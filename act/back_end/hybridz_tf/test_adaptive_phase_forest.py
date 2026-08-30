#!/usr/bin/env python3
"""Exact gates for the candidate-only adaptive HybridZ phase forest.

The discriminator is intentionally not the duplicate-ReLU toy already closed
by C49.  It is a seven-factor clique in signed HybridZ coordinates:

``z_i = (1+s_i)/2``, ``s_i in {-1,+1}``, and ``z_i + z_j <= 1``.

The exact integer maximum of ``sum(z_i)`` is one.  Even after applying the
complete existing C49 degree-one micro-RLT to every bit/edge pair, its parent
LP upper is ``7/3``.  Fixing one factor to active gives upper one immediately;
fixing it inactive leaves a smaller clique.  An adaptive forest consequently
uses 5 SAFE leaves / 8 child bounds, whereas blind depth-four enumeration uses
16 leaves.  The final UNKNOWN K4 node has two individually SAFE children.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from act.back_end.hybridz_tf import adaptive_phase_forest as forest_module
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    PhaseBoundWaveRequest,
    PhaseNodeBound,
    RivalSpec,
    RivalUpperBound,
    ordered_property_digest,
    run_adaptive_phase_forest_candidate,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.property_micro_rlt import (
    apply_property_micro_rlt,
    verify_property_micro_rlt_result,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _hz_attach_exact_phase_conditional_property_rows_from_operator,
    hz_enumerate_sparse_binary_phase_cover,
    hz_fresh_col_ids,
    hz_verify_sparse_binary_phase_child,
)


_SCALES = (Fraction(1), Fraction(3, 4), Fraction(1, 2))
_BASE_THRESHOLD = Fraction(9, 8)


def _clique_hz(n_binary: int) -> SparseHZono:
    """Return ``sum(z)`` over a complete-graph stable-set formulation."""

    edges = tuple(itertools.combinations(range(n_binary), 2))
    signed_edges = np.zeros(
        (len(edges), n_binary), dtype=np.float64
    )
    for row, (left, right) in enumerate(edges):
        # z_left + z_right <= 1  <=>  s_left + s_right <= 0.
        signed_edges[row, left] = 1.0
        signed_edges[row, right] = 1.0
    stable_ids = (
        hz_fresh_col_ids(n_binary, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    return SparseHZono(
        c=np.asarray([n_binary / 2.0], dtype=np.float64),
        Gc=sp.csr_matrix((1, 0), dtype=np.float64),
        Gb=sp.csr_matrix(
            np.full((1, n_binary), 0.5, dtype=np.float64)
        ),
        Ac=sp.csr_matrix((0, 0), dtype=np.float64),
        Ab=sp.csr_matrix((0, n_binary), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix((len(edges), 0), dtype=np.float64),
        Aub=sp.csr_matrix(signed_edges, dtype=np.float64),
        ub=np.zeros(len(edges), dtype=np.float64),
        col_ids=np.zeros(0, dtype=np.int64),
        bcol_ids=stable_ids,
    )


def _complete_c49_lift(n_binary: int) -> SparseHZono:
    base = _clique_hz(n_binary)
    source_rows = tuple(range(base.n_ub))
    result = apply_property_micro_rlt(
        base,
        source_rows_by_binary={
            binary: source_rows for binary in range(n_binary)
        },
        max_binary_factors=n_binary,
        max_source_rows_per_binary=len(source_rows),
        max_product_factors=n_binary * n_binary,
        max_selected_row_nnz=4 * n_binary * len(source_rows),
        max_requirement_scan_nnz=4
        * n_binary
        * len(source_rows),
    )
    if not verify_property_micro_rlt_result(result):
        raise AssertionError("complete C49 lift failed its live audit")
    return result.hz


def _attach_real_conditional_toy(hz: SparseHZono) -> None:
    """Attach the actual private conditional parent/seal metadata."""

    stable_id = int(hz.bcol_ids[0])
    rows = []
    for phase in (-1, 1):
        rows.append(
            {
                "binary_col_id": stable_id,
                "phase": phase,
                "layer_id": 17,
                "row": 0,
                "center": np.asarray([-0.25], dtype=np.float64),
                "generator": sp.csr_matrix(
                    (1, hz.n_cont), dtype=np.float64
                ),
                "error": np.zeros(1, dtype=np.float64),
                "rival_ids": (101,),
                "receipt": {
                    "schema": "c73_real_conditional_metadata_toy_v1",
                    "phase": phase,
                },
            }
        )
    _hz_attach_exact_phase_conditional_property_rows_from_operator(
        hz, rows
    )


def _output_upper(
    hz: SparseHZono,
    *,
    allow_infeasible: bool = False,
) -> float | None:
    """Independent HiGHS upper over the stored continuous relaxation."""

    objective = np.concatenate(
        [
            np.asarray(hz.Gc.getrow(0).toarray()).reshape(-1),
            np.asarray(hz.Gb.getrow(0).toarray()).reshape(-1),
        ]
    )
    upper_matrix = sp.hstack([hz.Auc, hz.Aub], format="csr")
    equality_matrix = sp.hstack([hz.Ac, hz.Ab], format="csr")
    result = linprog(
        -objective,
        A_ub=upper_matrix,
        b_ub=hz.ub,
        A_eq=equality_matrix,
        b_eq=hz.b,
        bounds=[(-1.0, 1.0)] * (hz.n_cont + hz.n_bin),
        method="highs",
    )
    if not result.success:
        if allow_infeasible and int(result.status) == 2:
            return None
        raise AssertionError(
            f"clique micro-RLT LP failed: {result.status}/{result.message}"
        )
    return float(hz.c[0] - result.fun)


def _fixed_phases(lineage):
    return tuple(int(sign) for _stable_id, sign in lineage)


def _fraction_integer_upper(
    n_binary: int,
    lineage,
) -> Fraction:
    """Exact enumeration of the original binary clique."""

    fixed_signs = _fixed_phases(lineage)
    best = None
    for signs in itertools.product((-1, 1), repeat=n_binary):
        if signs[: len(fixed_signs)] != fixed_signs:
            continue
        z = tuple(Fraction(1 + sign, 2) for sign in signs)
        if any(
            z[left] + z[right] > 1
            for left, right in itertools.combinations(
                range(n_binary), 2
            )
        ):
            continue
        value = sum(z, Fraction(0))
        best = value if best is None else max(best, value)
    if best is None:
        raise AssertionError("fixed clique phase unexpectedly empty")
    return best


def _fraction_c49_upper(
    n_binary: int,
    lineage,
) -> Fraction:
    """Exact degree-one RLT upper for this clique family.

    Multiplying an edge row by either endpoint forces every pair product to
    zero.  A complement product by a third bit then yields every triangle
    inequality.  Summing all triangle inequalities gives ``sum(z)<=n/3``;
    ``z_i=1/3`` and zero pair products attains it for at least four free bits.
    One active fixed bit forces all others to zero.  At most three free bits
    therefore have exact upper one.
    """

    signs = _fixed_phases(lineage)
    if 1 in signs:
        return Fraction(1)
    free = n_binary - len(signs)
    if free == 0:
        return Fraction(0)
    return max(Fraction(1), Fraction(free, 3))


def _rivals() -> tuple[RivalSpec, ...]:
    rival_ids = (101, 503, 907)
    return tuple(
        RivalSpec(
            rival_id=rival_id,
            objective=(float(scale),),
            threshold=float(scale * _BASE_THRESHOLD),
            assert_digest=hashlib.sha256(
                f"raw-assert-rival-{rival_id}".encode("ascii")
            ).hexdigest(),
        )
        for rival_id, scale in zip(rival_ids, _SCALES)
    )


def _identified_rival_bounds(
    rivals: tuple[RivalSpec, ...],
    upper_values,
) -> tuple[RivalUpperBound, ...]:
    values = tuple(upper_values)
    if len(values) != len(rivals):
        raise AssertionError("test rival bound/value count mismatch")
    return tuple(
        RivalUpperBound(
            rival_id=rival.rival_id,
            binding_digest=rival.binding_digest,
            upper=float(upper),
        )
        for rival, upper in zip(rivals, values)
    )


def _root_bound(
    hz: SparseHZono,
    n_binary: int,
    rivals: tuple[RivalSpec, ...] | None = None,
) -> PhaseNodeBound:
    rivals = _rivals() if rivals is None else rivals
    upper = _output_upper(hz)
    expected = _fraction_c49_upper(n_binary, ())
    if abs(upper - float(expected)) > 2.0e-9:
        raise AssertionError(
            f"root C49 upper {upper} != exact {expected}"
        )
    ids = tuple(int(value) for value in hz.bcol_ids.tolist())
    return PhaseNodeBound(
        node_id=0,
        lineage=(),
        remaining_bcol_ids=ids,
        rival_bounds=_identified_rival_bounds(
            rivals,
            (
                float(rival.objective[0]) * upper
                for rival in rivals
            ),
        ),
        property_digest=ordered_property_digest(rivals),
        node_semantic_digest=sparse_hz_semantic_digest(hz),
        verdict="UNKNOWN",
        binary_scores=tuple(
            (stable_id, float(len(ids) - offset))
            for offset, stable_id in enumerate(ids)
        ),
    )


def _valid_bound_wave(
    request: PhaseBoundWaveRequest,
    *,
    n_binary: int,
) -> tuple[PhaseNodeBound, ...]:
    thresholds = np.asarray(request.thresholds, dtype=np.float64)
    output = []
    for node in request.nodes:
        upper = _output_upper(node.hz)
        exact_rlt = _fraction_c49_upper(n_binary, node.lineage)
        if abs(upper - float(exact_rlt)) > 2.0e-8:
            raise AssertionError(
                f"lineage={node.lineage}: LP={upper}, "
                f"Fraction-RLT={exact_rlt}"
            )
        # The exact integer graph remains safe on every child; the RLT upper
        # is deliberately looser on the all-inactive prefix.
        if _fraction_integer_upper(n_binary, node.lineage) > 1:
            raise AssertionError("Fraction integer clique oracle failed")
        rival_upper = tuple(
            float(rival.objective[0]) * upper
            for rival in request.rivals
        )
        is_safe = bool(
            np.all(np.asarray(rival_upper) < thresholds)
        )
        ids = tuple(int(value) for value in node.hz.bcol_ids.tolist())
        output.append(
            PhaseNodeBound(
                node_id=node.node_id,
                lineage=node.lineage,
                remaining_bcol_ids=ids,
                rival_bounds=_identified_rival_bounds(
                    request.rivals, rival_upper
                ),
                property_digest=request.property_digest,
                node_semantic_digest=sparse_hz_semantic_digest(
                    node.hz
                ),
                verdict="SAFE" if is_safe else "UNKNOWN",
                binary_scores=tuple(
                    (stable_id, float(len(ids) - offset))
                    for offset, stable_id in enumerate(ids)
                ),
            )
        )
    return tuple(output)


class AdaptivePhaseForestExactDiscriminatorTests(unittest.TestCase):
    def test_complete_c49_parent_unknown_then_adaptive_prunes_twofold(
        self,
    ) -> None:
        n_binary = 7
        lifted = _complete_c49_lift(n_binary)
        self.assertEqual(
            _fraction_integer_upper(n_binary, ()), Fraction(1)
        )
        self.assertEqual(
            _fraction_c49_upper(n_binary, ()), Fraction(7, 3)
        )
        self.assertGreater(
            _output_upper(lifted), float(_BASE_THRESHOLD)
        )

        calls = []

        def bound_wave(request):
            calls.append((request.node_count, request.rival_count))
            return _valid_bound_wave(
                request, n_binary=n_binary
            )

        result = run_adaptive_phase_forest_candidate(
            lifted,
            _rivals(),
            _root_bound(lifted, n_binary),
            bound_wave,
            deadline=time.monotonic() + 10.0,
            max_depth=4,
            max_nodes=16,
        )
        self.assertEqual(
            result.status, "all_leaves_safe_candidate", result.receipt
        )
        self.assertTrue(result.all_leaves_safe)
        self.assertFalse(result.proof_authority)
        self.assertEqual(calls, [(2, 3)] * 4)

        receipt = result.receipt
        self.assertEqual(receipt["adaptive_safe_leaf_count"], 5)
        self.assertEqual(receipt["adaptive_child_bound_count"], 8)
        self.assertEqual(receipt["max_depth_reached"], 4)
        self.assertEqual(
            receipt["same_depth_full_cover_leaf_count"], 16
        )
        self.assertEqual(receipt["wave_sizes"], [2, 2, 2, 2])
        self.assertEqual(
            receipt["counters"],
            {
                "roots": 1,
                "children_expected": 8,
                "children_minted": 8,
                "processed": 9,
                "certified": 5,
                "branched": 4,
                "unresolved": 0,
                "active": 0,
            },
        )
        self.assertTrue(receipt["node_conservation"]["complete"])

        # The final all-inactive K4 parent has C49 upper 4/3 > 9/8,
        # while both exact-factor children have upper 1 < 9/8.
        safe_lineages = receipt["safe_leaf_lineages"]
        depth_four = [
            lineage for lineage in safe_lineages if len(lineage) == 4
        ]
        self.assertEqual(len(depth_four), 2)
        self.assertEqual(
            {lineage[-1][1] for lineage in depth_four}, {-1, 1}
        )

        # Build and solve the actual matched fixed-depth cover.  The 16 is
        # measured, not inferred from 2**4.
        fixed_cover = hz_enumerate_sparse_binary_phase_cover(
            lifted,
            positions=(0, 1, 2, 3),
            max_children=16,
            deadline=time.monotonic() + 5.0,
        )
        self.assertEqual(len(fixed_cover), 16)
        fixed_leaf_bounds = []
        for assignment, child in fixed_cover:
            self.assertTrue(
                hz_verify_sparse_binary_phase_child(
                    lifted, assignment, child
                )
            )
            upper = _output_upper(child, allow_infeasible=True)
            fixed_leaf_bounds.append(upper)
            if upper is not None:
                self.assertAlmostEqual(upper, 1.0, places=9)
                self.assertTrue(
                    all(
                        float(rival.objective[0]) * upper
                        < float(rival.threshold)
                        for rival in _rivals()
                    )
                )
        self.assertEqual(len(fixed_leaf_bounds), 16)
        self.assertEqual(
            sum(value is not None for value in fixed_leaf_bounds), 5
        )
        self.assertEqual(
            sum(value is None for value in fixed_leaf_bounds), 11
        )
        self.assertEqual(
            receipt["adaptive_child_bound_count"] * 2,
            len(fixed_leaf_bounds),
        )

    def _k4(self):
        n_binary = 4
        lifted = _complete_c49_lift(n_binary)
        root = _root_bound(lifted, n_binary)
        return n_binary, lifted, root

    def test_rival_permutation_objective_threshold_and_assert_swap_fail(
        self,
    ) -> None:
        base_rivals = _rivals()
        objective_swapped = (
            replace(
                base_rivals[0],
                objective=base_rivals[1].objective,
            ),
            replace(
                base_rivals[1],
                objective=base_rivals[0].objective,
            ),
            base_rivals[2],
        )
        threshold_swapped = (
            replace(
                base_rivals[0],
                threshold=base_rivals[1].threshold,
            ),
            replace(
                base_rivals[1],
                threshold=base_rivals[0].threshold,
            ),
            base_rivals[2],
        )
        assert_swapped = (
            replace(
                base_rivals[0],
                assert_digest=base_rivals[1].assert_digest,
            ),
            replace(
                base_rivals[1],
                assert_digest=base_rivals[0].assert_digest,
            ),
            base_rivals[2],
        )
        for label, altered, permute_bounds, expected in (
            (
                "permutation",
                tuple(reversed(base_rivals)),
                True,
                "bound_rival_id_mismatch",
            ),
            (
                "objective",
                objective_swapped,
                False,
                "bound_rival_binding_mismatch",
            ),
            (
                "threshold",
                threshold_swapped,
                False,
                "bound_rival_binding_mismatch",
            ),
            (
                "assert",
                assert_swapped,
                False,
                "bound_rival_binding_mismatch",
            ),
        ):
            with self.subTest(label=label):
                n_binary, lifted, root = self._k4()

                def bound_wave(request):
                    valid = _valid_bound_wave(
                        request, n_binary=n_binary
                    )
                    return tuple(
                        replace(
                            bound,
                            rival_bounds=(
                                tuple(reversed(bound.rival_bounds))
                                if permute_bounds
                                else tuple(
                                    replace(
                                        rival_bound,
                                        binding_digest=(
                                            altered_rival.binding_digest
                                        ),
                                    )
                                    for rival_bound, altered_rival in zip(
                                        bound.rival_bounds, altered
                                    )
                                )
                            ),
                        )
                        for bound in valid
                    )

                result = run_adaptive_phase_forest_candidate(
                    lifted,
                    base_rivals,
                    root,
                    bound_wave,
                    deadline=time.monotonic() + 5.0,
                    max_depth=1,
                )
                self.assertEqual(result.status, "fallback")
                self.assertEqual(result.reason, expected)
                self.assertFalse(result.proof_authority)

    def test_reversed_identified_upper_objects_cannot_fake_safe(
        self,
    ) -> None:
        """Reproduce the positional (100, 0) -> (0, 100) bypass."""

        n_binary, lifted, _unused_root = self._k4()
        rivals = (
            RivalSpec(
                rival_id=41,
                objective=(1.0,),
                threshold=50.0,
                assert_digest=hashlib.sha256(
                    b"positional-rival-41"
                ).hexdigest(),
            ),
            RivalSpec(
                rival_id=73,
                objective=(1.0,),
                threshold=200.0,
                assert_digest=hashlib.sha256(
                    b"positional-rival-73"
                ).hexdigest(),
            ),
        )
        root = _root_bound(lifted, n_binary, rivals)
        root = replace(
            root,
            rival_bounds=_identified_rival_bounds(
                rivals, (100.0, 0.0)
            ),
            verdict="UNKNOWN",
        )

        def bound_wave(request):
            honest = _identified_rival_bounds(
                request.rivals, (100.0, 0.0)
            )
            # If identities were stored separately, this numeric order would
            # be (0, 100), which is falsely SAFE against (50, 200).
            adversarial = tuple(reversed(honest))
            return tuple(
                PhaseNodeBound(
                    node_id=node.node_id,
                    lineage=node.lineage,
                    remaining_bcol_ids=tuple(
                        int(value)
                        for value in node.hz.bcol_ids.tolist()
                    ),
                    rival_bounds=adversarial,
                    property_digest=request.property_digest,
                    node_semantic_digest=sparse_hz_semantic_digest(
                        node.hz
                    ),
                    verdict="SAFE",
                )
                for node in request.nodes
            )

        result = run_adaptive_phase_forest_candidate(
            lifted,
            rivals,
            root,
            bound_wave,
            deadline=time.monotonic() + 5.0,
            max_depth=1,
        )
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.reason, "bound_rival_id_mismatch")
        self.assertFalse(result.all_leaves_safe)
        self.assertFalse(result.proof_authority)

    def test_hz_dense_csr_conditional_and_selector_mutation_fail(
        self,
    ) -> None:
        def mutate_center(node):
            node.hz.c[0] += 0.125

        def mutate_csr(node):
            node.hz.Aub.data[0] += 0.125

        def mutate_conditional(node):
            applied = dict(
                node.hz._solver_conditional_property_rows_applied
            )
            applied["proof_authority"] = False
            setattr(
                node.hz,
                "_solver_conditional_property_rows_applied",
                applied,
            )

        for label, mutator in (
            ("center", mutate_center),
            ("csr", mutate_csr),
            ("conditional", mutate_conditional),
        ):
            with self.subTest(label=label):
                n_binary, lifted, root = self._k4()
                if label == "conditional":
                    _attach_real_conditional_toy(lifted)
                    root = _root_bound(lifted, n_binary)

                def bound_wave(request):
                    valid = _valid_bound_wave(
                        request, n_binary=n_binary
                    )
                    mutator(request.nodes[0])
                    return valid

                result = run_adaptive_phase_forest_candidate(
                    lifted,
                    _rivals(),
                    root,
                    bound_wave,
                    deadline=time.monotonic() + 5.0,
                    max_depth=1,
                )
                self.assertEqual(result.status, "fallback")
                self.assertEqual(
                    result.reason,
                    "bound_wave_mutated_node_binding",
                )
                self.assertFalse(result.proof_authority)

        n_binary, lifted, root = self._k4()

        def mutating_selector(node, _bound):
            node.hz.c[0] += 0.25
            return int(node.hz.bcol_ids[0])

        selector_result = run_adaptive_phase_forest_candidate(
            lifted,
            _rivals(),
            root,
            lambda request: _valid_bound_wave(
                request, n_binary=n_binary
            ),
            deadline=time.monotonic() + 5.0,
            max_depth=1,
            select_binary=mutating_selector,
        )
        self.assertEqual(selector_result.status, "fallback")
        self.assertEqual(
            selector_result.reason,
            "selector_mutated_node_semantics",
        )

    def test_semantic_id_unsigned_wrap_and_duplicate_fail_closed(
        self,
    ) -> None:
        def unsigned_wrap(hz):
            wrapped = np.asarray(
                hz.bcol_ids, dtype=np.uint64
            ).copy()
            wrapped[0] = np.iinfo(np.uint64).max
            hz.bcol_ids = wrapped

        def duplicate(hz):
            duplicated = np.asarray(
                hz.bcol_ids, dtype=np.int64
            ).copy()
            duplicated[1] = duplicated[0]
            hz.bcol_ids = duplicated

        for label, mutator in (
            ("unsigned_wrap", unsigned_wrap),
            ("duplicate", duplicate),
        ):
            with self.subTest(label=label):
                _n_binary, lifted, _root = self._k4()
                mutator(lifted)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "semantic_bcol_ids_malformed",
                ):
                    sparse_hz_semantic_digest(lifted)

    def test_semantic_id_validation_matches_legacy_oracle_and_hash(
        self,
    ) -> None:
        def legacy_oracle(value, expected):
            try:
                array = np.asarray(value)
                return not (
                    array.dtype != np.dtype(np.int64)
                    or array.ndim != 1
                    or int(array.size) != int(expected)
                    or (array.size and np.any(array < 0))
                    or len(
                        set(int(item) for item in array.tolist())
                    )
                    != int(array.size)
                )
            except (TypeError, ValueError, OverflowError):
                return False

        rng = np.random.default_rng(1949)
        cases = [
            ("empty", np.zeros(0, dtype=np.int64), 0),
            ("sorted", np.arange(257, dtype=np.int64), 257),
            (
                "unsorted",
                rng.permutation(257).astype(np.int64, copy=False),
                257,
            ),
            (
                "duplicate",
                np.asarray([8, 3, 8, 1], dtype=np.int64),
                4,
            ),
            (
                "negative",
                np.asarray([8, 3, -1, 2], dtype=np.int64),
                4,
            ),
            ("unsigned", np.arange(4, dtype=np.uint64), 4),
            ("float", np.arange(4, dtype=np.float64), 4),
            (
                "two_dimensional",
                np.arange(4, dtype=np.int64).reshape(2, 2),
                4,
            ),
            ("wrong_length", np.arange(4, dtype=np.int64), 3),
            ("ragged", [[1], [2, 3]], 2),
        ]
        for index in range(24):
            size = int(rng.integers(0, 96))
            ids = rng.integers(
                0,
                max(1, 2 * size),
                size=size,
                dtype=np.int64,
            )
            cases.append((f"random_{index}", ids, size))

        for label, value, expected_size in cases:
            with self.subTest(label=label):
                expected_valid = legacy_oracle(value, expected_size)
                actual = hashlib.sha256()
                try:
                    forest_module._hash_semantic_ids(
                        actual,
                        name="stable_ids",
                        value=value,
                        expected=expected_size,
                    )
                    actual_valid = True
                except RuntimeError as exc:
                    self.assertEqual(
                        str(exc), "semantic_stable_ids_malformed"
                    )
                    actual_valid = False
                self.assertEqual(actual_valid, expected_valid)
                if expected_valid:
                    array = np.asarray(value)
                    legacy = hashlib.sha256()
                    forest_module._hash_framed_bytes(
                        legacy,
                        b"stable_ids_shape",
                        np.asarray(
                            array.shape, dtype=np.int64
                        ).tobytes(),
                    )
                    forest_module._hash_framed_bytes(
                        legacy, b"stable_ids", array.tobytes()
                    )
                    self.assertEqual(
                        actual.hexdigest(), legacy.hexdigest()
                    )

        source = inspect.getsource(forest_module._hash_semantic_ids)
        self.assertNotIn(".tolist(", source)
        self.assertNotIn("set(", source)

        class NoToListArray(np.ndarray):
            def tolist(self):
                raise AssertionError("semantic ID validation called tolist")

        no_tolist = np.arange(32, dtype=np.int64).view(NoToListArray)
        forest_module._hash_semantic_ids(
            hashlib.sha256(),
            name="stable_ids",
            value=no_tolist,
            expected=no_tolist.size,
        )

    def test_unsorted_semantic_id_peak_is_one_numpy_vector(self) -> None:
        script = textwrap.dedent(
            """
            import hashlib
            import json
            import sys
            import tracemalloc
            import numpy as np
            from act.back_end.hybridz_tf import adaptive_phase_forest as forest

            count = int(sys.argv[1])
            ids = np.arange(count - 1, -1, -1, dtype=np.int64)
            tracemalloc.start()
            forest._hash_semantic_ids(
                hashlib.sha256(),
                name="stable_ids",
                value=ids,
                expected=count,
            )
            _current, peak = tracemalloc.get_traced_memory()
            print(json.dumps({"peak": peak}))
            """
        )
        repository = Path(__file__).resolve().parents[3]
        observed = []
        for count in (250_000, 500_000, 1_000_000):
            completed = subprocess.run(
                [sys.executable, "-c", script, str(count)],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            )
            peak = int(
                json.loads(completed.stdout.splitlines()[-1])["peak"]
            )
            observed.append(peak)
            # One 8*N NumPy sort result, one chunk-sized boolean comparison,
            # and 2 MiB for interpreter/allocator measurement noise.  A
            # Python list+set of N boxed ints exceeds this bound by design.
            explicit_bound = (
                count * np.dtype(np.int64).itemsize
                + forest_module._SEMANTIC_HASH_CHUNK_ITEMS
                + 2 * 1024 * 1024
            )
            self.assertLessEqual(
                peak,
                explicit_bound,
                msg=(
                    f"{count} IDs used {peak} bytes, above "
                    f"the {explicit_bound}-byte bounded NumPy path"
                ),
            )
        self.assertLess(observed[-1], 16 * 1024 * 1024)

    def test_streamed_semantic_digest_is_legacy_bit_exact_and_chunked(
        self,
    ) -> None:
        _n_binary, lifted, _root = self._k4()
        _attach_real_conditional_toy(lifted)

        def framed(digest, label, payload):
            digest.update(len(label).to_bytes(8, "little"))
            digest.update(label)
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)

        def legacy_digest(hz):
            digest = hashlib.sha256()
            digest.update(
                b"hybridz_adaptive_phase_sparse_hz_semantic_v1"
            )
            for name in ("c", "b", "ub"):
                value = getattr(hz, name)
                if value is None:
                    framed(digest, name.encode("ascii"), b"NONE")
                    continue
                array = np.asarray(value)
                framed(
                    digest,
                    f"{name}_shape".encode("ascii"),
                    np.asarray(array.shape, dtype=np.int64).tobytes(),
                )
                framed(
                    digest,
                    name.encode("ascii"),
                    np.ascontiguousarray(
                        array, dtype=np.float64
                    ).tobytes(),
                )
            for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                matrix = getattr(hz, name)
                if matrix is None:
                    framed(digest, name.encode("ascii"), b"NONE")
                    continue
                framed(
                    digest,
                    f"{name}_shape".encode("ascii"),
                    np.asarray(matrix.shape, dtype=np.int64).tobytes(),
                )
                for suffix, value, dtype in (
                    ("indptr", matrix.indptr, np.int64),
                    ("indices", matrix.indices, np.int64),
                    ("data", matrix.data, np.float64),
                ):
                    framed(
                        digest,
                        f"{name}_{suffix}".encode("ascii"),
                        np.asarray(value, dtype=dtype).tobytes(),
                    )
            for name, expected in (
                ("col_ids", hz.n_cont),
                ("bcol_ids", hz.n_bin),
            ):
                array = np.asarray(getattr(hz, name), dtype=np.int64)
                self.assertEqual(array.size, expected)
                framed(
                    digest,
                    f"{name}_shape".encode("ascii"),
                    np.asarray(array.shape, dtype=np.int64).tobytes(),
                )
                framed(digest, name.encode("ascii"), array.tobytes())
            conditional_names = tuple(
                sorted(
                    name
                    for name in vars(hz)
                    if "conditional" in name.lower()
                )
            )
            forest_module._semantic_hash_value(
                digest, conditional_names
            )
            for name in conditional_names:
                forest_module._semantic_hash_value(digest, name)
                forest_module._semantic_hash_value(
                    digest, getattr(hz, name)
                )
            return digest.hexdigest()

        self.assertEqual(
            sparse_hz_semantic_digest(lifted), legacy_digest(lifted)
        )

        class RecordingDigest:
            def __init__(self):
                self.maximum_update = 0

            def update(self, payload):
                self.maximum_update = max(
                    self.maximum_update, len(payload)
                )

        item_count = 2 * forest_module._SEMANTIC_HASH_CHUNK_ITEMS + 7
        recorder = RecordingDigest()
        forest_module._hash_framed_numeric_array(
            recorder,
            label=b"indices",
            value=np.arange(item_count, dtype=np.int32),
            canonical_dtype=np.int64,
        )
        self.assertLessEqual(
            recorder.maximum_update,
            forest_module._SEMANTIC_HASH_CHUNK_ITEMS
            * np.dtype(np.int64).itemsize,
        )

    def test_vectorized_csr_recheck_masks_only_real_row_boundaries(
        self,
    ) -> None:
        valid = sp.csr_matrix(
            (
                np.array([1.0, 2.0, 3.0]),
                np.array([7, 1, 4], dtype=np.int32),
                np.array([0, 1, 1, 1, 3, 3], dtype=np.int32),
            ),
            shape=(5, 8),
        )
        self.assertTrue(
            forest_module._canonical_csr_structure_is_valid(valid)
        )
        for label, indices in (
            ("descending", (7, 4, 1)),
            ("duplicate", (7, 1, 1)),
        ):
            with self.subTest(label=label):
                malformed = valid.copy()
                self.assertTrue(malformed.has_canonical_format)
                malformed.indices[:] = np.asarray(
                    indices, dtype=malformed.indices.dtype
                )
                self.assertTrue(malformed.has_canonical_format)
                self.assertFalse(
                    forest_module._canonical_csr_structure_is_valid(
                        malformed
                    )
                )
                with self.assertRaisesRegex(
                    RuntimeError, "semantic_Auc_malformed"
                ):
                    forest_module._hash_csr_semantic_matrix(
                        hashlib.sha256(),
                        name="Auc",
                        value=malformed,
                    )

    def test_nonfirst_stable_binary_id_is_selected_and_removed(self) -> None:
        n_binary, lifted, root = self._k4()
        original_ids = tuple(
            int(value) for value in lifted.bcol_ids.tolist()
        )
        selected_id = original_ids[2]
        seen_child_ids = []

        def bound_wave(request):
            seen_child_ids.extend(
                tuple(int(value) for value in node.hz.bcol_ids.tolist())
                for node in request.nodes
            )
            return _valid_bound_wave(
                request, n_binary=n_binary
            )

        result = run_adaptive_phase_forest_candidate(
            lifted,
            _rivals(),
            root,
            bound_wave,
            deadline=time.monotonic() + 5.0,
            max_depth=1,
            select_binary=lambda _node, _bound: selected_id,
        )
        self.assertEqual(
            result.status, "all_leaves_safe_candidate", result.receipt
        )
        self.assertEqual(
            result.receipt["selected_bcol_ids"], [selected_id]
        )
        expected_remaining = tuple(
            value for value in original_ids if value != selected_id
        )
        self.assertEqual(seen_child_ids, [expected_remaining] * 2)
        self.assertEqual(
            {
                lineage[0][0]
                for lineage in result.receipt["safe_leaf_lineages"]
            },
            {selected_id},
        )

    def test_missing_child_and_duplicate_sign_overlap_fail_closed(
        self,
    ) -> None:
        from act.back_end.hybridz_tf import adaptive_phase_forest as module

        for mutation, expected in (
            (
                lambda cover: cover[:1],
                "split_cover_incomplete",
            ),
            (
                lambda cover: (cover[0], cover[0]),
                "split_assignments_overlap_and_complement_omitted",
            ),
        ):
            with self.subTest(expected=expected):
                n_binary, lifted, root = self._k4()
                original = (
                    module.hz_enumerate_sparse_binary_phase_cover
                )

                def corrupt(*args, **kwargs):
                    return mutation(original(*args, **kwargs))

                with patch.object(
                    module,
                    "hz_enumerate_sparse_binary_phase_cover",
                    side_effect=corrupt,
                ):
                    result = run_adaptive_phase_forest_candidate(
                        lifted,
                        _rivals(),
                        root,
                        lambda request: _valid_bound_wave(
                            request, n_binary=n_binary
                        ),
                        deadline=time.monotonic() + 5.0,
                        max_depth=1,
                    )
                self.assertEqual(result.status, "fallback")
                self.assertIn(expected, result.reason)
                self.assertFalse(result.all_leaves_safe)
                self.assertFalse(result.proof_authority)
                self.assertFalse(
                    result.receipt["node_conservation"]["complete"]
                )

    def test_wrong_split_and_wrong_child_copy_fail_closed(self) -> None:
        from act.back_end.hybridz_tf import adaptive_phase_forest as module

        for mutation, expected in (
            (
                lambda cover: (
                    (((1, -1),), cover[0][1]),
                    cover[1],
                ),
                "split_wrong_binary_position",
            ),
            (
                lambda cover: (
                    (cover[0][0], cover[1][1]),
                    (cover[1][0], cover[0][1]),
                ),
                "split_child_live_audit_failed",
            ),
        ):
            with self.subTest(expected=expected):
                n_binary, lifted, root = self._k4()
                original = (
                    module.hz_enumerate_sparse_binary_phase_cover
                )

                def corrupt(*args, **kwargs):
                    return mutation(original(*args, **kwargs))

                with patch.object(
                    module,
                    "hz_enumerate_sparse_binary_phase_cover",
                    side_effect=corrupt,
                ):
                    result = run_adaptive_phase_forest_candidate(
                        lifted,
                        _rivals(),
                        root,
                        lambda request: _valid_bound_wave(
                            request, n_binary=n_binary
                        ),
                        deadline=time.monotonic() + 5.0,
                        max_depth=1,
                    )
                self.assertEqual(result.status, "fallback")
                self.assertEqual(result.reason, expected)
                self.assertFalse(result.proof_authority)

    def test_bound_omission_wrong_binding_and_nan_fail_closed(self) -> None:
        mutations = (
            (
                lambda request, bounds: bounds[:1],
                "bound_wave_result_count_mismatch",
            ),
            (
                lambda request, bounds: tuple(reversed(bounds)),
                "bound_node_id_mismatch",
            ),
            (
                lambda request, bounds: (
                    PhaseNodeBound(
                        node_id=bounds[0].node_id,
                        lineage=bounds[0].lineage,
                        remaining_bcol_ids=(
                            bounds[0].remaining_bcol_ids
                        ),
                        rival_bounds=(
                            replace(
                                bounds[0].rival_bounds[0],
                                upper=float("nan"),
                            ),
                            *bounds[0].rival_bounds[1:],
                        ),
                        property_digest=bounds[0].property_digest,
                        node_semantic_digest=(
                            bounds[0].node_semantic_digest
                        ),
                        verdict="UNKNOWN",
                    ),
                    bounds[1],
                ),
                "bound_rival_nonfinite",
            ),
        )
        for mutation, expected in mutations:
            with self.subTest(expected=expected):
                n_binary, lifted, root = self._k4()

                def bound_wave(request):
                    valid = _valid_bound_wave(
                        request, n_binary=n_binary
                    )
                    return mutation(request, valid)

                result = run_adaptive_phase_forest_candidate(
                    lifted,
                    _rivals(),
                    root,
                    bound_wave,
                    deadline=time.monotonic() + 5.0,
                    max_depth=1,
                )
                self.assertEqual(result.status, "fallback")
                self.assertEqual(result.reason, expected)
                self.assertFalse(result.all_leaves_safe)
                self.assertGreater(
                    result.receipt["counters"]["active"], 0
                )

    def test_nan_root_and_expired_deadline_fail_before_callback(
        self,
    ) -> None:
        n_binary, lifted, root = self._k4()
        calls = 0

        def bound_wave(request):
            nonlocal calls
            calls += 1
            return _valid_bound_wave(
                request, n_binary=n_binary
            )

        nan_root = PhaseNodeBound(
            node_id=0,
            lineage=(),
            remaining_bcol_ids=root.remaining_bcol_ids,
            rival_bounds=(
                replace(
                    root.rival_bounds[0], upper=float("nan")
                ),
                *root.rival_bounds[1:],
            ),
            property_digest=root.property_digest,
            node_semantic_digest=root.node_semantic_digest,
            verdict="UNKNOWN",
        )
        nan_result = run_adaptive_phase_forest_candidate(
            lifted,
            _rivals(),
            nan_root,
            bound_wave,
            deadline=time.monotonic() + 5.0,
            max_depth=1,
        )
        self.assertEqual(nan_result.reason, "bound_rival_nonfinite")
        self.assertFalse(nan_result.proof_authority)

        deadline_result = run_adaptive_phase_forest_candidate(
            lifted,
            _rivals(),
            root,
            bound_wave,
            deadline=time.monotonic() - 1.0,
            max_depth=1,
        )
        self.assertEqual(deadline_result.reason, "deadline_before_root")
        self.assertFalse(deadline_result.proof_authority)
        self.assertEqual(calls, 0)

    def test_depth_and_node_caps_leave_explicit_unresolved_nodes(
        self,
    ) -> None:
        n_binary = 7
        for kwargs, expected in (
            ({"max_depth": 1, "max_nodes": 16}, "max_depth"),
            ({"max_depth": 4, "max_nodes": 4}, "max_nodes"),
        ):
            with self.subTest(expected=expected):
                lifted = _complete_c49_lift(n_binary)
                result = run_adaptive_phase_forest_candidate(
                    lifted,
                    _rivals(),
                    _root_bound(lifted, n_binary),
                    lambda request: _valid_bound_wave(
                        request, n_binary=n_binary
                    ),
                    deadline=time.monotonic() + 5.0,
                    **kwargs,
                )
                self.assertEqual(result.status, "fallback")
                self.assertEqual(result.reason, expected)
                self.assertGreater(
                    result.receipt["counters"]["unresolved"], 0
                )
                self.assertFalse(
                    result.receipt["node_conservation"]["complete"]
                )


if __name__ == "__main__":
    unittest.main()
