#!/usr/bin/env python3
"""Toy audits for the one-copy PCOH fresh SparseHZ materializer."""

from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction
import inspect
import itertools
import threading
import time
import tracemalloc
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf import (
    operator_phase_conditioned_objective_hull_fresh_materializer as fresh_module,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_fresh_materializer import (
    PCOHFreshMaterializationCaps,
    PhaseConditionedObjectiveHullFreshMaterializationError,
    consume_live_phase_conditioned_objective_hull_fresh_build,
    discard_live_phase_conditioned_objective_hull_fresh_build,
    issue_live_phase_conditioned_objective_hull_fresh_build,
    verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _k4_corner_build,
    _rivals,
)
from act.back_end.hybridz_tf.test_operator_phase_conditioned_live_adapter import (
    _sources,
)
from act.back_end.verifier import _hybridz_witness_input
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.solver.solver_hz import SparseHZono


def _issue(
    build,
    rivals,
    selection,
    stable_ids,
    certificates,
    pair_bundle,
    *,
    deadline=None,
    caps=None,
):
    kwargs = {}
    if caps is not None:
        kwargs["caps"] = caps
    return issue_live_phase_conditioned_objective_hull_fresh_build(
        build,
        rivals,
        selection,
        focused_rival_id=10,
        stable_bit_ids=stable_ids,
        conditional_certificates=certificates,
        pair_bundle=pair_bundle,
        deadline=(time.monotonic() + 60.0 if deadline is None else deadline),
        **kwargs,
    )


def _consume(issuance):
    return consume_live_phase_conditioned_objective_hull_fresh_build(
        issuance,
        issuance.capability,
        deadline=time.monotonic() + 60.0,
    )


def _factor_objective(hz, weights):
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    continuous = np.asarray(weights @ hz.Gc).reshape(-1)
    binary = np.asarray(weights @ hz.Gb).reshape(-1)
    center = float(weights @ hz.c)
    return center, continuous, binary


def _lp_maximum(hz, weights):
    center, continuous, binary = _factor_objective(hz, weights)
    objective = np.concatenate((continuous, binary))
    equality = np.hstack((hz.Ac.toarray(), hz.Ab.toarray()))
    upper = np.hstack((hz.Auc.toarray(), hz.Aub.toarray()))
    result = linprog(
        -objective,
        A_ub=upper,
        b_ub=hz.ub,
        A_eq=equality,
        b_eq=hz.b,
        bounds=[(-1.0, 1.0)] * int(objective.size),
        method="highs",
    )
    if not result.success:
        raise AssertionError(f"toy LP failed: {result.message}")
    return center + float(objective @ result.x)


def _enumerated_integer_maximum(hz, weights):
    center, continuous, binary = _factor_objective(hz, weights)
    values = []
    Auc = hz.Auc.toarray()
    Aub = hz.Aub.toarray()
    Ac = hz.Ac.toarray()
    Ab = hz.Ab.toarray()
    for assignment in itertools.product((-1.0, 1.0), repeat=hz.n_bin):
        binary_value = np.asarray(assignment, dtype=np.float64)
        result = linprog(
            -continuous,
            A_ub=Auc,
            b_ub=hz.ub - Aub @ binary_value,
            A_eq=Ac,
            b_eq=hz.b - Ab @ binary_value,
            bounds=[(-1.0, 1.0)] * hz.n_cont,
            method="highs",
        )
        if result.success:
            values.append(
                center
                + float(binary @ binary_value)
                + float(continuous @ result.x)
            )
    if not values:
        raise AssertionError("toy exact binary enumeration found no point")
    return max(values)


def _exact_csr_row_value(matrix, row, values):
    start = int(matrix.indptr[row])
    stop = int(matrix.indptr[row + 1])
    return sum(
        (
            Fraction.from_float(float(matrix.data[offset]))
            * values[int(matrix.indices[offset])]
            for offset in range(start, stop)
        ),
        Fraction(0),
    )


def _fraction_hz_feasible(hz, continuous, binary):
    """Replay every box/equality/upper row over exact binary64 Fractions."""

    if len(continuous) != hz.n_cont or len(binary) != hz.n_bin:
        return False
    if any(value < -1 or value > 1 for value in continuous):
        return False
    if any(value not in {-1, 1} for value in binary):
        return False
    for row in range(hz.n_eq):
        lhs = _exact_csr_row_value(hz.Ac, row, continuous)
        lhs += _exact_csr_row_value(hz.Ab, row, binary)
        if lhs != Fraction.from_float(float(hz.b[row])):
            return False
    for row in range(hz.n_ub):
        lhs = _exact_csr_row_value(hz.Auc, row, continuous)
        lhs += _exact_csr_row_value(hz.Aub, row, binary)
        if lhs > Fraction.from_float(float(hz.ub[row])):
            return False
    return True


def _fraction_hz_output(hz, continuous, binary):
    """Evaluate the complete HZ output map over exact binary64 Fractions."""

    result = []
    for row in range(hz.n_out):
        value = Fraction.from_float(float(hz.c[row]))
        value += _exact_csr_row_value(hz.Gc, row, continuous)
        value += _exact_csr_row_value(hz.Gb, row, binary)
        result.append(value)
    return tuple(result)


def _fraction_objective_projection(output, rival):
    value = sum(
        (
            Fraction.from_float(float(weight)) * coordinate
            for weight, coordinate in zip(rival.objective, output)
        ),
        Fraction(0),
    )
    return value - Fraction.from_float(float(rival.threshold))


def _k4_exact_parent_witness(pattern):
    """An exact full-factor witness for each of the five feasible corners."""

    if len(pattern) != 4 or sum(phase == 1 for phase in pattern) > 1:
        raise ValueError("K4 toy witness exists only for feasible patterns")
    continuous = [Fraction(0)] * 9
    if 1 in pattern:
        corners = ((1, 1), (1, -1), (-1, 1), (-1, -1))
        corner = corners[pattern.index(1)]
        continuous[0], continuous[1] = map(Fraction, corner)
    continuous[2:6] = tuple(Fraction(phase) for phase in pattern)
    binary = [Fraction(phase) for phase in pattern]
    return continuous, binary


def _with_zero_source_equality(build):
    """Clone the K4 source with one harmless old equality row and tag."""

    source = build.hz
    hz = SparseHZono(
        c=source.c.copy(),
        Gc=source.Gc.copy(),
        Gb=source.Gb.copy(),
        Ac=sp.vstack(
            (source.Ac, sp.csr_matrix((1, source.n_cont))), format="csr"
        ),
        Ab=sp.vstack(
            (source.Ab, sp.csr_matrix((1, source.n_bin))), format="csr"
        ),
        b=np.concatenate((source.b, np.asarray((0.0,), dtype=np.float64))),
        Auc=source.Auc.copy(),
        Aub=source.Aub.copy(),
        ub=source.ub.copy(),
        col_ids=source.col_ids.copy(),
        bcol_ids=source.bcol_ids.copy(),
    )
    for name in fresh_module._PROVENANCE_NAMES:
        setattr(hz, name, getattr(source, name).copy())
    setattr(
        hz,
        "_solver_constraint_row_tags",
        ("toy_source_equality:v1",)
        + tuple(source._solver_constraint_row_tags),
    )
    setattr(hz, "_solver_row_constraint_prefix_frames", {})
    return OperatorHZBuild(
        hz=hz,
        input_col_ids=build.input_col_ids.copy(),
        input_layer_id=build.input_layer_id,
        output_layer_id=build.output_layer_id,
        assert_layer_id=build.assert_layer_id,
        metadata={},
        property_upper_output=False,
        property_upper_row_groups=(),
        verified_preactivation_frame=None,
        constructive_nonempty_seal=None,
    )


def _with_stable_ids(
    build,
    *,
    col_ids=None,
    bcol_ids=None,
    input_col_ids=None,
):
    """Clone only the ID-bearing toy buffers needed by source validation."""

    source = build.hz
    chosen_col_ids = np.asarray(
        source.col_ids if col_ids is None else col_ids,
        dtype=np.int64,
    ).copy()
    chosen_bcol_ids = np.asarray(
        source.bcol_ids if bcol_ids is None else bcol_ids,
        dtype=np.int64,
    ).copy()
    chosen_input_ids = np.asarray(
        build.input_col_ids if input_col_ids is None else input_col_ids,
        dtype=np.int64,
    ).copy()
    hz = SparseHZono(
        c=source.c.copy(),
        Gc=source.Gc.copy(),
        Gb=source.Gb.copy(),
        Ac=source.Ac.copy(),
        Ab=source.Ab.copy(),
        b=source.b.copy(),
        Auc=source.Auc.copy(),
        Aub=source.Aub.copy(),
        ub=source.ub.copy(),
        col_ids=source.col_ids.copy(),
        bcol_ids=source.bcol_ids.copy(),
    )
    # Construct from a valid shell, then inject the adversarial ID buffers so
    # this test reaches the independent fresh-source admission gate.
    object.__setattr__(hz, "col_ids", chosen_col_ids)
    object.__setattr__(hz, "bcol_ids", chosen_bcol_ids)
    setattr(hz, "full_col_ids", chosen_input_ids.copy())
    for name in (
        "operator_input_center",
        "operator_input_radius",
        "_solver_continuous_column_layer_ids",
    ):
        setattr(hz, name, getattr(source, name).copy())
    setattr(
        hz,
        "_solver_constraint_row_tags",
        tuple(source._solver_constraint_row_tags),
    )
    setattr(hz, "_solver_row_constraint_prefix_frames", {})
    return OperatorHZBuild(
        hz=hz,
        input_col_ids=chosen_input_ids,
        input_layer_id=build.input_layer_id,
        output_layer_id=build.output_layer_id,
        assert_layer_id=build.assert_layer_id,
        metadata={},
        property_upper_output=False,
        property_upper_row_groups=(),
        verified_preactivation_frame=None,
        constructive_nonempty_seal=None,
    )


class PCOHFreshMaterializerK4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _k4_corner_build()
        (
            cls.rivals,
            cls.selection,
            cls.stable_ids,
            cls.certificates,
            cls.pair_bundle,
        ) = _sources(cls.build)

    def issue(self, **kwargs):
        return _issue(
            self.build,
            self.rivals,
            self.selection,
            self.stable_ids,
            self.certificates,
            self.pair_bundle,
            **kwargs,
        )

    def test_cleanup_only_discard_never_returns_or_reuses_private_build(self):
        issuance = self.issue()
        self.assertTrue(
            discard_live_phase_conditioned_objective_hull_fresh_build(
                issuance, issuance.capability
            )
        )
        self.assertFalse(
            discard_live_phase_conditioned_objective_hull_fresh_build(
                issuance, issuance.capability
            )
        )
        with self.assertRaises(
            PhaseConditionedObjectiveHullFreshMaterializationError
        ):
            _consume(issuance)

    def test_k4_dimensions_nnz_tags_provenance_and_authority_firewall(self):
        parent = self.build.hz
        parent_digest = sparse_hz_semantic_digest(parent)
        parent_tags = tuple(parent._solver_constraint_row_tags)
        issuance = self.issue()
        fresh = _consume(issuance)
        hz = fresh.hz

        self.assertEqual(
            issuance.receipt["source_dimensions"],
            (parent.n_out, parent.n_cont, parent.n_bin, parent.n_eq, parent.n_ub),
        )
        self.assertEqual(
            issuance.receipt["fresh_dimensions"],
            (parent.n_out, parent.n_cont + 16, parent.n_bin, parent.n_eq + 16, parent.n_ub + 1),
        )
        self.assertEqual(hz.Ac.nnz - parent.Ac.nnz, 16 * 5 + 11)
        self.assertEqual(hz.Ab.nnz - parent.Ab.nnz, 4)
        self.assertEqual(len(issuance.equality_row_tags), 16)
        self.assertEqual(len(issuance.upper_row_tags), 1)
        self.assertEqual(
            hz._solver_constraint_row_tags,
            parent_tags[: parent.n_eq]
            + issuance.equality_row_tags
            + parent_tags[parent.n_eq :]
            + issuance.upper_row_tags,
        )
        self.assertTrue(
            np.array_equal(
                hz._solver_continuous_column_layer_ids[: parent.n_cont],
                parent._solver_continuous_column_layer_ids,
            )
        )
        self.assertTrue(
            np.array_equal(
                hz._solver_continuous_column_layer_ids[parent.n_cont :],
                -np.ones(16, dtype=np.int64),
            )
        )
        self.assertTrue(np.array_equal(fresh.input_col_ids, self.build.input_col_ids))
        self.assertTrue(np.array_equal(hz.full_col_ids, parent.full_col_ids))
        self.assertIsNone(fresh.constructive_nonempty_seal)
        for forbidden in (
            "_solver_known_nonempty",
            "_solver_constructive_nonempty_token",
            "_property_full_input_replay_result",
            "operator_hz_metadata",
        ):
            self.assertNotIn(forbidden, vars(hz))
        self.assertFalse(issuance.proof_authority)
        self.assertFalse(issuance.verdict_authority)
        self.assertFalse(issuance.capability.proof_authority)
        self.assertIs(type(fresh.metadata), MappingProxyType)
        self.assertEqual(
            fresh.metadata["schema"],
            "act.hybridz_pcoh_private_fresh_build.toy.v1",
        )
        self.assertFalse(fresh.metadata["proof_authority"])
        with self.assertRaises(TypeError):
            fresh.metadata["proof_authority"] = True
        self.assertFalse(issuance.receipt["production_ready"])
        self.assertFalse(issuance.receipt["constructive_nonempty_reissued"])
        self.assertNotIn("absolute_deadline", issuance.receipt)
        self.assertTrue(
            issuance.receipt["shared_absolute_deadline_fail_closed"]
        )
        self.assertEqual(
            issuance.receipt["deadline_enforcement"],
            "cooperative_bulk_scans",
        )
        self.assertFalse(
            issuance.receipt["hard_wall_deadline_guaranteed"]
        )
        self.assertEqual(
            issuance.receipt["eta_id_allocator_route"],
            "solver_hz.hz_reserve_fresh_col_ids_above",
        )
        self.assertTrue(
            issuance.receipt["eta_id_allocator_global_lock_shared"]
        )
        self.assertTrue(issuance.receipt["eta_id_reservation_non_reusable"])
        self.assertNotIn(
            "module_local_eta_id_allocator_not_global_solver_allocator",
            issuance.receipt["production_blockers"],
        )
        self.assertNotIn(
            "row_materializer_uses_full_python_stable_id_maps",
            issuance.receipt["production_blockers"],
        )
        self.assertEqual(sparse_hz_semantic_digest(parent), parent_digest)
        self.assertEqual(
            sparse_hz_semantic_digest(hz), issuance.fresh_semantic_digest
        )

    def test_detached_readonly_output_jacobian_and_one_final_copy(self):
        source_arrays = []
        parent = self.build.hz
        for name in fresh_module._CORE_DENSE_NAMES:
            source_arrays.append(getattr(parent, name))
        for name in fresh_module._CORE_CSR_NAMES:
            matrix = getattr(parent, name)
            source_arrays.extend((matrix.data, matrix.indices, matrix.indptr))
        for name in fresh_module._PROVENANCE_NAMES:
            source_arrays.append(getattr(parent, name))
        source_arrays.append(self.build.input_col_ids)

        with mock.patch.object(
            fresh_module,
            "_copy_csr_with_tail",
            wraps=fresh_module._copy_csr_with_tail,
        ) as csr_copy:
            issuance = self.issue()
        fresh = _consume(issuance)
        hz = fresh.hz
        fresh_arrays = []
        for name in fresh_module._CORE_DENSE_NAMES:
            fresh_arrays.append(getattr(hz, name))
        for name in fresh_module._CORE_CSR_NAMES:
            matrix = getattr(hz, name)
            fresh_arrays.extend((matrix.data, matrix.indices, matrix.indptr))
        for name in fresh_module._PROVENANCE_NAMES:
            fresh_arrays.append(getattr(hz, name))
        fresh_arrays.append(fresh.input_col_ids)

        self.assertEqual(csr_copy.call_count, 6)
        self.assertFalse(
            any(
                np.shares_memory(source, target)
                for source in source_arrays
                for target in fresh_arrays
            )
        )
        self.assertTrue(all(not value.flags.writeable for value in fresh_arrays))
        self.assertTrue(np.array_equal(hz.c, parent.c))
        self.assertTrue(
            np.array_equal(hz.Gc[:, : parent.n_cont].toarray(), parent.Gc.toarray())
        )
        self.assertEqual(hz.Gc[:, parent.n_cont :].nnz, 0)
        self.assertTrue(np.array_equal(hz.Gb.toarray(), parent.Gb.toarray()))
        self.assertEqual(issuance.receipt["parent_snapshot_count"], 0)
        self.assertEqual(issuance.receipt["final_core_allocation_count"], 1)
        self.assertTrue(issuance.receipt["direct_live_to_final_detached_copy"])
        self.assertFalse(issuance.receipt["source_buffers_borrowed_by_fresh"])

    def test_global_eta_allocator_is_called_once_after_guarded_replay(self):
        with mock.patch.object(
            fresh_module,
            "hz_reserve_fresh_col_ids_above",
            wraps=fresh_module.hz_reserve_fresh_col_ids_above,
        ) as reserve:
            issuance = self.issue()
        self.assertEqual(reserve.call_count, 1)
        args, kwargs = reserve.call_args
        self.assertEqual(args, (16,))
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(
            kwargs["lower_bound_exclusive"],
            max(
                int(np.max(self.build.hz.col_ids)),
                int(np.max(self.build.hz.bcol_ids)),
            ),
        )
        _consume(issuance)

    def test_fraction_exact_one_hot_lift_and_empty_pattern_fix(self):
        issuance = self.issue()
        fresh = _consume(issuance)
        hz = fresh.hz
        parent = self.build.hz
        patterns = tuple(itertools.product((-1, 1), repeat=4))
        binary_positions = {
            int(stable_id): position
            for position, stable_id in enumerate(hz.bcol_ids.tolist())
        }
        selected_positions = tuple(
            binary_positions[stable_id] for stable_id in self.stable_ids
        )
        equality_start = parent.n_eq
        equality_stop = hz.n_eq
        self.assertEqual(equality_stop - equality_start, 16)

        feasible_count = 0
        for pattern_index, pattern in enumerate(patterns):
            eta = [Fraction(-1)] * 16
            eta[pattern_index] = Fraction(1)
            synthetic_continuous = [Fraction(0)] * hz.n_cont
            synthetic_continuous[parent.n_cont :] = eta
            synthetic_binary = [Fraction(-1)] * hz.n_bin
            for position, phase in zip(selected_positions, pattern):
                synthetic_binary[position] = Fraction(phase)
            residuals = []
            for row in range(equality_start, equality_stop):
                lhs = _exact_csr_row_value(
                    hz.Ac, row, synthetic_continuous
                )
                lhs += _exact_csr_row_value(
                    hz.Ab, row, synthetic_binary
                )
                residuals.append(lhs - Fraction.from_float(float(hz.b[row])))
            if sum(phase == 1 for phase in pattern) <= 1:
                self.assertTrue(all(value == 0 for value in residuals), pattern)
                parent_continuous, parent_binary = (
                    _k4_exact_parent_witness(pattern)
                )
                self.assertTrue(
                    _fraction_hz_feasible(
                        parent, parent_continuous, parent_binary
                    ),
                    pattern,
                )
                fresh_continuous = tuple(parent_continuous) + tuple(eta)
                self.assertTrue(
                    _fraction_hz_feasible(
                        hz, fresh_continuous, parent_binary
                    ),
                    pattern,
                )
                parent_output = _fraction_hz_output(
                    parent, parent_continuous, parent_binary
                )
                fresh_output = _fraction_hz_output(
                    hz, fresh_continuous, parent_binary
                )
                self.assertEqual(fresh_output, parent_output, pattern)
                for rival in self.rivals:
                    with self.subTest(pattern=pattern, rival=rival.rival_id):
                        self.assertEqual(
                            _fraction_objective_projection(
                                fresh_output, rival
                            ),
                            _fraction_objective_projection(
                                parent_output, rival
                            ),
                        )
                independent = (Fraction(1), Fraction(0), Fraction(0))
                self.assertEqual(
                    sum(
                        weight * value
                        for weight, value in zip(
                            independent, fresh_output
                        )
                    ),
                    sum(
                        weight * value
                        for weight, value in zip(
                            independent, parent_output
                        )
                    ),
                )
                feasible_count += 1
            else:
                self.assertTrue(any(value != 0 for value in residuals), pattern)
        self.assertEqual(feasible_count, 5)

    def test_exact_integer_projection_preserved_and_lp_strictly_tightened(self):
        issuance = self.issue()
        fresh = _consume(issuance)
        for rival in self.rivals:
            with self.subTest(rival=rival.rival_id):
                objective = np.asarray(rival.objective, dtype=np.float64)
                parent_milp = _enumerated_integer_maximum(
                    self.build.hz, objective
                )
                fresh_milp = _enumerated_integer_maximum(
                    fresh.hz, objective
                )
                parent_lp = _lp_maximum(self.build.hz, objective)
                fresh_lp = _lp_maximum(fresh.hz, objective)
                self.assertAlmostEqual(parent_milp, fresh_milp, places=7)
                self.assertLessEqual(parent_milp, fresh_lp + 1.0e-7)
                self.assertLessEqual(fresh_lp, parent_lp + 1.0e-7)

        focused = np.asarray(self.rivals[0].objective, dtype=np.float64)
        self.assertLess(
            _lp_maximum(fresh.hz, focused),
            _lp_maximum(self.build.hz, focused) - 1.0e-6,
        )

        # Independent negative control: output row zero is constant in this
        # toy, so the PCOH extension must neither invent nor lose tightness.
        independent = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        self.assertAlmostEqual(
            _lp_maximum(self.build.hz, independent),
            _lp_maximum(fresh.hz, independent),
            places=9,
        )

    def test_witness_decoder_ignores_eta_and_uses_stable_input_ids(self):
        issuance = self.issue()
        fresh = _consume(issuance)
        witness = np.zeros(fresh.hz.n_cont + fresh.hz.n_bin, dtype=np.float64)
        input_positions = {
            int(stable_id): position
            for position, stable_id in enumerate(fresh.hz.col_ids.tolist())
        }
        expected_xi = (0.25, -0.5)
        for stable_id, value in zip(fresh.input_col_ids.tolist(), expected_xi):
            witness[input_positions[int(stable_id)]] = value
        witness[self.build.hz.n_cont : fresh.hz.n_cont] = 1.0
        seed = Bounds(
            torch.tensor([[-1.0, -1.0]], dtype=torch.float64),
            torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        )
        decoded, reason = _hybridz_witness_input(
            fresh.hz,
            witness,
            seed,
            SimpleNamespace(_input_ids=fresh.input_col_ids),
        )
        self.assertEqual(reason, "stable_generator_ids")
        self.assertTrue(
            torch.allclose(
                decoded,
                torch.tensor([[0.25, -0.5]], dtype=torch.float64),
            )
        )

    def test_one_use_registry_copy_tamper_and_concurrent_consume(self):
        issuance = self.issue()
        copied = copy.copy(issuance.capability)
        self.assertIsNot(copied, issuance.capability)
        with self.assertRaises(PhaseConditionedObjectiveHullFreshMaterializationError):
            consume_live_phase_conditioned_objective_hull_fresh_build(
                issuance,
                copied,
                deadline=time.monotonic() + 60.0,
            )
        deep_copied = copy.deepcopy(issuance.capability)
        self.assertIsNot(deep_copied, issuance.capability)
        with self.assertRaises(PhaseConditionedObjectiveHullFreshMaterializationError):
            consume_live_phase_conditioned_objective_hull_fresh_build(
                issuance,
                deep_copied,
                deadline=time.monotonic() + 60.0,
            )

        forged_capability = copy.copy(issuance.capability)
        forged_pair = replace(
            issuance, capability=forged_capability
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullFreshMaterializationError,
            "owner_or_receipt_mismatch",
        ):
            consume_live_phase_conditioned_objective_hull_fresh_build(
                forged_pair,
                forged_capability,
                deadline=time.monotonic() + 60.0,
            )
        self.assertIn(issuance.capability.token, fresh_module._REGISTRY)

        barrier = threading.Barrier(2)
        results = []

        def consume_once():
            barrier.wait()
            try:
                results.append(("ok", _consume(issuance)))
            except Exception as exc:  # exact failure is asserted below
                results.append(("error", exc))

        threads = [threading.Thread(target=consume_once) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        self.assertEqual(sum(kind == "ok" for kind, _ in results), 1)
        self.assertEqual(sum(kind == "error" for kind, _ in results), 1)
        with self.assertRaises(PhaseConditionedObjectiveHullFreshMaterializationError):
            _consume(issuance)

        tampered = self.issue()
        object.__setattr__(tampered, "fresh_frame_sha256", "0" * 64)
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullFreshMaterializationError,
            "owner_or_receipt_mismatch",
        ):
            _consume(tampered)

    def test_registry_issue_baseexception_cleans_private_owner_and_traceback(self):
        registry_before = set(fresh_module._REGISTRY)
        original_check = fresh_module._check_deadline

        def interrupt_after_insert(deadline, stage):
            if stage == "registry_issue_complete":
                raise KeyboardInterrupt("interrupt after registry insert")
            return original_check(deadline, stage)

        caught = None
        with mock.patch.object(
            fresh_module,
            "_check_deadline",
            side_effect=interrupt_after_insert,
        ):
            try:
                self.issue()
            except PhaseConditionedObjectiveHullFreshMaterializationError as exc:
                caught = exc
        self.assertIsNotNone(caught)
        self.assertIn("fresh_registry_issue_interrupted", str(caught))
        self.assertEqual(set(fresh_module._REGISTRY), registry_before)
        cursor = caught.__traceback__
        while cursor is not None:
            frame = cursor.tb_frame
            if frame.f_code.co_filename == fresh_module.__file__:
                self.assertIsNone(frame.f_locals.get("record"))
                self.assertIsNone(frame.f_locals.get("fresh_build"))
            cursor = cursor.tb_next

    def test_private_buffer_tamper_is_terminally_rejected(self):
        issuance = self.issue()
        record = fresh_module._REGISTRY[issuance.capability.token]
        private = record.private_build.hz
        private.ub.setflags(write=True)
        private.ub[-1] = np.nextafter(private.ub[-1], -np.inf)
        private.ub.setflags(write=False)
        caught = None
        try:
            _consume(issuance)
        except PhaseConditionedObjectiveHullFreshMaterializationError as exc:
            caught = exc
        self.assertIsNotNone(caught)
        self.assertIn("terminal_digest_mismatch", str(caught))
        cursor = caught.__traceback__
        while cursor is not None:
            frame = cursor.tb_frame
            if frame.f_code.co_filename == fresh_module.__file__:
                self.assertIsNone(frame.f_locals.get("record"))
                self.assertIsNone(frame.f_locals.get("private_build"))
            cursor = cursor.tb_next

        class FatalValidation(BaseException):
            pass

        interrupted = self.issue()
        caught_fatal = None
        with mock.patch.object(
            fresh_module,
            "_validate_taken_registry_record",
            side_effect=FatalValidation("interrupt after pop"),
        ):
            try:
                _consume(interrupted)
            except PhaseConditionedObjectiveHullFreshMaterializationError as exc:
                caught_fatal = exc
        self.assertIsNotNone(caught_fatal)
        self.assertNotIn(interrupted.capability.token, fresh_module._REGISTRY)
        cursor = caught_fatal.__traceback__
        while cursor is not None:
            frame = cursor.tb_frame
            if frame.f_code.co_filename == fresh_module.__file__:
                self.assertIsNone(frame.f_locals.get("record"))
                self.assertIsNone(frame.f_locals.get("private_build"))
            cursor = cursor.tb_next

        hidden_authority = self.issue()
        hidden_record = fresh_module._REGISTRY[
            hidden_authority.capability.token
        ]
        setattr(
            hidden_record.private_build.hz,
            "_solver_constructive_nonempty_token",
            object(),
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullFreshMaterializationError,
            "attribute_whitelist_changed",
        ):
            _consume(hidden_authority)

    def test_private_metadata_authority_and_build_hidden_attr_tamper_rejected(self):
        metadata_tamper = self.issue()
        metadata_record = fresh_module._REGISTRY[
            metadata_tamper.capability.token
        ]
        forged_metadata = dict(metadata_record.private_build.metadata)
        forged_metadata["proof_authority"] = True
        object.__setattr__(
            metadata_record.private_build,
            "metadata",
            MappingProxyType(forged_metadata),
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullFreshMaterializationError,
            "mode_or_attribute_whitelist_changed",
        ):
            _consume(metadata_tamper)

        hidden_build_authority = self.issue()
        hidden_record = fresh_module._REGISTRY[
            hidden_build_authority.capability.token
        ]
        object.__setattr__(
            hidden_record.private_build,
            "_hidden_proof_authority",
            True,
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullFreshMaterializationError,
            "mode_or_attribute_whitelist_changed",
        ):
            _consume(hidden_build_authority)

    def test_registry_capacity_reserved_before_live_work_and_released_on_error(self):
        single_slot = replace(
            PCOHFreshMaterializationCaps(), max_registry_entries=1
        )
        held = self.issue(caps=single_slot)
        try:
            with mock.patch.object(
                fresh_module, "_validate_source"
            ) as validate_source, mock.patch.object(
                fresh_module,
                "build_live_phase_conditioned_objective_hull_candidate",
            ) as live_adapter:
                with self.assertRaisesRegex(
                    PhaseConditionedObjectiveHullFreshMaterializationError,
                    "capacity_exceeded_before_live_replay",
                ):
                    self.issue(caps=single_slot)
                validate_source.assert_not_called()
                live_adapter.assert_not_called()
        finally:
            _consume(held)

        reservation_before = set(fresh_module._REGISTRY_RESERVATIONS)
        registry_before = set(fresh_module._REGISTRY)
        with mock.patch.object(
            fresh_module,
            "build_live_phase_conditioned_objective_hull_candidate",
            side_effect=RuntimeError("injected live replay failure"),
        ):
            with self.assertRaisesRegex(
                PhaseConditionedObjectiveHullFreshMaterializationError,
                "live_adapter_replay_failed",
            ):
                self.issue(caps=single_slot)
        self.assertEqual(
            set(fresh_module._REGISTRY_RESERVATIONS), reservation_before
        )
        self.assertEqual(set(fresh_module._REGISTRY), registry_before)

        successful = self.issue(caps=single_slot)
        self.assertEqual(
            set(fresh_module._REGISTRY_RESERVATIONS), reservation_before
        )
        _consume(successful)

    def test_capability_ttl_expires_and_sweeps_private_build(self):
        short_caps = replace(
            PCOHFreshMaterializationCaps(),
            capability_ttl_seconds=0.01,
        )
        issuance = self.issue(caps=short_caps)
        token = issuance.capability.token
        self.assertIn(token, fresh_module._REGISTRY)
        time.sleep(0.02)
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullFreshMaterializationError,
            "missing_consumed_or_expired",
        ):
            _consume(issuance)
        self.assertNotIn(token, fresh_module._REGISTRY)

    def test_no_stack_source_and_caps_deadline_fail_closed(self):
        source = inspect.getsource(fresh_module._build_detached_fresh)
        self.assertNotIn("hstack", source)
        self.assertNotIn("vstack", source)
        self.assertNotIn("concatenate", source)
        with mock.patch.object(
            fresh_module,
            "build_live_phase_conditioned_objective_hull_candidate",
        ) as adapter:
            with self.assertRaises(PhaseConditionedObjectiveHullFreshMaterializationError):
                self.issue(deadline=time.monotonic() - 1.0)
            adapter.assert_not_called()

            tiny = replace(
                PCOHFreshMaterializationCaps(), max_parent_variables=1
            )
            with self.assertRaisesRegex(
                PhaseConditionedObjectiveHullFreshMaterializationError,
                "source_dimension_cap_exceeded",
            ):
                self.issue(caps=tiny)
            adapter.assert_not_called()

    def test_row_frame_tamper_is_replayed_and_rejected(self):
        original = fresh_module.materialize_phase_conditioned_objective_hull_row_frame

        def tamper(*args, **kwargs):
            frame = original(*args, **kwargs)
            frame.upper_rhs.setflags(write=True)
            frame.upper_rhs[0] = np.nextafter(frame.upper_rhs[0], -np.inf)
            frame.upper_rhs.setflags(write=False)
            return frame

        with mock.patch.object(
            fresh_module,
            "materialize_phase_conditioned_objective_hull_row_frame",
            side_effect=tamper,
        ):
            with self.assertRaisesRegex(
                PhaseConditionedObjectiveHullFreshMaterializationError,
                "strict_replay_failed",
            ):
                self.issue()

    def test_outer_terminal_source_mutation_fails_without_issuance(self):
        build = _k4_corner_build()
        rivals, selection, stable_ids, certificates, pair_bundle = _sources(build)
        original = fresh_module._build_detached_fresh

        def mutate_after_copy(*args, **kwargs):
            result = original(*args, **kwargs)
            build.hz.ub[0] = np.nextafter(build.hz.ub[0], np.inf)
            return result

        with mock.patch.object(
            fresh_module,
            "_build_detached_fresh",
            side_effect=mutate_after_copy,
        ):
            with self.assertRaisesRegex(
                PhaseConditionedObjectiveHullFreshMaterializationError,
                "outer_terminal_source_seal_mismatch",
            ):
                _issue(
                    build,
                    rivals,
                    selection,
                    stable_ids,
                    certificates,
                    pair_bundle,
                )


class PCOHFreshMaterializedTightnessSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _k4_corner_build()
        (
            cls.rivals,
            cls.selection,
            cls.stable_ids,
            cls.certificates,
            cls.pair_bundle,
        ) = _sources(cls.build, stable_count=2)

    def issue(self):
        return _issue(
            self.build,
            self.rivals,
            self.selection,
            self.stable_ids,
            self.certificates,
            self.pair_bundle,
        )

    def test_k2_empty_highest_upper_is_excluded_and_hlink_is_sound(self):
        issuance = self.issue()
        try:
            summary = issuance.materialized_tightness_summary
            uppers = tuple(
                Fraction.from_float(float.fromhex(value))
                for value in summary.pattern_upper_hex
            )
            active = tuple(
                value
                for value, keep in zip(uppers, summary.active_pattern_mask)
                if keep
            )
            self.assertEqual(
                summary.canonical_patterns,
                tuple(itertools.product((-1, 1), repeat=2)),
            )
            self.assertEqual(
                summary.active_pattern_mask, (True, True, True, False)
            )
            self.assertEqual(max(uppers), uppers[-1])
            self.assertGreater(max(uppers), max(active))
            self.assertEqual(
                Fraction.from_float(
                    float.fromhex(summary.ideal_union_upper_hex)
                ),
                max(active),
            )
            ideal = Fraction.from_float(
                float.fromhex(summary.ideal_union_upper_hex)
            )
            linked = Fraction(*summary.materialized_linked_upper_exact)
            direct = Fraction(*summary.materialized_direct_upper_exact)
            guarded = Fraction(*summary.materialized_guard_upper_exact)
            self.assertLess(ideal, linked)
            self.assertLessEqual(linked, direct)
            self.assertLessEqual(direct, guarded)
            self.assertEqual(
                Fraction(*summary.rounding_tax_exact), linked - ideal
            )
            self.assertLess(
                float.fromhex(summary.final_structural_upper_hex),
                float.fromhex(summary.global_cube_upper_hex),
            )
            self.assertFalse(summary.full_parent_lp_called)
            self.assertFalse(summary.proof_authority)
            self.assertFalse(summary.verdict_authority)
            self.assertEqual(
                issuance.receipt["materialized_tightness_summary_sha256"],
                summary.summary_sha256,
            )
            self.assertTrue(
                verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
                    issuance
                )
            )
        finally:
            discard_live_phase_conditioned_objective_hull_fresh_build(
                issuance, issuance.capability
            )
        self.assertFalse(
            verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
                issuance
            )
        )

    def test_rounding_subnormal_and_inward_hex_fail_closed(self):
        issuance = self.issue()
        try:
            summary = issuance.materialized_tightness_summary
            tiny = Fraction(1, 2**1075)
            self.assertEqual(
                fresh_module._outward_hex(tiny, name="test_subnormal"),
                float.fromhex("0x0.0000000000001p-1022").hex(),
            )
            self.assertEqual(
                fresh_module._strict_float_hex(
                    "0x0.0000000000001p-1022", name="test_subnormal"
                )[1],
                Fraction(1, 2**1074),
            )
            linked = float.fromhex(summary.materialized_linked_upper_hex)
            inward = float(np.nextafter(linked, -np.inf)).hex()
            tampered = replace(
                summary,
                materialized_linked_upper_hex=inward,
                summary_sha256="",
            )
            tampered = replace(
                tampered,
                summary_sha256=fresh_module._canonical_sha256(
                    fresh_module._materialized_tightness_payload(
                        tampered, include_digest=False
                    )
                ),
            )
            with self.assertRaisesRegex(
                PhaseConditionedObjectiveHullFreshMaterializationError,
                "upper_chain_invalid",
            ):
                fresh_module._strict_replay_materialized_tightness_summary(
                    tampered
                )
        finally:
            discard_live_phase_conditioned_objective_hull_fresh_build(
                issuance, issuance.capability
            )

    def test_active_mask_route_objective_and_issuance_tamper_fail_closed(self):
        issuance = self.issue()
        try:
            summary = issuance.materialized_tightness_summary
            cases = (
                (
                    {"active_pattern_mask": (True, True, True, True)},
                    "pattern_cover_invalid",
                    True,
                ),
                (
                    {"conditional_checker_route": "raw_candidate_lp"},
                    "header_invalid",
                    True,
                ),
                (
                    {"objective_binding_sha256": "0" * 64},
                    None,
                    False,
                ),
            )
            for changes, error, strict_fails in cases:
                with self.subTest(changes=changes):
                    tampered = replace(summary, **changes, summary_sha256="")
                    tampered = replace(
                        tampered,
                        summary_sha256=fresh_module._canonical_sha256(
                            fresh_module._materialized_tightness_payload(
                                tampered, include_digest=False
                            )
                        ),
                    )
                    if strict_fails:
                        with self.assertRaisesRegex(
                            PhaseConditionedObjectiveHullFreshMaterializationError,
                            error,
                        ):
                            fresh_module._strict_replay_materialized_tightness_summary(
                                tampered
                            )
                    else:
                        fresh_module._strict_replay_materialized_tightness_summary(
                            tampered
                        )
                    forged = replace(
                        issuance,
                        materialized_tightness_summary=tampered,
                        issuance_sha256="0" * 64,
                    )
                    self.assertFalse(
                        verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
                            forged
                        )
                    )
        finally:
            discard_live_phase_conditioned_objective_hull_fresh_build(
                issuance, issuance.capability
            )

    def test_public_verifier_has_no_full_lp_or_verdict_route(self):
        issuance = self.issue()
        try:
            source = inspect.getsource(
                verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness
            )
            for forbidden in (
                "linprog",
                "milp",
                "hz_base_feasibility",
                "hz_objbound_decide",
            ):
                self.assertNotIn(forbidden, source)
            self.assertTrue(
                verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
                    issuance
                )
            )
        finally:
            discard_live_phase_conditioned_objective_hull_fresh_build(
                issuance, issuance.capability
            )


class PCOHFreshStableIdValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _k4_corner_build()

    def validate(self, build):
        return fresh_module._validate_source(
            build,
            caps=PCOHFreshMaterializationCaps(),
            deadline=time.monotonic() + 60.0,
        )

    def test_sorted_and_unsorted_ids_without_unique_or_intersect(self):
        unsorted = _with_stable_ids(
            self.build,
            col_ids=self.build.hz.col_ids[::-1],
            bcol_ids=self.build.hz.bcol_ids[::-1],
            input_col_ids=self.build.input_col_ids[::-1],
        )
        with mock.patch.object(
            np,
            "unique",
            side_effect=AssertionError("np.unique must not be used"),
        ), mock.patch.object(
            np,
            "intersect1d",
            side_effect=AssertionError("np.intersect1d must not be used"),
        ):
            sorted_layout = self.validate(self.build)
            unsorted_layout = self.validate(unsorted)
        self.assertIs(sorted_layout.hz.col_ids, self.build.hz.col_ids)
        self.assertIs(unsorted_layout.hz.col_ids, unsorted.hz.col_ids)
        self.assertTrue(
            np.array_equal(
                unsorted_layout.input_col_ids,
                self.build.input_col_ids[::-1],
            )
        )

    def test_duplicate_negative_and_continuous_binary_overlap_fail_closed(self):
        duplicate_col = self.build.hz.col_ids.copy()
        duplicate_col[-1] = duplicate_col[0]
        duplicate_binary = self.build.hz.bcol_ids.copy()
        duplicate_binary[-1] = duplicate_binary[0]
        duplicate_input = self.build.input_col_ids.copy()
        duplicate_input[-1] = duplicate_input[0]
        negative_binary = self.build.hz.bcol_ids.copy()
        negative_binary[0] = -1
        negative_input = self.build.input_col_ids.copy()
        negative_input[0] = -1
        overlap_binary = self.build.hz.bcol_ids.copy()
        overlap_binary[0] = self.build.hz.col_ids[0]
        cases = (
            (
                _with_stable_ids(self.build, col_ids=duplicate_col),
                "source_stable_ids_invalid:col_ids",
            ),
            (
                _with_stable_ids(
                    self.build, bcol_ids=duplicate_binary
                ),
                "source_stable_ids_invalid:bcol_ids",
            ),
            (
                _with_stable_ids(
                    self.build, input_col_ids=duplicate_input
                ),
                "source_input_stable_ids_invalid",
            ),
            (
                _with_stable_ids(self.build, bcol_ids=negative_binary),
                "source_stable_ids_invalid:bcol_ids",
            ),
            (
                _with_stable_ids(
                    self.build, input_col_ids=negative_input
                ),
                "source_input_stable_ids_invalid",
            ),
            (
                _with_stable_ids(self.build, bcol_ids=overlap_binary),
                "source_continuous_binary_ids_overlap",
            ),
        )
        for build, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(
                    PhaseConditionedObjectiveHullFreshMaterializationError,
                    error,
                ):
                    self.validate(build)

    def test_bulk_operations_have_cooperative_deadline_checkpoints(self):
        value = np.arange(300_000, dtype=np.int64)
        value[0], value[-1] = value[-1], value[0]
        stages = []
        original = fresh_module._check_deadline

        def record(deadline, stage):
            stages.append(stage)
            return original(deadline, stage)

        with mock.patch.object(
            fresh_module, "_check_deadline", side_effect=record
        ):
            fresh_module._validate_stable_id_vector(
                value,
                name="checkpoint_ids",
                error="checkpoint_invalid",
                deadline=time.monotonic() + 60.0,
                retain_sorted=False,
            )
            fresh_module._reject_continuous_binary_id_overlap(
                value[:200_000],
                np.arange(400_000, 600_000, dtype=np.int64),
                deadline=time.monotonic() + 60.0,
            )
        for expected in (
            "checkpoint_ids_nonnegative_scan",
            "checkpoint_ids_strict_order_scan",
            "checkpoint_ids_sort_copy",
            "checkpoint_ids_inplace_sort",
            "checkpoint_ids_duplicate_scan",
            "source_stable_id_overlap_search",
            "source_stable_id_overlap_clip",
            "source_stable_id_overlap_compare",
        ):
            self.assertIn(expected, stages)

    def test_sorted_and_unsorted_id_validation_tracemalloc_slopes(self):
        sizes = (250_000, 500_000, 1_000_000)

        def peak_for(size, *, unsorted):
            value = np.arange(size, dtype=np.int64)
            if unsorted:
                value[0], value[-1] = value[-1], value[0]
            tracemalloc.start()
            try:
                fresh_module._validate_stable_id_vector(
                    value,
                    name="memory_slope_ids",
                    error="memory_slope_invalid",
                    deadline=time.monotonic() + 60.0,
                    retain_sorted=False,
                )
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return peak

        def overlap_peak_for(size):
            continuous = np.arange(size, dtype=np.int64)
            binary = np.arange(size, 2 * size, dtype=np.int64)
            tracemalloc.start()
            try:
                fresh_module._reject_continuous_binary_id_overlap(
                    continuous,
                    binary,
                    deadline=time.monotonic() + 60.0,
                )
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return peak

        sorted_peaks = tuple(
            peak_for(size, unsorted=False) for size in sizes
        )
        unsorted_peaks = tuple(
            peak_for(size, unsorted=True) for size in sizes
        )
        overlap_peaks = tuple(overlap_peak_for(size) for size in sizes)
        self.assertLessEqual(max(sorted_peaks) - min(sorted_peaks), 512_000)
        self.assertLessEqual(
            unsorted_peaks[1] - unsorted_peaks[0], 10 * 250_000
        )
        self.assertLessEqual(
            unsorted_peaks[2] - unsorted_peaks[1], 10 * 500_000
        )
        self.assertLessEqual(unsorted_peaks[-1], 10 * sizes[-1] + 1_000_000)
        self.assertLessEqual(
            max(overlap_peaks) - min(overlap_peaks), 512_000
        )


class PCOHFreshMaterializerKOneToFourTests(unittest.TestCase):
    def test_old_equality_tags_remain_before_new_equalities_and_old_uppers(self):
        build = _with_zero_source_equality(_k4_corner_build())
        rivals, selection, stable_ids, certificates, pair_bundle = _sources(
            build, stable_count=1
        )
        old_tags = tuple(build.hz._solver_constraint_row_tags)
        issuance = _issue(
            build,
            rivals,
            selection,
            stable_ids,
            certificates,
            pair_bundle,
        )
        fresh = _consume(issuance)
        self.assertEqual(
            fresh.hz._solver_constraint_row_tags,
            old_tags[:1]
            + issuance.equality_row_tags
            + old_tags[1:]
            + issuance.upper_row_tags,
        )
        self.assertEqual(fresh.hz.Ac.getrow(0).nnz, 0)
        self.assertEqual(float(fresh.hz.b[0]), 0.0)

    def test_eta_allocator_int64_overflow_and_conditional_source_fail_closed(self):
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullFreshMaterializationError,
            "eta_id_int64_overflow_or_global_reservation_failed",
        ):
            fresh_module._reserve_eta_ids(
                np.asarray((np.iinfo(np.int64).max,), dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                count=1,
            )

        barrier = threading.Barrier(4)
        ranges = []
        errors = []

        def reserve_two():
            barrier.wait()
            try:
                ranges.append(
                    tuple(
                        int(value)
                        for value in fresh_module._reserve_eta_ids(
                            np.asarray((100,), dtype=np.int64),
                            np.zeros(0, dtype=np.int64),
                            count=2,
                        ).tolist()
                    )
                )
            except Exception as exc:  # asserted empty below
                errors.append(exc)

        threads = [threading.Thread(target=reserve_two) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        self.assertFalse(errors)
        self.assertEqual(len(ranges), 4)
        self.assertTrue(all(stop == start + 1 for start, stop in ranges))
        flattened = tuple(value for pair in ranges for value in pair)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(value > 100 for value in flattened))

        build = _k4_corner_build()
        rivals, selection, stable_ids, certificates, pair_bundle = _sources(
            build, stable_count=1
        )
        setattr(build.hz, "_solver_conditional_tamper", object())
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullFreshMaterializationError,
            "conditional_metadata_unsupported",
        ):
            _issue(
                build,
                rivals,
                selection,
                stable_ids,
                certificates,
                pair_bundle,
            )

    def test_k_one_through_four_exact_counts_and_canonical_eta_ids(self):
        for k in range(1, 5):
            with self.subTest(k=k):
                build = _k4_corner_build()
                rivals, selection, stable_ids, certificates, pair_bundle = _sources(
                    build, stable_count=k
                )
                issuance = _issue(
                    build,
                    rivals,
                    selection,
                    stable_ids,
                    certificates,
                    pair_bundle,
                )
                fresh = _consume(issuance)
                M = 2**k
                H = M - (k + 1) if k >= 1 else 0
                self.assertEqual(fresh.hz.n_cont, build.hz.n_cont + M)
                self.assertEqual(fresh.hz.n_eq, build.hz.n_eq + 1 + k + H)
                self.assertEqual(fresh.hz.n_ub, build.hz.n_ub + 1)
                self.assertEqual(len(issuance.eta_col_ids), M)
                self.assertEqual(
                    issuance.eta_col_ids,
                    tuple(
                        range(
                            issuance.eta_col_ids[0],
                            issuance.eta_col_ids[0] + M,
                        )
                    ),
                )
                parent_ids = set(int(value) for value in build.hz.col_ids.tolist())
                parent_ids.update(int(value) for value in build.hz.bcol_ids.tolist())
                self.assertFalse(parent_ids.intersection(issuance.eta_col_ids))


if __name__ == "__main__":
    unittest.main()
