#!/usr/bin/env python3
"""Soundness and omission-firewall gates for fresh clique materialization."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from itertools import product
from pathlib import Path
import tempfile
import time
from unittest import mock
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf import (
    operator_exact_relu_phase_cliques as clique_module,
    operator_exact_relu_phase_clique_materializer as materializer_module,
    operator_hz as operator_hz_module,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_clique_materializer import (
    OperatorPhaseCliqueMaterialization,
    OperatorPhaseCliqueMaterializationError,
    materialize_verified_operator_phase_clique_cuts,
    maybe_materialize_verified_operator_phase_clique_cuts,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_cliques import (
    OperatorCertifiedPhaseClique,
    OperatorExactReLUPhaseCliqueResult,
    run_operator_exact_relu_phase_cliques_candidate,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge import (
    issue_raw_rival_exact_hardness_receipt,
    select_raw_focused_rivals,
    verify_raw_focused_rival_selection,
    verify_raw_rival_exact_hardness_receipt,
)
from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
    consume_raw_vnnlib_top1_candidate,
    issue_raw_vnnlib_top1_candidate,
    validate_consumed_raw_vnnlib_rival_batch,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _k4_corner_build,
    _relaxed_margin_upper,
    _rivals,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_constructively_nonempty,
)


def _chain(*, mutate_parent=None):
    build = _k4_corner_build()
    if mutate_parent is not None:
        mutate_parent(build)
    rivals = _rivals()
    selection = (
        derive_operator_exact_relu_property_phase_literals(
            build, rivals
        )
    )
    result = run_operator_exact_relu_phase_cliques_candidate(
        build,
        rivals,
        selection,
        deadline=time.monotonic() + 20.0,
    )
    return build, rivals, selection, result


def _materialize(build, rivals, selection, result):
    _ensure_constructive_nonempty_seal(build)
    return materialize_verified_operator_phase_clique_cuts(
        build,
        rivals,
        selection,
        result,
        deadline=time.monotonic() + 20.0,
        caps=result.caps,
    )


def _ensure_constructive_nonempty_seal(build) -> None:
    if build.constructive_nonempty_seal is not None:
        return
    seal = operator_hz_module._make_operator_hz_constructive_nonempty_seal(
        semantic_digest=sparse_hz_semantic_digest(build.hz),
        reason="operator_phase_clique_test_exact_builder_induction",
    )
    object.__setattr__(build, "constructive_nonempty_seal", seal)
    operator_hz_module._register_operator_hz_constructive_nonempty_seal(
        seal, build
    )


def _raw_top1_source() -> str:
    return """
    (set-logic QF_LRA)
    (declare-const X_0 Real)
    (declare-const X_1 Real)
    (declare-const Y_0 Real)
    (declare-const Y_1 Real)
    (declare-const Y_2 Real)
    (assert (>= X_0 -1))
    (assert (<= X_0 1))
    (assert (>= X_1 -1))
    (assert (<= X_1 1))
    (assert (or (<= Y_0 Y_1) (<= Y_0 Y_2)))
    """


def _consumed_raw_top1_batch():
    live = {
        "kind": "TOP1_ROBUST",
        "C": torch.tensor(
            [
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        "thresholds": torch.zeros(
            (1, 2), dtype=torch.float64
        ),
        "M": 2,
        "y_true": torch.tensor([0], dtype=torch.int64),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "top1.vnnlib"
        path.write_text(
            _raw_top1_source().strip() + "\n",
            encoding="utf-8",
        )
        source_sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        candidate = issue_raw_vnnlib_top1_candidate(
            path,
            expected_vnnlib_sha256=source_sha256,
            live_assert_params=live,
            deadline=time.monotonic() + 10.0,
        )
        batch = consume_raw_vnnlib_top1_candidate(
            candidate,
            live_assert_params=live,
            deadline=time.monotonic() + 10.0,
        )
    if not validate_consumed_raw_vnnlib_rival_batch(batch):
        raise AssertionError("raw TOP1 batch ownership was lost")
    return batch


def _live_output_interval_hardness(build, rivals):
    """Produce one exact scheduling vector from this live K4 HZ frame."""

    hz = build.hz
    radius = (
        np.asarray(abs(hz.Gc).sum(axis=1)).reshape(-1)
        + np.asarray(abs(hz.Gb).sum(axis=1)).reshape(-1)
    )
    lower = np.ascontiguousarray(
        hz.c - radius, dtype=np.float64
    )
    upper = np.ascontiguousarray(
        hz.c + radius, dtype=np.float64
    )
    digest = hashlib.sha256()
    digest.update(
        b"act.test.live_operator_hz_output_interval.v1\0"
    )
    digest.update(
        sparse_hz_semantic_digest(hz).encode("ascii")
    )
    for name, value in (("lower", lower), ("upper", upper)):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(
            np.asarray(value.shape, dtype=np.int64).tobytes()
        )
        digest.update(value.tobytes(order="C"))
    exact = []
    for rival in rivals:
        objective = np.asarray(
            rival.objective, dtype=np.float64
        )
        value = float(
            np.dot(
                np.maximum(objective, 0.0),
                upper,
            )
            + np.dot(
                np.minimum(objective, 0.0),
                lower,
            )
            - float(rival.threshold)
        )
        exact.append(value.as_integer_ratio())
    return tuple(exact), digest.hexdigest()


def _residual_property_sha256(rivals) -> str:
    C = np.ascontiguousarray(
        tuple(rival.objective for rival in rivals),
        dtype=np.float64,
    )
    thresholds = np.ascontiguousarray(
        tuple(rival.threshold for rival in rivals),
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    for value in (C, thresholds):
        digest.update(
            np.asarray(value.shape, dtype=np.int64).tobytes()
        )
        digest.update(value.tobytes(order="C"))
    digest.update(b"TOP1_ROBUST")
    return digest.hexdigest()


def _csr_exact(test: unittest.TestCase, left, right) -> None:
    test.assertIs(type(left), sp.csr_matrix)
    test.assertIs(type(right), sp.csr_matrix)
    test.assertEqual(left.shape, right.shape)
    test.assertEqual(left.dtype, right.dtype)
    test.assertTrue(np.array_equal(left.indptr, right.indptr))
    test.assertTrue(np.array_equal(left.indices, right.indices))
    test.assertTrue(np.array_equal(left.data, right.data))


def _legacy_core_copy(
    hz: SparseHZono,
    *,
    upper_rows: int | None = None,
) -> SparseHZono:
    rows = hz.n_ub if upper_rows is None else upper_rows
    return SparseHZono(
        c=hz.c.copy(),
        Gc=hz.Gc.copy(),
        Gb=hz.Gb.copy(),
        Ac=hz.Ac.copy(),
        Ab=hz.Ab.copy(),
        b=hz.b.copy(),
        Auc=hz.Auc[:rows, :].copy(),
        Aub=hz.Aub[:rows, :].copy(),
        ub=hz.ub[:rows].copy(),
        col_ids=hz.col_ids.copy(),
        bcol_ids=hz.bcol_ids.copy(),
    )


def _enumerated_integer_upper(hz, objective) -> float:
    """Exact {-1,+1} enumeration with an LP only over continuous factors."""

    objective = np.asarray(objective, dtype=np.float64).reshape(-1)
    continuous_objective = np.asarray(
        objective @ hz.Gc
    ).reshape(-1)
    binary_objective = np.asarray(
        objective @ hz.Gb
    ).reshape(-1)
    constant = float(np.dot(objective, hz.c))
    best = -np.inf
    for assignment in product((-1.0, 1.0), repeat=hz.n_bin):
        binary = np.asarray(assignment, dtype=np.float64)
        upper_rhs = hz.ub - np.asarray(
            hz.Aub @ binary
        ).reshape(-1)
        equality_rhs = hz.b - np.asarray(
            hz.Ab @ binary
        ).reshape(-1)
        solved = linprog(
            -continuous_objective,
            A_ub=hz.Auc if hz.n_ub else None,
            b_ub=upper_rhs if hz.n_ub else None,
            A_eq=hz.Ac if hz.n_eq else None,
            b_eq=equality_rhs if hz.n_eq else None,
            bounds=[(-1.0, 1.0)] * hz.n_cont,
            method="highs",
        )
        if solved.success:
            value = (
                constant
                + float(np.dot(binary_objective, binary))
                - float(solved.fun)
            )
            best = max(best, value)
    if not np.isfinite(best):
        raise AssertionError("enumerated parent unexpectedly empty")
    return float(best)


class OperatorPhaseCliqueMaterializerTests(unittest.TestCase):
    def test_raw_full_batch_residual_focus_to_fresh_hz_complete_chain(
        self,
    ) -> None:
        build = _k4_corner_build()
        batch = _consumed_raw_top1_batch()
        exact_hardness, interval_frame_sha256 = (
            _live_output_interval_hardness(
                build, batch.rivals
            )
        )
        hardness = issue_raw_rival_exact_hardness_receipt(
            batch,
            exact_hardness,
            live_interval_bounds_sha256=(
                interval_frame_sha256
            ),
            deadline=time.monotonic() + 10.0,
        )
        residual_property_sha256 = _residual_property_sha256(
            batch.rivals
        )
        residual_receipt = {
            "schema": "property_residual_selector_v1",
            "status": "selected",
            "candidate_only": True,
            "proof_authority": False,
            "property_sha256": residual_property_sha256,
            "selection_policy": (
                "facility_first_then_same_rival_joint"
            ),
            "joint_focus_rival_id": 0,
            "rival_ids": [0, 1],
            "targets_selected": 1,
        }
        focused = select_raw_focused_rivals(
            batch,
            hardness,
            focus_count=1,
            explicit_encoded_focus_row=0,
            residual_selector_receipt=residual_receipt,
            residual_selector_property_sha256=(
                residual_property_sha256
            ),
            expected_exact_upper_violations=exact_hardness,
            expected_live_interval_bounds_sha256=(
                interval_frame_sha256
            ),
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            verify_raw_rival_exact_hardness_receipt(
                batch,
                hardness,
                expected_exact_upper_violations=exact_hardness,
                expected_live_interval_bounds_sha256=(
                    interval_frame_sha256
                ),
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertTrue(
            verify_raw_focused_rival_selection(
                batch,
                hardness,
                focused,
                expected_exact_upper_violations=exact_hardness,
                expected_live_interval_bounds_sha256=(
                    interval_frame_sha256
                ),
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertEqual(len(hardness.entries), len(batch.rivals))
        self.assertEqual(
            focused.method,
            "caller_bound_residual_joint_focus_encoded_row_v1",
        )
        self.assertEqual(len(focused.rivals), 1)
        self.assertIs(focused.rivals[0], batch.rivals[0])

        selection = (
            derive_operator_exact_relu_property_phase_literals(
                build, focused.rivals
            )
        )
        clique_result = (
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                focused.rivals,
                selection,
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertEqual(len(clique_result.certificates), 6)
        self.assertIs(
            maybe_materialize_verified_operator_phase_clique_cuts(
                build,
                focused.rivals,
                selection,
                clique_result,
                enabled=False,
                deadline=object(),
                caps=object(),
            ),
            build,
        )
        materialized = _materialize(
            build,
            focused.rivals,
            selection,
            clique_result,
        )
        for rival in batch.rivals:
            self.assertLess(
                _relaxed_margin_upper(
                    materialized.build.hz,
                    rival.objective,
                ),
                0.0,
            )
        self.assertEqual(
            materialized.receipt["focused_property_digest"],
            selection.property_digest,
        )

    def test_actual_k4_fresh_copy_tightens_lp_and_preserves_integer_optimum(
        self,
    ) -> None:
        build, rivals, selection, result = _chain()
        parent = build.hz
        materialized = _materialize(
            build, rivals, selection, result
        )
        self.assertIsInstance(
            materialized, OperatorPhaseCliqueMaterialization
        )
        fresh_build = materialized.build
        fresh = fresh_build.hz

        self.assertIsNot(fresh_build, build)
        self.assertIsNot(fresh, parent)
        self.assertFalse(materialized.proof_authority)
        self.assertFalse(materialized.receipt["proof_authority"])
        self.assertEqual(
            materialized.receipt["verdict_path"],
            "hz_objbound_decide_only",
        )
        self.assertTrue(hz_constructively_nonempty(parent))
        self.assertFalse(hz_constructively_nonempty(fresh))
        private_solver_build = (
            materializer_module
            .consume_operator_phase_clique_materialization_solver_handoff(
                materialized,
                materialized.solver_handoff_capability,
                deadline=time.monotonic() + 5.0,
            )
        )
        self.assertTrue(
            hz_constructively_nonempty(private_solver_build.hz)
        )
        self.assertEqual(
            private_solver_build.hz._solver_constructive_nonempty_reason,
            (
                "operator_hz_redundant_exact_integer_"
                "phase_clique_cuts_v1"
            ),
        )
        self.assertEqual(fresh.n_ub, parent.n_ub + 1)

        for name in ("c", "b", "col_ids", "bcol_ids"):
            parent_array = getattr(parent, name)
            fresh_array = getattr(fresh, name)
            self.assertTrue(
                np.array_equal(parent_array, fresh_array)
            )
            self.assertFalse(
                np.shares_memory(parent_array, fresh_array)
            )
        for name in ("Gc", "Gb", "Ac", "Ab"):
            _csr_exact(
                self, getattr(parent, name), getattr(fresh, name)
            )
            self.assertIsNot(getattr(parent, name), getattr(fresh, name))

        self.assertTrue(
            np.array_equal(parent.Auc.data, fresh.Auc.data)
        )
        self.assertTrue(
            np.array_equal(parent.Auc.indices, fresh.Auc.indices)
        )
        self.assertTrue(
            np.array_equal(
                parent.Auc.indptr,
                fresh.Auc.indptr[: parent.n_ub + 1],
            )
        )
        self.assertTrue(
            np.array_equal(
                parent.Aub.data,
                fresh.Aub.data[: parent.Aub.nnz],
            )
        )
        self.assertTrue(
            np.array_equal(
                parent.Aub.indices,
                fresh.Aub.indices[: parent.Aub.nnz],
            )
        )
        self.assertTrue(
            np.array_equal(
                parent.Aub.indptr,
                fresh.Aub.indptr[: parent.n_ub + 1],
            )
        )
        self.assertTrue(
            np.array_equal(parent.ub, fresh.ub[: parent.n_ub])
        )
        self.assertFalse(np.shares_memory(parent.ub, fresh.ub))

        for name in materialized.receipt[
            "copied_parent_attributes"
        ]:
            self.assertTrue(
                np.array_equal(
                    getattr(parent, name), getattr(fresh, name)
                )
            )
            self.assertFalse(
                np.shares_memory(
                    getattr(parent, name), getattr(fresh, name)
                )
            )
        source_tags = tuple(parent._solver_constraint_row_tags)
        fresh_tags = tuple(fresh._solver_constraint_row_tags)
        self.assertEqual(
            fresh_tags[: len(source_tags)], source_tags
        )
        self.assertEqual(
            fresh_tags[len(source_tags) :],
            materialized.cut_row_tags,
        )
        self.assertTrue(
            all(
                tag.startswith(
                    "operator_exact_relu_phase_clique_cut:v1:"
                )
                for tag in materialized.cut_row_tags
            )
        )
        self.assertEqual(
            fresh._solver_row_constraint_prefix_frames, {}
        )

        before_lp = _relaxed_margin_upper(
            parent, rivals[0].objective
        )
        after_lp = _relaxed_margin_upper(
            fresh, rivals[0].objective
        )
        self.assertAlmostEqual(before_lp, 0.25, places=10)
        self.assertLess(after_lp, 0.0)
        before_integer = _enumerated_integer_upper(
            parent, rivals[0].objective
        )
        after_integer = _enumerated_integer_upper(
            fresh, rivals[0].objective
        )
        self.assertAlmostEqual(
            before_integer, after_integer, places=10
        )
        self.assertAlmostEqual(before_integer, -0.25, places=10)

    def test_success_path_has_one_full_copy_and_strict_readonly_views(
        self,
    ) -> None:
        build, rivals, selection, result = _chain()
        captured = {}
        real_consume = (
            materializer_module
            .consume_verified_operator_phase_clique_snapshot
        )
        real_parent_view = (
            materializer_module._parent_prefix_from_verified_cut
        )

        def capture_consumed(capability, *, deadline):
            snapshot = real_consume(capability, deadline=deadline)
            captured["consumed_cut"] = snapshot.cut_hz
            return snapshot

        def capture_parent_view(cut_hz, **kwargs):
            view = real_parent_view(cut_hz, **kwargs)
            captured["parent_view"] = view
            return view

        with (
            mock.patch.object(
                materializer_module,
                "consume_verified_operator_phase_clique_snapshot",
                side_effect=capture_consumed,
            ),
            mock.patch.object(
                materializer_module,
                "_parent_prefix_from_verified_cut",
                side_effect=capture_parent_view,
            ),
            mock.patch.object(
                materializer_module,
                "_snapshot_sparse_hz",
                wraps=materializer_module._snapshot_sparse_hz,
            ) as full_snapshot,
        ):
            materialized = _materialize(
                build, rivals, selection, result
            )

        public = materialized.build.hz
        parent_view = captured["parent_view"]
        self.assertIs(public, captured["consumed_cut"])
        self.assertEqual(full_snapshot.call_count, 1)
        self.assertEqual(
            materialized.receipt["materializer_full_core_copy_count"],
            1,
        )
        self.assertEqual(
            materialized.receipt["public_core_source"],
            "consumed_verified_cut_zero_copy",
        )
        self.assertEqual(
            materialized.receipt["parent_prefix_core"],
            "strict_readonly_zero_copy_view",
        )
        self.assertTrue(
            materialized.receipt["parent_prefix_readonly"]
        )
        self.assertTrue(
            materialized.receipt[
                "parent_prefix_aliases_public_cut"
            ]
        )
        self.assertTrue(
            materialized.receipt["public_core_readonly"]
        )
        self.assertTrue(
            materialized.receipt["public_private_core_no_alias"]
        )

        legacy_parent = _legacy_core_copy(
            public,
            upper_rows=materialized.receipt["source_upper_rows"],
        )
        legacy_fresh = _legacy_core_copy(public)
        self.assertEqual(
            sparse_hz_semantic_digest(parent_view),
            sparse_hz_semantic_digest(legacy_parent),
        )
        self.assertEqual(
            sparse_hz_semantic_digest(parent_view),
            materialized.parent_semantic_digest,
        )
        self.assertEqual(
            sparse_hz_semantic_digest(legacy_fresh),
            materialized.fresh_semantic_digest,
        )
        for name in ("c", "b", "col_ids", "bcol_ids", "ub"):
            source = getattr(public, name)
            view = getattr(parent_view, name)
            if source.size and view.size:
                self.assertTrue(np.shares_memory(source, view))
            self.assertFalse(view.flags.writeable)
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
            source = getattr(public, name)
            view = getattr(parent_view, name)
            for buffer_name in ("data", "indices", "indptr"):
                source_buffer = getattr(source, buffer_name)
                view_buffer = getattr(view, buffer_name)
                if source_buffer.size and view_buffer.size:
                    self.assertTrue(
                        np.shares_memory(source_buffer, view_buffer)
                    )
                self.assertFalse(view_buffer.flags.writeable)
        with self.assertRaises(ValueError):
            parent_view.ub[0] = 123.0

        private = (
            materializer_module
            .consume_operator_phase_clique_materialization_solver_handoff(
                materialized,
                materialized.solver_handoff_capability,
                deadline=time.monotonic() + 5.0,
            )
        )
        public_arrays = materializer_module._solver_owned_arrays(
            materialized.build
        )
        private_arrays = materializer_module._solver_owned_arrays(
            private
        )
        self.assertFalse(
            any(
                np.shares_memory(public_value, private_value)
                for public_value in public_arrays
                for private_value in private_arrays
            )
        )
        self.assertTrue(
            all(not value.flags.writeable for value in private_arrays)
        )
        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "solver_handoff_capability_invalid",
        ):
            materializer_module.consume_operator_phase_clique_materialization_solver_handoff(
                materialized,
                materialized.solver_handoff_capability,
                deadline=time.monotonic() + 5.0,
            )

    def test_verified_producer_seal_skips_witness_snapshot_and_milp(
        self,
    ) -> None:
        build, rivals, selection, result = _chain()
        with (
            mock.patch.object(
                clique_module,
                "_cut_has_exact_private_nonempty_witness",
                side_effect=AssertionError(
                    "validated producer seal must skip witness snapshot"
                ),
            ),
            mock.patch.object(
                clique_module,
                "milp",
                side_effect=AssertionError(
                    "validated producer seal must skip witness MILP"
                ),
            ),
        ):
            materialized = _materialize(
                build, rivals, selection, result
            )
        self.assertTrue(
            materialized.receipt[
                "producer_nonempty_seal_verified"
            ]
        )

    def test_missing_or_false_producer_seal_fails_closed(self) -> None:
        build, rivals, selection, result = _chain()
        self.assertIsNone(build.constructive_nonempty_seal)
        with (
            mock.patch.object(
                clique_module,
                "_cut_has_exact_private_nonempty_witness",
                side_effect=AssertionError(
                    "missing seal must fail before witness snapshot"
                ),
            ),
            mock.patch.object(
                clique_module,
                "milp",
                side_effect=AssertionError(
                    "missing seal must fail before witness MILP"
                ),
            ),
            self.assertRaisesRegex(
                OperatorPhaseCliqueMaterializationError,
                "producer_nonempty_seal_required",
            ),
        ):
            materialize_verified_operator_phase_clique_cuts(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
                caps=result.caps,
            )

        build, rivals, selection, result = _chain()
        real_consume = (
            materializer_module
            .consume_verified_operator_phase_clique_snapshot
        )

        def consume_with_false_seal(capability, *, deadline):
            snapshot = real_consume(capability, deadline=deadline)
            object.__setattr__(
                snapshot,
                "producer_nonempty_seal_verified",
                False,
            )
            return snapshot

        with (
            mock.patch.object(
                materializer_module,
                "consume_verified_operator_phase_clique_snapshot",
                side_effect=consume_with_false_seal,
            ),
            mock.patch.object(
                materializer_module,
                "verify_consumed_operator_phase_clique_snapshot_integrity",
                return_value=True,
            ),
            self.assertRaisesRegex(
                OperatorPhaseCliqueMaterializationError,
                "verified_snapshot_producer_nonempty_seal_not_verified",
            ),
        ):
            _materialize(build, rivals, selection, result)

    def test_default_off_is_identity_and_does_not_read_candidate(
        self,
    ) -> None:
        build = _k4_corner_build()

        class ExplodesOnRead:
            def __getattribute__(self, name):
                raise AssertionError(f"unexpected read: {name}")

        exploding = ExplodesOnRead()
        with mock.patch.object(
            materializer_module,
            "verify_and_issue_operator_phase_clique_snapshot",
            side_effect=AssertionError("verifier must not run"),
        ):
            unchanged = (
                maybe_materialize_verified_operator_phase_clique_cuts(
                    build,
                    exploding,
                    exploding,
                    exploding,
                    enabled=False,
                    deadline=exploding,
                    caps=exploding,
                )
            )
        self.assertIs(unchanged, build)

    def test_tags_and_provenance_are_rebound_but_caches_are_stripped(
        self,
    ) -> None:
        def decorate(build):
            hz = build.hz
            hz._solver_objbound_stats = {"verdict": "SAFE"}
            hz._solver_objbound_safe_token = object()
            hz._solver_objbound_safe_receipt = {
                "proof_authority": True
            }
            hz._solver_cache = {"pretend": "solver"}
            hz._candidate_receipt = {
                "proof_authority": True
            }

        build, rivals, selection, result = _chain(
            mutate_parent=decorate
        )
        materialized = _materialize(
            build, rivals, selection, result
        )
        fresh = materialized.build.hz
        forbidden = {
            "_solver_objbound_stats",
            "_solver_objbound_safe_token",
            "_solver_objbound_safe_receipt",
            "_solver_cache",
            "_candidate_receipt",
            "operator_hz_metadata",
            "_property_full_input_replay_result",
            "_property_micro_rlt_receipt",
        }
        self.assertFalse(forbidden.intersection(vars(fresh)))
        self.assertEqual(
            set(vars(fresh)).intersection(
                {
                    "full_col_ids",
                    "operator_input_center",
                    "operator_input_radius",
                    "_solver_continuous_column_layer_ids",
                }
            ),
            {
                "full_col_ids",
                "operator_input_center",
                "operator_input_radius",
                "_solver_continuous_column_layer_ids",
            },
        )
        self.assertEqual(
            materialized.build.metadata["schema"],
            "operator_hz_exact_relu_phase_clique_materialized_v1",
        )
        self.assertNotIn(
            "base_nonempty_certificate",
            materialized.build.metadata,
        )

    def test_conditional_prefix_micro_rlt_query_and_replay_modes_reject(
        self,
    ) -> None:
        cases = (
            (
                "conditional",
                lambda build: setattr(
                    build.hz,
                    "_audit_conditional_payload",
                    {"guard": (1, -1)},
                ),
                "conditional_metadata_unsupported",
            ),
            (
                "prefix",
                lambda build: setattr(
                    build.hz,
                    "_solver_row_constraint_prefix_frames",
                    {0: {"schema": "pretend"}},
                ),
                "row_constraint_prefix_frames_must_be_empty",
            ),
            (
                "micro_attr",
                lambda build: setattr(
                    build.hz,
                    "_property_micro_rlt_receipt",
                    {"proof_authority": True},
                ),
                "property_micro_rlt_receipt_unsupported",
            ),
            (
                "full_replay",
                lambda build: setattr(
                    build.hz,
                    "_property_full_input_replay_result",
                    object(),
                ),
                "property_full_input_replay_unsupported",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                build, rivals, selection, result = _chain()
                mutate(build)
                with self.assertRaisesRegex(
                    OperatorPhaseCliqueMaterializationError,
                    "hardened_clique_snapshot_issue_failed",
                ):
                    _materialize(
                        build, rivals, selection, result
                    )

        build, rivals, selection, result = _chain()
        build.metadata["verified_query_dual_feedback"] = {
            "proof_authority": True
        }
        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "hardened_clique_snapshot_issue_failed",
        ):
            _materialize(build, rivals, selection, result)

        build, rivals, selection, result = _chain()
        build.metadata["property_micro_rlt"] = dict(
            build.metadata["property_micro_rlt"],
            enabled=True,
            status="applied",
            live_result_validation_passed=True,
        )
        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "hardened_clique_snapshot_issue_failed",
        ):
            _materialize(build, rivals, selection, result)

    def test_tamper_deadline_caps_and_candidate_attributes_fail_closed(
        self,
    ) -> None:
        build, rivals, selection, result = _chain()
        tampered = replace(result, cliques=())
        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "hardened_clique_snapshot_issue_failed",
        ):
            _materialize(build, rivals, selection, tampered)

        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "deadline_expired",
        ):
            materialize_verified_operator_phase_clique_cuts(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() - 1.0,
                caps=result.caps,
            )

        mismatched_caps = replace(
            result.caps,
            max_top_literals=result.caps.max_top_literals - 1,
        )
        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "hardened_clique_snapshot_issue_failed",
        ):
            materialize_verified_operator_phase_clique_cuts(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
                caps=mismatched_caps,
            )

        result.hz._solver_objbound_safe_token = object()
        result.hz._solver_objbound_safe_receipt = {
            "proof_authority": True
        }
        materialized = _materialize(
            build, rivals, selection, result
        )
        self.assertFalse(
            {
                "_solver_objbound_safe_token",
                "_solver_objbound_safe_receipt",
            }.intersection(vars(materialized.build.hz))
        )

    def test_snapshot_readonly_tamper_fails_before_public_reuse(
        self,
    ) -> None:
        build, rivals, selection, result = _chain()
        real_consume = (
            materializer_module
            .consume_verified_operator_phase_clique_snapshot
        )

        def consume_then_tamper(capability, *, deadline):
            snapshot = real_consume(capability, deadline=deadline)
            snapshot.cut_hz.c.setflags(write=True)
            return snapshot

        with mock.patch.object(
            materializer_module,
            "consume_verified_operator_phase_clique_snapshot",
            side_effect=consume_then_tamper,
        ):
            with self.assertRaisesRegex(
                OperatorPhaseCliqueMaterializationError,
                "consumed_verified_cut_dense_buffer_writeable_c",
            ):
                _materialize(build, rivals, selection, result)

    def test_handoff_aba_pid_ttl_and_deadline_fail_closed(self) -> None:
        first = _materialize(*_chain())
        second = _materialize(*_chain())
        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "solver_handoff_capability_malformed",
        ):
            materializer_module.consume_operator_phase_clique_materialization_solver_handoff(
                second,
                first.solver_handoff_capability,
                deadline=time.monotonic() + 5.0,
            )

        process_id = materializer_module.os.getpid()
        with mock.patch.object(
            materializer_module.os,
            "getpid",
            return_value=process_id + 1,
        ):
            self.assertFalse(
                materializer_module.validate_operator_phase_clique_materialization_solver_handoff(
                    first
                )
            )
            with self.assertRaisesRegex(
                OperatorPhaseCliqueMaterializationError,
                "solver_handoff_capability_malformed",
            ):
                materializer_module.consume_operator_phase_clique_materialization_solver_handoff(
                    first,
                    first.solver_handoff_capability,
                    deadline=time.monotonic() + 5.0,
                )

        token = second.solver_handoff_capability.token
        with materializer_module._SOLVER_HANDOFF_REGISTRY_LOCK:
            record = materializer_module._SOLVER_HANDOFF_REGISTRY[token]
            materializer_module._SOLVER_HANDOFF_REGISTRY[token] = replace(
                record,
                expires_monotonic=time.monotonic() - 1.0,
            )
        self.assertFalse(
            materializer_module.validate_operator_phase_clique_materialization_solver_handoff(
                second
            )
        )
        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "solver_handoff_capability_invalid",
        ):
            materializer_module.consume_operator_phase_clique_materialization_solver_handoff(
                second,
                second.solver_handoff_capability,
                deadline=time.monotonic() + 5.0,
            )

        third = _materialize(*_chain())
        with self.assertRaisesRegex(
            OperatorPhaseCliqueMaterializationError,
            "deadline_expired_before_solver_handoff_consume",
        ):
            materializer_module.consume_operator_phase_clique_materialization_solver_handoff(
                third,
                third.solver_handoff_capability,
                deadline=time.monotonic() - 1.0,
            )

    def test_candidate_dataclass_equality_is_never_used(
        self,
    ) -> None:
        build, rivals, selection, result = _chain()

        def equality_bomb(*_args, **_kwargs):
            raise AssertionError("candidate equality was invoked")

        with (
            mock.patch.object(
                OperatorExactReLUPhaseCliqueResult,
                "__eq__",
                equality_bomb,
            ),
            mock.patch.object(
                OperatorCertifiedPhaseClique,
                "__eq__",
                equality_bomb,
            ),
            mock.patch.object(
                PhaseLiteral,
                "__eq__",
                equality_bomb,
            ),
        ):
            materialized = _materialize(
                build, rivals, selection, result
            )
        self.assertFalse(materialized.proof_authority)

    def test_live_objects_are_never_read_after_snapshot_consume(
        self,
    ) -> None:
        build, rivals, selection, result = _chain()
        original_parent_c = build.hz.c.copy()
        original_input_ids = build.input_col_ids.copy()
        real_consume = (
            materializer_module
            .consume_verified_operator_phase_clique_snapshot
        )

        def consume_then_mutate(capability, *, deadline):
            snapshot = real_consume(
                capability, deadline=deadline
            )
            build.hz.c[:] = 12345.0
            build.hz._solver_constraint_row_tags = ("forged",)
            build.input_col_ids[:] = -77
            result.hz.c[:] = -54321.0
            object.__setattr__(
                selection.caps, "max_rivals", 1
            )
            object.__setattr__(
                result.caps, "max_top_literals", 1
            )
            return snapshot

        with mock.patch.object(
            materializer_module,
            "consume_verified_operator_phase_clique_snapshot",
            side_effect=consume_then_mutate,
        ):
            materialized = _materialize(
                build, rivals, selection, result
            )
        fresh = materialized.build.hz
        self.assertTrue(
            np.array_equal(
                fresh.c,
                # The clique cut changes constraints, never output centers.
                original_parent_c,
            )
        )
        self.assertTrue(
            np.array_equal(
                materialized.build.input_col_ids,
                original_input_ids,
            )
        )
        self.assertLess(
            _relaxed_margin_upper(
                fresh, rivals[0].objective
            ),
            0.0,
        )

    def test_constructive_token_is_never_issued_before_verification(
        self,
    ) -> None:
        build, rivals, selection, result = _chain()
        with (
            mock.patch.object(
                materializer_module,
                "verify_and_issue_operator_phase_clique_snapshot",
                return_value=None,
            ) as verifier,
            mock.patch.object(
                materializer_module,
                "hz_mark_constructively_nonempty",
                side_effect=AssertionError(
                    "constructive token issued before verification"
                ),
            ) as issuer,
        ):
            with self.assertRaisesRegex(
                OperatorPhaseCliqueMaterializationError,
                "hardened_clique_snapshot_issue_failed",
            ):
                _materialize(
                    build, rivals, selection, result
                )
        verifier.assert_called_once()
        issuer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
