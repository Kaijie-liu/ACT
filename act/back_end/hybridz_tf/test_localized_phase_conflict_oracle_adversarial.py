#!/usr/bin/env python3
"""Adversarial regressions for the frozen localized conflict oracle.

These tests deliberately exercise caller-buffer races, full-width binary
semantics, source-row ordering, malformed parents, and hard resource caps.
They never give a localized candidate proof authority or materialize a cut.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import act.back_end.hybridz_tf.localized_phase_conflict_oracle as localized
import act.back_end.hybridz_tf.persistent_phase_conflict_oracle as pco
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
)
from act.back_end.solver.solver_hz import SparseHZono


_PROPERTY_DIGEST = "7" * 64


def _csr(rows, *, n_rows: int, n_columns: int) -> sp.csr_matrix:
    matrix = sp.csr_matrix(
        np.asarray(rows, dtype=np.float64).reshape(n_rows, n_columns),
        dtype=np.float64,
    )
    matrix.sort_indices()
    return matrix


def _parent(
    *,
    n_cont: int,
    n_bin: int,
    upper_cont,
    upper_bin,
    upper_rhs,
    equality_cont=(),
    equality_bin=(),
    equality_rhs=(),
) -> SparseHZono:
    n_upper = len(upper_rhs)
    n_equality = len(equality_rhs)
    return SparseHZono(
        c=np.zeros(1, dtype=np.float64),
        Gc=sp.csr_matrix((1, n_cont), dtype=np.float64),
        Gb=sp.csr_matrix((1, n_bin), dtype=np.float64),
        Ac=_csr(
            equality_cont,
            n_rows=n_equality,
            n_columns=n_cont,
        ),
        Ab=_csr(
            equality_bin,
            n_rows=n_equality,
            n_columns=n_bin,
        ),
        b=np.asarray(equality_rhs, dtype=np.float64),
        Auc=_csr(
            upper_cont,
            n_rows=n_upper,
            n_columns=n_cont,
        ),
        Aub=_csr(
            upper_bin,
            n_rows=n_upper,
            n_columns=n_bin,
        ),
        ub=np.asarray(upper_rhs, dtype=np.float64),
        col_ids=np.arange(101, 101 + n_cont, dtype=np.int64),
        bcol_ids=np.arange(1001, 1001 + n_bin, dtype=np.int64),
    )


def _direct_conflict() -> SparseHZono:
    # x+s <= 0, -x+t <= 0; (++ ) is infeasible.
    return _parent(
        n_cont=1,
        n_bin=2,
        upper_cont=((1.0,), (-1.0,)),
        upper_bin=((1.0, 0.0), (0.0, 1.0)),
        upper_rhs=(0.0, 0.0),
    )


def _two_hop_conflict() -> SparseHZono:
    # x+s<=0, -x+a<=0, -a+b<=0, -b+z<=0, -z+t<=0.
    return _parent(
        n_cont=4,
        n_bin=2,
        upper_cont=(
            (1.0, 0.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0, 0.0),
            (0.0, -1.0, 1.0, 0.0),
            (0.0, 0.0, -1.0, 1.0),
            (0.0, 0.0, 0.0, -1.0),
        ),
        upper_bin=(
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
        ),
        upper_rhs=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _seals(parent: SparseHZono) -> tuple[str, str]:
    parent_digest = sparse_hz_semantic_digest(parent)
    source_digest = pco._ordered_source_frame_digest(
        parent,
        parent_digest=parent_digest,
        deadline=time.monotonic() + 10.0,
    )
    return parent_digest, source_digest


def _bound_pair(
    parent: SparseHZono,
    phases: tuple[int, int] = (1, 1),
    *,
    parent_digest: str | None = None,
) -> tuple[PhaseLiteral, PhaseLiteral]:
    digest = (
        sparse_hz_semantic_digest(parent)
        if parent_digest is None
        else parent_digest
    )
    return tuple(
        PhaseLiteral(
            stable_bcol_id=int(parent.bcol_ids[index]),
            phase=int(phase),
            binding_digest=pco._literal_binding_digest(
                parent_digest=digest,
                property_digest=_PROPERTY_DIGEST,
                stable_bcol_id=int(parent.bcol_ids[index]),
                phase=int(phase),
            ),
        )
        for index, phase in enumerate(phases)
    )


def _run(
    parent: SparseHZono,
    phases: tuple[int, int] = (1, 1),
    *,
    seconds: float = 20.0,
):
    parent_digest, source_digest = _seals(parent)
    return localized.run_localized_phase_conflict_oracle_candidate(
        parent,
        _bound_pair(parent, phases, parent_digest=parent_digest),
        property_digest=_PROPERTY_DIGEST,
        parent_digest=parent_digest,
        source_frame_digest=source_digest,
        deadline=time.monotonic() + seconds,
        enabled=True,
    )


class LocalizedPhaseConflictAdversarialTests(unittest.TestCase):
    def test_nested_live_aba_restored_during_private_replay_is_safe(
        self,
    ) -> None:
        parent = _direct_conflict()
        original_verify = (
            localized.verify_exact_dual_ray_conflict_certificate
        )
        original_rhs = float(parent.ub[0])
        calls = []

        def verify_while_live_parent_is_transiently_changed(
            private_parent, *args, **kwargs
        ):
            self.assertIsNot(private_parent, parent)
            self.assertFalse(
                np.shares_memory(private_parent.ub, parent.ub)
            )
            parent.ub[0] = np.nextafter(original_rhs, np.inf)
            try:
                calls.append(private_parent)
                return original_verify(private_parent, *args, **kwargs)
            finally:
                parent.ub[0] = original_rhs

        with mock.patch.object(
            localized,
            "verify_exact_dual_ray_conflict_certificate",
            side_effect=verify_while_live_parent_is_transiently_changed,
        ):
            result = _run(parent)
        self.assertEqual(len(calls), 1)
        self.assertTrue(result.edge_accepted)
        self.assertTrue(result.parent_unchanged)
        self.assertEqual(float(parent.ub[0]), original_rhs)

    def test_nested_live_mutation_not_restored_revokes_candidate(
        self,
    ) -> None:
        parent = _direct_conflict()
        original_verify = (
            localized.verify_exact_dual_ray_conflict_certificate
        )

        def verify_then_leave_live_parent_changed(
            private_parent, *args, **kwargs
        ):
            self.assertIsNot(private_parent, parent)
            parent.ub[0] = np.nextafter(parent.ub[0], np.inf)
            return original_verify(private_parent, *args, **kwargs)

        with mock.patch.object(
            localized,
            "verify_exact_dual_ray_conflict_certificate",
            side_effect=verify_then_leave_live_parent_changed,
        ):
            result = _run(parent)
        self.assertEqual(result.status, "parent_mutated")
        self.assertEqual(result.reason, "live_parent_terminal_seal_mismatch")
        self.assertFalse(result.edge_accepted)
        self.assertFalse(result.parent_unchanged)
        self.assertIsNone(result.certificate)

    def test_every_private_snapshot_buffer_is_no_alias_and_read_only(
        self,
    ) -> None:
        parent = _direct_conflict()
        original_build = localized._build_incidence_frame
        snapshots = []

        def inspect_snapshot(snapshot, blocks, *, deadline):
            self.assertIsNot(snapshot, parent)
            for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
                private_value = getattr(snapshot, name)
                live_value = getattr(parent, name)
                self.assertIsNotNone(private_value)
                self.assertFalse(private_value.flags.writeable, name)
                self.assertFalse(
                    np.shares_memory(private_value, live_value), name
                )
            for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                private_matrix = getattr(snapshot, name)
                live_matrix = getattr(parent, name)
                self.assertIsNot(private_matrix, live_matrix)
                for field in ("data", "indices", "indptr"):
                    private_value = getattr(private_matrix, field)
                    live_value = getattr(live_matrix, field)
                    self.assertFalse(
                        private_value.flags.writeable, f"{name}.{field}"
                    )
                    self.assertFalse(
                        np.shares_memory(private_value, live_value),
                        f"{name}.{field}",
                    )
            snapshots.append(snapshot)
            return original_build(snapshot, blocks, deadline=deadline)

        with mock.patch.object(
            localized,
            "_build_incidence_frame",
            side_effect=inspect_snapshot,
        ):
            result = _run(parent)
        self.assertTrue(result.edge_accepted)
        self.assertEqual(len(snapshots), 1)

    def test_third_unfixed_binary_remains_in_full_width_feasible_model(
        self,
    ) -> None:
        # s+t+u<=1 is feasible at s=t=+1 because unfixed u may be -1.
        parent = _parent(
            n_cont=0,
            n_bin=3,
            upper_cont=((),),
            upper_bin=((1.0, 1.0, 1.0),),
            upper_rhs=(1.0,),
        )
        independent = linprog(
            np.zeros(3, dtype=np.float64),
            A_ub=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
            b_ub=np.asarray([1.0], dtype=np.float64),
            bounds=((1.0, 1.0), (1.0, 1.0), (-1.0, 1.0)),
            method="highs",
        )
        self.assertTrue(independent.success)
        self.assertEqual(float(independent.x[2]), -1.0)

        result = _run(parent)
        self.assertFalse(result.edge_accepted)
        self.assertIsNone(result.certificate)
        self.assertEqual(result.tiers[0].status, "feasible_or_unknown")
        self.assertEqual(result.tiers[0].model_columns, 3)
        self.assertEqual(result.tiers[0].model_nonzeros, 3)

    def test_asymmetric_upper_then_equality_row_identity_is_exact(
        self,
    ) -> None:
        # upper[0] is an unrelated y<=1 row.  The conflict uses upper[1]
        # (-x+t<=0) and equality[0] (x+s=0), whose global ids are 1 and 2.
        parent = _parent(
            n_cont=2,
            n_bin=2,
            upper_cont=((0.0, 1.0), (-1.0, 0.0)),
            upper_bin=((0.0, 0.0), (0.0, 1.0)),
            upper_rhs=(1.0, 0.0),
            equality_cont=((1.0, 0.0),),
            equality_bin=((1.0, 0.0),),
            equality_rhs=(0.0,),
        )
        result = _run(parent)
        self.assertTrue(result.edge_accepted)
        self.assertEqual(
            result.tiers[0].ordered_global_row_ids, (1, 2)
        )
        self.assertEqual(
            result.tiers[0].ordered_row_refs,
            (
                localized.RowRef("upper", 1),
                localized.RowRef("equality", 0),
            ),
        )
        self.assertEqual(
            {
                (
                    term.kind,
                    term.local_row_index,
                    term.global_row_index,
                )
                for term in result.certificate.source_terms
            },
            {("upper", 1, 1), ("equality_pos", 0, 2)},
        )

    def test_conditional_metadata_is_rejected_but_unrelated_metadata_is_not(
        self,
    ) -> None:
        unrelated = _direct_conflict()
        unrelated._audit_note = {"diagnostic": "not semantic"}
        self.assertTrue(_run(unrelated).edge_accepted)

        cases = (
            (
                "_audit_conditional_payload",
                {"nested": (np.zeros(2, dtype=np.float64),)},
            ),
            (
                "_solver_conditional_property_rows_applied",
                ("sealed", 1),
            ),
        )
        for name, payload in cases:
            with self.subTest(name=name):
                parent = _direct_conflict()
                setattr(parent, name, payload)
                parent_digest, source_digest = _seals(parent)
                with mock.patch.object(
                    localized,
                    "_build_incidence_frame",
                    side_effect=AssertionError(
                        "conditional parent reached incidence construction"
                    ),
                ), self.assertRaisesRegex(
                    localized.LocalizedPhaseConflictOracleError,
                    "parent_conditional_metadata_unsupported",
                ):
                    localized.run_localized_phase_conflict_oracle_candidate(
                        parent,
                        _bound_pair(
                            parent,
                            parent_digest=parent_digest,
                        ),
                        property_digest=_PROPERTY_DIGEST,
                        parent_digest=parent_digest,
                        source_frame_digest=source_digest,
                        deadline=time.monotonic() + 20.0,
                        enabled=True,
                    )

    def test_corrupt_csr_parents_fail_closed_before_solving(self) -> None:
        explicit_zero = _direct_conflict()
        explicit_zero.Auc.data[0] = 0.0
        parent_digest, source_digest = _seals(explicit_zero)
        with mock.patch.object(
            localized,
            "_solve_tier",
            side_effect=AssertionError("malformed CSR reached solver"),
        ), self.assertRaisesRegex(
            localized.LocalizedPhaseConflictOracleError,
            "Auc_malformed_csr",
        ):
            localized.run_localized_phase_conflict_oracle_candidate(
                explicit_zero,
                _bound_pair(
                    explicit_zero,
                    parent_digest=parent_digest,
                ),
                property_digest=_PROPERTY_DIGEST,
                parent_digest=parent_digest,
                source_frame_digest=source_digest,
                deadline=time.monotonic() + 20.0,
                enabled=True,
            )

        out_of_range = _direct_conflict()
        parent_digest, source_digest = _seals(out_of_range)
        pair = _bound_pair(out_of_range, parent_digest=parent_digest)
        out_of_range.Auc.indices[0] = out_of_range.n_cont
        with mock.patch.object(
            localized,
            "_solve_tier",
            side_effect=AssertionError("malformed CSR reached solver"),
        ), self.assertRaisesRegex(
            RuntimeError,
            "semantic_Auc_malformed",
        ):
            localized.run_localized_phase_conflict_oracle_candidate(
                out_of_range,
                pair,
                property_digest=_PROPERTY_DIGEST,
                parent_digest=parent_digest,
                source_frame_digest=source_digest,
                deadline=time.monotonic() + 20.0,
                enabled=True,
            )

    def test_snapshot_cap_is_explicitly_catchable(self) -> None:
        parent = _direct_conflict()
        parent_digest, source_digest = _seals(parent)
        with mock.patch.object(
            localized, "_HARD_SNAPSHOT_BUFFER_BYTES", 1
        ), self.assertRaisesRegex(
            MemoryError, "snapshot_buffer_byte_cap_exceeded"
        ):
            localized.run_localized_phase_conflict_oracle_candidate(
                parent,
                _bound_pair(parent, parent_digest=parent_digest),
                property_digest=_PROPERTY_DIGEST,
                parent_digest=parent_digest,
                source_frame_digest=source_digest,
                deadline=time.monotonic() + 20.0,
                enabled=True,
            )

    def test_pattern_and_frontier_caps_return_structured_no_edge(
        self,
    ) -> None:
        with mock.patch.object(
            localized, "_HARD_PATTERN_PEAK_BYTES", 0
        ):
            pattern = _run(_direct_conflict())
        self.assertEqual(pattern.status, "resource_cap")
        self.assertEqual(pattern.reason, "pattern_peak_byte_cap_exceeded")
        self.assertFalse(pattern.edge_accepted)
        self.assertIsNone(pattern.certificate)

        observed_known_column_counts = []
        original_rows_for_columns = localized._rows_for_columns

        def record_known_columns(frame, columns, *, deadline):
            observed_known_column_counts.append(len(set(columns)))
            return original_rows_for_columns(
                frame, columns, deadline=deadline
            )

        frontier_cap = 4
        with mock.patch.object(
            localized, "_HARD_FRONTIER_COLUMNS", frontier_cap
        ), mock.patch.object(
            localized,
            "_rows_for_columns",
            side_effect=record_known_columns,
        ):
            frontier = _run(_two_hop_conflict())
        self.assertEqual(frontier.status, "resource_cap")
        self.assertEqual(frontier.reason, "frontier_column_cap_exceeded")
        self.assertFalse(frontier.edge_accepted)
        self.assertIsNone(frontier.certificate)
        self.assertTrue(observed_known_column_counts)
        self.assertLessEqual(
            max(observed_known_column_counts), frontier_cap + 2
        )
        self.assertTrue(
            all(
                tier.selected_columns <= frontier_cap + 2
                for tier in frontier.tiers
            )
        )


if __name__ == "__main__":
    unittest.main()
