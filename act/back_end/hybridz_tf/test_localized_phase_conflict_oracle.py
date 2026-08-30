#!/usr/bin/env python3
"""Independent soundness toys for the localized phase-conflict candidate."""

from __future__ import annotations

from dataclasses import replace
import itertools
import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

import act.back_end.hybridz_tf.localized_phase_conflict_oracle as localized
import act.back_end.hybridz_tf.persistent_phase_conflict_oracle as pco
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
    _copy_parent_with_clique_cut,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _base_milp_matrices,
    hz_fresh_col_ids,
)


_PROPERTY_DIGEST = "d" * 64


def _fresh_ids(count: int) -> np.ndarray:
    return (
        hz_fresh_col_ids(count, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )


def _matrix(rows, *, n_rows: int, n_columns: int) -> sp.csr_matrix:
    return sp.csr_matrix(
        np.asarray(rows, dtype=np.float64).reshape(n_rows, n_columns)
    )


def _hz(
    *,
    n_cont: int,
    upper_cont,
    upper_bin,
    upper_rhs,
    equality_cont=(),
    equality_bin=(),
    equality_rhs=(),
) -> SparseHZono:
    n_ub = len(upper_rhs)
    n_eq = len(equality_rhs)
    n_bin = 2
    return SparseHZono(
        c=np.zeros(1, dtype=np.float64),
        Gc=sp.csr_matrix((1, n_cont), dtype=np.float64),
        Gb=sp.csr_matrix((1, n_bin), dtype=np.float64),
        Ac=_matrix(equality_cont, n_rows=n_eq, n_columns=n_cont),
        Ab=_matrix(equality_bin, n_rows=n_eq, n_columns=n_bin),
        b=np.asarray(equality_rhs, dtype=np.float64),
        Auc=_matrix(upper_cont, n_rows=n_ub, n_columns=n_cont),
        Aub=_matrix(upper_bin, n_rows=n_ub, n_columns=n_bin),
        ub=np.asarray(upper_rhs, dtype=np.float64),
        col_ids=_fresh_ids(n_cont),
        bcol_ids=_fresh_ids(n_bin),
    )


def _h0() -> SparseHZono:
    # x+s <= 0, -x+t <= 0.
    return _hz(
        n_cont=1,
        upper_cont=((1.0,), (-1.0,)),
        upper_bin=((1.0, 0.0), (0.0, 1.0)),
        upper_rhs=(0.0, 0.0),
    )


def _h1() -> SparseHZono:
    # x+s <= 0, -x+y <= 0, -y+t <= 0.
    return _hz(
        n_cont=2,
        upper_cont=((1.0, 0.0), (-1.0, 1.0), (0.0, -1.0)),
        upper_bin=((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
        upper_rhs=(0.0, 0.0, 0.0),
    )


def _h2() -> SparseHZono:
    # x+s <= 0, -x+a <= 0, -a+b <= 0, -b+z <= 0, -z+t <= 0.
    return _hz(
        n_cont=4,
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


def _mixed() -> SparseHZono:
    # x+s <= 0, -x-t <= 0; only (s,t)=(+1,-1) conflicts.
    return _hz(
        n_cont=1,
        upper_cont=((1.0,), (-1.0,)),
        upper_bin=((1.0, 0.0), (0.0, -1.0)),
        upper_rhs=(0.0, 0.0),
    )


def _equality_positive() -> SparseHZono:
    # x+s=0, -x+t<=0; the ray uses equality_pos.
    return _hz(
        n_cont=1,
        equality_cont=((1.0,),),
        equality_bin=((1.0, 0.0),),
        equality_rhs=(0.0,),
        upper_cont=((-1.0,),),
        upper_bin=((0.0, 1.0),),
        upper_rhs=(0.0,),
    )


def _equality_negative() -> SparseHZono:
    # x+s=0, x-t<=0 at (--); the ray uses equality_neg.
    return _hz(
        n_cont=1,
        equality_cont=((1.0,),),
        equality_bin=((1.0, 0.0),),
        equality_rhs=(0.0,),
        upper_cont=((1.0,),),
        upper_bin=((0.0, -1.0),),
        upper_rhs=(0.0,),
    )


def _full_width_witness() -> SparseHZono:
    # y+s+t<=1 remains feasible at (++), but becomes false if y is dropped.
    return _hz(
        n_cont=1,
        upper_cont=((1.0,),),
        upper_bin=((1.0, 1.0),),
        upper_rhs=(1.0,),
    )


def _bound_pair(
    parent: SparseHZono,
    phases: tuple[int, int],
    *,
    property_digest: str = _PROPERTY_DIGEST,
) -> tuple[PhaseLiteral, PhaseLiteral]:
    parent_digest = sparse_hz_semantic_digest(parent)
    return tuple(
        PhaseLiteral(
            stable_bcol_id=int(parent.bcol_ids[index]),
            phase=phase,
            binding_digest=pco._literal_binding_digest(
                parent_digest=parent_digest,
                property_digest=property_digest,
                stable_bcol_id=int(parent.bcol_ids[index]),
                phase=phase,
            ),
        )
        for index, phase in enumerate(phases)
    )


def _run(
    parent: SparseHZono,
    phases: tuple[int, int],
    *,
    enabled: bool = True,
    seconds: float = 20.0,
    max_source_terms: int = 128,
):
    parent_digest = sparse_hz_semantic_digest(parent)
    source_digest = pco._ordered_source_frame_digest(
        parent,
        parent_digest=parent_digest,
        deadline=time.monotonic() + max(10.0, seconds),
    )
    return localized.run_localized_phase_conflict_oracle_candidate(
        parent,
        _bound_pair(parent, phases),
        property_digest=_PROPERTY_DIGEST,
        parent_digest=parent_digest,
        source_frame_digest=source_digest,
        deadline=time.monotonic() + seconds,
        enabled=enabled,
        max_source_terms=max_source_terms,
    )


def _continuous_range(
    parent: SparseHZono,
    phases: tuple[int, int],
    variable: int,
) -> tuple[float, float] | None:
    phase = np.asarray(phases, dtype=np.float64)
    ub_rhs = np.asarray(parent.ub, dtype=np.float64) - parent.Aub @ phase
    eq_rhs = np.asarray(parent.b, dtype=np.float64) - parent.Ab @ phase
    objective = np.zeros(parent.n_cont, dtype=np.float64)
    objective[variable] = 1.0
    kwargs = {
        "A_ub": parent.Auc,
        "b_ub": ub_rhs,
        "A_eq": parent.Ac,
        "b_eq": eq_rhs,
        "bounds": [(-1.0, 1.0)] * parent.n_cont,
        "method": "highs",
    }
    lower = linprog(objective, **kwargs)
    if not lower.success:
        return None
    upper = linprog(-objective, **kwargs)
    if not upper.success:
        return None
    return float(lower.fun), float(-upper.fun)


def _milp_feasible(
    parent: SparseHZono,
    phases: tuple[int, int],
) -> bool:
    A, row_lower, row_upper, lower, upper, integrality = (
        _base_milp_matrices(parent)
    )
    for offset, phase in enumerate(phases):
        q_value = 0.5 * (phase + 1)
        lower[parent.n_cont + offset] = q_value
        upper[parent.n_cont + offset] = q_value
    result = milp(
        c=np.zeros(parent.n_cont + parent.n_bin, dtype=np.float64),
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A, row_lower, row_upper),
        options={"presolve": False, "time_limit": 5.0},
    )
    return bool(result.success)


class LocalizedPhaseConflictSoundnessTests(unittest.TestCase):
    def test_h0_direct_conflict_replays_exactly(self) -> None:
        result = _run(_h0(), (1, 1))
        self.assertEqual(result.status, "certified_conflict")
        self.assertTrue(result.edge_accepted)
        self.assertTrue(result.parent_unchanged)
        self.assertEqual(result.certificate.contradiction, -2)
        self.assertEqual(len(result.tiers), 1)
        self.assertEqual(result.tiers[0].expansion_depth, 0)
        self.assertEqual(result.tiers[0].exact_replay_status, "accepted")

    def test_h1_requires_one_incidence_expansion(self) -> None:
        result = _run(_h1(), (1, 1))
        self.assertTrue(result.edge_accepted)
        self.assertEqual(
            [(tier.expansion_depth, tier.status) for tier in result.tiers],
            [(0, "feasible_or_unknown"), (1, "infeasible_with_ray")],
        )
        self.assertEqual(result.certificate.contradiction, -2)

    def test_h2_requires_two_incidence_expansions(self) -> None:
        result = _run(_h2(), (1, 1))
        self.assertTrue(result.edge_accepted)
        self.assertEqual(
            [(tier.expansion_depth, tier.status) for tier in result.tiers],
            [
                (0, "feasible_or_unknown"),
                (1, "feasible_or_unknown"),
                (2, "infeasible_with_ray"),
            ],
        )
        self.assertEqual(result.certificate.contradiction, -2)

    def test_mixed_phase_conflict_is_not_reinterpreted(self) -> None:
        result = _run(_mixed(), (1, -1))
        self.assertTrue(result.edge_accepted)
        self.assertEqual(
            tuple(literal.phase for literal in result.literals), (1, -1)
        )
        self.assertEqual(result.certificate.contradiction, -2)

    def test_equality_positive_and_negative_orientations(self) -> None:
        cases = (
            (_equality_positive(), (1, 1), "equality_pos"),
            (_equality_negative(), (-1, -1), "equality_neg"),
        )
        for parent, phases, expected_kind in cases:
            with self.subTest(expected_kind=expected_kind):
                result = _run(parent, phases)
                self.assertTrue(result.edge_accepted)
                self.assertEqual(
                    result.tiers[0].ordered_global_row_ids, (0, 1)
                )
                self.assertIn(
                    expected_kind,
                    {term.kind for term in result.certificate.source_terms},
                )

    def test_full_width_free_column_prevents_false_conflict(self) -> None:
        parent = _full_width_witness()
        result = _run(parent, (1, 1))
        self.assertFalse(result.edge_accepted)
        self.assertIsNone(result.certificate)
        self.assertEqual(_continuous_range(parent, (1, 1), 0), (-1.0, -1.0))
        self.assertEqual(result.tiers[0].model_columns, 3)
        self.assertEqual(result.tiers[0].model_nonzeros, 3)

    def test_default_off_does_not_scan_or_build_the_parent(self) -> None:
        parent = _h0()
        parent_digest = sparse_hz_semantic_digest(parent)
        source_digest = pco._ordered_source_frame_digest(
            parent,
            parent_digest=parent_digest,
            deadline=time.monotonic() + 10.0,
        )
        pair = _bound_pair(parent, (1, 1))
        with mock.patch.object(
            localized,
            "_constraint_blocks",
            side_effect=AssertionError("disabled path scanned CSR"),
        ), mock.patch.object(
            localized,
            "sparse_hz_semantic_digest",
            side_effect=AssertionError("disabled path recomputed digest"),
        ), mock.patch.object(
            localized,
            "_build_incidence_frame",
            side_effect=AssertionError("disabled path built incidence"),
        ):
            result = localized.run_localized_phase_conflict_oracle_candidate(
                parent,
                pair,
                property_digest=_PROPERTY_DIGEST,
                parent_digest=parent_digest,
                source_frame_digest=source_digest,
                deadline=time.monotonic() + 10.0,
            )
        self.assertEqual(result.status, "disabled")
        self.assertFalse(result.edge_accepted)
        self.assertEqual(result.tiers, ())
        self.assertIsNone(result.terminal_parent_semantic_digest)

    def test_solver_and_replay_use_a_no_alias_private_snapshot(self) -> None:
        parent = _h0()
        original = localized._solve_tier
        observed = []

        def inspect_snapshot(snapshot, *args, **kwargs):
            self.assertIsNot(snapshot, parent)
            for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
                self.assertFalse(
                    np.shares_memory(getattr(snapshot, name), getattr(parent, name))
                )
            for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                fresh = getattr(snapshot, name)
                live = getattr(parent, name)
                for field in ("data", "indices", "indptr"):
                    self.assertFalse(
                        np.shares_memory(getattr(fresh, field), getattr(live, field))
                    )
            observed.append(snapshot)
            return original(snapshot, *args, **kwargs)

        with mock.patch.object(
            localized, "_solve_tier", side_effect=inspect_snapshot
        ):
            result = _run(parent, (1, 1))
        self.assertTrue(result.edge_accepted)
        self.assertEqual(len(observed), 1)
        self.assertGreater(result.snapshot_buffer_bytes, 0)
        self.assertGreaterEqual(result.snapshot_seconds, 0.0)

    def test_high_degree_postings_poll_deadline_and_cap_rows(self) -> None:
        count = 200_000
        block = localized._PatternBlock(
            column_indptr=np.asarray([0, count], dtype=np.int64),
            column_rows=np.arange(count, dtype=np.int64),
            n_rows=count,
            n_columns=1,
        )
        empty_cont = localized._PatternBlock(
            column_indptr=np.asarray([0, 0], dtype=np.int64),
            column_rows=np.zeros(0, dtype=np.int64),
            n_rows=0,
            n_columns=1,
        )
        empty_binary = localized._PatternBlock(
            column_indptr=np.asarray([0], dtype=np.int64),
            column_rows=np.zeros(0, dtype=np.int64),
            n_rows=0,
            n_columns=0,
        )
        frame = localized._IncidenceFrame(
            upper_continuous=block,
            upper_binary=empty_binary,
            equality_continuous=empty_cont,
            equality_binary=empty_binary,
            n_upper=count,
            n_equality=0,
            n_continuous=1,
            n_binary=0,
        )
        real_check = localized._check_deadline
        posting_polls = [0]

        def stop_inside_postings(deadline, reason):
            if reason == "deadline_during_row_postings":
                posting_polls[0] += 1
                if posting_polls[0] == 3:
                    raise TimeoutError("forced_posting_timeout")
            return real_check(deadline, reason)

        with mock.patch.object(
            localized, "_check_deadline", side_effect=stop_inside_postings
        ):
            with self.assertRaisesRegex(TimeoutError, "forced_posting_timeout"):
                localized._rows_for_columns(
                    frame, (0,), deadline=time.monotonic() + 10.0
                )
        self.assertEqual(posting_polls[0], 3)
        rows = localized._rows_for_columns(
            frame, (0,), deadline=time.monotonic() + 10.0
        )
        self.assertEqual(len(rows), 4096)
        self.assertEqual(min(rows), 0)
        self.assertEqual(max(rows), 4095)

    def test_exact_source_cap_rejects_three_row_chain(self) -> None:
        result = _run(_h1(), (1, 1), max_source_terms=2)
        self.assertFalse(result.edge_accepted)
        self.assertIsNone(result.certificate)
        self.assertEqual(result.tiers[-1].exact_replay_status, "rejected")

    def test_wrong_local_to_global_map_cannot_authorize_edge(self) -> None:
        parent = _h0()
        # Add a distractor after the two necessary source rows.
        parent.Auc = sp.vstack(
            [parent.Auc, sp.csr_matrix([[-1.0]])], format="csr"
        )
        parent.Aub = sp.vstack(
            [parent.Aub, sp.csr_matrix((1, 2))], format="csr"
        )
        parent.ub = np.concatenate([parent.ub, np.asarray([1.0])])
        original = localized._zero_pad_ray

        def wrong_map(local_ray, rows, *, full_rows):
            full = original(local_ray, rows, full_rows=full_rows)
            full[2] = full[1]
            full[1] = 0.0
            return full

        with mock.patch.object(localized, "_zero_pad_ray", side_effect=wrong_map):
            result = _run(parent, (1, 1))
        self.assertFalse(result.edge_accepted)
        self.assertIsNone(result.certificate)
        self.assertEqual(result.tiers[0].exact_replay_status, "rejected")

    def test_wrong_sign_fake_ray_is_rejected_by_exact_replay(self) -> None:
        original = localized._solve_tier

        def wrong_ray(*args, **kwargs):
            solved = original(*args, **kwargs)
            return replace(
                solved,
                status="infeasible_with_ray",
                local_ray=(1.0, -1.0),
            )

        with mock.patch.object(localized, "_solve_tier", side_effect=wrong_ray):
            result = _run(_h0(), (1, 1))
        self.assertFalse(result.edge_accepted)
        self.assertEqual(result.tiers[0].exact_replay_status, "rejected")

    def test_duplicate_and_unsorted_ray_maps_fail_closed(self) -> None:
        for rows in ((0, 0), (1, 0)):
            with self.subTest(rows=rows):
                with self.assertRaises(
                    localized.LocalizedPhaseConflictOracleError
                ):
                    localized._zero_pad_ray((1.0, 1.0), rows, full_rows=2)

    def test_live_parent_mutation_revokes_an_exact_candidate(self) -> None:
        parent = _h0()
        original = localized.exact_certificate_from_highs_dual_ray_candidate

        def mutate_after_replay(*args, **kwargs):
            certificate = original(*args, **kwargs)
            parent.ub[0] = np.nextafter(parent.ub[0], np.inf)
            return certificate

        with mock.patch.object(
            localized,
            "exact_certificate_from_highs_dual_ray_candidate",
            side_effect=mutate_after_replay,
        ):
            result = _run(parent, (1, 1))
        self.assertEqual(result.status, "parent_mutated")
        self.assertFalse(result.edge_accepted)
        self.assertFalse(result.parent_unchanged)
        self.assertIsNone(result.certificate)

    def test_aba_mutation_is_replayed_against_the_restored_parent(self) -> None:
        parent = _full_width_witness()
        solve_original = localized._solve_tier

        def solve_transient_tighter_parent(*args, **kwargs):
            parent.ub[0] = 0.0
            try:
                return solve_original(*args, **kwargs)
            finally:
                parent.ub[0] = 1.0

        with mock.patch.object(
            localized,
            "_solve_tier",
            side_effect=solve_transient_tighter_parent,
        ), mock.patch.object(
            localized,
            "exact_certificate_from_highs_dual_ray_candidate",
            side_effect=AssertionError(
                "private feasible snapshot unexpectedly produced a ray"
            ),
        ):
            result = _run(parent, (1, 1))
        self.assertFalse(result.edge_accepted)
        self.assertTrue(result.parent_unchanged)
        self.assertIsNone(result.certificate)
        self.assertEqual(result.status, "no_certified_conflict")

    def test_expired_and_internal_deadlines_never_leave_an_edge(self) -> None:
        parent = _h0()
        expired = _run(parent, (1, 1), seconds=-1.0)
        self.assertFalse(expired.edge_accepted)
        self.assertEqual(expired.status, "deadline_expired")
        with mock.patch.object(
            localized,
            "_build_incidence_frame",
            side_effect=TimeoutError("forced_selector_timeout"),
        ):
            timed = _run(_h0(), (1, 1))
        self.assertEqual(timed.status, "deadline_expired")
        self.assertFalse(timed.edge_accepted)
        self.assertEqual(timed.tiers, ())

    def test_literal_property_binding_mismatch_is_rejected(self) -> None:
        parent = _h0()
        parent_digest = sparse_hz_semantic_digest(parent)
        source_digest = pco._ordered_source_frame_digest(
            parent,
            parent_digest=parent_digest,
            deadline=time.monotonic() + 10.0,
        )
        with self.assertRaisesRegex(
            localized.LocalizedPhaseConflictOracleError,
            "literal_binding_invalid",
        ):
            localized.run_localized_phase_conflict_oracle_candidate(
                parent,
                _bound_pair(parent, (1, 1), property_digest="e" * 64),
                property_digest=_PROPERTY_DIGEST,
                parent_digest=parent_digest,
                source_frame_digest=source_digest,
                deadline=time.monotonic() + 10.0,
                enabled=True,
            )

    def test_frozen_telemetry_and_all_checksums_recompute(self) -> None:
        result = _run(_h1(), (1, 1))
        with self.assertRaises(Exception):
            result.status = "forged"
        for tier in result.tiers:
            self.assertEqual(
                tier.telemetry_sha256,
                localized._sha256(
                    localized._tier_payload(tier, include_digest=False)
                ),
            )
        self.assertEqual(
            result.telemetry_sha256,
            localized._sha256(
                {
                    "schema": "act.localized_phase_conflict.telemetry.v1",
                    "tiers": [
                        localized._tier_payload(tier, include_digest=True)
                        for tier in result.tiers
                    ],
                }
            ),
        )
        self.assertEqual(
            result.result_sha256,
            localized._sha256(
                localized._result_payload(result, include_digest=False)
            ),
        )

    def test_binary_enumeration_and_milp_match_the_pair_cut(self) -> None:
        cases = (
            (_h0(), (1, 1)),
            (_h1(), (1, 1)),
            (_h2(), (1, 1)),
            (_mixed(), (1, -1)),
        )
        assignments = tuple(itertools.product((-1, 1), repeat=2))
        for parent, forbidden in cases:
            with self.subTest(forbidden=forbidden, rows=parent.n_ub):
                result = _run(parent, forbidden)
                self.assertTrue(result.edge_accepted)
                cut = _copy_parent_with_clique_cut(parent, result.literals)
                before = {
                    assignment: _milp_feasible(parent, assignment)
                    for assignment in assignments
                }
                after = {
                    assignment: _milp_feasible(cut, assignment)
                    for assignment in assignments
                }
                self.assertFalse(before[forbidden])
                self.assertEqual(before, after)
                for assignment in assignments:
                    if not before[assignment]:
                        continue
                    for variable in range(parent.n_cont):
                        self.assertEqual(
                            _continuous_range(parent, assignment, variable),
                            _continuous_range(cut, assignment, variable),
                        )


if __name__ == "__main__":
    unittest.main()
