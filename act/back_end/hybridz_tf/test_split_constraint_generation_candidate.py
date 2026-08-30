#!/usr/bin/env python3
"""Toy-first tests for the split constraint-generation LP candidate."""

from __future__ import annotations

import gc
import time
import tracemalloc
import unittest
from unittest import mock

import numpy as np
from scipy import optimize
import scipy.sparse as sp

from act.back_end.hybridz_tf import (
    split_constraint_generation_candidate as scg,
)


def _csr(values, *, shape=None) -> sp.csr_matrix:
    result = sp.csr_matrix(values, shape=shape, dtype=np.float64)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _call(
    *,
    Auc,
    Aub,
    Ac,
    Ab,
    ub,
    b,
    q,
    lower,
    upper,
    seed_upper_rows=(),
    seed_upper_duals=(),
    seed_equality_rows=(),
    seed_equality_duals=(),
    seconds=5.0,
    **caps,
):
    return scg.propose_split_constraint_generation_candidate(
        Auc=Auc,
        Aub=Aub,
        Ac=Ac,
        Ab=Ab,
        ub=np.asarray(ub, dtype=np.float64),
        b=np.asarray(b, dtype=np.float64),
        q=np.asarray(q, dtype=np.float64),
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
        seed_upper_rows=seed_upper_rows,
        seed_upper_duals=seed_upper_duals,
        seed_equality_rows=seed_equality_rows,
        seed_equality_duals=seed_equality_duals,
        deadline=time.monotonic() + seconds,
        threads=1,
        **caps,
    )


def _chain_frame():
    return {
        "Auc": _csr(
            np.asarray(
                [
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, -1.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
        "Aub": _csr((3, 0)),
        "Ac": _csr((0, 3)),
        "Ab": _csr((0, 0)),
        "ub": np.zeros(3),
        "b": np.zeros(0),
        "q": np.asarray([1.0, 0.0, 0.0]),
        "lower": -np.ones(3),
        "upper": np.ones(3),
    }


class _HighsProxy:
    def __init__(self, *, bad_run=False, bad_close=False):
        self.inner = scg._highspy.Highs()
        self.bad_run = bad_run
        self.bad_close = bad_close
        self.clear_calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def run(self):
        if self.bad_run:
            return scg._highspy.HighsStatus.kError
        return self.inner.run()

    def clear(self):
        self.clear_calls += 1
        real_status = self.inner.clear()
        if self.bad_close:
            return scg._highspy.HighsStatus.kError
        return real_status


@unittest.skipIf(scg._highspy is None, "highspy is optional")
class SplitConstraintGenerationCandidateTests(unittest.TestCase):
    def test_chain_adds_one_row_per_round_and_keeps_zero_dual_seed(self):
        frame = _chain_frame()
        result = _call(
            **frame,
            seed_upper_rows=(0,),
            seed_upper_duals=(0.0,),
            max_rounds=6,
            add_batch=1,
            max_selected_upper_rows=3,
            max_equality_rows=0,
            max_binary_change_coefficients=0,
        )
        receipt = result.receipt
        self.assertEqual(receipt["status"], "full_scan_candidate_feasible")
        self.assertEqual(receipt["selected_upper_row_order"], [0, 1, 2])
        self.assertEqual(receipt["rounds_completed"], 3)
        self.assertEqual(receipt["full_split_scan_count"], 3)
        self.assertEqual(
            [row["added_upper_rows"] for row in receipt["rounds"]],
            [[1], [2], []],
        )
        np.testing.assert_allclose(result.factor_primal, np.zeros(3))
        np.testing.assert_allclose(result.upper_row_dual, -np.ones(3))
        self.assertEqual(result.solver_minimization_objective, 0.0)
        self.assertFalse(result.factor_primal.flags.writeable)
        self.assertFalse(result.upper_row_dual.flags.writeable)
        self.assertFalse(result.equality_row_dual.flags.writeable)
        self.assertFalse(result.proof_authority)
        self.assertFalse(result.verdict_authority)
        self.assertFalse(result.primal_feasibility_authority)
        self.assertFalse(receipt["parent_binding"])
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["verdict_authority"])
        self.assertFalse(receipt["primal_feasibility_authority"])
        self.assertTrue(receipt["native_model_closed_before_return"])
        self.assertTrue(receipt["explicit_zero_dual_seed_rows_retained"])

    def test_equality_and_binary_blocks_are_split_and_binary_is_relaxed(self):
        # x=z and x+z<=1/2, with z carrying the objective.  A true binary z
        # would choose zero; the intended HybridZ LP relaxation returns 1/4.
        result = _call(
            Auc=_csr([[1.0]]),
            Aub=_csr([[1.0]]),
            Ac=_csr([[1.0]]),
            Ab=_csr([[-1.0]]),
            ub=[0.5],
            b=[0.0],
            q=[0.0, 1.0],
            lower=[-1.0, -1.0],
            upper=[1.0, 1.0],
            seed_equality_rows=(0,),
            seed_equality_duals=(0.0,),
            max_rounds=4,
            add_batch=1,
            max_selected_upper_rows=1,
            max_equality_rows=1,
            max_binary_change_coefficients=2,
        )
        self.assertEqual(
            result.receipt["status"], "full_scan_candidate_feasible"
        )
        np.testing.assert_allclose(result.factor_primal, [0.25, 0.25])
        self.assertAlmostEqual(
            result.solver_minimization_objective, -0.25, places=10
        )
        self.assertEqual(result.receipt["selected_upper_row_order"], [0])
        self.assertEqual(result.receipt["loaded_binary_nnz_at_candidate_solve"], 2)
        self.assertEqual(result.receipt["binary_change_coefficient_calls"], 2)
        self.assertIn(
            "no_integrality", result.receipt["binary_factor_semantics"]
        )
        self.assertLess(result.upper_row_dual[0], 0.0)
        self.assertTrue(np.isfinite(result.equality_row_dual[0]))

    def test_zero_upper_and_equality_only_frame(self):
        result = _call(
            Auc=_csr((0, 1)),
            Aub=_csr((0, 0)),
            Ac=_csr([[1.0]]),
            Ab=_csr((1, 0)),
            ub=[],
            b=[0.0],
            q=[1.0],
            lower=[-1.0],
            upper=[1.0],
            max_rounds=2,
            add_batch=1,
            max_selected_upper_rows=1,
            max_equality_rows=1,
            max_binary_change_coefficients=0,
        )
        self.assertEqual(result.upper_row_dual.size, 0)
        self.assertEqual(result.equality_row_dual.size, 1)
        np.testing.assert_allclose(result.factor_primal, [0.0])
        self.assertEqual(
            result.receipt["loaded_equality_rows_at_candidate_solve"], 1
        )
        self.assertEqual(result.receipt["full_split_rows_scanned"], 1)

    def test_upper_row_cap_returns_infeasible_primal_candidate_without_claim(self):
        result = _call(
            **_chain_frame(),
            seed_upper_rows=(0,),
            seed_upper_duals=(0.0,),
            max_rounds=6,
            add_batch=1,
            max_selected_upper_rows=1,
            max_equality_rows=0,
            max_binary_change_coefficients=0,
        )
        self.assertEqual(result.receipt["status"], "upper_row_cap_reached")
        self.assertEqual(result.receipt["selected_upper_row_order"], [0])
        self.assertEqual(result.receipt["rounds_completed"], 1)
        self.assertGreater(
            result.receipt["rounds"][0]["omitted_upper_violated_rows"], 0
        )
        self.assertIn(
            "candidate_only", result.receipt["primal_candidate_status"]
        )
        self.assertFalse(result.receipt["primal_feasibility_authority"])

    def test_round_cap_does_not_load_an_unsolved_row(self):
        result = _call(
            **_chain_frame(),
            seed_upper_rows=(0,),
            seed_upper_duals=(0.0,),
            max_rounds=1,
            add_batch=1,
            max_selected_upper_rows=3,
            max_equality_rows=0,
            max_binary_change_coefficients=0,
        )
        self.assertEqual(result.receipt["status"], "round_cap_reached")
        self.assertEqual(result.receipt["selected_upper_row_order"], [0])
        self.assertEqual(result.receipt["physical_rows_before_close"], 1)
        self.assertFalse(result.receipt["discarded_unsolved_model_mutation"])

    def test_equal_violation_top_batch_is_deterministic_by_source_row(self):
        result = _call(
            Auc=_csr(np.ones((3, 1))),
            Aub=_csr((3, 0)),
            Ac=_csr((0, 1)),
            Ab=_csr((0, 0)),
            ub=np.zeros(3),
            b=[],
            q=[1.0],
            lower=[-1.0],
            upper=[1.0],
            max_rounds=2,
            add_batch=2,
            max_selected_upper_rows=3,
            max_equality_rows=0,
            max_binary_change_coefficients=0,
        )
        self.assertEqual(
            result.receipt["rounds"][0]["added_upper_rows"], [0, 1]
        )
        self.assertEqual(result.receipt["selected_upper_row_order"], [0, 1])
        self.assertEqual(
            result.receipt["status"], "full_scan_candidate_feasible"
        )

    def test_deadline_after_solve_keeps_primal_but_not_feasibility_claim(self):
        frame = _chain_frame()
        real_scan = scg._scan_split_frame
        called = {"value": False}

        def expire_once(**kwargs):
            called["value"] = True
            raise scg._DeadlineExpired("deadline_expired_during_test_scan")

        with mock.patch.object(scg, "_scan_split_frame", expire_once):
            result = _call(
                **frame,
                seed_upper_rows=(0,),
                seed_upper_duals=(0.0,),
                max_rounds=2,
                add_batch=1,
                max_selected_upper_rows=3,
                max_equality_rows=0,
                max_binary_change_coefficients=0,
            )
        self.assertIsNotNone(real_scan)
        self.assertTrue(called["value"])
        self.assertEqual(
            result.receipt["status"], "deadline_exhausted_after_candidate"
        )
        self.assertEqual(result.receipt["rounds_completed"], 1)
        self.assertFalse(
            result.receipt["rounds"][0][
                "complete_split_scan_candidate_only"
            ]
        )
        self.assertEqual(result.receipt["full_split_scan_count"], 0)
        self.assertFalse(result.receipt["primal_feasibility_authority"])
        self.assertTrue(result.receipt["native_model_closed_before_return"])

    def test_expired_entry_deadline_fails_before_model_creation(self):
        frame = _chain_frame()
        with (
            mock.patch.object(scg, "_new_highs_model") as constructor,
            self.assertRaisesRegex(
                scg.SplitConstraintGenerationCandidateError,
                "deadline_expired_during_entry",
            ),
        ):
            scg.propose_split_constraint_generation_candidate(
                Auc=frame["Auc"],
                Aub=frame["Aub"],
                Ac=frame["Ac"],
                Ab=frame["Ab"],
                ub=frame["ub"],
                b=frame["b"],
                q=frame["q"],
                lower_bounds=frame["lower"],
                upper_bounds=frame["upper"],
                deadline=time.monotonic() - 1.0,
            )
        constructor.assert_not_called()

    def test_nan_and_noncanonical_inputs_fail_before_highs(self):
        frame = _chain_frame()
        bad_q = frame["q"].copy()
        bad_q[0] = np.nan
        with (
            mock.patch.object(scg, "_new_highs_model") as constructor,
            self.assertRaisesRegex(
                scg.SplitConstraintGenerationCandidateError,
                "q_contains_nonfinite_value",
            ),
        ):
            _call(
                **{**frame, "q": bad_q},
                max_equality_rows=0,
                max_binary_change_coefficients=0,
            )
        constructor.assert_not_called()

        explicit_zero = frame["Auc"].copy()
        explicit_zero.data[0] = 0.0
        with self.assertRaisesRegex(
            scg.SplitConstraintGenerationCandidateError,
            "Auc_contains_explicit_zero",
        ):
            _call(
                **{**frame, "Auc": explicit_zero},
                max_equality_rows=0,
                max_binary_change_coefficients=0,
            )

    def test_overflowing_upper_matvec_fails_closed_without_candidate(self):
        # Regression: inf/inf in the old relative-violation calculation became
        # NaN, made the comparison false, and was mislabeled feasible.
        proxy = _HighsProxy()
        with (
            mock.patch.object(scg, "_new_highs_model", return_value=proxy),
            self.assertRaisesRegex(
                scg.SplitConstraintGenerationCandidateError,
                "nonfinite_split_scan_stage:upper_continuous_matvec",
            ),
        ):
            _call(
                Auc=_csr([[1.0e308, 1.0e308]]),
                Aub=_csr((1, 0)),
                Ac=_csr((0, 2)),
                Ab=_csr((0, 0)),
                ub=[1.0e308],
                b=[],
                q=[1.0, 1.0],
                lower=[1.0, 1.0],
                upper=[1.0, 1.0],
                max_rounds=1,
                add_batch=1,
                max_selected_upper_rows=1,
                max_equality_rows=0,
                max_binary_change_coefficients=0,
            )
        self.assertEqual(proxy.clear_calls, 1)

    def test_opposite_overflow_cancellation_fails_closed(self):
        # At primal (2,2), the sparse dot executes +inf + -inf.  It must not
        # pass through maximum/flatnonzero as an omitted-row count of zero.
        with self.assertRaisesRegex(
            scg.SplitConstraintGenerationCandidateError,
            "nonfinite_split_scan_stage:upper_continuous_matvec",
        ):
            _call(
                Auc=_csr([[1.0e308, -1.0e308]]),
                Aub=_csr((1, 0)),
                Ac=_csr((0, 2)),
                Ab=_csr((0, 0)),
                ub=[-1.0],
                b=[],
                q=[1.0, 1.0],
                lower=[0.0, 0.0],
                upper=[2.0, 2.0],
                max_rounds=1,
                add_batch=1,
                max_selected_upper_rows=1,
                max_equality_rows=0,
                max_binary_change_coefficients=0,
            )

    def test_split_combination_and_residual_overflow_have_distinct_stages(self):
        common = {
            "Ac": _csr((0, 1)),
            "Ab": _csr((0, 1)),
            "b": np.zeros(0, dtype=np.float64),
            "selected_upper": set(),
            "top_cap": 1,
            "scan_chunk_rows": 8,
            "absolute_tolerance": 0.0,
            "relative_tolerance": 0.0,
            "deadline": time.monotonic() + 2.0,
        }
        with self.assertRaisesRegex(
            scg.SplitConstraintGenerationCandidateError,
            "nonfinite_split_scan_stage:upper_combined_activity",
        ):
            scg._scan_split_frame(
                Auc=_csr([[1.0e308]]),
                Aub=_csr([[1.0e308]]),
                ub=np.asarray([1.0e308]),
                primal=np.asarray([1.0, 1.0]),
                n_continuous=1,
                **common,
            )
        with self.assertRaisesRegex(
            scg.SplitConstraintGenerationCandidateError,
            "nonfinite_split_scan_stage:upper_residual",
        ):
            scg._scan_split_frame(
                Auc=_csr([[1.0e308]]),
                Aub=_csr((1, 1)),
                ub=np.asarray([-1.0e308]),
                primal=np.asarray([1.0, 0.0]),
                n_continuous=1,
                **common,
            )

    def test_finite_subnormal_and_large_cancellation_remain_valid(self):
        tiny = np.nextafter(0.0, 1.0)
        scan = scg._scan_split_frame(
            Auc=_csr(
                np.asarray(
                    [
                        [tiny, tiny],
                        [1.0e308, -1.0e308],
                    ]
                )
            ),
            Aub=_csr((2, 0)),
            Ac=_csr((0, 2)),
            Ab=_csr((0, 0)),
            ub=np.asarray([tiny + tiny, 0.0]),
            b=np.zeros(0, dtype=np.float64),
            primal=np.asarray([1.0, 1.0]),
            n_continuous=2,
            selected_upper=set(),
            top_cap=2,
            scan_chunk_rows=2,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            deadline=time.monotonic() + 2.0,
        )
        self.assertEqual(scan.omitted_violated_rows, 0)
        self.assertEqual(scan.max_upper_violation, 0.0)
        self.assertEqual(scan.rows_scanned, 2)

    def test_solver_status_failure_still_closes_model(self):
        proxy = _HighsProxy(bad_run=True)
        with (
            mock.patch.object(scg, "_new_highs_model", return_value=proxy),
            self.assertRaisesRegex(
                scg.SplitConstraintGenerationCandidateError,
                "highs_candidate_run_failed",
            ),
        ):
            _call(
                **_chain_frame(),
                seed_upper_rows=(0,),
                seed_upper_duals=(0.0,),
                max_rounds=1,
                add_batch=1,
                max_selected_upper_rows=3,
                max_equality_rows=0,
                max_binary_change_coefficients=0,
            )
        self.assertEqual(proxy.clear_calls, 1)

    def test_close_failure_prevents_candidate_return(self):
        proxy = _HighsProxy(bad_close=True)
        with (
            mock.patch.object(scg, "_new_highs_model", return_value=proxy),
            self.assertRaisesRegex(
                scg.SplitConstraintGenerationCandidateError,
                "native_model_close_failed",
            ),
        ):
            _call(
                **_chain_frame(),
                seed_upper_rows=(0,),
                seed_upper_duals=(0.0,),
                max_rounds=1,
                add_batch=1,
                max_selected_upper_rows=1,
                max_equality_rows=0,
                max_binary_change_coefficients=0,
            )
        self.assertEqual(proxy.clear_calls, 1)

    def test_no_stack_merge_or_source_csr_copy_route(self):
        frame = _chain_frame()
        forbidden = AssertionError("full-frame stack/copy is forbidden")
        with (
            mock.patch.object(sp, "hstack", side_effect=forbidden),
            mock.patch.object(sp, "vstack", side_effect=forbidden),
            mock.patch.object(np, "hstack", side_effect=forbidden),
            mock.patch.object(np, "vstack", side_effect=forbidden),
            mock.patch.object(sp.csr_matrix, "copy", side_effect=forbidden),
        ):
            result = _call(
                **frame,
                seed_upper_rows=(0,),
                seed_upper_duals=(0.0,),
                max_rounds=6,
                add_batch=1,
                max_selected_upper_rows=3,
                max_equality_rows=0,
                max_binary_change_coefficients=0,
            )
        self.assertFalse(result.receipt["uses_sparse_hstack"])
        self.assertFalse(result.receipt["uses_sparse_vstack"])
        self.assertFalse(result.receipt["used_merged_sparse_frame"])
        self.assertFalse(result.receipt["materialized_full_candidate_csr"])

    def test_binary_cap_counts_only_loaded_upper_rows(self):
        # The complete Aub has four entries, but only one row is loaded under
        # the cap.  Rejecting on full-source nnz would defeat row generation.
        result = _call(
            Auc=_csr((4, 1)),
            Aub=_csr(np.ones((4, 1))),
            Ac=_csr((0, 1)),
            Ab=_csr((0, 1)),
            ub=np.ones(4),
            b=[],
            q=[1.0, 0.0],
            lower=[-1.0, -1.0],
            upper=[1.0, 1.0],
            seed_upper_rows=(0,),
            seed_upper_duals=(0.0,),
            max_rounds=1,
            add_batch=1,
            max_selected_upper_rows=1,
            max_equality_rows=0,
            max_binary_change_coefficients=1,
        )
        self.assertEqual(result.receipt["loaded_binary_nnz_at_candidate_solve"], 1)
        self.assertEqual(result.receipt["selected_upper_row_order"], [0])

    def test_binary_change_cap_stops_before_loading_an_unsolved_row(self):
        result = _call(
            Auc=_csr((1, 0)),
            Aub=_csr([[1.0]]),
            Ac=_csr((0, 0)),
            Ab=_csr((0, 1)),
            ub=[0.0],
            b=[],
            q=[1.0],
            lower=[-1.0],
            upper=[1.0],
            max_rounds=2,
            add_batch=1,
            max_selected_upper_rows=1,
            max_equality_rows=0,
            max_binary_change_coefficients=0,
        )
        self.assertEqual(
            result.receipt["status"],
            "binary_change_coefficient_cap_reached",
        )
        self.assertEqual(result.receipt["selected_upper_row_order"], [])
        self.assertEqual(result.receipt["physical_rows_before_close"], 0)
        self.assertEqual(
            result.receipt["rounds"][0]["omitted_upper_violated_rows"],
            1,
        )

    def test_random_small_candidates_match_full_scipy_lp(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            n_cont, n_bin, n_upper = 3, 2, 14
            matrix = rng.normal(size=(n_upper, n_cont + n_bin))
            matrix[np.abs(matrix) < 0.35] = 0.0
            equality = rng.normal(size=(1, n_cont + n_bin))
            upper_rhs = rng.uniform(0.1, 1.0, size=n_upper)
            q = rng.normal(size=n_cont + n_bin)
            lower = -np.ones(n_cont + n_bin)
            upper = np.ones(n_cont + n_bin)
            scipy_result = optimize.linprog(
                -q,
                A_ub=matrix,
                b_ub=upper_rhs,
                A_eq=equality,
                b_eq=np.zeros(1),
                bounds=list(zip(lower, upper)),
                method="highs",
            )
            self.assertTrue(scipy_result.success, scipy_result.message)
            result = _call(
                Auc=_csr(matrix[:, :n_cont]),
                Aub=_csr(matrix[:, n_cont:]),
                Ac=_csr(equality[:, :n_cont]),
                Ab=_csr(equality[:, n_cont:]),
                ub=upper_rhs,
                b=np.zeros(1),
                q=q,
                lower=lower,
                upper=upper,
                max_rounds=20,
                add_batch=3,
                max_selected_upper_rows=n_upper,
                max_equality_rows=1,
                max_binary_change_coefficients=100,
            )
            self.assertEqual(
                result.receipt["status"],
                "full_scan_candidate_feasible",
            )
            self.assertAlmostEqual(
                result.solver_minimization_objective,
                float(scipy_result.fun),
                places=7,
            )
            self.assertLessEqual(
                float(np.max(matrix @ result.factor_primal - upper_rhs)),
                1.0e-7,
            )
            self.assertLessEqual(
                float(np.max(np.abs(equality @ result.factor_primal))),
                1.0e-7,
            )
            self.assertFalse(result.receipt["proof_authority"])

    def test_250k_500k_1m_empty_row_allocation_slope_is_output_linear(self):
        peaks = []
        sizes = (250_000, 500_000, 1_000_000)
        for rows in sizes:
            Auc = _csr((rows, 1))
            Aub = _csr((rows, 0))
            Ac = _csr((0, 1))
            Ab = _csr((0, 0))
            ub = np.ones(rows, dtype=np.float64)
            gc.collect()
            tracemalloc.start()
            result = _call(
                Auc=Auc,
                Aub=Aub,
                Ac=Ac,
                Ab=Ab,
                ub=ub,
                b=np.zeros(0),
                q=np.zeros(1),
                lower=-np.ones(1),
                upper=np.ones(1),
                seconds=20.0,
                max_rounds=1,
                add_batch=1,
                max_selected_upper_rows=1,
                max_equality_rows=0,
                max_binary_change_coefficients=0,
                scan_chunk_rows=8192,
            )
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak)
            self.assertEqual(result.upper_row_dual.size, rows)
            self.assertEqual(
                result.receipt["zero_padded_output_binary64_bytes"],
                (rows + 1) * 8,
            )
            self.assertLessEqual(result.receipt["maximum_scan_dense_rows"], 8192)
            self.assertEqual(
                result.receipt["maximum_materialized_upper_continuous_nnz"],
                0,
            )
            self.assertFalse(result.receipt["materialized_full_candidate_csr"])
            del result, Auc, Aub, Ac, Ab, ub
        # The only required full-row allocation is the requested zero-padded
        # dual output.  Chunk workspace and Python overhead stay bounded.
        residuals = [
            peak - rows * np.dtype(np.float64).itemsize
            for peak, rows in zip(peaks, sizes)
        ]
        self.assertLess(max(residuals) - min(residuals), 3_000_000)
        self.assertLess(
            peaks[-1] - peaks[0],
            (sizes[-1] - sizes[0]) * 12 + 3_000_000,
        )


if __name__ == "__main__":
    unittest.main()
