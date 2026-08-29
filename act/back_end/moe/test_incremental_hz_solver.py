# ===- test_incremental_hz_solver.py - Incremental expert tests --------====#

from dataclasses import replace
import unittest

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.moe.incremental_hz_solver import IncrementalHZBranchSolver
from act.back_end.moe.hz_routing import condition_topk_membership
from act.back_end.solver.solver_hz import (
    HZSolver,
    hz_add_output_inequalities,
    hz_minimize_output,
    hz_support_bounds,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus


class IncrementalHZBranchSolverTests(unittest.TestCase):
    def test_only_explicit_run_time_limit_warning_is_reusable(self):
        import highspy

        self.assertTrue(
            IncrementalHZBranchSolver._run_status_is_expected_time_limit(
                highspy.HighsStatus.kWarning,
                highspy.HighsModelStatus.kTimeLimit,
            )
        )
        self.assertFalse(
            IncrementalHZBranchSolver._run_status_is_expected_time_limit(
                highspy.HighsStatus.kWarning,
                highspy.HighsModelStatus.kOptimal,
            )
        )
        self.assertFalse(
            IncrementalHZBranchSolver._run_status_is_expected_time_limit(
                highspy.HighsStatus.kError,
                highspy.HighsModelStatus.kTimeLimit,
            )
        )

    def _entry(self, lower=(-1.0, -1.0), upper=(1.0, 1.0), frame=1201):
        return sparse_hz_from_bounds(
            Bounds(
                torch.tensor([lower], dtype=torch.float64),
                torch.tensor([upper], dtype=torch.float64),
            ),
            frame_id=frame,
        )

    def test_support_and_minimum_match_scipy_without_rebuilding(self):
        entry = self._entry()
        entry = hz_add_output_inequalities(
            entry, [[1.0, 1.0], [-1.0, 0.25]], [0.6, 0.8]
        )
        output = sparse_hz_linear(
            entry,
            [[1.0, -2.0], [-0.5, 1.5], [2.0, 0.25]],
            [0.1, -0.2, 0.3],
        )
        session = IncrementalHZBranchSolver(output, time_limit=10.0)
        incremental = session.support_bounds([0, 1, 2])
        reference = hz_support_bounds(
            output, [0, 1, 2], time_limit=10.0, relax_binaries=False
        )
        np.testing.assert_allclose(
            incremental.bounds.lb, reference.bounds.lb, atol=2e-8, rtol=2e-8
        )
        np.testing.assert_allclose(
            incremental.bounds.ub, reference.bounds.ub, atol=2e-8, rtol=2e-8
        )
        high_min = session.minimize_output(
            1, input_hz=entry, input_shape=(1, 2)
        )
        old_min = hz_minimize_output(
            output,
            1,
            input_hz=entry,
            input_shape=(1, 2),
            time_limit=10.0,
        )
        self.assertEqual(high_min.status, old_min.status)
        self.assertAlmostEqual(high_min.minimum, old_min.minimum, places=7)
        telemetry = session.telemetry()
        self.assertEqual(telemetry.model_builds, 1)
        self.assertEqual(telemetry.model_build_failures, 0)
        self.assertGreaterEqual(telemetry.objective_update_calls, 7)
        self.assertEqual(telemetry.solves, telemetry.objective_update_calls)

    def test_point_domain_minimum_matches_legacy_path(self):
        point = sparse_hz_from_bounds(
            Bounds(torch.tensor([[1.25]]), torch.tensor([[1.25]])),
            frame_id=1202,
        )
        session = IncrementalHZBranchSolver(point, time_limit=10.0)
        high = session.minimize_output(
            0, input_hz=point, input_shape=(1, 1)
        )
        reference = hz_minimize_output(
            point,
            0,
            input_hz=point,
            input_shape=(1, 1),
            time_limit=10.0,
        )
        self.assertEqual(high.status, reference.status)
        self.assertEqual(high.minimum, reference.minimum)
        self.assertIsNotNone(high.candidate_input)

    def test_budget_escalation_reuses_the_same_model(self):
        entry = self._entry(lower=(-1.0,), upper=(1.0,), frame=1213)
        output = sparse_hz_linear(entry, [[1.0]])
        session = IncrementalHZBranchSolver(output, time_limit=0.0)
        first = session.minimize_output(
            0, input_hz=entry, input_shape=(1, 1)
        )
        self.assertEqual(first.status, "timeout")
        session.extend_budget(5.0)
        second = session.minimize_output(
            0, input_hz=entry, input_shape=(1, 1)
        )
        self.assertEqual(second.status, "optimal")
        telemetry = session.telemetry()
        self.assertEqual(telemetry.model_builds, 1)
        self.assertEqual(telemetry.budget_extension_calls, 1)
        self.assertEqual(telemetry.budget_extension_seconds, 5.0)

    def test_property_statuses_match_scipy_and_reuse_scratch_row(self):
        entry = self._entry(lower=(-0.1,), upper=(0.1,), frame=1203)
        safe_output = sparse_hz_linear(entry, [[1.0], [0.0], [-1.0]], [2.0, 0.0, -2.0])
        spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[0])
        old = HZSolver(time_limit=10.0).evaluate_spec(
            safe_output,
            spec,
            batch_size=1,
            n_out=3,
            input_hz=entry,
            input_shape=(1, 1),
            timelimit=10.0,
        )[0]
        session = IncrementalHZBranchSolver(safe_output, time_limit=10.0)
        new = session.evaluate_spec(
            spec,
            batch_size=1,
            n_out=3,
            input_hz=entry,
            input_shape=(1, 1),
        )[0]
        self.assertEqual(new.status, old.status)
        self.assertEqual(new.status, VerifyStatus.CERTIFIED)
        telemetry = session.telemetry()
        self.assertEqual(telemetry.model_builds, 1)
        self.assertEqual(telemetry.row_pool_additions, 1)
        self.assertGreaterEqual(telemetry.row_update_calls, 2)
        self.assertGreaterEqual(telemetry.row_bound_updates, 2)

    def test_exact_violation_matches_scipy_and_relaxed_one_is_unknown(self):
        entry = self._entry(lower=(-1.0,), upper=(1.0,), frame=1205)
        output = sparse_hz_linear(entry, [[1.0], [-1.0]])
        spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[0])
        old = HZSolver(time_limit=10.0).evaluate_spec(
            output,
            spec,
            batch_size=1,
            n_out=2,
            input_hz=entry,
            input_shape=(1, 1),
            timelimit=10.0,
        )[0]
        exact = IncrementalHZBranchSolver(output, time_limit=10.0).evaluate_spec(
            spec,
            batch_size=1,
            n_out=2,
            input_hz=entry,
            input_shape=(1, 1),
        )[0]
        self.assertEqual(exact.status, old.status)
        self.assertEqual(exact.status, VerifyStatus.FALSIFIED)
        self.assertIsNotNone(exact.counterexample)
        relaxed = IncrementalHZBranchSolver(
            replace(output, exact=False), time_limit=10.0
        ).evaluate_spec(
            spec,
            batch_size=1,
            n_out=2,
            input_hz=entry,
            input_shape=(1, 1),
        )[0]
        self.assertEqual(relaxed.status, VerifyStatus.UNKNOWN)
        self.assertIsNone(relaxed.counterexample)

    def test_small_matrix_value_preserves_router_scale_coefficients(self):
        entry = self._entry()
        guarded = hz_add_output_inequalities(
            entry, [[9.7e-10, 1.0]], [0.25]
        )
        output = sparse_hz_linear(guarded, [[1.0, -1.0]])
        incremental = IncrementalHZBranchSolver(output, time_limit=10.0)
        high = incremental.support_bounds([0])
        reference = hz_support_bounds(
            output, [0], time_limit=10.0, relax_binaries=False
        )
        self.assertTrue(incremental.available)
        np.testing.assert_allclose(
            high.bounds.lb, reference.bounds.lb, atol=2e-8, rtol=2e-8
        )
        np.testing.assert_allclose(
            high.bounds.ub, reference.bounds.ub, atol=2e-8, rtol=2e-8
        )

    def test_integral_support_matches_scipy_milp(self):
        scores = sparse_hz_from_bounds(
            Bounds(
                torch.full((1, 3), -1.0, dtype=torch.float64),
                torch.full((1, 3), 1.0, dtype=torch.float64),
            ),
            frame_id=1207,
        )
        branch = condition_topk_membership(scores, expert=0, top_k=2).hz
        self.assertGreater(branch.n_bin, 0)
        session = IncrementalHZBranchSolver(
            branch, time_limit=10.0, relax_binaries=False
        )
        high = session.support_bounds([0, 1, 2])
        reference = hz_support_bounds(
            branch, [0, 1, 2], time_limit=10.0, relax_binaries=False
        )
        self.assertTrue(high.exact)
        np.testing.assert_allclose(
            high.bounds.lb, reference.bounds.lb, atol=2e-8, rtol=2e-8
        )
        np.testing.assert_allclose(
            high.bounds.ub, reference.bounds.ub, atol=2e-8, rtol=2e-8
        )
        self.assertEqual(session.telemetry().integrality_update_calls, 1)

    def test_unsafe_linear_conjunction_matches_scipy(self):
        entry = self._entry(lower=(-1.0,), upper=(1.0,), frame=1209)
        output = sparse_hz_linear(entry, [[1.0]])
        spec = OutputSpec(
            kind=OutKind.UNSAFE_LINEAR,
            c=[[1.0], [-1.0]],
            d=[-0.25, 2.0],
        )
        old = HZSolver(time_limit=10.0).evaluate_spec(
            output,
            spec,
            batch_size=1,
            n_out=1,
            input_hz=entry,
            input_shape=(1, 1),
            timelimit=10.0,
        )[0]
        session = IncrementalHZBranchSolver(output, time_limit=10.0)
        new = session.evaluate_spec(
            spec,
            batch_size=1,
            n_out=1,
            input_hz=entry,
            input_shape=(1, 1),
        )[0]
        self.assertEqual(new.status, old.status)
        self.assertEqual(new.status, VerifyStatus.FALSIFIED)
        self.assertIsNotNone(new.counterexample)
        self.assertEqual(session.telemetry().row_pool_additions, 2)

    def test_property_row_warning_invalidates_session_and_returns_unknown(self):
        entry = self._entry(lower=(-1.0,), upper=(1.0,), frame=1211)
        output = sparse_hz_linear(entry, [[1.0]])
        session = IncrementalHZBranchSolver(output, time_limit=10.0)
        self.assertTrue(session.available)
        result = session.evaluate_spec(
            OutputSpec(kind=OutKind.LINEAR_LE, c=[5e-13], d=[0.0]),
            batch_size=1,
            n_out=1,
            input_hz=entry,
            input_shape=(1, 1),
        )[0]
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertFalse(session.available)
        self.assertIn("small_matrix_value", session.telemetry().build_error)

    def test_highs_warning_fails_closed(self):
        entry = self._entry()
        guarded = hz_add_output_inequalities(entry, [[5e-13, 1.0]], [0.25])
        output = sparse_hz_linear(guarded, [[1.0, -1.0]])
        session = IncrementalHZBranchSolver(output, time_limit=10.0)
        self.assertFalse(session.available)
        support = session.support_bounds([0])
        self.assertEqual(support.lower_status, ("fast_fallback",))
        self.assertEqual(support.upper_status, ("fast_fallback",))
        spec = OutputSpec(kind=OutKind.LINEAR_LE, c=[1.0], d=[0.0])
        result = session.evaluate_spec(
            spec,
            batch_size=1,
            n_out=1,
            input_hz=entry,
            input_shape=(1, 2),
        )[0]
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(session.telemetry().model_build_failures, 1)


if __name__ == "__main__":
    unittest.main()
