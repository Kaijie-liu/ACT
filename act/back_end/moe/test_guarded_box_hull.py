from __future__ import annotations

import unittest

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.moe.guarded_box_hull import (
    guarded_hz_box_hull_highs,
    guarded_hz_box_hull_scipy,
)
from act.back_end.moe.hz_routing import condition_topk_membership
from act.back_end.solver.solver_hz import (
    hz_add_output_inequalities,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)


class TestGuardedBoxHull(unittest.TestCase):
    def _domain(self, lower, upper, A=None, b=None):
        hz = sparse_hz_from_bounds(
            Bounds(
                torch.from_numpy(np.asarray(lower, dtype=np.float64)[None, :]),
                torch.from_numpy(np.asarray(upper, dtype=np.float64)[None, :]),
            ),
            frame_id=911,
        )
        if A is not None:
            hz = hz_add_output_inequalities(
                hz,
                torch.as_tensor(A, dtype=torch.float64),
                torch.as_tensor(b, dtype=torch.float64),
            )
        return hz

    def test_one_model_reused_for_coordinate_objectives(self):
        # -x <= -0.5 intersects [-1, 1] in [0.5, 1].
        domain = self._domain([-1.0], [1.0], [[-1.0]], [-0.5])
        result = guarded_hz_box_hull_highs(domain, time_limit=5.0)
        self.assertEqual(result.domain_status, "optimal")
        self.assertTrue(result.complete)
        self.assertTrue(result.exact)
        self.assertAlmostEqual(float(result.bounds.lb.item()), 0.5, places=7)
        self.assertAlmostEqual(float(result.bounds.ub.item()), 1.0, places=7)
        telemetry = result.telemetry
        self.assertEqual(telemetry.model_builds, 1)
        self.assertEqual(telemetry.objective_update_calls, 2)
        self.assertEqual(telemetry.solves, 2)
        self.assertEqual(
            telemetry.cold_start_solves,
            telemetry.solves - telemetry.basis_submissions_accepted,
        )
        self.assertIn("basis_semantics", telemetry.as_dict())

    def test_random_guarded_domains_match_scipy_reference(self):
        rng = np.random.default_rng(20260829)
        for trial in range(8):
            n = 6
            lower = rng.uniform(-2.0, -0.2, size=n)
            upper = rng.uniform(0.2, 2.0, size=n)
            witness = rng.uniform(lower, upper)
            A = rng.normal(size=(5, n))
            slack = rng.uniform(0.0, 1.0, size=5)
            b = A @ witness + slack
            domain = self._domain(lower, upper, A, b)
            incremental = guarded_hz_box_hull_highs(domain, time_limit=10.0)
            reference = guarded_hz_box_hull_scipy(domain, time_limit=10.0)
            self.assertTrue(incremental.complete, trial)
            self.assertTrue(reference.complete, trial)
            np.testing.assert_allclose(
                incremental.bounds.lb.numpy(),
                reference.bounds.lb.numpy(),
                atol=2e-8,
                rtol=2e-8,
            )
            np.testing.assert_allclose(
                incremental.bounds.ub.numpy(),
                reference.bounds.ub.numpy(),
                atol=2e-8,
                rtol=2e-8,
            )
            self.assertTrue(np.all(incremental.bounds.lb.numpy().reshape(-1) <= witness))
            self.assertTrue(np.all(incremental.bounds.ub.numpy().reshape(-1) >= witness))

    def test_small_guard_coefficient_is_not_silently_dropped(self):
        # HiGHS defaults to ignoring matrix entries below 1e-9.  Router guard
        # rows can contain legitimate coefficients on either side of that
        # threshold, so the incremental backend must preserve this entry.
        domain = self._domain(
            [-1.0, -1.0],
            [1.0, 1.0],
            [[9.7e-10, 1.0]],
            [0.25],
        )
        incremental = guarded_hz_box_hull_highs(domain, time_limit=5.0)
        reference = guarded_hz_box_hull_scipy(domain, time_limit=5.0)
        self.assertTrue(incremental.complete)
        self.assertTrue(reference.complete)
        np.testing.assert_allclose(
            incremental.bounds.lb.numpy(), reference.bounds.lb.numpy(), atol=2e-8
        )
        np.testing.assert_allclose(
            incremental.bounds.ub.numpy(), reference.bounds.ub.numpy(), atol=2e-8
        )

    def test_affine_outputs_share_the_guarded_input_frame(self):
        entry = self._domain([-1.0, -1.0], [1.0, 1.0], [[1.0, 1.0]], [0.25])
        output = sparse_hz_linear(
            entry,
            np.asarray([[1.0, -2.0], [-3.0, 0.5]], dtype=np.float64),
            np.asarray([0.1, -0.2], dtype=np.float64),
        )
        incremental = guarded_hz_box_hull_highs(output, time_limit=5.0)
        reference = guarded_hz_box_hull_scipy(output, time_limit=5.0)
        self.assertTrue(incremental.exact)
        np.testing.assert_allclose(
            incremental.bounds.lb.numpy(), reference.bounds.lb.numpy(), atol=2e-8
        )
        np.testing.assert_allclose(
            incremental.bounds.ub.numpy(), reference.bounds.ub.numpy(), atol=2e-8
        )

    def test_binary_hz_is_reported_as_relaxed_outer_hull(self):
        scores = self._domain([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        member = condition_topk_membership(scores, expert=0, top_k=2)
        self.assertGreater(member.hz.n_bin, 0)
        incremental = guarded_hz_box_hull_highs(member.hz, time_limit=5.0)
        reference = guarded_hz_box_hull_scipy(member.hz, time_limit=5.0)
        self.assertTrue(incremental.complete)
        self.assertFalse(incremental.exact)
        self.assertEqual(incremental.relaxed_binaries, member.hz.n_bin)
        np.testing.assert_allclose(
            incremental.bounds.lb.numpy(), reference.bounds.lb.numpy(), atol=2e-8
        )
        np.testing.assert_allclose(
            incremental.bounds.ub.numpy(), reference.bounds.ub.numpy(), atol=2e-8
        )

    def test_zero_budget_keeps_sound_fast_fallback(self):
        domain = self._domain([-1.0], [1.0], [[-1.0]], [-0.5])
        result = guarded_hz_box_hull_highs(domain, time_limit=0.0)
        self.assertEqual(result.domain_status, "unknown")
        self.assertFalse(result.complete)
        self.assertEqual(float(result.bounds.lb.item()), -1.0)
        self.assertEqual(float(result.bounds.ub.item()), 1.0)
        self.assertEqual(result.telemetry.solves, 0)

    def test_infeasible_guard_is_not_reported_as_complete(self):
        domain = self._domain([-1.0], [1.0], [[1.0]], [-2.0])
        result = guarded_hz_box_hull_highs(domain, time_limit=5.0)
        self.assertEqual(result.domain_status, "infeasible")
        self.assertFalse(result.complete)
        self.assertFalse(result.exact)


if __name__ == "__main__":
    unittest.main()
