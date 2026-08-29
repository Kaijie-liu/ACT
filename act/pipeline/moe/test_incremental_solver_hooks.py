import unittest

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.tf_mlp import _guarded_support_query
from act.back_end.moe.incremental_hz_solver import IncrementalHZBranchSolver
from act.back_end.moe.weighted_top2 import (
    build_weighted_top2_f0,
    shared_input_pair_hz,
)
from act.back_end.solver.solver_hz import (
    hz_add_output_inequalities,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)
from act.config.config import HybridZConfig
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.moe.experiment1 import Propagation
from act.pipeline.moe.experiment1c import _solve_staged
from act.pipeline.moe.experiment1d import _solve_f0_attempt
from act.util.stats import VerifyStatus


class _SupportTF:
    _guarded_support_solver_backend = "highspy_incremental"


class IncrementalSolverHookTests(unittest.TestCase):
    def _entry(self, frame=1401):
        return sparse_hz_from_bounds(
            Bounds(
                torch.tensor([[-1.0]], dtype=torch.float64),
                torch.tensor([[1.0]], dtype=torch.float64),
            ),
            frame_id=frame,
        )

    def test_config_defaults_remain_scipy_and_opt_in_is_validated(self):
        default = HybridZConfig()
        self.assertEqual(default.guarded_support_solver_backend, "scipy")
        self.assertEqual(default.expert_property_solver_backend, "scipy")
        opted = HybridZConfig(
            guarded_support_solver_backend="highspy_incremental",
            expert_property_solver_backend="highspy_incremental",
        )
        self.assertEqual(opted.guarded_support_solver_backend, "highspy_incremental")
        with self.assertRaises(ValueError):
            HybridZConfig(guarded_support_solver_backend="silent_auto")

    def test_guarded_support_hook_builds_one_session_for_batched_rows(self):
        entry = self._entry()
        guarded = hz_add_output_inequalities(entry, [[-1.0]], [-0.25])
        output = sparse_hz_linear(guarded, [[1.0], [-1.0]])
        support, telemetry = _guarded_support_query(
            _SupportTF(),
            output,
            [0, 1],
            time_limit=5.0,
            relax_binaries=True,
        )
        self.assertEqual(support.rows, (0, 1))
        self.assertEqual(telemetry["model_builds"], 1)
        self.assertEqual(telemetry["solves"], 4)
        self.assertEqual(telemetry["objective_update_calls"], 4)

    def test_staged_property_escalation_reuses_one_session(self):
        entry = self._entry(frame=1403)
        output = sparse_hz_linear(entry, [[1.0], [0.0]], [2.0, 0.0])
        propagation = Propagation(
            output_hz=output,
            input_hz=entry,
            output_bounds=Bounds(
                torch.tensor([[1.0, 0.0]]), torch.tensor([[3.0, 0.0]])
            ),
            unstable_per_relu=(),
            guarded_support=(),
            elapsed=0.0,
            solver_backend="highspy_incremental",
        )
        result, stages = _solve_staged(
            propagation,
            OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[0]),
            input_shape=(1, 1),
            low_budget=0.0,
            escalation_budget=5.0,
        )
        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(len(stages), 2)
        telemetry = stages[-1]["metadata"]["incremental_hz"]
        self.assertEqual(telemetry["model_builds"], 1)
        self.assertEqual(telemetry["budget_extension_calls"], 1)

    def test_f0_incremental_hook_preserves_encoding_and_decision(self):
        entry = self._entry(frame=1405)
        expert_a = sparse_hz_linear(entry, [[1.0]], [2.0])
        expert_b = sparse_hz_linear(entry, [[1.0]], [2.0])
        pair = shared_input_pair_hz(entry, expert_a, expert_b)
        router = sparse_hz_linear(entry, [[0.0], [0.0]])
        encoding = build_weighted_top2_f0(
            pair,
            router,
            (0, 1),
            [1.0],
            0.0,
            margin_time_limit=2.0,
            difference_time_limit=2.0,
        )
        scipy = _solve_f0_attempt(
            encoding,
            input_shape=(1, 1),
            time_limit=5.0,
            tolerance=1e-7,
            backend="scipy",
        )
        session = IncrementalHZBranchSolver(
            encoding.output_hz,
            time_limit=5.0,
            relax_binaries=False,
        )
        incremental = _solve_f0_attempt(
            encoding,
            input_shape=(1, 1),
            time_limit=5.0,
            tolerance=1e-7,
            backend="highspy_incremental",
            incremental_session=session,
        )
        self.assertEqual(incremental.status, scipy.status)
        self.assertEqual(incremental.reason, scipy.reason)
        self.assertAlmostEqual(incremental.minimum, scipy.minimum, places=7)
        self.assertEqual(session.telemetry().model_builds, 1)


if __name__ == "__main__":
    unittest.main()
