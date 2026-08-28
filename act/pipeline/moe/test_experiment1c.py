import unittest
from unittest.mock import patch

import torch

from act.back_end.moe import GateKind, OutputMoEFactoryConfig, build_output_moe
from act.pipeline.moe.audit_experiment1c import _wilson
from act.pipeline.moe.experiment1c import (
    _branch_reason,
    _inferred_row,
    exact_route_change_bracket,
)
from act.util.stats import VerifyResult, VerifyStatus


class Experiment1CTests(unittest.TestCase):
    def test_wilson_interval_contains_point_estimate(self):
        lower, upper = _wilson(2, 20)
        self.assertLess(lower, 0.1)
        self.assertGreater(upper, 0.1)

    def test_exact_route_change_bracket_is_stable_then_unstable(self):
        model = build_output_moe(
            OutputMoEFactoryConfig(
                input_shape=(1,),
                num_classes=2,
                num_experts=2,
                top_k=1,
                gate=GateKind.HARD_TOP1,
                router_hidden=(),
                expert_hidden=(),
                seed=0,
            )
        ).double()
        with torch.no_grad():
            linear = model.router[1]
            linear.weight.copy_(torch.tensor([[1.0], [0.0]], dtype=torch.float64))
            linear.bias.copy_(torch.tensor([-0.25, 0.0], dtype=torch.float64))
        bracket = exact_route_change_bracket(
            model,
            torch.zeros(1, 1, dtype=torch.float64),
            [1],
            0.5,
            steps=7,
            query_timeout=2.0,
        )
        self.assertLess(bracket["lower"], 0.25)
        self.assertGreaterEqual(bracket["upper"], 0.25)
        self.assertEqual(bracket["lower_status"], "stable")
        self.assertEqual(bracket["upper_status"], "unstable")

    def test_taxonomy_separates_gate_sufficiency(self):
        reason = _branch_reason(
            VerifyResult(VerifyStatus.FALSIFIED),
            full_witness_valid=False,
            clean_expert_prediction=1,
            clean_prediction=0,
            support={"fallback_sides": 0},
        )
        self.assertEqual(reason, "UNKNOWN_GATE_SUFFICIENCY")

    def test_large_input_route_bracket_avoids_degenerate_point_propagation(self):
        model = build_output_moe(
            OutputMoEFactoryConfig(
                input_shape=(1025,),
                num_classes=2,
                num_experts=2,
                top_k=1,
                gate=GateKind.HARD_TOP1,
                router_hidden=(),
                expert_hidden=(),
                seed=0,
            )
        ).double()
        with torch.no_grad():
            linear = model.router[1]
            linear.weight.zero_()
            linear.weight[0, 0] = 1.0
            linear.bias.copy_(torch.tensor([-0.25, 0.0], dtype=torch.float64))
        bracket = exact_route_change_bracket(
            model,
            torch.zeros(1, 1025, dtype=torch.float64),
            [1],
            0.5,
            steps=2,
            query_timeout=2.0,
        )
        self.assertEqual(
            bracket["history"][0]["certificate"],
            "strict_clean_router_margin",
        )
        self.assertEqual(bracket["lower_status"], "stable")
        self.assertEqual(bracket["upper_status"], "unstable")

    def test_unknown_midpoint_preserves_strict_bracket(self):
        model = build_output_moe(
            OutputMoEFactoryConfig(
                input_shape=(1,),
                num_classes=2,
                num_experts=2,
                top_k=1,
                gate=GateKind.HARD_TOP1,
                router_hidden=(),
                expert_hidden=(),
                seed=0,
            )
        ).double()
        with torch.no_grad():
            linear = model.router[1]
            linear.weight.copy_(torch.tensor([[1.0], [0.0]], dtype=torch.float64))
            linear.bias.copy_(torch.tensor([-0.25, 0.0], dtype=torch.float64))
        unstable = {
            "status": "unstable",
            "entering_experts": [0],
            "branch_statuses": {0: "feasible"},
            "elapsed": 0.01,
        }
        unknown = {
            "status": "unknown",
            "entering_experts": [],
            "branch_statuses": {0: "unknown"},
            "elapsed": 0.01,
        }
        with patch(
            "act.pipeline.moe.experiment1c._router_route_change",
            side_effect=[unstable, unknown, unknown],
        ):
            bracket = exact_route_change_bracket(
                model,
                torch.zeros(1, 1, dtype=torch.float64),
                [1],
                0.5,
                steps=7,
                query_timeout=2.0,
                retry_timeout=5.0,
            )
        self.assertEqual(bracket["lower"], 0.0)
        self.assertEqual(bracket["upper"], 0.5)
        self.assertFalse(bracket["bisection_complete"])
        self.assertEqual(bracket["termination"], "unknown_midpoint")

    def test_safe_monotonic_inference_points_to_source_radius(self):
        source = {
            "status": "SAFE",
            "reason": "SAFE_PROVED",
            "epsilon_multiplier": 1.10,
        }
        inferred = _inferred_row(
            source,
            epsilon=0.01,
            multiplier=1.01,
            rule="larger_radius_safe_implies_smaller_safe",
        )
        self.assertEqual(inferred["status"], "SAFE")
        self.assertEqual(inferred["monotonic_inference"]["source_multiplier"], 1.10)


if __name__ == "__main__":
    unittest.main()
