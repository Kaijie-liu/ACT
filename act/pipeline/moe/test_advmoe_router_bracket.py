import unittest

import numpy as np
import torch

from act.pipeline.moe.audit_advmoe_router_bracket_pilot import validate_accounting
from act.pipeline.moe.advmoe_router_bound_worker import (
    batchnorm_deployment_identity,
    crown_bound_options,
)
from act.pipeline.moe.advmoe_router_bracket import (
    aggregate_bracket,
    clean_margin_diagnostics,
    pgd_route_flip,
    strong_pgd_route_flip,
)


class TinyRouter(torch.nn.Module):
    def forward(self, inputs):
        scalar = inputs.flatten(1).sum(dim=1) - 0.5
        return torch.stack([scalar, -scalar], dim=1)


class AdvMoeRouterBracketTests(unittest.TestCase):
    def test_pgd_witness_is_replayed_and_bounded(self):
        router = TinyRouter()
        inputs = torch.tensor([[[[0.51]]], [[[0.49]]]])
        routes = router(inputs).argmax(dim=1)
        result = pgd_route_flip(
            router,
            inputs,
            routes,
            epsilon=0.6,
            steps=5,
            restarts=1,
            step_size=0.2,
            seed=0,
        )
        self.assertTrue(result["success"].all())
        self.assertTrue((result["linf"] <= 0.6 + 1e-7).all())

    def test_strong_pgd_reports_margin_compression_and_gradients(self):
        router = TinyRouter()
        inputs = torch.tensor([[[[0.51]]], [[[0.49]]]])
        routes = router(inputs).argmax(dim=1)
        diagnostics = clean_margin_diagnostics(router, inputs, routes)
        np.testing.assert_allclose(
            diagnostics["clean_margin"], [0.02, 0.02], atol=3e-8
        )
        np.testing.assert_allclose(diagnostics["gradient_l1"], [2.0, 2.0])
        np.testing.assert_allclose(diagnostics["gradient_l2"], [2.0, 2.0])
        np.testing.assert_allclose(diagnostics["gradient_linf"], [2.0, 2.0])
        result = strong_pgd_route_flip(
            router,
            inputs,
            routes,
            epsilon=0.1,
            steps=8,
            restarts=3,
            step_divisor=4.0,
            seed=4,
        )
        self.assertTrue(result["success"].all())
        self.assertTrue((result["margin_compression_fraction"] > 1).all())
        self.assertTrue((result["linf"] <= 0.1 + 1e-7).all())
        self.assertEqual(result["schedule"]["breaks"], [4, 6])

    def test_scalable_crown_options_and_batchnorm_identity(self):
        options = crown_bound_options(
            {
                "conv_mode": "patches",
                "sparse_alpha": True,
                "sparse_intermediate": True,
                "full_conv_alpha": False,
                "crown_batch_size": 128,
                "max_crown_size": 512,
                "batched_crown_max_vram_ratio": 0.5,
                "alpha_iterations": 20,
                "alpha_lr": 0.1,
                "share_alphas": True,
            }
        )
        self.assertTrue(options["sparse_intermediate_bounds"])
        self.assertFalse(options["use_full_conv_alpha"])
        self.assertEqual(options["crown_batch_size"], 128)
        self.assertEqual(options["max_crown_size"], 512)
        module = torch.nn.Sequential(torch.nn.BatchNorm2d(3)).eval()
        identity = batchnorm_deployment_identity(module)
        self.assertEqual(identity["layers"], 1)
        self.assertEqual(identity["training_layers"], 0)
        self.assertEqual(identity["maximum_abs_running_mean"], 0.0)
        self.assertEqual(identity["maximum_abs_running_variance_minus_one"], 0.0)

    def test_aggregation_keeps_numerical_filter_nonformal(self):
        summaries, issues = aggregate_bracket(
            indices=[0, 1, 2],
            epsilons=[0.1],
            attack_rows=[{"epsilon": 0.1, "success": [True, False, False]}],
            bound_rows=[
                {
                    "epsilon": 0.1,
                    "status": "COMPLETED_NUMERICAL_FILTER",
                    "lower_bounds": [-1.0, 0.5, -0.2],
                    "error": None,
                }
            ],
            tolerance=1e-7,
            method="IBP",
        )
        self.assertEqual(issues, [])
        self.assertEqual(summaries[0]["attack_confirmed_route_unstable"], 1)
        self.assertEqual(summaries[0]["positive_numerical_bound_filter"], 1)
        self.assertEqual(summaries[0]["undecided_band"], 1)
        self.assertEqual(summaries[0]["formal_route_stable"], 0)

    def test_filter_witness_conflict_fails_audit(self):
        _summaries, issues = aggregate_bracket(
            indices=[0],
            epsilons=[0.1],
            attack_rows=[{"epsilon": 0.1, "success": [True]}],
            bound_rows=[
                {
                    "epsilon": 0.1,
                    "status": "COMPLETED_NUMERICAL_FILTER",
                    "lower_bounds": [0.1],
                    "error": None,
                }
            ],
            tolerance=1e-7,
            method="CROWN",
        )
        self.assertEqual(len(issues), 1)

    def test_independent_accounting_rejects_mutated_summary(self):
        config = {
            "sample_indices": [0, 1],
            "epsilons": [0.1],
            "numerical": {"safe_positive_margin": 1e-7},
            "bound_worker": {"method": "IBP"},
        }
        prepare = {
            "attack_rows": [
                {
                    "epsilon": 0.1,
                    "success": [True, False],
                    "replay_routes": [1, 0],
                    "linf": [0.1, 0.1],
                }
            ]
        }
        bounds = {
            "rows": [
                {
                    "epsilon": 0.1,
                    "status": "COMPLETED_NUMERICAL_FILTER",
                    "lower_bounds": [-1.0, 0.2],
                }
            ]
        }
        correct = {
            "summaries": [
                {
                    "epsilon": 0.1,
                    "samples": 2,
                    "attack_confirmed_route_unstable": 1,
                    "positive_numerical_bound_filter": 1,
                    "undecided_band": 0,
                    "conflicts": 0,
                    "formal_route_stable": 0,
                    "formal_route_stable_reason": (
                        "backend lower bounds are not outward-rounded"
                    ),
                    "bound_method": "IBP",
                }
            ]
        }
        self.assertEqual(validate_accounting(config, prepare, bounds, correct), [])
        correct["summaries"][0]["undecided_band"] = 1
        issues = validate_accounting(config, prepare, bounds, correct)
        self.assertTrue(any("undecided_band" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
