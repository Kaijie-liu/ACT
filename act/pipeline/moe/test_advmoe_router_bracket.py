import unittest

import numpy as np
import torch

from act.pipeline.moe.audit_advmoe_router_bracket_pilot import validate_accounting
from act.pipeline.moe.advmoe_router_bracket import aggregate_bracket, pgd_route_flip


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
