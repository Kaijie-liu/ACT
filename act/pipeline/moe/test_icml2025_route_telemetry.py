import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from act.back_end.moe import affine_top1_route_boundary_batch
from act.pipeline.moe.icml2025_route_telemetry import (
    _validate_probe_witnesses,
    _grouped_official_forward,
    fold_official_router,
    summarize_boundaries,
)


class ICML2025RouteTelemetryTests(unittest.TestCase):
    def test_official_normalization_fold_preserves_scores(self):
        rng = np.random.default_rng(7)
        weight = rng.normal(size=(4, 3072))
        bias = rng.normal(size=4)
        pixels = rng.integers(0, 256, size=(3, 32, 32)).astype(np.float64) / 255.0
        folded_weight, folded_bias = fold_official_router(weight, bias)
        normalized = (
            pixels * 255.0
            - np.asarray([125.307, 122.961, 113.8575])[:, None, None]
        ) / np.asarray([51.5865, 50.847, 51.255])[:, None, None]
        expected = weight @ normalized.reshape(-1) + bias
        actual = folded_weight @ pixels.reshape(-1) + folded_bias
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_summary_separates_stable_reachable_and_undecided(self):
        summary = summarize_boundaries(
            np.asarray([0, 0, 1]),
            np.asarray([1, 1, 0]),
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([0.09, 0.19, 0.29]),
            np.asarray([0.11, 0.21, 0.31]),
            num_experts=2,
            epsilon_over_255=[25.5],
        )
        counts = summary["epsilon_counts"]["25.5"]
        self.assertEqual(counts["proven_reachable"], 0)
        self.assertEqual(counts["numerically_undecided"], 1)
        self.assertEqual(counts["proven_stable"], 2)
        self.assertEqual(summary["route_load_counts"], [2, 1])
        self.assertEqual(summary["directed_boundary_competitor_counts"]["0->1"], 2)

    def test_probe_witness_is_replayed_through_folded_router(self):
        weight = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
        bias = np.asarray([0.0, 1.0])
        points = np.asarray([[0.75, 0.5]])
        result = affine_top1_route_boundary_batch(
            weight,
            bias,
            points,
            input_lower=0.0,
            input_upper=1.0,
            include_witnesses=True,
        )
        audit = _validate_probe_witnesses(
            points, weight, bias, result, tolerance=1e-7
        )
        self.assertEqual(audit["checked"], 1)
        self.assertEqual(audit["failures"], 0)

    def test_grouped_forward_matches_official_per_sample_dispatch(self):
        router = SimpleNamespace(gate=nn.Linear(2, 2, bias=False))
        experts = nn.ModuleList(
            [nn.Sequential(nn.Flatten(), nn.Linear(2, 10, bias=False)) for _ in range(2)]
        )
        with torch.no_grad():
            router.gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            experts[0][1].weight.fill_(1.0)
            experts[1][1].weight.fill_(2.0)
        model = SimpleNamespace(router=router, experts=experts)
        inputs = torch.tensor([[[[2.0, 1.0]]], [[[1.0, 3.0]]]])
        grouped, scores = _grouped_official_forward(model, inputs)
        routes = scores.argmax(dim=1)
        expected = torch.stack(
            [experts[int(routes[index])](inputs[index : index + 1])[0] for index in range(2)]
        )
        torch.testing.assert_close(grouped, expected)


if __name__ == "__main__":
    unittest.main()
