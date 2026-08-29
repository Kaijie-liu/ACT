"""Tests for the frozen dynamic-MoE verifier-front-end probe."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch
from torch import nn

from act.pipeline.moe.verifier_parser_probe import (
    VectorizedOfficialHardMoE,
    diverse_probe_slots,
)


class DiverseProbeSelectionTest(unittest.TestCase):
    def test_prefers_first_distinct_routes_then_fills_in_order(self) -> None:
        keys = [(0,), (0,), (2,), (1,), (2,)]
        self.assertEqual(diverse_probe_slots(keys, 4), [0, 2, 3, 1])

    def test_rejects_impossible_sample_count(self) -> None:
        with self.assertRaises(RuntimeError):
            diverse_probe_slots([(0,)], 2)
        with self.assertRaises(ValueError):
            diverse_probe_slots([(0,)], 0)


class _ToyRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(12, 2, bias=False)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.weight[0, 0] = 1.0
            self.gate.weight[1, 1] = 1.0


class _ToyExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        total = value.flatten(1).sum(dim=1, keepdim=True)
        return torch.cat([total * self.scale, -total * self.scale], dim=1)


class _ToyOfficialMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _ToyRouter()
        self.experts = nn.ModuleList([_ToyExpert(1.0), _ToyExpert(3.0)])


class VectorizedOfficialHardMoETest(unittest.TestCase):
    def test_matches_explicit_per_sample_hard_dispatch(self) -> None:
        official = _ToyOfficialMoE().eval()
        wrapper = VectorizedOfficialHardMoE(official).eval()
        pixels = torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.5, 0.5], [0.5, 0.5]],
                ],
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.5, 0.5], [0.5, 0.5]],
                ],
            ],
            dtype=torch.float32,
        )
        normalized = wrapper.normalized(pixels)
        scores = official.router.gate(normalized.flatten(1))
        routes = scores.argmax(dim=1)
        expected = torch.stack(
            [
                official.experts[int(route)](value[None])[0]
                for value, route in zip(normalized, routes)
            ]
        )
        torch.testing.assert_close(wrapper(pixels), expected)
        torch.testing.assert_close(wrapper.route(pixels), routes)


class FrozenParserProbeConfigTest(unittest.TestCase):
    def test_config_freezes_models_and_non_claim_scope(self) -> None:
        path = Path(
            "/data1/Kane/MOE/ACT/act/pipeline/moe/configs/verifier_parser_probe.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "PREREGISTERED_NOT_RUN")
        self.assertEqual(config["export"]["opset"], 17)
        self.assertEqual(config["export"]["probe_samples"], 4)
        self.assertIn("frontend program-consumption", config["claim_scope"])
        self.assertNotIn("verification coverage", config["claim_scope"].split(";")[0])


if __name__ == "__main__":
    unittest.main()
