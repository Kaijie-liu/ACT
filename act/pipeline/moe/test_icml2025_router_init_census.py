"""Tests for the official RT-ER router-initialization census."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from act.pipeline.moe.icml2025_router_init_census import summarize_seed_fractions


class SeedFractionSummaryTest(unittest.TestCase):
    def test_reports_population_distribution_without_hidden_filtering(self) -> None:
        summary = summarize_seed_fractions([0.8, 1.0, 0.9])
        self.assertAlmostEqual(summary["mean"], 0.9)
        self.assertAlmostEqual(summary["median"], 0.9)
        self.assertAlmostEqual(summary["minimum"], 0.8)
        self.assertAlmostEqual(summary["maximum"], 1.0)

    def test_rejects_empty_or_nonfinite_vectors(self) -> None:
        with self.assertRaises(ValueError):
            summarize_seed_fractions([])
        with self.assertRaises(ValueError):
            summarize_seed_fractions([float("nan")])


class FrozenRouterInitCensusConfigTest(unittest.TestCase):
    def test_freezes_full_model_initialization_and_radius_grid(self) -> None:
        path = Path(
            "/data1/Kane/MOE/ACT/act/pipeline/moe/configs/icml2025_router_init_census_r2.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "PREREGISTERED_NOT_RUN")
        self.assertEqual(config["initialization"]["seeds"], list(range(20)))
        self.assertIn("complete official MOE_Resnet18", config["initialization"]["policy"])
        self.assertTrue(config["initialization"]["seed0_bitwise_checkpoint_match_required"])
        self.assertEqual(config["seed0_reference"]["reference_radius_atol"], 1e-12)
        self.assertIn("epsilon_classifications", config["seed0_reference"]["exact_fields"])
        self.assertEqual(config["epsilon_over_255"], [0.5, 1.0, 2.0, 4.0, 8.0])
        self.assertEqual(config["reporting"]["strict_point_estimate"], "radius < epsilon")


if __name__ == "__main__":
    unittest.main()
