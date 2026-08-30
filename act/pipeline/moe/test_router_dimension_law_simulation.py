"""Tests for the preregistered dimension-law simulation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from act.pipeline.moe.router_dimension_law_simulation import (
    local_affine_top1_radii,
    run,
    summarize_rows,
)
from act.pipeline.moe.audit_router_dimension_law_simulation import audit
from act.pipeline.moe.baseline_icml2025_b1_smoke import PROJECT_ROOT


class RouterDimensionLawSimulationTests(unittest.TestCase):
    def test_local_radius_matches_two_expert_closed_form(self) -> None:
        weights = np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        scores = np.asarray([[3.0, 1.0], [0.0, 4.0]], dtype=np.float64)
        observed = local_affine_top1_radii(scores, weights)
        np.testing.assert_allclose(observed, np.asarray([0.5, 1.0]))

    def test_summary_recovers_exact_inverse_square_root_slope(self) -> None:
        config = {
            "dimensions": [100, 400, 1600],
            "router_seeds": [0, 1, 2],
            "input_families": [
                {"label": "A", "fixed_second_moment": 4.0},
                {"label": "B", "fixed_second_moment": 1.0},
            ],
            "bootstrap_replicates": 20,
            "expected_slope": -0.5,
            "preregistered_consistency_rule": {
                "absolute_point_slope_error_at_most": 0.08
            },
        }
        rows = []
        for dimension_index, dimension in enumerate(config["dimensions"]):
            for seed in config["router_seeds"]:
                for moment_index, family in enumerate(config["input_families"]):
                    rows.append(
                        {
                            "dimension": dimension,
                            "dimension_index": dimension_index,
                            "router_seed": seed,
                            "moment_index": moment_index,
                            "input_family": family["label"],
                            "fixed_second_moment": family["fixed_second_moment"],
                            "sample_index": 0,
                            "local_radius": (
                                np.sqrt(family["fixed_second_moment"])
                                / np.sqrt(dimension)
                            ),
                            "weight_pair_l1_median": np.sqrt(dimension),
                        }
                    )
        summary = summarize_rows(rows, config)
        self.assertAlmostEqual(summary["fits"]["A"]["slope"], -0.5, places=12)
        self.assertAlmostEqual(summary["fits"]["B"]["slope"], -0.5, places=12)
        self.assertAlmostEqual(
            summary["moment_scale_check"]["median_observed_radius_ratio"],
            2.0,
            places=12,
        )

    def test_tiny_runner_and_independent_audit_close(self) -> None:
        config = {
            "schema_version": 1,
            "status": "PREREGISTERED_NOT_RUN",
            "scope": "TEST",
            "dimensions": [8, 16, 32],
            "num_experts": 3,
            "router_seeds": [0, 1, 2],
            "samples_per_seed_and_moment": 2,
            "synthetic_input_seed_base": 1234,
            "input_families": [
                {"label": "A", "fixed_second_moment": 1.0},
                {"label": "B", "fixed_second_moment": 0.5},
            ],
            "bootstrap_replicates": 20,
            "expected_slope": -0.5,
            "preregistered_consistency_rule": {
                "absolute_point_slope_error_at_most": 10.0
            },
            "claim_boundary": "test only",
        }
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            config_path = root / "config.json"
            output_dir = root / "run"
            audit_path = root / "audit.json"
            config_path.write_text(
                json.dumps(config) + "\n", encoding="utf-8"
            )
            result = run(config_path, output_dir)
            checked = audit(config_path, output_dir, audit_path)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(checked["status"], "PASS")
        self.assertEqual(checked["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
