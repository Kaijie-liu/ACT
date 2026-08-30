"""Tests for the frozen B1 landing sequence."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from act.pipeline.moe.baseline_icml2025_b1_endpoint import (
    _normalized_batch,
    _pixel_linf,
    _pixel_range,
)
from act.pipeline.moe.baseline_icml2025_b1_landing import (
    endpoint_decisions,
    validate_completed_epoch,
)
from act.pipeline.moe.baseline_icml2025_b1_smoke import _sha256


class B1LandingTests(unittest.TestCase):
    def test_endpoint_normalization_and_pixel_norm_use_official_255_domain(self) -> None:
        raw = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        raw[..., 0] = 125
        raw[..., 1] = 123
        raw[..., 2] = 114
        normalized = _normalized_batch(raw, torch.device("cpu"))
        self.assertEqual(normalized.dtype, torch.float16)
        expected = (
            torch.tensor([125.0, 123.0, 114.0], dtype=torch.float16)
            - torch.tensor([125.307, 122.961, 113.8575], dtype=torch.float16)
        ) / torch.tensor([51.5865, 50.847, 51.255], dtype=torch.float16)
        self.assertTrue(torch.equal(normalized[0, :, 0, 0], expected))
        adversarial = normalized.clone()
        adversarial[:, 0] += torch.tensor(8.0 / 51.5865, dtype=torch.float16)
        observed = _pixel_linf(normalized, adversarial)
        self.assertAlmostEqual(float(observed.item()), 8.0 / 255.0, places=4)
        minimum, maximum = _pixel_range(adversarial)
        self.assertGreaterEqual(float(minimum.item()), 0.0)
        self.assertLessEqual(float(maximum.item()), 255.0)

    def test_endpoint_decisions_use_inclusive_frozen_intervals(self) -> None:
        interpretation = {
            "primary_standard_accuracy_rule": {
                "inclusive_interval_percent": [72.81, 82.81],
                "inside": "SA_IN",
                "outside": "SA_OUT",
            },
            "secondary_pgd50_rule": {
                "inclusive_interval_percent": [64.09, 74.09],
                "inside": "RA_IN",
                "outside": "RA_OUT",
            },
        }
        self.assertEqual(endpoint_decisions(72.81, 74.09, interpretation)["standard_accuracy_branch"], "SA_IN")
        self.assertEqual(endpoint_decisions(72.81, 74.09, interpretation)["pgd50_accuracy_branch"], "RA_IN")
        self.assertEqual(endpoint_decisions(50.0, 20.0, interpretation)["standard_accuracy_branch"], "SA_OUT")

    def test_completed_epoch_validates_all_hashes_and_telemetry_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            checkpoint = root / "epoch_050.t7"
            metrics = root / "epoch_050.json"
            telemetry = root / "telemetry"
            telemetry.mkdir()
            torch.save({"epoch": 49}, checkpoint)
            metrics.write_text('{"epoch": 50}\n', encoding="utf-8")
            telemetry_summary = telemetry / "summary.json"
            telemetry_summary.write_text(
                json.dumps(
                    {
                        "epoch": 50,
                        "checkpoint": {"sha256": _sha256(checkpoint)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            progress = {
                "completed": [
                    {
                        "epoch": 50,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "metrics": str(metrics),
                        "metrics_sha256": _sha256(metrics),
                        "telemetry": str(telemetry),
                        "telemetry_summary_sha256": _sha256(telemetry_summary),
                    }
                ]
            }
            row = validate_completed_epoch(progress, 50)
            self.assertTrue(row["telemetry_checkpoint_identity_passed"])


if __name__ == "__main__":
    unittest.main()
