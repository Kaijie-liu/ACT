"""Tests for the frozen RT-ER B2 conformance stage."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from act.pipeline.moe.icml2025_b2_conformance import (
    _comparison_summary,
    endpoint_normalized_to_unit_pixels,
    unit_pixels_to_normalized,
)


CONFIG = Path(
    "/data1/Kane/MOE/ACT/act/pipeline/moe/configs/icml2025_b2_seed0_r1.json"
)


class PixelRoundTripTest(unittest.TestCase):
    def test_endpoint_normalization_round_trip(self) -> None:
        normalized = np.asarray(
            [[[[0.0]], [[1.0]], [[-1.0]]]], dtype=np.float32
        )
        pixels = endpoint_normalized_to_unit_pixels(normalized)
        observed = unit_pixels_to_normalized(torch.from_numpy(pixels)).numpy()
        np.testing.assert_allclose(observed, normalized, atol=5e-7, rtol=0.0)


class ComparisonSummaryTest(unittest.TestCase):
    def _config(self):
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_exact_fold_and_specialization_pass(self) -> None:
        scores = np.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32)
        logits = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        routes = np.asarray([0, 1], dtype=np.int64)
        selected = logits[np.arange(2), routes]
        summary = _comparison_summary(
            {
                "router_scores": scores,
                "folded_router_scores": scores.astype(np.float64),
                "expert_logits": logits,
                "wrapper_logits": logits.copy(),
                "selected_logits": selected,
                "predictions": selected.argmax(axis=1),
                "routes": routes,
            },
            self._config(),
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["nontie_route_agreement"], 1.0)

    def test_nontie_route_mismatch_fails(self) -> None:
        scores = np.asarray([[2.0, 1.0]], dtype=np.float32)
        logits = np.zeros((1, 2, 3), dtype=np.float32)
        summary = _comparison_summary(
            {
                "router_scores": scores,
                "folded_router_scores": np.asarray([[0.0, 3.0]]),
                "expert_logits": logits,
                "wrapper_logits": logits.copy(),
                "selected_logits": logits[:, 0],
                "predictions": np.asarray([0]),
                "routes": np.asarray([0]),
            },
            self._config(),
        )
        self.assertEqual(summary["status"], "FAIL")

    def test_config_freezes_first_1000_and_both_probe_families(self) -> None:
        config = self._config()
        self.assertEqual(config["status"], "PREREGISTERED_NOT_RUN")
        self.assertEqual(config["selection"]["samples"], 1000)
        self.assertEqual(
            config["selection"]["probe_families"],
            ["clean_uint8", "official_pgd50_endpoint"],
        )
        self.assertEqual(config["required_outcome"]["independent_audit"], "PASS")


if __name__ == "__main__":
    unittest.main()
