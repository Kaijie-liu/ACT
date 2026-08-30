"""Tests for AdvMoE large-epsilon and inflation diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from act.pipeline.moe.advmoe_init_large_epsilon_pgd import (
    attack_diagnostic_brackets,
    reuse_better_nested_endpoints,
)
from act.pipeline.moe.advmoe_relaxation_inflation import relaxation_inflation


class LargeEpsilonTests(unittest.TestCase):
    def test_nested_endpoint_reuse_is_atomic(self) -> None:
        result = {
            "adversarial": np.full((2, 1, 1, 1), 0.8, dtype=np.float32),
            "attacked_margin": np.array([0.2, 0.1]),
            "replay_routes": np.array([0, 1]),
            "success": np.array([False, True]),
            "linf": np.array([0.3, 0.3]),
            "margin_compression_fraction": np.array([0.5, 0.75]),
        }
        count = reuse_better_nested_endpoints(
            result,
            previous_endpoint=np.array([[[[0.6]]], [[[0.7]]]], dtype=np.float32),
            previous_margin=np.array([0.1, 0.2]),
            previous_routes=np.array([1, 0]),
            previous_success=np.array([True, False]),
            inputs=np.array([[[[0.5]]], [[[0.5]]]], dtype=np.float32),
            clean_margin=np.array([0.4, 0.4]),
        )
        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(result["adversarial"][0, 0, 0, 0]), 0.6)
        self.assertAlmostEqual(float(result["adversarial"][1, 0, 0, 0]), 0.8)
        np.testing.assert_allclose(result["attacked_margin"], [0.1, 0.1])
        np.testing.assert_array_equal(result["replay_routes"], [1, 1])
        np.testing.assert_array_equal(result["success"], [True, True])
        np.testing.assert_allclose(result["linf"], [0.1, 0.3], atol=1e-7)
        np.testing.assert_allclose(
            result["margin_compression_fraction"], [0.75, 0.75]
        )

    def test_attack_brackets(self) -> None:
        rows = attack_diagnostic_brackets(
            np.array([[False, False], [True, False], [True, False]]),
            np.array([16, 32, 64], dtype=float) / 255.0,
            8 / 255.0,
        )
        self.assertEqual(rows[0]["status"], "FLIP_FOUND")
        self.assertAlmostEqual(rows[0]["largest_tested_without_found_flip"], 16 / 255)
        self.assertAlmostEqual(rows[0]["smallest_epsilon_with_found_flip"], 32 / 255)
        self.assertEqual(rows[1]["status"], "NO_FLIP_FOUND_THROUGH_MAXIMUM")

    def test_inflation(self) -> None:
        value = relaxation_inflation(
            np.array([0.3]), np.array([-3.7e8]), np.array([0.297])
        )
        self.assertGreater(value[0], 1e11)

    def test_inflation_rejects_no_attack_drop(self) -> None:
        with self.assertRaises(ValueError):
            relaxation_inflation(
                np.array([0.3]), np.array([-1.0]), np.array([0.3])
            )


if __name__ == "__main__":
    unittest.main()
