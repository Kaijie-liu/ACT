"""Tests for AdvMoE large-epsilon and inflation diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from act.pipeline.moe.advmoe_init_large_epsilon_pgd import attack_diagnostic_brackets
from act.pipeline.moe.advmoe_relaxation_inflation import relaxation_inflation


class LargeEpsilonTests(unittest.TestCase):
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
