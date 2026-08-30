"""Tests for empirical AdvMoE route-boundary estimates."""

from __future__ import annotations

import unittest

import numpy as np

from act.pipeline.moe.advmoe_init_boundary_estimates import (
    compute_boundary_estimates,
)


class BoundaryEstimateTests(unittest.TestCase):
    def test_two_estimators(self) -> None:
        first, pgd = compute_boundary_estimates(
            np.array([0.3, 0.6]),
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            0.02,
        )
        np.testing.assert_allclose(first, [0.3, 0.3])
        np.testing.assert_allclose(pgd, [0.2, 0.1])

    def test_rejects_invalid_compression(self) -> None:
        with self.assertRaises(ValueError):
            compute_boundary_estimates(
                np.array([0.3]), np.array([1.0]), np.array([0.0]), 0.02
            )

    def test_rejects_misalignment(self) -> None:
        with self.assertRaises(ValueError):
            compute_boundary_estimates(
                np.array([0.3, 0.2]), np.array([1.0]), np.array([0.1]), 0.02
            )


if __name__ == "__main__":
    unittest.main()
