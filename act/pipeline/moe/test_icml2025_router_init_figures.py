"""Tests for the audited RT-ER router-init lottery figure."""

from __future__ import annotations

import unittest

import numpy as np

from act.pipeline.moe.plot_icml2025_router_init_census import (
    formal_stable_fraction_matrix,
    maximum_route_boundary,
)


class FormalStableFractionTest(unittest.TestCase):
    def test_uses_strict_epsilon_below_lower_bracket(self) -> None:
        lower = np.asarray([[0.5 / 255.0, 2.0 / 255.0], [3.0 / 255.0, 4.0 / 255.0]])
        result = formal_stable_fraction_matrix(lower, [0.5, 2.0])
        np.testing.assert_allclose(result, [[0.5, 0.0], [1.0, 1.0]])

    def test_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            formal_stable_fraction_matrix(np.asarray([]), [1.0])


class MaximumRouteBoundaryTest(unittest.TestCase):
    def test_reports_seed_and_sample_identity(self) -> None:
        result = maximum_route_boundary(
            np.asarray([[0.1, 0.2], [0.4, 0.3]]),
            np.asarray([7, 9]),
        )
        self.assertEqual(result["seed"], 9)
        self.assertEqual(result["sample_index"], 0)
        self.assertAlmostEqual(result["radius"], 0.4)


if __name__ == "__main__":
    unittest.main()
