"""Tests for the cross-dataset local route-radius scaling measurement."""

from __future__ import annotations

import unittest

import numpy as np

from act.pipeline.moe.icml2025_router_dimension_law import local_affine_top1_radii


class LocalAffineRadiusTest(unittest.TestCase):
    def test_matches_direct_two_expert_formula(self) -> None:
        weights = np.asarray([[1.0, -2.0], [-1.0, 1.0]])
        scores = np.asarray([[3.0, 1.0], [0.0, 0.0]])
        radii, clean, competitors = local_affine_top1_radii(scores, weights)
        np.testing.assert_array_equal(clean, [0, 0])
        np.testing.assert_array_equal(competitors, [1, 1])
        np.testing.assert_allclose(radii, [2.0 / 5.0, 0.0])

    def test_selects_minimum_over_all_competitors(self) -> None:
        weights = np.asarray([[0.0], [2.0], [10.0]])
        scores = np.asarray([[4.0, 3.0, 0.0]])
        radii, clean, competitors = local_affine_top1_radii(scores, weights)
        self.assertEqual(int(clean[0]), 0)
        self.assertEqual(int(competitors[0]), 2)
        self.assertAlmostEqual(float(radii[0]), 0.4)

    def test_rejects_duplicate_router_rows(self) -> None:
        with self.assertRaises(ValueError):
            local_affine_top1_radii(
                np.asarray([[1.0, 0.0]]), np.asarray([[1.0], [1.0]])
            )


if __name__ == "__main__":
    unittest.main()
