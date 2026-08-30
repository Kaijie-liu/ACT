"""Tests for the frozen RT-ER route-geometry figure accounting."""

from __future__ import annotations

import unittest

import numpy as np

from act.pipeline.moe.plot_icml2025_route_geometry import (
    strict_route_census,
    unordered_confusion_counts,
)


class StrictRouteCensusTest(unittest.TestCase):
    def test_uses_strict_less_than(self) -> None:
        radii = np.asarray([0.5 / 255.0, 2.0 / 255.0, 3.0 / 255.0])
        result = strict_route_census(radii, [2.0])["2.0"]
        self.assertEqual(result["strict_count"], 1)
        self.assertEqual(result["denominator"], 3)


class BoundaryConfusionTest(unittest.TestCase):
    def test_counts_each_unordered_pair_once(self) -> None:
        result = unordered_confusion_counts(
            np.asarray([0, 2, 1, 3]),
            np.asarray([2, 0, 3, 1]),
            experts=4,
        )
        self.assertEqual(result[0, 2], 2)
        self.assertEqual(result[1, 3], 2)
        self.assertEqual(int(result.sum()), 4)

    def test_rejects_self_competitor(self) -> None:
        with self.assertRaises(ValueError):
            unordered_confusion_counts(np.asarray([1]), np.asarray([1]), experts=2)


if __name__ == "__main__":
    unittest.main()
