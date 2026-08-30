"""Pure tests for the numerical CROWN reach worker."""

from __future__ import annotations

import math
import unittest

from act.pipeline.moe.advmoe_router_crown_reach_worker import geometric_midpoint


class NumericalReachWorkerTests(unittest.TestCase):
    def test_geometric_midpoint(self) -> None:
        self.assertAlmostEqual(geometric_midpoint(1e-12, 1e-8), 1e-10)

    def test_midpoint_is_strict(self) -> None:
        value = geometric_midpoint(1e-14, 0.5 / 255.0)
        self.assertTrue(1e-14 < value < 0.5 / 255.0)
        self.assertTrue(math.isfinite(value))

    def test_rejects_invalid_bracket(self) -> None:
        with self.assertRaises(ValueError):
            geometric_midpoint(0.0, 1.0)
        with self.assertRaises(ValueError):
            geometric_midpoint(1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
