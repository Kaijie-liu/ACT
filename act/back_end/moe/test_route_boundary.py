# ===- act/back_end/moe/test_route_boundary.py - Route Oracle Tests --====#

import math
import unittest

import numpy as np
from scipy.optimize import linprog

from act.back_end.moe.route_boundary import (
    affine_top1_route_boundary,
    fold_affine_input_map,
)


class AffineRouteBoundaryTests(unittest.TestCase):
    def test_unconstrained_radius_matches_l1_formula(self):
        report = affine_top1_route_boundary(
            [[1.0, 1.0], [0.0, 0.0]],
            [0.0, 0.0],
            [1.0, 1.0],
        )
        self.assertEqual(report.clean_expert, 0)
        self.assertEqual(report.boundary_competitor, 1)
        self.assertAlmostEqual(report.radius, 1.0)
        self.assertLessEqual(report.radius_lower, report.radius)
        self.assertGreaterEqual(report.radius_upper, report.radius)

    def test_input_box_can_increase_the_exact_radius(self):
        report = affine_top1_route_boundary(
            [[1.0, 1.0], [0.0, 0.0]],
            [-0.2, 0.0],
            [0.1, 0.9],
            input_lower=0.0,
            input_upper=1.0,
        )
        self.assertAlmostEqual(report.radius, 0.7)
        self.assertTrue(report.box_constrained)

    def test_box_can_make_a_competitor_unreachable(self):
        report = affine_top1_route_boundary(
            [[1.0], [0.0]],
            [1.0, 0.0],
            [0.0],
            input_lower=0.0,
            input_upper=1.0,
        )
        self.assertTrue(math.isinf(report.radius))
        self.assertIsNone(report.boundary_competitor)
        self.assertFalse(report.competitors[0].reachable)

    def test_clean_tie_has_zero_any_legal_route_radius(self):
        report = affine_top1_route_boundary(
            [[1.0], [1.0]],
            [0.0, 0.0],
            [0.25],
        )
        self.assertEqual(report.clean_expert, 0)
        self.assertEqual(report.radius, 0.0)

    def test_normalization_fold_preserves_router_scores(self):
        weight = np.asarray([[2.0], [-1.0]])
        bias = np.asarray([0.5, 0.25])
        scale = np.asarray([3.0])
        shift = np.asarray([-1.0])
        folded_weight, folded_bias = fold_affine_input_map(
            weight,
            bias,
            scale,
            shift,
        )
        point = np.asarray([0.4])
        expected = weight @ (scale * point + shift) + bias
        actual = folded_weight @ point + folded_bias
        self.assertTrue(np.array_equal(actual, expected))

    def test_box_oracle_matches_independent_lp(self):
        rng = np.random.default_rng(20260829)
        for _ in range(25):
            experts, width = 4, 5
            weight = rng.normal(size=(experts, width))
            bias = rng.normal(size=experts)
            point = rng.uniform(size=width)
            report = affine_top1_route_boundary(
                weight,
                bias,
                point,
                input_lower=0.0,
                input_upper=1.0,
            )
            clean = report.clean_expert
            lp_radii = []
            for competitor in range(experts):
                if competitor == clean:
                    continue
                rows, rhs = [], []
                for coordinate in range(width):
                    upper_row = np.zeros(width + 1)
                    upper_row[coordinate], upper_row[-1] = 1.0, -1.0
                    rows.append(upper_row)
                    rhs.append(point[coordinate])
                    lower_row = np.zeros(width + 1)
                    lower_row[coordinate], lower_row[-1] = -1.0, -1.0
                    rows.append(lower_row)
                    rhs.append(-point[coordinate])
                tie_row = np.zeros(width + 1)
                tie_row[:width] = weight[clean] - weight[competitor]
                rows.append(tie_row)
                rhs.append(-(bias[clean] - bias[competitor]))
                objective = np.zeros(width + 1)
                objective[-1] = 1.0
                solved = linprog(
                    objective,
                    A_ub=np.asarray(rows),
                    b_ub=np.asarray(rhs),
                    bounds=[(0.0, 1.0)] * width + [(0.0, None)],
                    method="highs",
                )
                lp_radii.append(float(solved.fun) if solved.success else math.inf)
            self.assertAlmostEqual(report.radius, min(lp_radii), places=12)


if __name__ == "__main__":
    unittest.main()
