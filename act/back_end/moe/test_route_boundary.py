# ===- act/back_end/moe/test_route_boundary.py - Route Oracle Tests --====#

import math
import unittest

import numpy as np
import torch
from scipy.optimize import linprog

from act.back_end.moe.route_boundary import (
    affine_top1_route_boundary,
    affine_top1_route_boundary_batch,
    fold_affine_input_map,
    fold_bilinear_resize_input_map,
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

    def test_upper_bracket_witness_reaches_the_reported_competitor(self):
        weight = np.asarray(
            [[1.0, 1.0, -0.5], [0.0, -0.25, 0.75], [-0.5, 0.5, 0.0]]
        )
        bias = np.asarray([-0.2, 0.1, -0.1])
        point = np.asarray([0.1, 0.9, 0.4])
        report = affine_top1_route_boundary(
            weight,
            bias,
            point,
            input_lower=0.0,
            input_upper=1.0,
        )
        self.assertIsNotNone(report.boundary_witness_delta)
        delta = np.asarray(report.boundary_witness_delta)
        witness = point + delta
        scores = weight @ witness + bias
        self.assertLessEqual(np.max(np.abs(delta)), report.radius_upper)
        self.assertTrue(np.all(witness >= 0.0))
        self.assertTrue(np.all(witness <= 1.0))
        self.assertGreaterEqual(
            scores[report.boundary_competitor] + 1e-12,
            scores[report.clean_expert],
        )

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
        self.assertIsNone(report.boundary_witness_delta)
        self.assertFalse(report.competitors[0].reachable)
        self.assertIsNone(report.competitors[0].witness_delta)

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

    def test_cifar_uint8_and_unit_pixel_domains_match_after_folding(self):
        rng = np.random.default_rng(917)
        pixels_uint8 = rng.integers(0, 256, size=(3, 4, 4), dtype=np.uint8)
        pixels_unit = pixels_uint8.astype(np.float64) / 255.0
        mean_255 = np.asarray([125.307, 122.961, 113.8575])
        std_255 = np.asarray([51.5865, 50.847, 51.255])
        feature_scale = np.repeat(255.0 / std_255, 16)
        feature_shift = np.repeat(-mean_255 / std_255, 16)
        weight = rng.normal(size=(4, 48))
        bias = rng.normal(size=4)
        folded_weight, folded_bias = fold_affine_input_map(
            weight,
            bias,
            feature_scale,
            feature_shift,
        )
        normalized = (
            pixels_uint8.astype(np.float64) - mean_255[:, None, None]
        ) / std_255[:, None, None]
        expected = weight @ normalized.reshape(-1) + bias
        actual = folded_weight @ pixels_unit.reshape(-1) + folded_bias
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_bilinear_resize_fold_matches_real_arithmetic_graph(self):
        import torch.nn.functional as functional

        generator = np.random.default_rng(17)
        source = generator.random((3, 3, 4))
        weight = generator.normal(size=(4, 3 * 7 * 6))
        bias = generator.normal(size=4)
        folded_weight, folded_bias = fold_bilinear_resize_input_map(
            weight,
            bias,
            channels=3,
            input_size=(3, 4),
            output_size=(7, 6),
            align_corners=False,
            antialias=True,
        )
        resized = functional.interpolate(
            torch.as_tensor(source[None], dtype=torch.float64),
            size=(7, 6),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).numpy().reshape(-1)
        expected = weight @ resized + bias
        actual = folded_weight @ source.reshape(-1) + folded_bias
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_normalization_then_resize_fold_preserves_composition(self):
        import torch.nn.functional as functional

        generator = np.random.default_rng(23)
        source = generator.random((3, 2, 3))
        weight = generator.normal(size=(2, 3 * 5 * 4))
        bias = generator.normal(size=2)
        scale = np.broadcast_to(
            np.asarray([1.2, 0.8, 1.5])[:, None, None], (3, 5, 4)
        ).reshape(-1)
        shift = np.broadcast_to(
            np.asarray([-0.2, 0.1, 0.3])[:, None, None], (3, 5, 4)
        ).reshape(-1)
        normalized_weight, normalized_bias = fold_affine_input_map(
            weight, bias, scale, shift
        )
        folded_weight, folded_bias = fold_bilinear_resize_input_map(
            normalized_weight,
            normalized_bias,
            channels=3,
            input_size=(2, 3),
            output_size=(5, 4),
        )
        resized = functional.interpolate(
            torch.as_tensor(source[None], dtype=torch.float64),
            size=(5, 4),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).numpy().reshape(-1)
        expected = weight @ (scale * resized + shift) + bias
        actual = folded_weight @ source.reshape(-1) + folded_bias
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_bilinear_resize_fold_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "resized output shape"):
            fold_bilinear_resize_input_map(
                np.zeros((2, 10)),
                np.zeros(2),
                channels=3,
                input_size=(2, 2),
                output_size=(2, 2),
            )

    def test_batch_oracle_matches_scalar_and_replays_witnesses(self):
        rng = np.random.default_rng(20260830)
        weight = rng.normal(size=(4, 7))
        bias = rng.normal(size=4)
        points = rng.uniform(size=(64, 7))
        batch = affine_top1_route_boundary_batch(
            weight,
            bias,
            points,
            input_lower=0.0,
            input_upper=1.0,
            include_witnesses=True,
        )
        for index, point in enumerate(points):
            scalar = affine_top1_route_boundary(
                weight,
                bias,
                point,
                input_lower=0.0,
                input_upper=1.0,
            )
            self.assertEqual(batch.clean_experts[index], scalar.clean_expert)
            self.assertEqual(
                batch.boundary_competitors[index],
                scalar.boundary_competitor,
            )
            self.assertAlmostEqual(batch.radii[index], scalar.radius, places=12)
            delta = batch.witness_deltas[index]
            witness = point + delta
            scores = weight @ witness + bias
            self.assertLessEqual(
                np.max(np.abs(delta)),
                batch.radius_uppers[index],
            )
            self.assertTrue(np.all(witness >= 0.0))
            self.assertTrue(np.all(witness <= 1.0))
            self.assertGreaterEqual(
                scores[batch.boundary_competitors[index]] + 1e-11,
                scores[batch.clean_experts[index]],
            )
        self.assertFalse(batch.radii.flags.writeable)
        self.assertFalse(batch.witness_deltas.flags.writeable)

    def test_batch_unconstrained_oracle_matches_scalar(self):
        rng = np.random.default_rng(20260831)
        weight = rng.normal(size=(3, 5))
        bias = rng.normal(size=3)
        points = rng.normal(size=(31, 5))
        batch = affine_top1_route_boundary_batch(weight, bias, points)
        expected = [
            affine_top1_route_boundary(weight, bias, point) for point in points
        ]
        np.testing.assert_allclose(
            batch.radii,
            [row.radius for row in expected],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_array_equal(
            batch.boundary_competitors,
            [row.boundary_competitor for row in expected],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_batch_breakpoints_match_numpy(self):
        rng = np.random.default_rng(20260901)
        weight = rng.normal(size=(4, 31))
        bias = rng.normal(size=4)
        points = rng.uniform(size=(127, 31))
        expected = affine_top1_route_boundary_batch(
            weight,
            bias,
            points,
            input_lower=0.0,
            input_upper=1.0,
        )
        actual = affine_top1_route_boundary_batch(
            weight,
            bias,
            points,
            input_lower=0.0,
            input_upper=1.0,
            compute_device="cuda",
        )
        np.testing.assert_allclose(
            actual.radii,
            expected.radii,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_array_equal(
            actual.boundary_competitors,
            expected.boundary_competitors,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_quantized_grid_matches_general_breakpoints(self):
        rng = np.random.default_rng(20260902)
        weight = rng.normal(size=(4, 47))
        bias = rng.normal(size=4)
        points = rng.integers(0, 256, size=(193, 47)).astype(np.float64) / 255.0
        expected = affine_top1_route_boundary_batch(
            weight,
            bias,
            points,
            input_lower=0.0,
            input_upper=1.0,
        )
        actual = affine_top1_route_boundary_batch(
            weight,
            bias,
            points,
            input_lower=0.0,
            input_upper=1.0,
            compute_device="cuda",
            capacity_grid_steps=255,
        )
        np.testing.assert_allclose(
            actual.radii,
            expected.radii,
            rtol=0.0,
            atol=1e-13,
        )
        np.testing.assert_array_equal(
            actual.boundary_competitors,
            expected.boundary_competitors,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_quantized_grid_rejects_non_grid_pixels(self):
        with self.assertRaisesRegex(ValueError, "declared finite grid"):
            affine_top1_route_boundary_batch(
                [[1.0], [0.0]],
                [0.0, 0.0],
                [[0.123456]],
                input_lower=0.0,
                input_upper=1.0,
                compute_device="cuda",
                capacity_grid_steps=255,
            )

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
