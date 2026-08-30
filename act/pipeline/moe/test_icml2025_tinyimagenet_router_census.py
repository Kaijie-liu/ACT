"""Tests for the official TinyImageNet router census."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from act.back_end.moe import affine_top1_route_boundary_batch
from act.pipeline.moe.icml2025_tinyimagenet_router_census import (
    fixed_epsilon_route_partition_torch,
)
from act.pipeline.moe.audit_icml2025_tinyimagenet_router_census import (
    _direct_fixed_partition,
)
from act.pipeline.moe.plot_icml2025_cross_dataset_router_census import (
    _tiny_fraction_matrix,
)


CONFIG = Path(
    "/data1/Kane/MOE/ACT/act/pipeline/moe/configs/"
    "icml2025_tinyimagenet_router_census_r2.json"
)


class FixedEpsilonRoutePartitionTest(unittest.TestCase):
    def test_matches_exact_radius_oracle_at_registered_endpoints(self) -> None:
        generator = np.random.default_rng(20260830)
        weight = generator.normal(size=(4, 11))
        bias = generator.normal(size=4)
        points = generator.uniform(size=(17, 11))
        epsilons = np.asarray([0.0, 0.01, 0.1, 0.4], dtype=np.float64)

        expected = affine_top1_route_boundary_batch(
            weight,
            bias,
            points,
            input_lower=0.0,
            input_upper=1.0,
            outward_absolute=1e-12,
            outward_relative=1e-12,
        )
        actual = fixed_epsilon_route_partition_torch(
            torch.as_tensor(weight, dtype=torch.float64),
            torch.as_tensor(bias, dtype=torch.float64),
            torch.as_tensor(points, dtype=torch.float64),
            torch.as_tensor(epsilons, dtype=torch.float64),
            input_lower=0.0,
            input_upper=1.0,
            outward_absolute=1e-12,
            outward_relative=1e-12,
        )

        stable = actual["formally_stable"].numpy()
        reachable = actual["formally_reachable"].numpy()
        undecided = actual["undecided"].numpy()
        np.testing.assert_array_equal(
            stable,
            epsilons[None, :] < expected.radius_lowers[:, None],
        )
        np.testing.assert_array_equal(
            reachable,
            expected.radius_uppers[:, None] <= epsilons[None, :],
        )
        np.testing.assert_array_equal(
            undecided,
            ~(
                (epsilons[None, :] < expected.radius_lowers[:, None])
                | (expected.radius_uppers[:, None] <= epsilons[None, :])
            ),
        )
        np.testing.assert_array_equal(
            actual["clean_experts"].numpy(), expected.clean_experts
        )

    def test_tie_is_never_misclassified_as_stable(self) -> None:
        weight = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        result = fixed_epsilon_route_partition_torch(
            weight,
            torch.zeros(2, dtype=torch.float64),
            torch.tensor([[0.25, 0.75]], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            input_lower=0.0,
            input_upper=1.0,
            outward_absolute=0.0,
            outward_relative=0.0,
        )
        self.assertFalse(bool(result["formally_stable"][0, 0]))
        self.assertTrue(
            bool(result["formally_reachable"][0, 0])
            or bool(result["undecided"][0, 0])
        )
        self.assertEqual(float(result["nominal_minimum_gap"][0, 0]), 0.0)
        self.assertEqual(int(result["boundary_competitors"][0, 0]), 1)

    def test_rejects_points_outside_registered_box(self) -> None:
        with self.assertRaises(ValueError):
            fixed_epsilon_route_partition_torch(
                torch.eye(2, dtype=torch.float64),
                torch.zeros(2, dtype=torch.float64),
                torch.tensor([[1.1, 0.0]], dtype=torch.float64),
                torch.tensor([0.1], dtype=torch.float64),
                input_lower=0.0,
                input_upper=1.0,
                outward_absolute=1e-9,
                outward_relative=1e-9,
            )

    def test_independent_scalar_audit_transcription_matches_runner(self) -> None:
        generator = np.random.default_rng(42)
        weight = generator.normal(size=(3, 7))
        bias = generator.normal(size=3)
        point = generator.uniform(size=7)
        epsilons = np.asarray([0.0, 0.03, 0.2], dtype=np.float64)
        runner = fixed_epsilon_route_partition_torch(
            torch.from_numpy(weight),
            torch.from_numpy(bias),
            torch.from_numpy(point[None]),
            torch.from_numpy(epsilons),
            input_lower=0.0,
            input_upper=1.0,
            outward_absolute=1e-12,
            outward_relative=1e-12,
        )
        clean, stable, reachable, undecided = _direct_fixed_partition(
            weight, bias, point, epsilons, 1e-12, 1e-12
        )
        self.assertEqual(clean, int(runner["clean_experts"][0]))
        np.testing.assert_array_equal(stable, runner["formally_stable"][0])
        np.testing.assert_array_equal(reachable, runner["formally_reachable"][0])
        np.testing.assert_array_equal(undecided, runner["undecided"][0])


class CrossDatasetFigureHelperTest(unittest.TestCase):
    def test_reduces_seed_sample_epsilon_array_only_along_samples(self) -> None:
        stable = np.asarray(
            [
                [[True, False], [False, False]],
                [[True, True], [True, False]],
            ],
            dtype=np.bool_,
        )
        actual = _tiny_fraction_matrix({"domain__stable": stable}, "domain")
        np.testing.assert_array_equal(actual, [[0.5, 0.0], [1.0, 0.5]])


class FrozenTinyImageNetCensusConfigTest(unittest.TestCase):
    def test_freezes_official_224_model_and_same_model_raw64_analysis(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "PREREGISTERED_NOT_RUN")
        self.assertEqual(config["supersedes"]["excluded_run"], "tinyimagenet_router_census_k20_20260830_r1")
        self.assertEqual(config["initialization"]["seeds"], list(range(20)))
        self.assertEqual(
            config["initialization"]["seed0_expected_router_sha256"],
            "de73e3c74832c8d41cb23ac70ea5ae697b5749ed42dd418b4719e2d544a5f79f",
        )
        self.assertEqual(config["router"]["graph"], "Flatten -> Linear(150528,4) -> argmax")
        self.assertEqual(config["router"]["num_experts"], 4)
        self.assertEqual(config["epsilon_over_255"], [0.5, 1.0, 2.0, 4.0, 8.0])
        self.assertEqual(config["preprocessing"]["resize"]["mode"], "bilinear")
        self.assertFalse(config["preprocessing"]["resize"]["align_corners"])
        self.assertTrue(config["preprocessing"]["resize"]["antialias"])
        self.assertEqual(
            config["domains"]["official_post_resize_224"]["features"],
            150528,
        )
        self.assertEqual(
            config["domains"]["official_composed_raw_64"]["features"],
            12288,
        )
        self.assertIn(
            "no new model",
            config["domains"]["official_composed_raw_64"]["description"],
        )
        self.assertFalse(config["dataset"]["labels_used"])


if __name__ == "__main__":
    unittest.main()
