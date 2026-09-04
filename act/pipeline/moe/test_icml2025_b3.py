"""Tests for the frozen official RT-ER B3 execution layer."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from act.pipeline.moe.certificate_constants import ConstantStatus
from act.pipeline.moe.icml2025_b3 import (
    PixelNormalizedExpert,
    audit_router_optimizer_state,
    formula_leaf,
    normalize_unit_pixel_box,
    route_applicability_census,
    route_status_at_epsilon,
    select_boundary_cohort,
    top1_guard,
)
from act.pipeline.moe.audit_icml2025_b3 import crown_backend_failure_counts


class BoundaryCohortTest(unittest.TestCase):
    def test_selection_is_first_eligible_index_without_cherry_picking(self) -> None:
        correct = np.asarray([True, False, True, True, True])
        upper = np.asarray([0.01, 0.001, np.inf, 0.02, 0.03])
        selected = select_boundary_cohort(
            correct, upper, samples=2, multiplier=1.05, cap=0.025
        )
        np.testing.assert_array_equal(selected, [0, 3])

    def test_selection_rejects_insufficient_frozen_cohort(self) -> None:
        with self.assertRaises(RuntimeError):
            select_boundary_cohort(
                np.asarray([True]),
                np.asarray([0.1]),
                samples=1,
                multiplier=1.05,
                cap=0.05,
            )

    def test_applicability_reports_all_and_clean_correct_denominators(self) -> None:
        census = route_applicability_census(
            np.asarray([0.02, 0.005, 0.001]),
            np.asarray([0.021, 0.006, 0.002]),
            np.asarray([True, True, False]),
            [1.0],
        )["1.0"]
        self.assertEqual(census["all_samples"], {
            "denominator": 3,
            "route_stable": 2,
            "route_unstable": 1,
            "numerically_undecided": 0,
        })
        self.assertEqual(census["clean_correct_samples"], {
            "denominator": 2,
            "route_stable": 2,
            "route_unstable": 0,
            "numerically_undecided": 0,
        })

    def test_fixed_radius_status_respects_outward_bracket(self) -> None:
        self.assertEqual(
            route_status_at_epsilon(0.02, 0.021, 0.019),
            "PROVEN_ROUTE_STABLE",
        )
        self.assertEqual(
            route_status_at_epsilon(0.02, 0.021, 0.0205),
            "UNDECIDED_NUMERICAL_ROUTE_BRACKET",
        )
        self.assertEqual(
            route_status_at_epsilon(0.02, 0.021, 0.021),
            "PROVEN_ROUTE_UNSTABLE",
        )


class TieInclusiveGuardTest(unittest.TestCase):
    def test_guard_accepts_tie_and_rejects_strictly_larger_competitor(self) -> None:
        weight = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        bias = np.zeros(3)
        matrix, rhs = top1_guard(weight, bias, expert=0)
        tie = np.asarray([1.0, 1.0])
        self.assertTrue(np.all(matrix @ tie <= rhs))
        competitor_wins = np.asarray([0.0, 1.0])
        self.assertFalse(np.all(matrix @ competitor_wins <= rhs))


class FormulaLeafTest(unittest.TestCase):
    def test_all_pre_registered_decision_outcomes(self) -> None:
        kwargs = {
            "route_lower": 0.02,
            "route_upper": 0.021,
            "smallest_registered_radius": 0.001,
        }
        self.assertEqual(
            formula_leaf(None, ConstantStatus.NOT_FORMALLY_INSTANTIATED, **kwargs),
            "L1_NOT_FORMALLY_INSTANTIATED",
        )
        self.assertEqual(
            formula_leaf(0.0005, ConstantStatus.FORMAL_BOUND, **kwargs),
            "L2_VACUOUS_AT_REGISTERED_RADII",
        )
        self.assertEqual(
            formula_leaf(0.01, ConstantStatus.FORMAL_BOUND, **kwargs),
            "L3_HARD_ROUTE_APPLICABILITY_ESTABLISHED",
        )
        self.assertEqual(
            formula_leaf(0.03, ConstantStatus.FORMAL_BOUND, **kwargs),
            "L4_ASSUMPTION_NOT_ESTABLISHED",
        )
        self.assertEqual(
            formula_leaf(0.0205, ConstantStatus.FORMAL_BOUND, **kwargs),
            "UNDECIDED_NUMERICAL_ROUTE_BRACKET_OVERLAP",
        )


class PixelNormalizationTest(unittest.TestCase):
    def test_wrapper_matches_explicit_official_normalization(self) -> None:
        expert = nn.Flatten()
        wrapper = PixelNormalizedExpert(expert)
        pixels = torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]])
        expected = (
            pixels * 255.0
            - torch.tensor([125.307, 122.961, 113.8575])[None, :, None, None]
        ) / torch.tensor([51.5865, 50.847, 51.255])[None, :, None, None]
        torch.testing.assert_close(wrapper(pixels), expected.flatten(1))

    def test_external_box_transform_is_exact_for_positive_affine_map(self) -> None:
        lower = torch.tensor(
            [[[[0.0, 0.2]], [[0.1, 0.3]], [[0.4, 0.8]]]],
            dtype=torch.float32,
        )
        upper = lower + 0.1
        center, normalized_lower, normalized_upper = normalize_unit_pixel_box(
            lower, upper
        )
        explicit_lower = PixelNormalizedExpert(nn.Identity())(lower)
        explicit_upper = PixelNormalizedExpert(nn.Identity())(upper)
        torch.testing.assert_close(normalized_lower, explicit_lower)
        torch.testing.assert_close(normalized_upper, explicit_upper)
        torch.testing.assert_close(center, (explicit_lower + explicit_upper) / 2.0)

    def test_external_box_transform_rejects_invalid_box(self) -> None:
        lower = torch.ones(1, 3, 1, 1)
        upper = torch.zeros(1, 3, 1, 1)
        with self.assertRaises(ValueError):
            normalize_unit_pixel_box(lower, upper)

    def test_config_freezes_final_seed0_endpoint(self) -> None:
        path = Path(
            "/data1/Kane/MOE/ACT/act/pipeline/moe/configs/icml2025_b3_seed0_r4.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["checkpoint"], {
            "seed": 0,
            "epoch": 130,
            "selection": "final scheduled checkpoint; no best-epoch selection",
        })
        self.assertFalse(config["numerical"]["outward_rounded_crown_safe_enabled"])
        self.assertFalse(config["monolithic"]["enabled_in_this_runner"])
        self.assertEqual(config["fixed_radius_expert_cohort"]["rows"], 100)
        self.assertEqual(config["b2_gate"]["required_issue_count"], 0)
        self.assertFalse(config["crown"]["gradient_tracking"])
        self.assertEqual(config["crown"]["method"], "CROWN")
        self.assertEqual(
            config["primary_table_epsilon_over_255"],
            [0.5, 1.0, 2.0, 4.0, 8.0],
        )
        self.assertIn(
            "small or zero absolute coverage",
            config["primary_table_interpretation"]["absolute_certification_expectation"],
        )
        identity = config["certificate_identity"]
        self.assertIn("torch_import_version", identity["required_manifest_fields"])
        self.assertIn("nvidia_driver_version", identity["required_manifest_fields"])
        self.assertIn("preprocessing_graph", identity["required_manifest_fields"])
        self.assertIn("missing or changed identity fields fail closed", identity["policy"])


class B3AuditHardeningTest(unittest.TestCase):
    def test_backend_errors_and_incomplete_bounds_fail_closed(self) -> None:
        rows = [
            {"crown": {"status": "ERROR", "complete": False}},
            {"crown": {"status": "UNKNOWN", "complete": False}},
            {"crown": {"status": "CERTIFIED_MARGIN_FILTER", "complete": True}},
        ]
        self.assertEqual(crown_backend_failure_counts(rows), (1, 2))


class RouterOptimizerProvenanceTest(unittest.TestCase):
    class ToyOfficial(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(2, 2)])
            self.router = nn.Module()
            self.router.gate = nn.Linear(2, 1)

    def test_detects_expert_only_optimizer_updates(self) -> None:
        model = self.ToyOfficial()
        reference = self.ToyOfficial()
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        model.experts[0](torch.ones(1, 2)).sum().backward()
        optimizer.step()
        result = audit_router_optimizer_state(
            model,
            {"optimizer": optimizer.state_dict()},
            reference_model=reference,
        )
        self.assertEqual(result["router_parameters_with_optimizer_state"], 0)
        self.assertEqual(result["expert_parameters_with_optimizer_state"], 2)
        self.assertTrue(result["router_equal_reference"])

    def test_detects_router_update(self) -> None:
        model = self.ToyOfficial()
        reference = self.ToyOfficial()
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        model.router.gate(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        result = audit_router_optimizer_state(
            model,
            {"optimizer": optimizer.state_dict()},
            reference_model=reference,
        )
        self.assertEqual(result["router_parameters_with_optimizer_state"], 2)
        self.assertFalse(result["router_equal_reference"])


if __name__ == "__main__":
    unittest.main()
