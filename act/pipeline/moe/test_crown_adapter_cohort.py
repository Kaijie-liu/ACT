import unittest
import typing
import importlib.util

import numpy as np
import torch
from torch import nn
from typing_extensions import override

if not hasattr(typing, "override"):
    typing.override = override  # type: ignore[attr-defined]

from act.pipeline.moe.crown_adapter_cohort import (
    TieSafeTopKSetImplication,
    _crown_bounds,
    _combine_experts,
    _safe_status,
    _summary,
)


def _constant_affine(outputs):
    layer = nn.Linear(1, len(outputs), dtype=torch.float64)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.as_tensor(outputs, dtype=torch.float64))
    return layer


class CrownAdapterCohortTests(unittest.TestCase):
    def _implication(self, scores, safety, eta=1e-7):
        return TieSafeTopKSetImplication(
            _constant_affine(scores),
            _constant_affine([safety]),
            (0, 1),
            len(scores),
            [[1.0]],
            [0.0],
            eta=eta,
        )

    def test_any_legal_topk_tie_requires_unsafe_member_property(self):
        module = self._implication([0.0, 0.0, 0.0, -1.0], -1.0)
        guard, safety, compiled = module.forward_components(
            torch.zeros((1, 1), dtype=torch.float64)
        )
        self.assertEqual(float(guard.item()), 0.0)
        self.assertEqual(float(safety.item()), -1.0)
        self.assertAlmostEqual(float(compiled.item()), -1e-7, places=14)

    def test_eta_overcheck_band_is_conservative_not_unsound(self):
        eta = 1e-7
        module = self._implication([0.0, 0.0, eta / 2.0, -1.0], -1.0, eta)
        guard, _safety, compiled = module.forward_components(
            torch.zeros((1, 1), dtype=torch.float64)
        )
        self.assertAlmostEqual(float(guard.item()), eta / 2.0, places=14)
        self.assertLess(float(compiled.item()), 0.0)

    def test_nonmember_pair_beyond_eta_may_discharge_implication(self):
        eta = 1e-7
        module = self._implication([0.0, 0.0, 2 * eta, -1.0], -1.0, eta)
        _guard, _safety, compiled = module.forward_components(
            torch.zeros((1, 1), dtype=torch.float64)
        )
        self.assertAlmostEqual(float(compiled.item()), eta, places=14)

    @unittest.skipUnless(
        importlib.util.find_spec("auto_LiRPA") is not None,
        "requires the isolated alpha-beta-CROWN environment",
    )
    def test_installed_crown_lowers_topk_tie_safe_graph(self):
        module = self._implication([0.0, 0.0, 0.0, -1.0], -1.0)
        center = torch.zeros((1, 1), dtype=torch.float64)
        result = _crown_bounds(
            module,
            center,
            center - 1.0,
            center + 1.0,
            property_rows=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tolerance=1e-7,
            method="CROWN",
        )
        self.assertIsNone(result["error"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["status"], "UNKNOWN_RELAXATION")
        self.assertLess(result["minimum_lower_bound"], 0.0)

    def test_negative_relaxation_never_becomes_unsafe(self):
        self.assertEqual(
            _safe_status([-1.0], complete=True, exact=False, tolerance=1e-7),
            "UNKNOWN_RELAXATION",
        )
        self.assertEqual(
            _safe_status([-1.0], complete=True, exact=True, tolerance=1e-7),
            "NOT_CERTIFIED_COMPLETE",
        )

    def test_pair_certificate_requires_both_member_experts(self):
        combined = _combine_experts(
            [
                {
                    "status": "CERTIFIED_MARGIN_FILTER",
                    "minimum_lower_bound": 0.5,
                    "complete": True,
                    "seconds": 1.0,
                },
                {
                    "status": "UNKNOWN_RELAXATION",
                    "minimum_lower_bound": -0.1,
                    "complete": True,
                    "seconds": 1.0,
                },
            ]
        )
        self.assertEqual(combined["status"], "UNKNOWN")

    def test_synthetic_summary_audits_four_variants_and_no_unsafe(self):
        variants = {
            name: {"status": "CERTIFIED", "seconds": 1.0}
            for name in (
                "hz_retained_guard",
                "crown_guarded_box",
                "crown_original_box",
                "crown_tie_safe_eta",
            )
        }
        row = {
            "branch_id": "rank110:pair0-3",
            "variants": variants,
            "soundness_issues": [],
            "unsafe_claimed": False,
            "error": None,
        }
        summary = _summary([row], 1)
        self.assertTrue(summary["audit_passed"])
        self.assertEqual(summary["soundness_issue_count"], 0)
        self.assertTrue(summary["no_unsafe_statuses_emitted"])
        self.assertEqual(
            summary["variant_status_counts"]["crown_guarded_box"],
            {"CERTIFIED": 1},
        )


if __name__ == "__main__":
    unittest.main()
