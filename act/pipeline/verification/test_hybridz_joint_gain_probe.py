"""Pure verdict-routing tests for the HybridZ joint-gain probe."""

from __future__ import annotations

import unittest

from act.pipeline.verification.hybridz_joint_gain_probe import (
    _discard_property_forest_live_capability,
    _probe_authority_fields,
    _safe_promotion_route,
)
from act.util.stats import VerifyStatus


class HybridZJointGainProbeRoutingTests(unittest.TestCase):
    def test_safe_validator_route_is_certified_only(self):
        self.assertEqual(
            _safe_promotion_route(
                VerifyStatus.CERTIFIED,
                property_separable_bab=True,
            ),
            ("pending_certified_validation", True),
        )
        self.assertEqual(
            _safe_promotion_route(
                VerifyStatus.FALSIFIED,
                property_separable_bab=True,
            ),
            ("not_applicable_falsified", False),
        )
        self.assertEqual(
            _safe_promotion_route(
                VerifyStatus.UNKNOWN,
                property_separable_bab=True,
            ),
            ("not_applicable_unknown", False),
        )
        self.assertEqual(
            _safe_promotion_route(
                VerifyStatus.CERTIFIED,
                property_separable_bab=False,
            ),
            ("disabled", False),
        )

    def test_authority_uses_only_the_matching_verdict_path(self):
        safe = {"proof_authority": True}
        counterexample = {
            "proof_authority": True,
            "strict_replay": {"valid_counterexample": True},
        }
        self.assertEqual(
            _probe_authority_fields(
                VerifyStatus.CERTIFIED,
                safe_proof_receipt=safe,
                counterexample_receipt=counterexample,
            ),
            {
                "proof_authority": True,
                "proof_authority_scope": (
                    "property_forest_safe_live_run"
                ),
                "safe_proof_authority": True,
                "counterexample_proof_authority": False,
            },
        )
        self.assertEqual(
            _probe_authority_fields(
                VerifyStatus.FALSIFIED,
                safe_proof_receipt=safe,
                counterexample_receipt=counterexample,
            ),
            {
                "proof_authority": True,
                "proof_authority_scope": (
                    "strict_counterexample_replay"
                ),
                "safe_proof_authority": False,
                "counterexample_proof_authority": True,
            },
        )
        self.assertEqual(
            _probe_authority_fields(
                VerifyStatus.UNKNOWN,
                safe_proof_receipt=safe,
                counterexample_receipt=counterexample,
            ),
            {
                "proof_authority": False,
                "proof_authority_scope": "diagnostic_only",
                "safe_proof_authority": False,
                "counterexample_proof_authority": False,
            },
        )
        self.assertEqual(
            _probe_authority_fields(
                VerifyStatus.CERTIFIED,
                safe_proof_receipt={"proof_authority": False},
                counterexample_receipt=counterexample,
            ),
            {
                "proof_authority": False,
                "proof_authority_scope": "diagnostic_only",
                "safe_proof_authority": False,
                "counterexample_proof_authority": False,
            },
        )
        self.assertEqual(
            _probe_authority_fields(
                VerifyStatus.FALSIFIED,
                safe_proof_receipt=safe,
                counterexample_receipt={
                    "proof_authority": True,
                    "strict_replay": {
                        "valid_counterexample": False,
                    },
                },
            ),
            {
                "proof_authority": False,
                "proof_authority_scope": "diagnostic_only",
                "safe_proof_authority": False,
                "counterexample_proof_authority": False,
            },
        )

    def test_inapplicable_safe_capability_is_removed(self):
        capability = object()
        metadata = {
            "_property_forest_live_capability": capability,
            "unrelated": 7,
        }
        self.assertTrue(
            _discard_property_forest_live_capability(metadata)
        )
        self.assertNotIn(
            "_property_forest_live_capability", metadata
        )
        self.assertEqual(metadata, {"unrelated": 7})
        self.assertFalse(
            _discard_property_forest_live_capability(metadata)
        )


if __name__ == "__main__":
    unittest.main()
