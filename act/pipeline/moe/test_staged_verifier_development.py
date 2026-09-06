import json
import unittest

from act.pipeline.moe.run_staged_verifier_development import (
    DEFAULT_CONFIG,
    _summary,
    _validate_selection,
)


class StagedVerifierDevelopmentTests(unittest.TestCase):
    def test_frozen_selection_reconstructs_all_seed2_residuals(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_selection(config)
        self.assertEqual(len(config["selection"]), 13)
        self.assertEqual(
            sum(
                row["epsilon_source"] == "partial_progress"
                for row in config["selection"]
            ),
            2,
        )

    def test_summary_uses_full_outcome_selected_denominator(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        rows = [
            {
                "sample_rank": 1,
                "source_status": "UNKNOWN",
                "production_status": "SAFE",
                "production_reason": "SAFE_WEIGHTED_RANGE",
                "evidence_audit_status": "PASS",
                "evidence_audit_issue_count": 0,
            },
            {
                "sample_rank": 3,
                "source_status": "TIMEOUT",
                "production_status": "TIMEOUT",
                "production_reason": "INSTANCE_HARD_DEADLINE",
                "evidence_audit_status": None,
                "evidence_audit_issue_count": None,
            },
        ]
        summary = _summary(rows, config)
        self.assertEqual(summary["selected_rows"], 2)
        self.assertEqual(summary["new_complete_outcomes"], 1)
        self.assertTrue(summary["primary_signal_met"])
        self.assertEqual(summary["hard_timeouts"], 1)


if __name__ == "__main__":
    unittest.main()
