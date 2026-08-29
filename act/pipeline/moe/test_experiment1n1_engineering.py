import unittest

from act.pipeline.moe.experiment1n1_engineering import _summary


class Experiment1N1EngineeringTests(unittest.TestCase):
    def test_summary_is_paired_and_preserves_confirmatory_endpoint(self):
        rows = [
            {
                "status": "SAFE",
                "reason": "SAFE_WEIGHTED_SEGMENTED",
                "baseline_status": "UNKNOWN",
                "baseline_seconds": 10.0,
                "total_seconds": 8.0,
                "paired_transition": "UNKNOWN->SAFE",
                "segmented_property_rows": 2,
                "active_segments": 4,
                "full_model_witness_valid": False,
            },
            {
                "status": "UNSAFE",
                "reason": "UNSAFE_FULL_FORWARD_FALLBACK",
                "baseline_status": "UNSAFE",
                "baseline_seconds": 5.0,
                "total_seconds": 7.0,
                "paired_transition": "UNSAFE->UNSAFE",
                "segmented_property_rows": 1,
                "active_segments": 2,
                "full_model_witness_valid": True,
            },
        ]
        summary = _summary(rows)
        self.assertEqual(summary["baseline_solved_rows"], 1)
        self.assertEqual(summary["n1_solved_rows"], 2)
        self.assertEqual(summary["net_solved_change"], 1)
        self.assertEqual(summary["segmented_property_rows"], 3)
        self.assertEqual(summary["active_segments"], 6)
        self.assertTrue(summary["all_unsafe_full_forward_validated"])
        self.assertEqual(summary["semantic_conflict_sample_ranks"], [])
        self.assertEqual(summary["audit_issues_pre_independent_audit"], 0)
        self.assertEqual(
            summary["original_confirmatory_overall_solved_rate_immutable"],
            0.56,
        )

    def test_unreplayed_unsafe_fails_audit_flag(self):
        summary = _summary(
            [
                {
                    "status": "UNSAFE",
                    "reason": "UNSAFE_FULL_FORWARD_FALLBACK",
                    "baseline_status": "UNKNOWN",
                    "baseline_seconds": 1.0,
                    "total_seconds": 1.0,
                    "paired_transition": "UNKNOWN->UNSAFE",
                    "sample_rank": 110,
                    "full_model_witness_valid": False,
                }
            ]
        )
        self.assertFalse(summary["all_unsafe_full_forward_validated"])
        self.assertEqual(summary["unreplayed_unsafe_sample_ranks"], [110])
        self.assertEqual(summary["audit_issues_pre_independent_audit"], 1)


if __name__ == "__main__":
    unittest.main()
