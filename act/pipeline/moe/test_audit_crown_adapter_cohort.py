import unittest

from act.pipeline.moe.audit_crown_adapter_cohort import summarize_rows


def _expert(hz_status, guarded, original, eta=-1.0):
    return {
        "hz_retained_guard": {
            "status": hz_status,
            "complete": hz_status != "UNKNOWN_INCOMPLETE",
            "exact": hz_status != "UNKNOWN_INCOMPLETE",
        },
        "crown_guarded_box": {
            "status": (
                "CERTIFIED_MARGIN_FILTER"
                if min(guarded) > 0
                else "UNKNOWN_RELAXATION"
            ),
            "lower_bounds": guarded,
        },
        "crown_original_box": {
            "status": (
                "CERTIFIED_MARGIN_FILTER"
                if min(original) > 0
                else "UNKNOWN_RELAXATION"
            ),
            "lower_bounds": original,
        },
        "crown_tie_safe_eta": {
            "status": (
                "CERTIFIED_MARGIN_FILTER" if eta > 0 else "UNKNOWN_RELAXATION"
            ),
            "lower_bounds": [eta],
        },
    }


class AuditCrownAdapterCohortTests(unittest.TestCase):
    def test_expert_counts_and_paired_bound_differences(self):
        experts = [
            _expert("CERTIFIED", [0.2, 0.3], [0.1, 0.3]),
            _expert("UNKNOWN_INCOMPLETE", [-0.2, 0.4], [-0.2, 0.4]),
        ]
        variants = {
            name: {"status": "UNKNOWN"}
            for name in (
                "hz_retained_guard",
                "crown_guarded_box",
                "crown_original_box",
                "crown_tie_safe_eta",
            )
        }
        summary = summarize_rows(
            [
                {
                    "branch_id": "b",
                    "experts": experts,
                    "variants": variants,
                    "error": None,
                }
            ]
        )
        self.assertEqual(summary["valid_branches"], 1)
        self.assertEqual(summary["valid_expert_obligations"], 2)
        self.assertEqual(
            summary["hz_expert_completeness"],
            {"complete_exact": 1, "incomplete": 1},
        )
        comparison = summary["guarded_vs_original_crown"]
        self.assertEqual(comparison["property_rows_compared"], 4)
        self.assertEqual(comparison["guarded_minimum_strictly_better_experts"], 1)
        self.assertEqual(comparison["guarded_minimum_equal_experts"], 1)
        self.assertEqual(comparison["guarded_minimum_strictly_worse_experts"], 0)
        self.assertFalse(
            summary["interpretation_limits"]["negative_relaxation_is_unsafe"]
        )

    def test_error_rows_are_excluded_not_silently_counted(self):
        summary = summarize_rows([{"branch_id": "bad", "error": "timeout"}])
        self.assertEqual(summary["valid_branches"], 0)
        self.assertEqual(summary["valid_expert_obligations"], 0)


if __name__ == "__main__":
    unittest.main()
