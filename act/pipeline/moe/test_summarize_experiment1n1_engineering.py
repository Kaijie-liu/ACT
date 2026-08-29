import unittest

from act.pipeline.moe.summarize_experiment1n1_engineering import (
    conditioned_support_counts,
    summarize,
)


class SummarizeExperiment1N1EngineeringTests(unittest.TestCase):
    def test_paired_table_and_new_results(self):
        rows = [
            {
                "sample_rank": 1,
                "baseline_status": "UNKNOWN",
                "status": "SAFE",
                "reason": "SAFE_WEIGHTED_RANGE",
                "paired_transition": "UNKNOWN->SAFE",
                "baseline_seconds": 10.0,
                "total_seconds": 5.0,
                "full_model_witness_valid": False,
                "n1": {"pairs": []},
            },
            {
                "sample_rank": 2,
                "baseline_status": "TIMEOUT",
                "status": "UNSAFE",
                "reason": "UNSAFE_FULL_FORWARD_FALLBACK",
                "paired_transition": "TIMEOUT->UNSAFE",
                "baseline_seconds": 10.0,
                "total_seconds": 15.0,
                "full_model_witness_valid": True,
                "n1": {"pairs": []},
            },
            {
                "sample_rank": 3,
                "baseline_status": "SAFE",
                "status": "SAFE",
                "reason": "SAFE_WEIGHTED_RANGE",
                "paired_transition": "SAFE->SAFE",
                "baseline_seconds": 4.0,
                "total_seconds": 4.0,
                "full_model_witness_valid": False,
                "n1": {"pairs": []},
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["baseline_solved"], 1)
        self.assertEqual(result["n1_solved"], 3)
        self.assertEqual(result["new_safe_ranks"], [1])
        self.assertEqual(result["new_full_forward_unsafe_ranks"], [2])
        self.assertEqual(
            result["paired_solved_table"]["baseline_unsolved_n1_solved_n01"], 2
        )
        self.assertTrue(result["all_unsafe_full_forward_validated"])

    def test_conditioned_support_tightening_is_counted(self):
        rows = [
            {
                "n1": {
                    "pairs": [
                        {
                            "property_rows": [
                                {
                                    "reused_parent": False,
                                    "unconditional_difference_bounds": [-2.0, 3.0],
                                    "segments": [
                                        {"difference_bounds": [-1.0, 3.0]},
                                        {"difference_bounds": [-2.0, 3.0]},
                                    ],
                                }
                            ]
                        }
                    ]
                }
            }
        ]
        counts = conditioned_support_counts(rows)
        self.assertEqual(counts["properties_with_any_strict_difference_tightening"], 1)
        self.assertEqual(counts["segments_with_strict_difference_tightening"], 1)


if __name__ == "__main__":
    unittest.main()
