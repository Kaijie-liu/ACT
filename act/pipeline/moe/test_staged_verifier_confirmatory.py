"""Tests for the fixed-radius production staged-verifier cohort."""

from __future__ import annotations

import json
import unittest

from act.pipeline.moe.freeze_staged_verifier_confirmatory import (
    OUTPUT,
    select_clean_correct,
)
from act.pipeline.moe.run_staged_verifier_confirmatory import (
    DEFAULT_CONFIG,
    _route_changing,
    _validate_registration,
    summarize_rows,
)


class StagedVerifierConfirmatoryTests(unittest.TestCase):
    def test_selection_is_ordered_clean_correct_and_excludes(self):
        rows = select_clean_correct(
            predictions=[0, 1, 9, 3, 4, 5],
            labels=[0, 2, 9, 3, 0, 5],
            start_index=1,
            sample_count=2,
            excluded_indices={2},
        )
        self.assertEqual([row["dataset_index"] for row in rows], [3, 5])
        self.assertEqual(
            [row["clean_correct_rank_after_start"] for row in rows], [1, 2]
        )

    def test_frozen_registration_is_hash_bound_and_disjoint(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        selection = _validate_registration(config)
        self.assertEqual(len(selection["samples"]), 100)
        self.assertTrue(
            all(int(row["dataset_index"]) >= 2000 for row in selection["samples"])
        )
        self.assertEqual(selection["request"]["epsilon_label"], "2/255")
        self.assertEqual(OUTPUT, DEFAULT_CONFIG.parent / OUTPUT.name)

    def test_route_changing_requires_exact_complete_coverage(self):
        exact = {
            "route_coverage": {
                "coverage_complete": True,
                "route_sets_exact": True,
                "feasible_route_sets": [[0, 1], [0, 2]],
            }
        }
        self.assertTrue(_route_changing(exact))
        exact["route_coverage"]["feasible_route_sets"] = [[0, 1]]
        self.assertFalse(_route_changing(exact))
        exact["route_coverage"]["coverage_complete"] = False
        self.assertIsNone(_route_changing(exact))

    def test_summary_uses_all_rows_and_replayed_unsafe(self):
        rows = [
            {
                "sample_rank": 0,
                "status": "SAFE",
                "reason": "SAFE_WEIGHTED_RANGE",
                "decision_tier": "TIER2_F0",
                "route_changing": True,
                "f0_invoked": True,
                "full_model_witness_valid": False,
                "verifier_total_seconds": 3.0,
                "outer_hard_timeout": False,
                "evidence_audit_status": "PASS",
                "evidence_audit_issue_count": 0,
            },
            {
                "sample_rank": 1,
                "status": "UNSAFE",
                "reason": "UNSAFE_FULL_FORWARD",
                "decision_tier": "TIER1_GATE_ELIMINATION",
                "route_changing": False,
                "f0_invoked": False,
                "full_model_witness_valid": True,
                "verifier_total_seconds": 5.0,
                "outer_hard_timeout": False,
                "evidence_audit_status": "PASS",
                "evidence_audit_issue_count": 0,
            },
        ]
        summary = summarize_rows(rows, 2)
        self.assertEqual(summary["complete_outcomes"], 2)
        self.assertEqual(summary["route_changing_safe"], 1)
        self.assertEqual(summary["timing"]["median_verifier_seconds"], 4.0)
        self.assertTrue(summary["preregistered_replication_signal_met"])


if __name__ == "__main__":
    unittest.main()
