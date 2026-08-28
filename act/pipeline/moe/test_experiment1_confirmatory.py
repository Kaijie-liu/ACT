# ===- act/pipeline/moe/test_experiment1_confirmatory.py - Tests -----====#

import json
import unittest

from act.back_end.solver.solver_hz import hz_numerical_policy_manifest
from act.pipeline.moe.audit_experiment1_confirmatory import _cluster_bootstrap
from act.pipeline.moe.experiment1_confirmatory import (
    DEFAULT_CONFIG,
    _boundary_summary,
    _census_summary,
)


class ConfirmatoryProtocolTests(unittest.TestCase):
    def test_tracked_numerical_policy_matches_implementation(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(
            config["numerical_safety"], hz_numerical_policy_manifest()
        )

    def test_census_summary_closes_guard_accounting(self):
        row = {
            "sample_rank": 100,
            "status": "COMPLETE",
            "route_set_unstable": True,
            "exact_candidate_count": 3,
            "ibp_candidate_count": 5,
            "zonotope_candidate_count": 4,
            "route_conditioned_max_width": 6,
            "candidate_pruned_monolithic_width": 12,
            "total_seconds": 1.0,
            "branches": [
                {
                    "guard_accounting": {
                        "binaries_before": 10,
                        "binaries_after": 5,
                        "binary_eliminated": 5,
                        "lp_support_eliminated": 2,
                        "milp_support_eliminated": 1,
                        "structural_or_propagation_eliminated": 2,
                    },
                    "support": {"seconds": 0.5},
                }
            ],
        }
        summary = _census_summary([row])
        self.assertEqual(summary["exact_reduces_ibp_rows"], 1)
        self.assertEqual(summary["exact_reduces_zonotope_rows"], 1)
        self.assertEqual(
            summary["width_ratio_candidate_gt_topk"]["median"], 0.5
        )
        self.assertEqual(
            summary["guard_accounting"][
                "structural_or_propagation_eliminated"
            ],
            2,
        )

    def test_boundary_summary_reports_f0_increment(self):
        rows = [
            {
                "sample_rank": 100,
                "status": "SAFE",
                "reason": "SAFE_WEIGHTED_RANGE",
                "unique_safe": True,
                "gate_reason": "UNKNOWN_GATE_SUFFICIENCY",
                "f0_invoked": True,
                "f0_seconds": 2.0,
                "total_seconds": 3.0,
                "gate": {"branches": []},
            },
            {
                "sample_rank": 101,
                "status": "UNKNOWN",
                "reason": "UNKNOWN_SOLVER_LIMIT",
                "unique_safe": False,
                "gate_reason": "UNKNOWN_SOLVER_LIMIT",
                "f0_invoked": False,
                "f0_seconds": 0.0,
                "total_seconds": 1.0,
                "gate": {"branches": []},
            },
        ]
        summary = _boundary_summary(rows)
        self.assertEqual(summary["base_semantic_incompleteness"], 1)
        self.assertEqual(summary["f0_invoked"], 1)
        self.assertEqual(summary["f0_resolved"], 1)
        self.assertEqual(summary["f0_added_safe"], 1)
        self.assertEqual(
            summary["f0_paired_runtime_overhead"]["median"], 2.0
        )

    def test_cluster_bootstrap_uses_sample_clusters(self):
        rows = [
            {"sample_rank": 100, "hit": True},
            {"sample_rank": 100, "hit": True},
            {"sample_rank": 101, "hit": False},
        ]
        interval = _cluster_bootstrap(
            rows,
            lambda row: row["hit"],
            replicates=200,
            seed=3,
        )
        self.assertLessEqual(interval[0], 2.0 / 3.0)
        self.assertGreaterEqual(interval[1], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
