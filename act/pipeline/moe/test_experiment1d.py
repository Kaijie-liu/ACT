import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from act.pipeline.moe.experiment1d import (
    DEFAULT_CONFIG,
    RowRecorder,
    _gate_support_record,
    _load_frozen_selection,
    _summary,
)


class Experiment1DTests(unittest.TestCase):
    def test_frozen_selection_is_exactly_twenty_applicable_rows(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            config = json.load(handle)
        rows = _load_frozen_selection(config)
        self.assertEqual(len(rows), 20)
        self.assertEqual(
            Counter(row["category"] for row in rows),
            Counter({
                "weighted_solver_limit": 14,
                "expert_timeout": 2,
                "hard_deadline": 2,
                "base_solver_limit": 1,
                "range_relaxation": 1,
            }),
        )
        self.assertNotIn(
            "NO_ROUTE_BOUNDARY_WITHIN_SEARCH",
            {row["parent"]["reason"] for row in rows},
        )

    def test_completion_before_300_still_writes_checkpoint(self):
        result_root = Path("/data1/Kane/MOE/ACT/data/moe/results")
        with tempfile.TemporaryDirectory(dir=result_root) as temporary:
            recorder = RowRecorder(Path(temporary), 300.0)
            recorder.progress(
                active={"tier": "F0", "property_index": 2},
                solver={"incumbent": 1.0, "dual_bound": 0.5, "gap": 0.5},
            )
            recorder.finish_checkpoint("SAFE")
            with (Path(temporary) / "checkpoint_300.json").open() as handle:
                checkpoint = json.load(handle)
        self.assertEqual(checkpoint["status"], "COMPLETED_BEFORE_CHECKPOINT")
        self.assertEqual(checkpoint["final_status"], "SAFE")

    def test_summary_keeps_parent_failure_and_conditional_closure_separate(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            config = json.load(handle)
        rows = [
            {"status": "SAFE", "reason": "SAFE_WEIGHTED_RANGE", "total_seconds": 1.0,
             "full_model_witness_valid": False}
            for _ in range(5)
        ] + [
            {"status": "UNKNOWN", "reason": "UNKNOWN_WEIGHTED_SOLVER_LIMIT", "total_seconds": 1.0,
             "full_model_witness_valid": False}
            for _ in range(15)
        ]
        summary = _summary(rows, config)
        self.assertEqual(summary["parent_overall_solved_rate_immutable"], 0.56)
        self.assertEqual(summary["parent_boundary_applicability"], "76/100")
        self.assertAlmostEqual(summary["closure_conditional_coverage"], 61 / 76)
        self.assertTrue(summary["baseline_unlock_pre_audit"])

    def test_gate_support_identity_uses_post_support_binary_universe(self):
        propagation = SimpleNamespace(
            guarded_support=({
                "fast_unstable": 69,
                "after_lp_unstable": 54,
                "after_milp_unstable": 52,
                "lp_eliminated": 15,
                "milp_eliminated": 2,
                "lp_seconds": 0.0,
                "milp_seconds": 0.0,
                "lp_fallback_sides": 2,
                "milp_fallback_sides": 2,
            },),
            binary_width=56,
        )
        record = _gate_support_record(propagation, shared_binary_width=4)
        self.assertEqual(record["fast_unstable"], 69)
        self.assertEqual(record["relu_binaries"], 52)
        self.assertEqual(record["binary_width"], 56)


if __name__ == "__main__":
    unittest.main()
