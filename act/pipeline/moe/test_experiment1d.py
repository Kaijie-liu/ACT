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
    _assert_support_identity,
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

    def test_fallback_side_drift_is_recorded_not_structural(self):
        actual = {
            "relu_binaries": 55, "binary_width": 38,
            "fast_unstable": 55, "after_lp_unstable": 38,
            "after_milp_unstable": 37, "lp_eliminated": 17,
            "milp_eliminated": 1, "fallback_sides": 6,
        }
        parent = {**actual, "fallback_sides": 5}
        report = _assert_support_identity(actual, parent)
        self.assertTrue(report["structural_identity"])
        self.assertEqual(report["fallback_side_drift"], 1)

    def test_structural_support_drift_remains_an_error(self):
        actual = {
            "relu_binaries": 55, "binary_width": 38,
            "fast_unstable": 55, "after_lp_unstable": 38,
            "after_milp_unstable": 37, "lp_eliminated": 17,
            "milp_eliminated": 1, "fallback_sides": 6,
        }
        with self.assertRaises(RuntimeError):
            _assert_support_identity(
                actual, {**actual, "binary_width": 39}
            )

    def test_explicit_engineering_backend_drift_is_recorded(self):
        actual = {
            "relu_binaries": 50, "binary_width": 34,
            "fast_unstable": 55, "after_lp_unstable": 35,
            "after_milp_unstable": 34, "lp_eliminated": 20,
            "milp_eliminated": 1, "fallback_sides": 0,
        }
        parent = {
            "relu_binaries": 52, "binary_width": 36,
            "fast_unstable": 55, "after_lp_unstable": 38,
            "after_milp_unstable": 36, "lp_eliminated": 17,
            "milp_eliminated": 2, "fallback_sides": 4,
        }
        report = _assert_support_identity(
            actual, parent, allow_backend_drift=True
        )
        self.assertFalse(report["structural_identity"])
        self.assertTrue(report["backend_drift_allowed"])
        self.assertIn("binary_width", report["structural_drift"])


if __name__ == "__main__":
    unittest.main()
