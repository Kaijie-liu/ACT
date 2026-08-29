import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from act.pipeline.moe.benchmark_guarded_box_hull import (
    _backend_order,
    _max_abs_bound_difference,
    _save_bounds_artifact,
    _summary,
)

import numpy as np


def _backend(*, complete=True, seconds=1.0, fallback_sides=0, model_builds=1):
    return {
        "complete": complete,
        "domain_status": "optimal" if complete else "partial",
        "exact": complete,
        "relaxed_binaries": 0,
        "fallback_sides": fallback_sides,
        "bounds_sha256": "synthetic",
        "wall_seconds": seconds,
        "telemetry": {
            "model_builds": model_builds,
            "model_build_seconds": 0.1,
            "objective_update_calls": 6,
            "objective_coefficients_changed": 12,
            "solves": 6,
            "cold_start_solves": model_builds,
            "basis_submission_attempts": 5 if model_builds == 1 else 0,
            "basis_submissions_accepted": 4 if model_builds == 1 else 0,
            "basis_valid_after_solve": 6 if model_builds == 1 else 0,
            "simplex_iterations": 9,
            "ipm_iterations": 0,
            "objective_update_seconds": 0.01,
            "solve_seconds": 0.8,
            "total_seconds": 0.9,
            "status_counts": {"optimal": 6},
        },
        "error": None,
    }


def _branch(rank, *, high_complete=True, scipy_complete=True, difference=1e-10):
    return {
        "branch_id": f"rank{rank}:pair0-1",
        "sample_rank": rank,
        "backend_order": list(_backend_order(rank)),
        "paired_complete": high_complete and scipy_complete,
        "bound_max_abs_diff": difference,
        "highspy": _backend(
            complete=high_complete,
            seconds=1.0,
            fallback_sides=0 if high_complete else 2,
            model_builds=1,
        ),
        "scipy": _backend(
            complete=scipy_complete,
            seconds=3.0,
            fallback_sides=0 if scipy_complete else 2,
            model_builds=6,
        ),
    }


class GuardedBoxHullBenchmarkTests(unittest.TestCase):
    def test_sample_rank_parity_freezes_alternating_backend_order(self):
        self.assertEqual(_backend_order(110), ("highspy", "scipy"))
        self.assertEqual(_backend_order(111), ("scipy", "highspy"))

    def test_bound_difference_compares_both_sides(self):
        left = (np.asarray([-1.0, 0.0]), np.asarray([2.0, 3.0]))
        right = (np.asarray([-1.25, 0.0]), np.asarray([2.0, 3.5]))
        self.assertEqual(_max_abs_bound_difference(left, right), 0.5)

    def test_bound_artifact_retains_every_compared_array(self):
        arrays = {
            "highspy": (np.asarray([-1.0]), np.asarray([2.0])),
            "scipy": (np.asarray([-1.25]), np.asarray([2.5])),
        }
        with TemporaryDirectory(dir="/data1/Kane/MOE/cache/tmp") as directory:
            relative, digest = _save_bounds_artifact(
                Path(directory), "rank1:pair0-1", arrays
            )
            path = Path(directory) / relative
            self.assertEqual(len(digest), 64)
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(payload["highspy_lower"].tolist(), [-1.0])
                self.assertEqual(payload["scipy_upper"].tolist(), [2.5])

    def test_complete_synthetic_summary_aggregates_paired_telemetry(self):
        branches = [_branch(110), _branch(111, difference=2e-10)]
        rows = [
            {"sample_rank": rank, "error": None}
            for rank in (110, 111)
        ]
        summary = _summary(
            branches,
            rows,
            expected_rows=2,
            comparison_tolerance=1e-8,
        )
        self.assertEqual(summary["paired_complete_branches"], 2)
        self.assertEqual(summary["highspy"]["model_builds"], 2)
        self.assertEqual(summary["scipy"]["model_builds"], 12)
        self.assertEqual(summary["highspy"]["solves"], 12)
        self.assertEqual(summary["highspy"]["basis_submissions_accepted"], 8)
        self.assertAlmostEqual(
            summary["complete_pair_bound_max_abs_diff"], 2e-10
        )
        self.assertTrue(summary["speed_conclusion"]["eligible"])
        self.assertEqual(
            summary["speed_conclusion"]["scipy_over_highspy_wall_ratio_median"],
            3.0,
        )

    def test_any_incomplete_backend_suppresses_speed_conclusion(self):
        branches = [_branch(110), _branch(111, high_complete=False)]
        rows = [
            {"sample_rank": 110, "error": None},
            {"sample_rank": 111, "error": None},
        ]
        summary = _summary(
            branches,
            rows,
            expected_rows=2,
            comparison_tolerance=1e-8,
        )
        self.assertEqual(summary["incomplete_or_fallback_branches"], 1)
        self.assertEqual(summary["highspy"]["fallback_sides"], 2)
        self.assertFalse(summary["speed_conclusion"]["eligible"])
        self.assertIsNone(
            summary["speed_conclusion"]["scipy_over_highspy_wall_ratio_median"]
        )

    def test_complete_bound_disagreement_is_auditable(self):
        summary = _summary(
            [_branch(110, difference=1e-4)],
            [{"sample_rank": 110, "error": None}],
            expected_rows=1,
            comparison_tolerance=1e-8,
        )
        self.assertEqual(
            summary["complete_pair_bound_disagreement_branch_ids"],
            ["rank110:pair0-1"],
        )
        self.assertFalse(summary["speed_conclusion"]["eligible"])


if __name__ == "__main__":
    unittest.main()
