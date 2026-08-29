"""Unit tests for the incremental-HiGHS engineering audit."""

from __future__ import annotations

import unittest

from act.pipeline.moe.audit_experiment1_highspy_engineering import (
    _highspy_safe_property_issues,
    _paired_summary,
    _session_issues,
)


def _telemetry(**updates):
    value = {
        "backend": "highspy_incremental_hz",
        "build_error": None,
        "model_build_failures": 0,
        "model_builds": 1,
        "warnings_fail_closed": True,
        "run_time_limit_warnings_accepted": 0,
        "status_counts": {"optimal": 1},
    }
    value.update(updates)
    return value


class IncrementalHighsAuditTest(unittest.TestCase):
    def test_accepts_sound_highspy_safe_property(self):
        prop = {
            "status": "SAFE",
            "reason": "SAFE_WEIGHTED_RANGE",
            "minimum": 0.25,
            "solver_status": 7,
            "solver_bound_kind": "highs_mip_dual_bound",
            "attempts": [
                {
                    "status": "SAFE",
                    "reason": "SAFE_WEIGHTED_RANGE",
                    "solver_status": 7,
                    "certified_lower_bound": 0.25,
                    "dual_bound": 0.250000001,
                    "gap": 0.0,
                    "incremental_hz": _telemetry(),
                }
            ],
        }
        self.assertEqual(
            _highspy_safe_property_issues(prop, tolerance=1e-7), []
        )

    def test_rejects_nonoptimal_or_nonpositive_safe_property(self):
        prop = {
            "status": "SAFE",
            "reason": "SAFE_WEIGHTED_RANGE",
            "minimum": -1e-5,
            "solver_status": 13,
            "solver_bound_kind": "highs_mip_dual_bound",
            "attempts": [
                {
                    "status": "SAFE",
                    "reason": "SAFE_WEIGHTED_RANGE",
                    "solver_status": 13,
                    "certified_lower_bound": -1e-5,
                    "dual_bound": -1e-5,
                    "gap": 1.0,
                    "incremental_hz": _telemetry(
                        build_error="mutation", status_counts={"time_limit_reached": 1}
                    ),
                }
            ],
        }
        issues = _highspy_safe_property_issues(prop, tolerance=1e-7)
        self.assertGreaterEqual(len(issues), 6)

    def test_accepts_reused_scipy_safe_property(self):
        prop = {
            "reused_parent": True,
            "minimum": 0.1,
            "solver_status": 0,
            "solver_bound_kind": "mip_dual_bound",
        }
        self.assertEqual(
            _highspy_safe_property_issues(prop, tolerance=1e-7), []
        )

    def test_pair_level_reuse_marks_copied_property_as_scipy(self):
        prop = {
            "minimum": 0.1,
            "solver_status": 0,
            "solver_bound_kind": "mip_dual_bound",
        }
        self.assertEqual(
            _highspy_safe_property_issues(
                prop, tolerance=1e-7, pair_reused=True
            ),
            [],
        )

    def test_time_limit_warning_accounting_is_exact(self):
        row = {
            "gate": {
                "branches": [
                    {
                        "support": None,
                        "attempts": [
                            {
                                "metadata": {
                                    "incremental_hz": _telemetry(
                                        run_time_limit_warnings_accepted=1,
                                        status_counts={"time_limit_reached": 1},
                                    )
                                }
                            }
                        ],
                    }
                ]
            },
            "f0": {"pairs": []},
        }
        issues, counts = _session_issues([row])
        self.assertEqual(issues, [])
        self.assertEqual(counts["accepted_time_limit_warnings"], 1)
        row["gate"]["branches"][0]["attempts"][0]["metadata"][
            "incremental_hz"
        ]["run_time_limit_warnings_accepted"] = 2
        issues, _ = _session_issues([row])
        self.assertTrue(any("differs" in issue for issue in issues))

    def test_paired_summary_does_not_confuse_parent_closure_with_gain(self):
        baseline = [
            {"sample_rank": 1, "status": "UNKNOWN", "total_seconds": 10.0},
            {"sample_rank": 2, "status": "SAFE", "total_seconds": 20.0},
        ]
        rows = [
            {"sample_rank": 1, "status": "SAFE", "total_seconds": 15.0},
            {"sample_rank": 2, "status": "SAFE", "total_seconds": 18.0},
        ]
        summary = _paired_summary(rows, baseline)
        self.assertEqual(summary["newly_solved_vs_d0"], 1)
        self.assertEqual(summary["net_solved_gain_vs_d0"], 1)
        self.assertEqual(summary["status_transitions"], {
            "UNKNOWN->SAFE": 1,
            "SAFE->SAFE": 1,
        })


if __name__ == "__main__":
    unittest.main()
