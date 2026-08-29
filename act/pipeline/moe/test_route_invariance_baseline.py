import unittest

from act.pipeline.moe.route_invariance_baseline import _summary


class RouteInvarianceBaselineTests(unittest.TestCase):
    def test_summary_keeps_precondition_and_property_coverage_separate(self):
        rows = [
            {
                "sample_rank": 100,
                "endpoint_kind": "NO_BOUNDARY_CAP",
                "route_precondition_status": "INVARIANT",
                "baseline_status": "SAFE",
                "route_a_status": "SAFE",
                "route_a_only_safe": False,
                "baseline_seconds": 2.0,
                "route_a_seconds": 2.0,
            },
            {
                "sample_rank": 101,
                "endpoint_kind": "ROUTE_BOUNDARY_PRIMARY",
                "route_precondition_status": "UNSTABLE",
                "baseline_status": "UNKNOWN",
                "route_a_status": "SAFE",
                "route_a_only_safe": True,
                "baseline_seconds": 0.1,
                "route_a_seconds": 3.0,
            },
        ]
        summary = _summary(rows)
        self.assertEqual(summary["baseline_solved"], 1)
        self.assertEqual(summary["route_a_solved"], 2)
        self.assertEqual(summary["coverage_difference"], 1)
        self.assertEqual(summary["route_a_only_safe_ranks"], [101])

    def test_followup_is_reported_but_does_not_overwrite_primary(self):
        row = {
            "sample_rank": 155,
            "endpoint_kind": "ROUTE_BOUNDARY_PRIMARY",
            "route_precondition_status": "UNSTABLE",
            "baseline_status": "UNKNOWN",
            "route_a_status": "TIMEOUT",
            "route_a_followup_status": "SAFE",
            "route_a_only_safe": False,
            "baseline_seconds": 1.0,
            "route_a_seconds": 300.0,
        }
        summary = _summary([row])
        self.assertEqual(summary["route_a_solved"], 0)
        self.assertEqual(summary["route_a_followup_solved"], 1)


if __name__ == "__main__":
    unittest.main()
