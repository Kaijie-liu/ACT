import unittest

from act.pipeline.moe.summarize_guarded_box_hull_benchmark import summarize


def _branch(order, highspy_seconds, scipy_seconds):
    return {
        "backend_order": order,
        "paired_complete": True,
        "within_tolerance": True,
        "bound_max_abs_diff": 0.0,
        "highspy": {
            "wall_seconds": highspy_seconds,
            "fallback_sides": 0,
            "telemetry": {"model_builds": 1, "solves": 4, "cold_start_solves": 1},
        },
        "scipy": {
            "wall_seconds": scipy_seconds,
            "fallback_sides": 0,
            "telemetry": {"model_builds": 4, "solves": 4, "cold_start_solves": 4},
        },
    }


class SummarizeGuardedBoxHullBenchmarkTests(unittest.TestCase):
    def test_paired_speed_and_model_reuse(self):
        result = summarize(
            [
                _branch(["highspy", "scipy"], 2.0, 20.0),
                _branch(["scipy", "highspy"], 4.0, 32.0),
            ],
            [{"complete": True}, {"complete": True}],
        )
        self.assertTrue(result["speed"]["eligible"])
        self.assertEqual(result["speed"]["scipy_over_highspy_wall_ratio_median"], 9.0)
        self.assertEqual(result["highspy"]["model_builds"], 2)
        self.assertEqual(result["scipy"]["model_builds"], 8)
        self.assertEqual(result["backend_order_counts"]["highspy->scipy"], 1)
        self.assertEqual(result["backend_order_counts"]["scipy->highspy"], 1)

    def test_incomplete_branch_disables_speed(self):
        branch = _branch(["highspy", "scipy"], 2.0, 20.0)
        branch["paired_complete"] = False
        self.assertFalse(summarize([branch], [{"complete": True}])["speed"]["eligible"])


if __name__ == "__main__":
    unittest.main()
