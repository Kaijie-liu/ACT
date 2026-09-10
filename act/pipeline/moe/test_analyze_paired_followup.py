import copy
import unittest

from act.pipeline.moe.analyze_paired_followup import describe_rows
from act.pipeline.moe.paired_followup import METHODS


class AnalysisTests(unittest.TestCase):
    def rows(self):
        return [{"model": f"seed{s}", "method": method, "rank": 0,
                 "status": "SAFE" if method == "monolithic_f0" else "TIMEOUT",
                 "outer_timeout": method != "monolithic_f0",
                 "wall_seconds": 1. if method == "monolithic_f0" else 300.,
                 "reason": "control"}
                for s in range(3) for method in METHODS]

    def test_losses_and_censored_cost_are_not_dropped(self):
        result = describe_rows(self.rows(), [0])["models"]["seed0"]
        self.assertEqual(result["paired"]["monolithic_f0"]["safe"]["net_gain"], -1)
        self.assertEqual(result["methods"]["staged"]["mean_observed_wall_seconds"], 300.)
        self.assertEqual(result["methods"]["staged"]["outer_timeout_count"], 1)

    def test_missing_or_duplicate_rows_fail(self):
        rows = self.rows()
        for modified in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                describe_rows(modified, [0])

    def test_conflict_and_nonfinite_cost_fail(self):
        for key, value in (("status", "UNSAFE"), ("wall_seconds", float("nan"))):
            rows = copy.deepcopy(self.rows())
            rows[0][key] = value
            with self.assertRaises(ValueError):
                describe_rows(rows, [0])


if __name__ == "__main__":
    unittest.main()
