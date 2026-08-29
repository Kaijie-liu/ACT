import unittest

from act.pipeline.moe.audit_guard_paired import paired_guard_statistics


def _branch(no_support, support, eliminated, no_seconds=2.0, yes_seconds=1.0):
    return {
        "matched_no_support_status": no_support,
        "branch_status": support,
        "matched_no_support_solve_seconds": no_seconds,
        "solve_time": yes_seconds,
        "guard_accounting": {"binary_eliminated": eliminated},
    }


class PairedGuardAuditTests(unittest.TestCase):
    def test_paired_table_keeps_direction(self):
        rows = [{"gate": {"branches": [
            _branch("unknown", "certified", 3),
            _branch("certified", "unknown", 0),
            _branch("certified", "certified", 1),
            _branch("unknown", "unknown", 0),
        ]}}]
        result = paired_guard_statistics(rows)
        self.assertEqual(result["branches"], 4)
        self.assertEqual(result["support_only_solved"], 1)
        self.assertEqual(result["no_support_only_solved"], 1)
        self.assertEqual(result["net_solved_gain"], 0)
        self.assertEqual(
            result["table"]["no_support_unsolved_support_unsolved_n00"], 1
        )


if __name__ == "__main__":
    unittest.main()
