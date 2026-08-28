# ===- act/pipeline/moe/test_accounting.py - Accounting Tests --------====#

import unittest

from act.pipeline.moe.accounting import guard_binary_accounting


class GuardAccountingTests(unittest.TestCase):
    def test_experiment1c_identity_closes(self):
        result = guard_binary_accounting(
            1550,
            1159,
            {"lp_eliminated": 158, "milp_eliminated": 175},
        )
        self.assertEqual(result.binary_eliminated, 391)
        self.assertEqual(result.structural_or_propagation_eliminated, 58)
        self.assertEqual(
            result.binary_eliminated,
            result.lp_support_eliminated
            + result.milp_support_eliminated
            + result.structural_or_propagation_eliminated,
        )

    def test_support_cannot_exceed_actual_reduction(self):
        with self.assertRaises(ValueError):
            guard_binary_accounting(
                10,
                9,
                {"lp_eliminated": 1, "milp_eliminated": 1},
            )

    def test_guard_cannot_increase_expert_binary_width(self):
        with self.assertRaises(ValueError):
            guard_binary_accounting(
                4,
                5,
                {"lp_eliminated": 0, "milp_eliminated": 0},
            )


if __name__ == "__main__":
    unittest.main()
