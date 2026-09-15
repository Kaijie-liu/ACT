import unittest
from scripts.analyze_proof_closure_costs import classify


class ClassificationTests(unittest.TestCase):
    def test_missing_does_not_become_checked_negative(self):
        self.assertEqual(classify(9, 7, 2), 'NO_COMPLETE_CHECKABLE_EVIDENCE')

    def test_checked_nonpositive_is_not_unsafe(self):
        self.assertEqual(classify(9, 7), 'CHECKED_NONPOSITIVE_BOUND')

    def test_all_positive_still_conditional(self):
        self.assertEqual(classify(9, 9), 'COMPLETE_CONDITIONAL_PROOF')


if __name__ == '__main__':
    unittest.main()
