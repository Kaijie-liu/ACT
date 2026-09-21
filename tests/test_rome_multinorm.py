import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rome_multinorm_batch import summarize


class MultiNormControls(unittest.TestCase):
    def test_timeout_is_not_robust(self):
        rows = [{'index': 1, 'norm': n, 'status': 'ATTACK_EVALUATION_COMPLETED',
                 'empirically_correct_after_attack': True} for n in ['Linf', 'L1', 'L2']]
        self.assertEqual(summarize(rows, [1])[0]['status'], 'NO_ATTACK_FOUND_ALL_THREE')
        rows[0] = {'index': 1, 'norm': 'Linf', 'status': 'TIMEOUT'}
        self.assertEqual(summarize(rows, [1])[0]['status'], 'INCOMPLETE_NOT_ROBUST')
        rows[1]['empirically_correct_after_attack'] = False
        self.assertEqual(summarize(rows, [1])[0]['status'], 'ATTACK_FOUND')

    def test_missing_or_duplicate_norm_rejected(self):
        with self.assertRaises(ValueError):
            summarize([], [1])
        with self.assertRaises(ValueError):
            summarize([{'index': 1, 'norm': 'Linf', 'status': 'TIMEOUT'}]*3, [1])


if __name__ == '__main__':
    unittest.main()
