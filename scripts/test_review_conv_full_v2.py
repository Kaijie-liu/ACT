import unittest

from scripts.review_conv_full_v2 import witness_context


class ReviewTests(unittest.TestCase):
    def test_method_runs_not_unique_inputs_or_promoted_timeouts(self):
        rows = [dict(dataset_index=8, method=a, status=s)
                for a, s in [('adaptive', 'UNSAFE'), ('crown', 'UNSAFE'),
                             ('monolithic', 'TIMEOUT')]]
        result = witness_context(rows)
        self.assertEqual(result['unsafe_method_runs'], 2)
        self.assertEqual(result['distinct_inputs_with_replayed_witness'], [8])
        self.assertEqual(result['timeout_with_other_arm_witness']['monolithic'], [8])
        self.assertEqual(rows[-1]['status'], 'TIMEOUT')

    def test_numerical_positive_does_not_create_witness(self):
        rows = [dict(dataset_index=98, method='crown', status='POSITIVE'),
                dict(dataset_index=98, method='adaptive', status='TIMEOUT')]
        self.assertEqual(witness_context(rows)['timeout_with_other_arm_witness']['adaptive'], [])


if __name__ == '__main__':
    unittest.main()
