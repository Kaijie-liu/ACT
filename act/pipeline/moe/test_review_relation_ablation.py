import json
import unittest
from act.pipeline.moe.review_relation_ablation import OUTPUT, gates


class ReviewRelationTests(unittest.TestCase):
    def test_archived_denominators_and_gain_sources(self):
        r = json.loads(OUTPUT.read_text())
        self.assertTrue(r['exact_reaudit_equals_saved'])
        a = r['experiments']['full']['summary']
        self.assertEqual((a['rows'], a['packages'], a['unsafe_replayed']), (60, 48, 16))
        self.assertEqual(a['common_fact_pairs_equal'], 30)
        self.assertEqual(len(r['missing_packages']), 12)
        gains = [d for d in r['discordances'] if d['shared']['status']=='SAFE']
        self.assertEqual(len(gains), 3)
        self.assertEqual(sorted(d['shared']['pair_count'] for d in gains), [1,2,3])
        self.assertEqual(sum('RELAXATION' in d['independent']['reason'] for d in gains), 2)
        self.assertEqual(r['recorded_gate_pairs']['different'], 0)

    def test_conflicting_gate_records_are_rejected(self):
        e = {'tier2': {'pairs': [{'pair': [0,1], 'property_rows': [
            {'margin_bounds': [0,1], 'lambda_bounds': [.5,.8]},
            {'margin_bounds': [0,2], 'lambda_bounds': [.5,.9]}]}]}}
        with self.assertRaises(ValueError): gates(e)


if __name__ == '__main__': unittest.main()
