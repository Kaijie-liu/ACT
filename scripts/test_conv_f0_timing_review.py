"""Read-only checks of the derived review against the retained raw trace."""
import json
import unittest

from scripts.conv_three_arm_contract import ROOT, read
from scripts.run_conv_f0_timing import DEFAULT
from scripts.archive_conv_f0_timing import summarize
from act.pipeline.moe.experiment1 import _sha256


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = read(ROOT/'act/pipeline/moe/results/conv_f0_timing_review_20260915_r1.json')
        cls.events = [json.loads(line) for line in (DEFAULT/'rank0_monolithic/trace.jsonl').read_text().splitlines()]
        cls.starts = {e['seq']:e for e in cls.events if e['kind']=='BEGIN'}
        cls.ends = {e['span']:e for e in cls.events if e['kind'] in ('END','RAISE')}

    def test_three_audits_and_derived_summary(self):
        automatic = read(DEFAULT/'audit.final.json')
        self.assertEqual(automatic, read(DEFAULT/'audit.review.json'))
        self.assertEqual(self.review['timing'], summarize(automatic))
        self.assertEqual(self.review['row']['status'], 'TIMEOUT')
        self.assertIsNone(self.review['row']['package'])

    def test_native_totals_recomputed_from_raw_events(self):
        calls = [b for b in self.starts.values() if b['name']=='scipy.optimize.milp'
                 and self.starts.get(b['parent'], {}).get('name','').endswith('.solve_monolithic_weighted_top2_f0')]
        completed = [b for b in calls if b['seq'] in self.ends]
        self.assertEqual((len(calls),len(completed)), (9,8))
        value = sum(self.ends[b['seq']]['elapsed']-b['elapsed'] for b in completed)
        self.assertAlmostEqual(value, self.review['timing']['property_native_completed_seconds'], places=10)
        for b in completed:
            self.assertEqual(self.ends[b['seq']]['result'], {'status':1})
        last = self.review['timing']['property_table'][-1]
        self.assertIsNone(last['native_seconds'])
        self.assertIsNone(last['native_result'])
        self.assertAlmostEqual(last['native_limit_excess_over_request_remaining'],
            last['native_requested_limit']-(300-last['native_start']), places=10)
        self.assertGreater(last['native_limit_excess_over_request_remaining'], 1.5)

    def test_support_exact_uses_inner_return_not_missing_dispatch_field(self):
        for label, relaxed in [('LP',True), ('MILP',False)]:
            parents = {b['seq'] for b in self.starts.values() if b['name'].endswith('._guarded_support_query')
                       and b['arguments']['relax_binaries'] is relaxed}
            children = [b for b in self.starts.values() if b['parent'] in parents
                        and b['name'].endswith('.hz_support_bounds')]
            self.assertEqual(len(children), 12)
            self.assertTrue(all('exact' not in self.ends[p]['result'] for p in parents))
            exact = sum(self.ends[b['seq']]['result']['exact'] is True for b in children)
            self.assertEqual(exact, self.review['timing']['guarded_support_within_pair_propagation'][label]['exact_completed_calls'])

    def test_union_coverage_not_caller_last_pair(self):
        for row in self.review['timing']['property_table']:
            self.assertEqual(row['encoding_count'], 2)
            self.assertEqual(row['covered_pairs'], [[1,2],[1,3]])
            self.assertEqual(row['property_index'], self.starts[row['solve_span_id']]['arguments']['property_index'])

    def test_archived_artifact_hashes(self):
        for item in self.review['artifact_inventory']:
            path = DEFAULT/item['path']
            self.assertEqual(path.stat().st_size, item['bytes'])
            self.assertEqual(_sha256(path), item['sha256'])


if __name__=='__main__':
    unittest.main()
