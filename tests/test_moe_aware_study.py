"""No solver/model execution: fixed queue, paired uncertainty and stop rules."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from act.pipeline.moe import moe_aware_study as S


class Study(unittest.TestCase):
    def test_four_budgets_and_no_transfer(self):
        rows = S.catalog(range(5000, 5020), S.MECHANISMS)
        self.assertEqual(len(rows), 1320)
        self.assertEqual(S.Counter(r['part'] for r in rows),
            {'main': 600, 'ablation': 360, 'radius': 240, 'boundary': 120})
        self.assertEqual(sum(t for _, t in S.CAPS.values()), 110*3600)
        self.assertEqual(len({(r['part'], r['model'], r['index'], r['arm']) for r in rows}), 1320)
        a = S.catalog(range(5000, 5020), ('M-A',))
        self.assertEqual(len(a), 1080)
        self.assertEqual(sum(r['part'] == 'boundary' for r in a), 120)
        self.assertEqual(sum(r['part'] == 'main' for r in a), 600)
        self.assertTrue(all(r['reference_part'] == 'main' for r in a if r['part'] == 'ablation'))
        self.assertTrue(all(not r['components'] for r in a if r['arm'] == 'minus_M-A'))

    def test_cluster_units_and_full_cost(self):
        rows = []
        for i in range(20):
            for m in ('a', 'b', 'c'):
                for arm in ('off', 'on'):
                    positive = (arm == 'on' and i < 4)
                    rows.append(dict(model=m, index=i, arm=arm, status='POSITIVE' if positive else 'TIMEOUT',
                        accepted=positive, seconds=10. if positive else 300.))
        result = S.contrast(rows, ('a', 'b', 'c'), range(20))
        self.assertEqual(result['net'], 12); self.assertEqual(result['positive_rate_delta'], .2)
        self.assertEqual(result['decision'], 'PASS'); self.assertEqual(result['cost']['median_ratio'], 1.)
        # Three identical models must give the same interval as one input series,
        # not spuriously narrower uncertainty from60 independent model pairs.
        one = S.contrast([r for r in rows if r['model'] == 'a'], ('a',), range(20))
        self.assertEqual(result['bootstrap95'], one['bootstrap95'])
        self.assertEqual(result['cost']['total_seconds']['off'], 18000.)
        self.assertEqual(S.contrast(rows[:-1], ('a', 'b', 'c'), range(20))['decision'],
                         'PENDING_INCOMPLETE_QUEUE')
        with self.assertRaises(ValueError): S.contrast(rows+rows[:1], ('a', 'b', 'c'), range(20))

    def test_decision_and_evidence_conflict(self):
        self.assertEqual(S.decision(15, .05, [.01, .09], 2.), 'PASS')
        self.assertEqual(S.decision(15, .05, [0., .09], 2.), 'GRAY_REPORT')
        self.assertEqual(S.decision(14, 14/300, [.01, .09], 1.), 'STOP_REGISTERED_GATE')
        self.assertEqual(S.decision(16, 16/300, [.01, .09], 2.001), 'STOP_REGISTERED_GATE')
        rows = [dict(model='a', index=1, arm=arm, status=status, accepted=True, seconds=1.)
                for arm, status in [('off', 'UNSAFE_REPLAYED'), ('on', 'POSITIVE')]]
        result = S.contrast(rows, ('a',), (1,), replicates=10)
        self.assertEqual(result['decision'], 'EVIDENCE_CONFLICT_REPORT')
        self.assertEqual(result['conflicts'], [('a', 1)])
        rows[1]['accepted'] = False
        with self.assertRaises(ValueError): S.contrast(rows, ('a',), (1,), replicates=10)


if __name__ == '__main__':
    unittest.main()
