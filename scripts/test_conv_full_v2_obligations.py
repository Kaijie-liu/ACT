from collections import Counter
import json
import unittest

from scripts.analyze_conv_full_v2_obligations import parse, scope_keys, reusable_keys, finite, RAW, OUTPUT


class ObligationTests(unittest.TestCase):
    def events(self):
        return [
            dict(seq=0, kind='PROPERTY_BEGIN', function='solve_monolithic_weighted_top2_f0',
                 scope={'pairs': [[0, 1], [0, 2]], 'row': [1., -1.] + [0.] * 8, 'constant': 0}),
            dict(seq=1, kind='NATIVE_READY', deadline=20.),
            dict(seq=2, kind='NATIVE_RETURN', token=1, entered=2., effective=18., clock_elapsed=21.,
                 result={'status': 1}),
            dict(seq=3, kind='PROPERTY_RESULT', token=0, result={'status': 'UNKNOWN', 'solver_status': 1})]

    def test_union_is_one_query_two_obligations_missing_bounds_not_zero(self):
        p, n, stack = parse(self.events(), [(0, 1), (0, 2)], 0)
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0]['keys'], [((0, 1), 1), ((0, 2), 1)])
        self.assertEqual(n[1]['owner'], 0)
        self.assertFalse(finite(n[1]['end']['result'].get('mip_dual_bound')))
        self.assertFalse(stack)

    def test_wrong_class_sign_constant_pair_and_duplicate_pair_rejected(self):
        for change in [dict(row=[-1., 1.] + [0.] * 8), dict(constant=1),
                       dict(pairs=[[1, 3]]), dict(pairs=[[0, 1], [0, 1]])]:
            s = {**self.events()[0]['scope'], **change}
            with self.assertRaises(ValueError): scope_keys(s, [(0, 1), (0, 2)], 0)

    def test_duplicate_query_rejected_in_no_retry_protocol(self):
        e = self.events(); e.append({**e[0], 'seq': 4})
        with self.assertRaises(ValueError): parse(e, [(0, 1), (0, 2)], 0)

    def test_unmatched_or_duplicate_native_terminal_rejected(self):
        for token in (99,):
            e = self.events(); e[2]['token'] = token
            with self.assertRaises(ValueError): parse(e, [(0, 1), (0, 2)], 0)
        e = self.events(); e.insert(3, {**e[2], 'seq': 7})
        with self.assertRaises(ValueError): parse(e, [(0, 1), (0, 2)], 0)

    def test_overallocation_rejected_but_late_return_allowed(self):
        e = self.events(); parse(e, [(0, 1), (0, 2)], 0)
        e[2]['effective'] = 19.
        with self.assertRaises(ValueError): parse(e, [(0, 1), (0, 2)], 0)

    def test_unreturned_not_fabricated_and_raise_not_result(self):
        e = self.events(); p, n, stack = parse(e[:2], [(0, 1), (0, 2)], 0)
        self.assertIsNone(p[0]['end']); self.assertIsNone(n[1]['end']); self.assertEqual(len(stack), 1)
        e[3] = dict(seq=3, kind='PROPERTY_RAISE', token=0, exception='BudgetExhausted')
        p, _, _ = parse(e, [(0, 1), (0, 2)], 0)
        self.assertNotIn('result', p[0]['end'])

    def test_misnested_scope_and_unbound_replay_rejected(self):
        e = self.events(); e.insert(1, dict(seq=9, kind='LOCAL_BEGIN', function='hz_minimize_output'))
        with self.assertRaises(ValueError): parse(e, [(0, 1), (0, 2)], 0)
        with self.assertRaises(ValueError): parse([dict(kind='PROPERTY_REPLAY', token=42)], [], 0)

    def test_reuse_needs_both_experts_and_frozen_fact_count(self):
        snapshot = {'identity': {'property': {'clean_prediction': 0}},
                    'scope': {'numerical_policy': {'outward_absolute': 1e-9, 'outward_relative': 1e-9,
                                                  'safe_positive_margin': 1e-7}},
                    'branches': [{'candidate': i, 'proof_output_bounds': {'lower': [2.] + [-1.] * 9,
                                                                        'upper': [3.] + [1.] * 9}}
                                 for i in (0, 1)], 'available_fact_count': 18,
                    'feasible_route_sets': [[0, 1], [0, 2]]}
        self.assertEqual(reusable_keys(snapshot), {((0, 1), j) for j in range(1, 10)})
        snapshot['available_fact_count'] = 17
        with self.assertRaises(ValueError): reusable_keys(snapshot)

    def test_archived_real_log_counts_and_obligation_partition(self):
        report = json.loads(OUTPUT.read_text())
        frozen = json.loads((RAW / 'audit.final.json').read_text())
        details = {d['job_id']: d for d in frozen['details']}
        for row in report['rows']:
            with self.subTest(job=row['job_id']):
                events = [json.loads(l) for l in (RAW / row['job_id'] / 'budget_journal.jsonl').read_text().splitlines()]
                results = [e for e in events if e['kind'] == 'PROPERTY_RESULT']
                self.assertEqual(len(results), len(row['property_queries']))
                self.assertEqual(dict(Counter(e['result']['reason'] for e in results)),
                                 details[row['job_id']]['stopping_observations']['property_reasons'])
                self.assertEqual(sum(row['obligation_counts'].values()), row['required_pair_property_count'])
                self.assertEqual(row['required_pair_property_count'], 9 * len(row['pairs']))
                self.assertIsNotNone(row['f0_entry_seconds'])
                self.assertFalse(row['native_unreturned_tokens'])
        self.assertEqual(report['all_act']['property_begins'], 419)
        self.assertEqual(report['all_act']['property_states'], {'RETURNED_SOLVER_LIMIT': 419})


if __name__ == '__main__':
    unittest.main()
