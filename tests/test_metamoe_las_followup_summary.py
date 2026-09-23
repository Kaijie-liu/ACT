"""Saved-analysis controls; no models, native calls or actual experiments."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_metamoe_las_followup import analyze, base_cost, query_observation


def archive():
    return {'audit': 'PASS', 'issues': 0, 'cost': {}, 'files': {}, 'rows': [
        {'id': 'mnist_7', 'arm': 'act', 'status': 'POSITIVE', 'seconds': 12.,
         'grade': 'HZ_POLICY_ACCEPTED', 'result': {'candidates': [0], 'unresolved': []},
         'details': {'experts': [], 'nonzero': []},
         'receipt': {'peak_sampled_group_rss_bytes': 20}},
        {'id': 'mnist_7', 'arm': 'author', 'status': 'TIMEOUT', 'seconds': 300.,
         'grade': 'NONE', 'result': None,
         'receipt': {'peak_sampled_group_rss_bytes': 30}},
    ]}


def query():
    return {'scope': {'phase': 'expanded', 'row': 1}, 'terminal': 'TIMEOUT',
            'returned_status': 'unknown', 'started_monotonic': 10.,
            'deadline_monotonic': 12., 'return_elapsed_seconds': 2.01,
            'native_started': True, 'native_result': None, 'token': 'current',
            'accepted_after_receipt': False}


def native():
    return {'token': 'current', 'started_monotonic': 10.1, 'options': {'time_limit': 1.9}}


class FollowupAnalysisTests(unittest.TestCase):
    def test_timeout_in_full_denominator_not_normal_pairs(self):
        a = archive()
        before = copy.deepcopy(a)
        v = analyze(a, Path('/unused'))
        self.assertEqual(a, before)
        g = v['groups']['all']
        self.assertEqual(g['author']['attempted'], 1)
        self.assertEqual(g['author']['charged_seconds_all_attempted']['sum'], 300.)
        self.assertEqual(g['both_returned_normally_ids'], [])
        self.assertEqual(g['positive_sets_on_registered_inputs']['act_only'], ['mnist_7'])
        self.assertEqual(g['paired_act_minus_author_seconds_all_attempted_pairs']['mean'], -288.)
        self.assertFalse(v['numerical_guarantees_equated'])

    def test_missing_execution_not_zero_cost_pair(self):
        a = archive()
        a['rows'][1] = {'id': 'mnist_7', 'arm': 'author', 'status': 'NOT_STARTED_AFTER_ERROR'}
        g = analyze(a, Path('/unused'))['groups']['all']
        self.assertEqual(g['missing_execution_ids'], ['mnist_7'])
        self.assertEqual(g['both_attempted_ids'], [])
        self.assertIsNone(g['paired_act_minus_author_seconds_all_attempted_pairs']['sum'])

    def test_unknown_is_not_unsafe(self):
        a = archive()
        a['rows'][0]['status'] = 'UNKNOWN'
        g = analyze(a, Path('/unused'))['groups']['all']
        self.assertEqual(g['decided_sets_on_registered_inputs'],
                         {'intersection': [], 'act_only': [], 'author_only': []})

    def test_fallback_base_cost_is_recorded_not_zero(self):
        e = {'queries': [{'scope': {'phase': 'base'}, 'return_elapsed_seconds': 2.9}]}
        self.assertEqual(base_cost(e), 2.9)
        with self.assertRaisesRegex(ValueError, 'base-query'):
            base_cost({'queries': []})

    def test_censored_native_time_not_zero(self):
        q = query_observation(query(), native())
        self.assertIsNone(q['native_seconds_on_return'])
        self.assertIsNone(q['native_status'])
        self.assertAlmostEqual(q['pre_native_seconds'], .1)
        self.assertAlmostEqual(q['remaining_at_native_entry_seconds'], 1.9)

    def test_completed_infeasible_stays_policy_evidence(self):
        q = query()
        q.update(terminal='COMPLETED', returned_status='infeasible', accepted_after_receipt=True,
                 native_result={'token': 'current', 'finished_monotonic': 11.,
                                'status': 2, 'native_seconds': .9, 'candidate_sha256': None})
        v = query_observation(q, native())
        self.assertEqual(v['native_status'], 2)
        self.assertEqual(v['native_seconds_on_return'], .9)
        self.assertFalse(v['native_candidate_recorded'])

    def test_wrong_token_and_impossible_clock_rejected(self):
        for changed in ({'token': 'other'}, {'started_monotonic': 9.}, {'started_monotonic': 12.}):
            n = native()
            n.update(changed)
            with self.assertRaises(ValueError):
                query_observation(query(), n)

    def test_native_start_presence_required(self):
        with self.assertRaises(ValueError):
            query_observation(query(), None)

    def test_late_native_return_not_promoted(self):
        raw = {'token': 'current', 'finished_monotonic': 12.1, 'status': 2, 'native_seconds': 1.99}
        v = query_observation(query(), native(), raw)
        self.assertTrue(v['native_return_recorded'])
        self.assertFalse(v['native_result_in_query_record'])
        self.assertFalse(v['accepted_after_receipt'])
        self.assertFalse(v['native_finish_timestamp_before_deadline'])
        self.assertEqual(v['returned_status'], 'unknown')

    def test_unbound_raw_artifact_rejected(self):
        a = archive()
        a['rows'][0]['details']['experts'] = [{
            'properties': [], 'queries': [{'scope': {'phase': 'expanded', 'row': 0},
                'terminal': 'TIMEOUT', 'return_elapsed_seconds': 1.}],
            'expert_elapsed_before_publication': 1., 'base_seconds': .1, 'native_property_calls': 1}]
        with self.assertRaisesRegex(ValueError, 'unbound'):
            analyze(a, Path('/unused'))


if __name__ == '__main__':
    unittest.main()
