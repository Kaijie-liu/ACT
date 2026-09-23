"""Accounting controls: no execution or native backend required."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_metamoe_checked_small import summarize, timing


def stopped_archive():
    # No normal ACT records: these tests exercise missing/error accounting only.
    rows = [
        {'id': 'mnist_7', 'arm': 'act', 'status': 'NOT_STARTED_AFTER_ERROR'},
        {'id': 'mnist_7', 'arm': 'author', 'status': 'ERROR', 'seconds': 250.,
         'result': {}, 'receipt': {'peak_sampled_group_rss_bytes': 10}},
        {'id': 'mnist_9', 'arm': 'act', 'status': 'NOT_STARTED_AFTER_ERROR'},
        {'id': 'mnist_9', 'arm': 'author', 'status': 'NOT_STARTED_AFTER_ERROR'},
    ]
    return {'audit': 'PASS', 'issues': 0, 'rows': rows, 'cost': {}}


class SummaryTests(unittest.TestCase):
    def test_missing_is_not_zero_or_negative(self):
        v = summarize(stopped_archive())['groups']['all']
        self.assertEqual(v['act']['registered_denominator'], 2)
        self.assertEqual(v['act']['attempted'], 0)
        self.assertIsNone(v['act']['charged_seconds_all_attempted']['mean'])
        self.assertEqual(v['author']['charged_seconds_all_attempted']['mean'], 250.)
        self.assertEqual(v['positive_sets_on_normal_pairs']['author_only'], [])
        self.assertEqual(v['not_comparable_ids'], ['mnist_7', 'mnist_9'])

    def test_unattempted_cost_refused(self):
        v = stopped_archive()
        v['rows'][0]['seconds'] = 0.
        with self.assertRaisesRegex(ValueError, 'unexecuted'):
            summarize(v)

    def test_missing_attempted_cost_refused(self):
        v = stopped_archive()
        del v['rows'][1]['seconds']
        with self.assertRaisesRegex(ValueError, 'attempted-request'):
            summarize(v)

    def test_roster_loss_or_duplication_refused(self):
        for mutation in ('loss', 'duplication'):
            v = stopped_archive()
            if mutation == 'loss':
                v['rows'].pop()
            else:
                v['rows'].append(copy.deepcopy(v['rows'][0]))
            with self.assertRaisesRegex(ValueError, 'roster'):
                summarize(v)

    def test_unaccepted_archive_refused(self):
        v = stopped_archive()
        v['issues'] = 1
        with self.assertRaisesRegex(ValueError, 'audit'):
            summarize(v)

    def test_empty_and_nonempty_time(self):
        self.assertEqual(timing([]), {'n': 0, 'sum': None, 'mean': None, 'median': None})
        self.assertEqual(timing([2., 5.])['mean'], 3.5)

    def test_paired_positive_grades_and_cost_direction(self):
        v = stopped_archive()
        v['rows'] = [
            {'id': 'cifar10_1', 'arm': 'act', 'status': 'UNKNOWN', 'seconds': 20.,
             'result': {'candidates': [0], 'unresolved': []},
             'details': {'experts': [], 'nonzero': []},
             'receipt': {'peak_sampled_group_rss_bytes': 10}},
            {'id': 'cifar10_1', 'arm': 'author', 'status': 'BACKEND_POSITIVE', 'seconds': 8.,
             'grade': 'AUTHOR_BACKEND_NUMERICAL_SUFFICIENT_FILTER', 'result': {},
             'receipt': {'peak_sampled_group_rss_bytes': 10}},
        ]
        s = summarize(v)
        self.assertFalse(s['numerical_guarantees_equated'])
        g = s['groups']['all']
        self.assertEqual(g['positive_sets_on_normal_pairs'],
                         {'intersection': [], 'act_only': [], 'author_only': ['cifar10_1']})
        self.assertEqual(g['paired_act_minus_author_seconds_normal_pairs']['mean'], 12.)


if __name__ == '__main__':
    unittest.main()
