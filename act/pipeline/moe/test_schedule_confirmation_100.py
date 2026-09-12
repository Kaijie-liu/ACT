"""Cohort/denominator extensions only; frozen solver semantics stay unchanged."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from act.pipeline.moe import schedule_confirmation as runner
from act.pipeline.moe import schedule_confirmation_100_selection as selection100
from act.pipeline.moe.schedule_confirmation_selection import source_indices, verify_exclusions


class HundredInputTests(unittest.TestCase):
    def fixture(self):
        config = json.loads(selection100.CONFIG.read_text())
        selection, methods = runner.artifacts(config, deep=True)
        return config, selection, methods

    def test_new_hundred_disjoint_and_nine_hundred_jobs(self):
        cfg, selection, methods = self.fixture()
        self.assertEqual(cfg['sample_count'], 100)
        old = json.loads(selection100.PREVIOUS_SELECTION.read_text())
        chosen = {s['dataset_index'] for s in selection['samples']}
        previous = {s['dataset_index'] for s in old['samples']}
        self.assertEqual(len(chosen), 100)
        self.assertTrue(chosen.isdisjoint(previous))
        self.assertTrue(previous.issubset(selection['excluded_indices']))
        self.assertTrue(chosen.isdisjoint(selection['excluded_indices']))
        self.assertEqual(selection['models'], old['models'])
        self.assertEqual(selection['smoke_samples'], old['smoke_samples'])
        self.assertNotIn('supersedes_selection', selection)
        jobs = runner.jobs(selection, False)
        self.assertEqual(len(jobs), 900)
        self.assertEqual(len(runner.jobs(selection, True)), 9)
        self.assertEqual(len({j['job_id'] for j in jobs}), 900)
        for model in selection['models']:
            for arm in runner.ARMS:
                counts = [sum(j['model']==model and j['method']==arm and j['position']==p for j in jobs)
                          for p in range(3)]
                self.assertEqual(sorted(counts), [33, 33, 34])
        original, oldmethods = runner.artifacts(json.loads(runner.DEFAULT.read_text()))
        self.assertEqual(methods, oldmethods)
        self.assertEqual(cfg['bootstrap'], json.loads(runner.DEFAULT.read_text())['bootstrap'])

    def test_no_resizing_old_config_or_changing_statistical_contract(self):
        cfg, selection, _ = self.fixture()
        for key, value in [('sample_count', 30), ('sample_count', 101), ('budget_seconds', 301),
                           ('classification', 'FROZEN_30_NEW_INPUT_ROUTE_COMPLEXITY_CONFIRMATION'),
                           ('primary_comparator', 'legacy'), ('classification', 'arbitrary')]:
            bad = copy.deepcopy(cfg); bad[key] = value
            with self.assertRaises(ValueError): runner.artifacts(bad)
        old = json.loads(runner.DEFAULT.read_text()); old['sample_count'] = 100
        with self.assertRaises(ValueError): runner.artifacts(old)
        bad = copy.deepcopy(cfg); bad['bootstrap']['seed'] += 1
        with self.assertRaises(ValueError): runner.artifacts(bad)
        # Validation defaults deliberately remain thirty; hundred requires its
        # explicitly registered profile, not an accidental len-based expansion.
        with self.assertRaisesRegex(ValueError, 'non-canonical'): verify_exclusions(selection)

    def test_previous_thirty_exclusion_and_identity_cannot_be_removed(self):
        _, selection, _ = self.fixture()
        old_index = json.loads(selection100.PREVIOUS_SELECTION.read_text())['samples'][0]['dataset_index']
        bad = copy.deepcopy(selection); bad['samples'][0]['dataset_index'] = old_index
        with self.assertRaisesRegex(ValueError, 'overlaps'): selection100.validate_prior(bad)
        bad = copy.deepcopy(selection); bad['excluded_indices'].remove(old_index)
        with self.assertRaisesRegex(ValueError, 'exclude prior'): selection100.validate_prior(bad)
        bad = copy.deepcopy(selection); bad['previous_completed_experiment']['review_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'identity drift'): selection100.validate_prior(bad)
        bad = copy.deepcopy(selection); bad['request']['epsilon'] = 1/255
        with self.assertRaisesRegex(ValueError, 'radius change'): selection100.validate_prior(bad)
        with self.assertRaisesRegex(ValueError, 'no overwrite'): selection100.freeze()
        with self.assertRaisesRegex(ValueError, 'no overwrite'): selection100.register()

    def test_matched_legacy_timeout_rows_excluded_even_without_packages(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            path = Path(tmp)/'rows.jsonl'
            path.write_text('\n'.join(json.dumps({'method': m, 'dataset_index': i,
                'status': 'TIMEOUT', 'package': None}) for m, i in [('matched', 4200), ('legacy', 4201)]))
            self.assertEqual(source_indices(path), {4200, 4201})

    def test_summary_uses_all_hundred_input_clusters_not_old_thirty(self):
        _, selection, _ = self.fixture()
        rows = []; details = {}
        for job in runner.jobs(selection, False):
            rows.append({**job, 'status': 'SAFE' if job['method']=='adaptive' and job['rank']==99 else 'UNKNOWN',
                         'wall_seconds': 300.})
            details[job['model'], job['rank'], job['method']] = {'facts': None, 'pair_count': 2, 'package': False}
        result = runner.summarize(rows, details, selection, False)
        self.assertAlmostEqual(result['input_clustered_contrasts']['matched']['SAFE']['mean_input_cluster_difference'], .01)
        self.assertEqual(result['common_fact_pairs_unavailable'], 300)
        for model in selection['models']:
            contrast = result['models'][model]['contrasts']['matched']['SAFE']
            self.assertEqual(contrast, {'gained': [99], 'lost': [], 'net': 1, 'denominator': 100})
        self.assertTrue(result['scope'].startswith('100 shared'))


if __name__ == '__main__': unittest.main()
