"""Completion accounting controls; no author environments, models, or solvers."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('baseline_readiness', ROOT/'scripts/review_baseline_readiness.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class BaselineReadiness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='baseline-ledger-', dir=ROOT.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in set(review.SOURCES.values()) | set(review.CONFIGS.values()) | {review.OUTPUT}:
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)

    def mutate(self, source, change):
        path = self.root/review.SOURCES[source]
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_real_archive_and_conservative_summary(self):
        result = review.check(self.root)
        self.assertEqual(result['status'], 'ARCHIVED_ACCOUNTING_PASS')
        self.assertEqual(result['summary']['works_with_real_execution'], 4)
        self.assertEqual(result['summary']['formal_cohort_act_comparisons_completed'], 0)
        self.assertEqual(result['summary']['same_object_act_smoke_works'], 1)
        self.assertTrue(result['not_submission_ready_judgment'])

    def test_dependency_free_cli(self):
        result = subprocess.run([sys.executable, '-I', '-S',
                                 str(ROOT/'scripts/review_baseline_readiness.py'), '--check'],
                                cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ARCHIVED_ACCOUNTING_PASS', result.stdout)

    def test_missing_source_rejects(self):
        (self.root/review.SOURCES['dual']).unlink()
        with self.assertRaises(FileNotFoundError):
            review.check(self.root)

    def test_final_training_checkpoint_must_match_certification(self):
        self.mutate('dual_train', lambda d: d['final_epoch'].update(file_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'checkpoint binding'):
            review.rebuild(self.root)

    def test_diagnostic_bound_to_the_smoke(self):
        self.mutate('meta_capacity', lambda d: d['result'].update(config_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'diagnostic config binding'):
            review.rebuild(self.root)

    def test_rome_interpretation_requires_all_rows(self):
        self.mutate('rome_interpretation', lambda d: d.update(rows=[]))
        with self.assertRaisesRegex(ValueError, 'unique request roster'):
            review.rebuild(self.root)

    def test_rome_duplicate_row_is_not_coverage(self):
        self.mutate('rome_interpretation', lambda d: d['rows'].__setitem__(0, d['rows'][1]))
        with self.assertRaisesRegex(ValueError, 'unique request roster'):
            review.rebuild(self.root)

    def test_rome_unchanged_counts_do_not_hide_wrong_interpretation(self):
        def swap(d):
            d['rows'][0]['interpretation'], d['rows'][3]['interpretation'] = (
                d['rows'][3]['interpretation'], d['rows'][0]['interpretation'])
        self.mutate('rome_interpretation', swap)
        with self.assertRaisesRegex(ValueError, 'per-row interpretation'):
            review.rebuild(self.root)

    def test_config_changed_without_rebinding_rejects(self):
        path = self.root/review.CONFIGS['meta']
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'config binding'):
            review.rebuild(self.root)

    def test_digest_detects_semantically_unmodified_source_change(self):
        path = self.root/review.SOURCES['dual']
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'stale or upgraded'):
            review.check(self.root)

    def test_audit_pass_cannot_open_meta_gate(self):
        self.mutate('meta', lambda d: d.update(execution_control_pass=True))
        with self.assertRaisesRegex(ValueError, 'not an execution gate'):
            review.rebuild(self.root)

    def test_missing_meta_arm_not_a_complete_smoke(self):
        self.mutate('meta', lambda d: d['rows'].pop())
        with self.assertRaisesRegex(ValueError, 'smoke roster'):
            review.rebuild(self.root)

    def test_smoothing_not_deterministic_safe(self):
        self.mutate('dual', lambda d: d.update(deterministic_formal_SAFE=True))
        with self.assertRaisesRegex(ValueError, 'guarantee upgrade'):
            review.rebuild(self.root)

    def test_training_aggregate_not_prediction_replay(self):
        self.mutate('robust', lambda d: d.update(prediction_replay=True))
        with self.assertRaisesRegex(ValueError, 'not attack replay'):
            review.rebuild(self.root)

    def test_partial_test_denominator_rejects(self):
        self.mutate('robust', lambda d: d['arms'][0]['evaluations']['PGD20'].update(examples=9999))
        with self.assertRaisesRegex(ValueError, 'accuracy denominator'):
            review.rebuild(self.root)

    def test_native_top2_cannot_be_relabelled_ste(self):
        self.mutate('robust_semantics', lambda d: d.update(top2_effective_ste=True))
        with self.assertRaisesRegex(ValueError, 'effective source'):
            review.rebuild(self.root)

    def test_rome_timeouts_stay_in_denominator(self):
        self.mutate('rome', lambda d: d['counts'].update(TIMEOUT=0))
        with self.assertRaisesRegex(ValueError, 'terminal denominator'):
            review.rebuild(self.root)

    def test_report_cannot_upgrade_scientific_or_execution_status(self):
        report = review.rebuild(self.root)
        for field, value in [('entire_paper_reproduction_completed', True),
                             ('next_execution_frozen', True),
                             ('act_comparison', 'FORMAL_COHORT_COMPLETED')]:
            changed = copy.deepcopy(report)
            changed['rows'][1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'stale or upgraded'):
                review.check(self.root, changed)

    def test_strict_json_and_output_nonoverwrite(self):
        for text in ['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                review.strict_json(text)
        target = self.root/'existing.json'
        target.write_text('preserve')
        run = subprocess.run([sys.executable, '-I', '-S',
                              str(ROOT/'scripts/review_baseline_readiness.py'), '--output', str(target)],
                             capture_output=True, text=True, timeout=15)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(target.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
