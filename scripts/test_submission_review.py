"""Review packaging/accounting controls; no numerical libraries or solver calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


builder = module('build_submission_review_kit')
checker = module('check_submission_review_kit')
tables = module('render_submission_tables')


class SubmissionReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='submission-review-control-', dir=ROOT.parent)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.kit = self.base / 'kit'
        self.record = builder.build(ROOT, self.kit)
        self.digest = self.record['manifest_sha256']

    def manifest(self):
        return json.loads((self.kit / 'manifest.json').read_bytes())

    def rewrite_manifest(self, value):
        data = json.dumps(value, sort_keys=True).encode()
        (self.kit / 'manifest.json').write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def test_accounting_rendering_and_grades(self):
        result = checker.check(self.kit, self.digest)
        self.assertEqual(result['table_rows'], 21)
        self.assertEqual(result['additional_metamoe_rows'], 2)
        self.assertEqual(result['evidence_grade'], 'ARCHIVED_ACCOUNTING_ONLY')
        self.assertEqual(result['limitations'], builder.LIMITS)
        self.assertEqual(tables.render(), (ROOT / tables.OUTPUT).read_text())
        for phrase in ('59 / 89', '57 / 81', '63 / 86', '138.11', '4.23'):
            self.assertIn(phrase, tables.render())
        for phrase in ('not source-complete strict certificates',
                       'All 23 primary gains', 'No ACT-only positive',
                       'ACT & HZ & 10 & 4 & 0 & 6 & 0 & 0 & 28.56',
                       'Repaired author & AF & 10 & 9 & 0 & 0 & 1 & 0 & 36.54'):
            self.assertIn(phrase, tables.render())

    def test_metamoe_missing_duplicate_grade_and_cost_rejected(self):
        original = json.loads((ROOT / tables.METAMOE_ARCHIVE).read_bytes())
        rows = tables.metamoe_accounting(original)
        self.assertEqual([(r['positive'], r['unknown'], r['timeout']) for r in rows],
                         [(4, 6, 0), (9, 0, 1)])
        mutations = [lambda d: d['rows'].pop(),
                     lambda d: d['rows'].__setitem__(0, copy.deepcopy(d['rows'][2])),
                     lambda d: d['rows'][0].update(grade='FORMAL_SAFE'),
                     lambda d: d['rows'][0].update(seconds=None),
                     lambda d: d['rows'][0].update(seconds=float('nan')),
                     lambda d: d['rows'][0].update(status='ERROR'),
                     lambda d: d.update(act_only_positive=['cifar10_1']),
                     lambda d: d.update(numerical_guarantees_equated=True)]
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            with self.assertRaises(ValueError):
                tables.metamoe_accounting(changed)

    def test_isolated_relocated_cli_without_git_or_models(self):
        moved = self.base / 'relocated'
        shutil.copytree(self.kit, moved)
        # Remove the original COPY, not any project evidence.
        shutil.rmtree(self.kit)
        result = subprocess.run([sys.executable, '-I', '-S',
                                 str(moved / 'scripts/check_submission_review_kit.py'),
                                 '--manifest-sha256', self.digest], cwd=self.base,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['evidence_grade'], 'ARCHIVED_ACCOUNTING_ONLY')
        self.assertFalse((moved / '.git').exists())
        self.assertFalse(list(moved.rglob('__pycache__')))
        self.assertFalse(any(p.suffix in ('.pt', '.pth', '.onnx', '.npz', '.zip') for p in moved.rglob('*')))

    def test_external_hash_and_missing_or_changed_files(self):
        with self.assertRaisesRegex(ValueError, 'manifest identity'):
            checker.check(self.kit, '0' * 64)
        path = self.kit / 'paper/review_main.tex'
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with self.assertRaises(ValueError):
            checker.check(self.kit, self.digest)
        path.unlink()
        with self.assertRaises(ValueError):
            checker.check(self.kit, self.digest)

    def test_extra_file_and_symlink_rejected(self):
        extra = self.kit / 'extra'
        extra.write_text('not in inventory')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            checker.check(self.kit, self.digest)
        extra.unlink()
        extra.symlink_to(self.base / 'absent')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            checker.check(self.kit, self.digest)

    def test_unsafe_duplicate_paths_and_sizes_rejected_even_if_rehashed(self):
        original = self.manifest()
        for path in ('../outside', '/etc/passwd', './paper/review_main.tex', 'paper//review_main.tex',
                     'paper\\review_main.tex', 'manifest.json'):
            with self.subTest(path=path):
                value = copy.deepcopy(original)
                value['files'][0]['path'] = path
                digest = self.rewrite_manifest(value)
                with self.assertRaises(ValueError):
                    checker.check(self.kit, digest)
        for modification in ('duplicate', 'size', 'digest'):
            value = copy.deepcopy(original)
            if modification == 'duplicate':
                value['files'].append(value['files'][0])
            elif modification == 'size':
                value['files'][0]['bytes'] = -1
            else:
                value['files'][0]['sha256'] = 'not a hash'
            with self.assertRaises(ValueError):
                checker.check(self.kit, self.rewrite_manifest(value))

    def test_guarantee_upgrades_rejected(self):
        original = self.manifest()
        changes = [{'evidence_grade': 'FORMAL_SAFE'}, {'schema': 'MODEL_PROOF'},
                   {'limitations': original['limitations'] | {'independent_network_reproof': True}},
                   {'limitations': original['limitations'] | {'human_review_completed': True}}]
        for change in changes:
            with self.assertRaises(ValueError):
                checker.check(self.kit, self.rewrite_manifest(original | change))

    def test_invalid_accounting_rejected_after_hashes_updated(self):
        manifest = self.manifest()
        name = 'act/pipeline/moe/results/schedule_confirmation_100_review_20260914_r1.json'
        path = self.kit / name
        content = json.loads(path.read_bytes())
        content['full']['audit']['models']['seed0']['methods']['adaptive']['states']['SAFE'] += 1
        data = json.dumps(content).encode()
        path.write_bytes(data)
        record, = [r for r in manifest['files'] if r['path'] == name]
        record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        with self.assertRaisesRegex(ValueError, 'denominator'):
            checker.check(self.kit, self.rewrite_manifest(manifest))

    def test_duplicate_json_keys_rejected(self):
        data = (self.kit / 'manifest.json').read_bytes()
        data = data[:-2] + b', "schema": "MOE_SUBMISSION_REVIEW_KIT_V1"}'
        (self.kit / 'manifest.json').write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            checker.check(self.kit, hashlib.sha256(data).hexdigest())

    def test_existing_output_not_overwritten(self):
        before = (self.kit / 'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            builder.build(ROOT, self.kit)
        self.assertEqual(before, (self.kit / 'manifest.json').read_bytes())

    def test_paper_keeps_major_adverse_results_and_conditional_theorem(self):
        paper = (ROOT / 'paper/review_main.tex').read_text()
        for phrase in ('Conditional complete-output composition', 'not nested',
                       '179, 156 and 141', 'zero complete positive',
                       '13 numerical positive filters', '138.11',
                       'not an exact', 'human independent method review remain open',
                       'four ACT policy acceptances versus nine author',
                       'zero ACT-only', 'compatibility is improved, but no new certificate',
                       'Corrected construction could change both route coverage and outcomes'):
            self.assertIn(phrase, paper.replace('\n', ' '))
        self.assertNotIn('\\write18', paper)
        import re
        self.assertEqual(set(re.findall(r'\\cite\{([^}]+)\}', paper)),
                         set(re.findall(r'\\bibitem\{([^}]+)\}', paper)))


if __name__ == '__main__':
    unittest.main()
