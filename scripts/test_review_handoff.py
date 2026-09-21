"""Read-only handoff bindings and links; no ACT, model, solver or writes."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'docs/external_ai_review_manifest_20260921.json'
DOCUMENTS = ('docs/EXTERNAL_AI_REVIEW.md', 'docs/EXTERNAL_AI_REVIEW_PROMPT.md')
BASELINE = '806443470b3eeda3e601510ea8a6fc71570f96bf'


def verify_record(record):
    p = (ROOT/record['path']).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise ValueError('missing/outside review material')
    # This manifest is a historical snapshot, not a ban on later manuscript
    # revisions. Check its original Git objects; current files are separately
    # hash-bound by review_revision_inventory_20260921_r2.json.
    try:
        b = subprocess.check_output(['git', 'show', f'{BASELINE}:{record["path"]}'],
                                    cwd=ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise ValueError('missing baseline review material') from exc
    if len(b) != record['bytes'] or hashlib.sha256(b).hexdigest() != record['sha256']:
        raise ValueError('review baseline drift')


class ReviewHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.loads(MANIFEST.read_bytes())
        cls.records = [r for group in cls.m['files'].values() for r in group]

    def test_unique_materials_match_scientific_baseline(self):
        self.assertEqual(self.m['scientific_baseline_commit'],
                         '806443470b3eeda3e601510ea8a6fc71570f96bf')
        self.assertEqual(len(self.records), self.m['file_count'])
        self.assertEqual(len({r['path'] for r in self.records}), len(self.records))
        for r in self.records:
            with self.subTest(path=r['path']): verify_record(r)

    def test_corrupt_hash_missing_path_or_size_rejected(self):
        r = self.records[0]
        for change in ({'sha256': '0'*64}, {'path': 'docs/NO_REVIEW_MATERIAL'},
                       {'bytes': r['bytes']+1}, {'path': '/dev/null'}):
            with self.assertRaises(ValueError): verify_record(r | change)

    def test_handoff_links_exist(self):
        for name in DOCUMENTS:
            p = ROOT/name
            for link in re.findall(r'\]\(([^\s)]+)\)', p.read_text()):
                if ':' in link or link.startswith('#'): continue
                target = (p.parent/link.split('#')[0]).resolve()
                with self.subTest(source=name, target=str(target)):
                    self.assertTrue(target.is_file())

    def test_review_does_not_authorize_new_experiment_or_claim_completion(self):
        self.assertFalse(self.m['review_completed'])
        self.assertFalse(self.m['third_party_human_review_completed'])
        self.assertFalse(self.m['raw_data_and_checkpoints_bundled'])
        for k, allowed in self.m['permissions'].items():
            self.assertEqual(allowed, k in ('read_source_and_saved_results', 'no_solver_readonly_checks'))
        self.assertEqual(self.m['scientific_constraints']['input98'], 'STOP_INPUT98_FOLLOWUP')
        self.assertFalse(self.m['scientific_constraints']['complete_source_positive_claim'])


if __name__ == '__main__':
    unittest.main()
