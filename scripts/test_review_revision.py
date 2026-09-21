"""No-solver review-response identities, scope, and witness arithmetic controls."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_bytes())


class Revision(unittest.TestCase):
    def test_complete_manuscript_and_response_identity(self):
        inv = read('docs/review_revision_inventory_20260921_r5.json')
        actual = {str(p.relative_to(ROOT)) for p in (ROOT/'paper').rglob('*')
                  if p.is_file() and p.suffix in ('.md', '.tex')}
        self.assertEqual(actual, {r['path'] for r in inv['current_manuscript']})
        for r in inv['current_manuscript']+inv['current_response_materials']+inv['protected_unchanged_scientific_materials']:
            data = (ROOT/r['path']).read_bytes()
            with self.subTest(path=r['path']):
                self.assertEqual(hashlib.sha256(data).hexdigest(), r['sha256'])
                self.assertEqual(len(data), r['bytes'])

    def test_composed_counts_and_both_directions(self):
        r = read('docs/main_table_input_composition_20260921.json')
        self.assertEqual((r['saved_representatives_decoded'], r['coordinates_checked']), (100, 307200))
        for kind, count in (('rational_epsilon', 257850), ('binary64_epsilon', 197704)):
            s = r['summary'][kind]
            self.assertEqual(s['inward_coordinates'], count)
            self.assertEqual(s['outward_coordinates'], 50512)
            self.assertEqual(s['inputs_with_inward_coordinates'], 100)
            self.assertEqual(s['inputs_with_outward_coordinates'], 100)
            self.assertEqual(s['gain_model_input_pairs_with_coverage_loss'], 23)
            self.assertEqual(s['distinct_gain_inputs_with_coverage_loss'], 21)
            for direction in ('inward', 'outward'):
                self.assertEqual(sum(b['comparisons'][kind][direction+'_coordinates'] for b in r['boxes']), s[direction+'_coordinates'])

    def test_witness_arithmetic_separately_reconstructed(self):
        # Independent of the producer module; checks every stored first witness.
        r = read('docs/main_table_input_composition_20260921.json')
        for b in r['boxes']:
            for kind, stat in b['comparisons'].items():
                eps = Q(2, 255) if kind == 'rational_epsilon' else Q(2/255)
                for direction in ('inward', 'outward'):
                    w = stat['first_'+direction+'_witness']
                    x, lo, hi = (float.fromhex(w[k]) for k in ('center', 'lower', 'upper'))
                    midpoint, radius = Q((lo+hi)/2), Q((hi-lo)/2)
                    if radius <= Q(1e-12): radius = Q(0)
                    left, right = max(Q(0), Q(x)-eps), min(Q(1), Q(x)+eps)
                    signed = (midpoint-radius-left, right-midpoint-radius)
                    if direction == 'outward': signed = tuple(-v for v in signed)
                    gaps = tuple(max(Q(0), v) for v in signed)
                    self.assertEqual(gaps, (Q(w['lower_gap_exact']), Q(w['upper_gap_exact'])))
                    self.assertGreater(max(gaps), 0)

    def test_parent_unchanged_and_no_new_execution_claim(self):
        r = read('docs/main_table_input_composition_20260921.json')
        self.assertEqual(hashlib.sha256((ROOT/r['parent']['path']).read_bytes()).hexdigest(), r['parent']['sha256'])
        for k in ('new_solver_calls', 'new_model_forwards', 'new_source_propagations', 'new_output_bounds'):
            self.assertEqual(r[k], 0)
        self.assertFalse(r['source_complete_positive_proof'])
        self.assertEqual(r['input98_followup'], 'STOP_INPUT98_FOLLOWUP')

    def test_primitive_risks_not_claimed_resolved(self):
        text = (ROOT/'paper/sections/05_soundness_engineering.md').read_text()
        for phrase in ('No independently verified propagation', 'not nested',
                       'other historical cohorts', 'not 181 independently', 'Counts are not cost'):
            haystack = text.replace('\n', ' ')
            if phrase == 'Counts are not cost':
                haystack = (ROOT/'docs/external_ai_review_response_20260921.md').read_text()
            self.assertIn(phrase, haystack)

    def test_unqualified_positive_count_phrases_removed_from_active_text(self):
        files = list((ROOT/'paper/sections').glob('*.md')) + [ROOT/'paper/baseline_delta_table.md', ROOT/'paper/claims_to_evidence.md']
        bad = (r'36 route-changing certificates', r'31 additional certificates',
               r'36 route-changing SAFE certificates', r'additional safe certificates')
        for p in files:
            for expression in bad:
                with self.subTest(file=str(p), phrase=expression):
                    self.assertIsNone(re.search(expression, p.read_text()))

    def test_portable_check_not_upgraded_or_released(self):
        r = read('docs/reviewer_artifact_readiness_20260921.json')
        old = read('docs/portable_conv_proof_v1_review.json')
        self.assertEqual(r['bundle_sha256'], old['bundle_sha256'])
        self.assertEqual(r['statement_sha256'], old['statement_sha256'])
        self.assertEqual(sum(f['bytes'] for f in r['files']), 7181520)
        self.assertEqual((r['positive_obligations'], r['required_obligations']), (9, 9))
        self.assertEqual(r['minimum_lower_bound'], old['minimum_lower_bound'])
        self.assertTrue(r['complete_result_matches_old'])
        for k in ('proof_regenerated', 'source_complete_positive_proof', 'deployed_float_SAFE',
                  'external_release_performed', 'clean_environment_install_tested', 'weights_or_dataset_distributed'):
            self.assertFalse(r[k])

    def test_response_and_manuscript_local_links(self):
        inv = read('docs/review_revision_inventory_20260921_r5.json')
        paths = [r['path'] for r in inv['current_response_materials'] if r['path'].endswith('.md')]
        paths += [r['path'] for r in inv['current_manuscript'] if r['path'].endswith('.md')
                  and not r['path'].startswith('paper/appendices/')]
        # Only Markdown links; plain-text historical path mentions aren't access claims.
        for name in paths:
            p = ROOT/name
            for link in re.findall(r'\]\(([^\s)]+)\)', p.read_text()):
                if ':' in link or link.startswith('#'): continue
                with self.subTest(source=name, link=link):
                    self.assertTrue((p.parent/link.split('#')[0]).resolve().is_file())


if __name__ == '__main__':
    unittest.main()
