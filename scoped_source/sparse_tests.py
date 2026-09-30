"""Complete synthetic graph proofs and fail-closed H1 mutations, no real data."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from act.back_end.solver.lp_certificate import propose
from scoped_source.sparse_build import build
from scoped_source.sparse_check import check
from scoped_source.sparse_controls import source, exact_routes
from source_enclosure.format import identity

ROOT = Path(__file__).resolve().parents[1]


def produce(doc, **kwargs):
    return build(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+300, **kwargs)


def verify(doc, package):
    return check(doc, package, expected_source_sha256=identity(doc), deadline=time.monotonic()+300)


class SparseSourceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = source(); cls.package = produce(cls.doc, proposer=propose)

    def test_complete_source_and_all_output_lp_proof(self):
        result = verify(self.doc, self.package)
        self.assertEqual(result['status'], 'CHECKED_DECLARED_SOURCE_POSITIVE')
        self.assertEqual((result['required'], result['lp_bounds_checked'], result['positive']), (6, 6, 6))
        self.assertFalse(result['deployed_float_SAFE'])
        self.assertLess(result['source_blocks_checked'], result['source_nodes'])

    def test_exact_distinct_route_witnesses(self):
        left = exact_routes(self.doc, [-1, 0, 0]); right = exact_routes(self.doc, [1, 0, 0])
        self.assertEqual(left['legal_pairs'], [[1, 2]])
        self.assertEqual(right['legal_pairs'], [[0, 2]])
        self.assertFalse(verify(self.doc, self.package)['route_change_witness_checked'])

    def test_relations_needed_beyond_interval_facts(self):
        doc = source(relational=True)
        package = produce(doc, proposer=propose, reuse_keys=[((0, 1), 1), ((1, 2), 1)])
        result = verify(doc, package)
        self.assertEqual(result['status'], 'CHECKED_DECLARED_SOURCE_POSITIVE')
        self.assertEqual(result['reused'], 1)  # expert 0's interval fact is -1/4 - margin
        self.assertEqual(package['obligations'][0]['kind'], 'weighted')
        omitted = [k for k in package['bank'] if '/layer/' in k]
        weak = verify(doc, produce(doc, proposer=propose, omit_nodes=omitted))
        self.assertEqual(weak['status'], 'UNKNOWN_NONPOSITIVE')
        self.assertLess(weak['positive'], result['positive'])

    def test_guard_only_source_coordinate_is_required(self):
        doc = source(relational=True, router_coordinate=2)
        package = produce(doc, proposer=propose)
        self.assertIn('input/2', package['bank'])
        verify(doc, package)
        package['bank'].pop('input/2')
        with self.assertRaises(ValueError): verify(doc, package)

    def test_exact_nonzero_mixed_residual(self):
        from scoped_source.sparse_ir import row, lp_record
        from scoped_source.sparse_check import check_bound
        lp = lp_record(['x', 'y'], {'x': (F(-1), F(1)), 'y': (F(0), F(2))},
                       [row('le', {'x': F(1), 'y': F(1)}, F(1)),
                        row('eq', {'x': F(1), 'y': F(-1)}, F(0))],
                       {'x': F(3), 'y': F(-5)}, F(1, 2))
        certificate = {'lp_sha256': identity(lp), 'inequality_dual': [-1], 'equality_dual': [2],
                       'claimed_lower_bound': '-13/2'}
        self.assertEqual(check_bound(lp, certificate)['checked_lower_bound'], '-13/2')
        certificate['claimed_lower_bound'] = '-649/100'
        with self.assertRaises(ValueError): check_bound(lp, certificate)

    def test_proposal_exception_preserves_all_missing_obligations(self):
        for exception in (ValueError, RuntimeError):
            def broken(*args, **kwargs): raise exception('controlled proposal failure')
            package = produce(self.doc, proposer=broken)
            result = verify(self.doc, package)
            self.assertEqual(result['status'], 'UNKNOWN_MISSING_EVIDENCE')
            self.assertEqual(result['required'], 6)
            self.assertTrue(all(v['proposal_error'] for v in package['obligations']))

    def test_all_tie_legal_pairs_and_changed_dimensions(self):
        for e, c, w in [(2, 2, 1), (4, 4, 2)]:
            doc = source(experts=e, classes=c, width=w, tied=True, constant=True)
            result = verify(doc, produce(doc, proposer=propose))
            self.assertEqual(result['required'], e*(e-1)//2*(c-1))
            self.assertEqual(result['positive'], result['required'])

    def test_partial_scoped_interval_reuse(self):
        package = produce(self.doc, proposer=propose, reuse_keys=[((0, 1), 1)])
        result = verify(self.doc, package)
        self.assertEqual((result['reused'], result['lp_bounds_checked']), (1, 5))
        self.assertEqual(result['positive'], 6)

    def test_same_row_universe_differential_and_constructor_saving(self):
        full = produce(self.doc, proposer=propose, mode='full')
        sliced = verify(self.doc, self.package); eager = verify(self.doc, full)
        self.assertEqual([r['lower_bound'] for r in sliced['obligations']],
                         [r['lower_bound'] for r in eager['obligations']])
        self.assertLess(self.package['stats']['constructed_blocks'], full['stats']['constructed_blocks'])
        self.assertLess(self.package['stats']['materialized_lp_rows'], full['stats']['materialized_lp_rows'])
        self.assertFalse(any(k.startswith('expert2/') for k in full['obligations'][0]['variables']))

    def test_dropped_constraints_are_relaxation_not_missing_obligations(self):
        omitted = [k for k in self.package['bank'] if '/layer/' in k]
        package = produce(self.doc, omit_nodes=omitted, proposer=propose)
        result = verify(self.doc, package)
        self.assertEqual(result['required'], 6)
        for old, new in zip(verify(self.doc, self.package)['obligations'], result['obligations']):
            self.assertLessEqual(F(new['lower_bound']), F(old['lower_bound']))

    def test_unsafe_source_negative_bounds_are_not_unsafe_verdict(self):
        doc = source(unsafe=True, tied=True); result = verify(doc, produce(doc, proposer=propose))
        self.assertEqual(result['status'], 'UNKNOWN_NONPOSITIVE')
        self.assertEqual(result['positive'], 0)

    def test_stable_zero_and_unstable_relu_ranges(self):
        for bias in (-2., 0., 2.):
            doc = source(relu_bias=bias); package = produce(doc, proposer=propose)
            self.assertEqual(verify(doc, package)['positive'], 6)

    def test_missing_duplicate_property_and_pair_fail_closed(self):
        for kind in ('missing', 'duplicate', 'wrong_property', 'reverse_pair', 'boolean'):
            bad = copy.deepcopy(self.package)
            if kind == 'missing': bad['obligations'].pop()
            if kind == 'duplicate': bad['obligations'][-1] = bad['obligations'][0]
            if kind == 'wrong_property': bad['obligations'][0]['competitor'] = 0
            if kind == 'reverse_pair': bad['obligations'][0]['pair'].reverse()
            if kind == 'boolean': bad['obligations'][0]['competitor'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): verify(self.doc, bad)

    def test_source_domain_property_and_model_binding(self):
        for key, value in [('radius', '1/2'), ('label', 1), ('margin', '0')]:
            changed = copy.deepcopy(self.doc); changed['request'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): verify(changed, self.package)
        with self.assertRaises(ValueError):
            check(self.doc, self.package, expected_source_sha256='0'*64, deadline=time.monotonic()+300)

    def test_range_row_identity_and_dependency_mutations(self):
        affine = next(k for k, b in self.package['bank'].items() if '/layer/0/' in k)
        relu = next(k for k, b in self.package['bank'].items() if '/layer/1/' in k)
        for kind in ('inward_input', 'bound_missing', 'coefficient', 'relu', 'cycle', 'alias', 'extra'):
            bad = copy.deepcopy(self.package)
            if kind == 'inward_input': bad['bank']['input/0']['bounds'][0] = '0'
            if kind == 'bound_missing': bad['bank'][affine].pop('bounds')
            if kind == 'coefficient': bad['bank'][affine]['rows'][0]['rhs'] = '1'
            if kind == 'relu': bad['bank'][relu]['rows'][2]['rhs'] = '-1'
            if kind == 'cycle': bad['bank'][affine]['parents'] = [affine]
            if kind == 'alias': bad['obligations'][0]['variables'][0] = bad['obligations'][0]['variables'][1]
            if kind == 'extra': bad['bank']['foreign'] = copy.deepcopy(bad['bank'][affine])
            with self.subTest(kind=kind), self.assertRaises(ValueError): verify(self.doc, bad)

    def test_product_range_row_inventory_and_stale_dual_binding(self):
        for kind in ('range', 'rows', 'hash', 'claim', 'sign', 'stale'):
            bad = copy.deepcopy(self.package); item = bad['obligations'][0]
            if kind == 'range': item['difference'] = ['0', '0']
            if kind == 'rows': item['blocks'].pop()
            if kind == 'hash': item['lp_sha256'] = '0'*64
            if kind == 'claim': item['certificate']['claimed_lower_bound'] = '1000000'
            if kind == 'sign': item['certificate']['inequality_dual'][0] = 1
            if kind == 'stale': item['certificate'] = bad['obligations'][-1]['certificate']
            with self.subTest(kind=kind), self.assertRaises(ValueError): verify(self.doc, bad)

    def test_missing_candidate_remains_unknown(self):
        package = produce(self.doc)
        result = verify(self.doc, package)
        self.assertEqual(result['status'], 'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(result['positive'], 0)

    def test_bad_source_fact_cannot_skip_solver(self):
        package = produce(self.doc, reuse_keys=[((0, 1), 1)], proposer=propose)
        for kind in ('bound', 'property', 'source'):
            bad = copy.deepcopy(package)
            if kind == 'bound': bad['obligations'][0]['facts'][0] = '100'
            if kind == 'property': bad['obligations'][0]['competitor'] = 2
            if kind == 'source': bad['source_sha256'] = '0'*64
            with self.subTest(kind=kind), self.assertRaises(ValueError): verify(self.doc, bad)

    def test_no_generator_or_solver_in_checker(self):
        with patch('scoped_source.sparse_build.build', side_effect=AssertionError('producer')):
            self.assertEqual(verify(self.doc, self.package)['positive'], 6)

    def test_cooperative_deadline_and_late_candidate(self):
        with self.assertRaises(TimeoutError):
            check(self.doc, self.package, expected_source_sha256=identity(self.doc), deadline=time.monotonic()-1)
        from scoped_source import sparse_check
        native = sparse_check.check_bound
        with patch('scoped_source.graph.time.monotonic', return_value=10) as now:
            def late(*args):
                result = native(*args); now.return_value = 12; return result
            with patch('scoped_source.sparse_check.check_bound', side_effect=late), self.assertRaises(TimeoutError):
                check(self.doc, self.package, expected_source_sha256=identity(self.doc), deadline=11)

    def test_relocation_and_standard_library_check(self):
        with tempfile.TemporaryDirectory(prefix='h1-check-', dir=ROOT) as temporary:
            path = Path(temporary)/'moved.json'
            path.write_text(json.dumps({'source': self.doc, 'proof': self.package}))
            code = '''import sys,json,time
sys.path.insert(0,sys.argv[1])
def audit(event,args):
    if event=='import' and (args[0].split('.')[0] in ('numpy','scipy','torch','highspy','gurobipy') or args[0]=='scoped_source.sparse_build'):
        raise RuntimeError('forbidden checker import')
sys.addaudithook(audit)
from scoped_source.sparse_check import check
from source_enclosure.format import identity
p=json.load(open(sys.argv[2]))
print(check(p['source'],p['proof'],expected_source_sha256=sys.argv[3],deadline=time.monotonic()+300)['status'])
'''
            output = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(ROOT), str(path), identity(self.doc)],
                                    check=True, capture_output=True, text=True, cwd=temporary, timeout=30)
            self.assertIn('CHECKED_DECLARED_SOURCE_POSITIVE', output.stdout)


if __name__ == '__main__': unittest.main()
