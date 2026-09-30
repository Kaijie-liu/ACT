"""H2 declared-source admission and independent-check controls, not timing tests."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from scoped_source.endpoint_source_build import build, source_request, mc_lp, common_certificate
from scoped_source.endpoint_source_check import check, reconstruct, reconstruct_mc
from scoped_source.endpoint_source_controls import cases, protocol, weighted_source, weighted_negative_point
from scoped_source.endpoint_source_audit import audit_case, point_value
from scoped_source.sparse_controls import source, exact_routes
from source_enclosure.format import identity
from upstream_source.checker import csr


def produce(doc, mode='endpoints', reuse=(), proposer=None):
    return build(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+300,
                 mode=mode, reuse_keys=reuse, proposer=proposer)


def verify(doc, package):
    return check(doc, package, expected_source_sha256=identity(doc),
                 expected_mode=package['mode'], deadline=time.monotonic()+300)


class SourceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from act.back_end.solver.lp_certificate import propose
        cls.propose = staticmethod(propose)
        cls.examples = {}
        for name, doc, reuse in cases():
            cls.examples[name] = (doc, {mode: produce(doc, mode, reuse, propose)
                                       for mode in ('endpoints', 'mccormick')})

    def test_same_source_control_results(self):
        expected = {'weighted_sign': (3, 2, 3), 'tied_partial_reuse': (18, 18, 18),
                    'unsafe_tied': (0, 0, 6), 'unresolved_sign': (6, 6, 6)}
        for name, (doc, arms) in self.examples.items():
            for i, mode in enumerate(('endpoints', 'mccormick')):
                with self.subTest(name=name, mode=mode):
                    result = verify(doc, arms[mode])
                    self.assertEqual((result['positive'], result['required']), (expected[name][i], expected[name][2]))
                    self.assertEqual(arms[mode]['proposal_errors'], [])
                    self.assertFalse(result['deployed_float_SAFE'])
                    self.assertFalse(result['hard_budget_supervision'])

    def test_binary64_source_separation_and_legal_route_witnesses(self):
        doc, arms = self.examples['weighted_sign']
        point = weighted_negative_point(doc, arms['mccormick'])
        self.assertEqual(F(point['checked_objective']), F(-225179981368525, 9007199254740992))
        self.assertNotEqual(F(point['checked_objective']), F(-1, 40))
        self.assertFalse(point['network_counterexample'])
        self.assertNotEqual(exact_routes(doc, [-1]), exact_routes(doc, [1]))
        self.assertEqual(len(exact_routes(doc, [F(-1, 2)])['legal_pairs']), 2)

    def test_independent_source_and_product_constructors_agree(self):
        for doc, arms in self.examples.values():
            expected = reconstruct(doc, identity(doc), lambda: None)
            self.assertEqual(source_request(doc, identity(doc), lambda: None), expected)
            self.assertEqual(arms['endpoints']['request'], arms['mccormick']['request'])
            for duty in expected[1]['duties']:
                lp = mc_lp(duty); base = duty['base']
                self.assertEqual(lp, reconstruct_mc(duty))
                self.assertEqual(csr(lp['A'])[:len(base['b'])], csr(base['A']))
                self.assertEqual(csr(lp['E']), csr(base['E']))
                self.assertEqual(lp['lower'][:-2], base['lower'])
                self.assertEqual(lp['upper'][:-2], base['upper'])
                self.assertEqual([lp['lower'][-2], lp['upper'][-2]], duty['gate'])

    def test_frozen_protocol_all_fields_bound(self):
        for field, value in [('real_requests', 1), ('hard_supervision', True),
                             ('acceptance_threshold', '0'), ('cases', ['weighted_sign']),
                             ('maximum_seconds_per_inprocess_arm', 301)]:
            changed = deepcopy(protocol()); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): protocol(changed)
        with patch('scoped_source.endpoint_source_controls.BUILD_THRESHOLD', F(0)), self.assertRaises(ValueError):
            protocol()

    def test_gate_sign_direction_and_unresolved_fallback(self):
        _, arms = self.examples['weighted_sign']
        duties = arms['endpoints']['request']['duties']
        self.assertEqual(duties[0]['gate'], ['0', '1/2'])
        self.assertEqual(duties[2]['gate'], ['1/2', '1'])
        _, arms = self.examples['tied_partial_reuse']
        self.assertTrue(all(d['gate'] == ['1/2', '1/2'] for d in arms['endpoints']['request']['duties']))
        _, arms = self.examples['unresolved_sign']
        self.assertTrue(all(d['gate'] == ['0', '1'] for d in arms['endpoints']['request']['duties']))

    def test_independent_negative_point_and_bound_matrix_binding(self):
        doc, arms = self.examples['weighted_sign']
        point = weighted_negative_point(doc, arms['mccormick'])
        result = audit_case(doc, arms, expected_source_sha256=identity(doc), negative_point=point)
        self.assertEqual(result['mc_negative_point'], point)
        self.assertEqual(result['route_witnesses'][0]['legal_pairs'], [[1, 2]])
        self.assertEqual(result['route_witnesses'][2]['legal_pairs'], [[0, 1]])
        with self.assertRaises(ValueError): weighted_negative_point(doc, arms['endpoints'])
        for kind in ('matrix', 'point', 'claim', 'semantic'):
            bad = deepcopy(point)
            if kind == 'matrix': bad['lp_sha256'] = '0'*64
            if kind == 'point': bad['point'][-1] = '-100'
            if kind == 'claim': bad['checked_objective'] = '-1'
            if kind == 'semantic': bad['network_counterexample'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_case(doc, arms, expected_source_sha256=identity(doc), negative_point=bad)

    def test_independent_audit_rejects_unequal_arm_facts(self):
        doc, arms = self.examples['weighted_sign']; packages = deepcopy(arms)
        # This request has no interval-positive reused duty; changing the unused
        # selection remains sound individually but is not the frozen fair comparison.
        packages['mccormick']['reuse_requested'] = [[[0, 1], 1]]
        verify(doc, packages['mccormick'])
        with self.assertRaises(ValueError): audit_case(doc, packages, expected_source_sha256=identity(doc))

    def test_external_source_arm_and_same_shape_other_source_binding(self):
        doc, arms = self.examples['weighted_sign']
        with self.assertRaises(ValueError):
            check(doc, arms['endpoints'], expected_source_sha256='0'*64,
                  expected_mode='endpoints', deadline=time.monotonic()+300)
        with self.assertRaises(ValueError):
            check(doc, arms['endpoints'], expected_source_sha256=identity(doc),
                  expected_mode='mccormick', deadline=time.monotonic()+300)
        changed = deepcopy(doc); changed['request']['radius'] = '1/2'
        package = deepcopy(arms['endpoints']); package['source_sha256'] = identity(changed)
        with self.assertRaises(ValueError): verify(changed, package)

    def test_source_rows_ranges_input_guard_gate_and_alias_tampering(self):
        doc, arms = self.examples['weighted_sign']
        for mode in arms:
            for kind in ('input', 'affine', 'relu', 'missing_range', 'extra_range', 'gate', 'guard', 'alias', 'fact'):
                package = deepcopy(arms[mode]); duty = package['request']['duties'][0]
                if kind == 'input': package['bank']['input/0']['bounds'][0] = '0'
                if kind == 'affine': package['bank']['expert0/layer/0/value/0']['rows'][0]['rhs'] = '10'
                if kind == 'relu': package['bank']['expert0/layer/1/value/0']['rows'].pop()
                if kind == 'missing_range': package['bank'].pop('router/layer/0/value/0')
                if kind == 'extra_range': package['bank']['cyclic/fake'] = {'parents': ['cyclic/fake']}
                if kind == 'gate': duty['gate'] = ['0', '0']
                if kind == 'guard': duty['base']['b'][-1] = '100'
                if kind == 'alias': duty['variables'][1] = duty['variables'][0]
                if kind == 'fact': package['scopes'][0]['facts'] = ['100', '100']
                # Rehashing the claimed request cannot override independently rebuilt source rows.
                package['proof']['request_sha256'] = identity(package['request'])
                with self.subTest(mode=mode, kind=kind), self.assertRaises(ValueError): verify(doc, package)

    def test_missing_duplicate_wrong_property_pair_or_endpoint(self):
        doc, arms = self.examples['weighted_sign']
        for mode in arms:
            for kind in ('missing', 'duplicate', 'property', 'pair'):
                package = deepcopy(arms[mode]); records = package['proof']['duties']
                if kind == 'missing': records.pop()
                if kind == 'duplicate': records[-1] = deepcopy(records[0])
                if kind == 'property': records[0]['competitor'] = 0
                if kind == 'pair': records[0]['pair'] = [1, 0]
                with self.subTest(mode=mode, kind=kind), self.assertRaises(ValueError): verify(doc, package)
        package = deepcopy(arms['endpoints']); package['proof']['duties'][0]['endpoints'].pop()
        with self.assertRaises(ValueError): verify(doc, package)

    def test_empty_guard_not_dropped_and_missing_bound_is_unknown(self):
        doc, arms = self.examples['weighted_sign']
        for mode in arms:
            package = deepcopy(arms[mode]); i = package['origins'].index('EMPTY_GUARD_DUAL')
            self.assertEqual(package['proof']['duties'][i]['pair'], [0, 2])
            record = package['proof']['duties'][i]
            if mode == 'endpoints': record['endpoints'][0]['certificate'] = None
            else: record['certificate'] = None
            result = verify(doc, package)
            self.assertEqual(result['status'], 'UNKNOWN_MISSING_EVIDENCE')
            self.assertEqual(result['required'], 3)

    def test_bad_dual_stale_matrix_and_forged_bound(self):
        doc, arms = self.examples['weighted_sign']
        for kind in ('dual', 'matrix', 'bound'):
            package = deepcopy(arms['endpoints'])
            cert = package['proof']['duties'][0]['endpoints'][0]['certificate']
            if kind == 'dual': cert['inequality_dual'][0] = '1'
            if kind == 'matrix': cert['lp_sha256'] = '0'*64
            if kind == 'bound': cert['claimed_lower_bound'] = '100'
            with self.subTest(kind=kind), self.assertRaises(ValueError): verify(doc, package)

    def test_partial_reuse_is_not_whole_request_or_mc_bound(self):
        doc, arms = self.examples['tied_partial_reuse']
        for mode, package in arms.items():
            self.assertEqual(package['origins'].count('SOURCE_BOX_REUSE'), 3)
            self.assertEqual(len(package['proof']['duties']), 18)
            self.assertTrue(all(s['fact_domain'] == 'GLOBAL_INPUT_BOX' for s in package['scopes']))
        package = deepcopy(arms['mccormick']); i = package['origins'].index('SOURCE_BOX_REUSE')
        self.assertEqual(package['proof']['duties'][i]['certificate']['kind'], 'SOURCE_BOX_FACT')
        package['proof']['duties'][i]['certificate']['lower_bound'] = '100'
        with self.assertRaises(ValueError): verify(doc, package)
        package = deepcopy(arms['endpoints']); package['reuse_requested'].append([[0, 1], 1])
        with self.assertRaises(ValueError): verify(doc, package)

    def test_only_proven_facts_reusable_and_unsafe_remains_unknown(self):
        doc, _ = self.examples['unsafe_tied']
        for mode in ('endpoints', 'mccormick'):
            package = produce(doc, mode, [((0, 1), 1)])
            self.assertNotIn('SOURCE_BOX_REUSE', package['origins'])
            self.assertEqual(verify(doc, package)['status'], 'UNKNOWN_MISSING_EVIDENCE')

    def test_nonzero_label_margin_and_private_or_router_only_coordinates(self):
        doc = source(width=3, router_coordinate=2, classes=4, constant=True)
        doc['request']['label'] = 1; doc['request']['margin'] = '2'
        package = produce(doc)
        for duty in package['request']['duties']:
            self.assertNotEqual(duty['competitor'], 1)
            self.assertEqual(duty['a']['offset'], '-2')
            self.assertIn('input/2', duty['variables'])
            for expert in duty['pair']:
                self.assertIn(f'expert{expert}/layer/1/value/0', duty['variables'])
        self.assertEqual(verify(doc, package)['required'], 9)

    def test_relu_stable_positive_negative_and_zero(self):
        for bias in (-2., 0., 2.):
            doc = source(experts=2, classes=2, width=1, relu_bias=bias)
            if bias == 0.: doc['request']['radius'] = '0'
            bank, request, scopes = source_request(doc, identity(doc), lambda: None)
            self.assertEqual((bank, request, scopes), reconstruct(doc, identity(doc), lambda: None))
            self.assertEqual(len(bank['expert0/layer/1/value/0']['rows']), 1)

    def test_candidate_errors_missing_and_alias_pollution(self):
        doc = weighted_source(); before = deepcopy(doc)
        def broken(lp, **kwargs):
            lp['lower'][0] = '100'; raise ValueError('synthetic candidate failure')
        for mode in ('endpoints', 'mccormick'):
            package = produce(doc, mode, proposer=broken)
            self.assertEqual(doc, before)
            self.assertTrue(package['proposal_errors'])
            self.assertEqual(verify(doc, package)['status'], 'UNKNOWN_MISSING_EVIDENCE')

    def test_expired_late_proposal_and_late_check(self):
        doc, arms = self.examples['weighted_sign']
        with self.assertRaises(TimeoutError):
            check(doc, arms['endpoints'], expected_source_sha256=identity(doc),
                  expected_mode='endpoints', deadline=time.monotonic()-1)
        for mode in ('endpoints', 'mccormick'):
            with patch('scoped_source.graph.time.monotonic', return_value=10) as now:
                def late(lp, **kwargs):
                    now.return_value = 12; return None
                with self.assertRaises(TimeoutError):
                    build(doc, expected_source_sha256=identity(doc), mode=mode, deadline=11, proposer=late)
        with patch('scoped_source.graph.time.monotonic', return_value=10) as now:
            from scoped_source.endpoint_source_check import check_bound
            def late_check(lp, certificate):
                result = check_bound(lp, certificate); now.return_value = 12; return result
            with patch('scoped_source.endpoint_source_check.check_bound', side_effect=late_check), self.assertRaises(TimeoutError):
                check(doc, arms['mccormick'], expected_source_sha256=identity(doc), expected_mode='mccormick', deadline=11)

    def test_checker_has_no_producer_or_solver_import(self):
        doc, arms = self.examples['weighted_sign']; root = str(Path(__file__).resolve().parents[1])
        code = '''import sys,json,time
sys.path.insert(0,sys.argv[1])
def audit(event,args):
    if event=='import' and (args[0].split('.')[0] in ('numpy','scipy','torch','highspy','act') or
                           args[0] in ('scoped_source.endpoint_source_build','scoped_source.endpoint_build','scoped_source.endpoint_source_controls')):
        raise RuntimeError('forbidden checker import')
sys.addaudithook(audit)
from scoped_source.endpoint_source_check import check
value=json.loads(sys.stdin.read())
for mode,package in value['arms'].items():
    print(check(value['doc'],package,expected_source_sha256=sys.argv[2],expected_mode=mode,deadline=time.monotonic()+300)['status'])
'''
        result = subprocess.run([sys.executable, '-B', '-I', '-S', '-c', code, root, identity(doc)],
                                input=json.dumps({'doc': doc, 'arms': arms}), text=True, capture_output=True,
                                timeout=30, check=True)
        self.assertIn('CHECKED_DECLARED_SOURCE_POSITIVE', result.stdout)
        self.assertIn('UNKNOWN_NONPOSITIVE', result.stdout)


if __name__ == '__main__': unittest.main()
