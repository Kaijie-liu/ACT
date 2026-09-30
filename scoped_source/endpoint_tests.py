"""Solver-free endpoint controls; no checkpoint, training data or sealed input."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from scoped_source.endpoint_build import build
from scoped_source.endpoint_check import check
from scoped_source.endpoint_controls import fixture, manual_proposer, report, certificate, exact_point, mccormick
from source_enclosure.format import identity
from upstream_source.checker import csr


def produce(request, proposer=None):
    return build(request, expected_request_sha256=identity(request), deadline=time.monotonic()+300,
                 proposer=proposer)


def verify(request, proof):
    return check(request, proof, expected_request_sha256=identity(request), deadline=time.monotonic()+300)


class EndpointControls(unittest.TestCase):
    def setUp(self):
        self.protocol, self.request, self.tags = fixture()
        self.proof = produce(self.request, manual_proposer(self.tags))

    def test_complete_private_factor_case_and_exact_relaxation_gap(self):
        result = report()
        check_ = result['endpoint_check']
        self.assertEqual((check_['required'], check_['positive'], check_['endpoint_bounds_checked']), (3, 3, 6))
        self.assertEqual(check_['duties'][0]['endpoint_bounds'], ['1/10', '1/10'])
        self.assertEqual(result['mccormick']['checked_objective'], '-1/40')
        self.assertEqual(result['mccormick']['narrowest_witness_compatible_range_objective'], '-1/40')
        self.assertEqual(result['expert_only_margin_at_legal_tie'], '-3/20')
        self.assertFalse(check_['source_complete'])
        self.assertFalse(check_['deployed_float_SAFE'])
        self.assertEqual(result['native_solver_calls'], 0)

    def test_both_legal_tie_routes_and_empty_pair_still_covered(self):
        result = report()
        self.assertEqual(result['route_witnesses'][1]['pairs'], [[0, 1], [1, 2]])
        self.assertEqual([d['pair'] for d in result['endpoint_check']['duties']], [[0, 1], [0, 2], [1, 2]])
        self.assertNotEqual(result['route_witnesses'][0]['pairs'], result['route_witnesses'][-1]['pairs'])

    def test_universal_gate_is_expertwise_and_cannot_claim_positive(self):
        for duty in self.request['duties']:
            duty['gate'] = ['0', '1']
        proof = produce(self.request, manual_proposer(self.tags))
        result = verify(self.request, proof)
        self.assertEqual(result['status'], 'UNKNOWN_NONPOSITIVE')
        self.assertEqual(result['duties'][0]['endpoint_bounds'][0], '1/10')
        self.assertLess(F(result['duties'][0]['endpoint_bounds'][1]), 0)

    def test_zero_width_gate_needs_one_distinct_endpoint(self):
        self.request['duties'][0]['gate'] = ['1/2', '1/2']
        proof = produce(self.request, manual_proposer(self.tags))
        self.assertEqual(verify(self.request, proof)['endpoint_bounds_checked'], 5)
        proof['duties'][0]['endpoints'].append(deepcopy(proof['duties'][0]['endpoints'][0]))
        with self.assertRaises(ValueError): verify(self.request, proof)

    def test_zero_difference_keeps_complete_endpoint_inventory(self):
        for duty in self.request['duties']:
            duty['a'] = deepcopy(duty['b'])
        observed = []
        def propose(duty, weight, lp):
            observed.append((duty['pair'], identity(lp)))
            return certificate(lp, {})
        proof = produce(self.request, propose)
        for i in range(0, len(observed), 2):
            self.assertEqual(observed[i], observed[i+1])
        self.assertEqual(verify(self.request, proof)['endpoint_bounds_checked'], 6)

    def test_base_constraints_and_inputs_are_not_modified(self):
        before = deepcopy(self.request)
        def propose(duty, weight, lp):
            for key in ('A', 'b', 'E', 'h', 'lower', 'upper'):
                self.assertEqual(lp[key], duty['base'][key])
            lp['A']['data'][0] = '999'  # callback cannot change the compiled LP
            duty['a']['offset'] = '999'  # nor the caller's expected duty
            return None
        proof = produce(self.request, propose)
        self.assertEqual(self.request, before)
        self.assertEqual(verify(self.request, proof)['status'], 'UNKNOWN_MISSING_EVIDENCE')

    def test_mccormick_preserves_same_base_and_gate(self):
        duty = self.request['duties'][0]; base = duty['base']; lp = mccormick(duty)
        self.assertEqual(csr(lp['A'])[:len(base['b'])], csr(base['A']))
        self.assertEqual(lp['b'][:len(base['b'])], base['b'])
        self.assertEqual(csr(lp['E']), csr(base['E']))
        self.assertEqual(lp['h'], base['h'])
        self.assertEqual(lp['lower'][:-2], base['lower'])
        self.assertEqual(lp['upper'][:-2], base['upper'])
        self.assertEqual([lp['lower'][-2], lp['upper'][-2]], duty['gate'])

    def test_frozen_fixture_rejects_unused_or_changed_protocol_fields(self):
        for kind in ('coefficients', 'input', 'threshold', 'roster', 'label', 'alias', 'flags'):
            protocol = deepcopy(self.protocol)
            if kind == 'coefficients': protocol['source']['expert_relu_coefficients'][0] = '1/3'
            if kind == 'input': protocol['source']['input'][0] = '-2'
            if kind == 'threshold': protocol['acceptance_threshold'] = '0'
            if kind == 'roster': protocol['retained_pairs'].pop()
            if kind == 'label': protocol['source']['label'] = 1
            if kind == 'alias': protocol['source']['private_relu_factors'] = False
            if kind == 'flags': protocol['production_integration'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): fixture(protocol)

    def test_swapped_experts_require_complementary_gate(self):
        request = deepcopy(self.request)
        request['experts'] = 2; request['duties'] = request['duties'][:1]
        original = produce(request, manual_proposer(self.tags))
        certs = {e['lp_sha256']: e['certificate'] for e in original['duties'][0]['endpoints']}
        duty = request['duties'][0]; duty['a'], duty['b'] = duty['b'], duty['a']
        lo, hi = map(F, duty['gate']); duty['gate'] = [str(1-hi), str(1-lo)]
        swapped = produce(request, lambda d, w, lp: certs[identity(lp)])
        self.assertEqual(verify(request, swapped)['duties'][0]['endpoint_bounds'], ['1/10', '1/10'])
        with self.assertRaises(ValueError): verify(request, original)

    def test_missing_duplicate_wrong_pair_property_and_endpoint_rejected(self):
        for kind in ('pair', 'property', 'endpoint', 'duplicate', 'reverse', 'boolean', 'weight'):
            proof = deepcopy(self.proof)
            if kind == 'pair': proof['duties'].pop()
            if kind == 'property': proof['duties'][0]['competitor'] = 0
            if kind == 'endpoint': proof['duties'][0]['endpoints'].pop()
            if kind == 'duplicate': proof['duties'][-1] = deepcopy(proof['duties'][0])
            if kind == 'reverse': proof['duties'][0]['pair'].reverse()
            if kind == 'boolean': proof['duties'][0]['pair'][0] = False
            if kind == 'weight': proof['duties'][0]['endpoints'][0]['weight'] = '1/4'
            with self.subTest(kind=kind), self.assertRaises(ValueError): verify(self.request, proof)

    def test_missing_certificate_is_unknown_not_partial_positive(self):
        self.proof['duties'][0]['endpoints'][1]['certificate'] = None
        result = verify(self.request, self.proof)
        self.assertEqual(result['status'], 'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(result['positive'], 2)

    def test_wrong_source_domain_property_gate_or_factor_identity(self):
        expected = identity(self.request)
        for kind in ('bound', 'property', 'gate', 'factor', 'alias'):
            request = deepcopy(self.request); duty = request['duties'][0]
            if kind == 'bound': duty['base']['lower'][0] = '0'
            if kind == 'property': duty['a']['offset'] = '100'
            if kind == 'gate': duty['gate'][1] = '1/4'
            if kind == 'factor': duty['variables'][1] = 'foreign/p0'
            if kind == 'alias': duty['variables'][1] = duty['variables'][2]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check(request, self.proof, expected_request_sha256=expected, deadline=time.monotonic()+300)
        self.request['duties'][0]['variables'][1] = self.request['duties'][0]['variables'][2]
        with self.assertRaises(ValueError): produce(self.request)

    def test_invalid_intervals_and_noncanonical_values_rejected(self):
        for interval in (['1', '0'], ['-1', '1'], ['0', '2'], ['0', '0.5'], [0, '1'], ['0', 'nan']):
            request = deepcopy(self.request); request['duties'][0]['gate'] = interval
            with self.subTest(interval=interval), self.assertRaises(ValueError): produce(request)

    def test_bad_dual_and_stale_lp_rejected(self):
        for kind in ('sign', 'claim', 'stale', 'lp'):
            proof = deepcopy(self.proof); endpoint = proof['duties'][0]['endpoints'][0]
            if kind == 'sign': endpoint['certificate']['inequality_dual'][0] = '1'
            if kind == 'claim': endpoint['certificate']['claimed_lower_bound'] = '1000'
            if kind == 'stale': endpoint['certificate'] = deepcopy(proof['duties'][1]['endpoints'][0]['certificate'])
            if kind == 'lp': endpoint['lp_sha256'] = '0'*64
            with self.subTest(kind=kind), self.assertRaises(ValueError): verify(self.request, proof)

    def test_dimension_and_all_property_coverage(self):
        request = deepcopy(self.request); request['classes'] = 4
        request['duties'] = [{**deepcopy(d), 'competitor': k} for d in self.request['duties'] for k in (1, 2, 3)]
        result = verify(request, produce(request, manual_proposer(self.tags)))
        self.assertEqual((result['required'], result['positive'], result['endpoint_bounds_checked']), (9, 9, 18))

    def test_feasible_relaxation_point_not_network_counterexample(self):
        lp = mccormick(self.request['duties'][0])
        with self.assertRaises(ValueError): exact_point(lp, ['0']*7+['1/4', '-1'])
        self.assertLess(exact_point(lp, ['0']*7+['1/4', '-1/8']), 0)

    def test_cooperative_deadline_and_late_proposal(self):
        with self.assertRaises(TimeoutError):
            check(self.request, self.proof, expected_request_sha256=identity(self.request), deadline=time.monotonic()-1)
        with patch('scoped_source.graph.time.monotonic', return_value=10) as now:
            def late(duty, weight, lp):
                now.return_value = 12
                return certificate(lp, {})
            with self.assertRaises(TimeoutError):
                build(self.request, expected_request_sha256=identity(self.request), deadline=11, proposer=late)

    def test_late_checker_cannot_publish_complete_positive(self):
        from scoped_source.endpoint_check import check_bound
        with patch('scoped_source.graph.time.monotonic', return_value=10) as now:
            def late(lp, cert):
                result = check_bound(lp, cert); now.return_value = 12; return result
            with patch('scoped_source.endpoint_check.check_bound', side_effect=late), self.assertRaises(TimeoutError):
                check(self.request, self.proof, expected_request_sha256=identity(self.request), deadline=11)

    def test_checker_does_not_import_constructor_or_numeric_stack(self):
        root = str(Path(__file__).resolve().parents[1])
        code = '''import sys,json,time
sys.path.insert(0,sys.argv[1])
def audit(event,args):
    if event=='import' and (args[0].split('.')[0] in ('numpy','scipy','torch','highspy') or args[0]=='scoped_source.endpoint_build'):
        raise RuntimeError('forbidden checker import')
sys.addaudithook(audit)
from scoped_source.endpoint_check import check
value=json.loads(sys.stdin.read())
print(check(value['request'],value['proof'],expected_request_sha256=sys.argv[2],deadline=time.monotonic()+300)['status'])
'''
        output = subprocess.run([sys.executable, '-B', '-I', '-S', '-c', code, root, identity(self.request)],
                                input=json.dumps({'request': self.request, 'proof': self.proof}),
                                text=True, capture_output=True, check=True, timeout=30)
        self.assertIn('CHECKED_POSITIVE_GIVEN_BASE_AND_GATE', output.stdout)


if __name__ == '__main__': unittest.main()
