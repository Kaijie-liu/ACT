"""Fixed four source controls and fail-closed source/endpoint mutations."""
from copy import deepcopy
from fractions import Fraction as F
import inspect
import json
import time
import unittest
from unittest.mock import patch

from source_enclosure.format import unpack, pack, identity
from source_enclosure import produce
from source_enclosure.check import check_affine
from scoped_source.endpoint_source_controls import cases
from scoped_source.sparse_controls import exact_routes
from scoped_source.hz_source_build import build, live
from scoped_source.hz_source_check import check, checked_state, exact_float


class SourceConnectionTests(unittest.TestCase):
    observations = {}

    @classmethod
    def setUpClass(cls):
        cls.observations = {}
        for name, doc, _historical_reuse in cases():
            begin = time.monotonic()
            deadline = begin+300
            cost = {}
            entry = {'source': doc, 'source_sha256': identity(doc), 'cost_seconds': cost}
            cls.observations[name] = entry
            try:
                package = build(doc, expected_source_sha256=identity(doc), deadline=deadline,
                                observe=lambda key, seconds: cost.update({key: seconds}))
                entry['package'] = package
                generated = time.monotonic()
                encoded = json.dumps(package, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                restored = json.loads(encoded)
                serialized = time.monotonic()
                entry['checked'] = check(doc, restored, expected_source_sha256=identity(doc), deadline=deadline)
                done = time.monotonic()
                cost.update(serialization=serialized-generated, check=done-serialized, total=done-begin)
                entry['package_bytes'] = len(encoded)
            except Exception as exc:
                entry['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                cost['total'] = time.monotonic()-begin
                raise
        doc = cls.observations['weighted_sign']['source']
        cls.observations['weighted_sign']['route_witnesses'] = [exact_routes(doc, [x]) for x in (-1, F(-1, 2), 1)]

    def item(self, name='weighted_sign'):
        item = self.observations[name]
        return item['source'], deepcopy(item['package'])

    def accept(self, doc, package):
        return check(doc, package, expected_source_sha256=identity(doc), deadline=time.monotonic()+300)

    def reject(self, doc, package):
        with self.assertRaises((ValueError, KeyError)):
            self.accept(doc, package)

    def test_complete_roster(self):
        from scripts.run_hz_source_controls import check_coverage, check_summary, protocol, PROTOCOL_SHA
        self.assertEqual(list(self.observations), ['weighted_sign', 'tied_partial_reuse', 'unsafe_tied', 'unresolved_sign'])
        for name, expected in [('weighted_sign', 3), ('tied_partial_reuse', 18), ('unsafe_tied', 6), ('unresolved_sign', 6)]:
            checked = self.observations[name]['checked']
            self.assertEqual(checked['required'], expected)
            self.assertEqual(checked['missing_endpoints'], 0)
            self.assertTrue(checked['source_lowering_checked'])
            self.assertFalse(checked['deployed_float_SAFE'])
            check_coverage(name, self.observations[name], checked, self.observations)
            incomplete = dict(checked, missing_endpoints=1, checked_endpoints=checked['checked_endpoints']-1)
            with self.assertRaisesRegex(ValueError, 'complete'):
                check_coverage(name, self.observations[name], incomplete, self.observations)
        self.assertEqual(self.observations['unsafe_tied']['checked']['positive'], 0)
        witnesses = self.observations['weighted_sign']['route_witnesses']
        self.assertEqual(witnesses[1]['legal_pairs'], [[0, 1], [1, 2]])
        self.assertNotEqual(witnesses[0]['legal_pairs'], witnesses[-1]['legal_pairs'])
        config = protocol()
        summary = {'status': 'PASS', 'tests': len(config['controls']), 'protocol_sha256': PROTOCOL_SHA,
                   'outcomes': [{'test': 'scoped_source.test_hz_source.SourceConnectionTests.test_'+n, 'status': 'PASS'}
                                for n in sorted(config['controls'])],
                   'exceptional_tests': {k: 0 for k in ('errors', 'failures', 'skipped', 'expectedFailures', 'unexpectedSuccesses')},
                   'native_solves': 0, 'cuda_calls': 0, 'real_requests': 0,
                   'hard_budget_supervision': False, 'performance_claim': False}
        check_summary(summary, config)
        for field in ('native_solves', 'cuda_calls', 'real_requests', 'hard_budget_supervision', 'performance_claim'):
            changed = deepcopy(summary); changed[field] = 1
            with self.assertRaises(ValueError): check_summary(changed, config)
        for field in summary['exceptional_tests']:
            changed = deepcopy(summary); changed['exceptional_tests'][field] = 1
            with self.assertRaises(ValueError): check_summary(changed, config)

    def test_input_enclosure(self):
        doc, p = self.item()
        p['input']['hz']['Gc']['data'][0] = '1/2'
        self.reject(doc, p)

    def test_affine_compensation(self):
        from act.back_end.solver.solver_hz import sparse_hz_linear
        from act.back_end.solver.hz_lp_export import snapshot
        import numpy as np
        s = produce.box([-F(.1)], [F(.1)])
        op, bias = [{0: F(.1)}], [F(0)]
        nominal = snapshot(sparse_hz_linear(live(s), np.array([[.1]]), np.array([0.])))
        target, cert = produce.affine(s, op, bias, nominal, 'rounding')
        result = check_affine(s, target, op, bias, nominal, cert, 'rounding')
        self.assertEqual(result['error_factors'], 1)
        live(target)
        missing, ids, bids = unpack(target)
        missing['Gc'][0].pop(len(ids)-1)
        with self.assertRaises(ValueError):
            check_affine(s, pack(missing, ids[:-1], bids), op, bias, nominal, cert, 'rounding')
        cert['error_bounds'][0] = '0'
        with self.assertRaisesRegex(ValueError, 'compensation'):
            check_affine(s, target, op, bias, nominal, cert, 'rounding')

    def test_relu_rows(self):
        doc, p = self.item()
        step = p['pairs'][0]['a'][1]
        h, c, b = unpack(step['target'])
        n = len(unpack(p['pairs'][0]['entry'])[0]['ub'])
        h['Aub'][n] = {k: -v for k, v in h['Aub'][n].items()}
        step['target'] = pack(h, c, b)
        self.reject(doc, p)
        doc, p = self.item()
        step = p['pairs'][0]['a'][1]
        self.assertEqual(step['certificate']['branches'].count('unstable'), 2)
        h, c, b = unpack(step['target'])
        for key in ('Auc', 'Aub', 'ub'):
            h[key][n+1], h[key][n+2] = h[key][n+2], h[key][n+1]
        step['target'] = pack(h, c, b)
        self.reject(doc, p)

    def test_relu_range(self):
        doc, p = self.item()
        p['pairs'][0]['a'][1]['certificate']['ranges'][0] = ['0', '1']
        self.reject(doc, p)

    def test_layer_inventory(self):
        doc, p = self.item()
        p['pairs'][0]['a'].pop()
        self.reject(doc, p)

    def test_guard_direction(self):
        doc, p = self.item()
        h, c, b = unpack(p['pairs'][0]['entry'])
        h['Auc'][-1] = {k: -v for k, v in h['Auc'][-1].items()}
        h['ub'][-1] = -h['ub'][-1]
        p['pairs'][0]['entry'] = pack(h, c, b)
        self.reject(doc, p)

    def test_factor_alias(self):
        doc, p = self.item()
        p['pairs'][0]['b'][1]['target']['continuous_ids'][-1] = p['pairs'][0]['a'][1]['target']['continuous_ids'][-1]
        self.reject(doc, p)

    def test_gate_orientation(self):
        doc, p = self.item()
        p['pairs'][0]['gate_evidence']['weight_expert'] = 1
        self.reject(doc, p)
        doc, p = self.item()
        p['endpoint_request']['pairs'][0]['gate']['bounds'] = ['1/2', '1']
        self.reject(doc, p)

    def test_endpoint_source(self):
        doc, p = self.item()
        p['endpoint_request']['pairs'][0]['sources']['a']['c'][0] += 1
        self.reject(doc, p)

    def test_source_identity(self):
        doc, p = self.item()
        changed = deepcopy(doc)
        changed['request']['margin'] = '1'
        with self.assertRaises(ValueError):
            check(changed, p, expected_source_sha256=identity(doc), deadline=time.monotonic()+300)
        self.reject(changed, p)

    def test_property_binding(self):
        doc, p = self.item()
        p['endpoint_request']['properties'][0]['offset'] = '1'
        self.reject(doc, p)

    def test_missing_pair(self):
        doc, p = self.item('tied_partial_reuse')
        p['pairs'].pop()
        p['endpoint_request']['pairs'].pop()
        self.reject(doc, p)

    def test_partial_proof(self):
        doc, p = self.item('tied_partial_reuse')
        p['proof']['pairs'][-1]['candidates'] = None
        checked = self.accept(doc, p)
        self.assertEqual(checked['status'], 'UNKNOWN_MISSING_EVIDENCE')
        self.assertGreater(checked['missing_endpoints'], 0)
        self.observations['partial_proof'] = {'source': doc, 'package': p, 'checked': checked}
        from scripts.run_hz_source_controls import check_coverage
        check_coverage('partial_proof', self.observations['partial_proof'], checked, self.observations)
        wrong = deepcopy(self.observations['partial_proof'])
        wrong['package']['proof']['pairs'][0]['candidates'] = None
        with self.assertRaises(ValueError):
            check_coverage('partial_proof', wrong, checked, self.observations)

    def test_stale_proof(self):
        doc, p = self.item()
        p['proof'] = deepcopy(self.observations['unresolved_sign']['package']['proof'])
        self.reject(doc, p)

    def test_nonrepresentable(self):
        with self.assertRaisesRegex(ValueError, 'representable'):
            exact_float(F(1, 10))
        doc, p = self.item()
        p['input']['hz']['Gc']['data'][0] = '1/10'
        with self.assertRaisesRegex(ValueError, 'representable'):
            checked_state(p['input'])

    def test_deadline(self):
        doc, p = self.item()
        with self.assertRaises(TimeoutError):
            check(doc, p, expected_source_sha256=identity(doc), deadline=time.monotonic()-1)
        with self.assertRaises(TimeoutError):
            build(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()-1)

    def test_independent_imports(self):
        doc, p = self.item()
        with patch('scoped_source.hz_source_build.propagate', side_effect=AssertionError('producer forbidden')), \
                patch('act.back_end.moe.batched_support.propose_batch', side_effect=AssertionError('proposal forbidden')):
            result = self.accept(doc, p)
        self.assertEqual(result, self.observations['weighted_sign']['checked'])
        import scoped_source.hz_source_check as checker
        text = inspect.getsource(checker)
        self.assertNotIn('import scipy', text)
        self.assertNotIn('import torch', text)
        self.assertNotIn('hz_source_build', text)


if __name__ == '__main__':
    unittest.main()
