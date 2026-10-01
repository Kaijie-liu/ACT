"""Fourteen bounded integration controls; no native solver, data or CUDA."""
import copy
from fractions import Fraction as F
import math
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, ConSet, Fact, Layer, Net
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.moe import checked_propagation as api
from act.back_end.moe.hz_routing import guarded_input_topk_set
from act.back_end.solver.hz_lp_export import snapshot
from act.back_end.solver.solver_hz import sparse_hz_from_bounds, sparse_hz_linear, sparse_hz_fast_bounds
from act.config.config import HybridZConfig
from act.util.device_manager import initialize_device
from scoped_source.rowwise_bound import identity


def fixture(n=1, two_layers=False):
    bounds = Bounds(torch.full((1, n), -1., dtype=torch.float64), torch.ones(1, n, dtype=torch.float64))
    hz = sparse_hz_from_bounds(bounds, frame_id=51)
    weights = np.zeros((3, n)); weights[0, 0] = 1; weights[1, 0] = -1
    router = sparse_hz_linear(hz, weights)
    v = list(range(n))
    layers = [Layer(0, 'INPUT', {'shape': (1, n), 'dtype': 'torch.float64'}, v, v),
              Layer(1, 'INPUT_SPEC', {'kind': 'BOX', 'lb': bounds.lb.clone(), 'ub': bounds.ub.clone()}, v, v)]
    for i in range(2 if two_layers else 1):
        k = len(layers)
        out = list(range((i*2+1)*n, (i*2+2)*n))
        layers.append(Layer(k, 'DENSE', {'weight': torch.eye(n, dtype=torch.float64),
                            'bias': torch.full((n,), .25 if i == 0 else -.25, dtype=torch.float64),
                            'in_features': n, 'out_features': n}, v, out))
        v = list(range((i*2+2)*n, (i*2+3)*n))
        layers.append(Layer(k+1, 'RELU', {}, out, v))
    layers.append(Layer(len(layers), 'ASSERT', {'kind': 'LINEAR_LE',
                       'C': torch.ones(1, n, dtype=torch.float64),
                       'thresholds': torch.zeros(1, dtype=torch.float64), 'M': 1}, v, v))
    net = Net(layers, {i: ([] if i == 0 else [i-1]) for i in range(len(layers))},
              {i: ([] if i == len(layers)-1 else [i+1]) for i in range(len(layers))})
    return net, hz, router, bounds


def config(**kwargs):
    return HybridZConfig(sparse_resource_policy='csr_bytes_v1', sparse_representation_bytes=16*2**20, **kwargs)


class CheckedPropagationTests(unittest.TestCase):
    observations = {}

    @classmethod
    def setUpClass(cls):
        initialize_device('cpu', 'float64')
        torch.set_num_threads(1)

    def setUp(self):
        self.end = time.monotonic()+300
        # Every native call in this stage is an error unless explicitly replaced
        # by the budget-only test seam below.
        self.no_native = patch('act.back_end.hybridz_tf.tf_mlp._guarded_support_query',
                               side_effect=AssertionError('native solver forbidden in controls'))
        self.no_native.start()
        self.addCleanup(self.no_native.stop)

    def run_case(self, *, data=None, name=None, **kwargs):
        net, hz, router, bounds = data or fixture()
        arguments = dict(input_hz=hz, router_hz=router, route=(0, 1), expert=0,
                         request='frozen-synthetic-integration', entry_bounds=bounds,
                         config=config(), deadline=self.end)
        arguments.update(kwargs)
        result = api.propagate_checked_guarded(net, **arguments)
        if name: self.observations[name] = result
        return result

    def tf(self, data=None, **kwargs):
        net, hz, router, bounds = data or fixture()
        arguments = dict(net=net, input_hz=hz, router_hz=router, route=(0, 1), expert=0,
                         request='scope-control', entry_bounds=bounds, deadline=self.end)
        arguments.update(kwargs)
        return api.CheckedGuardedHybridzTF(config(), **arguments)

    def test_actual_analyzer_relu_stabilization(self):
        p = self.run_case(name='retained_guard')
        self.assertEqual(p['events'][0]['status'], 'CHECKED_SUPPORT_APPLIED')
        self.assertEqual(p['events'][0]['fast_unstable'], 1)
        self.assertEqual(p['events'][0]['after_checked_unstable'], 0)
        self.assertEqual(p['output']['Gb']['shape'][1], 0)
        self.assertFalse(p['network_or_complete_moe_proof'])
        # An exact-zero preactivation uses the same two-sided obligation roster;
        # no claim is made that a finite proposal must establish exact zero.
        zero = fixture(); zero[0].layers[2].params['bias'].zero_()
        z = self.run_case(data=zero)
        self.assertEqual(len(z['events'][0]['package']['accepted']['results']), 2)
        # Negative preactivation uses the independently checked upper side.
        d = fixture(); d[0].layers[2].params['bias'].fill_(-.25)
        negative = self.run_case(data=d, name='negative_relu')
        self.assertEqual(negative['output']['Gc']['data'], [])
        self.assertEqual(negative['output']['c'], [0.])

    def test_disabled_matches_existing_propagation(self):
        net, hz, router, bounds = fixture()
        p = self.run_case(data=(net, hz, router, bounds), options=api.CheckedSupportOptions(enabled=False), name='disabled')
        from act.back_end.analyze import analyze
        from act.back_end.moe.route_a import _component_output_hz
        import act.back_end.transfer_functions as state
        old_tf, old_mode = state._current_tf, state.get_solver_mode()
        tf = HybridzTF(config()); tf.set_entry_hz(guarded_input_topk_set(hz, router, (0, 1)).hz)
        try:
            state.set_transfer_function(tf); state.set_solver_mode('hybridz')
            analyze(net, 0, Fact(bounds, ConSet()))
            self.assertEqual(p['output'], snapshot(_component_output_hz(net, tf)))
        finally:
            state.set_transfer_function(old_tf); state.set_solver_mode(old_mode)
        self.assertEqual(p['events'], [])
        self.assertEqual(p['output']['Gb']['shape'][1], 1)

    def test_guard_removed_sound_outer_control(self):
        p = self.run_case(retain_guard=False, name='guard_discarded')
        self.assertEqual(p['output']['Gb']['shape'][1], 1)
        self.assertEqual(p['events'][0]['status'], 'NO_CONSTRAINTS')
        self.assertEqual(p['scope']['guard_kind'], 'GUARD_DISCARDED_OUTER')

    def test_tie_legal_pairs_and_request_binding(self):
        keys = []
        for route in ((0,), (0, 1), (0, 2), (1, 2)):
            result = self.run_case(route=route, expert=route[0])
            keys.append(result['scope_sha256'])
            self.assertEqual(result['scope']['route'], list(route))
        self.assertEqual(len(set(keys)), 4)
        self.assertNotEqual(self.run_case(request='other')['scope_sha256'], keys[1])
        d = fixture(); d[2].frame_id = 52
        with self.assertRaises(ValueError): self.run_case(data=d)
        with self.assertRaises(ValueError): self.run_case(expert=2)

    def test_two_relu_layers_and_private_factors(self):
        p = self.run_case(data=fixture(n=2, two_layers=True), name='two_layers')
        self.assertEqual(len(p['events']), 2)
        self.assertEqual(p['events'][0]['after_checked_unstable'], 1)
        self.assertGreaterEqual(p['events'][1]['package']['accepted']['n_relaxed_binaries'], 1)
        batch = p['events'][1]['package']['batch']
        self.assertGreater(batch['source']['Gc']['shape'][1], 2)
        self.assertEqual(len(batch['base']['lower']), batch['source']['Gc']['shape'][1]+batch['source']['Gb']['shape'][1])
        self.assertEqual(p['output']['frame_id'], 51)

    def test_outward_rational_conversion(self):
        for q in (F(1, 10), F(-1, 10), F(1, 2**1075), F(-1, 2**1075), F(0), F(1), F(2**100)):
            self.assertLessEqual(F(api.outward(q, 'min')), q)
            self.assertGreaterEqual(F(api.outward(q, 'max')), q)
        with self.assertRaises((ValueError, OverflowError)): api.outward(F(2**2048), 'min')
        with self.assertRaises(ValueError): api.outward(F(1), 'bad')

    def test_source_router_network_and_entry_pollution(self):
        for kind in ('input', 'router', 'network', 'entry', 'scope', 'bounds', 'dispatch'):
            data = fixture(); tf = self.tf(data)
            if kind == 'input': data[1].c[0] += .1
            elif kind == 'router': data[2].c[0] += .1
            elif kind == 'network': data[0].layers[2].params['bias'][0] += .1
            elif kind == 'entry': tf._entry_sparse_hz_override.c[0] += .1
            elif kind == 'bounds': data[3].lb[0, 0] -= .1
            elif kind == 'dispatch': data[0].by_id[2] = data[0].layers[3]
            else: tf.scope['expert'] = 1
            with self.assertRaises(ValueError): tf._validate_scope(data[0])
        d = list(fixture())
        d[3] = Bounds(d[3].lb.float(), d[3].ub.float())
        with self.assertRaises(ValueError): self.run_case(data=d)
        from act.back_end.solver.solver_hz import hz_add_output_inequalities
        d = list(fixture())
        d[1] = hz_add_output_inequalities(d[1], torch.ones(1, 1), torch.tensor([.5]))
        # The old router no longer carries the new input-domain restriction.
        with self.assertRaisesRegex(ValueError, 'constraint prefix'): self.run_case(data=d)

    def test_missing_duplicate_wrong_side_and_wrong_source_candidates(self):
        original = api.propose_batch
        for kind in ('missing', 'duplicate', 'binding', 'side', 'source'):
            def bad(batch, **kw):
                candidate = original(batch, **kw)
                if kind == 'missing': candidate['entries'].pop()
                elif kind == 'duplicate': candidate['entries'][1] = copy.deepcopy(candidate['entries'][0])
                elif kind == 'binding': candidate['batch_sha256'] = '0'*64
                elif kind == 'side': batch['queries'][0]['side'] = 'max'
                else: batch['source']['c'][0] += 1
                return candidate
            with patch.object(api, 'propose_batch', side_effect=bad):
                p = self.run_case()
            self.assertIsNone(p['events'][0]['package'])
            self.assertEqual(p['output']['Gb']['shape'][1], 1)
        original_prepare = api.prepare_batch
        for kind in ('q', 'offset'):
            def wrong_objective(hz, queries, **kw):
                altered = copy.deepcopy(queries)
                for query in altered:
                    if kind == 'q': query['q'] = [0]*len(query['q'])
                    else: query['offset'] = 10
                return original_prepare(hz, altered, **kw)
            with patch.object(api, 'prepare_batch', side_effect=wrong_objective):
                p = self.run_case()
            self.assertIsNone(p['events'][0]['package'])
            self.assertIn('origin changed', p['events'][0]['reason'])

    def test_proposal_failure_uses_original_bounds(self):
        with patch.object(api, 'propose_batch', side_effect=RuntimeError('injected proposal failure')):
            p = self.run_case(name='candidate_failure')
        self.assertEqual(p['output']['Gb']['shape'][1], 1)
        self.assertEqual(p['events'][0]['status'], 'CANDIDATE_REJECTED_NATIVE_FALLBACK')

    def test_expired_or_late_evidence_rejected(self):
        with self.assertRaises(TimeoutError): self.run_case(deadline=time.monotonic()-1)
        # Inject expiry after successful independent checking, not before entry.
        original = api.check_batch
        def late(*args, **kwargs):
            value = original(*args, **kwargs)
            raise TimeoutError('controlled late candidate after check')
        with patch.object(api, 'check_batch', side_effect=late):
            p = self.run_case()
        self.assertIsNone(p['events'][0]['package'])
        self.assertEqual(p['output']['Gb']['shape'][1], 1)
        # Absolute expiry in the propagation return cannot publish a result.
        base = HybridzTF._propagate_sparse_hz
        def expire(tf, *args):
            value = base(tf, *args)
            if args[0].kind == 'RELU': tf.tick = lambda: (_ for _ in ()).throw(TimeoutError('deadline after propagation'))
            return value
        with patch.object(HybridzTF, '_propagate_sparse_hz', expire), self.assertRaises(TimeoutError):
            self.run_case()

    def test_native_fallback_remaining_budget(self):
        from types import SimpleNamespace
        observed = []
        def fake_native(tf, hz, rows, *, time_limit, relax_binaries):
            observed.append((time_limit, tf.deadline-time.monotonic(), relax_binaries))
            b = sparse_hz_fast_bounds(hz)
            result = SimpleNamespace(bounds=Bounds(b.lb.reshape(-1)[rows], b.ub.reshape(-1)[rows]),
                       lower_status=['controlled_fallback']*len(rows), upper_status=['controlled_fallback']*len(rows),
                       elapsed=0., solves=0, exact=False, solver_gap=[])
            return result, None
        cfg = config(guarded_support_enabled=True, guarded_support_lp_neurons=2,
                     guarded_support_milp_neurons=2, guarded_support_lp_time_limit=200,
                     guarded_support_milp_time_limit=200)
        with patch('act.back_end.hybridz_tf.tf_mlp._guarded_support_query', side_effect=fake_native):
            p = self.run_case(data=fixture(n=2), config=cfg)
        self.assertTrue(observed)
        event = p['events'][0]
        self.assertLessEqual(sum(event['native_allowances']), event['remaining_before_native'])
        self.assertLess(event['remaining_before_native'], 300)
        self.assertTrue(all(limit <= 200 for limit, _, _ in observed))
        self.assertEqual(len(p['events'][0]['native_stages']), len(observed))
        self.assertTrue(all(s['allowance'] <= s['remaining_before_call'] for s in p['events'][0]['native_stages']))
        with self.assertRaises(ValueError):
            self.run_case(config=cfg, options=api.CheckedSupportOptions(enabled=False))
        observed.clear()
        def expire_after_lp(tf, *args, **kw):
            result = fake_native(tf, *args, **kw)
            tf.tick = lambda: (_ for _ in ()).throw(TimeoutError('native LP overran'))
            return result
        with patch('act.back_end.hybridz_tf.tf_mlp._guarded_support_query', side_effect=expire_after_lp):
            with self.assertRaises(TimeoutError): self.run_case(data=fixture(n=2), config=cfg)
        self.assertEqual(len(observed), 1)

    def test_unsupported_capacity_does_not_truncate_obligations(self):
        p = self.run_case(data=fixture(n=5), name='five_rows')
        self.assertEqual(len(p['events'][0]['selected']), 4)
        self.assertEqual(len(p['events'][0]['package']['accepted']['results']), 8)
        self.assertEqual(p['output']['Gb']['shape'][1], 4)
        p = self.run_case(data=fixture(n=129), name='capacity_fallback')
        self.assertEqual(p['events'][0]['status'], 'CAPACITY_NATIVE_FALLBACK')
        self.assertEqual(p['output']['Gb']['shape'][1], 129)

    def test_global_state_restored_on_failure(self):
        import act.back_end.transfer_functions as state
        previous, mode = state._current_tf, state.get_solver_mode()
        with patch.object(api.CheckedGuardedHybridzTF, 'apply', side_effect=RuntimeError('apply fault')):
            with self.assertRaises(RuntimeError): self.run_case()
        self.assertIs(state._current_tf, previous)
        self.assertEqual(state.get_solver_mode(), mode)

    def test_full_cost_and_evidence_inventory(self):
        from scripts.run_hz_checked_propagation_controls import validate_query_roster
        p = self.run_case()
        self.assertGreaterEqual(p['total_seconds'], p['construction_seconds']+sum(e['total_seconds'] for e in p['events']))
        self.assertEqual(p['scope_sha256'], identity(p['scope']))
        for event in p['events']:
            self.assertLessEqual(sum(event[k] for k in ('prepare_seconds', 'proposal_seconds', 'check_seconds')), event['total_seconds'])
            accepted = event['package']['accepted']
            self.assertEqual(len(accepted['results']), 2*len(event['selected']))
            self.assertFalse(accepted['network_or_complete_moe_proof'])
            self.assertTrue(event['completed'])
            batch = event['package']['batch']
            validate_query_roster(batch, event, p)
            for mode in ('side', 'layer', 'property', 'offset', 'missing'):
                wrong = copy.deepcopy(batch)
                if mode == 'side': wrong['queries'][1]['side'] = 'min'
                elif mode == 'layer': wrong['context']['layer'] += 1
                elif mode == 'property': wrong['queries'][0]['q'][0] = '0'
                elif mode == 'offset': wrong['queries'][0]['offset'] = '1'
                else: wrong['queries'].pop()
                with self.assertRaises(ValueError): validate_query_roster(wrong, event, p)


if __name__ == '__main__':
    unittest.main()
