"""Analytic, identity, mutation and solver-free construction controls only."""
import copy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.capture import capture
from scoped_source.graph import operator, validate
from scoped_source.build import build
from scoped_source.check import check, check_join, check_guards
from source_enclosure.format import identity, unpack, pack
from full_source.obligations import materialize
from upstream_source.checker import csr


def fixture(experts=3, classes=3, width=2, depth=2, tied=False):
    import torch
    from act.back_end.moe.model import OutputLevelMoE
    from act.back_end.moe.schema import OutputLevelMoESpec, GateKind
    nn = torch.nn
    def network(out):
        layers = [nn.Flatten(1)]
        for _ in range(depth): layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, out))
        net = nn.Sequential(*layers).double()
        with torch.no_grad():
            for layer in net:
                if type(layer) is nn.Linear:
                    values = (torch.arange(layer.weight.numel()).reshape(layer.weight.shape) % 5 - 2) / 4
                    layer.weight.copy_(values); layer.bias.zero_()
        return net
    model = OutputLevelMoE(network(experts), [network(classes) for _ in range(experts)],
        OutputLevelMoESpec(experts, 2, GateKind.SELECTED_SOFTMAX)).eval()
    if tied:
        with torch.no_grad():
            for p in model.router.parameters(): p.zero_()
    center = torch.zeros((1, 1, width), dtype=torch.float64)
    doc = capture(model, center, label=classes-1, radius='1', margin='1/100', clip=['-1', '1'], deadline=time.monotonic()+300)
    bundle = build(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+300)
    return model, center, doc, bundle


def checked(doc, bundle):
    return check(doc, bundle, expected_source_sha256=identity(doc), deadline=time.monotonic()+300)


class GenericSourceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.center, cls.doc, cls.bundle = fixture()

    def test_generic_expert_class_depth_dimensions(self):
        for e, c, w, d in [(2, 2, 1, 0), (3, 4, 3, 1), (4, 2, 2, 3)]:
            with self.subTest(e=e, c=c, w=w, depth=d):
                _, _, doc, bundle = fixture(e, c, w, d)
                result = checked(doc, bundle)
                self.assertEqual(result['output_obligations'], e*(e-1)//2*(c-1))
                self.assertEqual(result['steps_checked'], (e+1)*(2*d+2))
                self.assertFalse(result['complete_output_positive_proof'])
                self.assertEqual(result['lower_bounds_checked'], 0)

    def test_all_tie_pairs_kept(self):
        _, _, doc, bundle = fixture(4, 3, 1, 1, tied=True)
        result = checked(doc, bundle)
        self.assertEqual(result['guarded_pairs'], 6)
        self.assertFalse(result['route_changing_established'])
        for p in bundle['pairs']:
            h, _, _ = unpack(p['guarded'])
            self.assertEqual(h['ub'][-4:], [0]*4)
            self.assertEqual(h['Auc'][-4:], [{}]*4)

    def test_252_synthetic_obligations_not_real_request(self):
        _, _, doc, bundle = fixture(8, 10, 1, 0, tied=True)
        result = checked(doc, bundle)
        self.assertEqual(result['guarded_pairs'], 28)
        self.assertEqual(result['output_obligations'], 252)
        self.assertEqual(result['lower_bounds_checked'], 0)

    def test_rational_radius_and_constant_input(self):
        import torch
        center = torch.tensor([[[.1, .3]]], dtype=torch.float64)
        for radius in ('2/255', '0'):
            doc = capture(self.model, center, label=2, radius=radius, margin='1/100', clip=['0', '1'], deadline=time.monotonic()+300)
            bundle = build(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+300)
            checked(doc, bundle)
            request, lo, hi = validate(doc, identity(doc), lambda: None)
            self.assertEqual(lo[0], F(.1)-F(radius))
            self.assertEqual(hi[1], F(.3)+F(radius))

    def test_constant_expert_difference_mccormick(self):
        import torch
        model = copy.deepcopy(self.model)
        with torch.no_grad():
            for expert in model.experts:
                for p in expert.parameters(): p.zero_()
        doc = capture(model, self.center, label=2, radius='1', margin='1/100', clip=['-1', '1'], deadline=time.monotonic()+300)
        bundle = build(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+300)
        checked(doc, bundle)
        self.assertTrue(all(row['difference'] == ['0', '0'] for p in bundle['pairs'] for row in p['obligations']['rows']))

    def test_capture_does_not_execute_model(self):
        import torch
        with patch.object(torch.nn.Module, '_call_impl', side_effect=AssertionError('no forward')):
            doc = capture(self.model, self.center, label=2, radius='1', margin='1/100', clip=['-1', '1'], deadline=time.monotonic()+300)
        self.assertEqual(doc, self.doc)

    def test_unsupported_models_and_semantics_reject(self):
        import torch
        for mode in ('train', 'operator', 'float32', 'hook'):
            with self.subTest(mode=mode):
                model = copy.deepcopy(self.model)
                if mode == 'train': model.train()
                if mode == 'operator': model.experts[0][2] = torch.nn.Sigmoid().eval()
                if mode == 'float32': model.float()
                if mode == 'hook': model.router[1].register_forward_hook(lambda m, i, o: o+1)
                with self.assertRaises(ValueError):
                    capture(model, self.center, label=2, radius='1', margin='1/100', clip=['-1', '1'], deadline=time.monotonic()+300)

    def test_external_source_anchor(self):
        with self.assertRaises(ValueError):
            check(self.doc, self.bundle, expected_source_sha256='0'*64, deadline=time.monotonic()+300)
        bad = copy.deepcopy(self.doc); bad['request']['margin'] = '0'
        with self.assertRaises(ValueError):
            check(bad, self.bundle, expected_source_sha256=identity(self.doc), deadline=time.monotonic()+300)

    def test_parameter_inventory_and_request_changes(self):
        for mode in ('parameter', 'inventory', 'name', 'classes', 'label', 'radius', 'gate', 'network'):
            with self.subTest(mode=mode):
                doc = copy.deepcopy(self.doc)
                if mode == 'parameter': doc['networks'][1]['layers'][1]['weight'] = doc['networks'][1]['layers'][1]['bias']
                if mode == 'inventory': doc['state_inventory'].pop()
                if mode == 'name': doc['networks'][1]['layers'][1]['bias_name'] = 'router.1.bias'
                if mode == 'classes': doc['request']['classes'] += 1
                if mode == 'label': doc['request']['label'] = 99
                if mode == 'radius': doc['request']['radius'] = '-1'
                if mode == 'gate': doc['request']['gate'] = 'HARD_TOP1'
                if mode == 'network': doc['networks'].pop()
                with self.assertRaises((ValueError, KeyError)):
                    validate(doc, identity(doc), lambda: None)

    def test_missing_layer_pair_property_and_old_bounds(self):
        for mode in ('layer', 'network', 'pair', 'property', 'duplicate', 'old_bound', 'partial'):
            with self.subTest(mode=mode):
                bad = copy.deepcopy(self.bundle)
                if mode == 'layer': bad['networks'][1]['steps'].pop()
                if mode == 'network': bad['networks'].pop()
                if mode == 'pair': bad['pairs'].pop()
                if mode == 'property': bad['pairs'][0]['obligations']['rows'].pop()
                if mode == 'duplicate': bad['pairs'][0] = bad['pairs'][1]
                if mode == 'old_bound': bad['pairs'][0]['obligations']['rows'][0]['lower_bound_certificate'] = {'bound': '10'}
                if mode == 'partial': bad['lower_bound_certificates'] = [{'missing_rest': True}]
                with self.assertRaises(ValueError): checked(self.doc, bad)

    def test_input_containment_and_affine_changes(self):
        for mode in ('input', 'affine', 'relu', 'frame'):
            with self.subTest(mode=mode):
                bad = copy.deepcopy(self.bundle)
                if mode == 'input': bad['input']['hz']['Gc']['data'][0] = '1/2'
                if mode == 'affine': bad['networks'][1]['steps'][1]['state']['hz']['c'][0] = '10'
                if mode == 'relu': bad['networks'][1]['steps'][2]['proof']['ranges'][0] = ['0', '0']
                if mode == 'frame': bad['networks'][1]['steps'][1]['state']['hz']['frame_id'] += 1
                with self.assertRaises(ValueError): checked(self.doc, bad)

    def test_guard_sign_order_margin_and_product(self):
        for mode in ('guard', 'rhs', 'expert_order', 'margin', 'plane', 'range'):
            with self.subTest(mode=mode):
                bad = copy.deepcopy(self.bundle); p = bad['pairs'][0]
                if mode == 'guard': p['guarded']['hz']['Auc']['data'][-1] = '17'
                if mode == 'rhs': p['guarded']['hz']['ub'][-1] = '-1'
                if mode == 'expert_order': p['joint']['continuous_ids'][-1] = 'wrong/expert'
                if mode == 'margin': p['projected']['hz']['c'][2] = '0'
                if mode == 'plane': p['obligations']['rows'][0]['A_extra']['data'][0] = '17'
                if mode == 'range': p['obligations']['rows'][0]['difference'] = ['0', '0']
                with self.assertRaises(ValueError): checked(self.doc, bad)

    def test_private_factor_alias_rejected(self):
        traces = self.bundle['networks']; router = traces[0]['steps'][-1]['state']; a = traces[1]['steps'][-1]['state']
        with self.assertRaises(ValueError):
            check_join(self.bundle['input'], [router, a, a], self.bundle['pairs'][0]['joint'])

    def test_independent_checker_never_calls_producer(self):
        with patch('scoped_source.build.build', side_effect=AssertionError('producer')), \
             patch('full_source.lift.affine', side_effect=AssertionError('lift producer')), \
             patch('source_enclosure.produce.relu', side_effect=AssertionError('ReLU producer')), \
             patch('full_source.obligations.build', side_effect=AssertionError('F0 producer')):
            checked(self.doc, self.bundle)

    def test_deadline_and_late_result(self):
        for deadline in (time.monotonic()-1, float('inf'), time.monotonic()+301):
            with self.assertRaises((ValueError, TimeoutError)):
                check(self.doc, self.bundle, expected_source_sha256=identity(self.doc), deadline=deadline)
        # A check which completes after expiry must not publish a success.
        from scoped_source.check import check_outputs
        with patch('scoped_source.graph.time.monotonic', return_value=10) as now:
            def late(*args):
                result = check_outputs(*args); now.return_value = 12; return result
            with patch('scoped_source.check.check_outputs', side_effect=late):
                with self.assertRaises(TimeoutError):
                    check(self.doc, self.bundle, expected_source_sha256=identity(self.doc), deadline=11)
        with self.assertRaises(TimeoutError):
            build(self.doc, expected_source_sha256=identity(self.doc), deadline=time.monotonic()-1)

    def test_standard_library_only_subprocess(self):
        # Read only serialized synthetic objects; no checkpoint/data/solver/model.
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as directory:
            p = Path(directory)/'objects.json'
            p.write_text(json.dumps([self.doc, self.bundle]))
            code = ("import json,sys,time; from scoped_source.check import check; "
                "d,b=json.load(open(sys.argv[1])); "
                "r=check(d,b,expected_source_sha256=sys.argv[2],deadline=time.monotonic()+30); "
                "assert not any(n.split('.')[0] in ('torch','numpy','scipy','act') for n in sys.modules); "
                "assert 'scoped_source.build' not in sys.modules; print(json.dumps(r))")
            result = subprocess.run([sys.executable, '-S', '-c', code, str(p), identity(self.doc)],
                cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True, text=True, timeout=35)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['output_obligations'], 6)

    def test_exact_feasible_extensions_and_weighted_lp(self):
        doc, bundle = self.doc, self.bundle; r = doc['request']
        for x in ([F(-1), F(1)], [F(0), F(0)], [F(1,3), F(-1,2)]):
            root, ci, _ = unpack(bundle['input'])
            assignment = {ci[i]: (x[i]-root['c'][i])/root['Gc'][i][i] for i in range(len(x))}
            ends = {}
            def value(state):
                h, c, b = unpack(state)
                return [h['c'][i] + sum(v*assignment[c[j]] for j, v in h['Gc'][i].items()) +
                    sum(v*assignment[b[j]] for j, v in h['Gb'][i].items()) for i in range(len(h['c']))]
            def feasible(state):
                h, c, b = unpack(state)
                self.assertTrue(all(-1 <= assignment[v] <= 1 for v in c))
                self.assertTrue(all(assignment[v] in (-1, 1) for v in b))
                for a, ab, rhs, equality in [('Ac','Ab','b',True), ('Auc','Aub','ub',False)]:
                    for ac, bc, rhs_value in zip(h[a], h[ab], h[rhs]):
                        lhs = sum(v*assignment[c[j]] for j,v in ac.items()) + sum(v*assignment[b[j]] for j,v in bc.items())
                        self.assertEqual(lhs, rhs_value) if equality else self.assertLessEqual(lhs, rhs_value)
            for graph, trace in zip(doc['networks'], bundle['networks']):
                source = bundle['input']; shape = r['center']['shape']
                for layer, step in zip(graph['layers'], trace['steps']):
                    old = value(source); shape, op, bias = operator(shape, layer)
                    target = step['state']; h, c, _ = unpack(target)
                    tag = graph['name']+'/layer/'+str(layer['index'])
                    if layer['kind'] == 'Linear':
                        expected = [bias[i]+sum(v*old[j] for j,v in row.items()) for i,row in enumerate(op)]
                        for i, row in enumerate(h['Gc']):
                            if row:
                                col, radius = next(iter(row.items())); assignment[c[col]] = (expected[i]-h['c'][i])/radius
                    elif layer['kind'] == 'ReLU':
                        expected = [max(F(0), v) for v in old]
                        for i, (bounds, branch) in enumerate(zip(step['proof']['ranges'], step['proof']['branches'])):
                            if branch != 'unstable': continue
                            lo, hi = map(F, bounds); positive = old[i] >= 0
                            assignment[tag+f'/sign/{i}'] = -1 if positive else 1
                            assignment[tag+f'/negative/{i}'] = F(1) if positive else 2*old[i]/lo-1
                            assignment[tag+f'/positive/{i}'] = 1-2*old[i]/hi if positive else F(1)
                    else: expected = old
                    feasible(target); self.assertEqual(value(target), expected); source = target
                ends[graph['name']] = value(source)
            scores = ends['router']
            for p in bundle['pairs']:
                a, b = p['pair']
                if any(scores[i] < scores[j] for i in (a,b) for j in range(r['experts']) if j not in (a,b)): continue
                feasible(p['guarded']); feasible(p['projected'])
                _, c, bi = unpack(p['projected']); factors = [assignment[v] for v in c+bi]
                for row in p['obligations']['rows']:
                    k = row['competitor']; y = r['label']; lam = F(1,3)
                    va = ends[f'expert{a}'][y]-ends[f'expert{a}'][k]
                    vb = ends[f'expert{b}'][y]-ends[f'expert{b}'][k]
                    point = factors+[lam, lam*(va-vb)]; lp = materialize(p['base'], row)
                    self.assertTrue(all(F(l)<=v<=F(u) for l,v,u in zip(lp['lower'],point,lp['upper'])))
                    for matrix, rhs, equality in [('A','b',False), ('E','h',True)]:
                        for coeff, rhs_value in zip(csr(lp[matrix]), map(F,lp[rhs])):
                            lhs = sum(v*point[j] for j,v in coeff.items())
                            self.assertEqual(lhs, rhs_value) if equality else self.assertLessEqual(lhs, rhs_value)
                    objective = F(lp['offset'])+sum(F(v)*pnt for v,pnt in zip(lp['c'],point))
                    self.assertEqual(objective, lam*va+(1-lam)*vb-F(r['margin']))


if __name__ == '__main__': unittest.main()
