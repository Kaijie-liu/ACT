"""S1 adapter semantics against analytic fixtures; no trained checkpoint."""
import copy
import time
import unittest
from pathlib import Path
import torch
from torch import nn
from act.pipeline.moe.architecture_evaluation import evaluate, Properties, dense_configuration
from act.back_end.moe.multilayer_external import verify_histories
from act.pipeline.moe.multilayer_study import coverage
from multilayer_stage0 import affine, fixture
from multilayer_stage1 import external_audit

DEADLINE = None  # Set once by the enclosing <=300s control, shared by all cases.
OBSERVED = []


def domain():
    x = torch.zeros((1, 1), dtype=torch.float64)
    return dict(center=x, lower=x-1, upper=x+1,
        rows=torch.tensor([[1., -1.]], dtype=torch.float64),
        thresholds=torch.zeros(1, dtype=torch.float64))


def check(model, tensors, family, mode, expected):
    result = evaluate(model, tensors, family=family, mode=mode, deadline=DEADLINE)
    OBSERVED.append(dict(family=family, mode=mode, expected=expected,
        status=result['status'], returned=result['returned_histories'],
        verified=result['verified_histories'], pruned=result['pruned_histories'],
        evidence=result['evidence_grade']))
    if family == 'moe':
        external_audit(result, [len(l.experts) for l in model.layers], len(tensors['rows']))
    if result['status'] != expected:
        raise AssertionError((family, mode, expected, result))
    return result


class Controls(unittest.TestCase):
    def test_dense_positive_both_methods(self):
        # min_{[-1,1]} (x+2) = 1. Multiple properties, nonzero RHS.
        model = nn.Sequential(affine([[1.], [0.]], [2., 0.]))
        tensors = domain()
        tensors['rows'] = torch.tensor([[1., -1.], [2., 0.]], dtype=torch.float64)
        tensors['thresholds'] = torch.tensor([.5, .25], dtype=torch.float64)
        for mode in ('incomplete', 'complete'):
            result = check(model, tensors, 'dense', mode, 'POSITIVE')
            if mode == 'complete':
                self.assertEqual(result['histories'][0]['outcome']['branching'], 'kfsb')
            else:
                self.assertEqual(len(result['histories'][0]['outcome']['bounds']), 2)

    def test_clean_wrong_is_counted_full_model_witness(self):
        model = affine([[1.], [0.]], [-.25, 0.])
        for mode in ('incomplete', 'complete'):
            result = check(model, domain(), 'dense', mode, 'UNSAFE_REPLAYED')
            self.assertEqual(result['attempted_histories'], 0)
            self.assertEqual(result['witness'], [[0.]])

    def test_zero_bound_not_positive_or_unsafe(self):
        model = affine([[0.], [0.]], [0., 0.])
        result = check(model, domain(), 'dense', 'incomplete', 'UNKNOWN')
        self.assertFalse(result['complete'])
        self.assertEqual(result['histories'][0]['outcome']['bounds'], [0.])

    def test_multilayer_safe_and_all_ties(self):
        for case in ('safe', 'tie'):
            model, tensors = fixture(case)
            if case == 'safe':
                # First-layer convex mixture lies in [-2,2]; all final experts
                # have margin >=18. This tests variable gates without demanding
                # that an incomplete method prove a cancellation-heavy fixture.
                with torch.no_grad():
                    for expert in model.layers[-1].experts:
                        expert.bias[0] = 20.
            for mode in ('incomplete', 'complete'):
                result = check(model, tensors, 'moe', mode, 'POSITIVE')
                self.assertEqual(result['pruned_histories']+result['verified_histories'], 9)
                if case == 'tie':
                    self.assertEqual(result['pruned_histories'], 0)
                    self.assertEqual(result['verified_histories'], 9)

    def test_multilayer_center_replay(self):
        model, tensors = fixture('unsafe')
        for mode in ('incomplete', 'complete'):
            check(model, tensors, 'moe', mode, 'UNSAFE_REPLAYED')

    def test_unknown_short_circuit_keeps_missing_count(self):
        model, tensors = fixture('tie')
        def unknown(*_): return dict(status='unknown'), None
        kw = dict(**tensors, pruning=True, deadline=DEADLINE,
            verify_graph=unknown, bound_router=lambda *_: {})
        short = verify_histories(model, **kw, stop_on_unknown=True)
        full = verify_histories(model, **kw)
        self.assertEqual((short['status'], short['returned_histories'], short['unvisited_histories']),
            ('UNKNOWN', 1, 8))
        self.assertFalse(short['proof_complete'])
        self.assertTrue(short['stopped_at_first_unproved_history'])
        self.assertEqual(full['returned_histories'], 9)
        self.assertEqual(full['status'], 'UNKNOWN')
        self.assertTrue(full['traversal_complete'])

    def test_unsafe_tie_route_not_hidden_by_clean_route(self):
        from act.back_end.moe.multilayer import RoutedLayer
        from act.back_end.moe.multilayer_external import SequentialTop2
        router = affine([[0.], [0.], [0.]], [0., 0., 0.])
        # Clean native tie choice is observed only to build a controlled missing
        # branch: put a negative expert outside it, leaving native output positive.
        selected = set(router(domain()['center']).topk(2, dim=1).indices[0].tolist())
        experts = [affine([[0.], [0.]], [2. if i in selected else -6., 0.]) for i in range(3)]
        model = SequentialTop2([RoutedLayer(router, experts, k=2, mode='selected_softmax')])
        self.assertGreater(float(model(domain()['center'])[0, 0]), 0)
        result = check(model, domain(), 'moe', 'incomplete', 'UNKNOWN')
        self.assertFalse(result['complete'])
        self.assertEqual(result['pruned_histories'], 0)

    def test_property_constants_and_input_guards(self):
        model = affine([[1.], [-1.]], [2., .5])
        tensors = domain(); tensors['thresholds'][0] = .25
        graph = Properties(model, tensors['rows'], tensors['thresholds'])
        with torch.no_grad():
            for p in (-1., 0., 1.):
                point = torch.tensor([[p]], dtype=torch.float64)
                torch.testing.assert_close(graph(point), torch.tensor([[2*p+1.25]], dtype=torch.float64))
        bad = copy.deepcopy(tensors); bad['thresholds'] = torch.zeros(2, dtype=torch.float64)
        with self.assertRaises(ValueError):
            evaluate(model, bad, family='dense', mode='incomplete', deadline=DEADLINE)
        bad = copy.deepcopy(tensors); bad['lower'] = bad['upper']+1
        with self.assertRaises(ValueError):
            evaluate(model, bad, family='dense', mode='incomplete', deadline=DEADLINE)
        with self.assertRaises(TimeoutError):
            evaluate(model, tensors, family='dense', mode='incomplete', deadline=time.monotonic()-1)

    def test_cross_environment_checkpoint(self):
        from act.pipeline.moe.architecture_dense import load_checkpoint
        path = Path('/data1/Kane/MOE/results/architecture_s1_control_20261029_r1/native/control.pt')
        model, payload = load_checkpoint(path)
        model = model.double()
        x = torch.arange(16, dtype=torch.float64).reshape(4, 4)/16
        y = x
        layers = list(model.children())[1:]
        for layer in layers:
            y = (torch.nn.functional.linear(y, layer.weight, layer.bias)
                 if isinstance(layer, nn.Linear) else y.clamp_min(0))
        torch.testing.assert_close(y, model(x), rtol=0, atol=0)
        self.assertEqual(payload['format'], 'act-dense-capacity-s1-v1')
