"""Ordinary synthetic math/graph/author-interface tests, no trained objects."""
from fractions import Fraction
import math
import os
import time
import unittest

import torch
from torch import nn

from act.back_end.moe.moe_aware_bab import (GuardObjectives, MarginCell,
    closed_cells, propagated_gate_scores, conditional_bounds, prefix_objectives,
    history_objectives, sliced_endpoint, GuardMaximum)
from act.back_end.moe.multilayer import RoutedLayer
from act.back_end.moe.multilayer_external import SequentialTop2

SOURCE = 'a'*64


def linear(weights, biases):
    layer = nn.Linear(len(weights[0]), len(weights), dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weights, dtype=torch.float64))
        layer.bias.copy_(torch.tensor(biases, dtype=torch.float64))
    return layer.eval()


class AnalyticBody(nn.Module):
    def __init__(self, relu_guard=False):
        super().__init__()
        self.affine = linear([[1.]], [0.])
        self.relu_guard = relu_guard

    def _run(self, x):
        x = self.affine(x)
        # Selected0 vs outside2 imposes x>=0 (or relu(x)-.25>=0).
        g = torch.relu(x)-.25 if self.relu_guard else x
        return x, [torch.cat((g, x*0+2., x*0), 1)]


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)

    def test_weak_dual_direction_and_constant_difference(self):
        # Exact finite grid is only a control of the algebra, not a verifier.
        for beta in (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2)):
            lower = Fraction(1, 10)-abs(1-beta)
            for n in range(17):
                x = Fraction(n, 16)
                self.assertLessEqual(lower, x+Fraction(1, 10))
        # d=3 constant still incurs .2*3 error; range oscillation would be zero.
        self.assertAlmostEqual(propagated_gate_scores([.2], [3.], [1.])[0], .6)
        self.assertEqual(propagated_gate_scores([.5, .25], [2., 4.], [2., 3.]), (3., 1.))
        with self.assertRaises(ValueError): propagated_gate_scores([.2], [-1.], [1.])

    def test_closed_cover_tie_degenerate_and_bad_bounds(self):
        cells = closed_cells(0, -.1, .2, 4)
        self.assertEqual((cells[0].lower, cells[-1].upper), (-.1, .2))
        self.assertTrue(all(a.upper >= b.lower for a, b in zip(cells, cells[1:])))
        self.assertEqual(closed_cells(0, 0., 0., 2), (MarginCell(0, 0., 0.),))
        for args in ((0, 2., 1., 2), (0, 0., math.inf, 2), (0, 0., 1., 3)):
            with self.assertRaises(ValueError): closed_cells(*args)

    def test_properties_constants_guard_direction_and_cells(self):
        graph = GuardObjectives(AnalyticBody().eval(), ((0, 1),), (3,),
            torch.tensor([[1.], [-2.]]), torch.tensor([.1, -.4]),
            source_id=SOURCE, cells=(MarginCell(0, -2., -1.),)).eval()
        x = torch.tensor([[-1.], [0.], [1.]])
        out = graph(x)
        torch.testing.assert_close(out[:, :2], x @ graph.rows.T+graph.constants)
        torch.testing.assert_close(out[:, 2], x[:, 0])
        self.assertEqual(float(out[1, 2].detach()), 0.)  # tie remains legal
        torch.testing.assert_close(out[:, 4], x[:, 0])  # m=x-2, m-(-2)=x
        torch.testing.assert_close(out[:, 5], 1.-x[:, 0])
        self.assertEqual(graph.guard_count, 4)
        with self.assertRaises(ValueError):
            GuardObjectives(AnalyticBody().eval(), ((1, 0),), (3,), graph.rows,
                            graph.constants, source_id=SOURCE)

    def test_prefix_has_only_prior_guards_and_preserves_all_properties(self):
        layers = []
        for _ in range(2):
            layers.append(RoutedLayer(linear([[1.], [-1.], [0.]], [0., 0., -2.]),
                [linear([[1.]], [0.]) for _ in range(3)], k=2, mode='selected_softmax'))
        model = SequentialTop2(layers).eval()
        p, ordered = prefix_objectives(model, ((0, 1),), source_id=SOURCE)
        self.assertEqual((p.objective_count, p.guard_count), (6, 2))
        self.assertEqual({i for i, _, _ in p.guard_rows}, {0})
        x = torch.tensor([[.2]])
        r = model.layers[1].router(x)
        torch.testing.assert_close(p(x)[:, :6], torch.stack([-r[:, a]+r[:, b] for a, b in ordered], 1))
        full = history_objectives(model, ((0, 1), (0, 1)), torch.ones((9, 1)),
                                  torch.arange(9.)/10, source_id=SOURCE)
        self.assertEqual((full.objective_count, full.guard_count), (9, 4))

    def test_slice_keeps_margin_guard_and_original_gate_embedding(self):
        from act.back_end.moe.gate_decoupling import GateDomain, sigmoid_interval, extend_box
        from test_gate_decoupling import fixture
        model, rows, th = fixture(2, classes=10)
        h = ((0, 1), (0, 1))
        parent = GateDomain(SOURCE, h, ((-.4, .4), (-2., 2.)),
                            (sigmoid_interval(-.4, .4), sigmoid_interval(-2., 2.)))
        cell = MarginCell(0, 0., .2)
        joint, domain = sliced_endpoint(model, parent, rows, th, cell,
            variant='V2_LO', input_features=1, expected_source_id=SOURCE)
        self.assertNotEqual(domain.source_id, parent.source_id)
        self.assertEqual(domain.margins[0], (0., .2))
        self.assertEqual((joint.objective_count, joint.guard_count), (9, 6))
        # An original x=.25 has first margin .1 inside the cell; embed its
        # actual first gate as the independent variable. Last endpoint is fixed.
        x = torch.tensor([[.25]])
        gate = torch.sigmoid(torch.tensor([[.1]]))
        z = torch.cat((x, gate), 1)
        values = joint(z)
        self.assertTrue((values[0, 9:] >= 0).all())
        torch.testing.assert_close(GuardMaximum(joint)(z), values[:, :9])
        # Same independent gate, but x=-.25 violates cell's *input* condition.
        bad = joint(torch.cat((-x, gate), 1))
        self.assertLess(float(bad[0, -2].detach()), 0.)
        box = extend_box(torch.zeros((1, 1)), -torch.ones((1, 1)), torch.ones((1, 1)),
            domain, variant='V2_LO', expected_source_id=domain.source_id)
        self.assertEqual(float(box['lower'][0, 1]), domain.intervals[0][0])
        with self.assertRaises(ValueError):
            sliced_endpoint(model, parent, rows, th, MarginCell(1, 0., 1.),
                variant='V2_LO', input_features=1, expected_source_id=SOURCE)

    @unittest.skipUnless(os.environ.get('MOE_NATIVE_CONTROLS') == '1', 'isolated author engine only')
    def test_native_two_layer_shared_sigmoid_graph_and_properties(self):
        from test_gate_decoupling import fixture
        model, rows, th = fixture(2, classes=3)
        graph = history_objectives(model, ((0, 1), (0, 1)), rows, th, source_id=SOURCE)
        x = torch.zeros((1, 1))
        on = conditional_bounds(graph, x, x-.25, x+.25,
            deadline=float(os.environ['MOE_CONTROL_DEADLINE']), use_guards=True, iterations=20)
        self.assertEqual((on['objective_count'], on['guard_count']), (2, 4))
        self.assertGreater(on['alpha_parameters'], 0)
        points = torch.linspace(-.25, .25, 65).reshape(-1, 1)
        with torch.no_grad():
            joint = graph(points)
            valid = (joint[:, 2:] >= 0).all(dim=1)
            sampled_min = joint[valid, :2].min(dim=0).values
        self.assertTrue((torch.tensor(on['lower']) <= sampled_min+1e-7).all())
        print('native two-layer bounds', on['lower'], 'alpha parameters', on['alpha_parameters'])

    @unittest.skipUnless(os.environ.get('MOE_NATIVE_CONTROLS') == '1', 'isolated author engine only')
    def test_native_guard_dual_affine_and_guard_only_relu(self):
        deadline = float(os.environ['MOE_CONTROL_DEADLINE'])
        x = torch.zeros((1, 1)); low, high = x-1, x+1
        records = []
        for nonlinear in (False, True):
            graph = GuardObjectives(AnalyticBody(nonlinear).eval(), ((0, 1),), (3,),
                torch.ones((1, 1)), torch.tensor([.1]), source_id=SOURCE).eval()
            off = conditional_bounds(graph, x, low, high, deadline=deadline, iterations=20)
            on = conditional_bounds(graph, x, low, high, deadline=deadline, use_guards=True, iterations=20)
            print('native bound receipt', nonlinear, on, flush=True)
            self.assertAlmostEqual(off['lower'][0], -.9, places=6)
            self.assertGreater(on['lower'][0], off['lower'][0]+.1)
            self.assertLessEqual(on['lower'][0], (.35 if nonlinear else .1)+1e-7)
            self.assertGreater(on['beta_nonzero'], 0)
            self.assertFalse(on['formal_SAFE']); self.assertFalse(on['complete_MoE'])
            records.append(dict(nonlinear=nonlinear, off=off['lower'], on=on['lower']))
        print('M-B native analytic controls:', records)
        with self.assertRaises(TimeoutError):
            conditional_bounds(graph, x, low, high, deadline=time.monotonic()-1)


if __name__ == '__main__':
    unittest.main()
