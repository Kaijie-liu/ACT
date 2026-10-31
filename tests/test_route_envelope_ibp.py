"""Analytic inclusion and training controls, without dataset/checkpoint queries."""
import copy
import io
import itertools
import unittest
import torch
from torch import nn
from act.back_end.moe.multilayer import RoutedLayer
from act.back_end.moe.multilayer_external import SequentialTop2
from act.pipeline.moe.route_envelope_ibp import (
    interval, projected_interval, margin_rows, model_bounds, route_envelope,
    route_exclusion_penalty, certified_loss,
)


def affine(weight, bias):
    m = nn.Linear(len(weight[0]), len(weight), dtype=torch.float64)
    with torch.no_grad():
        m.weight.copy_(torch.tensor(weight)); m.bias.copy_(torch.tensor(bias))
    return m


def layer(experts=4, seed=7):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        def net(out):
            return nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, out)).double()
        return RoutedLayer(net(experts), [net(3) for _ in range(experts)],
                           k=min(2, experts), mode='selected_softmax')


class RouteEnvelopeControls(unittest.TestCase):
    def test_random_all_tie_legal_routes_and_properties(self):
        lower = torch.tensor([[-1., -.3], [.1, .2]], dtype=torch.float64)
        upper = lower + torch.tensor([[1.7, 1.1], [.2, .5]], dtype=torch.float64)
        labels = torch.tensor([0, 2]); rows = margin_rows(labels, 3, dtype=lower.dtype)
        for seed in range(3):
            routed = layer(seed=seed)
            out = route_envelope(routed, lower, upper)
            prop = route_envelope(routed, lower, upper, rows, .125)
            generator = torch.Generator().manual_seed(seed)
            for _ in range(30):
                x = lower + torch.rand(lower.shape, generator=generator, dtype=lower.dtype)*(upper-lower)
                scores = routed.router(x)
                for k, (a, b) in enumerate(out.pairs):
                    others = [c for c in range(4) if c not in (a, b)]
                    legal = torch.minimum(scores[:, a], scores[:, b]) >= scores[:, others].amax(1)
                    self.assertTrue((~legal | out.candidates[:, k]).all())
                    weight = torch.sigmoid(scores[:, a]-scores[:, b])[:, None]
                    value = weight*routed.experts[a](x)+(1-weight)*routed.experts[b](x)
                    projected = torch.einsum('bpc,bc->bp', rows, value) + .125
                    for real, bounds in ((value, out), (projected, prop)):
                        self.assertTrue((real[legal] >= bounds.lower[legal]-1e-12).all())
                        self.assertTrue((real[legal] <= bounds.upper[legal]+1e-12).all())

    def test_ties_keep_all_pairs_zero_penalty_not_stable(self):
        routed = layer()
        for p in routed.router.parameters():
            with torch.no_grad(): p.zero_()
        lo = torch.zeros(1, 2, dtype=torch.float64); hi = lo+1
        bounds = route_envelope(routed, lo, hi)
        self.assertEqual(bounds.candidates.sum().item(), 6)
        self.assertFalse(bounds.single_route.item())
        self.assertEqual(route_exclusion_penalty(bounds, torch.tensor([[1, 3]])).item(), 0)
        self.assertTrue(torch.equal(bounds.gate_lower, bounds.gate_upper))
        self.assertTrue((bounds.gate_lower == .5).all())

    def test_tie_boundary_cannot_exclude_alternative(self):
        r = RoutedLayer(affine([[1.], [0.], [0.]], [0., 0., -1.]),
                        [affine([[0.]], [v]) for v in (2., -3., 5.)],
                        k=2, mode='selected_softmax')
        lo = torch.tensor([[-1.]], dtype=torch.float64); hi = lo+2
        b = route_envelope(r, lo, hi)
        self.assertEqual(b.candidates.tolist(), [[True, False, True]])
        self.assertFalse(b.single_route.item())

    def test_direction_constants_and_e2(self):
        # a has smaller weight; reverse-expert rewriting must preserve result.
        r = RoutedLayer(affine([[0.], [0.]], [-2., 1.]),
                        [affine([[0.]], [4.]), affine([[0.]], [-1.])],
                        k=2, mode='selected_softmax')
        lo = torch.tensor([[-1.]], dtype=torch.float64); hi = lo+2
        b = route_envelope(r, lo, hi, torch.tensor([[-2.]], dtype=lo.dtype), .7)
        expected = -2*(torch.sigmoid(torch.tensor(-3., dtype=lo.dtype))*5-1)+.7
        torch.testing.assert_close(b.lower[0, 0], expected)
        torch.testing.assert_close(b.lower, b.upper)
        self.assertTrue(b.single_route.item())
        rev = copy.deepcopy(r)
        rev.experts = nn.ModuleList(list(reversed(rev.experts)))
        with torch.no_grad(): rev.router.bias.copy_(rev.router.bias.flip(0))
        other = route_envelope(rev, lo, hi, torch.tensor([[-2.]], dtype=lo.dtype), .7)
        torch.testing.assert_close(other.lower, b.lower)

    def test_singleton_equals_dense_ibp(self):
        r = layer(experts=1)
        lo = torch.zeros(2, 2, dtype=torch.float64); hi = lo+1
        q = margin_rows(torch.tensor([0, 2]), 3, dtype=lo.dtype)
        b = route_envelope(r, lo, hi, q)
        dense = projected_interval(r.experts[0], lo, hi, q)
        self.assertTrue(torch.equal(b.lower, dense[0]))
        self.assertTrue(torch.equal(b.upper, dense[1]))
        r.k = 2
        with self.assertRaises(ValueError): route_envelope(r, lo, hi)

    def test_property_folding_retains_equal_logit_cancellation(self):
        net = affine([[1.], [1.]], [2., 0.])
        lo = torch.tensor([[-10.]], dtype=torch.float64); hi = -lo
        q = torch.tensor([[1., -1.]], dtype=lo.dtype)
        a, b = projected_interval(net, lo, hi, q)
        self.assertEqual(a.item(), 2); self.assertEqual(b.item(), 2)
        ol, ou = interval(net, lo, hi)
        self.assertLess((ol[:, 0]-ou[:, 1]).item(), 0)

    def test_candidate_box_has_no_false_exclusion(self):
        # Exhaust all vertices and ties of an independent three-score box.
        r = RoutedLayer(affine(torch.eye(3).tolist(), [0.]*3),
                        [affine([[0.]*3], [float(i)]) for i in range(3)],
                        k=2, mode='selected_softmax')
        lo = torch.tensor([[-1., 0., 0.]], dtype=torch.float64)
        hi = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
        b = route_envelope(r, lo, hi)
        for x in itertools.product(*[(float(lo[0, i]), float(hi[0, i])) for i in range(3)]):
            for k, pair in enumerate(b.pairs):
                other = next(i for i in range(3) if i not in pair)
                if min(x[pair[0]], x[pair[1]]) >= x[other]:
                    self.assertTrue(b.candidates[0, k])

    def test_gradient_and_save_restore_effective_training_step(self):
        model = SequentialTop2([layer()])
        lo = torch.tensor([[.1, .2], [.3, .4]], dtype=torch.float64); hi = lo+.01
        labels = torch.tensor([0, 1]); q = margin_rows(labels, 3, dtype=lo.dtype)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        def step(m, optimizer):
            optimizer.zero_grad()
            lb, _, b = model_bounds(m, lo, hi, q)
            clean = (lo+hi)*.5
            indices = m.layers[0].router(clean).topk(2, dim=1).indices
            loss = certified_loss(lb)+.1*route_exclusion_penalty(b, indices)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in m.parameters()))
            optimizer.step()
        before = copy.deepcopy(model.state_dict())
        step(model, opt)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        buffer = io.BytesIO()
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict()), buffer)
        buffer.seek(0); saved = torch.load(buffer, weights_only=True)
        restored = copy.deepcopy(model); restored.load_state_dict(saved['model'])
        resumed = torch.optim.AdamW(restored.parameters(), lr=.001); resumed.load_state_dict(saved['opt'])
        step(model, opt); step(restored, resumed)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters())))

    def test_train_eval_use_same_bounds_and_zero_radius(self):
        m = SequentialTop2([layer()]); x = torch.tensor([[.2, .3]], dtype=torch.float64)
        q = margin_rows(torch.tensor([1]), 3, dtype=x.dtype)
        m.train(); train = model_bounds(m, x, x, q)[0]
        m.eval(); evaluate = model_bounds(m, x, x, q)[0]
        self.assertTrue(torch.equal(train, evaluate))
        torch.testing.assert_close(train, torch.einsum('bpc,bc->bp', q, m(x)))

    def test_invalid_boxes_modes_operations_and_labels(self):
        r = layer(); lo = torch.zeros(1, 2, dtype=torch.float64); hi = lo+1
        for a, b in ((hi, lo), (lo+torch.nan, hi), (lo, hi+torch.inf)):
            with self.assertRaises(ValueError): route_envelope(r, a, b)
        r.mode = 'hard_top1'
        with self.assertRaises(ValueError): route_envelope(r, lo, hi)
        with self.assertRaises(ValueError): interval(nn.BatchNorm1d(2), lo, hi)
        with self.assertRaises(ValueError): margin_rows(torch.tensor([3]), 3, dtype=lo.dtype)
        with self.assertRaises(ValueError): model_bounds(SequentialTop2([layer(), layer()]), lo, hi)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
