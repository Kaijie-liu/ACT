"""Any-layer margin containment, error ordering and scoped M-B -> M-A use."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
import torch
from act.back_end.moe import gate_decoupling as G
from act.back_end.moe.moe_aware_bab import margin_endpoint, margin_priority, MarginCell, closed_cells
from act.back_end.moe.multilayer_external import StaticPrefix
import gate_cascade as C
import moe_aware_frontend as F
from test_gate_decoupling import fixture, prefix_records


def domain(depth):
    margins = ((-2., 2.),)*depth
    return G.GateDomain('a'*64, ((0, 1),)*depth, margins, tuple(G.sigmoid_interval(*m) for m in margins))


class MarginControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

    def test_later_cells_preserve_original_execution_and_last_endpoint_rule(self):
        model, rows, thresholds = fixture(3, classes=3)
        d = domain(3); x = torch.zeros((1, 1))
        actual, routers = StaticPrefix(model, d.history)._run(x)
        gates = [torch.sigmoid(r[:, :1]-r[:, 1:2]) for r in routers]
        z = torch.cat((x, *gates[:-1]), 1)
        for layer in range(3):
            margin = float((routers[layer][:, 0]-routers[layer][:, 1]).detach())
            cells = closed_cells(layer, -2., 2., 2)
            selected = [c for c in cells if c.lower <= margin <= c.upper]
            self.assertTrue(selected)
            if layer == 0: self.assertEqual(len(selected), 2)  # exact tie in both closed cells
            for cell in selected:
                low, scoped = margin_endpoint(model, d, rows, thresholds, cell,
                    variant='V2_LO', input_features=1, expected_source_id=d.source_id)
                high, same = margin_endpoint(model, d, rows, thresholds, cell,
                    variant='V2_HI', input_features=1, expected_source_id=d.source_id)
                self.assertEqual(scoped, same)
                self.assertEqual(low.body.independent_gate_count, 2)
                lv, hv = low(z), high(z)
                self.assertTrue((lv[:, len(rows):] >= 0).all())
                torch.testing.assert_close(lv[:, len(rows):], hv[:, len(rows):])
                last_lo, last_hi = scoped.intervals[-1]
                theta = (gates[-1]-last_lo)/(last_hi-last_lo)
                original_property = actual@rows.T-thresholds
                torch.testing.assert_close((1-theta)*lv[:, :len(rows)]+theta*hv[:, :len(rows)], original_property)
                for j in range(3):
                    if j != layer: self.assertEqual(scoped.intervals[j], d.intervals[j])
        with self.assertRaises(ValueError):
            margin_endpoint(model, d, rows, thresholds, MarginCell(3, 0., 1.),
                variant='V2_LO', input_features=1, expected_source_id=d.source_id)

    def test_priority_includes_residual_suffix_and_constant_difference(self):
        from act.pipeline.moe.multilayer_training import build
        cfg = dict(family='residual_stem_v1', input_dim=1, hidden_output=1, classes=2,
                   layers=2, experts=3, router_hidden=[], expert_hidden=[])
        model = build(cfg, 0).double().eval()
        with torch.no_grad():
            model.stem.weight.fill_(1.); model.stem.bias.zero_()
            model.head.weight.copy_(torch.tensor([[1.], [-1.]])); model.head.bias.zero_()
            for layer, difference in zip(model.layers, (1., 4.)):
                for i, expert in enumerate(layer.experts):
                    expert[-1].weight.zero_(); expert[-1].bias.fill_(difference if i == 1 else 0.)
        d = domain(2); x = torch.zeros((1, 1)); rows = torch.tensor([[1., -1.]])
        result = margin_priority(model, d, x-1, x+1, rows, deadline=time.monotonic()+5)
        self.assertEqual(result['difference_absolute_amplitudes'], [1., 4.])
        self.assertEqual(result['layer_lipschitz'], [1., 1.])
        self.assertEqual(result['property_head_norm'], 2.)
        self.assertAlmostEqual(result['scores'][1], 4*result['scores'][0])
        self.assertEqual(result['selected_layer'], 1)
        self.assertTrue(result['heuristic_only']); self.assertFalse(result['certified_error_bound'])
        with self.assertRaises(TimeoutError):
            margin_priority(model, d, x-1, x+1, rows, deadline=time.monotonic()-1)

    def test_scoped_range_is_explicit_and_cannot_borrow_another_history(self):
        from test_moe_aware_frontend import stub_support
        model, rows, thresholds = fixture(2, classes=3); h = ((0, 1),)*2
        request = dict(model=dict(architecture=dict(layers=2, experts=3, classes=3)),
            sample=dict(identity='synthetic'), selection_root='synthetic')
        x = torch.zeros((1, 1)); data = dict(center=x, lower=x-1, upper=x+1, rows=rows, thresholds=thresholds)
        frontend = F.GuardDual(model, data, request, 'b'*64, support=stub_support)
        bound = frontend.prefix(model, h[:1], x, x-1, x+1, time.monotonic()+5)
        records = prefix_records(h)
        records[1] = dict(prefix=[[0, 1]], upper=[[a,b,v] for (a,b),v in bound.items()],
            guarantee='GUARD_CONDITIONAL_CROWN_NUMERICAL_UPPER', guard_support=bound.guard_support)
        sid = C.identity(request, h, records, 'b'*64)
        with self.assertRaises(ValueError):
            G.domain_from_prefix_bounds(h, [3, 3], records, source_id=sid, expected_source_id=sid)
        received = C.scoped_domain(G, request, h, records, sid, 'b'*64, time.monotonic()+5)
        self.assertEqual(received.margins[1], (-1., 1.))
        bad = copy.deepcopy(records); bad[1]['guard_support']['history'] = [[1, 2]]
        with self.assertRaises(ValueError):
            C.scoped_domain(G, request, h, bad, sid, 'b'*64, time.monotonic()+5)
        bad = copy.deepcopy(records); bad[1]['guard_support']['source_id'] = 'c'*64
        with self.assertRaises(ValueError):
            C.scoped_domain(G, request, h, bad, sid, 'b'*64, time.monotonic()+5)

    @unittest.skipUnless(os.environ.get('MARGIN_NATIVE') == '1', 'isolated native M-B/M-A scope integration')
    def test_native_conditional_prefix_consumed_by_owned_v2_phase(self):
        from act.pipeline.moe.multilayer_training import build
        import genbab_cpu_ready as backend
        from multilayer_stage1 import fingerprint
        from scoped_proof.io import sha
        cfg = dict(input_dim=1, layers=2, experts=3, classes=3, hidden_output=1,
                   router_hidden=[], expert_hidden=[])
        model = build(cfg, 0).double().eval()
        with torch.no_grad():
            for layer in model.layers:
                layer.router[-1].weight.copy_(torch.tensor([[1.], [0.], [0.]]))
                layer.router[-1].bias.copy_(torch.tensor([0., 2., -4.]))
                for expert in layer.experts:
                    expert[-1].weight.zero_(); expert[-1].bias.fill_(.3)
                    if expert[-1].out_features == 3: expert[-1].bias.copy_(torch.tensor([2., 0., 0.]))
        x = torch.zeros((1, 1)); h = ((0, 1),)*2
        data = dict(center=x, lower=x-1, upper=x+1, rows=torch.tensor([[1., -1., 0.], [1., 0., -1.]]), thresholds=torch.zeros(2))
        deadline = float(os.environ['MARGIN_CONTROL_DEADLINE'])
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root = Path(tmp); checkpoint = root/'synthetic.pt'
            torch.save(dict(format='act-sequential-top2-v1', architecture=cfg, seed=0,
                state_dict=model.state_dict(), synthetic_control=True), checkpoint)
            torch.save(data, root/'input.pt')
            request = dict(model=dict(id='untrained_synthetic', checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint), architecture=cfg, seed=0),
                sample=dict(file='input.pt', sha256=sha(root/'input.pt'), tensors={k:fingerprint(v) for k,v in data.items()}),
                selection_root=str(root))
            engine = C.W.digest(backend.source_identity())
            frontend = F.GuardDual(model, data, request, engine)
            records = []
            for depth in range(2):
                bound = frontend.prefix(model, h[:depth], x, x-1, x+1, deadline)
                record = dict(prefix=[list(p) for p in h[:depth]], upper=[[a,b,v] for (a,b),v in bound.items()],
                    guarantee='WHOLE_BOX_CROWN_NUMERICAL_UPPER')
                if depth:
                    record.update(guarantee='GUARD_CONDITIONAL_CROWN_NUMERICAL_UPPER', guard_support=bound.guard_support)
                records.append(record)
            phase = C.run_phase(root/'phase', request, h, records, 2, engine, deadline, 128*2**20)
            self.assertTrue(phase['accepted']); self.assertEqual(phase['status'], 'POSITIVE')
            self.assertTrue(phase['evidence']['closed']); self.assertEqual(len(phase['evidence']['endpoints']), 2)
            print('conditional prefix -> V2', dict(status=phase['status'], phase_seconds=phase['seconds'],
                conditional_source_retained=records[1]['guarantee'], trained_calls=0), flush=True)


if __name__ == '__main__':
    unittest.main()
