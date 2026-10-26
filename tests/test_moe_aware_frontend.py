"""M-B prior-guard pruning and complete-history integration, no real queue."""
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
import moe_aware_frontend as F
from act.back_end.moe.multilayer_external import verify_histories
from test_gate_decoupling import fixture
from test_gate_cascade import tensors, outcome


def stub_support(graph, *args, deadline, use_guards, iterations):
    k, g = graph.objective_count, graph.guard_count
    values = [-1.]*k if graph.next_router is not None else [1.]*k
    return dict(source_id=graph.source_id, history=[list(p) for p in graph.history], cells=[],
        lower=values, zero_lower=values, objective_count=k, guard_count=g, use_guards=use_guards,
        selected_beta=[[0.]*g for _ in range(k)], beta_nonzero=0, beta_count=k*g,
        formal_SAFE=False, complete_MoE=False)


class Frontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

    def subject(self):
        model, rows, thresholds = fixture(2, classes=3)
        data = tensors(rows, thresholds)
        source = dict(model=dict(architecture=dict(layers=2, experts=3, classes=3)),
                      sample=dict(identity='fixed_synthetic'), selection_root='synthetic')
        return model, data, source

    def test_full_coverage_scoped_prefix_and_binding_rejection(self):
        model, data, request = self.subject(); deadline = time.monotonic()+15
        original_calls = []
        def root(m, prefix, *args):
            self.assertEqual(prefix, ()); original_calls.append(prefix)
            return {(i, j): 1. for i in range(3) for j in range(3) if i != j}
        frontend = F.GuardDual(model, data, request, 'a'*64, support=stub_support, router_fallback=root)
        result = verify_histories(model, **data, pruning=True, deadline=deadline,
            verify_graph=lambda *args: self.fail('positive guarded bounds must close without solver fallback'),
            bound_router=frontend.prefix, history_solver=frontend.history)
        self.assertEqual((result['status'], result['verified_histories']), ('POSITIVE', 9))
        self.assertEqual(original_calls, [()])
        self.assertEqual(result['prefix_bounds'][0]['guarantee'], 'WHOLE_BOX_CROWN_NUMERICAL_UPPER')
        self.assertEqual(result['prefix_bounds'][1]['guarantee'], 'GUARD_CONDITIONAL_CROWN_NUMERICAL_UPPER')
        self.assertTrue(all(v == 1 for _, _, v in result['prefix_bounds'][1]['upper']))
        F.audit_result(result, request, 'a'*64, deadline)
        bad = copy.deepcopy(result); bad['prefix_bounds'][1]['upper'][0][2] = -1.
        with self.assertRaises(ValueError): F.audit_result(bad, request, 'a'*64, deadline)
        bad_request = copy.deepcopy(request); bad_request['sample']['identity'] = 'different_input'
        with self.assertRaises(ValueError): F.audit_result(result, bad_request, 'a'*64, deadline)

    def test_nonpositive_history_falls_back_on_same_budget(self):
        model, data, request = self.subject(); deadline = time.monotonic()+10
        def support(graph, *args, **kwargs):
            result = stub_support(graph, *args, **kwargs)
            result['lower'] = result['zero_lower'] = [-1.]*graph.objective_count
            return result
        seen = []
        def fallback(history, records, end):
            self.assertEqual(end, deadline); seen.append(history)
            return dict(outcome(2), status='unknown'), None
        frontend = F.GuardDual(model, data, request, 'a'*64,
            support=support, history_fallback=fallback)
        value, _ = frontend.history(((0, 1), (0, 1)), [], deadline)
        self.assertEqual(value['status'], 'unknown'); self.assertFalse(value['guard_closed'])
        self.assertEqual(len(seen), 1)
        with self.assertRaises(TimeoutError): frontend.history(((0, 1), (0, 1)), [], time.monotonic()-1)

    def test_invalid_guard_multiplier_never_becomes_pruning(self):
        model, data, request = self.subject()
        def invalid(graph, *args, **kwargs):
            result = stub_support(graph, *args, **kwargs)
            result['selected_beta'][0][0] = -1.
            return result
        frontend = F.GuardDual(model, data, request, 'a'*64, support=invalid)
        with self.assertRaises(ValueError):
            frontend.prefix(model, ((0, 1),), data['center'], data['lower'], data['upper'], time.monotonic()+10)

    @unittest.skipUnless(os.environ.get('GUARD_NATIVE') == '1', 'isolated author engine synthetic integration')
    def test_native_prefix_guard_and_complete_request(self):
        from act.pipeline.moe.multilayer_training import build
        import multilayer_stage1 as stage
        import genbab_cpu_ready as backend
        from scoped_proof.io import sha
        from gate_decoupling_worker import digest
        config = dict(input_dim=1, layers=2, experts=3, classes=3, hidden_output=1,
                      router_hidden=[], expert_hidden=[])
        model = build(config, 0).double().eval()
        with torch.no_grad():
            a, b = model.layers
            a.router[-1].weight.copy_(torch.tensor([[1.], [0.], [0.]]))
            a.router[-1].bias.copy_(torch.tensor([0., 2., 0.]))
            b.router[-1].weight.copy_(torch.tensor([[-1.], [0.], [0.]]))
            b.router[-1].bias.copy_(torch.tensor([-.1, 2., 0.]))
            for expert in a.experts:
                expert[-1].weight.fill_(1.); expert[-1].bias.zero_()
            for expert in b.experts:
                expert[-1].weight.zero_(); expert[-1].bias.copy_(torch.tensor([2., 0., 0.]))
        rows = torch.tensor([[1., -1., 0.], [1., 0., -1.]])
        data = tensors(rows, torch.zeros(2)); data.update(lower=data['center']-1, upper=data['center']+1)
        deadline = float(os.environ['GUARD_CONTROL_DEADLINE'])
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root = Path(tmp); checkpoint = root/'synthetic.pt'
            torch.save(dict(format='act-sequential-top2-v1', architecture=config, seed=0,
                state_dict=model.state_dict(), synthetic_control=True), checkpoint)
            torch.save(data, root/'input.pt')
            request = dict(model=dict(id='untrained_synthetic', checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint), architecture=config, seed=0),
                sample=dict(file='input.pt', sha256=sha(root/'input.pt'),
                    tensors={k:stage.fingerprint(v) for k,v in data.items()}),
                selection_root=str(root), arm='genbab_pruned', guard_dual=True)
            stage.ARTIFACT_ROOT, stage.ARTIFACT_LIMIT = root, 128*2**20
            completed = stage.verify(root, request, model, data, deadline, torch)
            result = completed['result']
            self.assertEqual(result['status'], 'POSITIVE'); self.assertTrue(result['proof_complete'])
            self.assertEqual(result['expected_histories'], 9); self.assertFalse(result['formal_SAFE'])
            record = next(r for r in result['prefix_bounds'] if r['prefix'] == [[0, 1]])
            conditional = next(v for i, j, v in record['upper'] if (i, j) == (0, 2))
            # Prior pair01 imposes x>=0, hence next r0-r2=-x-.1<=-.1.
            # The unconstrained box maximum is .9, so strict-negative pruning
            # cannot come from the same unconditional next-margin bound.
            self.assertLess(conditional, 0.); self.assertGreaterEqual(conditional, -.1-1e-7)
            self.assertGreater(record['guard_support']['beta_nonzero'], 0)
            F.audit_result(result, request, digest(backend.source_identity()), deadline)
            print('M-B complete-request synthetic', dict(conditional_upper=conditional,
                exact_whole_box_upper=.9, exact_conditional_upper=-.1,
                pruned=result['pruned_histories'], verified=result['verified_histories']), flush=True)


if __name__ == '__main__':
    unittest.main()
