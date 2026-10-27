"""M-A full-history routing, same-deadline fallback and fixed native controls."""
import copy
import json
import os
import signal
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
import torch
from act.back_end.moe.multilayer_external import verify_histories
import gate_cascade as C
from scoped_proof.io import load, sha
from test_gate_decoupling import fixture


def tensors(rows, thresholds):
    x = torch.zeros((1, 1), dtype=torch.float64)
    return dict(center=x, lower=x-.25, upper=x+.25, rows=rows, thresholds=thresholds)


def outcome(k):
    return dict(status='verified', actual_raw_spec=[([[float(i == j) for j in range(k)]], [0.]) for i in range(k)])


class CascadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

    def test_full_coverage_and_remaining_deadline_fallback(self):
        model, rows, th = fixture(2, classes=3); data = tensors(rows, th)
        deadline = time.monotonic()+10; phases, fallback = [], []
        source = dict(model={}, sample={}, selection_root='synthetic')
        def runner(folder, source, history, records, k, backend, end, room, **kw):
            self.assertEqual(end, deadline); self.assertLessEqual(kw['phase_started'], time.monotonic())
            phases.append(tuple(history))
            # First local result closed; all other histories need fallback.
            closed = len(phases) == 1
            return dict(status='POSITIVE' if closed else 'TIMEOUT', accepted=closed,
                evidence=dict(closed=True, full_model_replays=[]) if closed else None,
                source_id='a'*64, seconds=.01)
        def coupled(graph, lower, upper, k, end):
            self.assertEqual(end, deadline); fallback.append(end)
            return outcome(k), None
        cascade = C.GateCascade(model, data, source, ROOT.parent/'results', 'b'*64,
            lambda: C.W.CAP, phase_runner=runner, coupled=coupled)
        def bounds(*args):
            return {(i, j): 10. for i in range(3) for j in range(3) if i != j}
        result = verify_histories(model, **data, pruning=True, deadline=deadline,
            verify_graph=coupled, bound_router=bounds, history_solver=cascade)
        self.assertEqual(result['expected_histories'], 9)
        self.assertEqual((result['verified_histories'], len(phases), len(fallback)), (9, 9, 8))
        self.assertEqual(result['status'], 'POSITIVE')
        self.assertTrue(result['proof_complete']); self.assertFalse(result['formal_SAFE'])

    def test_partial_or_expired_phase_never_closes_request(self):
        model, rows, th = fixture(2, classes=3); data = tensors(rows, th)
        source = dict(model={}, sample={}, selection_root='synthetic')
        def runner(*args, **kwargs):
            return dict(status='TIMEOUT', accepted=False, evidence=None, source_id='a'*64, seconds=.1)
        calls = []
        def unknown(graph, lower, upper, k, deadline):
            calls.append(deadline); return dict(outcome(k), status='unknown'), None
        cascade = C.GateCascade(model, data, source, ROOT.parent/'results', 'b'*64,
            lambda: C.W.CAP, phase_runner=runner, coupled=unknown)
        result = verify_histories(model, **data, pruning=False, deadline=time.monotonic()+5,
            verify_graph=unknown, history_solver=cascade)
        self.assertEqual(result['verified_histories'], 0); self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(len(calls), 9)
        with self.assertRaises(TimeoutError): cascade(((0, 1), (0, 1)), [], time.monotonic()-1)

    def test_nonfinite_gate_range_falls_back_without_spawn(self):
        from test_gate_decoupling import prefix_records
        h = ((0, 1), (0, 1)); records = prefix_records(h)
        records[0]['upper'][0][2] = None
        source = dict(model=dict(architecture=dict(experts=3)))
        result = C.run_phase(ROOT.parent/'results'/'must_not_be_created', source, h, records, 2,
                             'a'*64, time.monotonic()+10, C.W.CAP)
        self.assertEqual(result['status'], 'GATE_RANGE_UNAVAILABLE'); self.assertFalse(result['accepted'])

    def test_owned_phase_cutoff_and_wrong_session(self):
        from scoped_proof.owned_bounded import execute
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            before = time.monotonic(); root = Path(tmp)
            result = execute([sys.executable, '-I', '-c', 'import time; time.sleep(5)'],
                root/'child.log', run_deadline=before+.25, cleanup_deadline=before+2,
                env=dict(os.environ), rss_limit=None, owner_session=os.getsid(0), development_policy=C.POLICY)
            self.assertEqual(result['status'], 'TIMEOUT')
            self.assertEqual(result['cleanup_status'], 'LEADER_REAPED_NO_LIVE_GROUP')
            self.assertFalse(result['remaining_group']['live']); self.assertFalse(result['rss_acceptance_gate'])
            self.assertLess(result['seconds'], 2.)
            with self.assertRaises(ValueError):
                execute([], root/'bad.log', run_deadline=before+2, cleanup_deadline=before+2,
                    env={}, rss_limit=None, owner_session=os.getsid(0)+1000000, development_policy=C.POLICY)

    def test_deadline_alarm_is_timeout_only_after_worker_cutoff_and_cleanup(self):
        receipt=dict(status='ERROR',returncode=-signal.SIGALRM,error=None,
            cleanup_status='LEADER_REAPED_NO_LIVE_GROUP',exit_observed_at=100.,finished_at=100.1)
        self.assertTrue(C.deadline_alarm(receipt,100.))
        for change in (dict(exit_observed_at=99.9),dict(exit_observed_at=None),
            dict(exit_observed_at=float('nan')),dict(finished_at=99.9),dict(returncode=-signal.SIGSEGV),
            dict(returncode=-signal.SIGKILL),dict(returncode=1),dict(status='COMPLETED'),
            dict(error='observation failed'),dict(cleanup_status='CLEANUP_INCOMPLETE')):
            self.assertFalse(C.deadline_alarm(dict(receipt,**change),100.))

    def test_native_default_alarm_keeps_owned_cleanup_and_discards_result(self):
        from scoped_proof.owned_bounded import execute
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            before=time.monotonic();cutoff=before+.4;root=Path(tmp)
            code=('import signal,time,sys; signal.signal(signal.SIGALRM,signal.SIG_DFL); '
                'signal.setitimer(signal.ITIMER_REAL,max(.001,float(sys.argv[1])-time.monotonic())); '
                'time.sleep(10)')
            receipt=execute([sys.executable,'-I','-S','-c',code,str(cutoff)],root/'alarm.log',
                run_deadline=before+2,cleanup_deadline=before+4,env=dict(os.environ),rss_limit=None,
                owner_session=os.getsid(0),development_policy=C.POLICY)
            self.assertEqual(receipt['returncode'],-signal.SIGALRM)
            self.assertTrue(C.deadline_alarm(receipt,cutoff))
            self.assertFalse(receipt['remaining_group']['live'])
            self.assertLess(time.monotonic()-before,4.)

    def test_alarm_phase_discards_even_published_positive_and_keeps_errors(self):
        from test_gate_decoupling import prefix_records
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root=Path(tmp);history=((0,1),(0,1));records=prefix_records(history)
            source=dict(model=dict(architecture=dict(experts=3)))
            for name,offset,cleanup in (('late',.1,'LEADER_REAPED_NO_LIVE_GROUP'),
                ('early',-.1,'LEADER_REAPED_NO_LIVE_GROUP'),('unclean',.1,'CLEANUP_INCOMPLETE')):
                child=root/name
                def fake_execute(*args,**kwargs):
                    q=load(child/'phase_request.json');cutoff=q['worker_deadline']
                    self.assertEqual(kwargs['run_deadline'],q['phase_deadline']-4)
                    self.assertEqual(kwargs['cleanup_deadline'],q['phase_deadline']-2)
                    # Deliberately invalid positive-looking files must not be read.
                    (child/'outcome.json').write_text(json.dumps(dict(closed=True)))
                    (child/'published.json').write_text(json.dumps(dict(clock=cutoff-1)))
                    return dict(status='ERROR',returncode=-signal.SIGALRM,error=None,
                        cleanup_status=cleanup,exit_observed_at=cutoff+offset,finished_at=cutoff+.2)
                with patch.object(C,'execute',side_effect=fake_execute):
                    args=(child,source,history,records,2,'a'*64,time.monotonic()+40,C.W.CAP)
                    if name=='late':
                        result=C.run_phase(*args,phase_seconds=12)
                        self.assertEqual(result['status'],'TIMEOUT')
                        self.assertFalse(result['accepted']);self.assertIsNone(result['evidence'])
                        self.assertLess(result['phase_deadline'],result['request_deadline'])
                    else:
                        with self.assertRaises((RuntimeError,ValueError)):C.run_phase(*args,phase_seconds=12)

    @unittest.skipUnless(os.environ.get('CASCADE_NATIVE') == '1', 'isolated author engine synthetic integration')
    def test_native_complete_request_and_phase_receipt(self):
        from act.pipeline.moe.multilayer_training import build
        from multilayer_stage1 import fingerprint, external_audit
        import multilayer_stage1 as stage
        config = dict(input_dim=1, layers=2, experts=3, classes=3, hidden_output=1,
                      router_hidden=[], expert_hidden=[])
        model = build(config, 0).double().eval()
        with torch.no_grad():
            for depth, layer in enumerate(model.layers):
                router = layer.router[-1]
                router.weight.copy_(torch.tensor([[.2], [-.2], [0.]]))
                router.bias.copy_(torch.tensor([2., 2., -4.]))
                for expert in layer.experts:
                    expert[-1].weight.fill_(.1); expert[-1].bias.fill_(.3 if depth == 0 else 0.)
                    if depth == 1: expert[-1].bias[0] = 2.
        rows = torch.tensor([[1., -1., 0.], [1., 0., -1.]])
        data = tensors(rows, torch.tensor([0., .1]))
        deadline = float(os.environ['CASCADE_CONTROL_DEADLINE'])
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root = Path(tmp)
            checkpoint = root/'synthetic.pt'
            torch.save(dict(format='act-sequential-top2-v1', architecture=config, seed=0,
                state_dict=model.state_dict(), synthetic_control=True), checkpoint)
            torch.save(data, root/'input.pt')
            source = dict(model=dict(id='untrained_synthetic', checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint), architecture=config, seed=0),
                sample=dict(file='input.pt', sha256=sha(root/'input.pt'), tensors={k:fingerprint(v) for k,v in data.items()}),
                selection_root=str(root), arm='genbab_pruned', gate_cascade=True)
            # Use the same documented per-worker accounting state as worker().
            stage.ARTIFACT_ROOT, stage.ARTIFACT_LIMIT = root, C.W.CAP
            completed = stage.verify(root, source, model, data, deadline, torch)
            result = completed['result']
            self.assertEqual(result['status'], 'POSITIVE'); self.assertEqual(result['expected_histories'], 9)
            self.assertEqual(result['verified_histories'], 1); self.assertEqual(result['pruned_histories'], 8)
            phase = result['histories'][0]['outcome']['cascade']
            self.assertFalse(phase['used_fallback']); self.assertTrue(result['cascade_files'])
            self.assertEqual(external_audit(result, [3, 3], 2)['status'], 'PASS')
            folder = root/'v2_history_0000'; request = load(folder/'phase_request.json')
            value = load(folder/'outcome.json')['value']
            from act.back_end.moe import gate_decoupling as G
            domain = G.domain_from_prefix_bounds(((0, 1), (0, 1)), [3, 3], request['prefix_bounds'],
                source_id=request['source_id'], expected_source_id=request['source_id'])
            C.receive_phase(folder, request, value, domain, deadline)
            bad = copy.deepcopy(value); bad['endpoints'].pop()
            with self.assertRaises(ValueError): C.receive_phase(folder, request, bad, domain, deadline)
            with self.assertRaises(TimeoutError): C.receive_phase(folder, request, value, domain, time.monotonic()-1)
            print('M-A synthetic whole request', json.dumps(dict(verified=1, pruned=8,
                phase=phase, source='untrained fixed weights; no data')), flush=True)


if __name__ == '__main__':
    unittest.main()
