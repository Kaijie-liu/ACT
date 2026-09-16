"""Actual analytic upstream plus adversarial execution controls; no CIFAR verification."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import tempfile
import time
import unittest

from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest, original_bytes
from upstream_portable.flow import supervise, audit, review_candidate, phase_state

OBSERVATIONS = []


@contextmanager
def request_case():
    import torch
    from act.back_end.moe import OutputMoEFactoryConfig, GateKind
    from act.pipeline.moe.test_route_complexity_schedule import model, config
    from act.pipeline.moe.staged_verifier import _model_state_identity, _tensor_identity
    from scripts.test_general_evidence import fixture
    net = model(((-.2,0.,-2.), (1.,0.,-2.), (2.,0.,-2.)))
    fc = OutputMoEFactoryConfig(input_shape=(2,), num_classes=3, num_experts=3,
          top_k=2, gate=GateKind.SELECTED_SOFTMAX, router_hidden=(), expert_hidden=(), seed=7)
    x = torch.full((1,2), .5, dtype=torch.float64)
    tensors = {'center':x, 'lower':x-.01, 'upper':x+.01}
    with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
        base = Path(tmp); cp=base/'model.pt'; tp=base/'input.pt'; cfg=base/'config.json'
        torch.save({'format':'act-output-moe-v1','factory_config':asdict(fc),'state_dict':net.state_dict()},cp)
        torch.save(tensors,tp); save_new(cfg,config('monolithic_f0'))
        sample={'dataset_index':-1,'label':0,**{k:_tensor_identity(v) for k,v in tensors.items()}}
        r=fixture()[1]; r.update(epsilon=.01,model_state=_model_state_identity(net),
                               **{k:sample[k] for k in tensors})
        req={'subject':{'checkpoint':str(cp),'checkpoint_sha256':digest(cp.read_bytes()),'model_state':r['model_state']},
             'sample':sample,'epsilon':.01,'config':{'path':str(cfg),'sha256':digest(cfg.read_bytes())},
             'tensors':{'path':str(tp),'sha256':digest(tp.read_bytes())},'evidence_request':r,
             'head':'analytic','method':'evidence'}
        yield base,req


class FullFlowControls(unittest.TestCase):
    def test_actual_two_arm_upstream_and_tamper(self):
        results=[]
        with request_case() as (base,req):
            for arm in ('double_check','single_check'):
                root=base/arm; start=time.monotonic()
                out=supervise(req,arm,root,started=start)
                self.assertEqual(out['status'],'CHECKED_CONDITIONAL', (out,(root/'driver.log').read_text()))
                checked=audit(root); self.assertTrue(checked['complete_independent_check'])
                c=review_candidate(root)
                self.assertEqual([r['name'] for r in c['stages']],['capture','propose'])
                job=read(root/'tail/job.json'); self.assertEqual(job['started_monotonic'],start)
                self.assertGreater(job['upstream_elapsed_at_entry'],sum(r['elapsed_seconds'] for r in c['stages']))
                result=read(root/'tail/check.log')['result'];results.append(result)
                self.assertEqual((root/'tail/precheck.log').exists(),arm=='double_check')
                OBSERVATIONS.append({'arm':arm,'status':out['status'],'seconds':out['observed_seconds'],
                    'capture_seconds':c['stages'][0]['elapsed_seconds'],
                    'proposal_seconds':c['stages'][1]['elapsed_seconds'],
                    'tail_entry_seconds':job['upstream_elapsed_at_entry']})
                for rel in ('source/request.json','source/manifest.json','source/handoff.json','capture_stage.json'):
                    p=root/rel; raw=p.read_bytes()
                    try:
                        p.write_bytes(raw+b' ')
                        with self.assertRaises(ValueError): audit(root)
                    finally: p.write_bytes(raw)
                # Mutate a claimed upstream cost AND re-sign the candidate; audit still reconstructs it.
                candidate=root/'candidate.json'; raw=candidate.read_bytes(); outer=root/'outer.json'; ov=outer.read_bytes()
                pub=root/'publication.json'; pv=pub.read_bytes()
                try:
                    bad=read(candidate);bad['stages'][0]['elapsed_seconds']=0;candidate.write_bytes(original_bytes(bad))
                    v=read(outer);v['candidate_sha256']=digest(candidate.read_bytes());outer.write_bytes(original_bytes(v))
                    p=read(pub);p['outer_sha256']=digest(outer.read_bytes());pub.write_bytes(original_bytes(p))
                    with self.assertRaises(ValueError):audit(root)
                finally: candidate.write_bytes(raw);outer.write_bytes(ov);pub.write_bytes(pv)
                save_new(root/'publication_timeout.json',{'status':'TIMEOUT','complete_independent_check':False})
                self.assertEqual(audit(root)['status'],'TIMEOUT')
            self.assertEqual(results[0],results[1])

    def test_expired_clock_no_capture_and_no_retry(self):
        with request_case() as (base,req):
            root=base/'expired';v=supervise(req,'single_check',root,started=time.monotonic()-299)
            self.assertEqual(v['status'],'TIMEOUT'); self.assertEqual(audit(root)['status'],'TIMEOUT')
            self.assertFalse((root/'source').exists())
            with self.assertRaises(FileExistsError):supervise(req,'single_check',root,started=time.monotonic())

    def test_invalid_model_binding_is_error_not_proof(self):
        with request_case() as (base,req):
            req['subject']['checkpoint_sha256']='0'*64;root=base/'bad'
            result=supervise(req,'single_check',root,started=time.monotonic())
            self.assertEqual(result['status'],'ERROR');self.assertFalse(audit(root)['complete_independent_check'])
            self.assertFalse((root/'tail').exists())

    def test_process_and_arm_controls(self):
        for killed,code,end,expected in ((True,0,1,'TIMEOUT'),(False,3,1,'TIMEOUT'),
            (False,0,298,'TIMEOUT'),(False,1,1,'ERROR'),(False,0,2,'COMPLETED')):
            self.assertEqual(phase_state({'killed':killed,'return_code':code},end),expected)
        with self.assertRaises(ValueError):supervise({},'unknown',ROOT/'unused',started=time.monotonic())
        with self.assertRaises(ValueError):supervise({},'single_check',ROOT/'unused',started=time.monotonic()+100)

    def test_roster_rotation_error_stop_and_denominator(self):
        from upstream_portable.study import jobs,loop
        roster=jobs([{'dataset_index':i} for i in range(4)])
        self.assertEqual(len(roster),8)
        self.assertEqual([r['arm'] for r in roster[:4]],['double_check','single_check','single_check','double_check'])
        calls=[];published=[]
        def run(job):
            calls.append(job)
            return {'status':'TIMEOUT' if len(calls)==1 else 'ERROR','complete_independent_check':False}
        rows=loop(roster,run,published.append)
        self.assertEqual(len(calls),2);self.assertEqual(len(rows),8);self.assertEqual(rows,published)
        self.assertEqual([r['status'] for r in rows][2:],['NOT_RUN_AFTER_ERROR']*6)

    def test_deadline_between_upstream_and_tail_is_timeout(self):
        with request_case() as (base,req):
            root=base/'boundary';root.mkdir();(root/'source').mkdir()
            save_new(root/'plan.json',{'request':req,'arm':'single_check','started':1.})
            save_new(root/'source/request.json',req);stages=[]
            for i,name in enumerate(('capture','propose')):
                row={'name':name,'start_seconds':i*100.,'end_seconds':(i+1)*100.,'elapsed_seconds':100.,
                     'process':{'killed':False,'return_code':0},'state':'COMPLETED'}
                stages.append(row);save_new(root/(name+'_stage.json'),row)
            save_new(root/'candidate.json',{'status':'TIMEOUT','complete_independent_check':False,
                 'error':None,'wall_seconds':298.1,'stages':stages,'artifact_sha256':{},
                 'plan_sha256':digest((root/'plan.json').read_bytes())})
            self.assertEqual(review_candidate(root)['status'],'TIMEOUT')
            self.assertFalse((root/'tail').exists())


if __name__=='__main__': unittest.main()
