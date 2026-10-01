"""Finite CPU/stub wiring tests; no CUDA initialization or real queries."""
import copy
import json
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import ROOT, load, save, sha
from scoped_proof import device_lifecycle as engine
from source_enclosure.format import identity
from scripts import hz_physical_device as policy
from scripts.hz_device_lifecycle import cleanup_confirmed

EXTRA=('scripts/test_hz_physical_device.py','scripts/run_hz_physical_device.py')
TEST_NAMES=('test_fixed_roster','test_original_bound_and_cpu_isolation','test_busy_before_cuda',
            'test_failure_deadline_and_cleanup','test_candidate_metadata_and_cost','test_admission_binding',
            'test_missing_evidence_and_late_api','test_physical_freeze_required')


def roster():
    return {n:(policy.spec(device='cuda:0' if n in ('admission_busy_stub','pre_cuda_busy_stub') else 'cpu',control=n),b,status)
            for n,(b,status) in policy.protocol()['cpu_controls'].items()}


def fault_witness(root, name):
    if name=='normal': return
    phase=('admit' if name=='admission_busy_stub' else 'check' if name=='check_delay' else
           'release' if name=='release_delay' else 'produce')
    inv=load(root/'invocation.json'); term=load(root/'terminal.json'); stages={s['phase']:s for s in term['stages']}
    if load(root/(phase+'_fault.json'))!={'control':name,'invocation':inv['invocation'],'physical_cuda':False}:
        raise ValueError('target fault not reached')
    expected='TIMEOUT' if name in ('check_delay','release_delay') else 'ERROR' if name=='producer_exception' else 'COMPLETED'
    if stages[phase]['status']!=expected: raise ValueError('wrong fault terminal')
    if name=='admission_busy_stub' and list(stages)!=['admit']: raise ValueError('busy admission ran producer')
    if name=='pre_cuda_busy_stub':
        if load(root/'produce.json')['status']!='RESOURCE_UNAVAILABLE' or load(root/'pre_cuda.json')['ready']:
            raise ValueError('pre-CUDA refusal missing')
    if (root/'cuda_intent.json').exists(): raise ValueError('CPU/stub control tried CUDA')


class PhysicalWiringControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['HZ_PHYSICAL_CONTROLS_ROOT']); cls.root.mkdir(parents=True,exist_ok=False)
        bindings={}
        for name in dict.fromkeys((*policy.FILES,*EXTRA)):
            target=cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target); bindings[name]=sha(target)
        save(cls.root/'implementation.json',bindings); cls.calls={}
        for name,(spec,budget,status) in roster().items():
            begin=time.monotonic(); result=engine.supervise(policy,cls.root/name,spec,spec_sha256=identity(spec))
            observation={'begin':begin,'end':time.monotonic(),'result':result}; cls.calls[name]=observation
            save(cls.root/(name+'_observed.json'),observation)
            if any(not cleanup_confirmed(s) for s in load(cls.root/name/'terminal.json')['stages']):
                raise RuntimeError('actual cleanup unresolved: stop controls')
        save(cls.root/'calls.json',cls.calls)

    def clone(self,name,parent='normal'):
        path=self.root/('mutation_'+name); shutil.copytree(self.root/parent,path)
        return path,copy.deepcopy(self.calls[parent])

    def test_fixed_roster(self):
        self.assertEqual(len(self.calls),6)
        for name,(spec,budget,status) in roster().items():
            with self.subTest(name=name):
                self.assertEqual(self.calls[name]['result']['status'],status)
                result=engine.audit(policy,self.root/name,observation=self.calls[name],recheck=status==policy.DONE)
                fault_witness(self.root/name,name); self.assertFalse(result['complete_moe_proof'])

    def test_original_bound_and_cpu_isolation(self):
        result=load(self.root/'normal'/'receive.json')['checked']['results']
        old=load(ROOT/'docs/hz_device_candidates_20261001_r3.json')['checked']['guarded_two_sides_positive']['bounds']
        self.assertEqual(result,old)
        for name in self.calls:
            self.assertFalse((self.root/name/'cuda_intent.json').exists())
            for s in load(self.root/name/'terminal.json')['stages']: self.assertEqual(s['cuda_visible_devices'],'')

    def test_busy_before_cuda(self):
        for name,phases in [('admission_busy_stub',['admit']),('pre_cuda_busy_stub',['admit','produce','release'])]:
            t=load(self.root/name/'terminal.json'); self.assertEqual([s['phase'] for s in t['stages']],phases)
            self.assertIsNone(t['accepted']); self.assertGreater(t['seconds'],0)
        t=load(self.root/'pre_cuda_busy_stub'/'terminal.json')
        self.assertTrue(t['release']['release_confirmed']); self.assertFalse(t['release']['next_admission_established'])

    def test_failure_deadline_and_cleanup(self):
        for name in ('producer_exception','check_delay','release_delay'):
            t=load(self.root/name/'terminal.json'); self.assertIsNone(t['accepted'])
            self.assertTrue(all(cleanup_confirmed(s) for s in t['stages']))
            self.assertGreater(t['stages'][-1]['seconds'],0)
        t=load(self.root/'producer_exception'/'terminal.json')
        self.assertTrue(t['release']['release_confirmed'])
        self.assertEqual(t['stages'][-1]['phase'],'release')
        from scripts import run_hz_physical_device as runner
        freeze=self.root/'mock-freeze.json'; save(freeze,{'controls_root':str(self.root)})
        permit={'path':str(freeze),'sha256':sha(freeze)}
        for where in ('supervise','audit'):
            target=self.root/('mock-batch-'+where)
            def fake_supervise(_policy,path,_spec,**kw):
                path.mkdir(); save(path/'partial.json',{'synthetic':True})
                if where=='supervise': raise RuntimeError('injected caller failure')
                return {'status':policy.DONE,'synthetic':True}
            with patch.object(runner,'audit_controls',return_value={}),patch.object(policy,'validate_spec'),\
                 patch.object(runner,'preflight',return_value={'ready':True,'seconds':0,'synthetic':True}),\
                 patch.object(engine,'supervise',side_effect=fake_supervise) as called,\
                 patch.object(engine,'audit',side_effect=ValueError('injected audit failure')):
                summary=runner.batch(target,permit)
            self.assertEqual(called.call_count,1); self.assertEqual(summary['status'],'STOPPED_WITH_PREFIX')
            self.assertEqual(len(summary['calls']),1); self.assertEqual(summary['pending_slots'],list(range(1,8)))
            self.assertTrue((target/'call_00'/'partial.json').exists()); self.assertIsNotNone(summary['fatal_error'])

    def test_candidate_metadata_and_cost(self):
        for field,value in [('device','cuda:0'),('cost_scope','whole request'),('execution_context_sha256','0'*64)]:
            root,_=self.clone(field); p=load(root/'produce.json'); p['candidates'][field]=value
            (root/'produce.json').write_text(json.dumps(p)); stage=load(root/'produce_stage.json')
            stage['output_sha256']=sha(root/'produce.json'); (root/'produce_stage.json').write_text(json.dumps(stage))
            with self.assertRaises(ValueError): policy.payload(root,load(root/'invocation.json'),stage['output_sha256'])
        inv=load(self.root/'pre_cuda_busy_stub'/'invocation.json'); ctx=policy.context(inv,7.25)
        self.assertEqual(ctx['deadline'],7.25); self.assertEqual(ctx['gpu_uuid'],policy.protocol()['gpu_uuid'])
        for phase in ('admit','release','check','receive'): self.assertEqual(policy.visible(inv,phase),'')
        # Pure receipt fixture: not an executed CUDA result, not a new proposal.
        root,_=self.clone('cuda-shaped'); inv=load(root/'invocation.json'); inv['spec']=policy.spec(device='cuda:0')
        p=load(root/'produce.json'); plan=load(root/'produce_plan.json'); stage=load(root/'produce_stage.json')
        from scripts.hz_physical_worker import fake_snapshot
        row=fake_snapshot(inv['spec']['gpu_uuid'],busy=False)
        row['start']=plan['begin']+.01; row['end']=plan['begin']+.02
        pre={'invocation':inv['invocation'],'simulated':False,'ready':True,'observation':row}
        p['pre_cuda_sha256']=save(root/'pre_cuda.json',pre)['sha256']
        p['cuda_intent_sha256']=save(root/'cuda_intent.json',{'invocation':inv['invocation'],
            'pre_cuda_sha256':p['pre_cuda_sha256'],'producer_pid':stage['pid'],'not_completion_evidence':True})['sha256']
        p['spec_sha256']=identity(inv['spec']); p['context']=policy.context(inv,plan['run_deadline'])
        c=p['candidates']; c.update(device='cuda:0',execution_context_sha256=identity(p['context']),
            hardware={'gpu_uuid':inv['spec']['gpu_uuid'],'name':'SIMULATED_RECEIPT_ONLY','total_memory':2**36},
            allocator_memory={'peak_allocated':1024,'peak_reserved':2048})
        for entry in c['entries']:
            cert=entry['certificate']; cert['inequality_dual']=[0]*len(cert['inequality_dual'])
            cert['equality_dual']=[0]*len(cert['equality_dual']); cert['claimed_lower_bound']=entry['zero_candidate_lower_bound']
        def publish(value):
            (root/'produce.json').write_text(json.dumps(value)); stage['output_sha256']=sha(root/'produce.json')
            (root/'produce_stage.json').write_text(json.dumps(stage)); return stage['output_sha256']
        digest=publish(p); checked=policy.check(root,inv,digest,time.monotonic()+30)
        self.assertNotEqual(checked['result']['results'],load(self.root/'normal'/'receive.json')['checked']['results'])
        for mutate in (lambda v:v['candidates']['hardware'].__setitem__('gpu_uuid','GPU-wrong'),
                       lambda v:v['context'].__setitem__('deadline',v['context']['deadline']+1),
                       lambda v:v['candidates']['allocator_memory'].__setitem__('peak_allocated',4096),
                       lambda v:v['environment'].__setitem__('cpu_threads',2),
                       lambda v:v.__setitem__('cuda_intent_sha256','0'*64)):
            bad=copy.deepcopy(p); mutate(bad)
            with self.assertRaises(ValueError): policy.payload(root,inv,publish(bad))

    def test_admission_binding(self):
        root=self.root/'pre_cuda_busy_stub'; inv=load(root/'invocation.json'); pre=load(root/'pre_cuda.json')
        for mutate in (lambda r:r.__setitem__('invocation','old'),lambda r:r.__setitem__('simulated',False),
                       lambda r:r['observation'].__setitem__('gpu_uuid','GPU-wrong'),
                       lambda r:r['observation'].__setitem__('end',float('nan'))):
            bad=copy.deepcopy(pre); mutate(bad)
            with self.assertRaises(ValueError): policy.validate_pre(root,inv,bad)

    def test_missing_evidence_and_late_api(self):
        for field in ('release.json','check.json'):
            root,obs=self.clone(field); (root/field).unlink()
            with self.assertRaises(ValueError): engine.audit(policy,root,observation=obs)
        root=self.root/'normal'; obs=copy.deepcopy(self.calls['normal']); obs['end']+=30
        with self.assertRaises(ValueError): engine.audit(policy,root,observation=obs)
        self.assertFalse(engine.audit(policy,root)['budget_acceptance_observed'])

    def test_physical_freeze_required(self):
        for device in ('cpu','cuda:0'):
            with self.assertRaisesRegex(ValueError,'execution freeze'):
                policy.validate_spec(policy.spec(device=device),None)
        with self.assertRaises(ValueError): policy.spec(device='cuda:0',control='normal')
