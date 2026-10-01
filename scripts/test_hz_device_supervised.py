"""Fixed device-supervision CPU calls and simulated GPU refusals; no CUDA."""
import copy
import json
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import ROOT, load, save, sha
from scoped_proof.supervisor import group_rss
from source_enclosure.format import identity
from scripts import hz_device_supervised as flow
from scripts import hz_device_admission as admission

TEST='scripts/test_hz_device_supervised.py'
RUNNER='scripts/run_hz_device_supervision_controls.py'


def roster():
    cfg=flow.protocol()
    result={case:(flow.specification(case),30,cfg['rss_limit'],flow.DONE) for case in cfg['cases']}
    for fault,(budget,status) in cfg['faults'].items():
        result[fault]=(flow.specification(control=fault),budget,1 if fault=='low_rss' else cfg['rss_limit'],status)
    return result


def fault_witness(root,control):
    if not control: return
    terminal=load(root/'terminal.json'); inv=load(root/'invocation.json')
    phase=('admit' if control in ('admit_delay','admission_busy_stub','admission_error_stub','launch_failure','low_rss')
           else 'check' if control in ('partial_candidate','wrong_device_metadata','check_delay','check_exception','missing_stdout')
           else 'receive' if control in ('receive_delay','late_publish','rewrite_stdout') else 'produce')
    if terminal['stages'][-1]['phase']!=phase: raise ValueError('fault not reached')
    if control in ('device_oom_stub','device_sync_stub','readback_delay_stub'):
        events=[json.loads(line) for line in (root/'produce_events.jsonl').read_text().splitlines()]
        marks=[e for e in events if e['event']=='FAULT' and e['operation']==control]
        if len(marks)!=1 or marks[0].get('physical_cuda') is not False:
            raise ValueError('device fault must be explicitly simulated')
    if control.endswith('_delay') or control in ('late_publish','missing_stdout','readback_delay_stub'):
        name='readback' if control=='readback_delay_stub' else control[:-6] if control.endswith('_delay') else control
        if load(root/(name+'_unaccepted.json'))!={'control':name,'invocation':inv['invocation'],'accepted':False}:
            raise ValueError('fault marker binding')
        if control=='late_publish' and not (root/'late_candidate_unaccepted.json').is_file(): raise ValueError('publication not reached')
    elif control in ('candidate_exception','check_exception','admission_error_stub','device_oom_stub','device_sync_stub'):
        events=[json.loads(line) for line in (root/(phase+'_events.jsonl')).read_text().splitlines()]
        if not any(e['event']=='FAULT' and e['operation']==control for e in events): raise ValueError('exception not reached')
    elif control=='admission_busy_stub':
        record=load(root/'admission.json')
        if not record['simulated'] or record['status']!='RESOURCE_UNAVAILABLE' or (root/'produce.log').exists():
            raise ValueError('simulated refusal failed')
    elif control=='partial_candidate':
        p=load(root/'payload.json')
        if (len(p['batch']['queries']),len(p['candidates']['entries']))!=(4,3): raise ValueError('partial not retained')
    elif control=='wrong_device_metadata':
        if load(root/'payload.json')['candidates']['device']!='cuda:0': raise ValueError('device mutation missing')
    elif control=='rewrite_stdout':
        if sha(root/'check.stdout')==terminal['checker_stdout_sha256']: raise ValueError('rewrite not reached')
    elif control=='launch_failure':
        if terminal['stages'][0]['pid'] is not None or 'FileNotFoundError' not in str(terminal['stages'][0]['error']):
            raise ValueError('launch failure not reached')
    elif control=='low_rss':
        if terminal['stages'][0]['sampled_peak_rss']<=inv['rss_limit']: raise ValueError('resource fault not reached')


class DeviceSupervisionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['HZ_DEVICE_SUPERVISION_ROOT'])
        if not cls.root.is_relative_to(ROOT.parent/'baseline_runs'): raise ValueError('archive scope')
        cls.root.mkdir(parents=True,exist_ok=False)
        bindings={}
        for name in (*flow.FILES,TEST,RUNNER):
            target=cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target); bindings[name]=sha(target)
        save(cls.root/'implementation.json',bindings)
        cls.calls={}
        for name,(spec,budget,rss,status) in roster().items():
            save(cls.root/(name+'_started.json'),{'expected':status,'budget':budget,'rss':rss})
            start=time.monotonic()
            try: call=flow.supervise(cls.root/name,spec,expected_request_sha256=identity(spec),budget=budget,rss_limit=rss)
            except Exception as exc: call={'harness_exception':repr(exc),'seconds_observed':time.monotonic()-start}
            save(cls.root/(name+'_observed.json'),call); cls.calls[name]=call
        save(cls.root/'calls.json',cls.calls)

    def clone(self,name,parent='guarded_two_sides_positive'):
        target=self.root/name; shutil.copytree(self.root/parent,target)
        call=copy.deepcopy(self.calls[parent]); call['root']=str(target)
        return target,call

    def rebind(self,root,call,modify):
        t=load(root/'terminal.json'); modify(t)
        (root/'terminal.json').write_text(json.dumps(t))
        r=load(root/'receipt.json'); r['terminal_sha256']=sha(root/'terminal.json')
        (root/'receipt.json').write_text(json.dumps(r))
        f=load(root/'finish.json'); f['receipt_sha256']=sha(root/'receipt.json')
        (root/'finish.json').write_text(json.dumps(f)); call['finish_sha256']=sha(root/'finish.json')

    def test_fixed_roster_and_fault_reach(self):
        self.assertEqual(len(self.calls),22)
        for name,(spec,budget,rss,status) in roster().items():
            with self.subTest(name=name):
                self.assertEqual(self.calls[name]['status'],status)
                checked=flow.audit(self.root/name,self.calls[name],recheck=status==flow.DONE)
                self.assertFalse(checked['gpu_execution']); fault_witness(self.root/name,spec['control'])
                for s in load(self.root/name/'terminal.json')['stages']:
                    if s['pid'] is not None: self.assertEqual(group_rss(s['pid'])[1],[])

    def test_exact_bounds_unchanged(self):
        old=load(ROOT/'docs/hz_device_candidates_20261001_r3.json')['checked']
        for case in flow.protocol()['cases']:
            a=load(self.root/case/'accepted.json')
            self.assertEqual(a['result']['results'],old[case]['bounds'])
            self.assertFalse(a['complete_moe_proof']); self.assertFalse(a['gpu_execution'])

    def test_refusal_no_producer_or_zeroed_cost(self):
        for name in ('admission_busy_stub','admission_error_stub'):
            t=load(self.root/name/'terminal.json')
            self.assertEqual([s['phase'] for s in t['stages']],['admit'])
            self.assertGreater(t['stage_seconds'],0)
            self.assertIsNone(t['accepted']); self.assertFalse((self.root/name/'produce.log').exists())

    def test_partial_and_missing_stdout_keep_cost(self):
        for name in ('partial_candidate','missing_stdout','wrong_device_metadata'):
            t=load(self.root/name/'terminal.json')
            self.assertEqual(t['required'],4); self.assertIsNone(t['accepted'])
            self.assertEqual(t['stages'][-1]['phase'],'check'); self.assertGreater(t['stages'][-1]['seconds'],0)

    def test_candidate_cost_is_nested_not_added(self):
        for case in flow.protocol()['cases']:
            root=self.root/case; t=load(root/'terminal.json'); p=load(root/'payload.json')
            self.assertLessEqual(p['candidates']['cost_seconds']['total'],t['stages'][1]['seconds'])
            self.assertAlmostEqual(t['stage_seconds']+t['overhead_before_publication_seconds'],t['seconds_before_publication'])

    def test_cpu_isolation_and_cuda_disabled(self):
        for name in self.calls:
            for stage in load(self.root/name/'terminal.json')['stages']:
                self.assertEqual(stage['cuda_visible_devices'],'')
        with self.assertRaisesRegex(ValueError,'physical CUDA'):
            flow.specification(device='cuda:0',gpu_uuid='GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc')

    def test_request_hash_and_admission_anchor(self):
        root=self.root/'guarded_two_sides_positive'; spec=load(root/'spec.json'); t=load(root/'terminal.json')
        for bad in (None,'','0'*64):
            with self.assertRaises(ValueError):
                flow.receive(root,spec,t['invocation_sha256'],t['payload_sha256'],t['checker_stdout_sha256'],time.monotonic()+30,bad)
        with self.assertRaises(ValueError): flow.supervise(self.root/'bad-hash',spec,expected_request_sha256='0'*64)
        self.assertFalse((self.root/'bad-hash').exists())

    def test_api_observation_required_and_late_rejected(self):
        root=self.root/'guarded_two_sides_positive'
        self.assertFalse(flow.audit(root)['complete_support_execution'])
        c=copy.deepcopy(self.calls['guarded_two_sides_positive']); c['seconds']=301
        with self.assertRaises(ValueError): flow.audit(root,c)

    def test_rebound_cost_roster_and_environment_rejected(self):
        mutations=[lambda t:t.__setitem__('required',3),lambda t:t.__setitem__('stage_seconds',0),
                   lambda t:t['stages'].pop(),lambda t:t['accepted']['result']['results'].pop(),
                   lambda t:t.__setitem__('admission_sha256','0'*64)]
        for i,fn in enumerate(mutations):
            root,call=self.clone('mutation-'+str(i)); self.rebind(root,call,fn)
            with self.assertRaises(ValueError): flow.audit(root,call)
        root,call=self.clone('env-mutation')
        def mutate(t):
            t['stages'][1]['cuda_visible_devices']='0'
            (root/'produce_stage.json').write_text(json.dumps(t['stages'][1]))
        self.rebind(root,call,mutate)
        with self.assertRaises(ValueError): flow.audit(root,call)

    def test_admission_parse_busy_malformed_and_uuid(self):
        gpu='GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc'
        self.assertTrue(admission.parse(f'{gpu}, 10000, 100, 0','',gpu)['ready'])
        for used,util,process in ((100,90,''),(9000,0,''),(100,0,f'{gpu}, 4')):
            self.assertFalse(admission.parse(f'{gpu}, 10000, {used}, {util}',process,gpu)['ready'])
        for text,process in ((f'{gpu}, 1, 3, 0',''),(f'{gpu}, 1, 0, N/A',''),(f'{gpu}, 1, 0, 0','Unknown')):
            with self.assertRaises(ValueError): admission.parse(text,process,gpu)
        with self.assertRaises(ValueError): admission.parse(f'{gpu}, 10000, 0, 0','','GPU-wrong')

    def test_admission_raw_binding_and_chronology(self):
        root=self.root/'admission_busy_stub'; r=load(root/'admission.json'); spec=load(root/'spec.json'); inv=load(root/'invocation.json')
        admission.validate(r,inv['invocation'],spec,flow.phase_deadline(inv,'admit'))
        for field,value in (('status','READY'),('invocation','other'),('simulated',False)):
            v=copy.deepcopy(r); v[field]=value
            with self.assertRaises(ValueError): admission.validate(v,inv['invocation'],spec,flow.phase_deadline(inv,'admit'))
        v=copy.deepcopy(r); v['observations'][0]['compute_pids']=[]
        with self.assertRaises(ValueError): admission.validate(v,inv['invocation'],spec,flow.phase_deadline(inv,'admit'))
        v=copy.deepcopy(r); v['observations'][0]['end']=v['deadline']+1
        with self.assertRaises(ValueError): admission.validate(v,inv['invocation'],spec,flow.phase_deadline(inv,'admit'))

    def test_final_and_terminal_deadline_single_samples(self):
        with patch.object(flow.time,'monotonic',side_effect=[10.99,11.01]) as clock:
            status,result,elapsed=flow.terminal_observation(flow.DONE,{'ok':True},10.,11.)
        self.assertEqual(clock.call_count,1); self.assertEqual(status,flow.DONE)
        root=self.root/'final-clock'; root.mkdir()
        with patch.object(flow.time,'monotonic',side_effect=[11.01,11.02]):
            call=flow.finish_observation(root,flow.DONE,1.,10.,'0'*64,'injected')
        self.assertEqual(call['status'],'TIMEOUT'); self.assertFalse(call['complete_support_execution'])

    def test_failed_prefix_cannot_be_removed_or_rebound(self):
        for filename,parent in (('admission.json','candidate_exception'),('payload.json','partial_candidate')):
            root,call=self.clone('failed-prefix-'+filename,parent)
            value=load(root/filename); value['polluted']=True
            (root/filename).write_text(json.dumps(value))
            with self.assertRaises(ValueError): flow.audit(root,call)
        root,call=self.clone('failed-prefix-anchor','device_oom_stub')
        self.rebind(root,call,lambda t:t.__setitem__('admission_sha256','0'*64))
        with self.assertRaises(ValueError): flow.audit(root,call)
        root,call=self.clone('failed-prefix-both-anchors','device_oom_stub')
        def remove_anchors(t):
            t['stages'][0].pop('output_sha256'); t['admission_sha256']=None
            (root/'admit_stage.json').write_text(json.dumps(t['stages'][0]))
        self.rebind(root,call,remove_anchors)
        with self.assertRaises(ValueError): flow.audit(root,call)

    def test_admission_recheck_is_inside_phase_deadline(self):
        original_clock=time.monotonic; original=flow.bound_admission; shift=[0.]
        root=self.root/'admission_clock_control'; spec=flow.specification()
        def clock(): return original_clock()+shift[0]
        def checked(*args,**kwargs):
            value=original(*args,**kwargs); shift[0]=4.; return value
        with patch.object(flow.time,'monotonic',side_effect=clock),patch.object(flow,'bound_admission',side_effect=checked):
            call=flow.supervise(root,spec,expected_request_sha256=identity(spec),budget=30)
        self.assertEqual(call['status'],'TIMEOUT')
        t=load(root/'terminal.json'); self.assertEqual(len(t['stages']),1)
        self.assertEqual(t['stages'][0]['worker_status'],'COMPLETED'); self.assertIsNone(t['accepted'])
        flow.audit(root,call)
        save(self.root/'admission_clock_observed.json',{'synthetic_clock_injection':True,'call':call})

    def test_publication_overrun_is_not_success(self):
        original_clock=time.monotonic; original=flow.save; shift=[0.]
        root=self.root/'publication_clock_control'; spec=flow.specification()
        def clock(): return original_clock()+shift[0]
        def publish(path,value):
            record=original(path,value)
            if Path(path).name=='finish.json': shift[0]=30.
            return record
        with patch.object(flow.time,'monotonic',side_effect=clock),patch.object(flow,'save',side_effect=publish):
            call=flow.supervise(root,spec,expected_request_sha256=identity(spec),budget=30)
        self.assertEqual(call['status'],'TIMEOUT'); self.assertFalse(flow.audit(root,call)['complete_support_execution'])
        save(self.root/'publication_clock_observed.json',{'synthetic_clock_injection':True,'call':call})


if __name__=='__main__': unittest.main()
