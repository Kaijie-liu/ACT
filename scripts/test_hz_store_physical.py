"""Pure wrapper controls: no candidate generation and no device observation."""
import argparse
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import load, save, sha
from scoped_source.hz_request_store import identity
from scripts import run_hz_store_physical as p

NAMES=('test_roster_and_unchanged_math','test_v2_source_and_outcome_binding',
       'test_exact_freeze_not_membership','test_refusal_stop_and_unknown',
       'test_complete_pair_comparison','test_nested_cost_and_late_observation')


class Controls(unittest.TestCase):
    def test_roster_and_unchanged_math(self):
        rows=p.roster()
        self.assertEqual([r['name'] for r in rows],['half_cpu','half_cuda','negative_cuda','negative_cpu','partial_cpu','partial_cuda'])
        self.assertEqual(len({identity(p.policy.spec(r['case'],r['device'])) for r in rows}),6)
        for r in rows:
            s=p.policy.spec(r['case'],r['device'])
            self.assertEqual(p.policy.budget(s),300); self.assertFalse(s['control'])
            self.assertEqual(s['tile_width'],8)
        self.assertEqual(p.policy.spec('half')['source'],p.policy.spec('partial')['source'])
        self.assertNotEqual(p.policy.spec('half')['source'],p.policy.spec('negative')['source'])

    def test_v2_source_and_outcome_binding(self):
        p.prerequisites(); original=p.load
        for key,value in (('status','FAIL'),('audit_version',1),('extra_tests',0),('numerical_reruns',1),
                          ('cuda_calls',1),('calls',9),('tests',11),('sources',{}),
                          ('audit_extension_source_sha256','0'*64),('extra_tests_sha256','0'*64),
                          ('calls_sha256','0'*64),('freeze_sha256','0'*64),
                          ('original_audit_failure_sha256','0'*64)):
            def altered(path,*a,**k):
                data=original(path,*a,**k)
                if Path(path).name=='audit_v2.json': data[key]=value
                return data
            with self.subTest(key=key),patch.object(p,'load',side_effect=altered),self.assertRaises(ValueError):
                p.prerequisites()

    def test_exact_freeze_not_membership(self):
        # Mutation check of the wrapper equality, separate from prerequisite tests.
        reference=dict(wrapper_controls='receipt',roster=p.roster(),request_budget_seconds=300)
        permit=dict(path=str(p.ROOT/'configs/unpublished_control.json'),sha256='1'*64)
        for mutation in ('missing','duplicate','order','extra','budget'):
            f=copy.deepcopy(reference)
            if mutation=='missing': f['roster'].pop()
            elif mutation=='duplicate': f['roster'][1]=f['roster'][0]
            elif mutation=='order': f['roster'].reverse()
            elif mutation=='extra': f['roster'].append(f['roster'][0])
            else: f['request_budget_seconds']=301
            with self.subTest(mutation=mutation),patch.object(p,'load',return_value=f),patch.object(p,'make_freeze',return_value=reference),self.assertRaises(ValueError):
                p.validate(permit)

    def test_refusal_stop_and_unknown(self):
        for outcome in ('REFUSED','API_ERROR','AUDIT_ERROR','RESOURCE_UNAVAILABLE','DEVICE_RELEASE_UNCONFIRMED',p.policy.DONE):
            with self.subTest(outcome=outcome),tempfile.TemporaryDirectory(dir=p.ROOT.parent) as tmp:
                root=Path(tmp); seen=[]
                def invoke(*args,**kwargs):
                    seen.append(args)
                    if outcome=='API_ERROR': raise RuntimeError('controlled exception')
                    return dict(status=p.policy.DONE if outcome=='AUDIT_ERROR' else outcome)
                def check(*args):
                    if outcome=='AUDIT_ERROR': raise ValueError('controlled audit refusal')
                    return dict(mathematics=dict(status='UNKNOWN_NONPOSITIVE'))
                calls=p.execute_slots(root,{},dict(ready=outcome!='REFUSED'),time.monotonic()+2400,invoke=invoke,recheck=check)
                count=0 if outcome=='REFUSED' else 6 if outcome==p.policy.DONE else 1
                self.assertEqual(len(seen),count); self.assertEqual(len(calls),count)
                if count:
                    row=load(root/f'pending_after_{count-1}.json')
                    self.assertEqual(row['pending'],list(range(count,6)))
                    self.assertEqual(row['continue_allowed'],outcome==p.policy.DONE)
        with tempfile.TemporaryDirectory(dir=p.ROOT.parent) as tmp:
            with patch.object(p.engine,'supervise',side_effect=AssertionError('must not start')):
                calls=p.execute_slots(Path(tmp),{},dict(ready=True),time.monotonic()+1,
                                      invoke=lambda *a,**k:self.fail('no request allowance'))
            self.assertFalse(calls); self.assertEqual(load(Path(tmp)/'batch_deadline.json')['next_slot'],0)
        self.wrapper('REFUSED'); self.wrapper('API_ERROR'); self.wrapper('PREFIX_ERROR')
        # Startup publication consumes the remaining allowance; no invocation.
        with tempfile.TemporaryDirectory(dir=p.ROOT.parent) as tmp:
            original=p.save; now=time.monotonic()
            def publish(path,value):
                result=original(path,value)
                if Path(path).name=='started_0.json': clock.return_value=now+101.
                return result
            with patch.object(p.time,'monotonic',return_value=now) as clock,patch.object(p,'save',side_effect=publish):
                calls=p.execute_slots(Path(tmp),{},dict(ready=True),now+400,
                    invoke=lambda *a,**k:self.fail('allowance expired before invocation'))
            self.assertEqual(len(calls),1); self.assertIsNotNone(calls[0]['error'])

    def wrapper(self,mode):
        # Exercise batch + audit plumbing using explicit non-numerical seams.
        original_slots=p.execute_slots; original_inventory=p.inventory
        f=dict(batch_sources={}); prerequisite=dict(offline_seconds=0.,rows={})
        with tempfile.TemporaryDirectory(dir=p.ROOT.parent) as tmp:
            root=Path(tmp)/'batch'
            def gate(path):
                path.mkdir(); save(path/'launch.json',dict(start=time.monotonic()))
                return dict(ready=mode!='REFUSED',seconds=0.)
            def invoke(*args,**kwargs):
                if mode=='API_ERROR': raise RuntimeError('before owned directory')
                args[1].mkdir(); return dict(status=p.policy.DONE)
            count=[0]
            def capacity(path,**kwargs):
                if mode=='PREFIX_ERROR' and kwargs.get('enforce',True):
                    count[0]+=1
                    if count[0]==2: raise ValueError('controlled second-slot archive check')
                return original_inventory(path,**kwargs)
            with patch.object(p,'validate',return_value=f),patch.object(p.controlled,'audit',return_value=prerequisite),\
                 patch.object(p.probe,'preflight',side_effect=gate),patch.object(p.probe,'audit_preflight',return_value=dict(status='PASS')),\
                 patch.object(p,'inventory',side_effect=capacity),\
                 patch.object(p,'execute_slots',side_effect=lambda *a:original_slots(*a,invoke=invoke,recheck=lambda *a:dict(status='PASS'))):
                before=time.monotonic(); p.batch(root,dict(path='stub',sha256='1'*64)); end=time.monotonic()
                observation=dict(begin=before,end=end,summary_sha256=sha(root/'summary.json'))
                save(root.with_name(root.name+'_observed.json'),observation)
                s=load(root/'summary.json')
                if mode=='REFUSED':
                    self.assertEqual(p.audit(root)['started_calls'],0); self.assertEqual(s['pending_slots'],list(range(6)))
                    original_load=p.load
                    for mutation in ('launch_before_batch','early_return','late_batch'):
                        def read(path,*a,**k):
                            data=original_load(path,*a,**k)
                            if mutation=='launch_before_batch' and Path(path).name=='launch.json': data['start']=0.
                            if Path(path).name==root.name+'_observed.json':
                                if mutation=='early_return': data['end']=before
                                elif mutation=='late_batch': data['end']=end+2401.
                            return data
                        with patch.object(p,'load',side_effect=read):
                            if mutation=='late_batch': self.assertEqual(p.audit(root)['execution_status'],'BATCH_BUDGET_EXCEEDED')
                            else:
                                with self.assertRaises(ValueError): p.audit(root)
                else:
                    self.assertEqual(s['attempted_slots'],[0]); self.assertEqual(s['pending_slots'],list(range(1,6)))
                    self.assertEqual(len(s['calls']),1)
                    self.assertEqual(p.audit(root)['status'],'FAIL')

    def pair(self):
        return dict(source='source',store='store',contract='contract',environment=p.VERSIONS,seconds=1.,
            mathematics=dict(required=1,required_endpoints=2,checked_endpoints=2,missing_endpoints=0,
                status='CHECKED_POSITIVE_DECLARED_SOURCE',results=[dict(pair=[0,1],other=1,
                    endpoints=['left','right'],lower='1/2',positive=True)]))

    def test_complete_pair_comparison(self):
        cpu=self.pair(); gpu=self.pair(); gpu['mathematics']['results'][0]['lower']='3/4'
        self.assertEqual(p.compare(cpu,gpu)['rows'][0]['difference'],'1/4')
        gpu['mathematics']['results'][0].update(lower='-1/4',positive=False)
        self.assertEqual(p.compare(cpu,gpu)['lost'],1)
        for mutation in ('source','store','contract','env','missing','pair','property','endpoints','omission'):
            gpu=self.pair()
            if mutation in ('source','store','contract'): gpu[mutation]='foreign'
            elif mutation=='env': gpu['environment']={}
            elif mutation=='missing': gpu['mathematics']['missing_endpoints']=1
            elif mutation=='omission': gpu['mathematics']['results']=[]
            else:
                row=gpu['mathematics']['results'][0]
                row[{'pair':'pair','property':'other','endpoints':'endpoints'}[mutation]]=None
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): p.compare(cpu,gpu)

    def test_nested_cost_and_late_observation(self):
        from scripts.audit_store_device_supervision_v2 import Controls as C
        c=C(); stage,events=c.sample(); rows=p.accounting.parse(events,stage)
        self.assertAlmostEqual(sum(r['seconds'] for r in rows if r['additive']),1.1)
        stage['status']='TIMEOUT'; rows=p.accounting.parse(events[:2],stage)
        self.assertTrue(all(r['seconds'] is None and not r['span_is_operation_duration_bound'] for r in rows))
        from scripts.test_hz_store_device_supervised import Controls as DeviceControls
        DeviceControls().test_late_observation()


def run(root):
    root.mkdir(parents=True,exist_ok=False); begin=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    if [t._testMethodName for t in suite]!=sorted(NAMES): raise ValueError('fixed pure controls')
    class Result(unittest.TextTestResult):
        outcomes=[]
        def startTest(self,t): self.state='PASS'; super().startTest(t)
        def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
        def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
        def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
        def stopTest(self,t): self.outcomes.append(dict(name=t._testMethodName,status=self.state)); super().stopTest(t)
    with (root/'tests.log').open('x') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(suite)
    good=result.wasSuccessful() and all(r['status']=='PASS' for r in result.outcomes) and result.testsRun==len(NAMES)
    report=dict(status='PASS' if good else 'FAIL',tests=result.testsRun,outcomes=result.outcomes,sources=p.sources(),
        log_sha256=sha(root/'tests.log'),seconds=time.monotonic()-begin,numerical_calls=0,cuda_calls=0)
    for name in report['sources']:
        dest=root/'implementation'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        import shutil
        shutil.copyfile(p.ROOT/name,dest)
    save(root/'summary.json',report); print(report['status']); return good


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('root',type=Path); args=a.parse_args()
    if not args.root.resolve().is_relative_to(p.ROOT.parent/'baseline_runs'): raise ValueError('project controls archive')
    raise SystemExit(0 if run(args.root) else 1)
