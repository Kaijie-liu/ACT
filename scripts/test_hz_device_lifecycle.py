"""Twelve fixed CPU executions; mutations reuse saved bytes without solving."""
import copy
import json
import os
from pathlib import Path
import shutil
import time
import unittest

from scoped_proof.io import ROOT, load, save, sha
from scoped_proof.owned_bounded import observe
from source_enclosure.format import identity
from scripts import hz_device_lifecycle as life
from scripts.hz_device_lifecycle_audit import audit
from scripts.hz_device_admission import parse
from scripts.hz_device_release import validate

EXTRA=('scripts/hz_device_lifecycle_audit.py','scripts/test_hz_device_lifecycle.py',
       'scripts/run_hz_device_lifecycle_controls.py')


def roster():
    cfg,_=life.protocol()
    return {**{c:(life.spec(c),cfg['normal_budget_seconds'],life.DONE) for c in cfg['cases']},
            **{f:(life.spec(control=f),b,status) for f,(b,status) in cfg['faults'].items()}}


def fault_witness(root, control):
    if not control: return
    inv=load(root/'invocation.json'); term=load(root/'terminal.json')
    if control=='cleanup_unconfirmed':
        if ([s['phase'] for s in term['stages']]!=['admit','produce']
                or term['stages'][-1].get('cleanup_unconfirmed_stub') is not True
                or term['stages'][-1]['status']!='COMPLETED'
                or term['stages'][-1]['returncode']!=0
                or term['stages'][-1]['cleanup_status']!='LEADER_REAPED_NO_LIVE_GROUP'):
            raise ValueError('cleanup stop stub not reached')
        life.payload(root,inv,term['stages'][-1]['output_sha256'])
        return
    phase='produce' if control in ('producer_exception','readback_stall') else 'release'
    if load(root/(phase+'_fault.json'))!={'invocation':inv['invocation'],'control':control,'accepted':False}:
        raise ValueError('fault reach marker')
    events=[json.loads(line) for line in (root/(phase+'_events.jsonl')).read_text().splitlines()]
    if not any(e['event']=='FAULT' and e['operation']==control and e['physical_cuda'] is False for e in events):
        raise ValueError('fault evidence missing')
    stages={s['phase']:s for s in term['stages']}
    expected=('ERROR' if control in ('producer_exception','release_probe_error') else
              'TIMEOUT' if control in ('readback_stall','release_deadline') else 'COMPLETED')
    if stages[phase]['status']!=expected: raise ValueError('wrong fault phase terminal')
    if phase=='release' and stages['produce']['status']!='COMPLETED': raise ValueError('producer failed before release fault')
    if expected!='COMPLETED' and (root/(phase+'.json')).exists(): raise ValueError('unexpected completed fault output')
    if control in ('release_owned_stays','release_wrong_request'):
        record=load(root/'release.json'); producer=stages['produce']; plan=load(root/'release_plan.json')
        kw=dict(invocation=inv['invocation'],gpu_uuid=inv['spec']['gpu_uuid'],producer_sha=sha(root/'produce_stage.json'),
                producer_pid=producer['pid'],after=inv['start']+producer['end_seconds'],deadline=plan['run_deadline'],simulated=True)
        if control=='release_owned_stays':
            if (any(row['compute_pids']!=[producer['pid']] for row in record['observations'])
                    or validate(record,**kw)['release_confirmed']): raise ValueError('owned-PID fault not reached')
        else:
            if record['invocation']!='stale-other-request': raise ValueError('wrong-request fault not reached')
            record['invocation']=inv['invocation']
            if not validate(record,**kw)['release_confirmed']: raise ValueError('another release fault masked identity test')


class LifecycleControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['HZ_LIFECYCLE_ROOT'])
        if not cls.root.is_relative_to(ROOT.parent/'baseline_runs'): raise ValueError('archive scope')
        cls.root.mkdir(parents=True,exist_ok=False)
        bindings={}
        for name in dict.fromkeys((*life.FILES,*EXTRA)):
            target=cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target); bindings[name]=sha(target)
        save(cls.root/'implementation.json',bindings)
        cls.calls={}
        for name,(spec,budget,status) in roster().items():
            save(cls.root/(name+'_started.json'),{'spec':spec,'budget':budget,'expected':status})
            start=time.monotonic()
            result=life.supervise(cls.root/name,spec,request_sha256=identity(spec),budget=budget)
            record={'begin':start,'end':time.monotonic(),'result':result}
            save(cls.root/(name+'_observed.json'),record); cls.calls[name]=record
            if any(s['cleanup_status']=='CLEANUP_INCOMPLETE' for s in load(cls.root/name/'terminal.json')['stages']):
                raise RuntimeError('actual cleanup unconfirmed: no later call admitted')
        save(cls.root/'calls.json',cls.calls)

    def clone(self, name, parent='guarded_two_sides_positive'):
        root=self.root/('mutation_'+name); shutil.copytree(self.root/parent,root)
        return root,copy.deepcopy(self.calls[parent])

    def rebind(self, root, observation, modify):
        term=load(root/'terminal.json'); modify(term)
        (root/'terminal.json').write_text(json.dumps(term))
        finish=load(root/'finish.json'); finish['terminal_sha256']=sha(root/'terminal.json')
        (root/'finish.json').write_text(json.dumps(finish))
        observation['result']['finish_sha256']=sha(root/'finish.json')

    def test_fixed_calls_and_fault_reach(self):
        self.assertEqual(len(self.calls),12)
        for name,(spec,budget,status) in roster().items():
            with self.subTest(name=name):
                record=self.calls[name]; self.assertEqual(record['result']['status'],status)
                result=audit(self.root/name,observation=record,recheck=status==life.DONE)
                self.assertFalse(result['physical_cuda']); fault_witness(self.root/name,spec['control'])
                for stage in load(self.root/name/'terminal.json')['stages']:
                    if stage['pid'] is not None: self.assertEqual(observe(stage['pid'])['live'],[])

    def test_original_exact_bounds_unchanged(self):
        reference=load(ROOT/'docs/hz_device_candidates_20261001_r3.json')['checked']
        for case in life.protocol()[0]['cases']:
            current=load(self.root/case/'receive.json')['checked']['results']
            self.assertEqual(current,reference[case]['bounds'])

    def test_failed_producer_still_releases_and_charges(self):
        for name in ('producer_exception','readback_stall'):
            term=load(self.root/name/'terminal.json')
            self.assertEqual([s['phase'] for s in term['stages']],['admit','produce','release'])
            self.assertTrue(term['release']['release_confirmed']); self.assertIsNone(term['accepted'])
            self.assertGreater(term['stages'][-1]['seconds'],0)
        for name in ('live','returncode'):
            root,obs=self.clone('cleanup-'+name,'producer_exception')
            def mutate(t):
                stage=t['stages'][1]
                if name=='live': stage['remaining_group']['live']=[stage['pid']]
                else: stage['returncode']=None
                (root/'produce_stage.json').write_text(json.dumps(stage))
                t['inventory']['produce_stage.json']=sha(root/'produce_stage.json')
            self.rebind(root,obs,mutate)
            with self.assertRaises(ValueError): audit(root,observation=obs)

    def test_unconfirmed_release_prevents_later_launch(self):
        for name in ('release_owned_stays','release_probe_error','release_deadline','release_wrong_request'):
            term=load(self.root/name/'terminal.json')
            self.assertEqual([s['phase'] for s in term['stages']],['admit','produce','release'])
            self.assertIsNone(term['accepted']); self.assertGreater(term['stages'][-1]['seconds'],0)
            self.assertFalse((self.root/name/'check_plan.json').exists())

    def test_foreign_tenant_is_not_owned_release_failure(self):
        term=load(self.root/'release_foreign_tenant'/'terminal.json')
        self.assertTrue(term['release']['release_confirmed'])
        self.assertFalse(term['release']['next_admission_ready'])
        self.assertTrue(term['release']['simulated'])
        self.assertFalse(term['release']['physical_driver_release_proved'])
        self.assertFalse(term['release']['next_admission_established'])

    def release_fixture(self):
        root=self.root/'release_foreign_tenant'; inv=load(root/'invocation.json')
        producer=load(root/'produce_stage.json'); plan=load(root/'release_plan.json')
        kw=dict(invocation=inv['invocation'],gpu_uuid=inv['spec']['gpu_uuid'],producer_sha=sha(root/'produce_stage.json'),
                producer_pid=producer['pid'],after=inv['start']+producer['end_seconds'],deadline=plan['run_deadline'],simulated=True)
        return load(root/'release.json'),kw

    def test_release_binding_chronology_and_nan(self):
        record,kw=self.release_fixture()
        mutations=[lambda r:r.__setitem__('invocation','wrong'),lambda r:r.__setitem__('gpu_uuid','GPU-wrong'),
                   lambda r:r.__setitem__('producer_sha256','0'*64),lambda r:r.__setitem__('simulated',False),
                   lambda r:r['observations'].pop(),lambda r:r['observations'][0].__setitem__('end',float('nan')),
                   lambda r:r['observations'][1].__setitem__('start',r['after']-1),
                   lambda r:r['observations'][1].__setitem__('end',r['deadline'])]
        for mutate in mutations:
            bad=copy.deepcopy(record); mutate(bad)
            with self.assertRaises(ValueError): validate(bad,**kw)

    def test_owned_pid_reappearance_ambiguity_fails_closed(self):
        record,kw=self.release_fixture(); row=record['observations'][-1]
        row['process_text']=f"{kw['gpu_uuid']}, {kw['producer_pid']}"
        row.update(parse(row['gpu_text'],row['process_text'],kw['gpu_uuid']))
        result=validate(record,**kw)
        self.assertFalse(result['release_confirmed'])
        for row in record['observations']:
            row['process_text']=''; row['gpu_text']=f"{kw['gpu_uuid']}, 97887, 34, 0"
            row.update(parse(row['gpu_text'],row['process_text'],kw['gpu_uuid']))
        result=validate(record,**kw)
        self.assertTrue(result['release_confirmed'])
        self.assertTrue(result['last_snapshot_meets_admission_thresholds'])
        self.assertIsNone(result['next_admission_ready'])
        self.assertFalse(result['next_admission_established'])

    def test_nested_cost_and_api_deadline(self):
        for name in life.protocol()[0]['cases']:
            root=self.root/name; term=load(root/'terminal.json'); p=load(root/'produce.json')
            self.assertLessEqual(p['candidates']['cost_seconds']['total'],term['stages'][1]['seconds'])
            self.assertAlmostEqual(term['seconds'],term['stage_seconds']+term['parent_seconds'])
        root=self.root/'guarded_two_sides_positive'
        self.assertFalse(audit(root)['budget_acceptance_observed'])
        bad=copy.deepcopy(self.calls['guarded_two_sides_positive']); bad['end']+=30
        with self.assertRaises(ValueError): audit(root,observation=bad)
        for pending in life.PENDING:
            self.assertEqual(life.finalize_status(pending,31,30),pending)
        for field in ('cost_scope','cost_seconds'):
            root,_=self.clone('candidate-'+field); p=load(root/'produce.json')
            if field=='cost_scope': p['candidates'][field]='all costs'
            else:
                delta=1000; costs=p['candidates'][field]; costs['total']+=delta; costs['other']+=delta
            (root/'produce.json').write_text(json.dumps(p))
            stage=load(root/'produce_stage.json'); stage['output_sha256']=sha(root/'produce.json')
            (root/'produce_stage.json').write_text(json.dumps(stage))
            with self.assertRaises(ValueError): life.payload(root,load(root/'invocation.json'),stage['output_sha256'])

    def test_rebound_terminal_cost_and_deadline_rejected(self):
        for i,mutate in enumerate((lambda t:t.__setitem__('stage_seconds',0),
                                   lambda t:t.__setitem__('parent_seconds',float('nan')),
                                   lambda t:t['stages'][1].__setitem__('run_deadline',999999999),
                                   lambda t:t['accepted']['checked']['results'].pop())):
            root,obs=self.clone('cost'+str(i)); self.rebind(root,obs,mutate)
            with self.assertRaises(ValueError): audit(root,observation=obs)

    def test_missing_release_and_rewritten_checker_rejected(self):
        for filename in ('release.json','check.json'):
            root,obs=self.clone(filename)
            value=load(root/filename); value['polluted']=True
            (root/filename).write_text(json.dumps(value))
            with self.assertRaises(ValueError): audit(root,observation=obs)
        root,obs=self.clone('missing-release'); (root/'release.json').unlink()
        with self.assertRaises(ValueError): audit(root,observation=obs)

    def test_failed_prefix_removal_rejected(self):
        root,obs=self.clone('failed-prefix','producer_exception'); (root/'produce_fault.json').unlink()
        with self.assertRaises(ValueError): audit(root,observation=obs)

    def test_source_identity_and_cuda_refusal(self):
        spec=life.spec(); spec['device']='cuda:0'
        with self.assertRaises(ValueError): life.supervise(self.root/'forbidden',spec,request_sha256=identity(spec))
        self.assertFalse((self.root/'forbidden').exists())
        root,obs=self.clone('source')
        inv=load(root/'invocation.json'); inv['sources']['scoped_proof/owned_bounded.py']='0'*64
        (root/'invocation.json').write_text(json.dumps(inv))
        with self.assertRaises(ValueError): audit(root,observation=obs)
