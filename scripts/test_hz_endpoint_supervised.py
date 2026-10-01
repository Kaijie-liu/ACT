"""Frozen execution and receipt controls, not model performance tests."""
import copy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import load, sha, save
from scoped_proof import device_lifecycle as engine
from scripts import hz_endpoint_supervised as policy
from source_enclosure.format import identity

ROOT=None
CALLS={}


def roster():
    cfg=policy.protocol()
    return [(name,policy.spec(name)) for name in cfg['cases']]+[
        (name,policy.spec(control=name)) for name in cfg['faults']]


def fault_reached(root,s):
    name=s['control']
    if not name: return
    inv=load(root/'invocation.json'); found=[]
    for path in root.glob('*_events.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['event']=='FAULT_REACHED': found.append((path.stem.removesuffix('_events'),row))
    if len(found)!=1 or found[0][1]['fault']!=name or found[0][1]['invocation']!=inv['invocation']:
        raise ValueError('expected fault injection never reached or wrong identity')
    phase,row=found[0]; stage=load(root/(phase+'_stage.json'))
    if not stage['start_seconds']<=row['elapsed']<=stage['end_seconds']:
        raise ValueError('fault chronology')


def execute_calls(root,calls,*,invoke=None,read_terminal=load,publish=save):
    """Fail-stop batch ownership; retain all unstarted slots, never silently skip."""
    invoke=engine.supervise if invoke is None else invoke
    fixed=roster()
    for i,(name,s) in enumerate(fixed):
        begin=time.monotonic()
        result=invoke(policy,root/name,s,spec_sha256=identity(s))
        calls[name]={'begin':begin,'end':time.monotonic(),'result':result}
        publish(root/('observed_'+name+'.json'),calls[name])
        term=read_terminal(root/name/'terminal.json')
        pending=[n for n,_ in fixed[i+1:]]
        clean=(result['status']!='CLEANUP_INCOMPLETE'
               and all(policy.cleanup_confirmed(stage) for stage in term['stages']))
        publish(root/('pending_after_'+name+'.json'),{'pending':pending,'cleanup_confirmed':clean})
        if not clean:
            publish(root/'batch_stop.json',{'status':'CLEANUP_INCOMPLETE','after':name,'pending':pending})
            return False
    return True


class EndpointSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ROOT is None: raise ValueError('archive root must be configured')
        if not execute_calls(ROOT,CALLS):
            raise RuntimeError('unconfirmed cleanup: fixed batch stopped with remaining slots pending')

    def checked(self,name):
        root=ROOT/name; obs=CALLS[name]
        audit=engine.audit(policy,root,observation=obs,recheck=True)
        inv=load(root/'invocation.json'); fault_reached(root,inv['spec'])
        expected=(policy.protocol()['faults'][name][1] if name in policy.protocol()['faults'] else policy.DONE)
        self.assertEqual(audit['execution_status'],expected)
        return load(root/'terminal.json')

    def test_normal_positive(self):
        for name,count in [('separation',6),('shared',1)]:
            t=self.checked(name); a=t['accepted']
            self.assertTrue(a['positive']); self.assertTrue(a['obligations_complete'])
            self.assertEqual(len(a['checked']['results']),count)

    def test_complete_nonpositive(self):
        a=self.checked('independent')['accepted']
        self.assertFalse(a['positive']); self.assertTrue(a['obligations_complete'])
        self.assertEqual(a['checked']['aggregation']['status'],'UNKNOWN_NONPOSITIVE')

    def test_partial_unknown(self):
        a=self.checked('partial')['accepted']
        self.assertFalse(a['positive']); self.assertFalse(a['obligations_complete'])
        self.assertEqual(a['checked']['aggregation']['checked_endpoints'],4)
        self.assertEqual(a['checked']['aggregation']['missing_endpoints'],2)

    def test_producer_cutoffs(self):
        for name in ('creation_delay','partial_delay','serialization_delay'):
            self.assertIsNone(self.checked(name)['accepted'])
        root=ROOT/'partial_delay'; inv=load(root/'invocation.json')
        a=policy.recheck_prefix(root,inv,time.monotonic()+30)
        self.assertEqual((a['checked_endpoints'],a['missing_endpoints']),(2,4))

    def test_producer_errors(self):
        for name in ('proposal_exception','wrong_source','missing_endpoint'):
            self.assertIsNone(self.checked(name)['accepted'])

    def test_checker_failures(self):
        for name in ('check_exception','check_delay','missing_check'):
            self.assertIsNone(self.checked(name)['accepted'])

    def test_receiver_cutoff(self): self.assertIsNone(self.checked('receive_delay')['accepted'])

    def test_source_and_checker_binding(self):
        root=ROOT/'separation'; inv=load(root/'invocation.json'); p=load(root/'produce.json')
        q=copy.deepcopy(p['request']); q['pairs'][0]['sources']['a']['c'][0]='100'
        with self.assertRaises(ValueError): policy.validate_basis(q,inv['spec'])
        a=load(root/'terminal.json')['accepted']; inputs=dict(a['inputs'],check='0'*64)
        with self.assertRaises(ValueError): policy.receive(root,inv,inputs)
        for change in ('missing','positive','coverage'):
            agg=copy.deepcopy(a['checked']['aggregation'])
            if change=='missing': agg['results'].pop()
            elif change=='positive': agg['positive']=0
            else: agg['results'][0]['covered_gate_end_labels']=[0]
            with self.assertRaises(ValueError): policy.aggregate_rows(p,agg)

    def test_inventory_and_cost(self):
        root=ROOT/'separation'; real=engine.load
        for target in ('inventory','cost','cleanup'):
            def damaged(path,*args,**kwargs):
                data=real(path,*args,**kwargs)
                if Path(path).name=='terminal.json':
                    if target=='inventory': data['inventory'].pop('prefix_01.json')
                    elif target=='cost': data.pop('parent_seconds')
                    else: data['stages'][0]['cleanup_status']='CLEANUP_INCOMPLETE'
                return data
            with patch.object(engine,'load',side_effect=damaged), self.assertRaises((ValueError,KeyError)):
                engine.audit(policy,root,observation=CALLS['separation'])

    def test_late_api_not_accepted(self):
        root=ROOT/'separation'; inv=load(root/'invocation.json'); obs=copy.deepcopy(CALLS['separation'])
        obs['end']=inv['deadline']+.1
        with self.assertRaises(ValueError): engine.audit(policy,root,observation=obs)
        obs['result']['status']='TIMEOUT'; obs['result']['seconds']=obs['end']-inv['start']
        self.assertFalse(engine.audit(policy,root,observation=obs)['budget_acceptance_observed'])

    def test_no_observation_not_accepted(self):
        self.assertFalse(engine.audit(policy,ROOT/'separation',recheck=True)['budget_acceptance_observed'])

    def test_mandatory_call_roster(self):
        from scripts.run_hz_endpoint_supervision import validate_calls
        validate_calls(CALLS)
        bad=dict(CALLS); bad.pop('missing_check')
        with self.assertRaises(ValueError): validate_calls(bad)
        fake_calls={}; starts=[]; writes={}
        def invoke(*args,**kwargs):
            starts.append(args[1]); return {'status':'CLEANUP_INCOMPLETE'}
        def publish(path,value): writes[path.name]=copy.deepcopy(value)
        complete=execute_calls(ROOT/'simulated',fake_calls,invoke=invoke,
            read_terminal=lambda _: {'stages':[]},publish=publish)
        self.assertFalse(complete); self.assertEqual(len(starts),1)
        self.assertEqual(writes['batch_stop.json']['pending'],[n for n,_ in roster()[1:]])
        self.assertEqual(len(fake_calls)+len(writes['batch_stop.json']['pending']),14)
