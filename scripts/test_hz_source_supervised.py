"""Frozen whole-source process and receipt controls, no real-model intake."""
import copy
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import load,save
from scoped_proof import device_lifecycle as engine
from source_enclosure.format import identity
from scripts import hz_source_supervised as policy
from scripts.test_hz_endpoint_supervised import fault_reached

ROOT = None
CALLS = {}


def roster():
    cfg=policy.protocol()
    return [(name,policy.spec(name)) for name in cfg['cases']]+[
        (name,policy.spec(control=name)) for name in cfg['faults']]


def execute_calls(root,calls,*,invoke=None,read_terminal=load,publish=save):
    invoke=engine.supervise if invoke is None else invoke
    fixed=roster()
    for i,(name,s) in enumerate(fixed):
        pending=[n for n,_ in fixed[i+1:]]
        publish(root/('started_'+name+'.json'),{'name':name,'spec_sha256':identity(s),'pending':pending})
        begin=time.monotonic()
        try:
            result=invoke(policy,root/name,s,spec_sha256=identity(s))
            calls[name]={'begin':begin,'end':time.monotonic(),'result':result}
            publish(root/('observed_'+name+'.json'),calls[name])
            term=read_terminal(root/name/'terminal.json')
            clean=(result['status']!='CLEANUP_INCOMPLETE'
                   and all(policy.cleanup_confirmed(stage) for stage in term['stages']))
        except BaseException as exc:
            error={'begin':begin,'end':time.monotonic(),'api_error':repr(exc),'cleanup_confirmed':False}
            calls.setdefault(name,error)
            publish(root/('api_error_'+name+'.json'),error)
            clean=False
        publish(root/('pending_after_'+name+'.json'),{'pending':pending,'cleanup_confirmed':clean})
        if not clean:
            publish(root/'batch_stop.json',{'status':'CLEANUP_INCOMPLETE','after':name,'pending':pending})
            return False
    return True


class SourceSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ROOT is None: raise ValueError('fixed archive root required')
        if not execute_calls(ROOT,CALLS): raise RuntimeError('batch stopped; unresolved ownership remains')

    def checked(self,name):
        root=ROOT/name; inv=load(root/'invocation.json')
        result=engine.audit(policy,root,observation=CALLS[name],recheck=True)
        fault_reached(root,inv['spec'])
        expected=policy.protocol()['faults'][name][1] if name in policy.protocol()['faults'] else policy.DONE
        self.assertEqual(result['execution_status'],expected)
        return load(root/'terminal.json')

    def test_normal_source_positive(self):
        for name,count in [('weighted_sign',6),('tied_partial_reuse',18),('unresolved_sign',12)]:
            a=self.checked(name)['accepted']
            self.assertTrue(a['declared_source_positive']); self.assertTrue(a['obligations_complete'])
            self.assertFalse(a['complete_moe_proof']); self.assertFalse(a['real_model_proof'])
            self.assertEqual(len(a['checked']['results']),count)

    def test_complete_nonpositive(self):
        a=self.checked('unsafe_tied')['accepted']
        self.assertFalse(a['declared_source_positive']); self.assertTrue(a['obligations_complete'])
        self.assertEqual(a['checked']['aggregation']['status'],'UNKNOWN_NONPOSITIVE')

    def test_partial_unknown(self):
        a=self.checked('partial')['accepted']; agg=a['checked']['aggregation']
        self.assertFalse(a['declared_source_positive']); self.assertFalse(a['obligations_complete'])
        self.assertEqual((agg['required'],agg['checked_endpoints'],agg['missing_endpoints']),(18,15,3))

    def test_source_prefix(self):
        root=ROOT/'upstream_delay'; inv=load(root/'invocation.json')
        self.assertIsNone(self.checked('upstream_delay')['accepted'])
        prefix=policy.recheck_prefix(root,inv,time.monotonic()+30)
        self.assertEqual(prefix['source_graphs_checked'],1)
        self.assertEqual(prefix['checked_endpoints'],0); self.assertFalse(prefix['output_accepted'])
        self.assertFalse(list(root.glob('endpoint_*.json')))
        real=policy.load
        def changed(path,*args,**kwargs):
            p=real(path,*args,**kwargs)
            if Path(path).name=='lowering_01.json': p['input']['hz']['c'][0]='1'
            return p
        with patch.object(policy,'load',side_effect=changed), self.assertRaises(ValueError):
            policy.recheck_prefix(root,inv,time.monotonic()+30)
        # Finalized prefixes are not ignorable merely because an earlier file
        # vanished. These views leave the frozen artifacts untouched.
        complete=ROOT/'weighted_sign'; complete_inv=load(complete/'invocation.json')
        real_exists,real_glob=Path.exists,Path.glob
        def no_source(path):
            return False if path==complete/'prefix_source.json' else real_exists(path)
        with patch.object(Path,'exists',no_source), self.assertRaisesRegex(ValueError,'lacks source'):
            policy.recheck_prefix(complete,complete_inv,time.monotonic()+30)
        def no_lowering(path,pattern):
            return iter(()) if path==complete and pattern=='lowering_*.json' else real_glob(path,pattern)
        with patch.object(Path,'glob',no_lowering), self.assertRaisesRegex(ValueError,'lacks lowering'):
            policy.recheck_prefix(complete,complete_inv,time.monotonic()+30)
        with patch.object(Path,'exists',no_source), patch.object(Path,'glob',no_lowering), self.assertRaisesRegex(ValueError,'lacks source'):
            policy.recheck_prefix(complete,complete_inv,time.monotonic()+30)

    def test_producer_cutoffs(self):
        for name in ('creation_delay','partial_delay','serialization_delay'):
            self.assertIsNone(self.checked(name)['accepted'])
        root=ROOT/'partial_delay'; inv=load(root/'invocation.json')
        p=policy.recheck_prefix(root,inv,time.monotonic()+30)
        self.assertEqual((p['checked_endpoints'],p['aggregation']['missing_endpoints']),(2,4))
        self.assertFalse(p['output_accepted'])

    def test_producer_errors(self):
        for name in ('upstream_exception','proposal_exception','wrong_source','missing_endpoint'):
            self.assertIsNone(self.checked(name)['accepted'])
        root=ROOT/'missing_endpoint'; inv=load(root/'invocation.json')
        with self.assertRaises(ValueError): policy.recheck_prefix(root,inv,time.monotonic()+30)

    def test_checker_failures(self):
        for name in ('check_exception','check_delay','missing_check'):
            self.assertIsNone(self.checked(name)['accepted'])

    def test_receiver_cutoff(self): self.assertIsNone(self.checked('receive_delay')['accepted'])

    def test_source_and_checker_binding(self):
        root=ROOT/'weighted_sign'; inv=load(root/'invocation.json'); p=load(root/'produce.json')
        a=load(root/'terminal.json')['accepted']; inputs=dict(a['inputs'],check='0'*64)
        with self.assertRaises(ValueError): policy.receive(root,inv,inputs)
        for key,value in [('partial',True),('duties',4),('endpoints',7)]:
            changed=dict(inv['spec'],**{key:value})
            with self.assertRaises(ValueError): policy.execution_coverage(p,changed)
        for change in ('missing','positive','coverage','source','trust'):
            agg=copy.deepcopy(a['checked']['aggregation'])
            if change=='missing': agg['results'].pop()
            elif change=='positive': agg['positive']=0
            elif change=='coverage': agg['results'][0]['covered_gate_end_labels']=[0]
            elif change=='source': agg['source_lowering_checked']=False
            else: agg['remaining_trust']=[]
            with self.assertRaises(ValueError): policy.aggregate_rows(p,agg)
        real=policy.load
        def modified(path,*args,**kwargs):
            doc=real(path,*args,**kwargs)
            if Path(path).name=='prefix_source.json': doc['request']['margin']='1'
            return doc
        with patch.object(policy,'load',side_effect=modified), self.assertRaises(ValueError):
            policy.payload(root,inv,a['inputs']['produce'])

    def test_inventory_and_cost(self):
        root=ROOT/'weighted_sign'; real=engine.load
        for target in ('inventory','cost','cleanup'):
            def damaged(path,*args,**kwargs):
                data=real(path,*args,**kwargs)
                if Path(path).name=='terminal.json':
                    if target=='inventory': data['inventory'].pop('lowering_01.json')
                    elif target=='cost': data.pop('parent_seconds')
                    else: data['stages'][0]['cleanup_status']='CLEANUP_INCOMPLETE'
                return data
            with patch.object(engine,'load',side_effect=damaged), self.assertRaises((ValueError,KeyError)):
                engine.audit(policy,root,observation=CALLS['weighted_sign'])
        from scripts.run_hz_source_supervision import event_costs
        rows=event_costs(ROOT/'upstream_delay',load(ROOT/'upstream_delay'/'terminal.json'))
        self.assertEqual(rows['produce'][-1]['status'],'RIGHT_CENSORED')
        term=load(root/'terminal.json'); read=Path.read_text
        for damage in ('empty','finish','serialization','imports'):
            def missing_events(path,*args,**kwargs):
                text=read(path,*args,**kwargs)
                if path!=root/'produce_events.jsonl': return text
                if damage=='empty': return ''
                token={'finish':'WORKER_COMPLETE','serialization':'final_serialization','imports':'numerical_imports'}[damage]
                return '\n'.join(line for line in text.splitlines() if token not in line)
            with patch.object(Path,'read_text',missing_events),self.assertRaises(ValueError):
                event_costs(root,term)

    def test_late_api_not_accepted(self):
        root=ROOT/'weighted_sign'; inv=load(root/'invocation.json'); obs=copy.deepcopy(CALLS['weighted_sign'])
        obs['end']=inv['deadline']+.1
        with self.assertRaises(ValueError): engine.audit(policy,root,observation=obs)
        obs['result']['status']='TIMEOUT'; obs['result']['seconds']=obs['end']-inv['start']
        self.assertFalse(engine.audit(policy,root,observation=obs)['budget_acceptance_observed'])

    def test_no_observation_not_accepted(self):
        self.assertFalse(engine.audit(policy,ROOT/'weighted_sign',recheck=True)['budget_acceptance_observed'])

    def test_mandatory_call_roster(self):
        from scripts.run_hz_source_supervision import validate_calls
        validate_calls(CALLS)
        bad=dict(CALLS); bad.pop('missing_check')
        with self.assertRaises(ValueError): validate_calls(bad)
        for error in (False,True):
            calls={}; starts=[]; writes={}
            def invoke(*args,**kwargs):
                starts.append(args[1])
                if error: raise RuntimeError('API interruption')
                return {'status':'CLEANUP_INCOMPLETE'}
            def publish(path,value): writes[path.name]=copy.deepcopy(value)
            complete=execute_calls(ROOT/'simulated',calls,invoke=invoke,
                                   read_terminal=lambda _: {'stages':[]},publish=publish)
            self.assertFalse(complete); self.assertEqual(len(starts),1)
            self.assertIn('started_weighted_sign.json',writes)
            self.assertEqual(len(calls)+len(writes['batch_stop.json']['pending']),17)
            if error: self.assertIn('api_error_weighted_sign.json',writes)
