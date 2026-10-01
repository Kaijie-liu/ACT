"""Finite supervisor controls. All calls, including faults, retain their files."""
import copy
from fractions import Fraction
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
from scripts import hz_batch_support_supervised as flow


def roster():
    cfg = flow.protocol()
    out = {case:(flow.specification(case),cfg['normal_control_seconds'],cfg['rss_limit'],flow.DONE)
           for case in cfg['cases']}
    for fault,(budget,status) in cfg['faults'].items():
        out[fault] = (flow.specification(control=fault),budget,32*2**20 if fault=='low_rss' else cfg['rss_limit'],status)
    return out


def fault_witness(root, control):
    """A matching ERROR/TIMEOUT is insufficient unless injection was reached."""
    if not control: return
    inv=load(root/'invocation.json'); terminal=load(root/'terminal.json')
    phase = ('check' if control in ('partial_candidate','wrong_query','check_delay','check_exception','missing_stdout','wrong_invocation')
             else 'receive' if control in ('rewrite_stdout','receive_delay','late_publish') else 'produce')
    if terminal['stages'][-1]['phase']!=phase: raise ValueError('fault stopped before target phase')
    if control.endswith('_delay') or control in ('missing_stdout','late_publish'):
        name=control[:-6] if control.endswith('_delay') else control
        marker=load(root/(name+'_unaccepted.json'))
        if marker!={'invocation':inv['invocation'],'control':name,'accepted':False}:
            raise ValueError('fault marker binding')
        if control=='late_publish' and not (root/'late_candidate_unaccepted.json').is_file():
            raise ValueError('late candidate not reached')
    elif control.endswith('_exception'):
        events=[json.loads(line) for line in (root/(phase+'_events.jsonl')).read_text().splitlines()]
        if not any(e['event']=='FAULT' and e['operation']==control for e in events):
            raise ValueError('exception injection not reached')
    elif control in ('partial_candidate','wrong_query','wrong_invocation'):
        payload=load(root/'payload.json')
        if control=='partial_candidate' and (len(payload['batch']['queries']),len(payload['candidates']['entries']))!=(4,3):
            raise ValueError('partial candidate injection not reached')
        if control=='wrong_query' and identity(payload['batch'])==load(root/'spec.json')['batch_sha256']:
            raise ValueError('query mutation missing')
        if control=='wrong_query' and payload['batch']['queries'][0]['side']!='max': raise ValueError('query mutation missing')
        if control=='wrong_invocation' and payload['invocation']==inv['invocation']: raise ValueError('invocation mutation missing')
    elif control=='rewrite_stdout':
        if sha(root/'check.stdout')==terminal['checker_stdout_sha256']: raise ValueError('stdout mutation not reached')
    elif control=='descendant':
        child=load(root/'descendant_control.json')
        if child['invocation']!=inv['invocation'] or type(child['pid']) is not int:
            raise ValueError('descendant injection not reached')
        if 'live descendant' not in str(terminal['stages'][0]['error']): raise ValueError('descendant exit not detected')
    elif control=='launch_failure':
        stage=terminal['stages'][0]
        if (stage['pid'] is not None or stage['executable']!=str(root/'nonexistent-python')
                or 'FileNotFoundError' not in str(stage['error'])):
            raise ValueError('launch failure not reached')
    elif control=='low_rss':
        if terminal['stages'][0]['sampled_peak_rss']<=inv['rss_limit']: raise ValueError('resource stop not reached')


class SupervisionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['HZ_BATCH_SUPERVISION_ROOT'])
        if not cls.root.is_relative_to(ROOT.parent/'baseline_runs'): raise ValueError('test archive path')
        cls.root.mkdir(parents=True,exist_ok=False)
        cls.files = (*flow.FILES, 'scripts/test_hz_batch_support_supervised.py', 'scripts/run_hz_batch_supervision_controls.py')
        bindings = {}
        for name in cls.files:
            target=cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target); bindings[name]=sha(target)
        save(cls.root/'implementation.json',bindings)
        cls.calls = {}
        for name,(spec,budget,rss,expected) in roster().items():
            save(cls.root/(name+'_started.json'),{'expected':expected,'budget':budget,'rss':rss})
            begin=time.monotonic()
            try:
                call=flow.supervise(cls.root/name,spec,expected_request_sha256=identity(spec),budget=budget,rss_limit=rss)
            except Exception as exc:
                call={'harness_exception':repr(exc),'seconds_observed':time.monotonic()-begin}
            cls.calls[name]=call
            save(cls.root/(name+'_observed.json'),call)
        save(cls.root/'calls.json',cls.calls)

    def clone(self,name,parent='guarded_two_sides_positive'):
        target=self.root/name
        shutil.copytree(self.root/parent,target)
        call=copy.deepcopy(self.calls[parent]); call['root']=str(target)
        return target,call

    def rebind_terminal(self,root,call,mutate):
        terminal=load(root/'terminal.json'); mutate(terminal)
        # Only newly copied mutation controls are overwritten; historical originals stay intact.
        (root/'terminal.json').write_text(json.dumps(terminal,sort_keys=True))
        receipt=load(root/'receipt.json'); receipt['terminal_sha256']=sha(root/'terminal.json')
        (root/'receipt.json').write_text(json.dumps(receipt,sort_keys=True))
        finish=load(root/'finish.json'); finish['receipt_sha256']=sha(root/'receipt.json')
        (root/'finish.json').write_text(json.dumps(finish,sort_keys=True))
        call['finish_sha256']=sha(root/'finish.json')

    def test_all_fixed_calls_terminal_cost_and_no_process_leak(self):
        self.assertEqual(set(self.calls),set(roster()))
        for name,(_,budget,rss,status) in roster().items():
            call=self.calls[name]
            self.assertEqual(call.get('status'),status,(name,call))
            result=flow.audit(self.root/name,call)
            fault_witness(self.root/name,load(self.root/name/'spec.json')['control'])
            self.assertEqual(result['execution_status'],status)
            self.assertFalse(result['complete_moe_proof'])
            self.assertGreater(result['stage_seconds'],0)
            self.assertGreaterEqual(call['seconds'],result['stage_seconds'])
            for stage in load(self.root/name/'terminal.json')['stages']:
                if stage['pid'] is not None: self.assertEqual(group_rss(stage['pid'])[1],[],name)

    def test_normal_exact_differential_without_new_candidates(self):
        old=load(ROOT/'docs/hz_batch_support_controls_20261001_r2.json')['checked']
        results={}
        for case in flow.protocol()['cases']:
            root=self.root/case
            results[case]=flow.audit(root,self.calls[case],recheck=True)
            accepted=load(root/'accepted.json')
            self.assertEqual(accepted['result']['results'],old[case]['bounds'])
            self.assertFalse(accepted['result']['network_or_complete_moe_proof'])
        save(self.root/'exact_differential.json',results)

    def test_partial_keeps_denominator_and_completed_checker_error_cost(self):
        root=self.root/'partial_candidate'
        payload=load(root/'payload.json')
        self.assertEqual((len(payload['batch']['queries']),len(payload['candidates']['entries'])),(4,3))
        terminal=load(root/'terminal.json')
        self.assertEqual(terminal['required'],4)
        self.assertIsNone(terminal['accepted'])
        self.assertEqual([s['phase'] for s in terminal['stages']],['produce','check'])
        self.assertGreater(terminal['stages'][-1]['seconds'],0)

    def test_missing_stdout_preserves_executed_stage(self):
        terminal=load(self.root/'missing_stdout/terminal.json')
        self.assertEqual(terminal['stages'][-1]['phase'],'check')
        self.assertEqual(terminal['stages'][-1]['status'],'COMPLETED')
        self.assertIn('output_reception_error',terminal['stages'][-1])
        self.assertGreater(terminal['stages'][-1]['seconds'],0)
        self.assertIsNone(terminal['accepted'])

    def test_scope_request_and_mandatory_hashes(self):
        root=self.root/'guarded_two_sides_positive'; spec=load(root/'spec.json'); t=load(root/'terminal.json')
        for key in ('invocation','payload','stdout'):
            for bad in (None,'','0'*64):
                hashes={'invocation':t['invocation_sha256'],'payload':t['payload_sha256'],'stdout':t['checker_stdout_sha256']}
                hashes[key]=bad
                with self.assertRaises(ValueError):
                    flow.receive(root,spec,hashes['invocation'],hashes['payload'],hashes['stdout'],time.monotonic()+30)
        with self.assertRaises(ValueError):
            flow.supervise(self.root/'wrong-request',spec,expected_request_sha256='0'*64)
        self.assertFalse((self.root/'wrong-request').exists())

    def test_stored_record_without_external_return_not_execution_acceptance(self):
        result=flow.audit(self.root/'guarded_two_sides_positive')
        self.assertFalse(result['observed_completed_api'])
        self.assertFalse(result['complete_support_execution'])

    def test_rebound_cost_status_and_roster_mutations(self):
        mutations=[lambda t:t.__setitem__('stage_seconds',0),
                   lambda t:t.__setitem__('required',3),
                   lambda t:t.__setitem__('error','pretend exception'),
                   lambda t:t['stages'].pop(),
                   lambda t:t['accepted']['result']['results'].pop()]
        for i,mutate in enumerate(mutations):
            root,call=self.clone('terminal-mutation-'+str(i))
            self.rebind_terminal(root,call,mutate)
            with self.assertRaises(ValueError): flow.audit(root,call)

    def test_stage_semantics_after_rebinding_stage_and_chain(self):
        for field,value in (('seconds',1000.),('deadline_monotonic',0.),('cleanup_included',False),('returncode',7)):
            root,call=self.clone('stage-mutation-'+field)
            def mutate(t):
                t['stages'][0][field]=value
                (root/'produce_stage.json').write_text(json.dumps(t['stages'][0]))
            self.rebind_terminal(root,call,mutate)
            with self.assertRaises(ValueError): flow.audit(root,call)

    def test_error_elapsed_over_budget_must_be_timeout(self):
        call=copy.deepcopy(self.calls['candidate_exception'])
        call['seconds']=301.
        with self.assertRaises(ValueError): flow.audit(self.root/'candidate_exception',call)

    def test_publication_deadline_clock_injection(self):
        original_clock=time.monotonic; original_save=flow.save; shift=[0.]
        def clock(): return original_clock()+shift[0]
        def publish(path,value):
            record=original_save(path,value)
            if Path(path).name=='finish.json': shift[0]=300.
            return record
        spec=flow.specification(); root=self.root/'publication_clock_control'
        with patch.object(flow.time,'monotonic',side_effect=clock),patch.object(flow,'save',side_effect=publish):
            call=flow.supervise(root,spec,expected_request_sha256=identity(spec),budget=30)
        self.assertEqual(call['status'],'TIMEOUT')
        self.assertTrue((root/'publication_timeout.json').exists())
        self.assertFalse(flow.audit(root,call)['complete_support_execution'])
        save(self.root/'publication_clock_observed.json',{'synthetic_clock_injection':True,'call':call})

    def test_parent_hash_cost_obeys_phase_deadline(self):
        original_clock=time.monotonic; original_sha=flow.sha; shift=[0.]
        spec=flow.specification(); root=self.root/'hash_clock_control'
        def clock(): return original_clock()+shift[0]
        def hash_file(path):
            value=original_sha(path)
            if Path(path)==root/'payload.json': shift[0]=25.
            return value
        with patch.object(flow.time,'monotonic',side_effect=clock),patch.object(flow,'sha',side_effect=hash_file):
            call=flow.supervise(root,spec,expected_request_sha256=identity(spec),budget=30)
        self.assertEqual(call['status'],'TIMEOUT')
        terminal=load(root/'terminal.json')
        self.assertEqual(len(terminal['stages']),1)
        self.assertEqual(terminal['stages'][0]['worker_status'],'COMPLETED')
        self.assertIsNone(terminal['accepted'])
        flow.audit(root,call)
        save(self.root/'hash_clock_observed.json',{'synthetic_clock_injection':True,'call':call})

    def test_final_sample_drives_both_return_status_and_cost(self):
        root=self.root/'final-sample-control'; root.mkdir()
        with patch.object(flow.time,'monotonic',side_effect=[10.99,11.01]) as clock:
            value=flow.finish_observation(root,flow.DONE,1.,10.,'0'*64,'clock-control')
        self.assertEqual(clock.call_count,1)
        self.assertEqual(value['status'],flow.DONE)
        self.assertLess(value['seconds'],1.)
        with patch.object(flow.time,'monotonic',side_effect=[11.01,11.02]):
            value=flow.finish_observation(root,flow.DONE,1.,10.,'0'*64,'clock-control')
        self.assertEqual(value['status'],'TIMEOUT')
        self.assertGreaterEqual(value['seconds'],1.)

    def test_terminal_sample_drives_both_acceptance_and_charged_cost(self):
        with patch.object(flow.time,'monotonic',side_effect=[10.99,11.01]) as clock:
            status,accepted,charged=flow.terminal_observation(flow.DONE,{'checked':True},10.,11.)
        self.assertEqual(clock.call_count,1)
        self.assertEqual(status,flow.DONE)
        self.assertEqual(accepted,{'checked':True})
        self.assertLess(charged,1.)
        with patch.object(flow.time,'monotonic',return_value=11.01) as clock:
            status,accepted,charged=flow.terminal_observation(flow.DONE,{'checked':True},10.,11.)
        self.assertEqual(clock.call_count,1)
        self.assertEqual(status,'TIMEOUT')
        self.assertIsNone(accepted)
        self.assertGreaterEqual(charged,1.)

    def test_matching_timeout_without_fault_marker_rejected(self):
        root,_=self.clone('marker-mutation','check_delay')
        marker=load(root/'check_unaccepted.json'); marker['invocation']='wrong'
        (root/'check_unaccepted.json').write_text(json.dumps(marker))
        with self.assertRaises(ValueError): fault_witness(root,'check_delay')


if __name__=='__main__': unittest.main()
