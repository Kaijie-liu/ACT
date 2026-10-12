"""Pure contract controls. No numerical generation, CUDA or real requests."""
import copy
import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import save
from scoped_source.check_store_endpoints import MAX_VISITS
from scoped_source.hz_request_store import identity
from scripts import hz_store_supervised as p
from scripts.run_hz_store_supervision import execute_calls


class Controls(unittest.TestCase):
    def test_frozen_source_and_property(self):
        s=p.spec(); p.validate_spec(s,None)
        for key,value in [('source','0'*64),('classes',3),('label',1),('margin','1'),('tile_width',1),('algorithm','other')]:
            with self.subTest(key=key),self.assertRaises(ValueError): p.validate_spec(dict(s,**{key:value}),None)
        with self.assertRaises(ValueError): p.validate_spec(s,{'gpu':True})

    def test_cumulative_counter_resume(self):
        d=dict(row_visits=1,coefficient_visits=2,arithmetic=MAX_VISITS-3)
        w=p.resume_work(time.monotonic()+10,d); self.assertEqual(w.counts,d)
        with self.assertRaises(ValueError): w.visit('arithmetic')
        for v in ({},dict(d,arithmetic=-1),dict(d,arithmetic=True),dict(d,arithmetic=MAX_VISITS)):
            with self.assertRaises(ValueError): p.counts(v)

    def test_deadline_not_reset(self):
        with self.assertRaises(TimeoutError): p.resume_work(time.monotonic()-1,dict.fromkeys(p.COUNTERS,0))
        inv=dict(start=0,work=299,spec=p.spec())
        self.assertEqual(p.cutoff(inv,'produce',3),269)
        self.assertEqual(p.cutoff(inv,'check',280),296)
        self.assertEqual(p.cutoff(inv,'receive',290),299)
        for control,phase in [('check_delay','check'),('source_delay','produce'),('receive_delay','receive')]:
            inv['spec']=p.spec(control=control); self.assertEqual(p.cutoff(inv,phase,2),17)

    def test_checker_isolated_and_cpu_only(self):
        for phase in ('admit','release','check','receive'):
            cmd=p.worker_command(phase,Path('/x'),'a'*64,'b'*64)
            self.assertIn('-I',cmd); self.assertIn('-S',cmd)
            self.assertEqual(p.visible({},phase),'')
        self.assertNotIn('-S',p.worker_command('produce',Path('/x'),'a'*64,'b'*64))

    def test_atomic_publication_and_nested_tamper(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp); d=root/'evidence/store/rows'; d.mkdir(parents=True)
            save(d/'one.json',{'x':1}); before=p.evidence_inventory(root)
            with self.assertRaises(FileExistsError): save(d/'one.json',{'x':2})
            self.assertEqual((d/'one.json').read_text(),'{"x":1}')
            # Failed publication stays visible, not silently discarded.
            self.assertNotEqual(before,p.evidence_inventory(root))
            (d/'foreign').symlink_to(d/'one.json')
            with self.assertRaises(ValueError): p.evidence_inventory(root)

    def test_stop_on_api_exception_or_unknown_cleanup(self):
        for raised in (True,False):
            writes={}; starts=[]
            def invoke(*args,**kwargs):
                starts.append(args[1])
                if raised: raise RuntimeError('injected API interruption')
                return {'status':'CLEANUP_INCOMPLETE'}
            calls=execute_calls(Path('/not_written'),invoke=invoke,
                publish=lambda path,value:writes.update({path.name:value}),read=lambda _:dict(stages=[]))
            self.assertEqual(len(starts),1); self.assertEqual(len(calls),1)
            self.assertEqual(len(writes['batch_stop.json']['pending']),12)
            self.assertIn('started_half.json',writes)

    def receipt(self,case='half'):
        from scoped_source.check_store_endpoints import aggregate, Plan, EndpointWork
        s=p.spec(case); contract=dict(store='7'*64); anchor='8'*64
        expected=p.expected_rows(contract,anchor,s)
        duties=tuple((tuple(pair),k,tuple(slots)) for pair,k,slots in expected)
        keys=[k for _,_,slots in duties for k in slots]
        from types import SimpleNamespace
        plan=Plan(anchor,'checked_source',(),tuple(SimpleNamespace(slot=k) for k in keys),duties)
        from fractions import Fraction
        values={k:Fraction(2) for k in keys}
        if case=='partial': values[keys[-1]]=None
        agg=aggregate(plan,values,EndpointWork(time.monotonic()+10))
        inv=dict(invocation='call',spec=s); inputs=dict(produce='1'*64,check='2'*64,release='3'*64)
        payload=dict(counts=dict.fromkeys(p.COUNTERS,1),contract=anchor)
        checked=dict(invocation='call',payload_sha256=inputs['produce'],counts=dict.fromkeys(p.COUNTERS,2),
                     result=dict(results=agg['results'],aggregation=agg))
        return inv,inputs,payload,checked,contract

    def test_receipt_coverage_counters_and_guarantee(self):
        inv,inputs,payload,checked,contract=self.receipt()
        def reader(path,*args): return contract if path.name=='contract.json' else checked
        with patch.object(p,'payload',return_value=payload),patch.object(p,'load',side_effect=reader):
            self.assertTrue(p.receive(Path('/x'),inv,inputs)['declared_source_positive'])
        for change in ('missing','counter','source','positive','trust','endpoints','foreign_slot'):
            bad=copy.deepcopy(checked)
            if change=='missing': bad['result']['results'].pop()
            elif change=='counter': bad['counts']['row_visits']=0
            elif change=='source': bad['payload_sha256']='4'*64
            elif change=='positive': bad['result']['aggregation']['positive']=0
            elif change=='trust': bad['result']['aggregation']['native_float_proof']=True
            elif change=='endpoints': bad['result']['results'][0]['endpoints']=['a']
            else: bad['result']['results'][0]['endpoints'][0]='f'*64
            def reader(path,*args): return contract if path.name=='contract.json' else bad
            with self.subTest(change=change),patch.object(p,'payload',return_value=payload),patch.object(p,'load',side_effect=reader),self.assertRaises(ValueError):
                p.receive(Path('/x'),inv,inputs)

    def test_partial_cannot_claim_positive_or_hide_missing_duty(self):
        inv,inputs,payload,checked,contract=self.receipt('partial')
        def reader(path,*args): return contract if path.name=='contract.json' else checked
        with patch.object(p,'payload',return_value=payload),patch.object(p,'load',side_effect=reader):
            result=p.receive(Path('/x'),inv,inputs)
            self.assertFalse(result['declared_source_positive'])
            checked['result']['results'][-1].update(lower='2',positive=True)
            checked['result']['aggregation']['positive']=3
            with self.assertRaises(ValueError): p.receive(Path('/x'),inv,inputs)

    def test_censored_costs_and_missing_completed_events(self):
        from scripts.run_hz_store_supervision import costs
        import json
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp); stage=dict(phase='produce',status='TIMEOUT',start_seconds=0.,end_seconds=2.,cleanup_seconds=.1)
            events=[dict(event='ENTER',elapsed=1.,operation='source')]
            with patch.object(Path,'read_text',return_value='\n'.join(map(json.dumps,events))),patch.object(Path,'exists',return_value=True):
                value=costs(root,dict(stages=[stage]))['produce'][0]
                self.assertIsNone(value['seconds']); self.assertEqual(value['status'],'RIGHT_CENSORED')
                with self.assertRaises(ValueError): costs(root,dict(stages=[dict(stage,status='COMPLETED')]))
            events.append(dict(event='EXIT',elapsed=1.5,operation='source',seconds=10))
            with patch.object(Path,'read_text',return_value='\n'.join(map(json.dumps,events))),patch.object(Path,'exists',return_value=True),self.assertRaises(ValueError):
                costs(root,dict(stages=[stage]))

    def test_late_and_cleanup_classification(self):
        from scripts.hz_device_lifecycle import finalize_status
        self.assertEqual(finalize_status(p.DONE,300.,300.),'TIMEOUT')
        self.assertEqual(finalize_status('CLEANUP_INCOMPLETE',301.,300.),'CLEANUP_INCOMPLETE')
        self.assertFalse(p.cleanup_confirmed(dict(pid=1,returncode=0,remaining_group=dict(live=[2]),
                                                cleanup_status='LEADER_REAPED_NO_LIVE_GROUP')))

    def test_late_observer_and_cost_mutation(self):
        from scoped_proof import device_lifecycle as engine
        inv=dict(start=10.,deadline=310.); finish=dict(status=p.DONE,seconds=5.)
        obs=dict(begin=9.,end=16.,result=dict(invocation_sha256='a'*64,finish_sha256='b'*64,
                                           seconds=5.9,complete_moe_proof=False,status=p.DONE))
        with patch.object(engine,'sha',return_value='b'*64):
            self.assertEqual(engine.check_observation(Path('/x'),inv,'a'*64,finish,obs),p.DONE)
            self.assertIsNone(engine.check_observation(Path('/x'),inv,'a'*64,finish,None))
            obs['end']=311.
            with self.assertRaisesRegex(ValueError,'late API'): engine.check_observation(Path('/x'),inv,'a'*64,finish,obs)
            obs['result']['status']='TIMEOUT'
            self.assertEqual(engine.check_observation(Path('/x'),inv,'a'*64,finish,obs),'TIMEOUT')
            obs['result']['seconds']=0.
            with self.assertRaisesRegex(ValueError,'API anchor/cost'): engine.check_observation(Path('/x'),inv,'a'*64,finish,obs)

    def test_no_full_deadline_override_by_fault(self):
        from scripts.run_hz_store_supervision import roster
        self.assertEqual(len(roster()),13)
        self.assertTrue(all(p.budget(s)==300 for _,s in roster()))


TEST_NAMES=tuple('scripts.test_hz_store_supervised.Controls.test_'+n for n in (
    'atomic_publication_and_nested_tamper','censored_costs_and_missing_completed_events',
    'checker_isolated_and_cpu_only','cumulative_counter_resume','deadline_not_reset',
    'frozen_source_and_property','late_and_cleanup_classification','late_observer_and_cost_mutation',
    'no_full_deadline_override_by_fault','partial_cannot_claim_positive_or_hide_missing_duty',
    'receipt_coverage_counters_and_guarantee','stop_on_api_exception_or_unknown_cleanup'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--report',type=Path,required=True); a=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    class Result(unittest.TextTestResult):
        outcomes=[]
        def startTest(self,t): self.state='PASS'; super().startTest(t)
        def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
        def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
        def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
        def addExpectedFailure(self,t,e): self.state='EXPECTED_FAILURE'; super().addExpectedFailure(t,e)
        def addUnexpectedSuccess(self,t): self.state='UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
        def stopTest(self,t):
            self.outcomes.append(dict(name='scripts.test_hz_store_supervised.Controls.'+t._testMethodName,status=self.state))
            super().stopTest(t)
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    save(a.report,dict(tests=result.testsRun,outcomes=result.outcomes,
                      exceptional={k:len(getattr(result,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}))
    raise SystemExit(0 if result.wasSuccessful() and len(result.outcomes)==len(TEST_NAMES) else 1)
