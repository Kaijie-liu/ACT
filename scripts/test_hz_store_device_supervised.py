"""Pure lifecycle contracts; no CUDA, numerical producer or real source."""
import copy
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import save
from scoped_proof import device_lifecycle as engine
from scoped_source.hz_request_store import identity
from scripts import hz_store_device_supervised as p
from scripts import hz_device_release as release
from scripts.run_hz_store_device_supervision import may_continue, roster


class Controls(unittest.TestCase):
    def metadata(self,device='cpu'):
        s=p.spec(device=device); ctx=None if device=='cpu' else dict(binding='new request')
        c=dict(device=device,algorithm=s['algorithm'],dtype='float64',iterations_per_tile=128,
               proposal_tiles=3,proposal_columns=5,api_seconds=1.,native_preparation_seconds=.1,
               proposal_seconds=.2,exact_selection_seconds=.1,serialization_seconds=.1,device_report=dict(
                context_sha256=None if ctx is None else identity(ctx),hardware=None,memory=None,transfer_groups=10,
                completed_device_costs=dict(initialization_sync_seconds=.1,transfer_sync_seconds=.1,
                    optimization_sync_seconds=.1,readback_sync_seconds=.1),
                cost_scope='nested in inclusive API/preparation/proposal costs; do not add twice'))
        data=dict(candidate=c,context=ctx,pre_cuda_sha256=None,cuda_intent_sha256=None,
                  costs=dict(candidates=1.5),environment=dict(torch='frozen',cuda_build=None,cpu_threads=1))
        if device=='cuda:0':
            c['device_report'].update(hardware=dict(gpu_uuid=p.UUID,name='simulated',total_memory=2**34),
                                      memory=dict(peak_allocated=2**20,peak_reserved=2**21))
            data['environment']['cuda_build']='simulated'
        return data,s,ctx,dict(seconds=2.)

    def test_spec_and_permit(self):
        self.assertEqual(len(roster()),10)
        for _,s in roster(): p.validate_spec(s,None); self.assertEqual(p.budget(s),300)
        with self.assertRaisesRegex(ValueError,'freeze'): p.validate_spec(p.spec(device='cuda:0'),None)
        for key,v in [('source','a'*64),('classes',3),('tile_width',1),('margin','1'),('algorithm','old')]:
            with self.assertRaises(ValueError): p.validate_spec(dict(p.spec(),**{key:v}),None)
        with self.assertRaises(ValueError): p.validate_spec(p.spec(),{'path':'/tmp/foreign','sha256':'a'*64})

    def test_context_binds_actual_store_contract(self):
        inv=dict(spec=p.spec(device='cuda:0'),invocation='owned')
        c=p.context(inv,100.,dict(store='1'*64),'2'*64)
        self.assertEqual(c,dict(schema='STORE_DEVICE_CONTEXT_V1',store='1'*64,request=inv['spec']['source'],
            contract='2'*64,deadline=100.,invocation='owned',gpu_uuid=p.UUID,allocator_limit_bytes=2**30))
        self.assertNotEqual(identity(c),identity(p.context(inv,101.,dict(store='1'*64),'2'*64)))
        self.assertNotEqual(identity(c),identity(p.context(inv,100.,dict(store='3'*64),'2'*64)))
        inv['spec']=p.spec(); self.assertIsNone(p.context(inv,100.,dict(store='1'*64),'2'*64))

    def test_phase_clocks_and_isolation(self):
        inv=dict(start=0.,work=299.,spec=p.spec())
        self.assertEqual(p.cutoff(inv,'produce',5),269)
        self.assertEqual(p.cutoff(inv,'check',270),296)
        self.assertEqual(p.cutoff(inv,'release',272),277)
        inv['spec']=p.spec(control='readback_delay'); self.assertEqual(p.cutoff(inv,'produce',5),20)
        for phase in ('admit','release','check','receive'):
            cmd=p.worker_command(phase,Path('/x'),'a','b'); self.assertIn('-I',cmd); self.assertIn('-S',cmd)
            self.assertEqual(p.visible(inv,phase),'')
        inv['spec']=p.spec(device='cuda:0'); self.assertEqual(p.visible(inv,'produce'),p.UUID)
        inv['spec']=p.spec(device='cuda:0',control='pre_cuda_busy_stub'); self.assertEqual(p.visible(inv,'produce'),'')

    def test_cpu_metadata_refuses_cuda(self):
        data,s,ctx,stage=self.metadata(); p.device_metadata(data,s,ctx,stage)
        for key,v in [('device','cuda:0'),('algorithm','CUDA_BLOCK_PROJECTED_SUBGRADIENT_128_V1'),('dtype','float32'),
                      ('iterations_per_tile',127),('proposal_columns',4)]:
            bad=copy.deepcopy(data); bad['candidate'][key]=v
            with self.assertRaises(ValueError): p.device_metadata(bad,s,ctx,stage)
        for key,v in [('hardware',{}),('memory',{}),('context_sha256','a'*64)]:
            bad=copy.deepcopy(data); bad['candidate']['device_report'][key]=v
            with self.assertRaises(ValueError): p.device_metadata(bad,s,ctx,stage)

    def test_gpu_metadata_envelope(self):
        data,s,ctx,stage=self.metadata('cuda:0'); p.device_metadata(data,s,ctx,stage)
        for mem in (dict(peak_allocated=3,peak_reserved=2),dict(peak_allocated=0,peak_reserved=2**30+1),
                    dict(peak_allocated=True,peak_reserved=100)):
            bad=copy.deepcopy(data); bad['candidate']['device_report']['memory']=mem
            with self.assertRaises(ValueError): p.device_metadata(bad,s,ctx,stage)
        data['candidate']['device_report']['hardware']['gpu_uuid']='wrong'
        with self.assertRaises(ValueError): p.device_metadata(data,s,ctx,stage)

    def test_context_and_environment_mutations(self):
        data,s,ctx,stage=self.metadata('cuda:0')
        for change in ('context','hash','thread','cuda','torch'):
            bad=copy.deepcopy(data)
            if change=='context': bad['context']={'different':'request'}
            elif change=='hash': bad['candidate']['device_report']['context_sha256']='0'*64
            elif change=='thread': bad['environment']['cpu_threads']=2
            elif change=='cuda': bad['environment']['cuda_build']=None
            else: bad['environment']['torch']=''
            with self.assertRaises(ValueError): p.device_metadata(bad,s,ctx,stage)

    def test_cost_accounting_mutations(self):
        data,s,ctx,stage=self.metadata()
        for change in ('negative','nan','api','nested','sum','groups','scope','cpu_negative','cpu_sum'):
            bad=copy.deepcopy(data); c=bad['candidate']; dev=c['device_report']
            if change=='negative': c['api_seconds']=-1
            elif change=='nan': c['api_seconds']=float('nan')
            elif change=='api': c['api_seconds']=3.
            elif change=='nested': dev['completed_device_costs']['readback_sync_seconds']=2.
            elif change=='sum': dev['completed_device_costs']={k:.4 for k in dev['completed_device_costs']}
            elif change=='groups': dev['transfer_groups']=9
            elif change=='cpu_negative': c['serialization_seconds']=-1.
            elif change=='cpu_sum': c['proposal_seconds']=.9
            else: dev['cost_scope']='add to total'
            with self.assertRaises(ValueError): p.device_metadata(bad,s,ctx,stage)

    def release_record(self,control):
        now=time.monotonic(); inv=dict(invocation='r',start=now-2,spec=p.spec(device='cuda:0'))
        plan=dict(run_deadline=now+10,inputs=dict(produce='1'*64)); producer=dict(pid=123,end_seconds=1)
        record=release.simulated_record(inv,plan,producer,control)
        args=dict(invocation='r',gpu_uuid=p.UUID,producer_sha='1'*64,producer_pid=123,
                  after=now-1,deadline=now+10,simulated=True)
        return record,args

    def test_release_foreign_and_owned(self):
        for c,accepted in [('release_foreign_tenant',True),('release_owned_stays',False)]:
            record,args=self.release_record(c); result=release.validate(record,**args)
            self.assertEqual(result['release_confirmed'],accepted)
            self.assertFalse(result['next_admission_established']); self.assertFalse(result['physical_driver_release_proved'])

    def test_release_stale_wrong_binding(self):
        record,args=self.release_record('release_foreign_tenant')
        for change in ('invocation','owned_pids','clock','raw','missing'):
            bad=copy.deepcopy(record)
            if change=='invocation': bad['invocation']='other'
            elif change=='owned_pids': bad['owned_pids']=[999]
            elif change=='clock': bad['observations'][0]['start']=args['after']-1
            elif change=='raw': bad['observations'][0]['compute_pids']=[]
            else: bad['observations'].pop()
            with self.assertRaises(ValueError): release.validate(bad,**args)

    def test_receipt_partial_and_foreign_check(self):
        from scripts.test_hz_store_supervised import Controls as Prior
        for case in ('half','partial'):
            inv,inputs,payload,checked,contract=Prior().receipt(case); inv['spec']['device']='cpu'
            def reader(path,*args): return contract if path.name=='contract.json' else checked
            with patch.object(p,'payload',return_value=payload),patch.object(p.base,'load',side_effect=reader):
                result=p.receive(Path('/x'),inv,inputs)
                self.assertEqual(result['declared_source_positive'],case=='half')
                checked['payload_sha256']='9'*64
                with self.assertRaises(ValueError): p.receive(Path('/x'),inv,inputs)
            with patch.object(p,'payload',return_value=payload),patch.object(p.base,'load',side_effect=FileNotFoundError):
                with self.assertRaises(FileNotFoundError): p.receive(Path('/x'),inv,inputs)

    def test_late_observation(self):
        inv=dict(start=10.,deadline=310.); finish=dict(status=p.DONE,seconds=5.)
        obs=dict(begin=9.,end=311.,result=dict(invocation_sha256='a'*64,finish_sha256='b'*64,
                 seconds=5.9,complete_moe_proof=False,status=p.DONE))
        with patch.object(engine,'sha',return_value='b'*64),self.assertRaisesRegex(ValueError,'late API'):
            engine.check_observation(Path('/x'),inv,'a'*64,finish,obs)

    def test_batch_stop_on_unresolved_release(self):
        terminal=dict(stages=[{}])
        with patch.object(p,'cleanup_confirmed',return_value=True),patch.object(Path,'exists',return_value=False):
            self.assertFalse(may_continue(Path('/x'),'half',dict(status='DEVICE_RELEASE_UNCONFIRMED'),terminal))
            self.assertTrue(may_continue(Path('/x'),'release_owned_stays',dict(status='DEVICE_RELEASE_UNCONFIRMED'),terminal))
            with patch.object(Path,'exists',return_value=True):
                self.assertFalse(may_continue(Path('/x'),'release_owned_stays',dict(status='DEVICE_RELEASE_UNCONFIRMED'),terminal))
        with patch.object(p,'cleanup_confirmed',return_value=False):
            self.assertFalse(may_continue(Path('/x'),'release_delay',dict(status='DEVICE_RELEASE_UNCONFIRMED'),terminal))


TEST_NAMES=tuple('scripts.test_hz_store_device_supervised.Controls.test_'+s for s in (
    'batch_stop_on_unresolved_release','context_and_environment_mutations','context_binds_actual_store_contract',
    'cost_accounting_mutations','cpu_metadata_refuses_cuda','gpu_metadata_envelope','late_observation',
    'phase_clocks_and_isolation','receipt_partial_and_foreign_check','release_foreign_and_owned',
    'release_stale_wrong_binding','spec_and_permit'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--report',type=Path,required=True); a=parser.parse_args()
    class Result(unittest.TextTestResult):
        outcomes=[]
        def startTest(self,t): self.state='PASS'; super().startTest(t)
        def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
        def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
        def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
        def addExpectedFailure(self,t,e): self.state='EXPECTED_FAILURE'; super().addExpectedFailure(t,e)
        def addUnexpectedSuccess(self,t): self.state='UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(t)
        def stopTest(self,t):
            self.outcomes.append(dict(name='scripts.test_hz_store_device_supervised.Controls.'+t._testMethodName,status=self.state))
            super().stopTest(t)
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    save(a.report,dict(tests=result.testsRun,outcomes=result.outcomes,
        exceptional={k:len(getattr(result,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}))
    raise SystemExit(0 if result.wasSuccessful() and len(result.outcomes)==len(TEST_NAMES) else 1)
