import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import signal
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.budget_contract_v2 import Contract, Grant, verify_v2
from scripts.f0_timing_trace import Recorder
from scripts.check_budget_contract_v2 import check
from scripts.conv_three_arm_contract import ROOT
from act.pipeline.moe.request_budget import BudgetExhausted


class MemoryJournal:
    def __init__(self, now, cost=0): self.now, self.cost, self.events = now, cost, []
    def emit(self, kind, **fields):
        token = len(self.events)
        self.events.append({'kind':kind, **fields})
        self.now[0] += self.cost
        return token


class BudgetContractTests(unittest.TestCase):
    def context(self, now=None, cost=0):
        now = [0.] if now is None else now
        journal = MemoryJournal(now,cost)
        return now,journal,Contract(started=0, journal=journal, clock=lambda:now[0])

    def test_stale_limit_charges_construction_and_log(self):
        now,journal,ctx = self.context(cost=.1)
        options = {'time_limit':5., 'presolve':True, 'mip_rel_gap':0.}
        actual=[]; marker=object()
        def native(*, options): actual.append(options); return marker
        call=ctx.native_wrapper(native)
        def property(*, time_limit):
            now[0] += 3
            return call(options=options)
        self.assertIs(ctx.scope_wrapper(property)(time_limit=5.),marker)
        self.assertAlmostEqual(actual[0]['time_limit'],1.8)
        self.assertEqual(options['time_limit'],5.)
        self.assertEqual(actual[0]['presolve'],True)
        self.assertEqual(actual[0]['mip_rel_gap'],0.)

    def test_grant_deadline_is_not_reset(self):
        now,_,ctx=self.context()
        budget=ctx.make_budget_class()(300,started=0)
        grant=budget.limit('property',cap=10)
        now[0]=4
        got=[]
        def native(*, options): got.append(options['time_limit'])
        def local(*, time_limit): ctx.native_wrapper(native)(options={'time_limit':time_limit})
        ctx.scope_wrapper(local)(time_limit=grant)
        self.assertEqual(got,[6.])
        self.assertEqual(budget.record()['terminal_reserve_seconds'],5.)
        self.assertEqual(budget.record()['total_seconds'],300.)

    def test_global_deadline_does_not_reset_on_late_load(self):
        now,_,ctx=self.context([294.])
        got=[]
        def native(*, options): got.append(options['time_limit'])
        ctx.native_wrapper(native)(options={'time_limit':100.})
        self.assertEqual(got,[1.])
        now[0]=295.
        with self.assertRaises(BudgetExhausted):ctx.native_wrapper(native)(options={'time_limit':100.})
        self.assertEqual(got,[1.])

    def test_exhausted_during_journal_never_launches_zero(self):
        now,_,ctx=self.context([294.9],cost=.2)
        def native(*, options):self.fail('launched')
        with self.assertRaises(BudgetExhausted):
            ctx.native_wrapper(native)(options={'time_limit':1.})

    def test_nested_scope_minimum_and_exception_restoration(self):
        now,_,ctx=self.context()
        def inner(*, time_limit):
            self.assertEqual(min(ctx.deadlines),2.)
            raise RuntimeError('control')
        def outer(*, time_limit): ctx.scope_wrapper(inner)(time_limit=50.)
        with self.assertRaises(RuntimeError):ctx.scope_wrapper(outer)(time_limit=2.)
        self.assertEqual(ctx.deadlines,[])

    def test_both_property_paths_keep_exact_scope_and_return_identity(self):
        for arm in ('adaptive','monolithic'):
            now,journal,ctx=self.context()
            e=SimpleNamespace(pair=(0,1),property_row=(1.,-1.),property_constant=0.)
            marker=SimpleNamespace(status='UNKNOWN',candidate_input=object())
            if arm=='adaptive':
                def fn(encoding,*,time_limit): return marker
                arguments={'encoding':e}
            else:
                def fn(encodings,*,time_limit): return marker
                arguments={'encodings':[e]}
            self.assertIs(ctx.scope_wrapper(fn,property_query=True)(**arguments,time_limit=10),marker)
            self.assertEqual(journal.events[0]['scope'],{'pairs':[[0,1]],'row':[1.,-1.],'constant':0.})
            self.assertEqual(journal.events[1]['evidence_role'],'INTERMEDIATE_NOT_REQUEST_VERDICT')

    def test_complete_toy_both_arms_and_structural_audit(self):
        import torch
        from act.pipeline.moe.test_route_complexity_schedule import model,config
        from act.pipeline.moe.staged_verifier import verify_staged_linf,write_evidence_package
        from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
        from act.back_end.solver import solver_hz
        from scipy.optimize import milp
        net=model(((0.,1.,-2.),(3.,0.,-2.),(4.,0.,-2.)))
        center=torch.full((1,2),.5,dtype=torch.float64)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as directory:
            for arm in ('staged','monolithic_f0'):
                cfg=config(arm); original=copy.deepcopy(cfg)
                reference=verify_staged_linf(net,center,.1,cfg)
                journal=Path(directory)/f'{arm}.jsonl'
                report=verify_v2(net,center,.1,cfg,journal_path=journal,started=time.monotonic(),identity={'arm':arm})
                self.assertEqual(report.status,reference.status)
                self.assertEqual(cfg,original)
                self.assertIs(solver_hz.milp,milp)
                checked=check(journal,identity={'arm':arm})
                self.assertEqual(checked['status'],'PASS')
                self.assertTrue(checked['properties'])
                self.assertFalse(checked['journal_can_establish_SAFE'])
                package=Path(directory)/arm
                write_evidence_package(report,package)
                self.assertEqual(audit_evidence_package(package,replay_unsafe=True)['issues'],[])

    def test_reject_unsupported_mode_and_reentrant_adapter(self):
        now,journal,ctx=self.context()
        with ctx:
            with self.assertRaises(RuntimeError):
                with Contract(started=0,journal=journal,clock=lambda:now[0]):pass
        self.assertFalse(Contract.active)

    def test_reject_rehashed_budget_overallocation(self):
        # Rehash all records after tampering, so this exercises the budget
        # invariant rather than only a file-integrity check.
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as directory:
            path=Path(directory)/'journal.jsonl'; started=time.monotonic()
            journal=Recorder(path,started,{'test':True})
            with Contract(started=started,journal=journal) as ctx:
                ctx.native_wrapper(lambda *,options:SimpleNamespace(status=1))(
                    options={'time_limit':1.,'mip_rel_gap':0.})
                ctx.emit('WORK_COMPLETE',status='UNKNOWN')
            journal.close()
            self.assertEqual(check(path)['status'],'PASS')
            records=[json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                if record['kind']=='NATIVE_RETURN':record['effective']=500.
            previous='0'*64; lines=[]
            for record in records:
                record.pop('sha256');record['previous']=previous
                raw=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                previous=hashlib.sha256(raw).hexdigest()
                lines.append(json.dumps({**record,'sha256':previous}).encode()+b'\n')
            with patch.object(Path,'read_bytes',return_value=b''.join(lines)):
                with self.assertRaisesRegex(ValueError,'exceeded live deadline'):check(path)

    def test_actual_kill_retains_property_result_without_promotion(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as directory:
            path=Path(directory)/'journal.jsonl'
            code = ('import time; from pathlib import Path; from types import SimpleNamespace; '
                'from scripts.f0_timing_trace import Recorder; from scripts.budget_contract_v2 import Contract; '
                f'started=time.monotonic(); j=Recorder(Path({str(path)!r}),started,{{"test":True}}); '
                'ctx=Contract(started=started,journal=j); ctx.__enter__(); '
                'e=SimpleNamespace(pair=(0,1),property_row=(1.,-1.),property_constant=0.); '
                'ctx.scope_wrapper(lambda encoding,*,time_limit:SimpleNamespace(status="UNKNOWN",candidate_input=None),property_query=True)(e,time_limit=2.); '
                'ctx.native_wrapper(lambda *,options:time.sleep(60))(options={"time_limit":10.})')
            proc=subprocess.Popen([sys.executable,'-c',code],cwd=ROOT)
            try:
                until=time.monotonic()+15
                while time.monotonic()<until:
                    if path.exists() and b'NATIVE_READY' in path.read_bytes():break
                    time.sleep(.02)
                else:self.fail('child did not start native call')
                os.kill(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
                checked=check(path,killed=True)
                self.assertEqual(checked['native_unreturned'],1)
                self.assertEqual(len(checked['properties']),1)
                self.assertFalse(checked['work_complete'])
                self.assertFalse(checked['journal_can_establish_SAFE'])
                with self.assertRaises(ValueError):check(path,killed=False)
            finally:
                if proc.poll() is None:proc.kill();proc.wait()

    def test_full_api_exhaustion_cannot_promote_safe(self):
        import torch
        from act.pipeline.moe.test_route_complexity_schedule import model,config
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as directory:
            for arm in ('staged','monolithic_f0'):
                path=Path(directory)/f'{arm}.jsonl'
                report=verify_v2(model(((2.,0.),(3.,0.))),torch.full((1,2),.5,dtype=torch.float64),
                    .1,config(arm),journal_path=path,started=time.monotonic()-296,identity={'arm':arm})
                self.assertEqual(report.status,'TIMEOUT')
                self.assertFalse(report.evidence['route_coverage']['coverage_complete'])
                self.assertEqual(check(path)['status'],'PASS')

    def test_conv_worker_identity_gate_without_model_query(self):
        from scripts import conv_budget_worker_v2 as worker
        from scripts.conv_three_arm_contract import read,SELECTION,request_for
        from act.pipeline.moe.conv_training import atomic_json
        value=read(SELECTION)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as directory:
            root=Path(directory)
            request=request_for(value,value['smoke_jobs'][0],'test-head')
            atomic_json(root/'request.json',request)
            with self.assertRaises(ValueError):worker.validate(root)
            request['execution_budget_contract']=worker.execution_identity()
            atomic_json(root/'request.json',request)
            with patch.object(worker,'selection',return_value=value):
                self.assertEqual(worker.validate(root),request)
                for mutate in ('sample','subject'):
                    bad=copy.deepcopy(request)
                    if mutate=='sample':bad['sample']['dataset_index']=9999
                    else:bad['subject']['checkpoint_sha256']='0'*64
                    atomic_json(root/'request.json',bad)
                    with self.assertRaises(ValueError):worker.validate(root)
                atomic_json(root/'request.json',request)
                atomic_json(root/'budget_journal.jsonl',{})
                with self.assertRaises(ValueError):worker.validate(root)

    def test_policy_cannot_advertise_different_reserve(self):
        from scripts import conv_budget_worker_v2 as worker
        policy=worker.read(worker.POLICY)
        with patch.object(worker,'read',return_value={**policy,'terminal_reserve_seconds':20}):
            with self.assertRaises(ValueError):worker.execution_identity()


if __name__=='__main__': unittest.main()
