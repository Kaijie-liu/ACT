import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.conv_budget_smoke_v2 import (ROOT, schedule, selection, request, identity,
    publish_terminal, artifacts, run)
from scripts.audit_conv_budget_smoke_v2 import roster, journal_check
from scripts.audit_conv_three_arm import terminal_contract
from act.pipeline.moe.conv_training import atomic_json


class OuterV2Tests(unittest.TestCase):
    def test_frozen_roster_and_worker_binding(self):
        from scripts.conv_budget_worker_v2 import validate
        value=selection();jobs=schedule(value)
        self.assertEqual(len(jobs),4)
        for job in jobs:
            with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
                path=Path(d);req=request(value,job,'control',identity())
                atomic_json(path/'request.json',req)
                self.assertEqual(validate(path),req)
                req['execution_budget_contract']['policy']['terminal_reserve_seconds']=0
                atomic_json(path/'request.json',req)
                with self.assertRaises(ValueError):validate(path)

    def test_no_new_root_or_full_roster(self):
        with self.assertRaises(ValueError):run(ROOT/'data/moe/results/not_authorized')
        v=copy.deepcopy(selection());v['smoke_jobs']=v['smoke_jobs'][::-1]
        with self.assertRaises(ValueError):schedule(v)

    def test_roster_failstop_and_missing_denominator(self):
        jobs=schedule(selection());rows=[{**j,'status':'TIMEOUT'} for j in jobs]
        end={'state':'EXECUTION_COMPLETED','error':None,'completed_job_ids':[r['job_id'] for r in rows],
             'unattempted':[],'full_started':False}
        roster(rows,jobs,end)
        with self.assertRaises(ValueError):roster(rows[:-1],jobs,end)
        rows[0]['status']='ERROR'
        with self.assertRaises(ValueError):roster(rows,jobs,end)
        end.update(state='EXECUTION_ERROR',error='control',completed_job_ids=[rows[0]['job_id']],unattempted=jobs[1:])
        roster(rows[:1],jobs,end)

    def fixture(self,path):
        job={'job_id':'job','method':'adaptive','dataset_index':0,'rank':0,'position':0}
        d=path/'job';d.mkdir();atomic_json(d/'request.json',{})
        p=d/'package';p.mkdir();atomic_json(p/'manifest.json',{'status':'SAFE'})
        (d/'budget_journal.jsonl').write_text('partial')
        return job,d

    def test_late_safe_kept_but_never_accepted(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            path=Path(d);job,directory=self.fixture(path)
            row=publish_terminal(path,job,time.monotonic()-301,0,False,{})
            self.assertEqual(row['status'],'TIMEOUT');self.assertIsNone(row['package'])
            self.assertIn('package/manifest.json',row['artifacts'])
            self.assertEqual(artifacts(directory),row['artifacts']);terminal_contract(row)
            row.update(status='SAFE',package=str(directory/'package'))
            with self.assertRaises(ValueError):terminal_contract(row)

    def test_error_and_partial_inventory(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            path=Path(d);job,directory=self.fixture(path)
            row=publish_terminal(path,job,time.monotonic(),2,False,{},'control')
            self.assertEqual(row['status'],'ERROR');self.assertIsNone(row['package'])
            self.assertIn('budget_journal.jsonl',row['artifacts'])
            self.assertEqual(json.loads((path/'rows.jsonl').read_text()),row)

    def test_missing_journal_not_complete(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            self.assertIsNone(journal_check(Path(d),{'package':None},{}))
            with self.assertRaises(ValueError):journal_check(Path(d),{'package':'x'},{})

    def test_owned_supervisor_error_stops_remaining_jobs(self):
        from scripts import conv_budget_smoke_v2 as runner
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            root=Path(d)/'new'; calls=[]
            def fake_execute(command,log,started,env,**kwargs):
                calls.append(command)
                self.assertIn('scripts.conv_budget_worker_v2',command)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
                return 2,False
            def fake_audit(*args,**kwargs):
                atomic_json(root/'audit.final.json',{'smoke_gate':'FAIL'})
                return SimpleNamespace(returncode=0)
            def git(*args):
                return '' if args[0]=='status' else 'feat/moe-route-verification' if args[0]=='branch' else 'control'
            with patch.object(runner,'DEFAULT',root),patch.object(runner,'_git_value',side_effect=git),\
                    patch.object(runner,'execute',side_effect=fake_execute),\
                    patch.object(runner,'wait_resources',return_value={}),\
                    patch.object(runner,'subprocess',SimpleNamespace(run=fake_audit)):
                self.assertEqual(runner.run(root),1)
            self.assertEqual(len(calls),1)
            end=json.loads((root/'run_terminal.json').read_text())
            self.assertEqual(len(end['unattempted']),3)
            self.assertFalse(end['full_started'])
            self.assertEqual(end['state'],'EXECUTION_ERROR')

    def test_actual_toy_package_journal_and_identity_tamper(self):
        import torch
        from scripts.budget_contract_v2 import verify_v2
        from act.pipeline.moe.test_route_complexity_schedule import model,config
        from act.pipeline.moe.staged_verifier import write_evidence_package
        from act.pipeline.moe.experiment1 import _sha256
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            path=Path(d);req={'execution_budget_contract':identity()['budget']}
            atomic_json(path/'request.json',req);sha=_sha256(path/'request.json')
            binding={'request_sha256':sha,'execution_budget_contract':req['execution_budget_contract']}
            started=time.monotonic()
            report=verify_v2(model(((0.,1.,-2.),(3.,0.,-2.),(4.,0.,-2.))),
                torch.full((1,2),.5,dtype=torch.float64),.1,config(),
                journal_path=path/'budget_journal.jsonl',started=started,identity=binding)
            report.evidence['execution_budget_contract']['journal_sha256']=_sha256(path/'budget_journal.jsonl')
            write_evidence_package(report,path/'package')
            row={'package':str(path/'package'),'outer_timeout':False,'status':report.status,
                 'request_sha256':sha,'wall_seconds':time.monotonic()-started}
            self.assertTrue(journal_check(path,row,req)['work_complete'])
            row['request_sha256']='0'*64
            with self.assertRaises(ValueError):journal_check(path,row,req)


if __name__=='__main__':unittest.main()
