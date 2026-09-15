import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.conv_full_v2_contract import ROOT, identity, full_selection, request_for, validate_worker
from scripts.audit_conv_full_v2 import roster, summarize
from scripts.run_conv_full_v2 import terminal
from scripts.audit_conv_three_arm import terminal_contract, external_result
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256


class FullV2Tests(unittest.TestCase):
    def test_exact_original_ninety_and_three_arms(self):
        value=full_selection();execution=identity()
        self.assertEqual(len(value['full_jobs']),90)
        for job in value['full_jobs']:
            r=request_for(value,job,'test',execution)
            self.assertEqual(r['sample'],value['samples'][job['rank']])
            self.assertEqual(r['epsilon'],2/255)
            self.assertEqual('execution_budget_contract' in r,job['method']!='crown')
        with self.assertRaises(ValueError):request_for(value,value['smoke_jobs'][0],'test',execution)

    def test_worker_rejects_changed_request_or_resumed_directory(self):
        from scripts import conv_full_v2_contract as c
        value=full_selection();ex=identity();job=value['full_jobs'][0]
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            root=Path(d);freeze=root/'freeze.json';atomic_json(freeze,{'control':True})
            directory=root/job['job_id'];directory.mkdir()
            rt={'schema':'conv_three_arm_full_v2','execution':ex,'selection':value,'smoke':False,
                'freeze_sha256':_sha256(freeze),'git_head':'test'}
            atomic_json(root/'runtime.json',rt);req=request_for(value,job,'test',ex)
            atomic_json(directory/'request.json',req)
            with patch.object(c,'DEFAULT',root),patch.object(c,'FREEZE',freeze),patch.object(c,'identity',return_value=ex):
                self.assertEqual(validate_worker(directory),req)
                req['epsilon']=1/255;atomic_json(directory/'request.json',req)
                with self.assertRaises(ValueError):validate_worker(directory)
                req['epsilon']=2/255;atomic_json(directory/'request.json',req)
                atomic_json(directory/'terminal.json',{})
                with self.assertRaises(ValueError):validate_worker(directory)

    def test_terminal_no_external_positive_promotion_after_deadline(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            root=Path(d);job={'job_id':'job','method':'crown','rank':0};directory=root/'job';directory.mkdir()
            atomic_json(directory/'request.json',{});atomic_json(directory/'external.json',{'status':'POSITIVE'})
            row=terminal(root,job,time.monotonic()-301,0,False,{})
            self.assertEqual(row['status'],'TIMEOUT');self.assertTrue(row['outer_timeout'])
            self.assertEqual(row['evidence_level'],'CROWN_NUMERICAL_FILTER')
            self.assertIn('external.json',row['artifacts']);terminal_contract(row)
            row['status']='POSITIVE'
            with self.assertRaises(ValueError):terminal_contract(row)

    def test_crown_full_identity_same_frozen_checker(self):
        from scripts.conv_three_arm_contract import request_for as oldrequest
        root=ROOT/'data/moe/results/conv_three_arm_smoke_20260915_r1';rt=json.loads((root/'runtime.json').read_text())
        rows=list(map(json.loads,(root/'rows.jsonl').read_text().splitlines()))
        row=next(r for r in rows if r['method']=='crown')
        self.assertTrue(external_result(root,row,oldrequest(rt['selection'],row,rt['git_head']))['complete'])

    def records(self):
        jobs=full_selection()['full_jobs'];rows=[{**j,'status':'UNKNOWN','outer_timeout':False,
            'wall_seconds':1.,'resource_wait':{'seconds':0.}} for j in jobs]
        end={'state':'EXECUTION_COMPLETED','error':None,'full_started':True,
             'completed_job_ids':[j['job_id'] for j in jobs],'unattempted':[]}
        return jobs,rows,end

    def test_incomplete_and_error_roster_not_complete(self):
        jobs,rows,end=self.records();roster(rows,jobs,end)
        with self.assertRaises(ValueError):roster(rows[:-1],jobs,end)
        end.update(state='EXECUTION_ERROR',error='control',completed_job_ids=[rows[0]['job_id']],unattempted=jobs[1:])
        rows[0]['status']='ERROR';roster(rows[:1],jobs,end)
        with self.assertRaises(ValueError):roster(rows,jobs,end)

    def test_summary_evidence_levels_denominators_and_gains(self):
        jobs,rows,end=self.records()
        rows[0]['status']='SAFE';rows[2]['status']='POSITIVE'
        result=summarize(rows,[],True)
        self.assertEqual(result['comparisons']['monolithic']['adaptive_only_positive_ranks'],[0])
        self.assertEqual(result['comparisons']['crown']['shared_positive_ranks'],[0])
        self.assertFalse(result['comparisons']['crown']['evidence_levels_interchangeable'])
        self.assertEqual(result['methods']['crown']['denominator'],30)
        self.assertEqual(summarize(rows[:1],[],False)['comparisons'],{})
        self.assertEqual(summarize(rows[:1],[],False)['methods']['crown']['unattempted'],30)

    def test_conflicting_witness_and_positive_rejected(self):
        _,rows,_=self.records();rows[0]['status']='SAFE';rows[2]['status']='UNSAFE'
        with self.assertRaises(ValueError):summarize(rows,[],True)

    def test_runner_all_ninety_and_error_failstop(self):
        from scripts import run_conv_full_v2 as runner
        value=full_selection();ex=identity()
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
                root=Path(tmp)/'new';freeze=Path(tmp)/'freeze.json'
                atomic_json(freeze,{'status':'PASS','execution':ex,'parent_artifacts':{}})
                calls=[]
                def execute(command,log,started,env,**kw):
                    d=Path(command[command.index('--root')+1]);req=json.loads((d/'request.json').read_text());calls.append(req)
                    if fail:return 2,False
                    if req['method']=='crown':atomic_json(d/'external.json',{'status':'UNKNOWN'})
                    else:
                        (d/'package').mkdir();atomic_json(d/'package/manifest.json',{'status':'UNKNOWN'})
                    return 0,False
                def git(*a):return '' if a[0]=='status' else 'feat/moe-route-verification' if a[0]=='branch' else 'test'
                with patch.object(runner,'DEFAULT',root),patch.object(runner,'FREEZE',freeze),\
                    patch.object(runner,'identity',return_value=ex),patch.object(runner,'full_selection',return_value=value),\
                    patch.object(runner,'_git_value',side_effect=git),patch.object(runner,'execute',side_effect=execute),\
                    patch.object(runner,'wait_resources',return_value={}),patch.object(runner,'final_audits',return_value=not fail),\
                    patch('builtins.print'):
                    self.assertEqual(runner.run(root),int(fail))
                    with self.assertRaises(FileExistsError):runner.run(root)
                self.assertEqual(len(calls),1 if fail else 90)
                end=json.loads((root/'run_terminal.json').read_text())
                self.assertEqual(len(end['unattempted']),89 if fail else 0)
                self.assertTrue(end['no_follow_on_run'])


if __name__=='__main__':unittest.main()
