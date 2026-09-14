"""Orchestrator conformance controls; no model endpoint queries."""
import copy
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.conv_three_arm_contract import read, SELECTION, request_for, gate, ARMS, ROOT
from scripts.run_conv_three_arm import execute, resources_ok, terminal
from scripts.audit_conv_three_arm import route_inventory, terminal_contract, external_result


class ContractTests(unittest.TestCase):
    def test_frozen_requests_and_order(self):
        s=read(SELECTION)
        self.assertEqual([r['method'] for r in s['smoke_jobs']],
                         ['adaptive','monolithic','crown','monolithic','crown','adaptive'])
        for job in s['smoke_jobs']:
            r=request_for(s,{**job,'status':'UNKNOWN'},'head')
            self.assertEqual(r['topology'],{'num_experts':4,'top_k':2,'classes':10})
            self.assertEqual(r['tensors'],s['materialized_inputs'][str(job['dataset_index'])])
        with self.assertRaises(ValueError):request_for(s,{**s['smoke_jobs'][0],'dataset_index':99},'head')

    def test_gate_is_conformance_not_effect(self):
        rows=[{'status':'UNKNOWN'}]*6;complete={a:1 for a in ARMS}
        self.assertTrue(gate(rows,complete))
        self.assertFalse(gate(rows[:5],complete))
        self.assertFalse(gate(rows,{**complete,'crown':0}))
        self.assertFalse(gate(rows[:5]+[{'status':'ERROR'}],complete))

    def test_resource_gate(self):
        good={'available_ram_gib':16,'free_disk_gib':5,'load_per_core':.5}
        self.assertTrue(resources_ok(good))
        for k,v in [('available_ram_gib',15.9),('free_disk_gib',4.9),('load_per_core',.51)]:
            self.assertFalse(resources_ok({**good,k:v}))

    def test_six_route_universe(self):
        pairs=list(map(list,itertools.combinations(range(4),2)))
        record={'feasible':pairs,'infeasible':[],'unresolved':[],'exact':True,
                'branches':[{'route_set':p,'feasibility':'feasible'} for p in pairs]}
        self.assertTrue(route_inventory(record))
        for key in ('feasible','branches'):
            bad=copy.deepcopy(record);bad[key].pop()
            with self.assertRaises(ValueError):route_inventory(bad)
        bad=copy.deepcopy(record);bad['branches'][0]['feasibility']='infeasible'
        with self.assertRaises(ValueError):route_inventory(bad)
        record['exact']=False
        self.assertFalse(route_inventory(record))

    def test_timeout_never_promotes(self):
        row={'budget_seconds':300,'wall_seconds':301,'outer_timeout':True,'status':'TIMEOUT',
             'package':None,'return_code':-9}
        terminal_contract(row)
        for update in ({'status':'SAFE'},{'package':'late'},{'outer_timeout':False}, {'wall_seconds':float('nan')}):
            with self.assertRaises(ValueError):terminal_contract({**row,**update})

    def test_process_group_deadline(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            path=Path(d);pidfile=path/'child.pid'
            command=[sys.executable,'-c',
                'import subprocess,time,pathlib,sys; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)']
            code,expired=execute(command,path/'log',time.monotonic(),os.environ,budget=1)
            self.assertTrue(expired);self.assertNotEqual(code,0)
            pid=int(pidfile.read_text());stat=Path(f'/proc/{pid}/stat')
            # Reparented killed descendants may remain zombies briefly.
            self.assertTrue(not stat.exists() or stat.read_text().split()[2]=='Z')

    def test_exception_cleans_process_group(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            with patch('scripts.run_conv_three_arm.subprocess.Popen') as cls:
                proc=cls.return_value;proc.pid=999999999
                proc.wait.side_effect=[RuntimeError('wait failed'),-9]
                with patch('scripts.run_conv_three_arm.os.killpg') as kill:
                    with self.assertRaises(RuntimeError):
                        execute([sys.executable,'-c','pass'],Path(d)/'log',time.monotonic(),os.environ)
                    kill.assert_called_once()

    def test_late_package_preserved_but_not_counted(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            root=Path(d);job=read(SELECTION)['smoke_jobs'][0];directory=root/job['job_id'];directory.mkdir()
            (directory/'request.json').write_text('{}')
            (directory/'package').mkdir();(directory/'package/manifest.json').write_text('{"status":"SAFE"}')
            row=terminal(root,job,time.monotonic()-301,0,True,{'seconds':0,'at_launch':{}})
            self.assertEqual(row['status'],'TIMEOUT');self.assertIsNone(row['package'])
            self.assertTrue((directory/'package/manifest.json').exists())
            self.assertEqual(json.loads((directory/'terminal.json').read_text()),row)

    def test_external_incomplete_route_not_positive(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            root=Path(d);directory=root/'job';directory.mkdir();request={'test':True}
            pairs=list(map(list,itertools.combinations(range(4),2)))
            route={'request':request,'routes':{'feasible':[],'infeasible':[], 'unresolved':pairs,'exact':False,
                'branches':[{'route_set':p,'feasibility':'unknown'} for p in pairs]}}
            (directory/'routes.json').write_text(json.dumps(route))
            (directory/'external.json').write_text(json.dumps({'status':'POSITIVE'}))
            from act.pipeline.moe.experiment1 import _sha256
            row={'job_id':'job','outer_timeout':False,'return_code':0,'status':'POSITIVE',
                 'routes_sha256':_sha256(directory/'routes.json'),'external_sha256':_sha256(directory/'external.json')}
            with self.assertRaises(ValueError):external_result(root,row,request)

    def test_external_property_and_environment_mutations(self):
        from act.pipeline.moe.external_compatibility import TOOL
        from act.pipeline.moe.check_request_lp import property_row
        from act.pipeline.moe.experiment1 import _sha256
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as d:
            root=Path(d);directory=root/'job';directory.mkdir()
            request=request_for(read(SELECTION),read(SELECTION)['smoke_jobs'][2],'head')
            universe=list(map(list,itertools.combinations(range(4),2)))
            route={'request':request,'routes':{'feasible':[[0,1]],'infeasible':universe[1:],
                'unresolved':[],'exact':True,'branches':[{'route_set':p,'feasibility':'feasible' if p==[0,1] else 'infeasible'} for p in universe]}}
            (directory/'routes.json').write_text(json.dumps(route))
            result={'status':'POSITIVE','model_state':request['subject']['model_state'],
                'tensor_identity':{k:request['sample'][k] for k in ('center','lower','upper')},
                'C':[[property_row(10,request['sample']['label'],i) for i in range(9)]],
                'environment':{'python':'3.11.16 test','torch':'2.11.0+cu130','auto_lirpa_file':str(TOOL/'auto_LiRPA/auto_LiRPA/__init__.py'),
                               'device':'cpu','dtype':'float64','threads':1},
                'backend':{'method':'CROWN','bound_opts':{'conv_mode':'matrix'}},'formal_SAFE':False,
                'pairs':[{'pair':[0,1],'lower':[1.]*9,'upper':[2.]*9,'concrete_max_error':0.,'lowered_max_error':0.}]}
            def check(v):
                (directory/'external.json').write_text(json.dumps(v))
                row={'job_id':'job','outer_timeout':False,'return_code':0,'status':'POSITIVE',
                    'routes_sha256':_sha256(directory/'routes.json'),'external_sha256':_sha256(directory/'external.json')}
                return external_result(root,row,request)
            self.assertTrue(check(result)['complete'])
            for mutate in (lambda v:v['pairs'].clear(),lambda v:v['pairs'][0]['lower'].pop(),
                    lambda v:v['pairs'][0]['lower'].__setitem__(0,-1),
                    lambda v:v['C'][0][0].__setitem__(0,999),
                    lambda v:v['environment'].__setitem__('dtype','float32'),
                    lambda v:v.__setitem__('formal_SAFE',True)):
                bad=copy.deepcopy(result);mutate(bad)
                with self.assertRaises(ValueError):check(bad)


if __name__=='__main__':unittest.main()
