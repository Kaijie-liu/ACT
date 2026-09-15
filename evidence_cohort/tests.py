import copy
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from evidence_cohort.run import loop,wait_owned,wait_resource,request,environment,resource_ok
from evidence_cohort.audit import roster,summarize,check_one,envelope
from evidence_cohort.ownership import info,stop_owned
from evidence_cohort.contract import ARMS


def jobs(n=2):
    return [{'rank':i,'dataset_index':100+i,'arm':ARMS[(i+j)%3],
             'job_id':f'rank{i}_{ARMS[(i+j)%3]}','position':j} for i in range(n) for j in range(3)]


class CohortTests(unittest.TestCase):
    def test_roster_order_and_fail_stop(self):
        plan=jobs();rows=[]
        end=loop(plan,lambda j:{**j,'status':'ERROR' if j==plan[2] else 'TIMEOUT'},rows.append)
        self.assertEqual(len(rows),3);self.assertEqual(end['unattempted'],plan[3:]);roster(rows,plan,end)
        for mutation in ('omit','reorder','continue','complete','hidden'):
            r=copy.deepcopy(rows);e=copy.deepcopy(end)
            if mutation=='omit':r.pop()
            elif mutation=='reorder':r.reverse()
            elif mutation=='continue':r.append({**plan[3],'status':'UNKNOWN'})
            elif mutation=='complete':e['state']='EXECUTION_COMPLETED'
            else:e['unattempted']=[]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):roster(r,plan,e)

    def test_aborted_and_all_unknown(self):
        plan=jobs();rows=[]
        def fail(j):raise RuntimeError('disk/gate/driver exception')
        e=loop(plan,fail,rows.append);self.assertEqual(e['aborted_job'],plan[0]);roster(rows,plan,e)
        e=loop(plan,lambda j:{**j,'status':'UNKNOWN'},rows.append)
        self.assertEqual(e['state'],'EXECUTION_COMPLETED');roster(rows,plan,e)

    def test_resource_wait_cap(self):
        now=[0.];seen=[]
        def sleep(t):now[0]+=t
        bad={'ram_gib':1,'disk_gib':100,'load_per_core':0}
        with patch('evidence_cohort.run.EXECUTION',{'resource':{'minimum_ram_gib':16,'minimum_disk_gib':5,
             'maximum_load_per_core':.5,'wait_limit_seconds':60,'poll_seconds':30}}):
            with self.assertRaises(TimeoutError):wait_resource(seen.append,lambda:now[0],sleep,lambda:bad)
        self.assertEqual(now[0],60);self.assertEqual(len(seen),3)
        self.assertFalse(resource_ok(bad))

    def test_owned_nested_sessions_not_other_process(self):
        sentinel=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(20)'],start_new_session=True)
        code=('import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-I","-S","-c",'
              '"import time;time.sleep(20)"],start_new_session=True);print(p.pid,flush=True);time.sleep(20)')
        p=subprocess.Popen([sys.executable,'-I','-S','-c',code],stdout=subprocess.PIPE,text=True,start_new_session=True)
        try:
            self.assertTrue(select.select([p.stdout],[],[],3)[0]);child=int(p.stdout.readline())
            result=wait_owned(p,time.monotonic()+.05)
            self.assertTrue(result['killed']);self.assertIn(child,[r['pid'] for r in result['killed_processes']])
            self.assertNotIn(sentinel.pid,[r['pid'] for r in result['killed_processes']]);self.assertIsNone(sentinel.poll())
            record=info(sentinel.pid);record['start']+=1
            self.assertEqual(stop_owned(record),[]);self.assertIsNone(sentinel.poll())
        finally:
            if p.poll() is None:stop_owned(info(p.pid));p.wait()
            p.stdout.close();sentinel.terminate();sentinel.wait()

    def test_summary_grades_cluster_and_conflicts(self):
        plan=jobs();rows=[];details=[]
        for j in plan:
            status=('CHECKED_CONDITIONAL' if j['arm']=='evidence' else 'POSITIVE' if j['arm']=='crown' else 'UNKNOWN')
            rows.append({**j,'status':status,'wall_seconds':10,'resource_wait':{'seconds':0},'outer_timeout':False})
            details.append({**j,'pairs':[[0,1],[0,2]],'facts':None,'replayed':False})
        s=summarize(rows,details,plan,True)
        self.assertEqual(s['comparisons']['matched']['paired_positive_difference'],1)
        self.assertEqual(s['comparisons']['crown']['descriptive_input_interval95'],[0,0])
        self.assertFalse(s['comparisons']['crown']['formal_SAFE_comparison'])
        self.assertEqual(s['common_fact_pairs_unavailable'],2)
        self.assertEqual(summarize(rows[:1],details[:1],plan,False)['comparisons'],{})
        rows[0]['status']='UNSAFE'
        with self.assertRaises(ValueError):summarize(rows,details,plan,True)
        rows[0]['status']='UNKNOWN';details[0]['facts']={'bounds':[1]};details[1]['facts']={'bounds':[2]}
        with self.assertRaises(ValueError):summarize(rows,details,plan,True)

    def test_real_outer_driver_and_fresh_audit(self):
        import torch
        from dataclasses import asdict
        from act.back_end.moe import OutputMoEFactoryConfig,GateKind
        from act.pipeline.moe.test_route_complexity_schedule import model,config
        from act.pipeline.moe.staged_verifier import _model_state_identity,_tensor_identity
        from scripts.optional_evidence_dev_contract import ROOT,save
        from scripts.test_general_evidence import fixture
        from portable_proof.runtime import digest
        net=model(((-.2,0.,-2.),(1.,0.,-2.),(2.,0.,-2.)))
        fc=OutputMoEFactoryConfig(input_shape=(2,),num_classes=3,num_experts=3,top_k=2,
            gate=GateKind.SELECTED_SOFTMAX,router_hidden=(),expert_hidden=(),seed=7)
        x=torch.full((1,2),.5,dtype=torch.float64);tensors={'center':x,'lower':x-.01,'upper':x+.01}
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root=Path(tmp);(root/'control').mkdir();cp=root/'control.pt';tp=root/'input.pt';cfg=root/'config.json'
            torch.save({'format':'act-output-moe-v1','factory_config':asdict(fc),'state_dict':net.state_dict()},cp)
            torch.save(tensors,tp);save(cfg,config('monolithic_f0'))
            sample={'dataset_index':-1,'label':0,**{k:_tensor_identity(v) for k,v in tensors.items()}}
            r=fixture()[1];r.update(epsilon=.01,model_state=_model_state_identity(net),**{k:sample[k] for k in tensors})
            req={'subject':{'checkpoint':str(cp),'checkpoint_sha256':digest(cp.read_bytes()),'model_state':r['model_state']},
                'sample':sample,'epsilon':.01,'config':{'path':str(cfg),'sha256':digest(cfg.read_bytes())},
                'tensors':{'path':str(tp),'sha256':digest(tp.read_bytes())},'evidence_request':r,'head':'analytic'}
            facts=[]
            for arm,status in (('matched','SAFE'),('evidence','CHECKED_CONDITIONAL')):
                j={'rank':0,'dataset_index':-1,'arm':arm,'job_id':'control_'+arm,'position':0}
                q=dict(req,method=arm);row=request(j,q,root,environment())
                row['resource_wait']={'seconds':0,'at_launch':{'ram_gib':20,'disk_gib':20,'load_per_core':0}}
                self.assertEqual(row['status'],status,[(p.name,p.read_text()) for p in (root/j['job_id']).glob('*.log')])
                self.assertLess(row['wall_seconds'],300);checked=check_one(root,row,q);facts.append(checked['facts'])
                bad=copy.deepcopy(row);bad['terminal_sha256']='changed'
                with self.assertRaises(ValueError):envelope(root/j['job_id'],bad,q)
            self.assertEqual(facts[0],facts[1])


if __name__=='__main__':unittest.main()
