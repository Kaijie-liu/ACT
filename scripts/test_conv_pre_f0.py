import copy
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.solver.hz_lp_export import export,snapshot
from act.back_end.solver.lp_certificate import identity
from act.back_end.solver.rational_mccormick import build
from scripts.test_conv_request_sign_lp import fixture as old_fixture
from scripts.check_conv_pre_f0 import aggregate,order_bounds,TRUSTED
from scripts.check_conv_request_sign_lp import expected_scope,property_vector


def hz(c,coeff=None):
    return SparseHZono(c=np.array(c,dtype=float),Gc=sp.csr_matrix(np.array(coeff or [0]*len(c),dtype=float).reshape(-1,1)),
        Gb=sp.csr_matrix((len(c),0)),Ac=sp.csr_matrix((0,1)),Ab=sp.csr_matrix((0,0)),b=np.array([]),
        Auc=sp.csr_matrix((0,1)),Aub=sp.csr_matrix((0,0)),ub=np.array([]),frame_id='frame')


def fixture(base=2):
    _,s,j,_=old_fixture();j['case']={'expected_pairs':[[1,2]],'dataset_index':98,'job_id':'rank24_monolithic'}
    j['parent_request']['sample'].update(dataset_index=98,label=0);j['competitors']=list(range(1,10))
    req=expected_scope(j);s['identity']['property']['clean_prediction']=0;s['feasible_route_sets']=[[1,2]]
    joint=hz([base+1]+[0]*9+[base]+[0]*9,[.1]+[0]*9+[.1]+[0]*9);router=hz([0,.5,0,-1])
    records={'joint':snapshot(joint),'router':snapshot(router)}
    m={'schema':'CONV_PRE_F0_RATIONAL_R1_MANIFEST','request':req,'trusted_base':TRUSTED,'positive_threshold':1e-7,
       'routes':{'feasible':[[1,2]],'infeasible':[[0,1],[0,2],[0,3],[1,3],[2,3]],'unresolved':[],'exact':True},
       'joint_source':'joint','router_source':'router','expert_order':[1,2],'generation_complete':True,'supports':{},'obligations':[]}
    def support(key,kind,index,obj,q,claim):
        rec=export(obj,q,sparse=True);records[key]=rec
        records[key+'_cert']={'lp_sha256':identity(rec['lp']),'inequality_dual':[],'equality_dual':[],'claimed_lower_bound':claim}
        m['supports'][key]={'kind':kind,'property_index':index,'pair':[1,2],'request':req,'source_sha256':rec['source_sha256'],
                            'status':'PROPOSED','export':key,'certificate':key+'_cert'}
    support('gate_lower','router_order',None,router,[0,1,-1,0],'.5')
    support('gate_upper','router_order',None,router,[0,-1,1,0],'-.5')
    for i,c in enumerate(range(1,10)):
        q=property_vector(0,c);qd=q+[-v for v in q];lower=f'p{i}_lower';upper=f'p{i}_upper';key=f'p{i}_weighted'
        support(lower,'difference',i,joint,qd,1);support(upper,'difference',i,joint,[-v for v in qd],-1)
        rec=build(snapshot(joint),q,0,['1/2',1],[1,1]);records[key]=rec
        records[key+'_cert']={'lp_sha256':identity(rec['lp']),'inequality_dual':[0]*4,'equality_dual':[],'claimed_lower_bound':base}
        m['obligations'].append({'pair':[1,2],'property_index':i,'competitor':c,'q':q,'constant':0,'kind':'residual',
            'difference_lower':lower,'difference_upper':upper,'difference_bounds':['1','1'],'gate_bounds':['1/2','1'],
            'weighted_status':'PROPOSED','weighted':key,'certificate':key+'_cert'})
    return m,s,j,records


class PreF0Tests(unittest.TestCase):
    def check(self,a):return aggregate(*a[:3],a[3].__getitem__)

    def test_complete_without_builder_or_solver(self):
        a=fixture()
        with (patch('act.back_end.solver.rational_mccormick.build',side_effect=AssertionError('builder called')),
              patch('scipy.optimize.linprog',side_effect=AssertionError('solver called'))):
            r=self.check(a)
        self.assertEqual(r['positive_obligations'],9);self.assertFalse(r['floating_F0_construction_trusted'])
        self.assertEqual(r['status'],'CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_PRE_F0_LOWERING')

    def test_mutated_bindings_and_construction(self):
        for change in ('gate','diff','source','order','q','plane','missing','duplicate','route'):
            a=fixture();row=a[0]['obligations'][0]
            if change=='gate':row['gate_bounds']=['.75','1']
            elif change=='diff':row['difference_bounds']=['1.1','1.1']
            elif change=='source':a[0]['supports']['p0_lower']['source_sha256']='other'
            elif change=='order':a[0]['expert_order']=[2,1]
            elif change=='q':row['q'][0]=-1
            elif change=='plane':a[3][row['weighted']]['lp']['b'][0]='100'
            elif change=='missing':a[0]['obligations'].pop()
            elif change=='duplicate':a[0]['obligations'][-1]=copy.deepcopy(row)
            else:a[0]['routes']['infeasible'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(a)

    def test_unavailable_ranges_and_nonpositive(self):
        a=fixture();a[0]['supports']['p0_lower']['status']='UNAVAILABLE';r=a[0]['obligations'][0]
        r.update(weighted_status='RANGE_UNAVAILABLE',weighted=None)
        self.assertEqual(self.check(a)['status'],'UNKNOWN')
        negative=self.check(fixture(-2));self.assertEqual(negative['status'],'UNKNOWN')
        self.assertEqual(negative['positive_obligations'],0);self.assertFalse(negative['deployed_float_SAFE'])

    def test_order_including_ties_and_missing(self):
        self.assertEqual(order_bounds(None,None),[0,1])
        self.assertEqual(order_bounds(0,None),[.5,1])
        self.assertEqual(order_bounds(None,0),[0,.5])
        self.assertEqual(order_bounds(0,0),[.5,.5])
        with self.assertRaises(ValueError):order_bounds(1,1)

    def test_stdlib_only(self):
        code=('import json,sys;from scripts.check_conv_sign_lp import isolate;isolate();'
              'from scripts.check_conv_pre_f0 import aggregate;a=json.load(sys.stdin);'
              'r=aggregate(*a[:3],a[3].__getitem__);'
              'assert not any(x in sys.modules for x in ("torch","numpy","scipy","act.back_end.solver.rational_mccormick"));'
              'print(json.dumps(r))')
        r=subprocess.run([sys.executable,'-S','-c',code],input=json.dumps(fixture()),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(r.stdout)['positive_obligations'],9)

    def test_scoped_reuse_and_wrong_request(self):
        from act.pipeline.moe.scoped_f0_proofs import facts_from_branches,reuse_property
        a=fixture();lo=[3]+[0]*9;hi=[4]+[1]*9
        a[1]['branches']=[{'candidate':e,'proof_output_bounds':{'lower':lo,'upper':hi}} for e in (1,2)]
        facts=facts_from_branches(a[1]['branches'],a[1]['scope']);row=a[0]['obligations'][7]
        row.update(kind='reused',proof=reuse_property(facts,a[1]['scope'],(1,2),7))
        for k in ('p7_lower','p7_upper'):del a[0]['supports'][k]
        self.assertEqual(self.check(a)['reused_positive'],1)
        row['proof']['proof_sources'][0]['scope']=dict(a[1]['scope'],frame_id='other')
        with self.assertRaises(ValueError):self.check(a)

    def test_capture_stops_before_float_f0(self):
        from scripts import run_conv_pre_f0 as run
        import act.pipeline.moe.paired_monolithic as mono
        from act.back_end.moe.hz_routing import TopKSetReport
        a=fixture();j=a[2];req=j['parent_request'];req['config']={'path':'config'}
        req['subject'].update(checkpoint='cp',checkpoint_sha256='hash');j['protocol']={'positive_threshold':1e-7}
        config={'support':{'lp_neurons':0,'milp_neurons':0,'lp_time_limit':1,'milp_time_limit':1},
                'solver':{'margin_support_seconds':1,'difference_support_seconds':1}}
        router=hz([0,.5,0,-1]);joint=hz([3]+[0]*9+[2]+[0]*9)
        routes=TopKSetReport(((1,2),),tuple(tuple(p) for p in a[0]['routes']['infeasible']),(),(),True)
        internal={'route_sets':routes,'output_spec':'spec','program':SimpleNamespace(experts=list(range(4))),
                  'router':SimpleNamespace(output_hz=router,input_hz=router)}
        budget=SimpleNamespace(check=lambda *a:None,limit=lambda *a:1);original=mono._run_monolithic
        with tempfile.TemporaryDirectory(dir=run.ROOT/'data/moe/results') as tmp,ExitStack() as stack:
            d=Path(tmp);run.save(d/'job.json',j)
            def fake_verify(*args,**kw):
                run.save(kw['journal_path'],[]);kw['common_fact_callback']({'payload':a[1]})
                return mono._run_monolithic(model=None,center=None,clean_prediction=0,internal=internal,config=config,budget=budget)
            stack.enter_context(patch.object(run,'validate_job',return_value=j));stack.enter_context(patch.object(run,'read',return_value=config))
            stack.enter_context(patch('scripts.budget_contract_v2.verify_v2',side_effect=fake_verify))
            stack.enter_context(patch('act.pipeline.moe.external_pair_worker.load',return_value=(None,{'center':None})))
            stack.enter_context(patch.object(mono,'linear_safety_rows',return_value=[(property_vector(0,c),0) for c in range(1,10)]))
            for k in ('condition_topk_set','guarded_input_topk_set'):stack.enter_context(patch.object(mono,k,return_value=SimpleNamespace(hz=router)))
            stack.enter_context(patch.object(mono,'shared_input_pair_propagation',return_value=SimpleNamespace(joint=SimpleNamespace(output_hz=joint))))
            run.capture_worker(d);m=json.loads((d/'generation.json').read_text())
            self.assertTrue(m['generation_complete']);self.assertEqual(len(m['supports']),20);self.assertEqual(len(m['obligations']),9)
        self.assertIs(mono._run_monolithic,original)


if __name__=='__main__':unittest.main()
