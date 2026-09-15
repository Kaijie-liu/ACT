"""Analytic, mutation, portable and budget controls; never query real data."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from fractions import Fraction

from scripts.test_conv_pre_f0_r2 import hz
from act.back_end.solver.hz_lp_export import export,snapshot
from act.back_end.solver.lp_certificate import identity,rational
from act.back_end.solver.sparse_lp_certificate import evaluate
from act.back_end.solver.rational_mccormick import build
from moe_evidence.schema import (classification_properties,validate_request,pair_key,
    interval_lp,interval_certificate,gate_envelope)
from moe_evidence.generate import initial_manifest
from moe_evidence.checker import check_manifest
from portable_proof.runtime import original_bytes,digest


def box_certificate(lp):
    c={'lp_sha256':identity(lp),'inequality_dual':[0]*len(lp['b']),
       'equality_dual':[0]*len(lp['h'])}
    c['claimed_lower_bound']=str(evaluate(lp,c)[0]);return c


def fixture(experts=3,classes=3,base=3,partial=True,custom=False):
    request={'schema':'WEIGHTED_TOP2_REQUEST_V1','top_k':2,'experts':experts,'classes':classes,
        'gate':'selected_softmax','tie_policy':'ANY_LEGAL_TOPK','mode':'eval','epsilon':2/255,
        'clean_prediction':0,'properties':classification_properties(classes,0),
        **{k:{'sha256':k} for k in ('model_state','center','lower','upper')}}
    if custom:request['properties']=[{'q':['1/2']+[0]*(classes-1),'constant':'1/3'}]
    m=initial_manifest(request);rid=m['request_id'];files={}
    def put(name,obj):
        raw=original_bytes(obj);files[name]=obj;return {'file':name,'sha256':digest(raw)}
    pairs=list(itertools.combinations(range(experts),2))
    m['routes']={'feasible':[list(p) for p in pairs],'infeasible':[],'unresolved':[],'exact':True}
    scope={'request_id':rid,'frame_id':'frame','gate':request['gate'],'tie_policy':request['tie_policy']}
    # These intervals are a valid (deliberately loose) envelope for constants.
    common={'request_id':rid,'identity':{k:request[k] for k in ('model_state','center','lower','upper')},
        'scope':scope,'pairs':[list(p) for p in pairs],'branches':[
            {'expert':e,'interval':{'lower':[base]+[0]*(classes-1),'upper':[base]+[1]*(classes-1)}} for e in range(experts)]}
    m['common_facts']=put('common.json',common)
    joint=hz(([base]+[0]*(classes-1))*2);router=hz([0]*experts)
    def support(key,pair,index,kind,hz,q):
        rec=export(hz,q,sparse=True)
        m['supports'][key]={'request_id':rid,'pair':list(pair),'property_index':index,'kind':kind,
            'source_sha256':rec['source_sha256'],'export':put(key+'.json',rec),
            'certificate':put(key+'.cert.json',box_certificate(rec['lp'])),'status':'PROPOSED'}
    for pair in pairs:
        key=pair_key(pair);residual=False
        for i,prop in enumerate(request['properties']):
            row={'pair':list(pair),'property_index':i,'property':copy.deepcopy(prop)}
            if partial and i==0 and base>1:
                row.update(kind='reused',facts=[{'expert':e,'request_id':rid,'scope':copy.deepcopy(scope),
                    'property_index':i,'guard':'TOP2_MEMBERSHIP','interval':copy.deepcopy(common['branches'][e]['interval']),
                    'certificate':interval_certificate(interval_lp(common['branches'][e]['interval'],prop,classes))} for e in pair])
            else:
                residual=True;qd=prop['q']+[str(-rational(v)) for v in prop['q']];pkey=key+f'_p{i}'
                support(pkey+'_lo',pair,i,'difference',joint,qd)
                support(pkey+'_hi',pair,i,'difference',joint,[str(-rational(v)) for v in qd])
                rec=build(snapshot(joint),prop['q'],prop['constant'],['1/2','1/2'],['0','0'])
                row.update(kind='residual',gate_bounds=['1/2','1/2'],difference_bounds=['0','0'],weighted_status='PROPOSED',
                    weighted=put(pkey+'_weighted.json',rec),certificate=put(pkey+'_weighted.cert.json',box_certificate(rec['lp'])))
            m['obligations'].append(row)
        if residual:
            m['contexts'][key]={'request_id':rid,'pair':list(pair),'expert_order':list(pair),
                'joint_source':put(key+'_joint.json',snapshot(joint)),'router_source':put(key+'_router.json',snapshot(router))}
            q=[0]*experts;q[pair[0]]=1;q[pair[1]]=-1
            support(key+'_gate_lo',pair,None,'router_order',router,q)
            support(key+'_gate_hi',pair,None,'router_order',router,[-v for v in q])
    m['generation_complete']=True
    return m,request,files


def checked(a,**kw):
    def load(ref):
        obj=a[2][ref['file']]
        if digest(original_bytes(obj))!=ref['sha256']:raise ValueError('hash mismatch')
        return obj
    return check_manifest(a[0],a[1],load,**kw)


class GeneralEvidenceTests(unittest.TestCase):
    def test_dimensions_ties_and_partial_reuse(self):
        for e,c in ((2,2),(3,3),(5,4)):
            for partial in (False,True):
                with self.subTest(e=e,c=c,partial=partial):
                    r=checked(fixture(e,c,partial=partial))
                    self.assertEqual(r['status'],'CHECKED_CONDITIONAL')
                    self.assertEqual(r['positive_obligations'],e*(e-1)//2*(c-1))
                    self.assertFalse(r['deployed_float_SAFE'])

    def test_arbitrary_requested_property(self):
        r=checked(fixture(4,5,partial=False,custom=True))
        self.assertEqual(r['required_obligations'],6)
        self.assertEqual(Fraction(r['minimum_lower_bound']),Fraction(11,6))

    def test_no_solver_or_builder_in_check(self):
        a=fixture()
        with patch('scipy.optimize.linprog',side_effect=AssertionError),patch('act.back_end.solver.rational_mccormick.build',side_effect=AssertionError):
            self.assertEqual(checked(a)['status'],'CHECKED_CONDITIONAL')

    def test_binding_and_coverage_mutations(self):
        for what in ('request','missing','duplicate','pair','route','unresolved','order','property','scope','expert','source','context','kind','threshold'):
            a=fixture();m,r,files=a;row=m['obligations'][0]
            if what=='request':m['request_id']='other'
            elif what=='missing':m['obligations'].pop()
            elif what=='duplicate':m['obligations'][-1]=copy.deepcopy(row)
            elif what=='pair':row['pair']=[1,0]
            elif what=='route':m['routes']['feasible'].pop()
            elif what=='unresolved':m['routes']['unresolved']=[[0,1]]
            elif what=='order':m['contexts']['s0_1']['expert_order']=[1,0]
            elif what=='property':row['property']['constant']=1
            elif what=='scope':row['facts'][0]['scope']['frame_id']='other'
            elif what=='expert':row['facts'][0]['expert']=2
            elif what=='source':m['supports']['s0_1_p1_lo']['source_sha256']='other'
            elif what=='context':del m['contexts']['s0_1']
            elif what=='kind':row['kind']='auto_accept'
            else:m['positive_threshold']=0
            with self.subTest(what=what),self.assertRaises(ValueError):checked(a)

    def test_semantic_mutations_after_rehash(self):
        for what in ('plane','objective','cross_pair','inward','range','interval','frame'):
            a=fixture();m,r,files=a;row=m['obligations'][1]
            if what=='inward':row['gate_bounds']=['3/4','1'];ref=None
            elif what=='range':row['difference_bounds']=['1','1'];ref=None
            elif what=='plane':ref=row['weighted'];files[ref['file']]['lp']['b'][0]='99'
            elif what=='objective':ref=m['supports']['s0_1_p1_lo']['export'];files[ref['file']]['q'][0]=-1
            elif what=='cross_pair':
                item=m['supports']['s0_1_p1_lo'];item['pair']=[1,2];ref=None
            elif what=='interval':
                ref=m['common_facts'];files[ref['file']]['branches'][0]['interval']['lower'][0]=100
            else:
                ref=m['contexts']['s0_1']['joint_source'];files[ref['file']]['frame_id']='other'
            if ref:ref['sha256']=digest(original_bytes(files[ref['file']]))
            with self.subTest(what=what),self.assertRaises(ValueError):checked(a)

    def test_unknown_not_unsafe(self):
        a=fixture(base=-2,partial=False)
        self.assertEqual(checked(a)['status'],'UNKNOWN_NONPOSITIVE')
        a=fixture();m=a[0]
        m['routes']['exact']=False
        self.assertEqual(checked(a)['status'],'UNKNOWN_ROUTE_COVERAGE')
        a=fixture();row=a[0]['obligations'][1];item=a[0]['supports']['s0_1_p1_lo']
        item.update(status='UNAVAILABLE',certificate=None)
        row.update(weighted_status='RANGE_UNAVAILABLE',weighted=None,certificate=None)
        self.assertEqual(checked(a)['status'],'UNKNOWN_MISSING_EVIDENCE')

    def test_gate_rule(self):
        self.assertEqual(gate_envelope(None,None),['0','1'])
        self.assertEqual(gate_envelope(Fraction(0),None),['1/2','1'])
        self.assertEqual(gate_envelope(None,Fraction(0)),['0','1/2'])
        self.assertEqual(gate_envelope(Fraction(0),Fraction(0)),['1/2','1/2'])
        with self.assertRaises(ValueError):gate_envelope(Fraction(1),Fraction(1))

    def test_checker_deadline(self):
        from scripts.optional_evidence_budget import EvidenceBudget,EvidenceBudgetExpired,terminal_status
        now=[0.];b=EvidenceBudget(0.,clock=lambda:now[0])
        def tick():now[0]+=20;b.remaining(2)
        with self.assertRaises(EvidenceBudgetExpired):checked(fixture(),tick=tick)
        self.assertEqual(terminal_status('CHECKED_CONDITIONAL',301,300,True),'TIMEOUT')
        self.assertEqual(terminal_status('CHECKED_CONDITIONAL',100,300,False),'UNKNOWN_INCOMPLETE_EVIDENCE')

    def test_portable_outside_checkout(self):
        from moe_evidence.bundle import pack
        from scripts.optional_evidence_dev_contract import save
        a=fixture();result=checked(a)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            for name,obj in a[2].items():save(source/name,obj)
            save(source/'manifest.json',a[0]);meta=pack(source,root/'portable',result)
            command=[sys.executable,'-I','-S',str(root/'portable/verify.py'),'--bundle-hash',meta['bundle_sha256'],
                '--statement-hash',meta['statement_sha256']]
            run=subprocess.run(command,cwd=root,capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(run.stdout)['result'],result)
            (root/'portable/evidence.zip').write_bytes(b'changed')
            self.assertNotEqual(subprocess.run(command,cwd=root,capture_output=True).returncode,0)

    def test_real_capture_all_tie_pairs_and_partial_reuse(self):
        import time
        import torch
        from act.pipeline.moe.test_route_complexity_schedule import model,config
        from act.pipeline.moe.staged_verifier import _model_state_identity,_tensor_identity
        from moe_evidence.generate import capture,propose_all
        from moe_evidence.storage import loader
        from scripts.optional_evidence_budget import EvidenceBudget
        net=model(((-.2,0.,-2.),(1.,0.,-2.),(2.,0.,-2.)))
        x=torch.full((1,2),.5,dtype=torch.float64)
        tensors={'center':x,'lower':x-.01,'upper':x+.01}
        r=fixture()[1];r.update(epsilon=.01,model_state=_model_state_identity(net),
                               **{k:_tensor_identity(v) for k,v in tensors.items()})
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);budget=EvidenceBudget(time.monotonic())
            with patch('act.back_end.moe.weighted_top2.build_weighted_top2_f0',side_effect=AssertionError('float F0 called')):
                capture(net,tensors,r,config('monolithic_f0'),root,budget)
            m=json.loads((root/'manifest.json').read_text())
            self.assertEqual(len(m['obligations']),6)
            self.assertEqual(m['routes']['feasible'],[[0,1],[0,2],[1,2]])
            self.assertEqual(sum(v['kind']=='residual' for v in m['obligations']),2)
            propose_all(root,budget)
            result=check_manifest(json.loads((root/'manifest.json').read_text()),r,loader(root))
            self.assertEqual(result['status'],'CHECKED_CONDITIONAL')
            self.assertEqual(result['required_obligations'],6)

    def test_outer_timeout_and_late_positive(self):
        import os,time
        from moe_evidence.execution import phase,accept,PHASES
        from scripts.optional_evidence_budget import EvidenceBudget
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp)
            # Fixed 300s budget, with only 50ms work allowance remaining.
            b=EvidenceBudget(time.monotonic()-297.95)
            r=phase([sys.executable,'-I','-S','-c','import time;time.sleep(5)'],root,'check',b,os.environ.copy())
            self.assertEqual(r['state'],'OUTER_TIMEOUT')
            calls=[]
            all_ok={k:{'state':'COMPLETED'} for k in PHASES['evidence']}
            self.assertEqual(accept('evidence',all_ok,lambda:calls.append(True),301),('TIMEOUT',False))
            all_ok['check']=r
            self.assertEqual(accept('evidence',all_ok,lambda:calls.append(True),299),('TIMEOUT',False))
            self.assertFalse(calls)

    def test_process_pipeline_and_terminal_audit(self):
        import os
        from dataclasses import asdict
        import torch
        from act.back_end.moe import OutputMoEFactoryConfig,GateKind
        from act.pipeline.moe.test_route_complexity_schedule import model,config
        from act.pipeline.moe.staged_verifier import _model_state_identity,_tensor_identity
        from scripts.optional_evidence_dev_contract import save,ROOT
        from moe_evidence.execution import run_request
        from moe_evidence.audit import audit_request
        net=model(((-.2,0.,-2.),(1.,0.,-2.),(2.,0.,-2.)))
        x=torch.full((1,2),.5,dtype=torch.float64);values={'center':x,'lower':x-.01,'upper':x+.01}
        fc=OutputMoEFactoryConfig(input_shape=(2,),num_classes=3,num_experts=3,top_k=2,
            gate=GateKind.SELECTED_SOFTMAX,router_hidden=(),expert_hidden=(),seed=7)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root=Path(tmp);cp=root/'control.pt';tensor=root/'input.pt';cfg=root/'config.json'
            torch.save({'format':'act-output-moe-v1','factory_config':asdict(fc),'state_dict':net.state_dict()},cp)
            torch.save(values,tensor);save(cfg,config('monolithic_f0'))
            sample={'dataset_index':-1,'label':0,**{k:_tensor_identity(v) for k,v in values.items()}}
            r=fixture()[1];r.update(epsilon=.01,model_state=_model_state_identity(net),**{k:sample[k] for k in values})
            req={'subject':{'checkpoint':str(cp),'checkpoint_sha256':digest(cp.read_bytes()),'model_state':r['model_state']},
                'sample':sample,'epsilon':.01,'config':{'path':str(cfg),'sha256':digest(cfg.read_bytes())},
                'tensors':{'path':str(tensor),'sha256':digest(tensor.read_bytes())},'evidence_request':r,'head':'analytic-control'}
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
            facts=[]
            for arm,expected in (('matched','SAFE'),('evidence','CHECKED_CONDITIONAL')):
                terminal=run_request(arm,dict(req,method=arm),root/arm,env)
                self.assertEqual(terminal['status'],expected, (terminal,[(p.name,p.read_text()) for p in (root/arm).glob('*.log')]))
                checked=audit_request(root/arm,dict(req,method=arm),arm)
                self.assertEqual(checked['status'],'PASS');facts.append(checked['facts'])
            self.assertEqual(facts[0],facts[1])

    def test_clean_selection_is_endpoint_blind(self):
        from act.pipeline.moe.freeze_conv_three_arm import choose_indices
        from scripts.freeze_general_evidence import jobs
        rows=[{'dataset_index':i,'label':0,'prediction':int(i%3==0),'pair_count':i,
               'positive_bound':100-i} for i in range(12)]
        selected=choose_indices(rows,{1,2},4,0)
        for row in rows:row.update(pair_count=-1,positive_bound=-1000)
        self.assertEqual(choose_indices(rows,{1,2},4,0),selected)
        self.assertEqual(selected,([],[4,5,7,8]))
        roster=jobs([{'dataset_index':i} for i in range(20)])
        self.assertEqual(len(roster),60)
        for i in range(20):self.assertEqual({j['arm'] for j in roster if j['rank']==i},{'matched','evidence','crown'})

    def test_reserve_keeps_missing_obligations(self):
        from scripts.optional_evidence_dev_contract import save
        from scripts.optional_evidence_budget import EvidenceBudget
        from moe_evidence.generate import propose_all
        a=fixture();m=a[0]
        for item in m['supports'].values():item.update(status='PENDING',certificate=None)
        for row in m['obligations']:
            if row['kind']=='residual':row.update(weighted_status='RANGE_UNAVAILABLE',weighted=None,certificate=None)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            d=Path(tmp);save(d/'manifest.json',m)
            with patch('scipy.optimize.linprog',side_effect=AssertionError('reserve must be retained')):
                propose_all(d,EvidenceBudget(0,clock=lambda:221))
            self.assertEqual(json.loads((d/'manifest.json').read_text()),m)
        self.assertEqual(checked(a)['status'],'UNKNOWN_MISSING_EVIDENCE')


if __name__=='__main__':unittest.main()
