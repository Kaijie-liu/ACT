"""Frozen finite HZ endpoint controls. No solver, CUDA or model/data load."""
import ast
import copy
from fractions import Fraction as F
from itertools import combinations
import json
import math
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.moe import hz_endpoints as api
from act.back_end.moe import check_hz_endpoints as receive
from act.back_end.moe.check_batched_support import validated_records
from act.back_end.solver.solver_hz import SparseHZono
from scoped_source.endpoint_controls import mccormick, exact_point
from scoped_source.rowwise_bound import identity, rational, clock
from scoped_source.rowwise_native import _evaluate

ROOT=Path(__file__).resolve().parents[3]
CONTEXT={'request':'frozen-hz-endpoint-controls','domain':'x-minus-plus-one','guard':'all-pair-guards'}


def hz(c,gc,gb=None,ac=None,ab=None,b=(),auc=None,aub=None,ub=(),frame=771):
    gc=sp.csr_matrix(gc,dtype=np.float64); nc=gc.shape[1]
    gb=sp.csr_matrix((len(c),0)) if gb is None else sp.csr_matrix(gb,dtype=np.float64); nb=gb.shape[1]
    return SparseHZono(c=np.asarray(c,dtype=np.float64),Gc=gc,Gb=gb,
        Ac=sp.csr_matrix((len(b),nc)) if ac is None else sp.csr_matrix(ac,dtype=np.float64),
        Ab=sp.csr_matrix((len(b),nb)) if ab is None else sp.csr_matrix(ab,dtype=np.float64),b=np.asarray(b,dtype=np.float64),
        Auc=sp.csr_matrix((len(ub),nc)) if auc is None else sp.csr_matrix(auc,dtype=np.float64),
        Aub=sp.csr_matrix((len(ub),nb)) if aub is None else sp.csr_matrix(aub,dtype=np.float64),ub=np.asarray(ub,dtype=np.float64),frame_id=frame)


def separation():
    cfg=json.loads((ROOT/'configs/hz_endpoint_controls_20261001.json').read_text())['separation_fixture']
    scores=[[F(v) for v in row] for row in cfg['scores']]; pairs=[]
    for pair,gate in zip(combinations(range(3),2),cfg['gate_intervals_by_pair']):
        outsider=next(i for i in range(3) if i not in pair)
        rows=[[float(scores[outsider][0]-scores[i][0])] for i in pair]
        rhs=[float(scores[i][1]-scores[outsider][1]) for i in pair]
        entry=hz([0],[[1]],auc=rows,ub=rhs)
        experts=[]
        for i in pair:
            # p=(zp+1)/2 and n=(zn+1)/2, each expert's factors private.
            matrix=[r+[0,0] for r in rows]+[[1,-.5,0],[-1,0,-.5],[-1,1,0],[1,0,1]]
            experts.append(hz([float(F(cfg['margin_constant'])+F(1,4)),0],
                [[float(F(cfg['linear_coefficients'][i])),.125,.125],[0,0,0]],
                auc=matrix,ub=rhs+[.5,.5,0,0]))
        pairs.append({'pair':list(pair),'entry':entry,'a':experts[0],'b':experts[1],'gate':gate})
    return pairs,[{'id':'margin','q':[1,-1],'offset':0}],3,2


def relation(gate=('1/2','1/2'),offset=0):
    entry=hz([0],[[1]])
    a=hz([.25,0],[[1],[0]]); b=hz([.25,0],[[-1],[0]])
    return [{'pair':[0,1],'entry':entry,'a':a,'b':b,'gate':list(gate)}],[{'id':'margin','q':[1,-1],'offset':offset}],2,2


def manual_proof(request, deadline):
    proof={'schema':receive.PROOF,'request_sha256':identity(request),'pairs':[]}
    for p in request['pairs']:
        batch=p['batch']; entries=[]
        for q,lp in validated_records(batch,expected_batch_sha256=identity(batch),deadline=deadline):
            y=[F(0)]*len(lp['b']); t=F(q['q'][0])
            if p['pair']==[0,1]:
                if t==0: y[6]=F(-1,4)  # B p>=x
                else: y[3]=y[7]=F(-1,8)  # each n>=-x
            elif p['pair']==[1,2]:
                if t==1: y[2]=F(-1,4)
                else: y[2]=y[6]=F(-1,8)
            else: y[1]=F(-4)  # contradictory guard retained, not excluded
            cert={'lp_sha256':identity(lp),'inequality_dual':list(map(str,y)), 'equality_dual':[]}
            value=_evaluate(lp,cert,clock(deadline)); cert['claimed_lower_bound']=str(value)
            entries.append({'id':q['id'],'certificate':cert})
        proof['pairs'].append({'pair':p['pair'],'candidates':{'batch_sha256':identity(batch),'entries':entries}})
    return proof


class HZEndpointTests(unittest.TestCase):
    observations={}

    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.old_threads)

    def setUp(self): self.end=time.monotonic()+300

    def prepare(self,case=None,mode='shared_input'):
        pairs,props,e,c=case or relation()
        return api.prepare_request(pairs,props,experts=e,classes=c,context=CONTEXT,deadline=self.end,relation_mode=mode)

    def run_case(self,name,case=None,mode='shared_input'):
        pairs,props,e,c=case or relation()
        package=api.support_endpoints(pairs,props,experts=e,classes=c,context=CONTEXT,deadline=self.end,relation_mode=mode)
        self.observations[name]=package; return package

    def check(self,request,proof,anchor=None):
        return receive.check_request(request,proof,expected_request_sha256=anchor or identity(request),deadline=self.end)

    def test_multi_pair_reference(self):
        p=self.run_case('separation_projected',separation()); request=p['request']
        manual=manual_proof(request,self.end); checked=self.check(request,manual)
        self.observations['separation_analytic']={'request':request,'proof':manual,'checked':checked}
        self.assertEqual(checked['positive'],3); self.assertEqual(checked['checked_endpoints'],6)
        self.assertEqual(min(F(r['lower_bound']) for r in checked['results']),F(3,32))
        self.assertFalse(checked['source_complete']); self.assertFalse(checked['deployed_float_SAFE'])
        # Projected outcomes are measured, never tuned/required positive here.
        self.assertEqual(p['checked']['checked_endpoints'],6)
        self.assertEqual(len(request['pairs']),3)
        cfg=json.loads((ROOT/'configs/hz_endpoint_controls_20261001.json').read_text())['separation_fixture']
        witness=[]
        for point in map(F,cfg['route_witnesses']):
            scores=[F(a)*point+F(b) for a,b in cfg['scores']]
            legal=[list(pair) for pair in combinations(range(3),2)
                   if all(scores[i]>=scores[j] for i in pair for j in range(3) if j not in pair)]
            witness.append({'x':str(point),'legal_pairs':legal})
        self.assertEqual([w['legal_pairs'] for w in witness],[[[1,2]],[[0,1],[1,2]],[[0,1]]])
        self.observations['route_witnesses']=witness

    def test_range_only_mc_separation(self):
        request=self.prepare(separation()); batch=request['pairs'][0]['batch']; src=batch['source']
        parsed,nc,nb=receive._source(src,lambda:None)
        def form(row):
            c=[F(0)]*(nc+nb)
            for k,shift in [('Gc',0),('Gb',nc)]:
                for j,v in parsed[k][row].items(): c[j+shift]=v
            return {'c':list(map(str,c)),'offset':str(parsed['c'][row])}
        base=copy.deepcopy(batch['base']); base.update(c=['0']*(nc+nb),offset='0')
        duty={'base':base,'variables':[f'f{i}' for i in range(nc+nb)],'a':form(0),'b':form(2),'gate':['0','1/2']}
        lp=mccormick(duty); point=['0','-1','-1','-1','-1','1/4','-1/8']
        value=exact_point(lp,point); self.assertEqual(value,F(-1,32))
        # Exact difference box on the SAME given HZ, not a separately tightened arm.
        delta=[F(a)-F(b) for a,b in zip(duty['a']['c'],duty['b']['c'])]
        self.assertEqual(sum(map(abs,delta)),F(3,2))
        self.observations['mc_negative_point']={'request':request,'duty':duty,'lp':lp,'point':point,'value':str(value)}

    def test_shared_vs_independent(self):
        a=self.run_case('shared_relation'); b=self.run_case('independent_relation',mode='independent_inputs')
        self.assertEqual(F(a['checked']['results'][0]['lower_bound']),F(1,4))
        self.assertEqual(F(b['checked']['results'][0]['lower_bound']),F(-3,4))
        self.assertEqual(a['checked']['positive'],1); self.assertEqual(b['checked']['positive'],0)

    def test_private_factors(self):
        entry=hz([0],[[1]],[[1]],ac=[[1]],ab=[[1]],b=[0])
        expert=hz([.25,0],[[1,1],[0,0]],[[1,1],[0,0]],ac=[[1,0]],ab=[[1,0]],b=[0])
        case=([{'pair':[0,1],'entry':entry,'a':expert,'b':copy.deepcopy(expert),'gate':['1/2','1/2']}],relation()[1],2,2)
        p=self.run_case('private_factors',case); source=p['request']['pairs'][0]['batch']['source']
        self.assertEqual(source['Gc']['shape'],[4,3]); self.assertEqual(source['Gb']['shape'],[4,3])
        self.assertEqual(source['Gb']['indices'],[0,1,0,2])
        self.assertEqual(p['request']['pairs'][0]['batch']['n_relaxed_binaries'],3)

    def test_multiclass_rational_projection(self):
        entry=hz([0],[[1]]); pairs=[]
        for a,b in combinations(range(3),2):
            ea=hz([.1,1,.25],[[.1],[0],[.25]]); eb=hz([.1,1,.25],[[-.1],[0],[-.25]])
            pairs.append({'pair':[a,b],'entry':entry,'a':ea,'b':eb,'gate':['1/3','2/3']})
        props=[{'id':'label1-vs0','q':['-1/3','1','0'],'offset':'1/7'},
               {'id':'label1-vs2','q':[0,1,-1],'offset':0}]
        p=self.run_case('multiclass',(pairs,props,3,3))
        self.assertEqual(p['checked']['required'],6); self.assertEqual(p['checked']['checked_endpoints'],12)
        q=p['request']['pairs'][0]['batch']['queries'][0]
        self.assertEqual(q['q'],['-1/9','1/3','0','-2/9','2/3','0'])
        self.assertEqual(F(q['constant']),F(8,7)-F.from_float(.1)/3)

    def test_equal_gate_one_query_two_labels(self):
        p=self.run_case('equal_gate')
        self.assertEqual(len(p['request']['pairs'][0]['batch']['queries']),1)
        self.assertEqual(p['checked']['results'][0]['covered_gate_end_labels'],[0,1])

    def test_unit_gate_expertwise(self):
        p=self.run_case('unit_gate',relation(('0','1')))
        self.assertEqual([F(v) for v in p['checked']['results'][0]['bounds']],[F(-3,4)]*2)
        self.assertEqual([q['q'] for q in p['request']['pairs'][0]['batch']['queries']],
                         [['0','0','1','-1'],['1','-1','0','0']])

    def test_orientation(self):
        case=relation(('1/4','1/2')); a=self.run_case('orientation_a',case)
        pair=case[0][0]; pair['a'],pair['b']=pair['b'],pair['a']; pair['gate']=['1/2','3/4']
        b=self.run_case('orientation_b',case)
        self.assertEqual(a['checked']['results'][0]['lower_bound'],b['checked']['results'][0]['lower_bound'])
        with self.assertRaises(ValueError): self.check(b['request'],b['proof'],identity(a['request']))

    def test_nonpositive_not_unsafe(self):
        p=self.run_case('nonpositive',relation(offset=-1))
        self.assertEqual(p['checked']['status'],'UNKNOWN_NONPOSITIVE')
        self.assertEqual(F(p['checked']['results'][0]['lower_bound']),F(-3,4))

    def test_missing_pair_property_endpoint(self):
        r=self.prepare(separation()); proof=manual_proof(r,self.end)
        for mutate in (lambda p:p['pairs'].pop(),lambda p:p['pairs'][0]['candidates']['entries'].pop(),
                       lambda p:p['pairs'][0]['candidates']['entries'].append(p['pairs'][0]['candidates']['entries'][0])):
            p=copy.deepcopy(proof); mutate(p)
            with self.assertRaises(ValueError): self.check(r,p)
        for mutate in (lambda x:x['pairs'].pop(),lambda x:x['properties'].append(copy.deepcopy(x['properties'][0])),
                       lambda x:x['pairs'][0]['batch']['queries'].pop()):
            wrong=copy.deepcopy(r); mutate(wrong)
            with self.assertRaises(ValueError): receive.validate_request(wrong,expected_request_sha256=identity(wrong),deadline=self.end)

    def test_wrong_source_guard_gate_or_request(self):
        r=self.prepare(); p=api.propose_request(r,expected_request_sha256=identity(r),deadline=self.end)
        for mutate in (lambda x:x['context'].__setitem__('request','other'),
                       lambda x:x['pairs'][0]['gate'].__setitem__('domain','other'),
                       lambda x:x['pairs'][0]['gate'].__setitem__('bounds',['1/4','1/2']),
                       lambda x:x['pairs'][0]['sources']['a']['c'].__setitem__(0,.5)):
            wrong=copy.deepcopy(r); mutate(wrong)
            with self.assertRaises(ValueError): self.check(wrong,p,identity(r))
        for low,high in [('2','1'),('0','2'),('1/2','0')]:
            with self.assertRaises(ValueError): self.prepare(relation((low,high)))

    def test_bad_shared_prefix_frame(self):
        for kind in ('frame','private_prefix','shared_rhs'):
            case=separation(); a=case[0][0]['a']
            if kind=='frame': a.frame_id+=1
            elif kind=='private_prefix': a.Auc[0,1]=1
            else: a.ub[0]+=1
            with self.assertRaises(ValueError): self.prepare(case)

    def test_mutated_private_column_or_constraints(self):
        r=self.prepare(separation())
        for key in ('Gc','Auc'):
            wrong=copy.deepcopy(r); src=wrong['pairs'][0]['batch']['source']
            src[key]['data'][-1]+=1
            with self.assertRaises(ValueError): receive.validate_request(wrong,expected_request_sha256=identity(wrong),deadline=self.end)
        wrong=copy.deepcopy(r); wrong['pairs'][0]['relation_mode']='independent_inputs'
        with self.assertRaises(ValueError): receive.validate_request(wrong,expected_request_sha256=identity(wrong),deadline=self.end)

    def test_rounded_objective_rejected(self):
        r=self.prepare(relation(('1/3','2/3')))
        for field in ('q','c'):
            wrong=copy.deepcopy(r); q=wrong['pairs'][0]['batch']['queries'][0]
            q[field][0]=str(F.from_float(float(F(q[field][0]))))
            with self.assertRaises(ValueError): receive.validate_request(wrong,expected_request_sha256=identity(wrong),deadline=self.end)
        wrong=copy.deepcopy(r); wrong['pairs'][0]['batch']['queries'][0]['constant']='1'
        with self.assertRaises(ValueError): receive.validate_request(wrong,expected_request_sha256=identity(wrong),deadline=self.end)

    def test_candidate_sign_claim_nonfinite(self):
        r=self.prepare(separation()); proof=manual_proof(r,self.end)
        for field,value in [('inequality_dual',[1]*10),('claimed_lower_bound','100'),('equality_dual',[math.inf])]:
            p=copy.deepcopy(proof); p['pairs'][0]['candidates']['entries'][0]['certificate'][field]=value
            with self.assertRaises(ValueError): self.check(r,p)

    def test_deadline_and_partial(self):
        r=self.prepare(separation()); proof=manual_proof(r,self.end); proof['pairs'][1]['candidates']=None
        checked=self.check(r,proof); self.assertEqual(checked['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(checked['missing_endpoints'],2)
        self.observations['partial']={'request':r,'proof':proof,'checked':checked}
        with self.assertRaises(TimeoutError):
            receive.check_request(r,proof,expected_request_sha256=identity(r),deadline=time.monotonic()-1)

    def test_candidate_exception_no_cuda(self):
        r=self.prepare()
        with patch.object(api,'propose_batch',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError): api.propose_request(r,expected_request_sha256=identity(r),deadline=self.end)
        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden')):
            with self.assertRaises(ValueError): api.propose_request(r,expected_request_sha256=identity(r),deadline=self.end,device='cuda:0')

    def test_independent_checker(self):
        r=self.prepare(); p=api.propose_request(r,expected_request_sha256=identity(r),deadline=self.end)
        with (patch.object(api,'shared_input_pair_hz',side_effect=AssertionError('builder forbidden')),
              patch.object(api,'prepare_batch',side_effect=AssertionError('exporter forbidden'))):
            self.assertEqual(self.check(r,p)['positive'],1)
        tree=ast.parse(Path(receive.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(m and any(x in m for x in ('hz_endpoints','weighted_top2','batched_support.py')) for m in imports))
        from scripts.run_hz_endpoint_controls import check_control_links, check_frozen_fixtures, check_cost, check_reference_results
        same={'request':r}
        independent={'request':self.prepare(mode='independent_inputs')}
        obs={k:copy.deepcopy(same) for k in ('separation_analytic','separation_projected','mc_negative_point','shared_relation')}
        obs['independent_relation']=independent; check_control_links(obs)
        other=self.prepare(relation(('1/4','1/2')))
        wrong=copy.deepcopy(obs); wrong['mc_negative_point']['request']=other
        with self.assertRaises(ValueError): check_control_links(wrong)
        wrong=copy.deepcopy(obs); wrong['independent_relation']['request']['pairs'][0]['sources']['a']['c'][0]=.5
        with self.assertRaises(ValueError): check_control_links(wrong)
        fixtures={'separation_analytic':{'request':self.prepare(separation())},'shared_relation':same}
        check_frozen_fixtures(fixtures)
        damaged=copy.deepcopy(fixtures); damaged['separation_analytic']['request']['pairs'][0]['sources']['a']['c'][0]+=.25
        with self.assertRaises(ValueError): check_frozen_fixtures(damaged)
        with self.assertRaises(ValueError): check_cost('shared_relation',{})
        with self.assertRaises(ValueError): check_cost('shared_relation',{'cost_seconds':{'total':1}})
        references={}
        for name,status,required,positive,count,bound in [
            ('separation_analytic','CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE',3,3,6,'3/32'),
            ('shared_relation','CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE',1,1,1,'1/4'),
            ('independent_relation','UNKNOWN_NONPOSITIVE',1,0,1,'-3/4')]:
            references[name]={'status':status,'required':required,'positive':positive,
                              'checked_endpoints':count,'minimum_bound':bound,'missing_endpoints':0}
        references['partial']={'status':'UNKNOWN_MISSING_EVIDENCE','missing_endpoints':2,'checked_endpoints':4}
        check_reference_results(references)
        damaged=copy.deepcopy(references); damaged['separation_analytic'].update(status='UNKNOWN_MISSING_EVIDENCE',positive=2,checked_endpoints=4,missing_endpoints=2)
        with self.assertRaises(ValueError): check_reference_results(damaged)
