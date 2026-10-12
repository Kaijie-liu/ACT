"""Finite scaled-source and batched exact-consumption correctness controls."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(1)

from scoped_source.hz_request_store import Work,compact,digest,identity
from scoped_source.scaled_references import Builder,build
from scoped_source.check_scaled_references import Checker,CheckedScaledForest,check
from scoped_source.scaled_endpoints import Session,Receipt,contract,pullback,weighted_norm,check_proof
from scoped_source.scaled_endpoint_controls import fixture,CASES,expanded,scalar_bound,differential
from act.back_end.moe.scaled_forest_endpoints import Operator,produce


def work(): return Work(time.monotonic()+300)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir='/data1/Kane/MOE'); cls.root=Path(cls.tmp.name); cls.cases={}
        for name in CASES:
            root,source=fixture(cls.root/name,name); w=work(); raw,_=build(root,source,work=w)
            f=CheckedScaledForest(raw,root,source,expected=digest(raw),work=w)
            c=contract(f)
            with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden')):
                proof,result,stats=produce(f,c,work=w)
            cls.cases[name]=dict(root=root,source=source,raw=raw,doc=json.loads(raw),f=f,c=c,
                                 proof=proof,result=result,stats=stats,w=w)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def fresh(self,name='deep'):
        v=self.cases[name]; w=work(); f=CheckedScaledForest(v['raw'],v['root'],v['source'],expected=digest(v['raw']),work=w)
        return f,w

    def verify(self,d,name='deep'):
        v=self.cases[name]; raw=compact(d)
        return check(raw,v['root'],v['source'],expected=digest(raw),work=work())

    def session(self,name='deep',width=8):
        f,w=self.fresh(name); c=contract(f); return Session(f,c,expected=identity(c),work=w,width=width),f,w

    def replay(self,v,proof):
        f,w=self.fresh(v); c=contract(f)
        return check_proof(f,c,proof,expected_contract=identity(c),expected_proof=identity(proof),work=w)

    def test_source_roster_clipped_and_zero_scales(self):
        for name,v in self.cases.items():
            self.assertEqual(v['f'].receipt['full_request_certificates'],0)
            self.assertEqual(len(v['f'].roles['pairs']),3)
            self.assertTrue(all(d==1 for d,f in zip(v['f'].scales,v['f'].factors) if f[3]!=0))
        for name,centers,scales in [('clipped',(F(13,16),F(-3,4)),(F(3,16),F(1,4))),
                                    ('zero',(F(7,8),F(-3,4)),(F(0),F(0)))]:
            f=self.cases[name]['f']; inputs=f.views[0]['output']
            self.assertEqual(tuple(c for c,n in inputs),centers); self.assertEqual(f.scales[:2],scales)
            self.assertEqual([f.nodes[n][1] for c,n in inputs],[((0,F(1)),),((1,F(1)),)])
        f=self.cases['zero']['f']; first=next(v for v in f.views if v['name']=='router/1')
        self.assertTrue(any(c for c,n in first['output']))

    def test_source_algebra_eager_operator_and_triangle_radius(self):
        for name,v in self.cases.items():
            f,w=v['f'],v['w']; rows=expanded(f,w)
            self.assertTrue(all(sum(map(abs,row.values()))<=r for row,r in zip(rows,f.norms)))
            x=[[F((j%7)-3,16),F(j%3,8)] for j in range(len(f.factors))]
            a=[[F((j%5)-2,32),F(j%4,16)] for j in range(len(f.nodes))]
            expected=[[sum(q*x[j][k] for j,q in r.items()) for k in range(2)] for r in rows]
            trans=[[sum(r.get(j,F(0))*a[i][k] for i,r in enumerate(rows)) for k in range(2)] for j in range(len(x))]
            val=lambda q:torch.tensor([[float(z) for z in r] for r in q],dtype=torch.float64)
            n=Operator(f,w)
            torch.testing.assert_close(n.forward(val(x)),val(expected),rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(n.adjoint(val(a)),val(trans),rtol=1e-12,atol=1e-12)
            actual=pullback(f,[{i:r[k] for i,r in enumerate(a)} for k in range(2)],w)
            self.assertEqual([[f.scales[j]*actual[k].get(j,F(0)) for k in range(2)] for j in range(len(x))],trans)
            self.assertEqual(n.stats['expanded_constraint_entries'],0)
        self.assertGreaterEqual(Operator(self.cases['deep']['f'],self.cases['deep']['w']).stats['levels'],2)

    def test_all_candidates_eager_differential(self):
        for name,v in self.cases.items():
            result=differential(v['f'],v['c'],v['proof'],v['w'])
            self.assertEqual(result['candidates'],2*v['result']['required_endpoints'])
            self.assertEqual(result['result'],v['result'])

    def test_multicolumn_scalar_differential_and_cancellation(self):
        f,w=self.fresh(); a={len(f.nodes)-1:F(2),len(f.nodes)-2:F(-2)}; b={n:-q for n,q in a.items()}
        batch=pullback(f,[a,b],w); single=[pullback(f,[v],w)[0] for v in (a,b)]
        self.assertEqual(batch,single)
        merged={j:batch[0].get(j,F(0))+batch[1].get(j,F(0)) for j in range(len(f.factors))}
        self.assertEqual(weighted_norm(f,merged,w),0)
        self.assertGreater(weighted_norm(f,batch[0],w)+weighted_norm(f,batch[1],w),0)
        # Combining before propagation can cancel sooner; splitting is not universally cheaper.
        before=sum(w.counts.values()); pullback(f,[{}],w); combined_cost=sum(w.counts.values())-before
        before=sum(w.counts.values()); pullback(f,[a,b],w); split_cost=sum(w.counts.values())-before
        self.assertGreater(split_cost,combined_cost)

    def test_equal_scales_do_not_merge_distinct_factors(self):
        f,w=self.fresh('deep'); self.assertEqual(f.scales[0],f.scales[1])
        self.assertEqual(weighted_norm(f,{0:F(3),1:F(-3)},w),6*f.scales[0])
        self.assertEqual(weighted_norm(f,{0:F(0)},w),0)

    def test_full_ties_multiple_classes_and_endpoints(self):
        v=self.cases['guarded']; self.assertEqual(v['result']['required'],12)
        self.assertEqual(v['result']['required_endpoints'],24)
        v=self.cases['half']; self.assertEqual(v['result']['required_endpoints'],5)
        for name,v in self.cases.items(): self.assertEqual(self.replay(name,v['proof']),v['result'])

    def test_positive_nonpositive_and_missing_terminal(self):
        self.assertTrue(any(v['result']['status']=='CHECKED_POSITIVE_CAPTURED_REAL_SOURCE' for v in self.cases.values()))
        # With zero proposals a box-only check cannot certify this frozen guarded source.
        p=copy.deepcopy(self.cases['guarded']['proof'])
        for tile in p['tiles']:
            for r in tile: r['proposed']=copy.deepcopy(r['zero'])
        self.assertEqual(self.replay('guarded',p)['status'],'UNKNOWN_NONPOSITIVE')
        p=copy.deepcopy(self.cases['guarded']['proof']); p['tiles'][0][0]['zero']=p['tiles'][0][0]['proposed']=None
        result=self.replay('guarded',p); self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(result['missing_endpoints'],1)

    def test_zero_candidate_is_verified_not_a_label(self):
        p=copy.deepcopy(self.cases['guarded']['proof']); p['tiles'][0][0]['zero']['inequality'][0]='-1'
        with self.assertRaisesRegex(ValueError,'explicit zero'): self.replay('guarded',p)

    def test_raw_bounds_not_trusted_and_exact_tie_selects_proposal(self):
        s,f,w=self.session('half'); receipts=[]
        for i,tile in enumerate(self.cases['half']['proof']['tiles']):
            s.prepare(i); rows=copy.deepcopy(tile)
            for r in rows:
                r['proposed']=copy.deepcopy(r['zero']); r['proposed']['lower']='999999999'
                r['zero']['lower']='-999999999'
            rr=s.accept(rows); receipts.extend(rr)
            self.assertTrue(all(r.record['selected']=='proposed' for r in rr))
            self.assertTrue(all(r.record['lower']!='999999999' for r in rr))
        s.finish(receipts)

    def test_source_tampering_all_scale_center_factor_weight_bindings(self):
        base=self.cases['deep']['doc']; aux=next(i for i,f in enumerate(base['factors']) if f[3]!=0)
        n=base['views'][2]['output'][0][1]
        mutations=[lambda d:d.__setitem__('source','0'*64),lambda d:d.__setitem__('request','0'*64),
            lambda d:d.__setitem__('schema','SOURCE_AFFINE_REFERENCE_FOREST_V2'),
            lambda d:d['scales'].__setitem__(0,'-1'),lambda d:d['scales'].__setitem__(0,'1/3'),
            lambda d:d['scales'].__setitem__(0,'2'),lambda d:d['scales'].__setitem__(aux,'2'),
            lambda d:d['scales'].pop(),lambda d:d['factors'][0].__setitem__(0,'other'),
            lambda d:d['views'][0]['output'][0].__setitem__(0,'1'),
            lambda d:d['nodes'][n][1][0].__setitem__(1,'2'),lambda d:d['roles']['experts'].reverse(),
            lambda d:d['roles']['pairs'].pop(),lambda d:d['nodes'].pop()]
        for change in mutations:
            d=copy.deepcopy(base); change(d)
            with self.assertRaises((ValueError,IndexError)): self.verify(d)

    def test_guard_rhs_refs_and_private_factor_tampering(self):
        base=self.cases['deep']['doc']; vi=base['roles']['pairs'][0][2]; n=base['views'][vi]['ine'][0][1]
        for change in [lambda d:d['views'][vi]['ine'][0].__setitem__(0,'999'),
                       lambda d:d['nodes'][n][2][0].__setitem__(1,'-99'),
                       lambda d:d['nodes'][n][2][0].__setitem__(0,0)]:
            d=copy.deepcopy(base); change(d)
            with self.assertRaises(ValueError): self.verify(d)
        d=copy.deepcopy(base); j=next(j for j,f in enumerate(d['factors']) if f[2]==1); d['factors'][j][2]=0
        with self.assertRaises(ValueError): self.verify(d)

    def test_complete_suffix_and_prefix_refusal(self):
        v=self.cases['deep']; cut=['router',2]; w=work(); raw,_=build(v['root'],v['source'],work=w,cut=cut)
        f=CheckedScaledForest(raw,v['root'],v['source'],expected=digest(raw),work=w,cut=cut)
        with self.assertRaises(ValueError): contract(f)
        with self.assertRaises(ValueError): Operator(f,w)
        path=sorted((v['root']/'t').glob('*.bin'))[-1]; old=path.read_bytes()
        try:
            path.write_bytes(bytes([old[0]^1])+old[1:])
            with self.assertRaises(ValueError): check(raw,v['root'],v['source'],expected=digest(raw),work=work(),cut=cut)
        finally: path.write_bytes(old)

    def test_default_checkers_reject_v3_and_producer_is_not_checker(self):
        from scoped_source.check_source_dag import check as v1
        from scoped_source.check_affine_references import check as v2
        v=self.cases['deep']
        for fn in (v1,v2):
            with self.assertRaises(ValueError): fn(v['raw'],v['root'],v['source'],expected=digest(v['raw']),work=work())
        with (patch.object(Builder,'finish',side_effect=AssertionError('producer forbidden')),
              patch.object(Operator,'__init__',side_effect=AssertionError('native forbidden'))):
            self.verify(v['doc'])

    def test_candidate_rejects_missing_extra_wrong_property_and_inequality(self):
        base=self.cases['guarded']['proof']
        mutations=[lambda p:p['tiles'].pop(),lambda p:p['tiles'][0].pop(),
            lambda p:p['tiles'].append(copy.deepcopy(p['tiles'][0])),
            lambda p:p['tiles'][0][0].__setitem__('slot','0'*64),
            lambda p:p['tiles'][0][0]['proposed'].__setitem__('query','0'*64),
            lambda p:p['tiles'][0][0]['proposed']['inequality'].pop(),
            lambda p:p['tiles'][0][0]['proposed']['inequality'].__setitem__(0,'1'),
            lambda p:p['tiles'][0][0]['proposed']['inequality'].__setitem__(0,'NaN'),
            lambda p:p['tiles'][0][0]['proposed'].__setitem__('extra','x'),
            lambda p:p['tiles'][0][0].__setitem__('zero',None),lambda p:p.__setitem__('width',0)]
        for change in mutations:
            p=copy.deepcopy(base); change(p)
            with self.assertRaises((ValueError,IndexError)): self.replay('guarded',p)
        f,w=self.fresh('guarded'); c=contract(f); c['margin']='1'
        with self.assertRaises(ValueError): Session(f,c,expected=identity(c),work=w)

    def test_checked_object_alias_and_cross_session_work(self):
        f,w=self.fresh()
        for k in ('nodes','scales','session','_sealed'):
            with self.assertRaises(AttributeError): setattr(f,k,None)
            with self.assertRaises(AttributeError): delattr(f,k)
        with self.assertRaises(TypeError): f.views[0]['output'][0]=None
        with self.assertRaises(ValueError): Operator(f,work())
        with self.assertRaises(ValueError): Operator(object.__new__(CheckedScaledForest),w)

    def test_checker_owned_receipts_no_alias_or_forgery(self):
        s,f,w=self.session('half'); receipts=[]
        for i,tile in enumerate(self.cases['half']['proof']['tiles']):
            tasks,roots=s.prepare(i)
            with self.assertRaises(TypeError): roots[0][1][0]=F(99)
            r=copy.deepcopy(tile); rr=s.accept(r); receipts.extend(rr)
            r[0]['proposed']['lower']='888'; self.assertNotEqual(rr[0].record['lower'],'888')
            with self.assertRaises(TypeError): rr[0].record['lower']='888'
            with self.assertRaises(Exception): rr[0].record={}
        fabricated=list(receipts); fabricated[0]=Receipt(receipts[0].record)
        with self.assertRaisesRegex(ValueError,'receipt identity'): s.finish(fabricated)
        with self.assertRaises(ValueError): s.finish(receipts+receipts[:1])
        other,g,z=self.session('half'); foreign=[]
        for i,tile in enumerate(self.cases['half']['proof']['tiles']): other.prepare(i); foreign.extend(other.accept(tile))
        with self.assertRaisesRegex(ValueError,'receipt identity'): s.finish(foreign)
        s.finish(receipts)
        with self.assertRaises(ValueError): s.finish(receipts)

    def test_raw_candidate_mutation_after_accept_does_not_change_proof(self):
        s,f,w=self.session('half'); rr=[]
        for i,tile in enumerate(self.cases['half']['proof']['tiles']):
            raw=copy.deepcopy(tile); s.prepare(i); rr.extend(s.accept(raw)); raw.clear()
        s.finish(rr); p=s.proof(); self.assertEqual(p,self.cases['half']['proof'])

    def test_cumulative_work_deadline_and_late_positive_terminal(self):
        f,w=self.fresh(); w.counts['arithmetic']=8000000
        with self.assertRaisesRegex(ValueError,'work capacity'): Session(f,contract(f),expected=identity(contract(f)),work=w)
        f,w=self.fresh(); w.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): Operator(f,w)
        s,f,w=self.session('half'); receipts=[]
        for i,tile in enumerate(self.cases['half']['proof']['tiles']): s.prepare(i); receipts.extend(s.accept(tile))
        w.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): s.finish(receipts)

    def test_partial_exception_and_midcheck_deadline(self):
        f,w=self.fresh('guarded'); events=[]
        def fail(i,records): events.append(copy.deepcopy(records)); raise RuntimeError('injected publish failure')
        with self.assertRaises(RuntimeError): produce(f,contract(f),work=w,event=fail)
        self.assertEqual(len(events),1)
        s,f,w=self.session('guarded'); s.prepare(0); original=w.exact; count=[0]
        def expire(q):
            count[0]+=1
            if count[0]==3: w.deadline=time.monotonic()-1
            return original(q)
        with patch.object(w,'exact',side_effect=expire),self.assertRaises(TimeoutError):
            s.accept(self.cases['guarded']['proof']['tiles'][0])
        with self.assertRaises(TimeoutError): s.finish([])

    def test_proposals_published_before_exact_failure(self):
        f,w=self.fresh('guarded'); events=[]
        with patch.object(Session,'accept',side_effect=TimeoutError('injected exact cutoff')):
            with self.assertRaises(TimeoutError):
                produce(f,contract(f),work=w,event=lambda i,r:events.append(r))
        self.assertEqual(len(events),1)
        self.assertIs(type(events[0]),bytes)
        self.assertTrue(all(set(r)=={'slot','proposed','zero'} for r in json.loads(events[0])))

    def test_native_input_refusal_and_sequential_tile_contract(self):
        f,w=self.fresh(); n=Operator(f,w)
        for x in (torch.zeros((n.shape[1],9),dtype=torch.float64),torch.zeros((n.shape[1],1),dtype=torch.float32),
                  torch.full((n.shape[1],1),float('inf'),dtype=torch.float64)):
            with self.assertRaises(ValueError): n.forward(x)
        s,f,w=self.session()
        with self.assertRaises(ValueError): s.prepare(1)
        with self.assertRaises(ValueError): s.accept([])
        s.prepare(0)
        with self.assertRaises(ValueError): s.prepare(0)
        with self.assertRaises(ValueError): s.finish([])


if __name__=='__main__': unittest.main()
