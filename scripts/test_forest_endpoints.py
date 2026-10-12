"""Source-reference complete consumer controls; no real model or CUDA."""
import copy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(1)

from scoped_source.hz_request_store import Work, compact, digest, identity
from scoped_source.source_dag import build
from scoped_source.forest_endpoints import (
    CheckedForest, contract, plan, pullback, evaluate_tile, check_proof,
)
from scoped_source.forest_endpoint_controls import fixture, eager, differential
from act.back_end.moe.forest_endpoints import Native, produce


def work(): return Work(time.monotonic()+300)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir='/data1/Kane/MOE'); cls.root=Path(cls.tmp.name); cls.cases={}
        for name in ('small40','mixed','guarded','half'):
            w=work(); root,source=fixture(cls.root/name,name); raw,_=build(root,source,work=w)
            f=CheckedForest(raw,root,source,expected=digest(raw),work=w)
            c=contract(f); p=plan(f,c,expected=identity(c),work=w)
            with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA')):
                proof,stats=produce(f,p,width=8,work=w)
            result=check_proof(f,c,proof,expected_contract=identity(c),expected_proof=identity(proof),work=w)
            s,old=eager(root,source,raw,cls.root/(name+'-old'),w)
            n=differential(s,f,p,proof,w)
            cls.cases[name]=dict(root=root,source=source,raw=raw,f=f,c=c,p=p,proof=proof,result=result,
                                 w=w,store=s,differential=n,stats=stats)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def verify(self,v,proof=None,c=None,w=None):
        p=v['proof'] if proof is None else proof; c=v['c'] if c is None else c
        return check_proof(v['f'],c,p,expected_contract=identity(c),expected_proof=identity(p),work=w or work())

    def test_all_sources_same_candidate_eager_differential(self):
        for v in self.cases.values():
            self.assertEqual(v['differential'],len(v['p'].tasks))
            self.assertLess(sum(v['w'].counts.values()),8000000)
            self.assertFalse(v['result']['native_float_proof'])

    def test_positive_continuity_controls(self):
        for name,ends,duties in (('half',5,3),('guarded',24,12)):
            r=self.cases[name]['result']
            self.assertEqual((r['required_endpoints'],r['required']),(ends,duties))
            self.assertEqual(r['status'],'CHECKED_POSITIVE_CAPTURED_REAL_SOURCE')

    def test_nonpositive_is_not_unsafe(self):
        v=self.cases['half']; p=copy.deepcopy(v['proof'])
        for tasks,records in zip(v['p'].tiles(8),p['tiles']):
            for r in records:
                c=r['candidate']; c['equality']=['0']*len(c['equality']); c['inequality']=['-1000']*len(c['inequality'])
            bounds,_=evaluate_tile(v['f'],tasks,[r['candidate'] for r in records],work())
            for r,b in zip(records,bounds): r['candidate']['lower']=str(b)
        self.assertEqual(self.verify(v,p)['status'],'UNKNOWN_NONPOSITIVE')

    def test_immutable_handle_and_alias(self):
        v=self.cases['mixed']; f=v['f']
        for attr in ('_sealed','nodes','complete'):
            with self.assertRaises(AttributeError): delattr(f,attr)
            with self.assertRaises(AttributeError): setattr(f,attr,False)
        with self.assertRaises(TypeError): f.views[0]['parent']=3
        with self.assertRaises(TypeError): f.declaration['model_state']['sha256']='0'*64
        with self.assertRaises(ValueError): contract(dict(f.receipt))
        with self.assertRaises(ValueError): contract(object.__new__(CheckedForest))

    def test_property_source_and_identity(self):
        v=self.cases['half']
        for key,value in [('label',1),('margin','1'),('classes',3),('request','0'*64),('top_k',1)]:
            c=copy.deepcopy(v['c']); c[key]=value
            with self.assertRaises(ValueError): self.verify(v,c=c)
        with self.assertRaises(ValueError): CheckedForest(v['raw'],v['root'],v['source'],expected='0'*64,work=work())

    def test_prefix_refused(self):
        v=self.cases['small40']; w=work(); cut=['expert0',2]
        raw,_=build(v['root'],v['source'],work=w,cut=cut)
        f=CheckedForest(raw,v['root'],v['source'],expected=digest(raw),work=w,cut=cut)
        with self.assertRaises(ValueError): contract(f)

    def test_pair_domain_only_and_private_identity(self):
        f=self.cases['mixed']['f']; ids=f.rows((0,1),'ine')
        guard={v for a,b,v in f.roles['pairs'] if (a,b)==(0,1)}
        others={v for a,b,v in f.roles['pairs'] if (a,b)!=(0,1)}
        self.assertTrue(guard<={i for i,_,_ in ids}); self.assertFalse(others&{i for i,_,_ in ids})
        self.assertTrue(all(f.views[i]['scope']!=2 for i,_,_ in ids))
        private=[{j for j,t in enumerate(f.factors) if t[2]==e} for e in range(3)]
        self.assertTrue(all(private)); self.assertFalse(private[0]&private[1])

    def test_same_node_rhs_separate_and_zero_fallback(self):
        v=self.cases['mixed']; f=v['f']; tasks=next(v['p'].tiles(8))
        cc=[dict(query=t.query_id,lower='0',equality=['1/8']*len(f.rows(t.pair,'eq')),
                 inequality=['-1/4']*len(f.rows(t.pair,'ine'))) for t in tasks]
        from scoped_source.check_store_endpoints import evaluate_tile as old
        n,nr=evaluate_tile(f,tasks,cc,work()); o,orr=old(v['store'],tasks,cc,work())
        self.assertEqual(n,o); self.assertEqual(nr,[{j:q for j,q in r.items() if q} for r in orr])

    def test_sparse_operator_forward_transpose(self):
        f=self.cases['mixed']['f']; n=Native(f,work())
        expanded=[]
        for i in range(len(f.nodes)):
            _,local,refs=f.nodes[i]; d=dict(local)
            for p,s in refs:
                self.assertFalse(f.nodes[p][2])
                for j,q in f.nodes[p][1]: d[j]=d.get(j,F(0))+s*q
            expanded.append([float(d.get(j,0)) for j in range(len(f.factors))])
        m=torch.tensor(expanded,dtype=torch.float64,device='cpu')
        x=torch.arange(len(f.factors),dtype=torch.float64,device='cpu')[:,None]/16
        a=torch.arange(len(f.nodes),dtype=torch.float64,device='cpu')[:,None]/32
        torch.testing.assert_close(n.forward(x),m@x,rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(n.adjoint(a),m.T@a,rtol=1e-14,atol=1e-14)

    def test_missing_deleted_duplicate_wrong_binding(self):
        v=self.cases['half']; p=copy.deepcopy(v['proof']); p['tiles'][0][0]['candidate']=None
        self.assertEqual(self.verify(v,p)['status'],'UNKNOWN_MISSING_EVIDENCE')
        for mutate in (lambda p:p['tiles'].pop(),lambda p:p['tiles'][0].pop(),
                       lambda p:p['tiles'][0][0].__setitem__('slot','0'*64),
                       lambda p:p['tiles'][0][0]['candidate'].__setitem__('query','0'*64),
                       lambda p:p['tiles'][0][0]['candidate'].__setitem__('lower','100')):
            p=copy.deepcopy(v['proof']); mutate(p)
            with self.assertRaises(ValueError): self.verify(v,p)

    def test_dual_sign_missing_and_nonfinite_native(self):
        v=self.cases['half']; p=copy.deepcopy(v['proof']); p['tiles'][0][0]['candidate']['inequality'][0]='1'
        with self.assertRaises(ValueError): self.verify(v,p)
        p=copy.deepcopy(v['proof']); p['tiles'][0][0]['candidate']['equality'].pop()
        with self.assertRaises(ValueError): self.verify(v,p)
        n=Native(v['f'],work())
        with self.assertRaises(ValueError): n.tensor([float('nan')])

    def test_deadline_and_cumulative_work(self):
        v=self.cases['half']; w=work(); w.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): self.verify(v,w=w)
        w=work(); w.counts['arithmetic']=8000000
        with self.assertRaisesRegex(ValueError,'work capacity'): self.verify(v,w=w)

    def test_checker_does_not_need_native_or_producer(self):
        with patch('scoped_source.source_dag.build',side_effect=AssertionError()),patch.object(Native,'__init__',side_effect=AssertionError()):
            self.verify(self.cases['half'])

    def test_meta_default_does_not_move_to_gpu(self):
        v=self.cases['half']
        with torch.device('meta'),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError()):
            p,_=produce(v['f'],v['p'],width=8,work=work())
        self.assertEqual(p,v['proof'])

    def test_fabricated_plan_or_values_never_issue_checked_terminal(self):
        from scoped_source.forest_endpoints import Plan, _aggregate
        with self.assertRaises(ValueError): _aggregate(Plan('x','x',(),(),()),{},work())
        p=self.cases['half']['p']; r=_aggregate(p,{t.slot:F(100) for t in p.tasks},work())
        self.assertEqual(r['status'],'AGGREGATE_POSITIVE_UNVERIFIED_INPUTS')

    def test_producer_oversize_refusal_before_event(self):
        v=self.cases['half']; event=[]
        with patch('act.back_end.moe.forest_endpoints.compact',return_value=b'x'*(4*2**20+1)):
            with self.assertRaisesRegex(ValueError,'producer proof bytes'):
                produce(v['f'],v['p'],width=8,work=work(),event=lambda *x:event.append(x))
        self.assertFalse(event)

    def test_decimal_margin_source_semantics(self):
        import json
        import shutil
        v=self.cases['half']; root=self.root/'decimal-margin'
        shutil.copytree(v['root'],root)
        m=json.loads((root/'manifest.json').read_bytes()); m['declaration']['request']['margin']='0.1'
        (root/'manifest.json').write_bytes(compact(m)); source=identity(m); w=work()
        raw,_=build(root,source,work=w); f=CheckedForest(raw,root,source,expected=digest(raw),work=w)
        c=contract(f); p=plan(f,c,expected=identity(c),work=w)
        self.assertEqual({t.offset for t in p.tasks},{'-1/10'})

    def test_exact_zero_fallback_not_weaker(self):
        for v in self.cases.values():
            f=v['f']
            for tasks,records in zip(v['p'].tiles(8),v['proof']['tiles']):
                cc=[dict(query=t.query_id,lower='0',equality=['0']*len(f.rows(t.pair,'eq')),
                         inequality=['0']*len(f.rows(t.pair,'ine'))) for t in tasks]
                lower,_=evaluate_tile(f,tasks,cc,work())
                self.assertTrue(all(F(r['candidate']['lower'])>=b for r,b in zip(records,lower)))

    def test_partial_prefix_on_mid_execution_deadline_and_exception(self):
        v=self.cases['guarded']
        for fault in ('deadline','exception'):
            w=work(); partial=[]
            def event(i,records):
                partial.append(copy.deepcopy(records))
                if fault=='deadline': w.deadline=time.monotonic()-1
                else: raise OSError('simulated publishing failure')
            with self.assertRaises(TimeoutError if fault=='deadline' else OSError):
                produce(v['f'],v['p'],width=8,work=w,event=event)
            self.assertEqual(len(partial),1)

    def test_portable_requires_external_binding_anchor(self):
        from scoped_source.verify_forest_endpoints import verify
        for h in (None,'',False,'x'*64):
            with self.assertRaisesRegex(ValueError,'mandatory identity'): verify(self.root/'absent',h)


if __name__=='__main__': unittest.main()
