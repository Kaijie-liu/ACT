"""Finite V2 controls: no dense source, endpoint optimizer, solver or CUDA."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(1)

from scoped_source.affine_references import Builder,build
from scoped_source.check_affine_references import CheckedAffineForest,check
from scoped_source.affine_reference_controls import deep,expanded,exact_forward,exact_adjoint
from scoped_source.source_dag_controls import mixed
from scoped_source.hz_request_store import Work,compact,digest
from act.back_end.moe.affine_reference_operator import Operator


def work(): return Work(time.monotonic()+300)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir='/data1/Kane/MOE'); cls.root=Path(cls.tmp.name); cls.cases={}
        for name,make in [('mixed',mixed),('deep',deep)]:
            root,source=make(cls.root/name); w=work(); raw,stats=build(root,source,work=w)
            f=CheckedAffineForest(raw,root,source,expected=digest(raw),work=w)
            cls.cases[name]=dict(root=root,source=source,raw=raw,doc=json.loads(raw),f=f,work=w)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def verify(self,doc=None,case='deep',w=None):
        v=self.cases[case]; raw=v['raw'] if doc is None else compact(doc)
        return check(raw,v['root'],v['source'],expected=digest(raw),work=w or work())

    def test_full_source_roster_and_private_factors(self):
        for name,v in self.cases.items():
            self.assertEqual(self.verify(case=name)['full_request_certificates'],0)
            self.assertEqual([p[:2] for p in v['doc']['roles']['pairs']],[[0,1],[0,2],[1,2]])
            self.assertEqual({t[2] for t in v['f'].factors},{-1,0,1,2})
            self.assertTrue(v['f'].complete)

    def test_same_v2_exact_forward_adjoint_oracle(self):
        for v in self.cases.values():
            f=v['f']; w=work(); matrix=expanded(f.nodes,w)
            x=[[F((j%7)-3,16),F(j%3,8)] for j in range(len(f.factors))]
            a=[[F((i%5)-2,32),F(i%4,16)] for i in range(len(f.nodes))]
            expected=[[sum(q*x[j][k] for j,q in row.items()) for k in range(2)] for row in matrix]
            trans=[[sum(row.get(j,F(0))*a[i][k] for i,row in enumerate(matrix)) for k in range(2)] for j in range(len(x))]
            self.assertEqual(exact_forward(f.nodes,x,w),expected)
            self.assertEqual(exact_adjoint(f.nodes,a,len(x),w),trans)
            with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA')):
                n=Operator(f,w)
                val=lambda q:torch.tensor([[float(z) for z in row] for row in q],dtype=torch.float64,device='cpu')
                torch.testing.assert_close(n.forward(val(x)),val(expected),rtol=1e-12,atol=1e-12)
                torch.testing.assert_close(n.adjoint(val(a)),val(trans),rtol=1e-12,atol=1e-12)
            self.assertEqual(n.stats['expanded_entries'],0)
        f=self.cases['deep']['f']
        self.assertGreaterEqual(Operator(f,work()).stats['levels'],2)
        self.assertTrue(any(f.nodes[p][2] for _,_,refs in f.nodes for p,s in refs))

    def test_later_affine_never_expands_parent(self):
        from scoped_source.check_affine_references import Checker
        v=self.cases['deep']; original=Builder.expand
        def only_input(self,n):
            if self.nodes[n][0]!=0: raise AssertionError('later parent expansion')
            return original(self,n)
        with patch.object(Builder,'expand',only_input):
            raw,_=build(v['root'],v['source'],work=work())
        self.assertEqual(raw,v['raw'])
        with patch.object(Checker,'expand',side_effect=AssertionError('checker expansion')):
            self.verify()

    def test_same_node_different_centers_and_center_compensation(self):
        w=work(); b=Builder(None,w)
        b.factor('input/c/0','continuous',-1,0); n=b.node(0,{0:F(1)})
        b.view('input',-1,None,'input',0,[(F(0),n)])
        p=b.view('parent',0,0,'alias',1,[(F(1,10),n),(F(1,5),n)])
        layer=SimpleNamespace(graph='expert0',index=2,outputs=1)
        row=SimpleNamespace(graph='expert0',layer=2,row=0,bias=F(0),coefficients=(F(1),F(-1)))
        v=b.affine(p,layer,iter([row])); c,r=b.views[v]['output'][0]
        self.assertEqual(b.nodes[r][2],()); self.assertNotEqual(c,F(0))
        self.assertLessEqual(abs(c-F(-1,10)),b.norms[r])
        self.assertTrue(any('/error/' in t[0] for t in b.factors))

    def test_diamond_triangle_not_exact_norm(self):
        b=Builder(None,work()); b.factor('a','continuous',-1,0)
        a=b.node(0,{0:F(1)}); p=b.node(0,{},((a,F(1)),)); q=b.node(0,{},((a,F(-1)),))
        r=b.node(0,{},((p,F(1)),(q,F(1))))
        self.assertEqual(b.norms[r],2); self.assertEqual(expanded(b.nodes,work())[r],{})
        self.assertEqual(exact_adjoint(b.nodes,[[F(0)],[F(0)],[F(0)],[F(1)]],1,work()),[[F(0)]])

    def test_all_center_errors_precede_lifts_and_zero_weight(self):
        from scoped_source.check_affine_references import Checker
        w=work(); b=Builder(None,w); b.factor('input/c/0','continuous',-1,0)
        n=b.node(0,{0:F(1)}); b.view('input',-1,None,'input',0,[(F(0),n)])
        p=b.view('parent',0,0,'alias',1,[(F(1),n),(F(1,2**54),n),(F(100),n)])
        layer=SimpleNamespace(graph='expert0',index=2,outputs=2)
        rows=[SimpleNamespace(graph='expert0',layer=2,row=k,bias=F(1,2**110),
                              coefficients=(F(1),F(1),F(0))) for k in range(2)]
        v=b.affine(p,layer,iter(rows)); names=[f[0] for f in b.factors]
        self.assertEqual(names[1:],['expert0/2/error/0','expert0/2/error/1',
                                   'expert0/2/lift/output/0','expert0/2/lift/output/1'])
        for c,r in b.views[v]['output']:
            self.assertEqual(c,1); self.assertEqual(b.nodes[r][2],((n,F(2)),))
            self.assertGreaterEqual(sum(q for _,q in b.nodes[r][1]),F(1,2**54)+F(1,2**110))
        # Isolated checker method control, not a fabricated public checked handle.
        def component(factors):
            ck=object.__new__(Checker); ck.work=work(); ck.views=copy.deepcopy(b.views)
            ck.nodes=copy.deepcopy(b.nodes); ck.factors=tuple(map(tuple,factors))
            ck.local_norms=[sum(abs(q) for j,q in local) for o,local,refs in b.nodes]
            ck.norms=[F(1)]+[None]*(len(b.nodes)-1); ck.visible=[frozenset({0}),frozenset({0})]
            ck.depths=[1,2]; ck.fid=1; ck.vid=2; ck.nid=1
            ck.affine(p,layer,iter(rows))
        component(b.factors)
        wrong=copy.deepcopy(b.factors); wrong[2],wrong[3]=wrong[3],wrong[2]
        with self.assertRaises(ValueError): component(wrong)

    def test_triangle_bounds_expanded_norm_and_compensation_present(self):
        f=self.cases['deep']['f']; rows=expanded(f.nodes,work())
        self.assertTrue(all(sum(map(abs,row.values()))<=bound for row,bound in zip(rows,f.norms)))
        self.assertTrue(any('/error/' in x[0] for x in f.factors))
        later={i for i,v in enumerate(f.views) if v['op']=='Linear' and
               any(f.nodes[n][2] for _,n in v['output'])}
        self.assertTrue(any('/error/' in x[0] for x in f.factors if x[3] in later))

    def test_source_identity_and_schema(self):
        for key,value in [('source','0'*64),('request','0'*64),('frame',2),('schema','SOURCE_COEFFICIENT_FOREST_V1')]:
            d=copy.deepcopy(self.cases['deep']['doc']); d[key]=value
            with self.assertRaises(ValueError): self.verify(d)
        v=self.cases['deep']
        with self.assertRaises(ValueError): check(v['raw'],v['root'],v['source'],expected='0'*64,work=work())

    def test_affine_ref_sign_order_missing_and_wrong_expert(self):
        base=self.cases['deep']['doc']; vi=next(i for i,v in enumerate(base['views']) if v['name']=='expert1/2')
        n=base['views'][vi]['output'][0][1]
        foreign=base['views'][base['roles']['experts'][0]]['output'][0][1]
        for mutate in (lambda d:d['nodes'][n][2].pop(),lambda d:d['nodes'][n][2].reverse(),
                       lambda d:d['nodes'][n][2][0].__setitem__(1,'99'),
                       lambda d:d['nodes'][n][2][0].__setitem__(0,foreign),
                       lambda d:d['nodes'][n][2][0].__setitem__(0,n)):
            d=copy.deepcopy(base); mutate(d)
            with self.assertRaises(ValueError): self.verify(d)

    def test_guard_rhs_wrong_router_and_missing_pair(self):
        base=self.cases['deep']['doc']; vi=base['roles']['pairs'][0][2]; n=base['views'][vi]['ine'][0][1]
        expert=base['views'][base['roles']['experts'][0]]['output'][0][1]
        for mutate in (lambda d:d['roles']['pairs'].pop(),lambda d:d['views'][vi]['ine'].pop(),
                       lambda d:d['views'][vi]['ine'][0].__setitem__(0,'99'),
                       lambda d:d['nodes'][n][2][0].__setitem__(0,expert),
                       lambda d:d['nodes'][n][2][0].__setitem__(1,'99')):
            d=copy.deepcopy(base); mutate(d)
            with self.assertRaises(ValueError): self.verify(d)

    def test_extra_or_missing_nodes_factors_and_outputs(self):
        for mutate in (lambda d:d['nodes'].pop(),lambda d:d['nodes'].append(copy.deepcopy(d['nodes'][-1])),
                       lambda d:d['factors'].pop(),lambda d:d['views'][2]['output'].pop()):
            d=copy.deepcopy(self.cases['deep']['doc']); mutate(d)
            with self.assertRaises((ValueError,IndexError)): self.verify(d)

    def test_changed_source_and_complete_suffix(self):
        v=self.cases['deep']; path=sorted((v['root']/'t').glob('*.bin'))[-1]; original=path.read_bytes()
        try:
            path.write_bytes(bytes([original[0]^1])+original[1:])
            with self.assertRaises(ValueError): self.verify()
        finally: path.write_bytes(original)

    def test_prefix_still_binds_unlowered_expert_suffix(self):
        v=self.cases['deep']; cut=['router',2]; w=work()
        raw,_=build(v['root'],v['source'],work=w,cut=cut)
        manifest=json.loads((v['root']/'manifest.json').read_bytes())
        record=next(t for t in manifest['tensors'] if t['name']=='experts.2.5.weight')
        path=v['root']/record['chunks'][0]['file']; original=path.read_bytes()
        try:
            path.write_bytes(bytes([original[0]^1])+original[1:])
            with self.assertRaises(ValueError):
                check(raw,v['root'],v['source'],expected=digest(raw),work=w,cut=cut)
        finally: path.write_bytes(original)

    def test_immutable_checked_handle_and_prefix_rejection(self):
        f=self.cases['deep']['f']
        for k in ('nodes','_sealed','complete'):
            with self.assertRaises(AttributeError): setattr(f,k,None)
            with self.assertRaises(AttributeError): delattr(f,k)
        with self.assertRaises(TypeError): f.views[0]['parent']=1
        with self.assertRaises(ValueError): Operator(object.__new__(CheckedAffineForest),work())
        v=self.cases['deep']; w=work(); cut=['router',2]; raw,_=build(v['root'],v['source'],work=w,cut=cut)
        f=CheckedAffineForest(raw,v['root'],v['source'],expected=digest(raw),work=w,cut=cut)
        with self.assertRaises(ValueError): Operator(f,w)

    def test_deadline_and_work_capacity(self):
        w=work(); w.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): self.verify(w=w)
        w=work(); w.counts['arithmetic']=8000000
        with self.assertRaisesRegex(ValueError,'work capacity'): self.verify(w=w)
        w=work(); n=Operator(self.cases['deep']['f'],w); w.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError): n.forward(torch.zeros((n.shape[1],1),dtype=torch.float64))

    def test_checker_independent_and_operator_input_refusal(self):
        with patch.object(Builder,'finish',side_effect=AssertionError('producer')),patch.object(Operator,'__init__',side_effect=AssertionError('native')):
            self.verify()
        n=Operator(self.cases['deep']['f'],work())
        for x in (torch.zeros((n.shape[1],9),dtype=torch.float64),torch.zeros((n.shape[1],1),dtype=torch.float32),
                  torch.full((n.shape[1],1),float('nan'),dtype=torch.float64)):
            with self.assertRaises(ValueError): n.forward(x)


if __name__=='__main__': unittest.main()
