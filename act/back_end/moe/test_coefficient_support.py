"""Actual small SparseHZono interface; no solve or model evaluation."""
import copy
from fractions import Fraction as F
import time
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.moe.coefficient_support import prepare, native_rows, check_native_rows, support
from scoped_source.coefficient_dag import Graph, query, evaluate
from scoped_source.coefficient_dag_controls import expanded, REQUEST
from scoped_source.hz_request_store import Work, compact, decode, digest, identity


def fixture():
    # Original [.75 + x + 2y], exact unstable-ReLU auxiliary equations,
    # plus an alias with a different center; [-1,1] binary relaxation.
    return SparseHZono(c=np.array([.75,1.75,15/8]),
        Gc=sp.csr_matrix([[1.,2.,0.,0.],[1.,2.,0.,0.],[0.,0.,0.,-15/8]]),Gb=sp.csr_matrix((3,1)),
        Ac=sp.csr_matrix([[-1.,-2.,-9/8,-15/8]]),Ab=sp.csr_matrix([[-9/8]]),b=np.array([-9/8]),
        Auc=sp.csr_matrix([[0.,0.,-1.,0.],[0.,0.,0.,-1.]]),Aub=sp.csr_matrix([[-1.],[1.]]),
        ub=np.zeros(2),frame_id=717)


FACTORS=[['x','continuous',-1],['y','continuous',-1],['u','continuous',0],
         ['v','continuous',0],['z','binary',0]]
LINKS={1:(0,F(1)),3:(0,F(-1))}


class NativeControls(unittest.TestCase):
    def setUp(self):self.work=Work(time.monotonic()+300)

    def prepare(self):return prepare(fixture(),request=REQUEST,factors=FACTORS,links=LINKS,work=self.work)

    def test_all_original_native_coefficients_and_support(self):
        hz=fixture();g,raw,receipt=self.prepare()
        self.assertEqual(receipt['rows'],6)
        self.assertEqual(g.nodes[1].local,())
        self.assertEqual(g.nodes[3].local,((2,F(-9,8)),(3,F(-15,8)),(4,F(-9,8))))
        q=query(g,['1','-1','2'],'1/8');d=dict(eq=['1/4'],ine=['-1/2','-1/4'])
        b=evaluate(g,q,d,expected_query=identity(q),work=self.work)
        # Independent native dense reading, not the intake's CSR iterator.
        original=[]
        for kind,vector,left,right in (('output',hz.c,hz.Gc,hz.Gb),('eq',hz.b,hz.Ac,hz.Ab),('ine',hz.ub,hz.Auc,hz.Aub)):
            matrix=np.concatenate((left.toarray(),right.toarray()),axis=1)
            original.extend((kind,str(F(float(rhs))),{j:F(float(v)) for j,v in enumerate(row) if v})
                            for rhs,row in zip(vector,matrix,strict=True))
        self.assertEqual((b.lower,b.constant,dict(b.residual)),expanded(original,q['weights'],d))

    def test_changed_scale_rhs_root_and_factor_rejected_by_native_bridge(self):
        g,raw,_=self.prepare()
        for field in ('scale','rhs','missing','factor'):
            doc=decode(raw)
            if field=='scale':doc['nodes'][3][3][0][1]='1'
            elif field=='rhs':doc['roots'][3][3]='9/8'
            elif field=='missing':doc['roots'].pop();doc['nodes'].pop()
            else:doc['factors'][4][2]=1;doc['domains'][0]['experts']=[0,1]
            newraw=compact(doc);new=Graph(newraw,expected_graph=digest(newraw),expected_request=REQUEST,
                                        expected_source=doc['source'],work=self.work)
            with self.assertRaises(ValueError):check_native_rows(new,fixture(),FACTORS,work=self.work)

    def test_native_mutation_and_kind_mapping(self):
        g,_,_=self.prepare();hz=fixture();hz.c[0]+=1
        with self.assertRaisesRegex(ValueError,'source changed'):check_native_rows(g,hz,FACTORS,work=self.work)
        bad=copy.deepcopy(FACTORS);bad[4][1]='continuous'
        with self.assertRaisesRegex(ValueError,'factor map'):
            prepare(fixture(),request=REQUEST,factors=bad,links=LINKS,work=self.work)

    def test_full_api_and_deadline(self):
        out=support(fixture(),request=REQUEST,factors=FACTORS,links=LINKS,weights=['1','-1','2'],offset='1/8',
            duals=dict(eq=['1/4'],ine=['-1/2','-1/4']),deadline=time.monotonic()+300)
        self.assertEqual(out['bridge']['guarantee'],'GIVEN_NATIVE_HZ_ROWS_ONLY')
        with self.assertRaises(TimeoutError):
            support(fixture(),request=REQUEST,factors=FACTORS,links=LINKS,weights=[],offset='0',duals={},
                    deadline=time.monotonic()-1)

    def test_corrupt_csr_cannot_alias_a_binary_factor(self):
        for cached in (True,False):
            for change in ('column','negative','pointer','dtype'):
                hz=fixture()
                if cached:hz.Gc.has_canonical_format=True
                else:hz.Gc.__dict__.pop('_has_canonical_format',None)
                if change=='column':hz.Gc.indices[1]=hz.n_cont
                elif change=='negative':hz.Gc.indices[0]=-1
                elif change=='pointer':hz.Gc.indptr[-1]+=1
                else:hz.Gc.indices=hz.Gc.indices.astype(float)
                with self.assertRaises(ValueError):
                    prepare(hz,request=REQUEST,factors=FACTORS,links=LINKS,work=self.work)

    def test_hint_bits_before_multiplication(self):
        with self.assertRaisesRegex(ValueError,'hint rational capacity'):
            prepare(fixture(),request=REQUEST,factors=FACTORS,links={1:(0,F(2**4097))},work=self.work)

    def test_mutation_and_restore_during_native_comparison(self):
        from act.back_end.moe import coefficient_support as api
        g,raw,_=self.prepare();doc=decode(raw);doc['nodes'][0][2][0][1]='2'
        raw=compact(doc);changed=Graph(raw,expected_graph=digest(raw),expected_request=REQUEST,
                                     expected_source=g.source,work=self.work)
        original=api.native_rows;passes=[]
        def observed(hz,work):
            passes.append(True);comparing=len(passes)==2
            for kind,name,rhs,entries in original(hz,work):
                if comparing and name in ('output/0','output/1','eq/0'):
                    entries=tuple((j,v*2 if j==0 else v) for j,v in entries)
                yield kind,name,rhs,entries
        with patch.object(api,'native_rows',observed):
            with self.assertRaisesRegex(ValueError,'compared-row source identity'):
                check_native_rows(changed,fixture(),FACTORS,work=self.work)


if __name__=='__main__':unittest.main()
