import copy
import json
from pathlib import Path
import unittest
from fractions import Fraction
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.solver.hz_lp_export import export
from act.back_end.solver.lp_certificate import identity, propose
from act.pipeline.moe.check_request_lp import aggregate, TRUSTED
from act.pipeline.moe.request_lp_control import outward, frozen_request


class RequestLPTests(unittest.TestCase):
    def test_frozen_real_control_accounting(self):
        r=json.loads((Path(__file__).parent/'results/request_lp_review_20260914_r1.json').read_text())
        self.assertEqual(r['check']['status'],'UNKNOWN')
        self.assertEqual(r['checked_lp_count'],36)
        self.assertEqual(r['check']['counts'],{'reused':15,'residual':0,'unknown':3})
        self.assertEqual(len(r['obligations']),18)
        self.assertEqual(sum(p['kind']=='weighted' for p in r['proofs'].values()),3)
    def fixture(self, positive=True):
        request={'top_k':2,'tie_policy':'ANY_LEGAL_TOPK','experts':2,'classes':2,'clean_prediction':0}
        rid=identity(request)
        m={'schema':'request_lp_v1','request':request,'request_id':rid,'trusted_base':TRUSTED,
           'positive_threshold':1e-7,'routes':{'feasible':[[0,1]],'infeasible':[],'unresolved':[],'exact':True},
           'proofs':{},'obligations':[{'pair':[0,1],'property_index':0,'kind':'reused','sources':['a','b']}]}
        store={}
        for key,e in [('a',0),('b',1)]:
            hz=SparseHZono(c=np.array([2. if positive else -2.,0.]),Gc=sp.csr_matrix([[.1],[0.]]),
                Gb=sp.csr_matrix((2,0)),Ac=sp.csr_matrix((0,1)),Ab=sp.csr_matrix((0,0)),b=np.array([]),
                Auc=sp.csr_matrix((0,1)),Aub=sp.csr_matrix((0,0)),ub=np.array([]))
            record=export(hz,[1,-1],sparse=True); cert=propose(record['lp'])
            store[key+'e']=record;store[key+'c']=cert
            m['proofs'][key]={'request_id':rid,'kind':'expert','scope':{'membership':e},'property_index':0,
                'status':'CHECKED','export':key+'e','certificate':key+'c','hz_sha256':record['source_sha256']}
        return m,store

    def test_complete_reuse_without_solver(self):
        m,s=self.fixture()
        with patch('scipy.optimize.linprog',side_effect=AssertionError('solver called')):
            r=aggregate(m,s.__getitem__)
        self.assertEqual(r['status'],'CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING')
        self.assertEqual(r['counts'],{'reused':1,'residual':0,'unknown':0})

    def test_scope_inventory_and_property_mutations(self):
        for change in (lambda m:m['proofs']['b']['scope'].update(membership=0),
                       lambda m:m['proofs']['a'].update(request_id='wrong'),
                       lambda m:m['obligations'].clear(),
                       lambda m:m['obligations'].append(copy.deepcopy(m['obligations'][0])),
                       lambda m:m['routes']['infeasible'].append([0,1]),
                       lambda m:m.update(positive_threshold=0)):
            m,s=self.fixture();change(m)
            with self.assertRaises(ValueError):aggregate(m,s.__getitem__)
        m,s=self.fixture();s['ae']['q']=[-1,1]
        with self.assertRaises(ValueError):aggregate(m,s.__getitem__)

    def test_unknown_and_invalid_reuse_fail_closed(self):
        m,s=self.fixture(False)
        with self.assertRaises(ValueError):aggregate(m,s.__getitem__)
        m['obligations'][0]['kind']='unknown'
        self.assertEqual(aggregate(m,s.__getitem__)['status'],'UNKNOWN')
        m['routes'].update(feasible=[],unresolved=[[0,1]],exact=False)
        self.assertEqual(aggregate(m,s.__getitem__)['reason'],'INCOMPLETE_ROUTE_COVERAGE')

    def test_outward_and_frozen_identity(self):
        for v in [Fraction(1,3),Fraction(-1,3),Fraction(1,10)]:
            self.assertLessEqual(Fraction(outward(v)),v)
            self.assertGreaterEqual(Fraction(outward(v,True)),v)
        self.assertEqual(frozen_request()['dataset_index'],3000)

    def test_residual_and_cached_wrong_objective_rejected(self):
        m,s=self.fixture()
        def add(key,centers,q,kind):
            n=len(centers)
            hz=SparseHZono(c=np.array(centers,dtype=float),Gc=sp.csr_matrix((n,1)),Gb=sp.csr_matrix((n,0)),
                Ac=sp.csr_matrix((0,1)),Ab=sp.csr_matrix((0,0)),b=np.array([]),
                Auc=sp.csr_matrix((0,1)),Aub=sp.csr_matrix((0,0)),ub=np.array([]))
            e=export(hz,q,sparse=True);s[key+'e']=e;s[key+'c']=propose(e['lp'])
            m['proofs'][key]={'request_id':m['request_id'],'kind':kind,'scope':{'pair':[0,1]},
                'property_index':0,'status':'CHECKED','export':key+'e','certificate':key+'c','hz_sha256':e['source_sha256']}
        add('lo',[2,0,1,0],[1,-1,-1,1],'difference')
        add('hi',[2,0,1,0],[-1,1,1,-1],'difference')
        add('out',[1],[1],'weighted')
        row={'pair':[0,1],'property_index':0,'kind':'residual','difference_lower':'lo',
             'difference_upper':'hi','difference_bounds':[1,1],'lambda_bounds':[0,1],'source':'out'}
        m['obligations']=[row]
        self.assertEqual(aggregate(m,s.__getitem__)['counts']['residual'],1)
        row['difference_upper']='lo'
        with self.assertRaisesRegex(ValueError,'wrong output property'):aggregate(m,s.__getitem__)
        row['difference_upper']='hi';row['difference_bounds']=[1.1,1.2]
        with self.assertRaises(ValueError):aggregate(m,s.__getitem__)


if __name__=='__main__':unittest.main()
