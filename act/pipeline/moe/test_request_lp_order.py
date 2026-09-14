import copy
import unittest
from fractions import Fraction
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.solver.hz_lp_export import export
from act.back_end.solver.lp_certificate import identity, propose
from act.pipeline.moe.check_request_lp import TRUSTED, aggregate, order_envelope


class OrderLPTests(unittest.TestCase):
    def test_order_and_tie_enclosures(self):
        self.assertEqual(order_envelope(Fraction(1),Fraction(-2)),[.5,1])
        self.assertEqual(order_envelope(Fraction(-2),Fraction(1)),[0,.5])
        self.assertEqual(order_envelope(Fraction(0),Fraction(0)),[.5,.5])
        self.assertEqual(order_envelope(Fraction(-1),Fraction(-1)),[0,1])
        self.assertEqual(order_envelope(None,None),[0,1])
        self.assertEqual(order_envelope(None,Fraction(1)),[0,.5])
        with self.assertRaises(ValueError):order_envelope(Fraction(1),Fraction(1))

    def fixture(self):
        request={'top_k':2,'tie_policy':'ANY_LEGAL_TOPK','experts':2,'classes':2,'clean_prediction':0}
        rid=identity(request);store={}
        m={'schema':'request_lp_order_v2','request':request,'request_id':rid,'trusted_base':TRUSTED,
           'positive_threshold':1e-7,'routes':{'feasible':[[0,1]],'infeasible':[],'unresolved':[],'exact':True},
           'proofs':{},'obligations':[{'pair':[0,1],'property_index':0,'kind':'residual',
             'difference_lower':'dl','difference_upper':'du','difference_bounds':[1,1],
             'gate_lower':'gl','gate_upper':'gu','lambda_bounds':[0,.5],'source':'out'}]}
        for key,c,q,kind,prop in [('dl',[2,0,1,0],[1,-1,-1,1],'difference',0),
                                ('du',[2,0,1,0],[-1,1,1,-1],'difference',0),
                                ('gl',[-1,0],[1,-1],'router_order',None),
                                ('gu',[-1,0],[-1,1],'router_order',None),
                                ('out',[1],[1],'weighted',0)]:
            hz=SparseHZono(c=np.array(c,dtype=float),Gc=sp.csr_matrix((len(c),1)),Gb=sp.csr_matrix((len(c),0)),
                Ac=sp.csr_matrix((0,1)),Ab=sp.csr_matrix((0,0)),b=np.array([]),
                Auc=sp.csr_matrix((0,1)),Aub=sp.csr_matrix((0,0)),ub=np.array([]))
            record=export(hz,q,sparse=True);store[key+'e']=record;store[key+'c']=propose(record['lp'])
            m['proofs'][key]={'request_id':rid,'kind':kind,'scope':{'pair':[0,1]},'property_index':prop,
                'status':'CHECKED','export':key+'e','certificate':key+'c','hz_sha256':record['source_sha256']}
        return m,store

    def test_gate_bound_checked_without_solver_and_reject_mutations(self):
        m,s=self.fixture()
        with patch('scipy.optimize.linprog',side_effect=AssertionError('solver in checker')):
            self.assertEqual(aggregate(m,s.__getitem__)['counts']['residual'],1)
        for change in (lambda m:m['obligations'][0].update(lambda_bounds=[0,.1]),
                       lambda m:m['obligations'][0].update(gate_upper='gl'),
                       lambda m:m['proofs']['gu'].update(scope={'pair':[1,0]}),
                       lambda m:m['proofs']['gu'].update(property_index=0),
                       lambda m:m.update(schema='request_lp_v1')):
            changed=copy.deepcopy(m);change(changed)
            with self.assertRaises(ValueError):aggregate(changed,s.__getitem__)
        m['proofs']['gu']['status']='UNKNOWN';m['obligations'][0]['lambda_bounds']=[0,1]
        self.assertEqual(aggregate(m,s.__getitem__)['counts']['residual'],1)


if __name__=='__main__':unittest.main()
