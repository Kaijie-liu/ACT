import copy
from fractions import Fraction
import unittest
from unittest.mock import patch
from act.back_end.solver.hz_lp_export import export
from act.back_end.solver.check_hz_lp_export import check_export
from act.back_end.solver.lp_certificate import check, propose, identity
from act.back_end.solver.sparse_lp_certificate import rows
from act.back_end.solver.solver_hz import SparseHZono
import numpy as np
import scipy.sparse as sp


class SparseProofTests(unittest.TestCase):
    def record(self):
        hz=SparseHZono(c=np.array([.1]), Gc=sp.csr_matrix([[1.,2.]]), Gb=sp.csr_matrix((1,0)),
            Ac=sp.csr_matrix([[1.,1.]]), Ab=sp.csr_matrix((1,0)), b=np.array([0.]),
            Auc=sp.csr_matrix([[-1.,0.]]), Aub=sp.csr_matrix((1,0)), ub=np.array([0.]))
        return export(hz,[1],sparse=True),export(hz,[1])

    def test_sparse_dense_same_exact_bound_and_no_solver_in_check(self):
        a,b=self.record(); ca,cb=propose(a['lp']),propose(b['lp'])
        with patch('scipy.optimize.linprog',side_effect=AssertionError('checker called solver')):
            ra=check_export(a,ca,expected_source_sha256=a['source_sha256'])
            rb=check_export(b,cb,expected_source_sha256=b['source_sha256'])
        self.assertEqual(Fraction(ra['bound']['checked_lower_bound']),Fraction(rb['bound']['checked_lower_bound']))

    def test_shape_dual_claim_and_coefficient_mutations(self):
        r,_=self.record(); c=propose(r['lp'])
        for change in (lambda x:x['A']['data'].__setitem__(0,1.),
                       lambda x:x['A']['indices'].__setitem__(0,99),
                       lambda x:x['A']['indptr'].__setitem__(-1,0)):
            changed=copy.deepcopy(r); change(changed['lp'])
            with self.assertRaises(ValueError): check_export(changed,c,expected_source_sha256=r['source_sha256'])
        for change in (lambda x:x.__setitem__('claimed_lower_bound','1000'),
                       lambda x:x['inequality_dual'].__setitem__(0,1.)):
            d=copy.deepcopy(c);change(d)
            with self.assertRaises(ValueError):check(r['lp'],d)
        with self.assertRaises(ValueError):list(rows({'shape':[1,2],'data':[1,2],'indices':[0,0],'indptr':[0,2]},2))


if __name__=='__main__':unittest.main()
