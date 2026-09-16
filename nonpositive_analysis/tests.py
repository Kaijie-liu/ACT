import copy
import unittest
from fractions import Fraction as F
from act.back_end.solver.lp_certificate import identity
from nonpositive_analysis.analyze import decompose


def fixture(offset=1):
    # min offset+x; x in [-1,1], lambda in [0,1], w=0. Four harmless rows.
    lp={'matrix_format':'csr_v1','c':[1,0,0],'offset':offset,'lower':[-1,0,0],'upper':[1,1,0],
        'A':{'shape':[4,3],'indptr':[0,0,0,0,0],'indices':[],'data':[]},'b':[0]*4,
        'E':{'shape':[0,3],'indptr':[0],'indices':[],'data':[]},'h':[]}
    cert={'lp_sha256':identity(lp),'inequality_dual':[0]*4,'equality_dual':[],
          'claimed_lower_bound':str(offset-1)}
    return lp,cert


class Controls(unittest.TestCase):
    def test_exact_residual_accounting(self):
        lp,c=fixture();r=decompose(lp,c,1,0)
        self.assertEqual(r['lower']['exact'],'0');self.assertEqual(r['residual_box_term']['exact'],'-1')
        self.assertEqual(r['base_without_box_term_NOT_A_BOUND']['exact'],'1')

    def test_weak_lower_does_not_determine_lp_minimum(self):
        lp,c=fixture(0)
        # x>=1: LP optimum +1 but zero dual gives lower -1.
        lp['A']={'shape':[4,3],'indptr':[0,1,1,1,1],'indices':[0],'data':[-1]};lp['b'][0]=-1
        c['lp_sha256']=identity(lp);c['claimed_lower_bound']='-1'
        self.assertEqual(decompose(lp,c,1,0)['lower']['exact'],'-1')
        # The same lower bound also occurs without this constraint, optimum -1.
        a,b=fixture(0);self.assertEqual(decompose(a,b,1,0)['lower']['exact'],'-1')

    def test_identity_sign_and_overclaim_rejected(self):
        lp,c=fixture()
        for field,value in [('lp_sha256','0'*64),('inequality_dual',[1]*4),('claimed_lower_bound','1')]:
            bad=copy.deepcopy(c);bad[field]=value
            with self.assertRaises(ValueError):decompose(lp,bad,1,0)

    def test_zero_width_product_and_group_sum(self):
        lp,c=fixture(2);r=decompose(lp,c,1,0)
        self.assertEqual(sum(F(g['box_contribution']['exact']) for g in r['residual_groups'].values()),F(-1))
        self.assertEqual(r['residual_groups']['product']['box_contribution']['exact'],'0')
