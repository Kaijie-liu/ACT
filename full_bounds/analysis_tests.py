"""Read-only post-result analysis controls; not part of the frozen solve path."""
import copy
from fractions import Fraction as F
import unittest
from full_bounds.analysis import point_terms
from source_enclosure.format import sparse


class AnalysisControls(unittest.TestCase):
    def fixture(self):
        return {'gate':['0','1'],'difference':['-1','3'],'offset':'2',
            'A_extra':sparse([{}, {0:-2,1:3,2:-1},{},{}],3),'b_extra':['0','2','0','0'],
            'objective':sparse([{0:1,2:1}],3)}

    def test_point_arithmetic_is_not_a_verdict(self):
        r=point_terms(self.fixture(),['1/4','1/2','1/5'],3)
        self.assertEqual(F(r['u']),F(9,4));self.assertEqual(F(r['d']),F(1,2))
        self.assertEqual(F(r['relaxed_objective']),F(49,20))
        self.assertEqual(F(r['product_replaced_objective']),F(5,2))
        self.assertEqual(F(r['product_minus_relaxed_product']),F(1,20))
        self.assertNotIn('status',r)

    def test_wrong_formula_and_size_reject(self):
        for key,value in [('gate',['0','1/2']),('difference',['3','-1']),('b_extra',['0','0','0','0'])]:
            r=copy.deepcopy(self.fixture());r[key]=value
            with self.assertRaises(ValueError):point_terms(r,[0,0,0],3)
        with self.assertRaises(ValueError):point_terms(self.fixture(),[0,0],3)


if __name__ == '__main__':unittest.main()
