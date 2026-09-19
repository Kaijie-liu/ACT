import unittest
from copy import deepcopy
from lp_sandwich.tests import csr
from basis_compatibility.review import assess


def lp(n=2, rows=1):
    return {'c':[0]*n, 'lower':[0]*n, 'upper':[1]*n, 'b':[1]*rows, 'h':[],
            'A':csr([[1]*n for _ in range(rows)], n), 'E':csr([], n)}


class Controls(unittest.TestCase):
    def test_small_is_not_automatic_freeze(self):
        r=assess(lp())
        self.assertEqual(r['status'], 'NO_STATIC_BLOCKER_NOT_RUNTIME_APPROVAL')
        self.assertFalse(r['freeze_eligible'])

    def test_dimension_boundaries(self):
        self.assertNotIn('variables', assess(lp(64))['static_blockers'])
        self.assertIn('variables', assess(lp(65))['static_blockers'])
        self.assertNotIn('augmented_equations', assess(lp(2,64))['static_blockers'])
        self.assertIn('augmented_equations', assess(lp(2,65))['static_blockers'])

    def test_equations_include_both_original_row_types(self):
        v=lp(2,32);v['E']=csr([[1,0]]*33,2);v['h']=[1]*33
        self.assertEqual(assess(v)['dimensions']['augmented_equations'],65)

    def test_scalar_and_pivot_necessary_minima(self):
        v=lp(2,4097);r=assess(v)
        self.assertIn('full_rank_pivot_nnz_lower_bound',r['static_blockers'])
        self.assertEqual(r['dimensions']['initial_scalar_visits_lower_bound'],6+4097+1+8194)
        self.assertIn('input_nnz',r['static_blockers'])
        r=assess(lp(100,2100))
        self.assertIn('initial_scalar_visits_lower_bound',r['static_blockers'])

    def test_bad_shape_rejected(self):
        v=lp();v['E']['shape']=[0,3]
        with self.assertRaises(ValueError):assess(v)

    def test_csr_omission_rejected(self):
        v=lp();v['A']['indices'].pop()
        with self.assertRaises(ValueError):assess(v)


if __name__=='__main__':unittest.main()
