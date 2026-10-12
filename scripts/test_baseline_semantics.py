"""No models/solvers: geometry, quantifiers, tie cases and identity rejection."""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import analyze_baseline_semantics as a


class Contracts(unittest.TestCase):
    def test_box_corner_differential(self):
        c = [Q(1,3), Q(-2,3), 0]
        lo = [0, -1, -2]; hi = [1, 0, 3]; scales = [1, 2, Q(1,2)]
        explicit = max(sum((s*(x-y))**2 for x,y,s in zip(corner,c,scales))
                       for corner in product(*zip(lo,hi)))
        self.assertEqual(a.box_radius_squared(c,lo,hi,scales), explicit)

    def test_clipping_normalization_and_ball_boundary(self):
        self.assertEqual(a.box_radius_squared([0],[0],[Q(1,4)]), Q(1,16))
        self.assertEqual(a.box_radius_squared([0],[0],[Q(1,4)],[4]), 1)
        self.assertTrue(a.contained_in_l2([0],[0],[Q(1,4)],Q(1,4)))
        self.assertFalse(a.contained_in_l2([0],[0],[Q(1,4)],Q(1,4),open_ball=True))
        self.assertFalse(a.contained_in_l2([0],[0],[Q(1,4)],Q(1,4),[4]))
        self.assertTrue(a.contained_in_l2([0],[0],[0],0))
        self.assertFalse(a.contained_in_l2([0],[0],[0],0,open_ball=True))

    def test_invalid_and_implicit_float_geometry(self):
        for args in [([],[],[]),([0],[0],[1,2]),([0],[1],[0]),
                     ([0],[0],[1],[0]),([0.0],[0],[1])]:
            with self.assertRaises(ValueError): a.box_radius_squared(*args)
        with self.assertRaises(ValueError): a.contained_in_l2([0],[0],[1],-1)
        with self.assertRaises(ValueError): a.contained_in_l2([0],[0],[1],1,open_ball=1)

    def test_union_is_not_average_or_independence(self):
        self.assertNotEqual(Q.from_float(.0005),Q(1,2000))
        self.assertEqual(a.union_failure([Q(1,2000)]*2),Q(1,1000))
        self.assertEqual(a.union_failure([Q(1,2000)]*4),Q(1,500))
        self.assertEqual(a.union_failure([Q(3,4)]*2),1)
        for values in ([],[-1],[Q(11,10)],[.001],[True]):
            with self.assertRaises(ValueError): a.union_failure(values)

    def test_topk_ties_and_subnormalized_weights(self):
        delta = Q.from_float(1e-5)
        bound, low, high = a.selected_mass_bounds(4,2,delta)
        self.assertEqual(bound,Q(1,2)); self.assertLess(low,high); self.assertLess(high,1)
        # Exhaustive positive rational score controls, including all-equal ties.
        # They test the simplex arithmetic; no exp implementation is called.
        for raw in product(range(1,4),repeat=4):
            p = sorted((Q(v,sum(raw)) for v in raw),reverse=True)
            mass = sum(p[:2]); rho = mass/(mass+delta)
            self.assertGreaterEqual(mass,bound)
            self.assertLessEqual(low,rho); self.assertLessEqual(rho,high)
        for args in [(0,0,delta),(4,5,delta),(4,2,0),(4,2,-1),(4.,2,delta)]:
            with self.assertRaises(ValueError): a.selected_mass_bounds(*args)

    def test_not_a_normalized_affine_property_or_intermediate_equivalence(self):
        # Both experts satisfy z-1=0; subnormalization fails that same property.
        delta = Q(1,100000); s = Q(1,2); weighted = s/(s+delta)
        self.assertLess(weighted-1,0)
        # A later bias means an intermediate positive scaling need not preserve
        # even a final homogeneous margin.
        bias = -(1+weighted)/2
        self.assertGreater(1+bias,0); self.assertLess(weighted+bias,0)

    def test_history_count_is_not_k1_compatibility(self):
        self.assertEqual(a.history_count([(4,2)]*5),7776)
        self.assertEqual(a.history_count([(4,1)]*5),1024)
        self.assertEqual(a.history_count([]),1)
        with self.assertRaises(ValueError): a.history_count([(2,3)])

    def test_metamoe_zero_block_excludes_cross_domain_positive_margin(self):
        # Every y has a same-block competitor in the unselected block,
        # independent of the selected expert's arbitrary 10 logits.
        for selected in ((1,)*10,(-1,)*10,tuple(range(10))):
            for route in (0,1):
                full = selected+(0,)*10 if route == 0 else (0,)*10+selected
                offset = 10*(1-route)
                for y in range(offset,offset+10):
                    other = offset+(y-offset+1)%10
                    self.assertEqual(full[y]-full[other],0)
                    self.assertLess(full[y]-full[other],Q(1,10_000_000))

    def test_identity_mutation_and_no_model_admission(self):
        with patch.object(Path,'read_bytes',return_value=b'changed source'):
            with self.assertRaisesRegex(ValueError,'source identity mismatch'):
                a.verify_local_sources()
        r = a.report()
        self.assertEqual((r['model_queries'],r['solver_calls'],r['gpu_calls']),(0,0,0))
        self.assertFalse(r['new_execution_admitted'])
        self.assertFalse(r['dual_rs']['pilot_radius_reclassified'])
        self.assertFalse(r['dual_rs']['native_probability_tail_arithmetic_checked'])
        self.assertFalse(r['robust_experts']['all_histories_reachable_established'])


if __name__ == '__main__': unittest.main()
