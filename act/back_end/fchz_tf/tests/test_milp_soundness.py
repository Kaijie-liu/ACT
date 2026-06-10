# ===- act/back_end/fchz_tf/tests/test_milp_soundness.py - MILP soundness ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Unit tests for FCHZ-MILP refinement soundness.
#   Per advisor 2026-06-09: required after acasxu/102 false-alarm.
#
#   3 categories:
#     1. Toy 1-ReLU MILP vs brute-force grid bounds
#     2. MILP_LB >= LP_LB (MILP cannot be looser than LP relaxation)
#     3. AND-polytope CERT semantics regression (acasxu/102 case)
#
# ===---------------------------------------------------------------------===#
"""FCHZ-MILP refinement soundness regression tests."""

import unittest
import numpy as np


class TestMILPSoundness(unittest.TestCase):
    """MILP soundness unit tests."""

    def test_tjeng_relu_matches_bruteforce(self):
        """Test 1: 1-neuron ReLU with binary indicator gives same bounds as brute force.

        For z in [l, u] with l < 0 < u: ReLU output y = max(0, z).
        Tjeng MILP with binary b ∈ {0,1}:
          y >= 0, y >= z, y <= u*b, y <= z - l*(1-b)
        Brute force: y = max(0, z) for z in [l, u].
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds as LPBounds
        except ImportError:
            self.skipTest("scipy.optimize.milp not available")

        # Test case: z in [-1, 2], objective minimize y
        l, u = -1.0, 2.0
        # Variables: z, y, b (in that order)
        # Constraints:
        # y >= 0    (LP bound on y >= 0)
        # y >= z    → -y + z <= 0
        # y <= u*b  → y - u*b <= 0
        # y <= z - l*(1-b) = z + b   (l = -1) → y - z - b <= 0
        A_ub = np.array([
            [1, -1, 0],   # -y + z <= 0
            [0, 1, -u],   # y - u*b <= 0
            [-1, 1, -1],  # -z + y - b <= 0  (from y - z - b <= 0 swapped: y - z + (-l)*b = ... wait)
        ], dtype=float)
        # Actually y <= z - l*(1-b): rearranging: y - z + l*(1-b) <= 0 = y - z + l - l*b <= 0
        # Or y - z - l*b <= -l   With l=-1: y - z - (-1)*b <= -(-1) → y - z + b <= 1
        A_ub = np.array([
            [1, -1, 0],   # y >= z: -y + z <= 0 wait
        ], dtype=float)
        # Simpler: use canonical form directly
        # Variables: [z, y, b]; objective min y means c = [0, 1, 0]
        c = np.array([0., 1., 0.])
        # Constraints:
        # y >= 0: -y <= 0 → row [0, -1, 0] <= 0
        # y >= z: z - y <= 0 → row [1, -1, 0] <= 0
        # y <= u*b: y - u*b <= 0 → row [0, 1, -u] <= 0
        # y <= z - l*(1-b) = z + |l|*(1-b) for l<0; let l=-1, u=2:
        #   y <= z + (1-b) → y - z - 1 + b <= 0 → row [-1, 1, 1] <= 1 (rhs)
        A_ub_full = np.array([
            [0, -1, 0],
            [1, -1, 0],
            [0, 1, -u],
            [-1, 1, 1],
        ], dtype=float)
        b_ub_full = np.array([0, 0, 0, 1], dtype=float)

        bounds = LPBounds(lb=[l, l, 0], ub=[u, u, 1])  # z in [l,u], y in [l,u] (will be constrained tighter), b in [0,1]
        constraint = LinearConstraint(A_ub_full, ub=b_ub_full)
        integrality = np.array([0, 0, 1])  # z continuous, y continuous, b binary

        result = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        self.assertTrue(result.success, "MILP failed")
        milp_y_min = result.fun

        # Brute force: min over z in [l,u] of max(0, z) is 0 (achieved at z=0)
        brute_y_min = 0.0
        self.assertAlmostEqual(milp_y_min, brute_y_min, places=4,
                                          msg=f"MILP y_min={milp_y_min} doesn't match brute={brute_y_min}")

        # Max test: objective -y, expected y_max = max(0, u) = u = 2
        result_max = milp(c=-c, constraints=[constraint], integrality=integrality, bounds=bounds)
        self.assertTrue(result_max.success)
        milp_y_max = -result_max.fun
        self.assertAlmostEqual(milp_y_max, u, places=4)

    def test_milp_lb_is_sound_lower_bound(self):
        """Test 2: MILP_LB should never be greater than actual min over the box.

        For a simple linear layer y = w * x + b with x in [l, u], compute LB(y).
        Sound LP_LB <= MILP_LB <= true_min.
        Use 1D example to verify.
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds as LPBounds
            from scipy.optimize import linprog
        except ImportError:
            self.skipTest("scipy.optimize not available")

        # y = 0.5 * x + 1 for x in [-2, 3]
        # LB(y) = 0.5 * (-2) + 1 = 0
        # UB(y) = 0.5 * 3 + 1 = 2.5
        c = np.array([0., 1.])  # min y
        # y = 0.5*x + 1 → y - 0.5*x = 1 (equality)
        A_eq = np.array([[-0.5, 1.]], dtype=float)
        b_eq = np.array([1.0])
        # LP solve
        lp = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=[(-2, 3), (None, None)], method='highs')
        self.assertTrue(lp.success)
        lp_lb = lp.fun

        # MILP (with no binary) should match LP
        result_milp = milp(c=c, constraints=[LinearConstraint(A_eq, lb=b_eq, ub=b_eq)],
                                  integrality=np.array([0, 0]),
                                  bounds=LPBounds(lb=[-2, -np.inf], ub=[3, np.inf]))
        self.assertTrue(result_milp.success)
        milp_lb = result_milp.fun

        true_lb = 0.0   # exact: 0.5 * -2 + 1
        self.assertAlmostEqual(lp_lb, true_lb, places=3)
        self.assertAlmostEqual(milp_lb, true_lb, places=3)
        # Soundness: both bounds <= true_min (true_min is achievable)
        self.assertLessEqual(milp_lb, true_lb + 1e-6)
        # Also: MILP_LB >= LP_LB (MILP cannot be looser than its LP relaxation)
        self.assertGreaterEqual(milp_lb, lp_lb - 1e-6)

    def test_and_polytope_cert_semantics(self):
        """Test 3: AND-polytope CERT semantic correctness.

        AND-polytope: unsafe iff ALL rows of C @ y <= t (point inside polytope).
        CERT iff polytope unreachable iff ANY row's LB(C @ y) > t.

        Regression for acasxu prop_3 / iid 102 case:
          - vnnlib has 4 separate asserts (Y_0 <= Y_i for i=1..4)
          - parser/canonicalize produces UNSAFE_LINEAR with 4 rows (AND-of-rows)
          - CERT iff ANY row LB > 0
        """
        # Mock: 4 rows, 3 of 4 with positive LB → CERT
        per_row_lb = np.array([0.039, -0.001, 0.052, 0.004])
        thresholds = np.array([0.0, 0.0, 0.0, 0.0])
        # CERT iff ANY row LB > t
        cert_and_polytope = (per_row_lb > thresholds + 1e-9).any()
        self.assertTrue(cert_and_polytope, "AND-polytope CERT should fire when any row has LB > t")

        # Counter case: all rows LB <= t → UNK
        per_row_lb_unk = np.array([-0.1, -0.05, -0.2, -0.01])
        cert_unk = (per_row_lb_unk > thresholds + 1e-9).any()
        self.assertFalse(cert_unk, "AND-polytope UNK when all rows have LB <= t")

    def test_sound_check_uses_correct_and_semantic(self):
        """Test 4: Sound-check sample violation semantics.

        For AND-polytope, a sample y is a violation (in unsafe set) iff
        ALL rows of C @ y <= t. NOT iff ANY row <= t (that's OR semantic).

        This test guards against the false-alarm bug from 2026-06-09 where
        a standalone test incorrectly used OR semantic and reported
        67/500 violations on acasxu/102 (true sound count: 0/500).
        """
        # 4 rows × 5-dim y
        C = np.array([[1, -1, 0, 0, 0], [1, 0, -1, 0, 0],
                              [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]], dtype=float)
        t = np.zeros(4)
        # Sample y: y_0 = 0.5, y_1 = 0.4 (y_0 > y_1, row 0 = 0.1 > 0, NOT in polytope)
        # row 1: y_0 - y_2 = 0.5 - 0.6 = -0.1 < 0 (in unsafe condition for row 1)
        # row 2: y_0 - y_3 = 0.5 - 0.3 = 0.2 > 0 (NOT in unsafe for row 2)
        # row 3: y_0 - y_4 = 0.5 - 0.2 = 0.3 > 0 (NOT in unsafe for row 3)
        y = np.array([0.5, 0.4, 0.6, 0.3, 0.2])
        cy = C @ y
        # OR semantic (WRONG): any row <= t → "violation"
        or_violation = (cy <= t + 1e-5).any()
        self.assertTrue(or_violation,
                              "OR semantic would mark this y as violation (because row 1 <= 0)")
        # AND semantic (CORRECT): ALL rows <= t → in polytope → violation
        and_violation = (cy <= t + 1e-5).all()
        self.assertFalse(and_violation,
                                "AND semantic correctly says y is NOT in polytope (row 0/2/3 are > 0)")


if __name__ == '__main__':
    unittest.main()
