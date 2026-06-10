"""Unit tests for verify_once_fchz spec semantics (advisor M2b STEP 2).

Critical: verify each spec kind uses the correct bound direction
(LINEAR_LE/TOP1_ROBUST use UB, UNSAFE_LINEAR uses LB).

Sign-flip test ensures C and -C don't give same verdict.
"""
import sys
sys.path.insert(0, '/data1/Kane/ACT')

import numpy as np
import unittest

from act.back_end.fchz_tf.representations import FCHZState
from act.back_end.fchz_tf.verifier_fchz import (
    fchz_upper_bound, fchz_lower_bound, _decide_kind)


def make_state(c, G=None, tail_radius=None):
    """Build FCHZState quickly."""
    c = np.asarray(c, dtype=np.float64)
    n = c.shape[0]
    if G is None:
        G = np.zeros((n, 0), dtype=np.float64)
    else:
        G = np.asarray(G, dtype=np.float64)
    if tail_radius is not None:
        tail_radius = np.asarray(tail_radius, dtype=np.float64)
    return FCHZState(c=c, G=G, n_root=0, slack_records=[],
                          tail_radius=tail_radius)


class TestBoundDirections(unittest.TestCase):
    """Test UB/LB are correctly computed and OPPOSITE in direction."""

    def test_ub_lb_simple(self):
        # State: y = c + G·xi, xi in [-1, 1]
        # c=(0,0), G=I → R = [-1,1]^2
        c = np.array([0.0, 0.0])
        G = np.eye(2)
        s = make_state(c, G)
        # d=(1, 0): max y_0 = 0 + 1 = 1, min = -1
        C = np.array([[1.0, 0.0]])
        ub = fchz_upper_bound(s, C)
        lb = fchz_lower_bound(s, C)
        np.testing.assert_allclose(ub, [1.0])
        np.testing.assert_allclose(lb, [-1.0])

    def test_ub_lb_correlated(self):
        # Shared xi: G = [[1], [1]] → y = (xi, xi)
        # For d=(1, -1): d@y = xi - xi = 0 always.
        c = np.array([0.0, 0.0])
        G = np.array([[1.0], [1.0]])
        s = make_state(c, G)
        C = np.array([[1.0, -1.0]])
        ub = fchz_upper_bound(s, C)
        lb = fchz_lower_bound(s, C)
        # |d@G| = |[1, -1]@[1,1]| = |0| = 0 (correlation captured!)
        np.testing.assert_allclose(ub, [0.0])
        np.testing.assert_allclose(lb, [0.0])

    def test_tail_radius_added(self):
        c = np.array([0.0])
        G = np.zeros((1, 0))
        tail = np.array([0.5])
        s = make_state(c, G, tail)
        C = np.array([[1.0]])
        ub = fchz_upper_bound(s, C)
        lb = fchz_lower_bound(s, C)
        # UB = 0 + 0 + |1|*0.5 = 0.5
        # LB = 0 - 0 - 0.5 = -0.5
        np.testing.assert_allclose(ub, [0.5])
        np.testing.assert_allclose(lb, [-0.5])


class TestLinearLE(unittest.TestCase):
    """LINEAR_LE: safe iff C y <= d. CERT iff max(Cy) < d."""

    def test_safe_obvious(self):
        # y=1 with no noise → 2*1 = 2 < 3
        s = make_state([1.0])
        C = np.array([[2.0]])
        thresholds = np.array([3.0])
        result = _decide_kind('LINEAR_LE', s, C, thresholds, M_rows=1)
        self.assertEqual(result['verdict'], 'CERTIFIED')
        self.assertEqual(result['bound_type'], 'UB')

    def test_unsafe_obvious(self):
        # y=1 with no noise → 2*1 = 2 NOT < 1
        s = make_state([1.0])
        C = np.array([[2.0]])
        thresholds = np.array([1.0])
        result = _decide_kind('LINEAR_LE', s, C, thresholds, M_rows=1)
        self.assertEqual(result['verdict'], 'UNKNOWN')


class TestTop1Robust(unittest.TestCase):
    """TOP1_ROBUST: rows e_rival - e_true. CERT iff max(Cy) < 0."""

    def test_safe_top1(self):
        # 2-class with y_true=0, c=(2, 1) (true beats rival by 1)
        c = np.array([2.0, 1.0])
        G = np.eye(2) * 0.1   # small noise
        s = make_state(c, G)
        # Row: e_1 - e_0 (rival - true). max(y_1 - y_0) over noise
        C = np.array([[-1.0, 1.0]])
        thresholds = np.array([0.0])
        # max(y_1 - y_0) = max((1+0.1xi_1) - (2+0.1xi_0)) = -1 + 0.2 = -0.8 < 0 → CERT
        result = _decide_kind('TOP1_ROBUST', s, C, thresholds, M_rows=1)
        self.assertEqual(result['verdict'], 'CERTIFIED')

    def test_unsafe_top1(self):
        # Reverse: true=0 loses to rival
        c = np.array([1.0, 2.0])
        s = make_state(c)
        # Row: e_1 - e_0
        C = np.array([[-1.0, 1.0]])
        thresholds = np.array([0.0])
        # max(y_1 - y_0) = 2 - 1 = 1 > 0 → UNK
        result = _decide_kind('TOP1_ROBUST', s, C, thresholds, M_rows=1)
        self.assertEqual(result['verdict'], 'UNKNOWN')


class TestUnsafeLinear(unittest.TestCase):
    """UNSAFE_LINEAR: unsafe iff C y <= d. CERT iff min(Cy) > d.

    CRITICAL: must use LOWER bound, not upper!
    """

    def test_safe_via_lb(self):
        # State y = 3 (no noise). Unsafe: y <= 1. y=3 > 1 → SAFE.
        s = make_state([3.0])
        C = np.array([[1.0]])
        thresholds = np.array([1.0])
        # min(y) = 3 > 1 → CERT
        result = _decide_kind('UNSAFE_LINEAR', s, C, thresholds, M_rows=1)
        self.assertEqual(result['verdict'], 'CERTIFIED')
        self.assertEqual(result['bound_type'], 'LB')

    def test_unsafe_via_lb(self):
        # State y = 1.5 ± 1 (range [0.5, 2.5]). Unsafe: y <= 1.
        # min(y) = 0.5, NOT > 1 → UNK (overlap with unsafe set)
        s = make_state([1.5], G=np.array([[1.0]]))
        C = np.array([[1.0]])
        thresholds = np.array([1.0])
        result = _decide_kind('UNSAFE_LINEAR', s, C, thresholds, M_rows=1)
        self.assertEqual(result['verdict'], 'UNKNOWN')

    def test_sign_flip_safe(self):
        """Same state, opposite C direction.

        Both with their proper threshold sign should still give CERT.
        """
        # State: y = 3 (no noise)
        s = make_state([3.0])
        # UNSAFE_LINEAR with C=[1], d=1: unsafe iff y <= 1. y=3 > 1 → SAFE.
        # Using LB: min(y)=3 > 1 → CERT
        result_pos = _decide_kind('UNSAFE_LINEAR', s,
                                                np.array([[1.0]]),
                                                np.array([1.0]),
                                                M_rows=1)
        self.assertEqual(result_pos['verdict'], 'CERTIFIED')

        # UNSAFE_LINEAR with C=[-1], d=-1: unsafe iff -y <= -1, i.e. y >= 1. y=3 → UNSAFE region overlap.
        # Using LB: min(-y) = -3, NOT > -1 → UNK
        result_neg = _decide_kind('UNSAFE_LINEAR', s,
                                                np.array([[-1.0]]),
                                                np.array([-1.0]),
                                                M_rows=1)
        # This is the OPPOSITE direction encoding the SAME unsafe set (y >= 1 is different region)
        # We expect UNK because state y=3 IS in y >= 1
        self.assertEqual(result_neg['verdict'], 'UNKNOWN')


class TestSignFlip(unittest.TestCase):
    """Critical: C and -C must NOT give same verdict for same spec kind."""

    def test_linear_le_sign_flip(self):
        # State y=3
        s = make_state([3.0])
        # LINEAR_LE: C=[1], d=5 → safe iff y <= 5. y=3 ≤ 5 → CERT.
        r1 = _decide_kind('LINEAR_LE', s, np.array([[1.0]]),
                                  np.array([5.0]), M_rows=1)
        # LINEAR_LE: C=[-1], d=5 → safe iff -y <= 5, i.e. y >= -5. y=3 ≥ -5 → CERT.
        r2 = _decide_kind('LINEAR_LE', s, np.array([[-1.0]]),
                                  np.array([5.0]), M_rows=1)
        # Both CERT but for different reasons. NOT same verdict-from-error.
        # Test instead: opposite C with SAME threshold gives different SAFETY direction
        # C=[1], d=2 → y<=2 unsafe → y=3 UNSAFE → max(Cy)=3 NOT < 2 → UNK
        r3 = _decide_kind('LINEAR_LE', s, np.array([[1.0]]),
                                  np.array([2.0]), M_rows=1)
        # C=[-1], d=2 → -y<=2 → y>=-2 → y=3 satisfies → max(-y)=-3 < 2 → CERT
        r4 = _decide_kind('LINEAR_LE', s, np.array([[-1.0]]),
                                  np.array([2.0]), M_rows=1)
        # Critical: r3 != r4 (one says UNK, one says CERT for same y=3)
        self.assertNotEqual(r3['verdict'], r4['verdict'])


class TestUbLbConsistency(unittest.TestCase):
    """UB > LB always (when state has any noise)."""

    def test_ub_ge_lb(self):
        c = np.array([1.0, 2.0])
        G = np.array([[0.1, 0.2], [0.3, 0.4]])
        s = make_state(c, G)
        C = np.array([[1.0, -1.0], [0.5, 0.5]])
        ub = fchz_upper_bound(s, C)
        lb = fchz_lower_bound(s, C)
        for i in range(C.shape[0]):
            self.assertGreaterEqual(ub[i], lb[i],
                                                f"row {i}: UB={ub[i]} should be >= LB={lb[i]}")


if __name__ == '__main__':
    unittest.main()
