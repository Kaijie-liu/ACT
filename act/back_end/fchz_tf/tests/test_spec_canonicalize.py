"""Unit tests for spec_canonicalize (advisor M2c-fix)."""
import sys
sys.path.insert(0, '/data1/Kane/ACT')

import numpy as np
import unittest
import torch

from act.back_end.fchz_tf.spec_canonicalize import (
    canonicalize_queries, canon_verdict, _try_infer_top1, _build_top1_C,
)
from act.back_end.fchz_tf.representations import FCHZState
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind


def make_top1_unsafe_query(y_true: int, rival: int, n_out: int, convention='ACT'):
    """Build a single-row UNSAFE_LINEAR query for one rival.

    ACT convention: y_true - y_rival <= 0 → row[true]=+1, row[rival]=-1
    Research convention: y_rival - y_true >= 0 → row[rival]=+1, row[true]=-1
    """
    C = np.zeros((1, n_out), dtype=np.float32)
    if convention == 'ACT':
        C[0, y_true] = 1.0
        C[0, rival] = -1.0
    else:
        C[0, rival] = 1.0
        C[0, y_true] = -1.0
    in_spec = InputSpec(kind=InKind.LINF_BALL, center=torch.zeros(3), eps=0.1)
    out_spec = OutputSpec(
        kind=OutKind.UNSAFE_LINEAR,
        c=torch.tensor(C, dtype=torch.float32),
        d=torch.tensor([0.0], dtype=torch.float32))
    return (in_spec, out_spec)


class TestInferTop1(unittest.TestCase):
    def test_act_convention(self):
        # n=4, true=2, rivals 0,1,3
        n_out = 4
        rows = []
        for rival in [0, 1, 3]:
            r = np.zeros(n_out)
            r[2] = 1.0; r[rival] = -1.0
            rows.append(r)
        C = np.stack(rows)
        t = np.zeros(3)
        y_t = _try_infer_top1(C, t, n_out)
        self.assertEqual(y_t, 2)

    def test_research_convention(self):
        n_out = 4
        rows = []
        for rival in [0, 1, 3]:
            r = np.zeros(n_out)
            r[rival] = 1.0; r[2] = -1.0  # opposite
            rows.append(r)
        C = np.stack(rows)
        t = np.zeros(3)
        y_t = _try_infer_top1(C, t, n_out)
        self.assertEqual(y_t, 2)

    def test_missing_rival_no_top1(self):
        # Missing rival 1 → not all rivals covered
        n_out = 4
        rows = []
        for rival in [0, 3]:  # only 2 of 3 rivals
            r = np.zeros(n_out)
            r[2] = 1.0; r[rival] = -1.0
            rows.append(r)
        C = np.stack(rows)
        t = np.zeros(2)
        y_t = _try_infer_top1(C, t, n_out)
        # Wrong row count (2 vs n_out-1=3)
        self.assertIsNone(y_t)

    def test_non_zero_threshold_no_top1(self):
        n_out = 4
        rows = []
        for rival in [0, 1, 3]:
            r = np.zeros(n_out)
            r[2] = 1.0; r[rival] = -1.0
            rows.append(r)
        C = np.stack(rows)
        t = np.array([0.5, 0.0, 0.0])  # non-zero
        y_t = _try_infer_top1(C, t, n_out)
        self.assertIsNone(y_t)


class TestBinaryYTruePreservation(unittest.TestCase):
    """For n_out=2 (binary), y_true inference is ambiguous from a single row.
    Must preserve parser's y_true (advisor M3 fix 2026-06-08).
    """

    def test_binary_top1_preserves_y_true_0(self):
        """ACT TOP1_ROBUST y_true=0 must yield canon y_true=0 (not 1)."""
        in_spec = InputSpec(kind=InKind.LINF_BALL, center=torch.zeros(30), eps=0.1)
        out_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0]))
        queries = [(in_spec, out_spec)]
        canon = canonicalize_queries(queries, n_out=2)
        self.assertEqual(canon['kind'], 'TOP1_ROBUST')
        self.assertEqual(canon['y_true'], 0)   # preserved, not re-inferred to 1
        # C row should be e_rival - e_true = e_1 - e_0 = [-1, 1]
        np.testing.assert_allclose(canon['C'], [[-1.0, 1.0]])
        np.testing.assert_allclose(canon['t'], [0.0])

    def test_binary_top1_preserves_y_true_1(self):
        """y_true=1 must yield canon y_true=1 (not re-inferred to 0)."""
        in_spec = InputSpec(kind=InKind.LINF_BALL, center=torch.zeros(30), eps=0.1)
        out_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([1]))
        queries = [(in_spec, out_spec)]
        canon = canonicalize_queries(queries, n_out=2)
        self.assertEqual(canon['y_true'], 1)
        # C row should be e_rival - e_true = e_0 - e_1 = [1, -1]
        np.testing.assert_allclose(canon['C'], [[1.0, -1.0]])


class TestCanonicalizeQueries(unittest.TestCase):
    def test_top1_aggregation_from_99_queries(self):
        """The cifar/tiny case: K-1 single-row UNSAFE_LINEAR queries → 1 TOP1_ROBUST."""
        n_out = 100
        y_true = 42
        queries = []
        for rival in range(n_out):
            if rival == y_true: continue
            queries.append(make_top1_unsafe_query(y_true, rival, n_out, convention='ACT'))
        canon = canonicalize_queries(queries, n_out)
        self.assertEqual(canon['kind'], 'TOP1_ROBUST')
        self.assertEqual(canon['y_true'], y_true)
        self.assertEqual(canon['n_input_queries'], n_out - 1)
        self.assertTrue(canon['canonicalized'])
        self.assertEqual(canon['C'].shape, (n_out - 1, n_out))

    def test_top1_research_convention(self):
        """research-style direction → still inferred as TOP1."""
        n_out = 10
        y_true = 3
        queries = []
        for rival in range(n_out):
            if rival == y_true: continue
            queries.append(make_top1_unsafe_query(y_true, rival, n_out, convention='research'))
        canon = canonicalize_queries(queries, n_out)
        self.assertEqual(canon['kind'], 'TOP1_ROBUST')
        self.assertEqual(canon['y_true'], y_true)

    def test_arbitrary_unsafe_linear_NOT_aggregated(self):
        """SOUNDNESS gate (advisor 2026-06-08): arbitrary multi-query UNSAFE_LINEAR
        must NOT aggregate into single matrix (would be unsound).
        Should preserve as MULTI_QUERY for per-query AND evaluation.
        """
        n_out = 5
        queries = []
        # Row 0: y_0 + y_1 <= 1
        C0 = np.zeros((1, n_out), dtype=np.float32); C0[0, 0] = 1.0; C0[0, 1] = 1.0
        # Row 1: y_2 + y_3 <= 2
        C1 = np.zeros((1, n_out), dtype=np.float32); C1[0, 2] = 1.0; C1[0, 3] = 1.0
        in_spec = InputSpec(kind=InKind.LINF_BALL, center=torch.zeros(3), eps=0.1)
        queries.append((in_spec, OutputSpec(kind=OutKind.UNSAFE_LINEAR,
                                                          c=torch.tensor(C0), d=torch.tensor([1.0]))))
        queries.append((in_spec, OutputSpec(kind=OutKind.UNSAFE_LINEAR,
                                                          c=torch.tensor(C1), d=torch.tensor([2.0]))))
        canon = canonicalize_queries(queries, n_out)
        # MUST be MULTI_QUERY (preserves queries list), NOT aggregated UNSAFE_LINEAR
        self.assertEqual(canon['kind'], 'MULTI_QUERY')
        self.assertFalse(canon['canonicalized'])
        self.assertIsNone(canon['C'])   # no aggregation
        self.assertEqual(len(canon['queries']), 2)   # queries preserved
        self.assertIn('per-query AND', canon['reason'])

    def test_single_unsafe_linear_query_passes_through(self):
        """Single UNSAFE_LINEAR query (already conjunction-style) → keep as-is."""
        n_out = 5
        in_spec = InputSpec(kind=InKind.LINF_BALL, center=torch.zeros(3), eps=0.1)
        C = np.zeros((2, n_out), dtype=np.float32)
        C[0, 0] = 1.0; C[0, 1] = -1.0  # y_0 - y_1
        C[1, 2] = 1.0; C[1, 3] = -1.0
        queries = [(in_spec, OutputSpec(kind=OutKind.UNSAFE_LINEAR,
                                                          c=torch.tensor(C),
                                                          d=torch.tensor([0.0, 0.0])))]
        canon = canonicalize_queries(queries, n_out)
        self.assertEqual(canon['kind'], 'UNSAFE_LINEAR')
        self.assertTrue(canon['canonicalized'])
        self.assertEqual(canon['n_input_queries'], 1)


class TestCanonVerdict(unittest.TestCase):
    def test_top1_safe_via_canon(self):
        """Canonicalized TOP1_ROBUST on a safe state → CERT."""
        n_out = 3
        y_true = 1
        # State: c = (1, 5, 2), no noise → y_true=1 wins (5 > 1, 5 > 2)
        state = FCHZState(
            c=np.array([1.0, 5.0, 2.0]),
            G=np.zeros((3, 0), dtype=np.float64),
            n_root=0, slack_records=[], tail_radius=None)
        queries = []
        for rival in [0, 2]:
            queries.append(make_top1_unsafe_query(y_true, rival, n_out, convention='ACT'))
        canon = canonicalize_queries(queries, n_out)
        self.assertEqual(canon['kind'], 'TOP1_ROBUST')
        self.assertEqual(canon['y_true'], y_true)
        v = canon_verdict(canon, state)
        self.assertEqual(v, 'CERTIFIED')

    def test_top1_unsafe_via_canon(self):
        """Canonicalized TOP1 on unsafe state → UNKNOWN."""
        n_out = 3
        y_true = 1
        # State: c = (1, 2, 5), true=1 LOSES to class 2 (5 > 2)
        state = FCHZState(
            c=np.array([1.0, 2.0, 5.0]),
            G=np.zeros((3, 0), dtype=np.float64),
            n_root=0, slack_records=[], tail_radius=None)
        queries = []
        for rival in [0, 2]:
            queries.append(make_top1_unsafe_query(y_true, rival, n_out, convention='ACT'))
        canon = canonicalize_queries(queries, n_out)
        v = canon_verdict(canon, state)
        self.assertEqual(v, 'UNKNOWN')

    def test_multi_query_safe_via_per_query_AND(self):
        """MULTI_QUERY: each query evaluated independently, AND result."""
        n_out = 5
        # State: c = (10, 0, 0, 0, 0), no noise
        # Query 1 (UNSAFE_LINEAR): y_0 + y_1 <= 1 → y_0=10, y_1=0, sum=10 > 1 → SAFE
        # Query 2 (UNSAFE_LINEAR): y_2 + y_3 <= 2 → 0+0=0 ≤ 2 → UNSAFE
        # AND: any UNSAFE → overall UNKNOWN
        state = FCHZState(
            c=np.array([10.0, 0.0, 0.0, 0.0, 0.0]),
            G=np.zeros((5, 0), dtype=np.float64),
            n_root=0, slack_records=[], tail_radius=None)
        in_spec = InputSpec(kind=InKind.LINF_BALL, center=torch.zeros(3), eps=0.1)
        C0 = np.zeros((1, n_out), dtype=np.float32); C0[0, 0] = 1.0; C0[0, 1] = 1.0
        C1 = np.zeros((1, n_out), dtype=np.float32); C1[0, 2] = 1.0; C1[0, 3] = 1.0
        queries = [
            (in_spec, OutputSpec(kind=OutKind.UNSAFE_LINEAR,
                                          c=torch.tensor(C0), d=torch.tensor([1.0]))),
            (in_spec, OutputSpec(kind=OutKind.UNSAFE_LINEAR,
                                          c=torch.tensor(C1), d=torch.tensor([2.0]))),
        ]
        canon = canonicalize_queries(queries, n_out)
        self.assertEqual(canon['kind'], 'MULTI_QUERY')
        v = canon_verdict(canon, state)
        # Query 2 fails → UNKNOWN
        self.assertEqual(v, 'UNKNOWN')

    def test_multi_query_all_safe(self):
        """MULTI_QUERY: all queries safe → CERT."""
        n_out = 5
        # State: c = (10, 10, 10, 10, 10)
        # Query 1: y_0 + y_1 <= 1 → 20 > 1 → SAFE
        # Query 2: y_2 + y_3 <= 2 → 20 > 2 → SAFE
        state = FCHZState(
            c=np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
            G=np.zeros((5, 0), dtype=np.float64),
            n_root=0, slack_records=[], tail_radius=None)
        in_spec = InputSpec(kind=InKind.LINF_BALL, center=torch.zeros(3), eps=0.1)
        C0 = np.zeros((1, n_out), dtype=np.float32); C0[0, 0] = 1.0; C0[0, 1] = 1.0
        C1 = np.zeros((1, n_out), dtype=np.float32); C1[0, 2] = 1.0; C1[0, 3] = 1.0
        queries = [
            (in_spec, OutputSpec(kind=OutKind.UNSAFE_LINEAR,
                                          c=torch.tensor(C0), d=torch.tensor([1.0]))),
            (in_spec, OutputSpec(kind=OutKind.UNSAFE_LINEAR,
                                          c=torch.tensor(C1), d=torch.tensor([2.0]))),
        ]
        canon = canonicalize_queries(queries, n_out)
        self.assertEqual(canon['kind'], 'MULTI_QUERY')
        v = canon_verdict(canon, state)
        self.assertEqual(v, 'CERTIFIED')


if __name__ == '__main__':
    unittest.main()
