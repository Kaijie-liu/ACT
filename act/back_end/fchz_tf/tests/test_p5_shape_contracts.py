"""P5-1a: Shape contract tests for FchzTF nn4sys-style ops.

Per advisor 2026-06-08:
  Before fixing nn4sys, write tests that pin down the contract:
  - GATHER output flat dim = product of declared output_shape
  - DENSE after GATHER: state.c flat dim matches DENSE in_features
  - CONCAT multi-pred: state column counts aligned (no silent loss)
  - parallel pensieve: KeyError on preds → must fail-closed (no silent box-reset)
"""
import sys
sys.path.insert(0, '/data1/Kane/ACT')
import os
os.environ['HYZOR_FCHZ_USE_CUDA'] = '0'
os.environ.pop('HYZOR_FCHZ_G_MAX_COLS', None)

import unittest
import numpy as np
import torch


def _build_minimal_state(dim=4, K=3):
    """Build a minimal FCHZState for testing transfer functions."""
    from act.back_end.fchz_tf.representations import FCHZState
    c = np.arange(dim, dtype=np.float64) * 0.1
    G = np.eye(dim, K, dtype=np.float64) * 0.05
    tail = None
    return FCHZState(c=c, G=G, n_root=0, slack_records=[], tail_radius=tail)


def _build_runtime_with_state(dim=4, K=3):
    """Build minimal FchzTF runtime with a single pre-cached state."""
    from act.back_end.fchz_tf.fchz_tf import FchzTF
    from act.back_end.core import Bounds
    tf = FchzTF(G_max_cols=None)
    s = _build_minimal_state(dim, K)
    tf._state_cache[100] = s   # arbitrary L.id
    return tf, s


def _fake_layer(L_id, kind, params=None, preds=None):
    """Build a fake ACT Layer + register preds in a fake Net."""
    from act.back_end.core import Layer
    L = Layer(id=L_id, kind=kind, params=params or {}, in_vars=[], out_vars=[])
    return L


class FakeNet:
    def __init__(self, preds_map):
        self.preds = preds_map
        self.layers = []


def _bounds_from_state(state):
    from act.back_end.core import Bounds
    rad = np.abs(state.G).sum(axis=1) + (state.tail_radius if state.tail_radius is not None else 0)
    lb = state.c - rad
    ub = state.c + rad
    return Bounds(lb=torch.tensor(lb, dtype=torch.float32).reshape(1, -1),
                       ub=torch.tensor(ub, dtype=torch.float32).reshape(1, -1))


class TestGatherShapeContract(unittest.TestCase):
    """GATHER output flat dim must equal product of declared output_shape."""

    def test_gather_axis1_pick2_from_dim4(self):
        """Gather indices [0, 2] along axis 1 of (1, 4, 3) → output (1, 2, 3) flat 6."""
        from act.back_end.fchz_tf.fchz_tf import FchzTF
        from act.back_end.core import Bounds
        tf = FchzTF(G_max_cols=None)
        c = np.arange(12, dtype=np.float64).reshape(4, 3).flatten()
        G = np.zeros((12, 1), dtype=np.float64)
        from act.back_end.fchz_tf.representations import FCHZState
        s = FCHZState(c=c, G=G, n_root=0, slack_records=[], tail_radius=None)
        tf._state_cache[100] = s

        # GATHER axis=1 (dim 4) → output dim 2
        L = _fake_layer(101, 'GATHER', params={
            'indices': torch.tensor([0, 2]),
            'axis': 1,
            'input_shape': [1, 4, 3],
            'output_shape': [1, 2, 3],
        })
        tf._net = FakeNet({101: [100]})
        bounds_in = _bounds_from_state(s)
        fact = tf._tf_gather(L, bounds_in)
        s_out = tf._state_cache.get(101)
        self.assertIsNotNone(s_out)
        # Output flat dim = product(output_shape[1:]) = 2*3 = 6
        self.assertEqual(s_out.c.shape[0], 6,
                                f"GATHER flat dim {s_out.c.shape[0]} != product(output_shape[1:])=6")

    def test_gather_axis2_pick1_from_dim3(self):
        """Gather indices [1] along axis 2 of (1, 4, 3) → output (1, 4, 1) flat 4 (the typical nn4sys case)."""
        from act.back_end.fchz_tf.fchz_tf import FchzTF
        from act.back_end.fchz_tf.representations import FCHZState
        tf = FchzTF(G_max_cols=None)
        c = np.arange(12, dtype=np.float64)
        G = np.zeros((12, 1), dtype=np.float64)
        s = FCHZState(c=c, G=G, n_root=0, slack_records=[], tail_radius=None)
        tf._state_cache[100] = s
        L = _fake_layer(101, 'GATHER', params={
            'indices': torch.tensor([1]),
            'axis': 2,    # outermost data dim
            'input_shape': [1, 4, 3],
            'output_shape': [1, 4, 1],
        })
        tf._net = FakeNet({101: [100]})
        bounds_in = _bounds_from_state(s)
        tf._tf_gather(L, bounds_in)
        s_out = tf._state_cache.get(101)
        self.assertIsNotNone(s_out)
        # Output flat dim = product(output_shape[1:]) = 4*1 = 4
        self.assertEqual(s_out.c.shape[0], 4,
                                f"GATHER axis=2 pick1 flat dim {s_out.c.shape[0]} != 4")

    def test_gather_axis0_after_strip(self):
        """Gather indices [1, 3] along axis 0 of shape (4, 3) → output (2, 3) → flat 6."""
        from act.back_end.fchz_tf.fchz_tf import FchzTF
        from act.back_end.fchz_tf.representations import FCHZState
        tf = FchzTF(G_max_cols=None)
        c = np.arange(12, dtype=np.float64)
        G = np.zeros((12, 1), dtype=np.float64)
        s = FCHZState(c=c, G=G, n_root=0, slack_records=[], tail_radius=None)
        tf._state_cache[100] = s
        L = _fake_layer(101, 'GATHER', params={
            'indices': torch.tensor([1, 3]),
            'axis': 1,    # ACT convention: batch axis is 0, so axis=1 → strip to 0
            'input_shape': [1, 4, 3],
            'output_shape': [1, 2, 3],
        })
        tf._net = FakeNet({101: [100]})
        bounds_in = _bounds_from_state(s)
        tf._tf_gather(L, bounds_in)
        s_out = tf._state_cache.get(101)
        self.assertIsNotNone(s_out)
        self.assertEqual(s_out.c.shape[0], 6,
                                f"GATHER axis 1 (after strip 0) flat dim {s_out.c.shape[0]} != 6")


class TestDenseAfterGather(unittest.TestCase):
    """DENSE in_features must match GATHER output flat dim."""

    def test_dense_after_gather_shape_match(self):
        """DENSE in_features must match GATHER output flat dim. Build with correct sizes."""
        from act.back_end.fchz_tf.fchz_tf import FchzTF
        from act.back_end.fchz_tf.representations import FCHZState
        tf = FchzTF(G_max_cols=None)
        c = np.arange(12, dtype=np.float64)
        G = np.zeros((12, 1), dtype=np.float64)
        s = FCHZState(c=c, G=G, n_root=0, slack_records=[], tail_radius=None)
        tf._state_cache[100] = s
        # GATHER axis=2 picks 1 element → output flat 4
        L_gather = _fake_layer(101, 'GATHER', params={
            'indices': torch.tensor([1]),
            'axis': 2,
            'input_shape': [1, 4, 3],
            'output_shape': [1, 4, 1],
        })
        tf._net = FakeNet({101: [100], 102: [101]})
        tf._tf_gather(L_gather, _bounds_from_state(s))
        s_gather = tf._state_cache.get(101)
        self.assertEqual(s_gather.c.shape[0], 4)

        # DENSE in_features=4, out_features=3
        W = np.random.randn(3, 4).astype(np.float32)
        b = np.zeros(3, dtype=np.float32)
        L_dense = _fake_layer(102, 'DENSE', params={
            'weight': torch.tensor(W),
            'bias': torch.tensor(b),
            'in_features': 4,
            'out_features': 3,
            'input_shape': [1, 4],
            'output_shape': [1, 3],
        })
        tf._tf_dense(L_dense, _bounds_from_state(s_gather))
        s_dense = tf._state_cache.get(102)
        self.assertIsNotNone(s_dense)
        self.assertEqual(s_dense.c.shape[0], 3, "DENSE output must match out_features")

    def test_dense_after_gather_mismatch_detection(self):
        """If DENSE in_features != GATHER output flat dim, must raise (or fail-closed).
        This is the actual nn4sys/0 failure pattern."""
        from act.back_end.fchz_tf.fchz_tf import FchzTF
        from act.back_end.fchz_tf.representations import FCHZState
        tf = FchzTF(G_max_cols=None)
        c = np.arange(12, dtype=np.float64)
        G = np.zeros((12, 1), dtype=np.float64)
        s = FCHZState(c=c, G=G, n_root=0, slack_records=[], tail_radius=None)
        tf._state_cache[100] = s
        # GATHER axis=1 → output flat 6 (e.g. picking 2 of 4)
        L_gather = _fake_layer(101, 'GATHER', params={
            'indices': torch.tensor([0, 2]),
            'axis': 1,
            'input_shape': [1, 4, 3],
            'output_shape': [1, 2, 3],
        })
        tf._net = FakeNet({101: [100], 102: [101]})
        tf._tf_gather(L_gather, _bounds_from_state(s))
        s_gather = tf._state_cache.get(101)
        self.assertEqual(s_gather.c.shape[0], 6)
        # DENSE wrongly declares in_features=1 (this is the nn4sys/0 case)
        W = np.random.randn(3, 1).astype(np.float32)
        b = np.zeros(3, dtype=np.float32)
        L_dense = _fake_layer(102, 'DENSE', params={
            'weight': torch.tensor(W), 'bias': torch.tensor(b),
            'in_features': 1, 'out_features': 3,
            'input_shape': [1, 1], 'output_shape': [1, 3],
        })
        # Post P5-1a.3 harden: DENSE fails-closed with STATE_LOSS log entry
        # (used to raise ValueError; now records dense_input_dim_mismatch)
        tf._tf_dense(L_dense, _bounds_from_state(s_gather))
        loss_log = getattr(tf, '_state_loss_log', [])
        dense_violations = [e for e in loss_log
                                  if e.get('reason') == 'dense_input_dim_mismatch']
        self.assertEqual(len(dense_violations), 1,
                                f"Expected dense_input_dim_mismatch log, got {loss_log}")
        self.assertEqual(dense_violations[0]['W_in_features'], 1)
        self.assertEqual(dense_violations[0]['state_c_dim'], 6)
        # Must NOT have stored a wrong output state
        self.assertNotIn(102, tf._state_cache)


class TestStateLossTrace(unittest.TestCase):
    """STATE_LOSS trace: any fresh-box fallback must be logged, not silent."""

    def test_silent_fallback_is_recorded(self):
        """When pred state missing, _get_input_fchz must record STATE_LOSS."""
        from act.back_end.fchz_tf.fchz_tf import FchzTF
        from act.back_end.core import Bounds
        tf = FchzTF(G_max_cols=None)
        # Empty preds → fallback path
        tf._net = FakeNet({100: [99]})   # pred 99 not in cache
        L = _fake_layer(100, 'DENSE', params={'weight': torch.zeros(2, 4), 'bias': torch.zeros(2),
                                                              'in_features': 4, 'out_features': 2})
        bounds = Bounds(lb=torch.zeros(1, 4), ub=torch.ones(1, 4))
        tf._get_input_fchz(L, bounds)
        # STATE_LOSS log must record this
        self.assertTrue(hasattr(tf, '_state_loss_log'))
        self.assertGreaterEqual(len(tf._state_loss_log), 1)
        self.assertEqual(tf._state_loss_log[0]['layer_id'], 100)
        self.assertEqual(tf._state_loss_log[0]['reason'], 'pred_state_missing')


class TestKeyErrorFailClosed(unittest.TestCase):
    """Parallel pensieve KeyError: must not silently return cached/wrong state.
    Per advisor: 'if unsupported, must fail-closed UNKNOWN'.
    """

    def test_concat_multi_pred_NOT_resolved_via_helper(self):
        """CONCAT must NOT be resolved by _get_input_fchz's data-pred logic.
        Post P5-1a.1: only shape/index whitelist ops get data-pred resolution.
        Multi-input ops (CONCAT/ADD/MUL/etc.) must handle their preds explicitly.
        """
        from act.back_end.fchz_tf.fchz_tf import FchzTF
        from act.back_end.core import Bounds
        tf = FchzTF(G_max_cols=None)
        tf._net = FakeNet({200: [100, 101]})    # 2 preds
        L = _fake_layer(200, 'CONCAT', params={'concat_dim': 1})
        bounds = Bounds(lb=torch.zeros(1, 4), ub=torch.ones(1, 4))
        tf._get_input_fchz(L, bounds)
        self.assertTrue(hasattr(tf, '_state_loss_log'))
        self.assertGreaterEqual(len(tf._state_loss_log), 1)
        # CONCAT is NOT in whitelist → reason must be multi_pred_not_whitelisted
        self.assertEqual(tf._state_loss_log[0]['reason'], 'multi_pred_not_whitelisted',
                                f"CONCAT must not get data-pred resolution; got {tf._state_loss_log[0]}")

    def test_gather_IS_resolved_via_helper(self):
        """GATHER (whitelist) SHOULD get data-pred resolution when multi-pred."""
        from act.back_end.fchz_tf.fchz_tf import FchzTF
        from act.back_end.fchz_tf.representations import FCHZState
        from act.back_end.core import Bounds, Layer
        tf = FchzTF(G_max_cols=None)
        # Set up: data pred is layer 100 (cached), constant pred 99 (a CONSTANT layer)
        const_layer = Layer(id=99, kind='CONSTANT',
                                  params={'value': torch.zeros(1)},
                                  in_vars=[], out_vars=[])
        data_layer = Layer(id=100, kind='DENSE',
                                 params={'weight': torch.zeros(2, 2), 'bias': torch.zeros(2),
                                            'in_features': 2, 'out_features': 2},
                                 in_vars=[], out_vars=[])
        gather_layer = Layer(id=200, kind='GATHER',
                                    params={'indices': torch.tensor([0]),
                                              'axis': 0, 'input_shape': [1, 2]},
                                    in_vars=[], out_vars=[])

        class N: pass
        net = N()
        net.layers = [const_layer, data_layer, gather_layer]
        net.preds = {200: [99, 100]}
        tf._net = net
        # Cache a data state
        s = FCHZState(c=np.array([1.0, 2.0]), G=np.zeros((2, 0)),
                              n_root=0, slack_records=[], tail_radius=None)
        tf._state_cache[100] = s
        bounds = Bounds(lb=torch.zeros(1, 2), ub=torch.ones(1, 2))
        result = tf._get_input_fchz(gather_layer, bounds)
        # Should return the data pred's state
        np.testing.assert_array_equal(result.c, s.c)
        # Should be logged as 'multi_pred_resolved_to_data'
        self.assertEqual(tf._state_loss_log[-1]['reason'], 'multi_pred_resolved_to_data')


if __name__ == '__main__':
    unittest.main()
