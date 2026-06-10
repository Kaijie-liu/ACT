"""Unit tests for FchzTF — ACT-integrated FCHZ transfer function.

Tests cover:
  - representations.py: FCHZState invariants, compress_g_to_tail soundness
  - sigmoid_chord.py: analytical chord soundness on 200k fine grid
  - fchz_tf.py: per-layer dispatch + parity vs raw walker
"""

from __future__ import annotations
import sys
sys.path.insert(0, '/data1/Kane/ACT')

import numpy as np
import torch
import unittest

from act.back_end.transfer_functions import set_transfer_function
from act.back_end.fchz_tf import FchzTF
from act.back_end.fchz_tf.representations import (
    FCHZState, initial_state, hz_closed_form_ub,
    apply_dense, compress_g_to_tail,
)
from act.back_end.fchz_tf.sigmoid_chord import chord_params
from act.back_end.core import Layer, Bounds, Fact, Net, ConSet
from act.back_end.layer_schema import LayerKind


class TestFCHZStateInvariants(unittest.TestCase):
    """Invariant: R(state) bound by HZ closed-form."""

    def _sample_check(self, state, d, n_samples=5000, seed=0):
        rng = np.random.default_rng(seed)
        ub = hz_closed_form_ub(state, d)
        max_sampled = -float('inf')
        for _ in range(n_samples):
            xi = rng.uniform(-1, 1, size=state.K)
            y = state.c + state.G @ xi
            if state.tail_radius is not None:
                y = y + rng.uniform(-1, 1, size=state.n) * state.tail_radius
            v = float(d @ y)
            if v > max_sampled: max_sampled = v
        self.assertGreaterEqual(ub + 1e-9, max_sampled)

    def test_initial_state(self):
        c = np.array([0.0, 0.0]); r = np.array([1.0, 1.0])
        s = initial_state(c, r)
        self._sample_check(s, np.array([1.0, 1.0]))

    def test_dense_propagates(self):
        s = initial_state(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        W = np.array([[1.0, -2.0], [0.5, 1.0]])
        s2 = apply_dense(s, W, None)
        self._sample_check(s2, np.array([1.0, -1.0]))

    def test_dense_propagates_tail(self):
        c = np.array([0.0, 0.0]); G = np.zeros((2, 1))
        s = FCHZState(c=c, G=G, n_root=0, tail_radius=np.array([0.5, 0.3]))
        W = np.array([[1.0, -2.0], [0.5, 1.0]])
        s2 = apply_dense(s, W, None)
        np.testing.assert_allclose(s2.tail_radius, np.abs(W) @ np.array([0.5, 0.3]))


class TestCompressGToTail(unittest.TestCase):
    """Sparse-slack compression must be sound (UB grows or stays same)."""

    def test_compression_sound(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            n = 10; K = 20
            c = rng.standard_normal(n); G = rng.standard_normal((n, K))
            s = FCHZState(c=c, G=G, n_root=K)
            for K_max in [5, 10, 15]:
                s2 = compress_g_to_tail(s, K_max)
                self.assertEqual(s2.G.shape[1], min(K_max, K))
                d = rng.standard_normal(n)
                self.assertGreaterEqual(hz_closed_form_ub(s2, d) + 1e-9,
                                                  hz_closed_form_ub(s, d))


class TestSigmoidChordSoundness(unittest.TestCase):
    """Numerical sanity check for the analytical Sigmoid/Tanh chord.

    Soundness is ANALYTICAL (from critical-point derivation, see sigmoid_chord.py).
    This test is a regression guard that catches implementation bugs by checking
    that the 200k-sample max deviation does not exceed the analytically-derived
    radius. Sampling is NOT the soundness source.
    """

    def _check_one(self, kind, l, u):
        if kind == "Sigmoid":
            fn = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        else:
            fn = np.tanh
        alpha, beta, radius = chord_params(np.array([l]), np.array([u]), kind)
        alpha = float(alpha[0]); beta = float(beta[0]); radius = float(radius[0])
        xs = np.linspace(l, u, 200000)
        ys = fn(xs)
        chord_v = alpha * xs + beta
        max_abs_dev = np.max(np.abs(ys - chord_v))
        self.assertLessEqual(max_abs_dev, radius + 1e-12,
                                      f"{kind}[{l},{u}]: dev={max_abs_dev:.6e} > radius={radius:.6e}")

    def test_sigmoid_sound(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            l = rng.uniform(-5, 2); u = l + rng.uniform(0.01, 8)
            self._check_one("Sigmoid", l, u)

    def test_tanh_sound(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            l = rng.uniform(-3, 2); u = l + rng.uniform(0.01, 5)
            self._check_one("Tanh", l, u)


class TestFchzTFInterface(unittest.TestCase):
    """TransferFunction interface compliance."""

    def test_supports_layer(self):
        tf = FchzTF()
        for k in ['INPUT', 'INPUT_SPEC', 'ASSERT',
                     'DENSE', 'BIAS', 'SCALE',
                     'RELU', 'SIGMOID', 'TANH',
                     'CONV2D', 'BN', 'MAXPOOL2D']:
            self.assertTrue(tf.supports_layer(k), f"missing {k}")
        self.assertFalse(tf.supports_layer('UNKNOWN'))

    def test_name(self):
        self.assertEqual(FchzTF().name, "FchzTF")

    def test_mlp_pipeline(self):
        """End-to-end MLP via FchzTF."""
        tf = FchzTF()
        set_transfer_function(tf)
        in_l = Layer(id=0, kind='INPUT', params={'shape': (1, 2), 'dtype': 'torch.float64'},
                          in_vars=[], out_vars=[0, 1])
        spec = Layer(id=1, kind='INPUT_SPEC', params={'kind': 'BOX'},
                          in_vars=[0, 1], out_vars=[0, 1])
        W = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
        b = torch.tensor([0.1, -0.1], dtype=torch.float64)
        dense = Layer(id=2, kind='DENSE',
                            params={'weight': W, 'bias': b, 'in_features': 2, 'out_features': 2},
                            in_vars=[0, 1], out_vars=[2, 3])
        relu = Layer(id=3, kind='RELU', params={},
                          in_vars=[2, 3], out_vars=[4, 5])
        assert_l = Layer(id=4, kind='ASSERT',
                              params={'kind': 'LINEAR_LE',
                                          'C': torch.zeros(1, 2, dtype=torch.float64),
                                          'thresholds': torch.zeros(1, dtype=torch.float64), 'M': 1},
                              in_vars=[4, 5], out_vars=[4, 5])
        preds = {0:[], 1:[0], 2:[1], 3:[2], 4:[3]}
        succs = {0:[1], 1:[2], 2:[3], 3:[4], 4:[]}
        net = Net(layers=[in_l, spec, dense, relu, assert_l], preds=preds, succs=succs)
        input_bounds = Bounds(lb=torch.tensor([-1.0, -1.0], dtype=torch.float64),
                                      ub=torch.tensor([1.0, 1.0], dtype=torch.float64))
        before, after = {}, {}
        for L in net.layers:
            in_b = input_bounds if L.id == 0 else after[net.preds[L.id][0]].bounds
            after[L.id] = tf.apply(L, in_b, net, before, after)
        # After DENSE: bounds have shape (1, 2) (batch dim), row 0 = [-1.4, 1.6]
        d_bounds = after[2].bounds
        self.assertEqual(d_bounds.lb.shape, (1, 2))
        self.assertAlmostEqual(d_bounds.lb[0, 0].item(), -1.4, places=10)
        self.assertAlmostEqual(d_bounds.ub[0, 0].item(), 1.6, places=10)


class TestFchzTFParityVsRawWalker(unittest.TestCase):
    """FchzTF must give BIT-IDENTICAL results to raw walker."""

    def test_dense_relu_parity(self):
        """3-layer MLP: input → Dense → ReLU → Dense (raw vs FchzTF)."""
        from research.sc_hz.fc_hz_state import (
            FCHZState as OldS, initial_state as old_init,
            apply_dense as old_dense, hz_closed_form_ub as old_ub,
        )

        W1 = np.array([[1.0, -0.5, 0.3], [0.2, 1.0, -0.4],
                            [-0.3, 0.5, 0.8], [0.6, -0.2, 0.9]])
        b1 = np.array([0.1, -0.1, 0.2, 0.0])
        W2 = np.array([[0.5, -0.3, 0.2, 0.7], [0.1, 0.4, -0.6, 0.3]])
        b2 = np.array([0.0, 0.1])

        # OLD walker
        c = np.zeros(3); r = np.ones(3)
        s = old_init(c, r)
        s = old_dense(s, W1, b1)
        rad = np.abs(s.G).sum(axis=1) + (s.tail_radius if s.tail_radius is not None else 0)
        l = s.c - rad; u = s.c + rad
        is_active = l >= 0; is_inactive = u <= 0
        is_unstable = ~is_active & ~is_inactive
        den = np.where(is_unstable, u - l, 1.0)
        lam = np.where(is_unstable, u / np.maximum(den, 1e-300), 0.0)
        lam = np.where(is_active, 1.0, lam); lam = np.where(is_inactive, 0.0, lam)
        mu = np.where(is_unstable, -lam * l / 2.0, 0.0)
        new_c = lam * s.c + mu; new_G = s.G * lam[:, None]
        new_tail = (np.abs(mu) if np.any(is_unstable) else None)
        s = OldS(c=new_c, G=new_G, n_root=s.n_root, slack_records=s.slack_records,
                    tail_radius=new_tail)
        s = old_dense(s, W2, b2)
        old_lb = s.c - (np.abs(s.G).sum(axis=1) + (s.tail_radius if s.tail_radius is not None else 0))
        old_ub_v = s.c + (np.abs(s.G).sum(axis=1) + (s.tail_radius if s.tail_radius is not None else 0))

        # FchzTF (via ACT)
        tf = FchzTF()
        set_transfer_function(tf)
        in_l = Layer(id=0, kind='INPUT', params={'shape': (1, 3), 'dtype': 'torch.float64'},
                          in_vars=[], out_vars=[0, 1, 2])
        spec = Layer(id=1, kind='INPUT_SPEC', params={'kind': 'BOX'},
                          in_vars=[0, 1, 2], out_vars=[0, 1, 2])
        d1 = Layer(id=2, kind='DENSE',
                        params={'weight': torch.tensor(W1, dtype=torch.float64),
                                    'bias': torch.tensor(b1, dtype=torch.float64),
                                    'in_features': 3, 'out_features': 4},
                        in_vars=[0, 1, 2], out_vars=[3, 4, 5, 6])
        rl = Layer(id=3, kind='RELU', params={},
                        in_vars=[3, 4, 5, 6], out_vars=[7, 8, 9, 10])
        d2 = Layer(id=4, kind='DENSE',
                        params={'weight': torch.tensor(W2, dtype=torch.float64),
                                    'bias': torch.tensor(b2, dtype=torch.float64),
                                    'in_features': 4, 'out_features': 2},
                        in_vars=[7, 8, 9, 10], out_vars=[11, 12])
        ar = Layer(id=5, kind='ASSERT',
                        params={'kind': 'LINEAR_LE',
                                    'C': torch.zeros(1, 2, dtype=torch.float64),
                                    'thresholds': torch.zeros(1, dtype=torch.float64), 'M': 1},
                        in_vars=[11, 12], out_vars=[11, 12])
        preds = {0:[], 1:[0], 2:[1], 3:[2], 4:[3], 5:[4]}
        succs = {0:[1], 1:[2], 2:[3], 3:[4], 4:[5], 5:[]}
        net = Net(layers=[in_l, spec, d1, rl, d2, ar], preds=preds, succs=succs)
        input_bounds = Bounds(lb=torch.tensor([-1., -1., -1.], dtype=torch.float64),
                                      ub=torch.tensor([1., 1., 1.], dtype=torch.float64))
        before, after = {}, {}
        for L in net.layers:
            in_b = input_bounds if L.id == 0 else after[net.preds[L.id][0]].bounds
            after[L.id] = tf.apply(L, in_b, net, before, after)
        # Bounds now have batch dim (1, n_out); squeeze for comparison
        new_lb = after[4].bounds.lb.squeeze(0).numpy()
        new_ub = after[4].bounds.ub.squeeze(0).numpy()
        np.testing.assert_allclose(new_lb, old_lb, atol=1e-12)
        np.testing.assert_allclose(new_ub, old_ub_v, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
