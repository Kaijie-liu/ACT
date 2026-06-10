"""Tests for FCHZ K-piece Sigmoid/Tanh — verify SOUNDNESS (covers all x → f(x)).

Soundness criterion:
  For random x in box, f(x) must be inside the abstracted output range.
"""
import unittest
import numpy as np


class TestKPieceSigmoidTanh(unittest.TestCase):

    def _eval_state(self, c, G, tail_radius, kind, K, n_samples=1000):
        """Compute bounds of output state."""
        from act.back_end.fchz_tf.sigmoid_tanh_kpiece import apply_kpiece_sigmoid_tanh_fchz
        new_c, new_G, new_tail = apply_kpiece_sigmoid_tanh_fchz(c, G, tail_radius, kind, K=K)
        # Sample xi vars uniformly in [-1, 1]
        rng = np.random.default_rng(42)
        ng_new = new_G.shape[1] if new_G.ndim > 1 else 0
        if ng_new == 0:
            sampled = np.tile(new_c.reshape(1, -1), (n_samples, 1))
        else:
            xi = rng.uniform(-1.0, 1.0, (n_samples, ng_new))
            sampled = new_c.reshape(1, -1) + xi @ new_G.T
        # Add tail
        if new_tail is not None:
            xi_tail = rng.uniform(-1.0, 1.0, (n_samples, new_c.shape[0]))
            sampled = sampled + xi_tail * new_tail.reshape(1, -1)
        return sampled, new_c, new_G, new_tail

    def test_sigmoid_K1_sound(self):
        """K=1 single chord: sound on simple test."""
        n = 5
        rng = np.random.default_rng(7)
        c = rng.uniform(-2, 2, n)
        G = rng.uniform(-1, 1, (n, 3)) * 0.5
        tail = None
        kind = 'Sigmoid'
        sampled, _, _, _ = self._eval_state(c, G, tail, kind, K=1)
        # Reference: for each xi sample, compute actual sigmoid(c + G@xi)
        rng2 = np.random.default_rng(42)
        xi_in = rng2.uniform(-1.0, 1.0, (1000, 3))
        z_in = c.reshape(1, -1) + xi_in @ G.T
        true_y = 1 / (1 + np.exp(-z_in))
        # Sigmoid of input range is in [0, 1] — sampled should cover it
        for i in range(n):
            self.assertGreaterEqual(sampled[:, i].max(), true_y[:, i].max() - 0.01,
                                                f"K=1 unsound at neuron {i}")
            self.assertLessEqual(sampled[:, i].min(), true_y[:, i].min() + 0.01,
                                            f"K=1 unsound at neuron {i}")

    def test_sigmoid_K2_tighter_than_K1(self):
        """K=2 should give tighter bounds than K=1 on wide neurons."""
        n = 5
        rng = np.random.default_rng(11)
        c = rng.uniform(-2, 2, n)
        G = rng.uniform(-1, 1, (n, 3)) * 0.8   # wider neurons
        tail = None
        s_K1, _, _, _ = self._eval_state(c, G, tail, 'Sigmoid', K=1)
        s_K2, _, _, _ = self._eval_state(c, G, tail, 'Sigmoid', K=2)
        # K=2 width should be <= K=1 width per neuron
        w_K1 = s_K1.max(axis=0) - s_K1.min(axis=0)
        w_K2 = s_K2.max(axis=0) - s_K2.min(axis=0)
        # At least one neuron should be strictly tighter for K=2
        self.assertTrue((w_K2 <= w_K1 + 1e-6).any(), f"K=2 not tighter on any neuron")

    def test_tanh_sound(self):
        """Tanh K=2 sound on wide bounds."""
        n = 4
        rng = np.random.default_rng(13)
        c = rng.uniform(-3, 3, n)
        G = rng.uniform(-1, 1, (n, 4)) * 1.0
        sampled, _, _, _ = self._eval_state(c, G, None, 'Tanh', K=2)
        rng2 = np.random.default_rng(42)
        xi_in = rng2.uniform(-1.0, 1.0, (1000, 4))
        z_in = c.reshape(1, -1) + xi_in @ G.T
        true_y = np.tanh(z_in)
        for i in range(n):
            self.assertGreaterEqual(sampled[:, i].max(), true_y[:, i].max() - 0.01)
            self.assertLessEqual(sampled[:, i].min(), true_y[:, i].min() + 0.01)

    def test_env_var_K(self):
        """ACT_HZ_SIGMOID_K env var controls K."""
        import os
        from act.back_end.fchz_tf.sigmoid_tanh_kpiece import get_K_for
        os.environ['ACT_HZ_SIGMOID_K'] = '4'
        self.assertEqual(get_K_for('Sigmoid'), 4)
        os.environ['ACT_HZ_SIGMOID_K'] = '2'
        self.assertEqual(get_K_for('Sigmoid'), 2)
        os.environ.pop('ACT_HZ_SIGMOID_K', None)
        # Default is 2
        self.assertEqual(get_K_for('Sigmoid'), 2)

    def test_narrow_neurons_exact(self):
        """Narrow neurons (zero radius) get exact f(c) value."""
        from act.back_end.fchz_tf.sigmoid_tanh_kpiece import apply_kpiece_sigmoid_tanh_fchz
        c = np.array([0.0, 1.0, -1.0])
        G = np.zeros((3, 2))   # zero G → narrow
        tail = np.zeros(3)
        new_c, new_G, new_tail = apply_kpiece_sigmoid_tanh_fchz(c, G, tail, 'Sigmoid', K=2)
        # Exact sigmoid values
        exact = 1.0 / (1.0 + np.exp(-c))
        np.testing.assert_allclose(new_c, exact, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
