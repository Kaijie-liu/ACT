"""d_L^r chain correctness on a synthetic 3-Dense-layer net.

Per dc_hz_phase_a_plan.md §1.1 and INNOVATION_BRIEF §10:

    d_N^r  =  W_{N+1}[r, :]  -  W_{N+1}[y_t, :]    (in h_N space)
    d_L^r  =  W_{L+1}^T · d_{L+1}^r                 for L = N-1, ..., 0

So for a 3-weight net [W_1, W_2, W_3] (W_3 is the classifier, mapping
the last hidden dim to the n_classes output):
  - The chain has 3 entries [d_0, d_1, d_2] where:
      d_2 = W_3[rival] - W_3[y_true]     (h_2 space, dim = in_dim(W_3))
      d_1 = W_2^T @ d_2                  (h_1 space, dim = in_dim(W_2))
      d_0 = W_1^T @ d_1                  (input space, dim = in_dim(W_1))

This convention matches the §10 spec: there is one d_L per hidden
representation, indexed 0..N for N+1 hidden layers (input through
last hidden before the classifier).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.sc_hz.precompute_direction import precompute_d_per_layer  # noqa: E402


class TestDirectionChain(unittest.TestCase):
    """The d_L chain must equal W_{L+1}^T · d_{L+1} layer by layer."""

    def test_dchain_on_3weight_dense_net(self) -> None:
        rng = np.random.default_rng(20260604)
        # Net: x ∈ R^4 -- W1 --> h_1 ∈ R^6 -- W2 --> h_2 ∈ R^3 -- W3 --> y ∈ R^2
        W1 = rng.normal(size=(6, 4))
        W2 = rng.normal(size=(3, 6))
        W3 = rng.normal(size=(2, 3))   # classifier
        weights = [W1, W2, W3]
        y_true, rival = 0, 1

        d_per_layer = precompute_d_per_layer(weights, rival, y_true)

        # Expect 3 entries: [d_0 (input space), d_1 (h_1 space), d_2 (h_2 space)]
        self.assertEqual(len(d_per_layer), 3,
                          f"expected 3 d_L vectors, got {len(d_per_layer)}")

        # d_2 = W_3[rival] - W_3[y_true] (shape 3 = in_dim of W_3 = h_2 dim)
        np.testing.assert_allclose(
            d_per_layer[2], W3[rival] - W3[y_true],
            err_msg="d_2 must equal W_3[rival, :] - W_3[y_true, :]",
        )

        # d_1 = W_2^T @ d_2 (shape 6 = in_dim of W_2 = h_1 dim)
        np.testing.assert_allclose(
            d_per_layer[1], W2.T @ d_per_layer[2],
            err_msg="d_1 must equal W_2^T · d_2",
        )

        # d_0 = W_1^T @ d_1 (shape 4 = in_dim of W_1 = input dim)
        np.testing.assert_allclose(
            d_per_layer[0], W1.T @ d_per_layer[1],
            err_msg="d_0 must equal W_1^T · d_1",
        )

    def test_shape_per_layer(self) -> None:
        """Each d_L should have the shape of layer L's input dimension."""
        rng = np.random.default_rng(20260610)
        W1 = rng.normal(size=(6, 4))
        W2 = rng.normal(size=(3, 6))
        W3 = rng.normal(size=(2, 3))
        weights = [W1, W2, W3]
        d_per_layer = precompute_d_per_layer(weights, rival=1, y_true=0)
        self.assertEqual(d_per_layer[0].shape, (4,))   # input dim
        self.assertEqual(d_per_layer[1].shape, (6,))   # h_1 dim
        self.assertEqual(d_per_layer[2].shape, (3,))   # h_2 dim

    def test_d_does_not_depend_on_input_or_bounds(self) -> None:
        """Soundness-critical: d_L^r must depend ONLY on weights + rival/y_true."""
        rng = np.random.default_rng(20260608)
        W1 = rng.normal(size=(6, 4))
        W2 = rng.normal(size=(3, 6))
        W3 = rng.normal(size=(2, 3))
        d_a = precompute_d_per_layer([W1, W2, W3], rival=1, y_true=0)
        d_b = precompute_d_per_layer([W1, W2, W3], rival=1, y_true=0)
        for da, db in zip(d_a, d_b):
            np.testing.assert_array_equal(da, db,
                err_msg="precompute_d_per_layer must be deterministic")


if __name__ == "__main__":
    unittest.main()
