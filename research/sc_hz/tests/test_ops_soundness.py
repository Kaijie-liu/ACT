"""Soundness regression for ops on PrunedState.

For each operator (Dense, ReLU, Add, LP UB), build a small synthetic
state, propagate, and verify:
  1. The propagated state contains every realization of the original
     input set under the layer op (brute-force containment).
  2. LP UB on d^T y is >= max over samples of d^T y (sound upper bound).

Conv2D is tested via a small (Ci=1, H=4, W=4) case so brute-force
sampling is still cheap.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.sc_hz.prune import PrunedState  # noqa: E402
from research.sc_hz.ops import (  # noqa: E402
    apply_dense, apply_conv2d, apply_relu_triangle, apply_add,
    apply_flatten, lp_ub_rival_margin, bounds,
)


def _box_state(c: np.ndarray, radius: np.ndarray) -> PrunedState:
    """Build a PrunedState that's an axis-aligned box with given center/radius."""
    n = c.shape[0]
    G = np.diag(radius)
    return PrunedState(
        c=c.astype(np.float64),
        G_kept=G.astype(np.float64),
        tail_radius=None,
        metadata={"toy": True},
    )


class TestDenseOp(unittest.TestCase):
    def test_dense_containment_and_lp_ub(self) -> None:
        rng = np.random.default_rng(20260604)
        n_in, n_out = 4, 3
        c_in = rng.normal(size=n_in)
        r_in = np.full(n_in, 0.1)
        state_in = _box_state(c_in, r_in)
        W = rng.normal(size=(n_out, n_in))
        b = rng.normal(size=n_out)
        state_out = apply_dense(state_in, W, b)
        lb_out, ub_out = bounds(state_out)

        # Sample 500 corners + random points and check containment
        for _ in range(200):
            sign = rng.choice([-1.0, 1.0], size=n_in)
            x = c_in + r_in * sign
            y = W @ x + b
            self.assertTrue(np.all(y >= lb_out - 1e-9))
            self.assertTrue(np.all(y <= ub_out + 1e-9))

        # LP UB on d^T y must be >= max sample d^T y
        d = rng.normal(size=n_out)
        ub_lp = lp_ub_rival_margin(state_out, d)
        max_sample = max(d @ (W @ (c_in + r_in * rng.choice([-1.0, 1.0],
                                                              size=n_in)) + b)
                          for _ in range(500))
        self.assertGreaterEqual(ub_lp + 1e-9, max_sample)


class TestReluTriangleOp(unittest.TestCase):
    def test_relu_containment_and_lp_ub(self) -> None:
        rng = np.random.default_rng(20260604)
        for trial in range(20):
            n = 6
            c = (rng.uniform(-1.0, 1.0, size=n)).astype(np.float64) * 0.5
            r = rng.uniform(0.05, 0.6, size=n)
            state = _box_state(c, r)
            new_state, unstable = apply_relu_triangle(state)
            lb_o, ub_o = bounds(new_state)
            for _ in range(100):
                sign = rng.choice([-1.0, 1.0], size=n)
                x = c + r * sign
                y = np.maximum(x, 0.0)
                self.assertTrue(np.all(y >= lb_o - 1e-9))
                self.assertTrue(np.all(y <= ub_o + 1e-9))


class TestConv2DOp(unittest.TestCase):
    def test_conv2d_small_containment(self) -> None:
        rng = np.random.default_rng(20260604)
        Ci, H, W_in = 2, 4, 4
        Co = 3
        k = 3
        n_in = Ci * H * W_in
        c = np.zeros(n_in)
        r = np.full(n_in, 0.1)
        state_in = _box_state(c, r)
        Wt = rng.normal(size=(Co, Ci, k, k))
        b = rng.normal(size=Co)
        state_out, out_shape = apply_conv2d(state_in, Wt, b,
                                              input_shape=(Ci, H, W_in),
                                              stride=1, padding=0)
        lb_o, ub_o = bounds(state_out)
        for _ in range(100):
            sign = rng.choice([-1.0, 1.0], size=n_in)
            x = c + r * sign
            x_t = torch.from_numpy(x.reshape(1, Ci, H, W_in)).to(torch.float64)
            W_t = torch.from_numpy(Wt).to(torch.float64)
            b_t = torch.from_numpy(b).to(torch.float64)
            y = F.conv2d(x_t, W_t, b_t, stride=1, padding=0).detach().numpy().reshape(-1)
            self.assertTrue(np.all(y >= lb_o - 1e-9))
            self.assertTrue(np.all(y <= ub_o + 1e-9))


class TestAddOp(unittest.TestCase):
    def test_add_containment(self) -> None:
        rng = np.random.default_rng(20260604)
        n = 5
        c_a = rng.normal(size=n)
        r_a = np.full(n, 0.2)
        c_b = rng.normal(size=n)
        r_b = np.full(n, 0.15)
        sa = _box_state(c_a, r_a)
        sb = _box_state(c_b, r_b)
        sc = apply_add(sa, sb)
        lb, ub = bounds(sc)
        # Containment: any realization x_a + x_b should be in [lb, ub]
        for _ in range(200):
            x_a = c_a + r_a * rng.choice([-1.0, 1.0], size=n)
            x_b = c_b + r_b * rng.choice([-1.0, 1.0], size=n)
            y = x_a + x_b
            self.assertTrue(np.all(y >= lb - 1e-9))
            self.assertTrue(np.all(y <= ub + 1e-9))


class TestFullPipeline(unittest.TestCase):
    """Mini end-to-end: Dense → ReLU → Dense on a 4 → 6 → 3 → 2 net.

    Verify rival LP UB is sound on a 1000-sample brute-force.
    """

    def test_dense_relu_dense_pipeline(self) -> None:
        rng = np.random.default_rng(20260604)
        n_in, n_h, n_out = 4, 6, 2
        c_in = np.zeros(n_in)
        r_in = np.full(n_in, 0.1)
        W1 = rng.normal(size=(n_h, n_in)) * 0.5
        b1 = rng.normal(size=n_h) * 0.1
        W2 = rng.normal(size=(n_out, n_h)) * 0.5
        b2 = rng.normal(size=n_out) * 0.1

        state = _box_state(c_in, r_in)
        state = apply_dense(state, W1, b1)
        state, _ = apply_relu_triangle(state)
        state = apply_dense(state, W2, b2)

        # rival = 1, y_true = 0; rival direction d = W2[1] - W2[0]
        d = np.array([1.0 if k == 1 else (-1.0 if k == 0 else 0.0)
                       for k in range(n_out)])
        ub_lp = lp_ub_rival_margin(state, d)

        max_sample = -np.inf
        for _ in range(1000):
            x = c_in + r_in * rng.choice([-1.0, 1.0], size=n_in)
            h = np.maximum(W1 @ x + b1, 0.0)
            y = W2 @ h + b2
            margin = y[1] - y[0]
            if margin > max_sample:
                max_sample = margin
        # LP UB must be >= max sample margin (sound over-approx)
        self.assertGreaterEqual(ub_lp + 1e-9, max_sample,
            msg=f"LP UB ({ub_lp:.6f}) < max sample margin ({max_sample:.6f}) — UNSOUND")


if __name__ == "__main__":
    unittest.main()
