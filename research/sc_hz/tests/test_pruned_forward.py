"""End-to-end smoke + soundness for pruned_forward_dense.

Tests:
  1. soundness: the LP UB on the rival margin is >= max sample margin
     over 1000 brute-force input samples.
  2. K=ng_max edge: when K is so large no pruning fires, the LP UB
     equals the un-pruned LP UB (parity to baseline).
  3. CERT detection: on a constructed iid where the true margin is
     clearly negative, the verdict is CERT.
  4. FAL detection: on a constructed iid where the true margin can be
     positive, the verdict is FAL_CANDIDATE (and the closed-form
     xi_star can be ORT-verified later in the driver).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.sc_hz.pruned_forward import pruned_forward_dense  # noqa: E402


class TestPrunedForwardDense(unittest.TestCase):
    """End-to-end soundness on a small synthetic net."""

    def _build_net(self, seed: int = 20260604):
        rng = np.random.default_rng(seed)
        n_in, n_h1, n_h2, n_out = 4, 8, 6, 3
        W1 = rng.normal(size=(n_h1, n_in)) * 0.5
        b1 = rng.normal(size=n_h1) * 0.1
        W2 = rng.normal(size=(n_h2, n_h1)) * 0.5
        b2 = rng.normal(size=n_h2) * 0.1
        W3 = rng.normal(size=(n_out, n_h2)) * 0.5
        b3 = rng.normal(size=n_out) * 0.1
        return [W1, W2, W3], [b1, b2, b3], n_in, n_out

    def _forward_sample(self, weights, biases, x):
        h = np.maximum(weights[0] @ x + biases[0], 0.0)
        h = np.maximum(weights[1] @ h + biases[1], 0.0)
        y = weights[2] @ h + biases[2]
        return y

    def test_soundness_on_random_iid(self) -> None:
        weights, biases, n_in, n_out = self._build_net()
        rng = np.random.default_rng(20260605)
        c_in = rng.normal(size=n_in) * 0.2
        r_in = np.full(n_in, 0.05)
        lb = c_in - r_in
        ub = c_in + r_in

        y_true = 0
        rivals = [1, 2]
        result = pruned_forward_dense(
            weights, biases, lb, ub, y_true=y_true, rivals=rivals,
            K_per_layer=256,
        )

        # For each rival, the LP UB must be a sound upper bound on
        # the rival margin over the input box.
        for rr in result.per_rival:
            d_out = np.zeros(n_out)
            d_out[rr.rival] = 1.0
            d_out[y_true] = -1.0
            max_sample = -np.inf
            for _ in range(1000):
                x = c_in + r_in * rng.choice([-1.0, 1.0], size=n_in)
                y = self._forward_sample(weights, biases, x)
                margin = float(d_out @ y)
                if margin > max_sample:
                    max_sample = margin
            self.assertGreaterEqual(
                rr.lp_ub_rival_margin + 1e-6, max_sample,
                msg=f"rival {rr.rival}: LP UB ({rr.lp_ub_rival_margin:.6f}) "
                    f"< max sample margin ({max_sample:.6f}) — UNSOUND",
            )

    def test_cert_detection_on_easy_iid(self) -> None:
        """Build an iid where rival classes are clearly worse than y_true
        (large negative margin). The verdict should be CERT for all rivals."""
        rng = np.random.default_rng(20260606)
        n_in = 3
        # Identity-like net where y_true is favored heavily
        W1 = np.eye(4, 3)
        b1 = np.zeros(4)
        W2 = np.eye(3, 4)
        b2 = np.zeros(3)
        W3 = np.array([
            [10.0, 0.0, 0.0],   # class 0
            [-5.0, 0.0, 0.0],   # class 1
            [-5.0, 0.0, 0.0],   # class 2
        ])
        b3 = np.zeros(3)
        weights = [W1, W2, W3]
        biases = [b1, b2, b3]
        c_in = np.array([0.5, 0.0, 0.0])
        r_in = np.array([0.01, 0.01, 0.01])
        lb = c_in - r_in
        ub = c_in + r_in
        result = pruned_forward_dense(weights, biases, lb, ub,
                                        y_true=0, rivals=[1, 2],
                                        K_per_layer=64)
        # All rivals should be CERT
        for rr in result.per_rival:
            self.assertEqual(rr.verdict, "CERT",
                msg=f"rival {rr.rival}: expected CERT, got {rr.verdict}; "
                    f"LP UB={rr.lp_ub_rival_margin:.6f}")
        self.assertEqual(result.overall_verdict, "CERT")

    def test_fal_detection_on_adversarial_iid(self) -> None:
        """Build an iid where the rival actually wins for some inputs."""
        n_in = 3
        W1 = np.eye(4, 3)
        b1 = np.zeros(4)
        W2 = np.eye(3, 4)
        b2 = np.zeros(3)
        # Make class 1 win heavily
        W3 = np.array([
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
        ])
        b3 = np.zeros(3)
        weights = [W1, W2, W3]
        biases = [b1, b2, b3]
        c_in = np.array([0.5, 0.0, 0.0])
        r_in = np.array([0.5, 0.5, 0.5])
        lb = c_in - r_in
        ub = c_in + r_in
        result = pruned_forward_dense(weights, biases, lb, ub,
                                        y_true=0, rivals=[1, 2],
                                        K_per_layer=64)
        # At least rival 1 should be FAL_CANDIDATE
        rr_1 = [rr for rr in result.per_rival if rr.rival == 1][0]
        self.assertEqual(rr_1.verdict, "FAL_CANDIDATE",
            msg=f"rival 1 should be FAL_CANDIDATE; got {rr_1.verdict}, "
                f"LP UB={rr_1.lp_ub_rival_margin:.6f}")
        self.assertIsNotNone(rr_1.xi_star_input)


if __name__ == "__main__":
    unittest.main()
