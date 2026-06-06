"""F1 prototype unit tests for `constrained_lp_ub`.

Acceptance criteria:
  1. Constrained LP UB >= brute-force max (soundness)
  2. Constrained LP UB <= closed-form HZ LP UB (tightness)
  3. On simple synthetic data, constrained UB strictly tighter than closed-form
  4. No infeasible/unbounded LP solutions on well-formed input
"""
from __future__ import annotations

import unittest
import numpy as np

from research.sc_hz.constrained_lp import (
    LastReluRecord, constrained_lp_ub, closed_form_hz_lp_ub,
    brute_force_max_d_out_y,
)


def _make_synthetic_relu_record(
    n_pre: int = 8, K: int = 6, seed: int = 20260605,
    with_tail: bool = True,
) -> LastReluRecord:
    rng = np.random.default_rng(seed)
    c_z = rng.normal(scale=0.2, size=n_pre).astype(np.float64)
    G_z = rng.normal(scale=0.15, size=(n_pre, K)).astype(np.float64)
    tail = (np.abs(rng.normal(scale=0.05, size=n_pre)).astype(np.float64)
            if with_tail else None)
    # Set bounds to force mix of active/inactive/unstable
    # box bounds = c +/- |G|.sum() + tail
    rad = np.abs(G_z).sum(axis=1) + (tail if tail is not None else 0)
    l = c_z - rad
    u = c_z + rad
    return LastReluRecord(c_z=c_z, G_z=G_z, tail_z=tail, l=l, u=u)


class TestConstrainedLPSoundness(unittest.TestCase):
    """Constrained LP UB must >= brute-force max."""

    def test_sound_brute_force(self):
        rng = np.random.default_rng(20260605)
        relu_rec = _make_synthetic_relu_record(n_pre=8, K=6, with_tail=True)
        Co = 4
        W_remaining = rng.normal(scale=0.4, size=(Co, relu_rec.n_pre))
        b_remaining = rng.normal(scale=0.1, size=Co)
        d_out = rng.normal(size=Co)
        # 1. Constrained LP UB
        lp_ub, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
        # 2. Brute-force max
        bf_max = brute_force_max_d_out_y(
            relu_rec, W_remaining, b_remaining, d_out, n_samples=5000,
        )
        self.assertGreaterEqual(
            lp_ub + 1e-9, bf_max,
            msg=f"LP UB {lp_ub:.4e} < brute-force max {bf_max:.4e} → UNSOUND",
        )


class TestConstrainedTighterThanClosedForm(unittest.TestCase):
    """Constrained LP UB should be <= closed-form (and ideally strictly less)."""

    def test_constrained_no_looser_than_closed_form(self):
        rng = np.random.default_rng(20260605)
        relu_rec = _make_synthetic_relu_record(n_pre=8, K=6, with_tail=True)
        Co = 4
        W_remaining = rng.normal(scale=0.4, size=(Co, relu_rec.n_pre))
        b_remaining = rng.normal(scale=0.1, size=Co)
        d_out = rng.normal(size=Co)
        lp_ub, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
        cf_ub = closed_form_hz_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
        self.assertLessEqual(
            lp_ub, cf_ub + 1e-9,
            msg=f"constrained {lp_ub:.4e} > closed-form {cf_ub:.4e} (not tighter)"
        )

    def test_strictly_tighter_on_random_seeds(self):
        """Constrained should be strictly tighter on SOME random instances."""
        n_strict_tighter = 0
        n_trials = 8
        for seed in range(20260700, 20260700 + n_trials):
            rng = np.random.default_rng(seed)
            relu_rec = _make_synthetic_relu_record(n_pre=10, K=6, seed=seed, with_tail=True)
            Co = 4
            W_remaining = rng.normal(scale=0.4, size=(Co, relu_rec.n_pre))
            b_remaining = rng.normal(scale=0.1, size=Co)
            d_out = rng.normal(size=Co)
            lp_ub, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
            cf_ub = closed_form_hz_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
            if lp_ub < cf_ub - 1e-4:
                n_strict_tighter += 1
        self.assertGreater(
            n_strict_tighter, 0,
            msg=f"constrained LP not strictly tighter on any of {n_trials} seeds; "
                f"check implementation",
        )


class TestAllInactiveAllActive(unittest.TestCase):
    def test_all_active(self):
        rng = np.random.default_rng(20260605)
        n_pre = 6; K = 4
        c_z = np.full(n_pre, 1.0)
        G_z = rng.normal(scale=0.05, size=(n_pre, K)) * 0  # zero generators
        l = c_z - 0.1
        u = c_z + 0.1
        relu_rec = LastReluRecord(c_z=c_z, G_z=G_z, tail_z=None, l=l, u=u)
        W_remaining = np.eye(n_pre)
        b_remaining = np.zeros(n_pre)
        d_out = rng.normal(size=n_pre)
        lp_ub, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
        # All active: y = z, so max d·y = sum_i d_i * (c_i + 0.1*sign(d_i)) (within bounds)
        # since G_z=0, z = c_z exactly → y = c_z (active) → d·y = d·c_z
        expected = float(d_out @ c_z)
        self.assertAlmostEqual(lp_ub, expected, places=4)

    def test_all_inactive(self):
        rng = np.random.default_rng(20260605)
        n_pre = 6
        c_z = np.full(n_pre, -1.0)
        G_z = np.zeros((n_pre, 4))
        l = c_z - 0.1; u = c_z + 0.1
        relu_rec = LastReluRecord(c_z=c_z, G_z=G_z, tail_z=None, l=l, u=u)
        W_remaining = np.eye(n_pre)
        b_remaining = np.zeros(n_pre)
        d_out = rng.normal(size=n_pre)
        lp_ub, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
        # All y = 0; d·y = 0
        self.assertAlmostEqual(lp_ub, 0.0, places=4)


class TestConstrainedFeasibility(unittest.TestCase):
    """No LP should be reported infeasible/unbounded."""

    def test_no_infeasible_on_random_seeds(self):
        for seed in range(20260800, 20260820):
            rng = np.random.default_rng(seed)
            relu_rec = _make_synthetic_relu_record(n_pre=12, K=6, seed=seed)
            Co = 4
            W_remaining = rng.normal(scale=0.4, size=(Co, relu_rec.n_pre))
            b_remaining = rng.normal(scale=0.1, size=Co)
            d_out = rng.normal(size=Co)
            lp_ub, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
            self.assertNotEqual(
                lp_ub, float("inf"),
                msg=f"seed {seed} produced infeasible LP",
            )


if __name__ == "__main__":
    unittest.main()
