"""F2b prototype tests for multi_neuron_hull module.

Hard gates per advisor:
  1. multi-neuron LP UB <= F1 LP UB + 1e-8 (no widening, gate G9)
  2. multi-neuron LP UB >= brute-force max (soundness)
  3. Demonstrate strict tightening on at least some synthetic instances
"""
from __future__ import annotations

import unittest
import numpy as np

from research.sc_hz.constrained_lp import (
    LastReluRecord, constrained_lp_ub, brute_force_max_d_out_y,
)
from research.sc_hz.multi_neuron_hull import (
    multi_neuron_lp_ub, derive_pairwise_zonotope_cuts,
    _zonotope_2d_extremes, _exact_joint_relu_max_on_box,
)


def _make_corr_relu_record(
    n_pre: int = 8, K: int = 6, seed: int = 20260605,
    correlation_strength: float = 0.7,
) -> LastReluRecord:
    """Make a record where pre-activations are STRONGLY CORRELATED.

    Set G[i, k] = base[k] + (i-specific term) so that z_i's share
    most of their xi-dependence on the first column.
    """
    rng = np.random.default_rng(seed)
    c_z = rng.normal(scale=0.05, size=n_pre).astype(np.float64)
    G_z = np.zeros((n_pre, K), dtype=np.float64)
    # Strong shared column 0
    G_z[:, 0] = correlation_strength * rng.normal(scale=0.5, size=n_pre).astype(np.float64)
    # Per-neuron variation
    for k in range(1, K):
        G_z[:, k] = (1 - correlation_strength) * rng.normal(scale=0.1, size=n_pre).astype(np.float64)
    tail = np.abs(rng.normal(scale=0.02, size=n_pre)).astype(np.float64)
    rad = np.abs(G_z).sum(axis=1) + tail
    l = c_z - rad
    u = c_z + rad
    return LastReluRecord(c_z=c_z, G_z=G_z, tail_z=tail, l=l, u=u)


class TestF2bMonotonicityGate(unittest.TestCase):
    """F2b LP UB must be ≤ F1 LP UB (no widening)."""

    def test_no_widening_on_random_seeds(self):
        n_violations = 0
        for seed in range(20260800, 20260820):
            rng = np.random.default_rng(seed)
            relu_rec = _make_corr_relu_record(n_pre=10, K=6, seed=seed)
            Co = 4
            W_remaining = rng.normal(scale=0.3, size=(Co, relu_rec.n_pre))
            b_remaining = rng.normal(scale=0.05, size=Co)
            d_out = rng.normal(size=Co)
            ub_f1, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
            ub_f2b, _ = multi_neuron_lp_ub(
                relu_rec, W_remaining, b_remaining, d_out, top_k_neurons=4,
            )
            if ub_f2b > ub_f1 + 1e-8:
                n_violations += 1
                print(f"seed {seed}: F2b {ub_f2b:.4e} > F1 {ub_f1:.4e} (violation)")
        self.assertEqual(n_violations, 0,
            msg=f"{n_violations}/20 widening violations — F2b not sound vs F1")


class TestF2bSoundnessVsBruteForce(unittest.TestCase):
    """F2b LP UB must be ≥ brute-force max."""

    def test_sound_vs_brute_force(self):
        n_violations = 0
        for seed in range(20260700, 20260710):
            rng = np.random.default_rng(seed)
            relu_rec = _make_corr_relu_record(n_pre=8, K=6, seed=seed)
            Co = 4
            W_remaining = rng.normal(scale=0.3, size=(Co, relu_rec.n_pre))
            b_remaining = rng.normal(scale=0.05, size=Co)
            d_out = rng.normal(size=Co)
            ub_f2b, _ = multi_neuron_lp_ub(
                relu_rec, W_remaining, b_remaining, d_out, top_k_neurons=4,
            )
            bf_max = brute_force_max_d_out_y(
                relu_rec, W_remaining, b_remaining, d_out, n_samples=3000,
            )
            if ub_f2b + 1e-9 < bf_max:
                n_violations += 1
        self.assertEqual(n_violations, 0,
            msg=f"{n_violations}/10 brute-force violations — F2b UNSOUND")


class TestF2bStrictTighteningOnCorrelated(unittest.TestCase):
    """On STRONGLY correlated z's, F2b should strictly tighten F1."""

    def test_strict_tightening_on_correlated(self):
        n_strict = 0
        n_trials = 10
        improvements = []
        for seed in range(20260750, 20260750 + n_trials):
            rng = np.random.default_rng(seed)
            relu_rec = _make_corr_relu_record(n_pre=8, K=6, seed=seed,
                                                  correlation_strength=0.85)
            Co = 4
            W_remaining = rng.normal(scale=0.3, size=(Co, relu_rec.n_pre))
            b_remaining = rng.normal(scale=0.05, size=Co)
            d_out = rng.normal(size=Co)
            ub_f1, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d_out)
            ub_f2b, _ = multi_neuron_lp_ub(
                relu_rec, W_remaining, b_remaining, d_out, top_k_neurons=4,
            )
            if ub_f2b < ub_f1 - 1e-4:
                n_strict += 1
            improvements.append((ub_f1 - ub_f2b) / max(abs(ub_f1), 1e-9) * 100)
        print(f"  F2b strict tightening: {n_strict}/{n_trials} seeds")
        print(f"  median additional gain over F1: {sorted(improvements)[len(improvements)//2]:.1f}%")
        # We require at least ONE strict tighten on correlated data
        self.assertGreater(n_strict, 0,
            msg="F2b not strictly tighter on any of 10 correlated seeds — design may be wrong")


class TestZonotopeExtremes(unittest.TestCase):
    """Simple shape check on 2D zonotope extreme vertex enumeration."""

    def test_diagonal_zonotope(self):
        # Single generator (1, 1) → segment from (-1,-1) to (+1,+1)
        verts = _zonotope_2d_extremes(np.array([1.0]), np.array([1.0]))
        coords = sorted(verts)
        self.assertEqual(coords[0], (-1.0, -1.0))
        self.assertEqual(coords[-1], (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
