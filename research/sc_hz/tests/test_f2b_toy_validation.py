"""F2b soundness-first gates per advisor 2026-06-05 directive.

Three hard gates that must pass BEFORE any F2b cifar scoring run:

GATE A: Toy exact-hull benchmark
  - 2 strongly correlated ReLUs: z_1 = ξ_1+ξ_2, z_2 = ξ_1-ξ_2
  - Compute EXACT max y_1+y_2 by brute force (8 vertex combinations)
  - F1 per-neuron LP UB should be loose (proves the problem is real)
  - F2b LP UB should be TIGHTER than F1, approaching exact
  - If F2b is not strictly tighter than F1 on this toy, the mechanism is
    fundamentally insufficient — close F2b before any cifar.

GATE B: Convex-hull validity (cut exclusion test)
  - Sample many ξ → real (z_1, z_2, y_1=relu(z_1), y_2=relu(z_2))
  - Check that every cut admits these real points (cut.alpha_i*y_1 +
    cut.alpha_j*y_2 ≤ cut.rhs for all sampled points)
  - If ANY real point violates ANY cut, the cut is UNSOUND.

GATE C: Monotonicity on synthetic close-to-cifar shape
  - With K~thousands, n_pre~hundreds, F2b UB ≤ F1 UB on every instance.
"""
from __future__ import annotations

import unittest

import numpy as np

from research.sc_hz.constrained_lp import (
    LastReluRecord, constrained_lp_ub,
)
from research.sc_hz.multi_neuron_hull import (
    multi_neuron_lp_ub, derive_pairwise_zonotope_cuts,
)


class TestGateAToyExactHull(unittest.TestCase):
    """Toy 2-ReLU benchmark from advisor's directive:
      z_1 = ξ_1 + ξ_2
      z_2 = ξ_1 - ξ_2
      objective = y_1 + y_2  (i.e., d_eff = [1, 1])
    """

    def setUp(self):
        # G_z shape (2, 2): row i = pre-act i's coefficients
        # z_1 = c_1 + G_z[0, 0]*ξ_1 + G_z[0, 1]*ξ_2
        # We want z_1 = ξ_1 + ξ_2 → c_1 = 0, G_z[0] = [1, 1]
        # z_2 = ξ_1 - ξ_2 → c_2 = 0, G_z[1] = [1, -1]
        self.c_z = np.array([0.0, 0.0])
        self.G_z = np.array([[1.0, 1.0], [1.0, -1.0]])
        # No tail
        self.tail_z = None
        # Bounds: z_1 ∈ [-2, +2], z_2 ∈ [-2, +2], both unstable (l<0<u)
        self.l = np.array([-2.0, -2.0])
        self.u = np.array([+2.0, +2.0])
        self.rec = LastReluRecord(
            c_z=self.c_z, G_z=self.G_z, tail_z=self.tail_z, l=self.l, u=self.u,
        )
        # Identity W_remaining, zero bias — output = y itself
        self.W_remaining = np.eye(2)
        self.b_remaining = np.zeros(2)
        self.d_out = np.array([1.0, 1.0])  # objective y_1 + y_2

    def test_brute_force_exact_max(self):
        """Compute exact max y_1 + y_2 over (ξ_1, ξ_2) ∈ [-1,1]^2."""
        max_val = -np.inf
        # Dense grid is fine for 2D
        for xi_1 in np.linspace(-1, 1, 201):
            for xi_2 in np.linspace(-1, 1, 201):
                z_1 = xi_1 + xi_2
                z_2 = xi_1 - xi_2
                y_1 = max(0.0, z_1)
                y_2 = max(0.0, z_2)
                if y_1 + y_2 > max_val:
                    max_val = y_1 + y_2
        # True max: at ξ_1 = 1, ξ_2 = 0: z_1 = 1, z_2 = 1, y_1+y_2 = 2
        # Or at ξ_1 = 1, ξ_2 = 1: z_1 = 2, z_2 = 0, y_1+y_2 = 2
        self.assertAlmostEqual(max_val, 2.0, places=2)

    def test_f1_lp_ub(self):
        """F1 per-neuron triangle LP should give a SOUND but LOOSE bound."""
        ub_f1, _ = constrained_lp_ub(
            self.rec, self.W_remaining, self.b_remaining, self.d_out,
        )
        # Must be sound: >= 2.0 (exact)
        self.assertGreaterEqual(ub_f1, 2.0 - 1e-9,
            msg=f"F1 UB {ub_f1} < exact 2.0 → unsound")
        # Should be LOOSE (strictly larger than exact) — this proves the problem
        # exists. If F1 = exact already, there's nothing to improve.
        print(f"  F1 UB: {ub_f1:.4f}  (exact = 2.0; F1 looseness: {ub_f1 - 2.0:+.4f})")

    def test_f2b_lp_ub_tighter_than_f1(self):
        """F2b should be at least as tight as F1, ideally strictly tighter."""
        ub_f1, _ = constrained_lp_ub(
            self.rec, self.W_remaining, self.b_remaining, self.d_out,
        )
        ub_f2b, info = multi_neuron_lp_ub(
            self.rec, self.W_remaining, self.b_remaining, self.d_out,
            top_k_neurons=2, return_solution=True,
        )
        print(f"  F1 UB: {ub_f1:.4f}, F2b UB: {ub_f2b:.4f}, "
              f"cuts={info.get('n_cuts')}, exact=2.0")
        # MONOTONICITY: F2b ≤ F1
        self.assertLessEqual(ub_f2b, ub_f1 + 1e-8,
            msg=f"F2b UB {ub_f2b} > F1 UB {ub_f1} — WIDENING (G9 VIOLATION)")
        # SOUNDNESS: F2b ≥ 2.0 (exact)
        self.assertGreaterEqual(ub_f2b, 2.0 - 1e-9,
            msg=f"F2b UB {ub_f2b} < exact 2.0 → UNSOUND")
        # TIGHTNESS PROVING USEFULNESS: F2b strictly < F1 OR F1 already = 2.0
        if ub_f1 > 2.0 + 1e-4:
            self.assertLess(ub_f2b, ub_f1 - 1e-4,
                msg=f"F2b UB {ub_f2b} not strictly tighter than F1 UB {ub_f1} "
                    f"on a problem WHERE F1 IS LOOSE — F2b mechanism FAILED")


class TestGateBConvexHullValidity(unittest.TestCase):
    """Check no real (z, relu(z)) point is excluded by any F2b cut."""

    def test_no_real_point_excluded(self):
        """Sample real ξ points; verify they satisfy all derived cuts."""
        rng = np.random.default_rng(20260605)
        n_pre = 12; K = 8
        c_z = rng.normal(scale=0.1, size=n_pre).astype(np.float64)
        G_z = rng.normal(scale=0.3, size=(n_pre, K)).astype(np.float64)
        tail = np.abs(rng.normal(scale=0.05, size=n_pre)).astype(np.float64)
        rad = np.abs(G_z).sum(axis=1) + tail
        l = c_z - rad
        u = c_z + rad
        rec = LastReluRecord(c_z=c_z, G_z=G_z, tail_z=tail, l=l, u=u)
        # Random d_out and W
        Co = 4
        W_rem = rng.normal(scale=0.3, size=(Co, n_pre))
        d_out = rng.normal(size=Co)
        d_eff = W_rem.T @ d_out
        # Derive cuts
        cuts = derive_pairwise_zonotope_cuts(rec, d_eff, top_k=6)
        # Sample real ξ; compute (z, relu(z)); check cuts
        n_samples = 5000
        violations = 0
        max_viol = 0.0
        for _ in range(n_samples):
            xi = rng.uniform(-1, 1, K)
            xi_t = rng.uniform(-1, 1, n_pre)
            z = c_z + G_z @ xi + tail * xi_t
            y = np.maximum(0, z)
            for cut in cuts:
                lhs = cut.alpha_i * y[cut.i] + cut.alpha_j * y[cut.j]
                if lhs > cut.rhs + 1e-9:
                    violations += 1
                    max_viol = max(max_viol, lhs - cut.rhs)
        self.assertEqual(violations, 0,
            msg=f"{violations}/{n_samples * len(cuts)} cut violations on real "
                f"(z, relu(z)) points (max excess {max_viol:.4e}) — UNSOUND")


class TestGateCMonotonicityCloseToCifar(unittest.TestCase):
    """On larger synthetic close to cifar shape (K=1000, n_pre=100),
    F2b UB ≤ F1 UB on every instance."""

    def test_no_widening_on_larger_synthetic(self):
        rng = np.random.default_rng(20260605)
        n_pre = 100; K = 1000
        Co = 50
        n_violations = 0
        n_trials = 5
        for trial in range(n_trials):
            c_z = rng.normal(scale=0.1, size=n_pre).astype(np.float64)
            G_z = rng.normal(scale=0.05, size=(n_pre, K)).astype(np.float64)
            tail = np.abs(rng.normal(scale=0.02, size=n_pre)).astype(np.float64)
            rad = np.abs(G_z).sum(axis=1) + tail
            l = c_z - rad
            u = c_z + rad
            rec = LastReluRecord(c_z=c_z, G_z=G_z, tail_z=tail, l=l, u=u)
            W_rem = rng.normal(scale=0.3, size=(Co, n_pre))
            b_rem = rng.normal(scale=0.05, size=Co)
            d_out = rng.normal(size=Co)
            ub_f1, _ = constrained_lp_ub(rec, W_rem, b_rem, d_out)
            ub_f2b, _ = multi_neuron_lp_ub(rec, W_rem, b_rem, d_out, top_k_neurons=4)
            if ub_f2b > ub_f1 + 1e-8:
                n_violations += 1
        self.assertEqual(n_violations, 0,
            msg=f"{n_violations}/{n_trials} widening on larger synthetic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
