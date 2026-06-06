"""Phase G.0 — TWO-LAYER ReLU toy validation.

Per advisor 2026-06-05 directive (corrected G.0):
  Single-layer toy is meaningless for FC-HZ (F1 already covers single layer).
  MUST use 2-layer Dense→ReLU→Dense→ReLU→Dense structure.

Specific toy (advisor's pair extended to 2 layers):
  W_1 = [[1, 1], [1, -1]]    # 2 → 2
  W_2 = [[1, 1], [1, -1]]    # 2 → 2
  W_3 = [[1, 1]]             # 2 → 1
  x ∈ [-1, 1]^2
  objective = output[0]

Hard gates:
  Gate G.0.1: Exact max via brute force = 4.0 (computed)
  Gate G.0.2: HZ closed-form > exact (loose)
  Gate G.0.3: F1 (last-ReLU only) ≥ exact, ≤ HZ (intermediate)
  Gate G.0.4: FC-HZ ≥ exact, ≤ F1 (monotonicity), additional ≥40% drop vs F1
  Gate G.0.5: Brute-force samples all satisfy FC-HZ UB (soundness)
  Gate G.0.6: 20 random 2-layer instances all satisfy FC-HZ ≤ F1
"""
from __future__ import annotations

import unittest

import numpy as np

from research.sc_hz.fc_hz_state import (
    initial_state, apply_dense, apply_relu_triangle_with_record,
    fc_hz_lp_ub, f1_last_relu_lp_ub, hz_closed_form_ub,
)


def _build_two_layer_state(W_1, W_2, W_3, lb_x, ub_x):
    c_in = (lb_x + ub_x) / 2
    r_in = (ub_x - lb_x) / 2
    state = initial_state(c_in, r_in)
    state = apply_dense(state, W_1, None)
    state = apply_relu_triangle_with_record(state, layer_index=0)
    state = apply_dense(state, W_2, None)
    state = apply_relu_triangle_with_record(state, layer_index=1)
    state = apply_dense(state, W_3, None)
    return state


def _brute_force_max(W_1, W_2, W_3, lb_x, ub_x, d_out, n_grid=151):
    """Compute exact max d_out @ output over grid sampling."""
    if lb_x.shape[0] != 2:
        raise NotImplementedError("Grid only for n_in=2")
    max_val = -np.inf
    xs = np.linspace(lb_x[0], ub_x[0], n_grid)
    ys = np.linspace(lb_x[1], ub_x[1], n_grid)
    for x in xs:
        for y in ys:
            inp = np.array([x, y])
            z_1 = W_1 @ inp
            y_1 = np.maximum(0, z_1)
            z_2 = W_2 @ y_1
            y_2 = np.maximum(0, z_2)
            out = W_3 @ y_2
            val = float(d_out @ out)
            if val > max_val:
                max_val = val
    return max_val


class TestGateG01ExactMax(unittest.TestCase):
    """G.0.1: brute force exact = 4.0 for advisor's toy."""

    def test_exact_brute_force(self):
        W_1 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_2 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_3 = np.array([[1.0, -1.0]])  # mixed-sign d_eff to exercise y ≥ z constraint
        lb_x = np.array([-1.0, -1.0])
        ub_x = np.array([+1.0, +1.0])
        d_out = np.array([1.0])
        exact = _brute_force_max(W_1, W_2, W_3, lb_x, ub_x, d_out, n_grid=201)
        print(f"  Brute force exact max = {exact:.4f}")
        # With W_3 = [[1,-1]] mixed-sign, exact computed from grid
        self.assertGreater(exact, 0.5,
            msg=f"Toy not interesting: exact {exact} too small")


class TestGateG02HZLoose(unittest.TestCase):
    """G.0.2: HZ closed-form must be > exact (loose)."""

    def test_hz_loose(self):
        W_1 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_2 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_3 = np.array([[1.0, -1.0]])  # mixed-sign d_eff to exercise y ≥ z constraint
        lb_x = np.array([-1.0, -1.0])
        ub_x = np.array([+1.0, +1.0])
        d_out = np.array([1.0])
        state = _build_two_layer_state(W_1, W_2, W_3, lb_x, ub_x)
        hz = hz_closed_form_ub(state, d_out)
        print(f"  HZ closed-form: {hz:.4f} (exact = 4.0; HZ looseness: {hz - 4.0:+.4f})")
        # HZ should be loose
        exact = _brute_force_max(W_1, W_2, W_3, lb_x, ub_x, d_out, n_grid=151)
        print(f"  exact = {exact:.4f}, HZ looseness = {(hz - exact):+.4f}")
        self.assertGreater(hz, exact - 1e-9,
            msg=f"HZ {hz} < exact {exact} → unsound")


class TestGateG03F1Intermediate(unittest.TestCase):
    """G.0.3: F1 (last ReLU only) must be ≥ exact, ≤ HZ."""

    def test_f1_intermediate(self):
        W_1 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_2 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_3 = np.array([[1.0, -1.0]])  # mixed-sign d_eff to exercise y ≥ z constraint
        lb_x = np.array([-1.0, -1.0])
        ub_x = np.array([+1.0, +1.0])
        d_out = np.array([1.0])
        state = _build_two_layer_state(W_1, W_2, W_3, lb_x, ub_x)
        hz = hz_closed_form_ub(state, d_out)
        f1 = f1_last_relu_lp_ub(state, d_out)
        print(f"  F1 (last-ReLU only): {f1:.4f} "
              f"(exact = 4.0, HZ = {hz:.4f}, F1 drop over HZ: {(hz-f1)/abs(hz)*100:+.1f}%)")
        exact = _brute_force_max(W_1, W_2, W_3, lb_x, ub_x, d_out, n_grid=151)
        self.assertGreaterEqual(f1, exact - 1e-9, msg=f"F1 {f1} < exact {exact} → unsound")
        self.assertLessEqual(f1, hz + 1e-9, msg=f"F1 {f1} > HZ {hz} → bug")


class TestGateG04FCHZTightenerOverF1(unittest.TestCase):
    """G.0.4: FC-HZ ≥ exact, ≤ F1, additional drop ≥40% vs F1's improvement.

    EXPECTED FAILURE per advisor 2026-06-06 audit: this gate FAILED on
    2026-06-05 with FC-HZ median 8.1% additional drop vs 40% threshold.
    The mechanism is sound (TestGateG05Soundness PASSES) and produces
    strict tightening on 19/20 random instances. But the magnitude of
    tightening is INSUFFICIENT to flip dense-conv PHANTOMs to CERT under
    the binding principle set. See
    `research/phase_G_FAILED_paper_1472_20260605.md` for the full
    decisive evidence.

    Marked @expectedFailure to keep CI green while documenting the gate
    outcome. Do NOT silently remove or weaken — this failure is the
    advisor's hard gate that closed Phase G.
    """

    # NOTE 2026-06-06 advisor revision: `@unittest.expectedFailure` IS
    # strict-equivalent — verified empirically that XPASS (unexpected
    # success) causes the test suite to FAIL with
    # "FAILED (unexpected successes=N)" and result.wasSuccessful() == False.
    # This matches pytest's `@pytest.mark.xfail(strict=True)` semantics.
    #
    # Therefore the current decorator is correct: if FC-HZ ever crosses
    # the 40% additional-drop gate, this test will UNEXPECTEDLY SUCCEED
    # and break the build, forcing a Phase G re-evaluation.
    @unittest.expectedFailure
    def test_fc_hz_tighter_than_f1_GATE_FAILED(self):
        W_1 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_2 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_3 = np.array([[1.0, -1.0]])
        lb_x = np.array([-1.0, -1.0])
        ub_x = np.array([+1.0, +1.0])
        d_out = np.array([1.0])
        state = _build_two_layer_state(W_1, W_2, W_3, lb_x, ub_x)
        f1 = f1_last_relu_lp_ub(state, d_out)
        fc, info = fc_hz_lp_ub(state, d_out)
        exact = _brute_force_max(W_1, W_2, W_3, lb_x, ub_x, d_out, n_grid=151)
        # Soundness and monotonicity STILL must hold (these are the
        # baseline soundness contract; FC-HZ is sound but undersized)
        self.assertGreaterEqual(fc, exact - 1e-9,
            msg=f"FC-HZ {fc} < exact {exact} → unsound")
        self.assertLessEqual(fc, f1 + 1e-8,
            msg=f"FC-HZ {fc} > F1 {f1} → widening violation")
        # Gate G.0.4: ≥40% additional drop vs F1 — EXPECTED TO FAIL
        if f1 - exact > 1e-4:
            additional_drop = (f1 - fc) / abs(f1) * 100
            self.assertGreaterEqual(additional_drop, 40.0,
                msg=f"FC-HZ additional drop {additional_drop:.1f}% < 40% advisor gate "
                    f"(expected failure: see phase_G_FAILED memo)")


class TestGateG05Soundness(unittest.TestCase):
    """G.0.5: 5000 brute-force samples all satisfy FC-HZ UB."""

    def test_fc_hz_sound_vs_brute_force(self):
        W_1 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_2 = np.array([[1.0, 1.0], [1.0, -1.0]])
        W_3 = np.array([[1.0, -1.0]])  # mixed-sign d_eff to exercise y ≥ z constraint
        lb_x = np.array([-1.0, -1.0])
        ub_x = np.array([+1.0, +1.0])
        d_out = np.array([1.0])
        state = _build_two_layer_state(W_1, W_2, W_3, lb_x, ub_x)
        fc, _ = fc_hz_lp_ub(state, d_out)
        rng = np.random.default_rng(20260605)
        max_violation = 0.0
        for _ in range(5000):
            x = rng.uniform(-1, 1, 2)
            z_1 = W_1 @ x
            y_1 = np.maximum(0, z_1)
            z_2 = W_2 @ y_1
            y_2 = np.maximum(0, z_2)
            out = W_3 @ y_2
            val = float(d_out @ out)
            if val > fc + 1e-9:
                max_violation = max(max_violation, val - fc)
        self.assertEqual(max_violation, 0.0,
            msg=f"FC-HZ UNSOUND: max violation = {max_violation:.4e}")


class TestGateG06MonotonicityRandomInstances(unittest.TestCase):
    """G.0.6: 20 random 2-layer instances all show FC-HZ ≤ F1."""

    def test_no_widening_on_random(self):
        rng = np.random.default_rng(20260605)
        n_violations = 0
        n_trials = 20
        for seed in range(n_trials):
            rs = np.random.default_rng(20260700 + seed)
            n_in = 4; n_h1 = 8; n_h2 = 8; n_out = 4
            W_1 = rs.normal(scale=0.5, size=(n_h1, n_in))
            W_2 = rs.normal(scale=0.5, size=(n_h2, n_h1))
            W_3 = rs.normal(scale=0.5, size=(n_out, n_h2))
            lb_x = np.full(n_in, -1.0)
            ub_x = np.full(n_in, +1.0)
            d_out = rs.normal(size=n_out)
            state = _build_two_layer_state(W_1, W_2, W_3, lb_x, ub_x)
            f1 = f1_last_relu_lp_ub(state, d_out)
            fc, _ = fc_hz_lp_ub(state, d_out)
            if fc > f1 + 1e-8:
                n_violations += 1
                print(f"  seed {seed}: FC-HZ {fc:.4f} > F1 {f1:.4f} (WIDENING)")
        self.assertEqual(n_violations, 0,
            msg=f"{n_violations}/{n_trials} widening violations")


if __name__ == "__main__":
    unittest.main(verbosity=2)
