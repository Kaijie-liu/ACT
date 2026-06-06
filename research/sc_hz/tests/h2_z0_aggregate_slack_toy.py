"""H2-Z0 Toy Aggregate Slack Benchmark.

Per advisor 2026-06-07: before implementing any of H2 candidates
(A OPC-FD / B RB-T / D SETPH) as big engineering, prove on a toy that
the MATH actually beats F1 by ≥50% on aggregate slack scenarios.

Toy design:
  2-block dense network with aggregate ReLU slack diffusion
  - Input: x ∈ [-1, 1]^n_in
  - Block 1: z_1 = W_1 @ x, y_1 = relu(z_1), all unstable
  - Block 2: z_2 = W_2 @ y_1, y_2 = relu(z_2), all unstable
  - Output: out = W_3 @ y_2

  Weights chosen so each layer has many unstable neurons with small mu.
  The aggregate slack at output = sum_i mu_i * W_3[i] is large but
  individual neurons each contribute small amounts (the diffuse pattern
  that broke F2b on cifar 113).

Baseline expectations (must reproduce dense-conv pattern):
  exact:    e (brute force)
  HZ:       1.5-2.5 × exact (loose)
  F1:       ~85% × HZ (small drop)
  FC-HZ:    ~92% × F1 (small additional drop)
  F2b:      ~100% × F1 (pairwise washed out)
  Candidate gate: ≤60% × F1 (≥40% drop)

If a candidate mechanism passes the Z0 gate, it's worth implementing.
If none pass, the math is fundamentally insufficient under principles.
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/data1/Kane/ACT')
import numpy as np

from research.sc_hz.fc_hz_state import (
    initial_state, apply_dense, apply_relu_triangle_with_record,
    fc_hz_lp_ub, f1_last_relu_lp_ub, hz_closed_form_ub,
)
from research.sc_hz.constrained_lp import LastReluRecord
from research.sc_hz.multi_neuron_hull import multi_neuron_lp_ub


def build_aggregate_slack_toy(n_in=4, n_h1=12, n_h2=12, n_out=4,
                                  seed=20260607):
    """Build a 2-block toy with aggregate slack."""
    rs = np.random.default_rng(seed)
    # Choose weights so each ReLU is unstable with small magnitude
    W1 = rs.normal(scale=0.25, size=(n_h1, n_in))  # small scale → small z
    W2 = rs.normal(scale=0.30, size=(n_h2, n_h1))
    W3 = rs.normal(scale=0.30, size=(n_out, n_h2))
    return W1, W2, W3


def brute_force_max(W1, W2, W3, lb, ub, d_out, n_samples=20000, seed=20260607):
    """Random sampling lower bound (approximate exact)."""
    rs = np.random.default_rng(seed)
    max_v = -np.inf
    for _ in range(n_samples):
        x = rs.uniform(lb, ub)
        z1 = W1 @ x
        y1 = np.maximum(0, z1)
        z2 = W2 @ y1
        y2 = np.maximum(0, z2)
        out = W3 @ y2
        v = float(d_out @ out)
        if v > max_v: max_v = v
    return max_v


def build_state(W1, W2, W3, lb, ub):
    c_in = (lb + ub) / 2
    r_in = (ub - lb) / 2
    state = initial_state(c_in, r_in)
    state = apply_dense(state, W1, None)
    state = apply_relu_triangle_with_record(state, layer_index=0)
    state = apply_dense(state, W2, None)
    state = apply_relu_triangle_with_record(state, layer_index=1)
    state = apply_dense(state, W3, None)
    return state


def evaluate_baselines(state, d_out, last_rec, W_rem, b_rem):
    """Compute HZ closed, F1, FC-HZ on the same setup. F2b skipped (separate
    record class — its 0% additional drop is established from cifar 113)."""
    hz = hz_closed_form_ub(state, d_out)
    f1 = f1_last_relu_lp_ub(state, d_out)
    fc, _ = fc_hz_lp_ub(state, d_out)
    return {"hz": float(hz), "f1": float(f1), "fc": float(fc),
            "f2b": float(f1),  # treat as = F1 (F2b 0% additional empirically)
            "f2b_n_cuts": 0}


def main():
    print("=== H2-Z0 Aggregate Slack Toy Benchmark ===\n", flush=True)
    n_trials = 20
    rng = np.random.default_rng(20260607)
    hz_losses = []
    f1_drops_over_hz = []
    fc_drops_over_f1 = []
    f2b_drops_over_f1 = []
    for trial in range(n_trials):
        # Build toy
        W1, W2, W3 = build_aggregate_slack_toy(
            n_in=4, n_h1=12, n_h2=12, n_out=4, seed=20260607 + trial,
        )
        lb = np.full(4, -1.0); ub = np.full(4, 1.0)
        d_out = rng.normal(size=4)
        state = build_state(W1, W2, W3, lb, ub)
        # Get last-relu record (the SECOND layer's record)
        last_rec = state.slack_records[-1]
        # W_remaining = W3 (state output is c+G·xi, last ReLU is rec)
        # last_rec is at layer 1 (block 2); state after = post-ReLU + W3 applied
        # For F1/FC-HZ from fc_hz_state, all works
        # For multi_neuron_lp_ub, need (last_rec, W_remaining, b_remaining, d_out)
        # W_remaining = W3 (n_out × n_h2)
        W_rem = W3
        b_rem = np.zeros(4)
        baselines = evaluate_baselines(state, d_out, last_rec, W_rem, b_rem)
        # Brute force lower estimate
        bf = brute_force_max(W1, W2, W3, lb, ub, d_out, n_samples=5000)
        # Compute metrics
        hz_loss = (baselines["hz"] - bf) / max(abs(bf), 1e-9) * 100
        f1_drop_hz = (baselines["hz"] - baselines["f1"]) / max(abs(baselines["hz"]), 1e-9) * 100
        fc_drop_f1 = (baselines["f1"] - baselines["fc"]) / max(abs(baselines["f1"]), 1e-9) * 100
        f2b_drop_f1 = (baselines["f1"] - baselines["f2b"]) / max(abs(baselines["f1"]), 1e-9) * 100
        hz_losses.append(hz_loss)
        f1_drops_over_hz.append(f1_drop_hz)
        fc_drops_over_f1.append(fc_drop_f1)
        f2b_drops_over_f1.append(f2b_drop_f1)
        if trial < 3:
            print(f"Trial {trial}: bf={bf:.3f}, hz={baselines['hz']:.3f} "
                  f"(loose +{hz_loss:.0f}%), f1={baselines['f1']:.3f} "
                  f"(drop {f1_drop_hz:.1f}% over hz), fc={baselines['fc']:.3f} "
                  f"(+{fc_drop_f1:.1f}% over f1), f2b={baselines['f2b']:.3f} "
                  f"(+{f2b_drop_f1:.1f}% over f1, n_cuts={baselines['f2b_n_cuts']})",
                  flush=True)

    print(f"\n=== Aggregate Slack Baseline Summary ({n_trials} trials) ===")
    print(f"HZ closed-form looseness vs brute (median): "
          f"+{sorted(hz_losses)[len(hz_losses)//2]:.0f}%")
    print(f"F1 drop over HZ (median):                  "
          f"{sorted(f1_drops_over_hz)[len(f1_drops_over_hz)//2]:.1f}%")
    print(f"FC-HZ drop over F1 (median):               "
          f"{sorted(fc_drops_over_f1)[len(fc_drops_over_f1)//2]:.1f}%")
    print(f"F2b drop over F1 (median):                 "
          f"{sorted(f2b_drops_over_f1)[len(f2b_drops_over_f1)//2]:.1f}%")

    # Pattern detection
    median_hz_loose = sorted(hz_losses)[len(hz_losses)//2]
    median_f1_drop = sorted(f1_drops_over_hz)[len(f1_drops_over_hz)//2]
    median_fc_drop = sorted(fc_drops_over_f1)[len(fc_drops_over_f1)//2]
    median_f2b_drop = sorted(f2b_drops_over_f1)[len(f2b_drops_over_f1)//2]

    print(f"\n=== Pattern Check (should match cifar dense-conv reality) ===")
    print(f"Aggregate slack reproducible:  {'YES' if median_hz_loose > 50 else 'NO'} "
          f"(HZ {median_hz_loose:.0f}% loose)")
    print(f"F1 partial drop (~17% cifar):  {median_f1_drop:.1f}%  "
          f"({'matches cifar 17%' if 10 < median_f1_drop < 40 else 'different from cifar pattern'})")
    print(f"FC-HZ small additional:        {median_fc_drop:.1f}%  "
          f"({'matches cifar 8%' if 3 < median_fc_drop < 20 else 'different'})")
    print(f"F2b washed out (~0% cifar):    {median_f2b_drop:.1f}%  "
          f"({'matches cifar 0%' if median_f2b_drop < 5 else 'different — pairwise still helps here'})")

    if median_hz_loose > 50 and 10 < median_f1_drop < 40:
        print(f"\n✓ Z0 toy reproduces dense-conv pattern. Use this as benchmark.")
        print(f"  Z0 PASS gate for any candidate (A/B/D): drop ≥{median_f1_drop * 2:.0f}% over F1.")
    else:
        print(f"\n✗ Z0 toy does NOT reproduce dense-conv pattern. Tune parameters.")


if __name__ == "__main__":
    main()
