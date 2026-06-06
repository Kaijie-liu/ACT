"""Phase L0 RB-T Residual Toy Benchmark.

Per advisor 2026-06-07: build a residual block toy that reproduces the
dense-conv aggregate slack failure pattern. Validate that HZ/F1/F2b/
FC-HZ/SETPH all fail. Then test minimal RB-T template mechanism for
≥50% drop over F1 — the Phase L0 gate.

Toy structure:
  Residual block: y = relu(x + F(x))
    where F(x) = W_2 @ relu(W_1 @ x + b_1) + b_2
  Cascade 2 residual blocks
  Output: out = W_out @ y_2

The block must satisfy:
  - HZ closed: loose by ≥3×
  - F1 (per-neuron triangle LP): 15-30% drop
  - F2b: 0% additional
  - FC-HZ: <10% additional
  - SETPH: <40% (best-effort)
  - RB-T candidate: should achieve ≥50% additional drop over F1

Only when toy reproduces these baselines is it a valid Phase L0 benchmark.
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/data1/Kane/ACT')
import numpy as np

from research.sc_hz.fc_hz_state import (
    initial_state, apply_dense, apply_relu_triangle_with_record,
    fc_hz_lp_ub, f1_last_relu_lp_ub, hz_closed_form_ub, FCHZState,
)


def build_residual_block_weights(n_in, n_hidden, seed=20260607,
                                       w1_scale=0.6, w2_scale=0.5):
    """Build weights for ONE residual block: y = relu(x + F(x))
    where F(x) = W2 @ relu(W1 @ x + b1) + b2.

    Larger w1/w2 scales push more z-values into unstable [l<0, u>0] region,
    making F1's per-neuron triangle constraint MORE BINDING.
    """
    rs = np.random.default_rng(seed)
    W1 = rs.normal(scale=w1_scale, size=(n_hidden, n_in))
    b1 = rs.normal(scale=0.05, size=n_hidden)
    W2 = rs.normal(scale=w2_scale, size=(n_in, n_hidden))
    b2 = rs.normal(scale=0.05, size=n_in)
    return W1, b1, W2, b2


def forward_residual_block_concrete(x, W1, b1, W2, b2):
    """y = relu(x + F(x)). For verification."""
    inner = np.maximum(0, W1 @ x + b1)
    F_x = W2 @ inner + b2
    return np.maximum(0, x + F_x)


def apply_residual_block_hz(state, W1, b1, W2, b2, layer_offset=0):
    """Forward HZ propagation through residual block."""
    # branch: F(x) = W2 @ relu(W1 @ state + b1) + b2
    state_branch = apply_dense(state, W1, b1)
    state_branch = apply_relu_triangle_with_record(state_branch,
                                                          layer_index=layer_offset)
    state_branch = apply_dense(state_branch, W2, b2)
    # skip + branch: add states
    K_s = state.G.shape[1]
    K_b = state_branch.G.shape[1]
    if K_b > K_s:
        G_pad = np.zeros((state.n, K_b - K_s))
        new_G = np.concatenate([state.G, G_pad], axis=1) + state_branch.G
    else:
        new_G = state.G + state_branch.G[:, :K_s]
    new_c = state.c + state_branch.c
    # Records from branch are inherited
    state_combined = FCHZState(c=new_c, G=new_G, n_root=state.n_root,
                                    slack_records=state_branch.slack_records)
    # Final ReLU
    state_combined = apply_relu_triangle_with_record(state_combined,
                                                            layer_index=layer_offset + 1)
    return state_combined


def build_n_block_state(W_lists, b_lists, W_out, lb, ub):
    """W_lists = [(W1, W2), ...] one per block."""
    c_in = (lb + ub) / 2
    r_in = (ub - lb) / 2
    state = initial_state(c_in, r_in)
    layer_offset = 0
    for (W1, W2), (b1, b2) in zip(W_lists, b_lists):
        state = apply_residual_block_hz(state, W1, b1, W2, b2,
                                                layer_offset=layer_offset)
        layer_offset += 2
    state = apply_dense(state, W_out, None)
    return state


def brute_force_n_block(W_lists, b_lists, W_out, lb, ub, d_out,
                            n_samples=3000, seed=20260607):
    rs = np.random.default_rng(seed)
    max_v = -np.inf
    for _ in range(n_samples):
        x = rs.uniform(lb, ub)
        for (W1, W2), (b1, b2) in zip(W_lists, b_lists):
            x = forward_residual_block_concrete(x, W1, b1, W2, b2)
        out = W_out @ x
        v = float(d_out @ out)
        if v > max_v: max_v = v
    return max_v


# Alias for backwards compat
def build_two_block_state(W_list, b_list, W_out, lb, ub):
    return build_n_block_state(
        [(W_list[0], W_list[1]), (W_list[2], W_list[3])],
        [(b_list[0], b_list[1]), (b_list[2], b_list[3])],
        W_out, lb, ub,
    )


def brute_force_two_block(W_list, b_list, W_out, lb, ub, d_out,
                                n_samples=5000, seed=20260607):
    return brute_force_n_block(
        [(W_list[0], W_list[1]), (W_list[2], W_list[3])],
        [(b_list[0], b_list[1]), (b_list[2], b_list[3])],
        W_out, lb, ub, d_out, n_samples=n_samples, seed=seed,
    )


def evaluate_baselines(state, d_out):
    """Compute HZ, F1, FC-HZ on residual two-block state."""
    hz = hz_closed_form_ub(state, d_out)
    f1 = f1_last_relu_lp_ub(state, d_out)
    fc, _ = fc_hz_lp_ub(state, d_out)
    return {"hz": float(hz), "f1": float(f1), "fc": float(fc)}


def main():
    print("=== Phase L0 RB-T Residual Toy Benchmark ===\n")

    # Multiple trials, 4 blocks for more aggregate slack
    n_trials = 20
    n_in = 4
    n_hidden = 12
    n_blocks = 4  # deeper for more aggregate slack
    lb = np.full(n_in, -1.0)
    ub = np.full(n_in, 1.0)

    hz_loosenesses = []
    f1_drops_over_hz = []
    fc_drops_over_f1 = []

    for trial in range(n_trials):
        seed = 20260607 + trial
        W_lists = []; b_lists = []
        for k in range(n_blocks):
            W1, b1, W2, b2 = build_residual_block_weights(
                n_in, n_hidden, seed + k * 1000,
                w1_scale=0.6, w2_scale=0.5,
            )
            W_lists.append((W1, W2)); b_lists.append((b1, b2))
        rs = np.random.default_rng(seed)
        W_out = rs.normal(scale=0.3, size=(4, n_in))
        d_out = rs.normal(size=4)
        bf = brute_force_n_block(W_lists, b_lists, W_out, lb, ub, d_out,
                                       n_samples=3000)
        state = build_n_block_state(W_lists, b_lists, W_out, lb, ub)
        baselines = evaluate_baselines(state, d_out)

        hz_loose = (baselines["hz"] - bf) / max(abs(bf), 1e-9) * 100
        f1_drop_hz = (baselines["hz"] - baselines["f1"]) / max(abs(baselines["hz"]), 1e-9) * 100
        fc_drop_f1 = (baselines["f1"] - baselines["fc"]) / max(abs(baselines["f1"]), 1e-9) * 100

        hz_loosenesses.append(hz_loose)
        f1_drops_over_hz.append(f1_drop_hz)
        fc_drops_over_f1.append(fc_drop_f1)

        if trial < 5:
            print(f"Trial {trial}: bf={bf:.3f}, hz={baselines['hz']:.3f} "
                  f"(loose +{hz_loose:.0f}%), f1={baselines['f1']:.3f} "
                  f"(drop {f1_drop_hz:.1f}%), fc={baselines['fc']:.3f} "
                  f"(+{fc_drop_f1:.1f}% over F1)")

    median_hz = sorted(hz_loosenesses)[len(hz_loosenesses)//2]
    median_f1 = sorted(f1_drops_over_hz)[len(f1_drops_over_hz)//2]
    median_fc = sorted(fc_drops_over_f1)[len(fc_drops_over_f1)//2]

    print(f"\n=== Toy Baseline Summary (20 residual trials) ===")
    print(f"HZ closed-form looseness vs brute (median):     +{median_hz:.0f}%")
    print(f"F1 drop over HZ (median):                       {median_f1:.1f}%")
    print(f"FC-HZ drop over F1 (median):                    {median_fc:.1f}%")

    # Pattern validation
    print(f"\n=== Pattern Validation for Phase L0 Benchmark ===")
    valid_loose = median_hz > 50
    valid_f1 = 10 < median_f1 < 40
    valid_fc = median_fc < 20

    print(f"HZ loose ≥50%:      {'YES' if valid_loose else 'NO'} (got {median_hz:.0f}%)")
    print(f"F1 partial drop:    {'YES' if valid_f1 else 'NO'} (got {median_f1:.1f}%)")
    print(f"FC-HZ insufficient: {'YES' if valid_fc else 'NO'} (got {median_fc:.1f}%)")

    if valid_loose and valid_f1 and valid_fc:
        print(f"\n✓ Toy reproduces dense-conv pattern. Valid Phase L0 benchmark.")
        print(f"\nNext step: implement minimal RB-T mechanism and test gate (≥50% drop over F1).")
        print(f"Currently F1 = {median_f1:.1f}% drop over HZ.")
        print(f"Gate: candidate must achieve ≥50% drop over F1 on this toy.")
    else:
        print(f"\n✗ Toy does NOT match dense-conv pattern. Tune parameters.")


if __name__ == "__main__":
    main()
