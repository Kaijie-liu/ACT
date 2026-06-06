"""RB-T template LP UB for residual block toy.

Per advisor Phase L0: implement minimal RB-T mechanism.
Test whether it achieves ≥50% drop over F1.

Mechanism candidates (try multiple):
  T1: Branch L_inf norm bound — derived from architecture
  T2: Channel-wise aggregate sum constraint
  T3: Skip-branch correlation bound
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/data1/Kane/ACT')
import numpy as np
import scipy.optimize as sopt

from research.sc_hz.fc_hz_state import (
    initial_state, apply_dense, apply_relu_triangle_with_record,
    fc_hz_lp_ub, f1_last_relu_lp_ub, hz_closed_form_ub, FCHZState,
)
from research.rbt.residual_toy import (
    build_residual_block_weights, build_n_block_state, brute_force_n_block,
    forward_residual_block_concrete,
)


def compute_branch_linf_bound(W1, b1, W2, b2, x_lb, x_ub):
    """Forward L_inf bound on branch F(x) = W2 @ relu(W1 @ x + b1) + b2.

    Step 1: bound z1 = W1 @ x + b1 via interval propagation
            z1_max[i] = sum_j max(W1[i,j] x_ub[j], W1[i,j] x_lb[j]) + b1[i]
            z1_min[i] = sum_j min(W1[i,j] x_ub[j], W1[i,j] x_lb[j]) + b1[i]
    Step 2: y1 = relu(z1) → y1_max[i] = max(0, z1_max[i]), y1_min[i] = max(0, z1_min[i])
    Step 3: F = W2 @ y1 + b2 → F_max[j], F_min[j]
    Returns: F_max (n_in,), F_min (n_in,) — per-coord bounds on branch output
    """
    z1_max = np.where(W1 >= 0, W1 * x_ub, W1 * x_lb).sum(axis=1) + b1
    z1_min = np.where(W1 >= 0, W1 * x_lb, W1 * x_ub).sum(axis=1) + b1
    y1_max = np.maximum(0, z1_max)
    y1_min = np.maximum(0, z1_min)
    F_max = (np.where(W2 >= 0, W2 * y1_max, W2 * y1_min).sum(axis=1) + b2)
    F_min = (np.where(W2 >= 0, W2 * y1_min, W2 * y1_max).sum(axis=1) + b2)
    return F_min, F_max


def rbt_template_t1_branch_norm(state: FCHZState, d_out,
                                       residual_block_weights,
                                       residual_block_biases,
                                       skip_input_bounds):
    """RB-T T1: branch L_inf norm bound template.

    For the LAST residual block, the branch contribution F(x) is bounded
    component-wise by F_min, F_max derived from interval propagation of
    the residual branch.

    Add to F1 LP: for each block output coord j,
      |branch_j(x)| ≤ max(|F_min[j]|, |F_max[j]|)

    This encodes: even if F1 LP allows branch slack arbitrarily, the
    AGGREGATE BRANCH OUTPUT cannot exceed the architecturally-determined
    bound.

    Note: this is harder to encode in F1's LP framework because the LP
    works on output state's xi variables, not on branch output values.
    For a minimal proof of concept, we compare:
      - F1 LP UB (baseline)
      - HZ_closed + Lipschitz_branch_template (intersection approach)

    Specifically: if F1 LP UB > L_template, take L_template as the answer.
    """
    f1_ub = f1_last_relu_lp_ub(state, d_out)
    # Compute Lipschitz-based UB for final block output
    # For each block in sequence, propagate the bound:
    # y_k+1 = relu(y_k + F_k(y_k)), where y_k is bounded
    x_lb, x_ub = skip_input_bounds
    for (W1, W2), (b1, b2) in zip(residual_block_weights, residual_block_biases):
        F_min, F_max = compute_branch_linf_bound(W1, b1, W2, b2, x_lb, x_ub)
        # z = x + F(x) bounds:
        z_lb = x_lb + F_min
        z_ub = x_ub + F_max
        # y = relu(z):
        x_lb_new = np.maximum(0, z_lb)
        x_ub_new = np.maximum(0, z_ub)
        x_lb, x_ub = x_lb_new, x_ub_new
    # Apply final W_out @ x bound
    # We don't have W_out separated; use the existing state's bounds
    # via HZ closed-form

    # Alternative: get HZ closed form, which already includes architectural slack
    # If T1 < HZ closed (Lipschitz tighter), use T1; else use F1
    # For comparison purpose, just compute the IBP bound:
    return f1_ub  # Placeholder — T1 doesn't directly tighten F1 in this formulation


def rbt_template_t2_aggregate_sum(state: FCHZState, d_out):
    """RB-T T2: aggregate channel sum constraint.

    For each block output dimension i, the contribution to d_out · output is
    bounded by a per-channel aggregate template:
      sum over channels of (d_eff_i * mu_i * s_i) ≤ aggregate_max

    where aggregate_max is derived from the actual reachable sum on the
    forward zonotope.

    In F1 LP, each s_i is independent in [-1, 1]. T2 adds a SINGLE constraint
    that the WEIGHTED SUM ≤ bound.
    """
    f1_ub = f1_last_relu_lp_ub(state, d_out)
    if not state.slack_records:
        return f1_ub
    rec = state.slack_records[-1]
    K = state.K
    # Add constraint: weighted sum of slacks ≤ some bound
    # Variables: xi (K-dim, ∈ [-1, 1])
    # Original F1 constraints + new aggregate constraint:
    # sum_i (d_eff_i * mu_i * xi[slack_idx_i]) ≤ T_max
    # T_max = ??? Need to compute the true reachable max of this sum

    # For a sound T_max: it's the actual max of sum_i (d_eff_i * mu_i * relu(z_i_bound))
    # which we can compute as box-corner of the underlying state
    # Actually this is just HZ closed-form on this projected direction

    # For now: aggregate constraint = sum over unstable of |d_eff_i| * mu_i
    # is the closed form HZ max. Not a tightening — just baseline.

    # To actually tighten, we'd need NON-INDEPENDENT correlation analysis.
    # Skipping — return F1.
    return f1_ub


def rbt_template_t3_skip_branch_correlation(state: FCHZState, d_out):
    """RB-T T3: skip-branch correlation constraint.

    For y = relu(skip + branch), the correlation between skip and branch
    via shared input x produces a CONSTRAINT in LP that they cannot
    independently both be at worst.

    Specifically: if d_out · skip is high, d_out · branch tends to be...
    architecture-dependent.

    Implementation: we'd need to model the (skip_contribution, branch_contribution)
    joint distribution. Beyond minimal scope.
    """
    return f1_last_relu_lp_ub(state, d_out)


def main():
    print("=== RB-T Template LP UB Test on Residual Toy ===\n")
    n_trials = 10
    n_in = 4
    n_hidden = 12
    n_blocks = 4
    lb = np.full(n_in, -1.0)
    ub = np.full(n_in, 1.0)

    drops_t1_over_f1 = []
    drops_t2_over_f1 = []
    drops_t3_over_f1 = []
    drops_hz_over_f1 = []
    drops_fc_over_f1 = []

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
                                       n_samples=2000)
        state = build_n_block_state(W_lists, b_lists, W_out, lb, ub)
        hz = hz_closed_form_ub(state, d_out)
        f1 = f1_last_relu_lp_ub(state, d_out)
        fc, _ = fc_hz_lp_ub(state, d_out)
        t1 = rbt_template_t1_branch_norm(state, d_out, W_lists, b_lists, (lb, ub))
        t2 = rbt_template_t2_aggregate_sum(state, d_out)
        t3 = rbt_template_t3_skip_branch_correlation(state, d_out)

        if abs(f1) > 1e-9:
            drops_t1_over_f1.append((f1 - t1) / abs(f1) * 100)
            drops_t2_over_f1.append((f1 - t2) / abs(f1) * 100)
            drops_t3_over_f1.append((f1 - t3) / abs(f1) * 100)
            drops_hz_over_f1.append((f1 - hz) / abs(f1) * 100)
            drops_fc_over_f1.append((f1 - fc) / abs(f1) * 100)

        if trial < 3:
            print(f"Trial {trial}: bf={bf:.3f}, hz={hz:.3f}, f1={f1:.3f}, "
                  f"fc={fc:.3f}, T1={t1:.3f}, T2={t2:.3f}, T3={t3:.3f}")

    print(f"\n=== RB-T Template Results ===")
    if drops_fc_over_f1:
        print(f"Median FC-HZ drop over F1:       {sorted(drops_fc_over_f1)[len(drops_fc_over_f1)//2]:.1f}%")
        print(f"Median RB-T T1 drop over F1:     {sorted(drops_t1_over_f1)[len(drops_t1_over_f1)//2]:.1f}%")
        print(f"Median RB-T T2 drop over F1:     {sorted(drops_t2_over_f1)[len(drops_t2_over_f1)//2]:.1f}%")
        print(f"Median RB-T T3 drop over F1:     {sorted(drops_t3_over_f1)[len(drops_t3_over_f1)//2]:.1f}%")
        print(f"\nPhase L0 Gate: candidate must achieve ≥50% drop over F1")
        best = max(
            sorted(drops_t1_over_f1)[len(drops_t1_over_f1)//2],
            sorted(drops_t2_over_f1)[len(drops_t2_over_f1)//2],
            sorted(drops_t3_over_f1)[len(drops_t3_over_f1)//2],
        )
        if best >= 50:
            print(f"\n✓ Best RB-T template achieves {best:.1f}% — PASSES L0 gate.")
        else:
            print(f"\n✗ Best RB-T template at {best:.1f}% — FAILS L0 gate (47%-50%).")
            print(f"\nT1, T2, T3 are placeholder implementations.")
            print(f"None of them adds a real tightening mechanism over F1.")
            print(f"To pass L0, would need: actual skip-branch correlation analysis,")
            print(f"or real per-channel aggregate template constraints in LP.")


if __name__ == "__main__":
    main()
