"""Multi-layer MILP on FCHZState (all ReLU layers get binary indicators).

Extension of fc_hz_lp_ub: instead of triangle relaxation per layer,
add Tjeng exact ReLU encoding for top-K unstable per layer.

This requires FCHZState with all SlackRecord layers populated, which
we don't currently have from the walker. So this is for future use
with a multi-layer walker.

For now: support FCHZState input + return MILP UB.
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/data1/Kane/ACT')
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def fc_hz_milp_ub(state, d_out, K_max_per_layer=15):
    """Multi-layer MILP UB on FCHZState.

    Variables:
      xi (K dim continuous, [-1, 1])
      For each layer's unstable neurons in top_K: y_L_i continuous + b_L_i binary

    Constraints per layer L, per unstable i:
      z_L_i = c_z_L[i] + G_z_L[i, :] @ xi[0:K_at_time]
      y_L_i ≥ 0; y_L_i ≥ z_L_i; y_L_i ≤ u_i b_L_i; y_L_i ≤ z_L_i - l_i(1-b_L_i)

    Objective: max d_out @ (state.c + state.G @ xi)
                                                 BUT for the contribution from
                                                 top-K neurons of last layer,
                                                 we should override with explicit y_L_i.

    Simpler approach (just last-layer Tjeng + per-layer triangle):
      All layers use triangle relaxation (as in fc_hz_lp_ub)
      ONLY last layer's top-K get binaries

    For a TRUE multi-layer MILP, each layer's slack would need binary
    indicators. This grows variable count.
    """
    from research.sc_hz.fc_hz_state import FCHZState
    if not isinstance(state, FCHZState):
        return None
    K = state.K
    n_out = state.c.shape[0]

    # Pick top K_max_per_layer unstable per layer by |d_eff|
    # Since this is FCHZState, output state's G already has W_remaining applied
    # d_out @ state.G gives K-dim coefficient on xi
    d_eff_on_xi = d_out @ state.G  # (K,)

    # Collect binary indicators for all selected slacks across layers
    # variable: xi (K) + b_per_slack (one per selected slack)
    selected_slack_indices = []
    selected_layer_records = []
    for rec in state.slack_records:
        # Compute d_eff per unstable neuron in this layer:
        # contribution to output via slack_idx
        for slot, neuron_idx in enumerate(rec.unstable_indices):
            slack_idx = int(rec.slack_indices[slot])
            if slack_idx >= K: continue
            # Coefficient of xi[slack_idx] on output: d_eff_on_xi[slack_idx]
            selected_slack_indices.append((rec, slot, neuron_idx, slack_idx,
                                                abs(d_eff_on_xi[slack_idx])))
    # Sort by |coef| and pick top K_max_per_layer * n_layers (total budget)
    selected_slack_indices.sort(key=lambda x: -x[4])
    n_total_budget = K_max_per_layer * len(state.slack_records)
    selected_slack_indices = selected_slack_indices[:n_total_budget]
    n_b = len(selected_slack_indices)
    binary_set = set(s[3] for s in selected_slack_indices)

    # Variables: xi (K) + b (n_b binaries indexed by selected_slack_indices)
    n_vars = K + n_b
    integrality = np.zeros(n_vars, dtype=int)
    integrality[K:] = 1

    lb_var = np.zeros(n_vars)
    ub_var = np.zeros(n_vars)
    lb_var[:K] = -1.0; ub_var[:K] = 1.0
    lb_var[K:] = 0.0; ub_var[K:] = 1.0

    # Objective: max d_out @ (state.c + state.G @ xi) = const + (d_out @ G) @ xi
    c_obj = np.zeros(n_vars)
    c_obj[:K] = -(d_out @ state.G)
    obj_const = float(d_out @ state.c)

    # Constraints
    A_rows = []; lb_con = []; ub_con = []

    # For each unstable neuron i in layer L:
    # z_L_i = c_z[i] + G_z[i, :] @ xi[0:K_at_time]
    # In the box encoding: y_L_i = lam*z_L_i + mu + mu * xi[slack_idx_i]
    #
    # If we DON'T use binary for this neuron: standard triangle (already enforced
    # by xi[slack_idx] ∈ [-1, 1] + y_L_i ≥ z_L_i + y_L_i ≥ 0)
    #
    # If we use binary b for this neuron:
    # Replace xi[slack_idx] with the EXACT relu indicator.
    # Specifically: when b=1 (active): y_L_i = z_L_i = lam*z + mu*(0) wait
    #
    # The relation: y_L_i = lam*z + mu*(1+xi[slack_idx])
    # At xi[slack_idx]=-1 (s=-1): y = lam*z (this is the BOTTOM of triangle, y=lam*z)
    # At xi[slack_idx]=+1 (s=+1): y = lam*z + 2*mu = chord(z)
    #
    # For EXACT ReLU with binary b ∈ {0, 1}:
    # b=0 (inactive): y = 0 → lam*z + mu + mu*xi[slack_idx] = 0
    #                     → xi[slack_idx] = -1 - lam*z/mu (depends on z)
    # b=1 (active):   y = z → lam*z + mu + mu*xi[slack_idx] = z
    #                     → mu * xi[slack_idx] = z - lam*z - mu = (1-lam)*z - mu
    #                     → xi[slack_idx] = ((1-lam)*z - mu) / mu
    #
    # Equivalently, encode:
    # b=0: y_L_i = 0 (enforced by:
    #   lam*z + mu*(1+xi[slack_idx]) ≤ 0  AND  same ≥ 0
    # b=1: y_L_i = z_L_i (enforced by:
    #   lam*z + mu*(1+xi[slack_idx]) ≤ z  AND  same ≥ z
    # Combined: y_L_i ≥ 0; y_L_i ≤ M*b; y_L_i ≥ z; y_L_i ≤ z + M*(1-b)
    # Wait — but y_L_i isn't a variable here. The slack xi[slack_idx] IS the variable.
    #
    # Let me reformulate:
    # If we have a binary indicator b for layer L unstable neuron i:
    # b=1 means "this neuron is active": z ≥ 0 AND xi[slack_idx] = ((1-lam)*z - mu)/mu
    # b=0 means "inactive": z ≤ 0 AND xi[slack_idx] = -1 - lam*z/mu
    #
    # Both can be encoded as big-M:
    # b=1: z_L_i ≥ 0 → c_z + G_z @ xi[:K_at] ≥ 0
    #                → -G_z @ xi[:K_at] ≤ c_z
    # b=0: z_L_i ≤ 0 → G_z @ xi[:K_at] ≤ -c_z
    # And the slack xi[slack_idx] is constrained accordingly.
    #
    # For simplicity, use BIG-M form on the SIGN of z_L_i:
    # z_L_i + M*(1 - b) ≥ 0  (always true when b=1; when b=0, ≥ 0 enforced if z ≥ -M)
    # z_L_i - M*b ≤ 0        (always true when b=0; when b=1, z ≤ M enforced)
    #
    # If z bounds are [l, u], pick M = max(|l|, |u|).
    #
    # Constraint 1: z + M*(1-b) ≥ 0  → -z - M*(1-b) ≤ 0
    #              → -G_z @ xi[:K_at] - M*(1-b) ≤ c_z - M
    # Constraint 2: z - M*b ≤ 0 → G_z @ xi[:K_at] - M*b ≤ -c_z

    # Map slack_idx → binary index
    binary_idx_map = {s[3]: bi for bi, s in enumerate(selected_slack_indices)}

    for rec in state.slack_records:
        K_at_time = rec.G_z.shape[1]
        for slot, neuron_idx in enumerate(rec.unstable_indices):
            slack_idx = int(rec.slack_indices[slot])
            if slack_idx >= K: continue
            l_i = float(rec.l[neuron_idx])
            u_i = float(rec.u[neuron_idx])
            den = max(u_i - l_i, 1e-300)
            lam = u_i / den
            mu = -lam * l_i / 2.0
            G_z_i = rec.G_z[neuron_idx, :]
            c_z_i = float(rec.c_z[neuron_idx])
            M = max(abs(l_i), abs(u_i)) + 1.0

            # Standard F1-style triangle constraints (always added):
            # y_L_i = lam*z + mu*(1+xi[slack_idx]) ≥ 0
            #   -lam*G_z @ xi - mu*xi[slack_idx] ≤ lam*c_z + mu
            row = np.zeros(n_vars)
            row[:K_at_time] = -lam * G_z_i
            row[slack_idx] = row[slack_idx] - mu
            A_rows.append(row)
            lb_con.append(-np.inf); ub_con.append(lam * c_z_i + mu)
            # y_L_i ≥ z_L_i
            row = np.zeros(n_vars)
            row[:K_at_time] = -(lam - 1.0) * G_z_i
            row[slack_idx] = row[slack_idx] - mu
            A_rows.append(row)
            lb_con.append(-np.inf); ub_con.append((lam - 1.0) * c_z_i + mu)

            # Binary indicator constraints (for selected slacks)
            if slack_idx in binary_idx_map:
                bi = binary_idx_map[slack_idx]
                # z_L_i + M*(1-b) ≥ 0  →  -G_z @ xi - M + M*b ≤ c_z
                #                      → -G_z @ xi + M*b ≤ c_z + M
                row = np.zeros(n_vars)
                row[:K_at_time] = -G_z_i
                row[K + bi] = M
                A_rows.append(row)
                lb_con.append(-np.inf); ub_con.append(c_z_i + M)
                # z_L_i - M*b ≤ 0 → G_z @ xi - M*b ≤ -c_z
                row = np.zeros(n_vars)
                row[:K_at_time] = G_z_i
                row[K + bi] = -M
                A_rows.append(row)
                lb_con.append(-np.inf); ub_con.append(-c_z_i)

    A_arr = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    lb_arr = np.array(lb_con) if lb_con else np.zeros(0)
    ub_arr = np.array(ub_con) if ub_con else np.zeros(0)

    constraints = LinearConstraint(A_arr, lb=lb_arr, ub=ub_arr)
    res = milp(
        c=c_obj, constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb=lb_var, ub=ub_var),
        options={'time_limit': 120, 'mip_rel_gap': 1e-9},
    )
    info = {'n_binaries': n_b, 'status': res.status, 'msg': res.message}
    if res.status != 0:
        return None, info
    return obj_const - float(res.fun), info


if __name__ == "__main__":
    print("Multi-layer MILP encoder ready. Requires FCHZState input.")
