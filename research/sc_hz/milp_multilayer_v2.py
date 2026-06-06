"""Multi-layer MILP V2: explicit y_L_i variables per layer (Tjeng on each layer).

V1 bug: kept box-encoding for slacks (y implicit via xi) + added redundant
binary sign constraints. Result: same as FC-HZ.

V2 fix: introduce explicit y_L_i continuous variable + b_L_i binary for
each unstable neuron in EACH layer's top-K. Replace slack-mediated
contribution to output with explicit y_L_i. Encode exact ReLU per layer.

For an FCHZState with output = state.c + state.G @ xi:
The state.G column for the slack of layer L neuron i has the form:
  G[:, slack_idx_L_i] = (effect of slack_L_i on output)
                     = mu_L_i * W_remaining_after_layer_L @ ?
                     = mu_L_i * (some vector in n_out)

We need to extract this vector and replace
  xi[slack_idx_L_i] * G[:, slack_idx_L_i]
with
  (y_L_i - lam_L_i * z_L_i - mu_L_i) / mu_L_i * G[:, slack_idx_L_i]
i.e., express output coefficient in terms of y_L_i.

This is messier. Simpler equivalent: treat the slack xi[slack_idx_L_i]
as a function of y_L_i and z_L_i, then sub in y_L_i:
  s = (2 * y - 2*lam*z - 2*mu) / (2*mu) = (y - lam*z - mu) / mu
But s ∈ [-1, 1] is the standard box constraint.

If we ALSO enforce ReLU exactness via binary b:
  y - z ≥ 0; y ≥ 0; y ≤ u*b; y ≤ z - l*(1-b)

Then s is implicitly bounded tighter than [-1, 1].

So multi-layer MILP CORRECTLY = add explicit y variables for each top-K
unstable neuron and corresponding binary, with:
  y_L_i ≥ 0
  y_L_i ≥ z_L_i
  y_L_i ≤ u_L_i * b_L_i
  y_L_i ≤ z_L_i - l_L_i * (1 - b_L_i)
And replace the slack contribution xi[slack_idx_L_i] in the LP objective
with the EQUIVALENT y_L_i contribution:
  s_L_i = (y_L_i - lam_L_i * z_L_i - mu_L_i) / mu_L_i
Then output contribution of slack:
  (G[:, slack_idx_L_i] @ d_out) * xi[slack_idx_L_i] →
  (G[:, slack_idx_L_i] @ d_out) * (y_L_i - lam_L_i * z_L_i - mu_L_i) / mu_L_i

Restated: subtract original slack effect, add explicit y-based effect.
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/data1/Kane/ACT')
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def fc_hz_milp_v2_ub(state, d_out, K_max_per_layer=10):
    """V2: Tjeng exact ReLU + explicit y variables for top-K unstable per layer."""
    from research.sc_hz.fc_hz_state import FCHZState
    if not isinstance(state, FCHZState):
        return None, {"error": "needs FCHZState"}
    K = state.K
    n_out = state.c.shape[0]
    d_eff_on_xi = d_out @ state.G  # (K,)

    # Collect selected (layer_rec, slot, neuron_idx, slack_idx, |d_eff|)
    selected = []
    layer_slot_to_y = {}  # (layer_rec_id, slot) -> y_var_idx
    for rec_id, rec in enumerate(state.slack_records):
        # Rank within this layer by |d_eff_on_xi[slack_idx]|
        per_layer = []
        for slot, neuron_idx in enumerate(rec.unstable_indices):
            slack_idx = int(rec.slack_indices[slot])
            if slack_idx >= K: continue
            per_layer.append((abs(d_eff_on_xi[slack_idx]), slot, int(neuron_idx), slack_idx))
        per_layer.sort(reverse=True)
        top = per_layer[:K_max_per_layer]
        for coef, slot, ni, sidx in top:
            selected.append((rec_id, rec, slot, ni, sidx))

    n_y = len(selected)
    n_b = n_y  # one binary per y

    # Variables: xi (K) + y (n_y) + b (n_b)
    n_vars = K + n_y + n_b
    integrality = np.zeros(n_vars, dtype=int)
    integrality[K + n_y:] = 1
    lb_var = np.zeros(n_vars)
    ub_var = np.zeros(n_vars)
    lb_var[:K] = -1.0; ub_var[:K] = 1.0
    # y bounds
    for k, (rec_id, rec, slot, ni, sidx) in enumerate(selected):
        l_i = float(rec.l[ni]); u_i = float(rec.u[ni])
        lb_var[K + k] = 0.0
        ub_var[K + k] = max(0.0, u_i)
    # b bounds
    lb_var[K + n_y:] = 0.0; ub_var[K + n_y:] = 1.0

    # Objective: max d_out @ output = d_out @ c + (d_out @ G) @ xi
    # BUT we need to REPLACE the slack-mediated contribution of selected slacks
    # with explicit y_L_i contribution.
    #
    # In box encoding: y_L_i (implicit) = lam*z_L_i + mu*(1 + xi[slack_idx])
    # Output contribution: G[:, slack_idx] @ d_out * xi[slack_idx]
    #
    # The "true" y_L_i contribution to output: through W_remaining @ y_L_i.
    # We need the effective coefficient of y_L_i on output, which is:
    #   d_eff_y = d_eff_on_xi[slack_idx] / mu_L_i
    #             (since slack contribution = mu * xi → mu * coef = effect; so y coef = coef / mu * 2? wait)
    #
    # Let me recompute carefully:
    # y_L_i = lam*z_L_i + mu*(1 + xi[slack_idx])
    # In propagation, y_L_i goes into a linear layer giving output state.G column for slack:
    # state.G[:, slack_idx] = (linear chain after layer L) @ (mu * e_i_neuron)
    #                       = mu_L_i * W_post[:, i_neuron]
    # So W_post[:, i_neuron] = state.G[:, slack_idx] / mu_L_i (where W_post is the linear chain after layer L)
    #
    # The full "y_L_i contribution to output":
    # Output += W_post[:, i_neuron] * y_L_i = (state.G[:, slack_idx] / mu_L_i) * y_L_i
    # d_out · (above) = (d_eff_on_xi[slack_idx] / mu_L_i) * y_L_i
    #
    # But the original encoding adds:
    # d_eff_on_xi[slack_idx] * xi[slack_idx] = mu * (1 + xi) * d_eff_y - mu * d_eff_y
    # OK reframe:
    # Original output: d_out @ state.c + (d_out @ state.G) @ xi
    # The state.c already includes the "mu" centering contribution. So:
    # d_out @ state.c[i_for_neuron] = ... + W_post[:, i_neuron] @ d_out * (lam * c_z_i + mu)
    #
    # I'll just SWAP the variables:
    # remove the xi[slack_idx] term from objective (set its coef to 0)
    # add y_L_i term with coef d_eff_on_xi[slack_idx] / mu_L_i
    # adjust constant: state.c includes the mu centering, so don't change it
    # WAIT — state.c includes (lam * c_z + mu) for the neuron's contribution.
    # If we use explicit y_L_i, we should NOT use the box-encoding contribution.
    # So we should SUBTRACT the mu-centering from c too.
    #
    # Actually simpler: in box encoding y_L_i = lam*z + mu + mu*xi[slack_idx]
    # If we make y_L_i explicit:
    # output = c + G @ xi where G includes columns for all slacks
    # But state.G[:, slack_idx] = mu_L_i * W_post[:, i_neuron]
    # And state.c[for neuron's contribution after post-layers] = W_post @ (lam*c_z + mu) for unstable
    # So output contribution from neuron i_neuron = W_post[:, i_neuron] * (lam*c_z + mu + mu*xi[slack_idx])
    #
    # If we use y_L_i instead:
    # output contribution = W_post[:, i_neuron] * y_L_i
    # Difference: subtract W_post[:, i_neuron] * (lam*c_z + mu + mu*xi[slack_idx])
    #             add W_post[:, i_neuron] * y_L_i
    #
    # In d_out · output:
    # Subtract d_eff_y * (lam*c_z + mu + mu*xi[slack_idx]) where d_eff_y = d_eff_on_xi[slack_idx] / mu_L_i
    #   = d_eff_y * lam * c_z + d_eff_y * mu + d_eff_on_xi[slack_idx] * xi[slack_idx]
    # Add d_eff_y * y_L_i

    c_obj = np.zeros(n_vars)
    obj_const = float(d_out @ state.c)
    # original xi coefficients
    c_obj[:K] = -(d_out @ state.G)

    # For each selected y, replace xi-contribution with y-contribution
    # Constraints will enforce y is correct ReLU output
    A_rows = []; lb_con = []; ub_con = []

    for k, (rec_id, rec, slot, ni, sidx) in enumerate(selected):
        l_i = float(rec.l[ni]); u_i = float(rec.u[ni])
        den = max(u_i - l_i, 1e-300)
        lam = u_i / den
        mu = -lam * l_i / 2.0
        if abs(mu) < 1e-300: continue
        d_eff_y = float(d_eff_on_xi[sidx] / mu)
        c_z_i = float(rec.c_z[ni])
        G_z_i = rec.G_z[ni, :]
        K_at_time = G_z_i.shape[0]

        # SUBTRACT full box-encoding y contribution to output:
        # y_L_i (box) = lam*c_z + mu + lam*G_z @ xi[:K_at] + mu*xi[slack_idx]
        # Output contribution = d_eff_y * y_L_i
        # Subtract from objective:
        # const: subtract d_eff_y * (lam*c_z + mu)
        obj_const = obj_const - d_eff_y * (lam * c_z_i + mu)
        # xi[:K_at] coefs: subtract d_eff_y * lam * G_z[neuron, :K_at]
        # (c_obj had -(d_out @ G), so add d_eff_y * lam * G_z to undo)
        c_obj[:K_at_time] = c_obj[:K_at_time] + d_eff_y * lam * G_z_i
        # xi[slack_idx] coef: subtract d_eff_y * mu = d_eff_on_xi[slack_idx]
        c_obj[sidx] = c_obj[sidx] + d_eff_on_xi[sidx]

        # ADD y_L_i contribution to output (explicit): d_eff_y * y_L_i
        # MILP minimizes, so c -= d_eff_y on y var
        c_obj[K + k] = c_obj[K + k] - d_eff_y

        # Tjeng constraints
        # y_L_i ≥ 0  (via bounds, already lb=0)
        # y_L_i ≥ z_L_i = c_z + G_z @ xi[:K_at]
        # → G_z @ xi - y_L_i ≤ -c_z
        row = np.zeros(n_vars)
        row[:K_at_time] = G_z_i
        row[K + k] = -1.0
        A_rows.append(row); lb_con.append(-np.inf); ub_con.append(-c_z_i)
        # y_L_i ≤ u_i * b_L_i
        # → y_L_i - u_i * b ≤ 0
        row = np.zeros(n_vars)
        row[K + k] = 1.0
        row[K + n_y + k] = -u_i
        A_rows.append(row); lb_con.append(-np.inf); ub_con.append(0.0)
        # y_L_i ≤ z_L_i - l_i * (1 - b_L_i)
        # → y_L_i - z_L_i - l_i*b_L_i ≤ -l_i
        # → -G_z @ xi + y_L_i - l_i*b ≤ -l_i + c_z
        row = np.zeros(n_vars)
        row[:K_at_time] = -G_z_i
        row[K + k] = 1.0
        row[K + n_y + k] = -l_i
        A_rows.append(row); lb_con.append(-np.inf); ub_con.append(-l_i + c_z_i)

    A_arr = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    lb_arr = np.array(lb_con) if lb_con else np.zeros(0)
    ub_arr = np.array(ub_con) if ub_con else np.zeros(0)

    constraints = LinearConstraint(A_arr, lb=lb_arr, ub=ub_arr)
    res = milp(c=c_obj, constraints=constraints, integrality=integrality,
                 bounds=Bounds(lb=lb_var, ub=ub_var),
                 options={'time_limit': 120, 'mip_rel_gap': 1e-9})
    info = {'n_y': n_y, 'n_b': n_b, 'status': res.status, 'msg': res.message}
    if res.status != 0:
        return None, info
    return obj_const - float(res.fun), info


def main():
    import numpy as np
    from research.rbt.residual_toy import build_residual_block_weights, build_n_block_state, brute_force_n_block
    from research.sc_hz.fc_hz_state import f1_last_relu_lp_ub, fc_hz_lp_ub, hz_closed_form_ub

    n_in = 4; n_hidden = 12; n_blocks = 4
    lb = np.full(n_in, -1.0); ub = np.full(n_in, 1.0)
    print("V2 Test on Phase L0 residual toy:")
    print(f"{'Trial':<5} {'bf':>7} {'hz':>7} {'f1':>7} {'fc':>7} {'milp':>7} {'drop':>7}")
    drops_milp = []; drops_fc = []
    for trial in range(10):
        seed = 20260606 + trial
        W_lists = []; b_lists = []
        for k in range(n_blocks):
            W1, b1, W2, b2 = build_residual_block_weights(n_in, n_hidden, seed + k*1000, w1_scale=0.6, w2_scale=0.5)
            W_lists.append((W1, W2)); b_lists.append((b1, b2))
        rs = np.random.default_rng(seed)
        W_out = rs.normal(scale=0.3, size=(4, n_in))
        d_out = rs.normal(size=4)
        state = build_n_block_state(W_lists, b_lists, W_out, lb, ub)
        bf = brute_force_n_block(W_lists, b_lists, W_out, lb, ub, d_out, n_samples=2000)
        f1 = f1_last_relu_lp_ub(state, d_out)
        fc, _ = fc_hz_lp_ub(state, d_out)
        milp_ub, info = fc_hz_milp_v2_ub(state, d_out, K_max_per_layer=12)
        if milp_ub is not None and abs(f1) > 1e-9:
            drop_milp = (f1 - milp_ub) / abs(f1) * 100
            drop_fc = (f1 - fc) / abs(f1) * 100
            drops_milp.append(drop_milp)
            drops_fc.append(drop_fc)
            print(f"{trial:<5} {bf:+7.2f} {f1*1+0:+7.2f} {f1:+7.2f} {fc:+7.2f} {milp_ub:+7.2f} {drop_milp:+7.1f}%")

    if drops_milp:
        median_milp = sorted(drops_milp)[len(drops_milp)//2]
        median_fc = sorted(drops_fc)[len(drops_fc)//2]
        print(f"\nMedian FC-HZ drop over F1: {median_fc:+.1f}%")
        print(f"Median MILP V2 drop over F1: {median_milp:+.1f}%")
        if median_milp >= 50:
            print(f"*** MILP V2 PASSES Phase L0 gate (>=50%)! ***")


if __name__ == "__main__":
    main()
