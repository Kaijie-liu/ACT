# ===- research/fchz/m4_full_lp.py - Small-dense full-network LP ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor 2026-06-08:
#   Small-dense full-network continuous LP relaxation inside FCHZ.
#
#   - Forward FCHZ runs first to compute per-ReLU pre-activation bounds (l, u).
#   - For dense-only chain (no Conv / no Slice / no Concat), build a single
#     continuous LP over ALL ReLU layers using the original input noise vars
#     (NOT the accumulated slack), keeping triangle constraints EXPLICIT.
#   - HiGHS / scipy.linprog only; no MILP, no Gurobi, no integer vars, no BaB.
#
# This addresses the F1 LP failure mode (mid-layer correlation loss): by
# keeping all ReLU triangles explicit and coupling them via affine eqs,
# the LP captures the constraint that the same input xi must yield
# consistent (z, y) at every layer.
#
# Variables:
#   xi[K_in]              ∈ [-1, +1]     (input noise vars from FCHZ INPUT state)
#   y_layer_k[n_k]                       (post-activation per layer)
#     - active   : y_k_i = z_k_i (eq constraint)
#     - inactive : y_k_i = 0 (var bound)
#     - unstable : 0 ≤ y_k_i, y_k_i ≥ z_k_i, y_k_i ≤ λ_i (z_k_i - l_i)
#
# Constraints (each layer k, with composed_W_k, composed_b_k):
#   z_0 = composed_W_0 @ (input.c + input.G @ xi) + composed_b_0
#   z_k = composed_W_k @ y_{k-1} + composed_b_k   (k >= 1)
#
# Output:
#   z_out = output_W @ y_{R-1} + output_b
#   Objective: max d @ z_out  OR  min d @ z_out
#
# Status (advisor 2026-06-08): research SCOUT module.
#   M4A v2 sentinel: 22 sentinels, 0 V flip, mean UB drop 55%.
#   Pending: soundness audit + unit tests before any production integration.
#
# ===---------------------------------------------------------------------===#

"""Full-network continuous LP for small dense FCHZ refinement."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import numpy as np
import scipy.optimize as sopt


def is_dense_only_chain(net):
    """Check if the net is a dense-only chain.

    Allowed layer kinds: DENSE, RELU, BIAS, FLATTEN, INPUT, INPUT_SPEC, ASSERT.

    Returns True iff every layer is one of the allowed kinds.
    """
    allowed = {'INPUT', 'INPUT_SPEC', 'DENSE', 'RELU', 'BIAS', 'FLATTEN', 'ASSERT'}
    for L in net.layers:
        if L.kind not in allowed: return False
    return True


def extract_layers_for_lp(net, tf):
    """Walk net, extract per-ReLU info and post-ReLU affine composition.

    Returns (layers_data, output_affine, input_state) or None.
    - layers_data: list of dicts {composed_W, composed_b, pre_state}
        composed_W/b is the affine map from previous y (or input x) to z_k.
    - output_affine: {W, b} for post-last-ReLU composed dense → output.
    - input_state: FCHZState for INPUT_SPEC layer.
    """
    input_state = None
    for L in net.layers:
        if L.kind == 'INPUT_SPEC':
            input_state = tf._state_cache.get(L.id)
            break
        if L.kind == 'INPUT' and input_state is None:
            input_state = tf._state_cache.get(L.id)
    if input_state is None: return None

    layers_data = []
    dense_chain = []
    last_state = input_state
    for L in net.layers:
        if L.kind in ('INPUT', 'INPUT_SPEC', 'FLATTEN'):
            continue
        if L.kind == 'BIAS':
            b_l = L.params.get('c')
            if b_l is None: return None
            b_np = (b_l.detach().cpu().numpy().astype(np.float64).flatten()
                       if hasattr(b_l, 'detach')
                       else np.asarray(b_l, dtype=np.float64).flatten())
            dense_chain.append(('BIAS', None, b_np))
        elif L.kind == 'DENSE':
            W = L.params.get('weight')
            b_l = L.params.get('bias')
            if W is None: return None
            W_np = W.detach().cpu().numpy().astype(np.float64) if hasattr(W, 'detach') else np.asarray(W, dtype=np.float64)
            b_np = (b_l.detach().cpu().numpy().astype(np.float64).flatten()
                       if hasattr(b_l, 'detach') and b_l is not None
                       else (np.asarray(b_l, dtype=np.float64).flatten()
                                if b_l is not None else np.zeros(W_np.shape[0], dtype=np.float64)))
            dense_chain.append(('DENSE', W_np, b_np))
        elif L.kind == 'RELU':
            pred_id = net.preds.get(L.id, [None])[0]
            pre_state = tf._state_cache.get(pred_id)
            if pre_state is None: return None
            comp_W, comp_b = compose_chain(dense_chain, last_state.c.shape[0])
            layers_data.append({
                'composed_W': comp_W,
                'composed_b': comp_b,
                'pre_state': pre_state,
            })
            last_state = pre_state
            dense_chain = []
        elif L.kind == 'ASSERT':
            break

    # Remaining dense chain after last ReLU → output affine
    comp_W, comp_b = compose_chain(dense_chain, last_state.c.shape[0])
    output_affine = {'W': comp_W, 'b': comp_b}
    return layers_data, output_affine, input_state


def compose_chain(chain, in_dim):
    """Compose a list of (kind, W, b) into single affine y = W @ x + b."""
    W = np.eye(in_dim, dtype=np.float64)
    b = np.zeros(in_dim, dtype=np.float64)
    for kind, Wl, bl in chain:
        if kind == 'DENSE':
            b = Wl @ b + bl
            W = Wl @ W
        elif kind == 'BIAS':
            b = b + bl
    return W, b


def compute_pre_act_bounds(pre_state):
    """Return (l, u) from pre-ReLU FCHZState via interval closed-form."""
    c = pre_state.c
    G = pre_state.G
    tail = pre_state.tail_radius
    G_l1 = np.abs(G).sum(axis=1) if G is not None and G.size > 0 else np.zeros(c.shape[0])
    rad = G_l1 + (tail if tail is not None else 0.0)
    return c - rad, c + rad


def _build_lp(net, tf, d_out, sense='max'):
    """Build the LP constraint matrices.

    Args:
        d_out: direction vector (n_out,)
        sense: 'max' or 'min'

    Returns:
        (c_obj, A_ub, b_ub, A_eq, b_eq, var_bounds, const, total_vars) or None.
    """
    if not is_dense_only_chain(net):
        return None
    extract = extract_layers_for_lp(net, tf)
    if extract is None: return None
    layers_data, output_affine, input_state = extract
    if not layers_data: return None

    R = len(layers_data)
    K_in = input_state.G.shape[1] if input_state.G is not None else 0
    if K_in == 0: return None

    n_layers = [lyr['pre_state'].c.shape[0] for lyr in layers_data]

    # Pre-act bounds & ReLU masks per layer
    pre_bounds = []
    masks = []   # (active, inactive, unstable)
    for lyr in layers_data:
        l, u = compute_pre_act_bounds(lyr['pre_state'])
        pre_bounds.append((l, u))
        masks.append((l >= 0, u <= 0, (l < 0) & (u > 0)))

    # Variable layout: xi[K_in], y_layer_0[n_0], y_layer_1[n_1], ..., y_layer_{R-1}[n_{R-1}]
    y_offsets = [K_in]
    for n in n_layers:
        y_offsets.append(y_offsets[-1] + n)
    total = y_offsets[-1]

    var_bounds = []
    var_bounds.extend([(-1.0, 1.0)] * K_in)
    for k, n in enumerate(n_layers):
        a_mask, i_mask, u_mask = masks[k]
        l_pre, u_pre = pre_bounds[k]
        for i in range(n):
            if i_mask[i]:
                var_bounds.append((0.0, 0.0))
            elif a_mask[i]:
                var_bounds.append((max(0.0, l_pre[i]), max(0.0, u_pre[i])))
            else:
                var_bounds.append((0.0, max(0.0, u_pre[i])))

    A_eq, b_eq, A_ub, b_ub = [], [], [], []

    # Layer 0: z_0 = W_0 @ x + b_0; x = input_c + input_G @ xi
    W0 = layers_data[0]['composed_W']
    b0 = layers_data[0]['composed_b']
    G_in = input_state.G if input_state.G is not None else np.zeros((input_state.c.shape[0], K_in))
    z_xi = W0 @ G_in            # (n_0, K_in)
    z_const = W0 @ input_state.c + b0   # (n_0,)
    y0_off = y_offsets[0]
    for i in range(n_layers[0]):
        a_m, i_m, u_m = masks[0][0][i], masks[0][1][i], masks[0][2][i]
        if i_m: continue
        if a_m:
            row = np.zeros(total)
            row[0:K_in] = -z_xi[i]
            row[y0_off + i] = 1.0
            A_eq.append(row); b_eq.append(z_const[i])
        else:   # unstable
            l_pre_i = pre_bounds[0][0][i]
            u_pre_i = pre_bounds[0][1][i]
            lam = u_pre_i / max(u_pre_i - l_pre_i, 1e-300)
            # y >= z → z - y <= 0 → z_xi[i] @ xi - y <= -z_const[i]
            row = np.zeros(total)
            row[0:K_in] = z_xi[i]
            row[y0_off + i] = -1.0
            A_ub.append(row); b_ub.append(-z_const[i])
            # y <= lam*(z - l) → y - lam*z <= -lam*l
            #                       → -lam*z_xi[i]@xi + y <= -lam*l + lam*z_const[i]
            row = np.zeros(total)
            row[0:K_in] = -lam * z_xi[i]
            row[y0_off + i] = 1.0
            A_ub.append(row); b_ub.append(-lam * l_pre_i + lam * z_const[i])

    # Subsequent layers k=1..R-1: z_k = W_k @ y_{k-1} + b_k
    for k in range(1, R):
        W_k = layers_data[k]['composed_W']
        b_k = layers_data[k]['composed_b']
        y_prev_off = y_offsets[k - 1]
        y_curr_off = y_offsets[k]
        n_prev = n_layers[k - 1]
        for j in range(n_layers[k]):
            a_m, i_m, u_m = masks[k][0][j], masks[k][1][j], masks[k][2][j]
            if i_m: continue
            if a_m:
                row = np.zeros(total)
                row[y_prev_off:y_prev_off + n_prev] = -W_k[j]
                row[y_curr_off + j] = 1.0
                A_eq.append(row); b_eq.append(b_k[j])
            else:
                l_pre_j = pre_bounds[k][0][j]
                u_pre_j = pre_bounds[k][1][j]
                lam = u_pre_j / max(u_pre_j - l_pre_j, 1e-300)
                row = np.zeros(total)
                row[y_prev_off:y_prev_off + n_prev] = W_k[j]
                row[y_curr_off + j] = -1.0
                A_ub.append(row); b_ub.append(-b_k[j])
                row = np.zeros(total)
                row[y_prev_off:y_prev_off + n_prev] = -lam * W_k[j]
                row[y_curr_off + j] = 1.0
                A_ub.append(row); b_ub.append(-lam * l_pre_j + lam * b_k[j])

    # Output: z_out = output_affine['W'] @ y_{R-1} + output_affine['b']
    # Objective: max d @ z_out = d @ W_out @ y_{R-1} + d @ b_out
    W_out = output_affine['W']
    b_out = output_affine['b']
    d_eff = W_out.T @ d_out
    const = float(d_out @ b_out)

    c_obj = np.zeros(total)
    # linprog minimizes c @ x. For max d@z_out = max d_eff @ y_{R-1} + const,
    # set c = -d_eff (min negation). For min d@z_out, set c = +d_eff.
    sign = -1.0 if sense == 'max' else 1.0
    c_obj[y_offsets[R - 1]:y_offsets[R]] = sign * d_eff

    return c_obj, A_ub, b_ub, A_eq, b_eq, var_bounds, const, total, sense


def solve_full_lp(net, tf, d_out: np.ndarray, sense: str = 'max'):
    """Solve the full-network LP. sense='max' returns UB, sense='min' returns LB.

    Returns (value, dict_with_diagnostics). value is +inf for max-failure,
    -inf for min-failure.
    """
    built = _build_lp(net, tf, d_out, sense=sense)
    if built is None:
        return None, {'status': 'NOT_DENSE_ONLY'}
    c_obj, A_ub, b_ub, A_eq, b_eq, var_bounds, const, total, sense = built

    try:
        res = sopt.linprog(
            c=c_obj,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=var_bounds,
            method='highs',
        )
    except Exception as e:
        return ((float('inf') if sense == 'max' else float('-inf')),
                  {'status': 'EXCEPTION', 'exc': str(e)[:80]})

    if res.status != 0:
        return ((float('inf') if sense == 'max' else float('-inf')),
                  {'status': 'LP_FAIL', 'lp_status': res.status, 'lp_msg': res.message})

    # For max: value = const - res.fun (because we minimized -d_eff·y, so optimal value = -d_eff·y_opt = res.fun, then max d_eff·y = -res.fun)
    # For min: value = const + res.fun
    if sense == 'max':
        value = const - float(res.fun)
    else:
        value = const + float(res.fun)

    # Compute constraint residuals
    x = res.x
    A_ub_arr = np.array(A_ub) if A_ub else np.zeros((0, total))
    A_eq_arr = np.array(A_eq) if A_eq else np.zeros((0, total))
    ub_resid = float((A_ub_arr @ x - np.array(b_ub)).max()) if A_ub else 0.0
    eq_resid = float(np.abs(A_eq_arr @ x - np.array(b_eq)).max()) if A_eq else 0.0

    return value, {
        'status': 'OK',
        'lp_status': res.status,
        'sense': sense,
        'const': const,
        'res_fun': float(res.fun),
        'ub_resid': ub_resid,
        'eq_resid': eq_resid,
        'n_vars': total,
        'n_ub': len(A_ub), 'n_eq': len(A_eq),
    }


def solve_full_lp_ub(net, tf, d_out: np.ndarray):
    """Compute LP UB on d_out @ output. Returns scalar or +inf on failure."""
    val, _diag = solve_full_lp(net, tf, d_out, sense='max')
    return val if val is not None else float('inf')


def solve_full_lp_lb(net, tf, d_out: np.ndarray):
    """Compute LP LB on d_out @ output DIRECTLY (min d_out @ output).

    Per advisor: prefer direct min over -max(-d) trick for UNSAFE_LINEAR.
    Returns scalar or -inf on failure.
    """
    val, _diag = solve_full_lp(net, tf, d_out, sense='min')
    return val if val is not None else float('-inf')


def m4_full_lp_refine_ub(net, tf, C: np.ndarray, thresholds: np.ndarray):
    """Refine UB per spec row using full-network LP. (M,) array, +inf on fail."""
    if not is_dense_only_chain(net): return None
    M = C.shape[0]
    refined = np.full(M, np.inf, dtype=np.float64)
    for i in range(M):
        try:
            ub, _ = solve_full_lp(net, tf, C[i].astype(np.float64), sense='max')
            if ub is not None: refined[i] = ub
        except Exception:
            pass
    return refined


def m4_full_lp_refine_lb(net, tf, C: np.ndarray, thresholds: np.ndarray):
    """Refine LB per spec row using full-network LP DIRECTLY. (M,) array, -inf on fail."""
    if not is_dense_only_chain(net): return None
    M = C.shape[0]
    refined = np.full(M, -np.inf, dtype=np.float64)
    for i in range(M):
        try:
            lb, _ = solve_full_lp(net, tf, C[i].astype(np.float64), sense='min')
            if lb is not None: refined[i] = lb
        except Exception:
            pass
    return refined


# Back-compat alias used by older scripts
def solve_full_lp_ub_v2(net, tf, d_out):
    return solve_full_lp_ub(net, tf, d_out)


def m4_full_lp_refine(net, tf, C, thresholds):
    """Back-compat alias: same as m4_full_lp_refine_ub."""
    return m4_full_lp_refine_ub(net, tf, C, thresholds)
