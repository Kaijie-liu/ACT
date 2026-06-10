# ===- research/fchz/m4b_lp_with_head.py - M4B LP with affine head ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Extension of M4 full-network LP for "dense body + affine head" topology.
#
# The body LP is built the same way (xi + y_layer variables, ReLU triangles).
# The objective is extended:
#     output = W_xi_head @ xi + W_y_head @ y_{R-1} + b_head
#     d_out @ output = (d_out @ W_xi_head) @ xi + (d_out @ W_y_head) @ y_{R-1} + d_out @ b_head
#
# In LP: c_obj has nonzero on BOTH xi region AND y_R region.
# ===---------------------------------------------------------------------===#

"""M4B LP solver with affine head support."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'research/sc_hz'))

import numpy as np
import scipy.optimize as sopt


def _extract_body_only(net, tf, last_relu_id):
    """Extract body (up to last RELU). Output affine NOT computed (head handles)."""
    from research.fchz.m4_full_lp import compose_chain

    input_state = None
    for L in net.layers:
        if L.kind == 'INPUT_SPEC':
            input_state = tf._state_cache.get(L.id); break
        if L.kind == 'INPUT' and input_state is None:
            input_state = tf._state_cache.get(L.id)
    if input_state is None: return None

    layers_data = []
    dense_chain = []
    last_state = input_state
    for L in net.layers:
        if L.id > last_relu_id: break
        if L.kind in ('INPUT', 'INPUT_SPEC', 'FLATTEN'):
            continue
        if L.kind == 'BIAS':
            b_l = L.params.get('c')
            if b_l is None: return None
            b_np = (b_l.detach().cpu().numpy().astype(np.float64).flatten()
                       if hasattr(b_l, 'detach') else np.asarray(b_l, dtype=np.float64).flatten())
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
                'composed_W': comp_W, 'composed_b': comp_b, 'pre_state': pre_state,
            })
            last_state = pre_state
            dense_chain = []
        else:
            return None
    return layers_data, input_state


def solve_lp_with_head(net, tf, d_out, sense='max'):
    """Solve LP for d_out @ final_output, using affine head extractor.

    Returns (value, diagnostics) where value is +inf for max-failure, -inf for min-failure.
    """
    from research.fchz.m4_full_lp import (
        extract_layers_for_lp, compute_pre_act_bounds, is_dense_only_chain)
    from research.fchz.m4b_affine_head import (
        is_dense_body_with_affine_head, extract_affine_head)

    # Body extraction (need dense_only OR dense_body+affine_head)
    body_dense_only = is_dense_only_chain(net)
    has_head, last_relu_id = is_dense_body_with_affine_head(net)
    if not (body_dense_only or has_head):
        return None, {'status': 'NOT_SUPPORTED'}

    if has_head and last_relu_id is not None:
        # Use body-only extractor (doesn't try to compose multi-branch head)
        extract = _extract_body_only(net, tf, last_relu_id)
        if extract is None: return None, {'status': 'BODY_EXTRACT_FAIL'}
        layers_data, input_state = extract
    else:
        extract = extract_layers_for_lp(net, tf)
        if extract is None: return None, {'status': 'EXTRACT_FAIL'}
        layers_data, output_affine, input_state = extract

    if not layers_data: return None, {'status': 'NO_LAYERS'}

    R = len(layers_data)
    K_in = input_state.G.shape[1] if input_state.G is not None else 0
    if K_in == 0: return None, {'status': 'NO_XI'}

    n_layers = [lyr['pre_state'].c.shape[0] for lyr in layers_data]
    n_R = n_layers[-1]

    # Get affine head: (W_xi_head, W_y_head, b_head) where final = W_xi @ xi + W_y @ y_R + b
    if has_head:
        head = extract_affine_head(net, tf, last_relu_id, n_R, K_in, input_state)
        if head is None: return None, {'status': 'HEAD_EXTRACT_FAIL'}
        W_xi_head, W_y_head, b_head = head
    else:
        # body_dense_only — output_affine is just W @ y_R + b
        W_y_head = output_affine['W']
        W_xi_head = np.zeros((W_y_head.shape[0], K_in), dtype=np.float64)
        b_head = output_affine['b']

    # Pre-act bounds & masks per layer
    pre_bounds = []
    masks = []
    for lyr in layers_data:
        l, u = compute_pre_act_bounds(lyr['pre_state'])
        pre_bounds.append((l, u))
        masks.append((l >= 0, u <= 0, (l < 0) & (u > 0)))

    # Variable layout: xi[K_in], y_layer_0, ..., y_layer_{R-1}
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

    # Layer 0: z_0 = W_0 @ x + b_0, x = c + G @ xi
    W0 = layers_data[0]['composed_W']
    b0 = layers_data[0]['composed_b']
    G_in = input_state.G if input_state.G is not None else np.zeros((input_state.c.shape[0], K_in))
    z_xi = W0 @ G_in
    z_const = W0 @ input_state.c + b0
    y0_off = y_offsets[0]
    for i in range(n_layers[0]):
        a_m, i_m, u_m = masks[0][0][i], masks[0][1][i], masks[0][2][i]
        if i_m: continue
        if a_m:
            row = np.zeros(total)
            row[0:K_in] = -z_xi[i]
            row[y0_off + i] = 1.0
            A_eq.append(row); b_eq.append(z_const[i])
        else:
            l_pre_i = pre_bounds[0][0][i]
            u_pre_i = pre_bounds[0][1][i]
            lam = u_pre_i / max(u_pre_i - l_pre_i, 1e-300)
            row = np.zeros(total)
            row[0:K_in] = z_xi[i]
            row[y0_off + i] = -1.0
            A_ub.append(row); b_ub.append(-z_const[i])
            row = np.zeros(total)
            row[0:K_in] = -lam * z_xi[i]
            row[y0_off + i] = 1.0
            A_ub.append(row); b_ub.append(-lam * l_pre_i + lam * z_const[i])

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
                A_ub.append(row); b_ub.append(-lam * pre_bounds[k][0][j] + lam * b_k[j])

    # Objective: d_out @ output = (d_out @ W_xi_head) @ xi + (d_out @ W_y_head) @ y_R + const
    d_xi = W_xi_head.T @ d_out    # (K_in,)
    d_y = W_y_head.T @ d_out      # (n_R,)
    const = float(d_out @ b_head)

    c_obj = np.zeros(total)
    sign = -1.0 if sense == 'max' else 1.0
    c_obj[0:K_in] = sign * d_xi
    c_obj[y_offsets[R - 1]:y_offsets[R]] = sign * d_y

    try:
        res = sopt.linprog(
            c=c_obj,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=var_bounds, method='highs',
        )
    except Exception as e:
        return ((float('inf') if sense == 'max' else float('-inf')),
                  {'status': 'EXCEPTION', 'exc': str(e)[:80]})
    if res.status != 0:
        return ((float('inf') if sense == 'max' else float('-inf')),
                  {'status': 'LP_FAIL', 'lp_status': res.status, 'lp_msg': res.message})
    val = const - float(res.fun) if sense == 'max' else const + float(res.fun)
    return val, {
        'status': 'OK',
        'lp_status': res.status,
        'sense': sense, 'const': const, 'res_fun': float(res.fun),
        'n_vars': total, 'n_ub': len(A_ub), 'n_eq': len(A_eq),
        'has_affine_head': True,
    }


def m4b_refine_lb(net, tf, C, t):
    """Per-row LB refinement using LP-with-affine-head."""
    M = C.shape[0]
    refined = np.full(M, -np.inf, dtype=np.float64)
    for i in range(M):
        try:
            lb, _ = solve_lp_with_head(net, tf, C[i].astype(np.float64), sense='min')
            if lb is not None: refined[i] = lb
        except Exception: pass
    return refined


def m4b_refine_ub(net, tf, C, t):
    M = C.shape[0]
    refined = np.full(M, np.inf, dtype=np.float64)
    for i in range(M):
        try:
            ub, _ = solve_lp_with_head(net, tf, C[i].astype(np.float64), sense='max')
            if ub is not None: refined[i] = ub
        except Exception: pass
    return refined
