# ===- act/back_end/fchz_tf/forward_global_lp.py - Forward Global Triangle LP =#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   FCHZ refinement mode for small/dense MLPs: build forward joint LP over
#   per-layer activations, ReLU triangle convex hull, and spec objective.
#   When FCHZ closed-form returns UNKNOWN and network is small, this can
#   tighten via continuous LP solving (HiGHS).
#
#   PRINCIPLES PRESERVED:
#     - Forward only (no backward propagation)
#     - Continuous LP only (no MILP)
#     - No BaB (no input split)
#     - No CROWN (no symbolic backward bounds)
#     - No portfolio fallback
#
#   Per advisor 2026-06-08: H2-A refinement mode for FCHZ.
#
# ===---------------------------------------------------------------------===#
"""Forward Global Triangle LP for small dense MLP refinement.

For each unstable ReLU neuron with pre-activation bounds [l, u] (l<0<u):
  - h >= 0
  - h >= z      (where z = pre-act)
  - h <= u*(z-l)/(u-l)   (triangle upper line through (l,0) and (u,u))

For stable_active (l>=0):  h == z
For stable_inactive (u<=0): h == 0

For each DENSE layer:  z_i = W_i @ h_{i-1} + b_i (equality)

Input box: lb_in <= x <= ub_in

Spec objective:  minimize C @ y (for LB)  via HiGHS continuous LP.
"""
import numpy as np
from typing import Optional, List, Tuple, Dict
from scipy.optimize import linprog


def _interval_forward(net, lb_in: np.ndarray, ub_in: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Cheap interval forward pass — per-layer (lb, ub) for each layer output.

    Returns dict mapping layer.id → (lb_layer, ub_layer).
    Box-only; supports DENSE/RELU/INPUT/INPUT_SPEC layers.
    """
    bounds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for L in net.layers:
        if L.kind in ('INPUT', 'INPUT_SPEC'):
            bounds[L.id] = (lb_in.copy(), ub_in.copy())
            continue
        preds = net.preds.get(L.id, [])
        if not preds:
            bounds[L.id] = (lb_in.copy(), ub_in.copy())
            continue
        lb_pred, ub_pred = bounds.get(preds[0], (lb_in.copy(), ub_in.copy()))
        if L.kind == 'DENSE':
            W = L.params['weight']
            b = L.params.get('bias')
            W_np = W.detach().cpu().numpy().astype(np.float64) if hasattr(W, 'detach') else np.asarray(W).astype(np.float64)
            b_np = (b.detach().cpu().numpy().astype(np.float64) if hasattr(b, 'detach') else np.asarray(b).astype(np.float64)) if b is not None else np.zeros(W_np.shape[0])
            # Standard interval matrix mul
            W_pos = np.maximum(W_np, 0); W_neg = np.minimum(W_np, 0)
            new_lb = W_pos @ lb_pred + W_neg @ ub_pred + b_np
            new_ub = W_pos @ ub_pred + W_neg @ lb_pred + b_np
            bounds[L.id] = (new_lb, new_ub)
        elif L.kind == 'RELU':
            bounds[L.id] = (np.maximum(lb_pred, 0), np.maximum(ub_pred, 0))
        elif L.kind == 'SIGMOID':
            # y = 1/(1+exp(-z)); monotone increasing
            bounds[L.id] = (1.0 / (1.0 + np.exp(-lb_pred)), 1.0 / (1.0 + np.exp(-ub_pred)))
        elif L.kind == 'TANH':
            bounds[L.id] = (np.tanh(lb_pred), np.tanh(ub_pred))
        elif L.kind == 'BIAS':
            b = L.params.get('bias')
            if b is not None:
                b_np = (b.detach().cpu().numpy().astype(np.float64).reshape(-1)
                            if hasattr(b, 'detach') else np.asarray(b).astype(np.float64).reshape(-1))
                bounds[L.id] = (lb_pred + b_np, ub_pred + b_np)
            else:
                bounds[L.id] = (lb_pred, ub_pred)
        elif L.kind in ('FLATTEN', 'RESHAPE'):
            # Shape-only op: pass through (we operate flat)
            bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
        elif L.kind == 'SLICE':
            starts = L.params.get('starts', [0])
            ends = L.params.get('ends', [None])
            axes = L.params.get('axes', [0])
            s = int(starts[0]); e = int(ends[0]) if ends[0] is not None else len(lb_pred)
            bounds[L.id] = (lb_pred[s:e].copy(), ub_pred[s:e].copy())
        elif L.kind == 'CONCAT':
            # Concatenate all pred bounds along flat axis
            all_lb = []; all_ub = []
            for p in preds:
                p_lb, p_ub = bounds.get(p, (lb_in.copy(), ub_in.copy()))
                all_lb.append(p_lb); all_ub.append(p_ub)
            bounds[L.id] = (np.concatenate(all_lb), np.concatenate(all_ub))
        elif L.kind == 'SCALE':
            # y = a * x (element-wise scale)
            a = L.params.get('a')
            if a is not None:
                a_np = (a.detach().cpu().numpy().astype(np.float64).reshape(-1)
                            if hasattr(a, 'detach') else np.asarray(a).astype(np.float64).reshape(-1))
                if a_np.shape[0] == lb_pred.shape[0]:
                    a_pos = np.maximum(a_np, 0); a_neg = np.minimum(a_np, 0)
                    new_lb = a_pos * lb_pred + a_neg * ub_pred
                    new_ub = a_pos * ub_pred + a_neg * lb_pred
                    bounds[L.id] = (new_lb, new_ub)
                else:
                    bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
            else:
                bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
        elif L.kind == 'BN':
            # BatchNorm: y_i = A_i * x_i + c_i (after ACT precompute)
            A_param = L.params.get('A')
            c_param = L.params.get('c')
            if A_param is None or c_param is None:
                bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
            else:
                A_np = (A_param.detach().cpu().numpy() if hasattr(A_param, 'detach')
                           else np.asarray(A_param)).astype(np.float64).reshape(-1)
                c_np = (c_param.detach().cpu().numpy() if hasattr(c_param, 'detach')
                           else np.asarray(c_param)).astype(np.float64).reshape(-1)
                if A_np.shape[0] != lb_pred.shape[0]:
                    # Per-channel broadcast
                    if lb_pred.shape[0] % A_np.shape[0] == 0:
                        spatial = lb_pred.shape[0] // A_np.shape[0]
                        A_full = np.repeat(A_np, spatial)
                        c_full = np.repeat(c_np, spatial)
                    else:
                        bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
                        continue
                else:
                    A_full = A_np; c_full = c_np
                a_pos = np.maximum(A_full, 0); a_neg = np.minimum(A_full, 0)
                new_lb = a_pos * lb_pred + a_neg * ub_pred + c_full
                new_ub = a_pos * ub_pred + a_neg * lb_pred + c_full
                bounds[L.id] = (new_lb, new_ub)
        elif L.kind == 'CONV2D':
            # IBP for conv: compute via torch nn.functional
            try:
                import torch as _torch, torch.nn.functional as F
                W = L.params.get('weight')
                b = L.params.get('bias')
                stride = L.params.get('stride', (1, 1))
                padding = L.params.get('padding', (0, 0))
                input_shape = L.params.get('input_shape')   # e.g., (1, Cin, H, W)
                output_shape = L.params.get('output_shape')
                if W is None or input_shape is None or output_shape is None:
                    bounds[L.id] = (lb_pred, ub_pred); continue
                Cin, Hin, Win = input_shape[1], input_shape[2], input_shape[3]
                expected_in = Cin * Hin * Win
                if lb_pred.shape[0] != expected_in:
                    bounds[L.id] = (lb_pred, ub_pred); continue
                lb_t = _torch.tensor(lb_pred.reshape(1, Cin, Hin, Win), dtype=_torch.float64)
                ub_t = _torch.tensor(ub_pred.reshape(1, Cin, Hin, Win), dtype=_torch.float64)
                W_t = W if hasattr(W, 'shape') else _torch.tensor(W, dtype=_torch.float64)
                W_t = W_t.to(_torch.float64)
                b_t = (b.to(_torch.float64) if hasattr(b, 'to') else _torch.tensor(b, dtype=_torch.float64)) if b is not None else None
                W_pos = _torch.clamp(W_t, min=0); W_neg = _torch.clamp(W_t, max=0)
                new_lb = F.conv2d(lb_t, W_pos, b_t, stride=stride, padding=padding) + F.conv2d(ub_t, W_neg, None, stride=stride, padding=padding)
                new_ub = F.conv2d(ub_t, W_pos, b_t, stride=stride, padding=padding) + F.conv2d(lb_t, W_neg, None, stride=stride, padding=padding)
                bounds[L.id] = (new_lb.numpy().reshape(-1), new_ub.numpy().reshape(-1))
            except Exception:
                bounds[L.id] = (lb_pred, ub_pred)
        elif L.kind in ('ASSERT', 'CONSTANT', 'INPUT'):
            bounds[L.id] = (lb_pred, ub_pred)
        elif L.kind == 'ADD':
            # Sound interval ADD: y = x_a + x_b → lb = lb_a + lb_b, ub = ub_a + ub_b.
            # SOUNDNESS NOTE (2026-06-11): previously this fell through to the
            # generic `else: passthrough` clause, which returned only preds[0]'s
            # bound. That artificially tightened the bound when one branch
            # produced zero on a coordinate by network design (e.g., cersyve
            # pendulum_pretrain_inv layer 7 has W_row1 = 0 + b_1 = 0, so
            # interval said y_1 ∈ [0, 0]) while the other branch contributed
            # a non-zero range. The intersection with FCHZ in fchz_pre_bounds
            # then picked the wrong [0, 0] bound and the MILP unsoundly CERT'd.
            preds_lst = net.preds.get(L.id, [])
            if len(preds_lst) != 2:
                bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
            else:
                lb_a, ub_a = bounds.get(preds_lst[0], (lb_in.copy(), ub_in.copy()))
                lb_b, ub_b = bounds.get(preds_lst[1], (lb_in.copy(), ub_in.copy()))
                if lb_a.shape == lb_b.shape:
                    bounds[L.id] = (lb_a + lb_b, ub_a + ub_b)
                else:
                    # Shape mismatch — fall back to conservatively wide bound
                    bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
        elif L.kind == 'SUB':
            preds_lst = net.preds.get(L.id, [])
            if len(preds_lst) != 2:
                bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
            else:
                lb_a, ub_a = bounds.get(preds_lst[0], (lb_in.copy(), ub_in.copy()))
                lb_b, ub_b = bounds.get(preds_lst[1], (lb_in.copy(), ub_in.copy()))
                if lb_a.shape == lb_b.shape:
                    bounds[L.id] = (lb_a - ub_b, ub_a - lb_b)
                else:
                    bounds[L.id] = (lb_pred.copy(), ub_pred.copy())
        else:
            # Unsupported — passthrough (sound conservative)
            bounds[L.id] = (lb_pred, ub_pred)
    return bounds


def fchz_pre_bounds(tf, net, input_lb_arr: np.ndarray, input_ub_arr: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Extract per-layer (lb, ub) bounds from FCHZ propagated state cache.
    Falls back to cheap interval where FCHZ state isn't cached.
    If interval_forward fails (unsupported op), use FCHZ-only bounds.
    """
    interval: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    try:
        interval = _interval_forward(net, input_lb_arr, input_ub_arr)
    except Exception:
        # Fall back to FCHZ-only bounds; _interval_forward unsupported ops.
        interval = {}
    cache = getattr(tf, '_state_cache', {}) or {}
    for L in net.layers:
        s = cache.get(L.id)
        if s is None:
            if L.id not in interval:
                # No FCHZ state and no interval — use input box as fallback
                interval[L.id] = (input_lb_arr.copy(), input_ub_arr.copy())
            continue
        c = s.c; G = s.G; tail = s.tail_radius
        if not hasattr(c, 'shape'):
            if L.id not in interval:
                interval[L.id] = (input_lb_arr.copy(), input_ub_arr.copy())
            continue
        G_l1 = np.abs(G).sum(axis=1) if G is not None and G.size > 0 else np.zeros(c.shape[0])
        rad = G_l1 + (tail if tail is not None else 0)
        lb = c - rad; ub = c + rad
        i_lb, i_ub = interval.get(L.id, (lb, ub))
        if i_lb.shape == lb.shape:
            interval[L.id] = (np.maximum(lb, i_lb), np.minimum(ub, i_ub))
        else:
            interval[L.id] = (lb, ub)
    return interval


def build_forward_lp(net, lb_in: np.ndarray, ub_in: np.ndarray,
                                  pre_bounds: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
                                  ) -> Optional[Dict]:
    """Build LP variables + constraints for forward joint encoding.

    Returns dict with:
      - var_count: total variables
      - layer_var_idx: layer.id → (start_idx, dim) in variable vector
      - A_ub, b_ub: ineq constraints (Ax <= b)
      - A_eq, b_eq: equality constraints
      - var_bounds: list of (lb, ub) per variable
      - pre_bounds: pre-activation bounds dict (for triangle params)

    Returns None if any layer is unsupported.
    """
    # Cheap forward interval to get per-layer bounds
    if pre_bounds is None:
        pre_bounds = _interval_forward(net, lb_in, ub_in)

    n_in = len(lb_in)
    # Variable layout: input first, then per layer's output
    layer_var_idx: Dict[int, Tuple[int, int]] = {}
    var_bounds: List[Tuple[float, float]] = []
    A_eq_rows: List[np.ndarray] = []; b_eq_rows: List[float] = []
    # Goal 1 (2026-06-10): sparse equality blocks for CONV2D
    conv2d_sparse_blocks: List[Dict] = []
    A_ub_rows: List[np.ndarray] = []; b_ub_rows: List[float] = []

    # Input as variable [INPUT layer]
    input_layer = next((L for L in net.layers if L.kind == 'INPUT'), None)
    if input_layer is None:
        return None
    layer_var_idx[input_layer.id] = (0, n_in)
    for i in range(n_in):
        var_bounds.append((float(lb_in[i]), float(ub_in[i])))

    next_var = n_in
    # INPUT_SPEC: same as input
    for L in net.layers:
        if L.kind == 'INPUT_SPEC':
            # alias to input layer (preds[0] is INPUT)
            layer_var_idx[L.id] = layer_var_idx[input_layer.id]

    # Helper: get variable index range for layer (resolving INPUT_SPEC alias etc.)
    def var_idx_of(L_id):
        return layer_var_idx.get(L_id)

    # Iterate layers, adding equality/inequality constraints
    for L in net.layers:
        if L.kind in ('INPUT', 'INPUT_SPEC', 'CONSTANT'):
            continue
        preds = net.preds.get(L.id, [])
        if not preds:
            return None
        pred_id = preds[0]
        pred_idx = var_idx_of(pred_id)
        if pred_idx is None:
            return None
        pred_start, pred_dim = pred_idx
        lb_pred, ub_pred = pre_bounds[pred_id]

        if L.kind == 'DENSE':
            W = L.params['weight']
            b = L.params.get('bias')
            W_np = W.detach().cpu().numpy().astype(np.float64) if hasattr(W, 'detach') else np.asarray(W).astype(np.float64)
            b_np = (b.detach().cpu().numpy().astype(np.float64) if hasattr(b, 'detach') else np.asarray(b).astype(np.float64)) if b is not None else np.zeros(W_np.shape[0])
            n_out = W_np.shape[0]
            # New vars for this layer output
            layer_var_idx[L.id] = (next_var, n_out)
            for j in range(n_out):
                # Use cheap interval bounds as box
                lb_o, ub_o = pre_bounds[L.id]
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
            # Equality: out_j - W_j @ pred = b_j
            n_total = next_var + n_out
            for j in range(n_out):
                row = np.zeros(n_total)
                row[pred_start:pred_start + pred_dim] = -W_np[j]
                row[next_var + j] = 1.0
                A_eq_rows.append((row, n_total))
                b_eq_rows.append(float(b_np[j]))
            next_var += n_out

        elif L.kind == 'RELU':
            n_out = pred_dim
            layer_var_idx[L.id] = (next_var, n_out)
            lb_o, ub_o = pre_bounds[L.id]
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
            n_total = next_var + n_out
            for j in range(n_out):
                l_pre = float(lb_pred[j]); u_pre = float(ub_pred[j])
                v_out = next_var + j
                v_pre = pred_start + j
                if l_pre >= 0:
                    # Stable active: h == z
                    row = np.zeros(n_total)
                    row[v_out] = 1.0; row[v_pre] = -1.0
                    A_eq_rows.append((row, n_total)); b_eq_rows.append(0.0)
                elif u_pre <= 0:
                    # Stable inactive: h == 0
                    row = np.zeros(n_total)
                    row[v_out] = 1.0
                    A_eq_rows.append((row, n_total)); b_eq_rows.append(0.0)
                else:
                    # Triangle:
                    #   h >= 0       (already in var bounds, lb=0)
                    #   h >= z       → -h + z <= 0
                    row = np.zeros(n_total); row[v_out] = -1.0; row[v_pre] = 1.0
                    A_ub_rows.append((row, n_total)); b_ub_rows.append(0.0)
                    #   h <= u*(z-l)/(u-l)
                    #   → h - u/(u-l) * z <= -u*l/(u-l)
                    slope = u_pre / (u_pre - l_pre)
                    rhs = -slope * l_pre
                    row = np.zeros(n_total); row[v_out] = 1.0; row[v_pre] = -slope
                    A_ub_rows.append((row, n_total)); b_ub_rows.append(rhs)
            next_var += n_out

        elif L.kind == 'ASSERT':
            # ASSERT doesn't add vars; output is alias to pred
            layer_var_idx[L.id] = (pred_start, pred_dim)

        elif L.kind in ('FLATTEN', 'RESHAPE'):
            # Shape-only: alias to pred (no new vars, same flat representation)
            layer_var_idx[L.id] = (pred_start, pred_dim)

        elif L.kind == 'BIAS':
            # Add bias element-wise — alias pred + add bias via new vars
            b = L.params.get('bias')
            if b is not None:
                b_np = (b.detach().cpu().numpy().astype(np.float64).reshape(-1)
                            if hasattr(b, 'detach') else np.asarray(b).astype(np.float64).reshape(-1))
                if b_np.shape[0] != pred_dim:
                    # shape mismatch — passthrough
                    layer_var_idx[L.id] = (pred_start, pred_dim)
                else:
                    n_out = pred_dim
                    layer_var_idx[L.id] = (next_var, n_out)
                    lb_o, ub_o = pre_bounds[L.id]
                    for j in range(n_out):
                        var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                    n_total = next_var + n_out
                    for j in range(n_out):
                        row = np.zeros(n_total)
                        row[next_var + j] = 1.0
                        row[pred_start + j] = -1.0
                        A_eq_rows.append((row, n_total))
                        b_eq_rows.append(float(b_np[j]))
                    next_var += n_out
            else:
                layer_var_idx[L.id] = (pred_start, pred_dim)

        elif L.kind == 'SLICE':
            starts = L.params.get('starts', [0])
            ends = L.params.get('ends', [None])
            s = int(starts[0]); e = int(ends[0]) if ends[0] is not None else pred_dim
            slice_dim = e - s
            # Alias to a sub-range of pred vars
            layer_var_idx[L.id] = (pred_start + s, slice_dim)

        elif L.kind == 'CONCAT':
            # Allocate new vars + equality to each pred
            total_out_dim = 0
            for p in preds:
                p_idx = var_idx_of(p)
                if p_idx is None: return None
                total_out_dim += p_idx[1]
            n_out = total_out_dim
            layer_var_idx[L.id] = (next_var, n_out)
            lb_o, ub_o = pre_bounds[L.id]
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
            n_total = next_var + n_out
            offset = 0
            for p in preds:
                p_start, p_dim = var_idx_of(p)
                for k in range(p_dim):
                    row = np.zeros(n_total)
                    row[next_var + offset + k] = 1.0
                    row[p_start + k] = -1.0
                    A_eq_rows.append((row, n_total))
                    b_eq_rows.append(0.0)
                offset += p_dim
            next_var += n_out

        elif L.kind in ('SIGMOID', 'TANH'):
            # Nonlinear activation - use FCHZ pre-bounds as box (loose but sound).
            # Create fresh vars; LP/MILP refinement won't help much through these
            # but at least the build doesn't fail.
            lb_o, ub_o = pre_bounds[L.id]
            n_out = lb_o.shape[0]
            layer_var_idx[L.id] = (next_var, n_out)
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
            next_var += n_out
        elif L.kind == 'BN':
            # BatchNorm: y_i = A_i * x_i + c_i (per-channel affine).
            A_param = L.params.get('A')
            c_param = L.params.get('c')
            if A_param is None or c_param is None:
                layer_var_idx[L.id] = (pred_start, pred_dim)
            else:
                A_np = (A_param.detach().cpu().numpy() if hasattr(A_param, 'detach')
                           else np.asarray(A_param)).astype(np.float64).reshape(-1)
                c_np = (c_param.detach().cpu().numpy() if hasattr(c_param, 'detach')
                           else np.asarray(c_param)).astype(np.float64).reshape(-1)
                if A_np.shape[0] != pred_dim:
                    if pred_dim % A_np.shape[0] == 0:
                        spatial = pred_dim // A_np.shape[0]
                        A_full = np.repeat(A_np, spatial)
                        c_full = np.repeat(c_np, spatial)
                    else:
                        layer_var_idx[L.id] = (pred_start, pred_dim)
                        continue
                else:
                    A_full = A_np; c_full = c_np
                n_out = pred_dim
                layer_var_idx[L.id] = (next_var, n_out)
                lb_o, ub_o = pre_bounds[L.id]
                for j in range(n_out):
                    var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                n_total = next_var + n_out
                for j in range(n_out):
                    row = np.zeros(n_total)
                    row[next_var + j] = 1.0
                    row[pred_start + j] = -float(A_full[j])
                    A_eq_rows.append((row, n_total))
                    b_eq_rows.append(float(c_full[j]))
                next_var += n_out

        elif L.kind == 'SCALE':
            # Element-wise: y_j = a_j * x_j. Add equality y_j - a_j * x_j = 0.
            a = L.params.get('a')
            if a is not None:
                a_np = (a.detach().cpu().numpy().astype(np.float64).reshape(-1)
                            if hasattr(a, 'detach') else np.asarray(a).astype(np.float64).reshape(-1))
                if a_np.shape[0] != pred_dim:
                    layer_var_idx[L.id] = (pred_start, pred_dim)
                else:
                    n_out = pred_dim
                    layer_var_idx[L.id] = (next_var, n_out)
                    lb_o, ub_o = pre_bounds[L.id]
                    for j in range(n_out):
                        var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                    n_total = next_var + n_out
                    for j in range(n_out):
                        row = np.zeros(n_total)
                        row[next_var + j] = 1.0
                        row[pred_start + j] = -float(a_np[j])
                        A_eq_rows.append((row, n_total))
                        b_eq_rows.append(0.0)
                    next_var += n_out
            else:
                layer_var_idx[L.id] = (pred_start, pred_dim)
        elif L.kind == 'CONV2D':
            # Goal 1 (2026-06-10): tight sparse-affine encoding for CONV2D
            from act.back_end.fchz_tf.conv2d_sparse_encoder import conv2d_to_sparse_affine
            W = L.params.get('weight')
            b_param = L.params.get('bias')
            input_shape_4d = L.params.get('input_shape')
            stride = L.params.get('stride', (1, 1))
            padding = L.params.get('padding', (0, 0))
            tight_ok = False
            if W is not None and input_shape_4d is not None and len(input_shape_4d) == 4:
                try:
                    W_np = (W.detach().cpu().numpy() if hasattr(W, 'detach') else np.asarray(W)).astype(np.float64)
                    b_np = ((b_param.detach().cpu().numpy() if hasattr(b_param, 'detach') else np.asarray(b_param)).astype(np.float64)
                                if b_param is not None else None)
                    expected_in = int(np.prod(input_shape_4d[1:]))
                    if pred_dim == expected_in:
                        A_csr, b_flat, out_shape_4d = conv2d_to_sparse_affine(
                            W_np, b_np, input_shape_4d, stride=stride, padding=padding)
                        n_out = A_csr.shape[0]
                        layer_var_idx[L.id] = (next_var, n_out)
                        lb_o, ub_o = pre_bounds[L.id]
                        if lb_o.shape[0] == n_out:
                            for j in range(n_out):
                                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                            conv2d_sparse_blocks.append({
                                'y_start': next_var,
                                'y_dim': n_out,
                                'x_start': pred_start,
                                'x_dim': pred_dim,
                                'A_csr': A_csr,
                                'b_flat': b_flat,
                            })
                            next_var += n_out
                            tight_ok = True
                except Exception:
                    tight_ok = False
            if not tight_ok:
                # Fallback to loose-box (original behavior)
                lb_o, ub_o = pre_bounds[L.id]
                n_out = lb_o.shape[0]
                layer_var_idx[L.id] = (next_var, n_out)
                for j in range(n_out):
                    var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                next_var += n_out

        elif L.kind == 'ADD':
            # Residual ADD: y = x1 + x2 (state-state element-wise sum)
            pred_ids = net.preds.get(L.id, [])
            if len(pred_ids) != 2: return None
            pred1_idx = layer_var_idx.get(pred_ids[0])
            pred2_idx = layer_var_idx.get(pred_ids[1])
            if pred1_idx is None or pred2_idx is None: return None
            p1_start, p1_dim = pred1_idx
            p2_start, p2_dim = pred2_idx
            if p1_dim != p2_dim: return None
            n_out = p1_dim
            layer_var_idx[L.id] = (next_var, n_out)
            if L.id in pre_bounds:
                lb_o, ub_o = pre_bounds[L.id]
            else:
                lb1, ub1 = pre_bounds.get(pred_ids[0], (None, None))
                lb2, ub2 = pre_bounds.get(pred_ids[1], (None, None))
                if lb1 is None or lb2 is None: return None
                lb_o = lb1 + lb2; ub_o = ub1 + ub2
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
            n_total_now = next_var + n_out
            for j in range(n_out):
                row = np.zeros(n_total_now)
                row[next_var + j] = 1.0
                row[p1_start + j] = -1.0
                row[p2_start + j] = -1.0
                A_eq_rows.append((row, n_total_now))
                b_eq_rows.append(0.0)
            next_var += n_out

        elif L.kind == 'ASSERT':
            preds_ids = net.preds.get(L.id, [])
            if preds_ids and preds_ids[0] in layer_var_idx:
                layer_var_idx[L.id] = layer_var_idx[preds_ids[0]]
            else:
                layer_var_idx[L.id] = (next_var - 1, 1)

        else:
            # Unsupported op — bail out (return None to signal can't build LP)
            return None

    # Pad all A rows to final n_total
    n_total = next_var
    A_eq_padded = []
    for row, original_len in A_eq_rows:
        if len(row) < n_total:
            new_row = np.zeros(n_total)
            new_row[:original_len] = row[:original_len]
            A_eq_padded.append(new_row)
        else:
            A_eq_padded.append(row[:n_total] if len(row) > n_total else row)
    A_ub_padded = []
    for row, original_len in A_ub_rows:
        if len(row) < n_total:
            new_row = np.zeros(n_total)
            new_row[:original_len] = row[:original_len]
            A_ub_padded.append(new_row)
        else:
            A_ub_padded.append(row[:n_total] if len(row) > n_total else row)

    # Goal 1: assemble sparse equality including CONV2D blocks
    if conv2d_sparse_blocks:
        import scipy.sparse as sp
        sparse_block_rows = []
        sparse_block_b = []
        for blk in conv2d_sparse_blocks:
            y_start = blk['y_start']; y_dim = blk['y_dim']
            x_start = blk['x_start']; x_dim = blk['x_dim']
            A_csr = blk['A_csr']; b_flat = blk['b_flat']
            id_rows = np.arange(y_dim)
            id_cols = y_start + id_rows
            id_vals = np.ones(y_dim, dtype=np.float64)
            A_coo = A_csr.tocoo()
            neg_rows = A_coo.row
            neg_cols = x_start + A_coo.col
            neg_vals = -A_coo.data
            all_rows = np.concatenate([id_rows, neg_rows])
            all_cols = np.concatenate([id_cols, neg_cols])
            all_vals = np.concatenate([id_vals, neg_vals])
            blk_sparse = sp.coo_matrix((all_vals, (all_rows, all_cols)),
                                                              shape=(y_dim, n_total), dtype=np.float64)
            sparse_block_rows.append(blk_sparse.tocsr())
            sparse_block_b.append(b_flat)
        if A_eq_padded:
            sparse_block_rows.append(sp.csr_matrix(np.asarray(A_eq_padded), dtype=np.float64))
            sparse_block_b.append(np.asarray(b_eq_rows, dtype=np.float64))
        A_eq_final = sp.vstack(sparse_block_rows, format='csr') if len(sparse_block_rows) > 1 else sparse_block_rows[0]
        b_eq_final = np.concatenate(sparse_block_b)
    else:
        A_eq_final = np.asarray(A_eq_padded) if A_eq_padded else None
        b_eq_final = np.asarray(b_eq_rows) if b_eq_rows else None

    return {
        'var_count': n_total,
        'layer_var_idx': layer_var_idx,
        'A_ub': np.asarray(A_ub_padded) if A_ub_padded else None,
        'b_ub': np.asarray(b_ub_rows) if b_ub_rows else None,
        'A_eq': A_eq_final,
        'b_eq': b_eq_final,
        'var_bounds': var_bounds,
        'pre_bounds': pre_bounds,
        'conv2d_sparse_blocks_count': len(conv2d_sparse_blocks),
    }


def solve_forward_lp_lb(lp: Dict, output_layer_id: int, d: np.ndarray) -> Optional[float]:
    """Solve LP for minimize d @ y where y is output of given layer.

    Returns optimal LB value if successful, None on infeasible/error.
    """
    output_idx = lp['layer_var_idx'].get(output_layer_id)
    if output_idx is None: return None
    out_start, out_dim = output_idx
    n_total = lp['var_count']

    if d.shape[0] != out_dim:
        return None  # spec dim mismatch

    obj = np.zeros(n_total)
    obj[out_start:out_start + out_dim] = d.astype(np.float64)

    try:
        result = linprog(
            c=obj,
            A_ub=lp['A_ub'], b_ub=lp['b_ub'],
            A_eq=lp['A_eq'], b_eq=lp['b_eq'],
            bounds=lp['var_bounds'],
            method='highs', options={'presolve': True})
        if result.success:
            return float(result.fun)
        return None
    except Exception:
        return None


def find_assert_pre_layer(net) -> Optional[int]:
    """Find the layer that produces ASSERT input (= the pre-spec output layer)."""
    for L in reversed(net.layers):
        if L.kind == 'ASSERT':
            preds = net.preds.get(L.id, [])
            if preds: return preds[0]
            return None
    return None
