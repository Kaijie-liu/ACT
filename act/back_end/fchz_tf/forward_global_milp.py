# ===- act/back_end/fchz_tf/forward_global_milp.py - Forward MILP refinement =#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   FCHZ MILP refinement mode for small/dense MLPs: extend the forward
#   triangle LP (forward_global_lp.py) with binary indicators on TOP-K
#   most critical unstable ReLUs (Tjeng-style exact ReLU encoding).
#
#   PRINCIPLES (revised 2026-06-09 by advisor):
#     - Forward only (no backward propagation, no CROWN)
#     - Open-source MILP allowed (HiGHS via scipy.optimize.milp)
#     - NO Gurobi or other commercial solver
#     - No BaB on input box (B&B in MILP is on activation indicators, not input)
#     - No PGD / random sampling / gradient
#     - No portfolio fallback to other backends
#
# ===---------------------------------------------------------------------===#
"""Forward MILP refinement on top-K unstable ReLUs (Tjeng encoding, HiGHS solver)."""
import os
import numpy as np
from typing import Optional, Dict, Tuple, List

try:
    from scipy.optimize import milp, LinearConstraint, Bounds as LPBounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False


def build_forward_milp(net, lb_in: np.ndarray, ub_in: np.ndarray,
                                      pre_bounds: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
                                      K_per_layer: int = 10,
                                      d_objective: Optional[np.ndarray] = None,
                                      output_layer_id: Optional[int] = None,
                                      eq_layers: Optional[int] = None,
                                      K_eq_last: Optional[int] = None):
    """Build forward MILP with top-K binary refinement per ReLU layer.

    Args:
      net: ACT Net
      lb_in, ub_in: input bounds
      pre_bounds: per-layer (lb, ub) from FCHZ propagation
      K_per_layer: max binary variables per ReLU layer (default for non-last layers)
      d_objective: spec objective vector (for top-K selection by |d_eff|)
      output_layer_id: which layer is the output (for d_eff back-propagation)
      eq_layers: concentrate K_eq_last binary on LAST N ReLU layers (eq_lagr_v8 spirit).
                      If None, all layers use K_per_layer uniformly.
      K_eq_last: K for last N ReLU layers (used when eq_layers != None). Default K_per_layer*5.
                      Env var ACT_HZ_EQ_LAYERS / ACT_HZ_K_EQ_LAST override these args.

    Returns dict with:
      var_count, layer_var_idx, A_ub, b_ub, A_eq, b_eq, var_bounds, integrality
    """
    # Allow env-var override (so existing pushes can opt in without code change)
    eq_layers_env = os.environ.get('ACT_HZ_EQ_LAYERS')
    if eq_layers_env is not None:
        try: eq_layers = int(eq_layers_env)
        except ValueError: pass
    K_eq_last_env = os.environ.get('ACT_HZ_K_EQ_LAST')
    if K_eq_last_env is not None:
        try: K_eq_last = int(K_eq_last_env)
        except ValueError: pass
    if eq_layers is not None and K_eq_last is None:
        K_eq_last = K_per_layer * 5
    from act.back_end.fchz_tf.forward_global_lp import _interval_forward
    if pre_bounds is None:
        pre_bounds = _interval_forward(net, lb_in, ub_in)

    n_in = len(lb_in)
    layer_var_idx: Dict[int, Tuple[int, int]] = {}
    var_bounds: List[Tuple[float, float]] = []
    integrality: List[int] = []   # 0 = continuous, 1 = integer
    A_eq_rows: List[np.ndarray] = []; b_eq_rows: List[float] = []
    A_ub_rows: List[np.ndarray] = []; b_ub_rows: List[float] = []
    # Goal 1 (2026-06-10): sparse equality blocks for CONV2D (memory-efficient)
    conv2d_sparse_blocks: List[Dict] = []

    input_layer = next((L for L in net.layers if L.kind == 'INPUT'), None)
    if input_layer is None: return None
    layer_var_idx[input_layer.id] = (0, n_in)
    for i in range(n_in):
        var_bounds.append((float(lb_in[i]), float(ub_in[i])))
        integrality.append(0)

    for L in net.layers:
        if L.kind == 'INPUT_SPEC':
            layer_var_idx[L.id] = layer_var_idx[input_layer.id]

    def var_idx_of(L_id): return layer_var_idx.get(L_id)
    next_var = n_in

    # First pass: compute |d_eff| per neuron per ReLU layer for top-K selection
    # We approximate d_eff backward via |W| accumulation through subsequent layers.
    d_eff_per_relu: Dict[int, np.ndarray] = {}
    if d_objective is not None:
        # Walk backward from output layer collecting per-ReLU d_eff
        relus = [L for L in net.layers if L.kind == 'RELU']
        d_curr = np.abs(d_objective.reshape(-1))
        # For each subsequent affine after the ReLU, multiply by |W|.T
        # We process layers in reverse order tracking d_curr
        # Build layer-id → next layers map
        # Simple approach: iterate net.layers reverse from output to input
        layer_order = list(net.layers)
        out_idx = None
        if output_layer_id is not None:
            for i, L in enumerate(layer_order):
                if L.id == output_layer_id:
                    out_idx = i; break
        if out_idx is not None:
            for i in range(out_idx, -1, -1):
                L = layer_order[i]
                if L.kind == 'RELU':
                    if d_curr.shape[0] == pre_bounds[L.id][0].shape[0]:
                        d_eff_per_relu[L.id] = d_curr.copy()
                elif L.kind == 'DENSE':
                    W = L.params.get('weight')
                    if W is not None:
                        W_np = W.detach().cpu().numpy() if hasattr(W, 'detach') else np.asarray(W)
                        W_np = np.abs(W_np.astype(np.float64))
                        if d_curr.shape[0] == W_np.shape[0]:
                            d_curr = W_np.T @ d_curr
                # other layers: passthrough

    for L in net.layers:
        if L.kind in ('INPUT', 'INPUT_SPEC', 'CONSTANT'):
            continue
        preds = net.preds.get(L.id, [])
        if not preds: return None
        pred_id = preds[0]
        pred_idx = var_idx_of(pred_id)
        if pred_idx is None: return None
        pred_start, pred_dim = pred_idx
        lb_pred, ub_pred = pre_bounds[pred_id]

        if L.kind == 'DENSE':
            W = L.params['weight']
            b = L.params.get('bias')
            W_np = W.detach().cpu().numpy().astype(np.float64) if hasattr(W, 'detach') else np.asarray(W).astype(np.float64)
            b_np = (b.detach().cpu().numpy().astype(np.float64) if hasattr(b, 'detach') else np.asarray(b).astype(np.float64)) if b is not None else np.zeros(W_np.shape[0])
            n_out = W_np.shape[0]
            layer_var_idx[L.id] = (next_var, n_out)
            lb_o, ub_o = pre_bounds[L.id]
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                integrality.append(0)
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
            # Allocate continuous y vars
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                integrality.append(0)

            # eq_lagr_v8 spirit: concentrate K on LAST eq_layers RELU.
            # If eq_layers set, use K_eq_last for last N RELU, K_per_layer for earlier.
            if eq_layers is not None and K_eq_last is not None:
                # Compute position of this RELU layer from the end
                relu_layers_total = [LL for LL in net.layers if LL.kind == 'RELU']
                try:
                    pos_from_end = (len(relu_layers_total) - 1) - relu_layers_total.index(L)
                except ValueError:
                    pos_from_end = 999
                K_this = K_eq_last if pos_from_end < eq_layers else K_per_layer
            else:
                K_this = K_per_layer

            # Select top-K unstable for binary indicators (K_this depends on eq_layers)
            unstable_idx = [j for j in range(n_out) if lb_pred[j] < 0 < ub_pred[j]]
            if d_eff_per_relu.get(L.id) is not None:
                d_eff_layer = d_eff_per_relu[L.id]
                importance = np.abs(d_eff_layer[unstable_idx]) * (ub_pred[unstable_idx] - lb_pred[unstable_idx])
                order = np.argsort(-importance)[:K_this]
            else:
                order = np.arange(min(len(unstable_idx), K_this))
            binary_for = set(unstable_idx[i] for i in order if i < len(unstable_idx))

            # Allocate binary vars for selected unstable
            binary_var_idx: Dict[int, int] = {}
            for j in sorted(binary_for):
                binary_var_idx[j] = next_var + n_out + len(binary_var_idx)
            n_b = len(binary_var_idx)
            for _ in range(n_b):
                var_bounds.append((0.0, 1.0))
                integrality.append(1)
            n_total = next_var + n_out + n_b

            v_pre_start = pred_start
            v_out_start = next_var
            for j in range(n_out):
                l_pre = float(lb_pred[j]); u_pre = float(ub_pred[j])
                v_out = v_out_start + j
                v_pre = v_pre_start + j
                if l_pre >= 0:
                    # Stable active: y == z
                    row = np.zeros(n_total)
                    row[v_out] = 1.0; row[v_pre] = -1.0
                    A_eq_rows.append((row, n_total)); b_eq_rows.append(0.0)
                elif u_pre <= 0:
                    # Stable inactive: y == 0
                    row = np.zeros(n_total)
                    row[v_out] = 1.0
                    A_eq_rows.append((row, n_total)); b_eq_rows.append(0.0)
                elif j in binary_for:
                    # Tjeng exact ReLU:
                    #   y >= 0       (already in y bounds [max(0,l), max(0,u)])
                    #   y >= z       → -y + z <= 0
                    row = np.zeros(n_total); row[v_out] = -1.0; row[v_pre] = 1.0
                    A_ub_rows.append((row, n_total)); b_ub_rows.append(0.0)
                    #   y <= u * b   → y - u*b <= 0
                    v_bin = binary_var_idx[j]
                    row = np.zeros(n_total); row[v_out] = 1.0; row[v_bin] = -u_pre
                    A_ub_rows.append((row, n_total)); b_ub_rows.append(0.0)
                    #   y <= z - l*(1-b)  → y - z + l - l*b <= 0  → y - z - l*b <= -l
                    row = np.zeros(n_total); row[v_out] = 1.0; row[v_pre] = -1.0; row[v_bin] = -l_pre
                    A_ub_rows.append((row, n_total)); b_ub_rows.append(-l_pre)
                else:
                    # Triangle relaxation:
                    row = np.zeros(n_total); row[v_out] = -1.0; row[v_pre] = 1.0
                    A_ub_rows.append((row, n_total)); b_ub_rows.append(0.0)
                    slope = u_pre / (u_pre - l_pre)
                    rhs = -slope * l_pre
                    row = np.zeros(n_total); row[v_out] = 1.0; row[v_pre] = -slope
                    A_ub_rows.append((row, n_total)); b_ub_rows.append(rhs)
            next_var += n_out + n_b

        elif L.kind == 'ASSERT':
            layer_var_idx[L.id] = (pred_start, pred_dim)
        elif L.kind in ('FLATTEN', 'RESHAPE'):
            layer_var_idx[L.id] = (pred_start, pred_dim)
        elif L.kind == 'BIAS':
            b = L.params.get('bias')
            if b is None: b = L.params.get('c')
            if b is not None:
                b_np = (b.detach().cpu().numpy().astype(np.float64).reshape(-1)
                            if hasattr(b, 'detach') else np.asarray(b).astype(np.float64).reshape(-1))
                if b_np.shape[0] != pred_dim:
                    layer_var_idx[L.id] = (pred_start, pred_dim)
                else:
                    n_out = pred_dim
                    layer_var_idx[L.id] = (next_var, n_out)
                    lb_o, ub_o = pre_bounds[L.id]
                    for j in range(n_out):
                        var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                        integrality.append(0)
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
            layer_var_idx[L.id] = (pred_start + s, slice_dim)
        elif L.kind == 'CONCAT':
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
                integrality.append(0)
            n_total = next_var + n_out
            offset = 0
            for p in preds:
                p_start, p_dim = var_idx_of(p)
                for k in range(p_dim):
                    row = np.zeros(n_total)
                    row[next_var + offset + k] = 1.0
                    row[p_start + k] = -1.0
                    A_eq_rows.append((row, n_total)); b_eq_rows.append(0.0)
                offset += p_dim
            next_var += n_out
        elif L.kind in ('SIGMOID', 'TANH'):
            # Sigmoid/Tanh tiered encoding:
            #   ACT_HZ_SIGMOID_CHORD = '0' (default): loose box
            #   ACT_HZ_SIGMOID_CHORD = '1': K=1 chord+residual (TIGHTER)
            #   ACT_HZ_SIGMOID_CHORD = 'K2': K=2 with binary segment indicator (TIGHTEST)
            #   ACT_HZ_SIGMOID_TOPK = N: only top-N widest neurons get K=2 (memory control)
            chord_mode = os.environ.get('ACT_HZ_SIGMOID_CHORD', '0')
            if chord_mode == '0':
                # Loose box: fresh vars with box bounds from FCHZ; sound but loose.
                lb_o, ub_o = pre_bounds[L.id]
                n_out = lb_o.shape[0]
                layer_var_idx[L.id] = (next_var, n_out)
                for j in range(n_out):
                    var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                    integrality.append(0)
                next_var += n_out
                continue
            kind_name = 'Sigmoid' if L.kind == 'SIGMOID' else 'Tanh'
            func = (lambda x: 1.0 / (1.0 + np.exp(-x))) if kind_name == 'Sigmoid' else np.tanh
            n_out = pred_dim
            layer_var_idx[L.id] = (next_var, n_out)
            lb_o, ub_o = pre_bounds[L.id]
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                integrality.append(0)

            if chord_mode == 'K2':
                # K=2 with binary segment indicator (TIGHTEST). Sound, principled MILP.
                from act.back_end.fchz_tf.sigmoid_kpiece_milp import add_kpiece_sigmoid_tanh_milp
                top_K = int(os.environ.get('ACT_HZ_SIGMOID_TOPK', '0'))
                n_binary_added = add_kpiece_sigmoid_tanh_milp(
                    L.kind, lb_pred, ub_pred,
                    v_pre_start=pred_start, v_out_start=next_var,
                    n_total_initial=next_var + n_out,
                    var_bounds=var_bounds, integrality=integrality,
                    A_ub_rows=A_ub_rows, b_ub_rows=b_ub_rows,
                    A_eq_rows=A_eq_rows, b_eq_rows=b_eq_rows,
                    top_K=top_K)
                next_var += n_out + n_binary_added
                continue

            # K=1 chord (chord_mode == '1')
            n_total = next_var + n_out
            for j in range(n_out):
                l_pre = float(lb_pred[j]); u_pre = float(ub_pred[j])
                v_out = next_var + j
                v_pre = pred_start + j
                if u_pre - l_pre < 1e-12:
                    row = np.zeros(n_total)
                    row[v_out] = 1.0
                    A_eq_rows.append((row, n_total))
                    b_eq_rows.append(float(func(l_pre)))
                else:
                    fl = float(func(l_pre)); fu = float(func(u_pre))
                    slope = (fu - fl) / (u_pre - l_pre)
                    intercept = fl - slope * l_pre
                    sample_pts = np.linspace(l_pre, u_pre, 51)
                    chord_y = slope * sample_pts + intercept
                    true_y = func(sample_pts)
                    max_res = float(np.abs(true_y - chord_y).max()) + 1e-12
                    row = np.zeros(n_total)
                    row[v_out] = 1.0; row[v_pre] = -slope
                    A_ub_rows.append((row, n_total))
                    b_ub_rows.append(intercept + max_res)
                    row = np.zeros(n_total)
                    row[v_out] = -1.0; row[v_pre] = slope
                    A_ub_rows.append((row, n_total))
                    b_ub_rows.append(-(intercept - max_res))
            next_var += n_out
        elif L.kind == 'SCALE':
            # Element-wise y_j = a_j * x_j
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
                        integrality.append(0)
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
            # Goal 1 (2026-06-10): tight sparse-affine encoding.
            # y_flat = A_sparse @ x_flat + b_flat where A_sparse is the conv-as-affine matrix.
            # Falls back to loose-box encoding on any encoding failure.
            from act.back_end.fchz_tf.conv2d_sparse_encoder import conv2d_to_sparse_affine
            W = L.params.get('weight')
            b_param = L.params.get('bias')
            input_shape_4d = L.params.get('input_shape')
            stride = L.params.get('stride', (1, 1))
            padding = L.params.get('padding', (0, 0))
            tight_encoding_ok = False
            if W is not None and input_shape_4d is not None and len(input_shape_4d) == 4:
                try:
                    W_np = (W.detach().cpu().numpy() if hasattr(W, 'detach') else np.asarray(W)).astype(np.float64)
                    b_np = ((b_param.detach().cpu().numpy() if hasattr(b_param, 'detach') else np.asarray(b_param)).astype(np.float64)
                                if b_param is not None else None)
                    # Verify pred_dim matches expected input flat
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
                                integrality.append(0)
                            # Store sparse equality block (defer concat to end of build)
                            conv2d_sparse_blocks.append({
                                'y_start': next_var,
                                'y_dim': n_out,
                                'x_start': pred_start,
                                'x_dim': pred_dim,
                                'A_csr': A_csr,
                                'b_flat': b_flat,
                            })
                            next_var += n_out
                            tight_encoding_ok = True
                except Exception:
                    tight_encoding_ok = False
            if not tight_encoding_ok:
                # Fallback to loose-box encoding (original behavior)
                lb_o, ub_o = pre_bounds[L.id]
                n_out = lb_o.shape[0]
                layer_var_idx[L.id] = (next_var, n_out)
                for j in range(n_out):
                    var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                    integrality.append(0)
                next_var += n_out

        elif L.kind == 'ADD':
            # Residual ADD: y = x1 + x2 (state-state element-wise sum)
            # Sound encoding: equality row y - x1 - x2 = 0 per element.
            # ResNet residual blocks are the primary user.
            x_vars = L.params.get('x_vars')
            y_vars = L.params.get('y_vars')
            out_shape = L.params.get('output_shape') or L.params.get('input_shape')
            # Find predecessor layers via net.preds
            pred_ids = net.preds.get(L.id, [])
            if len(pred_ids) != 2:
                return None
            pred1_idx = layer_var_idx.get(pred_ids[0])
            pred2_idx = layer_var_idx.get(pred_ids[1])
            if pred1_idx is None or pred2_idx is None:
                return None
            p1_start, p1_dim = pred1_idx
            p2_start, p2_dim = pred2_idx
            if p1_dim != p2_dim:
                return None
            n_out = p1_dim
            layer_var_idx[L.id] = (next_var, n_out)
            if L.id in pre_bounds:
                lb_o, ub_o = pre_bounds[L.id]
            else:
                # Fallback: derive from predecessor bounds
                lb1, ub1 = pre_bounds.get(pred_ids[0], (None, None))
                lb2, ub2 = pre_bounds.get(pred_ids[1], (None, None))
                if lb1 is None or lb2 is None:
                    return None
                lb_o = lb1 + lb2
                ub_o = ub1 + ub2
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                integrality.append(0)
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
            # ASSERT is the spec output node — pass through (no new vars).
            preds_ids = net.preds.get(L.id, [])
            if preds_ids and preds_ids[0] in layer_var_idx:
                layer_var_idx[L.id] = layer_var_idx[preds_ids[0]]
            else:
                layer_var_idx[L.id] = (next_var - 1, 1)   # fallback

        elif L.kind == 'SUB':
            # SUB: y = x1 - x2 (state-state element-wise difference, linear)
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
                lb_o = lb1 - ub2; ub_o = ub1 - lb2
            for j in range(n_out):
                var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                integrality.append(0)
            n_total_now = next_var + n_out
            for j in range(n_out):
                row = np.zeros(n_total_now)
                row[next_var + j] = 1.0
                row[p1_start + j] = -1.0
                row[p2_start + j] = 1.0   # y - x1 + x2 = 0
                A_eq_rows.append((row, n_total_now))
                b_eq_rows.append(0.0)
            next_var += n_out

        elif L.kind in ('RESHAPE', 'EXPAND'):
            # Reshape/Expand: identity pass-through.
            # For EXPAND that changes element count, only handle 1→N broadcast.
            pred_ids = net.preds.get(L.id, [])
            if not pred_ids: return None
            pred_idx = layer_var_idx.get(pred_ids[0])
            if pred_idx is None: return None
            p_start, p_dim = pred_idx
            # If output_shape matches pred_dim, identity
            if L.id in pre_bounds:
                lb_o, ub_o = pre_bounds[L.id]
                if lb_o.shape[0] == p_dim:
                    layer_var_idx[L.id] = pred_idx
                    continue
                # Broadcast 1 → N: each output var = input var (pass through with repeat)
                if p_dim == 1 and lb_o.shape[0] > 1:
                    n_out = lb_o.shape[0]
                    layer_var_idx[L.id] = (next_var, n_out)
                    for j in range(n_out):
                        var_bounds.append((float(lb_o[j]), float(ub_o[j])))
                        integrality.append(0)
                    n_total_now = next_var + n_out
                    for j in range(n_out):
                        row = np.zeros(n_total_now)
                        row[next_var + j] = 1.0
                        row[p_start] = -1.0   # y_j - x_0 = 0
                        A_eq_rows.append((row, n_total_now))
                        b_eq_rows.append(0.0)
                    next_var += n_out
                    continue
            # Default: identity pass-through
            layer_var_idx[L.id] = pred_idx

        elif L.kind == 'CONSTANT':
            # CONSTANT: emit fresh vars at the constant's value (no slack).
            value = L.params.get('value')
            if value is None: return None
            v_np = (value.detach().cpu().numpy().astype(np.float64).reshape(-1)
                        if hasattr(value, 'detach') else np.asarray(value).astype(np.float64).reshape(-1))
            n_out = v_np.shape[0]
            layer_var_idx[L.id] = (next_var, n_out)
            for j in range(n_out):
                var_bounds.append((float(v_np[j]), float(v_np[j])))
                integrality.append(0)
            n_total_now = next_var + n_out
            for j in range(n_out):
                row = np.zeros(n_total_now)
                row[next_var + j] = 1.0
                A_eq_rows.append((row, n_total_now))
                b_eq_rows.append(float(v_np[j]))
            next_var += n_out

        else:
            return None

    n_total = next_var
    # Pad rows
    def pad(rows, n):
        out = []
        for row, original_len in rows:
            if len(row) < n:
                new_row = np.zeros(n)
                new_row[:original_len] = row[:original_len]
                out.append(new_row)
            else:
                out.append(row[:n])
        return out

    # Goal 1: assemble dense+sparse equality matrix
    # Each conv2d sparse block adds y_dim equality rows: y[k] - A_sparse[k,:] @ x = b[k]
    if A_eq_rows or conv2d_sparse_blocks:
        if conv2d_sparse_blocks:
            import scipy.sparse as sp
            sparse_block_rows = []
            sparse_block_b = []
            for blk in conv2d_sparse_blocks:
                y_start = blk['y_start']; y_dim = blk['y_dim']
                x_start = blk['x_start']; x_dim = blk['x_dim']
                A_csr = blk['A_csr']
                b_flat = blk['b_flat']
                # Build block matrix: rows are y_dim x n_total
                # For row k: +1 at col (y_start + k), -A_csr[k, :] at cols (x_start..x_start+x_dim)
                # Use COO incremental
                # Identity contribution for y vars
                id_rows = np.arange(y_dim)
                id_cols = y_start + id_rows
                id_vals = np.ones(y_dim, dtype=np.float64)
                # -A contribution for x vars: shift A_csr columns by x_start
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
            # Dense rows → sparse
            if A_eq_rows:
                dense_eq = np.asarray(pad(A_eq_rows, n_total))
                sparse_block_rows.append(sp.csr_matrix(dense_eq, dtype=np.float64))
                sparse_block_b.append(np.asarray(b_eq_rows, dtype=np.float64))
            A_eq = sp.vstack(sparse_block_rows, format='csr') if len(sparse_block_rows) > 1 else sparse_block_rows[0]
            b_eq_arr = np.concatenate(sparse_block_b)
        else:
            A_eq = np.asarray(pad(A_eq_rows, n_total))
            b_eq_arr = np.asarray(b_eq_rows)
    else:
        A_eq = None
        b_eq_arr = None
    A_ub = np.asarray(pad(A_ub_rows, n_total)) if A_ub_rows else None

    return {
        'var_count': n_total,
        'layer_var_idx': layer_var_idx,
        'A_ub': A_ub, 'b_ub': np.asarray(b_ub_rows) if b_ub_rows else None,
        'A_eq': A_eq, 'b_eq': b_eq_arr if b_eq_arr is not None else None,
        'conv2d_sparse_blocks_count': len(conv2d_sparse_blocks),
        'var_bounds': var_bounds,
        'integrality': np.asarray(integrality),
        'n_binary': int(sum(integrality)),
    }


def solve_forward_milp_lb(milp_data: Dict, output_layer_id: int, d: np.ndarray,
                                          time_limit_s: float = 30.0) -> Optional[float]:
    """Solve MILP for minimize d @ y. Returns LB or None on failure."""
    if not HAS_MILP: return None
    output_idx = milp_data['layer_var_idx'].get(output_layer_id)
    if output_idx is None: return None
    out_start, out_dim = output_idx
    n_total = milp_data['var_count']
    if d.shape[0] != out_dim: return None

    obj = np.zeros(n_total)
    obj[out_start:out_start + out_dim] = d.astype(np.float64)

    constraints = []
    if milp_data['A_ub'] is not None:
        constraints.append(LinearConstraint(milp_data['A_ub'], ub=milp_data['b_ub']))
    if milp_data['A_eq'] is not None:
        constraints.append(LinearConstraint(milp_data['A_eq'], lb=milp_data['b_eq'], ub=milp_data['b_eq']))

    lbs = np.array([b[0] for b in milp_data['var_bounds']])
    ubs = np.array([b[1] for b in milp_data['var_bounds']])
    bounds_obj = LPBounds(lb=lbs, ub=ubs)

    try:
        result = milp(
            c=obj,
            constraints=constraints if constraints else None,
            integrality=milp_data['integrality'],
            bounds=bounds_obj,
            options={'time_limit': time_limit_s, 'disp': False})
        if result.success:
            return float(result.fun)
        return None
    except Exception:
        return None
