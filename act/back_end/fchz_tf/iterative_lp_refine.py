# ===- act/back_end/fchz_tf/iterative_lp_refine.py - Iterative Forward LP =#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Iterative LP-based forward pre-act bound tightening.
#   For each ReLU layer L, solve LP to tighten its pre-act bounds using
#   all previous-layer constraints. These tighter bounds are then used
#   to (a) classify more stable neurons (smaller K), (b) tighter triangle
#   slopes for downstream layers, (c) tighter final-layer LB.
#
#   PRINCIPLES:
#     - Forward only (each layer's LP uses only PRIOR layers' constraints)
#     - Continuous LP (HiGHS), no MILP, no Gurobi
#     - No backward propagation, no CROWN
#     - Sound: every LP_bound[i] is a valid bound on neuron i
#
#   Per advisor 2026-06-09: open-source LP allowed.
#
# ===---------------------------------------------------------------------===#
"""Sound forward iterative LP refinement on pre-act bounds."""
import numpy as np
from typing import Optional, Dict, Tuple, List
from scipy.optimize import linprog


def iterative_lp_pre_bounds(net, lb_in: np.ndarray, ub_in: np.ndarray,
                                                 fchz_pre_bounds: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                                 max_neurons_per_layer: int = 50,
                                                 time_per_lp_s: float = 2.0) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """For each ReLU layer's PRE-act, solve up to max_neurons*2 LPs to tighten bounds.

    Returns updated pre_bounds (tighter or same).
    Sound: LP gives an outer bound on the true reachable set.
    """
    from act.back_end.fchz_tf.forward_global_lp import build_forward_lp, solve_forward_lp_lb

    # Refine bounds layer-by-layer
    refined = dict(fchz_pre_bounds)
    # Find ReLU layers
    relu_layers = [L for L in net.layers if L.kind == 'RELU']
    if not relu_layers: return refined

    for relu_L in relu_layers:
        # The pre-act = predecessor's output
        preds = net.preds.get(relu_L.id, [])
        if not preds: continue
        pre_id = preds[0]
        if pre_id not in refined: continue
        lb_pre, ub_pre = refined[pre_id]
        n_neurons = lb_pre.shape[0]

        # Identify unstable neurons (could become stable with tighter bounds)
        unstable_idx = [j for j in range(n_neurons) if lb_pre[j] < 0 < ub_pre[j]]
        if not unstable_idx: continue

        # Pick top-N unstable to refine (by closest to 0 i.e. largest |slope - 1|)
        slopes = np.array([ub_pre[j] / (ub_pre[j] - lb_pre[j]) for j in unstable_idx])
        importance = np.abs(slopes - 0.5) * (ub_pre[unstable_idx] - lb_pre[unstable_idx])
        top_n = min(max_neurons_per_layer, len(unstable_idx))
        top = np.argsort(-importance)[:top_n]
        target_neurons = [unstable_idx[i] for i in top]

        # Build LP up to this layer (using current refined bounds for prior layers)
        lp = build_forward_lp(net, lb_in, ub_in, pre_bounds=refined)
        if lp is None: continue

        # Find pre-id layer's var idx in LP
        pre_idx = lp['layer_var_idx'].get(pre_id)
        if pre_idx is None: continue
        pre_start, pre_dim = pre_idx

        # For each target neuron j, solve min/max of x[pre_start+j]
        # Using d vector with 1 at position pre_start+j, 0 elsewhere
        new_lb = lb_pre.copy(); new_ub = ub_pre.copy()
        for j in target_neurons:
            d = np.zeros(pre_dim, dtype=np.float64); d[j] = 1.0
            # LP gives LB(x[j]) directly
            try:
                # Solve min x[j]
                lp_lb = solve_forward_lp_lb(lp, pre_id, d)
                if lp_lb is not None and lp_lb > new_lb[j]:
                    new_lb[j] = lp_lb
                # Solve max x[j] = -min(-x[j])
                lp_neg = solve_forward_lp_lb(lp, pre_id, -d)
                if lp_neg is not None:
                    lp_ub = -lp_neg
                    if lp_ub < new_ub[j]: new_ub[j] = lp_ub
            except Exception:
                pass
        # Sound: refined bounds are tighter than original
        refined[pre_id] = (new_lb, new_ub)
        # Also refine the RELU output: max(0, pre)
        refined[relu_L.id] = (np.maximum(new_lb, 0), np.maximum(new_ub, 0))

    return refined
