# ===- act/back_end/fchz_tf/iterative_pre_bound_lp.py - Iterative bound tightening ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Iterative LP-based pre-activation bound tightening for CNN models.
#
#   Standard FCHZ pre-bounds (forward zonotope abstraction) can be wide for
#   CONV2D + ReLU cascades. With sparse CONV2D LP encoding, we can solve
#   per-neuron LPs to compute MUCH tighter pre-act bounds:
#
#     lb_tight[i] = min(z_i s.t. LP constraints)
#     ub_tight[i] = max(z_i s.t. LP constraints) = -min(-z_i s.t. ...)
#
#   These tighter bounds:
#     1) Make more ReLU neurons stable (l>=0 or u<=0), reducing unstable count
#     2) Tighten triangle relaxation for remaining unstable
#     3) Tighten final spec MILP
#
# Principle-compliance:
#   - Forward LP only (no backward CROWN)
#   - No PGD (concrete x* solver returns are box-feasible)
#   - No BaB (single LP per neuron, no input splitting)
#   - No Gurobi (HiGHS via scipy.optimize.linprog)
#   - Sound (LP gives proven LB/UB)
#
# Performance:
#   For CIFAR oval21 (~3000 unstable neurons), full tightening costs
#   ~3000 * 0.5s = 25 min. Selective top-K tightening reduces to 100-500 LPs.
#
# ===---------------------------------------------------------------------===#
"""Iterative LP pre-bound tightening for FCHZ + CONV2D."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linprog
import scipy.sparse as sp


def tighten_pre_bounds_with_lp(net, lb_in: np.ndarray, ub_in: np.ndarray,
                                                    pre_bounds: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                                    target_layers: Optional[List[int]] = None,
                                                    top_k_per_layer: int = 100,
                                                    selection_by_d_eff: bool = True,
                                                    d_objective: Optional[np.ndarray] = None,
                                                    time_per_lp: float = 5.0,
                                                    verbose: bool = False
                                                    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Tighten pre-activation bounds via per-neuron LP solving.

    Args:
      net: ACT net
      lb_in, ub_in: input box bounds (flat)
      pre_bounds: existing FCHZ-derived pre-bounds {layer_id: (lb, ub)}
      target_layers: which layers to tighten; None = all CONV2D-fed RELU layers
      top_k_per_layer: tighten this many neurons per layer (by |d_eff| importance)
      selection_by_d_eff: if True, prioritize neurons that contribute to final objective
      d_objective: objective row for d_eff computation (used for selection)
      time_per_lp: HiGHS time limit per LP (seconds)
      verbose: print progress

    Returns:
      Tightened pre_bounds dict (only modified layers replaced; rest passes through)

    Soundness: LP gives proven LB/UB; tightened bounds are sound.
    """
    from act.back_end.fchz_tf.forward_global_lp import build_forward_lp

    # Build initial LP with original pre_bounds
    lp = build_forward_lp(net, lb_in, ub_in, pre_bounds=pre_bounds)
    if lp is None:
        if verbose: print("LP build failed, return original bounds")
        return pre_bounds

    # Identify target layers (ReLU layers fed by CONV2D)
    if target_layers is None:
        target_layers = []
        for L in net.layers:
            if L.kind == 'RELU':
                preds = net.preds.get(L.id, [])
                for pid in preds:
                    pred_L = next((x for x in net.layers if x.id == pid), None)
                    if pred_L and pred_L.kind == 'CONV2D':
                        target_layers.append(pid)   # pre-act layer (CONV2D output)
                        break

    tightened = dict(pre_bounds)
    layer_var_idx = lp['layer_var_idx']
    n_total = lp['var_count']

    # Compute d_eff per layer if selection_by_d_eff and d_objective given
    d_eff_per_layer: Dict[int, np.ndarray] = {}
    if selection_by_d_eff and d_objective is not None:
        cur = np.abs(d_objective.astype(np.float64))
        # Walk backward through layers from output → input
        layer_by_id = {L.id: L for L in net.layers}
        # Find topological order
        for L_id in [L.id for L in reversed(net.layers)]:
            L = layer_by_id[L_id]
            if L.kind == 'DENSE' and L_id in layer_var_idx:
                # cur is for output of this DENSE; propagate to input via |W|
                W = L.params.get('weight')
                if W is not None:
                    W_np = (W.detach().cpu().numpy() if hasattr(W, 'detach') else np.asarray(W)).astype(np.float64)
                    if cur.shape[0] == W_np.shape[0]:
                        cur = np.abs(W_np).T @ cur
            elif L.kind == 'RELU':
                d_eff_per_layer[L.id] = cur.copy()
            elif L.kind == 'CONV2D':
                d_eff_per_layer[L.id] = cur.copy()

    n_neurons_tightened = 0
    total_layers_processed = 0
    for tgt_id in target_layers:
        if tgt_id not in pre_bounds: continue
        if tgt_id not in layer_var_idx: continue
        l_pre, u_pre = pre_bounds[tgt_id]
        n = l_pre.shape[0]
        out_start, out_dim = layer_var_idx[tgt_id]
        if out_dim != n: continue

        # Select which neurons to tighten
        unstable_mask = (l_pre < 0) & (u_pre > 0)
        unstable_indices = np.where(unstable_mask)[0]
        if len(unstable_indices) == 0: continue

        # Prioritize by |d_eff| if available, else by current width
        if tgt_id in d_eff_per_layer and d_eff_per_layer[tgt_id].shape[0] == n:
            scores = d_eff_per_layer[tgt_id][unstable_indices]
        else:
            scores = (u_pre - l_pre)[unstable_indices]
        order = np.argsort(-scores)   # descending
        select = unstable_indices[order[:top_k_per_layer]]

        new_lb = l_pre.copy()
        new_ub = u_pre.copy()
        for neuron_idx in select:
            # Solve LP: minimize z_i = c
            c = np.zeros(n_total)
            c[out_start + neuron_idx] = 1.0
            try:
                res_lb = linprog(c=c, A_ub=lp['A_ub'], b_ub=lp['b_ub'],
                                          A_eq=lp['A_eq'], b_eq=lp['b_eq'],
                                          bounds=lp['var_bounds'], method='highs',
                                          options={'time_limit': time_per_lp, 'presolve': True})
                if res_lb.success and res_lb.fun is not None:
                    lp_lb = float(res_lb.fun)
                    if lp_lb > new_lb[neuron_idx]:
                        new_lb[neuron_idx] = lp_lb
            except Exception:
                pass
            # Solve LP: maximize z_i = minimize -c
            c[out_start + neuron_idx] = -1.0
            try:
                res_ub = linprog(c=c, A_ub=lp['A_ub'], b_ub=lp['b_ub'],
                                          A_eq=lp['A_eq'], b_eq=lp['b_eq'],
                                          bounds=lp['var_bounds'], method='highs',
                                          options={'time_limit': time_per_lp, 'presolve': True})
                if res_ub.success and res_ub.fun is not None:
                    lp_ub = -float(res_ub.fun)
                    if lp_ub < new_ub[neuron_idx]:
                        new_ub[neuron_idx] = lp_ub
            except Exception:
                pass
            n_neurons_tightened += 1

        # Soundness check: tightened must be a subset of original
        new_lb = np.maximum(new_lb, l_pre)
        new_ub = np.minimum(new_ub, u_pre)
        tightened[tgt_id] = (new_lb, new_ub)
        total_layers_processed += 1
        if verbose:
            old_unstable = int(((l_pre < 0) & (u_pre > 0)).sum())
            new_unstable = int(((new_lb < 0) & (new_ub > 0)).sum())
            print(f"  layer {tgt_id}: tightened {min(top_k_per_layer, len(unstable_indices))} neurons, "
                      f"unstable {old_unstable} → {new_unstable}")

    if verbose:
        print(f"Total: tightened {n_neurons_tightened} neurons across {total_layers_processed} layers")

    return tightened


__all__ = ['tighten_pre_bounds_with_lp']
