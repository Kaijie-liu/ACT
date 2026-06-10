# ===- act/back_end/fchz_tf/sigmoid_kpiece_milp.py - K-piece Sigmoid/Tanh in MILP ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   K=2 Sigmoid/Tanh encoding for forward_global_milp.py.
#   This is the REAL K-piece innovation — uses MILP binary indicator
#   to select segment, gives tighter convex hull per segment than K=1 chord.
#
#   Mathematically: for each unstable Sigmoid neuron, split [l, u] at midpoint m.
#     Segment 0: z ∈ [l, m], chord_0(z) ± r_0
#     Segment 1: z ∈ [m, u], chord_1(z) ± r_1
#     Binary b: 0 = seg 0, 1 = seg 1
#
#   Per neuron: 1 binary + 6 inequalities (2 z-box + 4 chord constraints)
#   vs K=1 chord: 0 binary + 2 inequalities
#
#   Tighter LP relaxation because each segment chord is tighter than global chord.
#
# Configurable via env:
#   ACT_HZ_SIGMOID_CHORD = "0" (default loose box, preserves 1831)
#                                = "1" (K=1 chord, tighter)
#                                = "K2" (K=2 with binary, tightest)
#   ACT_HZ_SIGMOID_TOPK = 0 (apply to ALL unstable)
#                              = N (only top-N widest unstable neurons get K=2)
#
# Soundness: each segment chord ± residual is sample-verified to cover sigmoid curve.
#                The binary forces exactly one segment to be active in each MILP solution.
#
# ===---------------------------------------------------------------------===#
"""K=2 Sigmoid/Tanh MILP encoding with binary segment indicator."""
import os
import numpy as np
from typing import List, Tuple, Optional


def add_kpiece_sigmoid_tanh_milp(L_kind: str, lb_pre: np.ndarray, ub_pre: np.ndarray,
                                                            v_pre_start: int, v_out_start: int,
                                                            n_total_initial: int,
                                                            var_bounds: List[Tuple[float, float]],
                                                            integrality: List[int],
                                                            A_ub_rows: List[Tuple],
                                                            b_ub_rows: List[float],
                                                            A_eq_rows: List[Tuple],
                                                            b_eq_rows: List[float],
                                                            top_K: int = 0) -> int:
    """Add K=2 Sigmoid/Tanh encoding to MILP.

    Args:
      L_kind: 'SIGMOID' or 'TANH'
      lb_pre, ub_pre: pre-activation bounds per neuron
      v_pre_start: index of first pre-act var in MILP
      v_out_start: index of first post-act var in MILP (already allocated)
      n_total_initial: total vars before adding binary indicators
      var_bounds, integrality, A_ub_rows, b_ub_rows, A_eq_rows, b_eq_rows: MILP state
      top_K: 0 = all unstable get K=2; N>0 = only top-N widest get K=2

    Returns:
      n_binary_added: how many binary vars added

    Soundness: y ∈ [chord_seg(z) - r_seg, chord_seg(z) + r_seg] for active segment.
                  Binary enforces exactly one segment active.
    """
    kind_name = 'Sigmoid' if L_kind == 'SIGMOID' else 'Tanh'
    func = (lambda x: 1.0 / (1.0 + np.exp(-x))) if kind_name == 'Sigmoid' else np.tanh

    n_out = lb_pre.shape[0]
    # Identify wide (unstable) neurons
    is_wide = (ub_pre - lb_pre) > 1e-8
    wide_idx = np.where(is_wide)[0]
    widths = ub_pre[wide_idx] - lb_pre[wide_idx]

    # If top_K specified, restrict to widest
    if top_K > 0 and len(wide_idx) > top_K:
        order = np.argsort(-widths)[:top_K]
        wide_idx = wide_idx[order]

    n_wide = len(wide_idx)
    # New binary vars: 1 per wide neuron
    binary_var_start = n_total_initial
    for _ in range(n_wide):
        var_bounds.append((0.0, 1.0))
        integrality.append(1)
    n_total = n_total_initial + n_wide

    # Helper: emit a sparse row
    def add_ub_row(coeffs_dict, rhs):
        row = np.zeros(n_total)
        for col, val in coeffs_dict.items():
            row[col] = val
        A_ub_rows.append((row, n_total))
        b_ub_rows.append(rhs)

    M_z = float(max(np.max(ub_pre) - np.min(lb_pre) + 1.0, 1.0))
    # M_y bound (max y range): sigmoid ∈ [0, 1], tanh ∈ [-1, 1] + slack
    M_y = 2.0

    for slot, j in enumerate(wide_idx):
        l = float(lb_pre[j]); u = float(ub_pre[j])
        v_pre = v_pre_start + j
        v_out = v_out_start + j
        v_bin = binary_var_start + slot
        m = (l + u) / 2.0

        # Segment 0: z in [l, m], chord through (l, f(l)) and (m, f(m))
        fl = float(func(l)); fm = float(func(m)); fu = float(func(u))
        slope_0 = (fm - fl) / max(m - l, 1e-30)
        intercept_0 = fl - slope_0 * l
        sample_pts_0 = np.linspace(l, m, 51)
        chord_0 = slope_0 * sample_pts_0 + intercept_0
        true_0 = func(sample_pts_0)
        r_0 = float(np.abs(true_0 - chord_0).max()) + 1e-9

        # Segment 1: z in [m, u], chord through (m, f(m)) and (u, f(u))
        slope_1 = (fu - fm) / max(u - m, 1e-30)
        intercept_1 = fm - slope_1 * m
        sample_pts_1 = np.linspace(m, u, 51)
        chord_1 = slope_1 * sample_pts_1 + intercept_1
        true_1 = func(sample_pts_1)
        r_1 = float(np.abs(true_1 - chord_1).max()) + 1e-9

        # ── Segment selection constraints ──
        # z <= m when b=0:  z - m - M_z * b <= 0
        add_ub_row({v_pre: 1.0, v_bin: -M_z}, m)
        # z >= m when b=1:  m - z - M_z * (1-b) <= 0  →  -z + M_z * b <= -m + M_z
        add_ub_row({v_pre: -1.0, v_bin: M_z}, -m + M_z)

        # ── Segment 0 chord (active when b=0) ──
        # y - slope_0 * z <= intercept_0 + r_0 + M_y * b
        add_ub_row({v_out: 1.0, v_pre: -slope_0, v_bin: -M_y}, intercept_0 + r_0)
        # -(y - slope_0 * z) <= -(intercept_0 - r_0) + M_y * b
        add_ub_row({v_out: -1.0, v_pre: slope_0, v_bin: -M_y}, -(intercept_0 - r_0))

        # ── Segment 1 chord (active when b=1) ──
        # y - slope_1 * z <= intercept_1 + r_1 + M_y * (1-b) → ... + M_y - M_y * b
        add_ub_row({v_out: 1.0, v_pre: -slope_1, v_bin: M_y}, intercept_1 + r_1 + M_y)
        # -(y - slope_1 * z) <= -(intercept_1 - r_1) + M_y * (1-b)
        add_ub_row({v_out: -1.0, v_pre: slope_1, v_bin: M_y}, -(intercept_1 - r_1) + M_y)

    return n_wide


__all__ = ['add_kpiece_sigmoid_tanh_milp']
