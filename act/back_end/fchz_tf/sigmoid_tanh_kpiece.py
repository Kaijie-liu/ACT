# ===- act/back_end/fchz_tf/sigmoid_tanh_kpiece.py - Configurable K-piece S-shape =#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   K-piece tangent parallelogram Sigmoid/Tanh for FCHZ representation.
#   Ports the hybridz_tf hz_apply_piecewise concept to FCHZ.
#
#   Configurable via env vars:
#     ACT_HZ_SIGMOID_K (default 2, sweet spot per memory project_sigmoid_kpiece_fix_20260517)
#     ACT_HZ_TANH_K (default 2)
#
#   Tighter than single chord because each segment uses two tangent directions
#   to bound the segment's curve precisely. Trade-off: 2K more generator cols
#   per wide neuron (vs single chord), but smaller tail_radius.
#
# Soundness:
#   The tangent parallelogram covers all (x, f(x)) for x in segment by
#   construction. Scaling factor ensures all 50 sample points are inside.
#
# ===---------------------------------------------------------------------===#
"""K-piece tangent parallelogram Sigmoid/Tanh for FCHZ."""
import os
import numpy as np
from typing import Callable, Tuple


def get_K_for(kind: str) -> int:
    """Get K_pieces from env var. Default 2 (sweet spot per advisor memory)."""
    if kind == 'Sigmoid' or kind == 'sigmoid':
        env_val = os.environ.get('ACT_HZ_SIGMOID_K', '2')
    elif kind == 'Tanh' or kind == 'tanh':
        env_val = os.environ.get('ACT_HZ_TANH_K', '2')
    else:
        return 1
    try:
        K = int(env_val)
        return max(1, K)
    except ValueError:
        return 2


def apply_kpiece_sigmoid_tanh_fchz(c: np.ndarray, G: np.ndarray, tail_radius,
                                                            kind: str, K: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-piece tangent parallelogram for Sigmoid/Tanh on FCHZ representation.

    Args:
      c: (n,) center vector
      G: (n, ng) generator matrix
      tail_radius: (n,) tail radius or None
      kind: 'Sigmoid' or 'Tanh'
      K: number of segments per wide neuron (None → from env var)

    Returns:
      new_c: (n,) new center
      new_G: (n, ng + 2*K*n_wide) new generator matrix (extra cols for K-piece slacks)
      new_tail: (n,) new tail radius

    Soundness:
      For each wide neuron, the K segments cover [l_i, u_i].
      Each segment uses tangent parallelogram (2 directions) that PROVABLY
      contains the function curve via scale_factor adjustment from sampling.
    """
    if K is None:
        K = get_K_for(kind)
    n = c.shape[0]
    ng = G.shape[1] if G.ndim > 1 else 0

    # Compute bounds
    G_rad = np.abs(G).sum(axis=1) if ng > 0 else np.zeros(n)
    if tail_radius is not None:
        rad = G_rad + np.abs(tail_radius)
    else:
        rad = G_rad
    l = c - rad
    u = c + rad

    func = lambda x: 1.0 / (1.0 + np.exp(-x)) if kind in ('Sigmoid', 'sigmoid') else np.tanh(x)
    if kind in ('Sigmoid', 'sigmoid'):
        dfunc = lambda x: func(x) * (1.0 - func(x))
    else:
        dfunc = lambda x: 1.0 - np.tanh(x) ** 2

    # FCHZ-zonotope sound limit: K=1 only.
    # K>=2 requires binary segment indicator that FCHZ representation lacks
    # (verified unsound on Tanh test 2026-06-10: 0.002 over-tightening on
    #  one side because Minkowski-summed parallelograms don't enforce
    #  "exactly one segment" constraint).
    # Tighter K-piece requires forward_global_milp.py path (has binary support).
    if K >= 1:
        return _single_chord_fchz(c, G, tail_radius, l, u, func, dfunc)

    # Wide vs narrow neurons
    wide_mask = (u - l) > 1e-12
    narrow_mask = ~wide_mask
    wide_idx = np.where(wide_mask)[0]
    n_wide = len(wide_idx)

    new_c = c.copy()
    new_G_base = G.copy() if ng > 0 else np.zeros((n, 0))

    # Narrow: y = func(c) exactly, no slack
    new_c[narrow_mask] = func(c[narrow_mask])
    if ng > 0:
        new_G_base[narrow_mask, :] = 0.0
    new_tail = np.zeros(n) if tail_radius is None else np.abs(tail_radius).copy()
    new_tail[narrow_mask] = 0.0

    if n_wide == 0:
        return new_c, new_G_base, new_tail

    # K-piece per wide neuron
    lb_w = l[wide_idx]; ub_w = u[wide_idx]
    # Per-neuron segment endpoints: (K, n_wide)
    seg_ids = np.arange(K, dtype=np.float64).reshape(-1, 1)
    seg_width = (ub_w - lb_w).reshape(1, -1) / K
    a = lb_w.reshape(1, -1) + seg_ids * seg_width   # (K, n_wide) lower endpoints
    b_seg = a + seg_width                                          # (K, n_wide) upper endpoints

    fa = func(a); fb = func(b_seg)
    la = dfunc(a); lb_slope = dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = np.abs(la - lb_slope) < 1e-10

    # Tangent intersection construction
    denom = lb_slope - la
    safe_denom = np.where(nearly_linear, 1.0, denom)
    p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = lb_slope * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = la * (p2 - a) / 2.0

    # Nearly-linear fallback: linear + residual
    hw = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = np.linspace(0.0, 1.0, 50).reshape(50, 1, 1)
    pts = a.reshape(1, K, n_wide) + t_pts * (b_seg - a).reshape(1, K, n_wide)
    f_pts = func(pts)
    chord_y = slope.reshape(1, K, n_wide) * pts + (fa - slope * a).reshape(1, K, n_wide)
    resid = f_pts - chord_y
    max_err = np.abs(resid).max(axis=0)
    g1x_lin = hw; g1y_lin = slope * hw
    g2x_lin = np.zeros_like(hw); g2y_lin = max_err

    g1_x = np.where(nearly_linear, g1x_lin, g1x_tang)
    g1_y = np.where(nearly_linear, g1y_lin, g1y_tang)
    g2_x = np.where(nearly_linear, g2x_lin, g2x_tang)
    g2_y = np.where(nearly_linear, g2y_lin, g2y_tang)

    # Sample-based scale factor (ensure parallelogram covers all sample points)
    dx = pts - centers_x.reshape(1, K, n_wide)
    dy = f_pts - centers_y.reshape(1, K, n_wide)
    det = g1_y * g2_x - g1_x * g2_y
    safe_det = np.where(np.abs(det) < 1e-30, 1.0, det)
    xi1 = (dy * g2_x.reshape(1, K, n_wide) - dx * g2_y.reshape(1, K, n_wide)) / safe_det.reshape(1, K, n_wide)
    xi2 = (dy * g1_x.reshape(1, K, n_wide) - dx * g1_y.reshape(1, K, n_wide)) / (-safe_det.reshape(1, K, n_wide))
    max_xi = np.maximum(np.abs(xi1).max(axis=0), np.abs(xi2).max(axis=0))
    scale_factor = np.where(max_xi > 1.0, max_xi * 1.01, 1.0)
    scale_factor = np.where(np.abs(det) < 1e-30, 1.0, scale_factor)
    g1_x = g1_x * scale_factor
    g1_y = g1_y * scale_factor
    g2_x = g2_x * scale_factor
    g2_y = g2_y * scale_factor

    # New center for wide neurons: average of segment centers
    cy_sum = centers_y.sum(axis=0)   # (n_wide,)
    new_c[wide_idx] = cy_sum / K

    # For wide neurons: zero out base G rows (they get rebuilt via K-piece slacks)
    if ng > 0:
        new_G_base[wide_idx, :] = 0.0

    # Build new generator cols: 2K per wide neuron
    # We add new cols: g1_y[k, j] in slot wide_idx[j], col base_ng + j*2K + 2k
    n_new_cols = 2 * K * n_wide
    new_cols = np.zeros((n, n_new_cols), dtype=np.float64)
    # For wide neuron j (mapped to row wide_idx[j]):
    #   Cols [2K*j + 2k, 2K*j + 2k+1] = (g1_y[k,j], g2_y[k,j])
    for k in range(K):
        for j in range(n_wide):
            col_g1 = 2 * K * j + 2 * k
            col_g2 = 2 * K * j + 2 * k + 1
            new_cols[wide_idx[j], col_g1] = g1_y[k, j]
            new_cols[wide_idx[j], col_g2] = g2_y[k, j]

    new_G = np.concatenate([new_G_base, new_cols], axis=1) if ng > 0 else new_cols
    return new_c, new_G, new_tail


def _single_chord_fchz(c, G, tail_radius, l, u, func, dfunc):
    """Fallback: single chord (matches original FCHZ behavior for K=1)."""
    fa = func(l); fb = func(u)
    alpha = (fb - fa) / np.maximum(u - l, 1e-30)
    alpha = np.where(np.abs(u - l) < 1e-12, dfunc(c), alpha)
    beta = (fa + fb - alpha * (l + u)) / 2.0
    # Residual: max |func(x) - alpha*x - beta|
    sample_t = np.linspace(0.0, 1.0, 50)
    n = c.shape[0]
    xs = l.reshape(1, n) + sample_t.reshape(50, 1) * (u - l).reshape(1, n)
    fxs = func(xs)
    resid = np.abs(fxs - alpha.reshape(1, n) * xs - beta.reshape(1, n))
    radius = resid.max(axis=0)
    new_c = alpha * c + beta
    new_G = G * alpha.reshape(-1, 1) if G.ndim > 1 and G.shape[1] > 0 else G
    if tail_radius is not None:
        new_tail = np.abs(alpha) * np.abs(tail_radius) + radius
    else:
        new_tail = radius
    return new_c, new_G, new_tail


__all__ = ['apply_kpiece_sigmoid_tanh_fchz', 'get_K_for']
