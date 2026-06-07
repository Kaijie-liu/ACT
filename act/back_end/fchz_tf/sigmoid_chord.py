# ===- act/back_end/fchz_tf/sigmoid_chord.py - Sigmoid/Tanh chord -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Analytical sound chord relaxation for Sigmoid and Tanh.
#
#   For each row [l_i, u_i]:
#     1. Chord through (l_i, σ(l_i)) and (u_i, σ(u_i))
#     2. Compute critical x* where σ'(x*) = chord_slope (analytical)
#     3. Recenter β at midpoint of deviation, radius = half-spread
#
#   Sigmoid: σ'(x) = α  ⇒  σ(x) = (1 ± √(1-4α))/2  ⇒  x* = logit(σ)
#   Tanh:    σ'(x) = α  ⇒  σ(x) = ±√(1-α)         ⇒  x* = atanh(σ)
#
#   Per-row independent box error → added to tail_radius (sound).
#
# ===---------------------------------------------------------------------===#

"""Analytical sigmoid/tanh chord linear relaxation.

SOUNDNESS SOURCE (analytical, NOT sampling):
    σ(x) - α x - β has critical points where σ'(x*) = α
    - Sigmoid: σ(x*)(1 - σ(x*)) = α  ⇒  σ(x*) = (1 ± √(1−4α))/2,  x* = logit(σ)
    - Tanh:    1 - σ(x*)² = α        ⇒  σ(x*) = ±√(1−α),         x* = atanh(σ)
    Sound radius = max(|σ(x*) - α x* - β|) over valid x* ∈ (l, u).

The 200k fine-grid test in test_fchz_tf.py is a numerical sanity check
to catch implementation regressions. It is NOT the soundness source.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def chord_params(l: np.ndarray, u: np.ndarray, kind: str) -> Tuple[np.ndarray, ...]:
    """Compute analytical chord bound (alpha, beta, radius) for Sigmoid or Tanh.

    Returns:
        alpha: (n,) chord slope per row
        beta:  (n,) chord intercept (re-centered to midpoint of deviation)
        radius: (n,) ≥ 0 sound box error per row (added to tail_radius)
    """
    if kind == "Sigmoid":
        fn = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        max_slope = 0.25
    elif kind == "Tanh":
        fn = np.tanh
        max_slope = 1.0
    else:
        raise ValueError(f"Unknown kind: {kind}")

    fl = fn(l); fu = fn(u)
    den = np.maximum(u - l, 1e-12)
    alpha = (fu - fl) / den
    alpha = np.where(u - l < 1e-12, 0.0, alpha)
    beta = fl - alpha * l

    # Critical points where σ'(x) = α
    dev_max_row = np.zeros_like(l)
    dev_min_row = np.zeros_like(l)
    alpha_safe = np.minimum(alpha, max_slope - 1e-12)
    if kind == "Sigmoid":
        disc = np.sqrt(np.maximum(1.0 - 4.0 * alpha_safe, 0.0))
    else:  # Tanh
        disc = np.sqrt(np.maximum(1.0 - alpha_safe, 0.0))
    valid = (alpha > 1e-15) & (alpha < max_slope)

    for sign in [+1.0, -1.0]:
        if kind == "Sigmoid":
            s_val = (1.0 + sign * disc) / 2.0
            s_safe = np.clip(s_val, 1e-15, 1.0 - 1e-15)
            x_star = np.log(s_safe / (1.0 - s_safe))
        else:
            s_val = sign * disc
            s_safe = np.clip(s_val, -1.0 + 1e-15, 1.0 - 1e-15)
            x_star = 0.5 * np.log((1.0 + s_safe) / (1.0 - s_safe))
        in_range = valid & (x_star > l) & (x_star < u)
        dev_star = fn(x_star) - (alpha * x_star + beta)
        dev_max_row = np.where(in_range, np.maximum(dev_max_row, dev_star), dev_max_row)
        dev_min_row = np.where(in_range, np.minimum(dev_min_row, dev_star), dev_min_row)

    # Recenter
    mid_dev = (dev_max_row + dev_min_row) / 2.0
    radius = (dev_max_row - dev_min_row) / 2.0
    beta = beta + mid_dev

    return alpha, beta, radius
