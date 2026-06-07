# ===- act/back_end/fchz_tf/representations.py - FCHZ State -----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   FCHZState data structure for strict P1-P5 forward HZ propagation.
#
#   Mathematical representation:
#     R(s) = { c + G·ξ + δ : ξ ∈ [-1,1]^K, δ_i ∈ [-tail_radius_i, +tail_radius_i] }
#
#   Where:
#     c              : center (n,)
#     G              : generators (n, K)
#     tail_radius    : per-row INDEPENDENT box error (n,) ≥ 0
#     slack_records  : per-ReLU layer triangle constraints (for F1_LP audit)
#     n_root         : input dimension (n_in)
#
# Soundness invariants — see TAIL_RADIUS_SOUNDNESS_PROOF.md.
#
# ===---------------------------------------------------------------------===#

"""FCHZState — forward-only constrained hybrid zonotope representation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class SlackRecord:
    """Per-ReLU-layer triangle constraint record (for F1_LP audit)."""
    layer_index: int
    c_z: np.ndarray              # (n_layer,)  pre-activation center
    G_z: np.ndarray              # (n_layer, K)
    l: np.ndarray                # (n_layer,)  pre-act lower bound
    u: np.ndarray                # (n_layer,)
    slack_indices: np.ndarray    # (n_unstable,) which xi cols are slacks for this layer
    unstable_mask_arr: Optional[np.ndarray] = None  # (n_layer,) bool

    def stable_active_mask(self) -> np.ndarray:
        return self.l >= 0

    def stable_inactive_mask(self) -> np.ndarray:
        return self.u <= 0

    def unstable_mask(self) -> np.ndarray:
        if self.unstable_mask_arr is not None:
            return self.unstable_mask_arr
        return (self.l < 0) & (self.u > 0)


@dataclass
class FCHZState:
    """Forward Constrained Hybrid Zonotope state.

    Represents reachable set:
        R(s) = { c + G·ξ + δ :
                 ξ ∈ [-1, 1]^K,
                 δ_i ∈ [-tail_radius_i, +tail_radius_i] (per-row independent)
               }

    HZ closed-form upper bound:
        max d·x = d·c + Σ_k |d·G_k| + Σ_i |d_i|·tail_radius_i
    """
    c: np.ndarray                          # (n,)  center
    G: np.ndarray                          # (n, K) generators
    n_root: int                            # input dimension
    slack_records: List[SlackRecord] = field(default_factory=list)
    tail_radius: Optional[np.ndarray] = None  # (n,) ≥ 0  per-row independent box error

    @property
    def n(self) -> int:
        """State dimension."""
        return int(self.c.shape[0])

    @property
    def K(self) -> int:
        """Number of generator columns."""
        return int(self.G.shape[1])


def initial_state(c: np.ndarray, r: np.ndarray) -> FCHZState:
    """Build initial state from input box [c-r, c+r].

    State n = len(c), K = n (identity generator matrix).
    """
    n = c.shape[0]
    G = np.diag(r)
    return FCHZState(
        c=c.copy(),
        G=G,
        n_root=n,
        slack_records=[],
        tail_radius=None,
    )


def hz_closed_form_ub(state: FCHZState, d_out: np.ndarray) -> float:
    """HZ closed-form bound: max d·x over R(state).

    Sound formula:
        UB = d·c + Σ_k |d·G_k| + Σ_i |d_i|·tail_radius_i
    """
    val = float(d_out @ state.c) + float(np.abs(d_out @ state.G).sum())
    if state.tail_radius is not None:
        val += float(np.abs(d_out) @ state.tail_radius)
    return val


def apply_dense(state: FCHZState, W: np.ndarray, b: Optional[np.ndarray]) -> FCHZState:
    """Dense layer: y = W @ x + b.

    State propagation (sound):
        c' = W·c + b
        G' = W·G
        tail_radius' = |W| @ tail_radius
    """
    new_c = W @ state.c
    if b is not None:
        new_c = new_c + b
    new_G = W @ state.G
    new_tail = None
    if state.tail_radius is not None:
        new_tail = np.abs(W) @ state.tail_radius
    return FCHZState(
        c=new_c, G=new_G, n_root=state.n_root,
        slack_records=state.slack_records,
        tail_radius=new_tail,
    )


def compress_g_to_tail(state: FCHZState, K_max: int) -> FCHZState:
    """Sparse-slack compression: keep top-K_max G columns by L∞ norm.

    Sound: dropped columns absorbed into per-row tail_radius.
    See SPARSE_SLACK_DESIGN.md §6 for proof: R(s_new) ⊇ R(s_old).
    """
    if state.G.shape[1] <= K_max:
        return state
    col_inf = np.abs(state.G).max(axis=0)
    order = np.argsort(col_inf)[::-1]
    keep = order[:K_max]
    drop = order[K_max:]
    new_G = state.G[:, keep]
    extra_tail = np.abs(state.G[:, drop]).sum(axis=1)
    if state.tail_radius is not None:
        new_tail = state.tail_radius + extra_tail
    else:
        new_tail = extra_tail
    return FCHZState(
        c=state.c.copy(), G=new_G,
        n_root=state.n_root, slack_records=[],
        tail_radius=new_tail,
    )
