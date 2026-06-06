"""Phase G: Forward Constrained Hybrid Zonotope (FC-HZ) — Dense+ReLU only prototype.

Per advisor 2026-06-05 directive: F1 (last-ReLU triangle) gives 17% real
tightening but is insufficient (cifar 113 still PHANTOM). F2b (pairwise
same-layer cuts) gives 0% on real cifar (LP optimum spreads across many
neurons). The actual remaining looseness is AGGREGATE across MULTIPLE
ReLU LAYERS — earlier-layer slacks are treated as fully free, allowing
LP to push them all to worst simultaneously.

FC-HZ fixes this by carrying ALL per-layer triangle constraints forward
into the output LP.

PRINCIPLE compliance:
- Forward-only (no backward bound refinement)
- Continuous LP (no MILP, no integers)
- No BaB, no random falsifier, no gradient

This module implements the minimal Dense + ReLU prototype:
- FCHZState: (c, G, slack_records)
- apply_dense
- apply_relu_triangle_with_record  (records (z_pre_affine, l, u, slack_index))
- fc_hz_lp_ub: solve full constrained LP with ALL layer triangles

For Phase G.0 toy validation only. Conv/BN/MaxPool/Add later if G.0 passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.optimize as sopt


@dataclass
class SlackRecord:
    """For each ReLU layer L, record the per-unstable-neuron info needed for the
    output LP constraints.

    For neuron i in layer L (unstable):
      z_L_i = c_z_L[i] + G_z_L[i, :] @ xi   (linear in xi at time of layer L)
      lam_i = u_L[i] / (u_L[i] - l_L[i])
      mu_i  = -lam_i * l_L[i] / 2
      slack_idx_i: column in the FULL xi vector where this slack lives
      y_L_i = lam_i * z_L_i + mu_i * (1 + xi[slack_idx_i])
    """
    layer_index: int                  # which ReLU layer (0, 1, ...)
    c_z: np.ndarray                   # (n_layer,)
    G_z: np.ndarray                   # (n_layer, K_at_time)  pre-act coefficients
    l: np.ndarray                     # (n_layer,) pre-act lower bound
    u: np.ndarray                     # (n_layer,) pre-act upper bound
    unstable_indices: np.ndarray      # (n_unstable,) of neuron indices
    slack_indices: np.ndarray         # (n_unstable,) of slack column in extended xi


@dataclass
class FCHZState:
    """Forward Constrained HZ state.

    The reachable set is:
      { c + G @ xi + tail_disturbance : xi ∈ [-1, 1]^K,
        tail_disturbance_i ∈ [-tail_radius_i, +tail_radius_i] per row,
        for each layer L: y_L_i ≥ 0, y_L_i ≥ z_L_i, y_L_i ≤ chord(z_L_i) }

    The tail_radius captures per-row independent slack contributions
    that we don't store as full generator columns (for memory).
    Sound HZ closed-form: d·c + sum_k |d·G_k| + sum_i |d_i| · tail_radius_i.
    """
    c: np.ndarray                     # (n,)  output center
    G: np.ndarray                     # (n, K) generator matrix (root + ALL slacks)
    n_root: int                       # number of root xi columns
    slack_records: List[SlackRecord] = field(default_factory=list)
    tail_radius: Optional[np.ndarray] = None  # (n,) per-row tail (sound box bound)

    @property
    def n(self) -> int:
        return self.c.shape[0]

    @property
    def K(self) -> int:
        return self.G.shape[1]


def initial_state(c_in: np.ndarray, r_in: np.ndarray) -> FCHZState:
    """Initial box state: x ∈ [c_in - r_in, c_in + r_in]."""
    n = c_in.shape[0]
    return FCHZState(
        c=c_in.copy(),
        G=np.diag(r_in.astype(np.float64)),
        n_root=n,
        slack_records=[],
    )


def _propagate_tail(tail_r: Optional[np.ndarray],
                          abs_W: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Propagate tail_radius through linear op: new_tail = |W| @ tail."""
    if tail_r is None: return None
    if abs_W is None: return tail_r
    return abs_W @ tail_r


def apply_dense(state: FCHZState, W: np.ndarray, b: Optional[np.ndarray]) -> FCHZState:
    """Linear: y = W @ x + b."""
    new_c = W @ state.c
    if b is not None:
        new_c = new_c + b
    new_G = W @ state.G
    new_tail = _propagate_tail(state.tail_radius, np.abs(W) if state.tail_radius is not None else None)
    return FCHZState(
        c=new_c, G=new_G, n_root=state.n_root,
        slack_records=state.slack_records,  # unchanged — slack records keep G_z at recording time
        tail_radius=new_tail,
    )


def apply_relu_triangle_with_record(
    state: FCHZState, layer_index: int,
) -> FCHZState:
    """Apply ReLU triangle, recording per-unstable-neuron data."""
    n = state.n
    K = state.K
    # Bounds: z_i ∈ [c_i - |G_i|.sum(), c_i + |G_i|.sum()]
    rad = np.abs(state.G).sum(axis=1)
    l = state.c - rad
    u = state.c + rad

    is_active = l >= 0
    is_inactive = u <= 0
    is_unstable = ~is_active & ~is_inactive

    unstable_idx = np.where(is_unstable)[0]
    n_unstable = len(unstable_idx)

    # Record (BEFORE any modification — z_pre is current state's c, G)
    rec = SlackRecord(
        layer_index=layer_index,
        c_z=state.c.copy(),
        G_z=state.G.copy(),
        l=l.copy(),
        u=u.copy(),
        unstable_indices=unstable_idx.copy(),
        slack_indices=np.arange(K, K + n_unstable, dtype=np.int64),
    )

    # Build new state after ReLU:
    den = np.where(is_unstable, u - l, 1.0)
    lam = np.where(is_unstable, u / np.maximum(den, 1e-300), 0.0)
    lam = np.where(is_active, 1.0, lam)
    lam = np.where(is_inactive, 0.0, lam)
    mu = np.where(is_unstable, -lam * l / 2.0, 0.0)
    # The triangle relaxation gives: y = lam*z + mu + mu*s, s ∈ [-1, 1]
    new_c = lam * state.c + mu
    new_G = (lam[:, None] * state.G)  # (n, K)
    # Add aux slack columns: one per unstable neuron, value mu_i in row i
    if n_unstable > 0:
        slack_cols = np.zeros((n, n_unstable), dtype=np.float64)
        for slot, i in enumerate(unstable_idx):
            slack_cols[i, slot] = mu[i]
        new_G_extended = np.concatenate([new_G, slack_cols], axis=1)
    else:
        new_G_extended = new_G

    new_records = state.slack_records + [rec]
    return FCHZState(
        c=new_c, G=new_G_extended, n_root=state.n_root,
        slack_records=new_records,
    )


def fc_hz_lp_ub(state: FCHZState, d_out: np.ndarray) -> Tuple[float, dict]:
    """Solve constrained LP for max d_out @ output_value.

    Variables: xi (K,) ∈ [-1, 1]
    Objective: max d_out @ (c + G @ xi) = d_out @ c + (d_out @ G) @ xi
    Constraints (per layer L, per unstable neuron i in L):
      y_L_i = lam_L_i * z_L_i + mu_L_i * (1 + xi[slack_idx])
         where z_L_i = c_z_L[i] + G_z_L[i, :] @ xi[0:K_at_time]
      y_L_i ≥ 0  →  lam * z + mu + mu * xi[slack_idx] ≥ 0
                  →  -mu * xi[slack_idx] - lam * G_z[i, :] @ xi_prev ≤ lam * c_z[i] + mu
      y_L_i ≥ z_L_i  →  (lam - 1) * z + mu + mu * xi[slack_idx] ≥ 0
                  →  -(lam - 1) * G_z[i, :] @ xi_prev - mu * xi[slack_idx] ≤ (lam - 1) * c_z[i] + mu
                  →  (1 - lam) * G_z[i, :] @ xi_prev - mu * xi[slack_idx] ≤ -(1 - lam) * c_z[i] + mu
      y_L_i ≤ chord(z_L_i) = lam * (z - l)
                  →  lam * z + mu + mu * xi[slack_idx] ≤ lam * z - lam * l
                  →  mu + mu * xi[slack_idx] ≤ -lam * l
                  →  mu * xi[slack_idx] ≤ -lam * l - mu = -mu (since mu = -lam*l/2)
                  →  xi[slack_idx] ≤ -1 (already encoded by box)
       Wait — this is automatically satisfied by xi ∈ [-1, 1]. So upper triangle
       is FREE in our box-domain encoding.

    So the ONLY constraints we need to add are:
      y_L_i ≥ 0  AND  y_L_i ≥ z_L_i
    """
    K = state.K
    n_vars = K
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[:] = -(d_out @ state.G)  # min -(d_out @ G) @ xi
    obj_const = float(d_out @ state.c)
    bounds = [(-1.0, 1.0)] * n_vars

    A_ub_rows = []
    b_ub_rows = []

    for rec in state.slack_records:
        K_at_time = rec.G_z.shape[1]
        for slot, i in enumerate(rec.unstable_indices):
            l_i = float(rec.l[i])
            u_i = float(rec.u[i])
            den = max(u_i - l_i, 1e-300)
            lam = u_i / den
            mu = -lam * l_i / 2.0
            slack_idx = int(rec.slack_indices[slot])
            G_z_i = rec.G_z[i, :]
            c_z_i = float(rec.c_z[i])
            # Constraint: y_L_i ≥ 0
            #   lam * c_z + lam * G_z @ xi_prev + mu + mu * xi[slack_idx] ≥ 0
            #   - lam * G_z @ xi_prev - mu * xi[slack_idx] ≤ lam * c_z + mu
            row = np.zeros(n_vars)
            row[:K_at_time] = -lam * G_z_i
            row[slack_idx] = -mu
            A_ub_rows.append(row)
            b_ub_rows.append(lam * c_z_i + mu)
            # Constraint: y_L_i ≥ z_L_i
            #   lam * z + mu + mu * xi[slack_idx] ≥ z
            #   (lam - 1) * z + mu + mu * xi[slack_idx] ≥ 0
            #   (lam - 1) * c_z + (lam - 1) * G_z @ xi_prev + mu + mu * xi[slack_idx] ≥ 0
            #   -(lam - 1) * G_z @ xi_prev - mu * xi[slack_idx] ≤ (lam - 1) * c_z + mu
            row = np.zeros(n_vars)
            row[:K_at_time] = -(lam - 1.0) * G_z_i
            row[slack_idx] = -mu
            A_ub_rows.append(row)
            b_ub_rows.append((lam - 1.0) * c_z_i + mu)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    res = sopt.linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        return float("inf"), {"status": res.status, "n_constraints": len(A_ub_rows)}
    ub = obj_const - float(res.fun)
    return float(ub), {"status": 0, "n_constraints": len(A_ub_rows)}


def hz_closed_form_ub(state: FCHZState, d_out: np.ndarray) -> float:
    """Plain HZ closed-form (no constraints): d_out·c + |d_out·G|.sum() + |d|·tail_radius."""
    val = float(d_out @ state.c) + float(np.abs(d_out @ state.G).sum())
    if state.tail_radius is not None:
        val += float(np.abs(d_out) @ state.tail_radius)
    return val


def compress_g_to_tail(state: FCHZState, K_max: int) -> FCHZState:
    """Sparse-slack compression: keep top-K_max G columns by L∞ norm; absorb
    the rest into per-row tail_radius.

    Sound proof: see research/sc_hz/SPARSE_SLACK_DESIGN.md §6. For each
    dropped column k, add |G[:,k]| to tail_radius per row. R(s_new) ⊇ R(s_old).
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
    # Drop slack records whose generator columns we dropped — they are no
    # longer in G, so cannot be enforced as triangle constraints. Records
    # for retained columns are kept (column indices still valid).
    # Simplest sound: drop ALL slack records (their G_z entries reference
    # the OLD K-dim xi). New_G has fewer cols, so a stricter LP would index
    # out-of-bounds. Compression sacrifices the per-layer triangle inequality
    # constraint precision, replaced by per-row tail_radius bound.
    return FCHZState(c=state.c.copy(), G=new_G,
                          n_root=state.n_root, slack_records=[],
                          tail_radius=new_tail)


def f1_last_relu_lp_ub(state: FCHZState, d_out: np.ndarray) -> float:
    """F1-equivalent: only the LAST ReLU layer's triangle constraints in LP.

    For comparison vs FC-HZ.
    """
    K = state.K
    n_vars = K
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[:] = -(d_out @ state.G)
    obj_const = float(d_out @ state.c)
    bounds = [(-1.0, 1.0)] * n_vars

    A_ub_rows = []
    b_ub_rows = []

    if state.slack_records:
        rec = state.slack_records[-1]  # ONLY the last layer
        K_at_time = rec.G_z.shape[1]
        for slot, i in enumerate(rec.unstable_indices):
            l_i = float(rec.l[i])
            u_i = float(rec.u[i])
            den = max(u_i - l_i, 1e-300)
            lam = u_i / den
            mu = -lam * l_i / 2.0
            slack_idx = int(rec.slack_indices[slot])
            G_z_i = rec.G_z[i, :]
            c_z_i = float(rec.c_z[i])
            # y ≥ 0
            row = np.zeros(n_vars)
            row[:K_at_time] = -lam * G_z_i
            row[slack_idx] = -mu
            A_ub_rows.append(row)
            b_ub_rows.append(lam * c_z_i + mu)
            # y ≥ z
            row = np.zeros(n_vars)
            row[:K_at_time] = -(lam - 1.0) * G_z_i
            row[slack_idx] = -mu
            A_ub_rows.append(row)
            b_ub_rows.append((lam - 1.0) * c_z_i + mu)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    res = sopt.linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        return float("inf")
    return obj_const - float(res.fun)
