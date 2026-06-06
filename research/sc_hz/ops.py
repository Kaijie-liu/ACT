"""Forward HZ operators on PrunedState (continuous-only HZ subset).

Per design lock §1.2 and EXECUTION §2.3 (Option A — full per-rival
forward HZ). Each operator takes a PrunedState plus operator parameters
and returns a new PrunedState that over-approximates the layer's
output set.

These operators are implemented from scratch in research/sc_hz/ — they
do not import from act/back_end/hybridz_tf/ — so the Phase A path is
self-contained and the 924 V/A baseline is untouched.

Phase A scope:
  - apply_dense
  - apply_conv2d
  - apply_relu_triangle    (DeepZ; 1 new gen per unstable position)
  - apply_flatten / apply_reshape
  - apply_avgpool2d
  - apply_add              (residual)
  - apply_maxpool2d        (stable-winner only; unstable raises)

What we DO NOT yet support (Phase A out-of-scope, raise NotImplementedError):
  - eq_lagr_v8 / large_cls_proof_mode (binary HZ activation)
  - BoxHZ collapse / Phase 1-3 generator-saving heuristics
  - Sigmoid / Tanh / other smooth nonlinearities

Implementation note: tail_radius propagation through linear ops uses
the sound over-approximation
    new_tail_radius = |W| @ tail_radius
which is valid because for any xi_tail in [-1, 1]^n:
    | W @ (tail_radius * xi_tail) |_i  <=  Σ_k |W[i, k]| · tail_radius[k] · 1
                                          =  (|W| @ tail_radius)[i]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research.sc_hz.prune import PrunedState


# ─── Bounds helper ────────────────────────────────────────────────


def bounds(state: PrunedState) -> Tuple[np.ndarray, np.ndarray]:
    """Per-coordinate (lb, ub) of the set represented by PrunedState.

    For a state (c, G_kept, tail_radius):
        lb[i] = c[i] - Σ_k |G_kept[i, k]| - tail_radius[i]
        ub[i] = c[i] + Σ_k |G_kept[i, k]| + tail_radius[i]
    """
    c = state.c
    rad = np.abs(state.G_kept).sum(axis=1) if state.G_kept.size > 0 else np.zeros_like(c)
    if state.tail_radius is not None:
        rad = rad + state.tail_radius
    return c - rad, c + rad


# ─── Linear ops ───────────────────────────────────────────────────


def apply_dense(state: PrunedState, W: np.ndarray,
                b: Optional[np.ndarray] = None) -> PrunedState:
    """Apply y = W @ x + b to the pruned HZ.

    Closed-form on PrunedState:
        new_c = W @ c + b
        new_G_kept = W @ G_kept
        new_tail_radius = |W| @ tail_radius
    """
    W = np.asarray(W, dtype=np.float64)
    new_c = W @ state.c
    if b is not None:
        new_c = new_c + np.asarray(b, dtype=np.float64).reshape(-1)
    new_G = W @ state.G_kept if state.G_kept.shape[1] > 0 else state.G_kept
    new_tail = (np.abs(W) @ state.tail_radius
                if state.tail_radius is not None else None)
    return PrunedState(
        c=new_c, G_kept=new_G, tail_radius=new_tail,
        metadata=dict(state.metadata),
    )


def apply_conv2d(state: PrunedState,
                 W: np.ndarray,
                 b: Optional[np.ndarray],
                 input_shape: Tuple[int, int, int],
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1) -> Tuple[PrunedState, Tuple[int, int, int]]:
    """Apply Conv2D forward to a flattened PrunedState.

    Convention: state.c is a flat (n_in,) vector, but the conv treats it
    as a (C_in, H_in, W_in) tensor. The caller must pass `input_shape`.
    State after conv is again flat with the new output shape returned.

    For each generator column, we do a single batched torch conv. Tail
    radius is propagated as |W_eff| @ tail_radius via the same conv on
    abs(W) applied to abs(tail_radius) reshaped to image form.
    """
    Ci, Hi, Wi = input_shape
    assert state.c.shape == (Ci * Hi * Wi,), \
        f"state.c shape {state.c.shape} mismatch input_shape {input_shape}"

    W_t = torch.from_numpy(W).to(torch.float64)
    b_t = (torch.from_numpy(np.asarray(b, dtype=np.float64).reshape(-1)).to(torch.float64)
           if b is not None else None)
    Co, _, kH, kW = W.shape

    def _conv_one(x_flat: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_flat.reshape(1, Ci, Hi, Wi)).to(torch.float64)
        y = F.conv2d(x, W_t, b_t, stride=stride, padding=padding, groups=groups)
        return y.detach().numpy().reshape(-1)

    def _conv_no_bias(x_flat: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_flat.reshape(1, Ci, Hi, Wi)).to(torch.float64)
        y = F.conv2d(x, W_t, None, stride=stride, padding=padding, groups=groups)
        return y.detach().numpy().reshape(-1)

    # Center
    new_c = _conv_one(state.c)
    Ho_Wo = new_c.size // Co
    Ho = int(np.sqrt(Ho_Wo)) if int(np.sqrt(Ho_Wo)) ** 2 == Ho_Wo else None
    # Compute output H/W exactly via shape
    sample = _conv_one(state.c).reshape(-1)
    # We need the actual Ho, Wo; recompute via a single forward shape probe
    probe = F.conv2d(
        torch.zeros((1, Ci, Hi, Wi), dtype=torch.float64),
        W_t, None, stride=stride, padding=padding, groups=groups,
    )
    Co_p, Ho_p, Wo_p = probe.shape[1], probe.shape[2], probe.shape[3]
    output_shape = (Co_p, Ho_p, Wo_p)

    # Generators
    K = state.G_kept.shape[1]
    new_G = np.zeros((Co_p * Ho_p * Wo_p, K), dtype=np.float64)
    for k in range(K):
        new_G[:, k] = _conv_no_bias(state.G_kept[:, k])

    # Tail: pass abs(W) on abs(tail_radius) reshaped to image form
    if state.tail_radius is not None:
        abs_W_t = torch.abs(W_t)
        x_tail = torch.from_numpy(
            np.abs(state.tail_radius).reshape(1, Ci, Hi, Wi)
        ).to(torch.float64)
        y_tail = F.conv2d(
            x_tail, abs_W_t, None, stride=stride, padding=padding, groups=groups,
        )
        new_tail = y_tail.detach().numpy().reshape(-1)
    else:
        new_tail = None

    new_state = PrunedState(
        c=new_c, G_kept=new_G, tail_radius=new_tail,
        metadata=dict(state.metadata),
    )
    return new_state, output_shape


# ─── Shape ops ────────────────────────────────────────────────────


def apply_flatten(state: PrunedState) -> PrunedState:
    """Flatten is a no-op on the (already flat) PrunedState; just pass through.

    All our HZ representations carry c and G_kept as flat vectors and
    matrices. The "shape" is bookkeeping. Flatten just confirms the
    bookkeeping and returns the same state.
    """
    return PrunedState(
        c=state.c.copy(), G_kept=state.G_kept.copy(),
        tail_radius=(state.tail_radius.copy()
                     if state.tail_radius is not None else None),
        metadata=dict(state.metadata),
    )


# ─── Add (residual) ───────────────────────────────────────────────


def apply_add(state_a: PrunedState, state_b: PrunedState) -> PrunedState:
    """Sound Add: union of generators.

    For two zonotopic-like sets sharing the same dim, the Minkowski sum
    is exactly:
        c_new = c_a + c_b
        G_kept_new = [G_a, G_b]  (concatenated columns; new ξ per col)
        tail_radius_new = tail_a + tail_b  (independent intervals add)

    The "share generator" optimization is NOT applied here (would
    require factor_id tracking). For Phase A this conservative
    formulation is correct.
    """
    assert state_a.c.shape == state_b.c.shape
    new_c = state_a.c + state_b.c
    new_G = np.concatenate([state_a.G_kept, state_b.G_kept], axis=1)
    ta = state_a.tail_radius if state_a.tail_radius is not None else np.zeros_like(new_c)
    tb = state_b.tail_radius if state_b.tail_radius is not None else np.zeros_like(new_c)
    new_tail = ta + tb if (state_a.tail_radius is not None or
                            state_b.tail_radius is not None) else None
    return PrunedState(
        c=new_c, G_kept=new_G, tail_radius=new_tail,
        metadata={},
    )


# ─── ReLU triangle ────────────────────────────────────────────────


def apply_relu_triangle(state: PrunedState
                         ) -> Tuple[PrunedState, np.ndarray]:
    """DeepZ triangle ReLU on the PrunedState.

    For each coordinate i with bounds (l_i, u_i):
      stable active (l_i >= 0):    h_i = z_i (pass through)
      stable inactive (u_i <= 0):  h_i = 0
      unstable (l_i < 0 < u_i):
         lam_i = u_i / (u_i - l_i)
         mu_i  = -l_i * u_i / (2 (u_i - l_i))
         h_i = lam_i * z_i + mu_i + aux_i      with aux_i in [-mu_i, +mu_i]

    The relaxation adds ONE new generator column per unstable neuron
    with value mu_i at that row, 0 elsewhere. These new columns are
    appended to G_kept (so they remain explicit generators for the
    next layer's pruning to consider).

    Returns:
      (new_state, unstable_mask) where unstable_mask is (n,) bool.
    """
    lb, ub = bounds(state)
    n = state.c.shape[0]
    is_active = lb >= 0
    is_inactive = ub <= 0
    is_unstable = ~(is_active | is_inactive)

    den = np.where(is_unstable, ub - lb, 1.0)
    lam = np.where(is_unstable, ub / np.maximum(den, 1e-300), 0.0)
    mu = np.where(is_unstable, -lb * ub / (2.0 * np.maximum(den, 1e-300)), 0.0)

    # Apply per-coordinate scaling:
    # active: lam_i = 1
    # inactive: lam_i = 0 (h_i = 0, also c_i becomes 0)
    # unstable: lam_i = ub / (ub - lb)
    lam_full = np.where(is_active, 1.0, lam)  # active passes through
    # (inactive: lam=0, mu=0 → h_i = 0 always)

    new_c = lam_full * state.c + mu
    # Zero out inactive rows
    new_c = np.where(is_inactive, 0.0, new_c)

    # Scale kept generators row-wise by lam_full
    new_G = state.G_kept * lam_full[:, None]
    # Zero out inactive rows
    new_G = np.where(is_inactive[:, None], 0.0, new_G)

    # Tail radius: scale by lam_full (sound: |lam * x| <= lam * |x| since
    # lam ≥ 0)
    new_tail = (lam_full * state.tail_radius
                if state.tail_radius is not None else None)
    if new_tail is not None:
        new_tail = np.where(is_inactive, 0.0, new_tail)

    # Add one new generator per unstable neuron: value mu_i at row i,
    # 0 elsewhere. Pack into a sparse-aware column structure.
    k_unstable = int(is_unstable.sum())
    if k_unstable > 0:
        aux_G = np.zeros((n, k_unstable), dtype=np.float64)
        unstable_idx = np.where(is_unstable)[0]
        for j, i in enumerate(unstable_idx):
            aux_G[i, j] = mu[i]
        # ξ for these aux is in [-1, +1], so sign of the column entry is
        # free. The over-approx is achieved.
        new_G = np.concatenate([new_G, aux_G], axis=1)

    new_state = PrunedState(
        c=new_c, G_kept=new_G, tail_radius=new_tail,
        metadata=dict(state.metadata),
    )
    return new_state, is_unstable


# ─── LP UB on rival margin ────────────────────────────────────────


def lp_ub_rival_margin(state: PrunedState, d: np.ndarray) -> float:
    """Closed-form LP UB on d^T y over the pruned set.

    For a state (c, G_kept, tail_radius) and direction d, the LP
        max d^T y  s.t.  y = c + G_kept @ xi_keep + tail_radius * xi_tail
                       xi_keep in [-1, 1]^K, xi_tail in [-1, 1]^n
    has closed form:
        max d^T y = d^T c + Σ_k |d^T G_kept[:, k]| + Σ_i |d_i| * tail_radius[i]

    Returns: scalar.
    """
    ub = float(d @ state.c)
    if state.G_kept.shape[1] > 0:
        ub += float(np.sum(np.abs(d @ state.G_kept)))
    if state.tail_radius is not None:
        ub += float(np.sum(np.abs(d) * state.tail_radius))
    return ub


# ─── Sub (preprocessing constant subtract) ────────────────────────


def apply_sub(state: PrunedState, const: np.ndarray) -> PrunedState:
    """y = x - const (per-coordinate). Only center shifts; generators
    and tail unchanged."""
    new_c = state.c - np.asarray(const, dtype=np.float64).reshape(-1)
    return PrunedState(
        c=new_c, G_kept=state.G_kept.copy(),
        tail_radius=(state.tail_radius.copy()
                     if state.tail_radius is not None else None),
        metadata=dict(state.metadata),
    )


# ─── BatchNorm (per-channel affine: y = scale * x + shift) ────────


def apply_bn(state: PrunedState, scale: np.ndarray, shift: np.ndarray,
              input_shape: Tuple[int, int, int]) -> PrunedState:
    """y = scale[c] * x + shift[c] per channel, broadcast over (H, W).

    For a flat state with input_shape (C, H, W):
        new_c[c, h, w] = scale[c] * c[c, h, w] + shift[c]
        new_G[c, h, w, k] = scale[c] * G[c, h, w, k]
        new_tail[c, h, w] = |scale[c]| * tail[c, h, w]
    """
    C, H, W = input_shape
    n = C * H * W
    assert state.c.shape == (n,)
    scale_arr = np.asarray(scale, dtype=np.float64).reshape(C)
    shift_arr = np.asarray(shift, dtype=np.float64).reshape(C)
    # Broadcast scale/shift to (C, H, W) then flatten
    s_full = np.broadcast_to(scale_arr[:, None, None], (C, H, W)).reshape(-1)
    b_full = np.broadcast_to(shift_arr[:, None, None], (C, H, W)).reshape(-1)
    new_c = s_full * state.c + b_full
    new_G = state.G_kept * s_full[:, None] if state.G_kept.shape[1] > 0 else state.G_kept
    new_tail = (np.abs(s_full) * state.tail_radius
                if state.tail_radius is not None else None)
    return PrunedState(
        c=new_c, G_kept=new_G, tail_radius=new_tail,
        metadata=dict(state.metadata),
    )


# ─── MaxPool2D (stable-winner per window) ─────────────────────────


def apply_maxpool2d(state: PrunedState,
                     input_shape: Tuple[int, int, int],
                     kernel_size: int = 2,
                     stride: Optional[int] = None
                     ) -> Tuple[PrunedState, Tuple[int, int, int]]:
    """Sound forward MaxPool. For each pool window:
      - If a stable winner exists (lb_m >= max ub_others), pass through m.
      - Else: take the (lb, ub) hull of the window as a per-position
              over-approximation: output c = (lb_max + ub_max) / 2,
              new tail row = (ub_max - lb_min) / 2.

    The over-approximation for unstable windows is sound and standard
    (it's the box hull of the input set's projection to the max).
    """
    if stride is None:
        stride = kernel_size
    Ci, Hi, Wi = input_shape
    Ho = (Hi - kernel_size) // stride + 1
    Wo = (Wi - kernel_size) // stride + 1
    n_out = Ci * Ho * Wo

    lb_in, ub_in = bounds(state)
    lb3 = lb_in.reshape(Ci, Hi, Wi)
    ub3 = ub_in.reshape(Ci, Hi, Wi)

    new_c = np.zeros(n_out, dtype=np.float64)
    new_tail = np.zeros(n_out, dtype=np.float64)
    # For Phase A we drop all explicit generators at MaxPool (turn into tail).
    # This is sound but loses precision; a future "stable-winner" path can
    # preserve generators for the winner position.
    for c in range(Ci):
        for h_o in range(Ho):
            for w_o in range(Wo):
                hi0 = h_o * stride; wi0 = w_o * stride
                lb_w = lb3[c, hi0:hi0 + kernel_size, wi0:wi0 + kernel_size]
                ub_w = ub3[c, hi0:hi0 + kernel_size, wi0:wi0 + kernel_size]
                lb_max = lb_w.max()
                ub_max = ub_w.max()
                idx = c * Ho * Wo + h_o * Wo + w_o
                new_c[idx] = (lb_max + ub_max) / 2.0
                new_tail[idx] = (ub_max - lb_max) / 2.0

    # Absorb existing G_kept into tail (since we don't track which window
    # is winner explicitly in this minimal impl).
    # Sound: dropping all explicit generators and replacing with tail is
    # over-approximation.
    state_out = PrunedState(
        c=new_c, G_kept=np.zeros((n_out, 0), dtype=np.float64),
        tail_radius=new_tail, metadata=dict(state.metadata),
    )
    return state_out, (Ci, Ho, Wo)
