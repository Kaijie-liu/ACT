# ===- act/back_end/fchz_tf/f1_lp_refine.py - F1 LP refinement for FchzTF ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   F1 LP refinement (research advisor 2026-06-05 Phase F direction):
#   close the gap between FCHZ closed-form (independent ξ aux per slack)
#   and the true triangle-relaxation upper bound by explicitly enforcing
#   the ReLU triangle constraints in a CONTINUOUS LP at the LAST ReLU.
#
#   For specs C @ y < threshold (TOP1_ROBUST / UNSAFE_LINEAR / LINEAR_LE),
#   compute per-direction UB via LP and combine with closed-form via min:
#
#       UB_refined(d) = min(closed_form_UB, LP_UB)
#
#   Sound (LP is a tighter relaxation of the same set).
#
# Status (advisor 2026-06-08): EXPERIMENT — off-by-default in production.
#   Sentinel result: 0 NEW V / 100 across acasxu/linearizenn/sat_relu/tll/lsnc.
#   Diagnosis: for deep MLPs (acasxu has 7 ReLU layers), accumulated slack at
#   the pre-last-ReLU state already has ~200 generator cols. Last-ReLU-only
#   LP refinement cannot recover correlation loss from middle layers.
#   Use `verify_once_fchz(..., use_f1_lp=True)` only as opt-in research toggle.
#   M4 direction: multi-layer forward triangle LP (handle 2+ ReLUs jointly).
#
# Principle compliance:
#   - Forward-only (LP solved AFTER forward propagation)
#   - Continuous LP (HiGHS / scipy.optimize.linprog; no MILP, no integers)
#   - No backward gradient, no BaB, no PGD
#
# ===---------------------------------------------------------------------===#

"""F1 LP refinement — explicit ReLU triangle in LP at the last ReLU."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def find_last_relu_layer(net):
    """Find the last RELU layer in the net (by id descending)."""
    for L in reversed(net.layers):
        if L.kind == 'RELU':
            return L
    return None


def collect_post_relu_affine(net, last_relu_id: int, tf):
    """Compose W_remaining, b_remaining from layers AFTER last ReLU.

    Walks DENSE / BIAS / SCALE / etc. and folds into a single affine map.
    Returns (W_remaining: (n_out, n_pre), b_remaining: (n_out,)) or None
    if there's a non-affine layer after last ReLU that we can't handle.
    """
    # Collect affine layers after last_relu_id (excluding ASSERT)
    post_layers = []
    found_relu = False
    for L in net.layers:
        if L.id == last_relu_id:
            found_relu = True
            continue
        if not found_relu: continue
        if L.kind == 'ASSERT': break
        post_layers.append(L)
        if L.kind not in ('DENSE', 'BIAS', 'SCALE', 'FLATTEN'):
            return None, None    # unsupported non-affine layer
    # Compose: start with identity, apply each layer's affine map
    # Determine n_pre from FchzTF state at last_relu (post-ReLU state)
    state_after_relu = tf._state_cache.get(last_relu_id)
    if state_after_relu is None:
        return None, None
    n_pre = state_after_relu.c.shape[0]
    W = np.eye(n_pre, dtype=np.float64)
    b = np.zeros(n_pre, dtype=np.float64)
    for L in post_layers:
        if L.kind == 'DENSE':
            W_l = L.params.get('weight')
            b_l = L.params.get('bias')
            if W_l is None: return None, None
            W_np = W_l.detach().cpu().numpy().astype(np.float64) if hasattr(W_l, 'detach') else np.asarray(W_l, dtype=np.float64)
            b_np = (b_l.detach().cpu().numpy().astype(np.float64).flatten() if hasattr(b_l, 'detach')
                       else np.asarray(b_l, dtype=np.float64).flatten() if b_l is not None
                       else np.zeros(W_np.shape[0], dtype=np.float64))
            # y_new = W_np @ y + b_np = W_np @ (W @ x + b) + b_np = (W_np @ W) @ x + (W_np @ b + b_np)
            b = W_np @ b + b_np
            W = W_np @ W
        elif L.kind == 'BIAS':
            b_l = L.params.get('c') if 'c' in L.params else L.params.get('bias')
            if b_l is None: return None, None
            b_np = (b_l.detach().cpu().numpy().astype(np.float64).flatten() if hasattr(b_l, 'detach')
                       else np.asarray(b_l, dtype=np.float64).flatten())
            b = b + b_np
        elif L.kind == 'SCALE':
            a_l = L.params.get('a') if 'a' in L.params else L.params.get('scale')
            if a_l is None: return None, None
            a_np = (a_l.detach().cpu().numpy().astype(np.float64).flatten() if hasattr(a_l, 'detach')
                       else np.asarray(a_l, dtype=np.float64).flatten())
            # y = a * x (elementwise) → W rows scaled
            W = a_np[:, None] * W
            b = a_np * b
        elif L.kind == 'FLATTEN':
            pass    # no-op for affine composition
        else:
            return None, None
    return W, b


def build_last_relu_record(state_pre_relu, n_out_pre):
    """Build LastReluRecord from FCHZState just BEFORE the ReLU.

    Pre-act bounds: l[i] = c_z[i] - |G_z[i,:]|_1 - tail_z[i]
                       u[i] = c_z[i] + |G_z[i,:]|_1 + tail_z[i]
    """
    from research.sc_hz.constrained_lp import LastReluRecord
    c_z = state_pre_relu.c.astype(np.float64)
    G_z = state_pre_relu.G.astype(np.float64) if state_pre_relu.G is not None else np.zeros((c_z.shape[0], 0))
    tail_z = (state_pre_relu.tail_radius.astype(np.float64)
                  if state_pre_relu.tail_radius is not None else None)
    G_l1 = np.abs(G_z).sum(axis=1)
    rad = G_l1 + (tail_z if tail_z is not None else 0.0)
    l = c_z - rad
    u = c_z + rad
    return LastReluRecord(c_z=c_z, G_z=G_z, tail_z=tail_z, l=l, u=u)


def f1_lp_refine_ub(net, tf, C: np.ndarray, thresholds: np.ndarray) -> Optional[np.ndarray]:
    """Refine UB per spec row using F1 LP. Returns refined UB array, or None
    if F1 LP unavailable for this net topology.

    Args:
        net: ACT Net with at least one RELU and at least one DENSE after it
        tf:  active FchzTF with _state_cache
        C:   (M, n_out) spec directions
        thresholds: (M,) thresholds

    Returns:
        (M,) refined UB array (each <= closed-form UB).
    """
    last_relu = find_last_relu_layer(net)
    if last_relu is None: return None

    # Pre-ReLU state = state of predecessor of last_relu
    pred_id = net.preds.get(last_relu.id, [None])[0]
    state_pre_relu = tf._state_cache.get(pred_id) if pred_id is not None else None
    if state_pre_relu is None: return None

    W_remaining, b_remaining = collect_post_relu_affine(net, last_relu.id, tf)
    if W_remaining is None: return None
    # Sanity: W_remaining (n_out, n_pre)
    if W_remaining.shape[1] != state_pre_relu.c.shape[0]: return None

    relu_rec = build_last_relu_record(state_pre_relu, W_remaining.shape[0])
    if relu_rec.n_pre == 0: return None

    from research.sc_hz.constrained_lp import constrained_lp_ub

    refined = np.full(C.shape[0], np.inf, dtype=np.float64)
    for i in range(C.shape[0]):
        d = C[i].astype(np.float64)
        try:
            ub, _ = constrained_lp_ub(relu_rec, W_remaining, b_remaining, d, return_solution=False)
            refined[i] = ub
        except Exception:
            refined[i] = np.inf    # LP failure → fall back to closed form
    return refined
