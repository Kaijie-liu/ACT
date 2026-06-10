# ===- act/back_end/fchz_tf/verifier_fchz.py - FCHZ-native verifier ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   FCHZ-native verifier path (advisor M1/M2 fix).
#
#   Stock verify_once compresses FCHZ state to (lb, ub) interval before
#   margin check. This loses generator correlations and tail_radius
#   structure. FCHZ-native check uses:
#       UB(C[i] @ y) = C[i] @ state.c + |C[i] @ state.G|.sum() + |C[i]| @ tail
#       LB(C[i] @ y) = C[i] @ state.c - |C[i] @ state.G|.sum() - |C[i]| @ tail
#   This is TIGHTER than interval check when state has correlated G.
#
# Spec semantics (advisor 2026-06-08 — critical):
#   LINEAR_LE        : safe iff C y <= d. CERT iff max(C@y) < d (per row).
#   TOP1_ROBUST      : rows e_rival - e_true. CERT iff max(C@y) < threshold (=0).
#   MARGIN_ROBUST    : same as TOP1_ROBUST.
#   UNSAFE_LINEAR    : unsafe polytope C y <= d, defined by AND of rows
#                            (all rows must hold simultaneously for unsafe).
#                            Single-row: CERT iff min(C@y) > d (equiv max(-C@y) < -d).
#                            Multi-row conjunction: SAFE iff at least one row is
#                            provably violated for all y in the state.
#                            → CERT iff ANY row has LB(C@y) > d (any unreachable
#                                row breaks the AND-polytope).
#   RANGE            : lb_y <= y <= ub_y. Not implemented; returns UNKNOWN.
#
# ===---------------------------------------------------------------------===#

"""FCHZ-native verifier — semantically correct per-spec-kind check."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np

from act.back_end.core import Bounds, Layer, Net, ConSet, Fact


def fchz_upper_bound(state, C: np.ndarray) -> np.ndarray:
    """max_y (C @ y) over R(state) per row of C.

    UB(C@y) = C @ c + |C @ G|.sum() + |C| @ tail_radius
    """
    M = C.shape[0]
    margin_center = C @ state.c
    if state.G.shape[1] > 0:
        CG = C @ state.G
        margin_g = np.abs(CG).sum(axis=1)
    else:
        margin_g = np.zeros(M)
    if state.tail_radius is not None:
        margin_tail = np.abs(C) @ state.tail_radius
    else:
        margin_tail = np.zeros(M)
    return margin_center + margin_g + margin_tail


def fchz_lower_bound(state, C: np.ndarray) -> np.ndarray:
    """min_y (C @ y) over R(state) per row of C.

    LB(C@y) = C @ c - |C @ G|.sum() - |C| @ tail_radius
    """
    M = C.shape[0]
    margin_center = C @ state.c
    if state.G.shape[1] > 0:
        CG = C @ state.G
        margin_g = np.abs(CG).sum(axis=1)
    else:
        margin_g = np.zeros(M)
    if state.tail_radius is not None:
        margin_tail = np.abs(C) @ state.tail_radius
    else:
        margin_tail = np.zeros(M)
    return margin_center - margin_g - margin_tail


# Legacy alias (M1-fix Task 4)
def fchz_closed_form_margin(state, C: np.ndarray) -> np.ndarray:
    """Legacy alias for fchz_upper_bound (kept for back-compat tests)."""
    return fchz_upper_bound(state, C)


def _decide_kind(kind: str, state, C: np.ndarray, thresholds: np.ndarray,
                       M_rows: int) -> Dict[str, Any]:
    """Decide CERT/UNKNOWN per spec kind with correct semantics.

    Returns dict with verdict + diagnostic excess.
    """
    if kind in ('LINEAR_LE', 'TOP1_ROBUST', 'MARGIN_ROBUST'):
        # Safe iff C @ y <= threshold (and CERT iff strict).
        # Use upper bound: CERT iff max(C@y) < threshold per row.
        ub = fchz_upper_bound(state, C)
        excess = ub - thresholds
        certified = bool((excess < 0).all())
        return {
            'verdict': 'CERTIFIED' if certified else 'UNKNOWN',
            'bound_type': 'UB',
            'bound': ub,
            'thresholds': thresholds,
            'excess': excess,
            'max_excess': float(excess.max()),
        }
    elif kind == 'UNSAFE_LINEAR':
        # Unsafe polytope C y <= threshold. Safe iff polytope unreachable.
        # Per row: CERT iff LB(C@y) > threshold (= min(C@y) > d).
        # For multi-row (conjunction unsafe = all rows simultaneously),
        # conservative: only CERT if EVERY row has LB > threshold (any row
        # being unreachable means polytope is unreachable).
        lb = fchz_lower_bound(state, C)
        excess = lb - thresholds   # >0 means unreachable on that row
        # CERT iff ANY row excess > 0 (any single unreachable row breaks AND-polytope)
        # CONSERVATIVE: if M_rows > 1 (multi-row AND from same query),
        # we should require ANY row > 0. This is sound but loose.
        certified = bool((excess > 0).any())
        return {
            'verdict': 'CERTIFIED' if certified else 'UNKNOWN',
            'bound_type': 'LB',
            'bound': lb,
            'thresholds': thresholds,
            'excess': excess,  # positive means certified
            'max_excess': float(-excess.max()),  # for consistency: < 0 = CERT
        }
    elif kind == 'RANGE':
        # Safe iff lb_y <= y <= ub_y for all components.
        # We don't fully support RANGE yet.
        return {'verdict': 'UNKNOWN', 'bound_type': 'NA',
                    'reason': 'RANGE kind not implemented'}
    else:
        return {'verdict': 'UNKNOWN', 'bound_type': 'NA',
                    'reason': f'unknown kind: {kind}'}


def verify_once_fchz(net: Net, tf, queries=None, n_out=None,
                                use_f1_lp: bool = False) -> dict:
    """FCHZ-native verifier with semantically correct per-kind check.

    Two modes:
      (1) queries=None (legacy): read C/thresholds from ASSERT params.
      (2) queries provided (PRODUCTION): canonicalize first, then check.
          Use queries=parse_vnnlib_queries(...) for full TOP1 auto-fold +
          multi-query AND + soundness gate.

    Args:
        net:     ACT Net with ASSERT layer
        tf:      active FchzTF instance with _state_cache populated
        queries: optional list of (InputSpec, OutputSpec)
        n_out:   model output dim (required if queries provided)

    Returns:
        {'verdict': 'CERTIFIED' | 'UNKNOWN' | 'NO_ASSERT' | 'NO_STATE' | 'NO_SPEC',
         'kind':    spec kind, ...}
    """
    # Find ASSERT layer (last)
    assert_layer = None
    for L in reversed(net.layers):
        if L.kind == 'ASSERT':
            assert_layer = L
            break
    if assert_layer is None:
        return {'verdict': 'NO_ASSERT'}

    # Pre-ASSERT state
    pre_id = net.preds.get(assert_layer.id, [None])[0]
    state = tf._state_cache.get(pre_id) if hasattr(tf, '_state_cache') else None
    if state is None:
        return {'verdict': 'NO_STATE'}

    # PRODUCTION MODE: queries provided → canonicalize + check
    if queries is not None:
        from act.back_end.fchz_tf.spec_canonicalize import (
            canonicalize_queries, canon_verdict)
        if n_out is None:
            n_out = state.c.shape[0]
        canon = canonicalize_queries(queries, n_out)
        verdict = canon_verdict(canon, state)
        # M3 P1: F1 LP refinement (advisor 2026-06-08)
        # Closed-form returned UNKNOWN; try LP refinement at the last ReLU.
        # Only applies to single-kind canon types with C/t arrays (not MULTI_QUERY).
        f1_lp_used = False
        if (use_f1_lp and verdict == 'UNKNOWN'
                and canon.get('kind') in ('TOP1_ROBUST', 'LINEAR_LE', 'UNSAFE_LINEAR')
                and canon.get('C') is not None):
            try:
                from act.back_end.fchz_tf.f1_lp_refine import f1_lp_refine_ub
                C = canon['C']
                t = canon['t']
                refined_ub = f1_lp_refine_ub(net, tf, C, t)
                if refined_ub is not None:
                    f1_lp_used = True
                    kind = canon['kind']
                    # Combine with closed-form (take min — both sound)
                    from act.back_end.fchz_tf.verifier_fchz import (
                        fchz_upper_bound, fchz_lower_bound)
                    if kind in ('TOP1_ROBUST', 'LINEAR_LE'):
                        cf_ub = fchz_upper_bound(state, C)
                        eff_ub = np.minimum(cf_ub, refined_ub)
                        excess = eff_ub - t
                        if (excess < 0).all():
                            verdict = 'CERTIFIED'
                    elif kind == 'UNSAFE_LINEAR':
                        # For UNSAFE_LINEAR, LB > t means unreachable.
                        # F1 LP computes UB; for "any row LB > t" we use -UB(-d).
                        cf_lb = fchz_lower_bound(state, C)
                        # F1 LP on -C gives UB(-d); LB = -UB(-d)
                        try:
                            refined_neg = f1_lp_refine_ub(net, tf, -C, -t)
                            if refined_neg is not None:
                                lp_lb = -refined_neg
                                eff_lb = np.maximum(cf_lb, lp_lb)
                                excess = eff_lb - t
                                if (excess > 0).any():
                                    verdict = 'CERTIFIED'
                        except Exception:
                            pass
            except Exception:
                f1_lp_used = False

        return {
            'verdict': verdict,
            'kind': canon.get('kind'),
            'y_true': canon.get('y_true'),
            'n_input_queries': canon.get('n_input_queries'),
            'canonicalized': canon.get('canonicalized'),
            'canon_reason': canon.get('reason'),
            'f1_lp_used': f1_lp_used,
        }

    # LEGACY MODE: read from ASSERT.params (kept for back-compat)
    C_raw = assert_layer.params.get('C')
    th_raw = assert_layer.params.get('thresholds')
    kind = assert_layer.params.get('kind', 'LINEAR_LE')
    M = assert_layer.params.get('M', 1)

    if C_raw is None or th_raw is None:
        return {'verdict': 'NO_SPEC', 'kind': kind}

    C = (C_raw.detach().cpu().numpy() if hasattr(C_raw, 'detach') else np.asarray(C_raw))
    thresholds = (th_raw.detach().cpu().numpy() if hasattr(th_raw, 'detach') else np.asarray(th_raw))
    C = C.astype(np.float64)
    thresholds = thresholds.astype(np.float64).reshape(-1)

    if C.ndim == 1:
        C = C.reshape(1, -1)
    if thresholds.ndim == 0:
        thresholds = thresholds.reshape(1)
    if thresholds.shape[0] != C.shape[0]:
        if thresholds.shape[0] == 1:
            thresholds = np.broadcast_to(thresholds, (C.shape[0],))

    result = _decide_kind(kind, state, C, thresholds, M_rows=C.shape[0])
    result['kind'] = kind
    result['M'] = M
    return result
