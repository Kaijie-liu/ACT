# ===- research/fchz/m4_verdict.py - M4 LP verdict for canonicalized spec ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Per advisor 2026-06-08 audit feedback:
#   M4 LP refinement must respect MULTI_QUERY semantics:
#     - SAFE iff EVERY query is safe (AND).
#     - First-query shortcut is UNSOUND.
#   Each query is evaluated independently with the appropriate kind:
#     - LINEAR_LE / TOP1_ROBUST / MARGIN_ROBUST: every row's UB < threshold.
#     - UNSAFE_LINEAR: any row's LB > threshold (any unreachable row breaks
#       the AND-polytope for that disjunct).
#   Unsupported kind in any query → mark instance UNSUPPORTED_QUERY.
#
# ===---------------------------------------------------------------------===#

"""M4 LP verdict with sound MULTI_QUERY (AND-over-queries) semantics."""

from __future__ import annotations
import numpy as np

from research.fchz.m4_full_lp import (
    is_dense_only_chain, solve_full_lp_ub, solve_full_lp_lb,
)


def _spec_from_outspec(out_spec):
    """Extract (kind, C, t) from a single OutputSpec."""
    kind = out_spec.kind
    if kind in ('UNSAFE_LINEAR', 'LINEAR_LE'):
        c = out_spec.c
        d = out_spec.d
        C = c.detach().cpu().numpy().astype(np.float64) if hasattr(c, 'detach') else np.asarray(c, dtype=np.float64)
        t = d.detach().cpu().numpy().astype(np.float64).reshape(-1) if hasattr(d, 'detach') else np.asarray(d, dtype=np.float64).reshape(-1)
        if C.ndim == 1: C = C.reshape(1, -1)
        return kind, C, t
    if kind in ('TOP1_ROBUST', 'MARGIN_ROBUST'):
        y_t = out_spec.y_true
        if hasattr(y_t, 'item'):
            y_t = int(y_t.item()) if y_t.numel() == 1 else int(y_t[0].item())
        else:
            y_t = int(y_t)
        return kind, y_t, None
    return kind, None, None


def _build_top1_C(y_true: int, n_out: int):
    """e_rival - e_true rows; safe iff max(C@y) < 0."""
    M = n_out - 1
    C = np.zeros((M, n_out), dtype=np.float64)
    k = 0
    for i in range(n_out):
        if i == y_true: continue
        C[k, i] = 1.0; C[k, y_true] = -1.0; k += 1
    return C, np.zeros(M, dtype=np.float64)


def _query_safe_via_lp(net, tf, kind, C_or_yt, t, n_out, use_cf_fallback=True, cf_state=None):
    """Evaluate a single query's safety using M4 LP. Returns 'CERTIFIED' / 'UNKNOWN' / 'UNSUPPORTED'.

    For LINEAR_LE / TOP1_ROBUST: safe iff EVERY row's LP UB < threshold.
    For UNSAFE_LINEAR: safe iff ANY row's LP LB > threshold.
    """
    if kind in ('LINEAR_LE',):
        C, t_arr = C_or_yt, t
        for i in range(C.shape[0]):
            lp_ub = solve_full_lp_ub(net, tf, C[i])
            cf_ub = None
            if use_cf_fallback and cf_state is not None:
                from act.back_end.fchz_tf.verifier_fchz import fchz_upper_bound
                cf_ub = float(fchz_upper_bound(cf_state, C[i].reshape(1, -1))[0])
            eff_ub = min(lp_ub, cf_ub) if cf_ub is not None else lp_ub
            if eff_ub >= t_arr[i]:
                return 'UNKNOWN'
        return 'CERTIFIED'
    if kind in ('TOP1_ROBUST', 'MARGIN_ROBUST'):
        C, t_arr = _build_top1_C(C_or_yt, n_out)
        for i in range(C.shape[0]):
            lp_ub = solve_full_lp_ub(net, tf, C[i])
            cf_ub = None
            if use_cf_fallback and cf_state is not None:
                from act.back_end.fchz_tf.verifier_fchz import fchz_upper_bound
                cf_ub = float(fchz_upper_bound(cf_state, C[i].reshape(1, -1))[0])
            eff_ub = min(lp_ub, cf_ub) if cf_ub is not None else lp_ub
            if eff_ub >= 0.0:
                return 'UNKNOWN'
        return 'CERTIFIED'
    if kind == 'UNSAFE_LINEAR':
        C, t_arr = C_or_yt, t
        for i in range(C.shape[0]):
            lp_lb = solve_full_lp_lb(net, tf, C[i])
            cf_lb = None
            if use_cf_fallback and cf_state is not None:
                from act.back_end.fchz_tf.verifier_fchz import fchz_lower_bound
                cf_lb = float(fchz_lower_bound(cf_state, C[i].reshape(1, -1))[0])
            eff_lb = max(lp_lb, cf_lb) if cf_lb is not None else lp_lb
            if eff_lb > t_arr[i]:
                return 'CERTIFIED'   # this row unreachable → polytope unreachable
        return 'UNKNOWN'
    return 'UNSUPPORTED'


def m4_verdict_for_queries(net, tf, queries, n_out, cf_state=None):
    """Compute M4 LP verdict over a list of queries (AND semantics).

    Returns dict with verdict + per-query breakdown.
    """
    if not is_dense_only_chain(net):
        return {'verdict': 'UNKNOWN', 'reason': 'NOT_DENSE_ONLY'}

    per_query = []
    for in_spec, out_spec in queries:
        kind, C_or_yt, t = _spec_from_outspec(out_spec)
        if kind not in ('LINEAR_LE', 'TOP1_ROBUST', 'MARGIN_ROBUST', 'UNSAFE_LINEAR'):
            return {'verdict': 'UNSUPPORTED_QUERY', 'reason': f'kind={kind}',
                        'per_query': per_query}
        try:
            v = _query_safe_via_lp(net, tf, kind, C_or_yt, t, n_out,
                                              use_cf_fallback=True, cf_state=cf_state)
        except Exception as e:
            return {'verdict': 'UNKNOWN', 'reason': f'LP_EXCEPTION:{type(e).__name__}',
                        'per_query': per_query}
        per_query.append({'kind': kind, 'verdict': v})
        if v != 'CERTIFIED':
            return {'verdict': 'UNKNOWN', 'reason': f'query_not_safe ({kind}: {v})',
                        'per_query': per_query}
    return {'verdict': 'CERTIFIED', 'reason': 'all_queries_safe',
                'per_query': per_query}
