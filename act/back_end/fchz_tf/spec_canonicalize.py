# ===- act/back_end/fchz_tf/spec_canonicalize.py - VNNLIB spec canonicalize ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Per advisor 2026-06-08: VNNLIB top-1 robustness specs commonly arrive as
#   K-1 separate single-row UNSAFE_LINEAR queries (one per rival). Running
#   FchzTF.analyze N times for the same input box is wasteful and TIMEOUTs
#   on larger nets (cifar resnet_LARGE / tiny resnet_medium).
#
#   This module canonicalizes a list of (InputSpec, OutputSpec) queries from
#   parse_vnnlib_queries into one of these classifications:
#
#     1. TOP1_ROBUST: single TOP1_ROBUST input → preserve parser y_true
#                          exactly (critical for n_out=2 where single-row sign
#                          inference is ambiguous between conventions).
#     2. TOP1_ROBUST: multi-query UNSAFE_LINEAR pattern → fold into single
#                          TOP1_ROBUST if all rows match "+1 at one shared index
#                          (true class), -1 at unique rival" pattern.
#     3. UNSAFE_LINEAR / LINEAR_LE: single query with multi-row matrix → keep
#                                                as conjunction (rows AND'd in the original spec).
#     4. MULTI_QUERY: multiple non-TOP1 queries → preserve queries list,
#                          evaluator checks each separately + AND result
#                          (sound: vnnlib disjunctive query list semantics
#                           = safe iff every query is safe).
#     5. MIXED: mixed kinds across queries → fail-closed UNKNOWN.
#
#   Stacking arbitrary non-TOP1 queries' rows into a single C matrix is
#   UNSOUND (would let "any row unreachable" rule certify a disjunctive set).
#   Therefore aggregation into a single matrix is REFUSED for non-TOP1
#   multi-query cases; MULTI_QUERY path preserves correctness.
#
#   Structure-based, NOT bench-name-based. Sound (no semantics changed).
#
# ===---------------------------------------------------------------------===#

"""Spec canonicalization — collapse K-1 row queries into a single ASSERT layer."""

from __future__ import annotations
import numpy as np
import torch
from typing import List, Tuple, Any


def _spec_to_numpy(out_spec):
    """Extract C, t, kind from an OutputSpec as numpy arrays."""
    kind = out_spec.kind
    if kind == 'UNSAFE_LINEAR':
        C = out_spec.c.detach().cpu().numpy() if hasattr(out_spec.c, 'detach') else np.asarray(out_spec.c)
        t = out_spec.d.detach().cpu().numpy() if hasattr(out_spec.d, 'detach') else np.asarray(out_spec.d)
        if C.ndim == 1: C = C.reshape(1, -1)
        t = np.asarray(t).flatten()
        return kind, C.astype(np.float64), t.astype(np.float64)
    if kind == 'LINEAR_LE':
        C = out_spec.c.detach().cpu().numpy() if hasattr(out_spec.c, 'detach') else np.asarray(out_spec.c)
        t = out_spec.d.detach().cpu().numpy() if hasattr(out_spec.d, 'detach') else np.asarray(out_spec.d)
        if C.ndim == 1: C = C.reshape(1, -1)
        t = np.asarray(t).flatten()
        return kind, C.astype(np.float64), t.astype(np.float64)
    if kind == 'TOP1_ROBUST':
        y_t = out_spec.y_true
        if hasattr(y_t, 'item'):
            y_t = int(y_t.item()) if y_t.numel() == 1 else int(y_t[0].item())
        else:
            y_t = int(y_t)
        return kind, y_t, None
    return kind, None, None


def _try_infer_top1(rows_C: np.ndarray, rows_t: np.ndarray, n_out: int):
    """Infer (y_true) if all rows look like '+1 at rival, -1 at single shared true class'.

    Returns y_true if pattern matches, None otherwise.
    """
    M = rows_C.shape[0]
    if M != n_out - 1: return None
    if not np.allclose(rows_t, 0.0, atol=1e-9): return None

    # ACT UNSAFE_LINEAR encoding: y_true - y_rival <= 0
    # → row has -1 at rival index, +1 at true class index.
    # Or research: y_rival - y_true >= 0 (after sign flip: -y_true + y_rival ... different framing).
    # We accept BOTH conventions.

    # Try ACT convention: +1 at one shared index (true class), -1 at unique rival
    pos_idx_set = set()
    neg_idx_per_row = []
    for i in range(M):
        pos = np.where(rows_C[i] > 0.5)[0]
        neg = np.where(rows_C[i] < -0.5)[0]
        if len(pos) != 1 or len(neg) != 1: return None
        pos_idx_set.add(int(pos[0]))
        neg_idx_per_row.append(int(neg[0]))
    if len(pos_idx_set) == 1:
        y_true_candidate = list(pos_idx_set)[0]
        rivals = set(neg_idx_per_row)
        expected_rivals = set(range(n_out)) - {y_true_candidate}
        if rivals == expected_rivals:
            return y_true_candidate

    # Try research convention: -1 at shared true class, +1 at unique rival
    neg_idx_set = set()
    pos_idx_per_row = []
    for i in range(M):
        pos = np.where(rows_C[i] > 0.5)[0]
        neg = np.where(rows_C[i] < -0.5)[0]
        if len(pos) != 1 or len(neg) != 1: return None
        neg_idx_set.add(int(neg[0]))
        pos_idx_per_row.append(int(pos[0]))
    if len(neg_idx_set) == 1:
        y_true_candidate = list(neg_idx_set)[0]
        rivals = set(pos_idx_per_row)
        expected_rivals = set(range(n_out)) - {y_true_candidate}
        if rivals == expected_rivals:
            return y_true_candidate

    return None


def _build_top1_C(y_true: int, n_out: int):
    """Build canonical TOP1_ROBUST C matrix: rows e_rival - e_true."""
    M = n_out - 1
    C = np.zeros((M, n_out), dtype=np.float64)
    k = 0
    for i in range(n_out):
        if i == y_true: continue
        C[k, i] = 1.0
        C[k, y_true] = -1.0
        k += 1
    return C, np.zeros(M, dtype=np.float64)


def canonicalize_queries(queries: List[Tuple[Any, Any]], n_out: int) -> dict:
    """Canonicalize a list of (InputSpec, OutputSpec) queries.

    Returns dict with:
      'kind': 'TOP1_ROBUST' or 'UNSAFE_LINEAR' or 'LINEAR_LE' or 'MIXED'
      'in_spec': the common input spec (assumed shared across queries)
      'C': stacked C matrix (M, n_out) for the relevant kind
      't': thresholds (M,)
      'y_true': if TOP1_ROBUST
      'n_input_queries': original query count
      'canonicalized': True/False
      'reason': brief explanation
    """
    if not queries:
        return {'kind': 'EMPTY', 'canonicalized': False, 'reason': 'no queries'}

    # All queries share input spec (parse_vnnlib_queries semantics)
    in_spec0 = queries[0][0]

    # FAST PATH (advisor M3 fix 2026-06-08):
    # If queries are already in TOP1_ROBUST form, preserve y_true exactly.
    # For n_out=2 (binary), inference from a single row [-1, 1] is ambiguous
    # between ACT (true=1) and research (true=0) conventions; safe path is
    # to trust the parser's explicit y_true.
    if len(queries) == 1 and queries[0][1].kind == 'TOP1_ROBUST':
        y_t = queries[0][1].y_true
        if hasattr(y_t, 'item'):
            y_t = int(y_t.item()) if y_t.numel() == 1 else int(y_t[0].item())
        else:
            y_t = int(y_t)
        C_top, t_top = _build_top1_C(y_t, n_out)
        return {
            'kind': 'TOP1_ROBUST',
            'in_spec': in_spec0,
            'C': C_top, 't': t_top,
            'y_true': y_t,
            'n_input_queries': 1,
            'canonicalized': True,
            'reason': 'preserved TOP1_ROBUST y_true from parser',
        }

    # SOUNDNESS gate (advisor 2026-06-08, audit 2026-06-08):
    # _try_infer_top1 below ONLY safely applies when the original spec is
    # DISJUNCTIVE multi-query form (each query is single-row, AND-of-queries
    # = "every rival beaten by true" = top-1 robust).
    #
    # A SINGLE query with M=n_out-1 rows is CONJUNCTIVE (rows AND'd within
    # the polytope). Its semantics is "polytope unreachable iff ANY row LB > t",
    # NOT "every row UB < t" (which is TOP1).
    # → Folding single-query multi-row into TOP1 is unsound in tightness:
    #    canon TOP1 (all UB < 0) is STRICTER than original UNSAFE conjunction.
    #    No false CERT but UNSOUNDLY loose UNK results.
    #
    # So: only attempt TOP1 inference if every query has exactly 1 row
    # (= disjunctive form across multiple queries).
    single_row_per_query = all(
        out_s.kind in ('UNSAFE_LINEAR', 'LINEAR_LE') and
        (out_s.c.shape[0] if hasattr(out_s.c, 'shape') and out_s.c.dim() == 2 else 1) == 1
        for _, out_s in queries
        if out_s.kind in ('UNSAFE_LINEAR', 'LINEAR_LE')
    )

    # Collect kinds + (C, t) — only used for multi-query / non-TOP1 cases
    all_kinds = set()
    accum_rows_C = []
    accum_rows_t = []
    for in_s, out_s in queries:
        kind, C, t = _spec_to_numpy(out_s)
        all_kinds.add(kind)
        if kind in ('UNSAFE_LINEAR', 'LINEAR_LE'):
            for i in range(C.shape[0]):
                accum_rows_C.append(C[i])
                accum_rows_t.append(t[i] if i < len(t) else t[0])
        elif kind == 'TOP1_ROBUST':
            # Multiple queries case — preserve y_true via row building.
            # We trust parser's y_true rather than re-inferring.
            C_top, t_top = _build_top1_C(C, n_out)  # C here is y_true
            for i in range(C_top.shape[0]):
                accum_rows_C.append(C_top[i])
                accum_rows_t.append(t_top[i])

    if not accum_rows_C:
        return {'kind': list(all_kinds)[0] if all_kinds else 'UNKNOWN',
                    'canonicalized': False,
                    'reason': f'no rows: kinds={all_kinds}'}

    C_all = np.stack(accum_rows_C, axis=0)
    t_all = np.asarray(accum_rows_t, dtype=np.float64)

    # Try TOP1_ROBUST inference ONLY for disjunctive form
    # (every query has 1 row → multiple single-row queries = top-1 robust)
    y_true_inferred = None
    if single_row_per_query and len(queries) >= 2:
        y_true_inferred = _try_infer_top1(C_all, t_all, n_out)
    if y_true_inferred is not None:
        # Build canonical TOP1_ROBUST C
        C_top, t_top = _build_top1_C(y_true_inferred, n_out)
        return {
            'kind': 'TOP1_ROBUST',
            'in_spec': in_spec0,
            'C': C_top, 't': t_top,
            'y_true': y_true_inferred,
            'n_input_queries': len(queries),
            'canonicalized': True,
            'reason': 'top1 pattern inferred',
        }

    # SOUNDNESS gate (advisor 2026-06-08):
    # Only TOP1 pattern auto-folds into a single matrix.
    # Multi-query non-TOP1: do NOT aggregate into single C matrix (unsound).
    # Instead: evaluate each query independently and combine with AND
    # (vnnlib disjunctive query list semantics: safe iff every query is safe).
    if all_kinds == {'UNSAFE_LINEAR'} and len(queries) == 1:
        return {
            'kind': 'UNSAFE_LINEAR',
            'in_spec': in_spec0,
            'C': C_all, 't': t_all,
            'n_input_queries': 1,
            'canonicalized': True,
            'reason': 'single UNSAFE_LINEAR query (already conjunction)',
        }
    if all_kinds == {'LINEAR_LE'} and len(queries) == 1:
        return {
            'kind': 'LINEAR_LE',
            'in_spec': in_spec0,
            'C': C_all, 't': t_all,
            'n_input_queries': 1,
            'canonicalized': True,
            'reason': 'single LINEAR_LE query (already conjunction)',
        }
    # MULTI_QUERY: keep queries as list, evaluator checks each + AND.
    # This is sound (vnnlib semantics: safe iff every query is safe).
    # Note: aggregation into single C matrix is still REFUSED (unsound).
    if all_kinds <= {'UNSAFE_LINEAR', 'LINEAR_LE'}:
        return {
            'kind': 'MULTI_QUERY',
            'in_spec': in_spec0,
            'queries': queries,         # original list — evaluator checks each
            'C': None, 't': None,        # no aggregation
            'n_input_queries': len(queries),
            'canonicalized': False,
            'reason': f'multi-query non-TOP1 → per-query AND: kinds={all_kinds}, n_q={len(queries)}',
        }
    # Truly mixed kinds (UNSAFE + TOP1, etc): fail-closed
    return {
        'kind': 'MIXED',
        'in_spec': in_spec0,
        'C': None, 't': None,
        'n_input_queries': len(queries),
        'canonicalized': False,
        'reason': f'mixed kinds (fail-closed): {all_kinds}',
    }


def canon_verdict(canon: dict, state):
    """Apply correct verifier semantics to canonicalized spec.

    Returns 'CERTIFIED' / 'UNKNOWN'.

    Spec kinds:
      TOP1_ROBUST / UNSAFE_LINEAR / LINEAR_LE: single ASSERT via _decide_kind
      MULTI_QUERY: evaluate each query separately via _decide_kind, AND result
                       (sound: safe iff every query is safe — vnnlib disjunctive)
      MIXED / EMPTY: fail-closed UNKNOWN
    """
    from act.back_end.fchz_tf.verifier_fchz import _decide_kind

    if canon.get('kind') in ('EMPTY', 'UNKNOWN', 'MIXED'):
        return 'UNKNOWN'

    kind = canon['kind']

    # MULTI_QUERY: per-query check + AND (sound, no aggregation)
    if kind == 'MULTI_QUERY':
        queries = canon.get('queries', [])
        if not queries: return 'UNKNOWN'
        n_out = state.c.shape[0]
        for in_s, out_s in queries:
            q_kind = out_s.kind
            try:
                if q_kind == 'UNSAFE_LINEAR':
                    C_q = out_s.c.detach().cpu().numpy() if hasattr(out_s.c, 'detach') else np.asarray(out_s.c)
                    t_q = out_s.d.detach().cpu().numpy() if hasattr(out_s.d, 'detach') else np.asarray(out_s.d)
                    C_q = C_q.astype(np.float64)
                    if C_q.ndim == 1: C_q = C_q.reshape(1, -1)
                    t_q = np.asarray(t_q).flatten().astype(np.float64)
                    if t_q.shape[0] == 1 and C_q.shape[0] > 1:
                        t_q = np.broadcast_to(t_q, (C_q.shape[0],))
                elif q_kind == 'LINEAR_LE':
                    C_q = out_s.c.detach().cpu().numpy() if hasattr(out_s.c, 'detach') else np.asarray(out_s.c)
                    t_q = out_s.d.detach().cpu().numpy() if hasattr(out_s.d, 'detach') else np.asarray(out_s.d)
                    C_q = C_q.astype(np.float64)
                    if C_q.ndim == 1: C_q = C_q.reshape(1, -1)
                    t_q = np.asarray(t_q).flatten().astype(np.float64)
                else:
                    return 'UNKNOWN'   # unsupported kind in MULTI_QUERY
                res = _decide_kind(q_kind, state, C_q, t_q, M_rows=C_q.shape[0])
                if res['verdict'] != 'CERTIFIED':
                    return 'UNKNOWN'    # any query unsafe/unknown → overall UNKNOWN
            except Exception:
                return 'UNKNOWN'
        return 'CERTIFIED'

    # Single-spec kinds
    C = canon.get('C')
    t = canon.get('t')
    if C is None: return 'UNKNOWN'
    res = _decide_kind(kind, state, C, t, M_rows=C.shape[0])
    return res['verdict']
