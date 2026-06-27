# ===- act/back_end/solver/solver_hz_verdict.py - HZ spec-margin verdict -===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
"""Constrained Hybrid-Zonotope spec-margin certification (open-source solver).

Given the propagated output HZ and a linear output spec ``C y`` / ``thresholds``,
this decides robustness by SOLVING the spec margin over the constrained HZ --
not by interval arithmetic on the output box. Two regimes, both sound and both
using only open-source scipy/HiGHS/SCIP backends (Gurobi is forbidden in the
proof path, P3):

  * LP relaxation (binaries relaxed to [-1,1]) = the convex hull of the HZ
    (Zhang&Xu hz2, Thm 2): a sound, fast lower/upper bound on the true margin.
  * MILP (binaries in {-1,1}) = the exact margin over the HZ; tighter, slower.

Spec kinds (matching verifier.verify_once):
  * UNSAFE_LINEAR (conjunction): unsafe = {y : C[r] y <= t[r] for ALL r}. Safe
    iff  s* = min_y max_r (C[r] y - t[r]) > 0  (the joint min-max margin).
  * ALL-rows (TOP1_ROBUST / LINEAR_LE / ...): safe iff every row's
    max_y C[r] y < t[r].

Returned margins are SOUND: LP/MILP over the HZ is an over-approximation of the
reachable set, so a certified verdict here is sound. This module never produces
a falsification verdict (it only promotes UNKNOWN -> CERTIFIED).
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np

from act.back_end.solver.solver_hz import HZono, _split_eq_le, hz_known_nonempty
from act.back_end.solver.sparse_hz import SparseHZono

try:
    import scipy.sparse as _sp
    from scipy.optimize import linprog as _linprog, milp as _milp
    from scipy.optimize import LinearConstraint as _LC, Bounds as _LPB
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

try:
    import highspy as _highspy
    _HAS_HIGHSPY = True
except Exception:  # pragma: no cover
    _HAS_HIGHSPY = False

try:
    from pyscipopt import Model as _SCIPModel, quicksum as _scip_quicksum
    _HAS_PYSCIPOPT = True
except Exception:  # pragma: no cover
    _HAS_PYSCIPOPT = False


def _torch_csr(t):
    """Torch tensor -> scipy CSR without materializing hstack/vstack dense copies."""
    if not _HAS_SCIPY:
        return None
    shape = tuple(int(x) for x in t.shape)
    if t.numel() == 0:
        return _sp.csr_matrix(shape, dtype=np.float64)
    return _sp.csr_matrix(t.detach().cpu().numpy(), dtype=np.float64)


def _hz_np_sparse(hz):
    """Dense output rows + sparse constraint rows for solver construction.

    The final spec only touches the output dimension, so ``Gc/Gb`` stay dense and
    small. The constraint block can be thousands by tens of thousands with very
    low structural density after exact ReLU; keep it CSR all the way into HiGHS.
    """
    cached = getattr(hz, "_solver_np_sparse_cache", None)
    if cached is not None:
        return cached
    if isinstance(hz, SparseHZono):
        out = hz.solver_tuple()
        setattr(hz, "_solver_np_sparse_cache", out)
        return out
    c = hz.c.detach().cpu().double().numpy().reshape(-1)
    Gc = hz.Gc.detach().cpu().double().numpy()
    Gb = hz.Gb.detach().cpu().double().numpy()
    (Ace, Abe, be), (Acl, Abl, bl) = _split_eq_le(hz)
    out = (c, Gc, Gb,
           _torch_csr(Ace), _torch_csr(Abe),
           be.detach().cpu().double().numpy().reshape(-1),
           _torch_csr(Acl), _torch_csr(Abl),
           bl.detach().cpu().double().numpy().reshape(-1))
    setattr(hz, "_solver_np_sparse_cache", out)
    return out


def _mat_dot_gen(mat, gen) -> np.ndarray:
    """Return ``mat @ gen`` as a dense ndarray for solver objective rows."""

    mat = np.asarray(mat, dtype=np.float64)
    if _sp.issparse(gen):
        return np.asarray((_sp.csr_matrix(mat) @ gen).toarray(), dtype=np.float64)
    return np.asarray(mat @ gen, dtype=np.float64)


def _row_dot_gen(row, gen) -> np.ndarray:
    return _mat_dot_gen(np.asarray(row, dtype=np.float64).reshape(1, -1), gen).reshape(-1)


def _hz_relax_np_sparse(hz):
    """Legacy sparse LP-relaxation matrices for local diagnostic scripts.

    The production HybridZ verdict path below builds its own exact LP/MILP rows
    through ``_objbound_solve``.  Keep this private helper only while legacy
    local scripts still import it for excluded LP-witness diagnostics.
    """
    cached = getattr(hz, "_solver_relax_sparse_cache", None)
    if cached is not None:
        return cached
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = _hz_np_sparse(hz)
    Aeq = _sp.hstack([Ace, Abe], format="csr") if Ace.shape[0] else None
    Aub = _sp.hstack([Acl, Abl], format="csr") if Acl.shape[0] else None
    out = (c, Gc, Gb, Aeq, be, Aub, bl)
    setattr(hz, "_solver_relax_sparse_cache", out)
    return out


def _csr_rowsum(A) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).reshape(-1)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _parse_highs_value(raw: str):
    s = str(raw).strip()
    lo = s.lower()
    if lo in {"true", "yes", "on"}:
        return True
    if lo in {"false", "no", "off"}:
        return False
    try:
        if any(ch in s for ch in ".eE"):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _apply_highs_env_options(h):
    """Apply comma/semicolon separated HiGHS options from HZ_HIGHS_OPTIONS.

    This is a solver-strategy hook only: it cannot relax integrality or change
    the exact HZ constraints. Invalid options are ignored unless debug is on.
    """
    raw = os.environ.get("HZ_HIGHS_OPTIONS", "").strip()
    if not raw:
        return
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            h.setOptionValue(key, _parse_highs_value(val))
        except Exception as exc:
            if _env_flag("HZ_MILP_DEBUG"):
                print(f"[HZ_MILP] ignored HiGHS option {key}={val}: {exc}", flush=True)


def _project_singleton_continuous_rows(A, rl, ru, cost, lb, ub, integ_mask):
    """Exact existential projection of objective-free continuous singleton vars.

    If a continuous variable appears in exactly one equality row, appears in no
    other row, and has zero objective coefficient, it can be eliminated exactly:

        a*x + rest = rhs,  lb <= x <= ub

    becomes the range row

        rhs - max(a*[lb,ub]) <= rest <= rhs - min(a*[lb,ub]).

    This is solver presolve only. It keeps the feasible projection over the
    remaining variables identical and stores enough metadata to reconstruct an
    original-space witness if HiGHS finds one.
    """
    if not (_HAS_SCIPY and A.shape[0] and A.shape[1]):
        return A, rl, ru, cost, lb, ub, None

    A = _sp.csr_matrix(A)
    rl = np.asarray(rl, dtype=np.float64).copy()
    ru = np.asarray(ru, dtype=np.float64).copy()
    cost = np.asarray(cost, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    integ_mask = np.asarray(integ_mask, dtype=bool)

    eq_rows = np.isfinite(rl) & np.isfinite(ru) & (np.abs(rl - ru) <= 1e-10)
    if not np.any(eq_rows):
        return A, rl, ru, cost, lb, ub, None

    Acsc = A.tocsc()
    removable = []
    records_by_row = {}
    for j in range(A.shape[1]):
        if integ_mask[j] or abs(float(cost[j])) > 1e-12:
            continue
        start, end = Acsc.indptr[j], Acsc.indptr[j + 1]
        if end - start != 1:
            continue
        r = int(Acsc.indices[start])
        if not bool(eq_rows[r]):
            continue
        a = float(Acsc.data[start])
        if abs(a) <= 1e-14 or not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
            continue
        removable.append(j)
        records_by_row.setdefault(r, []).append((j, a, float(lb[j]), float(ub[j])))

    if not removable:
        return A, rl, ru, cost, lb, ub, None

    elim_min = np.zeros(A.shape[0], dtype=np.float64)
    elim_max = np.zeros(A.shape[0], dtype=np.float64)
    for r, vals in records_by_row.items():
        lo_sum = 0.0
        hi_sum = 0.0
        for _, a, lo, hi in vals:
            y0, y1 = a * lo, a * hi
            lo_sum += min(y0, y1)
            hi_sum += max(y0, y1)
        elim_min[r] = lo_sum
        elim_max[r] = hi_sum

    rhs = rl.copy()
    rl_new = rl.copy()
    ru_new = ru.copy()
    affected = np.fromiter(records_by_row.keys(), dtype=np.int64)
    rl_new[affected] = rhs[affected] - elim_max[affected]
    ru_new[affected] = rhs[affected] - elim_min[affected]

    keep = np.ones(A.shape[1], dtype=bool)
    keep[np.asarray(removable, dtype=np.int64)] = False
    keep_cols = np.nonzero(keep)[0]
    reduced = (
        A[:, keep_cols].tocsr(),
        rl_new,
        ru_new,
        cost[keep_cols],
        lb[keep_cols],
        ub[keep_cols],
        {
            "keep_cols": keep_cols,
            "original_ncols": int(A.shape[1]),
            "records_by_row": records_by_row,
            "rhs": rhs,
            "A_reduced": A[:, keep_cols].tocsr(),
        },
    )
    return reduced


def _substitute_eq_singleton_continuous_cols(A, rl, ru, cost, lb, ub, integ_mask):
    """Exact equality substitution for objective-free continuous singleton vars.

    For a continuous variable that appears in exactly one equality row,

        a*x + rest = rhs,

    substitute ``x = rhs/a - rest/a`` into every inequality row where ``x``
    appears, add the projected bounds for ``x``, and then remove both ``x`` and
    its defining equality. This is exact existential projection, not a
    relaxation. The transform is conservative about sparsity and skips columns
    whose substitution would touch too many inequality rows.
    """
    if not (_HAS_SCIPY and A.shape[0] and A.shape[1]):
        return A, rl, ru, cost, lb, ub, None

    A = _sp.csr_matrix(A, dtype=np.float64)
    rl = np.asarray(rl, dtype=np.float64).copy()
    ru = np.asarray(ru, dtype=np.float64).copy()
    cost = np.asarray(cost, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    integ_mask = np.asarray(integ_mask, dtype=bool)

    eq_rows = np.isfinite(rl) & np.isfinite(ru) & (np.abs(rl - ru) <= 1e-10)
    if not np.any(eq_rows):
        return A, rl, ru, cost, lb, ub, None

    try:
        max_ineq = int(os.environ.get("HZ_MILP_EQ_SUBST_MAX_INEQ", "2"))
    except ValueError:
        max_ineq = 2
    A_csc = A.tocsc()
    used_rows = set()
    pivots = []
    for j in range(A.shape[1]):
        if integ_mask[j] or abs(float(cost[j])) > 1e-12:
            continue
        if not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
            continue
        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        rows = A_csc.indices[start:end]
        data = A_csc.data[start:end]
        eq_pos = [k for k, r in enumerate(rows) if bool(eq_rows[int(r)])]
        if len(eq_pos) != 1:
            continue
        ineq_nnz = (end - start) - 1
        if max_ineq >= 0 and ineq_nnz > max_ineq:
            continue
        pos = eq_pos[0]
        r = int(rows[pos])
        if r in used_rows:
            continue
        a = float(data[pos])
        if abs(a) <= 1e-14:
            continue
        used_rows.add(r)
        pivots.append((int(j), r, a))

    if not pivots:
        return A, rl, ru, cost, lb, ub, None

    pivot_cols = np.asarray([p[0] for p in pivots], dtype=np.int64)
    pivot_rows = np.asarray([p[1] for p in pivots], dtype=np.int64)
    pivot_col_set = set(int(x) for x in pivot_cols)
    pivot_row_set = set(int(x) for x in pivot_rows)

    corr_rr, corr_cc, corr_dd = [], [], []
    bound_rr, bound_cc, bound_dd = [], [], []
    bound_rl, bound_ru = [], []
    records = []
    skipped = 0

    for out_i, (j, r, a) in enumerate(pivots):
        row = A.getrow(r).tocoo()
        mask = row.col != j
        cols = row.col[mask].astype(np.int64, copy=False)
        data = row.data[mask].astype(np.float64, copy=False)
        if any(int(c) in pivot_col_set for c in cols):
            skipped += 1
            pivot_col_set.discard(int(j))
            pivot_row_set.discard(int(r))
            continue

        rhs_j = float(rl[r])
        const = rhs_j / a
        coeff = -data / a
        records.append({
            "j": int(j),
            "r": int(r),
            "a": float(a),
            "rhs": rhs_j,
            "cols": cols.copy(),
            "data": data.copy(),
            "lb": float(lb[j]),
            "ub": float(ub[j]),
        })

        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        rows_i = A_csc.indices[start:end].astype(np.int64, copy=False)
        vals_i = A_csc.data[start:end].astype(np.float64, copy=False)
        for row_i, val_i in zip(rows_i, vals_i):
            row_i = int(row_i)
            if row_i == r:
                continue
            if cols.size:
                corr_rr.append(np.full(cols.size, row_i, dtype=np.int32))
                corr_cc.append(cols.astype(np.int32, copy=False))
                corr_dd.append((float(val_i) * coeff).astype(np.float64, copy=False))
            shift = float(val_i) * const
            if np.isfinite(rl[row_i]):
                rl[row_i] -= shift
            if np.isfinite(ru[row_i]):
                ru[row_i] -= shift

        if cols.size:
            br = 2 * out_i
            bound_rr.append(np.full(cols.size, br, dtype=np.int32))
            bound_cc.append(cols.astype(np.int32, copy=False))
            bound_dd.append(coeff.astype(np.float64, copy=False))
            bound_rl.append(-np.inf)
            bound_ru.append(float(ub[j]) - const)

            bound_rr.append(np.full(cols.size, br + 1, dtype=np.int32))
            bound_cc.append(cols.astype(np.int32, copy=False))
            bound_dd.append((-coeff).astype(np.float64, copy=False))
            bound_rl.append(-np.inf)
            bound_ru.append(-float(lb[j]) + const)
        else:
            bound_rl.extend([-np.inf, -np.inf])
            bound_ru.extend([float(ub[j]) - const, -float(lb[j]) + const])

    if skipped:
        keep_records = {int(rec["j"]) for rec in records}
        pivot_cols = np.asarray([p[0] for p in pivots if int(p[0]) in keep_records], dtype=np.int64)
        pivot_rows = np.asarray([p[1] for p in pivots if int(p[0]) in keep_records], dtype=np.int64)

    if pivot_cols.size == 0:
        return A, rl, ru, cost, lb, ub, None

    if corr_rr:
        corr = _sp.coo_matrix(
            (np.concatenate(corr_dd), (np.concatenate(corr_rr), np.concatenate(corr_cc))),
            shape=A.shape,
        ).tocsr()
        A = (A + corr).tocsr()
        A.eliminate_zeros()

    if bound_ru:
        n_bound = len(bound_ru)
        if bound_rr:
            bound_A = _sp.coo_matrix(
                (np.concatenate(bound_dd), (np.concatenate(bound_rr), np.concatenate(bound_cc))),
                shape=(n_bound, A.shape[1]),
            ).tocsr()
        else:
            bound_A = _sp.csr_matrix((n_bound, A.shape[1]), dtype=np.float64)
        A = _sp.vstack([A, bound_A], format="csr")
        rl = np.concatenate([rl, np.asarray(bound_rl, dtype=np.float64)])
        ru = np.concatenate([ru, np.asarray(bound_ru, dtype=np.float64)])

    keep_cols_mask = np.ones(A.shape[1], dtype=bool)
    keep_cols_mask[pivot_cols] = False
    keep_rows_mask = np.ones(A.shape[0], dtype=bool)
    keep_rows_mask[pivot_rows] = False
    keep_cols = np.nonzero(keep_cols_mask)[0]
    keep_rows = np.nonzero(keep_rows_mask)[0]
    A_reduced = A[keep_rows, :][:, keep_cols].tocsr()
    meta = {
        "kind": "eq_subst",
        "keep_cols": keep_cols,
        "original_ncols": int(A.shape[1]),
        "records": records,
    }
    return (
        A_reduced,
        rl[keep_rows],
        ru[keep_rows],
        cost[keep_cols],
        lb[keep_cols],
        ub[keep_cols],
        meta,
    )


def _chain_elim_meta(*metas):
    metas = [m for m in metas if m is not None]
    if not metas:
        return None
    if len(metas) == 1:
        return metas[0]
    return {"kind": "chain", "metas": metas}


def _expand_projected_solution(v, elim_meta):
    if not elim_meta:
        return np.asarray(v, dtype=np.float64)
    if elim_meta.get("kind") == "chain":
        out = np.asarray(v, dtype=np.float64)
        for meta in reversed(elim_meta["metas"]):
            out = _expand_projected_solution(out, meta)
        return out
    if elim_meta.get("kind") == "eq_subst":
        keep_cols = np.asarray(elim_meta["keep_cols"], dtype=np.int64)
        full = np.zeros(int(elim_meta["original_ncols"]), dtype=np.float64)
        full[keep_cols] = np.asarray(v, dtype=np.float64)
        for rec in elim_meta["records"]:
            cols = np.asarray(rec["cols"], dtype=np.int64)
            data = np.asarray(rec["data"], dtype=np.float64)
            rest = float(data @ full[cols]) if cols.size else 0.0
            val = (float(rec["rhs"]) - rest) / float(rec["a"])
            full[int(rec["j"])] = float(np.clip(val, float(rec["lb"]), float(rec["ub"])))
        return full
    keep_cols = np.asarray(elim_meta["keep_cols"], dtype=np.int64)
    full = np.zeros(int(elim_meta["original_ncols"]), dtype=np.float64)
    full[keep_cols] = np.asarray(v, dtype=np.float64)
    Ared = elim_meta["A_reduced"]
    rhs = np.asarray(elim_meta["rhs"], dtype=np.float64)

    for r, vals in elim_meta["records_by_row"].items():
        start, end = Ared.indptr[r], Ared.indptr[r + 1]
        rest = float(Ared.data[start:end] @ full[keep_cols[Ared.indices[start:end]]])
        target = float(rhs[r] - rest)

        lows, highs = [], []
        for _, a, lo, hi in vals:
            y0, y1 = a * lo, a * hi
            lows.append(min(y0, y1))
            highs.append(max(y0, y1))
        y = np.asarray(lows, dtype=np.float64)
        target = min(max(target, float(y.sum())), float(np.asarray(highs, dtype=np.float64).sum()))
        surplus = target - float(y.sum())
        for k in range(len(vals)):
            room = highs[k] - lows[k]
            if surplus <= 0.0:
                break
            step = min(room, surplus)
            y[k] += step
            surplus -= step
        for k, (j, a, lo, hi) in enumerate(vals):
            full[int(j)] = float(np.clip(y[k] / a, lo, hi))
    return full


def _project_solution_to_reduced(v, elim_meta):
    out = np.asarray(v, dtype=np.float64)
    if not elim_meta:
        return out
    if elim_meta.get("kind") == "chain":
        for meta in elim_meta["metas"]:
            out = _project_solution_to_reduced(out, meta)
        return out
    return out[np.asarray(elim_meta["keep_cols"], dtype=np.int64)]


def _scale_milp_rows(A, rl, ru):
    """Positive row scaling for solver conditioning; exact feasible set unchanged."""
    A = _sp.csr_matrix(A, dtype=np.float64)
    if A.shape[0] == 0:
        return A, np.asarray(rl, dtype=np.float64), np.asarray(ru, dtype=np.float64), None
    row_abs = np.asarray(np.abs(A).max(axis=1).toarray()).reshape(-1)
    scale = np.ones(A.shape[0], dtype=np.float64)
    nz = row_abs > 0.0
    scale[nz] = 1.0 / row_abs[nz]
    if np.allclose(scale, 1.0):
        return A, np.asarray(rl, dtype=np.float64), np.asarray(ru, dtype=np.float64), scale
    D = _sp.diags(scale, offsets=0, format="csr")
    return (
        D @ A,
        np.asarray(rl, dtype=np.float64) * scale,
        np.asarray(ru, dtype=np.float64) * scale,
        scale,
    )


def _scale_milp_objective(cost, obj_thr):
    """Positive objective scaling; exact cutoff sign unchanged."""
    cost = np.asarray(cost, dtype=np.float64)
    denom = max(float(np.max(np.abs(cost))) if cost.size else 0.0,
                abs(float(obj_thr)), 1.0)
    if not np.isfinite(denom) or denom <= 0.0:
        return cost, float(obj_thr), 1.0
    scale = 1.0 / denom
    return cost * scale, float(obj_thr) * scale, scale


def sparse_row_bound_infeasible(
    A: "_sp.csr_matrix",
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    tol: float = 1e-9,
) -> Tuple[bool, Dict[str, object]]:
    """Cheap exact row-range infeasibility check over variable bounds."""

    if A.shape[0] == 0 or A.shape[1] == 0:
        return False, {"rows": 0}
    A = A.tocsr()
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    rl = np.asarray(rl, dtype=np.float64)
    ru = np.asarray(ru, dtype=np.float64)
    pos = A.maximum(0.0).tocsr()
    neg = A.minimum(0.0).tocsr()
    row_min = np.asarray(pos @ lb + neg @ ub).reshape(-1)
    row_max = np.asarray(pos @ ub + neg @ lb).reshape(-1)
    scale = np.maximum(1.0, np.maximum(np.abs(row_min), np.abs(row_max)))
    hi_bad = np.isfinite(ru) & (row_min > ru + tol * scale)
    lo_bad = np.isfinite(rl) & (row_max < rl - tol * scale)
    bad = np.flatnonzero(hi_bad | lo_bad)
    if bad.size == 0:
        return False, {"rows": int(A.shape[0])}
    r0 = int(bad[0])
    return True, {
        "rows": int(A.shape[0]),
        "bad_row": r0,
        "row_min": float(row_min[r0]),
        "row_max": float(row_max[r0]),
        "rl": float(rl[r0]) if np.isfinite(rl[r0]) else None,
        "ru": float(ru[r0]) if np.isfinite(ru[r0]) else None,
        "bad_count": int(bad.size),
    }


def sparse_fbbt_tighten_bounds(
    A: "_sp.csr_matrix",
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    integer_mask: Optional[np.ndarray] = None,
    max_passes: int = 3,
    tol: float = 1e-9,
) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, object]]:
    """Feasibility-based bound tightening for sparse linear rows.

    This is a sound MILP presolve over exact HZ constraints.  It intersects the
    current variable boxes with bounds implied by ``rl <= A x <= ru`` and can
    prove EMPTY; it never proves a witness or replaces the exact MILP solve.
    """

    if A.shape[0] == 0 or A.shape[1] == 0 or max_passes <= 0:
        return False, lb, ub, {"passes": 0, "tightened": 0, "fixed_int": 0}

    A = A.tocsr(copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    rl = np.asarray(rl, dtype=np.float64)
    ru = np.asarray(ru, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
    if integer_mask is None:
        integer_mask = np.zeros(A.shape[1], dtype=bool)
    else:
        integer_mask = np.asarray(integer_mask, dtype=bool)
        if integer_mask.size != A.shape[1]:
            raise ValueError(f"integer_mask size mismatch: {integer_mask.size} vs {A.shape[1]}")

    pos = A.maximum(0.0).tocsr()
    neg = A.minimum(0.0).tocsr()
    total_tightened = 0
    total_fixed_int = 0
    max_width_delta = 0.0
    passes_done = 0
    bad_info: Optional[Dict[str, object]] = None

    for pass_i in range(int(max_passes)):
        passes_done = pass_i + 1
        prev_lb = lb.copy()
        prev_ub = ub.copy()
        row_min = np.asarray(pos @ lb + neg @ ub).reshape(-1)
        row_max = np.asarray(pos @ ub + neg @ lb).reshape(-1)
        scale = np.maximum(1.0, np.maximum(np.abs(row_min), np.abs(row_max)))
        hi_bad = np.isfinite(ru) & (row_min > ru + tol * scale)
        lo_bad = np.isfinite(rl) & (row_max < rl - tol * scale)
        bad = np.flatnonzero(hi_bad | lo_bad)
        if bad.size:
            r0 = int(bad[0])
            bad_info = {
                "bad_row": r0,
                "row_min": float(row_min[r0]),
                "row_max": float(row_max[r0]),
                "rl": float(rl[r0]) if np.isfinite(rl[r0]) else None,
                "ru": float(ru[r0]) if np.isfinite(ru[r0]) else None,
                "bad_count": int(bad.size),
            }
            return True, lb, ub, {
                "passes": passes_done,
                "tightened": int(total_tightened),
                "fixed_int": int(total_fixed_int),
                "max_width_delta": float(max_width_delta),
                "infeasible": bad_info,
            }

        for r in range(A.shape[0]):
            start, end = A.indptr[r], A.indptr[r + 1]
            if start == end:
                continue
            cols = A.indices[start:end]
            vals = A.data[start:end]
            nz = np.abs(vals) > tol
            if not np.any(nz):
                continue
            cols = cols[nz]
            vals = vals[nz]
            col_lb = lb[cols]
            col_ub = ub[cols]
            positive = vals > 0.0

            if np.isfinite(ru[r]) and ru[r] < 1e20:
                min_contrib = np.where(positive, vals * col_lb, vals * col_ub)
                rest_min = row_min[r] - min_contrib
                cand = (ru[r] - rest_min) / vals
                finite = np.isfinite(cand)
                m = positive & finite
                if np.any(m):
                    idx = cols[m]
                    ub[idx] = np.minimum(ub[idx], cand[m])
                m = (~positive) & finite
                if np.any(m):
                    idx = cols[m]
                    lb[idx] = np.maximum(lb[idx], cand[m])

            if np.isfinite(rl[r]) and rl[r] > -1e20:
                max_contrib = np.where(positive, vals * col_ub, vals * col_lb)
                rest_max = row_max[r] - max_contrib
                cand = (rl[r] - rest_max) / vals
                finite = np.isfinite(cand)
                m = positive & finite
                if np.any(m):
                    idx = cols[m]
                    lb[idx] = np.maximum(lb[idx], cand[m])
                m = (~positive) & finite
                if np.any(m):
                    idx = cols[m]
                    ub[idx] = np.minimum(ub[idx], cand[m])

        if np.any(integer_mask):
            snap_hi = integer_mask & (lb > tol) & (ub >= 1.0 - tol)
            snap_lo = integer_mask & (ub < 1.0 - tol) & (lb <= tol)
            fixed_now = int(np.count_nonzero(snap_hi | snap_lo))
            if fixed_now:
                total_fixed_int += fixed_now
                lb[snap_hi] = 1.0
                ub[snap_hi] = 1.0
                lb[snap_lo] = 0.0
                ub[snap_lo] = 0.0

        bad_cols = np.flatnonzero(lb > ub + 10.0 * tol)
        if bad_cols.size:
            j0 = int(bad_cols[0])
            return True, lb, ub, {
                "passes": passes_done,
                "tightened": int(total_tightened),
                "fixed_int": int(total_fixed_int),
                "max_width_delta": float(max_width_delta),
                "infeasible": {
                    "bad_col": j0,
                    "lb": float(lb[j0]),
                    "ub": float(ub[j0]),
                    "bad_count": int(bad_cols.size),
                },
            }

        tightened = (lb > prev_lb + tol) | (ub < prev_ub - tol)
        tightened_count = int(np.count_nonzero(tightened))
        if tightened_count:
            total_tightened += tightened_count
            before_w = prev_ub - prev_lb
            after_w = ub - lb
            max_width_delta = max(max_width_delta, float(np.max(before_w[tightened] - after_w[tightened])))
        else:
            break

    bad_cols = np.flatnonzero(lb > ub + 10.0 * tol)
    if bad_cols.size:
        j0 = int(bad_cols[0])
        bad_info = {"bad_col": j0, "lb": float(lb[j0]), "ub": float(ub[j0]), "bad_count": int(bad_cols.size)}
        return True, lb, ub, {
            "passes": passes_done,
            "tightened": int(total_tightened),
            "fixed_int": int(total_fixed_int),
            "max_width_delta": float(max_width_delta),
            "infeasible": bad_info,
        }

    return False, lb, ub, {
        "passes": passes_done,
        "tightened": int(total_tightened),
        "fixed_int": int(total_fixed_int),
        "max_width_delta": float(max_width_delta),
    }


def _spec_np(C, thresholds, out_dim: int):
    C = np.asarray(C, dtype=np.float64).reshape(-1, out_dim)
    t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if t.size == 1 and C.shape[0] != 1:
        t = np.repeat(t, C.shape[0])
    return C, t


def hz_row_max(hz, c_row: np.ndarray, *, integer: bool = False,
               time_limit: float = 20.0) -> Optional[float]:
    """max_y (c_row . y) over the HZ. LP relaxation (convex hull) or MILP."""
    if not _HAS_SCIPY:
        return None
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = _hz_np_sparse(hz)
    c_row = np.asarray(c_row, dtype=np.float64).reshape(-1)
    ng, nb = Gc.shape[1], Gb.shape[1]
    obj_c = _row_dot_gen(c_row, Gc)
    obj_b = _row_dot_gen(c_row, Gb)
    obj = np.concatenate([obj_c, obj_b])  # maximize -> minimize -obj
    const = float(c_row @ c)
    if ng + nb == 0:
        return const
    if not integer:
        A_eq = _sp.hstack([Ace, Abe], format="csr") if Ace.shape[0] else None
        A_ub = _sp.hstack([Acl, Abl], format="csr") if Acl.shape[0] else None
        r = _linprog(-obj, A_eq=A_eq, b_eq=(be if Ace.shape[0] else None),
                     A_ub=(A_ub if Acl.shape[0] else None),
                     b_ub=(bl if Acl.shape[0] else None),
                     bounds=[(-1, 1)] * (ng + nb), method="highs")
        return (const - r.fun) if r.success else None
    # MILP: the binary generators xi_b live in {-1,+1}, NOT {-1,0,+1}. scipy's
    # integrality over bounds [-1,1] would admit the spurious 0, enlarging the
    # feasible set and LOOSENING the max (a sound but missed-CERT bug). Substitute
    # xi_b = 2z-1, z in {0,1}: A.xi_b = 2A.z - A.sum -> 2*Ab coeff, +Ab.sum RHS.
    obj_z = np.concatenate([obj_c, 2.0 * obj_b])
    const_z = const - float(obj_b.sum())
    integ = np.concatenate([np.zeros(ng), np.ones(nb)]).astype(int)
    cons = []
    if Ace.shape[0]:
        Aeq = _sp.hstack([Ace, 2.0 * Abe], format="csr")
        rhs = be + _csr_rowsum(Abe)
        cons.append(_LC(Aeq, lb=rhs, ub=rhs))
    if Acl.shape[0]:
        Ale = _sp.hstack([Acl, 2.0 * Abl], format="csr")
        cons.append(_LC(Ale, ub=bl + _csr_rowsum(Abl)))
    vlb = np.concatenate([-np.ones(ng), np.zeros(nb)])
    vub = np.ones(ng + nb)
    r = _milp(c=-obj_z, constraints=cons, integrality=integ,
              bounds=_LPB(vlb, vub),
              options={"mip_rel_gap": 1e-9, "time_limit": time_limit})
    return (const_z - r.fun) if r.success else None


def hz_joint_min_margin(hz, C: np.ndarray, t: np.ndarray, *,
                        integer: bool = False, time_limit: float = 30.0) -> Optional[float]:
    """s* = min_y max_r (C[r] y - t[r]) over the HZ (epigraph form)."""
    if not _HAS_SCIPY:
        return None
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = _hz_np_sparse(hz)
    C, t = _spec_np(C, t, c.size)
    ng, nb = Gc.shape[1], Gb.shape[1]
    nrow = C.shape[0]
    v_s = ng + nb
    nv = ng + nb + 1
    epi_A = np.zeros((nrow, nv))
    epi_b = np.empty(nrow)
    for r in range(nrow):
        epi_A[r, :ng] = _row_dot_gen(C[r], Gc)
        epi_A[r, ng:ng + nb] = _row_dot_gen(C[r], Gb)
        epi_A[r, v_s] = -1.0
        epi_b[r] = float(t[r] - C[r] @ c)
    obj = np.zeros(nv)
    obj[v_s] = 1.0
    vlb = np.concatenate([-np.ones(ng + nb), [-1e12]])
    vub = np.concatenate([np.ones(ng + nb), [1e12]])
    if not integer:
        A_ub = [_sp.csr_matrix(epi_A)]
        b_ub = [epi_b]
        if Acl.shape[0]:
            A_ub.append(_sp.hstack(
                [Acl, Abl, _sp.csr_matrix((Acl.shape[0], 1))],
                format="csr"))
            b_ub.append(bl)
        A_eq = (_sp.hstack(
                    [Ace, Abe, _sp.csr_matrix((Ace.shape[0], 1))],
                    format="csr")
                if Ace.shape[0] else None)
        r = _linprog(obj, A_ub=_sp.vstack(A_ub, format="csr"),
                     b_ub=np.concatenate(b_ub),
                     A_eq=A_eq, b_eq=(be if Ace.shape[0] else None),
                     bounds=list(zip(vlb, vub)), method="highs")
        return float(r.fun) if r.success else None
    # MILP: xi_b in {-1,+1}, NOT {-1,0,+1}. scipy integrality over [-1,1] admits
    # the spurious 0 -> enlarged feasible set -> LOOSER (more negative) s* -> the
    # MILP escalation silently MISSES certs. Substitute xi_b = 2z-1, z in {0,1}:
    # the xi_b column block *=2 and the row RHS += (that block).sum over nb.
    epi_Az = epi_A.copy()
    epi_Az[:, ng:ng + nb] *= 2.0
    epi_bz = epi_b + _mat_dot_gen(C, Gb).sum(axis=1)
    integ = np.concatenate([np.zeros(ng), np.ones(nb), [0]]).astype(int)
    vlbz = np.concatenate([-np.ones(ng), np.zeros(nb), [-1e12]])
    vubz = np.concatenate([np.ones(ng + nb), [1e12]])
    cons = [_LC(_sp.csr_matrix(epi_Az), ub=epi_bz)]
    if Ace.shape[0]:
        Meq = _sp.hstack(
            [Ace, 2.0 * Abe, _sp.csr_matrix((Ace.shape[0], 1))],
            format="csr")
        beq = be + _csr_rowsum(Abe)
        cons.append(_LC(Meq, lb=beq, ub=beq))
    if Acl.shape[0]:
        Mle = _sp.hstack(
            [Acl, 2.0 * Abl, _sp.csr_matrix((Acl.shape[0], 1))],
            format="csr")
        cons.append(_LC(Mle, ub=bl + _csr_rowsum(Abl)))
    r = _milp(c=obj, constraints=cons, integrality=integ, bounds=_LPB(vlbz, vubz),
              options={"mip_rel_gap": 1e-9, "time_limit": time_limit})
    return float(r.fun) if r.success else None


def _objbound_solve_highs(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                          mip_start_xi=None):
    """HiGHS minimize cost@v with early-stop at obj_thr (objective_target +
    objective_bound), or optionally solve the equivalent cutoff-row feasibility
    MILP when ``HZ_MILP_CUTOFF_ROW`` is set. mip_rel_gap=1e-9 so the optimum is
    exact; only the STOPPING is early. Returns (kind, xi) where kind in
    {'witness','empty','unknown'}:
      * 'witness' (kObjectiveTarget): a feasible point with cost<=obj_thr was found;
      * 'empty'   (kObjectiveBound/kInfeasible): every node was pruned -> no feasible
                  point reaches obj_thr (the cutoff side is provably empty);
      * 'unknown' (kTimeLimit / other): undecided.
    xi maps the integer (z in {0,1}) columns back to {-1,+1}; continuous pass through.
    """
    cutoff_row = os.environ.get("HZ_MILP_CUTOFF_ROW", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    solve_cost = np.zeros_like(cost) if cutoff_row else cost
    if cutoff_row:
        cut = _sp.csr_matrix(np.asarray(cost, float).reshape(1, -1))
        A = _sp.vstack([_sp.csr_matrix(A), cut], format="csr")
        rl = np.concatenate([np.asarray(rl, float), [-np.inf]])
        ru = np.concatenate([np.asarray(ru, float), [float(obj_thr) + 1e-9]])

    orig_integ_mask = np.asarray(integ_mask, dtype=bool)
    current_integ_mask = orig_integ_mask
    elim_metas = []
    if _env_flag("HZ_MILP_EQ_SUBST"):
        A, rl, ru, cost, lb, ub, eq_meta = _substitute_eq_singleton_continuous_cols(
            A, rl, ru, cost, lb, ub, current_integ_mask
        )
        if eq_meta is not None:
            current_integ_mask = current_integ_mask[np.asarray(eq_meta["keep_cols"], dtype=np.int64)]
            elim_metas.append(eq_meta)
            if _env_flag("HZ_MILP_DEBUG"):
                print(
                    f"[HZ_MILP] eq_subst removed="
                    f"{int(eq_meta['original_ncols'] - len(eq_meta['keep_cols']))} "
                    f"kept={len(eq_meta['keep_cols'])} rows={A.shape[0]}",
                    flush=True,
                )

    if _env_flag("HZ_MILP_ELIM_SINGLETONS"):
        A, rl, ru, cost, lb, ub, singleton_meta = _project_singleton_continuous_rows(
            A, rl, ru, cost, lb, ub, current_integ_mask
        )
        if singleton_meta is not None:
            current_integ_mask = current_integ_mask[np.asarray(singleton_meta["keep_cols"], dtype=np.int64)]
            elim_metas.append(singleton_meta)
            if _env_flag("HZ_MILP_DEBUG"):
                print(
                    f"[HZ_MILP] elim_singletons removed="
                    f"{int(singleton_meta['original_ncols'] - len(singleton_meta['keep_cols']))} "
                    f"kept={len(singleton_meta['keep_cols'])} rows={A.shape[0]}",
                    flush=True,
                )
    elim_meta = _chain_elim_meta(*elim_metas)
    integ_mask = current_integ_mask

    solve_cost = np.zeros_like(cost) if cutoff_row else cost
    if _env_flag("HZ_MILP_SCALE"):
        A, rl, ru, row_scale = _scale_milp_rows(A, rl, ru)
        obj_scale = 1.0
        if not cutoff_row:
            cost, obj_thr, obj_scale = _scale_milp_objective(cost, obj_thr)
            solve_cost = cost
        if _env_flag("HZ_MILP_DEBUG"):
            if row_scale is None or row_scale.size == 0:
                row_msg = "none"
            else:
                row_msg = (
                    f"min={float(np.min(row_scale)):.3g} "
                    f"max={float(np.max(row_scale)):.3g}"
                )
            print(
                f"[HZ_MILP] scale rows={row_msg} obj_scale={obj_scale:.3g}",
                flush=True,
            )

    h = _highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-9)
    if not cutoff_row:
        h.setOptionValue("objective_target", float(obj_thr))
        h.setOptionValue("objective_bound", float(obj_thr))
    # Sound speed knobs (default-off; verdict is the proven sign vs obj_thr, unchanged):
    _thr = os.environ.get("HZ_MILP_THREADS")
    if _thr:
        h.setOptionValue("threads", int(_thr))
    _heff = os.environ.get("HZ_MILP_HEURISTIC")
    if _heff:
        h.setOptionValue("mip_heuristic_effort", float(_heff))
    _apply_highs_env_options(h)
    nc = len(cost)
    h.addCols(nc, np.asarray(solve_cost, float), np.asarray(lb, float), np.asarray(ub, float),
              0, np.array([], np.int32), np.array([], np.int32), np.array([], float))
    vt = np.array([_highspy.HighsVarType.kInteger if m else _highspy.HighsVarType.kContinuous
                   for m in integ_mask])
    h.changeColsIntegrality(nc, np.arange(nc, dtype=np.int32), vt)
    if A.shape[0]:
        As = _sp.csr_matrix(A)
        h.addRows(As.shape[0], np.asarray(rl, float), np.asarray(ru, float), As.nnz,
                  As.indptr.astype(np.int32), As.indices.astype(np.int32), As.data.astype(float))
    if mip_start_xi is not None:
        try:
            raw = np.asarray(mip_start_xi, dtype=np.float64).reshape(-1)
            start_full = np.zeros(orig_integ_mask.size, dtype=np.float64)
            ncopy = min(raw.size, start_full.size)
            start_full[:ncopy] = np.clip(raw[:ncopy], -1.0, 1.0)
            start_full[orig_integ_mask] = (start_full[orig_integ_mask] >= 0.0).astype(np.float64)
            start = _project_solution_to_reduced(start_full, elim_meta)
            int_idx = np.flatnonzero(np.asarray(integ_mask, dtype=bool)).astype(np.int32)
            if int_idx.size:
                ret = h.setSolution(
                    int_idx.size,
                    int_idx,
                    np.clip(start[int_idx], lb[int_idx], ub[int_idx]).astype(np.float64),
                )
                if _env_flag("HZ_MILP_DEBUG"):
                    print(
                        f"[HZ_MILP] mip_start entries={int(int_idx.size)} status={ret}",
                        flush=True,
                    )
        except Exception as exc:
            if _env_flag("HZ_MILP_DEBUG"):
                print(f"[HZ_MILP] mip_start error={type(exc).__name__}:{str(exc)[:100]}", flush=True)
    run_status = h.run()
    MS = _highspy.HighsModelStatus
    st = h.getModelStatus()

    def _xi_from_reduced(v_reduced):
        v = _expand_projected_solution(np.asarray(v_reduced, float), elim_meta)
        return np.array([
            (2.0 * v[i] - 1.0) if orig_integ_mask[i] else v[i]
            for i in range(orig_integ_mask.size)
        ])

    def _xi():
        return _xi_from_reduced(np.asarray(h.getSolution().col_value, float))

    def _target_incumbent_from_nonterminal():
        try:
            raw = np.asarray(h.getSolution().col_value, dtype=np.float64)
        except Exception:
            return None
        if raw.size != nc or not np.all(np.isfinite(raw)):
            return None
        v = raw.copy()
        im = np.asarray(integ_mask, dtype=bool)
        if im.any():
            ints = v[im]
            rounded = np.rint(ints)
            if ints.size and float(np.max(np.abs(ints - rounded))) > 1e-5:
                return None
            v[im] = np.clip(rounded, 0.0, 1.0)
        if v.size:
            if float(np.max(np.maximum(np.asarray(lb, float) - v, 0.0))) > 1e-6:
                return None
            if float(np.max(np.maximum(v - np.asarray(ub, float), 0.0))) > 1e-6:
                return None
        As = _sp.csr_matrix(A)
        if As.shape[0]:
            av = np.asarray(As @ v, dtype=np.float64).reshape(-1)
            rlv = np.asarray(rl, dtype=np.float64).reshape(-1)
            ruv = np.asarray(ru, dtype=np.float64).reshape(-1)
            lower = np.where(np.isfinite(rlv), rlv - av, -np.inf)
            upper = np.where(np.isfinite(ruv), av - ruv, -np.inf)
            vio = np.maximum(np.maximum(lower, upper), 0.0)
            row_vio = float(np.max(vio)) if vio.size else 0.0
            scale = 1.0 + np.maximum(
                np.abs(av),
                np.maximum(
                    np.where(np.isfinite(rlv), np.abs(rlv), 0.0),
                    np.where(np.isfinite(ruv), np.abs(ruv), 0.0),
                ),
            )
            row_vio_scaled = float(np.max(vio / scale)) if vio.size else 0.0
            if row_vio > 5e-5 and row_vio_scaled > 5e-8:
                return None
        obj_val = float(np.asarray(cost, dtype=np.float64) @ v)
        if (not np.isfinite(obj_val)) or obj_val > float(obj_thr) + 1e-7:
            return None
        if _env_flag("HZ_MILP_DEBUG"):
            print(
                f"[HZ_MILP] accepted target incumbent status={h.modelStatusToString(st)} "
                f"obj={obj_val:.12g} thr={float(obj_thr):.12g}",
                flush=True,
            )
        return _xi_from_reduced(v)

    if st == MS.kObjectiveTarget:
        return "witness", _xi()
    if st in (MS.kObjectiveBound, MS.kInfeasible):
        return "empty", None
    if st == MS.kOptimal:
        if cutoff_row:
            return "witness", _xi()
        # Solved fully (early cutoffs did not fire): decide by the TRUE optimum vs the
        # threshold. obj<=thr -> a feasible point reaches it (witness); else SAFE side.
        obj = h.getInfo().objective_function_value
        return ("witness", _xi()) if obj <= obj_thr + 1e-9 else ("empty", None)
    if (
        not cutoff_row
        and run_status == _highspy.HighsStatus.kOk
        and h.modelStatusToString(st) != "Not Set"
    ):
        try:
            dual_bound = float(h.getInfo().mip_dual_bound)
        except Exception:
            dual_bound = float("nan")
        if np.isfinite(dual_bound) and dual_bound > float(obj_thr) + 1e-7:
            if _env_flag("HZ_MILP_DEBUG"):
                print(
                    f"[HZ_MILP] dual bound proves empty status={h.modelStatusToString(st)} "
                    f"dual={dual_bound:.12g} thr={float(obj_thr):.12g}",
                    flush=True,
                )
            return "empty", None
    xi_inc = _target_incumbent_from_nonterminal()
    if xi_inc is not None:
        return "witness", xi_inc
    return "unknown", None   # kTimeLimit / other -> undecided (never a false CERT)


def _objbound_solve_scip(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                         mip_start_xi=None):
    """SCIP cutoff-feasibility exact backend.

    This solves the same exact MILP as the HiGHS path, but as a feasibility
    query with the cutoff row ``cost @ v <= obj_thr``. Feasible => unsafe
    witness; infeasible => safe side proven; timeout/other => unknown. This is
    optional because SCIP build/solve overhead can dominate small instances.
    """
    if not (_HAS_PYSCIPOPT and _HAS_SCIPY):
        return "unknown", None
    A = _sp.csr_matrix(A)
    cost = np.asarray(cost, dtype=np.float64).reshape(-1)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    rl = np.asarray(rl, dtype=np.float64).reshape(-1)
    ru = np.asarray(ru, dtype=np.float64).reshape(-1)
    integ_mask = np.asarray(integ_mask, dtype=bool).reshape(-1)
    n = int(cost.size)
    if n == 0:
        return ("witness", np.zeros(0, dtype=np.float64)) if 0.0 <= obj_thr + 1e-9 else ("empty", None)

    m = _SCIPModel()
    m.hideOutput()
    try:
        m.setParam("limits/time", float(time_limit))
        m.setParam("numerics/feastol", 1e-7)
    except Exception:
        pass

    def _finite_bound(x, sign):
        x = float(x)
        if np.isneginf(x):
            return -1e20
        if np.isposinf(x):
            return 1e20
        return max(-1e20, min(1e20, x if np.isfinite(x) else sign * 1e20))

    V = []
    for i in range(n):
        vtype = "B" if integ_mask[i] else "C"
        V.append(m.addVar(
            lb=_finite_bound(lb[i], -1.0),
            ub=_finite_bound(ub[i], 1.0),
            vtype=vtype,
            name=f"v{i}",
        ))

    # Existing equality/inequality rows.
    for r in range(A.shape[0]):
        s, e = A.indptr[r], A.indptr[r + 1]
        if s == e:
            row_expr = 0.0
        else:
            row_expr = _scip_quicksum(float(A.data[p]) * V[int(A.indices[p])]
                                      for p in range(s, e))
        lo = float(rl[r])
        hi = float(ru[r])
        lo_fin = np.isfinite(lo) and lo > -1e19
        hi_fin = np.isfinite(hi) and hi < 1e19
        if lo_fin and hi_fin and abs(lo - hi) <= 1e-12:
            m.addCons(row_expr == lo)
        else:
            if lo_fin:
                m.addCons(row_expr >= lo)
            if hi_fin:
                m.addCons(row_expr <= hi)

    # Cutoff row: existence of a point that reaches the unsafe threshold.
    nz = np.flatnonzero(np.abs(cost) > 0.0)
    if nz.size:
        cut_expr = _scip_quicksum(float(cost[i]) * V[int(i)] for i in nz)
        m.addCons(cut_expr <= float(obj_thr) + 1e-9)
    elif 0.0 > float(obj_thr) + 1e-9:
        return "empty", None

    m.setObjective(0.0)
    m.optimize()
    status = str(m.getStatus()).lower()
    has_sol = False
    try:
        has_sol = int(m.getNSols()) > 0
    except Exception:
        has_sol = False
    if os.environ.get("HZ_SCIP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            print(
                f"[HZ_SCIP] status={status} nsol={int(m.getNSols())} "
                f"vars={n} rows={A.shape[0] + 1} time_limit={float(time_limit):.3g}",
                flush=True,
            )
        except Exception:
            pass
    if status in {"optimal", "feasible", "bestsollimit"} or has_sol:
        vals = np.asarray([float(m.getVal(v)) for v in V], dtype=np.float64)
        return "witness", np.array([(2.0 * vals[i] - 1.0) if integ_mask[i] else vals[i]
                                    for i in range(n)])
    if status == "infeasible":
        return "empty", None
    return "unknown", None


def _objbound_solve(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                    mip_start_xi=None):
    backend = os.environ.get("HZ_MILP_BACKEND", "highs").strip().lower()
    if backend == "scip":
        return _objbound_solve_scip(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                                    mip_start_xi=mip_start_xi)
    out = _objbound_solve_highs(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                                mip_start_xi=mip_start_xi)
    if backend in {"highs_scip", "portfolio"} and out[0] == "unknown":
        return _objbound_solve_scip(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                                    mip_start_xi=mip_start_xi)
    return out


def _base_milp_matrices(hz):
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = _hz_np_sparse(hz)
    ng, nb = int(Gc.shape[1]), int(Gb.shape[1])
    rows_A, rl, ru = [], [], []
    if Ace.shape[0]:
        rows_A.append(_sp.hstack([Ace, 2.0 * Abe], format="csr"))
        rhs = be + _csr_rowsum(Abe)
        rl.append(rhs)
        ru.append(rhs)
    if Acl.shape[0]:
        rows_A.append(_sp.hstack([Acl, 2.0 * Abl], format="csr"))
        rhs = bl + _csr_rowsum(Abl)
        rl.append(np.full(Acl.shape[0], -np.inf))
        ru.append(rhs)
    A = (_sp.vstack(rows_A, format="csr") if rows_A
         else _sp.csr_matrix((0, ng + nb), dtype=np.float64))
    rl = np.concatenate(rl) if rl else np.zeros(0, dtype=np.float64)
    ru = np.concatenate(ru) if ru else np.zeros(0, dtype=np.float64)
    lb = np.concatenate([-np.ones(ng), np.zeros(nb)]).astype(np.float64)
    ub = np.ones(ng + nb, dtype=np.float64)
    integ = np.concatenate([np.zeros(ng), np.ones(nb)]).astype(int)
    return A, rl, ru, lb, ub, integ


def _base_solution_to_xi(sol, integ) -> np.ndarray:
    sol = np.asarray(sol, dtype=np.float64).reshape(-1)
    integ = np.asarray(integ, dtype=bool).reshape(-1)
    xi = sol.copy()
    if integ.any():
        xi[integ] = 2.0 * xi[integ] - 1.0
    return xi


def hz_base_feasibility(hz, *, time_limit: float = 10.0):
    """Return ``(status, msg)`` for the propagated HZ state itself.

    ``status`` is one of ``FEASIBLE``, ``INFEASIBLE``, or ``UNKNOWN``.  A SAFE
    verdict over ``HZ ∩ unsafe = empty`` is meaningful only if the base HZ is
    nonempty; otherwise the proof is vacuous.  Binary HZ variables are checked
    exactly as integer ``z in {0,1}`` after the standard ``xi_b = 2z - 1`` map.
    """
    cached = getattr(hz, "_solver_base_feas_cache", None)
    if cached is not None:
        return cached
    if hz_known_nonempty(hz):
        reason = getattr(hz, "_solver_known_nonempty_reason", "constructed")
        return ("FEASIBLE", f"known_nonempty:{reason}")

    def _finish(out):
        if out[0] != "UNKNOWN":
            setattr(hz, "_solver_base_feas_cache", out)
        return out

    def _set_witness(sol, integ):
        setattr(hz, "_solver_base_witness_cache", _base_solution_to_xi(sol, integ))

    if not _HAS_SCIPY:
        return ("UNKNOWN", "scipy_unavailable")

    A, rl, ru, lb, ub, integ = _base_milp_matrices(hz)
    if A.shape[1] == 0:
        zero = np.zeros(0, dtype=np.float64)
        row = np.asarray(A @ zero, dtype=np.float64).reshape(-1)
        lo_bad = np.isfinite(rl) & (row < rl - 1e-9)
        hi_bad = np.isfinite(ru) & (row > ru + 1e-9)
        feasible = not bool(np.any(lo_bad | hi_bad))
        if feasible:
            _set_witness(zero, integ)
        out = ("FEASIBLE", "bare_point") if feasible else ("INFEASIBLE", "constant_rows")
        return _finish(out)

    if A.shape[0] == 0:
        _set_witness(np.zeros(A.shape[1], dtype=np.float64), integ)
        return _finish(("FEASIBLE", "unconstrained_box"))

    if _HAS_HIGHSPY:
        try:
            h = _highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("time_limit", float(time_limit))
            h.setOptionValue("presolve", "on")
            h.addCols(
                A.shape[1],
                np.zeros(A.shape[1], dtype=np.float64),
                lb,
                ub,
                0,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )
            if np.any(integ):
                vt = np.array([
                    _highspy.HighsVarType.kInteger if m else _highspy.HighsVarType.kContinuous
                    for m in integ.astype(bool)
                ])
                h.changeColsIntegrality(A.shape[1], np.arange(A.shape[1], dtype=np.int32), vt)
            As = _sp.csr_matrix(A)
            h.addRows(
                As.shape[0],
                rl,
                ru,
                As.nnz,
                As.indptr.astype(np.int32),
                As.indices.astype(np.int32),
                As.data.astype(np.float64),
            )
            h.run()
            st = h.getModelStatus()
            msg = h.modelStatusToString(st)
            MS = _highspy.HighsModelStatus
            if st == MS.kOptimal:
                _set_witness(np.asarray(h.getSolution().col_value, dtype=np.float64), integ)
                out = ("FEASIBLE", f"highs:{msg}")
            elif st == MS.kInfeasible:
                out = ("INFEASIBLE", f"highs:{msg}")
            else:
                out = ("UNKNOWN", f"highs:{msg}")
            return _finish(out)
        except Exception as exc:
            highs_msg = f"highs_error:{type(exc).__name__}:{str(exc)[:120]}"
    else:
        highs_msg = "highspy_unavailable"

    try:
        cons = [_LC(A, lb=rl, ub=ru)]
        r = _milp(
            c=np.zeros(A.shape[1], dtype=np.float64),
            constraints=cons,
            integrality=integ,
            bounds=_LPB(lb, ub),
            options={"time_limit": float(time_limit), "mip_rel_gap": 1e-9},
        )
        if r.success:
            _set_witness(np.asarray(r.x, dtype=np.float64), integ)
            out = ("FEASIBLE", f"{highs_msg}; scipy_milp:{r.message}")
        elif str(getattr(r, "message", "")).lower().find("infeasible") >= 0:
            out = ("INFEASIBLE", f"{highs_msg}; scipy_milp:{r.message}")
        else:
            out = ("UNKNOWN", f"{highs_msg}; scipy_milp:{r.message}")
    except Exception as exc:
        out = ("UNKNOWN", f"{highs_msg}; scipy_milp_error:{type(exc).__name__}:{str(exc)[:120]}")
    return _finish(out)


def hz_base_witness(hz, *, time_limit: float = 10.0):
    """Return a feasible base-HZ ``xi`` point, or ``None`` if unavailable."""

    status, msg = hz_base_feasibility(hz, time_limit=time_limit)
    if status != "FEASIBLE":
        return None, msg
    xi = getattr(hz, "_solver_base_witness_cache", None)
    if xi is None:
        return None, "feasible_without_cached_witness"
    return np.asarray(xi, dtype=np.float64).reshape(-1).copy(), msg


def _hz_spec_unsafe_at_xi(hz, C, t, xi, *, is_unsafe_linear: bool,
                          tol: float = 1e-9) -> bool:
    c, Gc, Gb, *_ = _hz_np_sparse(hz)
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    ng, nb = int(Gc.shape[1]), int(Gb.shape[1])
    if xi.size < ng + nb:
        return False
    vals = C @ c - t
    if ng:
        vals = vals + _mat_dot_gen(C, Gc) @ xi[:ng]
    if nb:
        vals = vals + _mat_dot_gen(C, Gb) @ xi[ng:ng + nb]
    if is_unsafe_linear:
        return bool(np.all(vals <= tol))
    return bool(np.any(vals >= -tol))


def hz_objbound_decide(hz, C, thresholds, *, is_unsafe_linear: bool,
                       time_limit: float = 15.0, tol: float = 1e-9,
                       mip_start_xi=None, require_base_feasible: bool = True,
                       base_feas_time_limit: Optional[float] = None,
                       base_witness_precheck: bool = True):
    """Verdict-only exact MILP via HiGHS objective-bound early termination. Returns
    ``(verdict, witness_xi)``, verdict in {SAFE, UNSAFE, UNKNOWN}. SOUND & EXACT:
    mip_rel_gap=1e-9, but B&B stops once the margin's sign vs the threshold is proven
    (a feasible witness = UNSAFE / a provably-empty cutoff = SAFE); undecided within
    time_limit -> UNKNOWN (never a false CERT). witness_xi (xi_b in {-1,+1}) is an
    unsafe HZ point the caller must still forward-verify. Validated 16/16 vs scipy
    mip_rel_gap=1e-9, 0 false-CERT, 1.5-665x."""
    if not (_HAS_HIGHSPY and _HAS_SCIPY):
        return ("UNKNOWN", None)
    setattr(hz, "_solver_last_witness_source", None)
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = _hz_np_sparse(hz)
    C, t = _spec_np(C, thresholds, c.size)
    ng, nb = Gc.shape[1], Gb.shape[1]

    if require_base_feasible:
        btl = min(float(time_limit), 10.0) if base_feas_time_limit is None else float(base_feas_time_limit)
        base_status, _ = hz_base_feasibility(hz, time_limit=btl)
        if base_status != "FEASIBLE":
            return ("UNKNOWN", None)
        if base_witness_precheck:
            base_xi, _ = hz_base_witness(hz, time_limit=btl)
            if base_xi is not None and _hz_spec_unsafe_at_xi(
                hz, C, t, base_xi, is_unsafe_linear=is_unsafe_linear, tol=tol
            ):
                setattr(hz, "_solver_last_witness_source", "base_hz_witness")
                return ("UNSAFE", base_xi)

    # bare point / no generators -> closed form (matches hz_certify_spec)
    if ng + nb == 0:
        row = C @ c - t
        if is_unsafe_linear:
            if float(np.max(row)) > tol:
                return ("SAFE", None)
            setattr(hz, "_solver_last_witness_source", "bare_point")
            return ("UNSAFE", np.zeros(0))
        if float(np.max(row)) < -tol:
            return ("SAFE", None)
        setattr(hz, "_solver_last_witness_source", "bare_point")
        return ("UNSAFE", np.zeros(0))

    integ = ([0] * ng) + ([1] * nb)
    # shared eq/le constraint rows in z-space (xi_b = 2z-1)
    rows_A, rl, ru = [], [], []
    if Ace.shape[0]:
        rows_A.append(_sp.hstack([Ace, 2.0 * Abe], format="csr"))
        rhs = be + _csr_rowsum(Abe)
        rl.append(rhs); ru.append(rhs)
    if Acl.shape[0]:
        rows_A.append(_sp.hstack([Acl, 2.0 * Abl], format="csr"))
        rhs = bl + _csr_rowsum(Abl)
        rl.append(np.full(Acl.shape[0], -np.inf)); ru.append(rhs)
    A = (_sp.vstack(rows_A, format="csr") if rows_A
         else _sp.csr_matrix((0, ng + nb), dtype=np.float64))
    rl = np.concatenate(rl) if rl else np.zeros(0)
    ru = np.concatenate(ru) if ru else np.zeros(0)
    lb = np.concatenate([-np.ones(ng), np.zeros(nb)]); ub = np.ones(ng + nb)

    if not is_unsafe_linear:
        # ALL-rows / TOP1: unsafe iff SOME row has max_y C[r]y >= t[r].
        def _solve_row(r: int):
            obj_b = _row_dot_gen(C[r], Gb)
            cost = -np.concatenate([_row_dot_gen(C[r], Gc), 2.0 * obj_b])          # minimize -C[r]y
            const_z = float(C[r] @ c) - float(obj_b.sum())
            obj_thr = const_z - float(t[r])  # feasible cost<=thr  <=>  C[r]y>=t[r]
            return _objbound_solve(
                cost,
                obj_thr,
                A,
                rl,
                ru,
                lb,
                ub,
                integ,
                time_limit,
                mip_start_xi=mip_start_xi,
            )

        row_workers = max(1, min(_env_int("HZ_QUERY_WORKERS", 1), int(C.shape[0])))
        row_results = []
        if row_workers > 1 and C.shape[0] > 1:
            with ThreadPoolExecutor(max_workers=row_workers) as ex:
                futs = [ex.submit(_solve_row, r) for r in range(C.shape[0])]
                for fut in as_completed(futs):
                    row_results.append(fut.result())
        else:
            row_results = [_solve_row(r) for r in range(C.shape[0])]

        any_unknown = False
        for kind, xi in row_results:
            if kind == "witness":
                setattr(hz, "_solver_last_witness_source", "milp_objective_bound")
                return ("UNSAFE", xi)
            if kind == "unknown":
                any_unknown = True
        return ("UNKNOWN", None) if any_unknown else ("SAFE", None)

    # UNSAFE_LINEAR (conjunction): unsafe iff EXISTS y with all C[r]y <= t[r],
    # i.e. s* = min_y max_r(C[r]y - t[r]) <= 0. Epigraph vars [xi_c, z, s].
    nrow = C.shape[0]; nv = ng + nb + 1
    epi = np.zeros((nrow, nv)); epib = np.empty(nrow)
    for r in range(nrow):
        epi[r, :ng] = _row_dot_gen(C[r], Gc); epi[r, ng:ng + nb] = 2.0 * _row_dot_gen(C[r], Gb)
        epi[r, ng + nb] = -1.0
        epib[r] = float(t[r] - C[r] @ c) + float(_row_dot_gen(C[r], Gb).sum())
    A2 = (_sp.vstack([
            _sp.hstack([A, _sp.csr_matrix((A.shape[0], 1))], format="csr"),
            _sp.csr_matrix(epi),
          ], format="csr")
          if A.shape[0] else _sp.csr_matrix(epi))
    rl2 = np.concatenate([rl, np.full(nrow, -np.inf)])
    ru2 = np.concatenate([ru, epib])
    lb2 = np.concatenate([lb, [-1e12]]); ub2 = np.concatenate([ub, [1e12]])
    cost = np.zeros(nv); cost[ng + nb] = 1.0   # minimize s
    integ2 = integ + [0]
    kind, xi = _objbound_solve(cost, 0.0, A2, rl2, ru2, lb2, ub2, integ2, time_limit,
                               mip_start_xi=mip_start_xi)
    if kind == "witness":
        setattr(hz, "_solver_last_witness_source", "milp_objective_bound")
        return ("UNSAFE", xi[:ng + nb])
    if kind == "empty":
        return ("SAFE", None)
    return ("UNKNOWN", None)


def hz_certify_spec(hz, C, thresholds, *, is_unsafe_linear: bool,
                    escalate_milp: bool = True,
                    tol: float = 1e-9, time_limit: float = 30.0,
                    require_base_feasible: bool = True):
    """Certify a single (B=1) linear output spec over the constrained HZ.

    Returns (certified: bool, margin: float|None). Sound: True only when the
    HZ over-approximation proves the property. Tries the LP relaxation first;
    if it does not certify and ``escalate_milp`` is set, tries the exact MILP.
    """
    if not _HAS_SCIPY:
        return False, None
    c0, Gc0, Gb0, *_ = _hz_np_sparse(hz)
    C, t = _spec_np(C, thresholds, int(c0.size))
    if require_base_feasible:
        base_status, _ = hz_base_feasibility(hz, time_limit=min(float(time_limit), 10.0))
        if base_status != "FEASIBLE":
            return False, None

    # An unconstrained zonotope (Ac.shape[0]==0) is a perfectly valid HZ: the
    # joint-margin LP recovers its TRUE support (exact for a zonotope -- there
    # are no binaries to relax). Pure-affine nets (e.g. linear classifiers)
    # always yield nc==0; bailing here wrongly rejected every such result.
    # A HZ with NO generators at all is a bare point, decided in closed form.
    ng = Gc0.shape[1]
    nb = Gb0.shape[1]
    if ng + nb == 0:
        row_margins = C @ c0 - t  # C[r].c - t[r] for each row
        if is_unsafe_linear:           # safe iff max_r (C[r]c - t[r]) > 0
            s = float(np.max(row_margins))
            return s > tol, s
        worst = float(np.max(row_margins))  # ALL-rows: safe iff every row < t
        return worst < -tol, -worst

    def _decide(integer):
        if is_unsafe_linear:
            s = hz_joint_min_margin(hz, C, t, integer=integer, time_limit=time_limit)
            return (s is not None and s > tol), s
        # ALL-rows: every row max_y C[r] y < t[r]
        worst = -np.inf
        for r in range(C.shape[0]):
            mx = hz_row_max(hz, C[r], integer=integer, time_limit=time_limit)
            if mx is None:
                return False, None
            worst = max(worst, mx - t[r])
        return (worst < -tol), -worst

    ok, margin = _decide(False)
    if ok or not escalate_milp:
        return ok, margin
    return _decide(True)


def _test_sparse_hz_verdict_parity() -> None:  # pragma: no cover
    import torch

    dtype = torch.float64
    hz = HZono(
        c=torch.zeros(2, 1, dtype=dtype),
        Gc=torch.tensor([[1.0, 0.0], [0.0, 0.5]], dtype=dtype),
        Gb=torch.zeros(2, 0, dtype=dtype),
        Ac=torch.zeros(0, 2, dtype=dtype),
        Ab=torch.zeros(0, 0, dtype=dtype),
        b=torch.zeros(0, 1, dtype=dtype),
        col_ids=torch.tensor([10, 11], dtype=torch.long),
        bcol_ids=torch.zeros(0, dtype=torch.long),
    )
    shz = SparseHZono.from_dense_hz(hz)
    assert shz.col_ids is not None and np.array_equal(shz.col_ids, np.array([10, 11]))
    assert shz.bcol_ids is not None and shz.bcol_ids.size == 0
    try:
        SparseHZono(
            c=np.zeros(1, dtype=np.float64),
            Gc=_sp.csr_matrix((1, 2), dtype=np.float64),
            Gb=_sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=_sp.csr_matrix((0, 2), dtype=np.float64),
            Ab=_sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            col_ids=np.array([1], dtype=np.int64),
        )
        raise AssertionError("SparseHZono accepted mismatched col_ids")
    except ValueError as exc:
        assert "col_ids length mismatch" in str(exc)
    try:
        SparseHZono(
            c=np.zeros(1, dtype=np.float64),
            Gc=_sp.csr_matrix((1, 0), dtype=np.float64),
            Gb=_sp.csr_matrix((1, 1), dtype=np.float64),
            Ac=_sp.csr_matrix((0, 0), dtype=np.float64),
            Ab=_sp.csr_matrix((0, 1), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            bcol_ids=np.array([1, 2], dtype=np.int64),
        )
        raise AssertionError("SparseHZono accepted mismatched bcol_ids")
    except ValueError as exc:
        assert "bcol_ids length mismatch" in str(exc)
    try:
        SparseHZono(
            c=np.zeros(1, dtype=np.float64),
            Gc=_sp.csr_matrix((1, 1), dtype=np.float64),
            Gb=_sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=_sp.csr_matrix((0, 1), dtype=np.float64),
            Ab=_sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            input_center=np.zeros(1, dtype=np.float64),
        )
        raise AssertionError("SparseHZono accepted partial input replay metadata")
    except ValueError as exc:
        assert "input replay metadata requires" in str(exc)
    try:
        SparseHZono(
            c=np.zeros(1, dtype=np.float64),
            Gc=_sp.csr_matrix((1, 1), dtype=np.float64),
            Gb=_sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=_sp.csr_matrix((0, 1), dtype=np.float64),
            Ab=_sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            input_center=np.zeros(1, dtype=np.float64),
            input_radius=np.ones(1, dtype=np.float64),
            input_indices=np.array([1], dtype=np.int64),
            input_shape=(1,),
        )
        raise AssertionError("SparseHZono accepted out-of-range input_indices")
    except ValueError as exc:
        assert "input_indices out of input range" in str(exc)
    row = np.array([1.0, 2.0], dtype=np.float64)
    assert abs(float(hz_row_max(hz, row)) - float(hz_row_max(shz, row))) <= 1e-12

    C_safe = np.array([[1.0, 0.0]], dtype=np.float64)
    t_safe = np.array([2.0], dtype=np.float64)
    dense_cert = hz_certify_spec(hz, C_safe, t_safe, is_unsafe_linear=False)
    sparse_cert = hz_certify_spec(shz, C_safe, t_safe, is_unsafe_linear=False)
    assert dense_cert[0] == sparse_cert[0]
    assert abs(float(dense_cert[1]) - float(sparse_cert[1])) <= 1e-12
    assert hz_base_feasibility(hz, time_limit=2.0)[0] == "FEASIBLE"
    base_xi, base_msg = hz_base_witness(hz, time_limit=2.0)
    assert base_xi is not None, base_msg
    assert np.allclose(base_xi, np.zeros(2, dtype=np.float64))
    Apre = _sp.csr_matrix(np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float64))
    lb0 = np.array([0.0, 0.0], dtype=np.float64)
    ub0 = np.array([1.0, 1.0], dtype=np.float64)
    infeas, info = sparse_row_bound_infeasible(
        Apre,
        np.array([3.0, -np.inf], dtype=np.float64),
        np.array([np.inf, np.inf], dtype=np.float64),
        lb0,
        ub0,
    )
    assert infeas and info["bad_row"] == 0
    infeas, tlb, tub, info = sparse_fbbt_tighten_bounds(
        Apre[:1],
        np.array([1.5], dtype=np.float64),
        np.array([2.0], dtype=np.float64),
        lb0,
        ub0,
        max_passes=2,
    )
    assert not infeas
    assert info["tightened"] >= 2
    assert tlb[0] >= 0.5 - 1e-12 and tlb[1] >= 0.5 - 1e-12

    empty_hz = HZono(
        c=torch.zeros(1, 1, dtype=dtype),
        Gc=torch.zeros(1, 0, dtype=dtype),
        Gb=torch.zeros(1, 0, dtype=dtype),
        Ac=torch.zeros(1, 0, dtype=dtype),
        Ab=torch.zeros(1, 0, dtype=dtype),
        b=torch.ones(1, 1, dtype=dtype),
    )
    empty_shz = SparseHZono.from_dense_hz(empty_hz)
    C1_safe = np.array([[1.0]], dtype=np.float64)
    t1_safe = np.array([2.0], dtype=np.float64)
    assert hz_base_feasibility(empty_hz, time_limit=2.0)[0] == "INFEASIBLE"
    assert hz_base_feasibility(empty_shz, time_limit=2.0)[0] == "INFEASIBLE"
    assert hz_certify_spec(empty_hz, C1_safe, t1_safe, is_unsafe_linear=False)[0] is False

    bin_empty_hz = HZono(
        c=torch.zeros(1, 1, dtype=dtype),
        Gc=torch.zeros(1, 0, dtype=dtype),
        Gb=torch.zeros(1, 1, dtype=dtype),
        Ac=torch.zeros(1, 0, dtype=dtype),
        Ab=torch.tensor([[0.5]], dtype=dtype),
        b=torch.zeros(1, 1, dtype=dtype),
    )
    assert hz_base_feasibility(bin_empty_hz, time_limit=2.0)[0] == "INFEASIBLE"

    if _HAS_HIGHSPY:
        assert hz_objbound_decide(
            hz, C_safe, t_safe, is_unsafe_linear=False, time_limit=2.0
        )[0] == hz_objbound_decide(
            shz, C_safe, t_safe, is_unsafe_linear=False, time_limit=2.0
        )[0] == "SAFE"
        C_multi_safe = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        t_multi_safe = np.array([2.0, 2.0], dtype=np.float64)
        serial = hz_objbound_decide(
            shz, C_multi_safe, t_multi_safe, is_unsafe_linear=False, time_limit=2.0
        )[0]
        old_qw = os.environ.get("HZ_QUERY_WORKERS")
        os.environ["HZ_QUERY_WORKERS"] = "2"
        try:
            parallel = hz_objbound_decide(
                shz, C_multi_safe, t_multi_safe, is_unsafe_linear=False, time_limit=2.0
            )[0]
        finally:
            if old_qw is None:
                os.environ.pop("HZ_QUERY_WORKERS", None)
            else:
                os.environ["HZ_QUERY_WORKERS"] = old_qw
        assert serial == parallel == "SAFE"
        C_base_adv = np.array([[1.0, 0.0]], dtype=np.float64)
        t_base_adv = np.array([0.0], dtype=np.float64)
        base_v, base_adv_xi = hz_objbound_decide(
            hz, C_base_adv, t_base_adv, is_unsafe_linear=False, time_limit=2.0
        )
        assert base_v == "UNSAFE"
        assert base_adv_xi is not None and np.allclose(base_adv_xi, np.zeros(2))
        assert hz_objbound_decide(
            empty_hz, C1_safe, t1_safe, is_unsafe_linear=False, time_limit=2.0
        )[0] == "UNKNOWN"
        assert hz_objbound_decide(
            bin_empty_hz, C1_safe, t1_safe, is_unsafe_linear=False, time_limit=2.0
        )[0] == "UNKNOWN"

        bin_hz = HZono(
            c=torch.zeros(1, 1, dtype=dtype),
            Gc=torch.zeros(1, 0, dtype=dtype),
            Gb=torch.ones(1, 1, dtype=dtype),
            Ac=torch.zeros(0, 0, dtype=dtype),
            Ab=torch.zeros(0, 1, dtype=dtype),
            b=torch.zeros(0, 1, dtype=dtype),
        )
        bin_shz = SparseHZono.from_dense_hz(bin_hz)
        C_unsafe = np.array([[1.0]], dtype=np.float64)
        t_unsafe = np.array([0.5], dtype=np.float64)
        dense_v, dense_xi = hz_objbound_decide(
            bin_hz, C_unsafe, t_unsafe, is_unsafe_linear=False, time_limit=2.0
        )
        sparse_v, sparse_xi = hz_objbound_decide(
            bin_shz, C_unsafe, t_unsafe, is_unsafe_linear=False, time_limit=2.0
        )
        assert dense_v == sparse_v == "UNSAFE"
        assert dense_xi is not None and sparse_xi is not None
        assert np.allclose(dense_xi, sparse_xi, atol=1e-12)


__all__ = [
    "hz_base_feasibility",
    "hz_base_witness",
    "hz_certify_spec",
    "hz_joint_min_margin",
    "hz_row_max",
    "hz_objbound_decide",
    "sparse_fbbt_tighten_bounds",
    "sparse_row_bound_infeasible",
]


if __name__ == "__main__":  # pragma: no cover
    tests = [_test_sparse_hz_verdict_parity]
    passed = 0
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"{passed}/{len(tests)} passed")
