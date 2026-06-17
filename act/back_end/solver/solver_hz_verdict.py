# ===- act/back_end/solver/solver_hz_verdict.py - HZ spec-margin verdict -===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
"""Constrained Hybrid-Zonotope spec-margin certification (open-source solver).

Given the propagated output HZ and a linear output spec ``C y`` / ``thresholds``,
this decides robustness by SOLVING the spec margin over the constrained HZ --
not by interval arithmetic on the output box. Two regimes, both sound and both
using only scipy/HiGHS (Gurobi is forbidden in the proof path, P3):

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
from typing import Optional
import numpy as np

from act.back_end.solver.solver_hz import HZono, _split_eq_le

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


def _hz_np(hz: HZono):
    c = hz.c.detach().cpu().double().numpy().reshape(-1)
    Gc = hz.Gc.detach().cpu().double().numpy()
    Gb = hz.Gb.detach().cpu().double().numpy()
    (Ace, Abe, be), (Acl, Abl, bl) = _split_eq_le(hz)
    return (c, Gc, Gb,
            Ace.detach().cpu().double().numpy(), Abe.detach().cpu().double().numpy(),
            be.detach().cpu().double().numpy().reshape(-1),
            Acl.detach().cpu().double().numpy(), Abl.detach().cpu().double().numpy(),
            bl.detach().cpu().double().numpy().reshape(-1))


def _spec_np(C, thresholds, out_dim: int):
    C = np.asarray(C, dtype=np.float64).reshape(-1, out_dim)
    t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if t.size == 1 and C.shape[0] != 1:
        t = np.repeat(t, C.shape[0])
    return C, t


def hz_row_max(hz: HZono, c_row: np.ndarray, *, integer: bool = False,
               time_limit: float = 20.0) -> Optional[float]:
    """max_y (c_row . y) over the HZ. LP relaxation (convex hull) or MILP."""
    if not _HAS_SCIPY:
        return None
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = _hz_np(hz)
    c_row = np.asarray(c_row, dtype=np.float64).reshape(-1)
    ng, nb = Gc.shape[1], Gb.shape[1]
    obj = np.concatenate([c_row @ Gc, c_row @ Gb])  # maximize -> minimize -obj
    const = float(c_row @ c)
    if not integer:
        A_eq = np.hstack([Ace, Abe]) if Ace.shape[0] else None
        A_ub = np.hstack([Acl, Abl]) if Acl.shape[0] else None
        r = _linprog(-obj, A_eq=A_eq, b_eq=(be if Ace.shape[0] else None),
                     A_ub=(A_ub if Acl.shape[0] else None),
                     b_ub=(bl if Acl.shape[0] else None),
                     bounds=[(-1, 1)] * (ng + nb), method="highs")
        return (const - r.fun) if r.success else None
    # MILP: the binary generators xi_b live in {-1,+1}, NOT {-1,0,+1}. scipy's
    # integrality over bounds [-1,1] would admit the spurious 0, enlarging the
    # feasible set and LOOSENING the max (a sound but missed-CERT bug). Substitute
    # xi_b = 2z-1, z in {0,1}: A.xi_b = 2A.z - A.sum -> 2*Ab coeff, +Ab.sum RHS.
    obj_z = np.concatenate([c_row @ Gc, 2.0 * (c_row @ Gb)])
    const_z = const - float((c_row @ Gb).sum())
    integ = np.concatenate([np.zeros(ng), np.ones(nb)]).astype(int)
    cons = []
    if Ace.shape[0]:
        cons.append(_LC(_sp.csr_matrix(np.hstack([Ace, 2.0 * Abe])),
                        lb=be + Abe.sum(axis=1), ub=be + Abe.sum(axis=1)))
    if Acl.shape[0]:
        cons.append(_LC(_sp.csr_matrix(np.hstack([Acl, 2.0 * Abl])),
                        ub=bl + Abl.sum(axis=1)))
    vlb = np.concatenate([-np.ones(ng), np.zeros(nb)])
    vub = np.ones(ng + nb)
    r = _milp(c=-obj_z, constraints=cons, integrality=integ,
              bounds=_LPB(vlb, vub),
              options={"mip_rel_gap": 1e-9, "time_limit": time_limit})
    return (const_z - r.fun) if r.success else None


def hz_joint_min_margin(hz: HZono, C: np.ndarray, t: np.ndarray, *,
                        integer: bool = False, time_limit: float = 30.0) -> Optional[float]:
    """s* = min_y max_r (C[r] y - t[r]) over the HZ (epigraph form)."""
    if not _HAS_SCIPY:
        return None
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = _hz_np(hz)
    C, t = _spec_np(C, t, c.size)
    ng, nb = Gc.shape[1], Gb.shape[1]
    nrow = C.shape[0]
    v_s = ng + nb
    nv = ng + nb + 1
    epi_A = np.zeros((nrow, nv))
    epi_b = np.empty(nrow)
    for r in range(nrow):
        epi_A[r, :ng] = C[r] @ Gc
        epi_A[r, ng:ng + nb] = C[r] @ Gb
        epi_A[r, v_s] = -1.0
        epi_b[r] = float(t[r] - C[r] @ c)
    obj = np.zeros(nv)
    obj[v_s] = 1.0
    vlb = np.concatenate([-np.ones(ng + nb), [-1e12]])
    vub = np.concatenate([np.ones(ng + nb), [1e12]])
    if not integer:
        A_ub = [epi_A]
        b_ub = [epi_b]
        if Acl.shape[0]:
            A_ub.append(np.hstack([Acl, Abl, np.zeros((Acl.shape[0], 1))]))
            b_ub.append(bl)
        A_eq = (np.hstack([Ace, Abe, np.zeros((Ace.shape[0], 1))])
                if Ace.shape[0] else None)
        r = _linprog(obj, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                     A_eq=A_eq, b_eq=(be if Ace.shape[0] else None),
                     bounds=list(zip(vlb, vub)), method="highs")
        return float(r.fun) if r.success else None
    # MILP: xi_b in {-1,+1}, NOT {-1,0,+1}. scipy integrality over [-1,1] admits
    # the spurious 0 -> enlarged feasible set -> LOOSER (more negative) s* -> the
    # MILP escalation silently MISSES certs. Substitute xi_b = 2z-1, z in {0,1}:
    # the xi_b column block *=2 and the row RHS += (that block).sum over nb.
    epi_Az = epi_A.copy()
    epi_Az[:, ng:ng + nb] *= 2.0
    epi_bz = epi_b + (C @ Gb).sum(axis=1)
    integ = np.concatenate([np.zeros(ng), np.ones(nb), [0]]).astype(int)
    vlbz = np.concatenate([-np.ones(ng), np.zeros(nb), [-1e12]])
    vubz = np.concatenate([np.ones(ng + nb), [1e12]])
    cons = [_LC(_sp.csr_matrix(epi_Az), ub=epi_bz)]
    if Ace.shape[0]:
        Meq = np.hstack([Ace, 2.0 * Abe, np.zeros((Ace.shape[0], 1))])
        beq = be + Abe.sum(axis=1)
        cons.append(_LC(_sp.csr_matrix(Meq), lb=beq, ub=beq))
    if Acl.shape[0]:
        Mle = np.hstack([Acl, 2.0 * Abl, np.zeros((Acl.shape[0], 1))])
        cons.append(_LC(_sp.csr_matrix(Mle), ub=bl + Abl.sum(axis=1)))
    r = _milp(c=obj, constraints=cons, integrality=integ, bounds=_LPB(vlbz, vubz),
              options={"mip_rel_gap": 1e-9, "time_limit": time_limit})
    return float(r.fun) if r.success else None


def _objbound_solve(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit):
    """HiGHS minimize cost@v with early-stop at obj_thr (objective_target +
    objective_bound). mip_rel_gap=1e-9 so the optimum is exact; only the STOPPING
    is early. Returns (kind, xi) where kind in {'witness','empty','unknown'}:
      * 'witness' (kObjectiveTarget): a feasible point with cost<=obj_thr was found;
      * 'empty'   (kObjectiveBound/kInfeasible): every node was pruned -> no feasible
                  point reaches obj_thr (the cutoff side is provably empty);
      * 'unknown' (kTimeLimit / other): undecided.
    xi maps the integer (z in {0,1}) columns back to {-1,+1}; continuous pass through.
    """
    h = _highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-9)
    h.setOptionValue("objective_target", float(obj_thr))
    h.setOptionValue("objective_bound", float(obj_thr))
    nc = len(cost)
    h.addCols(nc, np.asarray(cost, float), np.asarray(lb, float), np.asarray(ub, float),
              0, np.array([], np.int32), np.array([], np.int32), np.array([], float))
    vt = np.array([_highspy.HighsVarType.kInteger if m else _highspy.HighsVarType.kContinuous
                   for m in integ_mask])
    h.changeColsIntegrality(nc, np.arange(nc, dtype=np.int32), vt)
    if A.shape[0]:
        As = _sp.csr_matrix(A)
        h.addRows(As.shape[0], np.asarray(rl, float), np.asarray(ru, float), As.nnz,
                  As.indptr.astype(np.int32), As.indices.astype(np.int32), As.data.astype(float))
    h.run()
    MS = _highspy.HighsModelStatus
    st = h.getModelStatus()

    def _xi():
        v = np.asarray(h.getSolution().col_value, float)
        return np.array([(2.0 * v[i] - 1.0) if integ_mask[i] else v[i] for i in range(nc)])

    if st == MS.kObjectiveTarget:
        return "witness", _xi()
    if st in (MS.kObjectiveBound, MS.kInfeasible):
        return "empty", None
    if st == MS.kOptimal:
        # Solved fully (early cutoffs did not fire): decide by the TRUE optimum vs the
        # threshold. obj<=thr -> a feasible point reaches it (witness); else SAFE side.
        obj = h.getInfo().objective_function_value
        return ("witness", _xi()) if obj <= obj_thr + 1e-9 else ("empty", None)
    return "unknown", None   # kTimeLimit / other -> undecided (never a false CERT)


def hz_objbound_decide(hz: HZono, C, thresholds, *, is_unsafe_linear: bool,
                       time_limit: float = 15.0, tol: float = 1e-9):
    """Verdict-only exact MILP via HiGHS objective-bound early termination. Returns
    ``(verdict, witness_xi)``, verdict in {SAFE, UNSAFE, UNKNOWN}. SOUND & EXACT:
    mip_rel_gap=1e-9, but B&B stops once the margin's sign vs the threshold is proven
    (a feasible witness = UNSAFE / a provably-empty cutoff = SAFE); undecided within
    time_limit -> UNKNOWN (never a false CERT). witness_xi (xi_b in {-1,+1}) is an
    unsafe HZ point the caller must still forward-verify. Validated 16/16 vs scipy
    mip_rel_gap=1e-9, 0 false-CERT, 1.5-665x."""
    if not (_HAS_HIGHSPY and _HAS_SCIPY):
        return ("UNKNOWN", None)
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = _hz_np(hz)
    C, t = _spec_np(C, thresholds, c.size)
    ng, nb = Gc.shape[1], Gb.shape[1]

    # bare point / no generators -> closed form (matches hz_certify_spec)
    if ng + nb == 0:
        row = C @ c - t
        if is_unsafe_linear:
            return ("SAFE", None) if float(np.max(row)) > tol else ("UNSAFE", np.zeros(0))
        return ("SAFE", None) if float(np.max(row)) < -tol else ("UNSAFE", np.zeros(0))

    integ = ([0] * ng) + ([1] * nb)
    # shared eq/le constraint rows in z-space (xi_b = 2z-1)
    rows_A, rl, ru = [], [], []
    if Ace.shape[0]:
        rows_A.append(np.hstack([Ace, 2.0 * Abe])); rhs = be + Abe.sum(1); rl.append(rhs); ru.append(rhs)
    if Acl.shape[0]:
        rows_A.append(np.hstack([Acl, 2.0 * Abl])); rhs = bl + Abl.sum(1)
        rl.append(np.full(Acl.shape[0], -np.inf)); ru.append(rhs)
    A = np.vstack(rows_A) if rows_A else np.zeros((0, ng + nb))
    rl = np.concatenate(rl) if rl else np.zeros(0)
    ru = np.concatenate(ru) if ru else np.zeros(0)
    lb = np.concatenate([-np.ones(ng), np.zeros(nb)]); ub = np.ones(ng + nb)

    if not is_unsafe_linear:
        # ALL-rows / TOP1: unsafe iff SOME row has max_y C[r]y >= t[r].
        any_unknown = False
        for r in range(C.shape[0]):
            obj_b = C[r] @ Gb
            cost = -np.concatenate([C[r] @ Gc, 2.0 * obj_b])          # minimize -C[r]y
            const_z = float(C[r] @ c) - float(obj_b.sum())
            obj_thr = const_z - float(t[r])  # feasible cost<=thr  <=>  C[r]y>=t[r]
            kind, xi = _objbound_solve(cost, obj_thr, A, rl, ru, lb, ub, integ, time_limit)
            if kind == "witness":
                return ("UNSAFE", xi)
            if kind == "unknown":
                any_unknown = True
        return ("UNKNOWN", None) if any_unknown else ("SAFE", None)

    # UNSAFE_LINEAR (conjunction): unsafe iff EXISTS y with all C[r]y <= t[r],
    # i.e. s* = min_y max_r(C[r]y - t[r]) <= 0. Epigraph vars [xi_c, z, s].
    nrow = C.shape[0]; nv = ng + nb + 1
    epi = np.zeros((nrow, nv)); epib = np.empty(nrow)
    for r in range(nrow):
        epi[r, :ng] = C[r] @ Gc; epi[r, ng:ng + nb] = 2.0 * (C[r] @ Gb)
        epi[r, ng + nb] = -1.0
        epib[r] = float(t[r] - C[r] @ c) + float((C[r] @ Gb).sum())
    A2 = np.vstack([np.hstack([A, np.zeros((A.shape[0], 1))]), epi]) if A.shape[0] else epi
    rl2 = np.concatenate([rl, np.full(nrow, -np.inf)])
    ru2 = np.concatenate([ru, epib])
    lb2 = np.concatenate([lb, [-1e12]]); ub2 = np.concatenate([ub, [1e12]])
    cost = np.zeros(nv); cost[ng + nb] = 1.0   # minimize s
    integ2 = integ + [0]
    kind, xi = _objbound_solve(cost, 0.0, A2, rl2, ru2, lb2, ub2, integ2, time_limit)
    if kind == "witness":
        return ("UNSAFE", xi[:ng + nb])
    if kind == "empty":
        return ("SAFE", None)
    return ("UNKNOWN", None)


def hz_certify_spec(hz: HZono, C, thresholds, *, is_unsafe_linear: bool,
                    escalate_milp: bool = True,
                    tol: float = 1e-9, time_limit: float = 30.0):
    """Certify a single (B=1) linear output spec over the constrained HZ.

    Returns (certified: bool, margin: float|None). Sound: True only when the
    HZ over-approximation proves the property. Tries the LP relaxation first;
    if it does not certify and ``escalate_milp`` is set, tries the exact MILP.
    """
    if not _HAS_SCIPY:
        return False, None
    C, t = _spec_np(C, thresholds, int(hz.c.numel()))

    # An unconstrained zonotope (Ac.shape[0]==0) is a perfectly valid HZ: the
    # joint-margin LP recovers its TRUE support (exact for a zonotope -- there
    # are no binaries to relax). Pure-affine nets (e.g. linear classifiers)
    # always yield nc==0; bailing here wrongly rejected every such result.
    # A HZ with NO generators at all is a bare point, decided in closed form.
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    if ng + nb == 0:
        c = hz.c.detach().cpu().double().numpy().reshape(-1)
        row_margins = C @ c - t  # C[r].c - t[r] for each row
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


__all__ = ["hz_certify_spec", "hz_joint_min_margin", "hz_row_max", "hz_objbound_decide"]
