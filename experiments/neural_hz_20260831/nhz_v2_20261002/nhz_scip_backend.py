"""SCIP backend for terminal plans (same matrix, bounds, integrality and cutoff as HiGHS).

minimise cost^T v  s.t.  lo <= M v <= hi,  col_lo <= v <= col_hi,  v_j integer for integ_j;
the objective limit cc + pad + margin makes "infeasible" mean "no point reaches the
violation threshold" (excluded), exactly as HiGHS objective_bound.  Returns the same dict
fields as run_portfolio_v7."""
import time
import numpy as np


def run_scip(M, lo, hi, col_lo, col_hi, cost, integ, K, cc, pad, time_limit, margin=1e-4, target=None, threads=1):
    from pyscipopt import Model, quicksum
    from nhz_terminal_v7 import _fold_tiny
    t0 = time.time()
    if not (np.isfinite(M.data).all() and np.isfinite(col_lo).all() and np.isfinite(col_hi).all() and np.isfinite(cost).all()):
        return {"upper": None, "excluded": False, "status": "scip:nonfinite-plan", "binaries": int(np.sum(integ)), "rows": int(M.shape[0]),
                "wall_s": time.time() - t0, "build_s": 0.0, "incumbent_w": None, "incumbent_violation": -np.inf}
    M, lo, hi = _fold_tiny(M, lo, hi, col_lo, col_hi)   # LOG N085: SCIP drops |a| <= 1e-9 as well
    m = Model(); m.hideOutput()
    m.setParam("limits/time", float(max(time_limit, 0.1)))
    m.setParam("parallel/maxnthreads", int(threads))
    n = M.shape[1]
    xs = [m.addVar(lb=float(col_lo[j]), ub=float(col_hi[j]), vtype=("B" if integ[j] and col_lo[j] >= 0 and col_hi[j] <= 1 else ("I" if integ[j] else "C")),
                   obj=float(cost[j])) for j in range(n)]
    Mr = M.tocsr()
    for i in range(Mr.shape[0]):
        s, e = Mr.indptr[i], Mr.indptr[i + 1]
        if s == e:
            continue
        expr = quicksum(float(v) * xs[int(c)] for c, v in zip(Mr.indices[s:e], Mr.data[s:e]))
        l_, h_ = lo[i], hi[i]
        if np.isfinite(l_) and np.isfinite(h_) and l_ == h_:
            m.addCons(expr == float(l_))
        else:
            if np.isfinite(h_):
                m.addCons(expr <= float(h_))
            if np.isfinite(l_):
                m.addCons(expr >= float(l_))
    m.setObjlimit(float(cc + pad + margin))
    build = time.time() - t0
    m.optimize()
    st = m.getStatus()
    excluded = st == "infeasible"
    upper = -margin if excluded else None
    if not excluded:
        try:
            db = m.getDualbound()
            if np.isfinite(db):
                upper = -db + cc + pad
                excluded = upper < -margin
        except Exception:
            pass
    inc = None; best_v = -np.inf
    if m.getNSols() > 0:
        sol = m.getBestSol(); vals = np.array([m.getSolVal(sol, x) for x in xs[:K]])
        inc = vals; best_v = cc - m.getSolObjVal(sol)
    return {"upper": upper, "excluded": bool(excluded), "status": f"scip:{st}", "binaries": int(np.sum(integ)),
            "rows": int(M.shape[0]), "wall_s": time.time() - t0, "build_s": build, "incumbent_w": inc, "incumbent_violation": best_v}
