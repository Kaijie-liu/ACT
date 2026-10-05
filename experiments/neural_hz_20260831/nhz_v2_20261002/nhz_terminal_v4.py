"""Terminal MILP v4: fixed seed portfolio in threads + early stop on the violation target.

Same relaxation and acceptance rule as nhz_terminal_v3.plan_milp_highs_v3.  Differences:
  * `objective_target`: a solver stops at the first solution with violation >= 1e-6
    (LOG N041: solutions found early were otherwise lost to continued optimisation);
  * seeds (0, 1, 2, 3) run concurrently (HiGHS releases the GIL); the first decisive
    thread (Infeasible = excluded, or target reached) cancels the others.  The seed set
    is fixed for every instance (no per-instance choice), like the frozen baseline's
    parallel portfolio branches.
The reported upper bound is the minimum of the threads' valid dual bounds.
"""
import threading
import time

import numpy as np
import scipy.sparse as sp

from nhz_terminal import _build_plan

SEEDS = (0, 1, 2, 3)
TERMINAL_V4_VERSION = "terminal-v4.2"  # .2: interrupt the other seeds through the MIP interrupt callback


def plan_milp_portfolio(A, b, phases, K, g, cc, pad, extra, gamma_layers, lambda_layers, time_limit,
                        margin=1e-4, target=1e-6, seeds=SEEDS):
    import highspy
    t0 = time.time()
    M, rhs, nb = _build_plan(A, b, phases, K, [], gamma_layers, lambda_layers)
    M = M.tocsr()
    M = sp.hstack([M[:, :K], sp.csr_matrix((M.shape[0], 1)), M[:, K:]]).tocsr()
    blocks, rr = [M], [rhs]
    for ak, bk in extra:
        blocks.append(sp.hstack([sp.csr_matrix(ak.reshape(1, -1)), sp.csr_matrix((1, nb))])); rr.append(np.array([bk]))
    M = sp.vstack(blocks).tocsc(); rhs = np.concatenate(rr)
    n = K + 1 + nb
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = M.shape[0]
    lp.col_cost_ = np.concatenate([-g_full, np.zeros(nb)])
    lp.col_lower_ = np.concatenate([-np.ones(K + 1), np.zeros(nb)]); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32)
    lp.a_matrix_.value_ = M.data.astype(np.float64)
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * (K + 1) + [highspy.HighsVarType.kInteger] * nb
    solvers = []
    done = threading.Event()

    def _cb(callback_type, message, data_out, data_in, user_data):
        if done.is_set():
            data_in.user_interrupt = True

    for sd in (seeds if nb else seeds[:1]):
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", float(max(time_limit, 0.1))); h.setOptionValue("random_seed", int(sd))
        h.passModel(lp)
        h.setOptionValue("objective_bound", float(cc + pad + margin))
        h.setOptionValue("objective_target", float(cc - target))
        if nb:
            h.setCallback(_cb, None)
            h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipInterrupt)
        solvers.append(h)
    results = [None] * len(solvers)

    def run(i):
        solvers[i].run()
        st = solvers[i].modelStatusToString(solvers[i].getModelStatus())
        results[i] = st
        if st in ("Infeasible", "Target for objective reached", "Optimal"):
            done.set()

    th = [threading.Thread(target=run, args=(i,)) for i in range(len(solvers))]
    for x in th:
        x.start()
    for x in th:
        x.join()
    excluded = any(r == "Infeasible" for r in results)
    upper = -margin if excluded else None
    inc = None; best_v = -np.inf
    for h, st in zip(solvers, results):
        info = h.getInfo()
        if not excluded:
            dual = info.mip_dual_bound if nb else (info.objective_function_value if st == "Optimal" else None)
            if dual is not None and np.isfinite(dual) and dual > -1e300:
                u_ = -dual + cc + pad
                upper = u_ if upper is None else min(upper, u_)
        if np.isfinite(info.objective_function_value):
            v = cc - info.objective_function_value
            if v > best_v:
                sol = h.getSolution()
                if len(sol.col_value):
                    best_v = v; inc = np.array(sol.col_value[:K])
    if upper is not None and upper < -margin:
        excluded = True
    return {"upper": upper, "excluded": bool(excluded), "status": "|".join(str(r) for r in results), "binaries": nb,
            "rows": int(M.shape[0]), "wall_s": time.time() - t0, "incumbent_w": inc, "incumbent_violation": best_v}
