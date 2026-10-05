"""Terminal MILP v5 = terminal v4 (seed portfolio, early stop, interrupt) + sign-aware
binaries on the last ReLU layer, extended to conjunctive (epigraph) objectives.

Proposition 1' (THEORY.md Section 9, extension).  Let L be the last ReLU layer with
phases; its fresh factor eta_j occurs only in y_j's value map, its two LP rows, its two
binary rows, the objective (coefficient g_j) and the extra epigraph rows a_k^T [w;tau] <= b_k
(coefficient a_kj).  If g_j <= 0 and a_kj >= 0 for every extra row, then lowering eta_j
never lowers the objective and never violates an extra row, so an optimum of the plan
without unit j's binary rows can be moved to y_j = ReLU(x_j), which satisfies the binary
rows with d_j = [x_j >= 0].  Dropping those binaries therefore leaves the MILP optimum and
its feasibility (hence the Infeasible-under-cutoff verdict) unchanged.  For a single atom
there are no extra rows and the rule is Proposition 1.

The selection reads only signs of specification coefficients in latent coordinates
(observable structure); no LP state, margin, family or instance identity.
"""
import threading
import time

import numpy as np
import scipy.sparse as sp

SEEDS = (0, 1, 2, 3)
TERMINAL_V5_VERSION = "terminal-v5.0"


def needs_binary(p, g_full, extra):
    m = int(p["idx"].numel()); e0 = int(p["eta0"])
    need = np.asarray(g_full)[e0:e0 + m] > 0
    for ak, _ in extra:
        need |= np.asarray(ak)[e0:e0 + m] < 0
    return need


def build_plan_v5(A, b, phases, K, gamma_layers, lambda_layers, g_full, extra, sign_aware=True):
    """Rows over columns [w (K) | tau (1) | d (nb)]."""
    nz = [p for p in phases if int(p["idx"].numel())]
    lam = nz[-lambda_layers:] if lambda_layers > 0 else []
    gam = nz[-gamma_layers:] if gamma_layers > 0 else []
    An = A.double().cpu().numpy(); bn = b.double().cpu().numpy()
    keep = np.zeros(An.shape[0], bool)
    for p in lam:
        m = int(p["idx"].numel()); keep[p["row0"]:p["row0"] + 2 * m] = True
    sel = []
    for j, p in enumerate(gam):
        m = int(p["idx"].numel())
        last = (p is nz[-1])
        sel.append(np.flatnonzero(needs_binary(p, g_full, extra)) if (sign_aware and last) else np.arange(m))
    nb = int(sum(s.size for s in sel))
    blocks = [sp.hstack([sp.csr_matrix(An[keep]), sp.csr_matrix((int(keep.sum()), 1 + nb))])]
    rhs = [bn[keep]]; col = 0
    for p, s in zip(gam, sel):
        if s.size == 0:
            continue
        m = s.size
        def pad(t):
            z = np.zeros((m, K)); t = t.double().cpu().numpy()[s]; z[:, :t.shape[1]] = t; return z
        gx, gy = pad(p["gx"]), pad(p["gy"])
        cx, cy, ex, ey, l, u = (p[k].double().cpu().numpy()[s] for k in ("cx", "cy", "ex", "ey", "l", "u"))
        r = np.arange(m); zt = sp.csr_matrix((m, 1))
        blocks.append(sp.hstack([sp.csr_matrix(gy), zt, sp.csr_matrix((-u, (r, col + r)), shape=(m, nb))])); rhs.append(-cy + ey)
        blocks.append(sp.hstack([sp.csr_matrix(gy - gx), zt, sp.csr_matrix((-l, (r, col + r)), shape=(m, nb))]))
        rhs.append(cx - cy - l + ex + ey)
        col += m
    for ak, bk in extra:
        blocks.append(sp.hstack([sp.csr_matrix(np.asarray(ak).reshape(1, -1)), sp.csr_matrix((1, nb))])); rhs.append(np.array([bk]))
    return sp.vstack(blocks).tocsc(), np.concatenate(rhs), nb, sum(int(p["idx"].numel()) for p in gam)


def run_portfolio(M, rhs, nb, K, g_full, cc, pad, time_limit, margin=1e-4, target=1e-6, seeds=SEEDS):
    import highspy
    t0 = time.time()
    n = K + 1 + nb
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = M.shape[0]
    lp.col_cost_ = np.concatenate([-np.asarray(g_full, dtype=np.float64), np.zeros(nb)])
    lp.col_lower_ = np.concatenate([-np.ones(K + 1), np.zeros(nb)]); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32)
    lp.a_matrix_.value_ = M.data.astype(np.float64)
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * (K + 1) + [highspy.HighsVarType.kInteger] * nb
    solvers = []; done = threading.Event()

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


def plan_milp_v5(A, b, phases, K, g, cc, pad, extra, gamma_layers, lambda_layers, time_limit, margin=1e-4,
                 target=1e-6, seeds=SEEDS, sign_aware=True):
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M, rhs, nb, n_units = build_plan_v5(A, b, phases, K, gamma_layers, lambda_layers, g_full, extra, sign_aware)
    r = run_portfolio(M, rhs, nb, K, g_full, cc, pad, time_limit, margin, target, seeds)
    r["gamma_units"] = n_units
    return r
