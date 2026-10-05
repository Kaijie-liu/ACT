"""Terminal MILP v7.11 (copy of v7.10 + numerical safety width NUM_TOL = 1e-5 on every unit row, LOG N112; the frozen nhz_terminal_v7.py of N039 v14 is unchanged).

Terminal MILP v7: disaggregated (Balas) encoding of the gamma units, sign/monotone-aware
integrality, seed portfolio with early stop (terminal v5 runner semantics).

Columns: w (K latent) | tau | x_k (explicit pre-activation of every gamma unit) |
         x0_j, x1_j, d_j (binary units only).
Rows:
  lambda-only layers (not gamma): their two dense LP rows, as in every earlier plan;
  every gamma unit k:   x_k - gx_k^T w = cx_k                                 (equality)
     y_k := lam_k x_k + mu_k (1 + eta_k)   with  |y_true - y_k| <= ey_k + delta_k
     delta_k = sum_i |gy_ki - lam_k gx_ki| + |cy_k - lam_k cx_k - mu_k|  (float32 storage gap
               between the stored value map gy and the aligned formula; padded)
  non-binary gamma unit:  -y_k <= ey+delta,  x_k - y_k <= ex+ey+delta
  binary unit:            x_k = x0 + x1 (+-ex),  y_k = x1 (+-(ey+delta)),
                          x0 >= l (1-d),  x1 <= u d,  x0 in [l,0],  x1 in [0,u].
The projection onto (w, d) contains the exact concretisation (every real point with
y = ReLU(x) and d = [x>=0] satisfies all rows), so exclusion is sound; integrality is
skipped only for units selected by nhz_monotone (Proposition 2, = Proposition 1' on the
last layer), which only relaxes."""
import numpy as np
import scipy.sparse as sp

from nhz_monotone import _lam_mu, monotone_units
from nhz_sound import gamma
from nhz_terminal_v5 import SEEDS, needs_binary
import torch

TERMINAL_V7_VERSION = "terminal-v7.11"  # .11: NUM_TOL safety width on unit rows and piece bounds (LOG N112); .10: empty plan (no rows) handled (LOG N103); .9: optional MIP start on integer columns; .8: dual bounds only from valid non-error statuses; tiny coefficients folded into row bounds (LOG N085); .7: portfolio = HiGHS seeds 0-2 + SCIP (forked child) when nnz <= 5e5; .6: fail closed on non-finite plan data; .5: presolve off above 5e5 nonzeros; .4: wall-clock watchdog through simplex/IPM/MIP interrupt callbacks; .3: last-layer sign rule only; build time charged to the stage;  # .1: target=None disables the early stop; .2: vectorised builder, monotone analysis only for small multi-layer plans


PRESOLVE_MAX_NNZ = 500_000
NUM_TOL = 1e-5   # v7.11: numerical safety width added to every unit row (LOG N112); a relaxation
MONO_CELLS = 2e7   # resource guard for the multi-layer monotone analysis (dense unit-coordinate matrices)


def _need_masks(phases, nz, gam, K, g_full, extra, sign_aware):
    if not (sign_aware and gam):
        return [np.ones(int(p["idx"].numel()), bool) for p in gam]
    last_need = needs_binary(nz[-1], g_full, extra)              # Proposition 1' (= Prop. 2 on the last layer)
    # v7.3: the multi-layer analysis (Proposition 2) is not used: it never selected more than the
    # last-layer rule (LOG N047) and its CPU cost broke the stage deadline on Cora (LOG N059)
    return [np.ones(int(p["idx"].numel()), bool) for p in gam[:-1]] + [last_need]


def build_plan_v7(A, b, phases, K, gamma_layers, lambda_layers, g_full, extra, sign_aware=True):
    nz = [p for p in phases if int(p["idx"].numel())]
    lam_l = nz[-lambda_layers:] if lambda_layers > 0 else []
    gam = nz[-gamma_layers:] if gamma_layers > 0 else []
    gam_ids = set(id(p) for p in gam)
    An = A.double().cpu().numpy(); bn = b.double().cpu().numpy()
    keep = np.zeros(An.shape[0], bool)
    for p in lam_l:
        if id(p) not in gam_ids:
            m = int(p["idx"].numel()); keep[p["row0"]:p["row0"] + 2 * m] = True
    need = _need_masks(phases, nz, gam, K, g_full, extra, sign_aware)
    m_tot = sum(int(p["idx"].numel()) for p in gam); nb = int(sum(int(n_.sum()) for n_ in need))
    nW = K + 1; X0 = nW; P0 = X0 + m_tot; Q0 = P0 + nb; D0 = Q0 + nb; n = D0 + nb
    blocks, LO, HI = [], [], []
    Ak = An[keep]
    if Ak.shape[0]:
        blocks.append(sp.hstack([sp.csr_matrix(Ak), sp.csr_matrix((Ak.shape[0], n - K))]).tocsr())
        LO.append(np.full(Ak.shape[0], -np.inf)); HI.append(bn[keep])
    xl, xu = [], []; x0l, x0u, x1l, x1u = [], [], [], []
    xoff = 0; boff = 0; g2 = float(gamma(4, torch.float64))
    for p, nd in zip(gam, need):
        m = int(p["idx"].numel()); e0 = int(p["eta0"]); r = np.arange(m)
        lam, mu = (v.cpu().numpy() for v in _lam_mu(p))
        gx = p["gx"].double().cpu().numpy(); gy = p["gy"].double().cpu().numpy()
        cx = p["cx"].double().cpu().numpy(); cy = p["cy"].double().cpu().numpy()
        ex = p["ex"].double().cpu().numpy(); ey = p["ey"].double().cpu().numpy()
        l = p["l"].double().cpu().numpy(); u = p["u"].double().cpu().numpy()
        kx = gx.shape[1]
        gyz = gy.copy(); gyz[r, e0 + r] -= mu
        delta = (np.abs(gyz[:, :kx] - lam[:, None] * gx).sum(1) + np.abs(gyz[:, kx:]).sum(1)
                 + np.abs(cy - lam * cx - mu)) * (1 + g2) + g2 * (np.abs(cy) + np.abs(gy).sum(1)) + 1e-300
        tol = ey + delta
        xc = X0 + xoff + r; ec = e0 + r
        # x equality: x_k - gx_k^T w = cx_k
        Gx = sp.csr_matrix(gx); Gx = sp.hstack([-Gx, sp.csr_matrix((m, n - kx))]).tocsr()
        Gx = Gx + sp.csr_matrix((np.ones(m), (r, xc)), shape=(m, n))
        blocks.append(Gx); LO.append(cx); HI.append(cx)
        xl.append(l - ex - 1.0); xu.append(u + ex + 1.0)
        nbm = ~nd; ib = np.flatnonzero(nd); inb = np.flatnonzero(nbm)
        tol = tol + NUM_TOL; exn = ex + NUM_TOL          # v7.11: numerical safety width (relaxation)
        if inb.size:
            k = inb; rr = np.arange(k.size)
            def two(cx_, ce_):
                return sp.csr_matrix((np.concatenate([cx_, ce_]), (np.concatenate([rr, rr]), np.concatenate([xc[k], ec[k]]))), shape=(k.size, n))
            blocks.append(two(-lam[k], -mu[k])); LO.append(np.full(k.size, -np.inf)); HI.append(mu[k] + tol[k])
            blocks.append(two(1 - lam[k], -mu[k])); LO.append(np.full(k.size, -np.inf)); HI.append(mu[k] + exn[k] + tol[k])
        if ib.size:
            k = ib; rr = np.arange(k.size); jj = boff + rr
            def mk(cols, vals):
                return sp.csr_matrix((np.concatenate(vals), (np.concatenate([rr] * len(cols)), np.concatenate(cols))), shape=(k.size, n))
            blocks.append(mk([xc[k], P0 + jj, Q0 + jj], [np.ones(k.size), -np.ones(k.size), -np.ones(k.size)])); LO.append(-exn[k]); HI.append(exn[k])
            blocks.append(mk([xc[k], ec[k], Q0 + jj], [lam[k], mu[k], -np.ones(k.size)])); LO.append(-mu[k] - tol[k]); HI.append(-mu[k] + tol[k])
            ln = l[k] - NUM_TOL; un = u[k] + NUM_TOL
            blocks.append(mk([P0 + jj, D0 + jj], [np.ones(k.size), ln])); LO.append(ln); HI.append(np.full(k.size, np.inf))
            blocks.append(mk([Q0 + jj, D0 + jj], [np.ones(k.size), -un])); LO.append(np.full(k.size, -np.inf)); HI.append(np.zeros(k.size))
            x0l.append(ln); x0u.append(np.zeros(k.size)); x1l.append(np.zeros(k.size)); x1u.append(un)
            boff += k.size
        xoff += m
    for ak, bk_ in extra:
        a = np.asarray(ak).reshape(1, -1)
        blocks.append(sp.hstack([sp.csr_matrix(a), sp.csr_matrix((1, n - a.shape[1]))]).tocsr()); LO.append(np.array([-np.inf])); HI.append(np.array([bk_]))
    if not blocks:          # v7.10: no unstable unit, no lambda row, no atom row -> the plan is the latent box alone (LOG N103)
        M = sp.csc_matrix((0, n)); LO = [np.zeros(0)]; HI = [np.zeros(0)]
    else:
        M = sp.vstack(blocks).tocsc()
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)
    col_lo = np.concatenate([-np.ones(nW), cat(xl), cat(x0l), cat(x1l), np.zeros(nb)]).astype(np.float64)
    col_hi = np.concatenate([np.ones(nW), cat(xu), cat(x0u), cat(x1u), np.ones(nb)]).astype(np.float64)
    cost = np.concatenate([-g_full, np.zeros(m_tot + 3 * nb)])
    integ = np.concatenate([np.zeros(nW + m_tot + 2 * nb), np.ones(nb)]).astype(bool)
    return M, np.concatenate(LO), np.concatenate(HI), col_lo, col_hi, cost, integ, nb, m_tot



TINY_COEF = 1e-9


def _fold_tiny(M, lo, hi, col_lo, col_hi):
    """Remove matrix entries with |a| <= TINY_COEF and relax the row bounds by their worst-case
    contribution over the column bounds:  lo - max(a v) <= sum_kept <= hi - min(a v)."""
    import scipy.sparse as _sp
    C = M.tocoo()
    small = np.abs(C.data) <= TINY_COEF
    if not small.any():
        return M, lo, hi
    r, c, a = C.row[small], C.col[small], C.data[small]
    v1 = a * col_lo[c]; v2 = a * col_hi[c]
    cmin = np.minimum(v1, v2); cmax = np.maximum(v1, v2)
    dmin = np.bincount(r, weights=cmin, minlength=M.shape[0]); dmax = np.bincount(r, weights=cmax, minlength=M.shape[0])
    lo2 = np.where(np.isfinite(lo), lo - dmax - 1e-300, lo); hi2 = np.where(np.isfinite(hi), hi - dmin + 1e-300, hi)
    keep = ~small
    M2 = _sp.csc_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=M.shape)
    return M2, lo2, hi2


def _scip_child(conn, M, lo, hi, col_lo, col_hi, cost, integ, K, cc, pad, time_limit, margin):
    try:
        from nhz_scip_backend import run_scip
        r = run_scip(M, lo, hi, col_lo, col_hi, cost, integ, K, cc, pad, time_limit, margin)
        conn.send({k: r[k] for k in ("upper", "excluded", "status", "incumbent_w", "incumbent_violation")})
    except Exception as ex:  # fail closed: report nothing decisive
        conn.send({"upper": None, "excluded": False, "status": f"scip-error:{type(ex).__name__}", "incumbent_w": None,
                   "incumbent_violation": -np.inf})
    finally:
        conn.close()


SCIP_MAX_NNZ = 500_000


def run_portfolio_v7(M, lo, hi, col_lo, col_hi, cost, integ, K, cc, pad, time_limit, margin=1e-4, target=1e-6,
                     seeds=SEEDS, start_int=None):
    import threading, time
    import highspy
    t0 = time.time(); nb = int(integ.sum())
    # v7.6: fail closed on non-finite plan data (never an exclusion)
    if not (np.isfinite(M.data).all() and not np.isnan(lo).any() and not np.isnan(hi).any()
            and np.isfinite(col_lo).all() and np.isfinite(col_hi).all() and np.isfinite(cost).all()):
        return {"upper": None, "excluded": False, "status": "nonfinite-plan", "binaries": nb, "rows": int(M.shape[0]),
                "wall_s": time.time() - t0, "incumbent_w": None, "incumbent_violation": -np.inf}
    M, lo, hi = _fold_tiny(M, lo, hi, col_lo, col_hi)        # v7.8: solvers drop |a| <= 1e-9 silently (LOG N085)
    lp = highspy.HighsLp(); lp.num_col_ = M.shape[1]; lp.num_row_ = M.shape[0]
    lp.col_cost_ = cost; lp.col_lower_ = col_lo; lp.col_upper_ = col_hi
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -highspy.kHighsInf)
    lp.row_upper_ = np.where(np.isfinite(hi), hi, highspy.kHighsInf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32)
    lp.a_matrix_.value_ = M.data.astype(np.float64)
    if nb:
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
    solvers = []; done = threading.Event()
    hard = t0 + float(max(time_limit, 0.1)) + 0.5            # v7.4: wall-clock watchdog (root LP / IPM included)

    def _cb(callback_type, message, data_out, data_in, user_data):
        if done.is_set() or time.time() > hard:
            data_in.user_interrupt = True

    use_scip = bool(nb) and M.nnz <= SCIP_MAX_NNZ and len(seeds) >= 2
    hseeds = (seeds[:-1] if use_scip else seeds) if nb else seeds[:1]
    for sd in hseeds:
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", float(max(time_limit, 0.1))); h.setOptionValue("random_seed", int(sd))
        if M.nnz > PRESOLVE_MAX_NNZ:
            h.setOptionValue("presolve", "off")     # v7.5: HiGHS presolve ignores the time limit on large plans (LOG N059)
        h.passModel(lp)
        h.setOptionValue("objective_bound", float(cc + pad + margin))
        if target is not None:
            h.setOptionValue("objective_target", float(cc - target))
        if start_int is not None and nb and len(start_int) == nb:
            # v7.9: MIP start on the integer columns only (a hint; HiGHS checks and completes it)
            iidx = np.flatnonzero(integ).astype(np.int32)
            try:
                h.setSolution(int(iidx.size), iidx, np.asarray(start_int, dtype=np.float64))
            except Exception:
                pass
        h.setCallback(_cb, None)
        for ct in (highspy.cb.HighsCallbackType.kCallbackSimplexInterrupt, highspy.cb.HighsCallbackType.kCallbackIpmInterrupt):
            h.startCallback(ct)
        if nb:
            h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipInterrupt)
        solvers.append(h)
    results = [None] * len(solvers)

    def run(i):
        try:
            solvers[i].run()
            st = solvers[i].modelStatusToString(solvers[i].getModelStatus())
        except Exception as ex:          # v7.8: an exception is an error status, never a bound
            st = f"error:{type(ex).__name__}"
        results[i] = st
        if st in ("Infeasible", "Target for objective reached", "Optimal"):
            done.set()

    th = [threading.Thread(target=run, args=(i,)) for i in range(len(solvers))]
    proc = None; parent = None; scip_res = None
    if use_scip:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        parent, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_scip_child, args=(child_conn, M, lo, hi, col_lo, col_hi, cost, integ, K, cc, pad,
                                                     float(max(time_limit, 0.1)), margin), daemon=True)
        proc.start(); child_conn.close()
    for x in th:
        x.start()
    while True:
        alive = any(x.is_alive() for x in th)
        if proc is not None and scip_res is None and parent.poll(0.05):
            try:
                scip_res = parent.recv()
            except EOFError:
                scip_res = {"upper": None, "excluded": False, "status": "scip-eof", "incumbent_w": None, "incumbent_violation": -np.inf}
            if scip_res.get("excluded"):
                done.set()
        if proc is not None and scip_res is None and not proc.is_alive() and not parent.poll(0):
            scip_res = {"upper": None, "excluded": False, "status": "scip-died", "incumbent_w": None, "incumbent_violation": -np.inf}
        scip_open = proc is not None and scip_res is None
        if not alive and (not scip_open or done.is_set() or time.time() > hard):
            break
        if not alive and scip_open:
            time.sleep(0.05)
        elif proc is None:
            for x in th:
                x.join()
            break
        else:
            time.sleep(0.02)
    if proc is not None:
        if proc.is_alive():
            proc.terminate(); proc.join(1.0)
            if proc.is_alive():
                proc.kill(); proc.join(1.0)
    for x in th:
        x.join()
    excluded = any(r == "Infeasible" for r in results) or bool(scip_res and scip_res.get("excluded")
                                                                and scip_res.get("upper") is not None
                                                                and np.isfinite(scip_res["upper"]))
    upper = -margin if excluded else None
    inc = None; best_v = -np.inf
    BOUND_OK = ("Optimal", "Time limit reached", "Interrupted by user", "Target for objective reached")
    for h, st in zip(solvers, results):
        info = h.getInfo()
        if not excluded and st in BOUND_OK and bool(getattr(info, "valid", True)):
            # v7.8: a dual bound is used only from a valid info block of a non-error status (LOG N085)
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
    if scip_res is not None and not excluded:
        su = scip_res.get("upper")
        if su is not None and np.isfinite(su):
            upper = su if upper is None else min(upper, su)
    if scip_res is not None and scip_res.get("incumbent_w") is not None and scip_res.get("incumbent_violation", -np.inf) > best_v:
        best_v = scip_res["incumbent_violation"]; inc = np.asarray(scip_res["incumbent_w"])
    if upper is not None and upper < -margin:
        excluded = True
    results = list(results) + ([scip_res.get("status")] if scip_res is not None else (["scip-none"] if use_scip else []))
    return {"upper": upper, "excluded": bool(excluded), "status": "|".join(str(r) for r in results), "binaries": nb,
            "rows": int(M.shape[0]), "wall_s": time.time() - t0, "incumbent_w": inc, "incumbent_violation": best_v}


def plan_milp_v7(A, b, phases, K, g, cc, pad, extra, gamma_layers, lambda_layers, time_limit, margin=1e-4,
                 target=1e-6, seeds=SEEDS, sign_aware=True):
    import time as _t
    t0 = _t.time()
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M, lo, hi, clo, chi, cost, integ, nb, m_tot = build_plan_v7(A, b, phases, K, gamma_layers, lambda_layers, g_full,
                                                                 extra, sign_aware)
    left = time_limit - (_t.time() - t0)                      # v7.3: build time is charged to the stage
    r = run_portfolio_v7(M, lo, hi, clo, chi, cost, integ, K, cc, pad, max(left, 0.1), margin, target, seeds)
    r["build_s"] = _t.time() - t0 - r["wall_s"]
    r["gamma_units"] = m_tot
    return r
