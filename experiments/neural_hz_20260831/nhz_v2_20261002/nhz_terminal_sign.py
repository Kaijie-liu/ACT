"""Sign-aware selective exactness for the last ReLU layer (THEORY.md Section 9).

For the last ReLU layer with phases, the violation objective is
  v(w) = g^T w + cc,  and eta_k (the fresh factor of unit k) appears only in
  y_k's value map, its two LP rows and its two binary rows.
If g[eta_k] <= 0, maximising v pushes y_k onto its lower envelope max(0, x_k),
which the LP rows already describe exactly, so unit k needs no binary.  Only
units with g[eta_k] > 0 keep integer d_k.  The selection reads only the
objective coefficients of the specification (no LP state, no margin).
"""
import numpy as np
import scipy.sparse as sp

from nhz_terminal import _build_plan  # noqa: F401  (for reference)


def build_sign_plan(A, b, phases, K, g, extra):
    """All LP rows (lambda everywhere) + integer rows only for positive-coefficient
    units of the last ReLU layer with phases."""
    nz = [p for p in phases if int(p["idx"].numel())]
    last = nz[-1]
    eta0 = last["eta0"]; m = int(last["idx"].numel())
    coef = np.asarray(g)[eta0:eta0 + m]
    keep = np.flatnonzero(coef > 0)
    An = A.double().cpu().numpy(); bn = b.double().cpu().numpy()
    nb = int(keep.size)
    blocks = [sp.hstack([sp.csr_matrix(An), sp.csr_matrix((An.shape[0], nb))])]; rhs = [bn]
    gx = np.zeros((m, K)); g0 = last["gx"].double().cpu().numpy(); gx[:, :g0.shape[1]] = g0
    gy = np.zeros((m, K)); g1 = last["gy"].double().cpu().numpy(); gy[:, :g1.shape[1]] = g1
    cx = last["cx"].double().cpu().numpy(); cy = last["cy"].double().cpu().numpy()
    ex = last["ex"].double().cpu().numpy(); ey = last["ey"].double().cpu().numpy()
    l = last["l"].double().cpu().numpy(); u = last["u"].double().cpu().numpy()
    gx, gy, cx, cy, ex, ey, l, u = (v[keep] for v in (gx, gy, cx, cy, ex, ey, l, u))
    r = np.arange(nb)
    blocks.append(sp.hstack([sp.csr_matrix(gy), sp.csr_matrix((-u, (r, r)), shape=(nb, nb))])); rhs.append(-cy + ey)
    blocks.append(sp.hstack([sp.csr_matrix(gy - gx), sp.csr_matrix((-l, (r, r)), shape=(nb, nb))])); rhs.append(cx - cy - l + ex + ey)
    for ak, bk in extra:
        blocks.append(sp.hstack([sp.csr_matrix(ak.reshape(1, -1)), sp.csr_matrix((1, nb))])); rhs.append(np.array([bk]))
    return sp.vstack(blocks).tocsc(), np.concatenate(rhs), nb, m


def sign_milp_highs(A, b, phases, K, g, cc, pad, extra, time_limit, margin=1e-4, threads=1):
    import time
    import highspy
    t0 = time.time()
    M, rhs, nb, m_all = build_sign_plan(A, b, phases, K, g, extra)
    n = K + nb
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = M.shape[0]
    lp.col_cost_ = np.concatenate([-np.asarray(g), np.zeros(nb)])
    lp.col_lower_ = np.concatenate([-np.ones(K), np.zeros(nb)]); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32)
    lp.a_matrix_.value_ = M.data.astype(np.float64)
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * K + [highspy.HighsVarType.kInteger] * nb
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", int(threads)); h.setOptionValue("time_limit", float(time_limit))
    h.passModel(lp); h.setOptionValue("objective_bound", float(cc + pad + margin)); h.run()
    st = h.modelStatusToString(h.getModelStatus()); info = h.getInfo()
    excluded = st == "Infeasible"; upper = -margin if excluded else None
    if not excluded and nb and np.isfinite(info.mip_dual_bound) and info.mip_dual_bound > -1e300:
        upper = -info.mip_dual_bound + cc + pad; excluded = upper < -margin
    inc = None
    if np.isfinite(info.objective_function_value):
        sol = h.getSolution(); inc = np.array(sol.col_value[:K]) if len(sol.col_value) else None
    return {"upper": upper, "excluded": bool(excluded), "status": st, "binaries": nb, "last_layer_units": m_all,
            "wall_s": time.time() - t0, "incumbent_w": inc}
