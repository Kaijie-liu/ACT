"""Terminal queries v3: epigraph (min-slack) objective for multi-atom unsafe disjuncts.

For an unsafe disjunct  AND_k (a_k^T Y <= b_k)  define the slack s(w) = min_k (b_k - a_k^T Y(w)).
The disjunct is empty iff  max_w s(w) < 0.  We write s via an extra latent factor
tau in [-1,1] scaled by T (an upper bound of all |slacks| from the shadow):
    maximise T tau  s.t.  T tau - (b_k - a_k^T Y_hat(w)) <= pad_k   for every atom k
(each atom row is relaxed by its rigorous coefficient/rounding pad).  For a single
atom this reduces to the v1/v2 objective.  An optimal point with T tau > 0 lies
strictly inside every atom, which makes MILP incumbents usable witnesses at the
boundary of conjunctions (LOG N034).
"""
import time

import numpy as np
import scipy.sparse as sp
import torch

from nhz_sound import gamma, optimise_nu, sound_eval
from nhz_terminal import _build_plan


def epigraph_objective(out, K, atoms):
    """Return (g, cc, pad, extra) in the v2 format but over K+1 columns (last = tau)."""
    c = out.c.reshape(-1).double(); G = out.G.reshape(K, c.numel()).double(); e = out.e.reshape(-1).double()
    rad = G.abs().sum(0); n_out = c.numel(); dev = c.device
    if len(atoms) == 1:
        a0 = torch.as_tensor(atoms[0][0], device=dev, dtype=torch.float64); b0 = float(atoms[0][1])
        g = -(G @ a0); cc = float(b0 - c @ a0)
        pad = float(gamma(n_out + 2, torch.float64) * (abs(b0) + a0.abs() @ (c.abs() + rad)) + a0.abs() @ e)
        return np.concatenate([g.cpu().numpy(), [0.0]]), cc, pad, [], None
    # T: bound on every slack magnitude over the shadow
    T = 0.0; rows = []
    for ak, bk in atoms:
        at = torch.as_tensor(ak, device=dev, dtype=torch.float64)
        sc = float(bk - c @ at); sg = (G @ at)
        T = max(T, abs(sc) + float(sg.abs().sum()))
        pk = float(gamma(n_out + 2, torch.float64) * (abs(bk) + at.abs() @ (c.abs() + rad)) + at.abs() @ e)
        rows.append((sg.cpu().numpy(), sc, pk))
    T = T * 1.01 + 1e-9
    extra = []
    for sg, sc, pk in rows:   # T tau - (sc - sg^T w) <= pk   ->   sg^T w + T tau <= sc + pk
        extra.append((np.concatenate([sg, [T]]), sc + pk))
    g = np.zeros(K + 1); g[K] = T
    return g, 0.0, 0.0, extra, T


def lp_bound_epigraph(A, b, g, cc, extra, iters, lr):
    """Rigorous LP upper bound of cc + g^T [w; tau] over rows (A padded with a zero tau column) + extra."""
    dev = A.device
    A1 = torch.cat([A, torch.zeros((A.shape[0], 1), device=dev, dtype=A.dtype)], 1)
    if extra:
        EA = torch.as_tensor(np.stack([x for x, _ in extra]), device=dev, dtype=torch.float64)
        Eb = torch.as_tensor([y for _, y in extra], device=dev, dtype=torch.float64)
        A1 = torch.cat([A1, EA]); b1 = torch.cat([b, Eb])
    else:
        b1 = b
    gt = torch.as_tensor(g, device=dev, dtype=torch.float64).unsqueeze(0)
    ct = torch.as_tensor([cc], device=dev, dtype=torch.float64)
    nu = optimise_nu(gt.float(), A1.float(), b1.float(), iters, lr, ct.float())
    ub = sound_eval(gt, ct, A1, b1, nu)
    cand = torch.sign(gt - nu.double() @ A1)[0].cpu().numpy()
    return float(ub[0]), cand


def plan_milp_highs_v3(A, b, phases, K, g, cc, pad, extra, gamma_layers, lambda_layers, time_limit, margin=1e-4):
    """Level-plan MILP over [w; tau; d]; tau is the last continuous column."""
    import highspy
    t0 = time.time()
    M, rhs, nb = _build_plan(A, b, phases, K, [], gamma_layers, lambda_layers)   # columns: K w + nb d
    # insert tau column after w (index K), shift d columns
    M = M.tocsr()
    Mw, Md = M[:, :K], M[:, K:]
    M = sp.hstack([Mw, sp.csr_matrix((M.shape[0], 1)), Md]).tocsr()
    blocks, rr = [M], [rhs]
    for ak, bk in extra:
        blocks.append(sp.hstack([sp.csr_matrix(ak.reshape(1, -1)), sp.csr_matrix((1, nb))])); rr.append(np.array([bk]))
    M = sp.vstack(blocks).tocsc(); rhs = np.concatenate(rr)
    n = K + 1 + nb
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = M.shape[0]
    lp.col_cost_ = np.concatenate([-np.asarray(g), np.zeros(nb)])
    lp.col_lower_ = np.concatenate([-np.ones(K + 1), np.zeros(nb)]); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32)
    lp.a_matrix_.value_ = M.data.astype(np.float64)
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * (K + 1) + [highspy.HighsVarType.kInteger] * nb
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(time_limit)); h.passModel(lp)
    h.setOptionValue("objective_bound", float(cc + pad + margin)); h.run()
    st = h.modelStatusToString(h.getModelStatus()); info = h.getInfo()
    excluded = st == "Infeasible"; upper = -margin if excluded else None
    dual = info.mip_dual_bound if nb else info.objective_function_value
    if not excluded and dual is not None and np.isfinite(dual) and dual > -1e300:
        upper = -dual + cc + pad; excluded = upper < -margin
    inc = None
    if np.isfinite(info.objective_function_value):
        sol = h.getSolution(); inc = np.array(sol.col_value[:K]) if len(sol.col_value) else None
    return {"upper": upper, "excluded": bool(excluded), "status": st, "binaries": nb, "rows": int(M.shape[0]),
            "wall_s": time.time() - t0, "incumbent_w": inc}
