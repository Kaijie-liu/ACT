"""Rigorous terminal queries for the sound engine (n004.2+): LP and level-plan MILP.

`sound_objective` builds the violation objective of one unsafe disjunct with its
rigorous coefficient pad.  `plan_milp` solves a level-plan relaxation of the
exact state with HiGHS:

  * gamma layers (last `gamma_layers` ReLU layers with phases): integer d and
    the binary rows, relaxed by the stored rounding radii
        y_hat <= u d + e_y,      y_hat - x_hat <= -l (1 - d) + e_x + e_y ;
  * lambda layers (last `lambda_layers` layers): their two LP rows;
  * remaining layers: eta box only (sigma level).

Every plan is a relaxation of the exact concretisation (THEORY.md Lemmas 2, 4),
so its optimum bounds the true violation.  The returned upper bound is the
HiGHS MILP dual bound plus the coefficient pad; a disjunct is reported excluded
only if this is below -`margin` (default 1e-4), to absorb solver tolerances.
This is the same solver standard as the frozen baseline (HiGHS MILP).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from nhz_sound import gamma


def sound_objective(out, K: int, atoms):
    c = out.c.reshape(-1).double(); G = out.G.reshape(K, c.numel()).double(); e = out.e.reshape(-1).double()
    rad = G.abs().sum(0); n_out = c.numel(); dev = c.device
    a0 = torch.as_tensor(atoms[0][0], device=dev, dtype=torch.float64); b0 = float(atoms[0][1])
    g = -(G @ a0); cc = float(b0 - c @ a0)
    pad = float(gamma(n_out + 2, torch.float64) * (abs(b0) + a0.abs() @ (c.abs() + rad)) + a0.abs() @ e)
    extra = []
    for ak, bk in atoms[1:]:
        at = torch.as_tensor(ak, device=dev, dtype=torch.float64)
        pk = float(gamma(n_out + 2, torch.float64) * (abs(bk) + at.abs() @ (c.abs() + rad)) + at.abs() @ e)
        extra.append(((G @ at).cpu().numpy(), float(bk - c @ at + pk)))
    return g.cpu().numpy(), cc, pad, extra


def plan_milp(A: torch.Tensor, b: torch.Tensor, phases: List[dict], K: int, g: np.ndarray, cc: float, pad: float,
              extra, gamma_layers: int, lambda_layers: int, time_limit: float, margin: float = 1e-4):
    t0 = time.time()
    nz = [p for p in phases if int(p["idx"].numel())]
    lam = nz[-lambda_layers:] if lambda_layers > 0 else []
    gam = nz[-gamma_layers:] if gamma_layers > 0 else []
    An = A.double().cpu().numpy(); bn = b.double().cpu().numpy()
    keep = np.zeros(An.shape[0], bool)
    for p in lam:
        m = int(p["idx"].numel()); keep[p["row0"]:p["row0"] + 2 * m] = True
    nb = sum(int(p["idx"].numel()) for p in gam)
    blocks = [sp.hstack([sp.csr_matrix(An[keep]), sp.csr_matrix((int(keep.sum()), nb))])]
    rhs = [bn[keep]]
    col = 0
    for p in gam:
        m = int(p["idx"].numel())
        gx = np.zeros((m, K)); g0 = p["gx"].double().cpu().numpy(); gx[:, :g0.shape[1]] = g0
        gy = np.zeros((m, K)); g1 = p["gy"].double().cpu().numpy(); gy[:, :g1.shape[1]] = g1
        cx = p["cx"].double().cpu().numpy(); cy = p["cy"].double().cpu().numpy()
        ex = p["ex"].double().cpu().numpy(); ey = p["ey"].double().cpu().numpy()
        l = p["l"].double().cpu().numpy(); u = p["u"].double().cpu().numpy()
        r = np.arange(m)
        D1 = sp.csr_matrix((-u, (r, col + r)), shape=(m, nb))
        D2 = sp.csr_matrix((-l, (r, col + r)), shape=(m, nb))
        blocks.append(sp.hstack([sp.csr_matrix(gy), D1])); rhs.append(-cy + ey)                    # y <= u d (+e)
        blocks.append(sp.hstack([sp.csr_matrix(gy - gx), D2])); rhs.append(cx - cy - l + ex + ey)  # y - x <= -l(1-d) (+e)
        col += m
    for ak, bk in extra:
        blocks.append(sp.hstack([sp.csr_matrix(ak.reshape(1, -1)), sp.csr_matrix((1, nb))])); rhs.append(np.array([bk]))
    AA = sp.vstack(blocks).tocsr(); BB = np.concatenate(rhs)
    obj = np.concatenate([-g, np.zeros(nb)])
    integ = np.concatenate([np.zeros(K), np.ones(nb)])
    bnds = Bounds(np.concatenate([-np.ones(K), np.zeros(nb)]), np.ones(K + nb))
    res = milp(obj, constraints=LinearConstraint(AA, -np.inf, BB), integrality=integ, bounds=bnds,
               options={"time_limit": float(time_limit), "disp": False})
    if nb == 0:
        dual = res.fun if res.status == 0 else None
    else:
        dual = getattr(res, "mip_dual_bound", None)
    upper = (-dual + cc + pad) if dual is not None and np.isfinite(dual) else None
    inc_w = res.x[:K] if res.x is not None else None
    return {"upper": upper, "excluded": bool(upper is not None and upper < -margin), "status": int(res.status),
            "binaries": nb, "rows": int(AA.shape[0]), "wall_s": time.time() - t0, "incumbent_w": inc_w,
            "incumbent": (-res.fun + cc) if res.fun is not None else None}


def _build_plan(A, b, phases, K, extra, gamma_layers, lambda_layers):
    nz = [p for p in phases if int(p["idx"].numel())]
    lam = nz[-lambda_layers:] if lambda_layers > 0 else []
    gam = nz[-gamma_layers:] if gamma_layers > 0 else []
    An = A.double().cpu().numpy(); bn = b.double().cpu().numpy()
    keep = np.zeros(An.shape[0], bool)
    for p in lam:
        m = int(p["idx"].numel()); keep[p["row0"]:p["row0"] + 2 * m] = True
    nb = sum(int(p["idx"].numel()) for p in gam)
    blocks = [sp.hstack([sp.csr_matrix(An[keep]), sp.csr_matrix((int(keep.sum()), nb))])]
    rhs = [bn[keep]]; col = 0
    for p in gam:
        m = int(p["idx"].numel())
        gx = np.zeros((m, K)); g0 = p["gx"].double().cpu().numpy(); gx[:, :g0.shape[1]] = g0
        gy = np.zeros((m, K)); g1 = p["gy"].double().cpu().numpy(); gy[:, :g1.shape[1]] = g1
        cx = p["cx"].double().cpu().numpy(); cy = p["cy"].double().cpu().numpy()
        ex = p["ex"].double().cpu().numpy(); ey = p["ey"].double().cpu().numpy()
        l = p["l"].double().cpu().numpy(); u = p["u"].double().cpu().numpy()
        r = np.arange(m)
        blocks.append(sp.hstack([sp.csr_matrix(gy), sp.csr_matrix((-u, (r, col + r)), shape=(m, nb))])); rhs.append(-cy + ey)
        blocks.append(sp.hstack([sp.csr_matrix(gy - gx), sp.csr_matrix((-l, (r, col + r)), shape=(m, nb))])); rhs.append(cx - cy - l + ex + ey)
        col += m
    for ak, bk in extra:
        blocks.append(sp.hstack([sp.csr_matrix(ak.reshape(1, -1)), sp.csr_matrix((1, nb))])); rhs.append(np.array([bk]))
    return sp.vstack(blocks).tocsc(), np.concatenate(rhs), nb


def plan_milp_highs(A, b, phases, K, g, cc, pad, extra, gamma_layers, lambda_layers, time_limit, margin=1e-4,
                    threads=1):
    """Same relaxation as plan_milp, solved with highspy and an objective cutoff:
    minimise f = -g^T w; the disjunct is excluded iff no point has
    f < cc + pad + margin (HiGHS reports Infeasible under objective_bound)."""
    import highspy
    t0 = time.time()
    M, rhs, nb = _build_plan(A, b, phases, K, extra, gamma_layers, lambda_layers)
    n = K + nb
    lp = highspy.HighsLp()
    lp.num_col_ = n; lp.num_row_ = M.shape[0]
    lp.col_cost_ = np.concatenate([-g, np.zeros(nb)])
    lp.col_lower_ = np.concatenate([-np.ones(K), np.zeros(nb)]); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32)
    lp.a_matrix_.value_ = M.data.astype(np.float64)
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * K + [highspy.HighsVarType.kInteger] * nb
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", int(threads)); h.setOptionValue("time_limit", float(time_limit))
    h.passModel(lp)
    cutoff = cc + pad + margin
    h.setOptionValue("objective_bound", float(cutoff))
    h.run()
    st = h.modelStatusToString(h.getModelStatus()); info = h.getInfo()
    excluded = st == "Infeasible"
    dual = info.mip_dual_bound if nb else info.objective_function_value
    upper = None
    if excluded:
        upper = -margin            # certified below the threshold (exact value not computed)
    elif dual is not None and np.isfinite(dual) and dual > -1e300:
        upper = -dual + cc + pad
        excluded = upper < -margin
    inc = None
    if st == "Optimal" or (info.objective_function_value is not None and np.isfinite(info.objective_function_value)):
        sol = h.getSolution()
        inc = np.array(sol.col_value[:K]) if len(sol.col_value) else None
    return {"upper": upper, "excluded": bool(excluded), "status": st, "binaries": nb, "rows": int(M.shape[0]),
            "wall_s": time.time() - t0, "incumbent_w": inc}


def sound_lp_batch(out, rows, K: int, disjuncts, iters: int, lr: float, chunk: int = 256):
    """Rigorous LP upper bounds for every unsafe disjunct (batched for single-atom
    disjuncts), plus box-maximiser witness candidates.  Returns a list of dicts
    with keys upper, g (numpy), cc, pad, extra, cand (numpy, w-space)."""
    from nhz_sound import optimise_nu, sound_eval
    dev = out.c.device
    A, b = rows.dense(K); A = A.to(dev); b = b.to(dev)
    A32, b32 = A.float(), b.float()
    res = [None] * len(disjuncts)
    objs = [sound_objective(out, K, atoms) for atoms in disjuncts]
    single = [i for i, d in enumerate(disjuncts) if len(d) == 1]
    for s in range(0, len(single), chunk):
        ids = single[s:s + chunk]
        g = torch.as_tensor(np.stack([objs[i][0] for i in ids]), device=dev, dtype=torch.float64)
        cc = torch.as_tensor([objs[i][1] for i in ids], device=dev, dtype=torch.float64)
        nu = optimise_nu(g.float(), A32, b32, iters, lr, cc.float())
        ub = sound_eval(g, cc, A, b, nu)
        cand = torch.sign(g - nu.double() @ A) if A.shape[0] else torch.sign(g)
        for j, i in enumerate(ids):
            gi, ci, pi, ex = objs[i]
            res[i] = {"upper": float(ub[j]) + pi, "g": gi, "cc": ci, "pad": pi, "extra": ex,
                      "cand": cand[j].cpu().numpy()}
    for i, atoms in enumerate(disjuncts):
        if len(atoms) == 1:
            continue
        gi, ci, pi, ex = objs[i]
        g = torch.as_tensor(gi, device=dev, dtype=torch.float64).unsqueeze(0)
        cc = torch.as_tensor([ci], device=dev, dtype=torch.float64)
        AA = torch.cat([A, torch.as_tensor(np.stack([a for a, _ in ex]), device=dev, dtype=torch.float64)])
        BB = torch.cat([b, torch.as_tensor([bb for _, bb in ex], device=dev, dtype=torch.float64)])
        nu = optimise_nu(g.float(), AA.float(), BB.float(), iters, lr, cc.float())
        ub = sound_eval(g, cc, AA, BB, nu)
        cand = torch.sign(g - nu.double() @ AA)
        res[i] = {"upper": float(ub[0]) + pi, "g": gi, "cc": ci, "pad": pi, "extra": ex, "cand": cand[0].cpu().numpy()}
    return res
