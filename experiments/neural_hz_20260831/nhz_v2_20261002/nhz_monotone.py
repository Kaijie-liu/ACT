"""Monotone-unit analysis (THEORY.md Section 12.3, Proposition 2).

Rewrites every pre-activation of an unstable unit in "unit coordinates":
    x_t = const + P_t eps + sum_{s<t} J_ts y_s ,
by eliminating eta_s = (y_s - lam_s x_s)/mu_s - 1 layer by layer (stable units are
already folded into the affine maps).  The same elimination is applied to the
query functionals (objective and atom rows).  An interval reverse pass with ReLU
slopes in [0, 1] then encloses d f / d y_s over every point whose downstream
layers are exact.  A unit may skip integrality in a plan whose downstream layers
are all exact iff lowering it never hurts any functional:
    objective (maximised):  sup d obj / d y_s <= 0
    atom rows (<= b_k):     inf d row / d y_s >= 0 .
Dropping integrality only relaxes the plan, so soundness never depends on this
analysis; Proposition 2 says that it also costs no precision.
"""
import numpy as np
import torch

F64 = torch.float64


def _lam_mu(p):
    l = p["l"].double(); u = p["u"].double()
    lam = (u / (u - l)).clamp(0.0, 1.0)
    m = int(p["idx"].numel()); e0 = int(p["eta0"])
    mu = p["gy"][torch.arange(m, device=p["gy"].device), e0 + torch.arange(m, device=p["gy"].device)].double()
    return lam, mu


def monotone_units(phases, K, functionals, signs, device="cpu"):
    """functionals: list of length-K(+1) arrays; signs: +1 if the functional is maximised
    (needs sup <= 0), -1 if it is a <= row (needs inf >= 0).
    Returns a list (one bool array per phase with units) of 'needs integrality'."""
    nz = [p for p in phases if int(p["idx"].numel())]
    if not nz:
        return []
    X = []
    for p in nz:
        g = torch.zeros((int(p["idx"].numel()), K), dtype=F64, device=device)
        g0 = p["gx"].to(device).double(); g[:, :g0.shape[1]] = g0
        X.append(g)
    Fm = torch.zeros((len(functionals), K), dtype=F64, device=device)
    for i, f in enumerate(functionals):
        f = np.asarray(f)[:K]; Fm[i, :f.size] = torch.as_tensor(f, dtype=F64)
    lm = [_lam_mu(p) for p in nz]
    for s, p in enumerate(nz):
        e0 = int(p["eta0"]); m = int(p["idx"].numel()); lam, mu = (v.to(device) for v in lm[s])
        Xs = X[s][:, :e0]                                   # already in unit coordinates
        for R in [X[t] for t in range(s + 1, len(nz))] + [Fm]:
            ce = R[:, e0:e0 + m]                            # coefficients on eta_s
            R[:, :e0] -= (ce * (lam / mu)) @ Xs
            R[:, e0:e0 + m] = ce / mu
    need = []
    for i, sg in enumerate(signs):
        # reverse pass: adjoint of y_s = F[y_s] + sum_t xbar_t J_ts,  xbar_t = slope * ybar_t
        lo = [None] * len(nz); hi = [None] * len(nz)
        for s in range(len(nz) - 1, -1, -1):
            e0 = int(nz[s]["eta0"]); m = int(nz[s]["idx"].numel())
            a = Fm[i, e0:e0 + m].clone(); b = a.clone()
            for t in range(s + 1, len(nz)):
                J = X[t][:, e0:e0 + m]                       # (m_t, m_s)
                xl = torch.clamp(lo[t], max=0.0); xh = torch.clamp(hi[t], min=0.0)
                Jp = J.clamp(min=0.0); Jn = J.clamp(max=0.0)
                a += xl @ Jp + xh @ Jn; b += xh @ Jp + xl @ Jn
            lo[s], hi[s] = a, b
        ok = [(h <= 0) if sg > 0 else (l_ >= 0) for l_, h in zip(lo, hi)]
        need.append([~o for o in ok])
    out = []
    for s in range(len(nz)):
        n_s = torch.zeros(int(nz[s]["idx"].numel()), dtype=torch.bool, device=device)
        for i in range(len(signs)):
            n_s |= need[i][s]
        out.append(n_s.cpu().numpy())
    return out
