"""Terminal MILP v6 = terminal v5 with the monotone-unit rule (Proposition 2) on every
gamma layer whose downstream ReLU layers are all gamma (in practice: the all-layers
stage E, and the last layer in stage D where it coincides with Proposition 1')."""
import numpy as np
import scipy.sparse as sp

from nhz_monotone import monotone_units
from nhz_terminal_v5 import run_portfolio, SEEDS

TERMINAL_V6_VERSION = "terminal-v6.0"


def build_plan_v6(A, b, phases, K, gamma_layers, lambda_layers, g_full, extra, sign_aware=True):
    nz = [p for p in phases if int(p["idx"].numel())]
    lam = nz[-lambda_layers:] if lambda_layers > 0 else []
    gam = nz[-gamma_layers:] if gamma_layers > 0 else []
    An = A.double().cpu().numpy(); bn = b.double().cpu().numpy()
    keep = np.zeros(An.shape[0], bool)
    for p in lam:
        m = int(p["idx"].numel()); keep[p["row0"]:p["row0"] + 2 * m] = True
    if sign_aware and gam:
        funcs = [g_full] + [ak for ak, _ in extra]; signs = [1] + [-1] * len(extra)
        need_all = monotone_units(phases, K, funcs, signs)
        need = need_all[len(nz) - len(gam):]
        sel = [np.flatnonzero(nd) for nd in need]
    else:
        sel = [np.arange(int(p["idx"].numel())) for p in gam]
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


def plan_milp_v6(A, b, phases, K, g, cc, pad, extra, gamma_layers, lambda_layers, time_limit, margin=1e-4,
                 target=1e-6, seeds=SEEDS, sign_aware=True):
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M, rhs, nb, n_units = build_plan_v6(A, b, phases, K, gamma_layers, lambda_layers, g_full, extra, sign_aware)
    r = run_portfolio(M, rhs, nb, K, g_full, cc, pad, time_limit, margin, target, seeds)
    r["gamma_units"] = n_units
    return r
