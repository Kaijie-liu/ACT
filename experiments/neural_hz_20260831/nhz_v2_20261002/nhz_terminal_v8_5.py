"""Terminal MILP v8.4 (copy of v8.3 on terminal v7.10; the frozen nhz_terminal_v8.py of N039 v14 is unchanged).

Terminal MILP v8 = terminal v7 + smooth units in the plan (THEORY.md Section 14).

Every recorded smooth unit j (engine n016) gets an explicit pre-activation column x_j
(x_j - gx_j^T w = cx_j) and its rigorous chord/tangent rows written sparsely in (x_j, eta_j):
    y_true in lam x_j + nu eta_j + (yc - lam cx) +- (ye + delta),   delta = 2^-24 lam sum|gx| (float32 storage)
    y_true <= a x_true + b_up,  y_true >= a x_true + b_lo,  x_true in x_j +- ex .
Plans before v8 dropped these rows (they kept only ReLU phase rows), so in the MILP stages a
smooth unit was only its DeepZ parallelogram.  Optionally, selected units get phase segments
(Balas disaggregation, one binary per segment, sum d = 1): the input range is split at the
inflection point and evenly inside each convexity region; each segment carries its own
rigorous lines.  The union of segment hulls contains the graph of f on [l, u], so every
plan remains a relaxation of the exact network."""
import numpy as np
import scipy.sparse as sp
import torch

from nhz_sound_mp2 import rigorous_lines
from nhz_terminal_v7_11 import build_plan_v7, run_portfolio_v7, SEEDS

TERMINAL_V8_VERSION = "terminal-v8.5"  # .5: on terminal v7.11, smooth rows widened by 1e-5;  # .3: optional MIP start (true phases at a latent point);  # .2: piece columns bounded by their segment hull;  # .1: sign rule off when a smooth layer follows the last ReLU layer
U24 = 2.0 ** -24


def _breaks(l, u, K):
    """per-unit breakpoints (n, K+1): split at 0 when l < 0 < u (K//2 left, rest right), else even."""
    n = l.shape[0]; B = np.zeros((n, K + 1))
    for j in range(n):
        if l[j] < 0 < u[j] and K >= 2:
            k1 = K // 2; k2 = K - k1
            left = np.linspace(l[j], 0.0, k1 + 1); right = np.linspace(0.0, u[j], k2 + 1)
            B[j] = np.concatenate([left, right[1:]])
        else:
            B[j] = np.linspace(l[j], u[j], K + 1)
    return B


def exact_preacts(phases, smooth, K, w_in):
    """Exact HZ semantics at the input latent point w_in (length = number of input factors):
    returns {id(layer): x vector} for every ReLU phase layer and smooth layer, filling the eta
    factors with their exact values in latent order (as test N066)."""
    from nhz_monotone import _lam_mu
    nzp = [p for p in phases if int(p["idx"].numel())]
    layers = sorted([("relu", p) for p in nzp] + [("smooth", s_) for s_ in smooth], key=lambda t: int(t[1]["eta0"]))
    w = np.zeros(K); w[:len(w_in)] = w_in; out = {}
    for kind, p in layers:
        e0 = int(p["eta0"]); gx = p["gx"].double().cpu().numpy(); cx = p["cx"].double().cpu().numpy()
        x = cx + gx @ w[:gx.shape[1]]
        if kind == "relu":
            lam, mu = (v.cpu().numpy() for v in _lam_mu(p))
            w[e0:e0 + x.size] = np.clip((np.maximum(x, 0) - lam * x) / mu - 1, -1, 1)
        else:
            lam = p["lam"].cpu().numpy(); nu = p["nu"].cpu().numpy(); c0 = p["yc"].cpu().numpy() - lam * cx
            f = 1 / (1 + np.exp(-x)) if p["kind"] == "Sigmoid" else np.tanh(x)
            w[e0:e0 + x.size] = np.clip((f - lam * x - c0) / nu, -1, 1)
        out[id(p)] = x
    return out


def build_plan_v8(A, b, phases, smooth, K, gamma_layers, lambda_layers, g_full, extra, seg_select=None, K_seg=2,
                  sign_aware=True, start_w=None):
    nz = [p for p in phases if int(p["idx"].numel())]
    if nz and any(int(s_["eta0"]) > int(nz[-1]["eta0"]) for s_ in smooth):
        sign_aware = False      # a smooth layer after the last ReLU layer: Proposition 1' premise fails
    M7, lo7, hi7, clo7, chi7, cost7, integ7, nb7, m_tot = build_plan_v7(A, b, phases, K, gamma_layers, lambda_layers,
                                                                        g_full, extra, sign_aware)
    n7 = M7.shape[1]
    start_int = None
    if start_w is not None:
        # v8.3: MIP start = true phases of the element at start_w (binaries only; the solver completes it)
        from nhz_terminal_v7_11 import _need_masks
        X = exact_preacts(phases, smooth, K, start_w)
        gam = nz[-gamma_layers:] if gamma_layers > 0 else []
        need = _need_masks(phases, nz, gam, K, g_full, extra, sign_aware)
        start_int = [((X[id(p)][nd] >= 0).astype(np.float64)) for p, nd in zip(gam, need)]
        start_int = np.concatenate(start_int) if start_int else np.zeros(0)
    units = []
    for li, sp_ in enumerate(smooth):
        n = int(sp_["n"])
        for j in range(n):
            units.append((li, j))
    Ns = len(units)
    if Ns == 0:
        build_plan_v8.last_start_int = start_int
        return M7, lo7, hi7, clo7, chi7, cost7, integ7, nb7, m_tot, 0
    sel = np.zeros(Ns, bool) if seg_select is None else np.asarray(seg_select, bool)
    Nsel = int(sel.sum()); S = int(K_seg)
    X0 = n7; XK0 = X0 + Ns; YK0 = XK0 + S * Nsel; DK0 = YK0 + S * Nsel; ntot = DK0 + S * Nsel
    blocks, LO, HI = [sp.hstack([M7, sp.csr_matrix((M7.shape[0], ntot - n7))]).tocsr()], [lo7], [hi7]
    clo = [clo7]; chi = [chi7]
    xl, xu = [], []
    sidx = 0; selpos = np.cumsum(sel) - 1; seg_bounds = {}; seg_start = []
    rows_r, rows_c, rows_v, rlo, rhi = [], [], [], [], []
    r = [0]

    def add(cols, vals, lo_, hi_):
        rows_r.extend([r[0]] * len(cols)); rows_c.extend(cols); rows_v.extend(vals); rlo.append(lo_); rhi.append(hi_); r[0] += 1

    for li, sp_ in enumerate(smooth):
        n = int(sp_["n"]); e0 = int(sp_["eta0"]); kind = sp_["kind"]
        gx = sp_["gx"].double().cpu().numpy(); cx = sp_["cx"].double().cpu().numpy(); ex = sp_["ex"].double().cpu().numpy()
        lam = sp_["lam"].double().cpu().numpy(); nu = sp_["nu"].double().cpu().numpy()
        yc = sp_["yc"].double().cpu().numpy(); ye = sp_["ye"].double().cpu().numpy()
        l = sp_["l"].double(); u = sp_["u"].double()
        delta = U24 * lam * np.abs(gx).sum(1) * (1 + 1e-12) + 1e-300
        tol = ye + delta + 1e-5; c0 = yc - lam * cx          # v8.5: numerical safety width (LOG N112)
        ex = ex + 1e-5
        # x equality rows (dense in w, built as a sparse block)
        rr = np.arange(n)
        Gx = sp.csr_matrix(gx); Gx = sp.hstack([-Gx, sp.csr_matrix((n, ntot - gx.shape[1]))]).tocsr()
        Gx = Gx + sp.csr_matrix((np.ones(n), (rr, X0 + sidx + rr)), shape=(n, ntot))
        blocks.append(Gx); LO.append(cx); HI.append(cx)
        ln, un = l.cpu().numpy(), u.cpu().numpy()
        xl.append(ln - ex - 1.0); xu.append(un + ex + 1.0)
        lows, ups = rigorous_lines(kind, l, u)
        for (a_, b_) in ups:                       # (lam - a) x + nu eta <= b_up - c0 + tol + |a| ex
            a_ = a_.cpu().numpy(); b_ = b_.cpu().numpy()
            for j in range(n):
                add([X0 + sidx + j, e0 + j], [lam[j] - a_[j], nu[j]], -np.inf, b_[j] - c0[j] + tol[j] + abs(a_[j]) * ex[j])
        for (a_, b_) in lows:                      # (a - lam) x - nu eta <= c0 + tol + |a| ex - b_lo
            a_ = a_.cpu().numpy(); b_ = b_.cpu().numpy()
            for j in range(n):
                add([X0 + sidx + j, e0 + j], [a_[j] - lam[j], -nu[j]], -np.inf, c0[j] + tol[j] + abs(a_[j]) * ex[j] - b_[j])
        # segments
        my = [j for j in range(n) if sel[sidx + j]]
        if my:
            my = np.array(my); Bk = _breaks(ln[my], un[my], S)
            if start_w is not None:
                xs = X[id(sp_)][my]
                ks = np.clip(np.array([np.searchsorted(Bk[t], xs[t], side="right") - 1 for t in range(len(my))]), 0, S - 1)
                seg_start.append((my, ks))
            for k in range(S):
                lk = torch.as_tensor(Bk[:, k], dtype=torch.float64); uk = torch.as_tensor(Bk[:, k + 1], dtype=torch.float64)
                slows, sups = rigorous_lines(kind, lk, uk)
                for t, j in enumerate(my):
                    q = int(selpos[sidx + j]); xk = XK0 + q * S + k; yk = YK0 + q * S + k; dk = DK0 + q * S + k
                    seg_bounds[(q, k)] = (float(Bk[t, k]), float(Bk[t, k + 1]))
                    add([xk, dk], [1.0, -Bk[t, k + 1]], -np.inf, 0.0)        # x_k <= b_k d_k
                    add([xk, dk], [1.0, -Bk[t, k]], 0.0, np.inf)              # x_k >= b_{k-1} d_k
                    for (a_, b_) in sups:
                        add([yk, xk, dk], [1.0, -float(a_[t]), -float(b_[t])], -np.inf, 0.0)
                    for (a_, b_) in slows:
                        add([yk, xk, dk], [1.0, -float(a_[t]), -float(b_[t])], 0.0, np.inf)
            for t, j in enumerate(my):
                q = int(selpos[sidx + j]); ek = e0 + j
                add([DK0 + q * S + k for k in range(S)], [1.0] * S, 1.0, 1.0)
                add([XK0 + q * S + k for k in range(S)] + [X0 + sidx + j], [1.0] * S + [-1.0], -ex[j], ex[j])
                # y_true = sum y_k  in  lam x + nu eta + c0 +- tol
                add([YK0 + q * S + k for k in range(S)] + [X0 + sidx + j, ek], [1.0] * S + [-lam[j], -nu[j]], c0[j] - tol[j], c0[j] + tol[j])
        sidx += n
    R = sp.csr_matrix((rows_v, (rows_r, rows_c)), shape=(r[0], ntot))
    blocks.append(R); LO.append(np.array(rlo)); HI.append(np.array(rhi))
    M = sp.vstack(blocks).tocsc()
    # v8.2: piece columns bounded by their own segment hull (not by a fixed +-1e3; LOG N085)
    xk_lo = np.full(S * Nsel, 0.0); xk_hi = np.full(S * Nsel, 0.0)
    for (q, k), (blo, bhi) in seg_bounds.items():
        xk_lo[q * S + k] = min(0.0, blo); xk_hi[q * S + k] = max(0.0, bhi)
    clo_all = np.concatenate([clo7, np.concatenate(xl), xk_lo, np.full(S * Nsel, -2.0), np.zeros(S * Nsel)])
    chi_all = np.concatenate([chi7, np.concatenate(xu), xk_hi, np.full(S * Nsel, 2.0), np.ones(S * Nsel)])
    cost = np.concatenate([cost7, np.zeros(ntot - n7)])
    integ = np.concatenate([integ7, np.zeros(Ns + 2 * S * Nsel, bool), np.ones(S * Nsel, bool)])
    if start_int is not None:
        dk = np.zeros(S * Nsel)
        q = 0
        for my, ks in seg_start:
            for t in range(len(my)):
                dk[q * S + ks[t]] = 1.0; q += 1
        start_int = np.concatenate([start_int, dk])
    build_plan_v8.last_start_int = start_int
    return M, np.concatenate(LO), np.concatenate(HI), clo_all, chi_all, cost, integ, nb7 + S * Nsel, m_tot, Nsel


def plan_milp_v8(A, b, phases, smooth, K, g, cc, pad, extra, gamma_layers, lambda_layers, time_limit, seg_select=None,
                 K_seg=2, margin=1e-4, target=1e-6, seeds=SEEDS, start_w=None):
    import time as _t
    t0 = _t.time()
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M, lo, hi, clo, chi, cost, integ, nb, m_tot, nsel = build_plan_v8(A, b, phases, smooth, K, gamma_layers, lambda_layers,
                                                                       g_full, extra, seg_select, K_seg, start_w=start_w)
    start_int = build_plan_v8.last_start_int
    left = time_limit - (_t.time() - t0)
    r = run_portfolio_v7(M, lo, hi, clo, chi, cost, integ, K, cc, pad, max(left, 0.1), margin, target, seeds,
                         start_int=start_int)
    r.update({"build_s": _t.time() - t0 - r["wall_s"], "segment_units": nsel, "nnz": int(M.nnz)})
    return r
