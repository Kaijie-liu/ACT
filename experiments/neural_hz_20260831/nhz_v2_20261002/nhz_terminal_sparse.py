"""Sparse but equivalent encoding of the terminal plan (solver-level, no semantic change).

For every gamma/lambda unit of a single-ReLU-layer plan whose x value map depends only on
latent columns before the layer, introduce an explicit continuous x_k with the equality
x_k - gx_k^T w = cx_k (one dense row per unit) and write the unit's rows in (x_k, eta_k, d_k):
    y_k = lam_k x_k + mu_k (1 + eta_k)       (value map; lam, mu read from the phase)
  LP rows:     -y <= ey,   x - y <= ex + ey
  binary rows:  y - u d <= ey,   y - x + l (1 - d) <= ex + ey
The projection onto (w, d) is the plan of nhz_terminal_v5 up to the float32 storage of
lam*gx (whose effect is bounded by the radii already in ex/ey), so verdicts must agree.
Only used for single-layer plans (gamma = lambda = the one ReLU layer with phases)."""
import numpy as np
import scipy.sparse as sp

from nhz_monotone import _lam_mu
from nhz_terminal_v5 import needs_binary


def build_sparse_single(phases, K, g_full, extra, sign_aware=True):
    nz = [p for p in phases if int(p["idx"].numel())]
    assert len(nz) == 1
    p = nz[0]; m = int(p["idx"].numel()); e0 = int(p["eta0"])
    need = needs_binary(p, g_full, extra) if sign_aware else np.ones(m, bool)
    sel = np.flatnonzero(need); nb = sel.size
    lam, mu = (v.cpu().numpy() for v in _lam_mu(p))
    gx = p["gx"].double().cpu().numpy(); cx = p["cx"].double().cpu().numpy()
    ex = p["ex"].double().cpu().numpy(); ey = p["ey"].double().cpu().numpy()
    l = p["l"].double().cpu().numpy(); u = p["u"].double().cpu().numpy()
    # columns: w (K) | tau (1) | x (m) | d (nb)
    nW = K + 1; X0 = nW; D0 = nW + m; n = D0 + nb
    R, C, V, lo, hi = [], [], [], [], []
    row = 0
    def add(cols, vals, lo_, hi_):
        nonlocal row
        R.extend([row] * len(cols)); C.extend(cols); V.extend(vals); lo.append(lo_); hi.append(hi_); row += 1
    for k in range(m):
        nzc = np.flatnonzero(gx[k]); add([X0 + k] + list(nzc), [1.0] + list(-gx[k][nzc]), cx[k], cx[k])
    for k in range(m):
        ek = e0 + k; L, Mu = lam[k], mu[k]
        # y = L x + Mu eta + Mu
        add([X0 + k, ek], [-L, -Mu], -np.inf, Mu + ey[k])                    # -y <= ey
        add([X0 + k, ek], [1 - L, -Mu], -np.inf, Mu + ex[k] + ey[k])        # x - y <= ex+ey
    for j, k in enumerate(sel):
        ek = e0 + k; L, Mu = lam[k], mu[k]
        add([X0 + k, ek, D0 + j], [L, Mu, -u[k]], -np.inf, -Mu + ey[k])     # y - u d <= ey
        add([X0 + k, ek, D0 + j], [L - 1, Mu, -l[k]], -np.inf, -Mu - l[k] + ex[k] + ey[k])  # y - x - l d <= -l + ...
    for ak, bk in extra:
        a = np.asarray(ak); nzc = np.flatnonzero(a); add(list(nzc), list(a[nzc]), -np.inf, bk)
    M = sp.csc_matrix((V, (R, C)), shape=(row, n))
    xb = np.maximum(np.abs(l), np.abs(u)) * 2 + 1.0
    col_lo = np.concatenate([-np.ones(nW), -xb, np.zeros(nb)]); col_hi = np.concatenate([np.ones(nW), xb, np.ones(nb)])
    cost = np.concatenate([-g_full, np.zeros(m + nb)])
    integ = np.concatenate([np.zeros(nW + m), np.ones(nb)])
    return M, np.array(lo), np.array(hi), col_lo, col_hi, cost, integ, nb


def build_balas_single(phases, K, g_full, extra, sign_aware=True):
    """Disaggregated (Balas) variant of build_sparse_single for the binary units:
    x_k = x0_k + x1_k,  l (1 - d) <= x0 <= 0,  0 <= x1 <= u d,  y_k = x1_k  (y_k from the value map).
    Non-binary units keep the two LP rows.  Same projection onto (w, d) as the big-M plan."""
    nz = [p for p in phases if int(p["idx"].numel())]
    assert len(nz) == 1
    p = nz[0]; m = int(p["idx"].numel()); e0 = int(p["eta0"])
    need = needs_binary(p, g_full, extra) if sign_aware else np.ones(m, bool)
    sel = np.flatnonzero(need); nb = sel.size
    lam, mu = (v.cpu().numpy() for v in _lam_mu(p))
    gx = p["gx"].double().cpu().numpy(); cx = p["cx"].double().cpu().numpy()
    ex = p["ex"].double().cpu().numpy(); ey = p["ey"].double().cpu().numpy()
    l = p["l"].double().cpu().numpy(); u = p["u"].double().cpu().numpy()
    nW = K + 1; X0 = nW; P0 = X0 + m; Q0 = P0 + nb; D0 = Q0 + nb; n = D0 + nb   # x | x0 | x1 | d
    R, C, V, lo, hi = [], [], [], [], []
    row = 0
    def add(cols, vals, lo_, hi_):
        nonlocal row
        R.extend([row] * len(cols)); C.extend(cols); V.extend(vals); lo.append(lo_); hi.append(hi_); row += 1
    for k in range(m):
        nzc = np.flatnonzero(gx[k]); add([X0 + k] + list(nzc), [1.0] + list(-gx[k][nzc]), cx[k], cx[k])
    bset = set(sel.tolist())
    for k in range(m):
        if k in bset:
            continue
        ek = e0 + k; L, Mu = lam[k], mu[k]
        add([X0 + k, ek], [-L, -Mu], -np.inf, Mu + ey[k])
        add([X0 + k, ek], [1 - L, -Mu], -np.inf, Mu + ex[k] + ey[k])
    for j, k in enumerate(sel):
        ek = e0 + k; L, Mu = lam[k], mu[k]
        add([X0 + k, P0 + j, Q0 + j], [1.0, -1.0, -1.0], -ex[k], ex[k])                 # x = x0 + x1 (+-ex)
        add([X0 + k, ek, Q0 + j], [L, Mu, -1.0], -Mu - ey[k], -Mu + ey[k])              # y = x1 (+-ey)
        add([P0 + j, D0 + j], [1.0, l[k]], l[k], np.inf)                                # x0 >= l (1 - d)
        add([Q0 + j, D0 + j], [1.0, -u[k]], -np.inf, 0.0)                               # x1 <= u d
    for ak, bk in extra:
        a = np.asarray(ak); nzc = np.flatnonzero(a); add(list(nzc), list(a[nzc]), -np.inf, bk)
    M = sp.csc_matrix((V, (R, C)), shape=(row, n))
    xb = np.maximum(np.abs(l), np.abs(u)) * 2 + 1.0
    col_lo = np.concatenate([-np.ones(nW), -xb, l[sel] - 1e-9, np.zeros(nb), np.zeros(nb)])
    col_hi = np.concatenate([np.ones(nW), xb, np.zeros(nb), u[sel] + 1e-9, np.ones(nb)])
    cost = np.concatenate([-g_full, np.zeros(m + 3 * nb)])
    integ = np.concatenate([np.zeros(nW + m + 2 * nb), np.ones(nb)])
    return M, np.array(lo), np.array(hi), col_lo, col_hi, cost, integ, nb
