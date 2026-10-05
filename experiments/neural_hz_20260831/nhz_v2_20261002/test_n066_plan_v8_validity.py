"""N066 test: every row and column bound of the terminal v8 plan holds at exact points.

Exact points are built from random box inputs with the engine's own value maps (exact HZ
semantics): x_hat = c + G w, ReLU y = max(x_hat, 0), eta from the aligned form; smooth
y = f(x_hat), eta from the shadow form.  Columns of the plan are filled accordingly (Balas
pieces for ReLU binary units, the active segment for smooth segments).  A violation means a
construction error that could make the plan exclude real points."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v16 import SoundEngineV16
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v8 import build_plan_v8, _breaks
from nhz_monotone import _lam_mu
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", "0.05")))
U = {(r["family"], r["iid"]): r for r in universe()}
fam, iid = sys.argv[1], int(sys.argv[2]); S = int(sys.argv[3]) if len(sys.argv) > 3 else 2
r = U[(fam, iid)]
e = SoundEngineV16(os.path.normpath(os.path.join(ROOT, fam, r["onnx"])), "cuda")
n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
nz = [p for p in ph if int(p["idx"].numel())]; sm = e.smooth_phases; nl = len(nz)
g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
g_full = np.zeros(K + 1); g_full[:len(g)] = g
Ns = sum(int(s_["n"]) for s_ in sm)
sel = np.ones(Ns, bool)
M, lo, hi, clo, chi, cost, integ, nb, m_tot, nsel = build_plan_v8(A, b, ph, sm, K, nl, 99, g_full, extra, sel, S, sign_aware=False)
print("plan", M.shape, "nnz", M.nnz, "binaries", nb, "segment units", nsel)
Mr = M.tocsr()
K0 = int(e.input_factor_index.numel())
layers = sorted([("relu", p) for p in nz] + [("smooth", s_) for s_ in sm], key=lambda t: int(t[1]["eta0"]))
rng = np.random.default_rng(1); worst_row = 0.0; worst_col = 0.0
for trial in range(100):
    w = np.zeros(K); w[:K0] = rng.uniform(-1, 1, K0)
    vals = {}
    for kind, p in layers:
        e0 = int(p["eta0"]); gx = p["gx"].double().cpu().numpy(); cx = p["cx"].double().cpu().numpy()
        x = cx + gx @ w[:gx.shape[1]]
        if kind == "relu":
            lam, mu = (v.cpu().numpy() for v in _lam_mu(p)); y = np.maximum(x, 0)
            w[e0:e0 + x.size] = (y - lam * x) / mu - 1
        else:
            lam = p["lam"].cpu().numpy(); nu = p["nu"].cpu().numpy(); c0 = p["yc"].cpu().numpy() - lam * cx
            f = 1 / (1 + np.exp(-x)) if p["kind"] == "Sigmoid" else np.tanh(x)
            w[e0:e0 + x.size] = (f - lam * x - c0) / nu
        vals[id(p)] = x
    # column vector: w | tau | x_relu (gamma = all) | x0 | x1 | d | x_smooth | xk | yk | dk
    v = np.zeros(M.shape[1]); v[:K] = w; v[K] = 0.0
    X0 = K + 1; xs = np.concatenate([vals[id(p)] for p in nz]); m_all = xs.size
    v[X0:X0 + m_all] = xs
    P0 = X0 + m_all; nbr = m_all; Q0 = P0 + nbr; D0 = Q0 + nbr
    v[P0:P0 + nbr] = np.minimum(xs, 0); v[Q0:Q0 + nbr] = np.maximum(xs, 0); v[D0:D0 + nbr] = (xs >= 0)
    n7 = D0 + nbr
    xsm = np.concatenate([vals[id(s_)] for s_ in sm]); v[n7:n7 + Ns] = xsm
    XK0 = n7 + Ns; YK0 = XK0 + S * Ns; DK0 = YK0 + S * Ns
    ls = np.concatenate([s_["l"].cpu().numpy() for s_ in sm]); us = np.concatenate([s_["u"].cpu().numpy() for s_ in sm])
    Bk = _breaks(ls, us, S)
    kinds = np.concatenate([[s_['kind']] * int(s_['n']) for s_ in sm])
    f = np.where(kinds == 'Sigmoid', 1 / (1 + np.exp(-xsm)), np.tanh(xsm))
    for q in range(Ns):
        k = int(np.clip(np.searchsorted(Bk[q], xsm[q], side="right") - 1, 0, S - 1))
        v[XK0 + q * S + k] = xsm[q]; v[YK0 + q * S + k] = f[q]; v[DK0 + q * S + k] = 1
    Mv = Mr @ v
    worst_row = max(worst_row, float(np.max(np.maximum(lo - Mv, 0) / (1 + np.abs(lo)))), float(np.max(np.maximum(Mv - hi, 0) / (1 + np.abs(hi)))))
    worst_col = max(worst_col, float(np.max(np.maximum(clo - v, 0))), float(np.max(np.maximum(v - chi, 0))))
print("worst relative row violation", worst_row, "worst column-bound violation", worst_col)
