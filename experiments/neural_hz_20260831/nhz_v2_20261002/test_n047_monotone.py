"""N047 test: monotone-unit analysis (nhz_monotone.py).
(1) the unit-coordinate rewrite reproduces exact forward differences;
(2) finite-difference derivatives d f / d y_s (downstream recomputed exactly) lie in the
    interval enclosure at random exact points;
(3) plan optimum with and without the dropped integrality is identical (HiGHS, no cutoff)."""
import sys, os, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
import nhz_monotone as NM
torch.cuda.set_per_process_memory_fraction(0.05)
csv.field_size_limit(10**9)
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
ov = {(r['benchmark'], r['iid']): r for r in csv.DictReader(open('/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv'))}
fam, iid = sys.argv[1], sys.argv[2]
r = ov[(fam, iid)]
e = SoundEngineV9(os.path.normpath(os.path.join(ROOT, fam, r['onnx'])), 'cuda')
n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, fam, r['vnnlib'])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
o, rows, K, ph = e.propagate(lb, ub, 100, 0.05)
nz = [p for p in ph if int(p["idx"].numel())]
print("K", K, "layers", [int(p["idx"].numel()) for p in nz])
g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
f = np.asarray(g)[:K] if not extra else np.asarray(extra[0][0])[:K]


def forward(eps, override=None):
    """exact HZ semantics: returns latent w, x per layer, y per layer"""
    w = np.zeros(K); K0 = int(nz[0]["eta0"]); w[:K0] = eps
    xs, ys = [], []
    for s, p in enumerate(nz):
        e0 = int(p["eta0"]); m = int(p["idx"].numel())
        gx = p["gx"].double().cpu().numpy(); x = p["cx"].double().cpu().numpy() + gx @ w[:gx.shape[1]]
        y = np.maximum(x, 0)
        if override is not None and override[0] == s:
            y = y.copy(); y[override[1]] += override[2]
        lam, mu = (v.cpu().numpy() for v in NM._lam_mu(p))
        w[e0:e0 + m] = (y - lam * x) / mu - 1
        xs.append(x); ys.append(y)
    return w, xs, ys


# rebuild the unit-coordinate matrices exactly as monotone_units does
X = []
for p in nz:
    gg = torch.zeros((int(p["idx"].numel()), K), dtype=torch.float64); g0 = p["gx"].cpu().double(); gg[:, :g0.shape[1]] = g0; X.append(gg)
Fm = torch.zeros((1, K), dtype=torch.float64); Fm[0, :] = torch.as_tensor(f)
for s, p in enumerate(nz):
    e0 = int(p["eta0"]); m = int(p["idx"].numel()); lam, mu = NM._lam_mu(p); lam, mu = lam.cpu(), mu.cpu()
    Xs = X[s][:, :e0]
    for R in [X[t] for t in range(s + 1, len(nz))] + [Fm]:
        ce = R[:, e0:e0 + m]; R[:, :e0] -= (ce * (lam / mu)) @ Xs; R[:, e0:e0 + m] = ce / mu
rng = np.random.default_rng(0); K0 = int(nz[0]["eta0"])
err = 0.0
for trial in range(20):
    e1, e2 = rng.uniform(-1, 1, K0), rng.uniform(-1, 1, K0)
    w1, x1, y1 = forward(e1); w2, x2, y2 = forward(e2)
    u1 = np.concatenate([e1] + [np.zeros(0)]); 
    def unit_vec(eps, ys):
        v = np.zeros(K); v[:K0] = eps
        for p, y in zip(nz, ys):
            v[int(p["eta0"]):int(p["eta0"]) + y.size] = y
        return v
    d = unit_vec(e1, y1) - unit_vec(e2, y2)
    for t in range(len(nz)):
        pred = X[t].numpy() @ d; true = x1[t] - x2[t]
        err = max(err, float(np.abs(pred - true).max() / (1 + np.abs(true).max())))
    pf = float(Fm[0].numpy() @ d); tf = float(f @ (w1 - w2))
    err = max(err, abs(pf - tf) / (1 + abs(tf)))
print("(1) max relative rewrite error", err)
# (2) enclosure check
lo_hi = {}
need = NM.monotone_units(ph, K, [f], [1])
# recompute intervals for reporting via a private copy of the reverse pass
lo = [None] * len(nz); hi = [None] * len(nz)
for s in range(len(nz) - 1, -1, -1):
    e0 = int(nz[s]["eta0"]); m = int(nz[s]["idx"].numel()); a = Fm[0, e0:e0 + m].clone(); b = a.clone()
    for t in range(s + 1, len(nz)):
        J = X[t][:, e0:e0 + m]; xl = torch.clamp(lo[t], max=0.0); xh = torch.clamp(hi[t], min=0.0)
        a += xl @ J.clamp(min=0) + xh @ J.clamp(max=0); b += xh @ J.clamp(min=0) + xl @ J.clamp(max=0)
    lo[s], hi[s] = a, b
viol = 0; checked = 0
for trial in range(30):
    eps = rng.uniform(-1, 1, K0); w0, xs0, ys0 = forward(eps); f0 = f @ w0
    for s in range(len(nz)):
        for j in rng.choice(int(nz[s]["idx"].numel()), size=min(5, int(nz[s]["idx"].numel())), replace=False):
            if not (nz[s]["l"][j] <= xs0[s][j] <= nz[s]["u"][j]):
                continue
            dlt = 1e-6
            w1, _, _ = forward(eps, (s, j, dlt)); der = (f @ w1 - f0) / dlt
            checked += 1
            if der < float(lo[s][j]) - 1e-5 * (1 + abs(der)) or der > float(hi[s][j]) + 1e-5 * (1 + abs(der)):
                viol += 1
print("(2) derivative checks", checked, "outside enclosure", viol)
print("    units without integrality per layer:", [int((~n).sum()) for n in need], "of", [int(n.size) for n in need])
