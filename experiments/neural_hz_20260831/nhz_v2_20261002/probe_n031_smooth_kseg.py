"""N031 probe: gamma-level segment unions for sigmoid units (float engine n005.3).

For the top-N sigmoid units ranked by |g[eta_j]| * nu_j (objective coefficient of the
unit's shadow error factor times its width: specification and weights only), split
[l_j, u_j] at 0 (if inside) and at equal-width points into S segments; per segment,
valid lower/upper lines (shifted chord and tangents, smooth_lines on the segment);
one-hot binaries choose the segment; big-M rows enforce the chosen segment's lines and
x in that segment.  All ReLU phases exact.  HiGHS MILP, worst open disjunct.
"""
import sys, os, csv, time
import numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_attn import AttnEngine, smooth_lines
from nhz_engine import parse_vnnlib, lp_upper
from run_n011_partial_milp import build_partial
from scipy.optimize import milp, LinearConstraint, Bounds
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/dist_shift_2023"
torch.cuda.set_per_process_memory_fraction(0.1)
TOPN, S, TL = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
eng = {}
for ri in [int(x) for x in sys.argv[1].split(",")]:
    o, s = inst[ri][:2]; mp = os.path.normpath(f"{ROOT}/{o}")
    if mp not in eng: eng[mp] = AttnEngine(mp, "cuda", torch.float32)
    e = eng[mp]
    z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
    spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{s}"), int(np.prod(e.input_shape)), int(o0.c.numel())); lb, ub = spec.boxes[0]
    # capture sigmoid input forms by wrapping smooth_state
    import nhz_attn
    captured = {}
    orig = nhz_attn.smooth_state
    def cap(kind, x, K):
        y, n, extra = orig(kind, x, K)
        captured["x"] = (extra[1].clone(), extra[0].clone(), K, n)      # fc, fG [K, n], eta0, n
        return y, n, extra
    nhz_attn.smooth_state = cap
    out, rows, K, ph = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
    nhz_attn.smooth_state = orig
    c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
    gg = torch.stack([-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts])
    cc = torch.as_tensor([d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts], device="cuda")
    lpb = lp_upper(gg, cc, A, b, 1000, 0.05); k = int(torch.argmax(lpb))
    fc, fG, eta0, n = captured["x"]
    r = fG.abs().sum(0); l = fc - r; u = torch.maximum(fc + r, fc - r + 1e-9)
    # sigmoid output forms: y = lam x + mu + nu eta (recompute as in smooth_state)
    from nhz_attn import _df, _f
    lam = torch.minimum(_df("Sigmoid", l), _df("Sigmoid", u))
    lo_ = _f("Sigmoid", l) - lam * l; hi_ = _f("Sigmoid", u) - lam * u
    mu = (lo_ + hi_) / 2; nu = ((hi_ - lo_) / 2).clamp(min=0)
    score = gg[k, eta0:eta0 + n].abs() * nu
    sel = torch.argsort(score, descending=True)[:TOPN].tolist()
    AA, bb, nb = build_partial(A, b, ph, K, 99)          # all ReLU phases exact
    ncol0 = K + nb; extra_rows = []; extra_rhs = []; nseg_total = 0
    gx_full = torch.zeros((K,), device="cuda")
    for j in sel:
        lj, uj = float(l[j]), float(u[j])
        bps = sorted(set([lj, uj] + ([0.0] if lj < 0 < uj else []) + [lj + (uj - lj) * q / S for q in range(1, S)]))
        segs = list(zip(bps[:-1], bps[1:]))
        gx = np.zeros(K); gx[:fG.shape[0]] = fG[:, j].double().cpu().numpy(); cx = float(fc[j])
        gy = np.zeros(K); gy[:fG.shape[0]] = float(lam[j]) * gx[:fG.shape[0]]; gy[eta0 + j] = float(nu[j]); cy = float(lam[j] * fc[j] + mu[j])
        ylo, yhi = float(_f("Sigmoid", torch.tensor(lj))), float(_f("Sigmoid", torch.tensor(uj)))
        base = ncol0 + nseg_total; m = len(segs)
        # one-hot: sum delta = 1
        row = sp.lil_matrix((2, ncol0 + nseg_total + m)) if False else None
        for q, (a0, a1) in enumerate(segs):
            lows, ups = smooth_lines("Sigmoid", torch.tensor([a0], dtype=torch.float64), torch.tensor([a1], dtype=torch.float64))
            dcol = base + q
            for (sa, sb) in ups:   # y - a x - b <= M (1 - d)
                a_, b_ = float(sa[0]), float(sb[0]); M = yhi - min(a_ * lj, a_ * uj) - b_ + 1e-6
                extra_rows.append(({**{i: gy[i] - a_ * gx[i] for i in np.flatnonzero(gy - a_ * gx)}, dcol: M}, b_ + a_ * cx - cy + M))
            for (sa, sb) in lows:  # a x + b - y <= M (1 - d)
                a_, b_ = float(sa[0]), float(sb[0]); M = max(a_ * lj, a_ * uj) + b_ - ylo + 1e-6
                extra_rows.append(({**{i: a_ * gx[i] - gy[i] for i in np.flatnonzero(a_ * gx - gy)}, dcol: M}, cy - a_ * cx - b_ + M))
            # x >= a0 when d:  a0 - x <= (a0 - lj)(1 - d) ;  x <= a1 when d: x - a1 <= (uj - a1)(1 - d)
            extra_rows.append(({**{i: -gx[i] for i in np.flatnonzero(gx)}, dcol: (a0 - lj)}, cx - a0 + (a0 - lj)))
            extra_rows.append(({**{i: gx[i] for i in np.flatnonzero(gx)}, dcol: (uj - a1)}, a1 - cx + (uj - a1)))
        extra_rows.append(({base + q: 1.0 for q in range(m)}, 1.0)); extra_rows.append(({base + q: -1.0 for q in range(m)}, -1.0))
        nseg_total += m
    ncol = ncol0 + nseg_total
    data, ri_, ci_ = [], [], []
    for t_, (dct, rhs) in enumerate(extra_rows):
        for cidx, v in dct.items():
            data.append(v); ri_.append(t_); ci_.append(int(cidx))
    E = sp.csr_matrix((data, (ri_, ci_)), shape=(len(extra_rows), ncol))
    AAx = sp.vstack([sp.hstack([AA, sp.csr_matrix((AA.shape[0], nseg_total))]), E]).tocsr()
    bbx = np.concatenate([bb, np.array([r_ for _, r_ in extra_rows])])
    obj = np.concatenate([-gg[k].double().cpu().numpy(), np.zeros(nb + nseg_total)])
    integ = np.concatenate([np.zeros(K), np.ones(nb + nseg_total)])
    bnds = Bounds(np.concatenate([-np.ones(K), np.zeros(nb + nseg_total)]), np.ones(ncol))
    t0 = time.time()
    res = milp(obj, constraints=LinearConstraint(AAx, -np.inf, bbx), integrality=integ, bounds=bnds, options={"time_limit": TL, "disp": False})
    db = getattr(res, "mip_dual_bound", None)
    print(ri, "LP", round(float(lpb[k]), 4), "relu bin", nb, "sigmoid units", len(sel), "segment bin", nseg_total, "MILP upper",
          round((-db + float(cc[k])) if db is not None else float("nan"), 4), "status", res.status, round(time.time() - t0, 1), "s", flush=True)
