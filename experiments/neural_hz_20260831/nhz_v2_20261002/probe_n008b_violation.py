import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_n008_pair_hull as P
from nhz_engine import Engine, parse_vnnlib, lp_upper
from scipy.optimize import linprog
import csv
fam, row = sys.argv[1], int(sys.argv[2])
inst = list(csv.reader(open(f"{P.ROOT}/{fam}/instances.csv"))); o, s = inst[row][:2]
e = Engine(os.path.normpath(f"{P.ROOT}/{fam}/{o}"), "cuda", torch.float32)
z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
spec = parse_vnnlib(os.path.normpath(f"{P.ROOT}/{fam}/{s}"), int(np.prod(e.input_shape)), int(o0.c.numel()))
lb, ub = spec.boxes[0]
out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
objs = [(-(G @ torch.as_tensor(a0, device="cuda", dtype=torch.float32)), b0 - float(c @ torch.as_tensor(a0, device="cuda", dtype=torch.float32))) for (a0, b0), in [d[:1] for d in spec.disjuncts]]
g = torch.stack([x[0] for x in objs]); cc = torch.as_tensor([x[1] for x in objs], device="cuda")
k = int(torch.argmax(lp_upper(g, cc, A, b, 600, 0.05)))
An, bn = A.double().cpu().numpy(), b.double().cpu().numpy()
res = linprog(-g[k].double().cpu().numpy(), A_ub=An, b_ub=bn, bounds=(-1, 1), method="highs")
w = res.x; print("HiGHS base LP value", -res.fun + float(cc[k]))
tot = 0; viol = []
for p in phases:
    m = int(p.idx.numel())
    if m < 2: continue
    gx = p.gx.double().cpu().numpy(); cx = p.cx.double().cpu().numpy(); gy = p.gy.double().cpu().numpy(); cy = p.cy.double().cpu().numpy()
    l = p.l.double().cpu().numpy(); u = p.u.double().cpu().numpy()
    xs = cx + gx @ w[:gx.shape[1]]; ys = cy + gy @ w[:gy.shape[1]]
    gn = p.gx / (p.gx.norm(dim=1, keepdim=True) + 1e-30); sim = (gn @ gn.t()).abs(); sim.fill_diagonal_(-1)
    top = torch.topk(sim, 1, dim=1).indices.cpu().numpy()
    vmax = 0; nviol = 0; nrows = 0
    for i in range(m):
        j = int(top[i][0])
        for a, rhs in P.pair_rows(cx, gx, cy, gy, [(l[i], u[i]), (l[j], u[j])], i, j, K):
            v = float(a @ w - rhs); nrows += 1
            if v > 1e-7: nviol += 1; vmax = max(vmax, v)
    # slack of triangle: y - relu(x)
    slack = ys - np.maximum(xs, 0)
    print(f"{p.layer}: m={m} rows={nrows} violated={nviol} max_violation={vmax:.4g}  mean triangle slack at w*={slack.mean():.4g} max={slack.max():.4g}")
