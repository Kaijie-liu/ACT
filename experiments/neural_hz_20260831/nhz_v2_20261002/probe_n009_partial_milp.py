"""N009 diagnostic: terminal MILP with exact binaries only on the last k ReLU layers.

Query-time relaxation of the exact state: phases of the last k ReLU layers keep
their binary rows  y <= u d, y <= x - l(1-d)  (d integer); all earlier phases are
LP-relaxed (rows implied, Lemma 2).  Sound upper bound of the worst disjunct
violation (float, HiGHS tolerances).  Layer choice is structural (position).
"""
import sys, os, time, csv
import numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import Engine, parse_vnnlib, lp_upper
from scipy.optimize import milp, LinearConstraint, Bounds
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam, row, kl, tl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv"))); o, s = inst[row][:2]
e = Engine(os.path.normpath(f"{ROOT}/{fam}/{o}"), "cuda", torch.float32)
z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{fam}/{s}"), int(np.prod(e.input_shape)), int(o0.c.numel()))
lb, ub = spec.boxes[0]
out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
objs = [(-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)), d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32))) for d in spec.disjuncts]
gg = torch.stack([x[0] for x in objs]); cc = torch.as_tensor([x[1] for x in objs], device="cuda")
lpb = lp_upper(gg, cc, A, b, 800, 0.05)
order = torch.argsort(lpb, descending=True)[:3].tolist()
sel = [p for p in phases if int(p.idx.numel())][-kl:] if kl > 0 else []
nb = sum(int(p.idx.numel()) for p in sel)
An = sp.csr_matrix(A.double().cpu().numpy()); bn = b.double().cpu().numpy()
blocks = []; rhs = []; col = 0
Mrows = []
for p in sel:
    m = int(p.idx.numel())
    gx = p.gx.double().cpu().numpy(); cx = p.cx.double().cpu().numpy(); gy = p.gy.double().cpu().numpy(); cy = p.cy.double().cpu().numpy()
    l = p.l.double().cpu().numpy(); u = p.u.double().cpu().numpy()
    gxp = np.zeros((m, K)); gxp[:, :gx.shape[1]] = gx
    gyp = np.zeros((m, K)); gyp[:, :gy.shape[1]] = gy
    D = np.zeros((m, nb)); 
    # y - u d <= 0  ->  gy w - u d <= -cy
    D1 = np.zeros((m, nb)); D1[np.arange(m), col + np.arange(m)] = -u
    Mrows.append((np.hstack([gyp, D1]), -cy))
    # y - x + l(1-d) <= 0 -> (gy - gx) w - l d <= cx - cy - l
    D2 = np.zeros((m, nb)); D2[np.arange(m), col + np.arange(m)] = -l
    Mrows.append((np.hstack([gyp - gxp, D2]), cx - cy - l))
    col += m
AA = sp.hstack([An, sp.csr_matrix((An.shape[0], nb))])
if Mrows:
    AA = sp.vstack([AA] + [sp.csr_matrix(r[0]) for r in Mrows]); bb = np.concatenate([bn] + [r[1] for r in Mrows])
else:
    bb = bn
integ = np.concatenate([np.zeros(K), np.ones(nb)])
bnds = Bounds(np.concatenate([-np.ones(K), np.zeros(nb)]), np.concatenate([np.ones(K), np.ones(nb)]))
print(f"row {row}: last {kl} ReLU layers exact -> {nb} binaries; LP bounds {[round(float(lpb[k]),4) for k in order]}", flush=True)
for k in order:
    t0 = time.time()
    obj = np.concatenate([-gg[k].double().cpu().numpy(), np.zeros(nb)])
    res = milp(obj, constraints=LinearConstraint(AA, -np.inf, bb), integrality=integ, bounds=bnds,
               options={"time_limit": tl, "disp": False})
    dual_bound = -res.mip_dual_bound + float(cc[k]) if getattr(res, "mip_dual_bound", None) is not None else None
    prim = (-res.fun + float(cc[k])) if res.fun is not None else None
    print(f"  disjunct {k}: LP {float(lpb[k]):.4f}  MILP upper(dual bound) {dual_bound}  incumbent {prim}  status {res.status} {time.time()-t0:.1f}s", flush=True)
