"""N014 diagnostic: per-layer level plan for the terminal partial MILP.

Phases are grouped by ReLU layer (creation order).  The plan keeps integer
binaries for the last `g` layers (gamma), LP rows for the last `r` layers
(lambda), and only the eta box (sigma = DeepZ) for earlier layers.  Every plan is
a relaxation of the exact state, hence a sound bound.  Structural (layer
position) choice only.
"""
import sys, os, time, csv
import numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import Engine, parse_vnnlib, lp_upper
from scipy.optimize import milp, LinearConstraint, Bounds
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam, row = sys.argv[1], int(sys.argv[2]); plans = [tuple(int(v) for v in p.split(":")) for p in sys.argv[3].split(",")]
tl = float(sys.argv[4])
torch.cuda.set_per_process_memory_fraction(0.2)
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv"))); o, s = inst[row][:2]
e = Engine(os.path.normpath(f"{ROOT}/{fam}/{o}"), "cuda", torch.float32)
z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{fam}/{s}"), int(np.prod(e.input_shape)), int(o0.c.numel()))
lb, ub = spec.boxes[0]
out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
gg = torch.stack([-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts])
cc = torch.as_tensor([d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts], device="cuda")
lpb = lp_upper(gg, cc, A, b, 1000, 0.05)
k = int(torch.argmax(lpb)); print("row", row, "worst disjunct", k, "LP", round(float(lpb[k]), 4), flush=True)
blocks = [p for p in phases if int(p.idx.numel())]
# row ranges: rows were added two per phase block in order
ranges = []; r0 = 0
for p in phases:
    m = int(p.idx.numel()); ranges.append((r0, r0 + 2 * m)); r0 += 2 * m
nzb = [i for i, p in enumerate(phases) if int(p.idx.numel())]
An = A.double().cpu().numpy(); bn = b.double().cpu().numpy()
for (gl, rl) in plans:
    keep_rows = np.zeros(An.shape[0], bool)
    for i in nzb[-rl:] if rl > 0 else []:
        a0, a1 = ranges[i]; keep_rows[a0:a1] = True
    sel = [phases[i] for i in (nzb[-gl:] if gl > 0 else [])]
    nb = sum(int(p.idx.numel()) for p in sel)
    Ab = sp.csr_matrix(An[keep_rows]); bb = bn[keep_rows]
    blocks_ = [sp.hstack([Ab, sp.csr_matrix((Ab.shape[0], nb))])]; rhs = [bb]; col = 0
    for p in sel:
        m = int(p.idx.numel())
        gx = np.zeros((m, K)); g0 = p.gx.double().cpu().numpy(); gx[:, :g0.shape[1]] = g0
        gy = np.zeros((m, K)); g1 = p.gy.double().cpu().numpy(); gy[:, :g1.shape[1]] = g1
        cx = p.cx.double().cpu().numpy(); cy = p.cy.double().cpu().numpy(); l = p.l.double().cpu().numpy(); u = p.u.double().cpu().numpy()
        D1 = sp.csr_matrix((-u, (np.arange(m), col + np.arange(m))), shape=(m, nb)); D2 = sp.csr_matrix((-l, (np.arange(m), col + np.arange(m))), shape=(m, nb))
        blocks_ += [sp.hstack([sp.csr_matrix(gy), D1]), sp.hstack([sp.csr_matrix(gy - gx), D2])]; rhs += [-cy, cx - cy - l]; col += m
    AA = sp.vstack(blocks_).tocsr(); BB = np.concatenate(rhs)
    obj = np.concatenate([-gg[k].double().cpu().numpy(), np.zeros(nb)])
    t0 = time.time()
    res = milp(obj, constraints=LinearConstraint(AA, -np.inf, BB), integrality=np.concatenate([np.zeros(K), np.ones(nb)]),
               bounds=Bounds(np.concatenate([-np.ones(K), np.zeros(nb)]), np.ones(K + nb)), options={"time_limit": tl, "disp": False})
    db = getattr(res, "mip_dual_bound", None)
    if db is None or nb == 0:
        db = res.fun if res.fun is not None else float("nan")
    print(f"  plan gamma-layers={gl} lambda-layers={rl}: rows {AA.shape[0]} bin {nb}  upper {(-db + float(cc[k])):.4f}  status {res.status}  {time.time()-t0:.1f}s", flush=True)
