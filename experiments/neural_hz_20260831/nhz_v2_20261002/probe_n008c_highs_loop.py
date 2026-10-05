"""Diagnostic: exact LP value of the base relaxation plus all violated pair-hull rows (HiGHS loop)."""
import sys, os, time, csv
import numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_n008_pair_hull as P
from nhz_engine import Engine, parse_vnnlib, lp_upper
from scipy.optimize import linprog
fam, row, partners = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
inst = list(csv.reader(open(f"{P.ROOT}/{fam}/instances.csv"))); o, s = inst[row][:2]
e = Engine(os.path.normpath(f"{P.ROOT}/{fam}/{o}"), "cuda", torch.float32)
z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
spec = parse_vnnlib(os.path.normpath(f"{P.ROOT}/{fam}/{s}"), int(np.prod(e.input_shape)), int(o0.c.numel()))
lb, ub = spec.boxes[0]
out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
objs = [(-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)), d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32))) for d in spec.disjuncts]
gg = torch.stack([x[0] for x in objs]); cc = torch.as_tensor([x[1] for x in objs], device="cuda")
k = int(torch.argmax(lp_upper(gg, cc, A, b, 600, 0.05)))
obj = -gg[k].double().cpu().numpy(); off = float(cc[k])
pool = []
for p in phases:
    m = int(p.idx.numel())
    if m < 2: continue
    gx = p.gx.double().cpu().numpy(); cx = p.cx.double().cpu().numpy(); gy = p.gy.double().cpu().numpy(); cy = p.cy.double().cpu().numpy()
    l = p.l.double().cpu().numpy(); u = p.u.double().cpu().numpy()
    gn = p.gx / (p.gx.norm(dim=1, keepdim=True) + 1e-30); sim = (gn @ gn.t()).abs(); sim.fill_diagonal_(-1)
    top = torch.topk(sim, min(partners, m - 1), dim=1).indices.cpu().numpy()
    seen = set()
    for i in range(m):
        for j in top[i]:
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen: continue
            seen.add(key)
            pool += P.pair_rows(cx, gx, cy, gy, [(l[i], u[i]), (l[j], u[j])], i, int(j), K)
PA = sp.csr_matrix(np.stack([r[0] for r in pool])); Pb = np.array([r[1] for r in pool])
print("pool rows", PA.shape[0], "nnz", PA.nnz, flush=True)
An = sp.csr_matrix(A.double().cpu().numpy()); bn = b.double().cpu().numpy()
act = np.zeros(PA.shape[0], bool)
for rnd in range(8):
    t0 = time.time()
    AA = sp.vstack([An, PA[act]]) if act.any() else An
    bb = np.concatenate([bn, Pb[act]])
    res = linprog(obj, A_ub=AA, b_ub=bb, bounds=(-1, 1), method="highs")
    val = -res.fun + off
    v = PA @ res.x - Pb
    new = (v > 1e-7) & ~act
    print(f"round {rnd}: LP value {val:.5f}  active pair rows {int(act.sum())}  newly violated {int(new.sum())}  {time.time()-t0:.1f}s", flush=True)
    if not new.any(): break
    act |= new
