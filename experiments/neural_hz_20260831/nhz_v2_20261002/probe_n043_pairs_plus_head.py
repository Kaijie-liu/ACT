"""N043 diagnostic: last-layer MILP (gamma head) + zonogon pair-hull rows on the k ReLU layers
before it (structural: layer position and |cosine| partner), float engine n003, HiGHS."""
import sys, os, csv, time
import numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import Engine, parse_vnnlib, lp_upper
from run_n011_partial_milp import build_partial
import probe_n008_pair_hull as P
from scipy.optimize import milp, LinearConstraint, Bounds
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024"
torch.cuda.set_per_process_memory_fraction(0.15)
kpre, tl = int(sys.argv[2]), float(sys.argv[3])
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
for ri in [int(x) for x in sys.argv[1].split(",")]:
    o, s = inst[ri][:2]
    e = Engine(f"{ROOT}/{o}", "cuda", torch.float32)
    z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
    spec = parse_vnnlib(f"{ROOT}/{s}", int(np.prod(e.input_shape)), int(o0.c.numel())); lb, ub = spec.boxes[0]
    out, rows, K, ph = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
    c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
    gg = torch.stack([-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts])
    cc = torch.as_tensor([d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts], device="cuda")
    k = int(torch.argmax(lp_upper(gg, cc, A, b, 1000, 0.05)))
    AA, bb, nb = build_partial(A, b, ph, K, 1)
    nzp = [p for p in ph if int(p.idx.numel())]
    pool = []
    for p in nzp[-1 - kpre:-1]:
        m = int(p.idx.numel())
        if m < 2: continue
        gx = p.gx.double().cpu().numpy(); cx = p.cx.double().cpu().numpy(); gy = p.gy.double().cpu().numpy(); cy = p.cy.double().cpu().numpy()
        l = p.l.double().cpu().numpy(); u = p.u.double().cpu().numpy()
        gn = p.gx / (p.gx.norm(dim=1, keepdim=True) + 1e-30); sim = (gn @ gn.t()).abs(); sim.fill_diagonal_(-1)
        top = torch.topk(sim, min(2, m - 1), dim=1).indices.cpu().numpy(); seen = set()
        for i in range(m):
            for j in top[i]:
                key = (min(i, int(j)), max(i, int(j)))
                if key in seen: continue
                seen.add(key); pool += P.pair_rows(cx, gx, cy, gy, [(l[i], u[i]), (l[j], u[j])], i, int(j), K)
    obj = np.concatenate([-gg[k].double().cpu().numpy(), np.zeros(nb)])
    integ = np.concatenate([np.zeros(K), np.ones(nb)]); bnds = Bounds(np.concatenate([-np.ones(K), np.zeros(nb)]), np.ones(K + nb))
    res = []
    for label, extraA in (("head only", None), (f"head + pairs({kpre} layers, {len(pool)} rows)", pool)):
        if extraA:
            PA = sp.hstack([sp.csr_matrix(np.stack([r_[0] for r_ in extraA])), sp.csr_matrix((len(extraA), nb))])
            M = sp.vstack([AA, PA]).tocsr(); rhs = np.concatenate([bb, np.array([r_[1] for r_ in extraA])])
        else:
            M, rhs = AA, bb
        t0 = time.time()
        r = milp(obj, constraints=LinearConstraint(M, -np.inf, rhs), integrality=integ, bounds=bnds, options={"time_limit": tl, "disp": False})
        db = getattr(r, "mip_dual_bound", None)
        res.append((label, round((-db + float(cc[k])) if db is not None else float("nan"), 4), r.status, round(time.time() - t0, 1)))
    print(ri, res, flush=True)
