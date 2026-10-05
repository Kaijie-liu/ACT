"""N030 diagnostic: dist_shift rows that the lambda level cannot certify; MILP with
all ReLU binaries exact and sigmoid at the lambda level (float engine n005.3)."""
import sys, os, csv, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_attn import AttnEngine
from nhz_engine import parse_vnnlib, lp_upper
from run_n011_partial_milp import build_partial
from scipy.optimize import milp, LinearConstraint, Bounds
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/dist_shift_2023"
torch.cuda.set_per_process_memory_fraction(0.1)
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
eng = {}
for ri in [int(x) for x in sys.argv[1].split(",")]:
    o, s = inst[ri][:2]; mp = os.path.normpath(f"{ROOT}/{o}")
    if mp not in eng: eng[mp] = AttnEngine(mp, "cuda", torch.float32)
    e = eng[mp]
    z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
    spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{s}"), int(np.prod(e.input_shape)), int(o0.c.numel())); lb, ub = spec.boxes[0]
    out, rows, K, ph = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
    c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
    gg = torch.stack([-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts])
    cc = torch.as_tensor([d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts], device="cuda")
    lpb = lp_upper(gg, cc, A, b, 1000, 0.05)
    open_ids = [int(k) for k in torch.argsort(lpb, descending=True) if float(lpb[k]) >= 0]
    AA, bb, nb = build_partial(A, b, ph, K, 99)
    t0 = time.time(); res_all = []
    for k in open_ids:
        res = milp(np.concatenate([-gg[k].double().cpu().numpy(), np.zeros(nb)]), constraints=LinearConstraint(AA, -np.inf, bb),
                   integrality=np.concatenate([np.zeros(K), np.ones(nb)]), bounds=Bounds(np.concatenate([-np.ones(K), np.zeros(nb)]), np.ones(K + nb)),
                   options={"time_limit": 60, "disp": False})
        db = getattr(res, "mip_dual_bound", None)
        res_all.append(round((-db + float(cc[k])) if db is not None else float("nan"), 4))
        if res_all[-1] >= 0: break
    print(ri, "LP", round(float(lpb.max()), 4), "open", len(open_ids), "ReLU binaries", nb, "MILP uppers", res_all, round(time.time() - t0, 1), "s", flush=True)
