import sys, os, csv, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_attn import AttnEngine
from nhz_engine import parse_vnnlib, lp_upper
from run_n011_partial_milp import build_partial
from scipy.optimize import milp, LinearConstraint, Bounds
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/vit_2023"
ri, layers, tl = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
torch.cuda.set_per_process_memory_fraction(0.15)
inst = list(csv.reader(open(f"{ROOT}/instances.csv"))); o, s = inst[ri][:2]
e = AttnEngine(f"{ROOT}/{o}", "cuda", torch.float32)
spec = parse_vnnlib(f"{ROOT}/{s}", 3072, 10); lb, ub = spec.boxes[0]
out, rows, K, ph = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
gg = torch.stack([-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts])
cc = torch.as_tensor([d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts], device="cuda")
lpb = lp_upper(gg, cc, A, b, 1000, 0.05); k = int(torch.argmax(lpb))
print("row", ri, "K", K, "rows", A.shape[0], "phase blocks", [int(p.idx.numel()) for p in ph], "worst", k, round(float(lpb[k]), 4), flush=True)
AA, bb, nb = build_partial(A, b, ph, K, layers)
t0 = time.time()
res = milp(np.concatenate([-gg[k].double().cpu().numpy(), np.zeros(nb)]), constraints=LinearConstraint(AA, -np.inf, bb),
           integrality=np.concatenate([np.zeros(K), np.ones(nb)]), bounds=Bounds(np.concatenate([-np.ones(K), np.zeros(nb)]), np.ones(K + nb)),
           options={"time_limit": tl, "disp": False})
db = getattr(res, "mip_dual_bound", None)
print("exact layers", layers, "binaries", nb, "MILP upper", (-db + float(cc[k])) if db is not None else None, "status", res.status, round(time.time() - t0, 1), "s")
