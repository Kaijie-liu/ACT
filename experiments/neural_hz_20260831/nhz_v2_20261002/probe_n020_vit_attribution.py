import sys, os, csv
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_attn import AttnEngine
from nhz_engine import parse_vnnlib, lp_upper


def solve_nu(g, c, A, b, iters, lr):
    nu = g.new_zeros((1, A.shape[0])); m1 = torch.zeros_like(nu); m2 = torch.zeros_like(nu)
    best = None; best_nu = nu.clone()
    for it in range(1, iters + 1):
        resid = g - nu @ A
        val = float(c + nu @ b + resid.abs().sum())
        if best is None or val < best:
            best, best_nu = val, nu.clone()
        grad = b.unsqueeze(0) - torch.sign(resid) @ A.t()
        m1.mul_(0.9).add_(grad, alpha=0.1); m2.mul_(0.999).addcmul_(grad, grad, value=0.001)
        nu = (nu - lr * (m1 / (1 - 0.9 ** it)) / ((m2 / (1 - 0.999 ** it)).sqrt() + 1e-8)).clamp_(min=0)
    return best, best_nu
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/vit_2023"
torch.cuda.set_per_process_memory_fraction(0.15)
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
for ri in [int(x) for x in sys.argv[1].split(",")]:
    o, s = inst[ri][:2]
    e = AttnEngine(f"{ROOT}/{o}", "cuda", torch.float32)
    spec = parse_vnnlib(f"{ROOT}/{s}", 3072, 10); lb, ub = spec.boxes[0]
    out, rows, K, ph = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
    c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
    gg = torch.stack([-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts])
    cc = torch.as_tensor([d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts], device="cuda")
    k = int(torch.argmax(lp_upper(gg, cc, A, b, 600, 0.05)))
    val, nu = solve_nu(gg[k:k+1], float(cc[k]), A, b, 1500, 0.05)
    resid = (gg[k:k+1] - nu @ A).reshape(-1).abs()
    tot = {}
    for kind, s0, n in e.factor_groups:
        tot[kind] = tot.get(kind, 0.0) + float(resid[s0:s0 + n].sum())
    print(ri, "bound", round(val, 4), "c", round(float(cc[k]), 4), "nu^T b", round(float(nu @ b), 4), {k_: round(v, 4) for k_, v in tot.items()},
          "groups", [(g[0], g[2]) for g in e.factor_groups])
