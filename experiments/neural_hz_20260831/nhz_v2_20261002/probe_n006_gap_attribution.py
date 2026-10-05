"""N006 probe: attribute the terminal LP bound to factor groups.

At the final multipliers nu of the worst disjunct, the dual bound is
  c + nu^T b + sum_k |(g - A^T nu)_k| .
The box term of each eta_j is the price paid for neuron j's triangle
relaxation (its vertical slack); the box terms of input factors are the price
of the input region; nu^T b collects the row offsets.  We report the eta
contribution per ReLU layer and the top individual neurons.  Diagnostic only.
"""

import sys, os, csv, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import Engine, parse_vnnlib

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"


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


fam, row = sys.argv[1], int(sys.argv[2])
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
onnx_rel, spec_rel = inst[row][:2]
e = Engine(os.path.normpath(f"{ROOT}/{fam}/{onnx_rel}"), "cuda", torch.float32)
z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{fam}/{spec_rel}"), int(np.prod(e.input_shape)), int(o0.c.numel()))
lb, ub = spec.boxes[0]
out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 400, 0.05)
c = out.c.reshape(-1); G = out.G.reshape(K, c.numel())
A, b = rows.dense(K)
worst = None
for atoms in spec.disjuncts:
    a0, b0 = atoms[0]
    a = torch.as_tensor(a0, device="cuda", dtype=torch.float32)
    g = -(G @ a).unsqueeze(0); cc = float(b0 - c @ a)
    val, nu = solve_nu(g, cc, A, b, 1500, 0.05)
    if worst is None or val > worst[0]:
        worst = (val, g, cc, nu, atoms)
val, g, cc, nu, atoms = worst
resid = (g - nu @ A).reshape(-1).abs()
n_in = int(e.input_factor_index.numel())
print(f"worst disjunct bound {val:.4f}  c={cc:.4f}  nu^T b={float(nu @ b):.4f}  input-box={float(resid[:n_in].sum()):.4f}  eta-box={float(resid[n_in:].sum()):.4f}")
for p in phases:
    m = int(p.idx.numel())
    seg = resid[p.eta0:p.eta0 + m]
    print(f"  {p.layer:24s} unstable={m:5d}  eta contribution={float(seg.sum()):8.4f}  top5={[round(float(v),4) for v in torch.topk(seg, min(5, m)).values] if m else []}")
