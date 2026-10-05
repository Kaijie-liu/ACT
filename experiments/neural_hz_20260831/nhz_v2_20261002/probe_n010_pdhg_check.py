import sys, os, time, csv
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import Engine, parse_vnnlib, lp_upper
from lp_pdhg import pdhg_upper
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam, row = sys.argv[1], int(sys.argv[2])
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv"))); o, s = inst[row][:2]
e = Engine(os.path.normpath(f"{ROOT}/{fam}/{o}"), "cuda", torch.float32)
z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{fam}/{s}"), int(np.prod(e.input_shape)), int(o0.c.numel()))
lb, ub = spec.boxes[0]
out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
c = out.c.reshape(-1); G = out.G.reshape(K, c.numel()); A, b = rows.dense(K)
gg = torch.stack([-(G @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts])
cc = torch.as_tensor([d[0][1] - float(c @ torch.as_tensor(d[0][0], device="cuda", dtype=torch.float32)) for d in spec.disjuncts], device="cuda")
t0 = time.time(); adam = lp_upper(gg, cc, A, b, 1000, 0.05); torch.cuda.synchronize(); ta = time.time() - t0
k = torch.argsort(adam, descending=True)[:3]
for it in (500, 2000, 5000):
    t0 = time.time(); pb, _, _ = pdhg_upper(gg, cc, A, b, it); torch.cuda.synchronize()
    print(f"PDHG {it} it ({time.time()-t0:.2f}s, all {gg.shape[0]} objectives): top {[round(float(pb[i]),4) for i in k]}")
print(f"Adam 1000 it ({ta:.2f}s): top {[round(float(adam[i]),4) for i in k]}")
