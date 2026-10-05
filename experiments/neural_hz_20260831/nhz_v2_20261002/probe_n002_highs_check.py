"""Ground-truth check of the GPU dual LP bounds against Gurobi on one row."""
import sys, os, csv, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_shadow import Graph, parse_vnnlib_box_top1
from gpu_aligned_lp import run, lp_upper_bounds
from run_n001_shadow_census import EVID, ROOT

fam, row, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
onnx_rel, spec_rel, _ = inst[row]
g = Graph(f"{ROOT}/{fam}/{onnx_rel}", "cuda", torch.float32)
in_shape = (3, 32, 32) if "CIFAR" in onnx_rel else (3, 56, 56)
lb, ub, t, others = parse_vnnlib_box_top1(f"{ROOT}/{fam}/{spec_rel}", int(np.prod(in_shape)))
lbt = torch.as_tensor(lb.reshape(in_shape), device="cuda", dtype=torch.float32)
ubt = torch.as_tensor(ub.reshape(in_shape), device="cuda", dtype=torch.float32)
out, rs, K = run(g, lbt, ubt, iters=iters, lr=0.05)
out = out.pad_to(K); c = out.c.reshape(-1); G = out.G.reshape(K, -1)
A, b = rs.dense(K)
gobj = (G[:, others] - G[:, t:t+1]).t().contiguous(); cobj = c[others] - c[t]
# worst few margins by shadow
sh = cobj + gobj.abs().sum(1)
order = torch.argsort(sh, descending=True)[:3]
for lr_, it_ in [(0.05, 400), (0.02, 2000), (0.1, 2000)]:
    t0 = time.time(); v = lp_upper_bounds(gobj[order], cobj[order], A, b, it_, lr_); torch.cuda.synchronize()
    print('gpu dual lr', lr_, 'iters', it_, [round(float(x), 4) for x in v], round(time.time()-t0, 2), 's')
An = A.double().cpu().numpy(); bn = b.double().cpu().numpy()
print('LP size rows', An.shape, 'nnz', int((An != 0).sum()))
from scipy.optimize import linprog
for j in order.tolist():
    t0 = time.time()
    res = linprog(-gobj[j].double().cpu().numpy(), A_ub=An, b_ub=bn, bounds=(-1, 1), method="highs")
    print('highs max (Y_j - Y_t) for', others[j], round(-res.fun + float(cobj[j]), 5), res.status, round(time.time()-t0, 2), 's')
