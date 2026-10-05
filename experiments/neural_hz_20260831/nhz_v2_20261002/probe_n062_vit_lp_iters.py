"""N062: is the GPU weak-duality LP converged on ViT IBP rows?  Terminal LP bound with 1000,
5000, 20000 iterations (engine n014) and the number of rows."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v14 import SoundEngineV14
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.12)
U = {(r["family"], r["iid"]): r for r in universe()}
for iid in [int(x) for x in sys.argv[1].split(",")]:
    r = U[("vit_2023", iid)]
    e = SoundEngineV14(os.path.normpath(os.path.join(ROOT, "vit_2023", r["onnx"])), "cuda")
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, "vit_2023", r["vnnlib"])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    for it_prop in (300, 1500):
        o, rows, K, ph = e.propagate(lb, ub, it_prop, 0.05); A, b = rows.dense(K); A = A.to(o.c.device); b = b.to(o.c.device)
        res = {}
        for it in (1000, 5000, 20000):
            ups = []
            for atoms in spec.disjuncts:
                g, cc, pad, extra, T = epigraph_objective(o, K, atoms)
                u_, _ = lp_bound_epigraph(A, b, g, cc, extra, it, 0.05); ups.append(u_ + pad)
            res[it] = round(max(ups), 4)
        print(iid, "prop_iters", it_prop, "rows", A.shape[0], "K", K, "unstable", sum(int(p["idx"].numel()) for p in ph), res, flush=True)
