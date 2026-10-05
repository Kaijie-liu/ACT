"""N069: E0 CIFAR rows closest to exclusion: MILP upper bound at selective-exactness levels
gamma = 1, 2, 3 (last 1/2/3 ReLU layers exact, all layers lambda), 60 s each, terminal v8.1,
engine n017.  Diagnostic for a stage between D and E."""
import sys, os, csv, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v17 import SoundEngineV17
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph
from nhz_terminal_v8 import plan_milp_v8
torch.cuda.set_per_process_memory_fraction(0.2)
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam = sys.argv[1]; rows_ = [int(x) for x in sys.argv[2].split(",")]; out = open(sys.argv[3], "a"); tl = float(sys.argv[4])
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
eng = {}
for ri in rows_:
    o_, s_, _ = inst[ri]; mp = f"{ROOT}/{fam}/{o_}"; sp = f"{ROOT}/{fam}/{s_}"
    if mp not in eng:
        eng.clear(); torch.cuda.empty_cache(); eng[mp] = SoundEngineV17(mp, "cuda")
    e = eng[mp]; n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(sp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    t0 = time.time(); o, rset, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rset.dense(K); A = A.to(o.c.device); b = b.to(o.c.device)
    tp = time.time() - t0
    ups = []
    for atoms in spec.disjuncts:
        g, cc, pad, extra, T = epigraph_objective(o, K, atoms); u_, _ = lp_bound_epigraph(A, b, g, cc, extra, 1000, 0.05); ups.append(u_ + pad)
    i = int(np.argmax(ups)); g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[i])
    nz = [p for p in ph if int(p["idx"].numel())]
    res = {"row": ri, "lp": max(ups), "n_open": int(sum(u >= -1e-4 for u in ups)), "units_per_layer": [int(p["idx"].numel()) for p in nz], "propagate_s": tp}
    for gl in (1, 2, 3):
        m = plan_milp_v8(A, b, ph, [], K, g, cc, pad, extra, gl, 99, tl, target=None)
        res[f"g{gl}"] = (m["upper"], m["excluded"], m["binaries"], round(m["wall_s"], 1))
    out.write(json.dumps(res, default=float) + "\n"); out.flush(); print(res, flush=True)
    del o, rset, A, b; torch.cuda.empty_cache()
