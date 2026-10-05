import sys, os, csv, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_mp2 import SoundEngineMP2
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph, plan_milp_highs_v3
import onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.1)
fam = sys.argv[1]; ROOT = f"/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/{fam}"
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
for ri in [int(x) for x in sys.argv[2].split(",")]:
    o, s = inst[ri][:2]
    e = SoundEngineMP2(os.path.normpath(f"{ROOT}/{o}"), "cuda"); sess = ort.InferenceSession(os.path.normpath(f"{ROOT}/{o}"), providers=["CPUExecutionProvider"])
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{s}"), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    t0 = time.time()
    out, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    nzi = e.input_factor_index.cpu().numpy()
    res = []
    for atoms in spec.disjuncts:
        g, cc, pad, extra, T = epigraph_objective(out, K, atoms)
        ubd, cand = lp_bound_epigraph(A.to("cuda"), b.to("cuda"), g, cc, extra, 1000, 0.05)
        m = plan_milp_highs_v3(A, b, ph, K, g, cc, pad, extra, 99, 99, 60)
        w = m["incumbent_w"]; ok = None
        if w is not None:
            xi = np.zeros(n_in); xi[nzi] = w[: nzi.size]; x = np.clip((lb + ub) / 2 + (ub - lb) / 2 * xi, lb, ub).astype(np.float32)
            y = sess.run(None, {sess.get_inputs()[0].name: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
            ok = any(all(float(a @ y) <= bb for a, bb in d) for d in spec.disjuncts)
        res.append((round(ubd, 4), m["status"], round(m["upper"], 4) if m["upper"] is not None else None, ok))
    print(fam, ri, res, round(time.time() - t0, 1), "s", flush=True)
