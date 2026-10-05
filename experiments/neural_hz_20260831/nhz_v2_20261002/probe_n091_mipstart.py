"""N091: MIP start from the element centre (true phases at w = 0) on exact all-layers plans;
ACAS Xu ADV rows lost by v11.  60 s per variant; ORT check of the incumbent."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v17 import SoundEngineV17
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v8 import plan_milp_v8
from run_n039_full_replay import universe, ROOT
import onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.05)
U = {(r["family"], r["iid"]): r for r in universe()}
out = open(sys.argv[2], "a"); tl = float(sys.argv[3])
for spec_ in sys.argv[1].split(","):
    fam, iid = spec_.split(":"); r = U[(fam, int(iid))]
    mp = os.path.normpath(os.path.join(ROOT, fam, r["onnx"])); sp = os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"]))
    e = SoundEngineV17(mp, "cuda"); s = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(sp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    nl = sum(1 for p in ph if int(p["idx"].numel())); K0 = int(e.input_factor_index.numel())
    nzi = e.input_factor_index.cpu().numpy(); c = (lb + ub) / 2; rd = (ub - lb) / 2
    def ort_ok(w):
        xi = np.zeros(n_in); xi[nzi] = np.asarray(w)[:nzi.size]; x = np.clip(c + rd * xi, lb, ub).astype(np.float32)
        y = s.run(None, {s.get_inputs()[0].name: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
        return any(all(float(a @ y) <= bb for a, bb in d) for d in spec.disjuncts)
    for di, atoms in enumerate(spec.disjuncts):
        g, cc, pad, extra, T = epigraph_objective(o, K, atoms)
        for v in ("nostart", "centre_start"):
            t0 = time.time()
            m = plan_milp_v8(A, b, ph, e.smooth_phases, K, g, cc, pad, extra, nl, 99, tl, start_w=(np.zeros(K0) if v == "centre_start" else None))
            ok = ort_ok(m["incumbent_w"]) if m["incumbent_w"] is not None else None
            rec = {"row": spec_, "disjunct": di, "variant": v, "status": m["status"][:80], "excluded": m["excluded"], "wall_s": time.time() - t0,
                   "incumbent_violation": m.get("incumbent_violation"), "ort_valid": ok}
            out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
