"""N065: dist_shift with smooth rows in the MILP plan (terminal v8) and optional phase segments.
Variants per row (all ReLU layers gamma, 300 s budget each, 4 seeds):
  v7     : terminal v7 (smooth rows dropped, as in every earlier plan)
  v8     : smooth rows written sparsely, no segments
  v8seg2 : + 2 segments (split at the inflection) on every non-saturated sigmoid unit"""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v16 import SoundEngineV16
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph
from nhz_terminal_v7 import plan_milp_v7
from nhz_terminal_v8 import plan_milp_v8
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.05)
U = {(r["family"], r["iid"]): r for r in universe()}
variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["v7", "v8", "v8seg2"]
tl = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
out = open(sys.argv[2], "a")
for iid in [int(x) for x in sys.argv[1].split(",")]:
    r = U[("dist_shift_2023", iid)]
    e = SoundEngineV16(os.path.normpath(os.path.join(ROOT, "dist_shift_2023", r["onnx"])), "cuda")
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, "dist_shift_2023", r["vnnlib"])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    nl = sum(1 for p in ph if int(p["idx"].numel()))
    g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
    sm = e.smooth_phases
    lsm = torch.cat([s_["l"] for s_ in sm]).cpu().numpy(); usm = torch.cat([s_["u"] for s_ in sm]).cpu().numpy()
    nonsat = (1 / (1 + np.exp(-usm)) - 1 / (1 + np.exp(-lsm))) > 1e-4
    for v in variants:
        t0 = time.time()
        if v == "v7":
            m = plan_milp_v7(A, b, ph, K, g, cc, pad, extra, nl, 99, tl)
        else:
            seg = None if v == "v8" else nonsat
            m = plan_milp_v8(A, b, ph, sm, K, g, cc, pad, extra, nl, 99, tl, seg_select=seg, K_seg=int(v[-1]) if "seg" in v else 2)
        rec = {"iid": iid, "baseline": r["baseline"], "variant": v, "excluded": m["excluded"], "upper": m["upper"],
               "status": m["status"], "binaries": m["binaries"], "wall_s": time.time() - t0, "nnz": m.get("nnz"),
               "segment_units": m.get("segment_units"), "nonsat": int(nonsat.sum())}
        out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
