"""N044: sign-aware terminal v5 vs full binaries (terminal v4 plan) on the SafeNLP rows that
N039 v2 lost at the 20 s budget.  Capability/diagnostic probe, same engine n009.2."""
import sys, os, csv, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v5 import plan_milp_v5
import onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.05)
csv.field_size_limit(10**9)
ov = {(r['benchmark'], r['iid']): r for r in csv.DictReader(open('/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv'))}
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
eng = {}
out = open(sys.argv[2], "a")
for iid in sys.argv[1].split(","):
    r = ov[('safenlp_2024', iid)]
    mp = os.path.normpath(os.path.join(ROOT, 'safenlp_2024', r['onnx'])); spp = os.path.normpath(os.path.join(ROOT, 'safenlp_2024', r['vnnlib']))
    if mp not in eng:
        eng.clear(); eng[mp] = (SoundEngineV9(mp, 'cuda'), ort.InferenceSession(mp, providers=['CPUExecutionProvider']))
    e, s = eng[mp]
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(spp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
    for sign in (True, False):
        m = plan_milp_v5(A, b, ph, K, g, cc, pad, extra, 1, 99, 20.0, sign_aware=sign)
        ok = None
        if m["incumbent_w"] is not None:
            nzi = e.input_factor_index.cpu().numpy(); xi = np.zeros(n_in); xi[nzi] = m["incumbent_w"][:nzi.size]
            x = np.clip((lb + ub) / 2 + (ub - lb) / 2 * xi, lb, ub).astype(np.float32)
            y = s.run(None, {s.get_inputs()[0].name: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
            ok = any(all(float(a_ @ y) <= bb for a_, bb in d) for d in spec.disjuncts)
        rec = {"iid": int(iid), "baseline": r["raw_verdict"], "sign_aware": sign, "binaries": m["binaries"],
               "units": m["gamma_units"], "excluded": m["excluded"], "status": m["status"], "wall_s": m["wall_s"],
               "upper": m["upper"], "incumbent_violation": m["incumbent_violation"], "ort_valid": ok,
               "n_disjuncts": len(spec.disjuncts), "atoms": len(spec.disjuncts[0])}
        out.write(json.dumps(rec, default=float) + "\n"); out.flush()
        print(rec, flush=True)
