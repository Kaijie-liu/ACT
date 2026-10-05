"""N116 (ViT softmax coupling, engine n021 vs n020.1); derived from N054: terminal LP bound (rigorous, 1000 iterations) of engine n009.2 vs n011 (per-coordinate
softmax enclosure choice) on ViT rows; diagnostic."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_sound_v11 import SoundEngineV11
from nhz_sound_v11b import SoundEngineV11b
from nhz_sound_v20 import SoundEngineV20
from nhz_sound_v21 import SoundEngineV21
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(float(sys.argv[3]))
U = {(r["family"], r["iid"]): r for r in universe()}
out = open(sys.argv[2], "a")
for iid in [int(x) for x in sys.argv[1].split(",")]:
    r = U[("vit_2023", iid)]
    mp = os.path.normpath(os.path.join(ROOT, "vit_2023", r["onnx"])); sp = os.path.normpath(os.path.join(ROOT, "vit_2023", r["vnnlib"]))
    for name, cls in (("n020.1", SoundEngineV20), ("n021.0", SoundEngineV21)):
        e = cls(mp, "cuda"); n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
        spec = parse_vnnlib(sp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
        t0 = time.time(); o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); tp = time.time() - t0
        A, b = rows.dense(K); A = A.to(o.c.device); b = b.to(o.c.device)
        ups = []
        for atoms in spec.disjuncts:
            g, cc, pad, extra, T = epigraph_objective(o, K, atoms)
            u_, _ = lp_bound_epigraph(A, b, g, cc, extra, 1000, 0.05); ups.append(u_ + pad)
        rec = {"iid": iid, "baseline": r["baseline"], "engine": name, "lp_worst": max(ups), "n_open": sum(u >= -1e-4 for u in ups),
               "unstable": sum(int(p["idx"].numel()) for p in ph), "K": K, "propagate_s": tp,
               "softmax_interval_coords": getattr(e, "softmax_interval_coords", None), "softmax_coords": getattr(e, "softmax_coords", None), "coupled": [getattr(e, "sm_lp_tightened", None), getattr(e, "sm_ratio_rows", None)], "engine_version": name}
        out.write(json.dumps(rec) + "\n"); out.flush(); print(rec, flush=True)
        del e, o, rows, A, b; torch.cuda.empty_cache()
