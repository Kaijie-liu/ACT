"""N080: effect of the exact-LP polishing caps (n009.2: <=100 unstable per layer, <=600 LPs per
propagation, first 25% of the budget) on rows where the MILP is weak.  Variant 'nocap' keeps only
the 25% time share.  Runs the full path v6.4 per row with the official budget."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhz_sound_v9 as V9
from nhz_sound_v17 import SoundEngineV17
from nhz_engine import parse_vnnlib
from nhz_path_v6 import solve_box_v6
from run_n039_full_replay import universe, ROOT
import onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.1)
U = {(r["family"], r["iid"]): r for r in universe()}
out = open(sys.argv[2], "a"); variants = sys.argv[3].split(",")
for spec_ in sys.argv[1].split(","):
    fam, iid = spec_.split(":"); r = U[(fam, int(iid))]
    mp = os.path.normpath(os.path.join(ROOT, fam, r["onnx"])); sp = os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"]))
    for v in variants:
        if v == "nocap":
            V9.POLISH_LAYER_MAX = 10 ** 9; V9.POLISH_TOTAL_MAX = 10 ** 9
        else:
            V9.POLISH_LAYER_MAX = 100; V9.POLISH_TOTAL_MAX = 600
        e = SoundEngineV17(mp, "cuda"); s = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
        spec = parse_vnnlib(sp, n_in, int(o0.c.numel()))
        t0 = time.time(); outcome = "CERT"; infos = []
        for lb, ub in spec.boxes:
            oc, info, wit = solve_box_v6(e, s, s.get_inputs()[0].name, spec, lb, ub, t0 + r["timeout"])
            infos.append({k: info[k] for k in ("propagate_s", "lp_worst", "lp_open") if k in info})
            if oc != "CERT":
                outcome = oc; break
        wall = time.time() - t0
        if outcome in ("CERT", "ADV") and wall > r["timeout"]:
            outcome = "TIMEOUT"
        rec = {"row": spec_, "baseline": r["baseline"], "variant": v, "outcome": outcome, "wall_s": wall, "polished": e.polished, "boxes": infos}
        out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
