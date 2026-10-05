"""N079: generic influence score s_j = |g[eta_j]| (objective coefficient on smooth unit j's own fresh
factor = linearised downstream sensitivity x shadow half-width); segment (2 per unit) the
non-saturated units in decreasing s_j until they cover 90 percent of sum s_j.  300 s per row."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v17 import SoundEngineV17
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v8 import plan_milp_v8
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.05)
U = {(r["family"], r["iid"]): r for r in universe()}
out = open(sys.argv[2], "a"); tl = float(sys.argv[3]); cover = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
KSEG = int(sys.argv[5]) if len(sys.argv) > 5 else 2
for iid in [int(x) for x in sys.argv[1].split(",")]:
    r = U[("dist_shift_2023", iid)]
    e = SoundEngineV17(os.path.normpath(os.path.join(ROOT, "dist_shift_2023", r["onnx"])), "cuda")
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, "dist_shift_2023", r["vnnlib"])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    nl = sum(1 for p in ph if int(p["idx"].numel())); sm = e.smooth_phases
    g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
    gf = np.asarray(g)
    scores = []; nonsat = []
    for s_ in sm:
        e0 = int(s_["eta0"]); n = int(s_["n"]); sc = np.abs(gf[e0:e0 + n])
        for ak, _ in extra:
            sc = np.maximum(sc, np.abs(np.asarray(ak)[e0:e0 + n]))
        scores.append(sc)
        f = torch.sigmoid if s_["kind"] == "Sigmoid" else torch.tanh
        nonsat.append(((f(s_["u"]) - f(s_["l"])) > 1e-4).cpu().numpy())
    sc = np.concatenate(scores); ns = np.concatenate(nonsat); sc = np.where(ns, sc, 0.0)
    order = np.argsort(-sc); cum = np.cumsum(sc[order]); k = int(np.searchsorted(cum, cover * cum[-1]) + 1)
    sel = np.zeros(sc.size, bool); sel[order[:k]] = True
    t0 = time.time()
    m = plan_milp_v8(A, b, ph, sm, K, g, cc, pad, extra, nl, 99, tl, seg_select=sel, K_seg=KSEG, target=None)
    rec = {"iid": iid, "baseline": r["baseline"], "cover": cover, "K_seg": KSEG, "N": int(sel.sum()), "nonsat": int(ns.sum()), "excluded": m["excluded"],
           "upper": m["upper"], "status": m["status"][:70], "binaries": m["binaries"], "wall_s": time.time() - t0}
    out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
