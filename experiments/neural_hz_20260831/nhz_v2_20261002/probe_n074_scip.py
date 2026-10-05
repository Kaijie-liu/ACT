"""N074: SCIP vs the HiGHS portfolio on the same terminal plan (all ReLU layers gamma, v8)."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v17 import SoundEngineV17
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph
from nhz_terminal_v8 import build_plan_v8
from nhz_terminal_v7 import run_portfolio_v7
from nhz_scip_backend import run_scip
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.05)
U = {(r["family"], r["iid"]): r for r in universe()}
out = open(sys.argv[2], "a"); tl = float(sys.argv[3])
for spec_ in sys.argv[1].split(","):
    fam, iid = spec_.split(":"); r = U[(fam, int(iid))]
    e = SoundEngineV17(os.path.normpath(os.path.join(ROOT, fam, r["onnx"])), "cuda")
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    nl = sum(1 for p in ph if int(p["idx"].numel()))
    Ad, bd = A.to(o.c.device), b.to(o.c.device); ups = []
    for atoms in spec.disjuncts:
        g, cc, pad, extra, T = epigraph_objective(o, K, atoms); u_, _ = lp_bound_epigraph(Ad, bd, g, cc, extra, 1000, 0.05); ups.append(u_ + pad)
    i = int(np.argmax(ups)); g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[i])
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M, lo, hi, clo, chi, cost, integ, nb, m_tot, _ = build_plan_v8(A, b, ph, e.smooth_phases, K, nl, 99, g_full, extra)
    for name in ("highs4", "scip1"):
        if name == "highs4":
            m = run_portfolio_v7(M, lo, hi, clo, chi, cost, integ, K, cc, pad, tl, target=None)
        else:
            m = run_scip(M, lo, hi, clo, chi, cost, integ, K, cc, pad, tl)
        rec = {"row": spec_, "baseline": r["baseline"], "solver": name, "excluded": m["excluded"], "upper": m["upper"], "status": m["status"],
               "wall_s": m["wall_s"], "binaries": nb}
        out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
