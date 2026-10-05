"""N046: time-to-exclusion of the sign-aware plan (terminal v5) on the SafeNLP CERT rows still
open after N044, for single HiGHS configurations run side by side (no early cancel):
seed 0, seed 2, mip_pscost_minreliable=1, heuristic options (the option values the frozen
baseline portfolio used).  Diagnostic only."""
import sys, os, csv, json, time, threading, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v5 import build_plan_v5
import highspy
torch.cuda.set_per_process_memory_fraction(0.05)
csv.field_size_limit(10**9)
ov = {(r['benchmark'], r['iid']): r for r in csv.DictReader(open('/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv'))}
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
CONF = {"seed0": {"random_seed": 0}, "seed2": {"random_seed": 2}, "pscost1": {"mip_pscost_minreliable": 1},
        "heur": {"mip_heuristic_effort": 1.0, "mip_heuristic_run_shifting": True, "mip_heuristic_run_zi_round": True}}
fam = sys.argv[1]; iids = sys.argv[2].split(","); outp = sys.argv[3]; tl = float(sys.argv[4])
out = open(outp, "a"); eng = {}
for iid in iids:
    r = ov[(fam, iid)]
    mp = os.path.normpath(os.path.join(ROOT, fam, r['onnx'])); spp = os.path.normpath(os.path.join(ROOT, fam, r['vnnlib']))
    if mp not in eng:
        eng.clear(); eng[mp] = SoundEngineV9(mp, 'cuda')
    e = eng[mp]
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(spp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M, rhs, nb, _ = build_plan_v5(A, b, ph, K, 1, 99, g_full, extra, True)
    n = K + 1 + nb
    res = {}
    def run(name, opts):
        lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = M.shape[0]
        lp.col_cost_ = np.concatenate([-g_full, np.zeros(nb)]); lp.col_lower_ = np.concatenate([-np.ones(K + 1), np.zeros(nb)]); lp.col_upper_ = np.ones(n)
        lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data
        lp.integrality_ = [highspy.HighsVarType.kContinuous] * (K + 1) + [highspy.HighsVarType.kInteger] * nb
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.setOptionValue("time_limit", tl)
        for k, v in opts.items():
            h.setOptionValue(k, v)
        h.passModel(lp); h.setOptionValue("objective_bound", float(cc + pad + 1e-4)); h.setOptionValue("objective_target", float(cc - 1e-6))
        t0 = time.time(); h.run(); info = h.getInfo()
        res[name] = {"status": h.modelStatusToString(h.getModelStatus()), "wall_s": time.time() - t0, "nodes": int(info.mip_node_count),
                     "upper": (-info.mip_dual_bound + cc + pad) if np.isfinite(info.mip_dual_bound) else None}
    th = [threading.Thread(target=run, args=(k, v)) for k, v in CONF.items()]
    [x.start() for x in th]; [x.join() for x in th]
    rec = {"family": fam, "iid": int(iid), "baseline": r["raw_verdict"], "binaries": nb, "time_limit": tl, "configs": res}
    out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
