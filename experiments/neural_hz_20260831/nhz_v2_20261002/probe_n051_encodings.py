"""N051 (variant of N049): sparse big-M, Balas disaggregation, and sparse with HiGHS default threads; first written as N049: sparse equivalent encoding vs the dense v5 plan, same sign-aware binaries, single
HiGHS thread (seed 0), objective cutoff; SafeNLP CERT rows still open."""
import sys, os, csv, json, time, threading, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v5 import build_plan_v5
from nhz_terminal_sparse import build_sparse_single, build_balas_single
import highspy
torch.cuda.set_per_process_memory_fraction(0.05)
csv.field_size_limit(10**9)
ov = {(r['benchmark'], r['iid']): r for r in csv.DictReader(open('/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv'))}
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam = sys.argv[1]; iids = sys.argv[2].split(","); outp = sys.argv[3]; tl = float(sys.argv[4])
out = open(outp, "a"); eng = {}


def solve(M, lo, hi, clo, chi, cost, integ, cutoff, seed, threads=1):
    lp = highspy.HighsLp(); lp.num_col_ = M.shape[1]; lp.num_row_ = M.shape[0]; M = M.tocsc()
    lp.col_cost_ = cost; lp.col_lower_ = clo; lp.col_upper_ = chi
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -highspy.kHighsInf); lp.row_upper_ = hi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
    h = highspy.Highs(); h.setOptionValue("output_flag", False); (h.setOptionValue("threads", threads) if threads else None); h.setOptionValue("time_limit", tl)
    h.setOptionValue("random_seed", seed); h.passModel(lp); h.setOptionValue("objective_bound", cutoff)
    t0 = time.time(); h.run(); info = h.getInfo()
    return {"status": h.modelStatusToString(h.getModelStatus()), "wall_s": time.time() - t0, "nodes": int(info.mip_node_count),
            "dual": info.mip_dual_bound, "nnz": int(M.nnz), "rows": int(M.shape[0])}


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
    cutoff = float(cc + pad + 1e-4)
    Md, rhs, nb, _ = build_plan_v5(A, b, ph, K, 1, 99, g_full, extra, True)
    nd = K + 1 + nb
    dense = (Md, np.full(Md.shape[0], -np.inf), rhs, np.concatenate([-np.ones(K + 1), np.zeros(nb)]), np.ones(nd),
             np.concatenate([-g_full, np.zeros(nb)]), np.concatenate([np.zeros(K + 1), np.ones(nb)]))
    S = build_sparse_single(ph, K, g_full, extra, True)
    res = {}
    B = build_balas_single(ph, K, g_full, extra, True)
    def run(name, args, thr=1):
        res[name] = solve(*args, cutoff, 0, thr)
    th = [threading.Thread(target=run, args=("sparse", S[:7])), threading.Thread(target=run, args=("balas", B[:7])),
          threading.Thread(target=run, args=("sparse_thr_default", S[:7], 0))]
    [x.start() for x in th]; [x.join() for x in th]
    rec = {"family": fam, "iid": int(iid), "binaries": nb, "sparse_binaries": S[7], "res": res,
           "upper": {k: (-v["dual"] + cc + pad) if np.isfinite(v["dual"]) else None for k, v in res.items()}}
    out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
