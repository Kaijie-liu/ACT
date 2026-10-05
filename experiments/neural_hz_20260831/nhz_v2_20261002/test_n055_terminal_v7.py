"""N055 test: terminal v7 (vectorised, disaggregated) vs terminal v5 (dense big-M) on the
same plan levels.  (1) LP relaxations (integrality dropped) must have equal optima up to the
row tolerances; (2) MILP exclusion verdicts and incumbent-free optima must agree (60 s)."""
import sys, os, json, time, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v11b import SoundEngineV11b
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v5 import build_plan_v5
from nhz_terminal_v7 import build_plan_v7
from run_n039_full_replay import universe, ROOT
import highspy
torch.cuda.set_per_process_memory_fraction(0.05)
U = {(r["family"], r["iid"]): r for r in universe()}


def solve(M, lo, hi, clo, chi, cost, integ, tl, cutoff=None):
    lp = highspy.HighsLp(); lp.num_col_ = M.shape[1]; lp.num_row_ = M.shape[0]; M = M.tocsc()
    lp.col_cost_ = cost; lp.col_lower_ = clo; lp.col_upper_ = chi
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -highspy.kHighsInf); lp.row_upper_ = np.where(np.isfinite(hi), hi, highspy.kHighsInf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data
    if integ is not None and integ.any():
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.setOptionValue("time_limit", tl)
    h.passModel(lp)
    if cutoff is not None:
        h.setOptionValue("objective_bound", cutoff)
    h.run(); info = h.getInfo()
    return h.modelStatusToString(h.getModelStatus()), info.objective_function_value, getattr(info, "mip_dual_bound", None)


for spec_ in sys.argv[1].split(","):
    fam, iid = spec_.split(":"); r = U[(fam, int(iid))]
    mp = os.path.normpath(os.path.join(ROOT, fam, r["onnx"])); sp_ = os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"]))
    e = SoundEngineV11b(mp, "cuda"); n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(sp_, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    nl = sum(1 for p in ph if int(p["idx"].numel()))
    g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    for gl in sorted(set([1, nl])):
        Md, rhs, nbd, _ = build_plan_v5(A, b, ph, K, gl, 99, g_full, extra, False)
        nd = K + 1 + nbd
        dense = (Md, np.full(Md.shape[0], -np.inf), rhs, np.concatenate([-np.ones(K + 1), np.zeros(nbd)]), np.ones(nd),
                 np.concatenate([-g_full, np.zeros(nbd)]))
        t0 = time.time(); V = build_plan_v7(A, b, ph, K, gl, 99, g_full, extra, False); tb = time.time() - t0
        st1, f1, _ = solve(*dense, None, 60); st2, f2, _ = solve(*V[:6], None, 60)
        Vs = build_plan_v7(A, b, ph, K, gl, 99, g_full, extra, True)
        st3, f3, d3 = solve(*Vs[:7], 60, float(cc + pad + 1e-4)); st4, f4, d4 = solve(*dense, np.concatenate([np.zeros(K + 1, bool), np.ones(nbd, bool)]), 60, float(cc + pad + 1e-4))
        rec = {"row": spec_, "gamma_layers": gl, "lp_dense": cc - f1 + pad, "lp_v7": cc - f2 + pad, "build_s": tb,
               "milp_v7_sign": [st3, cc - d3 + pad if d3 is not None and np.isfinite(d3) else None, int(Vs[7])],
               "milp_dense_full": [st4, cc - d4 + pad if d4 is not None and np.isfinite(d4) else None, int(nbd)]}
        print(json.dumps(rec, default=float), flush=True)
