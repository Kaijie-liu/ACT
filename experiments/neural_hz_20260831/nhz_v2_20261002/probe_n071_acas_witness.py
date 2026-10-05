"""N071: why the exact all-layers MILP finds no incumbent on ACAS Xu 60 (baseline ADV in 2.6 s).
Variants (single seed, 60 s): epigraph objective as in path v6 (with target), the same with
HiGHS heuristic options, and a pure feasibility version (atom rows as constraints, zero
objective).  Reports time to first feasible point and whether ORT confirms it."""
import sys, os, json, time, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v17 import SoundEngineV17
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v8 import build_plan_v8
from run_n039_full_replay import universe, ROOT
import highspy, onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.05)
U = {(r["family"], r["iid"]): r for r in universe()}
fam, iid = sys.argv[1], int(sys.argv[2]); r = U[(fam, iid)]
mp = os.path.normpath(os.path.join(ROOT, fam, r["onnx"])); spp = os.path.normpath(os.path.join(ROOT, fam, r["vnnlib"]))
e = SoundEngineV17(mp, "cuda"); s = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
spec = parse_vnnlib(spp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
nl = sum(1 for p in ph if int(p["idx"].numel()))
print("disjuncts", len(spec.disjuncts), "atoms", [len(d) for d in spec.disjuncts], "unstable", [int(p["idx"].numel()) for p in ph if int(p["idx"].numel())])
nzi = e.input_factor_index.cpu().numpy(); center = (lb + ub) / 2; rad = (ub - lb) / 2
def ort_ok(w):
    xi = np.zeros(n_in); xi[nzi] = np.asarray(w)[:nzi.size]; x = np.clip(center + rad * xi, lb, ub).astype(np.float32)
    y = s.run(None, {s.get_inputs()[0].name: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
    return any(all(float(a @ y) <= bb for a, bb in d) for d in spec.disjuncts)
for di, atoms in enumerate(spec.disjuncts):
    g, cc, pad, extra, T = epigraph_objective(o, K, atoms)
    g_full = np.zeros(K + 1); g_full[:len(g)] = g
    M, lo, hi, clo, chi, cost, integ, nb, m_tot, _ = build_plan_v8(A, b, ph, [], K, nl, 99, g_full, extra)
    print("disjunct", di, "T", T, "binaries", nb, "rows", M.shape)
    for name, opts, feas in (("epigraph", {}, False), ("epigraph+heur", {"mip_heuristic_effort": 1.0, "mip_heuristic_run_shifting": True, "mip_heuristic_run_zi_round": True}, False), ("feasibility", {}, True)):
        Mx, lox, hix, cst, clox, chix = M, lo, hi, cost, clo.copy(), chi.copy()
        if feas:
            cst = np.zeros_like(cost); clox[K] = 1e-6 / max(T, 1e-12) if T else 1e-9   # tau >= tiny > 0 : strictly inside every atom
        lp = highspy.HighsLp(); lp.num_col_ = Mx.shape[1]; lp.num_row_ = Mx.shape[0]
        lp.col_cost_ = cst; lp.col_lower_ = clox; lp.col_upper_ = chix
        lp.row_lower_ = np.where(np.isfinite(lox), lox, -highspy.kHighsInf); lp.row_upper_ = np.where(np.isfinite(hix), hix, highspy.kHighsInf)
        Mc = Mx.tocsc(); lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = Mc.indptr.astype(np.int32); lp.a_matrix_.index_ = Mc.indices.astype(np.int32); lp.a_matrix_.value_ = Mc.data
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.setOptionValue("time_limit", 60.0)
        for k_, v_ in opts.items(): h.setOptionValue(k_, v_)
        h.passModel(lp)
        if not feas:
            h.setOptionValue("objective_bound", float(cc + pad + 1e-4)); h.setOptionValue("objective_target", float(cc - 1e-6))
        t0 = time.time(); h.run(); st = h.modelStatusToString(h.getModelStatus()); dt = time.time() - t0
        sol = h.getSolution(); ok = None
        if len(sol.col_value) and np.isfinite(h.getInfo().objective_function_value):
            ok = ort_ok(np.array(sol.col_value[:K]))
        print(" ", name, st, round(dt, 1), "s", "ORT-valid", ok, flush=True)
