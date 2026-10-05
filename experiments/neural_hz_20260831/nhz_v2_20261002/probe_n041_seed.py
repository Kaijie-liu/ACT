import sys, os, csv, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal import _build_plan
import highspy, onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.05)
csv.field_size_limit(10**9)
ov = {(r['benchmark'], r['iid']): r for r in csv.DictReader(open('/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv'))}
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
for iid in sys.argv[1].split(","):
    r = ov[('safenlp_2024', iid)]
    mp = os.path.normpath(os.path.join(ROOT, 'safenlp_2024', r['onnx'])); sp = os.path.normpath(os.path.join(ROOT, 'safenlp_2024', r['vnnlib']))
    e = SoundEngineV9(mp, 'cuda'); s = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(sp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    out, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    g, cc, pad, extra, T = epigraph_objective(out, K, spec.disjuncts[0])
    g = g[:K]
    M, rhs, nb = _build_plan(A, b, ph, K, [], 99, 99)
    for seed in (0, 1, 2, 3):
        for cutoff in (True,):
            lp = highspy.HighsLp(); n = K + nb; lp.num_col_ = n; lp.num_row_ = M.shape[0]
            lp.col_cost_ = np.concatenate([-g, np.zeros(nb)]); lp.col_lower_ = np.concatenate([-np.ones(K), np.zeros(nb)]); lp.col_upper_ = np.ones(n)
            lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf); lp.row_upper_ = rhs
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; lp.a_matrix_.start_ = M.indptr.astype(np.int32); lp.a_matrix_.index_ = M.indices.astype(np.int32); lp.a_matrix_.value_ = M.data
            lp.integrality_ = [highspy.HighsVarType.kContinuous] * K + [highspy.HighsVarType.kInteger] * nb
            h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.setOptionValue("time_limit", 20.0)
            h.setOptionValue("random_seed", seed); h.passModel(lp)
            h.setOptionValue("objective_bound", float(cc + pad + 1e-4))
            if cutoff: h.setOptionValue("objective_target", float(cc - 1e-6))   # stop at the first solution with violation >= 1e-6
            t0 = time.time(); h.run(); st = h.modelStatusToString(h.getModelStatus()); info = h.getInfo()
            v = cc - info.objective_function_value if np.isfinite(info.objective_function_value) else None
            w = np.array(h.getSolution().col_value[:K]) if np.isfinite(info.objective_function_value) else None
            ok = None
            if w is not None:
                nzi = e.input_factor_index.cpu().numpy(); xi = np.zeros(n_in); xi[nzi] = w[:nzi.size]
                x = np.clip((lb + ub) / 2 + (ub - lb) / 2 * xi, lb, ub).astype(np.float32)
                y = s.run(None, {s.get_inputs()[0].name: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
                ok = any(all(float(a_ @ y) <= bb for a_, bb in d) for d in spec.disjuncts)
            print(iid, "seed", seed, "target-stop", cutoff, st, "ORT-valid", ok, "best violation", None if v is None else round(v, 4), round(time.time() - t0, 1), "s", flush=True)
