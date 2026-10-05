"""N111c: (a) MILP without cutoff; (b) LP with d fixed to the witness phases, with cutoff;
(c) same with all finite row bounds relaxed by 1e-5; (d) types of rows violated at the exact point."""
import sys, os, csv, time, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v20 import SoundEngineV20
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v8 import build_plan_v8
from nhz_terminal_v7 import _need_masks
from nhz_monotone import _lam_mu
import highspy
torch.cuda.set_per_process_memory_fraction(0.3)
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"; fam = "cifar100_2024"; ri = 164
WIT = "/data1/Kane/HyZor/audit_results/resnet_recovery/cifar100_2024_probe_20260420_143209/sidecar_artifacts/largecls_sat_preflight_clip_cifar100_2024_164/cifar100_2024_164_large_cls_sat_preflight_0.x_star.npy"
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv"))); o, s, _ = inst[ri]; mp = f"{ROOT}/{fam}/{o}"
e = SoundEngineV20(mp, "cuda"); n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
spec = parse_vnnlib(f"{ROOT}/{fam}/{s}", n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
e.deadline = time.time() + 100; e.t_start = time.time()
out, rows, K, ph = e.propagate(lb, ub, 300, 0.05); nz = [p for p in ph if int(p["idx"].numel())]
A, b = rows.dense(K)
g, cc, pad, extra, T = epigraph_objective(out, K, spec.disjuncts[3]); g_full = np.zeros(K + 1); g_full[:len(g)] = g
M, lo, hi, clo, chi, cost, integ, nb, m_tot, _ = build_plan_v8(A, b, ph, [], K, len(nz), 99, g_full, extra, None, 2, sign_aware=True)
cutoff = cc + pad + 1e-4
# exact point from the witness (eta from TRUE pre-activations via ORT)
m = onnx.load(mp); relu_nodes = [n for n in m.graph.node if n.op_type == "Relu"]; outs = set(x_.name for x_ in m.graph.output)
for n in relu_nodes:
    if n.input[0] not in outs: m.graph.output.extend([onnx.helper.make_tensor_value_info(n.input[0], onnx.TensorProto.FLOAT, None)])
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]); iname = sess.get_inputs()[0].name
x = np.load(WIT).astype(np.float64).reshape(-1); center = (lb + ub) / 2; rad = (ub - lb) / 2; nzi = e.input_factor_index.cpu().numpy()
vals = dict(zip([x_.name for x_ in sess.get_outputs()], sess.run(None, {iname: x.astype(np.float32).reshape(e.input_shape)})))
w = np.zeros(K); w[:nzi.size] = (x - center)[nzi] / rad[nzi]
xtrue_all = []
for li, p in enumerate(nz):
    e0 = int(p["eta0"]); idx = p["idx"].cpu().numpy(); node = ([n for n in relu_nodes if n.name == p["layer"]] or [relu_nodes[li]])[0]
    xt = vals[node.input[0]].reshape(-1).astype(np.float64)[idx]; lam, mu = (v.cpu().numpy() for v in _lam_mu(p))
    w[e0:e0 + idx.size] = np.clip((np.maximum(xt, 0) - lam * xt) / mu - 1, -1, 1); xtrue_all.append(xt)
xtrue = np.concatenate(xtrue_all)
need = _need_masks(ph, nz, nz, K, g_full, extra, True)
sel = np.concatenate([np.flatnonzero(nd) + sum(int(q["idx"].numel()) for q in nz[:i]) for i, nd in enumerate(need)])
dpat = (xtrue[sel] >= 0).astype(float)
n7 = M.shape[1]; X0 = K + 1; P0 = X0 + m_tot; Q0 = P0 + nb; D0 = Q0 + nb
# row types
nrow_x = m_tot; print("rows:", M.shape[0], "x-eq rows", nrow_x, "binaries", nb, "non-binary units", m_tot - nb)


def solve(M, lo, hi, clo, chi, cost, integ, cutoff, use_int, tl=60):
    lp = highspy.HighsLp(); lp.num_col_ = M.shape[1]; lp.num_row_ = M.shape[0]; Mc = M.tocsc()
    lp.col_cost_ = cost; lp.col_lower_ = clo; lp.col_upper_ = chi
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -highspy.kHighsInf); lp.row_upper_ = np.where(np.isfinite(hi), hi, highspy.kHighsInf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Mc.indptr.astype(np.int32); lp.a_matrix_.index_ = Mc.indices.astype(np.int32); lp.a_matrix_.value_ = Mc.data.astype(np.float64)
    if use_int:
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1); h.setOptionValue("time_limit", tl)
    h.setOptionValue("presolve", "off"); h.passModel(lp)
    if cutoff is not None:
        h.setOptionValue("objective_bound", float(cutoff))
    t0 = time.time(); h.run(); info = h.getInfo()
    sol = np.array(h.getSolution().col_value) if len(h.getSolution().col_value) else None
    return h.modelStatusToString(h.getModelStatus()), round(time.time() - t0, 1), info.objective_function_value, info.mip_dual_bound if use_int else None, sol


st, dt, f, db, sol = solve(M, lo, hi, clo, chi, cost, integ, None, True, 60)
print("(a) MILP no cutoff:", st, dt, "obj", f, "dual bound", db, "-> violation cc-f", (cc - f) if np.isfinite(f) else None)
clo2, chi2 = clo.copy(), chi.copy(); clo2[D0:D0 + nb] = dpat; chi2[D0:D0 + nb] = dpat
st, dt, f, _, sol = solve(M, lo, hi, clo2, chi2, cost, integ, cutoff, False)
print("(b) LP, d fixed to witness phases, cutoff:", st, dt, "obj", f)
st, dt, f, _, sol = solve(M, lo, hi, clo2, chi2, cost, integ, None, False)
print("(b') LP, d fixed, no cutoff:", st, dt, "obj", f)
lo3 = np.where(np.isfinite(lo), lo - 1e-5, lo); hi3 = np.where(np.isfinite(hi), hi + 1e-5, hi)
st, dt, f, _, sol = solve(M, lo3, hi3, clo2, chi2, cost, integ, None, False)
print("(c) LP, d fixed, rows relaxed 1e-5:", st, dt, "obj", f)
st, dt, f, db, sol = solve(M, lo3, hi3, clo, chi, cost, integ, cutoff, True, 60)
print("(c') MILP, rows relaxed 1e-5, cutoff:", st, dt, "obj", f, "dual", db)
# (d) violated rows at the exact point, by type
v = np.zeros(M.shape[1]); v[:K] = w
xs = np.concatenate([p["cx"].double().cpu().numpy() + p["gx"].double().cpu().numpy() @ w[:p["gx"].shape[1]] for p in nz])
v[X0:X0 + m_tot] = xs; v[P0:P0 + nb] = np.minimum(xs[sel], 0); v[Q0:Q0 + nb] = np.maximum(xs[sel], 0); v[D0:D0 + nb] = dpat
Mv = M.tocsr() @ v; vio = np.maximum(lo - Mv, 0) + np.maximum(Mv - hi, 0); bad = np.flatnonzero(vio > 1e-7)
Mr = M.tocsr()
kinds = {}
for r_ in bad:
    cols = Mr.indices[Mr.indptr[r_]:Mr.indptr[r_ + 1]]
    k_ = "x-eq" if (cols.size > 3 and (X0 <= cols.max() < P0)) else ("piece-x" if P0 <= cols.max() < Q0 and cols.size == 3 else ("y-rel" if cols.size == 3 else ("d-row" if cols.max() >= D0 else "other")))
    kinds[k_] = kinds.get(k_, 0) + 1
print("(d) violated rows at exact point by type:", kinds, "max violation", float(vio.max()))
# how much does the exact point violate the sign-rule LP rows / tolerance: report ex, ey, delta scale
p = nz[-1]; print("last layer ex median", float(np.median(p["ex"].cpu().numpy())), "ey median", float(np.median(p["ey"].cpu().numpy())))
