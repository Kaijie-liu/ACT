"""N111: root-cause probe for the false CERT on E0 CIFAR row 164.
At the baseline witness x*: (1) list ONNX ops; (2) propagate engine n020.1 (same as N109) with the
row box; (3) read true intermediate values from ORT (graph outputs added); (4) for every ReLU phase
layer compare the true pre-activation with the engine's [l, u] and with its value map x_hat = cx +
gx w (eta filled exactly, as test N066); (5) evaluate every row of the stage-E plan at the exact
latent point and report violations."""
import sys, os, csv, json, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v20 import SoundEngineV20
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v8 import build_plan_v8
from nhz_monotone import _lam_mu
torch.cuda.set_per_process_memory_fraction(0.3)
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"; fam = "cifar100_2024"; ri = 164
WIT = "/data1/Kane/HyZor/audit_results/resnet_recovery/cifar100_2024_probe_20260420_143209/sidecar_artifacts/largecls_sat_preflight_clip_cifar100_2024_164/cifar100_2024_164_large_cls_sat_preflight_0.x_star.npy"
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv"))); o, s, _ = inst[ri]
mp, sp = f"{ROOT}/{fam}/{o}", f"{ROOT}/{fam}/{s}"
m = onnx.load(mp); import collections
print("ops:", dict(collections.Counter(n.op_type for n in m.graph.node)))
e = SoundEngineV20(mp, "cuda"); n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
spec = parse_vnnlib(sp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
import time; e.deadline = time.time() + 100; e.t_start = time.time()
out, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
nz = [p for p in ph if int(p["idx"].numel())]
print("K", K, "phase layers", [(p["layer"], int(p["idx"].numel())) for p in nz])
x = np.load(WIT).astype(np.float64).reshape(-1); center = (lb + ub) / 2; rad = (ub - lb) / 2
nzi = e.input_factor_index.cpu().numpy(); w_in = ((x - center)[nzi] / rad[nzi]); print("latent |w_in| max", float(np.abs(w_in).max()))
# true intermediates: add every Relu input as a graph output
relu_inputs = [n.input[0] for n in m.graph.node if n.op_type == "Relu"]
outs = set(x_.name for x_ in m.graph.output)
for nm in relu_inputs:
    if nm not in outs:
        m.graph.output.extend([onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None)])
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]); iname = sess.get_inputs()[0].name
names = [x_.name for x_ in sess.get_outputs()]; vals = dict(zip(names, sess.run(None, {iname: x.astype(np.float32).reshape(e.input_shape)})))
# engine value maps at the exact latent point
w = np.zeros(K); w[:len(w_in)] = w_in
layers = sorted(nz, key=lambda p: int(p["eta0"]))
relu_nodes = [n for n in m.graph.node if n.op_type == "Relu"]
for li, p in enumerate(layers):
    e0 = int(p["eta0"]); idx = p["idx"].cpu().numpy(); gx = p["gx"].double().cpu().numpy(); cx = p["cx"].double().cpu().numpy()
    ex = p["ex"].double().cpu().numpy(); l = p["l"].double().cpu().numpy(); u = p["u"].double().cpu().numpy()
    xhat = cx + gx @ w[:gx.shape[1]]
    # true pre-activation of the matching Relu node (order of Relu nodes = order of phase layers)
    node = [n for n in relu_nodes if n.name == p["layer"]]
    node = node[0] if node else relu_nodes[li]
    xt_full = vals[node.input[0]].reshape(-1).astype(np.float64); xt = xt_full[idx]
    lam, mu = (v.cpu().numpy() for v in _lam_mu(p)); y = np.maximum(xt, 0)
    w[e0:e0 + idx.size] = (y - lam * xt) / mu - 1
    below = l - xt; above = xt - u; gap = np.abs(xt - xhat) - ex
    print(f"layer {li} {p['layer']} n={idx.size}: true x outside [l,u]: below {int((below>1e-9).sum())} (max {below.max():.3e}), above {int((above>1e-9).sum())} (max {above.max():.3e}); |x_true-x_hat|-ex max {gap.max():.3e}; eta range [{w[e0:e0+idx.size].min():.3f},{w[e0:e0+idx.size].max():.3f}]")
    # also check stable units: any unit with l>=0 (declared active) whose true pre-activation is negative, or u<=0 with positive true value?
    fc_all = None
# final output check
yout = vals[m.graph.output[0].name].reshape(-1).astype(np.float64) if m.graph.output[0].name in vals else None
A, b = rows.dense(K)
g, cc, pad, extra, T = epigraph_objective(out, K, spec.disjuncts[3])
g_full = np.zeros(K + 1); g_full[:len(g)] = g
M, lo, hi, clo, chi, cost, integ, nb, m_tot, _ = build_plan_v8(A, b, ph, [], K, len(nz), 99, g_full, extra, None, 2, sign_aware=True)
# fill plan columns at the exact point: w | tau=0 | x_k (all gamma units) | x0 | x1 | d
v = np.zeros(M.shape[1]); v[:K] = w; X0 = K + 1; off = 0
xs_all = []
for li, p in enumerate(layers):
    idx = p["idx"].cpu().numpy(); gx = p["gx"].double().cpu().numpy(); cx = p["cx"].double().cpu().numpy()
    xs_all.append(cx + gx @ w[:gx.shape[1]])
xs = np.concatenate(xs_all); v[X0:X0 + xs.size] = xs
from nhz_terminal_v7 import _need_masks
need = _need_masks(ph, nz, nz, K, g_full, extra, True); sel = np.concatenate([np.flatnonzero(nd) + sum(int(q["idx"].numel()) for q in nz[:i]) for i, nd in enumerate(need)])
P0 = X0 + xs.size; Q0 = P0 + nb; D0 = Q0 + nb
v[P0:P0 + nb] = np.minimum(xs[sel], 0); v[Q0:Q0 + nb] = np.maximum(xs[sel], 0); v[D0:D0 + nb] = (xs[sel] >= 0)
Mv = M.tocsr() @ v
viol_lo = np.maximum(lo - Mv, 0); viol_hi = np.maximum(Mv - hi, 0)
print("plan rows", M.shape, "binaries", nb, "row violations >1e-7: lo", int((viol_lo > 1e-7).sum()), "hi", int((viol_hi > 1e-7).sum()), "max", float(max(viol_lo.max(), viol_hi.max())))
print("column bound violations:", float(np.max(np.maximum(clo - v, 0))), float(np.max(np.maximum(v - chi, 0))))
# objective value at the point vs cutoff
f = float(cost @ v); print("objective f =", f, "cutoff cc+pad+margin =", cc + pad + 1e-4, "violation cc - f =", cc - f, "(ORT slack -0.022 expected)")
# which rows violated: classify
bad = np.flatnonzero((viol_lo > 1e-7) | (viol_hi > 1e-7))
print("first violated rows:", bad[:10].tolist(), "of", M.shape[0]); nA = int(sum(1 for _ in []))
