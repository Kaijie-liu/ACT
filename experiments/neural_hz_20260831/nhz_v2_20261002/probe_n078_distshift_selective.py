"""N078 (diagnostic): dist_shift rows not retained after N065/N067: all-ReLU-gamma plan with smooth
rows (v8) + 2 segments on the top-N sigmoid units by influence (one-sided-free relaxation gap x
|d obj / d y| interval magnitude, as N058), N in {50, 100, 200, all non-saturated}; 300 s each."""
import sys, os, json, time, numpy as np, torch, onnx
from onnx import numpy_helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v17 import SoundEngineV17
from nhz_engine import parse_vnnlib
from nhz_terminal_v3 import epigraph_objective
from nhz_terminal_v8 import plan_milp_v8
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.05)
U = {(r["family"], r["iid"]): r for r in universe()}
m_ = onnx.load(os.path.join(ROOT, "dist_shift_2023", "onnx/mnist_concat.onnx"))
W = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in m_.graph.initializer}
W1, b1 = W["cls/fc1.weight"], W["cls/fc1.bias"]; W2, b2 = W["cls/fc2.weight"], W["cls/fc2.bias"]; W3 = W["cls/fc3.weight"]
sig = lambda x: 1 / (1 + np.exp(-x))
out = open(sys.argv[2], "a"); tl = float(sys.argv[3]); Ns = [int(x) for x in sys.argv[4].split(",")]
for iid in [int(x) for x in sys.argv[1].split(",")]:
    r = U[("dist_shift_2023", iid)]
    e = SoundEngineV17(os.path.normpath(os.path.join(ROOT, "dist_shift_2023", r["onnx"])), "cuda")
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, "dist_shift_2023", r["vnnlib"])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05); A, b = rows.dense(K)
    nl = sum(1 for p in ph if int(p["idx"].numel())); sm = e.smooth_phases
    g, cc, pad, extra, T = epigraph_objective(o, K, spec.disjuncts[0])
    l = sm[0]["l"].cpu().numpy(); u = sm[0]["u"].cpu().numpy(); ylo, yhi = sig(l), sig(u)
    def aff(Wm, bb, lo, hi):
        Wp, Wn = np.maximum(Wm, 0), np.minimum(Wm, 0); return Wp @ lo + Wn @ hi + bb, Wp @ hi + Wn @ lo + bb
    z1l, z1h = aff(W1, b1, ylo, yhi); z2l, z2h = aff(W2, b2, np.maximum(z1l, 0), np.maximum(z1h, 0))
    s1 = (np.where(z1l >= 0, 1., 0.), np.where(z1h <= 0, 0., 1.)); s2 = (np.where(z2l >= 0, 1., 0.), np.where(z2h <= 0, 0., 1.))
    a0, bb0 = spec.disjuncts[0][0]; gvec = -np.asarray(a0) @ W3
    def smul(gl, gh, s_):
        c_ = np.stack([gl * s_[0], gl * s_[1], gh * s_[0], gh * s_[1]]); return c_.min(0), c_.max(0)
    def mmul(gl, gh, Wm):
        Wp, Wn = np.maximum(Wm, 0), np.minimum(Wm, 0); return gl @ Wp + gh @ Wn, gh @ Wp + gl @ Wn
    gl, gh = smul(gvec, gvec, s2); gl, gh = mmul(gl, gh, W2); gl, gh = smul(gl, gh, s1); gl, gh = mmul(gl, gh, W1)
    gap = np.zeros_like(l)
    for j in range(l.size):
        xs = np.linspace(l[j], u[j], 65); fx = sig(xs); ch = sig(l[j]) + (sig(u[j]) - sig(l[j])) * (xs - l[j]) / max(u[j] - l[j], 1e-12)
        gap[j] = np.abs(fx - ch).max()
    infl = gap * np.maximum(np.abs(gl), np.abs(gh)); order = np.argsort(-infl)
    nonsat = (yhi - ylo) > 1e-4
    for N in Ns:
        sel = np.zeros(l.size, bool)
        if N <= 0:
            sel = nonsat.copy()
        else:
            sel[order[:N]] = True; sel &= nonsat
        t0 = time.time()
        m = plan_milp_v8(A, b, ph, sm, K, g, cc, pad, extra, nl, 99, tl, seg_select=sel, K_seg=2, target=None)
        rec = {"iid": iid, "N": int(sel.sum()), "excluded": m["excluded"], "upper": m["upper"], "status": m["status"][:60], "binaries": m["binaries"], "wall_s": time.time() - t0}
        out.write(json.dumps(rec, default=float) + "\n"); out.flush(); print(rec, flush=True)
