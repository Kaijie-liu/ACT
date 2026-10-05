"""N058: sign/curvature classification of the 784 sigmoid units of dist_shift (diagnostic).

For each sigmoid unit j with sound input bounds [l_j, u_j] (engine n011.2 propagation),
enclose d v / d y_j (v = violation objective in output space) through the classifier
784 -> 32 ReLU -> 32 ReLU -> 10 by interval reverse mode, slopes from interval bounds of the
classifier pre-activations computed from y in [sigma(l), sigma(u)].  Classes:
  free  : (dec and u <= 0) or (inc and l >= 0)   -> lambda tangents are exact for the query
  one   : dec or inc but the curvature region does not match -> one-sided segments needed
  both  : sign unknown -> two-sided
Reports counts and the relaxation area that matters (one-sided gap x |derivative|)."""
import sys, os, json, numpy as np, torch, onnx
from onnx import numpy_helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v11b import SoundEngineV11b
from nhz_engine import parse_vnnlib
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.05)
U = {(r["family"], r["iid"]): r for r in universe()}


class Rec(SoundEngineV11b):
    def _smooth(self, kind, x):
        out = super()._smooth(kind, x)
        K = x.G.shape[0]
        self.smooth_rec = getattr(self, "smooth_rec", [])
        return out

    def _bounds(self, fc, fG32, fe, idx, rows, K, iters, lr, rad_all=None):
        l, u = super()._bounds(fc, fG32, fe, idx, rows, K, iters, lr, rad_all)
        self.last_bounds = (l, u, int(idx.numel()))
        return l, u


def sig(x): return 1 / (1 + np.exp(-x))


m = onnx.load(os.path.join(ROOT, "dist_shift_2023", "onnx/mnist_concat.onnx"))
W = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in m.graph.initializer}
W1, b1 = W["cls/fc1.weight"], W["cls/fc1.bias"]; W2, b2 = W["cls/fc2.weight"], W["cls/fc2.bias"]; W3 = W["cls/fc3.weight"]
out = open(sys.argv[2], "a")
for iid in [int(x) for x in sys.argv[1].split(",")]:
    r = U[("dist_shift_2023", iid)]
    sp = os.path.normpath(os.path.join(ROOT, "dist_shift_2023", r["vnnlib"]))
    e = Rec(os.path.normpath(os.path.join(ROOT, "dist_shift_2023", r["onnx"])), "cuda")
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(sp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    # capture the sigmoid input bounds: wrap _smooth to grab l,u right after its _bounds calls
    cap = {}
    orig = Rec._smooth
    def _sm(self, kind, x, cap=cap):
        res = orig(self, kind, x); cap["lu"] = self.last_bounds; return res
    Rec._smooth = _sm
    o, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
    Rec._smooth = orig
    l, u, n = cap["lu"]; l = l.cpu().numpy(); u = u.cpu().numpy()
    ylo, yhi = sig(l), sig(u)
    # classifier interval forward
    def aff(Wm, b, lo, hi):
        Wp, Wn = np.maximum(Wm, 0), np.minimum(Wm, 0)
        return Wp @ lo + Wn @ hi + b, Wp @ hi + Wn @ lo + b
    z1l, z1h = aff(W1, b1, ylo, yhi); a1l, a1h = np.maximum(z1l, 0), np.maximum(z1h, 0)
    z2l, z2h = aff(W2, b2, a1l, a1h)
    s1 = (np.where(z1l >= 0, 1., 0.), np.where(z1h <= 0, 0., 1.)); s2 = (np.where(z2l >= 0, 1., 0.), np.where(z2h <= 0, 0., 1.))
    if os.environ.get("N058_ENGINE_SLOPES"):
        # diagnostic only: classifier slopes from the engine's LP-tightened phase sets (stable side by centre sign)
        nzp = [p for p in ph]
        cls = [p for p in ph if int(p["idx"].numel()) and p["layer"].startswith("cls")] if any(p["layer"].startswith("cls") for p in ph) else []
        def slopes_from(p, zl, zh):
            unst = np.zeros(zl.size, bool); unst[p["idx"].cpu().numpy()] = True
            ctr = (zl + zh) / 2
            lo_ = np.where(unst, 0., np.where(ctr >= 0, 1., 0.)); hi_ = np.where(unst, 1., np.where(ctr >= 0, 1., 0.))
            return lo_, hi_
        L = [p for p in ph if int(p["idx"].numel())]
        if len(L) >= 3:
            s1 = slopes_from(L[-2], z1l, z1h); s2 = slopes_from(L[-1], z2l, z2h)
    (a0, bb0), = spec.disjuncts[0] if len(spec.disjuncts[0]) == 1 else [spec.disjuncts[0][0]]
    c = -np.asarray(a0)                          # maximise b - a^T Y
    g = c @ W3                                  # (32,) exact
    def slope_mul(gl, gh, s):                   # [gl,gh] * [s_lo, s_hi], s in {0,1,[0,1]}
        cands = np.stack([gl * s[0], gl * s[1], gh * s[0], gh * s[1]])
        return cands.min(0), cands.max(0)
    def mat_mul(gl, gh, Wm):                    # row interval times matrix
        Wp, Wn = np.maximum(Wm, 0), np.minimum(Wm, 0)
        return gl @ Wp + gh @ Wn, gh @ Wp + gl @ Wn
    gl, gh = slope_mul(g, g, s2); gl, gh = mat_mul(gl, gh, W2); gl, gh = slope_mul(gl, gh, s1); gl, gh = mat_mul(gl, gh, W1)
    dec = gh <= 0; inc = gl >= 0; flat = (yhi - ylo) < 1e-6
    free = (dec & (u <= 0)) | (inc & (l >= 0)) | flat
    one = (dec | inc) & ~free
    both = ~(dec | inc) & ~flat
    # one-sided gap that matters: chord-vs-curve on the mismatched curvature part (crude: max over 64-point grid)
    gap = np.zeros_like(l)
    for j in np.flatnonzero(one | both):
        xs = np.linspace(l[j], u[j], 65); fx = sig(xs)
        ch = sig(l[j]) + (sig(u[j]) - sig(l[j])) * (xs - l[j]) / max(u[j] - l[j], 1e-12)
        gap[j] = np.abs(fx - ch).max()
    infl = gap * np.maximum(np.abs(gl), np.abs(gh))
    order = np.argsort(-infl)
    rec = {"iid": iid, "baseline": r["baseline"], "units": int(l.size), "free": int(free.sum()), "one_sided": int(one.sum()),
           "both": int(both.sum()), "flat": int(flat.sum()), "dec": int(dec.sum()), "inc": int(inc.sum()),
           "unstable_cls_relu": int(((z1l < 0) & (z1h > 0)).sum() + ((z2l < 0) & (z2h > 0)).sum()),
           "influence_top10": [round(float(infl[j]), 4) for j in order[:10]], "influence_sum": float(infl.sum()),
           "n_infl_gt_1e-2": int((infl > 1e-2).sum()), "n_infl_gt_1e-3": int((infl > 1e-3).sum())}
    out.write(json.dumps(rec) + "\n"); out.flush(); print(rec, flush=True)
