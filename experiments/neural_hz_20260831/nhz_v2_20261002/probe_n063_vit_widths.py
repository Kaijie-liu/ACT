"""N063: per-node over-approximation on a ViT row: engine shadow width vs sampled width
(ORT on 3000 random box points), mean over coordinates.  Diagnostic only."""
import sys, os, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v14 import SoundEngineV14
from nhz_sound import SAZ
from nhz_sound_mp import radius64
from nhz_engine import parse_vnnlib
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.12)
U = {(r["family"], r["iid"]): r for r in universe()}
iid = int(sys.argv[1]); r = U[("vit_2023", iid)]
mp = os.path.normpath(os.path.join(ROOT, "vit_2023", r["onnx"]))
widths = {}


class Hook(SoundEngineV14):
    pass


orig_prop = SoundEngineV14.propagate
e = Hook(mp, "cuda"); n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, "vit_2023", r["vnnlib"])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
# wrap every op to record output widths
for name in [n for n in dir(e) if n.startswith("op_")]:
    f = getattr(e, name)
    def w(node, ins, a, f=f):
        out = f(node, ins, a)
        if isinstance(out, SAZ):
            G = out.G.reshape(out.G.shape[0], -1)
            rad = radius64(G).reshape(-1) + out.e.reshape(-1)
            widths[node.output[0]] = (node.op_type, (2 * rad).cpu().numpy())
        return out
    setattr(e, name, w)
o, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
m = onnx.load(mp)
outs = [x.name for x in m.graph.output]
for n in m.graph.node:
    for oo in n.output:
        if oo in widths and oo not in outs:
            m.graph.output.extend([onnx.helper.make_tensor_value_info(oo, onnx.TensorProto.FLOAT, None)])
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
names = [x.name for x in s.get_outputs()]
rng = np.random.default_rng(0); mins = {}; maxs = {}
for t in range(3000):
    x = (lb + (ub - lb) * rng.uniform(0, 1, lb.size)).astype(np.float32).reshape(e.input_shape)
    if t % 3 == 0:
        x = np.where(rng.uniform(0, 1, lb.size) < 0.5, lb, ub).astype(np.float32).reshape(e.input_shape)
    vals = s.run(None, {s.get_inputs()[0].name: x})
    for nm, v in zip(names, vals):
        v = v.reshape(-1).astype(np.float64)
        mins[nm] = np.minimum(mins.get(nm, v), v); maxs[nm] = np.maximum(maxs.get(nm, v), v)
prev = None
for n in m.graph.node:
    oo = n.output[0]
    if oo in widths and oo in mins:
        op, wd = widths[oo]; sw = maxs[oo] - mins[oo]
        if wd.size != sw.size:
            continue
        ok = sw > 1e-9
        ratio = float(np.mean(wd[ok]) / max(np.mean(sw[ok]), 1e-12)) if ok.any() else float('nan')
        print(f"{op:20s} {oo[-45:]:45s} ours {np.mean(wd):9.4f} sampled {np.mean(sw):9.4f} ratio {ratio:7.2f}")
