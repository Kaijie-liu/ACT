"""N077: composition of the softmax output radius on ViT rows (engine n017): linear part
sum|pG|, remainder rem, error ep, and the interval alternative, per softmax node (means)."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhz_sound_v14 as V14
from nhz_sound_v17 import SoundEngineV17
from nhz_engine import parse_vnnlib
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.1)
U = {(r["family"], r["iid"]): r for r in universe()}
stats = []
orig = torch.where
src = open(V14.__file__).read()
# instrument by wrapping the module-level softmax computation: re-exec op_Softmax with a hook
import types
code = src[src.index("    def op_Softmax(self, node, ins, a):"):]
code = code.replace("        use_int = (rem + ep) >= r_int      # n011.2: switch only where the remainder alone is uninformative",
 "        use_int = (rem + ep) >= r_int      # n011.2: switch only where the remainder alone is uninformative\n        stats.append({'lin': float(pG.abs().sum(0).mean()), 'rem': float(rem.mean()), 'ep': float(ep.mean()), 'int': float(r_int.mean()), 'frac_int': float(use_int.double().mean()), 'dr_mean': float(dr.mean())})")
ns = {}
exec("import torch\nfrom nhz_sound import SAZ, gamma\nfrom nhz_sound_v9 import F64, _up32\nclass _T:\n" + code, {**V14.__dict__, "stats": stats}, ns)
SoundEngineV17.op_Softmax = ns["_T"].op_Softmax
for iid in [int(x) for x in sys.argv[1].split(",")]:
    r = U[("vit_2023", iid)]
    e = SoundEngineV17(os.path.normpath(os.path.join(ROOT, "vit_2023", r["onnx"])), "cuda")
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(os.path.normpath(os.path.join(ROOT, "vit_2023", r["vnnlib"])), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    stats.clear(); o, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
    for i, st in enumerate(stats):
        print(iid, "softmax", i, {k: round(v, 5) for k, v in st.items()})
