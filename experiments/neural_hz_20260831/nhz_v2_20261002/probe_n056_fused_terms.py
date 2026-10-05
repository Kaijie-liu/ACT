"""N056: which term of the fused attention-mix bound dominates (vs the n011.2 composite)?"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhz_sound_v12 as V12
from nhz_sound_v12 import SoundEngineV12
from nhz_sound_mp import radius64
from nhz_engine import parse_vnnlib
from run_n039_full_replay import universe, ROOT
torch.cuda.set_per_process_memory_fraction(0.12)
U = {(r["family"], r["iid"]): r for r in universe()}
stats = []


class Probe(SoundEngineV12):
    def _attention_mix(self, s, v):
        K0 = self.K
        out_f = super()._attention_mix(s, v)
        Kf = self.K; self.K = K0
        # composite for comparison (n011.2 path): need P state -> recompute via parent softmax on the fly is complex;
        # report the fused pieces instead
        rf = radius64(out_f.G.reshape(out_f.G.shape[0], -1)).reshape(out_f.c.shape)
        fresh = out_f.G[K0:].double().abs().sum(0)
        stats.append({"layer": len(stats), "fused_total_mean": float(rf.mean()), "fused_fresh_mean": float(fresh.mean()),
                      "fused_lin_mean": float((rf - fresh).mean()), **self.last_fused_terms})
        self.K = Kf
        return out_f


for iid in [int(x) for x in sys.argv[1].split(",")]:
    r = U[("vit_2023", iid)]
    mp = os.path.normpath(os.path.join(ROOT, "vit_2023", r["onnx"])); sp = os.path.normpath(os.path.join(ROOT, "vit_2023", r["vnnlib"]))
    # instrument RB / RC inside the fused op
    orig = V12.SoundEngineV12._attention_mix
    e = Probe(mp, "cuda"); n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(sp, n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    stats.clear(); o, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
    print(iid, stats)
