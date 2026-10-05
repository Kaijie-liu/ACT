import sys, os, csv, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_mp import SoundEngineMP
from nhz_sound import SoundEngine
from nhz_engine import parse_vnnlib
from nhz_terminal import sound_lp_batch
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam, rows_, which = sys.argv[1], sys.argv[2], sys.argv[3]
torch.cuda.set_per_process_memory_fraction(float(sys.argv[4]) if len(sys.argv) > 4 else 0.4)
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
for ri in [int(x) for x in rows_.split(",")]:
    o, s = inst[ri][:2]
    for kind in which.split(","):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        E = SoundEngineMP(f"{ROOT}/{fam}/{o}", "cuda") if kind == "mp" else SoundEngine(f"{ROOT}/{fam}/{o}", "cuda", torch.float64)
        n_in = int(np.prod(E.input_shape)); o0, *_ = E.propagate(np.zeros(n_in), np.zeros(n_in), 0)
        spec = parse_vnnlib(f"{ROOT}/{fam}/{s}", n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
        t0 = time.time()
        out, rws, K, ph = E.propagate(lb, ub, 300, 0.05)
        lp = sound_lp_batch(out, rws, K, spec.disjuncts, 1000, 0.05)
        torch.cuda.synchronize()
        print(fam, ri, kind, "worst rigorous LP", round(max(d["upper"] for d in lp), 5), "e_out_max", float(out.e.max()),
              "unstable", sum(int(p["idx"].numel()) for p in ph), "wall", round(time.time() - t0, 1), "peak GB", round(torch.cuda.max_memory_allocated() / 2**30, 1), flush=True)
        del E, out, rws, ph, lp
