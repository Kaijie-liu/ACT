import sys, os, csv, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_mp2 import SoundEngineMP2
from nhz_sound_mp3 import SoundEngineMP3
from nhz_engine import parse_vnnlib
from nhz_terminal import sound_lp_batch
torch.cuda.set_per_process_memory_fraction(0.1)
fam = sys.argv[1]; ROOT = f"/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/{fam}"
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
for ri in [int(x) for x in sys.argv[2].split(",")]:
    o, s = inst[ri][:2]
    for Eng in (SoundEngineMP2, SoundEngineMP3):
        e = Eng(os.path.normpath(f"{ROOT}/{o}"), "cuda")
        n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
        spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{s}"), n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
        t0 = time.time(); out, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
        lp = sound_lp_batch(out, rows, K, spec.disjuncts, 1000, 0.05)
        print(fam, ri, Eng.__name__, "unstable", sum(int(p["idx"].numel()) for p in ph), "lp_worst", round(max(d["upper"] for d in lp), 4), round(time.time() - t0, 1), "s", flush=True)
