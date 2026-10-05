import sys, os, time, csv
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound import SoundEngine, sound_terminal
from nhz_engine import parse_vnnlib
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam = sys.argv[1]
for row in [int(x) for x in sys.argv[2].split(",")]:
    inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv"))); o, s = inst[row][:2]
    se = SoundEngine(os.path.normpath(f"{ROOT}/{fam}/{o}"), "cuda", torch.float32)
    n_in = int(np.prod(se.input_shape))
    t0 = time.time()
    out0, *_ = se.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{fam}/{s}"), n_in, int(out0.c.numel()))
    lb, ub = spec.boxes[0]
    out, rows, K, ph = se.propagate(lb, ub, 300, 0.05)
    sb = sound_terminal(out, rows, K, spec.disjuncts, 1000, 0.05)
    torch.cuda.synchronize()
    print(row, "fp32-rigorous worst", round(max(sb), 5), "e_out_max", float(out.e.max()), "wall", round(time.time() - t0, 1), flush=True)
