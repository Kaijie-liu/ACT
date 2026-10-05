import sys, os, csv, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound import SoundEngine
from nhz_engine import parse_vnnlib
from nhz_terminal import sound_lp_batch, plan_milp_highs
from nhz_terminal_sign import sign_milp_highs
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
fam = sys.argv[1]
torch.cuda.set_per_process_memory_fraction(0.5)
inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
for ri in [int(x) for x in sys.argv[2].split(",")]:
    o, s = inst[ri][:2]
    e = SoundEngine(f"{ROOT}/{fam}/{o}", "cuda", torch.float64)
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(f"{ROOT}/{fam}/{s}", n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    out, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
    lp = sound_lp_batch(out, rows, K, spec.disjuncts, 1000, 0.05)
    i = int(np.argmax([d["upper"] for d in lp])); d = lp[i]
    A, b = rows.dense(K)
    full = plan_milp_highs(A, b, ph, K, d["g"], d["cc"], d["pad"], d["extra"], 1, 99, 200)
    sign = sign_milp_highs(A, b, ph, K, d["g"], d["cc"], d["pad"], d["extra"], 200)
    print(ri, "LP", round(d["upper"], 4), "| full-last-layer:", full["binaries"], "bin", full["status"], round(full["upper"] or 0, 4), round(full["wall_s"], 1), "s",
          "| sign-aware:", sign["binaries"], "/", sign["last_layer_units"], "bin", sign["status"], round(sign["upper"] or 0, 4), round(sign["wall_s"], 1), "s",
          "| peak GB", round(torch.cuda.max_memory_allocated() / 2**30, 1), flush=True)
    del out, rows, ph, lp, e, A, b
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
