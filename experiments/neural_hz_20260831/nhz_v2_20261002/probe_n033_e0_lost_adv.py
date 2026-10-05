"""N033 diagnostic: path v2 (n007 engine) on the E0 CIFAR ADV rows the float probe lost."""
import sys, os, csv, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_mp2 import SoundEngineMP2
from nhz_engine import parse_vnnlib
from nhz_path_v2 import solve_box_v2
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024"
torch.cuda.set_per_process_memory_fraction(0.25)
import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 1
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
budget = float(sys.argv[2])
for ri in [int(x) for x in sys.argv[1].split(",")]:
    o, s = inst[ri][:2]
    e = SoundEngineMP2(f"{ROOT}/{o}", "cuda"); sess = ort.InferenceSession(f"{ROOT}/{o}", so, providers=["CPUExecutionProvider"])
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(f"{ROOT}/{s}", n_in, int(o0.c.numel())); lb, ub = spec.boxes[0]
    t0 = time.time()
    oc, info, wit = solve_box_v2(e, sess, sess.get_inputs()[0].name, spec, lb, ub, t0 + budget)
    print(ri, oc, round(time.time() - t0, 1), "lp_worst", round(info["lp_worst"], 4), "open", info["lp_open"],
          {k: (v["status"], v["binaries"], round(v["upper"], 4) if v.get("upper") is not None else None) for k, v in info.get("milp", {}).items()},
          wit[0] if wit else None, flush=True)
    del e
    torch.cuda.empty_cache()
