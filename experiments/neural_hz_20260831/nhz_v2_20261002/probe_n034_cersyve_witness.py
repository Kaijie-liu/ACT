import sys, os, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_mp2 import SoundEngineMP2
from nhz_engine import parse_vnnlib
from nhz_terminal import sound_lp_batch, plan_milp_highs
import onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.1)
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cersyve"
inst = list(csv.reader(open(f"{ROOT}/instances.csv")))
for ri in [int(x) for x in sys.argv[1].split(",")]:
    o, s = inst[ri][:2]
    e = SoundEngineMP2(f"{ROOT}/{o}", "cuda"); sess = ort.InferenceSession(f"{ROOT}/{o}", providers=["CPUExecutionProvider"])
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(f"{ROOT}/{s}", n_in, int(o0.c.numel()))
    print(ri, o, "boxes", len(spec.boxes), "disjuncts", len(spec.disjuncts), "atoms per disjunct", [len(d) for d in spec.disjuncts][:5], "n_in", n_in, "n_out", int(o0.c.numel()))
    lb, ub = spec.boxes[0]
    out, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
    lp = sound_lp_batch(out, rows, K, spec.disjuncts, 1000, 0.05)
    i = int(np.argmax([d["upper"] for d in lp])); d = lp[i]
    A, b = rows.dense(K)
    m = plan_milp_highs(A, b, ph, K, d["g"], d["cc"], d["pad"], d["extra"], 99, 99, 60)
    w = m["incumbent_w"]; nzi = e.input_factor_index.cpu().numpy()
    xi = np.zeros(n_in); xi[nzi] = w[: nzi.size]; x = np.clip((lb + ub) / 2 + (ub - lb) / 2 * xi, lb, ub).astype(np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
    c = out.c.reshape(-1).double(); G = out.G.reshape(K, -1).double()
    yhat = (c + torch.as_tensor(w, device="cuda", dtype=torch.float64) @ G).cpu().numpy()
    print("  MILP", m["status"], "upper", round(m["upper"], 5), "| atoms at ORT y:", [round(float(a @ y - bb), 6) for a, bb in spec.disjuncts[i]],
          "| atoms at HZ yhat:", [round(float(a @ yhat - bb), 6) for a, bb in spec.disjuncts[i]], "| max|y-yhat|", float(np.abs(y - yhat).max()))
