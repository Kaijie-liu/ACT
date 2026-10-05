"""Smoke/self-test of engine n009 on several families: ORT centre check, containment of random
ORT outputs in the rigorous output shadow, and the rigorous terminal LP bound."""
import sys, os, csv, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v10 import SoundEngineV10 as SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_terminal import sound_lp_batch
import onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.2)
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
for item in sys.argv[1].split(","):
    fam, ri = item.rsplit(":", 1); ri = int(ri)
    inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv"))); o, s = inst[ri][:2]
    mp = os.path.normpath(f"{ROOT}/{fam}/{o}"); sp = os.path.normpath(f"{ROOT}/{fam}/{s}")
    try:
        e = SoundEngineV9(mp, "cuda"); sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"]); nm = sess.get_inputs()[0].name
        n_in = int(np.prod(e.input_shape))
        x = np.random.default_rng(0).uniform(0, 1, size=n_in)
        oc, *_ = e.propagate(x, x, 0)
        ref = sess.run(None, {nm: x.astype(np.float32).reshape(e.input_shape)})[0].reshape(-1)
        cerr = float(np.abs(oc.c.reshape(-1).cpu().numpy() - ref).max())
        spec = parse_vnnlib(sp, n_in, int(oc.c.numel())); lb, ub = spec.boxes[0]
        t0 = time.time(); out, rows, K, ph = e.propagate(lb, ub, 300, 0.05)
        lp = sound_lp_batch(out, rows, K, spec.disjuncts, 1000, 0.05); t1 = time.time() - t0
        c = out.c.reshape(-1).double(); r = out.G.reshape(K, -1).double().abs().sum(0); ee = out.e.reshape(-1).double()
        lo = (c - r - ee).cpu().numpy(); hi = (c + r + ee).cpu().numpy(); worst = -np.inf
        for _ in range(64):
            xx = np.random.default_rng(_).uniform(lb, ub).astype(np.float32)
            yy = sess.run(None, {nm: xx.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
            sl = 1e-4 * (1 + np.abs(yy)); worst = max(worst, float(np.max(np.maximum(lo - yy - sl, yy - hi - sl))))
        print(item, "centre err", f"{cerr:.2e}", "containment", "OK" if worst <= 0 else f"VIOLATED {worst}", "rigorous LP worst", round(max(d['upper'] for d in lp), 4),
              "e_out", f"{float(ee.max()):.1e}", "K", K, round(t1, 1), "s", flush=True)
    except Exception as ex:
        import traceback; print(item, "ERROR", type(ex).__name__, str(ex)[:200]); traceback.print_exc(limit=2)
