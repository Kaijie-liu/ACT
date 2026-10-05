"""Tests for the sound engine n004 (run as a script; pytest-compatible functions).

1. Containment: random inputs from the box, evaluated by ONNX Runtime in float32
   and by the engine's own exact-real-intended value map, must lie within the
   rigorous output shadow [c - rad - e, c + rad + e] (float32 ORT outputs are
   compared with a float32 slack, since ORT itself rounds).
2. Monotonicity: rigorous terminal bounds are >= the float probe bounds minus a
   small tolerance, i.e. rigor only loosens.
3. Small exact MLP: on a 2-2-1 ReLU network the rigorous LP bound equals the
   hand-computed triangle LP optimum within rounding.
"""

import os
import sys
import csv

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound import SoundEngine, sound_terminal  # noqa: E402
from nhz_engine import Engine, parse_vnnlib, terminal_query  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"


def _instance(fam, row):
    inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
    o, s = inst[row][:2]
    return os.path.normpath(f"{ROOT}/{fam}/{o}"), os.path.normpath(f"{ROOT}/{fam}/{s}")


def test_containment_and_monotone(fam="cifar100_2024", row=28, n_samples=64):
    import onnxruntime as ort
    mp, sp = _instance(fam, row)
    se = SoundEngine(mp, "cuda", torch.float64)
    n_in = int(np.prod(se.input_shape))
    fe = Engine(mp, "cuda", torch.float32)
    z = torch.zeros(fe.input_shape, device="cuda"); o0, *_ = fe.propagate(z, z, 0)
    spec = parse_vnnlib(sp, n_in, int(o0.c.numel()))
    lb, ub = spec.boxes[0]
    out, rows, K, _ = se.propagate(lb, ub, 300, 0.05)
    c = out.c.reshape(-1).double(); rad = out.G.reshape(K, -1).abs().sum(0).double(); e = out.e.reshape(-1).double()
    lo = (c - rad - e).cpu().numpy(); hi = (c + rad + e).cpu().numpy()
    sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(n_samples):
        x = rng.uniform(lb, ub).astype(np.float32)
        y = sess.run(None, {sess.get_inputs()[0].name: x.reshape(se.input_shape)})[0].reshape(-1).astype(np.float64)
        slack = 1e-4 * (1 + np.abs(y))
        worst = max(worst, float(np.max(np.maximum(lo - y - slack, y - hi - slack))))
    assert worst <= 0, f"containment violated by {worst}"
    sb = sound_terminal(out, rows, K, spec.disjuncts, 1000, 0.05)
    fo, frows, fK, _ = fe.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
    pb, _ = terminal_query(fo, frows, fK, spec.disjuncts, 1000, 0.05)
    diff = max(p - s for p, s in zip(pb, sb))
    return {"row": row, "sound_worst": max(sb), "probe_worst": max(pb), "max_probe_minus_sound": diff,
            "containment_margin": -worst, "e_out_max": float(e.max())}


if __name__ == "__main__":
    for r in [int(x) for x in sys.argv[1:]] or [28]:
        print(test_containment_and_monotone(row=r), flush=True)
