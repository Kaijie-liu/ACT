"""N003: family probe for the GPU Neural-HZ engine (diagnostic, float, no rounding control).

For each instances.csv row: parse spec, propagate the aligned HZ with batched
LP tightening, bound every unsafe disjunct with the terminal LP, and record
whether the LP relaxation already excludes all disjuncts ("lp_cert_probe").
The engine's center propagation is checked against ONNX Runtime once per model
as an implementation self-test (never a verdict source).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import AZ, Engine, ReluLog, certify_disjuncts, parse_vnnlib  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"


def ort_check(engine: Engine, onnx_path: str) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).uniform(0, 1, size=engine.input_shape).astype(np.float32)
    ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
    t = torch.as_tensor(x, device=engine.device, dtype=engine.dtype)
    out, _, _, _ = engine.propagate(t, t, 0)
    return float(np.abs(out.c.reshape(-1).double().cpu().numpy() - ref).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--rows", default="all")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--final-iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    inst = list(csv.reader(open(f"{ROOT}/{args.family}/instances.csv")))
    sel = range(len(inst)) if args.rows == "all" else [int(x) for x in args.rows.split(",")]
    engines = {}
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for ri in sel:
            onnx_rel, spec_rel, tmo = inst[ri][:3]
            onnx_p = os.path.normpath(f"{ROOT}/{args.family}/{onnx_rel}")
            spec_p = os.path.normpath(f"{ROOT}/{args.family}/{spec_rel}")
            rec = {"family": args.family, "row": ri, "onnx": onnx_rel, "spec": spec_rel, "timeout": tmo}
            try:
                if onnx_p not in engines:
                    e = Engine(onnx_p, "cuda", dtype)
                    rec["ort_center_err"] = ort_check(e, onnx_p)
                    engines[onnx_p] = e
                e = engines[onnx_p]
                n_in = int(np.prod(e.input_shape))
                z = torch.zeros(e.input_shape, device="cuda", dtype=dtype)
                o0, _, _, _ = e.propagate(z, z, 0)
                n_out = int(o0.c.numel())
                spec = parse_vnnlib(spec_p, n_in, n_out)
                rec["n_boxes"] = len(spec.boxes); rec["n_disjuncts"] = len(spec.disjuncts)
                torch.cuda.synchronize(); t0 = time.time()
                worst = []
                logs_all = []
                for (lb, ub) in spec.boxes:
                    log: list = []
                    out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), args.iters, args.lr, log)
                    ubs = certify_disjuncts(out, rows, K, spec.disjuncts, args.final_iters, args.lr)
                    worst.append(max(ubs))
                    logs_all.append([[l.unstable_shadow, l.unstable_final] for l in log])
                torch.cuda.synchronize()
                rec.update({"K": K, "lp_rows": rows.n_rows, "worst_disjunct_ub": max(worst),
                            "lp_cert_probe": bool(max(worst) < 0), "unstable": logs_all,
                            "wall_s": time.time() - t0})
            except Exception as ex:  # diagnostic probe: record and continue
                rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(ri, rec.get("worst_disjunct_ub"), rec.get("lp_cert_probe"), rec.get("wall_s"), rec.get("error", ""), flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
