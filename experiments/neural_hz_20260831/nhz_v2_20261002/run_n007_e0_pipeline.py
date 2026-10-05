"""N007: single-path GPU Neural-HZ diagnostic over the whole E0 universe (400 rows).

One fixed configuration for every row (no per-row menu):
  1. aligned HZ propagation with batched LP tightening (iters, lr fixed);
  2. batched terminal LP bound for every unsafe disjunct;
  3. if every disjunct bound < 0: outcome LP_CERT_PROBE (diagnostic; rigorous
     rounding re-evaluation pending, so NOT a certificate yet);
  4. otherwise decode the box maximiser of each still-open disjunct (largest
     bound first) into an input, validate with ONNX Runtime against the
     original VNNLIB property: outcome ADV_ORT if any candidate is a true
     counterexample, else OPEN.
Witness decoding is a terminal-query primal heuristic, reported separately and
never credited as a representation gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import ENGINE_VERSION, Engine, parse_vnnlib, terminal_query  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
EVID = "/data1/Kane/FSE/ACT/experiments/neural_hz_20260831/evidence"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--rows", default="all")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--final-iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mem-fraction", type=float, default=0.45)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    ledger = json.load(open(f"{EVID}/{args.family}_evidence_baseline_v2.json"))
    inst = list(csv.reader(open(f"{ROOT}/{args.family}/instances.csv")))
    rows = ledger["rows"]
    if args.rows == "unknown":
        rows = [r for r in rows if r["baseline_verdict"] == "UNKNOWN"]
    elif args.rows == "adv":
        rows = [r for r in rows if r["baseline_verdict"] != "UNKNOWN"]
    elif args.rows != "all":
        want = {int(x) for x in args.rows.split(",")}
        rows = [r for r in rows if r["row_index"] in want]
    engines, sessions = {}, {}
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for r in rows:
            onnx_rel, spec_rel, _ = inst[r["row_index"]]
            mp = f"{ROOT}/{args.family}/{onnx_rel}"; sp = f"{ROOT}/{args.family}/{spec_rel}"
            rec = {"engine": ENGINE_VERSION, "family": args.family, "row_index": r["row_index"],
                   "baseline": r["baseline_verdict"], "iters": args.iters, "final_iters": args.final_iters, "lr": args.lr}
            try:
                if mp not in engines:
                    if sha256(mp) != r["model_sha256"]:
                        raise RuntimeError("model hash mismatch")
                    engines[mp] = Engine(mp, "cuda", torch.float32)
                    sessions[mp] = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
                if sha256(sp) != r["spec_sha256"]:
                    raise RuntimeError("spec hash mismatch")
                e = engines[mp]; sess = sessions[mp]; iname = sess.get_inputs()[0].name
                n_in = int(np.prod(e.input_shape))
                z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
                n_out = int(o0.c.numel())
                spec = parse_vnnlib(sp, n_in, n_out)
                if len(spec.boxes) != 1:
                    raise RuntimeError("multi-box spec")
                lb, ub = spec.boxes[0]
                torch.cuda.synchronize(); t0 = time.time()
                out, rset, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), args.iters, args.lr)
                bounds, cands = terminal_query(out, rset, K, spec.disjuncts, args.final_iters, args.lr)
                torch.cuda.synchronize(); t_bound = time.time() - t0
                rec.update({"K": K, "unstable": sum(int(p.idx.numel()) for p in phases),
                            "worst_bound": max(bounds), "n_open": sum(b >= 0 for b in bounds),
                            "bound_wall_s": t_bound})
                if max(bounds) < 0:
                    rec["outcome"] = "LP_CERT_PROBE"
                else:
                    nzi = e.input_factor_index.cpu().numpy()
                    center = (lb + ub) / 2; rad = (ub - lb) / 2
                    order = sorted(range(len(bounds)), key=lambda i: -bounds[i])
                    rec["outcome"] = "OPEN"; tried = 0
                    for i in order:
                        if bounds[i] < 0:
                            break
                        w = cands[i].double().cpu().numpy()
                        xi = np.zeros(n_in); xi[nzi] = w[: nzi.size]
                        x = np.clip(center + rad * xi, lb, ub).astype(np.float32)
                        y = sess.run(None, {iname: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
                        tried += 1
                        hit = next((k for k, atoms in enumerate(spec.disjuncts)
                                    if all(float(a @ y) <= bb for a, bb in atoms)), None)
                        if hit is not None:
                            rec["outcome"] = "ADV_ORT"
                            rec["witness_disjunct"] = hit
                            rec["witness_x_sha256"] = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                            np.save(f"{os.path.splitext(args.out)[0]}_x_{r['row_index']}.npy", x)
                            break
                    rec["decode_tried"] = tried
                rec["wall_s"] = time.time() - t0
            except Exception as ex:
                rec["outcome"] = "ERROR"; rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(r["row_index"], r["baseline_verdict"], rec["outcome"], round(rec.get("worst_bound", float("nan")), 4),
                  rec.get("unstable"), round(rec.get("wall_s", 0), 1), rec.get("error", ""), flush=True)
            del e
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
