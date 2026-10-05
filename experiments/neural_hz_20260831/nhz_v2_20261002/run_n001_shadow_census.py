"""N001: shadow-parametrisation census on the E0 CIFAR100/TinyImageNet universe.

Diagnostic only (no verdict authority, float32/float64 without directed
rounding).  For each requested E0 row it records, per ReLU layer, the
stable/unstable phase counts under the ``fresh`` (current ACT HZ fast-bound
shadow), ``aligned`` (N001 projection-aligned) and ``interval`` shadows, and
the minimum lower bound of the robustness margins.

Writes one JSON-lines record per instance with exclusive-create semantics.
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
from gpu_shadow import Graph, margins_lower, parse_vnnlib_box_top1, propagate  # noqa: E402

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
    ap.add_argument("--family", required=True, choices=["cifar100_2024", "tinyimagenet_2024"])
    ap.add_argument("--rows", default="unknown", help="'unknown', 'all', or comma list")
    ap.add_argument("--modes", default="fresh,aligned,interval")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    ledger = json.load(open(f"{EVID}/{args.family}_evidence_baseline_v2.json"))
    inst = list(csv.reader(open(f"{ROOT}/{args.family}/instances.csv")))
    rows = ledger["rows"]
    if args.rows == "unknown":
        sel = [r for r in rows if r["baseline_verdict"] == "UNKNOWN"]
    elif args.rows == "all":
        sel = rows
    else:
        want = {int(x) for x in args.rows.split(",")}
        sel = [r for r in rows if r["row_index"] in want]
    if args.limit:
        sel = sel[: args.limit]
    graphs = {}
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for r in sel:
            onnx_rel, spec_rel, _to = inst[r["row_index"]]
            onnx_p = f"{ROOT}/{args.family}/{onnx_rel}"
            spec_p = f"{ROOT}/{args.family}/{spec_rel}"
            if onnx_p not in graphs:
                if sha256(onnx_p) != r["model_sha256"]:
                    raise SystemExit("model hash mismatch")
                graphs[onnx_p] = Graph(onnx_p, "cuda", dtype)
            if sha256(spec_p) != r["spec_sha256"]:
                raise SystemExit(f"spec hash mismatch row {r['row_index']}")
            g = graphs[onnx_p]
            in_shape = (3, 32, 32) if "CIFAR" in onnx_rel else (3, 56, 56)
            lb, ub, t, others = parse_vnnlib_box_top1(spec_p, int(np.prod(in_shape)))
            lbt = torch.as_tensor(lb.reshape(in_shape), device="cuda", dtype=dtype)
            ubt = torch.as_tensor(ub.reshape(in_shape), device="cuda", dtype=dtype)
            rec = {"family": args.family, "row_index": r["row_index"], "baseline": r["baseline_verdict"],
                   "onnx": onnx_rel, "spec": spec_rel, "true": t, "dtype": args.dtype, "modes": {}}
            for mode in args.modes.split(","):
                torch.cuda.synchronize(); t0 = time.time()
                out, stats = propagate(g, lbt.unsqueeze(0) if mode == "interval" else lbt,
                                       ubt.unsqueeze(0) if mode == "interval" else ubt, mode)
                m = margins_lower(out, t, others)
                torch.cuda.synchronize()
                rec["modes"][mode] = {
                    "min_margin_lb": float(m.min()),
                    "n_margin_positive": int((m > 0).sum()),
                    "n_margins": int(m.numel()),
                    "shadow_cert": bool((m > 0).all()),
                    "unstable_per_relu": [s.unstable for s in stats],
                    "total_unstable": int(sum(s.unstable for s in stats)),
                    "wall_s": time.time() - t0,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
                torch.cuda.reset_peak_memory_stats()
                del out
                torch.cuda.empty_cache()
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(r["row_index"], {k: (v["min_margin_lb"], v["total_unstable"], round(v["wall_s"], 2)) for k, v in rec["modes"].items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
