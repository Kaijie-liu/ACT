"""N002 probe runner: aligned HZ + GPU LP tightening on E0 rows (diagnostic)."""

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
from gpu_shadow import Graph, parse_vnnlib_box_top1  # noqa: E402
from gpu_aligned_lp import margin_lower_bounds, run  # noqa: E402
from run_n001_shadow_census import EVID, ROOT, sha256  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--rows", default="unknown")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--final-iters", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ledger = json.load(open(f"{EVID}/{args.family}_evidence_baseline_v2.json"))
    inst = list(csv.reader(open(f"{ROOT}/{args.family}/instances.csv")))
    rows = ledger["rows"]
    if args.rows == "unknown":
        sel = [r for r in rows if r["baseline_verdict"] == "UNKNOWN"]
    elif args.rows == "adv":
        sel = [r for r in rows if r["baseline_verdict"] != "UNKNOWN"]
    else:
        want = {int(x) for x in args.rows.split(",")}
        sel = [r for r in rows if r["row_index"] in want]
    if args.limit:
        sel = sel[: args.limit]
    graphs = {}
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fout:
        for r in sel:
            onnx_rel, spec_rel, _ = inst[r["row_index"]]
            onnx_p = f"{ROOT}/{args.family}/{onnx_rel}"; spec_p = f"{ROOT}/{args.family}/{spec_rel}"
            if onnx_p not in graphs:
                assert sha256(onnx_p) == r["model_sha256"]
                graphs[onnx_p] = Graph(onnx_p, "cuda", torch.float32)
            assert sha256(spec_p) == r["spec_sha256"]
            in_shape = (3, 32, 32) if "CIFAR" in onnx_rel else (3, 56, 56)
            lb, ub, t, others = parse_vnnlib_box_top1(spec_p, int(np.prod(in_shape)))
            lbt = torch.as_tensor(lb.reshape(in_shape), device="cuda", dtype=torch.float32)
            ubt = torch.as_tensor(ub.reshape(in_shape), device="cuda", dtype=torch.float32)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
            log = []
            out, rs, K = run(graphs[onnx_p], lbt, ubt, iters=args.iters, lr=args.lr, log=log)
            m_sh = margin_lower_bounds(out, rs, K, t, others, 0, args.lr, lp=False)
            m_lp = margin_lower_bounds(out, rs, K, t, others, args.final_iters, args.lr, lp=True)
            torch.cuda.synchronize()
            rec = {"family": args.family, "row_index": r["row_index"], "baseline": r["baseline_verdict"],
                   "true": t, "K": K, "lp_rows": rs.n_rows,
                   "min_margin_shadow": float(m_sh.min()), "min_margin_lp": float(m_lp.min()),
                   "n_pos_lp": int((m_lp > 0).sum()), "n_margins": int(m_lp.numel()),
                   "lp_cert_probe": bool((m_lp > 0).all()),
                   "layers": [dict(name=l.name, n=l.n, unstable_shadow=l.unstable_shadow, unstable_lp=l.unstable_lp,
                                   wall_s=round(l.wall_s, 3)) for l in log],
                   "wall_s": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                   "iters": args.iters, "final_iters": args.final_iters, "lr": args.lr}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(r["row_index"], r["baseline_verdict"], "K", K, "shadow", round(rec["min_margin_shadow"], 3),
                  "lp", round(rec["min_margin_lp"], 4), "unst", [l.unstable_lp for l in log],
                  "wall", round(rec["wall_s"], 1), "mem", round(rec["peak_mem_gb"], 1), flush=True)
            del out, rs
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
