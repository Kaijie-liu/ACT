"""Gate 2: 40-sentinel memory gate for streaming-prune Conv.

Per advisor 2026-06-05 Gate 2 spec:
  - Goal: confirm streaming-prune lets cifar/tiny iids run with peak RAM
    under 80 GB and zero OOM, on the EXACT 40-iid sentinel set used in
    the Day-of dense-conv pilot.
  - NO V/A scoring at this stage. Goal is purely memory correctness.
  - Sentinel iids per advisor:
      cifar100:    0,2,6,8,24,29,57,72,86,113,118,130,145,156,168,180,185,190,195,199
      tinyimagenet: 0,6,12,24,30,45,60,73,86,99,116,130,145,158,170,180,190,195,198,199

Per-iid output:
  - peak_rss_gb (measured via resource.getrusage)
  - wall_s
  - output_ng (post-streaming-prune)
  - output_tail_norm (size of folded-tail contribution)
  - n_processed, n_skipped (parser coverage)
  - max_excess (LP UB - threshold) for monitoring; NOT a score

Subprocess isolation per iid: parent driver spawns single-iid worker
processes so OOM-kill on one iid doesn't take down the parent.

G10 pre-flight + RLIMIT_AS enforced.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ACT_ROOT = Path(__file__).resolve().parents[2]

CIFAR_IIDS = [0,2,6,8,24,29,57,72,86,113,118,130,145,156,168,180,185,190,195,199]
TINY_IIDS  = [0,6,12,24,30,45,60,73,86,99,116,130,145,158,170,180,190,195,198,199]

SINGLE_RUNNER = '''
import sys, json, time, resource
sys.path.insert(0, "/data1/Kane/ACT")
resource.setrlimit(resource.RLIMIT_AS, (80 * 1024**3, resource.RLIM_INFINITY))
import numpy as np
from research.canonical_provenance import load_instance, build_provenance
from research.sc_hz.onnx_walker_resnet import forward_resnet
from research.sc_hz.vnnlib_parse import parse_vnnlib
from research.sc_hz.ops import lp_ub_rival_margin

bench, iid, K_target = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out_path = sys.argv[4]
rec = {"bench": bench, "iid": iid, "K_target": K_target}
t0 = time.perf_counter()
try:
    prov = build_provenance(bench, iid)
    rec.update({
        "canonical_root": str(prov.canonical_root),
        "onnx_sha256": prov.onnx_sha256,
        "vnnlib_sha256": prov.vnnlib_sha256,
    })
    onnx_p, vnn_p = load_instance(bench, iid)
    if bench.startswith("cifar100"):
        n_in, n_classes = 3072, 100
    else:
        n_in, n_classes = 3*56*56, 200
    lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_p), n_in, n_classes)
    result = forward_resnet(str(onnx_p), lb_x, ub_x, K_per_layer=100000,
                              streaming_K_target=K_target,
                              streaming_chunk_size=256)
    wall = time.perf_counter() - t0
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
    rec.update({
        "wall_s": wall,
        "peak_rss_gb": rss_gb,
        "n_processed": result.n_nodes_processed,
        "n_skipped": len(result.nodes_skipped),
        "output_ng": int(result.output_state.G_kept.shape[1]),
        "output_tail_norm": float(result.output_state.tail_radius.sum()
                                     if result.output_state.tail_radius is not None
                                     else 0.0),
        "n_unsafe_conditions": len(unsafe),
    })
    max_excess = -float("inf")
    for d_out, t_thr, lbl in unsafe:
        ub = lp_ub_rival_margin(result.output_state, d_out)
        excess = float(ub) - float(t_thr)
        if excess > max_excess: max_excess = excess
    rec["max_excess"] = float(max_excess)
    rec["status"] = "OK"
except MemoryError:
    rec["status"] = "OOM"
    rec["wall_s"] = time.perf_counter() - t0
    rec["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
except Exception as e:
    rec["status"] = "ERROR"
    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["wall_s"] = time.perf_counter() - t0

with open(out_path, "w") as f:
    json.dump(rec, f, indent=2, default=float)
print(f"DONE iid={iid} status={rec['status']} rss={rec.get('peak_rss_gb',0):.1f}GB "
      f"wall={rec.get('wall_s',0):.0f}s ng={rec.get('output_ng', 0)}")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K-target", type=int, default=5000)
    ap.add_argument("--subprocess-timeout-s", type=int, default=900)
    args = ap.parse_args()

    # G10 pre-flight
    try:
        free_gb_out = subprocess.check_output(["free", "-g"], text=True).strip().split("\n")[1].split()
        available = int(free_gb_out[6])
        if available < 90:
            print(f"REFUSE: G10 violation — only {available} GB available")
            return 1
        print(f"G10 pre-flight OK: {available} GB available")
    except Exception as e:
        print(f"WARNING G10 check: {e}")

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    runner = out_root / "_runner.py"
    runner.write_text(SINGLE_RUNNER)

    work = [("cifar100_2024", i) for i in CIFAR_IIDS] + \
            [("tinyimagenet_2024", i) for i in TINY_IIDS]
    print(f"Gate 2 memory pilot: {len(work)} iids, K_target={args.K_target}, "
          f"subprocess_timeout={args.subprocess_timeout_s}s, n_workers=1 sequential",
          flush=True)

    t0 = time.perf_counter()
    results = []
    for i, (bench, iid) in enumerate(work):
        receipt = out_root / bench / f"iid{iid:04d}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i+1}/{len(work)}] {bench} iid {iid} (elapsed={time.perf_counter()-t0:.0f}s)...",
              flush=True)
        cmd = [
            "/data1/Kane/miniconda3/envs/act-py312/bin/python",
            str(runner), bench, str(iid), str(args.K_target), str(receipt),
        ]
        try:
            proc = subprocess.run(cmd, timeout=args.subprocess_timeout_s,
                                    capture_output=True, text=True)
            tail = (proc.stdout or "").strip().split("\n")[-3:]
            for line in tail: print(f"  {line}", flush=True)
            if proc.returncode != 0 and not receipt.exists():
                # write OOM/RC-killed receipt
                rec = {"bench": bench, "iid": iid, "K_target": args.K_target,
                        "status": "RC_KILL", "returncode": proc.returncode,
                        "stderr_tail": (proc.stderr or "")[-200:]}
                receipt.write_text(json.dumps(rec, indent=2))
                print(f"  RC_KILL — wrote receipt as RC_KILL", flush=True)
            if receipt.exists():
                results.append(json.load(open(receipt)))
        except subprocess.TimeoutExpired:
            rec = {"bench": bench, "iid": iid, "K_target": args.K_target,
                    "status": "TIMEOUT"}
            receipt.write_text(json.dumps(rec, indent=2))
            results.append(rec)
            print(f"  TIMEOUT after {args.subprocess_timeout_s}s", flush=True)
        # Intermediate summary
        cc = Counter(r.get("status") for r in results)
        n_ok = sum(1 for r in results if r.get("status") == "OK")
        max_rss = max((r.get("peak_rss_gb", 0) for r in results if r.get("status") == "OK"), default=0)
        with open(out_root / "intermediate.json", "w") as f:
            json.dump({"i_done": i+1, "counts": dict(cc),
                         "n_ok": n_ok, "max_rss_gb_so_far": max_rss}, f, indent=2)

    # Final summary
    wall = time.perf_counter() - t0
    by_bench = {}
    for b in ["cifar100_2024", "tinyimagenet_2024"]:
        sub = [r for r in results if r["bench"] == b]
        ok = [r for r in sub if r.get("status") == "OK"]
        by_bench[b] = {
            "n": len(sub),
            "n_OK": len(ok),
            "n_OOM": sum(1 for r in sub if r.get("status") == "OOM"),
            "n_TIMEOUT": sum(1 for r in sub if r.get("status") == "TIMEOUT"),
            "n_RC_KILL": sum(1 for r in sub if r.get("status") == "RC_KILL"),
            "n_ERROR": sum(1 for r in sub if r.get("status") == "ERROR"),
            "max_peak_rss_gb": max((r.get("peak_rss_gb", 0) for r in ok), default=0),
            "median_wall_s": sorted([r.get("wall_s", 0) for r in ok])[len(ok)//2] if ok else 0,
            "median_max_excess": sorted([r.get("max_excess", 0) for r in ok])[len(ok)//2] if ok else 0,
        }
        print(f"\n{b}: {by_bench[b]}", flush=True)
    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K_target": args.K_target,
        "wall_seconds": wall,
        "by_bench": by_bench,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== GATE 2 RESULT ===")
    print(f"cifar OK={by_bench['cifar100_2024']['n_OK']}/20, "
          f"OOM={by_bench['cifar100_2024']['n_OOM']}, "
          f"max_RSS={by_bench['cifar100_2024']['max_peak_rss_gb']:.1f} GB")
    print(f"tiny OK={by_bench['tinyimagenet_2024']['n_OK']}/20, "
          f"OOM={by_bench['tinyimagenet_2024']['n_OOM']}, "
          f"max_RSS={by_bench['tinyimagenet_2024']['max_peak_rss_gb']:.1f} GB")
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
