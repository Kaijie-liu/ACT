"""Horizontal extension: SC-HZ + ORT replay on 5 dense / small-dense benchmarks.

Per advisor 2026-06-04 review: extend the same SC-HZ directional-witness
sidecar mechanism that produced 358 NEW A on safenlp_2024 to:
  - malbeware       (150 inst, small-dense)
  - linearizenn_2024 (60 inst, small-dense)
  - cersyve         (12 inst, small-dense)
  - relusplitter    (220 inst, ReLU-split tests)
  - cgan_2023       (21 inst, generative — may fail-closed on conv)

Each iid: SC-HZ → if FAL_CANDIDATE → ORT replay → A_CONFIRMED or PHANTOM.
Failures (parser, shape, etc.) fail-closed to UNK. Multi-process parallel.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def _process_one(args: Tuple[str, int, int, int, str]) -> Dict[str, Any]:
    bench, iid, K, wall_s, out_dir = args
    from research.sc_hz.run_sentinels import run_iid
    from research.sc_hz.ort_replay import promote_iid

    t0 = time.perf_counter()
    try:
        rec = run_iid(bench, iid, K=K, wall_s=wall_s)
    except Exception as e:
        rec = {"verdict": "UNK", "fail_closed_reason": f"run_iid: {type(e).__name__}: {str(e)[:200]}"}

    replay_result = None
    if rec.get("verdict") == "FAL_CANDIDATE":
        try:
            replay_result = promote_iid(bench, iid, rec)
        except Exception as e:
            replay_result = {"replay_verdict": "REPLAY_ERROR",
                              "error": str(e)[:200]}

    wall = time.perf_counter() - t0
    combined = {
        "bench": bench, "iid": iid,
        "sc_hz_verdict": rec.get("verdict"),
        "max_excess": rec.get("max_excess"),
        "wall_total_s": wall,
        "replay_verdict": (replay_result.get("replay_verdict")
                            if replay_result else None),
        "fail_closed_reason": rec.get("fail_closed_reason"),
    }
    p = Path(out_dir) / bench / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    full = {**rec, "replay": replay_result, "wall_total_s": wall}
    with open(p, "w") as f:
        json.dump(full, f, indent=2, default=float)
    return combined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--wall-per-iid-s", type=int, default=30)
    ap.add_argument("--n-workers", type=int, default=12)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = [
        ("malbeware", 150),
        ("linearizenn_2024", 60),
        ("cersyve", 12),
        ("relusplitter", 220),
        ("cgan_2023", 21),
    ]

    all_work: List[Tuple[str, int, int, int, str]] = []
    for bench, n in targets:
        for iid in range(n):
            all_work.append((bench, iid, args.K, args.wall_per_iid_s, str(out_root)))

    print(f"Horizontal extension: SC-HZ + ORT on {len(all_work)} iids "
          f"across {len(targets)} benchmarks, K={args.K}, "
          f"wall={args.wall_per_iid_s}s, workers={args.n_workers}")
    t0 = time.perf_counter()

    results = []
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_process_one, all_work, chunksize=2)):
            results.append(rec)
            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                a_count = sum(1 for r in results if r.get("replay_verdict") == "A_CONFIRMED")
                cert_count = sum(1 for r in results if r.get("sc_hz_verdict") == "CERT")
                print(f"  {i+1}/{len(all_work)} ({rate:.1f} iid/s) — "
                      f"A so far: {a_count}, CERT: {cert_count}", flush=True)

    wall = time.perf_counter() - t0

    # Per-benchmark aggregate
    per_bench: Dict[str, Dict[str, Any]] = {}
    for bench, n in targets:
        bench_results = [r for r in results if r["bench"] == bench]
        sc_hz_counts = Counter(r["sc_hz_verdict"] for r in bench_results)
        replay_counts = Counter(r["replay_verdict"] for r in bench_results
                                if r["replay_verdict"] is not None)
        a_iids = sorted(r["iid"] for r in bench_results
                          if r["replay_verdict"] == "A_CONFIRMED")
        cert_iids = sorted(r["iid"] for r in bench_results
                            if r["sc_hz_verdict"] == "CERT")
        per_bench[bench] = {
            "n_total": len(bench_results),
            "sc_hz_counts": dict(sc_hz_counts),
            "replay_counts": dict(replay_counts),
            "a_iids": a_iids,
            "cert_iids": cert_iids,
            "n_a_confirmed": len(a_iids),
            "n_cert": len(cert_iids),
        }
        print(f"  {bench}: n={len(bench_results)}, "
              f"A_CONFIRMED={len(a_iids)}, CERT={len(cert_iids)}, "
              f"SC-HZ={dict(sc_hz_counts)}")

    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K": args.K,
        "wall_per_iid_s": args.wall_per_iid_s,
        "n_workers": args.n_workers,
        "n_total_iids": len(all_work),
        "wall_seconds": wall,
        "per_benchmark": per_bench,
        "total_a_confirmed": sum(b["n_a_confirmed"] for b in per_bench.values()),
        "total_cert": sum(b["n_cert"] for b in per_bench.values()),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== Horizontal extension RESULT ===")
    print(f"total A_CONFIRMED: {summary['total_a_confirmed']}")
    print(f"total CERT: {summary['total_cert']}")
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
