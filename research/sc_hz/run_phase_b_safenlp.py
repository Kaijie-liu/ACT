"""Phase B: SC-HZ + ORT replay on ALL 1080 safenlp_2024 instances.

Per Phase A continuation memo §5 + §8. Parallel via multiprocessing.
Each worker processes a slice of iids; receipts written per-iid.

Phase B-1 output: A_CONFIRMED iid list for Phase B-2 production comparison.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def _process_one_iid(args: Tuple[int, int, int, str]) -> Dict[str, Any]:
    """Worker: run SC-HZ + ORT replay on one iid."""
    iid, K, wall_s, out_dir = args
    # Re-import inside worker to avoid forking issues
    from research.sc_hz.run_sentinels import run_iid
    from research.sc_hz.ort_replay import promote_iid

    t0 = time.perf_counter()
    rec = run_iid("safenlp_2024", iid, K=K, wall_s=wall_s)

    replay_result = None
    if rec.get("verdict") == "FAL_CANDIDATE":
        try:
            replay_result = promote_iid("safenlp_2024", iid, rec)
        except Exception as e:
            replay_result = {"replay_verdict": "REPLAY_ERROR",
                              "error": str(e)[:200]}

    wall = time.perf_counter() - t0

    combined = {
        "iid": iid,
        "sc_hz_verdict": rec.get("verdict"),
        "max_excess": rec.get("max_excess"),
        "wall_total_s": wall,
        "replay_verdict": (replay_result.get("replay_verdict")
                            if replay_result else None),
        "fail_closed_reason": rec.get("fail_closed_reason"),
    }
    # Save full receipt
    p = Path(out_dir) / "per_iid" / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    full_receipt = {**rec, "replay": replay_result, "wall_total_s": wall}
    with open(p, "w") as f:
        json.dump(full_receipt, f, indent=2, default=float)
    return combined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--wall-per-iid-s", type=int, default=30)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--max-iids", type=int, default=1080)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    iids = list(range(args.max_iids))
    work = [(iid, args.K, args.wall_per_iid_s, str(out_root)) for iid in iids]

    print(f"Phase B: SC-HZ + ORT on {len(work)} safenlp iids, "
          f"K={args.K}, wall={args.wall_per_iid_s}s, workers={args.n_workers}")
    t0 = time.perf_counter()

    results: List[Dict[str, Any]] = []
    a_iids: List[int] = []
    phantom_iids: List[int] = []
    cert_iids: List[int] = []

    # Use multiprocessing for parallelism
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_process_one_iid, work, chunksize=4)):
            results.append(rec)
            if rec.get("replay_verdict") == "A_CONFIRMED":
                a_iids.append(rec["iid"])
            elif rec.get("replay_verdict") == "PHANTOM_LP_SAT":
                phantom_iids.append(rec["iid"])
            elif rec.get("sc_hz_verdict") == "CERT":
                cert_iids.append(rec["iid"])
            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(work) - (i + 1)) / rate if rate > 0 else 0
                print(f"  progress {i+1}/{len(work)} "
                      f"({rate:.1f} iid/s, ETA {eta:.0f}s) — "
                      f"A_CONFIRMED so far: {len(a_iids)}",
                      flush=True)

    total_wall = time.perf_counter() - t0
    print(f"\nTotal wall: {total_wall:.1f}s")

    # Aggregate
    sc_hz_counts = Counter(r["sc_hz_verdict"] for r in results)
    replay_counts = Counter(r["replay_verdict"] for r in results
                              if r["replay_verdict"] is not None)
    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K": args.K,
        "wall_per_iid_s": args.wall_per_iid_s,
        "n_workers": args.n_workers,
        "n_total": len(results),
        "wall_seconds": total_wall,
        "sc_hz_counts": dict(sc_hz_counts),
        "replay_counts": dict(replay_counts),
        "n_a_confirmed": len(a_iids),
        "n_phantom_lp_sat": len(phantom_iids),
        "n_cert": len(cert_iids),
        "a_iids": sorted(a_iids),
        "cert_iids": sorted(cert_iids),
        "phantom_iids": sorted(phantom_iids)[:50],
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(out_root / "a_iids.txt", "w") as f:
        f.write(",".join(str(i) for i in sorted(a_iids)))
    print(f"\n=== Phase B-1 RESULT ===")
    print(f"n_total: {len(results)}")
    print(f"SC-HZ verdicts: {dict(sc_hz_counts)}")
    print(f"ORT replay verdicts: {dict(replay_counts)}")
    print(f"A_CONFIRMED: {len(a_iids)} ({100*len(a_iids)/len(results):.1f}%)")
    print(f"PHANTOM_LP_SAT: {len(phantom_iids)}")
    print(f"SC-HZ CERT: {len(cert_iids)}")
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
