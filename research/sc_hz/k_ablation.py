"""K ablation diagnostic: run SC-HZ on 40 working iids at K∈{64,128,256,512,1024}.

Hypothesis check: if d_L-relevance pruning works as the design lock §1.3 thesis
claims, then LP UB should MONOTONICALLY DECREASE as K decreases (we keep fewer
but more rival-relevant generators, with sound tail absorbing the rest).

If LP UB INCREASES as K decreases (i.e. pruning costs precision), the thesis
does NOT hold on this benchmark class — the LP UB is dominated by the
interval-tail magnitude, not by the pruned generators' precision.

This is the operationalization of the relevance-savings gap pinned in
test_relevance_score_ablations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.sc_hz.run_sentinels import run_iid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentinels", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K-list", type=str, default="64,128,256,512,1024")
    ap.add_argument("--wall-per-iid-s", type=int, default=120)
    args = ap.parse_args()

    K_list = [int(k) for k in args.K_list.split(",")]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    sentinels = json.load(open(args.sentinels))
    # Only the 2 working benchmarks
    work_benches = ["acasxu_2023", "safenlp_2024"]

    results: List[Dict[str, Any]] = []
    for bench in work_benches:
        iids = sentinels[bench]["iids"]
        for iid in iids:
            for K in K_list:
                t0 = time.perf_counter()
                rec = run_iid(bench, iid, K=K, wall_s=args.wall_per_iid_s)
                wall = time.perf_counter() - t0
                results.append({
                    "benchmark": bench, "iid": iid, "K": K,
                    "verdict": rec.get("verdict"),
                    "max_excess": rec.get("max_excess"),
                    "wall_s": wall,
                    "fail_closed_reason": rec.get("fail_closed_reason"),
                })
                print(f"  {bench} iid {iid:>3} K={K:>4} verdict={rec.get('verdict'):<14} "
                      f"max_excess={rec.get('max_excess')}", flush=True)

    # Aggregate per-iid: compute LP UB vs K curve
    from collections import defaultdict
    by_iid = defaultdict(dict)
    for r in results:
        by_iid[(r["benchmark"], r["iid"])][r["K"]] = r["max_excess"]

    # For each iid, check monotonicity
    monotone_decreasing = 0
    monotone_increasing = 0
    non_monotone = 0
    for key, k_to_ub in by_iid.items():
        ub_vals = [k_to_ub.get(K) for K in K_list]
        if any(v is None for v in ub_vals):
            non_monotone += 1
            continue
        diffs = [ub_vals[i+1] - ub_vals[i] for i in range(len(ub_vals)-1)]
        if all(d <= 1e-9 for d in diffs):
            monotone_decreasing += 1
        elif all(d >= -1e-9 for d in diffs):
            monotone_increasing += 1
        else:
            non_monotone += 1

    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K_list": K_list,
        "n_iids": len(by_iid),
        "n_runs": len(results),
        "interpretation": {
            "K_INCREASES_so_UB_INCREASES": "PRUNE COSTS precision (thesis FAILS — heuristic has no signal)",
            "K_INCREASES_so_UB_DECREASES": "PRUNE HELPS precision (thesis HOLDS — d_L relevance has signal)",
        },
        "per_iid_monotone_decreasing_with_K_decrease": monotone_decreasing,
        "per_iid_monotone_increasing_with_K_decrease": monotone_increasing,
        "per_iid_non_monotone": non_monotone,
        "by_iid_lp_ub": {f"{b}_iid{i}": dict(d) for (b, i), d in by_iid.items()},
    }
    with open(out_root / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(out_root / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {out_root}/ablation_summary.json")
    print(f"  per-iid monotone DECREASING (UB falls as K falls): {monotone_decreasing}")
    print(f"  per-iid monotone INCREASING (UB rises as K falls): {monotone_increasing}")
    print(f"  per-iid non-monotone: {non_monotone}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
