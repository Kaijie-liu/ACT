"""No-prune (K=∞) SC-HZ ablation on the iids that did NOT yield V/A at K=256.

Hypothesis: PRUNE thesis is falsified — LP UB strictly tightens as K grows.
With K=∞ (no pruning, all generators retained), the LP UB should be tighter
than at K=256. Effects we expect:
  - Some previously-UNK iids cross over → CERT (good)
  - Some previously-FAL_CANDIDATE iids drop below threshold → UNK
    (lose A but those A were phantom anyway; less of those)
  - Among remaining FAL_CANDIDATEs, A_CONFIRMED rate (via ORT) goes up

Target list:
  - relusplitter: 149 non-CERT iids (80 UNK + 69 PHANTOM)
  - safenlp:     559 PHANTOM iids (currently UNK)

Goal per advisor: +50 V/A combined, enough to push 1346 → 1396, beating
PyRAT[con_z] 1393.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
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
        rec = {"verdict": "UNK", "fail_closed_reason": f"{type(e).__name__}: {str(e)[:200]}"}

    replay_result = None
    if rec.get("verdict") == "FAL_CANDIDATE":
        try:
            replay_result = promote_iid(bench, iid, rec)
        except Exception as e:
            replay_result = {"replay_verdict": "REPLAY_ERROR", "error": str(e)[:200]}

    wall = time.perf_counter() - t0
    combined = {
        "bench": bench, "iid": iid, "K": K,
        "sc_hz_verdict": rec.get("verdict"),
        "max_excess": rec.get("max_excess"),
        "wall_s": wall,
        "replay_verdict": (replay_result.get("replay_verdict")
                            if replay_result else None),
    }
    p = Path(out_dir) / bench / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    full = {**rec, "replay": replay_result, "wall_s": wall, "K_used": K}
    with open(p, "w") as f:
        json.dump(full, f, indent=2, default=float)
    return combined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    # K_LARGE = 100000 acts as no-prune (will be capped to ng if smaller)
    ap.add_argument("--K", type=int, default=100000)
    ap.add_argument("--wall-per-iid-s", type=int, default=60)
    ap.add_argument("--n-workers", type=int, default=12)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build target lists
    p_b1 = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_phase_b_safenlp_*/"))[-1]
    p_h = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_horizontal_*/"))[-1]

    # safenlp: 559 PHANTOM iids
    safenlp_targets: List[int] = []
    for f in glob.glob(f"{p_b1}/per_iid/iid*.json"):
        d = json.load(open(f))
        if (d.get("replay", {}) or {}).get("replay_verdict") == "PHANTOM_LP_SAT":
            safenlp_targets.append(d["iid"])

    # relusplitter: 80 UNK + 69 PHANTOM_LP_SAT
    h_sum = json.load(open(f"{p_h}/summary.json"))
    relu_cert = set(h_sum["per_benchmark"]["relusplitter"]["cert_iids"])
    relu_a = set(h_sum["per_benchmark"]["relusplitter"]["a_iids"])
    relu_targets: List[int] = [i for i in range(220)
                                  if i not in relu_cert and i not in relu_a]

    print(f"safenlp PHANTOM targets: {len(safenlp_targets)}")
    print(f"relusplitter non-CERT/non-A targets: {len(relu_targets)}")
    print(f"K={args.K} (no-prune cap), wall={args.wall_per_iid_s}s, "
          f"workers={args.n_workers}")

    all_work = []
    for iid in safenlp_targets:
        all_work.append(("safenlp_2024", iid, args.K, args.wall_per_iid_s,
                          str(out_root)))
    for iid in relu_targets:
        all_work.append(("relusplitter", iid, args.K, args.wall_per_iid_s,
                          str(out_root)))
    print(f"Total work: {len(all_work)} iids")

    t0 = time.perf_counter()
    results = []
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_process_one, all_work,
                                                       chunksize=2)):
            results.append(rec)
            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                cert = sum(1 for r in results if r["sc_hz_verdict"] == "CERT")
                a = sum(1 for r in results
                          if r.get("replay_verdict") == "A_CONFIRMED")
                print(f"  {i+1}/{len(all_work)} ({rate:.1f} iid/s) — "
                      f"CERT={cert}, A={a}", flush=True)

    wall = time.perf_counter() - t0

    # Aggregate per-bench
    by_bench = {}
    for bench in ["safenlp_2024", "relusplitter"]:
        bench_recs = [r for r in results if r["bench"] == bench]
        sc_counts = Counter(r["sc_hz_verdict"] for r in bench_recs)
        replay_counts = Counter(r["replay_verdict"] for r in bench_recs
                                  if r["replay_verdict"] is not None)
        new_cert = sorted(r["iid"] for r in bench_recs
                            if r["sc_hz_verdict"] == "CERT")
        new_a = sorted(r["iid"] for r in bench_recs
                         if r.get("replay_verdict") == "A_CONFIRMED")
        by_bench[bench] = {
            "n_targets": len(bench_recs),
            "sc_hz_counts": dict(sc_counts),
            "replay_counts": dict(replay_counts),
            "new_cert_iids_under_no_prune": new_cert,
            "new_a_iids_under_no_prune": new_a,
            "n_new_cert": len(new_cert),
            "n_new_a": len(new_a),
        }
        print(f"\n{bench}: targets={len(bench_recs)}, "
              f"SC-HZ={dict(sc_counts)}, ORT={dict(replay_counts)}")
        print(f"  NEW CERT under K=∞: {len(new_cert)}")
        print(f"  NEW A_CONFIRMED under K=∞: {len(new_a)}")

    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K_used": args.K,
        "wall_per_iid_s": args.wall_per_iid_s,
        "n_workers": args.n_workers,
        "wall_seconds": wall,
        "n_total_targets": len(all_work),
        "by_bench": by_bench,
        "total_new_cert": sum(b["n_new_cert"] for b in by_bench.values()),
        "total_new_a": sum(b["n_new_a"] for b in by_bench.values()),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== NO-PRUNE ABLATION ===")
    print(f"Total new CERT: {summary['total_new_cert']}")
    print(f"Total new A_CONFIRMED: {summary['total_new_a']}")
    print(f"Total new V+A: {summary['total_new_cert'] + summary['total_new_a']}")
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
