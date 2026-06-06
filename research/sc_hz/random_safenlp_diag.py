"""50 random safenlp iids (disjoint from sentinel set) — SC-HZ + ORT replay.

Generates a sample-bias check: if the 5/20=25% A_CONFIRMED rate from the
Phase A sentinels was due to favorable sample selection, a fresh random
sample should show a different (likely lower) rate.

Excludes the 20 iids already in the Phase A sentinel set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.sc_hz.run_sentinels import run_iid  # noqa: E402
from research.sc_hz.ort_replay import promote_iid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--wall-per-iid-s", type=int, default=60)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--exclude-sentinels", type=str,
                    default="audit_results/sc_hz_phase_a_sentinels_20260604.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    excluded = set()
    if Path(args.exclude_sentinels).exists():
        d = json.load(open(args.exclude_sentinels))
        excluded = set(d.get("safenlp_2024", {}).get("iids", []))

    # Pick N random iids from safenlp's pool of 1080
    pool = [i for i in range(1080) if i not in excluded]
    sample_iids = sorted(rng.sample(pool, min(args.n, len(pool))))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_iid").mkdir(exist_ok=True)

    results = []
    for iid in sample_iids:
        t0 = time.perf_counter()
        rec = run_iid("safenlp_2024", iid, K=args.K, wall_s=args.wall_per_iid_s)
        wall_sc = time.perf_counter() - t0

        replay_result = None
        if rec.get("verdict") == "FAL_CANDIDATE":
            try:
                replay_result = promote_iid("safenlp_2024", iid, rec)
            except Exception as e:
                replay_result = {"replay_verdict": "REPLAY_ERROR",
                                  "error": str(e)[:200]}

        combined = {
            "iid": iid,
            "sc_hz_verdict": rec.get("verdict"),
            "max_excess": rec.get("max_excess"),
            "wall_sc_hz_s": wall_sc,
            "replay_verdict": (replay_result.get("replay_verdict")
                                if replay_result else None),
        }
        results.append(combined)
        with open(out_root / "per_iid" / f"iid{iid:04d}.json", "w") as f:
            json.dump({**rec, "replay": replay_result}, f, indent=2, default=float)
        print(f"  iid {iid:>4} SC-HZ={rec.get('verdict'):<14} "
              f"ORT={(replay_result.get('replay_verdict') if replay_result else 'n/a'):<15}", flush=True)

    # Aggregate
    from collections import Counter
    sc_hz_counts = Counter(r["sc_hz_verdict"] for r in results)
    replay_counts = Counter(r["replay_verdict"] for r in results
                              if r["replay_verdict"] is not None)
    n_a = replay_counts.get("A_CONFIRMED", 0)
    n_phantom = replay_counts.get("PHANTOM_LP_SAT", 0)
    n_cert = sc_hz_counts.get("CERT", 0)

    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "seed": args.seed,
        "n_sampled": len(results),
        "sample_pool_size": len(pool),
        "excluded_count": len(excluded),
        "sc_hz_verdict_counts": dict(sc_hz_counts),
        "ort_replay_verdict_counts": dict(replay_counts),
        "n_a_confirmed": n_a,
        "n_phantom": n_phantom,
        "n_cert": n_cert,
        "a_rate": n_a / len(results) if results else 0.0,
        "cert_rate": n_cert / len(results) if results else 0.0,
        "compare_to_phase_a_sentinel_5_in_20_a_rate": 0.25,
        "results": results,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== RESULT ===")
    print(f"sample size: {len(results)}")
    print(f"new A_CONFIRMED: {n_a} ({100*n_a/len(results):.1f}%)")
    print(f"phantom_lp_sat: {n_phantom}")
    print(f"SC-HZ CERT: {n_cert} ({100*n_cert/len(results):.1f}%)")
    print(f"Phase A sentinel rate was 25% A; this sample is {100*n_a/len(results):.1f}%")
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
