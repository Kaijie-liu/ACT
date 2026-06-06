"""Aggregate horizontal-extension SC-HZ results + production baseline.

Reads:
  - Horizontal sweep summary (5 benchmarks, 463 iids)
  - 4 parallel production baselines on the CERT/A candidates

Reports:
  - NEW V per benchmark (SC-HZ CERT where production != CERTIFIED)
  - NEW A per benchmark (SC-HZ A_CONFIRMED where production != FALSIFIED)
  - Updated combined headline including horizontal extension
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    p_h = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_horizontal_*/"))[-1]
    p_p = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_horiz_prod_baseline_*/"))[-1]
    print(f"horizontal sweep: {p_h}")
    print(f"prod baseline:    {p_p}")

    h = json.load(open(f"{p_h}/summary.json"))

    # Aggregate production verdicts per subdir
    subdir_to_iids = {
        "mal_cert": ("malbeware", h["per_benchmark"]["malbeware"]["cert_iids"]),
        "mal_a":    ("malbeware", h["per_benchmark"]["malbeware"]["a_iids"]),
        "relu_cert_0": ("relusplitter", None),
        "relu_cert_1": ("relusplitter", None),
    }

    prod_verdicts = {}  # (bench, iid) -> production verdict
    for sub in ["mal_cert", "mal_a", "relu_cert_0", "relu_cert_1"]:
        for f in glob.glob(f"{p_p}/{sub}/per_instance*.json"):
            d = json.load(open(f))
            bench_from_sub = subdir_to_iids.get(sub, ("?", []))[0]
            for r in d.get("per_instance", []):
                iid = int(r["official_instance_id"])
                v = r.get("reportable_status", "MISSING")
                prod_verdicts[(bench_from_sub, iid)] = v

    print(f"\ncollected {len(prod_verdicts)} production verdicts")

    # NEW V per bench
    new_v_per_bench: dict = {}
    matched_v_per_bench: dict = {}
    for bench in ["malbeware", "relusplitter"]:
        cert_iids = h["per_benchmark"][bench]["cert_iids"]
        new_v = [i for i in cert_iids
                  if prod_verdicts.get((bench, i), "MISSING") != "CERTIFIED"]
        matched_v = [i for i in cert_iids
                      if prod_verdicts.get((bench, i), "MISSING") == "CERTIFIED"]
        new_v_per_bench[bench] = new_v
        matched_v_per_bench[bench] = matched_v
        verdict_counter = Counter(prod_verdicts.get((bench, i), "MISSING")
                                   for i in cert_iids)
        print(f"\n{bench} CERT side: {len(cert_iids)} SC-HZ CERTs")
        print(f"  production verdicts: {dict(verdict_counter)}")
        print(f"  NEW V (prod != CERT): {len(new_v)}")
        print(f"  MATCHED V (prod == CERT): {len(matched_v)}")

    # NEW A: malbeware has 1
    mal_a_iids = h["per_benchmark"]["malbeware"]["a_iids"]
    new_a = []
    matched_a = []
    for iid in mal_a_iids:
        v = prod_verdicts.get(("malbeware", iid), "MISSING")
        if v == "FALSIFIED":
            matched_a.append(iid)
        else:
            new_a.append(iid)
    print(f"\nmalbeware A side: {len(mal_a_iids)} SC-HZ A_CONFIRMED")
    print(f"  NEW A: {len(new_a)}")
    print(f"  MATCHED A: {len(matched_a)}")

    # Combined
    total_new_v = sum(len(v) for v in new_v_per_bench.values())
    total_new_a = len(new_a)

    print(f"\n=== HORIZONTAL TOTAL ===")
    print(f"NEW V across malbeware+relusplitter: {total_new_v}")
    print(f"NEW A: {total_new_a}")
    print(f"")
    print(f"Phase A+B (safenlp): +358 NEW A → 1282 V/A")
    print(f"Phase C (horizontal): +{total_new_v + total_new_a} NEW (V+A)")
    print(f"Combined: 1282 + {total_new_v + total_new_a} = {1282 + total_new_v + total_new_a}")

    aggregate = {
        "horizontal_sweep_summary": p_h,
        "prod_baseline_dir": p_p,
        "n_prod_verdicts": len(prod_verdicts),
        "new_v_per_bench": {b: len(v) for b, v in new_v_per_bench.items()},
        "matched_v_per_bench": {b: len(v) for b, v in matched_v_per_bench.items()},
        "new_v_iids_per_bench": {b: v for b, v in new_v_per_bench.items()},
        "new_a_malbeware": new_a,
        "matched_a_malbeware": matched_a,
        "total_new_v": total_new_v,
        "total_new_a": total_new_a,
        "with_horizontal_combined": 1282 + total_new_v + total_new_a,
    }
    p_out = Path(p_h) / "horizontal_aggregate.json"
    with open(p_out, "w") as f:
        json.dump(aggregate, f, indent=2, default=float)
    print(f"\nwrote {p_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
