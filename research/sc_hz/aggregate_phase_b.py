"""Aggregate Phase B-1/B-2/B-3 results into a comprehensive NEW V/A report.

Reads:
  - Phase B-1 sweep: SC-HZ + ORT verdicts on all 1080 safenlp iids
  - Phase B-2 production baseline: production verdict on 368 A_CONFIRMED iids
  - Phase B-3 production baseline: production verdict on 153 SC-HZ CERT iids

Produces:
  - NEW V count (SC-HZ CERT iids where production != CERTIFIED in 60s)
  - NEW A count (SC-HZ A_CONFIRMED iids where production != FALSIFIED in 60s)
  - MATCHED V (SC-HZ CERT iids also production CERT)
  - MATCHED A (SC-HZ A iids also production FAL)
  - Overlap with canonical ACT result on safenlp (V=335, A=10)
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    # Find the Phase B-1 + B-2 + B-3 directories
    p_b1 = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_phase_b_safenlp_*/"))[-1]
    p_b2 = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_phase_b_prod_baseline_*/"))[-1]
    p_b3 = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_phase_b3_cert_baseline_*/"))[-1]
    print(f"B-1: {p_b1}")
    print(f"B-2: {p_b2}")
    print(f"B-3: {p_b3}")

    # Load Phase B-1 summary
    b1 = json.load(open(f"{p_b1}/summary.json"))
    a_iids = set(b1["a_iids"])
    cert_iids = set(b1["cert_iids"])
    print(f"\nPhase B-1: {b1['n_total']} iids, A={b1['n_a_confirmed']}, CERT={b1['n_cert']}")

    # Load Phase B-2 production verdicts on A iids
    b2_verdicts = {}
    for f in glob.glob(f"{p_b2}/b*/per_instance*.json"):
        d = json.load(open(f))
        for r in d.get("per_instance", []):
            iid = int(r["official_instance_id"])
            v = r.get("reportable_status", "MISSING")
            b2_verdicts[iid] = v
    print(f"Phase B-2: {len(b2_verdicts)} production verdicts on A iids")

    # Load Phase B-3 production verdicts on CERT iids
    b3_verdicts = {}
    for f in glob.glob(f"{p_b3}/b*/per_instance*.json"):
        d = json.load(open(f))
        for r in d.get("per_instance", []):
            iid = int(r["official_instance_id"])
            v = r.get("reportable_status", "MISSING")
            b3_verdicts[iid] = v
    print(f"Phase B-3: {len(b3_verdicts)} production verdicts on CERT iids")

    # NEW A: SC-HZ A_CONFIRMED iids where production != FALSIFIED
    new_a = []
    matched_a = []
    missing_a = []
    a_prod_counts = Counter()
    for iid in a_iids:
        v = b2_verdicts.get(iid, "MISSING")
        a_prod_counts[v] += 1
        if v == "FALSIFIED":
            matched_a.append(iid)
        elif v == "MISSING":
            missing_a.append(iid)
        else:
            new_a.append(iid)

    # NEW V: SC-HZ CERT iids where production != CERTIFIED
    new_v = []
    matched_v = []
    missing_v = []
    v_prod_counts = Counter()
    for iid in cert_iids:
        v = b3_verdicts.get(iid, "MISSING")
        v_prod_counts[v] += 1
        if v == "CERTIFIED":
            matched_v.append(iid)
        elif v == "MISSING":
            missing_v.append(iid)
        else:
            new_v.append(iid)

    print(f"\n=== A side ===")
    print(f"SC-HZ A_CONFIRMED: {len(a_iids)}")
    print(f"  Production verdicts: {dict(a_prod_counts)}")
    print(f"  NEW A (prod != FAL): {len(new_a)}")
    print(f"  MATCHED A (prod == FAL): {len(matched_a)}")
    print(f"  MISSING from prod baseline: {len(missing_a)}")

    print(f"\n=== V side ===")
    print(f"SC-HZ CERT: {len(cert_iids)}")
    print(f"  Production verdicts: {dict(v_prod_counts)}")
    print(f"  NEW V (prod != CERT): {len(new_v)}")
    print(f"  MATCHED V (prod == CERT): {len(matched_v)}")
    print(f"  MISSING from prod baseline: {len(missing_v)}")

    print(f"\n=== Combined ===")
    print(f"NEW V/A from SC-HZ Phase B: {len(new_v) + len(new_a)} ({len(new_v)} V + {len(new_a)} A)")
    print(f"924 V/A baseline + SC-HZ NEW: {924 + len(new_v) + len(new_a)}")

    # Save aggregate
    aggregate = {
        "phase_b_1_sc_hz_total": {
            "n_iids": 1080,
            "n_a_confirmed": len(a_iids),
            "n_cert": len(cert_iids),
        },
        "phase_b_2_prod_baseline_on_a_iids": {
            "n_processed": len(b2_verdicts),
            "verdict_counts": dict(a_prod_counts),
            "n_new_a": len(new_a),
            "n_matched_a": len(matched_a),
            "n_missing": len(missing_a),
        },
        "phase_b_3_prod_baseline_on_cert_iids": {
            "n_processed": len(b3_verdicts),
            "verdict_counts": dict(v_prod_counts),
            "n_new_v": len(new_v),
            "n_matched_v": len(matched_v),
            "n_missing": len(missing_v),
        },
        "combined_new_v_a_from_sc_hz_phase_b": len(new_v) + len(new_a),
        "canonical_baseline": 924,
        "with_sc_hz_phase_b_combined": 924 + len(new_v) + len(new_a),
        "brief_17_phase_b_target": "924 → 1100/1300",
        "phase_b_target_achieved": (924 + len(new_v) + len(new_a)) >= 1100,
    }
    out_path = Path(p_b1) / "phase_b_aggregate.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=float)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
