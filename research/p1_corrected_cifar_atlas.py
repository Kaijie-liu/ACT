"""P1 — corrected CIFAR atlas, per advisor 2026-06-03 productionization spec.

Replaces `atlas_v2_bucketed.json`. Fixes the methodological bugs:

1. `y_true` now parsed from VNNLIB first-line label
   ('; CIFAR100 property with label: N.'), NOT from dataset labels.
2. Records top-K rival LP upper bounds and per-rival ORT replay results
   (not just a single picked rival).
3. Center / random ORT outputs are DIAGNOSTIC ONLY — never used as FAL
   scoring.
4. Buckets are dropped; verdict is from the actual P0 dispatch run.

Output: `corrected_atlas_v1.json`. Source data: the per-iid summaries
under `PHASE2_P0_UNKNOWN185/`.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ACT_ROOT = Path(__file__).resolve().parent.parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


P0_DIR = Path(
    "/data1/Kane/ACT/audit_results/cifar_unknown_margin_atlas_20260603/"
    "PHASE2_P0_UNKNOWN185"
)
ATLAS_V2 = Path(
    "/data1/Kane/ACT/audit_results/cifar_unknown_margin_atlas_20260603/"
    "atlas_v2_bucketed.json"
)
OUT = Path(
    "/data1/Kane/ACT/audit_results/cifar_unknown_margin_atlas_20260603/"
    "corrected_atlas_v1.json"
)


def load_atlas_v2() -> Dict[int, Dict[str, Any]]:
    with open(ATLAS_V2) as f:
        a = json.load(f)
    return {e["iid"]: e for e in a}


def load_p0_summary(iid: int) -> Dict[str, Any] | None:
    # FAL receipts are at p0_iid<NNN>_FAL_rival<NNN>.json
    # UNK / ALL_SAFE summaries are at p0_iid<NNN>_summary.json
    fal_glob = list(P0_DIR.glob(f"p0_iid{iid:03d}_FAL_rival*.json"))
    if fal_glob:
        with open(fal_glob[0]) as f:
            return json.load(f)
    summary = P0_DIR / f"p0_iid{iid:03d}_summary.json"
    if summary.exists():
        with open(summary) as f:
            return json.load(f)
    return None


def classify_unk(top1_lp_ub: float, candidate_log: list) -> str:
    """Sub-classify UNK_NO_REPLAY into actionable buckets.

    - UNK_LP_TIGHT_NEAR_FAL : top-1 LP UB in (0, 2] and top-1 phantom_gap
                              in [0, 2] — the relaxation is tight but
                              the candidate just barely didn't replay.
                              Worth trying topK>5 or eq_lagr+LP tail.
    - UNK_PHANTOM_TOO_LOOSE : top-1 phantom_gap > 5 — the relaxation
                              is too loose to point at a real attack
                              from box-LP alone.
    - UNK_TAIL_RELAXATION   : everything else (moderate phantom).
    """
    if not candidate_log:
        return "UNK_NO_POSITIVE_RIVAL"
    top = candidate_log[0]
    phantom = top["lp_ub"] - top["ort_margin"]
    if top1_lp_ub <= 2.0 and 0 < phantom <= 2.0:
        return "UNK_LP_TIGHT_NEAR_FAL"
    if phantom > 5.0:
        return "UNK_PHANTOM_TOO_LOOSE"
    return "UNK_TAIL_RELAXATION"


def main() -> int:
    atlas_v2 = load_atlas_v2()
    print(f"atlas_v2 has {len(atlas_v2)} iids")

    rows = list(csv.DictReader(open(P0_DIR / "summary.csv")))
    print(f"P0 dispatch covered {len(rows)} iids")

    corrected: List[Dict[str, Any]] = []
    y_true_mismatch_count = 0
    for r in rows:
        iid = int(r["iid"])
        v2 = atlas_v2.get(iid, {})
        summary = load_p0_summary(iid)
        if summary is None:
            continue
        # vnnlib y_true (correct) is in the summary as y_true.
        y_true_vnnlib = int(summary.get("y_true", -1))
        atlas_v2_y_true = int(v2.get("y_true", -1)) if v2 else -1
        if atlas_v2_y_true != y_true_vnnlib:
            y_true_mismatch_count += 1

        verdict = r["verdict"]
        candidate_log = summary.get("candidate_log", []) if "candidate_log" in summary \
            else []
        top = summary.get("topK_rivals_lp_ub", [])

        entry: Dict[str, Any] = {
            "iid": iid,
            "onnx_path": summary.get("onnx_path", ""),
            "vnnlib_path": summary.get("vnnlib_path", ""),
            "y_true_vnnlib": y_true_vnnlib,
            "atlas_v2_y_true_INCORRECT": atlas_v2_y_true,
            "atlas_v2_y_true_BUG": atlas_v2_y_true != y_true_vnnlib,
            "verdict_strict": verdict,
            "topK_rivals_lp_ub": [
                {"rival": rv, "lp_upper_bound": ub} for rv, ub in top
            ],
        }
        if verdict == "FALSIFIED":
            entry["fal_rival"] = int(summary.get("target_rival", -1))
            entry["fal_lp_upper_bound"] = float(summary.get("lp_upper_bound_y_r_minus_y_t", float("nan")))
            entry["fal_ort_actual_margin"] = float(summary.get("ort_actual_margin", float("nan")))
            entry["fal_phantom_gap"] = entry["fal_lp_upper_bound"] - entry["fal_ort_actual_margin"]
        elif verdict == "ALL_RIVALS_LP_SAFE":
            entry["lp_safe_top1_rival"] = int(top[0][0]) if top else -1
            entry["lp_safe_top1_upper_bound"] = float(top[0][1]) if top else float("nan")
            entry["lp_safe_status"] = "WATCHLIST (closed-form box-LP CERT; not a baseline CERT)"
        else:
            sub = classify_unk(float(r["top1_lp_ub"]), candidate_log)
            entry["unk_subclass"] = sub
            entry["unk_topK_replay_attempts"] = candidate_log

        # Diagnostic-only fields (not part of FAL scoring).
        entry["diagnostic"] = {
            "v2_legacy_lp_unsafe": v2.get("lp_unsafe"),
            "v2_legacy_ort_unsafe": v2.get("ort_unsafe"),
            "v2_legacy_picked_worst_rival": v2.get("worst_rival"),
            "v2_legacy_bucket": v2.get("bucket"),
            "note": "v2 fields are LEGACY/POLLUTED by y_true bug; kept for traceability only",
        }
        corrected.append(entry)

    n_fal = sum(1 for e in corrected if e["verdict_strict"] == "FALSIFIED")
    n_lp_safe = sum(1 for e in corrected if e["verdict_strict"] == "ALL_RIVALS_LP_SAFE")
    n_unk = sum(1 for e in corrected if e["verdict_strict"].startswith("UNKNOWN"))
    print()
    print(f"Corrected atlas v1: {len(corrected)} iids")
    print(f"  y_true MISMATCHES v2 atlas: {y_true_mismatch_count}/{len(corrected)}")
    print(f"  FAL (strict): {n_fal}")
    print(f"  ALL_RIVALS_LP_SAFE (watchlist, not CERT): {n_lp_safe}")
    print(f"  UNK (subclassified): {n_unk}")
    sub_counts: Dict[str, int] = {}
    for e in corrected:
        if e["verdict_strict"].startswith("UNKNOWN"):
            sub_counts[e["unk_subclass"]] = sub_counts.get(e["unk_subclass"], 0) + 1
    for k, v in sorted(sub_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<30} : {v:>3}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "version": "corrected_v1",
            "date": "2026-06-03",
            "source": "PHASE2_P0_UNKNOWN185 dispatch + vnnlib first-line label",
            "methodology": {
                "y_true_source": "vnnlib first-line '; CIFAR100 property with label: N.'",
                "verdict_source": "P0 closed-form box-LP + strict ORT replay",
                "no_random_or_pgd": True,
                "all_rival_lp_safe_status": "watchlist; not a baseline CERT",
            },
            "stats": {
                "n_iids": len(corrected),
                "n_y_true_mismatch_with_v2_atlas": y_true_mismatch_count,
                "n_fal_strict": n_fal,
                "n_all_rivals_lp_safe_watchlist": n_lp_safe,
                "n_unk_subclassified": n_unk,
                "unk_subclass_counts": sub_counts,
            },
            "entries": corrected,
        }, f, indent=2)
    print()
    print(f"Corrected atlas v1: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
