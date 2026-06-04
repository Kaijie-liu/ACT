"""VGG mini-atlas merge — combine the 2026-06-03 original 18-iid atlas
with the 2026-06-04 missing-iid reruns into a single evidence table.

Per advisor 2026-06-04: do NOT overwrite the original. Produce a
side-by-side merged JSON + CSV with origin labeling and compute the
final §6b gate verdict.

Usage:
    python research/vgg_mini_atlas_merge.py \
        --base    <original_vgg_mini_atlas_root> \
        --rerun   <missing_rerun_root>... \
        --out     <merged_root>

If --out omitted, the merged result is written under
`audit_results/vgg_mini_atlas_canonical_plus_missing_<STAMP>/`.

Both --base and --rerun roots are expected to have been pre-processed
by `vgg_mini_atlas_reparse.py` so each `per_iid/iid*.json` carries the
corrected `layer_trace`, `girard_fires`, `n_layers_traced`,
`n_girard_fires` fields.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ACT_ROOT = Path("/data1/Kane/ACT")


N_INPUT_VGG = 3 * 224 * 224  # 150528


def load_per_iid(root: Path) -> Dict[int, Dict[str, Any]]:
    per_iid: Dict[int, Dict[str, Any]] = {}
    pid_dir = root / "per_iid"
    if not pid_dir.exists():
        return per_iid
    for f in sorted(pid_dir.glob("iid*.json")):
        try:
            d = json.load(open(f))
            per_iid[int(d["iid"])] = d
        except Exception as e:
            print(f"[merge] skip {f}: {e}", file=sys.stderr)
    return per_iid


def gate_decision(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the §6b ImageHZ go/no-go gate from the merged entries.

    Per advisor 2026-06-04:
      - PROCEED iff (≥ 5% of iids show root_factor_preserved_ratio < 0.95
                     AND the loss is concentrated at MAXPOOL or wide ReLU)
      - The Girard cap layer evidence now comes from the corrected
        layer_trace / girard_fires fields (reparse 2026-06-04).
    """
    n_total = len(entries)
    with_snap = [e for e in entries if (e.get("root_ng_at_flatten") is not None
                                         or (e.get("snapshot") or {}).get("root_ng") is not None)]
    correlation_lost = []
    for e in with_snap:
        rng = (e.get("root_ng_at_flatten")
               or (e.get("snapshot") or {}).get("root_ng"))
        if rng is None:
            continue
        ratio = rng / N_INPUT_VGG
        if ratio < 0.95:
            correlation_lost.append(e)
    # Girard fire concentration: layers L11/L18/L25/L32 MAXPOOL + L17/L29/L35 RELU
    target_layers = {11, 18, 25, 32, 17, 29, 35}
    iids_with_targeted_girard = []
    for e in entries:
        fires = e.get("girard_fires") or []
        if any(int(f["layer_id"]) in target_layers for f in fires):
            iids_with_targeted_girard.append(int(e["iid"]))
    share_corr = (len(correlation_lost) / len(with_snap) * 100) if with_snap else 0.0
    share_girard = (len(iids_with_targeted_girard) / n_total * 100) if n_total else 0.0
    proceed = (
        share_corr >= 5.0
        and len(iids_with_targeted_girard) >= 1
    )
    return {
        "n_total_iids": n_total,
        "n_with_snapshot": len(with_snap),
        "n_correlation_lost": len(correlation_lost),
        "correlation_lost_share_pct": share_corr,
        "iids_with_targeted_girard_fire": sorted(iids_with_targeted_girard),
        "n_iids_with_targeted_girard": len(iids_with_targeted_girard),
        "targeted_girard_share_pct": share_girard,
        "target_layers_definition": "L11/L18/L25/L32 MAXPOOL + L17/L29/L35 RELU",
        "decision": "PROCEED" if proceed else "STOP",
        "decision_basis": (
            "advisor 2026-06-04 §6b gate: ≥5% correlation-loss + ≥1 iid "
            "with Girard fire at MAXPOOL or wide ReLU"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True,
                    help="original vgg_mini_atlas_canonical_* root (must contain per_iid/)")
    ap.add_argument("--rerun", type=str, action="append", default=[],
                    help="missing-iid rerun root; can be passed multiple times")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    base_root = Path(args.base)
    rerun_roots = [Path(r) for r in args.rerun]
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_root = (Path(args.out) if args.out else
                ACT_ROOT / "audit_results" / f"vgg_mini_atlas_canonical_plus_missing_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base   = {base_root}")
    for r in rerun_roots:
        print(f"[merge] rerun  = {r}")
    print(f"[merge] out    = {out_root}")

    base_entries = load_per_iid(base_root)
    print(f"[merge] base loaded: {len(base_entries)} iids")

    merged: Dict[int, Dict[str, Any]] = {}
    # Start from base (mark origin).
    for iid, e in base_entries.items():
        e2 = dict(e)
        e2["_merge_origin"] = "base_2026-06-03"
        merged[iid] = e2
    # Apply reruns in order, overwriting base if rerun has snapshot data
    # AND base did not.
    for r in rerun_roots:
        r_entries = load_per_iid(r)
        print(f"[merge] rerun {r.name} loaded: {len(r_entries)} iids")
        for iid, e in r_entries.items():
            base_has_snap = (
                iid in merged
                and merged[iid].get("root_ng_at_flatten") is not None
            )
            r_has_snap = (e.get("root_ng_at_flatten") is not None
                          or (e.get("snapshot") or {}).get("root_ng") is not None)
            if r_has_snap and not base_has_snap:
                e2 = dict(e)
                e2["_merge_origin"] = f"rerun_{r.name}"
                # Preserve the base diag's iid id semantics.
                merged[iid] = e2
                print(f"[merge]   iid {iid}: rerun replaces base (rerun has snapshot, base did not)")
            elif r_has_snap and base_has_snap:
                # Keep base; record alternate.
                merged[iid].setdefault("_alternate_rerun_iids", []).append({
                    "rerun_root": str(r),
                    "rerun_root_ng": e.get("root_ng_at_flatten")
                                      or (e.get("snapshot") or {}).get("root_ng"),
                })
                print(f"[merge]   iid {iid}: both base and rerun have snapshot; keep base, record alt")
            else:
                print(f"[merge]   iid {iid}: rerun also has no snapshot; skip")

    entries = [merged[iid] for iid in sorted(merged.keys())]
    gate = gate_decision(entries)
    print()
    print("[merge] === final §6b gate ===")
    for k, v in gate.items():
        print(f"  {k}: {v}")

    # Write outputs.
    out_root.joinpath("per_iid").mkdir(parents=True, exist_ok=True)
    for iid, e in merged.items():
        with open(out_root / "per_iid" / f"iid{iid:03d}.json", "w") as f:
            json.dump(e, f, indent=2, default=float)
    with open(out_root / "vgg_mini_atlas_merged.json", "w") as f:
        json.dump({
            "stamp": stamp,
            "base_root": str(base_root),
            "rerun_roots": [str(r) for r in rerun_roots],
            "n_iids": len(entries),
            "gate": gate,
            "entries": entries,
        }, f, indent=2, default=float)

    keys = [
        "iid", "_merge_origin", "production_verdict", "production_wall_s",
        "n_layers_traced", "n_girard_fires",
        "root_ng_at_flatten", "ng_at_flatten",
        "nc_at_flatten", "nb_at_flatten", "boxhz_collapse",
    ]
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for e in entries:
            w.writerow([e.get(k, "") for k in keys])

    print(f"[merge] wrote {out_root}/vgg_mini_atlas_merged.json")
    print(f"[merge] wrote {out_root}/summary.csv")
    print(f"[merge] wrote {out_root}/per_iid/iid*.json (count={len(merged)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
