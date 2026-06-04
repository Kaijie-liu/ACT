"""VGG mini-atlas re-parse: rebuild summary.csv from the EXISTING watchdog
logs without rerunning any iid.

Per advisor 2026-06-04: the original driver read `stdout_tail` from
per_instance.json, which `watchdog_runner.py` truncates to the last
2000 bytes. With 38 VGG layers × 2 lines per layer that truncation
loses every layer trace except the last 5-10. The driver therefore
reported `n_layers_traced = 1` and `n_girard_fires = 0` on every iid,
which made the §6b text gate look like STOP even though the snapshot
root_ng signal already implied PROCEED.

`watchdog_runner.py` persists the FULL subprocess stdout to
`out_dir / watchdog_<benchmark>_<iid>.log` (see watchdog_runner.py:333).
This script reads that file directly, parses every `[HZ-PROGRESS]` line,
and rebuilds the trace + Girard-fire detection.

This is a verification-only pass. It does NOT rerun any iid. It
overwrites `summary.csv` and `vgg_mini_atlas.json` for the targeted
atlas directory.

Usage:
    python research/vgg_mini_atlas_reparse.py [<atlas_dir>]

If <atlas_dir> is omitted, the most recent `vgg_mini_atlas_canonical_*`
is used.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ACT_ROOT = Path("/data1/Kane/ACT")
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


HZ_TRACE_RE_STRICT = re.compile(
    r"\[HZ-PROGRESS\]\s+L(\d+)\s+(\w+).*?(?:->|ng=)"
    r".*?ng=(\d+)\s+nb=(\d+)\s+nc=(\d+)"
)


def parse_hz_trace_full(text: str) -> List[Dict[str, Any]]:
    """Parse every end-state HZ-PROGRESS line in ``text``. End-state
    lines look like::

        [HZ-PROGRESS] L30 CONV2D -> dim=100352 ng=31278 nb=0 nc=0

    Start lines (``[HZ-PROGRESS] start L30 CONV2D in=...``) are skipped
    so each layer is recorded once.
    """
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if "[HZ-PROGRESS]" not in line:
            continue
        # Skip ``start`` lines (we want post-op state per layer).
        if "[HZ-PROGRESS] start " in line:
            continue
        m = HZ_TRACE_RE_STRICT.search(line)
        if not m:
            continue
        layer_id, op, ng, nb, nc = m.groups()
        dim_m = re.search(r"dim=(\d+)", line)
        out.append({
            "layer_id": int(layer_id),
            "op": op,
            "dim": int(dim_m.group(1)) if dim_m else None,
            "ng": int(ng),
            "nb": int(nb),
            "nc": int(nc),
        })
    return out


def find_girard_fires(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Layers where ng_post < ng_pre."""
    fires: List[Dict[str, Any]] = []
    for i in range(1, len(trace)):
        prev, cur = trace[i - 1], trace[i]
        if cur["ng"] < prev["ng"]:
            ng_pre = prev["ng"]
            fires.append({
                "layer_id": cur["layer_id"],
                "op": cur["op"],
                "ng_pre": ng_pre,
                "ng_post": cur["ng"],
                "ng_change": cur["ng"] - ng_pre,
                "ng_change_pct": (cur["ng"] - ng_pre) / ng_pre * 100.0
                if ng_pre > 0 else 0.0,
            })
    return fires


def reparse_one(per_iid_path: Path, run_outputs_root: Path) -> Dict[str, Any]:
    """Load the existing per-iid diag JSON and re-derive its layer
    trace / Girard fires from the full subprocess log.
    """
    d = json.load(open(per_iid_path))
    iid = int(d["iid"])
    iid_dir = run_outputs_root / f"iid{iid:03d}"
    log_path = iid_dir / f"watchdog_vggnet16_2022_{iid}.log"
    if not log_path.exists():
        d["_reparse_note"] = f"watchdog log missing at {log_path}"
        return d
    try:
        text = log_path.read_text(errors="replace")
    except Exception as e:
        d["_reparse_note"] = f"log read failed: {e}"
        return d
    trace = parse_hz_trace_full(text)
    fires = find_girard_fires(trace)
    d["layer_trace"] = trace
    d["n_layers_traced"] = len(trace)
    d["girard_fires"] = fires
    d["n_girard_fires"] = len(fires)
    d["_reparse_note"] = (
        f"reparsed from full watchdog log "
        f"({len(text)} bytes -> {len(trace)} layer rows, {len(fires)} Girard fires)"
    )
    return d


def main() -> int:
    if len(sys.argv) > 1:
        atlas_dir = Path(sys.argv[1])
    else:
        candidates = sorted(
            (ACT_ROOT / "audit_results").glob(
                "vgg_mini_atlas_canonical_*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print("no vgg_mini_atlas_canonical_* dir found", file=sys.stderr)
            return 1
        atlas_dir = candidates[-1]
    print(f"[reparse] atlas_dir = {atlas_dir}")
    per_iid_dir = atlas_dir / "per_iid"
    run_outputs_root = atlas_dir / "run_outputs"
    if not per_iid_dir.exists() or not run_outputs_root.exists():
        print(f"[reparse] missing per_iid/ or run_outputs/ under {atlas_dir}",
              file=sys.stderr)
        return 1

    entries: List[Dict[str, Any]] = []
    n_layers_total = 0
    n_girard_total = 0
    for piid in sorted(per_iid_dir.glob("iid*.json")):
        out = reparse_one(piid, run_outputs_root)
        entries.append(out)
        n_layers = out.get("n_layers_traced", 0) or 0
        n_girard = out.get("n_girard_fires", 0) or 0
        n_layers_total += n_layers
        n_girard_total += n_girard
        # Also re-write the per-iid JSON with the reparsed fields so
        # downstream consumers get the corrected data.
        with open(piid, "w") as f:
            json.dump(out, f, indent=2, default=float)

    # Rewrite aggregate vgg_mini_atlas.json with the existing top-level
    # metadata + the updated entries list.
    agg_path = atlas_dir / "vgg_mini_atlas.json"
    agg = {"entries": entries}
    if agg_path.exists():
        try:
            with open(agg_path) as f:
                old = json.load(f)
            old["entries"] = entries
            old["_reparse_stamp"] = "2026-06-04 advisor-directed parser fix"
            old["_reparse_n_layers_total"] = n_layers_total
            old["_reparse_n_girard_total"] = n_girard_total
            agg = old
        except Exception:
            pass
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=float)

    # Rewrite summary.csv with the reparsed fields.
    keys = [
        "iid", "production_verdict", "production_wall_s",
        "n_layers_traced", "n_girard_fires",
        "root_ng_at_flatten", "ng_at_flatten",
        "nc_at_flatten", "nb_at_flatten", "boxhz_collapse",
    ]
    csv_path = atlas_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for d in entries:
            w.writerow([d.get(k, "") for k in keys])

    # Quick sanity print.
    print()
    print(f"[reparse] total iids: {len(entries)}")
    print(f"[reparse] sum n_layers_traced: {n_layers_total}")
    print(f"[reparse] sum n_girard_fires : {n_girard_total}")
    print(f"[reparse] iids with >=1 girard fire: "
          f"{sum(1 for e in entries if (e.get('n_girard_fires') or 0) > 0)}")
    print()
    print("Per-iid trace summary:")
    for e in entries:
        layers = e.get("layer_trace") or []
        fires = e.get("girard_fires") or []
        layer_ids = sorted({L["layer_id"] for L in layers})
        fire_summary = ", ".join(
            f"L{f['layer_id']}({f['op']}: {f['ng_pre']}→{f['ng_post']})"
            for f in fires
        )
        print(f"  iid {e['iid']:>2}: layers={len(layer_ids)} "
              f"range=[L{min(layer_ids) if layer_ids else '-'}.."
              f"L{max(layer_ids) if layer_ids else '-'}] "
              f"fires=[{fire_summary or 'none'}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
