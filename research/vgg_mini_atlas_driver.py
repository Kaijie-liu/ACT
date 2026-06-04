"""VGG mini-atlas driver — §6b ImageHZ go/no-go gate.

Per advisor 2026-06-04. For up to 18 canonical VGG iids (full pool is
small enough to run all of them as the "mini" atlas), record:

  - per-layer trace from `[HZ-PROGRESS]`: layer ID, op kind, dim, ng, nc, nb
  - Girard cap fires: layers where ng_post < ng_pre
  - root_ng at final FLATTEN snapshot vs n_input → root_factor_preserved_ratio
  - BoxHZ fallback: snapshot's `has_Gc` False / nc=nb=0
  - production verdict + wall + peak RSS

Output: `audit_results/vgg_mini_atlas_canonical_<STAMP>/`
  - per_iid/iid<NNN>.json
  - vgg_mini_atlas.json (aggregate)
  - summary.csv

Gate decision (advisor §6b):
  PROCEED to ImageHZ prototype only if
    root_factor_preserved_ratio < 0.95 in >= 5% of VGG iids
    AND loss is concentrated at a Girard cap site
  STOP otherwise.

Principles preserved: forward-only diagnostic, no CROWN/backward/Gurobi/
B&B/PGD/random. No FAL claims; replay margins are diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ACT_ROOT = Path("/data1/Kane/ACT")
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.canonical_provenance import (  # noqa: E402
    CANONICAL_ROOT, build_provenance, load_instance,
)


PY = "/data1/Kane/miniconda3/envs/act-py312/bin/python"

HZ_TRACE_RE = re.compile(
    r"\[HZ-PROGRESS\]\s+(?:start\s+)?L(\d+)\s+(\w+)"
    r".*?(?:->|in).*?(?:dim=(\d+))?\s*"
    r"ng=(\d+)\s+nb=(\d+)\s+nc=(\d+)"
)


def parse_hz_trace(stdout: str) -> List[Dict[str, Any]]:
    """Parse [HZ-PROGRESS] lines from watchdog_runner stdout.
    Returns one record per ``L<id> OP -> dim=... ng=... nb=... nc=...``
    line (only end-state lines, not start lines).
    """
    out: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        if "[HZ-PROGRESS]" not in line:
            continue
        if "->" not in line:
            continue  # skip "start" lines
        m = HZ_TRACE_RE.search(line)
        if not m:
            # Fallback: simpler regex
            simple = re.search(
                r"\[HZ-PROGRESS\]\s+L(\d+)\s+(\w+).*?ng=(\d+)\s+nb=(\d+)\s+nc=(\d+)",
                line,
            )
            if not simple:
                continue
            layer_id, op, ng, nb, nc = simple.groups()
            dim = ""
            d_m = re.search(r"dim=(\d+)", line)
            if d_m:
                dim = d_m.group(1)
        else:
            layer_id, op, dim, ng, nb, nc = m.groups()
        out.append({
            "layer_id": int(layer_id),
            "op": op,
            "dim": int(dim) if dim else None,
            "ng": int(ng),
            "nb": int(nb),
            "nc": int(nc),
        })
    return out


def find_girard_fires(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect layers where ng_post < ng_pre (Girard cap fired).
    Returns list of ``{layer_id, op, ng_pre, ng_post, ng_change}``.
    """
    fires: List[Dict[str, Any]] = []
    if len(trace) < 2:
        return fires
    for i, cur in enumerate(trace):
        if i == 0:
            continue
        prev = trace[i - 1]
        if cur["ng"] < prev["ng"]:
            fires.append({
                "layer_id": cur["layer_id"],
                "op": cur["op"],
                "ng_pre": prev["ng"],
                "ng_post": cur["ng"],
                "ng_change": cur["ng"] - prev["ng"],
                "ng_change_pct": (
                    (cur["ng"] - prev["ng"]) / prev["ng"] * 100.0
                    if prev["ng"] > 0 else 0.0
                ),
            })
    return fires


def run_one_vgg_iid(
    iid: int, out_dir: Path, *,
    wall_s: int = 500, rss_gb: int = 40,
) -> Dict[str, Any]:
    onnx_p, vnn_p = load_instance("vggnet16_2022", iid)
    prov = build_provenance("vggnet16_2022", iid).as_dict()

    snap_dir = out_dir / "snapshots" / f"iid{iid:03d}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    run_out = out_dir / "run_outputs" / f"iid{iid:03d}"
    run_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/data1/Kane/ACT"
    env["ACT_VNNLIB_ROOT"] = str(CANONICAL_ROOT)
    # Trace + memory-safe path per VGG forensic finding.
    env["ACT_HZ_LAYER_PROGRESS"] = "1"
    env["ACT_HZ_CONV_DEBUG"] = "1"
    env["ACT_HZ_CONV_FALLBACK_SAFE"] = "1"
    env["ACT_HZ_GIRARD_PRESERVE_ROOT"] = "1"
    # Snapshot for the final FLATTEN.
    env["ACT_HZ_ENDCAP_SNAPSHOT_DIR"] = str(snap_dir)
    env["ACT_HZ_ENDCAP_SNAPSHOT_KIND"] = "FLATTEN"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    cmd = [
        PY, "-m", "act.pipeline.watchdog_runner",
        "--benchmark", "vggnet16_2022",
        "--instance-ids", str(iid),
        "--wall-s", str(wall_s),
        "--device", "cuda", "--dtype", "float64",
        "--rss-cap-gb", str(rss_gb),
        "--out-dir", str(run_out),
        "--canonical-root", str(CANONICAL_ROOT),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=str(ACT_ROOT),
            capture_output=True, text=True, timeout=wall_s + 60,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        rc = -1
        stdout = ""
        stderr = "TIMEOUT_AT_DRIVER"
    wall = time.perf_counter() - t0

    # The stdout written by watchdog_runner is embedded inside its own
    # per_instance.json's `stdout_tail` field; grep there too.
    trace_lines = stdout
    pij = list(run_out.glob("per_instance*.json"))
    verdict = ""
    if pij:
        try:
            d = json.load(open(pij[0]))
            for r in d.get("per_instance", []):
                verdict = r.get("reportable_status", "")
                # The stdout_tail has the [HZ-PROGRESS] lines.
                tail = r.get("stdout_tail", "")
                if tail:
                    trace_lines += "\n" + tail
                break
        except Exception:
            pass

    trace = parse_hz_trace(trace_lines)
    fires = find_girard_fires(trace)

    # Load snapshot if present.
    snap_files = sorted(snap_dir.glob("L*_FLATTEN.pkl"))
    snap_info: Dict[str, Any] = {}
    if snap_files:
        try:
            with open(snap_files[0], "rb") as f:
                snap = pickle.load(f)
            snap_info = {
                "snapshot_path": str(snap_files[0]),
                "has_Gc": "Gc" in snap,
                "has_Gc_sparse": "Gc_sparse" in snap,
                "has_lb_ub": "lb" in snap and "ub" in snap,
                "dim": int(snap.get("c").numel() if hasattr(snap.get("c"), "numel")
                          else len(snap.get("c", []))),
                "ng": int(snap.get("ng", 0)),
                "nb": int(snap.get("nb", 0)),
                "nc": int(snap.get("nc", 0)),
                "root_ng": int(snap.get("root_ng", 0)),
            }
        except Exception as e:
            snap_info = {"snapshot_load_error": f"{type(e).__name__}: {e}"}

    return {
        "iid": iid,
        "provenance": prov,
        "production_returncode": rc,
        "production_verdict": verdict,
        "production_wall_s": wall,
        "n_layers_traced": len(trace),
        "layer_trace": trace,
        "girard_fires": fires,
        "n_girard_fires": len(fires),
        "snapshot": snap_info,
        # Diagnostic fields the gate cares about
        "root_ng_at_flatten": snap_info.get("root_ng"),
        "ng_at_flatten": snap_info.get("ng"),
        "nc_at_flatten": snap_info.get("nc"),
        "nb_at_flatten": snap_info.get("nb"),
        "boxhz_collapse": (
            bool(snap_info.get("nc") == 0 and snap_info.get("nb") == 0)
            if snap_info.get("ng") is not None else None
        ),
    }


# ── Driver ────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iids", type=str, default="",
                    help="comma-separated iids; default = all 18 canonical VGG iids")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--wall-s", type=int, default=500)
    ap.add_argument("--rss-gb", type=int, default=40)
    args = ap.parse_args()

    if args.iids:
        iids = [int(x) for x in args.iids.split(",") if x.strip()]
    else:
        from research.canonical_provenance import canonical_instances_rows
        n = len(canonical_instances_rows("vggnet16_2022"))
        iids = list(range(n))

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_base = (
        Path(args.out) if args.out else
        Path(f"/data1/Kane/ACT/audit_results/vgg_mini_atlas_canonical_{stamp}")
    )
    out_base.mkdir(parents=True, exist_ok=True)
    per_iid_dir = out_base / "per_iid"
    per_iid_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    t_all0 = time.perf_counter()
    for k, iid in enumerate(iids):
        print(f"--- [{k+1:>3}/{len(iids)}] VGG iid {iid} ---", flush=True)
        try:
            diag = run_one_vgg_iid(
                iid, out_base, wall_s=args.wall_s, rss_gb=args.rss_gb,
            )
        except Exception as e:
            diag = {
                "iid": iid, "verdict": "ERROR",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:500],
            }
        with open(per_iid_dir / f"iid{iid:03d}.json", "w") as f:
            json.dump(diag, f, indent=2, default=float)
        summary.append(diag)
        s = diag.get("snapshot", {}) or {}
        print(
            f"  verdict={diag.get('production_verdict'):<24} "
            f"wall={diag.get('production_wall_s', 0):.1f}s  "
            f"n_layers={diag.get('n_layers_traced', 0)}  "
            f"girard_fires={diag.get('n_girard_fires', 0)}  "
            f"root_ng={s.get('root_ng')}  "
            f"ng={s.get('ng')}  "
            f"nc={s.get('nc')}  nb={s.get('nb')}  "
            f"BoxHZ={diag.get('boxhz_collapse')}",
            flush=True,
        )

    t_all = time.perf_counter() - t_all0

    # Roll up
    with open(out_base / "vgg_mini_atlas.json", "w") as f:
        json.dump({
            "stamp": stamp,
            "canonical_root": str(CANONICAL_ROOT),
            "n_iids": len(summary),
            "wall_s_total": t_all,
            "entries": summary,
        }, f, indent=2, default=float)

    keys = [
        "iid", "production_verdict", "production_wall_s",
        "n_layers_traced", "n_girard_fires",
        "root_ng_at_flatten", "ng_at_flatten",
        "nc_at_flatten", "nb_at_flatten", "boxhz_collapse",
    ]
    with open(out_base / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for d in summary:
            w.writerow([d.get(k, "") for k in keys])

    # Gate computation
    print()
    print(f"=== VGG mini-atlas roll-up (wall {t_all:.1f}s) ===")
    ok = [d for d in summary if isinstance(d.get("snapshot"), dict)
          and d["snapshot"].get("root_ng") is not None]
    if not ok:
        print("  no usable snapshots — atlas inconclusive")
        return 1

    # We don't know n_input per VGG iid without parsing vnnlib;
    # for VGG, the snapshot's root_ng is most directly compared to
    # the network's nominal n_input = 3*224*224 = 150528.
    N_INPUT = 3 * 224 * 224
    correlation_lost = [
        d for d in ok
        if d["snapshot"]["root_ng"] is not None
        and d["snapshot"]["root_ng"] / N_INPUT < 0.95
    ]
    n_girard = sum(1 for d in ok if d.get("n_girard_fires", 0) > 0)
    print(f"  total ok: {len(ok)}")
    print(f"  root_ng/n_input < 0.95 (correlation-loss):  "
          f"{len(correlation_lost)} ({len(correlation_lost)/len(ok)*100:.1f}%)")
    print(f"  iids with Girard fires:  {n_girard} ({n_girard/len(ok)*100:.1f}%)")
    print()
    pct = (len(correlation_lost) / len(ok)) * 100 if ok else 0
    if pct >= 5 and any(d.get("n_girard_fires", 0) > 0 for d in correlation_lost):
        print("  GATE: PROCEED to ImageHZ prototype on VGG (>= 5% correlation-loss)")
    else:
        print(f"  GATE: STOP ImageHZ (correlation-loss share = {pct:.1f}% < 5%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
