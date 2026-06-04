"""P0 step 2 — 20-sentinel topK=5 LP-witness dispatch.

Per advisor 2026-06-03 P0 directive. For each iid in the sentinel list:
  1. Parse y_true from vnnlib first-line label comment (NOT atlas — atlas
     was confirmed to use dataset labels instead of vnnlib labels).
  2. Forward HZ through conv body + bridge + tail (Step 2.1/2.2 path).
  3. Closed-form top-K rival LP upper bounds on (Y[r] - Y[t]).
  4. For each rival with LP UB > 0 (potential attack), extract closed-form
     xi_star, reconstruct x_cand, strict ORT replay.
  5. First replay with argmax != y_true → FAL receipt.
  6. If top-K[0] UB ≤ 0 → "all rivals safe by LP" (still UNK without a
     CERT receipt — this is a weaker upper bound than baseline).
  7. Else UNKNOWN.

Outputs per iid:
  - JSON receipt (the FAL one if found; otherwise the UNK summary).
Roll-up:
  - summary.csv listing verdict / top-1 LP UB / top-1 ORT margin per iid.

Principles respected: same as p0_all_rival_witness_smoke (no CROWN,
no backward, no Gurobi/MILP, no BaB, no PGD/random; witness from HZ/LP
feasibility, ORT only confirms).

Usage:
    python research/p0_batch_topk_dispatch.py
    python research/p0_batch_topk_dispatch.py --iids 0,11,84 --topK 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch

ACT_ROOT = Path(__file__).resolve().parent.parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.imagehz_cifar_prototype import load_instance, load_input_box  # noqa: E402
import os  # noqa: E402
from research.p0_all_rival_witness_smoke import (  # noqa: E402
    forward_to_bridge, walk_tail_track_xi, topk_rival_bounds,
    closed_form_witness_for_rival, strict_ort_replay,
)
from research.canonical_provenance import build_provenance  # noqa: E402


SENTINELS = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 21, 32, 33, 50, 84, 103]

LABEL_RE = re.compile(r";\s*CIFAR100\s+property\s+with\s+label:\s*(\d+)")


def parse_y_true_from_vnnlib(vnnlib_path: str) -> int:
    """Parse y_true from the first-line label comment.
    Format: '; CIFAR100 property with label: N.'
    """
    with open(vnnlib_path) as f:
        first = f.readline()
    m = LABEL_RE.search(first)
    if not m:
        raise RuntimeError(f"could not parse y_true from {vnnlib_path}: {first!r}")
    return int(m.group(1))


def dispatch_one_iid(
    iid: int, out_dir: Path, K: int,
) -> Dict[str, Any]:
    onnx_path, vnn_path = load_instance(iid)
    # 2026-06-03 REC2/REC3: every receipt records provenance.
    prov = build_provenance("cifar100_2024", iid).as_dict()
    y_true = parse_y_true_from_vnnlib(vnn_path)
    print(f"--- iid {iid} (y_true={y_true}) "
          f"canon-csv-sha={prov['instances_csv_sha256'][:8]} "
          f"vnnlib-sha={prov['vnnlib_sha256'][:8]} ---")

    m_onnx = onnx.load(onnx_path)
    in_dims = [d.dim_value for d in m_onnx.graph.input[0].type.tensor_type.shape.dim]
    C, H, W = int(in_dims[1]), int(in_dims[2]), int(in_dims[3])
    n_in = C * H * W
    lb, ub = load_input_box(vnn_path, n_in)
    c_box = (lb + ub) / 2.0
    half = (ub - lb) / 2.0

    t0 = time.perf_counter()
    sg_bridge, col_xi_bridge, fwd_meta = forward_to_bridge(onnx_path, lb, ub)
    t_walk = time.perf_counter() - t0
    t1 = time.perf_counter()
    sg_out, col_xi_out, _ = walk_tail_track_xi(
        sg_bridge, col_xi_bridge, onnx_path,
        next_xi=fwd_meta["next_xi_after_walk"],
    )
    t_tail = time.perf_counter() - t1
    n_classes = int(sg_out.n)

    top = topk_rival_bounds(sg_out, y_true=y_true, n_classes=n_classes, K=K)
    print(f"  conv_body+bridge={t_walk:.1f}s  tail={t_tail:.2f}s "
          f"(n_out={n_classes}, ng={sg_out.ng})")
    print(f"  top-{K} rivals:")
    for r, ub_val in top:
        print(f"    rival={r:3d}  LP UB = {ub_val:+.4f}")

    # Verdict shortcut: if top-1 UB <= 0, every rival is LP-safe.
    if top[0][1] <= 0:
        verdict = "ALL_RIVALS_LP_SAFE"
        print(f"  → {verdict}: top-1 LP UB = {top[0][1]:+.4f} ≤ 0")
        receipt = {
            "iid": iid, "y_true": y_true, "verdict": verdict,
            "topK_rivals_lp_ub": top, "n_classes": n_classes,
            "wall_conv_body_s": t_walk, "wall_tail_s": t_tail,
            "onnx_path": onnx_path, "vnnlib_path": vnn_path,
            "provenance": prov,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"p0_iid{iid:03d}_summary.json", "w") as f:
            json.dump(receipt, f, indent=2)
        return receipt

    # Try witness for each rival in top-K that has positive LP UB.
    candidate_log: List[Dict[str, Any]] = []
    fal_receipt = None
    for rival, lp_ub in top:
        if lp_ub <= 0:
            continue
        t_c0 = time.perf_counter()
        lp_max, xi_input, _ = closed_form_witness_for_rival(
            sg_out, col_xi_out, y_true=y_true, y_rival=rival,
            n_input=n_in,
        )
        x_cand = np.clip(c_box + half * xi_input, lb, ub)
        in_box = bool(np.all(x_cand >= lb - 1e-12)
                      and np.all(x_cand <= ub + 1e-12))
        y_ort = strict_ort_replay(onnx_path, x_cand, in_shape=(C, H, W))
        y_argmax = int(np.argmax(y_ort))
        margin_actual = float(y_ort[rival] - y_ort[y_true])
        t_c = time.perf_counter() - t_c0
        is_fal = (y_argmax != y_true) and (margin_actual > 0.0)
        entry = {
            "rival": rival,
            "lp_ub": lp_ub,
            "ort_margin": margin_actual,
            "ort_argmax": y_argmax,
            "phantom_gap": lp_ub - margin_actual,
            "wall_s": t_c,
            "input_box_holds": in_box,
            "is_fal": is_fal,
        }
        candidate_log.append(entry)
        tag = "FAL" if is_fal else "no-replay"
        print(f"    [rival {rival:3d}] LP UB={lp_ub:+.4f}  "
              f"ORT margin={margin_actual:+.4f}  argmax={y_argmax}  "
              f"phantom={lp_ub - margin_actual:+.4f}  → {tag}")
        if is_fal:
            fal_receipt = {
                "source": "p0_all_rival_imagehz_box_lp_witness",
                "iid": iid,
                "onnx_path": onnx_path,
                "vnnlib_path": vnn_path,
                "y_true": y_true,
                "target_rival": rival,
                "lp_upper_bound_y_r_minus_y_t": lp_ub,
                "input_box_holds": in_box,
                "ort_y_true_logit": float(y_ort[y_true]),
                "ort_y_rival_logit": float(y_ort[rival]),
                "ort_y_argmax": y_argmax,
                "ort_actual_margin": margin_actual,
                "argmax_matches_y_true": False,
                "is_falsified": True,
                "verdict": "FALSIFIED",
                "topK_rivals_lp_ub": top,
                "wall_conv_body_s": t_walk,
                "wall_tail_s": t_tail,
                "provenance": prov,
            }
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    if fal_receipt is not None:
        verdict = "FALSIFIED"
        path = out_dir / f"p0_iid{iid:03d}_FAL_rival{fal_receipt['target_rival']:03d}.json"
        with open(path, "w") as f:
            json.dump(fal_receipt, f, indent=2)
        print(f"  → FALSIFIED via rival {fal_receipt['target_rival']}")
        return {**fal_receipt, "candidate_log": candidate_log}

    # No replay produced FAL.
    verdict = "UNKNOWN_NO_FAL_REPLAYED"
    summary = {
        "iid": iid, "y_true": y_true, "verdict": verdict,
        "topK_rivals_lp_ub": top,
        "candidate_log": candidate_log,
        "n_classes": n_classes,
        "wall_conv_body_s": t_walk, "wall_tail_s": t_tail,
        "onnx_path": onnx_path, "vnnlib_path": vnn_path,
        "provenance": prov,
    }
    with open(out_dir / f"p0_iid{iid:03d}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {verdict}: {len(candidate_log)} rivals tried, 0 replay")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iids", type=str, default=",".join(str(i) for i in SENTINELS),
                    help="Comma-separated iids; default = 20-sentinel set")
    ap.add_argument("--topK", type=int, default=5)
    ap.add_argument(
        "--out", type=str,
        default="audit_results/cifar_unknown_margin_atlas_20260603/PHASE2_P0_SENTINEL20"
    )
    args = ap.parse_args()

    iids = [int(x) for x in args.iids.split(",") if x.strip()]
    out_dir = Path(args.out)

    results: List[Dict[str, Any]] = []
    skipped: List[int] = []
    t_all0 = time.perf_counter()
    for iid in iids:
        # Skip iids whose vnnlib is missing on disk (instances.csv can
        # reference files we don't have).
        _o, _v = load_instance(iid)
        if not os.path.exists(_v):
            print(f"--- iid {iid}: vnnlib not on disk, skipping ---")
            skipped.append(iid)
            continue
        r = dispatch_one_iid(iid, out_dir, K=args.topK)
        results.append(r)
    t_all = time.perf_counter() - t_all0

    # Roll-up.
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iid", "y_true", "verdict",
            "top1_rival", "top1_lp_ub",
            "first_fal_rival", "first_fal_ort_margin",
            "n_candidates_tried", "n_certified_safe_by_lp",
            "wall_conv_body_s", "wall_tail_s",
        ])
        for r in results:
            iid = r["iid"]
            y_true = r["y_true"]
            verdict = r["verdict"]
            top = r["topK_rivals_lp_ub"]
            top1_r, top1_ub = top[0]
            fal_rival = ""
            fal_margin = ""
            cand_log = r.get("candidate_log", [])
            n_tried = len(cand_log)
            n_safe = sum(1 for (_, ub_v) in top if ub_v <= 0)
            if verdict == "FALSIFIED":
                fal_rival = r["target_rival"]
                fal_margin = r["ort_actual_margin"]
            w.writerow([
                iid, y_true, verdict,
                top1_r, f"{top1_ub:+.4f}",
                fal_rival, fal_margin if isinstance(fal_margin, str) else f"{fal_margin:+.4f}",
                n_tried, n_safe,
                f"{r.get('wall_conv_body_s', 0):.2f}",
                f"{r.get('wall_tail_s', 0):.2f}",
            ])

    print()
    print(f"=== P0 sentinel-20 roll-up (wall {t_all:.1f}s) ===")
    n_fal = sum(1 for r in results if r["verdict"] == "FALSIFIED")
    n_safe = sum(1 for r in results if r["verdict"] == "ALL_RIVALS_LP_SAFE")
    n_unk = sum(1 for r in results if r["verdict"] == "UNKNOWN_NO_FAL_REPLAYED")
    print(f"  FAL receipts      : {n_fal}/{len(results)}")
    print(f"  ALL_RIVALS_LP_SAFE: {n_safe}/{len(results)}  (CIFAR HZ-box-LP certifies)")
    print(f"  UNKNOWN_NO_REPLAY : {n_unk}/{len(results)}")
    print(f"  summary CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
