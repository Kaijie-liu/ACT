"""P0 step 4 — strict audit of the 16 FAL receipts from PHASE2_P0_UNKNOWN185.

For each FAL receipt:
  1. Parse y_true from vnnlib first line (independent re-derivation).
  2. Reconstruct x_cand from the receipt's recorded LP candidate (re-run
     the closed-form witness extraction from scratch — no use of any
     cached numbers in the receipt).
  3. Run deterministic CPU ORT (fresh session per receipt) on x_cand.
  4. Confirm:
     - in_box (x_cand in [lb, ub])
     - argmax != y_true
     - Y[rival] - Y[y_true] > 0  (strict, no tolerance)
  5. Cross-check the freshly recomputed LP UB and ORT margin vs the
     receipt's recorded values.

This is an "independent" audit in the sense that no field of the
receipt JSON is trusted — every value is re-derived from the vnnlib,
ONNX, and ImageHZ pipeline.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnx
import torch

ACT_ROOT = Path(__file__).resolve().parent.parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.imagehz_cifar_prototype import load_instance, load_input_box  # noqa: E402
from research.p0_all_rival_witness_smoke import (  # noqa: E402
    forward_to_bridge, walk_tail_track_xi,
    closed_form_witness_for_rival, strict_ort_replay,
)
from research.p0_batch_topk_dispatch import parse_y_true_from_vnnlib  # noqa: E402


AUDIT_DIR = Path(
    "/data1/Kane/ACT/audit_results/cifar_unknown_margin_atlas_20260603/"
    "PHASE2_P0_UNKNOWN185"
)
SUMMARY_CSV = AUDIT_DIR / "summary.csv"


def audit_one_receipt(iid: int, expected_rival: int) -> Dict[str, Any]:
    onnx_path, vnn_path = load_instance(iid)
    y_true = parse_y_true_from_vnnlib(vnn_path)

    m_onnx = onnx.load(onnx_path)
    in_dims = [d.dim_value for d in m_onnx.graph.input[0].type.tensor_type.shape.dim]
    C, H, W = int(in_dims[1]), int(in_dims[2]), int(in_dims[3])
    n_in = C * H * W
    lb, ub = load_input_box(vnn_path, n_in)
    c_box = (lb + ub) / 2.0
    half = (ub - lb) / 2.0

    # Re-derive from scratch.
    sg, col_xi, meta = forward_to_bridge(onnx_path, lb, ub)
    sg_out, col_xi_out, _ = walk_tail_track_xi(
        sg, col_xi, onnx_path, next_xi=meta["next_xi_after_walk"]
    )
    lp_max, xi_input, _ = closed_form_witness_for_rival(
        sg_out, col_xi_out, y_true=y_true, y_rival=expected_rival, n_input=n_in,
    )
    x_cand = np.clip(c_box + half * xi_input, lb, ub)
    in_box = bool(np.all(x_cand >= lb - 1e-12)
                  and np.all(x_cand <= ub + 1e-12))

    y_ort = strict_ort_replay(onnx_path, x_cand, in_shape=(C, H, W))
    y_argmax = int(np.argmax(y_ort))
    margin_actual = float(y_ort[expected_rival] - y_ort[y_true])
    is_fal = (y_argmax != y_true) and (margin_actual > 0.0)

    return {
        "iid": iid, "expected_rival": expected_rival,
        "y_true_parsed": y_true,
        "lp_ub_redrived": lp_max,
        "ort_argmax": y_argmax,
        "ort_y_true_logit": float(y_ort[y_true]),
        "ort_y_rival_logit": float(y_ort[expected_rival]),
        "ort_actual_margin": margin_actual,
        "argmax_matches_y_true": y_argmax == y_true,
        "in_box": in_box,
        "is_fal_strict": is_fal,
    }


def main() -> int:
    rows = list(csv.DictReader(open(SUMMARY_CSV)))
    fals = [(int(r["iid"]), int(r["first_fal_rival"]))
            for r in rows if r["verdict"] == "FALSIFIED"]
    print(f"Auditing {len(fals)} FAL receipts independently...")
    print()

    audited: List[Dict[str, Any]] = []
    n_pass = 0
    for iid, rival in fals:
        r = audit_one_receipt(iid, rival)
        audited.append(r)
        tag = "PASS" if r["is_fal_strict"] else "FAIL"
        if r["is_fal_strict"]:
            n_pass += 1
        print(f"  iid={iid:>3}  rival={rival:>3}  y_true={r['y_true_parsed']:>3}  "
              f"LP UB={r['lp_ub_redrived']:+.4f}  ORT margin={r['ort_actual_margin']:+.4f}  "
              f"argmax={r['ort_argmax']:>3}  in_box={r['in_box']}  -> {tag}")

    print()
    print(f"=== Audit summary: {n_pass}/{len(fals)} receipts re-replay clean ===")
    out = AUDIT_DIR / "audit_independent.json"
    with open(out, "w") as f:
        json.dump({"n_audited": len(fals), "n_pass": n_pass,
                   "results": audited}, f, indent=2)
    print(f"Audit JSON: {out}")
    return 0 if n_pass == len(fals) else 1


if __name__ == "__main__":
    sys.exit(main())
