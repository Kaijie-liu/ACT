"""Strict audit of the 546 forward-coefficient A_CONFIRMED receipts.

Same 4-check structure as audit_368_a_receipts.py but uses the
forward-coefficient decoder (decode_xi_star_forward) instead of the
backward W^T chain decoder. The audit is independent of the LP-UB bug
because it re-derives x_star from raw inputs and re-checks via ORT
replay at strict tolerance.

Checks per iid:
  1. input_box_holds: x_star ∈ [lb, ub] without clip
  2. spec_zero_tol_holds: d·y >= threshold at strict tolerance after ORT
  3. provenance_complete: 4 bundle keys present
  4. x_star_clip_required: false (witness is at the actual box corner)
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.canonical_provenance import load_instance, build_provenance  # noqa: E402
from research.sc_hz.onnx_walker import parse_onnx_to_layers  # noqa: E402
from research.sc_hz.vnnlib_parse import parse_vnnlib  # noqa: E402
from research.sc_hz.ort_replay import ort_replay_one  # noqa: E402
from research.sc_hz.forward_witness import (  # noqa: E402
    initial_state_with_lineage, forward_propagate_no_backward,
    decode_xi_star_forward,
)


def audit_one(bench: str, iid: int) -> dict:
    out = {
        "bench": bench, "iid": iid,
        "input_box_holds": None,
        "x_star_clip_required": None,
        "spec_zero_tol_holds": None,
        "provenance_complete": None,
        "overall_pass": False,
        "notes": [],
    }
    try:
        prov = build_provenance(bench, iid)
        out["provenance_complete"] = bool(
            prov.canonical_root and prov.instances_csv_sha256
            and prov.onnx_sha256 and prov.vnnlib_sha256
        )
        if not out["provenance_complete"]:
            out["notes"].append("provenance incomplete")
            return out

        onnx_path, vnn_path = load_instance(bench, iid)
        layers, input_shape, n_classes = parse_onnx_to_layers(str(onnx_path))
        n_in = int(np.prod(input_shape))
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_path), n_in, n_classes)
        c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2

        init = initial_state_with_lineage(c_in, r_in)
        state_out, _ = forward_propagate_no_backward(
            init, layers, K_per_layer=100000, initial_shape=input_shape,
        )

        any_pass = False
        for d_out, threshold, label in unsafe:
            x_star_uncl, _ = decode_xi_star_forward(
                state_out, d_out, c_in, r_in,
            )
            in_box = bool(
                np.all(x_star_uncl >= lb_x - 1e-12) and
                np.all(x_star_uncl <= ub_x + 1e-12)
            )
            x_star = np.clip(x_star_uncl, lb_x, ub_x)
            clip_req = bool(np.max(np.abs(x_star - x_star_uncl)) > 1e-12)
            try:
                y = ort_replay_one(str(onnx_path), x_star, input_shape)
            except Exception as e:
                out["notes"].append(f"{label}: ORT error: {str(e)[:100]}")
                continue
            cond = float(d_out @ y) > float(threshold)
            if cond:
                out["input_box_holds"] = in_box
                out["x_star_clip_required"] = clip_req
                out["spec_zero_tol_holds"] = True
                out["violating_label"] = label
                out["d_dot_y"] = float(d_out @ y)
                out["threshold"] = float(threshold)
                any_pass = True
                break

        if any_pass:
            out["overall_pass"] = (
                out["input_box_holds"]
                and out["spec_zero_tol_holds"]
                and out["provenance_complete"]
            )
        else:
            out["notes"].append("no unsafe condition holds at strict tolerance")
    except Exception as e:
        out["notes"].append(f"audit raised: {type(e).__name__}: {str(e)[:200]}")
    return out


def main() -> int:
    # Load 546 A iids from forward sweep
    out_dir = sorted(glob.glob(
        "/data1/Kane/ACT/audit_results/sc_hz_forward_safenlp_1080_*/"
    ))[-1]
    s = json.load(open(f"{out_dir}/summary.json"))
    a_iids = s["a_iids"]
    print(f"auditing {len(a_iids)} forward-coeff A_CONFIRMED iids...")

    audit_root = Path(out_dir) / "audit_546"
    audit_root.mkdir(exist_ok=True)
    results = []
    counters = Counter()
    for i, iid in enumerate(a_iids):
        if i % 100 == 0:
            print(f"  progress {i}/{len(a_iids)}...", flush=True)
        r = audit_one("safenlp_2024", iid)
        results.append(r)
        if r["overall_pass"]:
            counters["pass_strict"] += 1
        elif r.get("x_star_clip_required") and r.get("spec_zero_tol_holds"):
            counters["pass_with_clip_caveat"] += 1
        elif r.get("spec_zero_tol_holds") is None:
            counters["fail_no_witness"] += 1
        else:
            counters["fail_other"] += 1

    with open(audit_root / "audit_per_iid.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    summary = {
        "n_audited": len(a_iids), "categories": dict(counters),
        "strict_pass_count": counters["pass_strict"],
    }
    with open(audit_root / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== AUDIT RESULT on {len(a_iids)} forward A_CONFIRMED ===")
    for k, v in counters.items():
        print(f"  {k}: {v}")
    print(f"\nSTRICT-PASS: {counters['pass_strict']}/{len(a_iids)}")
    print(f"wrote {audit_root}/audit_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
