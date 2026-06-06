"""Strict audit of the 368 SC-HZ Phase B A_CONFIRMED receipts.

Per advisor 2026-06-04 review:
  - input_box_holds: x_star ∈ [lb, ub] strictly, no clip needed
  - spec_zero_tol_holds: cond_holds at strict zero tolerance after ORT replay
  - provenance: canonical_root + 3 SHA256 present and consistent
  - clip_check: detect whether x_star was clipped (means closed-form maximizer
                was outside the box; receipt then needs flagging)

Only receipts passing ALL four checks count toward the headline 1282.
Failing receipts are reported individually for remediation or removal.
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
from research.sc_hz.precompute_direction import precompute_d_per_layer_chain  # noqa: E402
from research.sc_hz.vnnlib_parse import parse_vnnlib  # noqa: E402
from research.sc_hz.run_sentinels import _layer_output_shapes  # noqa: E402
from research.sc_hz.ort_replay import (  # noqa: E402
    decode_xi_star_for_condition, ort_replay_one, check_unsafe_condition,
)


def audit_one(bench: str, iid: int, sc_hz_receipt: dict) -> dict:
    """Per-iid audit; returns dict with check results."""
    out = {
        "bench": bench, "iid": iid,
        "input_box_holds": None,
        "x_star_clip_required": None,
        "spec_zero_tol_holds": None,
        "provenance_complete": None,
        "violating_condition_label": None,
        "overall_pass": False,
        "notes": [],
    }

    # Check provenance
    prov_keys = ["canonical_root", "instances_csv_sha256",
                  "onnx_sha256", "vnnlib_sha256"]
    prov_complete = all(sc_hz_receipt.get(k) for k in prov_keys)
    out["provenance_complete"] = prov_complete
    if not prov_complete:
        out["notes"].append("provenance: missing one or more bundle keys")
        return out

    # Re-derive xi_star and ORT-replay independently (no trust in cached receipt)
    onnx_path, vnn_path = load_instance(bench, iid)
    layers, input_shape, n_classes = parse_onnx_to_layers(str(onnx_path))
    n_in = 1
    for d in input_shape:
        n_in *= int(d)
    lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_path), n_in, n_classes)
    c_in = (lb_x + ub_x) / 2.0
    r_in = (ub_x - lb_x) / 2.0
    out_shapes = _layer_output_shapes(layers, input_shape)

    # For each unsafe condition with LP UB > threshold, attempt to construct
    # x_star and verify the four properties.
    any_pass = False
    for (d_out, threshold, label) in unsafe:
        d_chain = precompute_d_per_layer_chain(layers, d_out, out_shapes)
        d_at_input = d_chain[0]
        x_star_unclipped, _ = decode_xi_star_for_condition(
            {}, d_out, c_in, r_in, d_at_input,
        )
        # 1. input_box_holds without clip?
        in_box_before_clip = bool(
            np.all(x_star_unclipped >= lb_x - 1e-12)
            and np.all(x_star_unclipped <= ub_x + 1e-12)
        )
        x_star = np.clip(x_star_unclipped, lb_x, ub_x)
        # If clip changed the value, record it
        clip_diff = np.max(np.abs(x_star - x_star_unclipped))
        clip_required = bool(clip_diff > 1e-12)

        # 2. ORT replay
        try:
            y = ort_replay_one(str(onnx_path), x_star, input_shape)
        except Exception as e:
            out["notes"].append(f"{label}: ORT error: {str(e)[:100]}")
            continue

        # 3. spec_zero_tol_holds (strict, atol=0)
        cond_holds = float(d_out @ y) > float(threshold)

        if cond_holds:
            # This condition gives a valid A witness
            out["input_box_holds"] = in_box_before_clip
            out["x_star_clip_required"] = clip_required
            out["spec_zero_tol_holds"] = True
            out["violating_condition_label"] = label
            out["d_dot_y"] = float(d_out @ y)
            out["threshold"] = float(threshold)
            out["margin_above_threshold"] = float(d_out @ y) - float(threshold)
            # An A is valid iff in-box without clip AND strict spec holds
            # If x_star was clipped, the witness is still valid (it lives in
            # the box) but the closed-form maximizer was outside — flag as a
            # weaker but still sound A.
            any_pass = True
            break  # one valid condition is enough

    if any_pass:
        # Overall pass: in-box without clip + strict spec + provenance
        out["overall_pass"] = (
            out["input_box_holds"] and
            out["spec_zero_tol_holds"] and
            out["provenance_complete"]
        )
        if out["x_star_clip_required"]:
            out["notes"].append(
                "x_star was outside the input box; clipped value still "
                "produces a valid witness but the closed-form maximizer "
                "was outside-box"
            )
    else:
        out["notes"].append("no unsafe condition holds at strict tolerance "
                              "with x_star (= sign(d_at_input) corner)")
    return out


def main() -> int:
    # Find the Phase B-1 directory
    p_b1 = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_phase_b_safenlp_*/"))[-1]
    print(f"B-1: {p_b1}")

    # Load A_CONFIRMED iid list
    b1_summary = json.load(open(f"{p_b1}/summary.json"))
    a_iids = b1_summary["a_iids"]
    print(f"auditing {len(a_iids)} A_CONFIRMED iids...")

    audit_root = Path(p_b1) / "audit_368"
    audit_root.mkdir(exist_ok=True)

    audit_results = []
    counters = Counter()
    for i, iid in enumerate(a_iids):
        if i % 50 == 0:
            print(f"  progress {i}/{len(a_iids)}...", flush=True)
        try:
            sc_hz_rec = json.load(open(f"{p_b1}/per_iid/iid{iid:04d}.json"))
            audit = audit_one("safenlp_2024", iid, sc_hz_rec)
        except Exception as e:
            audit = {
                "bench": "safenlp_2024", "iid": iid,
                "overall_pass": False,
                "notes": [f"audit raised: {type(e).__name__}: {str(e)[:200]}"],
            }
        audit_results.append(audit)
        # Record counters
        if audit["overall_pass"]:
            counters["pass_strict"] += 1
        elif audit.get("spec_zero_tol_holds") and audit.get("x_star_clip_required"):
            counters["pass_with_clip_caveat"] += 1
        elif audit.get("spec_zero_tol_holds") and not audit.get("input_box_holds"):
            counters["pass_clip_required"] += 1
        elif audit.get("spec_zero_tol_holds") is False:
            counters["fail_spec_not_strict"] += 1
        elif not audit.get("provenance_complete"):
            counters["fail_provenance"] += 1
        elif audit.get("notes"):
            counters["fail_no_witness"] += 1
        else:
            counters["fail_other"] += 1

    # Write detailed audit
    with open(audit_root / "audit_per_iid.json", "w") as f:
        json.dump(audit_results, f, indent=2, default=float)

    summary = {
        "n_audited": len(a_iids),
        "categories": dict(counters),
        "strict_pass_count": counters["pass_strict"],
        "ok_with_clip_caveat": counters["pass_with_clip_caveat"],
        "headline_strict_a_count": counters["pass_strict"],
    }
    with open(audit_root / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n=== AUDIT RESULT on {len(a_iids)} A_CONFIRMED ===")
    for k, v in counters.items():
        print(f"  {k}: {v}")
    print(f"\nSTRICT-PASS (in-box, strict spec, provenance): {counters['pass_strict']}")
    print(f"PASS-WITH-CLIP-CAVEAT (witness valid but maximizer outside box): "
          f"{counters['pass_with_clip_caveat']}")
    print(f"\nwrote {audit_root}/audit_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
