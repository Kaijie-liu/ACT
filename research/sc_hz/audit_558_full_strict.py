"""Gate 0: full strict-> audit on ALL 558 safenlp A_CONFIRMED.

Per advisor 2026-06-05 post-S3 directive: the 53-sample audit is not enough.
Run the full 558 audit under STRICT `>` rule (G4 binding), with:
  - independent re-derive of forward state
  - all S1 candidate menu tried per condition
  - require d.y > threshold STRICTLY (margin > 1e-12)
  - input_box_holds with NO clip
  - provenance bundle complete

Outputs per iid:
  - STRICT_PASS: bool
  - witness_label: str or None
  - margin: float
  - in_box_no_clip: bool
  - provenance_complete: bool

Aggregate:
  - Of 558 A, how many strict-pass
  - Headline correction if any boundary witnesses found
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def audit_one_iid(bench: str, iid: int) -> dict:
    from research.canonical_provenance import load_instance, build_provenance
    from research.sc_hz.onnx_walker import parse_onnx_to_layers
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.forward_witness import (
        initial_state_with_lineage, forward_propagate_no_backward,
    )
    from research.sc_hz.ops import lp_ub_rival_margin
    from research.sc_hz.ort_replay import ort_replay_one
    from research.sc_hz.s1_phantom_repair import generate_candidates

    out = {"bench": bench, "iid": iid, "STRICT_PASS": False}
    try:
        prov = build_provenance(bench, iid)
        prov_ok = bool(prov.canonical_root and prov.instances_csv_sha256
                        and prov.onnx_sha256 and prov.vnnlib_sha256)
        out["provenance_complete"] = prov_ok
        if not prov_ok:
            out["fail_reason"] = "provenance"
            return out

        onnx_p, vnn_p = load_instance(bench, iid)
        layers, in_shape, n_classes = parse_onnx_to_layers(str(onnx_p))
        n_in = int(np.prod(in_shape))
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_p), n_in, n_classes)
        c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2

        init = initial_state_with_lineage(c_in, r_in)
        state_out, _ = forward_propagate_no_backward(
            init, layers, K_per_layer=100000, initial_shape=in_shape,
        )

        best_margin = -np.inf
        best_witness = None
        for d_out, threshold, label in unsafe:
            if lp_ub_rival_margin(state_out, d_out) < float(threshold):
                continue
            for cand_label, x_star in generate_candidates(
                state_out, d_out, c_in, r_in, lb_x, ub_x,
            ):
                in_box = bool(
                    np.all(x_star >= lb_x - 1e-12)
                    and np.all(x_star <= ub_x + 1e-12)
                )
                if not in_box:
                    continue
                x_star_c = np.clip(x_star, lb_x, ub_x)
                if np.max(np.abs(x_star_c - x_star)) > 1e-12:
                    continue  # clip required, reject
                try:
                    y = ort_replay_one(str(onnx_p), x_star_c, in_shape)
                except Exception:
                    continue
                d_y = float(d_out @ y)
                margin = d_y - float(threshold)
                if margin > best_margin:
                    best_margin = margin
                    best_witness = {
                        "cond_label": label, "cand_label": cand_label,
                        "d_dot_y": d_y, "threshold": float(threshold),
                        "margin": margin,
                    }
                # STRICT >: must have margin > tolerance
                if margin > 1e-12:
                    break  # found strict witness, can stop this condition
            if best_witness is not None and best_witness["margin"] > 1e-12:
                break  # no need to try other conditions

        out["best_witness"] = best_witness
        out["best_margin"] = float(best_margin) if best_margin > -np.inf else None
        out["STRICT_PASS"] = (
            best_witness is not None
            and best_witness["margin"] > 1e-12
            and prov_ok
        )
        if not out["STRICT_PASS"]:
            if best_witness is None:
                out["fail_reason"] = "no_witness_found"
            elif best_margin <= 1e-12:
                out["fail_reason"] = f"margin_not_strict ({best_margin:.3e})"
    except Exception as e:
        out["fail_reason"] = f"audit_exception: {type(e).__name__}: {str(e)[:200]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load union of forward 546 + S1 12
    p_fwd = sorted(glob.glob(
        "/data1/Kane/ACT/audit_results/sc_hz_forward_safenlp_1080_*/"
    ))[-1]
    p_s1 = sorted(glob.glob(
        "/data1/Kane/ACT/audit_results/sc_hz_s1_phantom_repair_full381_*/"
    ))[-1]
    fwd_a = set(json.load(open(f"{p_fwd}/summary.json"))["a_iids"])
    s1_a = set(json.load(open(f"{p_s1}/summary.json"))["a_iids"])
    all_a = sorted(fwd_a | s1_a)
    print(f"Auditing {len(all_a)} safenlp A_CONFIRMED ({len(fwd_a)} fwd + {len(s1_a)} S1, "
          f"union)", flush=True)

    t0 = time.perf_counter()
    results = []
    n_strict = 0; n_boundary = 0; n_fail = 0
    boundary_iids = []; fail_iids = []
    for i, iid in enumerate(all_a):
        if i % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i}/{len(all_a)} ({elapsed:.0f}s) — STRICT={n_strict}, "
                  f"boundary={n_boundary}, fail={n_fail}", flush=True)
        r = audit_one_iid("safenlp_2024", iid)
        results.append(r)
        if r["STRICT_PASS"]:
            n_strict += 1
        else:
            m = r.get("best_margin")
            if m is not None and abs(m) <= 1e-12:
                n_boundary += 1
                boundary_iids.append(iid)
            else:
                n_fail += 1
                fail_iids.append(iid)

    wall = time.perf_counter() - t0
    summary = {
        "n_audited": len(all_a),
        "n_strict_pass": n_strict,
        "n_boundary_margin_eq_zero": n_boundary,
        "n_fail_other": n_fail,
        "boundary_iids": boundary_iids,
        "fail_iids": fail_iids,
        "wall_seconds": wall,
        "headline_holds_if_n_strict_equals_n_audited": n_strict == len(all_a),
    }
    with open(out_dir / "audit_per_iid.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n=== GATE 0 FULL AUDIT RESULT ===")
    print(f"n_audited:            {len(all_a)}")
    print(f"n_STRICT_PASS:        {n_strict}")
    print(f"n_boundary (margin=0): {n_boundary}")
    print(f"n_fail_other:         {n_fail}")
    print(f"wall:                 {wall:.1f}s")
    if n_strict == len(all_a):
        print(f"\n✓ HEADLINE 1472 HOLDS — all {len(all_a)} pass strict G4")
    else:
        n_lost = len(all_a) - n_strict
        new_a = n_strict - 10  # subtract 10 matched-production
        print(f"\n✗ HEADLINE CORRECTION: {n_lost} A demoted")
        print(f"  New A count: {n_strict} - 10 matched = {new_a}")
        print(f"  Corrected headline: 924 + {new_a} = {924 + new_a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
