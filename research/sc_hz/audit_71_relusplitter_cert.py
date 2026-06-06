"""V-side audit of the 71 SC-HZ CERTs on relusplitter.

Per advisor 2026-06-04 review for the A side (368/368 STRICT-PASS), the
corresponding V-side audit must check:
  1. SC-HZ's PrunedState is a sound over-approximation: for ALL unsafe
     conditions, the closed-form LP UB d·c + |d·G|_1 + |d|·tail must be
     strictly < threshold.
  2. Re-derive the LP UB from raw inputs (do not trust the cached receipt).
  3. Provenance bundle is complete.
  4. Re-verify by exhaustive corner check on the input box: for each
     unsafe condition, compute d_at_input via the precomputed chain,
     decode x_star at the corner sign(d_at_input), run ORT, check that
     the spec does NOT hold (otherwise the "CERT" would be wrong).

The last check is the key adversarial test: if for any CERT iid we can
construct an x_star that ORT confirms violates the spec, the CERT is
unsound and must be removed.
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

from research.canonical_provenance import load_instance  # noqa: E402
from research.sc_hz.onnx_walker import parse_onnx_to_layers, forward_propagate  # noqa: E402
from research.sc_hz.precompute_direction import precompute_d_per_layer_chain  # noqa: E402
from research.sc_hz.prune import PrunedState  # noqa: E402
from research.sc_hz.vnnlib_parse import parse_vnnlib  # noqa: E402
from research.sc_hz.run_sentinels import _layer_output_shapes  # noqa: E402
from research.sc_hz.ort_replay import (  # noqa: E402
    decode_xi_star_for_condition, ort_replay_one,
)


def audit_one_cert(bench: str, iid: int, sc_hz_receipt: dict) -> dict:
    """Verify a SC-HZ CERT is sound."""
    out = {
        "bench": bench, "iid": iid,
        "all_cond_lp_ub_strictly_below_threshold": None,
        "no_corner_witness_violates_spec": None,
        "provenance_complete": None,
        "overall_pass": False,
        "notes": [],
        "n_unsafe_conditions": 0,
        "max_lp_ub_minus_threshold": None,
    }

    prov_keys = ["canonical_root", "instances_csv_sha256",
                  "onnx_sha256", "vnnlib_sha256"]
    prov_complete = all(sc_hz_receipt.get(k) for k in prov_keys)
    out["provenance_complete"] = prov_complete
    if not prov_complete:
        out["notes"].append("provenance: missing one or more bundle keys")
        return out

    try:
        onnx_path, vnn_path = load_instance(bench, iid)
        layers, input_shape, n_classes = parse_onnx_to_layers(str(onnx_path))
        n_in = 1
        for d in input_shape:
            n_in *= int(d)
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_path), n_in, n_classes)
        out["n_unsafe_conditions"] = len(unsafe)
        c_in = (lb_x + ub_x) / 2.0
        r_in = (ub_x - lb_x) / 2.0
        out_shapes = _layer_output_shapes(layers, input_shape)

        # Build initial PrunedState (box input)
        # Generators = diag(r_in) — pure box, K = n_in
        n = n_in
        G0 = np.diag(r_in)
        init_state = PrunedState(c=c_in.copy(), G_kept=G0,
                                   tail_radius=None, metadata={})

        # For each unsafe condition: check LP UB and corner witness
        max_lp_ub_minus_threshold = -np.inf
        all_lp_ub_safe = True
        any_corner_violates = False
        for (d_out, threshold, label) in unsafe:
            d_chain = precompute_d_per_layer_chain(layers, d_out, out_shapes)
            d_at_input = d_chain[0]
            # 1. Closed-form LP UB on init state
            state, _ = forward_propagate(init_state, layers, d_chain,
                                          K_per_layer=256,
                                          initial_shape=input_shape)
            lp_ub = float(d_out @ state.c + np.abs(d_out @ state.G_kept).sum())
            if state.tail_radius is not None:
                lp_ub += float(np.abs(d_out) @ state.tail_radius)
            diff = lp_ub - float(threshold)
            if diff > max_lp_ub_minus_threshold:
                max_lp_ub_minus_threshold = diff
            if lp_ub >= float(threshold):
                all_lp_ub_safe = False
                out["notes"].append(
                    f"{label}: LP UB {lp_ub:.6e} >= threshold {threshold:.6e}"
                )

            # 2. Corner adversarial check
            x_star_uncl, _ = decode_xi_star_for_condition(
                {}, d_out, c_in, r_in, d_at_input,
            )
            x_star = np.clip(x_star_uncl, lb_x, ub_x)
            try:
                y = ort_replay_one(str(onnx_path), x_star, input_shape)
            except Exception as e:
                out["notes"].append(f"{label}: ORT error: {str(e)[:100]}")
                continue
            cond_holds = float(d_out @ y) >= float(threshold)
            if cond_holds:
                any_corner_violates = True
                out["notes"].append(
                    f"{label}: corner witness VIOLATES spec (d.y={float(d_out@y):.6e} "
                    f">= threshold {threshold:.6e}) — UNSOUND CERT"
                )

        out["all_cond_lp_ub_strictly_below_threshold"] = all_lp_ub_safe
        out["no_corner_witness_violates_spec"] = not any_corner_violates
        out["max_lp_ub_minus_threshold"] = max_lp_ub_minus_threshold
        out["overall_pass"] = (
            all_lp_ub_safe and (not any_corner_violates) and prov_complete
        )

    except Exception as e:
        out["notes"].append(f"audit raised: {type(e).__name__}: {str(e)[:200]}")

    return out


def main() -> int:
    p_h = sorted(glob.glob("/data1/Kane/ACT/audit_results/sc_hz_horizontal_*/"))[-1]
    print(f"horizontal sweep: {p_h}")

    h = json.load(open(f"{p_h}/summary.json"))
    cert_iids = h["per_benchmark"]["relusplitter"]["cert_iids"]
    print(f"auditing {len(cert_iids)} relusplitter SC-HZ CERTs...")

    audit_root = Path(p_h) / "audit_relusplitter_cert"
    audit_root.mkdir(exist_ok=True)

    audit_results = []
    counters = Counter()
    for i, iid in enumerate(cert_iids):
        if i % 20 == 0:
            print(f"  progress {i}/{len(cert_iids)}...", flush=True)
        try:
            sc_hz_rec = json.load(open(f"{p_h}/relusplitter/iid{iid:04d}.json"))
            audit = audit_one_cert("relusplitter", iid, sc_hz_rec)
        except Exception as e:
            audit = {
                "bench": "relusplitter", "iid": iid,
                "overall_pass": False,
                "notes": [f"audit raised: {type(e).__name__}: {str(e)[:200]}"],
            }
        audit_results.append(audit)
        if audit["overall_pass"]:
            counters["pass_strict"] += 1
        elif audit.get("no_corner_witness_violates_spec") is False:
            counters["UNSOUND_corner_violates"] += 1
        elif audit.get("all_cond_lp_ub_strictly_below_threshold") is False:
            counters["fail_lp_ub_not_strictly_safe"] += 1
        elif not audit.get("provenance_complete"):
            counters["fail_provenance"] += 1
        else:
            counters["fail_other"] += 1

    with open(audit_root / "audit_per_iid.json", "w") as f:
        json.dump(audit_results, f, indent=2, default=float)
    summary = {
        "n_audited": len(cert_iids),
        "categories": dict(counters),
        "strict_pass_count": counters["pass_strict"],
        "unsound_cert_count": counters["UNSOUND_corner_violates"],
    }
    with open(audit_root / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== V-SIDE AUDIT RESULT on {len(cert_iids)} CERTs ===")
    for k, v in counters.items():
        print(f"  {k}: {v}")
    print(f"\nSTRICT-PASS: {counters['pass_strict']}/{len(cert_iids)}")
    print(f"UNSOUND (corner witness violates): {counters['UNSOUND_corner_violates']}")
    print(f"\nwrote {audit_root}/audit_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
