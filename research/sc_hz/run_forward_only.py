"""SC-HZ runner using FORWARD-ONLY witness extractor (no W^T backward chain).

Per advisor 2026-06-04 review §1: gate-test the forward-coefficient extractor
on safenlp 100 sample + relusplitter 71 CERT, then full sweep if A_CONFIRMED
and CERT counts are preserved (within ~1%).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def _process_one(args: Tuple[str, int, int, str]) -> Dict[str, Any]:
    bench, iid, K, out_dir = args
    from research.canonical_provenance import load_instance, build_provenance
    from research.sc_hz.onnx_walker import parse_onnx_to_layers
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.forward_witness import (
        initial_state_with_lineage,
        forward_propagate_no_backward,
        decode_xi_star_forward,
    )
    from research.sc_hz.ops import lp_ub_rival_margin
    from research.sc_hz.ort_replay import ort_replay_one, check_unsafe_condition

    t0 = time.perf_counter()
    record: Dict[str, Any] = {
        "bench": bench, "iid": iid,
        "method": "forward_coefficient_no_backward_chain",
    }
    try:
        onnx_path, vnn_path = load_instance(bench, iid)
        prov = build_provenance(bench, iid)
        record.update({
            "canonical_root": str(prov.canonical_root),
            "instances_csv_sha256": prov.instances_csv_sha256,
            "onnx_sha256": prov.onnx_sha256,
            "vnnlib_sha256": prov.vnnlib_sha256,
        })
        layers, input_shape, n_classes = parse_onnx_to_layers(str(onnx_path))
        n_in = 1
        for d in input_shape:
            n_in *= int(d)
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_path), n_in, n_classes)
        c_in = (lb_x + ub_x) / 2.0
        r_in = (ub_x - lb_x) / 2.0

        init_state = initial_state_with_lineage(c_in, r_in)
        # Forward propagate with no backward d chain
        state_out, traces = forward_propagate_no_backward(
            init_state, layers, K_per_layer=K, initial_shape=input_shape,
        )

        # Per-condition LP UB + witness decode
        cond_results = []
        any_fal_candidate = False
        any_a_confirmed = False
        max_excess = -np.inf
        for (d_out, threshold, label) in unsafe:
            ub = lp_ub_rival_margin(state_out, d_out)
            excess = ub - float(threshold)
            if excess > max_excess:
                max_excess = excess
            cond_rec: Dict[str, Any] = {
                "label": label, "lp_ub": float(ub),
                "threshold": float(threshold), "excess": float(excess),
            }
            if ub >= float(threshold):
                any_fal_candidate = True
                x_star, meta = decode_xi_star_forward(
                    state_out, d_out, c_in, r_in,
                )
                cond_rec.update(meta)
                # ORT replay
                try:
                    y = ort_replay_one(str(onnx_path), x_star, input_shape)
                    d_dot_y = float(d_out @ y)
                    cond_holds = d_dot_y >= float(threshold)
                    cond_rec["d_dot_y_at_x_star"] = d_dot_y
                    cond_rec["cond_holds_on_x_star"] = cond_holds
                    if cond_holds:
                        any_a_confirmed = True
                        cond_rec["witness_iv"] = {
                            "x_star_first_5": x_star[:5].tolist(),
                            "n_input_dim": int(n_in),
                        }
                except Exception as e:
                    cond_rec["ort_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            cond_results.append(cond_rec)

        # Verdict
        if any_a_confirmed:
            verdict = "A_CONFIRMED"
        elif any_fal_candidate:
            verdict = "PHANTOM_LP_SAT"
        else:
            verdict = "CERT"

        record.update({
            "verdict": verdict,
            "max_excess": float(max_excess),
            "cond_results": cond_results,
            "wall_s": time.perf_counter() - t0,
        })
    except Exception as e:
        record.update({
            "verdict": "UNK",
            "fail_closed_reason": f"{type(e).__name__}: {str(e)[:200]}",
            "wall_s": time.perf_counter() - t0,
        })

    p = Path(out_dir) / bench / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(record, f, indent=2, default=float)
    return {
        "bench": bench, "iid": iid,
        "verdict": record.get("verdict"),
        "max_excess": record.get("max_excess"),
        "wall_s": record.get("wall_s"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--bench", type=str, required=True,
                      help="benchmark name (e.g. safenlp_2024, relusplitter)")
    ap.add_argument("--iids", type=str, required=True,
                      help="comma-separated iid list, or 'all:N' for range(N)")
    args = ap.parse_args()

    if args.iids.startswith("all:"):
        iids = list(range(int(args.iids.split(":")[1])))
    else:
        iids = [int(x) for x in args.iids.split(",") if x.strip()]

    work = [(args.bench, iid, args.K, args.out) for iid in iids]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Forward-only SC-HZ: {len(work)} iids on {args.bench}, "
          f"K={args.K}, workers={args.n_workers}")
    t0 = time.perf_counter()

    results = []
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_process_one, work,
                                                       chunksize=2)):
            results.append(rec)
            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                a = sum(1 for r in results if r["verdict"] == "A_CONFIRMED")
                cert = sum(1 for r in results if r["verdict"] == "CERT")
                print(f"  {i+1}/{len(work)} ({(i+1)/elapsed:.1f}/s) "
                      f"— A={a}, CERT={cert}", flush=True)

    wall = time.perf_counter() - t0
    counts = Counter(r["verdict"] for r in results)
    a_iids = sorted(r["iid"] for r in results if r["verdict"] == "A_CONFIRMED")
    cert_iids = sorted(r["iid"] for r in results if r["verdict"] == "CERT")

    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "method": "forward_coefficient_no_backward_chain",
        "bench": args.bench,
        "K": args.K, "n_workers": args.n_workers,
        "n_total": len(results),
        "wall_seconds": wall,
        "verdict_counts": dict(counts),
        "n_a_confirmed": len(a_iids),
        "n_cert": len(cert_iids),
        "a_iids": a_iids,
        "cert_iids": cert_iids,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== FORWARD-ONLY RESULT ({args.bench}) ===")
    print(f"verdicts: {dict(counts)}")
    print(f"A_CONFIRMED: {len(a_iids)}, CERT: {len(cert_iids)}")
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
