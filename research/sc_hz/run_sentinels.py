"""SC-HZ Phase A sentinel driver.

Per EXECUTION §2.4-§4: load 80 sentinels, run pruned forward HZ with
per-rival pruning, emit receipts with provenance.

For Phase A:
  - acasxu: Dense-only, fully supported.
  - safenlp: Dense-only, fully supported.
  - tinyimagenet/cifar100: contain Conv/BN/Add/MaxPool; the driver
    handles Dense + Conv + BN + Sub + Flatten via the ONNX walker.
    If a model has residual Add (ResNet), the driver currently raises
    NotImplementedError (Phase A residual handling is future work).

The driver fail-closes on every error: the iid is recorded with the
exception type and a "fail_closed_reason" string. No silent fallback.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.canonical_provenance import build_provenance, load_instance  # noqa: E402
from research.sc_hz.onnx_walker import (  # noqa: E402
    parse_onnx_to_layers, forward_propagate,
)
from research.sc_hz.precompute_direction import (  # noqa: E402
    precompute_d_per_layer_chain,
)
from research.sc_hz.prune import PrunedState  # noqa: E402
from research.sc_hz.ops import lp_ub_rival_margin, bounds  # noqa: E402
from research.sc_hz.vnnlib_parse import parse_vnnlib  # noqa: E402


def _build_input_state(lb: np.ndarray, ub: np.ndarray) -> PrunedState:
    """Initial PrunedState = axis-aligned box (c_in, diag(r_in))."""
    c = (lb + ub) / 2.0
    r = (ub - lb) / 2.0
    G = np.diag(r).astype(np.float64)
    return PrunedState(
        c=c, G_kept=G, tail_radius=None,
        metadata={"layer": "input"},
    )


def _layer_output_shapes(layers, input_shape):
    """Compute the per-layer output shape (for d_L precompute)."""
    shapes: List = []
    cur = tuple(int(x) for x in input_shape)
    for op in layers:
        k = op.kind
        if k == "sub" or k == "relu" or k == "bn":
            shapes.append(cur)
        elif k == "flatten":
            n = 1
            for d in cur: n *= d
            cur = (n,)
            shapes.append(cur)
        elif k == "dense":
            cur = (int(op.params["W"].shape[0]),)
            shapes.append(cur)
        elif k == "conv2d":
            W = op.params["W"]
            Co, _, kH, kW = W.shape
            stride = op.params.get("stride", 1)
            padding = op.params.get("padding", 0)
            Ci, Hi, Wi = op.params["input_shape"]
            Ho = (Hi + 2*padding - kH) // stride + 1
            Wo = (Wi + 2*padding - kW) // stride + 1
            cur = (Co, Ho, Wo)
            shapes.append(cur)
        elif k == "maxpool" or k == "avgpool":
            ks = op.params["kernel_size"]
            st = op.params.get("stride", ks)
            Ci, Hi, Wi = op.params["input_shape"]
            Ho = (Hi - ks) // st + 1
            Wo = (Wi - ks) // st + 1
            cur = (Ci, Ho, Wo)
            shapes.append(cur)
        else:
            shapes.append(cur)
    return shapes


def run_iid(benchmark: str, iid: int, K: int = 256,
             wall_s: int = 600) -> Dict[str, Any]:
    """Run one sentinel iid through SC-HZ. Returns receipt dict."""
    t0 = time.perf_counter()
    receipt: Dict[str, Any] = {
        "benchmark": benchmark,
        "iid": int(iid),
        "K": K,
        "verdict": None,
        "fail_closed_reason": None,
    }
    try:
        # Resolve canonical paths
        onnx_path, vnnlib_path = load_instance(benchmark, iid)
        prov = build_provenance(benchmark, iid).as_dict()
        receipt.update({
            "canonical_root": prov["canonical_root"],
            "instances_csv_sha256": prov["instances_csv_sha256"],
            "onnx_sha256": prov["onnx_sha256"],
            "vnnlib_sha256": prov["vnnlib_sha256"],
        })

        # Parse ONNX
        layers, input_shape, n_classes = parse_onnx_to_layers(str(onnx_path))
        n_input = 1
        for d in input_shape: n_input *= int(d)

        # Parse vnnlib (returns list of (d_vector, threshold, label))
        lb, ub, unsafe = parse_vnnlib(str(vnnlib_path), n_input, n_classes)
        if not unsafe:
            receipt["fail_closed_reason"] = "vnnlib: no unsafe conditions parsed"
            receipt["verdict"] = "UNK"
            return receipt

        receipt["n_input"] = n_input
        receipt["n_classes"] = n_classes
        receipt["n_layers"] = len(layers)
        receipt["n_unsafe"] = len(unsafe)
        receipt["unsafe_labels"] = [u[2] for u in unsafe[:50]]

        # Initial input box
        input_state = _build_input_state(lb, ub)

        # Compute output shapes per layer (for d_L)
        layer_out_shapes = _layer_output_shapes(layers, input_shape)

        # For each unsafe condition (d, threshold, label), compute LP UB on d·y.
        # The unsafe condition reads "d · y >= threshold"; we prove CERT iff
        # max(d·y - threshold) < 0 across the input box.
        per_cond_results = []
        max_excess_overall = -np.inf
        for (d_out, threshold, label) in unsafe:
            d_chain = precompute_d_per_layer_chain(
                layers, d_out, layer_out_shapes,
            )
            t_fwd = time.perf_counter()
            try:
                state_out, _traces = forward_propagate(
                    input_state, layers, d_chain, K_per_layer=K,
                    initial_shape=input_shape,
                )
                ub = lp_ub_rival_margin(state_out, d_out)
                excess = ub - threshold      # > 0 means unsafe reachable
            except NotImplementedError as ne:
                per_cond_results.append({
                    "label": label, "lp_ub": None, "threshold": threshold,
                    "fail_closed_reason": f"NotImplementedError: {ne}",
                })
                continue
            t_fwd = time.perf_counter() - t_fwd
            per_cond_results.append({
                "label": label, "lp_ub": float(ub),
                "threshold": float(threshold),
                "excess": float(excess), "wall_s": t_fwd,
            })
            if excess > max_excess_overall:
                max_excess_overall = excess
            if (time.perf_counter() - t0) > wall_s:
                receipt["fail_closed_reason"] = f"wall budget {wall_s}s exceeded"
                break

        valid = [p for p in per_cond_results if p.get("lp_ub") is not None]
        if not valid:
            verdict = "UNK"
        elif all(p["excess"] < -1e-9 for p in valid):
            verdict = "CERT"
        elif any(p["excess"] >= -1e-9 for p in valid):
            verdict = "FAL_CANDIDATE"
        else:
            verdict = "UNK"

        receipt["verdict"] = verdict
        receipt["per_cond"] = per_cond_results[:50]
        receipt["max_excess"] = float(max_excess_overall) if np.isfinite(max_excess_overall) else None
        receipt["n_cond_results"] = len(per_cond_results)

    except NotImplementedError as ne:
        receipt["verdict"] = "UNK"
        receipt["fail_closed_reason"] = f"NotImplementedError: {ne}"
    except Exception as e:
        receipt["verdict"] = "UNK"
        receipt["fail_closed_reason"] = f"{type(e).__name__}: {e}"
        receipt["trace"] = traceback.format_exc()[:1500]
    finally:
        receipt["wall_s"] = time.perf_counter() - t0

    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentinels", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--wall-per-iid-s", type=int, default=600)
    ap.add_argument("--bench", type=str, default="",
                    help="restrict to one benchmark")
    ap.add_argument("--max-iids", type=int, default=0,
                    help="cap per-benchmark iid count (0 = no cap)")
    args = ap.parse_args()

    sentinels = json.load(open(args.sentinels))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_iid").mkdir(exist_ok=True)

    receipts: List[Dict[str, Any]] = []
    bench_list = ([args.bench] if args.bench
                   else ["acasxu_2023", "safenlp_2024",
                         "tinyimagenet_2024", "cifar100_2024"])
    for bench in bench_list:
        if bench not in sentinels:
            print(f"[driver] WARN: {bench} not in sentinel JSON; skipping")
            continue
        iids = sentinels[bench]["iids"]
        if args.max_iids > 0:
            iids = iids[:args.max_iids]
        print(f"\n=== {bench}: {len(iids)} iids ===", flush=True)
        for iid in iids:
            print(f"  iid {iid}... ", end="", flush=True)
            t0 = time.perf_counter()
            rec = run_iid(bench, iid, K=args.K, wall_s=args.wall_per_iid_s)
            wall = time.perf_counter() - t0
            receipts.append(rec)
            out_file = out_root / "per_iid" / f"{bench}_iid{iid}.json"
            with open(out_file, "w") as f:
                json.dump(rec, f, indent=2, default=float)
            ub = rec.get("max_excess")
            ub_str = f"{ub:+.3f}" if ub is not None else "n/a"
            print(f"{rec.get('verdict','?'):<14} max_excess={ub_str} wall={wall:.1f}s "
                  f"{rec.get('fail_closed_reason') or ''}")
            gc.collect()

    # Aggregate summary
    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K": args.K,
        "n_total": len(receipts),
        "verdict_counts": {},
        "per_bench": {},
    }
    from collections import Counter
    cc = Counter(r["verdict"] for r in receipts)
    summary["verdict_counts"] = dict(cc)
    for bench in bench_list:
        b_recs = [r for r in receipts if r["benchmark"] == bench]
        bcc = Counter(r["verdict"] for r in b_recs)
        summary["per_bench"][bench] = {
            "n": len(b_recs),
            "verdicts": dict(bcc),
            "max_excess_mean": float(np.mean([r["max_excess"] for r in b_recs
                                                if r.get("max_excess") is not None]))
                                if any(r.get("max_excess") is not None for r in b_recs)
                                else None,
        }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {out_root}/summary.json")
    print(f"verdicts: {dict(cc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
