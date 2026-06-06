"""Phase D pilot: forward-coeff + fixed prune on dense-conv ResNet benchmarks.

Per advisor 2026-06-05 directive: prove SC-HZ + forward-coeff sidecar
works on dense-conv ResNet (cifar100 / tinyimagenet). Bounded pilot of
20-40 sentinel iids per benchmark; NOT a full sweep. Goal: verify
mechanism viability and measure LP margin tightness.

For each iid: forward HZ via ResNet walker, closed-form LP UB on each
unsafe condition, decode forward-coeff x_star and ORT replay. Verdict
follows the standard A_CONFIRMED / PHANTOM_LP_SAT / CERT trichotomy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def _process_one(args: Tuple[str, int, int, int, str]) -> Dict[str, Any]:
    bench, iid, K, wall_s, out_dir = args
    from research.canonical_provenance import load_instance, build_provenance
    from research.sc_hz.onnx_walker_resnet import forward_resnet
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.forward_witness import decode_xi_star_forward
    from research.sc_hz.ops import lp_ub_rival_margin
    from research.sc_hz.ort_replay import ort_replay_one

    t0 = time.perf_counter()
    rec: Dict[str, Any] = {"bench": bench, "iid": iid,
                              "method": "resnet_forward_coeff"}
    try:
        prov = build_provenance(bench, iid)
        rec.update({
            "canonical_root": str(prov.canonical_root),
            "instances_csv_sha256": prov.instances_csv_sha256,
            "onnx_sha256": prov.onnx_sha256,
            "vnnlib_sha256": prov.vnnlib_sha256,
        })
        onnx_p, vnn_p = load_instance(bench, iid)
        # Best-effort n_in / n_classes derivation
        if bench.startswith("cifar100"):
            n_in, n_classes = 3072, 100
        elif bench.startswith("tinyimagenet"):
            n_in, n_classes = 3*56*56, 200
        else:
            # Fall back via onnx
            import onnx
            m = onnx.load(str(onnx_p))
            dims = [d.dim_value if d.dim_value > 0 else 1
                     for d in m.graph.input[0].type.tensor_type.shape.dim]
            n_in = int(np.prod(dims[1:])) if dims[0] in (0, 1) else int(np.prod(dims))
            odims = [d.dim_value if d.dim_value > 0 else 1
                      for d in m.graph.output[0].type.tensor_type.shape.dim]
            n_classes = int(np.prod(odims[1:])) if len(odims) > 1 else odims[0]
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_p), n_in, n_classes)
        c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2

        result = forward_resnet(str(onnx_p), lb_x, ub_x, K_per_layer=K)
        rec["forward_wall_s"] = time.perf_counter() - t0
        rec["n_processed"] = result.n_nodes_processed
        rec["n_skipped"] = len(result.nodes_skipped)
        rec["output_ng"] = int(result.output_state.G_kept.shape[1])
        rec["n_unsafe_conditions"] = len(unsafe)

        any_fal = False; any_a = False
        max_excess = -np.inf
        cond_results = []
        for d_out, threshold, label in unsafe:
            if time.perf_counter() - t0 > wall_s:
                rec["timeout"] = True
                break
            ub = lp_ub_rival_margin(result.output_state, d_out)
            excess = float(ub) - float(threshold)
            if excess > max_excess:
                max_excess = excess
            cr = {"label": label, "lp_ub": float(ub),
                  "threshold": float(threshold), "excess": float(excess)}
            if ub >= float(threshold):
                any_fal = True
                x_star, _ = decode_xi_star_forward(
                    result.output_state, d_out, c_in, r_in,
                )
                in_box = bool(np.all(x_star >= lb_x - 1e-12) and
                                np.all(x_star <= ub_x + 1e-12))
                x_star_c = np.clip(x_star, lb_x, ub_x)
                try:
                    y = ort_replay_one(str(onnx_p), x_star_c, result.input_shape)
                    d_y = float(d_out @ y)
                    cr["d_dot_y"] = d_y
                    cr["cond_holds_strict"] = d_y > float(threshold)
                    cr["x_star_in_box_no_clip"] = in_box
                    if d_y > float(threshold) and in_box:
                        any_a = True
                        cond_results.append(cr)
                        break  # one A is enough
                except Exception as e:
                    cr["ort_error"] = f"{type(e).__name__}: {str(e)[:100]}"
            cond_results.append(cr)

        if any_a:
            verdict = "A_CONFIRMED"
        elif any_fal:
            verdict = "PHANTOM_LP_SAT"
        else:
            verdict = "CERT"
        rec["verdict"] = verdict
        rec["max_excess"] = float(max_excess)
        rec["cond_results"] = cond_results
        rec["wall_total_s"] = time.perf_counter() - t0
    except Exception as e:
        rec["verdict"] = "UNK"
        rec["fail_closed_reason"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["wall_total_s"] = time.perf_counter() - t0

    p = Path(out_dir) / bench / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(rec, f, indent=2, default=float)
    return {"bench": bench, "iid": iid,
            "verdict": rec.get("verdict"),
            "max_excess": rec.get("max_excess"),
            "wall_total_s": rec.get("wall_total_s")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=100000)
    ap.add_argument("--wall-per-iid-s", type=int, default=300)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--targets", type=str, required=True,
                      help="bench:iids; comma-sep iids. E.g. "
                           "cifar100_2024:0,10,20;tinyimagenet_2024:0,10")
    args = ap.parse_args()

    work: List[Tuple[str, int, int, int, str]] = []
    for chunk in args.targets.split(";"):
        bench, iids = chunk.split(":")
        for iid in iids.split(","):
            iid = iid.strip()
            if iid:
                work.append((bench, int(iid), args.K, args.wall_per_iid_s, args.out))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Phase D pilot: {len(work)} iids, K={args.K}, wall={args.wall_per_iid_s}s, "
          f"workers={args.n_workers}")

    t0 = time.perf_counter()
    results = []
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_process_one, work, chunksize=1)):
            results.append(rec)
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(work)}] elapsed={elapsed:.0f}s — "
                  f"{rec['bench']} iid {rec['iid']}: {rec['verdict']} "
                  f"(wall={rec.get('wall_total_s', 0):.0f}s)", flush=True)

    wall = time.perf_counter() - t0
    by_bench = {}
    for b in set(w[0] for w in work):
        sub = [r for r in results if r["bench"] == b]
        cc = Counter(r["verdict"] for r in sub)
        a_iids = sorted(r["iid"] for r in sub if r["verdict"] == "A_CONFIRMED")
        cert_iids = sorted(r["iid"] for r in sub if r["verdict"] == "CERT")
        by_bench[b] = {"n": len(sub), "counts": dict(cc),
                         "a_iids": a_iids, "cert_iids": cert_iids}
        print(f"\n{b}: n={len(sub)}, {dict(cc)}, A={len(a_iids)}, CERT={len(cert_iids)}")
    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K": args.K, "wall_per_iid_s": args.wall_per_iid_s,
        "n_workers": args.n_workers,
        "wall_seconds": wall,
        "by_bench": by_bench,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
