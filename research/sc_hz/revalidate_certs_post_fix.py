"""Re-validate all prior SC-HZ CERT verdicts under the bug-fixed prune.

The 2026-06-04 soundness bug: prune() did not preserve incoming
state.tail_radius, causing LP UB to UNDER-approximate the true reach.
This invalidates any CERT verdict that relied on LP UB < threshold
strictly via the buggy path.

This script re-runs the forward propagation with the fixed prune (now
sums incoming tail + new dropped-cols tail) on every prior CERT iid:
  - 153 safenlp_2024 CERTs (Phase B)
  - 71 relusplitter CERTs (horizontal extension)
  - 48 malbeware CERTs (horizontal extension)

For each, reports the NEW verdict under the fixed prune (CERT / PHANTOM /
A_CONFIRMED via ORT) at K=256.
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
    from research.canonical_provenance import load_instance
    from research.sc_hz.onnx_walker import parse_onnx_to_layers, forward_propagate
    from research.sc_hz.precompute_direction import precompute_d_per_layer_chain
    from research.sc_hz.prune import PrunedState
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.run_sentinels import _layer_output_shapes
    from research.sc_hz.ops import lp_ub_rival_margin
    from research.sc_hz.ort_replay import (
        decode_xi_star_for_condition, ort_replay_one,
    )

    t0 = time.perf_counter()
    out = {"bench": bench, "iid": iid, "K": K}
    try:
        onnx_path, vnn_path = load_instance(bench, iid)
        layers, input_shape, n_classes = parse_onnx_to_layers(str(onnx_path))
        n_in = int(np.prod(input_shape))
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_path), n_in, n_classes)
        c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2
        out_shapes = _layer_output_shapes(layers, input_shape)
        G_0 = np.diag(r_in)
        init = PrunedState(c=c_in.copy(), G_kept=G_0,
                            tail_radius=None, metadata={})
        any_fal_candidate = False
        any_a_confirmed = False
        max_excess = -np.inf
        for d_out, threshold, label in unsafe:
            d_chain = precompute_d_per_layer_chain(layers, d_out, out_shapes)
            state, _ = forward_propagate(init, layers, d_chain,
                                            K_per_layer=K,
                                            initial_shape=input_shape)
            ub = lp_ub_rival_margin(state, d_out)
            excess = float(ub) - float(threshold)
            if excess > max_excess:
                max_excess = excess
            if ub >= float(threshold):
                any_fal_candidate = True
                # ORT replay via OLD decoder (back-compat audit)
                x_star, _ = decode_xi_star_for_condition(
                    {}, d_out, c_in, r_in, d_chain[0],
                )
                x_star = np.clip(x_star, lb_x, ub_x)
                try:
                    y = ort_replay_one(str(onnx_path), x_star, input_shape)
                    if float(d_out @ y) > float(threshold):
                        any_a_confirmed = True
                        break
                except Exception:
                    pass

        if any_a_confirmed:
            verdict = "A_CONFIRMED"
        elif any_fal_candidate:
            verdict = "PHANTOM_LP_SAT"
        else:
            verdict = "CERT"
        out.update({"verdict": verdict, "max_excess": float(max_excess),
                     "wall_s": time.perf_counter() - t0})
    except Exception as e:
        out.update({"verdict": "ERROR",
                     "error": f"{type(e).__name__}: {str(e)[:200]}",
                     "wall_s": time.perf_counter() - t0})
    p = Path(out_dir) / bench / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--n-workers", type=int, default=16)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect CERT iid sets from prior runs
    targets: List[Tuple[str, int]] = []
    # 153 safenlp CERTs from Phase B
    p_b1 = sorted(glob.glob(
        "/data1/Kane/ACT/audit_results/sc_hz_phase_b_safenlp_*/"
    ))[-1]
    safe_cert = json.load(open(f"{p_b1}/summary.json"))["cert_iids"]
    for iid in safe_cert:
        targets.append(("safenlp_2024", iid))
    # 71 relusplitter + 48 malbeware CERTs from horizontal
    p_h = sorted(glob.glob(
        "/data1/Kane/ACT/audit_results/sc_hz_horizontal_*/"
    ))[-1]
    h_sum = json.load(open(f"{p_h}/summary.json"))
    for iid in h_sum["per_benchmark"]["relusplitter"]["cert_iids"]:
        targets.append(("relusplitter", iid))
    for iid in h_sum["per_benchmark"]["malbeware"]["cert_iids"]:
        targets.append(("malbeware", iid))

    print(f"Re-validating {len(targets)} prior CERT iids (153 safenlp + "
          f"71 relusplitter + 48 malbeware) under FIXED prune at K={args.K}")

    work = [(b, i, args.K, str(out_root)) for (b, i) in targets]
    t0 = time.perf_counter()
    results = []
    with mp.get_context("spawn").Pool(processes=args.n_workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_process_one, work,
                                                       chunksize=2)):
            results.append(rec)
            if (i + 1) % 50 == 0:
                cert = sum(1 for r in results if r["verdict"] == "CERT")
                a = sum(1 for r in results if r["verdict"] == "A_CONFIRMED")
                print(f"  {i+1}/{len(work)} — CERT={cert}, A={a}", flush=True)

    wall = time.perf_counter() - t0
    by_bench = {}
    for bench in ["safenlp_2024", "relusplitter", "malbeware"]:
        sub = [r for r in results if r["bench"] == bench]
        cc = Counter(r["verdict"] for r in sub)
        cert_surv = sorted(r["iid"] for r in sub if r["verdict"] == "CERT")
        a_new = sorted(r["iid"] for r in sub if r["verdict"] == "A_CONFIRMED")
        by_bench[bench] = {
            "n_prior_cert": len(sub),
            "verdict_counts": dict(cc),
            "n_cert_surviving": len(cert_surv),
            "n_lost_to_phantom": cc.get("PHANTOM_LP_SAT", 0),
            "n_lost_to_a": cc.get("A_CONFIRMED", 0),
            "cert_surviving_iids": cert_surv,
            "a_new_iids": a_new,
        }
        print(f"\n{bench}: prior_CERT={len(sub)}, FIXED-CERT={len(cert_surv)}, "
              f"PHANTOM={cc.get('PHANTOM_LP_SAT', 0)}, A={len(a_new)}")

    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K": args.K, "wall_seconds": wall,
        "n_workers": args.n_workers,
        "by_bench": by_bench,
        "total_cert_surviving": sum(b["n_cert_surviving"] for b in by_bench.values()),
        "total_lost_to_unk": sum(b["n_lost_to_phantom"] for b in by_bench.values()),
        "total_lost_to_a": sum(b["n_lost_to_a"] for b in by_bench.values()),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n=== POST-FIX CERT SURVIVAL ===")
    print(f"safenlp_2024: {by_bench['safenlp_2024']['n_cert_surviving']}/153")
    print(f"relusplitter: {by_bench['relusplitter']['n_cert_surviving']}/71")
    print(f"malbeware:    {by_bench['malbeware']['n_cert_surviving']}/48")
    print(f"TOTAL:        {summary['total_cert_surviving']}/272")
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
