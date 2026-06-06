"""Goal 2: Dense-conv ResNet Phase D 40-sentinel pilot.

User-specified iids:
  cifar100_2024:      0,2,6,8,24,29,57,72,86,113,118,130,145,156,168,180,185,190,195,199
  tinyimagenet_2024:  0,6,12,24,30,45,60,73,86,99,116,130,145,158,170,180,190,195,198,199

n_workers=1 mandatory (each iid peak RSS ~70 GB on 125 GB system).
K=∞ (no prune; headline anchored at K=∞ per G7).

Per-iid output: verdict / max_excess / output_ng / peak_mem / cond_results
with LP UB + ORT margin + clip flag per unsafe condition.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run_one_iid(bench: str, iid: int, K: int, wall_s: int,
                  out_dir: Path) -> Dict[str, Any]:
    from research.canonical_provenance import load_instance, build_provenance
    from research.sc_hz.onnx_walker_resnet import forward_resnet
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.forward_witness import decode_xi_star_forward
    from research.sc_hz.ops import lp_ub_rival_margin
    from research.sc_hz.ort_replay import ort_replay_one

    rss_baseline = _peak_rss_gb()
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {
        "bench": bench, "iid": iid, "K": K,
        "method": "resnet_forward_coeff_K_inf",
        "rss_baseline_gb": rss_baseline,
    }
    try:
        prov = build_provenance(bench, iid)
        rec.update({
            "canonical_root": str(prov.canonical_root),
            "instances_csv_sha256": prov.instances_csv_sha256,
            "onnx_sha256": prov.onnx_sha256,
            "vnnlib_sha256": prov.vnnlib_sha256,
        })
        onnx_p, vnn_p = load_instance(bench, iid)
        if bench.startswith("cifar100"):
            n_in, n_classes = 3072, 100
        else:
            n_in, n_classes = 3*56*56, 200
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_p), n_in, n_classes)
        c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2

        t_fwd = time.perf_counter()
        result = forward_resnet(str(onnx_p), lb_x, ub_x, K_per_layer=K)
        rec["forward_wall_s"] = time.perf_counter() - t_fwd
        rec["peak_rss_gb"] = _peak_rss_gb()
        rec["n_processed"] = result.n_nodes_processed
        rec["n_skipped"] = len(result.nodes_skipped)
        rec["output_ng"] = int(result.output_state.G_kept.shape[1])
        rec["initial_ng"] = n_in
        rec["ng_blowup_ratio"] = rec["output_ng"] / max(1, n_in)
        rec["n_unsafe_conditions"] = len(unsafe)

        any_fal = False; any_a = False
        max_excess = -np.inf
        cond_results: List[Dict[str, Any]] = []
        for j, (d_out, threshold, label) in enumerate(unsafe):
            elapsed = time.perf_counter() - t0
            if elapsed > wall_s:
                rec["timeout_at_cond"] = j
                break
            ub = lp_ub_rival_margin(result.output_state, d_out)
            excess = float(ub) - float(threshold)
            max_excess = max(max_excess, excess)
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
                    cr["ort_margin_minus_threshold"] = d_y - float(threshold)
                    cr["cond_holds_strict"] = d_y > float(threshold)
                    cr["x_star_in_box_no_clip"] = in_box
                    if d_y > float(threshold) and in_box:
                        any_a = True
                        cond_results.append(cr)
                        break
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
        rec["fail_closed_reason"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["wall_total_s"] = time.perf_counter() - t0
        rec["peak_rss_gb"] = _peak_rss_gb()

    p = out_dir / bench / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(rec, f, indent=2, default=float)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=100000)
    ap.add_argument("--wall-per-iid-s", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cifar_iids = [0,2,6,8,24,29,57,72,86,113,118,130,145,156,168,180,185,190,195,199]
    tiny_iids  = [0,6,12,24,30,45,60,73,86,99,116,130,145,158,170,180,190,195,198,199]
    work = [("cifar100_2024", i) for i in cifar_iids] + \
            [("tinyimagenet_2024", i) for i in tiny_iids]

    print(f"Goal 2 pilot: {len(work)} iids, K={args.K}, wall={args.wall_per_iid_s}s, "
          f"n_workers=1 SEQUENTIAL", flush=True)
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for i, (bench, iid) in enumerate(work):
        print(f"\n[{i+1}/{len(work)}] {bench} iid {iid} (elapsed={time.perf_counter()-t0:.0f}s)...",
              flush=True)
        r = run_one_iid(bench, iid, args.K, args.wall_per_iid_s, out)
        results.append(r)
        v = r.get("verdict"); wall = r.get("wall_total_s", 0)
        ng = r.get("output_ng", 0); rss = r.get("peak_rss_gb", 0)
        mx = r.get("max_excess")
        mx_str = f"{mx:.2e}" if isinstance(mx, float) and not (mx is None) else "n/a"
        print(f"  --> {v}, wall={wall:.0f}s, output_ng={ng}, "
              f"peak_rss={rss:.1f}GB, max_excess={mx_str}", flush=True)
        cc = Counter(rr.get("verdict") for rr in results)
        with open(out / "summary_intermediate.json", "w") as f:
            json.dump({"i_done": i+1, "total": len(work),
                         "counts": dict(cc),
                         "wall_seconds_so_far": time.perf_counter() - t0},
                       f, indent=2, default=float)

    wall = time.perf_counter() - t0
    by_bench: Dict[str, Any] = {}
    for b in ["cifar100_2024", "tinyimagenet_2024"]:
        sub = [r for r in results if r["bench"] == b]
        cc = Counter(r.get("verdict") for r in sub)
        a_iids = sorted(r["iid"] for r in sub if r.get("verdict") == "A_CONFIRMED")
        c_iids = sorted(r["iid"] for r in sub if r.get("verdict") == "CERT")
        by_bench[b] = {
            "n": len(sub),
            "verdict_counts": dict(cc),
            "a_iids": a_iids, "cert_iids": c_iids,
            "n_a_confirmed": len(a_iids), "n_cert": len(c_iids),
        }
        print(f"\n{b}: n={len(sub)}, {dict(cc)}, A={len(a_iids)}, CERT={len(c_iids)}",
              flush=True)
    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K": args.K, "wall_per_iid_s": args.wall_per_iid_s,
        "wall_seconds": wall,
        "by_bench": by_bench,
        "cifar_iids": cifar_iids, "tiny_iids": tiny_iids,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {out}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
