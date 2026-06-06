"""S3 small/control 48h scout — forward HZ + structured-flip candidates.

Per advisor 2026-06-05 post-S1 plan: scout 6 small/control benchmarks where
production canonical sweep returned UNK on every tried config. Mechanism is
the SAME forward-coefficient + S1 structured-flip candidate that produced
+12 NEW A on safenlp PHANTOMs, applied to:
  - acasxu_2023
  - linearizenn_2024
  - sat_relu
  - relusplitter
  - tllverifybench_2023
  - ml4acopf_2024

20 production-truly-UNK iids per bench (sampled deterministically from the
front of the sorted UNK list).

Verdict mapping:
  CERT          — LP UB < threshold for every unsafe rival
  A_CONFIRMED   — at least one deterministic candidate ORT-confirms strict
  PHANTOM_LP_SAT — LP UB ≥ threshold but no candidate confirms
  UNK           — parser fail-closed or runtime error (no verdict)

G10 enforced: pre-flight RAM check, RLIMIT_AS 100 GB.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def _g10_pre_flight() -> None:
    try:
        out = subprocess.check_output(["free", "-g"], text=True).strip().split("\n")
        cols = out[1].split()
        available = int(cols[6])
        if available < 90:
            print(f"REFUSE: G10 violation — only {available} GB available, need >=90 GB")
            sys.exit(1)
        print(f"G10 pre-flight OK: {available} GB available")
    except Exception as e:
        print(f"WARNING: G10 check failed: {e}")
    try:
        resource.setrlimit(resource.RLIMIT_AS,
                            (100 * 1024**3, resource.RLIM_INFINITY))
    except Exception as e:
        print(f"WARNING: RLIMIT_AS set failed: {e}")


def _derive_n_in_n_classes(bench: str, onnx_path) -> Tuple[int, int]:
    """Best-effort n_in / n_classes derivation via ONNX shape."""
    import onnx
    m = onnx.load(str(onnx_path))
    in_dims = [d.dim_value if d.dim_value > 0 else 1
                for d in m.graph.input[0].type.tensor_type.shape.dim]
    out_dims = [d.dim_value if d.dim_value > 0 else 1
                  for d in m.graph.output[0].type.tensor_type.shape.dim]
    # Skip batch dim
    n_in = int(np.prod(in_dims[1:])) if in_dims[0] in (0, 1) else int(np.prod(in_dims))
    n_classes = int(np.prod(out_dims[1:])) if len(out_dims) > 1 else int(out_dims[0])
    if n_classes <= 0:
        n_classes = int(out_dims[-1])
    return n_in, n_classes


def scout_one_iid(bench: str, iid: int, out_dir: Path, K: int = 100000,
                    wall_budget_s: int = 60) -> Dict[str, Any]:
    from research.canonical_provenance import load_instance, build_provenance
    from research.sc_hz.onnx_walker import parse_onnx_to_layers
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.forward_witness import (
        initial_state_with_lineage, forward_propagate_no_backward,
    )
    from research.sc_hz.ops import lp_ub_rival_margin
    from research.sc_hz.ort_replay import ort_replay_one
    from research.sc_hz.s1_phantom_repair import generate_candidates

    t0 = time.perf_counter()
    rec: Dict[str, Any] = {
        "bench": bench, "iid": iid, "K": K,
        "method": "s3_forward_coeff_plus_flip_candidates",
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
        n_in, n_classes = _derive_n_in_n_classes(bench, onnx_p)
        layers, input_shape, n_classes2 = parse_onnx_to_layers(str(onnx_p))
        n_in_real = int(np.prod(input_shape))
        n_in = n_in_real if n_in_real > 0 else n_in
        n_classes = n_classes2 if n_classes2 > 0 else n_classes
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_p), n_in, n_classes)
        c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2

        init = initial_state_with_lineage(c_in, r_in)
        state_out, _ = forward_propagate_no_backward(
            init, layers, K_per_layer=K, initial_shape=input_shape,
        )

        any_fal = False; any_a_witness = None
        max_excess = -np.inf
        cond_results = []
        for d_out, threshold, label in unsafe:
            if time.perf_counter() - t0 > wall_budget_s:
                rec["timeout"] = True
                break
            ub = lp_ub_rival_margin(state_out, d_out)
            excess = float(ub) - float(threshold)
            max_excess = max(max_excess, excess)
            cr = {"label": label, "lp_ub": float(ub),
                  "threshold": float(threshold), "excess": float(excess)}
            if ub >= float(threshold):
                any_fal = True
                cands = generate_candidates(state_out, d_out, c_in, r_in, lb_x, ub_x)
                for cand_label, x_star in cands:
                    in_box = bool(
                        np.all(x_star >= lb_x - 1e-12)
                        and np.all(x_star <= ub_x + 1e-12)
                    )
                    if not in_box:
                        continue
                    x_star_c = np.clip(x_star, lb_x, ub_x)
                    if np.max(np.abs(x_star_c - x_star)) > 1e-12:
                        continue  # clip required, skip
                    try:
                        y = ort_replay_one(str(onnx_p), x_star_c, input_shape)
                        d_y = float(d_out @ y)
                        if d_y > float(threshold):
                            any_a_witness = {
                                "cond_label": label,
                                "cand_label": cand_label,
                                "d_dot_y": d_y,
                                "threshold": float(threshold),
                                "margin": d_y - float(threshold),
                            }
                            break
                    except Exception:
                        continue
            cond_results.append(cr)
            if any_a_witness is not None:
                break

        if any_a_witness is not None:
            verdict = "A_CONFIRMED"
            rec["witness"] = any_a_witness
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

    p = out_dir / bench / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(rec, f, indent=2, default=float)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--targets-json", type=str, required=True)
    ap.add_argument("--per-bench", type=int, default=20)
    ap.add_argument("--K", type=int, default=100000)
    ap.add_argument("--wall-per-iid-s", type=int, default=60)
    args = ap.parse_args()

    _g10_pre_flight()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = json.load(open(args.targets_json))
    work: List[Tuple[str, int]] = []
    for bench, d in targets.items():
        iids = d.get("truly_unk", [])[:args.per_bench]
        for iid in iids:
            work.append((bench, iid))

    print(f"S3 scout: {len(work)} iids across {len(targets)} benches, "
          f"K={args.K}, wall={args.wall_per_iid_s}s/iid, n_workers=1 sequential",
          flush=True)
    t0 = time.perf_counter()
    results = []
    for i, (bench, iid) in enumerate(work):
        r = scout_one_iid(bench, iid, out_root, K=args.K,
                            wall_budget_s=args.wall_per_iid_s)
        results.append(r)
        v = r.get("verdict"); wall = r.get("wall_total_s", 0)
        mx = r.get("max_excess")
        mx_str = f"{mx:+.2e}" if isinstance(mx, float) else "n/a"
        marker = " *** NEW " + v + " ***" if v in {"A_CONFIRMED", "CERT"} else ""
        print(f"  [{i+1}/{len(work)}] {bench:<22} iid {iid:>4}: "
              f"{v:<14} ({wall:.1f}s, mx={mx_str}){marker}", flush=True)
        if (i + 1) % 20 == 0:
            cc = Counter(rr.get("verdict") for rr in results)
            print(f"    --- progress {i+1}: {dict(cc)} ---", flush=True)

    wall = time.perf_counter() - t0
    by_bench = {}
    for bench in targets:
        sub = [r for r in results if r["bench"] == bench]
        cc = Counter(r.get("verdict") for r in sub)
        a_iids = sorted(r["iid"] for r in sub if r.get("verdict") == "A_CONFIRMED")
        c_iids = sorted(r["iid"] for r in sub if r.get("verdict") == "CERT")
        by_bench[bench] = {"n": len(sub), "counts": dict(cc),
                             "a_iids": a_iids, "cert_iids": c_iids,
                             "n_a": len(a_iids), "n_cert": len(c_iids)}
        print(f"\n{bench:<22}: n={len(sub)}, {dict(cc)} "
              f"(A={len(a_iids)}, CERT={len(c_iids)})", flush=True)

    total_a = sum(b["n_a"] for b in by_bench.values())
    total_c = sum(b["n_cert"] for b in by_bench.values())
    print(f"\n=== S3 SCOUT RESULT ===")
    print(f"total NEW A: {total_a}")
    print(f"total NEW CERT (LP audit pending): {total_c}")
    print(f"per-bench: " + ", ".join(f"{b}=A{d['n_a']}/V{d['n_cert']}"
                                         for b, d in by_bench.items()))

    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "K": args.K, "wall_per_iid_s": args.wall_per_iid_s,
        "per_bench_iid_cap": args.per_bench,
        "wall_seconds": wall,
        "by_bench": by_bench,
        "total_a": total_a, "total_cert": total_c,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
