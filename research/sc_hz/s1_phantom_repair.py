"""S1: safenlp PHANTOM structured deterministic multi-candidate repair.

Per advisor 2026-06-05 directive: take each PHANTOM_LP_SAT iid from the
safenlp forward-coeff sweep, and try a DETERMINISTIC set of LP-derived
witness candidates (NOT random, NOT PGD). Each candidate is one
structured repair of the base box-corner; strict ORT replay confirms or
rejects.

Candidate menu per unsafe condition (~21-32 candidates total):
  1.  base:                  x* = c + r * sign(alpha_input)
  2.  reverse_sign:           x* = c - r * sign(alpha_input)
  3.  top-K single flips:     x* with x*[top_k_i] flipped (K=8)
  4.  top-K single centers:   x* with x*[top_k_i] set to c_in[i]
  5.  top-K pair flips:       x* with x*[pair] flipped (top-4 → C(4,2)=6)
  6.  top-K zero-then-flip:   center top-3 + flip next-3 (3 candidates)

Each candidate must:
  - lie strictly inside [lb, ub] (no clip required)
  - ORT replay returns d·y >= threshold at strict tolerance

Receipts written per iid with `verdict ∈ {A_CONFIRMED, PHANTOM_LP_SAT}`.

G10 enforced: pre-flight available RAM check + per-process RLIMIT_AS.
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
    """G10: refuse to start if available RAM < 90 GB."""
    try:
        out = subprocess.check_output(["free", "-g"], text=True).strip().split("\n")
        # Parse "Mem: total used free shared buff/cache available"
        cols = out[1].split()
        available = int(cols[6])
        if available < 90:
            print(f"REFUSE: G10 violation — only {available} GB available, need >=90 GB")
            sys.exit(1)
        print(f"G10 pre-flight OK: {available} GB available")
    except Exception as e:
        print(f"WARNING: G10 check failed: {e}")
    # Self-cap process address space at 100 GB
    try:
        resource.setrlimit(resource.RLIMIT_AS,
                            (100 * 1024**3, resource.RLIM_INFINITY))
    except Exception as e:
        print(f"WARNING: RLIMIT_AS set failed: {e}")


def generate_candidates(
    state_out: Any, d_out: np.ndarray,
    c_in: np.ndarray, r_in: np.ndarray, lb_x: np.ndarray, ub_x: np.ndarray,
    K_FLIP: int = 8, K_PAIR: int = 4, K_ZERO_FLIP: int = 3,
) -> List[Tuple[str, np.ndarray]]:
    """Generate structured deterministic witness candidates.

    Returns list of (label, x_star_unclipped) tuples. Each candidate
    must be in-box to be a valid witness; the caller checks that.
    """
    n_in = c_in.shape[0]
    origin = state_out.metadata.get("input_coord_origin", None)
    G_kept = state_out.G_kept
    K_kept = G_kept.shape[1]
    if origin is None or K_kept == 0:
        # Fallback: assume first n_in cols are input-coord
        origin = np.concatenate([
            np.arange(min(n_in, K_kept), dtype=np.int64),
            -np.ones(max(0, K_kept - n_in), dtype=np.int64),
        ])

    # alpha[i] = projected coefficient onto input coord i
    alpha = np.zeros(n_in, dtype=np.float64)
    alpha_kept = d_out @ G_kept
    for k in range(K_kept):
        coord = int(origin[k])
        if 0 <= coord < n_in:
            alpha[coord] += alpha_kept[k]

    # Base sign vector
    sign_alpha = np.sign(alpha)
    sign_alpha[sign_alpha == 0] = 1.0  # tie-break to +1
    base = c_in + r_in * sign_alpha

    candidates: List[Tuple[str, np.ndarray]] = []
    candidates.append(("base", base.copy()))
    candidates.append(("reverse_sign", c_in - r_in * sign_alpha))

    # Top-k by |alpha|
    abs_alpha = np.abs(alpha)
    top_indices = np.argsort(-abs_alpha)
    top_flip = top_indices[:K_FLIP]
    top_pair = top_indices[:K_PAIR]
    top_zero = top_indices[:K_ZERO_FLIP * 2]  # 2K for zero-then-flip

    # 3. Single flips at top-K_FLIP
    for j in top_flip:
        x = base.copy()
        x[j] = c_in[j] - r_in[j] * sign_alpha[j]
        candidates.append((f"flip_{j}", x))

    # 4. Single centers at top-K_FLIP
    for j in top_flip:
        x = base.copy()
        x[j] = c_in[j]
        candidates.append((f"center_{j}", x))

    # 5. Pair flips on top-K_PAIR
    for ii in range(K_PAIR):
        for jj in range(ii + 1, K_PAIR):
            i, j = top_pair[ii], top_pair[jj]
            x = base.copy()
            x[i] = c_in[i] - r_in[i] * sign_alpha[i]
            x[j] = c_in[j] - r_in[j] * sign_alpha[j]
            candidates.append((f"pair_{i}_{j}", x))

    # 6. Zero top-3 then flip next-3
    if len(top_zero) >= K_ZERO_FLIP * 2:
        zero_set = top_zero[:K_ZERO_FLIP]
        flip_set = top_zero[K_ZERO_FLIP:K_ZERO_FLIP * 2]
        for k in range(K_ZERO_FLIP):
            x = base.copy()
            for z in zero_set:
                x[z] = c_in[z]
            x[flip_set[k]] = c_in[flip_set[k]] - r_in[flip_set[k]] * sign_alpha[flip_set[k]]
            candidates.append((f"zero_top3_flip_{flip_set[k]}", x))

    return candidates


def repair_one_iid(bench: str, iid: int, out_dir: Path,
                    K_per_layer: int = 100000) -> Dict[str, Any]:
    """Run structured repair on one PHANTOM iid; return record."""
    from research.canonical_provenance import load_instance, build_provenance
    from research.sc_hz.onnx_walker import parse_onnx_to_layers
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.forward_witness import (
        initial_state_with_lineage, forward_propagate_no_backward,
    )
    from research.sc_hz.ops import lp_ub_rival_margin
    from research.sc_hz.ort_replay import ort_replay_one

    t0 = time.perf_counter()
    rec: Dict[str, Any] = {
        "bench": bench, "iid": iid,
        "method": "s1_structured_phantom_repair",
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
        layers, input_shape, n_classes = parse_onnx_to_layers(str(onnx_p))
        n_in = int(np.prod(input_shape))
        lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_p), n_in, n_classes)
        c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2

        init = initial_state_with_lineage(c_in, r_in)
        state_out, _ = forward_propagate_no_backward(
            init, layers, K_per_layer=K_per_layer, initial_shape=input_shape,
        )

        any_a_witness: Dict[str, Any] = None
        per_cond_summary: List[Dict[str, Any]] = []
        for j, (d_out, threshold, label) in enumerate(unsafe):
            ub = lp_ub_rival_margin(state_out, d_out)
            if ub < float(threshold):
                continue  # this rival is CERT (cannot witness)
            cands = generate_candidates(state_out, d_out, c_in, r_in, lb_x, ub_x)
            n_tried = 0; n_inbox = 0; n_holds = 0
            for cand_label, x_star in cands:
                n_tried += 1
                in_box = bool(
                    np.all(x_star >= lb_x - 1e-12)
                    and np.all(x_star <= ub_x + 1e-12)
                )
                if not in_box:
                    continue
                n_inbox += 1
                try:
                    y = ort_replay_one(str(onnx_p), x_star, input_shape)
                    d_y = float(d_out @ y)
                    if d_y > float(threshold):
                        n_holds += 1
                        any_a_witness = {
                            "cond_label": label,
                            "cand_label": cand_label,
                            "d_dot_y": d_y,
                            "threshold": float(threshold),
                            "margin": d_y - float(threshold),
                        }
                        break
                except Exception as e:
                    continue
            per_cond_summary.append({
                "cond_label": label, "lp_ub": float(ub),
                "n_tried": n_tried, "n_inbox": n_inbox, "n_holds": n_holds,
            })
            if any_a_witness is not None:
                break

        if any_a_witness is not None:
            rec["verdict"] = "A_CONFIRMED"
            rec["witness"] = any_a_witness
        else:
            rec["verdict"] = "PHANTOM_LP_SAT"
        rec["per_cond_summary"] = per_cond_summary
        rec["wall_total_s"] = time.perf_counter() - t0
    except Exception as e:
        rec["verdict"] = "UNK"
        rec["fail_closed_reason"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["wall_total_s"] = time.perf_counter() - t0

    p = out_dir / bench / f"iid{iid:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(rec, f, indent=2, default=float)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--bench", type=str, default="safenlp_2024")
    ap.add_argument("--phantom-list", type=str, required=True,
                      help="path to file with comma-separated iids")
    ap.add_argument("--n-pilot", type=int, default=100,
                      help="how many PHANTOM iids to attempt (deterministic prefix)")
    args = ap.parse_args()

    _g10_pre_flight()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    iids = [int(x) for x in open(args.phantom_list).read().split(",") if x.strip()]
    iids = iids[:args.n_pilot]
    print(f"S1 PHANTOM repair: bench={args.bench}, n={len(iids)} iids", flush=True)

    t0 = time.perf_counter()
    results = []
    for i, iid in enumerate(iids):
        rec = repair_one_iid(args.bench, iid, out_root)
        results.append(rec)
        v = rec.get("verdict")
        wall = rec.get("wall_total_s", 0)
        marker = " *** NEW A ***" if v == "A_CONFIRMED" else ""
        print(f"  [{i+1}/{len(iids)}] iid {iid:>4}: {v} ({wall:.1f}s){marker}",
              flush=True)
        # Intermediate save
        if (i + 1) % 20 == 0:
            cc = Counter(r.get("verdict") for r in results)
            with open(out_root / "intermediate.json", "w") as f:
                json.dump({"i_done": i+1, "counts": dict(cc)}, f, indent=2)

    wall = time.perf_counter() - t0
    cc = Counter(r.get("verdict") for r in results)
    n_a = cc.get("A_CONFIRMED", 0)
    a_iids = sorted(r["iid"] for r in results if r.get("verdict") == "A_CONFIRMED")
    print(f"\n=== S1 RESULT ===")
    print(f"n_processed: {len(results)}")
    print(f"verdict counts: {dict(cc)}")
    print(f"NEW A (vs PHANTOM_LP_SAT baseline): {n_a}")
    print(f"A iids: {a_iids[:20]}{'...' if len(a_iids) > 20 else ''}")
    print(f"wall: {wall:.1f}s")

    summary = {
        "stamp": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "bench": args.bench, "n_pilot": len(iids),
        "verdict_counts": dict(cc),
        "n_new_a": n_a, "a_iids": a_iids,
        "wall_seconds": wall,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {out_root}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
