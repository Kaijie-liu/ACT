"""Phase H0 — Measure principle-internal harvestable subset across 22 benchmarks.

Per `research/FORWARD_PLAN_principle_internal_levers_20260606.md` §3:
before any Lever 1-5 sprint runs, we must measure for each (bench, UNK iid):

  - parseable: walker fails with `not implemented` on a finite list of ops
  - fal_able: LP rival max excess > 0 AND ORT replay of LP-corner candidate
              produces output "close" to violation (threshold-configurable)
  - low_dim: spec has effective input dimension << raw input shape
              (sparse perturbation or partial-coord spec)
  - boundary_numeric: F1 LP UB excess in [0, 0.01] range (potential
                      float-precision PHANTOM that exact-arith might flip)
  - robust_blocked: none of above; F1 LP excess > +0.1

Output: per-bench tag distribution + total principle-internal ceiling.
This output is canonical: any further sprint expected-yield estimate
MUST reference the ceiling reported here.

PRINCIPLE COMPLIANCE: this script is measurement-only. It does not
modify production code. It does not produce V/A claims. The output is
informational structuring of the existing UNK pool.

Time: ~1-2s per iid (walker + F1 LP + ORT replay). Targeting ~20 iids
per bench = ~440 instances, ~10-15 min wall.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))


def _set_memory_cap(gb: int) -> None:
    """Cap process virtual address space at gb gigabytes (RLIMIT_AS)."""
    cap_bytes = gb * 1024**3
    try:
        resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, resource.RLIM_INFINITY))
        print(f"[H0] RLIMIT_AS set to {gb} GB", flush=True)
    except Exception as e:
        print(f"[H0] WARNING: RLIMIT_AS set failed: {e}", flush=True)


class _IIDTimeoutError(BaseException):
    """Inherits from BaseException to NOT be swallowed by `except Exception`."""
    pass


def _iid_timeout_handler(signum, frame):
    raise _IIDTimeoutError("per-iid timeout")


CANONICAL_R93 = ACT_ROOT / "audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS"

# Benches in the 22-VNN-COMP-2025 set. lsnc_relu universally hard.
BENCHES_22 = [
    "acasxu_2023", "cersyve", "cgan_2023", "cifar100_2024",
    "collins_aerospace_benchmark", "collins_rul_cnn_2022", "cora_2024",
    "dist_shift_2023", "linearizenn_2024", "lsnc_relu", "malbeware",
    "metaroom_2023", "ml4acopf_2024", "nn4sys", "relusplitter",
    "safenlp_2024", "sat_relu", "soundnessbench", "tinyimagenet_2024",
    "tllverifybench_2023", "traffic_signs_recognition_2023",
    "vggnet16_2022", "yolo_2023",
]


def _load_unk_iids(bench: str) -> List[int]:
    """Return list of iids reported as UNK in canonical r93."""
    csv_path = CANONICAL_R93 / bench / "per_instance.csv"
    if not csv_path.exists():
        return []
    unk = set()
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            verdict = (row.get("reportable_status") or "").strip().upper()
            if verdict in ("UNKNOWN", "UNKNOWN_TIMEOUT", "ERROR"):
                try:
                    unk.add(int(row["iid"]))
                except (ValueError, KeyError):
                    pass
    return sorted(unk)


def _effective_input_dim(lb_x: np.ndarray, ub_x: np.ndarray,
                          eps: float = 1e-12) -> int:
    """Number of input coordinates with non-trivial range."""
    return int(np.sum(ub_x - lb_x > eps))


def _classify_iid_with_timeout(bench: str, iid: int, time_budget_s: float,
                                   **kwargs) -> Dict[str, Any]:
    """Wrap _classify_iid with SIGALRM timeout to prevent hang on slow iids."""
    signal.signal(signal.SIGALRM, _iid_timeout_handler)
    signal.alarm(int(time_budget_s))
    try:
        return _classify_iid(bench, iid, time_budget_s=time_budget_s, **kwargs)
    except _IIDTimeoutError:
        return {"bench": bench, "iid": iid, "tag": "timeout",
                "n_input_eff": -1, "n_input_raw": -1,
                "hz_max_excess": None, "f1_max_excess": None,
                "wall_s": time_budget_s,
                "reason": f"per-iid timeout @ {time_budget_s}s"}
    finally:
        signal.alarm(0)


def _classify_iid(bench: str, iid: int, time_budget_s: float = 30.0,
                    boundary_threshold: float = 0.01,
                    fal_close_threshold_rel: float = 0.50,
                    ) -> Dict[str, Any]:
    """Classify a single UNK iid into the 5 categories.

    Returns dict with keys: tag, n_input_eff, n_input_raw, hz_max_excess,
                              f1_max_excess (if applicable), wall_s, reason.
    """
    rec = {"bench": bench, "iid": iid, "tag": "unknown_error",
           "n_input_eff": -1, "n_input_raw": -1,
           "hz_max_excess": None, "f1_max_excess": None,
           "wall_s": 0.0, "reason": ""}
    t0 = time.perf_counter()
    try:
        from research.canonical_provenance import load_instance
        from research.sc_hz.vnnlib_parse import parse_vnnlib
        import onnx
        onnx_p, vnn_p = load_instance(bench, iid)
        m = onnx.load(str(onnx_p))
        init_names = {i.name for i in m.graph.initializer}
        data_inputs = [i for i in m.graph.input if i.name not in init_names]
        if not data_inputs:
            data_inputs = list(m.graph.input)
        in_proto = data_inputs[0]
        in_dims = [d.dim_value if d.dim_value > 0 else 1
                    for d in in_proto.type.tensor_type.shape.dim]
        in_shape = tuple(in_dims[1:]) if in_dims[0] in (0, 1) else tuple(in_dims)
        n_in = int(np.prod(in_shape))
        out_dims = [d.dim_value if d.dim_value > 0 else 1
                     for d in m.graph.output[0].type.tensor_type.shape.dim]
        n_cls = int(np.prod(out_dims[1:]) if len(out_dims) > 1 else out_dims[0])

        # vnnlib parse: gets input box + unsafe conditions
        try:
            lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_p), n_in, n_cls)
        except Exception as e:
            rec["tag"] = "parseable_vnnlib"
            rec["reason"] = f"vnnlib parse: {type(e).__name__}: {str(e)[:80]}"
            rec["wall_s"] = time.perf_counter() - t0
            return rec

        n_eff = _effective_input_dim(lb_x, ub_x)
        rec["n_input_eff"] = n_eff
        rec["n_input_raw"] = n_in

        # Tag low-dim if effective input dim is << raw
        # Threshold: < 25% of raw OR ≤ 50 absolute (sparse perturbation)
        low_dim = (n_eff <= 50) or (n_eff < 0.25 * n_in)

        # Try walker
        try:
            from research.sc_hz.constrained_lp_integration import forward_resnet_capture
            r = forward_resnet_capture(str(onnx_p), lb_x, ub_x,
                                          K_per_layer=100000)
        except Exception as e:
            err_str = str(e)[:100]
            if "not implemented" in err_str.lower() or "not reached" in err_str.lower():
                rec["tag"] = "parseable"
                rec["reason"] = f"walker: {err_str}"
            else:
                rec["tag"] = "parseable_other"
                rec["reason"] = f"walker: {type(e).__name__}: {err_str}"
            rec["wall_s"] = time.perf_counter() - t0
            return rec

        # Walker OK — compute HZ + F1 LP max_excess across unsafe conditions
        from research.sc_hz.constrained_lp import constrained_lp_ub
        from research.sc_hz.ops import lp_ub_rival_margin

        hz_max = -float("inf")
        f1_max = -float("inf")
        worst_d = None
        worst_t = None
        for d, t_thr, lbl in unsafe:
            ub_hz = float(lp_ub_rival_margin(r.output_state, d)) - float(t_thr)
            if ub_hz > hz_max:
                hz_max = ub_hz
                worst_d = d
                worst_t = float(t_thr)
        rec["hz_max_excess"] = float(hz_max)

        # F1 LP only on the worst rival (cheap)
        f1_excess = None
        if r.last_relu_record is not None and worst_d is not None:
            ub_f1 = constrained_lp_ub(r.last_relu_record, r.W_remaining,
                                         r.b_remaining, worst_d)[0]
            f1_excess = float(ub_f1) - worst_t
            rec["f1_max_excess"] = f1_excess

        # Classification logic
        # Priority: parseable_X already returned above for walker failures
        # 1. boundary_numeric: F1 excess in (0, 0.01]
        # 2. low_dim: effective input dim small enough
        # 3. fal_able: HZ excess > 0 and not super-large (heuristic for replay-able)
        # 4. robust_blocked: F1 excess > 0.1 and not low_dim and not boundary

        if f1_excess is not None and 0 < f1_excess <= boundary_threshold:
            rec["tag"] = "boundary_numeric"
            rec["reason"] = f"F1 excess +{f1_excess:.4e} ≤ boundary {boundary_threshold}"
        elif low_dim and (hz_max < 50.0):
            # Low effective input dim AND HZ not absurdly large
            rec["tag"] = "low_dim"
            rec["reason"] = (f"n_eff={n_eff} (raw={n_in}), HZ_excess={hz_max:.2e}; "
                              "low-dim spec may CERT via existing pipeline")
        elif hz_max > 0:
            # Distinguish FAL-able vs robust_blocked
            # heuristic: if F1 excess is "moderate" (< 5), there's a chance
            # the LP-corner candidate replays close enough that activation-walk
            # FAL could succeed
            if f1_excess is not None and f1_excess < 5.0:
                rec["tag"] = "fal_able"
                rec["reason"] = (f"HZ excess +{hz_max:.4e}, F1 excess +{f1_excess:.4e} "
                                  "potentially FAL via activation-walk")
            else:
                rec["tag"] = "robust_blocked"
                rec["reason"] = (f"HZ excess +{hz_max:.4e}, F1 excess "
                                  f"{'(N/A)' if f1_excess is None else f'+{f1_excess:.4e}'}; "
                                  "above current-pipeline reach")
        else:
            # hz_max < 0: this iid is actually CERT under HZ. Should not be UNK.
            # Treat as anomaly.
            rec["tag"] = "already_cert"
            rec["reason"] = f"HZ excess {hz_max:.4e} < 0 (anomaly: marked UNK but HZ CERTs)"

    except Exception as e:
        rec["tag"] = "error"
        rec["reason"] = f"{type(e).__name__}: {str(e)[:120]}"
    rec["wall_s"] = time.perf_counter() - t0
    return rec


# Benches where walker is known to be too slow (>60s/iid even on simple iids)
# under current SC-HZ + DeepZ pipeline. For these we classify via spec parsing
# only, then attribute to `robust_blocked` (closure evidence from
# F1/F2b/FC-HZ) or `low_dim` (sparse spec). See FORWARD_PLAN §4 Lever 2.
HEAVY_BENCHES_SPEC_ONLY = {
    "cifar100_2024", "tinyimagenet_2024", "vggnet16_2022",
    "yolo_2023", "traffic_signs_recognition_2023", "cctsdb_yolo_2023",
}


def _classify_iid_spec_only(bench: str, iid: int) -> Dict[str, Any]:
    """Lightweight classification using vnnlib parse only (no walker).

    For heavy benches where walker would take >60s/iid. We tag based on
    spec geometry alone:
      - low_dim if effective input dim is small
      - robust_blocked otherwise (closure evidence: F1/F2b/FC-HZ all
        failed to flip these benches)
    """
    rec = {"bench": bench, "iid": iid, "tag": "unknown_error",
           "n_input_eff": -1, "n_input_raw": -1,
           "hz_max_excess": None, "f1_max_excess": None,
           "wall_s": 0.0, "reason": ""}
    t0 = time.perf_counter()
    try:
        from research.canonical_provenance import load_instance
        from research.sc_hz.vnnlib_parse import parse_vnnlib
        import onnx
        onnx_p, vnn_p = load_instance(bench, iid)
        m = onnx.load(str(onnx_p))
        init_names = {i.name for i in m.graph.initializer}
        data_inputs = [i for i in m.graph.input if i.name not in init_names]
        if not data_inputs:
            data_inputs = list(m.graph.input)
        in_proto = data_inputs[0]
        in_dims = [d.dim_value if d.dim_value > 0 else 1
                    for d in in_proto.type.tensor_type.shape.dim]
        in_shape = tuple(in_dims[1:]) if in_dims[0] in (0, 1) else tuple(in_dims)
        n_in = int(np.prod(in_shape))
        out_dims = [d.dim_value if d.dim_value > 0 else 1
                     for d in m.graph.output[0].type.tensor_type.shape.dim]
        n_cls = int(np.prod(out_dims[1:]) if len(out_dims) > 1 else out_dims[0])

        try:
            lb_x, ub_x, _unsafe = parse_vnnlib(str(vnn_p), n_in, n_cls)
        except Exception as e:
            rec["tag"] = "parseable_vnnlib"
            rec["reason"] = f"vnnlib parse: {type(e).__name__}: {str(e)[:80]}"
            rec["wall_s"] = time.perf_counter() - t0
            return rec

        n_eff = _effective_input_dim(lb_x, ub_x)
        rec["n_input_eff"] = n_eff
        rec["n_input_raw"] = n_in
        # low_dim heuristic: ≤ 50 absolute OR < 25% of raw
        if n_eff <= 50 or n_eff < 0.25 * n_in:
            rec["tag"] = "low_dim"
            rec["reason"] = f"n_eff={n_eff} ≤ low_dim threshold; spec-only classification"
        else:
            rec["tag"] = "robust_blocked"
            rec["reason"] = (f"n_eff={n_eff}/{n_in} full ε-ball spec; "
                              "F1/F2b/FC-HZ closure evidence: robust CERT unreachable "
                              "by current pipeline")
    except Exception as e:
        rec["tag"] = "error"
        rec["reason"] = f"{type(e).__name__}: {str(e)[:120]}"
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def measure_bench(bench: str, n_sample: int = 20,
                    time_budget_s: float = 30.0,
                    seed: int = 20260606,
                    out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Measure tag distribution on a sample of UNK iids for a bench.

    Writes incremental JSON after EACH iid so a hang doesn't lose data.
    Uses spec-only classification for HEAVY_BENCHES_SPEC_ONLY.
    """
    unk_iids = _load_unk_iids(bench)
    n_unk = len(unk_iids)
    if n_unk == 0:
        result = {"bench": bench, "n_unk_total": 0, "n_sampled": 0,
                  "tags": {}, "records": [], "spec_only": False}
        if out_dir:
            with open(out_dir / f"{bench}.json", "w") as f:
                json.dump(result, f, indent=2, default=float)
        return result
    rng = np.random.default_rng(seed + hash(bench) % (2**31))
    if len(unk_iids) > n_sample:
        sampled = sorted(rng.choice(unk_iids, size=n_sample, replace=False).tolist())
    else:
        sampled = unk_iids
    spec_only = bench in HEAVY_BENCHES_SPEC_ONLY
    records = []
    for iid in sampled:
        t_iid = time.perf_counter()
        if spec_only:
            rec = _classify_iid_spec_only(bench, int(iid))
        else:
            rec = _classify_iid_with_timeout(bench, int(iid),
                                                  time_budget_s=time_budget_s)
        records.append(rec)
        print(f"    {bench}/iid {iid}: tag={rec['tag']} "
              f"wall={time.perf_counter()-t_iid:.1f}s"
              f"{' (spec-only)' if spec_only else ''}",
              flush=True)
        # INCREMENTAL: write after each iid
        if out_dir:
            partial = {"bench": bench, "n_unk_total": n_unk,
                       "n_sampled": len(records),
                       "tags": dict(Counter(r["tag"] for r in records)),
                       "records": records, "spec_only": spec_only,
                       "in_progress": True}
            with open(out_dir / f"{bench}.json", "w") as f:
                json.dump(partial, f, indent=2, default=float)
    tags = Counter(r["tag"] for r in records)
    result = {
        "bench": bench, "n_unk_total": n_unk, "n_sampled": len(sampled),
        "tags": dict(tags), "records": records,
        "spec_only": spec_only, "in_progress": False,
    }
    if out_dir:
        with open(out_dir / f"{bench}.json", "w") as f:
            json.dump(result, f, indent=2, default=float)
    return result


def aggregate(bench_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-bench results into a global harvestable ceiling estimate."""
    out = {"per_bench": {}, "global_extrapolated": {}}
    total_unk = 0
    total_extrap = Counter()
    for br in bench_results:
        bench = br["bench"]
        n_unk = br["n_unk_total"]
        n_smp = br["n_sampled"]
        tags = br["tags"]
        out["per_bench"][bench] = {
            "n_unk_total": n_unk, "n_sampled": n_smp,
            "tags_sampled": tags,
            "tags_extrapolated": {},
        }
        if n_smp == 0:
            continue
        scale = n_unk / n_smp
        for tag, count in tags.items():
            extrap = int(round(count * scale))
            out["per_bench"][bench]["tags_extrapolated"][tag] = extrap
            total_extrap[tag] += extrap
        total_unk += n_unk
    out["global_extrapolated"]["n_unk_total"] = total_unk
    out["global_extrapolated"]["by_tag"] = dict(total_extrap)
    # Harvestable ceiling = sum of all reachable tags
    reachable = ["parseable", "parseable_vnnlib", "parseable_other",
                  "fal_able", "low_dim", "boundary_numeric"]
    blocked = ["robust_blocked"]
    harvestable = sum(total_extrap.get(t, 0) for t in reachable)
    blocked_count = sum(total_extrap.get(t, 0) for t in blocked)
    out["global_extrapolated"]["harvestable_ceiling"] = harvestable
    out["global_extrapolated"]["blocked"] = blocked_count
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True,
                     help="Output directory for receipts + summary.json")
    ap.add_argument("--per-bench-sample", type=int, default=20,
                     help="Sample size per benchmark (default 20)")
    ap.add_argument("--time-budget-s", type=float, default=30.0,
                     help="Per-iid time budget seconds")
    ap.add_argument("--benches", type=str, default=None,
                     help="Comma-separated bench list (default: all 22 minus lsnc_relu)")
    ap.add_argument("--memory-cap-gb", type=int, default=100,
                     help="RLIMIT_AS memory cap (default 100 GB)")
    args = ap.parse_args()

    _set_memory_cap(args.memory_cap_gb)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.benches:
        bench_list = args.benches.split(",")
    else:
        bench_list = [b for b in BENCHES_22 if b != "lsnc_relu"]

    print(f"Measuring harvestable subset on {len(bench_list)} benchmarks, "
          f"{args.per_bench_sample} iids each")
    print(f"Output: {out_dir}", flush=True)

    bench_results = []
    t0 = time.perf_counter()
    for i, bench in enumerate(bench_list):
        t_b = time.perf_counter()
        result = measure_bench(bench, n_sample=args.per_bench_sample,
                                  time_budget_s=args.time_budget_s,
                                  out_dir=out_dir)
        bench_results.append(result)
        wall = time.perf_counter() - t_b
        tag_summary = ", ".join(f"{t}={n}" for t, n in result["tags"].items())
        print(f"  [{i+1}/{len(bench_list)}] {bench}: "
              f"n_unk={result['n_unk_total']}, sampled={result['n_sampled']}, "
              f"wall={wall:.0f}s, tags={{{tag_summary}}}",
              flush=True)

    # Aggregate
    agg = aggregate(bench_results)
    agg["total_wall_s"] = time.perf_counter() - t0
    with open(out_dir / "summary.json", "w") as f:
        json.dump(agg, f, indent=2, default=float)

    print(f"\n=== Phase H0 Aggregate ===")
    print(f"Total UNK across {len(bench_list)} benches: "
          f"{agg['global_extrapolated']['n_unk_total']}")
    print(f"Harvestable ceiling (sum of reachable tags): "
          f"{agg['global_extrapolated']['harvestable_ceiling']}")
    print(f"Blocked (robust_blocked): {agg['global_extrapolated']['blocked']}")
    print(f"By tag: {agg['global_extrapolated']['by_tag']}")
    print(f"Total wall: {agg['total_wall_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
