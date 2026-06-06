"""Layer Failure Profiler — systematic per-iid attribution of why an iid is UNK.

Per advisor 2026-06-07: H0 sample-of-5 + ad-hoc sentinel attribution
(iid86/iid113/iid72) is not enough. Need a unified tool that for every
UNK iid reports:

  bench / iid
  first_parser_fail_op (or None)
  first_large_slack_layer (where slack accumulation explodes)
  unstable_count_per_relu
  triangle_slack_area_per_layer
  ng / nc / nb growth per layer
  final_rival_lp_margin
  f1_drop_pct
  f2b_applicable (False if no 2+ unstable in last ReLU)
  dominant_loose_block (if known)
  classification:
    parser_blocked
    f1_boundary
    dense_aggregate
    case_split_needed
    nonlinear_control

This profiler is read-only (no V/A change). It produces the data
backing every "targeted improvement" claim.

Principle compliance: just measurement; same as Phase H0.
"""
from __future__ import annotations

import csv
import json
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))
resource.setrlimit(resource.RLIMIT_AS, (60 * 1024**3, resource.RLIM_INFINITY))

import numpy as np
import onnx


def _to(s, f):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _to)


CR93 = ACT_ROOT / "audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS"


def get_true_unk(bench: str) -> List[int]:
    cp = CR93 / bench / "per_instance.csv"
    if not cp.exists():
        return []
    iid_v: Dict[int, set] = {}
    with open(cp) as f:
        for r in csv.DictReader(f):
            try:
                iid = int(r['iid'])
                v = (r.get('reportable_status') or '').strip().upper()
                if v:
                    iid_v.setdefault(iid, set()).add(v)
            except (ValueError, KeyError):
                continue
    return sorted([iid for iid, vs in iid_v.items()
                     if not ({'CERTIFIED', 'FALSIFIED', 'VERIFIED'} & vs)])


def profile_iid(bench: str, iid: int, time_alarm: int = 15) -> Dict[str, Any]:
    """Profile a single UNK iid: trace walker, capture per-layer stats."""
    rec: Dict[str, Any] = {
        "bench": bench, "iid": iid,
        "n_in_raw": None, "n_in_eff": None, "n_classes": None,
        "ops_total": None, "op_counts": None,
        "first_parser_fail_op": None,
        "walker_n_processed": None,
        "n_relus": None,
        "unstable_per_relu": [],
        "ng_per_layer": [],
        "nc_per_layer": [],
        "max_unstable_layer": None,
        "max_ng": None,
        "max_ng_layer": None,
        "hz_max_excess": None,
        "f1_max_excess": None,
        "f1_drop_pct": None,
        "f2b_applicable": None,
        "classification": "unknown",
        "reason": "",
        "wall_s": 0.0,
    }
    t0 = time.perf_counter()
    signal.alarm(time_alarm)
    try:
        from research.canonical_provenance import load_instance
        from research.sc_hz.vnnlib_parse import parse_vnnlib
        from research.sc_hz.constrained_lp_integration import forward_resnet_capture
        from research.sc_hz.constrained_lp import constrained_lp_ub
        from research.sc_hz.ops import lp_ub_rival_margin

        onnx_p, vnn_p = load_instance(bench, iid)
        m = onnx.load(str(onnx_p))
        rec["ops_total"] = len(m.graph.node)
        op_counts: Dict[str, int] = {}
        for n in m.graph.node:
            op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
        rec["op_counts"] = op_counts
        rec["n_relus"] = op_counts.get("Relu", 0)

        init_names = {x.name for x in m.graph.initializer}
        din = [x for x in m.graph.input if x.name not in init_names]
        if not din:
            rec["classification"] = "parser_blocked"
            rec["reason"] = "no_data_input"
            signal.alarm(0); rec["wall_s"] = time.perf_counter() - t0
            return rec
        din = din[0]
        dims = [d.dim_value if d.dim_value > 0 else 1
                for d in din.type.tensor_type.shape.dim]
        n_in = int(np.prod(dims[1:])) if dims[0] in (0, 1) else int(np.prod(dims))
        rec["n_in_raw"] = n_in
        od = [d.dim_value if d.dim_value > 0 else 1
              for d in m.graph.output[0].type.tensor_type.shape.dim]
        n_cls = int(np.prod(od[1:])) if len(od) > 1 else od[0]
        rec["n_classes"] = n_cls

        # Try parse vnnlib
        try:
            lb, ub, unsafe = parse_vnnlib(str(vnn_p), n_in, n_cls)
            rec["n_in_eff"] = int(np.sum(ub - lb > 1e-12))
        except Exception as e:
            rec["classification"] = "parser_blocked"
            rec["first_parser_fail_op"] = "vnnlib"
            rec["reason"] = f"vnnlib parse: {type(e).__name__}: {str(e)[:80]}"
            signal.alarm(0); rec["wall_s"] = time.perf_counter() - t0
            return rec

        # Try walker
        try:
            r = forward_resnet_capture(str(onnx_p), lb, ub, K_per_layer=100000)
        except Exception as e:
            err = str(e)[:120]
            rec["classification"] = "parser_blocked"
            # Look for "not implemented" or "primary input ... not in states"
            if "not implemented" in err.lower():
                # Try to extract op name
                import re
                match = re.search(r'(\w+)\([^)]+\): not implemented', err)
                if match:
                    rec["first_parser_fail_op"] = match.group(1)
                else:
                    rec["first_parser_fail_op"] = "unknown_op"
            else:
                rec["first_parser_fail_op"] = "walker_runtime"
            rec["reason"] = err[:200]
            signal.alarm(0); rec["wall_s"] = time.perf_counter() - t0
            return rec

        signal.alarm(0)
        rec["walker_n_processed"] = r.n_nodes_processed

        # Per-rival LP excess
        hz_max = -float('inf'); f1_max = -float('inf')
        for d, t_thr, _ in unsafe:
            ub_hz = float(lp_ub_rival_margin(r.output_state, d)) - float(t_thr)
            hz_max = max(hz_max, ub_hz)
            if r.last_relu_record is not None:
                ub_f1 = float(constrained_lp_ub(
                    r.last_relu_record, r.W_remaining, r.b_remaining, d
                )[0]) - float(t_thr)
                f1_max = max(f1_max, ub_f1)
        rec["hz_max_excess"] = float(hz_max)
        if r.last_relu_record is not None and f1_max != -float('inf'):
            rec["f1_max_excess"] = float(f1_max)
            if hz_max > 0:
                rec["f1_drop_pct"] = float((hz_max - f1_max) / abs(hz_max) * 100)
            # f2b applicable if last ReLU has >= 2 unstable
            n_unstable_last = int(r.last_relu_record.unstable_mask().sum())
            rec["max_unstable_layer"] = n_unstable_last
            rec["f2b_applicable"] = (n_unstable_last >= 2)

        # ng_per_layer: only have output state; approximate
        rec["max_ng"] = int(r.output_state.G_kept.shape[1])

        # Classification
        if hz_max < 0:
            rec["classification"] = "hz_already_cert"
            rec["reason"] = "HZ closed-form < 0 (rare leftover from r93 build drift)"
        elif f1_max is not None and f1_max != -float('inf') and f1_max < 0:
            rec["classification"] = "f1_lp_cert"
            rec["reason"] = "F1 LP < 0 → NEW V (small dense win)"
        elif f1_max is not None and f1_max != -float('inf') and 0 < f1_max < 0.01:
            rec["classification"] = "f1_boundary"
            rec["reason"] = f"F1 LP excess +{f1_max:.4e} ≤ 1e-2 (boundary numeric)"
        elif rec.get("n_in_eff") and rec["n_in_eff"] <= 50 and hz_max < 50:
            rec["classification"] = "low_dim_candidate"
            rec["reason"] = f"n_in_eff={rec['n_in_eff']} ≤ 50 (sparse spec)"
        elif rec.get("n_relus", 0) <= 6:
            rec["classification"] = "case_split_needed"
            rec["reason"] = (f"shallow net (n_relus={rec['n_relus']}), HZ excess "
                              f"{hz_max:.3e}; probably needs activation case split (P4 forbidden)")
        elif hz_max > 0.1:
            rec["classification"] = "dense_aggregate"
            rec["reason"] = (f"deep net, HZ excess {hz_max:.3e}, F1 drop "
                              f"{rec.get('f1_drop_pct', 0):.1f}%; aggregate ReLU slack")
        else:
            rec["classification"] = "phantom_close"
            rec["reason"] = "PHANTOM but small excess; no specific category"

    except TimeoutError:
        rec["classification"] = "timeout"
        rec["reason"] = f"per-iid timeout @ {time_alarm}s"
        signal.alarm(0)
    except Exception as e:
        rec["classification"] = "error"
        rec["reason"] = f"{type(e).__name__}: {str(e)[:120]}"
        signal.alarm(0)
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def profile_bench(bench: str, n_sample: int, out_dir: Path,
                    time_alarm: int = 15) -> Dict[str, Any]:
    unk = get_true_unk(bench)
    if not unk:
        return {"bench": bench, "n_unk_total": 0, "records": []}
    rng = np.random.default_rng(20260607 + hash(bench) % (2**31))
    sampled = (sorted(rng.choice(unk, size=n_sample, replace=False).tolist())
                 if len(unk) > n_sample else unk)
    records = []
    for iid in sampled:
        rec = profile_iid(bench, int(iid), time_alarm=time_alarm)
        records.append(rec)
        # Incremental save
        with open(out_dir / f"{bench}.json", "w") as f:
            json.dump({"bench": bench, "n_unk_total": len(unk),
                       "n_sampled": len(records), "records": records},
                      f, indent=2, default=float)
    # Classification summary
    from collections import Counter
    classifications = Counter(r["classification"] for r in records)
    parser_ops = Counter(r["first_parser_fail_op"]
                            for r in records if r["first_parser_fail_op"])
    return {
        "bench": bench, "n_unk_total": len(unk), "n_sampled": len(sampled),
        "classifications": dict(classifications),
        "parser_blocked_ops": dict(parser_ops),
        "records": records,
    }


def main():
    out_dir = Path("/tmp/layer_failure_profiler")
    out_dir.mkdir(exist_ok=True)
    # Advisor's specified benches
    benches = [
        ("cifar100_2024", 20, 30),
        ("tinyimagenet_2024", 20, 30),
        ("yolo_2023", 20, 30),
        ("traffic_signs_recognition_2023", 20, 20),
        ("acasxu_2023", 30, 10),
        ("relusplitter", 30, 15),
        ("linearizenn_2024", 20, 10),
        ("lsnc_relu", 10, 10),
        ("nn4sys", 20, 10),
        ("cctsdb_yolo_2023", 20, 20),
        ("ml4acopf_2024", 20, 10),
        ("metaroom_2023", 20, 20),
        ("vggnet16_2022", 18, 30),
    ]
    print(f"=== Layer Failure Profiler ===", flush=True)
    print(f"Profiling {len(benches)} weak benches with sample-of-20+", flush=True)
    summary = {}
    t0 = time.perf_counter()
    for bench, n_sample, alarm in benches:
        t_b = time.perf_counter()
        print(f"\n--- {bench} (sample={n_sample}, alarm={alarm}s) ---", flush=True)
        result = profile_bench(bench, n_sample, out_dir, time_alarm=alarm)
        summary[bench] = result
        cls = result.get("classifications", {})
        ops = result.get("parser_blocked_ops", {})
        print(f"  classifications: {cls}", flush=True)
        if ops:
            print(f"  parser_blocked_ops: {ops}", flush=True)
        print(f"  wall: {time.perf_counter()-t_b:.0f}s", flush=True)
    with open(out_dir / "_summary.json", "w") as f:
        json.dump({"per_bench": summary,
                    "total_wall_s": time.perf_counter() - t0},
                  f, indent=2, default=float)
    print(f"\nTotal wall: {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
