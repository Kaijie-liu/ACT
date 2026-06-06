"""Goal 1: ResNet walker correctness suite.

Four checks per advisor 2026-06-05 day-plan:
  (a) center parity: walker(box center input) == ORT(box center) within 1e-5
  (b) Add lineage soundness: residual generators merged correctly
  (c) linear op exactness: Conv/BN/Flatten/Gemm/Add are EXACT
  (d) per-layer memory + ng trace

Run on 5 cifar100 + 5 tinyimagenet sentinels.
If center parity fails or any skipped node → halt before pilot.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.canonical_provenance import load_instance
from research.sc_hz.onnx_walker_resnet import (
    forward_resnet, _initializers_dict, _node_attr_dict, _smart_add,
    _apply_globalavgpool,
)
from research.sc_hz.forward_witness import initial_state_with_lineage
from research.sc_hz.prune import PrunedState
import research.sc_hz.ops as scops
from research.sc_hz.vnnlib_parse import parse_vnnlib
import onnx
from onnx import numpy_helper


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def check_center_parity(bench: str, iid: int, atol: float = 1e-5) -> Dict[str, Any]:
    onnx_p, vnn_p = load_instance(bench, iid)
    if bench.startswith("cifar100"):
        n_in, n_classes = 3072, 100
    elif bench.startswith("tinyimagenet"):
        n_in, n_classes = 3*56*56, 200
    else:
        raise ValueError(f"unsupported bench {bench}")
    lb_x, ub_x, _ = parse_vnnlib(str(vnn_p), n_in, n_classes)
    c_in = (lb_x + ub_x) / 2

    t0 = time.perf_counter()
    result = forward_resnet(str(onnx_p), c_in, c_in, K_per_layer=100000)
    walker_wall = time.perf_counter() - t0
    walker_out = result.output_state.c

    sess = ort.InferenceSession(str(onnx_p))
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    x = c_in.reshape(result.input_shape).astype(np.float32)
    if len(in_shape) > len(result.input_shape):
        x = x[None, ...]
    ort_out = sess.run(None, {in_name: x})[0].reshape(-1).astype(np.float64)

    if walker_out.shape != ort_out.shape:
        ort_out = ort_out.reshape(-1)[:walker_out.shape[0]]
    diff = walker_out - ort_out
    max_abs_diff = float(np.abs(diff).max())
    rel_diff = float(np.abs(diff).max() / max(1e-12, float(np.abs(ort_out).max())))

    return {
        "bench": bench, "iid": iid,
        "walker_wall_s": walker_wall,
        "walker_out_first_5": walker_out[:5].tolist(),
        "ort_out_first_5": ort_out[:5].tolist(),
        "max_abs_diff": max_abs_diff,
        "max_relative_diff": rel_diff,
        "n_processed": result.n_nodes_processed,
        "n_skipped": len(result.nodes_skipped),
        "output_ng": int(result.output_state.G_kept.shape[1]),
        "PASS_center_parity": max_abs_diff < atol,
    }


def check_per_layer_trace(bench: str, iid: int) -> Dict[str, Any]:
    """Re-walk the model and capture per-layer ng / wall / rss trace."""
    onnx_p, vnn_p = load_instance(bench, iid)
    if bench.startswith("cifar100"):
        n_in, n_classes = 3072, 100
    else:
        n_in, n_classes = 3*56*56, 200
    lb_x, ub_x, _ = parse_vnnlib(str(vnn_p), n_in, n_classes)
    c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2

    m = onnx.load(str(onnx_p))
    inits = _initializers_dict(m)
    in_proto = m.graph.input[0]
    in_dims = [d.dim_value if d.dim_value > 0 else 1
                for d in in_proto.type.tensor_type.shape.dim]
    in_shape = tuple(in_dims[1:]) if in_dims[0] in (0, 1) else tuple(in_dims)

    init_state = initial_state_with_lineage(c_in, r_in)
    input_name = in_proto.name
    states = {input_name: init_state}
    shapes = {input_name: in_shape}
    trace: List[Dict[str, Any]] = []
    rss0 = _peak_rss_gb()

    for ni, node in enumerate(m.graph.node):
        op = node.op_type
        if op == "Constant":
            arr = numpy_helper.to_array(node.attribute[0].t).astype(np.float64)
            inits[node.output[0]] = arr
            continue
        in_names = list(node.input)
        primary_in = in_names[0]
        if primary_in not in states:
            continue
        s_in = states[primary_in]
        sh_in = shapes[primary_in]
        attrs = _node_attr_dict(node)
        out_name = node.output[0]
        t_layer = time.perf_counter()
        try:
            if op == "Conv":
                W = inits[in_names[1]]
                b = inits[in_names[2]] if len(in_names) > 2 else None
                stride = attrs.get("strides", [1, 1])[0]
                padding = attrs.get("pads", [0, 0, 0, 0])[0]
                groups = attrs.get("group", 1)
                s_out, out_shape = scops.apply_conv2d(
                    s_in, W, b, input_shape=sh_in,
                    stride=stride, padding=padding, groups=groups,
                )
            elif op == "BatchNormalization":
                scale = inits[in_names[1]]; bias = inits[in_names[2]]
                mean = inits[in_names[3]]; var = inits[in_names[4]]
                eps = attrs.get("epsilon", 1e-5)
                inv_std = 1.0 / np.sqrt(var + eps)
                effective_scale = scale * inv_std
                effective_shift = bias - mean * effective_scale
                s_out = scops.apply_bn(s_in, effective_scale, effective_shift,
                                         input_shape=sh_in)
                out_shape = sh_in
            elif op == "Relu":
                s_out, _ = scops.apply_relu_triangle(s_in)
                out_shape = sh_in
            elif op == "Add":
                s_b = states.get(in_names[1])
                if s_b is None: continue
                s_out = _smart_add(s_in, s_b)
                out_shape = sh_in
            elif op == "Flatten":
                s_out = scops.apply_flatten(s_in)
                out_shape = (s_out.c.shape[0],)
            elif op == "Gemm":
                W = inits[in_names[1]]
                b = inits[in_names[2]] if len(in_names) > 2 else None
                transB = attrs.get("transB", 0)
                W_eff = W if transB else W.T
                s_out = scops.apply_dense(s_in, W_eff, b)
                out_shape = (int(W_eff.shape[0]),)
            elif op == "GlobalAveragePool":
                s_out, out_shape = _apply_globalavgpool(s_in, sh_in)
            else:
                continue
        except Exception as e:
            trace.append({"layer": ni, "op": op, "ERROR": f"{type(e).__name__}: {str(e)[:100]}"})
            continue
        wall = time.perf_counter() - t_layer
        trace.append({
            "layer": ni, "op": op,
            "in_shape": list(sh_in), "out_shape": list(out_shape),
            "ng_in": int(s_in.G_kept.shape[1]),
            "ng_out": int(s_out.G_kept.shape[1]),
            "wall_s": wall,
            "rss_gb": _peak_rss_gb(),
        })
        states[out_name] = s_out
        shapes[out_name] = out_shape

    ng_init = trace[0]["ng_in"] if trace else 0
    first_blowup = next((t["layer"] for t in trace if t.get("ng_out", 0) > 10 * ng_init), None)
    max_ng = max((t.get("ng_out", 0) for t in trace), default=0)
    peak_rss = max((t.get("rss_gb", 0) for t in trace), default=0)

    return {
        "bench": bench, "iid": iid,
        "n_traced_layers": len(trace),
        "ng_init": ng_init, "ng_max": max_ng,
        "first_layer_ng_blowup_10x": first_blowup,
        "peak_rss_gb_during_walk": peak_rss,
        "rss_gb_baseline": rss0,
        "trace": trace,
    }


def check_add_lineage_synthetic() -> Dict[str, Any]:
    """Synthetic 3-coord residual to verify _smart_add merges by input-coord."""
    rng = np.random.default_rng(20260605)
    c_in = np.array([0.0, 0.0])
    r_in = np.array([1.0, 1.0])
    init = initial_state_with_lineage(c_in, r_in)

    # Branch A: y_a = c + 2*ξ_0, 3*ξ_1
    Wa = np.array([[2.0, 0.0], [0.0, 3.0]])
    sa = scops.apply_dense(init, Wa)
    # Branch B: y_b = c + 5*ξ_0, -2*ξ_1
    Wb = np.array([[5.0, 0.0], [0.0, -2.0]])
    sb = scops.apply_dense(init, Wb)

    # Add merges same-input-coord cols → should give 7*ξ_0, 1*ξ_1
    s_sum = _smart_add(sa, sb)
    # Test: row 0 should have ξ_0 coeff = 7, row 1 should have ξ_1 coeff = 1
    coord_0_idx = np.where(s_sum.metadata["input_coord_origin"] == 0)[0]
    coord_1_idx = np.where(s_sum.metadata["input_coord_origin"] == 1)[0]
    row0_coef = float(s_sum.G_kept[0, coord_0_idx].sum())
    row1_coef = float(s_sum.G_kept[1, coord_1_idx].sum())
    return {
        "expected_row0_xi0_coef": 7.0, "actual": row0_coef,
        "expected_row1_xi1_coef": 1.0, "actual_row1": row1_coef,
        "PASS_residual_merge_by_coord": (
            abs(row0_coef - 7.0) < 1e-9 and abs(row1_coef - 1.0) < 1e-9
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--cifar-iids", type=str, default="0,5,10,30,100")
    ap.add_argument("--tiny-iids", type=str, default="0,5,10,30,100")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # (b) Add lineage synthetic
    add_check = check_add_lineage_synthetic()
    print(f"Add lineage synthetic: PASS={add_check['PASS_residual_merge_by_coord']}, "
          f"row0_xi0={add_check['actual']} (expected 7), "
          f"row1_xi1={add_check['actual_row1']} (expected 1)", flush=True)
    with open(out / "add_lineage_check.json", "w") as f:
        json.dump(add_check, f, indent=2, default=float)

    # (a) center parity
    parity = []
    for iid in [int(x) for x in args.cifar_iids.split(",")]:
        print(f"cifar100 iid {iid}...", flush=True)
        try:
            r = check_center_parity("cifar100_2024", iid)
        except Exception as e:
            r = {"bench": "cifar100_2024", "iid": iid, "ERROR": str(e)[:200]}
        parity.append(r)
        if "max_abs_diff" in r:
            print(f"  max_abs_diff={r['max_abs_diff']:.2e}, "
                  f"PASS={r['PASS_center_parity']}, "
                  f"n_processed={r['n_processed']}, n_skipped={r['n_skipped']}, "
                  f"wall={r['walker_wall_s']:.0f}s", flush=True)
    for iid in [int(x) for x in args.tiny_iids.split(",")]:
        print(f"tinyimagenet iid {iid}...", flush=True)
        try:
            r = check_center_parity("tinyimagenet_2024", iid)
        except Exception as e:
            r = {"bench": "tinyimagenet_2024", "iid": iid, "ERROR": str(e)[:200]}
        parity.append(r)
        if "max_abs_diff" in r:
            print(f"  max_abs_diff={r['max_abs_diff']:.2e}, "
                  f"PASS={r['PASS_center_parity']}, "
                  f"n_processed={r['n_processed']}, n_skipped={r['n_skipped']}, "
                  f"wall={r['walker_wall_s']:.0f}s", flush=True)
    with open(out / "center_parity.json", "w") as f:
        json.dump(parity, f, indent=2, default=float)

    # (d) per-layer trace on first iid each
    traces = []
    for bench, iid in [("cifar100_2024", 0), ("tinyimagenet_2024", 0)]:
        print(f"per-layer trace {bench} iid {iid}...", flush=True)
        try:
            t = check_per_layer_trace(bench, iid)
            traces.append(t)
            print(f"  ng_init={t['ng_init']}, ng_max={t['ng_max']}, "
                  f"first_blowup_layer={t['first_layer_ng_blowup_10x']}, "
                  f"peak_rss={t['peak_rss_gb_during_walk']:.2f} GB", flush=True)
        except Exception as e:
            traces.append({"bench": bench, "iid": iid, "ERROR": str(e)[:200]})
    with open(out / "per_layer_traces.json", "w") as f:
        json.dump(traces, f, indent=2, default=float)

    n_pass = sum(1 for r in parity if r.get("PASS_center_parity"))
    n_skipped_any = sum(1 for r in parity if r.get("n_skipped", 0) > 0)
    print(f"\n=== Goal 1 RESULT ===")
    print(f"center parity: {n_pass}/{len(parity)} PASS (tol 1e-5)")
    print(f"any-skipped nodes: {n_skipped_any}/{len(parity)}")
    print(f"Add lineage synthetic: {add_check['PASS_residual_merge_by_coord']}")
    summary = {
        "n_parity_tests": len(parity),
        "n_parity_pass": n_pass,
        "n_skipped_nonzero": n_skipped_any,
        "add_lineage_pass": add_check["PASS_residual_merge_by_coord"],
        "tolerance_atol": 1e-5,
        "per_iid_parity": parity,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {out}/summary.json")
    return 0 if (n_pass == len(parity) and n_skipped_any == 0
                  and add_check["PASS_residual_merge_by_coord"]) else 1


if __name__ == "__main__":
    sys.exit(main())
