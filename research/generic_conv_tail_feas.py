#!/usr/bin/env python3
"""Generic final-Conv tail feasibility scout for root-only HZ snapshots.

This is a structured FAL candidate generator for networks whose remaining
tail after a snapshot is a final Conv (often 1x1) followed by Flatten:

    HZ snapshot z -> Conv -> Flatten -> VNNLIB unsafe halfspaces

It solves only the continuous LP over root factors and promotes a result only
when the reconstructed input passes strict raw-ONNX replay. It is candidate-only:
LP infeasibility is not a certificate because the root-only snapshot is an
underapproximation of the full relaxed HZ.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

HYZOR_DIR = Path("/data1/Kane/HyZor")
if str(HYZOR_DIR) not in sys.path:
    sys.path.insert(0, str(HYZOR_DIR))

from receipt_factor_aware_endcap_lp import (  # noqa: E402
    _disjunct_holds_zero_tol,
    _parse_vnnlib_full,
)


def _ort_input_shape(onnx_path: str, n_flat: int) -> Tuple[str, List[int], np.dtype]:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [1 if d is None or isinstance(d, str) else int(d) for d in inp.shape]
    if int(np.prod(shape)) != int(n_flat):
        if len(shape) >= 2 and int(np.prod(shape[1:])) == int(n_flat):
            shape[0] = 1
        else:
            shape = [1, int(n_flat)]
    dtype = np.float32 if inp.type == "tensor(float)" else np.float64
    return inp.name, shape, dtype


def _strict_replay(
    onnx_path: str,
    lb_x: np.ndarray,
    ub_x: np.ndarray,
    disjuncts: List[List[Dict[str, Any]]],
    xi_root: np.ndarray,
) -> Dict[str, Any]:
    inp_name, shape, dtype = _ort_input_shape(onnx_path, int(lb_x.size))
    center = (lb_x + ub_x) / 2.0
    rad = (ub_x - lb_x) / 2.0
    x = center.copy()
    if int(xi_root.size) == int(lb_x.size):
        x = center + rad * np.clip(xi_root, -1.0, 1.0)
    else:
        active = np.flatnonzero(np.abs(rad) > 1e-12)
        if int(active.size) != int(xi_root.size):
            raise RuntimeError(
                f"cannot map xi_root size {xi_root.size} to input dim {lb_x.size}"
            )
        x[active] = center[active] + rad[active] * np.clip(xi_root, -1.0, 1.0)
    in_box = bool(np.all(x >= lb_x - 1e-12) and np.all(x <= ub_x + 1e-12))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y = sess.run(None, {inp_name: x.reshape(shape).astype(dtype)})[0].reshape(-1)
    unsafe = any(_disjunct_holds_zero_tol(y, d) for d in disjuncts)
    return {
        "input_box_holds": in_box,
        "vnnlib_query_holds": bool(unsafe),
        "spec_zero_tol_holds": bool(unsafe),
        "all_checks_pass": bool(in_box and unsafe),
        "ort_argmax": int(np.argmax(y)),
        "ort_max": float(np.max(y)),
    }


def _load_last_conv(onnx_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    model = onnx.load(onnx_path)
    init = {
        x.name: numpy_helper.to_array(x).astype(np.float64)
        for x in model.graph.initializer
    }
    for node in reversed(model.graph.node):
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2 or node.input[1] not in init:
            continue
        W = init[node.input[1]].astype(np.float64)
        b = (
            init[node.input[2]].reshape(-1).astype(np.float64)
            if len(node.input) >= 3 and node.input[2] in init
            else np.zeros((W.shape[0],), dtype=np.float64)
        )
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        strides = tuple(int(x) for x in attrs.get("strides", [1, 1]))
        pads_raw = tuple(int(x) for x in attrs.get("pads", [0, 0, 0, 0]))
        dilations = tuple(int(x) for x in attrs.get("dilations", [1, 1]))
        groups = int(attrs.get("group", 1))
        if groups != 1:
            raise RuntimeError("grouped final Conv is not supported")
        return W, b, {
            "node_name": node.name,
            "weight": node.input[1],
            "bias": node.input[2] if len(node.input) >= 3 else None,
            "strides": strides,
            "pads": pads_raw,
            "dilations": dilations,
            "groups": groups,
        }
    raise RuntimeError(f"no supported final Conv in {onnx_path}")


def _infer_hw(dim: int, cin: int) -> Tuple[int, int]:
    if dim % cin != 0:
        raise RuntimeError(f"snapshot dim {dim} not divisible by Cin {cin}")
    area = dim // cin
    h = int(round(math.sqrt(area)))
    if h * h == area:
        return h, h
    for cand in range(1, int(math.sqrt(area)) + 1):
        if area % cand == 0:
            return cand, area // cand
    raise RuntimeError(f"cannot infer H/W for area {area}")


def _conv_output_hw(h: int, w: int, k_h: int, k_w: int, info: Dict[str, Any]) -> Tuple[int, int]:
    s_h, s_w = info["strides"]
    d_h, d_w = info["dilations"]
    p_top, p_left, p_bot, p_right = info["pads"]
    out_h = (h + p_top + p_bot - d_h * (k_h - 1) - 1) // s_h + 1
    out_w = (w + p_left + p_right - d_w * (k_w - 1) - 1) // s_w + 1
    return int(out_h), int(out_w)


def _make_row_getter(
    c: np.ndarray,
    G: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    info: Dict[str, Any],
):
    cout, cin, k_h, k_w = W.shape
    h_in, w_in = _infer_hw(int(c.size), int(cin))
    h_out, w_out = _conv_output_hw(h_in, w_in, k_h, k_w, info)
    s_h, s_w = info["strides"]
    d_h, d_w = info["dilations"]
    p_top, p_left, _, _ = info["pads"]
    n_out = int(cout * h_out * w_out)

    @lru_cache(maxsize=None)
    def row(j: int) -> Tuple[float, np.ndarray]:
        if j < 0 or j >= n_out:
            raise RuntimeError(f"output index {j} out of range {n_out}")
        co = j // (h_out * w_out)
        rem = j % (h_out * w_out)
        oh = rem // w_out
        ow = rem % w_out
        cy = float(b[co])
        gy = np.zeros((G.shape[1],), dtype=np.float64)
        for ci in range(cin):
            for kh in range(k_h):
                ih = oh * s_h + kh * d_h - p_top
                if ih < 0 or ih >= h_in:
                    continue
                for kw in range(k_w):
                    iw = ow * s_w + kw * d_w - p_left
                    if iw < 0 or iw >= w_in:
                        continue
                    coeff = float(W[co, ci, kh, kw])
                    if coeff == 0.0:
                        continue
                    idx = ci * h_in * w_in + ih * w_in + iw
                    cy += coeff * float(c[idx])
                    gy += coeff * G[idx, :]
        return cy, gy

    return row, {"input_shape": [int(cin), h_in, w_in], "output_shape": [int(cout), h_out, w_out]}


def _add_halfspace_rows(h, row_getter, disjunct: List[Dict[str, Any]], n: int) -> None:
    import highspy

    INF = highspy.kHighsInf
    idx = np.arange(n, dtype=np.int32)
    for con in disjunct:
        k = con["kind"]
        if k == "Yj_le":
            cy, gy = row_getter(int(con["j"]))
            h.addRow(-INF, float(con["c"] - cy), n, idx, gy.astype(np.float64, copy=False))
        elif k == "Yj_ge":
            cy, gy = row_getter(int(con["j"]))
            h.addRow(-INF, float(cy - con["c"]), n, idx, (-gy).astype(np.float64, copy=False))
        elif k == "YjYt":
            cj, gj = row_getter(int(con["j"]))
            ct, gt = row_getter(int(con["t"]))
            h.addRow(-INF, float(cj - ct), n, idx, (-(gj - gt)).astype(np.float64, copy=False))
        elif k == "YjYt_le":
            cj, gj = row_getter(int(con["j"]))
            ct, gt = row_getter(int(con["t"]))
            h.addRow(-INF, float(-(cj - ct)), n, idx, (gj - gt).astype(np.float64, copy=False))
        else:
            raise RuntimeError(f"unsupported halfspace kind {k}")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import highspy

    t0 = time.time()
    snap = pickle.load(open(args.snapshot, "rb"))
    c = snap["c"].numpy().reshape(-1).astype(np.float64)
    G = snap["Gc"].numpy().astype(np.float64)
    if int(snap.get("nb", 0)) != 0 or int(snap.get("nc", 0)) != 0:
        raise RuntimeError("conv-tail scout expects pure root-only snapshot")
    W, b, conv_info = _load_last_conv(args.onnx)
    row_getter, shape_info = _make_row_getter(c, G, W, b, conv_info)
    lb_x, ub_x, disjuncts = _parse_vnnlib_full(args.vnnlib)
    n = int(G.shape[1])

    per_disjunct = []
    fal_receipt = None
    for di, disj in enumerate(disjuncts):
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(args.time_limit_s))
        h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        h.addCols(
            n,
            np.zeros(n, dtype=np.float64),
            -np.ones(n, dtype=np.float64),
            np.ones(n, dtype=np.float64),
            0,
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )
        _add_halfspace_rows(h, row_getter, disj, n)
        h.run()
        status = h.getModelStatus()
        rec: Dict[str, Any] = {
            "disjunct": di,
            "n_halfspaces": len(disj),
            "status": str(status),
        }
        if status == highspy.HighsModelStatus.kOptimal:
            xi = np.asarray(h.getSolution().col_value, dtype=np.float64)
            replay = _strict_replay(args.onnx, lb_x, ub_x, disjuncts, xi)
            rec["replay"] = replay
            if replay["all_checks_pass"] and fal_receipt is None:
                fal_receipt = {
                    "disjunct": di,
                    "n_halfspaces": len(disj),
                    **replay,
                }
        per_disjunct.append(rec)

    return {
        "source": "generic_conv_tail_feas",
        "snapshot": args.snapshot,
        "onnx": args.onnx,
        "vnnlib": args.vnnlib,
        "snapshot_shape": {
            "dim": int(c.size),
            "root_ng": int(G.shape[1]),
            "ng_full": int(snap.get("ng_full", snap.get("ng", G.shape[1]))),
            "type": snap.get("type"),
        },
        "conv": {**conv_info, **shape_info},
        "n_disjuncts": len(disjuncts),
        "verdict": "FAL" if fal_receipt else "UNKNOWN",
        "fal_receipt": fal_receipt,
        "per_disjunct": per_disjunct,
        "wall_s": float(time.time() - t0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--vnnlib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit-s", type=float, default=10.0)
    args = ap.parse_args()
    out = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(
        f"verdict={out['verdict']} disjuncts={out['n_disjuncts']} "
        f"wall={out['wall_s']:.2f}s out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
