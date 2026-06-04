#!/usr/bin/env python3
"""Generic affine-tail feasibility scout for root-only HZ snapshots.

This pilot targets models where a reconstructable HZ snapshot is taken
immediately before a final affine layer:

    HZ snapshot z -> y = W z + b -> VNNLIB unsafe halfspaces

It solves continuous LP feasibility over the root factors xi_root in
[-1, 1]. It is used only as a FAL candidate generator: any feasible LP
candidate must pass strict raw-ONNX replay against the full VNNLIB unsafe
query before it can be counted.

Principles:
  - forward HZ snapshot + continuous LP only
  - no CROWN/backward/gradients
  - no MILP/Gurobi/BaB
  - no random or corner sampling
  - fail closed to UNKNOWN unless strict ORT replay passes
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

HYZOR_DIR = Path("/data1/Kane/HyZor")
if str(HYZOR_DIR) not in sys.path:
    sys.path.insert(0, str(HYZOR_DIR))

from receipt_factor_aware_endcap_lp import (  # noqa: E402
    _parse_vnnlib_full,
    _disjunct_holds_zero_tol,
)


def _load_last_affine(onnx_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    model = onnx.load(onnx_path)
    init = {x.name: numpy_helper.to_array(x).astype(np.float64) for x in model.graph.initializer}
    for node in reversed(model.graph.node):
        if node.op_type not in ("Gemm", "MatMul"):
            continue
        if len(node.input) < 2 or node.input[1] not in init:
            continue
        W = init[node.input[1]]
        b = np.zeros((W.shape[0],), dtype=np.float64)
        if len(node.input) >= 3 and node.input[2] in init:
            b = init[node.input[2]].reshape(-1).astype(np.float64)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Gemm":
            trans_b = int(attrs.get("transB", 0))
            if trans_b == 0:
                W_eff = W.T
            else:
                W_eff = W
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            W_eff = alpha * W_eff
            b_eff = beta * b
        else:
            # ONNX MatMul with constant RHS. If dimensions are ambiguous,
            # fail later on W/input mismatch.
            W_eff = W.T
            b_eff = b
        return W_eff.astype(np.float64), b_eff.astype(np.float64), {
            "node_op": node.op_type,
            "node_name": node.name,
            "weight": node.input[1],
            "bias": node.input[2] if len(node.input) >= 3 else None,
        }
    raise RuntimeError(f"no supported final Gemm/MatMul affine in {onnx_path}")


def _ort_input_shape(onnx_path: str, n_flat: int) -> Tuple[str, List[int], np.dtype]:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = []
    for d in inp.shape:
        shape.append(1 if d is None or isinstance(d, str) else int(d))
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
    x = ((lb_x + ub_x) / 2.0) + ((ub_x - lb_x) / 2.0) * np.clip(xi_root, -1.0, 1.0)
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


def _add_halfspace_rows(h, cy: np.ndarray, Gy: np.ndarray, disjunct: List[Dict[str, Any]]) -> None:
    import highspy

    INF = highspy.kHighsInf
    n = int(Gy.shape[1])
    idx = np.arange(n, dtype=np.int32)
    for c in disjunct:
        k = c["kind"]
        if k == "Yj_ge":
            # cy_j + g_j xi >= c  ->  -g_j xi <= cy_j - c
            row = -Gy[int(c["j"])]
            ub = float(cy[int(c["j"])] - c["c"])
        elif k == "Yj_le":
            # cy_j + g_j xi <= c  ->   g_j xi <= c - cy_j
            row = Gy[int(c["j"])]
            ub = float(c["c"] - cy[int(c["j"])])
        elif k == "YjYt":
            # y_j >= y_t -> -(g_j-g_t) xi <= cy_j-cy_t
            j, t = int(c["j"]), int(c["t"])
            row = -(Gy[j] - Gy[t])
            ub = float(cy[j] - cy[t])
        elif k == "YjYt_le":
            # y_j <= y_t -> (g_j-g_t) xi <= -(cy_j-cy_t)
            j, t = int(c["j"]), int(c["t"])
            row = Gy[j] - Gy[t]
            ub = float(-(cy[j] - cy[t]))
        else:
            raise RuntimeError(f"unsupported halfspace kind {k}")
        h.addRow(-INF, ub, n, idx, row.astype(np.float64, copy=False))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import highspy

    t0 = time.time()
    snap = pickle.load(open(args.snapshot, "rb"))
    c = snap["c"].numpy().reshape(-1).astype(np.float64)
    G = snap["Gc"].numpy().astype(np.float64)
    root_ng = int(snap.get("root_ng", G.shape[1]))
    if int(G.shape[1]) != root_ng:
        raise RuntimeError(f"snapshot G columns {G.shape[1]} != root_ng {root_ng}")
    if int(snap.get("nb", 0)) != 0 or int(snap.get("nc", 0)) != 0:
        raise RuntimeError("affine-tail scout expects pure root-only snapshot")

    W, b, tail_info = _load_last_affine(args.onnx)
    if int(W.shape[1]) != int(c.size):
        raise RuntimeError(f"last affine input {W.shape[1]} != snapshot dim {c.size}")
    cy = W @ c + b
    Gy = W @ G
    lb_x, ub_x, disjuncts = _parse_vnnlib_full(args.vnnlib)
    if int(lb_x.size) != root_ng:
        raise RuntimeError(f"input dim {lb_x.size} != root_ng {root_ng}")

    per_disjunct = []
    fal_receipt = None
    for di, disj in enumerate(disjuncts):
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(args.time_limit_s))
        h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        n = root_ng
        empty_starts = np.zeros(0, dtype=np.int32)
        empty_indices = np.zeros(0, dtype=np.int32)
        empty_values = np.zeros(0, dtype=np.float64)
        h.addCols(
            n,
            np.zeros(n, dtype=np.float64),
            -np.ones(n, dtype=np.float64),
            np.ones(n, dtype=np.float64),
            0,
            empty_starts,
            empty_indices,
            empty_values,
        )
        _add_halfspace_rows(h, cy, Gy, disj)
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

    verdict = "FAL" if fal_receipt else "UNKNOWN"
    out = {
        "source": "generic_affine_tail_feas",
        "snapshot": args.snapshot,
        "onnx": args.onnx,
        "vnnlib": args.vnnlib,
        "snapshot_shape": {
            "dim": int(c.size),
            "root_ng": root_ng,
            "ng_full": int(snap.get("ng_full", snap.get("ng", G.shape[1]))),
            "type": snap.get("type"),
        },
        "tail": tail_info,
        "n_disjuncts": len(disjuncts),
        "verdict": verdict,
        "fal_receipt": fal_receipt,
        "per_disjunct": per_disjunct,
        "wall_s": float(time.time() - t0),
    }
    return out


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
