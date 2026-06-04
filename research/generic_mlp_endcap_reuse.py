#!/usr/bin/env python3
"""Generic one-ReLU MLP end-cap LP with reusable HiGHS model.

This is a research/pilot tool for HZ snapshots taken at a FLATTEN layer
immediately before a classifier tail of the form:

    Flatten snapshot -> linear1 -> ReLU -> linear2 -> logits

It builds the ReLU triangle relaxation once and then changes only the LP
objective for each top-1 rival. This avoids rebuilding identical rows for
every class, which is the main cost of the older CIFAR pilot.

Principles:
  - forward-only HZ snapshot + continuous LP only
  - no gradients, no CROWN/backward, no MILP/Gurobi, no BaB, no sampling
  - FAL is promoted only if the LP xi_root candidate passes strict raw-ONNX
    ORT replay against the full VNNLIB unsafe query
  - CERT is promoted only if every rival LP has strictly positive margin
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort


HYZOR_DIR = Path("/data1/Kane/HyZor")
if str(HYZOR_DIR) not in sys.path:
    sys.path.insert(0, str(HYZOR_DIR))

from pilot_cifar_endcap_lp import (  # noqa: E402
    _build_h39_affine,
    _h39_bounds,
    _parse_top1_spec,
)
from receipt_factor_aware_endcap_lp import (  # noqa: E402
    _disjunct_holds_zero_tol,
    _parse_vnnlib_full,
)


def _extract_tail_dense_layers(onnx_path: str):
    """Return [(W, b)] for the affine layers AFTER the last Flatten.

    Walks the ONNX graph forward from the last Flatten; collects up to
    two consecutive Gemm/MatMul layers (with an optional Relu between).
    Each (W, b) is returned in (out_dim, in_dim) PyTorch convention so
    that ``y = W @ x + b``.

    Supported tail patterns:
    - ``Flatten -> Gemm/MatMul``                     (single Dense head)
    - ``Flatten -> Gemm -> Relu -> Gemm``            (2-layer MLP head)
    - ``Flatten -> Gemm -> Gemm``                    (degenerate 2-layer)

    Honors Gemm ``transB`` attribute and ignores Dropout/Identity/Cast
    noise. Raises ``RuntimeError`` on any other tail shape (e.g. Conv
    after Flatten, or extra activations).
    """
    import onnx
    from onnx import numpy_helper as _np_helper
    m = onnx.load(onnx_path)
    init = {i.name: _np_helper.to_array(i) for i in m.graph.initializer}
    nodes = list(m.graph.node)
    last_flatten = max(
        (i for i, n in enumerate(nodes) if n.op_type == "Flatten"),
        default=None,
    )
    if last_flatten is None:
        raise RuntimeError("ONNX has no Flatten op")
    # Walk forward from after the last Flatten
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    last_was_dense = False
    for n in nodes[last_flatten + 1:]:
        op = n.op_type
        if op in ("Identity", "Cast", "Constant", "Dropout"):
            continue
        if op == "Relu":
            if not last_was_dense or len(layers) >= 2:
                raise RuntimeError(
                    f"unexpected Relu in tail at {n.name}")
            last_was_dense = False
            continue
        if op in ("Gemm", "MatMul"):
            W_name = n.input[1]
            if W_name not in init:
                raise RuntimeError(
                    f"missing weight initializer {W_name!r} for {n.name}")
            W = init[W_name].astype(np.float64, copy=True)
            if op == "Gemm":
                trans_b = next(
                    (a.i for a in n.attribute if a.name == "transB"), 0
                )
                # With transB=1 the stored shape is already (out, in).
                # With transB=0 the stored shape is (in, out) and the
                # matmul forms `A @ B` so we transpose to canonicalize
                # as (out, in).
                if trans_b == 0:
                    W = W.T
                # Bias
                if len(n.input) >= 3:
                    b_name = n.input[2]
                    if b_name in init:
                        b = init[b_name].astype(
                            np.float64, copy=True).reshape(-1)
                    else:
                        b = np.zeros(W.shape[0], dtype=np.float64)
                else:
                    b = np.zeros(W.shape[0], dtype=np.float64)
            else:  # MatMul has no transB attribute; assume stored as (in, out)
                W = W.T
                b = np.zeros(W.shape[0], dtype=np.float64)
            layers.append((W, b))
            last_was_dense = True
            if len(layers) == 2:
                break
            continue
        # Unsupported tail op
        raise RuntimeError(
            f"unsupported tail op {op!r} after last Flatten")
    if not layers:
        raise RuntimeError("tail has no Gemm/MatMul layers")
    return layers


def _ort_input_shape(onnx_path: str, n_flat: int) -> Tuple[str, List[int], np.dtype]:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = []
    for d in inp.shape:
        if d is None or isinstance(d, str):
            shape.append(1)
        else:
            shape.append(int(d))
    prod = int(np.prod(shape))
    if prod != int(n_flat):
        if len(shape) >= 2 and int(np.prod(shape[1:])) == int(n_flat):
            shape[0] = 1
        elif int(n_flat) > 0:
            shape = [1, int(n_flat)]
        else:
            raise RuntimeError(f"cannot infer ORT input shape for n={n_flat}")
    dtype = np.float32 if inp.type == "tensor(float)" else np.float64
    return inp.name, shape, dtype


def _build_reusable_model(
    c39: np.ndarray,
    G39: np.ndarray,
    l39: np.ndarray,
    u39: np.ndarray,
    time_limit_s: float,
):
    import highspy

    ng = int(G39.shape[1])
    stable_active = np.where(l39 >= 0.0)[0]
    stable_inactive = np.where(u39 <= 0.0)[0]
    unstable = np.where((l39 < 0.0) & (u39 > 0.0))[0]
    n_unstable = int(unstable.size)
    n_vars = ng + n_unstable

    h = highspy.Highs()
    h.silent()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit_s))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    empty_starts = np.zeros(0, dtype=np.int32)
    empty_indices = np.zeros(0, dtype=np.int32)
    empty_values = np.zeros(0, dtype=np.float64)
    h.addCols(
        n_vars,
        np.zeros(n_vars, dtype=np.float64),
        np.concatenate([
            -np.ones(ng, dtype=np.float64),
            np.zeros(n_unstable, dtype=np.float64),
        ]),
        np.concatenate([
            np.ones(ng, dtype=np.float64),
            u39[unstable].astype(np.float64, copy=False),
        ]),
        0,
        empty_starts,
        empty_indices,
        empty_values,
    )

    xi_indices = np.arange(ng, dtype=np.int32)
    INF = highspy.kHighsInf
    for ridx, i in enumerate(unstable):
        l_i = float(l39[i])
        u_i = float(u39[i])
        if (u_i - l_i) <= 0.0:
            continue
        y_idx = np.asarray([ng + ridx], dtype=np.int32)

        # y - G_i xi >= c_i
        idx1 = np.concatenate([xi_indices, y_idx])
        val1 = np.concatenate([
            -G39[i, :].astype(np.float64, copy=False),
            np.asarray([1.0], dtype=np.float64),
        ])
        h.addRow(float(c39[i]), INF, int(idx1.size), idx1, val1)

        # y - slope * G_i xi <= slope * (c_i - l_i)
        slope = u_i / (u_i - l_i)
        idx2 = idx1
        val2 = np.concatenate([
            (-slope * G39[i, :]).astype(np.float64, copy=False),
            np.asarray([1.0], dtype=np.float64),
        ])
        h.addRow(
            -INF,
            slope * (float(c39[i]) - l_i),
            int(idx2.size),
            idx2,
            val2,
        )

    return h, stable_active, stable_inactive, unstable


def _set_objective(
    h,
    W41: np.ndarray,
    b41: np.ndarray,
    c39: np.ndarray,
    G39: np.ndarray,
    stable_active: np.ndarray,
    unstable: np.ndarray,
    y_true: int,
    rival: int,
) -> float:
    coef = W41[y_true, :] - W41[rival, :]
    ng = int(G39.shape[1])
    n_vars = ng + int(unstable.size)
    cost = np.zeros(n_vars, dtype=np.float64)
    offset = float(b41[y_true] - b41[rival])
    if stable_active.size:
        cost[:ng] = coef[stable_active] @ G39[stable_active, :]
        offset += float(coef[stable_active] @ c39[stable_active])
    if unstable.size:
        cost[ng:] = coef[unstable]
    idx = np.arange(n_vars, dtype=np.int32)
    st = h.changeColsCost(int(n_vars), idx, cost)
    if str(st) != "HighsStatus.kOk":
        raise RuntimeError(f"changeColsCost failed: {st}")
    st = h.changeObjectiveOffset(float(offset))
    if str(st) != "HighsStatus.kOk":
        raise RuntimeError(f"changeObjectiveOffset failed: {st}")
    return offset


def _set_objective_single_dense(
    h,
    W1: np.ndarray,
    b1: np.ndarray,
    c: np.ndarray,
    Gc: np.ndarray,
    y_true: int,
    rival: int,
) -> float:
    """Single-Dense head: objective is just the linear margin.

    Margin = (W1[y_true] - W1[rival]) @ (c + Gc @ xi) +
             (b1[y_true] - b1[rival])

    The HiGHS model has only the ng xi columns (each in [-1, 1]); no
    ReLU triangle constraints. Caller swaps costs per rival via
    `changeColsCost`.
    """
    coef = W1[y_true, :] - W1[rival, :]
    c_flat = c.reshape(-1)
    cost = (coef @ Gc).astype(np.float64, copy=False)
    ng = int(Gc.shape[1])
    offset = float(b1[y_true] - b1[rival]) + float(coef @ c_flat)
    idx = np.arange(ng, dtype=np.int32)
    st = h.changeColsCost(int(ng), idx, cost)
    if str(st) != "HighsStatus.kOk":
        raise RuntimeError(f"changeColsCost failed (single-dense): {st}")
    st = h.changeObjectiveOffset(float(offset))
    if str(st) != "HighsStatus.kOk":
        raise RuntimeError(f"changeObjectiveOffset failed (single-dense): {st}")
    return offset


def _strict_replay(
    onnx_path: str,
    vnnlib_path: str,
    lb_x: np.ndarray,
    ub_x: np.ndarray,
    disjuncts: List[List[Dict[str, Any]]],
    xi_root: np.ndarray,
) -> Dict[str, Any]:
    inp_name, shape, dtype = _ort_input_shape(onnx_path, int(lb_x.size))
    x = ((lb_x + ub_x) / 2.0) + ((ub_x - lb_x) / 2.0) * np.clip(
        xi_root, -1.0, 1.0
    )
    in_box = bool(
        np.all(x >= lb_x - 1e-12) and np.all(x <= ub_x + 1e-12)
    )
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.snapshot, "rb") as f:
        snap = pickle.load(f)
    c = snap["c"].numpy().astype(np.float64, copy=False)
    Gc = snap["Gc"].numpy().astype(np.float64, copy=False)
    nb = int(snap.get("nb", 0))
    nc = int(snap.get("nc", 0))
    root_ng = int(snap.get("root_ng", 0))

    lb_x, ub_x, disjuncts = _parse_vnnlib_full(args.vnnlib)
    input_dim = int(lb_x.size)
    if nb != 0 or nc != 0:
        raise RuntimeError(f"snapshot must be pure HZono, got nb={nb} nc={nc}")
    if root_ng != input_dim:
        raise RuntimeError(
            f"root_ng={root_ng} != input_dim={input_dim}; cannot reconstruct"
        )
    if root_ng > int(snap.get("ng", Gc.shape[1])):
        raise RuntimeError(
            f"root_ng={root_ng} > ng={snap.get('ng')}; root compressed"
        )

    # Robust ONNX tail extractor — works for both PyTorch (linear1.weight)
    # and ONNX-name-mangled (fc.weight, model.12.weight) initializers.
    tail = _extract_tail_dense_layers(args.onnx)
    n_tail = len(tail)
    if n_tail not in (1, 2):
        raise RuntimeError(
            f"tail must be 1 or 2 Gemm layers, got {n_tail}")
    if int(tail[0][0].shape[1]) != int(c.shape[0]):
        raise RuntimeError(
            f"tail[0] input dim {tail[0][0].shape[1]} != snapshot dim "
            f"{c.shape[0]}"
        )
    y_true, rivals = _parse_top1_spec(args.vnnlib)
    if args.max_rivals > 0:
        rivals = rivals[: args.max_rivals]

    if n_tail == 2:
        W39, b39 = tail[0]
        W41, b41 = tail[1]
        c39, G39 = _build_h39_affine(W39, b39, c, Gc)
        l39, u39 = _h39_bounds(c39, G39)
        h, stable_active, stable_inactive, unstable = _build_reusable_model(
            c39, G39, l39, u39, args.time_limit_s
        )
        _single_dense_path = False
    else:
        # 1-layer head: tail is Flatten -> Gemm -> logits.
        # Margin per rival:
        #   m[r] = (W[r] - W[y_true]) @ c + (b[r] - b[y_true]) +
        #          (W[r] - W[y_true]) @ Gc @ xi
        # min over xi in [-1, 1]^ng is closed-form: const - sum(|coef|).
        # We still drive HiGHS for consistency with the 2-layer path
        # (objective swap via changeColsCost), but the model has no
        # ReLU triangle, only the box xi in [-1, 1]^ng.
        W1, b1 = tail[0]
        # No middle representation; use the raw snapshot
        W39 = W1
        b39 = b1
        W41 = np.zeros((W1.shape[0], 0), dtype=np.float64)
        b41 = np.zeros(W1.shape[0], dtype=np.float64)
        c39 = c.reshape(-1).copy()
        G39 = Gc.copy()
        l39, u39 = _h39_bounds(c39, G39)
        # Empty unstable set => no ReLU aux columns
        ng = int(G39.shape[1])
        import highspy as _hsp
        h = _hsp.Highs()
        h.silent()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(args.time_limit_s))
        h.changeObjectiveSense(_hsp.ObjSense.kMinimize)
        h.addCols(
            ng,
            np.zeros(ng, dtype=np.float64),
            -np.ones(ng, dtype=np.float64),
            np.ones(ng, dtype=np.float64),
            0,
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )
        stable_active = np.zeros(0, dtype=np.int64)
        stable_inactive = np.zeros(0, dtype=np.int64)
        unstable = np.zeros(0, dtype=np.int64)
        _single_dense_path = True

    import highspy

    t0 = time.time()
    per_rival = {}
    fal_receipt = None
    n_ok = 0
    col_values_for_worst = None
    worst_rival = None
    worst_lp = float("inf")
    n_vars = int(G39.shape[1]) + int(unstable.size)
    for k, rival in enumerate(rivals):
        if _single_dense_path:
            _set_objective_single_dense(
                h, W39, b39, c39, G39, y_true, rival
            )
        else:
            _set_objective(
                h, W41, b41, c39, G39, stable_active, unstable,
                y_true, rival
            )
        h.run()
        ms = h.getModelStatus()
        if ms != highspy.HighsModelStatus.kOptimal:
            per_rival[int(rival)] = {
                "status": f"model_status:{ms!s}",
                "lp_min": None,
            }
            continue
        lp = float(h.getObjectiveValue())
        n_ok += 1
        per_rival[int(rival)] = {"status": "ok", "lp_min": lp}
        if lp < worst_lp:
            worst_lp = lp
            worst_rival = int(rival)
            col_values_for_worst = np.asarray(
                h.getSolution().col_value, dtype=np.float64
            )
        if lp <= args.fal_replay_threshold:
            xi_root = np.asarray(h.getSolution().col_value[:root_ng], dtype=np.float64)
            replay = _strict_replay(
                args.onnx, args.vnnlib, lb_x, ub_x, disjuncts, xi_root
            )
            per_rival[int(rival)]["replay"] = replay
            if replay["all_checks_pass"] and fal_receipt is None:
                fal_receipt = {
                    "rival": int(rival),
                    "lp_min": lp,
                    **replay,
                }
        if args.progress_every and (
            k < 5 or (k + 1) % args.progress_every == 0 or k + 1 == len(rivals)
        ):
            print(
                f"rival {k + 1}/{len(rivals)} id={rival} lp={lp:.6g} "
                f"worst={worst_lp:.6g}",
                flush=True,
            )
    wall = time.time() - t0
    ok_vals = [
        r["lp_min"] for r in per_rival.values()
        if r["status"] == "ok" and r["lp_min"] is not None
    ]
    n_pos = int(sum(1 for v in ok_vals if v > args.cert_tol))
    cert = bool(n_ok == len(rivals) and n_pos == len(rivals))
    verdict = "CERT" if cert else ("FAL" if fal_receipt else "UNKNOWN")
    out = {
        "source": "generic_mlp_endcap_reuse",
        "snapshot": args.snapshot,
        "onnx": args.onnx,
        "vnnlib": args.vnnlib,
        "snapshot_shape": {
            "dim": int(snap.get("dim", c.shape[0])),
            "ng": int(snap.get("ng", Gc.shape[1])),
            "nb": nb,
            "nc": nc,
            "root_ng": root_ng,
        },
        "tail": {
            "n_dense_layers": 1 if _single_dense_path else 2,
            "linear1": list(W39.shape),
            "linear2": (list(W41.shape)
                        if not _single_dense_path else None),
            "stable_active": int(stable_active.size),
            "stable_inactive": int(stable_inactive.size),
            "unstable": int(unstable.size),
        },
        "y_true": int(y_true),
        "n_rivals": int(len(rivals)),
        "n_optimal": int(n_ok),
        "n_positive": int(n_pos),
        "lp_min": float(min(ok_vals)) if ok_vals else None,
        "lp_max": float(max(ok_vals)) if ok_vals else None,
        "worst_rival": worst_rival,
        "worst_lp": float(worst_lp) if ok_vals else None,
        "verdict": verdict,
        "fal_receipt": fal_receipt,
        "wall_s": float(wall),
        "per_rival": per_rival,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--vnnlib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit-s", type=float, default=15.0)
    ap.add_argument("--max-rivals", type=int, default=0)
    ap.add_argument("--cert-tol", type=float, default=1e-8)
    ap.add_argument("--fal-replay-threshold", type=float, default=0.0)
    ap.add_argument("--progress-every", type=int, default=20)
    args = ap.parse_args()
    result = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(
        f"verdict={result['verdict']} n_pos={result['n_positive']}/"
        f"{result['n_rivals']} lp_min={result['lp_min']} "
        f"wall={result['wall_s']:.2f}s out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
