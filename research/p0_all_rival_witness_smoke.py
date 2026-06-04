"""P0 / iid 11 / rival 48 — structured LP witness smoke.

Per advisor 2026-06-03 P0 directive:
    For each UNKNOWN iid, compute HZ/LP upper bound on (Y[r] - Y[t])
    for ALL rivals (or top-K), sort, take top-K. For each, extract an
    LP witness candidate xi, reconstruct the input, ORT strict replay.
    The first replay that produces a real FAL is the FAL receipt.

This smoke focuses on iid 11 / rival 48 (atlas-missed rival) on the
CIFAR100 resnet_medium top-1 robust spec. It uses the ImageHZ + bridge
+ box-LP path from Step 2.1/2.2 (no constraints anywhere → LP optimum
has a closed-form sign vector for xi).

Principles respected:
- No CROWN, no backward
- No Gurobi/MILP (closed-form for box-LP)
- No BaB
- No random / PGD (the candidate xi comes from HZ/LP feasibility, not
  sampling; ORT only confirms strictly)
- FAL strictly requires ORT-replay receipt

Usage:
    python research/p0_all_rival_witness_smoke.py --iid 11 --topK 5
    python research/p0_all_rival_witness_smoke.py --iid 11 --rival 48
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import torch

ACT_ROOT = Path(__file__).resolve().parent.parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from act.back_end.hybridz_tf.representations import SparseGcZ  # noqa: E402
from act.back_end.imagehz import (  # noqa: E402
    apply_add, apply_conv2d, apply_relu_triangle, flatten_to_sparsegcz,
)
from research.imagehz_cifar_prototype import (  # noqa: E402
    CIFAR_BENCH, load_instance, load_input_box,
    _build_initial_hz, _initializer_map, _apply_channel_affine,
    _fold_bn, _bn_eps, _conv_attrs,
)
from research.imagehz_cifar_step22 import apply_linear_sparsegcz  # noqa: E402


# ── Bridge that ALSO returns the col → xi_id map ─────────────────


def flatten_with_xi_map(hz, *, n_input: int) -> Tuple[SparseGcZ, np.ndarray]:
    """Like flatten_to_sparsegcz but also returns a 1D numpy array
    ``col_xi_id[k] = xi_id of column k in the SparseGcZ``.

    Default merging by xi_id makes the column ordering equal to the
    set of unique xi_ids in the order they first appear; we follow the
    same ``merge_generators_by_xi_id`` pass.
    """
    from act.back_end.imagehz.representation import merge_generators_by_xi_id
    merged = merge_generators_by_xi_id(
        hz.generators, dtype=hz.dtype, device=hz.device,
    )
    col_xi_id = np.array([g.xi_id for g in merged], dtype=np.int64)
    sg = flatten_to_sparsegcz(hz)
    assert sg.ng == col_xi_id.size, "merge / bridge column count mismatch"
    return sg, col_xi_id


# ── Forward through the model ────────────────────────────────────


def forward_to_bridge(
    onnx_path: str, lb: np.ndarray, ub: np.ndarray,
) -> Tuple[SparseGcZ, np.ndarray, Dict[str, Any]]:
    """Walk the ONNX up to the Flatten bridge. Returns
    (bridge SparseGcZ, col_xi_id at bridge, metadata)."""
    m = onnx.load(onnx_path)
    in_dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    C, H, W = int(in_dims[1]), int(in_dims[2]), int(in_dims[3])
    n_in = C * H * W

    inits = _initializer_map(m)
    hz0, next_xi = _build_initial_hz(lb, ub, (C, H, W))
    activations = {m.graph.input[0].name: hz0}

    bridge_hz = None
    pre_flatten_image = None
    for node in m.graph.node:
        op = node.op_type
        ins, outs = list(node.input), list(node.output)
        if op == "Conv":
            x = activations[ins[0]]
            w = torch.from_numpy(inits[ins[1]].astype(np.float64))
            b = torch.from_numpy(inits[ins[2]].astype(np.float64)) if len(ins) > 2 else None
            attrs = _conv_attrs(node)
            activations[outs[0]] = apply_conv2d(
                x, w, b,
                stride=attrs.get("stride", 1),
                padding=attrs.get("padding", 0),
            )
        elif op == "BatchNormalization":
            x = activations[ins[0]]
            bn = {"scale": inits[ins[1]], "B": inits[ins[2]],
                  "mean": inits[ins[3]], "var": inits[ins[4]]}
            gamma, beta = _fold_bn(bn, _bn_eps(node))
            activations[outs[0]] = _apply_channel_affine(x, gamma, beta)
        elif op == "Relu":
            x = activations[ins[0]]
            y, next_xi = apply_relu_triangle(x, next_aux_id=next_xi)
            activations[outs[0]] = y
        elif op == "Add":
            activations[outs[0]] = apply_add(
                activations[ins[0]], activations[ins[1]])
        elif op == "Flatten":
            pre_flatten_image = activations[ins[0]]
            bridge_hz, col_xi_id = flatten_with_xi_map(
                pre_flatten_image, n_input=n_in)
            break

    assert bridge_hz is not None
    meta = {"n_input": n_in, "C": C, "H": H, "W": W,
            "next_xi_after_walk": next_xi}
    return bridge_hz, col_xi_id, meta


def walk_tail_track_xi(
    sg: SparseGcZ, col_xi_id: np.ndarray, onnx_path: str,
    next_xi: int,
) -> Tuple[SparseGcZ, np.ndarray, int]:
    """Apply Gemm_56 + Relu_57(triangle) + Gemm_58 to the bridge
    SparseGcZ, tracking col_xi_id end to end.

    Triangle on SparseGcZ appends k new columns with FRESH xi_ids
    starting at ``next_xi``. Linear preserves column count and order.
    """
    m = onnx.load(onnx_path)
    inits = _initializer_map(m)
    cur_xi = next_xi
    for node in m.graph.node:
        if node.name == "Gemm_56":
            w = torch.from_numpy(inits[node.input[1]].astype(np.float64))
            b = torch.from_numpy(inits[node.input[2]].astype(np.float64))
            sg = apply_linear_sparsegcz(sg, w, b)
            # col_xi_id unchanged.
        elif node.name == "Relu_57":
            # Apply triangle and append fresh xi_ids for the k aux gens.
            sg_before = sg
            sg = sg.apply_relu_triangle()
            k_new = sg.ng - sg_before.ng
            assert k_new >= 0
            new_xi_ids = np.arange(cur_xi, cur_xi + k_new, dtype=np.int64)
            cur_xi += k_new
            col_xi_id = np.concatenate([col_xi_id, new_xi_ids])
        elif node.name == "Gemm_58":
            w = torch.from_numpy(inits[node.input[1]].astype(np.float64))
            b = torch.from_numpy(inits[node.input[2]].astype(np.float64))
            sg = apply_linear_sparsegcz(sg, w, b)
    return sg, col_xi_id, cur_xi


# ── Closed-form box-LP witness extraction ────────────────────────


def closed_form_witness_for_rival(
    sg_out: SparseGcZ, col_xi_id: np.ndarray,
    y_true: int, y_rival: int,
    n_input: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Maximize (Y[r] - Y[t]) under xi ∈ [-1,+1]^ng with no
    constraints. Closed-form: xi_star[k] = sign(Gc[r,k] - Gc[t,k]).

    Returns (lp_max, xi_input_pixels, xi_star_full).
    xi_input_pixels: shape (n_input,) — values in [-1,+1] for each
    input pixel (defaults to 0 for pixels with no generator coupling).
    """
    c = sg_out.c.detach().cpu().numpy().reshape(-1)
    ng = int(sg_out.ng)
    Gc_dense = sg_out.Gc_sparse.to_dense().detach().cpu().numpy()  # (n_out, ng)
    diff_row = Gc_dense[y_rival, :] - Gc_dense[y_true, :]
    xi_star = np.sign(diff_row).astype(np.float64)
    # 0 → +1 by convention (boundary; either sign gives the same max).
    xi_star[xi_star == 0.0] = 1.0
    lp_max = float(c[y_rival] - c[y_true] + np.abs(diff_row).sum())

    # Map xi to input pixels.
    xi_input = np.zeros(n_input, dtype=np.float64)
    # Mask columns whose xi_id is an input pixel (xi_id < n_input).
    is_input = col_xi_id < n_input
    pixel_id = col_xi_id[is_input]
    xi_vals = xi_star[is_input]
    # If multiple columns somehow alias to the same input pixel
    # (shouldn't, because merge_by_xi_id collapses them), we'd take
    # the last one. With merge it's bijective.
    xi_input[pixel_id] = xi_vals
    return lp_max, xi_input, xi_star


def topk_rival_bounds(
    sg_out: SparseGcZ, y_true: int, n_classes: int, K: int,
) -> List[Tuple[int, float]]:
    """Return top-K rivals sorted by descending upper bound on
    (Y[r] - Y[t]).
    """
    c = sg_out.c.detach().cpu().numpy().reshape(-1)
    Gc_dense = sg_out.Gc_sparse.to_dense().detach().cpu().numpy()
    diff_const = c - c[y_true]
    diff_gen = Gc_dense - Gc_dense[y_true:y_true + 1, :]
    ub = diff_const + np.abs(diff_gen).sum(axis=1)
    ub[y_true] = -np.inf
    order = np.argsort(-ub)
    return [(int(i), float(ub[i])) for i in order[:K]]


# ── Strict ORT replay ────────────────────────────────────────────


def strict_ort_replay(
    onnx_path: str, x_cand: np.ndarray,
    in_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Single-input deterministic CPU ORT eval. Returns flat logits."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"],
    )
    inp_name = sess.get_inputs()[0].name
    C, H, W = in_shape
    x_in = x_cand.astype(np.float32).reshape(1, C, H, W)
    y = sess.run(None, {inp_name: x_in})[0]
    return np.asarray(y, dtype=np.float64).reshape(-1)


# ── Driver ──────────────────────────────────────────────────────


def smoke_one_iid_one_rival(
    iid: int, y_true: int, target_rival: int,
    out_dir: Path, K_for_topk: int = 5,
) -> Dict[str, Any]:
    onnx_path, vnn_path = load_instance(iid)
    print(f"[iid={iid}] onnx={onnx_path}")
    print(f"[iid={iid}] vnnlib={vnn_path}")
    print(f"[iid={iid}] target rival={target_rival}, y_true={y_true}")

    m = onnx.load(onnx_path)
    in_dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    C, H, W = int(in_dims[1]), int(in_dims[2]), int(in_dims[3])
    n_in = C * H * W
    lb, ub = load_input_box(vnn_path, n_in)
    c_box = (lb + ub) / 2.0
    half = (ub - lb) / 2.0

    # Forward through conv body, bridge to SparseGcZ.
    t0 = time.perf_counter()
    sg_bridge, col_xi_id_bridge, fwd_meta = forward_to_bridge(onnx_path, lb, ub)
    t_walk = time.perf_counter() - t0
    print(f"[iid={iid}] conv-body+bridge: {t_walk:.2f}s "
          f"(n={sg_bridge.n}, ng={sg_bridge.ng})")

    # Tail (Gemm+ReLU triangle+Gemm).
    t1 = time.perf_counter()
    sg_out, col_xi_id_out, next_xi = walk_tail_track_xi(
        sg_bridge, col_xi_id_bridge, onnx_path,
        next_xi=fwd_meta["next_xi_after_walk"],
    )
    t_tail = time.perf_counter() - t1
    print(f"[iid={iid}] tail: {t_tail:.3f}s "
          f"(n_out={sg_out.n}, ng={sg_out.ng})")

    # Top-K rival bounds.
    top_k = topk_rival_bounds(sg_out, y_true=y_true, n_classes=sg_out.n,
                              K=K_for_topk)
    print(f"[iid={iid}] top-{K_for_topk} rivals by LP upper bound:")
    for r, ub_val in top_k:
        print(f"   rival={r:3d}  Y[r]-Y[t] upper bound = {ub_val:+.4f}")

    # Target rival witness.
    lp_max, xi_input, xi_star_full = closed_form_witness_for_rival(
        sg_out, col_xi_id_out, y_true=y_true, y_rival=target_rival,
        n_input=n_in,
    )
    print(f"[iid={iid}] target rival {target_rival}: LP upper bound "
          f"Y[r]-Y[t] = {lp_max:+.4f}")
    # Reconstruct input candidate.
    x_cand = c_box + half * xi_input
    # Clamp to box (safety).
    x_cand = np.clip(x_cand, lb, ub)
    in_box = bool(np.all(x_cand >= lb - 1e-12)
                  and np.all(x_cand <= ub + 1e-12))
    print(f"[iid={iid}] x_cand reconstructed; in_box={in_box}")

    # Strict ORT replay.
    y_ort = strict_ort_replay(onnx_path, x_cand, in_shape=(C, H, W))
    y_argmax = int(np.argmax(y_ort))
    margin_actual = float(y_ort[target_rival] - y_ort[y_true])
    print(f"[iid={iid}] ORT replay logits:")
    print(f"   Y[{y_true}] (y_true) = {y_ort[y_true]:+.4f}")
    print(f"   Y[{target_rival}] (target rival) = {y_ort[target_rival]:+.4f}")
    print(f"   argmax = {y_argmax}")
    print(f"   actual Y[r] - Y[t] = {margin_actual:+.4f}")

    # Strict FAL check: vnnlib top-1 robust says spec is SAFE iff
    # argmax == y_true. UNSAFE / FAL iff argmax != y_true.
    is_fal = (y_argmax != y_true) and (margin_actual > 0.0)
    verdict = "FALSIFIED" if is_fal else "candidate-did-not-replay"

    print()
    print(f"[iid={iid}] === smoke verdict ===")
    print(f"   LP candidate margin upper bound = {lp_max:+.4f}")
    print(f"   actual ORT margin               = {margin_actual:+.4f}")
    print(f"   ORT argmax matches y_true        = {y_argmax == y_true}")
    print(f"   verdict                          = {verdict}")

    receipt = {
        "source": "p0_all_rival_imagehz_box_lp_witness",
        "iid": iid,
        "onnx_path": onnx_path,
        "vnnlib_path": vnn_path,
        "y_true": int(y_true),
        "target_rival": int(target_rival),
        "lp_upper_bound_y_r_minus_y_t": lp_max,
        "input_box_holds": in_box,
        "x_cand_l2_to_center": float(np.linalg.norm(x_cand - c_box)),
        "ort_y_true_logit": float(y_ort[y_true]),
        "ort_y_rival_logit": float(y_ort[target_rival]),
        "ort_y_argmax": y_argmax,
        "ort_actual_margin": margin_actual,
        "argmax_matches_y_true": bool(y_argmax == y_true),
        "is_falsified": bool(is_fal),
        "verdict": verdict,
        "topK_rivals_lp_ub": top_k,
        "wall_conv_body_s": t_walk,
        "wall_tail_s": t_tail,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / f"p0_iid{iid:03d}_rival{target_rival:03d}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"   receipt: {receipt_path}")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid", type=int, default=11)
    ap.add_argument("--y-true", type=int, default=None,
                    help="y_true from vnnlib parse; default reads atlas")
    ap.add_argument("--rival", type=int, default=48,
                    help="Target rival class for witness")
    ap.add_argument("--topK", type=int, default=5)
    ap.add_argument("--out", type=str,
                    default="audit_results/cifar_unknown_margin_atlas_20260603/PHASE2_P0_SMOKE")
    args = ap.parse_args()

    y_true = args.y_true
    if y_true is None:
        # Fall back to atlas
        atlas_path = (
            "/data1/Kane/ACT/audit_results/cifar_unknown_margin_atlas_20260603/"
            "atlas_v2_bucketed.json"
        )
        with open(atlas_path) as f:
            atlas = json.load(f)
        entry = next((e for e in atlas if e.get("iid") == args.iid), None)
        if entry is None:
            raise RuntimeError(f"iid {args.iid} not in atlas; pass --y-true")
        y_true = int(entry["y_true"])

    out_dir = Path(args.out)
    smoke_one_iid_one_rival(
        iid=args.iid, y_true=y_true, target_rival=args.rival,
        out_dir=out_dir, K_for_topk=args.topK,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
