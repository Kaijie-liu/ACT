"""ImageHZ Step 2.1 — single-iid smoke driver (CONV BODY ONLY, NO LP).

Per advisor 2026-06-03 phased plan:
- Goal: prove the ImageHZ representation can walk the CIFAR resnet_medium
  conv body end-to-end and exit at the Flatten point as a SparseGcZ
  whose `num_unique_xi_id == num_generators` (merge correct), with no
  OOM and wall-time < 20 min.
- Scope: ONE iid (default 11), conv body + bridge ONLY.
- NOT in scope yet: tail LP, margins, V/A verdict, batch sentinel runs.

Usage:
    python research/imagehz_cifar_prototype.py --iid 11
    python research/imagehz_cifar_prototype.py --iid 11 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import torch

# Make package importable when run as a script.
ACT_ROOT = Path(__file__).resolve().parent.parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from act.back_end.imagehz import (  # noqa: E402
    BoundingBox,
    ImageHZ,
    SpatialGenerator,
    apply_add,
    apply_conv2d,
    apply_maxpool2d,  # noqa: F401  (kept import so failure is loud if reached)
    apply_relu_triangle,
    flatten_to_sparsegcz,
)

# 2026-06-03 REC3: canonical-root provenance guard. The prior LOCAL
# constants below are intentionally kept (commented out) as a tombstone
# so future readers see the bug we recovered from.
#
#   _LOCAL_BUG_CIFAR_BENCH = "/data1/Kane/ACT/data/vnnlib/cifar100_2024"
#   _LOCAL_BUG_INSTANCES_CSV = f"{_LOCAL_BUG_CIFAR_BENCH}/instances.csv"
#
# These pointed at a vnnlib pool that has zero file-overlap with the
# canonical vnncomp2025 set, so every iid in the prior P0 dispatch
# referred to a different vnnlib file than what the baseline runner
# used. See memory: project_p0_fal_diff_clean.md (RETRACTED).

from research.canonical_provenance import (  # noqa: E402
    CANONICAL_ROOT,
    load_instance as _canonical_load_instance,
)

CIFAR_BENCH = str(CANONICAL_ROOT / "cifar100_2024")
INSTANCES_CSV = str(CANONICAL_ROOT / "cifar100_2024" / "instances.csv")


# ── Spec loading ──────────────────────────────────────────────────


def load_instance(iid: int) -> Tuple[str, str]:
    """Delegates to canonical_provenance.load_instance. Fail-closed
    on any non-canonical path. Returns (onnx_path, vnnlib_path) as
    str (not Path) to preserve the prior call sites' signature.
    """
    onnx_p, vnn_p = _canonical_load_instance("cifar100_2024", iid)
    return (str(onnx_p), str(vnn_p))


def _legacy_unused_call(iid: int) -> Tuple[str, str]:
    """Tombstone for the prior load_instance body. Do NOT call.
    Retained so audit tools can grep for the pattern.
    """
    raise RuntimeError(
        "REC3 guard: the prior LOCAL-pool load_instance has been "
        "retired. Use canonical_provenance.load_instance instead."
    )
    with open(INSTANCES_CSV) as f:  # pragma: no cover  # noqa
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    onnx_rel, vnnlib_rel, _ = rows[iid]
    return (
        os.path.join(CIFAR_BENCH, onnx_rel),
        os.path.join(CIFAR_BENCH, vnnlib_rel),
    )


def load_input_box(vnnlib_path: str, n_in: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quick lb/ub parser for CIFAR-style box specs.

    The CIFAR100 vnnlib files are simple per-pixel box constraints:
        (assert (>= X_i lb_i))
        (assert (<= X_i ub_i))
    No OR, no joint constraints on inputs.
    """
    lb = np.full(n_in, -np.inf, dtype=np.float64)
    ub = np.full(n_in, np.inf, dtype=np.float64)
    import re
    with open(vnnlib_path) as f:
        text = f.read()
    pat_geq = re.compile(r"\(assert\s+\(>=\s+X_(\d+)\s+([-0-9eE.+]+)\s*\)\s*\)")
    pat_leq = re.compile(r"\(assert\s+\(<=\s+X_(\d+)\s+([-0-9eE.+]+)\s*\)\s*\)")
    for m in pat_geq.finditer(text):
        i = int(m.group(1)); v = float(m.group(2))
        lb[i] = max(lb[i], v) if np.isfinite(lb[i]) else v
    for m in pat_leq.finditer(text):
        i = int(m.group(1)); v = float(m.group(2))
        ub[i] = min(ub[i], v) if np.isfinite(ub[i]) else v
    if np.isinf(lb).any() or np.isinf(ub).any():
        raise ValueError("vnnlib parse left unbounded inputs")
    return lb, ub


# ── ONNX walker ───────────────────────────────────────────────────


def _initializer_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _build_initial_hz(
    lb: np.ndarray, ub: np.ndarray, shape_chw: Tuple[int, int, int],
) -> Tuple[ImageHZ, int]:
    """Build the initial ImageHZ from a per-pixel input box.

    Each input pixel gets its own xi_id (0 .. n_in - 1) and one
    single-pixel generator. The center is (lb + ub) / 2; the value is
    (ub - lb) / 2.
    """
    C, H, W = shape_chw
    n = C * H * W
    assert lb.shape == (n,)
    center = ((lb + ub) / 2.0).reshape(C, H, W)
    radius = ((ub - lb) / 2.0).reshape(C, H, W)
    c_t = torch.from_numpy(center).to(torch.float64)
    gens: List[SpatialGenerator] = []
    next_id = 0
    for ci in range(C):
        for hi in range(H):
            for wi in range(W):
                r = float(radius[ci, hi, wi])
                if r == 0.0:
                    next_id += 1
                    continue
                region = BoundingBox(ci, ci + 1, hi, hi + 1, wi, wi + 1)
                val = torch.tensor([[[r]]], dtype=torch.float64)
                gens.append(SpatialGenerator(
                    region=region, values=val, xi_id=next_id,
                ))
                next_id += 1
    hz = ImageHZ(c=c_t, generators=gens)
    return hz, next_id


def _fold_bn(
    bn_inits: Dict[str, np.ndarray],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fold a BatchNormalization into a per-channel (gamma, beta) pair.

    Output of BN: y = gamma_eff * x + beta_eff   (channel-wise)
    where:
      gamma_eff = scale / sqrt(running_var + eps)
      beta_eff  = bias - running_mean * gamma_eff
    """
    scale = bn_inits["scale"].astype(np.float64)
    bias = bn_inits["B"].astype(np.float64)
    mean = bn_inits["mean"].astype(np.float64)
    var = bn_inits["var"].astype(np.float64)
    gamma_eff = scale / np.sqrt(var + eps)
    beta_eff = bias - mean * gamma_eff
    return gamma_eff, beta_eff


def _apply_channel_affine(
    hz: ImageHZ, gamma: np.ndarray, beta: np.ndarray,
) -> ImageHZ:
    """Apply y[c,h,w] = gamma[c] * x[c,h,w] + beta[c] in-place on a copy.

    This is the post-BN-fold operator. Linear in generators (each
    generator's values get scaled by gamma per channel; center gets
    scaled + shifted).
    """
    g_t = torch.from_numpy(gamma).to(torch.float64).view(-1, 1, 1)
    b_t = torch.from_numpy(beta).to(torch.float64).view(-1, 1, 1)
    new_c = hz.c * g_t + b_t
    new_gens: List[SpatialGenerator] = []
    for gen in hz.generators:
        r = gen.region
        # gamma slice for this generator's channel range
        g_slice = g_t[r.c_lo:r.c_hi, :, :]  # (cR, 1, 1)
        new_vals = gen.values * g_slice
        if (new_vals.abs() == 0).all():
            continue
        new_gens.append(SpatialGenerator(
            region=gen.region, values=new_vals.contiguous(), xi_id=gen.xi_id,
        ))
    return ImageHZ(c=new_c, generators=new_gens)


def _conv_attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for a in node.attribute:
        if a.name == "strides":
            attrs["stride"] = int(a.ints[0])
        elif a.name == "pads":
            # ONNX uses [start_h, start_w, end_h, end_w]
            attrs["padding"] = int(a.ints[0])
        elif a.name == "dilations":
            attrs["dilations"] = list(a.ints)
        elif a.name == "kernel_shape":
            attrs["kernel_shape"] = list(a.ints)
        elif a.name == "group":
            attrs["group"] = int(a.i)
    return attrs


def _bn_eps(node: onnx.NodeProto) -> float:
    for a in node.attribute:
        if a.name == "epsilon":
            return float(a.f)
    return 1e-5


# ── Layer-level dispatch ──────────────────────────────────────────


def run_conv_body(
    onnx_path: str, lb: np.ndarray, ub: np.ndarray,
    log_path: Path,
) -> Dict[str, Any]:
    """Walk the ONNX graph node-by-node from the input until the
    Flatten node, propagating an ImageHZ activation per tensor.

    Returns a report dict with per-layer telemetry + final stats.
    """
    model = onnx.load(onnx_path)
    inits = _initializer_map(model)

    input_name = model.graph.input[0].name
    # CIFAR is NCHW with N=1; ignore N.
    in_dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    C, H, W = int(in_dims[1]), int(in_dims[2]), int(in_dims[3])
    n_in = C * H * W

    t_init_start = time.perf_counter()
    hz0, next_xi = _build_initial_hz(lb, ub, (C, H, W))
    init_wall = time.perf_counter() - t_init_start

    activations: Dict[str, ImageHZ] = {input_name: hz0}

    per_layer: List[Dict[str, Any]] = []
    per_layer.append({
        "step": 0,
        "op": "INPUT_BOX",
        "output": input_name,
        "wall_s": init_wall,
        **hz0.stats(),
    })

    # Walk.
    t_walk_start = time.perf_counter()
    flatten_seen = False
    bridge_sparsegcz = None
    for step, node in enumerate(model.graph.node, start=1):
        op = node.op_type
        ins = list(node.input)
        outs = list(node.output)
        # Skip BN params, Conv weight tensors — they're initializers, not
        # active tensors.
        op_start = time.perf_counter()

        if op == "Conv":
            x_name = ins[0]
            w = inits[ins[1]].astype(np.float64)
            b = inits[ins[2]].astype(np.float64) if len(ins) > 2 else None
            w_t = torch.from_numpy(w).to(torch.float64)
            b_t = torch.from_numpy(b).to(torch.float64) if b is not None else None
            attrs = _conv_attrs(node)
            stride = attrs.get("stride", 1)
            padding = attrs.get("padding", 0)
            x = activations[x_name]
            y = apply_conv2d(x, w_t, b_t, stride=stride, padding=padding)
            activations[outs[0]] = y
            stats = y.stats()
            stats["_conv_kernel"] = list(w.shape)
            stats["_conv_stride"] = stride
            stats["_conv_pad"] = padding
        elif op == "BatchNormalization":
            x_name = ins[0]
            bn = {
                "scale": inits[ins[1]],
                "B": inits[ins[2]],
                "mean": inits[ins[3]],
                "var": inits[ins[4]],
            }
            gamma, beta = _fold_bn(bn, _bn_eps(node))
            x = activations[x_name]
            y = _apply_channel_affine(x, gamma, beta)
            activations[outs[0]] = y
            stats = y.stats()
        elif op == "Relu":
            x = activations[ins[0]]
            y, next_xi = apply_relu_triangle(x, next_aux_id=next_xi)
            activations[outs[0]] = y
            stats = y.stats()
        elif op == "Add":
            a = activations[ins[0]]
            b = activations[ins[1]]
            y = apply_add(a, b)
            activations[outs[0]] = y
            stats = y.stats()
        elif op == "Flatten":
            flatten_seen = True
            x = activations[ins[0]]
            bridge_t0 = time.perf_counter()
            bridge_sparsegcz = flatten_to_sparsegcz(x)
            bridge_wall = time.perf_counter() - bridge_t0
            stats = x.stats()
            stats["_bridge_wall_s"] = bridge_wall
            stats["_sparsegcz_n"] = int(bridge_sparsegcz.c.numel())
            stats["_sparsegcz_ng"] = int(bridge_sparsegcz.Gc_sparse.shape[1])
            stats["_sparsegcz_nnz"] = int(bridge_sparsegcz.Gc_sparse._nnz())
        elif op == "MaxPool":
            raise NotImplementedError(
                "ImageHZ guard: MaxPool encountered in conv body — "
                "this model is incompatible with the prototype."
            )
        elif op in {"Gemm", "MatMul"}:
            # Tail (post-flatten) — Step 2.1 stops at the bridge.
            stats = {
                "_note": "tail op skipped (Step 2.1 stops at bridge)",
            }
        else:
            # Unknown op — flag and stop.
            raise NotImplementedError(f"unhandled op {op} at step {step}")

        wall = time.perf_counter() - op_start
        per_layer.append({
            "step": step,
            "op": op,
            "output": outs[0] if outs else "",
            "wall_s": wall,
            **stats,
        })

        if flatten_seen:
            # Bridge done; Step 2.1 STOPS here per advisor plan.
            break

    total_walk = time.perf_counter() - t_walk_start

    report: Dict[str, Any] = {
        "iid_onnx": onnx_path,
        "input_shape_chw": [C, H, W],
        "n_in": n_in,
        "next_xi_at_bridge": next_xi,
        "wall_init_s": init_wall,
        "wall_walk_s": total_walk,
        "wall_total_s": init_wall + total_walk,
        "per_layer": per_layer,
        "reached_flatten": flatten_seen,
    }
    if bridge_sparsegcz is not None:
        report["sparsegcz_n"] = int(bridge_sparsegcz.c.numel())
        report["sparsegcz_ng"] = int(bridge_sparsegcz.Gc_sparse.shape[1])
        report["sparsegcz_nnz"] = int(bridge_sparsegcz.Gc_sparse._nnz())

    # Acceptance: the bridge SparseGcZ ng must equal num_unique_xi_id in
    # the pre-flatten ImageHZ (because merge_by_xi_id collapses each
    # xi_id to one column).
    pre_flat = next(
        e for e in reversed(per_layer)
        if e["op"] == "Flatten"
    )
    report["acceptance_merge_correct"] = bool(
        report.get("sparsegcz_ng") == pre_flat["num_unique_xi_id"]
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ── CLI ───────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid", type=int, default=11)
    ap.add_argument(
        "--out", type=str,
        default="audit_results/cifar_unknown_margin_atlas_20260603/PHASE2_STEP21",
    )
    args = ap.parse_args()

    onnx_path, vnn_path = load_instance(args.iid)
    print(f"[iid={args.iid}] onnx={onnx_path}")
    print(f"[iid={args.iid}] vnnlib={vnn_path}")

    # Quick n_in from ONNX
    m = onnx.load(onnx_path)
    in_dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    n_in = int(in_dims[1]) * int(in_dims[2]) * int(in_dims[3])

    lb, ub = load_input_box(vnn_path, n_in)
    print(f"[iid={args.iid}] input box: lb min={lb.min():.4f}, ub max={ub.max():.4f}, "
          f"max width={float((ub - lb).max()):.4f}")

    out_dir = Path(args.out)
    log_path = out_dir / f"iid_{args.iid:03d}_step21.json"

    t0 = time.perf_counter()
    report = run_conv_body(onnx_path, lb, ub, log_path)
    wall = time.perf_counter() - t0

    print()
    print(f"[iid={args.iid}] DONE in {wall:.2f}s")
    print(f"[iid={args.iid}] reached_flatten = {report['reached_flatten']}")
    if "sparsegcz_n" in report:
        print(f"[iid={args.iid}] sparsegcz n={report['sparsegcz_n']}  "
              f"ng={report['sparsegcz_ng']}  nnz={report['sparsegcz_nnz']}")
    print(f"[iid={args.iid}] acceptance_merge_correct = "
          f"{report.get('acceptance_merge_correct')}")
    # Print last 8 per-layer entries for a quick visual check.
    print()
    print("Last 8 layers:")
    for e in report["per_layer"][-8:]:
        print(f"  step={e['step']:3d} op={e['op']:<22} "
              f"wall={e['wall_s']:7.3f}s "
              f"n_gen={e.get('num_generators', '-'):>6} "
              f"n_xi={e.get('num_unique_xi_id', '-'):>6} "
              f"max_r={e.get('max_region_numel', '-'):>6}")
    print()
    print(f"Report written to: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
