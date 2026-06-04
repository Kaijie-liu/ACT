"""ImageHZ Step 2.2 — single-iid tail LP + margin comparison.

Per advisor 2026-06-03 phased plan:
- Extend Step 2.1: walk the dense tail (Gemm → ReLU triangle → Gemm)
  on the SparseGcZ produced at the Flatten bridge.
- Compute ImageHZ upper bound on (Y[rival] - Y[true]) (the "lp_unsafe"
  metric) and compare to:
    (a) the existing HZ baseline atlas number (atlas_v2_bucketed.json)
    (b) the ORT replay number (already in atlas)

A KEY simplification: at the Flatten bridge the SparseGcZ has
``Ac/b/eq_mask`` empty. Both ``apply_linear`` and the DeepZ triangle
ReLU add generators but NO constraints, so the final SparseGcZ remains
unconstrained. The "LP" therefore has a closed form:

    max_{xi in [-1,+1]^ng}  (Gc[r,:] - Gc[t,:]) @ xi  =  ||Gc[r,:] - Gc[t,:]||_1

so ``margin_diff_max = (c[r] - c[t]) + ||Gc[r,:] - Gc[t,:]||_1``.

No LP solver needed for Step 2.2 — the closed-form is exact for the
empty-constraint case. Step 2.3+ may revisit if we add LP-style
constraints.

NOT in scope: V/A verdict change, batch sentinels, production wiring.

Usage:
    python research/imagehz_cifar_step22.py --iid 11
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
from research.imagehz_cifar_prototype import (  # noqa: E402
    INSTANCES_CSV,
    CIFAR_BENCH,
    load_instance,
    load_input_box,
    run_conv_body,
)

ATLAS_PATH = (
    "/data1/Kane/ACT/audit_results/cifar_unknown_margin_atlas_20260603/"
    "atlas_v2_bucketed.json"
)


# ── SparseGcZ tail ops ────────────────────────────────────────────


def apply_linear_sparsegcz(
    hz: SparseGcZ, weight: torch.Tensor, bias: Optional[torch.Tensor],
) -> SparseGcZ:
    """Apply y = W @ x + b on SparseGcZ.

    Both center and Gc transform by left-mul with W. Bias adds to
    center only. Constraints (Ac/b/eq_mask) are unchanged (they
    operate on xi-space, which is untouched by a linear map on x).
    """
    W = weight.to(dtype=hz.dtype, device=hz.device)
    n_in = int(hz.c.numel())
    if W.shape[1] != n_in:
        raise ValueError(f"linear: W cols {W.shape[1]} != input dim {n_in}")
    n_out = int(W.shape[0])

    # Center: y_c = W @ c + b
    c_out = W @ hz.c.view(-1)
    if bias is not None:
        c_out = c_out + bias.to(dtype=hz.dtype, device=hz.device).view(-1)

    # Gc: y_Gc = W @ Gc_sparse  (dense @ sparse via torch.sparse.mm
    # with W.T). torch.sparse.mm expects (sparse, dense), returns dense;
    # we want W @ Gc_sparse where Gc_sparse is (n_in, ng).
    # Equivalent: (Gc_sparse.T @ W.T).T, but easier to densify the
    # output since n_out << n_in usually (here 100 << 2048) and ng is
    # small enough.
    Gc_dense_out = torch.sparse.mm(hz.Gc_sparse.T, W.T).T  # (n_out, ng)
    # Convert back to sparse.
    nz_mask = Gc_dense_out.abs() > 0
    if nz_mask.any():
        nz = nz_mask.nonzero(as_tuple=False)
        rows = nz[:, 0]
        cols = nz[:, 1]
        vals = Gc_dense_out[rows, cols]
        indices = torch.stack([rows, cols], dim=0)
    else:
        indices = torch.zeros((2, 0), dtype=torch.long, device=hz.device)
        vals = torch.zeros((0,), dtype=hz.dtype, device=hz.device)
    Gc_out = torch.sparse_coo_tensor(
        indices, vals, (n_out, hz.ng),
        dtype=hz.dtype, device=hz.device,
    ).coalesce()

    # Ac/b/eq_mask carry over as-is (xi-space same shape).
    return SparseGcZ(
        c=c_out,
        Gc_sparse=Gc_out,
        Ac_sparse=hz.Ac_sparse,
        b=hz.b,
        eq_mask=hz.eq_mask,
        dtype=hz.dtype,
        device=hz.device,
    )


# ── Margin computation ───────────────────────────────────────────


def margin_upper_bound(
    hz: SparseGcZ, y_true: int, y_rival: int,
) -> float:
    """Closed-form max of (Y[rival] - Y[true]) on unconstrained
    SparseGcZ:

        max = (c[r] - c[t]) + ||Gc[r,:] - Gc[t,:]||_1

    Requires hz.nc == 0 (no Ac constraints) and hz.nb == 0.
    """
    if hz.nc != 0 or hz.nb != 0:
        raise ValueError(
            f"closed-form margin requires nc=0, nb=0 (got nc={hz.nc}, nb={hz.nb})"
        )
    c = hz.c.view(-1)
    # Materialize the two rows of Gc as dense vectors of length ng.
    ng = int(hz.ng)
    Gc_row_r = torch.zeros(ng, dtype=hz.dtype, device=hz.device)
    Gc_row_t = torch.zeros(ng, dtype=hz.dtype, device=hz.device)
    ind = hz.Gc_sparse.indices()
    val = hz.Gc_sparse.values()
    if val.numel() > 0:
        sel_r = ind[0] == y_rival
        sel_t = ind[0] == y_true
        Gc_row_r.index_add_(0, ind[1][sel_r], val[sel_r])
        Gc_row_t.index_add_(0, ind[1][sel_t], val[sel_t])
    diff_const = float(c[y_rival].item() - c[y_true].item())
    diff_gen_l1 = float((Gc_row_r - Gc_row_t).abs().sum().item())
    return diff_const + diff_gen_l1


def worst_rival_upper_bound(
    hz: SparseGcZ, y_true: int, n_classes: int,
) -> Tuple[int, float, List[Tuple[int, float]]]:
    """Find the rival with the largest upper bound on Y[r] - Y[t]."""
    ng = int(hz.ng)
    c = hz.c.view(-1)
    # Vector form: for each row r, compute ||Gc[r,:] - Gc[t,:]||_1.
    # Densify Gc to (n_classes, ng); it's small (100 x ng).
    Gc_dense = hz.Gc_sparse.to_dense()
    diff_const = c - c[y_true]
    diff_gen_l1 = (Gc_dense - Gc_dense[y_true:y_true + 1, :]).abs().sum(dim=1)
    margin_max = diff_const + diff_gen_l1
    margin_max[y_true] = float("-inf")  # exclude self
    worst_r = int(torch.argmax(margin_max).item())
    worst_v = float(margin_max[worst_r].item())
    rivals_sorted = sorted(
        [(int(i), float(margin_max[i].item()))
         for i in range(n_classes) if i != y_true],
        key=lambda x: -x[1],
    )
    return worst_r, worst_v, rivals_sorted


# ── ORT independent ground truth ─────────────────────────────────


def ort_replay_min_margin(
    onnx_path: str,
    lb: np.ndarray, ub: np.ndarray,
    y_true: int, y_rival: int,
    n_samples: int = 5000,
    seed: int = 0,
) -> float:
    """Sample inputs uniformly in the box, run through ORT, return
    max observed (logit[y_rival] - logit[y_true]).
    """
    import onnxruntime as ort
    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"],
    )
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    # Symbolic / zero / None dims become batch=1 marker; rest are literal ints
    def _resolve_dim(d):
        if d is None:
            return 1
        if isinstance(d, str):
            return 1  # symbolic, e.g. "batch_size"
        if isinstance(d, int) and d <= 0:
            return 1
        return int(d)
    inp_shape = [_resolve_dim(d) for d in inp_shape]

    rng = np.random.default_rng(seed)
    n_in = int(lb.size)
    samples = rng.uniform(low=lb, high=ub, size=(n_samples, n_in)).astype(np.float32)
    # Reshape into ONNX input
    samples_r = samples.reshape(n_samples, *inp_shape[1:])

    best = -np.inf
    # Batch through to avoid OOM
    BS = 256
    for i in range(0, n_samples, BS):
        x = samples_r[i:i + BS]
        out = sess.run(None, {inp_name: x})[0]  # (B, 100)
        diff = out[:, y_rival] - out[:, y_true]
        best = max(best, float(diff.max()))
    return float(best)


# ── Driver ────────────────────────────────────────────────────────


def step22_one_iid(
    iid: int, out_dir: Path, n_ort_samples: int = 5000,
) -> Dict[str, Any]:
    onnx_path, vnn_path = load_instance(iid)
    print(f"[iid={iid}] onnx={onnx_path}")
    print(f"[iid={iid}] vnnlib={vnn_path}")

    # Resolve y_true from atlas (saves reading vnnlib output rows).
    with open(ATLAS_PATH) as f:
        atlas = json.load(f)
    atlas_entry = next((e for e in atlas if e.get("iid") == iid), None)
    if atlas_entry is None:
        raise RuntimeError(f"iid {iid} not in atlas")
    y_true = int(atlas_entry["y_true"])
    atlas_worst_rival = int(atlas_entry["worst_rival"])
    atlas_lp_unsafe = float(atlas_entry["lp_unsafe"])
    atlas_ort_unsafe = float(atlas_entry["ort_unsafe"])
    print(f"[iid={iid}] atlas: y_true={y_true} worst_rival={atlas_worst_rival} "
          f"lp_unsafe={atlas_lp_unsafe:.4f} ort_unsafe={atlas_ort_unsafe:.4f}")

    # Load input box.
    m_onnx = onnx.load(onnx_path)
    in_dims = [d.dim_value for d in m_onnx.graph.input[0].type.tensor_type.shape.dim]
    n_in = int(in_dims[1]) * int(in_dims[2]) * int(in_dims[3])
    lb, ub = load_input_box(vnn_path, n_in)

    # Step 2.1: walk conv body, get SparseGcZ at bridge.
    print(f"[iid={iid}] Step 2.1 conv-body walk...")
    t_walk0 = time.perf_counter()
    log_dummy = out_dir / f"iid_{iid:03d}_step21_logs.json"
    body_report = run_conv_body(onnx_path, lb, ub, log_dummy)
    t_walk = time.perf_counter() - t_walk0
    print(f"[iid={iid}]   conv body: {t_walk:.2f}s  "
          f"-> SparseGcZ(n={body_report['sparsegcz_n']}, "
          f"ng={body_report['sparsegcz_ng']})")

    # Rebuild the SparseGcZ for the tail (run_conv_body discards it;
    # easier to re-walk than refactor in this prototype).
    # Quick rebuild by reusing the same code path...
    from research.imagehz_cifar_prototype import (  # noqa: E402
        _build_initial_hz, _initializer_map, _apply_channel_affine,
        _fold_bn, _bn_eps, _conv_attrs,
    )
    from act.back_end.imagehz import (  # noqa: E402
        apply_add, apply_conv2d, apply_relu_triangle, flatten_to_sparsegcz,
    )

    inits = _initializer_map(m_onnx)
    input_name = m_onnx.graph.input[0].name
    in_C, in_H, in_W = int(in_dims[1]), int(in_dims[2]), int(in_dims[3])
    hz0, next_xi = _build_initial_hz(lb, ub, (in_C, in_H, in_W))
    activations = {input_name: hz0}

    bridge_hz = None
    flatten_in_tensor = None
    for node in m_onnx.graph.node:
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
            x = activations[ins[0]]
            flatten_in_tensor = ins[0]
            bridge_hz = flatten_to_sparsegcz(x)
            activations[outs[0]] = bridge_hz
            break

    assert bridge_hz is not None
    print(f"[iid={iid}]   bridge SparseGcZ: nc={bridge_hz.nc}, "
          f"nb={bridge_hz.nb} (must be 0 for closed-form margin)")

    # Tail walk on SparseGcZ.
    t_tail0 = time.perf_counter()
    sg = bridge_hz
    tail_per_layer: List[Dict[str, Any]] = []
    for node in m_onnx.graph.node:
        if node.name in {"Gemm_56", "Relu_57", "Gemm_58"}:
            op = node.op_type
            ins = list(node.input)
            t0 = time.perf_counter()
            if op == "Gemm":
                w = torch.from_numpy(inits[ins[1]].astype(np.float64))
                b = torch.from_numpy(inits[ins[2]].astype(np.float64))
                sg = apply_linear_sparsegcz(sg, w, b)
            elif op == "Relu":
                sg = sg.apply_relu_triangle()
            tail_per_layer.append({
                "node": node.name,
                "op": op,
                "wall_s": time.perf_counter() - t0,
                "n": sg.n,
                "ng": sg.ng,
                "nnz": int(sg.Gc_sparse._nnz()),
                "nc": sg.nc,
                "nb": sg.nb,
            })
    t_tail = time.perf_counter() - t_tail0
    print(f"[iid={iid}]   tail walk: {t_tail:.3f}s -> final "
          f"(n={sg.n}, ng={sg.ng}, nnz={int(sg.Gc_sparse._nnz())})")
    for e in tail_per_layer:
        print(f"      {e['node']:<10} {e['op']:<6} "
              f"wall={e['wall_s']:.3f}s  ng={e['ng']:>5} nnz={e['nnz']:>7}")

    # Compute the ImageHZ margin upper bound vs all rivals.
    n_classes = int(sg.n)
    t_m0 = time.perf_counter()
    worst_r, worst_v, rivals_sorted = worst_rival_upper_bound(
        sg, y_true=y_true, n_classes=n_classes,
    )
    t_m = time.perf_counter() - t_m0
    print(f"[iid={iid}]   margin closed-form: {t_m:.3f}s")
    print(f"[iid={iid}]   ImageHZ worst rival = {worst_r}  "
          f"upper bound on Y[r]-Y[t] = {worst_v:+.4f}")

    # Also evaluate the SAME rival the atlas picked.
    img_at_atlas_rival = margin_upper_bound(sg, y_true, atlas_worst_rival)
    print(f"[iid={iid}]   ImageHZ at atlas_rival ({atlas_worst_rival}) = "
          f"{img_at_atlas_rival:+.4f}")

    # ORT replay.
    print(f"[iid={iid}]   ORT replay ({n_ort_samples} samples)...")
    t_o0 = time.perf_counter()
    ort_at_worst_r = ort_replay_min_margin(
        onnx_path, lb, ub, y_true=y_true, y_rival=worst_r,
        n_samples=n_ort_samples, seed=0,
    )
    ort_at_atlas_r = (
        ort_replay_min_margin(
            onnx_path, lb, ub, y_true=y_true, y_rival=atlas_worst_rival,
            n_samples=n_ort_samples, seed=0,
        ) if worst_r != atlas_worst_rival else ort_at_worst_r
    )
    t_o = time.perf_counter() - t_o0
    print(f"[iid={iid}]   ORT (worst_r={worst_r}) = {ort_at_worst_r:+.4f}  "
          f"({t_o:.2f}s)")

    report = {
        "iid": iid,
        "y_true": y_true,
        "atlas_worst_rival": atlas_worst_rival,
        "atlas_lp_unsafe": atlas_lp_unsafe,
        "atlas_ort_unsafe": atlas_ort_unsafe,
        "imagehz_worst_rival": worst_r,
        "imagehz_lp_unsafe_at_worst": worst_v,
        "imagehz_lp_unsafe_at_atlas_rival": img_at_atlas_rival,
        "ort_at_imagehz_worst_r": ort_at_worst_r,
        "ort_at_atlas_worst_r": ort_at_atlas_r,
        "top5_rivals_imagehz": rivals_sorted[:5],
        "bridge_sparsegcz": {
            "n": int(bridge_hz.n), "ng": int(bridge_hz.ng),
            "nnz": int(bridge_hz.Gc_sparse._nnz()),
        },
        "final_sparsegcz": {
            "n": int(sg.n), "ng": int(sg.ng),
            "nnz": int(sg.Gc_sparse._nnz()),
        },
        "wall_conv_body_s": t_walk,
        "wall_tail_s": t_tail,
        "wall_margin_s": t_m,
        "wall_ort_s": t_o,
        "n_ort_samples": n_ort_samples,
        "tail_per_layer": tail_per_layer,
        # Verdict shorthand.
        "verdict_imagehz": "CERT" if worst_v < 0 else "UNKNOWN",
        "delta_vs_atlas_lp_unsafe": worst_v - atlas_lp_unsafe,
        # Phantom: ImageHZ over-approximation vs ORT.
        "imagehz_phantom_gap_at_worst": worst_v - ort_at_worst_r,
    }
    out = out_dir / f"iid_{iid:03d}_step22.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(f"[iid={iid}] === Step 2.2 summary ===")
    print(f"   atlas baseline  : lp_unsafe = {atlas_lp_unsafe:+.4f}  ort_unsafe = {atlas_ort_unsafe:+.4f}")
    print(f"   ImageHZ         : lp_unsafe = {worst_v:+.4f}  (Δ vs atlas = {worst_v - atlas_lp_unsafe:+.4f})")
    print(f"   ImageHZ phantom : {worst_v - ort_at_worst_r:+.4f} (vs atlas phantom {atlas_lp_unsafe - atlas_ort_unsafe:+.4f})")
    print(f"   verdict (ImageHZ): {report['verdict_imagehz']}")
    print(f"   report: {out}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid", type=int, default=11)
    ap.add_argument("--out", type=str,
                    default="audit_results/cifar_unknown_margin_atlas_20260603/PHASE2_STEP22")
    ap.add_argument("--n-ort-samples", type=int, default=5000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    step22_one_iid(args.iid, out_dir, n_ort_samples=args.n_ort_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
