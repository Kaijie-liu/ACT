"""F1 Integration: capture pre-last-ReLU state from `forward_resnet` and
solve constrained LP per rival.

Mechanism:
  1. Pre-pass ONNX graph to find the LAST Relu node by topological order.
  2. Re-walk graph: process normally, but RIGHT BEFORE applying the last
     Relu triangle, snapshot the pre-activation state into LastReluRecord.
  3. Continue forward; for ops AFTER the captured Relu, accumulate them
     into a linear transformation (W_remaining, b_remaining) that maps
     post-Relu y → final output.
  4. For each unsafe rival (d_out, threshold): solve constrained LP UB.
     If LP UB < threshold strictly → CERT for that rival.

Supported ops in post-last-Relu chain (composing W_remaining):
  Conv, BatchNormalization, Flatten, Gemm, GlobalAveragePool, AveragePool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from research.sc_hz.constrained_lp import (
    LastReluRecord, closed_form_hz_lp_ub, constrained_lp_ub,
)
from research.sc_hz.prune import PrunedState
from research.sc_hz.forward_witness import initial_state_with_lineage
from research.sc_hz.onnx_walker_resnet import (
    _initializers_dict, _node_attr_dict, _smart_add, _apply_globalavgpool,
)
from research.sc_hz.conv_streaming_prune import apply_conv2d_streaming_prune
import research.sc_hz.ops as scops


@dataclass
class ConstrainedLPWalkResult:
    """Like ResNetWalkResult but also carries last-ReLU capture + remaining linear chain."""
    output_state: PrunedState
    input_shape: Tuple[int, ...]
    output_name: str
    n_classes: int
    n_nodes_processed: int
    nodes_skipped: List[str]
    last_relu_record: Optional[LastReluRecord]
    W_remaining: Optional[np.ndarray]   # (n_classes, n_after_relu_flat)
    b_remaining: Optional[np.ndarray]   # (n_classes,)
    n_after_relu_flat: Optional[int]


def _find_last_relu_node(graph) -> Optional[str]:
    """Return the output value name of the LAST Relu node in topological order."""
    last = None
    for node in graph.node:
        if node.op_type == "Relu":
            last = node.output[0]
    return last


def _compose_gemm(W_in: np.ndarray, b_in: np.ndarray,
                    W_gemm: np.ndarray, b_gemm: Optional[np.ndarray],
                    transB: int, alpha: float = 1.0, beta: float = 1.0,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compose W_in @ x + b_in followed by Gemm.

    Gemm: y = alpha * x @ W_gemm[.T if not transB] + beta * b_gemm
    With ONNX standard: if transB=0 then we multiply by W.T (W has shape
    (in, out)); if transB=1 W has shape (out, in) and we multiply by W.
    """
    if transB == 1:
        W_use = W_gemm   # (out, in)
    else:
        W_use = W_gemm.T  # (in, out) → (out, in) via transpose
    W_use = W_use * alpha
    # y = W_use @ x + alpha (or beta) * b
    # composition: (W_use @ W_in) @ original_input + (W_use @ b_in + beta * b_gemm)
    new_W = W_use @ W_in
    if b_gemm is not None:
        new_b = W_use @ b_in + beta * b_gemm
    else:
        new_b = W_use @ b_in
    return new_W, new_b


def _compose_flatten(W_in: np.ndarray, b_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten is a no-op on a 1-D vector. Pass-through."""
    return W_in, b_in


def _compose_globalavgpool(
    W_in: np.ndarray, b_in: np.ndarray, in_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """GlobalAveragePool: (C, H, W) → (C, 1, 1). Linear via averaging."""
    C, H, W_ = in_shape
    n_in = C * H * W_
    n_out = C
    # Build averaging matrix A: (C, C*H*W) where A[c, c*H*W + h*W + w] = 1/(H*W)
    A = np.zeros((n_out, n_in), dtype=np.float64)
    for c in range(C):
        for h in range(H):
            for w in range(W_):
                A[c, c * H * W_ + h * W_ + w] = 1.0 / (H * W_)
    new_W = A @ W_in
    new_b = A @ b_in
    return new_W, new_b, (C, 1, 1)


def _compose_bn(
    W_in: np.ndarray, b_in: np.ndarray,
    scale: np.ndarray, bias: np.ndarray, mean: np.ndarray, var: np.ndarray,
    eps: float, in_shape: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose BN as per-channel scale + shift."""
    inv_std = 1.0 / np.sqrt(var + eps)
    effective_scale = scale * inv_std
    effective_shift = bias - mean * effective_scale
    if len(in_shape) == 3:
        C, H, W_ = in_shape
        # Per-channel scale repeated for each spatial position
        scale_flat = np.repeat(effective_scale, H * W_)  # (n_in,)
        shift_flat = np.repeat(effective_shift, H * W_)
    elif len(in_shape) == 1:
        scale_flat = effective_scale
        shift_flat = effective_shift
    else:
        raise NotImplementedError(f"BN compose shape {in_shape}")
    new_W = W_in * scale_flat[:, None]
    new_b = b_in * scale_flat + shift_flat
    return new_W, new_b


def _compose_conv(
    W_in: np.ndarray, b_in: np.ndarray,
    W_conv: np.ndarray, b_conv: Optional[np.ndarray],
    in_shape: Tuple[int, int, int],
    stride: int, padding: int, groups: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Compose Conv2D by applying conv to each column of W_in interpreted as
    an image. W_in has shape (n_in, n_orig); each column becomes (Ci, Hi, Wi).
    Output is W_out: (n_out, n_orig)."""
    Ci, Hi, Wi = in_shape
    n_in = Ci * Hi * Wi
    n_orig = W_in.shape[1]
    W_t = torch.from_numpy(W_conv).to(torch.float64)
    b_t = (torch.from_numpy(b_conv.astype(np.float64).reshape(-1)).to(torch.float64)
           if b_conv is not None else None)
    # Probe shape
    probe = F.conv2d(torch.zeros((1, Ci, Hi, Wi), dtype=torch.float64),
                      W_t, None, stride=stride, padding=padding, groups=groups)
    Co_p, Ho_p, Wo_p = int(probe.shape[1]), int(probe.shape[2]), int(probe.shape[3])
    n_out = Co_p * Ho_p * Wo_p
    # Reshape W_in (n_in, n_orig) → (n_orig, Ci, Hi, Wi) batched
    chunk = W_in.T.reshape(n_orig, Ci, Hi, Wi)
    chunk_t = torch.from_numpy(np.ascontiguousarray(chunk)).to(torch.float64)
    out = F.conv2d(chunk_t, W_t, None,
                     stride=stride, padding=padding, groups=groups)
    new_W = out.detach().numpy().reshape(n_orig, n_out).T  # (n_out, n_orig)
    # Bias: y = W_in @ x + b_in conv → conv(b_in_image) + conv_b
    b_image = b_in.reshape(Ci, Hi, Wi)
    b_t_image = torch.from_numpy(b_image[None, ...].copy()).to(torch.float64)
    b_conv_out = F.conv2d(b_t_image, W_t, b_t,
                             stride=stride, padding=padding, groups=groups)
    new_b = b_conv_out.detach().numpy().reshape(-1)
    return new_W, new_b, (Co_p, Ho_p, Wo_p)


def forward_resnet_capture(
    onnx_path: str, lb_x: np.ndarray, ub_x: np.ndarray,
    K_per_layer: int = 100000,
    streaming_K_target: Optional[int] = None,
    streaming_chunk_size: int = 256,
) -> ConstrainedLPWalkResult:
    """Like forward_resnet but also captures (last_relu_record, W_remaining, b_remaining)."""
    m = onnx.load(onnx_path)
    inits = _initializers_dict(m)
    # Find the TRUE data input: graph.input entries NOT in initializers
    init_names = {init.name for init in m.graph.initializer}
    data_inputs = [i for i in m.graph.input if i.name not in init_names]
    in_proto = data_inputs[0] if data_inputs else m.graph.input[0]
    in_dims = [d.dim_value if d.dim_value > 0 else 1
                for d in in_proto.type.tensor_type.shape.dim]
    in_shape = tuple(in_dims[1:]) if in_dims[0] in (0, 1) else tuple(in_dims)
    n_in = int(np.prod(in_shape))
    assert lb_x.shape == (n_in,)

    c_in = (lb_x + ub_x) / 2; r_in = (ub_x - lb_x) / 2
    init_state = initial_state_with_lineage(c_in, r_in)
    input_name = in_proto.name
    states: Dict[str, PrunedState] = {input_name: init_state}
    shapes: Dict[str, Tuple[int, ...]] = {input_name: in_shape}

    # Liveness
    use_count: Dict[str, int] = {}
    for node in m.graph.node:
        for nm in node.input:
            if nm: use_count[nm] = use_count.get(nm, 0) + 1
    for out in m.graph.output:
        use_count[out.name] = use_count.get(out.name, 0) + 1

    last_relu_out_name = _find_last_relu_node(m.graph)

    # SOUNDNESS GUARD (added 2026-06-06 per CRITICAL_F1_LP_DAG_SOUNDNESS_BUG):
    # F1 LP integration assumes a sequential network (one "last ReLU" by
    # topological order = the actual last ReLU on every input→output path).
    # For DAGs with parallel ReLU branches, that assumption is wrong and
    # F1 LP UB is unsound (12 cersyve iids falsely CERT'd by F1 LP, all
    # found to have 70-99/100 ORT sample violations).
    #
    # Detect DAG branchiness: build successor count per node output. If any
    # value has multiple consumers AND the network has parallel ReLU paths,
    # disable F1 capture and fall back to HZ closed-form only.
    out_consumer_count: Dict[str, int] = {}
    for node in m.graph.node:
        for nm in node.input:
            if nm:
                out_consumer_count[nm] = out_consumer_count.get(nm, 0) + 1
    branchy = any(c > 1 for c in out_consumer_count.values())
    # Count Relu nodes
    n_relus = sum(1 for n in m.graph.node if n.op_type == "Relu")
    # Disable F1 capture if branchy AND >1 ReLU (parallel ReLU branches likely)
    disable_f1_capture = branchy and n_relus > 1
    if disable_f1_capture:
        last_relu_out_name = None  # never capture

    # Capture variables
    last_relu_record: Optional[LastReluRecord] = None
    W_remaining: Optional[np.ndarray] = None
    b_remaining: Optional[np.ndarray] = None
    cur_post_shape: Optional[Tuple[int, ...]] = None
    capturing = False

    skipped: List[str] = []
    n_processed = 0

    def _evict_consumed(consumed):
        for nm in consumed:
            if nm == input_name: continue
            if nm in use_count:
                use_count[nm] -= 1
                if use_count[nm] <= 0 and nm in states:
                    del states[nm]
                    if nm in shapes: del shapes[nm]

    for node in m.graph.node:
        op = node.op_type
        in_names = list(node.input)
        if op == "Constant":
            arr = numpy_helper.to_array(node.attribute[0].t).astype(np.float64)
            inits[node.output[0]] = arr
            continue
        primary_in = in_names[0]
        if primary_in not in states:
            skipped.append(f"{op}({node.name}): primary input {primary_in} not in states")
            continue
        s_in = states[primary_in]; sh_in = shapes[primary_in]
        out_name = node.output[0]
        attrs = _node_attr_dict(node)

        # BEFORE processing: if this is the LAST Relu, snapshot pre-activation state
        if op == "Relu" and out_name == last_relu_out_name and not capturing:
            from research.sc_hz.ops import bounds as _bounds
            l, u = _bounds(s_in)
            last_relu_record = LastReluRecord(
                c_z=s_in.c.copy(),
                G_z=s_in.G_kept.copy(),
                tail_z=(s_in.tail_radius.copy()
                          if s_in.tail_radius is not None else None),
                l=l.copy(), u=u.copy(),
            )
            n_after = int(np.prod(sh_in))
            W_remaining = np.eye(n_after, dtype=np.float64)
            b_remaining = np.zeros(n_after, dtype=np.float64)
            cur_post_shape = sh_in
            capturing = True

        # Normal forward op
        try:
            if op == "Conv":
                W = inits[in_names[1]]
                b = inits[in_names[2]] if len(in_names) > 2 else None
                stride = attrs.get("strides", [1, 1])[0]
                padding = attrs.get("pads", [0, 0, 0, 0])[0]
                groups = attrs.get("group", 1)
                if streaming_K_target is not None:
                    s_out, out_shape = apply_conv2d_streaming_prune(
                        s_in, W, b, input_shape=sh_in,
                        stride=stride, padding=padding, groups=groups,
                        chunk_size=streaming_chunk_size,
                        K_target=streaming_K_target,
                    )
                else:
                    s_out, out_shape = scops.apply_conv2d(
                        s_in, W, b, input_shape=sh_in,
                        stride=stride, padding=padding, groups=groups,
                    )
                if capturing:
                    W_remaining, b_remaining, cur_post_shape = _compose_conv(
                        W_remaining, b_remaining, W, b, cur_post_shape,
                        stride=stride, padding=padding, groups=groups,
                    )
            elif op == "BatchNormalization":
                scale = inits[in_names[1]].astype(np.float64)
                bias = inits[in_names[2]].astype(np.float64)
                mean = inits[in_names[3]].astype(np.float64)
                var = inits[in_names[4]].astype(np.float64)
                eps = attrs.get("epsilon", 1e-5)
                inv_std = 1.0 / np.sqrt(var + eps)
                effective_scale = scale * inv_std
                effective_shift = bias - mean * effective_scale
                s_out = scops.apply_bn(s_in, effective_scale, effective_shift,
                                         input_shape=sh_in)
                out_shape = sh_in
                if capturing:
                    W_remaining, b_remaining = _compose_bn(
                        W_remaining, b_remaining,
                        scale, bias, mean, var, eps, cur_post_shape,
                    )
            elif op == "Relu":
                s_out, _ = scops.apply_relu_triangle(s_in)
                # Origin
                prev_origin = s_in.metadata.get("input_coord_origin")
                if prev_origin is not None:
                    cur_K = s_out.G_kept.shape[1]; prev_K = prev_origin.shape[0]
                    if cur_K > prev_K:
                        new_origin = np.concatenate([
                            prev_origin,
                            -np.ones(cur_K - prev_K, dtype=np.int64),
                        ])
                    else:
                        new_origin = prev_origin[:cur_K].copy()
                    s_out.metadata["input_coord_origin"] = new_origin
                out_shape = sh_in
                # Don't compose ReLU into W_remaining (nonlinear); but the
                # captured ReLU is the LAST one — any later ReLU we cannot
                # compose, so reset capturing.
                if capturing and out_name != last_relu_out_name:
                    capturing = False
                    last_relu_record = None
                    W_remaining = None; b_remaining = None
            elif op == "Add":
                # Two cases: (1) state + state (residual), (2) state + const (bias)
                if in_names[1] in inits:
                    const = inits[in_names[1]].astype(np.float64).reshape(-1)
                    if const.size == s_in.c.shape[0]:
                        shift = const
                    elif const.size == 1:
                        shift = const[0] * np.ones_like(s_in.c)
                    else:
                        skipped.append(f"Add({node.name}): const shape mismatch")
                        continue
                    s_out = PrunedState(
                        c=s_in.c + shift, G_kept=s_in.G_kept.copy(),
                        tail_radius=(s_in.tail_radius.copy()
                                      if s_in.tail_radius is not None else None),
                        metadata=dict(s_in.metadata),
                    )
                    out_shape = sh_in
                    if capturing:
                        b_remaining = b_remaining + shift
                else:
                    s_b = states.get(in_names[1])
                    if s_b is None:
                        skipped.append(f"Add({node.name}): second in {in_names[1]} not in states/inits")
                        continue
                    s_out = _smart_add(s_in, s_b)
                    out_shape = sh_in
                    if capturing:
                        # Multi-parent residual; cannot compose into single W_remaining chain
                        capturing = False
                        last_relu_record = None
                        W_remaining = None; b_remaining = None
            elif op == "Flatten":
                s_out = scops.apply_flatten(s_in)
                out_shape = (s_out.c.shape[0],)
                if capturing:
                    W_remaining, b_remaining = _compose_flatten(W_remaining, b_remaining)
                    cur_post_shape = out_shape
            elif op == "Gemm":
                W = inits[in_names[1]]
                b = inits[in_names[2]] if len(in_names) > 2 else None
                transB = attrs.get("transB", 0)
                alpha = attrs.get("alpha", 1.0)
                beta = attrs.get("beta", 1.0)
                W_eff = (W if transB else W.T) * alpha
                b_eff = b * beta if b is not None else None
                s_out = scops.apply_dense(s_in, W_eff, b_eff)
                out_shape = (int(W_eff.shape[0]),)
                if capturing:
                    W_remaining, b_remaining = _compose_gemm(
                        W_remaining, b_remaining, W, b, transB, alpha, beta,
                    )
                    cur_post_shape = out_shape
            elif op == "GlobalAveragePool":
                s_out, out_shape = _apply_globalavgpool(s_in, sh_in)
                if capturing:
                    W_remaining, b_remaining, cur_post_shape = _compose_globalavgpool(
                        W_remaining, b_remaining, cur_post_shape,
                    )
            elif op == "Reshape":
                if in_names[1] in inits:
                    target = inits[in_names[1]].astype(np.int64).tolist()
                    actual = [d for d in target if d > 0]
                    new_shape = tuple(int(d) for d in actual if d != 1) or (1,)
                    s_out = PrunedState(
                        c=s_in.c.copy(), G_kept=s_in.G_kept.copy(),
                        tail_radius=(s_in.tail_radius.copy()
                                      if s_in.tail_radius is not None else None),
                        metadata=dict(s_in.metadata),
                    )
                    out_shape = new_shape
                    if capturing:
                        cur_post_shape = new_shape
                else:
                    skipped.append(f"Reshape({node.name}): non-const target")
                    continue
            elif op == "Sub":
                # state - constant: just shift c
                if in_names[1] in inits:
                    const = inits[in_names[1]].astype(np.float64).reshape(-1)
                    if const.size == s_in.c.shape[0]:
                        shift = -const
                    elif const.size == 1:
                        shift = -const[0] * np.ones_like(s_in.c)
                    else:
                        skipped.append(f"Sub({node.name}): const shape mismatch")
                        continue
                    s_out = PrunedState(
                        c=s_in.c + shift, G_kept=s_in.G_kept.copy(),
                        tail_radius=(s_in.tail_radius.copy()
                                      if s_in.tail_radius is not None else None),
                        metadata=dict(s_in.metadata),
                    )
                    out_shape = sh_in
                    if capturing:
                        b_remaining = b_remaining + shift
                else:
                    skipped.append(f"Sub({node.name}): non-const second operand")
                    continue
            elif op == "ConvTranspose":
                # ConvTranspose: spatial upsampling Conv. For HZ propagation, the
                # output value linearly depends on input via torch's conv_transpose2d.
                # Build a custom apply_convtranspose that mirrors apply_conv2d.
                W = inits[in_names[1]]
                b = inits[in_names[2]] if len(in_names) > 2 else None
                stride = attrs.get("strides", [1, 1])[0]
                padding = attrs.get("pads", [0, 0, 0, 0])[0]
                output_padding = attrs.get("output_padding", [0, 0])[0]
                groups = attrs.get("group", 1)
                # Reshape state to (C, H, W) per generator
                Ci, Hi, Wi = sh_in
                n_in_ct = Ci * Hi * Wi
                K_ct = s_in.G_kept.shape[1]
                W_t = torch.from_numpy(W).to(torch.float64)
                b_t = (torch.from_numpy(b.astype(np.float64).reshape(-1)).to(torch.float64)
                        if b is not None else None)
                # Apply to center
                c_image = torch.from_numpy(s_in.c.reshape(1, Ci, Hi, Wi)).to(torch.float64)
                c_out_image = F.conv_transpose2d(c_image, W_t, b_t,
                                                       stride=stride, padding=padding,
                                                       output_padding=output_padding,
                                                       groups=groups)
                Co, Ho, Wo = (int(c_out_image.shape[1]),
                                int(c_out_image.shape[2]),
                                int(c_out_image.shape[3]))
                new_c = c_out_image.detach().numpy().reshape(-1)
                # Apply to generators (chunked)
                n_out_ct = Co * Ho * Wo
                new_G = np.zeros((n_out_ct, K_ct), dtype=np.float64)
                chunk = 64
                for kk in range(0, K_ct, chunk):
                    G_chunk = s_in.G_kept[:, kk:kk+chunk].T.reshape(-1, Ci, Hi, Wi)
                    G_t = torch.from_numpy(np.ascontiguousarray(G_chunk)).to(torch.float64)
                    G_out = F.conv_transpose2d(G_t, W_t, None,
                                                     stride=stride, padding=padding,
                                                     output_padding=output_padding,
                                                     groups=groups)
                    new_G[:, kk:kk+chunk] = G_out.detach().numpy().reshape(G_t.shape[0], -1).T
                # Tail: also apply abs(W) to abs(tail)
                if s_in.tail_radius is not None:
                    tail_image = torch.from_numpy(s_in.tail_radius.reshape(1, Ci, Hi, Wi)).to(torch.float64)
                    W_abs = torch.from_numpy(np.abs(W)).to(torch.float64)
                    new_tail = F.conv_transpose2d(tail_image, W_abs, None,
                                                         stride=stride, padding=padding,
                                                         output_padding=output_padding,
                                                         groups=groups).detach().numpy().reshape(-1)
                else:
                    new_tail = None
                s_out = PrunedState(c=new_c, G_kept=new_G,
                                        tail_radius=new_tail,
                                        metadata=dict(s_in.metadata))
                out_shape = (Co, Ho, Wo)
                # Don't compose ConvTranspose into W_remaining easily; abort capture
                if capturing:
                    capturing = False
                    last_relu_record = None
                    W_remaining = None; b_remaining = None
            elif op == "Unsqueeze":
                # Unsqueeze: add a dim. State is 1D (flat); shape is logical.
                # For our walker which stores state as flat vector, unsqueeze
                # is essentially a no-op except shape change.
                s_out = PrunedState(c=s_in.c.copy(), G_kept=s_in.G_kept.copy(),
                                        tail_radius=(s_in.tail_radius.copy()
                                                      if s_in.tail_radius is not None else None),
                                        metadata=dict(s_in.metadata))
                # Update shape: insert 1 at axis position(s)
                if len(in_names) >= 2 and in_names[1] in inits:
                    axes = inits[in_names[1]].astype(np.int64).reshape(-1).tolist()
                else:
                    axes = attrs.get("axes", [0])
                new_shape = list(sh_in)
                for ax in sorted(axes):
                    if ax < 0: ax = len(new_shape) + ax + 1
                    new_shape.insert(ax, 1)
                # Remove leading 1 if it's the batch dim
                while new_shape and new_shape[0] == 1 and len(new_shape) > 1:
                    new_shape = new_shape[1:]
                out_shape = tuple(new_shape) if new_shape else (1,)
                if capturing:
                    cur_post_shape = out_shape
            elif op == "Squeeze":
                # Squeeze: remove dims of size 1. State unchanged.
                s_out = PrunedState(c=s_in.c.copy(), G_kept=s_in.G_kept.copy(),
                                        tail_radius=(s_in.tail_radius.copy()
                                                      if s_in.tail_radius is not None else None),
                                        metadata=dict(s_in.metadata))
                out_shape = tuple(d for d in sh_in if d != 1) or (1,)
                if capturing:
                    cur_post_shape = out_shape
            elif op == "Transpose":
                # Permute axes: for flat vector with shape, just update shape.
                # NOTE: state.c is flat, so permutation requires re-ordering elements.
                perm = attrs.get("perm")
                if perm is None:
                    skipped.append(f"Transpose({node.name}): no perm attr")
                    continue
                perm = tuple(perm)
                # Trim leading batch dim if present
                if perm[0] == 0:
                    perm_trim = tuple(p - 1 for p in perm[1:])
                else:
                    perm_trim = perm
                full_shape = sh_in
                if len(perm_trim) != len(full_shape):
                    skipped.append(f"Transpose({node.name}): perm dim mismatch")
                    continue
                n_total = int(np.prod(full_shape))
                # Reindex: build permutation map
                old_idx = np.arange(n_total).reshape(full_shape)
                new_idx = np.transpose(old_idx, perm_trim).reshape(-1)
                s_out = PrunedState(
                    c=s_in.c[new_idx],
                    G_kept=s_in.G_kept[new_idx, :],
                    tail_radius=(s_in.tail_radius[new_idx]
                                  if s_in.tail_radius is not None else None),
                    metadata=dict(s_in.metadata),
                )
                new_shape = tuple(full_shape[p] for p in perm_trim)
                out_shape = new_shape
                if capturing:
                    cur_post_shape = new_shape
            elif op == "Split":
                # Split: divide along axis. ONNX outputs multiple tensors;
                # one node, multiple outputs. State currently 1D flat.
                # Read split sizes from input[1] (init) or attribute.
                axis = attrs.get("axis", 0)
                if axis < 0: axis = len(sh_in) + axis
                # batch dim strip — common case axis=1 with sh_in[0]=batch_remaining
                if axis >= len(sh_in):
                    axis = axis - 1
                if axis < 0 or axis >= len(sh_in):
                    skipped.append(f"Split({node.name}): axis {axis} invalid for {sh_in}")
                    continue
                split_sizes = None
                if len(in_names) >= 2 and in_names[1] in inits:
                    split_sizes = inits[in_names[1]].astype(np.int64).reshape(-1).tolist()
                elif "split" in attrs:
                    split_sizes = list(attrs["split"])
                else:
                    n_outputs = len(node.output)
                    if sh_in[axis] % n_outputs == 0:
                        split_sizes = [sh_in[axis] // n_outputs] * n_outputs
                    else:
                        skipped.append(f"Split({node.name}): cannot infer split sizes")
                        continue
                # Compute index slices for each split
                full_shape = sh_in
                n_total = int(np.prod(full_shape))
                full_idx = np.arange(n_total).reshape(full_shape)
                offset = 0
                for k, sz in enumerate(split_sizes):
                    if k >= len(node.output):
                        break
                    sl = [slice(None)] * len(full_shape)
                    sl[axis] = slice(offset, offset + sz)
                    keep_idx = full_idx[tuple(sl)].reshape(-1)
                    new_shape = list(full_shape)
                    new_shape[axis] = sz
                    new_shape = tuple(new_shape)
                    s_k = PrunedState(
                        c=s_in.c[keep_idx],
                        G_kept=s_in.G_kept[keep_idx, :],
                        tail_radius=(s_in.tail_radius[keep_idx]
                                      if s_in.tail_radius is not None else None),
                        metadata=dict(s_in.metadata),
                    )
                    states[node.output[k]] = s_k
                    shapes[node.output[k]] = new_shape
                    offset += sz
                # No standard s_out; loop body's standard write would override
                # — we already wrote all outputs in the loop.
                # Capture: abort F1 capture since Split fans out (multi-output)
                if capturing:
                    capturing = False
                    last_relu_record = None
                    W_remaining = None; b_remaining = None
                n_processed += 1
                _evict_consumed(in_names)
                continue  # skip the standard write below
            elif op == "Gather":
                # Gather: pick indices along axis.
                axis = attrs.get("axis", 0)
                if in_names[1] not in inits:
                    skipped.append(f"Gather({node.name}): non-const indices")
                    continue
                indices = inits[in_names[1]].astype(np.int64).reshape(-1)
                if axis < 0: axis = len(sh_in) + axis
                if axis >= len(sh_in):
                    axis = axis - 1
                if axis < 0 or axis >= len(sh_in):
                    skipped.append(f"Gather({node.name}): axis {axis} invalid for {sh_in}")
                    continue
                full_shape = sh_in
                n_total = int(np.prod(full_shape))
                full_idx = np.arange(n_total).reshape(full_shape)
                sl = [slice(None)] * len(full_shape)
                sl[axis] = indices
                keep_idx = full_idx[tuple(sl)].reshape(-1)
                new_shape = list(full_shape)
                new_shape[axis] = len(indices)
                new_shape = tuple(new_shape)
                s_out = PrunedState(
                    c=s_in.c[keep_idx],
                    G_kept=s_in.G_kept[keep_idx, :],
                    tail_radius=(s_in.tail_radius[keep_idx]
                                  if s_in.tail_radius is not None else None),
                    metadata=dict(s_in.metadata),
                )
                out_shape = new_shape
                if capturing:
                    W_remaining = W_remaining[keep_idx, :]
                    b_remaining = b_remaining[keep_idx]
                    cur_post_shape = new_shape
            elif op == "Slice":
                # Slice with constant starts/ends/axes (and optional steps)
                # Inputs after data: starts, ends, axes, steps (all initializers)
                def _safe_int64(arr):
                    """Cast to int64 safely, treating any value larger than 2^30 as
                    'slice-to-end' sentinel (clamped to 2^30 so downstream clamps work)."""
                    arr = np.asarray(arr).reshape(-1)
                    if arr.dtype == np.int64:
                        # Convert any INT64_MAX-like sentinel to 2^30 for safe arithmetic
                        BIG = 1 << 30
                        return np.where(np.abs(arr) > BIG,
                                            np.sign(arr).astype(np.int64) * BIG, arr)
                    BIG = 1 << 30
                    arr_f = arr.astype(np.float64)
                    arr_f = np.where(np.abs(arr_f) > BIG, np.sign(arr_f) * BIG, arr_f)
                    return arr_f.astype(np.int64)
                try:
                    starts = _safe_int64(inits[in_names[1]])
                    ends = _safe_int64(inits[in_names[2]])
                    axes = (_safe_int64(inits[in_names[3]])
                             if len(in_names) > 3 and in_names[3] in inits
                             else np.arange(len(starts)))
                    steps = (_safe_int64(inits[in_names[4]])
                              if len(in_names) > 4 and in_names[4] in inits
                              else np.ones(len(starts), dtype=np.int64))
                except KeyError:
                    skipped.append(f"Slice({node.name}): non-const slice params")
                    continue
                if not np.all(steps == 1):
                    skipped.append(f"Slice({node.name}): non-unit steps not supported")
                    continue
                # Slice indices into the flat representation
                # For now: only support slicing along the LAST or single dim
                # state is 1D (flat); apply via index mask
                if len(sh_in) == 0 or sh_in == (1,):
                    # Trivial pass-through
                    s_out = PrunedState(c=s_in.c.copy(),
                                            G_kept=s_in.G_kept.copy(),
                                            tail_radius=(s_in.tail_radius.copy()
                                                          if s_in.tail_radius is not None else None),
                                            metadata=dict(s_in.metadata))
                    out_shape = sh_in
                elif len(starts) == 1:
                    # Single-axis slice
                    ax = int(axes[0]) if axes[0] >= 0 else len(sh_in) + int(axes[0])
                    # Batch dim may have been stripped — if ax >= len(sh_in), shift down by 1
                    if ax >= len(sh_in):
                        ax = ax - 1
                    if ax < 0 or ax >= len(sh_in):
                        skipped.append(f"Slice({node.name}): axis {axes[0]} invalid for shape {sh_in}")
                        continue
                    st = int(starts[0])
                    # ends may be INT64_MAX (slice to end)
                    en_raw = int(ends[0])
                    if en_raw > sh_in[ax]:
                        en = sh_in[ax]
                    elif en_raw < 0:
                        en = sh_in[ax] + en_raw
                    else:
                        en = en_raw
                    if st < 0: st = max(0, sh_in[ax] + st)
                    if en < 0: en = sh_in[ax] + en
                    en = min(en, sh_in[ax])
                    # Build index mask on flat array
                    full_shape = sh_in
                    n_total = int(np.prod(full_shape))
                    full_idx = np.arange(n_total).reshape(full_shape)
                    sl = [slice(None)] * len(full_shape)
                    sl[ax] = slice(st, en)
                    keep_idx = full_idx[tuple(sl)].reshape(-1)
                    new_shape = list(full_shape)
                    new_shape[ax] = en - st
                    new_shape = tuple(new_shape)
                    s_out = PrunedState(
                        c=s_in.c[keep_idx],
                        G_kept=s_in.G_kept[keep_idx, :],
                        tail_radius=(s_in.tail_radius[keep_idx]
                                      if s_in.tail_radius is not None else None),
                        metadata=dict(s_in.metadata),
                    )
                    out_shape = new_shape
                    if capturing:
                        W_remaining = W_remaining[keep_idx, :]
                        b_remaining = b_remaining[keep_idx]
                        cur_post_shape = new_shape
                else:
                    skipped.append(f"Slice({node.name}): multi-axis slice not supported")
                    continue
            elif op == "Concat":
                # Concat along given axis; all inputs must be state-like
                axis = attrs.get("axis", 0)
                states_to_concat = []
                shapes_to_concat = []
                ok = True
                for nm in in_names:
                    if nm in states:
                        states_to_concat.append(states[nm])
                        shapes_to_concat.append(shapes[nm])
                    elif nm in inits:
                        # Constant input — convert to PrunedState
                        arr = inits[nm].astype(np.float64).reshape(-1)
                        states_to_concat.append(PrunedState(
                            c=arr, G_kept=np.zeros((arr.shape[0],
                                                       s_in.G_kept.shape[1])),
                            tail_radius=None,
                            metadata={},
                        ))
                        shapes_to_concat.append((arr.shape[0],))
                    else:
                        ok = False
                        skipped.append(f"Concat({node.name}): input {nm} not in states/inits")
                        break
                if not ok:
                    continue
                new_c = np.concatenate([s.c for s in states_to_concat], axis=0)
                # Align G dimensions: pad smaller ng with zeros so all states
                # have the same generator count. This is sound because adding
                # zero generators doesn't expand the set.
                max_ng = max(s.G_kept.shape[1] for s in states_to_concat)
                padded_Gs = []
                for s in states_to_concat:
                    if s.G_kept.shape[1] < max_ng:
                        pad = np.zeros((s.G_kept.shape[0],
                                            max_ng - s.G_kept.shape[1]))
                        padded_Gs.append(np.concatenate([s.G_kept, pad], axis=1))
                    else:
                        padded_Gs.append(s.G_kept)
                new_G = np.concatenate(padded_Gs, axis=0)
                tails = [s.tail_radius for s in states_to_concat]
                if any(t is not None for t in tails):
                    new_tail = np.concatenate([
                        t if t is not None else np.zeros(states_to_concat[i].c.shape[0])
                        for i, t in enumerate(tails)
                    ], axis=0)
                else:
                    new_tail = None
                s_out = PrunedState(c=new_c, G_kept=new_G,
                                        tail_radius=new_tail,
                                        metadata=dict(s_in.metadata))
                # Compute output shape (just flat concat for now)
                out_shape = (sum(np.prod(sh) for sh in shapes_to_concat),)
                if capturing:
                    capturing = False
                    last_relu_record = None
                    W_remaining = None
                    b_remaining = None
            elif op == "Dropout":
                # Inference mode: identity
                s_out = PrunedState(c=s_in.c.copy(), G_kept=s_in.G_kept.copy(),
                                        tail_radius=(s_in.tail_radius.copy()
                                                      if s_in.tail_radius is not None else None),
                                        metadata=dict(s_in.metadata))
                out_shape = sh_in
                # W_remaining unchanged (identity)
            elif op == "Sigmoid":
                # Sigmoid triangle relaxation per neuron
                # σ(z) ≈ slope*z + bias + slack*aux, aux ∈ [-1, 1]
                # For unstable: use chord from σ(l) to σ(u); slope = (σ(u)-σ(l))/(u-l)
                #               midpoint contains both ends; aux captures gap
                # For stable (l > 4 or u < -4): nearly constant → use point estimate
                rad = np.abs(s_in.G_kept).sum(axis=1)
                if s_in.tail_radius is not None:
                    rad = rad + s_in.tail_radius
                l = s_in.c - rad
                u = s_in.c + rad
                sig_l = 1.0 / (1.0 + np.exp(-l))
                sig_u = 1.0 / (1.0 + np.exp(-u))
                # Chord slope and intercept
                den = np.maximum(u - l, 1e-300)
                slope = (sig_u - sig_l) / den
                intercept = sig_l - slope * l
                # Max gap = max |σ(z) - chord(z)| approximated by half max - min
                # Conservative: slack mu = (σ_max_on_interval - chord_at_some_z) / 2
                # For monotonic σ: max gap is bounded by (σ(u) - σ(l)) / 4 at midpoint
                mu = np.abs(sig_u - sig_l) / 4.0
                new_c = slope * s_in.c + intercept + 0  # midpoint chord (gap mu)
                new_G = slope[:, None] * s_in.G_kept
                # Add slack column per coord
                slack_cols = np.diag(mu)
                if np.any(mu > 1e-12):
                    new_G_ext = np.concatenate([new_G, slack_cols], axis=1)
                else:
                    new_G_ext = new_G
                new_tail = (slope * s_in.tail_radius
                              if s_in.tail_radius is not None else None)
                s_out = PrunedState(c=new_c, G_kept=new_G_ext,
                                        tail_radius=new_tail,
                                        metadata=dict(s_in.metadata))
                # Don't compose Sigmoid into W_remaining (nonlinear); abort capture
                if capturing:
                    capturing = False
                    last_relu_record = None
                    W_remaining = None; b_remaining = None
                out_shape = sh_in
            elif op == "Mul":
                # state * constant: scale c, G, tail elementwise
                if in_names[1] in inits:
                    const = inits[in_names[1]].astype(np.float64).reshape(-1)
                    if const.size == s_in.c.shape[0]:
                        scale = const
                    elif const.size == 1:
                        scale = const[0] * np.ones_like(s_in.c)
                    else:
                        skipped.append(f"Mul({node.name}): const shape mismatch n={const.size} vs state {s_in.c.shape[0]}")
                        continue
                    new_c = s_in.c * scale
                    new_G = s_in.G_kept * scale[:, None]
                    new_tail = (np.abs(scale) * s_in.tail_radius
                                  if s_in.tail_radius is not None else None)
                    s_out = PrunedState(c=new_c, G_kept=new_G,
                                            tail_radius=new_tail,
                                            metadata=dict(s_in.metadata))
                    out_shape = sh_in
                    if capturing:
                        # Mul is diag scaling — equivalent to A = diag(scale)
                        W_remaining = scale[:, None] * W_remaining
                        b_remaining = b_remaining * scale
                else:
                    skipped.append(f"Mul({node.name}): non-const second operand")
                    continue
            elif op == "Div":
                # state / constant: scale by 1/constant
                if in_names[1] in inits:
                    const = inits[in_names[1]].astype(np.float64).reshape(-1)
                    if np.any(np.abs(const) < 1e-30):
                        skipped.append(f"Div({node.name}): divide by zero")
                        continue
                    if const.size == s_in.c.shape[0]:
                        scale = 1.0 / const
                    elif const.size == 1:
                        scale = (1.0 / const[0]) * np.ones_like(s_in.c)
                    else:
                        skipped.append(f"Div({node.name}): const shape mismatch")
                        continue
                    new_c = s_in.c * scale
                    new_G = s_in.G_kept * scale[:, None]
                    new_tail = (np.abs(scale) * s_in.tail_radius
                                  if s_in.tail_radius is not None else None)
                    s_out = PrunedState(c=new_c, G_kept=new_G,
                                            tail_radius=new_tail,
                                            metadata=dict(s_in.metadata))
                    out_shape = sh_in
                    if capturing:
                        W_remaining = scale[:, None] * W_remaining
                        b_remaining = b_remaining * scale
                else:
                    skipped.append(f"Div({node.name}): non-const second operand")
                    continue
            elif op == "MatMul":
                W = inits.get(in_names[1])
                if W is None:
                    skipped.append(f"MatMul({node.name}): non-const W")
                    continue
                W = W.astype(np.float64)
                # ONNX MatMul: y = x @ W where x is (batch, in), W is (in, out)
                # For our 1-D state c (in,), x @ W = c @ W (out,)
                # apply_dense expects W_eff of shape (out, in)
                W_eff = W.T
                s_out = scops.apply_dense(s_in, W_eff, None)
                out_shape = (int(W_eff.shape[0]),)
                if capturing:
                    W_remaining = W_eff @ W_remaining
                    b_remaining = W_eff @ b_remaining
                    cur_post_shape = out_shape
            else:
                skipped.append(f"{op}({node.name}): not implemented")
                if capturing:
                    # Cannot compose this op; abort capture
                    capturing = False
                    last_relu_record = None
                    W_remaining = None; b_remaining = None
                continue
        except Exception as e:
            skipped.append(f"{op}({node.name}): {type(e).__name__}: {str(e)[:80]}")
            if capturing:
                capturing = False
                last_relu_record = None
                W_remaining = None; b_remaining = None
            continue

        states[out_name] = s_out
        shapes[out_name] = out_shape
        n_processed += 1
        _evict_consumed(in_names)

    out_name = m.graph.output[0].name
    if out_name not in states:
        raise RuntimeError(f"Output {out_name} not reached. Skipped: {skipped}")
    out_dims = [d.dim_value if d.dim_value > 0 else 1
                 for d in m.graph.output[0].type.tensor_type.shape.dim]
    n_classes = int(np.prod(out_dims[1:]) if len(out_dims) > 1 else out_dims[0])

    n_after_relu_flat = None
    if last_relu_record is not None:
        n_after_relu_flat = int(last_relu_record.c_z.shape[0])

    return ConstrainedLPWalkResult(
        output_state=states[out_name], input_shape=in_shape,
        output_name=out_name, n_classes=n_classes,
        n_nodes_processed=n_processed, nodes_skipped=skipped,
        last_relu_record=last_relu_record,
        W_remaining=W_remaining, b_remaining=b_remaining,
        n_after_relu_flat=n_after_relu_flat,
    )
