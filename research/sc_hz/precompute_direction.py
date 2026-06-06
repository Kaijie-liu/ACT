"""Pre-compute the per-rival direction d_L^r through arbitrary ONNX layers.

Now supports Dense, Conv2D, BN, AvgPool, Add (residual), Flatten, Sub
(constant preprocessing). MaxPool is supported in the FORWARD path but
its adjoint is identity-on-the-winner (only valid when stable winner
identified — handled by setting a per-position mask).

The chain is given as a list of `LayerOp` records. Each LayerOp has:
  - kind:  "dense" | "conv2d" | "bn" | "relu" | "maxpool" | "avgpool"
           | "add" | "flatten" | "sub" | "reshape"
  - params: dict of per-op parameters

The d_chain is built FROM the output direction BACKWARD through these
ops, in reverse order, applying the corresponding adjoint.

Forward order: layer[0] is applied first; layer[-1] is the output classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LayerOp:
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


# ─── Adjoint dispatch ─────────────────────────────────────────────


def _adjoint(op: LayerOp, d_out: np.ndarray,
              out_shape: Optional[Tuple[int, ...]] = None) -> Tuple[np.ndarray, Optional[Tuple]]:
    """Return (d_in, in_shape) — the cotangent at this op's input.

    d_out is the cotangent at this op's OUTPUT (flat). out_shape is the
    output's image shape (C, H, W) when relevant (for conv/maxpool/etc).
    """
    k = op.kind
    if k == "dense":
        W = op.params["W"]                       # (out_dim, in_dim)
        return W.T @ d_out, None
    if k == "conv2d":
        W = op.params["W"]                       # (Co, Ci, kH, kW)
        stride = op.params.get("stride", 1)
        padding = op.params.get("padding", 0)
        groups = op.params.get("groups", 1)
        in_shape = op.params["input_shape"]       # (Ci, Hi, Wi)
        Co, Ci_per_group, kH, kW = W.shape
        Ho, Wo = out_shape[1], out_shape[2]
        d_out_4d = torch.from_numpy(
            d_out.reshape(1, Co, Ho, Wo)
        ).to(torch.float64)
        W_t = torch.from_numpy(W).to(torch.float64)
        d_in_4d = F.conv_transpose2d(d_out_4d, W_t, None,
                                       stride=stride, padding=padding,
                                       groups=groups)
        # Crop/pad to match in_shape if conv_transpose's output differs
        Ci, Hi, Wi = in_shape
        d_in_arr = d_in_4d.detach().numpy()[0]
        # Truncate to Hi/Wi (in some stride configs conv_transpose
        # over-produces by 1 row/col)
        d_in_arr = d_in_arr[:, :Hi, :Wi]
        return d_in_arr.reshape(-1), in_shape
    if k == "bn":
        # y = scale * x + shift   (per-channel, applied after Conv)
        # d_in = scale * d_out (per channel, broadcast over spatial)
        scale = op.params["scale"]               # (C,)
        # out_shape is (C, H, W); reshape d_out to broadcast
        C = scale.shape[0]
        if out_shape is None or len(out_shape) != 3:
            # Treat as per-coordinate scaling
            return d_out * scale, None
        H, W = out_shape[1], out_shape[2]
        d_out_4d = d_out.reshape(C, H, W)
        d_in_4d = d_out_4d * scale[:, None, None]
        return d_in_4d.reshape(-1), out_shape
    if k == "relu":
        # The adjoint of forward ReLU at a pre-activation z is multiplication
        # by an "approximate slope" per coordinate. For our heuristic d_L,
        # we use the linearization slope: for stable active -> 1, stable
        # inactive -> 0, unstable -> u/(u-l). But at d_L precompute time
        # we don't have bounds. Use IDENTITY (slope=1 everywhere). Soundness
        # is independent of d_L; this just affects pruning quality.
        return d_out, out_shape
    if k == "add":
        # y = x_a + x_b → d_in_a = d_in_b = d_out (cotangent broadcasts)
        return d_out, out_shape
    if k == "flatten" or k == "reshape":
        # Shape change only
        in_shape = op.params.get("input_shape")
        return d_out, in_shape
    if k == "sub":
        # y = x - const → d_in = d_out
        return d_out, out_shape
    if k == "maxpool":
        # MaxPool adjoint is identity on the winner. At precompute time we
        # use a "uniform spread" approximation: d_in[winner_or_any] = d_out
        # over the pool window. For per-rival pruning, this is a heuristic;
        # soundness of the FORWARD HZ does not depend on it.
        Ci, Hi, Wi = op.params["input_shape"]
        kernel = op.params.get("kernel_size", 2)
        stride = op.params.get("stride", kernel)
        Ho = (Hi - kernel) // stride + 1
        Wo = (Wi - kernel) // stride + 1
        d_out_3d = d_out.reshape(Ci, Ho, Wo)
        # Spread uniformly: d_in[c, h*s..h*s+k, w*s..w*s+k] += d_out[c,h,w] / k^2
        d_in = np.zeros((Ci, Hi, Wi), dtype=np.float64)
        for h in range(Ho):
            for w in range(Wo):
                d_in[:, h*stride:h*stride+kernel,
                       w*stride:w*stride+kernel] += d_out_3d[:, h:h+1, w:w+1] / (kernel * kernel)
        return d_in.reshape(-1), (Ci, Hi, Wi)
    if k == "avgpool":
        Ci, Hi, Wi = op.params["input_shape"]
        kernel = op.params.get("kernel_size", 2)
        stride = op.params.get("stride", kernel)
        Ho = (Hi - kernel) // stride + 1
        Wo = (Wi - kernel) // stride + 1
        d_out_3d = d_out.reshape(Ci, Ho, Wo)
        d_in = np.zeros((Ci, Hi, Wi), dtype=np.float64)
        for h in range(Ho):
            for w in range(Wo):
                d_in[:, h*stride:h*stride+kernel,
                       w*stride:w*stride+kernel] += d_out_3d[:, h:h+1, w:w+1] / (kernel * kernel)
        return d_in.reshape(-1), (Ci, Hi, Wi)
    raise NotImplementedError(f"adjoint for op kind '{k}' not yet implemented")


def precompute_d_per_layer_chain(
    layers: Sequence[LayerOp],
    rival_direction_at_output: np.ndarray,
    layer_output_shapes: Sequence[Optional[Tuple[int, ...]]],
) -> List[np.ndarray]:
    """Compute d_L^r for a general layer chain.

    Args:
      layers: forward-ordered list of LayerOp. layers[0] applies first.
      rival_direction_at_output: (n_classes,) vector — e_r - e_{y_t}.
      layer_output_shapes: per-layer output shape (for conv etc); None if irrelevant.

    Returns:
      List of (N+1) arrays: d_per_layer[i] is the cotangent at layer i's
      INPUT (so d_per_layer[0] is at the model input). The LAST entry
      d_per_layer[N] is at the output of the entire chain (just == rival_direction_at_output).
    """
    N = len(layers)
    # d_at_output[N] = rival_direction_at_output (in output space)
    d_per_layer: List[np.ndarray] = [None] * (N + 1)
    d_per_layer[N] = rival_direction_at_output.astype(np.float64)
    # Walk backward through layers
    d_cur = d_per_layer[N]
    cur_shape = layer_output_shapes[N - 1] if N > 0 else None
    for i in range(N - 1, -1, -1):
        op = layers[i]
        out_shape_for_this_op = cur_shape
        d_in, in_shape = _adjoint(op, d_cur, out_shape_for_this_op)
        d_per_layer[i] = d_in
        d_cur = d_in
        # Determine input shape for the NEXT (earlier) op
        cur_shape = in_shape if in_shape is not None else (
            layer_output_shapes[i - 1] if i > 0 else None
        )
    return d_per_layer


# ─── Backward-compat wrapper for Dense-only nets ───────────────────


def precompute_d_per_layer(
    weights: Sequence[np.ndarray],
    rival: int,
    y_true: int,
) -> List[np.ndarray]:
    """Legacy Dense-only wrapper (kept for the original test).

    Returns [d_0, ..., d_N] where d_L lives in space of layer L's input.
    For Dense-only: d_N = W_{N+1}[rival] - W_{N+1}[y_true], etc.
    """
    if len(weights) < 1:
        raise ValueError("weights must have at least one entry")
    W_clf = np.asarray(weights[-1], dtype=np.float64)
    if W_clf.ndim != 2:
        raise NotImplementedError("classifier must be 2-D Dense")
    n_classes = W_clf.shape[0]
    if not (0 <= rival < n_classes) or not (0 <= y_true < n_classes):
        raise ValueError("rival or y_true out of range")
    if rival == y_true:
        raise ValueError("rival must differ from y_true")
    d_N = (W_clf[rival, :] - W_clf[y_true, :]).astype(np.float64)
    d_chain: List[np.ndarray] = [d_N]
    for W_l in reversed(list(weights[:-1])):
        W_l = np.asarray(W_l, dtype=np.float64)
        d_chain.append(W_l.T @ d_chain[-1])
    d_chain.reverse()
    return d_chain
