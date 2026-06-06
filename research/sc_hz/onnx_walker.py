"""ONNX → LayerOp list extractor + forward pruned propagator.

For Phase A: parse a feedforward ONNX model into a list of
`LayerOp` records (consumed by precompute_direction_chain) AND
walk the forward path applying ops on PrunedState + PRUNE between
layers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from research.sc_hz.precompute_direction import LayerOp
from research.sc_hz.prune import PrunedState, prune
from research.sc_hz import ops as scops


# ─── ONNX → LayerOp list ──────────────────────────────────────────


def parse_onnx_to_layers(model_path: str) -> Tuple[List[LayerOp],
                                                     Tuple[int, ...],
                                                     int]:
    """Walk the ONNX graph and return (layers, input_shape, n_classes).

    Phase A scope ops: Sub, Flatten, MatMul, Gemm, Add, Relu, Conv,
    AveragePool, MaxPool, BatchNormalization, Reshape.
    """
    m = onnx.load(model_path)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    in_dims = [d.dim_value if d.dim_value > 0 else 1
                for d in m.graph.input[0].type.tensor_type.shape.dim]
    # Strip leading 1's (batch)
    while in_dims and in_dims[0] == 1 and len(in_dims) > 1:
        in_dims = in_dims[1:]
    input_shape = tuple(in_dims)

    layers: List[LayerOp] = []
    # value_shapes[name] tracks the inferred output shape of each tensor
    cur_in_name = m.graph.input[0].name
    cur_shape: Tuple[int, ...] = tuple(int(x) for x in input_shape)

    def _flat(shape):
        n = 1
        for d in shape:
            n *= int(d)
        return n

    pending_matmul_W: Optional[np.ndarray] = None
    pending_matmul_out: Optional[str] = None

    for node in m.graph.node:
        op = node.op_type
        if op == "Sub":
            # x - const
            for inp in node.input:
                if inp in inits:
                    layers.append(LayerOp("sub",
                                           {"const": inits[inp].astype(np.float64).reshape(-1)}))
                    break
        elif op == "Flatten" or op == "Reshape":
            layers.append(LayerOp("flatten", {"input_shape": cur_shape}))
            cur_shape = (_flat(cur_shape),)
        elif op == "MatMul":
            # Identify weight initializer
            W = None
            for inp in node.input:
                if inp in inits:
                    W = inits[inp].astype(np.float64)
                    break
            if W is None:
                raise NotImplementedError("dynamic MatMul (no init) unsupported")
            # ONNX MatMul: y = x @ W where W has shape (in_dim, out_dim).
            # Convert to our Dense convention: y = W' @ x where W' = W.T.
            W_T = W.T  # shape (out_dim, in_dim)
            pending_matmul_W = W_T
            pending_matmul_out = node.output[0]
        elif op == "Gemm":
            # Gemm: y = alpha * A @ B + beta * C; usually alpha=1, beta=1
            A_name, B_name = node.input[0], node.input[1]
            C_name = node.input[2] if len(node.input) > 2 else None
            B = inits.get(B_name)
            if B is None:
                raise NotImplementedError("Gemm with non-init B unsupported")
            transA = 0
            transB = 0
            for a in node.attribute:
                if a.name == "transA": transA = a.i
                if a.name == "transB": transB = a.i
            W = B if transB == 1 else B.T  # bring to (out, in)
            W = W.astype(np.float64)
            bias = inits.get(C_name) if C_name else None
            params = {"W": W}
            if bias is not None:
                params["b"] = bias.astype(np.float64).reshape(-1)
            layers.append(LayerOp("dense", params))
            cur_shape = (int(W.shape[0]),)
        elif op == "Add":
            # Could be (a) bias add following MatMul, or (b) residual add
            # If pending MatMul AND one input is an init bias → fuse
            bias_init = None
            for inp in node.input:
                if inp in inits:
                    bias_init = inits[inp].astype(np.float64).reshape(-1)
                    break
            if pending_matmul_W is not None and bias_init is not None:
                # Fuse into dense
                layers.append(LayerOp("dense",
                                       {"W": pending_matmul_W, "b": bias_init}))
                cur_shape = (int(pending_matmul_W.shape[0]),)
                pending_matmul_W = None
                pending_matmul_out = None
            elif bias_init is None:
                # Residual add
                layers.append(LayerOp("add", {}))
            else:
                # Lone Add with bias (no preceding MatMul) — treat as dense identity + bias
                # Skip for Phase A
                raise NotImplementedError(f"lone Add at {node.name} with bias only")
        elif op == "Relu":
            layers.append(LayerOp("relu", {}))
        elif op == "Conv":
            # Conv: x (Ci, Hi, Wi) → y (Co, Ho, Wo)
            W_t = inits[node.input[1]].astype(np.float64)
            b_t = (inits[node.input[2]].astype(np.float64)
                   if len(node.input) > 2 else None)
            stride = 1; padding = 0; groups = 1
            for a in node.attribute:
                if a.name == "strides": stride = int(a.ints[0])
                elif a.name == "pads": padding = int(a.ints[0])
                elif a.name == "group": groups = int(a.i)
            params = {"W": W_t, "stride": stride, "padding": padding,
                       "groups": groups, "input_shape": cur_shape}
            if b_t is not None:
                params["b"] = b_t
            layers.append(LayerOp("conv2d", params))
            # Compute output shape
            Co, _, kH, kW = W_t.shape
            Ci, Hi, Wi = cur_shape
            Ho = (Hi + 2 * padding - kH) // stride + 1
            Wo = (Wi + 2 * padding - kW) // stride + 1
            cur_shape = (Co, Ho, Wo)
        elif op == "BatchNormalization":
            scale = inits[node.input[1]].astype(np.float64)
            B_bn = inits[node.input[2]].astype(np.float64)
            mean = inits[node.input[3]].astype(np.float64)
            var = inits[node.input[4]].astype(np.float64)
            eps = 1e-5
            for a in node.attribute:
                if a.name == "epsilon": eps = a.f
            inv_std = 1.0 / np.sqrt(var + eps)
            eff_scale = scale * inv_std
            eff_shift = B_bn - mean * eff_scale
            layers.append(LayerOp("bn",
                                    {"scale": eff_scale, "shift": eff_shift,
                                     "input_shape": cur_shape}))
            # Shape unchanged
        elif op == "MaxPool":
            kernel = 2; stride = 2
            for a in node.attribute:
                if a.name == "kernel_shape": kernel = int(a.ints[0])
                elif a.name == "strides": stride = int(a.ints[0])
            layers.append(LayerOp("maxpool",
                                    {"kernel_size": kernel, "stride": stride,
                                     "input_shape": cur_shape}))
            Ci, Hi, Wi = cur_shape
            Ho = (Hi - kernel) // stride + 1
            Wo = (Wi - kernel) // stride + 1
            cur_shape = (Ci, Ho, Wo)
        elif op == "AveragePool" or op == "GlobalAveragePool":
            if op == "GlobalAveragePool":
                Ci, Hi, Wi = cur_shape
                kernel = Hi
                stride = Hi
            else:
                kernel = 2; stride = 2
                for a in node.attribute:
                    if a.name == "kernel_shape": kernel = int(a.ints[0])
                    elif a.name == "strides": stride = int(a.ints[0])
            layers.append(LayerOp("avgpool",
                                    {"kernel_size": kernel, "stride": stride,
                                     "input_shape": cur_shape}))
            Ci, Hi, Wi = cur_shape
            Ho = (Hi - kernel) // stride + 1
            Wo = (Wi - kernel) // stride + 1
            cur_shape = (Ci, Ho, Wo)
        elif op == "Constant":
            continue  # initializer-like
        elif op == "Identity":
            continue
        else:
            raise NotImplementedError(
                f"ONNX op '{op}' not in Phase A scope (node {node.name})"
            )

    # If a pending MatMul without Add, flush as dense without bias
    if pending_matmul_W is not None:
        layers.append(LayerOp("dense", {"W": pending_matmul_W}))
        cur_shape = (int(pending_matmul_W.shape[0]),)

    n_classes = int(cur_shape[-1]) if cur_shape else 0
    return layers, input_shape, n_classes


# ─── Forward propagator on PrunedState ────────────────────────────


def forward_propagate(
    state: PrunedState,
    layers: List[LayerOp],
    d_per_layer: List[np.ndarray],
    K_per_layer: int,
    initial_shape: Tuple[int, ...],
) -> Tuple[PrunedState, List[Dict[str, Any]]]:
    """Walk the layer list, applying each op and PRUNE after each."""
    cur_shape: Tuple[int, ...] = tuple(int(x) for x in initial_shape)
    traces: List[Dict[str, Any]] = []

    # Initial prune on input using d_per_layer[0]
    if state.G_kept.shape[1] > K_per_layer:
        state = prune(state.c, state.G_kept, d_per_layer[0], K_per_layer,
                       return_metadata=True,
                       incoming_tail_radius=state.tail_radius)

    for i, op in enumerate(layers):
        k = op.kind
        if k == "sub":
            state = scops.apply_sub(state, op.params["const"])
        elif k == "flatten":
            # Just no-op; shape becomes flat
            state = scops.apply_flatten(state)
            cur_shape = (state.c.shape[0],)
        elif k == "dense":
            state = scops.apply_dense(state, op.params["W"],
                                        op.params.get("b"))
            cur_shape = (int(op.params["W"].shape[0]),)
        elif k == "conv2d":
            state, cur_shape = scops.apply_conv2d(
                state, op.params["W"], op.params.get("b"),
                input_shape=cur_shape,
                stride=op.params.get("stride", 1),
                padding=op.params.get("padding", 0),
                groups=op.params.get("groups", 1),
            )
        elif k == "bn":
            state = scops.apply_bn(state, op.params["scale"], op.params["shift"],
                                     input_shape=cur_shape)
        elif k == "relu":
            state, _ = scops.apply_relu_triangle(state)
        elif k == "maxpool":
            state, cur_shape = scops.apply_maxpool2d(
                state, input_shape=cur_shape,
                kernel_size=op.params.get("kernel_size", 2),
                stride=op.params.get("stride", None),
            )
        elif k == "avgpool":
            # For Phase A simplification, treat avgpool as identity-summed/scaled
            # Properly: y[h,w] = mean over window of x. This is a linear op.
            # Build it as a conv with a fixed averaging kernel.
            Ci, Hi, Wi = cur_shape
            ks = op.params["kernel_size"]
            st = op.params.get("stride", ks)
            W = np.zeros((Ci, Ci, ks, ks), dtype=np.float64)
            for c in range(Ci):
                W[c, c, :, :] = 1.0 / (ks * ks)
            state, cur_shape = scops.apply_conv2d(
                state, W, None, input_shape=cur_shape, stride=st, padding=0,
            )
        elif k == "add":
            # Add is a residual — needs two parent states. Phase A: skip
            # (treat as identity). This is unsound for residual nets;
            # raise for clarity.
            raise NotImplementedError(
                "residual Add requires multi-parent tracking; not in Phase A driver"
            )
        else:
            raise NotImplementedError(f"forward for op '{k}' not implemented")

        traces.append({
            "layer": i, "op": k, "ng": state.G_kept.shape[1],
            "tail_sum": (float(state.tail_radius.sum())
                          if state.tail_radius is not None else 0.0),
        })

        # PRUNE using d at this layer's OUTPUT
        # d_per_layer[i+1] is the cotangent at this op's OUTPUT
        if (i + 1) < len(d_per_layer) and state.G_kept.shape[1] > K_per_layer:
            d_here = d_per_layer[i + 1]
            if d_here is not None and d_here.shape[0] == state.c.shape[0]:
                state = prune(state.c, state.G_kept, d_here, K_per_layer,
                               return_metadata=True,
                               incoming_tail_radius=state.tail_radius)

    return state, traces
