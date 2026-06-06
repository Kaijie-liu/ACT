"""Value-DAG ONNX walker for ResNet-style topologies (Add residual + conv body).

Per advisor 2026-06-05 Phase D directive: extend SC-HZ forward HZ + forward-
coefficient sidecar to dense-conv benchmarks (cifar100 / tinyimagenet / yolo /
traffic). The current onnx_walker.py walks a linear layer list and raises
NotImplementedError on Add; we need to track HZ state per VALUE NAME and
handle multi-input ops.

Design:
  - One pass: walk ONNX graph in topological order (model.graph.node is
    already topologically sorted by ONNX convention).
  - Maintain states: dict[value_name → PrunedState].
  - Maintain shapes: dict[value_name → tuple[int, ...]].
  - For multi-input ops (Add), look up both inputs and combine.
  - For input-coord lineage: track origin metadata through the DAG;
    when Add merges two states, sum their input-coord generator columns
    (they refer to the same xi_k) and concatenate non-input-coord
    generators (independent ReLU slacks etc).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from research.sc_hz.prune import PrunedState
from research.sc_hz.forward_witness import initial_state_with_lineage
import research.sc_hz.ops as scops


def _initializers_dict(m: onnx.ModelProto) -> Dict[str, np.ndarray]:
    return {t.name: numpy_helper.to_array(t).astype(np.float64)
             for t in m.graph.initializer}


def _node_attr_dict(node) -> Dict[str, Any]:
    d = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            d[a.name] = int(a.i)
        elif a.type == onnx.AttributeProto.INTS:
            d[a.name] = [int(x) for x in a.ints]
        elif a.type == onnx.AttributeProto.FLOAT:
            d[a.name] = float(a.f)
        elif a.type == onnx.AttributeProto.FLOATS:
            d[a.name] = [float(x) for x in a.floats]
        elif a.type == onnx.AttributeProto.STRING:
            d[a.name] = a.s.decode("utf-8") if isinstance(a.s, bytes) else a.s
        elif a.type == onnx.AttributeProto.TENSOR:
            d[a.name] = numpy_helper.to_array(a.t)
    return d


def _smart_add(state_a: PrunedState, state_b: PrunedState) -> PrunedState:
    """Residual Add of two PrunedStates with lineage-aware generator merging.

    Inputs derive from the SAME input box, so columns with the same
    input_coord_origin index refer to the same xi_k. For those columns,
    G_y[:, k] = G_a[:, k] + G_b[:, k] (linearity of xi-parameterization).

    For columns with origin -1 (e.g. ReLU slack), the two branches'
    slacks are INDEPENDENT variables → concatenate.

    The tail_radius (independent interval) terms simply add per-coordinate.
    """
    assert state_a.c.shape == state_b.c.shape, \
        f"Add shape mismatch: {state_a.c.shape} vs {state_b.c.shape}"
    new_c = state_a.c + state_b.c

    origin_a = state_a.metadata.get("input_coord_origin",
                                       -np.ones(state_a.G_kept.shape[1], dtype=np.int64))
    origin_b = state_b.metadata.get("input_coord_origin",
                                       -np.ones(state_b.G_kept.shape[1], dtype=np.int64))

    # Build merged input-coord generators
    n_in_max = max(int(origin_a.max() if origin_a.size > 0 else -1),
                    int(origin_b.max() if origin_b.size > 0 else -1)) + 1
    G_input_coord_cols: List[np.ndarray] = []
    new_input_coord_origin_list: List[int] = []
    n = state_a.c.shape[0]
    for coord in range(n_in_max):
        ka = np.where(origin_a == coord)[0]
        kb = np.where(origin_b == coord)[0]
        if ka.size == 0 and kb.size == 0:
            continue
        col_sum = np.zeros(n, dtype=np.float64)
        if ka.size > 0:
            col_sum += state_a.G_kept[:, ka].sum(axis=1)
        if kb.size > 0:
            col_sum += state_b.G_kept[:, kb].sum(axis=1)
        G_input_coord_cols.append(col_sum)
        new_input_coord_origin_list.append(coord)

    # Non-input-coord cols (slacks): concatenate (independent variables)
    nonic_a = np.where(origin_a == -1)[0]
    nonic_b = np.where(origin_b == -1)[0]
    G_slack_cols: List[np.ndarray] = []
    if nonic_a.size > 0:
        G_slack_cols.append(state_a.G_kept[:, nonic_a])
    if nonic_b.size > 0:
        G_slack_cols.append(state_b.G_kept[:, nonic_b])

    # Assemble G
    parts = []
    if G_input_coord_cols:
        parts.append(np.column_stack(G_input_coord_cols))
    if G_slack_cols:
        parts.append(np.concatenate(G_slack_cols, axis=1))
    if parts:
        new_G = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    else:
        new_G = np.zeros((n, 0), dtype=np.float64)

    new_origin = np.concatenate([
        np.array(new_input_coord_origin_list, dtype=np.int64),
        -np.ones(sum(c.shape[1] for c in G_slack_cols), dtype=np.int64),
    ])

    new_tail: Optional[np.ndarray] = None
    if state_a.tail_radius is not None and state_b.tail_radius is not None:
        new_tail = state_a.tail_radius + state_b.tail_radius
    elif state_a.tail_radius is not None:
        new_tail = state_a.tail_radius.copy()
    elif state_b.tail_radius is not None:
        new_tail = state_b.tail_radius.copy()

    return PrunedState(
        c=new_c, G_kept=new_G, tail_radius=new_tail,
        metadata={"input_coord_origin": new_origin},
    )


def _apply_globalavgpool(state: PrunedState, input_shape: Tuple[int, int, int]
                          ) -> Tuple[PrunedState, Tuple[int, int, int]]:
    """GlobalAveragePool: (C, H, W) → (C, 1, 1). Mean over spatial dims.

    Linear op: y_c = mean_{h, w} x_{c, h, w}.
    """
    C, H, W = input_shape
    flat = C * H * W
    assert state.c.shape == (flat,), \
        f"GlobalAvgPool expects flat input ({flat},), got {state.c.shape}"

    # Build linear operator W_op of shape (C, flat) where W_op[c, c*H*W + h*W + w] = 1/(H*W)
    rec = state.c.reshape(C, H, W).mean(axis=(1, 2))  # (C,)
    Gn = state.G_kept.shape[1]
    G_new = np.zeros((C, Gn), dtype=np.float64)
    G_re = state.G_kept.reshape(C, H, W, Gn)
    G_new[:, :] = G_re.mean(axis=(1, 2))
    tail_new = None
    if state.tail_radius is not None:
        t_re = state.tail_radius.reshape(C, H, W)
        # |W_op| @ tail in fully connected form: per-channel mean of absolute
        # row-radii is the per-channel bound after averaging — but actually
        # for tail (independent box per coord), the average is
        #   tail_c <= (1/(H*W)) * sum_{h,w} tail_{c,h,w}   (independent → conservative)
        tail_new = t_re.mean(axis=(1, 2))

    return PrunedState(
        c=rec, G_kept=G_new, tail_radius=tail_new,
        metadata=dict(state.metadata),
    ), (C, 1, 1)


@dataclass
class ResNetWalkResult:
    output_state: PrunedState
    input_shape: Tuple[int, ...]
    output_name: str
    n_classes: int
    n_nodes_processed: int
    nodes_skipped: List[str] = field(default_factory=list)


def forward_resnet(onnx_path: str, lb_x: np.ndarray, ub_x: np.ndarray,
                    K_per_layer: int = 100000,
                    streaming_K_target: Optional[int] = None,
                    streaming_chunk_size: int = 256) -> ResNetWalkResult:
    """Forward propagate a ResNet-style ONNX model into a PrunedState.

    Args:
      onnx_path: path to .onnx
      lb_x, ub_x: input box bounds, shape (n_in,)
      K_per_layer: prune budget per layer (default ∞ = no prune)
      streaming_K_target: if set, every Conv op uses
        `apply_conv2d_streaming_prune` with this K_target instead of the
        dense `apply_conv2d`. This bounds the resident generator matrix
        per layer to (n_out × K_target). Root-coord generators are
        prioritized over ReLU slacks; dropped columns fold into tail.
        None = use dense apply_conv2d (current behavior).
      streaming_chunk_size: chunk size for streaming-prune Conv. Only
        relevant when streaming_K_target is set.

    Returns ResNetWalkResult with output_state at the graph output.
    """
    m = onnx.load(onnx_path)
    inits = _initializers_dict(m)

    # Input setup
    in_proto = m.graph.input[0]
    in_dims = [d.dim_value if d.dim_value > 0 else 1
                for d in in_proto.type.tensor_type.shape.dim]
    in_shape = tuple(in_dims[1:]) if in_dims[0] in (0, 1) else tuple(in_dims)
    n_in = int(np.prod(in_shape))
    assert lb_x.shape == (n_in,), f"lb_x shape {lb_x.shape} vs n_in {n_in}"

    c_in = (lb_x + ub_x) / 2
    r_in = (ub_x - lb_x) / 2
    init_state = initial_state_with_lineage(c_in, r_in)

    input_name = in_proto.name
    states: Dict[str, PrunedState] = {input_name: init_state}
    shapes: Dict[str, Tuple[int, ...]] = {input_name: in_shape}

    # Liveness analysis: count how many remaining nodes consume each value.
    # After a node is processed, decrement use_count of its inputs; when 0,
    # free the state from `states` dict to bound memory.
    use_count: Dict[str, int] = {}
    for node in m.graph.node:
        for in_name in node.input:
            if in_name:
                use_count[in_name] = use_count.get(in_name, 0) + 1
    # Graph outputs must survive: bump their count so they never get evicted
    for out in m.graph.output:
        use_count[out.name] = use_count.get(out.name, 0) + 1

    skipped: List[str] = []
    n_processed = 0

    def _evict_consumed(consumed_names):
        """After a node uses these input names, decrement use_count and
        del the state from `states`/`shapes` when count reaches zero.
        The input ONNX value (initial input) is never evicted (its
        use_count was bumped by the loop above OR it's never consumed).
        """
        for in_name in consumed_names:
            if in_name == input_name:
                continue
            if in_name in use_count:
                use_count[in_name] -= 1
                if use_count[in_name] <= 0 and in_name in states:
                    del states[in_name]
                    if in_name in shapes:
                        del shapes[in_name]

    for node in m.graph.node:
        op = node.op_type
        in_names = list(node.input)
        # Constant ops define initializers; skip
        if op == "Constant":
            arr = numpy_helper.to_array(node.attribute[0].t).astype(np.float64)
            inits[node.output[0]] = arr
            continue
        primary_in = in_names[0]
        if primary_in not in states:
            # Op references a name that's an initializer (e.g. Reshape target)
            # or we don't have the state — skip and record
            skipped.append(f"{op}({node.name}): primary input {primary_in} not in states")
            continue
        s_in = states[primary_in]
        sh_in = shapes[primary_in]
        out_name = node.output[0]
        attrs = _node_attr_dict(node)

        if op == "Conv":
            W = inits[in_names[1]]
            b = inits[in_names[2]] if len(in_names) > 2 else None
            stride = attrs.get("strides", [1, 1])[0]
            padding = attrs.get("pads", [0, 0, 0, 0])[0]
            groups = attrs.get("group", 1)
            if streaming_K_target is not None:
                from research.sc_hz.conv_streaming_prune import (
                    apply_conv2d_streaming_prune,
                )
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
            states[out_name] = s_out
            shapes[out_name] = out_shape
            n_processed += 1
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
            states[out_name] = s_out
            shapes[out_name] = sh_in
            n_processed += 1
        elif op == "Relu":
            s_out, _ = scops.apply_relu_triangle(s_in)
            # Origin: existing cols preserve origin; new cols appended → -1
            prev_origin = s_in.metadata.get("input_coord_origin")
            if prev_origin is not None:
                cur_K = s_out.G_kept.shape[1]
                prev_K = prev_origin.shape[0]
                if cur_K > prev_K:
                    new_origin = np.concatenate([
                        prev_origin,
                        -np.ones(cur_K - prev_K, dtype=np.int64),
                    ])
                else:
                    new_origin = prev_origin[:cur_K].copy()
                s_out.metadata["input_coord_origin"] = new_origin
            states[out_name] = s_out
            shapes[out_name] = sh_in
            n_processed += 1
        elif op == "Add":
            s_b = states.get(in_names[1])
            if s_b is None:
                skipped.append(f"Add({node.name}): second input {in_names[1]} not in states")
                continue
            s_out = _smart_add(s_in, s_b)
            states[out_name] = s_out
            shapes[out_name] = sh_in
            n_processed += 1
        elif op == "Flatten":
            s_out = scops.apply_flatten(s_in)
            states[out_name] = s_out
            shapes[out_name] = (s_out.c.shape[0],)
            n_processed += 1
        elif op == "Gemm":
            W = inits[in_names[1]]
            b = inits[in_names[2]] if len(in_names) > 2 else None
            transA = attrs.get("transA", 0)
            transB = attrs.get("transB", 0)
            alpha = attrs.get("alpha", 1.0)
            beta = attrs.get("beta", 1.0)
            W_eff = (W if transB else W.T) * alpha
            b_eff = b * beta if b is not None else None
            assert not transA, "transA Gemm not supported"
            s_out = scops.apply_dense(s_in, W_eff, b_eff)
            states[out_name] = s_out
            shapes[out_name] = (int(W_eff.shape[0]),)
            n_processed += 1
        elif op == "GlobalAveragePool":
            s_out, out_shape = _apply_globalavgpool(s_in, sh_in)
            states[out_name] = s_out
            shapes[out_name] = out_shape
            n_processed += 1
        elif op == "MaxPool":
            ks = attrs.get("kernel_shape", [2, 2])[0]
            st = attrs.get("strides", [ks, ks])[0]
            s_out, out_shape = scops.apply_maxpool2d(
                s_in, input_shape=sh_in, kernel_size=ks, stride=st,
            )
            states[out_name] = s_out
            shapes[out_name] = out_shape
            n_processed += 1
        elif op == "AveragePool":
            ks = attrs.get("kernel_shape", [2, 2])[0]
            st = attrs.get("strides", [ks, ks])[0]
            # Build avg kernel and use conv
            C, H, W_ = sh_in
            kernel = np.zeros((C, 1, ks, ks), dtype=np.float64)
            kernel[:, 0, :, :] = 1.0 / (ks * ks)
            s_out, out_shape = scops.apply_conv2d(
                s_in, kernel, None, input_shape=sh_in,
                stride=st, padding=0, groups=C,
            )
            states[out_name] = s_out
            shapes[out_name] = out_shape
            n_processed += 1
        elif op == "Reshape":
            # If reshape target is constant initializer, follow it.
            if in_names[1] in inits:
                target = inits[in_names[1]].astype(np.int64).tolist()
                # ONNX reshape: -1 means infer; 0 means use input dim
                # Skip the batch dim (usually first), use remaining
                actual = [d for d in target if d > 0]
                if len(actual) == 0:
                    actual = [s_in.c.shape[0]]
                # Build new shape excluding leading 1s
                new_shape = tuple(int(d) for d in actual if d != 1) or (1,)
                # Just update shape; c/G are flat
                states[out_name] = PrunedState(
                    c=s_in.c.copy(), G_kept=s_in.G_kept.copy(),
                    tail_radius=(s_in.tail_radius.copy()
                                  if s_in.tail_radius is not None else None),
                    metadata=dict(s_in.metadata),
                )
                shapes[out_name] = new_shape
                n_processed += 1
            else:
                skipped.append(f"Reshape({node.name}): non-constant target")
                continue
        else:
            skipped.append(f"{op}({node.name}): not implemented")
            continue

        # Value liveness eviction: after a successful op, decrement
        # use_count for each input value. When count reaches 0, the
        # state can be freed.
        _evict_consumed(in_names)

    # Output
    out_name = m.graph.output[0].name
    if out_name not in states:
        raise RuntimeError(f"Output {out_name} not reached. Skipped: {skipped}")
    # Determine n_classes
    out_dims = [d.dim_value if d.dim_value > 0 else 1
                 for d in m.graph.output[0].type.tensor_type.shape.dim]
    n_classes = int(np.prod(out_dims[1:]) if len(out_dims) > 1 else out_dims[0])
    return ResNetWalkResult(
        output_state=states[out_name], input_shape=in_shape,
        output_name=out_name, n_classes=n_classes,
        n_nodes_processed=n_processed, nodes_skipped=skipped,
    )
