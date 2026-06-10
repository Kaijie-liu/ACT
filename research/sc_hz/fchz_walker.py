"""FCHZState walker for ONNX (minimal: sequential MLPs).

Produces FCHZState with all SlackRecord layers populated.
This unlocks multi-layer MILP application on real ONNX benchmarks.

Supported ops (sequential MLP path):
  Gemm, MatMul, Add (bias), Relu, Flatten, Reshape, Identity
  Squeeze, Unsqueeze, Constant (initializer ingestion)

For now: NOT Conv, NOT residual Add, NOT Slice/Concat/Split DAG.
Just sequential dense networks (acasxu, linearizenn main chain,
dist_shift, safenlp, malbeware, sat_relu).
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, '/data1/Kane/ACT')
import numpy as np
import onnx
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from research.sc_hz.fc_hz_state import (
    FCHZState, SlackRecord, initial_state, apply_dense,
    apply_relu_triangle_with_record, compress_g_to_tail,
)


@dataclass
class FCHZWalkResult:
    state: FCHZState
    n_processed: int
    n_skipped: int
    skipped_ops: List[str]
    output_name: str


def forward_fchz(onnx_path: str, lb: np.ndarray, ub: np.ndarray,
                    hz_only: bool = False,
                    G_max_cols: Optional[int] = None) -> FCHZWalkResult:
    """Walk through ONNX building FCHZState. Sequential MLP only.

    G_max_cols: if set, after each Conv/Dense compress G to at most this many
    columns (drop smallest L∞-norm columns into per-row tail_radius). Sound,
    enables deep CNN like tinyimagenet. See SPARSE_SLACK_DESIGN.md.
    """
    m = onnx.load(onnx_path)
    inits = {init.name: onnx.numpy_helper.to_array(init).astype(np.float64)
             for init in m.graph.initializer}
    init_names = set(inits.keys())

    # Identify data input
    din_list = [x for x in m.graph.input if x.name not in init_names]
    if not din_list:
        raise RuntimeError("no data input found")
    din = din_list[0]
    dims = [d.dim_value if d.dim_value > 0 else 1
            for d in din.type.tensor_type.shape.dim]
    n_in = int(np.prod(dims[1:])) if dims[0] in (0, 1) else int(np.prod(dims))

    # Build initial state from input bounds
    c_in = (lb + ub) / 2
    r_in = (ub - lb) / 2
    state = initial_state(c_in, r_in)

    # Track states per output name; constants stored in inits
    states: Dict[str, FCHZState] = {din.name: state}
    # Track shapes (C, H, W) for state tensors with spatial layout
    shapes: Dict[str, Tuple[int, ...]] = {}
    # Initial input shape (strip batch dim)
    if len(dims) >= 4 and dims[0] in (0, 1):
        shapes[din.name] = tuple(dims[1:])
    elif len(dims) >= 3:
        shapes[din.name] = tuple(dims)
    elif len(dims) >= 2:
        # 2D input: keep batch dim for ops like Slice that may reference axes
        shapes[din.name] = tuple(dims)
    elif len(dims) == 1:
        shapes[din.name] = tuple(dims)
    # Constant value cache (intermediate Constant nodes)
    const_cache: Dict[str, np.ndarray] = {}

    n_processed = 0
    n_skipped = 0
    skipped_ops = []
    layer_index = 0  # for ReLU records

    def get_const(name):
        if name in inits: return inits[name]
        if name in const_cache: return const_cache[name]
        return None

    out_name = m.graph.output[0].name

    # Pre-compute consumer count per output name for eviction.
    consumer_count: Dict[str, int] = {}
    for node in m.graph.node:
        for nm in node.input:
            if nm:
                consumer_count[nm] = consumer_count.get(nm, 0) + 1
    # Output of the graph must not be evicted.
    consumer_count[out_name] = consumer_count.get(out_name, 0) + 1

    def _decref_inputs(in_names_iter):
        """After processing a node, decrement consumer count for its inputs.
        If count reaches 0, free the state from `states` dict."""
        for nm in in_names_iter:
            if not nm: continue
            if nm in consumer_count:
                consumer_count[nm] -= 1
                if consumer_count[nm] <= 0:
                    if nm in states and nm != din.name:
                        del states[nm]
                    if nm in shapes and nm != din.name:
                        del shapes[nm]

    for node in m.graph.node:
        op = node.op_type
        in_names = list(node.input)

        # === Handle Constant nodes ===
        if op == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    arr = onnx.numpy_helper.to_array(a.t).astype(np.float64)
                    const_cache[node.output[0]] = arr
                    break
            n_processed += 1
            continue

        # Identify primary state input
        primary = None
        for nm in in_names:
            if nm in states:
                primary = nm
                break

        try:
            if op in ("Identity", "Flatten"):
                # No-op on flat state — keep state, mark as 1D shape.
                if primary is None:
                    skipped_ops.append(f"{op}({node.name}): no state input")
                    n_skipped += 1
                    continue
                states[node.output[0]] = states[primary]
                # Flatten gives 1D, Identity inherits
                if op == "Flatten":
                    shapes[node.output[0]] = (states[primary].c.shape[0],)
                elif primary in shapes:
                    shapes[node.output[0]] = shapes[primary]
            elif op == "Gemm":
                # Standard: y = alpha * x @ W^[T] + beta * b
                if primary is None:
                    skipped_ops.append(f"Gemm({node.name}): no state input")
                    n_skipped += 1
                    continue
                W_arr = get_const(in_names[1])
                if W_arr is None:
                    skipped_ops.append(f"Gemm({node.name}): non-const W")
                    n_skipped += 1
                    continue
                b_arr = get_const(in_names[2]) if len(in_names) >= 3 else None
                attrs = {a.name: a for a in node.attribute}
                transB = attrs["transB"].i if "transB" in attrs else 0
                alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
                beta = attrs["beta"].f if "beta" in attrs else 1.0
                W = W_arr.copy()
                if not transB:
                    W = W.T
                W = W * alpha
                b = b_arr * beta if b_arr is not None else None
                new_state = apply_dense(states[primary], W, b)
                states[node.output[0]] = new_state
            elif op == "MatMul":
                # x @ W (W is (in, out)); handle 1D state as well as 2D-batched state.
                if primary is None:
                    skipped_ops.append(f"MatMul({node.name}): no state input")
                    n_skipped += 1
                    continue
                W_arr = get_const(in_names[1])
                if W_arr is None:
                    skipped_ops.append(f"MatMul({node.name}): non-const W")
                    n_skipped += 1
                    continue
                s = states[primary]
                n_state = s.c.shape[0]
                if W_arr.ndim == 2:
                    in_dim, out_dim = W_arr.shape
                    if n_state == in_dim:
                        # 1D state (vec) @ W
                        W = W_arr.T  # (in, out) → (out, in) for apply_dense W @ c
                        new_state = apply_dense(s, W, None)
                        states[node.output[0]] = new_state
                        if primary in shapes:
                            shapes[node.output[0]] = (out_dim,)
                    elif n_state % in_dim == 0:
                        # 2D batched state: state is (batch, in_dim) flattened.
                        # Output: (batch, out_dim) flattened.
                        batch = n_state // in_dim
                        # Build block-diagonal effective W of shape (batch*out_dim, batch*in_dim)
                        # Sparse: just apply W per batch row.
                        K_st = s.G.shape[1]
                        new_c = np.zeros(batch * out_dim, dtype=s.c.dtype)
                        new_G = np.zeros((batch * out_dim, K_st), dtype=s.G.dtype)
                        new_tail = (np.zeros(batch * out_dim, dtype=s.tail_radius.dtype)
                                          if s.tail_radius is not None else None)
                        c_2d = s.c.reshape(batch, in_dim)
                        G_2d = s.G.reshape(batch, in_dim, K_st)
                        tail_2d = (s.tail_radius.reshape(batch, in_dim)
                                          if s.tail_radius is not None else None)
                        for b in range(batch):
                            new_c[b*out_dim:(b+1)*out_dim] = c_2d[b] @ W_arr
                            if K_st > 0:
                                # (in,out).T @ (in, K) = (out, K)
                                new_G[b*out_dim:(b+1)*out_dim] = W_arr.T @ G_2d[b]
                            if new_tail is not None:
                                new_tail[b*out_dim:(b+1)*out_dim] = np.abs(W_arr).T @ tail_2d[b]
                        new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                                  slack_records=s.slack_records,
                                                  tail_radius=new_tail)
                        states[node.output[0]] = new_state
                        # Update output shape
                        if primary in shapes:
                            in_shape = shapes[primary]
                            if len(in_shape) >= 1 and in_shape[-1] == in_dim:
                                shapes[node.output[0]] = in_shape[:-1] + (out_dim,)
                            else:
                                shapes[node.output[0]] = (batch, out_dim)
                    else:
                        skipped_ops.append(f"MatMul({node.name}): n_state={n_state} not multiple of in_dim={in_dim}")
                        n_skipped += 1
                        continue
                elif W_arr.ndim == 1:
                    # 1D MatMul: state (M, K) @ W (K,) → (M,).
                    # In flat form: state has M*K elements, output M elements.
                    K = W_arr.shape[0]
                    if n_state % K == 0:
                        M = n_state // K
                        K_st = s.G.shape[1]
                        c_2d = s.c.reshape(M, K)
                        new_c = c_2d @ W_arr  # (M,)
                        if K_st > 0:
                            G_2d = s.G.reshape(M, K, K_st)
                            new_G = (W_arr[None, :, None] * G_2d).sum(axis=1)  # (M, K_st)
                        else:
                            new_G = np.zeros((M, 0))
                        new_tail = None
                        if s.tail_radius is not None:
                            tail_2d = s.tail_radius.reshape(M, K)
                            new_tail = (np.abs(W_arr)[None, :] * tail_2d).sum(axis=1)
                        new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                                  slack_records=s.slack_records,
                                                  tail_radius=new_tail)
                        states[node.output[0]] = new_state
                        # Output shape: drop last dim
                        if primary in shapes:
                            in_shape = shapes[primary]
                            if len(in_shape) >= 1 and in_shape[-1] == K:
                                shapes[node.output[0]] = in_shape[:-1] if len(in_shape) > 1 else (M,)
                            else:
                                shapes[node.output[0]] = (M,)
                    else:
                        skipped_ops.append(f"MatMul({node.name}): n_state={n_state} not multiple of K={K}")
                        n_skipped += 1
                        continue
                else:
                    skipped_ops.append(f"MatMul({node.name}): W ndim={W_arr.ndim} not 1 or 2")
                    n_skipped += 1
                    continue
            elif op == "Add":
                in0, in1 = in_names[0], in_names[1]
                if in0 in states and in1 in states:
                    # RESIDUAL ADD: y = state0 + state1 (both are states)
                    s0 = states[in0]; s1 = states[in1]
                    if s0.c.shape != s1.c.shape:
                        skipped_ops.append(f"Add({node.name}): residual shape mismatch {s0.c.shape} {s1.c.shape}")
                        n_skipped += 1; continue
                    # Pad G to same K (smaller padded with zeros)
                    K0 = s0.G.shape[1]; K1 = s1.G.shape[1]
                    K_max = max(K0, K1)
                    G0_pad = np.concatenate([s0.G, np.zeros((s0.c.shape[0], K_max - K0))], axis=1) if K0 < K_max else s0.G
                    G1_pad = np.concatenate([s1.G, np.zeros((s1.c.shape[0], K_max - K1))], axis=1) if K1 < K_max else s1.G
                    new_c = s0.c + s1.c
                    new_G = G0_pad + G1_pad
                    # Merge slack records by union (preserve all from both branches)
                    # Note: when both branches share a slack via same slack_idx,
                    # the merged G has DOUBLED contribution which is sound but loose.
                    # For sequential CIFAR-style res: skip+branch share earlier ξ but each has own branch slacks.
                    # We just take union of slack_records lists.
                    merged_records = list(s0.slack_records)
                    # Avoid duplicating records that share (layer_index, slack_indices)
                    existing_layers = {rec.layer_index for rec in s0.slack_records}
                    for rec in s1.slack_records:
                        if rec.layer_index not in existing_layers:
                            merged_records.append(rec)
                    # Combine tail_radius from both branches (sound: sum independent boxes)
                    new_tail_res = None
                    if s0.tail_radius is not None or s1.tail_radius is not None:
                        t0_v = s0.tail_radius if s0.tail_radius is not None else 0.0
                        t1_v = s1.tail_radius if s1.tail_radius is not None else 0.0
                        new_tail_res = t0_v + t1_v
                    new_state = FCHZState(c=new_c, G=new_G, n_root=s0.n_root,
                                              slack_records=merged_records,
                                              tail_radius=new_tail_res)
                    states[node.output[0]] = new_state
                    if in0 in shapes: shapes[node.output[0]] = shapes[in0]
                    n_processed += 1
                    continue
                # Otherwise bias add (constant + state)
                bias = None
                src = None
                if in0 in states and in1 not in states:
                    src = in0; bias = get_const(in1)
                elif in1 in states and in0 not in states:
                    src = in1; bias = get_const(in0)
                if src is None or bias is None:
                    skipped_ops.append(f"Add({node.name}): mixed-type Add (need both state or one state + one const)")
                    n_skipped += 1
                    continue
                # Bias add: c += bias
                s = states[src]
                bias = bias.reshape(-1)
                if bias.shape[0] != s.c.shape[0]:
                    # Try per-channel broadcasting (if shape is C-dim)
                    sh = shapes.get(src)
                    if sh is not None and len(sh) == 3 and bias.shape[0] == sh[0]:
                        bias = np.repeat(bias, sh[1] * sh[2])
                    elif bias.shape[0] == 1:
                        bias = np.full(s.c.shape[0], float(bias[0]))
                    else:
                        skipped_ops.append(f"Add({node.name}): bias shape mismatch {bias.shape} vs c {s.c.shape}")
                        n_skipped += 1
                        continue
                # SOUNDNESS FIX 2026-06-07: preserve tail_radius across bias-Add.
                # Bias is constant — adds to c only; tail_radius (error magnitude)
                # is unchanged. Previous code dropped tail_radius silently.
                new_state = FCHZState(c=s.c + bias, G=s.G.copy(),
                                          n_root=s.n_root,
                                          slack_records=s.slack_records,
                                          tail_radius=(s.tail_radius.copy()
                                                              if s.tail_radius is not None else None))
                states[node.output[0]] = new_state
                if src in shapes: shapes[node.output[0]] = shapes[src]
            elif op == "Sub":
                # Constant - state OR state - constant OR state - state (residual).
                in0, in1 = in_names[0], in_names[1]
                # State - state case (residual subtraction)
                if in0 in states and in1 in states:
                    s0 = states[in0]; s1 = states[in1]
                    if s0.c.shape != s1.c.shape:
                        skipped_ops.append(f"Sub({node.name}): state-state shape mismatch")
                        n_skipped += 1; continue
                    K0 = s0.G.shape[1]; K1 = s1.G.shape[1]
                    K_max = max(K0, K1)
                    n = s0.c.shape[0]
                    G0_pad = np.concatenate([s0.G, np.zeros((n, K_max - K0))], axis=1) if K0 < K_max else s0.G
                    G1_pad = np.concatenate([s1.G, np.zeros((n, K_max - K1))], axis=1) if K1 < K_max else s1.G
                    new_c = s0.c - s1.c
                    new_G = G0_pad - G1_pad
                    t0 = s0.tail_radius if s0.tail_radius is not None else 0.0
                    t1 = s1.tail_radius if s1.tail_radius is not None else 0.0
                    new_tail = (t0 + t1) if (s0.tail_radius is not None or s1.tail_radius is not None) else None
                    if new_tail is not None and not isinstance(new_tail, np.ndarray):
                        new_tail = np.full(n, float(new_tail))
                    new_state = FCHZState(c=new_c, G=new_G, n_root=s0.n_root,
                                              slack_records=list(s0.slack_records),
                                              tail_radius=new_tail)
                    states[node.output[0]] = new_state
                    if in0 in shapes: shapes[node.output[0]] = shapes[in0]
                    n_processed += 1
                    continue
                if in0 in states and in1 not in states:
                    src = in0; const_arr = get_const(in1)
                    state_left = True
                elif in1 in states and in0 not in states:
                    src = in1; const_arr = get_const(in0)
                    state_left = False
                else:
                    skipped_ops.append(f"Sub({node.name}): need const operand")
                    n_skipped += 1
                    continue
                if const_arr is None:
                    skipped_ops.append(f"Sub({node.name}): no const value")
                    n_skipped += 1
                    continue
                s = states[src]
                arr = const_arr.astype(np.float64).reshape(-1)
                n_s = s.c.shape[0]; n_c = arr.shape[0]
                # Broadcast: state and const broadcast to max(n_s, n_c)
                if n_s == n_c:
                    new_c_s = s.c
                    new_G_s = s.G
                    new_tail_s = s.tail_radius
                    new_arr = arr
                    out_n = n_s
                elif n_s == 1:
                    # State scalar -> replicate to const size
                    new_c_s = np.full(n_c, float(s.c[0]))
                    new_G_s = np.repeat(s.G, n_c, axis=0)
                    new_tail_s = (np.full(n_c, float(s.tail_radius[0]))
                                          if s.tail_radius is not None else None)
                    new_arr = arr
                    out_n = n_c
                elif n_c == 1:
                    new_c_s = s.c
                    new_G_s = s.G
                    new_tail_s = s.tail_radius
                    new_arr = np.full(n_s, float(arr[0]))
                    out_n = n_s
                else:
                    # General outer-broadcast: state (M, *) and const (N, *) → (M, N).
                    # Common ml4acopf pattern: state (M, 1) - const (N,) → (M, N).
                    # Use input shapes to verify. State shape after Unsqueeze is (M, 1).
                    sh_s = shapes.get(src, (n_s,))
                    if len(sh_s) >= 2 and sh_s[-1] == 1 and int(np.prod(sh_s[:-1])) == n_s:
                        # State (..., 1) - const (n_c,) → (..., n_c)
                        M = n_s; N = n_c
                        # Each row m: c[m] - arr[n] for n in range(N)
                        K_st = s.G.shape[1]
                        new_c_s = np.repeat(s.c, N)  # (M*N,) row-major
                        new_G_s = np.repeat(s.G, N, axis=0)  # (M*N, K)
                        new_tail_s = (np.repeat(s.tail_radius, N)
                                              if s.tail_radius is not None else None)
                        # broadcast const tiled: [arr, arr, ..., arr] M times
                        new_arr = np.tile(arr, M)
                        out_n = M * N
                        # Output shape: sh_s[:-1] + (N,)
                        shapes[node.output[0]] = tuple(sh_s[:-1]) + (N,)
                        if state_left:
                            new_state = FCHZState(c=new_c_s - new_arr,
                                                          G=new_G_s.copy(),
                                                          n_root=s.n_root,
                                                          slack_records=s.slack_records,
                                                          tail_radius=(new_tail_s.copy()
                                                                              if new_tail_s is not None else None))
                        else:
                            new_state = FCHZState(c=new_arr - new_c_s,
                                                          G=-new_G_s.copy(),
                                                          n_root=s.n_root,
                                                          slack_records=s.slack_records,
                                                          tail_radius=(new_tail_s.copy()
                                                                              if new_tail_s is not None else None))
                        states[node.output[0]] = new_state
                        continue
                    skipped_ops.append(f"Sub({node.name}): shape mismatch n_s={n_s} n_c={n_c} sh_s={sh_s}")
                    n_skipped += 1
                    continue
                if state_left:  # state - const
                    new_state = FCHZState(c=new_c_s - new_arr, G=new_G_s.copy(),
                                              n_root=s.n_root,
                                              slack_records=s.slack_records,
                                              tail_radius=(new_tail_s.copy()
                                                                  if new_tail_s is not None else None))
                else:  # const - state
                    new_state = FCHZState(c=new_arr - new_c_s, G=-new_G_s.copy(),
                                              n_root=s.n_root,
                                              slack_records=s.slack_records,
                                              tail_radius=(new_tail_s.copy()
                                                                  if new_tail_s is not None else None))
                states[node.output[0]] = new_state
                # output shape: same as broadcast result; default 1D
                shapes[node.output[0]] = (out_n,)
            elif op == "Mul":
                in0, in1 = in_names[0], in_names[1]
                # State * state case: fallback to interval bound (nonlinear)
                if in0 in states and in1 in states:
                    s0 = states[in0]; s1 = states[in1]
                    if s0.c.shape != s1.c.shape:
                        skipped_ops.append(f"Mul({node.name}): state-state shape mismatch")
                        n_skipped += 1; continue
                    # Bound y = x0 * x1 via interval bounds at each row
                    r0 = np.abs(s0.G).sum(axis=1) + (s0.tail_radius if s0.tail_radius is not None else 0)
                    r1 = np.abs(s1.G).sum(axis=1) + (s1.tail_radius if s1.tail_radius is not None else 0)
                    l0, u0 = s0.c - r0, s0.c + r0
                    l1, u1 = s1.c - r1, s1.c + r1
                    corners = np.stack([l0*l1, l0*u1, u0*l1, u0*u1], axis=0)
                    lb_out = corners.min(axis=0)
                    ub_out = corners.max(axis=0)
                    n = s0.c.shape[0]
                    new_c = (lb_out + ub_out) / 2.0
                    new_tail = (ub_out - lb_out) / 2.0
                    new_state = FCHZState(c=new_c, G=np.zeros((n, 0), dtype=np.float64),
                                              n_root=s0.n_root, slack_records=s0.slack_records,
                                              tail_radius=new_tail)
                    states[node.output[0]] = new_state
                    if in0 in shapes: shapes[node.output[0]] = shapes[in0]
                    n_processed += 1
                    continue
                if in0 in states and in1 not in states:
                    src = in0; scale = get_const(in1)
                elif in1 in states and in0 not in states:
                    src = in1; scale = get_const(in0)
                else:
                    skipped_ops.append(f"Mul({node.name}): need const operand")
                    n_skipped += 1
                    continue
                if scale is None:
                    skipped_ops.append(f"Mul({node.name}): no const value")
                    n_skipped += 1
                    continue
                s = states[src]
                arr = scale.reshape(-1)
                if arr.shape[0] != s.c.shape[0]:
                    if arr.shape[0] == 1:
                        arr = np.full(s.c.shape[0], float(arr[0]))
                    else:
                        skipped_ops.append(f"Mul({node.name}): shape mismatch")
                        n_skipped += 1
                        continue
                # SOUNDNESS FIX 2026-06-07: scale tail_radius by |arr|.
                # y = arr * x → tail magnitude scales by |arr| per row.
                # Previous code dropped tail_radius silently → unsound tightening.
                new_tail_mul = None
                if s.tail_radius is not None:
                    new_tail_mul = np.abs(arr) * s.tail_radius
                new_state = FCHZState(c=s.c * arr, G=s.G * arr[:, None],
                                          n_root=s.n_root,
                                          slack_records=s.slack_records,
                                          tail_radius=new_tail_mul)
                states[node.output[0]] = new_state
            elif op == "Relu":
                if primary is None:
                    skipped_ops.append(f"Relu({node.name}): no state input")
                    n_skipped += 1
                    continue
                if hz_only:
                    # Memory-efficient ReLU using per-row tail_radius (sound).
                    # y_i = lam_i * z_i + mu_i + mu_i * s_i, s_i ∈ [-1, 1] independent
                    # HZ closed-form contribution from this slack: |d_i| * mu_i (per-row).
                    # So we add |mu| to tail_radius (sound, treats slacks as independent boxes).
                    s = states[primary]
                    # Include tail_radius in bound computation
                    rad = np.abs(s.G).sum(axis=1)
                    if s.tail_radius is not None:
                        rad = rad + s.tail_radius
                    l = s.c - rad
                    u = s.c + rad
                    is_active = l >= 0
                    is_inactive = u <= 0
                    is_unstable = ~is_active & ~is_inactive
                    den = np.where(is_unstable, u - l, 1.0)
                    lam = np.where(is_unstable, u / np.maximum(den, 1e-300), 0.0)
                    lam = np.where(is_active, 1.0, lam)
                    lam = np.where(is_inactive, 0.0, lam)
                    mu = np.where(is_unstable, -lam * l / 2.0, 0.0)
                    new_c = lam * s.c + mu
                    # In-place scale G by lam (avoids full copy via *=)
                    new_G = s.G.copy()
                    new_G *= lam[:, None]
                    # Propagate existing tail_radius through ReLU: lam-scaled
                    new_tail = None
                    if s.tail_radius is not None:
                        new_tail = lam * s.tail_radius
                    # Add per-row mu as new tail (sound: independent slacks)
                    if np.any(is_unstable):
                        if new_tail is None:
                            new_tail = np.abs(mu)
                        else:
                            new_tail = new_tail + np.abs(mu)
                    new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                              slack_records=s.slack_records,
                                              tail_radius=new_tail)
                else:
                    new_state = apply_relu_triangle_with_record(
                        states[primary], layer_index=layer_index,
                    )
                states[node.output[0]] = new_state
                if primary in shapes:
                    shapes[node.output[0]] = shapes[primary]
                layer_index += 1
            elif op in ("Sigmoid", "Tanh"):
                # Sound chord linear relaxation: y ≈ α·x + β + ε, ε ∈ [-r, +r].
                # Chord passes through (l, σ(l)) and (u, σ(u)).
                # Per-row independent box error → adds to tail_radius (sound).
                if primary is None:
                    skipped_ops.append(f"{op}({node.name}): no state input")
                    n_skipped += 1
                    continue
                s = states[primary]
                rad = np.abs(s.G).sum(axis=1)
                if s.tail_radius is not None:
                    rad = rad + s.tail_radius
                l_arr = s.c - rad
                u_arr = s.c + rad
                if op == "Sigmoid":
                    fn = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
                else:  # Tanh
                    fn = np.tanh
                fl = fn(l_arr); fu = fn(u_arr)
                den = np.maximum(u_arr - l_arr, 1e-12)
                alpha = (fu - fl) / den
                # Zero alpha (and zero G impact) where l == u (degenerate)
                alpha = np.where(u_arr - l_arr < 1e-12, 0.0, alpha)
                beta = fl - alpha * l_arr
                # ANALYTICAL sound bound (no sampling).
                # For σ ∈ {sigmoid, tanh}, the chord through (l, σ(l)) and (u, σ(u))
                # has slope α. Let dev(x) = σ(x) - (α x + β_init), β_init = σ(l) - α l.
                # dev(l) = dev(u) = 0. Critical points: σ'(x*) = α.
                #
                # Sigmoid: σ'(x) = σ(x)(1-σ(x)) = α  ⇒  σ(x) = (1 ± √(1-4α))/2
                #          x* = logit(σ(x)) = ln(σ/(1-σ))
                # Tanh:    σ'(x) = 1 - σ(x)²  = α  ⇒  σ(x) = ±√(1-α)
                #          x* = atanh(σ(x))
                #
                # For each row, find critical x* in (l, u) (if any); compute dev at x*.
                # Sound radius = (dev_max - dev_min) / 2 over {dev=0 at endpoints, dev(x*) for valid x*}.
                # Re-center β so error is ε ∈ [-radius, +radius] symmetric.
                # See research/sc_hz/TAIL_RADIUS_SOUNDNESS_PROOF.md §sigmoid for proof.
                #
                # Numerical: when α > 0.25 (sigmoid) or α > 1.0 (tanh), no interior
                # critical point (chord steeper than σ' anywhere) → dev = 0 everywhere,
                # radius = 0. When α very small, two critical points (left + right).
                dev_max_row = np.zeros_like(l_arr)
                dev_min_row = np.zeros_like(l_arr)
                if op == "Sigmoid":
                    max_slope = 0.25
                    # σ(x) = s where s ∈ (0, 1): x = logit(s) = ln(s/(1-s))
                    # Roots: s = (1 ± √(1-4α))/2; requires 1 - 4α ≥ 0
                    alpha_safe = np.minimum(alpha, max_slope - 1e-12)
                    disc = np.sqrt(np.maximum(1.0 - 4.0 * alpha_safe, 0.0))
                    valid = (alpha > 1e-15) & (alpha < max_slope)
                else:  # Tanh
                    max_slope = 1.0
                    # σ(x) = ±√(1-α). x = atanh(s).
                    alpha_safe = np.minimum(alpha, max_slope - 1e-12)
                    disc = np.sqrt(np.maximum(1.0 - alpha_safe, 0.0))
                    valid = (alpha > 1e-15) & (alpha < max_slope)
                # Two critical x* per row (sigmoid) or two (tanh)
                for sign in [+1.0, -1.0]:
                    if op == "Sigmoid":
                        s_val = (1.0 + sign * disc) / 2.0
                        s_safe = np.clip(s_val, 1e-15, 1.0 - 1e-15)
                        x_star = np.log(s_safe / (1.0 - s_safe))
                    else:
                        s_val = sign * disc
                        s_safe = np.clip(s_val, -1.0 + 1e-15, 1.0 - 1e-15)
                        x_star = 0.5 * np.log((1.0 + s_safe) / (1.0 - s_safe))
                    in_range = valid & (x_star > l_arr) & (x_star < u_arr)
                    dev_star = fn(x_star) - (alpha * x_star + beta)
                    dev_max_row = np.where(in_range,
                                                  np.maximum(dev_max_row, dev_star),
                                                  dev_max_row)
                    dev_min_row = np.where(in_range,
                                                  np.minimum(dev_min_row, dev_star),
                                                  dev_min_row)
                # Endpoint contribution is 0 (chord matches σ at l, u).
                # Final symmetric box: center β at (dev_max + dev_min)/2, half-width radius.
                mid_dev = (dev_max_row + dev_min_row) / 2.0
                radius = (dev_max_row - dev_min_row) / 2.0
                beta = beta + mid_dev
                # New state: y = α·x + β + ε with ε ∈ [-r, r]
                new_c = alpha * s.c + beta
                new_G = s.G * alpha[:, None]
                new_tail = None
                if s.tail_radius is not None:
                    new_tail = np.abs(alpha) * s.tail_radius
                if np.any(radius > 0):
                    if new_tail is None:
                        new_tail = radius
                    else:
                        new_tail = new_tail + radius
                new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                          slack_records=s.slack_records,
                                          tail_radius=new_tail)
                states[node.output[0]] = new_state
                if primary in shapes:
                    shapes[node.output[0]] = shapes[primary]
            elif op == "Cast":
                # Numeric type conversion. For float state, treat as identity.
                # Walker only handles float; if Cast to int (e.g. for indexing),
                # we still pass through — downstream ops may fail safely.
                if primary is None:
                    skipped_ops.append(f"Cast({node.name}): no state")
                    n_skipped += 1; continue
                states[node.output[0]] = states[primary]
                if primary in shapes:
                    shapes[node.output[0]] = shapes[primary]
            elif op == "Shape":
                # ONNX Shape returns shape tensor of input. We treat it as a
                # constant int tensor. Downstream Reshape ops may consume this.
                if primary is None:
                    # If we have shapes recorded, return them as a constant
                    in_name = in_names[0] if in_names else None
                    if in_name in shapes:
                        sh = shapes[in_name]
                        # Output is 1D int64 tensor with shape values
                        const_cache[node.output[0]] = np.asarray(sh, dtype=np.int64)
                        n_processed += 1
                        continue
                    skipped_ops.append(f"Shape({node.name}): no state input")
                    n_skipped += 1
                    continue
                # If state exists, get its shape and produce as constant
                in_name = in_names[0]
                if in_name in shapes:
                    sh = shapes[in_name]
                    const_cache[node.output[0]] = np.asarray(sh, dtype=np.int64)
                else:
                    # Use state c shape (flat)
                    const_cache[node.output[0]] = np.asarray([states[primary].c.shape[0]], dtype=np.int64)
            elif op == "Identity":
                # Pure identity
                if primary is None:
                    skipped_ops.append(f"Identity({node.name}): no state input")
                    n_skipped += 1
                    continue
                states[node.output[0]] = states[primary]
                if primary in shapes:
                    shapes[node.output[0]] = shapes[primary]
            elif op == "Tanh":
                # Same as Sigmoid: analytical chord relaxation
                if primary is None:
                    skipped_ops.append(f"Tanh({node.name}): no state input")
                    n_skipped += 1
                    continue
                s = states[primary]
                rad = np.abs(s.G).sum(axis=1)
                if s.tail_radius is not None:
                    rad = rad + s.tail_radius
                l_arr = s.c - rad
                u_arr = s.c + rad
                fn = np.tanh
                fl = fn(l_arr); fu = fn(u_arr)
                den = np.maximum(u_arr - l_arr, 1e-12)
                alpha = (fu - fl) / den
                alpha = np.where(u_arr - l_arr < 1e-12, 0.0, alpha)
                beta = fl - alpha * l_arr
                max_slope = 1.0
                alpha_safe = np.minimum(alpha, max_slope - 1e-12)
                disc = np.sqrt(np.maximum(1.0 - alpha_safe, 0.0))
                valid = (alpha > 1e-15) & (alpha < max_slope)
                dev_max_row = np.zeros_like(l_arr)
                dev_min_row = np.zeros_like(l_arr)
                for sign in [+1.0, -1.0]:
                    s_val = sign * disc
                    s_safe = np.clip(s_val, -1.0 + 1e-15, 1.0 - 1e-15)
                    x_star = 0.5 * np.log((1.0 + s_safe) / (1.0 - s_safe))
                    in_range = valid & (x_star > l_arr) & (x_star < u_arr)
                    dev_star = fn(x_star) - (alpha * x_star + beta)
                    dev_max_row = np.where(in_range, np.maximum(dev_max_row, dev_star), dev_max_row)
                    dev_min_row = np.where(in_range, np.minimum(dev_min_row, dev_star), dev_min_row)
                mid_dev = (dev_max_row + dev_min_row) / 2.0
                radius = (dev_max_row - dev_min_row) / 2.0
                beta = beta + mid_dev
                new_c = alpha * s.c + beta
                new_G = s.G * alpha[:, None]
                new_tail = None
                if s.tail_radius is not None:
                    new_tail = np.abs(alpha) * s.tail_radius
                if np.any(radius > 0):
                    new_tail = (new_tail + radius) if new_tail is not None else radius
                new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                              slack_records=s.slack_records,
                                              tail_radius=new_tail)
                states[node.output[0]] = new_state
                if primary in shapes:
                    shapes[node.output[0]] = shapes[primary]
            elif op == "Sign":
                # sign(x) is non-smooth: -1 if x<0, 0 if x==0, +1 if x>0.
                # Sound box relaxation: each row interval [-1, 1] → bound with l<=0<u → [-1,1]
                # Otherwise (all positive or all negative): point value.
                if primary is None:
                    skipped_ops.append(f"Sign({node.name}): no state input")
                    n_skipped += 1
                    continue
                s = states[primary]
                rad = np.abs(s.G).sum(axis=1)
                if s.tail_radius is not None:
                    rad = rad + s.tail_radius
                l_arr = s.c - rad
                u_arr = s.c + rad
                is_pos = l_arr > 0  # output = +1
                is_neg = u_arr < 0  # output = -1
                # Otherwise interval [-1, 1]: use box [c_new=0, tail=1]
                new_c = np.where(is_pos, 1.0, np.where(is_neg, -1.0, 0.0))
                new_G = np.zeros((s.c.shape[0], 0), dtype=s.G.dtype)
                # Sound radius: 0 for stable rows, 1 for uncertain rows
                new_tail = np.where(is_pos | is_neg, 0.0, 1.0)
                new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                              slack_records=s.slack_records,
                                              tail_radius=new_tail)
                states[node.output[0]] = new_state
                if primary in shapes:
                    shapes[node.output[0]] = shapes[primary]
            elif op == "AveragePool":
                # Window-average pool (CNN). Treat as Conv with 1/window weight.
                if primary is None:
                    skipped_ops.append(f"AveragePool({node.name}): no state input")
                    n_skipped += 1
                    continue
                s = states[primary]
                in_shape = shapes.get(primary)
                if in_shape is None or len(in_shape) != 3:
                    skipped_ops.append(f"AveragePool({node.name}): need 3D shape")
                    n_skipped += 1
                    continue
                attrs = {a.name: a for a in node.attribute}
                ks = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else [2, 2]
                strides = list(attrs["strides"].ints) if "strides" in attrs else ks
                pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
                Ci, Hi, Wi = in_shape
                kH, kW = ks[0], ks[1]
                sH, sW = strides[0], strides[1]
                pH = pads[0]; pW = pads[1] if len(pads) > 1 else pH
                Ho = (Hi + 2*pH - kH) // sH + 1
                Wo = (Wi + 2*pW - kW) // sW + 1
                # Average pool = Conv with weight = 1/(kH*kW) over each channel
                weight = np.zeros((Ci, 1, kH, kW), dtype=s.c.dtype)
                weight[:, 0, :, :] = 1.0 / (kH * kW)
                # Build a depthwise conv input shape
                try:
                    import torch
                    import torch.nn.functional as F
                    c_in = torch.from_numpy(s.c.reshape(1, Ci, Hi, Wi))
                    w_t = torch.from_numpy(weight)
                    c_out = F.conv2d(c_in, w_t, None, stride=(sH, sW), padding=(pH, pW), groups=Ci)
                    new_c = c_out.numpy().reshape(-1)
                    K = s.G.shape[1]
                    new_G = np.zeros((Ci*Ho*Wo, K), dtype=s.G.dtype)
                    if K > 0:
                        for k in range(K):
                            g_in = torch.from_numpy(s.G[:, k].reshape(1, Ci, Hi, Wi))
                            g_out = F.conv2d(g_in, w_t, None, stride=(sH, sW), padding=(pH, pW), groups=Ci)
                            new_G[:, k] = g_out.numpy().reshape(-1)
                    new_tail = None
                    if s.tail_radius is not None:
                        tail_in = torch.from_numpy(np.abs(s.tail_radius).reshape(1, Ci, Hi, Wi))
                        w_abs = torch.from_numpy(np.abs(weight))
                        tail_out = F.conv2d(tail_in, w_abs, None, stride=(sH, sW), padding=(pH, pW), groups=Ci)
                        new_tail = tail_out.numpy().reshape(-1)
                    new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                                  slack_records=s.slack_records,
                                                  tail_radius=new_tail)
                    states[node.output[0]] = new_state
                    shapes[node.output[0]] = (Ci, Ho, Wo)
                except Exception as e:
                    skipped_ops.append(f"AveragePool({node.name}): {type(e).__name__}: {str(e)[:60]}")
                    n_skipped += 1
                    continue
            elif op == "Expand":
                # Broadcast state to target shape (replicate values).
                if primary is None:
                    skipped_ops.append(f"Expand({node.name}): no state input")
                    n_skipped += 1; continue
                target_shape = None
                if len(in_names) > 1 and in_names[1] in const_cache:
                    target_shape = const_cache[in_names[1]].astype(np.int64).reshape(-1)
                elif len(in_names) > 1:
                    target_shape = get_const(in_names[1])
                    if target_shape is not None: target_shape = target_shape.astype(np.int64).reshape(-1)
                if target_shape is None:
                    skipped_ops.append(f"Expand({node.name}): no shape")
                    n_skipped += 1; continue
                s = states[primary]
                # Strip batch
                ts = tuple(int(x) for x in target_shape)
                if ts[0] == 1 and len(ts) > 1:
                    ts_nb = ts[1:]
                else:
                    ts_nb = ts
                in_sh = shapes.get(primary, (s.c.shape[0],))
                try:
                    n_out = int(np.prod(ts_nb))
                    # If input is scalar (n=1), replicate
                    if s.c.shape[0] == 1 and n_out > 1:
                        new_c = np.full(n_out, float(s.c[0]))
                        K = s.G.shape[1]
                        new_G = np.tile(s.G, (n_out, 1)) if K > 0 else np.zeros((n_out, 0))
                        new_tail = None
                        if s.tail_radius is not None:
                            new_tail = np.full(n_out, float(s.tail_radius[0]))
                        new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                                      slack_records=s.slack_records,
                                                      tail_radius=new_tail)
                        states[node.output[0]] = new_state
                        shapes[node.output[0]] = ts_nb
                    else:
                        # Try broadcast via numpy reshape + broadcast
                        c_in = s.c.reshape(in_sh)
                        c_out = np.broadcast_to(c_in, ts_nb).copy().reshape(-1)
                        K = s.G.shape[1]
                        if K > 0:
                            G_in = s.G.reshape(in_sh + (K,))
                            G_out = np.broadcast_to(G_in, ts_nb + (K,)).copy().reshape(-1, K)
                        else:
                            G_out = np.zeros((n_out, 0), dtype=s.G.dtype)
                        new_tail = None
                        if s.tail_radius is not None:
                            tail_in = s.tail_radius.reshape(in_sh)
                            new_tail = np.broadcast_to(tail_in, ts_nb).copy().reshape(-1)
                        new_state = FCHZState(c=c_out, G=G_out, n_root=s.n_root,
                                                      slack_records=s.slack_records,
                                                      tail_radius=new_tail)
                        states[node.output[0]] = new_state
                        shapes[node.output[0]] = ts_nb
                except Exception as e:
                    skipped_ops.append(f"Expand({node.name}): {type(e).__name__}")
                    n_skipped += 1
                    continue
            elif op == "ConstantOfShape":
                # Produces a constant tensor with given shape. Shape from input.
                shape_arr = get_const(in_names[0]) if in_names else None
                if shape_arr is None and in_names[0] in const_cache:
                    shape_arr = const_cache[in_names[0]]
                if shape_arr is None:
                    skipped_ops.append(f"ConstantOfShape({node.name}): no shape input")
                    n_skipped += 1
                    continue
                attrs = {a.name: a for a in node.attribute}
                val = 0.0
                if "value" in attrs:
                    val_tensor = onnx.numpy_helper.to_array(attrs["value"].t)
                    val = float(val_tensor.flat[0])
                n_out = int(np.prod(shape_arr))
                c = np.full(n_out, val, dtype=np.float64)
                new_state = FCHZState(c=c, G=np.zeros((n_out, 0), dtype=np.float64),
                                              n_root=0, slack_records=[], tail_radius=None)
                states[node.output[0]] = new_state
                shapes[node.output[0]] = tuple(int(x) for x in shape_arr)
            elif op == "Pow":
                # x ** y where y is constant scalar (or per-element).
                # For state input x with bounds [l, u] (assuming x >= 0 for sound bound),
                # and integer y: monotone, so [l^y, u^y].
                # Fall back to interval bound (sound box).
                if primary is None:
                    skipped_ops.append(f"Pow({node.name}): no state input")
                    n_skipped += 1; continue
                exp_arr = get_const(in_names[1]) if len(in_names) > 1 else None
                if exp_arr is None:
                    skipped_ops.append(f"Pow({node.name}): non-const exponent")
                    n_skipped += 1; continue
                exp_val = float(exp_arr.flat[0]) if exp_arr.size > 0 else 1.0
                s = states[primary]
                rad = np.abs(s.G).sum(axis=1) + (s.tail_radius if s.tail_radius is not None else 0)
                l = s.c - rad; u = s.c + rad
                # For sound bound: if exp is even or x >= 0, use [l^p, u^p].
                # Compute pointwise; if l < 0 and exp not int, use 0 lower bound.
                try:
                    if abs(exp_val - round(exp_val)) < 1e-9:
                        # Integer exponent
                        e_int = int(round(exp_val))
                        if e_int % 2 == 0:
                            # Even: parabola → min at 0 if range crosses
                            lb_out = np.where((l <= 0) & (u >= 0), 0.0,
                                                   np.minimum(np.power(l, e_int), np.power(u, e_int)))
                            ub_out = np.maximum(np.power(l, e_int), np.power(u, e_int))
                        else:
                            # Odd: monotone
                            lb_out = np.power(l, e_int)
                            ub_out = np.power(u, e_int)
                    else:
                        # Non-integer: assume input non-negative
                        l_safe = np.maximum(l, 1e-12)
                        u_safe = np.maximum(u, 1e-12)
                        lb_out = np.power(l_safe, exp_val)
                        ub_out = np.power(u_safe, exp_val)
                    # Box state
                    n = s.c.shape[0]
                    new_c = (lb_out + ub_out) / 2.0
                    new_tail = (ub_out - lb_out) / 2.0
                    new_state = FCHZState(c=new_c, G=np.zeros((n, 0), dtype=np.float64),
                                                  n_root=s.n_root, slack_records=s.slack_records,
                                                  tail_radius=new_tail)
                    states[node.output[0]] = new_state
                    if primary in shapes:
                        shapes[node.output[0]] = shapes[primary]
                except Exception as e:
                    skipped_ops.append(f"Pow({node.name}): {type(e).__name__}")
                    n_skipped += 1
                    continue
            elif op == "Upsample" or op == "Resize":
                # Upsample by integer scale via nearest neighbor.
                # Sound (perfectly correlated state across replicated positions).
                if primary is None:
                    skipped_ops.append(f"{op}({node.name}): no state input")
                    n_skipped += 1; continue
                s = states[primary]
                in_shape = shapes.get(primary)
                if in_shape is None or len(in_shape) != 3:
                    skipped_ops.append(f"{op}({node.name}): need 3D shape, got {in_shape}")
                    n_skipped += 1; continue
                # Get scales from inputs or attributes
                scales_arr = None
                for i in range(1, len(in_names)):
                    if in_names[i] in const_cache:
                        scales_arr = const_cache[in_names[i]]; break
                    sc = get_const(in_names[i])
                    if sc is not None:
                        scales_arr = sc; break
                if scales_arr is None:
                    attrs = {a.name: a for a in node.attribute}
                    if "scales" in attrs:
                        scales_arr = np.array(attrs["scales"].floats)
                if scales_arr is None:
                    skipped_ops.append(f"{op}({node.name}): no scales")
                    n_skipped += 1; continue
                scales = np.asarray(scales_arr).flatten()
                if len(scales) == 4:
                    s_H, s_W = float(scales[2]), float(scales[3])
                elif len(scales) == 2:
                    s_H, s_W = float(scales[0]), float(scales[1])
                else:
                    skipped_ops.append(f"{op}({node.name}): bad scales {scales}")
                    n_skipped += 1; continue
                Ci, Hi, Wi = in_shape
                Ho = int(Hi * s_H); Wo = int(Wi * s_W)
                # Nearest neighbor: replicate via np.repeat
                try:
                    c_nd = s.c.reshape(Ci, Hi, Wi)
                    c_up = np.repeat(np.repeat(c_nd, int(s_H), axis=1), int(s_W), axis=2)
                    new_c = c_up[:, :Ho, :Wo].reshape(-1).copy()
                    K = s.G.shape[1]
                    if K > 0:
                        G_nd = s.G.reshape(Ci, Hi, Wi, K)
                        G_up = np.repeat(np.repeat(G_nd, int(s_H), axis=1), int(s_W), axis=2)
                        new_G = G_up[:, :Ho, :Wo, :].reshape(-1, K).copy()
                    else:
                        new_G = np.zeros((Ci*Ho*Wo, 0), dtype=s.G.dtype)
                    new_tail = None
                    if s.tail_radius is not None:
                        tail_nd = s.tail_radius.reshape(Ci, Hi, Wi)
                        tail_up = np.repeat(np.repeat(tail_nd, int(s_H), axis=1), int(s_W), axis=2)
                        new_tail = tail_up[:, :Ho, :Wo].reshape(-1).copy()
                    new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                                  slack_records=s.slack_records,
                                                  tail_radius=new_tail)
                    states[node.output[0]] = new_state
                    shapes[node.output[0]] = (Ci, Ho, Wo)
                except Exception as e:
                    skipped_ops.append(f"{op}({node.name}): {type(e).__name__}")
                    n_skipped += 1
                    continue
            elif op == "Dropout":
                # Inference: identity
                if primary is None:
                    skipped_ops.append(f"Dropout({node.name}): no state input")
                    n_skipped += 1
                    continue
                states[node.output[0]] = states[primary]
            elif op == "Conv":
                # Apply Conv2D to FCHZState
                if primary is None:
                    skipped_ops.append(f"Conv({node.name}): no state input")
                    n_skipped += 1; continue
                W_arr = get_const(in_names[1])
                if W_arr is None:
                    skipped_ops.append(f"Conv({node.name}): non-const W")
                    n_skipped += 1; continue
                b_arr = get_const(in_names[2]) if len(in_names) >= 3 else None
                attrs = {a.name: a for a in node.attribute}
                kernel_shape = tuple(attrs['kernel_shape'].ints) if 'kernel_shape' in attrs else None
                strides = tuple(attrs['strides'].ints) if 'strides' in attrs else (1, 1)
                pads = tuple(attrs['pads'].ints) if 'pads' in attrs else (0, 0, 0, 0)
                groups = attrs['group'].i if 'group' in attrs else 1
                s = states[primary]
                W = W_arr.astype(np.float64)
                Co, Ci_per_g, kH, kW = W.shape
                Ci = Ci_per_g * groups
                # Get input shape from shapes dict
                in_shape = shapes.get(primary)
                if in_shape is None or len(in_shape) != 3:
                    # try square assumption
                    n = s.c.shape[0]
                    spatial = n // Ci
                    Hi_guess = int(np.sqrt(spatial))
                    if Hi_guess * Hi_guess == spatial:
                        Hi = Wi = Hi_guess
                    else:
                        skipped_ops.append(f"Conv({node.name}): cannot infer shape ({Ci}, ?, ?)")
                        n_skipped += 1; continue
                else:
                    _, Hi, Wi = in_shape
                import torch
                import torch.nn.functional as F
                # GPU acceleration if HYZOR_FCHZ_USE_CUDA=1 and torch.cuda available
                use_cuda = (os.environ.get('HYZOR_FCHZ_USE_CUDA', '0') == '1'
                                  and torch.cuda.is_available())
                dev = torch.device('cuda') if use_cuda else torch.device('cpu')
                W_t = torch.from_numpy(W).to(torch.float64).to(dev)
                b_t = (torch.from_numpy(np.asarray(b_arr).astype(np.float64).reshape(-1)).to(torch.float64).to(dev)
                          if b_arr is not None else None)
                pad_h, pad_w = pads[0], pads[1]
                # Center
                c_in = torch.from_numpy(s.c.reshape(1, Ci, Hi, Wi)).to(torch.float64).to(dev)
                c_out = F.conv2d(c_in, W_t, b_t, stride=strides, padding=(pad_h, pad_w), groups=groups)
                new_c = c_out.detach().cpu().numpy().reshape(-1)
                _, Co_p, Ho_p, Wo_p = c_out.shape
                # G in chunks to bound peak memory (chunk K dim)
                K_st = s.G.shape[1]
                n_out = Co_p * Ho_p * Wo_p
                # Float32 for G to halve memory (HZ closed-form precision OK)
                new_G = np.zeros((n_out, K_st), dtype=np.float32)
                if K_st > 0:
                    # Chunk size targeting ~256 MB per intermediate tensor at f32
                    bytes_per_col = Co_p * Ho_p * Wo_p * 4
                    chunk_max = max(64, min(K_st, int(2.5e8 // max(bytes_per_col, 1))))
                    W_t_f32 = W_t.to(torch.float32)
                    G_T = s.G.T.astype(np.float32, copy=False)
                    for start in range(0, K_st, chunk_max):
                        end = min(K_st, start + chunk_max)
                        chunk = G_T[start:end].reshape(end - start, Ci, Hi, Wi)
                        G_chunk_t = torch.from_numpy(chunk).to(dev)
                        G_out_chunk = F.conv2d(G_chunk_t, W_t_f32, None,
                                                    stride=strides,
                                                    padding=(pad_h, pad_w),
                                                    groups=groups)
                        new_G[:, start:end] = G_out_chunk.detach().cpu().numpy().reshape(end - start, -1).T
                        del G_chunk_t, G_out_chunk
                    del G_T
                # Promote new_G back to float64 for downstream (compatibility)
                new_G = new_G.astype(np.float64)
                # Propagate tail_radius through Conv: new_tail = |W| applied to tail as image
                new_tail = None
                if s.tail_radius is not None:
                    abs_W_t = torch.abs(W_t_f32)
                    tail_in = torch.from_numpy(s.tail_radius.astype(np.float32).reshape(1, Ci, Hi, Wi)).to(dev)
                    tail_out = F.conv2d(tail_in, abs_W_t, None,
                                            stride=strides,
                                            padding=(pad_h, pad_w),
                                            groups=groups)
                    new_tail = tail_out.detach().cpu().numpy().reshape(-1).astype(np.float64)
                new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                          slack_records=s.slack_records,
                                          tail_radius=new_tail)
                states[node.output[0]] = new_state
                shapes[node.output[0]] = (Co_p, Ho_p, Wo_p)
            elif op == "ConvTranspose":
                # Transposed Conv2D — used in cgan for upsampling.
                if primary is None:
                    skipped_ops.append(f"ConvTranspose({node.name}): no state input")
                    n_skipped += 1; continue
                W_arr = get_const(in_names[1])
                if W_arr is None:
                    skipped_ops.append(f"ConvTranspose({node.name}): non-const W")
                    n_skipped += 1; continue
                b_arr = get_const(in_names[2]) if len(in_names) >= 3 else None
                attrs = {a.name: a for a in node.attribute}
                strides = tuple(attrs['strides'].ints) if 'strides' in attrs else (1, 1)
                pads = tuple(attrs['pads'].ints) if 'pads' in attrs else (0, 0, 0, 0)
                output_padding = tuple(attrs['output_padding'].ints) if 'output_padding' in attrs else (0, 0)
                groups = attrs['group'].i if 'group' in attrs else 1
                s = states[primary]
                W = W_arr.astype(np.float64)
                # ONNX ConvTranspose W: (Ci, Co/groups, kH, kW)
                Ci, Co_per_g, kH, kW = W.shape
                Co = Co_per_g * groups
                in_shape = shapes.get(primary)
                if in_shape is None or len(in_shape) != 3:
                    n = s.c.shape[0]
                    spatial = n // Ci
                    Hi_guess = int(np.sqrt(spatial))
                    if Hi_guess * Hi_guess == spatial:
                        Hi = Wi = Hi_guess
                    else:
                        skipped_ops.append(f"ConvTranspose({node.name}): cannot infer shape")
                        n_skipped += 1; continue
                else:
                    _, Hi, Wi = in_shape
                import torch
                import torch.nn.functional as F
                use_cuda = (os.environ.get('HYZOR_FCHZ_USE_CUDA', '0') == '1'
                                  and torch.cuda.is_available())
                dev = torch.device('cuda') if use_cuda else torch.device('cpu')
                W_t = torch.from_numpy(W).to(torch.float32).to(dev)
                b_t = (torch.from_numpy(np.asarray(b_arr).astype(np.float32).reshape(-1)).to(dev)
                          if b_arr is not None else None)
                pad_h, pad_w = pads[0], pads[1]
                stride_h, stride_w = strides
                op_h, op_w = output_padding
                # Center
                c_in = torch.from_numpy(s.c.reshape(1, Ci, Hi, Wi).astype(np.float32)).to(dev)
                c_out = F.conv_transpose2d(c_in, W_t, b_t, stride=strides,
                                                  padding=(pad_h, pad_w),
                                                  output_padding=(op_h, op_w), groups=groups)
                new_c = c_out.detach().cpu().numpy().reshape(-1).astype(np.float64)
                _, Co_p, Ho_p, Wo_p = c_out.shape
                # G chunked
                K_st = s.G.shape[1]
                n_out = Co_p * Ho_p * Wo_p
                new_G = np.zeros((n_out, K_st), dtype=np.float32)
                if K_st > 0:
                    bytes_per_col = n_out * 4
                    chunk_max = max(64, min(K_st, int(2.5e8 // max(bytes_per_col, 1))))
                    G_T = s.G.T.astype(np.float32, copy=False)
                    for start in range(0, K_st, chunk_max):
                        end = min(K_st, start + chunk_max)
                        chunk = G_T[start:end].reshape(end - start, Ci, Hi, Wi)
                        G_chunk_t = torch.from_numpy(chunk).to(dev)
                        G_out_chunk = F.conv_transpose2d(G_chunk_t, W_t, None,
                                                                  stride=strides,
                                                                  padding=(pad_h, pad_w),
                                                                  output_padding=(op_h, op_w),
                                                                  groups=groups)
                        new_G[:, start:end] = G_out_chunk.detach().cpu().numpy().reshape(end - start, -1).T
                        del G_chunk_t, G_out_chunk
                    del G_T
                new_G = new_G.astype(np.float64)
                new_tail_ct = None
                if s.tail_radius is not None:
                    abs_W_t = torch.abs(W_t)
                    tail_in = torch.from_numpy(s.tail_radius.astype(np.float32).reshape(1, Ci, Hi, Wi))
                    tail_out = F.conv_transpose2d(tail_in, abs_W_t, None,
                                                          stride=strides,
                                                          padding=(pad_h, pad_w),
                                                          output_padding=(op_h, op_w),
                                                          groups=groups)
                    new_tail_ct = tail_out.detach().numpy().reshape(-1).astype(np.float64)
                new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                          slack_records=s.slack_records,
                                          tail_radius=new_tail_ct)
                states[node.output[0]] = new_state
                shapes[node.output[0]] = (Co_p, Ho_p, Wo_p)
            elif op == "Pad":
                if primary is None:
                    skipped_ops.append(f"Pad({node.name}): no state")
                    n_skipped += 1; continue
                mode = "constant"
                for a in node.attribute:
                    if a.name == "mode":
                        mode = a.s.decode() if isinstance(a.s, bytes) else a.s
                if mode != "constant":
                    skipped_ops.append(f"Pad({node.name}): mode {mode} unsupported")
                    n_skipped += 1; continue
                pads = None
                if in_names[1] in const_cache:
                    pads = const_cache[in_names[1]].astype(np.int64).reshape(-1)
                else:
                    try:
                        pads = get_const(in_names[1]).astype(np.int64).reshape(-1)
                    except Exception:
                        pads = None
                if pads is None:
                    skipped_ops.append(f"Pad({node.name}): non-const pads")
                    n_skipped += 1; continue
                pad_val = 0.0
                if len(in_names) > 2 and get_const(in_names[2]) is not None:
                    pad_val = float(get_const(in_names[2]).reshape(-1)[0])
                s = states[primary]
                sh = shapes.get(primary)
                if sh is None:
                    skipped_ops.append(f"Pad({node.name}): no shape")
                    n_skipped += 1; continue
                rank = len(sh)
                if pads.size != 2 * rank and pads.size == 2 * (rank + 1):
                    pads = np.concatenate([pads[1:rank+1], pads[rank+2:]])
                pad_pairs = [(int(pads[i]), int(pads[i+rank])) for i in range(rank)]
                c_img = s.c.reshape(sh)
                K_st = s.G.shape[1]
                c_padded = np.pad(c_img, pad_pairs, constant_values=pad_val)
                if K_st > 0:
                    G_img = s.G.reshape(sh + (K_st,))
                    G_padded = np.pad(G_img, pad_pairs + [(0, 0)], constant_values=0.0)
                    new_G = G_padded.reshape(-1, K_st)
                else:
                    new_G = np.zeros((c_padded.size, 0))
                new_shape = c_padded.shape
                new_c = c_padded.reshape(-1)
                new_tail = None
                if s.tail_radius is not None:
                    new_tail = np.pad(s.tail_radius.reshape(sh), pad_pairs,
                                              constant_values=0.0).reshape(-1)
                states[node.output[0]] = FCHZState(c=new_c, G=new_G,
                                                              n_root=s.n_root,
                                                              slack_records=s.slack_records,
                                                              tail_radius=new_tail)
                shapes[node.output[0]] = new_shape
            elif op == "Transpose":
                if primary is None:
                    skipped_ops.append(f"Transpose({node.name}): no state")
                    n_skipped += 1; continue
                perm = None
                for a in node.attribute:
                    if a.name == "perm": perm = list(a.ints)
                s = states[primary]
                sh = shapes.get(primary)
                if sh is None or perm is None:
                    skipped_ops.append(f"Transpose({node.name}): no shape/perm")
                    n_skipped += 1; continue
                rank = len(sh)
                if len(perm) == rank + 1:
                    perm = [p - 1 for p in perm[1:]]
                if len(perm) != rank or sorted(perm) != list(range(rank)):
                    skipped_ops.append(f"Transpose({node.name}): bad perm {perm} for {sh}")
                    n_skipped += 1; continue
                K_st = s.G.shape[1]
                c_img = s.c.reshape(sh).transpose(perm)
                new_shape = tuple(c_img.shape)
                new_c = c_img.reshape(-1)
                if K_st > 0:
                    G_img = s.G.reshape(sh + (K_st,)).transpose(perm + [rank])
                    new_G = G_img.reshape(-1, K_st)
                else:
                    new_G = np.zeros((new_c.size, 0))
                new_tail = None
                if s.tail_radius is not None:
                    new_tail = s.tail_radius.reshape(sh).transpose(perm).reshape(-1)
                states[node.output[0]] = FCHZState(c=new_c, G=new_G,
                                                              n_root=s.n_root,
                                                              slack_records=s.slack_records,
                                                              tail_radius=new_tail)
                shapes[node.output[0]] = new_shape
            elif op == "Unsqueeze":
                # Add singleton dim at given axes. Pure shape op — state stays the
                # same flat array; only shape metadata changes.
                if primary is None:
                    skipped_ops.append(f"Unsqueeze({node.name}): no state")
                    n_skipped += 1; continue
                axes = None
                for a in node.attribute:
                    if a.name == "axes":
                        axes = list(a.ints)
                if axes is None and len(in_names) > 1:
                    arr = get_const(in_names[1])
                    if arr is not None:
                        axes = list(np.asarray(arr).reshape(-1).astype(int))
                if axes is None:
                    skipped_ops.append(f"Unsqueeze({node.name}): no axes")
                    n_skipped += 1; continue
                s = states[primary]
                sh = shapes.get(primary, (s.c.shape[0],))
                new_shape = list(sh)
                # Normalize negative axes and insert in sorted order
                ranks = len(sh) + len(axes)
                axes_norm = sorted([(a + ranks) if a < 0 else a for a in axes])
                for ax in axes_norm:
                    new_shape.insert(ax, 1)
                states[node.output[0]] = s  # data unchanged
                shapes[node.output[0]] = tuple(new_shape)
            elif op == "Squeeze":
                # Remove singleton dims. Same: pure shape op.
                if primary is None:
                    skipped_ops.append(f"Squeeze({node.name}): no state")
                    n_skipped += 1; continue
                axes = None
                for a in node.attribute:
                    if a.name == "axes":
                        axes = list(a.ints)
                if axes is None and len(in_names) > 1:
                    arr = get_const(in_names[1])
                    if arr is not None:
                        axes = list(np.asarray(arr).reshape(-1).astype(int))
                s = states[primary]
                sh = shapes.get(primary, (s.c.shape[0],))
                if axes is None:
                    new_shape = tuple(d for d in sh if d != 1)
                else:
                    axes_norm = sorted([(a + len(sh)) if a < 0 else a
                                                 for a in axes], reverse=True)
                    new_shape = list(sh)
                    for ax in axes_norm:
                        if 0 <= ax < len(new_shape) and new_shape[ax] == 1:
                            new_shape.pop(ax)
                    new_shape = tuple(new_shape)
                states[node.output[0]] = s
                shapes[node.output[0]] = new_shape
            elif op == "Reshape":
                # Pure shape op (state flat already; just track output shape).
                if primary is None:
                    skipped_ops.append(f"Reshape({node.name}): no state")
                    n_skipped += 1; continue
                try:
                    shape_arr = get_const(in_names[1])
                    if shape_arr is None:
                        skipped_ops.append(f"Reshape({node.name}): non-const shape")
                        n_skipped += 1; continue
                    target = list(np.asarray(shape_arr).reshape(-1).astype(int))
                except Exception:
                    skipped_ops.append(f"Reshape({node.name}): bad shape arg")
                    n_skipped += 1; continue
                s = states[primary]
                n = s.c.shape[0]
                # Resolve -1 first (using FULL target, not batch-stripped)
                target_full = list(target)
                neg_idx_full = [i for i, d in enumerate(target_full) if d == -1]
                prod_known_full = 1
                for d in target_full:
                    if d > 0: prod_known_full *= d
                if len(neg_idx_full) == 1:
                    target_full[neg_idx_full[0]] = n // max(prod_known_full, 1)
                # Try with and without batch dim stripping. Prefer the one matching n_state.
                full_prod = int(np.prod([max(d, 1) for d in target_full]))
                stripped = target_full[1:] if (len(target_full) > 1 and target_full[0] in (0, 1)) else target_full
                stripped_prod = int(np.prod([max(d, 1) for d in stripped]))
                if full_prod == n:
                    final_shape = target_full
                elif stripped_prod == n:
                    final_shape = stripped
                else:
                    skipped_ops.append(f"Reshape({node.name}): {target_full} size mismatch {n}")
                    n_skipped += 1; continue
                states[node.output[0]] = s
                shapes[node.output[0]] = tuple(final_shape)
            elif op == "Gather":
                # Constant index Gather: pick rows/cols of state. Sound: like Slice.
                if primary is None:
                    skipped_ops.append(f"Gather({node.name}): no state")
                    n_skipped += 1; continue
                axis = 0
                for a in node.attribute:
                    if a.name == "axis": axis = a.i
                try:
                    indices = get_const(in_names[1])
                    if indices is None:
                        skipped_ops.append(f"Gather({node.name}): non-const indices")
                        n_skipped += 1; continue
                    indices = np.asarray(indices).reshape(-1).astype(int)
                except Exception:
                    skipped_ops.append(f"Gather({node.name}): bad indices")
                    n_skipped += 1; continue
                s = states[primary]
                sh = shapes.get(primary, (s.c.shape[0],))
                if len(sh) == 1:
                    # 1D state — gather rows directly
                    new_c = s.c[indices]
                    new_G = s.G[indices, :]
                    new_tail = (s.tail_radius[indices]
                                      if s.tail_radius is not None else None)
                    states[node.output[0]] = FCHZState(c=new_c, G=new_G,
                                                                  n_root=s.n_root,
                                                                  slack_records=s.slack_records,
                                                                  tail_radius=new_tail)
                    shapes[node.output[0]] = (indices.size,)
                else:
                    # Multi-dim: gather along axis
                    ax = axis if axis >= 0 else len(sh) + axis
                    K_st = s.G.shape[1]
                    c_img = s.c.reshape(sh)
                    new_c_img = np.take(c_img, indices, axis=ax)
                    new_shape = new_c_img.shape
                    new_c = new_c_img.reshape(-1)
                    if K_st > 0:
                        G_img = s.G.reshape(sh + (K_st,))
                        new_G_img = np.take(G_img, indices, axis=ax)
                        new_G = new_G_img.reshape(-1, K_st)
                    else:
                        new_G = np.zeros((new_c.size, 0))
                    new_tail = None
                    if s.tail_radius is not None:
                        tail_img = s.tail_radius.reshape(sh)
                        new_tail = np.take(tail_img, indices, axis=ax).reshape(-1)
                    states[node.output[0]] = FCHZState(c=new_c, G=new_G,
                                                                  n_root=s.n_root,
                                                                  slack_records=s.slack_records,
                                                                  tail_radius=new_tail)
                    shapes[node.output[0]] = new_shape
            elif op == "Slice":
                # Slice with constant starts/ends/axes/steps. State is 1D-flat.
                if primary is None:
                    skipped_ops.append(f"Slice({node.name}): no state input")
                    n_skipped += 1; continue
                def _safe_int64(arr):
                    arr = np.asarray(arr).reshape(-1)
                    if arr.dtype == np.int64:
                        BIG = 1 << 30
                        return np.where(np.abs(arr) > BIG,
                                            np.sign(arr).astype(np.int64) * BIG, arr)
                    BIG = 1 << 30
                    arr_f = arr.astype(np.float64)
                    arr_f = np.where(np.abs(arr_f) > BIG, np.sign(arr_f) * BIG, arr_f)
                    return arr_f.astype(np.int64)
                try:
                    starts = _safe_int64(get_const(in_names[1]))
                    ends = _safe_int64(get_const(in_names[2]))
                    axes = (_safe_int64(get_const(in_names[3]))
                                if len(in_names) > 3 and get_const(in_names[3]) is not None
                                else np.arange(len(starts)))
                    steps = (_safe_int64(get_const(in_names[4]))
                                if len(in_names) > 4 and get_const(in_names[4]) is not None
                                else np.ones(len(starts), dtype=np.int64))
                except Exception:
                    skipped_ops.append(f"Slice({node.name}): non-const params")
                    n_skipped += 1; continue
                if not np.all(steps == 1):
                    skipped_ops.append(f"Slice({node.name}): non-unit steps unsupported")
                    n_skipped += 1; continue
                s = states[primary]
                sh_in = shapes.get(primary)
                if sh_in is None:
                    sh_in = (s.c.shape[0],)
                if len(starts) == 1:
                    ax = int(axes[0]) if axes[0] >= 0 else len(sh_in) + int(axes[0])
                    # Batch dim correction
                    if ax >= len(sh_in):
                        ax = ax - 1
                    if ax < 0 or ax >= len(sh_in):
                        skipped_ops.append(f"Slice({node.name}): axis {axes[0]} invalid for shape {sh_in}")
                        n_skipped += 1; continue
                    st = int(starts[0])
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
                    full_idx = np.arange(int(np.prod(sh_in))).reshape(sh_in)
                    sl = [slice(None)] * len(sh_in)
                    sl[ax] = slice(st, en)
                    keep_idx = full_idx[tuple(sl)].reshape(-1)
                    new_shape = list(sh_in); new_shape[ax] = en - st
                    new_shape = tuple(new_shape)
                    new_state = FCHZState(
                        c=s.c[keep_idx], G=s.G[keep_idx, :],
                        n_root=s.n_root, slack_records=s.slack_records,
                        tail_radius=(s.tail_radius[keep_idx]
                                          if s.tail_radius is not None else None),
                    )
                    states[node.output[0]] = new_state
                    shapes[node.output[0]] = new_shape
                else:
                    skipped_ops.append(f"Slice({node.name}): multi-axis slice not supported")
                    n_skipped += 1; continue
            elif op == "Split":
                # Split a state along an axis into multiple outputs.
                # Pure shape/index op: each output is a Slice of input.
                if primary is None:
                    skipped_ops.append(f"Split({node.name}): no state input")
                    n_skipped += 1; continue
                axis = 0
                for a in node.attribute:
                    if a.name == "axis": axis = a.i
                # split sizes can be from attribute "split" or input[1]
                splits = None
                for a in node.attribute:
                    if a.name == "split":
                        splits = list(a.ints); break
                if splits is None and len(in_names) > 1:
                    arr = get_const(in_names[1])
                    if arr is not None:
                        splits = list(np.asarray(arr).reshape(-1).astype(int))
                s = states[primary]
                sh = shapes.get(primary, (s.c.shape[0],))
                if splits is None:
                    # Equal splits by number of outputs
                    n_out = len(node.output)
                    ax = axis if axis >= 0 else len(sh) + axis
                    if sh[ax] % n_out != 0:
                        skipped_ops.append(f"Split({node.name}): equal split impossible")
                        n_skipped += 1; continue
                    splits = [sh[ax] // n_out] * n_out
                # Apply per-output slice
                ax = axis if axis >= 0 else len(sh) + axis
                offset = 0
                K_st = s.G.shape[1]
                c_img = s.c.reshape(sh)
                G_img = s.G.reshape(sh + (K_st,)) if K_st > 0 else None
                tail_img = (s.tail_radius.reshape(sh)
                                  if s.tail_radius is not None else None)
                for out_name, w in zip(node.output, splits):
                    sl = [slice(None)] * len(sh)
                    sl[ax] = slice(offset, offset + w)
                    new_c_sp = c_img[tuple(sl)].reshape(-1)
                    if K_st > 0:
                        new_G_sp = G_img[tuple(sl + [slice(None)])].reshape(-1, K_st)
                    else:
                        new_G_sp = np.zeros((new_c_sp.size, 0))
                    new_tail_sp = (tail_img[tuple(sl)].reshape(-1)
                                          if tail_img is not None else None)
                    new_shape = list(sh); new_shape[ax] = w
                    states[out_name] = FCHZState(c=new_c_sp, G=new_G_sp,
                                                          n_root=s.n_root,
                                                          slack_records=s.slack_records,
                                                          tail_radius=new_tail_sp)
                    shapes[out_name] = tuple(new_shape)
                    offset += w
            elif op == "Concat":
                # Concat along axis; all inputs must be states (or constants).
                axis = 0
                for a in node.attribute:
                    if a.name == "axis": axis = a.i
                states_to_concat = []
                shapes_to_concat = []
                ok = True
                for nm in in_names:
                    if nm in states:
                        states_to_concat.append(states[nm])
                        shapes_to_concat.append(shapes.get(nm, (states[nm].c.shape[0],)))
                    elif nm in inits or nm in const_cache:
                        arr = get_const(nm)
                        if arr is None:
                            ok = False
                            skipped_ops.append(f"Concat({node.name}): {nm} not found")
                            break
                        arr = arr.astype(np.float64).reshape(-1)
                        # Lift constant to state with zero G/tail
                        # need K width matching others
                        s_const = FCHZState(c=arr, G=np.zeros((arr.shape[0], 1)),
                                                 n_root=0, slack_records=[])
                        states_to_concat.append(s_const)
                        shapes_to_concat.append((arr.shape[0],))
                    else:
                        ok = False
                        skipped_ops.append(f"Concat({node.name}): {nm} not state/const")
                        break
                if not ok:
                    n_skipped += 1; continue
                # Pad all G's to same K (zero-pad shorter ones)
                max_K = max(s.G.shape[1] for s in states_to_concat)
                pad_G = []
                for s in states_to_concat:
                    if s.G.shape[1] < max_K:
                        pad_G.append(np.concatenate(
                            [s.G, np.zeros((s.c.shape[0], max_K - s.G.shape[1]))], axis=1))
                    else:
                        pad_G.append(s.G)
                new_c = np.concatenate([s.c for s in states_to_concat], axis=0)
                new_G = np.concatenate(pad_G, axis=0)
                # Tails: keep per-row, concatenate; zeros for those without tail
                any_tail = any(s.tail_radius is not None for s in states_to_concat)
                new_tail_cc = None
                if any_tail:
                    tails = []
                    for s in states_to_concat:
                        if s.tail_radius is not None:
                            tails.append(s.tail_radius)
                        else:
                            tails.append(np.zeros(s.c.shape[0]))
                    new_tail_cc = np.concatenate(tails, axis=0)
                # Merge slack_records (preserve all)
                merged_records = []
                seen_layers = set()
                for s in states_to_concat:
                    for rec in s.slack_records:
                        if rec.layer_index not in seen_layers:
                            merged_records.append(rec)
                            seen_layers.add(rec.layer_index)
                new_state = FCHZState(c=new_c, G=new_G,
                                          n_root=states_to_concat[0].n_root,
                                          slack_records=merged_records,
                                          tail_radius=new_tail_cc)
                states[node.output[0]] = new_state
                # Output shape: flat concat for now
                out_dim = sum(int(np.prod(sh)) for sh in shapes_to_concat)
                shapes[node.output[0]] = (out_dim,)
            elif op == "MaxPool":
                # MaxPool is NONLINEAR — replace with relaxation: bound by box.
                # For HZ: each output is in [min, max] over the kernel window of inputs.
                # Sound but loose: take max over each window's box-bounds.
                if primary is None:
                    skipped_ops.append(f"MaxPool({node.name}): no state input")
                    n_skipped += 1; continue
                attrs = {a.name: a for a in node.attribute}
                kernel = tuple(attrs['kernel_shape'].ints) if 'kernel_shape' in attrs else (2, 2)
                strides = tuple(attrs['strides'].ints) if 'strides' in attrs else kernel
                pads = tuple(attrs['pads'].ints) if 'pads' in attrs else (0, 0, 0, 0)
                s = states[primary]
                sh = shapes.get(primary)
                if sh is None or len(sh) != 3:
                    skipped_ops.append(f"MaxPool({node.name}): no 3D shape")
                    n_skipped += 1; continue
                Ci, Hi, Wi = sh
                kH, kW = kernel
                sH, sW = strides
                pH, pW = pads[0], pads[1]
                Ho = (Hi + 2*pH - kH) // sH + 1
                Wo = (Wi + 2*pW - kW) // sW + 1
                # Sound bound: per output element is max over kernel window.
                # Compute c_per_pixel ± rad_per_pixel per input.
                rad_in = np.abs(s.G).sum(axis=1)
                if s.tail_radius is not None:
                    rad_in = rad_in + s.tail_radius
                c_in_img = s.c.reshape(Ci, Hi, Wi)
                rad_in_img = rad_in.reshape(Ci, Hi, Wi)
                # Pad
                if pH or pW:
                    c_in_pad = np.pad(c_in_img, ((0,0),(pH,pH),(pW,pW)),
                                            constant_values=-1e30)
                    rad_in_pad = np.pad(rad_in_img, ((0,0),(pH,pH),(pW,pW)),
                                              constant_values=0)
                else:
                    c_in_pad = c_in_img; rad_in_pad = rad_in_img
                new_c_arr = np.zeros((Ci, Ho, Wo))
                new_tail_arr = np.zeros((Ci, Ho, Wo))
                for ho in range(Ho):
                    for wo in range(Wo):
                        hs = ho * sH; ws = wo * sW
                        win_c = c_in_pad[:, hs:hs+kH, ws:ws+kW]
                        win_r = rad_in_pad[:, hs:hs+kH, ws:ws+kW]
                        win_lo = (win_c - win_r).reshape(Ci, -1)
                        win_hi = (win_c + win_r).reshape(Ci, -1)
                        # max of window: lower=max(lo's), upper=max(hi's)
                        out_lo = win_lo.max(axis=1)
                        out_hi = win_hi.max(axis=1)
                        new_c_arr[:, ho, wo] = (out_lo + out_hi) / 2
                        new_tail_arr[:, ho, wo] = (out_hi - out_lo) / 2
                new_c = new_c_arr.reshape(-1)
                # MaxPool destroys G dependency entirely (relaxed to box). Sound but loose.
                K_st = s.G.shape[1]
                new_G = np.zeros((Ci*Ho*Wo, K_st)) if K_st > 0 else s.G
                new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                          slack_records=s.slack_records,
                                          tail_radius=new_tail_arr.reshape(-1))
                states[node.output[0]] = new_state
                shapes[node.output[0]] = (Ci, Ho, Wo)
            elif op == "GlobalAveragePool":
                # GAP: (C, H, W) -> (C,) via mean over H, W. Linear op.
                if primary is None:
                    skipped_ops.append(f"GAP({node.name}): no state")
                    n_skipped += 1; continue
                s = states[primary]
                sh = shapes.get(primary)
                if sh is None or len(sh) != 3:
                    skipped_ops.append(f"GAP({node.name}): no 3D shape")
                    n_skipped += 1; continue
                Ci, Hi, Wi = sh
                spatial = Hi * Wi
                # Average per channel
                c_img = s.c.reshape(Ci, Hi, Wi)
                new_c = c_img.mean(axis=(1, 2))
                G_img = s.G.reshape(Ci, Hi, Wi, s.G.shape[1])
                new_G = G_img.mean(axis=(1, 2))
                new_tail = None
                if s.tail_radius is not None:
                    new_tail = s.tail_radius.reshape(Ci, Hi, Wi).mean(axis=(1, 2))
                new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                          slack_records=s.slack_records,
                                          tail_radius=new_tail)
                states[node.output[0]] = new_state
                shapes[node.output[0]] = (Ci,)
            elif op == "BatchNormalization":
                # Inference BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
                # = a * x + b where a = gamma / sqrt(var + eps), b = beta - a*mean
                if primary is None:
                    skipped_ops.append(f"BN({node.name}): no state input")
                    n_skipped += 1; continue
                gamma = get_const(in_names[1])
                beta_v = get_const(in_names[2])
                mean = get_const(in_names[3])
                var = get_const(in_names[4])
                if any(x is None for x in [gamma, beta_v, mean, var]):
                    skipped_ops.append(f"BN({node.name}): non-const params")
                    n_skipped += 1; continue
                attrs = {a.name: a for a in node.attribute}
                eps = attrs['epsilon'].f if 'epsilon' in attrs else 1e-5
                s = states[primary]
                a = gamma.reshape(-1) / np.sqrt(var.reshape(-1) + eps)
                b = beta_v.reshape(-1) - a * mean.reshape(-1)
                # For 2D state, a/b apply per-channel; need to broadcast
                # Assume state.c is (Co * H * W,) and a is (Co,)
                Co = a.shape[0]
                spatial = s.c.shape[0] // Co
                a_full = np.repeat(a, spatial)
                b_full = np.repeat(b, spatial)
                new_c = s.c * a_full + b_full
                # In-place scaling on G to avoid allocating a new (n, K) array
                if s.G.dtype != np.float64:
                    new_G = s.G.astype(np.float64, copy=True)
                else:
                    new_G = s.G.copy()
                new_G *= a_full[:, None]
                # Propagate tail through BN: scaled by |a|
                new_tail_bn = None
                if s.tail_radius is not None:
                    new_tail_bn = np.abs(a_full) * s.tail_radius
                new_state = FCHZState(c=new_c, G=new_G, n_root=s.n_root,
                                          slack_records=s.slack_records,
                                          tail_radius=new_tail_bn)
                states[node.output[0]] = new_state
                # BN preserves shape
                if primary in shapes:
                    shapes[node.output[0]] = shapes[primary]
            elif op == "Sigmoid":
                # Linear chord relaxation on sigmoid
                # sigma(z) is monotonic; chord from (l, sigma(l)) to (u, sigma(u))
                # Upper bound: chord above sigma in [l, u] when l < 0 < u
                # Use box bound: state.c, state.G stay; just clip the c via chord midpoint
                if primary is None:
                    skipped_ops.append(f"Sigmoid({node.name}): no state input")
                    n_skipped += 1
                    continue
                s = states[primary]
                rad = np.abs(s.G).sum(axis=1)
                l = s.c - rad
                u = s.c + rad
                # Sigmoid bounds
                def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
                sl = sig(l); su = sig(u)
                # Chord: y ~= a*z + b where a = (su-sl)/(u-l), b = sl - a*l (using endpoints)
                # For sound LP bound: y in [sl, su] (interval), and y ≤ linear chord
                # Just use centered interval approx: y center = (sl+su)/2, slack = (su-sl)/2
                new_c = (sl + su) / 2
                new_rad = (su - sl) / 2
                # Add as fresh independent slack column
                K = s.G.shape[1]
                n = s.c.shape[0]
                new_G = np.concatenate([s.G * 0, np.diag(new_rad)], axis=1)
                # Actually we lose all input dependency. Sound but loose.
                # For our walker purposes, the SECOND ReLU layer's slack records would
                # be invalidated. Skip for now; mark unsupported on benches that need it.
                skipped_ops.append(f"Sigmoid({node.name}): forward chord not implemented "
                                       f"(would lose multi-layer dependency)")
                n_skipped += 1
                continue
            else:
                skipped_ops.append(f"{op}({node.name}): not supported in fchz walker")
                n_skipped += 1
                continue
            n_processed += 1
            # Sparse-slack compression: if requested + this op produced a new
            # state, compress G when it exceeds the column budget. Sound by
            # construction (see SPARSE_SLACK_DESIGN.md §6).
            if G_max_cols is not None and len(node.output) > 0 and node.output[0] in states:
                s_new = states[node.output[0]]
                if s_new.G.shape[1] > G_max_cols:
                    states[node.output[0]] = compress_g_to_tail(s_new, G_max_cols)
            # Free states whose consumers are all processed
            _decref_inputs(in_names)
        except Exception as e:
            skipped_ops.append(f"{op}({node.name}): {type(e).__name__}: {str(e)[:60]}")
            n_skipped += 1

    if out_name not in states:
        raise RuntimeError(f"Output {out_name} not reached. Skipped: {skipped_ops[:3]}")

    return FCHZWalkResult(
        state=states[out_name],
        n_processed=n_processed,
        n_skipped=n_skipped,
        skipped_ops=skipped_ops,
        output_name=out_name,
    )


def main():
    print("=== FCHZ walker self-test ===\n")
    import signal
    def _to(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _to)
    from research.canonical_provenance import load_instance
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.fc_hz_state import f1_last_relu_lp_ub, fc_hz_lp_ub, hz_closed_form_ub
    from research.sc_hz.milp_multilayer_v2 import fc_hz_milp_v2_ub
    import time

    # Test on metaroom iid 22 (known MILP CERT)
    targets = [
        ('metaroom_2023', 22),
        ('dist_shift_2023', 64),
        ('acasxu_2023', 0),
        ('acasxu_2023', 50),
    ]
    for bench, iid in targets:
        print(f"\n--- {bench} iid {iid} ---")
        signal.alarm(30)
        try:
            onnx_p, vnn_p = load_instance(bench, iid)
            m = onnx.load(str(onnx_p))
            init_names = {x.name for x in m.graph.initializer}
            din = [x for x in m.graph.input if x.name not in init_names][0]
            dims = [d.dim_value if d.dim_value > 0 else 1
                    for d in din.type.tensor_type.shape.dim]
            n_in = int(np.prod(dims[1:])) if dims[0] in (0, 1) else int(np.prod(dims))
            od = [d.dim_value if d.dim_value > 0 else 1
                  for d in m.graph.output[0].type.tensor_type.shape.dim]
            n_cls = int(np.prod(od[1:])) if len(od) > 1 else od[0]
            lb, ub, unsafe = parse_vnnlib(str(vnn_p), n_in, n_cls)
            t0 = time.perf_counter()
            r = forward_fchz(str(onnx_p), lb, ub)
            wall = time.perf_counter() - t0
            signal.alarm(0)
            print(f"  walker {wall:.1f}s, processed={r.n_processed}, skipped={r.n_skipped}")
            if r.n_skipped > 0:
                print(f"  skipped[:2]: {r.skipped_ops[:2]}")
                continue
            n_layers = len(r.state.slack_records)
            print(f"  FCHZState: n_layers={n_layers}, K={r.state.K}, n_out={r.state.n}")
            # Compute bounds
            f1_max = -float('inf'); fc_max = -float('inf'); milp_max = -float('inf'); hz_max = -float('inf')
            for d, t, _ in unsafe:
                hz_ub = hz_closed_form_ub(r.state, d)
                f1_ub = f1_last_relu_lp_ub(r.state, d)
                fc_ub, _ = fc_hz_lp_ub(r.state, d)
                m_ub, _ = fc_hz_milp_v2_ub(r.state, d, K_max_per_layer=10)
                hz_ex = hz_ub - float(t)
                f1_ex = f1_ub - float(t)
                fc_ex = fc_ub - float(t)
                m_ex = (m_ub - float(t)) if m_ub is not None else float('inf')
                if hz_ex > hz_max: hz_max = hz_ex
                if f1_ex > f1_max: f1_max = f1_ex
                if fc_ex > fc_max: fc_max = fc_ex
                if m_ex > milp_max: milp_max = m_ex
            print(f"  HZ={hz_max:+.3e}, F1={f1_max:+.3e}, FC-HZ={fc_max:+.3e}, MILP={milp_max:+.3e}")
            if milp_max < 0: print(f"  *** {bench} iid {iid} MILP CERT! ***")
        except Exception as e:
            signal.alarm(0); print(f"  {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
