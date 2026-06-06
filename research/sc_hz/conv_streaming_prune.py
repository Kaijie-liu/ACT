"""S2 v3 (advisor 2026-06-05 corrections): streaming-prune Conv with
output-L1 priority + K_target >= n_root enforcement.

Changes from v2:
  - K_target is FORCED to be >= n_root_columns (input-coord lineage cols).
    Smaller K_target silently demotes root columns into tail, which breaks
    witness decode (the forward-coefficient decoder reads sign(d_out·G_out)
    on the input-coord columns to construct x_star).
  - Priority for slack-column ranking is now OUTPUT-L1 (computed AFTER the
    Conv on each chunk), not input-L1. After Conv, the relevant magnitude
    is the per-OUTPUT-column L1, which is what determines its contribution
    to subsequent LP UB / downstream ops.
  - Root columns are ALWAYS kept (origin >= 0), per Phase A design lock.
    Slack columns (origin == -1) are ranked by output-L1 and the bottom
    `K_target - n_root` are folded into tail.

Memory profile per layer (n_in_flat = Ci*Hi*Wi, n_out_flat = Co*Ho*Wo):
  - peak transient:  chunk_size × (n_in_flat + n_out_flat) × 8 bytes
  - peak resident:    n_out_flat × K_target × 8 bytes (kept matrix)
                       + n_out_flat × 8 bytes (tail vector)

Soundness invariants (tested in test_conv_streaming_prune_soundness.py):
  - K_target >= ng_in → identity (no prune, bit-equal chunked output)
  - K_target < ng_in → LP UB ≥ no-prune LP UB for any direction
  - Brute-force samples from no-prune set lie in streaming-prune box
  - Root-coord generators NEVER dropped
  - chunk_size does not affect result (modulo float epsilon in tail)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research.sc_hz.prune import PrunedState


def _torchify(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float64)


def _conv_chunk_compute(
    G_slice: np.ndarray,  # (n_in_flat, cs)
    W_t: torch.Tensor,
    input_shape: Tuple[int, int, int],
    stride: int, padding: int, groups: int,
) -> np.ndarray:
    """Compute Conv on a chunk of generator columns. Returns (n_out_flat, cs)."""
    Ci, Hi, Wi = input_shape
    cs = G_slice.shape[1]
    chunk_input = G_slice.T.reshape(cs, Ci, Hi, Wi)
    chunk_t = _torchify(chunk_input)
    chunk_out = F.conv2d(chunk_t, W_t, None,
                          stride=stride, padding=padding, groups=groups)
    out_np = chunk_out.detach().numpy().reshape(cs, -1)
    return out_np.T  # (n_out_flat, cs)


def apply_conv2d_streaming_prune(
    state: PrunedState,
    W: np.ndarray,
    b: Optional[np.ndarray],
    input_shape: Tuple[int, int, int],
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    chunk_size: int = 256,
    K_target: int = 100000,
    enforce_root_minimum: bool = True,
) -> Tuple[PrunedState, Tuple[int, int, int]]:
    """Conv2D forward with streaming-prune + output-L1 priority.

    Args:
      state: incoming PrunedState
      W: conv weights (C_out, C_in/groups, kH, kW)
      b: optional bias (C_out,)
      input_shape: (C_in, H_in, W_in)
      stride, padding, groups: as in F.conv2d
      chunk_size: per-chunk batched conv batch size
      K_target: max output kept columns. If less than n_root_in,
        auto-promoted to n_root_in (if enforce_root_minimum=True) or
        function raises (if enforce_root_minimum=False).
      enforce_root_minimum: if True (default), silently promote K_target
        to n_root when smaller. Set False to require caller to pick a
        valid K_target.

    Returns (new_state, output_shape).
    """
    Ci, Hi, Wi = input_shape
    n_in_flat = Ci * Hi * Wi
    if state.c.shape != (n_in_flat,):
        raise ValueError(f"state.c shape {state.c.shape} mismatch input_shape")

    W_t = _torchify(W)
    Co, _, kH, kW = W.shape
    b_t = (_torchify(np.asarray(b, dtype=np.float64).reshape(-1))
           if b is not None else None)

    # Probe output shape
    probe = F.conv2d(
        torch.zeros((1, Ci, Hi, Wi), dtype=torch.float64),
        W_t, None, stride=stride, padding=padding, groups=groups,
    )
    Co_p, Ho_p, Wo_p = int(probe.shape[1]), int(probe.shape[2]), int(probe.shape[3])
    n_out_flat = Co_p * Ho_p * Wo_p

    # 1. Center: single Conv with bias
    c_in_4d = _torchify(state.c).reshape(1, Ci, Hi, Wi)
    new_c_4d = F.conv2d(c_in_4d, W_t, b_t,
                         stride=stride, padding=padding, groups=groups)
    new_c = new_c_4d.detach().numpy().reshape(-1)

    K_old = state.G_kept.shape[1]
    # Origin / root detection
    origin = state.metadata.get("input_coord_origin", None)
    if origin is None:
        # Best-effort: first n_in cols are root
        origin = -np.ones(K_old, dtype=np.int64)
        origin[:min(K_old, n_in_flat)] = np.arange(min(K_old, n_in_flat), dtype=np.int64)

    root_mask = origin >= 0
    n_root = int(root_mask.sum())

    # K_target enforcement
    if K_target < n_root:
        if enforce_root_minimum:
            K_target = n_root  # promote silently
        else:
            raise ValueError(
                f"K_target={K_target} < n_root={n_root}. Root columns must "
                f"all be kept; either raise K_target or set "
                f"enforce_root_minimum=True."
            )

    # 2. Tail propagation through |W|
    if state.tail_radius is not None:
        abs_W_t = torch.abs(W_t)
        abs_tail_4d = _torchify(np.abs(state.tail_radius)).reshape(1, Ci, Hi, Wi)
        out_tail = F.conv2d(abs_tail_4d, abs_W_t, None,
                              stride=stride, padding=padding, groups=groups)
        in_tail_after_W = out_tail.detach().numpy().reshape(-1)
    else:
        in_tail_after_W = np.zeros(n_out_flat, dtype=np.float64)

    # 3. K_old == 0 path: nothing to convolve
    if K_old == 0:
        return PrunedState(
            c=new_c, G_kept=np.empty((n_out_flat, 0), dtype=np.float64),
            tail_radius=in_tail_after_W, metadata={"input_coord_origin": np.empty(0, dtype=np.int64)},
        ), (Co_p, Ho_p, Wo_p)

    # 4. No-prune path: K_target >= K_old
    if K_target >= K_old:
        new_G = np.empty((n_out_flat, K_old), dtype=np.float64)
        for start in range(0, K_old, chunk_size):
            end = min(start + chunk_size, K_old)
            new_G[:, start:end] = _conv_chunk_compute(
                state.G_kept[:, start:end], W_t, input_shape,
                stride, padding, groups,
            )
        return PrunedState(
            c=new_c, G_kept=new_G, tail_radius=in_tail_after_W,
            metadata={"input_coord_origin": origin.copy()},
        ), (Co_p, Ho_p, Wo_p)

    # 5. Streaming-prune path with output-L1 priority for slack cols
    # n_root cols always kept; remaining (K_target - n_root) slots fill by
    # top-output-L1 slack cols; rest fold to tail.
    slack_budget = K_target - n_root
    slack_indices_old = np.where(~root_mask)[0]
    n_slack = len(slack_indices_old)

    # Step 5a: Compute Conv on all SLACK columns in chunks to get output-L1
    # We need to know L1 BEFORE deciding which to keep. To avoid full output
    # buffer, do per-chunk compute + accumulate (L1, original_idx) tuples.
    slack_L1 = np.zeros(n_slack, dtype=np.float64)
    # Need to keep slack output cols around in case they make the top-K.
    # Strategy: two-pass. First pass computes L1 only; second pass selects
    # and stores top by L1 + folds rest to tail.
    for start in range(0, n_slack, chunk_size):
        end = min(start + chunk_size, n_slack)
        idxs = slack_indices_old[start:end]
        chunk_out = _conv_chunk_compute(
            state.G_kept[:, idxs], W_t, input_shape,
            stride, padding, groups,
        )
        slack_L1[start:end] = np.abs(chunk_out).sum(axis=0)
        del chunk_out

    # Step 5b: pick top slack_budget by slack_L1
    if slack_budget >= n_slack:
        slack_keep_mask = np.ones(n_slack, dtype=bool)
        slack_drop_mask = np.zeros(n_slack, dtype=bool)
    else:
        slack_order = np.argsort(-slack_L1)
        slack_keep_idx = slack_order[:slack_budget]
        slack_drop_idx = slack_order[slack_budget:]
        slack_keep_mask = np.zeros(n_slack, dtype=bool)
        slack_keep_mask[slack_keep_idx] = True
        slack_drop_mask = ~slack_keep_mask

    n_slack_kept = int(slack_keep_mask.sum())
    K_actual = n_root + n_slack_kept

    # Step 5c: Allocate output and fill in by streaming Conv on:
    # (a) root cols (all)
    # (b) slack cols that are kept (slack_keep_mask)
    # (c) compute tail_drop for slack_drop_mask cols
    new_G = np.empty((n_out_flat, K_actual), dtype=np.float64)
    new_origin = np.empty(K_actual, dtype=np.int64)
    new_tail_drop = np.zeros(n_out_flat, dtype=np.float64)

    # (a) Root columns
    root_indices_old = np.where(root_mask)[0]
    write_pos = 0
    for start in range(0, len(root_indices_old), chunk_size):
        end = min(start + chunk_size, len(root_indices_old))
        idxs = root_indices_old[start:end]
        chunk_out = _conv_chunk_compute(
            state.G_kept[:, idxs], W_t, input_shape,
            stride, padding, groups,
        )
        cs = end - start
        new_G[:, write_pos:write_pos+cs] = chunk_out
        new_origin[write_pos:write_pos+cs] = origin[idxs]
        write_pos += cs
        del chunk_out

    # (b) Kept slack columns
    slack_indices_kept_old = slack_indices_old[slack_keep_mask]
    for start in range(0, len(slack_indices_kept_old), chunk_size):
        end = min(start + chunk_size, len(slack_indices_kept_old))
        idxs = slack_indices_kept_old[start:end]
        chunk_out = _conv_chunk_compute(
            state.G_kept[:, idxs], W_t, input_shape,
            stride, padding, groups,
        )
        cs = end - start
        new_G[:, write_pos:write_pos+cs] = chunk_out
        new_origin[write_pos:write_pos+cs] = origin[idxs]
        write_pos += cs
        del chunk_out

    # (c) Dropped slack columns: fold to tail
    slack_indices_dropped_old = slack_indices_old[slack_drop_mask]
    for start in range(0, len(slack_indices_dropped_old), chunk_size):
        end = min(start + chunk_size, len(slack_indices_dropped_old))
        idxs = slack_indices_dropped_old[start:end]
        chunk_out = _conv_chunk_compute(
            state.G_kept[:, idxs], W_t, input_shape,
            stride, padding, groups,
        )
        new_tail_drop += np.abs(chunk_out).sum(axis=1)
        del chunk_out

    new_tail = in_tail_after_W + new_tail_drop

    new_state = PrunedState(
        c=new_c, G_kept=new_G, tail_radius=new_tail,
        metadata={"input_coord_origin": new_origin},
    )
    return new_state, (Co_p, Ho_p, Wo_p)


@dataclass
class StreamingMemoryProfile:
    chunk_size: int
    K_target: int
    transient_input_bytes: int
    transient_output_bytes: int
    resident_kept_bytes: int
    resident_tail_bytes: int

    @property
    def transient_gb(self) -> float:
        return (self.transient_input_bytes + self.transient_output_bytes) / (1024**3)

    @property
    def resident_gb(self) -> float:
        return (self.resident_kept_bytes + self.resident_tail_bytes) / (1024**3)


def estimate_streaming_memory(
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    chunk_size: int, K_target: int,
) -> StreamingMemoryProfile:
    Ci, Hi, Wi = input_shape
    Co, Ho, Wo = output_shape
    n_out = Co * Ho * Wo
    transient_in = 8 * chunk_size * Ci * Hi * Wi
    transient_out = 8 * chunk_size * n_out
    resident_kept = 8 * n_out * K_target
    resident_tail = 8 * n_out
    return StreamingMemoryProfile(
        chunk_size=chunk_size, K_target=K_target,
        transient_input_bytes=transient_in,
        transient_output_bytes=transient_out,
        resident_kept_bytes=resident_kept,
        resident_tail_bytes=resident_tail,
    )


def n_root_in_state(state: PrunedState) -> int:
    """Number of input-coord (root) generator columns in a state."""
    origin = state.metadata.get("input_coord_origin", None)
    if origin is None:
        return 0
    return int((origin >= 0).sum())
