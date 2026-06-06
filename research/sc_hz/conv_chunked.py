"""S2 Day 1: chunked Conv propagation for forward HZ.

Per advisor 2026-06-05 path-A plan: lift the memory ceiling that blocked
tinyimagenet + cifar deep-variant pilots. The existing `apply_conv2d` in
ops.py iterates generator columns one at a time but still allocates the
full `(C_out * H_out * W_out, K)` output before-the-fact; peak memory at
deep cifar variants reached ~80 GB per iid.

This module provides `apply_conv2d_chunked`:
  - Processes generator columns in chunks of `chunk_size` (default 256)
  - Within each chunk, batches `F.conv2d` over `(chunk_size, C_in, H_in, W_in)`
  - Writes results directly into the pre-allocated output slice
  - Optionally separates ROOT generators (input-coord lineage) from
     RELU-SLACK generators for downstream optimization

Numerical equivalence with `apply_conv2d` is guaranteed modulo float
summation order (which Conv does not introduce — each column is a
deterministic linear transform of the input column).

G10 enforced via per-process RLIMIT_AS at the driver level (this module
is import-clean and does NOT call setrlimit on its own).

Unit tests live in `tests/test_conv_chunked_parity.py`.
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


def apply_conv2d_chunked(
    state: PrunedState,
    W: np.ndarray,
    b: Optional[np.ndarray],
    input_shape: Tuple[int, int, int],
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    chunk_size: int = 256,
) -> Tuple[PrunedState, Tuple[int, int, int]]:
    """Conv2D forward propagation with generator-column chunking.

    Memory profile per chunk:
      input  buf: (chunk_size, C_in, H_in, W_in) * 8 bytes
      output buf: (chunk_size, C_out, H_out, W_out) * 8 bytes
    Total transient: ~chunk_size * (Ci*Hi*Wi + Co*Ho*Wo) * 8.

    Output state has same total memory as `apply_conv2d` (the full
    `(n_out, K_kept)` G_kept), but PEAK is bounded by the per-chunk
    transient. For Conv layers where K_kept is the dominant memory term,
    this can reduce peak by `(K_kept / chunk_size)`x.

    Args:
      state: incoming PrunedState
      W: conv weights, shape (C_out, C_in/groups, kH, kW)
      b: optional bias, shape (C_out,) or None
      input_shape: (C_in, H_in, W_in) — caller must pass; the PrunedState
                    stores flat vectors
      stride, padding, groups: as in F.conv2d
      chunk_size: number of generator columns processed per chunk
                   (default 256). Pure peak-memory knob — does NOT affect
                   numerical output (modulo cross-column sum order which
                   does not arise in Conv).

    Returns (new_state, output_shape).
    """
    Ci, Hi, Wi = input_shape
    n_in_flat = Ci * Hi * Wi
    if state.c.shape != (n_in_flat,):
        raise ValueError(
            f"state.c shape {state.c.shape} mismatch input_shape "
            f"{input_shape} -> flat {n_in_flat}"
        )

    W_t = _torchify(W)
    Co, _, kH, kW = W.shape
    b_t = (_torchify(np.asarray(b, dtype=np.float64).reshape(-1))
           if b is not None else None)

    # Probe output shape via a zero-input conv (avoids materializing arange)
    probe = F.conv2d(
        torch.zeros((1, Ci, Hi, Wi), dtype=torch.float64),
        W_t, None, stride=stride, padding=padding, groups=groups,
    )
    Co_p, Ho_p, Wo_p = int(probe.shape[1]), int(probe.shape[2]), int(probe.shape[3])
    n_out_flat = Co_p * Ho_p * Wo_p

    # --- Center: single conv with bias
    c_in_4d = _torchify(state.c).reshape(1, Ci, Hi, Wi)
    new_c_4d = F.conv2d(c_in_4d, W_t, b_t,
                         stride=stride, padding=padding, groups=groups)
    new_c = new_c_4d.detach().numpy().reshape(-1)

    # --- Generators: chunked (no bias)
    K = state.G_kept.shape[1]
    new_G = np.empty((n_out_flat, K), dtype=np.float64)
    if K > 0:
        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            cs = end - start
            # G[:, start:end] has shape (n_in_flat, cs); reshape to
            # (cs, Ci, Hi, Wi) for batched conv
            chunk_input = state.G_kept[:, start:end].T.reshape(cs, Ci, Hi, Wi)
            chunk_t = _torchify(chunk_input)
            chunk_out = F.conv2d(chunk_t, W_t, None,
                                   stride=stride, padding=padding, groups=groups)
            chunk_out_np = chunk_out.detach().numpy().reshape(cs, n_out_flat)
            new_G[:, start:end] = chunk_out_np.T
            # Free per-chunk torch tensor immediately
            del chunk_t, chunk_out, chunk_out_np

    # --- Tail: |W| @ tail propagated via conv with abs(W) on abs(tail) image
    new_tail: Optional[np.ndarray] = None
    if state.tail_radius is not None:
        abs_W_t = torch.abs(W_t)
        abs_tail_4d = _torchify(np.abs(state.tail_radius)).reshape(1, Ci, Hi, Wi)
        out_tail = F.conv2d(abs_tail_4d, abs_W_t, None,
                              stride=stride, padding=padding, groups=groups)
        new_tail = out_tail.detach().numpy().reshape(-1)

    new_state = PrunedState(
        c=new_c, G_kept=new_G, tail_radius=new_tail,
        metadata=dict(state.metadata),
    )
    return new_state, (Co_p, Ho_p, Wo_p)


@dataclass
class ChunkedMemoryProfile:
    """Per-call estimate of peak transient memory."""
    chunk_size: int
    transient_input_bytes: int
    transient_output_bytes: int
    total_transient_bytes: int

    @property
    def total_transient_gb(self) -> float:
        return self.total_transient_bytes / (1024**3)


def estimate_chunk_memory(
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    chunk_size: int,
) -> ChunkedMemoryProfile:
    """Estimate the peak transient memory per chunk in bytes.

    Useful for choosing `chunk_size` against a budget.
    """
    Ci, Hi, Wi = input_shape
    Co, Ho, Wo = output_shape
    in_b = 8 * chunk_size * Ci * Hi * Wi
    out_b = 8 * chunk_size * Co * Ho * Wo
    return ChunkedMemoryProfile(
        chunk_size=chunk_size,
        transient_input_bytes=in_b,
        transient_output_bytes=out_b,
        total_transient_bytes=in_b + out_b,
    )


def adaptive_chunk_size(
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    budget_gb: float = 4.0,
    min_chunk: int = 16,
    max_chunk: int = 1024,
) -> int:
    """Pick chunk_size so the per-chunk transient fits in `budget_gb`.

    Returns at least `min_chunk` even if the budget is tight (smaller would
    be over-fragmented). Returns at most `max_chunk` (diminishing returns
    on PyTorch batching beyond this scale).
    """
    Ci, Hi, Wi = input_shape
    Co, Ho, Wo = output_shape
    per_col_bytes = 8 * (Ci * Hi * Wi + Co * Ho * Wo)
    budget_bytes = budget_gb * (1024**3)
    cs = max(min_chunk, int(budget_bytes // max(1, per_col_bytes)))
    return min(cs, max_chunk)
