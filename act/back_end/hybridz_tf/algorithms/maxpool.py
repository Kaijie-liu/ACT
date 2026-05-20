"""HZono MaxPool 2D with stable-winner row preservation.

Faithful port of HyZor's ``HybridZonotope.max_pool_node_evaluate``
(``HybridZonotope.py`` :6605-:6744). Solver-free algorithm:

  1. Rearrange the input HZ rows into pooling-window blocks.
  2. Cheap interval-hull bounds per block (no LP).
  3. If a block has a STABLE WINNER (``lb_winner >= ub_others`` for
     every other entry in the block), output the winner row exactly —
     full generator correlation preserved.
  4. Otherwise relax that block to an interval ``[max(lb), max(ub)]``
     with one fresh independent continuous generator.

Adds at most ``k_uns`` new continuous generators (one per unstable
block) and 0 binary / 0 constraint rows. This is the same shape as the
DeepZ-style max-pool relaxation but tighter on the dominant-pixel case.
"""
from __future__ import annotations
from typing import Tuple

import torch

from act.back_end.solver.solver_hz import HZono


__all__ = ["hz_maxpool2d"]


# ---------------------------------------------------------------------------
# Helpers (mirrors HyZor's HybridZonotope._take_rows and
# _build_maxpool_index_map_chw)
# ---------------------------------------------------------------------------


def _take_rows(hz: HZono, idx: torch.Tensor) -> HZono:
    """Slice ``hz`` to a new HZono with rows at ``idx`` (constraints unchanged).

    Mirrors ``HybridZonotope._take_rows`` (HZ :6569).
    """
    idx = idx.to(device=hz.c.device, dtype=torch.long).view(-1)
    new_c = hz.c[idx, :]
    if hz.Gc.numel():
        new_Gc = hz.Gc[idx, :]
    else:
        new_Gc = hz.Gc.new_zeros((idx.numel(), 0))
    if hz.Gb.numel():
        new_Gb = hz.Gb[idx, :]
    else:
        new_Gb = hz.Gb.new_zeros((idx.numel(), 0))
    return HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
        eq_mask=hz.eq_mask,
    )


def _build_maxpool_index_map_chw(
    H_in: int, W_in: int, C_in: int,
    window_h: int, window_w: int,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
) -> Tuple[torch.Tensor, int, int, int]:
    """Return ``(idx, H_out, W_out, block_size)`` for the CHW layout.

    ``idx`` is a flat ``[num_blocks * block_size]`` Long tensor: entry
    ``i`` is the source row in the flattened CHW HZ for the ``i``-th
    block-position pair. ``-1`` entries represent padding positions
    (the caller appends a zero-valued pad row).
    """
    H_out = (H_in - window_h + 2 * pad_h) // stride_h + 1
    W_out = (W_in - window_w + 2 * pad_w) // stride_w + 1
    block_size = window_h * window_w

    idx = []
    for co in range(C_in):
        base_c = co * H_in * W_in
        for h_out in range(H_out):
            in_h0 = h_out * stride_h - pad_h
            for w_out in range(W_out):
                in_w0 = w_out * stride_w - pad_w
                for kh in range(window_h):
                    h_in = in_h0 + kh
                    for kw in range(window_w):
                        w_in = in_w0 + kw
                        if 0 <= h_in < H_in and 0 <= w_in < W_in:
                            idx.append(base_c + h_in * W_in + w_in)
                        else:
                            idx.append(-1)
    idx_t = torch.tensor(idx, dtype=torch.long)
    return idx_t, H_out, W_out, block_size


def _bounds_box(hz: HZono) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cheap interval-hull bounds (ignores constraints, sound)."""
    radius = hz.Gc.abs().sum(dim=1) + hz.Gb.abs().sum(dim=1)
    c_flat = hz.c.flatten()
    return c_flat - radius, c_flat + radius


# ---------------------------------------------------------------------------
# Public: hz_maxpool2d
# ---------------------------------------------------------------------------


def hz_maxpool2d(
    hz: HZono, *,
    kernel_size, stride=None, padding=0, input_shape,
) -> HZono:
    """2D max-pool on an HZono with stable-winner row preservation.

    Args:
        hz: input HZono with ``n = C * H * W`` rows in CHW order.
        kernel_size: int or ``(kh, kw)``.
        stride: int or ``(sh, sw)``; defaults to ``kernel_size``.
        padding: int (max over a 4-tuple if given as 4-tuple, matching
            HyZor's convention).
        input_shape: ``(C, H, W)`` or ``(1, C, H, W)``.

    Returns:
        HZono with ``n_out = C * H_out * W_out`` rows. Adds at most
        ``k_uns`` (number of unstable blocks) new continuous generators
        and 0 new constraints.
    """
    if isinstance(kernel_size, (list, tuple)):
        kh, kw = int(kernel_size[0]), int(kernel_size[1])
    else:
        kh = kw = int(kernel_size)

    if stride is None:
        sh = sw = kh
    elif isinstance(stride, (list, tuple)):
        sh, sw = int(stride[0]), int(stride[1])
    else:
        sh = sw = int(stride)

    if isinstance(padding, (list, tuple)):
        pad_use = max(int(p) for p in padding)
    else:
        pad_use = int(padding)

    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"hz_maxpool2d: bad input_shape {input_shape}")
    C, H, W = int(C), int(H), int(W)

    device = hz.c.device
    dtype = hz.c.dtype
    n = int(hz.c.shape[0])
    assert n == C * H * W, f"hz_maxpool2d: rows {n} != C*H*W = {C * H * W}"

    idx, H_out, W_out, block_size = _build_maxpool_index_map_chw(
        H, W, C, kh, kw, sh, sw, pad_use, pad_use,
    )
    idx = idx.to(device=device)

    # Padding-aware rearrange: append a zero pad row and remap -1 to it.
    has_pad = bool((idx < 0).any().item())
    if has_pad:
        pad_row_idx = n  # index of the appended pad row
        ng0 = int(hz.Gc.shape[1])
        nb0 = int(hz.Gb.shape[1])
        pad_c = torch.zeros((1, 1), device=device, dtype=dtype)
        pad_Gc = torch.zeros((1, ng0), device=device, dtype=dtype)
        pad_Gb = torch.zeros((1, nb0), device=device, dtype=dtype)
        padded_hz = HZono(
            c=torch.cat([hz.c, pad_c], dim=0),
            Gc=torch.cat([hz.Gc, pad_Gc], dim=0),
            Gb=torch.cat([hz.Gb, pad_Gb], dim=0),
            Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
            eq_mask=hz.eq_mask,
        )
        idx_rm = idx.clone()
        idx_rm[idx_rm < 0] = pad_row_idx
        rearranged = _take_rows(padded_hz, idx_rm)
    else:
        rearranged = _take_rows(hz, idx)

    lb_flat, ub_flat = _bounds_box(rearranged)
    num_blocks = C * H_out * W_out
    lb_blocks = lb_flat.view(num_blocks, block_size)
    ub_blocks = ub_flat.view(num_blocks, block_size)

    best_lb, best_in_block = lb_blocks.max(dim=1)

    if block_size > 1:
        ub_others = ub_blocks.clone()
        ub_others[
            torch.arange(num_blocks, device=device),
            best_in_block,
        ] = float("-inf")
        max_others_ub = ub_others.max(dim=1).values
    else:
        max_others_ub = torch.full(
            (num_blocks,), float("-inf"),
            device=device, dtype=dtype,
        )

    stable_mask = best_lb >= max_others_ub
    block_offsets = torch.arange(num_blocks, device=device) * block_size
    best_rows = block_offsets + best_in_block

    ng_base = int(rearranged.Gc.shape[1])
    nb_base = int(rearranged.Gb.shape[1])
    nc_base = int(rearranged.b.shape[0])

    stable_blocks = torch.nonzero(stable_mask, as_tuple=False).view(-1)
    unstable_blocks = torch.nonzero(~stable_mask, as_tuple=False).view(-1)
    k_uns = int(unstable_blocks.numel())

    ng_new = ng_base + k_uns
    out_c = torch.zeros((num_blocks, 1), device=device, dtype=dtype)
    out_Gc = torch.zeros((num_blocks, ng_new), device=device, dtype=dtype)
    out_Gb = torch.zeros((num_blocks, nb_base), device=device, dtype=dtype)

    if stable_blocks.numel() > 0:
        stable_rows = best_rows[stable_blocks]
        out_c[stable_blocks, :] = rearranged.c[stable_rows, :]
        if ng_base > 0:
            out_Gc[stable_blocks, :ng_base] = rearranged.Gc[stable_rows, :]
        if nb_base > 0:
            out_Gb[stable_blocks, :] = rearranged.Gb[stable_rows, :]

    if k_uns > 0:
        lb_uns = lb_blocks[unstable_blocks].max(dim=1).values
        ub_uns = ub_blocks[unstable_blocks].max(dim=1).values
        ctr_uns = (lb_uns + ub_uns) / 2.0
        rad_uns = (ub_uns - lb_uns) / 2.0
        out_c[unstable_blocks, 0] = ctr_uns
        uns_cols = ng_base + torch.arange(k_uns, device=device)
        out_Gc[unstable_blocks, uns_cols] = rad_uns

    # Preserve original constraints on old factors. New per-block
    # interval generators are unconstrained.
    if nc_base > 0:
        out_Ac = torch.zeros((nc_base, ng_new), device=device, dtype=dtype)
        if ng_base > 0:
            out_Ac[:, :ng_base] = rearranged.Ac
        out_Ab = rearranged.Ab.clone()
        out_b = rearranged.b.clone()
        em_out = None if rearranged.eq_mask is None else rearranged.eq_mask.clone()
    else:
        out_Ac = torch.zeros((0, ng_new), device=device, dtype=dtype)
        out_Ab = torch.zeros((0, nb_base), device=device, dtype=dtype)
        out_b = torch.zeros((0, 1), device=device, dtype=dtype)
        em_out = None

    return HZono(
        c=out_c, Gc=out_Gc, Gb=out_Gb,
        Ac=out_Ac, Ab=out_Ab, b=out_b,
        eq_mask=em_out,
    )
