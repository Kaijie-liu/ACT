#===- act/back_end/hybridz_tf/tf_cnn.py - HybridZ CNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ CNN Transfer Functions. Implements HybridZ-based transfer functions
#   for CNN layers including convolution, pooling, and tensor reshaping
#   operations.
#
#===---------------------------------------------------------------------===#

from typing import Tuple

import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import HZono, _eq_mask_of
from act.back_end.hybridz_tf.tf_mlp import _hz_fact
import act.back_end.interval_tf.tf_cnn as interval


# --- HZ transfer functions (CNN) ---

def tf_conv2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        input_shape = L.params.get("input_shape")
        if input_shape is not None:
            tf._hz_cache[L.id] = hz_conv2d(
                hz_in, L.params["weight"], L.params.get("bias"),
                L.params.get("stride", 1), L.params.get("padding", 0),
                L.params.get("dilation", 1), L.params.get("groups", 1), input_shape,
            )
        else:
            hz_in = None
    fact = interval.tf_conv2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_maxpool2d(L, bounds, tf):
    """Per-window max with stable-winner correlation preservation.

    For each pooling window we compute interval bounds on the K² candidates
    from the HZ. When one candidate's lower bound dominates every other
    candidate's upper bound, the max is that single row and we keep its HZ
    representation verbatim — full correlation through the window. When no
    single candidate dominates we fall back to ``[max(lb), max(ub)]`` with
    a fresh independent generator, which is sound but loses correlation.
    """
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_maxpool2d(L, bounds)
    if hz_in is None:
        return fact
    params = L.params
    input_shape = params.get("input_shape")
    if input_shape is None:
        tf._hz_cache[L.id] = None
        return fact
    try:
        tf._hz_cache[L.id] = hz_maxpool2d(
            hz_in,
            kernel_size=params["kernel_size"],
            stride=params.get("stride"),
            padding=params.get("padding", 0),
            input_shape=input_shape,
        )
    except Exception:
        # Any shape mismatch falls back to the interval path soundly.
        tf._hz_cache[L.id] = None
        return fact
    return _hz_fact(fact, tf._hz_cache[L.id])


# --- HZ conv2d (zonotope domain) ---

def _conv2d_generators(
    G, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
):
    """Apply conv2d to a generator matrix ``(B*C*H*W, ng)`` and return
    a generator matrix ``(B*n_out_per_sample, ng)``. Each generator
    column is convolved independently per batch element by stacking
    ``ng * B`` images into conv2d's leading "batch" axis.
    """
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    # (B*C*H*W, ng) → (ng, B*C*H*W) → (ng, B, C, H, W) → (ng*B, C, H, W)
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = F.conv2d(
        imgs,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out.shape
    # (ng*B, Cp, Hp, Wp) → (ng, B, Cp, Hp, Wp) → (B, Cp, Hp, Wp, ng)
    return (
        out.view(ng, B, Cp, Hp, Wp)
        .permute(1, 2, 3, 4, 0)
        .contiguous()
        .reshape(B * Cp * Hp * Wp, ng)
    )


def hz_conv2d(
    hz: HZono, weight, bias, stride, padding, dilation, groups, input_shape
) -> HZono:
    """Apply conv2d to a hybrid zonotope: convolve the center as one
    ``(B, C, H, W)`` image and each generator column as ``B`` per-batch
    images. ``B`` is recovered from ``hz.c.numel() // (C*H*W)`` so this
    works uniformly for B=1 and B>1 without materialising a
    block-diagonal weight.
    """
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}, expected 3D or 4D")
    weight = weight.to(hz.c)

    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    c_img = hz.c.view(B, C, H, W)
    out_c = F.conv2d(
        c_img,
        weight,
        bias=bias.to(hz.c) if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out_c.shape
    new_c = out_c.reshape(-1, 1)
    n_out_per_sample = Cp * Hp * Wp

    new_Gc = _conv2d_generators(
        hz.Gc, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )
    new_Gb = _conv2d_generators(
        hz.Gb, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )

    # Conv leaves the factor-space constraint rows untouched, so eq_mask
    # passes through unchanged.
    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
    )


# --- HZ MaxPool 2D (stable-winner row preservation) ----------------------


def _take_rows_hz(hz: HZono, idx: torch.Tensor) -> HZono:
    """Index ``hz``'s output dim by ``idx``; constraint rows unchanged."""
    idx = idx.to(device=hz.c.device, dtype=torch.long).view(-1)
    Gc = hz.Gc[idx, :] if hz.Gc.numel() else hz.Gc.new_zeros(idx.numel(), 0)
    Gb = hz.Gb[idx, :] if hz.Gb.numel() else hz.Gb.new_zeros(idx.numel(), 0)
    return HZono(
        c=hz.c[idx, :], Gc=Gc, Gb=Gb,
        Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
        eq_mask=hz.eq_mask,
    )


def _build_maxpool_index_map_chw(
    H_in: int, W_in: int, C_in: int,
    kh: int, kw: int, sh: int, sw: int, pad_h: int, pad_w: int,
) -> Tuple[torch.Tensor, int, int, int]:
    """Return ``(idx, H_out, W_out, block_size)``. ``idx`` is a flat
    ``(num_blocks * block_size,)`` Long tensor giving the source flattened-
    CHW row for every pooling slot, with ``-1`` for padding positions.
    """
    H_out = (H_in - kh + 2 * pad_h) // sh + 1
    W_out = (W_in - kw + 2 * pad_w) // sw + 1
    block_size = kh * kw
    idx = []
    for co in range(C_in):
        base_c = co * H_in * W_in
        for h_out in range(H_out):
            in_h0 = h_out * sh - pad_h
            for w_out in range(W_out):
                in_w0 = w_out * sw - pad_w
                for dh in range(kh):
                    h_in = in_h0 + dh
                    for dw in range(kw):
                        w_in = in_w0 + dw
                        if 0 <= h_in < H_in and 0 <= w_in < W_in:
                            idx.append(base_c + h_in * W_in + w_in)
                        else:
                            idx.append(-1)
    return torch.tensor(idx, dtype=torch.long), H_out, W_out, block_size


def _hz_interval_bounds(hz: HZono) -> Tuple[torch.Tensor, torch.Tensor]:
    radius = hz.Gc.abs().sum(dim=1) + hz.Gb.abs().sum(dim=1)
    c_flat = hz.c.flatten()
    return c_flat - radius, c_flat + radius


def hz_maxpool2d(
    hz: HZono, *, kernel_size, stride=None, padding=0, input_shape,
) -> HZono:
    """2D max-pool on an HZono with per-window stable-winner preservation.

    A pooling block whose largest lower bound dominates every other entry's
    upper bound is reduced to that single row exactly (full HZ correlation
    kept). Other blocks fall back to ``[max(lb), max(ub)]`` with one fresh
    continuous generator. Adds at most ``k_unstable`` new generators and
    zero new constraint rows; eq_mask passes through.
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
        pad = max(int(p) for p in padding)
    else:
        pad = int(padding)
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"hz_maxpool2d: bad input_shape {input_shape}")
    C, H, W = int(C), int(H), int(W)

    device, dtype = hz.c.device, hz.c.dtype
    n = int(hz.c.shape[0])
    assert n == C * H * W, f"hz_maxpool2d: rows {n} != C*H*W = {C * H * W}"

    idx, H_out, W_out, block_size = _build_maxpool_index_map_chw(
        H, W, C, kh, kw, sh, sw, pad, pad,
    )
    idx = idx.to(device=device)

    # Pad-aware rearrange.
    if bool((idx < 0).any().item()):
        pad_row = n
        ng0 = int(hz.Gc.shape[1])
        nb0 = int(hz.Gb.shape[1])
        padded = HZono(
            c=torch.cat([hz.c, torch.zeros(1, 1, device=device, dtype=dtype)], dim=0),
            Gc=torch.cat([hz.Gc, torch.zeros(1, ng0, device=device, dtype=dtype)], dim=0),
            Gb=torch.cat([hz.Gb, torch.zeros(1, nb0, device=device, dtype=dtype)], dim=0),
            Ac=hz.Ac, Ab=hz.Ab, b=hz.b, eq_mask=hz.eq_mask,
        )
        idx_rm = idx.clone()
        idx_rm[idx_rm < 0] = pad_row
        rearranged = _take_rows_hz(padded, idx_rm)
    else:
        rearranged = _take_rows_hz(hz, idx)

    lb_flat, ub_flat = _hz_interval_bounds(rearranged)
    num_blocks = C * H_out * W_out
    lb_blocks = lb_flat.view(num_blocks, block_size)
    ub_blocks = ub_flat.view(num_blocks, block_size)
    best_lb, best_in_block = lb_blocks.max(dim=1)

    if block_size > 1:
        ub_others = ub_blocks.clone()
        ub_others[torch.arange(num_blocks, device=device), best_in_block] = float("-inf")
        max_others_ub = ub_others.max(dim=1).values
    else:
        max_others_ub = torch.full(
            (num_blocks,), float("-inf"), device=device, dtype=dtype,
        )

    stable = best_lb >= max_others_ub
    block_offsets = torch.arange(num_blocks, device=device) * block_size
    best_rows = block_offsets + best_in_block

    ng_base = int(rearranged.Gc.shape[1])
    nb_base = int(rearranged.Gb.shape[1])
    stable_idx = torch.nonzero(stable, as_tuple=False).view(-1)
    unstable_idx = torch.nonzero(~stable, as_tuple=False).view(-1)
    k_unstable = int(unstable_idx.numel())

    ng_new = ng_base + k_unstable
    out_c = torch.zeros((num_blocks, 1), device=device, dtype=dtype)
    out_Gc = torch.zeros((num_blocks, ng_new), device=device, dtype=dtype)
    out_Gb = torch.zeros((num_blocks, nb_base), device=device, dtype=dtype)

    if stable_idx.numel() > 0:
        rows = best_rows[stable_idx]
        out_c[stable_idx, :] = rearranged.c[rows, :]
        if ng_base > 0:
            out_Gc[stable_idx, :ng_base] = rearranged.Gc[rows, :]
        if nb_base > 0:
            out_Gb[stable_idx, :] = rearranged.Gb[rows, :]

    if k_unstable > 0:
        lb_u = lb_blocks[unstable_idx].max(dim=1).values
        ub_u = ub_blocks[unstable_idx].max(dim=1).values
        ctr = (lb_u + ub_u) / 2.0
        rad = (ub_u - lb_u) / 2.0
        out_c[unstable_idx, 0] = ctr
        out_Gc[unstable_idx, ng_base + torch.arange(k_unstable, device=device)] = rad

    nc_base = int(rearranged.b.shape[0])
    if nc_base > 0:
        out_Ac = torch.zeros((nc_base, ng_new), device=device, dtype=dtype)
        if ng_base > 0:
            out_Ac[:, :ng_base] = rearranged.Ac
        out_Ab = rearranged.Ab.clone()
        out_b = rearranged.b.clone()
        em = None if rearranged.eq_mask is None else rearranged.eq_mask.clone()
    else:
        out_Ac = torch.zeros((0, ng_new), device=device, dtype=dtype)
        out_Ab = torch.zeros((0, nb_base), device=device, dtype=dtype)
        out_b = torch.zeros((0, 1), device=device, dtype=dtype)
        em = None

    return HZono(
        c=out_c, Gc=out_Gc, Gb=out_Gb,
        Ac=out_Ac, Ab=out_Ab, b=out_b, eq_mask=em,
    )
