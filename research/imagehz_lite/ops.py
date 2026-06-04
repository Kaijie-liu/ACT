"""ImageHZ-lite Phase 0 operators.

Implements four operators per §3 of the prototype plan:
    apply_conv2d
    apply_relu_triangle      (DeepZ; 1 aux per unstable position)
    apply_maxpool2d          (sound-first per §9R-2)
    apply_flatten            (Phase0FlattenSnapshot)

All operations are forward-only, no gradients, no random sampling.
The MaxPool implementation uses the sound-first semantics from §9R-2:
exact selection on a stable winner; single-candidate over-approximation
on the unstable case. No multi-candidate convex hull.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from research.imagehz_lite.budget import Budget
from research.imagehz_lite.domain import (
    ImageHZLite,
    Phase0FlattenSnapshot,
    TileBlock,
)


# ── Conv2D ────────────────────────────────────────────────────────


def apply_conv2d(
    hz: ImageHZLite,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> ImageHZLite:
    """Apply a Conv2D to the center and each TileBlock independently.

    Conv is linear; the same convolution applied to each generator
    column independently produces a sound (in fact exact) result.
    Each block grows spatially by the kernel half-size; channels remap
    according to the conv's (C_out, C_in, kH, kW) weight.
    """
    if hz.c.dtype != torch.float64 or weight.dtype != torch.float64:
        raise ValueError(
            f"apply_conv2d requires float64; c={hz.c.dtype} w={weight.dtype}"
        )
    c_in = int(weight.shape[1])
    if hz.C != c_in:
        raise ValueError(
            f"Conv2D in_channels mismatch: hz.C={hz.C} weight.C_in={c_in}"
        )

    # Center: standard conv, bias on center.
    new_c = F.conv2d(
        hz.c.unsqueeze(0), weight, bias=bias,
        stride=stride, padding=padding,
    ).squeeze(0)

    new_tiles: List[TileBlock] = []
    _, H_out, W_out = new_c.shape
    for tile in hz.tiles:
        # Embed each generator column into the full (C, H, W) frame,
        # run conv (no bias for generators), crop to its non-zero
        # bounding box.
        c0, h0, w0 = tile.origin_chw
        tc, th, tw = tile.shape
        # Generator stack: (n_gen, C, H, W) full-frame.
        full = torch.zeros(
            (tile.n_gen_tile, hz.C, hz.H, hz.W),
            dtype=torch.float64, device=hz.c.device,
        )
        full[:, c0:c0 + tc, h0:h0 + th, w0:w0 + tw] = tile.G_tile
        # Conv per generator. Treat n_gen as batch.
        out = F.conv2d(
            full, weight, bias=None,
            stride=stride, padding=padding,
        )  # (n_gen, C_out, H_out, W_out)

        # Find the smallest bounding box that contains all non-zero
        # output positions across all generators of this tile.
        # If everything is zero (degenerate), drop the tile.
        any_nz = (out.abs().sum(dim=0) > 0)  # (C_out, H_out, W_out)
        if not any_nz.any():
            continue
        nz_idx = any_nz.nonzero(as_tuple=False)
        c_lo = int(nz_idx[:, 0].min().item())
        c_hi = int(nz_idx[:, 0].max().item()) + 1
        h_lo = int(nz_idx[:, 1].min().item())
        h_hi = int(nz_idx[:, 1].max().item()) + 1
        w_lo = int(nz_idx[:, 2].min().item())
        w_hi = int(nz_idx[:, 2].max().item()) + 1
        new_origin = (c_lo, h_lo, w_lo)
        new_shape = (c_hi - c_lo, h_hi - h_lo, w_hi - w_lo)
        G_new = out[
            :, c_lo:c_hi, h_lo:h_hi, w_lo:w_hi,
        ].contiguous()
        new_tiles.append(TileBlock(
            origin_chw=new_origin,
            shape=new_shape,
            G_tile=G_new,
            factor_ids=tuple(tile.factor_ids),
            aux_meta=dict(tile.aux_meta),
        ))
    return ImageHZLite(c=new_c, tiles=new_tiles)


# ── ReLU triangle (DeepZ; one aux per unstable position) ─────────


def apply_relu_triangle(
    hz: ImageHZLite,
    budget: Budget,
    layer_id: int,
    next_aux_id: int,
) -> Tuple[ImageHZLite, int]:
    """DeepZ-style ReLU.

    Per §9R-3: one aux generator per unstable position; if the count
    would exceed the budget, raise `BudgetExceeded`.

    Per (un)stable position i with bounds [l_i, u_i]:
      active   (l_i >= 0): y_i = x_i               (pass through)
      inactive (u_i <= 0): y_i = 0                 (zero out)
      unstable (l_i <  0 < u_i):
          lam_i = u_i / (u_i - l_i)
          mu_i  = -l_i * u_i / (2 (u_i - l_i))
          y_i   = lam_i * c_i + mu_i  + sum over old gens scaled by lam_i
                  + new_aux_i (value mu_i, single-position TileBlock)
    """
    lb, ub = hz.bounds()
    is_active = (lb >= 0)
    is_inactive = (ub <= 0)
    is_unstable = ~(is_active | is_inactive)

    unstable_idx = is_unstable.nonzero(as_tuple=False)
    k_unstable = int(unstable_idx.shape[0])
    budget.spend_relu_aux(layer_id=layer_id, n=k_unstable)

    den = (ub - lb).clamp_min(1e-300)
    lam = ub / den
    mu = -lb * ub / (2.0 * den)

    # Center update.
    new_c = hz.c.clone()
    new_c[is_inactive] = 0.0
    new_c[is_unstable] = lam[is_unstable] * hz.c[is_unstable] + mu[is_unstable]

    # Build a (C, H, W) row_scale: 1 for active, 0 for inactive, lam for unstable.
    row_scale = torch.zeros_like(hz.c)
    row_scale[is_active] = 1.0
    row_scale[is_unstable] = lam[is_unstable]

    # Apply row_scale to each tile's generators.
    new_tiles: List[TileBlock] = []
    for tile in hz.tiles:
        c0, h0, w0 = tile.origin_chw
        tc, th, tw = tile.shape
        scale_slice = row_scale[c0:c0 + tc, h0:h0 + th, w0:w0 + tw]
        # Broadcast over the n_gen axis.
        G_new = tile.G_tile * scale_slice.unsqueeze(0)
        if (G_new.abs() == 0).all():
            continue
        new_tiles.append(TileBlock(
            origin_chw=tile.origin_chw,
            shape=tile.shape,
            G_tile=G_new.contiguous(),
            factor_ids=tuple(tile.factor_ids),
            aux_meta=dict(tile.aux_meta),
        ))

    # Add one aux per unstable position. Each aux is a single-position
    # tile (1, 1, 1, 1) with value mu_i and a fresh aux factor id.
    aux_id = next_aux_id
    for row in unstable_idx:
        ci, hi, wi = int(row[0].item()), int(row[1].item()), int(row[2].item())
        mu_val = float(mu[ci, hi, wi].item())
        G = torch.tensor(
            [[[[mu_val]]]],
            dtype=torch.float64, device=hz.c.device,
        )
        new_tiles.append(TileBlock(
            origin_chw=(ci, hi, wi),
            shape=(1, 1, 1),
            G_tile=G,
            factor_ids=(aux_id,),
            aux_meta={
                "kind": "relu_aux",
                "spawn_layer": int(layer_id),
                "spawn_op": "relu_triangle",
                "parent_block": None,
            },
        ))
        aux_id += 1
    return ImageHZLite(c=new_c, tiles=new_tiles), aux_id


# ── MaxPool2D (sound-first per §9R-2) ─────────────────────────────


def apply_maxpool2d(
    hz: ImageHZLite,
    kernel_size: int,
    stride: int | None = None,
    budget: Budget | None = None,
    layer_id: int = 0,
    next_aux_id: int = 0,
) -> Tuple[ImageHZLite, int, Dict[str, Any]]:
    """Sound-first MaxPool2D per §9R-2.

    For each pooling window:
      - stable winner case: output = chosen position's TileBlocks
        copied verbatim. NO aux. Exact.
      - unstable case: pick deterministic candidate m = argmax_i ub_i
        (ties broken by lowest flat index). Output center =
            lb[m] + D_m / 2     with D_m = max_i ub_i - lb[m].
        Output keeps m's TileBlocks (preserving one candidate's
        root provenance). Add one new aux TileBlock with value D_m / 2
        at the output position.

    Returns (hz_out, next_aux_id_after, stats).
    The stats dict carries n_stable / n_unstable / n_output_positions_with_root_provenance
    so the driver can compute the gate metric directly without re-scanning.
    """
    if stride is None:
        stride = kernel_size

    lb, ub = hz.bounds()  # (C, H, W)
    C, H_in, W_in = hz.shape
    H_out = (H_in - kernel_size) // stride + 1
    W_out = (W_in - kernel_size) // stride + 1

    # For each output (c, h_o, w_o) we need to:
    #   - identify the in-window positions (c, h_o*s..h_o*s+k, w_o*s..w_o*s+k)
    #   - find m = position with lowest lb among stable winners, OR if
    #     no stable winner, argmax ub
    #   - decide stable vs unstable
    # Implement vectorized.

    # Build per-window unfold: F.unfold expects (N, C, H, W). We treat
    # C as the batch-of-channels because max is per channel.
    # Reshape lb / ub to (C, 1, H, W) and use unfold along H/W only.
    # Easier: do it with simple python loops over output positions and
    # channels — correctness first, vectorize later if Phase 0 wall budget
    # is tight.

    new_c = torch.zeros(
        (C, H_out, W_out), dtype=torch.float64, device=hz.c.device,
    )
    # Per-position parent map: list of (origin_chw, factor_ids) telling
    # which (c, h, w) of the input was chosen as the winner m. None if
    # everything inactive in that pool window.
    # output_winner_map[c, h_o, w_o] = (c, h_in, w_in) | None
    output_winner_map: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    # Output positions where we added a relu_aux-style D/2 tile.
    aux_positions: List[Tuple[int, int, int, float, int]] = []  # (c, h_o, w_o, mu_val, factor_id)

    n_stable = 0
    n_unstable = 0

    aux_id = int(next_aux_id)

    for c in range(C):
        for h_o in range(H_out):
            for w_o in range(W_out):
                hi0 = h_o * stride
                wi0 = w_o * stride
                lb_w = lb[c, hi0:hi0 + kernel_size, wi0:wi0 + kernel_size]
                ub_w = ub[c, hi0:hi0 + kernel_size, wi0:wi0 + kernel_size]
                # Flatten.
                lb_f = lb_w.reshape(-1)
                ub_f = ub_w.reshape(-1)
                # Stable winner exists iff some m with lb_m >= max_{i != m} ub_i.
                # Equivalently, iff max(lb) >= max_{i not argmax(lb)} ub_i.
                m_lb = int(torch.argmax(lb_f).item())
                lb_m = float(lb_f[m_lb].item())
                # max of ub excluding m_lb:
                if lb_f.numel() <= 1:
                    max_other_ub = float("-inf")
                else:
                    mask = torch.ones_like(ub_f, dtype=torch.bool)
                    mask[m_lb] = False
                    max_other_ub = float(ub_f[mask].max().item())
                stable = lb_m >= max_other_ub
                if stable:
                    n_stable += 1
                    winner = m_lb
                else:
                    n_unstable += 1
                    # argmax ub, tie -> lowest index
                    max_ub = float(ub_f.max().item())
                    cand_idx = (ub_f >= max_ub - 1e-15).nonzero(as_tuple=False).flatten()
                    winner = int(cand_idx[0].item())
                # Position in the (h, w) window.
                h_in = hi0 + winner // kernel_size
                w_in = wi0 + winner % kernel_size
                output_winner_map[(c, h_o, w_o)] = (c, h_in, w_in)
                if stable:
                    # Exact: output center = c_input_at_winner
                    new_c[c, h_o, w_o] = hz.c[c, h_in, w_in]
                else:
                    # unstable over-approx
                    lb_m_val = float(lb_f[winner].item())
                    D_m = float(ub_f.max().item()) - lb_m_val
                    if D_m < 0:
                        D_m = 0.0
                    new_c[c, h_o, w_o] = lb_m_val + 0.5 * D_m
                    aux_positions.append(
                        (c, h_o, w_o, 0.5 * D_m, aux_id)
                    )
                    aux_id += 1

    # Build output tiles by copying each chosen-winner's tile blocks
    # into the output position with the same generator coefficients
    # (Phase 0 simplification: pre-pool generators map 1:1 to output
    # position; this is sound because the winner's bound is what we
    # selected, not a multi-position combination).
    new_tiles: List[TileBlock] = []
    n_output_positions_with_root_provenance = 0
    for out_pos, in_pos in output_winner_map.items():
        c_o, h_o, w_o = out_pos
        c_in, h_in, w_in = in_pos
        # Find all input tiles overlapping (c_in, h_in, w_in).
        local_root = False
        for tile in hz.tiles:
            c0, h0, w0 = tile.origin_chw
            tc, th, tw = tile.shape
            if not (c0 <= c_in < c0 + tc and
                    h0 <= h_in < h0 + th and
                    w0 <= w_in < w0 + tw):
                continue
            # The slice of G_tile at (c_in - c0, h_in - h0, w_in - w0)
            # is a (n_gen_tile, 1, 1, 1) row.
            slc = tile.G_tile[
                :, c_in - c0: c_in - c0 + 1,
                h_in - h0: h_in - h0 + 1,
                w_in - w0: w_in - w0 + 1,
            ]
            if (slc.abs() == 0).all():
                continue
            new_tiles.append(TileBlock(
                origin_chw=(c_o, h_o, w_o),
                shape=(1, 1, 1),
                G_tile=slc.clone().contiguous(),
                factor_ids=tuple(tile.factor_ids),
                aux_meta=dict(tile.aux_meta),
            ))
            if tile.aux_meta.get("kind") == "root":
                local_root = True
        if local_root:
            n_output_positions_with_root_provenance += 1

    # Add the unstable aux tiles.
    for (c_o, h_o, w_o, mu_val, fid) in aux_positions:
        G = torch.tensor(
            [[[[mu_val]]]],
            dtype=torch.float64, device=hz.c.device,
        )
        new_tiles.append(TileBlock(
            origin_chw=(c_o, h_o, w_o),
            shape=(1, 1, 1),
            G_tile=G,
            factor_ids=(fid,),
            aux_meta={
                "kind": "relu_aux",
                "spawn_layer": int(layer_id),
                "spawn_op": "maxpool_unstable",
                "parent_block": None,
            },
        ))

    stats = {
        "n_stable": n_stable,
        "n_unstable": n_unstable,
        "n_output_positions": C * H_out * W_out,
        "n_output_positions_with_root_provenance":
            n_output_positions_with_root_provenance,
    }
    if budget is not None and aux_positions:
        # Maxpool aux count is bounded by the unstable window count;
        # it shares the same "relu_aux" budget kind so the global cap
        # holds across operators.
        budget.spend_relu_aux(layer_id=layer_id, n=len(aux_positions))
    return ImageHZLite(c=new_c, tiles=new_tiles), aux_id, stats


# ── Flatten — Phase 0 snapshot, no verifier link ──────────────────


def apply_flatten(
    hz: ImageHZLite,
    girard_fires: List[Dict[str, Any]],
    peak_memory_bytes: int,
    wall_s: float,
) -> Phase0FlattenSnapshot:
    """Build a Phase0FlattenSnapshot per §9R-4.

    Deterministic block ordering:
      - sort by aux_meta['kind'] ('root' before 'relu_aux')
      - then by spawn_layer
      - then by origin_chw

    n_flat = C * H * W.
    """
    C, H, W = hz.shape
    n_flat = C * H * W
    c_flat = hz.c.reshape(-1).contiguous()
    sorted_tiles = sorted(
        hz.tiles,
        key=lambda t: (
            0 if t.aux_meta.get("kind") == "root" else 1,
            int(t.aux_meta.get("spawn_layer", 0)),
            tuple(t.origin_chw),
        ),
    )
    blocks_meta: List[Dict[str, Any]] = []
    root_factor_ids: set = set()
    total_aux_count = 0
    for tile in sorted_tiles:
        kind = tile.aux_meta.get("kind", "root")
        blocks_meta.append({
            "origin_chw": tuple(tile.origin_chw),
            "shape": tuple(tile.shape),
            "n_gen_tile": tile.n_gen_tile,
            "factor_ids": list(tile.factor_ids),
            "aux_kind": kind,
            "spawn_layer": int(tile.aux_meta.get("spawn_layer", -1)),
        })
        if kind == "root":
            root_factor_ids.update(tile.factor_ids)
        else:
            total_aux_count += tile.n_gen_tile
    return Phase0FlattenSnapshot(
        c_flat=c_flat,
        blocks_meta=blocks_meta,
        root_ng_at_flatten=len(root_factor_ids),
        total_aux_count=total_aux_count,
        per_layer_girard_fires_observed=list(girard_fires),
        peak_memory_bytes=int(peak_memory_bytes),
        wall_s=float(wall_s),
    )


# ── Structural gate (§9R-5) ───────────────────────────────────────


ALLOWED_PREFIX_OPS = frozenset(
    {"Conv2D", "Conv", "ReLU", "Relu", "MaxPool2D", "MaxPool"}
)


def structural_gate_passes(
    layer_op_names: List[str],
    flatten_index: int,
    trace_has_girard_root_loss_at_maxpool_or_relu: bool,
) -> bool:
    """Return True iff the network's prefix (everything before the
    Flatten layer) contains only Conv2D / ReLU / MaxPool2D and the
    trace shows at least one Girard root-loss event at a MaxPool or
    ReLU layer on this iid.

    This is the only gating function. No benchmark-name check.
    """
    if flatten_index < 0 or flatten_index > len(layer_op_names):
        return False
    prefix = layer_op_names[:flatten_index]
    if not prefix:
        return False
    for op in prefix:
        if op not in ALLOWED_PREFIX_OPS:
            return False
    return bool(trace_has_girard_root_loss_at_maxpool_or_relu)
