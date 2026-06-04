"""ImageHZ operators — sound forward dispatch on the spatial body.

Per advisor 2026-06-03 rev-2:
- MaxPool2D is FORBIDDEN; calling it raises `NotImplementedError`.
- No constraints (`Ac/Ab/b`) introduced anywhere; ReLU triangle
  adds aux generators only.
- All math float64.
"""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F

from act.back_end.imagehz.representation import (
    BoundingBox,
    ImageHZ,
    SpatialGenerator,
    merge_generators_by_xi_id,
)


# ── Conv2D ────────────────────────────────────────────────────────


def apply_conv2d(
    hz: ImageHZ,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int = 1,
    padding: int = 0,
) -> ImageHZ:
    """Apply a conv2d to the center and each generator.

    weight shape: (C_out, C_in, kH, kW). Each generator is convolved
    using only its local region (zero-padded back to a full image
    only for the math; the output region is the Minkowski
    dilation of the input region with the kernel, then clipped to
    output spatial dims).

    Soundness: conv is linear, so applying it generator-by-generator
    is the same as applying it to the whole HZ (each xi
    coefficient maps independently). The center carries the bias.
    """
    if hz.dtype != torch.float64:
        raise ValueError(f"ImageHZ float64 required; got {hz.dtype}")
    if weight.dtype != torch.float64:
        weight = weight.to(torch.float64)
    if bias is not None and bias.dtype != torch.float64:
        bias = bias.to(torch.float64)

    c_in = int(weight.shape[1])
    c_out = int(weight.shape[0])
    kH = int(weight.shape[2])
    kW = int(weight.shape[3])
    if hz.C != c_in:
        raise ValueError(
            f"conv2d C_in mismatch: hz.C={hz.C} vs weight C_in={c_in}"
        )

    # Center: standard conv.
    c_in_btn = hz.c.unsqueeze(0)  # (1, C, H, W)
    c_out_btn = F.conv2d(
        c_in_btn, weight, bias=bias, stride=stride, padding=padding
    )
    new_c = c_out_btn.squeeze(0)
    _, H_out, W_out = new_c.shape

    new_gens: List[SpatialGenerator] = []
    for g in hz.generators:
        # Build a full-image padded tensor for this generator, run
        # the conv, then crop the result to the bounding region of
        # the affected output positions. For correctness we just
        # convolve the whole padded thing — operator cost is bounded
        # by region size for sparse regions, but here we accept the
        # simple correct implementation; locality benefit shows up
        # in MEMORY (per-generator storage), not in this op's CPU.
        full = hz.embed_generator(g)            # (C_in, H, W)
        out_full = F.conv2d(
            full.unsqueeze(0), weight,
            bias=None,                           # bias goes to center only
            stride=stride, padding=padding,
        ).squeeze(0)                             # (C_out, H_out, W_out)

        # Find the smallest bounding box in (c_out, h_out, w_out)
        # space that contains all non-zero values of out_full.
        nz_mask = out_full.abs() > 0
        if not nz_mask.any():
            continue
        nz_idx = nz_mask.nonzero(as_tuple=False)
        c_lo = int(nz_idx[:, 0].min())
        c_hi = int(nz_idx[:, 0].max()) + 1
        h_lo = int(nz_idx[:, 1].min())
        h_hi = int(nz_idx[:, 1].max()) + 1
        w_lo = int(nz_idx[:, 2].min())
        w_hi = int(nz_idx[:, 2].max()) + 1
        region = BoundingBox(c_lo, c_hi, h_lo, h_hi, w_lo, w_hi)
        values = out_full[c_lo:c_hi, h_lo:h_hi, w_lo:w_hi].contiguous()
        new_gens.append(SpatialGenerator(
            region=region, values=values, xi_id=g.xi_id,
        ))

    return ImageHZ(c=new_c, generators=new_gens)


# ── AvgPool2D ─────────────────────────────────────────────────────


def apply_avgpool2d(
    hz: ImageHZ,
    kernel_size: int,
    stride: int | None = None,
) -> ImageHZ:
    """Average pooling. Linear in generators."""
    if stride is None:
        stride = kernel_size
    c_in_btn = hz.c.unsqueeze(0)
    c_out_btn = F.avg_pool2d(c_in_btn, kernel_size=kernel_size, stride=stride)
    new_c = c_out_btn.squeeze(0)

    new_gens: List[SpatialGenerator] = []
    for g in hz.generators:
        full = hz.embed_generator(g)
        out_full = F.avg_pool2d(
            full.unsqueeze(0), kernel_size=kernel_size, stride=stride
        ).squeeze(0)
        nz_mask = out_full.abs() > 0
        if not nz_mask.any():
            continue
        nz_idx = nz_mask.nonzero(as_tuple=False)
        region = BoundingBox(
            c_lo=int(nz_idx[:, 0].min()),
            c_hi=int(nz_idx[:, 0].max()) + 1,
            h_lo=int(nz_idx[:, 1].min()),
            h_hi=int(nz_idx[:, 1].max()) + 1,
            w_lo=int(nz_idx[:, 2].min()),
            w_hi=int(nz_idx[:, 2].max()) + 1,
        )
        values = out_full[
            region.c_lo:region.c_hi,
            region.h_lo:region.h_hi,
            region.w_lo:region.w_hi,
        ].contiguous()
        new_gens.append(SpatialGenerator(
            region=region, values=values, xi_id=g.xi_id,
        ))
    return ImageHZ(c=new_c, generators=new_gens)


# ── MaxPool2D — FORBIDDEN ─────────────────────────────────────────


def apply_maxpool2d(*args, **kwargs) -> ImageHZ:
    """FORBIDDEN. Advisor 2026-06-03 rev-2 guard 1.

    Per-window argmax on the IBP center is NOT sound under input
    perturbation. Three options exist:
    (a) restrict to models without MaxPool (CIFAR resnet_medium qualifies)
    (b) implement sound MaxPool over-approximation (future work)
    (c) fail-closed when encountered (this prototype)

    Prototype picks (c). Callers receive a clear error.
    """
    raise NotImplementedError(
        "ImageHZ.apply_maxpool2d is intentionally not implemented in "
        "the prototype. Per advisor 2026-06-03 rev-2 guard 1, "
        "center-argmax MaxPool is unsound. Use a model without "
        "MaxPool, or implement a sound over-approximation "
        "separately before re-enabling this operator."
    )


# ── ADD (residual) ────────────────────────────────────────────────


def apply_add(hz_a: ImageHZ, hz_b: ImageHZ) -> ImageHZ:
    """Add two ImageHZs.

    Soundness contract per advisor rev-2 test case 3:
    - generators with the SAME `xi_id` across the two HZs merge
      into one generator whose region = union and values = sum
      (extended to the union region; zero outside each operand's
      original region).
    - generators with distinct `xi_id` are unioned (kept as
      separate generators in the output).

    This is the *factor-aware* sum identical in semantics to
    `hz_factor_aware_sum` in `solver_hz.py`. Different xi_ids must
    not be aliased.
    """
    if hz_a.shape != hz_b.shape:
        raise ValueError(
            f"ADD shape mismatch: {hz_a.shape} vs {hz_b.shape}"
        )

    new_c = hz_a.c + hz_b.c
    new_gens = merge_generators_by_xi_id(
        list(hz_a.generators) + list(hz_b.generators),
        dtype=hz_a.dtype, device=hz_a.device,
    )
    return ImageHZ(c=new_c, generators=new_gens)


# ── ReLU triangle ─────────────────────────────────────────────────


def apply_relu_triangle(hz: ImageHZ, next_aux_id: int) -> tuple[ImageHZ, int]:
    """DeepZ-style ReLU triangle relaxation.

    For each (c, h, w) position:
      lb = c - rad, ub = c + rad   (from `hz.bounds()`)
      if lb >= 0: pass through (no change)
      if ub <= 0: zero out (center + scale all existing generator
                  contributions at that position by 0)
      else:        unstable. λ = ub / (ub - lb), μ = -lb*ub / (2*(ub-lb)).
                   new center value at (c,h,w) becomes λ·c[c,h,w] + μ
                   each existing generator's contribution at (c,h,w)
                   is scaled by λ; ONE new aux generator is added with
                   single-pixel region around (c,h,w), value μ, and a
                   fresh xi_id.

    Returns (new_hz, next_aux_id_after).

    Soundness contract per advisor test 2: sampled exact points
    must lie within the new bounds.
    """
    lb, ub = hz.bounds()
    is_active = lb >= 0
    is_inactive = ub <= 0
    is_unstable = ~(is_active | is_inactive)

    new_c = hz.c.clone()
    new_c[is_inactive] = 0.0

    # Compute lam, mu only where unstable. Use safe denominator.
    den = (ub - lb).clamp_min(1e-300)
    lam = ub / den
    mu = -lb * ub / (2.0 * den)

    # Apply at unstable positions: c[u] = lam[u]*c[u] + mu[u]
    new_c[is_unstable] = (
        lam[is_unstable] * hz.c[is_unstable] + mu[is_unstable]
    )

    # Build mask of (active OR unstable lam-scale) for existing
    # generator contribution scaling.
    # - active positions: scale by 1
    # - inactive positions: scale by 0
    # - unstable positions: scale by lam
    row_scale = torch.zeros_like(hz.c)
    row_scale[is_active] = 1.0
    row_scale[is_unstable] = lam[is_unstable]

    # Scale each existing generator's values by row_scale over its
    # region.
    new_gens: List[SpatialGenerator] = []
    for g in hz.generators:
        r = g.region
        scale_slice = row_scale[
            r.c_lo:r.c_hi, r.h_lo:r.h_hi, r.w_lo:r.w_hi,
        ]
        new_values = g.values * scale_slice
        if (new_values.abs() == 0).all():
            continue
        new_gens.append(SpatialGenerator(
            region=g.region,
            values=new_values.contiguous(),
            xi_id=g.xi_id,
        ))

    # Add one aux generator per unstable position. Each aux is a
    # 1×1×1 region (single pixel) with value mu at that position.
    unstable_idx = is_unstable.nonzero(as_tuple=False)
    aux_id = next_aux_id
    for entry in unstable_idx:
        c_i, h_i, w_i = int(entry[0]), int(entry[1]), int(entry[2])
        region = BoundingBox(
            c_lo=c_i, c_hi=c_i + 1,
            h_lo=h_i, h_hi=h_i + 1,
            w_lo=w_i, w_hi=w_i + 1,
        )
        val = mu[c_i, h_i, w_i].clone().reshape(1, 1, 1)
        new_gens.append(SpatialGenerator(
            region=region, values=val, xi_id=aux_id,
        ))
        aux_id += 1

    return ImageHZ(c=new_c, generators=new_gens), aux_id
