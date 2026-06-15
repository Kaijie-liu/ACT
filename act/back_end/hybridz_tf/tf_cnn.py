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

import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import HZono
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


def _pair(x):
    return (int(x), int(x)) if isinstance(x, int) else (int(x[0]), int(x[1]))


def hz_maxpool2d(hz: HZono, kernel_size, stride, padding, dilation, input_shape,
                 binary_budget: int = 0) -> HZono:
    """MaxPool2d as a fold of pairwise maxima: max(a,b) = b + relu(a - b).

    The K*K window candidates of each output are gathered as HZ row-subsets
    (sharing the input factors via col_ids) and folded with hz_sub / hz_apply_relu
    / hz_sgm_add. Padding is handled by padding the HZ with a constant below the
    real range (it can never win a max -> sound). ``binary_budget`` controls the
    per-max encoding: 0 (default) = convex DeepZ relaxation (no binaries, cheap,
    still correlation-preserving -> tighter than interval); None = exact binary.
    """
    from act.back_end.hybridz_tf.tf_mlp import hz_apply_relu, _hz_gather_rows
    from act.back_end.hybridz_tf.algorithms.sgm import hz_sub, hz_sgm_add
    from act.back_end.solver.solver_hz import hz_compute_bounds

    ishape = [int(d) for d in input_shape]
    if len(ishape) == 4:
        _, C, H, W = ishape
    else:
        C, H, W = ishape
    Kh, Kw = _pair(kernel_size)
    Sh, Sw = _pair(stride) if stride is not None else (Kh, Kw)
    Ph, Pw = _pair(padding)
    Dh, Dw = _pair(dilation)
    n = int(hz.c.shape[0])
    B = n // (C * H * W)
    Hp = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
    Wp = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1

    work = hz
    Hpad, Wpad = H, W
    if Ph > 0 or Pw > 0:
        big_neg = float(hz_compute_bounds(hz, exact=False).lb.min().item()) - 1.0
        Hpad, Wpad = H + 2 * Ph, W + 2 * Pw
        npad = B * C * Hpad * Wpad
        pad_idx = torch.full((B, C, Hpad, Wpad), -1, dtype=torch.long)
        pad_idx[:, :, Ph:Ph + H, Pw:Pw + W] = torch.arange(n).view(B, C, H, W)
        pad_flat = pad_idx.reshape(-1)
        valid = pad_flat >= 0
        c_new = hz.c.new_full((npad, 1), big_neg)
        Gc_new = hz.c.new_zeros(npad, hz.Gc.shape[1])
        Gb_new = hz.c.new_zeros(npad, hz.Gb.shape[1])
        c_new[valid] = hz.c[pad_flat[valid]]
        Gc_new[valid] = hz.Gc[pad_flat[valid]]
        Gb_new[valid] = hz.Gb[pad_flat[valid]]
        work = HZono(c=c_new, Gc=Gc_new, Gb=Gb_new, Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
                     eq_mask=hz.eq_mask, col_ids=hz.col_ids, bcol_ids=hz.bcol_ids)

    idx = torch.arange(B * C * Hpad * Wpad).view(B, C, Hpad, Wpad)
    cands = []
    for ki in range(Kh):
        for kj in range(Kw):
            ri = idx[:, :, ki * Dh: ki * Dh + Sh * Hp: Sh,
                     kj * Dw: kj * Dw + Sw * Wp: Sw]
            cands.append(ri[:, :, :Hp, :Wp].reshape(-1))
    m = _hz_gather_rows(work, cands[0])
    for ri in cands[1:]:
        cand = _hz_gather_rows(work, ri)
        m = hz_sgm_add(m, hz_apply_relu(hz_sub(cand, m), binary_budget=binary_budget))
    return m


def tf_maxpool2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            budget = getattr(tf, "_maxpool_binary_budget", 0)
            tf._hz_cache[L.id] = hz_maxpool2d(
                hz_in, L.params["kernel_size"], L.params.get("stride"),
                L.params.get("padding", 0), L.params.get("dilation", 1),
                ishape, binary_budget=budget)
        else:
            hz_in = None
    fact = interval.tf_maxpool2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


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

    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        # conv is affine: each generator COLUMN is convolved independently, so
        # the factor ids carry through unchanged.
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _spatial_op_generators(G, op_fn, B, C, H, W, n_out_per_sample):
    """Apply a spatial op ``op_fn(imgs)->(N,Cp,Hp,Wp)`` to a generator matrix
    ``(B*C*H*W, ng)`` independently per generator column + batch element,
    returning ``(B*n_out_per_sample, ng)``. Shared by avg-pool / conv-transpose
    (each is an affine spatial map, so generators transform like the center)."""
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = op_fn(imgs)
    _, Cp, Hp, Wp = out.shape
    return (out.view(ng, B, Cp, Hp, Wp).permute(1, 2, 3, 4, 0)
            .contiguous().reshape(B * Cp * Hp * Wp, ng))


def _hz_spatial_affine(hz: HZono, op_fn, input_shape, bias=None) -> HZono:
    """Exact-affine spatial transfer (avg-pool, conv-transpose): apply ``op_fn``
    to the center image and each generator column. Affine => exact, constraints
    + factor ids carried through unchanged."""
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    out_c = op_fn(hz.c.view(B, C, H, W))
    _, Cp, Hp, Wp = out_c.shape
    if bias is not None:
        out_c = out_c + bias.to(hz.c).view(1, -1, 1, 1)
    new_c = out_c.reshape(-1, 1)
    n_out = Cp * Hp * Wp
    new_Gc = _spatial_op_generators(hz.Gc, op_fn, B, C, H, W, n_out)
    new_Gb = _spatial_op_generators(hz.Gb, op_fn, B, C, H, W, n_out)
    return HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def hz_avgpool2d(hz, kernel_size, stride, padding, input_shape) -> HZono:
    """AvgPool2d is affine (a fixed averaging linear map) => exact in HZ."""
    op = lambda x: F.avg_pool2d(x, kernel_size=kernel_size,
                                stride=stride if stride is not None else kernel_size,
                                padding=padding)
    return _hz_spatial_affine(hz, op, input_shape)


def tf_avgpool2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            tf._hz_cache[L.id] = hz_avgpool2d(
                hz_in, L.params.get("kernel_size"), L.params.get("stride"),
                L.params.get("padding", 0), ishape)
        else:
            hz_in = None
    fact = interval.tf_avgpool2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def hz_convtranspose2d(hz, weight, bias, stride, padding, output_padding,
                       dilation, groups, input_shape) -> HZono:
    """ConvTranspose2d is affine (index-mapped linear map) => exact in HZ.
    Bias is added to the center only (not the generators)."""
    weight = weight.to(hz.c)
    op = lambda x: F.conv_transpose2d(
        x, weight, bias=None, stride=stride, padding=padding,
        output_padding=output_padding, dilation=dilation, groups=groups)
    return _hz_spatial_affine(hz, op, input_shape, bias=bias)


def tf_convtranspose2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            tf._hz_cache[L.id] = hz_convtranspose2d(
                hz_in, L.params["weight"], L.params.get("bias"),
                L.params.get("stride", 1), L.params.get("padding", 0),
                L.params.get("output_padding", 0), L.params.get("dilation", 1),
                L.params.get("groups", 1), ishape)
        else:
            hz_in = None
    fact = interval.tf_convtranspose2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact
