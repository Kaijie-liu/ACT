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

from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from act.back_end.core import Bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import HZono, SparseHZono, hz_inherit_known_nonempty
from act.back_end.hybridz_tf.tf_mlp import (
    _hz_fact,
    _to_numpy,
    sparse_empty,
    sparse_hz_add_same_frame,
    sparse_hz_apply_relu_exact,
    sparse_hz_gather_rows_like,
    sparse_hz_linear,
    sparse_hz_sub_same_frame,
)
import act.back_end.interval_tf.tf_cnn as interval


def sparse_hz_apply_layer(L, hz: SparseHZono, input_bounds: Bounds, result, tf):
    """Apply sparse-HZ propagation for CNN/pooling layer kinds."""

    k = L.kind.upper()
    if k == LayerKind.CONV2D.value:
        if L.params.get("input_shape") is None:
            return True, None, "missing_conv2d_input_shape"
        return True, sparse_hz_apply_conv2d_layer(hz, L), None
    if k == LayerKind.CONVTRANSPOSE2D.value:
        if L.params.get("input_shape") is None:
            return True, None, "missing_convtranspose2d_input_shape"
        return True, sparse_hz_apply_convtranspose2d_layer(hz, L), None
    if k == LayerKind.AVGPOOL2D.value:
        if L.params.get("input_shape") is None:
            return True, None, "missing_avgpool2d_input_shape"
        return True, sparse_hz_apply_avgpool2d_layer(hz, L), None
    if k == LayerKind.MAXPOOL2D.value:
        if L.params.get("input_shape") is None:
            return True, None, "missing_maxpool2d_input_shape"
        return True, sparse_hz_apply_maxpool2d_layer(
            hz,
            L,
            input_bounds,
            compressed_relu=getattr(tf, "_relu_compressed", False),
            valid_cuts=getattr(tf, "_relu_valid_cuts", False),
        ), None
    return False, None, None


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


def _shape4(shape) -> Tuple[int, int, int, int]:
    dims = tuple(int(d) for d in shape)
    if len(dims) == 4:
        return dims
    if len(dims) == 3:
        c, h, w = dims
        return 1, c, h, w
    raise ValueError(f"expected 3D or 4D NCHW shape, got {shape!r}")


def hz_maxpool2d(hz: HZono, kernel_size, stride, padding, dilation, input_shape,
                 *, exact: bool = True, cell_budget: int | None = None) -> HZono | None:
    """MaxPool2d as a fold of pairwise maxima: max(a,b) = b + relu(a - b).

    The K*K window candidates of each output are gathered as HZ row-subsets
    (sharing the input factors via col_ids) and folded with hz_sub /
    ReLU / hz_sgm_add. Padding is handled by padding the HZ with a constant below
    the real range (it can never win a max -> sound). The backend path is
    exact-only: pairwise max uses the same exact binary ReLU encoding as
    activations. ``exact=False`` is accepted only for legacy call-site
    compatibility and returns ``None`` so strict HybridZ treats the layer as an
    unsupported representation rather than using a triangle relaxation.
    """
    from act.back_end.hybridz_tf.tf_mlp import (
        hz_apply_relu, hz_reduce, _hz_gather_rows,
    )
    from act.back_end.solver.solver_hz import hz_sub, hz_sgm_add
    from act.back_end.solver.solver_hz import hz_compute_bounds

    if not exact:
        return None

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
        work = hz_inherit_known_nonempty(
            HZono(c=c_new, Gc=Gc_new, Gb=Gb_new, Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
                  eq_mask=hz.eq_mask, col_ids=hz.col_ids, bcol_ids=hz.bcol_ids),
            hz,
            reason="maxpool_padding",
        )

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
        diff = hz_sub(cand, m)
        relu = hz_apply_relu(diff, cell_budget=cell_budget)
        if relu is None:
            return None
        m = hz_reduce(hz_sgm_add(m, relu))
    return m


def tf_maxpool2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            out = hz_maxpool2d(
                hz_in, L.params["kernel_size"], L.params.get("stride"),
                L.params.get("padding", 0), L.params.get("dilation", 1),
                ishape, cell_budget=getattr(tf, "_hz_cell_budget", 100_000_000))
            if out is None:
                tf._hz_cache.pop(L.id, None)
            else:
                tf._hz_cache[L.id] = out
        else:
            hz_in = None
    fact = interval.tf_maxpool2d(L, bounds)
    if hz_in is not None and L.id in tf._hz_cache:
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

    return hz_inherit_known_nonempty(HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    ), hz, reason="conv2d")


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
    return hz_inherit_known_nonempty(HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    ), hz, reason="spatial_affine")


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


def sparse_conv2d_matrix_from_layer(layer) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build the exact NCHW sparse affine matrix for an ACT CONV2D layer."""

    weight = _to_numpy(layer.params["weight"]).astype(np.float64, copy=False)
    bias = layer.params.get("bias")
    bias_np = None if bias is None else _to_numpy(bias).astype(np.float64, copy=False).reshape(-1)
    out_ch, in_ch_per_group, kh, kw = [int(v) for v in weight.shape]
    groups = int(layer.params.get("groups", 1))
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    bsz, in_ch, in_h, in_w = _shape4(layer.params["input_shape"])
    out_bsz, out_ch_shape, out_h, out_w = _shape4(layer.params["output_shape"])
    if bsz != out_bsz or out_ch != out_ch_shape:
        raise ValueError(f"conv2d shape mismatch at layer {getattr(layer, 'id', '?')}")
    if in_ch_per_group * groups != in_ch:
        raise ValueError(f"conv2d group/input mismatch at layer {getattr(layer, 'id', '?')}")

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    out_ch_per_group = out_ch // groups
    for n in range(bsz):
        for co in range(out_ch):
            group = co // out_ch_per_group
            ci_base = group * in_ch_per_group
            for oh in range(out_h):
                ih0 = oh * stride[0] - padding[0]
                for ow in range(out_w):
                    iw0 = ow * stride[1] - padding[1]
                    out_idx = ((n * out_ch + co) * out_h + oh) * out_w + ow
                    cur_cols: List[int] = []
                    cur_vals: List[float] = []
                    for ci_local in range(in_ch_per_group):
                        ci = ci_base + ci_local
                        for rr in range(kh):
                            ih = ih0 + rr * dilation[0]
                            if ih < 0 or ih >= in_h:
                                continue
                            for cc in range(kw):
                                iw = iw0 + cc * dilation[1]
                                if iw < 0 or iw >= in_w:
                                    continue
                                val = float(weight[co, ci_local, rr, cc])
                                if val == 0.0:
                                    continue
                                cur_cols.append(((n * in_ch + ci) * in_h + ih) * in_w + iw)
                                cur_vals.append(val)
                    if cur_cols:
                        arr_c = np.asarray(cur_cols, dtype=np.int32)
                        rows.append(np.full(arr_c.size, out_idx, dtype=np.int32))
                        cols.append(arr_c)
                        data.append(np.asarray(cur_vals, dtype=np.float64))

    if rows:
        rr = np.concatenate(rows)
        cc = np.concatenate(cols)
        dd = np.concatenate(data)
    else:
        rr = cc = np.empty(0, dtype=np.int32)
        dd = np.empty(0, dtype=np.float64)
    mat = sp.csr_matrix(
        (dd, (rr, cc)),
        shape=(bsz * out_ch * out_h * out_w, bsz * in_ch * in_h * in_w),
        dtype=np.float64,
    )
    mat.eliminate_zeros()
    if bias_np is None:
        bvec = np.zeros(mat.shape[0], dtype=np.float64)
    else:
        bvec = np.tile(np.repeat(bias_np, out_h * out_w), bsz)
    return mat, bvec


def sparse_hz_apply_conv2d_layer(hz: SparseHZono, layer) -> SparseHZono:
    W, b = sparse_conv2d_matrix_from_layer(layer)
    return sparse_hz_linear(hz, W, b)


def sparse_convtranspose2d_matrix_from_layer(layer) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build the exact NCHW sparse affine matrix for ACT CONVTRANSPOSE2D."""

    weight = _to_numpy(layer.params["weight"]).astype(np.float64, copy=False)
    bias = layer.params.get("bias")
    bias_np = None if bias is None else _to_numpy(bias).astype(np.float64, copy=False).reshape(-1)
    in_ch, out_ch_per_group, kh, kw = [int(v) for v in weight.shape]
    groups = int(layer.params.get("groups", 1))
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    bsz, in_ch_shape, in_h, in_w = _shape4(layer.params["input_shape"])
    out_bsz, out_ch, out_h, out_w = _shape4(layer.params["output_shape"])
    if bsz != out_bsz or in_ch != in_ch_shape:
        raise ValueError(f"convtranspose2d shape mismatch at layer {getattr(layer, 'id', '?')}")
    if out_ch != out_ch_per_group * groups:
        raise ValueError(f"convtranspose2d group/output mismatch at layer {getattr(layer, 'id', '?')}")

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    in_ch_per_group = in_ch // groups
    for n in range(bsz):
        for ci in range(in_ch):
            group = ci // in_ch_per_group
            co_base = group * out_ch_per_group
            for ih in range(in_h):
                oh0 = ih * stride[0] - padding[0]
                for iw in range(in_w):
                    ow0 = iw * stride[1] - padding[1]
                    in_idx = ((n * in_ch + ci) * in_h + ih) * in_w + iw
                    cur_rows: List[int] = []
                    cur_vals: List[float] = []
                    for co_local in range(out_ch_per_group):
                        co = co_base + co_local
                        for rr in range(kh):
                            oh = oh0 + rr * dilation[0]
                            if oh < 0 or oh >= out_h:
                                continue
                            for cc in range(kw):
                                ow = ow0 + cc * dilation[1]
                                if ow < 0 or ow >= out_w:
                                    continue
                                val = float(weight[ci, co_local, rr, cc])
                                if val == 0.0:
                                    continue
                                cur_rows.append(((n * out_ch + co) * out_h + oh) * out_w + ow)
                                cur_vals.append(val)
                    if cur_rows:
                        arr_r = np.asarray(cur_rows, dtype=np.int32)
                        rows.append(arr_r)
                        cols.append(np.full(arr_r.size, in_idx, dtype=np.int32))
                        data.append(np.asarray(cur_vals, dtype=np.float64))

    if rows:
        rr = np.concatenate(rows)
        cc = np.concatenate(cols)
        dd = np.concatenate(data)
    else:
        rr = cc = np.empty(0, dtype=np.int32)
        dd = np.empty(0, dtype=np.float64)
    mat = sp.csr_matrix(
        (dd, (rr, cc)),
        shape=(bsz * out_ch * out_h * out_w, bsz * in_ch * in_h * in_w),
        dtype=np.float64,
    )
    mat.eliminate_zeros()
    if bias_np is None:
        bvec = np.zeros(mat.shape[0], dtype=np.float64)
    else:
        bvec = np.tile(np.repeat(bias_np, out_h * out_w), bsz)
    return mat, bvec


def sparse_hz_apply_convtranspose2d_layer(hz: SparseHZono, layer) -> SparseHZono:
    W, b = sparse_convtranspose2d_matrix_from_layer(layer)
    return sparse_hz_linear(hz, W, b)


def sparse_avgpool2d_matrix_from_layer(layer) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build the exact NCHW sparse affine matrix for ACT AVGPOOL2D."""

    bsz, ch, in_h, in_w = _shape4(layer.params["input_shape"])
    out_bsz, out_ch, out_h, out_w = _shape4(layer.params["output_shape"])
    if bsz != out_bsz or ch != out_ch:
        raise ValueError(f"avgpool2d shape mismatch at layer {getattr(layer, 'id', '?')}")
    kh, kw = _pair(layer.params["kernel_size"])
    stride = layer.params.get("stride", layer.params["kernel_size"])
    sh, sw = _pair(stride)
    ph, pw = _pair(layer.params.get("padding", 0))
    denom = float(kh * kw)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for n in range(bsz):
        for c in range(ch):
            for oh in range(out_h):
                ih0 = oh * sh - ph
                for ow in range(out_w):
                    iw0 = ow * sw - pw
                    out_idx = ((n * ch + c) * out_h + oh) * out_w + ow
                    cur_cols: List[int] = []
                    for ki in range(kh):
                        ih = ih0 + ki
                        if ih < 0 or ih >= in_h:
                            continue
                        for kj in range(kw):
                            iw = iw0 + kj
                            if iw < 0 or iw >= in_w:
                                continue
                            cur_cols.append(((n * ch + c) * in_h + ih) * in_w + iw)
                    if cur_cols:
                        rows.append(np.full(len(cur_cols), out_idx, dtype=np.int32))
                        cols.append(np.asarray(cur_cols, dtype=np.int32))
    rr = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    cc = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    dd = np.full(rr.size, 1.0 / denom, dtype=np.float64)
    mat = sp.csr_matrix(
        (dd, (rr, cc)),
        shape=(bsz * ch * out_h * out_w, bsz * ch * in_h * in_w),
        dtype=np.float64,
    )
    mat.eliminate_zeros()
    return mat, np.zeros(mat.shape[0], dtype=np.float64)


def sparse_hz_apply_avgpool2d_layer(hz: SparseHZono, layer) -> SparseHZono:
    W, b = sparse_avgpool2d_matrix_from_layer(layer)
    return sparse_hz_linear(hz, W, b)


def sparse_maxpool2d_candidate_rows(layer) -> List[np.ndarray]:
    """Return one output-to-input row map per MaxPool2D window candidate.

    Invalid padding locations are encoded as ``-1``.  The exact MaxPool fold
    replaces those with a constant below the input lower bound, so padding
    locations cannot win a maximum.
    """

    bsz, ch, in_h, in_w = _shape4(layer.params["input_shape"])
    kh, kw = _pair(layer.params["kernel_size"])
    stride = layer.params.get("stride", layer.params["kernel_size"])
    sh, sw = _pair(stride)
    ph, pw = _pair(layer.params.get("padding", 0))
    dh, dw = _pair(layer.params.get("dilation", 1))
    output_shape = layer.params.get("output_shape")
    if output_shape is None:
        out_bsz, out_ch = bsz, ch
        out_h = (in_h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (in_w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    else:
        out_bsz, out_ch, out_h, out_w = _shape4(output_shape)
    if bsz != out_bsz or ch != out_ch:
        raise ValueError(f"maxpool2d shape mismatch at layer {getattr(layer, 'id', '?')}")

    n_out = bsz * ch * out_h * out_w
    candidates: List[np.ndarray] = []
    for ki in range(kh):
        for kj in range(kw):
            rows = np.full(n_out, -1, dtype=np.int64)
            for n in range(bsz):
                for c in range(ch):
                    for oh in range(out_h):
                        ih = oh * sh - ph + ki * dh
                        for ow in range(out_w):
                            iw = ow * sw - pw + kj * dw
                            out_idx = ((n * ch + c) * out_h + oh) * out_w + ow
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                rows[out_idx] = ((n * ch + c) * in_h + ih) * in_w + iw
            candidates.append(rows)
    return candidates

def sparse_hz_apply_maxpool2d_layer(
    hz: SparseHZono,
    layer,
    input_bounds: Bounds,
    *,
    compressed_relu: bool = False,
    valid_cuts: bool = False,
) -> SparseHZono:
    """Exact sparse MaxPool2D as a fold of ``max(a,b)=b+ReLU(a-b)``."""

    candidates = sparse_maxpool2d_candidate_rows(layer)
    if not candidates:
        raise ValueError(f"maxpool2d layer {getattr(layer, 'id', '?')} has no candidates")
    in_lb = input_bounds.lb.detach().cpu().double().numpy().reshape(-1)
    in_ub = input_bounds.ub.detach().cpu().double().numpy().reshape(-1)
    fill_value = float(np.min(in_lb) - 1.0) if in_lb.size else -1.0

    def gather_bounds(rows: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        row_idx = np.asarray(rows, dtype=np.int64).reshape(-1)
        valid = row_idx >= 0
        lb = np.full(row_idx.size, fill_value, dtype=np.float64)
        ub = np.full(row_idx.size, fill_value, dtype=np.float64)
        if np.any(valid):
            lb[valid] = in_lb[row_idx[valid]]
            ub[valid] = in_ub[row_idx[valid]]
        return lb, ub

    out = sparse_hz_gather_rows_like(hz, candidates[0], fill_value=fill_value)
    cur_lb, cur_ub = gather_bounds(candidates[0])
    for rows in candidates[1:]:
        nxt = sparse_hz_gather_rows_like(hz, rows, fill_value=fill_value, template=out)
        nxt_lb, nxt_ub = gather_bounds(rows)
        diff = sparse_hz_sub_same_frame(out, nxt)
        pre_bounds = Bounds(
            lb=torch.from_numpy(cur_lb - nxt_ub).reshape(1, -1).double(),
            ub=torch.from_numpy(cur_ub - nxt_lb).reshape(1, -1).double(),
        )
        relu = sparse_hz_apply_relu_exact(
            diff,
            pre_bounds=pre_bounds,
            compressed=compressed_relu,
            valid_cuts=valid_cuts,
        )
        out = sparse_hz_add_same_frame(nxt, relu)
        cur_lb = np.maximum(cur_lb, nxt_lb)
        cur_ub = np.maximum(cur_ub, nxt_ub)
    return out
