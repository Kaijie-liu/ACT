# ===- act/back_end/hybridz_tf/tf_mlp.py - HybridZ MLP Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ MLP Transfer Functions. Implements HybridZ-based transfer functions
#   for MLP layers including dense, activation, and element-wise operations.
#
# ===---------------------------------------------------------------------===#

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from act.back_end.core import Bounds, Fact
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import (
    HZono,
    SparseHZono,
    hz_multiply,
    hz_add_const,
    hz_from_bounds,
    hz_compute_bounds,
    hz_fresh_col_ids,
    hz_inherit_known_nonempty,
    hz_mark_known_nonempty,
)
import act.back_end.interval_tf.tf_mlp as interval
import act.back_end.interval_tf.tf_cnn as interval_cnn


def sparse_hz_apply_layer(L, hz: SparseHZono, input_bounds: Bounds, result: Fact, tf):
    """Apply sparse-HZ propagation for MLP/structural layer kinds.

    Returns ``(handled, out, drop_reason)``.  ``handled=False`` means another
    layer-family module may handle the layer.  ``drop_reason`` means this layer
    is in-family but cannot be represented by the sparse exact-HZ carrier.
    """

    k = L.kind.upper()
    if k == LayerKind.DENSE.value:
        return True, sparse_hz_apply_dense_layer(hz, L), None
    if k == LayerKind.BIAS.value:
        return True, sparse_hz_add_const(hz, L.params["c"]), None
    if k == LayerKind.SCALE.value:
        return True, sparse_hz_scale(hz, L.params["a"]), None
    if k == LayerKind.BN.value:
        return True, sparse_hz_add_const(sparse_hz_scale(hz, L.params["A"]), L.params["c"]), None
    if k == LayerKind.RELU.value:
        return True, sparse_hz_apply_relu_exact(
            hz,
            pre_bounds=input_bounds,
            compressed=getattr(tf, "_relu_compressed", False),
            valid_cuts=getattr(tf, "_relu_valid_cuts", False),
        ), None
    if k == LayerKind.SIGMOID.value:
        return True, sparse_hz_apply_sigmoid_piecewise(
            hz,
            K=getattr(tf, "_sigmoid_K", 2),
            domain_cuts=getattr(tf, "_scurve_domain_cuts", False),
            graph_cuts=getattr(tf, "_scurve_graph_cuts", False),
        ), None
    if k == LayerKind.TANH.value:
        return True, sparse_hz_apply_tanh_piecewise(
            hz,
            K=getattr(tf, "_tanh_K", 1),
            domain_cuts=getattr(tf, "_scurve_domain_cuts", False),
            graph_cuts=getattr(tf, "_scurve_graph_cuts", False),
        ), None
    if k == LayerKind.MATMUL.value:
        preds = tf._net.preds.get(L.id, [])
        if len(preds) != 2:
            return True, None, "unsupported_sparse_matmul_routing"
        left = tf._sparse_hz_cache.get(preds[0])
        right = tf._sparse_hz_cache.get(preds[1])
        if left is None or right is None:
            return True, None, "missing_sparse_matmul_input"
        if sparse_hz_is_point(right):
            return True, sparse_hz_apply_matmul_const_layer(
                left, right, L, variable_is_left=True,
            ), None
        if sparse_hz_is_point(left):
            return True, sparse_hz_apply_matmul_const_layer(
                right, left, L, variable_is_left=False,
            ), None
        return True, None, "unsupported_sparse_matmul_var_var"
    if k in {
        LayerKind.FLATTEN.value,
        LayerKind.RESHAPE.value,
        LayerKind.SQUEEZE.value,
        LayerKind.UNSQUEEZE.value,
        LayerKind.TRANSPOSE.value,
    }:
        return True, hz, None
    if k == LayerKind.SLICE.value:
        rows = sparse_slice_row_indices(L, hz.n_out)
        if rows is None or rows.size != result.bounds.lb.numel():
            return True, None, "unsupported_slice_row_map"
        return True, sparse_hz_gather_rows(hz, rows), None
    if k == LayerKind.GATHER.value:
        rows = sparse_gather_row_indices(L, hz.n_out)
        if rows is None or rows.size != result.bounds.lb.numel():
            return True, None, "unsupported_gather_row_map"
        return True, sparse_hz_gather_rows(hz, rows), None
    if k == LayerKind.EXPAND.value:
        rows = sparse_expand_row_indices(L, hz.n_out)
        if rows is None or rows.size != result.bounds.lb.numel():
            return True, None, "unsupported_expand_row_map"
        return True, sparse_hz_gather_rows(hz, rows), None
    if k == LayerKind.REDUCE_SUM.value:
        rows = sparse_reduce_sum_row_indices(L, hz.n_out, result.bounds.lb.numel())
        if rows is None:
            return True, None, "unsupported_reduce_sum_row_map"
        return True, sparse_hz_reduce_sum_rows(hz, rows, result.bounds.lb.numel()), None
    if k == LayerKind.UPSAMPLE.value:
        rows = sparse_upsample_nearest_row_indices(L, hz.n_out, result.bounds.lb.numel())
        if rows is None:
            return True, None, "unsupported_upsample_row_map"
        return True, sparse_hz_gather_rows(hz, rows), None
    if k == LayerKind.ADD.value:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._sparse_hz_cache.get(pid) for pid in preds[:2]]
        if len(parts) != 2 or any(part is None for part in parts):
            return True, None, "missing_sparse_add_input"
        return True, sparse_hz_add_same_frame(parts[0], parts[1]), None
    if k == LayerKind.SUB.value:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._sparse_hz_cache.get(pid) for pid in preds[:2]]
        if len(parts) != 2 or any(part is None for part in parts):
            return True, None, "missing_sparse_sub_input"
        return True, sparse_hz_sub_same_frame(parts[0], parts[1]), None
    if k == LayerKind.CONCAT.value:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._sparse_hz_cache.get(pid) for pid in preds]
        if not parts or any(part is None for part in parts):
            return True, None, "missing_sparse_concat_input"
        return True, sparse_hz_concat(parts), None
    if k == LayerKind.CONSTANT.value:
        return True, sparse_hz_from_bounds(result.bounds), None
    return False, None, None


def _hz_fact(fact: Fact, hz: HZono) -> Fact:
    """Combine HZ-refined bounds (flat ``(n, 1)`` shape) with interval's
    batch-aware fact: reshape HZ bounds to match ``fact.bounds`` and keep
    interval's constraint set. Use everywhere a hybridz handler returns
    after refining the HZ cache.
    """
    hb = hz_compute_bounds(hz)
    lb = torch.maximum(hb.lb.reshape_as(fact.bounds.lb), fact.bounds.lb)
    ub = torch.minimum(hb.ub.reshape_as(fact.bounds.ub), fact.bounds.ub)
    return Fact(bounds=Bounds(lb=lb, ub=ub), cons=fact.cons)


# ============================================================================
# Batch-native HZ helpers
# ----------------------------------------------------------------------------
# HZono stores ``c: (n, 1)``, ``Gc: (n, ng)``, ``Gb: (n, nb)`` where the
# leading dimension ``n`` is the *flattened* output size of the encoded
# layer including any leading batch axis ``B``. For per-channel ops
# (DENSE, BIAS, SCALE) we recover ``B`` from ``n // per_channel`` and
# operate via broadcasted 3D matmul / per-row scaling so that no
# block-diagonal weight is materialised.
# ============================================================================


def _hz_apply_per_batch_linear(hz: HZono, W: torch.Tensor, B: int) -> HZono:
    """Apply ``y = W x`` independently to each of ``B`` instances stacked
    along the leading axis of ``hz``. Equivalent to
    ``hz_multiply(hz, block_diag(W, ...))`` without materialising the
    block-diagonal matrix.
    """
    in_dim = W.shape[1]
    out_dim = W.shape[0]
    if B == 1:
        return hz_multiply(hz, W)
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    # (out, in) @ (B, in, *) broadcasts → (B, out, *)
    c3 = hz.c.view(B, in_dim, 1)
    new_c = (W @ c3).reshape(B * out_dim, 1)
    if ng:
        new_Gc = (W @ hz.Gc.view(B, in_dim, ng)).reshape(B * out_dim, ng)
    else:
        new_Gc = hz.Gc.new_zeros(B * out_dim, 0)
    if nb:
        new_Gb = (W @ hz.Gb.view(B, in_dim, nb)).reshape(B * out_dim, nb)
    else:
        new_Gb = hz.Gb.new_zeros(B * out_dim, 0)
    return hz_inherit_known_nonempty(HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    ), hz, reason="affine")


def _hz_add_per_channel(hz: HZono, v: torch.Tensor, B: int) -> HZono:
    """Add per-channel constant ``v: (out,)`` to each of ``B`` stacked
    instances in ``hz.c``. ``hz.c`` has shape ``(B*out, 1)``.
    """
    v = v.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        v = v.repeat(B)
    return hz_add_const(hz, v.view(-1, 1))


def _hz_scale_per_channel(hz: HZono, a: torch.Tensor, B: int) -> HZono:
    """Multiply hz fields by per-channel ``a: (out,)``. ``hz.c`` shape
    is ``(B*out, 1)``; we broadcast ``a`` once per batch via repeat.
    Equivalent to ``hz_multiply(hz, diag(a_repeated))`` without building
    the diagonal matrix.
    """
    a = a.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        a = a.repeat(B)
    a_col = a.view(-1, 1)
    return hz_inherit_known_nonempty(HZono(
        c=a_col * hz.c,
        Gc=a_col * hz.Gc,
        Gb=a_col * hz.Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    ), hz, reason="affine")


def _hz_is_point(hz: HZono) -> bool:
    gc_zero = hz.Gc.numel() == 0 or bool((hz.Gc.abs() <= 1e-12).all())
    gb_zero = hz.Gb.numel() == 0 or bool((hz.Gb.abs() <= 1e-12).all())
    return gc_zero and gb_zero


def _broadcast_flat(v: torch.Tensor, n: int) -> torch.Tensor:
    v = v.flatten()
    if v.numel() == n:
        return v
    if v.numel() == 1:
        return v.expand(n)
    if n % v.numel() == 0:
        return v.repeat(n // v.numel())
    raise ValueError(f"cannot broadcast {v.numel()} values to {n}")


def _hz_scale_elementwise(hz: HZono, a: torch.Tensor) -> HZono:
    """Exact elementwise scale by a point tensor."""
    a = _broadcast_flat(a.to(dtype=hz.c.dtype, device=hz.c.device), hz.c.shape[0])
    acol = a.view(-1, 1)
    return hz_inherit_known_nonempty(HZono(
        c=acol * hz.c,
        Gc=acol * hz.Gc,
        Gb=acol * hz.Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    ), hz, reason="affine")


def _prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out


def _hz_matmul_const(L, hz: HZono, const: torch.Tensor, *, variable_is_left: bool) -> HZono | None:
    """Exact linear map for MatMul when one operand is a point tensor."""
    dtype, device = hz.c.dtype, hz.c.device
    x_shape = tuple(int(d) for d in L.params["x_shape"])
    y_shape = tuple(int(d) for d in L.params["y_shape"])
    in_shape = x_shape if variable_is_left else y_shape
    in_dim = _prod(in_shape)
    if in_dim == 0 or hz.c.shape[0] % in_dim != 0:
        return None
    C = const.to(dtype=dtype, device=device).flatten()
    if variable_is_left:
        W = C.view(*y_shape)
        eye = torch.eye(in_dim, dtype=dtype, device=device).view(in_dim, *x_shape)
        out = torch.matmul(eye, W).reshape(in_dim, -1)
    else:
        W = C.view(*x_shape)
        eye = torch.eye(in_dim, dtype=dtype, device=device).view(in_dim, *y_shape)
        out = torch.matmul(W, eye).reshape(in_dim, -1)
    R = out.t().contiguous()
    return _hz_apply_per_batch_linear(hz, R, hz.c.shape[0] // in_dim)


# ============================================================================
# HZ layer functions: HZono -> Optional[HZono] per layer kind
# Each takes (L, hz_in, tf) and returns the transformed HZono or None.
# ============================================================================


# --- HZ transfer functions (MLP) ---


def tf_dense(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        W = L.params["weight"].to(hz_in.c)
        in_dim = W.shape[1]
        B = hz_in.c.shape[0] // in_dim
        hz = _hz_apply_per_batch_linear(hz_in, W, B)
        bias = L.params.get("bias")
        if bias is not None:
            hz = _hz_add_per_channel(hz, bias, B)
        tf._hz_cache[L.id] = hz
    fact = interval.tf_dense(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_bias(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        c = L.params["c"].to(hz_in.c)
        if c.ndim == 1:
            B = hz_in.c.shape[0] // c.numel()
            tf._hz_cache[L.id] = _hz_add_per_channel(hz_in, c, B)
        else:
            tf._hz_cache[L.id] = hz_add_const(hz_in, c)
    fact = interval.tf_bias(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_scale(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        a = L.params["a"].to(hz_in.c).flatten()
        B = hz_in.c.shape[0] // a.numel()
        tf._hz_cache[L.id] = _hz_scale_per_channel(hz_in, a, B)
    fact = interval.tf_scale(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_relu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_relu(L, bounds)
    if hz_in is not None:
        tight = getattr(tf, "_relu_tight_bounds", False)
        cuts = getattr(tf, "_relu_valid_cuts", False)
        compressed = getattr(tf, "_relu_compressed", False)
        cell_budget = getattr(tf, "_hz_cell_budget", 100_000_000)
        out = hz_apply_relu(
            hz_in, tight_bounds=tight, valid_cuts=cuts,
            cell_budget=cell_budget, compressed=compressed)
        if out is None:
            tf._hz_cache.pop(L.id, None)
            return fact
        tf._hz_cache[L.id] = hz_reduce(out)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_lrelu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(
            hz_apply_leaky_relu(hz_in, float(L.params.get("negative_slope", 0.01)))
        )
    fact = interval.tf_lrelu(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_tanh(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        out = hz_apply_tanh(
            hz_in, K=tf._tanh_K,
            cell_budget=getattr(tf, "_hz_cell_budget", 100_000_000))
        if out is None:
            tf._hz_cache.pop(L.id, None)
        else:
            tf._hz_cache[L.id] = hz_reduce(out)
    fact = interval.tf_tanh(L, bounds)
    if hz_in is not None and L.id in tf._hz_cache:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_sigmoid(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        out = hz_apply_sigmoid(
            hz_in, K=tf._sigmoid_K,
            cell_budget=getattr(tf, "_hz_cell_budget", 100_000_000))
        if out is None:
            tf._hz_cache.pop(L.id, None)
        else:
            tf._hz_cache[L.id] = hz_reduce(out)
    fact = interval.tf_sigmoid(L, bounds)
    if hz_in is not None and L.id in tf._hz_cache:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_abs(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        bds = hz_compute_bounds(hz_in)
        lb_out = torch.where(
            bds.lb >= 0,
            bds.lb,
            torch.where(bds.ub <= 0, -bds.ub, torch.zeros_like(bds.lb)),
        )
        tf._hz_cache[L.id] = hz_from_bounds(
            Bounds(lb=lb_out, ub=torch.maximum(bds.lb.abs(), bds.ub.abs())),
            dtype,
            device,
        )
    fact = interval.tf_abs(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_bn(L, bounds, tf):
    # BatchNorm is the per-channel affine map y = A*x + c (A,c per channel).
    # The per-channel HZ helpers already handle B>1 via repeat, so this is an
    # EXACT HZ transfer — no interval fallback needed (was a spurious fallback).
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        A, c = L.params["A"], L.params["c"]
        B = hz_in.c.shape[0] // A.numel()
        tf._hz_cache[L.id] = _hz_add_per_channel(
            _hz_scale_per_channel(hz_in, A, B), c, B)
    fact = interval.tf_bn(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_add(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            from act.back_end.solver.solver_hz import hz_sgm_add
            tf._hz_cache[L.id] = hz_sgm_add(hz_in, hz2)
        else:
            hz_in = None
    fact = interval.tf_add(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_mul(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_mul(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            if _hz_is_point(hz2):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz_in, hz2.c)
            elif _hz_is_point(hz_in):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz2, hz_in.c)
            else:
                tf._hz_cache.pop(L.id, None)
                hz_in = None
                return fact
        else:
            hz_in = None
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_div(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_div(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None and _hz_is_point(hz2):
            denom = _broadcast_flat(hz2.c.to(dtype=hz_in.c.dtype, device=hz_in.c.device),
                                    hz_in.c.shape[0])
            if bool((denom.abs() > 1e-12).all()):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz_in, 1.0 / denom)
                return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_constant(L, bounds, tf):
    val = L.params["value"].flatten()
    n = val.numel()
    # When the surrounding net is batched (e.g., upstream ADD sibling is
    # ``[B, *shape]``), replicate the constant per batch element so the
    # downstream HZ Minkowski-sum / element-wise ops see matching sizes.
    if bounds is not None and n > 0:
        in_numel = int(bounds.lb.numel())
        if in_numel > 0 and in_numel % n == 0:
            B = in_numel // n
            if B > 1:
                val = val.repeat(B)
                n = val.numel()
    tf._hz_cache[L.id] = hz_mark_known_nonempty(HZono(
        c=val.view(-1, 1),
        Gc=val.new_zeros(n, 0),
        Gb=val.new_zeros(n, 0),
        Ac=val.new_zeros(0, 0),
        Ab=val.new_zeros(0, 0),
        b=val.new_zeros(0, 1),
    ), "constant")
    return interval.tf_constant(L, bounds)


def tf_sign(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_sign(L, bounds)


def tf_compare(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_compare(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )


def tf_where(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_where(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2),
    )


def tf_matmul(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_matmul(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            out = None
            if _hz_is_point(hz2):
                out = _hz_matmul_const(L, hz_in, hz2.c, variable_is_left=True)
            elif _hz_is_point(hz_in):
                out = _hz_matmul_const(L, hz2, hz_in.c, variable_is_left=False)
            if out is not None:
                tf._hz_cache[L.id] = out
                return _hz_fact(fact, tf._hz_cache[L.id])
    tf._hz_cache.pop(L.id, None)
    return fact


def tf_arg_extremum(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_arg_extremum(L, bounds)


def tf_upsample(L, bounds, tf):
    fact = interval_cnn.tf_upsample(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is None:
        return fact
    rows = sparse_upsample_nearest_row_indices(L, hz_in.c.shape[0], len(L.out_vars))
    if rows is None:
        tf._hz_cache.pop(L.id, None)
        return fact
    row_idx = torch.as_tensor(rows, dtype=torch.long, device=hz_in.c.device)
    tf._hz_cache[L.id] = _hz_gather_rows(hz_in, row_idx)
    return _hz_fact(fact, tf._hz_cache[L.id])


def tf_scatter_nd(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_scatter_nd(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2),
    )


def tf_reduce_sum(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_reduce_sum(L, bounds)
    if hz_in is not None:
        rows = sparse_reduce_sum_row_indices(L, hz_in.c.shape[0], fact.bounds.lb.numel())
        if rows is not None:
            ri = torch.as_tensor(rows, dtype=torch.long, device=hz_in.c.device)
            out_n = int(fact.bounds.lb.numel())
            c = hz_in.c.new_zeros(out_n, 1)
            Gc = hz_in.Gc.new_zeros(out_n, hz_in.Gc.shape[1])
            Gb = hz_in.Gb.new_zeros(out_n, hz_in.Gb.shape[1])
            c.index_add_(0, ri, hz_in.c)
            if hz_in.Gc.shape[1]:
                Gc.index_add_(0, ri, hz_in.Gc)
            if hz_in.Gb.shape[1]:
                Gb.index_add_(0, ri, hz_in.Gb)
            tf._hz_cache[L.id] = hz_inherit_known_nonempty(HZono(
                c=c, Gc=Gc, Gb=Gb,
                Ac=hz_in.Ac, Ab=hz_in.Ab, b=hz_in.b,
                eq_mask=hz_in.eq_mask,
                col_ids=hz_in.col_ids, bcol_ids=hz_in.bcol_ids),
                hz_in, reason="reduce_sum")
        else:
            tf._hz_cache[L.id] = hz_from_bounds(
                fact.bounds, fact.bounds.lb.dtype, fact.bounds.lb.device)
    return fact


def tf_concat(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._hz_cache.get(pid) for pid in preds]
        if all(p is not None for p in parts):
            from act.back_end.solver.solver_hz import hz_concat
            tf._hz_cache[L.id] = hz_concat(parts)
        else:
            hz_in = None
    fact = interval.tf_concat(
        L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_sub(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            # x1 - x2 with share-merged subtraction (correlated => exact).
            from act.back_end.solver.solver_hz import hz_sub as _hz_sub
            tf._hz_cache[L.id] = _hz_sub(hz_in, hz2)
        else:
            hz_in = None
    fact = interval.tf_sub(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_flatten(L, bounds, tf):
    # Flatten/Reshape only reorder the value layout; the HZ row order is already
    # the flattened layout, so this is a literal pass-through (exact). Keeping the
    # inherited HZ in the cache stops apply()'s box re-seed from destroying it.
    fact = interval_cnn.tf_flatten(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_reshape(L, bounds, tf):
    fact = interval.tf_reshape(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def _hz_rebind(hz: HZono) -> HZono:
    """Return a NEW HZono wrapping the same tensors. Flatten/Reshape are layout
    no-ops for the HZ (rows already stored flattened), but the handler must hand
    back a *distinct* object so apply()'s identity-based box re-seed does not fire."""
    return hz_inherit_known_nonempty(
        HZono(c=hz.c, Gc=hz.Gc, Gb=hz.Gb, Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
              eq_mask=hz.eq_mask, col_ids=hz.col_ids, bcol_ids=hz.bcol_ids),
        hz,
        reason="rebind",
    )


def _hz_gather_rows(hz: HZono, row_idx: torch.Tensor) -> HZono:
    """Select / permute / repeat HZ output ROWS by ``row_idx`` (n_out,). The
    constraint block (Ac/Ab/b) and ``col_ids`` reference generator COLUMNS, so
    they are unchanged -- a structural op only remaps which output coordinate
    reads which latent row. Exact (the output is an exact gather of inputs)."""
    ri = row_idx.to(device=hz.c.device, dtype=torch.long)
    return hz_inherit_known_nonempty(
        HZono(c=hz.c[ri], Gc=hz.Gc[ri], Gb=hz.Gb[ri],
              Ac=hz.Ac, Ab=hz.Ab, b=hz.b, eq_mask=hz.eq_mask,
              col_ids=hz.col_ids, bcol_ids=hz.bcol_ids),
        hz,
        reason="gather_rows",
    )


def tf_squeeze(L, bounds, tf):
    # Squeeze/Unsqueeze/Transpose only change tensor SHAPE; the framework's
    # interval handler returns identity bounds (the permutation is tracked in
    # the layer's EQ constraint), so the flattened HZ row order is unchanged ->
    # pass-through, consistent with interval.
    fact = interval.tf_squeeze(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_unsqueeze(L, bounds, tf):
    fact = interval.tf_unsqueeze(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_transpose(L, bounds, tf):
    fact = interval.tf_transpose(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_slice(L, bounds, tf):
    fact = interval.tf_slice(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = sparse_slice_row_indices(L, hz_in.c.shape[0])
        if rows is not None and rows.size == fact.bounds.lb.shape[-1]:
            ri = torch.as_tensor(rows, dtype=torch.long, device=hz_in.c.device)
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, ri)
            return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_gather(L, bounds, tf):
    fact = interval.tf_gather(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = sparse_gather_row_indices(L, hz_in.c.shape[0])
        if rows is not None and rows.size == fact.bounds.lb.numel():
            ri = torch.as_tensor(rows, dtype=torch.long, device=hz_in.c.device)
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, ri)
            return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_expand(L, bounds, tf):
    fact = interval.tf_expand(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = sparse_expand_row_indices(L, hz_in.c.shape[0])
        if rows is not None and rows.size == fact.bounds.lb.shape[-1]:
            ri = torch.as_tensor(rows, dtype=torch.long, device=hz_in.c.device)
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, ri)
            return _hz_fact(fact, tf._hz_cache[L.id])
    return fact




_RELU_TIGHT_MAX_DIM = 4096  # LP-tight does 2n scipy solves; above this (wide conv


def _relu_preact_bounds(hz: HZono, tight: bool):
    """Pre-activation [alpha,beta] for ReLU classification. Fast |Gc|+|Gb| box by
    default; the constrained LP relaxation (scipy/HiGHS, sound) when ``tight``
    and constraints exist AND the layer isn't too wide.
    Both enclose the true range; LP-relax is tighter, so it feeds a tighter ReLU
    encoding. Fast-box stable rows are already soundly classified, so tight LPs
    are only needed for ambiguous rows unless HZ_RELU_TIGHT_ALL_ROWS=1 requests
    full-layer tightening."""
    fb = hz_compute_bounds(hz)
    if not tight or hz.Ac.shape[0] == 0 or hz.c.shape[0] > _RELU_TIGHT_MAX_DIM:
        return fb
    flat_lb = fb.lb.reshape(-1)
    flat_ub = fb.ub.reshape(-1)
    ambiguous = torch.nonzero((flat_lb < 0) & (flat_ub > 0), as_tuple=False).reshape(-1)
    if ambiguous.numel() == 0:
        return fb
    try:
        from act.back_end.solver.solver_hz import hz_compute_lp_bounds
        if os.environ.get("HZ_RELU_TIGHT_ALL_ROWS", "").strip().lower() in {
            "1", "true", "yes", "on"}:
            tb = hz_compute_lp_bounds(hz)
            return Bounds(lb=torch.maximum(tb.lb, fb.lb),
                          ub=torch.minimum(tb.ub, fb.ub))
        try:
            max_rows = int(os.environ.get("HZ_RELU_TIGHT_MAX_ROWS", "0") or 0)
        except Exception:
            max_rows = 0
        if max_rows > 0 and ambiguous.numel() > max_rows:
            score = torch.minimum(flat_lb[ambiguous].abs(), flat_ub[ambiguous].abs())
            keep = torch.argsort(score)[:max_rows]
            ambiguous = ambiguous[keep]
        idx = ambiguous.to(device=flat_lb.device)
        amb_np = ambiguous.detach().cpu().numpy()
        tb = hz_compute_lp_bounds(
            hz,
            rows=amb_np,
            base_lb=flat_lb[idx].detach().cpu().numpy(),
            base_ub=flat_ub[idx].detach().cpu().numpy(),
            relu_stability=True,
        )
        lb = flat_lb.clone()
        ub = flat_ub.clone()
        tb_lb = tb.lb.reshape(-1).to(device=flat_lb.device, dtype=flat_lb.dtype)
        tb_ub = tb.ub.reshape(-1).to(device=flat_ub.device, dtype=flat_ub.dtype)
        lb[idx] = torch.maximum(tb_lb, flat_lb[idx])
        ub[idx] = torch.minimum(tb_ub, flat_ub[idx])
        return Bounds(lb=lb.reshape_as(fb.lb), ub=ub.reshape_as(fb.ub))
    except Exception:
        return fb


def hz_apply_relu(
    hz: HZono, tight_bounds=False, valid_cuts=False,
    cell_budget: int | None = None,
    compressed: bool = False,
) -> HZono | None:
    """ReLU encoded EXACTLY for every unstable neuron (no convex relaxation).

    Each unstable neuron i (alpha<0<beta) gets the EXACT eq_lagr graph (binary):
    ng+=4, nb+=1, nc+=3 -- the HZ thesis form that makes the domain non-convex/
    exact. There is intentionally NO triangle/DeepZ convex fallback: the HZ domain
    represents ReLU exactly via its binary generators, and a convex relaxation
    would only inject (sound but loose) over-approximation. If the resulting exact
    MILP is intractable, the verdict is honestly UNKNOWN -- we never silently fall
    back to a looser convex encoding.

    ``tight_bounds`` (precision mode): compute the per-neuron [alpha,beta] from
    the CONSTRAINED LP-relaxation bound (scipy, convex hull, sound) instead of
    the fast |Gc|+|Gb| box. The LP-tight box encloses the true pre-activation
    range (sound) but is much tighter, so the eq_lagr encoding it feeds is
    tighter too. Costs an LP per dimension -- use on the verdict/precision path,
    not the fast path.

    ``valid_cuts`` optionally appends two redundant ReLU graph facets
    (``x - y <= 0`` and ``y - s*x <= -s*alpha``) as inequality rows on top
    of the exact binary encoding. These cuts are solver-only tightening; they
    do not replace eq_lagr and do not propagate as a triangle relaxation.

    ``compressed`` exactly projects the ``xi3/xi4`` slack equalities into two
    inequality rows:
    ``-xi1 - z <= 0`` and ``-xi2 + z <= 0``. The linking equality and binary
    phase variable remain unchanged, so the represented ReLU graph is identical.
    It is opt-in because equivalent formulations can behave differently in
    HiGHS on some benchmarks.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = _relu_preact_bounds(hz, tight_bounds)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    unstable_idx = torch.where(unstable)[0]
    k = len(unstable_idx)

    if k == 0:
        out_Gc = hz.c.new_zeros(n, ng)
        out_Gb = hz.c.new_zeros(n, nb)
        out_c = hz.c.new_zeros(n, 1)
        if active.any():
            out_c[active] = hz.c[active]
            out_Gc[active, :ng] = hz.Gc[active]
            out_Gb[active, :nb] = hz.Gb[active]
        return hz_inherit_known_nonempty(HZono(
            c=out_c, Gc=out_Gc, Gb=out_Gb,
            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
            col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
            bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
        ), hz, reason="relu")

    exact_sel = unstable_idx
    ke = int(exact_sel.shape[0])

    relu_cont = 2 * ke if compressed else 4 * ke
    ng_new = ng + relu_cont
    nb_new = nb + ke
    n_proj = 2 * ke if (compressed and ke > 0) else 0
    n_cut = 2 * ke if (valid_cuts and ke > 0) else 0
    if cell_budget is not None:
        rows = nc + (ke if compressed else 3 * ke) + n_proj + n_cut
        cells = n * (ng_new + nb_new) + rows * (ng_new + nb_new)
        if cells > int(cell_budget):
            return None
    out_Gc = hz.c.new_zeros(n, ng_new)
    out_Gb = hz.c.new_zeros(n, nb_new)
    out_c = hz.c.new_zeros(n, 1)
    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    n_relu_eq = ke if compressed else 3 * ke
    Ac_out = hz.c.new_zeros(nc + n_relu_eq + n_proj + n_cut, ng_new)
    Ab_out = hz.c.new_zeros(nc + n_relu_eq + n_proj + n_cut, nb_new)
    b_out = hz.c.new_zeros(nc + n_relu_eq + n_proj + n_cut, 1)
    if nc > 0:
        Ac_out[:nc, :ng] = hz.Ac
        Ab_out[:nc, :nb] = hz.Ab
        b_out[:nc] = hz.b

    if ke > 0:
        a_e = lb[exact_sel]
        b_e = ub[exact_sel]
        te = torch.arange(ke, device=device)
        col_xi1 = ng + te
        col_xi2 = ng + ke + te
        col_z = nb + te
        out_c[exact_sel, 0] = b_e / 2.0
        out_Gc[exact_sel, col_xi2] = -b_e / 2.0
        if compressed:
            r3 = nc + te
        else:
            col_xi3 = ng + 2 * ke + te
            col_xi4 = ng + 3 * ke + te
            r1 = nc + 3 * te
            r2 = r1 + 1
            r3 = r1 + 2
            Ac_out[r1, col_xi1] = 1.0
            Ac_out[r1, col_xi3] = 1.0
            Ab_out[r1, col_z] = 1.0
            b_out[r1, 0] = 1.0
            Ac_out[r2, col_xi2] = 1.0
            Ac_out[r2, col_xi4] = 1.0
            Ab_out[r2, col_z] = -1.0
            b_out[r2, 0] = 1.0
        Ac_out[r3, col_xi1] = a_e / 2.0
        Ac_out[r3, col_xi2] = -b_e / 2.0
        Ab_out[r3, col_z] = a_e / 2.0
        b_out[r3, 0] = hz.c[exact_sel, 0] - b_e / 2.0

        chunk = 512
        for s in range(0, ke, chunk):
            e = min(s + chunk, ke)
            idx = exact_sel[s:e]
            Ac_out[r3[s:e], :ng] = -hz.Gc[idx]
            if nb > 0:
                Ab_out[r3[s:e], :nb] = -hz.Gb[idx]

        if compressed:
            pr1 = nc + n_relu_eq + te
            pr2 = pr1 + ke
            Ac_out[pr1, col_xi1] = -1.0
            Ab_out[pr1, col_z] = -1.0
            Ac_out[pr2, col_xi2] = -1.0
            Ab_out[pr2, col_z] = 1.0

    if valid_cuts and ke > 0:
        cr1 = nc + n_relu_eq + n_proj + te
        cr2 = cr1 + ke
        slope = b_e / (b_e - a_e).clamp_min(1e-12)

        Ac_out[cr1, col_xi2] = b_e / 2.0
        b_out[cr1, 0] = b_e / 2.0 - hz.c[exact_sel, 0]

        Ac_out[cr2, col_xi2] = -b_e / 2.0
        b_out[cr2, 0] = -slope * a_e - b_e / 2.0 + slope * hz.c[exact_sel, 0]
        chunk = 512
        for s in range(0, ke, chunk):
            e = min(s + chunk, ke)
            idx = exact_sel[s:e]
            Ac_out[cr1[s:e], :ng] = hz.Gc[idx]
            Ac_out[cr2[s:e], :ng] = -slope[s:e].unsqueeze(1) * hz.Gc[idx]
            if nb > 0:
                Ab_out[cr1[s:e], :nb] = hz.Gb[idx]
                Ab_out[cr2[s:e], :nb] = -slope[s:e].unsqueeze(1) * hz.Gb[idx]
    if hz.eq_mask is None and not compressed and not (valid_cuts and ke > 0):
        eq_mask_out = None
    else:
        old_mask = (hz.eq_mask.to(device) if hz.eq_mask is not None
                    else torch.ones(nc, dtype=torch.bool, device=device))
        proj_mask = (torch.zeros(n_proj, dtype=torch.bool, device=device)
                     if n_proj > 0
                     else torch.zeros(0, dtype=torch.bool, device=device))
        cut_mask = (torch.zeros(2 * ke, dtype=torch.bool, device=device)
                    if valid_cuts and ke > 0
                    else torch.zeros(0, dtype=torch.bool, device=device))
        eq_mask_out = torch.cat(
            [old_mask, torch.ones(n_relu_eq, dtype=torch.bool, device=device),
             proj_mask, cut_mask])

    new_col_ids, new_bcol_ids = _relu_extend_ids(hz, ke, compressed=compressed)
    return hz_inherit_known_nonempty(HZono(
        c=out_c, Gc=out_Gc, Gb=out_Gb,
        Ac=Ac_out, Ab=Ab_out, b=b_out,
        eq_mask=eq_mask_out,
        col_ids=new_col_ids, bcol_ids=new_bcol_ids,
    ), hz, reason="relu")


def _relu_extend_ids(hz: HZono, k: int, *, compressed: bool = False):
    """Extend factor ids for the ReLU encoding.

    The original ng/nb columns keep their ids. The new continuous factors
    (xi1/xi2, plus xi3/xi4 in the uncompressed form) and k new binary phase
    factors get FRESH ids because they are brand-new latent factors.
    """
    if hz.col_ids is None:
        return None, None
    dev = hz.col_ids.device
    new_col = torch.cat([
        hz.col_ids,
        hz_fresh_col_ids((2 if compressed else 4) * k, device=dev),
    ])
    base_b = (hz.bcol_ids if hz.bcol_ids is not None
              else torch.zeros(0, dtype=torch.long, device=dev))
    new_bcol = torch.cat([base_b, hz_fresh_col_ids(k, device=dev)])
    return new_col, new_bcol


def hz_apply_leaky_relu(hz: HZono, alpha_arg: float) -> HZono:
    """Exact LeakyReLU via the same encoding as ReLU.

    Per unstable neuron: ng += 4 (xi1, xi2, xi3, xi4), nb += 1 (z), nc += 3
    (graph eq 1, graph eq 2, linking eq) -- identical to hz_apply_relu.

    Decomposition: y = max(s*x, x) where s = alpha_arg. On the unstable
    branch, using the same switching mechanism as ReLU (z=+1 -> inactive
    with xi2 forced to 1; z=-1 -> active with xi1 forced to 1), we set
    the output as::

        y_h = beta/2 + (s*alpha/2) xi1 - (beta/2) xi2 + (s*alpha/2) z

    which degenerates exactly to ReLU's ``y_h = (beta/2)(1 - xi2)`` when
    s = 0. The graph equalities (xi1+xi3+z=1, xi2+xi4-z=1) and the linking
    equality (that ties x_h to xi1, xi2, z) are identical to ReLU.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]
    s = alpha_arg
    assert 0.0 <= s <= 1.0, f"hz_apply_leaky_relu: slope must be in [0, 1], got {s}"

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    unstable_idx = torch.where(unstable)[0]
    k = len(unstable_idx)

    out_Gc = hz.c.new_zeros(n, ng + 4 * k)
    out_Gb = hz.c.new_zeros(n, nb + k)
    out_c = hz.c.new_zeros(n, 1)

    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    if inactive.any():
        out_c[inactive] = s * hz.c[inactive]
        out_Gc[inactive, :ng] = s * hz.Gc[inactive]
        out_Gb[inactive, :nb] = s * hz.Gb[inactive]

    if k == 0:
        return hz_inherit_known_nonempty(HZono(
            c=out_c,
            Gc=out_Gc[:, :ng],
            Gb=out_Gb[:, :nb],
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
            col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
            bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
        ), hz, reason="leaky_relu")

    alpha = lb[unstable_idx]
    beta = ub[unstable_idx]
    t = torch.arange(k, device=device)

    col_xi1 = ng + t
    col_xi2 = ng + k + t
    col_xi3 = ng + 2 * k + t
    col_xi4 = ng + 3 * k + t
    col_z = nb + t

    # Output encoding: y_h = beta/2 + (s*alpha/2) xi1 - (beta/2) xi2 + (s*alpha/2) z
    out_c[unstable_idx, 0] = beta / 2.0
    out_Gc[unstable_idx, col_xi1] = s * alpha / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0
    out_Gb[unstable_idx, col_z] = s * alpha / 2.0

    ng_new = ng + 4 * k
    nb_new = nb + k

    eq_Ac = hz.c.new_zeros(3 * k, ng_new)
    eq_Ab = hz.c.new_zeros(3 * k, nb_new)
    eq_b = hz.c.new_zeros(3 * k, 1)

    r1 = 3 * t
    r2 = 3 * t + 1

    # Graph equality 1: xi1 + xi3 + z = 1
    eq_Ac[r1, col_xi1] = 1.0
    eq_Ac[r1, col_xi3] = 1.0
    eq_Ab[r1, col_z] = 1.0
    eq_b[r1, 0] = 1.0

    # Graph equality 2: xi2 + xi4 - z = 1
    eq_Ac[r2, col_xi2] = 1.0
    eq_Ac[r2, col_xi4] = 1.0
    eq_Ab[r2, col_z] = -1.0
    eq_b[r2, 0] = 1.0

    # Linking equality: ties x_h to (xi1, xi2, z)
    # Same form as ReLU; x_h has the same input expression.
    r3 = 3 * t + 2
    eq_Ac[r3, col_xi1] = alpha / 2.0
    eq_Ac[r3, col_xi2] = -beta / 2.0
    eq_Ac[r3, :ng] = -hz.Gc[unstable_idx]
    eq_Ab[r3, :nb] = -hz.Gb[unstable_idx]
    eq_Ab[r3, col_z] = alpha / 2.0
    eq_b[r3, 0] = hz.c[unstable_idx, 0] - beta / 2.0

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, 4 * k)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, k)], dim=1
    )

    new_col_ids, new_bcol_ids = _relu_extend_ids(hz, k)
    return hz_inherit_known_nonempty(HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
        eq_mask=(None if hz.eq_mask is None else torch.cat(
            [hz.eq_mask.to(device), torch.ones(3 * k, dtype=torch.bool, device=device)])),
        col_ids=new_col_ids,
        bcol_ids=new_bcol_ids,
    ), hz, reason="leaky_relu")


def _hz_apply_piecewise_compressed_pruned(
    hz: HZono,
    func,
    dfunc,
    K: int,
    inflection: float,
    cell_budget: int | None,
) -> HZono | None:
    """Compressed S-shape encoding with exact zero-width segment deletion.

    When the clamped inflection point equals a neuron bound, one side contributes
    K zero-width segments. Those segments only duplicate the shared boundary
    already represented by the nonzero side, so deleting them preserves the
    segment union exactly while removing binary/continuous columns.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    wide = (ub - lb) > 1e-12
    narrow = ~wide
    wide_idx = torch.where(wide)[0]
    m = int(wide_idx.numel())

    new_c = hz.c.clone()
    new_c[narrow] = func(hz.c[narrow])
    new_Gc_base = hz.Gc.clone()
    new_Gc_base[narrow] = 0.0
    new_Gb_base = hz.Gb.clone()
    new_Gb_base[narrow] = 0.0

    if m == 0:
        return hz_inherit_known_nonempty(HZono(
            c=new_c,
            Gc=new_Gc_base,
            Gb=new_Gb_base,
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
            col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
            bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
        ), hz, reason="piecewise")

    lb_w, ub_w = lb[wide_idx], ub[wide_idx]
    p = torch.maximum(
        torch.minimum(torch.full_like(lb_w, float(inflection)), ub_w), lb_w)
    sid = torch.arange(K, dtype=dtype, device=device).unsqueeze(1)
    wL = (p - lb_w).unsqueeze(0) / K
    wR = (ub_w - p).unsqueeze(0) / K
    aL = lb_w.unsqueeze(0) + sid * wL
    aR = p.unsqueeze(0) + sid * wR
    a_grid = torch.cat([aL, aR], dim=0)
    b_grid = torch.cat([aL + wL, aR + wR], dim=0)
    owner_grid = torch.arange(m, device=device).unsqueeze(0).expand(2 * K, -1)
    nondeg = (b_grid - a_grid) > 1e-12
    a = a_grid[nondeg]
    b_seg = b_grid[nondeg]
    owner = owner_grid[nondeg].to(dtype=torch.long)
    r = int(a.numel())
    if r == 0:
        return None

    fa, fb = func(a), func(b_seg)
    la, lb_slope = dfunc(a), dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = (la - lb_slope).abs() < 1e-10

    denom = lb_slope - la
    safe_denom = torch.where(nearly_linear, torch.ones_like(denom), denom)
    p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = lb_slope * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = la * (p2 - a) / 2.0

    hw = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = torch.linspace(0.0, 1.0, 50, dtype=dtype, device=device).view(50, 1)
    pts = a.unsqueeze(0) + t_pts * (b_seg - a).unsqueeze(0)
    f_pts = func(pts)
    resid = f_pts - (slope.unsqueeze(0) * pts + (fa - slope * a).unsqueeze(0))
    max_err = resid.abs().max(dim=0).values
    g1x_lin, g1y_lin = hw, slope * hw
    g2x_lin, g2y_lin = torch.zeros_like(hw), max_err

    g1_x = torch.where(nearly_linear, g1x_lin, g1x_tang)
    g1_y = torch.where(nearly_linear, g1y_lin, g1y_tang)
    g2_x = torch.where(nearly_linear, g2x_lin, g2x_tang)
    g2_y = torch.where(nearly_linear, g2y_lin, g2y_tang)

    dx = pts - centers_x.unsqueeze(0)
    dy = f_pts - centers_y.unsqueeze(0)
    det = g1_y * g2_x - g1_x * g2_y
    safe_det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
    xi1 = (dy * g2_x.unsqueeze(0) - dx * g2_y.unsqueeze(0)) / safe_det.unsqueeze(0)
    xi2 = (dy * g1_x.unsqueeze(0) - dx * g1_y.unsqueeze(0)) / (-safe_det.unsqueeze(0))
    max_xi = torch.maximum(xi1.abs().amax(dim=0), xi2.abs().amax(dim=0))
    scale_factor = torch.where(max_xi > 1.0, max_xi * 1.01, torch.ones_like(max_xi))
    scale_factor = torch.where(det.abs() < 1e-30, torch.ones_like(scale_factor), scale_factor)
    g1_x = g1_x * scale_factor
    g1_y = g1_y * scale_factor
    g2_x = g2_x * scale_factor
    g2_y = g2_y * scale_factor

    center_y_sum = torch.bincount(owner, weights=centers_y, minlength=m).to(dtype)
    center_x_sum = torch.bincount(owner, weights=centers_x, minlength=m).to(dtype)
    seg_count = torch.bincount(owner, minlength=m).to(dtype)
    new_c[wide_idx] = (center_y_sum / 2.0).unsqueeze(1)
    new_Gc_base[wide_idx] = 0.0
    new_Gb_base[wide_idx] = 0.0

    n_real = 2 * r
    ng_total = ng + n_real
    nb_total = nb + r
    n_eq_total = 2 * m
    n_le_total = 4 * r
    if cell_budget is not None:
        rows_total = nc + n_eq_total + n_le_total
        cells = n * (ng_total + nb_total) + rows_total * (ng_total + nb_total)
        if cells > int(cell_budget):
            return None

    g1_cols = ng + torch.arange(r, device=device)
    g2_cols = ng + r + torch.arange(r, device=device)
    z_cols = nb + torch.arange(r, device=device)
    wide_rows = wide_idx[owner]

    Gc_new = hz.c.new_zeros(n, n_real)
    Gc_new[wide_rows, g1_cols - ng] = g1_y
    Gc_new[wide_rows, g2_cols - ng] = g2_y
    Gb_new = hz.c.new_zeros(n, r)
    Gb_new[wide_rows, z_cols - nb] = -centers_y / 2.0
    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)

    eq_Ac = hz.c.new_zeros(n_eq_total, ng_total)
    eq_Ab = hz.c.new_zeros(n_eq_total, nb_total)
    eq_b = hz.c.new_zeros(n_eq_total, 1)
    link_rows = torch.arange(m, device=device)
    sum_rows = m + link_rows
    eq_Ac[link_rows[owner], g1_cols] = -g1_x
    eq_Ac[link_rows[owner], g2_cols] = -g2_x
    eq_Ab[link_rows[owner], z_cols] = centers_x / 2.0
    eq_Ac[link_rows, :ng] = hz.Gc[wide_idx]
    eq_Ab[link_rows, :nb] = hz.Gb[wide_idx]
    eq_b[link_rows, 0] = center_x_sum / 2.0 - hz.c[wide_idx, 0]
    eq_Ab[sum_rows[owner], z_cols] = 1.0
    eq_b[sum_rows, 0] = seg_count - 2.0

    ineq_Ac = hz.c.new_zeros(n_le_total, ng_total)
    ineq_Ab = hz.c.new_zeros(n_le_total, nb_total)
    ineq_b = hz.c.new_full((n_le_total, 1), 0.5)
    box_rows = 4 * torch.arange(r, device=device)
    ineq_Ac[box_rows, g1_cols] = 1.0
    ineq_Ac[box_rows + 1, g1_cols] = -1.0
    ineq_Ac[box_rows + 2, g2_cols] = 1.0
    ineq_Ac[box_rows + 3, g2_cols] = -1.0
    ineq_Ab[box_rows, z_cols] = 0.5
    ineq_Ab[box_rows + 1, z_cols] = 0.5
    ineq_Ab[box_rows + 2, z_cols] = 0.5
    ineq_Ab[box_rows + 3, z_cols] = 0.5

    old_Ac_ext = torch.cat([hz.Ac, hz.c.new_zeros(nc, n_real)], dim=1)
    old_Ab_ext = torch.cat([hz.Ab, hz.c.new_zeros(nc, r)], dim=1)

    new_col_ids = None
    new_bcol_ids = None
    if hz.col_ids is not None:
        id_dev = hz.col_ids.device
        new_col_ids = torch.cat([hz.col_ids, hz_fresh_col_ids(n_real, device=id_dev)])
        base_b = (hz.bcol_ids if hz.bcol_ids is not None
                  else torch.zeros(0, dtype=torch.long, device=id_dev))
        new_bcol_ids = torch.cat([base_b.to(id_dev), hz_fresh_col_ids(r, device=id_dev)])

    out = HZono(
        c=new_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac, ineq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab, ineq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b, ineq_b], dim=0),
        eq_mask=torch.cat([
            (hz.eq_mask.to(device) if hz.eq_mask is not None
             else torch.ones(nc, dtype=torch.bool, device=device)),
            torch.ones(n_eq_total, dtype=torch.bool, device=device),
            torch.zeros(n_le_total, dtype=torch.bool, device=device),
        ]),
        col_ids=new_col_ids,
        bcol_ids=new_bcol_ids,
    )
    if hasattr(hz, "full_col_ids"):
        out.full_col_ids = hz.full_col_ids
    return hz_inherit_known_nonempty(out, hz, reason="piecewise")


def hz_apply_piecewise(
    hz: HZono,
    func,
    dfunc,
    K: int = 2,
    inflection=None,
    compressed: bool = True,
    cell_budget: int | None = None,
) -> HZono | None:
    """Piecewise linear approximation for monotone activations (tangent parallelogram).

    ``inflection`` (e.g. 0.0 for sigmoid/tanh): force the inflection point to be a
    SEGMENT BOUNDARY. The tangent parallelogram assumes the function is convex OR
    concave on each segment; a segment straddling the inflection violates that, so
    the soundness slack inflates and tightness is not monotone in K. Splitting K
    segments on each side of the inflection (=> 2K segments, inflection always a
    boundary) keeps every segment convex/concave so tightness is monotone in K.

    ``compressed`` projects the per-segment slack-box equalities exactly into
    two inequality rows per local generator. The represented HZ approximation is
    unchanged, but the dense state drops four continuous slack columns per
    segment, which is the dominant memory cost in wide sigmoid/tanh networks.
    """
    if compressed and inflection is not None:
        return _hz_apply_piecewise_compressed_pruned(
            hz, func, dfunc, K, float(inflection), cell_budget)

    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    wide = (ub - lb) > 1e-12
    narrow = ~wide
    wide_idx = torch.where(wide)[0]
    m = int(wide_idx.sum() if wide_idx.ndim == 0 else wide_idx.shape[0])

    new_c = hz.c.clone()
    new_c[narrow] = func(hz.c[narrow])
    new_Gc_base = hz.Gc.clone()
    new_Gc_base[narrow] = 0.0
    new_Gb_base = hz.Gb.clone()
    new_Gb_base[narrow] = 0.0

    if m == 0:
        return hz_inherit_known_nonempty(HZono(
            c=new_c,
            Gc=new_Gc_base,
            Gb=new_Gb_base,
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
            col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
            bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
        ), hz, reason="piecewise")

    lb_w, ub_w = lb[wide_idx], ub[wide_idx]
    if inflection is not None:
        p = torch.maximum(
            torch.minimum(torch.full_like(lb_w, float(inflection)), ub_w), lb_w)
        sid = torch.arange(K, dtype=dtype, device=device).unsqueeze(1)
        wL = (p - lb_w).unsqueeze(0) / K
        wR = (ub_w - p).unsqueeze(0) / K
        aL = lb_w.unsqueeze(0) + sid * wL
        aR = p.unsqueeze(0) + sid * wR
        a = torch.cat([aL, aR], dim=0)
        b_seg = torch.cat([aL + wL, aR + wR], dim=0)
        K = 2 * K
    else:
        segment_ids = torch.arange(K, dtype=dtype, device=device).unsqueeze(1)
        segment_width = (ub_w - lb_w).unsqueeze(0) / K
        a = lb_w.unsqueeze(0) + segment_ids * segment_width
        b_seg = a + segment_width
    fa, fb = func(a), func(b_seg)
    la, lb_slope = dfunc(a), dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = (la - lb_slope).abs() < 1e-10

    denom = lb_slope - la
    safe_denom = torch.where(nearly_linear, torch.ones_like(denom), denom)
    p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = lb_slope * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = la * (p2 - a) / 2.0

    hw = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = torch.linspace(0.0, 1.0, 50, dtype=dtype, device=device).view(50, 1, 1)
    pts = a.unsqueeze(0) + t_pts * (b_seg - a).unsqueeze(0)
    f_pts = func(pts)
    resid = f_pts - (
        slope.unsqueeze(0) * pts + (fa - slope * a).unsqueeze(0)
    )
    max_err = resid.abs().max(dim=0).values
    g1x_lin, g1y_lin = hw, slope * hw
    g2x_lin, g2y_lin = torch.zeros_like(hw), max_err

    g1_x = torch.where(nearly_linear, g1x_lin, g1x_tang)
    g1_y = torch.where(nearly_linear, g1y_lin, g1y_tang)
    g2_x = torch.where(nearly_linear, g2x_lin, g2x_tang)
    g2_y = torch.where(nearly_linear, g2y_lin, g2y_tang)

    dx = pts - centers_x.unsqueeze(0)
    dy = f_pts - centers_y.unsqueeze(0)
    det = g1_y * g2_x - g1_x * g2_y
    safe_det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
    xi1 = (dy * g2_x.unsqueeze(0) - dx * g2_y.unsqueeze(0)) / safe_det.unsqueeze(0)
    xi2 = (dy * g1_x.unsqueeze(0) - dx * g1_y.unsqueeze(0)) / (-safe_det.unsqueeze(0))
    max_xi = torch.maximum(xi1.abs().amax(dim=0), xi2.abs().amax(dim=0))
    scale_factor = torch.where(max_xi > 1.0, max_xi * 1.01, torch.ones_like(max_xi))
    scale_factor = torch.where(det.abs() < 1e-30, torch.ones_like(scale_factor), scale_factor)
    g1_x = g1_x * scale_factor
    g1_y = g1_y * scale_factor
    g2_x = g2_x * scale_factor
    g2_y = g2_y * scale_factor

    cy_sum = centers_y.sum(dim=0)
    new_c[wide_idx] = (cy_sum / 2.0).unsqueeze(1)
    new_Gc_base[wide_idx] = 0.0
    new_Gb_base[wide_idx] = 0.0

    n_real = 2 * K * m
    n_slack = 0 if compressed else 4 * K * m
    ng_total = ng + n_real + n_slack
    nb_total = nb + K * m
    n_box = 0 if compressed else 4 * K * m
    n_eq_total = n_box + m + m
    n_le_total = 4 * K * m if compressed else 0
    if cell_budget is not None:
        rows_total = nc + n_eq_total + n_le_total
        cells = (
            n * (ng_total + nb_total)
            + rows_total * (ng_total + nb_total)
        )
        if cells > int(cell_budget):
            return None

    Gc_new = hz.c.new_zeros(n, n_real + n_slack)
    g1_cols = torch.arange(K * m, device=device).reshape(K, m)
    g2_cols = (K * m + torch.arange(K * m, device=device)).reshape(K, m)
    wide_rows = wide_idx.unsqueeze(0).expand(K, -1)
    Gc_new[wide_rows, g1_cols] = g1_y
    Gc_new[wide_rows, g2_cols] = g2_y

    Gb_new = hz.c.new_zeros(n, K * m)
    z_cols = torch.arange(K * m, device=device).reshape(K, m)
    Gb_new[wide_rows, z_cols] = -centers_y / 2.0

    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)
    eq_Ac = hz.c.new_zeros(n_eq_total, ng_total)
    eq_Ab = hz.c.new_zeros(n_eq_total, nb_total)
    eq_b = hz.c.new_zeros(n_eq_total, 1)

    segment_grid = torch.arange(K * m, device=device).reshape(K, m)
    g1_col_grid = ng + segment_grid
    g2_col_grid = ng + K * m + segment_grid
    z_col_grid = nb + segment_grid
    slack_base_grid = ng + n_real + 4 * segment_grid
    row_grid = 4 * segment_grid

    flat_rows = row_grid.reshape(-1)
    flat_g1_cols = g1_col_grid.reshape(-1)
    flat_g2_cols = g2_col_grid.reshape(-1)
    flat_z_cols = z_col_grid.reshape(-1)
    flat_slack_bases = slack_base_grid.reshape(-1)

    if not compressed:
        eq_Ac[flat_rows, flat_g1_cols] = 1.0
        eq_Ac[flat_rows, flat_slack_bases] = 1.0
        eq_Ab[flat_rows, flat_z_cols] = -0.5
        eq_b[flat_rows, 0] = 0.5

        eq_Ac[flat_rows + 1, flat_g1_cols] = -1.0
        eq_Ac[flat_rows + 1, flat_slack_bases + 1] = 1.0
        eq_Ab[flat_rows + 1, flat_z_cols] = -0.5
        eq_b[flat_rows + 1, 0] = 0.5

        eq_Ac[flat_rows + 2, flat_g2_cols] = 1.0
        eq_Ac[flat_rows + 2, flat_slack_bases + 2] = 1.0
        eq_Ab[flat_rows + 2, flat_z_cols] = -0.5
        eq_b[flat_rows + 2, 0] = 0.5

        eq_Ac[flat_rows + 3, flat_g2_cols] = -1.0
        eq_Ac[flat_rows + 3, flat_slack_bases + 3] = 1.0
        eq_Ab[flat_rows + 3, flat_z_cols] = -0.5
        eq_b[flat_rows + 3, 0] = 0.5

    link_rows = n_box + torch.arange(m, device=device)
    link_row_grid = link_rows.unsqueeze(1).expand(-1, K)
    eq_Ac[link_row_grid, g1_col_grid.transpose(0, 1)] = -g1_x.transpose(0, 1)
    eq_Ac[link_row_grid, g2_col_grid.transpose(0, 1)] = -g2_x.transpose(0, 1)
    eq_Ab[link_row_grid, z_col_grid.transpose(0, 1)] = centers_x.transpose(0, 1) / 2.0
    eq_Ac[link_rows, :ng] = hz.Gc[wide_idx]
    eq_Ab[link_rows, :nb] = hz.Gb[wide_idx]
    eq_b[link_rows, 0] = centers_x.sum(dim=0) / 2.0 - hz.c[wide_idx, 0]

    sum_rows = n_box + m + torch.arange(m, device=device)
    sum_row_grid = sum_rows.unsqueeze(1).expand(-1, K)
    eq_Ab[sum_row_grid, z_col_grid.transpose(0, 1)] = 1.0
    eq_b[sum_rows, 0] = hz.c.new_full((m,), float(K - 2))

    if compressed:
        ineq_Ac = hz.c.new_zeros(n_le_total, ng_total)
        ineq_Ab = hz.c.new_zeros(n_le_total, nb_total)
        ineq_b = hz.c.new_full((n_le_total, 1), 0.5)
        if n_le_total > 0:
            box_rows = 4 * segment_grid.reshape(-1)
            ineq_Ac[box_rows, flat_g1_cols] = 1.0
            ineq_Ac[box_rows + 1, flat_g1_cols] = -1.0
            ineq_Ac[box_rows + 2, flat_g2_cols] = 1.0
            ineq_Ac[box_rows + 3, flat_g2_cols] = -1.0
            ineq_Ab[box_rows, flat_z_cols] = 0.5
            ineq_Ab[box_rows + 1, flat_z_cols] = 0.5
            ineq_Ab[box_rows + 2, flat_z_cols] = 0.5
            ineq_Ab[box_rows + 3, flat_z_cols] = 0.5
    else:
        ineq_Ac = hz.c.new_zeros(0, ng_total)
        ineq_Ab = hz.c.new_zeros(0, nb_total)
        ineq_b = hz.c.new_zeros(0, 1)

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, n_real + n_slack)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, K * m)], dim=1
    )

    new_col_ids = None
    new_bcol_ids = None
    if hz.col_ids is not None:
        id_dev = hz.col_ids.device
        new_col_ids = torch.cat([
            hz.col_ids,
            hz_fresh_col_ids(n_real + n_slack, device=id_dev),
        ])
        base_b = (hz.bcol_ids if hz.bcol_ids is not None
                  else torch.zeros(0, dtype=torch.long, device=id_dev))
        new_bcol_ids = torch.cat([
            base_b.to(id_dev),
            hz_fresh_col_ids(K * m, device=id_dev),
        ])

    out = HZono(
        c=new_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac, ineq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab, ineq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b, ineq_b], dim=0),
        eq_mask=(
            torch.cat([
                (hz.eq_mask.to(device) if hz.eq_mask is not None
                 else torch.ones(nc, dtype=torch.bool, device=device)),
                torch.ones(n_eq_total, dtype=torch.bool, device=device),
                torch.zeros(n_le_total, dtype=torch.bool, device=device),
            ])
            if (compressed or hz.eq_mask is not None)
            else None
        ),
        col_ids=new_col_ids,
        bcol_ids=new_bcol_ids,
    )
    if hasattr(hz, "full_col_ids"):
        out.full_col_ids = hz.full_col_ids
    return hz_inherit_known_nonempty(out, hz, reason="piecewise")


def hz_apply_sigmoid(
    hz: HZono, K: int = 2, *, cell_budget: int | None = None
) -> HZono | None:
    """Piecewise linear sigmoid via tangent parallelogram encoding. Sigmoid has
    its inflection at 0 -> split segments there for monotone-in-K tightness."""
    return hz_apply_piecewise(
        hz, torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)), K,
        inflection=0.0,
        cell_budget=cell_budget,
    )


def hz_apply_tanh(
    hz: HZono, K: int = 2, *, cell_budget: int | None = None
) -> HZono | None:
    """Piecewise linear tanh via tangent parallelogram encoding. Tanh inflection
    at 0 -> split segments there for monotone-in-K tightness."""
    return hz_apply_piecewise(hz, torch.tanh, lambda x: 1 - torch.tanh(x) ** 2, K,
                              inflection=0.0, cell_budget=cell_budget)


# --- HZ order reduction ---


def hz_reduce(hz: HZono, max_order: float = 3.0) -> HZono:
    """Exact HZ redundancy removal for the strict product path.

    ``max_order`` is accepted for legacy call-site compatibility, but ignored.
    Strict HybridZ does not silently relax binaries or invoke Girard reduction:
    if an exact HZ grows too large, the frontend must return UNKNOWN rather than
    continue with a lossy HZ representation.
    """
    n = hz.c.shape[0]
    if n == 0:
        return hz
    from act.back_end.solver.solver_hz import hz_remove_redundancy
    return hz_remove_redundancy(hz)





def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def sparse_empty(rows: int, cols: int) -> sp.csr_matrix:
    return sp.csr_matrix((int(rows), int(cols)), dtype=np.float64)


def sparse_pad_cols(mat: sp.csr_matrix, cols: int) -> sp.csr_matrix:
    mat = mat.tocsr()
    cols = int(cols)
    if mat.shape[1] == cols:
        return mat
    if mat.shape[1] > cols:
        raise ValueError(f"cannot shrink sparse matrix from {mat.shape[1]} to {cols}")
    return sp.hstack([mat, sparse_empty(mat.shape[0], cols - mat.shape[1])], format="csr")


def sparse_hz_pad_frame(hz: SparseHZono, n_cont: int, n_bin: int) -> SparseHZono:
    Auc = None if hz.Auc is None else sparse_pad_cols(hz.Auc, n_cont)
    Aub = None if hz.Aub is None else sparse_pad_cols(hz.Aub, n_bin)
    return SparseHZono(
        c=hz.c,
        Gc=sparse_pad_cols(hz.Gc, n_cont),
        Gb=sparse_pad_cols(hz.Gb, n_bin),
        Ac=sparse_pad_cols(hz.Ac, n_cont),
        Ab=sparse_pad_cols(hz.Ab, n_bin),
        b=hz.b,
        Auc=Auc,
        Aub=Aub,
        ub=hz.ub,
    )


def sparse_hz_from_bounds(bounds: Bounds, *, drop_zero_radius: bool = True) -> SparseHZono:
    """Create a sparse HZ box from ACT ``Bounds``.

    By default, zero-radius dimensions do not allocate useless generator
    columns.  The represented set is identical to the dense ``hz_from_bounds``
    set, just with structurally-zero columns removed.
    """

    lb = bounds.lb.detach().cpu().double().numpy().reshape(-1)
    ub = bounds.ub.detach().cpu().double().numpy().reshape(-1)
    center = (lb + ub) * 0.5
    rad = (ub - lb) * 0.5
    if drop_zero_radius:
        rows = np.nonzero(np.abs(rad) > 1e-12)[0].astype(np.int32)
    else:
        rows = np.arange(rad.size, dtype=np.int32)
    cols = np.arange(rows.size, dtype=np.int32)
    Gc = sp.csr_matrix((rad[rows], (rows, cols)), shape=(rad.size, rows.size), dtype=np.float64)
    return SparseHZono(
        c=center,
        Gc=Gc,
        Gb=sparse_empty(rad.size, 0),
        Ac=sparse_empty(0, rows.size),
        Ab=sparse_empty(0, 0),
        b=np.zeros(0, dtype=np.float64),
        Auc=sparse_empty(0, rows.size),
        Aub=sparse_empty(0, 0),
        ub=np.zeros(0, dtype=np.float64),
    )


def sparse_hz_linear(hz: SparseHZono, W, bias: Optional[Sequence[float]] = None) -> SparseHZono:
    """Apply an exact affine map ``W @ z + bias``."""

    Wsp = W.tocsr().astype(np.float64) if sp.issparse(W) else sp.csr_matrix(np.asarray(W, dtype=np.float64))
    if Wsp.shape[1] != hz.n_out:
        raise ValueError(f"linear shape mismatch: W={Wsp.shape}, hz.n_out={hz.n_out}")
    b = (
        np.zeros(Wsp.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != Wsp.shape[0]:
        raise ValueError(f"bias shape mismatch: bias={b.size}, rows={Wsp.shape[0]}")
    Gc = (Wsp @ hz.Gc).tocsr()
    Gb = (Wsp @ hz.Gb).tocsr() if hz.n_bin else sparse_empty(Wsp.shape[0], 0)
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=np.asarray(Wsp @ hz.c).reshape(-1) + b,
        Gc=Gc,
        Gb=Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
    )


def sparse_dense_matrix_from_layer(layer) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return exact sparse matrix/bias for an ACT DENSE layer."""

    W = _to_numpy(layer.params["weight"]).astype(np.float64, copy=False)
    bias = layer.params.get("bias")
    b = (
        np.zeros(W.shape[0], dtype=np.float64)
        if bias is None
        else _to_numpy(bias).astype(np.float64, copy=False).reshape(-1)
    )
    mat = sp.csr_matrix(W)
    mat.eliminate_zeros()
    return mat, b


def sparse_hz_apply_dense_layer(hz: SparseHZono, layer) -> SparseHZono:
    W, b = sparse_dense_matrix_from_layer(layer)
    return sparse_hz_linear(hz, W, b)


def sparse_hz_is_point(hz: SparseHZono) -> bool:
    return hz.n_cont == 0 and hz.n_bin == 0 and hz.n_eq == 0 and hz.n_ub == 0


def sparse_matmul_const_matrix_from_layer(
    layer,
    const,
    *,
    variable_is_left: bool,
) -> sp.csr_matrix:
    """Exact linear matrix for MATMUL when one operand is a point tensor."""

    x_shape = tuple(int(d) for d in layer.params["x_shape"])
    y_shape = tuple(int(d) for d in layer.params["y_shape"])
    in_shape = x_shape if variable_is_left else y_shape
    in_dim = _prod(in_shape)
    if in_dim <= 0:
        raise ValueError("MATMUL variable operand has empty shape")
    const_shape = y_shape if variable_is_left else x_shape
    W = np.asarray(const, dtype=np.float64).reshape(const_shape)
    eye = np.eye(in_dim, dtype=np.float64).reshape((in_dim, *in_shape))
    out = np.matmul(eye, W) if variable_is_left else np.matmul(W, eye)
    mat = sp.csr_matrix(out.reshape(in_dim, -1).T)
    mat.eliminate_zeros()
    return mat


def sparse_hz_apply_per_batch_linear(hz: SparseHZono, W: sp.csr_matrix) -> SparseHZono:
    W = W.tocsr()
    in_dim = int(W.shape[1])
    if in_dim <= 0 or hz.n_out % in_dim:
        raise ValueError(f"per-batch linear mismatch: hz.n_out={hz.n_out}, W={W.shape}")
    B = hz.n_out // in_dim
    if B == 1:
        big = W
    else:
        big = sp.kron(sp.eye(B, format="csr", dtype=np.float64), W, format="csr")
    return sparse_hz_linear(hz, big, np.zeros(big.shape[0], dtype=np.float64))


def sparse_hz_apply_matmul_const_layer(
    variable_hz: SparseHZono,
    const_hz: SparseHZono,
    layer,
    *,
    variable_is_left: bool,
) -> SparseHZono:
    if not sparse_hz_is_point(const_hz):
        raise ValueError("constant-side MATMUL requires a point SparseHZono")
    W = sparse_matmul_const_matrix_from_layer(
        layer,
        const_hz.c,
        variable_is_left=variable_is_left,
    )
    return sparse_hz_apply_per_batch_linear(variable_hz, W)


def sparse_upsample_nearest_row_indices(layer, n_in: int, n_out: int) -> Optional[np.ndarray]:
    """Output-to-input row map for nearest-neighbor upsample."""

    mode = str(layer.params.get("mode", "nearest")).lower()
    if mode != "nearest":
        return None
    in_shape = layer.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    if _prod(in_shape) != int(n_in) or len(in_shape) < 3:
        return None
    view_shape = (1, *in_shape) if len(in_shape) == 3 else in_shape
    spatial_rank = len(view_shape) - 2
    size = layer.params.get("size")
    scale_factor = layer.params.get("scale_factor")
    if size is not None and isinstance(size, (list, tuple)):
        size = tuple(int(s) for s in size)
        if len(size) > spatial_rank:
            size = size[-spatial_rank:]
    if scale_factor is not None and isinstance(scale_factor, (list, tuple)):
        scale_factor = tuple(float(s) for s in scale_factor)
        if len(scale_factor) > spatial_rank:
            scale_factor = scale_factor[-spatial_rank:]
    if size is None and scale_factor is None:
        out_shape = layer.params.get("output_shape")
        if out_shape is None:
            return None
        out_shape = tuple(int(d) for d in out_shape)
        out_view_shape = (1, *out_shape) if len(out_shape) == 3 else out_shape
        if len(out_view_shape) != len(view_shape):
            return None
        size = out_view_shape[2:]
    base = torch.arange(int(n_in), dtype=torch.float64).view(*view_shape)
    out = torch.nn.functional.interpolate(
        base,
        size=size,
        scale_factor=scale_factor,
        mode="nearest",
    )
    idx = out.reshape(-1).detach().cpu().numpy().astype(np.int64)
    if idx.size != int(n_out):
        return None
    return idx


def sparse_slice_row_indices(layer, n: int) -> Optional[np.ndarray]:
    """Output-to-input row map for Slice, mirroring interval.tf_slice."""

    if "input_shape" not in layer.params:
        return None
    inp_shape = tuple(int(d) for d in layer.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    idx = torch.arange(int(n)).view(batch, *inp_shape)
    starts = layer.params.get("starts", [])
    ends = layer.params.get("ends", [])
    axes = layer.params.get("axes", list(range(len(inp_shape))))
    steps = layer.params.get("steps", [1] * len(axes))
    slices = [slice(None)] * (len(inp_shape) + 1)
    for i, axis in enumerate(axes):
        axis = int(axis)
        end = ends[i]
        if end > inp_shape[axis]:
            end = inp_shape[axis]
        slices[axis + 1] = slice(starts[i], end, steps[i])
    return idx[tuple(slices)].reshape(-1).detach().cpu().numpy().astype(np.int64)


def sparse_gather_row_indices(layer, n: int) -> Optional[np.ndarray]:
    """Output-to-input row map for Gather, mirroring interval.tf_gather."""

    if "input_shape" not in layer.params:
        return None
    inp_shape = tuple(int(d) for d in layer.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    axis = int(layer.params.get("axis", 0))
    if axis < 0:
        axis += len(inp_shape)
    raw_idx = layer.params["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long)
    elif hasattr(raw_idx, "detach"):
        indices = raw_idx.detach().cpu().long()
    else:
        indices = torch.as_tensor(raw_idx, dtype=torch.long)
    idx = torch.arange(int(n)).view(batch, *inp_shape)
    return (
        torch.index_select(idx, dim=axis + 1, index=indices.reshape(-1))
        .reshape(-1)
        .numpy()
        .astype(np.int64)
    )


def sparse_expand_row_indices(layer, n: int) -> Optional[np.ndarray]:
    """Output-to-input row map for Expand/broadcast, mirroring interval.tf_expand."""

    in_shape = layer.params.get("input_shape")
    out_shape = layer.params.get("output_shape") or layer.params.get("shape")
    if in_shape is None or out_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    out_shape = tuple(int(d) for d in out_shape)
    per = _prod(in_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    try:
        rows = torch.arange(int(n)).view(batch, *in_shape).broadcast_to(batch, *out_shape)
    except RuntimeError:
        return None
    return rows.reshape(-1).detach().cpu().numpy().astype(np.int64)


def sparse_reduce_sum_row_indices(layer, n_in: int, n_out: int) -> Optional[np.ndarray]:
    """Input-to-output row map for ReduceSum, mirroring interval.tf_reduce_sum."""

    in_shape = layer.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    per = _prod(in_shape)
    if per == 0 or int(n_in) % per != 0:
        return None
    batch = int(n_in) // per
    axes = layer.params.get("axes")
    axes = list(range(len(in_shape))) if not axes else [int(a) for a in axes]
    axes = [(a + len(in_shape)) if a < 0 else a for a in axes]
    keepdims = bool(layer.params.get("keepdims", 0))
    out_shape: List[int] = []
    for i, dim in enumerate(in_shape):
        if i in axes:
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(dim)
    if _prod(out_shape) * batch != int(n_out):
        return None
    out_idx = np.arange(int(n_out), dtype=np.int64).reshape((batch, *out_shape))
    view_shape = [batch]
    for i, dim in enumerate(in_shape):
        view_shape.append(1 if i in axes else dim)
    return np.broadcast_to(out_idx.reshape(tuple(view_shape)), (batch, *in_shape)).reshape(-1)


def sparse_hz_reduce_sum_rows(
    hz: SparseHZono,
    out_rows: Sequence[int],
    n_out: int,
) -> SparseHZono:
    """Exact ReduceSum row aggregation using a sparse linear map."""

    rows = np.asarray(out_rows, dtype=np.int64).reshape(-1)
    if rows.size != hz.n_out:
        raise ValueError(f"reduce-sum row map length {rows.size} != hz.n_out {hz.n_out}")
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= int(n_out)):
        raise ValueError("reduce-sum row map contains output index outside n_out")
    mat = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, np.arange(rows.size, dtype=np.int64))),
        shape=(int(n_out), hz.n_out),
        dtype=np.float64,
    )
    return sparse_hz_linear(hz, mat, np.zeros(int(n_out), dtype=np.float64))


def sparse_hz_gather_rows_like(
    base: SparseHZono,
    rows: Sequence[int],
    *,
    fill_value: float,
    template: Optional[SparseHZono] = None,
) -> SparseHZono:
    """Gather rows from ``base`` while using ``template``'s variable frame."""

    row_idx = np.asarray(rows, dtype=np.int64).reshape(-1)
    tmpl = template if template is not None else base
    n = int(row_idx.size)
    valid = row_idx >= 0
    c = np.full(n, float(fill_value), dtype=np.float64)
    if np.any(valid):
        pos = np.nonzero(valid)[0].astype(np.int32)
        src = row_idx[valid].astype(np.int64)
        c[pos] = base.c[src]
        sub_c = sparse_pad_cols(base.Gc[src].tocsr(), tmpl.n_cont).tocoo()
        Gc = sp.coo_matrix(
            (sub_c.data, (pos[sub_c.row], sub_c.col)),
            shape=(n, tmpl.n_cont),
            dtype=np.float64,
        ).tocsr()
        if tmpl.n_bin:
            sub_b = sparse_pad_cols(base.Gb[src].tocsr(), tmpl.n_bin).tocoo()
            Gb = sp.coo_matrix(
                (sub_b.data, (pos[sub_b.row], sub_b.col)),
                shape=(n, tmpl.n_bin),
                dtype=np.float64,
            ).tocsr()
        else:
            Gb = sparse_empty(n, 0)
    else:
        Gc = sparse_empty(n, tmpl.n_cont)
        Gb = sparse_empty(n, tmpl.n_bin)
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=c,
        Gc=Gc,
        Gb=Gb,
        Ac=tmpl.Ac,
        Ab=tmpl.Ab,
        b=tmpl.b,
        Auc=tmpl.Auc,
        Aub=tmpl.Aub,
        ub=tmpl.ub,
    )


def sparse_hz_fast_bounds(hz: SparseHZono) -> Bounds:
    """Sound unconstrained box bounds for a sparse HZ."""

    rad_c = np.asarray(np.abs(hz.Gc).sum(axis=1)).reshape(-1)
    rad_b = np.asarray(np.abs(hz.Gb).sum(axis=1)).reshape(-1) if hz.n_bin else 0.0
    rad = rad_c + rad_b
    lb = torch.from_numpy(hz.c - rad).reshape(1, -1).double()
    ub = torch.from_numpy(hz.c + rad).reshape(1, -1).double()
    return Bounds(lb=lb, ub=ub)


def _bounds_arrays(bounds: Bounds, n: int) -> Tuple[np.ndarray, np.ndarray]:
    lb = bounds.lb.detach().cpu().double().numpy().reshape(-1)
    ub = bounds.ub.detach().cpu().double().numpy().reshape(-1)
    if lb.size != n or ub.size != n:
        raise ValueError(f"bounds shape mismatch: bounds={lb.size}/{ub.size}, hz.n_out={n}")
    return lb.astype(np.float64, copy=False), ub.astype(np.float64, copy=False)


def _coo_matrix_from_parts(
    rows: List[np.ndarray],
    cols: List[np.ndarray],
    data: List[np.ndarray],
    shape: Tuple[int, int],
) -> sp.csr_matrix:
    if not rows:
        return sparse_empty(*shape)
    return sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=shape,
        dtype=np.float64,
    ).tocsr()


def _coo_appender(
    rows: List[np.ndarray],
    cols: List[np.ndarray],
    data: List[np.ndarray],
):
    def append(row_idx, col_idx, values) -> None:
        row_arr = np.asarray(row_idx, dtype=np.int32).reshape(-1)
        if row_arr.size:
            rows.append(row_arr)
            cols.append(np.asarray(col_idx, dtype=np.int32).reshape(-1))
            data.append(np.asarray(values, dtype=np.float64).reshape(-1))

    return append


def sparse_hz_apply_relu_exact(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    *,
    compressed: bool = False,
    valid_cuts: bool = False,
    return_info: bool = False,
):
    """Exact sparse eq_lagr ReLU.

    Every unstable neuron gets the exact binary HZ graph.  ``compressed`` keeps
    the same exact graph after projecting the two slack equalities into two
    upper-inequality rows.  ``valid_cuts`` appends redundant graph facets for
    solver tightening; it never replaces the exact binary construction.
    """

    if pre_bounds is None:
        pre_bounds = sparse_hz_fast_bounds(hz)
    lb, ub = _bounds_arrays(pre_bounds, hz.n_out)
    active = lb >= 0.0
    inactive = ub <= 0.0
    unstable = ~(active | inactive)
    active_idx = np.nonzero(active)[0].astype(np.int32)
    unstable_idx = np.nonzero(unstable)[0].astype(np.int32)
    k = int(unstable_idx.size)
    info = {
        "lb": lb.copy(),
        "ub": ub.copy(),
        "active": active.copy(),
        "inactive": inactive.copy(),
        "unstable_idx": unstable_idx.copy(),
        "compressed": bool(compressed),
    }

    n = hz.n_out
    ng = hz.n_cont
    nb = hz.n_bin
    ng_new = ng + (2 * k if compressed else 4 * k)
    nb_new = nb + k

    out_c = np.zeros(n, dtype=np.float64)
    blocks_c: List[sp.csr_matrix] = []
    blocks_b: List[sp.csr_matrix] = []
    if active_idx.size:
        out_c[active_idx] = hz.c[active_idx]
        act_c = hz.Gc[active_idx].tocoo()
        blocks_c.append(
            sp.coo_matrix(
                (act_c.data, (active_idx[act_c.row], act_c.col)),
                shape=(n, ng_new),
                dtype=np.float64,
            ).tocsr()
        )
        if nb:
            act_b = hz.Gb[active_idx].tocoo()
            blocks_b.append(
                sp.coo_matrix(
                    (act_b.data, (active_idx[act_b.row], act_b.col)),
                    shape=(n, nb_new),
                    dtype=np.float64,
                ).tocsr()
            )
    if k:
        beta = ub[unstable_idx]
        out_c[unstable_idx] = beta / 2.0
        xi2_cols = ng + k + np.arange(k, dtype=np.int32)
        blocks_c.append(
            sp.csr_matrix(
                (-beta / 2.0, (unstable_idx, xi2_cols)),
                shape=(n, ng_new),
                dtype=np.float64,
            )
        )

    out_Gc = sum(blocks_c[1:], blocks_c[0]).tocsr() if blocks_c else sparse_empty(n, ng_new)
    out_Gb = sum(blocks_b[1:], blocks_b[0]).tocsr() if blocks_b else sparse_empty(n, nb_new)
    out_Gc.eliminate_zeros()
    out_Gb.eliminate_zeros()

    old_Ac = sparse_pad_cols(hz.Ac, ng_new)
    old_Ab = sparse_pad_cols(hz.Ab, nb_new)
    if k:
        alpha = lb[unstable_idx]
        beta = ub[unstable_idx]
        te = np.arange(k, dtype=np.int32)
        col_xi1 = ng + te
        col_xi2 = ng + k + te
        col_z = nb + te

        rr_c: List[np.ndarray] = []
        cc_c: List[np.ndarray] = []
        dd_c: List[np.ndarray] = []
        rr_b: List[np.ndarray] = []
        cc_b: List[np.ndarray] = []
        dd_b: List[np.ndarray] = []
        add_c = _coo_appender(rr_c, cc_c, dd_c)
        add_b = _coo_appender(rr_b, cc_b, dd_b)

        if compressed:
            r3 = te
            n_relu_eq = k
        else:
            col_xi3 = ng + 2 * k + te
            col_xi4 = ng + 3 * k + te
            r1 = 3 * te
            r2 = r1 + 1
            r3 = r1 + 2
            add_c(r1, col_xi1, np.ones(k))
            add_c(r1, col_xi3, np.ones(k))
            add_c(r2, col_xi2, np.ones(k))
            add_c(r2, col_xi4, np.ones(k))
            add_b(r1, col_z, np.ones(k))
            add_b(r2, col_z, -np.ones(k))
            n_relu_eq = 3 * k

        add_c(r3, col_xi1, alpha / 2.0)
        add_c(r3, col_xi2, -beta / 2.0)
        add_b(r3, col_z, alpha / 2.0)

        pre_gc = hz.Gc[unstable_idx].tocoo()
        if pre_gc.nnz:
            add_c(r3[pre_gc.row], pre_gc.col, -pre_gc.data)
        if nb:
            pre_gb = hz.Gb[unstable_idx].tocoo()
            if pre_gb.nnz:
                add_b(r3[pre_gb.row], pre_gb.col, -pre_gb.data)

        eq_Ac = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (n_relu_eq, ng_new))
        eq_Ab = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (n_relu_eq, nb_new))
        eq_b = np.zeros(n_relu_eq, dtype=np.float64)
        if not compressed:
            eq_b[r1] = 1.0
            eq_b[r2] = 1.0
        eq_b[r3] = hz.c[unstable_idx] - beta / 2.0
    else:
        eq_Ac = sparse_empty(0, ng_new)
        eq_Ab = sparse_empty(0, nb_new)
        eq_b = np.zeros(0, dtype=np.float64)

    Ac = sp.vstack([old_Ac, eq_Ac], format="csr")
    Ab = sp.vstack([old_Ab, eq_Ab], format="csr")
    b = np.concatenate([hz.b, eq_b])
    Ac.eliminate_zeros()
    Ab.eliminate_zeros()

    Auc_base = sparse_pad_cols(hz.Auc if hz.Auc is not None else sparse_empty(0, hz.n_cont), ng_new)
    Aub_base = sparse_pad_cols(hz.Aub if hz.Aub is not None else sparse_empty(0, hz.n_bin), nb_new)
    ub_base = hz.ub if hz.ub is not None else np.zeros(0, dtype=np.float64)
    upper_blocks_c = [Auc_base]
    upper_blocks_b = [Aub_base]
    upper_rhs = [ub_base]

    if compressed and k:
        rows = np.arange(k, dtype=np.int32)
        proj_Ac = sp.coo_matrix(
            (
                np.concatenate([-np.ones(k), -np.ones(k)]),
                (
                    np.concatenate([rows, k + rows]),
                    np.concatenate([ng + rows, ng + k + rows]),
                ),
            ),
            shape=(2 * k, ng_new),
            dtype=np.float64,
        ).tocsr()
        proj_Ab = sp.coo_matrix(
            (
                np.concatenate([-np.ones(k), np.ones(k)]),
                (
                    np.concatenate([rows, k + rows]),
                    np.concatenate([nb + rows, nb + rows]),
                ),
            ),
            shape=(2 * k, nb_new),
            dtype=np.float64,
        ).tocsr()
        upper_blocks_c.append(proj_Ac)
        upper_blocks_b.append(proj_Ab)
        upper_rhs.append(np.zeros(2 * k, dtype=np.float64))

    if valid_cuts and k:
        rr_c = []
        cc_c = []
        dd_c = []
        rr_b = []
        cc_b = []
        dd_b = []
        rhs = np.zeros(2 * k, dtype=np.float64)
        add_cut_c = _coo_appender(rr_c, cc_c, dd_c)
        add_cut_b = _coo_appender(rr_b, cc_b, dd_b)

        row1 = np.arange(k, dtype=np.int32)
        row2 = k + row1
        alpha = lb[unstable_idx]
        beta = ub[unstable_idx]
        slope = beta / np.maximum(beta - alpha, 1e-12)
        xi2_cols = ng + k + np.arange(k, dtype=np.int32)
        pre_gc = hz.Gc[unstable_idx].tocoo()
        if pre_gc.nnz:
            add_cut_c(pre_gc.row, pre_gc.col, pre_gc.data)
            add_cut_c(k + pre_gc.row, pre_gc.col, -slope[pre_gc.row] * pre_gc.data)
        if nb:
            pre_gb = hz.Gb[unstable_idx].tocoo()
            if pre_gb.nnz:
                add_cut_b(pre_gb.row, pre_gb.col, pre_gb.data)
                add_cut_b(k + pre_gb.row, pre_gb.col, -slope[pre_gb.row] * pre_gb.data)
        add_cut_c(row1, xi2_cols, beta / 2.0)
        add_cut_c(row2, xi2_cols, -beta / 2.0)
        rhs[row1] = beta / 2.0 - hz.c[unstable_idx]
        rhs[row2] = -slope * alpha - beta / 2.0 + slope * hz.c[unstable_idx]
        upper_blocks_c.append(_coo_matrix_from_parts(rr_c, cc_c, dd_c, (2 * k, ng_new)))
        upper_blocks_b.append(_coo_matrix_from_parts(rr_b, cc_b, dd_b, (2 * k, nb_new)))
        upper_rhs.append(rhs)

    Auc = sp.vstack(upper_blocks_c, format="csr")
    Aub = sp.vstack(upper_blocks_b, format="csr")
    ub_rhs = np.concatenate(upper_rhs)
    Auc.eliminate_zeros()
    Aub.eliminate_zeros()
    out = SparseHZono(
        out_c, out_Gc, out_Gb, Ac, Ab, b, Auc, Aub, ub_rhs,
    )
    if return_info:
        return out, (int(active.sum()), int(inactive.sum()), k), info
    return out


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_deriv_np(x: np.ndarray) -> np.ndarray:
    y = _sigmoid_np(x)
    return y * (1.0 - y)


def _tanh_np(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0))


def _tanh_deriv_np(x: np.ndarray) -> np.ndarray:
    y = _tanh_np(x)
    return 1.0 - y * y


def _scurve_breakpoints(
    lo: float,
    hi: float,
    K: int,
    grid: str,
    func,
    dfunc,
) -> np.ndarray:
    """Segment breakpoints for one convex/concave side of an S-curve."""

    K = max(1, int(K))
    lo = float(lo)
    hi = float(hi)
    if K == 1 or hi - lo <= 1e-12 or grid == "uniform":
        return np.linspace(lo, hi, K + 1, dtype=np.float64)
    if grid != "curvature":
        raise ValueError(f"unknown S-curve grid {grid!r}")

    xs = np.linspace(lo, hi, max(33, 32 * K + 1), dtype=np.float64)
    h = 1e-4 * (1.0 + np.abs(xs))
    with np.errstate(all="ignore"):
        curv = np.abs((dfunc(xs + h) - dfunc(xs - h)) / (2.0 * h))
    curv = np.where(np.isfinite(curv), curv, 0.0)
    dens = np.sqrt(np.maximum(curv, 0.0)) + 1e-9
    dx = np.diff(xs)
    area = 0.5 * (dens[:-1] + dens[1:]) * dx
    cum = np.concatenate([[0.0], np.cumsum(area)])
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 1e-12:
        return np.linspace(lo, hi, K + 1, dtype=np.float64)
    cuts = np.interp(np.linspace(0.0, total, K + 1), cum, xs).astype(np.float64)
    cuts[0] = lo
    cuts[-1] = hi
    for i in range(1, cuts.size):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]
    return cuts


def _scurve_domain_cut_matrices(
    hz: SparseHZono,
    wide_idx: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    owner: np.ndarray,
    z_cols: np.ndarray,
    n_cont: int,
    n_bin: int,
    lb_w: np.ndarray,
    ub_w: np.ndarray,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Exact-valid cuts tying selected S-curve segments to their x-domain."""

    owner = np.asarray(owner, dtype=np.int32).reshape(-1)
    r = int(owner.size)
    if r == 0:
        return sparse_empty(0, n_cont), sparse_empty(0, n_bin), np.zeros(0, dtype=np.float64)
    seg_a = np.asarray(seg_a, dtype=np.float64).reshape(-1)
    seg_b = np.asarray(seg_b, dtype=np.float64).reshape(-1)
    z_cols = np.asarray(z_cols, dtype=np.int32).reshape(-1)
    if not (seg_a.size == seg_b.size == z_cols.size == r):
        raise ValueError("S-curve domain cut metadata size mismatch")

    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []
    add_c = _coo_appender(rr_c, cc_c, dd_c)
    add_b = _coo_appender(rr_b, cc_b, dd_b)

    upper_rows = np.arange(r, dtype=np.int32)
    lower_rows = r + upper_rows
    g_lo = np.asarray(lb_w, dtype=np.float64).reshape(-1)[owner]
    g_hi = np.asarray(ub_w, dtype=np.float64).reshape(-1)[owner]
    x_center = hz.c[np.asarray(wide_idx, dtype=np.int64)[owner]]
    rhs = np.empty(2 * r, dtype=np.float64)
    rhs[upper_rows] = (g_hi + seg_b) / 2.0 - x_center
    rhs[lower_rows] = -(g_lo + seg_a) / 2.0 + x_center

    add_b(upper_rows, z_cols, -(g_hi - seg_b) / 2.0)
    add_b(lower_rows, z_cols, -(seg_a - g_lo) / 2.0)

    pre_gc = hz.Gc[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    pre_gb = hz.Gb[np.asarray(wide_idx, dtype=np.int64)].tocsr() if hz.n_bin else sparse_empty(len(wide_idx), 0)
    for j in range(len(wide_idx)):
        flats = np.nonzero(owner == j)[0].astype(np.int32)
        if flats.size == 0:
            continue
        c0, c1 = pre_gc.indptr[j], pre_gc.indptr[j + 1]
        if c1 > c0:
            cols = pre_gc.indices[c0:c1].astype(np.int32)
            data = pre_gc.data[c0:c1].astype(np.float64)
            rows = np.repeat(flats, cols.size)
            add_c(rows, np.tile(cols, flats.size), np.tile(data, flats.size))
            add_c(r + rows, np.tile(cols, flats.size), -np.tile(data, flats.size))
        if hz.n_bin:
            b0, b1 = pre_gb.indptr[j], pre_gb.indptr[j + 1]
            if b1 > b0:
                cols = pre_gb.indices[b0:b1].astype(np.int32)
                data = pre_gb.data[b0:b1].astype(np.float64)
                rows = np.repeat(flats, cols.size)
                add_b(rows, np.tile(cols, flats.size), np.tile(data, flats.size))
                add_b(r + rows, np.tile(cols, flats.size), -np.tile(data, flats.size))

    Auc = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (2 * r, n_cont))
    Aub = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (2 * r, n_bin))
    Auc.eliminate_zeros()
    Aub.eliminate_zeros()
    return Auc, Aub, rhs


def _scurve_range_cut_matrices(
    out_c: np.ndarray,
    out_Gc: sp.csr_matrix,
    out_Gb: sp.csr_matrix,
    wide_idx: np.ndarray,
    seg_y_lo: np.ndarray,
    seg_y_hi: np.ndarray,
    owner: np.ndarray,
    z_cols: np.ndarray,
    n_cont: int,
    n_bin: int,
    y_lo_w: np.ndarray,
    y_hi_w: np.ndarray,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Exact-valid cuts tying selected S-curve segments to their y-range."""

    owner = np.asarray(owner, dtype=np.int32).reshape(-1)
    r = int(owner.size)
    if r == 0:
        return sparse_empty(0, n_cont), sparse_empty(0, n_bin), np.zeros(0, dtype=np.float64)
    seg_y_lo = np.asarray(seg_y_lo, dtype=np.float64).reshape(-1)
    seg_y_hi = np.asarray(seg_y_hi, dtype=np.float64).reshape(-1)
    z_cols = np.asarray(z_cols, dtype=np.int32).reshape(-1)
    if not (seg_y_lo.size == seg_y_hi.size == z_cols.size == r):
        raise ValueError("S-curve range cut metadata size mismatch")

    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []
    add_c = _coo_appender(rr_c, cc_c, dd_c)
    add_b = _coo_appender(rr_b, cc_b, dd_b)

    upper_rows = np.arange(r, dtype=np.int32)
    lower_rows = r + upper_rows
    g_lo = np.asarray(y_lo_w, dtype=np.float64).reshape(-1)[owner]
    g_hi = np.asarray(y_hi_w, dtype=np.float64).reshape(-1)[owner]
    y_center = out_c[np.asarray(wide_idx, dtype=np.int64)[owner]]
    rhs = np.empty(2 * r, dtype=np.float64)
    rhs[upper_rows] = (g_hi + seg_y_hi) / 2.0 - y_center
    rhs[lower_rows] = -(g_lo + seg_y_lo) / 2.0 + y_center

    add_b(upper_rows, z_cols, -(g_hi - seg_y_hi) / 2.0)
    add_b(lower_rows, z_cols, -(seg_y_lo - g_lo) / 2.0)

    y_gc = out_Gc[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    y_gb = out_Gb[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    for j in range(len(wide_idx)):
        flats = np.nonzero(owner == j)[0].astype(np.int32)
        if flats.size == 0:
            continue
        c0, c1 = y_gc.indptr[j], y_gc.indptr[j + 1]
        if c1 > c0:
            cols = y_gc.indices[c0:c1].astype(np.int32)
            data = y_gc.data[c0:c1].astype(np.float64)
            rows = np.repeat(flats, cols.size)
            add_c(rows, np.tile(cols, flats.size), np.tile(data, flats.size))
            add_c(r + rows, np.tile(cols, flats.size), -np.tile(data, flats.size))
        b0, b1 = y_gb.indptr[j], y_gb.indptr[j + 1]
        if b1 > b0:
            cols = y_gb.indices[b0:b1].astype(np.int32)
            data = y_gb.data[b0:b1].astype(np.float64)
            rows = np.repeat(flats, cols.size)
            add_b(rows, np.tile(cols, flats.size), np.tile(data, flats.size))
            add_b(r + rows, np.tile(cols, flats.size), -np.tile(data, flats.size))

    Auc = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (2 * r, n_cont))
    Aub = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (2 * r, n_bin))
    Auc.eliminate_zeros()
    Aub.eliminate_zeros()
    return Auc, Aub, rhs


def _scurve_graph_cut_matrices(
    hz: SparseHZono,
    out_c: np.ndarray,
    out_Gc: sp.csr_matrix,
    out_Gb: sp.csr_matrix,
    wide_idx: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    owner: np.ndarray,
    z_cols: np.ndarray,
    n_cont: int,
    n_bin: int,
    lb_w: np.ndarray,
    ub_w: np.ndarray,
    func,
    dfunc,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Conditional convex/concave graph cuts for S-curve segments."""

    owner = np.asarray(owner, dtype=np.int32).reshape(-1)
    r = int(owner.size)
    if r == 0:
        return sparse_empty(0, n_cont), sparse_empty(0, n_bin), np.zeros(0, dtype=np.float64)
    seg_a = np.asarray(seg_a, dtype=np.float64).reshape(-1)
    seg_b = np.asarray(seg_b, dtype=np.float64).reshape(-1)
    z_cols = np.asarray(z_cols, dtype=np.int32).reshape(-1)
    if not (seg_a.size == seg_b.size == z_cols.size == r):
        raise ValueError("S-curve graph cut metadata size mismatch")

    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []
    rhs_vals: List[float] = []

    pre_gc = hz.Gc[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    pre_gb = hz.Gb[np.asarray(wide_idx, dtype=np.int64)].tocsr() if hz.n_bin else sparse_empty(len(wide_idx), 0)
    y_gc = out_Gc[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    y_gb = out_Gb[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    x_center = hz.c[np.asarray(wide_idx, dtype=np.int64)]
    y_center = out_c[np.asarray(wide_idx, dtype=np.int64)]
    x_lo = np.asarray(lb_w, dtype=np.float64).reshape(-1)
    x_hi = np.asarray(ub_w, dtype=np.float64).reshape(-1)
    y_lo = func(x_lo)
    y_hi = func(x_hi)

    def add_c(row: int, cols: np.ndarray, data: np.ndarray, scale: float) -> None:
        if cols.size and abs(scale) > 0.0:
            rr_c.append(np.full(cols.size, int(row), dtype=np.int32))
            cc_c.append(cols.astype(np.int32, copy=False))
            dd_c.append((float(scale) * data).astype(np.float64, copy=False))

    def add_b(row: int, cols: np.ndarray, data: np.ndarray, scale: float) -> None:
        if cols.size and abs(scale) > 0.0:
            rr_b.append(np.full(cols.size, int(row), dtype=np.int32))
            cc_b.append(cols.astype(np.int32, copy=False))
            dd_b.append((float(scale) * data).astype(np.float64, copy=False))

    def append_linear_cut(seg_i: int, ax: float, ay: float, c0: float) -> None:
        seg_owner = int(owner[seg_i])
        max_g = (
            ax * (x_hi[seg_owner] if ax >= 0.0 else x_lo[seg_owner])
            + ay * (y_hi[seg_owner] if ay >= 0.0 else y_lo[seg_owner])
            + c0
        )
        M = max(0.0, float(max_g))
        row = len(rhs_vals)
        rhs_vals.append(
            M / 2.0 - ax * float(x_center[seg_owner]) - ay * float(y_center[seg_owner]) - c0
        )

        c0p, c1p = pre_gc.indptr[seg_owner], pre_gc.indptr[seg_owner + 1]
        add_c(row, pre_gc.indices[c0p:c1p], pre_gc.data[c0p:c1p], ax)
        b0p, b1p = pre_gb.indptr[seg_owner], pre_gb.indptr[seg_owner + 1]
        add_b(row, pre_gb.indices[b0p:b1p], pre_gb.data[b0p:b1p], ax)
        c0y, c1y = y_gc.indptr[seg_owner], y_gc.indptr[seg_owner + 1]
        add_c(row, y_gc.indices[c0y:c1y], y_gc.data[c0y:c1y], ay)
        b0y, b1y = y_gb.indptr[seg_owner], y_gb.indptr[seg_owner + 1]
        add_b(row, y_gb.indices[b0y:b1y], y_gb.data[b0y:b1y], ay)
        rr_b.append(np.array([row], dtype=np.int32))
        cc_b.append(np.array([int(z_cols[seg_i])], dtype=np.int32))
        dd_b.append(np.array([-M / 2.0], dtype=np.float64))

    fa = func(seg_a)
    fb = func(seg_b)
    da = dfunc(seg_a)
    db = dfunc(seg_b)
    for s in range(r):
        a = float(seg_a[s])
        b = float(seg_b[s])
        if b - a <= 1e-12:
            continue
        if a < -1e-12 and b > 1e-12:
            continue
        sec_slope = float((fb[s] - fa[s]) / (b - a))
        sec_inter = float(fa[s] - sec_slope * a)
        tan_x = np.array([a, b], dtype=np.float64)
        tan_y = func(tan_x)
        tan_slope = dfunc(tan_x)
        tan_inter = tan_y - tan_slope * tan_x
        convex = b <= 1e-12
        if convex:
            append_linear_cut(s, -sec_slope, 1.0, -sec_inter)
            for slope_i, inter_i in zip(tan_slope, tan_inter):
                append_linear_cut(s, float(slope_i), -1.0, float(inter_i))
        else:
            append_linear_cut(s, sec_slope, -1.0, sec_inter)
            for slope_i, inter_i in zip(tan_slope, tan_inter):
                append_linear_cut(s, -float(slope_i), 1.0, -float(inter_i))

    n_rows = len(rhs_vals)
    if n_rows == 0:
        return sparse_empty(0, n_cont), sparse_empty(0, n_bin), np.zeros(0, dtype=np.float64)
    Auc = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (n_rows, n_cont))
    Aub = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (n_rows, n_bin))
    Auc.eliminate_zeros()
    Aub.eliminate_zeros()
    return Auc, Aub, np.asarray(rhs_vals, dtype=np.float64)


def sparse_hz_apply_scurve_piecewise(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    K: int = 2,
    func=_sigmoid_np,
    dfunc=_sigmoid_deriv_np,
    inflection: float = 0.0,
    torch_func=None,
    torch_dfunc=None,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
    return_info: bool = False,
):
    """Compressed sparse HZ S-curve encoding with pruned zero-width segments.

    This mirrors the dense ``hz_apply_piecewise(..., compressed=True,
    inflection=0)`` path: each nondegenerate segment gets two continuous local
    generators and one binary selector; the local generator box is represented
    by exact inequality rows rather than four slack equality columns.
    """

    if pre_bounds is None:
        pre_bounds = sparse_hz_fast_bounds(hz)
    grid = str(grid or "uniform")
    lb, ub = _bounds_arrays(pre_bounds, hz.n_out)
    wide = (ub - lb) > 1e-12
    wide_idx = np.nonzero(wide)[0].astype(np.int32)
    narrow_idx = np.nonzero(~wide)[0].astype(np.int32)
    m = int(wide_idx.size)
    n = hz.n_out
    ng = hz.n_cont
    nb = hz.n_bin

    out_c = np.zeros(n, dtype=np.float64)
    if narrow_idx.size:
        out_c[narrow_idx] = func(hz.c[narrow_idx])
    if m == 0:
        out = SparseHZono(
            c=out_c,
            Gc=sparse_empty(n, ng),
            Gb=sparse_empty(n, nb),
            Ac=hz.Ac,
            Ab=hz.Ab,
            b=hz.b,
            Auc=hz.Auc,
            Aub=hz.Aub,
            ub=hz.ub,
        )
        if return_info:
            info = {
                "wide_idx": wide_idx,
                "m": 0,
                "compressed": True,
                "pruned": True,
                "r": 0,
                "owner_arr": np.zeros(0, dtype=np.int32),
                "seg_count": np.zeros(0, dtype=np.int32),
                "a": np.zeros(0, dtype=np.float64),
                "b_seg": np.zeros(0, dtype=np.float64),
                "centers_x": np.zeros(0, dtype=np.float64),
                "centers_y": np.zeros(0, dtype=np.float64),
                "g1_x": np.zeros(0, dtype=np.float64),
                "g1_y": np.zeros(0, dtype=np.float64),
                "g2_x": np.zeros(0, dtype=np.float64),
                "g2_y": np.zeros(0, dtype=np.float64),
                "grid": grid,
            }
            return out, (0, int(narrow_idx.size)), info
        return out

    K_side = max(1, int(K))
    lb_w = lb[wide_idx]
    ub_w = ub[wide_idx]
    pivot = np.maximum(np.minimum(float(inflection), ub_w), lb_w)
    if grid == "uniform":
        sid = np.arange(K_side, dtype=np.float64).reshape(-1, 1)
        w_left = (pivot - lb_w).reshape(1, -1) / K_side
        w_right = (ub_w - pivot).reshape(1, -1) / K_side
        a_left = lb_w.reshape(1, -1) + sid * w_left
        a_right = pivot.reshape(1, -1) + sid * w_right
        a_grid = np.vstack([a_left, a_right])
        b_grid = np.vstack([a_left + w_left, a_right + w_right])
        owner_grid = np.broadcast_to(
            np.arange(m, dtype=np.int32).reshape(1, -1),
            (2 * K_side, m),
        )
        nondeg = (b_grid - a_grid) > 1e-12
        a = a_grid[nondeg].astype(np.float64, copy=False)
        b_seg = b_grid[nondeg].astype(np.float64, copy=False)
        owner = owner_grid[nondeg].astype(np.int32, copy=False)
    else:
        seg_a: List[float] = []
        seg_b: List[float] = []
        seg_owner: List[int] = []
        for j in range(m):
            for lo, hi in ((lb_w[j], pivot[j]), (pivot[j], ub_w[j])):
                if float(hi) - float(lo) <= 1e-12:
                    continue
                cuts = _scurve_breakpoints(
                    float(lo),
                    float(hi),
                    K_side,
                    grid=grid,
                    func=func,
                    dfunc=dfunc,
                )
                for s in range(K_side):
                    if float(cuts[s + 1]) - float(cuts[s]) <= 1e-12:
                        continue
                    seg_a.append(float(cuts[s]))
                    seg_b.append(float(cuts[s + 1]))
                    seg_owner.append(j)
        a = np.asarray(seg_a, dtype=np.float64)
        b_seg = np.asarray(seg_b, dtype=np.float64)
        owner = np.asarray(seg_owner, dtype=np.int32)
    r = int(a.size)
    if r == 0:
        raise ValueError("wide S-curve neuron produced no nondegenerate segment")

    if torch_func is not None and torch_dfunc is not None:
        at = torch.from_numpy(a).double()
        bt = torch.from_numpy(b_seg).double()
        fa_t = torch_func(at)
        fb_t = torch_func(bt)
        la_t = torch_dfunc(at)
        lb_slope_t = torch_dfunc(bt)
        centers_x_t = (at + bt) / 2.0
        centers_y_t = (fa_t + fb_t) / 2.0
        nearly_linear_t = (la_t - lb_slope_t).abs() < 1e-10

        denom_t = lb_slope_t - la_t
        safe_denom_t = torch.where(nearly_linear_t, torch.ones_like(denom_t), denom_t)
        p1_t = (fb_t - fa_t + lb_slope_t * at - la_t * bt) / safe_denom_t
        p2_t = at + bt - p1_t
        g1x_tang_t = (p1_t - at) / 2.0
        g1y_tang_t = lb_slope_t * (p1_t - at) / 2.0
        g2x_tang_t = (p2_t - at) / 2.0
        g2y_tang_t = la_t * (p2_t - at) / 2.0

        half_width_t = (bt - at) / 2.0
        slope_t = (fb_t - fa_t) / (bt - at + 1e-30)
        t_pts = torch.linspace(0.0, 1.0, 50, dtype=torch.float64).view(50, 1)
        pts_t = at.unsqueeze(0) + t_pts * (bt - at).unsqueeze(0)
        f_pts_t = torch_func(pts_t)
        resid_t = f_pts_t - (slope_t.unsqueeze(0) * pts_t + (fa_t - slope_t * at).unsqueeze(0))
        max_err_t = resid_t.abs().max(dim=0).values
        g1x_lin_t = half_width_t
        g1y_lin_t = slope_t * half_width_t
        g2x_lin_t = torch.zeros_like(half_width_t)
        g2y_lin_t = max_err_t

        g1_x_t = torch.where(nearly_linear_t, g1x_lin_t, g1x_tang_t)
        g1_y_t = torch.where(nearly_linear_t, g1y_lin_t, g1y_tang_t)
        g2_x_t = torch.where(nearly_linear_t, g2x_lin_t, g2x_tang_t)
        g2_y_t = torch.where(nearly_linear_t, g2y_lin_t, g2y_tang_t)

        dx_t = pts_t - centers_x_t.unsqueeze(0)
        dy_t = f_pts_t - centers_y_t.unsqueeze(0)
        det_t = g1_y_t * g2_x_t - g1_x_t * g2_y_t
        safe_det_t = torch.where(det_t.abs() < 1e-30, torch.ones_like(det_t), det_t)
        xi1_t = (dy_t * g2_x_t.unsqueeze(0) - dx_t * g2_y_t.unsqueeze(0)) / safe_det_t.unsqueeze(0)
        xi2_t = (dy_t * g1_x_t.unsqueeze(0) - dx_t * g1_y_t.unsqueeze(0)) / (-safe_det_t.unsqueeze(0))
        max_xi_t = torch.maximum(xi1_t.abs().amax(dim=0), xi2_t.abs().amax(dim=0))
        scale_factor_t = torch.where(max_xi_t > 1.0, max_xi_t * 1.01, torch.ones_like(max_xi_t))
        scale_factor_t = torch.where(det_t.abs() < 1e-30, torch.ones_like(scale_factor_t), scale_factor_t)
        g1_x = (g1_x_t * scale_factor_t).numpy()
        g1_y = (g1_y_t * scale_factor_t).numpy()
        g2_x = (g2_x_t * scale_factor_t).numpy()
        g2_y = (g2_y_t * scale_factor_t).numpy()
        centers_x = centers_x_t.numpy()
        centers_y = centers_y_t.numpy()
        fa = fa_t.numpy()
        fb = fb_t.numpy()
    else:
        fa = func(a)
        fb = func(b_seg)
        la = dfunc(a)
        lb_slope = dfunc(b_seg)
        centers_x = (a + b_seg) / 2.0
        centers_y = (fa + fb) / 2.0
        nearly_linear = np.abs(la - lb_slope) < 1e-10

        denom = lb_slope - la
        safe_denom = np.where(nearly_linear, 1.0, denom)
        p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
        p2 = a + b_seg - p1
        g1x_tang = (p1 - a) / 2.0
        g1y_tang = lb_slope * (p1 - a) / 2.0
        g2x_tang = (p2 - a) / 2.0
        g2y_tang = la * (p2 - a) / 2.0

        half_width = (b_seg - a) / 2.0
        slope = (fb - fa) / (b_seg - a + 1e-30)
        t_pts = np.linspace(0.0, 1.0, 50, dtype=np.float64).reshape(50, 1)
        pts = a.reshape(1, r) + t_pts * (b_seg - a).reshape(1, r)
        f_pts = func(pts)
        resid = f_pts - (slope.reshape(1, r) * pts + (fa - slope * a).reshape(1, r))
        max_err = np.max(np.abs(resid), axis=0)
        g1x_lin = half_width
        g1y_lin = slope * half_width
        g2x_lin = np.zeros_like(half_width)
        g2y_lin = max_err

        g1_x = np.where(nearly_linear, g1x_lin, g1x_tang)
        g1_y = np.where(nearly_linear, g1y_lin, g1y_tang)
        g2_x = np.where(nearly_linear, g2x_lin, g2x_tang)
        g2_y = np.where(nearly_linear, g2y_lin, g2y_tang)

        dx = pts - centers_x.reshape(1, r)
        dy = f_pts - centers_y.reshape(1, r)
        det = g1_y * g2_x - g1_x * g2_y
        safe_det = np.where(np.abs(det) < 1e-30, 1.0, det)
        xi1 = (dy * g2_x.reshape(1, r) - dx * g2_y.reshape(1, r)) / safe_det.reshape(1, r)
        xi2 = (dy * g1_x.reshape(1, r) - dx * g1_y.reshape(1, r)) / (-safe_det.reshape(1, r))
        max_xi = np.maximum(np.max(np.abs(xi1), axis=0), np.max(np.abs(xi2), axis=0))
        scale_factor = np.where(max_xi > 1.0, max_xi * 1.01, 1.0)
        scale_factor = np.where(np.abs(det) < 1e-30, 1.0, scale_factor)
        g1_x *= scale_factor
        g1_y *= scale_factor
        g2_x *= scale_factor
        g2_y *= scale_factor

    out_c[wide_idx] = np.bincount(owner, weights=centers_y, minlength=m) / 2.0
    seg_count = np.bincount(owner, minlength=m).astype(np.float64)
    center_x_sum = np.bincount(owner, weights=centers_x, minlength=m)

    ng_total = ng + 2 * r
    nb_total = nb + r
    g1_cols = ng + np.arange(r, dtype=np.int32)
    g2_cols = ng + r + np.arange(r, dtype=np.int32)
    z_cols = nb + np.arange(r, dtype=np.int32)
    wide_rows = wide_idx[owner]
    out_Gc = sp.coo_matrix(
        (
            np.concatenate([g1_y, g2_y]),
            (
                np.concatenate([wide_rows, wide_rows]),
                np.concatenate([g1_cols, g2_cols]),
            ),
        ),
        shape=(n, ng_total),
        dtype=np.float64,
    ).tocsr()
    out_Gb = sp.coo_matrix(
        (-centers_y / 2.0, (wide_rows, z_cols)),
        shape=(n, nb_total),
        dtype=np.float64,
    ).tocsr()
    out_Gc.eliminate_zeros()
    out_Gb.eliminate_zeros()

    n_eq_new = 2 * m
    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []
    add_c = _coo_appender(rr_c, cc_c, dd_c)
    add_b = _coo_appender(rr_b, cc_b, dd_b)

    link_rows = np.arange(m, dtype=np.int32)
    sum_rows = m + link_rows
    add_c(link_rows[owner], g1_cols, -g1_x)
    add_c(link_rows[owner], g2_cols, -g2_x)
    add_b(link_rows[owner], z_cols, centers_x / 2.0)
    pre_gc = hz.Gc[wide_idx].tocoo()
    if pre_gc.nnz:
        add_c(link_rows[pre_gc.row], pre_gc.col, pre_gc.data)
    if nb:
        pre_gb = hz.Gb[wide_idx].tocoo()
        if pre_gb.nnz:
            add_b(link_rows[pre_gb.row], pre_gb.col, pre_gb.data)
    add_b(sum_rows[owner], z_cols, np.ones(r, dtype=np.float64))

    eq_Ac = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (n_eq_new, ng_total))
    eq_Ab = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (n_eq_new, nb_total))
    eq_b = np.zeros(n_eq_new, dtype=np.float64)
    eq_b[link_rows] = center_x_sum / 2.0 - hz.c[wide_idx]
    eq_b[sum_rows] = seg_count - 2.0

    Ac = sp.vstack([sparse_pad_cols(hz.Ac, ng_total), eq_Ac], format="csr")
    Ab = sp.vstack([sparse_pad_cols(hz.Ab, nb_total), eq_Ab], format="csr")
    Ac.eliminate_zeros()
    Ab.eliminate_zeros()
    b = np.concatenate([hz.b, eq_b])

    base_Auc = sparse_pad_cols(
        hz.Auc if hz.Auc is not None else sparse_empty(0, hz.n_cont),
        ng_total,
    )
    base_Aub = sparse_pad_cols(
        hz.Aub if hz.Aub is not None else sparse_empty(0, hz.n_bin),
        nb_total,
    )
    base_ub = hz.ub if hz.ub is not None else np.zeros(0, dtype=np.float64)
    box_rows = 4 * np.arange(r, dtype=np.int32)
    box_Auc = sp.coo_matrix(
        (
            np.concatenate([np.ones(r), -np.ones(r), np.ones(r), -np.ones(r)]),
            (
                np.concatenate([box_rows, box_rows + 1, box_rows + 2, box_rows + 3]),
                np.concatenate([g1_cols, g1_cols, g2_cols, g2_cols]),
            ),
        ),
        shape=(4 * r, ng_total),
        dtype=np.float64,
    ).tocsr()
    box_Aub = sp.coo_matrix(
        (
            0.5 * np.ones(4 * r, dtype=np.float64),
            (
                np.concatenate([box_rows, box_rows + 1, box_rows + 2, box_rows + 3]),
                np.concatenate([z_cols, z_cols, z_cols, z_cols]),
            ),
        ),
        shape=(4 * r, nb_total),
        dtype=np.float64,
    ).tocsr()
    Auc = sp.vstack([base_Auc, box_Auc], format="csr")
    Aub = sp.vstack([base_Aub, box_Aub], format="csr")
    ub_rhs = np.concatenate([base_ub, 0.5 * np.ones(4 * r, dtype=np.float64)])

    if domain_cuts:
        dom_Auc, dom_Aub, dom_rhs = _scurve_domain_cut_matrices(
            hz,
            wide_idx,
            a,
            b_seg,
            owner,
            z_cols,
            ng_total,
            nb_total,
            lb_w,
            ub_w,
        )
        rng_Auc, rng_Aub, rng_rhs = _scurve_range_cut_matrices(
            out_c,
            out_Gc,
            out_Gb,
            wide_idx,
            fa,
            fb,
            owner,
            z_cols,
            ng_total,
            nb_total,
            func(lb_w),
            func(ub_w),
        )
        Auc = sp.vstack([Auc, dom_Auc, rng_Auc], format="csr")
        Aub = sp.vstack([Aub, dom_Aub, rng_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, dom_rhs, rng_rhs])

    if graph_cuts:
        graph_Auc, graph_Aub, graph_rhs = _scurve_graph_cut_matrices(
            hz,
            out_c,
            out_Gc,
            out_Gb,
            wide_idx,
            a,
            b_seg,
            owner,
            z_cols,
            ng_total,
            nb_total,
            lb_w,
            ub_w,
            func=func,
            dfunc=dfunc,
        )
        Auc = sp.vstack([Auc, graph_Auc], format="csr")
        Aub = sp.vstack([Aub, graph_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, graph_rhs])

    Auc.eliminate_zeros()
    Aub.eliminate_zeros()

    out = SparseHZono(
        out_c, out_Gc, out_Gb, Ac, Ab, b, Auc, Aub, ub_rhs,
    )
    if return_info:
        info = {
            "wide_idx": wide_idx,
            "m": m,
            "compressed": True,
            "pruned": True,
            "r": r,
            "owner_arr": owner,
            "seg_count": seg_count.astype(np.int32, copy=False),
            "a": a,
            "b_seg": b_seg,
            "centers_x": centers_x,
            "centers_y": centers_y,
            "g1_x": g1_x,
            "g1_y": g1_y,
            "g2_x": g2_x,
            "g2_y": g2_y,
            "grid": grid,
        }
        return out, (m, int(narrow_idx.size)), info
    return out


def sparse_hz_apply_sigmoid_piecewise(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    K: int = 2,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
) -> SparseHZono:
    return sparse_hz_apply_scurve_piecewise(
        hz,
        pre_bounds,
        K=K,
        func=_sigmoid_np,
        dfunc=_sigmoid_deriv_np,
        inflection=0.0,
        torch_func=torch.sigmoid,
        torch_dfunc=lambda x: torch.sigmoid(x) * (1.0 - torch.sigmoid(x)),
        domain_cuts=domain_cuts,
        graph_cuts=graph_cuts,
        grid=grid,
    )


def sparse_hz_apply_tanh_piecewise(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    K: int = 1,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
) -> SparseHZono:
    return sparse_hz_apply_scurve_piecewise(
        hz,
        pre_bounds,
        K=K,
        func=_tanh_np,
        dfunc=_tanh_deriv_np,
        inflection=0.0,
        torch_func=torch.tanh,
        torch_dfunc=lambda x: 1.0 - torch.tanh(x) ** 2,
        domain_cuts=domain_cuts,
        graph_cuts=graph_cuts,
        grid=grid,
    )



def _broadcast_param(value, n: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().double().numpy().reshape(-1)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if n % arr.size == 0:
        return np.tile(arr, n // arr.size)
    raise ValueError(f"cannot broadcast parameter of size {arr.size} to {n}")


def sparse_hz_add_const(hz: SparseHZono, bias) -> SparseHZono:
    return SparseHZono(
        c=hz.c + _broadcast_param(bias, hz.n_out),
        Gc=hz.Gc,
        Gb=hz.Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
    )


def sparse_hz_scale(hz: SparseHZono, scale) -> SparseHZono:
    a = _broadcast_param(scale, hz.n_out)
    D = sp.diags(a, format="csr")
    return sparse_hz_linear(hz, D, None)


def sparse_hz_gather_rows(hz: SparseHZono, rows: Sequence[int]) -> SparseHZono:
    row_idx = np.asarray(rows, dtype=np.int64).reshape(-1)
    return SparseHZono(
        c=hz.c[row_idx].copy(),
        Gc=hz.Gc[row_idx].tocsr(),
        Gb=hz.Gb[row_idx].tocsr(),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
    )


def _csr_equal(a: sp.csr_matrix, b: sp.csr_matrix, *, atol: float = 0.0) -> bool:
    if a.shape != b.shape:
        return False
    d = (a - b).tocsr()
    d.eliminate_zeros()
    if d.nnz == 0:
        return True
    return bool(np.max(np.abs(d.data)) <= atol)


def _constraints_start_with(
    Ac_big: sp.csr_matrix,
    Ab_big: sp.csr_matrix,
    b_big: np.ndarray,
    Ac_small: sp.csr_matrix,
    Ab_small: sp.csr_matrix,
    b_small: np.ndarray,
) -> bool:
    n = int(Ac_small.shape[0])
    if n > Ac_big.shape[0] or n != Ab_small.shape[0] or n > Ab_big.shape[0]:
        return False
    return (
        _csr_equal(Ac_big[:n], Ac_small)
        and _csr_equal(Ab_big[:n], Ab_small)
        and np.array_equal(b_big[:n], b_small)
    )


def _merge_equalities(parts: List[SparseHZono], n_cont: int, n_bin: int):
    Ac = sparse_empty(0, n_cont)
    Ab = sparse_empty(0, n_bin)
    b = np.zeros(0, dtype=np.float64)
    for part in parts:
        pAc = sparse_pad_cols(part.Ac, n_cont)
        pAb = sparse_pad_cols(part.Ab, n_bin)
        if _constraints_start_with(pAc, pAb, part.b, Ac, Ab, b):
            Ac, Ab, b = pAc, pAb, part.b
        elif _constraints_start_with(Ac, Ab, b, pAc, pAb, part.b):
            continue
        elif pAc.shape[0]:
            Ac = sp.vstack([Ac, pAc], format="csr")
            Ab = sp.vstack([Ab, pAb], format="csr")
            b = np.concatenate([b, part.b])
    return Ac, Ab, b


def _merge_uppers(parts: List[SparseHZono], n_cont: int, n_bin: int):
    Auc = sparse_empty(0, n_cont)
    Aub = sparse_empty(0, n_bin)
    ub = np.zeros(0, dtype=np.float64)
    for part in parts:
        pAuc = sparse_pad_cols(part.Auc if part.Auc is not None else sparse_empty(0, part.n_cont), n_cont)
        pAub = sparse_pad_cols(part.Aub if part.Aub is not None else sparse_empty(0, part.n_bin), n_bin)
        pub = part.ub if part.ub is not None else np.zeros(0, dtype=np.float64)
        if _constraints_start_with(pAuc, pAub, pub, Auc, Aub, ub):
            Auc, Aub, ub = pAuc, pAub, pub
        elif _constraints_start_with(Auc, Aub, ub, pAuc, pAub, pub):
            continue
        elif pAuc.shape[0]:
            Auc = sp.vstack([Auc, pAuc], format="csr")
            Aub = sp.vstack([Aub, pAub], format="csr")
            ub = np.concatenate([ub, pub])
    return Auc, Aub, ub


def sparse_hz_concat(parts: Iterable[SparseHZono]) -> SparseHZono:
    parts = list(parts)
    if not parts:
        raise ValueError("sparse_hz_concat requires at least one part")
    n_cont = max(p.n_cont for p in parts)
    n_bin = max(p.n_bin for p in parts)
    padded = [sparse_hz_pad_frame(p, n_cont, n_bin) for p in parts]
    Ac, Ab, b = _merge_equalities(padded, n_cont, n_bin)
    Auc, Aub, ub = _merge_uppers(padded, n_cont, n_bin)
    Gc = sp.vstack([p.Gc for p in padded], format="csr")
    Gb = sp.vstack([p.Gb for p in padded], format="csr")
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=np.concatenate([p.c for p in padded]),
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
    )


def sparse_hz_add_same_frame(x: SparseHZono, y: SparseHZono) -> SparseHZono:
    """Exact sum when both HZs use the same global generator frame."""

    n_cont = max(x.n_cont, y.n_cont)
    n_bin = max(x.n_bin, y.n_bin)
    xp = sparse_hz_pad_frame(x, n_cont, n_bin)
    yp = sparse_hz_pad_frame(y, n_cont, n_bin)
    if xp.n_out != yp.n_out:
        raise ValueError(f"add shape mismatch: {xp.n_out} vs {yp.n_out}")
    Ac, Ab, b = _merge_equalities([xp, yp], n_cont, n_bin)
    Auc, Aub, ub = _merge_uppers([xp, yp], n_cont, n_bin)
    Gc = (xp.Gc + yp.Gc).tocsr()
    Gb = (xp.Gb + yp.Gb).tocsr()
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=xp.c + yp.c,
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
    )


def sparse_hz_sub_same_frame(x: SparseHZono, y: SparseHZono) -> SparseHZono:
    return sparse_hz_add_same_frame(x, sparse_hz_scale(y, -1.0))
