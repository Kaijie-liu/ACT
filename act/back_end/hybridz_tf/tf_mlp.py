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
import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import (
    HZono,
    _fresh_col_ids,
    hz_multiply,
    hz_add_const,
    hz_minkowski_sum,
    hz_from_bounds,
    hz_compute_bounds,
    hz_inherit_known_nonempty,
    hz_mark_known_nonempty,
)
import act.back_end.interval_tf.tf_mlp as interval
import act.back_end.interval_tf.tf_cnn as interval_cnn


def _hz_fact(fact: Fact, hz: HZono) -> Fact:
    """Combine HZ-refined bounds (flat ``(n, 1)`` shape) with interval's
    batch-aware fact: reshape HZ bounds to match ``fact.bounds`` and keep
    interval's constraint set. Use everywhere a hybridz handler returns
    after refining the HZ cache.
    """
    hb = hz_compute_bounds(hz)
    # Intersect the HZ box with interval's box (Bird Prop 3.2.10: intersect with
    # the interval hull). Both are sound, so the tighter of the two is sound --
    # this guarantees the HZ fast bound is NEVER worse than interval (e.g. the
    # maxpool fold over-approximates the per-output bound while still tracking
    # correlation in the cached HZ for the verdict).
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
        # Precision knob: LP-tight pre-activation [alpha,beta]. ReLU is always
        # encoded EXACTLY (eq_lagr binary) -- no triangle/convex relaxation exists.
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
            # Share-merge generators by factor id when both branches descend
            # from a common ancestor (residual y = x + f(x)); avoids ng-doubling
            # AND the correlation loss of independent Minkowski. Falls back to
            # Minkowski when ids are untracked. Sound either way.
            from act.back_end.hybridz_tf.algorithms.sgm import hz_sgm_add
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
                # Var-var multiplication is not an exact HybridZ operator in
                # this backend.  Drop the HZ state and let the caller return
                # UNKNOWN instead of silently installing a relaxation.
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
    row_idx = _upsample_nearest_row_idx(
        L, hz_in.c.shape[0], len(L.out_vars), hz_in.c.device)
    if row_idx is None:
        tf._hz_cache.pop(L.id, None)
        return fact
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
        ri = _reduce_sum_row_map(L, hz_in.c.shape[0], fact.bounds.lb.numel())
        if ri is not None:
            ri = ri.to(device=hz_in.c.device, dtype=torch.long)
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
            # Concat = stack output rows (NOT minkowski_sum, which sums in one
            # coordinate space => wrong output dimension + crashes on unequal
            # branch dims). hz_concat row-stacks with col_id alignment so the
            # cross-branch correlation is preserved exactly.
            from act.back_end.hybridz_tf.algorithms.sgm import hz_concat
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
            from act.back_end.hybridz_tf.algorithms.sgm import hz_sub as _hz_sub
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


def _upsample_nearest_row_idx(
    L, n_in: int, n_out: int, device: torch.device
) -> torch.Tensor | None:
    """Row map for nearest-neighbor upsample. This is exact: each output row is
    a copy of one input row."""
    mode = str(L.params.get("mode", "nearest")).lower()
    if mode != "nearest":
        return None
    in_shape = L.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    if _prod(in_shape) != int(n_in) or len(in_shape) < 3:
        return None
    view_shape = (1, *in_shape) if len(in_shape) == 3 else in_shape
    spatial_rank = len(view_shape) - 2
    size = L.params.get("size")
    scale_factor = L.params.get("scale_factor")
    if size is not None and isinstance(size, (list, tuple)):
        size = tuple(int(s) for s in size)
        if len(size) > spatial_rank:
            size = size[-spatial_rank:]
    if scale_factor is not None and isinstance(scale_factor, (list, tuple)):
        scale_factor = tuple(float(s) for s in scale_factor)
        if len(scale_factor) > spatial_rank:
            scale_factor = scale_factor[-spatial_rank:]
    if size is None and scale_factor is None:
        out_shape = L.params.get("output_shape")
        if out_shape is None:
            return None
        out_shape = tuple(int(d) for d in out_shape)
        out_view_shape = (1, *out_shape) if len(out_shape) == 3 else out_shape
        if len(out_view_shape) != len(view_shape):
            return None
        size = out_view_shape[2:]
    base = torch.arange(n_in, dtype=torch.float64, device=device).view(*view_shape)
    out = F.interpolate(
        base,
        size=size,
        scale_factor=scale_factor,
        mode="nearest",
    )
    idx = out.reshape(-1).to(dtype=torch.long)
    if idx.numel() != int(n_out):
        return None
    return idx


def _reduce_sum_row_map(L, n_in: int, n_out: int) -> torch.Tensor | None:
    in_shape = L.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    per = _prod(in_shape)
    if per == 0 or n_in % per != 0:
        return None
    B = n_in // per
    axes = L.params.get("axes")
    axes = list(range(len(in_shape))) if not axes else [int(a) for a in axes]
    axes = [(a + len(in_shape)) if a < 0 else a for a in axes]
    keepdims = bool(L.params.get("keepdims", 0))
    out_shape = []
    for i, d in enumerate(in_shape):
        if i in axes:
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(d)
    if _prod(out_shape) * B != n_out:
        return None
    out_idx = torch.arange(n_out).view(B, *out_shape)
    view_shape = [B]
    for i, d in enumerate(in_shape):
        view_shape.append(1 if i in axes else d)
    return out_idx.view(*view_shape).expand(B, *in_shape).reshape(-1)


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


def _slice_row_idx(L, n):
    """Output->input row map for a Slice, mirroring interval.tf_slice exactly."""
    if "input_shape" not in L.params:
        return None
    inp_shape = tuple(int(d) for d in L.params["input_shape"])
    per = 1
    for d in inp_shape:
        per *= d
    if per == 0 or n % per != 0:
        return None
    B = n // per
    idx = torch.arange(n).view(B, *inp_shape)
    starts = L.params.get("starts", [])
    ends = L.params.get("ends", [])
    axes = L.params.get("axes", list(range(len(inp_shape))))
    steps = L.params.get("steps", [1] * len(axes))
    slices = [slice(None)] * (len(inp_shape) + 1)
    for i, axis in enumerate(axes):
        axis = int(axis)
        e = ends[i]
        if e > inp_shape[axis]:
            e = inp_shape[axis]
        slices[axis + 1] = slice(starts[i], e, steps[i])
    return idx[tuple(slices)].reshape(-1)


def tf_slice(L, bounds, tf):
    fact = interval.tf_slice(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ri = _slice_row_idx(L, hz_in.c.shape[0])
        if ri is not None and ri.numel() == fact.bounds.lb.shape[-1]:
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, ri)
            return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def _gather_row_idx(L, n):
    if "input_shape" not in L.params:
        return None
    inp_shape = tuple(int(d) for d in L.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or n % per != 0:
        return None
    B = n // per
    axis = int(L.params.get("axis", 0))
    if axis < 0:
        axis += len(inp_shape)
    raw_idx = L.params["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long)
    elif hasattr(raw_idx, "detach"):
        indices = raw_idx.detach().cpu().long()
    else:
        indices = torch.as_tensor(raw_idx, dtype=torch.long)
    idx = torch.arange(n).view(B, *inp_shape)
    return torch.index_select(idx, dim=axis + 1, index=indices.reshape(-1)).reshape(-1)


def tf_gather(L, bounds, tf):
    fact = interval.tf_gather(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ri = _gather_row_idx(L, hz_in.c.shape[0])
        if ri is not None and ri.numel() == fact.bounds.lb.numel():
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, ri)
            return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def _expand_row_idx(L, n):
    """Output->input row map for an Expand/broadcast (repeated rows)."""
    in_shape = L.params.get("input_shape")
    out_shape = L.params.get("output_shape") or L.params.get("shape")
    if in_shape is None or out_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    out_shape = tuple(int(d) for d in out_shape)
    per = 1
    for d in in_shape:
        per *= d
    if per == 0 or n % per != 0:
        return None
    B = n // per
    try:
        return torch.arange(n).view(B, *in_shape).broadcast_to(B, *out_shape).reshape(-1)
    except RuntimeError:
        return None


def tf_expand(L, bounds, tf):
    fact = interval.tf_expand(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ri = _expand_row_idx(L, hz_in.c.shape[0])
        if ri is not None and ri.numel() == fact.bounds.lb.shape[-1]:
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, ri)
            return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


# --- HZ activation encodings (zonotope domain) ---


_RELU_TIGHT_MAX_DIM = 4096  # LP-tight does 2n scipy solves; above this (wide conv
                            # feature maps) it's thousands of LPs -> use fast box.


def _relu_preact_bounds(hz: HZono, tight: bool):
    """Pre-activation [alpha,beta] for ReLU classification. Fast |Gc|+|Gb| box by
    default; the CONSTRAINED LP-relaxation (scipy convex hull, sound, Gurobi-free
    per P3) when ``tight`` and constraints exist AND the layer isn't too wide.
    Both enclose the true range; LP-relax is tighter, so it feeds a tighter ReLU
    encoding. Fast-box stable rows are already soundly classified, so tight LPs
    are only needed for ambiguous rows unless HZ_RELU_TIGHT_ALL_ROWS=1 requests
    the old full-layer diagnostic path."""
    fb = hz_compute_bounds(hz)
    if not tight or hz.Ac.shape[0] == 0 or hz.c.shape[0] > _RELU_TIGHT_MAX_DIM:
        return fb
    flat_lb = fb.lb.reshape(-1)
    flat_ub = fb.ub.reshape(-1)
    ambiguous = torch.nonzero((flat_lb < 0) & (flat_ub > 0), as_tuple=False).reshape(-1)
    if ambiguous.numel() == 0:
        return fb
    try:
        from act.back_end.solver.solver_hz import _hz_compute_bounds_scipy
        if os.environ.get("HZ_RELU_TIGHT_ALL_ROWS", "").strip().lower() in {
            "1", "true", "yes", "on"}:
            tb = _hz_compute_bounds_scipy(hz)
            return Bounds(lb=torch.maximum(tb.lb, fb.lb),
                          ub=torch.minimum(tb.ub, fb.ub))
        try:
            max_rows = int(os.environ.get("HZ_RELU_TIGHT_MAX_ROWS", "0") or 0)
        except Exception:
            max_rows = 0
        if max_rows > 0 and ambiguous.numel() > max_rows:
            # Tightening is only useful when it can flip a fast-box ambiguous
            # row to stable. Prioritize rows closest to a phase boundary and
            # leave the rest at the original sound fast bound.
            score = torch.minimum(flat_lb[ambiguous].abs(), flat_ub[ambiguous].abs())
            keep = torch.argsort(score)[:max_rows]
            ambiguous = ambiguous[keep]
        idx = ambiguous.to(device=flat_lb.device)
        amb_np = ambiguous.detach().cpu().numpy()
        tb = _hz_compute_bounds_scipy(
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
        # Both sound -> take the tighter (intersection). Numerically robust.
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
    the fast |Gc|+|Gb| box. Toy validation showed the fast box is the dominant
    looseness source (1.93x vs the HZ structure's 1.2x); the LP-tight box
    encloses the true pre-activation range (sound) but is much tighter, so the
    eq_lagr/triangle relaxation it feeds is tighter too. Costs an LP per
    dimension -- use on the verdict/precision path, not the fast path.

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

    # Exact-only: EVERY unstable neuron gets the exact eq_lagr binary graph.
    # No triangle/convex partition -- the HZ domain is exact by construction.
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

    # --- EXACT eq_lagr block on exact_sel (ke neurons) ---
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

        # Avoid a full hz.Gc[exact_sel] / hz.Gb[exact_sel] temporary on wide HZs.
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

        # Redundant exact facets: x - y <= 0.
        Ac_out[cr1, col_xi2] = b_e / 2.0
        b_out[cr1, 0] = b_e / 2.0 - hz.c[exact_sel, 0]

        # Redundant exact facets: y - s*x <= -s*alpha.
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
    # eq_mask: original senses + ReLU equality rows + projected/cut inequality
    # rows. The uncompressed, no-cut path preserves all-equality None semantics.
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
    from act.back_end.solver.solver_hz import _fresh_col_ids
    dev = hz.col_ids.device
    new_col = torch.cat([
        hz.col_ids,
        _fresh_col_ids((2 if compressed else 4) * k, device=dev),
    ])
    base_b = (hz.bcol_ids if hz.bcol_ids is not None
              else torch.zeros(0, dtype=torch.long, device=dev))
    new_bcol = torch.cat([base_b, _fresh_col_ids(k, device=dev)])
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
        # No unstable neurons: columns unchanged -> factor ids carry through.
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
        new_col_ids = torch.cat([hz.col_ids, _fresh_col_ids(n_real, device=id_dev)])
        base_b = (hz.bcol_ids if hz.bcol_ids is not None
                  else torch.zeros(0, dtype=torch.long, device=id_dev))
        new_bcol_ids = torch.cat([base_b.to(id_dev), _fresh_col_ids(r, device=id_dev)])

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
    *,
    compressed: bool = True,
    cell_budget: int | None = None,
) -> HZono | None:
    """Piecewise linear approximation for monotone activations (tangent parallelogram).

    ``inflection`` (e.g. 0.0 for sigmoid/tanh): force the inflection point to be a
    SEGMENT BOUNDARY. The tangent parallelogram assumes the function is convex OR
    concave on each segment; a segment straddling the inflection violates that, so
    the soundness slack inflates and tightness becomes ERRATIC in K (toy: K=3/8
    loosen). Splitting K segments on each side of the inflection (=> 2K segments,
    inflection always a boundary) keeps every segment convex/concave so tightness
    is MONOTONE in K.

    ``compressed`` projects the per-segment slack-box equalities exactly into
    two inequality rows per local generator. The represented HZ approximation is
    unchanged, but the dense state drops four continuous slack columns per
    segment, which is the dominant sigmoid/tanh memory cost on dist_shift/cGAN.
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
        # Inflection-aware: K segments on each side of the (per-neuron clamped)
        # inflection point so none straddles it. -> 2K segments; K reassigned to
        # the actual segment count so all downstream code is unchanged.
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
            _fresh_col_ids(n_real + n_slack, device=id_dev),
        ])
        base_b = (hz.bcol_ids if hz.bcol_ids is not None
                  else torch.zeros(0, dtype=torch.long, device=id_dev))
        new_bcol_ids = torch.cat([
            base_b.to(id_dev),
            _fresh_col_ids(K * m, device=id_dev),
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
    from act.back_end.hybridz_tf.algorithms.order_reduce import hz_remove_redundancy
    return hz_remove_redundancy(hz)


def _test_hz_mul_exact_point_and_var_drop() -> None:  # pragma: no cover
    """Lock strict-HZ multiplication semantics.

    HZ times a point tensor is exact and should keep the HZ cache.  HZ times HZ
    would need a non-exact bilinear relaxation in this backend, so strict
    HybridZ must drop the carried HZ instead of silently installing a relaxed
    state.
    """
    from act.back_end.core import ConSet, Layer

    dtype = torch.float64
    device = torch.device("cpu")

    class DummyNet:
        preds = {2: [0, 1]}

        @staticmethod
        def get_predecessor_bounds(layer_id, after, before, pred_index=0):
            pred = DummyNet.preds[layer_id][pred_index]
            return after[pred].bounds if pred in after else before[pred].bounds

    class DummyTF:
        pass

    layer = Layer(
        id=2,
        kind="MUL",
        params={"x_vars": [0, 1], "y_vars": [2, 3]},
        in_vars=[0, 1, 2, 3],
        out_vars=[4, 5],
    )
    bx = Bounds(
        lb=torch.tensor([-1.0, 0.25], dtype=dtype, device=device),
        ub=torch.tensor([2.0, 1.25], dtype=dtype, device=device),
    )
    by_point = Bounds(
        lb=torch.tensor([2.0, -1.5], dtype=dtype, device=device),
        ub=torch.tensor([2.0, -1.5], dtype=dtype, device=device),
    )
    before = {}
    after_point = {
        0: Fact(bounds=bx, cons=ConSet()),
        1: Fact(bounds=by_point, cons=ConSet()),
    }

    tf = DummyTF()
    tf._net = DummyNet()
    tf._before = before
    tf._after = after_point
    tf._hz_cache = {
        2: hz_from_bounds(bx, dtype, device, track_ids=True),
        1: hz_from_bounds(by_point, dtype, device, track_ids=True),
    }
    tf_mul(layer, None, tf)
    if 2 not in tf._hz_cache:
        raise AssertionError("point multiply unexpectedly dropped exact HZ")
    hb = hz_compute_bounds(tf._hz_cache[2])
    expected = interval.tf_mul(layer, bx, by_point).bounds
    if not (torch.allclose(hb.lb, expected.lb) and torch.allclose(hb.ub, expected.ub)):
        raise AssertionError("point multiply HZ bounds mismatch")

    by_var = Bounds(
        lb=torch.tensor([-0.5, 0.5], dtype=dtype, device=device),
        ub=torch.tensor([1.5, 2.0], dtype=dtype, device=device),
    )
    after_var = {
        0: Fact(bounds=bx, cons=ConSet()),
        1: Fact(bounds=by_var, cons=ConSet()),
    }
    tf = DummyTF()
    tf._net = DummyNet()
    tf._before = before
    tf._after = after_var
    tf._hz_cache = {
        2: hz_from_bounds(bx, dtype, device, track_ids=True),
        1: hz_from_bounds(by_var, dtype, device, track_ids=True),
    }
    tf_mul(layer, None, tf)
    if 2 in tf._hz_cache:
        raise AssertionError("var-var multiply must drop strict exact-HZ state")


def _test_upsample_nearest_row_idx_3d4d() -> None:  # pragma: no cover
    """Nearest upsample row maps must match NCHW row-copy semantics."""

    from types import SimpleNamespace

    def manual_rows(in_shape, out_shape):
        if len(in_shape) == 3:
            bsz, ch, in_h, in_w = (1, *in_shape)
            out_bsz, out_ch, out_h, out_w = (1, *out_shape)
        else:
            bsz, ch, in_h, in_w = in_shape
            out_bsz, out_ch, out_h, out_w = out_shape
        if (bsz, ch) != (out_bsz, out_ch):
            raise AssertionError("manual upsample shape mismatch")
        rows = []
        for n in range(bsz):
            for c in range(ch):
                for oh in range(out_h):
                    ih = min(int(oh * in_h // out_h), in_h - 1)
                    for ow in range(out_w):
                        iw = min(int(ow * in_w // out_w), in_w - 1)
                        rows.append(((n * ch + c) * in_h + ih) * in_w + iw)
        return torch.tensor(rows, dtype=torch.long, device=torch.device("cpu"))

    for in_shape, out_shape in [
        ((2, 3, 2, 2), (2, 3, 4, 6)),
        ((3, 2, 2), (3, 4, 6)),
    ]:
        layer = SimpleNamespace(params={
            "mode": "nearest",
            "input_shape": in_shape,
            "output_shape": out_shape,
        })
        rows = _upsample_nearest_row_idx(
            layer,
            int(_prod(in_shape)),
            int(_prod(out_shape)),
            torch.device("cpu"),
        )
        expected = manual_rows(in_shape, out_shape)
        if rows is None or not torch.equal(rows.cpu(), expected):
            raise AssertionError(
                f"nearest upsample row-map mismatch: "
                f"in={in_shape}, out={out_shape}, rows={None if rows is None else rows[:8]}, "
                f"expected={expected[:8]}"
            )
