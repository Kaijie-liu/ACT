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

import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import (
    HZono,
    hz_multiply,
    hz_add_const,
    hz_minkowski_sum,
    hz_from_bounds,
    hz_compute_bounds,
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
    return HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
    )


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
    return HZono(
        c=a_col * hz.c,
        Gc=a_col * hz.Gc,
        Gb=a_col * hz.Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


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
    if hz_in is not None:
        # Precision knob: LP-tight pre-activation [alpha,beta]. ReLU is always
        # encoded EXACTLY (eq_lagr binary) -- no triangle/convex relaxation exists.
        tight = getattr(tf, "_relu_tight_bounds", False)
        tf._hz_cache[L.id] = hz_reduce(
            hz_apply_relu(hz_in, tight_bounds=tight))
    fact = interval.tf_relu(L, bounds)
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
        tf._hz_cache[L.id] = hz_reduce(hz_apply_tanh(hz_in, K=tf._tanh_K))
    fact = interval.tf_tanh(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_sigmoid(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_sigmoid(hz_in, K=tf._sigmoid_K))
    fact = interval.tf_sigmoid(L, bounds)
    if hz_in is not None:
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
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            b1, b2 = hz_compute_bounds(hz_in), hz_compute_bounds(hz2)
            corners = torch.stack(
                [b1.lb * b2.lb, b1.lb * b2.ub, b1.ub * b2.lb, b1.ub * b2.ub]
            )
            tf._hz_cache[L.id] = hz_from_bounds(
                Bounds(lb=corners.min(0)[0], ub=corners.max(0)[0]), dtype, device
            )
        else:
            hz_in = None
    fact = interval.tf_mul(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
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
    tf._hz_cache[L.id] = HZono(
        c=val.view(-1, 1),
        Gc=val.new_zeros(n, 0),
        Gb=val.new_zeros(n, 0),
        Ac=val.new_zeros(0, 0),
        Ab=val.new_zeros(0, 0),
        b=val.new_zeros(0, 1),
    )
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
    tf._hz_cache.pop(L.id, None)
    return interval.tf_matmul(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )


def tf_arg_extremum(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_arg_extremum(L, bounds)


def tf_upsample(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval_cnn.tf_upsample(L, bounds)


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
        dtype, device = hz_in.c.dtype, hz_in.c.device
        tf._hz_cache[L.id] = hz_from_bounds(fact.bounds, dtype, device)
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
    return HZono(c=hz.c, Gc=hz.Gc, Gb=hz.Gb, Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
                 eq_mask=hz.eq_mask, col_ids=hz.col_ids, bcol_ids=hz.bcol_ids)


def _hz_gather_rows(hz: HZono, row_idx: torch.Tensor) -> HZono:
    """Select / permute / repeat HZ output ROWS by ``row_idx`` (n_out,). The
    constraint block (Ac/Ab/b) and ``col_ids`` reference generator COLUMNS, so
    they are unchanged -- a structural op only remaps which output coordinate
    reads which latent row. Exact (the output is an exact gather of inputs)."""
    ri = row_idx.to(device=hz.c.device, dtype=torch.long)
    return HZono(c=hz.c[ri], Gc=hz.Gc[ri], Gb=hz.Gb[ri],
                 Ac=hz.Ac, Ab=hz.Ab, b=hz.b, eq_mask=hz.eq_mask,
                 col_ids=hz.col_ids, bcol_ids=hz.bcol_ids)


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
    encoding -- but on a 28k-wide conv it would cost ~56k LPs (timeout), and such
    wide-conv nets don't certify anyway, so fall back to the fast box there."""
    if not tight or hz.Ac.shape[0] == 0 or hz.c.shape[0] > _RELU_TIGHT_MAX_DIM:
        return hz_compute_bounds(hz)
    try:
        from act.back_end.solver.solver_hz import _hz_compute_bounds_scipy
        tb = _hz_compute_bounds_scipy(hz)
        fb = hz_compute_bounds(hz)
        # Both sound -> take the tighter (intersection). Numerically robust.
        return Bounds(lb=torch.maximum(tb.lb, fb.lb),
                      ub=torch.minimum(tb.ub, fb.ub))
    except Exception:
        return hz_compute_bounds(hz)


def hz_apply_relu(hz: HZono, tight_bounds=False) -> HZono:
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
        return HZono(
            c=out_c, Gc=out_Gc, Gb=out_Gb,
            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
            col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
            bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
        )

    # Exact-only: EVERY unstable neuron gets the exact eq_lagr binary graph.
    # No triangle/convex partition -- the HZ domain is exact by construction.
    exact_sel = unstable_idx
    ke = int(exact_sel.shape[0])

    ng_new = ng + 4 * ke
    nb_new = nb + ke
    out_Gc = hz.c.new_zeros(n, ng_new)
    out_Gb = hz.c.new_zeros(n, nb_new)
    out_c = hz.c.new_zeros(n, 1)
    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    # --- EXACT eq_lagr block on exact_sel (ke neurons) ---
    if ke > 0:
        a_e = lb[exact_sel]
        b_e = ub[exact_sel]
        te = torch.arange(ke, device=device)
        col_xi1 = ng + te
        col_xi2 = ng + ke + te
        col_xi3 = ng + 2 * ke + te
        col_xi4 = ng + 3 * ke + te
        col_z = nb + te
        out_c[exact_sel, 0] = b_e / 2.0
        out_Gc[exact_sel, col_xi2] = -b_e / 2.0
        eq_Ac = hz.c.new_zeros(3 * ke, ng_new)
        eq_Ab = hz.c.new_zeros(3 * ke, nb_new)
        eq_b = hz.c.new_zeros(3 * ke, 1)
        r1 = 3 * te
        r2 = 3 * te + 1
        r3 = 3 * te + 2
        eq_Ac[r1, col_xi1] = 1.0
        eq_Ac[r1, col_xi3] = 1.0
        eq_Ab[r1, col_z] = 1.0
        eq_b[r1, 0] = 1.0
        eq_Ac[r2, col_xi2] = 1.0
        eq_Ac[r2, col_xi4] = 1.0
        eq_Ab[r2, col_z] = -1.0
        eq_b[r2, 0] = 1.0
        eq_Ac[r3, col_xi1] = a_e / 2.0
        eq_Ac[r3, col_xi2] = -b_e / 2.0
        eq_Ac[r3, :ng] = -hz.Gc[exact_sel]
        eq_Ab[r3, :nb] = -hz.Gb[exact_sel]
        eq_Ab[r3, col_z] = a_e / 2.0
        eq_b[r3, 0] = hz.c[exact_sel, 0] - b_e / 2.0
    else:
        eq_Ac = hz.c.new_zeros(0, ng_new)
        eq_Ab = hz.c.new_zeros(0, nb_new)
        eq_b = hz.c.new_zeros(0, 1)

    old_Ac_ext = torch.cat([hz.Ac, hz.c.new_zeros(nc, 4 * ke)], dim=1)
    old_Ab_ext = torch.cat([hz.Ab, hz.c.new_zeros(nc, ke)], dim=1)
    Ac_out = torch.cat([old_Ac_ext, eq_Ac], dim=0)
    Ab_out = torch.cat([old_Ab_ext, eq_Ab], dim=0)
    b_out = torch.cat([hz.b, eq_b], dim=0)
    # eq_mask: original senses + 3*ke equality rows (True). None stays None
    # since the eq_lagr rows are equalities (all-equality semantics preserved).
    if hz.eq_mask is None:
        eq_mask_out = None
    else:
        eq_mask_out = torch.cat(
            [hz.eq_mask.to(device),
             torch.ones(3 * ke, dtype=torch.bool, device=device)])

    new_col_ids, new_bcol_ids = _relu_extend_ids(hz, ke)
    return HZono(
        c=out_c, Gc=out_Gc, Gb=out_Gb,
        Ac=Ac_out, Ab=Ab_out, b=b_out,
        eq_mask=eq_mask_out,
        col_ids=new_col_ids, bcol_ids=new_bcol_ids,
    )


def hz_apply_relu_convex(hz: HZono) -> HZono:
    """Convex (DeepZ/triangle) ReLU over-approximation -- NOT the activation path.

    Used ONLY as the convex max-relaxation inside MaxPool (max(a,b) = b + relu(a-b)),
    where the exact binary encoding explodes the MILP (a 2x2 pool already needs
    ~50 binaries per tiny tile). Each unstable neuron gets ONE fresh generator
    (ng+=1, nb+=0, nc+=0): y in [lam*x, lam*(x-alpha)], lam=beta/(beta-alpha) -- the
    hz2 Thm-2 convex hull, sound over-approximation. The ReLU *activation*
    (``hz_apply_relu``) remains exact-only; this convex form exists solely so the
    structural MaxPool operator stays tractable, and is never used for activations.
    """
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]
    bounds = _relu_preact_bounds(hz, tight=False)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()
    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    uidx = torch.where(unstable)[0]
    kt = int(uidx.shape[0])

    ng_new = ng + kt
    out_Gc = hz.c.new_zeros(n, ng_new)
    out_Gb = hz.c.new_zeros(n, nb)
    out_c = hz.c.new_zeros(n, 1)
    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]
    if kt > 0:
        a_t = lb[uidx]
        b_t = ub[uidx]
        tt = torch.arange(kt, device=hz.c.device)
        lam = b_t / (b_t - a_t)            # upper-chord slope in (0,1)
        half_gap = 0.5 * lam * (-a_t)      # half the vertical gap (>=0)
        out_c[uidx, 0] = lam * hz.c[uidx, 0] + half_gap
        out_Gc[uidx, :ng] = lam.unsqueeze(1) * hz.Gc[uidx]
        out_Gc[uidx, ng + tt] = half_gap
        if nb:
            out_Gb[uidx, :nb] = lam.unsqueeze(1) * hz.Gb[uidx]
    Ac_out = torch.cat([hz.Ac, hz.c.new_zeros(nc, kt)], dim=1)
    new_col_ids = None
    if hz.col_ids is not None:
        from act.back_end.solver.solver_hz import _fresh_col_ids
        new_col_ids = torch.cat(
            [hz.col_ids, _fresh_col_ids(kt, device=hz.col_ids.device)])
    return HZono(
        c=out_c, Gc=out_Gc, Gb=out_Gb,
        Ac=Ac_out, Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=new_col_ids,
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _relu_extend_ids(hz: HZono, k: int):
    """Extend factor ids for the ReLU encoding: the original ng/nb columns keep
    their ids; the 4k new continuous (xi1..xi4) and k new binary (z) columns get
    FRESH ids (they are brand-new latent factors, shared with nothing)."""
    if hz.col_ids is None:
        return None, None
    from act.back_end.solver.solver_hz import _fresh_col_ids
    dev = hz.col_ids.device
    new_col = torch.cat([hz.col_ids, _fresh_col_ids(4 * k, device=dev)])
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
        return HZono(
            c=out_c,
            Gc=out_Gc[:, :ng],
            Gb=out_Gb[:, :nb],
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
            col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
            bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
        )

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
    return HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
        col_ids=new_col_ids,
        bcol_ids=new_bcol_ids,
    )


def hz_apply_piecewise(hz: HZono, func, dfunc, K: int = 2, inflection=None) -> HZono:
    """Piecewise linear approximation for monotone activations (tangent parallelogram).

    ``inflection`` (e.g. 0.0 for sigmoid/tanh): force the inflection point to be a
    SEGMENT BOUNDARY. The tangent parallelogram assumes the function is convex OR
    concave on each segment; a segment straddling the inflection violates that, so
    the soundness slack inflates and tightness becomes ERRATIC in K (toy: K=3/8
    loosen). Splitting K segments on each side of the inflection (=> 2K segments,
    inflection always a boundary) keeps every segment convex/concave so tightness
    is MONOTONE in K."""
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
        return HZono(
            c=new_c,
            Gc=new_Gc_base,
            Gb=new_Gb_base,
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
        )

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
    n_slack = 4 * K * m
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
    ng_total = ng + n_real + n_slack
    nb_total = nb + K * m

    n_box = 4 * K * m
    n_eq_total = n_box + m + m
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

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, n_real + n_slack)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, K * m)], dim=1
    )

    return HZono(
        c=new_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
    )


def hz_apply_sigmoid(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear sigmoid via tangent parallelogram encoding. Sigmoid has
    its inflection at 0 -> split segments there for monotone-in-K tightness."""
    return hz_apply_piecewise(
        hz, torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)), K,
        inflection=0.0
    )


def hz_apply_tanh(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear tanh via tangent parallelogram encoding. Tanh inflection
    at 0 -> split segments there for monotone-in-K tightness."""
    return hz_apply_piecewise(hz, torch.tanh, lambda x: 1 - torch.tanh(x) ** 2, K,
                              inflection=0.0)


# --- HZ order reduction ---


# Absolute size caps above which the LOSSY reduction steps are allowed to fire.
# Toy validation (2026-06-14) showed the lossy steps (binary-relax + Girard) cost
# 80-215% exact-width inflation for only a ~20% generator cut on moderate nets,
# while the EXACT Step-0 redundancy removal is free AND gives a 1.4-2.9x LP
# speedup. So below these caps reduce = Step-0 only (keep precision + the
# speedup); the lossy steps engage only when the HZ is genuinely too large for
# the open-source LP/MILP solver (tractability emergency).
_HZ_REDUCE_NB_CAP = 2048   # relax binaries to continuous only above this
_HZ_REDUCE_NG_CAP = 16384  # Girard-reduce continuous generators only above this


def hz_reduce(hz: HZono, max_order: float = 3.0) -> HZono:
    """Reduce HZ complexity (sound over-approximation), Bird PhD 2022 Ch.6.

    Step 0 (always) is the EXACT redundancy removal (Bird 6.1) -- free, and on
    constrained/residual HZ it cuts nc 57-84% with a 1.4-2.9x LP speedup and
    ZERO precision loss. Step 1 (binary relax, Bird 6.2.4) and Step 2 (lifted
    Girard, Bird 6.2.3) are LOSSY and engage only above absolute size caps,
    because per the toy validation they otherwise just throw away precision on
    moderate nets. All steps carry ``col_ids`` / ``eq_mask`` so downstream
    share-merge (sgm) and constraint senses survive.
    """
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    if n == 0:
        return hz

    # Step 0: EXACT redundancy removal (Bird 6.1 / zonopy). Free -- shrinks
    # ng/nc/nb with ZERO over-approximation, so always worth doing first.
    from act.back_end.hybridz_tf.algorithms.order_reduce import hz_remove_redundancy
    hz = hz_remove_redundancy(hz)
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    # The lossy steps only engage when genuinely huge. Girard's target stays
    # ABOVE the lifted-box floor (n+nc) so it retains real generators rather than
    # collapsing to a pure box (which is the counterproductive case).
    # nc-AWARE: the lossy steps are exact-free at nc==0 but LOOSEN a constrained
    # HZ (toy: Girard +46%, binary-relax +8% on the true set at nc>0), and the
    # scipy engine is blind to the binary-relax loss. So at nc>0 raise the
    # emergency cap -> lossy reduction becomes a genuine last resort there.
    nc_guard = 2 if nc > 0 else 1
    max_nb = max(2 * n, _HZ_REDUCE_NB_CAP * nc_guard)
    max_ng = max(int(max_order * n), n + nc + n, _HZ_REDUCE_NG_CAP * nc_guard)

    # Step 1: Relax excess binary generators to continuous (Bird 6.2.4). The
    # relaxed binary's columns move Gb->Gc and Ab->Ac; ids move with them.
    if nb > max_nb:
        col_norms = hz.Gb.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort()
        n_relax = nb - max_nb
        relax_idx = sorted_idx[:n_relax]
        keep_idx = sorted_idx[n_relax:]
        extra_Gc = hz.Gb[:, relax_idx]
        extra_Ac = (
            hz.Ab[:, relax_idx] if nc > 0 else hz.c.new_zeros(0, n_relax)
        )
        new_col_ids = None
        new_bcol_ids = None
        if hz.col_ids is not None and hz.bcol_ids is not None:
            new_col_ids = torch.cat([hz.col_ids, hz.bcol_ids[relax_idx.to(hz.bcol_ids.device)]])
            new_bcol_ids = hz.bcol_ids[keep_idx.to(hz.bcol_ids.device)]
        hz = HZono(
            c=hz.c,
            Gc=torch.cat([hz.Gc, extra_Gc], dim=1),
            Gb=hz.Gb[:, keep_idx],
            Ac=torch.cat([hz.Ac, extra_Ac], dim=1)
            if nc > 0
            else hz.c.new_zeros(0, ng + n_relax),
            Ab=hz.Ab[:, keep_idx]
            if nc > 0
            else hz.c.new_zeros(0, max_nb),
            b=hz.b.clone(),
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
            col_ids=new_col_ids,
            bcol_ids=new_bcol_ids,
        )
        ng = hz.Gc.shape[1]
        nb = hz.Gb.shape[1]

    # Step 2: Reduce continuous generators via lifted Girard (constraint- and
    # id-preserving). Replaces the old destructive row-dropping reduction.
    if ng > max_ng:
        from act.back_end.hybridz_tf.algorithms.order_reduce import hz_girard_reduce
        hz = hz_girard_reduce(hz, max_ng)

    return hz
