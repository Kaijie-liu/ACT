"""HyZor-compat surface for solver_hyzor.consume_cons.

This module exposes HyZor's function names (``hz_dense``, ``hz_apply_relu_v8``,
etc.) backed by ACT-native implementations whenever possible. Functions
not yet ported delegate to HyZor at runtime, with a single import path
``from act.back_end.hybridz_tf.hyzor_compat import ...``.

Migration plan
--------------
Each function listed below has one of three states:

  ✓ NATIVE   — ACT-native; **parity-tested 0.0 element-wise error**.
  ⚙ WRAPPED  — Thin wrapper around ACT modules (simple composition).
  ◌ LEGACY   — Still calls HyZor at runtime; awaits port.

Phase 6.x status:

  hz_from_bounds            ✓ NATIVE  (re-export solver_hz.hz_from_bounds)
  hz_add_const              ✓ NATIVE  (re-export solver_hz.hz_add_const)
  hz_minkowski_sum          ✓ NATIVE  (re-export solver_hz.hz_minkowski_sum)
  hz_sgm_add                ✓ NATIVE  (re-export algorithms.sgm.hz_sgm_add)
  shares_generator          ✓ NATIVE  (re-export algorithms.sgm.shares_generator)
  check_unsafe_for_act      ✓ NATIVE  (re-export algorithms.lp_verify.check_unsafe_for_act)
  lp_witness_to_input       ✓ NATIVE  (re-export algorithms.lp_verify.lp_witness_to_input)
  hz_apply_sigmoid          ✓ NATIVE  (re-export tf_mlp.hz_apply_sigmoid)
  hz_apply_tanh             ✓ NATIVE  (re-export tf_mlp.hz_apply_tanh)
  hz_conv2d                 ✓ NATIVE  (wraps tf_cnn.hz_conv2d; 5/5 parity)
  hz_maxpool2d              ✓ NATIVE  (algorithms.maxpool; 6/6 parity)
  strict_replay_for_act     ✓ NATIVE  (algorithms.strict_replay; ORT/torch dual path)
  hz_dense                  ⚙ WRAPPED (hz_multiply + hz_add_const)
  hz_scale                  ⚙ WRAPPED (per-channel diag multiply)
  hz_bn                     ⚙ WRAPPED (= hz_scale + hz_add_const)
  hz_concat                 ⚙ WRAPPED (HZono-native; no Lazy/Box routing)
  hz_intersect_polytope     ⚙ WRAPPED (linear constraint add, eq_mask=False)
  hz_apply_relu_v8          ⚙ WRAPPED (dispatch by method to ACT relu_methods)
  hz_apply_leaky_relu_v8    ⚙ WRAPPED (re-export tf_mlp.hz_apply_leaky_relu)

  All entries above are ACT-only at runtime; HyZor pkg deletion blocked
  only on the Phase 1-3 representation routing (BoxHZ / LazyChainHZ /
  SparseGcZ) used by `_from_hyzor` for callers that hand in those
  flavours. cifar/tiny / small-dense paths never produce those types,
  so the conversion bridge is dead code for production benchmarks.
"""
from __future__ import annotations
from typing import Optional, List, Tuple
import os
import sys

import torch

from act.back_end.solver.solver_hz import (
    HZono,
    hz_from_bounds as _act_hz_from_bounds,
    hz_add_const as _act_hz_add_const,
    hz_multiply,
    hz_minkowski_sum as _act_hz_minkowski,
    _eq_mask_of,
)


def hz_from_bounds(bounds, *, dtype=None, device=None):
    """Tiered HZ construction (HyZor's three-tier strategy).

    For large inputs (e.g. cifar/tiny conv root), ACT's
    ``solver_hz.hz_from_bounds`` always emits a dense diagonal Gc which
    can exceed ``n^2 * elem`` bytes — OOM on ResNet. HyZor's
    ``__init__.py:936`` instead picks one of:

      1. SparseGcZ if ``≤ HYZOR_SPARSE_INPUT_THRESHOLD`` pixels have
         ``rad > 0`` (DeepFool-style sparse attack).
      2. BoxHZ if ``n > HYZOR_LARGE_HZ_DIM_CAP``.
      3. Dense HybridZonotope otherwise.

    Phase 6.7 routes this entry point through HyZor's logic; once Phase
    1-3 representations are full ACT-native this will switch.
    """
    _ensure_hyzor_on_path()
    from HyZor import hz_from_bounds as _hyzor_hzfb
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    return _hyzor_hzfb(bounds, **kwargs)


def hz_add_const(hz, v):
    """Translate by constant ``v``.

    Phase 1-3 → delegate to HyZor (preserves type).
    HZono → ACT-native.
    """
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_add_const as _hyzor_addc
        return _hyzor_addc(hz, v)
    return _act_hz_add_const(hz, v)


def hz_minkowski_sum(hz_x, hz_y):
    """Minkowski sum.

    Mixed-type or Phase 1-3 → delegate to HyZor.
    Both HZono → ACT-native.
    """
    if _is_phase13(hz_x) or _is_phase13(hz_y):
        _ensure_hyzor_on_path()
        from HyZor import hz_minkowski_sum as _hyzor_minkowski
        return _hyzor_minkowski(hz_x, hz_y)
    return _act_hz_minkowski(hz_x, hz_y)
from act.back_end.hybridz_tf.algorithms.sgm import (
    shares_generator as _act_shares_generator,
    hz_sgm_add as _act_hz_sgm_add,
)


def shares_generator(hz_x, hz_y) -> bool:
    """True if hz_x and hz_y share input generators (SGM heuristic).

    Phase 1-3 types always return False (HyZor convention — no Gc to
    compare). HZono pair → ACT-native bitwise Gc compare.
    """
    if _is_phase13(hz_x) or _is_phase13(hz_y):
        return False
    return _act_shares_generator(hz_x, hz_y)


def hz_sgm_add(hz_x, hz_y):
    """Shared Generator Merge add. Phase 1-3 falls back to interval add."""
    if _is_phase13(hz_x) or _is_phase13(hz_y):
        _ensure_hyzor_on_path()
        from HyZor import hz_sgm_add as _hyzor_sgm
        return _hyzor_sgm(hz_x, hz_y)
    return _act_hz_sgm_add(hz_x, hz_y)
from act.back_end.hybridz_tf.algorithms.lp_verify import (
    check_unsafe_for_act,
    lp_witness_to_input,
)
from act.back_end.hybridz_tf.algorithms.relu_methods import (
    hz_apply_relu_triangle,
    hz_apply_relu_compact,
    hz_apply_relu_bigM_fast,
)
from act.back_end.hybridz_tf.tf_mlp import (
    hz_apply_relu,
    hz_apply_leaky_relu,
    hz_apply_sigmoid,
    hz_apply_tanh,
)


# ============================================================================
# Native re-exports (already parity-tested)
# ============================================================================

__all__ = [
    "hz_from_bounds",
    "hz_add_const",
    "hz_minkowski_sum",
    "hz_sgm_add",
    "shares_generator",
    "check_unsafe_for_act",
    "lp_witness_to_input",
    "hz_apply_sigmoid",
    "hz_apply_tanh",
    "hz_apply_relu",
    "hz_apply_relu_triangle",
    "hz_apply_relu_compact",
    "hz_apply_relu_bigM_fast",
    "hz_apply_leaky_relu",
    # Wrapped + legacy ports below.
    "hz_dense",
    "hz_scale",
    "hz_bn",
    "hz_concat",
    "hz_intersect_polytope",
    "hz_apply_relu_v8",
    "hz_apply_leaky_relu_v8",
    "hz_conv2d",
    "hz_maxpool2d",
    "strict_replay_for_act",
]


# ============================================================================
# Wrapped: linear ops
# ============================================================================


def _is_phase13(hz) -> bool:
    """True if hz is a HyZor type (BoxHZ / LazyChainHZ / SparseGcZ /
    HybridZonotope). We dispatch these to HyZor's own ops to preserve
    Phase 1-3 routing — converting back-and-forth between HyZor classes
    and HZono breaks memory routing AND tries to call HZono on HyZor
    methods like ``_bounds_unconstrained`` that don't exist there.

    Rule: once a layer's HZ is a HyZor class, the rest of the chain
    stays HyZor. ACT path only runs when input HZ is HZono throughout.
    """
    cls = type(hz).__name__
    return cls in ("BoxHZ", "LazyChainHZ", "SparseGcZ", "HybridZonotope")


def hz_dense(hz, W, b=None):
    """``y = W x + b``.

    HZono input  → ACT-native (hz_multiply + hz_add_const).
    BoxHZ / LazyChainHZ / SparseGcZ → delegate to HyZor.hz_dense
    (Phase 1-3 routing preserves memory; ACT port pending).
    """
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_dense as _hyzor_dense
        return _hyzor_dense(hz, W, b)

    W_t = torch.as_tensor(W, dtype=hz.c.dtype, device=hz.c.device)
    out = hz_multiply(hz, W_t)
    if b is not None:
        b_t = torch.as_tensor(b, dtype=hz.c.dtype, device=hz.c.device).flatten()
        out = hz_add_const(out, b_t)
    return out


def hz_scale(hz, a):
    """``y = a ⊙ x`` (per-channel scaling).

    HZono input → ACT-native diagonal multiply.
    Phase 1-3 types → delegate to HyZor.hz_scale.
    """
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_scale as _hyzor_scale
        return _hyzor_scale(hz, a)

    a_t = torch.as_tensor(a, dtype=hz.c.dtype, device=hz.c.device).flatten()
    n = int(hz.c.shape[0])
    if a_t.numel() == 1:
        a_t = a_t.expand(n)
    if a_t.numel() != n:
        raise ValueError(f"hz_scale: size mismatch {a_t.numel()} vs {n}")
    scale_col = a_t.view(-1, 1)
    return HZono(
        c=hz.c * scale_col,
        Gc=hz.Gc * scale_col,
        Gb=hz.Gb * scale_col,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
    )


def hz_bn(hz, A, c):
    """``y = A ⊙ x + c`` (batch-norm fused affine)."""
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_bn as _hyzor_bn
        return _hyzor_bn(hz, A, c)
    return hz_add_const(hz_scale(hz, A), c)


def hz_intersect_polytope(hz, A, b):
    """Intersect with ``A x ≤ b`` halfspace polytope.

    Phase 1-3 types → delegate to HyZor (its impl freezes / densifies
    appropriately before adding LP rows).
    HZono → ACT-native row append with eq_mask=False.
    """
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_intersect_polytope as _hyzor_ip
        return _hyzor_ip(hz, A, b)

    A_t = torch.as_tensor(A, dtype=hz.c.dtype, device=hz.c.device)
    if A_t.dim() == 1:
        A_t = A_t.view(1, -1)
    b_t = torch.as_tensor(b, dtype=hz.c.dtype, device=hz.c.device).view(-1, 1)
    new_rows = int(A_t.shape[0])

    Ac_add = A_t @ hz.Gc
    Ab_add = A_t @ hz.Gb if int(hz.Gb.shape[1]) > 0 else torch.zeros(
        (new_rows, 0), dtype=hz.c.dtype, device=hz.c.device,
    )
    b_add = b_t - A_t @ hz.c

    new_Ac = torch.cat([hz.Ac, Ac_add], dim=0)
    new_Ab = torch.cat([hz.Ab, Ab_add], dim=0)
    new_b = torch.cat([hz.b, b_add], dim=0)

    em_old = _eq_mask_of(hz)
    em_new = torch.cat([
        em_old,
        torch.zeros(new_rows, dtype=torch.bool, device=hz.c.device),
    ])

    return HZono(
        c=hz.c.clone(), Gc=hz.Gc.clone(), Gb=hz.Gb.clone(),
        Ac=new_Ac, Ab=new_Ab, b=new_b, eq_mask=em_new,
    )


def hz_concat(hz_list):
    """Concatenate HZ instances along dim 0.

    Mixed-type inputs (any element is Phase 1-3) → delegate to HyZor
    (its `_coerce_for_combine` handles type unification).
    All HZono → ACT-native block-diagonal layout.
    """
    hz_list = list(hz_list)
    if any(_is_phase13(h) for h in hz_list):
        _ensure_hyzor_on_path()
        from HyZor import hz_concat as _hyzor_concat
        return _hyzor_concat(hz_list)
    if len(hz_list) == 1:
        return hz_list[0]
    if len(hz_list) == 0:
        raise ValueError("hz_concat: empty input list")

    dtype = hz_list[0].c.dtype
    device = hz_list[0].c.device
    # Validate consistent dtype/device.
    for hz in hz_list[1:]:
        if hz.c.dtype != dtype or hz.c.device != device:
            raise ValueError(
                f"hz_concat: dtype/device mismatch "
                f"({hz.c.dtype}/{hz.c.device} vs {dtype}/{device})"
            )

    # Concat coords c, vertically stack Gc and Gb as block-diagonal.
    cs = [hz.c for hz in hz_list]
    Gcs = [hz.Gc for hz in hz_list]
    Gbs = [hz.Gb for hz in hz_list]
    Acs = [hz.Ac for hz in hz_list]
    Abs = [hz.Ab for hz in hz_list]
    bs = [hz.b for hz in hz_list]
    em_list = [_eq_mask_of(hz) for hz in hz_list]

    c_new = torch.cat(cs, dim=0)
    # Block-diagonal Gc / Gb across factor axes.
    Gc_blocks = []
    Gb_blocks = []
    Ac_blocks = []
    Ab_blocks = []
    n_total = sum(int(c.shape[0]) for c in cs)
    ng_offsets = [0]
    nb_offsets = [0]
    for Gc, Gb in zip(Gcs, Gbs):
        ng_offsets.append(ng_offsets[-1] + int(Gc.shape[1]))
        nb_offsets.append(nb_offsets[-1] + int(Gb.shape[1]))
    ng_total = ng_offsets[-1]
    nb_total = nb_offsets[-1]

    # Build Gc and Gb as block-diagonal across COLUMNS and stacked across
    # ROWS — same total dim n_total. Each input contributes its rows;
    # zero-pad to align cols.
    Gc_full = torch.zeros((n_total, ng_total), dtype=dtype, device=device)
    Gb_full = torch.zeros((n_total, nb_total), dtype=dtype, device=device)
    row_off = 0
    for i, hz in enumerate(hz_list):
        ni = int(hz.c.shape[0])
        ngi = int(hz.Gc.shape[1])
        nbi = int(hz.Gb.shape[1])
        if ngi > 0:
            Gc_full[row_off:row_off + ni, ng_offsets[i]:ng_offsets[i + 1]] = hz.Gc
        if nbi > 0:
            Gb_full[row_off:row_off + ni, nb_offsets[i]:nb_offsets[i + 1]] = hz.Gb
        row_off += ni

    # Stack constraint blocks; same block-diag pattern.
    nc_total = sum(int(b.shape[0]) for b in bs)
    Ac_full = torch.zeros((nc_total, ng_total), dtype=dtype, device=device)
    Ab_full = torch.zeros((nc_total, nb_total), dtype=dtype, device=device)
    b_full = torch.zeros((nc_total, 1), dtype=dtype, device=device)
    em_full = torch.zeros(nc_total, dtype=torch.bool, device=device)
    row_c = 0
    for i, hz in enumerate(hz_list):
        nci = int(hz.b.shape[0])
        ngi = int(hz.Gc.shape[1])
        nbi = int(hz.Gb.shape[1])
        if nci > 0:
            if ngi > 0:
                Ac_full[row_c:row_c + nci, ng_offsets[i]:ng_offsets[i + 1]] = hz.Ac
            if nbi > 0:
                Ab_full[row_c:row_c + nci, nb_offsets[i]:nb_offsets[i + 1]] = hz.Ab
            b_full[row_c:row_c + nci, :] = hz.b
            em_full[row_c:row_c + nci] = em_list[i]
        row_c += nci

    return HZono(
        c=c_new, Gc=Gc_full, Gb=Gb_full,
        Ac=Ac_full, Ab=Ab_full, b=b_full,
        eq_mask=em_full,
    )


# ============================================================================
# Wrapped: ReLU dispatcher (HyZor hz_apply_relu_v8 signature)
# ============================================================================


def hz_apply_relu_v8(hz, *, method: str = "eq_lagr_v8",
                     mace: bool = True, girard_cap: int = 6000):
    """Apply ReLU encoding by method name (HyZor's facade).

    Phase 1-3 types → delegate to HyZor.hz_apply_relu_v8 (which handles
    BoxHZ IBP-clamp, LazyChainHZ freeze, SparseGcZ triangle).
    HZono → dispatch to ACT method-specific encoding.

    HZono method mapping:
      ``eq_lagr_v8`` / ``eq_native`` / ``exact``    → ``hz_apply_relu``
      ``triangle``                                  → ``hz_apply_relu_triangle``
      ``compact``                                   → ``hz_apply_relu_compact``
      ``bigM`` / ``bigM_fast`` / ``exact_box``      → ``hz_apply_relu_bigM_fast``

    ``mace`` and ``girard_cap`` parameters are accepted for signature
    compatibility but are not currently used (Girard reduction lives
    in the solver-side ``_maybe_reduce``).
    """
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_apply_relu_v8 as _hyzor_relu_v8
        return _hyzor_relu_v8(hz, method=method, mace=mace, girard_cap=girard_cap)

    if method in ("eq_lagr_v8", "eq_native", "exact"):
        return hz_apply_relu(hz)
    if method == "triangle":
        return hz_apply_relu_triangle(hz)
    if method == "compact":
        return hz_apply_relu_compact(hz)
    if method in ("bigM", "bigM_fast", "exact_box"):
        return hz_apply_relu_bigM_fast(hz)
    raise ValueError(f"hz_apply_relu_v8: unsupported method={method!r}")


def hz_apply_leaky_relu_v8(hz, alpha: float):
    """Leaky ReLU with slope α ∈ [0, 1].

    Phase 1-3 types → delegate to HyZor; HZono → ACT-native.
    """
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_apply_leaky_relu_v8 as _hyzor_lrelu
        return _hyzor_lrelu(hz, alpha)
    return hz_apply_leaky_relu(hz, alpha)


# ============================================================================
# HZono <-> HybridZonotope conversion bridge
# ============================================================================
#
# ACT-native paths consume HZono. HyZor-delegated paths (hz_conv2d /
# hz_maxpool2d / Phase 1-3 representation routing) consume HyZor's
# HybridZonotope (and may return BoxHZ / LazyChainHZ / SparseGcZ). The
# helpers below convert at the boundary so that solver_hyzor's
# consume_cons sees a single carrier type (HZono) throughout while still
# delegating the not-yet-ported ops to HyZor.


_HYZOR_ROOT_DEFAULT = "/data1/Kane/HyZor"


def _ensure_hyzor_on_path() -> None:
    root = os.environ.get("HYZOR_ROOT", _HYZOR_ROOT_DEFAULT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _to_hyzor(hz):
    """Convert HZono (or HyZor type) to HyZor's HybridZonotope.

    Pass-through for any HyZor type. For HZono, materialises a
    HybridZonotope sharing the same tensors. eq_mask is filled with
    ones (all-eq) when missing — matches HZono's None=all-eq default.
    """
    _ensure_hyzor_on_path()
    from HybridZonotope import HybridZonotope
    if isinstance(hz, HybridZonotope):
        return hz
    em = hz.eq_mask
    if em is None and int(hz.b.shape[0]) > 0:
        em = torch.ones(int(hz.b.shape[0]), dtype=torch.bool, device=hz.c.device)
    return HybridZonotope(
        Gc=hz.Gc, Gb=hz.Gb, c=hz.c,
        Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
        device=hz.c.device, dtype=hz.c.dtype,
        eq_mask=em,
    )


def _from_hyzor(hz):
    """Convert HyZor return type back to HZono.

    Handles HybridZonotope directly. BoxHZ / LazyChainHZ / SparseGcZ
    are first reduced to a full HZ via their freeze / densify / to_hz
    methods, then wrapped as HZono. This is a no-op for cifar/tiny
    workloads where Phase 1-3 never triggers; VGG-scale paths pay
    a freeze cost.
    """
    if isinstance(hz, HZono):
        return hz
    cls = type(hz).__name__
    if cls == "BoxHZ":
        # BoxHZ stores lb/ub. Build a single-generator HZono.
        from act.back_end.core import Bounds
        return hz_from_bounds(
            Bounds(lb=hz.lb, ub=hz.ub),
            dtype=hz.dtype, device=hz.device,
        )
    if cls == "LazyChainHZ":
        # Freeze to full HZ then convert.
        if hasattr(hz, "freeze"):
            hz = hz.freeze()
        elif hasattr(hz, "to_full_hz"):
            hz = hz.to_full_hz()
    if cls == "SparseGcZ":
        hz = hz.to_dense_hz()
    # HybridZonotope or any duck-typed 6-tuple object.
    em = getattr(hz, "eq_mask", None)
    return HZono(
        c=hz.c, Gc=hz.Gc, Gb=hz.Gb,
        Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
        eq_mask=em,
    )


# ============================================================================
# Legacy: still delegate to HyZor (port pending), but convert at boundary
# ============================================================================


def hz_conv2d(hz, weight, bias=None, *, input_shape,
              stride=1, padding=0, dilation=1, groups=1):
    """2D conv.

    Phase 1-3 types → delegate to HyZor.hz_conv2d (its Phase 1-3 routing
    preserves memory by keeping BoxHZ/LazyChain/Sparse as-is). Densifying
    to HZono here would OOM on cifar/tiny ResNet (v108 confirmed).

    HZono → ACT-native (tf_cnn.hz_conv2d, parity-tested 5/5 with
    0.0e+00 element-wise error).
    """
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_conv2d as _hyzor_conv
        return _hyzor_conv(
            hz, weight, bias,
            input_shape=input_shape, stride=stride, padding=padding,
            dilation=dilation, groups=groups,
        )

    from act.back_end.hybridz_tf.tf_cnn import hz_conv2d as _act_hz_conv2d

    # Normalise stride/padding/dilation to tuple-2.
    def _t2(v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        v = int(v)
        return (v, v)

    weight_t = torch.as_tensor(weight, dtype=hz.c.dtype, device=hz.c.device)
    bias_t = None
    if bias is not None:
        bias_t = torch.as_tensor(bias, dtype=hz.c.dtype, device=hz.c.device).flatten()
        # ACT's per-spatial broadcast collapse: if bias has more elems than
        # output channels, take the channel slice (mirrors HyZor at
        # __init__.py:1033).
        C_out = int(weight_t.shape[0])
        if bias_t.numel() != C_out:
            try:
                spatial = bias_t.numel() // C_out
                if C_out * spatial == bias_t.numel():
                    bias_t = bias_t.view(C_out, spatial)[:, 0]
            except Exception:
                pass

    return _act_hz_conv2d(
        hz, weight_t, bias_t,
        stride=_t2(stride), padding=padding,
        dilation=_t2(dilation), groups=int(groups),
        input_shape=input_shape,
    )


def hz_maxpool2d(hz, *, kernel_size, stride=None, padding=0, input_shape):
    """MaxPool 2D.

    Phase 1-3 types → delegate to HyZor (its BoxHZ branch uses
    F.max_pool2d on lb/ub directly, very cheap).
    HZono → ACT-native (algorithms.maxpool; 6/6 parity tests with
    0.0e+00 element-wise error).
    """
    if _is_phase13(hz):
        _ensure_hyzor_on_path()
        from HyZor import hz_maxpool2d as _hyzor_mp
        return _hyzor_mp(
            hz, kernel_size=kernel_size, stride=stride,
            padding=padding, input_shape=input_shape,
        )
    from act.back_end.hybridz_tf.algorithms.maxpool import (
        hz_maxpool2d as _act_maxpool,
    )
    return _act_maxpool(
        hz, kernel_size=kernel_size, stride=stride,
        padding=padding, input_shape=input_shape,
    )


def strict_replay_for_act(*, net, x_star, assert_layer) -> bool:
    """Strict zero-tol witness replay. ACT-native port of HyZor's
    ``strict_replay_for_act``. Lives in
    ``algorithms.strict_replay`` — see that module for details.
    """
    from act.back_end.hybridz_tf.algorithms.strict_replay import (
        strict_replay_for_act as _act_strict,
    )
    return _act_strict(net=net, x_star=x_star, assert_layer=assert_layer)
