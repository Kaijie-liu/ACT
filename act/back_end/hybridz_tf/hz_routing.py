#===- act/back_end/hybridz_tf/hz_routing.py - HZ 4-Way Routing Dispatch -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Routes HZ ops to ACT-native dispatch by flavour (HZono / BoxHZ /
#   LazyChainHZ / SparseGcZ). HZono uses dense ops in tf_mlp / tf_cnn /
#   algorithms; lazy/sparse variants defer materialisation.
#
#===---------------------------------------------------------------------===#

"""ACT-native Phase 1-3 routing layer (Y2 Stage 3b).

This is the ACT equivalent of HyZor's ``__init__.py`` top-level
routing functions (``hz_from_bounds`` / ``hz_dense`` / ``hz_conv2d`` /
``hz_apply_relu_v8`` / ``hz_intersect_polytope`` / etc.). Each function
dispatches by HZ flavour:

  - ``SparseGcZ``    -> sparse-on-sparse op when feasible
  - ``LazyChainHZ``  -> extend chain (no allocation)
  - ``BoxHZ``        -> IBP (interval propagation)
  - ``HZono``        -> ACT-native dense ops (tf_mlp / tf_cnn / algorithms)

All four classes live in this package
(``representations.py`` for BoxHZ/Lazy/Sparse; ``solver_hz.HZono``).
No HyZor pkg imports — this module is part of the Y2 self-contained ACT path.

Faithful behaviour of HyZor ``__init__.py`` routing (lines 936-1470):
each function preserves the same isinstance dispatch and the same
output-shape conventions.
"""
from __future__ import annotations
from typing import List, Optional
import os

import torch

from act.back_end.solver.solver_hz import (
    HZono,
    hz_multiply,
    hz_add_const as _hz_add_const_native,
    hz_minkowski_sum as _hz_minkowski_native,
    _eq_mask_of,
    _propagate_base,
)


def _propagate_base_any(parent, child):
    """Propagate _base_ng/_base_nb from parent to child if both are HZono.

    Tolerant of non-HZono shapes (BoxHZ/Sparse/Lazy) so we can call this
    unconditionally at the end of ops with 4-way dispatch.
    """
    if isinstance(parent, HZono) and isinstance(child, HZono):
        _propagate_base(parent, child)
    return child


def _propagate_base_multi(parents, child):
    """min() propagation from multiple parents (concat / sgm / minkowski)."""
    if not isinstance(child, HZono):
        return child
    hz_parents = [p for p in parents if isinstance(p, HZono)]
    if not hz_parents:
        return child
    base_ng = min(int(getattr(p, "_base_ng", int(p.Gc.shape[1])))
                  for p in hz_parents)
    base_nb = min(int(getattr(p, "_base_nb", int(p.Gb.shape[1])))
                  for p in hz_parents)
    object.__setattr__(child, "_base_ng",
                       int(min(base_ng, int(child.Gc.shape[1]))))
    object.__setattr__(child, "_base_nb",
                       int(min(base_nb, int(child.Gb.shape[1]))))
    return child
from act.back_end.hybridz_tf.representations import (
    BoxHZ, LazyChainHZ, SparseGcZ,
    _make_or_box, _to_box, _bounds_of,
    _enable_lazy_chain, _enable_sparse_gc,
    _sparse_gc_budget_bytes,
)
from act.back_end.hybridz_tf.algorithms.sgm import (
    shares_generator as _act_shares_generator,
    hz_sgm_add as _act_hz_sgm_add,
)
from act.back_end.hybridz_tf.tf_mlp import (
    hz_apply_relu, hz_apply_leaky_relu,
    hz_apply_sigmoid, hz_apply_tanh,
)
from act.back_end.hybridz_tf.tf_cnn import (
    hz_maxpool2d as _act_maxpool_hzono,
)
from act.back_end.hybridz_tf.algorithms.relu_methods import (
    hz_apply_relu_triangle, hz_apply_relu_compact, hz_apply_relu_bigM_fast,
)


# ---------------------------------------------------------------------------
# Env knobs
# ---------------------------------------------------------------------------


def _sparse_input_threshold() -> int:
    """Max alive-pixel count to route directly to SparseGcZ at construction."""
    return int(os.environ.get("HYZOR_SPARSE_INPUT_THRESHOLD", "0"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_for_combine(z):
    """For multi-input ops: freeze LazyChainHZ; densify SparseGcZ; HZono and
    BoxHZ pass through."""
    if isinstance(z, LazyChainHZ):
        return z.freeze()
    if isinstance(z, SparseGcZ):
        return z.to_dense_hz()
    return z


# ---------------------------------------------------------------------------
# Tier-1: hz_from_bounds
# ---------------------------------------------------------------------------


def hz_from_bounds(bounds, *, dtype=torch.float64, device="cpu"):
    """Build axis-aligned HZ from box bounds with three-tier strategy.

      1. SparseGcZ if ≤ HYZOR_SPARSE_INPUT_THRESHOLD pixels have rad > 0
         (default 0 = DISABLED; sparse-input path has open LP-interaction issue)
      2. BoxHZ if dim > HYZOR_LARGE_HZ_DIM_CAP (default 8192)
      3. Dense HZono otherwise

    Faithful port of HyZor ``hz_from_bounds`` (__init__.py:936).
    """
    lb = bounds.lb.flatten().to(dtype=dtype, device=device)
    ub = bounds.ub.flatten().to(dtype=dtype, device=device)
    n = int(lb.numel())
    rad = ((ub - lb) / 2.0).abs()

    if _enable_sparse_gc():
        alive_mask = rad > 0
        n_alive = int(alive_mask.sum().item())
        if 0 < n_alive <= _sparse_input_threshold() and n_alive < n:
            c = ((lb + ub) / 2.0)
            alive_idx = torch.nonzero(alive_mask, as_tuple=False).view(-1)
            cols = torch.arange(n_alive, dtype=torch.long, device=device)
            ind = torch.stack([alive_idx, cols])
            vals = rad[alive_idx]
            Gc_sparse = torch.sparse_coo_tensor(
                ind, vals, (n, n_alive),
                dtype=dtype, device=device,
            ).coalesce()
            return SparseGcZ(c=c, Gc_sparse=Gc_sparse,
                             dtype=dtype, device=device)

    return _make_or_box(lb, ub, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Affine ops
# ---------------------------------------------------------------------------


def hz_dense(hz, W, b=None):
    """``y = W x + b``. 4-way dispatch faithful to HyZor (__init__.py:974).

      - SparseGcZ → ``hz.apply_dense(W, b)`` (densifies, returns HZono)
      - LazyChainHZ → ``hz.with_dense(W, b)`` (extends chain)
      - BoxHZ → IBP snapshot (rebuilds via _make_or_box)
      - HZono → ``hz_multiply(hz, W) + hz_add_const(b)``
    """
    W_t = torch.as_tensor(W, dtype=hz.dtype if hasattr(hz, "dtype") else hz.c.dtype,
                          device=hz.device if hasattr(hz, "device") else hz.c.device)
    b_t = (torch.as_tensor(b, dtype=W_t.dtype, device=W_t.device).flatten()
           if b is not None else None)

    if isinstance(hz, SparseGcZ):
        return hz.apply_dense(W_t, b_t)
    if isinstance(hz, LazyChainHZ):
        return hz.with_dense(W_t, b_t)
    if isinstance(hz, BoxHZ):
        if _enable_lazy_chain():
            return LazyChainHZ.from_box(hz).with_dense(W_t, b_t)
        # IBP snapshot.
        c_mid = (hz.lb + hz.ub) / 2.0
        rad = ((hz.ub - hz.lb) / 2.0).abs()
        c_out = W_t @ c_mid
        rad_out = W_t.abs() @ rad
        if b_t is not None:
            c_out = c_out + b_t
        return _make_or_box(c_out - rad_out, c_out + rad_out,
                            dtype=hz.dtype, device=hz.device)
    # HZono
    out = hz_multiply(hz, W_t)
    if b_t is not None:
        out = _hz_add_const_native(out, b_t)
    return _propagate_base_any(hz, out)


def hz_scale(hz, a):
    """``y = a ⊙ x``. 4-way dispatch faithful to HyZor (__init__.py:1142)."""
    a_t = torch.as_tensor(a, dtype=hz.dtype if hasattr(hz, "dtype") else hz.c.dtype,
                          device=hz.device if hasattr(hz, "device") else hz.c.device).flatten()
    if isinstance(hz, LazyChainHZ):
        return hz.with_scale(a_t)
    if isinstance(hz, SparseGcZ):
        return hz.apply_scale(a_t)
    if isinstance(hz, BoxHZ):
        a_pos = a_t.clamp(min=0); a_neg = a_t.clamp(max=0)
        lb_out = a_pos * hz.lb + a_neg * hz.ub
        ub_out = a_pos * hz.ub + a_neg * hz.lb
        return _make_or_box(lb_out, ub_out, dtype=hz.dtype, device=hz.device)
    # HZono
    n = int(hz.c.shape[0])
    if a_t.numel() == 1:
        a_t = a_t.expand(n)
    if a_t.numel() != n:
        raise ValueError(f"hz_scale: size mismatch {a_t.numel()} vs {n}")
    scale_col = a_t.view(-1, 1)
    out = HZono(
        c=hz.c * scale_col, Gc=hz.Gc * scale_col, Gb=hz.Gb * scale_col,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
    )
    return _propagate_base_any(hz, out)


def hz_add_const(hz, c):
    """``y = x + c``. 4-way dispatch faithful to HyZor (__init__.py:1131)."""
    if isinstance(hz, BoxHZ):
        c_t = torch.as_tensor(c, dtype=hz.dtype, device=hz.device).flatten()
        return BoxHZ(hz.lb + c_t, hz.ub + c_t, dtype=hz.dtype, device=hz.device)
    if isinstance(hz, LazyChainHZ):
        c_t = torch.as_tensor(c, dtype=hz.dtype, device=hz.device).flatten()
        return hz.with_add_const(c_t)
    if isinstance(hz, SparseGcZ):
        c_t = torch.as_tensor(c, dtype=hz.dtype, device=hz.device).flatten()
        return SparseGcZ(c=hz.c + c_t, Gc_sparse=hz.Gc_sparse,
                         dtype=hz.dtype, device=hz.device)
    return _propagate_base_any(hz, _hz_add_const_native(hz, c))


def hz_bn(hz, A, c):
    """``y = A ⊙ x + c`` (batch-norm fused affine)."""
    return hz_add_const(hz_scale(hz, A), c)


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------


def hz_conv2d(hz, weight, bias=None, *, input_shape,
              stride=1, padding=0, dilation=1, groups=1):
    """2D conv. 4-way dispatch faithful to HyZor (__init__.py:1007)."""
    if isinstance(stride, (list, tuple)):
        stride = tuple(int(s) for s in stride)
    else:
        stride = (int(stride), int(stride))
    if isinstance(padding, (list, tuple)):
        pad_use = max(int(p) for p in padding)
    else:
        pad_use = int(padding)

    if len(input_shape) == 4:
        _, C, H, W_ = input_shape
    elif len(input_shape) == 3:
        C, H, W_ = input_shape
    else:
        raise ValueError(f"hz_conv2d: bad input_shape {input_shape}")

    dtype = hz.dtype if hasattr(hz, "dtype") else hz.c.dtype
    device = hz.device if hasattr(hz, "device") else hz.c.device

    weight_t = torch.as_tensor(weight, dtype=dtype, device=device)
    Cout = int(weight_t.shape[0])
    if bias is None:
        bias_t = torch.zeros(Cout, dtype=dtype, device=device)
    else:
        bias_t = torch.as_tensor(bias, dtype=dtype, device=device).flatten()
        if bias_t.numel() != Cout:
            try:
                spatial = bias_t.numel() // Cout
                if Cout * spatial == bias_t.numel():
                    bias_t = bias_t.view(Cout, spatial)[:, 0]
            except Exception:
                pass

    if isinstance(hz, SparseGcZ):
        try:
            return hz.apply_conv(weight_t, bias_t, (C, H, W_), stride, pad_use)
        except (MemoryError, RuntimeError) as e:
            msg = str(e).lower()
            if "memory" not in msg and "alloc" not in msg:
                raise
            hz = hz.to_dense_hz()  # density fallback

    if isinstance(hz, LazyChainHZ):
        # Predict output dim and decide if extending chain stays affordable.
        import torch.nn.functional as F
        Hout = (H + 2 * pad_use - weight_t.shape[2]) // stride[0] + 1
        Wout = (W_ + 2 * pad_use - weight_t.shape[3]) // stride[1] + 1
        out_dim = Cout * Hout * Wout
        return hz.with_conv(weight_t, bias_t, stride, pad_use,
                            (C, H, W_), (Cout, Hout, Wout), out_dim)

    if isinstance(hz, BoxHZ):
        # HyZor parity (__init__.py:1063): upgrade BoxHZ → LazyChainHZ so
        # input-pixel correlation is preserved through conv. Plain IBP
        # collapses correlation (output ng = output dim, MUCH bigger
        # than input ng), breaking downstream Girard caps and adding
        # spurious looseness. Only fall back to IBP if lazy-chain
        # disabled by env knob.
        if _enable_lazy_chain():
            kH, kW = int(weight_t.shape[2]), int(weight_t.shape[3])
            out_H = (H + 2 * pad_use - kH) // stride[0] + 1
            out_W = (W_ + 2 * pad_use - kW) // stride[1] + 1
            out_dim = Cout * out_H * out_W
            return LazyChainHZ.from_box(hz).with_conv(
                weight_t, bias_t, stride, pad_use,
                (C, H, W_), (Cout, out_H, out_W), out_dim)
        # Lazy-chain disabled: plain IBP fallback (sound but loose).
        import torch.nn.functional as F
        c_mid = (hz.lb + hz.ub) / 2.0
        rad = ((hz.ub - hz.lb) / 2.0).abs()
        c4 = c_mid.view(1, C, H, W_)
        r4 = rad.view(1, C, H, W_)
        out_c = F.conv2d(c4, weight_t, bias_t,
                          stride=stride, padding=pad_use).view(-1)
        out_r = F.conv2d(r4, weight_t.abs(), None,
                          stride=stride, padding=pad_use).view(-1)
        return _make_or_box(out_c - out_r, out_c + out_r,
                            dtype=dtype, device=device)

    # HZono path: use ACT tf_cnn.hz_conv2d
    from act.back_end.hybridz_tf.tf_cnn import hz_conv2d as _act_conv2d_inner
    out = _act_conv2d_inner(
        hz, weight_t, bias_t,
        stride=stride, padding=pad_use,
        dilation=(int(dilation), int(dilation)) if not isinstance(dilation, tuple) else dilation,
        groups=int(groups), input_shape=input_shape,
    )
    return _propagate_base_any(hz, out)


# ---------------------------------------------------------------------------
# ReLU (eq_lagr_v8 / triangle / compact / bigM)
# ---------------------------------------------------------------------------


def _box_fallback_relu(hz):
    """Sound IBP-clamp ReLU; output reshaped via _make_or_box."""
    lb, ub = _bounds_of(hz)
    relu_lb = torch.clamp(lb, min=0.0)
    relu_ub = torch.clamp(ub, min=0.0)
    dtype = hz.dtype if hasattr(hz, "dtype") else hz.c.dtype
    device = hz.device if hasattr(hz, "device") else hz.c.device
    return _make_or_box(relu_lb, relu_ub, dtype=dtype, device=device)


def _hzono_tight_bounds(hz):
    """Compute pre-ReLU tight bounds for an HZono. Mirrors HyZor's
    cascade in ``HybridZReLU.forward`` for ``eq_lagr_v8`` (HybridZVerifier.py:349-377):

      eq_mask any True  → hz_bounds_eq_elim_lp (QR-reduced LP)  ── Tier 3
      nc > 0 no eq      → hz_bounds_hz_dual    (Adam Lagrangian) ── Tier 2
      else              → hz_bounds_unconstrained                ── Tier 1

    Returns ``(lb, ub)`` flat tensors. All tiers are sound; later tiers
    are no looser than earlier ones.
    """
    from act.back_end.hybridz_tf.algorithms.bounds_tighten import (
        hz_bounds_unconstrained, hz_bounds_eq_elim_lp, hz_bounds_hz_dual,
    )
    em = _eq_mask_of(hz) if int(hz.b.shape[0]) > 0 else None
    if em is not None and bool(em.any().item()):
        try:
            lb, ub = hz_bounds_eq_elim_lp(hz)
            return lb.flatten(), ub.flatten()
        except Exception:
            pass
    if int(hz.b.shape[0]) > 0:
        try:
            # HyZor uses max_iter=min(100, max(20, nc)) for nc>0 no-eq tier
            # (HybridZVerifier.py:371-374).
            nc = int(hz.b.shape[0])
            mi = min(100, max(20, nc))
            lb, ub = hz_bounds_hz_dual(
                hz, max_iter=mi, lr=0.1,
                selective_lp=True, unconstrained_lambda=False,
            )
            return lb.flatten(), ub.flatten()
        except Exception:
            pass
    lb, ub = hz_bounds_unconstrained(hz)
    return lb.flatten(), ub.flatten()


def hz_apply_relu_v8(hz, *, method: str = "eq_lagr_v8"):
    """HZ ReLU dispatch by representation flavor.

    Dispatch:
      LazyChainHZ → freeze if affordable; else triangle on sparse-Gc; else BoxHZ
      SparseGcZ → ``apply_relu_triangle()``
      BoxHZ → IBP clamp
      HZono → method-specific ACT encoding (eq_lagr_v8 / triangle / compact / bigM)
    """
    if isinstance(hz, LazyChainHZ):
        if hz.can_materialize():
            hz = hz.to_full_hz()  # full HZono, fall through to method-dispatch
        else:
            sparse = hz.to_sparse_gc_hz()
            if sparse is not None:
                return sparse.apply_relu_triangle()
            box = hz.snapshot_to_box()
            relu_lb = torch.clamp(box.lb, min=0.0)
            relu_ub = torch.clamp(box.ub, min=0.0)
            return _make_or_box(relu_lb, relu_ub,
                                dtype=hz.dtype, device=hz.device)
    if isinstance(hz, SparseGcZ):
        return hz.apply_relu_triangle()
    if isinstance(hz, BoxHZ):
        relu_lb = torch.clamp(hz.lb, min=0.0)
        relu_ub = torch.clamp(hz.ub, min=0.0)
        return _make_or_box(relu_lb, relu_ub,
                            dtype=hz.dtype, device=hz.device)

    # HZono: method-specific dispatch. Mirrors HyZor's HybridZReLU but
    # uses ACT's parity-tested encodings. Memory pre-check is the same.
    n = int(hz.c.shape[0])
    ng = int(hz.Gc.shape[1])
    nb = int(hz.Gb.shape[1])
    nc = int(hz.b.shape[0])
    k_max = n
    new_ng = ng + 4 * k_max
    new_nb = nb + k_max
    new_nc = nc + 3 * k_max
    bytes_per = 8
    peak = (n * new_ng + n * new_nb + new_nc * (new_ng + new_nb)) * bytes_per * 3

    budget_gb = float(os.environ.get("HYZOR_RELU_MEM_BUDGET_GB", "8.0"))
    budget_bytes = int(budget_gb * (1024 ** 3))

    if peak > budget_bytes:
        try:
            return hz_apply_relu_triangle(hz)
        except Exception:
            return _box_fallback_relu(hz)

    try:
        if method in ("eq_lagr_v8", "eq_native", "exact"):
            # Stage 4d+4e: HyZor's eq_lagr_v8 = bounds-cascade +
            # intersect_box + applyReLU_eq_native + binary_probe_v8 +
            # project_eq_elim. Wire all five steps for HZono path.
            from act.back_end.hybridz_tf.algorithms.bounds_tighten import (
                hz_intersect_box,
            )
            from act.back_end.hybridz_tf.algorithms.binary_probe import (
                binary_probe, binary_probe_v8,
            )
            from act.back_end.hybridz_tf.algorithms.eq_elim import (
                project_eq_elim,
            )

            # 1. Tight pre-ReLU bounds (cascade).
            lb_t, ub_t = _hzono_tight_bounds(hz)
            # 2. intersect_box — adds 2n new rows. HyZor (HybridZVerifier.py:401)
            #    mem-skips this when the resulting Ac/Ab block would exceed
            #    ~55% of free GPU budget (matters most for conv layers with
            #    n in the 16K-65K range where 2n is huge).
            from act.back_end.hybridz_tf.algorithms.v8_memaware import (
                apply_relu_v8_memaware as _v8_relu_memaware,
                get_v8_mem_budget_bytes as _v8_budget,
                estimate_intersect_box_peak_bytes as _v8_est_ib,
            )
            ng_base = max(int(getattr(hz, "_base_ng", int(hz.Gc.shape[1]))), 5)
            skip_intersect = False
            budget_b = _v8_budget(hz)
            if budget_b is not None and int(hz.b.shape[0]) > 0:
                elem_now = 8 if hz.c.dtype == torch.float64 else 4
                est_ib = _v8_est_ib(hz, elem_now)
                if est_ib > int(0.55 * budget_b):
                    skip_intersect = True
            hz_clipped = (hz_intersect_box(hz, lb_t, ub_t)
                          if (int(hz.b.shape[0]) > 0 and not skip_intersect)
                          else hz)
            # 3. Mem-aware ReLU encoding: picks eq_native/compact/triangle
            #    and f64/f32 to fit budget. HyZor:160 port.
            hz_after_relu = _v8_relu_memaware(hz_clipped, lb_t, ub_t)
            # 4. Binary probe — for v8, dispatch to binary_probe_v8 which
            #    adds the LP pairwise tail on hard cases (HyZor:498-540).
            #    Adaptive budgets from layer difficulty (HyZor:510-540).
            if method == "eq_lagr_v8" and int(hz_after_relu.Gb.shape[1]) > 0:
                try:
                    nb_now = int(hz_after_relu.Gb.shape[1])
                    n_eq_now = (int(hz_after_relu.eq_mask.sum().item())
                                if hz_after_relu.eq_mask is not None
                                else int(hz_after_relu.b.shape[0]))
                    hard_nb = min(1.0, float(nb_now) / 256.0)
                    hard_eq = min(1.0, float(n_eq_now) / 1024.0)
                    hard = 0.55 * hard_nb + 0.45 * hard_eq
                    probe_timeout = 2.2 + 1.6 * hard
                    if nb_now < 128:
                        hz_after_relu = binary_probe_v8(
                            hz_after_relu,
                            timeout=max(1.2, 0.9 * probe_timeout),
                            max_pairs=0,
                            enable_pairwise=False,
                        )
                    else:
                        hz_after_relu = binary_probe_v8(
                            hz_after_relu,
                            timeout=probe_timeout,
                            max_pairs=8 + int(round(8.0 * hard)),
                            enable_pairwise=True,
                            pairwise_min_nb=192,
                            pairwise_min_eq=64,
                            pairwise_min_cooc=2,
                            pairwise_time_cap=0.20 + 0.40 * hard,
                            warmup_pairs=6 + int(round(4.0 * hard)),
                        )
                except Exception:
                    pass
            # 5. project_eq_elim with adaptive bonus (HyZor: line 569-574).
            #    adapt_bonus = 1 iff inp.dim>=256 AND unstable_ratio>0.55.
            if method == "eq_lagr_v8" and ng_base > 0:
                unstable = int(
                    ((lb_t.view(-1) < 0) & (ub_t.view(-1) > 0)).sum().item()
                )
                ratio = float(unstable) / max(1, int(hz.c.shape[0]))
                adapt_bonus = 1 if (int(hz.c.shape[0]) >= 256 and ratio > 0.55) else 0
                try:
                    hz_after_relu = project_eq_elim(
                        hz_after_relu, ng_base=ng_base + adapt_bonus
                    )
                except Exception:
                    pass
            return hz_after_relu
        if method == "triangle":
            return _propagate_base_any(hz, hz_apply_relu_triangle(hz))
        if method == "compact":
            return _propagate_base_any(hz, hz_apply_relu_compact(hz))
        if method in ("bigM", "bigM_fast", "exact_box"):
            return _propagate_base_any(hz, hz_apply_relu_bigM_fast(hz))
        # Unknown method → exact (eq_native).
        return _propagate_base_any(hz, hz_apply_relu(hz))
    except (ValueError, AssertionError):
        try:
            return _propagate_base_any(hz, hz_apply_relu(hz))
        except Exception:
            return _box_fallback_relu(hz)
    except (MemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if "memory" in msg or "alloc" in msg:
            try:
                return hz_apply_relu_triangle(hz)
            except Exception:
                return _box_fallback_relu(hz)
        raise


def hz_apply_leaky_relu_v8(hz, alpha: float):
    """LeakyReLU. Faithful port of HyZor (__init__.py:1470)."""
    if isinstance(hz, LazyChainHZ):
        if hz.can_materialize():
            hz = hz.to_full_hz()
        else:
            # No sparse-Gc leaky encoding; snapshot to box and apply IBP leaky.
            box = hz.snapshot_to_box()
            return _leaky_box(box, alpha)
    if isinstance(hz, SparseGcZ):
        return _leaky_box(_to_box(hz), alpha)
    if isinstance(hz, BoxHZ):
        return _leaky_box(hz, alpha)
    return hz_apply_leaky_relu(hz, alpha)


def _leaky_box(box: BoxHZ, alpha: float):
    """IBP-style leaky ReLU on BoxHZ: lb' = min(lb, alpha*lb)+0 if pos etc."""
    a = float(alpha)
    lb, ub = box.lb, box.ub
    lb_pos = lb.clamp(min=0.0)
    lb_neg = lb.clamp(max=0.0) * a
    ub_pos = ub.clamp(min=0.0)
    ub_neg = ub.clamp(max=0.0) * a
    return _make_or_box(lb_pos + lb_neg, ub_pos + ub_neg,
                         dtype=box.dtype, device=box.device)


# ---------------------------------------------------------------------------
# Multi-input ops
# ---------------------------------------------------------------------------


def hz_minkowski_sum(hz_x, hz_y):
    """Minkowski sum. HyZor parity (__init__.py:1285): when both operands
    are dense HZ, dispatch to the prefix-sharing ``add`` (= ``hz_sgm_add``)
    — HyZor's ``hz_minkowski_sum`` also calls ``hz_x.add(hz_y)`` so the
    shared input-pixel base does NOT get duplicated. The previous ACT
    impl did block-diagonal concat, inflating ng through every resnet
    add block.
    """
    hz_x = _coerce_for_combine(hz_x)
    hz_y = _coerce_for_combine(hz_y)
    if isinstance(hz_x, BoxHZ) or isinstance(hz_y, BoxHZ):
        bx = _to_box(hz_x); by = _to_box(hz_y)
        return _make_or_box(bx.lb + by.lb, bx.ub + by.ub,
                            dtype=bx.dtype, device=bx.device)
    # Dense HZ path: use prefix-sharing SGM add.
    return _act_hz_sgm_add(hz_x, hz_y)


def hz_sgm_add(hz_x, hz_y):
    """Shared Generator Merge add. Faithful port of HyZor (__init__.py:1295)."""
    hz_x = _coerce_for_combine(hz_x)
    hz_y = _coerce_for_combine(hz_y)
    if isinstance(hz_x, BoxHZ) or isinstance(hz_y, BoxHZ):
        bx = _to_box(hz_x); by = _to_box(hz_y)
        return _make_or_box(bx.lb + by.lb, bx.ub + by.ub,
                            dtype=bx.dtype, device=bx.device)
    return _act_hz_sgm_add(hz_x, hz_y)


def shares_generator(hz_x, hz_y) -> bool:
    """SGM heuristic. Faithful port of HyZor (__init__.py:1307)."""
    if isinstance(hz_x, (BoxHZ, LazyChainHZ, SparseGcZ)) or \
       isinstance(hz_y, (BoxHZ, LazyChainHZ, SparseGcZ)):
        return False
    return _act_shares_generator(hz_x, hz_y)


def hz_concat(hz_list):
    """Concatenate HZ instances. Faithful port of HyZor (__init__.py:1320).

    All-HZono path implements block-diagonal layout inline (avoiding
    circular dependency on hz_ops).
    """
    hz_list = list(hz_list)
    if len(hz_list) == 1:
        return hz_list[0]
    if not hz_list:
        raise ValueError("hz_concat: empty list")
    hz_list = [_coerce_for_combine(h) for h in hz_list]
    if any(isinstance(h, BoxHZ) for h in hz_list):
        boxes = [_to_box(h) for h in hz_list]
        lb_cat = torch.cat([b.lb for b in boxes], dim=0)
        ub_cat = torch.cat([b.ub for b in boxes], dim=0)
        return _make_or_box(lb_cat, ub_cat,
                            dtype=boxes[0].dtype, device=boxes[0].device)

    # All HZono: build block-diagonal layout inline.
    dtype = hz_list[0].c.dtype
    device = hz_list[0].c.device
    cs = [h.c for h in hz_list]
    n_total = sum(int(c.shape[0]) for c in cs)
    ng_offs = [0]
    nb_offs = [0]
    for h in hz_list:
        ng_offs.append(ng_offs[-1] + int(h.Gc.shape[1]))
        nb_offs.append(nb_offs[-1] + int(h.Gb.shape[1]))
    ng_total = ng_offs[-1]
    nb_total = nb_offs[-1]
    c_new = torch.cat(cs, dim=0)
    Gc_full = torch.zeros((n_total, ng_total), dtype=dtype, device=device)
    Gb_full = torch.zeros((n_total, nb_total), dtype=dtype, device=device)
    row_off = 0
    for i, h in enumerate(hz_list):
        ni = int(h.c.shape[0])
        ngi = int(h.Gc.shape[1])
        nbi = int(h.Gb.shape[1])
        if ngi > 0:
            Gc_full[row_off:row_off + ni, ng_offs[i]:ng_offs[i + 1]] = h.Gc
        if nbi > 0:
            Gb_full[row_off:row_off + ni, nb_offs[i]:nb_offs[i + 1]] = h.Gb
        row_off += ni
    nc_total = sum(int(h.b.shape[0]) for h in hz_list)
    Ac_full = torch.zeros((nc_total, ng_total), dtype=dtype, device=device)
    Ab_full = torch.zeros((nc_total, nb_total), dtype=dtype, device=device)
    b_full = torch.zeros((nc_total, 1), dtype=dtype, device=device)
    em_full = torch.zeros(nc_total, dtype=torch.bool, device=device)
    row_c = 0
    for i, h in enumerate(hz_list):
        nci = int(h.b.shape[0])
        ngi = int(h.Gc.shape[1])
        nbi = int(h.Gb.shape[1])
        if nci > 0:
            if ngi > 0:
                Ac_full[row_c:row_c + nci, ng_offs[i]:ng_offs[i + 1]] = h.Ac
            if nbi > 0:
                Ab_full[row_c:row_c + nci, nb_offs[i]:nb_offs[i + 1]] = h.Ab
            b_full[row_c:row_c + nci, :] = h.b
            em_full[row_c:row_c + nci] = _eq_mask_of(h)
        row_c += nci
    out = HZono(c=c_new, Gc=Gc_full, Gb=Gb_full,
                Ac=Ac_full, Ab=Ab_full, b=b_full, eq_mask=em_full)
    return _propagate_base_multi(hz_list, out)


def hz_intersect_polytope(hz, A, b):
    """Intersect with A·x ≤ b. Faithful port of HyZor (__init__.py:1336).

    Phase 1-3 types are coerced to dense HZ (no LP slots) first.
    """
    if isinstance(hz, LazyChainHZ):
        hz = hz.freeze()
    if isinstance(hz, SparseGcZ):
        hz = hz.to_dense_hz()
    if isinstance(hz, BoxHZ):
        hz = hz.to_hzono()

    A_t = torch.as_tensor(A, dtype=hz.c.dtype, device=hz.c.device)
    if A_t.dim() == 1:
        A_t = A_t.view(1, -1)
    b_t = torch.as_tensor(b, dtype=hz.c.dtype, device=hz.c.device).view(-1, 1)
    new_rows = int(A_t.shape[0])

    Ac_add = A_t @ hz.Gc
    Ab_add = (A_t @ hz.Gb if int(hz.Gb.shape[1]) > 0
              else torch.zeros((new_rows, 0),
                               dtype=hz.c.dtype, device=hz.c.device))
    b_add = b_t - A_t @ hz.c

    new_Ac = torch.cat([hz.Ac, Ac_add], dim=0)
    new_Ab = torch.cat([hz.Ab, Ab_add], dim=0)
    new_b = torch.cat([hz.b, b_add], dim=0)
    em_old = _eq_mask_of(hz)
    em_new = torch.cat([
        em_old,
        torch.zeros(new_rows, dtype=torch.bool, device=hz.c.device),
    ])
    out = HZono(
        c=hz.c.clone(), Gc=hz.Gc.clone(), Gb=hz.Gb.clone(),
        Ac=new_Ac, Ab=new_Ab, b=new_b, eq_mask=em_new,
    )
    return _propagate_base_any(hz, out)


# ---------------------------------------------------------------------------
# MaxPool
# ---------------------------------------------------------------------------


def hz_maxpool2d(hz, *, kernel_size, stride=None, padding=0, input_shape):
    """2D max-pool. Faithful port of HyZor (__init__.py:1180)."""
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
        _, C, H, W_ = input_shape
    elif len(input_shape) == 3:
        C, H, W_ = input_shape
    else:
        raise ValueError(f"hz_maxpool2d: bad input_shape {input_shape}")
    C, H, W_ = int(C), int(H), int(W_)

    # Coerce non-dense flavors. HZ MaxPool needs explicit Gc/Gb for row select.
    if isinstance(hz, LazyChainHZ):
        hz = hz.freeze()
    if isinstance(hz, SparseGcZ):
        try:
            hz = hz.to_dense_hz()
        except Exception:
            lb, ub = hz.bounds()
            hz = BoxHZ(lb, ub, dtype=hz.dtype, device=hz.device)

    if isinstance(hz, BoxHZ):
        # Per-pixel IBP max-pool via F.max_pool2d on lb/ub.
        import torch.nn.functional as F
        lb4 = hz.lb.view(1, C, H, W_)
        ub4 = hz.ub.view(1, C, H, W_)
        lb_out = F.max_pool2d(lb4, kernel_size=(kh, kw),
                               stride=(sh, sw), padding=pad_use)
        ub_out = F.max_pool2d(ub4, kernel_size=(kh, kw),
                               stride=(sh, sw), padding=pad_use)
        return _make_or_box(lb_out.flatten(), ub_out.flatten(),
                             dtype=hz.dtype, device=hz.device)

    # HZono path: use ACT.s tf_cnn.hz_maxpool2d.
    return _act_maxpool_hzono(
        hz, kernel_size=(kh, kw), stride=(sh, sw),
        padding=pad_use, input_shape=(C, H, W_),
    )


__all__ = [
    "hz_from_bounds", "hz_dense", "hz_scale", "hz_add_const", "hz_bn",
    "hz_conv2d", "hz_apply_relu_v8", "hz_apply_leaky_relu_v8",
    "hz_minkowski_sum", "hz_sgm_add", "shares_generator",
    "hz_concat", "hz_intersect_polytope", "hz_maxpool2d",
    "_coerce_for_combine", "_box_fallback_relu",
]
