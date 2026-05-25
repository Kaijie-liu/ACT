#===- act/back_end/hybridz_tf/algorithms/relu_methods.py - Alternate HZ ReLU Encodings -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Triangle / compact / bigM ReLU encodings for HZ. Lower-precision than
#   eq_lagr but cheaper; the cascade scheduler picks per-layer.
#
#===---------------------------------------------------------------------===#

"""Per-encoding ReLU methods for HZono.

This module contains alternative ReLU encodings beyond the tight
``hz_apply_relu`` (eq_lagr_v8 in HyZor parlance) that lives in
``tf_mlp.py``. The eq_lagr_v8 encoding is the most precise but
allocates 4 continuous + 1 binary generator + 3 equality rows per
unstable neuron — too expensive for early layers of deep CNNs.

This module provides:

  - ``hz_apply_relu_triangle``: DeepZ-style chord relaxation
    (Singh et al., 2018). ZERO equality rows, ZERO binary slots,
    one new continuous generator per unstable neuron. Memory-friendly.

  - ``pick_relu_method_for_layer``: a memaware dispatcher that picks
    between ``eq_native`` (=hz_apply_relu) and ``triangle`` based on
    estimated peak GPU bytes and free memory.

Both methods are faithful ports of their HyZor counterparts
(``HybridZonotope.applyReLU_triangle`` HZ:5857 and the dispatch
logic in ``HybridZVerifier._apply_relu_v8_memaware`` HZV:160+).

Sigmoid / Tanh / Leaky ReLU all use their own encodings already in
tf_mlp.py and are not affected by this module.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple

import torch

from act.back_end.solver.solver_hz import HZono, hz_compute_bounds


# ---------------------------------------------------------------------------
# Method 1: DeepZ-style triangle (cheap, lossy)
# ---------------------------------------------------------------------------


def hz_apply_relu_triangle(hz: HZono, external_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> HZono:
    """Sound DeepZ-style ReLU relaxation.

    For unstable neuron i with bounds ``[l, u]`` (``l < 0 < u``)::

        λ_i = u / (u - l)            in (0, 1)
        μ_i = -l u / (2 (u - l))     > 0
        y_i = λ_i x_i + μ_i (1 + ε_new_i)   with ε_new_i ∈ [-1, 1]

    Adds **one new continuous generator** per unstable neuron, NO new
    equality rows, NO new binary slot. Bit-identical port of
    ``HybridZonotope.applyReLU_triangle`` (HZ:5857-5946).

    For ``external_bounds`` arg: if provided, used for active/inactive
    classification (HyZor's `external_bounds` semantic). Otherwise the
    unconstrained box bounds are used.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = int(hz.c.shape[0])
    ng0 = int(hz.Gc.shape[1])
    nb0 = int(hz.Gb.shape[1])
    nc0 = int(hz.b.shape[0])

    if external_bounds is not None:
        lb = external_bounds[0].to(device=device, dtype=dtype).view(-1)
        ub = external_bounds[1].to(device=device, dtype=dtype).view(-1)
    else:
        # Unconstrained box bounds = c ± |Gc|·1 ± |Gb|·1.
        radius = hz.Gc.abs().sum(dim=1) + hz.Gb.abs().sum(dim=1)
        c_flat = hz.c.flatten()
        lb = c_flat - radius
        ub = c_flat + radius

    is_active = (lb >= 0)
    is_inactive = (ub <= 0)
    is_unstable = ~(is_active | is_inactive)

    unstable_idx = torch.nonzero(is_unstable, as_tuple=False).view(-1)
    k = int(unstable_idx.numel())
    em_old = hz.eq_mask if (hz.eq_mask is not None and int(hz.eq_mask.numel()) == nc0) \
        else torch.ones(nc0, dtype=torch.bool, device=device)

    if k == 0:
        # All stable: just zero out inactive rows.
        new_c = hz.c.clone()
        new_Gc = hz.Gc.clone()
        new_Gb = hz.Gb.clone()
        if is_inactive.any():
            new_c[is_inactive] = 0.0
            if ng0 > 0:
                new_Gc[is_inactive] = 0.0
            if nb0 > 0:
                new_Gb[is_inactive] = 0.0
        return HZono(c=new_c, Gc=new_Gc, Gb=new_Gb,
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(),
                     b=hz.b.clone(), eq_mask=em_old.clone())

    l_uns = lb[unstable_idx]
    u_uns = ub[unstable_idx]
    lam = u_uns / (u_uns - l_uns)
    mu = -l_uns * u_uns / (2.0 * (u_uns - l_uns))

    ng = ng0 + k

    c_out = torch.zeros((n, 1), device=device, dtype=dtype)
    if is_active.any():
        c_out[is_active] = hz.c[is_active]
    c_out[unstable_idx, 0] = lam * hz.c[unstable_idx, 0] + mu

    Gc_out = torch.zeros((n, ng), device=device, dtype=dtype)
    if ng0 > 0:
        if is_active.any():
            Gc_out[is_active, :ng0] = hz.Gc[is_active]
        Gc_out[unstable_idx, :ng0] = lam.unsqueeze(1) * hz.Gc[unstable_idx]
    j_idx = torch.arange(k, device=device)
    Gc_out[unstable_idx, ng0 + j_idx] = mu

    Gb_out = torch.zeros((n, nb0), device=device, dtype=dtype)
    if nb0 > 0:
        if is_active.any():
            Gb_out[is_active, :nb0] = hz.Gb[is_active]
        Gb_out[unstable_idx, :nb0] = lam.unsqueeze(1) * hz.Gb[unstable_idx]

    # Constraint rows: extend Ac with k zero columns (new cont gens
    # don't appear in old constraints); Ab and b unchanged.
    if nc0 > 0:
        Ac_out = torch.cat([
            hz.Ac, torch.zeros((nc0, k), device=device, dtype=dtype)
        ], dim=1)
    else:
        Ac_out = torch.zeros((0, ng), device=device, dtype=dtype)
    Ab_out = hz.Ab
    b_out = hz.b

    return HZono(c=c_out, Gc=Gc_out, Gb=Gb_out,
                 Ac=Ac_out, Ab=Ab_out, b=b_out,
                 eq_mask=em_old.clone())


# ---------------------------------------------------------------------------
# Method 2: compact (v-z coupling, no linking)
# ---------------------------------------------------------------------------


def hz_apply_relu_compact(hz: HZono) -> HZono:
    """Sound ReLU overapprox with v-z coupling but NO linking constraints.

    Tighter than triangle (enforces z=-1 ⇒ v=0 ⇒ y=0 exactly), looser
    than eq_native (no linking equality). Allocates:

      +k continuous gen (v), +k binary (z), +2k sparse inequality rows.

    Compare:
      eq_native: +4k cont, +k bin, +3k eq rows
      triangle:  +k cont,  0 bin,  0 rows

    Per unstable neuron i with domain bound a_i = |c_i| + Σ|Gc_i| + Σ|Gb_i|::

        y_i = a_i/4 + (a_i/2) v_i + (a_i/4) z_i,
              with v_i ∈ [-1,+1] and |v_i| ≤ (z_i + 1)/2

    The |v| ≤ (z+1)/2 constraint encodes "z=-1 forces v=0" (so y=0).

    Faithful port of ``HybridZonotope.applyReLU_compact`` (HZ:6114-6235).
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = int(hz.c.shape[0])
    ng0 = int(hz.Gc.shape[1])
    nb0 = int(hz.Gb.shape[1])
    nc0 = int(hz.b.shape[0])

    radius = hz.Gc.abs().sum(dim=1) + hz.Gb.abs().sum(dim=1)
    c_flat = hz.c.flatten()
    lb_unc = c_flat - radius
    ub_unc = c_flat + radius

    is_active = (lb_unc >= 0)
    is_inactive = (ub_unc <= 0)
    is_unstable = ~(is_active | is_inactive)
    unstable_idx = torch.nonzero(is_unstable, as_tuple=False).view(-1)
    k = int(unstable_idx.numel())

    em_old = hz.eq_mask if (hz.eq_mask is not None and int(hz.eq_mask.numel()) == nc0) \
        else torch.ones(nc0, dtype=torch.bool, device=device)

    if k == 0:
        new_c = hz.c.clone()
        new_Gc = hz.Gc.clone()
        new_Gb = hz.Gb.clone()
        if is_inactive.any():
            new_c[is_inactive] = 0.0
            if ng0 > 0:
                new_Gc[is_inactive] = 0.0
            if nb0 > 0:
                new_Gb[is_inactive] = 0.0
        return HZono(c=new_c, Gc=new_Gc, Gb=new_Gb,
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(),
                     b=hz.b.clone(), eq_mask=em_old.clone())

    a_uns = hz.c[unstable_idx, 0].abs()
    if ng0 > 0:
        a_uns = a_uns + hz.Gc[unstable_idx].abs().sum(dim=1)
    if nb0 > 0:
        a_uns = a_uns + hz.Gb[unstable_idx].abs().sum(dim=1)
    a_uns = torch.clamp(a_uns, min=1e-12)

    ng = ng0 + k
    nb = nb0 + k
    j_idx = torch.arange(k, device=device)
    v_cols = ng0 + j_idx
    z_cols = nb0 + j_idx

    c_out = torch.zeros((n, 1), device=device, dtype=dtype)
    if is_active.any():
        c_out[is_active] = hz.c[is_active]
    c_out[unstable_idx] = (a_uns / 4.0).unsqueeze(1)

    Gc_out = torch.zeros((n, ng), device=device, dtype=dtype)
    if ng0 > 0 and is_active.any():
        Gc_out[is_active, :ng0] = hz.Gc[is_active]
    Gc_out[unstable_idx, v_cols] = a_uns / 2.0

    Gb_out = torch.zeros((n, nb), device=device, dtype=dtype)
    if nb0 > 0 and is_active.any():
        Gb_out[is_active, :nb0] = hz.Gb[is_active]
    Gb_out[unstable_idx, z_cols] = a_uns / 4.0

    # Extend old constraints with zero cols for new v / z.
    if nc0 > 0:
        Ac0 = torch.zeros((nc0, ng), device=device, dtype=dtype)
        Ab0 = torch.zeros((nc0, nb), device=device, dtype=dtype)
        if ng0 > 0:
            Ac0[:, :ng0] = hz.Ac
        if nb0 > 0:
            Ab0[:, :nb0] = hz.Ab
        b0 = hz.b.clone()
    else:
        Ac0 = torch.zeros((0, ng), device=device, dtype=dtype)
        Ab0 = torch.zeros((0, nb), device=device, dtype=dtype)
        b0 = torch.zeros((0, 1), device=device, dtype=dtype)

    # 2k v-z graph rows (|v| <= (z+1)/2):
    #   v - 0.5 z <= 0.5
    #  -v - 0.5 z <= 0.5
    Ac_graph = torch.zeros((2 * k, ng), device=device, dtype=dtype)
    Ab_graph = torch.zeros((2 * k, nb), device=device, dtype=dtype)
    b_graph = torch.full((2 * k, 1), 0.5, device=device, dtype=dtype)
    r0 = 2 * j_idx
    r1 = 2 * j_idx + 1
    Ac_graph[r0, v_cols] = 1.0
    Ab_graph[r0, z_cols] = -0.5
    Ac_graph[r1, v_cols] = -1.0
    Ab_graph[r1, z_cols] = -0.5

    Ac_out = torch.cat([Ac0, Ac_graph], dim=0)
    Ab_out = torch.cat([Ab0, Ab_graph], dim=0)
    b_out = torch.cat([b0, b_graph], dim=0)
    em_new = torch.cat(
        [em_old, torch.zeros(2 * k, dtype=torch.bool, device=device)],
        dim=0,
    )

    return HZono(c=c_out, Gc=Gc_out, Gb=Gb_out,
                 Ac=Ac_out, Ab=Ab_out, b=b_out,
                 eq_mask=em_new)


# ---------------------------------------------------------------------------
# Method 3: Big-M (η, z + 3 inequalities per unstable)
# ---------------------------------------------------------------------------


def hz_apply_relu_bigM_fast(
    hz: HZono,
    external_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> HZono:
    """Big-M ReLU encoding for HZono.

    Per unstable neuron i with bounds [l, u] (l<0<u), introduce fresh
    eta_i ∈ [-1,+1] and z_i ∈ {-1,+1}, parameterize y in [0, u]::

        y_i = u/2 + (u/2) eta_i

    with three Big-M-style inequalities::

        x_i - y_i <= 0                      (active branch enforces y=x)
        y_i - (u/2) z_i <= u/2              (z=-1 forces y=0)
        y_i - x_i - (l/2) z_i <= -l/2       (z=+1 forces y=x)

    Adds +k cont gens, +k binary, +3k ineq rows. NO equality rows.

    Stable rows pass through (active=identity, inactive=zero). Faithful
    port of ``HybridZonotope.applyReLU_bigM_fast`` (HZ:5377-5540).

    Builds augmented (2n) coords [x;y] then slices the y rows -- mirrors
    HyZor's projection step without materialising the 2n×n selector.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = int(hz.c.shape[0])
    ng0 = int(hz.Gc.shape[1])
    nb0 = int(hz.Gb.shape[1])
    nc0 = int(hz.b.shape[0])

    if external_bounds is not None:
        lb = external_bounds[0].to(device=device, dtype=dtype).view(-1)
        ub = external_bounds[1].to(device=device, dtype=dtype).view(-1)
    else:
        radius = hz.Gc.abs().sum(dim=1) + hz.Gb.abs().sum(dim=1)
        c_flat = hz.c.flatten()
        lb = c_flat - radius
        ub = c_flat + radius

    pos_mask = (lb >= 0)
    neg_mask = (ub <= 0)
    mix_mask = ~(pos_mask | neg_mask)
    mix_idx = torch.nonzero(mix_mask, as_tuple=False).view(-1)
    k = int(mix_idx.numel())

    em_old = hz.eq_mask if (hz.eq_mask is not None and int(hz.eq_mask.numel()) == nc0) \
        else torch.ones(nc0, dtype=torch.bool, device=device)

    if k == 0:
        if not torch.any(neg_mask):
            return HZono(c=hz.c.clone(), Gc=hz.Gc.clone(), Gb=hz.Gb.clone(),
                         Ac=hz.Ac.clone(), Ab=hz.Ab.clone(),
                         b=hz.b.clone(), eq_mask=em_old.clone())
        keep = pos_mask.to(dtype=dtype).unsqueeze(1)
        return HZono(
            c=hz.c * keep,
            Gc=hz.Gc * keep if hz.Gc.numel() else hz.Gc,
            Gb=hz.Gb * keep if hz.Gb.numel() else hz.Gb,
            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(),
            b=hz.b.clone(), eq_mask=em_old.clone(),
        )

    # Augmented [x; y] coords (2n rows). Old factors get k zero columns
    # for the new eta variables; new binary z columns stay zero on coords.
    Gc_aug = torch.zeros((2 * n, ng0 + k), device=device, dtype=dtype)
    Gb_aug = torch.zeros((2 * n, nb0 + k), device=device, dtype=dtype)
    c_aug = torch.zeros((2 * n, 1), device=device, dtype=dtype)

    # x-part (top n) = old HZ.
    c_aug[:n, :] = hz.c
    if ng0 > 0:
        Gc_aug[:n, :ng0] = hz.Gc
    if nb0 > 0:
        Gb_aug[:n, :nb0] = hz.Gb

    # y-part (bottom n) starts as old HZ then patches per neuron class.
    c_aug[n:, :] = hz.c
    if ng0 > 0:
        Gc_aug[n:, :ng0] = hz.Gc
    if nb0 > 0:
        Gb_aug[n:, :nb0] = hz.Gb

    # Inactive: y = 0.
    neg_idx = torch.nonzero(neg_mask, as_tuple=False).view(-1)
    if int(neg_idx.numel()) > 0:
        c_aug[n + neg_idx, :] = 0.0
        if ng0 > 0:
            Gc_aug[n + neg_idx, :ng0] = 0.0
        if nb0 > 0:
            Gb_aug[n + neg_idx, :nb0] = 0.0

    # Mix: y_i = u/2 + (u/2) eta_t. Clear old factor contribution, set eta col.
    u_mix = ub[mix_idx].clamp(min=0)
    l_mix = lb[mix_idx]
    c_mix = hz.c[mix_idx, 0]
    t_idx = torch.arange(k, device=device)

    c_aug[n + mix_idx, 0] = u_mix / 2.0
    if ng0 > 0:
        Gc_aug[n + mix_idx, :ng0] = 0.0
    if nb0 > 0:
        Gb_aug[n + mix_idx, :nb0] = 0.0
    Gc_aug[n + mix_idx, ng0 + t_idx] = u_mix / 2.0

    # Extend old constraints with zero cols for eta and z.
    if nc0 > 0:
        Ac0 = torch.cat([hz.Ac, torch.zeros((nc0, k), device=device, dtype=dtype)], dim=1)
        Ab0 = torch.cat([hz.Ab, torch.zeros((nc0, k), device=device, dtype=dtype)], dim=1)
        b0 = hz.b
    else:
        Ac0 = torch.zeros((0, ng0 + k), device=device, dtype=dtype)
        Ab0 = torch.zeros((0, nb0 + k), device=device, dtype=dtype)
        b0 = torch.zeros((0, 1), device=device, dtype=dtype)

    # 3k Big-M rows over [xi_c; eta] and [xi_b; z]:
    #   row1: x - y <= 0 (active enforces y=x)
    #   row2: y - (u/2) z <= u/2 (z=-1 forces y=0)
    #   row3: y - x - (l/2) z <= -l/2 (z=+1 forces y=x)
    row1 = 3 * t_idx
    row2 = 3 * t_idx + 1
    row3 = 3 * t_idx + 2
    col_eta = ng0 + t_idx
    col_z = nb0 + t_idx

    Ac_relu = torch.zeros((3 * k, ng0 + k), device=device, dtype=dtype)
    Ab_relu = torch.zeros((3 * k, nb0 + k), device=device, dtype=dtype)
    b_relu = torch.zeros(3 * k, device=device, dtype=dtype)

    if ng0 > 0:
        ng_cols = torch.arange(ng0, device=device)
        Ac_relu[row1[:, None], ng_cols[None, :]] = hz.Gc[mix_idx, :]
        Ac_relu[row3[:, None], ng_cols[None, :]] = -hz.Gc[mix_idx, :]
    Ac_relu[row1, col_eta] = -u_mix / 2.0
    Ac_relu[row2, col_eta] = u_mix / 2.0
    Ac_relu[row3, col_eta] = u_mix / 2.0

    if nb0 > 0:
        nb_cols = torch.arange(nb0, device=device)
        Ab_relu[row1[:, None], nb_cols[None, :]] = hz.Gb[mix_idx, :]
        Ab_relu[row3[:, None], nb_cols[None, :]] = -hz.Gb[mix_idx, :]
    Ab_relu[row2, col_z] = -u_mix / 2.0
    Ab_relu[row3, col_z] = -l_mix / 2.0

    b_relu[row1] = u_mix / 2.0 - c_mix
    # b_relu[row2] = 0
    b_relu[row3] = -l_mix / 2.0 + c_mix - u_mix / 2.0
    b_relu = b_relu.view(-1, 1)

    newAc = torch.cat([Ac0, Ac_relu], dim=0)
    newAb = torch.cat([Ab0, Ab_relu], dim=0)
    newb = torch.cat([b0, b_relu], dim=0)
    em_new = torch.cat(
        [em_old, torch.zeros(3 * k, dtype=torch.bool, device=device)],
        dim=0,
    )

    # Project to y (take bottom n rows of augmented Gc/Gb/c).
    return HZono(
        c=c_aug[n:, :],
        Gc=Gc_aug[n:, :],
        Gb=Gb_aug[n:, :],
        Ac=newAc, Ab=newAb, b=newb,
        eq_mask=em_new,
    )


# ---------------------------------------------------------------------------
# Method 4: convex_hull_cont — continuous-only convex-hull ReLU
# ---------------------------------------------------------------------------


def hz_apply_relu_convex_hull(
    hz: HZono,
    external_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> HZono:
    """Continuous-only convex-hull ReLU encoding.

    Per unstable neuron i with bounds [l, u] (l<0<u), introduce one
    fresh continuous factor ``eta_t ∈ [-1,+1]`` and parameterise::

        y_i = u/2 + (u/2) eta_t        (so y_i ∈ [0, u] automatically)

    with TWO inequality rows (the y>=0 / y<=u bounds are absorbed by η)::

        x_i - y_i <= 0                  (lower facet y >= x)
        y_i - lam x_i <= -lam l_i       (upper chord, lam = u/(u-l))

    Per unstable: +1 continuous gen, +0 binary, +2 inequality rows.

    Algebraically yields the **same convex set in (x, y)** as the
    LP-relaxation of ``eq_lagr_v8`` (which uses +4 cont + 1 binary +
    3 equalities per neuron), but with a strictly smaller LP
    representation.  Equivalence proof: project both polytopes onto
    (x_i, y_i) and observe both are exactly the triangle
    ``conv(ReLU)``.

    NOT comparable to ``hz_apply_relu_triangle`` (DeepZ parallelogram,
    which is **strictly looser** because it forces y to lie on a
    parallelogram chord rather than the true convex hull).

    Stable rows pass through (active=identity, inactive=zero).
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = int(hz.c.shape[0])
    ng0 = int(hz.Gc.shape[1])
    nb0 = int(hz.Gb.shape[1])
    nc0 = int(hz.b.shape[0])

    if external_bounds is not None:
        lb = external_bounds[0].to(device=device, dtype=dtype).view(-1)
        ub = external_bounds[1].to(device=device, dtype=dtype).view(-1)
    else:
        radius = hz.Gc.abs().sum(dim=1) + hz.Gb.abs().sum(dim=1)
        c_flat = hz.c.flatten()
        lb = c_flat - radius
        ub = c_flat + radius

    pos_mask = (lb >= 0)
    neg_mask = (ub <= 0)
    mix_mask = ~(pos_mask | neg_mask)
    mix_idx = torch.nonzero(mix_mask, as_tuple=False).view(-1)
    k = int(mix_idx.numel())

    em_old = hz.eq_mask if (hz.eq_mask is not None and int(hz.eq_mask.numel()) == nc0) \
        else torch.ones(nc0, dtype=torch.bool, device=device)

    if k == 0:
        # All stable: zero out inactive rows, identity on active.
        if not torch.any(neg_mask):
            return HZono(c=hz.c.clone(), Gc=hz.Gc.clone(), Gb=hz.Gb.clone(),
                         Ac=hz.Ac.clone(), Ab=hz.Ab.clone(),
                         b=hz.b.clone(), eq_mask=em_old.clone())
        keep = pos_mask.to(dtype=dtype).unsqueeze(1)
        return HZono(
            c=hz.c * keep,
            Gc=hz.Gc * keep if hz.Gc.numel() else hz.Gc,
            Gb=hz.Gb * keep if hz.Gb.numel() else hz.Gb,
            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(),
            b=hz.b.clone(), eq_mask=em_old.clone(),
        )

    u_mix = ub[mix_idx].clamp(min=0)
    l_mix = lb[mix_idx]
    c_mix = hz.c[mix_idx, 0]
    t_idx = torch.arange(k, device=device)
    lam = u_mix / (u_mix - l_mix)  # in (0, 1)

    # ── Build new HZ coords (y rows only — direct projection) ──
    # Stable (active): y = x  → carry old c/Gc/Gb.
    # Stable (inactive): y = 0  → zeros.
    # Mix: y = u/2 + (u/2) eta_t  → c=u/2, Gc col for eta_t = u/2.
    ng_new = ng0 + k
    nb_new = nb0  # NO new binary

    c_out = torch.zeros((n, 1), device=device, dtype=dtype)
    Gc_out = torch.zeros((n, ng_new), device=device, dtype=dtype)
    Gb_out = torch.zeros((n, nb_new), device=device, dtype=dtype)

    # Active rows: pass-through.
    if pos_mask.any():
        c_out[pos_mask] = hz.c[pos_mask]
        if ng0 > 0:
            Gc_out[pos_mask, :ng0] = hz.Gc[pos_mask]
        if nb0 > 0:
            Gb_out[pos_mask, :nb0] = hz.Gb[pos_mask]

    # Mix rows: parametrise y = u/2 + (u/2) eta_t (decoupled from old factors).
    c_out[mix_idx, 0] = u_mix / 2.0
    Gc_out[mix_idx, ng0 + t_idx] = u_mix / 2.0
    # (no old-factor contribution on mix y rows — by parametrisation)

    # ── Extend old constraints with zero cols for new eta vars ──
    if nc0 > 0:
        Ac0 = torch.cat([hz.Ac, torch.zeros((nc0, k), device=device, dtype=dtype)], dim=1)
        # Ab0 keeps its original (nc0, nb0) shape — chull adds no new binaries.
        Ab0 = hz.Ab if nb0 > 0 else torch.zeros((nc0, 0), device=device, dtype=dtype)
        b0 = hz.b
    else:
        Ac0 = torch.zeros((0, ng_new), device=device, dtype=dtype)
        Ab0 = torch.zeros((0, nb_new), device=device, dtype=dtype)
        b0 = torch.zeros((0, 1), device=device, dtype=dtype)

    # ── 2k inequality rows: facet y>=x and upper chord ──
    #
    # row1 (y >= x), i.e.  x_i - y_i <= 0:
    #   x_i = c_i + Gc_i · xi_c + Gb_i · xi_b
    #   y_i = u/2 + (u/2) eta_t
    #   →  Gc_i · xi_c + Gb_i · xi_b - (u/2) eta_t <= u/2 - c_i
    #
    # row2 (upper chord):  y_i - lam x_i <= -lam l_i
    #   →  -lam Gc_i · xi_c - lam Gb_i · xi_b + (u/2) eta_t <= -lam l_i - u/2 + lam c_i
    row1 = 2 * t_idx
    row2 = 2 * t_idx + 1
    col_eta = ng0 + t_idx

    Ac_relu = torch.zeros((2 * k, ng_new), device=device, dtype=dtype)
    Ab_relu = torch.zeros((2 * k, nb_new), device=device, dtype=dtype)
    b_relu = torch.zeros(2 * k, device=device, dtype=dtype)

    if ng0 > 0:
        ng_cols = torch.arange(ng0, device=device)
        Ac_relu[row1[:, None], ng_cols[None, :]] = hz.Gc[mix_idx, :]
        Ac_relu[row2[:, None], ng_cols[None, :]] = -lam.unsqueeze(1) * hz.Gc[mix_idx, :]
    Ac_relu[row1, col_eta] = -u_mix / 2.0
    Ac_relu[row2, col_eta] = u_mix / 2.0

    if nb0 > 0:
        nb_cols = torch.arange(nb0, device=device)
        Ab_relu[row1[:, None], nb_cols[None, :]] = hz.Gb[mix_idx, :]
        Ab_relu[row2[:, None], nb_cols[None, :]] = -lam.unsqueeze(1) * hz.Gb[mix_idx, :]

    b_relu[row1] = u_mix / 2.0 - c_mix
    b_relu[row2] = -lam * l_mix - u_mix / 2.0 + lam * c_mix
    b_relu = b_relu.view(-1, 1)

    newAc = torch.cat([Ac0, Ac_relu], dim=0)
    # Always concatenate: even with nb0==0, row count must match Ac/b/eq_mask.
    newAb = torch.cat([Ab0, Ab_relu], dim=0)
    newb = torch.cat([b0, b_relu], dim=0)
    em_new = torch.cat(
        [em_old, torch.zeros(2 * k, dtype=torch.bool, device=device)],
        dim=0,
    )

    return HZono(
        c=c_out, Gc=Gc_out, Gb=Gb_out,
        Ac=newAc, Ab=newAb, b=newb,
        eq_mask=em_new,
    )


# ---------------------------------------------------------------------------
# External chull mask registry (for property-directed dual-sensitivity masks)
# ---------------------------------------------------------------------------
#
# Set by the driver via ``set_chull_mask_for_relu(ridx, mask)``. Read by
# the SparseGcZ early path and HZono late path in hz_routing.py when
# ``ACT_HZ_DUAL_MASK_ACTIVE=1``. Module-level state is a deliberate hack
# for experimental property-direction integration; production work would
# thread per-call.

_CHULL_MASK_REGISTRY: dict = {}
_CURRENT_RIDX: int = 0


def set_chull_mask_for_relu(ridx: int, mask: torch.Tensor) -> None:
    """Register a per-neuron chull mask for the ``ridx``-th ReLU."""
    _CHULL_MASK_REGISTRY[int(ridx)] = mask


def set_current_relu_idx(ridx: int) -> None:
    """Set by HZVerifier._dispatch before calling hz_apply_relu_v8."""
    global _CURRENT_RIDX
    _CURRENT_RIDX = int(ridx)


def get_chull_mask_for_current_relu() -> Optional[torch.Tensor]:
    return _CHULL_MASK_REGISTRY.get(_CURRENT_RIDX)


def clear_chull_masks() -> None:
    _CHULL_MASK_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Method 5: selective_chull — DeepZ triangle for most + chull tightening
#           on a top-K mask of unstable neurons (sparse cut)
# ---------------------------------------------------------------------------


def hz_apply_relu_selective_chull(
    hz: HZono,
    *,
    external_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    chull_mask: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
) -> HZono:
    """DeepZ triangle ReLU with chull tightening on a selective subset.

    All unstable neurons get DeepZ-style parametrisation
    ``y_i = lam_i x_i + mu_i (1 + eps_new_i)`` (one new continuous gen
    per unstable). Then for each MASKED unstable neuron we add two
    sparse inequality rows that tighten the DeepZ parallelogram into
    the proper convex hull:

      row1 (y_i >= x_i, the lower facet):
        x_i - y_i = (1 - lam_i) x_i - mu_i (1 + eps_new_i)  <= 0
        → (1 - lam_i) Gc_i · xi_c + (1 - lam_i) Gb_i · xi_b
          - mu_i eps_new_i <= mu_i - (1 - lam_i) c_i

      row2 (y_i >= 0, the inactive facet):
        -y_i = -lam_i x_i - mu_i (1 + eps_new_i)  <= 0
        → -lam_i Gc_i · xi_c - lam_i Gb_i · xi_b
          - mu_i eps_new_i <= lam_i c_i + mu_i

    Cost: ALL unstable get +1 cont gen (the DeepZ slack); MASKED unstable
    additionally get +2 inequality rows. The Gc structure is unchanged
    from triangle. Non-masked unstable see no constraint rows — strictly
    cheaper than full chull, strictly tighter than full triangle.

    Mask selection:
      - If ``chull_mask`` is given: 1-D bool tensor over n, True ⇒ apply
        chull tightening to that neuron (if also unstable).
      - Else if ``top_k`` is given: select top-k unstable neurons by
        width ``ub - lb`` (forward heuristic; property-direction is a
        follow-up).
      - If both None: identity-DeepZ (no chull tightening, equivalent
        to ``hz_apply_relu_triangle``).
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = int(hz.c.shape[0])
    ng0 = int(hz.Gc.shape[1])
    nb0 = int(hz.Gb.shape[1])
    nc0 = int(hz.b.shape[0])

    if external_bounds is not None:
        lb = external_bounds[0].to(device=device, dtype=dtype).view(-1)
        ub = external_bounds[1].to(device=device, dtype=dtype).view(-1)
    else:
        radius = hz.Gc.abs().sum(dim=1) + hz.Gb.abs().sum(dim=1)
        c_flat = hz.c.flatten()
        lb = c_flat - radius
        ub = c_flat + radius

    is_active = (lb >= 0)
    is_inactive = (ub <= 0)
    is_unstable = ~(is_active | is_inactive)
    unstable_idx = torch.nonzero(is_unstable, as_tuple=False).view(-1)
    k = int(unstable_idx.numel())

    em_old = hz.eq_mask if (hz.eq_mask is not None and int(hz.eq_mask.numel()) == nc0) \
        else torch.ones(nc0, dtype=torch.bool, device=device)

    if k == 0:
        # No unstable neurons — same as triangle's k==0 branch.
        new_c = hz.c.clone()
        new_Gc = hz.Gc.clone()
        new_Gb = hz.Gb.clone()
        if is_inactive.any():
            new_c[is_inactive] = 0.0
            if ng0 > 0:
                new_Gc[is_inactive] = 0.0
            if nb0 > 0:
                new_Gb[is_inactive] = 0.0
        return HZono(c=new_c, Gc=new_Gc, Gb=new_Gb,
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(),
                     b=hz.b.clone(), eq_mask=em_old.clone())

    # ── DeepZ triangle skeleton on ALL unstable (identical to hz_apply_relu_triangle) ──
    l_uns = lb[unstable_idx]
    u_uns = ub[unstable_idx]
    lam = u_uns / (u_uns - l_uns)
    mu = -l_uns * u_uns / (2.0 * (u_uns - l_uns))

    ng = ng0 + k

    c_out = torch.zeros((n, 1), device=device, dtype=dtype)
    if is_active.any():
        c_out[is_active] = hz.c[is_active]
    c_out[unstable_idx, 0] = lam * hz.c[unstable_idx, 0] + mu

    Gc_out = torch.zeros((n, ng), device=device, dtype=dtype)
    if ng0 > 0:
        if is_active.any():
            Gc_out[is_active, :ng0] = hz.Gc[is_active]
        Gc_out[unstable_idx, :ng0] = lam.unsqueeze(1) * hz.Gc[unstable_idx]
    j_idx = torch.arange(k, device=device)
    Gc_out[unstable_idx, ng0 + j_idx] = mu

    Gb_out = torch.zeros((n, nb0), device=device, dtype=dtype)
    if nb0 > 0:
        if is_active.any():
            Gb_out[is_active] = hz.Gb[is_active]
        Gb_out[unstable_idx] = lam.unsqueeze(1) * hz.Gb[unstable_idx]

    # ── Old constraints get k zero cols for new eps gens ──
    if nc0 > 0:
        Ac0 = torch.cat([hz.Ac, torch.zeros((nc0, k), device=device, dtype=dtype)], dim=1)
        Ab0 = hz.Ab if nb0 > 0 else torch.zeros((nc0, 0), device=device, dtype=dtype)
        b0 = hz.b
    else:
        Ac0 = torch.zeros((0, ng), device=device, dtype=dtype)
        Ab0 = torch.zeros((0, nb0), device=device, dtype=dtype)
        b0 = torch.zeros((0, 1), device=device, dtype=dtype)

    # ── Derive the chull mask within unstable indices ──
    width_uns = u_uns - l_uns  # > 0 for all unstable

    if chull_mask is not None:
        # External mask over all n neurons; intersect with unstable.
        cm = chull_mask.to(device=device, dtype=torch.bool).view(-1)
        in_unst = cm[unstable_idx]
        selected_local = torch.nonzero(in_unst, as_tuple=False).view(-1)
    elif top_k is not None and top_k > 0:
        kk = min(int(top_k), k)
        _, selected_local = torch.topk(width_uns, kk)
    else:
        selected_local = torch.zeros(0, dtype=torch.long, device=device)

    n_sel = int(selected_local.numel())

    if n_sel == 0:
        # Pure triangle: no extra constraints needed.
        return HZono(
            c=c_out, Gc=Gc_out, Gb=Gb_out,
            Ac=Ac0, Ab=Ab0, b=b0,
            eq_mask=em_old.clone(),
        )

    # ── 2 * n_sel chull-tightening inequality rows ──
    sel_orig_idx = unstable_idx[selected_local]   # original neuron indices
    sel_lam = lam[selected_local]
    sel_mu = mu[selected_local]
    sel_c = hz.c[sel_orig_idx, 0]

    row1 = 2 * torch.arange(n_sel, device=device)
    row2 = row1 + 1
    eps_col = ng0 + selected_local                 # the new eps gen for each selected

    Ac_cut = torch.zeros((2 * n_sel, ng), device=device, dtype=dtype)
    Ab_cut = torch.zeros((2 * n_sel, nb0), device=device, dtype=dtype)
    b_cut = torch.zeros(2 * n_sel, device=device, dtype=dtype)

    # row1: (1-lam) Gc_i . xi_c + (1-lam) Gb_i . xi_b - mu eps_i <= mu - (1-lam) c
    one_minus_lam = (1.0 - sel_lam)
    if ng0 > 0:
        Ac_cut[row1[:, None], torch.arange(ng0, device=device)[None, :]] = \
            one_minus_lam.unsqueeze(1) * hz.Gc[sel_orig_idx, :]
        Ac_cut[row2[:, None], torch.arange(ng0, device=device)[None, :]] = \
            -sel_lam.unsqueeze(1) * hz.Gc[sel_orig_idx, :]
    Ac_cut[row1, eps_col] = -sel_mu
    Ac_cut[row2, eps_col] = -sel_mu

    if nb0 > 0:
        Ab_cut[row1[:, None], torch.arange(nb0, device=device)[None, :]] = \
            one_minus_lam.unsqueeze(1) * hz.Gb[sel_orig_idx, :]
        Ab_cut[row2[:, None], torch.arange(nb0, device=device)[None, :]] = \
            -sel_lam.unsqueeze(1) * hz.Gb[sel_orig_idx, :]

    b_cut[row1] = sel_mu - one_minus_lam * sel_c
    b_cut[row2] = sel_lam * sel_c + sel_mu
    b_cut = b_cut.view(-1, 1)

    newAc = torch.cat([Ac0, Ac_cut], dim=0)
    newAb = torch.cat([Ab0, Ab_cut], dim=0)
    newb = torch.cat([b0, b_cut], dim=0)
    em_new = torch.cat(
        [em_old, torch.zeros(2 * n_sel, dtype=torch.bool, device=device)],
        dim=0,
    )

    return HZono(
        c=c_out, Gc=Gc_out, Gb=Gb_out,
        Ac=newAc, Ab=newAb, b=newb,
        eq_mask=em_new,
    )


# ---------------------------------------------------------------------------
# Method dispatcher (memaware)
# ---------------------------------------------------------------------------


def _estimate_eq_native_peak_bytes(hz: HZono) -> int:
    """Conservative estimate of peak bytes for ``hz_apply_relu`` (eq_native).

    Per unstable neuron adds 4 cont + 1 binary gen + 3 eq rows. We
    can't know k a priori without computing bounds, so use a rough
    proxy: assume HALF the neurons could be unstable.
    """
    n = int(hz.c.shape[0])
    ng = int(hz.Gc.shape[1])
    nb = int(hz.Gb.shape[1])
    nc = int(hz.b.shape[0])
    elem = 8  # float64
    k_est = n // 2
    ng_new = ng + 4 * k_est
    nb_new = nb + k_est
    nc_new = nc + 3 * k_est
    # Final HZ matrix sizes.
    bytes_Gc = n * ng_new * elem
    bytes_Gb = n * nb_new * elem
    bytes_Ac = nc_new * ng_new * elem
    bytes_Ab = nc_new * nb_new * elem
    return bytes_Gc + bytes_Gb + bytes_Ac + bytes_Ab


def _get_free_gpu_bytes(device: torch.device) -> Optional[int]:
    """Query free GPU memory. Returns None on CPU device or query failure."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        free_b, _ = torch.cuda.mem_get_info(device)
        return int(free_b)
    except Exception:
        return None


def pick_relu_method_for_layer(
    hz: HZono,
    *,
    cascade_late: bool = True,
    env_budget_gb: Optional[float] = None,
    env_reserve_gb: float = 1.0,
) -> str:
    """Return the encoding name to use for this layer's ReLU.

    Decision tree (mirrors HyZor ``_apply_relu_v8_memaware``):

      - If cascade says "use early method" (not in last-k tail): return
        ``"triangle"`` regardless of memory.
      - If on CUDA and free memory minus reserve is below
        ``_estimate_eq_native_peak_bytes(hz)``: return ``"triangle"``
        (downgrade to memory-friendly).
      - Otherwise: return ``"eq_native"`` (the tight encoding).

    Args:
        hz: input HZono (needed to estimate peak bytes).
        cascade_late: True if the cascade controller placed this ReLU
            in the "late" (precision-critical) bucket. False forces
            triangle regardless of memory.
        env_budget_gb: optional fixed cap on per-call memory budget.
            If set, the actual budget is ``min(free - reserve, cap)``.
            Mirrors ``HYZOR_V8_MEM_BUDGET_GB``.
        env_reserve_gb: headroom to keep free for allocator / kernels.
    """
    if not cascade_late:
        return "triangle"

    device = hz.c.device
    free_b = _get_free_gpu_bytes(device)
    if free_b is None:
        # CPU or unable to query: assume eq_native is OK.
        return "eq_native"

    reserve_b = int(env_reserve_gb * (1024.0 ** 3))
    budget_b = max(256 * 1024 * 1024, free_b - reserve_b)
    if env_budget_gb is not None:
        cap_b = int(max(0.25, env_budget_gb) * (1024.0 ** 3))
        budget_b = min(budget_b, cap_b)
    budget_b = int(max(128 * 1024 * 1024, budget_b))

    est_b = _estimate_eq_native_peak_bytes(hz)
    if est_b > budget_b:
        return "triangle"
    return "eq_native"
