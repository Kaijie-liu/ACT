#===- act/back_end/hybridz_tf/algorithms/sgm.py - Shared Generator Merge -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HZ addition with prefix-shared continuous generators. For ResNet-style
#   residuals y = x + f(x) where x and f(x) share input-pixel ancestry,
#   this avoids the ng-doubling of naive minkowski_sum.
#
#===---------------------------------------------------------------------===#

"""Shared Generator Merge (SGM) for ResNet-style residual ``y = x + f(x)``.

Background
----------
Standard HZ addition (``hz_minkowski_sum``) treats the two summands as
independent and concatenates their generator matrices, which doubles
``ng`` at every skip-connection. When ``x`` and ``f(x)`` descend from a
common upstream HZ and inherit the same continuous-generator factors,
that doubling is over-approximation: the sum ``x + f(x)`` should still
reference the SAME underlying generators, not two independent copies.

This module provides the SGM detection (``shares_generator``) and a
merge routine (``hz_sgm_add``) that keeps a single shared block when
the two summands' ``Gc`` matrices are pointwise equal (the strong form
of "fully shared", appropriate for HZ representations that have not
been transformed apart on either branch). For the broader notion of
shared ancestry that survives transformation, callers must track
``_base_ng`` / ``_base_nb`` and call into the HZ class's own ``add``
method (HyZor's HybridZonotope has this; ACT's HZono does not yet).
"""

from __future__ import annotations
import torch
from act.back_end.solver.solver_hz import HZono, hz_minkowski_sum


def shares_generator(hz_x: HZono, hz_y: HZono, *, tol: float = 1e-12) -> bool:
    """Detect whether two HZs share the entire continuous-generator block.

    Returns True iff ``hz_x.Gc`` and ``hz_y.Gc`` have the same shape AND
    are pointwise equal to within ``tol``. An all-zero (or empty) Gc on
    either side returns False (no meaningful sharing).

    The all-zero / empty-Gc guard is required because torch's
    ``.abs().max()`` raises ``RuntimeError`` on a zero-element tensor;
    the equivalent guard in HyZor's repo (``__init__.py`` line ~1314)
    was a latent crash discovered when the ResNet ``add`` dispatch hit
    a degenerate HZ. Keep this guard in any SGM port.
    """
    if hz_x.Gc.shape != hz_y.Gc.shape:
        return False
    if hz_x.Gc.numel() == 0:
        return False
    return bool((hz_x.Gc - hz_y.Gc).abs().max().item() < tol)


def hz_sgm_add(hz_x: HZono, hz_y: HZono) -> HZono:
    """Sum two HZs with PREFIX-SHARED generator merge. Faithful port of
    HyZor ``HybridZonotope.add`` (HybridZonotope.py:6402).

    For HZs that descend from a common ancestor (tracked via ``_base_ng``
    / ``_base_nb``), the first ``min(x._base_ng, y._base_ng)`` continuous
    generators reference the **same latent factors** ξ_c[:shared_ng].
    Adding the two HZs is therefore **algebraic**:

        Gc_shared = x.Gc[:, :shared_ng] + y.Gc[:, :shared_ng]
        Gc_tail_x = x.Gc[:, shared_ng:]    (independent)
        Gc_tail_y = y.Gc[:, shared_ng:]    (independent)
        new_Gc = [Gc_shared | Gc_tail_x | Gc_tail_y]

    This is NOT an over-approximation — it's the exact result of summing
    two zonotopes that share input-pixel factors. Falling back to plain
    Minkowski sum here (concatenating ALL gens) would inflate ng by
    duplicating the shared base factor.

    The previous ACT port only triggered SGM when ``Gc`` matrices were
    BIT-IDENTICAL across operands, which essentially never holds for
    residual blocks (the main branch grows ng via ReLU). Always taking
    the minkowski fallback caused ng to balloon through every resnet
    add block — ~50% larger ng on cifar med, blowing the LP budget.
    """
    nx = int(hz_x.c.shape[0])
    ny = int(hz_y.c.shape[0])
    if nx != ny:
        raise ValueError(f"hz_sgm_add: shape mismatch {nx} vs {ny}")

    dtype = hz_x.c.dtype
    device = hz_x.c.device
    z = torch.zeros

    ng_x = int(hz_x.Gc.shape[1])
    ng_y = int(hz_y.Gc.shape[1])
    nb_x = int(hz_x.Gb.shape[1])
    nb_y = int(hz_y.Gb.shape[1])
    nc_x = int(hz_x.Ac.shape[0])
    nc_y = int(hz_y.Ac.shape[0])

    bx_ng = int(getattr(hz_x, "_base_ng", ng_x))
    by_ng = int(getattr(hz_y, "_base_ng", ng_y))
    bx_nb = int(getattr(hz_x, "_base_nb", nb_x))
    by_nb = int(getattr(hz_y, "_base_nb", nb_y))

    # Cap shared length at the actual generator counts (defensive).
    shared_ng = min(bx_ng, by_ng, ng_x, ng_y)
    shared_nb = min(bx_nb, by_nb, nb_x, nb_y)

    if shared_ng > 0:
        Gc_shared = (
            hz_x.Gc[:, :shared_ng]
            + hz_y.Gc[:, :shared_ng].to(dtype=dtype, device=device)
        )
    else:
        Gc_shared = z(nx, 0, device=device, dtype=dtype)

    if shared_nb > 0:
        Gb_shared = (
            hz_x.Gb[:, :shared_nb]
            + hz_y.Gb[:, :shared_nb].to(dtype=dtype, device=device)
        )
    else:
        Gb_shared = z(nx, 0, device=device, dtype=dtype)

    Gc_A = hz_x.Gc[:, shared_ng:]
    Gc_B = hz_y.Gc[:, shared_ng:].to(dtype=dtype, device=device)
    Gb_A = hz_x.Gb[:, shared_nb:]
    Gb_B = hz_y.Gb[:, shared_nb:].to(dtype=dtype, device=device)

    new_Gc = torch.cat([Gc_shared, Gc_A, Gc_B], dim=1)
    new_Gb = torch.cat([Gb_shared, Gb_A, Gb_B], dim=1)
    new_c = hz_x.c + hz_y.c.to(dtype=dtype, device=device)

    ng_new = int(new_Gc.shape[1])
    nb_new = int(new_Gb.shape[1])
    ng_A_indep = ng_x - shared_ng
    ng_B_indep = ng_y - shared_ng
    nb_A_indep = nb_x - shared_nb
    nb_B_indep = nb_y - shared_nb

    # Block-pad constraint rows: x's Ac references [shared | x's tail | 0...],
    # y's Ac references [shared | 0... | y's tail].
    if nc_x > 0:
        Ac_self = torch.cat(
            [hz_x.Ac, z(nc_x, ng_B_indep, device=device, dtype=dtype)], dim=1
        )
        Ab_self = torch.cat(
            [hz_x.Ab, z(nc_x, nb_B_indep, device=device, dtype=dtype)], dim=1
        )
    else:
        Ac_self = z(0, ng_new, device=device, dtype=dtype)
        Ab_self = z(0, nb_new, device=device, dtype=dtype)

    if nc_y > 0:
        Ac_other = torch.cat([
            hz_y.Ac[:, :shared_ng].to(dtype=dtype, device=device),
            z(nc_y, ng_A_indep, device=device, dtype=dtype),
            hz_y.Ac[:, shared_ng:].to(dtype=dtype, device=device),
        ], dim=1)
        Ab_other = torch.cat([
            hz_y.Ab[:, :shared_nb].to(dtype=dtype, device=device),
            z(nc_y, nb_A_indep, device=device, dtype=dtype),
            hz_y.Ab[:, shared_nb:].to(dtype=dtype, device=device),
        ], dim=1)
    else:
        Ac_other = z(0, ng_new, device=device, dtype=dtype)
        Ab_other = z(0, nb_new, device=device, dtype=dtype)

    new_Ac = torch.cat([Ac_self, Ac_other], dim=0)
    new_Ab = torch.cat([Ab_self, Ab_other], dim=0)
    new_b = torch.cat([hz_x.b, hz_y.b.to(dtype=dtype, device=device)], dim=0)

    eq1 = (hz_x.eq_mask if (hz_x.eq_mask is not None and int(hz_x.eq_mask.numel()) == nc_x)
           else torch.zeros(nc_x, dtype=torch.bool, device=device))
    eq2 = (hz_y.eq_mask if (hz_y.eq_mask is not None and int(hz_y.eq_mask.numel()) == nc_y)
           else torch.zeros(nc_y, dtype=torch.bool, device=device))

    out = HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=new_Ac, Ab=new_Ab, b=new_b,
        eq_mask=torch.cat([eq1, eq2], dim=0),
    )
    # _base_ng / _base_nb tracking: the output's shared block has size
    # ``shared_ng`` (matches HyZor:6467-6468). Use object.__setattr__ to
    # bypass the dataclass post-init default that picks new_Gc.shape[1].
    object.__setattr__(out, "_base_ng", int(shared_ng))
    object.__setattr__(out, "_base_nb", int(shared_nb))
    return out
