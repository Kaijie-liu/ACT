# ===- act/back_end/hybridz_tf/algorithms/sgm.py - Shared Generator Merge -==#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
"""Shared-Generator Merge (SGM) for ResNet-style residual ``y = x + f(x)``.

Plain ``hz_minkowski_sum`` treats the two summands as INDEPENDENT and
concatenates their generator matrices, doubling ``ng`` at every skip
connection AND losing the correlation that ``x`` and ``f(x)`` both descend
from the same input factors. That is a (sound) over-approximation.

``hz_sgm_add`` instead aligns generators by their factor ``col_ids``: a
factor id present in BOTH operands is the SAME latent ``xi`` (ids are
globally-unique-monotonic — see ``solver_hz.hz_fresh_col_ids`` — so distinct
factors can never collide), so its two generator columns are ADDED; factors
present in only one operand are concatenated as independent. The result is
the EXACT sum ``{x + y}`` for shared-ancestry zonotopes, not an
over-approximation.

Soundness rests entirely on the id invariant: same id <=> same factor.
That is guaranteed by the monotonic id source + the propagation contract
(affine ops preserve ids; ReLU appends fresh ids; a fork inherits the
parent's ids by reference). If either operand is id-untracked we fall back
to the sound ``hz_minkowski_sum``.
"""
from __future__ import annotations
import torch

from act.back_end.solver.solver_hz import (
    HZono,
    hz_inherit_known_nonempty,
    hz_minkowski_sum,
)


def _align(ids_x: torch.Tensor, ids_y: torch.Tensor,
           Gx: torch.Tensor, Gy: torch.Tensor):
    """Merge two generator blocks by factor id.

    Returns (G_merged (n, n_merged), merged_ids (n_merged,),
             x_map (ng_x,), y_map (ng_y,)) where ``*_map[j]`` is the merged
    column index that operand column ``j`` lands in. Shared ids accumulate
    both operands' columns.
    """
    n = Gx.shape[0]
    dtype, device = Gx.dtype, Gx.device
    lx = ids_x.tolist()
    ly = ids_y.tolist()
    pos: dict = {}
    merged_ids: list = []
    # x ids first (in x order: shared + x-only), then y-only ids.
    for idv in lx:
        if idv not in pos:
            pos[idv] = len(merged_ids)
            merged_ids.append(idv)
    for idv in ly:
        if idv not in pos:
            pos[idv] = len(merged_ids)
            merged_ids.append(idv)
    n_merged = len(merged_ids)
    x_map = torch.tensor([pos[v] for v in lx], dtype=torch.long, device=device)
    y_map = torch.tensor([pos[v] for v in ly], dtype=torch.long, device=device)
    G = torch.zeros(n, n_merged, dtype=dtype, device=device)
    if Gx.shape[1]:
        G.index_add_(1, x_map, Gx)
    if Gy.shape[1]:
        G.index_add_(1, y_map, Gy.to(dtype=dtype, device=device))
    merged = torch.tensor(merged_ids, dtype=torch.long, device=device)
    return G, merged, x_map, y_map


def _scatter_cols(A: torch.Tensor, col_map: torch.Tensor, n_merged: int) -> torch.Tensor:
    """Place A's columns into a width-``n_merged`` matrix at ``col_map`` positions
    (col_map entries are distinct because each operand's ids are unique)."""
    out = A.new_zeros(A.shape[0], n_merged)
    if A.shape[1]:
        out[:, col_map] = A
    return out


def hz_sgm_add(hz_x: HZono, hz_y: HZono) -> HZono:
    """Exact sum of two HZs that may share generator factors (by ``col_ids``)."""
    if hz_x.col_ids is None or hz_y.col_ids is None:
        return hz_minkowski_sum(hz_x, hz_y)
    n = int(hz_x.c.shape[0])
    if int(hz_y.c.shape[0]) != n:
        raise ValueError(f"hz_sgm_add: shape mismatch {n} vs {hz_y.c.shape[0]}")
    dtype, device = hz_x.c.dtype, hz_x.c.device

    bx = (hz_x.bcol_ids if hz_x.bcol_ids is not None
          else torch.zeros(0, dtype=torch.long, device=device))
    by = (hz_y.bcol_ids if hz_y.bcol_ids is not None
          else torch.zeros(0, dtype=torch.long, device=device))

    Gc, cids, xc_map, yc_map = _align(hz_x.col_ids, hz_y.col_ids, hz_x.Gc, hz_y.Gc)
    Gb, bids, xb_map, yb_map = _align(bx, by, hz_x.Gb, hz_y.Gb)
    ngm = Gc.shape[1]
    nbm = Gb.shape[1]

    # Remap each operand's constraint rows into the merged column layout.
    Ac_x = _scatter_cols(hz_x.Ac, xc_map, ngm)
    Ac_y = _scatter_cols(hz_y.Ac.to(dtype=dtype, device=device), yc_map, ngm)
    Ab_x = _scatter_cols(hz_x.Ab, xb_map, nbm)
    Ab_y = _scatter_cols(hz_y.Ab.to(dtype=dtype, device=device), yb_map, nbm)

    new_Ac = torch.cat([Ac_x, Ac_y], dim=0)
    new_Ab = torch.cat([Ab_x, Ab_y], dim=0)
    new_b = torch.cat([hz_x.b, hz_y.b.to(dtype=dtype, device=device)], dim=0)

    nc_x = int(hz_x.Ac.shape[0])
    nc_y = int(hz_y.Ac.shape[0])
    if hz_x.eq_mask is None and hz_y.eq_mask is None:
        new_eq_mask = None
    else:
        mx = (hz_x.eq_mask if hz_x.eq_mask is not None
              else torch.ones(nc_x, dtype=torch.bool, device=device))
        my = (hz_y.eq_mask if hz_y.eq_mask is not None
              else torch.ones(nc_y, dtype=torch.bool, device=device))
        new_eq_mask = torch.cat([mx.to(device), my.to(device)], dim=0)

    return hz_inherit_known_nonempty(HZono(
        c=hz_x.c + hz_y.c.to(dtype=dtype, device=device),
        Gc=Gc, Gb=Gb, Ac=new_Ac, Ab=new_Ab, b=new_b,
        eq_mask=new_eq_mask, col_ids=cids, bcol_ids=bids,
    ), hz_x, hz_y, reason="sgm_add")


def hz_negate(hz: HZono) -> HZono:
    """-hz: flip the center + generators (constraints unchanged, ids preserved)."""
    return hz_inherit_known_nonempty(HZono(
        c=-hz.c, Gc=-hz.Gc, Gb=-hz.Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    ), hz, reason="negate")


def hz_sub(hz_x: HZono, hz_y: HZono) -> HZono:
    """x - y as a share-merged sum x + (-y): correlated subtraction is EXACT
    (e.g. x - x = 0 exactly when they share factor ids), not interval-loose."""
    return hz_sgm_add(hz_x, hz_negate(hz_y))


def hz_concat(parts) -> "HZono | None":
    """Concatenate HZs along the output dimension (stack rows). When the parts
    share factor ids (common input ancestry) the columns are ALIGNED so the
    correlation between the stacked blocks is preserved exactly; the feasible
    factor set is the intersection of the parts' constraints (row-stacked).
    Falls back to a block-diagonal (independent-factor) stack when ids are
    untracked — sound, but loses cross-block correlation.
    """
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    dtype, device = parts[0].c.dtype, parts[0].c.device
    if any(p.col_ids is None for p in parts):
        return _hz_concat_independent(parts)

    # Union continuous + binary ids across all parts (stable first-seen order).
    cpos, cids = {}, []
    for p in parts:
        for idv in p.col_ids.tolist():
            if idv not in cpos:
                cpos[idv] = len(cids); cids.append(idv)
    bpos, bids = {}, []
    for p in parts:
        pb = p.bcol_ids if p.bcol_ids is not None else torch.zeros(0, dtype=torch.long)
        for idv in pb.tolist():
            if idv not in bpos:
                bpos[idv] = len(bids); bids.append(idv)
    ngm, nbm = len(cids), len(bids)

    cs, Gcs, Gbs, Acs, Abs_, bs, eqs = [], [], [], [], [], [], []
    for p in parts:
        cmap = torch.tensor([cpos[v] for v in p.col_ids.tolist()],
                            dtype=torch.long, device=device)
        pb = (p.bcol_ids if p.bcol_ids is not None
              else torch.zeros(0, dtype=torch.long, device=device))
        bmap = torch.tensor([bpos[v] for v in pb.tolist()],
                            dtype=torch.long, device=device)
        n_p = p.c.shape[0]
        Gc_p = p.c.new_zeros(n_p, ngm)
        if p.Gc.shape[1]:
            Gc_p[:, cmap] = p.Gc
        Gb_p = p.c.new_zeros(n_p, nbm)
        if p.Gb.shape[1] and nbm:
            Gb_p[:, bmap] = p.Gb.to(dtype=dtype, device=device)
        cs.append(p.c.to(dtype=dtype, device=device))
        Gcs.append(Gc_p); Gbs.append(Gb_p)
        nc_p = p.Ac.shape[0]
        Ac_p = p.Ac.new_zeros(nc_p, ngm)
        if p.Ac.shape[1]:
            Ac_p[:, cmap] = p.Ac
        Ab_p = p.Ab.new_zeros(nc_p, nbm)
        if p.Ab.shape[1] and nbm:
            Ab_p[:, bmap] = p.Ab.to(dtype=dtype, device=device)
        Acs.append(Ac_p); Abs_.append(Ab_p)
        bs.append(p.b.to(dtype=dtype, device=device))
        eqs.append(p.eq_mask if p.eq_mask is not None
                   else torch.ones(nc_p, dtype=torch.bool, device=device))
    eq_mask = torch.cat(eqs) if any(e is not None for e in eqs) else None
    return hz_inherit_known_nonempty(HZono(
        c=torch.cat(cs, 0), Gc=torch.cat(Gcs, 0), Gb=torch.cat(Gbs, 0),
        Ac=torch.cat(Acs, 0), Ab=torch.cat(Abs_, 0), b=torch.cat(bs, 0),
        eq_mask=eq_mask,
        col_ids=torch.tensor(cids, dtype=torch.long, device=device),
        bcol_ids=torch.tensor(bids, dtype=torch.long, device=device),
    ), *parts, reason="concat")


def _hz_concat_independent(parts) -> HZono:
    """Block-diagonal concat (parts treated as independent). Sound over-approx
    of concat when factor ids are unknown."""
    dtype, device = parts[0].c.dtype, parts[0].c.device
    ng_tot = sum(int(p.Gc.shape[1]) for p in parts)
    nb_tot = sum(int(p.Gb.shape[1]) for p in parts)
    nc_tot = sum(int(p.Ac.shape[0]) for p in parts)
    cs, Gcs, Gbs = [], [], []
    Ac = torch.zeros(nc_tot, ng_tot, dtype=dtype, device=device)
    Ab = torch.zeros(nc_tot, nb_tot, dtype=dtype, device=device)
    bs, eqs = [], []
    goff = boff = roff = 0
    for p in parts:
        n_p, ng_p = int(p.c.shape[0]), int(p.Gc.shape[1])
        nb_p, nc_p = int(p.Gb.shape[1]), int(p.Ac.shape[0])
        Gc_p = torch.zeros(n_p, ng_tot, dtype=dtype, device=device)
        Gc_p[:, goff:goff + ng_p] = p.Gc.to(dtype=dtype, device=device)
        Gb_p = torch.zeros(n_p, nb_tot, dtype=dtype, device=device)
        Gb_p[:, boff:boff + nb_p] = p.Gb.to(dtype=dtype, device=device)
        cs.append(p.c.to(dtype=dtype, device=device)); Gcs.append(Gc_p); Gbs.append(Gb_p)
        if nc_p:
            Ac[roff:roff + nc_p, goff:goff + ng_p] = p.Ac.to(dtype=dtype, device=device)
            Ab[roff:roff + nc_p, boff:boff + nb_p] = p.Ab.to(dtype=dtype, device=device)
            bs.append(p.b.to(dtype=dtype, device=device))
            eqs.append(p.eq_mask if p.eq_mask is not None
                       else torch.ones(nc_p, dtype=torch.bool, device=device))
        goff += ng_p; boff += nb_p; roff += nc_p
    b = torch.cat(bs, 0) if bs else torch.zeros(0, 1, dtype=dtype, device=device)
    eq_mask = torch.cat(eqs) if eqs else None
    return hz_inherit_known_nonempty(
        HZono(c=torch.cat(cs, 0), Gc=torch.cat(Gcs, 0), Gb=torch.cat(Gbs, 0),
              Ac=Ac, Ab=Ab, b=b, eq_mask=eq_mask),
        *parts,
        reason="concat_independent",
    )


__all__ = ["hz_sgm_add", "hz_sub", "hz_negate", "hz_concat"]
