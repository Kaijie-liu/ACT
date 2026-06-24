# ===- act/back_end/hybridz_tf/algorithms/order_reduce.py - HZ order reduce -=#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
"""Lifted Girard order reduction for HZono (Bird PhD 2022, Section 6.2.3).

Standard Girard reduction on a CONSTRAINED zonotope is unsound if the dropped
generators are simply deleted together with the constraint rows that reference
them (that silently enlarges the set in an uncontrolled way and collapses the
HZ toward its interval hull). Bird's fix is to LIFT the constraints into the
generator matrix and reduce jointly:

    M = [Gc; Ac]   (an (n+nc) x ng matrix: value rows stacked over constraint rows)

The lowest-L1-norm columns of ``M`` are box-merged into an INDEPENDENT diagonal
generator per lifted dimension. Because the merge happens in the lifted space,
each dropped generator's contribution to the value AND to every constraint is
over-approximated (the value/constraint contributions are decoupled into
independent box factors) -- a sound over-approximation that KEEPS all
constraints. Splitting ``M`` back gives the reduced ``Gc`` / ``Ac``.

eq_mask (constraint senses) is untouched (rows are preserved, only re-expressed).
col_ids: surviving generators keep their ids; the diagonal box generators are
fresh factors.
"""
from __future__ import annotations
import torch

from act.back_end.solver.solver_hz import HZono


def hz_remove_redundancy(hz: HZono, *, tol: float = 1e-9,
                         parallel: bool = True) -> HZono:
    """EXACT redundancy removal (Bird PhD 6.1 / zonopy redundant_*_hz). Zero
    over-approximation -- the represented set is unchanged, only its description
    shrinks. Apply FIRST (free), before any lossy reduction. Three passes:

      A. drop continuous/binary generators that are zero in the lifted [G; A]
         (they affect neither the value nor any constraint);
      B. merge generators that are parallel in lifted [Gc; Ac] (their ranges
         add exactly: a single generator of magnitude sum|.| reproduces the
         combined extent over one factor) -- only when ``parallel`` and ng is
         moderate (the grouping is O(ng*(n+nc)));
      C. drop all-zero constraint rows (0 = 0) and rows parallel to another.

    Preserves eq_mask; surviving generators keep their ids, merged generators
    get fresh ids (a merged factor is a new latent factor).
    """
    n = int(hz.c.shape[0])
    ng = int(hz.Gc.shape[1])
    nb = int(hz.Gb.shape[1])
    nc = int(hz.Ac.shape[0])
    device, dtype = hz.c.device, hz.c.dtype
    Gc, Gb, Ac, Ab, b = hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.b
    col_ids, bcol_ids, eq_mask = hz.col_ids, hz.bcol_ids, hz.eq_mask

    # --- A: drop zero generators (zero in BOTH value and constraints) ---
    if ng > 0:
        mass = Gc.abs().sum(dim=0)
        if nc > 0:
            mass = mass + Ac.abs().sum(dim=0)
        keep = mass > tol
        if not bool(keep.all()):
            Gc = Gc[:, keep]
            # Filter Ac columns too, ALWAYS (even nc==0): Ac is [nc, ng] and must
            # keep Ac.shape[1] == Gc.shape[1]. The nc==0 case is [0, ng] -> [0,
            # ng_new] (cheap); skipping it leaves a malformed HZ (Ac claims more
            # generators than exist) that crashes hz_sgm_add/minkowski downstream.
            Ac = Ac[:, keep]
            col_ids = col_ids[keep] if col_ids is not None else None
            ng = int(Gc.shape[1])
    if nb > 0:
        mass = Gb.abs().sum(dim=0)
        if nc > 0:
            mass = mass + Ab.abs().sum(dim=0)
        keep = mass > tol
        if not bool(keep.all()):
            Gb = Gb[:, keep]
            Ab = Ab[:, keep]
            bcol_ids = bcol_ids[keep] if bcol_ids is not None else None
            nb = int(Gb.shape[1])

    # --- B: merge parallel continuous generators in lifted [Gc; Ac] ---
    # Hash-based grouping (O(ng) keys), so it scales to large nets.
    if (parallel and ng > 1 and ng <= _PARALLEL_MAX
            and _fits_parallel_merge(n + nc, ng)):
        Mc = torch.cat([Gc, Ac], dim=0) if nc > 0 else Gc
        norms = Mc.norm(dim=0)
        nz = norms > tol
        keys, sign = _canonical_keys(Mc, norms, tol)
        groups = {}
        for j in range(ng):
            if bool(nz[j]):
                groups.setdefault(keys[j], []).append(j)
        if any(len(v) > 1 for v in groups.values()):
            from act.back_end.solver.solver_hz import _fresh_col_ids
            units = Mc / norms.clamp_min(tol)
            new_cols, keep_id, n_fresh = [], [], 0
            for js in groups.values():
                if len(js) == 1:
                    new_cols.append(Mc[:, js[0]])
                    keep_id.append(int(col_ids[js[0]]) if col_ids is not None else -1)
                else:
                    new_cols.append(units[:, js[0]] * sign[js[0]] * norms[js].sum())
                    keep_id.append(None); n_fresh += 1
            Mc_new = torch.stack(new_cols, dim=1)
            Gc = Mc_new[:n]
            Ac = Mc_new[n:] if nc > 0 else hz.c.new_zeros(0, Mc_new.shape[1])
            ng = int(Gc.shape[1])
            if col_ids is not None:
                fresh = _fresh_col_ids(n_fresh, device=device).tolist()
                fi = 0; ids = []
                for v in keep_id:
                    if v is None:
                        ids.append(fresh[fi]); fi += 1
                    else:
                        ids.append(v)
                col_ids = torch.tensor(ids, dtype=torch.long, device=device)

    # --- C: drop zero + duplicate constraint rows (hash-based, O(nc)) ---
    if (nc > 0 and nc <= _PARALLEL_MAX
            and _fits_parallel_merge(nc, Ac.shape[1] + Ab.shape[1] + 1)):
        A_full = torch.cat([Ac, Ab, b], dim=1)
        rnorm = A_full.norm(dim=1)
        coeff_norm = (Ac.abs().sum(1) + Ab.abs().sum(1))
        keys, _ = _canonical_keys(A_full.T, rnorm, tol)
        # Include the constraint SENSE in the dedup key: an equality (== b) and an
        # inequality (<= b) row with identical coefficients mean different things
        # and must NOT be merged. (No-op today since all rows are equalities, but
        # defensive for any future le-row producer.)
        if eq_mask is not None:
            senses = eq_mask.tolist()
            keys = [(bool(senses[k]), keys[k]) for k in range(nc)]
        keep_rows, seen = [], set()
        for k in range(nc):
            if float(coeff_norm[k]) <= tol:
                continue  # 0 = b (b~0 for a feasible HZ) -> redundant
            if keys[k] in seen:
                continue
            seen.add(keys[k]); keep_rows.append(k)
        if len(keep_rows) < nc:
            idx = torch.tensor(keep_rows, dtype=torch.long, device=device)
            Ac = Ac[idx]; Ab = Ab[idx]; b = b[idx]
            eq_mask = eq_mask[idx] if eq_mask is not None else None
            nc = int(Ac.shape[0])

    return HZono(c=hz.c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b,
                 eq_mask=eq_mask, col_ids=col_ids, bcol_ids=bcol_ids)


_PARALLEL_MAX = 20000  # skip the (still O(ncols)) dedup passes above this size
_PARALLEL_CELL_MAX = 25_000_000  # avoid multi-GB canonicalization tensors


def _fits_parallel_merge(nrow: int, ncol: int) -> bool:
    return int(nrow) * int(ncol) <= _PARALLEL_CELL_MAX


def _canonical_keys(M, norms, tol):
    """Hashable direction keys for columns of M (sign-canonicalized unit vectors,
    rounded). Parallel/anti-parallel columns share a key. Returns (keys, sign)."""
    units = M / norms.clamp_min(tol)
    # sign = sign of the first significant entry of each column (for canonicalization)
    absu = units.abs()
    sig = absu > 1e-6
    first = torch.argmax(sig.to(torch.int8), dim=0)
    cols = torch.arange(units.shape[1], device=M.device)
    fv = units[first, cols]
    sign = torch.where(fv < 0, -1.0, 1.0)
    canon = (units * sign).round(decimals=6)
    rounded = canon.T.tolist()
    return [tuple(r) for r in rounded], sign


def hz_girard_reduce(hz: HZono, target_ng: int) -> HZono:
    """Reduce continuous generators toward ``target_ng`` via lifted Girard.

    Returns ``hz`` unchanged when reduction would not actually shrink ng (e.g.
    when the constraint count makes the lifted box larger than the current ng);
    keeping the exact set is sound and avoids growth.
    """
    n = int(hz.c.shape[0])
    ng = int(hz.Gc.shape[1])
    nc = int(hz.Ac.shape[0])
    if ng == 0 or ng <= target_ng:
        return hz
    device = hz.c.device

    # Lifted continuous generator matrix [Gc; Ac].
    M = torch.cat([hz.Gc, hz.Ac], dim=0) if nc > 0 else hz.Gc
    nl = M.shape[0]  # n + nc
    keep_count = max(target_ng - nl, 0)
    new_ng = keep_count + nl
    if new_ng >= ng:
        # Lifted box (nl cols) would not reduce ng -> keep exact (sound).
        return hz

    col_norms = M.abs().sum(dim=0)
    order = torch.argsort(col_norms, descending=True)
    keep_idx = order[:keep_count]
    drop_idx = order[keep_count:]
    box_widths = M[:, drop_idx].abs().sum(dim=1)  # (nl,)
    M_new = torch.cat([M[:, keep_idx], torch.diag(box_widths)], dim=1)
    new_Gc = M_new[:n]
    new_Ac = M_new[n:] if nc > 0 else hz.c.new_zeros(0, M_new.shape[1])

    new_col_ids = None
    if hz.col_ids is not None:
        from act.back_end.solver.solver_hz import _fresh_col_ids
        new_col_ids = torch.cat(
            [hz.col_ids[keep_idx.to(hz.col_ids.device)],
             _fresh_col_ids(nl, device=device)])

    return HZono(
        c=hz.c, Gc=new_Gc, Gb=hz.Gb, Ac=new_Ac, Ab=hz.Ab, b=hz.b,
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=new_col_ids,
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


__all__ = ["hz_remove_redundancy", "hz_girard_reduce"]
