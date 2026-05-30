#===- act/back_end/hybridz_tf/algorithms/sparse_eq_lagr.py - B3 sparse eq_lagr ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   B3 — sparse-eq_lagr ReLU on SparseGcZ. Implements the same tight
#   bound as dense eq_lagr_v8 but without densifying Gc/Ac. Uses the
#   STRUCTURED form of eq_lagr's 3 equality rows per unstable neuron
#   to do algebraic elimination of (xi1, xi3, xi4) directly, leaving
#   only (xi2, z) plus 6 inequality rows per unstable neuron.
#
#   This is a HZ-internal representation flavor extension. Stays
#   forward-only, no CROWN, no Gurobi, no fallback, no B&B.
#
#===---------------------------------------------------------------------===#

"""Sparse equality-Lagrangian ReLU for SparseGcZ.

Math
====

Dense ``hz_apply_relu`` (tf_mlp.py:405) adds, per unstable neuron i
with pre-activation bounds ``alpha < 0 < beta``:

  * ng += 4 (xi1, xi2, xi3, xi4)
  * nb += 1 (z, binary in {-1, +1})
  * nc += 3 equality rows

  Row r1: xi1 + xi3 + z = 1               (graph eq 1)
  Row r2: xi2 + xi4 - z = 1               (graph eq 2)
  Row r3: (alpha/2) xi1 - (beta/2) xi2
        - Gc[i,:] xi_old - Gb[i,:] xi_b
        + (alpha/2) z                       (linking eq)
        = c[i] - beta/2

  Output: y[i] = beta/2 + (-beta/2) xi2

The PEE step (project_eq_elim) later QR-eliminates these equality rows
by picking pivot columns. For eq_lagr's structured rows, the pivot
choice is OBVIOUS: r1 → eliminate xi3, r2 → eliminate xi4, r3 →
eliminate xi1 (the only xi with non-trivial Gc/Gb coupling).

This module pre-applies that algebraic elimination directly to the
sparse representation, producing the post-PEE structure without ever
storing the dense Ac.

Post-elimination per unstable neuron:

  * +1 continuous generator (xi2) — KEPT
  * +1 binary generator (z)       — KEPT
  * +6 inequality rows from substituted box constraints
       xi1 ∈ [-1, 1]  →  2 rows on (xi_old, xi_b, xi2, z)
       xi3 ∈ [-1, 1]  →  2 rows on (xi1, z) [substitute xi1] → 2 rows on (xi_old, xi_b, xi2, z)
       xi4 ∈ [-1, 1]  →  2 rows on (xi2, z) [no Gc/Gb coupling]

  The xi3/xi4 box constraints, after substituting xi1's formula, are
  sparse in xi_old (inheriting from Gc[i,:]).

Algebra
=======

From r3 solve xi1:
    xi1 = (2/alpha) * [c[i] - beta/2 + (beta/2) xi2 + Gc[i,:] xi_old
                        + Gb[i,:] xi_b - (alpha/2) z]
        = (2/alpha)*(c[i] - beta/2) + (beta/alpha) xi2
          + (2/alpha) Gc[i,:] xi_old + (2/alpha) Gb[i,:] xi_b - z

Let ``coef_old_c = (2/alpha) Gc[i,:]`` (sparse), ``coef_old_b = (2/alpha) Gb[i,:]``
(sparse), ``coef_xi2 = beta/alpha``, ``coef_z = -1``, ``rhs_xi1 = (2/alpha)*(c[i] - beta/2)``.

So xi1 = rhs_xi1 + coef_old_c · xi_old + coef_old_b · xi_b
         + coef_xi2 * xi2 + coef_z * z.

Box constraint xi1 ≤ 1:
    coef_old_c · xi_old + coef_old_b · xi_b + coef_xi2 xi2 + coef_z z
    ≤ 1 - rhs_xi1

Box constraint xi1 ≥ -1, i.e., -xi1 ≤ 1:
    -coef_old_c · xi_old - coef_old_b · xi_b - coef_xi2 xi2 - coef_z z
    ≤ 1 + rhs_xi1

From r1: xi3 = 1 - xi1 - z. Substitute xi1:
    xi3 = 1 - rhs_xi1 - coef_old_c·xi_old - coef_old_b·xi_b
            - coef_xi2 xi2 - coef_z z - z
        = (1 - rhs_xi1) - coef_old_c·xi_old - coef_old_b·xi_b
            - coef_xi2 xi2 - (coef_z + 1) z

Box constraint xi3 ≤ 1:
    -coef_old_c·xi_old - coef_old_b·xi_b - coef_xi2 xi2 - (coef_z + 1) z
    ≤ rhs_xi1     (i.e. 1 - (1 - rhs_xi1))

Box constraint xi3 ≥ -1, i.e. -xi3 ≤ 1:
    coef_old_c·xi_old + coef_old_b·xi_b + coef_xi2 xi2 + (coef_z + 1) z
    ≤ 2 - rhs_xi1

From r2: xi4 = 1 - xi2 + z. No Gc/Gb coupling.
    Box: -1 ≤ 1 - xi2 + z ≤ 1
       →  xi2 - z ≤ 2  (xi4 ≥ -1)
           -xi2 + z ≤ 0  (xi4 ≤ 1)

Output: y[i] = beta/2 + (-beta/2) xi2.

Soundness
=========

The structured elimination is EXACT — it does not introduce any
over-approximation beyond the dense eq_lagr_v8 + PEE pipeline. The
projection of (xi1, xi2, xi3, xi4, z) ∈ [-1,1]^4 × {-1,+1} satisfying
all 3 equality rows onto (xi2, z, xi_old, xi_b) is exactly described
by these 6 inequality rows (algebraically equivalent to dense PEE on
the same rows). No CROWN, no backward, no Gurobi, no fallback.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.representations import SparseGcZ


# ─── Helpers ───────────────────────────────────────────────────────────


def _row_slice_sparse(sp: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """Return rows of a 2D sparse_coo_tensor in row_idx order. Output is
    coalesced. row_idx is a 1D LongTensor; the result has shape (k, ng)."""
    sp = sp.coalesce()
    n_in, ng = sp.shape
    k = int(row_idx.numel())
    if sp._nnz() == 0 or k == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=sp.device),
            torch.zeros(0, dtype=sp.dtype, device=sp.device),
            (k, ng), dtype=sp.dtype, device=sp.device,
        ).coalesce()
    # Build inverse map row_old -> row_new
    row_map = torch.full((n_in,), -1, dtype=torch.long, device=sp.device)
    row_map[row_idx] = torch.arange(k, dtype=torch.long, device=sp.device)
    ind = sp.indices()
    val = sp.values()
    keep_mask = row_map[ind[0]] >= 0
    if not keep_mask.any():
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=sp.device),
            torch.zeros(0, dtype=sp.dtype, device=sp.device),
            (k, ng), dtype=sp.dtype, device=sp.device,
        ).coalesce()
    kept_ind = ind[:, keep_mask].clone()
    kept_val = val[keep_mask]
    kept_ind[0] = row_map[kept_ind[0]]
    return torch.sparse_coo_tensor(
        kept_ind, kept_val, (k, ng), dtype=sp.dtype, device=sp.device,
    ).coalesce()


def _scale_sparse_rows(sp: torch.Tensor, row_scale: torch.Tensor) -> torch.Tensor:
    """Multiply each row of a 2D sparse_coo_tensor by ``row_scale[row]``.
    row_scale is 1D of length sp.shape[0]."""
    sp = sp.coalesce()
    if sp._nnz() == 0:
        return sp
    ind = sp.indices()
    val = sp.values() * row_scale[ind[0]]
    return torch.sparse_coo_tensor(
        ind, val, sp.shape, dtype=sp.dtype, device=sp.device,
    ).coalesce()


def _set_sparse_col_block(sp: torch.Tensor, col_offset: int, n_cols: int,
                          rows: torch.Tensor, cols_local: torch.Tensor,
                          vals: torch.Tensor) -> torch.Tensor:
    """Return a new sparse_coo_tensor with shape (sp.shape[0],
    sp.shape[1] + n_cols) where the original columns are preserved and
    new entries are placed at (rows, col_offset + cols_local) = vals."""
    sp = sp.coalesce()
    n_rows = sp.shape[0]
    ng_old = sp.shape[1]
    ng_new = ng_old + n_cols
    if vals.numel() > 0:
        new_ind = torch.stack([rows, col_offset + cols_local])
        all_ind = torch.cat([sp.indices(), new_ind], dim=1)
        all_val = torch.cat([sp.values(), vals])
    else:
        all_ind = sp.indices()
        all_val = sp.values()
    return torch.sparse_coo_tensor(
        all_ind, all_val, (n_rows, ng_new), dtype=sp.dtype, device=sp.device,
    ).coalesce()


def _pad_cols(sp: torch.Tensor, extra_cols: int) -> torch.Tensor:
    """Return a sparse_coo_tensor with extra_cols zero columns appended."""
    if extra_cols <= 0:
        return sp.coalesce()
    sp = sp.coalesce()
    n_rows = sp.shape[0]
    ng_old = sp.shape[1]
    return torch.sparse_coo_tensor(
        sp.indices(), sp.values(),
        (n_rows, ng_old + extra_cols),
        dtype=sp.dtype, device=sp.device,
    ).coalesce()


# ─── Main operator ─────────────────────────────────────────────────────


def apply_relu_eq_lagr_sparse(
    hz: SparseGcZ,
    *,
    external_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    compact_rows: bool = False,
) -> SparseGcZ:
    """Sparse equality-Lagrangian ReLU with structured PEE.

    Equivalent to dense ``hz_apply_relu`` (eq_lagr_v8) followed by
    ``project_eq_elim``, but operating on SparseGcZ directly so that
    Gc/Ac never need to be densified. The 3 equality rows added by
    eq_lagr per unstable neuron are immediately substituted out using
    their known algebraic structure (see module docstring).

    Adds per unstable neuron: 1 continuous generator (xi2), 1 binary
    generator (z), 6 inequality rows. Active and inactive neurons pass
    through unchanged (active: y = x; inactive: y = 0).

    Soundness: exact on the (xi1, xi2, xi3, xi4, z) projection — no
    over-approximation introduced beyond what dense eq_lagr_v8 + PEE
    would yield on the same input.
    """
    n = hz.dim
    ng0 = hz.ng
    nb0 = hz.nb
    nc0 = hz.nc
    dtype = hz.dtype
    device = hz.device

    if external_bounds is None:
        lb, ub = hz.bounds()
    else:
        lb, ub = external_bounds
        lb = lb.to(dtype=dtype, device=device).view(-1)
        ub = ub.to(dtype=dtype, device=device).view(-1)
        if int(lb.numel()) != n or int(ub.numel()) != n:
            raise ValueError("sparse eq_lagr external bounds dim mismatch")

    is_active = (lb >= 0)
    is_inactive = (ub <= 0)
    is_unstable = ~(is_active | is_inactive)
    unstable_idx = torch.nonzero(is_unstable, as_tuple=False).view(-1)
    k = int(unstable_idx.numel())

    # ── Output center: 0 on inactive, c on active, beta/2 on unstable ──
    c_out = torch.zeros(n, dtype=dtype, device=device)
    c_out[is_active] = hz.c[is_active]
    if k > 0:
        beta = ub[unstable_idx]
        c_out[unstable_idx] = beta / 2.0

    # ── Output Gc rows ──
    # Active rows pass through old Gc unchanged.
    # Inactive rows go to zero.
    # Unstable rows: y[i] = beta/2 + (-beta/2) xi2_i; old Gc[i,:] is
    # ZEROED (the linking constraint moves it to inequality rows).
    # Pattern: row-scale Gc by (1 if active, 0 if inactive or unstable).
    row_scale_c = torch.zeros(n, dtype=dtype, device=device)
    row_scale_c[is_active] = 1.0
    Gc_out_old = _scale_sparse_rows(hz.Gc_sparse, row_scale_c)
    # Pad with k new columns for xi2.
    Gc_out = _pad_cols(Gc_out_old, k)
    if k > 0:
        new_rows = unstable_idx
        new_cols = torch.arange(k, dtype=torch.long, device=device)
        new_vals = -beta / 2.0
        all_ind = torch.cat([Gc_out.indices(),
                              torch.stack([new_rows, ng0 + new_cols])], dim=1)
        all_val = torch.cat([Gc_out.values(), new_vals])
        Gc_out = torch.sparse_coo_tensor(
            all_ind, all_val, (n, ng0 + k),
            dtype=dtype, device=device,
        ).coalesce()

    # ── Output Gb rows: scale active rows by 1, zero unstable/inactive ──
    Gb_out_old = _scale_sparse_rows(hz.Gb_sparse, row_scale_c)
    Gb_out = _pad_cols(Gb_out_old, k)  # +k new columns for z (binary)
    # Active rows already inherit old Gb. Unstable rows: y[i] doesn't
    # depend on z directly (it's only in the linking eq). No new entry
    # in Gb_out for unstable rows.

    if k == 0:
        return SparseGcZ(
            c=c_out,
            Gc_sparse=Gc_out,
            Gb_sparse=Gb_out,
            dtype=dtype, device=device,
            Ac_sparse=_pad_cols(hz.Ac_sparse, 0),
            Ab_sparse=_pad_cols(hz.Ab_sparse, 0),
            b=hz.b.clone(),
            eq_mask=hz.eq_mask.clone(),
        )

    ng_new = ng0 + k
    nb_new = nb0 + k

    # ── Structured elimination: build the 6 new inequality rows per
    # unstable neuron in (xi_old, xi_b, xi2, z) coordinates. ──
    alpha = lb[unstable_idx]  # < 0
    # Per-neuron scalars
    inv_alpha = 1.0 / alpha                  # (k,)
    coef_xi2 = beta * inv_alpha              # = beta/alpha
    coef_z = -torch.ones(k, dtype=dtype, device=device)  # = -1
    c_unstable = hz.c[unstable_idx]
    rhs_xi1 = 2.0 * inv_alpha * (c_unstable - beta / 2.0)

    # Sparse Gc[unstable, :] / Gb[unstable, :] in (k, ng0)
    Gc_uns_sparse = _row_slice_sparse(hz.Gc_sparse, unstable_idx)
    Gb_uns_sparse = _row_slice_sparse(hz.Gb_sparse, unstable_idx)
    # coef_old_c = (2/alpha) * Gc[unstable, :]
    coef_old_c = _scale_sparse_rows(Gc_uns_sparse, 2.0 * inv_alpha)
    coef_old_b = _scale_sparse_rows(Gb_uns_sparse, 2.0 * inv_alpha)

    # Build 6 row blocks. Each block has k rows referencing:
    #   - xi_old columns 0..ng0-1 (sparse, from coef_old_c)
    #   - xi2_i column ng0+i      (one dense entry per row)
    #   - xi_b columns 0..nb0-1   (sparse, from coef_old_b)
    #   - z_i column nb0+i        (one dense entry per row)
    #
    # All 6 blocks share the same sparsity pattern on xi_old/xi_b (just
    # with a sign flip for some). We construct them as 6 separate sparse
    # matrices and stack.

    def _build_row(scale_c, scale_b, val_xi2, val_z, rhs):
        """Returns (Ac_block, Ab_block, b_block) for k rows where:
        Ac_block[i, j] = scale_c * coef_old_c[i, j]   (j in ng_old)
        Ac_block[i, ng0+i] = val_xi2[i]
        Ab_block[i, j] = scale_b * coef_old_b[i, j]   (j in nb_old)
        Ab_block[i, nb0+i] = val_z[i]
        b_block[i] = rhs[i]
        """
        # Ac block: scale * coef_old_c (k, ng0), then extend to ng_new
        # by appending column for xi2_i per row.
        Ac_main = _scale_sparse_rows(coef_old_c.t() if False else coef_old_c,
                                      torch.full((k,), scale_c, dtype=dtype, device=device))
        Ac_block = _pad_cols(Ac_main, k)  # (k, ng0 + k)
        # Set diagonal xi2_i column entries
        rows = torch.arange(k, dtype=torch.long, device=device)
        cols = ng0 + rows
        val_xi2_t = val_xi2 if isinstance(val_xi2, torch.Tensor) else \
            torch.full((k,), float(val_xi2), dtype=dtype, device=device)
        # Filter out zero values to avoid storing exact zeros
        nz = val_xi2_t != 0
        if nz.any():
            new_ind = torch.stack([rows[nz], cols[nz]])
            Ac_block = torch.sparse_coo_tensor(
                torch.cat([Ac_block.indices(), new_ind], dim=1),
                torch.cat([Ac_block.values(), val_xi2_t[nz]]),
                (k, ng_new), dtype=dtype, device=device,
            ).coalesce()

        Ab_main = _scale_sparse_rows(coef_old_b,
                                      torch.full((k,), scale_b, dtype=dtype, device=device))
        Ab_block = _pad_cols(Ab_main, k)  # (k, nb0 + k)
        val_z_t = val_z if isinstance(val_z, torch.Tensor) else \
            torch.full((k,), float(val_z), dtype=dtype, device=device)
        nz_b = val_z_t != 0
        if nz_b.any():
            cols_b = nb0 + rows
            new_ind_b = torch.stack([rows[nz_b], cols_b[nz_b]])
            Ab_block = torch.sparse_coo_tensor(
                torch.cat([Ab_block.indices(), new_ind_b], dim=1),
                torch.cat([Ab_block.values(), val_z_t[nz_b]]),
                (k, nb_new), dtype=dtype, device=device,
            ).coalesce()

        b_block = rhs.view(k, 1).to(dtype=dtype, device=device)
        return Ac_block, Ab_block, b_block

    # Row 1: xi1 <= 1
    #   coef_old_c·xi_old + coef_old_b·xi_b + coef_xi2 xi2 + coef_z z <= 1 - rhs_xi1
    blk1 = _build_row(1.0, 1.0, coef_xi2, coef_z, 1.0 - rhs_xi1)

    # Row 2: -xi1 <= 1
    #   -coef_old_c·xi_old - coef_old_b·xi_b - coef_xi2 xi2 - coef_z z <= 1 + rhs_xi1
    blk2 = _build_row(-1.0, -1.0, -coef_xi2, -coef_z, 1.0 + rhs_xi1)

    # Row 3: xi3 <= 1 → -xi1 - z <= 0 → -coef_old_c·xi_old - coef_old_b·xi_b
    #                                    - coef_xi2 xi2 - (coef_z + 1) z <= rhs_xi1
    blk3 = _build_row(-1.0, -1.0, -coef_xi2, -(coef_z + 1.0), rhs_xi1)

    # Row 4: -xi3 <= 1 → xi1 + z <= 2 → coef_old_c·xi_old + coef_old_b·xi_b
    #                                    + coef_xi2 xi2 + (coef_z + 1) z <= 2 - rhs_xi1
    blk4 = _build_row(1.0, 1.0, coef_xi2, (coef_z + 1.0), 2.0 - rhs_xi1)

    # Row 5: xi4 <= 1 → -xi2 + z <= 0  (no Gc/Gb coupling)
    blk5 = _build_row(0.0, 0.0,
                      -torch.ones(k, dtype=dtype, device=device),
                      torch.ones(k, dtype=dtype, device=device),
                      torch.zeros(k, dtype=dtype, device=device))

    # Row 6: -xi4 <= 1 → xi2 - z <= 2  (no Gc/Gb coupling)
    blk6 = _build_row(0.0, 0.0,
                      torch.ones(k, dtype=dtype, device=device),
                      -torch.ones(k, dtype=dtype, device=device),
                      2.0 * torch.ones(k, dtype=dtype, device=device))

    # Stack 6 blocks into one big sparse matrix.
    # ``compact_rows=True`` drops the THREE rows proven LP-redundant
    # (under variable boxes xi_c, xi_b ∈ [-1, 1]; z ∈ [-1, 1] in
    # LP-relax): blk2, blk4, blk6 (the "−xi_j ≤ 1" / negated forms).
    #
    # Soundness + zero precision loss: LP-redundancy means these rows
    # are implied by the kept rows + variable box constraints. The LP
    # solver enforces the boxes, so dropping these rows doesn't change
    # the feasible region nor the optimum. Verified by exhaustive LP
    # tests across (single-neuron α/β trials, multi-neuron, multi-layer)
    # — see research/t2_sparse_gc/verify_redundancy_multi.py.
    #
    # Net: 6 rows per unstable → 3 rows. ~50% Ac storage reduction,
    # sound, no precision loss.
    if compact_rows:
        blocks = [blk1, blk3, blk5]
        rows_per_neuron = 3
    else:
        blocks = [blk1, blk2, blk3, blk4, blk5, blk6]
        rows_per_neuron = 6
    Ac_new_blocks_ind = []
    Ac_new_blocks_val = []
    Ab_new_blocks_ind = []
    Ab_new_blocks_val = []
    b_new_blocks = []
    for bi, (Ac_b, Ab_b, b_b) in enumerate(blocks):
        Ac_b = Ac_b.coalesce()
        Ab_b = Ab_b.coalesce()
        if Ac_b._nnz() > 0:
            ind = Ac_b.indices().clone()
            ind[0] = ind[0] + bi * k
            Ac_new_blocks_ind.append(ind)
            Ac_new_blocks_val.append(Ac_b.values())
        if Ab_b._nnz() > 0:
            ind_b = Ab_b.indices().clone()
            ind_b[0] = ind_b[0] + bi * k
            Ab_new_blocks_ind.append(ind_b)
            Ab_new_blocks_val.append(Ab_b.values())
        b_new_blocks.append(b_b)

    if Ac_new_blocks_ind:
        Ac_extra_ind = torch.cat(Ac_new_blocks_ind, dim=1)
        Ac_extra_val = torch.cat(Ac_new_blocks_val)
    else:
        Ac_extra_ind = torch.zeros((2, 0), dtype=torch.long, device=device)
        Ac_extra_val = torch.zeros(0, dtype=dtype, device=device)
    if Ab_new_blocks_ind:
        Ab_extra_ind = torch.cat(Ab_new_blocks_ind, dim=1)
        Ab_extra_val = torch.cat(Ab_new_blocks_val)
    else:
        Ab_extra_ind = torch.zeros((2, 0), dtype=torch.long, device=device)
        Ab_extra_val = torch.zeros(0, dtype=dtype, device=device)
    Ac_extra = torch.sparse_coo_tensor(
        Ac_extra_ind, Ac_extra_val, (rows_per_neuron * k, ng_new),
        dtype=dtype, device=device,
    ).coalesce()
    Ab_extra = torch.sparse_coo_tensor(
        Ab_extra_ind, Ab_extra_val, (rows_per_neuron * k, nb_new),
        dtype=dtype, device=device,
    ).coalesce()
    b_extra = torch.cat(b_new_blocks, dim=0)

    # Stack old + new constraint blocks. Old Ac (nc0, ng0) → pad to
    # (nc0, ng_new); old Ab (nc0, nb0) → pad to (nc0, nb_new).
    Ac_old_pad = _pad_cols(hz.Ac_sparse, k)  # (nc0, ng_new)
    Ab_old_pad = _pad_cols(hz.Ab_sparse, k)

    # Concatenate by rows
    nc_new = nc0 + rows_per_neuron * k
    if Ac_old_pad._nnz() > 0 or Ac_extra._nnz() > 0:
        Ac_full_ind_list = []
        Ac_full_val_list = []
        if Ac_old_pad._nnz() > 0:
            Ac_full_ind_list.append(Ac_old_pad.indices())
            Ac_full_val_list.append(Ac_old_pad.values())
        if Ac_extra._nnz() > 0:
            ind_e = Ac_extra.indices().clone()
            ind_e[0] = ind_e[0] + nc0
            Ac_full_ind_list.append(ind_e)
            Ac_full_val_list.append(Ac_extra.values())
        Ac_full_ind = torch.cat(Ac_full_ind_list, dim=1)
        Ac_full_val = torch.cat(Ac_full_val_list)
    else:
        Ac_full_ind = torch.zeros((2, 0), dtype=torch.long, device=device)
        Ac_full_val = torch.zeros(0, dtype=dtype, device=device)
    Ac_full = torch.sparse_coo_tensor(
        Ac_full_ind, Ac_full_val, (nc_new, ng_new),
        dtype=dtype, device=device,
    ).coalesce()

    if Ab_old_pad._nnz() > 0 or Ab_extra._nnz() > 0:
        Ab_full_ind_list = []
        Ab_full_val_list = []
        if Ab_old_pad._nnz() > 0:
            Ab_full_ind_list.append(Ab_old_pad.indices())
            Ab_full_val_list.append(Ab_old_pad.values())
        if Ab_extra._nnz() > 0:
            ind_e2 = Ab_extra.indices().clone()
            ind_e2[0] = ind_e2[0] + nc0
            Ab_full_ind_list.append(ind_e2)
            Ab_full_val_list.append(Ab_extra.values())
        Ab_full_ind = torch.cat(Ab_full_ind_list, dim=1)
        Ab_full_val = torch.cat(Ab_full_val_list)
    else:
        Ab_full_ind = torch.zeros((2, 0), dtype=torch.long, device=device)
        Ab_full_val = torch.zeros(0, dtype=dtype, device=device)
    Ab_full = torch.sparse_coo_tensor(
        Ab_full_ind, Ab_full_val, (nc_new, nb_new),
        dtype=dtype, device=device,
    ).coalesce()

    b_full = torch.cat([hz.b, b_extra], dim=0)
    eq_mask_full = torch.cat([
        hz.eq_mask,
        torch.zeros(rows_per_neuron * k, dtype=torch.bool, device=device),
    ])

    return SparseGcZ(
        c=c_out,
        Gc_sparse=Gc_out,
        Gb_sparse=Gb_out,
        dtype=dtype, device=device,
        Ac_sparse=Ac_full,
        Ab_sparse=Ab_full,
        b=b_full,
        eq_mask=eq_mask_full,
    )
