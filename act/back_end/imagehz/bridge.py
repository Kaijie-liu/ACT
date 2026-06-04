"""Flatten bridge — ImageHZ → SparseGcZ for tail LP.

At the Flatten point, ImageHZ's spatial structure is no longer
needed (the tail is dense / 1-D). We materialize each spatial
generator as a sparse COO column in a flat (n = C*H*W) × (ng = # gens)
SparseGcZ.

Soundness invariant per advisor rev-2 test 4: the SparseGcZ
concretization bounds must NOT be narrower than the ImageHZ
bounds at the bridge point. (They are exactly equal by construction
when each xi_id appears once; if an xi_id appears multiple times,
the bridge currently keeps them as separate columns, which is
sound because the LP downstream will treat them as a single
variable through column merging — handled at the SparseGcZ level
on the existing path.)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from act.back_end.hybridz_tf.representations import SparseGcZ
from act.back_end.imagehz.representation import (
    ImageHZ,
    merge_generators_by_xi_id,
)


def flatten_to_sparsegcz(
    hz: ImageHZ,
    *,
    strict_unique_xi_id: bool = False,
) -> SparseGcZ:
    """Materialize the ImageHZ as a SparseGcZ over the flattened
    (n = C * H * W) coordinate, with one sparse column per UNIQUE
    xi_id.

    Per advisor 2026-06-03 Step 2.0 guard 1: generators sharing an
    `xi_id` are MERGED before emitting columns so the downstream LP
    treats them as a single variable. Without this merge, the LP
    would see shared-xi columns as independent and produce a
    spurious phantom margin (sound but useless).

    With ``strict_unique_xi_id=True`` the merge is skipped and a
    duplicate-id input raises instead (used by tests that need to
    detect upstream mistakes).

    Output:
      c       — (n,) center vector (flatten of hz.c in C-major order)
      Gc      — sparse_coo_tensor of shape (n, ng) with one column
                per generator; nonzeros are exactly the generator's
                local values at its region positions.
      Ac, b   — empty (no constraints in the prototype body)
      eq_mask — empty
      nb=0
    """
    if hz.c.dtype != torch.float64:
        raise ValueError(
            f"flatten_to_sparsegcz requires float64; got {hz.c.dtype}"
        )

    device = hz.device
    dtype = hz.dtype
    C, H, W = hz.shape
    n = C * H * W

    # Per advisor 2026-06-03 Step 2.0 guard 1: enforce xi_id uniqueness
    # before materializing columns.
    if strict_unique_xi_id:
        seen: set[int] = set()
        for g in hz.generators:
            if g.xi_id in seen:
                raise ValueError(
                    f"flatten_to_sparsegcz strict mode saw duplicate "
                    f"xi_id {g.xi_id}; upstream operator failed to merge"
                )
            seen.add(g.xi_id)
        gens = list(hz.generators)
    else:
        gens = merge_generators_by_xi_id(
            hz.generators, dtype=dtype, device=device,
        )
    ng = len(gens)

    c_flat = hz.c.reshape(-1).contiguous()

    # Collect sparse COO entries for Gc.
    indices_rows: list[torch.Tensor] = []
    indices_cols: list[torch.Tensor] = []
    values_all: list[torch.Tensor] = []
    for col_idx, g in enumerate(gens):
        # Build the (C, H, W) row indices for this generator's
        # region.
        r = g.region
        # The flatten order is row-major: idx = c*H*W + h*W + w
        c_idx = torch.arange(r.c_lo, r.c_hi, device=device)
        h_idx = torch.arange(r.h_lo, r.h_hi, device=device)
        w_idx = torch.arange(r.w_lo, r.w_hi, device=device)
        cc, hh, ww = torch.meshgrid(c_idx, h_idx, w_idx, indexing="ij")
        flat_rows = (cc * (H * W) + hh * W + ww).reshape(-1)
        # values in the same order.
        vals = g.values.reshape(-1)
        # filter zeros to keep sparsity
        nz = vals.abs() > 0
        if not nz.any():
            continue
        indices_rows.append(flat_rows[nz])
        indices_cols.append(torch.full(
            (int(nz.sum()),), col_idx, dtype=torch.long, device=device,
        ))
        values_all.append(vals[nz])

    if indices_rows:
        rows = torch.cat(indices_rows)
        cols = torch.cat(indices_cols)
        vals_t = torch.cat(values_all)
        indices = torch.stack([rows, cols], dim=0)
    else:
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        vals_t = torch.zeros((0,), dtype=dtype, device=device)

    Gc_sparse = torch.sparse_coo_tensor(
        indices, vals_t, (n, ng), dtype=dtype, device=device,
    ).coalesce()

    # No constraints in the prototype body.
    Ac_sparse = torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long, device=device),
        torch.zeros((0,), dtype=dtype, device=device),
        (0, ng), dtype=dtype, device=device,
    ).coalesce()
    b = torch.zeros((0,), dtype=dtype, device=device)
    eq_mask = torch.zeros((0,), dtype=torch.bool, device=device)

    return SparseGcZ(
        c=c_flat,
        Gc_sparse=Gc_sparse,
        Ac_sparse=Ac_sparse,
        b=b,
        eq_mask=eq_mask,
        dtype=dtype,
        device=device,
    )
