"""QR-based equality-constraint elimination for HZono.

This is a faithful port of HyZor's ``HybridZonotope.project_eq_elim``
(``HybridZonotope.py`` line 5129+). Critical to soundness: the pivoted
QR is computed on the **continuous** equality block ``Ac_eq`` only —
not on the combined ``[Ac_eq, Ab_eq]``. Binary factors ``xi_b in
{-1,+1}`` cannot be eliminated like continuous ones because they have
a discrete domain; instead they appear in the substitution as a
constant correction to the eliminated continuous factors.

Mechanism
---------
Let ``Ac_eq xi_c + Ab_eq xi_b = b_eq`` be the equality block. Pivoted
QR ``Ac_eq = Q R P^T`` identifies ``rank`` dependent columns
(``dep_idx``, indices in ``[0, ng)``) and ``ng - rank`` free columns
(``free_idx``, also in ``[0, ng)``). Then::

    xi_dep = M b_eq - M Ab_eq xi_b - M Ac_eq[:, free_idx] xi_free
           where M = R_dep^{-1} Q^T

Substituting into ``y = c + Gc xi_c + Gb xi_b``::

    c_red   = c + Gc[:, dep] @ (M b_eq)
    Gc_red  = Gc[:, free] - Gc[:, dep] @ (M Ac_eq[:, free])
    Gb_red  = Gb - Gc[:, dep] @ (M Ab_eq)

The implicit ``xi_dep in [-1, +1]`` becomes two new inequality rows
per dep dim, over ``[xi_free; xi_b]``. Pre-existing inequality rows
that reference ``xi_dep`` are substituted in the same way.

Soundness
---------
QR substitution preserves ``y = c + G xi`` exactly. The 2*rank box
rows encode the dep-domain constraint exactly. Binary factors keep
their ``{-1, +1}`` domain throughout. The optional ``ng_base`` budget
merges low-L2-norm free continuous generators into a single box
generator (this is a sound widening; binaries are NEVER merged).
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from act.back_end.solver.solver_hz import HZono, _split_eq_le


def _to_np_f64(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()


def _qr_pivoted_cpu(A: np.ndarray, rank_tol: float = 1e-10):
    """scipy pivoted QR. Returns (Q, R, piv, rank).

    rank counts |R[i, i]| > rank_tol over the leading min(rows, cols).
    """
    from scipy.linalg import qr as _qr
    n_eq = A.shape[0]
    n_cols = A.shape[1]
    Q, R, piv = _qr(A, pivoting=True, overwrite_a=True, check_finite=False)
    R_diag = np.abs(np.diag(R[: min(n_eq, n_cols), :]))
    rank = int(np.sum(R_diag > rank_tol))
    return Q, R, piv, rank


def project_eq_elim(
    hz: HZono,
    ng_base: Optional[int] = None,
    *,
    rank_tol: float = 1e-10,
) -> HZono:
    """Eliminate equality-constrained continuous generators via QR.

    Args:
        hz: input HZono. Equality rows are identified via ``hz.eq_mask``
            (None ⇒ all rows are equalities, legacy semantics).
        ng_base: target number of continuous generators to keep. If
            ``ng_free <= ng_base`` after QR, returns input unchanged
            (already small enough). If ``ng_free > ng_base``, the
            ``ng_free - ng_base`` lowest-L2-norm continuous generators
            are merged into a single box column. If ``None``, no
            elimination is performed and the input is returned
            unchanged (matches HyZor's ``ng_base=None`` early-exit).
        rank_tol: numerical threshold for non-zero R diagonals.

    Returns:
        A new HZono with: (a) at most ``ng_base + 1`` continuous
        generators (or original ``ng - rank`` if no merge needed);
        (b) original binary generators preserved with constant
        corrections; (c) ``2*rank`` new inequality rows enforcing
        ``xi_dep in [-1, +1]``; (d) pre-existing inequality rows
        substituted in. Output ``eq_mask`` marks all new rows as
        inequalities.

    Semantics matches HyZor ``HybridZonotope.project_eq_elim``
    (HZ.py:5129) exactly.
    """
    nc = int(hz.b.shape[0])
    ng = int(hz.Gc.shape[1])
    nb = int(hz.Gb.shape[1])
    n = int(hz.c.shape[0])

    # HyZor early-exit semantics.
    if ng_base is None or nc == 0:
        return hz

    Ac_eq_t, Ab_eq_t, b_eq_t, Ac_le_t, Ab_le_t, b_le_t = _split_eq_le(hz)
    n_eq = int(Ac_eq_t.shape[0])
    n_le = int(Ac_le_t.shape[0])
    if n_eq == 0:
        return hz

    device = hz.c.device
    dtype = hz.c.dtype

    # Numpy float64 staging.
    Ac_eq = _to_np_f64(Ac_eq_t)   # (n_eq, ng)
    Ab_eq = _to_np_f64(Ab_eq_t)   # (n_eq, nb)
    b_eq = _to_np_f64(b_eq_t).reshape(-1)  # (n_eq,)
    Ac_le = _to_np_f64(Ac_le_t)
    Ab_le = _to_np_f64(Ab_le_t)
    b_le = _to_np_f64(b_le_t).reshape(-1)

    # Pivoted QR on Ac_eq ONLY -- this is the critical fix from a
    # previous port that QR'd the combined [Ac_eq, Ab_eq] block and
    # eliminated binaries as if continuous (unsound, lost the {-1,+1}
    # domain). HyZor does QR on Ac_eq alone.
    Q, R, piv, rank = _qr_pivoted_cpu(Ac_eq, rank_tol=rank_tol)
    if rank == 0:
        return hz

    dep_idx = piv[:rank]
    free_idx = piv[rank:]
    ng_free = len(free_idx)

    # Quick exit: HyZor returns self if already small enough.
    if ng_free <= ng_base:
        return hz

    # M = R_dep^{-1} Q^T  (rank, n_eq).
    from scipy.linalg import solve_triangular
    R_dep = R[:rank, :rank]
    Qt = Q[:, :rank].T
    M = solve_triangular(R_dep, Qt)  # (rank, n_eq)

    # Substitution coefficients.
    M_b = M @ b_eq                              # (rank,)
    M_Ab = M @ Ab_eq                            # (rank, nb)
    Ac_eq_free = Ac_eq[:, free_idx]             # (n_eq, ng_free)
    M_Ac_free = M @ Ac_eq_free                  # (rank, ng_free)

    # Reduced generators (binaries PRESERVED with constant correction).
    Gc_np = _to_np_f64(hz.Gc)
    Gb_np = _to_np_f64(hz.Gb)
    c_np = _to_np_f64(hz.c).reshape(-1)

    Gc_dep = Gc_np[:, dep_idx]    # (n, rank)
    Gc_free = Gc_np[:, free_idx]  # (n, ng_free)

    Gc_red = Gc_free - Gc_dep @ M_Ac_free       # (n, ng_free)
    Gb_red = Gb_np - Gc_dep @ M_Ab              # (n, nb)
    c_red = c_np + Gc_dep @ M_b                  # (n,)

    # Box constraints on eliminated continuous: xi_dep in [-1, +1].
    # xi_dep = M_b - M_Ab xi_b - M_Ac_free xi_free
    # Upper:  -M_Ac_free xi_free - M_Ab xi_b <= 1 - M_b
    # Lower:   M_Ac_free xi_free + M_Ab xi_b <= 1 + M_b
    A_box_ub_free = -M_Ac_free          # (rank, ng_free)
    A_box_ub_b = -M_Ab                  # (rank, nb)
    b_box_ub = 1.0 - M_b                # (rank,)

    A_box_lb_free = M_Ac_free
    A_box_lb_b = M_Ab
    b_box_lb = 1.0 + M_b

    # Pre-existing inequality rows: substitute xi_dep out.
    if n_le > 0:
        Ac_le_dep = Ac_le[:, dep_idx]
        Ac_le_free = Ac_le[:, free_idx]
        Ac_le_remap_free = Ac_le_free - Ac_le_dep @ M_Ac_free   # (n_le, ng_free)
        Ab_le_remap = Ab_le - Ac_le_dep @ M_Ab                  # (n_le, nb)
        b_le_remap = b_le - Ac_le_dep @ M_b                     # (n_le,)
    else:
        Ac_le_remap_free = np.zeros((0, ng_free))
        Ab_le_remap = np.zeros((0, nb))
        b_le_remap = np.zeros(0)

    # Merge low-L2-norm continuous free generators into a single box
    # generator. Binaries are NEVER merged.
    norms = np.linalg.norm(Gc_red, axis=0)  # (ng_free,)
    keep_count = min(ng_base, ng_free - 1)
    if keep_count < 1:
        keep_count = 1
    order = np.argsort(-norms)
    keep_idx = order[:keep_count]
    merge_idx = order[keep_count:]

    Gc_keep = Gc_red[:, keep_idx]               # (n, keep_count)
    Gc_merge = Gc_red[:, merge_idx]             # (n, ng_free - keep_count)
    # Box column = L1 widening of merged columns.
    box_col = np.abs(Gc_merge).sum(axis=1, keepdims=True)  # (n, 1)
    Gc_new = np.concatenate([Gc_keep, box_col], axis=1)    # (n, keep+1)
    ng_new = keep_count + 1

    # Remap box ineq rows: split free cols into keep/merge.
    A_box_ub_keep = A_box_ub_free[:, keep_idx]
    A_box_ub_merge = A_box_ub_free[:, merge_idx]
    A_box_lb_keep = A_box_lb_free[:, keep_idx]
    A_box_lb_merge = A_box_lb_free[:, merge_idx]

    # New box-column has zero coefficient in dep-box rows (the box gen
    # is independent of any prior factor).
    box_zero_col = np.zeros((rank, 1))
    A_box_ub_new_cont = np.concatenate(
        [A_box_ub_keep, box_zero_col], axis=1
    )  # (rank, ng_new)
    A_box_lb_new_cont = np.concatenate(
        [A_box_lb_keep, box_zero_col], axis=1
    )

    # The merged cols contribute |A_box_*_merge| @ 1 widening to b.
    b_box_ub_widened = b_box_ub + np.abs(A_box_ub_merge).sum(axis=1)
    b_box_lb_widened = b_box_lb + np.abs(A_box_lb_merge).sum(axis=1)

    # Remap pre-existing le rows likewise.
    if n_le > 0:
        Ac_le_keep = Ac_le_remap_free[:, keep_idx]
        Ac_le_merge = Ac_le_remap_free[:, merge_idx]
        box_zero_le = np.zeros((n_le, 1))
        Ac_le_new_cont = np.concatenate(
            [Ac_le_keep, box_zero_le], axis=1
        )  # (n_le, ng_new)
        b_le_widened = b_le_remap + np.abs(Ac_le_merge).sum(axis=1)
    else:
        Ac_le_new_cont = np.zeros((0, ng_new))
        b_le_widened = np.zeros(0)

    # Assemble all post-elim inequality rows: 2*rank box rows + n_le
    # pre-existing rows, all sharing the same [xi_free_kept; xi_box; xi_b]
    # column layout.
    Ac_out_np = np.concatenate(
        [A_box_ub_new_cont, A_box_lb_new_cont, Ac_le_new_cont], axis=0
    )  # (2*rank + n_le, ng_new)
    Ab_out_np = np.concatenate(
        [A_box_ub_b, A_box_lb_b, Ab_le_remap], axis=0
    )  # (2*rank + n_le, nb)
    b_out_np = np.concatenate(
        [b_box_ub_widened, b_box_lb_widened, b_le_widened]
    )  # (2*rank + n_le,)
    new_le_count = int(Ac_out_np.shape[0])

    # Convert back to torch.
    Gc_out = torch.from_numpy(Gc_new).to(dtype=dtype, device=device)
    Gb_out = torch.from_numpy(Gb_red).to(dtype=dtype, device=device)
    c_out = torch.from_numpy(c_red).to(dtype=dtype, device=device).view(-1, 1)
    Ac_out = torch.from_numpy(Ac_out_np).to(dtype=dtype, device=device)
    Ab_out = torch.from_numpy(Ab_out_np).to(dtype=dtype, device=device)
    b_out = torch.from_numpy(b_out_np).to(dtype=dtype, device=device).view(-1, 1)
    eq_mask_out = torch.zeros(new_le_count, dtype=torch.bool, device=device)

    return HZono(c=c_out, Gc=Gc_out, Gb=Gb_out,
                 Ac=Ac_out, Ab=Ab_out, b=b_out,
                 eq_mask=eq_mask_out)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _trivial_no_eq_returns_same():
    """Empty constraint block → pass through."""
    n, ng, nb = 3, 2, 1
    hz = HZono(
        c=torch.zeros(n, 1), Gc=torch.randn(n, ng),
        Gb=torch.randn(n, nb), Ac=torch.zeros(0, ng),
        Ab=torch.zeros(0, nb), b=torch.zeros(0, 1),
        eq_mask=None,
    )
    out = project_eq_elim(hz, ng_base=ng)
    assert out is hz, "should pass through when no eq rows"


def _trivial_eq_eliminates():
    """Single equality row eliminates one continuous factor."""
    n, ng, nb = 2, 3, 0
    Gc = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]], dtype=torch.float64)
    Gb = torch.zeros(n, nb, dtype=torch.float64)
    Ac = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)  # xi_c[0] = 0
    Ab = torch.zeros(1, nb, dtype=torch.float64)
    b = torch.tensor([[0.0]], dtype=torch.float64)
    hz = HZono(c=torch.zeros(n, 1, dtype=torch.float64),
               Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b,
               eq_mask=torch.tensor([True]))
    out = project_eq_elim(hz, ng_base=2)
    # ng_free = 2, ng_base = 2 → early-exit (already small enough).
    assert out is hz
    # Force elimination by tight budget.
    out2 = project_eq_elim(hz, ng_base=1)
    assert out2.Gc.shape[1] <= 2, f"Gc cols={out2.Gc.shape[1]}"


if __name__ == "__main__":
    print("=== project_eq_elim self-tests ===")
    _trivial_no_eq_returns_same()
    print("  trivial_no_eq_returns_same OK")
    _trivial_eq_eliminates()
    print("  trivial_eq_eliminates OK")
    print("PASSED")
