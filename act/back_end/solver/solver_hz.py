from __future__ import annotations

import logging

import torch
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import Solver, SolverCaps

if TYPE_CHECKING:
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution

logger = logging.getLogger(__name__)

try:
    from act.back_end.solver.solver_gurobi import GurobiSolver, is_gurobi_available

    _HAS_GUROBI = is_gurobi_available()
except ImportError:
    _HAS_GUROBI = False

try:
    import numpy as np
    from scipy.optimize import linprog

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# 1. HZono dataclass
# ============================================================================


@dataclass
class HZono:
    """Z = {c + Gc @ xi_c + Gb @ xi_b | (Ac @ xi_c + Ab @ xi_b) [op] b,
    xi_c in [-1,1]^ng, xi_b in {-1,1}^nb}

    eq_mask (optional, (nc,) bool): per-row constraint sense. True/None =
    equality (== b); False = inequality (<= b). Default None means ALL rows
    are equalities (backward-compatible with the original all-equality form).
    The inequality sense is the general HZono form (Ac xi <= b); the verdict
    splits rows by sense via _split_eq_le."""

    c: torch.Tensor  # (n, 1)
    Gc: torch.Tensor  # (n, ng)
    Gb: torch.Tensor  # (n, nb)
    Ac: torch.Tensor  # (nc, ng)
    Ab: torch.Tensor  # (nc, nb)
    b: torch.Tensor  # (nc, 1)
    eq_mask: Optional[torch.Tensor] = None  # (nc,) bool; None = all-equality
    # Per-generator identity tags (optional). col_ids: (ng,) long, bcol_ids:
    # (nb,) long. Two columns with the SAME id across HZs are the SAME latent
    # factor xi (e.g. a shared input pixel surviving into two residual
    # branches). ids are globally-unique-monotonic (see _fresh_col_ids), so
    # distinct factors never collide -> shared-generator merge (hz_sgm_add) is
    # sound by construction. None = untracked (ops then fall back to the
    # independent-factor Minkowski sum, also sound but looser for residuals).
    col_ids: Optional[torch.Tensor] = None
    bcol_ids: Optional[torch.Tensor] = None


# Monotonic source of globally-unique generator ids. Monotonic (never reset
# mid-process is fine) guarantees two independently-created factors get
# distinct ids, so hz_sgm_add can only merge columns that are *literally* the
# same factor -> sound. Reset between nets only keeps the integers small.
_NEXT_COL_ID = [0]


def _fresh_col_ids(k: int, device=None) -> torch.Tensor:
    start = _NEXT_COL_ID[0]
    _NEXT_COL_ID[0] = start + k
    return torch.arange(start, start + k, dtype=torch.long, device=device)


def reset_col_ids() -> None:
    """Reset the id counter (optional; call at the start of a propagation)."""
    _NEXT_COL_ID[0] = 0


# ============================================================================
# 2. Algebraic operations
# ============================================================================


def _clone_ids(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.clone()


def hz_multiply(hz: HZono, R: torch.Tensor) -> HZono:
    # Left-multiply mixes ROWS (output dims), not generator COLUMNS, so each
    # generator factor xi is preserved -> col ids carry through unchanged.
    R = R.to(dtype=hz.c.dtype, device=hz.c.device)
    return HZono(
        c=R @ hz.c,
        Gc=R @ hz.Gc,
        Gb=R @ hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def hz_add_const(hz: HZono, v: torch.Tensor) -> HZono:
    v = v.to(dtype=hz.c.dtype, device=hz.c.device)
    if v.ndim == 1:
        v = v.view(-1, 1)
    return HZono(
        c=hz.c + v,
        Gc=hz.Gc.clone(),
        Gb=hz.Gb.clone(),
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def hz_minkowski_sum(hz1: HZono, hz2: HZono) -> HZono:
    dtype, device = hz1.c.dtype, hz1.c.device

    new_c = hz1.c + hz2.c.to(dtype=dtype, device=device)
    new_Gc = torch.cat([hz1.Gc, hz2.Gc.to(dtype=dtype, device=device)], dim=1)
    new_Gb = torch.cat([hz1.Gb, hz2.Gb.to(dtype=dtype, device=device)], dim=1)

    nc1, nc2 = hz1.Ac.shape[0], hz2.Ac.shape[0]
    ng1, ng2 = hz1.Gc.shape[1], hz2.Gc.shape[1]
    nb1, nb2 = hz1.Gb.shape[1], hz2.Gb.shape[1]

    Ac_top = torch.cat(
        [hz1.Ac, torch.zeros((nc1, ng2), dtype=dtype, device=device)], dim=1
    )
    Ac_bot = torch.cat(
        [
            torch.zeros((nc2, ng1), dtype=dtype, device=device),
            hz2.Ac.to(dtype=dtype, device=device),
        ],
        dim=1,
    )
    new_Ac = torch.cat([Ac_top, Ac_bot], dim=0)

    Ab_top = torch.cat(
        [hz1.Ab, torch.zeros((nc1, nb2), dtype=dtype, device=device)], dim=1
    )
    Ab_bot = torch.cat(
        [
            torch.zeros((nc2, nb1), dtype=dtype, device=device),
            hz2.Ab.to(dtype=dtype, device=device),
        ],
        dim=1,
    )
    new_Ab = torch.cat([Ab_top, Ab_bot], dim=0)

    new_b = torch.cat([hz1.b, hz2.b.to(dtype=dtype, device=device)], dim=0)
    # eq_mask: concat per-operand senses (None operand = all-equality rows).
    if hz1.eq_mask is None and hz2.eq_mask is None:
        new_eq_mask = None
    else:
        m1 = (hz1.eq_mask if hz1.eq_mask is not None
              else torch.ones(nc1, dtype=torch.bool, device=device))
        m2 = (hz2.eq_mask if hz2.eq_mask is not None
              else torch.ones(nc2, dtype=torch.bool, device=device))
        new_eq_mask = torch.cat([m1.to(device), m2.to(device)], dim=0)
    # Minkowski treats the two summands' factors as INDEPENDENT, so their id
    # blocks are simply concatenated (kept only if both sides are tracked).
    if hz1.col_ids is not None and hz2.col_ids is not None:
        new_col_ids = torch.cat([hz1.col_ids.to(device), hz2.col_ids.to(device)])
    else:
        new_col_ids = None
    if hz1.bcol_ids is not None and hz2.bcol_ids is not None:
        new_bcol_ids = torch.cat([hz1.bcol_ids.to(device), hz2.bcol_ids.to(device)])
    else:
        new_bcol_ids = None
    return HZono(c=new_c, Gc=new_Gc, Gb=new_Gb, Ac=new_Ac, Ab=new_Ab, b=new_b,
                 eq_mask=new_eq_mask, col_ids=new_col_ids, bcol_ids=new_bcol_ids)


def _split_eq_le(hz: HZono):
    """Split constraint rows by sense → ((Ac_eq, Ab_eq, b_eq), (Ac_le, Ab_le, b_le)).

    eq_mask True/None → equality (== b); False → inequality (<= b). Used by the
    bound/verdict LP so equality rows go to A_eq and inequality rows to A_ub."""
    nc = hz.Ac.shape[0]
    if nc == 0 or hz.eq_mask is None:
        return (hz.Ac, hz.Ab, hz.b), (
            hz.Ac.new_zeros(0, hz.Ac.shape[1]),
            hz.Ab.new_zeros(0, hz.Ab.shape[1]),
            hz.b.new_zeros(0, 1))
    m = hz.eq_mask.to(torch.bool)
    return ((hz.Ac[m], hz.Ab[m], hz.b[m]),
            (hz.Ac[~m], hz.Ab[~m], hz.b[~m]))


def hz_from_bounds(bounds: Bounds, dtype, device, *, track_ids: bool = False,
                   col_ids: Optional[torch.Tensor] = None) -> HZono:
    """Box -> HZ. If ``track_ids``, assign fresh monotonic ids to the n box
    generators (so a downstream residual ADD can share-merge them). Pass an
    explicit ``col_ids`` to reuse another HZ's factor identities (e.g. a
    floating residual root that reads the SAME network input as the main
    branch must inherit the input's ids, not fresh ones)."""
    lb = bounds.lb.flatten().to(dtype=dtype, device=device)
    ub = bounds.ub.flatten().to(dtype=dtype, device=device)
    n = lb.shape[0]
    c = ((lb + ub) / 2.0).view(-1, 1)
    rad = (ub - lb) / 2.0
    ids = None
    if col_ids is not None:
        ids = col_ids.to(device=device)
    elif track_ids:
        ids = _fresh_col_ids(n, device=device)
    return HZono(
        c=c,
        Gc=torch.diag(rad),
        Gb=torch.zeros((n, 0), dtype=dtype, device=device),
        Ac=torch.zeros((0, n), dtype=dtype, device=device),
        Ab=torch.zeros((0, 0), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
        col_ids=ids,
        bcol_ids=(torch.zeros(0, dtype=torch.long, device=device)
                  if ids is not None else None),
    )


# ============================================================================
# 3. Bounds computation
# ============================================================================


def _hz_is_unconstrained(hz: HZono) -> bool:
    tol = 1e-12
    return (
        torch.all(torch.abs(hz.Ac) < tol).item()
        and torch.all(torch.abs(hz.Ab) < tol).item()
        and torch.all(torch.abs(hz.b) < tol).item()
    )


def _hz_bounds_unconstrained(hz: HZono) -> Bounds:
    n = hz.c.shape[0]
    dtype, device = hz.c.dtype, hz.c.device
    absGc = (
        hz.Gc.abs().sum(dim=1, keepdim=True)
        if hz.Gc.numel()
        else torch.zeros((n, 1), dtype=dtype, device=device)
    )
    absGb = (
        hz.Gb.abs().sum(dim=1, keepdim=True)
        if hz.Gb.numel()
        else torch.zeros((n, 1), dtype=dtype, device=device)
    )
    rad = absGc + absGb
    return Bounds(lb=(hz.c - rad).reshape(1, -1), ub=(hz.c + rad).reshape(1, -1))


def _hz_compute_bounds_gurobi(hz: HZono) -> Bounds:
    return GurobiSolver.compute_bounds(hz)


def _hz_compute_bounds_scipy(hz: HZono) -> Bounds:
    n = int(hz.c.shape[0])
    p = int(hz.Gc.shape[1])
    q = int(hz.Gb.shape[1])
    c_np = hz.c.detach().cpu().numpy().astype("float64").reshape(-1)
    Gc_np = hz.Gc.detach().cpu().numpy().astype("float64")
    Gb_np = hz.Gb.detach().cpu().numpy().astype("float64")
    Ac_np = hz.Ac.detach().cpu().numpy().astype("float64")
    Ab_np = hz.Ab.detach().cpu().numpy().astype("float64")
    b_np = hz.b.detach().cpu().numpy().astype("float64").reshape(-1)

    A_all = (
        np.concatenate([Ac_np, Ab_np], axis=1) if (Ac_np.size or Ab_np.size) else None
    )
    # Split rows by sense: equality (eq_mask True/None) vs inequality (<= b).
    if A_all is not None and hz.eq_mask is not None:
        m = hz.eq_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        A_eq = A_all[m] if m.any() else None
        b_eq = b_np[m] if m.any() else None
        A_ub = A_all[~m] if (~m).any() else None
        b_ub = b_np[~m] if (~m).any() else None
    else:
        A_eq = A_all
        b_eq = b_np if (A_all is not None) else None
        A_ub = b_ub = None
    var_bounds = [(-1.0, 1.0)] * (p + q)

    LB = np.empty((n,), dtype=np.float64)
    UB = np.empty((n,), dtype=np.float64)
    for i in range(n):
        obj = np.concatenate([Gc_np[i], Gb_np[i]], axis=0)
        res_min = linprog(
            c=obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
            bounds=var_bounds, method="highs"
        )
        if not res_min.success:
            raise RuntimeError(
                f"[linprog] MIN infeasible at dim {i}: {res_min.message}"
            )
        LB[i] = c_np[i] + res_min.fun
        res_max = linprog(
            c=-obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
            bounds=var_bounds, method="highs"
        )
        if not res_max.success:
            raise RuntimeError(
                f"[linprog] MAX infeasible at dim {i}: {res_max.message}"
            )
        UB[i] = c_np[i] - res_max.fun

    dtype, device = hz.c.dtype, hz.c.device
    return Bounds(
        lb=torch.from_numpy(LB).to(device=device, dtype=dtype).reshape(1, -1),
        ub=torch.from_numpy(UB).to(device=device, dtype=dtype).reshape(1, -1),
    )


def hz_compute_bounds(hz: HZono, *, exact: bool = False) -> Bounds:
    """Compute box bounds from a hybrid zonotope.

    Args:
        hz: The hybrid zonotope.
        exact: If False (default), always use the fast unconstrained
            over-approximation (|Gc| + |Gb| radius). This is sound but
            may be wider than necessary.  If True, solve per-dimension
            LP/MILP to obtain tight bounds when equality constraints
            exist.  Use ``exact=True`` only at the final output layer
            where tight bounds matter for verification; intermediate
            layers benefit from the 1000×+ speed-up of the fast path
            with negligible precision loss (the full zonotope structure
            is still propagated via ``_hz_cache``).
    """
    if _hz_is_unconstrained(hz):
        return _hz_bounds_unconstrained(hz)
    if not exact:
        return _hz_bounds_unconstrained(hz)
    # The Gurobi path treats every constraint row as an equality; if this HZ
    # carries inequality (le) rows (eq_mask with any False entry, e.g. a
    # post-PEE box-encoded zonotope) it would mis-solve. Route those to scipy,
    # which splits eq/le by eq_mask.
    _has_le = hz.eq_mask is not None and bool((~hz.eq_mask).any().item())
    if _HAS_GUROBI and not _has_le:
        try:
            return _hz_compute_bounds_gurobi(hz)
        except Exception as e:
            # Intentional: Gurobi failures (license/timeout/numerical) fall back to scipy/unconstrained.
            logger.debug("suppressed: %s", e)
    if _HAS_SCIPY:
        try:
            return _hz_compute_bounds_scipy(hz)
        except Exception as e:
            # Intentional: scipy linprog failures fall back to the unconstrained bounds estimate.
            logger.debug("suppressed: %s", e)
    return _hz_bounds_unconstrained(hz)


# ============================================================================
# 4. HZSolver
# ============================================================================


class HZSolver(Solver):
    """Hybrid Zonotope bounds solver.

    Precision hierarchy:
      GurobiSolver (MILP, exact) > HZSolver (HZ, tight) > TorchLPSolver (box, fast)
    """

    def __init__(self):
        self._last_bounds: Optional[Bounds] = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=False, supports_csp=False, supports_hz=True)

    def compute_bounds(self, hz: HZono, *, exact: bool = False) -> Bounds:
        self._last_bounds = hz_compute_bounds(hz, exact=exact)
        return self._last_bounds

    def solve_batch(
        self,
        problem: "BatchLPProblem",
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":
        """HZSolver does not accept BatchLPProblem inputs.

        HZSolver operates on HZono (hybrid zonotope) domains via
        compute_bounds(), not on LP/CSP batch problems.  Callers that
        need batch LP solving should use TorchLPSolver or GurobiSolver.
        """
        raise NotImplementedError(
            "HZSolver does not solve CSPs; use compute_bounds() for HZ domain analysis."
        )
