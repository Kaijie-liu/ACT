from __future__ import annotations

import logging
import os

import torch
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import Solver, SolverCaps

if TYPE_CHECKING:
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution

logger = logging.getLogger(__name__)

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
    splits rows by sense via hz_split_constraints."""

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
    # branches). ids are globally-unique-monotonic (see hz_fresh_col_ids), so
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


def hz_fresh_col_ids(k: int, device=None) -> torch.Tensor:
    start = _NEXT_COL_ID[0]
    _NEXT_COL_ID[0] = start + k
    return torch.arange(start, start + k, dtype=torch.long, device=device)


_fresh_col_ids = hz_fresh_col_ids


def reset_col_ids() -> None:
    """Reset the id counter (optional; call at the start of a propagation)."""
    _NEXT_COL_ID[0] = 0


# ============================================================================
# 2. Algebraic operations
# ============================================================================


def _clone_ids(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.clone()


def hz_mark_known_nonempty(hz: HZono, reason: str = "constructed") -> HZono:
    """Attach a lightweight non-emptiness certificate to a constructed HZ.

    Exact transfer functions in this backend preserve non-emptiness from a
    non-empty input box. The verdict layer still has a MILP fallback for objects
    without this construction evidence, but it should not spend a second hard
    MIP just to rediscover that an exactly-propagated HZ is non-empty.
    """
    setattr(hz, "_solver_known_nonempty", True)
    setattr(hz, "_solver_known_nonempty_reason", str(reason))
    return hz


def hz_known_nonempty(hz) -> bool:
    return bool(getattr(hz, "_solver_known_nonempty", False))


def hz_inherit_known_nonempty(out: HZono, *sources, reason: str = "inherited") -> HZono:
    if sources and all(hz_known_nonempty(src) for src in sources):
        return hz_mark_known_nonempty(out, reason)
    return out


def hz_multiply(hz: HZono, R: torch.Tensor) -> HZono:
    # Left-multiply mixes ROWS (output dims), not generator COLUMNS, so each
    # generator factor xi is preserved -> col ids carry through unchanged.
    R = R.to(dtype=hz.c.dtype, device=hz.c.device)
    return hz_inherit_known_nonempty(HZono(
        c=R @ hz.c,
        Gc=R @ hz.Gc,
        Gb=R @ hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    ), hz, reason="affine")


def hz_add_const(hz: HZono, v: torch.Tensor) -> HZono:
    v = v.to(dtype=hz.c.dtype, device=hz.c.device)
    if v.ndim == 1:
        v = v.view(-1, 1)
    return hz_inherit_known_nonempty(HZono(
        c=hz.c + v,
        Gc=hz.Gc.clone(),
        Gb=hz.Gb.clone(),
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    ), hz, reason="affine")


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
    return hz_inherit_known_nonempty(
        HZono(c=new_c, Gc=new_Gc, Gb=new_Gb, Ac=new_Ac, Ab=new_Ab, b=new_b,
              eq_mask=new_eq_mask, col_ids=new_col_ids, bcol_ids=new_bcol_ids),
        hz1,
        hz2,
        reason="minkowski_sum",
    )


def hz_split_constraints(hz: HZono):
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


_split_eq_le = hz_split_constraints


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
        ids = hz_fresh_col_ids(n, device=device)
    hz = HZono(
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
    if bool(torch.all(lb <= ub).item()):
        hz_mark_known_nonempty(hz, "input_box")
    return hz


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
    from act.back_end.solver.solver_gurobi import GurobiSolver, is_gurobi_available

    if not is_gurobi_available():
        raise RuntimeError("gurobipy is not available")
    return GurobiSolver.compute_bounds(hz)


def hz_compute_lp_bounds(
    hz: HZono,
    rows=None,
    *,
    base_lb=None,
    base_ub=None,
    relu_stability: bool = False,
) -> Bounds:
    n = int(hz.c.shape[0])
    p = int(hz.Gc.shape[1])
    q = int(hz.Gb.shape[1])
    if rows is None:
        row_idx = np.arange(n, dtype=np.int64)
    else:
        row_idx = np.asarray(rows, dtype=np.int64).reshape(-1)
        if row_idx.size and ((row_idx < 0).any() or (row_idx >= n).any()):
            raise IndexError("HZ bound row index out of range")
    base_lb_np = None if base_lb is None else np.asarray(base_lb, dtype=np.float64).reshape(-1)
    base_ub_np = None if base_ub is None else np.asarray(base_ub, dtype=np.float64).reshape(-1)
    if relu_stability and (base_lb_np is None or base_ub_np is None):
        relu_stability = False
    if relu_stability and (base_lb_np.size != row_idx.size or base_ub_np.size != row_idx.size):
        raise ValueError("base ReLU bounds must match requested row count")
    lp_time_limit = 0.0
    if relu_stability:
        try:
            lp_time_limit = max(0.0, float(os.environ.get("HZ_RELU_TIGHT_LP_TIMEOUT", "0") or 0.0))
        except Exception:
            lp_time_limit = 0.0
    lp_options = {"time_limit": lp_time_limit} if lp_time_limit > 0.0 else None
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

    work_n = int(row_idx.size)
    LB = np.empty((work_n,), dtype=np.float64)
    UB = np.empty((work_n,), dtype=np.float64)

    def _solve_dim(pos):
        # Per-dim min/max over the (shared, read-only) HZ constraints. Independent
        # across i, so safe to run concurrently; HiGHS releases the GIL during solve.
        i = int(row_idx[pos])
        obj = np.concatenate([Gc_np[i], Gb_np[i]], axis=0)

        def _min():
            res = linprog(c=obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                          bounds=var_bounds, method="highs", options=lp_options)
            if not res.success:
                if relu_stability and base_lb_np is not None:
                    return float(base_lb_np[pos])
                raise RuntimeError(f"[linprog] MIN infeasible at dim {i}: {res.message}")
            return c_np[i] + res.fun

        def _max():
            res = linprog(c=-obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                          bounds=var_bounds, method="highs", options=lp_options)
            if not res.success:
                if relu_stability and base_ub_np is not None:
                    return float(base_ub_np[pos])
                raise RuntimeError(f"[linprog] MAX infeasible at dim {i}: {res.message}")
            return c_np[i] - res.fun

        if relu_stability:
            bl = float(base_lb_np[pos])
            bu = float(base_ub_np[pos])
            if abs(bl) <= abs(bu):
                lb_i = _min()
                if lb_i >= 0.0:
                    return pos, lb_i, bu
                return pos, lb_i, _max()
            ub_i = _max()
            if ub_i <= 0.0:
                return pos, bl, ub_i
            return pos, _min(), ub_i

        return pos, _min(), _max()

    # The 2n LP solves are independent -> parallelize the wide ones (the tight-bounds
    # bottleneck on wide nets). IDENTICAL result to serial (same LPs); threads give a
    # real speedup because HiGHS runs in C and releases the GIL. Gated by env
    # HZ_TIGHT_THREADS (default 1 = serial), and only when n is wide enough to amortize.
    _nthr = int(os.environ.get("HZ_TIGHT_THREADS", "1"))
    if _nthr > 1 and work_n >= 16:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=_nthr) as _ex:
            for pos, lb_i, ub_i in _ex.map(_solve_dim, range(work_n)):
                LB[pos] = lb_i; UB[pos] = ub_i
    else:
        for pos in range(work_n):
            _, LB[pos], UB[pos] = _solve_dim(pos)

    dtype, device = hz.c.dtype, hz.c.device
    return Bounds(
        lb=torch.from_numpy(LB).to(device=device, dtype=dtype).reshape(1, -1),
        ub=torch.from_numpy(UB).to(device=device, dtype=dtype).reshape(1, -1),
    )


_hz_compute_bounds_scipy = hz_compute_lp_bounds


def hz_compute_bounds(hz: HZono, *, exact: bool = False) -> Bounds:
    """Compute box bounds from a hybrid zonotope.

    Args:
        hz: The hybrid zonotope.
        exact: If False (default), always use the fast unconstrained
            over-approximation (|Gc| + |Gb| radius). This is sound but
            may be wider than necessary.  If True, solve per-dimension LPs
            with the open-source scipy/HiGHS backend to obtain tight bounds
            when equality constraints exist.  Gurobi remains available only
            as an explicit diagnostic oracle via ``HZ_BOUNDS_BACKEND=gurobi``
            or ``HZ_BOUNDS_GUROBI=1``.  Use ``exact=True`` only at the final
            output layer
            where tight bounds matter for verification; intermediate
            layers benefit from the 1000×+ speed-up of the fast path
            with negligible precision loss (the full zonotope structure
            is still propagated via ``_hz_cache``).
    """
    if not exact:
        return _hz_bounds_unconstrained(hz)
    if _hz_is_unconstrained(hz):
        return _hz_bounds_unconstrained(hz)
    prefer_gurobi = (
        os.environ.get("HZ_BOUNDS_BACKEND", "").strip().lower() == "gurobi"
        or os.environ.get("HZ_BOUNDS_GUROBI", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if _HAS_SCIPY and not prefer_gurobi:
        try:
            return hz_compute_lp_bounds(hz)
        except Exception as e:
            # Intentional: scipy linprog failures fall back to the unconstrained bounds estimate.
            logger.debug("suppressed: %s", e)
    # The optional Gurobi diagnostic path treats every constraint row as an
    # equality; if this HZ carries inequality rows (eq_mask with any False
    # entry) it would mis-solve. Route those to scipy/open-source instead.
    _has_le = hz.eq_mask is not None and bool((~hz.eq_mask).any().item())
    if prefer_gurobi and not _has_le:
        try:
            return _hz_compute_bounds_gurobi(hz)
        except Exception as e:
            # Intentional: Gurobi failures (license/timeout/numerical) fall back to scipy/unconstrained.
            logger.debug("suppressed: %s", e)
    if _HAS_SCIPY and prefer_gurobi:
        try:
            return hz_compute_lp_bounds(hz)
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
      HZSolver exact bounds (scipy/HiGHS LP, tight) > HZSolver fast box >
      TorchLPSolver box.  Gurobi is an opt-in diagnostic bounds oracle only,
      not a counted HybridZ proof dependency.
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
        need batch LP solving should use TorchLPSolver or another BatchLP
        implementation.
        """
        raise NotImplementedError(
            "HZSolver does not solve CSPs; use compute_bounds() for HZ domain analysis."
        )
