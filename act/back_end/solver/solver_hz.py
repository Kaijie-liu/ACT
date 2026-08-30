"""Hybrid Zonotope domain and dense torch representation.

ACT uses one Hybrid Zonotope abstract domain:

    z = c + Gc xi_c + Gb xi_b
    Ac xi_c + Ab xi_b == b
    Auc xi_c + Aub xi_b <= ub

This module contains ``HZono`` and ``SparseHZono``, the dense torch and
scipy-CSR representations of the same Hybrid Zonotope domain. It also contains
the HZ algebra helpers and final LP/MILP verdict routines used by HybridzTF.
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
import time
import weakref
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from fractions import Fraction
from itertools import product
from threading import Lock

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import Solver, SolverCaps

if TYPE_CHECKING:
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import linprog

    _HAS_SCIPY = True
except ImportError:
    np = None
    sp = None
    linprog = None
    _HAS_SCIPY = False


# ============================================================================
# 1. HZono dataclass
# ============================================================================


@dataclass
class HZono:
    """Dense torch representation of the Hybrid Zonotope domain.

    Z = {c + Gc @ xi_c + Gb @ xi_b | (Ac @ xi_c + Ab @ xi_b) [op] b,
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
    col_ids: Optional[torch.Tensor] = None
    bcol_ids: Optional[torch.Tensor] = None


_NEXT_COL_ID = [0]
_COL_ID_LOCK = Lock()
_COL_ID_INT64_MAX = int(torch.iinfo(torch.int64).max)


def _hz_reserve_col_id_range(
    k: int,
    *,
    lower_bound_exclusive: Optional[int] = None,
) -> tuple[int, int]:
    """Atomically reserve one non-reusable signed-int64 ID interval."""

    k = int(k)
    if k < 0:
        raise ValueError(f"generator id count must be nonnegative, got {k}")
    if lower_bound_exclusive is None:
        floor = -1
    else:
        if isinstance(lower_bound_exclusive, bool):
            raise TypeError("generator id lower bound must be an integer")
        floor = int(lower_bound_exclusive)
        if floor < -1 or floor > _COL_ID_INT64_MAX:
            raise ValueError(
                "generator id lower bound is outside signed int64"
            )
    with _COL_ID_LOCK:
        current = int(_NEXT_COL_ID[0])
        if k == 0:
            return current, current
        start = max(current, floor + 1)
        if (
            start < 0
            or start > _COL_ID_INT64_MAX
            or k - 1 > _COL_ID_INT64_MAX - start
        ):
            raise OverflowError(
                "generator id reservation exceeds signed int64"
            )
        stop = start + k
        # Reservations are never rolled back.  A downstream failure burns the
        # interval rather than allowing a stale HZ to collide with reused IDs.
        _NEXT_COL_ID[0] = stop
        return start, stop


def hz_fresh_col_ids(k: int, device=None) -> torch.Tensor:
    start, stop = _hz_reserve_col_id_range(k)
    return (
        torch.arange(stop - start, dtype=torch.long, device=device)
        + start
    )


def hz_reserve_fresh_col_ids_above(
    k: int,
    *,
    lower_bound_exclusive: int,
    device=None,
) -> torch.Tensor:
    """Reserve fresh global IDs strictly above a live parent ID floor.

    The floor adjustment and allocation share the same lock used by every
    normal HybridZ factor allocation.  This is intended for detached HZ
    extensions whose parent may have been constructed before the current
    process-local allocator state was established.
    """

    start, stop = _hz_reserve_col_id_range(
        k,
        lower_bound_exclusive=lower_bound_exclusive,
    )
    return (
        torch.arange(stop - start, dtype=torch.long, device=device)
        + start
    )




# ============================================================================
# 2. Algebraic operations
# ============================================================================


def _clone_ids(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.clone()


def hz_mark_known_nonempty(hz: HZono, reason: str = "constructed") -> HZono:
    """Mark an HZ produced by exact transfer functions as non-empty."""
    setattr(hz, "_solver_known_nonempty", True)
    setattr(hz, "_solver_known_nonempty_reason", str(reason))
    return hz


def hz_known_nonempty(hz) -> bool:
    return bool(getattr(hz, "_solver_known_nonempty", False))


# A plain ``known_nonempty`` marker is intentionally only a hint: numerical
# propagation or a caller can attach it to an inconsistent HZ.  Strict SAFE
# authorization is reserved for either an exact stored-float witness or this
# module-private construction theorem token.  The operator-HZ builder earns
# the latter by induction over a nonempty input box and total, outward transfer
# relations (affine error envelopes, equality bands, and ReLU graph outers).
_HZ_CONSTRUCTIVE_NONEMPTY_TOKEN = object()
_HZ_EXACT_PHASE_COVER_MEMBER_TOKEN = object()
_HZ_CONDITIONAL_PROPERTY_ROWS_PRODUCER_TOKEN = object()
_HZ_OBJBOUND_SAFE_TOKEN = object()


def hz_mark_constructively_nonempty(
    hz: HZono,
    reason: str,
) -> HZono:
    """Attach the strict construction-theorem certificate used by base gating."""

    hz_mark_known_nonempty(hz, reason)
    setattr(hz, "_solver_constructive_nonempty_token", _HZ_CONSTRUCTIVE_NONEMPTY_TOKEN)
    setattr(hz, "_solver_constructive_nonempty_reason", str(reason))
    return hz


def hz_constructively_nonempty(hz) -> bool:
    return (
        getattr(hz, "_solver_constructive_nonempty_token", None)
        is _HZ_CONSTRUCTIVE_NONEMPTY_TOKEN
    )


def _hz_exact_phase_cover_member(hz) -> bool:
    """Return whether ``hz`` is an exact child of a constructively valid HZ."""

    return (
        getattr(hz, "_solver_exact_phase_cover_member_token", None)
        is _HZ_EXACT_PHASE_COVER_MEMBER_TOKEN
    )


def hz_inherit_known_nonempty(out: HZono, *sources, reason: str = "inherited") -> HZono:
    if sources and all(hz_known_nonempty(src) for src in sources):
        return hz_mark_known_nonempty(out, reason)
    return out


def hz_multiply(hz: HZono, R: torch.Tensor) -> HZono:
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
    if hz1.eq_mask is None and hz2.eq_mask is None:
        new_eq_mask = None
    else:
        m1 = (hz1.eq_mask if hz1.eq_mask is not None
              else torch.ones(nc1, dtype=torch.bool, device=device))
        m2 = (hz2.eq_mask if hz2.eq_mask is not None
              else torch.ones(nc2, dtype=torch.bool, device=device))
        new_eq_mask = torch.cat([m1.to(device), m2.to(device)], dim=0)
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


def hz_compute_lp_bounds(
    hz: HZono,
    rows=None,
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




def hz_compute_bounds(hz: HZono, *, exact: bool = False) -> Bounds:
    """Compute box bounds from a hybrid zonotope.

    Args:
        hz: The hybrid zonotope.
        exact: If False (default), always use the fast unconstrained
            over-approximation (|Gc| + |Gb| radius). This is sound but
            may be wider than necessary. If True, solve per-dimension LPs
            with the open-source scipy/HiGHS backend to obtain tight bounds
            when equality constraints exist. Use ``exact=True`` only at the final
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
    if _HAS_SCIPY:
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
      TorchLPSolver box.
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

def _as_csr(mat, *, shape: Optional[Tuple[int, int]] = None) -> sp.csr_matrix:
    if mat is None:
        if shape is None:
            raise ValueError("shape is required when mat is None")
        return sp.csr_matrix(shape, dtype=np.float64)
    if sp.issparse(mat):
        out = mat.tocsr().astype(np.float64)
    else:
        out = sp.csr_matrix(np.asarray(mat, dtype=np.float64))
    if shape is not None and out.shape != shape:
        raise ValueError(f"sparse matrix shape mismatch: got {out.shape}, expected {shape}")
    return out


def _torch_to_csr(t) -> sp.csr_matrix:
    shape = tuple(int(x) for x in t.shape)
    if t.numel() == 0:
        return sp.csr_matrix(shape, dtype=np.float64)
    return sp.csr_matrix(t.detach().cpu().double().numpy(), dtype=np.float64)


def _torch_to_np(t) -> np.ndarray:
    return t.detach().cpu().double().numpy().reshape(-1).astype(np.float64, copy=False)


@dataclass
class SparseHZono:
    """Sparse CSR representation of the Hybrid Zonotope domain.

    This is a native sparse propagation backend, not a view of ``HZono`` and
    not a second abstract domain.  The dense ``HZono`` and sparse
    ``SparseHZono`` representations lower to the same verdict layer.

    Continuous variables use ``xi_c in [-1, 1]``.  Binary variables use
    ``xi_b in {-1, 1}``.  Inequality rows are optional; absent rows are exposed
    to solver code as zero-row CSR blocks.
    """

    c: np.ndarray
    Gc: sp.csr_matrix
    Gb: sp.csr_matrix
    Ac: sp.csr_matrix
    Ab: sp.csr_matrix
    b: np.ndarray
    Auc: Optional[sp.csr_matrix] = None
    Aub: Optional[sp.csr_matrix] = None
    ub: Optional[np.ndarray] = None
    # Stable generator identities.  Sparse branch-local nonlinear factors
    # cannot be aligned by their positional column index: sibling branches
    # commonly append different variables at the same local offset.  Joins
    # therefore require and align these ids exactly, just like dense HZono.
    col_ids: Optional[np.ndarray] = None
    bcol_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=np.float64).reshape(-1)
        self.b = np.asarray(self.b, dtype=np.float64).reshape(-1)
        self.Gc = _as_csr(self.Gc)
        self.Gb = _as_csr(self.Gb)
        self.Ac = _as_csr(self.Ac)
        self.Ab = _as_csr(self.Ab)

        n_out = int(self.c.size)
        n_cont = int(self.Gc.shape[1])
        n_bin = int(self.Gb.shape[1])
        if self.col_ids is not None:
            self.col_ids = np.asarray(self.col_ids, dtype=np.int64).reshape(-1)
            if self.col_ids.size != n_cont:
                raise ValueError(
                    "SparseHZono continuous id mismatch: "
                    f"ids={self.col_ids.size}, Gc_cols={n_cont}"
                )
            if np.unique(self.col_ids).size != self.col_ids.size:
                raise ValueError("SparseHZono continuous generator ids must be unique")
        if self.bcol_ids is not None:
            self.bcol_ids = np.asarray(self.bcol_ids, dtype=np.int64).reshape(-1)
            if self.bcol_ids.size != n_bin:
                raise ValueError(
                    "SparseHZono binary id mismatch: "
                    f"ids={self.bcol_ids.size}, Gb_cols={n_bin}"
                )
            if np.unique(self.bcol_ids).size != self.bcol_ids.size:
                raise ValueError("SparseHZono binary generator ids must be unique")
        if self.Gc.shape[0] != n_out or self.Gb.shape[0] != n_out:
            raise ValueError(
                "SparseHZono value shape mismatch: "
                f"c={n_out}, Gc={self.Gc.shape}, Gb={self.Gb.shape}"
            )
        if self.Ac.shape[1] != n_cont or self.Ab.shape[1] != n_bin:
            raise ValueError(
                "SparseHZono equality column mismatch: "
                f"Gc_cols={n_cont}, Gb_cols={n_bin}, Ac={self.Ac.shape}, Ab={self.Ab.shape}"
            )
        if self.Ac.shape[0] != self.Ab.shape[0] or self.Ac.shape[0] != self.b.size:
            raise ValueError(
                "SparseHZono equality row mismatch: "
                f"Ac={self.Ac.shape}, Ab={self.Ab.shape}, b={self.b.size}"
            )

        has_upper = self.Auc is not None or self.Aub is not None or self.ub is not None
        if has_upper:
            if self.Auc is None or self.Aub is None or self.ub is None:
                raise ValueError("upper constraints require Auc, Aub, and ub together")
            self.Auc = _as_csr(self.Auc, shape=(self.Auc.shape[0], n_cont))
            self.Aub = _as_csr(self.Aub, shape=(self.Aub.shape[0], n_bin))
            self.ub = np.asarray(self.ub, dtype=np.float64).reshape(-1)
            if self.Auc.shape[0] != self.Aub.shape[0] or self.Auc.shape[0] != self.ub.size:
                raise ValueError(
                    "SparseHZono upper row mismatch: "
                    f"Auc={self.Auc.shape}, Aub={self.Aub.shape}, ub={self.ub.size}"
                )

    @classmethod
    def from_dense_hz(cls, hz: HZono) -> "SparseHZono":
        """Convert a dense torch-backed ``HZono`` to CSR form."""

        (Ace, Abe, be), (Acl, Abl, bl) = hz_split_constraints(hz)
        out = cls(
            c=_torch_to_np(hz.c),
            Gc=_torch_to_csr(hz.Gc),
            Gb=_torch_to_csr(hz.Gb),
            Ac=_torch_to_csr(Ace),
            Ab=_torch_to_csr(Abe),
            b=_torch_to_np(be),
            Auc=_torch_to_csr(Acl),
            Aub=_torch_to_csr(Abl),
            ub=_torch_to_np(bl),
            col_ids=(
                None
                if hz.col_ids is None
                else hz.col_ids.detach().cpu().numpy().astype(np.int64, copy=True)
            ),
            bcol_ids=(
                None
                if hz.bcol_ids is None
                else hz.bcol_ids.detach().cpu().numpy().astype(np.int64, copy=True)
            ),
        )
        if getattr(hz, "_solver_known_nonempty", False):
            setattr(out, "_solver_known_nonempty", True)
            setattr(
                out,
                "_solver_known_nonempty_reason",
                getattr(hz, "_solver_known_nonempty_reason", "dense_conversion"),
            )
        return out

    @property
    def n_out(self) -> int:
        return int(self.c.size)

    @property
    def n_cont(self) -> int:
        return int(self.Gc.shape[1])

    @property
    def n_bin(self) -> int:
        return int(self.Gb.shape[1])

    @property
    def n_eq(self) -> int:
        return int(self.Ac.shape[0])

    @property
    def n_ub(self) -> int:
        return 0 if self.Auc is None else int(self.Auc.shape[0])

    @property
    def value_nnz(self) -> int:
        return int(self.Gc.nnz + self.Gb.nnz)

    @property
    def eq_nnz(self) -> int:
        return int(self.Ac.nnz + self.Ab.nnz)

    @property
    def ub_nnz(self) -> int:
        if self.Auc is None or self.Aub is None:
            return 0
        return int(self.Auc.nnz + self.Aub.nnz)

    @property
    def constraint_nnz(self) -> int:
        return int(self.eq_nnz + self.ub_nnz)

    def solver_tuple(self):
        """Return arrays in the layout consumed by ``solver_hz``."""

        Auc = self.Auc
        Aub = self.Aub
        ub = self.ub
        if Auc is None or Aub is None or ub is None:
            Auc = sp.csr_matrix((0, self.n_cont), dtype=np.float64)
            Aub = sp.csr_matrix((0, self.n_bin), dtype=np.float64)
            ub = np.zeros(0, dtype=np.float64)
        return (
            self.c,
            self.Gc,
            self.Gb,
            self.Ac,
            self.Ab,
            self.b,
            Auc,
            Aub,
            ub,
        )


class _HZImmutableMapping(Mapping[Any, Any]):
    """Small deeply immutable mapping used by proof-bearing receipts."""

    __slots__ = ("_items",)

    def __init__(self, items: Sequence[Tuple[Any, Any]]) -> None:
        object.__setattr__(self, "_items", tuple(items))

    def __getitem__(self, key: Any) -> Any:
        for stored_key, value in self._items:
            if stored_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("proof-bearing mappings are immutable")

    def __deepcopy__(self, _memo):
        return self


class _HZConditionalPropertyRowsSeal:
    """Unique immutable seal binding producer authority to one parent hash."""

    __slots__ = ("_live_content_sha256",)

    def __init__(
        self,
        live_content_sha256: str,
        *,
        _producer_capability: Any,
    ) -> None:
        if (
            _producer_capability
            is not _HZ_CONDITIONAL_PROPERTY_ROWS_PRODUCER_TOKEN
        ):
            raise PermissionError(
                "conditional property seal requires the private producer"
            )
        object.__setattr__(
            self,
            "_live_content_sha256",
            str(live_content_sha256),
        )

    @property
    def live_content_sha256(self) -> str:
        return self._live_content_sha256

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("conditional property seals are immutable")

    def __deepcopy__(self, _memo):
        return self


def _hz_freeze_conditional_value(value: Any) -> Any:
    """Return a detached immutable form of JSON-like proof metadata."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(
                "conditional property receipt contains a non-finite float"
            )
        return value
    if isinstance(value, np.generic):
        return _hz_freeze_conditional_value(value.item())
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value).copy()
        if array.dtype.kind not in {"b", "i", "u", "f"}:
            raise TypeError(
                "conditional property receipt array has an unsupported dtype"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError(
                "conditional property receipt contains a non-finite array"
            )
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        frozen_items = []
        raw_keys = list(value)
        if any(
            not isinstance(raw_key, (str, int))
            or isinstance(raw_key, bool)
            for raw_key in raw_keys
        ):
            raise TypeError(
                "conditional property receipt keys must be strings or integers"
            )
        raw_keys.sort(
            key=lambda raw_key: (
                0 if isinstance(raw_key, str) else 1,
                raw_key,
            )
        )
        for raw_key in raw_keys:
            if not isinstance(raw_key, (str, int)):
                raise TypeError(
                    "conditional property receipt key is unsupported"
                )
            frozen_items.append(
                (
                    raw_key,
                    _hz_freeze_conditional_value(value[raw_key]),
                )
            )
        return _HZImmutableMapping(frozen_items)
    if isinstance(value, (list, tuple)):
        return tuple(_hz_freeze_conditional_value(item) for item in value)
    raise TypeError(
        "conditional property receipt contains an unsupported value"
    )


def _hz_hash_conditional_value(
    digest: "hashlib._Hash",
    value: Any,
) -> None:
    """Hash one immutable proof value with explicit type/length framing."""

    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, (bool, np.bool_)):
        digest.update(b"B1" if bool(value) else b"B0")
        return
    if isinstance(value, (int, np.integer)):
        payload = str(int(value)).encode("ascii")
        digest.update(b"I" + len(payload).to_bytes(8, "little") + payload)
        return
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError("cannot hash a non-finite proof float")
        payload = scalar.hex().encode("ascii")
        digest.update(b"F" + len(payload).to_bytes(8, "little") + payload)
        return
    if isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"S" + len(payload).to_bytes(8, "little") + payload)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind not in {"b", "i", "u", "f"}:
            raise TypeError("cannot hash an unsupported proof array")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("cannot hash a non-finite proof array")
        dtype = array.dtype.str.encode("ascii")
        digest.update(b"A" + len(dtype).to_bytes(8, "little") + dtype)
        digest.update(int(array.ndim).to_bytes(8, "little"))
        digest.update(
            np.asarray(array.shape, dtype=np.int64).tobytes(order="C")
        )
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        return
    if isinstance(value, Mapping):
        keys = list(value)
        if any(
            not isinstance(key, (str, int))
            or isinstance(key, bool)
            for key in keys
        ):
            raise TypeError("proof mapping keys must be strings or integers")
        keys.sort(
            key=lambda key: (
                0 if isinstance(key, str) else 1,
                key,
            )
        )
        digest.update(b"M" + len(keys).to_bytes(8, "little"))
        for key in keys:
            _hz_hash_conditional_value(digest, key)
            _hz_hash_conditional_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"L" + len(value).to_bytes(8, "little"))
        for item in value:
            _hz_hash_conditional_value(digest, item)
        return
    raise TypeError("cannot hash an unsupported proof value")


def _hz_readonly_conditional_array(
    values: Any,
    *,
    dtype,
) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _hz_readonly_conditional_csr(matrix: Any) -> sp.csr_matrix:
    csr = sp.csr_matrix(matrix, dtype=np.float64, copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    csr.eliminate_zeros()
    csr.data.setflags(write=False)
    csr.indices.setflags(write=False)
    csr.indptr.setflags(write=False)
    return csr


def _hz_conditional_parent_content_sha256(
    hz: SparseHZono,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash all proof-relevant parent conditional-plane content."""

    digest = hashlib.sha256()
    digest.update(b"hz_exact_phase_conditional_property_rows_parent_v2")
    _hz_hash_conditional_value(digest, int(hz.n_cont))
    _hz_hash_conditional_value(
        digest,
        ()
        if hz.bcol_ids is None
        else tuple(int(value) for value in hz.bcol_ids),
    )
    _hz_hash_conditional_value(digest, len(rows))
    for item in rows:
        _hz_hash_conditional_value(digest, item["binary_guards"])
        _hz_hash_conditional_value(digest, item["layer_id"])
        _hz_hash_conditional_value(digest, item["row"])
        _hz_hash_conditional_value(digest, item["center"])
        generator = sp.csr_matrix(
            item["generator"], dtype=np.float64, copy=True
        )
        generator.sum_duplicates()
        generator.sort_indices()
        generator.eliminate_zeros()
        _hz_hash_conditional_value(
            digest, tuple(int(value) for value in generator.shape)
        )
        _hz_hash_conditional_value(
            digest,
            np.asarray(generator.indptr, dtype=np.int64),
        )
        _hz_hash_conditional_value(
            digest,
            np.asarray(generator.indices, dtype=np.int64),
        )
        _hz_hash_conditional_value(
            digest,
            np.asarray(generator.data, dtype=np.float64),
        )
        _hz_hash_conditional_value(digest, item["error"])
        _hz_hash_conditional_value(digest, item["rival_ids"])
        _hz_hash_conditional_value(digest, item["receipt"])
    return digest.hexdigest()


def _hz_conditional_applied_content_sha256(
    payload: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"hz_exact_phase_conditional_property_rows_child_v2")
    _hz_hash_conditional_value(digest, payload)
    return digest.hexdigest()


def _hz_live_conditional_parent_rows(
    hz: SparseHZono,
) -> Optional[Tuple[Tuple[Mapping[str, Any], ...], str]]:
    """Return live hash-bound rows, fail-closed on any partial/tampered state."""

    names = (
        "_solver_conditional_property_rows_token",
        "_solver_conditional_property_rows",
        "_solver_conditional_property_rows_receipt",
    )
    present = tuple(hasattr(hz, name) for name in names)
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "conditional property parent has an incomplete proof state"
        )
    seal = getattr(hz, "_solver_conditional_property_rows_token", None)
    if not isinstance(seal, _HZConditionalPropertyRowsSeal):
        raise ValueError(
            "conditional property parent capability is invalid"
        )
    rows = getattr(hz, "_solver_conditional_property_rows")
    receipt = getattr(hz, "_solver_conditional_property_rows_receipt")
    if (
        not isinstance(rows, tuple)
        or not rows
        or not isinstance(receipt, Mapping)
        or receipt.get("schema")
        != "hz_exact_phase_conditional_property_rows_parent_v2"
        or set(receipt)
        != {
            "schema",
            "proof_rule",
            "record_count",
            "live_content_sha256",
            "proof_authority",
        }
        or receipt.get("proof_rule")
        != (
            "private_operator_producer+immutable_normalization+"
            "live_parent_content_hash"
        )
        or receipt.get("proof_authority") is not True
        or isinstance(receipt.get("record_count"), (bool, np.bool_))
        or not isinstance(
            receipt.get("record_count"), (int, np.integer)
        )
        or receipt.get("record_count") != len(rows)
        or not isinstance(receipt.get("live_content_sha256"), str)
        or len(receipt["live_content_sha256"]) != 64
    ):
        raise ValueError(
            "conditional property parent receipt is malformed"
        )
    available_ids = (
        set()
        if hz.bcol_ids is None
        else set(int(value) for value in hz.bcol_ids)
    )
    assignment_covers: Dict[
        Tuple[int, ...], set[Tuple[int, ...]]
    ] = {}
    seen = set()
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValueError(
                "conditional property parent record is malformed"
            )
        guards = item.get("binary_guards")
        center = np.asarray(item.get("center"), dtype=np.float64).reshape(-1)
        error = np.asarray(item.get("error"), dtype=np.float64).reshape(-1)
        generator = sp.csr_matrix(
            item.get("generator"), dtype=np.float64, copy=True
        )
        rivals = tuple(int(value) for value in item.get("rival_ids", ()))
        if (
            not isinstance(guards, tuple)
            or not guards
            or isinstance(item.get("layer_id"), (bool, np.bool_))
            or not isinstance(
                item.get("layer_id"), (int, np.integer)
            )
            or isinstance(item.get("row"), (bool, np.bool_))
            or not isinstance(item.get("row"), (int, np.integer))
            or center.size == 0
            or error.shape != center.shape
            or generator.shape != (center.size, hz.n_cont)
            or len(rivals) != center.size
            or len(set(rivals)) != len(rivals)
            or not rivals
            or min(rivals) < 0
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(error))
            or np.any(error < 0.0)
            or (
                generator.nnz
                and not np.all(np.isfinite(generator.data))
            )
            or not isinstance(item.get("receipt"), Mapping)
        ):
            raise ValueError(
                "conditional property parent affine record is malformed"
            )
        guard_key = []
        guard_ids = []
        prior_id = None
        for guard in guards:
            if not isinstance(guard, Mapping):
                raise ValueError(
                    "conditional property parent guard is malformed"
                )
            binary_id = int(guard["binary_col_id"])
            phase = int(guard["phase"])
            if (
                binary_id not in available_ids
                or phase not in {-1, 1}
                or (prior_id is not None and binary_id <= prior_id)
            ):
                raise ValueError(
                    "conditional property parent guard is invalid"
                )
            prior_id = binary_id
            guard_ids.append(binary_id)
            guard_key.append((binary_id, phase))
        guard_key_tuple = tuple(guard_key)
        if guard_key_tuple in seen:
            raise ValueError(
                "conditional property parent has duplicate guards"
            )
        seen.add(guard_key_tuple)
        ids = tuple(guard_ids)
        assignment_covers.setdefault(ids, set()).add(
            tuple(phase for _binary_id, phase in guard_key_tuple)
        )
    if any(
        assignments != set(product((-1, 1), repeat=len(ids)))
        for ids, assignments in assignment_covers.items()
    ):
        raise ValueError(
            "conditional property parent phase cover is incomplete"
        )
    claimed = receipt["live_content_sha256"]
    actual = _hz_conditional_parent_content_sha256(hz, rows)
    if claimed != actual or seal.live_content_sha256 != actual:
        raise ValueError(
            "conditional property parent live-content seal mismatch"
        )
    return rows, actual


def hz_attach_exact_phase_conditional_property_rows(
    hz: SparseHZono,
    rows: Sequence[Mapping[str, Any]],
    *,
    _producer_capability: Any = None,
) -> SparseHZono:
    """Attach builder-authenticated property planes valid on one exact phase.

    The rows are deliberately not part of the parent output.  They become
    available only after :func:`hz_fix_sparse_binary_assignment` fixes the
    matching stable binary id to the recorded phase.  This prevents a
    conditional affine inequality from being consumed outside its guard.
    """

    if (
        _producer_capability
        is not _HZ_CONDITIONAL_PROPERTY_ROWS_PRODUCER_TOKEN
    ):
        raise PermissionError(
            "conditional property rows require the private proof producer"
        )
    if not isinstance(hz, SparseHZono):
        raise TypeError("conditional property rows require SparseHZono")
    if any(
        hasattr(hz, name)
        for name in (
            "_solver_conditional_property_rows_token",
            "_solver_conditional_property_rows",
            "_solver_conditional_property_rows_receipt",
        )
    ):
        raise ValueError(
            "conditional property rows are already attached"
        )
    normalized = []
    seen = set()
    available_ids = (
        set()
        if hz.bcol_ids is None
        else set(int(value) for value in hz.bcol_ids)
    )
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("conditional property row record must be a mapping")
        raw_guards = raw.get("binary_guards")
        if raw_guards is None:
            raw_guards = (
                {
                    "binary_col_id": raw["binary_col_id"],
                    "phase": raw["phase"],
                    "layer_id": raw["layer_id"],
                    "row": raw["row"],
                },
            )
        guards = []
        guard_ids = set()
        for raw_guard in raw_guards:
            if not isinstance(raw_guard, Mapping):
                raise TypeError(
                    "conditional property binary guard must be a mapping"
                )
            guard = {
                "binary_col_id": int(raw_guard["binary_col_id"]),
                "phase": int(raw_guard["phase"]),
                "layer_id": int(raw_guard["layer_id"]),
                "row": int(raw_guard["row"]),
            }
            if (
                guard["binary_col_id"] not in available_ids
                or guard["phase"] not in {-1, 1}
                or guard["binary_col_id"] in guard_ids
            ):
                raise ValueError(
                    "conditional property row has invalid binary guards"
                )
            guard_ids.add(guard["binary_col_id"])
            guards.append(guard)
        if not guards:
            raise ValueError(
                "conditional property row requires at least one guard"
            )
        guards.sort(key=lambda item: item["binary_col_id"])
        key = tuple(
            (guard["binary_col_id"], guard["phase"])
            for guard in guards
        )
        if key in seen:
            raise ValueError("duplicate conditional property phase record")
        seen.add(key)
        layer_id = int(raw.get("layer_id", guards[0]["layer_id"]))
        neuron = int(raw.get("row", guards[0]["row"]))
        center = np.asarray(raw["center"], dtype=np.float64).reshape(-1)
        generator = sp.csr_matrix(
            raw["generator"], dtype=np.float64
        )
        if generator.shape[0] != center.size or generator.shape[1] > hz.n_cont:
            raise ValueError(
                "conditional property generator has invalid shape"
            )
        if generator.shape[1] < hz.n_cont:
            generator = sp.hstack(
                [
                    generator,
                    sp.csr_matrix(
                        (
                            center.size,
                            hz.n_cont - generator.shape[1],
                        ),
                        dtype=np.float64,
                    ),
                ],
                format="csr",
            )
        error = np.asarray(raw["error"], dtype=np.float64).reshape(-1)
        rival_ids = tuple(
            int(value)
            for value in raw.get(
                "rival_ids", tuple(range(int(center.size)))
            )
        )
        if (
            center.size == 0
            or error.shape != center.shape
            or len(rival_ids) != center.size
            or len(set(rival_ids)) != len(rival_ids)
            or min(rival_ids) < 0
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(error))
            or np.any(error < 0.0)
            or (generator.nnz and not np.all(np.isfinite(generator.data)))
        ):
            raise ValueError("conditional property affine rows are malformed")
        immutable_guards = tuple(
            _HZImmutableMapping(
                tuple(
                    (name, int(guard[name]))
                    for name in (
                        "binary_col_id",
                        "phase",
                        "layer_id",
                        "row",
                    )
                )
            )
            for guard in guards
        )
        normalized.append(
            _HZImmutableMapping(
                (
                    ("binary_guards", immutable_guards),
                    ("layer_id", layer_id),
                    ("row", neuron),
                    (
                        "center",
                        _hz_readonly_conditional_array(
                            center, dtype=np.float64
                        ),
                    ),
                    (
                        "generator",
                        _hz_readonly_conditional_csr(generator),
                    ),
                    (
                        "error",
                        _hz_readonly_conditional_array(
                            error, dtype=np.float64
                        ),
                    ),
                    ("rival_ids", rival_ids),
                    (
                        "receipt",
                        _hz_freeze_conditional_value(
                            raw.get("receipt", {})
                        ),
                    ),
                )
            )
        )
    if normalized:
        assignment_covers: Dict[
            Tuple[int, ...], set[Tuple[int, ...]]
        ] = {}
        for item in normalized:
            ids = tuple(
                int(guard["binary_col_id"])
                for guard in item["binary_guards"]
            )
            assignment = tuple(
                int(guard["phase"])
                for guard in item["binary_guards"]
            )
            assignment_covers.setdefault(ids, set()).add(assignment)
        for ids, assignments in assignment_covers.items():
            expected = set(product((-1, 1), repeat=len(ids)))
            if assignments != expected:
                raise ValueError(
                    "conditional property rows require a complete exact "
                    "phase cover for every guard set"
                )
        if not assignment_covers:
            raise ValueError(
                "conditional property rows have no exact phase cover"
            )
        normalized.sort(
            key=lambda item: tuple(
                (
                    int(guard["binary_col_id"]),
                    int(guard["phase"]),
                )
                for guard in item["binary_guards"]
            )
        )
        immutable_rows = tuple(normalized)
        content_sha256 = _hz_conditional_parent_content_sha256(
            hz, immutable_rows
        )
        parent_receipt = _HZImmutableMapping(
            (
                (
                    "schema",
                    "hz_exact_phase_conditional_property_rows_parent_v2",
                ),
                (
                    "proof_rule",
                    "private_operator_producer+immutable_normalization+"
                    "live_parent_content_hash",
                ),
                ("record_count", len(immutable_rows)),
                ("live_content_sha256", content_sha256),
                ("proof_authority", True),
            )
        )
        setattr(hz, "_solver_conditional_property_rows", immutable_rows)
        setattr(
            hz,
            "_solver_conditional_property_rows_receipt",
            parent_receipt,
        )
        setattr(
            hz,
            "_solver_conditional_property_rows_token",
            _HZConditionalPropertyRowsSeal(
                content_sha256,
                _producer_capability=(
                    _HZ_CONDITIONAL_PROPERTY_ROWS_PRODUCER_TOKEN
                ),
            ),
        )
    return hz


def _hz_attach_exact_phase_conditional_property_rows_from_operator(
    hz: SparseHZono,
    rows: Sequence[Mapping[str, Any]],
) -> SparseHZono:
    """Private proof-producing entry used by Operator-HZ and controlled toys."""

    return hz_attach_exact_phase_conditional_property_rows(
        hz,
        rows,
        _producer_capability=(
            _HZ_CONDITIONAL_PROPERTY_ROWS_PRODUCER_TOKEN
        ),
    )


def _hz_apply_exact_phase_conditional_property_rows(
    parent: SparseHZono,
    child: SparseHZono,
    *,
    fixed_ids: Sequence[Optional[int]],
    fixed_values: Sequence[int],
) -> SparseHZono:
    live_parent = _hz_live_conditional_parent_rows(parent)
    if live_parent is None:
        return child
    parent_rows, parent_content_sha256 = live_parent
    fixed = {
        int(col_id): int(value)
        for col_id, value in zip(fixed_ids, fixed_values)
        if col_id is not None
    }
    applicable = [
        item
        for item in parent_rows
        if all(
            fixed.get(int(guard["binary_col_id"]))
            == int(guard["phase"])
            for guard in item["binary_guards"]
        )
    ]
    if not applicable:
        return child

    old_out = int(child.n_out)
    old_cont = int(child.n_cont)
    added_rows = sum(int(item["center"].size) for item in applicable)
    error_rows = []
    error_values = []
    conditional_centers = []
    conditional_generators = []
    rival_to_rows: Dict[int, List[int]] = {}
    cursor = old_out
    for item in applicable:
        center = np.asarray(item["center"], dtype=np.float64).reshape(-1)
        generator = sp.csr_matrix(
            item["generator"],
            dtype=np.float64,
            shape=(center.size, old_cont),
        )
        error = np.asarray(item["error"], dtype=np.float64).reshape(-1)
        conditional_centers.append(center)
        conditional_generators.append(generator)
        for local_row, rival in enumerate(item["rival_ids"]):
            rival_to_rows.setdefault(int(rival), []).append(
                int(cursor + local_row)
            )
        nz_error = np.flatnonzero(error > 0.0)
        error_rows.extend(int(cursor + row) for row in nz_error)
        error_values.extend(float(error[row]) for row in nz_error)
        cursor += int(center.size)

    error_count = len(error_rows)
    new_cont = old_cont + error_count
    base_G = sp.hstack(
        [
            child.Gc,
            sp.csr_matrix((old_out, error_count), dtype=np.float64),
        ],
        format="csr",
    )
    conditional_G = sp.vstack(conditional_generators, format="csr")
    conditional_G = sp.hstack(
        [
            conditional_G,
            sp.csr_matrix(
                (added_rows, error_count), dtype=np.float64
            ),
        ],
        format="csr",
    )
    Gc = sp.vstack([base_G, conditional_G], format="lil")
    if error_count:
        for local_col, (output_row, value) in enumerate(
            zip(error_rows, error_values)
        ):
            Gc[int(output_row), old_cont + local_col] = float(value)
    Gc = Gc.tocsr()
    Gc.eliminate_zeros()
    zero_output_binary = sp.csr_matrix(
        (added_rows, child.n_bin), dtype=np.float64
    )
    Gb = sp.vstack([child.Gb, zero_output_binary], format="csr")
    Ac = sp.hstack(
        [
            child.Ac,
            sp.csr_matrix((child.n_eq, error_count), dtype=np.float64),
        ],
        format="csr",
    )
    Auc = (
        None
        if child.Auc is None
        else sp.hstack(
            [
                child.Auc,
                sp.csr_matrix(
                    (child.n_ub, error_count), dtype=np.float64
                ),
            ],
            format="csr",
        )
    )
    fresh_ids = (
        np.zeros(0, dtype=np.int64)
        if error_count == 0
        else hz_fresh_col_ids(
            error_count, device="cpu"
        ).detach().cpu().numpy().astype(np.int64, copy=False)
    )
    col_ids = (
        None
        if child.col_ids is None
        else np.concatenate([child.col_ids, fresh_ids])
    )
    augmented = SparseHZono(
        c=np.concatenate(
            [
                np.asarray(child.c, dtype=np.float64),
                *conditional_centers,
            ]
        ),
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=child.Ab.copy(),
        b=child.b.copy(),
        Auc=Auc,
        Aub=None if child.Aub is None else child.Aub.copy(),
        ub=None if child.ub is None else child.ub.copy(),
        col_ids=col_ids,
        bcol_ids=(
            None if child.bcol_ids is None else child.bcol_ids.copy()
        ),
    )
    for name, value in vars(child).items():
        if name in vars(augmented):
            continue
        setattr(augmented, name, value)
    raw_column_layers = getattr(
        child, "_solver_continuous_column_layer_ids", None
    )
    if raw_column_layers is not None:
        column_layers = np.asarray(
            raw_column_layers, dtype=np.int64
        ).reshape(-1)
        if column_layers.size != old_cont:
            raise ValueError(
                "conditional property column provenance mismatch"
            )
        setattr(
            augmented,
            "_solver_continuous_column_layer_ids",
            np.concatenate(
                [
                    column_layers,
                    np.full(error_count, -2, dtype=np.int64),
                ]
            ),
        )
    applied_payload: Dict[str, Any] = {
        "schema": "hz_exact_phase_conditional_property_rows_child_v2",
        "proof_rule": (
            "live_parent_content_hash+exact_binary_phase_guard+"
            "independently_replayed_suffix_upper_plane+"
            "explicit_roundoff_generators+live_child_map_hash"
        ),
        "parent_live_content_sha256": parent_content_sha256,
        "fixed_binary_assignments": tuple(
            (int(col_id), int(value))
            for col_id, value in sorted(fixed.items())
        ),
        "parent_output_rows": int(old_out),
        "conditional_output_rows": int(added_rows),
        "error_generators": int(error_count),
        "applied_guard_sets": tuple(
            tuple(
                _HZImmutableMapping(
                    tuple(
                        (name, int(guard[name]))
                        for name in (
                            "binary_col_id",
                            "phase",
                            "layer_id",
                            "row",
                        )
                    )
                )
                for guard in item["binary_guards"]
            )
            for item in applicable
        ),
        "rival_to_output_rows": _HZImmutableMapping(
            tuple(
                (
                    int(rival),
                    tuple(int(row) for row in rows),
                )
                for rival, rows in sorted(rival_to_rows.items())
            )
        ),
        "proof_authority": True,
    }

    # Constraint-prefix receipts hash the full continuous matrix shape, so
    # adding output-only error columns requires deterministic re-binding.
    raw_prefix = getattr(
        augmented, "_solver_row_constraint_prefix_frames", None
    )
    if isinstance(raw_prefix, dict):
        rebound = {}
        prefix_hash_cache: Dict[
            Tuple[int, int], Tuple[str, str]
        ] = {}
        for raw_row, raw_entry in raw_prefix.items():
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            eq_rows = int(entry["eq_rows"])
            ub_rows = int(entry["ub_rows"])
            prefix_key = (eq_rows, ub_rows)
            hashes = prefix_hash_cache.get(prefix_key)
            if hashes is None:
                hashes = (
                    _solver_csr_sha256(augmented.Ac[:eq_rows, :]),
                    _solver_csr_sha256(
                        (
                            sp.csr_matrix(
                                (0, new_cont), dtype=np.float64
                            )
                            if augmented.Auc is None
                            else augmented.Auc[:ub_rows, :]
                        )
                    ),
                )
                prefix_hash_cache[prefix_key] = hashes
            entry["eq_csr_sha256"], entry["ub_csr_sha256"] = hashes
            rebound[int(raw_row)] = entry
        # Every conditional row uses the same stop prefix as the existing
        # shared-suffix rows.  Clone one validated frame per new output row.
        template = next(iter(rebound.values()), None)
        if isinstance(template, dict):
            for rows in rival_to_rows.values():
                for output_row in rows:
                    entry = dict(template)
                    entry["spec_row"] = int(output_row)
                    entry["output_row"] = int(output_row)
                    rebound[int(output_row)] = entry
        setattr(
            augmented, "_solver_row_constraint_prefix_frames", rebound
        )
    applied_hash = _hz_conditional_applied_content_sha256(
        applied_payload
    )
    applied_payload["live_content_sha256"] = applied_hash
    setattr(
        augmented,
        "_solver_conditional_property_rows_applied",
        _hz_freeze_conditional_value(applied_payload),
    )
    return augmented


def _hz_fraction_binary64_with_error(
    exact: Fraction,
    *,
    name: str,
    row: int,
) -> Tuple[float, float]:
    """Return a finite binary64 nominal and an outward absolute error."""

    try:
        nominal = float(exact)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name} row {row} cannot be represented by finite binary64"
        ) from exc
    if not np.isfinite(nominal):
        raise ValueError(
            f"{name} row {row} cannot be represented by finite binary64"
        )
    residual = abs(exact - Fraction.from_float(nominal))
    if residual == 0:
        return nominal, 0.0
    try:
        radius = float(residual)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name} row {row} roundoff radius overflowed"
        ) from exc
    if Fraction.from_float(radius) < residual:
        radius = float(np.nextafter(radius, np.inf))
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or Fraction.from_float(radius) < residual
    ):
        raise ValueError(
            f"{name} row {row} roundoff radius cannot be widened safely"
        )
    return nominal, radius


def _hz_fraction_binary64_upper(
    exact: Fraction,
    *,
    name: str,
    row: int,
) -> Tuple[float, bool]:
    """Round an exact dyadic RHS toward ``+inf`` or fail closed."""

    try:
        upper = float(exact)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name} row {row} cannot be outward-rounded finitely"
        ) from exc
    if not np.isfinite(upper):
        raise ValueError(
            f"{name} row {row} cannot be outward-rounded finitely"
        )
    if Fraction.from_float(upper) < exact:
        upper = float(np.nextafter(upper, np.inf))
    if (
        not np.isfinite(upper)
        or Fraction.from_float(upper) < exact
    ):
        raise ValueError(
            f"{name} row {row} cannot be outward-rounded finitely"
        )
    return upper, Fraction.from_float(upper) != exact


def _hz_fixed_binary_affine_shift(
    base,
    binary_matrix,
    *,
    fixed_positions: np.ndarray,
    fixed_values: np.ndarray,
    subtract: bool,
    mode: str,
    name: str,
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[float, ...]]:
    """Apply a fixed signed-binary shift with exact dyadic accumulation.

    ``mode`` is ``"affine"`` for a nominal plus explicit absolute-error
    generators, or ``"upper"`` for one-sided RHS rounding toward ``+inf``.
    """

    if mode not in {"affine", "upper"}:
        raise ValueError(f"unsupported fixed-binary shift mode {mode!r}")
    values = np.asarray(base, dtype=np.float64).reshape(-1)
    matrix = sp.csr_matrix(binary_matrix, dtype=np.float64, copy=False)
    if matrix.shape[0] != values.size:
        raise ValueError(
            f"{name} row mismatch: matrix={matrix.shape}, base={values.shape}"
        )
    if not np.all(np.isfinite(values)) or (
        matrix.nnz and not np.all(np.isfinite(matrix.data))
    ):
        raise ValueError(f"{name} received non-finite stored data")
    if not matrix.has_canonical_format:
        raise ValueError(
            f"{name} requires canonical CSR without duplicate entries"
        )
    if fixed_positions.size != fixed_values.size:
        raise ValueError(f"{name} fixed assignment shape mismatch")

    selected = matrix[:, fixed_positions].tocsr(copy=False)
    out = values.copy()
    affected_rows: List[int] = []
    error_or_marker: List[float] = []
    for row in range(matrix.shape[0]):
        start = int(selected.indptr[row])
        end = int(selected.indptr[row + 1])
        if start == end:
            continue
        exact = Fraction.from_float(float(values[row]))
        for offset in range(start, end):
            local_col = int(selected.indices[offset])
            term = (
                Fraction.from_float(float(selected.data[offset]))
                * int(fixed_values[local_col])
            )
            exact = exact - term if subtract else exact + term
        if mode == "upper":
            rounded, widened = _hz_fraction_binary64_upper(
                exact,
                name=name,
                row=row,
            )
            out[row] = rounded
            if widened:
                affected_rows.append(int(row))
                error_or_marker.append(0.0)
        else:
            nominal, radius = _hz_fraction_binary64_with_error(
                exact,
                name=name,
                row=row,
            )
            out[row] = nominal
            if radius > 0.0:
                affected_rows.append(int(row))
                error_or_marker.append(float(radius))
    return out, tuple(affected_rows), tuple(error_or_marker)


def _hz_fresh_sparse_col_ids(
    count: int,
    existing: np.ndarray,
) -> np.ndarray:
    """Reserve enough global ids that ``count`` cannot collide with a parent."""

    count = int(count)
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    forbidden = {
        int(value)
        for value in np.asarray(existing, dtype=np.int64).reshape(-1)
    }
    candidates = (
        hz_fresh_col_ids(
            count + len(forbidden),
            device="cpu",
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    selected = [
        int(value) for value in candidates if int(value) not in forbidden
    ][:count]
    if len(selected) != count:
        raise ValueError("cannot allocate collision-free roundoff column ids")
    return np.asarray(selected, dtype=np.int64)


def _hz_phase_deadline_check(
    deadline: Optional[float],
    *,
    stage: str,
) -> None:
    """Raise ``TimeoutError`` once an optional absolute deadline expires."""

    if deadline is None:
        return
    if isinstance(deadline, (bool, np.bool_)):
        raise TypeError("binary phase deadline must be numeric")
    try:
        absolute = float(deadline)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("binary phase deadline must be numeric") from exc
    if not np.isfinite(absolute):
        raise ValueError("binary phase deadline must be finite")
    if time.monotonic() >= absolute:
        raise TimeoutError(f"binary phase deadline expired at {stage}")


def hz_fix_sparse_binary_assignment(
    hz: SparseHZono,
    assignment: Mapping[int, int] | Sequence[Tuple[int, int]],
    *,
    deadline: Optional[float] = None,
) -> SparseHZono:
    """Fix selected ``{-1,+1}`` factors with a sound outward projection.

    For ``z = c + Gc*xc + Gb*xb`` and constraint rows

    ``Ac*xc + Ab*xb == b`` and ``Auc*xc + Aub*xb <= ub``,

    The fixed shifts are accumulated exactly as dyadic rationals.  A
    non-representable output center or equality RHS gets a fresh continuous
    error factor whose binary64 radius is rounded outward.  An inequality RHS
    is rounded only toward ``+inf``.  Thus every exact fixed-phase point is in
    its child even when its shifted constants are not representable as one
    binary64 value.  Enumerating both values of every selected factor gives a
    sound cover of the parent HZ.

    A child is deliberately *not* marked constructively non-empty: a fixed
    phase can be empty even when its parent is not.  The ordinary base
    feasibility firewall must find and exactly recheck a child witness before
    that child can participate in a SAFE proof.
    """

    if not isinstance(hz, SparseHZono):
        raise TypeError("binary phase fixing requires SparseHZono")
    _hz_phase_deadline_check(deadline, stage="phase_fix_entry")
    raw_items = (
        list(assignment.items())
        if isinstance(assignment, Mapping)
        else list(assignment)
    )
    if not raw_items:
        raise ValueError("binary phase assignment must not be empty")

    normalized: Dict[int, int] = {}
    for raw_position, raw_value in raw_items:
        if isinstance(raw_position, (bool, np.bool_)) or not isinstance(
            raw_position, (int, np.integer)
        ):
            raise TypeError("binary phase position must be an integer")
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, (int, np.integer)
        ):
            raise TypeError("binary phase value must be an integer")
        position = int(raw_position)
        value = int(raw_value)
        if not 0 <= position < hz.n_bin:
            raise ValueError(
                f"binary phase position {position} is outside [0, {hz.n_bin})"
            )
        if value not in {-1, 1}:
            raise ValueError("binary phase values must be exactly -1 or +1")
        if position in normalized and normalized[position] != value:
            raise ValueError(
                f"binary phase position {position} has conflicting values"
            )
        normalized[position] = value

    fixed_positions = np.asarray(sorted(normalized), dtype=np.int64)
    fixed_values = np.asarray(
        [normalized[int(position)] for position in fixed_positions],
        dtype=np.int64,
    )
    keep_mask = np.ones(hz.n_bin, dtype=bool)
    keep_mask[fixed_positions] = False
    keep_positions = np.flatnonzero(keep_mask).astype(np.int64, copy=False)
    _hz_phase_deadline_check(
        deadline, stage="phase_fix_after_assignment"
    )

    stored_matrices = (
        ("Gc", hz.Gc),
        ("Gb", hz.Gb),
        ("Ac", hz.Ac),
        ("Ab", hz.Ab),
    )
    if hz.Auc is not None:
        stored_matrices += (("Auc", hz.Auc), ("Aub", hz.Aub))
    for matrix_name, matrix in stored_matrices:
        matrix = sp.csr_matrix(matrix, dtype=np.float64, copy=False)
        if matrix.nnz and not np.all(np.isfinite(matrix.data)):
            raise ValueError(
                "binary phase fixing received non-finite "
                f"{matrix_name} data"
            )
        if matrix.nnz and np.any(matrix.data == 0.0):
            raise ValueError(
                "binary phase fixing requires CSR without explicit zeros "
                f"in {matrix_name}"
            )
        if not matrix.has_canonical_format:
            raise ValueError(
                "binary phase fixing requires canonical CSR without "
                f"duplicate entries in {matrix_name}"
            )

    c, center_error_rows, center_error_values = (
        _hz_fixed_binary_affine_shift(
            hz.c,
            hz.Gb,
            fixed_positions=fixed_positions,
            fixed_values=fixed_values,
            subtract=False,
            mode="affine",
            name="fixed output center",
        )
    )
    b, equality_error_rows, equality_error_values = (
        _hz_fixed_binary_affine_shift(
            hz.b,
            hz.Ab,
            fixed_positions=fixed_positions,
            fixed_values=fixed_values,
            subtract=True,
            mode="affine",
            name="fixed equality RHS",
        )
    )
    center_error_count = len(center_error_rows)
    equality_error_count = len(equality_error_rows)
    roundoff_count = center_error_count + equality_error_count
    old_cont = int(hz.n_cont)
    new_cont = old_cont + roundoff_count
    _hz_phase_deadline_check(
        deadline, stage="phase_fix_after_exact_shifts"
    )

    center_error_block = sp.csr_matrix(
        (
            np.asarray(center_error_values, dtype=np.float64),
            (
                np.asarray(center_error_rows, dtype=np.int64),
                np.arange(center_error_count, dtype=np.int64),
            ),
        ),
        shape=(hz.n_out, roundoff_count),
        dtype=np.float64,
    )
    Gc = sp.hstack([hz.Gc, center_error_block], format="csr")

    equality_error_block = sp.csr_matrix(
        (
            np.asarray(equality_error_values, dtype=np.float64),
            (
                np.asarray(equality_error_rows, dtype=np.int64),
                center_error_count
                + np.arange(equality_error_count, dtype=np.int64),
            ),
        ),
        shape=(hz.n_eq, roundoff_count),
        dtype=np.float64,
    )
    Ac = sp.hstack([hz.Ac, equality_error_block], format="csr")

    if hz.Auc is None:
        Auc = None
        Aub = None
        ub = None
        upper_outward_rows: Tuple[int, ...] = ()
    else:
        _hz_phase_deadline_check(
            deadline, stage="phase_fix_before_upper_copy"
        )
        Auc = sp.hstack(
            [
                hz.Auc,
                sp.csr_matrix(
                    (hz.n_ub, roundoff_count),
                    dtype=np.float64,
                ),
            ],
            format="csr",
        )
        Aub = hz.Aub[:, keep_positions].tocsr()
        ub, upper_outward_rows, _unused_upper_markers = (
            _hz_fixed_binary_affine_shift(
                hz.ub,
                hz.Aub,
                fixed_positions=fixed_positions,
                fixed_values=fixed_values,
                subtract=True,
                mode="upper",
                name="fixed upper RHS",
            )
        )

    fresh_ids = (
        np.zeros(0, dtype=np.int64)
        if hz.col_ids is None
        else _hz_fresh_sparse_col_ids(roundoff_count, hz.col_ids)
    )
    col_ids = (
        None
        if hz.col_ids is None
        else np.concatenate([hz.col_ids.copy(), fresh_ids])
    )
    child = SparseHZono(
        c=c,
        Gc=Gc,
        Gb=hz.Gb[:, keep_positions].tocsr(),
        Ac=Ac,
        Ab=hz.Ab[:, keep_positions].tocsr(),
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        col_ids=col_ids,
        bcol_ids=(
            None
            if hz.bcol_ids is None
            else hz.bcol_ids[keep_positions].copy()
        ),
    )

    fixed_ids = (
        [None] * int(fixed_positions.size)
        if hz.bcol_ids is None
        else [
            int(hz.bcol_ids[int(position)])
            for position in fixed_positions
        ]
    )
    setattr(
        child,
        "_solver_binary_phase_fix",
        {
            "schema": "sparse_hz_binary_phase_fix_v2",
            "proof_rule": (
                "exact_fraction_fixed_binary_substitution_with_explicit_"
                "center_and_equality_roundoff_generators_and_upper_rhs_"
                "rounding_toward_positive_infinity;all_sign_assignments_"
                "form_sound_parent_cover"
            ),
            "projection_relation": (
                "exact_fixed_phase_projection_subset_of_child"
            ),
            "arithmetic": "Fraction.from_float_exact_dyadic",
            "parent_n_bin": int(hz.n_bin),
            "child_n_bin": int(child.n_bin),
            "parent_n_cont": int(hz.n_cont),
            "child_n_cont": int(child.n_cont),
            "fixed_positions": [
                int(position) for position in fixed_positions
            ],
            "fixed_values": [int(value) for value in fixed_values],
            "fixed_bcol_ids": fixed_ids,
            "center_roundoff_generator_rows": [
                int(row) for row in center_error_rows
            ],
            "center_roundoff_radii_hex": [
                float(value).hex() for value in center_error_values
            ],
            "equality_rhs_roundoff_generator_rows": [
                int(row) for row in equality_error_rows
            ],
            "equality_rhs_roundoff_radii_hex": [
                float(value).hex() for value in equality_error_values
            ],
            "upper_rhs_outward_rounded_rows": [
                int(row) for row in upper_outward_rows
            ],
            "roundoff_generator_count": int(roundoff_count),
            "roundoff_col_ids": [
                int(value) for value in fresh_ids
            ],
            "proof_authority": True,
        },
    )
    if hz_constructively_nonempty(hz):
        setattr(
            child,
            "_solver_exact_phase_cover_member_token",
            _HZ_EXACT_PHASE_COVER_MEMBER_TOKEN,
        )
        setattr(
            child,
            "_solver_exact_phase_cover_parent_reason",
            getattr(
                hz,
                "_solver_constructive_nonempty_reason",
                "constructive_parent",
            ),
        )

    # C16/C17 row-prefix scheduling remains valid after exact substitution.
    # Continuous matrices and row order are unchanged; binary coefficients are
    # consumed into the stored right-hand sides above.  Update only the exact
    # number of binary factors which existed at each historical prefix.
    raw_prefix = getattr(hz, "_solver_row_constraint_prefix_frames", None)
    if isinstance(raw_prefix, dict):
        child_prefix: Dict[int, Dict[str, Any]] = {}
        # Every property row in one replayed suffix normally points at the
        # same constraint prefix.  Hashing that multi-million-nnz prefix once
        # per output row turns a four-child cover into hundreds of redundant
        # full CSR scans.  The hash is a pure function of the two row counts
        # and the already-fixed child matrices, so one value per distinct
        # prefix is both bit-identical and sufficient for the live validator.
        prefix_hash_cache: Dict[
            Tuple[int, int], Tuple[str, str]
        ] = {}
        for raw_row, raw_entry in raw_prefix.items():
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            try:
                frame_n_bin = int(entry["n_bin"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            removed_before_frame = int(
                np.count_nonzero(fixed_positions < frame_n_bin)
            )
            entry["n_bin"] = int(frame_n_bin - removed_before_frame)
            entry["n_cont"] = int(child.n_cont)
            try:
                eq_rows = int(entry["eq_rows"])
                ub_rows = int(entry["ub_rows"])
                if (
                    not 0 <= eq_rows <= child.n_eq
                    or not 0 <= ub_rows <= child.n_ub
                ):
                    raise ValueError("row-prefix bounds are invalid")
                prefix_key = (eq_rows, ub_rows)
                hashes = prefix_hash_cache.get(prefix_key)
                if hashes is None:
                    _hz_phase_deadline_check(
                        deadline,
                        stage="phase_fix_before_distinct_prefix_hash",
                    )
                    hashes = (
                        _solver_csr_sha256(child.Ac[:eq_rows, :]),
                        _solver_csr_sha256(
                            (
                                sp.csr_matrix(
                                    (0, child.n_cont),
                                    dtype=np.float64,
                                )
                                if child.Auc is None
                                else child.Auc[:ub_rows, :]
                            )
                        ),
                    )
                    prefix_hash_cache[prefix_key] = hashes
                entry["eq_csr_sha256"], entry["ub_csr_sha256"] = hashes
            except (KeyError, TypeError, ValueError, OverflowError):
                # Keep malformed audit metadata non-authoritative.  The
                # downstream validator will reject it without affecting the
                # child feasible set.
                pass
            child_prefix[int(raw_row)] = entry
        setattr(
            child,
            "_solver_row_constraint_prefix_frames",
            child_prefix,
        )

    # These fields are audit/provenance data only.  Copying them does not grant
    # non-emptiness or verdict authority, but keeps witness decoding and
    # experiment receipts bound to the same raw input box.
    for name in (
        "full_col_ids",
        "operator_input_center",
        "operator_input_radius",
        "operator_hz_metadata",
        "_property_full_input_replay_result",
        "_solver_continuous_column_layer_ids",
        "_solver_constraint_row_tags",
    ):
        if hasattr(hz, name):
            value = getattr(hz, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            elif isinstance(value, dict):
                value = dict(value)
            if (
                name == "_solver_continuous_column_layer_ids"
                and roundoff_count > 0
            ):
                value = np.asarray(value, dtype=np.int64).reshape(-1)
                if value.size != hz.n_cont:
                    raise ValueError(
                        "binary phase continuous-column provenance mismatch"
                    )
                value = np.concatenate(
                    [
                        value,
                        np.full(
                            roundoff_count,
                            -2,
                            dtype=np.int64,
                        ),
                    ]
                )
            setattr(child, name, value)
    _hz_phase_deadline_check(
        deadline, stage="phase_fix_before_conditional_rows"
    )
    child = _hz_apply_exact_phase_conditional_property_rows(
        hz,
        child,
        fixed_ids=fixed_ids,
        fixed_values=[int(value) for value in fixed_values],
    )
    _hz_phase_deadline_check(
        deadline, stage="phase_fix_after_conditional_rows"
    )
    return child


def _hz_csr_exact_equal(left, right) -> bool:
    """Bitwise equality of two already-canonical live CSR matrices.

    Phase construction rejects non-canonical parents and creates canonical
    children.  A live matrix that later acquires duplicates, unsorted indices,
    explicit zeros, or another sparse format is therefore a mutation and must
    fail closed rather than be normalized into acceptance.  Comparing the CSR
    buffers in place avoids four full copies per matrix in every child audit.
    """

    if not sp.isspmatrix_csr(left) or not sp.isspmatrix_csr(right):
        return False
    lhs = left
    rhs = right
    for matrix in (lhs, rhs):
        if (
            matrix.dtype != np.dtype(np.float64)
            or not matrix.has_canonical_format
            or not matrix.has_sorted_indices
            or (
                matrix.nnz
                and (
                    not np.all(np.isfinite(matrix.data))
                    or np.any(matrix.data == 0.0)
                )
            )
        ):
            return False
    return bool(
        lhs.shape == rhs.shape
        and np.array_equal(lhs.indptr, rhs.indptr)
        and np.array_equal(lhs.indices, rhs.indices)
        and np.array_equal(lhs.data, rhs.data)
    )


def _hz_csr_exact_prefix_with_trailing_zero_columns(
    live: sp.csr_matrix,
    prefix: sp.csr_matrix,
    *,
    total_columns: int,
) -> bool:
    """Check ``live == hstack(prefix, zero trailing columns)`` in place."""

    if (
        not sp.isspmatrix_csr(live)
        or not sp.isspmatrix_csr(prefix)
        or isinstance(total_columns, (bool, np.bool_))
        or not isinstance(total_columns, (int, np.integer))
    ):
        return False
    total_columns = int(total_columns)
    if (
        total_columns < int(prefix.shape[1])
        or live.shape != (int(prefix.shape[0]), total_columns)
    ):
        return False
    for matrix in (live, prefix):
        if (
            matrix.dtype != np.dtype(np.float64)
            or not matrix.has_canonical_format
            or not matrix.has_sorted_indices
            or (
                matrix.nnz
                and (
                    not np.all(np.isfinite(matrix.data))
                    or np.any(matrix.data == 0.0)
                )
            )
        ):
            return False
    return bool(
        np.array_equal(live.indptr, prefix.indptr)
        and np.array_equal(live.indices, prefix.indices)
        and np.array_equal(live.data, prefix.data)
    )


def hz_verify_sparse_binary_phase_child(
    parent: SparseHZono,
    assignment: Mapping[int, int] | Sequence[Tuple[int, int]],
    child: SparseHZono,
    *,
    deadline: Optional[float] = None,
) -> bool:
    """Independently validate one live outward fixed-phase child.

    This validator trusts neither the public receipt nor copied metadata.  It
    reconstructs all fixed shifts with exact ``Fraction`` arithmetic, checks
    the complete live matrices (including conditional trailing rows), and
    requires the module-private cover capability exactly when the parent has
    the constructive non-emptiness capability.
    """

    try:
        _hz_phase_deadline_check(
            deadline, stage="phase_child_audit_entry"
        )
        if not isinstance(parent, SparseHZono) or not isinstance(
            child, SparseHZono
        ):
            return False
        raw_items = (
            list(assignment.items())
            if isinstance(assignment, Mapping)
            else list(assignment)
        )
        if not raw_items:
            return False
        normalized: Dict[int, int] = {}
        for raw_position, raw_value in raw_items:
            if (
                isinstance(raw_position, (bool, np.bool_))
                or not isinstance(raw_position, (int, np.integer))
                or isinstance(raw_value, (bool, np.bool_))
                or not isinstance(raw_value, (int, np.integer))
            ):
                return False
            position = int(raw_position)
            value = int(raw_value)
            if (
                not 0 <= position < parent.n_bin
                or value not in {-1, 1}
                or (
                    position in normalized
                    and normalized[position] != value
                )
            ):
                return False
            normalized[position] = value
        fixed_positions = np.asarray(
            sorted(normalized),
            dtype=np.int64,
        )
        fixed_values = np.asarray(
            [
                normalized[int(position)]
                for position in fixed_positions
            ],
            dtype=np.int64,
        )
        keep_mask = np.ones(parent.n_bin, dtype=bool)
        keep_mask[fixed_positions] = False
        keep_positions = np.flatnonzero(keep_mask).astype(
            np.int64,
            copy=False,
        )

        expected_c, center_rows, center_radii = (
            _hz_fixed_binary_affine_shift(
                parent.c,
                parent.Gb,
                fixed_positions=fixed_positions,
                fixed_values=fixed_values,
                subtract=False,
                mode="affine",
                name="validated fixed output center",
            )
        )
        expected_b, equality_rows, equality_radii = (
            _hz_fixed_binary_affine_shift(
                parent.b,
                parent.Ab,
                fixed_positions=fixed_positions,
                fixed_values=fixed_values,
                subtract=True,
                mode="affine",
                name="validated fixed equality RHS",
            )
        )
        if parent.Auc is None:
            expected_ub = None
            upper_rows: Tuple[int, ...] = ()
        else:
            expected_ub, upper_rows, _upper_markers = (
                _hz_fixed_binary_affine_shift(
                    parent.ub,
                    parent.Aub,
                    fixed_positions=fixed_positions,
                    fixed_values=fixed_values,
                    subtract=True,
                    mode="upper",
                    name="validated fixed upper RHS",
                )
            )
        _hz_phase_deadline_check(
            deadline, stage="phase_child_audit_after_exact_shifts"
        )

        center_count = len(center_rows)
        equality_count = len(equality_rows)
        substitution_error_count = center_count + equality_count
        base_cont = parent.n_cont + substitution_error_count
        expected_center_error_block = sp.csr_matrix(
            (
                np.asarray(center_radii, dtype=np.float64),
                (
                    np.asarray(center_rows, dtype=np.int64),
                    np.arange(center_count, dtype=np.int64),
                ),
            ),
            shape=(parent.n_out, substitution_error_count),
            dtype=np.float64,
        )
        expected_base_Gc = sp.hstack(
            [parent.Gc, expected_center_error_block],
            format="csr",
        )

        expected_equality_error_block = sp.csr_matrix(
            (
                np.asarray(equality_radii, dtype=np.float64),
                (
                    np.asarray(equality_rows, dtype=np.int64),
                    center_count
                    + np.arange(equality_count, dtype=np.int64),
                ),
            ),
            shape=(parent.n_eq, substitution_error_count),
            dtype=np.float64,
        )
        expected_base_Ac = sp.hstack(
            [parent.Ac, expected_equality_error_block],
            format="csr",
        )
        expected_Ab = parent.Ab[:, keep_positions].tocsr()
        expected_Aub = (
            None
            if parent.Aub is None
            else parent.Aub[:, keep_positions].tocsr()
        )
        expected_base_Gb = parent.Gb[:, keep_positions].tocsr()
        _hz_phase_deadline_check(
            deadline, stage="phase_child_audit_after_base_projection"
        )

        fixed_ids = (
            [None] * int(fixed_positions.size)
            if parent.bcol_ids is None
            else [
                int(parent.bcol_ids[int(position)])
                for position in fixed_positions
            ]
        )
        fixed_by_id = {
            int(col_id): int(value)
            for col_id, value in zip(fixed_ids, fixed_values)
            if col_id is not None
        }
        live_parent_conditional = _hz_live_conditional_parent_rows(
            parent
        )
        if live_parent_conditional is None:
            parent_conditional_rows: Tuple[
                Mapping[str, Any], ...
            ] = ()
            parent_conditional_sha256: Optional[str] = None
        else:
            (
                parent_conditional_rows,
                parent_conditional_sha256,
            ) = live_parent_conditional
        applicable = []
        if parent_conditional_rows:
            applicable = [
                item
                for item in parent_conditional_rows
                if all(
                    fixed_by_id.get(
                        int(guard["binary_col_id"])
                    )
                    == int(guard["phase"])
                    for guard in item["binary_guards"]
                )
            ]

        conditional_centers: List[np.ndarray] = []
        conditional_generators: List[sp.csr_matrix] = []
        conditional_error_rows: List[int] = []
        conditional_error_values: List[float] = []
        rival_to_rows: Dict[int, List[int]] = {}
        output_cursor = int(parent.n_out)
        for item in applicable:
            center = np.asarray(
                item["center"],
                dtype=np.float64,
            ).reshape(-1)
            generator = sp.csr_matrix(
                item["generator"],
                dtype=np.float64,
            )
            if (
                generator.shape
                != (center.size, parent.n_cont)
                or not np.all(np.isfinite(center))
                or (
                    generator.nnz
                    and not np.all(np.isfinite(generator.data))
                )
            ):
                return False
            generator = sp.hstack(
                [
                    generator,
                    sp.csr_matrix(
                        (
                            center.size,
                            substitution_error_count,
                        ),
                        dtype=np.float64,
                    ),
                ],
                format="csr",
            )
            error = np.asarray(
                item["error"],
                dtype=np.float64,
            ).reshape(-1)
            if (
                error.shape != center.shape
                or not np.all(np.isfinite(error))
                or np.any(error < 0.0)
            ):
                return False
            conditional_centers.append(center)
            conditional_generators.append(generator)
            for local_row, rival in enumerate(item["rival_ids"]):
                rival_to_rows.setdefault(int(rival), []).append(
                    output_cursor + int(local_row)
                )
            for local_row in np.flatnonzero(error > 0.0):
                conditional_error_rows.append(
                    output_cursor + int(local_row)
                )
                conditional_error_values.append(
                    float(error[int(local_row)])
                )
            output_cursor += int(center.size)

        conditional_output_count = (
            sum(int(value.size) for value in conditional_centers)
        )
        conditional_error_count = len(conditional_error_rows)
        expected_out = parent.n_out + conditional_output_count
        expected_cont = base_cont + conditional_error_count

        expected_top_Gc = sp.hstack(
            [
                expected_base_Gc,
                sp.csr_matrix(
                    (parent.n_out, conditional_error_count),
                    dtype=np.float64,
                ),
            ],
            format="csr",
        )
        if conditional_output_count:
            expected_bottom_Gc = sp.vstack(
                conditional_generators,
                format="csr",
            )
            expected_bottom_Gc = sp.hstack(
                [
                    expected_bottom_Gc,
                    sp.csr_matrix(
                        (
                            conditional_output_count,
                            conditional_error_count,
                        ),
                        dtype=np.float64,
                    ),
                ],
                format="lil",
            )
            for local_col, (output_row, radius) in enumerate(
                zip(
                    conditional_error_rows,
                    conditional_error_values,
                )
            ):
                expected_bottom_Gc[
                    int(output_row - parent.n_out),
                    base_cont + local_col,
                ] = float(radius)
            expected_bottom_Gc = expected_bottom_Gc.tocsr()
            expected_Gc = sp.vstack(
                [expected_top_Gc, expected_bottom_Gc],
                format="csr",
            )
        else:
            expected_Gc = expected_top_Gc
        expected_Gc.eliminate_zeros()
        expected_Gb = sp.vstack(
            [
                expected_base_Gb,
                sp.csr_matrix(
                    (
                        conditional_output_count,
                        keep_positions.size,
                    ),
                    dtype=np.float64,
                ),
            ],
            format="csr",
        )
        expected_Ac = sp.hstack(
            [
                expected_base_Ac,
                sp.csr_matrix(
                    (parent.n_eq, conditional_error_count),
                    dtype=np.float64,
                ),
            ],
            format="csr",
        )
        expected_full_c = (
            expected_c
            if not conditional_centers
            else np.concatenate(
                [expected_c, *conditional_centers]
            )
        )
        _hz_phase_deadline_check(
            deadline, stage="phase_child_audit_before_live_matrix_compare"
        )

        if (
            child.n_out != expected_out
            or child.n_cont != expected_cont
            or child.n_bin != int(keep_positions.size)
            or not np.array_equal(child.c, expected_full_c)
            or not np.array_equal(child.b, expected_b)
            or not _hz_csr_exact_equal(child.Gc, expected_Gc)
            or not _hz_csr_exact_equal(child.Gb, expected_Gb)
            or not _hz_csr_exact_equal(child.Ac, expected_Ac)
            or not _hz_csr_exact_equal(child.Ab, expected_Ab)
        ):
            return False
        if parent.Auc is None:
            if (
                child.Auc is not None
                or child.Aub is not None
                or child.ub is not None
            ):
                return False
        elif (
            child.Auc is None
            or child.Aub is None
            or child.ub is None
            or not _hz_csr_exact_prefix_with_trailing_zero_columns(
                child.Auc,
                parent.Auc,
                total_columns=expected_cont,
            )
            or not _hz_csr_exact_equal(child.Aub, expected_Aub)
            or not np.array_equal(child.ub, expected_ub)
        ):
            return False

        expected_bcolids = (
            None
            if parent.bcol_ids is None
            else parent.bcol_ids[keep_positions]
        )
        if expected_bcolids is None:
            if child.bcol_ids is not None:
                return False
        elif (
            child.bcol_ids is None
            or not np.array_equal(child.bcol_ids, expected_bcolids)
        ):
            return False
        if parent.col_ids is None:
            if child.col_ids is not None:
                return False
            roundoff_ids: List[int] = []
        else:
            if (
                child.col_ids is None
                or child.col_ids.size != expected_cont
                or not np.array_equal(
                    child.col_ids[: parent.n_cont],
                    parent.col_ids,
                )
                or np.unique(child.col_ids).size != child.col_ids.size
            ):
                return False
            roundoff_ids = [
                int(value)
                for value in child.col_ids[
                    parent.n_cont:base_cont
                ]
            ]

        receipt = getattr(child, "_solver_binary_phase_fix", None)
        proof_rule = (
            "exact_fraction_fixed_binary_substitution_with_explicit_"
            "center_and_equality_roundoff_generators_and_upper_rhs_"
            "rounding_toward_positive_infinity;all_sign_assignments_"
            "form_sound_parent_cover"
        )
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema")
            != "sparse_hz_binary_phase_fix_v2"
            or receipt.get("proof_rule") != proof_rule
            or receipt.get("projection_relation")
            != "exact_fixed_phase_projection_subset_of_child"
            or receipt.get("arithmetic")
            != "Fraction.from_float_exact_dyadic"
            or receipt.get("proof_authority") is not True
            or receipt.get("parent_n_bin") != parent.n_bin
            or receipt.get("child_n_bin") != child.n_bin
            or receipt.get("parent_n_cont") != parent.n_cont
            or receipt.get("child_n_cont") != base_cont
            or receipt.get("fixed_positions")
            != [int(value) for value in fixed_positions]
            or receipt.get("fixed_values")
            != [int(value) for value in fixed_values]
            or receipt.get("fixed_bcol_ids") != fixed_ids
            or receipt.get("center_roundoff_generator_rows")
            != [int(value) for value in center_rows]
            or receipt.get("center_roundoff_radii_hex")
            != [float(value).hex() for value in center_radii]
            or receipt.get(
                "equality_rhs_roundoff_generator_rows"
            )
            != [int(value) for value in equality_rows]
            or receipt.get("equality_rhs_roundoff_radii_hex")
            != [float(value).hex() for value in equality_radii]
            or receipt.get("upper_rhs_outward_rounded_rows")
            != [int(value) for value in upper_rows]
            or receipt.get("roundoff_generator_count")
            != substitution_error_count
            or receipt.get("roundoff_col_ids") != roundoff_ids
        ):
            return False

        parent_authorized = hz_constructively_nonempty(parent)
        child_authorized = (
            getattr(
                child,
                "_solver_exact_phase_cover_member_token",
                None,
            )
            is _HZ_EXACT_PHASE_COVER_MEMBER_TOKEN
        )
        if child_authorized != parent_authorized:
            return False
        if parent_authorized and (
            getattr(
                child,
                "_solver_exact_phase_cover_parent_reason",
                None,
            )
            != getattr(
                parent,
                "_solver_constructive_nonempty_reason",
                "constructive_parent",
            )
        ):
            return False

        applied = getattr(
            child,
            "_solver_conditional_property_rows_applied",
            None,
        )
        if applicable:
            expected_guard_sets = tuple(
                tuple(
                    {
                        "binary_col_id": int(
                            guard["binary_col_id"]
                        ),
                        "phase": int(guard["phase"]),
                        "layer_id": int(guard["layer_id"]),
                        "row": int(guard["row"]),
                    }
                    for guard in item["binary_guards"]
                )
                for item in applicable
            )
            expected_applied_payload: Dict[str, Any] = {
                "schema": (
                    "hz_exact_phase_conditional_property_rows_child_v2"
                ),
                "proof_rule": (
                    "live_parent_content_hash+exact_binary_phase_guard+"
                    "independently_replayed_suffix_upper_plane+"
                    "explicit_roundoff_generators+live_child_map_hash"
                ),
                "parent_live_content_sha256": (
                    parent_conditional_sha256
                ),
                "fixed_binary_assignments": tuple(
                    (int(col_id), int(value))
                    for col_id, value in sorted(fixed_by_id.items())
                ),
                "parent_output_rows": int(parent.n_out),
                "conditional_output_rows": conditional_output_count,
                "error_generators": conditional_error_count,
                "applied_guard_sets": expected_guard_sets,
                "rival_to_output_rows": {
                    int(rival): tuple(int(row) for row in rows)
                    for rival, rows in sorted(rival_to_rows.items())
                },
                "proof_authority": True,
            }
            expected_applied_hash = (
                _hz_conditional_applied_content_sha256(
                    expected_applied_payload
                )
            )
            if (
                not isinstance(applied, Mapping)
                or set(applied)
                != {
                    *expected_applied_payload.keys(),
                    "live_content_sha256",
                }
                or applied.get("live_content_sha256")
                != expected_applied_hash
            ):
                return False
            live_applied_payload = {
                key: applied[key]
                for key in applied
                if key != "live_content_sha256"
            }
            if (
                _hz_conditional_applied_content_sha256(
                    live_applied_payload
                )
                != expected_applied_hash
            ):
                return False
        elif applied is not None:
            return False

        raw_layers = getattr(
            parent,
            "_solver_continuous_column_layer_ids",
            None,
        )
        child_layers = getattr(
            child,
            "_solver_continuous_column_layer_ids",
            None,
        )
        if raw_layers is not None:
            raw_layers = np.asarray(
                raw_layers,
                dtype=np.int64,
            ).reshape(-1)
            if (
                raw_layers.size != parent.n_cont
                or child_layers is None
                or not np.array_equal(
                    np.asarray(child_layers, dtype=np.int64).reshape(-1),
                    np.concatenate(
                        [
                            raw_layers,
                            np.full(
                                substitution_error_count
                                + conditional_error_count,
                                -2,
                                dtype=np.int64,
                            ),
                        ]
                    ),
                )
            ):
                return False
        _hz_phase_deadline_check(
            deadline, stage="phase_child_audit_complete"
        )
        return True
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        IndexError,
    ):
        return False


def hz_enumerate_sparse_binary_phase_cover(
    hz: SparseHZono,
    positions: Optional[Sequence[int]] = None,
    *,
    max_children: int = 16,
    deadline: Optional[float] = None,
) -> Tuple[Tuple[Tuple[Tuple[int, int], ...], SparseHZono], ...]:
    """Enumerate a deterministic sound outward binary-phase cover of ``hz``.

    The guard prevents accidental exponential exploration.  Callers should
    select one or two property-relevant ReLUs first, then use this routine.
    """

    if not isinstance(hz, SparseHZono):
        raise TypeError("binary phase cover requires SparseHZono")
    _hz_phase_deadline_check(deadline, stage="phase_cover_entry")
    if isinstance(max_children, (bool, np.bool_)) or not isinstance(
        max_children, (int, np.integer)
    ):
        raise TypeError("max_children must be an integer")
    max_children = int(max_children)
    if max_children <= 0:
        raise ValueError("max_children must be positive")

    if positions is None:
        selected = tuple(range(hz.n_bin))
    else:
        selected_list = []
        seen = set()
        for raw_position in positions:
            if isinstance(raw_position, (bool, np.bool_)) or not isinstance(
                raw_position, (int, np.integer)
            ):
                raise TypeError("binary phase position must be an integer")
            position = int(raw_position)
            if not 0 <= position < hz.n_bin:
                raise ValueError(
                    f"binary phase position {position} is outside "
                    f"[0, {hz.n_bin})"
                )
            if position in seen:
                raise ValueError("binary phase positions must be unique")
            seen.add(position)
            selected_list.append(position)
        selected = tuple(sorted(selected_list))
    if not selected:
        raise ValueError("binary phase cover requires at least one position")

    child_count = 1 << len(selected)
    if child_count > max_children:
        raise ValueError(
            f"binary phase cover would create {child_count} children; "
            f"limit={max_children}"
        )
    children = []
    for bits in range(child_count):
        _hz_phase_deadline_check(
            deadline, stage=f"phase_cover_before_child_{bits}"
        )
        assignment = tuple(
            (
                int(position),
                1 if ((bits >> index) & 1) else -1,
            )
            for index, position in enumerate(selected)
        )
        children.append(
            (
                assignment,
                hz_fix_sparse_binary_assignment(
                    hz,
                    assignment,
                    deadline=deadline,
                ),
            )
        )
    return tuple(children)

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

    if ng > 0:
        mass = Gc.abs().sum(dim=0)
        if nc > 0:
            mass = mass + Ac.abs().sum(dim=0)
        keep = mass > tol
        if not bool(keep.all()):
            Gc = Gc[:, keep]
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
                fresh = hz_fresh_col_ids(n_fresh, device=device).tolist()
                fi = 0; ids = []
                for v in keep_id:
                    if v is None:
                        ids.append(fresh[fi]); fi += 1
                    else:
                        ids.append(v)
                col_ids = torch.tensor(ids, dtype=torch.long, device=device)

    if (nc > 0 and nc <= _PARALLEL_MAX
            and _fits_parallel_merge(nc, Ac.shape[1] + Ab.shape[1] + 1)):
        A_full = torch.cat([Ac, Ab, b], dim=1)
        rnorm = A_full.norm(dim=1)
        coeff_norm = (Ac.abs().sum(1) + Ab.abs().sum(1))
        keys, _ = _canonical_keys(A_full.T, rnorm, tol)
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

    return hz_inherit_known_nonempty(
        HZono(c=hz.c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b,
              eq_mask=eq_mask, col_ids=col_ids, bcol_ids=bcol_ids),
        hz,
        reason="remove_redundancy",
    )


_PARALLEL_MAX = 20000  # skip the (still O(ncols)) dedup passes above this size
_PARALLEL_CELL_MAX = 25_000_000  # avoid multi-GB canonicalization tensors


def _fits_parallel_merge(nrow: int, ncol: int) -> bool:
    return int(nrow) * int(ncol) <= _PARALLEL_CELL_MAX


def _canonical_keys(M, norms, tol):
    """Hashable direction keys for columns of M (sign-canonicalized unit vectors,
    rounded). Parallel/anti-parallel columns share a key. Returns (keys, sign)."""
    units = M / norms.clamp_min(tol)
    absu = units.abs()
    sig = absu > 1e-6
    first = torch.argmax(sig.to(torch.int8), dim=0)
    cols = torch.arange(units.shape[1], device=M.device)
    fv = units[first, cols]
    sign = torch.where(fv < 0, -1.0, 1.0)
    canon = (units * sign).round(decimals=6)
    rounded = canon.T.tolist()
    return [tuple(r) for r in rounded], sign


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


def _shared_constraint_prefix(Ac_x: torch.Tensor, Ac_y: torch.Tensor,
                              Ab_x: torch.Tensor, Ab_y: torch.Tensor,
                              b_x: torch.Tensor, b_y: torch.Tensor,
                              eq_x: "torch.Tensor | None",
                              eq_y: "torch.Tensor | None") -> int:
    m = min(int(Ac_x.shape[0]), int(Ac_y.shape[0]))
    if m == 0:
        return 0
    same = (Ac_x[:m] == Ac_y[:m]).all(dim=1)
    if Ab_x.shape[1]:
        same &= (Ab_x[:m] == Ab_y[:m]).all(dim=1)
    same &= (b_x[:m] == b_y[:m]).reshape(m, -1).all(dim=1)
    if eq_x is not None or eq_y is not None:
        ex = eq_x if eq_x is not None else torch.ones(int(Ac_x.shape[0]), dtype=torch.bool, device=Ac_x.device)
        ey = eq_y if eq_y is not None else torch.ones(int(Ac_y.shape[0]), dtype=torch.bool, device=Ac_y.device)
        same &= ex[:m].to(Ac_x.device) == ey[:m].to(Ac_x.device)
    return m if bool(same.all()) else int((~same).nonzero()[0, 0])


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

    Ac_x = _scatter_cols(hz_x.Ac, xc_map, ngm)
    Ac_y = _scatter_cols(hz_y.Ac.to(dtype=dtype, device=device), yc_map, ngm)
    Ab_x = _scatter_cols(hz_x.Ab, xb_map, nbm)
    Ab_y = _scatter_cols(hz_y.Ab.to(dtype=dtype, device=device), yb_map, nbm)

    b_x = hz_x.b.to(dtype=dtype, device=device)
    b_y = hz_y.b.to(dtype=dtype, device=device)
    k = _shared_constraint_prefix(Ac_x, Ac_y, Ab_x, Ab_y, b_x, b_y,
                                  hz_x.eq_mask, hz_y.eq_mask)
    new_Ac = torch.cat([Ac_x, Ac_y[k:]], dim=0)
    new_Ab = torch.cat([Ab_x, Ab_y[k:]], dim=0)
    new_b = torch.cat([b_x, b_y[k:]], dim=0)

    nc_x = int(hz_x.Ac.shape[0])
    nc_y = int(hz_y.Ac.shape[0])
    if hz_x.eq_mask is None and hz_y.eq_mask is None:
        new_eq_mask = None
    else:
        mx = (hz_x.eq_mask if hz_x.eq_mask is not None
              else torch.ones(nc_x, dtype=torch.bool, device=device))
        my = (hz_y.eq_mask if hz_y.eq_mask is not None
              else torch.ones(nc_y, dtype=torch.bool, device=device))
        new_eq_mask = torch.cat([mx.to(device), my.to(device)[k:]], dim=0)

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
        reason="concat_independent",
    )

try:
    import scipy.sparse as _sp
    from scipy.optimize import linprog as _linprog, milp as _milp
    from scipy.optimize import LinearConstraint as _LC, Bounds as _LPB
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

try:
    import highspy as _highspy
    _HAS_HIGHSPY = True
except Exception:  # pragma: no cover
    _HAS_HIGHSPY = False

try:
    from pyscipopt import Model as _SCIPModel, quicksum as _scip_quicksum
    _HAS_PYSCIPOPT = True
except Exception:  # pragma: no cover
    _HAS_PYSCIPOPT = False



def _torch_csr(t):
    """Torch tensor -> scipy CSR without materializing hstack/vstack dense copies."""
    if not _HAS_SCIPY:
        return None
    shape = tuple(int(x) for x in t.shape)
    if t.numel() == 0:
        return _sp.csr_matrix(shape, dtype=np.float64)
    return _sp.csr_matrix(t.detach().cpu().numpy(), dtype=np.float64)


def hz_np_sparse(hz):
    """Dense output rows + sparse constraint rows for solver construction.

    The final spec only touches the output dimension, so ``Gc/Gb`` stay dense and
    small. The constraint block can be thousands by tens of thousands with very
    low structural density after exact ReLU; keep it CSR all the way into HiGHS.
    """
    cached = getattr(hz, "_solver_np_sparse_cache", None)
    if cached is not None:
        return cached
    if isinstance(hz, SparseHZono):
        out = hz.solver_tuple()
        setattr(hz, "_solver_np_sparse_cache", out)
        return out
    c = hz.c.detach().cpu().double().numpy().reshape(-1)
    Gc = hz.Gc.detach().cpu().double().numpy()
    Gb = hz.Gb.detach().cpu().double().numpy()
    (Ace, Abe, be), (Acl, Abl, bl) = hz_split_constraints(hz)
    out = (c, Gc, Gb,
           _torch_csr(Ace), _torch_csr(Abe),
           be.detach().cpu().double().numpy().reshape(-1),
           _torch_csr(Acl), _torch_csr(Abl),
           bl.detach().cpu().double().numpy().reshape(-1))
    setattr(hz, "_solver_np_sparse_cache", out)
    return out




def _mat_dot_gen(mat, gen) -> np.ndarray:
    """Return ``mat @ gen`` as a dense ndarray for solver objective rows."""

    mat = np.asarray(mat, dtype=np.float64)
    if _sp.issparse(gen):
        return np.asarray((_sp.csr_matrix(mat) @ gen).toarray(), dtype=np.float64)
    return np.asarray(mat @ gen, dtype=np.float64)


def _row_dot_gen(row, gen) -> np.ndarray:
    return _mat_dot_gen(np.asarray(row, dtype=np.float64).reshape(1, -1), gen).reshape(-1)


def hz_relax_np_sparse(hz):
    """Sparse LP-relaxation matrices for HybridZ LP prefilters.

    The production HybridZ verdict path below builds its own exact LP/MILP rows
    through ``_objbound_solve``.  Packaged HybridZ workers use this public helper
    for lightweight LP prefilters.
    """
    cached = getattr(hz, "_solver_relax_sparse_cache", None)
    if cached is not None:
        return cached
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = hz_np_sparse(hz)
    Aeq = _sp.hstack([Ace, Abe], format="csr") if Ace.shape[0] else None
    Aub = _sp.hstack([Acl, Abl], format="csr") if Acl.shape[0] else None
    out = (c, Gc, Gb, Aeq, be, Aub, bl)
    setattr(hz, "_solver_relax_sparse_cache", out)
    return out




def _csr_rowsum(A) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).reshape(-1)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _highs_process_threads() -> int:
    """One immutable HiGHS scheduler size for the lifetime of a worker.

    HiGHS 1.14 owns a process-global scheduler: initializing one model with
    one thread count and a later model with another makes ``run`` fail.  Use
    the per-solver gate cap everywhere, including the persistent LP.
    """

    raw = os.environ.get(
        "HZ_MILP_THREADS",
        os.environ.get("HZ_LP_PREFILTER_THREADS", "1"),
    )
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = 1
    return max(1, value)


def _parse_highs_value(raw: str):
    s = str(raw).strip()
    lo = s.lower()
    if lo in {"true", "yes", "on"}:
        return True
    if lo in {"false", "no", "off"}:
        return False
    try:
        if any(ch in s for ch in ".eE"):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _apply_highs_env_options(h):
    """Apply comma/semicolon separated HiGHS options from HZ_HIGHS_OPTIONS."""
    raw = os.environ.get("HZ_HIGHS_OPTIONS", "").strip()
    if not raw:
        return
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            h.setOptionValue(key, _parse_highs_value(val))
        except Exception:
            pass


def _project_singleton_continuous_rows(A, rl, ru, cost, lb, ub, integ_mask):
    """Exact existential projection of objective-free continuous singleton vars.

    If a continuous variable appears in exactly one equality row, appears in no
    other row, and has zero objective coefficient, it can be eliminated exactly:

        a*x + rest = rhs,  lb <= x <= ub

    becomes the range row

        rhs - max(a*[lb,ub]) <= rest <= rhs - min(a*[lb,ub]).

    This is solver presolve only. It keeps the feasible projection over the
    remaining variables identical and stores enough metadata to reconstruct an
    original-space witness if HiGHS finds one.
    """
    if not (_HAS_SCIPY and A.shape[0] and A.shape[1]):
        return A, rl, ru, cost, lb, ub, None

    A = _sp.csr_matrix(A)
    rl = np.asarray(rl, dtype=np.float64).copy()
    ru = np.asarray(ru, dtype=np.float64).copy()
    cost = np.asarray(cost, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    integ_mask = np.asarray(integ_mask, dtype=bool)

    eq_rows = np.isfinite(rl) & np.isfinite(ru) & (np.abs(rl - ru) <= 1e-10)
    if not np.any(eq_rows):
        return A, rl, ru, cost, lb, ub, None

    Acsc = A.tocsc()
    removable = []
    records_by_row = {}
    for j in range(A.shape[1]):
        if integ_mask[j] or abs(float(cost[j])) > 1e-12:
            continue
        start, end = Acsc.indptr[j], Acsc.indptr[j + 1]
        if end - start != 1:
            continue
        r = int(Acsc.indices[start])
        if not bool(eq_rows[r]):
            continue
        a = float(Acsc.data[start])
        if abs(a) <= 1e-14 or not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
            continue
        removable.append(j)
        records_by_row.setdefault(r, []).append((j, a, float(lb[j]), float(ub[j])))

    if not removable:
        return A, rl, ru, cost, lb, ub, None

    elim_min = np.zeros(A.shape[0], dtype=np.float64)
    elim_max = np.zeros(A.shape[0], dtype=np.float64)
    for r, vals in records_by_row.items():
        lo_sum = 0.0
        hi_sum = 0.0
        for _, a, lo, hi in vals:
            y0, y1 = a * lo, a * hi
            lo_sum += min(y0, y1)
            hi_sum += max(y0, y1)
        elim_min[r] = lo_sum
        elim_max[r] = hi_sum

    rhs = rl.copy()
    rl_new = rl.copy()
    ru_new = ru.copy()
    affected = np.fromiter(records_by_row.keys(), dtype=np.int64)
    rl_new[affected] = rhs[affected] - elim_max[affected]
    ru_new[affected] = rhs[affected] - elim_min[affected]

    keep = np.ones(A.shape[1], dtype=bool)
    keep[np.asarray(removable, dtype=np.int64)] = False
    keep_cols = np.nonzero(keep)[0]
    reduced = (
        A[:, keep_cols].tocsr(),
        rl_new,
        ru_new,
        cost[keep_cols],
        lb[keep_cols],
        ub[keep_cols],
        {
            "keep_cols": keep_cols,
            "original_ncols": int(A.shape[1]),
            "records_by_row": records_by_row,
            "rhs": rhs,
            "A_reduced": A[:, keep_cols].tocsr(),
        },
    )
    return reduced


def _substitute_eq_singleton_continuous_cols(A, rl, ru, cost, lb, ub, integ_mask):
    """Exact equality substitution for objective-free continuous singleton vars.

    For a continuous variable that appears in exactly one equality row,

        a*x + rest = rhs,

    substitute ``x = rhs/a - rest/a`` into every inequality row where ``x``
    appears, add the projected bounds for ``x``, and then remove both ``x`` and
    its defining equality. This is exact existential projection, not a
    relaxation. The transform is conservative about sparsity and skips columns
    whose substitution would touch too many inequality rows.
    """
    if not (_HAS_SCIPY and A.shape[0] and A.shape[1]):
        return A, rl, ru, cost, lb, ub, None

    A = _sp.csr_matrix(A, dtype=np.float64)
    rl = np.asarray(rl, dtype=np.float64).copy()
    ru = np.asarray(ru, dtype=np.float64).copy()
    cost = np.asarray(cost, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    integ_mask = np.asarray(integ_mask, dtype=bool)

    eq_rows = np.isfinite(rl) & np.isfinite(ru) & (np.abs(rl - ru) <= 1e-10)
    if not np.any(eq_rows):
        return A, rl, ru, cost, lb, ub, None

    try:
        max_ineq = int(os.environ.get("HZ_MILP_EQ_SUBST_MAX_INEQ", "2"))
    except ValueError:
        max_ineq = 2
    A_csc = A.tocsc()
    used_rows = set()
    pivots = []
    for j in range(A.shape[1]):
        if integ_mask[j] or abs(float(cost[j])) > 1e-12:
            continue
        if not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
            continue
        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        rows = A_csc.indices[start:end]
        data = A_csc.data[start:end]
        eq_pos = [k for k, r in enumerate(rows) if bool(eq_rows[int(r)])]
        if len(eq_pos) != 1:
            continue
        ineq_nnz = (end - start) - 1
        if max_ineq >= 0 and ineq_nnz > max_ineq:
            continue
        pos = eq_pos[0]
        r = int(rows[pos])
        if r in used_rows:
            continue
        a = float(data[pos])
        if abs(a) <= 1e-14:
            continue
        used_rows.add(r)
        pivots.append((int(j), r, a))

    if not pivots:
        return A, rl, ru, cost, lb, ub, None

    pivot_cols = np.asarray([p[0] for p in pivots], dtype=np.int64)
    pivot_rows = np.asarray([p[1] for p in pivots], dtype=np.int64)
    pivot_col_set = set(int(x) for x in pivot_cols)
    pivot_row_set = set(int(x) for x in pivot_rows)

    corr_rr, corr_cc, corr_dd = [], [], []
    bound_rr, bound_cc, bound_dd = [], [], []
    bound_rl, bound_ru = [], []
    records = []
    skipped = 0

    for out_i, (j, r, a) in enumerate(pivots):
        row = A.getrow(r).tocoo()
        mask = row.col != j
        cols = row.col[mask].astype(np.int64, copy=False)
        data = row.data[mask].astype(np.float64, copy=False)
        if any(int(c) in pivot_col_set for c in cols):
            skipped += 1
            pivot_col_set.discard(int(j))
            pivot_row_set.discard(int(r))
            continue

        rhs_j = float(rl[r])
        const = rhs_j / a
        coeff = -data / a
        records.append({
            "j": int(j),
            "r": int(r),
            "a": float(a),
            "rhs": rhs_j,
            "cols": cols.copy(),
            "data": data.copy(),
            "lb": float(lb[j]),
            "ub": float(ub[j]),
        })

        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        rows_i = A_csc.indices[start:end].astype(np.int64, copy=False)
        vals_i = A_csc.data[start:end].astype(np.float64, copy=False)
        for row_i, val_i in zip(rows_i, vals_i):
            row_i = int(row_i)
            if row_i == r:
                continue
            if cols.size:
                corr_rr.append(np.full(cols.size, row_i, dtype=np.int32))
                corr_cc.append(cols.astype(np.int32, copy=False))
                corr_dd.append((float(val_i) * coeff).astype(np.float64, copy=False))
            shift = float(val_i) * const
            if np.isfinite(rl[row_i]):
                rl[row_i] -= shift
            if np.isfinite(ru[row_i]):
                ru[row_i] -= shift

        if cols.size:
            br = 2 * out_i
            bound_rr.append(np.full(cols.size, br, dtype=np.int32))
            bound_cc.append(cols.astype(np.int32, copy=False))
            bound_dd.append(coeff.astype(np.float64, copy=False))
            bound_rl.append(-np.inf)
            bound_ru.append(float(ub[j]) - const)

            bound_rr.append(np.full(cols.size, br + 1, dtype=np.int32))
            bound_cc.append(cols.astype(np.int32, copy=False))
            bound_dd.append((-coeff).astype(np.float64, copy=False))
            bound_rl.append(-np.inf)
            bound_ru.append(-float(lb[j]) + const)
        else:
            bound_rl.extend([-np.inf, -np.inf])
            bound_ru.extend([float(ub[j]) - const, -float(lb[j]) + const])

    if skipped:
        keep_records = {int(rec["j"]) for rec in records}
        pivot_cols = np.asarray([p[0] for p in pivots if int(p[0]) in keep_records], dtype=np.int64)
        pivot_rows = np.asarray([p[1] for p in pivots if int(p[0]) in keep_records], dtype=np.int64)

    if pivot_cols.size == 0:
        return A, rl, ru, cost, lb, ub, None

    if corr_rr:
        corr = _sp.coo_matrix(
            (np.concatenate(corr_dd), (np.concatenate(corr_rr), np.concatenate(corr_cc))),
            shape=A.shape,
        ).tocsr()
        A = (A + corr).tocsr()
        A.eliminate_zeros()

    if bound_ru:
        n_bound = len(bound_ru)
        if bound_rr:
            bound_A = _sp.coo_matrix(
                (np.concatenate(bound_dd), (np.concatenate(bound_rr), np.concatenate(bound_cc))),
                shape=(n_bound, A.shape[1]),
            ).tocsr()
        else:
            bound_A = _sp.csr_matrix((n_bound, A.shape[1]), dtype=np.float64)
        A = _sp.vstack([A, bound_A], format="csr")
        rl = np.concatenate([rl, np.asarray(bound_rl, dtype=np.float64)])
        ru = np.concatenate([ru, np.asarray(bound_ru, dtype=np.float64)])

    keep_cols_mask = np.ones(A.shape[1], dtype=bool)
    keep_cols_mask[pivot_cols] = False
    keep_rows_mask = np.ones(A.shape[0], dtype=bool)
    keep_rows_mask[pivot_rows] = False
    keep_cols = np.nonzero(keep_cols_mask)[0]
    keep_rows = np.nonzero(keep_rows_mask)[0]
    A_reduced = A[keep_rows, :][:, keep_cols].tocsr()
    meta = {
        "kind": "eq_subst",
        "keep_cols": keep_cols,
        "original_ncols": int(A.shape[1]),
        "records": records,
    }
    return (
        A_reduced,
        rl[keep_rows],
        ru[keep_rows],
        cost[keep_cols],
        lb[keep_cols],
        ub[keep_cols],
        meta,
    )


def _chain_elim_meta(*metas):
    metas = [m for m in metas if m is not None]
    if not metas:
        return None
    if len(metas) == 1:
        return metas[0]
    return {"kind": "chain", "metas": metas}


def _expand_projected_solution(v, elim_meta):
    if not elim_meta:
        return np.asarray(v, dtype=np.float64)
    if elim_meta.get("kind") == "chain":
        out = np.asarray(v, dtype=np.float64)
        for meta in reversed(elim_meta["metas"]):
            out = _expand_projected_solution(out, meta)
        return out
    if elim_meta.get("kind") == "eq_subst":
        keep_cols = np.asarray(elim_meta["keep_cols"], dtype=np.int64)
        full = np.zeros(int(elim_meta["original_ncols"]), dtype=np.float64)
        full[keep_cols] = np.asarray(v, dtype=np.float64)
        for rec in elim_meta["records"]:
            cols = np.asarray(rec["cols"], dtype=np.int64)
            data = np.asarray(rec["data"], dtype=np.float64)
            rest = float(data @ full[cols]) if cols.size else 0.0
            val = (float(rec["rhs"]) - rest) / float(rec["a"])
            full[int(rec["j"])] = float(np.clip(val, float(rec["lb"]), float(rec["ub"])))
        return full
    keep_cols = np.asarray(elim_meta["keep_cols"], dtype=np.int64)
    full = np.zeros(int(elim_meta["original_ncols"]), dtype=np.float64)
    full[keep_cols] = np.asarray(v, dtype=np.float64)
    Ared = elim_meta["A_reduced"]
    rhs = np.asarray(elim_meta["rhs"], dtype=np.float64)

    for r, vals in elim_meta["records_by_row"].items():
        start, end = Ared.indptr[r], Ared.indptr[r + 1]
        rest = float(Ared.data[start:end] @ full[keep_cols[Ared.indices[start:end]]])
        target = float(rhs[r] - rest)

        lows, highs = [], []
        for _, a, lo, hi in vals:
            y0, y1 = a * lo, a * hi
            lows.append(min(y0, y1))
            highs.append(max(y0, y1))
        y = np.asarray(lows, dtype=np.float64)
        target = min(max(target, float(y.sum())), float(np.asarray(highs, dtype=np.float64).sum()))
        surplus = target - float(y.sum())
        for k in range(len(vals)):
            room = highs[k] - lows[k]
            if surplus <= 0.0:
                break
            step = min(room, surplus)
            y[k] += step
            surplus -= step
        for k, (j, a, lo, hi) in enumerate(vals):
            full[int(j)] = float(np.clip(y[k] / a, lo, hi))
    return full


def _project_solution_to_reduced(v, elim_meta):
    out = np.asarray(v, dtype=np.float64)
    if not elim_meta:
        return out
    if elim_meta.get("kind") == "chain":
        for meta in elim_meta["metas"]:
            out = _project_solution_to_reduced(out, meta)
        return out
    return out[np.asarray(elim_meta["keep_cols"], dtype=np.int64)]


def _scale_milp_rows(A, rl, ru):
    """Positive row scaling for solver conditioning; exact feasible set unchanged."""
    A = _sp.csr_matrix(A, dtype=np.float64)
    A.sum_duplicates()
    A.sort_indices()
    if A.shape[0] == 0:
        return A, np.asarray(rl, dtype=np.float64), np.asarray(ru, dtype=np.float64), None
    row_abs = np.asarray(np.abs(A).max(axis=1).toarray()).reshape(-1)
    scale = np.ones(A.shape[0], dtype=np.float64)
    nz = row_abs > 0.0
    scale[nz] = 1.0 / row_abs[nz]
    if np.allclose(scale, 1.0):
        return A, np.asarray(rl, dtype=np.float64), np.asarray(ru, dtype=np.float64), scale
    D = _sp.diags(scale, offsets=0, format="csr")
    scaled = _sp.csr_matrix(D @ A, dtype=np.float64)
    scaled.sum_duplicates()
    scaled.sort_indices()
    return (
        scaled,
        np.asarray(rl, dtype=np.float64) * scale,
        np.asarray(ru, dtype=np.float64) * scale,
        scale,
    )


def _highs_candidate_csr(
    A,
    *,
    small_matrix_value: float = 1e-12,
):
    """Build an auditable HiGHS-only CSR without granting it proof authority.

    HiGHS silently removes coefficients whose magnitude is at most
    ``small_matrix_value`` and reports only ``kWarning`` from ``addRows``.
    Perform that perturbation explicitly on a solver copy so the accepted
    status and the exact number/mass of omitted entries are checkable.  The
    caller must retain the original matrix for witness or certificate
    validation; this helper does *not* claim that arbitrary signed deletion
    is a relaxation.
    """

    threshold = float(small_matrix_value)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("HiGHS small-matrix threshold must be finite and positive")
    candidate = _sp.csr_matrix(A, dtype=np.float64).copy()
    if not candidate.has_canonical_format or not candidate.has_sorted_indices:
        raise ValueError("HiGHS candidate CSR must be canonical and sorted")
    if candidate.nnz and not np.all(np.isfinite(candidate.data)):
        raise ValueError("HiGHS candidate CSR contains non-finite coefficients")

    original_nnz = int(candidate.nnz)
    dropped = np.abs(candidate.data) <= threshold
    dropped_values = candidate.data[dropped].copy()
    if dropped_values.size:
        candidate.data[dropped] = 0.0
        candidate.eliminate_zeros()
        candidate.sort_indices()
    if candidate.nnz and (
        not np.all(np.isfinite(candidate.data))
        or np.any(np.abs(candidate.data) <= threshold)
    ):
        raise ValueError("HiGHS candidate CSR tiny-value filtering failed")
    stats = {
        "input_nnz": original_nnz,
        "loaded_nnz": int(candidate.nnz),
        "dropped_nnz": int(dropped_values.size),
        "dropped_abs_mass": float(
            np.sum(np.abs(dropped_values), dtype=np.longdouble)
        ),
        "dropped_abs_max": (
            float(np.max(np.abs(dropped_values)))
            if dropped_values.size
            else 0.0
        ),
        "small_matrix_value": threshold,
    }
    if stats["loaded_nnz"] + stats["dropped_nnz"] != original_nnz:
        raise ValueError("HiGHS candidate CSR nnz accounting mismatch")
    return candidate, stats


def _scale_milp_objective(cost, obj_thr):
    """Positive objective scaling; exact cutoff sign unchanged."""
    cost = np.asarray(cost, dtype=np.float64)
    denom = max(float(np.max(np.abs(cost))) if cost.size else 0.0,
                abs(float(obj_thr)), 1.0)
    if not np.isfinite(denom) or denom <= 0.0:
        return cost, float(obj_thr), 1.0
    scale = 1.0 / denom
    return cost * scale, float(obj_thr) * scale, scale


def sparse_row_bound_infeasible(
    A: "_sp.csr_matrix",
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    tol: float = 1e-9,
) -> Tuple[bool, Dict[str, object]]:
    """Cheap exact row-range infeasibility check over variable bounds."""

    if A.shape[0] == 0 or A.shape[1] == 0:
        return False, {"rows": 0}
    A = A.tocsr()
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    rl = np.asarray(rl, dtype=np.float64)
    ru = np.asarray(ru, dtype=np.float64)
    pos = A.maximum(0.0).tocsr()
    neg = A.minimum(0.0).tocsr()
    row_min = np.asarray(pos @ lb + neg @ ub).reshape(-1)
    row_max = np.asarray(pos @ ub + neg @ lb).reshape(-1)
    scale = np.maximum(1.0, np.maximum(np.abs(row_min), np.abs(row_max)))
    hi_bad = np.isfinite(ru) & (row_min > ru + tol * scale)
    lo_bad = np.isfinite(rl) & (row_max < rl - tol * scale)
    bad = np.flatnonzero(hi_bad | lo_bad)
    if bad.size == 0:
        return False, {"rows": int(A.shape[0])}
    r0 = int(bad[0])
    return True, {
        "rows": int(A.shape[0]),
        "bad_row": r0,
        "row_min": float(row_min[r0]),
        "row_max": float(row_max[r0]),
        "rl": float(rl[r0]) if np.isfinite(rl[r0]) else None,
        "ru": float(ru[r0]) if np.isfinite(ru[r0]) else None,
        "bad_count": int(bad.size),
    }


def sparse_fbbt_tighten_bounds(
    A: "_sp.csr_matrix",
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    integer_mask: Optional[np.ndarray] = None,
    max_passes: int = 3,
    tol: float = 1e-9,
) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, object]]:
    """Feasibility-based bound tightening for sparse linear rows.

    This is a sound MILP presolve over exact HZ constraints.  It intersects the
    current variable boxes with bounds implied by ``rl <= A x <= ru`` and can
    prove EMPTY; it never proves a witness or replaces the exact MILP solve.
    """

    if A.shape[0] == 0 or A.shape[1] == 0 or max_passes <= 0:
        return False, lb, ub, {"passes": 0, "tightened": 0, "fixed_int": 0}

    A = A.tocsr(copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    rl = np.asarray(rl, dtype=np.float64)
    ru = np.asarray(ru, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
    if integer_mask is None:
        integer_mask = np.zeros(A.shape[1], dtype=bool)
    else:
        integer_mask = np.asarray(integer_mask, dtype=bool)
        if integer_mask.size != A.shape[1]:
            raise ValueError(f"integer_mask size mismatch: {integer_mask.size} vs {A.shape[1]}")

    pos = A.maximum(0.0).tocsr()
    neg = A.minimum(0.0).tocsr()
    total_tightened = 0
    total_fixed_int = 0
    max_width_delta = 0.0
    passes_done = 0
    bad_info: Optional[Dict[str, object]] = None

    for pass_i in range(int(max_passes)):
        passes_done = pass_i + 1
        prev_lb = lb.copy()
        prev_ub = ub.copy()
        row_min = np.asarray(pos @ lb + neg @ ub).reshape(-1)
        row_max = np.asarray(pos @ ub + neg @ lb).reshape(-1)
        scale = np.maximum(1.0, np.maximum(np.abs(row_min), np.abs(row_max)))
        hi_bad = np.isfinite(ru) & (row_min > ru + tol * scale)
        lo_bad = np.isfinite(rl) & (row_max < rl - tol * scale)
        bad = np.flatnonzero(hi_bad | lo_bad)
        if bad.size:
            r0 = int(bad[0])
            bad_info = {
                "bad_row": r0,
                "row_min": float(row_min[r0]),
                "row_max": float(row_max[r0]),
                "rl": float(rl[r0]) if np.isfinite(rl[r0]) else None,
                "ru": float(ru[r0]) if np.isfinite(ru[r0]) else None,
                "bad_count": int(bad.size),
            }
            return True, lb, ub, {
                "passes": passes_done,
                "tightened": int(total_tightened),
                "fixed_int": int(total_fixed_int),
                "max_width_delta": float(max_width_delta),
                "infeasible": bad_info,
            }

        for r in range(A.shape[0]):
            start, end = A.indptr[r], A.indptr[r + 1]
            if start == end:
                continue
            cols = A.indices[start:end]
            vals = A.data[start:end]
            nz = np.abs(vals) > tol
            if not np.any(nz):
                continue
            cols = cols[nz]
            vals = vals[nz]
            col_lb = lb[cols]
            col_ub = ub[cols]
            positive = vals > 0.0

            if np.isfinite(ru[r]) and ru[r] < 1e20:
                min_contrib = np.where(positive, vals * col_lb, vals * col_ub)
                rest_min = row_min[r] - min_contrib
                cand = (ru[r] - rest_min) / vals
                finite = np.isfinite(cand)
                m = positive & finite
                if np.any(m):
                    idx = cols[m]
                    ub[idx] = np.minimum(ub[idx], cand[m])
                m = (~positive) & finite
                if np.any(m):
                    idx = cols[m]
                    lb[idx] = np.maximum(lb[idx], cand[m])

            if np.isfinite(rl[r]) and rl[r] > -1e20:
                max_contrib = np.where(positive, vals * col_ub, vals * col_lb)
                rest_max = row_max[r] - max_contrib
                cand = (rl[r] - rest_max) / vals
                finite = np.isfinite(cand)
                m = positive & finite
                if np.any(m):
                    idx = cols[m]
                    lb[idx] = np.maximum(lb[idx], cand[m])
                m = (~positive) & finite
                if np.any(m):
                    idx = cols[m]
                    ub[idx] = np.minimum(ub[idx], cand[m])

        if np.any(integer_mask):
            snap_hi = integer_mask & (lb > tol) & (ub >= 1.0 - tol)
            snap_lo = integer_mask & (ub < 1.0 - tol) & (lb <= tol)
            fixed_now = int(np.count_nonzero(snap_hi | snap_lo))
            if fixed_now:
                total_fixed_int += fixed_now
                lb[snap_hi] = 1.0
                ub[snap_hi] = 1.0
                lb[snap_lo] = 0.0
                ub[snap_lo] = 0.0

        bad_cols = np.flatnonzero(lb > ub + 10.0 * tol)
        if bad_cols.size:
            j0 = int(bad_cols[0])
            return True, lb, ub, {
                "passes": passes_done,
                "tightened": int(total_tightened),
                "fixed_int": int(total_fixed_int),
                "max_width_delta": float(max_width_delta),
                "infeasible": {
                    "bad_col": j0,
                    "lb": float(lb[j0]),
                    "ub": float(ub[j0]),
                    "bad_count": int(bad_cols.size),
                },
            }

        tightened = (lb > prev_lb + tol) | (ub < prev_ub - tol)
        tightened_count = int(np.count_nonzero(tightened))
        if tightened_count:
            total_tightened += tightened_count
            before_w = prev_ub - prev_lb
            after_w = ub - lb
            max_width_delta = max(max_width_delta, float(np.max(before_w[tightened] - after_w[tightened])))
        else:
            break

    bad_cols = np.flatnonzero(lb > ub + 10.0 * tol)
    if bad_cols.size:
        j0 = int(bad_cols[0])
        bad_info = {"bad_col": j0, "lb": float(lb[j0]), "ub": float(ub[j0]), "bad_count": int(bad_cols.size)}
        return True, lb, ub, {
            "passes": passes_done,
            "tightened": int(total_tightened),
            "fixed_int": int(total_fixed_int),
            "max_width_delta": float(max_width_delta),
            "infeasible": bad_info,
        }

    return False, lb, ub, {
        "passes": passes_done,
        "tightened": int(total_tightened),
        "fixed_int": int(total_fixed_int),
        "max_width_delta": float(max_width_delta),
    }


def sparse_solver_start_from_xi(
    base_xi: Optional[np.ndarray],
    ncols: int,
    n_cont: int,
    n_bin: int,
) -> np.ndarray:
    """Convert an HZ ``xi`` witness into solver variable coordinates."""

    full = np.zeros(int(ncols), dtype=np.float64)
    if base_xi is None:
        return full
    raw = np.asarray(base_xi, dtype=np.float64).reshape(-1)
    ncopy = min(raw.size, n_cont + n_bin, full.size)
    if ncopy <= 0:
        return full
    n_cont_copy = min(n_cont, ncopy)
    if n_cont_copy:
        full[:n_cont_copy] = np.clip(raw[:n_cont_copy], -1.0, 1.0)
    if ncopy > n_cont:
        b_end = min(n_cont + n_bin, ncopy)
        full[n_cont:b_end] = (np.clip(raw[n_cont:b_end], -1.0, 1.0) + 1.0) / 2.0
    return full


def sparse_highs_relaxation_empty_precheck(
    highspy_module,
    A: "_sp.csr_matrix",
    rl: np.ndarray,
    ru: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    cost: np.ndarray,
    cutoff: float,
    const_z: float,
    time_limit: float,
    cutoff_as_row: bool,
    multirow_feas: bool,
    highs_threads: int = 0,
    highs_parallel: str = "",
    highs_options: Optional[Dict[str, object]] = None,
) -> Tuple[str, Optional[float], Dict[str, object]]:
    """Continuous relaxation precheck for EMPTY only.

    If a relaxation of the exact HZ MILP is infeasible, then the integer HZ
    problem is infeasible.  For objective formulations, an optimal relaxation
    lower bound above the cutoff also proves EMPTY.  Feasible/timeout statuses
    are not used as ADV evidence.
    """

    ts = time.time()
    h = highspy_module.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("presolve", "on")
    if highs_threads > 0:
        h.setOptionValue("threads", int(highs_threads))
    if highs_parallel:
        h.setOptionValue("parallel", highs_parallel)
    if highs_options:
        for key, value in highs_options.items():
            h.setOptionValue(str(key), value)
    ncols = int(cost.size)
    h.addCols(
        ncols,
        np.asarray(cost, dtype=np.float64),
        np.asarray(lb, dtype=np.float64),
        np.asarray(ub, dtype=np.float64),
        0,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=float),
    )
    A = A.tocsr()
    if A.shape[0]:
        h.addRows(
            A.shape[0],
            np.asarray(rl, dtype=np.float64),
            np.asarray(ru, dtype=np.float64),
            A.nnz,
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(float),
        )
    h.run()
    st = h.getModelStatus()
    status = h.modelStatusToString(st)
    info = h.getInfo()

    def stat_float(name: str) -> Optional[float]:
        val = getattr(info, name, None)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    obj_value = stat_float("objective_function_value")
    margin = None if obj_value is None else const_z + float(obj_value)
    stats = {
        "status": status,
        "obj": obj_value,
        "margin": margin,
        "dual_bound": stat_float("mip_dual_bound"),
        "margin_dual_bound": None,
        "sec": round(time.time() - ts, 3),
        "nodes": None,
    }
    MS = highspy_module.HighsModelStatus
    if st == MS.kInfeasible:
        return "EMPTY:relax_infeasible", None, stats
    if st == MS.kOptimal and (not cutoff_as_row or multirow_feas):
        if margin is not None and margin > float(cutoff) + 1e-7:
            return "EMPTY:relax_bound", margin, stats
    return status, margin, stats


def sparse_milp_cutoff_highs(
    hz: SparseHZono,
    C: np.ndarray,
    t: np.ndarray,
    time_limit: float,
    cutoff: float = 0.0,
    elim_singletons: bool = False,
    highs_threads: int = 0,
    highs_parallel: str = "",
    highs_heuristic_effort: Optional[float] = None,
    cutoff_as_row: bool = False,
    highs_options: Optional[Dict[str, object]] = None,
    mip_start_xi: Optional[np.ndarray] = None,
    mip_start_binary_only: bool = False,
    connected_presolve: bool = False,
    base_xi: Optional[np.ndarray] = None,
    elim_eq_subst: bool = False,
    fbbt_passes: int = 0,
    relax_precheck_timeout: float = 0.0,
) -> Tuple[str, Optional[float], Optional[np.ndarray]]:
    try:
        import highspy
    except Exception as exc:
        return f"no_highspy:{exc}", None, None

    Cmat = C.reshape(C.shape[0], -1)
    tvec = t.reshape(-1)
    if Cmat.shape[0] != tvec.size:
        raise ValueError(f"C/t row mismatch: {Cmat.shape} vs {tvec.shape}")
    multirow_feas = Cmat.shape[0] > 1
    base_ncols = hz.n_cont + hz.n_bin
    extra_epigraph_cols = 1 if multirow_feas else 0
    if multirow_feas:
        cost = np.zeros(base_ncols + 1, dtype=np.float64)
        cost[-1] = 1.0
        const_z = 0.0
        obj_thr = float(cutoff)
    else:
        c_row = Cmat[0]
        obj_c = np.asarray(c_row @ hz.Gc).reshape(-1)
        obj_b = np.asarray(c_row @ hz.Gb).reshape(-1) if hz.n_bin else np.zeros(0)
        const = float(c_row @ hz.c - tvec[0])
        cost = np.concatenate([obj_c, 2.0 * obj_b])
        const_z = const - float(obj_b.sum())
        obj_thr = float(cutoff - const_z)

    if hz.n_eq:
        A = sp.hstack([hz.Ac, 2.0 * hz.Ab], format="csr")
        rhs = hz.b + np.asarray(hz.Ab.sum(axis=1)).reshape(-1)
    else:
        A = sp.csr_matrix((0, hz.n_cont + hz.n_bin), dtype=np.float64)
        rhs = np.zeros(0, dtype=np.float64)
    if extra_epigraph_cols:
        A = sp.hstack([A, sp.csr_matrix((A.shape[0], extra_epigraph_cols))], format="csr")
    if hz.n_ub:
        Ale = sp.hstack([hz.Auc, 2.0 * hz.Aub], format="csr")
        ble = hz.ub + np.asarray(hz.Aub.sum(axis=1)).reshape(-1)
    else:
        Ale = sp.csr_matrix((0, hz.n_cont + hz.n_bin), dtype=np.float64)
        ble = np.zeros(0, dtype=np.float64)
    if extra_epigraph_cols:
        Ale = sp.hstack([Ale, sp.csr_matrix((Ale.shape[0], extra_epigraph_cols))], format="csr")
    if multirow_feas:
        Csp = sp.csr_matrix(Cmat)
        unsafe_Ac = Csp @ hz.Gc
        unsafe_Ab = Csp @ hz.Gb if hz.n_bin else sp.csr_matrix((Cmat.shape[0], 0), dtype=np.float64)
        unsafe_A = sp.hstack(
            [
                unsafe_Ac,
                2.0 * unsafe_Ab,
                -sp.csr_matrix(np.ones((Cmat.shape[0], 1))),
            ],
            format="csr",
        )
        unsafe_b = tvec - Cmat @ hz.c + np.asarray(unsafe_Ab.sum(axis=1)).reshape(-1)
        Ale = sp.vstack([Ale, unsafe_A], format="csr")
        ble = np.concatenate([ble, unsafe_b.astype(np.float64)])

    lb = np.concatenate([-np.ones(hz.n_cont), np.zeros(hz.n_bin)])
    ub = np.ones(base_ncols)
    if extra_epigraph_cols:
        lb = np.concatenate([lb, np.array([-1e12], dtype=np.float64)])
        ub = np.concatenate([ub, np.array([1e12], dtype=np.float64)])

    mip_start_values: Optional[np.ndarray] = None
    mip_start_indices: Optional[np.ndarray] = None
    mip_start_status = "off"
    if mip_start_xi is not None:
        raw_start = np.asarray(mip_start_xi, dtype=np.float64).reshape(-1)
        if raw_start.size < base_ncols:
            mip_start_status = f"skipped:short_xi:{raw_start.size}<{base_ncols}"
        elif mip_start_binary_only and not hz.n_bin:
            mip_start_status = "skipped:no_binary"
        else:
            start = np.zeros(cost.size, dtype=np.float64)
            start[:hz.n_cont] = np.clip(raw_start[:hz.n_cont], -1.0, 1.0)
            if hz.n_bin:
                relaxed = np.clip(raw_start[hz.n_cont:base_ncols], -1.0, 1.0)
                start[hz.n_cont:base_ncols] = (relaxed >= 0.0).astype(np.float64)
            if extra_epigraph_cols:
                xi_c = start[:hz.n_cont]
                if hz.n_bin:
                    xi_b = 2.0 * start[hz.n_cont:base_ncols] - 1.0
                    y = hz.c + np.asarray(hz.Gc @ xi_c).reshape(-1) + np.asarray(hz.Gb @ xi_b).reshape(-1)
                else:
                    y = hz.c + np.asarray(hz.Gc @ xi_c).reshape(-1)
                start[-1] = float(np.max(Cmat @ y - tvec))
            start = np.clip(start, lb, ub)
            if np.all(np.isfinite(start)):
                mip_start_values = start
                if mip_start_binary_only:
                    mip_start_indices = np.arange(hz.n_cont, hz.n_cont + hz.n_bin, dtype=np.int64)
                mip_start_status = "prepared"
            else:
                mip_start_status = "skipped:nonfinite"

    rl = rhs.astype(np.float64).copy()
    ru = rhs.astype(np.float64).copy()
    keep_cols = None
    original_ncols = int(cost.size)
    original_margin_cost = cost.copy()
    original_const_z = float(const_z)
    reconstruct_base = sparse_solver_start_from_xi(base_xi, original_ncols, hz.n_cont, hz.n_bin)
    solve_obj_offset = 0.0
    elim_count = 0
    eq_subst_count = 0
    fixed_subst_stats = None
    if elim_eq_subst and A.shape[0] and A.shape[1]:
        Aeq = A.tocsr()
        Aeq_csc = Aeq.tocsc()
        Ale_csr = Ale.tocsr()
        Ale_csc = Ale_csr.tocsc() if Ale_csr.shape[0] else sp.csc_matrix((0, Aeq.shape[1]))
        used_rows: set[int] = set()
        pivots: List[Tuple[int, int, float]] = []
        for j in range(hz.n_cont):
            if abs(float(cost[j])) > 1e-12:
                continue
            es, ee = Aeq_csc.indptr[j], Aeq_csc.indptr[j + 1]
            if ee - es != 1:
                continue
            r = int(Aeq_csc.indices[es])
            if r in used_rows:
                continue
            ls, le = Ale_csc.indptr[j], Ale_csc.indptr[j + 1]
            if le - ls > 2:
                continue
            a = float(Aeq_csc.data[es])
            if abs(a) < 1e-12:
                continue
            used_rows.add(r)
            pivots.append((j, r, a))
        if pivots:
            pivot_cols = np.asarray([p[0] for p in pivots], dtype=np.int64)
            pivot_rows = np.asarray([p[1] for p in pivots], dtype=np.int64)
            pivot_col_set = set(int(x) for x in pivot_cols)
            pivot_row_set = set(int(x) for x in pivot_rows)
            corr_rr: List[np.ndarray] = []
            corr_cc: List[np.ndarray] = []
            corr_dd: List[np.ndarray] = []
            bound_rr: List[np.ndarray] = []
            bound_cc: List[np.ndarray] = []
            bound_dd: List[np.ndarray] = []
            bound_rhs: List[float] = []
            ble = ble.astype(np.float64, copy=True)
            skipped = 0

            for out_i, (j, r, a) in enumerate(pivots):
                row = Aeq.getrow(r).tocoo()
                mask = row.col != j
                cols = row.col[mask].astype(np.int64, copy=False)
                data = row.data[mask].astype(np.float64, copy=False)
                if any(int(c) in pivot_col_set for c in cols):
                    skipped += 1
                    pivot_col_set.discard(int(j))
                    pivot_row_set.discard(int(r))
                    continue
                rhs_j = float(rhs[r])
                const = rhs_j / a
                coeff = -data / a
                ls, le = Ale_csc.indptr[j], Ale_csc.indptr[j + 1]
                if le > ls and cols.size:
                    rows_i = Ale_csc.indices[ls:le].astype(np.int64, copy=False)
                    vals_i = Ale_csc.data[ls:le].astype(np.float64, copy=False)
                    for row_i, val_i in zip(rows_i, vals_i):
                        corr_rr.append(np.full(cols.size, int(row_i), dtype=np.int32))
                        corr_cc.append(cols.astype(np.int32, copy=False))
                        corr_dd.append((float(val_i) * coeff).astype(np.float64, copy=False))
                        ble[int(row_i)] -= float(val_i) * const
                if cols.size:
                    br = 2 * out_i
                    bound_rr.append(np.full(cols.size, br, dtype=np.int32))
                    bound_cc.append(cols.astype(np.int32, copy=False))
                    bound_dd.append(coeff.astype(np.float64, copy=False))
                    bound_rhs.append(float(ub[j]) - const)
                    bound_rr.append(np.full(cols.size, br + 1, dtype=np.int32))
                    bound_cc.append(cols.astype(np.int32, copy=False))
                    bound_dd.append((-coeff).astype(np.float64, copy=False))
                    bound_rhs.append(-float(lb[j]) + const)
                else:
                    bound_rhs.extend([float(ub[j]) - const, -float(lb[j]) + const])

            if skipped:
                pivots = [(j, r, a) for (j, r, a) in pivots if int(j) in pivot_col_set]
                pivot_cols = np.asarray([p[0] for p in pivots], dtype=np.int64)
                pivot_rows = np.asarray([p[1] for p in pivots], dtype=np.int64)
            if pivot_cols.size:
                if corr_rr:
                    corr = sp.coo_matrix(
                        (np.concatenate(corr_dd), (np.concatenate(corr_rr), np.concatenate(corr_cc))),
                        shape=Ale_csr.shape,
                    ).tocsr()
                    Ale_csr = (Ale_csr + corr).tocsr()
                    Ale_csr.eliminate_zeros()
                if bound_rhs:
                    n_bound = len(bound_rhs)
                    if bound_rr:
                        bound_A = sp.coo_matrix(
                            (np.concatenate(bound_dd), (np.concatenate(bound_rr), np.concatenate(bound_cc))),
                            shape=(n_bound, Aeq.shape[1]),
                        ).tocsr()
                    else:
                        bound_A = sp.csr_matrix((n_bound, Aeq.shape[1]), dtype=np.float64)
                    Ale_csr = sp.vstack([Ale_csr, bound_A], format="csr")
                    ble = np.concatenate([ble, np.asarray(bound_rhs, dtype=np.float64)])
                keep = np.ones(Aeq.shape[1], dtype=bool)
                keep[pivot_cols] = False
                keep_rows = np.ones(Aeq.shape[0], dtype=bool)
                keep_rows[pivot_rows] = False
                keep_idx = np.nonzero(keep)[0]
                A = Aeq[keep_rows, :][:, keep_idx].tocsr()
                rhs = rhs[keep_rows]
                Ale = Ale_csr[:, keep_idx].tocsr()
                cost = cost[keep_idx]
                lb = lb[keep_idx]
                ub = ub[keep_idx]
                keep_cols = keep_idx.astype(np.int64)
                rl = rhs.astype(np.float64).copy()
                ru = rhs.astype(np.float64).copy()
                if mip_start_values is not None:
                    mip_start_values = mip_start_values[keep_cols]
                    if mip_start_indices is not None:
                        reverse = {int(col): int(pos) for pos, col in enumerate(keep_cols)}
                        mip_start_indices = np.asarray(
                            [reverse[int(col)] for col in mip_start_indices if int(col) in reverse],
                            dtype=np.int64,
                        )
                        if mip_start_indices.size == 0:
                            mip_start_values = None
                            mip_start_status = "skipped:no_start_cols_after_eq_subst"
                eq_subst_count = int(pivot_cols.size)
                logger.debug(
                    "elim_eq_subst removed_cont=%s removed_eq=%s kept_cols=%s "
                    "eq_rows=%s ineq_rows=%s skipped=%s",
                    eq_subst_count,
                    eq_subst_count,
                    A.shape[1],
                    A.shape[0],
                    Ale.shape[0],
                    skipped,
                )
    if elim_singletons and A.shape[0] and A.shape[1]:
        Acsc = A.tocsc()
        Alecsc = Ale.tocsc() if Ale.shape[0] else None
        removable = []
        current_orig_cols = (
            np.arange(A.shape[1], dtype=np.int64)
            if keep_cols is None
            else np.asarray(keep_cols, dtype=np.int64)
        )
        current_cont_positions = np.nonzero(current_orig_cols < hz.n_cont)[0]
        for j in current_cont_positions:
            if abs(cost[int(j)]) > 1e-12:
                continue
            if Alecsc is not None and Alecsc.indptr[int(j) + 1] > Alecsc.indptr[int(j)]:
                continue
            start, end = Acsc.indptr[int(j)], Acsc.indptr[int(j) + 1]
            if end - start == 1:
                removable.append(int(j))
        if removable:
            elim_min = np.zeros(A.shape[0], dtype=np.float64)
            elim_max = np.zeros(A.shape[0], dtype=np.float64)
            for j in removable:
                start, end = Acsc.indptr[j], Acsc.indptr[j + 1]
                r = int(Acsc.indices[start])
                a = float(Acsc.data[start])
                vals = (a * lb[j], a * ub[j])
                elim_min[r] += min(vals)
                elim_max[r] += max(vals)
            rl = rhs - elim_max
            ru = rhs - elim_min
            keep = np.ones(A.shape[1], dtype=bool)
            keep[np.asarray(removable, dtype=np.int64)] = False
            local_keep_cols = np.nonzero(keep)[0]
            A = A[:, local_keep_cols].tocsr()
            Ale = Ale[:, local_keep_cols].tocsr()
            cost = cost[local_keep_cols]
            lb = lb[local_keep_cols]
            ub = ub[local_keep_cols]
            keep_cols = local_keep_cols if keep_cols is None else keep_cols[local_keep_cols]
            if mip_start_values is not None:
                mip_start_values = mip_start_values[local_keep_cols]
                if mip_start_indices is not None:
                    reverse = {int(col): int(pos) for pos, col in enumerate(local_keep_cols)}
                    mip_start_indices = np.asarray(
                        [reverse[int(col)] for col in mip_start_indices if int(col) in reverse],
                        dtype=np.int64,
                    )
                    if mip_start_indices.size == 0:
                        mip_start_values = None
                        mip_start_status = "skipped:no_start_cols_after_elim"
            elim_count = len(removable)
            logger.debug(
                "elim_singletons removed_cont=%s kept_cols=%s rows=%s",
                elim_count,
                A.shape[1],
                A.shape[0],
            )
    if Ale.shape[0]:
        A = sp.vstack([A, Ale], format="csr")
        rl = np.concatenate([rl, np.full(Ale.shape[0], -1e30, dtype=np.float64)])
        ru = np.concatenate([ru, ble.astype(np.float64)])
    margin_cost = cost.copy()
    if cutoff_as_row and not multirow_feas:
        A = sp.vstack([A, sp.csr_matrix(margin_cost.reshape(1, -1))], format="csr")
        rl = np.concatenate([rl, np.array([-1e30], dtype=np.float64)])
        ru = np.concatenate([ru, np.array([obj_thr], dtype=np.float64)])
        solve_cost = np.zeros_like(margin_cost)
    else:
        solve_cost = margin_cost
    connected_stats = None
    if connected_presolve and A.shape[0] and A.shape[1]:
        root = np.flatnonzero(
            (np.abs(margin_cost) > 1e-12) | (np.abs(solve_cost) > 1e-12)
        )
        if root.size:
            A_csr = A.tocsr()
            A_csc = A_csr.tocsc()
            keep_col = np.zeros(A_csr.shape[1], dtype=bool)
            keep_row = np.zeros(A_csr.shape[0], dtype=bool)
            stack = [int(x) for x in root]
            for x in stack:
                keep_col[x] = True
            while stack:
                col = stack.pop()
                for p in range(A_csc.indptr[col], A_csc.indptr[col + 1]):
                    row = int(A_csc.indices[p])
                    if keep_row[row]:
                        continue
                    keep_row[row] = True
                    for q in range(A_csr.indptr[row], A_csr.indptr[row + 1]):
                        nxt = int(A_csr.indices[q])
                        if not keep_col[nxt]:
                            keep_col[nxt] = True
                            stack.append(nxt)
            conn_cols = np.flatnonzero(keep_col)
            conn_rows = np.flatnonzero(keep_row)
            connected_stats = {
                "cols_before": int(A.shape[1]),
                "rows_before": int(A.shape[0]),
                "nnz_before": int(A.nnz),
                "cols_after": int(conn_cols.size),
                "rows_after": int(conn_rows.size),
            }
            if conn_cols.size < A.shape[1] or conn_rows.size < A.shape[0]:
                prev_cols, prev_rows, prev_nnz = A.shape[1], A.shape[0], A.nnz
                A = A[conn_rows, :][:, conn_cols].tocsr()
                connected_stats["nnz_after"] = int(A.nnz)
                rl = rl[conn_rows]
                ru = ru[conn_rows]
                cost = cost[conn_cols]
                margin_cost = margin_cost[conn_cols]
                solve_cost = solve_cost[conn_cols]
                lb = lb[conn_cols]
                ub = ub[conn_cols]
                if keep_cols is None:
                    keep_cols = conn_cols.astype(np.int64)
                else:
                    keep_cols = keep_cols[conn_cols]
                if mip_start_values is not None:
                    mip_start_values = mip_start_values[conn_cols]
                    if mip_start_indices is not None:
                        reverse = {int(col): int(pos) for pos, col in enumerate(conn_cols)}
                        mip_start_indices = np.asarray(
                            [reverse[int(col)] for col in mip_start_indices if int(col) in reverse],
                            dtype=np.int64,
                        )
                        if mip_start_indices.size == 0:
                            mip_start_values = None
                            mip_start_status = "skipped:no_start_cols_after_connected"
                logger.debug(
                    "connected_presolve cols=%s->%s rows=%s->%s nnz=%s->%s",
                    prev_cols,
                    A.shape[1],
                    prev_rows,
                    A.shape[0],
                    prev_nnz,
                    A.nnz,
                )
            else:
                connected_stats["nnz_after"] = int(A.nnz)
                logger.debug(
                    "connected_presolve no_reduction cols=%s rows=%s nnz=%s",
                    A.shape[1],
                    A.shape[0],
                    A.nnz,
                )
    int_mask = None
    if A.shape[1]:
        if keep_cols is None:
            col_orig = np.arange(A.shape[1], dtype=np.int64)
        else:
            col_orig = np.asarray(keep_cols, dtype=np.int64)
        int_mask = (col_orig >= hz.n_cont) & (col_orig < hz.n_cont + hz.n_bin)
    fbbt_stats = None
    if int(fbbt_passes) > 0 and A.shape[0] and A.shape[1]:
        fbbt_empty, lb, ub, fbbt_stats = sparse_fbbt_tighten_bounds(
            A, rl, ru, lb, ub,
            integer_mask=int_mask,
            max_passes=int(fbbt_passes),
        )
        logger.debug(
            "fbbt_presolve status=%s passes=%s tightened=%s fixed_int=%s "
            "max_width_delta=%s",
            "infeasible" if fbbt_empty else "ok",
            fbbt_stats.get("passes"),
            fbbt_stats.get("tightened"),
            fbbt_stats.get("fixed_int"),
            fbbt_stats.get("max_width_delta"),
        )
        if fbbt_empty:
            stats = {
                "status": "fbbt_infeasible",
                "nodes": 0,
                "dual_bound": None,
                "obj": None,
                "gap": None,
                "max_integrality": None,
                "obj_thr": obj_thr,
                "const_z": const_z,
                "margin_dual_bound": None,
                "elim_singletons": elim_count,
                "elim_eq_subst": eq_subst_count,
                "cutoff_as_row": bool(cutoff_as_row),
                "mip_start": bool(mip_start_values is not None),
                "mip_start_status": mip_start_status,
                "connected_presolve": connected_stats,
                "fbbt": fbbt_stats,
                "fbbt_fixed_subst": fixed_subst_stats,
            }
            sparse_milp_cutoff_highs.last_stats = stats
            return "EMPTY:fbbt_infeasible", None, None
        if mip_start_values is not None:
            mip_start_values = np.clip(mip_start_values, lb, ub)
        fixed_mask = np.isfinite(lb) & np.isfinite(ub) & ((ub - lb) <= 1e-9)
        if np.any(fixed_mask):
            fixed_pos = np.flatnonzero(fixed_mask)
            fixed_vals = 0.5 * (lb[fixed_pos] + ub[fixed_pos])
            current_cols = (
                np.arange(A.shape[1], dtype=np.int64)
                if keep_cols is None
                else np.asarray(keep_cols, dtype=np.int64)
            )
            fixed_orig = current_cols[fixed_pos]
            contrib = np.asarray(A[:, fixed_pos] @ fixed_vals, dtype=np.float64).reshape(-1)
            rl = rl - contrib
            ru = ru - contrib
            solve_obj_offset += float(solve_cost[fixed_pos] @ fixed_vals)
            reconstruct_base[fixed_orig] = fixed_vals

            keep_local = np.ones(A.shape[1], dtype=bool)
            keep_local[fixed_pos] = False
            before_cols, before_rows, before_nnz = int(A.shape[1]), int(A.shape[0]), int(A.nnz)
            A = A[:, keep_local].tocsr()
            A.eliminate_zeros()
            cost = cost[keep_local]
            margin_cost = margin_cost[keep_local]
            solve_cost = solve_cost[keep_local]
            lb = lb[keep_local]
            ub = ub[keep_local]
            keep_cols = current_cols[keep_local]
            if mip_start_values is not None:
                mip_start_values = mip_start_values[keep_local]
                if mip_start_indices is not None:
                    old_to_new = -np.ones(keep_local.size, dtype=np.int64)
                    old_to_new[np.flatnonzero(keep_local)] = np.arange(int(np.count_nonzero(keep_local)))
                    mip_start_indices = np.asarray(
                        [
                            int(old_to_new[int(col)])
                            for col in mip_start_indices
                            if 0 <= int(col) < old_to_new.size and old_to_new[int(col)] >= 0
                        ],
                        dtype=np.int64,
                    )
                    if mip_start_indices.size == 0:
                        mip_start_values = None
                        mip_start_status = "skipped:no_start_cols_after_fbbt_fixed"

            row_nnz = np.diff(A.indptr)
            zero_rows = np.flatnonzero(row_nnz == 0)
            dropped_zero_rows = 0
            if zero_rows.size:
                bad_zero = zero_rows[
                    ((np.isfinite(rl[zero_rows])) & (rl[zero_rows] > 1e-9))
                    | ((np.isfinite(ru[zero_rows])) & (ru[zero_rows] < -1e-9))
                ]
                if bad_zero.size:
                    r0 = int(bad_zero[0])
                    fixed_subst_stats = {
                        "fixed_cols": int(fixed_pos.size),
                        "cols_before": before_cols,
                        "cols_after": int(A.shape[1]),
                        "rows_before": before_rows,
                        "rows_after": int(A.shape[0]),
                        "nnz_before": before_nnz,
                        "nnz_after": int(A.nnz),
                        "zero_row_infeasible": {
                            "bad_row": r0,
                            "rl": float(rl[r0]) if np.isfinite(rl[r0]) else None,
                            "ru": float(ru[r0]) if np.isfinite(ru[r0]) else None,
                        },
                    }
                    stats = {
                        "status": "fbbt_fixed_zero_row_infeasible",
                        "nodes": 0,
                        "dual_bound": None,
                        "obj": None,
                        "gap": None,
                        "max_integrality": None,
                        "obj_thr": obj_thr,
                        "solver_obj_thr": obj_thr - solve_obj_offset,
                        "const_z": const_z,
                        "solve_obj_offset": solve_obj_offset,
                        "margin_dual_bound": None,
                        "elim_singletons": elim_count,
                        "elim_eq_subst": eq_subst_count,
                        "cutoff_as_row": bool(cutoff_as_row),
                        "mip_start": bool(mip_start_values is not None),
                        "mip_start_status": mip_start_status,
                        "connected_presolve": connected_stats,
                        "fbbt": fbbt_stats,
                        "fbbt_fixed_subst": fixed_subst_stats,
                    }
                    sparse_milp_cutoff_highs.last_stats = stats
                    logger.debug(
                        "fbbt_fixed_subst fixed_cols=%s cols=%s->%s rows=%s->%s "
                        "nnz=%s->%s zero_row_infeasible=%s",
                        fixed_pos.size,
                        before_cols,
                        A.shape[1],
                        before_rows,
                        A.shape[0],
                        before_nnz,
                        A.nnz,
                        r0,
                    )
                    return "EMPTY:fbbt_fixed_zero_row_infeasible", None, None
                keep_rows = np.ones(A.shape[0], dtype=bool)
                keep_rows[zero_rows] = False
                dropped_zero_rows = int(zero_rows.size)
                A = A[keep_rows, :].tocsr()
                rl = rl[keep_rows]
                ru = ru[keep_rows]

            fixed_subst_stats = {
                "fixed_cols": int(fixed_pos.size),
                "cols_before": before_cols,
                "cols_after": int(A.shape[1]),
                "rows_before": before_rows,
                "rows_after": int(A.shape[0]),
                "nnz_before": before_nnz,
                "nnz_after": int(A.nnz),
                "dropped_zero_rows": dropped_zero_rows,
                "solve_obj_offset": float(solve_obj_offset),
            }
            logger.debug(
                "fbbt_fixed_subst fixed_cols=%s cols=%s->%s rows=%s->%s nnz=%s->%s",
                fixed_pos.size,
                before_cols,
                A.shape[1],
                before_rows,
                A.shape[0],
                before_nnz,
                A.nnz,
            )

    rb_empty, rb_stats = sparse_row_bound_infeasible(A, rl, ru, lb, ub)
    if rb_empty:
        stats = {
            "status": "row_bound_infeasible",
            "nodes": 0,
            "dual_bound": None,
            "obj": None,
            "gap": None,
            "max_integrality": None,
            "obj_thr": obj_thr,
            "const_z": const_z,
            "margin_dual_bound": None,
            "elim_singletons": elim_count,
            "elim_eq_subst": eq_subst_count,
            "cutoff_as_row": bool(cutoff_as_row),
            "mip_start": bool(mip_start_values is not None),
            "mip_start_status": mip_start_status,
            "connected_presolve": connected_stats,
            "fbbt": fbbt_stats,
            "fbbt_fixed_subst": fixed_subst_stats,
            "row_bound_infeasible": rb_stats,
        }
        sparse_milp_cutoff_highs.last_stats = stats
        logger.debug(
            "row_bound_infeasible bad_row=%s bad_count=%s row_min=%s row_max=%s "
            "rl=%s ru=%s",
            rb_stats.get("bad_row"),
            rb_stats.get("bad_count"),
            rb_stats.get("row_min"),
            rb_stats.get("row_max"),
            rb_stats.get("rl"),
            rb_stats.get("ru"),
        )
        return "EMPTY:row_bound_infeasible", None, None

    relax_stats = None
    if float(relax_precheck_timeout) > 0.0 and A.shape[0] and A.shape[1]:
        relax_status, relax_margin, relax_stats = sparse_highs_relaxation_empty_precheck(
            highspy,
            A,
            rl,
            ru,
            lb,
            ub,
            solve_cost,
            cutoff=cutoff,
            const_z=const_z + solve_obj_offset,
            time_limit=float(relax_precheck_timeout),
            cutoff_as_row=bool(cutoff_as_row),
            multirow_feas=bool(multirow_feas),
            highs_threads=highs_threads,
            highs_parallel=highs_parallel,
            highs_options=highs_options,
        )
        logger.debug(
            "relax_precheck status=%s margin=%s sec=%s nodes=%s",
            relax_status,
            relax_margin,
            relax_stats.get("sec"),
            relax_stats.get("nodes"),
        )
        if relax_status.startswith("EMPTY:"):
            stats = {
                "status": relax_status,
                "nodes": 0,
                "dual_bound": relax_stats.get("dual_bound"),
                "obj": relax_stats.get("obj"),
                "gap": None,
                "max_integrality": None,
                "obj_thr": obj_thr,
                "const_z": const_z,
                "margin_dual_bound": relax_stats.get("margin_dual_bound"),
                "elim_singletons": elim_count,
                "elim_eq_subst": eq_subst_count,
                "cutoff_as_row": bool(cutoff_as_row),
                "mip_start": bool(mip_start_values is not None),
                "mip_start_status": mip_start_status,
                "connected_presolve": connected_stats,
                "fbbt": fbbt_stats,
                "fbbt_fixed_subst": fixed_subst_stats,
                "relax_precheck": relax_stats,
            }
            sparse_milp_cutoff_highs.last_stats = stats
            return relax_status, relax_margin, None
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-9)
    solver_obj_thr = float(obj_thr - solve_obj_offset)
    if (not cutoff_as_row) or multirow_feas:
        h.setOptionValue("objective_target", solver_obj_thr)
        h.setOptionValue("objective_bound", solver_obj_thr)
    h.setOptionValue("presolve", "on")
    if highs_threads > 0:
        h.setOptionValue("threads", int(highs_threads))
    if highs_parallel:
        h.setOptionValue("parallel", highs_parallel)
    if highs_heuristic_effort is not None:
        h.setOptionValue("mip_heuristic_effort", float(highs_heuristic_effort))
    if highs_options:
        for key, value in highs_options.items():
            h.setOptionValue(str(key), value)
    ncols = cost.size
    h.addCols(
        ncols,
        solve_cost.astype(float),
        lb.astype(float),
        ub.astype(float),
        0,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=float),
    )
    if keep_cols is None:
        int_idx = np.arange(hz.n_cont, hz.n_cont + hz.n_bin, dtype=np.int32)
    else:
        int_idx = np.nonzero((keep_cols >= hz.n_cont) & (keep_cols < hz.n_cont + hz.n_bin))[0].astype(np.int32)
    if int_idx.size:
        types = np.array([highspy.HighsVarType.kInteger] * int_idx.size)
        h.changeColsIntegrality(int_idx.size, int_idx, types)
    if A.shape[0]:
        h.addRows(
            A.shape[0],
            rl.astype(float),
            ru.astype(float),
            A.nnz,
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(float),
        )
    if mip_start_values is not None:
        start_entry_count = 0
        try:
            if mip_start_indices is None:
                start_indices = np.arange(ncols, dtype=np.int32)
            else:
                start_indices = mip_start_indices.astype(np.int32)
            start_entry_count = int(start_indices.size)
            start_values = np.clip(mip_start_values[start_indices], lb[start_indices], ub[start_indices]).astype(np.float64)
            ret = h.setSolution(start_indices.size, start_indices, start_values)
            mip_start_status = str(ret)
        except Exception as exc:
            mip_start_status = f"error:{type(exc).__name__}:{str(exc)[:80]}"
        logger.debug(
            "highs_mip_start status=%s entries=%s binary_only=%s",
            mip_start_status,
            start_entry_count,
            bool(mip_start_binary_only),
        )
    run_status = h.run()
    MS = highspy.HighsModelStatus
    st = h.getModelStatus()
    status = h.modelStatusToString(st)
    info = h.getInfo()
    def stat_float(name: str) -> Optional[float]:
        val = getattr(info, name, None)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    dual_bound = stat_float("mip_dual_bound")
    obj_value = stat_float("objective_function_value")
    margin_dual_bound = None if dual_bound is None else const_z + dual_bound
    nodes_raw = getattr(info, "mip_node_count", None)
    try:
        nodes = None if nodes_raw is None else int(nodes_raw)
    except (TypeError, ValueError):
        nodes = None
    stats = {
        "run_status": str(run_status),
        "status": status,
        "nodes": nodes,
        "dual_bound": dual_bound,
        "obj": obj_value,
        "gap": stat_float("mip_gap"),
        "max_integrality": stat_float("max_integrality_violation"),
        "obj_thr": obj_thr,
        "solver_obj_thr": solver_obj_thr,
        "const_z": const_z,
        "solve_obj_offset": solve_obj_offset,
        "margin_dual_bound": None if dual_bound is None else const_z + solve_obj_offset + dual_bound,
        "elim_singletons": elim_count,
        "elim_eq_subst": eq_subst_count,
        "cutoff_as_row": bool(cutoff_as_row),
        "mip_start": bool(mip_start_values is not None),
        "mip_start_status": mip_start_status,
        "connected_presolve": connected_stats,
        "fbbt": fbbt_stats,
        "fbbt_fixed_subst": fixed_subst_stats,
        "relax_precheck": relax_stats,
    }
    sparse_milp_cutoff_highs.last_stats = stats
    logger.debug(
        "highs_stats status=%s nodes=%s dual_bound=%s obj=%s gap=%s "
        "max_integrality=%s obj_thr=%s const_z=%s margin_dual_bound=%s",
        status,
        stats["nodes"],
        stats["dual_bound"],
        stats["obj"],
        stats["gap"],
        stats["max_integrality"],
        obj_thr,
        const_z,
        stats["margin_dual_bound"],
    )

    def dual_bound_proves_empty() -> bool:
        if cutoff_as_row and not multirow_feas:
            return False
        if run_status != highspy.HighsStatus.kOk or status == "Not Set":
            return False
        mb = stats.get("margin_dual_bound")
        return mb is not None and np.isfinite(float(mb)) and float(mb) > float(cutoff) + 1e-7

    def solution(solver_values: Optional[np.ndarray] = None):
        v = (
            np.asarray(h.getSolution().col_value, dtype=np.float64)
            if solver_values is None
            else np.asarray(solver_values, dtype=np.float64)
        )
        if keep_cols is None:
            full = v.copy()
        else:
            full = reconstruct_base.copy()
            full[np.asarray(keep_cols, dtype=np.int64)] = v
        xi = full[:base_ncols].copy()
        if hz.n_bin:
            xi[hz.n_cont:] = 2.0 * xi[hz.n_cont:] - 1.0
        if multirow_feas:
            y = hz.c + np.asarray(hz.Gc @ xi[:hz.n_cont]).reshape(-1)
            if hz.n_bin:
                y = y + np.asarray(hz.Gb @ xi[hz.n_cont:]).reshape(-1)
            val = float(np.max(Cmat @ y - tvec))
        else:
            val = original_const_z + float(original_margin_cost @ full)
        return val, xi

    def target_incumbent_from_nonterminal():
        """Accept a HiGHS incumbent only after checking MILP residuals."""
        incumbent_check = {
            "available": False,
            "accepted": False,
            "reason": "not_checked",
        }
        try:
            raw = np.asarray(h.getSolution().col_value, dtype=np.float64)
        except Exception as exc:
            incumbent_check.update({"reason": f"get_solution_error:{type(exc).__name__}:{str(exc)[:80]}"})
            stats["incumbent_check"] = incumbent_check
            return None
        incumbent_check["available"] = bool(raw.size)
        if raw.size != ncols or not np.all(np.isfinite(raw)):
            incumbent_check.update({"reason": f"bad_solution_shape:{raw.size}!={ncols}"})
            stats["incumbent_check"] = incumbent_check
            return None

        v = raw.copy()
        int_vio = 0.0
        if int_idx.size:
            ints = v[int_idx]
            rounded = np.rint(ints)
            int_vio = float(np.max(np.abs(ints - rounded))) if ints.size else 0.0
            incumbent_check["max_integrality"] = int_vio
            if int_vio > 1e-5:
                incumbent_check.update({"reason": "integrality_violation"})
                stats["incumbent_check"] = incumbent_check
                return None
            v[int_idx] = np.clip(rounded, 0.0, 1.0)
        else:
            incumbent_check["max_integrality"] = 0.0

        lb_vio = float(np.max(np.maximum(lb - v, 0.0))) if lb.size else 0.0
        ub_vio = float(np.max(np.maximum(v - ub, 0.0))) if ub.size else 0.0
        bound_vio = max(lb_vio, ub_vio)
        incumbent_check["max_bound_vio"] = bound_vio
        if bound_vio > 1e-6:
            incumbent_check.update({"reason": "bound_violation"})
            stats["incumbent_check"] = incumbent_check
            return None

        row_vio = 0.0
        row_vio_scaled = 0.0
        if A.shape[0]:
            av = np.asarray(A @ v, dtype=np.float64).reshape(-1)
            finite_lower = rl > -1e20
            finite_upper = ru < 1e20
            lower = np.where(finite_lower, rl - av, -np.inf)
            upper = np.where(finite_upper, av - ru, -np.inf)
            vio = np.maximum(np.maximum(lower, upper), 0.0)
            row_vio = float(np.max(vio)) if vio.size else 0.0
            scale = 1.0 + np.maximum(
                np.abs(av),
                np.maximum(
                    np.where(finite_lower, np.abs(rl), 0.0),
                    np.where(finite_upper, np.abs(ru), 0.0),
                ),
            )
            row_vio_scaled = float(np.max(vio / scale)) if vio.size else 0.0
        incumbent_check["max_row_vio"] = row_vio
        incumbent_check["max_row_vio_scaled"] = row_vio_scaled
        if row_vio > 5e-5:
            incumbent_check.update({"reason": "row_violation"})
            stats["incumbent_check"] = incumbent_check
            return None

        val, xi = solution(v)
        incumbent_check["margin"] = float(val)
        if not np.isfinite(val) or val > cutoff + 1e-7:
            incumbent_check.update({"reason": "cutoff_not_met"})
            stats["incumbent_check"] = incumbent_check
            return None
        incumbent_check.update({"accepted": True, "reason": "target_incumbent"})
        stats["incumbent_check"] = incumbent_check
        return val, xi

    if multirow_feas:
        if st == MS.kObjectiveTarget:
            val, xi = solution()
            return f"TARGET:{status}", val, xi
        if st in (MS.kObjectiveBound, MS.kInfeasible):
            return f"EMPTY:{status}", None, None
        if st == MS.kOptimal:
            val, xi = solution()
            if val <= cutoff + 1e-7:
                return f"TARGET:{status}", val, xi
            return f"EMPTY:{status}", val, None
        if dual_bound_proves_empty():
            return f"EMPTY:dual_bound:{status}", stats["margin_dual_bound"], None
        incumbent = target_incumbent_from_nonterminal()
        if incumbent is not None:
            val, xi = incumbent
            return f"TARGET:{status}:incumbent", val, xi
        return status, None, None

    if cutoff_as_row:
        if st == MS.kOptimal:
            val, xi = solution()
            if val <= cutoff + 1e-7:
                return f"TARGET:{status}", val, xi
            return f"EMPTY:{status}", val, None
        if st == MS.kInfeasible:
            return f"EMPTY:{status}", None, None
        incumbent = target_incumbent_from_nonterminal()
        if incumbent is not None:
            val, xi = incumbent
            return f"TARGET:{status}:incumbent", val, xi
        return status, None, None

    if st == MS.kObjectiveTarget:
        val, xi = solution()
        return f"TARGET:{status}", val, xi
    if st in (MS.kObjectiveBound, MS.kInfeasible):
        return f"EMPTY:{status}", None, None
    if st == MS.kOptimal:
        val, xi = solution()
        return f"OPTIMAL:{status}", val, xi
    if dual_bound_proves_empty():
        return f"EMPTY:dual_bound:{status}", stats["margin_dual_bound"], None
    incumbent = target_incumbent_from_nonterminal()
    if incumbent is not None:
        val, xi = incumbent
        return f"TARGET:{status}:incumbent", val, xi
    return status, None, None


def sparse_milp_cutoff_scip(
    hz: SparseHZono,
    C: np.ndarray,
    t: np.ndarray,
    time_limit: float,
    cutoff: float = 0.0,
    elim_singletons: bool = False,
    cutoff_as_row: bool = False,
    fbbt_passes: int = 0,
    scip_threads: int = 0,
    scip_options: Optional[Dict[str, object]] = None,
) -> Tuple[str, Optional[float], Optional[np.ndarray]]:
    try:
        from pyscipopt import Model, quicksum
    except Exception as exc:
        return f"no_pyscipopt:{exc}", None, None

    Cmat = C.reshape(C.shape[0], -1)
    tvec = t.reshape(-1)
    if Cmat.shape[0] != tvec.size:
        raise ValueError(f"C/t row mismatch: {Cmat.shape} vs {tvec.shape}")
    multirow_feas = Cmat.shape[0] > 1
    base_ncols = hz.n_cont + hz.n_bin
    extra_epigraph_cols = 1 if multirow_feas else 0
    if multirow_feas:
        cost = np.zeros(base_ncols + 1, dtype=np.float64)
        cost[-1] = 1.0
        const_z = 0.0
        obj_thr = float(cutoff)
    else:
        c_row = Cmat[0]
        obj_c = np.asarray(c_row @ hz.Gc).reshape(-1)
        obj_b = np.asarray(c_row @ hz.Gb).reshape(-1) if hz.n_bin else np.zeros(0)
        const = float(c_row @ hz.c - tvec[0])
        cost = np.concatenate([obj_c, 2.0 * obj_b])
        const_z = const - float(obj_b.sum())
        obj_thr = float(cutoff - const_z)

    if hz.n_eq:
        A = sp.hstack([hz.Ac, 2.0 * hz.Ab], format="csr")
        rhs = hz.b + np.asarray(hz.Ab.sum(axis=1)).reshape(-1)
    else:
        A = sp.csr_matrix((0, hz.n_cont + hz.n_bin), dtype=np.float64)
        rhs = np.zeros(0, dtype=np.float64)
    if extra_epigraph_cols:
        A = sp.hstack([A, sp.csr_matrix((A.shape[0], extra_epigraph_cols))], format="csr")
    if hz.n_ub:
        Ale = sp.hstack([hz.Auc, 2.0 * hz.Aub], format="csr")
        ble = hz.ub + np.asarray(hz.Aub.sum(axis=1)).reshape(-1)
    else:
        Ale = sp.csr_matrix((0, hz.n_cont + hz.n_bin), dtype=np.float64)
        ble = np.zeros(0, dtype=np.float64)
    if extra_epigraph_cols:
        Ale = sp.hstack([Ale, sp.csr_matrix((Ale.shape[0], extra_epigraph_cols))], format="csr")
    if multirow_feas:
        Csp = sp.csr_matrix(Cmat)
        unsafe_Ac = Csp @ hz.Gc
        unsafe_Ab = Csp @ hz.Gb if hz.n_bin else sp.csr_matrix((Cmat.shape[0], 0), dtype=np.float64)
        unsafe_A = sp.hstack(
            [
                unsafe_Ac,
                2.0 * unsafe_Ab,
                -sp.csr_matrix(np.ones((Cmat.shape[0], 1))),
            ],
            format="csr",
        )
        unsafe_b = tvec - Cmat @ hz.c + np.asarray(unsafe_Ab.sum(axis=1)).reshape(-1)
        Ale = sp.vstack([Ale, unsafe_A], format="csr")
        ble = np.concatenate([ble, unsafe_b.astype(np.float64)])

    lb = np.concatenate([-np.ones(hz.n_cont), np.zeros(hz.n_bin)])
    ub = np.ones(base_ncols)
    if extra_epigraph_cols:
        lb = np.concatenate([lb, np.array([-1e12], dtype=np.float64)])
        ub = np.concatenate([ub, np.array([1e12], dtype=np.float64)])

    rl = rhs.astype(np.float64).copy()
    ru = rhs.astype(np.float64).copy()
    keep_cols = None
    original_ncols = int(cost.size)
    reconstruct_base = np.zeros(original_ncols, dtype=np.float64)
    solve_obj_offset = 0.0
    elim_count = 0
    fbbt_stats = None
    fixed_subst_stats = None
    if elim_singletons and A.shape[0] and A.shape[1]:
        Acsc = A.tocsc()
        Alecsc = Ale.tocsc() if Ale.shape[0] else None
        removable = []
        for j in range(hz.n_cont):
            if abs(cost[j]) > 1e-12:
                continue
            if Alecsc is not None and Alecsc.indptr[j + 1] > Alecsc.indptr[j]:
                continue
            start, end = Acsc.indptr[j], Acsc.indptr[j + 1]
            if end - start == 1:
                removable.append(j)
        if removable:
            elim_min = np.zeros(A.shape[0], dtype=np.float64)
            elim_max = np.zeros(A.shape[0], dtype=np.float64)
            for j in removable:
                start, end = Acsc.indptr[j], Acsc.indptr[j + 1]
                r = int(Acsc.indices[start])
                a = float(Acsc.data[start])
                vals = (a * lb[j], a * ub[j])
                elim_min[r] += min(vals)
                elim_max[r] += max(vals)
            rl = rhs - elim_max
            ru = rhs - elim_min
            keep = np.ones(A.shape[1], dtype=bool)
            keep[np.asarray(removable, dtype=np.int64)] = False
            keep_cols = np.nonzero(keep)[0]
            A = A[:, keep_cols].tocsr()
            Ale = Ale[:, keep_cols].tocsr()
            cost = cost[keep_cols]
            lb = lb[keep_cols]
            ub = ub[keep_cols]
            elim_count = len(removable)
            logger.debug(
                "scip_elim_singletons removed_cont=%s kept_cols=%s rows=%s",
                elim_count,
                A.shape[1],
                A.shape[0],
            )
    if Ale.shape[0]:
        A = sp.vstack([A, Ale], format="csr")
        rl = np.concatenate([rl, np.full(Ale.shape[0], -1e30, dtype=np.float64)])
        ru = np.concatenate([ru, ble.astype(np.float64)])
    margin_cost = cost.copy()
    if cutoff_as_row and not multirow_feas:
        A = sp.vstack([A, sp.csr_matrix(margin_cost.reshape(1, -1))], format="csr")
        rl = np.concatenate([rl, np.array([-1e30], dtype=np.float64)])
        ru = np.concatenate([ru, np.array([obj_thr], dtype=np.float64)])
        solve_cost = np.zeros_like(margin_cost)
    else:
        solve_cost = margin_cost

    if int(fbbt_passes) > 0 and A.shape[0] and A.shape[1]:
        if keep_cols is None:
            col_orig = np.arange(A.shape[1], dtype=np.int64)
        else:
            col_orig = np.asarray(keep_cols, dtype=np.int64)
        int_mask = (col_orig >= hz.n_cont) & (col_orig < hz.n_cont + hz.n_bin)
        fbbt_empty, lb, ub, fbbt_stats = sparse_fbbt_tighten_bounds(
            A, rl, ru, lb, ub,
            integer_mask=int_mask,
            max_passes=int(fbbt_passes),
        )
        logger.debug(
            "scip_fbbt_presolve status=%s passes=%s tightened=%s fixed_int=%s "
            "max_width_delta=%s",
            "infeasible" if fbbt_empty else "ok",
            fbbt_stats.get("passes"),
            fbbt_stats.get("tightened"),
            fbbt_stats.get("fixed_int"),
            fbbt_stats.get("max_width_delta"),
        )
        if fbbt_empty:
            sparse_milp_cutoff_scip.last_stats = {
                "status": "fbbt_infeasible",
                "nodes": 0,
                "dual_bound": None,
                "obj": None,
                "gap": None,
                "max_integrality": None,
                "obj_thr": obj_thr,
                "solver_obj_thr": obj_thr - solve_obj_offset,
                "const_z": const_z,
                "solve_obj_offset": solve_obj_offset,
                "margin_dual_bound": None,
                "elim_singletons": elim_count,
                "cutoff_as_row": bool(cutoff_as_row),
                "fbbt": fbbt_stats,
            }
            return "EMPTY:fbbt_infeasible", None, None
        fixed_mask = np.isfinite(lb) & np.isfinite(ub) & ((ub - lb) <= 1e-9)
        if np.any(fixed_mask):
            fixed_pos = np.flatnonzero(fixed_mask)
            fixed_vals = 0.5 * (lb[fixed_pos] + ub[fixed_pos])
            current_cols = (
                np.arange(A.shape[1], dtype=np.int64)
                if keep_cols is None
                else np.asarray(keep_cols, dtype=np.int64)
            )
            fixed_orig = current_cols[fixed_pos]
            contrib = np.asarray(A[:, fixed_pos] @ fixed_vals, dtype=np.float64).reshape(-1)
            rl = rl - contrib
            ru = ru - contrib
            solve_obj_offset += float(solve_cost[fixed_pos] @ fixed_vals)
            reconstruct_base[fixed_orig] = fixed_vals

            keep_local = np.ones(A.shape[1], dtype=bool)
            keep_local[fixed_pos] = False
            before_cols, before_rows, before_nnz = int(A.shape[1]), int(A.shape[0]), int(A.nnz)
            A = A[:, keep_local].tocsr()
            A.eliminate_zeros()
            cost = cost[keep_local]
            margin_cost = margin_cost[keep_local]
            solve_cost = solve_cost[keep_local]
            lb = lb[keep_local]
            ub = ub[keep_local]
            keep_cols = current_cols[keep_local]

            row_nnz = np.diff(A.indptr)
            zero_rows = np.flatnonzero(row_nnz == 0)
            dropped_zero_rows = 0
            if zero_rows.size:
                bad_zero = zero_rows[
                    ((np.isfinite(rl[zero_rows])) & (rl[zero_rows] > 1e-9))
                    | ((np.isfinite(ru[zero_rows])) & (ru[zero_rows] < -1e-9))
                ]
                if bad_zero.size:
                    r0 = int(bad_zero[0])
                    fixed_subst_stats = {
                        "fixed_cols": int(fixed_pos.size),
                        "cols_before": before_cols,
                        "cols_after": int(A.shape[1]),
                        "rows_before": before_rows,
                        "rows_after": int(A.shape[0]),
                        "nnz_before": before_nnz,
                        "nnz_after": int(A.nnz),
                        "zero_row_infeasible": {
                            "bad_row": r0,
                            "rl": float(rl[r0]) if np.isfinite(rl[r0]) else None,
                            "ru": float(ru[r0]) if np.isfinite(ru[r0]) else None,
                        },
                    }
                    sparse_milp_cutoff_scip.last_stats = {
                        "status": "fbbt_fixed_zero_row_infeasible",
                        "nodes": 0,
                        "dual_bound": None,
                        "obj": None,
                        "gap": None,
                        "max_integrality": None,
                        "obj_thr": obj_thr,
                        "solver_obj_thr": obj_thr - solve_obj_offset,
                        "const_z": const_z,
                        "solve_obj_offset": solve_obj_offset,
                        "margin_dual_bound": None,
                        "elim_singletons": elim_count,
                        "cutoff_as_row": bool(cutoff_as_row),
                        "fbbt": fbbt_stats,
                        "fbbt_fixed_subst": fixed_subst_stats,
                    }
                    return "EMPTY:fbbt_fixed_zero_row_infeasible", None, None
                keep_rows = np.ones(A.shape[0], dtype=bool)
                keep_rows[zero_rows] = False
                dropped_zero_rows = int(zero_rows.size)
                A = A[keep_rows, :].tocsr()
                rl = rl[keep_rows]
                ru = ru[keep_rows]

            fixed_subst_stats = {
                "fixed_cols": int(fixed_pos.size),
                "cols_before": before_cols,
                "cols_after": int(A.shape[1]),
                "rows_before": before_rows,
                "rows_after": int(A.shape[0]),
                "nnz_before": before_nnz,
                "nnz_after": int(A.nnz),
                "dropped_zero_rows": dropped_zero_rows,
                "solve_obj_offset": float(solve_obj_offset),
            }
            logger.debug(
                "scip_fbbt_fixed_subst fixed_cols=%s cols=%s->%s rows=%s->%s nnz=%s->%s",
                fixed_pos.size,
                before_cols,
                A.shape[1],
                before_rows,
                A.shape[0],
                before_nnz,
                A.nnz,
            )

    rb_empty, rb_stats = sparse_row_bound_infeasible(A, rl, ru, lb, ub)
    if rb_empty:
        sparse_milp_cutoff_scip.last_stats = {
            "status": "row_bound_infeasible",
            "nodes": 0,
            "dual_bound": None,
            "obj": None,
            "gap": None,
            "max_integrality": None,
            "obj_thr": obj_thr,
            "solver_obj_thr": obj_thr - solve_obj_offset,
            "const_z": const_z,
            "solve_obj_offset": solve_obj_offset,
            "margin_dual_bound": None,
            "elim_singletons": elim_count,
            "cutoff_as_row": bool(cutoff_as_row),
            "fbbt": fbbt_stats,
            "fbbt_fixed_subst": fixed_subst_stats,
            "row_bound_infeasible": rb_stats,
        }
        logger.debug(
            "scip_row_bound_infeasible bad_row=%s bad_count=%s row_min=%s "
            "row_max=%s rl=%s ru=%s",
            rb_stats.get("bad_row"),
            rb_stats.get("bad_count"),
            rb_stats.get("row_min"),
            rb_stats.get("row_max"),
            rb_stats.get("rl"),
            rb_stats.get("ru"),
        )
        return "EMPTY:row_bound_infeasible", None, None

    model = Model()
    model.hideOutput(True)
    model.setParam("limits/time", float(time_limit))
    if int(scip_threads) > 0:
        for key in ("parallel/maxnthreads", "lp/threads"):
            try:
                model.setParam(key, int(scip_threads))
            except Exception:
                pass
    if scip_options:
        for key, value in scip_options.items():
            try:
                model.setParam(str(key), value)
            except Exception:
                pass
    solver_obj_thr = float(obj_thr - solve_obj_offset)
    try:
        model.setParam("limits/primal", solver_obj_thr)
    except Exception:
        pass
    vars_ = []
    col_map = keep_cols if keep_cols is not None else np.arange(cost.size, dtype=np.int64)
    for j, orig_j in enumerate(col_map):
        lo = float(lb[j])
        hi = float(ub[j])
        if int(orig_j) >= hz.n_cont and int(orig_j) < hz.n_cont + hz.n_bin:
            vars_.append(model.addVar(vtype="B", lb=max(0.0, lo), ub=min(1.0, hi), name=f"z{int(orig_j - hz.n_cont)}"))
        else:
            vars_.append(model.addVar(vtype="C", lb=lo, ub=hi, name=f"x{int(orig_j)}"))

    def row_expr(row: int):
        start, end = A.indptr[row], A.indptr[row + 1]
        return quicksum(float(A.data[p]) * vars_[int(A.indices[p])] for p in range(start, end))

    infeas_const = False
    for r in range(A.shape[0]):
        lhs = float(rl[r])
        rhs_v = float(ru[r])
        if A.indptr[r] == A.indptr[r + 1]:
            if lhs > 1e-9 or rhs_v < -1e-9:
                infeas_const = True
                break
            continue
        expr = row_expr(r)
        lhs_finite = lhs > -1e20
        rhs_finite = rhs_v < 1e20
        if lhs_finite and rhs_finite and abs(lhs - rhs_v) <= 1e-9:
            model.addCons(expr == rhs_v)
        else:
            if lhs_finite:
                model.addCons(expr >= lhs)
            if rhs_finite:
                model.addCons(expr <= rhs_v)
    if infeas_const:
        sparse_milp_cutoff_scip.last_stats = {
            "status": "constant_infeasible",
            "nodes": 0,
            "dual_bound": None,
            "obj": None,
            "gap": None,
            "max_integrality": None,
            "obj_thr": obj_thr,
            "const_z": const_z,
            "margin_dual_bound": None,
            "elim_singletons": elim_count,
            "cutoff_as_row": bool(cutoff_as_row),
        }
        return "EMPTY:constant_infeasible", None, None

    obj_terms = [(j, float(v)) for j, v in enumerate(solve_cost) if abs(float(v)) > 1e-12]
    if obj_terms:
        model.setObjective(quicksum(v * vars_[j] for j, v in obj_terms), "minimize")
    else:
        model.setObjective(0.0, "minimize")
    model.optimize()
    status = str(model.getStatus())
    sol = model.getBestSol()

    def solution() -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if sol is None:
            return None, None, None
        v = np.asarray([model.getSolVal(sol, var) for var in vars_], dtype=np.float64)
        if keep_cols is None:
            full = v.copy()
        else:
            full = reconstruct_base.copy()
            full[np.asarray(keep_cols, dtype=np.int64)] = v
        xi = full[:base_ncols].copy()
        if hz.n_bin:
            xi[hz.n_cont:] = 2.0 * xi[hz.n_cont:] - 1.0
        y = hz.c + np.asarray(hz.Gc @ xi[:hz.n_cont]).reshape(-1)
        if hz.n_bin:
            y = y + np.asarray(hz.Gb @ xi[hz.n_cont:]).reshape(-1)
        if multirow_feas:
            val = float(np.max(Cmat @ y - tvec))
        else:
            val = float(Cmat[0] @ y - tvec[0])
        return val, xi, v

    val, xi, solver_v = solution()
    try:
        obj_val = float(model.getObjVal()) if sol is not None else None
    except Exception:
        obj_val = None
    try:
        dual_bound = float(model.getDualbound())
    except Exception:
        dual_bound = None
    stats = {
        "status": status,
        "nodes": int(model.getNNodes()),
        "dual_bound": dual_bound,
        "obj": obj_val,
        "gap": float(model.getGap()) if sol is not None else None,
        "max_integrality": None,
        "obj_thr": obj_thr,
        "solver_obj_thr": solver_obj_thr,
        "const_z": const_z,
        "solve_obj_offset": solve_obj_offset,
        "margin_dual_bound": None if dual_bound is None else const_z + solve_obj_offset + dual_bound,
        "elim_singletons": elim_count,
        "cutoff_as_row": bool(cutoff_as_row),
        "fbbt": fbbt_stats,
        "fbbt_fixed_subst": fixed_subst_stats,
    }
    if solver_v is not None:
        int_vio = 0.0
        if vars_:
            int_positions = [
                j for j, orig_j in enumerate(col_map)
                if int(orig_j) >= hz.n_cont and int(orig_j) < hz.n_cont + hz.n_bin
            ]
            if int_positions:
                ints = solver_v[np.asarray(int_positions, dtype=np.int64)]
                int_vio = float(np.max(np.abs(ints - np.rint(ints)))) if ints.size else 0.0
        lb_vio = float(np.max(np.maximum(lb - solver_v, 0.0))) if lb.size else 0.0
        ub_vio = float(np.max(np.maximum(solver_v - ub, 0.0))) if ub.size else 0.0
        row_vio = 0.0
        row_vio_scaled = 0.0
        if A.shape[0]:
            av = np.asarray(A @ solver_v, dtype=np.float64).reshape(-1)
            lower = np.where(rl > -1e20, rl - av, -np.inf)
            upper = np.where(ru < 1e20, av - ru, -np.inf)
            vio = np.maximum(np.maximum(lower, upper), 0.0)
            row_vio = float(np.max(vio)) if vio.size else 0.0
            scale = 1.0 + np.maximum(
                np.abs(av),
                np.maximum(
                    np.where(rl > -1e20, np.abs(rl), 0.0),
                    np.where(ru < 1e20, np.abs(ru), 0.0),
                ),
            )
            row_vio_scaled = float(np.max(vio / scale)) if vio.size else 0.0
            if cutoff_as_row and not multirow_feas:
                stats["cutoff_row_lhs"] = float(av[-1])
                stats["cutoff_row_rhs"] = float(ru[-1])
        stats["max_integrality"] = int_vio
        stats["max_bound_vio"] = max(lb_vio, ub_vio)
        stats["max_row_vio"] = row_vio
        stats["max_row_vio_scaled"] = row_vio_scaled
    sparse_milp_cutoff_scip.last_stats = stats
    logger.debug(
        "scip_stats status=%s nodes=%s dual_bound=%s obj=%s gap=%s obj_thr=%s "
        "const_z=%s margin=%s row_vio=%s int_vio=%s",
        status,
        stats["nodes"],
        stats["dual_bound"],
        stats["obj"],
        stats["gap"],
        obj_thr,
        const_z,
        val,
        stats.get("max_row_vio"),
        stats.get("max_integrality"),
    )

    st = status.lower()
    feasible_incumbent = (
        solver_v is not None
        and stats.get("max_bound_vio", 0.0) <= 1e-6
        and stats.get("max_integrality", 0.0) <= 1e-5
        and stats.get("max_row_vio", 0.0) <= 5e-5
    )
    if feasible_incumbent and val is not None and val <= cutoff + 1e-7:
        return f"TARGET:{status}", val, xi
    if st in {"infeasible", "inforunbd"}:
        return f"EMPTY:{status}", None, None
    if st == "optimal":
        if val is not None and val > cutoff + 1e-7:
            return f"EMPTY:{status}", val, None
        if feasible_incumbent:
            return f"TARGET:{status}", val, xi
        return status, val, xi if val is not None else None
    if not (cutoff_as_row and not multirow_feas):
        mb = stats.get("margin_dual_bound")
        if mb is not None and np.isfinite(float(mb)) and float(mb) > float(cutoff) + 1e-7:
            return f"EMPTY:dual_bound:{status}", float(mb), None
    return status, val, xi if val is not None else None




def _spec_np(C, thresholds, out_dim: int):
    C = np.asarray(C, dtype=np.float64).reshape(-1, out_dim)
    t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if t.size == 1 and C.shape[0] != 1:
        t = np.repeat(t, C.shape[0])
    if (
        C.shape[0] == 0
        or t.size != C.shape[0]
        or not np.all(np.isfinite(C))
        or not np.all(np.isfinite(t))
    ):
        raise ValueError(
            f"objective C/t are malformed or non-finite: "
            f"C={C.shape}, t={t.shape}"
        )
    return C, t


def _normalize_safe_row_groups(value, n_rows: int):
    """Validate an exact partition of sound alternative upper-plane rows."""

    if value is None:
        return None
    try:
        raw_groups = tuple(value)
    except TypeError as exc:
        raise ValueError("safe_row_groups must be an iterable") from exc
    if not raw_groups:
        raise ValueError("safe_row_groups must not be empty")
    groups = []
    flattened = []
    for group_index, raw_group in enumerate(raw_groups):
        if isinstance(raw_group, (str, bytes)):
            raise ValueError(
                f"safe_row_groups[{group_index}] is not a row iterable"
            )
        try:
            raw_rows = tuple(raw_group)
        except TypeError as exc:
            raise ValueError(
                f"safe_row_groups[{group_index}] is not iterable"
            ) from exc
        if not raw_rows:
            raise ValueError(
                f"safe_row_groups[{group_index}] must not be empty"
            )
        rows = []
        for raw_row in raw_rows:
            if isinstance(raw_row, (bool, np.bool_)) or not isinstance(
                raw_row, (int, np.integer)
            ):
                raise ValueError(
                    "safe_row_groups entries must be integer row ids"
                )
            row = int(raw_row)
            if row < 0 or row >= int(n_rows):
                raise ValueError(
                    f"safe_row_groups row {row} is outside [0,{n_rows})"
                )
            rows.append(row)
            flattened.append(row)
        if len(set(rows)) != len(rows):
            raise ValueError(
                f"safe_row_groups[{group_index}] repeats a row"
            )
        groups.append(tuple(rows))
    if (
        len(flattened) != int(n_rows)
        or len(set(flattened)) != int(n_rows)
        or set(flattened) != set(range(int(n_rows)))
    ):
        raise ValueError(
            "safe_row_groups must partition every objective row exactly once"
        )
    return tuple(groups)


def _dyadic_pair_cube_candidate(
    center0,
    generators0,
    center1,
    generators1,
    *,
    denominator: int,
    max_union_terms: int = 250_000,
    deadline=None,
):
    """Return the best strict interior dyadic free-cube mixture candidate.

    Every finite binary64 center/generator is converted with
    ``as_integer_ratio`` and shifted to one common power-of-two denominator.
    The following *stored-float* discrete convex objective is then minimized
    with exact Python-integer forward differences:

    ``F(k) = (k*c0 + (Q-k)*c1
                 + sum_j |k*g0[j] + (Q-k)*g1[j]|) / Q``

    A logarithmic search finds the first nonnegative forward difference, so
    the chosen ``k`` is an exact grid argmin of that proxy (ties choose the
    smallest ``k``).  This exactness applies only to candidate selection:
    this routine has no proof authority, and only the downstream outward
    cube/Lagrangian checker may certify SAFE.

    Only the union of the two CSR supports is materialized.  Thus the memory
    cost is ``O(nnz(g0 union g1))``, independent of the generator dimension.
    Term/deadline limits fail closed by returning no candidate.
    """

    denominator = int(denominator)
    if (
        denominator < 2
        or denominator & (denominator - 1)
    ):
        raise ValueError(
            "dyadic mixture denominator must be a power of two >= 2"
        )
    max_union_terms = int(max_union_terms)
    if max_union_terms <= 0:
        raise ValueError("dyadic mixture term cap must be positive")
    g0 = sp.csr_matrix(
        generators0, dtype=np.float64, copy=True
    )
    g1 = sp.csr_matrix(
        generators1, dtype=np.float64, copy=True
    )
    if g0.shape[0] != 1 or g1.shape != g0.shape:
        raise ValueError("dyadic mixture rows have incompatible shapes")
    for row in (g0, g1):
        row.sum_duplicates()
        row.sort_indices()
        row.eliminate_zeros()
    if (
        not np.isfinite(float(center0))
        or not np.isfinite(float(center1))
        or np.any(~np.isfinite(g0.data))
        or np.any(~np.isfinite(g1.data))
    ):
        return None

    # Align only the sparse union.  np.union1d is implemented in C and avoids
    # the large per-coordinate Python dictionaries used by the first probe.
    union_indices = np.union1d(g0.indices, g1.indices)
    if union_indices.size > max_union_terms:
        return None
    if deadline is not None and time.monotonic() >= float(deadline):
        return None
    values0 = np.zeros(union_indices.size, dtype=np.float64)
    values1 = np.zeros(union_indices.size, dtype=np.float64)
    if g0.nnz:
        values0[
            np.searchsorted(union_indices, g0.indices)
        ] = g0.data
    if g1.nnz:
        values1[
            np.searchsorted(union_indices, g1.indices)
        ] = g1.data

    # Convert each stored binary64 exactly, but avoid per-term Fraction
    # construction.  Denominators returned by as_integer_ratio are powers of
    # two, so a left shift puts all values on one exact integer scale.
    raw_ratios0 = []
    raw_ratios1 = []
    common_exponent = 0
    center0_ratio = float(center0).as_integer_ratio()
    center1_ratio = float(center1).as_integer_ratio()
    common_exponent = max(
        center0_ratio[1].bit_length() - 1,
        center1_ratio[1].bit_length() - 1,
    )
    for index, (value0, value1) in enumerate(zip(values0, values1)):
        if (
            deadline is not None
            and index % 4096 == 0
            and time.monotonic() >= float(deadline)
        ):
            return None
        ratio0 = float(value0).as_integer_ratio()
        ratio1 = float(value1).as_integer_ratio()
        raw_ratios0.append(ratio0)
        raw_ratios1.append(ratio1)
        common_exponent = max(
            common_exponent,
            ratio0[1].bit_length() - 1,
            ratio1[1].bit_length() - 1,
        )

    def _scaled_integer(ratio):
        numerator, ratio_denominator = ratio
        exponent = ratio_denominator.bit_length() - 1
        return int(numerator) << int(common_exponent - exponent)

    center0_integer = _scaled_integer(center0_ratio)
    center1_integer = _scaled_integer(center1_ratio)
    integers0 = tuple(
        _scaled_integer(ratio) for ratio in raw_ratios0
    )
    integers1 = tuple(
        _scaled_integer(ratio) for ratio in raw_ratios1
    )
    deltas = tuple(
        value0 - value1
        for value0, value1 in zip(integers0, integers1)
    )
    center_delta = center0_integer - center1_integer
    objective_cache = {}
    forward_cache = {}

    def _check_deadline():
        if deadline is not None and time.monotonic() >= float(deadline):
            raise TimeoutError("dyadic exact candidate deadline reached")

    def _proxy_integer(numerator):
        numerator = int(numerator)
        cached = objective_cache.get(numerator)
        if cached is not None:
            return cached
        _check_deadline()
        other = denominator - numerator
        value = (
            numerator * center0_integer
            + other * center1_integer
            + sum(
                abs(numerator * value0 + other * value1)
                for value0, value1 in zip(integers0, integers1)
            )
        )
        objective_cache[numerator] = int(value)
        return int(value)

    def _forward_difference_integer(numerator):
        """Return the exact scaled Q*(F(k+1)-F(k))."""

        numerator = int(numerator)
        cached = forward_cache.get(numerator)
        if cached is not None:
            return cached
        _check_deadline()
        value = center_delta
        for value1, delta in zip(integers1, deltas):
            at_k = denominator * value1 + numerator * delta
            value += abs(at_k + delta) - abs(at_k)
        forward_cache[numerator] = int(value)
        return int(value)

    try:
        # Search [0,Q] with Q as a sentinel meaning every forward difference
        # is negative.  Exact convexity makes the first nonnegative difference
        # the smallest grid minimizer.
        lo = 0
        hi = denominator
        while lo < hi:
            middle = (lo + hi) // 2
            if (
                middle == denominator
                or _forward_difference_integer(middle) >= 0
            ):
                hi = middle
            else:
                lo = middle + 1
        numerator = int(lo)
        left_ok = (
            numerator == 0
            or _forward_difference_integer(numerator - 1) < 0
        )
        right_ok = (
            numerator == denominator
            or _forward_difference_integer(numerator) >= 0
        )
        if not (left_ok and right_ok):
            return None
        endpoint0_integer = _proxy_integer(0)
        endpoint1_integer = _proxy_integer(denominator)
        candidate_integer = _proxy_integer(numerator)
    except TimeoutError:
        return None

    endpoint_best_integer = min(
        endpoint0_integer, endpoint1_integer
    )
    if (
        numerator <= 0
        or numerator >= denominator
        or candidate_integer >= endpoint_best_integer
    ):
        return None
    exact_denominator = denominator << common_exponent
    endpoint_best = float(
        Fraction(endpoint_best_integer, exact_denominator)
    )
    candidate_value = float(
        Fraction(candidate_integer, exact_denominator)
    )
    if not np.isfinite(endpoint_best) or not np.isfinite(candidate_value):
        return None
    numerical_guard = (
        64.0
        * np.finfo(np.float64).eps
        * (1.0 + abs(endpoint_best))
    )
    improvement_integer = endpoint_best_integer - candidate_integer
    guard_numerator, guard_denominator = (
        float(numerical_guard).as_integer_ratio()
    )
    if (
        improvement_integer * guard_denominator
        <= guard_numerator * exact_denominator
    ):
        return None
    exact_digest = hashlib.sha256()
    for value in (
        denominator,
        common_exponent,
        numerator,
        endpoint_best_integer,
        candidate_integer,
        improvement_integer,
    ):
        magnitude = abs(int(value))
        encoded = magnitude.to_bytes(
            max(1, (magnitude.bit_length() + 7) // 8),
            byteorder="little",
            signed=False,
        )
        exact_digest.update(
            b"-" if int(value) < 0 else b"+"
        )
        exact_digest.update(
            len(encoded).to_bytes(8, byteorder="little")
        )
        exact_digest.update(encoded)
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "proxy_upper": float(candidate_value),
        "endpoint_proxy_upper": float(endpoint_best),
        "proxy_improvement": float(endpoint_best - candidate_value),
        "proxy_numerical_guard": float(numerical_guard),
        "discrete_search_method": (
            "exact_integer_forward_difference_binary"
        ),
        "discrete_bracket_validated": bool(left_ok and right_ok),
        "exact_stored_float_grid_argmin_validated": True,
        "exact_common_denominator_bits": int(common_exponent),
        "exact_objective_sha256": exact_digest.hexdigest(),
        "aligned_generator_nnz": int(union_indices.size),
        "objective_evaluations": int(len(objective_cache)),
        "forward_difference_evaluations": int(len(forward_cache)),
        "candidate_arithmetic": (
            "python_int_exact_binary64_dyadic"
        ),
        # Private comparison fields are removed before the JSON receipt.
        "_proxy_exact_numerator": int(candidate_integer),
        "_proxy_exact_denominator": int(exact_denominator),
    }


def _augment_safe_groups_with_dyadic_mixtures(
    c,
    Gc,
    Gb,
    C,
    t,
    safe_groups,
    *,
    grid_bits: int,
    candidate_deadline=None,
    exact_total_term_cap: int = 1_000_000,
    exact_pair_term_cap: int = 250_000,
):
    """Append at most one dyadic convex-mixture objective per property group."""

    receipt = {
        "schema": "hz_safe_group_dyadic_mixture_v1",
        "enabled": bool(int(grid_bits) > 0),
        "status": "disabled" if int(grid_bits) == 0 else "pending",
        "candidate_only": True,
        "proof_authority": False,
        "grid_bits": int(grid_bits),
        "selected_groups": 0,
        "appended_rows": 0,
        "exact_total_term_cap": int(exact_total_term_cap),
        "exact_pair_term_cap": int(exact_pair_term_cap),
        "exact_search_deadline_enforced": bool(
            candidate_deadline is not None
        ),
    }
    if int(grid_bits) == 0:
        return C, t, safe_groups, receipt
    if safe_groups is None:
        raise ValueError("dyadic group mixtures require safe row groups")
    if not (1 <= int(grid_bits) <= 24):
        raise ValueError("dyadic group mixture grid bits must lie in [1,24]")
    C = np.asarray(C, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    identity = np.eye(int(c.size), dtype=np.float64)
    if (
        C.shape != identity.shape
        or not np.array_equal(C, identity)
        or t.shape != (int(c.size),)
        or np.any(t != 0.0)
    ):
        raise ValueError(
            "dyadic group mixtures require exact identity/zero objectives"
        )
    denominator = 1 << int(grid_bits)
    generators = sp.hstack([Gc, Gb], format="csr")
    search_started = time.monotonic()
    receipt.update(
        {
            "exact_integer_search": True,
            "exact_search_initial_budget_s": (
                max(
                    0.0,
                    float(candidate_deadline) - search_started,
                )
                if candidate_deadline is not None
                else None
            ),
        }
    )
    selected = []
    pairs_considered = 0
    skipped_large_groups = 0
    exact_terms_upper = 0
    search_budget_reason = None
    for group_index, group in enumerate(safe_groups):
        if search_budget_reason is not None:
            break
        rows = tuple(int(row) for row in group)
        if len(rows) < 2:
            continue
        if len(rows) > 8:
            skipped_large_groups += 1
            continue
        group_best = None
        for left_position in range(len(rows)):
            if search_budget_reason is not None:
                break
            for right_position in range(left_position + 1, len(rows)):
                pairs_considered += 1
                left = int(rows[left_position])
                right = int(rows[right_position])
                left_generators = generators.getrow(left)
                right_generators = generators.getrow(right)
                pair_terms_upper = int(
                    left_generators.nnz + right_generators.nnz
                )
                if pair_terms_upper > int(exact_pair_term_cap):
                    search_budget_reason = "pair_term_cap"
                    break
                if (
                    exact_terms_upper + pair_terms_upper
                    > int(exact_total_term_cap)
                ):
                    search_budget_reason = "total_term_cap"
                    break
                if (
                    candidate_deadline is not None
                    and time.monotonic() >= float(candidate_deadline)
                ):
                    search_budget_reason = "candidate_deadline"
                    break
                exact_terms_upper += pair_terms_upper
                candidate = _dyadic_pair_cube_candidate(
                    c[left],
                    left_generators,
                    c[right],
                    right_generators,
                    denominator=denominator,
                    max_union_terms=exact_pair_term_cap,
                    deadline=candidate_deadline,
                )
                if (
                    candidate_deadline is not None
                    and time.monotonic() >= float(candidate_deadline)
                ):
                    search_budget_reason = "candidate_deadline"
                    break
                if candidate is None:
                    continue
                item = {
                    **candidate,
                    "group": int(group_index),
                    "left_row": left,
                    "right_row": right,
                }
                exact_left = int(item["_proxy_exact_numerator"])
                exact_left_denominator = int(
                    item["_proxy_exact_denominator"]
                )
                if group_best is not None:
                    exact_right = int(
                        group_best["_proxy_exact_numerator"]
                    )
                    exact_right_denominator = int(
                        group_best["_proxy_exact_denominator"]
                    )
                    exact_order = (
                        exact_left * exact_right_denominator
                        - exact_right * exact_left_denominator
                    )
                else:
                    exact_order = -1
                if (
                    group_best is None
                    or (
                        int(exact_order),
                        int(item["left_row"]),
                        int(item["right_row"]),
                        int(item["numerator"]),
                    ) < (
                        0,
                        int(group_best["left_row"]),
                        int(group_best["right_row"]),
                        int(group_best["numerator"]),
                    )
                ):
                    group_best = item
        if group_best is not None:
            selected.append(group_best)
    if search_budget_reason is not None:
        receipt.update(
            {
                "status": "exact_search_budget_exceeded_no_append",
                "denominator": int(denominator),
                "pairs_considered": int(pairs_considered),
                "skipped_large_groups": int(skipped_large_groups),
                "exact_terms_upper_processed": int(exact_terms_upper),
                "exact_search_complete": False,
                "exact_search_budget_reason": str(
                    search_budget_reason
                ),
                "exact_search_elapsed_s": float(
                    time.monotonic() - search_started
                ),
                "dyadic_convexity_validated": False,
            }
        )
        return C, t, safe_groups, receipt
    if not selected:
        receipt.update(
            {
                "status": "no_strict_proxy_improvement",
                "denominator": int(denominator),
                "pairs_considered": int(pairs_considered),
                "skipped_large_groups": int(skipped_large_groups),
                "exact_terms_upper_processed": int(exact_terms_upper),
                "exact_search_complete": bool(
                    skipped_large_groups == 0
                ),
                "exact_search_elapsed_s": float(
                    time.monotonic() - search_started
                ),
                "dyadic_convexity_validated": True,
            }
        )
        return C, t, safe_groups, receipt

    for item in selected:
        item.pop("_proxy_exact_numerator", None)
        item.pop("_proxy_exact_denominator", None)

    old_rows = int(C.shape[0])
    extra_C = np.zeros(
        (len(selected), int(C.shape[1])), dtype=np.float64
    )
    extra_t = np.zeros(len(selected), dtype=np.float64)
    augmented_groups = [tuple(group) for group in safe_groups]
    digest = hashlib.sha256()
    digest.update(
        np.asarray(
            [int(grid_bits), int(denominator), len(selected)],
            dtype=np.int64,
        ).tobytes()
    )
    for offset, item in enumerate(selected):
        numerator = int(item["numerator"])
        left_weight = float(numerator / denominator)
        right_weight = float((denominator - numerator) / denominator)
        exact_left_weight = Fraction(numerator, denominator)
        exact_right_weight = Fraction(
            denominator - numerator, denominator
        )
        if (
            exact_left_weight + exact_right_weight != 1
            or Fraction.from_float(left_weight) != exact_left_weight
            or Fraction.from_float(right_weight) != exact_right_weight
            or left_weight < 0.0
            or right_weight < 0.0
            or left_weight + right_weight != 1.0
        ):
            raise ValueError("dyadic group mixture weights are invalid")
        extra_C[offset, int(item["left_row"])] = left_weight
        extra_C[offset, int(item["right_row"])] = right_weight
        appended_row = int(old_rows + offset)
        group_index = int(item["group"])
        augmented_groups[group_index] = (
            *augmented_groups[group_index],
            appended_row,
        )
        item["appended_row"] = appended_row
        item["left_weight"] = left_weight
        item["right_weight"] = right_weight
        item["left_weight_numerator"] = int(numerator)
        item["right_weight_numerator"] = int(
            denominator - numerator
        )
        item["stored_dyadic_weights_validated"] = True
        digest.update(
            np.asarray(
                [
                    group_index,
                    int(item["left_row"]),
                    int(item["right_row"]),
                    numerator,
                    denominator,
                    appended_row,
                ],
                dtype=np.int64,
            ).tobytes()
        )
        digest.update(
            np.asarray(
                [left_weight, right_weight], dtype=np.float64
            ).tobytes()
        )
        if (
            Fraction.from_float(
                float(extra_C[offset, int(item["left_row"])])
            )
            != exact_left_weight
            or Fraction.from_float(
                float(extra_C[offset, int(item["right_row"])])
            )
            != exact_right_weight
        ):
            raise ValueError(
                "stored objective row changed a dyadic mixture weight"
            )
    augmented_C = np.vstack([C, extra_C])
    augmented_t = np.concatenate([t, extra_t])
    normalized_groups = _normalize_safe_row_groups(
        tuple(augmented_groups), int(augmented_C.shape[0])
    )
    improvements = np.asarray(
        [float(item["proxy_improvement"]) for item in selected],
        dtype=np.float64,
    )
    receipt.update(
        {
            "status": "generated",
            "denominator": int(denominator),
            "pairs_considered": int(pairs_considered),
            "skipped_large_groups": int(skipped_large_groups),
            "exact_terms_upper_processed": int(exact_terms_upper),
            "exact_search_complete": bool(
                skipped_large_groups == 0
            ),
            "exact_search_elapsed_s": float(
                time.monotonic() - search_started
            ),
            "selected_groups": int(len(selected)),
            "appended_rows": int(len(selected)),
            "original_objective_rows": int(old_rows),
            "augmented_objective_rows": int(augmented_C.shape[0]),
            "proxy_improvement_sum": float(np.sum(improvements)),
            "proxy_improvement_max": float(np.max(improvements)),
            "weights_sha256": digest.hexdigest(),
            "stored_dyadic_weights_validated": True,
            "dyadic_convexity_validated": True,
            "selected_pair_exact_grid_argmins_validated": True,
            "construction_rule": (
                "exact_identity_rows+nonnegative_power_of_two_weights_"
                "summing_exactly_one"
            ),
            "selected": [dict(item) for item in selected],
        }
    )
    return augmented_C, augmented_t, normalized_groups, receipt


def _hz_cube_row_upper_bounds(c, Gc, Gb, C, t):
    """Constraint-free, outward-guarded upper bounds for ``C[r] y - t[r]``.

    Every HZ factor lies in ``[-1, 1]`` before the equality/inequality
    constraints are considered.  Dropping those constraints therefore gives
    the sound (possibly loose) box support

        C[r] c - t[r] + ||C[r] Gc||_1 + ||C[r] Gb||_1.

    The returned value includes a conservative first-order floating-point
    accumulation guard.  Callers may use a negative upper bound to discard an
    unsafe rival without invoking an LP/MILP, but must send every remaining
    row to an exact decision procedure.
    """

    c = np.asarray(c, dtype=np.float64).reshape(-1)
    C = np.asarray(C, dtype=np.float64).reshape(-1, c.size)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if C.shape[0] != t.size:
        raise ValueError(f"cube prefilter C/t row mismatch: {C.shape} vs {t.shape}")

    abs_gc = abs(Gc) if _sp.issparse(Gc) else np.abs(np.asarray(Gc, dtype=np.float64))
    abs_gb = abs(Gb) if _sp.issparse(Gb) else np.abs(np.asarray(Gb, dtype=np.float64))
    ng = int(Gc.shape[1])
    nb = int(Gb.shape[1])
    eps = np.finfo(np.float64).eps
    # This deliberately over-counts the relevant reductions.  For the target
    # models it remains far below the proof tolerance while preventing a
    # rounded-down support value from becoming a false row prune.
    op_count = max(1, 4 * int(c.size) + ng + nb + 16)
    gamma = (op_count * eps) / max(0.5, 1.0 - op_count * eps)

    upper = np.empty(C.shape[0], dtype=np.float64)
    guards = np.empty(C.shape[0], dtype=np.float64)
    abs_c = np.abs(c)
    for r in range(C.shape[0]):
        row = C[r]
        row_abs = np.abs(row)
        gc_row = _row_dot_gen(row, Gc)
        gb_row = _row_dot_gen(row, Gb)
        center = float(row @ c) - float(t[r])
        radius = float(np.abs(gc_row).sum() + np.abs(gb_row).sum())

        # Bound the magnitude of every product accumulated by the two sparse
        # matmuls, not merely the already-rounded resulting coefficients.
        product_magnitude = float(row_abs @ abs_c) + abs(float(t[r]))
        if ng:
            product_magnitude += float(_row_dot_gen(row_abs, abs_gc).sum())
        if nb:
            product_magnitude += float(_row_dot_gen(row_abs, abs_gb).sum())
        guard = gamma * (1.0 + product_magnitude + abs(center) + radius)
        guards[r] = guard
        upper[r] = np.nextafter(center + radius + guard, np.inf)
    return upper, guards


def _hz_exact_point_margins(c, C, t):
    """Evaluate fixed-point margins exactly over stored binary64 constants."""

    c = np.asarray(c, dtype=np.float64).reshape(-1)
    C = np.asarray(C, dtype=np.float64).reshape(-1, c.size)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    margins = []
    for row, threshold in zip(C, t):
        value = -Fraction.from_float(float(threshold))
        for coefficient, coordinate in zip(row, c):
            coefficient = float(coefficient)
            coordinate = float(coordinate)
            if coefficient != 0.0 and coordinate != 0.0:
                value += (
                    Fraction.from_float(coefficient)
                    * Fraction.from_float(coordinate)
                )
        margins.append(value)
    return margins


def hz_row_max(hz, c_row: np.ndarray, *, integer: bool = False,
               time_limit: float = 20.0) -> Optional[float]:
    """max_y (c_row . y) over the HZ. LP relaxation (convex hull) or MILP."""
    if not _HAS_SCIPY:
        return None
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = hz_np_sparse(hz)
    c_row = np.asarray(c_row, dtype=np.float64).reshape(-1)
    ng, nb = Gc.shape[1], Gb.shape[1]
    obj_c = _row_dot_gen(c_row, Gc)
    obj_b = _row_dot_gen(c_row, Gb)
    obj = np.concatenate([obj_c, obj_b])  # maximize -> minimize -obj
    const = float(c_row @ c)
    if ng + nb == 0:
        return const
    if not integer:
        A_eq = _sp.hstack([Ace, Abe], format="csr") if Ace.shape[0] else None
        A_ub = _sp.hstack([Acl, Abl], format="csr") if Acl.shape[0] else None
        r = _linprog(-obj, A_eq=A_eq, b_eq=(be if Ace.shape[0] else None),
                     A_ub=(A_ub if Acl.shape[0] else None),
                     b_ub=(bl if Acl.shape[0] else None),
                     bounds=[(-1, 1)] * (ng + nb), method="highs")
        return (const - r.fun) if r.success else None
    obj_z = np.concatenate([obj_c, 2.0 * obj_b])
    const_z = const - float(obj_b.sum())
    integ = np.concatenate([np.zeros(ng), np.ones(nb)]).astype(int)
    cons = []
    if Ace.shape[0]:
        Aeq = _sp.hstack([Ace, 2.0 * Abe], format="csr")
        rhs = be + _csr_rowsum(Abe)
        cons.append(_LC(Aeq, lb=rhs, ub=rhs))
    if Acl.shape[0]:
        Ale = _sp.hstack([Acl, 2.0 * Abl], format="csr")
        cons.append(_LC(Ale, ub=bl + _csr_rowsum(Abl)))
    vlb = np.concatenate([-np.ones(ng), np.zeros(nb)])
    vub = np.ones(ng + nb)
    r = _milp(c=-obj_z, constraints=cons, integrality=integ,
              bounds=_LPB(vlb, vub),
              options={"mip_rel_gap": 1e-9, "time_limit": time_limit})
    return (const_z - r.fun) if r.success else None


def hz_joint_min_margin(hz, C: np.ndarray, t: np.ndarray, *,
                        integer: bool = False, time_limit: float = 30.0) -> Optional[float]:
    """s* = min_y max_r (C[r] y - t[r]) over the HZ (epigraph form)."""
    if not _HAS_SCIPY:
        return None
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = hz_np_sparse(hz)
    C, t = _spec_np(C, t, c.size)
    ng, nb = Gc.shape[1], Gb.shape[1]
    nrow = C.shape[0]
    v_s = ng + nb
    nv = ng + nb + 1
    epi_A = np.zeros((nrow, nv))
    epi_b = np.empty(nrow)
    for r in range(nrow):
        epi_A[r, :ng] = _row_dot_gen(C[r], Gc)
        epi_A[r, ng:ng + nb] = _row_dot_gen(C[r], Gb)
        epi_A[r, v_s] = -1.0
        epi_b[r] = float(t[r] - C[r] @ c)
    obj = np.zeros(nv)
    obj[v_s] = 1.0
    vlb = np.concatenate([-np.ones(ng + nb), [-1e12]])
    vub = np.concatenate([np.ones(ng + nb), [1e12]])
    if not integer:
        A_ub = [_sp.csr_matrix(epi_A)]
        b_ub = [epi_b]
        if Acl.shape[0]:
            A_ub.append(_sp.hstack(
                [Acl, Abl, _sp.csr_matrix((Acl.shape[0], 1))],
                format="csr"))
            b_ub.append(bl)
        A_eq = (_sp.hstack(
                    [Ace, Abe, _sp.csr_matrix((Ace.shape[0], 1))],
                    format="csr")
                if Ace.shape[0] else None)
        r = _linprog(obj, A_ub=_sp.vstack(A_ub, format="csr"),
                     b_ub=np.concatenate(b_ub),
                     A_eq=A_eq, b_eq=(be if Ace.shape[0] else None),
                     bounds=list(zip(vlb, vub)), method="highs")
        return float(r.fun) if r.success else None
    epi_Az = epi_A.copy()
    epi_Az[:, ng:ng + nb] *= 2.0
    epi_bz = epi_b + _mat_dot_gen(C, Gb).sum(axis=1)
    integ = np.concatenate([np.zeros(ng), np.ones(nb), [0]]).astype(int)
    vlbz = np.concatenate([-np.ones(ng), np.zeros(nb), [-1e12]])
    vubz = np.concatenate([np.ones(ng + nb), [1e12]])
    cons = [_LC(_sp.csr_matrix(epi_Az), ub=epi_bz)]
    if Ace.shape[0]:
        Meq = _sp.hstack(
            [Ace, 2.0 * Abe, _sp.csr_matrix((Ace.shape[0], 1))],
            format="csr")
        beq = be + _csr_rowsum(Abe)
        cons.append(_LC(Meq, lb=beq, ub=beq))
    if Acl.shape[0]:
        Mle = _sp.hstack(
            [Acl, 2.0 * Abl, _sp.csr_matrix((Acl.shape[0], 1))],
            format="csr")
        cons.append(_LC(Mle, ub=bl + _csr_rowsum(Abl)))
    r = _milp(c=obj, constraints=cons, integrality=integ, bounds=_LPB(vlbz, vubz),
              options={"mip_rel_gap": 1e-9, "time_limit": time_limit})
    return float(r.fun) if r.success else None


def _objbound_solve_highs(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                          mip_start_xi=None):
    """HiGHS minimize cost@v with early-stop at obj_thr (objective_target +
    objective_bound), or optionally solve the equivalent cutoff-row feasibility
    MILP when ``HZ_MILP_CUTOFF_ROW`` is set. mip_rel_gap=1e-9 so the optimum is
    exact; only the STOPPING is early. Returns (kind, xi) where kind in
    {'witness','empty','unknown'}:
                  point reaches obj_thr (the cutoff side is provably empty);
    xi maps the integer (z in {0,1}) columns back to {-1,+1}; continuous pass through.
    """
    # The caller's budget covers Python-side validation, matrix preparation,
    # native model loading, and the solve.  Giving the original duration to
    # ``Highs.run`` after loading a multi-million-nnz model can overrun the
    # request deadline by the entire load time.
    total_solve_budget = max(0.0, float(time_limit))
    # Native presolve/termination can return slightly after its configured
    # limit.  Keep one bounded tail inside this function's own budget so the
    # caller receives UNKNOWN and diagnostics instead of being hard-killed.
    native_return_reserve = min(
        2.0,
        0.25 * total_solve_budget,
        max(0.05, 0.10 * total_solve_budget),
    )
    solve_deadline = (
        time.monotonic()
        + max(0.0, total_solve_budget - native_return_reserve)
    )

    # Strict verdicts use cutoff feasibility, not a cancellation-prone
    # objective comparison.  A historical objective-bound formulation could
    # report false EMPTY on ill-conditioned rows.  Keep that formulation out
    # of the proof path until it has an independently validated dual
    # certificate.
    cutoff_row = True
    cost = np.asarray(cost, dtype=np.float64).reshape(-1)
    A = _sp.csr_matrix(A, dtype=np.float64)
    A.sum_duplicates()
    A.sort_indices()
    rl = np.asarray(rl, dtype=np.float64).reshape(-1)
    ru = np.asarray(ru, dtype=np.float64).reshape(-1)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    integ_mask = np.asarray(integ_mask, dtype=bool).reshape(-1)
    if (
        A.shape != (rl.size, cost.size)
        or ru.size != rl.size
        or lb.size != cost.size
        or ub.size != cost.size
        or integ_mask.size != cost.size
    ):
        raise ValueError("HiGHS cutoff model shape mismatch")
    if (
        (A.nnz and not np.all(np.isfinite(A.data)))
        or not np.all(np.isfinite(cost))
        or not np.isfinite(float(obj_thr))
        or not np.isfinite(float(time_limit))
        or not np.all(np.isfinite(lb))
        or not np.all(np.isfinite(ub))
        or np.any(np.isnan(rl))
        or np.any(np.isnan(ru))
        or np.any(lb > ub)
    ):
        raise ValueError("HiGHS cutoff model contains invalid numerical data")
    solve_cost = np.zeros_like(cost) if cutoff_row else cost
    if cutoff_row:
        cut = _sp.csr_matrix(np.asarray(cost, float).reshape(1, -1))
        A = _sp.vstack([_sp.csr_matrix(A), cut], format="csr")
        A.sum_duplicates()
        A.sort_indices()
        rl = np.concatenate([np.asarray(rl, float), [-np.inf]])
        ru = np.concatenate([np.asarray(ru, float), [float(obj_thr) + 1e-9]])

    # Preserve the unscaled, unprojected feasibility system for independent
    # incumbent validation.  Solver presolve/scaling coordinates are never
    # trusted directly as an UNSAFE witness.
    validation_A = _sp.csr_matrix(A).copy()
    validation_rl = np.asarray(rl, dtype=np.float64).copy()
    validation_ru = np.asarray(ru, dtype=np.float64).copy()
    validation_lb = np.asarray(lb, dtype=np.float64).copy()
    validation_ub = np.asarray(ub, dtype=np.float64).copy()

    orig_integ_mask = np.asarray(integ_mask, dtype=bool)
    current_integ_mask = orig_integ_mask
    elim_metas = []
    if _env_flag("HZ_MILP_EQ_SUBST"):
        A, rl, ru, cost, lb, ub, eq_meta = _substitute_eq_singleton_continuous_cols(
            A, rl, ru, cost, lb, ub, current_integ_mask
        )
        if eq_meta is not None:
            current_integ_mask = current_integ_mask[np.asarray(eq_meta["keep_cols"], dtype=np.int64)]
            elim_metas.append(eq_meta)

    if _env_flag("HZ_MILP_ELIM_SINGLETONS"):
        A, rl, ru, cost, lb, ub, singleton_meta = _project_singleton_continuous_rows(
            A, rl, ru, cost, lb, ub, current_integ_mask
        )
        if singleton_meta is not None:
            current_integ_mask = current_integ_mask[np.asarray(singleton_meta["keep_cols"], dtype=np.int64)]
            elim_metas.append(singleton_meta)
    elim_meta = _chain_elim_meta(*elim_metas)
    integ_mask = current_integ_mask

    solve_cost = np.zeros_like(cost) if cutoff_row else cost
    if cutoff_row or _env_flag("HZ_MILP_SCALE"):
        A, rl, ru, row_scale = _scale_milp_rows(A, rl, ru)
        obj_scale = 1.0
        if not cutoff_row:
            cost, obj_thr, obj_scale = _scale_milp_objective(cost, obj_thr)
            solve_cost = cost
    A = _sp.csr_matrix(A, dtype=np.float64)
    A.sum_duplicates()
    A.sort_indices()

    h = _highspy.Highs()
    HS = _highspy.HighsStatus

    def _require_ok(status, operation):
        if status != HS.kOk:
            raise RuntimeError(f"HiGHS {operation} returned {status}")

    _require_ok(h.setOptionValue("output_flag", False), "set output_flag")
    _require_ok(
        h.setOptionValue("time_limit", float(time_limit)),
        "set time_limit",
    )
    _require_ok(h.setOptionValue("mip_rel_gap", 1e-9), "set mip_rel_gap")
    # HiGHS otherwise drops every matrix coefficient with magnitude <= 1e-9.
    # Learned networks contain many legitimate coefficients below that
    # default, so retain everything down to the smallest value accepted by
    # HiGHS.  The remaining coefficients at/below 1e-12 are removed
    # explicitly from the candidate copy below; the original matrix remains
    # the witness-validation authority.
    _require_ok(
        h.setOptionValue("small_matrix_value", 1e-12),
        "set small_matrix_value",
    )
    if not cutoff_row:
        _require_ok(
            h.setOptionValue("objective_target", float(obj_thr)),
            "set objective_target",
        )
        _require_ok(
            h.setOptionValue("objective_bound", float(obj_thr)),
            "set objective_bound",
        )
    _require_ok(
        h.setOptionValue("threads", _highs_process_threads()),
        "set threads",
    )
    _heff = os.environ.get("HZ_MILP_HEURISTIC")
    if _heff:
        _require_ok(
            h.setOptionValue("mip_heuristic_effort", float(_heff)),
            "set mip_heuristic_effort",
        )
    _apply_highs_env_options(h)
    nc = len(cost)
    _require_ok(
        h.addCols(
            nc,
            np.asarray(solve_cost, float),
            np.asarray(lb, float),
            np.asarray(ub, float),
            0,
            np.array([], np.int32),
            np.array([], np.int32),
            np.array([], float),
        ),
        "add columns",
    )
    vt = np.array([_highspy.HighsVarType.kInteger if m else _highspy.HighsVarType.kContinuous
                   for m in integ_mask])
    _require_ok(
        h.changeColsIntegrality(nc, np.arange(nc, dtype=np.int32), vt),
        "change integrality",
    )
    if A.shape[0]:
        As, _candidate_matrix_stats = _highs_candidate_csr(
            A,
            small_matrix_value=1e-12,
        )
        _require_ok(
            h.addRows(
            As.shape[0],
            np.asarray(rl, float),
            np.asarray(ru, float),
            As.nnz,
            As.indptr.astype(np.int32),
            As.indices.astype(np.int32),
            As.data.astype(float),
            ),
            "add rows",
        )
        if (
            int(h.getNumRow()) != int(As.shape[0])
            or int(h.getNumCol()) != int(As.shape[1])
            or int(h.getNumNz()) != int(As.nnz)
        ):
            raise RuntimeError(
                "HiGHS cutoff candidate matrix postcondition failed"
            )
    if mip_start_xi is not None:
        try:
            raw = np.asarray(mip_start_xi, dtype=np.float64).reshape(-1)
            start_full = np.zeros(orig_integ_mask.size, dtype=np.float64)
            ncopy = min(raw.size, start_full.size)
            start_full[:ncopy] = np.clip(raw[:ncopy], -1.0, 1.0)
            start_full[orig_integ_mask] = (start_full[orig_integ_mask] >= 0.0).astype(np.float64)
            start = _project_solution_to_reduced(start_full, elim_meta)
            int_idx = np.flatnonzero(np.asarray(integ_mask, dtype=bool)).astype(np.int32)
            if int_idx.size:
                h.setSolution(
                    int_idx.size,
                    int_idx,
                    np.clip(start[int_idx], lb[int_idx], ub[int_idx]).astype(np.float64),
                )
        except Exception:
            pass
    run_time_limit = solve_deadline - time.monotonic()
    if run_time_limit <= 0.0:
        return "unknown", None
    _require_ok(
        h.setOptionValue("time_limit", float(run_time_limit)),
        "refresh time_limit after model load",
    )
    run_status = h.run()
    MS = _highspy.HighsModelStatus
    st = h.getModelStatus()

    def _validated_xi_from_reduced(v_reduced):
        """Validate a solver incumbent in the original cutoff system."""

        try:
            reduced = np.asarray(v_reduced, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if reduced.size != nc or not np.all(np.isfinite(reduced)):
            return None
        try:
            v = _expand_projected_solution(reduced, elim_meta)
        except Exception:
            return None
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.size != orig_integ_mask.size or not np.all(np.isfinite(v)):
            return None

        im = np.asarray(orig_integ_mask, dtype=bool)
        if im.any():
            ints = v[im]
            rounded = np.rint(ints)
            if ints.size and float(np.max(np.abs(ints - rounded))) > 1e-7:
                return None
            v[im] = np.clip(rounded, 0.0, 1.0)
        if v.size:
            if float(
                np.max(np.maximum(validation_lb - v, 0.0))
            ) > 1e-7:
                return None
            if float(
                np.max(np.maximum(v - validation_ub, 0.0))
            ) > 1e-7:
                return None
        if validation_A.shape[0]:
            av = np.asarray(validation_A @ v, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(av)):
                return None
            lower = np.where(
                np.isfinite(validation_rl),
                validation_rl - av,
                -np.inf,
            )
            upper = np.where(
                np.isfinite(validation_ru),
                av - validation_ru,
                -np.inf,
            )
            vio = np.maximum(np.maximum(lower, upper), 0.0)
            if not np.all(np.isfinite(vio)):
                return None
            row_vio = float(np.max(vio)) if vio.size else 0.0
            scale = 1.0 + np.maximum(
                np.abs(av),
                np.maximum(
                    np.where(
                        np.isfinite(validation_rl),
                        np.abs(validation_rl),
                        0.0,
                    ),
                    np.where(
                        np.isfinite(validation_ru),
                        np.abs(validation_ru),
                        0.0,
                    ),
                ),
            )
            row_vio_scaled = float(np.max(vio / scale)) if vio.size else 0.0
            if row_vio > 5e-7 or row_vio_scaled > 5e-9:
                return None
        return np.array(
            [
                (2.0 * v[i] - 1.0) if orig_integ_mask[i] else v[i]
                for i in range(orig_integ_mask.size)
            ],
            dtype=np.float64,
        )

    def _validated_solver_incumbent():
        try:
            raw = np.asarray(h.getSolution().col_value, dtype=np.float64)
        except Exception:
            return None
        return _validated_xi_from_reduced(raw)

    if run_status != HS.kOk:
        return "unknown", None
    if st == MS.kObjectiveTarget:
        xi = _validated_solver_incumbent()
        return ("witness", xi) if xi is not None else ("unknown", None)
    if st in (MS.kObjectiveBound, MS.kInfeasible):
        # A floating-point solver status is not by itself a checkable
        # infeasibility certificate.  In particular, HiGHS can report
        # kInfeasible for a stored-binary64 feasible cutoff system after
        # numerical presolve/row scaling.  Until a Farkas/MIP certificate is
        # independently validated, this status has no SAFE authority.
        return "unknown", None
    if st == MS.kOptimal:
        if cutoff_row:
            xi = _validated_solver_incumbent()
            return ("witness", xi) if xi is not None else ("unknown", None)
        obj = h.getInfo().objective_function_value
        if obj <= obj_thr + 1e-9:
            xi = _validated_solver_incumbent()
            return ("witness", xi) if xi is not None else ("unknown", None)
        return "empty", None
    if (
        not cutoff_row
        and run_status == _highspy.HighsStatus.kOk
        and h.modelStatusToString(st) != "Not Set"
    ):
        try:
            dual_bound = float(h.getInfo().mip_dual_bound)
        except Exception:
            dual_bound = float("nan")
        if np.isfinite(dual_bound) and dual_bound > float(obj_thr) + 1e-7:
            return "empty", None
    xi_inc = _validated_solver_incumbent()
    if xi_inc is not None:
        return "witness", xi_inc
    return "unknown", None   # kTimeLimit / other -> undecided (never a false CERT)


def _objbound_solve_scip(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                         mip_start_xi=None):
    """SCIP cutoff-feasibility exact backend.

    This solves the same exact MILP as the HiGHS path, but as a feasibility
    query with the cutoff row ``cost @ v <= obj_thr``. Feasible => unsafe
    witness; infeasible => safe side proven; timeout/other => unknown. This is
    optional because SCIP build/solve overhead can dominate small instances.
    """
    if not (_HAS_PYSCIPOPT and _HAS_SCIPY):
        return "unknown", None
    A = _sp.csr_matrix(A)
    cost = np.asarray(cost, dtype=np.float64).reshape(-1)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    rl = np.asarray(rl, dtype=np.float64).reshape(-1)
    ru = np.asarray(ru, dtype=np.float64).reshape(-1)
    integ_mask = np.asarray(integ_mask, dtype=bool).reshape(-1)
    n = int(cost.size)
    if n == 0:
        return ("witness", np.zeros(0, dtype=np.float64)) if 0.0 <= obj_thr + 1e-9 else ("empty", None)

    m = _SCIPModel()
    m.hideOutput()
    try:
        m.setParam("limits/time", float(time_limit))
        m.setParam("numerics/feastol", 1e-7)
    except Exception:
        pass

    def _finite_bound(x, sign):
        x = float(x)
        if np.isneginf(x):
            return -1e20
        if np.isposinf(x):
            return 1e20
        return max(-1e20, min(1e20, x if np.isfinite(x) else sign * 1e20))

    V = []
    for i in range(n):
        vtype = "B" if integ_mask[i] else "C"
        V.append(m.addVar(
            lb=_finite_bound(lb[i], -1.0),
            ub=_finite_bound(ub[i], 1.0),
            vtype=vtype,
            name=f"v{i}",
        ))

    for r in range(A.shape[0]):
        s, e = A.indptr[r], A.indptr[r + 1]
        if s == e:
            row_expr = 0.0
        else:
            row_expr = _scip_quicksum(float(A.data[p]) * V[int(A.indices[p])]
                                      for p in range(s, e))
        lo = float(rl[r])
        hi = float(ru[r])
        lo_fin = np.isfinite(lo) and lo > -1e19
        hi_fin = np.isfinite(hi) and hi < 1e19
        if lo_fin and hi_fin and abs(lo - hi) <= 1e-12:
            m.addCons(row_expr == lo)
        else:
            if lo_fin:
                m.addCons(row_expr >= lo)
            if hi_fin:
                m.addCons(row_expr <= hi)

    nz = np.flatnonzero(np.abs(cost) > 0.0)
    if nz.size:
        cut_expr = _scip_quicksum(float(cost[i]) * V[int(i)] for i in nz)
        m.addCons(cut_expr <= float(obj_thr) + 1e-9)
    elif 0.0 > float(obj_thr) + 1e-9:
        return "empty", None

    m.setObjective(0.0)
    m.optimize()
    status = str(m.getStatus()).lower()
    has_sol = False
    try:
        has_sol = int(m.getNSols()) > 0
    except Exception:
        has_sol = False
    if status in {"optimal", "feasible", "bestsollimit"} or has_sol:
        vals = np.asarray([float(m.getVal(v)) for v in V], dtype=np.float64)
        return "witness", np.array([(2.0 * vals[i] - 1.0) if integ_mask[i] else vals[i]
                                    for i in range(n)])
    if status == "infeasible":
        # As for HiGHS, a SCIP status string is not an independently checked
        # exact infeasibility certificate over the stored binary64 model.
        return "unknown", None
    return "unknown", None


def _objbound_solve(cost, obj_thr, A, rl, ru, lb, ub, integ_mask, time_limit,
                    mip_start_xi=None, deadline=None):
    def _remaining():
        budget = max(0.0, float(time_limit))
        if deadline is not None:
            budget = min(budget, max(0.0, float(deadline) - time.monotonic()))
        return budget

    first_limit = _remaining()
    if first_limit <= 0.0:
        return ("unknown", None)
    backend = os.environ.get("HZ_MILP_BACKEND", "highs").strip().lower()
    if backend == "scip":
        try:
            out = _objbound_solve_scip(
                cost,
                obj_thr,
                A,
                rl,
                ru,
                lb,
                ub,
                integ_mask,
                first_limit,
                mip_start_xi=mip_start_xi,
            )
        except Exception:
            logger.exception("HybridZ SCIP cutoff query failed closed")
            out = ("unknown", None)
        if deadline is not None and time.monotonic() >= deadline:
            return ("unknown", None)
        return out
    try:
        out = _objbound_solve_highs(
            cost,
            obj_thr,
            A,
            rl,
            ru,
            lb,
            ub,
            integ_mask,
            first_limit,
            mip_start_xi=mip_start_xi,
        )
    except Exception:
        logger.exception("HybridZ HiGHS cutoff query failed closed")
        out = ("unknown", None)
    if backend in {"highs_scip", "portfolio"} and out[0] == "unknown":
        fallback_limit = _remaining()
        if fallback_limit <= 0.0:
            return ("unknown", None)
        out = _objbound_solve_scip(
            cost,
            obj_thr,
            A,
            rl,
            ru,
            lb,
            ub,
            integ_mask,
            fallback_limit,
            mip_start_xi=mip_start_xi,
        )
    if deadline is not None and time.monotonic() >= deadline:
        return ("unknown", None)
    return out


def _binary_shift_rhs_exact_or_outward(base, binary_matrix, *,
                                       equality: bool) -> np.ndarray:
    """Map ``xi_b = 2 z - 1`` without silently shrinking the HZ.

    A stored-float row ``A_b xi_b <= b`` becomes

        ``2 A_b z <= b + sum(A_b)``.

    The right-hand sum is a dyadic rational but need not itself be exactly
    representable as binary64.  For an upper inequality we round it toward
    ``+inf``, which only enlarges the feasible set.  An equality cannot be
    rounded in either direction without changing the set, so a
    non-representable result fails closed.
    """

    B = _sp.csr_matrix(binary_matrix, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    if B.shape[0] != base.size:
        raise ValueError(
            f"binary RHS row mismatch: matrix={B.shape}, rhs={base.shape}"
        )
    if not np.all(np.isfinite(base)) or (
        B.nnz and not np.all(np.isfinite(B.data))
    ):
        raise ValueError("binary RHS transform received non-finite data")

    out = np.empty(base.size, dtype=np.float64)
    row_nnz = np.diff(B.indptr)

    # Exact-ReLU rows contain zero or one binary coefficient.  The old path
    # constructed several Fraction objects for every such row even though an
    # IEEE-754 TwoSum gives an error-free expansion of the two stored values:
    #
    #     exact(base + coefficient) == rounded + residual.
    #
    # A positive residual means the nearest rounded value lies below the
    # exact dyadic sum, so one nextafter toward +inf is the directed upper
    # result.  Equality accepts only a zero residual.  This vectorized common
    # path allocates no second constraint representation and leaves rows with
    # multiple binary terms on the established Fraction audit below.
    zero_rows = row_nnz == 0
    out[zero_rows] = base[zero_rows]
    out[zero_rows & (base == 0.0)] = 0.0
    completed = zero_rows.copy()
    one_rows = np.flatnonzero(row_nnz == 1).astype(
        np.int64, copy=False
    )
    if one_rows.size:
        left = base[one_rows]
        right = B.data[B.indptr[one_rows]]
        # Do not inherit a caller's NumPy warning policy: overflow is checked
        # explicitly below and must fail through this function's ValueError
        # contract rather than leak FloatingPointError.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            rounded = left + right
            bridge = rounded - left
            residual = (left - (rounded - bridge)) + (right - bridge)
        finite = np.isfinite(rounded)
        if not np.all(finite):
            row = int(one_rows[np.flatnonzero(~finite)[0]])
            raise ValueError(f"binary RHS row {row} overflows binary64")
        if equality:
            inexact = residual != 0.0
            if np.any(inexact):
                row = int(one_rows[np.flatnonzero(inexact)[0]])
                raise ValueError(
                    "binary equality RHS is not exactly representable at "
                    f"row {row}"
                )
        else:
            inward = residual > 0.0
            if np.any(inward):
                rounded[inward] = np.nextafter(
                    rounded[inward], np.inf
                )
                finite = np.isfinite(rounded)
                if not np.all(finite):
                    row = int(one_rows[np.flatnonzero(~finite)[0]])
                    raise ValueError(
                        "cannot outward-round binary inequality RHS row "
                        f"{row}"
                    )
        # Fraction(0) converts back to positive zero.  Preserve those exact
        # established bits even when both stored inputs were negative zero.
        rounded[(rounded == 0.0) & (residual == 0.0)] = 0.0
        out[one_rows] = rounded
        completed[one_rows] = True

    for row in np.flatnonzero(~completed):
        row = int(row)
        exact = Fraction.from_float(float(base[row]))
        start, end = int(B.indptr[row]), int(B.indptr[row + 1])
        for value in B.data[start:end]:
            exact += Fraction.from_float(float(value))

        try:
            rounded = float(exact)
        except OverflowError as exc:
            raise ValueError(
                f"binary RHS row {row} overflows binary64"
            ) from exc
        if not np.isfinite(rounded):
            raise ValueError(f"binary RHS row {row} is not finite")

        rounded_exact = Fraction.from_float(rounded)
        if equality:
            if rounded_exact != exact:
                raise ValueError(
                    "binary equality RHS is not exactly representable at "
                    f"row {row}"
                )
        elif rounded_exact < exact:
            rounded = float(np.nextafter(rounded, np.inf))
            if (
                not np.isfinite(rounded)
                or Fraction.from_float(rounded) < exact
            ):
                raise ValueError(
                    f"cannot outward-round binary inequality RHS row {row}"
                )
        out[row] = rounded
    return out


def _tagged_upper_band_compaction_plan(Acl, Abl, upper_row_tags):
    """Return one exact RANGE plan for tagged forward/reverse row blocks.

    Operator affine-chain cuts and stable-active ReLU equalities publish one
    contiguous forward block followed by its bitwise coefficient negation.
    HiGHS accepts their conjunction as one ranged row.  Tags only locate the
    common block; every continuous and binary coefficient is independently
    checked before any row is removed.
    """

    if upper_row_tags is None:
        return None
    if type(upper_row_tags) is not tuple:
        raise ValueError("upper-row tags must be an exact tuple")
    row_count = int(Acl.shape[0])
    if len(upper_row_tags) != row_count:
        raise ValueError("upper-row tag count does not match source rows")
    if any(type(tag) is not str or not tag for tag in upper_row_tags):
        raise ValueError("upper-row tags must be nonempty exact strings")

    Acl = _sp.csr_matrix(Acl, dtype=np.float64)
    Abl = _sp.csr_matrix(Abl, dtype=np.float64)
    if Acl.shape[0] != Abl.shape[0]:
        raise ValueError("continuous/binary upper-row counts differ")

    sign_mask = np.uint64(0x8000000000000000)

    def _block_is_bitwise_negative(matrix, fs, fe, rs, re):
        if fe - fs != re - rs:
            return False
        f_counts = np.diff(matrix.indptr[fs : fe + 1])
        r_counts = np.diff(matrix.indptr[rs : re + 1])
        if not np.array_equal(f_counts, r_counts):
            return False
        f0, f1 = int(matrix.indptr[fs]), int(matrix.indptr[fe])
        r0, r1 = int(matrix.indptr[rs]), int(matrix.indptr[re])
        if not np.array_equal(
            matrix.indices[f0:f1], matrix.indices[r0:r1]
        ):
            return False
        return np.array_equal(
            np.bitwise_xor(matrix.data[f0:f1].view(np.uint64), sign_mask),
            matrix.data[r0:r1].view(np.uint64),
        )

    keep = np.ones(row_count, dtype=bool)
    forward_rows = []
    reverse_rows = []
    compacted_tags = []
    pair_count = 0
    paired_prefixes = ("affine_chain_cut:", "relu_active:")
    row = 0
    while row < row_count:
        tag = upper_row_tags[row]
        stop = row + 1
        while stop < row_count and upper_row_tags[stop] == tag:
            stop += 1
        is_forward = (
            tag.startswith(paired_prefixes)
            and tag.endswith(":forward")
        )
        is_reverse = (
            tag.startswith(paired_prefixes)
            and tag.endswith(":reverse")
        )
        if is_reverse:
            raise ValueError("orphan paired upper-band reverse block")
        if not is_forward:
            compacted_tags.extend(upper_row_tags[row:stop])
            row = stop
            continue

        reverse_tag = tag[: -len("forward")] + "reverse"
        reverse_stop = stop
        while (
            reverse_stop < row_count
            and upper_row_tags[reverse_stop] == reverse_tag
        ):
            reverse_stop += 1
        count = stop - row
        if reverse_stop - stop != count:
            raise ValueError("paired upper-band block size mismatch")
        if not _block_is_bitwise_negative(
            Acl, row, stop, stop, reverse_stop
        ) or not _block_is_bitwise_negative(
            Abl, row, stop, stop, reverse_stop
        ):
            raise ValueError(
                "paired upper-band rows are not a bitwise coefficient negation"
            )

        keep[stop:reverse_stop] = False
        forward_rows.extend(range(row, stop))
        reverse_rows.extend(range(stop, reverse_stop))
        compacted_tags.extend(
            [tag[: -len("forward")] + "range"] * count
        )
        pair_count += count
        row = reverse_stop

    if pair_count == 0:
        return None
    keep_rows = np.flatnonzero(keep).astype(np.int64, copy=False)
    old_to_new = np.full(row_count, -1, dtype=np.int64)
    old_to_new[keep_rows] = np.arange(keep_rows.size, dtype=np.int64)
    lower_positions = old_to_new[
        np.asarray(forward_rows, dtype=np.int64)
    ]
    if np.any(lower_positions < 0):
        raise ValueError("paired upper-band forward row was removed")
    return {
        "keep_rows": keep_rows,
        "lower_positions": lower_positions,
        "reverse_rows": np.asarray(reverse_rows, dtype=np.int64),
        "compacted_tags": tuple(compacted_tags),
        "pair_count": int(pair_count),
        "source_rows": int(row_count),
    }


def _base_milp_matrices_from_blocks(
    Gc,
    Gb,
    Ace,
    Abe,
    be,
    Acl,
    Abl,
    bl,
    *,
    upper_compaction_plan=None,
):
    ng, nb = int(Gc.shape[1]), int(Gb.shape[1])
    rows_A, rl, ru = [], [], []
    if Ace.shape[0]:
        doubled = 2.0 * _sp.csr_matrix(Abe, dtype=np.float64)
        if doubled.nnz and not np.all(np.isfinite(doubled.data)):
            raise ValueError("binary equality coefficients overflowed in xi-to-z map")
        rows_A.append(_sp.hstack([Ace, doubled], format="csr"))
        rhs = _binary_shift_rhs_exact_or_outward(
            be,
            Abe,
            equality=True,
        )
        rl.append(rhs)
        ru.append(rhs)
    if Acl.shape[0]:
        rhs_full = _binary_shift_rhs_exact_or_outward(
            bl,
            Abl,
            equality=False,
        )
        if upper_compaction_plan is None:
            compact_Acl = Acl
            compact_Abl = Abl
            lower = np.full(Acl.shape[0], -np.inf)
            rhs = rhs_full
        else:
            keep_rows = np.asarray(
                upper_compaction_plan["keep_rows"], dtype=np.int64
            ).reshape(-1)
            lower_positions = np.asarray(
                upper_compaction_plan["lower_positions"], dtype=np.int64
            ).reshape(-1)
            reverse_rows = np.asarray(
                upper_compaction_plan["reverse_rows"], dtype=np.int64
            ).reshape(-1)
            if (
                int(upper_compaction_plan.get("source_rows", -1))
                != int(Acl.shape[0])
                or keep_rows.size == 0
                or lower_positions.size != reverse_rows.size
                or np.any(keep_rows < 0)
                or np.any(keep_rows >= Acl.shape[0])
                or np.any(keep_rows[1:] <= keep_rows[:-1])
                or np.any(lower_positions < 0)
                or np.any(lower_positions >= keep_rows.size)
                or np.any(reverse_rows < 0)
                or np.any(reverse_rows >= Acl.shape[0])
            ):
                raise ValueError("upper-row compaction plan is malformed")
            compact_Acl = _sp.csr_matrix(Acl, dtype=np.float64)[
                keep_rows, :
            ].tocsr()
            compact_Abl = _sp.csr_matrix(Abl, dtype=np.float64)[
                keep_rows, :
            ].tocsr()
            rhs = rhs_full[keep_rows]
            lower = np.full(keep_rows.size, -np.inf)
            lower[lower_positions] = -rhs_full[reverse_rows]
            if np.any(lower > rhs):
                raise ValueError("compacted upper-row band is contradictory")
        doubled = 2.0 * _sp.csr_matrix(compact_Abl, dtype=np.float64)
        if doubled.nnz and not np.all(np.isfinite(doubled.data)):
            raise ValueError("binary inequality coefficients overflowed in xi-to-z map")
        rows_A.append(_sp.hstack([compact_Acl, doubled], format="csr"))
        rl.append(lower)
        ru.append(rhs)
    A = (_sp.vstack(rows_A, format="csr") if rows_A
         else _sp.csr_matrix((0, ng + nb), dtype=np.float64))
    rl = np.concatenate(rl) if rl else np.zeros(0, dtype=np.float64)
    ru = np.concatenate(ru) if ru else np.zeros(0, dtype=np.float64)
    lb = np.concatenate([-np.ones(ng), np.zeros(nb)]).astype(np.float64)
    ub = np.ones(ng + nb, dtype=np.float64)
    integ = np.concatenate([np.zeros(ng), np.ones(nb)]).astype(int)
    return A, rl, ru, lb, ub, integ


def _base_milp_matrices(hz):
    _, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = hz_np_sparse(hz)
    return _base_milp_matrices_from_blocks(Gc, Gb, Ace, Abe, be, Acl, Abl, bl)


def _base_solution_to_xi(sol, integ) -> np.ndarray:
    sol = np.asarray(sol, dtype=np.float64).reshape(-1)
    integ = np.asarray(integ, dtype=bool).reshape(-1)
    xi = sol.copy()
    if integ.any():
        xi[integ] = 2.0 * xi[integ] - 1.0
    return xi


def _exact_stored_float_feasible_candidate(
    A,
    rl,
    ru,
    lb,
    ub,
    integ,
    candidate,
):
    """Validate one base-MILP point in exact stored-binary64 arithmetic.

    All finite binary64 inputs are interpreted as their exact dyadic rational
    values.  There is deliberately no primal, row, scaled, or integrality
    tolerance.  Integer columns are rounded to a concrete 0/1 assignment and
    that resulting point is then checked from scratch; this is sound because
    the rounded point itself, rather than the solver incumbent, is the
    existence witness.

    Exact Fraction accumulation is intentionally budgeted.  A model beyond
    the audit budget returns no authority (and therefore UNKNOWN), never an
    approximate FEASIBLE result.
    """

    A = _sp.csr_matrix(A, dtype=np.float64)
    rl = np.asarray(rl, dtype=np.float64).reshape(-1)
    ru = np.asarray(ru, dtype=np.float64).reshape(-1)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    integ = np.asarray(integ, dtype=bool).reshape(-1)
    try:
        v = np.asarray(candidate, dtype=np.float64).reshape(-1).copy()
    except Exception:
        return None, "candidate_conversion"

    nrow, ncol = A.shape
    if (
        v.size != ncol
        or lb.size != ncol
        or ub.size != ncol
        or integ.size != ncol
        or rl.size != nrow
        or ru.size != nrow
    ):
        return None, "shape_mismatch"
    if (
        not np.all(np.isfinite(v))
        or not np.all(np.isfinite(lb))
        or not np.all(np.isfinite(ub))
        or (A.nnz and not np.all(np.isfinite(A.data)))
        or np.any(np.isnan(rl))
        or np.any(np.isnan(ru))
    ):
        return None, "nonfinite"

    if np.any(integ):
        rounded = np.rint(v[integ])
        if np.any((rounded != 0.0) & (rounded != 1.0)):
            return None, "integrality"
        v[integ] = rounded

    # Stored-float ordering is exact ordering of the corresponding dyadics.
    if np.any(v < lb) or np.any(v > ub):
        return None, "bounds"

    try:
        max_terms = int(os.environ.get(
            "HZ_EXACT_BASE_WITNESS_MAX_TERMS",
            "250000",
        ))
    except ValueError:
        max_terms = 250000
    max_terms = max(0, max_terms)
    term_count = int(A.nnz) + int(ncol) + 2 * int(nrow)
    if ncol and term_count > max_terms:
        return None, f"exact_budget:{term_count}>{max_terms}"

    try:
        vf = [Fraction.from_float(float(value)) for value in v]
        for row in range(nrow):
            total = Fraction(0)
            start, end = int(A.indptr[row]), int(A.indptr[row + 1])
            for pos in range(start, end):
                total += (
                    Fraction.from_float(float(A.data[pos]))
                    * vf[int(A.indices[pos])]
                )
            if np.isfinite(rl[row]):
                if total < Fraction.from_float(float(rl[row])):
                    return None, f"row_{row}_lower"
            if np.isfinite(ru[row]):
                if total > Fraction.from_float(float(ru[row])):
                    return None, f"row_{row}_upper"
    except (OverflowError, ValueError, ZeroDivisionError):
        return None, "exact_arithmetic"
    return v, "exact"


def hz_base_feasibility(hz, *, time_limit: float = 10.0):
    """Return ``(status, msg)`` for the propagated HZ state itself.

    ``status`` is one of ``FEASIBLE``, ``INFEASIBLE``, or ``UNKNOWN``.  A SAFE
    verdict over ``HZ ∩ unsafe = empty`` is meaningful only if the base HZ is
    nonempty; otherwise the proof is vacuous.  Binary HZ variables are checked
    exactly as integer ``z in {0,1}`` after the standard ``xi_b = 2z - 1`` map.
    """
    cached = getattr(hz, "_solver_base_feas_cache", None)
    if cached is not None and (
        cached[0] != "FEASIBLE"
        or bool(getattr(hz, "_solver_base_feas_exact", False))
    ):
        return cached

    def _finish(out, *, exact_feasible: bool = False):
        if out[0] != "UNKNOWN":
            setattr(hz, "_solver_base_feas_cache", out)
            if out[0] == "FEASIBLE":
                setattr(
                    hz,
                    "_solver_base_feas_exact",
                    bool(exact_feasible),
                )
        return out

    def _set_witness(sol, integ):
        setattr(hz, "_solver_base_witness_cache", _base_solution_to_xi(sol, integ))

    if hz_constructively_nonempty(hz):
        reason = getattr(
            hz,
            "_solver_constructive_nonempty_reason",
            "outward_transfer_induction",
        )
        return _finish(
            ("FEASIBLE", f"constructive:{reason}"),
            exact_feasible=True,
        )

    if not _HAS_SCIPY:
        return ("UNKNOWN", "scipy_unavailable")

    try:
        A, rl, ru, lb, ub, integ = _base_milp_matrices(hz)
    except Exception as exc:
        return (
            "UNKNOWN",
            f"base_matrix_error:{type(exc).__name__}:{str(exc)[:120]}",
        )
    if A.shape[1] == 0:
        zero = np.zeros(0, dtype=np.float64)
        exact, reason = _exact_stored_float_feasible_candidate(
            A, rl, ru, lb, ub, integ, zero
        )
        if exact is None:
            return _finish(("INFEASIBLE", f"constant_rows_exact:{reason}"))
        _set_witness(exact, integ)
        return _finish(
            ("FEASIBLE", "bare_point_exact"),
            exact_feasible=True,
        )

    if A.shape[0] == 0:
        exact, reason = _exact_stored_float_feasible_candidate(
            A,
            rl,
            ru,
            lb,
            ub,
            integ,
            np.zeros(A.shape[1], dtype=np.float64),
        )
        if exact is None:
            return ("UNKNOWN", f"unconstrained_exact:{reason}")
        _set_witness(exact, integ)
        return _finish(
            ("FEASIBLE", "unconstrained_box_exact"),
            exact_feasible=True,
        )

    if _HAS_HIGHSPY:
        try:
            h = _highspy.Highs()
            HS = _highspy.HighsStatus

            def _require_ok(status, operation):
                if status != HS.kOk:
                    raise RuntimeError(f"{operation} returned {status}")

            _require_ok(h.setOptionValue("output_flag", False), "set output_flag")
            _require_ok(
                h.setOptionValue("time_limit", float(time_limit)),
                "set time_limit",
            )
            _require_ok(h.setOptionValue("presolve", "on"), "set presolve")
            _require_ok(
                h.setOptionValue("threads", _highs_process_threads()),
                "set threads",
            )
            _require_ok(h.addCols(
                A.shape[1],
                np.zeros(A.shape[1], dtype=np.float64),
                lb,
                ub,
                0,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            ), "add columns")
            if np.any(integ):
                vt = np.array([
                    _highspy.HighsVarType.kInteger if m else _highspy.HighsVarType.kContinuous
                    for m in integ.astype(bool)
                ])
                _require_ok(
                    h.changeColsIntegrality(
                        A.shape[1],
                        np.arange(A.shape[1], dtype=np.int32),
                        vt,
                    ),
                    "change integrality",
                )
            As = _sp.csr_matrix(A)
            _require_ok(h.addRows(
                As.shape[0],
                rl,
                ru,
                As.nnz,
                As.indptr.astype(np.int32),
                As.indices.astype(np.int32),
                As.data.astype(np.float64),
            ), "add rows")
            run_status = h.run()
            if run_status != HS.kOk:
                raise RuntimeError(f"run returned {run_status}")
            st = h.getModelStatus()
            msg = h.modelStatusToString(st)
            MS = _highspy.HighsModelStatus
            if st == MS.kOptimal:
                sol = np.asarray(
                    h.getSolution().col_value,
                    dtype=np.float64,
                ).reshape(-1)
                exact, reason = _exact_stored_float_feasible_candidate(
                    A, rl, ru, lb, ub, integ, sol
                )
                if exact is not None:
                    _set_witness(exact, integ)
                    out = ("FEASIBLE", f"highs:{msg}:exact")
                else:
                    out = (
                        "UNKNOWN",
                        f"highs:{msg}:not_exact:{reason}",
                    )
            elif st == MS.kInfeasible:
                out = ("INFEASIBLE", f"highs:{msg}")
            else:
                out = ("UNKNOWN", f"highs:{msg}")
            return _finish(
                out,
                exact_feasible=out[0] == "FEASIBLE",
            )
        except Exception as exc:
            highs_msg = f"highs_error:{type(exc).__name__}:{str(exc)[:120]}"
    else:
        highs_msg = "highspy_unavailable"

    try:
        cons = [_LC(A, lb=rl, ub=ru)]
        r = _milp(
            c=np.zeros(A.shape[1], dtype=np.float64),
            constraints=cons,
            integrality=integ,
            bounds=_LPB(lb, ub),
            options={"time_limit": float(time_limit), "mip_rel_gap": 1e-9},
        )
        if r.success:
            exact, reason = _exact_stored_float_feasible_candidate(
                A,
                rl,
                ru,
                lb,
                ub,
                integ,
                np.asarray(r.x, dtype=np.float64),
            )
            if exact is not None:
                _set_witness(exact, integ)
                out = (
                    "FEASIBLE",
                    f"{highs_msg}; scipy_milp:{r.message}:exact",
                )
            else:
                out = (
                    "UNKNOWN",
                    f"{highs_msg}; scipy_milp:{r.message}:"
                    f"not_exact:{reason}",
                )
        elif str(getattr(r, "message", "")).lower().find("infeasible") >= 0:
            out = ("INFEASIBLE", f"{highs_msg}; scipy_milp:{r.message}")
        else:
            out = ("UNKNOWN", f"{highs_msg}; scipy_milp:{r.message}")
    except Exception as exc:
        out = ("UNKNOWN", f"{highs_msg}; scipy_milp_error:{type(exc).__name__}:{str(exc)[:120]}")
    return _finish(
        out,
        exact_feasible=out[0] == "FEASIBLE",
    )


def hz_base_witness(hz, *, time_limit: float = 10.0):
    """Return a feasible base-HZ ``xi`` point, or ``None`` if unavailable."""

    status, msg = hz_base_feasibility(hz, time_limit=time_limit)
    if status != "FEASIBLE":
        return None, msg
    xi = getattr(hz, "_solver_base_witness_cache", None)
    if xi is None:
        return None, "feasible_without_cached_witness"
    return np.asarray(xi, dtype=np.float64).reshape(-1).copy(), msg


def _hz_spec_unsafe_at_xi(hz, C, t, xi, *, is_unsafe_linear: bool,
                          tol: float = 1e-9) -> bool:
    c, Gc, Gb, *_ = hz_np_sparse(hz)
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    ng, nb = int(Gc.shape[1]), int(Gb.shape[1])
    if xi.size < ng + nb:
        return False
    vals = C @ c - t
    if ng:
        vals = vals + _mat_dot_gen(C, Gc) @ xi[:ng]
    if nb:
        vals = vals + _mat_dot_gen(C, Gb) @ xi[ng:ng + nb]
    if is_unsafe_linear:
        return bool(np.all(vals <= tol))
    return bool(np.any(vals >= -tol))


def _hz_run_row_queries(
    row_count: int,
    row_workers: int,
    solve_row,
    deadline: float,
    *,
    clock=time.monotonic,
):
    """Run rival-row MILPs under one wall-clock deadline.

    ``solve_row`` must derive its solver time limit from ``deadline`` immediately
    before entering the solver.  At most ``row_workers`` queries are in flight,
    so a witness can stop submission and cancel work which has not started.
    ``empty`` is returned only after every row has been proved empty.
    """
    row_count = int(row_count)
    if row_count <= 0:
        return ("empty", None)

    row_workers = max(1, min(int(row_workers), row_count))
    any_unknown = False

    if row_workers == 1:
        for row in range(row_count):
            if clock() >= deadline:
                return ("unknown", None)
            try:
                kind, xi = solve_row(row)
            except Exception:
                logger.exception("HybridZ rival-row query failed")
                kind, xi = "unknown", None
            if clock() >= deadline:
                return ("unknown", None)
            if kind == "witness":
                return ("witness", xi)
            if kind != "empty":
                any_unknown = True
        return ("unknown", None) if any_unknown else ("empty", None)

    executor = ThreadPoolExecutor(max_workers=row_workers)
    pending = {}
    next_row = 0
    abandon = False
    try:
        while next_row < row_count and len(pending) < row_workers:
            if clock() >= deadline:
                abandon = True
                break
            fut = executor.submit(solve_row, next_row)
            pending[fut] = next_row
            next_row += 1

        while pending and not abandon:
            remaining = deadline - clock()
            if remaining <= 0.0:
                abandon = True
                break
            done, _ = wait(
                tuple(pending),
                timeout=remaining,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                abandon = True
                break

            for fut in done:
                pending.pop(fut, None)
                try:
                    kind, xi = fut.result()
                except Exception:
                    logger.exception("HybridZ parallel rival-row query failed")
                    kind, xi = "unknown", None

                if clock() >= deadline:
                    abandon = True
                    break
                if kind == "witness":
                    for other in pending:
                        other.cancel()
                    abandon = True
                    return ("witness", xi)
                if kind != "empty":
                    any_unknown = True

            while (
                not abandon
                and next_row < row_count
                and len(pending) < row_workers
            ):
                if clock() >= deadline:
                    abandon = True
                    break
                fut = executor.submit(solve_row, next_row)
                pending[fut] = next_row
                next_row += 1
    finally:
        if abandon:
            for fut in pending:
                fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)

    if abandon or next_row < row_count:
        return ("unknown", None)
    return ("unknown", None) if any_unknown else ("empty", None)


def _hz_shared_deadline_self_test():
    """Dependency-free scheduler regression checks; raises on any failure."""
    class _FakeClock:
        now = 0.0

        def __call__(self):
            return self.now

    fake = _FakeClock()
    calls = []

    def _budget_consumer(row):
        calls.append(row)
        fake.now += 0.6
        return ("empty", None)

    assert _hz_run_row_queries(
        4, 1, _budget_consumer, 1.0, clock=fake
    ) == ("unknown", None)
    assert calls == [0, 1]

    fake = _FakeClock()

    def _unknown_then_witness(row):
        return ("unknown", None) if row == 0 else ("witness", row)

    assert _hz_run_row_queries(
        4, 1, _unknown_then_witness, 1.0, clock=fake
    ) == ("witness", 1)

    fake = _FakeClock()
    certified_rows = []

    def _all_empty(row):
        certified_rows.append(row)
        return ("empty", None)

    assert _hz_run_row_queries(
        4, 1, _all_empty, 1.0, clock=fake
    ) == ("empty", None)
    assert certified_rows == [0, 1, 2, 3]

    # Force one worker to remain in flight while another yields a witness.
    # The scheduler must return without waiting for the blocked worker.
    from threading import Event

    blocker_started = Event()
    release_blocker = Event()

    def _parallel_witness(row):
        if row == 0:
            assert blocker_started.wait(1.0)
            return ("witness", row)
        blocker_started.set()
        assert release_blocker.wait(1.0)
        return ("empty", None)

    try:
        parallel_result = _hz_run_row_queries(
            4, 2, _parallel_witness, time.monotonic() + 2.0
        )
    finally:
        release_blocker.set()
    assert parallel_result == ("witness", 0)
    return True


_HZ_LP_CERTIFICATE_SCHEMA = "hz_lp_lagrangian_longdouble_v1"
_HZ_SPLIT_BLOCK_LP_CERTIFICATE_SCHEMA = (
    "hz_lp_lagrangian_split_blocks_longdouble_v1"
)
_HZ_PREFORMED_FACTOR_OBJECTIVE_SCHEMA = (
    "hz_exact_preformed_factor_objective_envelope_v1"
)
_HZ_PREFORMED_SPLIT_BLOCK_LP_CERTIFICATE_SCHEMA = (
    "hz_lp_lagrangian_preformed_objective_split_blocks_longdouble_v1"
)
_HZ_PREFORMED_FACTOR_OBJECTIVE_TOKEN = object()
_HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_COLUMNS = 1_000_000
_HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_EXACT_TERMS = 250_000
_HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_OUTPUTS = 250_000
_HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_SOURCE_NNZ = 2_000_000
_HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY_LOCK = Lock()
_HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY = {}


@dataclass(frozen=True)
class _HZPreformedFactorObjectiveRegistryRecord:
    envelope_ref: Any
    original_values: Tuple[Any, ...]
    original_seal: Any
    process_id: int


class _HZSplitBlockCertificateDeadline(TimeoutError):
    """Internal fail-closed deadline signal for the split checker."""


def _hz_split_certificate_deadline(deadline, stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise _HZSplitBlockCertificateDeadline(stage)


def _hz_split_certificate_csr(
    matrix,
    *,
    rows: int,
    columns: int,
    name: str,
):
    """Validate one binary64 CSR block without converting or copying it."""

    if (
        not _sp.isspmatrix_csr(matrix)
        or matrix.dtype != np.dtype(np.float64)
        or matrix.shape != (int(rows), int(columns))
        or not matrix.has_canonical_format
        or np.asarray(matrix.indptr).ndim != 1
        or np.asarray(matrix.indices).ndim != 1
        or np.asarray(matrix.data).ndim != 1
        or int(matrix.indptr.size) != int(rows) + 1
        or int(matrix.indices.size) != int(matrix.data.size)
    ):
        raise ValueError(
            f"split LP certificate {name} must be canonical finite "
            "binary64 CSR with the exact declared shape"
        )
    # ``np.isfinite(matrix.data)`` over the complete array would itself be an
    # O(nnz) boolean allocation.  Validate in the same bounded chunks used by
    # the arithmetic path so malformed input cannot recreate a full-CSR peak.
    for start in range(0, int(matrix.nnz), 65536):
        if not np.all(np.isfinite(matrix.data[start : start + 65536])):
            raise ValueError(
                f"split LP certificate {name} contains non-finite data"
            )
    return matrix


def _hz_ld_sparse_weighted_columns_split(
    matrix,
    weights,
    *,
    name: str,
    deadline=None,
    chunk_nonzeros: int = 65536,
):
    """Compute ``matrix.T @ weights`` with bounded temporary storage.

    The CSR is never sliced, transposed, copied, or made absolute.  Canonical
    rows have unique column indices, so each row can update the three dense
    column accumulators directly.  A very wide row is processed in bounded
    chunks.  ``mass`` is rounded outward after every product and addition and
    therefore safely dominates the exact absolute product mass used by the
    subsequent long-double gamma guard.
    """

    dtype = np.longdouble
    inf = dtype(np.inf)
    weight = np.asarray(weights, dtype=dtype).reshape(-1)
    if (
        weight.size != matrix.shape[0]
        or not np.all(np.isfinite(weight))
        or isinstance(chunk_nonzeros, (bool, np.bool_))
        or int(chunk_nonzeros) <= 0
    ):
        raise ValueError(f"{name} has invalid weights or chunk bound")
    chunk_nonzeros = int(chunk_nonzeros)
    estimate = np.zeros(matrix.shape[1], dtype=dtype)
    mass = np.zeros(matrix.shape[1], dtype=dtype)
    counts = np.zeros(matrix.shape[1], dtype=np.int64)
    active_rows = np.flatnonzero(weight != 0.0).astype(
        np.int64, copy=False
    )
    _hz_split_certificate_deadline(deadline, f"before_{name}")
    for active_position, row_raw in enumerate(active_rows):
        if active_position % 128 == 0:
            _hz_split_certificate_deadline(deadline, f"during_{name}")
        row = int(row_raw)
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        while start < stop:
            chunk_stop = min(stop, start + chunk_nonzeros)
            indices = matrix.indices[start:chunk_stop]
            # Conversion allocates at most ``chunk_nonzeros`` long doubles.
            products = np.asarray(
                matrix.data[start:chunk_stop], dtype=dtype
            )
            products *= weight[row]
            estimate[indices] += products
            np.abs(products, out=products)
            positive = products > 0.0
            if np.any(positive):
                products[positive] = np.nextafter(
                    products[positive], inf
                )
            updated_mass = mass[indices] + products
            mass[indices] = np.nextafter(updated_mass, inf)
            counts[indices] += 1
            start = chunk_stop
    _hz_split_certificate_deadline(deadline, f"after_{name}")
    guard = _hz_ld_roundoff_guard(
        mass,
        2 * counts + 6,
        name=name,
    )
    if (
        not np.all(np.isfinite(estimate))
        or not np.all(np.isfinite(mass))
        or not np.all(np.isfinite(guard))
    ):
        raise ValueError(f"{name} produced a non-finite enclosure")
    return estimate, mass, counts, guard


def _hz_longdouble_certificate_platform() -> Tuple[bool, str]:
    """Check the arithmetic assumptions used by the LP certificate verifier.

    Binary64 inputs convert exactly to the x86 extended ``long double`` used
    by the benchmark environment.  Its exponent range also contains every
    product of two finite binary64 values, so the standard gamma bounds below
    do not need an uncheckable underflow exception.
    """

    info = np.finfo(np.longdouble)
    if (
        int(getattr(info, "nmant", 0)) < 63
        or int(getattr(info, "maxexp", 0)) < 2048
        or int(getattr(info, "minexp", 0)) > -2148
        or np.longdouble(info.eps) > np.longdouble(2.0) ** np.longdouble(-63)
    ):
        return False, (
            f"insufficient_longdouble:nmant={getattr(info, 'nmant', None)},"
            f"minexp={getattr(info, 'minexp', None)},"
            f"maxexp={getattr(info, 'maxexp', None)}"
        )
    return True, "extended_binary_longdouble"


def _hz_longdouble_to_outward_float64_upper(value) -> float:
    """Materialize a finite binary64 upper without downward rounding.

    Python's direct ``float(np.longdouble(...))`` uses round-to-nearest and
    can therefore fall below an authoritative long-double upper by a fraction
    of one binary64 ulp.  This conversion explicitly advances toward
    ``+inf`` only when the nearest binary64 lies below the input.
    """

    exact = np.asarray(value, dtype=np.longdouble).reshape(-1)
    if exact.size != 1 or not np.isfinite(exact[0]):
        raise ValueError("longdouble upper is not one finite scalar")
    exact_scalar = exact[0]
    rounded = float(exact_scalar)
    if rounded == -np.inf:
        rounded = -float(np.finfo(np.float64).max)
    if not np.isfinite(rounded):
        raise ValueError("longdouble upper has no finite binary64 upper")
    if np.longdouble(rounded) < exact_scalar:
        rounded = float(np.nextafter(rounded, np.inf))
    if (
        not np.isfinite(rounded)
        or np.longdouble(rounded) < exact_scalar
    ):
        raise ValueError("binary64 upper conversion was not outward")
    return rounded


def _hz_ld_roundoff_guard(mass, op_count, *, name: str):
    """Conservative outward error for a long-double reduction.

    ``16 * N * eps`` deliberately dominates Higham's ``gamma_N`` as well as
    the rounding used to form the nonnegative mass estimate.  The explicit
    ``N*eps < 1/64`` gate keeps that simple domination proof in its small-error
    regime.  A zero mass is structurally exact and retains a zero guard.
    """

    dtype = np.longdouble
    inf = dtype(np.inf)
    value = np.asarray(mass, dtype=dtype)
    count = np.broadcast_to(np.asarray(op_count, dtype=dtype), value.shape)
    if (
        np.any(value < 0)
        or not np.all(np.isfinite(value))
        or np.any(count < 0)
        or not np.all(np.isfinite(count))
    ):
        raise ValueError(f"{name} has invalid mass/operation count")
    scaled = count * dtype(np.finfo(dtype).eps)
    if np.any(scaled >= dtype(1.0) / dtype(64.0)):
        raise ValueError(f"{name} exceeds the long-double gamma regime")
    factor = np.nextafter(dtype(16.0) * scaled, inf)
    guard = np.zeros(value.shape, dtype=dtype)
    active = value > 0
    if np.any(active):
        guard[active] = np.nextafter(
            factor[active] * value[active],
            inf,
        )
    if not np.all(np.isfinite(guard)):
        raise ValueError(f"{name} roundoff guard overflowed")
    return guard


def _hz_ld_sum_products_upper(left, right, *, name: str):
    """Outward upper bound on the exact sum of long-double products."""

    dtype = np.longdouble
    inf = dtype(np.inf)
    lhs = np.asarray(left, dtype=dtype).reshape(-1)
    rhs = np.asarray(right, dtype=dtype).reshape(-1)
    if lhs.size != rhs.size:
        raise ValueError(f"{name} product shape mismatch")
    if lhs.size == 0:
        return dtype(0.0), dtype(0.0)
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError(f"{name} has a non-finite product input")
    products = lhs * rhs
    if not np.all(np.isfinite(products)):
        raise ValueError(f"{name} product overflowed")
    rounded = np.sum(products, dtype=dtype)
    mass = np.sum(np.abs(products), dtype=dtype)
    guard = _hz_ld_roundoff_guard(
        mass,
        2 * int(products.size) + 4,
        name=name,
    ).reshape(())
    upper = np.nextafter(rounded + guard, inf)
    if not np.isfinite(upper):
        raise ValueError(f"{name} upper bound overflowed")
    return upper, guard


def _hz_ld_objective_enclosure(
    c,
    Gc,
    C_row,
    threshold,
    *,
    center_error=None,
):
    """Independently enclose ``q=C@Gc`` and ``kappa=C@c-threshold``."""

    dtype = np.longdouble
    inf = dtype(np.inf)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    row = np.asarray(C_row, dtype=np.float64).reshape(-1)
    if c.size != row.size:
        raise ValueError("LP certificate objective/output shape mismatch")
    threshold = float(threshold)
    G = _sp.csr_matrix(Gc, dtype=np.float64, copy=False)
    if G.shape[0] != c.size:
        raise ValueError("LP certificate generator/output shape mismatch")
    if (
        not G.has_canonical_format
        or not np.all(np.isfinite(c))
        or not np.all(np.isfinite(row))
        or not np.isfinite(threshold)
        or (G.nnz and not np.all(np.isfinite(G.data)))
    ):
        raise ValueError("LP certificate objective data are not canonical finite CSR")
    if center_error is None:
        center_error_ld = np.zeros(c.size, dtype=dtype)
    else:
        center_error_ld = np.asarray(
            center_error,
            dtype=dtype,
        ).reshape(-1)
        if (
            center_error_ld.size != c.size
            or not np.all(np.isfinite(center_error_ld))
            or np.any(center_error_ld < 0)
        ):
            raise ValueError(
                "LP certificate center transformation error is invalid"
            )

    selected = np.flatnonzero(row != 0.0).astype(np.int64, copy=False)
    if selected.size:
        local_G = G[selected, :].tocsr()
        coefficients = row[selected].astype(dtype)
        q_hat = np.asarray(
            local_G.transpose() @ coefficients,
            dtype=dtype,
        ).reshape(-1)
        abs_G = local_G.copy()
        abs_G.data = np.abs(abs_G.data)
        q_mass = np.asarray(
            abs_G.transpose() @ np.abs(coefficients),
            dtype=dtype,
        ).reshape(-1)
        q_counts = np.bincount(
            local_G.indices,
            minlength=G.shape[1],
        ).astype(np.int64, copy=False)
    else:
        q_hat = np.zeros(G.shape[1], dtype=dtype)
        q_mass = np.zeros(G.shape[1], dtype=dtype)
        q_counts = np.zeros(G.shape[1], dtype=np.int64)
    q_error = _hz_ld_roundoff_guard(
        q_mass,
        2 * q_counts + 6,
        name="LP certificate objective coefficients",
    )

    center_products = row[selected].astype(dtype) * c[selected].astype(dtype)
    if not np.all(np.isfinite(center_products)):
        raise ValueError("LP certificate objective center overflowed")
    kappa_hat = (
        np.sum(center_products, dtype=dtype)
        - dtype(threshold)
    )
    kappa_mass = (
        np.sum(np.abs(center_products), dtype=dtype)
        + np.abs(dtype(threshold))
    )
    kappa_arithmetic_error = _hz_ld_roundoff_guard(
        kappa_mass,
        2 * int(selected.size) + 6,
        name="LP certificate objective constant",
    ).reshape(())
    if center_error is None:
        # Preserve the established continuous-only certificate bit for bit.
        kappa_error = kappa_arithmetic_error
    else:
        center_uncertainty_upper, _ = _hz_ld_sum_products_upper(
            np.abs(row[selected].astype(dtype)),
            center_error_ld[selected],
            name="LP certificate center transformation uncertainty",
        )
        kappa_error = np.nextafter(
            kappa_arithmetic_error + center_uncertainty_upper,
            inf,
        )
    kappa_upper = np.nextafter(kappa_hat + kappa_error, inf)
    if (
        not np.all(np.isfinite(q_hat))
        or not np.all(np.isfinite(q_error))
        or not np.isfinite(kappa_hat)
        or not np.isfinite(kappa_upper)
    ):
        raise ValueError("LP certificate objective enclosure is non-finite")
    return q_hat, q_error, kappa_hat, kappa_error


def _hz_binary_relaxed_output_frame(c, Gc, Gb):
    """Map signed binary HZ factors to the base LP's ``z in [0, 1]`` frame.

    The base matrices use ``xi_b = 2*z - 1``.  Therefore the same output is

    ``(c - Gb@1) + [Gc, 2*Gb] @ [xi_c, z]``.

    ``center_error`` encloses the difference between the exact expression over
    the stored binary64 coefficients and the rounded binary64 center returned
    here.  The independent certificate consumes that error one-sided through
    ``abs(C_row)``; candidate-solver arithmetic is never trusted.
    """

    dtype = np.longdouble
    inf = dtype(np.inf)
    center_source = np.asarray(c, dtype=np.float64).reshape(-1)
    cont = _sp.csr_matrix(Gc, dtype=np.float64, copy=False)
    binary = _sp.csr_matrix(Gb, dtype=np.float64, copy=False)
    if (
        cont.shape[0] != center_source.size
        or binary.shape[0] != center_source.size
        or binary.shape[1] == 0
        or not cont.has_canonical_format
        or not binary.has_canonical_format
        or not np.all(np.isfinite(center_source))
        or (cont.nnz and not np.all(np.isfinite(cont.data)))
        or (binary.nnz and not np.all(np.isfinite(binary.data)))
    ):
        raise ValueError("binary-relaxed LP output frame is invalid")

    doubled_binary = binary.copy()
    doubled_binary.data *= 2.0
    if doubled_binary.nnz and not np.all(
        np.isfinite(doubled_binary.data)
    ):
        raise ValueError("binary-relaxed LP output generator overflowed")
    combined = _sp.hstack(
        [cont, doubled_binary],
        format="csr",
    )
    combined.sum_duplicates()
    combined.sort_indices()

    center = np.empty(center_source.size, dtype=np.float64)
    center_error = np.empty(center_source.size, dtype=dtype)
    for row_index in range(center_source.size):
        start = int(binary.indptr[row_index])
        end = int(binary.indptr[row_index + 1])
        values = binary.data[start:end].astype(dtype)
        row_sum = np.sum(values, dtype=dtype)
        abs_mass = (
            np.abs(dtype(center_source[row_index]))
            + np.sum(np.abs(values), dtype=dtype)
        )
        arithmetic_error = _hz_ld_roundoff_guard(
            abs_mass,
            2 * int(values.size) + 8,
            name="binary-relaxed LP center transformation",
        ).reshape(())
        center_ld = dtype(center_source[row_index]) - row_sum
        rounded = float(center_ld)
        if not np.isfinite(rounded):
            raise ValueError("binary-relaxed LP center overflowed")
        center[row_index] = rounded
        representation_error = np.abs(dtype(rounded) - center_ld)
        center_error[row_index] = np.nextafter(
            representation_error + arithmetic_error,
            inf,
        )
    if not np.all(np.isfinite(center_error)):
        raise ValueError(
            "binary-relaxed LP center transformation guard overflowed"
        )
    return center, combined, center_error


class _HZPreformedFactorObjectiveEnvelope:
    """Immutable solver-private enclosure of one exact stored objective.

    The packed byte strings are the authority-bearing representation.  They
    expose read-only NumPy views, cannot be changed in place, and are bound to
    a private construction seal.  The independent checker validates that
    seal in O(1); it deliberately does not re-hash O(number_of_factors) bytes
    for every conditional pattern.
    """

    __slots__ = (
        "_schema",
        "_parent_semantic_digest",
        "_objective_id",
        "_objective_source_sha256",
        "_stable_ids_sha256",
        "_exact_objective_sha256",
        "_objective_binding_sha256",
        "_objective_center_exact",
        "_continuous_terms_exact",
        "_binary_terms_exact",
        "_envelope_sha256",
        "_n_continuous",
        "_n_binary",
        "_source_generator_nnz",
        "_exact_term_count",
        "_q_continuous_hat_bytes",
        "_q_continuous_error_bytes",
        "_q_binary_hat_bytes",
        "_q_binary_error_bytes",
        "_kappa_hat",
        "_kappa_error",
        "_platform",
        "_process_id",
        "_receipt",
        "_seal",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        parent_semantic_digest: str,
        objective_id: str,
        objective_source_sha256: str,
        stable_ids_sha256: str,
        exact_objective_sha256: str,
        objective_binding_sha256: str,
        objective_center_exact: Fraction,
        continuous_terms_exact: Tuple[Tuple[int, Fraction], ...],
        binary_terms_exact: Tuple[Tuple[int, Fraction], ...],
        envelope_sha256: str,
        n_continuous: int,
        n_binary: int,
        source_generator_nnz: int,
        exact_term_count: int,
        q_continuous_hat_bytes: bytes,
        q_continuous_error_bytes: bytes,
        q_binary_hat_bytes: bytes,
        q_binary_error_bytes: bytes,
        kappa_hat: float,
        kappa_error: float,
        platform: str,
        receipt: Mapping[str, Any],
        _producer_capability: Any,
    ) -> None:
        if _producer_capability is not _HZ_PREFORMED_FACTOR_OBJECTIVE_TOKEN:
            raise PermissionError(
                "preformed factor objective requires the private producer"
            )
        process_id = int(os.getpid())
        values = (
            _HZ_PREFORMED_FACTOR_OBJECTIVE_SCHEMA,
            str(parent_semantic_digest),
            str(objective_id),
            str(objective_source_sha256),
            str(stable_ids_sha256),
            str(exact_objective_sha256),
            str(objective_binding_sha256),
            objective_center_exact,
            continuous_terms_exact,
            binary_terms_exact,
            str(envelope_sha256),
            int(n_continuous),
            int(n_binary),
            int(source_generator_nnz),
            int(exact_term_count),
            q_continuous_hat_bytes,
            q_continuous_error_bytes,
            q_binary_hat_bytes,
            q_binary_error_bytes,
            float(kappa_hat),
            float(kappa_error),
            str(platform),
            process_id,
            receipt,
        )
        for name, value in zip(self.__slots__[:-2], values):
            object.__setattr__(self, name, value)
        # The seal retains the exact immutable byte objects and scalar
        # metadata installed by the producer.  Attribute replacement through
        # ``object.__setattr__`` is therefore detected without re-hashing.
        object.__setattr__(
            self,
            "_seal",
            values,
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("preformed factor objective envelopes are immutable")

    def __copy__(self):
        raise TypeError("preformed factor objective envelopes cannot be copied")

    def __deepcopy__(self, _memo):
        raise TypeError("preformed factor objective envelopes cannot be copied")

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def parent_semantic_digest(self) -> str:
        return self._parent_semantic_digest

    @property
    def objective_id(self) -> str:
        return self._objective_id

    @property
    def objective_source_sha256(self) -> str:
        return self._objective_source_sha256

    @property
    def stable_ids_sha256(self) -> str:
        return self._stable_ids_sha256

    @property
    def exact_objective_sha256(self) -> str:
        return self._exact_objective_sha256

    @property
    def objective_binding_sha256(self) -> str:
        return self._objective_binding_sha256

    @property
    def objective_center_exact(self) -> Fraction:
        return self._objective_center_exact

    @property
    def continuous_terms_exact(self) -> Tuple[Tuple[int, Fraction], ...]:
        return self._continuous_terms_exact

    @property
    def binary_terms_exact(self) -> Tuple[Tuple[int, Fraction], ...]:
        return self._binary_terms_exact

    @property
    def envelope_sha256(self) -> str:
        return self._envelope_sha256

    @property
    def n_continuous(self) -> int:
        return self._n_continuous

    @property
    def n_binary(self) -> int:
        return self._n_binary

    @property
    def q_continuous_hat(self):
        return np.frombuffer(
            self._q_continuous_hat_bytes, dtype=np.dtype("<f8")
        )

    @property
    def q_continuous_error(self):
        return np.frombuffer(
            self._q_continuous_error_bytes, dtype=np.dtype("<f8")
        )

    @property
    def q_binary_hat(self):
        return np.frombuffer(
            self._q_binary_hat_bytes, dtype=np.dtype("<f8")
        )

    @property
    def q_binary_error(self):
        return np.frombuffer(
            self._q_binary_error_bytes, dtype=np.dtype("<f8")
        )

    @property
    def kappa_hat(self) -> float:
        return self._kappa_hat

    @property
    def kappa_error(self) -> float:
        return self._kappa_error

    @property
    def receipt(self) -> Mapping[str, Any]:
        return self._receipt


def _hz_register_preformed_factor_objective_envelope(
    envelope: _HZPreformedFactorObjectiveEnvelope,
    *,
    _producer_capability: Any,
) -> None:
    """Register one reusable envelope by exact process-local identity."""

    if (
        _producer_capability is not _HZ_PREFORMED_FACTOR_OBJECTIVE_TOKEN
        or type(envelope) is not _HZPreformedFactorObjectiveEnvelope
    ):
        raise PermissionError("preformed objective registry producer is invalid")
    identity = id(envelope)

    def _cleanup(reference, *, identity=identity):
        with _HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY_LOCK:
            current = _HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY.get(identity)
            if current is not None and current.envelope_ref is reference:
                _HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY.pop(identity, None)

    reference = weakref.ref(envelope, _cleanup)
    original_values = tuple(
        getattr(envelope, name) for name in envelope.__slots__[:-2]
    )
    record = _HZPreformedFactorObjectiveRegistryRecord(
        envelope_ref=reference,
        original_values=original_values,
        original_seal=envelope._seal,
        process_id=int(os.getpid()),
    )
    with _HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY_LOCK:
        existing = _HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY.get(identity)
        if existing is not None and existing.envelope_ref() is not None:
            raise PermissionError("preformed objective identity is already registered")
        _HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY[identity] = record


def _hz_preformed_factor_objective_registry_record(envelope):
    """Return the original registry record or reject copies/forgeries."""

    identity = id(envelope)
    with _HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY_LOCK:
        record = _HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY.get(identity)
        if (
            type(record) is not _HZPreformedFactorObjectiveRegistryRecord
            or record.process_id != int(os.getpid())
            or record.envelope_ref() is not envelope
        ):
            raise PermissionError(
                "preformed objective is not the registered live envelope"
            )
        return record


def _hz_preformed_require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one canonical lowercase SHA-256")
    return value


def _hz_preformed_stable_ids(values, *, size: int, name: str) -> np.ndarray:
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.dtype(np.int64)
        and values.ndim == 1
    ):
        if int(values.size) != int(size):
            raise ValueError(f"{name} has the wrong length")
        out = np.ascontiguousarray(values).copy()
        if np.any(out < 0):
            raise ValueError(f"{name} contains a negative stable id")
        if np.unique(out).size != out.size:
            raise ValueError(f"{name} contains duplicate stable ids")
        out.setflags(write=False)
        return out
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer sequence") from exc
    if len(raw) != int(size):
        raise ValueError(f"{name} has the wrong length")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ValueError(f"{name} must contain strict integers")
    try:
        out = np.asarray(raw, dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is outside signed int64") from exc
    if np.any(out < 0):
        raise ValueError(f"{name} contains a negative stable id")
    if len(set(int(value) for value in raw)) != len(raw):
        raise ValueError(f"{name} contains duplicate stable ids")
    out.setflags(write=False)
    return out


def _hz_preformed_hash_array(
    digest,
    *,
    name: str,
    array,
    dtype,
    deadline,
) -> None:
    canonical = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    digest.update(name.encode("ascii") + b"\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(np.dtype(dtype).str.encode("ascii") + b"\0")
    raw = canonical.view(np.uint8).reshape(-1)
    for start in range(0, int(raw.size), 1048576):
        _hz_split_certificate_deadline(
            deadline, f"during_preformed_source_hash_{name}"
        )
        digest.update(memoryview(raw[start : start + 1048576]))


def _hz_preformed_hash_csr(
    digest,
    *,
    name: str,
    matrix,
    deadline,
) -> None:
    digest.update(name.encode("ascii") + b"\0")
    _hz_preformed_hash_array(
        digest,
        name=f"{name}_shape",
        array=np.asarray(matrix.shape, dtype=np.int64),
        dtype=np.dtype("<i8"),
        deadline=deadline,
    )
    _hz_preformed_hash_array(
        digest,
        name=f"{name}_indptr",
        array=matrix.indptr,
        dtype=np.dtype("<i8"),
        deadline=deadline,
    )
    _hz_preformed_hash_array(
        digest,
        name=f"{name}_indices",
        array=matrix.indices,
        dtype=np.dtype("<i8"),
        deadline=deadline,
    )
    _hz_preformed_hash_array(
        digest,
        name=f"{name}_data",
        array=matrix.data,
        dtype=np.dtype("<f8"),
        deadline=deadline,
    )


def _hz_preformed_hash_fraction(digest, value: Fraction) -> None:
    for integer in (value.numerator, value.denominator):
        magnitude = abs(int(integer))
        encoded = magnitude.to_bytes(
            max(1, (magnitude.bit_length() + 7) // 8),
            byteorder="little",
            signed=False,
        )
        digest.update(b"-" if integer < 0 else b"+")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)


def _hz_preformed_rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _hz_preformed_objective_binding_sha256(
    *,
    objective_id: str,
    parent_semantic_digest: str,
    center: Fraction,
    continuous_ids: np.ndarray,
    continuous_exact: Mapping[int, Fraction],
    binary_ids: np.ndarray,
    binary_exact: Mapping[int, Fraction],
) -> str:
    """Reproduce the core ObjectiveBinding v1 canonical digest exactly."""

    if (
        type(objective_id) is not str
        or not objective_id
        or len(objective_id) > 256
        or any(ord(character) < 32 or ord(character) > 126 for character in objective_id)
    ):
        raise ValueError("preformed objective_id is not canonical printable ASCII")

    def _terms(ids, exact_values):
        values = [
            (int(ids[position]), exact)
            for position, exact in exact_values.items()
            if exact != 0
        ]
        values.sort(key=lambda item: item[0])
        return [
            [stable_id, _hz_preformed_rational_text(exact)]
            for stable_id, exact in values
        ]

    payload = {
        "schema": "act.hybridz_pc_objective_binding.v1",
        "objective_id": objective_id,
        "parent_semantic_digest": parent_semantic_digest,
        "center": _hz_preformed_rational_text(center),
        "continuous_terms": _terms(continuous_ids, continuous_exact),
        "binary_terms": _terms(binary_ids, binary_exact),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _hz_preformed_fraction_binary64_with_error(
    exact: Fraction,
    *,
    name: str,
    position: int,
) -> Tuple[float, float]:
    """Nearest binary64 nominal plus a finite outward absolute error."""

    try:
        nominal = float(exact)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name} {position} nominal is not finite binary64"
        ) from exc
    if not np.isfinite(nominal):
        raise ValueError(f"{name} {position} nominal is not finite binary64")
    residual = abs(exact - Fraction.from_float(nominal))
    if residual == 0:
        return nominal, 0.0
    try:
        error = float(residual)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name} {position} error is not finite binary64"
        ) from exc
    if error == 0.0:
        error = float(np.nextafter(0.0, np.inf))
    elif Fraction.from_float(error) < residual:
        error = float(np.nextafter(error, np.inf))
    if (
        not np.isfinite(error)
        or error <= 0.0
        or Fraction.from_float(error) < residual
    ):
        raise ValueError(
            f"{name} {position} error cannot be enclosed finitely"
        )
    return nominal, error


def _hz_form_exact_factor_objective_envelope_from_live_split_blocks(
    *,
    c,
    Gc,
    Gb,
    C_row,
    threshold,
    continuous_col_ids,
    binary_col_ids,
    objective_id: str,
    parent_semantic_digest: str,
    deadline=None,
):
    """Form one exact, sealed factor-objective envelope from live blocks.

    Stored binary64 values are interpreted as exact dyadic rationals.  This
    toy-first v1 intentionally has strict output/factor/exact-term caps and is
    marked ``production_ready=False``.  Failure or deadline expiry returns no
    partially formed envelope.
    """

    receipt = {
        "schema": _HZ_PREFORMED_FACTOR_OBJECTIVE_SCHEMA,
        "status": "not_started",
        "route": "exact_fraction_live_split_objective_to_outward_f64_v1",
        "proof_authority": False,
        "verdict_authority": False,
        "pcoh_authorization": False,
        "requires_external_objective_binding_sha256": True,
        "production_ready": False,
        "generator_validation_pass_count": 0,
        "source_hash_pass_count": 0,
        "exact_expansion_pass_count": 0,
        "objective_expansion_count": 0,
        "packed_factor_bytes": None,
        "envelope_sha256": None,
    }
    platform_ok, platform_reason = _hz_longdouble_certificate_platform()
    receipt["platform"] = platform_reason
    if not platform_ok:
        receipt["status"] = "platform_unsupported"
        return None, receipt
    try:
        if deadline is not None:
            if (
                isinstance(deadline, (bool, np.bool_))
                or not np.isscalar(deadline)
                or not np.isfinite(float(deadline))
            ):
                raise ValueError("preformed objective deadline must be finite")
            deadline = float(deadline)
        _hz_split_certificate_deadline(deadline, "preformed_formation_entry")
        parent_digest = _hz_preformed_require_sha256(
            parent_semantic_digest, name="parent_semantic_digest"
        )

        center_raw = np.asarray(c)
        objective_raw = np.asarray(C_row)
        if (
            center_raw.dtype != np.dtype(np.float64)
            or objective_raw.dtype != np.dtype(np.float64)
        ):
            raise ValueError(
                "preformed objective c and C_row must be stored binary64"
            )
        center = center_raw.reshape(-1)
        objective_row = objective_raw.reshape(-1)
        if (
            center.size != objective_row.size
            or center.size > _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_OUTPUTS
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(objective_row))
            or isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (float, np.float64))
            or not np.isfinite(float(threshold))
        ):
            raise ValueError("preformed objective output frame is invalid")
        threshold_f64 = float(threshold)

        raw_gc = Gc
        raw_gb = Gb
        if not _sp.isspmatrix_csr(raw_gc) or not _sp.isspmatrix_csr(raw_gb):
            raise ValueError("preformed objective generators must be CSR")
        if int(raw_gc.nnz) + int(raw_gb.nnz) > (
            _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_SOURCE_NNZ
        ):
            raise ValueError("preformed objective exceeds the source-nnz cap")
        n_continuous = int(raw_gc.shape[1])
        n_binary = int(raw_gb.shape[1])
        total_columns = n_continuous + n_binary
        if total_columns > _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_COLUMNS:
            raise ValueError("preformed objective exceeds the factor cap")
        continuous_ids = _hz_preformed_stable_ids(
            continuous_col_ids,
            size=n_continuous,
            name="continuous_col_ids",
        )
        binary_ids = _hz_preformed_stable_ids(
            binary_col_ids,
            size=n_binary,
            name="binary_col_ids",
        )
        Gc_live = _hz_split_certificate_csr(
            raw_gc,
            rows=center.size,
            columns=n_continuous,
            name="preformed_Gc",
        )
        Gb_live = _hz_split_certificate_csr(
            raw_gb,
            rows=center.size,
            columns=n_binary,
            name="preformed_Gb",
        )
        receipt["generator_validation_pass_count"] = 1
        _hz_split_certificate_deadline(
            deadline, "preformed_formation_after_validation"
        )

        stable_digest = hashlib.sha256()
        stable_digest.update(b"act.hz.preformed.stable_ids.v1\0")
        _hz_preformed_hash_array(
            stable_digest,
            name="continuous_col_ids",
            array=continuous_ids,
            dtype=np.dtype("<i8"),
            deadline=deadline,
        )
        _hz_preformed_hash_array(
            stable_digest,
            name="binary_col_ids",
            array=binary_ids,
            dtype=np.dtype("<i8"),
            deadline=deadline,
        )
        stable_ids_sha256 = stable_digest.hexdigest()

        source_digest = hashlib.sha256()
        source_digest.update(b"act.hz.preformed.objective_source.v1\0")
        source_digest.update(parent_digest.encode("ascii"))
        source_digest.update(stable_ids_sha256.encode("ascii"))
        _hz_preformed_hash_array(
            source_digest,
            name="c",
            array=center,
            dtype=np.dtype("<f8"),
            deadline=deadline,
        )
        _hz_preformed_hash_csr(
            source_digest,
            name="Gc",
            matrix=Gc_live,
            deadline=deadline,
        )
        _hz_preformed_hash_csr(
            source_digest,
            name="Gb",
            matrix=Gb_live,
            deadline=deadline,
        )
        _hz_preformed_hash_array(
            source_digest,
            name="C_row",
            array=objective_row,
            dtype=np.dtype("<f8"),
            deadline=deadline,
        )
        _hz_preformed_hash_array(
            source_digest,
            name="threshold",
            array=np.asarray([threshold_f64], dtype=np.float64),
            dtype=np.dtype("<f8"),
            deadline=deadline,
        )
        objective_source_sha256 = source_digest.hexdigest()
        receipt["source_hash_pass_count"] = 1

        selected_outputs = np.flatnonzero(objective_row != 0.0).astype(
            np.int64, copy=False
        )
        selected_generator_terms = 0
        for offset, output_raw in enumerate(selected_outputs):
            if offset % 4096 == 0:
                _hz_split_certificate_deadline(
                    deadline, "preformed_formation_term_count"
                )
            output = int(output_raw)
            selected_generator_terms += int(
                Gc_live.indptr[output + 1] - Gc_live.indptr[output]
            )
            selected_generator_terms += int(
                Gb_live.indptr[output + 1] - Gb_live.indptr[output]
            )
        exact_term_count = (
            1 + int(selected_outputs.size) + selected_generator_terms
        )
        if exact_term_count > _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_EXACT_TERMS:
            raise ValueError("preformed objective exceeds the exact-term cap")

        q_continuous_exact = {}
        q_binary_exact = {}
        kappa_exact = -Fraction.from_float(threshold_f64)
        for offset, output_raw in enumerate(selected_outputs):
            if offset % 128 == 0:
                _hz_split_certificate_deadline(
                    deadline, "preformed_formation_exact_expansion"
                )
            output = int(output_raw)
            weight = Fraction.from_float(float(objective_row[output]))
            kappa_exact += weight * Fraction.from_float(float(center[output]))
            for matrix, target in (
                (Gc_live, q_continuous_exact),
                (Gb_live, q_binary_exact),
            ):
                start = int(matrix.indptr[output])
                stop = int(matrix.indptr[output + 1])
                for position in range(start, stop):
                    if (position - start) % 4096 == 0:
                        _hz_split_certificate_deadline(
                            deadline,
                            "preformed_formation_exact_generator_terms",
                        )
                    column = int(matrix.indices[position])
                    contribution = weight * Fraction.from_float(
                        float(matrix.data[position])
                    )
                    target[column] = target.get(column, Fraction(0)) + contribution
        receipt["objective_expansion_count"] = 1
        receipt["exact_expansion_pass_count"] = 1
        _hz_split_certificate_deadline(
            deadline, "preformed_formation_after_exact_expansion"
        )

        q_continuous_hat = np.zeros(n_continuous, dtype=np.float64)
        q_continuous_error = np.zeros(n_continuous, dtype=np.float64)
        q_binary_hat = np.zeros(n_binary, dtype=np.float64)
        q_binary_error = np.zeros(n_binary, dtype=np.float64)
        exact_digest = hashlib.sha256()
        exact_digest.update(b"act.hz.preformed.exact_objective.v1\0")
        exact_digest.update(parent_digest.encode("ascii"))
        exact_digest.update(objective_source_sha256.encode("ascii"))
        _hz_preformed_hash_fraction(exact_digest, kappa_exact)
        for kind, ids, exact_values, nominal, error in (
            (
                b"continuous\0",
                continuous_ids,
                q_continuous_exact,
                q_continuous_hat,
                q_continuous_error,
            ),
            (
                b"binary\0",
                binary_ids,
                q_binary_exact,
                q_binary_hat,
                q_binary_error,
            ),
        ):
            exact_digest.update(kind)
            exact_digest.update(
                int(ids.size).to_bytes(8, "little", signed=False)
            )
            # Absent positions are canonical exact zeros.  Hashing the stable
            # id vector once plus only nonzero exact coefficients avoids a
            # million Python Fraction operations for a sparse large frame.
            nonzero_exact = tuple(
                sorted(
                    (column, exact)
                    for column, exact in exact_values.items()
                    if exact != 0
                )
            )
            exact_digest.update(
                len(nonzero_exact).to_bytes(8, "little", signed=False)
            )
            for offset, (column, exact) in enumerate(nonzero_exact):
                if offset % 4096 == 0:
                    _hz_split_certificate_deadline(
                        deadline, "preformed_formation_pack"
                    )
                nominal[column], error[column] = (
                    _hz_preformed_fraction_binary64_with_error(
                        exact,
                        name=kind.decode("ascii").rstrip("\0"),
                        position=column,
                    )
                )
                exact_digest.update(
                    int(ids[column]).to_bytes(8, "little", signed=True)
                )
                _hz_preformed_hash_fraction(exact_digest, exact)
        kappa_hat, kappa_error = _hz_preformed_fraction_binary64_with_error(
            kappa_exact,
            name="kappa",
            position=0,
        )
        exact_objective_sha256 = exact_digest.hexdigest()
        continuous_terms_exact = tuple(sorted(
            (
                (int(continuous_ids[position]), exact)
                for position, exact in q_continuous_exact.items()
                if exact != 0
            ),
            key=lambda item: item[0],
        ))
        binary_terms_exact = tuple(sorted(
            (
                (int(binary_ids[position]), exact)
                for position, exact in q_binary_exact.items()
                if exact != 0
            ),
            key=lambda item: item[0],
        ))
        objective_binding_sha256 = _hz_preformed_objective_binding_sha256(
            objective_id=objective_id,
            parent_semantic_digest=parent_digest,
            center=kappa_exact,
            continuous_ids=continuous_ids,
            continuous_exact=q_continuous_exact,
            binary_ids=binary_ids,
            binary_exact=q_binary_exact,
        )

        packed = tuple(
            np.asarray(values, dtype="<f8").tobytes(order="C")
            for values in (
                q_continuous_hat,
                q_continuous_error,
                q_binary_hat,
                q_binary_error,
            )
        )
        envelope_digest = hashlib.sha256()
        envelope_digest.update(b"act.hz.preformed.envelope.v1\0")
        for value in (
            parent_digest,
            objective_source_sha256,
            stable_ids_sha256,
            exact_objective_sha256,
            objective_binding_sha256,
            platform_reason,
        ):
            envelope_digest.update(value.encode("ascii") + b"\0")
        envelope_digest.update(
            np.asarray(
                [n_continuous, n_binary, exact_term_count], dtype="<i8"
            ).tobytes()
        )
        envelope_digest.update(
            np.asarray([kappa_hat, kappa_error], dtype="<f8").tobytes()
        )
        for raw in packed:
            envelope_digest.update(raw)
        envelope_sha256 = envelope_digest.hexdigest()
        packed_factor_bytes = 16 * total_columns
        receipt.update({
            "status": "formed",
            "parent_semantic_digest": parent_digest,
            "objective_source_sha256": objective_source_sha256,
            "objective_id": objective_id,
            "stable_ids_sha256": stable_ids_sha256,
            "exact_objective_sha256": exact_objective_sha256,
            "objective_binding_sha256": objective_binding_sha256,
            "objective_binding_schema": (
                "act.hybridz_pc_objective_binding.v1"
            ),
            "objective_binding_digest_formation": (
                "exact_core_canonical_json_reproduction_v1"
            ),
            "exact_binding_material_readonly": True,
            "exact_binding_terms_persisted": True,
            "exact_binding_term_count": int(
                len(continuous_terms_exact) + len(binary_terms_exact)
            ),
            "exact_binding_fraction_storage_production_blocker": True,
            "envelope_sha256": envelope_sha256,
            "n_outputs": int(center.size),
            "n_continuous": n_continuous,
            "n_binary": n_binary,
            "source_generator_nnz": int(Gc_live.nnz + Gb_live.nnz),
            "selected_generator_terms": int(selected_generator_terms),
            "exact_term_count": int(exact_term_count),
            "max_outputs": _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_OUTPUTS,
            "max_factor_columns": _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_COLUMNS,
            "max_exact_terms": _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_EXACT_TERMS,
            "max_source_generator_nnz": (
                _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_SOURCE_NNZ
            ),
            "packed_factor_bytes": int(packed_factor_bytes),
            "packed_factor_bytes_is_persistent_lower_bound_only": True,
            "packed_scalar_bytes": 16,
            "total_persistent_bytes_bounded": False,
            "total_persistent_bytes_blocker": (
                "python_fraction_exact_binding_material_v1"
            ),
            "coefficient_storage": "little_endian_binary64_hat_plus_outward_abs_error",
            "exact_arithmetic": "Fraction.from_float_exact_dyadic",
            "threshold_application_count": 1,
            "arrays_readonly": True,
            "envelope_rehash_per_checker": False,
            "process_local_registry": True,
            "registry_lifetime": "weakref_multi_replay_until_envelope_gc",
            "registry_one_use": False,
            "trust_boundary": (
                "process_local_registry_and_solver_module_state_trusted_v1"
            ),
            "uses_sparse_hstack": False,
            "uses_sparse_vstack": False,
            "process_id": int(os.getpid()),
        })
        frozen_receipt = _HZImmutableMapping(
            tuple((key, _hz_freeze_conditional_value(value)) for key, value in receipt.items())
        )
        envelope = _HZPreformedFactorObjectiveEnvelope(
            parent_semantic_digest=parent_digest,
            objective_id=objective_id,
            objective_source_sha256=objective_source_sha256,
            stable_ids_sha256=stable_ids_sha256,
            exact_objective_sha256=exact_objective_sha256,
            objective_binding_sha256=objective_binding_sha256,
            objective_center_exact=kappa_exact,
            continuous_terms_exact=continuous_terms_exact,
            binary_terms_exact=binary_terms_exact,
            envelope_sha256=envelope_sha256,
            n_continuous=n_continuous,
            n_binary=n_binary,
            source_generator_nnz=int(Gc_live.nnz + Gb_live.nnz),
            exact_term_count=int(exact_term_count),
            q_continuous_hat_bytes=packed[0],
            q_continuous_error_bytes=packed[1],
            q_binary_hat_bytes=packed[2],
            q_binary_error_bytes=packed[3],
            kappa_hat=kappa_hat,
            kappa_error=kappa_error,
            platform=platform_reason,
            receipt=frozen_receipt,
            _producer_capability=_HZ_PREFORMED_FACTOR_OBJECTIVE_TOKEN,
        )
        _hz_register_preformed_factor_objective_envelope(
            envelope,
            _producer_capability=_HZ_PREFORMED_FACTOR_OBJECTIVE_TOKEN,
        )
        _hz_split_certificate_deadline(
            deadline, "preformed_formation_before_return"
        )
        return envelope, frozen_receipt
    except _HZSplitBlockCertificateDeadline as exc:
        receipt["status"] = f"deadline_exhausted:{str(exc)[:120]}"
        return None, receipt
    except Exception as exc:
        receipt["status"] = f"invalid:{type(exc).__name__}:{str(exc)[:120]}"
        return None, receipt


def _hz_validate_preformed_factor_objective_envelope(
    envelope,
    *,
    expected_parent_semantic_digest: str,
    expected_exact_objective_sha256: str,
    expected_objective_binding_sha256: str,
):
    """O(1) structural validation of a producer-sealed immutable envelope."""

    if type(envelope) is not _HZPreformedFactorObjectiveEnvelope:
        raise TypeError("preformed objective envelope has the wrong type")
    parent = _hz_preformed_require_sha256(
        expected_parent_semantic_digest,
        name="expected_parent_semantic_digest",
    )
    objective = _hz_preformed_require_sha256(
        expected_exact_objective_sha256,
        name="expected_exact_objective_sha256",
    )
    binding = _hz_preformed_require_sha256(
        expected_objective_binding_sha256,
        name="expected_objective_binding_sha256",
    )
    record = _hz_preformed_factor_objective_registry_record(envelope)
    values = tuple(getattr(envelope, name) for name in envelope.__slots__[:-2])
    original = dict(zip(envelope.__slots__[:-2], record.original_values))
    if (
        envelope._seal is not record.original_seal
        or type(record.original_seal) is not tuple
        or len(record.original_values) != len(values)
        or any(
            current is not original
            for current, original in zip(values, record.original_values)
        )
    ):
        raise PermissionError(
            "preformed objective registry identity seal is invalid"
        )
    if original["_process_id"] != int(os.getpid()):
        raise PermissionError("preformed objective belongs to another process")
    platform_ok, platform_reason = _hz_longdouble_certificate_platform()
    if not platform_ok or platform_reason != original["_platform"]:
        raise RuntimeError("preformed objective platform contract is stale")
    if original["_schema"] != _HZ_PREFORMED_FACTOR_OBJECTIVE_SCHEMA:
        raise ValueError("preformed objective schema mismatch")
    if original["_parent_semantic_digest"] != parent:
        raise ValueError("preformed objective parent digest mismatch")
    if original["_exact_objective_sha256"] != objective:
        raise ValueError("preformed objective exact digest mismatch")
    if original["_objective_binding_sha256"] != binding:
        raise ValueError("preformed objective binding digest mismatch")
    n_continuous = original["_n_continuous"]
    n_binary = original["_n_binary"]
    if (
        not (0 <= n_continuous <= _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_COLUMNS)
        or not (0 <= n_binary <= _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_COLUMNS)
        or n_continuous + n_binary > _HZ_PREFORMED_FACTOR_OBJECTIVE_MAX_COLUMNS
        or len(original["_q_continuous_hat_bytes"]) != 8 * n_continuous
        or len(original["_q_continuous_error_bytes"]) != 8 * n_continuous
        or len(original["_q_binary_hat_bytes"]) != 8 * n_binary
        or len(original["_q_binary_error_bytes"]) != 8 * n_binary
    ):
        raise ValueError("preformed objective packed shape mismatch")
    return (
        np.frombuffer(original["_q_continuous_hat_bytes"], dtype="<f8"),
        np.frombuffer(original["_q_continuous_error_bytes"], dtype="<f8"),
        np.frombuffer(original["_q_binary_hat_bytes"], dtype="<f8"),
        np.frombuffer(original["_q_binary_error_bytes"], dtype="<f8"),
        np.longdouble(original["_kappa_hat"]),
        np.longdouble(original["_kappa_error"]),
        original,
    )


def _hz_read_exact_objective_binding_material_from_factor_envelope(
    envelope,
    *,
    expected_parent_semantic_digest: str,
    expected_objective_id: str,
):
    """Return sealed exact core-binding material without rescanning live G.

    This accessor conveys no proof or verdict authority.  It exists so a
    bundle context can construct the core ``ObjectiveBinding`` from the exact
    Fraction expansion already performed by the private solver formation,
    then cross-check the resulting core digest against the sealed digest.
    """

    if type(envelope) is not _HZPreformedFactorObjectiveEnvelope:
        raise TypeError("preformed objective envelope has the wrong type")
    parent = _hz_preformed_require_sha256(
        expected_parent_semantic_digest,
        name="expected_parent_semantic_digest",
    )
    if type(expected_objective_id) is not str:
        raise ValueError("expected_objective_id must be a strict string")
    # Reuse the complete O(1) private-seal/process/platform validation.  The
    # two objective digests supplied here are themselves sealed fields; the
    # caller independently checks the returned core binding against the
    # binding digest before any numeric replay.
    validated = _hz_validate_preformed_factor_objective_envelope(
        envelope,
        expected_parent_semantic_digest=parent,
        expected_exact_objective_sha256=(
            envelope.exact_objective_sha256
        ),
        expected_objective_binding_sha256=(
            envelope.objective_binding_sha256
        ),
    )
    original = validated[-1]
    if original["_objective_id"] != expected_objective_id:
        raise ValueError("preformed objective id mismatch")
    if (
        type(original["_objective_center_exact"]) is not Fraction
        or type(original["_continuous_terms_exact"]) is not tuple
        or type(original["_binary_terms_exact"]) is not tuple
    ):
        raise PermissionError("preformed exact binding material is malformed")
    return (
        original["_objective_center_exact"],
        original["_continuous_terms_exact"],
        original["_binary_terms_exact"],
        original["_objective_binding_sha256"],
    )


def _hz_independent_lp_lagrangian_upper(
    *,
    c,
    Gc,
    C_row,
    threshold,
    A,
    rl,
    ru,
    lb,
    ub,
    row_dual,
    center_error=None,
):
    """Verify one HZ LP-relaxation upper bound without trusting solver status.

    HiGHS minimizes ``-q @ v`` in :func:`_hz_persistent_lp_filter`, so its row
    dual is negated to obtain the multiplier ``d`` for the maximization
    certificate.  For every feasible ``v`` and every sign-legal finite ``d``::

        kappa + q@v
          <= kappa
             + support_[rl,ru](d)
             + support_[lb,ub](q - A.T@d).

    Stationarity is not assumed: the residual box support makes *any* legal
    multiplier a valid certificate.  The returned upper bound is authoritative
    only as a ``numpy.longdouble``; receipt floats are diagnostics.
    """

    receipt = {
        "schema": _HZ_LP_CERTIFICATE_SCHEMA,
        "status": "not_started",
        "illegal_sign_projected": 0,
        "nonfinite_dual_zeroed": 0,
        "dual_nnz": 0,
        "upper": None,
        "upper_float64_rounding": (
            "toward_positive_infinity_from_longdouble_v1"
        ),
        "objective_guard": None,
        "residual_guard": None,
        "roundoff_guard": None,
    }
    platform_ok, platform_reason = _hz_longdouble_certificate_platform()
    receipt["platform"] = platform_reason
    if not platform_ok:
        receipt["status"] = "platform_unsupported"
        return None, receipt

    try:
        A = _sp.csr_matrix(A, dtype=np.float64, copy=False)
        rl = np.asarray(rl, dtype=np.float64).reshape(-1)
        ru = np.asarray(ru, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        raw_dual = np.asarray(row_dual, dtype=np.float64).reshape(-1)
        if (
            A.shape != (rl.size, lb.size)
            or ru.size != rl.size
            or ub.size != lb.size
            or raw_dual.size != rl.size
        ):
            raise ValueError("LP certificate base shape mismatch")
        if (
            not A.has_canonical_format
            or (A.nnz and not np.all(np.isfinite(A.data)))
            or not np.all(np.isfinite(lb))
            or not np.all(np.isfinite(ub))
            or np.any(lb > ub)
            or np.any(np.isnan(rl))
            or np.any(np.isnan(ru))
            or np.any(rl > ru)
            or np.any(np.isposinf(rl))
            or np.any(np.isneginf(ru))
        ):
            raise ValueError("LP certificate base data are invalid")

        # HiGHS row_dual uses the minimization convention.  Replacing an
        # unusable multiplier by zero can only weaken, never invalidate, the
        # independently recomputed bound.
        d = -raw_dual.copy()
        nonfinite = ~np.isfinite(d)
        receipt["nonfinite_dual_zeroed"] = int(np.count_nonzero(nonfinite))
        d[nonfinite] = 0.0
        upper_only = np.isneginf(rl) & np.isfinite(ru)
        lower_only = np.isfinite(rl) & np.isposinf(ru)
        free_row = np.isneginf(rl) & np.isposinf(ru)
        illegal_upper = upper_only & (d < 0.0)
        illegal_lower = lower_only & (d > 0.0)
        illegal_free = free_row & (d != 0.0)
        receipt["illegal_sign_projected"] = int(np.count_nonzero(
            illegal_upper | illegal_lower | illegal_free
        ))
        d[illegal_upper | illegal_lower | illegal_free] = 0.0
        nonzero = np.flatnonzero(d != 0.0).astype(np.int64, copy=False)
        receipt["dual_nnz"] = int(nonzero.size)

        q_hat, q_error, kappa_hat, kappa_error = (
            _hz_ld_objective_enclosure(
                c,
                Gc,
                C_row,
                threshold,
                center_error=center_error,
            )
        )
        if q_hat.size != lb.size:
            raise ValueError("LP certificate objective/base column mismatch")
        if center_error is not None:
            center_error_ld = np.asarray(
                center_error,
                dtype=np.longdouble,
            ).reshape(-1)
            receipt["center_transform_guard_max"] = float(
                np.max(center_error_ld)
                if center_error_ld.size
                else np.longdouble(0.0)
            )

        dtype = np.longdouble
        inf = dtype(np.inf)
        if nonzero.size:
            d_local = d[nonzero].astype(dtype)
            selected_side = np.where(
                d[nonzero] >= 0.0,
                ru[nonzero],
                rl[nonzero],
            )
            if not np.all(np.isfinite(selected_side)):
                raise ValueError("LP certificate multiplier selected an infinite side")
            row_upper, row_guard = _hz_ld_sum_products_upper(
                d_local,
                selected_side.astype(dtype),
                name="LP certificate row support",
            )
            local_A = A[nonzero, :].tocsr()
            residual_hat = q_hat - np.asarray(
                local_A.transpose() @ d_local,
                dtype=dtype,
            ).reshape(-1)
            abs_A = local_A.copy()
            abs_A.data = np.abs(abs_A.data)
            residual_mass = np.abs(q_hat) + np.asarray(
                abs_A.transpose() @ np.abs(d_local),
                dtype=dtype,
            ).reshape(-1)
            residual_counts = np.bincount(
                local_A.indices,
                minlength=A.shape[1],
            ).astype(np.int64, copy=False)
            residual_arithmetic_error = _hz_ld_roundoff_guard(
                residual_mass,
                2 * residual_counts + 6,
                name="LP certificate residual",
            )
        else:
            row_upper = dtype(0.0)
            row_guard = dtype(0.0)
            residual_hat = q_hat.copy()
            residual_arithmetic_error = np.zeros(q_hat.size, dtype=dtype)

        residual_error = np.nextafter(
            residual_arithmetic_error + q_error,
            inf,
        )
        if (
            not np.all(np.isfinite(residual_hat))
            or not np.all(np.isfinite(residual_error))
        ):
            raise ValueError("LP certificate residual enclosure is non-finite")
        lb_ld = lb.astype(dtype)
        ub_ld = ub.astype(dtype)
        selected_box_side = np.where(residual_hat >= 0.0, ub_ld, lb_ld)
        box_upper, box_guard = _hz_ld_sum_products_upper(
            residual_hat,
            selected_box_side,
            name="LP certificate residual box support",
        )
        box_radius = np.maximum(np.abs(lb_ld), np.abs(ub_ld))
        uncertainty_upper, uncertainty_guard = _hz_ld_sum_products_upper(
            residual_error,
            box_radius,
            name="LP certificate residual uncertainty",
        )
        objective_guard_upper, _ = _hz_ld_sum_products_upper(
            q_error,
            box_radius,
            name="LP certificate objective-map uncertainty",
        )
        pieces = np.asarray(
            [
                kappa_hat,
                kappa_error,
                row_upper,
                box_upper,
                uncertainty_upper,
            ],
            dtype=dtype,
        )
        upper, final_guard = _hz_ld_sum_products_upper(
            np.ones(pieces.size, dtype=dtype),
            pieces,
            name="LP certificate final upper",
        )
        if not np.isfinite(upper):
            raise ValueError("LP certificate final upper is non-finite")

        total_guard = np.nextafter(
            kappa_error
            + row_guard
            + box_guard
            + uncertainty_guard
            + final_guard,
            inf,
        )
        receipt.update({
            "status": "verified_upper",
            "upper": _hz_longdouble_to_outward_float64_upper(upper),
            "objective_guard": float(np.nextafter(
                kappa_error + objective_guard_upper,
                inf,
            )),
            "residual_guard": float(
                np.max(residual_arithmetic_error)
                if residual_arithmetic_error.size
                else dtype(0.0)
            ),
            "roundoff_guard": float(total_guard),
            "longdouble_nmant": int(np.finfo(dtype).nmant),
            "longdouble_eps": float(np.finfo(dtype).eps),
        })
        return upper, receipt
    except Exception as exc:
        receipt["status"] = f"invalid:{type(exc).__name__}:{str(exc)[:120]}"
        return None, receipt


def _hz_independent_split_block_lp_lagrangian_upper(
    *,
    c,
    Gc,
    Gb,
    C_row,
    threshold,
    Auc,
    Aub,
    Ac,
    Ab,
    ub,
    b,
    continuous_lb,
    continuous_ub,
    binary_lb,
    binary_ub,
    upper_row_dual,
    equality_row_dual,
    center_error=None,
    deadline=None,
):
    """Verify an LP upper certificate without assembling a full LP frame.

    This is the low-peak counterpart of
    :func:`_hz_independent_lp_lagrangian_upper`.  It consumes the native HZ
    blocks directly::

        Auc*xc + Aub*xb <= ub
        Ac *xc + Ab *xb == b

    and keeps continuous/binary columns and upper/equality rows split for the
    entire verification.  In particular it never calls sparse ``hstack`` or
    ``vstack`` and never constructs ``[Gc,Gb]`` or the combined constraint
    CSR.  All input sparse blocks must already be canonical binary64 CSR; this
    strict contract prevents a hidden dtype/format conversion from recreating
    the memory peak that the API is designed to remove.

    ``upper_row_dual`` and ``equality_row_dual`` use HiGHS' minimization-dual
    convention.  The checker negates them to obtain maximization multipliers.
    Non-finite multipliers and sign-illegal upper-row multipliers are replaced
    by zero.  This can only weaken the independently recomputed upper bound.
    Solver status, objective values, and primal points have no authority.

    A finite ``deadline`` is optional.  Expiry at any guarded stage returns no
    bound; a partially accumulated value is never exposed.
    """

    receipt = {
        "schema": _HZ_SPLIT_BLOCK_LP_CERTIFICATE_SCHEMA,
        "status": "not_started",
        "route": "native_hz_split_csr_blocks_no_stack_v1",
        "illegal_sign_projected": 0,
        "nonfinite_dual_zeroed": 0,
        "dual_nnz": 0,
        "upper": None,
        "upper_float64_rounding": (
            "toward_positive_infinity_from_longdouble_v1"
        ),
        "objective_guard": None,
        "residual_guard": None,
        "roundoff_guard": None,
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "assembled_sparse_nnz": 0,
        "input_sparse_nnz": None,
        "temporary_nonzero_chunk_cap": 65536,
        "analytical_dense_workspace_bytes_ceiling": None,
    }
    platform_ok, platform_reason = _hz_longdouble_certificate_platform()
    receipt["platform"] = platform_reason
    if not platform_ok:
        receipt["status"] = "platform_unsupported"
        return None, receipt

    try:
        if deadline is not None:
            if (
                isinstance(deadline, (bool, np.bool_))
                or not np.isscalar(deadline)
                or not np.isfinite(float(deadline))
            ):
                raise ValueError(
                    "split LP certificate deadline must be finite"
                )
            deadline = float(deadline)
        _hz_split_certificate_deadline(deadline, "entry")

        center = np.asarray(c, dtype=np.float64).reshape(-1)
        objective_row = np.asarray(
            C_row, dtype=np.float64
        ).reshape(-1)
        threshold = float(threshold)
        if (
            center.size != objective_row.size
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(objective_row))
            or not np.isfinite(threshold)
        ):
            raise ValueError(
                "split LP certificate objective data are invalid"
            )

        continuous_lower = np.asarray(
            continuous_lb, dtype=np.float64
        ).reshape(-1)
        continuous_upper = np.asarray(
            continuous_ub, dtype=np.float64
        ).reshape(-1)
        binary_lower = np.asarray(
            binary_lb, dtype=np.float64
        ).reshape(-1)
        binary_upper = np.asarray(
            binary_ub, dtype=np.float64
        ).reshape(-1)
        n_continuous = int(continuous_lower.size)
        n_binary = int(binary_lower.size)
        if (
            continuous_upper.size != n_continuous
            or binary_upper.size != n_binary
            or not np.all(np.isfinite(continuous_lower))
            or not np.all(np.isfinite(continuous_upper))
            or not np.all(np.isfinite(binary_lower))
            or not np.all(np.isfinite(binary_upper))
            or np.any(continuous_lower > continuous_upper)
            or np.any(binary_lower > binary_upper)
        ):
            raise ValueError(
                "split LP certificate variable bounds are invalid"
            )

        upper_bound = np.asarray(ub, dtype=np.float64).reshape(-1)
        equality_bound = np.asarray(b, dtype=np.float64).reshape(-1)
        n_upper = int(upper_bound.size)
        n_equality = int(equality_bound.size)
        if (
            not np.all(np.isfinite(upper_bound))
            or not np.all(np.isfinite(equality_bound))
        ):
            raise ValueError(
                "split LP certificate row bounds are invalid"
            )

        Gc = _hz_split_certificate_csr(
            Gc,
            rows=center.size,
            columns=n_continuous,
            name="Gc",
        )
        Gb = _hz_split_certificate_csr(
            Gb,
            rows=center.size,
            columns=n_binary,
            name="Gb",
        )
        Auc = _hz_split_certificate_csr(
            Auc,
            rows=n_upper,
            columns=n_continuous,
            name="Auc",
        )
        Aub = _hz_split_certificate_csr(
            Aub,
            rows=n_upper,
            columns=n_binary,
            name="Aub",
        )
        Ac = _hz_split_certificate_csr(
            Ac,
            rows=n_equality,
            columns=n_continuous,
            name="Ac",
        )
        Ab = _hz_split_certificate_csr(
            Ab,
            rows=n_equality,
            columns=n_binary,
            name="Ab",
        )
        receipt["input_sparse_nnz"] = int(
            Gc.nnz
            + Gb.nnz
            + Auc.nnz
            + Aub.nnz
            + Ac.nnz
            + Ab.nnz
        )
        receipt["block_shapes"] = {
            "Gc": [int(x) for x in Gc.shape],
            "Gb": [int(x) for x in Gb.shape],
            "Auc": [int(x) for x in Auc.shape],
            "Aub": [int(x) for x in Aub.shape],
            "Ac": [int(x) for x in Ac.shape],
            "Ab": [int(x) for x in Ab.shape],
        }

        raw_upper_dual = np.asarray(
            upper_row_dual, dtype=np.float64
        ).reshape(-1)
        raw_equality_dual = np.asarray(
            equality_row_dual, dtype=np.float64
        ).reshape(-1)
        if (
            raw_upper_dual.size != n_upper
            or raw_equality_dual.size != n_equality
        ):
            raise ValueError(
                "split LP certificate dual shape mismatch"
            )
        d_upper = -raw_upper_dual.copy()
        d_equality = -raw_equality_dual.copy()
        nonfinite_upper = ~np.isfinite(d_upper)
        nonfinite_equality = ~np.isfinite(d_equality)
        receipt["nonfinite_dual_zeroed"] = int(
            np.count_nonzero(nonfinite_upper)
            + np.count_nonzero(nonfinite_equality)
        )
        d_upper[nonfinite_upper] = 0.0
        d_equality[nonfinite_equality] = 0.0
        illegal_upper = d_upper < 0.0
        receipt["illegal_sign_projected"] = int(
            np.count_nonzero(illegal_upper)
        )
        d_upper[illegal_upper] = 0.0
        receipt["dual_nnz"] = int(
            np.count_nonzero(d_upper)
            + np.count_nonzero(d_equality)
        )

        # This ceiling counts every dense long-double/int64 block that may be
        # live together plus two bounded chunk temporaries.  It depends on
        # topology, never on sparse nnz; the source CSR storage is borrowed.
        total_columns = n_continuous + n_binary
        total_rows = n_upper + n_equality
        chunk_cap = int(receipt["temporary_nonzero_chunk_cap"])
        receipt["analytical_dense_workspace_bytes_ceiling"] = int(
            176 * total_columns
            + 40 * total_rows
            + 40 * center.size
            + 40 * min(chunk_cap, max(1, total_columns))
            + 1048576
        )
        _hz_split_certificate_deadline(deadline, "after_validation")

        dtype = np.longdouble
        inf = dtype(np.inf)
        objective_weights = objective_row.astype(dtype)
        (
            q_continuous,
            q_continuous_mass,
            q_continuous_counts,
            q_continuous_error,
        ) = _hz_ld_sparse_weighted_columns_split(
            Gc,
            objective_weights,
            name="split LP continuous objective coefficients",
            deadline=deadline,
            chunk_nonzeros=chunk_cap,
        )
        del q_continuous_mass, q_continuous_counts
        (
            q_binary,
            q_binary_mass,
            q_binary_counts,
            q_binary_error,
        ) = _hz_ld_sparse_weighted_columns_split(
            Gb,
            objective_weights,
            name="split LP binary objective coefficients",
            deadline=deadline,
            chunk_nonzeros=chunk_cap,
        )
        del q_binary_mass, q_binary_counts

        selected_outputs = np.flatnonzero(
            objective_row != 0.0
        ).astype(np.int64, copy=False)
        center_products = (
            objective_row[selected_outputs].astype(dtype)
            * center[selected_outputs].astype(dtype)
        )
        if not np.all(np.isfinite(center_products)):
            raise ValueError(
                "split LP certificate objective center overflowed"
            )
        kappa_hat = (
            np.sum(center_products, dtype=dtype) - dtype(threshold)
        )
        kappa_mass = (
            np.sum(np.abs(center_products), dtype=dtype)
            + np.abs(dtype(threshold))
        )
        kappa_arithmetic_error = _hz_ld_roundoff_guard(
            kappa_mass,
            2 * int(selected_outputs.size) + 6,
            name="split LP objective constant",
        ).reshape(())
        if center_error is None:
            center_error_ld = None
            kappa_error = kappa_arithmetic_error
        else:
            center_error_ld = np.asarray(
                center_error, dtype=dtype
            ).reshape(-1)
            if (
                center_error_ld.size != center.size
                or not np.all(np.isfinite(center_error_ld))
                or np.any(center_error_ld < 0.0)
            ):
                raise ValueError(
                    "split LP certificate center error is invalid"
                )
            center_uncertainty, _ = _hz_ld_sum_products_upper(
                np.abs(
                    objective_row[selected_outputs].astype(dtype)
                ),
                center_error_ld[selected_outputs],
                name="split LP center transformation uncertainty",
            )
            kappa_error = np.nextafter(
                kappa_arithmetic_error + center_uncertainty,
                inf,
            )
            receipt["center_transform_guard_max"] = float(
                np.max(center_error_ld)
                if center_error_ld.size
                else dtype(0.0)
            )
        if not np.isfinite(kappa_hat) or not np.isfinite(kappa_error):
            raise ValueError(
                "split LP certificate objective constant is non-finite"
            )

        def _residual_block(
            q_hat,
            q_error,
            upper_matrix,
            equality_matrix,
            *,
            block_name: str,
        ):
            residual = q_hat.copy()
            combined_mass = np.nextafter(
                np.abs(q_hat), inf
            )
            combined_counts = np.zeros(q_hat.size, dtype=np.int64)
            accumulated_error = np.zeros(q_hat.size, dtype=dtype)
            for matrix, multipliers, row_kind in (
                (upper_matrix, d_upper, "upper"),
                (equality_matrix, d_equality, "equality"),
            ):
                estimate, mass, counts, arithmetic_error = (
                    _hz_ld_sparse_weighted_columns_split(
                        matrix,
                        multipliers.astype(dtype),
                        name=(
                            f"split LP {block_name} {row_kind} A.T d"
                        ),
                        deadline=deadline,
                        chunk_nonzeros=chunk_cap,
                    )
                )
                residual -= estimate
                combined_mass = np.nextafter(
                    combined_mass + mass, inf
                )
                combined_counts += counts
                accumulated_error = np.nextafter(
                    accumulated_error + arithmetic_error, inf
                )
                del estimate, mass, counts, arithmetic_error
            combination_guard = _hz_ld_roundoff_guard(
                combined_mass,
                2 * combined_counts + 16,
                name=f"split LP {block_name} residual combination",
            )
            residual_arithmetic_error = np.nextafter(
                accumulated_error + combination_guard, inf
            )
            residual_error = np.nextafter(
                residual_arithmetic_error + q_error, inf
            )
            if (
                not np.all(np.isfinite(residual))
                or not np.all(np.isfinite(residual_error))
            ):
                raise ValueError(
                    f"split LP {block_name} residual is non-finite"
                )
            return residual, residual_error, residual_arithmetic_error

        (
            residual_continuous,
            residual_continuous_error,
            residual_continuous_arithmetic_error,
        ) = _residual_block(
            q_continuous,
            q_continuous_error,
            Auc,
            Ac,
            block_name="continuous",
        )
        (
            residual_binary,
            residual_binary_error,
            residual_binary_arithmetic_error,
        ) = _residual_block(
            q_binary,
            q_binary_error,
            Aub,
            Ab,
            block_name="binary",
        )
        _hz_split_certificate_deadline(deadline, "after_residuals")

        def _row_support(multipliers, side, *, support_name: str):
            nonzero = np.flatnonzero(multipliers != 0.0).astype(
                np.int64, copy=False
            )
            if nonzero.size == 0:
                return dtype(0.0), dtype(0.0), dtype(0.0), 0
            local_multiplier = multipliers[nonzero].astype(dtype)
            local_side = side[nonzero].astype(dtype)
            upper_value, guard = _hz_ld_sum_products_upper(
                local_multiplier,
                local_side,
                name=support_name,
            )
            mass_upper, _ = _hz_ld_sum_products_upper(
                np.abs(local_multiplier),
                np.abs(local_side),
                name=f"{support_name} absolute mass",
            )
            return upper_value, guard, mass_upper, int(nonzero.size)

        (
            upper_row_support,
            upper_row_guard,
            upper_row_mass,
            upper_row_count,
        ) = _row_support(
            d_upper,
            upper_bound,
            support_name="split LP upper-row support",
        )
        (
            equality_row_support,
            equality_row_guard,
            equality_row_mass,
            equality_row_count,
        ) = _row_support(
            d_equality,
            equality_bound,
            support_name="split LP equality-row support",
        )
        row_mass = np.nextafter(
            upper_row_mass + equality_row_mass, inf
        )
        row_cross_guard = _hz_ld_roundoff_guard(
            row_mass,
            2 * (upper_row_count + equality_row_count) + 16,
            name="split LP cross-block row support",
        ).reshape(())
        row_upper = np.nextafter(
            upper_row_support
            + equality_row_support
            + row_cross_guard,
            inf,
        )

        def _box_support(
            residual,
            residual_error,
            q_error,
            lower,
            upper,
            *,
            block_name: str,
        ):
            lower_ld = lower.astype(dtype)
            upper_ld = upper.astype(dtype)
            selected_side = np.where(
                residual >= 0.0, upper_ld, lower_ld
            )
            box_upper, box_guard = _hz_ld_sum_products_upper(
                residual,
                selected_side,
                name=f"split LP {block_name} box support",
            )
            box_mass, _ = _hz_ld_sum_products_upper(
                np.abs(residual),
                np.abs(selected_side),
                name=f"split LP {block_name} box support mass",
            )
            radius = np.maximum(np.abs(lower_ld), np.abs(upper_ld))
            uncertainty_upper, uncertainty_guard = (
                _hz_ld_sum_products_upper(
                    residual_error,
                    radius,
                    name=(
                        f"split LP {block_name} residual uncertainty"
                    ),
                )
            )
            uncertainty_mass, _ = _hz_ld_sum_products_upper(
                residual_error,
                radius,
                name=(
                    f"split LP {block_name} residual uncertainty mass"
                ),
            )
            objective_uncertainty, _ = _hz_ld_sum_products_upper(
                q_error,
                radius,
                name=(
                    f"split LP {block_name} objective-map uncertainty"
                ),
            )
            return (
                box_upper,
                box_guard,
                box_mass,
                uncertainty_upper,
                uncertainty_guard,
                uncertainty_mass,
                objective_uncertainty,
            )

        continuous_box = _box_support(
            residual_continuous,
            residual_continuous_error,
            q_continuous_error,
            continuous_lower,
            continuous_upper,
            block_name="continuous",
        )
        binary_box = _box_support(
            residual_binary,
            residual_binary_error,
            q_binary_error,
            binary_lower,
            binary_upper,
            block_name="binary",
        )
        box_cross_guard = _hz_ld_roundoff_guard(
            np.nextafter(continuous_box[2] + binary_box[2], inf),
            2 * total_columns + 16,
            name="split LP cross-block box support",
        ).reshape(())
        uncertainty_cross_guard = _hz_ld_roundoff_guard(
            np.nextafter(continuous_box[5] + binary_box[5], inf),
            2 * total_columns + 16,
            name="split LP cross-block residual uncertainty",
        ).reshape(())

        pieces = np.asarray(
            [
                kappa_hat,
                kappa_error,
                row_upper,
                continuous_box[0],
                binary_box[0],
                box_cross_guard,
                continuous_box[3],
                binary_box[3],
                uncertainty_cross_guard,
            ],
            dtype=dtype,
        )
        upper, final_guard = _hz_ld_sum_products_upper(
            np.ones(pieces.size, dtype=dtype),
            pieces,
            name="split LP final upper",
        )
        if not np.isfinite(upper):
            raise ValueError(
                "split LP certificate final upper is non-finite"
            )
        _hz_split_certificate_deadline(deadline, "before_authorization")

        total_guard = np.nextafter(
            kappa_error
            + upper_row_guard
            + equality_row_guard
            + row_cross_guard
            + continuous_box[1]
            + binary_box[1]
            + box_cross_guard
            + continuous_box[4]
            + binary_box[4]
            + uncertainty_cross_guard
            + final_guard,
            inf,
        )
        objective_guard = np.nextafter(
            kappa_error
            + continuous_box[6]
            + binary_box[6],
            inf,
        )
        residual_guard = max(
            np.max(residual_continuous_arithmetic_error)
            if residual_continuous_arithmetic_error.size
            else dtype(0.0),
            np.max(residual_binary_arithmetic_error)
            if residual_binary_arithmetic_error.size
            else dtype(0.0),
        )
        receipt.update({
            "status": "verified_upper",
            "upper": _hz_longdouble_to_outward_float64_upper(upper),
            "objective_guard": float(objective_guard),
            "residual_guard": float(residual_guard),
            "roundoff_guard": float(total_guard),
            "longdouble_nmant": int(np.finfo(dtype).nmant),
            "longdouble_eps": float(np.finfo(dtype).eps),
        })
        return upper, receipt
    except _HZSplitBlockCertificateDeadline as exc:
        receipt["status"] = f"deadline_exhausted:{str(exc)[:120]}"
        return None, receipt
    except Exception as exc:
        receipt["status"] = (
            f"invalid:{type(exc).__name__}:{str(exc)[:120]}"
        )
        return None, receipt


def _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope(
    *,
    objective_envelope,
    expected_parent_semantic_digest,
    expected_exact_objective_sha256,
    expected_objective_binding_sha256,
    Auc,
    Aub,
    Ac,
    Ab,
    ub,
    b,
    continuous_lb,
    continuous_ub,
    binary_lb,
    binary_ub,
    upper_row_dual,
    equality_row_dual,
    deadline=None,
):
    """Replay a split-block upper from a sealed preformed objective.

    This route has no ``c/Gc/Gb/C/threshold`` arguments and consequently
    cannot revisit the live generator blocks.  Only objective formation is
    reused: constraints, raw duals, residuals, support functions, roundoff
    guards, and the final outward binary64 upper are independently recomputed
    for every call.  The legacy split checker above is intentionally left
    untouched.
    """

    receipt = {
        "schema": _HZ_PREFORMED_SPLIT_BLOCK_LP_CERTIFICATE_SCHEMA,
        "status": "not_started",
        "route": (
            "native_hz_preformed_objective_split_csr_"
            "no_generator_read_v1"
        ),
        "proof_authority": False,
        "verdict_authority": False,
        "pcoh_authorization": False,
        "illegal_sign_projected": 0,
        "nonfinite_dual_zeroed": 0,
        "dual_nnz": 0,
        "upper": None,
        "upper_float64_rounding": (
            "toward_positive_infinity_from_longdouble_v1"
        ),
        "objective_guard": None,
        "residual_guard": None,
        "roundoff_guard": None,
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "assembled_sparse_nnz": 0,
        "input_constraint_sparse_nnz": None,
        "generator_source_read_count": 0,
        "envelope_rehash_bytes": 0,
        "temporary_nonzero_chunk_cap": 65536,
        "analytical_dense_workspace_bytes_ceiling": None,
        "packed_factor_persistent_bytes": None,
        "packed_factor_persistent_bytes_lower_bound_only": True,
        "total_persistent_bytes_bounded": False,
        "total_persistent_bytes_blocker": (
            "python_fraction_exact_binding_material_v1"
        ),
        "trust_boundary": (
            "process_local_registry_and_solver_module_state_trusted_v1"
        ),
    }
    platform_ok, platform_reason = _hz_longdouble_certificate_platform()
    receipt["platform"] = platform_reason
    if not platform_ok:
        receipt["status"] = "platform_unsupported"
        return None, receipt

    try:
        if deadline is not None:
            if (
                isinstance(deadline, (bool, np.bool_))
                or not np.isscalar(deadline)
                or not np.isfinite(float(deadline))
            ):
                raise ValueError(
                    "preformed split LP certificate deadline must be finite"
                )
            deadline = float(deadline)
        _hz_split_certificate_deadline(deadline, "preformed_checker_entry")

        (
            q_continuous_f64,
            q_continuous_error_f64,
            q_binary_f64,
            q_binary_error_f64,
            kappa_hat,
            kappa_error,
            sealed_fields,
        ) = _hz_validate_preformed_factor_objective_envelope(
            objective_envelope,
            expected_parent_semantic_digest=(
                expected_parent_semantic_digest
            ),
            expected_exact_objective_sha256=(
                expected_exact_objective_sha256
            ),
            expected_objective_binding_sha256=(
                expected_objective_binding_sha256
            ),
        )
        n_continuous = int(sealed_fields["_n_continuous"])
        n_binary = int(sealed_fields["_n_binary"])
        receipt.update({
            "parent_semantic_digest": sealed_fields[
                "_parent_semantic_digest"
            ],
            "objective_source_sha256": sealed_fields[
                "_objective_source_sha256"
            ],
            "stable_ids_sha256": sealed_fields["_stable_ids_sha256"],
            "exact_objective_sha256": sealed_fields[
                "_exact_objective_sha256"
            ],
            "objective_binding_sha256": sealed_fields[
                "_objective_binding_sha256"
            ],
            "objective_binding_cross_checked": True,
            "objective_envelope_sha256": sealed_fields[
                "_envelope_sha256"
            ],
            "packed_factor_persistent_bytes": int(
                16 * (n_continuous + n_binary)
            ),
            "source_generator_nnz": int(
                sealed_fields["_source_generator_nnz"]
            ),
            "objective_formation_reused": True,
        })

        continuous_lower = np.asarray(
            continuous_lb, dtype=np.float64
        ).reshape(-1)
        continuous_upper = np.asarray(
            continuous_ub, dtype=np.float64
        ).reshape(-1)
        binary_lower = np.asarray(
            binary_lb, dtype=np.float64
        ).reshape(-1)
        binary_upper = np.asarray(
            binary_ub, dtype=np.float64
        ).reshape(-1)
        if (
            continuous_lower.size != n_continuous
            or continuous_upper.size != n_continuous
            or binary_lower.size != n_binary
            or binary_upper.size != n_binary
            or not np.all(np.isfinite(continuous_lower))
            or not np.all(np.isfinite(continuous_upper))
            or not np.all(np.isfinite(binary_lower))
            or not np.all(np.isfinite(binary_upper))
            or np.any(continuous_lower > continuous_upper)
            or np.any(binary_lower > binary_upper)
        ):
            raise ValueError(
                "preformed split LP certificate variable bounds are invalid"
            )

        upper_bound = np.asarray(ub, dtype=np.float64).reshape(-1)
        equality_bound = np.asarray(b, dtype=np.float64).reshape(-1)
        n_upper = int(upper_bound.size)
        n_equality = int(equality_bound.size)
        if (
            not np.all(np.isfinite(upper_bound))
            or not np.all(np.isfinite(equality_bound))
        ):
            raise ValueError(
                "preformed split LP certificate row bounds are invalid"
            )

        Auc = _hz_split_certificate_csr(
            Auc,
            rows=n_upper,
            columns=n_continuous,
            name="preformed_Auc",
        )
        Aub = _hz_split_certificate_csr(
            Aub,
            rows=n_upper,
            columns=n_binary,
            name="preformed_Aub",
        )
        Ac = _hz_split_certificate_csr(
            Ac,
            rows=n_equality,
            columns=n_continuous,
            name="preformed_Ac",
        )
        Ab = _hz_split_certificate_csr(
            Ab,
            rows=n_equality,
            columns=n_binary,
            name="preformed_Ab",
        )
        receipt["input_constraint_sparse_nnz"] = int(
            Auc.nnz + Aub.nnz + Ac.nnz + Ab.nnz
        )
        receipt["block_shapes"] = {
            "Auc": [int(x) for x in Auc.shape],
            "Aub": [int(x) for x in Aub.shape],
            "Ac": [int(x) for x in Ac.shape],
            "Ab": [int(x) for x in Ab.shape],
        }

        raw_upper_dual = np.asarray(
            upper_row_dual, dtype=np.float64
        ).reshape(-1)
        raw_equality_dual = np.asarray(
            equality_row_dual, dtype=np.float64
        ).reshape(-1)
        if (
            raw_upper_dual.size != n_upper
            or raw_equality_dual.size != n_equality
        ):
            raise ValueError(
                "preformed split LP certificate dual shape mismatch"
            )
        d_upper = -raw_upper_dual.copy()
        d_equality = -raw_equality_dual.copy()
        nonfinite_upper = ~np.isfinite(d_upper)
        nonfinite_equality = ~np.isfinite(d_equality)
        receipt["nonfinite_dual_zeroed"] = int(
            np.count_nonzero(nonfinite_upper)
            + np.count_nonzero(nonfinite_equality)
        )
        d_upper[nonfinite_upper] = 0.0
        d_equality[nonfinite_equality] = 0.0
        illegal_upper = d_upper < 0.0
        receipt["illegal_sign_projected"] = int(
            np.count_nonzero(illegal_upper)
        )
        d_upper[illegal_upper] = 0.0
        receipt["dual_nnz"] = int(
            np.count_nonzero(d_upper)
            + np.count_nonzero(d_equality)
        )

        total_columns = n_continuous + n_binary
        total_rows = n_upper + n_equality
        chunk_cap = int(receipt["temporary_nonzero_chunk_cap"])
        receipt["analytical_dense_workspace_bytes_ceiling"] = int(
            176 * total_columns
            + 40 * total_rows
            + 40 * min(chunk_cap, max(1, total_columns))
            + 1048576
        )
        _hz_split_certificate_deadline(
            deadline, "preformed_checker_after_validation"
        )

        dtype = np.longdouble
        inf = dtype(np.inf)
        q_continuous = np.asarray(
            q_continuous_f64, dtype=dtype
        ).reshape(-1)
        q_continuous_error = np.asarray(
            q_continuous_error_f64, dtype=dtype
        ).reshape(-1)
        q_binary = np.asarray(q_binary_f64, dtype=dtype).reshape(-1)
        q_binary_error = np.asarray(
            q_binary_error_f64, dtype=dtype
        ).reshape(-1)
        if (
            not np.all(np.isfinite(q_continuous))
            or not np.all(np.isfinite(q_continuous_error))
            or not np.all(np.isfinite(q_binary))
            or not np.all(np.isfinite(q_binary_error))
            or np.any(q_continuous_error < 0.0)
            or np.any(q_binary_error < 0.0)
            or not np.isfinite(kappa_hat)
            or not np.isfinite(kappa_error)
            or kappa_error < 0.0
        ):
            raise ValueError(
                "preformed split LP objective enclosure is invalid"
            )

        def _residual_block(
            q_hat,
            q_error,
            upper_matrix,
            equality_matrix,
            *,
            block_name: str,
        ):
            residual = q_hat.copy()
            combined_mass = np.nextafter(np.abs(q_hat), inf)
            combined_counts = np.zeros(q_hat.size, dtype=np.int64)
            accumulated_error = np.zeros(q_hat.size, dtype=dtype)
            for matrix, multipliers, row_kind in (
                (upper_matrix, d_upper, "upper"),
                (equality_matrix, d_equality, "equality"),
            ):
                estimate, mass, counts, arithmetic_error = (
                    _hz_ld_sparse_weighted_columns_split(
                        matrix,
                        multipliers.astype(dtype),
                        name=(
                            f"preformed split LP {block_name} "
                            f"{row_kind} A.T d"
                        ),
                        deadline=deadline,
                        chunk_nonzeros=chunk_cap,
                    )
                )
                residual -= estimate
                combined_mass = np.nextafter(combined_mass + mass, inf)
                combined_counts += counts
                accumulated_error = np.nextafter(
                    accumulated_error + arithmetic_error, inf
                )
                del estimate, mass, counts, arithmetic_error
            combination_guard = _hz_ld_roundoff_guard(
                combined_mass,
                2 * combined_counts + 16,
                name=f"preformed split LP {block_name} residual combination",
            )
            residual_arithmetic_error = np.nextafter(
                accumulated_error + combination_guard, inf
            )
            residual_error = np.nextafter(
                residual_arithmetic_error + q_error, inf
            )
            if (
                not np.all(np.isfinite(residual))
                or not np.all(np.isfinite(residual_error))
            ):
                raise ValueError(
                    f"preformed split LP {block_name} residual is non-finite"
                )
            return residual, residual_error, residual_arithmetic_error

        (
            residual_continuous,
            residual_continuous_error,
            residual_continuous_arithmetic_error,
        ) = _residual_block(
            q_continuous,
            q_continuous_error,
            Auc,
            Ac,
            block_name="continuous",
        )
        (
            residual_binary,
            residual_binary_error,
            residual_binary_arithmetic_error,
        ) = _residual_block(
            q_binary,
            q_binary_error,
            Aub,
            Ab,
            block_name="binary",
        )
        _hz_split_certificate_deadline(
            deadline, "preformed_checker_after_residuals"
        )

        def _row_support(multipliers, side, *, support_name: str):
            nonzero = np.flatnonzero(multipliers != 0.0).astype(
                np.int64, copy=False
            )
            if nonzero.size == 0:
                return dtype(0.0), dtype(0.0), dtype(0.0), 0
            local_multiplier = multipliers[nonzero].astype(dtype)
            local_side = side[nonzero].astype(dtype)
            upper_value, guard = _hz_ld_sum_products_upper(
                local_multiplier,
                local_side,
                name=support_name,
            )
            mass_upper, _ = _hz_ld_sum_products_upper(
                np.abs(local_multiplier),
                np.abs(local_side),
                name=f"{support_name} absolute mass",
            )
            return upper_value, guard, mass_upper, int(nonzero.size)

        (
            upper_row_support,
            upper_row_guard,
            upper_row_mass,
            upper_row_count,
        ) = _row_support(
            d_upper,
            upper_bound,
            support_name="preformed split LP upper-row support",
        )
        (
            equality_row_support,
            equality_row_guard,
            equality_row_mass,
            equality_row_count,
        ) = _row_support(
            d_equality,
            equality_bound,
            support_name="preformed split LP equality-row support",
        )
        row_mass = np.nextafter(
            upper_row_mass + equality_row_mass, inf
        )
        row_cross_guard = _hz_ld_roundoff_guard(
            row_mass,
            2 * (upper_row_count + equality_row_count) + 16,
            name="preformed split LP cross-block row support",
        ).reshape(())
        row_upper = np.nextafter(
            upper_row_support + equality_row_support + row_cross_guard,
            inf,
        )

        def _box_support(
            residual,
            residual_error,
            q_error,
            lower,
            upper,
            *,
            block_name: str,
        ):
            lower_ld = lower.astype(dtype)
            upper_ld = upper.astype(dtype)
            selected_side = np.where(residual >= 0.0, upper_ld, lower_ld)
            box_upper, box_guard = _hz_ld_sum_products_upper(
                residual,
                selected_side,
                name=f"preformed split LP {block_name} box support",
            )
            box_mass, _ = _hz_ld_sum_products_upper(
                np.abs(residual),
                np.abs(selected_side),
                name=(
                    f"preformed split LP {block_name} box support mass"
                ),
            )
            radius = np.maximum(np.abs(lower_ld), np.abs(upper_ld))
            uncertainty_upper, uncertainty_guard = (
                _hz_ld_sum_products_upper(
                    residual_error,
                    radius,
                    name=(
                        f"preformed split LP {block_name} "
                        "residual uncertainty"
                    ),
                )
            )
            uncertainty_mass, _ = _hz_ld_sum_products_upper(
                residual_error,
                radius,
                name=(
                    f"preformed split LP {block_name} "
                    "residual uncertainty mass"
                ),
            )
            objective_uncertainty, _ = _hz_ld_sum_products_upper(
                q_error,
                radius,
                name=(
                    f"preformed split LP {block_name} "
                    "objective-map uncertainty"
                ),
            )
            return (
                box_upper,
                box_guard,
                box_mass,
                uncertainty_upper,
                uncertainty_guard,
                uncertainty_mass,
                objective_uncertainty,
            )

        continuous_box = _box_support(
            residual_continuous,
            residual_continuous_error,
            q_continuous_error,
            continuous_lower,
            continuous_upper,
            block_name="continuous",
        )
        binary_box = _box_support(
            residual_binary,
            residual_binary_error,
            q_binary_error,
            binary_lower,
            binary_upper,
            block_name="binary",
        )
        box_cross_guard = _hz_ld_roundoff_guard(
            np.nextafter(continuous_box[2] + binary_box[2], inf),
            2 * total_columns + 16,
            name="preformed split LP cross-block box support",
        ).reshape(())
        uncertainty_cross_guard = _hz_ld_roundoff_guard(
            np.nextafter(continuous_box[5] + binary_box[5], inf),
            2 * total_columns + 16,
            name=(
                "preformed split LP cross-block residual uncertainty"
            ),
        ).reshape(())

        pieces = np.asarray(
            [
                kappa_hat,
                kappa_error,
                row_upper,
                continuous_box[0],
                binary_box[0],
                box_cross_guard,
                continuous_box[3],
                binary_box[3],
                uncertainty_cross_guard,
            ],
            dtype=dtype,
        )
        upper, final_guard = _hz_ld_sum_products_upper(
            np.ones(pieces.size, dtype=dtype),
            pieces,
            name="preformed split LP final upper",
        )
        if not np.isfinite(upper):
            raise ValueError(
                "preformed split LP certificate final upper is non-finite"
            )
        _hz_split_certificate_deadline(
            deadline, "preformed_checker_before_authorization"
        )

        total_guard = np.nextafter(
            kappa_error
            + upper_row_guard
            + equality_row_guard
            + row_cross_guard
            + continuous_box[1]
            + binary_box[1]
            + box_cross_guard
            + continuous_box[4]
            + binary_box[4]
            + uncertainty_cross_guard
            + final_guard,
            inf,
        )
        objective_guard = np.nextafter(
            kappa_error + continuous_box[6] + binary_box[6], inf
        )
        residual_guard = max(
            np.max(residual_continuous_arithmetic_error)
            if residual_continuous_arithmetic_error.size
            else dtype(0.0),
            np.max(residual_binary_arithmetic_error)
            if residual_binary_arithmetic_error.size
            else dtype(0.0),
        )
        upper_float64 = _hz_longdouble_to_outward_float64_upper(upper)
        longdouble_nmant = int(np.finfo(dtype).nmant)
        longdouble_eps = float(np.finfo(dtype).eps)
        # Diagnostic reductions above are O(number of factors).  They are
        # deliberately completed before this terminal authorization barrier:
        # crossing the shared absolute deadline during diagnostics must never
        # leave a proof-bearing receipt.
        _hz_split_certificate_deadline(
            deadline,
            "preformed_checker_after_diagnostics_before_receipt",
        )
        receipt.update({
            "status": "verified_upper",
            "proof_authority": True,
            # This checker authorizes only the finite numeric upper.  Pattern
            # coverage, live terminal seals, and external handle issuance
            # remain the conditional adapter's responsibility.
            "pcoh_authorization": False,
            "upper": upper_float64,
            "objective_guard": float(objective_guard),
            "residual_guard": float(residual_guard),
            "roundoff_guard": float(total_guard),
            "longdouble_nmant": longdouble_nmant,
            "longdouble_eps": longdouble_eps,
        })
        return upper, receipt
    except _HZSplitBlockCertificateDeadline as exc:
        receipt["status"] = f"deadline_exhausted:{str(exc)[:120]}"
        return None, receipt
    except Exception as exc:
        receipt["status"] = (
            f"invalid:{type(exc).__name__}:{str(exc)[:120]}"
        )
        return None, receipt


def _hz_candidate_dual_support(
    *,
    q,
    A,
    rl,
    ru,
    lb,
    ub,
    row_dual,
):
    """Evaluate one unguarded candidate support in original coordinates."""

    q = np.asarray(q, dtype=np.float64).reshape(-1)
    row_dual = np.asarray(row_dual, dtype=np.float64).reshape(-1)
    d = -row_dual
    finite_l = np.isfinite(rl)
    finite_u = np.isfinite(ru)
    if (
        row_dual.size != A.shape[0]
        or q.size != A.shape[1]
        or np.any(d[(~finite_l) & finite_u] < -1e-10)
        or np.any(d[finite_l & (~finite_u)] > 1e-10)
        or np.any(np.abs(d[(~finite_l) & (~finite_u)]) > 1e-10)
    ):
        return None
    side = np.where(d >= 0.0, ru, rl)
    finite_side = np.isfinite(side)
    row_support = float(np.dot(d[finite_side], side[finite_side]))
    residual = np.asarray(q - A.transpose() @ d, dtype=np.float64).reshape(-1)
    box_side = np.where(residual >= 0.0, ub, lb)
    value = row_support + float(np.dot(residual, box_side))
    return float(value) if np.isfinite(value) else None


def _hz_candidate_support_attribution(
    *,
    q,
    A,
    rl,
    ru,
    lb,
    ub,
    row_dual,
    column_layer_ids=None,
    constraint_row_tags=None,
    topk: int = 12,
    deadline=None,
):
    """Diagnose a candidate support by factor origin and constraint tag.

    This receipt has no proof authority and never changes a multiplier,
    bound, row schedule, or verdict.  It evaluates the same stored candidate
    as :func:`_hz_candidate_dual_support` and partitions its box term by the
    builder-recorded layer which created each normalized factor.
    """

    started = time.monotonic()
    receipt = {
        "schema": "hz_candidate_support_attribution_v1",
        "status": "unavailable",
        "proof_authority": False,
        "elapsed_seconds": 0.0,
        "tag_aggregation": "single_pass",
        "column_layer_support": [],
        "constraint_tag_contribution": [],
    }
    if deadline is not None and time.monotonic() >= float(deadline):
        receipt["status"] = "skipped_deadline"
        receipt["elapsed_seconds"] = float(time.monotonic() - started)
        return receipt
    try:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        A = _sp.csr_matrix(A, dtype=np.float64)
        rl = np.asarray(rl, dtype=np.float64).reshape(-1)
        ru = np.asarray(ru, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        row_dual = np.asarray(row_dual, dtype=np.float64).reshape(-1)
        if (
            A.shape != (row_dual.size, q.size)
            or rl.size != row_dual.size
            or ru.size != row_dual.size
            or lb.size != q.size
            or ub.size != q.size
        ):
            raise ValueError("support attribution frame mismatch")
        d = -row_dual
        residual = np.asarray(
            q - A.transpose() @ d, dtype=np.float64
        ).reshape(-1)
        if deadline is not None and time.monotonic() >= float(deadline):
            receipt["status"] = "skipped_deadline"
            receipt["elapsed_seconds"] = float(time.monotonic() - started)
            return receipt
        box_side = np.where(residual >= 0.0, ub, lb)
        column_contribution = residual * box_side
        if not np.all(np.isfinite(column_contribution)):
            raise ValueError("support attribution has non-finite box term")

        if column_layer_ids is None:
            layers = np.full(q.size, -1, dtype=np.int64)
        else:
            layers = np.asarray(
                column_layer_ids, dtype=np.int64
            ).reshape(-1)
            if layers.size != q.size:
                raise ValueError(
                    "support attribution column provenance mismatch"
                )
        layer_rows = []
        for layer_id in np.unique(layers):
            mask = layers == int(layer_id)
            contribution = float(np.sum(column_contribution[mask]))
            layer_rows.append(
                {
                    "layer_id": int(layer_id),
                    "support": contribution,
                    "absolute_support": float(
                        np.sum(np.abs(column_contribution[mask]))
                    ),
                    "residual_nnz": int(
                        np.count_nonzero(residual[mask])
                    ),
                    "column_count": int(np.count_nonzero(mask)),
                }
            )
        layer_rows.sort(
            key=lambda item: (
                -float(item["absolute_support"]),
                int(item["layer_id"]),
            )
        )

        finite_l = np.isfinite(rl)
        finite_u = np.isfinite(ru)
        side = np.where(d >= 0.0, ru, rl)
        finite_side = np.isfinite(side)
        row_contribution = np.zeros(row_dual.size, dtype=np.float64)
        row_contribution[finite_side] = (
            d[finite_side] * side[finite_side]
        )
        if constraint_row_tags is None:
            tags = np.full(row_dual.size, "unattributed", dtype=object)
        else:
            tags = np.asarray(
                tuple(constraint_row_tags), dtype=object
            ).reshape(-1)
            if tags.size != row_dual.size:
                raise ValueError(
                    "support attribution constraint tags mismatch"
                )
        # Property micro-RLT gives every generated row a distinct audit tag.
        # Rebuilding a full-size mask once per tag made this proof-neutral
        # receipt O(number_of_rows * number_of_tags): the real 106k-row /
        # 8.2k-tag frame could outlive the worker's hard deadline after the
        # candidate itself had already completed.  Aggregate once in row
        # order.  This changes no multiplier, bound, schedule, or verdict.
        tag_accumulators = {}
        for index, raw_tag in enumerate(tags):
            if (
                deadline is not None
                and (index & 4095) == 0
                and time.monotonic() >= float(deadline)
            ):
                receipt["status"] = "skipped_deadline"
                receipt["elapsed_seconds"] = float(
                    time.monotonic() - started
                )
                return receipt
            tag = str(raw_tag)
            accumulator = tag_accumulators.get(tag)
            if accumulator is None:
                accumulator = [0.0, 0.0, 0.0, 0]
                tag_accumulators[tag] = accumulator
            contribution = float(row_contribution[index])
            dual_abs = abs(float(d[index]))
            accumulator[0] += contribution
            accumulator[1] += abs(contribution)
            accumulator[2] += dual_abs
            accumulator[3] += int(d[index] != 0.0)
        tag_rows = [
            {
                "tag": tag,
                "contribution": float(values[0]),
                "absolute_contribution": float(values[1]),
                "dual_l1": float(values[2]),
                "dual_nnz": int(values[3]),
            }
            for tag, values in tag_accumulators.items()
        ]
        tag_rows.sort(
            key=lambda item: (
                -float(item["dual_l1"]),
                str(item["tag"]),
            )
        )
        receipt.update(
            {
                "status": "computed",
                "generator_box_support": float(
                    np.sum(column_contribution)
                ),
                "generator_box_absolute_support": float(
                    np.sum(np.abs(column_contribution))
                ),
                "constraint_row_support": float(
                    np.sum(row_contribution)
                ),
                "candidate_support": float(
                    np.sum(column_contribution)
                    + np.sum(row_contribution)
                ),
                "residual_nnz": int(np.count_nonzero(residual)),
                "dual_nnz": int(np.count_nonzero(d)),
                "one_sided_lower_rows": int(
                    np.count_nonzero(finite_l & ~finite_u)
                ),
                "one_sided_upper_rows": int(
                    np.count_nonzero(~finite_l & finite_u)
                ),
                "column_layer_support": layer_rows[: int(topk)],
                "constraint_tag_contribution": tag_rows[: int(topk)],
                "constraint_tag_group_count": int(len(tag_rows)),
            }
        )
    except Exception as exc:
        receipt.update(
            {
                "status": f"error:{type(exc).__name__}",
                "error": str(exc)[:240],
            }
        )
    receipt["elapsed_seconds"] = float(time.monotonic() - started)
    return receipt


def _hz_constraint_generation_dual_candidate(
    *,
    q,
    A,
    rl,
    ru,
    lb,
    ub,
    seed_row_dual,
    deadline,
    max_rounds: int = 24,
    add_batch: int = 1024,
    max_selected_rows: int = 24576,
):
    """Solve a property-conditioned sequence of small outer-relaxation LPs.

    Only the current active constraint rows are loaded into HiGHS.  Its primal
    candidate is checked against *all* original rows with one sparse matvec;
    the most violated omitted rows are added and the model is re-solved.
    Every exported dual is expanded with zeros to the original row order and
    therefore remains eligible for the independent long-double checker.

    This helper is candidate-only.  Solver status, primal feasibility, row
    selection, and the unguarded support reported here have no proof authority.
    """

    started = time.monotonic()
    stats = {
        "schema": "hz_property_constraint_generation_candidate_v1",
        "status": "not_started",
        "proof_authority": False,
        "rounds_completed": 0,
        "rows_seeded": 0,
        "rows_selected": 0,
        "rows_added_by_violation": 0,
        "full_rows": int(A.shape[0]),
        "full_nnz": int(A.nnz),
        "loaded_nnz": 0,
        "best_support": None,
        "initial_support": None,
        "best_improvement": 0.0,
        "last_max_violation": None,
        "last_violated_rows": None,
        "full_primal_feasible_candidate": False,
        "elapsed_s": 0.0,
    }
    if not (_HAS_HIGHSPY and _HAS_SCIPY):
        stats["status"] = "unavailable"
        return None, stats
    if time.monotonic() >= float(deadline):
        stats["status"] = "no_budget"
        return None, stats
    try:
        A = _sp.csr_matrix(A, dtype=np.float64, copy=False)
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        rl = np.asarray(rl, dtype=np.float64).reshape(-1)
        ru = np.asarray(ru, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        seed_row_dual = np.asarray(
            seed_row_dual, dtype=np.float64
        ).reshape(-1)
        if (
            A.shape != (rl.size, q.size)
            or ru.size != rl.size
            or lb.size != q.size
            or ub.size != q.size
            or seed_row_dual.size != rl.size
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(lb))
            or not np.all(np.isfinite(ub))
        ):
            raise ValueError("constraint-generation frame is malformed")

        zero_dual = np.zeros(A.shape[0], dtype=np.float64)
        initial_support = _hz_candidate_dual_support(
            q=q,
            A=A,
            rl=rl,
            ru=ru,
            lb=lb,
            ub=ub,
            row_dual=zero_dual,
        )
        stats["initial_support"] = initial_support
        best_support = initial_support
        best_row_dual = zero_dual

        raw_seed = np.flatnonzero(
            np.isfinite(seed_row_dual) & (seed_row_dual != 0.0)
        ).astype(np.int64, copy=False)
        if raw_seed.size > int(max_selected_rows):
            order = np.argpartition(
                np.abs(seed_row_dual[raw_seed]),
                -int(max_selected_rows),
            )[-int(max_selected_rows):]
            raw_seed = raw_seed[order]
        selected = [int(row) for row in np.sort(raw_seed)]
        selected_set = set(selected)
        stats["rows_seeded"] = int(len(selected))

        candidate_A, _candidate_matrix_stats = _highs_candidate_csr(
            A,
            small_matrix_value=1e-12,
        )
        h = _highspy.Highs()
        HS = _highspy.HighsStatus
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "on")
        h.setOptionValue("small_matrix_value", 1e-12)
        h.setOptionValue("threads", _highs_process_threads())
        add_status = h.addCols(
            int(q.size),
            -q,
            lb,
            ub,
            0,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )
        if add_status != HS.kOk:
            raise RuntimeError(f"addCols returned {add_status}")

        loaded_rows: List[int] = []

        def add_rows(rows_to_add):
            rows_array = np.asarray(rows_to_add, dtype=np.int64).reshape(-1)
            if rows_array.size == 0:
                return
            block = candidate_A[rows_array, :].tocsr()
            status = h.addRows(
                int(rows_array.size),
                rl[rows_array],
                ru[rows_array],
                int(block.nnz),
                block.indptr.astype(np.int32),
                block.indices.astype(np.int32),
                block.data.astype(np.float64),
            )
            if status != HS.kOk:
                raise RuntimeError(f"addRows returned {status}")
            loaded_rows.extend(int(row) for row in rows_array)
            stats["loaded_nnz"] += int(block.nnz)

        add_rows(selected)
        for round_index in range(int(max_rounds)):
            now = time.monotonic()
            if now >= float(deadline):
                stats["status"] = "budget_exhausted"
                break
            h.setOptionValue(
                "time_limit",
                max(1e-3, float(deadline) - now),
            )
            run_status = h.run()
            stats["rounds_completed"] += 1
            solution = h.getSolution()
            primal = np.asarray(
                solution.col_value, dtype=np.float64
            ).reshape(-1)
            selected_dual = np.asarray(
                solution.row_dual, dtype=np.float64
            ).reshape(-1)
            if (
                primal.size != q.size
                or selected_dual.size != len(loaded_rows)
                or not np.all(np.isfinite(primal))
            ):
                stats["status"] = "invalid_solver_candidate"
                break
            full_row_dual = np.zeros(A.shape[0], dtype=np.float64)
            if loaded_rows and np.all(np.isfinite(selected_dual)):
                full_row_dual[
                    np.asarray(loaded_rows, dtype=np.int64)
                ] = selected_dual
                support = _hz_candidate_dual_support(
                    q=q,
                    A=A,
                    rl=rl,
                    ru=ru,
                    lb=lb,
                    ub=ub,
                    row_dual=full_row_dual,
                )
                if (
                    support is not None
                    and (
                        best_support is None
                        or support < best_support
                    )
                ):
                    best_support = float(support)
                    best_row_dual = full_row_dual

            activity = np.asarray(A @ primal, dtype=np.float64).reshape(-1)
            lower_violation = np.where(
                np.isfinite(rl), rl - activity, 0.0
            )
            upper_violation = np.where(
                np.isfinite(ru), activity - ru, 0.0
            )
            violation = np.maximum(
                np.maximum(lower_violation, upper_violation), 0.0
            )
            scale = 1.0 + np.maximum(
                np.abs(activity),
                np.maximum(
                    np.where(np.isfinite(rl), np.abs(rl), 0.0),
                    np.where(np.isfinite(ru), np.abs(ru), 0.0),
                ),
            )
            relative = violation / scale
            if selected_set:
                relative[
                    np.fromiter(selected_set, dtype=np.int64)
                ] = 0.0
            violated = np.flatnonzero(
                (violation > 5e-8) & (relative > 5e-10)
            )
            stats["last_max_violation"] = float(
                np.max(violation) if violation.size else 0.0
            )
            stats["last_violated_rows"] = int(violated.size)
            if violated.size == 0:
                stats["full_primal_feasible_candidate"] = True
                stats["status"] = "full_primal_candidate_feasible"
                break
            remaining_cap = int(max_selected_rows) - len(selected_set)
            if remaining_cap <= 0:
                stats["status"] = "row_cap_reached"
                break
            count = min(int(add_batch), int(violated.size), remaining_cap)
            if count < violated.size:
                chosen = violated[
                    np.argpartition(relative[violated], -count)[-count:]
                ]
            else:
                chosen = violated
            chosen = chosen[
                np.argsort(-relative[chosen], kind="mergesort")
            ]
            new_rows = [
                int(row)
                for row in chosen
                if int(row) not in selected_set
            ]
            if not new_rows:
                stats["status"] = "no_new_rows"
                break
            add_rows(new_rows)
            selected_set.update(new_rows)
            stats["rows_added_by_violation"] += int(len(new_rows))
            stats["status"] = "running"
            if run_status not in {HS.kOk, HS.kWarning}:
                stats["status"] = f"solver_status:{run_status}"
                break
        else:
            stats["status"] = "round_cap_reached"

        stats["rows_selected"] = int(len(selected_set))
        stats["best_support"] = best_support
        if best_support is not None and initial_support is not None:
            stats["best_improvement"] = float(
                initial_support - best_support
            )
        stats["elapsed_s"] = float(time.monotonic() - started)
        return best_row_dual, stats
    except Exception as exc:
        stats["status"] = (
            f"error:{type(exc).__name__}:{str(exc)[:160]}"
        )
        stats["elapsed_s"] = float(time.monotonic() - started)
        return None, stats


def _hz_property_micro_rlt_source_candidate_rows(
    hz,
    *,
    constraint_row_tags,
    matrix_row_count,
):
    """Map the bounded RLT source-row receipt into the base LP row order.

    These rows seed candidate generation only.  The receipt has no authority
    over the accepted upper bound: malformed metadata returns an empty
    selection, and every generated multiplier is still independently checked
    against the complete live matrix.
    """

    empty = np.zeros(0, dtype=np.int64)
    receipt = getattr(hz, "_property_micro_rlt_receipt", None)
    if not isinstance(receipt, Mapping):
        return empty
    try:
        if receipt.get("schema") != "act.property_micro_rlt.v1":
            return empty
        base_n_eq = int(receipt["base_n_eq"])
        base_n_ub = int(receipt["base_n_ub"])
        result_n_ub = int(receipt["result_n_ub"])
        if (
            base_n_eq != int(hz.n_eq)
            or result_n_ub != int(hz.n_ub)
            or not (0 <= base_n_ub <= result_n_ub)
            or int(matrix_row_count) != int(hz.n_eq + hz.n_ub)
        ):
            return empty
        raw_selection = receipt.get("selection")
        if not isinstance(raw_selection, list) or len(raw_selection) > 2:
            return empty
        source_rows = []
        for entry in raw_selection:
            if not isinstance(entry, Mapping):
                return empty
            raw_rows = entry.get("source_upper_rows")
            if not isinstance(raw_rows, list) or len(raw_rows) > 4:
                return empty
            for raw_row in raw_rows:
                if (
                    isinstance(raw_row, bool)
                    or not isinstance(raw_row, (int, np.integer))
                ):
                    return empty
                upper_row = int(raw_row)
                if upper_row < 0 or upper_row >= base_n_ub:
                    return empty
                source_rows.append(base_n_eq + upper_row)
        source = np.unique(
            np.asarray(source_rows, dtype=np.int64)
        )
        if source.size > 8:
            return empty
        tags = tuple(constraint_row_tags)
        if (
            len(tags) != int(matrix_row_count)
            or (
                source.size
                and any(
                    str(tags[int(row)]).startswith(
                        "property_micro_rlt:"
                    )
                    for row in source
                )
            )
        ):
            return empty
        return source
    except (KeyError, TypeError, ValueError, OverflowError):
        return empty


def _hz_property_micro_rlt_bridge_candidate_rows(
    hz,
    *,
    constraint_row_tags,
    matrix_row_count,
    deadline=None,
    max_materialization_tag_blocks: int = 4,
    max_rows: int = 32_768,
):
    """Select a bounded ordinary DAG bridge for a micro-RLT candidate.

    A property objective at the final ReLU is expressed in the freshly
    materialized output factors, while the packet products constrain the
    exact-ReLU preactivation and binary factors.  Restricting candidate
    generation to packet/source rows can therefore leave a missing path
    through the ordinary ReLU and residual-ADD materialization constraints.

    This helper selects only already-live rows from the pre-RLT base prefix:

    * every ordinary ReLU row at the packet's common exact-ReLU layer; and
    * at most two closest complete ``add_materialize`` forward/reverse pairs.

    The selection is a proof-neutral scheduling heuristic.  It never creates
    a row, changes a bound, or authorizes a verdict; any multiplier using the
    returned rows is still rechecked against the complete live matrix.
    Malformed metadata, incomplete materialization pairs, caps, or deadlines
    return an empty selection.
    """

    empty = np.zeros(0, dtype=np.int64)
    try:
        max_materialization_tag_blocks = int(
            max_materialization_tag_blocks
        )
        max_rows = int(max_rows)
        if (
            max_materialization_tag_blocks < 0
            or max_materialization_tag_blocks > 4
            or max_materialization_tag_blocks % 2
            or max_rows < 0
            or max_rows > 32_768
        ):
            return empty
        if deadline is not None and time.monotonic() >= float(deadline):
            return empty
        metadata = getattr(hz, "operator_hz_metadata", None)
        if not isinstance(metadata, Mapping):
            return empty
        micro = metadata.get("property_micro_rlt")
        if (
            not isinstance(micro, Mapping)
            or micro.get("schema")
            != "operator_hz_property_micro_rlt_v1"
            or micro.get("status") != "applied"
        ):
            return empty
        records = micro.get("exact_relu_records")
        if (
            not isinstance(records, list)
            or not 1 <= len(records) <= 2
            or any(not isinstance(record, Mapping) for record in records)
        ):
            return empty
        focus_layers = {
            int(record["layer_id"]) for record in records
        }
        if len(focus_layers) != 1:
            return empty
        focus_layer = int(next(iter(focus_layers)))
        base_counts = micro.get("base_counts")
        result_counts = micro.get("result_counts")
        if not isinstance(base_counts, Mapping) or not isinstance(
            result_counts, Mapping
        ):
            return empty
        base_n_eq = int(base_counts["n_eq"])
        base_n_ub = int(base_counts["n_ub"])
        result_n_eq = int(result_counts["n_eq"])
        result_n_ub = int(result_counts["n_ub"])
        ordinary_row_limit = base_n_eq + base_n_ub
        if (
            min(base_n_eq, base_n_ub, result_n_eq, result_n_ub) < 0
            or result_n_eq != base_n_eq
            or result_n_ub < base_n_ub
            or ordinary_row_limit > int(matrix_row_count)
            or result_n_eq + result_n_ub != int(matrix_row_count)
        ):
            return empty
        tags = tuple(constraint_row_tags)
        if len(tags) != int(matrix_row_count):
            return empty

        relu_rows = []
        materialization = {}
        for row, raw_tag in enumerate(tags[:ordinary_row_limit]):
            if (
                deadline is not None
                and (row & 4095) == 0
                and time.monotonic() >= float(deadline)
            ):
                return empty
            tag = str(raw_tag)
            if tag.startswith("property_micro_rlt:"):
                return empty
            pieces = tag.split(":")
            if (
                len(pieces) >= 2
                and pieces[0].startswith("relu_")
                and int(pieces[1]) == focus_layer
            ):
                relu_rows.append(int(row))
                continue
            if (
                len(pieces) == 3
                and pieces[0] == "add_materialize"
                and pieces[2] in {"forward", "reverse"}
            ):
                layer = int(pieces[1])
                if layer < focus_layer:
                    materialization.setdefault(
                        layer,
                        {"forward": [], "reverse": []},
                    )[pieces[2]].append(int(row))

        selected = list(relu_rows)
        if len(selected) > max_rows:
            return empty
        pair_cap = max_materialization_tag_blocks // 2
        pairs_used = 0
        for layer in sorted(materialization, reverse=True):
            if pairs_used >= pair_cap:
                break
            pair = materialization[layer]
            forward = pair["forward"]
            reverse = pair["reverse"]
            if not forward or len(forward) != len(reverse):
                return empty
            pair_rows = [*forward, *reverse]
            if len(selected) + len(pair_rows) > max_rows:
                continue
            selected.extend(pair_rows)
            pairs_used += 1
        result = np.unique(np.asarray(selected, dtype=np.int64))
        if (
            result.size > max_rows
            or (
                result.size
                and (
                    int(result.min()) < 0
                    or int(result.max()) >= ordinary_row_limit
                )
            )
        ):
            return empty
        return result
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return empty


def _hz_property_micro_rlt_focused_objective_schedule(
    hz,
    *,
    safe_groups,
    candidate_rows,
    cube_upper,
):
    """Choose the packet's adjoint-focused rival before generic hardness.

    The two exact ReLUs and their RLT source rows are selected for one explicit
    property rival.  Testing an unrelated globally-hard group can therefore
    have zero first-order packet signal.  This helper only changes which one
    of the already-live rows receives candidate-generation time; every other
    row remains a survivor and every accepted bound still uses the full
    independent checker.
    """

    rows = np.asarray(candidate_rows, dtype=np.int64).reshape(-1)
    no_focus = (None, rows.copy(), None, None)
    metadata = getattr(hz, "operator_hz_metadata", None)
    if not isinstance(metadata, Mapping) or safe_groups is None:
        return no_focus
    try:
        micro = metadata.get("property_micro_rlt")
        if (
            not isinstance(micro, Mapping)
            or micro.get("status") != "applied"
            or micro.get("common_focused_rival_id") is None
        ):
            return no_focus
        focus = int(micro["common_focused_rival_id"])
        tail = metadata.get("property_tail_upper")
        if not isinstance(tail, Mapping):
            return no_focus
        baseline_count = int(tail["baseline_plane_count"])
        declared_groups = tail.get("property_row_groups")
        if (
            focus < 0
            or focus >= len(safe_groups)
            or baseline_count != len(safe_groups)
            or not isinstance(declared_groups, list)
            or len(declared_groups) != baseline_count
        ):
            return no_focus
        declared_focus_group = tuple(
            int(row) for row in declared_groups[focus]
        )
        if (
            not declared_focus_group
            or int(focus) not in declared_focus_group
            or int(focus) not in tuple(
                int(row) for row in safe_groups[focus]
            )
        ):
            return no_focus
        cube = np.asarray(cube_upper, dtype=np.float64).reshape(-1)
        available = set(int(row) for row in rows)
        focused_rows = [
            int(row)
            for row in safe_groups[focus]
            if int(row) in available
        ]
        if not focused_rows:
            return no_focus
        # The selector that created the exact ReLUs consumes the original
        # property-C rows, which occupy the first baseline block in the
        # exported grouped planes.  Prefer that exact baseline row.  A
        # shared-suffix alternative can be cube-tighter while having no
        # coefficient path through the selected exact neurons.
        if int(focus) in focused_rows:
            chosen = int(focus)
            plane_kind = "baseline_property_plane"
        else:
            chosen = min(
                focused_rows,
                key=lambda row: (float(cube[row]), int(row)),
            )
            plane_kind = "best_available_plane"
        deferred = np.asarray(
            [int(row) for row in rows if int(row) != int(chosen)],
            dtype=np.int64,
        )
        return (
            np.asarray([chosen], dtype=np.int64),
            deferred,
            int(focus),
            str(plane_kind),
        )
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return no_focus


def _hz_pc_cbde_stats_defaults():
    """Bounded proof-neutral diagnostics for the optional PC-CBDE candidate."""

    return {
        "gpu_dual_pc_cbde_status": "not_started",
        "gpu_dual_pc_cbde_elapsed_s": 0.0,
        "gpu_dual_pc_cbde_budget_s": 0.0,
        "gpu_dual_pc_cbde_deadline_reached": False,
        "gpu_dual_pc_cbde_cone_rows": [],
        "gpu_dual_pc_cbde_cone_row_count": 0,
        "gpu_dual_pc_cbde_local_row_count": 0,
        "gpu_dual_pc_cbde_bridge_row_count": 0,
        "gpu_dual_pc_cbde_generated_row_count": 0,
        "gpu_dual_pc_cbde_generated_warm_nonzero_count": 0,
        "gpu_dual_pc_cbde_generated_warm_truncated_count": 0,
        "gpu_dual_pc_cbde_source_row_count": 0,
        "gpu_dual_pc_cbde_ignored_source_row_count": 0,
        "gpu_dual_pc_cbde_full_nnz": 0,
        "gpu_dual_pc_cbde_updates": 0,
        "gpu_dual_pc_cbde_checked_upper_full": None,
        "gpu_dual_pc_cbde_checked_upper_without_generated": None,
        "gpu_dual_pc_cbde_checked_upper_without_bridge": None,
        "gpu_dual_pc_cbde_checked_upper_without_both": None,
        "gpu_dual_pc_cbde_all_ablations_verified": False,
        "gpu_dual_pc_cbde_strict_family_ablation": False,
        "gpu_dual_pc_cbde_strict_family_ablation_tol": None,
        "gpu_dual_pc_cbde_full_vs_without_generated_gap": None,
        "gpu_dual_pc_cbde_full_vs_without_bridge_gap": None,
        "gpu_dual_pc_cbde_old_support": None,
        "gpu_dual_pc_cbde_full_support": None,
        "gpu_dual_pc_cbde_support_improvement": 0.0,
        "gpu_dual_pc_cbde_support_improvement_tol": None,
        "gpu_dual_pc_cbde_replaced_old_candidate": False,
        "gpu_dual_pc_cbde_error_type": None,
        "gpu_dual_pc_cbde_error_message": None,
        "gpu_dual_pc_cbde_proof_authority": False,
    }


def _hz_try_pc_cbde_candidate(
    *,
    rows,
    row_topk,
    candidates,
    candidate_constraint_rows,
    bridge_only_rows,
    micro_rlt_rows,
    source_rows,
    constraint_row_tags,
    frame,
    q,
    row_dual_matrix,
    candidate_support,
    A,
    rl,
    ru,
    lb,
    ub,
    certificate_c,
    certificate_G,
    certificate_center_error,
    C,
    t,
    phase_deadline,
):
    """Try one bounded PC-CBDE replacement without disturbing the old path.

    The old packet candidate is returned byte-for-byte on every skip, error, or
    deadline.  Four independent full-frame checker calls are a causal
    diagnostic gate only; the chosen full candidate is checked again by the
    ordinary outer certificate loop before it can prune a property row.
    """

    pc_stats = _hz_pc_cbde_stats_defaults()
    started = time.monotonic()
    original_row_dual = np.asarray(
        row_dual_matrix, dtype=np.float64
    )
    original_support = np.asarray(
        candidate_support, dtype=np.float64
    ).reshape(-1)
    original_bridge_rows = np.asarray(
        bridge_only_rows, dtype=np.int64
    ).reshape(-1)

    def finish(status):
        pc_stats["gpu_dual_pc_cbde_status"] = str(status)
        pc_stats["gpu_dual_pc_cbde_elapsed_s"] = float(
            time.monotonic() - started
        )
        return (
            original_row_dual,
            original_support,
            original_bridge_rows,
            pc_stats,
        )

    if np.asarray(micro_rlt_rows).size == 0:
        return finish("skipped_no_micro_rows")
    if np.asarray(rows).size != 1:
        return finish("skipped_objective_count")
    if (
        candidate_constraint_rows is None
        or str(getattr(candidates, "device", "")) != "cpu_packet_core"
    ):
        return finish("skipped_old_packet_unavailable")
    if np.asarray(bridge_only_rows).size:
        return finish("skipped_static_bridge")
    if int(row_topk) != 0:
        return finish("skipped_row_topk")
    if constraint_row_tags is None:
        return finish("skipped_missing_row_tags")

    now = time.monotonic()
    checker_reserve = 0.30
    available = float(phase_deadline) - now - checker_reserve
    if available <= 0.0:
        pc_stats["gpu_dual_pc_cbde_deadline_reached"] = bool(
            now >= float(phase_deadline)
        )
        return finish("skipped_checker_reserve")
    pc_deadline = min(
        now + 1.75,
        float(phase_deadline) - checker_reserve,
    )
    pc_stats["gpu_dual_pc_cbde_budget_s"] = float(
        max(0.0, pc_deadline - now)
    )

    try:
        warm_d = -np.asarray(
            original_row_dual[0], dtype=np.float64
        ).reshape(-1)
        micro = np.asarray(micro_rlt_rows, dtype=np.int64).reshape(-1)
        finite_nonzero = [
            int(row)
            for row in micro
            if np.isfinite(warm_d[int(row)])
            and warm_d[int(row)] != 0.0
        ]
        optimization_packet = np.asarray(
            sorted(
                finite_nonzero,
                key=lambda row: (-abs(float(warm_d[row])), int(row)),
            )[:64],
            dtype=np.int64,
        )
        pc_stats["gpu_dual_pc_cbde_generated_warm_nonzero_count"] = int(
            len(finite_nonzero)
        )
        pc_stats["gpu_dual_pc_cbde_generated_warm_truncated_count"] = int(
            max(0, len(finite_nonzero) - optimization_packet.size)
        )
        if optimization_packet.size == 0:
            return finish("skipped_no_generated_warm")
        # Every generated row outside the bounded optimization packet is fixed
        # to zero.  Otherwise an unselected warm multiplier would survive the
        # ``without_generated`` ablation and invalidate its family semantics.
        pc_warm_d = warm_d.copy()
        pc_warm_d[micro] = 0.0
        pc_warm_d[optimization_packet] = warm_d[optimization_packet]

        from act.back_end.hybridz_tf.property_causal_block_integration import (
            property_causal_block_integration,
        )

        tags = tuple(str(tag) for tag in constraint_row_tags)
        if len(tags) != int(A.shape[0]):
            return finish("skipped_invalid_row_tags")
        upper_only = np.isneginf(rl) & np.isfinite(ru)
        ordinary = np.asarray(
            [
                not tag.startswith("property_micro_rlt:")
                for tag in tags
            ],
            dtype=np.bool_,
        )
        allowed = np.asarray(upper_only & ordinary, dtype=np.bool_)
        pc_frame = type(frame)(
            A=A,
            rl=rl,
            ru=ru,
            lb=lb,
            ub=ub,
            row_tags=tags,
        )
        integrated = property_causal_block_integration(
            pc_frame,
            np.asarray(q, dtype=np.float64),
            np.asarray([pc_warm_d], dtype=np.float64),
            incidence_packet_rows=micro,
            optimization_packet_rows=optimization_packet,
            source_rows=np.asarray(source_rows, dtype=np.int64).reshape(-1),
            allowed_row_mask=allowed,
            row_tags=tags,
            deadline=pc_deadline,
        )
        if not bool(integrated.success):
            pc_stats["gpu_dual_pc_cbde_deadline_reached"] = bool(
                integrated.deadline_reached
            )
            return finish(f"integration_{integrated.status}")
        if time.monotonic() >= pc_deadline:
            pc_stats["gpu_dual_pc_cbde_deadline_reached"] = True
            return finish("deadline_after_integration")

        pc_stats.update(
            {
                "gpu_dual_pc_cbde_cone_rows": [
                    int(row) for row in integrated.cone_rows
                ],
                "gpu_dual_pc_cbde_cone_row_count": int(
                    integrated.cone_rows.size
                ),
                "gpu_dual_pc_cbde_local_row_count": int(
                    integrated.local_rows.size
                ),
                "gpu_dual_pc_cbde_bridge_row_count": int(
                    integrated.bridge_rows.size
                ),
                "gpu_dual_pc_cbde_generated_row_count": int(
                    integrated.generated_rows.size
                ),
                "gpu_dual_pc_cbde_source_row_count": int(
                    integrated.source_rows.size
                ),
                "gpu_dual_pc_cbde_ignored_source_row_count": int(
                    integrated.ignored_source_rows.size
                ),
                "gpu_dual_pc_cbde_full_nnz": int(
                    np.count_nonzero(integrated.ablation("full").d)
                ),
                "gpu_dual_pc_cbde_updates": int(
                    sum(
                        int(item.optimizer.updates)
                        for item in integrated.ablations
                    )
                ),
            }
        )

        checked = {}
        for name in (
            "full",
            "without_generated",
            "without_bridge",
            "without_both",
        ):
            if time.monotonic() >= pc_deadline:
                pc_stats["gpu_dual_pc_cbde_deadline_reached"] = True
                return finish("deadline_during_ablation_check")
            ablation = integrated.ablation(name)
            upper, receipt = _hz_independent_lp_lagrangian_upper(
                c=certificate_c,
                Gc=certificate_G,
                C_row=C[int(np.asarray(rows).reshape(-1)[0])],
                threshold=t[int(np.asarray(rows).reshape(-1)[0])],
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                row_dual=ablation.row_dual[0],
                center_error=certificate_center_error,
            )
            if (
                upper is None
                or receipt.get("status") != "verified_upper"
                or not np.isfinite(upper)
            ):
                return finish("ablation_checker_incomplete")
            checked[name] = np.longdouble(upper)
            if time.monotonic() >= pc_deadline:
                pc_stats["gpu_dual_pc_cbde_deadline_reached"] = True
                return finish("deadline_during_ablation_check")

        full_upper = checked["full"]
        without_generated = checked["without_generated"]
        without_bridge = checked["without_bridge"]
        comparison_scale = max(
            np.longdouble(1.0),
            abs(full_upper),
            abs(without_generated),
            abs(without_bridge),
        )
        family_tolerance = (
            np.longdouble(512.0)
            # The checked arithmetic is long-double, but every input
            # coefficient and optimizer multiplier originated in float64.
            # Attribute only gaps that dominate the source-precision noise.
            * np.longdouble(np.finfo(np.float64).eps)
            * comparison_scale
        )
        strict_family_ablation = bool(
            full_upper + family_tolerance < without_generated
            and full_upper + family_tolerance < without_bridge
        )
        pc_stats.update(
            {
                "gpu_dual_pc_cbde_checked_upper_full": float(full_upper),
                "gpu_dual_pc_cbde_checked_upper_without_generated": float(
                    without_generated
                ),
                "gpu_dual_pc_cbde_checked_upper_without_bridge": float(
                    without_bridge
                ),
                "gpu_dual_pc_cbde_checked_upper_without_both": float(
                    checked["without_both"]
                ),
                "gpu_dual_pc_cbde_all_ablations_verified": True,
                "gpu_dual_pc_cbde_strict_family_ablation": (
                    strict_family_ablation
                ),
                "gpu_dual_pc_cbde_strict_family_ablation_tol": float(
                    family_tolerance
                ),
                "gpu_dual_pc_cbde_full_vs_without_generated_gap": float(
                    without_generated - full_upper
                ),
                "gpu_dual_pc_cbde_full_vs_without_bridge_gap": float(
                    without_bridge - full_upper
                ),
            }
        )

        old_float_support = _hz_candidate_dual_support(
            q=np.asarray(q, dtype=np.float64).reshape(1, -1)[0],
            A=A,
            rl=rl,
            ru=ru,
            lb=lb,
            ub=ub,
            row_dual=original_row_dual[0],
        )
        full = integrated.ablation("full")
        full_float_support = float(full.candidate_support[0])
        if (
            old_float_support is None
            or not np.isfinite(float(old_float_support))
            or not np.isfinite(full_float_support)
        ):
            return finish("verified_invalid_float_support")
        improvement = float(old_float_support) - full_float_support
        improvement_tolerance = (
            512.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                abs(float(old_float_support)),
                abs(full_float_support),
            )
        )
        pc_stats.update(
            {
                "gpu_dual_pc_cbde_old_support": float(old_float_support),
                "gpu_dual_pc_cbde_full_support": full_float_support,
                "gpu_dual_pc_cbde_support_improvement": float(
                    max(0.0, improvement)
                ),
                "gpu_dual_pc_cbde_support_improvement_tol": float(
                    improvement_tolerance
                ),
            }
        )
        if not (
            full_float_support + improvement_tolerance
            < float(old_float_support)
        ):
            return finish("verified_no_strict_support_improvement")

        replacement_row_dual = original_row_dual.copy()
        replacement_support = original_support.copy()
        replacement_row_dual[0] = np.asarray(
            full.row_dual[0], dtype=np.float64
        )
        replacement_support[0] = full_float_support
        pc_stats["gpu_dual_pc_cbde_replaced_old_candidate"] = True
        pc_stats["gpu_dual_pc_cbde_status"] = "verified_replaced"
        pc_stats["gpu_dual_pc_cbde_elapsed_s"] = float(
            time.monotonic() - started
        )
        return (
            replacement_row_dual,
            replacement_support,
            np.asarray(integrated.bridge_rows, dtype=np.int64).copy(),
            pc_stats,
        )
    except Exception as exc:
        pc_stats["gpu_dual_pc_cbde_error_type"] = type(exc).__name__
        pc_stats["gpu_dual_pc_cbde_error_message"] = (
            str(exc).replace("\n", " ")[:512]
        )
        if time.monotonic() >= pc_deadline:
            pc_stats["gpu_dual_pc_cbde_deadline_reached"] = True
        return finish(f"error:{type(exc).__name__}")


def _hz_gpu_dual_candidate_filter(
    *,
    c,
    Gc,
    Gb,
    C,
    t,
    candidate_rows,
    A,
    rl,
    ru,
    lb,
    ub,
    deadline,
    time_budget,
    steps,
    row_topk,
    learning_rate,
    tol,
    column_layer_ids=None,
    constraint_row_tags=None,
    packet_core_seed_rows=None,
    packet_core_bridge_rows=None,
):
    """Try untrusted CUDA duals, then re-prove every accepted row on ``A``.

    The CUDA optimizer is deliberately isolated from verdict authority.  Its
    output is interpreted as a proposed maximization multiplier ``d`` and is
    converted back to the independent checker's convention
    ``row_dual = -d`` only after optional top-k sparsification.  Neither its
    support estimate, its optimizer status, nor CUDA success can prune a
    rival.  Signed binary factors are mapped to the same ``z in [0, 1]``
    relaxation used by the base matrix.  The rounded center transformation is
    guarded again by the independent long-double certificate; GPU candidates
    remain ineligible to authorize witnesses.
    """

    rows = np.asarray(candidate_rows, dtype=np.int64).reshape(-1)
    binary_factor_count = int(Gb.shape[1])
    stats = {
        "gpu_dual_enabled": bool(
            int(steps) > 0 and float(time_budget) > 0.0
        ),
        "gpu_dual_status": "not_started",
        "gpu_dual_input_rows": int(rows.size),
        "gpu_dual_certified_rows": 0,
        "gpu_dual_certified_row_ids": [],
        "gpu_dual_uncertified_rows": int(rows.size),
        "gpu_dual_coverage_ok": True,
        "gpu_dual_elapsed_s": 0.0,
        "gpu_dual_time_budget_s": max(0.0, float(time_budget)),
        "gpu_dual_steps_requested": int(steps),
        "gpu_dual_steps_completed": 0,
        "gpu_dual_learning_rate": float(learning_rate),
        "gpu_dual_row_topk": int(row_topk),
        "gpu_dual_deadline_reached": False,
        "gpu_dual_deadline_stage": None,
        "gpu_dual_errors": 0,
        "gpu_dual_error_type": None,
        "gpu_dual_error_message": None,
        "gpu_dual_error_stage": None,
        "gpu_dual_certificate_attempted_rows": 0,
        "gpu_dual_certificate_errors": 0,
        "gpu_dual_initial_support_min": None,
        "gpu_dual_initial_support_max": None,
        "gpu_dual_candidate_support_min": None,
        "gpu_dual_candidate_support_max": None,
        "gpu_dual_support_improved_rows": 0,
        "gpu_dual_support_best_improvement": None,
        "gpu_dual_candidate_dual_nnz_total": 0,
        "gpu_dual_candidate_dual_nnz_max": 0,
        "gpu_dual_checked_dual_nnz_total": 0,
        "gpu_dual_checked_dual_nnz_max": 0,
        "gpu_dual_checked_generated_nnz_total": 0,
        "gpu_dual_checked_generated_nnz_max": 0,
        "gpu_dual_checked_source_nnz_total": 0,
        "gpu_dual_checked_source_nnz_max": 0,
        "gpu_dual_checked_bridge_nnz_total": 0,
        "gpu_dual_checked_bridge_nnz_max": 0,
        "gpu_dual_checked_other_nnz_total": 0,
        "gpu_dual_checked_other_nnz_max": 0,
        "gpu_dual_wavefront_updates": 0,
        "gpu_dual_wavefront_support_improved_rows": 0,
        "gpu_dual_wavefront_best_improvement": 0.0,
        "gpu_dual_wavefront_elapsed_s": 0.0,
        "gpu_dual_wavefront_selected_constraint_count": 0,
        "gpu_dual_constraint_generation_attempted_rows": 0,
        "gpu_dual_constraint_generation_improved_rows": 0,
        "gpu_dual_constraint_generation_best_improvement": 0.0,
        "gpu_dual_constraint_generation_elapsed_s": 0.0,
        "gpu_dual_constraint_generation_status": "not_started",
        "gpu_dual_constraint_generation_receipts": [],
        "gpu_dual_support_attributions": [],
        "gpu_dual_support_attribution_elapsed_s": 0.0,
        "gpu_dual_independent_certificate_elapsed_s": 0.0,
        "gpu_dual_candidate_constraint_scope": "full_original_frame",
        "gpu_dual_candidate_constraint_rows_total": int(A.shape[0]),
        "gpu_dual_candidate_constraint_rows_selected": int(A.shape[0]),
        "gpu_dual_candidate_constraint_rows_deferred": 0,
        "gpu_dual_packet_generated_rows_selected": 0,
        "gpu_dual_packet_source_rows_selected": 0,
        "gpu_dual_packet_bridge_rows_selected": 0,
        "gpu_dual_bridge_base_updates": 0,
        "gpu_dual_bridge_packet_updates": 0,
        "gpu_dual_bridge_base_nnz": 0,
        "gpu_dual_bridge_packet_nnz": 0,
        "gpu_dual_bridge_base_support_improvement": 0.0,
        "gpu_dual_bridge_combined_support_improvement": 0.0,
        "gpu_dual_candidate_constraint_selection_proof_authority": False,
        "gpu_dual_cert_upper_max": None,
        "gpu_dual_cert_min_gap_to_cutoff": None,
        "gpu_dual_cert_center_transform_guard_max": 0.0,
        "gpu_dual_checked_upper_min": None,
        "gpu_dual_checked_upper_max": None,
        "gpu_dual_device": "cuda",
        "gpu_dual_device_requested": "cuda",
        "gpu_dual_packet_core_cpu_fallback": False,
        "gpu_dual_proof_authority": False,
        "gpu_dual_binary_factor_count": int(binary_factor_count),
        "gpu_dual_binary_relaxation_enabled": bool(
            binary_factor_count > 0
        ),
        "gpu_dual_candidate_witness_eligible": bool(
            binary_factor_count == 0
        ),
        **_hz_pc_cbde_stats_defaults(),
    }
    if rows.size == 0:
        stats["gpu_dual_status"] = "empty_input"
        return rows, stats
    if int(steps) <= 0 or float(time_budget) <= 0.0:
        stats["gpu_dual_status"] = "disabled"
        return rows, stats

    started = time.monotonic()
    phase_deadline = min(
        float(deadline),
        started + max(0.0, float(time_budget)),
    )
    if phase_deadline <= started:
        stats["gpu_dual_status"] = "no_budget"
        stats["gpu_dual_deadline_reached"] = True
        stats["gpu_dual_deadline_stage"] = "before_candidate"
        return rows, stats

    def _deadline_closed(stage):
        if time.monotonic() < phase_deadline:
            return False
        stats["gpu_dual_status"] = "deadline_reached"
        stats["gpu_dual_deadline_reached"] = True
        stats["gpu_dual_deadline_stage"] = str(stage)
        stats["gpu_dual_elapsed_s"] = float(
            time.monotonic() - started
        )
        return True

    error_stage = "lazy_import"
    try:
        # Import only on the explicitly enabled path.  In particular, a
        # default-off production run must not initialize CUDA or acquire any
        # accelerator resources.
        from act.back_end.hybridz_tf.gpu_dual_candidates import (
            BatchedDualCandidates,
            OriginalFrameLP,
            batched_original_frame_row_duals,
            property_conditioned_coordinate_wavefront_duals,
        )

        error_stage = "original_frame_validation"
        A = _sp.csr_matrix(A, dtype=np.float64, copy=False)
        rl = np.asarray(rl, dtype=np.float64).reshape(-1)
        ru = np.asarray(ru, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        if (
            A.shape != (rl.size, lb.size)
            or ru.size != rl.size
            or ub.size != lb.size
            or not A.has_canonical_format
            or (A.nnz and not np.all(np.isfinite(A.data)))
        ):
            raise ValueError("GPU dual original-frame matrix is invalid")
        if _deadline_closed("after_original_frame_validation"):
            return rows, stats
        micro_rlt_rows = np.zeros(0, dtype=np.int64)
        if (
            binary_factor_count > 0
            and constraint_row_tags is not None
        ):
            tags = tuple(constraint_row_tags)
            if len(tags) == int(A.shape[0]):
                micro_rlt_rows = np.asarray(
                    [
                        index
                        for index, tag in enumerate(tags)
                        if str(tag).startswith("property_micro_rlt:")
                    ],
                    dtype=np.int64,
                )
        if micro_rlt_rows.size and rows.size != 1:
            raise ValueError(
                "restricted packet-core candidate requires exactly one "
                "scheduled objective"
            )
        if _deadline_closed("after_constraint_scope_selection"):
            return rows, stats
        error_stage = "objective_map"
        if binary_factor_count:
            (
                certificate_c,
                certificate_G,
                certificate_center_error,
            ) = _hz_binary_relaxed_output_frame(c, Gc, Gb)
        else:
            certificate_c = np.asarray(
                c, dtype=np.float64
            ).reshape(-1)
            certificate_G = Gc
            certificate_center_error = None
        if _deadline_closed("after_binary_output_frame"):
            return rows, stats
        q = _mat_dot_gen(C[rows], certificate_G)
        if q.shape != (rows.size, A.shape[1]) or not np.all(np.isfinite(q)):
            raise ValueError("GPU dual objective map is invalid")
        if _deadline_closed("after_objective_map"):
            return rows, stats

        # The optimizer never consumes row tags.  A lazy range preserves its
        # length audit without materializing one Python object per row (and,
        # critically, never one tag per rival-row pair).
        frame = OriginalFrameLP(
            A=A,
            rl=rl,
            ru=ru,
            lb=lb,
            ub=ub,
            row_tags=range(int(A.shape[0])),
        )
        candidate_constraint_rows = None
        source_rows = np.zeros(0, dtype=np.int64)
        bridge_rows = np.zeros(0, dtype=np.int64)
        bridge_only_rows = np.zeros(0, dtype=np.int64)
        if micro_rlt_rows.size:
            source_rows = np.asarray(
                (
                    []
                    if packet_core_seed_rows is None
                    else packet_core_seed_rows
                ),
                dtype=np.int64,
            ).reshape(-1)
            if (
                source_rows.size > 8
                or (
                    source_rows.size
                    and (
                        int(source_rows.min()) < 0
                        or int(source_rows.max()) >= int(A.shape[0])
                        or np.unique(source_rows).size
                        != source_rows.size
                    )
                )
            ):
                source_rows = np.zeros(0, dtype=np.int64)
            bridge_rows = np.asarray(
                (
                    []
                    if packet_core_bridge_rows is None
                    else packet_core_bridge_rows
                ),
                dtype=np.int64,
            ).reshape(-1)
            if (
                bridge_rows.size > 32_768
                or (
                    bridge_rows.size
                    and (
                        int(bridge_rows.min()) < 0
                        or int(bridge_rows.max()) >= int(A.shape[0])
                        or np.unique(bridge_rows).size
                        != bridge_rows.size
                        or (
                            constraint_row_tags is not None
                            and any(
                                str(
                                    constraint_row_tags[int(row)]
                                ).startswith("property_micro_rlt:")
                                for row in bridge_rows
                            )
                        )
                    )
                )
            ):
                bridge_rows = np.zeros(0, dtype=np.int64)
            bridge_only_rows = np.setdiff1d(
                bridge_rows,
                source_rows,
                assume_unique=False,
            )
            # Row selection is only a candidate-generation heuristic.  The
            # returned sparse multiplier is expanded to the full original row
            # order and independently checked on all A rows, so a bad
            # selection can only weaken the bound.
            candidate_constraint_rows = np.unique(
                np.concatenate(
                    [micro_rlt_rows, source_rows, bridge_rows]
                )
            )
            stats.update(
                {
                    "gpu_dual_candidate_constraint_scope": (
                        (
                            "property_micro_rlt_plus_constraint_cone_bridge"
                            if bridge_only_rows.size
                            else
                            "property_micro_rlt_generated_plus_source_rows"
                            if source_rows.size
                            else "property_micro_rlt_generated_rows"
                        )
                    ),
                    "gpu_dual_candidate_constraint_rows_selected": int(
                        candidate_constraint_rows.size
                    ),
                    "gpu_dual_candidate_constraint_rows_deferred": int(
                        A.shape[0] - candidate_constraint_rows.size
                    ),
                    "gpu_dual_packet_generated_rows_selected": int(
                        micro_rlt_rows.size
                    ),
                    "gpu_dual_packet_source_rows_selected": int(
                        source_rows.size
                    ),
                    "gpu_dual_packet_bridge_rows_selected": int(
                        bridge_only_rows.size
                    ),
                }
            )
        error_stage = "candidate_generation"
        if candidate_constraint_rows is None:
            candidates = batched_original_frame_row_duals(
                frame,
                q,
                device="cuda",
                steps=int(steps),
                learning_rate=float(learning_rate),
                candidate_rows=None,
                deadline=phase_deadline,
            )
        else:
            # Two bounded real probes showed that the CUDA sparse candidate
            # can remain inside an uninterruptible kernel past the parent and
            # worker deadlines.  The selected packet core is tiny enough for
            # the deterministic CPU coordinate wavefront.  It observes the
            # same absolute deadline between sparse updates and has no proof
            # authority; every multiplier is still checked on full A below.
            stats["gpu_dual_device"] = (
                "cpu_packet_bridge"
                if bridge_only_rows.size
                else "cpu_packet_core"
            )
            stats["gpu_dual_packet_core_cpu_fallback"] = True
            scale = np.maximum(np.max(np.abs(q), axis=1), 1.0)
            q_normalized = q / scale[:, None]
            if bridge_only_rows.size:
                # C65: jointly move the property residual through the packet
                # and already-live ordinary ReLU/materialization rows.  A
                # joint first stage is important for equality-band bridges:
                # the useful forward move can be support-flat until a packet
                # row fires, while optimizing the forward/reverse band alone
                # can simply shuttle the residual back.  A bounded packet-only
                # polish follows.  All selected rows are upper inequalities,
                # so nonnegative first/refinement multipliers add in the same
                # Lagrangian convention.  The combined multiplier is still
                # untrusted until the full-A long-double checker below.
                bridge_stage_rows = candidate_constraint_rows
                packet_stage_rows = np.unique(
                    np.concatenate([micro_rlt_rows, source_rows])
                )
                if (
                    bridge_stage_rows.size == 0
                    or packet_stage_rows.size == 0
                    or np.any(np.isfinite(rl[bridge_stage_rows]))
                    or np.any(~np.isfinite(ru[bridge_stage_rows]))
                    or np.any(np.isfinite(rl[packet_stage_rows]))
                    or np.any(~np.isfinite(ru[packet_stage_rows]))
                ):
                    raise ValueError(
                        "packet bridge stages require upper-only rows"
                    )

                def _selected_frame(selected_rows):
                    return OriginalFrameLP(
                        A=_sp.csr_matrix(
                            A[selected_rows, :], dtype=np.float64
                        ),
                        rl=np.asarray(
                            rl[selected_rows], dtype=np.float64
                        ),
                        ru=np.asarray(
                            ru[selected_rows], dtype=np.float64
                        ),
                        lb=lb,
                        ub=ub,
                        row_tags=tuple(
                            int(row) for row in selected_rows
                        ),
                    )

                bridge_frame = _selected_frame(bridge_stage_rows)
                if _deadline_closed("after_bridge_materialization"):
                    return rows, stats
                now = time.monotonic()
                bridge_deadline = min(
                    float(phase_deadline) - 0.40,
                    now + 0.85,
                )
                if bridge_deadline <= now:
                    return rows, stats
                bridge_wavefront = (
                    property_conditioned_coordinate_wavefront_duals(
                        bridge_frame,
                        q_normalized,
                        max_updates=max(
                            8, min(64, 4 * int(steps))
                        ),
                        frontier_topk=64,
                        refresh_batch=4,
                        deadline=bridge_deadline,
                    )
                )
                if bridge_wavefront.deadline_reached:
                    stats["gpu_dual_status"] = "deadline_reached"
                    stats["gpu_dual_deadline_reached"] = True
                    stats["gpu_dual_deadline_stage"] = (
                        "constraint_cone_bridge_wavefront"
                    )
                    stats["gpu_dual_elapsed_s"] = float(
                        time.monotonic() - started
                    )
                    return rows, stats
                bridge_d = np.asarray(
                    bridge_wavefront.d, dtype=np.float64
                )
                residual_q = q_normalized - np.asarray(
                    bridge_frame.A.transpose() @ bridge_d.transpose(),
                    dtype=np.float64,
                ).transpose()
                if not np.all(np.isfinite(residual_q)):
                    raise ValueError(
                        "packet bridge residual is non-finite"
                    )
                packet_frame = _selected_frame(packet_stage_rows)
                now = time.monotonic()
                packet_deadline = min(
                    float(phase_deadline) - 0.18,
                    now + 0.25,
                )
                if packet_deadline <= now:
                    return rows, stats
                packet_wavefront = (
                    property_conditioned_coordinate_wavefront_duals(
                        packet_frame,
                        residual_q,
                        max_updates=max(
                            8, min(32, 2 * int(steps))
                        ),
                        frontier_topk=64,
                        refresh_batch=4,
                        deadline=packet_deadline,
                    )
                )
                if packet_wavefront.deadline_reached:
                    stats["gpu_dual_status"] = "deadline_reached"
                    stats["gpu_dual_deadline_reached"] = True
                    stats["gpu_dual_deadline_stage"] = (
                        "packet_refine_wavefront"
                    )
                    stats["gpu_dual_elapsed_s"] = float(
                        time.monotonic() - started
                    )
                    return rows, stats

                d_normalized = np.zeros(
                    (rows.size, A.shape[0]), dtype=np.float64
                )
                d_normalized[:, bridge_stage_rows] += bridge_d
                d_normalized[:, packet_stage_rows] += np.asarray(
                    packet_wavefront.d, dtype=np.float64
                )
                if (
                    not np.all(np.isfinite(d_normalized))
                    or int(np.count_nonzero(d_normalized)) > 96
                ):
                    raise ValueError(
                        "packet bridge combined dual exceeds its cap"
                    )
                row_dual = -d_normalized * scale[:, None]
                initial_support = np.empty(
                    rows.size, dtype=np.float64
                )
                candidate_support = np.empty(
                    rows.size, dtype=np.float64
                )
                zero_row_dual = np.zeros(
                    A.shape[0], dtype=np.float64
                )
                for local_pos in range(rows.size):
                    initial = _hz_candidate_dual_support(
                        q=q[local_pos],
                        A=A,
                        rl=rl,
                        ru=ru,
                        lb=lb,
                        ub=ub,
                        row_dual=zero_row_dual,
                    )
                    candidate = _hz_candidate_dual_support(
                        q=q[local_pos],
                        A=A,
                        rl=rl,
                        ru=ru,
                        lb=lb,
                        ub=ub,
                        row_dual=row_dual[local_pos],
                    )
                    if (
                        initial is None
                        or candidate is None
                        or candidate > initial
                    ):
                        row_dual[local_pos].fill(0.0)
                        candidate = initial
                    initial_support[local_pos] = float(initial)
                    candidate_support[local_pos] = float(candidate)

                bridge_improvement = (
                    np.asarray(
                        bridge_wavefront.initial_support,
                        dtype=np.float64,
                    )
                    - np.asarray(
                        bridge_wavefront.candidate_support,
                        dtype=np.float64,
                    )
                ) * scale
                combined_improvement = (
                    initial_support - candidate_support
                )
                combined_selected = int(
                    np.count_nonzero(
                        np.any(d_normalized != 0.0, axis=0)
                    )
                )
                total_updates = int(
                    bridge_wavefront.updates
                    + packet_wavefront.updates
                )
                total_elapsed = float(
                    bridge_wavefront.elapsed_seconds
                    + packet_wavefront.elapsed_seconds
                )
                stats.update(
                    {
                        "gpu_dual_bridge_base_updates": int(
                            bridge_wavefront.updates
                        ),
                        "gpu_dual_bridge_packet_updates": int(
                            packet_wavefront.updates
                        ),
                        "gpu_dual_bridge_base_nnz": int(
                            np.count_nonzero(bridge_d)
                        ),
                        "gpu_dual_bridge_packet_nnz": int(
                            np.count_nonzero(packet_wavefront.d)
                        ),
                        "gpu_dual_bridge_base_support_improvement": float(
                            np.max(bridge_improvement)
                            if bridge_improvement.size
                            else 0.0
                        ),
                        "gpu_dual_bridge_combined_support_improvement": float(
                            np.max(combined_improvement)
                            if combined_improvement.size
                            else 0.0
                        ),
                    }
                )
                candidates = BatchedDualCandidates(
                    row_dual=row_dual,
                    initial_support=initial_support,
                    candidate_support=candidate_support,
                    selected_rows=candidate_constraint_rows.copy(),
                    device="cpu_packet_bridge",
                    dtype="numpy.float64",
                    steps_requested=0,
                    steps_completed=0,
                    elapsed_seconds=total_elapsed,
                    deadline_reached=False,
                    optimization_method=(
                        "property_conditioned_constraint_cone_"
                        "bridge_then_packet_v1"
                    ),
                    wavefront_updates=total_updates,
                    wavefront_support_improved_rows=int(
                        np.count_nonzero(
                            candidate_support < initial_support
                        )
                    ),
                    wavefront_best_improvement=float(
                        np.max(combined_improvement)
                        if combined_improvement.size
                        else 0.0
                    ),
                    wavefront_elapsed_seconds=total_elapsed,
                    wavefront_selected_constraint_count=(
                        combined_selected
                    ),
                )
            else:
                selected_frame = OriginalFrameLP(
                    A=_sp.csr_matrix(
                        A[candidate_constraint_rows, :],
                        dtype=np.float64,
                    ),
                    rl=np.asarray(
                        rl[candidate_constraint_rows], dtype=np.float64
                    ),
                    ru=np.asarray(
                        ru[candidate_constraint_rows], dtype=np.float64
                    ),
                    lb=lb,
                    ub=ub,
                    row_tags=tuple(
                        int(row) for row in candidate_constraint_rows
                    ),
                )
                if _deadline_closed("after_packet_core_materialization"):
                    return rows, stats
                wavefront = property_conditioned_coordinate_wavefront_duals(
                    selected_frame,
                    q_normalized,
                    max_updates=max(8, min(128, 4 * int(steps))),
                    frontier_topk=64,
                    refresh_batch=4,
                    deadline=phase_deadline,
                )
                if (
                    bool(wavefront.deadline_reached)
                    or _deadline_closed("after_packet_core_wavefront")
                ):
                    if stats["gpu_dual_deadline_stage"] is None:
                        stats["gpu_dual_status"] = "deadline_reached"
                        stats["gpu_dual_deadline_reached"] = True
                        stats["gpu_dual_deadline_stage"] = (
                            "packet_core_wavefront"
                        )
                        stats["gpu_dual_elapsed_s"] = float(
                            time.monotonic() - started
                        )
                    return rows, stats
                row_dual = np.zeros(
                    (rows.size, A.shape[0]), dtype=np.float64
                )
                row_dual[:, candidate_constraint_rows] = (
                    -np.asarray(wavefront.d, dtype=np.float64)
                    * scale[:, None]
                )
                candidates = BatchedDualCandidates(
                    row_dual=row_dual,
                    initial_support=(
                        np.asarray(
                            wavefront.initial_support, dtype=np.float64
                        )
                        * scale
                    ),
                    candidate_support=(
                        np.asarray(
                            wavefront.candidate_support, dtype=np.float64
                        )
                        * scale
                    ),
                    selected_rows=candidate_constraint_rows.copy(),
                    device="cpu_packet_core",
                    dtype="numpy.float64",
                    steps_requested=0,
                    steps_completed=0,
                    elapsed_seconds=float(wavefront.elapsed_seconds),
                    deadline_reached=bool(wavefront.deadline_reached),
                    optimization_method=str(wavefront.method),
                    wavefront_updates=int(wavefront.updates),
                    wavefront_support_improved_rows=int(
                        np.count_nonzero(
                            wavefront.candidate_support
                            < wavefront.initial_support
                        )
                    ),
                    wavefront_best_improvement=float(
                        np.max(
                            (
                                wavefront.initial_support
                                - wavefront.candidate_support
                            )
                            * scale
                        )
                        if wavefront.initial_support.size
                        else 0.0
                    ),
                    wavefront_elapsed_seconds=float(
                        wavefront.elapsed_seconds
                    ),
                    wavefront_selected_constraint_count=int(
                        wavefront.selected_constraint_count
                    ),
                )
        error_stage = "candidate_validation"
        if _deadline_closed("after_candidate_generation"):
            return rows, stats
        if (
            candidate_constraint_rows is None
            and not str(candidates.device).startswith("cuda")
        ) or (
            candidate_constraint_rows is not None
            and str(candidates.device)
            not in {"cpu_packet_core", "cpu_packet_bridge"}
        ):
            raise RuntimeError(
                "dual candidate generator used an unexpected device"
            )
        row_dual_matrix = np.asarray(
            candidates.row_dual,
            dtype=np.float64,
        )
        initial_support = np.asarray(
            candidates.initial_support,
            dtype=np.float64,
        ).reshape(-1)
        candidate_support = np.asarray(
            candidates.candidate_support,
            dtype=np.float64,
        ).reshape(-1)
        if (
            row_dual_matrix.shape != (rows.size, A.shape[0])
            or initial_support.size != rows.size
            or candidate_support.size != rows.size
            or not np.all(np.isfinite(initial_support))
            or not np.all(np.isfinite(candidate_support))
        ):
            raise ValueError("GPU dual candidate result has invalid shape/data")
        stats.update({
            "gpu_dual_steps_completed": int(candidates.steps_completed),
            "gpu_dual_deadline_reached": bool(candidates.deadline_reached),
            "gpu_dual_initial_support_min": float(np.min(initial_support)),
            "gpu_dual_initial_support_max": float(np.max(initial_support)),
            "gpu_dual_candidate_support_min": float(np.min(candidate_support)),
            "gpu_dual_candidate_support_max": float(np.max(candidate_support)),
            "gpu_dual_support_improved_rows": int(np.count_nonzero(
                candidate_support < initial_support
            )),
            "gpu_dual_support_best_improvement": float(np.max(
                initial_support - candidate_support
            )),
            "gpu_dual_device": str(candidates.device),
            "gpu_dual_wavefront_updates": int(
                getattr(candidates, "wavefront_updates", 0)
            ),
            "gpu_dual_wavefront_support_improved_rows": int(
                getattr(
                    candidates,
                    "wavefront_support_improved_rows",
                    0,
                )
            ),
            "gpu_dual_wavefront_best_improvement": float(
                getattr(candidates, "wavefront_best_improvement", 0.0)
            ),
            "gpu_dual_wavefront_elapsed_s": float(
                getattr(candidates, "wavefront_elapsed_seconds", 0.0)
            ),
            "gpu_dual_wavefront_selected_constraint_count": int(
                getattr(
                    candidates,
                    "wavefront_selected_constraint_count",
                    0,
                )
            ),
        })
        # C25: turn the sparse property wavefront into an active-set outer LP.
        # Candidate models see only selected rows; every returned multiplier
        # is expanded to the exact original row order before the independent
        # checker can use it.
        cg_receipts = []
        cg_improvements = []
        cg_started = time.monotonic()
        if candidate_constraint_rows is None:
            stats["gpu_dual_constraint_generation_status"] = "running"
            for local_pos in range(rows.size):
                now = time.monotonic()
                rows_left = max(1, int(rows.size) - int(local_pos))
                reserve = 0.35 * float(rows_left)
                available = float(phase_deadline) - now - reserve
                if available <= 0.05:
                    break
                candidate_deadline = min(
                    float(phase_deadline) - reserve,
                    now + min(3.0, available / float(rows_left)),
                )
                cg_row_dual, cg_stats = (
                    _hz_constraint_generation_dual_candidate(
                        q=q[local_pos],
                        A=A,
                        rl=rl,
                        ru=ru,
                        lb=lb,
                        ub=ub,
                        seed_row_dual=row_dual_matrix[local_pos],
                        deadline=candidate_deadline,
                    )
                )
                cg_stats = dict(cg_stats)
                cg_stats["objective_row"] = int(rows[local_pos])
                cg_receipts.append(cg_stats)
                stats[
                    "gpu_dual_constraint_generation_attempted_rows"
                ] += 1
                if cg_row_dual is None:
                    continue
                cg_support = cg_stats.get("best_support")
                if (
                    cg_support is not None
                    and np.isfinite(float(cg_support))
                    and float(cg_support)
                    < float(candidate_support[local_pos])
                ):
                    improvement = float(
                        candidate_support[local_pos]
                        - float(cg_support)
                    )
                    row_dual_matrix[local_pos] = np.asarray(
                        cg_row_dual, dtype=np.float64
                    )
                    candidate_support[local_pos] = float(cg_support)
                    cg_improvements.append(improvement)
                    stats[
                        "gpu_dual_constraint_generation_improved_rows"
                    ] += 1
            stats["gpu_dual_constraint_generation_status"] = "completed"
        else:
            # A packet-core multiplier is intentionally tested as-is against
            # the full original A.  Launching the generic full-frame cut
            # generator here defeats the bounded scope and can spend an
            # uninterruptible HiGHS presolve beyond the parent deadline.
            stats["gpu_dual_constraint_generation_status"] = (
                "skipped_restricted_constraint_scope"
            )
        stats["gpu_dual_constraint_generation_receipts"] = cg_receipts
        stats["gpu_dual_constraint_generation_elapsed_s"] = float(
            time.monotonic() - cg_started
        )
        if cg_improvements:
            stats[
                "gpu_dual_constraint_generation_best_improvement"
            ] = float(max(cg_improvements))
            stats["gpu_dual_candidate_support_min"] = float(
                np.min(candidate_support)
            )
            stats["gpu_dual_candidate_support_max"] = float(
                np.max(candidate_support)
            )
            stats["gpu_dual_support_improved_rows"] = int(
                np.count_nonzero(candidate_support < initial_support)
            )
            stats["gpu_dual_support_best_improvement"] = float(
                np.max(initial_support - candidate_support)
            )

        (
            row_dual_matrix,
            candidate_support,
            bridge_only_rows,
            pc_cbde_stats,
        ) = _hz_try_pc_cbde_candidate(
            rows=rows,
            row_topk=row_topk,
            candidates=candidates,
            candidate_constraint_rows=candidate_constraint_rows,
            bridge_only_rows=bridge_only_rows,
            micro_rlt_rows=micro_rlt_rows,
            source_rows=source_rows,
            constraint_row_tags=constraint_row_tags,
            frame=frame,
            q=q,
            row_dual_matrix=row_dual_matrix,
            candidate_support=candidate_support,
            A=A,
            rl=rl,
            ru=ru,
            lb=lb,
            ub=ub,
            certificate_c=certificate_c,
            certificate_G=certificate_G,
            certificate_center_error=certificate_center_error,
            C=C,
            t=t,
            phase_deadline=phase_deadline,
        )
        stats.update(pc_cbde_stats)
        if bool(
            pc_cbde_stats.get(
                "gpu_dual_pc_cbde_replaced_old_candidate", False
            )
        ):
            stats["gpu_dual_candidate_support_min"] = float(
                np.min(candidate_support)
            )
            stats["gpu_dual_candidate_support_max"] = float(
                np.max(candidate_support)
            )
            stats["gpu_dual_support_improved_rows"] = int(
                np.count_nonzero(candidate_support < initial_support)
            )
            stats["gpu_dual_support_best_improvement"] = float(
                np.max(initial_support - candidate_support)
            )
    except Exception as exc:
        # CUDA absence, OOM, import errors, malformed candidates, and all
        # other generator failures retain the complete rival set.
        logger.warning(
            "HybridZ GPU dual candidate failed closed at %s: %s: %s",
            error_stage,
            type(exc).__name__,
            str(exc).replace("\n", " ")[:512],
        )
        stats["gpu_dual_status"] = (
            "cuda_unavailable"
            if isinstance(exc, RuntimeError)
            and "CUDA" in str(exc)
            and "unavailable" in str(exc)
            else f"candidate_error:{type(exc).__name__}"
        )
        stats["gpu_dual_errors"] = 1
        stats["gpu_dual_error_type"] = type(exc).__name__
        stats["gpu_dual_error_message"] = (
            str(exc).replace("\n", " ")[:512]
        )
        stats["gpu_dual_error_stage"] = str(error_stage)
        stats["gpu_dual_elapsed_s"] = float(time.monotonic() - started)
        stats["gpu_dual_deadline_reached"] = bool(
            time.monotonic() >= phase_deadline
        )
        return rows, stats

    survivors: List[int] = []
    certified: List[int] = []
    certified_uppers: List[np.longdouble] = []
    certified_gaps: List[np.longdouble] = []
    checked_uppers: List[np.longdouble] = []
    topk = int(row_topk)
    for pos, r_raw in enumerate(rows):
        if time.monotonic() >= phase_deadline:
            survivors.extend(int(x) for x in rows[pos:])
            stats["gpu_dual_status"] = "deadline_reached"
            stats["gpu_dual_deadline_reached"] = True
            break

        # The candidate module returns row_dual=-d.  Sparsify d explicitly,
        # then restore the checker convention.  This makes the sign handoff
        # reviewable and prevents an accidental double-negation.
        d = -np.asarray(row_dual_matrix[pos], dtype=np.float64).reshape(-1)
        candidate_nnz = int(np.count_nonzero(d))
        stats["gpu_dual_candidate_dual_nnz_total"] += candidate_nnz
        stats["gpu_dual_candidate_dual_nnz_max"] = max(
            int(stats["gpu_dual_candidate_dual_nnz_max"]),
            candidate_nnz,
        )
        if topk > 0 and candidate_nnz > topk:
            scores = np.abs(d)
            # Non-finite candidates have no heuristic value.  Zeroing them is
            # a weakening; the independent checker would also project them to
            # zero.
            scores[~np.isfinite(scores)] = -np.inf
            selected = np.argpartition(scores, -topk)[-topk:]
            sparse_d = np.zeros_like(d)
            sparse_d[selected] = d[selected]
            sparse_d[~np.isfinite(sparse_d)] = 0.0
            d = sparse_d
        checked_nnz = int(np.count_nonzero(d))
        stats["gpu_dual_checked_dual_nnz_total"] += checked_nnz
        stats["gpu_dual_checked_dual_nnz_max"] = max(
            int(stats["gpu_dual_checked_dual_nnz_max"]),
            checked_nnz,
        )
        checked_indices = np.flatnonzero(d != 0.0).astype(
            np.int64, copy=False
        )
        if constraint_row_tags is None:
            generated_nnz = 0
        else:
            tags = tuple(constraint_row_tags)
            generated_nnz = (
                0
                if len(tags) != int(A.shape[0])
                else sum(
                    str(tags[int(row)]).startswith(
                        "property_micro_rlt:"
                    )
                    for row in checked_indices
                )
            )
        source_set = set(int(row) for row in source_rows)
        source_nnz = sum(
            int(row) in source_set for row in checked_indices
        )
        bridge_set = set(int(row) for row in bridge_only_rows)
        bridge_nnz = sum(
            int(row) in bridge_set for row in checked_indices
        )
        other_nnz = int(
            checked_nnz - generated_nnz - source_nnz - bridge_nnz
        )
        for prefix, count in (
            ("generated", generated_nnz),
            ("source", source_nnz),
            ("bridge", bridge_nnz),
            ("other", other_nnz),
        ):
            stats[f"gpu_dual_checked_{prefix}_nnz_total"] += int(count)
            stats[f"gpu_dual_checked_{prefix}_nnz_max"] = max(
                int(stats[f"gpu_dual_checked_{prefix}_nnz_max"]),
                int(count),
            )
        certificate_started = time.monotonic()
        certificate_elapsed_recorded = False
        try:
            upper, receipt = _hz_independent_lp_lagrangian_upper(
                c=certificate_c,
                Gc=certificate_G,
                C_row=C[int(r_raw)],
                threshold=t[int(r_raw)],
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                row_dual=-d,
                center_error=certificate_center_error,
            )
            stats["gpu_dual_independent_certificate_elapsed_s"] += float(
                time.monotonic() - certificate_started
            )
            certificate_elapsed_recorded = True
            stats["gpu_dual_certificate_attempted_rows"] += 1
            receipt_verified = (
                receipt.get("status") == "verified_upper"
            )
            if (
                upper is not None
                and receipt_verified
                and np.isfinite(upper)
            ):
                checked_uppers.append(np.longdouble(upper))
            if not receipt_verified:
                stats["gpu_dual_certificate_errors"] += 1
            center_guard = receipt.get("center_transform_guard_max")
            if (
                center_guard is not None
                and np.isfinite(float(center_guard))
            ):
                stats["gpu_dual_cert_center_transform_guard_max"] = max(
                    float(
                        stats[
                            "gpu_dual_cert_center_transform_guard_max"
                        ]
                    ),
                    float(center_guard),
                )
            # A certificate that finishes outside either absolute deadline is
            # not credited to the budgeted run.
            if time.monotonic() >= phase_deadline:
                survivors.extend(int(x) for x in rows[pos:])
                stats["gpu_dual_status"] = "deadline_reached"
                stats["gpu_dual_deadline_reached"] = True
                break
            if (
                upper is not None
                and receipt_verified
                and np.isfinite(upper)
                and np.longdouble(upper) < -np.longdouble(tol)
            ):
                certified.append(int(r_raw))
                certified_uppers.append(np.longdouble(upper))
                certified_gaps.append(
                    -np.longdouble(tol) - np.longdouble(upper)
                )
            else:
                survivors.append(int(r_raw))
        except Exception as exc:
            if not certificate_elapsed_recorded:
                stats[
                    "gpu_dual_independent_certificate_elapsed_s"
                ] += float(time.monotonic() - certificate_started)
            stats["gpu_dual_errors"] += 1
            stats["gpu_dual_certificate_errors"] += 1
            if stats["gpu_dual_error_type"] is None:
                stats["gpu_dual_error_type"] = type(exc).__name__
            survivors.append(int(r_raw))

        # Attribution is proof-neutral diagnostics.  Run it only after the
        # independent checker so it can never consume the row's certificate
        # opportunity, and leave a small explicit checker-priority reserve.
        attribution_started = time.monotonic()
        if phase_deadline - attribution_started <= 0.10:
            attribution = {
                "schema": "hz_candidate_support_attribution_v1",
                "status": "skipped_checker_priority",
                "proof_authority": False,
                "elapsed_seconds": 0.0,
                "tag_aggregation": "single_pass",
                "column_layer_support": [],
                "constraint_tag_contribution": [],
            }
        else:
            attribution = _hz_candidate_support_attribution(
                q=q[pos],
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                row_dual=-d,
                column_layer_ids=column_layer_ids,
                constraint_row_tags=constraint_row_tags,
                deadline=phase_deadline,
            )
        stats["gpu_dual_support_attributions"].append(
            {
                "objective_row": int(r_raw),
                **attribution,
            }
        )
        stats["gpu_dual_support_attribution_elapsed_s"] += float(
            time.monotonic() - attribution_started
        )
        if time.monotonic() >= phase_deadline:
            survivors.extend(int(x) for x in rows[pos + 1 :])
            stats["gpu_dual_status"] = "deadline_reached"
            stats["gpu_dual_deadline_reached"] = True
            break
    else:
        stats["gpu_dual_status"] = (
            "completed_with_errors"
            if int(stats["gpu_dual_errors"]) > 0
            else "completed"
        )

    out = np.asarray(survivors, dtype=np.int64)
    stats["gpu_dual_certified_rows"] = int(len(certified))
    stats["gpu_dual_certified_row_ids"] = [int(x) for x in certified]
    stats["gpu_dual_uncertified_rows"] = int(out.size)
    covered = np.asarray(certified + survivors, dtype=np.int64)
    stats["gpu_dual_coverage_ok"] = bool(
        covered.size == rows.size
        and np.unique(covered).size == rows.size
        and set(int(x) for x in covered) == set(int(x) for x in rows)
    )
    stats["gpu_dual_proof_authority"] = bool(certified)
    if certified_uppers:
        stats["gpu_dual_cert_upper_max"] = float(max(certified_uppers))
    if checked_uppers:
        stats["gpu_dual_checked_upper_min"] = float(min(checked_uppers))
        stats["gpu_dual_checked_upper_max"] = float(max(checked_uppers))
    if certified_gaps:
        stats["gpu_dual_cert_min_gap_to_cutoff"] = float(min(certified_gaps))
    stats["gpu_dual_elapsed_s"] = float(time.monotonic() - started)
    return out, stats


def _hz_persistent_lp_filter(
    *,
    c,
    Gc,
    Gb,
    C,
    t,
    candidate_rows,
    A,
    rl,
    ru,
    lb,
    ub,
    deadline,
    time_budget,
    tol,
    alternative_row_groups=None,
):
    """Prune OR rivals only through checked Lagrangian upper certificates.

    HiGHS remains an untrusted candidate generator.  Its row dual is checked
    from scratch against the complete original base-LP matrix by
    :func:`_hz_independent_lp_lagrangian_upper`.  Signed binary HZ factors are
    relaxed to the base model's ``z in [0, 1]`` coordinates and may authorize
    SAFE only.  A relaxed binary primal point never authorizes a witness.
    """

    rows = np.asarray(candidate_rows, dtype=np.int64).reshape(-1)
    binary_factor_count = int(Gb.shape[1])
    expected_factor_columns = int(Gc.shape[1]) + binary_factor_count
    structurally_complete_frame = (
        int(A.shape[1]) == expected_factor_columns
        and int(Gc.shape[0]) == int(np.asarray(c).size)
        and int(Gb.shape[0]) == int(np.asarray(c).size)
    )
    continuous_v1_eligible = (
        binary_factor_count == 0 and structurally_complete_frame
    )
    candidate_witness_eligible = bool(continuous_v1_eligible)
    row_to_alternative_group = {}
    if alternative_row_groups is not None:
        for group_index, group in enumerate(alternative_row_groups):
            for row in group:
                row_to_alternative_group[int(row)] = int(group_index)
    stats = {
        "lp_certificate_schema": _HZ_LP_CERTIFICATE_SCHEMA,
        "lp_input_rows": int(rows.size),
        "lp_pruned_rows": 0,
        "lp_certified_rows": 0,
        "lp_certified_row_ids": [],
        "lp_uncertified_rows": int(rows.size),
        "lp_candidate_empty_rows": 0,
        "lp_survivor_rows": int(rows.size),
        "lp_completed_rows": 0,
        "lp_full_resolve_rows": 0,
        "lp_optimal_certificate_candidates": 0,
        "lp_nonoptimal_certificate_candidates": 0,
        "lp_elapsed_s": 0.0,
        "lp_model_reused": False,
        "lp_persistent_model_builds": 0,
        "lp_basis_warmup_attempted": False,
        "lp_basis_warmup_seconds": 0.0,
        "lp_basis_warmup_run_status": None,
        "lp_basis_warmup_model_status": None,
        "lp_status": "not_started",
        "lp_alternative_groups_enabled": bool(
            alternative_row_groups is not None
        ),
        "lp_group_redundant_rows": 0,
        "lp_base_feasibility_conflict": False,
        "lp_proof_authority": False,
        # Preserve the legacy field's continuous-only meaning while exposing
        # the two authorities explicitly below.
        "lp_certificate_v1_eligible": bool(continuous_v1_eligible),
        "lp_safe_certificate_eligible": bool(
            structurally_complete_frame
        ),
        "lp_binary_relaxation_certificate_eligible": bool(
            binary_factor_count > 0 and structurally_complete_frame
        ),
        "lp_candidate_witness_eligible": bool(
            candidate_witness_eligible
        ),
        "lp_binary_factor_count": int(binary_factor_count),
        "lp_certificate_factor_columns": int(expected_factor_columns),
        "lp_candidate_witness_rows": 0,
        "lp_relaxed_nonwitness_rows": 0,
        "lp_validated_witness_rows": 0,
        "lp_validated_witness_row_id": None,
        "lp_certificate_attempted_rows": 0,
        "lp_zero_dual_certificate_skips": 0,
        "lp_solver_run_status_histogram": {},
        "lp_model_status_histogram": {},
        "lp_last_run_status": None,
        "lp_last_model_status": None,
        "lp_row_time_slice_min_s": None,
        "lp_row_time_slice_max_s": None,
        "lp_deadline_exhausted_stage": None,
        "lp_certificate_failures": 0,
        "lp_cert_max_upper": None,
        "lp_cert_min_gap_to_cutoff": None,
        "lp_cert_max_roundoff_guard": 0.0,
        "lp_cert_objective_guard_max": 0.0,
        "lp_cert_residual_guard_max": 0.0,
        "lp_cert_center_transform_guard_max": 0.0,
        "lp_cert_nnz_dual_total": 0,
        "lp_cert_nnz_dual_max": 0,
        "lp_cert_illegal_sign_projected": 0,
        "lp_cert_nonfinite_dual_zeroed": 0,
        "lp_cert_longdouble_nmant": int(np.finfo(np.longdouble).nmant),
        "lp_cert_longdouble_eps": float(np.finfo(np.longdouble).eps),
        "lp_matrix_small_value": 1e-12,
        "lp_matrix_input_nnz": int(A.nnz),
        "lp_matrix_loaded_nnz": 0,
        "lp_matrix_dropped_nnz": 0,
        "lp_matrix_dropped_abs_mass": 0.0,
        "lp_matrix_dropped_abs_max": 0.0,
        "lp_matrix_load_status": "not_started",
        "lp_matrix_load_warning": False,
        "lp_input_validation_s": 0.0,
        "lp_binary_frame_s": 0.0,
        "lp_candidate_csr_s": 0.0,
        "lp_highs_setup_s": 0.0,
        "lp_highs_add_columns_s": 0.0,
        "lp_highs_add_rows_s": 0.0,
        "lp_model_build_elapsed_s": 0.0,
        "lp_coverage_ok": True,
    }
    if rows.size == 0 or not (_HAS_HIGHSPY and _HAS_SCIPY):
        stats["lp_status"] = "unavailable" if rows.size else "empty_input"
        return rows, stats, None

    started = time.monotonic()
    lp_deadline = min(float(deadline), started + max(0.0, float(time_budget)))
    if lp_deadline <= started:
        stats["lp_status"] = "no_budget"
        return rows, stats, None

    model_build_started = time.monotonic()
    try:
        input_validation_started = time.monotonic()
        A = _sp.csr_matrix(A, dtype=np.float64)
        rl = np.asarray(rl, dtype=np.float64).reshape(-1)
        ru = np.asarray(ru, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        if (
            A.shape != (rl.size, lb.size)
            or ru.size != rl.size
            or ub.size != lb.size
            or (A.nnz and not np.all(np.isfinite(A.data)))
            or np.any(np.isnan(rl))
            or np.any(np.isnan(ru))
            or not np.all(np.isfinite(lb))
            or not np.all(np.isfinite(ub))
            or np.any(lb > ub)
            or A.shape[1] != expected_factor_columns
        ):
            raise ValueError("persistent LP base model has invalid numerical data")
        stats["lp_input_validation_s"] = float(
            time.monotonic() - input_validation_started
        )
        if time.monotonic() >= lp_deadline:
            stats["lp_status"] = "budget_exhausted"
            stats["lp_deadline_exhausted_stage"] = (
                "after_input_validation"
            )
            stats["lp_elapsed_s"] = float(time.monotonic() - started)
            stats["lp_model_build_elapsed_s"] = float(
                time.monotonic() - model_build_started
            )
            return rows, stats, None

        binary_frame_started = time.monotonic()
        if binary_factor_count:
            certificate_c, certificate_G, certificate_center_error = (
                _hz_binary_relaxed_output_frame(c, Gc, Gb)
            )
        else:
            # Preserve the established continuous-HZ arithmetic exactly.
            certificate_c = c
            certificate_G = Gc
            certificate_center_error = None
        stats["lp_binary_frame_s"] = float(
            time.monotonic() - binary_frame_started
        )
        stats["lp_safe_certificate_eligible"] = True
        stats["lp_binary_relaxation_certificate_eligible"] = bool(
            binary_factor_count
        )
        candidate_csr_started = time.monotonic()
        As, candidate_matrix_stats = _highs_candidate_csr(
            A,
            small_matrix_value=float(stats["lp_matrix_small_value"]),
        )
        stats["lp_candidate_csr_s"] = float(
            time.monotonic() - candidate_csr_started
        )
        stats.update({
            "lp_matrix_input_nnz": int(candidate_matrix_stats["input_nnz"]),
            "lp_matrix_loaded_nnz": int(candidate_matrix_stats["loaded_nnz"]),
            "lp_matrix_dropped_nnz": int(candidate_matrix_stats["dropped_nnz"]),
            "lp_matrix_dropped_abs_mass": float(
                candidate_matrix_stats["dropped_abs_mass"]
            ),
            "lp_matrix_dropped_abs_max": float(
                candidate_matrix_stats["dropped_abs_max"]
            ),
        })
        if time.monotonic() >= lp_deadline:
            stats["lp_status"] = "budget_exhausted"
            stats["lp_deadline_exhausted_stage"] = (
                "after_candidate_csr"
            )
            stats["lp_elapsed_s"] = float(time.monotonic() - started)
            stats["lp_model_build_elapsed_s"] = float(
                time.monotonic() - model_build_started
            )
            return rows, stats, None

        highs_setup_started = time.monotonic()
        h = _highspy.Highs()
        HS = _highspy.HighsStatus

        def _require_ok(status, operation):
            if status != HS.kOk:
                raise RuntimeError(f"{operation} returned {status}")

        _require_ok(h.setOptionValue("output_flag", False), "set output_flag")
        _require_ok(h.setOptionValue("presolve", "on"), "set presolve")
        _require_ok(
            h.setOptionValue(
                "small_matrix_value",
                float(stats["lp_matrix_small_value"]),
            ),
            "set LP small_matrix_value",
        )
        _require_ok(h.setOptionValue(
            "threads",
            _highs_process_threads(),
        ), "set threads")
        stats["lp_highs_setup_s"] = float(
            time.monotonic() - highs_setup_started
        )
        ncol = int(A.shape[1])
        all_cols = np.arange(ncol, dtype=np.int32)
        add_columns_started = time.monotonic()
        _require_ok(h.addCols(
            ncol,
            np.zeros(ncol, dtype=np.float64),
            np.asarray(lb, dtype=np.float64),
            np.asarray(ub, dtype=np.float64),
            0,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        ), "add LP columns")
        stats["lp_highs_add_columns_s"] = float(
            time.monotonic() - add_columns_started
        )
        if A.shape[0]:
            add_rows_started = time.monotonic()
            _require_ok(
                h.addRows(
                    As.shape[0],
                    rl,
                    ru,
                    As.nnz,
                    As.indptr.astype(np.int32),
                    As.indices.astype(np.int32),
                    As.data.astype(np.float64),
                ),
                "add LP rows",
            )
            stats["lp_highs_add_rows_s"] = float(
                time.monotonic() - add_rows_started
            )
            if (
                int(h.getNumRow()) != int(As.shape[0])
                or int(h.getNumCol()) != int(As.shape[1])
                or int(h.getNumNz()) != int(As.nnz)
            ):
                raise RuntimeError(
                    "persistent LP candidate matrix postcondition failed"
                )
        stats["lp_matrix_load_status"] = str(HS.kOk)
        stats["lp_model_reused"] = True
        stats["lp_persistent_model_builds"] = 1
        stats["lp_model_build_elapsed_s"] = float(
            time.monotonic() - model_build_started
        )
        if time.monotonic() >= lp_deadline:
            stats["lp_status"] = "budget_exhausted"
            stats["lp_deadline_exhausted_stage"] = (
                "after_highs_model_load"
            )
            stats["lp_elapsed_s"] = float(time.monotonic() - started)
            return rows, stats, None

        warmup_remaining = lp_deadline - time.monotonic()
        warmup_limit = min(
            2.0,
            max(0.0, 0.05 * float(time_budget)),
            max(0.0, warmup_remaining),
        )
        if warmup_limit > 1e-3:
            stats["lp_basis_warmup_attempted"] = True
            _require_ok(
                h.setOptionValue("time_limit", warmup_limit),
                "set LP basis warmup time_limit",
            )
            warmup_started = time.monotonic()
            warmup_run_status = h.run()
            warmup_model_status = h.getModelStatus()
            stats["lp_basis_warmup_seconds"] = float(
                time.monotonic() - warmup_started
            )
            stats["lp_basis_warmup_run_status"] = str(
                warmup_run_status
            )
            stats["lp_basis_warmup_model_status"] = str(
                warmup_model_status
            )
            if time.monotonic() >= lp_deadline:
                stats["lp_status"] = "budget_exhausted"
                stats["lp_deadline_exhausted_stage"] = (
                    "after_basis_warmup"
                )
                stats["lp_elapsed_s"] = float(
                    time.monotonic() - started
                )
                return rows, stats, None
    except Exception as exc:
        stats["lp_model_build_elapsed_s"] = float(
            time.monotonic() - model_build_started
        )
        stats["lp_status"] = f"build_error:{type(exc).__name__}"
        stats["lp_elapsed_s"] = float(time.monotonic() - started)
        return rows, stats, None

    reachable_rows: List[int] = []
    predicted_empty_rows: List[int] = []
    certified_rows: List[int] = []
    redundant_rows: List[int] = []
    certified_alternative_groups = set()
    evaluated_uppers: List[np.longdouble] = []
    certified_gaps: List[np.longdouble] = []
    validated_witness_xi: Optional[np.ndarray] = None

    def _validated_continuous_candidate(r: int, obj_thr: float):
        """Check a candidate-generator point against the original LP rows."""

        if not stats["lp_candidate_witness_eligible"]:
            return None
        try:
            v = np.asarray(
                h.getSolution().col_value,
                dtype=np.float64,
            ).reshape(-1)
        except Exception:
            return None
        if v.size != lb.size or not np.all(np.isfinite(v)):
            return None
        if (
            np.any(v < lb - 1e-7)
            or np.any(v > ub + 1e-7)
        ):
            return None
        if A.shape[0]:
            av = np.asarray(A @ v, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(av)):
                return None
            lower_vio = np.where(
                np.isfinite(rl),
                rl - av,
                -np.inf,
            )
            upper_vio = np.where(
                np.isfinite(ru),
                av - ru,
                -np.inf,
            )
            violation = np.maximum(
                np.maximum(lower_vio, upper_vio),
                0.0,
            )
            scale = 1.0 + np.maximum(
                np.abs(av),
                np.maximum(
                    np.where(np.isfinite(rl), np.abs(rl), 0.0),
                    np.where(np.isfinite(ru), np.abs(ru), 0.0),
                ),
            )
            if (
                not np.all(np.isfinite(violation))
                or float(np.max(violation)) > 5e-7
                or float(np.max(violation / scale)) > 5e-9
            ):
                return None
        objective_value = float(np.dot(-cost, v))
        if not np.isfinite(objective_value):
            return None
        margin = float(obj_thr + objective_value)
        if not np.isfinite(margin) or margin < -float(tol):
            return None
        return v.copy()

    def _checked_certificate_upper(r: int, row_dual):
        upper, receipt = _hz_independent_lp_lagrangian_upper(
            c=certificate_c,
            Gc=certificate_G,
            C_row=C[r],
            threshold=t[r],
            A=A,
            rl=rl,
            ru=ru,
            lb=lb,
            ub=ub,
            row_dual=row_dual,
            center_error=certificate_center_error,
        )
        stats["lp_certificate_attempted_rows"] += 1
        stats["lp_cert_nnz_dual_total"] += int(
            receipt.get("dual_nnz", 0)
        )
        stats["lp_cert_nnz_dual_max"] = max(
            int(stats["lp_cert_nnz_dual_max"]),
            int(receipt.get("dual_nnz", 0)),
        )
        stats["lp_cert_illegal_sign_projected"] += int(
            receipt.get("illegal_sign_projected", 0)
        )
        stats["lp_cert_nonfinite_dual_zeroed"] += int(
            receipt.get("nonfinite_dual_zeroed", 0)
        )
        for key, receipt_key in (
            ("lp_cert_max_roundoff_guard", "roundoff_guard"),
            ("lp_cert_objective_guard_max", "objective_guard"),
            ("lp_cert_residual_guard_max", "residual_guard"),
        ):
            value = receipt.get(receipt_key)
            if value is not None and np.isfinite(float(value)):
                stats[key] = max(float(stats[key]), float(value))
        center_guard = receipt.get("center_transform_guard_max")
        if center_guard is not None and np.isfinite(float(center_guard)):
            stats["lp_cert_center_transform_guard_max"] = max(
                float(stats["lp_cert_center_transform_guard_max"]),
                float(center_guard),
            )
        if upper is not None:
            if receipt.get("status") == "verified_upper":
                evaluated_uppers.append(np.longdouble(upper))
                return upper
        return None

    MS = _highspy.HighsModelStatus
    status_finished = False
    for pos, r_raw in enumerate(rows):
        now = time.monotonic()
        if now >= lp_deadline:
            reachable_rows.extend(int(x) for x in rows[pos:])
            stats["lp_status"] = "budget_exhausted"
            status_finished = True
            break
        r = int(r_raw)
        alternative_group = row_to_alternative_group.get(r)
        if (
            alternative_group is not None
            and alternative_group in certified_alternative_groups
        ):
            redundant_rows.append(r)
            stats["lp_group_redundant_rows"] += 1
            continue
        try:
            obj_b = _row_dot_gen(C[r], Gb)
            cost = -np.concatenate(
                [_row_dot_gen(C[r], Gc), 2.0 * obj_b],
            ).astype(np.float64, copy=False)
            const_z = float(C[r] @ c) - float(obj_b.sum())
            if (
                not np.all(np.isfinite(cost))
                or not np.isfinite(const_z)
                or not np.isfinite(float(t[r]))
            ):
                raise ValueError("persistent LP objective is non-finite")
            obj_thr = const_z - float(t[r])
            # Keep equality/tie cases and a numerical halo in the survivor
            # set.  Only an objective strictly beyond this guarded cutoff is
            # allowed to prune a rival.
            proof_guard = max(
                float(tol),
                1e-7 * (1.0 + abs(obj_thr) + float(np.abs(cost).sum()) * np.finfo(np.float64).eps),
            )
            guarded_thr = obj_thr + proof_guard
            if row_to_alternative_group:
                remaining_group_ids = {
                    row_to_alternative_group.get(int(value), -1)
                    for value in rows[pos:]
                } - certified_alternative_groups
                rows_remaining = max(1, len(remaining_group_ids))
            else:
                rows_remaining = max(1, int(rows.size) - int(pos))
            fair_share = max(
                1e-4,
                (lp_deadline - now) / float(rows_remaining),
            )
            # Leave part of each fair share for the independent original-A
            # certificate and bookkeeping.  A single hard rival must never
            # consume the budget reserved for the other 98 classes.
            row_time_limit = min(
                max(1e-4, lp_deadline - now),
                max(1e-3, 0.65 * fair_share),
            )
            stats["lp_row_time_slice_min_s"] = (
                row_time_limit
                if stats["lp_row_time_slice_min_s"] is None
                else min(
                    float(stats["lp_row_time_slice_min_s"]),
                    row_time_limit,
                )
            )
            stats["lp_row_time_slice_max_s"] = (
                row_time_limit
                if stats["lp_row_time_slice_max_s"] is None
                else max(
                    float(stats["lp_row_time_slice_max_s"]),
                    row_time_limit,
                )
            )
            _require_ok(
                h.setOptionValue("time_limit", row_time_limit),
                "set LP time_limit",
            )
            if stats["lp_safe_certificate_eligible"]:
                # The independently checked SAFE certificate benefits from a
                # full LP dual.  Only continuous states are also eligible to
                # turn the corresponding primal into a witness.
                _require_ok(
                    h.setOptionValue(
                        "objective_target",
                        -_highspy.kHighsInf,
                    ),
                    "clear LP objective_target",
                )
                _require_ok(
                    h.setOptionValue(
                        "objective_bound",
                        _highspy.kHighsInf,
                    ),
                    "clear LP objective_bound",
                )
            else:
                _require_ok(
                    h.setOptionValue("objective_target", float(guarded_thr)),
                    "set LP objective_target",
                )
                _require_ok(
                    h.setOptionValue("objective_bound", float(guarded_thr)),
                    "set LP objective_bound",
                )
            _require_ok(
                h.changeColsCost(ncol, all_cols, cost),
                "change LP objective",
            )
            run_status = h.run()
            st = h.getModelStatus()
            stats["lp_completed_rows"] += 1
            if stats["lp_safe_certificate_eligible"]:
                stats["lp_full_resolve_rows"] += 1
            run_key = str(run_status)
            model_key = str(st)
            stats["lp_last_run_status"] = run_key
            stats["lp_last_model_status"] = model_key
            run_hist = stats["lp_solver_run_status_histogram"]
            model_hist = stats["lp_model_status_histogram"]
            run_hist[run_key] = int(run_hist.get(run_key, 0)) + 1
            model_hist[model_key] = int(model_hist.get(model_key, 0)) + 1
            if time.monotonic() >= lp_deadline:
                reachable_rows.extend(int(x) for x in rows[pos:])
                stats["lp_status"] = "budget_exhausted"
                stats["lp_deadline_exhausted_stage"] = "after_solver_run"
                status_finished = True
                break

            predicted_empty = st == MS.kObjectiveBound
            if st == MS.kOptimal:
                obj = float(h.getInfo().objective_function_value)
                predicted_empty = np.isfinite(obj) and obj > guarded_thr
            if st == MS.kInfeasible:
                # The exact base state was checked feasible before this helper;
                # disagreement is a numerical conflict and must fail closed.
                stats["lp_status"] = "base_feasibility_conflict"
                stats["lp_base_feasibility_conflict"] = True
                reachable_rows = [int(x) for x in rows]
                predicted_empty_rows = []
                status_finished = True
                break
            if stats["lp_safe_certificate_eligible"]:
                # Even a time-limited/nonterminal solve may expose a useful
                # dual iterate.  Solver status has no authority: verify that
                # arbitrary multiplier against the original A with residual
                # box correction before pruning anything.
                if st == MS.kOptimal:
                    stats["lp_optimal_certificate_candidates"] += 1
                else:
                    stats["lp_nonoptimal_certificate_candidates"] += 1
                try:
                    row_dual = np.asarray(
                        h.getSolution().row_dual,
                        dtype=np.float64,
                    ).reshape(-1)
                except Exception:
                    row_dual = np.full(
                        A.shape[0],
                        np.nan,
                        dtype=np.float64,
                    )
                finite_nonzero_dual = np.any(
                    np.isfinite(row_dual) & (row_dual != 0.0)
                )
                if finite_nonzero_dual:
                    upper = _checked_certificate_upper(r, row_dual)
                else:
                    # A zero multiplier reproduces the already-failed cube
                    # bound, so recomputing a 10M-nnz residual cannot certify
                    # this survivor.
                    stats["lp_zero_dual_certificate_skips"] += 1
                    upper = None
                if time.monotonic() >= lp_deadline:
                    # A checker result which crosses its allocated LP slice
                    # is not credited, even if the global verifier deadline
                    # has not yet expired.
                    reachable_rows.extend(int(x) for x in rows[pos:])
                    stats["lp_status"] = "budget_exhausted"
                    stats["lp_deadline_exhausted_stage"] = (
                        "after_independent_certificate_check"
                    )
                    status_finished = True
                    break
                if (
                    upper is not None
                    and np.isfinite(upper)
                    and upper < -np.longdouble(tol)
                ):
                    stats["lp_candidate_empty_rows"] += 1
                    certified_rows.append(r)
                    if alternative_group is not None:
                        certified_alternative_groups.add(
                            int(alternative_group)
                        )
                    certified_gaps.append(
                        -np.longdouble(tol) - np.longdouble(upper)
                    )
                    continue
                stats["lp_certificate_failures"] += 1
                if stats["lp_candidate_witness_eligible"]:
                    stats["lp_candidate_witness_rows"] += 1
                    candidate = _validated_continuous_candidate(
                        r, obj_thr
                    )
                    if candidate is not None:
                        stats["lp_validated_witness_rows"] += 1
                        if validated_witness_xi is None:
                            validated_witness_xi = candidate
                            stats["lp_validated_witness_row_id"] = r
                else:
                    stats["lp_relaxed_nonwitness_rows"] += 1
                reachable_rows.append(r)
                continue

            if run_status != HS.kOk:
                reachable_rows.extend(int(x) for x in rows[pos:])
                stats["lp_status"] = "solver_nonterminal"
                status_finished = True
                break
            if predicted_empty:
                stats["lp_candidate_empty_rows"] += 1
                predicted_empty_rows.append(r)
            else:
                reachable_rows.append(r)
        except Exception as exc:
            reachable_rows.extend(int(x) for x in rows[pos:])
            stats["lp_status"] = f"query_error:{type(exc).__name__}"
            status_finished = True
            break
    if not status_finished:
        stats["lp_status"] = "complete"

    # LP-near/feasible rows are prioritized; potential-safe rows which failed
    # independent checking retain mandatory exact coverage.  Only certified
    # rows disappear from the returned survivor list.
    out = np.asarray(reachable_rows + predicted_empty_rows, dtype=np.int64)
    covered = np.asarray(
        reachable_rows
        + predicted_empty_rows
        + certified_rows
        + redundant_rows,
        dtype=np.int64,
    )
    coverage_ok = (
        covered.size == rows.size
        and np.unique(covered).size == rows.size
        and set(int(x) for x in covered) == set(int(x) for x in rows)
    )
    stats["lp_coverage_ok"] = bool(coverage_ok)
    if not coverage_ok:
        stats["lp_status"] = "coverage_internal_error"
        stats["lp_elapsed_s"] = float(time.monotonic() - started)
        stats["lp_proof_authority"] = False
        stats["lp_pruned_rows"] = 0
        stats["lp_certified_rows"] = 0
        stats["lp_certified_row_ids"] = []
        stats["lp_survivor_rows"] = int(rows.size)
        stats["lp_uncertified_rows"] = int(rows.size)
        return rows, stats, None

    stats["lp_pruned_rows"] = int(len(certified_rows))
    stats["lp_certified_rows"] = int(len(certified_rows))
    stats["lp_certified_row_ids"] = [int(x) for x in certified_rows]
    stats["lp_certified_alternative_groups"] = int(
        len(certified_alternative_groups)
    )
    stats["lp_uncertified_rows"] = int(out.size)
    stats["lp_proof_authority"] = bool(certified_rows)
    stats["lp_survivor_rows"] = int(out.size)
    if evaluated_uppers:
        stats["lp_cert_max_upper"] = float(max(evaluated_uppers))
    if certified_gaps:
        stats["lp_cert_min_gap_to_cutoff"] = float(min(certified_gaps))
    stats["lp_elapsed_s"] = float(time.monotonic() - started)
    return out, stats, validated_witness_xi


def _solver_csr_sha256(matrix) -> str:
    """Hash one canonical stored-float CSR matrix.

    This mirrors the Operator-HZ receipt hash without importing the builder
    into the solver layer.
    """

    csr = _sp.csr_matrix(matrix, dtype=np.float64, copy=False)
    csr.sum_duplicates()
    csr.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _hz_objbound_live_sha256(hz) -> Optional[str]:
    """Hash every live stored-float field relevant to an objbound proof."""

    try:
        if not isinstance(hz, SparseHZono):
            return None
        digest = hashlib.sha256()
        digest.update(b"act.hz_objbound.live_hz.v1\0")

        def update_array(name: str, value, dtype) -> None:
            array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
            if (
                np.issubdtype(array.dtype, np.floating)
                and not np.all(np.isfinite(array))
            ):
                raise ValueError(f"non-finite live array {name}")
            digest.update(name.encode("ascii") + b"\0")
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.dtype.str.encode("ascii") + b"\0")
            digest.update(array.tobytes(order="C"))

        def update_matrix(name: str, value) -> None:
            matrix = sp.csr_matrix(
                value, dtype=np.float64, copy=False
            )
            if (
                not matrix.has_canonical_format
                or (
                    matrix.nnz
                    and not np.all(np.isfinite(matrix.data))
                )
            ):
                raise ValueError(
                    f"non-canonical/non-finite live matrix {name}"
                )
            digest.update(name.encode("ascii") + b"\0")
            digest.update(
                _solver_csr_sha256(matrix.copy()).encode("ascii")
            )

        update_array("c", hz.c, np.float64)
        update_matrix("Gc", hz.Gc)
        update_matrix("Gb", hz.Gb)
        update_matrix("Ac", hz.Ac)
        update_matrix("Ab", hz.Ab)
        update_array("b", hz.b, np.float64)
        if hz.Auc is None or hz.Aub is None or hz.ub is None:
            if not (
                hz.Auc is None and hz.Aub is None and hz.ub is None
            ):
                return None
            digest.update(b"upper:none\0")
        else:
            digest.update(b"upper:present\0")
            update_matrix("Auc", hz.Auc)
            update_matrix("Aub", hz.Aub)
            update_array("ub", hz.ub, np.float64)
        if hz.col_ids is None:
            digest.update(b"col_ids:none\0")
        else:
            update_array("col_ids", hz.col_ids, np.int64)
        if hz.bcol_ids is None:
            digest.update(b"bcol_ids:none\0")
        else:
            update_array("bcol_ids", hz.bcol_ids, np.int64)
        return digest.hexdigest()
    except (
        TypeError,
        ValueError,
        OverflowError,
        UnicodeError,
    ):
        return None


def _hz_objbound_call_sha256(
    C: np.ndarray,
    thresholds: np.ndarray,
    safe_groups: Optional[Tuple[Tuple[int, ...], ...]],
    *,
    tol: float,
    is_unsafe_linear: bool,
    require_base_feasible: bool,
    base_witness_precheck: bool,
) -> str:
    """Bind a SAFE capability to one normalized objective and contract."""

    digest = hashlib.sha256()
    digest.update(b"act.hz_objbound.call.v1\0")
    for name, value in (("C", C), ("thresholds", thresholds)):
        array = np.ascontiguousarray(
            np.asarray(value, dtype=np.float64)
        )
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    digest.update(float(tol).hex().encode("ascii") + b"\0")
    digest.update(
        bytes(
            (
                int(bool(is_unsafe_linear)),
                int(bool(require_base_feasible)),
                int(bool(base_witness_precheck)),
            )
        )
    )
    if safe_groups is None:
        digest.update(b"groups:none\0")
    else:
        digest.update(
            np.asarray([len(safe_groups)], dtype=np.int64).tobytes()
        )
        for group in safe_groups:
            digest.update(
                np.asarray([len(group)], dtype=np.int64).tobytes()
            )
            digest.update(
                np.asarray(group, dtype=np.int64).tobytes()
            )
    return digest.hexdigest()


def _hz_clear_objbound_safe_capability(hz) -> None:
    for name in (
        "_solver_objbound_safe_token",
        "_solver_objbound_safe_receipt",
    ):
        try:
            delattr(hz, name)
        except AttributeError:
            pass


def hz_objbound_safe_capability_receipt(
    hz,
    C,
    thresholds,
    *,
    is_unsafe_linear: bool,
    tol: float,
    require_base_feasible: bool,
    base_witness_precheck: bool,
    safe_row_groups,
    expected_safe_group_count: Optional[int],
    require_binary_relaxation_lp: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return a validated copy of the current private SAFE capability.

    Diagnostic stats and a string verdict never authorize promotion.  This
    validator recomputes the live HZ/objective/group hashes and checks the
    private module token plus every base/group/deadline obligation issued by
    :func:`hz_objbound_decide`.
    """

    try:
        if not isinstance(hz, SparseHZono):
            return None
        Cn, tn = _spec_np(C, thresholds, hz.n_out)
        groups = _normalize_safe_row_groups(
            safe_row_groups, Cn.shape[0]
        )
        if groups is None:
            return None
        if (
            isinstance(expected_safe_group_count, (bool, np.bool_))
            or not isinstance(
                expected_safe_group_count, (int, np.integer)
            )
            or int(expected_safe_group_count) != len(groups)
        ):
            return None
        tol_value = float(tol)
        if not np.isfinite(tol_value) or tol_value < 0.0:
            return None
        if (
            getattr(hz, "_solver_objbound_safe_token", None)
            is not _HZ_OBJBOUND_SAFE_TOKEN
        ):
            return None
        receipt = getattr(
            hz, "_solver_objbound_safe_receipt", None
        )
        if not isinstance(receipt, dict):
            return None
        call_sha256 = _hz_objbound_call_sha256(
            Cn,
            tn,
            groups,
            tol=tol_value,
            is_unsafe_linear=bool(is_unsafe_linear),
            require_base_feasible=bool(require_base_feasible),
            base_witness_precheck=bool(base_witness_precheck),
        )
        live_sha256 = _hz_objbound_live_sha256(hz)
        if (
            live_sha256 is None
            or receipt.get("schema") != "hz_objbound_safe_v1"
            or receipt.get("proof_authority") is not True
            or receipt.get("witness_none") is not True
            or receipt.get("deadline_respected_at_issue") is not True
            or receipt.get("hz_live_sha256") != live_sha256
            or receipt.get("C_t_groups_sha256") != call_sha256
            or receipt.get("tol_hex") != tol_value.hex()
            or receipt.get("is_unsafe_linear")
            is not bool(is_unsafe_linear)
            or receipt.get("require_base_feasible")
            is not bool(require_base_feasible)
            or receipt.get("base_witness_precheck")
            is not bool(base_witness_precheck)
            or receipt.get("safe_row_group_count") != len(groups)
            or receipt.get("all_groups_resolved") is not True
            or receipt.get("safe_row_groups_resolved") != len(groups)
            or receipt.get("safe_row_groups_unresolved") != 0
            or receipt.get("all_rivals_covered") is not True
            or receipt.get("binary_factor_count") != hz.n_bin
        ):
            return None
        stats = getattr(hz, "_solver_objbound_stats", None)
        if (
            not isinstance(stats, dict)
            or stats.get("safe_row_groups_enabled") is not True
            or stats.get("safe_row_group_count") != len(groups)
            or stats.get("safe_row_groups_resolved") != len(groups)
            or stats.get("safe_row_groups_unresolved") != 0
            or stats.get("all_rivals_covered") is not True
        ):
            return None
        if require_base_feasible:
            if (
                receipt.get("base_discharge")
                != "FEASIBLE_CHECKED"
                or stats.get("base_feasibility_status") != "FEASIBLE"
            ):
                return None
        elif (
            receipt.get("base_discharge")
            != "SOUND_PHASE_COVER_MEMBER_V2"
            or stats.get("base_feasibility_status")
            != "EXACT_COVER_MEMBER_NOT_REQUIRED"
            or stats.get("exact_phase_cover_member") is not True
            or not _hz_exact_phase_cover_member(hz)
        ):
            return None
        if require_binary_relaxation_lp:
            persistent_binary_lp = bool(
                receipt.get("proof_stage")
                == "persistent_lp_lagrangian"
                and stats.get(
                    "lp_binary_relaxation_certificate_eligible"
                )
                is True
                and stats.get("lp_candidate_witness_eligible") is False
                and stats.get("lp_safe_certificate_eligible") is True
                and stats.get("lp_proof_authority") is True
                and stats.get("lp_coverage_ok") is True
                and stats.get("lp_binary_factor_count") == hz.n_bin
                and stats.get("lp_certificate_factor_columns")
                == hz.n_cont + hz.n_bin
            )
            gpu_binary_lp = bool(
                receipt.get("proof_stage") == "gpu_dual_lagrangian"
                and stats.get("gpu_dual_binary_relaxation_enabled")
                is True
                and stats.get("gpu_dual_candidate_witness_eligible")
                is False
                and stats.get("gpu_dual_proof_authority") is True
                and stats.get("gpu_dual_coverage_ok") is True
                and stats.get("gpu_dual_binary_factor_count")
                == hz.n_bin
                and int(
                    stats.get("gpu_dual_certificate_attempted_rows", 0)
                )
                > 0
            )
            if hz.n_bin <= 0 or not (
                persistent_binary_lp or gpu_binary_lp
            ):
                return None
        return dict(receipt)
    except (
        TypeError,
        ValueError,
        OverflowError,
        KeyError,
    ):
        return None


def _validated_row_constraint_prefix_models(
    *,
    hz,
    C,
    t,
    Ace,
    Acl,
    n_cont: int,
    n_bin: int,
):
    """Validate process-local row-to-constraint-prefix scheduling metadata.

    A valid entry binds an identity objective row to exact leading equality
    and inequality rows of the *actual* final HybridZ matrices.  The candidate
    LP retains every variable and its original box; it merely omits later
    constraint rows.  Consequently the full feasible set is a subset of the
    candidate set even if the frame metadata itself was supplied by an
    untrusted caller.
    """

    raw = getattr(hz, "_solver_row_constraint_prefix_frames", None)
    stats = {
        "row_prefix_lp_metadata_present": raw is not None,
        "row_prefix_lp_metadata_entries": 0,
        "row_prefix_lp_valid_entries": 0,
        "row_prefix_lp_rejected_entries": 0,
        "row_prefix_lp_rejection_histogram": {},
        "row_prefix_lp_model_count": 0,
    }
    if raw is None:
        return {}, stats
    if not isinstance(raw, dict):
        stats["row_prefix_lp_rejected_entries"] = 1
        stats["row_prefix_lp_rejection_histogram"] = {
            "metadata_not_dict": 1
        }
        return {}, stats
    stats["row_prefix_lp_metadata_entries"] = int(len(raw))
    if int(n_bin) != 0:
        # The independent LP proof path currently certifies continuous HZs
        # only.  Keep binary prefix models entirely outside proof authority.
        stats["row_prefix_lp_rejected_entries"] = int(len(raw))
        stats["row_prefix_lp_rejection_histogram"] = {
            "binary_state_ineligible": int(len(raw))
        }
        return {}, stats

    models = {}
    constraint_hash_cache = {}

    def reject(reason: str):
        stats["row_prefix_lp_rejected_entries"] += 1
        histogram = stats["row_prefix_lp_rejection_histogram"]
        histogram[reason] = int(histogram.get(reason, 0)) + 1

    for raw_key, value in raw.items():
        try:
            if (
                isinstance(raw_key, (bool, np.bool_))
                or not isinstance(raw_key, (int, np.integer))
                or not isinstance(value, dict)
            ):
                raise ValueError("entry_type")
            row = int(raw_key)
            if (
                value.get("schema")
                != "operator_hz_row_constraint_prefix_v1"
            ):
                raise ValueError("schema")
            for field in (
                "spec_row",
                "output_row",
                "stop_layer_id",
                "n_cont",
                "n_bin",
                "eq_rows",
                "ub_rows",
            ):
                item = value.get(field)
                if isinstance(item, (bool, np.bool_)) or not isinstance(
                    item, (int, np.integer)
                ):
                    raise ValueError(f"{field}_type")
            spec_row = int(value["spec_row"])
            output_row = int(value["output_row"])
            frame_n_cont = int(value["n_cont"])
            frame_n_bin = int(value["n_bin"])
            eq_rows = int(value["eq_rows"])
            ub_rows = int(value["ub_rows"])
            if row != spec_row or not 0 <= row < C.shape[0]:
                raise ValueError("spec_row")
            if not 0 <= output_row < C.shape[1]:
                raise ValueError("output_row")
            objective_nz = np.flatnonzero(C[row] != 0.0)
            if (
                objective_nz.size != 1
                or int(objective_nz[0]) != output_row
                or float(C[row, output_row]) != 1.0
                or float(t[row]) != 0.0
            ):
                raise ValueError("nonidentity_objective")
            if not (
                0 <= frame_n_cont <= int(n_cont)
                and frame_n_bin == 0
                and 0 <= eq_rows <= int(Ace.shape[0])
                and 0 <= ub_rows <= int(Acl.shape[0])
            ):
                raise ValueError("frame_range")
            eq_hash = value.get("eq_csr_sha256")
            ub_hash = value.get("ub_csr_sha256")
            if (
                not isinstance(eq_hash, str)
                or len(eq_hash) != 64
                or not isinstance(ub_hash, str)
                or len(ub_hash) != 64
            ):
                raise ValueError("hash_type")
            hash_key = (int(eq_rows), int(ub_rows))
            cached_hashes = constraint_hash_cache.get(hash_key)
            if cached_hashes is None:
                cached_hashes = (
                    _solver_csr_sha256(Ace[:eq_rows, :]),
                    _solver_csr_sha256(Acl[:ub_rows, :]),
                )
                constraint_hash_cache[hash_key] = cached_hashes
            actual_eq_hash, actual_ub_hash = cached_hashes
            if eq_hash != actual_eq_hash or ub_hash != actual_ub_hash:
                raise ValueError("constraint_hash")
            key = (
                int(eq_rows),
                int(ub_rows),
                str(actual_eq_hash),
                str(actual_ub_hash),
            )
            models.setdefault(key, []).append(int(row))
            stats["row_prefix_lp_valid_entries"] += 1
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            reason = str(exc) or type(exc).__name__
            reject(reason[:120])

    out = {
        key: np.asarray(sorted(set(rows)), dtype=np.int64)
        for key, rows in models.items()
    }
    stats["row_prefix_lp_model_count"] = int(len(out))
    return out, stats


def _ordered_row_constraint_prefix_models(prefix_models, input_rows):
    """Preserve the outer verifier's hardest-first row schedule.

    Prefix metadata is grouped and sorted by structural key during validation.
    Reusing that storage order here silently changed a property-group schedule
    such as ``[hard suffix, ..., easy suffix]`` back into objective-row order.
    Under a short GPU/LP budget this could spend the entire slice on easy
    rivals and never inspect the bottleneck.  The returned models and rows
    change scheduling only; no row, variable, or constraint is removed.
    """

    ordered_input = tuple(
        int(row)
        for row in np.asarray(input_rows, dtype=np.int64).reshape(-1)
    )
    input_rank = {
        int(row): int(position)
        for position, row in enumerate(ordered_input)
    }
    scheduled = []
    for key, model_rows in prefix_models.items():
        model_set = {
            int(row)
            for row in np.asarray(model_rows, dtype=np.int64).reshape(-1)
            if int(row) in input_rank
        }
        rows = np.asarray(
            [row for row in ordered_input if row in model_set],
            dtype=np.int64,
        )
        if rows.size:
            scheduled.append((int(input_rank[int(rows[0])]), key, rows))
    scheduled.sort(
        key=lambda item: (
            int(item[0]),
            -int(item[2].size),
            int(item[1][0]),
            int(item[1][1]),
        )
    )
    return [(key, rows) for _rank, key, rows in scheduled]


def _hz_row_constraint_prefix_lp_filter(
    *,
    c,
    Gc,
    Gb,
    C,
    t,
    candidate_rows,
    A,
    rl,
    ru,
    lb,
    ub,
    full_eq_rows: int,
    prefix_models,
    deadline,
    time_budget,
    tol,
    alternative_row_groups=None,
):
    """Certify selected rows on exact constraint-prefix relaxations.

    HiGHS remains candidate-only.  Each returned certification comes from the
    same independent long-double Lagrangian checker used by the full model,
    but its authority matrix is an exact row subset of ``A``.  Since all
    variables and box bounds are retained, omitting constraints is a pure
    outer relaxation.
    """

    started = time.monotonic()
    input_rows = np.asarray(candidate_rows, dtype=np.int64).reshape(-1)
    stats = {
        "row_prefix_lp_enabled": bool(prefix_models),
        "row_prefix_lp_input_rows": int(input_rows.size),
        "row_prefix_lp_eligible_rows": 0,
        "row_prefix_lp_certified_rows": 0,
        "row_prefix_lp_certified_row_ids": [],
        "row_prefix_lp_uncertified_rows": int(input_rows.size),
        "row_prefix_lp_model_count": int(len(prefix_models)),
        "row_prefix_lp_models_attempted": 0,
        "row_prefix_lp_full_constraint_rows": int(A.shape[0]),
        "row_prefix_lp_selected_constraint_rows_min": None,
        "row_prefix_lp_selected_constraint_rows_max": None,
        "row_prefix_lp_full_nnz": int(A.nnz),
        "row_prefix_lp_selected_nnz_min": None,
        "row_prefix_lp_selected_nnz_max": None,
        "row_prefix_lp_constraint_rows_dropped": 0,
        "row_prefix_lp_constraint_nnz_dropped": 0,
        "row_prefix_lp_proof_rule": (
            "independent_lagrangian_upper_over_exact_constraint_"
            "row_subset_with_all_variables_retained"
        ),
        "row_prefix_lp_coverage_ok": True,
        "row_prefix_lp_status": (
            "not_started" if prefix_models else "disabled"
        ),
        "row_prefix_lp_elapsed_s": 0.0,
        "row_prefix_lp_model_receipts": [],
    }
    if not prefix_models or input_rows.size == 0:
        stats["row_prefix_lp_status"] = (
            "empty_input" if input_rows.size == 0 else "disabled"
        )
        return input_rows, stats
    local_deadline = min(
        float(deadline),
        started + max(0.0, float(time_budget)),
    )
    if local_deadline <= started:
        stats["row_prefix_lp_status"] = "no_budget"
        return input_rows, stats

    ordered_models = _ordered_row_constraint_prefix_models(
        prefix_models, input_rows
    )
    eligible_union = set()
    certified = set()
    model_receipts = []
    prefix_row_counts = []
    prefix_nnzs = []
    for model_index, (key, scheduled_rows) in enumerate(ordered_models):
        now = time.monotonic()
        if now >= local_deadline:
            stats["row_prefix_lp_status"] = "budget_exhausted"
            break
        rows = np.asarray(
            [
                int(row)
                for row in scheduled_rows
                if int(row) not in certified
            ],
            dtype=np.int64,
        )
        if rows.size == 0:
            continue
        eq_rows, ub_rows = int(key[0]), int(key[1])
        selected_rows = np.concatenate(
            [
                np.arange(eq_rows, dtype=np.int64),
                int(full_eq_rows)
                + np.arange(ub_rows, dtype=np.int64),
            ]
        )
        if (
            np.any(selected_rows < 0)
            or np.any(selected_rows >= A.shape[0])
        ):
            stats["row_prefix_lp_coverage_ok"] = False
            stats["row_prefix_lp_status"] = "invalid_selected_rows"
            break
        prefix_A = _sp.csr_matrix(A[selected_rows, :], dtype=np.float64)
        prefix_rl = np.asarray(rl[selected_rows], dtype=np.float64)
        prefix_ru = np.asarray(ru[selected_rows], dtype=np.float64)
        prefix_row_counts.append(int(prefix_A.shape[0]))
        prefix_nnzs.append(int(prefix_A.nnz))
        eligible_union.update(int(row) for row in rows)
        remaining_models = max(1, len(ordered_models) - model_index)
        model_budget = max(
            0.0,
            (local_deadline - now) / float(remaining_models),
        )
        survivors, inner, _ignored_witness = _hz_persistent_lp_filter(
            c=c,
            Gc=Gc,
            Gb=Gb,
            C=C,
            t=t,
            candidate_rows=rows,
            A=prefix_A,
            rl=prefix_rl,
            ru=prefix_ru,
            lb=lb,
            ub=ub,
            deadline=local_deadline,
            time_budget=model_budget,
            tol=tol,
            alternative_row_groups=alternative_row_groups,
        )
        stats["row_prefix_lp_models_attempted"] += 1
        inner_certified = {
            int(row)
            for row in inner.get("lp_certified_row_ids", [])
        }
        if (
            not bool(inner.get("lp_coverage_ok", False))
            or bool(inner.get("lp_base_feasibility_conflict", False))
            or not inner_certified.issubset(set(int(row) for row in rows))
        ):
            inner_certified = set()
            stats["row_prefix_lp_coverage_ok"] = False
        certified.update(inner_certified)
        model_receipts.append(
            {
                "model_index": int(model_index),
                "eq_rows": int(eq_rows),
                "ub_rows": int(ub_rows),
                "constraint_rows": int(prefix_A.shape[0]),
                "constraint_nnz": int(prefix_A.nnz),
                "input_rows": int(rows.size),
                "survivor_rows": int(np.asarray(survivors).size),
                "certified_rows": int(len(inner_certified)),
                "certified_row_ids": sorted(inner_certified),
                "status": str(inner.get("lp_status")),
                "elapsed_s": float(inner.get("lp_elapsed_s", 0.0)),
                "optimal_certificate_candidates": int(
                    inner.get("lp_optimal_certificate_candidates", 0)
                ),
                "nonoptimal_certificate_candidates": int(
                    inner.get("lp_nonoptimal_certificate_candidates", 0)
                ),
                "certificate_attempted_rows": int(
                    inner.get("lp_certificate_attempted_rows", 0)
                ),
                "certificate_failures": int(
                    inner.get("lp_certificate_failures", 0)
                ),
                "cert_max_upper": inner.get("lp_cert_max_upper"),
                "matrix_loaded_nnz": int(
                    inner.get("lp_matrix_loaded_nnz", 0)
                ),
                "last_run_status": inner.get("lp_last_run_status"),
                "last_model_status": inner.get("lp_last_model_status"),
            }
        )
    remaining = np.asarray(
        [int(row) for row in input_rows if int(row) not in certified],
        dtype=np.int64,
    )
    stats["row_prefix_lp_eligible_rows"] = int(len(eligible_union))
    stats["row_prefix_lp_certified_rows"] = int(len(certified))
    stats["row_prefix_lp_certified_row_ids"] = sorted(certified)
    stats["row_prefix_lp_uncertified_rows"] = int(remaining.size)
    stats["row_prefix_lp_model_receipts"] = model_receipts
    if prefix_row_counts:
        stats["row_prefix_lp_selected_constraint_rows_min"] = int(
            min(prefix_row_counts)
        )
        stats["row_prefix_lp_selected_constraint_rows_max"] = int(
            max(prefix_row_counts)
        )
        stats["row_prefix_lp_constraint_rows_dropped"] = int(
            A.shape[0] - max(prefix_row_counts)
        )
    if prefix_nnzs:
        stats["row_prefix_lp_selected_nnz_min"] = int(min(prefix_nnzs))
        stats["row_prefix_lp_selected_nnz_max"] = int(max(prefix_nnzs))
        stats["row_prefix_lp_constraint_nnz_dropped"] = int(
            A.nnz - max(prefix_nnzs)
        )
    if stats["row_prefix_lp_status"] == "not_started":
        stats["row_prefix_lp_status"] = "complete"
    stats["row_prefix_lp_elapsed_s"] = float(
        time.monotonic() - started
    )
    return remaining, stats


def _hz_row_constraint_prefix_gpu_filter(
    *,
    c,
    Gc,
    Gb,
    C,
    t,
    candidate_rows,
    A,
    rl,
    ru,
    lb,
    ub,
    full_eq_rows: int,
    prefix_models,
    deadline,
    time_budget,
    steps,
    row_topk,
    learning_rate,
    tol,
    column_layer_ids=None,
    constraint_row_tags=None,
):
    """Run batched untrusted CUDA duals on exact constraint-prefix rows."""

    started = time.monotonic()
    input_rows = np.asarray(candidate_rows, dtype=np.int64).reshape(-1)
    stats = {
        "row_prefix_gpu_dual_enabled": bool(
            prefix_models and int(steps) > 0 and float(time_budget) > 0.0
        ),
        "row_prefix_gpu_dual_input_rows": int(input_rows.size),
        "row_prefix_gpu_dual_eligible_rows": 0,
        "row_prefix_gpu_dual_certified_rows": 0,
        "row_prefix_gpu_dual_certified_row_ids": [],
        "row_prefix_gpu_dual_uncertified_rows": int(input_rows.size),
        "row_prefix_gpu_dual_model_count": int(len(prefix_models)),
        "row_prefix_gpu_dual_models_attempted": 0,
        "row_prefix_gpu_dual_coverage_ok": True,
        "row_prefix_gpu_dual_status": (
            "not_started" if prefix_models else "disabled"
        ),
        "row_prefix_gpu_dual_elapsed_s": 0.0,
        "row_prefix_gpu_dual_model_receipts": [],
    }
    if (
        not prefix_models
        or input_rows.size == 0
        or int(steps) <= 0
        or float(time_budget) <= 0.0
    ):
        if input_rows.size == 0:
            stats["row_prefix_gpu_dual_status"] = "empty_input"
        elif not prefix_models:
            stats["row_prefix_gpu_dual_status"] = "disabled"
        else:
            stats["row_prefix_gpu_dual_status"] = "disabled_config"
        return input_rows, stats
    local_deadline = min(
        float(deadline),
        started + max(0.0, float(time_budget)),
    )
    if local_deadline <= started:
        stats["row_prefix_gpu_dual_status"] = "no_budget"
        return input_rows, stats

    ordered_models = _ordered_row_constraint_prefix_models(
        prefix_models, input_rows
    )
    eligible_union = set()
    certified = set()
    receipts = []
    for model_index, (key, scheduled_rows) in enumerate(ordered_models):
        now = time.monotonic()
        if now >= local_deadline:
            stats["row_prefix_gpu_dual_status"] = "budget_exhausted"
            break
        rows = np.asarray(
            [
                int(row)
                for row in scheduled_rows
                if int(row) not in certified
            ],
            dtype=np.int64,
        )
        if rows.size == 0:
            continue
        eq_rows, ub_rows = int(key[0]), int(key[1])
        selected_rows = np.concatenate(
            [
                np.arange(eq_rows, dtype=np.int64),
                int(full_eq_rows)
                + np.arange(ub_rows, dtype=np.int64),
            ]
        )
        if (
            np.any(selected_rows < 0)
            or np.any(selected_rows >= A.shape[0])
        ):
            stats["row_prefix_gpu_dual_coverage_ok"] = False
            stats["row_prefix_gpu_dual_status"] = (
                "invalid_selected_rows"
            )
            break
        prefix_A = _sp.csr_matrix(A[selected_rows, :], dtype=np.float64)
        prefix_rl = np.asarray(rl[selected_rows], dtype=np.float64)
        prefix_ru = np.asarray(ru[selected_rows], dtype=np.float64)
        prefix_constraint_tags = (
            None
            if constraint_row_tags is None
            else tuple(
                np.asarray(
                    tuple(constraint_row_tags), dtype=object
                ).reshape(-1)[selected_rows].tolist()
            )
        )
        eligible_union.update(int(row) for row in rows)
        remaining_models = max(1, len(ordered_models) - model_index)
        model_budget = max(
            0.0,
            (local_deadline - now) / float(remaining_models),
        )
        survivors, inner = _hz_gpu_dual_candidate_filter(
            c=c,
            Gc=Gc,
            Gb=Gb,
            C=C,
            t=t,
            candidate_rows=rows,
            A=prefix_A,
            rl=prefix_rl,
            ru=prefix_ru,
            lb=lb,
            ub=ub,
            deadline=local_deadline,
            time_budget=model_budget,
            steps=steps,
            row_topk=row_topk,
            learning_rate=learning_rate,
            tol=tol,
            column_layer_ids=column_layer_ids,
            constraint_row_tags=prefix_constraint_tags,
        )
        stats["row_prefix_gpu_dual_models_attempted"] += 1
        inner_certified = {
            int(row)
            for row in inner.get("gpu_dual_certified_row_ids", [])
        }
        if (
            not bool(inner.get("gpu_dual_coverage_ok", False))
            or not inner_certified.issubset(set(int(row) for row in rows))
        ):
            inner_certified = set()
            stats["row_prefix_gpu_dual_coverage_ok"] = False
        certified.update(inner_certified)
        receipts.append(
            {
                "model_index": int(model_index),
                "eq_rows": int(eq_rows),
                "ub_rows": int(ub_rows),
                "constraint_rows": int(prefix_A.shape[0]),
                "constraint_nnz": int(prefix_A.nnz),
                "input_rows": int(rows.size),
                "first_scheduled_row": (
                    int(rows[0]) if rows.size else None
                ),
                "scheduled_row_ids_head": [
                    int(row) for row in rows[:64]
                ],
                "survivor_rows": int(np.asarray(survivors).size),
                "certified_rows": int(len(inner_certified)),
                "certified_row_ids": sorted(inner_certified),
                "status": str(inner.get("gpu_dual_status")),
                "elapsed_s": float(inner.get("gpu_dual_elapsed_s", 0.0)),
                "steps_completed": int(
                    inner.get("gpu_dual_steps_completed", 0)
                ),
                "deadline_reached": bool(
                    inner.get("gpu_dual_deadline_reached", False)
                ),
                "support_improved_rows": int(
                    inner.get("gpu_dual_support_improved_rows", 0)
                ),
                "support_best_improvement": inner.get(
                    "gpu_dual_support_best_improvement"
                ),
                "initial_support_min": inner.get(
                    "gpu_dual_initial_support_min"
                ),
                "candidate_support_min": inner.get(
                    "gpu_dual_candidate_support_min"
                ),
                "candidate_support_max": inner.get(
                    "gpu_dual_candidate_support_max"
                ),
                "certificate_attempted_rows": int(
                    inner.get("gpu_dual_certificate_attempted_rows", 0)
                ),
                "certificate_errors": int(
                    inner.get("gpu_dual_certificate_errors", 0)
                ),
                "cert_upper_max": inner.get("gpu_dual_cert_upper_max"),
                "checked_upper_min": inner.get(
                    "gpu_dual_checked_upper_min"
                ),
                "checked_upper_max": inner.get(
                    "gpu_dual_checked_upper_max"
                ),
                "candidate_dual_nnz_total": int(
                    inner.get("gpu_dual_candidate_dual_nnz_total", 0)
                ),
                "checked_dual_nnz_total": int(
                    inner.get("gpu_dual_checked_dual_nnz_total", 0)
                ),
                "wavefront_updates": int(
                    inner.get("gpu_dual_wavefront_updates", 0)
                ),
                "wavefront_support_improved_rows": int(
                    inner.get(
                        "gpu_dual_wavefront_support_improved_rows", 0
                    )
                ),
                "wavefront_best_improvement": inner.get(
                    "gpu_dual_wavefront_best_improvement"
                ),
                "wavefront_elapsed_s": float(
                    inner.get("gpu_dual_wavefront_elapsed_s", 0.0)
                ),
                "wavefront_selected_constraint_count": int(
                    inner.get(
                        "gpu_dual_wavefront_selected_constraint_count", 0
                    )
                ),
                "constraint_generation_attempted_rows": int(
                    inner.get(
                        "gpu_dual_constraint_generation_attempted_rows", 0
                    )
                ),
                "constraint_generation_improved_rows": int(
                    inner.get(
                        "gpu_dual_constraint_generation_improved_rows", 0
                    )
                ),
                "constraint_generation_best_improvement": inner.get(
                    "gpu_dual_constraint_generation_best_improvement"
                ),
                "constraint_generation_elapsed_s": float(
                    inner.get(
                        "gpu_dual_constraint_generation_elapsed_s", 0.0
                    )
                ),
                "constraint_generation_receipts": inner.get(
                    "gpu_dual_constraint_generation_receipts", []
                ),
                "support_attributions": inner.get(
                    "gpu_dual_support_attributions", []
                ),
                "error_type": inner.get("gpu_dual_error_type"),
                "error_stage": inner.get("gpu_dual_error_stage"),
            }
        )
    remaining = np.asarray(
        [int(row) for row in input_rows if int(row) not in certified],
        dtype=np.int64,
    )
    stats["row_prefix_gpu_dual_eligible_rows"] = int(len(eligible_union))
    stats["row_prefix_gpu_dual_certified_rows"] = int(len(certified))
    stats["row_prefix_gpu_dual_certified_row_ids"] = sorted(certified)
    stats["row_prefix_gpu_dual_uncertified_rows"] = int(remaining.size)
    stats["row_prefix_gpu_dual_model_receipts"] = receipts
    if stats["row_prefix_gpu_dual_status"] == "not_started":
        stats["row_prefix_gpu_dual_status"] = "complete"
    stats["row_prefix_gpu_dual_elapsed_s"] = float(
        time.monotonic() - started
    )
    return remaining, stats


def hz_objbound_decide(hz, C, thresholds, *, is_unsafe_linear: bool,
                       time_limit: float = 15.0, tol: float = 1e-9,
                       mip_start_xi=None, require_base_feasible: bool = True,
                       base_feas_time_limit: Optional[float] = None,
                       base_witness_precheck: bool = True,
                       lp_prefilter_fraction: Optional[float] = None,
                       lp_prefilter_max_seconds: Optional[float] = None,
                       gpu_dual_steps: int = 0,
                       gpu_dual_time_limit: float = 0.0,
                       gpu_dual_row_topk: int = 0,
                       gpu_dual_learning_rate: float = 0.08,
                       safe_row_groups=None,
                       expected_safe_group_count: Optional[int] = None,
                       safe_group_mixture_grid_bits: int = 0):
    """Decide a linear property with proof-firewalled HybridZ candidates.

    Returns ``(verdict, witness_xi)`` with verdict in
    ``{SAFE, UNSAFE, UNKNOWN}``.  Cube enclosures and independently recomputed
    long-double Lagrangian bounds may authorize SAFE.  HiGHS/SCIP statuses
    never do so directly.  Solver incumbents are checked against the original
    stored-float HZ and production callers must additionally replay the
    decoded input against raw ONNX/VNNLIB before exposing FALSIFIED.
    ``time_limit`` is one shared wall budget for base feasibility, optional
    CUDA dual candidates, persistent LP certification, and any binary cutoff
    queries.  CUDA is strictly default-off and has no proof authority.

    ``safe_row_groups`` is a safe-only alternative-plane mode.  It must
    partition every objective row exactly once.  A property group is
    certified when any one of its affine upper-plane rows receives an
    independent negative upper bound; every group must be certified for SAFE,
    and this mode never emits UNSAFE.
    """
    call_started = time.monotonic()
    _hz_clear_objbound_safe_capability(hz)
    try:
        total_budget = max(0.0, float(time_limit))
    except (TypeError, ValueError):
        return ("UNKNOWN", None)
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        return ("UNKNOWN", None)
    if not (0.0 <= tol < float("inf")):
        return ("UNKNOWN", None)
    try:
        gpu_dual_steps = int(gpu_dual_steps)
        gpu_dual_time_limit = float(gpu_dual_time_limit)
        gpu_dual_row_topk = int(gpu_dual_row_topk)
        gpu_dual_learning_rate = float(gpu_dual_learning_rate)
    except (TypeError, ValueError, OverflowError):
        return ("UNKNOWN", None)
    if (
        gpu_dual_steps < 0
        or gpu_dual_row_topk < 0
        or not (0.0 <= gpu_dual_time_limit < float("inf"))
        or not (0.0 < gpu_dual_learning_rate < float("inf"))
    ):
        return ("UNKNOWN", None)
    if isinstance(safe_group_mixture_grid_bits, (bool, np.bool_)):
        return ("UNKNOWN", None)
    try:
        safe_group_mixture_grid_bits = int(
            safe_group_mixture_grid_bits
        )
    except (TypeError, ValueError, OverflowError):
        return ("UNKNOWN", None)
    if not (0 <= safe_group_mixture_grid_bits <= 24):
        return ("UNKNOWN", None)
    deadline = time.monotonic() + total_budget
    if not (_HAS_HIGHSPY and _HAS_SCIPY):
        return ("UNKNOWN", None)
    if total_budget <= 0.0:
        return ("UNKNOWN", None)
    c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = hz_np_sparse(hz)
    try:
        C, t = _spec_np(C, thresholds, c.size)
    except (TypeError, ValueError, OverflowError):
        return ("UNKNOWN", None)
    try:
        safe_groups = _normalize_safe_row_groups(
            safe_row_groups, C.shape[0]
        )
    except (TypeError, ValueError):
        return ("UNKNOWN", None)
    if safe_groups is not None and is_unsafe_linear:
        return ("UNKNOWN", None)
    if expected_safe_group_count is not None:
        if isinstance(expected_safe_group_count, (bool, np.bool_)):
            return ("UNKNOWN", None)
        try:
            expected_group_count = int(expected_safe_group_count)
        except (TypeError, ValueError, OverflowError):
            return ("UNKNOWN", None)
        if (
            safe_groups is None
            or expected_group_count <= 0
            or len(safe_groups) != expected_group_count
        ):
            return ("UNKNOWN", None)
    capability_C = np.asarray(C, dtype=np.float64).copy()
    capability_t = np.asarray(t, dtype=np.float64).copy()
    capability_safe_groups = (
        None
        if safe_groups is None
        else tuple(
            tuple(int(row) for row in group)
            for group in safe_groups
        )
    )
    capability_call_sha256 = _hz_objbound_call_sha256(
        capability_C,
        capability_t,
        capability_safe_groups,
        tol=tol,
        is_unsafe_linear=bool(is_unsafe_linear),
        require_base_feasible=bool(require_base_feasible),
        base_witness_precheck=bool(base_witness_precheck),
    )
    exact_cover_child = _hz_exact_phase_cover_member(hz)
    if (
        safe_groups is not None
        and not require_base_feasible
        and not exact_cover_child
    ):
        # Grouped planes are a one-sided proof path and may never certify a
        # vacuous/unchecked ordinary base state.  The sole exception is a
        # module-authenticated sound outward phase child of a constructively
        # nonempty parent.  Universal safety may be proved vacuously on an
        # empty child because the verifier separately enumerates every sibling
        # assignment before lifting the result back to that parent.
        return ("UNKNOWN", None)
    original_safe_row_plane_count = int(C.shape[0])
    mixture_candidate_deadline = min(
        deadline, time.monotonic() + 5.0
    )
    try:
        C, t, safe_groups, safe_group_mixture_receipt = (
            _augment_safe_groups_with_dyadic_mixtures(
                c,
                Gc,
                Gb,
                C,
                t,
                safe_groups,
                grid_bits=safe_group_mixture_grid_bits,
                candidate_deadline=mixture_candidate_deadline,
            )
        )
    except (TypeError, ValueError, OverflowError):
        return ("UNKNOWN", None)
    if time.monotonic() >= deadline:
        return ("UNKNOWN", None)
    group_certified_rows = set()
    group_winners = {}

    def _record_group_winners(row_ids, *, stage, upper_bounds=None):
        if safe_groups is None:
            return
        available = set(int(row) for row in row_ids)
        for group_index, group in enumerate(safe_groups):
            if int(group_index) in group_winners:
                continue
            eligible_rows = [
                int(row) for row in group if int(row) in available
            ]
            if not eligible_rows:
                continue
            if upper_bounds is None:
                winner = min(eligible_rows)
                bound = None
            else:
                winner = min(
                    eligible_rows,
                    key=lambda row: (float(upper_bounds[row]), int(row)),
                )
                bound = float(upper_bounds[winner])
            group_winners[int(group_index)] = {
                "group": int(group_index),
                "row": int(winner),
                "stage": str(stage),
                "upper": bound,
            }

    def _group_winner_receipt():
        return [
            dict(group_winners[index])
            for index in sorted(group_winners)
        ]

    def _group_resolved_count():
        if safe_groups is None:
            return 0
        return sum(
            any(int(row) in group_certified_rows for row in group)
            for group in safe_groups
        )

    def _unresolved_group_rows(rows):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if safe_groups is None:
            return rows
        available = set(int(row) for row in rows)
        unresolved = []
        for group in safe_groups:
            if any(int(row) in group_certified_rows for row in group):
                continue
            unresolved.extend(
                int(row) for row in group if int(row) in available
            )
        return np.asarray(unresolved, dtype=np.int64)

    def _schedule_group_rows(rows, cube_upper):
        rows = _unresolved_group_rows(rows)
        if safe_groups is None or rows.size == 0:
            return rows
        available = set(int(row) for row in rows)
        candidates = []
        for group_index, group in enumerate(safe_groups):
            if any(int(row) in group_certified_rows for row in group):
                continue
            ordered = sorted(
                (int(row) for row in group if int(row) in available),
                key=lambda row: (float(cube_upper[row]), int(row)),
            )
            if ordered:
                candidates.append(
                    (
                        int(group_index),
                        float(cube_upper[ordered[0]]),
                        ordered,
                    )
                )
        # Hard groups first, but visit every group's best plane before any
        # second-choice baseline/candidate.  This preserves rival fairness.
        candidates.sort(key=lambda item: (-item[1], item[0]))
        scheduled = []
        max_choices = max(len(item[2]) for item in candidates)
        for choice in range(max_choices):
            for _group_index, _hardness, ordered in candidates:
                if choice < len(ordered):
                    scheduled.append(int(ordered[choice]))
        return np.asarray(scheduled, dtype=np.int64)

    ng, nb = Gc.shape[1], Gb.shape[1]
    setattr(
        hz,
        "_solver_objbound_stats",
        {
            "base_feasibility_status": (
                "NOT_CHECKED"
                if require_base_feasible
                else (
                    "EXACT_COVER_MEMBER_NOT_REQUIRED"
                    if exact_cover_child
                    else "NOT_REQUIRED"
                )
            ),
            "base_feasibility_reason": (
                "pending"
                if require_base_feasible
                else (
                    "exact_phase_child_of_constructively_nonempty_parent"
                    if exact_cover_child
                    else "caller_disabled"
                )
            ),
            "exact_phase_cover_member": bool(exact_cover_child),
            "exact_phase_cover_vacuous_child_allowed": bool(
                exact_cover_child and not require_base_feasible
            ),
            "safe_row_groups_enabled": bool(safe_groups is not None),
            "safe_row_group_count": (
                int(len(safe_groups)) if safe_groups is not None else 0
            ),
            "safe_row_plane_count": int(C.shape[0]),
            "safe_row_original_plane_count": int(
                original_safe_row_plane_count
            ),
            "safe_row_dyadic_mixture": dict(
                safe_group_mixture_receipt
            ),
            "safe_row_group_proof_rule": (
                "each_property_group_has_one_independently_certified_"
                "affine_upper_plane"
                if safe_groups is not None
                else None
            ),
            # Proof-neutral observability for the bounded parent call.  These
            # fields never participate in a verdict or capability check; they
            # only identify where the shared wall budget was consumed.
            "parent_stage_timing_schema": (
                "hz_objbound_parent_stage_timing_v1"
            ),
            "parent_stage_timings_diagnostic_only": True,
            "parent_stage_timings_proof_authority": False,
            "parent_last_stage": "initialized",
            "parent_exit_reason": None,
            "parent_elapsed_s": float(
                max(0.0, time.monotonic() - call_started)
            ),
            "parent_cube_complete_elapsed_s": None,
            "parent_base_matrix_materialization_status": "not_started",
            "parent_base_matrix_materialization_elapsed_s": 0.0,
            "parent_base_matrix_rows": None,
            "parent_base_matrix_columns": None,
            "parent_base_matrix_nnz": None,
            "parent_base_matrix_error_type": None,
            "parent_persistent_lp_status": "not_started",
            "parent_persistent_lp_elapsed_s": 0.0,
            "parent_persistent_lp_input_rows": None,
            "parent_persistent_lp_output_rows": None,
            "parent_persistent_lp_budget_s": None,
            "parent_persistent_lp_error_type": None,
        },
    )

    def _record_parent_stage(
        stage: str,
        *,
        exit_reason: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        """Update diagnostic-only parent timing without proof authority."""

        observed = time.monotonic() if now is None else float(now)
        stats = getattr(hz, "_solver_objbound_stats", None)
        if not isinstance(stats, dict):
            return
        stats["parent_last_stage"] = str(stage)
        stats["parent_elapsed_s"] = float(
            max(0.0, observed - call_started)
        )
        if exit_reason is not None:
            stats["parent_exit_reason"] = str(exit_reason)

    def _parent_unknown(exit_reason: str):
        observed = time.monotonic()
        _record_parent_stage(
            str(exit_reason),
            exit_reason=str(exit_reason),
            now=observed,
        )
        return ("UNKNOWN", None)

    def _return_group_safe(stage: str):
        """Issue the only capability accepted for grouped SAFE promotion."""

        now = time.monotonic()
        if (
            safe_groups is None
            or capability_safe_groups is None
            or now >= deadline
        ):
            return ("UNKNOWN", None)
        stats = getattr(hz, "_solver_objbound_stats", None)
        if not isinstance(stats, dict):
            return ("UNKNOWN", None)
        resolved = int(stats.get("safe_row_groups_resolved", -1))
        unresolved = int(stats.get("safe_row_groups_unresolved", -1))
        if resolved != len(safe_groups) or unresolved != 0:
            return ("UNKNOWN", None)
        if require_base_feasible:
            if stats.get("base_feasibility_status") != "FEASIBLE":
                return ("UNKNOWN", None)
            base_discharge = "FEASIBLE_CHECKED"
        else:
            if (
                not exact_cover_child
                or stats.get("base_feasibility_status")
                != "EXACT_COVER_MEMBER_NOT_REQUIRED"
            ):
                return ("UNKNOWN", None)
            base_discharge = "SOUND_PHASE_COVER_MEMBER_V2"

        stats["all_rivals_covered"] = True
        live_sha256 = _hz_objbound_live_sha256(hz)
        if live_sha256 is None:
            return ("UNKNOWN", None)
        receipt = {
            "schema": "hz_objbound_safe_v1",
            "proof_authority": True,
            "verdict": "SAFE",
            "witness_none": True,
            "proof_stage": str(stage),
            "hz_live_sha256": live_sha256,
            "C_t_groups_sha256": capability_call_sha256,
            "tol_hex": float(tol).hex(),
            "is_unsafe_linear": bool(is_unsafe_linear),
            "require_base_feasible": bool(require_base_feasible),
            "base_witness_precheck": bool(base_witness_precheck),
            "base_discharge": base_discharge,
            "safe_row_group_count": int(len(safe_groups)),
            "safe_row_groups_resolved": int(resolved),
            "safe_row_groups_unresolved": int(unresolved),
            "all_groups_resolved": True,
            "all_rivals_covered": True,
            "binary_factor_count": int(nb),
            "requested_budget_seconds": float(total_budget),
            "solver_elapsed_seconds_at_issue": float(
                now - call_started
            ),
            "remaining_seconds_at_issue": float(deadline - now),
            "deadline_respected_at_issue": True,
        }
        setattr(
            hz,
            "_solver_objbound_safe_receipt",
            receipt,
        )
        setattr(
            hz,
            "_solver_objbound_safe_token",
            _HZ_OBJBOUND_SAFE_TOKEN,
        )
        stats.update(
            {
                "safe_capability_schema": "hz_objbound_safe_v1",
                "safe_capability_issued": True,
                "safe_capability_stage": str(stage),
                "safe_capability_base_discharge": base_discharge,
            }
        )
        return ("SAFE", None)

    if time.monotonic() >= deadline:
        return ("UNKNOWN", None)

    if require_base_feasible:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return ("UNKNOWN", None)
        try:
            requested_btl = (
                10.0
                if base_feas_time_limit is None
                else max(0.0, float(base_feas_time_limit))
            )
        except (TypeError, ValueError):
            return ("UNKNOWN", None)
        btl = min(remaining, requested_btl)
        if btl <= 0.0:
            return ("UNKNOWN", None)
        base_status, base_reason = hz_base_feasibility(hz, time_limit=btl)
        getattr(hz, "_solver_objbound_stats").update({
            "base_feasibility_status": str(base_status),
            "base_feasibility_reason": str(base_reason),
        })
        if time.monotonic() >= deadline:
            return _parent_unknown("deadline_after_base_feasibility")
        if base_status != "FEASIBLE":
            return ("UNKNOWN", None)
        if base_witness_precheck and safe_groups is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return ("UNKNOWN", None)
            base_xi, _ = hz_base_witness(hz, time_limit=min(remaining, btl))
            if time.monotonic() >= deadline:
                return ("UNKNOWN", None)
            if base_xi is not None and _hz_spec_unsafe_at_xi(
                hz, C, t, base_xi, is_unsafe_linear=is_unsafe_linear, tol=tol
            ):
                return ("UNSAFE", base_xi)

    if time.monotonic() >= deadline:
        return ("UNKNOWN", None)
    if ng + nb == 0:
        from fractions import Fraction

        row = _hz_exact_point_margins(c, C, t)
        tol_exact = Fraction.from_float(tol)
        if time.monotonic() >= deadline:
            return _parent_unknown(
                "deadline_before_exact_point_evaluation"
            )
        if safe_groups is not None:
            safe = all(
                any(row[int(index)] < -tol_exact for index in group)
                for group in safe_groups
            )
            getattr(hz, "_solver_objbound_stats").update(
                {
                    "safe_row_groups_resolved": int(
                        sum(
                            any(
                                row[int(index)] < -tol_exact
                                for index in group
                            )
                            for group in safe_groups
                        )
                    ),
                    "safe_row_groups_unresolved": int(
                        len(safe_groups)
                        - sum(
                            any(
                                row[int(index)] < -tol_exact
                                for index in group
                            )
                            for group in safe_groups
                        )
                    ),
                }
            )
            return (
                _return_group_safe("exact_point_fraction")
                if safe
                else ("UNKNOWN", None)
            )
        if is_unsafe_linear:
            if max(row) > tol_exact:
                return ("SAFE", None)
            return ("UNSAFE", np.zeros(0))
        if max(row) < -tol_exact:
            return ("SAFE", None)
        return ("UNSAFE", np.zeros(0))

    survivor_rows = np.arange(C.shape[0], dtype=np.int64)
    if not is_unsafe_linear:
        _record_parent_stage("cube")
        cube_started = time.monotonic()
        cube_ub, cube_guard = _hz_cube_row_upper_bounds(c, Gc, Gb, C, t)
        cube_elapsed = time.monotonic() - cube_started
        cube_pruned = cube_ub < -float(tol)
        if safe_groups is not None:
            cube_certified_rows = [
                int(row) for row in np.flatnonzero(cube_pruned)
            ]
            _record_group_winners(
                cube_certified_rows,
                stage="cube",
                upper_bounds=cube_ub,
            )
            group_certified_rows.update(cube_certified_rows)
            mixture_selected = safe_group_mixture_receipt.get(
                "selected", []
            )
            mixture_rows = [
                int(item["appended_row"])
                for item in mixture_selected
            ]
            original_group_best = np.asarray(
                [
                    min(
                        float(cube_ub[int(row)])
                        for row in group
                        if int(row) < original_safe_row_plane_count
                    )
                    for group in safe_groups
                ],
                dtype=np.float64,
            )
            selected_group_best = np.asarray(
                [
                    float(original_group_best[int(item["group"])])
                    for item in mixture_selected
                ],
                dtype=np.float64,
            )
            selected_endpoint_best = np.asarray(
                [
                    min(
                        float(cube_ub[int(item["left_row"])]),
                        float(cube_ub[int(item["right_row"])]),
                    )
                    for item in mixture_selected
                ],
                dtype=np.float64,
            )
            selected_mixed_upper = np.asarray(
                [
                    float(cube_ub[int(item["appended_row"])])
                    for item in mixture_selected
                ],
                dtype=np.float64,
            )
            mixture_improvements = np.asarray(
                selected_group_best - selected_mixed_upper,
                dtype=np.float64,
            )
            endpoint_improvements = np.asarray(
                selected_endpoint_best - selected_mixed_upper,
                dtype=np.float64,
            )
            guarded_records_digest = hashlib.sha256()
            for selected_index, item in enumerate(mixture_selected):
                appended_row = int(item["appended_row"])
                item.update(
                    {
                        "guarded_endpoint_best_upper": float(
                            selected_endpoint_best[selected_index]
                        ),
                        "guarded_original_group_best_upper": float(
                            selected_group_best[selected_index]
                        ),
                        "guarded_mixed_cube_upper": float(
                            selected_mixed_upper[selected_index]
                        ),
                        "guarded_cube_gain": float(
                            mixture_improvements[selected_index]
                        ),
                        "guarded_endpoint_gain": float(
                            endpoint_improvements[selected_index]
                        ),
                        "guarded_cube_certified": bool(
                            cube_pruned[appended_row]
                        ),
                    }
                )
                guarded_records_digest.update(
                    np.asarray(
                        [
                            int(item["group"]),
                            int(item["left_row"]),
                            int(item["right_row"]),
                            appended_row,
                        ],
                        dtype=np.int64,
                    ).tobytes()
                )
                guarded_records_digest.update(
                    np.asarray(
                        [
                            selected_endpoint_best[selected_index],
                            selected_group_best[selected_index],
                            selected_mixed_upper[selected_index],
                            mixture_improvements[selected_index],
                        ],
                        dtype=np.float64,
                    ).tobytes()
                )

            def _guarded_array_sha256(values):
                values = np.asarray(
                    values, dtype=np.float64
                ).reshape(-1)
                array_digest = hashlib.sha256()
                array_digest.update(
                    np.asarray(
                        [values.size], dtype=np.int64
                    ).tobytes()
                )
                array_digest.update(values.tobytes(order="C"))
                return array_digest.hexdigest()

            def _guarded_summary(prefix, values):
                values = np.asarray(
                    values, dtype=np.float64
                ).reshape(-1)
                if values.size == 0:
                    return {
                        f"{prefix}_count": 0,
                        f"{prefix}_min": None,
                        f"{prefix}_max": None,
                        f"{prefix}_sum": 0.0,
                        f"{prefix}_sha256": (
                            _guarded_array_sha256(values)
                        ),
                    }
                return {
                    f"{prefix}_count": int(values.size),
                    f"{prefix}_min": float(np.min(values)),
                    f"{prefix}_max": float(np.max(values)),
                    f"{prefix}_sum": float(np.sum(values)),
                    f"{prefix}_sha256": (
                        _guarded_array_sha256(values)
                    ),
                }

            safe_group_mixture_receipt.update(
                {
                    "guarded_cube_improved_groups": int(
                        np.count_nonzero(mixture_improvements > 0.0)
                    ),
                    "guarded_cube_improvement_sum": float(
                        np.sum(
                            np.maximum(mixture_improvements, 0.0)
                        )
                    ),
                    "guarded_cube_improvement_max": float(
                        np.max(
                            np.maximum(mixture_improvements, 0.0)
                        )
                        if mixture_improvements.size else 0.0
                    ),
                    **_guarded_summary(
                        "guarded_all_group_best_cube_upper",
                        original_group_best,
                    ),
                    **_guarded_summary(
                        "guarded_selected_group_best_cube_upper",
                        selected_group_best,
                    ),
                    **_guarded_summary(
                        "guarded_selected_endpoint_best_cube_upper",
                        selected_endpoint_best,
                    ),
                    **_guarded_summary(
                        "guarded_selected_mixed_cube_upper",
                        selected_mixed_upper,
                    ),
                    **_guarded_summary(
                        "guarded_cube_gain",
                        mixture_improvements,
                    ),
                    **_guarded_summary(
                        "guarded_endpoint_gain",
                        endpoint_improvements,
                    ),
                    "guarded_selected_records_sha256": (
                        guarded_records_digest.hexdigest()
                    ),
                    "guarded_cube_authority": (
                        "outward_hz_cube_checker"
                    ),
                    "guarded_cube_certified_rows": int(
                        sum(
                            bool(cube_pruned[row])
                            for row in mixture_rows
                        )
                    ),
                    "guarded_cube_certified_groups": int(
                        len(
                            {
                                int(item["group"])
                                for item in mixture_selected
                                if cube_pruned[
                                    int(item["appended_row"])
                                ]
                            }
                        )
                    ),
                }
            )
            survivor_rows = _schedule_group_rows(
                np.flatnonzero(~cube_pruned), cube_ub
            )
        else:
            survivor_rows = np.flatnonzero(~cube_pruned).astype(
                np.int64, copy=False
            )
        # Hardest-looking rivals first.  This changes only scheduling: every
        # survivor still has to be proved empty before SAFE is returned.
        if survivor_rows.size and safe_groups is None:
            survivor_rows = survivor_rows[
                np.argsort(-cube_ub[survivor_rows], kind="stable")
            ]
        objbound_stats = getattr(hz, "_solver_objbound_stats")
        objbound_stats.update({
            "cube_total_rows": int(C.shape[0]),
            "cube_pruned_rows": int(np.count_nonzero(cube_pruned)),
            "cube_survivor_rows": int(survivor_rows.size),
            "cube_max_upper": float(np.max(cube_ub)) if cube_ub.size else None,
            "cube_min_upper": float(np.min(cube_ub)) if cube_ub.size else None,
            "cube_max_roundoff_guard": (
                float(np.max(cube_guard)) if cube_guard.size else 0.0
            ),
            "cube_elapsed_s": float(cube_elapsed),
            "all_rivals_covered": True,
            "safe_row_cube_certified_groups": int(
                _group_resolved_count()
            ),
            "safe_row_groups_resolved": int(_group_resolved_count()),
            "safe_row_groups_unresolved": (
                int(len(safe_groups) - _group_resolved_count())
                if safe_groups is not None else 0
            ),
            "safe_row_group_winners": _group_winner_receipt(),
            "safe_row_dyadic_mixture": dict(
                safe_group_mixture_receipt
            ),
        })
        cube_complete = time.monotonic()
        objbound_stats["parent_cube_complete_elapsed_s"] = float(
            max(0.0, cube_complete - call_started)
        )
        _record_parent_stage("cube_complete", now=cube_complete)
        if time.monotonic() >= deadline:
            return _parent_unknown("deadline_after_cube")
        if (
            safe_groups is not None
            and _group_resolved_count() == len(safe_groups)
        ):
            return _return_group_safe("cube_outward_support")
        if survivor_rows.size == 0:
            return (
                ("UNKNOWN", None)
                if safe_groups is not None
                else ("SAFE", None)
            )

    _record_parent_stage("base_matrix_materialization")
    matrix_started = time.monotonic()
    parent_stats = getattr(hz, "_solver_objbound_stats", None)
    if isinstance(parent_stats, dict):
        parent_stats["parent_base_matrix_materialization_status"] = (
            "running"
        )
    constraint_row_tags = getattr(
        hz, "_solver_constraint_row_tags", None
    )
    upper_compaction_plan = None
    try:
        if constraint_row_tags is not None:
            constraint_row_tags = tuple(constraint_row_tags)
            if len(constraint_row_tags) != int(
                Ace.shape[0] + Acl.shape[0]
            ):
                raise ValueError(
                    "constraint-row tag count does not match source rows"
                )
            equality_tags = constraint_row_tags[: int(Ace.shape[0])]
            upper_tags = constraint_row_tags[int(Ace.shape[0]) :]
            upper_compaction_plan = (
                _tagged_upper_band_compaction_plan(
                    Acl,
                    Abl,
                    upper_tags,
                )
            )
            if upper_compaction_plan is not None:
                constraint_row_tags = (
                    equality_tags
                    + upper_compaction_plan["compacted_tags"]
                )
        A, rl, ru, lb, ub, integ = _base_milp_matrices_from_blocks(
            Gc,
            Gb,
            Ace,
            Abe,
            be,
            Acl,
            Abl,
            bl,
            upper_compaction_plan=upper_compaction_plan,
        )
    except Exception as exc:
        matrix_finished = time.monotonic()
        parent_stats = getattr(hz, "_solver_objbound_stats", None)
        if isinstance(parent_stats, dict):
            parent_stats.update(
                {
                    "parent_base_matrix_materialization_status": "error",
                    "parent_base_matrix_materialization_elapsed_s": float(
                        max(0.0, matrix_finished - matrix_started)
                    ),
                    "parent_base_matrix_error_type": type(exc).__name__,
                }
            )
        _record_parent_stage(
            "base_matrix_materialization_error",
            exit_reason="base_matrix_materialization_error",
            now=matrix_finished,
        )
        logger.exception("HybridZ base matrix construction failed closed")
        return ("UNKNOWN", None)
    matrix_finished = time.monotonic()
    parent_stats = getattr(hz, "_solver_objbound_stats", None)
    if isinstance(parent_stats, dict):
        parent_stats.update(
            {
                "parent_base_matrix_materialization_status": "completed",
                "parent_base_matrix_materialization_elapsed_s": float(
                    max(0.0, matrix_finished - matrix_started)
                ),
                "parent_base_matrix_rows": int(A.shape[0]),
                "parent_base_matrix_columns": int(A.shape[1]),
                "parent_base_matrix_nnz": int(A.nnz),
                "parent_ranged_row_pair_count": int(
                    0
                    if upper_compaction_plan is None
                    else upper_compaction_plan["pair_count"]
                ),
                "parent_ranged_row_compaction_applied": bool(
                    upper_compaction_plan is not None
                ),
            }
        )
    _record_parent_stage(
        "base_matrix_materialization_complete",
        now=matrix_finished,
    )
    if time.monotonic() >= deadline:
        return _parent_unknown(
            "deadline_after_base_matrix_materialization"
        )
    column_layer_ids = getattr(
        hz, "_solver_continuous_column_layer_ids", None
    )
    if column_layer_ids is not None:
        column_layer_ids = np.asarray(
            column_layer_ids, dtype=np.int64
        ).reshape(-1)
        if column_layer_ids.size != A.shape[1]:
            column_layer_ids = None
    if constraint_row_tags is not None:
        constraint_row_tags = tuple(constraint_row_tags)
        if len(constraint_row_tags) != A.shape[0]:
            constraint_row_tags = None
    packet_core_seed_rows = (
        np.zeros(0, dtype=np.int64)
        if constraint_row_tags is None
        else _hz_property_micro_rlt_source_candidate_rows(
            hz,
            constraint_row_tags=constraint_row_tags,
            matrix_row_count=int(A.shape[0]),
        )
    )
    # C66 closed the static tag/layer-nearest bridge: on the one authorized
    # real probe it selected 8,390 ordinary rows, added 0.700 s, and reached
    # exactly the C62/C63 packet plateau with zero generated-row multiplier.
    # Keep the proof-neutral selector and its causal toys for audit history,
    # but do not silently enable a failed scheduling heuristic in the live
    # path.  A future bridge may populate this only after a bounded
    # row<->column incidence selector and family-ablation gate are integrated.
    packet_core_bridge_rows = np.zeros(0, dtype=np.int64)
    prefix_models, prefix_metadata_stats = (
        _validated_row_constraint_prefix_models(
            hz=hz,
            C=C,
            t=t,
            Ace=Ace,
            Acl=Acl,
            n_cont=ng,
            n_bin=nb,
        )
    )
    if safe_groups is None:
        # Prefix rows are a one-sided alternative-plane optimization.  Never
        # alter the ordinary SAFE/UNSAFE path even if a caller attaches
        # otherwise structurally valid metadata.
        prefix_models = {}
        prefix_metadata_stats["row_prefix_lp_disabled_reason"] = (
            "requires_safe_row_groups"
        )
    obj_stats = getattr(hz, "_solver_objbound_stats", None)
    if isinstance(obj_stats, dict):
        obj_stats.update(prefix_metadata_stats)

    if not is_unsafe_linear:
        try:
            lp_fraction_raw = (
                os.environ.get("HZ_LP_PREFILTER_FRACTION", "0.20")
                if lp_prefilter_fraction is None
                else lp_prefilter_fraction
            )
            lp_cap_raw = (
                os.environ.get("HZ_LP_PREFILTER_MAX_SECONDS", "8.0")
                if lp_prefilter_max_seconds is None
                else lp_prefilter_max_seconds
            )
            lp_fraction = max(0.0, min(1.0, float(lp_fraction_raw)))
            lp_cap = max(0.0, float(lp_cap_raw))
        except (TypeError, ValueError):
            lp_fraction, lp_cap = 0.20, 8.0
        remaining_before_lp = max(0.0, deadline - time.monotonic())
        lp_budget = min(lp_cap, remaining_before_lp * lp_fraction)
        gpu_stats = {
            "gpu_dual_enabled": False,
            "gpu_dual_status": "disabled",
            "gpu_dual_input_rows": int(survivor_rows.size),
            "gpu_dual_certified_rows": 0,
            "gpu_dual_certified_row_ids": [],
            "gpu_dual_uncertified_rows": int(survivor_rows.size),
            "gpu_dual_coverage_ok": True,
            "gpu_dual_elapsed_s": 0.0,
            "gpu_dual_time_budget_s": 0.0,
            "gpu_dual_steps_requested": int(gpu_dual_steps),
            "gpu_dual_steps_completed": 0,
            "gpu_dual_learning_rate": float(gpu_dual_learning_rate),
            "gpu_dual_row_topk": int(gpu_dual_row_topk),
            "gpu_dual_deadline_reached": False,
            "gpu_dual_deadline_stage": None,
            "gpu_dual_errors": 0,
            "gpu_dual_error_type": None,
            "gpu_dual_error_message": None,
            "gpu_dual_error_stage": None,
            "gpu_dual_certificate_attempted_rows": 0,
            "gpu_dual_certificate_errors": 0,
            "gpu_dual_initial_support_min": None,
            "gpu_dual_initial_support_max": None,
            "gpu_dual_candidate_support_min": None,
            "gpu_dual_candidate_support_max": None,
            "gpu_dual_support_improved_rows": 0,
            "gpu_dual_support_best_improvement": None,
            "gpu_dual_candidate_dual_nnz_total": 0,
            "gpu_dual_candidate_dual_nnz_max": 0,
            "gpu_dual_checked_dual_nnz_total": 0,
            "gpu_dual_checked_dual_nnz_max": 0,
            "gpu_dual_checked_generated_nnz_total": 0,
            "gpu_dual_checked_generated_nnz_max": 0,
            "gpu_dual_checked_source_nnz_total": 0,
            "gpu_dual_checked_source_nnz_max": 0,
            "gpu_dual_checked_bridge_nnz_total": 0,
            "gpu_dual_checked_bridge_nnz_max": 0,
            "gpu_dual_checked_other_nnz_total": 0,
            "gpu_dual_checked_other_nnz_max": 0,
            "gpu_dual_support_attribution_elapsed_s": 0.0,
            "gpu_dual_independent_certificate_elapsed_s": 0.0,
            "gpu_dual_packet_generated_rows_selected": 0,
            "gpu_dual_packet_source_rows_selected": 0,
            "gpu_dual_packet_bridge_rows_selected": 0,
            "gpu_dual_bridge_base_updates": 0,
            "gpu_dual_bridge_packet_updates": 0,
            "gpu_dual_bridge_base_nnz": 0,
            "gpu_dual_bridge_packet_nnz": 0,
            "gpu_dual_bridge_base_support_improvement": 0.0,
            "gpu_dual_bridge_combined_support_improvement": 0.0,
            "gpu_dual_cert_upper_max": None,
            "gpu_dual_cert_min_gap_to_cutoff": None,
            "gpu_dual_cert_center_transform_guard_max": 0.0,
            "gpu_dual_device": "cuda",
            "gpu_dual_device_requested": "cuda",
            "gpu_dual_packet_core_cpu_fallback": False,
            "gpu_dual_proof_authority": False,
            "gpu_dual_binary_factor_count": int(nb),
            "gpu_dual_binary_relaxation_enabled": bool(nb > 0),
            "gpu_dual_candidate_witness_eligible": bool(nb == 0),
            "gpu_dual_objective_scope": "all_surviving_rows",
            "gpu_dual_total_input_rows": int(survivor_rows.size),
            "gpu_dual_objective_rows_scheduled": int(
                survivor_rows.size
            ),
            "gpu_dual_objective_rows_deferred": 0,
            "gpu_dual_first_scheduled_objective_row": (
                int(survivor_rows[0]) if survivor_rows.size else None
            ),
            "gpu_dual_objective_focus_rival_id": None,
            "gpu_dual_objective_focus_plane_kind": None,
            "gpu_dual_objective_focus_mapping_valid": False,
            "gpu_dual_objective_selection_proof_authority": False,
            **_hz_pc_cbde_stats_defaults(),
        }
        if (
            gpu_dual_steps > 0
            and gpu_dual_time_limit > 0.0
            and not prefix_models
        ):
            gpu_budget = min(
                float(gpu_dual_time_limit),
                max(0.0, deadline - time.monotonic()),
            )
            gpu_input_rows = np.asarray(
                survivor_rows, dtype=np.int64
            ).reshape(-1)
            gpu_scheduled_rows = gpu_input_rows
            gpu_deferred_rows = np.zeros(0, dtype=np.int64)
            gpu_focus_rival = None
            gpu_focus_plane_kind = None
            has_micro_rlt_rows = bool(
                nb > 0
                and constraint_row_tags is not None
                and len(constraint_row_tags) == int(A.shape[0])
                and any(
                    str(tag).startswith("property_micro_rlt:")
                    for tag in constraint_row_tags
                )
            )
            if (
                has_micro_rlt_rows
                and safe_groups is not None
                and gpu_input_rows.size > 1
            ):
                (
                    aligned_rows,
                    aligned_deferred,
                    gpu_focus_rival,
                    gpu_focus_plane_kind,
                ) = _hz_property_micro_rlt_focused_objective_schedule(
                    hz,
                    safe_groups=safe_groups,
                    candidate_rows=gpu_input_rows,
                    cube_upper=cube_ub,
                )
                if aligned_rows is None:
                    # Fall back to the ordinary hardest unresolved group's
                    # best plane when no bounded focus receipt is available.
                    gpu_scheduled_rows = gpu_input_rows[:1]
                    gpu_deferred_rows = gpu_input_rows[1:]
                else:
                    gpu_scheduled_rows = aligned_rows
                    gpu_deferred_rows = aligned_deferred
            gpu_survivor_rows, gpu_stats = _hz_gpu_dual_candidate_filter(
                c=c,
                Gc=Gc,
                Gb=Gb,
                C=C,
                t=t,
                candidate_rows=gpu_scheduled_rows,
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                deadline=deadline,
                time_budget=gpu_budget,
                steps=gpu_dual_steps,
                row_topk=gpu_dual_row_topk,
                learning_rate=gpu_dual_learning_rate,
                tol=tol,
                column_layer_ids=column_layer_ids,
                constraint_row_tags=constraint_row_tags,
                packet_core_seed_rows=packet_core_seed_rows,
                packet_core_bridge_rows=packet_core_bridge_rows,
            )
            certified_gpu_rows = {
                int(row)
                for row in gpu_stats.get(
                    "gpu_dual_certified_row_ids", []
                )
            }
            survivor_rows = np.concatenate(
                [
                    np.asarray(
                        gpu_survivor_rows, dtype=np.int64
                    ).reshape(-1),
                    gpu_deferred_rows,
                ]
            )
            overall_covered = np.asarray(
                [
                    *sorted(certified_gpu_rows),
                    *(int(row) for row in survivor_rows),
                ],
                dtype=np.int64,
            )
            gpu_stats.update(
                {
                    "gpu_dual_objective_scope": (
                        (
                            "property_micro_rlt_focused_rival_best_plane"
                            if gpu_focus_rival is not None
                            and gpu_focus_plane_kind
                            != "baseline_property_plane"
                            else
                            "property_micro_rlt_focused_rival_baseline_plane"
                            if gpu_focus_rival is not None
                            else
                            "hardest_unresolved_group_first_plane"
                        )
                        if gpu_deferred_rows.size
                        else "all_surviving_rows"
                    ),
                    "gpu_dual_total_input_rows": int(
                        gpu_input_rows.size
                    ),
                    "gpu_dual_objective_rows_scheduled": int(
                        gpu_scheduled_rows.size
                    ),
                    "gpu_dual_objective_rows_deferred": int(
                        gpu_deferred_rows.size
                    ),
                    "gpu_dual_first_scheduled_objective_row": (
                        int(gpu_scheduled_rows[0])
                        if gpu_scheduled_rows.size
                        else None
                    ),
                    "gpu_dual_objective_focus_rival_id": (
                        None
                        if gpu_focus_rival is None
                        else int(gpu_focus_rival)
                    ),
                    "gpu_dual_objective_focus_plane_kind": (
                        None
                        if gpu_focus_plane_kind is None
                        else str(gpu_focus_plane_kind)
                    ),
                    "gpu_dual_objective_focus_mapping_valid": bool(
                        gpu_focus_rival is not None
                    ),
                    "gpu_dual_objective_selection_proof_authority": (
                        False
                    ),
                    "gpu_dual_uncertified_rows": int(
                        survivor_rows.size
                    ),
                    "gpu_dual_coverage_ok": bool(
                        gpu_stats.get("gpu_dual_coverage_ok", False)
                        and overall_covered.size == gpu_input_rows.size
                        and np.unique(overall_covered).size
                        == gpu_input_rows.size
                        and set(int(row) for row in overall_covered)
                        == set(int(row) for row in gpu_input_rows)
                    ),
                }
            )
        elif (
            prefix_models
            and gpu_dual_steps > 0
            and gpu_dual_time_limit > 0.0
        ):
            gpu_stats["gpu_dual_enabled"] = True
            gpu_stats["gpu_dual_status"] = (
                "redirected_to_row_constraint_prefix"
            )

        obj_stats = getattr(hz, "_solver_objbound_stats", None)
        if isinstance(obj_stats, dict):
            obj_stats.update(gpu_stats)
        else:
            setattr(hz, "_solver_objbound_stats", dict(gpu_stats))
            obj_stats = getattr(hz, "_solver_objbound_stats")
        if not bool(gpu_stats.get("gpu_dual_coverage_ok", False)):
            obj_stats["all_rivals_covered"] = False
            return ("UNKNOWN", None)
        if safe_groups is not None:
            resolved_before_gpu = int(_group_resolved_count())
            gpu_certified_rows = [
                int(row)
                for row in gpu_stats.get(
                    "gpu_dual_certified_row_ids", []
                )
            ]
            _record_group_winners(
                gpu_certified_rows, stage="gpu_lagrangian"
            )
            group_certified_rows.update(
                gpu_certified_rows
            )
            survivor_rows = _schedule_group_rows(
                survivor_rows, cube_ub
            )
            obj_stats.update(
                {
                    "safe_row_gpu_certified_groups": int(
                        _group_resolved_count() - resolved_before_gpu
                    ),
                    "safe_row_groups_resolved_after_gpu": int(
                        _group_resolved_count()
                    ),
                    "safe_row_groups_resolved": int(
                        _group_resolved_count()
                    ),
                    "safe_row_groups_unresolved": int(
                        len(safe_groups) - _group_resolved_count()
                    ),
                    "safe_row_group_winners": (
                        _group_winner_receipt()
                    ),
                }
            )
        if time.monotonic() >= deadline:
            return _parent_unknown("deadline_after_gpu_dual")
        if (
            safe_groups is not None
            and _group_resolved_count() == len(safe_groups)
        ):
            obj_stats.update({
                "lp_status": "skipped_all_property_groups_certified",
                "lp_input_rows": 0,
                "lp_certified_rows": 0,
                "lp_uncertified_rows": 0,
                "lp_survivor_rows": 0,
                "lp_elapsed_s": 0.0,
                "lp_persistent_model_builds": 0,
                "lp_model_reused": False,
                "lp_coverage_ok": True,
                "all_rivals_covered": True,
            })
            return _return_group_safe("gpu_dual_lagrangian")
        if survivor_rows.size == 0:
            if safe_groups is not None:
                return ("UNKNOWN", None)
            obj_stats.update({
                "lp_status": "skipped_all_gpu_certified",
                "lp_input_rows": 0,
                "lp_certified_rows": 0,
                "lp_uncertified_rows": 0,
                "lp_survivor_rows": 0,
                "lp_elapsed_s": 0.0,
                "lp_persistent_model_builds": 0,
                "lp_model_reused": False,
                "lp_coverage_ok": True,
                "all_rivals_covered": bool(
                    int(obj_stats.get("cube_pruned_rows", 0))
                    + int(gpu_stats.get("gpu_dual_certified_rows", 0))
                    == int(C.shape[0])
                ),
            })
            if obj_stats["all_rivals_covered"]:
                return ("SAFE", None)
            return ("UNKNOWN", None)

        # GPU candidate generation has its own explicit budget.  Persistent
        # HiGHS retains its independently configured LP slice; both remain
        # bounded by the same absolute verification deadline.
        highs_lp_budget = float(lp_budget)
        prefix_gpu_budget = min(
            float(gpu_dual_time_limit),
            max(0.0, deadline - time.monotonic()),
        )
        survivor_rows, prefix_gpu_stats = (
            _hz_row_constraint_prefix_gpu_filter(
                c=c,
                Gc=Gc,
                Gb=Gb,
                C=C,
                t=t,
                candidate_rows=survivor_rows,
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                full_eq_rows=int(Ace.shape[0]),
                prefix_models=prefix_models,
                deadline=deadline,
                time_budget=prefix_gpu_budget,
                steps=gpu_dual_steps,
                row_topk=gpu_dual_row_topk,
                learning_rate=gpu_dual_learning_rate,
                tol=tol,
                column_layer_ids=column_layer_ids,
                constraint_row_tags=constraint_row_tags,
            )
        )
        obj_stats = getattr(hz, "_solver_objbound_stats", None)
        if isinstance(obj_stats, dict):
            obj_stats.update(prefix_gpu_stats)
        if not bool(
            prefix_gpu_stats.get(
                "row_prefix_gpu_dual_coverage_ok", False
            )
        ):
            return ("UNKNOWN", None)
        if safe_groups is not None:
            resolved_before_prefix_gpu = int(_group_resolved_count())
            prefix_gpu_certified_rows = [
                int(row)
                for row in prefix_gpu_stats.get(
                    "row_prefix_gpu_dual_certified_row_ids", []
                )
            ]
            _record_group_winners(
                prefix_gpu_certified_rows,
                stage="row_constraint_prefix_gpu_lagrangian",
            )
            group_certified_rows.update(prefix_gpu_certified_rows)
            survivor_rows = _schedule_group_rows(
                survivor_rows, cube_ub
            )
            if isinstance(obj_stats, dict):
                obj_stats.update(
                    {
                        "safe_row_prefix_gpu_certified_groups": int(
                            _group_resolved_count()
                            - resolved_before_prefix_gpu
                        ),
                        "safe_row_groups_resolved_after_prefix_gpu": int(
                            _group_resolved_count()
                        ),
                        "safe_row_groups_resolved": int(
                            _group_resolved_count()
                        ),
                        "safe_row_groups_unresolved": int(
                            len(safe_groups) - _group_resolved_count()
                        ),
                        "safe_row_group_winners": (
                            _group_winner_receipt()
                        ),
                    }
                )
            if _group_resolved_count() == len(safe_groups):
                if isinstance(obj_stats, dict):
                    obj_stats.update(
                        {
                            "row_prefix_lp_enabled": False,
                            "row_prefix_lp_status": (
                                "skipped_all_property_groups_certified_"
                                "by_prefix_gpu"
                            ),
                            "row_prefix_lp_input_rows": 0,
                            "row_prefix_lp_certified_rows": 0,
                            "row_prefix_lp_uncertified_rows": 0,
                            "row_prefix_lp_elapsed_s": 0.0,
                            "row_prefix_lp_coverage_ok": True,
                            "lp_status": (
                                "skipped_all_property_groups_certified_"
                                "by_row_constraint_prefix_gpu"
                            ),
                            "lp_input_rows": 0,
                            "lp_certified_rows": 0,
                            "lp_uncertified_rows": 0,
                            "lp_survivor_rows": 0,
                            "lp_elapsed_s": 0.0,
                            "lp_persistent_model_builds": 0,
                            "lp_model_reused": False,
                            "lp_coverage_ok": True,
                            "all_rivals_covered": True,
                        }
                    )
                return _return_group_safe(
                    "row_prefix_gpu_lagrangian"
                )
        if time.monotonic() >= deadline:
            return _parent_unknown(
                "deadline_after_row_constraint_prefix_gpu"
            )
        prefix_lp_started = time.monotonic()
        survivor_rows, prefix_lp_stats = (
            _hz_row_constraint_prefix_lp_filter(
                c=c,
                Gc=Gc,
                Gb=Gb,
                C=C,
                t=t,
                candidate_rows=survivor_rows,
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                full_eq_rows=int(Ace.shape[0]),
                prefix_models=prefix_models,
                deadline=deadline,
                time_budget=highs_lp_budget,
                tol=tol,
                alternative_row_groups=safe_groups,
            )
        )
        prefix_lp_consumed = max(
            0.0, time.monotonic() - prefix_lp_started
        )
        obj_stats = getattr(hz, "_solver_objbound_stats", None)
        if isinstance(obj_stats, dict):
            obj_stats.update(prefix_lp_stats)
        if not bool(
            prefix_lp_stats.get("row_prefix_lp_coverage_ok", False)
        ):
            return ("UNKNOWN", None)
        if safe_groups is not None:
            resolved_before_prefix = int(_group_resolved_count())
            prefix_certified_rows = [
                int(row)
                for row in prefix_lp_stats.get(
                    "row_prefix_lp_certified_row_ids", []
                )
            ]
            _record_group_winners(
                prefix_certified_rows,
                stage="row_constraint_prefix_lp_lagrangian",
            )
            group_certified_rows.update(prefix_certified_rows)
            survivor_rows = _schedule_group_rows(
                survivor_rows, cube_ub
            )
            if isinstance(obj_stats, dict):
                obj_stats.update(
                    {
                        "safe_row_prefix_lp_certified_groups": int(
                            _group_resolved_count()
                            - resolved_before_prefix
                        ),
                        "safe_row_groups_resolved_after_prefix_lp": int(
                            _group_resolved_count()
                        ),
                        "safe_row_groups_resolved": int(
                            _group_resolved_count()
                        ),
                        "safe_row_groups_unresolved": int(
                            len(safe_groups) - _group_resolved_count()
                        ),
                        "safe_row_group_winners": (
                            _group_winner_receipt()
                        ),
                    }
                )
            if _group_resolved_count() == len(safe_groups):
                if isinstance(obj_stats, dict):
                    obj_stats.update(
                        {
                            "lp_status": (
                                "skipped_all_property_groups_certified_"
                                "by_row_constraint_prefix_lp"
                            ),
                            "lp_input_rows": 0,
                            "lp_certified_rows": 0,
                            "lp_uncertified_rows": 0,
                            "lp_survivor_rows": 0,
                            "lp_elapsed_s": 0.0,
                            "lp_persistent_model_builds": 0,
                            "lp_model_reused": False,
                            "lp_coverage_ok": True,
                            "all_rivals_covered": True,
                        }
                    )
                return _return_group_safe(
                    "row_prefix_lp_lagrangian"
                )
        remaining_highs_lp_budget = max(
            0.0, highs_lp_budget - prefix_lp_consumed
        )
        if time.monotonic() >= deadline:
            return _parent_unknown("deadline_before_persistent_lp")
        persistent_lp_started = time.monotonic()
        obj_stats = getattr(hz, "_solver_objbound_stats", None)
        if isinstance(obj_stats, dict):
            obj_stats.update(
                {
                    "parent_persistent_lp_status": "running",
                    "parent_persistent_lp_input_rows": int(
                        survivor_rows.size
                    ),
                    "parent_persistent_lp_budget_s": float(
                        remaining_highs_lp_budget
                    ),
                }
            )
        _record_parent_stage(
            "persistent_lp",
            now=persistent_lp_started,
        )
        try:
            survivor_rows, lp_stats, lp_witness_xi = (
                _hz_persistent_lp_filter(
                    c=c,
                    Gc=Gc,
                    Gb=Gb,
                    C=C,
                    t=t,
                    candidate_rows=survivor_rows,
                    A=A,
                    rl=rl,
                    ru=ru,
                    lb=lb,
                    ub=ub,
                    deadline=deadline,
                    time_budget=remaining_highs_lp_budget,
                    tol=tol,
                    alternative_row_groups=safe_groups,
                )
            )
        except Exception as exc:
            persistent_lp_finished = time.monotonic()
            obj_stats = getattr(hz, "_solver_objbound_stats", None)
            if isinstance(obj_stats, dict):
                obj_stats.update(
                    {
                        "parent_persistent_lp_status": "error",
                        "parent_persistent_lp_elapsed_s": float(
                            max(
                                0.0,
                                persistent_lp_finished
                                - persistent_lp_started,
                            )
                        ),
                        "parent_persistent_lp_error_type": (
                            type(exc).__name__
                        ),
                    }
                )
            _record_parent_stage(
                "persistent_lp_error",
                exit_reason="persistent_lp_error",
                now=persistent_lp_finished,
            )
            raise
        persistent_lp_finished = time.monotonic()
        obj_stats = getattr(hz, "_solver_objbound_stats", None)
        if isinstance(obj_stats, dict):
            obj_stats.update(lp_stats)
            obj_stats.update(
                {
                    "parent_persistent_lp_status": (
                        "completed_after_deadline"
                        if persistent_lp_finished >= deadline
                        else "completed"
                    ),
                    "parent_persistent_lp_elapsed_s": float(
                        max(
                            0.0,
                            persistent_lp_finished
                            - persistent_lp_started,
                        )
                    ),
                    "parent_persistent_lp_output_rows": int(
                        survivor_rows.size
                    ),
                }
            )
        _record_parent_stage(
            "persistent_lp_complete",
            now=persistent_lp_finished,
        )
        if time.monotonic() >= deadline:
            return _parent_unknown("deadline_after_persistent_lp")
        if isinstance(obj_stats, dict):
            if safe_groups is None:
                obj_stats["all_rivals_covered"] = bool(
                    gpu_stats.get("gpu_dual_coverage_ok", False)
                    and lp_stats.get("lp_coverage_ok", False)
                    and int(obj_stats.get("cube_pruned_rows", 0))
                    + int(gpu_stats.get("gpu_dual_certified_rows", 0))
                    + int(lp_stats.get("lp_certified_rows", 0))
                    + int(survivor_rows.size)
                    == int(C.shape[0])
                )
            else:
                resolved_before_lp = int(_group_resolved_count())
                lp_certified_rows = [
                    int(row)
                    for row in lp_stats.get(
                        "lp_certified_row_ids", []
                    )
                ]
                _record_group_winners(
                    lp_certified_rows, stage="persistent_lp_lagrangian"
                )
                group_certified_rows.update(
                    lp_certified_rows
                )
                survivor_rows = _schedule_group_rows(
                    survivor_rows, cube_ub
                )
                obj_stats.update(
                    {
                        "safe_row_lp_certified_groups": int(
                            _group_resolved_count() - resolved_before_lp
                        ),
                        "safe_row_groups_resolved_after_lp": int(
                            _group_resolved_count()
                        ),
                        "safe_row_groups_resolved": int(
                            _group_resolved_count()
                        ),
                        "safe_row_groups_unresolved": int(
                            len(safe_groups)
                            - _group_resolved_count()
                        ),
                        "safe_row_group_input_coverage_ok": bool(
                            gpu_stats.get(
                                "gpu_dual_coverage_ok", False
                            )
                            and prefix_lp_stats.get(
                                "row_prefix_lp_coverage_ok", False
                            )
                            and prefix_gpu_stats.get(
                                "row_prefix_gpu_dual_coverage_ok",
                                False,
                            )
                            and lp_stats.get("lp_coverage_ok", False)
                        ),
                        "all_rivals_covered": bool(
                            _group_resolved_count()
                            == len(safe_groups)
                        ),
                        "safe_row_group_winners": (
                            _group_winner_receipt()
                        ),
                    }
                )
        else:
            setattr(hz, "_solver_objbound_stats", dict(lp_stats))
        if lp_stats.get("lp_base_feasibility_conflict"):
            return ("UNKNOWN", None)
        if safe_groups is not None:
            if not bool(
                getattr(hz, "_solver_objbound_stats").get(
                    "safe_row_group_input_coverage_ok", False
                )
            ):
                return ("UNKNOWN", None)
            if _group_resolved_count() == len(safe_groups):
                return _return_group_safe(
                    "persistent_lp_lagrangian"
                )
            # Alternative upper planes are a one-sided certificate mechanism.
            # A feasible point of one plane says nothing about the original
            # network property, and all unresolved groups remain UNKNOWN.
            getattr(hz, "_solver_objbound_stats").update(
                {
                    "safe_row_group_exit": (
                        "unknown_unresolved_after_independent_filters"
                    ),
                    "safe_row_group_witness_suppressed": bool(
                        lp_witness_xi is not None
                    ),
                }
            )
            return ("UNKNOWN", None)
        if not getattr(hz, "_solver_objbound_stats").get(
            "all_rivals_covered",
            False,
        ):
            return ("UNKNOWN", None)
        if time.monotonic() >= deadline:
            return ("UNKNOWN", None)
        if survivor_rows.size == 0:
            return ("SAFE", None)
        if lp_witness_xi is not None:
            # This is still only an abstract-HZ candidate.  Production
            # verification must decode its input and pass strict raw ONNX /
            # VNNLIB replay before exposing FALSIFIED.
            return ("UNSAFE", lp_witness_xi)
        if nb == 0 and survivor_rows.size > 1:
            # The persistent model has already solved the same continuous
            # relaxation that the per-rival cutoff path would rebuild.  A
            # cutoff infeasibility status has no independent SAFE authority,
            # so rebuilding the 10M-nnz base once per survivor can add no
            # conclusive outcome here.
            getattr(hz, "_solver_objbound_stats").update({
                "continuous_cutoff_fallback_skipped": True,
                "continuous_cutoff_skip_reason": (
                    "multiple_survivors_and_no_cutoff_safe_authority"
                ),
            })
            return ("UNKNOWN", None)

        def _solve_row(query_index: int):
            r = int(survivor_rows[query_index])
            obj_b = _row_dot_gen(C[r], Gb)
            cost = -np.concatenate([_row_dot_gen(C[r], Gc), 2.0 * obj_b])          # minimize -C[r]y
            const_z = float(C[r] @ c) - float(obj_b.sum())
            # Widen the cutoff by the outward support-accumulation guard used
            # by the cube prefilter.  This covers cancellation error in C@c,
            # C@Gc and C@Gb: every true unsafe point remains feasible in the
            # numerical cutoff model, so roundoff can only lose a proof.
            obj_thr = np.nextafter(
                const_z - float(t[r]) + float(cube_guard[r]),
                np.inf,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return ("unknown", None)
            result = _objbound_solve(
                cost,
                obj_thr,
                A,
                rl,
                ru,
                lb,
                ub,
                integ,
                remaining,
                mip_start_xi=mip_start_xi,
                deadline=deadline,
            )
            if time.monotonic() >= deadline:
                return ("unknown", None)
            return result

        row_workers = max(
            1,
            min(_env_int("HZ_QUERY_WORKERS", 1), int(survivor_rows.size)),
        )
        kind, xi = _hz_run_row_queries(
            int(survivor_rows.size),
            row_workers,
            _solve_row,
            deadline,
        )
        if kind == "witness":
            return ("UNSAFE", xi)
        if kind == "empty":
            return ("SAFE", None)
        return ("UNKNOWN", None)

    # For joint unsafe conjunctions, widen every epigraph row by the same
    # outward accumulation guard used for OR rivals.  Thus a true point with
    # C[r]y <= t[r] cannot be excluded by cancellation in C@c/C@G.
    _, joint_guard = _hz_cube_row_upper_bounds(c, Gc, Gb, C, t)
    nrow = C.shape[0]; nv = ng + nb + 1
    epi = np.zeros((nrow, nv)); epib = np.empty(nrow)
    for r in range(nrow):
        epi[r, :ng] = _row_dot_gen(C[r], Gc); epi[r, ng:ng + nb] = 2.0 * _row_dot_gen(C[r], Gb)
        epi[r, ng + nb] = -1.0
        epib[r] = np.nextafter(
            float(t[r] - C[r] @ c)
            + float(_row_dot_gen(C[r], Gb).sum())
            + float(joint_guard[r]),
            np.inf,
        )
    A2 = (_sp.vstack([
            _sp.hstack([A, _sp.csr_matrix((A.shape[0], 1))], format="csr"),
            _sp.csr_matrix(epi),
          ], format="csr")
          if A.shape[0] else _sp.csr_matrix(epi))
    rl2 = np.concatenate([rl, np.full(nrow, -np.inf)])
    ru2 = np.concatenate([ru, epib])
    lb2 = np.concatenate([lb, [-1e12]]); ub2 = np.concatenate([ub, [1e12]])
    cost = np.zeros(nv); cost[ng + nb] = 1.0   # minimize s
    integ2 = np.concatenate([np.asarray(integ, dtype=int), np.array([0], dtype=int)])
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return ("UNKNOWN", None)
    kind, xi = _objbound_solve(
        cost,
        0.0,
        A2,
        rl2,
        ru2,
        lb2,
        ub2,
        integ2,
        remaining,
        mip_start_xi=mip_start_xi,
        deadline=deadline,
    )
    if time.monotonic() >= deadline:
        return ("UNKNOWN", None)
    if kind == "witness":
        return ("UNSAFE", xi[:ng + nb])
    if kind == "empty":
        return ("SAFE", None)
    return ("UNKNOWN", None)


def hz_certify_spec(hz, C, thresholds, *, is_unsafe_linear: bool,
                    escalate_milp: bool = True,
                    tol: float = 1e-9, time_limit: float = 30.0,
                    require_base_feasible: bool = True):
    """Certify a single (B=1) linear output spec over the constrained HZ.

    Returns (certified: bool, margin: float|None). Sound: True only when the
    HZ over-approximation proves the property. Tries the LP relaxation first;
    if it does not certify and ``escalate_milp`` is set, tries the exact MILP.
    """
    if not _HAS_SCIPY:
        return False, None
    c0, Gc0, Gb0, *_ = hz_np_sparse(hz)
    C, t = _spec_np(C, thresholds, int(c0.size))
    if require_base_feasible:
        base_status, _ = hz_base_feasibility(hz, time_limit=min(float(time_limit), 10.0))
        if base_status != "FEASIBLE":
            return False, None

    ng = Gc0.shape[1]
    nb = Gb0.shape[1]
    if ng + nb == 0:
        row_margins = C @ c0 - t  # C[r].c - t[r] for each row
        if is_unsafe_linear:           # safe iff max_r (C[r]c - t[r]) > 0
            s = float(np.max(row_margins))
            return s > tol, s
        worst = float(np.max(row_margins))  # ALL-rows: safe iff every row < t
        return worst < -tol, -worst

    def _decide(integer):
        if is_unsafe_linear:
            s = hz_joint_min_margin(hz, C, t, integer=integer, time_limit=time_limit)
            return (s is not None and s > tol), s
        worst = -np.inf
        for r in range(C.shape[0]):
            mx = hz_row_max(hz, C[r], integer=integer, time_limit=time_limit)
            if mx is None:
                return False, None
            worst = max(worst, mx - t[r])
        return (worst < -tol), -worst

    ok, margin = _decide(False)
    if ok or not escalate_milp:
        return ok, margin
    return _decide(True)

__all__ = [
    "HZono",
    "SparseHZono",
    "HZSolver",
    "hz_add_const",
    "hz_attach_exact_phase_conditional_property_rows",
    "hz_base_feasibility",
    "hz_base_witness",
    "hz_constructively_nonempty",
    "hz_certify_spec",
    "hz_compute_bounds",
    "hz_compute_lp_bounds",
    "hz_concat",
    "hz_enumerate_sparse_binary_phase_cover",
    "hz_fix_sparse_binary_assignment",
    "hz_verify_sparse_binary_phase_child",
    "hz_fresh_col_ids",
    "hz_reserve_fresh_col_ids_above",
    "hz_from_bounds",
    "hz_inherit_known_nonempty",
    "hz_joint_min_margin",
    "hz_known_nonempty",
    "hz_mark_constructively_nonempty",
    "hz_mark_known_nonempty",
    "hz_minkowski_sum",
    "hz_multiply",
    "hz_negate",
    "hz_np_sparse",
    "hz_objbound_decide",
    "hz_objbound_safe_capability_receipt",
    "hz_relax_np_sparse",
    "hz_remove_redundancy",
    "hz_row_max",
    "hz_sgm_add",
    "hz_split_constraints",
    "hz_sub",
    "sparse_fbbt_tighten_bounds",
    "sparse_highs_relaxation_empty_precheck",
    "sparse_milp_cutoff_highs",
    "sparse_milp_cutoff_scip",
    "sparse_row_bound_infeasible",
    "sparse_solver_start_from_xi",
]
