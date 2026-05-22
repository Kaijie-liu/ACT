from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer, Net
from act.back_end.solver.solver_base import Solver, SolverCaps, SolveStatus

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

# ─── Legacy HyZor pkg path (escape hatch; raises ImportError if absent) ──
# Path setup retained for GlobalTriangleLP / WitnessExtract research drafts
# that may live at HYZOR_ROOT as paper artifacts.
_HYZOR_ROOT = os.environ.get(
    "HYZOR_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  "..", "..", "..", "..", "HyZor")),
)
_HYZOR_PARENT = os.path.dirname(_HYZOR_ROOT)
if _HYZOR_ROOT not in sys.path:
    sys.path.insert(0, _HYZOR_ROOT)
if _HYZOR_PARENT not in sys.path:
    sys.path.insert(0, _HYZOR_PARENT)


# ============================================================================
# 1. HZono dataclass
# ============================================================================


@dataclass
class HZono:
    """Hybrid zonotope with mixed equality / inequality constraints.

    Set definition::

        Z = { c + Gc @ xi_c + Gb @ xi_b
              | (Ac @ xi_c + Ab @ xi_b) [op] b,
                xi_c in [-1, 1]^ng,  xi_b in {-1, +1}^nb }

    where ``[op]`` is row-wise: equality ``=`` for rows where
    ``eq_mask[i] == True`` and inequality ``<=`` for rows where
    ``eq_mask[i] == False``.

    Shapes::

        c   : (n, 1)
        Gc  : (n, ng)         continuous generators
        Gb  : (n, nb)         binary generators
        Ac  : (nc, ng)
        Ab  : (nc, nb)
        b   : (nc, 1)
        eq_mask : (nc,)  torch.bool  --  None ⇒ all rows are equalities
                                          (backward-compat with pre-eq_mask
                                          HZono usage in tf_mlp/cnn/rnn)

    The mixed-constraint form is required by algorithms that introduce
    inequality rows during forward propagation, in particular
    ``project_eq_elim`` (after QR elimination of equality rows, the
    implicit box constraint ``xi_dep in [-1, 1]`` on eliminated
    variables becomes an inequality on the surviving variables) and
    binary-probing residuals. Algorithms that don't need this slot can
    leave ``eq_mask=None`` and the system behaves exactly as the
    pre-extension all-equality HZono.
    """

    c: torch.Tensor  # (n, 1)
    Gc: torch.Tensor  # (n, ng)
    Gb: torch.Tensor  # (n, nb)
    Ac: torch.Tensor  # (nc, ng)
    Ab: torch.Tensor  # (nc, nb)
    b: torch.Tensor  # (nc, 1)
    eq_mask: Optional[torch.Tensor] = None  # (nc,) bool, None = all True
    # Base generator tracking (HyZor parity, HybridZonotope.py:188). The
    # "base" count records how many generators were inherited from an
    # ancestor HZ. project_eq_elim uses this as the retention budget so
    # the input-pixel correlation is preserved through ReLU layers. Each
    # generator-producing op should propagate via _propagate_base().
    _base_ng: Optional[int] = None
    _base_nb: Optional[int] = None

    def __post_init__(self):
        if self._base_ng is None:
            object.__setattr__(self, "_base_ng", int(self.Gc.shape[1]))
        if self._base_nb is None:
            object.__setattr__(self, "_base_nb", int(self.Gb.shape[1]))

    # ─── Convenience shape accessors (parity with HyZor's HybridZonotope) ──
    # HyZor exposes ``.dim, .ng, .nb, .nc`` as cached ints; the cons-walker
    # in ``HZVerifier`` reads them for per-layer logging and dispatch
    # heuristics. Provide them as zero-cost @property's so the same code
    # works on either HZono or HybridZonotope.
    @property
    def dim(self) -> int:
        return int(self.c.shape[0])

    @property
    def ng(self) -> int:
        return int(self.Gc.shape[1])

    @property
    def nb(self) -> int:
        return int(self.Gb.shape[1])

    @property
    def nc(self) -> int:
        return int(self.b.shape[0])


def _propagate_base(parent: HZono, child: HZono) -> HZono:
    """Mirror HyZor ``HybridZonotope._with_base``: child base = min(parent, child.ng).

    Returns child (mutated). Use after every HZ-producing op so the
    base-generator budget tracks through the cascade.
    """
    parent_base_ng = getattr(parent, "_base_ng", None)
    parent_base_nb = getattr(parent, "_base_nb", None)
    if parent_base_ng is not None:
        object.__setattr__(child, "_base_ng",
                           int(min(int(parent_base_ng), int(child.Gc.shape[1]))))
    if parent_base_nb is not None:
        object.__setattr__(child, "_base_nb",
                           int(min(int(parent_base_nb), int(child.Gb.shape[1]))))
    return child


def _hz_reduce_constraints(
    hz: HZono,
    *,
    nc_budget: Optional[int] = None,
    ng_budget: Optional[int] = None,
    nb_budget: Optional[int] = None,
    tol: float = 1e-12,
) -> HZono:
    """Reduce HZono constraint / generator count. Faithful port of HyZor
    ``HybridZonotope.reduce_constraints`` (HybridZonotope.py:4399). Sound
    over-approximation.

    Phases (subset; the full HyZor port has 5 phases):
      P1: remove trivially redundant constraint rows  (exact)
      P2: remove zero generator columns               (exact)
      P5: continuous generator order reduction (Girard) — caps ng to
          ng_budget, merging dropped cols into a single box column with
          eq-row split + widening.

    Phases skipped (parity-deferred): 1.3 (QR rank), 1.5 (parallel rows),
    2.5 (parallel gens), 3 (nc-budget topk), 4 (nb relaxation). The
    skipped phases are deduplications — sound to skip, may produce
    slightly larger but still valid HZ.
    """
    import torch as _t
    has_budget = (
        nc_budget is not None or ng_budget is not None or nb_budget is not None
    )
    nc = int(hz.b.shape[0])
    if nc == 0 and not has_budget:
        return hz

    device = hz.c.device
    dtype = hz.c.dtype
    Ac, Ab, b = hz.Ac, hz.Ab, hz.b
    Gc, Gb, c = hz.Gc, hz.Gb, hz.c
    eq_m = (
        hz.eq_mask.clone()
        if (hz.eq_mask is not None and int(hz.eq_mask.numel()) == nc)
        else _t.zeros(nc, dtype=_t.bool, device=device)
    )
    n = int(Gc.shape[0])

    # ---- Phase 1: trivially redundant constraints ----
    if Ac.shape[0] > 0:
        lhs_max = Ac.abs().sum(dim=1) + Ab.abs().sum(dim=1)
        slack = b.view(-1) - lhs_max
        # Equalities are never trivially redundant in ≤ sense, so keep them.
        keep_mask = (slack < -tol) | eq_m[: Ac.shape[0]]
        if int((~keep_mask).sum().item()) > 0:
            Ac = Ac[keep_mask]
            Ab = Ab[keep_mask]
            b = b[keep_mask]
            eq_m = eq_m[keep_mask]

    # ---- Phase 1.3: QR rank-revealing row removal ----
    # Linearly dependent constraint rows are sound to drop (LP feasibility
    # over a subset of rank-many independent rows = full system). HyZor:4449.
    # Size cap: QR on (ng+nb, nc) is O((ng+nb)·nc²); skip when product
    # exceeds threshold so reduce overhead stays bounded.
    nc_now = int(Ac.shape[0])
    _PHASE_1_3_CAP = 1024  # max nc for QR phase
    if 1 < nc_now <= _PHASE_1_3_CAP:
        try:
            M = _t.cat([Ac, Ab], dim=1).to(dtype=_t.float64)
            Q, R = _t.linalg.qr(M.T)  # QR on transpose for row-rank.
            diag = R.diag().abs()
            if diag.numel() > 0:
                rank_tol = max(tol, 1e-10) * max(1.0, float(diag[0]))
            else:
                rank_tol = tol
            rank = int((diag > rank_tol).sum().item())
            if rank < nc_now:
                import scipy.linalg as _sla
                _Q, _R, _piv = _sla.qr(
                    M.T.cpu().numpy(), pivoting=True, mode="economic"
                )
                keep_idx = sorted(_piv[:rank].tolist())
                keep_t = _t.tensor(keep_idx, device=device, dtype=_t.long)
                Ac = Ac[keep_t]
                Ab = Ab[keep_t]
                b = b[keep_t]
                eq_m = eq_m[keep_t]
        except Exception:
            pass  # QR failed; sound fallback (keep all rows).

    # ---- Phase 1.5: parallel / duplicate constraint rows ----
    # Rows with cosine similarity ≈ 1 represent the same constraint
    # direction; the tightest (smallest normalized RHS) dominates. HyZor:4484.
    # Size cap: O(nc²) cosine similarity + O(nc) python loop is expensive
    # for huge nc (intermediate layers). Gate at 2048 — for nc smaller
    # than this the savings dominate; for larger let Phase 5 Girard handle
    # downstream redundancy.
    nc_now = int(Ac.shape[0])
    _PHASE_1_5_CAP = 2048
    if 1 < nc_now <= _PHASE_1_5_CAP:
        rows = _t.cat([Ac, Ab], dim=1)
        norms = rows.norm(dim=1, keepdim=True).clamp(min=tol)
        dirs = rows / norms
        sim = dirs @ dirs.T
        parallel = sim > 1.0 - 1e-6
        parallel.fill_diagonal_(False)
        # Early-exit gate: skip Python loop if no parallel rows.
        if bool(parallel.any().item()):
            b_norm = b.view(-1) / norms.view(-1)
            keep = _t.ones(nc_now, dtype=_t.bool, device=device)
            for i in range(nc_now):
                if not keep[i]:
                    continue
                group = _t.where(parallel[i] & keep)[0]
                if group.numel() == 0:
                    continue
                group = _t.cat([_t.tensor([i], device=device), group])
                best = group[_t.argmin(b_norm[group])]
                for j in group:
                    if j.item() != best.item():
                        keep[j] = False
            if int((~keep).sum().item()) > 0:
                Ac, Ab, b = Ac[keep], Ab[keep], b[keep]
                eq_m = eq_m[keep]

    # ---- Phase 2: zero generator columns ----
    # When nc=0 we must keep Ac as (0, new_ng) (not (0, 0)) so downstream
    # ops that cat onto Ac (e.g. hz_apply_relu eq encoding) preserve the
    # invariant Ac.shape[1] == Gc.shape[1].
    nc_now = int(Ac.shape[0])
    if Gc.numel() > 0 and Gc.shape[1] > 0:
        gc_used = Gc.abs().sum(dim=0) > tol
        if nc_now > 0:
            gc_used = gc_used | (Ac.abs().sum(dim=0) > tol)
        if int((~gc_used).sum().item()) > 0:
            Gc = Gc[:, gc_used]
            if nc_now > 0:
                Ac = Ac[:, gc_used]
            else:
                Ac = _t.empty(0, int(Gc.shape[1]), device=device, dtype=dtype)
    if Gb.numel() > 0 and Gb.shape[1] > 0:
        gb_used = Gb.abs().sum(dim=0) > tol
        if nc_now > 0:
            gb_used = gb_used | (Ab.abs().sum(dim=0) > tol)
        if int((~gb_used).sum().item()) > 0:
            Gb = Gb[:, gb_used]
            if nc_now > 0:
                Ab = Ab[:, gb_used]
            else:
                Ab = _t.empty(0, int(Gb.shape[1]), device=device, dtype=dtype)

    # ---- Phase 2.5: parallel generator merging ----
    # Continuous generator columns with cosine sim ≈ 1 represent the same
    # direction in the joint (Gc; Ac) space; merge by summing norms. HyZor:4548.
    # Size cap: O(ng²) cosine + O(ng) python loop is expensive for ng=6000+.
    # Gate at 2048 — Phase 5 Girard handles the budget cap for larger ng.
    nc_now = int(Ac.shape[0])
    ng_cur = int(Gc.shape[1])
    _PHASE_2_5_CAP = 2048
    if 1 < ng_cur <= _PHASE_2_5_CAP:
        Lc = _t.cat([Gc, Ac], dim=0) if nc_now > 0 else Gc
        lc_norms = Lc.norm(dim=0).clamp(min=tol)
        lc_dirs = Lc / lc_norms.unsqueeze(0)
        sim_c = lc_dirs.T @ lc_dirs
        par_c = sim_c > 1.0 - 1e-6
        par_c.fill_diagonal_(False)
        # Early-exit gate.
        if bool(par_c.any().item()):
            visited_c = _t.zeros(ng_cur, dtype=_t.bool, device=device)
            merged_Gc_cols = []
            merged_Ac_cols = []
            for i in range(ng_cur):
                if visited_c[i]:
                    continue
                grp = _t.where(par_c[i] & ~visited_c)[0]
                grp = _t.cat([_t.tensor([i], device=device), grp])
                visited_c[grp] = True
                if grp.numel() == 1:
                    merged_Gc_cols.append(Gc[:, i])
                    if nc_now > 0:
                        merged_Ac_cols.append(Ac[:, i])
                else:
                    total_norm = lc_norms[grp].sum()
                    merged_Gc_cols.append(lc_dirs[:n, grp[0]] * total_norm)
                    if nc_now > 0:
                        merged_Ac_cols.append(lc_dirs[n:, grp[0]] * total_norm)
            if len(merged_Gc_cols) < ng_cur:
                Gc = _t.stack(merged_Gc_cols, dim=1) if merged_Gc_cols else _t.empty(n, 0, device=device, dtype=dtype)
                if nc_now > 0:
                    Ac = _t.stack(merged_Ac_cols, dim=1) if merged_Ac_cols else _t.empty(nc_now, 0, device=device, dtype=dtype)
                else:
                    Ac = _t.empty(0, int(Gc.shape[1]), device=device, dtype=dtype)

    # ---- Phase 5: continuous generator order reduction (Girard) ----
    ng_now = int(Gc.shape[1])
    if ng_budget is not None and ng_now > ng_budget and ng_budget >= 1:
        nc_now = int(Ac.shape[0])
        scores = Gc.abs().sum(dim=0)
        keep_k = ng_budget - 1  # reserve 1 slot for box column
        if keep_k < 0:
            keep_k = 0
        _, topk_idx = _t.topk(scores, min(keep_k, ng_now))
        topk_idx, _ = topk_idx.sort()
        remove_mask = _t.ones(ng_now, dtype=_t.bool, device=device)
        remove_mask[topk_idx] = False
        removed_Gc = Gc[:, remove_mask]
        box_col = removed_Gc.abs().sum(dim=1, keepdim=True)  # (n, 1)
        Gc = _t.cat([Gc[:, topk_idx], box_col], dim=1)

        if nc_now > 0:
            old_Ac_keep = Ac[:, topk_idx]
            removed_Ac = Ac[:, remove_mask]
            widen = removed_Ac.abs().sum(dim=1, keepdim=True)
            Ac_box = _t.zeros(nc_now, 1, device=device, dtype=dtype)
            em_now = (
                eq_m if int(eq_m.numel()) == nc_now
                else _t.zeros(nc_now, dtype=_t.bool, device=device)
            )
            ineq_mask = ~em_now
            # Inequality rows: widen b by |removed_Ac|·1.
            Ac_in = _t.cat([old_Ac_keep[ineq_mask], Ac_box[ineq_mask]], dim=1)
            Ab_in = Ab[ineq_mask]
            b_in = b[ineq_mask] + widen[ineq_mask]
            if bool(em_now.any()):
                # Equality rows: split into 2 inequalities, both widened.
                eAc = old_Ac_keep[em_now]
                eAb = Ab[em_now]
                eb = b[em_now]
                ewiden = widen[em_now]
                n_eq = int(em_now.sum().item())
                up_Ac = _t.cat([eAc, _t.zeros(n_eq, 1, device=device, dtype=dtype)], dim=1)
                up_b = eb + ewiden
                lo_Ac = _t.cat([-eAc, _t.zeros(n_eq, 1, device=device, dtype=dtype)], dim=1)
                lo_b = -eb + ewiden
                Ac = _t.cat([Ac_in, up_Ac, lo_Ac], dim=0)
                Ab = _t.cat([Ab_in, eAb, -eAb], dim=0)
                b = _t.cat([b_in, up_b, lo_b], dim=0)
                eq_m = _t.cat([
                    _t.zeros(int(ineq_mask.sum().item()), dtype=_t.bool, device=device),
                    _t.zeros(2 * n_eq, dtype=_t.bool, device=device),
                ], dim=0)
            else:
                Ac = Ac_in
                Ab = Ab_in
                b = b_in
                eq_m = _t.zeros(int(ineq_mask.sum().item()), dtype=_t.bool, device=device)
        else:
            Ac = _t.empty(0, Gc.shape[1], device=device, dtype=dtype)

    # Avoid construction if nothing changed.
    if (
        Gc.data_ptr() == hz.Gc.data_ptr()
        and Gb.data_ptr() == hz.Gb.data_ptr()
        and Ac.data_ptr() == hz.Ac.data_ptr()
    ):
        return hz

    out = HZono(
        c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b,
        eq_mask=(eq_m if int(eq_m.numel()) == int(Ac.shape[0]) else None),
    )
    _propagate_base(hz, out)
    return out


# Bind reduce_constraints as method on HZono so HZVerifier._maybe_reduce
# (which calls hz.reduce_constraints(ng_budget=cap)) works on ACT HZono.
def _hz_reduce_constraints_method(self, *, nc_budget=None, ng_budget=None,
                                    nb_budget=None, tol=1e-12, verbose=False):
    return _hz_reduce_constraints(
        self, nc_budget=nc_budget, ng_budget=ng_budget,
        nb_budget=nb_budget, tol=tol,
    )


HZono.reduce_constraints = _hz_reduce_constraints_method


# ----------------------------------------------------------------------------
# eq_mask helpers
# ----------------------------------------------------------------------------


def _eq_mask_of(hz: HZono) -> torch.Tensor:
    """Return hz.eq_mask if set, else a length-nc all-True mask."""
    nc = hz.b.shape[0]
    if hz.eq_mask is not None:
        if int(hz.eq_mask.numel()) != int(nc):
            raise ValueError(
                f"HZono.eq_mask length {int(hz.eq_mask.numel())} != "
                f"nc {int(nc)}"
            )
        return hz.eq_mask
    return torch.ones(nc, dtype=torch.bool, device=hz.b.device)


def _split_eq_le(hz: HZono):
    """Return ``(Ac_eq, Ab_eq, b_eq, Ac_le, Ab_le, b_le)`` slices of hz
    according to its eq_mask. b's leading dim is flattened to (n,)."""
    em = _eq_mask_of(hz)
    le = ~em
    b_flat = hz.b.reshape(-1)
    return (
        hz.Ac[em], hz.Ab[em], b_flat[em],
        hz.Ac[le], hz.Ab[le], b_flat[le],
    )


def _concat_eq_masks(em1, em2, nc1: int, nc2: int, device) -> torch.Tensor:
    """Concat eq_masks from two HZonos, materialising defaults (all True)
    on either side when the input mask is None."""
    if em1 is None:
        em1 = torch.ones(nc1, dtype=torch.bool, device=device)
    if em2 is None:
        em2 = torch.ones(nc2, dtype=torch.bool, device=device)
    return torch.cat(
        [em1.to(device=device), em2.to(device=device)], dim=0
    )


# ============================================================================
# 2. Algebraic operations
# ============================================================================


def hz_multiply(hz: HZono, R: torch.Tensor) -> HZono:
    """Linear map ``y = R x``. Constraint rows untouched, so eq_mask passes
    through unchanged."""
    R = R.to(dtype=hz.c.dtype, device=hz.c.device)
    return HZono(
        c=R @ hz.c,
        Gc=R @ hz.Gc,
        Gb=R @ hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
    )


def hz_add_const(hz: HZono, v: torch.Tensor) -> HZono:
    """Translate by constant ``v``. Constraint rows untouched."""
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
    )


def hz_minkowski_sum(hz1: HZono, hz2: HZono) -> HZono:
    """Block-diagonal stacking of the two constraint systems. Each branch
    keeps its own row semantics; eq_mask is concatenated row-wise."""
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

    # Preserve eq_mask through stacking. If both inputs left it as None we
    # leave it None to keep the legacy all-True default; otherwise we
    # materialise both masks first so the result has a well-defined per-row
    # eq/le label.
    new_eq_mask = None
    if hz1.eq_mask is not None or hz2.eq_mask is not None:
        new_eq_mask = _concat_eq_masks(
            hz1.eq_mask, hz2.eq_mask, nc1, nc2, device
        )

    return HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=new_Ac, Ab=new_Ab, b=new_b,
        eq_mask=new_eq_mask,
    )


def hz_from_bounds(bounds: Bounds, dtype, device) -> HZono:
    """Box-from-bounds factory. No constraints, eq_mask trivially None."""
    lb = bounds.lb.flatten().to(dtype=dtype, device=device)
    ub = bounds.ub.flatten().to(dtype=dtype, device=device)
    n = lb.shape[0]
    c = ((lb + ub) / 2.0).view(-1, 1)
    rad = (ub - lb) / 2.0
    return HZono(
        c=c,
        Gc=torch.diag(rad),
        Gb=torch.zeros((n, 0), dtype=dtype, device=device),
        Ac=torch.zeros((0, n), dtype=dtype, device=device),
        Ab=torch.zeros((0, 0), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
        eq_mask=None,
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
    """Per-dimension MILP-tight bounds using scipy.optimize.milp.

    HZ semantics: y = c + Gc ξ_c + Gb ξ_b with ξ_c ∈ [-1,1]^p, ξ_b ∈ {-1,+1}^q,
    Ac ξ_c + Ab ξ_b = b. We must enforce ξ_b integrality; relaxing it to
    continuous [-1,1] gives narrower-than-MILP bounds and is therefore
    UNSOUND for verification (the LP-relaxed feasible polytope is the convex
    hull of {-1,+1}^q which the true HZ does not visit interiorly).

    Implementation: scipy.optimize.milp with integrality on the last q vars,
    and bounds [-1, 1] interpreted as the integer set {-1, +1} for those vars.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds as SciBounds
    n = int(hz.c.shape[0])
    p = int(hz.Gc.shape[1])
    q = int(hz.Gb.shape[1])
    c_np = hz.c.detach().cpu().numpy().astype("float64").reshape(-1)
    Gc_np = hz.Gc.detach().cpu().numpy().astype("float64")
    Gb_np = hz.Gb.detach().cpu().numpy().astype("float64")
    Ac_np = hz.Ac.detach().cpu().numpy().astype("float64")
    Ab_np = hz.Ab.detach().cpu().numpy().astype("float64")
    b_np = hz.b.detach().cpu().numpy().astype("float64").reshape(-1)

    # Variables: [ξ_c (p continuous in [-1,1]); ξ_b (q integer with lb=-1, ub=+1)]
    # integrality flag: 0 for continuous, 1 for integer
    integrality = np.concatenate([np.zeros(p, dtype=int), np.ones(q, dtype=int)])
    var_lb = np.concatenate([-np.ones(p), -np.ones(q)])
    var_ub = np.concatenate([np.ones(p),   np.ones(q)])
    var_bounds = SciBounds(lb=var_lb, ub=var_ub)
    # Split eq vs le rows when eq_mask is set. Legacy callers leave
    # eq_mask=None, in which case all rows are equalities (treated as
    # bidirectional LP rhs).
    nc = int(b_np.size)
    constraints = []
    if nc > 0:
        if hz.eq_mask is None:
            A_full = np.concatenate([Ac_np, Ab_np], axis=1)
            constraints.append(LinearConstraint(A=A_full, lb=b_np, ub=b_np))
        else:
            em_np = hz.eq_mask.detach().cpu().numpy().astype(bool)
            le_np = ~em_np
            if em_np.any():
                A_eq = np.concatenate(
                    [Ac_np[em_np], Ab_np[em_np]], axis=1
                )
                b_eq = b_np[em_np]
                constraints.append(
                    LinearConstraint(A=A_eq, lb=b_eq, ub=b_eq)
                )
            if le_np.any():
                A_le = np.concatenate(
                    [Ac_np[le_np], Ab_np[le_np]], axis=1
                )
                b_le = b_np[le_np]
                constraints.append(
                    LinearConstraint(A=A_le, lb=-np.inf, ub=b_le)
                )

    LB = np.empty((n,), dtype=np.float64)
    UB = np.empty((n,), dtype=np.float64)
    for i in range(n):
        obj = np.concatenate([Gc_np[i], Gb_np[i]], axis=0)
        res_min = milp(c=obj, constraints=constraints, integrality=integrality,
                        bounds=var_bounds)
        if not res_min.success:
            raise RuntimeError(
                f"[milp] MIN infeasible at dim {i}: {res_min.message}"
            )
        LB[i] = c_np[i] + res_min.fun
        res_max = milp(c=-obj, constraints=constraints, integrality=integrality,
                        bounds=var_bounds)
        if not res_max.success:
            raise RuntimeError(
                f"[milp] MAX infeasible at dim {i}: {res_max.message}"
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
    if _HAS_GUROBI:
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


# ─── HZVerifier: main HZ-based network verifier ─────────────────────────────


class HZVerifier(Solver):
    """Tier-2 verification solver. Walks ACT cons IR and dispatches to
    HZ ops (eq_lagr_v8 family) via hz_ops/hz_routing, with ACT sigmoid/tanh
    K-piece for smooth activations."""

    # v9.2: benchmark-tuned profile presets (ports HyZor's 4 canonical overrides)
    # Each profile is an overlay dict applied on top of constructor defaults.
    BENCHMARK_PROFILES = {
        # safenlp_2024: B14 profile. Heavy proof techniques + eq_lagr_v8 last 3.
        # NOTE: dual_cert_n=1 (LP single cert is enough; UNC too loose).
        # The B14 power comes from extra_env enabling HyZor-internal LP techniques.
        "safenlp_b14": {
            "relu_method": "eq_lagr_v8",
            "girard_cap": 6000,
            "mace_enabled": True,
            "sgm_enabled": True,
            "dual_cert_n": 1,
            "large_cls_proof_mode": "off",  # safenlp is small-out, large_cls not relevant
            "extra_env": {
                "HYZOR_L2_EXACTLP_TAIL": "1",
                "HYZOR_L2_EXACTLP_RELU_CAP": "128",
                "HYZOR_L2_EXACTLP_OUT_CAP": "1024",
                "HYZOR_MARGIN_LAGR": "1",
                "HYZOR_MARGIN_LAGR_ITERS": "260",
            },
        },
        # nn4sys: A0_triangle profile. Force triangle (no eq_lagr_v8, faster).
        "nn4sys_triangle": {
            "relu_method": "triangle",
            "girard_cap": 6000,
            "mace_enabled": False,
            "sgm_enabled": False,
            "dual_cert_n": 1,
            "large_cls_proof_mode": "off",
            "extra_env": {},
        },
        # metaroom_2023: PDNT N=1 bypass override.
        "metaroom_n1": {
            "relu_method": "eq_lagr_v8",
            "girard_cap": 6000,
            "mace_enabled": True,
            "sgm_enabled": True,
            "dual_cert_n": 1,                # ★ key knob for metaroom
            "large_cls_proof_mode": "auto",
            "extra_env": {},
        },
        # dist_shift_2023: B9 profile (lighter than B14).
        "dist_shift_b9": {
            "relu_method": "eq_lagr_v8",
            "girard_cap": 6000,
            "mace_enabled": True,
            "sgm_enabled": True,
            "dual_cert_n": 1,
            "large_cls_proof_mode": "off",
            "extra_env": {
                "HYZOR_L2_EXACTLP_TAIL": "1",
                "HYZOR_L2_EXACTLP_RELU_CAP": "96",
                "HYZOR_L2_EXACTLP_OUT_CAP": "512",
                "HYZOR_MARGIN_LAGR": "1",
                "HYZOR_MARGIN_LAGR_ITERS": "220",
            },
        },
        "default": {
            "relu_method": "eq_lagr_v8",
            "girard_cap": 6000,
            "mace_enabled": True,
            "sgm_enabled": True,
            "dual_cert_n": 1,
            "large_cls_proof_mode": "auto",
            "extra_env": {},
        },
    }

    def __init__(
        self, *,
        # Profile shortcut (v9.2): applies a benchmark-tuned overlay
        benchmark_profile: Optional[str] = None,
        # Individual knobs (override profile if specified)
        relu_method: str = "eq_lagr_v8",
        girard_cap: int = 6000,
        mace_enabled: bool = True,
        sgm_enabled: bool = True,
        strict_replay: bool = True,
        sigmoid_K: int = 2,
        tanh_K: int = 2,
        # large_cls_proof_mode (HyZor scheduling for cifar100/tinyimagenet)
        large_cls_proof_mode: str = "auto",   # "on" | "off" | "auto"
        large_cls_eq_layers: int = 3,         # last N relus use eq_lagr_v8
        large_cls_conv_threshold: int = 4,    # min conv count to trigger
        large_cls_out_dim_threshold: int = 100,  # min output dim
        # PDNT multi-cert (v9.2): N independent sound certs required.
        # N=1 (default) = trust LP single cert (sound; matches metaroom override).
        # N≥2 = belt-and-suspenders mode: require additional cert(s) to confirm.
        # NOTE: only UNC cert ("U") implemented in v9.2. Full E/Z/F pool deferred.
        # Since UNC is much looser than LP, requiring N≥2 UNC-confirmation will
        # lose precision (LP-verified + UNC-can't-confirm → UNKNOWN). Use N=1
        # unless you specifically need the extra paranoia.
        dual_cert_n: int = 1,
        dual_cert_pool: str = "U",
        dual_cert_margin: float = 1e-8,
        timeout_s: float = 300.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        onnx_path: Optional[str] = None,
        vnnlib_path: Optional[str] = None,
        # Tier 1A: LP-aggressive tail pass. After cons-walker returns UNKNOWN,
        # retry with `exact_lp` ReLU encoding (more LP cons per ReLU = tighter).
        # Gating mirrors HyZor's HYZOR_L2_EXACTLP_TAIL: small/medium nets only.
        l2_exactlp_tail: bool = False,        # off by default (opt-in)
        l2_exactlp_relu_cap: int = 128,       # only retry if ≤ this many ReLUs
        l2_exactlp_out_cap: int = 512,        # only retry if output dim ≤ this
    ):
        # Apply benchmark profile if specified
        if benchmark_profile is not None:
            if benchmark_profile not in self.BENCHMARK_PROFILES:
                raise ValueError(
                    f"Unknown benchmark_profile {benchmark_profile!r}. "
                    f"Valid: {list(self.BENCHMARK_PROFILES.keys())}"
                )
            profile = self.BENCHMARK_PROFILES[benchmark_profile]
            relu_method = profile.get("relu_method", relu_method)
            girard_cap = profile.get("girard_cap", girard_cap)
            # Tier 1A: profiles that set HYZOR_L2_EXACTLP_TAIL in extra_env
            # also enable the cons-walker's LP-aggressive retry path. Caps
            # come from profile env (RELU_CAP / OUT_CAP) when set.
            _eenv = profile.get("extra_env", {})
            if _eenv.get("HYZOR_L2_EXACTLP_TAIL") == "1":
                l2_exactlp_tail = True
                l2_exactlp_relu_cap = int(float(_eenv.get("HYZOR_L2_EXACTLP_RELU_CAP", l2_exactlp_relu_cap)))
                l2_exactlp_out_cap = int(float(_eenv.get("HYZOR_L2_EXACTLP_OUT_CAP", l2_exactlp_out_cap)))
            mace_enabled = profile.get("mace_enabled", mace_enabled)
            sgm_enabled = profile.get("sgm_enabled", sgm_enabled)
            large_cls_proof_mode = profile.get("large_cls_proof_mode",
                                                large_cls_proof_mode)
            dual_cert_n = profile.get("dual_cert_n", dual_cert_n)
            # Set HyZor-internal env vars that affect proof aggressiveness
            for k, v in profile.get("extra_env", {}).items():
                os.environ.setdefault(k, v)

        self.cfg = dict(
            benchmark_profile=benchmark_profile,
            relu_method=relu_method, girard_cap=girard_cap,
            mace_enabled=mace_enabled, sgm_enabled=sgm_enabled,
            strict_replay=strict_replay,
            sigmoid_K=sigmoid_K, tanh_K=tanh_K,
            large_cls_proof_mode=large_cls_proof_mode,
            large_cls_eq_layers=large_cls_eq_layers,
            large_cls_conv_threshold=large_cls_conv_threshold,
            large_cls_out_dim_threshold=large_cls_out_dim_threshold,
            dual_cert_n=dual_cert_n,
            dual_cert_pool=dual_cert_pool,
            dual_cert_margin=dual_cert_margin,
            timeout_s=timeout_s, device=device, dtype=dtype,
            onnx_path=onnx_path,
            vnnlib_path=vnnlib_path,
            l2_exactlp_tail=l2_exactlp_tail,
            l2_exactlp_relu_cap=l2_exactlp_relu_cap,
            l2_exactlp_out_cap=l2_exactlp_out_cap,
        )
        self._reset_state()

    def _reset_state(self):
        self._status: str = SolveStatus.UNKNOWN
        self._witness: Optional[np.ndarray] = None
        self._has_solution: bool = False
        self._var_count: int = 0
        self._stats: Dict[str, Any] = {}

    # ----- Solver interface stubs (no-op; HyZor uses consume_cons) -----
    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=True, supports_hz=True)

    # ─── LEGACY_SHIM_TO_REMOVE_AT_P3 ───────────────────────────────────
    # HyZor's verification path walks the ACT cons IR via consume_cons()
    # (a custom analyze-walking pipeline) rather than consuming a
    # BatchLPProblem. Until eq_lagr_v8 / project_eq_elim / Phase-1-3
    # representations land in act/back_end/hybridz_tf/ and the cascade
    # controller moves to hybridz_tf/algorithms/, the new
    # setup_and_solve_batch + verify_once entry points cannot drive
    # HyZor end-to-end. solve_batch therefore mirrors HZSolver's design
    # (raise with redirect message); callers must use
    # ``verify_once_legacy_batch1`` (defined below) for HyZor-mode
    # verification of a single (model, vnnlib) instance.
    def solve_batch(self, problem, timelimit: Optional[float] = None):  # noqa: D401
        """HZVerifier does not accept BatchLPProblem inputs.

        HyZor walks the ACT cons IR directly via ``consume_cons``; it
        does not consume a pre-built LP. Callers verifying a single
        instance through HyZor should use
        ``verify_once_legacy_batch1(net, solver=..., timelimit=...)``
        from this module. Batch-native HyZor integration via
        ``hybridz_tf`` is a follow-up (see comment block above).
        """
        raise NotImplementedError(
            "HZVerifier does not consume BatchLPProblem; use "
            "act.back_end.solver.solver_hz.verify_once_legacy_batch1"
            "(net, solver=..., timelimit=...) for single-instance HyZor "
            "verification until hybridz_tf integration lands."
        )

    def begin(self, name: str = "verify", device: Optional[str] = None):
        self._reset_state()
        if device is not None:
            self.cfg["device"] = device
        # Aggressive GPU memory cleanup to avoid fragmentation across
        # sequential verify_once calls (esp. on large nets like cifar100/
        # tinyimagenet — fragmented allocator causes spurious OOM).
        try:
            import torch as _t
            if _t.cuda.is_available() and self.cfg.get("device", "cpu").startswith("cuda"):
                _t.cuda.empty_cache()
                _t.cuda.synchronize()
        except Exception:
            pass

    @property
    def n(self) -> int:
        return self._var_count

    def add_vars(self, n: int) -> None:
        self._var_count += n

    def add_binary_vars(self, n: int) -> List[int]:
        ids = list(range(self._var_count, self._var_count + n))
        self._var_count += n
        return ids

    def set_bounds(self, idxs, lb, ub): pass
    def add_lin_eq(self, vids, coeffs, rhs): pass
    def add_lin_le(self, vids, coeffs, rhs): pass
    def add_lin_ge(self, vids, coeffs, rhs): pass
    def add_sum_eq(self, vids, rhs): pass
    def add_ge_zero(self, vids): pass
    def add_sos2(self, var_ids, weights=None): pass
    def set_objective_linear(self, vids, coeffs, const=0.0, sense="min"): pass
    def optimize(self, timelimit: Optional[float] = None) -> None: pass

    # ----- Real entry: cons walker -----
    def consume_cons(
        self, globalC: ConSet, before: Dict[int, Fact], after: Dict[int, Fact],
        *, net: Net, input_ids: List[int], output_ids: List[int],
        assert_layer: Layer,
    ) -> str:
        # ACT-native HZ ops by default. HYZOR_USE_ACT=0 emergency escape
        # hatch tries to import the legacy HyZor pkg and raises if absent.
        _use_act = os.environ.get("HYZOR_USE_ACT", "1") == "1"
        if not _use_act:
            try:
                from HyZor import (
                    hz_from_bounds, hz_dense, hz_conv2d, hz_add_const, hz_scale,
                    hz_bn, hz_minkowski_sum, hz_sgm_add, shares_generator,
                    hz_concat, hz_intersect_polytope,
                    hz_apply_relu_v8, hz_apply_leaky_relu_v8,
                )
                # check_unsafe_for_act, lp_witness_to_input, strict_replay_for_act
                # are ACT-native (defined later in this module).
            except ImportError as e:
                raise RuntimeError(
                    f"HZVerifier: HYZOR_USE_ACT=0 requested LEGACY HyZor "
                    f"pkg but cannot import (HyZor pkg deleted post-port). "
                    f"Set HYZOR_USE_ACT=1 (default) to use ACT-native impl. "
                    f"Underlying: {e}"
                )
        else:
            from act.back_end.hybridz_tf.hz_routing import (
                hz_from_bounds, hz_dense, hz_conv2d, hz_add_const, hz_scale,
                hz_bn, hz_minkowski_sum, hz_sgm_add, shares_generator,
                hz_concat, hz_intersect_polytope,
                hz_apply_relu_v8, hz_apply_leaky_relu_v8,
            )
            # check_unsafe_for_act, lp_witness_to_input, strict_replay_for_act
            # are defined later in this same module (post-consolidation).

        # ACT operators (sigmoid/tanh K-piece -- ACT innovation)
        from act.back_end.hybridz_tf.tf_mlp import (
            hz_apply_sigmoid as act_hz_apply_sigmoid,
            hz_apply_tanh as act_hz_apply_tanh,
        )

        # ─── Phase 1: cons walker ───
        cons_by_layer: Dict[int, List[Con]] = {}
        global_polys: List[Con] = []
        for con in globalC:
            tag = con.meta.get("tag", "")
            if tag == "in:linpoly":
                global_polys.append(con); continue
            if tag.startswith("box:"): continue
            if ":" in tag:
                try:
                    lid = int(tag.split(":")[-1])
                    cons_by_layer.setdefault(lid, []).append(con)
                except ValueError: pass

        # Build initial input HZ
        input_box = self._extract_input_box(globalC, input_ids, before)
        device = torch.device(self.cfg["device"])
        dtype = self.cfg["dtype"]
        input_hz = hz_from_bounds(input_box, dtype=dtype, device=device)
        for poly_con in global_polys:
            input_hz = hz_intersect_polytope(
                input_hz, poly_con.meta["A"], poly_con.meta["b"])

        var_to_hz: Dict[Tuple[int, ...], Any] = {tuple(input_ids): input_hz}

        # ── Pre-scan: detect large_cls_proof_mode + count relus/convs ──
        relu_layer_ids: List[int] = []
        conv_count = 0
        for L in net.layers:
            ku = L.kind.upper()
            if ku == "RELU": relu_layer_ids.append(L.id)
            elif ku in ("CONV2D", "CONV1D", "CONV3D"): conv_count += 1
        total_relu = len(relu_layer_ids)
        out_dim = len(output_ids)

        lc_mode = self.cfg["large_cls_proof_mode"]
        if lc_mode == "auto":
            large_cls_active = (
                self.cfg["relu_method"] == "eq_lagr_v8"
                and conv_count >= self.cfg["large_cls_conv_threshold"]
                and out_dim >= self.cfg["large_cls_out_dim_threshold"]
            )
        else:
            large_cls_active = (lc_mode == "on")

        # ReLU index bookkeeping: which relu (1..total) is the next one we hit
        relu_idx_map: Dict[int, int] = {
            lid: i + 1 for i, lid in enumerate(relu_layer_ids)
        }
        eq_last = self.cfg["large_cls_eq_layers"]

        if large_cls_active:
            print(f"  [hyzor] large_cls_proof_mode ACTIVE: "
                  f"conv={conv_count} out_dim={out_dim} relus={total_relu} "
                  f"(triangle for relu 1..{total_relu - eq_last}, "
                  f"eq_lagr_v8 for last {eq_last})", flush=True)
            self._stats["large_cls_active"] = True
            self._stats["total_relu"] = total_relu
            self._stats["eq_last"] = eq_last
        self._lc_active = large_cls_active
        self._relu_idx_map = relu_idx_map
        self._total_relu = total_relu

        # ─── Optional dispatch: SpecAwareLP for small-dense networks ───
        # Small-dense optional dispatch to GlobalTriangleLP + SpecAwareLP.
        # NOTE: cross-check vs arXiv-2512.19007v1 VNN-COMP 2025 ground truth
        # showed this path is UNSOUND on acasxu (44 false positives: GT=sat
        # reported as verified). Default flipped to "off" until the
        # SpecAwareLP infeasibility encoding is audited; explicit "on" or
        # "auto" still available via env for research / paper artifacts.
        _gtlp_mode = os.environ.get("HYZOR_USE_GLOBAL_LP", "off").lower()
        _onnx_p = self.cfg.get("onnx_path")
        _vnn_p = self.cfg.get("vnnlib_path")
        if _gtlp_mode != "off" and _onnx_p and _vnn_p and conv_count == 0:
            try:
                if os.environ.get("HYZOR_USE_ACT", "1") == "1":
                    # is_small_dense + verify_with_falsification live later in
                    # this same module (post-consolidation).
                    _is_sd = is_small_dense
                    _we_verify = verify_with_falsification
                else:
                    import sys as _sys_gtlp
                    _hz_root = os.environ.get("HYZOR_ROOT", _HYZOR_ROOT)
                    if _hz_root not in _sys_gtlp.path:
                        _sys_gtlp.path.insert(0, _hz_root)
                    from GlobalTriangleLP import is_small_dense as _is_sd
                    from WitnessExtract import verify_with_falsification as _we_verify
                _dispatch = (_gtlp_mode == "on") or (
                    _gtlp_mode == "auto" and _is_sd(_onnx_p)
                )
                if _dispatch:
                    print(f"  [hyzor] WitnessExtract(SA+falsify) dispatch: "
                          f"mode={_gtlp_mode} onnx={Path(_onnx_p).name} "
                          f"vnnlib={Path(_vnn_p).name}", flush=True)
                    _verdict, _x_wit, _y_wit, _elapsed = _we_verify(
                        _onnx_p, _vnn_p,
                        time_limit_per_lp=15.0,
                        max_refinement_passes=3,
                        return_witness=True,
                    )
                    self._stats["global_lp_dispatch"] = True
                    self._stats["global_lp_verdict"] = _verdict
                    self._stats["global_lp_elapsed_s"] = _elapsed
                    if _verdict == "verified":
                        self._status = SolveStatus.UNSAT  # safe
                        return self._status
                    elif _verdict == "falsified":
                        # Stash witness so verify_once promotes SAT → FALSIFIED
                        import numpy as _np_wit
                        self._witness = _np_wit.asarray(_x_wit, dtype=_np_wit.float64).ravel()
                        self._has_solution = True
                        self._status = SolveStatus.SAT
                        return self._status
                    else:
                        # WitnessExtract demotes all non-verified-non-falsified
                        # (unknown OR solver fail) to 'unknown'. No fallback to
                        # HZ cascade for small-dense — HZ K-sweep best is 5/186
                        # vs WE 74V+15A=89/186.
                        self._status = SolveStatus.UNKNOWN
                        return self._status
            except Exception as _e_gtlp:
                print(f"  [hyzor] SpecAwareLP dispatch error "
                      f"({type(_e_gtlp).__name__}: {_e_gtlp}); falling back",
                      flush=True)
                self._stats["global_lp_dispatch"] = False
                self._stats["global_lp_error"] = f"{type(_e_gtlp).__name__}: {_e_gtlp}"

        op_counts: Dict[str, int] = {}
        for L in net.layers:
            if L.kind in ("INPUT", "INPUT_SPEC", "ASSERT"):
                continue

            in_var_tuple = tuple(L.in_vars)
            hz_in = var_to_hz.get(in_var_tuple)
            multi_in_hzs = (
                self._collect_multi_input_hzs(L, var_to_hz, net)
                if hz_in is None else None
            )

            cons_list = cons_by_layer.get(L.id, [])
            op_con = next(
                (c for c in cons_list
                 if not c.meta.get("tag", "").startswith("box:")),
                None
            )

            # Per-layer logging (so OOM-kill leaves a trail)
            tag_for_log = (op_con.meta["tag"] if op_con else f"box-fallback({L.kind})")
            in_dim = hz_in.dim if hz_in is not None else "n/a"
            in_ng = hz_in.ng if hz_in is not None else "n/a"
            print(f"  [hyzor L{L.id}] {tag_for_log}  "
                  f"in: dim={in_dim} ng={in_ng}", flush=True)
            try:
                # MaxPool: ACT cons_exporter doesn't generate constraints
                # for max-pool (it's not a linear op), so op_con is None
                # and the generic _dispatch fall-through would land in
                # _box_fallback (discarding all HZ correlation). Instead,
                # detect MAXPOOL2D explicitly here and route to the HZ
                # max_pool_node_evaluate via the hz_maxpool2d facade,
                # which preserves stable-winner rows exactly and falls
                # back to interval only on unstable blocks.
                if op_con is None and L.kind == "MAXPOOL2D" and hz_in is not None:
                    try:
                        if os.environ.get("HYZOR_USE_ACT", "1") == "1":
                            from act.back_end.hybridz_tf.hz_routing import hz_maxpool2d
                        else:
                            from HyZor import hz_maxpool2d
                        params = L.params
                        in_shape = params.get("input_shape")
                        if in_shape is None:
                            # fall back to box if shape missing
                            hz_out = self._box_fallback(L, after, hz_from_bounds)
                        else:
                            hz_out = hz_maxpool2d(
                                hz_in,
                                kernel_size=params["kernel_size"],
                                stride=params.get("stride"),
                                padding=params.get("padding", 0),
                                input_shape=in_shape,
                            )
                    except Exception as e:
                        # Sound fallback on any failure (shape mismatch etc.)
                        self._stats[f"maxpool_fallback@{L.id}"] = f"{type(e).__name__}: {e}"
                        hz_out = self._box_fallback(L, after, hz_from_bounds)
                else:
                    hz_out = self._dispatch(
                        L, op_con, hz_in, multi_in_hzs, before, after,
                        # HyZor ops
                        hz_dense=hz_dense, hz_conv2d=hz_conv2d,
                        hz_add_const=hz_add_const, hz_scale=hz_scale, hz_bn=hz_bn,
                        hz_sgm_add=hz_sgm_add, hz_minkowski_sum=hz_minkowski_sum,
                        shares_generator=shares_generator, hz_concat=hz_concat,
                        hz_apply_relu_v8=hz_apply_relu_v8,
                        hz_apply_leaky_relu_v8=hz_apply_leaky_relu_v8,
                        hz_from_bounds=hz_from_bounds,
                        # ACT ops (sigmoid/tanh K-piece)
                        act_hz_apply_sigmoid=act_hz_apply_sigmoid,
                        act_hz_apply_tanh=act_hz_apply_tanh,
                    )

                # Girard reduction: cap ng to keep memory bounded
                hz_out = self._maybe_reduce(hz_out)

                print(f"  [hyzor L{L.id}] done  "
                      f"out: dim={hz_out.dim} ng={hz_out.ng} nb={hz_out.nb} nc={hz_out.nc}",
                      flush=True)
            except Exception as e:
                # Sound fallback on any per-layer failure
                self._stats[f"error@{L.id}"] = f"{type(e).__name__}: {e}"
                hz_out = self._box_fallback(L, after, hz_from_bounds)
                import os as _osdbg, traceback as _tb
                if _osdbg.environ.get("HYZOR_DEBUG_FALLBACK", "0") == "1":
                    print(f"  [hyzor L{L.id}] FALLBACK ({type(e).__name__}): {e}",
                          flush=True)
                    _tb.print_exc()
                else:
                    print(f"  [hyzor L{L.id}] FALLBACK ({type(e).__name__})",
                          flush=True)

            var_to_hz[tuple(L.out_vars)] = hz_out
            op_kind = (op_con.meta["tag"].split(":")[0]
                       if op_con else f"box-fallback({L.kind})")
            op_counts[op_kind] = op_counts.get(op_kind, 0) + 1

        out_hz = var_to_hz.get(tuple(output_ids))
        if out_hz is None:
            self._status = SolveStatus.UNKNOWN
            self._stats["error"] = "no output HZ"
            return self._status
        self._stats["op_counts"] = op_counts

        # Stash for Tier 1A LP-aggressive retry: if Phase 2/4 ends UNKNOWN,
        # we re-walk with relu_method='exact_lp' (tighter ReLU encoding).
        self._first_pass_method = self.cfg["relu_method"]

        # ─── Phase 2: LP feasibility ───
        try:
            feas, xi_star = check_unsafe_for_act(
                out_hz, assert_layer,
                output_ids=output_ids,
                timeout_s=self.cfg["timeout_s"]
            )
        except Exception as e:
            self._status = SolveStatus.UNKNOWN
            self._stats["feasibility_error"] = f"{type(e).__name__}: {e}"
            return self._status

        # Tier 1A: LP-aggressive tail. Try retry with exact_lp on UNKNOWN-likely
        # outcomes (timeout or feasible-but-replay-rejected). Gating: small/med
        # nets only, not large_cls, not already exact_lp.
        def _should_try_lp_tail() -> bool:
            return (self.cfg.get("l2_exactlp_tail", False)
                    and not large_cls_active
                    and self.cfg["relu_method"] != "exact_lp"
                    and total_relu <= self.cfg.get("l2_exactlp_relu_cap", 128)
                    and out_dim <= self.cfg.get("l2_exactlp_out_cap", 512))

        if feas == "infeasible":
            # v9.2: PDNT multi-cert. dual_cert_n=1 trusts LP single cert.
            # dual_cert_n>=2 requires additional independent cert(s) to
            # confirm before declaring "verified".
            n_certs_required = self.cfg["dual_cert_n"]
            if n_certs_required <= 1:
                self._status = SolveStatus.UNSAT
                self._stats["dual_cert_n"] = 1
                return self._status
            # Run additional cert: UNC bound on out_hz vs assert_layer
            extra_certs_pass = self._extra_certs_verified(
                out_hz, assert_layer, output_ids)
            n_passing = 1 + sum(extra_certs_pass.values())  # +1 for LP
            self._stats["dual_cert_n_required"] = n_certs_required
            self._stats["dual_cert_passes"] = n_passing
            self._stats["dual_cert_detail"] = extra_certs_pass
            if n_passing >= n_certs_required:
                self._status = SolveStatus.UNSAT
                return self._status
            # Multi-cert disagreement → conservative downgrade to UNKNOWN
            self._status = SolveStatus.UNKNOWN
            self._stats["dual_cert_rejected"] = True
            return self._status
        if feas == "timeout":
            # Tier 1A: retry with exact_lp before giving up
            if _should_try_lp_tail() and not getattr(self, "_lp_tail_active", False):
                self._stats["l2_exactlp_tail_attempted"] = True
                return self._retry_with_exact_lp(
                    globalC, before, after,
                    net=net, input_ids=input_ids,
                    output_ids=output_ids, assert_layer=assert_layer,
                    reason="phase2_timeout")
            self._status = SolveStatus.UNKNOWN
            self._stats["timeout"] = True
            return self._status

        # ─── Phase 3: witness back to input space ───
        try:
            x_star = lp_witness_to_input(xi_star, input_hz)
        except Exception as e:
            self._status = SolveStatus.UNKNOWN
            self._stats["witness_error"] = f"{type(e).__name__}: {e}"
            return self._status

        # ─── Phase 4: strict replay ───
        if self.cfg["strict_replay"]:
            try:
                # F1: prefer ORT replay over torch fallback (avoids 1e-6 fp drift
                # that caused phantom_rejected on acasxu).
                if self.cfg.get("onnx_path") and not getattr(net, "onnx_path", None):
                    try: net.onnx_path = self.cfg["onnx_path"]
                    except Exception: pass
                ok = strict_replay_for_act(
                    net=net, x_star=x_star, assert_layer=assert_layer
                )
            except Exception as e:
                ok = False
                self._stats["replay_error"] = f"{type(e).__name__}: {e}"
            if not ok:
                # Tier 1A: retry with exact_lp before giving up
                if _should_try_lp_tail() and not getattr(self, "_lp_tail_active", False):
                    self._stats["l2_exactlp_tail_attempted"] = True
                    return self._retry_with_exact_lp(
                        globalC, before, after,
                        net=net, input_ids=input_ids,
                        output_ids=output_ids, assert_layer=assert_layer,
                        reason="phase4_replay_rejected")
                self._status = SolveStatus.UNKNOWN
                self._stats["phantom_rejected"] = True
                return self._status

        self._status = SolveStatus.SAT
        self._witness = np.asarray(x_star, dtype=np.float64).ravel()
        self._has_solution = True
        # Final cleanup: release intermediate HZ tensors held in var_to_hz
        try:
            del var_to_hz, out_hz, input_hz
            import torch as _t
            if _t.cuda.is_available() and self.cfg.get("device","cpu").startswith("cuda"):
                _t.cuda.empty_cache()
        except Exception:
            pass
        return self._status

    # ----- Per-layer dispatch (cons tag -> HyZor or ACT op) -----
    def _dispatch(self, L, op_con, hz_in, multi_in_hzs, before, after, **ops):
        if op_con is None:
            return self._box_fallback(L, after, ops["hz_from_bounds"])
        tag = op_con.meta["tag"]; op = tag.split(":")[0]; meta = op_con.meta

        # PR3 memory guard: predict output dim; if dim × ng exceeds budget,
        # fall back to box (interval) to avoid OOM. Used for ImageNet-scale
        # models like VGG16 (n=3.2M after first conv, ng=150K input pixels).
        # Budget defaults to 4 GB (configurable via env).
        import os as _os
        guard_gb = float(_os.environ.get("HYZOR_DISPATCH_GUARD_GB", "4.0"))
        guard_bytes = int(guard_gb * 1024 ** 3)
        def _would_oom(out_dim, in_ng):
            est = out_dim * in_ng * 8 * 3  # ×3 for intermediate copies
            return est > guard_bytes
        # Estimate output dim for the layer
        try:
            if op == "dense":
                out_dim = int(meta["W"].shape[0])
            elif op == "conv2d":
                out_shape = meta.get("output_shape")
                out_dim = int(out_shape[1] * out_shape[2] * out_shape[3]) if out_shape else 0
            else:
                out_dim = 0
            in_ng = int(hz_in.ng) if hz_in is not None else 0
            if out_dim > 0 and in_ng > 0 and _would_oom(out_dim, in_ng):
                self._stats[f"oom_guard@{L.id}"] = f"{op}: out_dim={out_dim}, ng={in_ng}, est_GB={out_dim*in_ng*8*3/(1024**3):.1f}"
                return self._box_fallback(L, after, ops["hz_from_bounds"])
        except Exception:
            pass  # if estimation fails, just attempt the op normally

        # ── HyZor ops ──
        if op == "dense":
            return ops["hz_dense"](hz_in, meta["W"], meta.get("b"))
        if op == "conv2d":
            cp = meta.get("conv_params", {})
            return ops["hz_conv2d"](
                hz_in, meta["weight"], meta.get("b"),
                input_shape=meta["input_shape"],
                stride=cp.get("stride", 1), padding=cp.get("padding", 0),
                dilation=cp.get("dilation", 1), groups=cp.get("groups", 1)
            )
        if op == "bias":  return ops["hz_add_const"](hz_in, meta["c"])
        if op == "scale": return ops["hz_scale"](hz_in, meta["a"])
        if op == "bn":    return ops["hz_bn"](hz_in, meta["A"], meta["c"])
        if op == "add":
            if multi_in_hzs is None or len(multi_in_hzs) < 2:
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            hz_x, hz_y = multi_in_hzs[0], multi_in_hzs[1]
            return (ops["hz_sgm_add"](hz_x, hz_y)
                    if (self.cfg["sgm_enabled"] and
                        ops["shares_generator"](hz_x, hz_y))
                    else ops["hz_minkowski_sum"](hz_x, hz_y))
        if op == "sub":
            # z = x - y. Negate hz_y, then minkowski sum.
            if multi_in_hzs is None or len(multi_in_hzs) < 2:
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            hz_x, hz_y = multi_in_hzs[0], multi_in_hzs[1]
            from act.back_end.solver.solver_hz import HZono as _HZ
            hz_y_neg = _HZ(
                c=-hz_y.c, Gc=-hz_y.Gc, Gb=-hz_y.Gb,
                Ac=hz_y.Ac.clone(), Ab=hz_y.Ab.clone(), b=hz_y.b.clone(),
                eq_mask=(hz_y.eq_mask.clone()
                          if hz_y.eq_mask is not None else None),
            )
            return ops["hz_minkowski_sum"](hz_x, hz_y_neg)
        if op == "concat":
            if multi_in_hzs is None: multi_in_hzs = [hz_in]
            return ops["hz_concat"](multi_in_hzs)
        if op == "relu":
            # large_cls_proof_mode: triangle for early relus, eq_lagr_v8 for last N
            method = self.cfg["relu_method"]
            if getattr(self, "_lc_active", False):
                ridx = self._relu_idx_map.get(L.id, 0)
                eq_last = self.cfg["large_cls_eq_layers"]
                if ridx <= self._total_relu - eq_last:
                    method = "triangle"
            return ops["hz_apply_relu_v8"](
                hz_in,
                method=method,
                mace=self.cfg["mace_enabled"],
                girard_cap=self.cfg["girard_cap"]
            )
        if op == "lrelu":
            return ops["hz_apply_leaky_relu_v8"](hz_in, alpha=meta["alpha"])

        # ── ACT ops (sigmoid/tanh K-piece -- ACT innovation) ──
        # Dim guard: ACT's hz_apply_piecewise has a Python-level loop over
        # wide neurons; for dim > sigmoid_dim_cap it becomes prohibitively
        # slow. Fall back to box (sound) on large dims.
        if op == "sigmoid":
            cap = int(os.environ.get("HYZOR_SIGMOID_DIM_CAP", "256"))
            if int(hz_in.dim) > cap:
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            hzono_in = self._hyzor_to_hzono(hz_in)
            hzono_out = ops["act_hz_apply_sigmoid"](
                hzono_in, K=self.cfg["sigmoid_K"]
            )
            return self._hzono_to_hyzor(hzono_out)
        if op == "tanh":
            cap = int(os.environ.get("HYZOR_TANH_DIM_CAP", "256"))
            if int(hz_in.dim) > cap:
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            hzono_in = self._hyzor_to_hzono(hz_in)
            hzono_out = ops["act_hz_apply_tanh"](
                hzono_in, K=self.cfg["tanh_K"]
            )
            return self._hzono_to_hyzor(hzono_out)

        # ── Shape ops ──
        if op in ("flatten", "reshape", "transpose", "squeeze",
                  "unsqueeze", "tile", "expand"):
            return hz_in
        # SLICE actually subsets dims; box-fallback is sound (looser but correct)
        if op == "slice":
            return self._box_fallback(L, after, ops["hz_from_bounds"])

        # ── Fallback ──
        return self._box_fallback(L, after, ops["hz_from_bounds"])

    # ----- Helpers -----
    def _retry_with_exact_lp(
        self, globalC, before, after, *,
        net, input_ids, output_ids, assert_layer, reason: str
    ):
        """Tier 1A: re-run consume_cons with relu_method='exact_lp'.

        exact_lp encoding has more LP constraints per ReLU than eq_lagr_v8
        (tighter relaxation). For UNKNOWN cases on small/medium nets, this
        often promotes to verified. Only called once per query (guarded by
        self._lp_tail_active flag) — sound by construction.
        """
        prev_method = self.cfg["relu_method"]
        self._lp_tail_active = True
        self.cfg["relu_method"] = "exact_lp"
        try:
            print(f"  [hyzor] LP-aggressive retry (reason={reason}, method=exact_lp)",
                  flush=True)
            result = self.consume_cons(
                globalC, before, after,
                net=net, input_ids=input_ids,
                output_ids=output_ids, assert_layer=assert_layer,
            )
            self._stats["l2_exactlp_tail_outcome"] = result
            return result
        finally:
            self.cfg["relu_method"] = prev_method
            self._lp_tail_active = False

    def _extract_input_box(self, globalC, input_ids, before):
        for con in globalC:
            tag = con.meta.get("tag", "")
            if tag.startswith("box:") and set(con.var_ids) == set(input_ids):
                return Bounds(lb=con.meta["lb"], ub=con.meta["ub"])
        for lid, fact in before.items():
            return fact.bounds
        raise RuntimeError("HZVerifier: cannot find input box")

    def _collect_multi_input_hzs(self, L, var_to_hz, net):
        out = []
        for pid in net.preds.get(L.id, []):
            pred_layer = net.by_id[pid]
            tup = tuple(pred_layer.out_vars)
            if tup in var_to_hz:
                out.append(var_to_hz[tup])
        return out

    def _box_fallback(self, L, after, hz_from_bounds):
        b = after[L.id].bounds
        return hz_from_bounds(
            Bounds(b.lb, b.ub),
            dtype=self.cfg["dtype"],
            device=torch.device(self.cfg["device"])
        )

    def _maybe_reduce(self, hz):
        """Apply Girard generator reduction if ng exceeds budget.

        Bounds memory by capping continuous generators at girard_cap.
        Sound (over-approximation)."""
        cap = self.cfg["girard_cap"]
        if int(hz.ng) <= cap:
            return hz
        try:
            return hz.reduce_constraints(ng_budget=cap)
        except Exception:
            return hz

    def _extra_certs_verified(self, out_hz, assert_layer, output_ids) -> dict:
        """v9.2 PDNT extra certs (besides the LP cert that just said infeasible).

        Returns dict of {cert_name: bool}. Each True means that independent
        cert also proves the unsafe predicate is false (i.e. spec verified).

        Pool currently implemented:
          - U: UNC bound (cheapest, sound). Just compute Girard interval
               bounds on out_hz and check spec is unsafe-infeasible.

        Future certs (E/Z/F) deferred to a follow-up minor; this 2-cert
        (LP + UNC) is the minimum useful multi-cert for the "directly verify"
        comparison setting.
        """
        pool = set(self.cfg["dual_cert_pool"].upper())
        results = {}
        if "U" in pool:
            results["U"] = self._cert_U_unc_verified(out_hz, assert_layer)
        return results

    def _cert_U_unc_verified(self, out_hz, assert_layer) -> bool:
        """UNC cert: use Girard interval bounds to prove unsafe-infeasible."""
        try:
            lb_t, ub_t = out_hz._bounds_unconstrained()
            import numpy as np
            lb = lb_t.detach().cpu().numpy().reshape(-1)
            ub = ub_t.detach().cpu().numpy().reshape(-1)
            kind = assert_layer.params.get("kind")
            kstr = str(kind).split(".")[-1] if hasattr(kind, "__class__") else str(kind)
            eps = self.cfg["dual_cert_margin"]

            def _unwrap_int(x):
                if hasattr(x, "item"): return int(x.item())
                if hasattr(x, "__len__"): return int(x[0])
                return int(x)

            def _to_np(x):
                import torch as _t
                if _t.is_tensor(x): return x.detach().cpu().numpy()
                return np.asarray(x)

            if kstr == "TOP1_ROBUST":
                t = _unwrap_int(assert_layer.params["y_true"])
                # Spec: y[t] > y[j] for all j ≠ t
                # Spec verified iff lb[t] > max_{j≠t} ub[j]   (with eps)
                others_ub = np.delete(ub, t)
                return float(lb[t]) > float(others_ub.max()) + eps
            if kstr == "MARGIN_ROBUST":
                t = _unwrap_int(assert_layer.params["y_true"])
                m = float(_to_np(assert_layer.params["margin"]).reshape(-1)[0])
                others_ub = np.delete(ub, t)
                return float(lb[t]) > float(others_ub.max()) + m + eps
            if kstr == "LINEAR_LE":
                # Spec: c·y ≤ d.   Verified iff max c·y ≤ d.
                coef = _to_np(assert_layer.params["c"]).reshape(-1)
                d = float(_to_np(assert_layer.params["d"]).reshape(-1)[0])
                pos = np.clip(coef, 0, None); neg = np.clip(coef, None, 0)
                cy_max = float(pos @ ub + neg @ lb)
                return cy_max <= d - eps
            if kstr == "UNSAFE_LINEAR":
                # Spec violated (unsafe) iff ALL rows of C·y ≤ d.
                # Verified (NOT unsafe) iff ANY row C[i]·y > d[i] always.
                C = _to_np(assert_layer.params["c"])
                d_vec = _to_np(assert_layer.params["d"]).reshape(-1)
                if C.ndim == 1: C = C.reshape(1, -1)
                for i in range(C.shape[0]):
                    coef = C[i]
                    pos = np.clip(coef, 0, None); neg = np.clip(coef, None, 0)
                    cy_min = float(pos @ lb + neg @ ub)
                    if cy_min > float(d_vec[i]) + eps:
                        return True
                return False
            if kstr == "RANGE":
                # Spec: lb_t ≤ y ≤ ub_t. Verified iff actual bounds
                # are strictly inside spec bounds.
                lb_spec = assert_layer.params.get("lb")
                ub_spec = assert_layer.params.get("ub")
                if lb_spec is not None:
                    lb_v = _to_np(lb_spec).reshape(-1)
                    if (lb < lb_v + eps).any(): return False
                if ub_spec is not None:
                    ub_v = _to_np(ub_spec).reshape(-1)
                    if (ub > ub_v - eps).any(): return False
                return True
            return False
        except Exception:
            return False

    # ----- HZ conversion helpers (no-ops for ACT HZono; legacy rollback only) -----
    def _hyzor_to_hzono(self, hyzor_hz):
        from act.back_end.solver.solver_hz import HZono
        if isinstance(hyzor_hz, HZono):
            return hyzor_hz
        return HZono(
            c=hyzor_hz.c, Gc=hyzor_hz.Gc, Gb=hyzor_hz.Gb,
            Ac=hyzor_hz.Ac, Ab=hyzor_hz.Ab, b=hyzor_hz.b,
            eq_mask=getattr(hyzor_hz, "eq_mask", None),
        )

    def _hzono_to_hyzor(self, hzono):
        # Identity: downstream code only reads HZ-shape fields which
        # HZono exposes the same way as HyZor's HybridZonotope.
        return hzono

    # ----- Result accessors -----
    def status(self) -> str:
        return self._status

    def has_solution(self) -> bool:
        return self._has_solution

    def get_values(self, vids: List[int]) -> np.ndarray:
        if self._witness is None:
            return np.zeros(len(vids), dtype=np.float64)
        return self._witness[: len(vids)]

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        return self.get_values(input_ids)

    def stats(self) -> dict:
        return dict(self._stats)


def _slice_facts_lane(facts: Dict[int, Fact], lane: int) -> Dict[int, Fact]:
    """Return a new facts dict where each Fact's batched bounds are
    sliced to the requested batch lane. Cons (if any) are passed through
    unchanged — legacy consume_cons only reads bounds from facts."""
    out: Dict[int, Fact] = {}
    for lid, f in facts.items():
        lb, ub = f.bounds.lb, f.bounds.ub
        if lb.dim() >= 2:
            lb = lb[lane]
            ub = ub[lane]
        out[lid] = Fact(bounds=Bounds(lb=lb, ub=ub), cons=f.cons)
    return out


def _slice_assert_layer_lane(assert_layer: Layer, lane: int) -> Layer:
    """Build a de-batched copy of ASSERT layer for single-instance consumers.

    PR #66's ASSERT params carry leading B axis on per-kind fields
    (``y_true: [B]``, ``c: [B, ...]``, ``d: [B, ...]``, ``margin: [B]``,
    ``lb/ub: [B, n_out]``, ``thresholds: [B, M]``, ``C: [B*M, n_out]``).
    HyZor's ``check_unsafe_for_act`` / ``_cert_U_unc_verified`` predate
    this and expect the unbatched per-kind layout. This helper slices
    lane ``lane`` so those readers see the same shapes they always did.
    """
    new_params: Dict[str, Any] = {}
    B_hint: Optional[int] = None
    for k, v in assert_layer.params.items():
        if k in ("kind", "M"):
            new_params[k] = v
            continue
        if hasattr(v, "dim") and hasattr(v, "shape"):
            if v.dim() >= 1 and v.shape[0] > 0:
                # Heuristic: leading dim is B for per-kind fields. For the
                # pre-encoded C of shape [B*M, n_out] we keep as-is — readers
                # of "C" key are PR-#66 callers, not the legacy path.
                if k == "C":
                    new_params[k] = v
                else:
                    new_params[k] = v[lane]
                    if B_hint is None:
                        B_hint = int(v.shape[0])
            else:
                new_params[k] = v
        else:
            new_params[k] = v
    # Construct a shallow Layer-like wrapper. ASSERT consumers only read
    # `.params`; building a fresh Layer keeps the Layer dataclass invariants
    # of the host module untouched.
    from dataclasses import replace as _dc_replace
    try:
        return _dc_replace(assert_layer, params=new_params)
    except Exception:
        # Fallback if Layer isn't a dataclass on this main: use a thin proxy
        class _LayerProxy:  # noqa: D401
            pass
        proxy = _LayerProxy()
        for attr in ("id", "kind", "in_vars", "out_vars"):
            if hasattr(assert_layer, attr):
                setattr(proxy, attr, getattr(assert_layer, attr))
        proxy.params = new_params
        return proxy  # type: ignore[return-value]


def _slice_globalC_lane(globalC: ConSet, lane: int) -> ConSet:
    """Slice batched ``box:`` meta-stored lb/ub to a single batch lane.
    Non-box cons are passed through unchanged."""
    out = ConSet()
    for sig, con in list(globalC.S.items()):
        meta = dict(con.meta) if con.meta else {}
        lb_meta = meta.get("lb")
        ub_meta = meta.get("ub")
        sliced = False
        if lb_meta is not None and hasattr(lb_meta, "dim") and lb_meta.dim() >= 2:
            meta["lb"] = lb_meta[lane].reshape(-1)
            sliced = True
        if ub_meta is not None and hasattr(ub_meta, "dim") and ub_meta.dim() >= 2:
            meta["ub"] = ub_meta[lane].reshape(-1)
            sliced = True
        if sliced:
            out.replace(Con(kind=con.kind, var_ids=con.var_ids, meta=meta))
        else:
            out.S[sig] = con
    return out


# ─── LEGACY_SHIM_TO_REMOVE_AT_P3 ───────────────────────────────────────
# Single-instance verifier entry that mirrors the pre-PR-#66
# ``verify_once(net, solver=..., timelimit=...)`` semantics on top of
# HyZor's ``consume_cons`` cons-IR walker. Driver scripts that pre-date
# the batch-native verifier (v100/v101/v102 and similar) call this
# helper instead of ``act.back_end.verifier.verify_once`` (which no
# longer accepts a ``solver=`` argument).
#
# Inputs:
#   - ``net``: ACT Net whose first layer is INPUT, last is ASSERT.
#     INPUT_SPEC may be batched [B, *shape]; this helper takes lane
#     ``batch_lane`` only (default 0; raises if B>1 and lane unset).
#   - ``solver``: HZVerifier instance.
#   - ``timelimit``: optional wall-clock budget (seconds).
#
# Returns: ``(status: str, ce_input: Optional[np.ndarray], stats: dict)``
# matching the pre-PR-#66 return type.
#
# REMOVAL PLAN: once HyZor's HZ propagation lives in hybridz_tf and the
# cascade controller in hybridz_tf/algorithms, the new
# ``setup_and_solve_batch`` will dispatch to HyZor natively and this
# helper can be deleted.
def verify_once_legacy_batch1(
    net,
    *,
    solver: "HZVerifier",
    timelimit: Optional[float] = None,
    batch_lane: int = 0,
) -> Tuple[str, Optional[np.ndarray], Dict[str, Any]]:
    """Pre-PR-#66 verify_once API on top of HZVerifier.consume_cons.

    See module-level comment ``LEGACY_SHIM_TO_REMOVE_AT_P3``.
    """
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import set_transfer_function_mode
    from act.back_end.verifier import (
        find_entry_layer_id, get_input_ids, get_output_ids,
        gather_input_spec_layers, get_assert_layer,
        seed_from_input_specs, add_all_input_specs, validate_constraints,
    )

    # consume_cons reads `.bounds` from before/after for _box_fallback
    # paths, so we need TIGHT bounds. An earlier attempt forced
    # interval-only TF for speed; it caused a regression on
    # cifar100_resnet_large (20 V down from baseline 52) because
    # looser box fallbacks turned verified instances into "unknown".
    # Default mode honored (hybridz). Set HYZOR_TF_MODE=interval to
    # force the fast-but-lossy path when speed matters more than recall.
    _tf_mode = os.environ.get("HYZOR_TF_MODE", "").strip().lower()
    if _tf_mode in ("interval", "hybridz"):
        set_transfer_function_mode(_tf_mode)

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    # Run analyze with the batched seed (new TFs require [B, *shape]).
    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2:
        # Legacy 1-D seed: synthesize a B=1 batch so new TFs accept it.
        seed_bounds = Bounds(
            lb=seed_bounds.lb.unsqueeze(0), ub=seed_bounds.ub.unsqueeze(0)
        )
    B = int(seed_bounds.lb.shape[0])
    if batch_lane >= B:
        raise IndexError(
            f"verify_once_legacy_batch1: batch_lane={batch_lane} out of "
            f"range [0, {B})"
        )

    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    before, after, globalC = analyze(net, entry_id, entry_fact)
    validate_constraints(globalC, after, net)

    # HyZor's consume_cons predates batch-native analyze and expects
    # single-lane (1-D) bounds in before/after Facts. Slice lane
    # ``batch_lane`` so consume_cons sees the same input shape it always
    # has. globalC may also carry per-batch box rows; rewrite them to
    # the single-lane view.
    before_b1, after_b1 = _slice_facts_lane(before, batch_lane), _slice_facts_lane(after, batch_lane)
    globalC_b1 = _slice_globalC_lane(globalC, batch_lane)
    assert_layer_b1 = _slice_assert_layer_lane(assert_layer, batch_lane)

    if timelimit is not None and hasattr(solver, "cfg"):
        solver.cfg["timeout_s"] = float(timelimit)

    st = solver.consume_cons(
        globalC_b1, before_b1, after_b1,
        net=net, input_ids=input_ids, output_ids=output_ids,
        assert_layer=assert_layer_b1,
    )
    ce_input = None
    if st == SolveStatus.SAT and solver.has_solution():
        ce_input = solver.get_values(input_ids)

    stats: Dict[str, Any] = {
        "status": st, "ncons": len(globalC), "solver": "hyzor",
    }
    try:
        stats.update(solver.stats() if hasattr(solver, "stats") else {})
    except Exception:
        pass
    return st, ce_input, stats


# ======================================================================
# Final LP verification (unsafe-set feasibility + witness)
# (inlined from former hz_lp_verify.py)
# ======================================================================

from typing import Optional, Tuple, List




# OutKind values from act.back_end.layer_schema (don't import the enum at
# module load time; just compare against the string form).


def _to_np_f64(t):
    if torch.is_tensor(t):
        return t.detach().cpu().double().numpy()
    return np.asarray(t, dtype=np.float64)


def _kind_str(kind) -> str:
    """Coerce any kind representation to its trailing identifier string."""
    if kind is None:
        return ""
    s = str(kind)
    return s.split(".")[-1]


def _build_factor_lp(hz: HZono):
    """Build the constant parts of every LP we'll solve over hz.

    Returns dict with numpy arrays:
        Gc_n (n, p), Gb_n (n, q), c_n (n,)
        A_eq (m_eq, p+q), b_eq (m_eq,)
        A_le (m_le, p+q), b_le (m_le,)
        p, q
    """
    em_t = _eq_mask_of(hz)
    em = em_t.detach().cpu().numpy().astype(bool)
    le = ~em
    Ac_np = _to_np_f64(hz.Ac)
    Ab_np = _to_np_f64(hz.Ab)
    b_np = _to_np_f64(hz.b).reshape(-1)
    A_eq = np.concatenate([Ac_np[em], Ab_np[em]], axis=1) \
        if em.any() else np.zeros((0, hz.Ac.shape[1] + hz.Ab.shape[1]))
    b_eq = b_np[em] if em.any() else np.zeros(0)
    A_le = np.concatenate([Ac_np[le], Ab_np[le]], axis=1) \
        if le.any() else np.zeros((0, hz.Ac.shape[1] + hz.Ab.shape[1]))
    b_le = b_np[le] if le.any() else np.zeros(0)
    return {
        "Gc": _to_np_f64(hz.Gc),
        "Gb": _to_np_f64(hz.Gb),
        "c": _to_np_f64(hz.c).reshape(-1),
        "A_eq": A_eq, "b_eq": b_eq,
        "A_le": A_le, "b_le": b_le,
        "p": int(hz.Gc.shape[1]),
        "q": int(hz.Gb.shape[1]),
    }


def _lp_feas_or_minimize(prob, obj_row: np.ndarray, rhs_threshold: Optional[float],
                          sense: str = "maximize",
                          timeout_s: Optional[float] = None
                          ) -> Tuple[str, Optional[np.ndarray]]:
    """Solve one LP: ``sense (obj_row . xi + obj_const)`` s.t. hz constraints.

    Returns ``("feasible", xi_star)`` / ``("infeasible", None)`` /
    ``("timeout", None)``.

    If ``rhs_threshold`` is given, we EARLY-EXIT: once we have a feasible
    point whose objective beats the threshold we return feasible (the
    LP also gives back xi_star). If the LP's optimum is bounded by
    ``rhs_threshold`` the disjunct is infeasible (spec holds on that
    branch).
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return "feasible", None  # conservative

    p, q = prob["p"], prob["q"]
    nvars = p + q
    bounds = [(-1.0, 1.0)] * nvars
    # linprog minimizes; for maximize we negate.
    c = -obj_row.copy() if sense == "maximize" else obj_row.copy()
    A_eq = prob["A_eq"] if prob["A_eq"].shape[0] > 0 else None
    b_eq = prob["b_eq"] if prob["A_eq"].shape[0] > 0 else None
    A_ub = prob["A_le"] if prob["A_le"].shape[0] > 0 else None
    b_ub = prob["b_le"] if prob["A_le"].shape[0] > 0 else None
    options = {}
    if timeout_s is not None:
        options["time_limit"] = float(timeout_s)
    try:
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method="highs", options=options)
    except Exception:
        return "feasible", None
    if res.status == 0 and res.success:
        # Optimal found.
        obj_val = -res.fun if sense == "maximize" else res.fun
        if rhs_threshold is not None:
            # spec says obj_row . y <= threshold (LINEAR_LE) or analogous.
            # Disjunct infeasible iff max obj_row . y <= threshold.
            if (sense == "maximize" and obj_val <= rhs_threshold) or \
               (sense == "minimize" and obj_val >= rhs_threshold):
                return "infeasible", None
        return "feasible", res.x
    if res.status == 2:
        # Primal infeasible.
        return "infeasible", None
    if res.status == 1:
        return "timeout", None
    return "feasible", None  # conservative on unknown status


def check_unsafe_for_act(out_hz: HZono, assert_layer, *,
                         output_ids=None,
                         timeout_s: float = 30.0
                         ) -> Tuple[str, Optional[np.ndarray]]:
    """Decide whether the unsafe set is reachable from ``out_hz`` for the
    spec stored in ``assert_layer.params``.

    Returns:
        ``("feasible", xi_star)`` -- unsafe set has a candidate witness
                                     in factor space; spec may be
                                     violated. Caller must replay.
        ``("infeasible", None)``  -- spec verified on every disjunct.
        ``("timeout", None)``     -- gave up; report UNKNOWN.

    ``out_hz``'s LP relaxation is used (binary xi_b relaxed to [-1, 1]).
    This is sound for ``infeasible`` (no integer point can satisfy what
    the relaxation rules out) but ``feasible`` may have spurious
    witnesses; caller must concretely replay and confirm.
    """
    prob = _build_factor_lp(out_hz)
    p, q = prob["p"], prob["q"]
    Gc, Gb, c_vec = prob["Gc"], prob["Gb"], prob["c"]
    nvars = p + q
    kind = _kind_str(assert_layer.params.get("kind"))

    def _row_to_obj_y(coef: np.ndarray) -> Tuple[np.ndarray, float]:
        """y = c + Gc xi_c + Gb xi_b ⇒ coef·y = (coef·c) + obj_row · xi
        where obj_row = [coef·Gc, coef·Gb] of length nvars."""
        obj_row = np.concatenate([coef @ Gc, coef @ Gb], axis=0)
        obj_const = float(coef @ c_vec)
        return obj_row, obj_const

    t0 = time.perf_counter()
    def _remaining():
        return max(0.05, timeout_s - (time.perf_counter() - t0))

    if kind == "LINEAR_LE":
        coef = _to_np_f64(assert_layer.params["c"]).reshape(-1)
        d = float(_to_np_f64(assert_layer.params["d"]).reshape(-1)[0])
        obj_row, obj_const = _row_to_obj_y(coef)
        st, x = _lp_feas_or_minimize(
            prob, obj_row, rhs_threshold=d - obj_const,
            sense="maximize", timeout_s=_remaining(),
        )
        return st, x

    if kind == "UNSAFE_LINEAR":
        C = _to_np_f64(assert_layer.params["c"])
        d_vec = _to_np_f64(assert_layer.params["d"]).reshape(-1)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        # Unsafe set = {y : Cy <= d}. Polytope is unsafe-reachable iff
        # exists xi with C(c + Gc xi_c + Gb xi_b) <= d AND hz constraints.
        # Build one combined LP where we add C·y <= d as extra <=
        # inequalities and check feasibility (any feasible point works).
        # Equivalent to: find any xi satisfying hz + C·y <= d.
        N = C.shape[0]
        # Augment A_le with N rows: C @ Gc | C @ Gb, rhs = d - C @ c
        A_le_aug = np.concatenate(
            [prob["A_le"],
             np.concatenate([C @ Gc, C @ Gb], axis=1)],
            axis=0,
        )
        b_le_aug = np.concatenate([prob["b_le"], d_vec - C @ c_vec])
        prob2 = dict(prob)
        prob2["A_le"] = A_le_aug
        prob2["b_le"] = b_le_aug
        obj_row = np.zeros(nvars)  # pure feasibility
        st, x = _lp_feas_or_minimize(
            prob2, obj_row, rhs_threshold=None,
            sense="minimize", timeout_s=_remaining(),
        )
        return st, x

    if kind == "TOP1_ROBUST":
        t = int(_to_np_f64(assert_layer.params["y_true"]).reshape(-1)[0])
        n_out = c_vec.size
        # === OR pre-filter (HyZor __init__.py:1800) ===
        # Cheap cube UB: y_j - y_t = (c_j - c_t) + (Gc[j]-Gc[t]) xi_c + (Gb[j]-Gb[t]) xi_b
        # UB on xi ∈ [-1,1]^nvars = |coef_y|.sum(). Disjunct infeasible iff UB < 0.
        # For 200-class output (tinyimagenet) this skips most LPs.
        diffY = np.concatenate([Gc, Gb], axis=1)  # (n_out, nvars)
        ub_cube = c_vec + np.abs(diffY).sum(axis=1)  # (n_out,)
        ub_diff = ub_cube - c_vec[t]  # rough UB of y_j - y_t (loose)
        # Tighter: use vectorized actual diff
        diff_rows = diffY - diffY[t:t+1]
        diff_c = c_vec - c_vec[t]
        ub_diff_tight = diff_c + np.abs(diff_rows).sum(axis=1)  # (n_out,)
        # Surviving disjuncts: ub_diff_tight >= 0 (could be unsafe)
        # Sort by ub_diff_tight DESC so most-likely-unsafe checked first.
        candidates = [j for j in range(n_out)
                       if j != t and ub_diff_tight[j] >= 0.0]
        candidates.sort(key=lambda j: -ub_diff_tight[j])
        for j in candidates:
            coef = np.zeros(n_out)
            coef[j] = 1.0
            coef[t] = -1.0
            obj_row, obj_const = _row_to_obj_y(coef)
            st, x = _lp_feas_or_minimize(
                prob, obj_row, rhs_threshold=-obj_const,
                sense="maximize", timeout_s=_remaining(),
            )
            if st == "feasible":
                return "feasible", x
            if st == "timeout":
                return "timeout", None
        return "infeasible", None

    if kind == "MARGIN_ROBUST":
        t = int(_to_np_f64(assert_layer.params["y_true"]).reshape(-1)[0])
        m = float(_to_np_f64(assert_layer.params["margin"]).reshape(-1)[0])
        n_out = c_vec.size
        # OR pre-filter (with margin): skip disjuncts where cube UB(y_j-y_t) < -m.
        diffY = np.concatenate([Gc, Gb], axis=1)
        diff_rows = diffY - diffY[t:t+1]
        diff_c = c_vec - c_vec[t]
        ub_diff_tight = diff_c + np.abs(diff_rows).sum(axis=1)
        candidates = [j for j in range(n_out)
                       if j != t and ub_diff_tight[j] >= -m]
        candidates.sort(key=lambda j: -ub_diff_tight[j])
        for j in candidates:
            coef = np.zeros(n_out)
            coef[j] = 1.0
            coef[t] = -1.0
            obj_row, obj_const = _row_to_obj_y(coef)
            st, x = _lp_feas_or_minimize(
                prob, obj_row, rhs_threshold=-m - obj_const,
                sense="maximize", timeout_s=_remaining(),
            )
            if st == "feasible":
                return "feasible", x
            if st == "timeout":
                return "timeout", None
        return "infeasible", None

    if kind == "RANGE":
        n_out = c_vec.size
        lb_spec = assert_layer.params.get("lb")
        ub_spec = assert_layer.params.get("ub")
        # Disjunct i_low: y[i] < lb_spec[i]  →  -y[i] > -lb_spec[i]
        if lb_spec is not None:
            lb_v = _to_np_f64(lb_spec).reshape(-1)
            for i in range(n_out):
                coef = np.zeros(n_out); coef[i] = -1.0
                obj_row, obj_const = _row_to_obj_y(coef)
                st, x = _lp_feas_or_minimize(
                    prob, obj_row, rhs_threshold=-lb_v[i] - obj_const,
                    sense="maximize", timeout_s=_remaining(),
                )
                if st == "feasible":
                    return "feasible", x
                if st == "timeout":
                    return "timeout", None
        if ub_spec is not None:
            ub_v = _to_np_f64(ub_spec).reshape(-1)
            for i in range(n_out):
                coef = np.zeros(n_out); coef[i] = 1.0
                obj_row, obj_const = _row_to_obj_y(coef)
                st, x = _lp_feas_or_minimize(
                    prob, obj_row, rhs_threshold=ub_v[i] - obj_const,
                    sense="maximize", timeout_s=_remaining(),
                )
                if st == "feasible":
                    return "feasible", x
                if st == "timeout":
                    return "timeout", None
        return "infeasible", None

    # Unknown kind ⇒ conservative report.
    return "feasible", None


def lp_witness_to_input(xi_star: np.ndarray, input_hz: HZono) -> np.ndarray:
    """Map a factor-space witness xi_star back to input space.

    For a network whose first HZ corresponds to the input box, the
    "input" coordinates are ``c_in + Gc_in @ xi_c + Gb_in @ xi_b`` for
    the relevant factor slots. If ``input_hz`` is a BoxHZ-style HZono
    (diagonal Gc, no Gb), the first p_in entries of ``xi_star`` are the
    factor coordinates of the input pixels.
    """
    p_in = int(input_hz.Gc.shape[1])
    q_in = int(input_hz.Gb.shape[1])
    xi_c = xi_star[:p_in]
    xi_b = xi_star[p_in: p_in + q_in] if q_in > 0 else np.zeros(0)
    c = _to_np_f64(input_hz.c).reshape(-1)
    Gc = _to_np_f64(input_hz.Gc)
    Gb = _to_np_f64(input_hz.Gb)
    x = c + (Gc @ xi_c if p_in > 0 else 0.0)
    if q_in > 0:
        x = x + Gb @ xi_b
    return x


# Strict witness replay via ONNX runtime
# (inlined from former hz_strict_replay.py)
# ======================================================================

from typing import Any



__all__ = ["strict_replay_for_act"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unwrap_int(x) -> int:
    if torch.is_tensor(x):
        return int(x.item() if x.numel() == 1 else x.flatten()[0].item())
    if hasattr(x, "__len__"):
        return int(x[0])
    return int(x)


def _unwrap_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.item() if x.numel() == 1 else x.flatten()[0].item())
    if hasattr(x, "__len__"):
        return float(x[0])
    return float(x)


def _eval_unsafe_strict(y: np.ndarray, assert_layer) -> bool:
    """Return True iff ``y`` violates the safety spec (i.e. is in the
    unsafe set). ZERO tolerance — exact comparisons.
    """
    kind = assert_layer.params.get("kind")
    kstr = str(kind).split(".")[-1] if kind is not None else ""

    if kstr == "TOP1_ROBUST":
        t = int(_unwrap_int(assert_layer.params["y_true"]))
        return any(y[j] >= y[t] for j in range(len(y)) if j != t)

    if kstr == "MARGIN_ROBUST":
        t = int(_unwrap_int(assert_layer.params["y_true"]))
        m = float(_unwrap_float(assert_layer.params["margin"]))
        return any(y[j] >= y[t] - m for j in range(len(y)) if j != t)

    if kstr == "LINEAR_LE":
        coef = _to_np(assert_layer.params["c"]).reshape(-1)
        d = float(_unwrap_float(assert_layer.params["d"]))
        return float(coef @ y) > d

    if kstr == "UNSAFE_LINEAR":
        C = _to_np(assert_layer.params["c"])
        d_vec = _to_np(assert_layer.params["d"]).reshape(-1)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        return bool(np.all(C @ y <= d_vec))

    if kstr == "RANGE":
        lb_t = assert_layer.params.get("lb")
        ub_t = assert_layer.params.get("ub")
        if lb_t is not None and np.any(y < _to_np(lb_t).reshape(-1)):
            return True
        if ub_t is not None and np.any(y > _to_np(ub_t).reshape(-1)):
            return True
        return False

    return False


def _ort_replay(onnx_path: str, x_t: torch.Tensor, assert_layer) -> bool:
    """Run onnxruntime forward and evaluate unsafe predicate.

    Reshapes ``x_t`` to match the ONNX input shape (overrides batch dim
    to 1) and casts to float32 (ONNX models default precision).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    in_name = sess.get_inputs()[0].name
    in_shape = list(sess.get_inputs()[0].shape)
    in_shape[0] = 1
    x_in = x_t.numpy().reshape(in_shape).astype(np.float32)
    y = sess.run(None, {in_name: x_in})[0].ravel()
    return _eval_unsafe_strict(y, assert_layer)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def strict_replay_for_act(*, net, x_star, assert_layer) -> bool:
    """Strict (zero-tol) witness replay for the ACT verifier.

    Args:
        net: ACT ``Net`` — may carry ``.onnx_path`` for ORT fast path.
        x_star: witness in input space, ``np.ndarray`` of length ``n_in``.
        assert_layer: ACT ASSERT layer carrying spec params.

    Returns:
        ``True`` iff the model's output at ``x_star`` violates the spec
        (i.e. is in the unsafe set). Used to confirm SAT witnesses; if
        False, the LP cert is spurious and the verdict downgrades.
    """
    x_arr = np.asarray(x_star, dtype=np.float64)

    # Path 1: ORT replay (preferred — matches VNN-COMP scorer).
    onnx_path = getattr(net, "onnx_path", None)
    if onnx_path is not None and os.path.exists(onnx_path):
        try:
            x_t = torch.from_numpy(x_arr.astype(np.float32))
            return _ort_replay(onnx_path, x_t, assert_layer)
        except Exception:
            # Fall through to torch path.
            pass

    # Path 2: ACTToTorch conversion + torch forward.
    try:
        from act.pipeline.verification.act2torch import ACTToTorch
        from act.back_end.layer_schema import LayerKind
    except Exception:
        # Pipeline unavailable → can't replay. Sound: reject witness.
        return False

    try:
        torch_model = ACTToTorch(net).run()
        torch_model.eval()
        input_layer = next(
            L for L in net.layers if L.kind == LayerKind.INPUT.value
        )
        in_shape = input_layer.params.get("shape")
        if in_shape is None:
            x_t = torch.from_numpy(x_arr.astype(np.float64)).unsqueeze(0)
        else:
            x_t = torch.from_numpy(x_arr.astype(np.float64)).reshape(in_shape)
        with torch.no_grad():
            y = torch_model(x_t)
            if isinstance(y, dict):
                y = y["output"]
            y_np = y.detach().cpu().numpy().reshape(-1)
        return _eval_unsafe_strict(y_np, assert_layer)
    except Exception:
        return False

# ======================================================================
# Small-dense LP verifier (GlobalTriangleLP)
# (inlined from former hz_smalldense_lp.py)
# ======================================================================

from typing import List, Tuple, Optional


# Lazy imports — only loaded when needed
_onnx = None
_numpy_helper = None
_hp = None


def _lazy_imports():
    global _onnx, _numpy_helper, _hp
    if _onnx is None:
        import onnx as _onnx_mod
        from onnx import numpy_helper as _nh
        import highspy as _hp_mod
        _onnx = _onnx_mod
        _numpy_helper = _nh
        _hp = _hp_mod


# ─── ONNX extraction (Sub-optional, MatMul+Add OR Gemm) ─────────────────────
def extract_layers(onnx_path: Path):
    """Return (sub_const, layers, output_layer):
      sub_const: (n_in,) input shift, OR None if model has no input Sub
      layers:    list of (W, b) for each hidden layer (with ReLU after)
      output_layer: (W, b) for final layer (no ReLU)
                    OR (W, b, skip_W, skip_slice) for linearizenn-style
                    models with an input-slice skip into the output.

    Handles:
      * optional Sub at input
      * optional Flatten
      * MatMul+Add chains and Gemm nodes (both as affine layers)
      * ReLU markers between layers
      * linearizenn-style tail: MatMul-(Slice+MatMul)-Concat-MatMul that
        produces output = main_path + skip_W @ x[slice_range]
    """
    _lazy_imports()
    m = _onnx.load(str(onnx_path))
    inits = {init.name: _numpy_helper.to_array(init) for init in m.graph.initializer}
    nodes = list(m.graph.node)

    sub_const = None
    matmul_b_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    relu_after: List[bool] = []

    # Track per-tensor symbolic linear ops for the post-Gemm-chain tail.
    # Map name -> ('main', last_matmul_b_index)  OR  ('input_slice', start, stop, step)
    #          OR ('linear', (A, b))  meaning value = A @ x_in + b
    # We compose linear maps symbolically through MatMul / Slice / Concat / MatMul.
    pending_W = None
    consumed_outputs = set()
    constants = {}  # name -> np.ndarray for Constant nodes
    main_chain_output = None  # the name of the final Add/Gemm output before tail

    # First pass: build hidden + initial output layer via the Gemm/MatMul+Add pattern
    for n in nodes:
        if n.op_type == "Sub":
            const_name = [i for i in n.input if i in inits][0]
            sub_const = inits[const_name].astype(np.float64).reshape(-1)
        elif n.op_type == "Flatten":
            pass
        elif n.op_type == "MatMul":
            const_inputs = [i for i in n.input if i in inits]
            if const_inputs:
                # Always set pending_W on MatMul. If a subsequent Add+init follows
                # (main-chain MatMul+Add pattern), it gets consumed. If not (tail
                # MatMul before Slice/Concat), pending_W is just left uncommitted.
                pending_W = inits[const_inputs[0]].astype(np.float64)
        elif n.op_type == "Add":
            const_inputs = [i for i in n.input if i in inits]
            if const_inputs and pending_W is not None:
                b = inits[const_inputs[0]].astype(np.float64).reshape(-1)
                matmul_b_pairs.append((pending_W, b))
                relu_after.append(False)
                pending_W = None
                main_chain_output = n.output[0]
        elif n.op_type == "Gemm":
            # Gemm has 2 or 3 inputs: A, B (weight), optional C (bias).
            # When B is an initializer, treat as standard affine: y = alpha*A@B + beta*C
            # ONNX default alpha=1, beta=1, transA=0, transB=0.
            B_name = n.input[1]
            C_name = n.input[2] if len(n.input) >= 3 else None
            if B_name in inits:
                W = inits[B_name].astype(np.float64)
                # Honor transB attribute
                transB = 0
                transA = 0
                alpha = 1.0
                beta = 1.0
                for attr in n.attribute:
                    if attr.name == 'transB': transB = attr.i
                    elif attr.name == 'transA': transA = attr.i
                    elif attr.name == 'alpha': alpha = attr.f
                    elif attr.name == 'beta': beta = attr.f
                if transB:
                    W = W.T
                W = alpha * W
                if C_name and C_name in inits:
                    b = beta * inits[C_name].astype(np.float64).reshape(-1)
                else:
                    b = np.zeros(W.shape[-1], dtype=np.float64)
                matmul_b_pairs.append((W, b))
                relu_after.append(False)
                main_chain_output = n.output[0]
        elif n.op_type == "Relu":
            if relu_after:
                relu_after[-1] = True
        elif n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == 'value':
                    constants[n.output[0]] = _numpy_helper.to_array(attr.t)
        # Slice / Concat / MatMul on non-init handled in tail pass below

    # Compose consecutive non-ReLU affine layers. The composition rule is:
    #   (x @ W1 + b1) @ W2 + b2 = x @ (W1 @ W2) + (b1 @ W2 + b2)
    # The first ReLU after a composed block terminates that hidden layer;
    # any trailing composed affines (no ReLU after) become the output layer.
    composed_layers = []          # list of (W, b)
    composed_has_relu = []         # parallel: True if this layer is followed by ReLU
    cur_W, cur_b = None, None
    for (W, b), has_relu in zip(matmul_b_pairs, relu_after):
        if cur_W is None:
            cur_W, cur_b = W, b
        else:
            # Compose: x @ cur_W + cur_b → (...) @ W + b
            cur_W = cur_W @ W
            cur_b = cur_b @ W + b
        if has_relu:
            composed_layers.append((cur_W, cur_b))
            composed_has_relu.append(True)
            cur_W, cur_b = None, None
    # Trailing affine(s) without ReLU = the output layer
    if cur_W is not None:
        composed_layers.append((cur_W, cur_b))
        composed_has_relu.append(False)

    hidden = [layer for layer, hr in zip(composed_layers, composed_has_relu) if hr]
    out_layers = [layer for layer, hr in zip(composed_layers, composed_has_relu) if not hr]
    if len(out_layers) != 1:
        raise ValueError(
            f"GlobalTriangleLP: expected exactly 1 output (no-ReLU) layer "
            f"after affine composition, got {len(out_layers)}; "
            f"model not supported."
        )
    output_layer = out_layers[0]

    # ─── Tail pass: detect linearizenn-style skip pattern ───────────────────
    # Look for nodes AFTER main_chain_output (= last Gemm/Add output):
    #   MatMul(main_chain_output, W_a)  -> u
    #   Slice(input, ...)               -> x_slice
    #   MatMul(x_slice, W_b)            -> v
    #   Concat([u, v])                  -> c
    #   MatMul(c, W_c)                  -> output
    # If detected, fold all tail linears into:
    #   output_layer = (W_out_main, b_out_main) such that
    #     output = post_L @ W_out_main + b_out_main + x_in[slice] @ W_out_skip
    if main_chain_output is None:
        return sub_const, hidden, output_layer

    # Trace tail: detect and fold linearizenn-style Slice+Concat+MatMul.
    # See _maybe_fold_linearizenn_tail for the symbolic-linear composition.
    return _maybe_fold_linearizenn_tail(sub_const, hidden, output_layer,
                                          nodes, inits, constants, main_chain_output,
                                          m.graph.input[0].name)


def _slice_size(slice_tuple, n_in):
    """Resolve slice (start, end, step) against n_in to a concrete count."""
    if slice_tuple is None: return 0
    start, end, step = slice_tuple
    if end is None or end > n_in: end = n_in
    if start < 0: start = max(0, n_in + start)
    if step <= 0: step = 1
    return max(0, (end - start + step - 1) // step)


def _maybe_fold_linearizenn_tail(sub_const, hidden, output_layer,
                                   nodes, inits, constants, main_chain_output,
                                   input_name):
    """If we find a Slice+Concat+MatMul tail, fold it into a richer output_layer
    that also includes a skip term from input. Returns:
        (sub_const, hidden, (W_main, b_main, W_skip, slice_idx_range))
    where slice_idx_range is a tuple (start, stop, step).
    If no tail, returns (sub_const, hidden, output_layer) as before.
    """
    # Collect tail nodes (those that consume main_chain_output transitively
    # or take input as a side branch via Slice).
    # Symbolic tensors: dict name -> {kind, ...}
    # kind: 'main' — value derived from post_L via affine (A_post, b_post)
    #       'input' — equals the model input tensor x_in
    #       'slice' — slice of input: (start, stop, step)
    #       'linear' — A_post @ post_L + A_skip @ x_in[slice] + b
    # We propagate symbolic values through MatMul/Slice/Concat.

    # The output_layer's affine is: last_pre = post_L @ output_layer[0] + output_layer[1]
    # (where post_L is the previous ReLU's output). But linearizenn has NO ReLU
    # after the final Gemm — so main_chain_output IS this value, treat it as the
    # "starting symbolic" linear value:
    #   main_chain_output = post_L @ W + b
    # For tail tracing, we let post_L be a placeholder; A_post starts as W and
    # b starts as b for main_chain_output.

    W_init, b_init = output_layer  # last Gemm/MatMul+Add weights
    # Determine model input dim for slice resolution
    n_in_full = W_init.shape[0] if sub_const is None else sub_const.shape[0]
    if hidden:
        # First hidden layer's W has shape (n_in, n_h0); first dim is n_in
        n_in_full = hidden[0][0].shape[0]
    # Tail symbolic state per tensor name:
    syms = {}
    # main_chain_output represents post_L @ W_init + b_init
    syms[main_chain_output] = {
        'A_post': W_init.copy(),     # shape (n_L, n_dim)
        'A_skip': None,              # set when skip term enters
        'b': b_init.copy(),          # shape (n_dim,)
        'slice': None,
    }
    syms[input_name] = {
        'A_post': None,
        'A_skip': None,
        'b': None,
        'slice': (0, None, 1),       # full input
        '_is_input': True,
    }

    has_tail = False
    for n in nodes:
        if all((inp in inits) or (inp not in syms and inp not in constants) for inp in n.input):
            continue
        if n.op_type == "Slice":
            # Slice(input, starts, ends, axes, steps) — assume axis=-1 if axes present
            in_name = n.input[0]
            if in_name not in syms:
                continue
            # Read starts/ends/steps from constants
            starts = constants.get(n.input[1], None)
            ends = constants.get(n.input[2], None)
            # axes optional (n.input[3]), steps optional (n.input[4])
            steps = constants.get(n.input[4], None) if len(n.input) >= 5 else None
            if starts is None or ends is None:
                continue
            start = int(starts.flatten()[0])
            end = int(ends.flatten()[0])
            step = int(steps.flatten()[0]) if steps is not None else 1
            # Clamp end to input dim (ONNX uses INT64_MAX for "to end")
            if end > n_in_full or end < 0: end = n_in_full
            if start < 0: start = max(0, n_in_full + start)
            out_name = n.output[0]
            syms[out_name] = {
                'A_post': None,
                'A_skip': None,
                'b': None,
                'slice': (start, end, step),
                '_is_input_slice': True,
            }
            has_tail = True
        elif n.op_type == "MatMul":
            # tail MatMul: tensor @ W
            in0, in1 = n.input[0], n.input[1]
            W = inits[in1].astype(np.float64) if in1 in inits else None
            if W is None or in0 not in syms:
                continue
            sym0 = syms[in0]
            out_name = n.output[0]
            new_sym = {'A_post': None, 'A_skip': None, 'b': None, 'slice': None}
            if sym0.get('_is_input_slice'):
                # value = x_in[slice] @ W
                new_sym['A_skip'] = W.copy()
                new_sym['slice'] = sym0['slice']
                new_sym['b'] = np.zeros(W.shape[-1], dtype=np.float64)
            elif sym0.get('_is_input'):
                new_sym['A_skip'] = W.copy()
                new_sym['slice'] = (0, None, 1)
                new_sym['b'] = np.zeros(W.shape[-1], dtype=np.float64)
            else:
                # sym0 = A_post @ post_L + (A_skip @ x_slice) + b
                # value = sym0 @ W
                if sym0['A_post'] is not None:
                    new_sym['A_post'] = sym0['A_post'] @ W
                if sym0['A_skip'] is not None:
                    new_sym['A_skip'] = sym0['A_skip'] @ W
                    new_sym['slice'] = sym0['slice']
                new_sym['b'] = (sym0['b'] @ W) if sym0['b'] is not None else np.zeros(W.shape[-1], dtype=np.float64)
            syms[out_name] = new_sym
            has_tail = True
        elif n.op_type == "Concat":
            # Concat along last axis. We concatenate (A_post @ post_L + A_skip @ x_slice + b) blocks.
            # All inputs must be symbolic; concat along axis -1.
            in_syms = [syms.get(i) for i in n.input]
            if any(s is None for s in in_syms):
                continue
            # Pad each block's A_post/A_skip/b to vectors; concat along last dim
            blocks_post = []; blocks_skip = []; blocks_b = []
            # Resolve a common slice. For simplicity, require all skip slices match
            # or be None.
            slices = [s['slice'] for s in in_syms if s.get('A_skip') is not None]
            common_slice = slices[0] if slices else None
            sk_n_common = _slice_size(common_slice, n_in_full) if common_slice else 0
            for s in in_syms:
                A_p = s.get('A_post')
                A_s = s.get('A_skip')
                bb = s.get('b')
                dim_out = (A_p.shape[1] if A_p is not None
                            else (A_s.shape[1] if A_s is not None
                                    else (bb.shape[0] if bb is not None else 0)))
                if A_p is None:
                    A_p = np.zeros((W_init.shape[0], dim_out), dtype=np.float64)
                if A_s is None and sk_n_common > 0:
                    A_s = np.zeros((sk_n_common, dim_out), dtype=np.float64)
                if bb is None:
                    bb = np.zeros(dim_out, dtype=np.float64)
                blocks_post.append(A_p)
                if A_s is not None: blocks_skip.append(A_s)
                blocks_b.append(bb)
            new_sym = {
                'A_post': np.concatenate(blocks_post, axis=-1) if blocks_post else None,
                'A_skip': (np.concatenate(blocks_skip, axis=-1) if blocks_skip else None),
                'b': np.concatenate(blocks_b, axis=-1) if blocks_b else None,
                'slice': common_slice,
            }
            syms[n.output[0]] = new_sym
            has_tail = True
        # Other ops (Constant) are tracked in `constants` dict (already populated)

    if not has_tail:
        return sub_const, hidden, output_layer

    # Find the model graph's final output name to get the folded result
    # (the last symbolic tensor whose name matches model output).
    # We assume `output` is the model's output name.
    final_name = 'output'
    if final_name not in syms:
        # fallback: pick last sym
        candidate_outputs = [k for k in syms if k not in ('input', input_name)]
        if not candidate_outputs:
            return sub_const, hidden, output_layer
        final_name = candidate_outputs[-1]
    final_sym = syms[final_name]
    W_main = final_sym['A_post']
    b_main = final_sym['b']
    W_skip = final_sym['A_skip']
    slice_range = final_sym['slice']
    if W_main is None:
        return sub_const, hidden, output_layer
    if W_skip is None:
        return sub_const, hidden, (W_main, b_main)
    return sub_const, hidden, (W_main, b_main, W_skip, slice_range)


def is_small_dense(onnx_path: Path,
                   in_dim_max: int = 32,
                   total_relu_max: int = 500) -> bool:
    """Heuristic: dispatch only on small dense ReLU networks.

    Conditions:
      - No Conv ops in the graph
      - Input dim <= in_dim_max (default 32)
      - Total ReLU count <= total_relu_max (default 500)
    """
    _lazy_imports()
    try:
        m = _onnx.load(str(onnx_path))
    except Exception:
        return False
    has_conv = False
    n_relu = 0
    in_dim = 0
    for n in m.graph.node:
        if n.op_type.startswith("Conv"):
            has_conv = True
        elif n.op_type == "Relu":
            n_relu += 1
    if has_conv:
        return False
    if n_relu == 0 or n_relu > total_relu_max:
        return False
    # Input dim from graph input shape (last dim)
    if m.graph.input:
        try:
            shape = m.graph.input[0].type.tensor_type.shape.dim
            dims = [d.dim_value if d.dim_value > 0 else 1 for d in shape]
            in_dim = int(np.prod(dims))
        except Exception:
            return False
    if in_dim == 0 or in_dim > in_dim_max:
        return False
    return True


# ─── vnnlib parser ──────────────────────────────────────────────────────────
def _split_top_level_groups(s: str, opener: str = '(', closer: str = ')'):
    out = []
    depth = 0; start = -1
    for i, ch in enumerate(s):
        if ch == opener:
            if depth == 0: start = i
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(s[start:i+1])
                start = -1
    return out


def parse_vnnlib(vnn_path: Path, n_in: int, n_out: int):
    """Returns disjuncts: list of (lb_x_disj, ub_x_disj, unsafe_rows).
    Handles top-level X bounds, top-level Y constraints, and arbitrary mix
    of (assert (or (and ...) (and ...))) blocks; final disjuncts are the
    cartesian product over all OR-blocks.
    """
    import re
    raw = open(vnn_path).read()
    lines = []
    for ln in raw.split('\n'):
        i = ln.find(';')
        if i >= 0: ln = ln[:i]
        if ln.strip(): lines.append(ln)
    txt = '\n'.join(lines)

    lb_x_global = np.full(n_in, -np.inf)
    ub_x_global = np.full(n_in,  np.inf)
    for m in re.finditer(r'\(assert \(>= X_(\d+) ([\-\d.eE]+)\)\)', txt):
        lb_x_global[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r'\(assert \(<= X_(\d+) ([\-\d.eE]+)\)\)', txt):
        ub_x_global[int(m.group(1))] = float(m.group(2))

    def parse_y_in(blk):
        disj = []
        for m in re.finditer(r'\(>= Y_(\d+) ([\-\d.eE]+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = -1.0
            disj.append((c, -float(m.group(2))))
        for m in re.finditer(r'\(<= Y_(\d+) ([\-\d.eE]+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = 1.0
            disj.append((c, float(m.group(2))))
        for m in re.finditer(r'\(>= \(\- Y_(\d+) Y_(\d+)\) ([\-\d.eE]+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = -1.0; c[int(m.group(2))] = 1.0
            disj.append((c, -float(m.group(3))))
        for m in re.finditer(r'\(<= \(\- Y_(\d+) Y_(\d+)\) ([\-\d.eE]+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = 1.0; c[int(m.group(2))] = -1.0
            disj.append((c, float(m.group(3))))
        for m in re.finditer(r'\(<= Y_(\d+) Y_(\d+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = 1.0; c[int(m.group(2))] = -1.0
            disj.append((c, 0.0))
        for m in re.finditer(r'\(>= Y_(\d+) Y_(\d+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(2))] = 1.0; c[int(m.group(1))] = -1.0
            disj.append((c, 0.0))
        return disj

    def parse_x_in(blk):
        lb = lb_x_global.copy(); ub = ub_x_global.copy()
        for m in re.finditer(r'\(>= X_(\d+) ([\-\d.eE]+)\)', blk):
            lb[int(m.group(1))] = float(m.group(2))
        for m in re.finditer(r'\(<= X_(\d+) ([\-\d.eE]+)\)', blk):
            ub[int(m.group(1))] = float(m.group(2))
        return lb, ub

    # Extract OR blocks first (we need their spans to compute top_level_y on
    # the text WITHOUT OR contents — otherwise per-disjunct Y rows get
    # duplicated as "top-level" too).
    or_block_alternatives = []
    or_block_spans = []  # (start, end) spans to mask out for top-level parse
    for m in re.finditer(r'\(assert \(or\b', txt):
        start = m.start()
        depth = 0; end = -1
        for i in range(start, len(txt)):
            if txt[i] == '(': depth += 1
            elif txt[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1; break
        if end < 0: continue
        assert_block = txt[start:end]
        body_match = re.match(r'\(assert \(or\s+(.+)\)\s*\)', assert_block, re.DOTALL)
        if not body_match: continue
        body = body_match.group(1)
        ands = _split_top_level_groups(body)
        if ands:
            or_block_alternatives.append(ands)
            or_block_spans.append((start, end))

    # Build a text without OR-block contents for top-level Y extraction
    if or_block_spans:
        pieces = []
        cursor = 0
        for s, e in sorted(or_block_spans):
            pieces.append(txt[cursor:s])
            cursor = e
        pieces.append(txt[cursor:])
        txt_without_or = ''.join(pieces)
    else:
        txt_without_or = txt
    top_level_y = parse_y_in(txt_without_or)

    disjuncts = []
    if not or_block_alternatives:
        if top_level_y:
            disjuncts.append((lb_x_global, ub_x_global, top_level_y))
    else:
        from itertools import product
        for combo in product(*or_block_alternatives):
            lb_d = lb_x_global.copy(); ub_d = ub_x_global.copy()
            y_rows = list(top_level_y)
            for and_block in combo:
                lb_blk, ub_blk = parse_x_in(and_block)
                lb_d = np.maximum(lb_d, lb_blk)
                ub_d = np.minimum(ub_d, ub_blk)
                y_rows.extend(parse_y_in(and_block))
            if y_rows:
                disjuncts.append((lb_d, ub_d, y_rows))
    return disjuncts


# ─── LP build + solve ───────────────────────────────────────────────────────
def _solve_one_obj(h, nvars, obj_coefs, sense='min', time_limit=10):
    """Set objective and solve. obj_coefs: dict col→coef."""
    obj = [0.0] * nvars
    for c, v in obj_coefs.items():
        obj[c] = float(v) if sense == 'min' else -float(v)
    h.changeColsCost(nvars, np.arange(nvars, dtype=np.int32), obj)
    h.run()
    sm = h.getModelStatus()
    if sm == _hp.HighsModelStatus.kOptimal:
        val = h.getObjectiveValue()
        return ('ok', val if sense == 'min' else -val)
    if sm == _hp.HighsModelStatus.kInfeasible:
        return ('infeasible', None)
    return ('fail', None)


def output_affine_rows(output_layer, n_li, n_in, n_out,
                         s_post: int, s_xin: int, s_y: int):
    """Build the LP rows for output y = post_L @ W_main + b_main [+ x_in[slc] @ W_skip].

    Handles both 2-tuple `(W_main, b_main)` and 4-tuple
    `(W_main, b_main, W_skip, slice_range)` output_layer formats.

    Each row encodes: y[j] - Σ wjk·post_L[k] - Σ wjm·x_in[m] = b_main[j]
    """
    if len(output_layer) == 2:
        W_main, b_main = output_layer
        W_skip = None; slc = None
    else:
        W_main, b_main, W_skip, slc = output_layer
    in_to_out_main = (W_main.shape[0] == n_li)
    rows = []
    # Resolve skip indices
    skip_idx = None
    if W_skip is not None and slc is not None:
        s, e, step = slc
        if e is None or e > n_in: e = n_in
        skip_idx = list(range(s, e, step))
        in_to_out_skip = (W_skip.shape[0] == len(skip_idx))
    for j in range(n_out):
        coefs = {s_y + j: 1.0}
        for k in range(n_li):
            wkj = W_main[k, j] if in_to_out_main else W_main[j, k]
            if wkj != 0:
                coefs[s_post + k] = -wkj
        if W_skip is not None:
            for pos, i_in in enumerate(skip_idx):
                wkj = W_skip[pos, j] if in_to_out_skip else W_skip[j, pos]
                if wkj != 0:
                    coefs[s_xin + i_in] = coefs.get(s_xin + i_in, 0.0) - wkj
        rows.append(('eq', float(b_main[j]), list(coefs.items())))
    return rows


def _add_rows_to_lp(h, rows_data):
    """rows_data: list of (sense, rhs, [(col, val), ...])"""
    for sense, rhs, entries in rows_data:
        cols = np.array([e[0] for e in entries], dtype=np.int32)
        vals = np.array([e[1] for e in entries], dtype=np.float64)
        if sense == 'le':
            h.addRow(-_hp.kHighsInf, rhs, len(cols), cols, vals)
        elif sense == 'ge':
            h.addRow(rhs, _hp.kHighsInf, len(cols), cols, vals)
        else:
            h.addRow(rhs, rhs, len(cols), cols, vals)


def _verify_one_disjunct(sub_const, layers, output_layer,
                          lb_x, ub_x, unsafe_rows, time_limit_per_lp=5):
    n_in = lb_x.shape[0]
    n_out = output_layer[1].shape[0]
    n_l = [b.shape[0] for W, b in layers]
    var_offsets = {}
    cur = 0
    var_offsets['x_in'] = (cur, cur + n_in); cur += n_in
    for li in range(len(layers)):
        var_offsets[f'pre_{li}'] = (cur, cur + n_l[li]); cur += n_l[li]
        var_offsets[f'post_{li}'] = (cur, cur + n_l[li]); cur += n_l[li]
    var_offsets['y'] = (cur, cur + n_out); cur += n_out
    nvars = cur

    lb_arr = np.full(nvars, -_hp.kHighsInf)
    ub_arr = np.full(nvars,  _hp.kHighsInf)
    s, e = var_offsets['x_in']
    lb_arr[s:e] = lb_x; ub_arr[s:e] = ub_x

    h = _hp.Highs()
    h.silent()
    h.setOptionValue("time_limit", float(time_limit_per_lp))
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", "simplex")
    lp = _hp.HighsLp()
    lp.num_col_ = nvars
    lp.num_row_ = 0
    lp.col_cost_ = [0.0] * nvars
    lp.col_lower_ = lb_arr.tolist()
    lp.col_upper_ = ub_arr.tolist()
    lp.row_lower_ = []
    lp.row_upper_ = []
    lp.a_matrix_.format_ = _hp.MatrixFormat.kColwise
    lp.a_matrix_.start_ = [0] * (nvars + 1)
    lp.a_matrix_.index_ = []
    lp.a_matrix_.value_ = []
    h.passModel(lp)

    # Layer 0 affine: pre_0 = W_0 @ (x_in - sub_const) + b_0
    W0, b0 = layers[0]
    in_to_out = (W0.shape[0] == n_in)
    s_pre0, _ = var_offsets['pre_0']
    s_xin, _ = var_offsets['x_in']
    if sub_const is not None:
        b_eff0 = b0 - (sub_const @ W0 if in_to_out else W0 @ sub_const)
    else:
        b_eff0 = b0
    rows = []
    for j in range(n_l[0]):
        coefs = {s_pre0 + j: 1.0}
        for k in range(n_in):
            wkj = W0[k, j] if in_to_out else W0[j, k]
            if wkj != 0:
                coefs[s_xin + k] = -wkj
        rows.append(('eq', float(b_eff0[j]),
                     [(c, v) for c, v in coefs.items()]))
    _add_rows_to_lp(h, rows)

    layer_pre_bounds = []
    for li in range(len(layers)):
        n_li = n_l[li]
        s_pre, _ = var_offsets[f'pre_{li}']
        lb_pre = np.zeros(n_li); ub_pre = np.zeros(n_li)
        for j in range(n_li):
            st_lb, lb_j = _solve_one_obj(h, nvars, {s_pre + j: 1.0}, sense='min',
                                          time_limit=time_limit_per_lp)
            st_ub, ub_j = _solve_one_obj(h, nvars, {s_pre + j: 1.0}, sense='max',
                                          time_limit=time_limit_per_lp)
            if st_lb != 'ok' or st_ub != 'ok':
                return f'fail(lp_bound_l{li}_n{j}_{st_lb}/{st_ub})'
            lb_pre[j] = lb_j; ub_pre[j] = ub_j
        layer_pre_bounds.append((lb_pre, ub_pre))

        s_post, _ = var_offsets[f'post_{li}']
        new_lb = np.maximum(0, lb_pre)
        new_ub = np.maximum(0, ub_pre)
        for j in range(n_li):
            h.changeColBounds(s_post + j, float(new_lb[j]), float(new_ub[j]))
        rows = []
        for j in range(n_li):
            l, u = float(lb_pre[j]), float(ub_pre[j])
            if l >= 0:
                rows.append(('eq', 0.0, [(s_post + j, 1.0), (s_pre + j, -1.0)]))
            elif u <= 0:
                rows.append(('eq', 0.0, [(s_post + j, 1.0)]))
            else:
                rows.append(('le', 0.0, [(s_pre + j, 1.0), (s_post + j, -1.0)]))
                lam = u / (u - l)
                rhs = -lam * l
                rows.append(('le', float(rhs),
                             [(s_post + j, 1.0), (s_pre + j, -lam)]))
        _add_rows_to_lp(h, rows)

        if li + 1 < len(layers):
            W_next, b_next = layers[li + 1]
            n_next = n_l[li + 1]
            in_to_out_n = (W_next.shape[0] == n_li)
            s_pre_next, _ = var_offsets[f'pre_{li + 1}']
            rows = []
            for j in range(n_next):
                coefs = {s_pre_next + j: 1.0}
                for k in range(n_li):
                    wkj = W_next[k, j] if in_to_out_n else W_next[j, k]
                    if wkj != 0:
                        coefs[s_post + k] = -wkj
                rows.append(('eq', float(b_next[j]),
                             [(c, v) for c, v in coefs.items()]))
            _add_rows_to_lp(h, rows)
        else:
            s_y, _ = var_offsets['y']
            s_xin_eff, _ = var_offsets['x_in']
            rows = output_affine_rows(output_layer, n_li, n_in, n_out,
                                        s_post=s_post, s_xin=s_xin_eff, s_y=s_y)
            _add_rows_to_lp(h, rows)

    s_y, _ = var_offsets['y']
    for c_vec, d in unsafe_rows:
        entries = [(s_y + k, float(c_vec[k])) for k in range(n_out) if c_vec[k] != 0]
        cols = np.array([e[0] for e in entries], dtype=np.int32)
        vals = np.array([e[1] for e in entries], dtype=np.float64)
        h.addRow(-_hp.kHighsInf, float(d), len(cols), cols, vals)
    h.changeColsCost(nvars, np.arange(nvars, dtype=np.int32), [0.0] * nvars)
    h.run()
    sm = h.getModelStatus()
    if sm == _hp.HighsModelStatus.kOptimal:
        return 'unknown'
    if sm == _hp.HighsModelStatus.kInfeasible:
        return 'verified'
    return f'fail(spec_lp_{str(sm)})'


# ─── Public API ─────────────────────────────────────────────────────────────
def verify(onnx_path, vnnlib_path, time_limit_per_lp: float = 5.0):
    """Verify a single (onnx, vnnlib) instance via global triangle LP.

    Returns ('verified'|'unknown'|'fail(...)', elapsed_s).

    Sound: LP relaxation contains the integer-feasible set; LP-infeasible
    disjunct ⇒ unsafe set empty for that disjunct; verified iff all disjuncts
    infeasible. Cannot output 'falsified' (no witness extraction in this
    version; a feasible LP does not prove a counterexample exists).
    """
    _lazy_imports()
    t0 = time.time()
    onnx_path = Path(onnx_path); vnnlib_path = Path(vnnlib_path)
    try:
        sub_const, layers, output_layer = extract_layers(onnx_path)
    except Exception as e:
        return (f'fail(extract:{type(e).__name__})', time.time() - t0)
    n_in = layers[0][0].shape[0] if (sub_const is None) else sub_const.shape[0]
    if sub_const is None:
        # Infer n_in from layer-0 weight orientation
        W0 = layers[0][0]; b0_dim = layers[0][1].shape[0]
        n_in = W0.shape[0] if W0.shape[1] == b0_dim else W0.shape[1]
    n_out = output_layer[1].shape[0]
    disjuncts = parse_vnnlib(vnnlib_path, n_in, n_out)
    if not disjuncts:
        return ('fail(no_disjuncts)', time.time() - t0)
    for lb_d, ub_d, _ in disjuncts:
        if not (np.isfinite(lb_d).all() and np.isfinite(ub_d).all()):
            return ('fail(unbounded_input)', time.time() - t0)
    for lb_d, ub_d, unsafe_rows in disjuncts:
        st = _verify_one_disjunct(sub_const, layers, output_layer,
                                    lb_d, ub_d, unsafe_rows,
                                    time_limit_per_lp=time_limit_per_lp)
        if st == 'unknown':
            return ('unknown', time.time() - t0)
        if st.startswith('fail'):
            return (st, time.time() - t0)
    return ('verified', time.time() - t0)

# ======================================================================
# Small-dense witness extractor (SpecAware)
# (inlined from former hz_smalldense_witness.py)
# ======================================================================

from typing import List, Tuple, Optional
# extract_layers, parse_vnnlib, _lazy_imports, _add_rows_to_lp, _solve_one_obj
# are all defined earlier in this module (the "Small-dense LP verifier" section).


def _build_and_refine_lp(sub_const, layers, output_layer, lb_x, ub_x, unsafe_rows,
                           time_limit_per_lp=15.0, max_refinement_passes=3):
    """Build the full SpecAware LP (including refinement) and return the
    HiGHS handle + variable offsets, ready for objective changes."""
    _lazy_imports()
    hp = _hp
    n_in = lb_x.shape[0]
    n_out = output_layer[1].shape[0]
    n_l = [b.shape[0] for W, b in layers]
    var_offsets = {}; cur = 0
    var_offsets['x_in'] = (cur, cur + n_in); cur += n_in
    for li in range(len(layers)):
        var_offsets[f'pre_{li}'] = (cur, cur + n_l[li]); cur += n_l[li]
        var_offsets[f'post_{li}'] = (cur, cur + n_l[li]); cur += n_l[li]
    var_offsets['y'] = (cur, cur + n_out); cur += n_out
    nvars = cur

    lb_arr = np.full(nvars, -hp.kHighsInf); ub_arr = np.full(nvars, hp.kHighsInf)
    s, e = var_offsets['x_in']; lb_arr[s:e] = lb_x; ub_arr[s:e] = ub_x

    h = hp.Highs(); h.silent()
    h.setOptionValue("time_limit", float(time_limit_per_lp))
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", "simplex")
    lp = hp.HighsLp(); lp.num_col_ = nvars; lp.num_row_ = 0
    lp.col_cost_ = [0.0] * nvars
    lp.col_lower_ = lb_arr.tolist(); lp.col_upper_ = ub_arr.tolist()
    lp.row_lower_ = []; lp.row_upper_ = []
    lp.a_matrix_.format_ = hp.MatrixFormat.kColwise
    lp.a_matrix_.start_ = [0] * (nvars + 1)
    lp.a_matrix_.index_ = []; lp.a_matrix_.value_ = []
    h.passModel(lp)

    # Layer 0 affine
    W0, b0 = layers[0]
    in_to_out = (W0.shape[0] == n_in)
    s_pre0, _ = var_offsets['pre_0']; s_xin, _ = var_offsets['x_in']
    if sub_const is not None:
        b_eff0 = b0 - (sub_const @ W0 if in_to_out else W0 @ sub_const)
    else:
        b_eff0 = b0
    rows = []
    for j in range(n_l[0]):
        coefs = {s_pre0 + j: 1.0}
        for k in range(n_in):
            wkj = W0[k, j] if in_to_out else W0[j, k]
            if wkj != 0: coefs[s_xin + k] = -wkj
        rows.append(('eq', float(b_eff0[j]), list(coefs.items())))
    _add_rows_to_lp(h, rows)

    # Build hidden layers with LP-tight bounds + triangles
    layer_pre_bounds = []
    for li in range(len(layers)):
        n_li = n_l[li]
        s_pre, _ = var_offsets[f'pre_{li}']; s_post, _ = var_offsets[f'post_{li}']
        lb_pre = np.zeros(n_li); ub_pre = np.zeros(n_li)
        for j in range(n_li):
            st_lb, lb_j = _solve_one_obj(h, nvars, {s_pre + j: 1.0}, sense='min',
                                           time_limit=time_limit_per_lp)
            st_ub, ub_j = _solve_one_obj(h, nvars, {s_pre + j: 1.0}, sense='max',
                                           time_limit=time_limit_per_lp)
            if st_lb != 'ok' or st_ub != 'ok':
                return None  # fail
            lb_pre[j] = lb_j; ub_pre[j] = ub_j
        layer_pre_bounds.append((lb_pre, ub_pre))

        new_lb_post = np.maximum(0, lb_pre); new_ub_post = np.maximum(0, ub_pre)
        for j in range(n_li):
            h.changeColBounds(s_post + j, float(new_lb_post[j]), float(new_ub_post[j]))
        rows = []
        for j in range(n_li):
            l, u = float(lb_pre[j]), float(ub_pre[j])
            if l >= 0:
                rows.append(('eq', 0.0, [(s_post + j, 1.0), (s_pre + j, -1.0)]))
            elif u <= 0:
                rows.append(('eq', 0.0, [(s_post + j, 1.0)]))
            else:
                rows.append(('le', 0.0, [(s_pre + j, 1.0), (s_post + j, -1.0)]))
                lam = u / (u - l); rhs = -lam * l
                rows.append(('le', float(rhs), [(s_post + j, 1.0), (s_pre + j, -lam)]))
        _add_rows_to_lp(h, rows)

        if li + 1 < len(layers):
            W_n, b_n = layers[li + 1]; n_next = n_l[li + 1]
            in_to_out_n = (W_n.shape[0] == n_li)
            s_pre_n, _ = var_offsets[f'pre_{li + 1}']
            rows_n = []
            for j in range(n_next):
                coefs = {s_pre_n + j: 1.0}
                for k in range(n_li):
                    wkj = W_n[k, j] if in_to_out_n else W_n[j, k]
                    if wkj != 0: coefs[s_post + k] = -wkj
                rows_n.append(('eq', float(b_n[j]), list(coefs.items())))
            _add_rows_to_lp(h, rows_n)
        else:
            s_y, _ = var_offsets['y']
            s_xin_eff, _ = var_offsets['x_in']
            rows_y = output_affine_rows(output_layer, n_li, n_in, n_out,
                                          s_post=s_post, s_xin=s_xin_eff, s_y=s_y)
            _add_rows_to_lp(h, rows_y)

    # Spec rows (track first index so we can change row bounds for max-margin)
    s_y, _ = var_offsets['y']
    spec_first_row = h.getNumRow()
    for c_vec, d in unsafe_rows:
        entries = [(s_y + k, float(c_vec[k])) for k in range(n_out) if c_vec[k] != 0]
        cols = np.array([e[0] for e in entries], dtype=np.int32)
        vals = np.array([e[1] for e in entries], dtype=np.float64)
        h.addRow(-hp.kHighsInf, float(d), len(cols), cols, vals)

    return (h, var_offsets, nvars, layer_pre_bounds, spec_first_row, n_in, n_out)


def _ort_replay(onnx_path: Path, x_in: np.ndarray, unsafe_rows):
    """Run ORT inference, return (y, is_real_counterex)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    in_meta = sess.get_inputs()[0]
    # ONNX shapes can contain symbolic dims (str) or unknown (0). Use 1 for both.
    in_shape = tuple(d if (isinstance(d, int) and d > 0) else 1 for d in in_meta.shape)
    x_arr = None
    y = None
    # Try the declared shape first, then common fallbacks
    candidate_shapes = [in_shape, (1, len(x_in)), (1, len(x_in), 1, 1), (len(x_in),), (1, 1, 1, len(x_in))]
    for shp in candidate_shapes:
        try:
            x_arr = np.asarray(x_in, dtype=np.float32).reshape(shp)
            y = sess.run(None, {in_meta.name: x_arr})[0].flatten()
            break
        except Exception:
            continue
    if y is None:
        return None, False
    # All unsafe rows must hold for a true counterexample
    eps = 1e-6
    for c_vec, d in unsafe_rows:
        if float(np.dot(c_vec, y)) > d + eps:
            return y, False
    return y, True


def try_falsify_disjunct(onnx_path, sub_const, layers, output_layer,
                          lb_x, ub_x, unsafe_rows,
                          time_limit_per_lp=15.0):
    """Run S1/S2/S3 in turn; return ('falsified', x, y) on first success, else ('unknown', None, None)."""
    _lazy_imports(); hp = _hp

    built = _build_and_refine_lp(sub_const, layers, output_layer, lb_x, ub_x, unsafe_rows,
                                   time_limit_per_lp=time_limit_per_lp)
    if built is None:
        return ('unknown', None, None)
    h, var_offsets, nvars, layer_pre_bounds, spec_first_row, n_in, n_out = built

    # Strategy S1: solve as feasibility (zero obj), grab x_in
    h.changeColsCost(nvars, np.arange(nvars, dtype=np.int32), [0.0] * nvars)
    h.run()
    sm = h.getModelStatus()
    if sm == hp.HighsModelStatus.kInfeasible:
        # Verified already — shouldn't happen if we were called on an unknown
        return ('verified', None, None)
    if sm != hp.HighsModelStatus.kOptimal:
        return ('unknown', None, None)
    sol = h.getSolution()
    cv = np.asarray(sol.col_value, dtype=np.float64)
    s_xin, e_xin = var_offsets['x_in']
    x1 = cv[s_xin:e_xin].copy()
    y_ort, real = _ort_replay(Path(onnx_path), x1, unsafe_rows)
    if real:
        return ('falsified', x1, y_ort)

    # Strategy S2: per-row maximize the LHS of each unsafe row, see if any
    # individual maximizer corresponds to a real counterexample
    s_y, _ = var_offsets['y']
    for c_vec, d in unsafe_rows:
        obj = {}
        for k in range(n_out):
            if c_vec[k] != 0:
                obj[s_y + k] = float(c_vec[k])
        st, val = _solve_one_obj(h, nvars, obj, sense='max', time_limit=time_limit_per_lp)
        if st != 'ok':
            continue
        sol = h.getSolution()
        cv = np.asarray(sol.col_value, dtype=np.float64)
        x_cand = cv[s_xin:e_xin].copy()
        y_ort, real = _ort_replay(Path(onnx_path), x_cand, unsafe_rows)
        if real:
            return ('falsified', x_cand, y_ort)

    # Strategy S3: small random perturbations around each LP candidate
    rng = np.random.default_rng(0)
    for _ in range(20):
        # Perturb x1 within box
        delta = rng.uniform(-1, 1, size=len(x1)) * 1e-3 * (ub_x - lb_x)
        x_pert = np.clip(x1 + delta, lb_x, ub_x)
        y_ort, real = _ort_replay(Path(onnx_path), x_pert, unsafe_rows)
        if real:
            return ('falsified', x_pert, y_ort)

    return ('unknown', None, None)


def verify_with_falsification(onnx_path, vnnlib_path,
                                time_limit_per_lp: float = 15.0,
                                max_refinement_passes: int = 3,
                                return_witness: bool = False):
    """SpecAwareLP then witness extraction on unknowns.

    Returns ('verified' | 'falsified' | 'unknown', elapsed_s) by default.
    If return_witness=True, returns
        ('falsified', x_witness, y_ort, elapsed_s) on falsification
        ('verified' | 'unknown', None, None, elapsed_s) otherwise.
    """
    import SpecAwareLP as sa
    _lazy_imports()
    t0 = time.time()
    op = Path(onnx_path); vp = Path(vnnlib_path)

    # First try standard SpecAware verification
    st, sa_elapsed = sa.verify(op, vp,
                                 time_limit_per_lp=time_limit_per_lp,
                                 max_refinement_passes=max_refinement_passes)
    if st == 'verified':
        if return_witness:
            return ('verified', None, None, time.time() - t0)
        return ('verified', time.time() - t0)

    # Unknown — try to falsify each disjunct
    sub_const, layers, output_layer = extract_layers(op)
    if sub_const is None:
        W0 = layers[0][0]; b0_dim = layers[0][1].shape[0]
        n_in = W0.shape[0] if W0.shape[1] == b0_dim else W0.shape[1]
    else:
        n_in = sub_const.shape[0]
    n_out = output_layer[1].shape[0]
    disjuncts = parse_vnnlib(vp, n_in, n_out)
    for lb_d, ub_d, unsafe_rows in disjuncts:
        if not (np.isfinite(lb_d).all() and np.isfinite(ub_d).all()):
            continue
        st2, x_w, y_w = try_falsify_disjunct(op, sub_const, layers, output_layer,
                                                lb_d, ub_d, unsafe_rows,
                                                time_limit_per_lp=time_limit_per_lp)
        if st2 == 'falsified':
            if return_witness:
                return ('falsified', x_w, y_w, time.time() - t0)
            return ('falsified', time.time() - t0)
    if return_witness:
        return ('unknown', None, None, time.time() - t0)
    return ('unknown', time.time() - t0)