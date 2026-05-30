from __future__ import annotations

import logging
import importlib
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
    constraint_keep_weight: float = 0.0,
    tol: float = 1e-12,
) -> HZono:
    """Reduce HZono constraint / generator count. Faithful port of HyZor
    ``HybridZonotope.reduce_constraints`` (HybridZonotope.py:4399). Sound
    over-approximation.

    Phases (subset; the full HyZor port has 5 phases):
      P1: remove trivially redundant constraint rows  (exact)
      P2: remove zero generator columns               (exact)
      P5: continuous generator order reduction (Girard) — when a sound
          independent-slack representation fits ``ng_budget``, reduce to
          that budget with eq-row split + widening. Otherwise retain the
          larger HZ rather than introduce unsound shared slack.

    Additional exact/precision-preserving passes implement equality-only
    row rank removal, parallel row handling, and parallel generator merging.
    Phases skipped (parity-deferred): 3 (nc-budget topk), 4 (nb relaxation).
    The skipped phases are sound to omit and may leave a larger HZ.
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
    # eq_mask=None ⇒ all rows are equalities (matches HZono dataclass docstring
    # at line 77). Previous code used _t.zeros (= all inequalities), which
    # contradicted the docstring and could drop equality rows in Phase 1
    # trivial-row removal — a latent soundness bug.
    eq_m = (
        hz.eq_mask.clone()
        if (hz.eq_mask is not None and int(hz.eq_mask.numel()) == nc)
        else _t.ones(nc, dtype=_t.bool, device=device)
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

    # ---- Phase 1.3: QR rank-revealing EQUALITY-row removal ----
    # A dependent equality can be removed without losing a useful tight
    # inequality: if its RHS is inconsistent, removing it is a sound
    # widening of an empty set; otherwise it is redundant. Applying the
    # same row-rank rule to inequalities is still sound, but needlessly
    # destroys precision. For example, xi <= 0.8 and xi <= 0.2 have
    # rank-one LHS rows, yet removing the tighter row enlarges the factor
    # set. Inequalities are preserved here and may only be deduplicated by
    # the direction-and-RHS-aware Phase 1.5 below.
    n_eq_now = int(eq_m.sum().item())
    _PHASE_1_3_CAP = 1024  # max nc for QR phase
    if 1 < n_eq_now <= _PHASE_1_3_CAP:
        try:
            eq_idx = _t.where(eq_m)[0]
            M = _t.cat([Ac[eq_idx], Ab[eq_idx]], dim=1).to(dtype=_t.float64)
            Q, R = _t.linalg.qr(M.T)  # QR on transpose for row-rank.
            diag = R.diag().abs()
            if diag.numel() > 0:
                rank_tol = max(tol, 1e-10) * max(1.0, float(diag[0]))
            else:
                rank_tol = tol
            rank = int((diag > rank_tol).sum().item())
            if rank < n_eq_now:
                import scipy.linalg as _sla
                _Q, _R, _piv = _sla.qr(
                    M.T.cpu().numpy(), pivoting=True, mode="economic"
                )
                keep_eq_local = _t.tensor(
                    sorted(_piv[:rank].tolist()), device=device, dtype=_t.long
                )
                keep_mask = ~eq_m
                keep_mask[eq_idx[keep_eq_local]] = True
                keep_t = _t.where(keep_mask)[0]
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
    # Sound: collapses removed Gc cols into n INDEPENDENT diagonal slack
    # generators (one per output dim), NOT a single shared box column.
    # The shared-column version introduces artificial cross-output
    # correlation and strictly shrinks the set in opposing directions
    # (test: 2D Gc=I, ng_budget=1 → support in (1,-1) goes 2.0 → 0.0).
    #
    # Skip guard: diagonal slack adds n_dim cols. If n_dim >= ng_budget
    # the cap is unachievable soundly, AND at large n the dense diag
    # is O(n^2) (n=25088 → ~5 GiB). Skip reduction in that regime; the
    # HZ passes through unchanged. The cap was meant to limit growth
    # past the budget, not to force shrinkage beyond what's representable.
    ng_now = int(Gc.shape[1])
    n_dim = int(Gc.shape[0])
    if (ng_budget is not None and ng_now > ng_budget and ng_budget >= 1
            and n_dim < ng_budget):
        nc_now = int(Ac.shape[0])
        scores = Gc.abs().sum(dim=0)
        if nc_now > 0 and constraint_keep_weight > 0.0:
            # A column with a small current output coefficient can still be
            # carrying a strong hull inequality. Dropping it widens the RHS
            # by |Ac[:, col]| and immediately discards that tightening.
            # Reweighting only chooses which columns remain exact; Girard's
            # independent-slack widening below remains unchanged and sound.
            ac_scores = Ac.abs().sum(dim=0)
            if bool((ac_scores > tol).any().item()):
                output_scale = scores.max().clamp(min=tol)
                ac_scale = ac_scores.max().clamp(min=tol)
                scores = scores + (
                    float(constraint_keep_weight) * output_scale
                    * ac_scores / ac_scale
                )
        keep_k = max(ng_budget - n_dim, 0)  # reserve n slots for diagonal slack
        _, topk_idx = _t.topk(scores, min(keep_k, ng_now))
        topk_idx, _ = topk_idx.sort()
        remove_mask = _t.ones(ng_now, dtype=_t.bool, device=device)
        remove_mask[topk_idx] = False
        removed_Gc = Gc[:, remove_mask]
        # Diagonal slack: per-output independent box generators.
        box_widths = removed_Gc.abs().sum(dim=1)  # (n,)
        box_diag = _t.diag(box_widths)            # (n, n)
        Gc = _t.cat([Gc[:, topk_idx], box_diag], dim=1)  # (n, keep_k + n)

        if nc_now > 0:
            old_Ac_keep = Ac[:, topk_idx]
            removed_Ac = Ac[:, remove_mask]
            widen = removed_Ac.abs().sum(dim=1, keepdim=True)
            # The n new diagonal box gens are independent of any prior
            # factor → zero coefficient in all existing constraint rows.
            Ac_box = _t.zeros(nc_now, n_dim, device=device, dtype=dtype)
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
                up_Ac = _t.cat([eAc, _t.zeros(n_eq, n_dim, device=device, dtype=dtype)], dim=1)
                up_b = eb + ewiden
                lo_Ac = _t.cat([-eAc, _t.zeros(n_eq, n_dim, device=device, dtype=dtype)], dim=1)
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
                                    nb_budget=None, constraint_keep_weight=0.0,
                                    tol=1e-12, verbose=False):
    return _hz_reduce_constraints(
        self, nc_budget=nc_budget, ng_budget=ng_budget,
        nb_budget=nb_budget, constraint_keep_weight=constraint_keep_weight,
        tol=tol,
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
    rad = ((ub - lb) / 2.0).clamp_min(0)
    active = torch.nonzero(rad > 0, as_tuple=False).view(-1)
    ng = int(active.numel())
    if ng == n:
        Gc = torch.diag(rad)
    else:
        Gc = torch.zeros((n, ng), dtype=dtype, device=device)
        if ng > 0:
            cols = torch.arange(ng, dtype=torch.long, device=device)
            Gc[active, cols] = rad[active]
    return HZono(
        c=c,
        Gc=Gc,
        Gb=torch.zeros((n, 0), dtype=dtype, device=device),
        Ac=torch.zeros((0, ng), dtype=dtype, device=device),
        Ab=torch.zeros((0, 0), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
        eq_mask=None,
    )


def _zero_factor_hz_feasible(hz: HZono, *, tol: float = 1e-9) -> bool:
    """Feasibility of an HZ with no continuous/binary factors.

    With ng=nb=0 the factor-space has exactly one assignment. Equality rows
    require 0 == b; inequality rows require 0 <= b. This is an exact check
    and is used only for singleton-input fast paths.
    """
    if int(hz.ng) != 0 or int(hz.nb) != 0:
        return False
    if int(hz.nc) == 0:
        return True
    b_flat = hz.b.reshape(-1)
    em = _eq_mask_of(hz).reshape(-1)
    eq_ok = bool((torch.abs(b_flat[em]) <= tol).all().item()) if bool(em.any().item()) else True
    le = ~em
    le_ok = bool((b_flat[le] >= -tol).all().item()) if bool(le.any().item()) else True
    return eq_ok and le_ok


def _hz_upsample_nearest_nchw(hz: HZono, params: Dict[str, Any]) -> HZono:
    """Exact HZ transfer for nearest-neighbor NCHW upsample.

    Nearest-neighbor upsample is a linear row-selection/duplication map.  It
    must preserve the existing factor space instead of materializing an
    interval box; otherwise a 3-generator latent cGAN input becomes thousands
    of independent pixel generators after the first resize.
    """
    mode = str(params.get("mode", "nearest")).lower()
    if mode != "nearest":
        raise ValueError(f"unsupported UPSAMPLE mode for HZ exact path: {mode}")
    if not isinstance(hz, HZono):
        if hasattr(hz, "materialization_bytes") and hasattr(hz, "to_full_hzono"):
            budget = int(
                float(os.environ.get("HYZOR_UPSAMPLE_MAT_BUDGET_GB", "1.0"))
                * (1024 ** 3)
            )
            if int(hz.materialization_bytes()) > budget:
                raise ValueError(
                    "UPSAMPLE LazyChain materialization would exceed budget: "
                    f"{int(hz.materialization_bytes())} bytes"
                )
            hz = hz.to_full_hzono()
        elif hasattr(hz, "to_hzono"):
            hz = hz.to_hzono()
        else:
            raise ValueError(
                f"UPSAMPLE exact path cannot coerce {type(hz).__name__} to HZono"
            )
    in_shape = tuple(int(x) for x in params.get("input_shape", ()))
    out_shape = tuple(int(x) for x in params.get("output_shape", ()))
    if len(in_shape) != 4 or len(out_shape) != 4:
        raise ValueError("UPSAMPLE exact path currently supports NCHW rank-4 only")
    n, c, h, w = in_shape
    n2, c2, h2, w2 = out_shape
    if n != n2 or c != c2:
        raise ValueError("UPSAMPLE exact path requires unchanged N and C")
    if int(hz.dim) != n * c * h * w:
        raise ValueError(
            f"UPSAMPLE input dim mismatch: hz.dim={hz.dim}, shape={in_shape}"
        )

    device = hz.c.device
    hh = torch.div(
        torch.arange(h2, device=device, dtype=torch.float64) * h,
        max(h2, 1),
        rounding_mode="floor",
    ).to(torch.long).clamp_(0, h - 1)
    ww = torch.div(
        torch.arange(w2, device=device, dtype=torch.float64) * w,
        max(w2, 1),
        rounding_mode="floor",
    ).to(torch.long).clamp_(0, w - 1)
    nn = torch.arange(n, device=device, dtype=torch.long)
    cc = torch.arange(c, device=device, dtype=torch.long)
    grid_n, grid_c, grid_h, grid_w = torch.meshgrid(
        nn, cc, hh, ww, indexing="ij"
    )
    idx = (((grid_n * c + grid_c) * h + grid_h) * w + grid_w).reshape(-1)

    return HZono(
        c=hz.c.index_select(0, idx),
        Gc=hz.Gc.index_select(0, idx),
        Gb=hz.Gb.index_select(0, idx),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=(hz.eq_mask.clone() if hz.eq_mask is not None else None),
    )


def _hz_gather_exact(hz: "HZono", params: Dict[str, Any]) -> "HZono":
    """Exact HZ transfer for GATHER (ONNX axis-wise index selection).

    GATHER is a linear row-selection: ``output[..., k, ...] = input[..., indices[k], ...]``
    along the gather axis. This is mathematically equivalent to a permutation /
    sparse selection matrix; HZ's c, Gc, Gb propagate exactly via ``index_select``
    on the corresponding flat positions. No new generators, no relaxation.

    Required ``params`` (matching the interval-tf gather meta in tf_mlp.py):
        - ``axis``: int, ONNX axis (relative to inp_shape, NOT including batch).
        - ``indices``: list of int — the positions to gather (may be a scalar list).
        - ``input_shape``: tuple, the shape of the input tensor (no batch dim).
        - ``output_shape``: tuple, the shape of the output tensor.

    Implementation: compute a flat index permutation from output flat index to
    input flat index, then apply ``index_select`` to c, Gc, Gb.
    """
    if not isinstance(hz, HZono):
        if hasattr(hz, "to_hzono"):
            hz = hz.to_hzono()
        elif hasattr(hz, "to_full_hzono"):
            hz = hz.to_full_hzono()
        else:
            raise ValueError(
                f"GATHER exact path requires HZono, got {type(hz).__name__}"
            )

    indices_raw = params.get("indices", None)
    if indices_raw is None:
        raise ValueError("GATHER meta missing 'indices'")
    if not isinstance(indices_raw, torch.Tensor):
        indices = torch.tensor(indices_raw, dtype=torch.long)
    else:
        indices = indices_raw.to(torch.long)
    indices = indices.reshape(-1)  # always 1-D for flat selection

    axis = int(params.get("axis", 0))
    in_shape = tuple(int(x) for x in params.get("input_shape", ()))
    if not in_shape:
        raise ValueError("GATHER meta missing 'input_shape'")
    rank = len(in_shape)
    if axis < 0:
        axis = rank + axis
    if axis < 0 or axis >= rank:
        raise ValueError(f"GATHER invalid axis={axis} for rank={rank}")

    n_in = 1
    for d in in_shape:
        n_in *= int(d)
    if int(hz.dim) != n_in:
        raise ValueError(
            f"GATHER input dim mismatch: hz.dim={hz.dim}, input_shape={in_shape}, n_in={n_in}"
        )

    # Compute prefix/axis/suffix split for flat index arithmetic
    prefix_size = 1
    for d in in_shape[:axis]:
        prefix_size *= int(d)
    axis_dim = int(in_shape[axis])
    suffix_size = 1
    for d in in_shape[axis + 1:]:
        suffix_size *= int(d)

    n_indices = int(indices.shape[0])
    device = hz.c.device

    # Bounds-check indices (interval_tf already wrapped negatives but be defensive)
    indices = indices.to(device).clamp_(0, axis_dim - 1)

    # For each (p, k, s) output position the input flat index is:
    #   p * axis_dim * suffix_size + indices[k] * suffix_size + s
    p_idx = torch.arange(prefix_size, device=device, dtype=torch.long).reshape(-1, 1, 1)
    k_idx = indices.reshape(1, -1, 1)
    s_idx = torch.arange(suffix_size, device=device, dtype=torch.long).reshape(1, 1, -1)

    flat_in_idx = (
        p_idx * (axis_dim * suffix_size) + k_idx * suffix_size + s_idx
    ).reshape(-1)

    return HZono(
        c=hz.c.index_select(0, flat_in_idx),
        Gc=hz.Gc.index_select(0, flat_in_idx),
        Gb=hz.Gb.index_select(0, flat_in_idx),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=(hz.eq_mask.clone() if hz.eq_mask is not None else None),
    )


def _hz_slice_exact(hz: "HZono", params: Dict[str, Any]) -> "HZono":
    """Exact HZ transfer for SLICE (ONNX axis-wise slicing).

    SLICE selects a contiguous strided subset of indices along one or more axes;
    it is a linear permutation/selection map and is exactly representable in HZ
    via index_select on c, Gc, Gb. No new generators, no relaxation.

    Required ``params`` (matching tf_slice meta):
        - ``starts``, ``ends``, ``axes``, ``steps``: lists describing the slice.
        - ``input_shape``: tuple of input dims (no batch).

    If the ROUND9 shape-mismatch sentinel is set the meta cannot describe a real
    permutation; fall back to caller's box-fallback by raising.
    """
    if params.get("ROUND9_shape_mismatch_sentinel", False):
        raise ValueError("SLICE exact path: ROUND9 shape sentinel — cannot do exact")
    if not isinstance(hz, HZono):
        if hasattr(hz, "to_hzono"):
            hz = hz.to_hzono()
        elif hasattr(hz, "to_full_hzono"):
            hz = hz.to_full_hzono()
        else:
            raise ValueError(
                f"SLICE exact path requires HZono, got {type(hz).__name__}"
            )

    in_shape = tuple(int(x) for x in params.get("input_shape", ()))
    if not in_shape:
        raise ValueError("SLICE meta missing 'input_shape'")
    rank = len(in_shape)
    n_in = 1
    for d in in_shape:
        n_in *= int(d)
    if int(hz.dim) != n_in:
        raise ValueError(
            f"SLICE input dim mismatch: hz.dim={hz.dim}, input_shape={in_shape}, n_in={n_in}"
        )

    starts = list(params.get("starts", []))
    ends = list(params.get("ends", []))
    axes = list(params.get("axes", list(range(len(starts)))))
    steps = list(params.get("steps", [1] * len(axes)))

    # Build per-axis index ranges
    device = hz.c.device
    axis_ranges = []
    for d in range(rank):
        axis_ranges.append(
            torch.arange(int(in_shape[d]), device=device, dtype=torch.long)
        )
    for i, axis in enumerate(axes):
        axis = int(axis)
        if axis < 0:
            axis = rank + axis
        s = int(starts[i])
        e = int(ends[i])
        st = int(steps[i]) if i < len(steps) else 1
        # Handle negative indices and clamping per ONNX semantics
        dim = int(in_shape[axis])
        if s < 0:
            s = max(0, s + dim)
        if e < 0:
            e = max(0, e + dim)
        s = max(0, min(s, dim))
        e = max(0, min(e, dim))
        # PyTorch arange with step (incl. negative step)
        if st > 0:
            axis_ranges[axis] = torch.arange(s, e, st, device=device, dtype=torch.long)
        elif st < 0:
            axis_ranges[axis] = torch.arange(s, e, st, device=device, dtype=torch.long)
        else:
            raise ValueError(f"SLICE invalid step=0 on axis={axis}")

    # Build flat index = prod for the cartesian product over axis_ranges
    # Use meshgrid then ravel
    grids = torch.meshgrid(*axis_ranges, indexing="ij")
    # Compute flat strides
    strides = [1] * rank
    for d in range(rank - 2, -1, -1):
        strides[d] = strides[d + 1] * int(in_shape[d + 1])
    flat_in_idx = torch.zeros_like(grids[0])
    for d in range(rank):
        flat_in_idx = flat_in_idx + grids[d] * int(strides[d])
    flat_in_idx = flat_in_idx.reshape(-1)

    return HZono(
        c=hz.c.index_select(0, flat_in_idx),
        Gc=hz.Gc.index_select(0, flat_in_idx),
        Gb=hz.Gb.index_select(0, flat_in_idx),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=(hz.eq_mask.clone() if hz.eq_mask is not None else None),
    )


def _hz_convtranspose2d_native(hz: HZono, params: Dict[str, Any]) -> HZono:
    """Exact HZ transfer for ``ConvTranspose2d`` without dense W materialization."""
    if not isinstance(hz, HZono):
        if hasattr(hz, "to_hzono"):
            hz = hz.to_hzono()
        elif hasattr(hz, "to_full_hzono"):
            hz = hz.to_full_hzono()
        else:
            raise ValueError(
                f"ConvTranspose2d exact path requires HZono, got {type(hz).__name__}"
            )

    import torch.nn.functional as F

    weight = params.get("weight")
    if weight is None:
        raise ValueError("ConvTranspose2d exact path missing weight")
    weight_t = torch.as_tensor(weight, dtype=hz.c.dtype, device=hz.c.device)
    bias = params.get("b", params.get("bias", None))
    bias_t = (
        torch.as_tensor(bias, dtype=hz.c.dtype, device=hz.c.device).flatten()
        if bias is not None else None
    )
    conv_params = dict(params.get("conv_params", {}) or {})
    stride = conv_params.get("stride", params.get("stride", 1))
    padding = conv_params.get("padding", params.get("padding", 0))
    output_padding = conv_params.get(
        "output_padding", params.get("output_padding", 0)
    )
    dilation = conv_params.get("dilation", params.get("dilation", 1))
    groups = int(conv_params.get("groups", params.get("groups", 1)))
    input_shape = tuple(int(x) for x in params.get("input_shape", ()))
    output_shape = tuple(int(x) for x in params.get("output_shape", ()))
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise ValueError("ConvTranspose2d exact path requires rank-4 shapes")
    _, c_in, h_in, w_in = input_shape
    _, c_out, h_out, w_out = output_shape
    if int(hz.dim) != c_in * h_in * w_in:
        raise ValueError(
            "ConvTranspose2d input dim mismatch: "
            f"hz.dim={hz.dim}, input_shape={input_shape}"
        )

    def _apply_cols(mat: torch.Tensor) -> torch.Tensor:
        if int(mat.shape[1]) == 0:
            return torch.zeros(
                (c_out * h_out * w_out, 0),
                dtype=hz.c.dtype,
                device=hz.c.device,
            )
        chunk = int(os.environ.get("HYZOR_CONVTRANSPOSE_COL_CHUNK", "128"))
        outs = []
        for start in range(0, int(mat.shape[1]), chunk):
            block = mat[:, start:start + chunk].transpose(0, 1).contiguous()
            block4 = block.view(-1, c_in, h_in, w_in)
            out4 = F.conv_transpose2d(
                block4, weight_t, None,
                stride=stride, padding=padding,
                output_padding=output_padding, dilation=dilation,
                groups=groups,
            )
            outs.append(out4.reshape(out4.shape[0], -1).transpose(0, 1))
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    c4 = hz.c.reshape(1, c_in, h_in, w_in)
    c_out_t = F.conv_transpose2d(
        c4, weight_t, bias_t,
        stride=stride, padding=padding,
        output_padding=output_padding, dilation=dilation,
        groups=groups,
    ).reshape(-1, 1)
    expected_dim = c_out * h_out * w_out
    if int(c_out_t.numel()) != expected_dim:
        raise ValueError(
            "ConvTranspose2d output dim mismatch: "
            f"got={int(c_out_t.numel())}, expected={expected_dim}"
        )
    return HZono(
        c=c_out_t,
        Gc=_apply_cols(hz.Gc),
        Gb=_apply_cols(hz.Gb),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=(hz.eq_mask.clone() if hz.eq_mask is not None else None),
    )


def _assert_is_order_only_zero_threshold(assert_layer: Layer) -> bool:
    """True iff the output spec only asks pairwise class ordering.

    For a final softmax layer, pairwise zero-threshold comparisons are
    exactly preserved by the monotone exponential normalization:
    ``softmax_i >= softmax_j`` iff ``logit_i >= logit_j``.  This helper is
    intentionally narrow; margins, arbitrary linear rows, ranges, and
    non-zero thresholds must keep the normal softmax fallback.
    """
    try:
        kind = _kind_str(assert_layer.params.get("kind"))
        if kind == "TOP1_ROBUST":
            return True
        if kind != "UNSAFE_LINEAR":
            return False
        C = _to_np_f64(
            assert_layer.params.get("c", assert_layer.params.get("C"))
        )
        d = _to_np_f64(
            assert_layer.params.get("d", assert_layer.params.get("thresholds"))
        ).reshape(-1)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        else:
            C = C.reshape(-1, C.shape[-1])
        if C.shape[0] != d.shape[0] or not np.all(np.abs(d) <= 1e-12):
            return False
        for row in C:
            nz = np.flatnonzero(np.abs(row) > 1e-12)
            if nz.size != 2:
                return False
            vals = sorted(float(row[i]) for i in nz)
            if abs(vals[0] + 1.0) > 1e-12 or abs(vals[1] - 1.0) > 1e-12:
                return False
        return True
    except Exception:
        return False


def _hz_sign_convex_hull(hz: HZono, in_bounds: Bounds) -> HZono:
    """Sound convex-hull transfer for elementwise ``torch.sign``.

    Stable inputs become constants.  For crossing intervals ``l < 0 < u``,
    introduce one continuous output variable ``y`` in ``[-1, 1]`` and add
    the two hull facets of the discontinuous sign graph:

      ``y >= 2 x / u - 1`` and ``y <= 2 x / (-l) + 1``.

    This keeps a forward-only HZ relation between the pre-sign affine form
    and the post-sign abstraction.  It is an over-approximation of
    torch.sign, including the exact-zero point, and uses no splitting.
    """
    if not isinstance(hz, HZono):
        raise ValueError("SIGN hull currently requires dense HZono")
    lb = in_bounds.lb.flatten().to(dtype=hz.c.dtype, device=hz.c.device)
    ub = in_bounds.ub.flatten().to(dtype=hz.c.dtype, device=hz.c.device)
    n = int(hz.dim)
    if int(lb.numel()) != n or int(ub.numel()) != n:
        raise ValueError(
            f"SIGN hull bound shape mismatch: bounds={int(lb.numel())}, hz.dim={n}"
        )
    tol = float(os.environ.get("HYZOR_SIGN_HULL_TOL", "1e-12"))
    neg = ub < -tol
    pos = lb > tol
    exact_zero = (lb.abs() <= tol) & (ub.abs() <= tol)
    cross = (lb < -tol) & (ub > tol)
    nonneg_zero = (~pos) & (~exact_zero) & (lb.abs() <= tol) & (ub > tol)
    nonpos_zero = (~neg) & (~exact_zero) & (ub.abs() <= tol) & (lb < -tol)
    variable = cross | nonneg_zero | nonpos_zero
    var_idx = torch.nonzero(variable, as_tuple=False).view(-1)
    k = int(var_idx.numel())
    cross_idx = torch.nonzero(cross, as_tuple=False).view(-1)
    kc = int(cross_idx.numel())
    max_unstable = int(os.environ.get("HYZOR_SIGN_HULL_MAX_UNSTABLE", "6000"))
    if k > max_unstable:
        raise ValueError(f"SIGN hull unstable count {k} exceeds cap {max_unstable}")

    p0 = int(hz.ng)
    q0 = int(hz.nb)
    keep_old = kc > 0
    p_out = (p0 if keep_old else 0) + k
    budget = int(
        float(os.environ.get("HYZOR_SIGN_HULL_MAT_BUDGET_GB", "3.0"))
        * (1024 ** 3)
    )
    est = n * max(p_out + q0, 1) * hz.c.element_size()
    if est > budget:
        raise ValueError(
            "SIGN hull materialization would exceed budget: "
            f"{est / (1024 ** 3):.2f} GiB > {budget / (1024 ** 3):.2f} GiB"
        )

    c_out = torch.zeros((n, 1), dtype=hz.c.dtype, device=hz.c.device)
    c_out[pos, 0] = 1.0
    c_out[neg, 0] = -1.0
    c_out[nonneg_zero, 0] = 0.5
    c_out[nonpos_zero, 0] = -0.5
    Gc_out = torch.zeros((n, p_out), dtype=hz.c.dtype, device=hz.c.device)
    Gb_out = torch.zeros((n, q0), dtype=hz.c.dtype, device=hz.c.device)
    if k > 0:
        new_col0 = p0 if keep_old else 0
        cols = new_col0 + torch.arange(k, dtype=torch.long, device=hz.c.device)
        coeff = torch.ones(k, dtype=hz.c.dtype, device=hz.c.device)
        coeff[nonneg_zero[var_idx] | nonpos_zero[var_idx]] = 0.5
        Gc_out[var_idx, cols] = coeff

    if not keep_old:
        return HZono(
            c=c_out,
            Gc=Gc_out,
            Gb=Gb_out,
            Ac=torch.zeros((0, p_out), dtype=hz.c.dtype, device=hz.c.device),
            Ab=torch.zeros((0, q0), dtype=hz.c.dtype, device=hz.c.device),
            b=torch.zeros((0, 1), dtype=hz.c.dtype, device=hz.c.device),
            eq_mask=None,
        )

    nc0 = int(hz.nc)
    Ac_base = torch.cat(
        [
            hz.Ac,
            torch.zeros((nc0, k), dtype=hz.c.dtype, device=hz.c.device),
        ],
        dim=1,
    )
    Ab_base = hz.Ab.clone()
    b_base = hz.b.clone()
    eq_base = _eq_mask_of(hz).clone()

    col_of_dim = torch.full((n,), -1, dtype=torch.long, device=hz.c.device)
    col_of_dim[var_idx] = torch.arange(k, dtype=torch.long, device=hz.c.device)
    cross_cols = p0 + col_of_dim[cross_idx]
    a1 = (2.0 / ub[cross_idx]).view(-1, 1)
    a2 = (2.0 / (-lb[cross_idx])).view(-1, 1)
    Gx = hz.Gc.index_select(0, cross_idx)
    Bx = hz.Gb.index_select(0, cross_idx)
    cx = hz.c.index_select(0, cross_idx)

    Ac_new = torch.zeros((2 * kc, p_out), dtype=hz.c.dtype, device=hz.c.device)
    Ab_new = torch.zeros((2 * kc, q0), dtype=hz.c.dtype, device=hz.c.device)
    b_new = torch.zeros((2 * kc, 1), dtype=hz.c.dtype, device=hz.c.device)
    rows = torch.arange(kc, dtype=torch.long, device=hz.c.device)
    Ac_new[rows, :p0] = a1 * Gx
    Ac_new[rows, cross_cols] = -1.0
    Ab_new[rows, :] = a1 * Bx
    b_new[rows, :] = 1.0 - a1 * cx

    rows2 = rows + kc
    Ac_new[rows2, :p0] = -a2 * Gx
    Ac_new[rows2, cross_cols] = 1.0
    Ab_new[rows2, :] = -a2 * Bx
    b_new[rows2, :] = 1.0 + a2 * cx

    return HZono(
        c=c_out,
        Gc=Gc_out,
        Gb=Gb_out,
        Ac=torch.cat([Ac_base, Ac_new], dim=0),
        Ab=torch.cat([Ab_base, Ab_new], dim=0),
        b=torch.cat([b_base, b_new], dim=0),
        eq_mask=torch.cat(
            [
                eq_base,
                torch.zeros((2 * kc,), dtype=torch.bool, device=hz.c.device),
            ],
            dim=0,
        ),
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
    """HZ verdict-producing verifier. Walks ACT cons IR, propagates HZono
    through hz_routing, and runs final unsafe-set LP feasibility + strict
    ONNX replay to produce CERTIFIED / FALSIFIED / UNKNOWN.

    Companion to HZSolver (which only computes bounds without verdicts).
    """

    def __init__(
        self, *,
        relu_method: str = "eq_lagr_v8",
        girard_cap: int = 6000,
        sgm_enabled: bool = True,
        strict_replay: bool = True,
        sigmoid_K: int = 2,
        tanh_K: int = 2,
        # large_cls scheduling for cifar100 / tinyimagenet-shaped nets
        large_cls_proof_mode: str = "auto",   # "on" | "off" | "auto"
        large_cls_eq_layers: int = 3,
        large_cls_conv_threshold: int = 4,
        large_cls_out_dim_threshold: int = 100,
        # Property-facing tail correlation preservation: optionally skip
        # Girard reduce on layers whose output dim is small. Always sound:
        # skip = identity, not a widening. Default-off because it did not
        # improve the default eq_lagr_v8 path in the controlled tiny slice;
        # it remains useful for chull-tail research runs.
        tail_preserve_dim: int = 0,
        # Generator retention policy inside sound Girard widening. A positive
        # value preserves columns participating strongly in carried hull
        # inequalities, instead of ranking only by current output magnitude.
        # This changes precision, never containment.
        constraint_keep_weight: float = 0.0,
        timeout_s: float = 300.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        onnx_path: Optional[str] = None,
        vnnlib_path: Optional[str] = None,
        # Sound small-dense forward-LP portfolio. ``base`` selects
        # GlobalTriangleLP; ``specaware`` selects conditional unsafe-set
        # refinement; ``witness``/``auto`` also performs ORT-validated
        # counterexample extraction. The backend is explicitly supplied from
        # a module root until it is fully vendored into ACT.
        small_dense_lp: str = "specaware",    # "off" | "base" | "specaware" | "witness" | "auto"
        small_dense_lp_root: Optional[str] = None,
        small_dense_lp_time_limit_s: float = 5.0,
        # 0 selects the audited cost-aware policy: 20 passes for a single
        # hidden ReLU layer with <=128 units, otherwise 3. A positive value
        # overrides the policy. More passes only tighten the sound LP.
        small_dense_lp_refinement_passes: int = 0,
        small_dense_lp_fallback_on_unknown: bool = False,
        # FAL receipt provenance (advisor 2026-05-24): callers MUST pass
        # these for receipts to be unique. ``benchmark`` defaults to the
        # ONNX path stem; ``instance_id`` defaults to -1 (a sentinel that
        # causes filename collisions if the caller forgets — fail loud).
        benchmark: Optional[str] = None,
        instance_id: int = -1,
        query_index: int = 0,
    ):
        if small_dense_lp not in ("off", "base", "specaware", "witness", "auto"):
            raise ValueError(
                "small_dense_lp must be one of off/base/specaware/witness/auto; "
                f"got {small_dense_lp!r}"
            )
        if int(small_dense_lp_refinement_passes) < 0:
            raise ValueError(
                "small_dense_lp_refinement_passes must be >= 0; "
                f"got {small_dense_lp_refinement_passes!r}"
            )
        self.cfg = dict(
            relu_method=relu_method, girard_cap=girard_cap,
            sgm_enabled=sgm_enabled,
            strict_replay=strict_replay,
            sigmoid_K=sigmoid_K, tanh_K=tanh_K,
            large_cls_proof_mode=large_cls_proof_mode,
            large_cls_eq_layers=large_cls_eq_layers,
            large_cls_conv_threshold=large_cls_conv_threshold,
            large_cls_out_dim_threshold=large_cls_out_dim_threshold,
            tail_preserve_dim=tail_preserve_dim,
            constraint_keep_weight=constraint_keep_weight,
            timeout_s=timeout_s, device=device, dtype=dtype,
            onnx_path=onnx_path,
            vnnlib_path=vnnlib_path,
            small_dense_lp=small_dense_lp,
            small_dense_lp_root=small_dense_lp_root,
            small_dense_lp_time_limit_s=small_dense_lp_time_limit_s,
            small_dense_lp_refinement_passes=int(small_dense_lp_refinement_passes),
            small_dense_lp_fallback_on_unknown=small_dense_lp_fallback_on_unknown,
            benchmark=benchmark,
            instance_id=int(instance_id),
            query_index=int(query_index),
        )
        self._reset_state()

    def _reset_state(self):
        self._status: str = SolveStatus.UNKNOWN
        self._witness: Optional[np.ndarray] = None
        self._has_solution: bool = False
        self._var_count: int = 0
        self._stats: Dict[str, Any] = {}
        self._final_output_ids: Tuple[int, ...] = ()
        self._final_softmax_order_only: bool = False
        self._convtranspose_triangle_profile_active: bool = False

    def _emit_sat_with_receipt(
        self,
        *,
        x_star: np.ndarray,
        assert_layer,
        source: str,
        model_path_fallback: Optional[str] = None,
    ) -> bool:
        """Emit a SAT verdict and compute its formal-result reportability.

        CONTRACT (advisor 2026-05-24 Round 4, correcting Round 3):

            Mathematical truth (internal ``_status``): SAT is set
            unconditionally whenever this method is called — caller has
            already verified the witness via ``strict_replay_for_act``,
            so the adversary genuinely exists. Pretending otherwise (the
            Round 3 over-downgrade to UNKNOWN) would make real
            counter-examples disappear from solver state, which is
            DISHONEST.

            Reportability (``_stats["formal_result"]``): orthogonal field
            that records whether this SAT is admissible into paper-grade
            FAL counts. CLI consumes formal_result in formal mode.

        formal_result values (formal-mode only; None otherwise):
            REPORTABLE_FALSIFIED    — receipt written, zero_tol_holds=True
            ERROR_RECEIPT_MISSING   — no receipt_dir / sentinel iid
            ERROR_RECEIPT_COLLISION — would overwrite prior receipt
            ERROR_RECEIPT_WRITE     — IO / hash / ORT failure
            ERROR_RECEIPT_READBACK  — wrote OK but couldn't re-parse
            ERROR_INTERNAL_INCONSISTENCY
                — receipt's recomputed zero_tol disagrees with the
                  strict_replay_for_act result that gated us here.
                  The witness x* is the SAME for both checks, so this
                  is a genuine internal bug (ORT non-determinism /
                  reshape mismatch / _eval_unsafe_strict vs receipt
                  evaluator drift). Run should be flagged broken.

        Returns:
            True (caller can rely on _status == SAT). The bool is kept
            for API parity with prior callers; reportability is in stats.
        """
        import os as _os, json as _json
        from act.back_end.solver.fal_receipt import (
            write_receipt, ReceiptCollisionError,
        )
        formal = _os.environ.get("ACT_FAL_RECEIPT_FORMAL", "").strip().lower() \
            in ("1", "true", "yes", "on")
        rp: Optional[Path] = None
        receipt_error: Optional[str] = None
        receipt_error_kind: Optional[str] = None
        model_path = self.cfg.get("onnx_path") or model_path_fallback or ""
        spec_path = self.cfg.get("vnnlib_path", "")
        # R9.3: forward the input-box check result (if available) so
        # the receipt records audit-grade provenance, not just y.
        ib_holds = None; ib_reason = None
        try:
            net_ref = getattr(self, "_last_net_for_replay", None)
            if net_ref is not None:
                cached = getattr(net_ref, "_last_input_box_check", None)
                if cached is not None:
                    ib_holds, ib_reason = cached
        except Exception:
            pass
        try:
            rp = write_receipt(
                x_star=x_star,
                model_path=model_path,
                spec_path=spec_path,
                assert_layer=assert_layer,
                benchmark=(self.cfg.get("benchmark")
                           or (Path(model_path).stem if model_path else "unknown")),
                instance_id=int(self.cfg.get("instance_id", -1)),
                query_index=int(self.cfg.get("query_index", 0)),
                source=source,
                formal_mode=formal,
                input_box_holds=ib_holds,
                input_box_reason=ib_reason,
            )
        except ReceiptCollisionError as e:
            receipt_error = f"ReceiptCollisionError: {e}"
            msg_lower = str(e).lower()
            if "collision" in msg_lower:
                receipt_error_kind = "ERROR_RECEIPT_COLLISION"
            elif "instance_id" in msg_lower or "sentinel" in msg_lower or "receipt_dir" in msg_lower:
                receipt_error_kind = "ERROR_RECEIPT_MISSING"
            else:
                receipt_error_kind = "ERROR_RECEIPT_WRITE"
        except Exception as e:
            receipt_error = f"{type(e).__name__}: {e}"
            receipt_error_kind = "ERROR_RECEIPT_WRITE"

        # Truth: set SAT internally (strict_replay_for_act already
        # verified the witness — do not pretend otherwise).
        self._status = SolveStatus.SAT
        self._witness = x_star
        self._has_solution = True
        if rp is not None:
            self._stats["fal_receipt_path"] = str(rp)
        if receipt_error:
            self._stats["receipt_error"] = receipt_error

        # Compute formal_result (only in formal mode)
        if not formal:
            return True

        if rp is None:
            self._stats["formal_result"] = (
                receipt_error_kind or "ERROR_RECEIPT_WRITE"
            )
            return True

        # Receipt was written; verify its recomputed zero_tol matches
        # the strict_replay_for_act outcome that brought us here.
        try:
            rec = _json.loads(Path(rp).read_text())
            zero_holds = bool(rec.get("spec_zero_tol_holds"))
        except Exception as e:
            self._stats["formal_result"] = "ERROR_RECEIPT_READBACK"
            self._stats["receipt_error"] = (
                self._stats.get("receipt_error") or f"readback: {e}"
            )
            return True

        if not zero_holds:
            # strict_replay_for_act returned True but the receipt's
            # independent ORT recomputation says zero_tol fails. The
            # same x* gave inconsistent verdicts — investigate.
            self._stats["formal_result"] = "ERROR_INTERNAL_INCONSISTENCY"
            return True

        self._stats["formal_result"] = "REPORTABLE_FALSIFIED"
        return True

    def capabilities(self) -> SolverCaps:
        # supports_csp=False: HZVerifier walks the ACT cons IR via
        # ``consume_cons`` (HZ-native), it does not accept BatchLPProblem.
        # Mirrors HZSolver / DualSolver, which are also non-CSP solvers.
        return SolverCaps(supports_gpu=True, supports_csp=False, supports_hz=True)

    def solve_batch(self, problem, timelimit: Optional[float] = None):  # noqa: D401
        """HZVerifier does not accept BatchLPProblem inputs.

        Callers verifying a single (model, spec) instance through HZ should
        use ``verify_once_hz(net, solver=..., timelimit=...)`` from this
        module, which drives the cons-walker on the analyzed ACT Net.
        """
        raise NotImplementedError(
            "HZVerifier does not consume BatchLPProblem; use "
            "act.back_end.solver.solver_hz.verify_once_hz"
            "(net, solver=..., timelimit=...) for single-instance HZ "
            "verification."
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

    def _try_small_dense_lp(
        self, *, conv_count: int,
        net=None, assert_layer=None,
    ) -> Optional[str]:
        """Try the sound forward-LP portfolio before the HZ walk.

        The optional backend is only allowed to prove safety.  Once an
        eligible network was successfully checked, its ``unknown`` is
        terminal by default because the HZ cascade is known to be both
        slower and weaker on this small-dense family.  Callers can opt into
        fallback if they need the HZ counterexample attempt.

        SOUNDNESS gate (added 2026-05-24 per advisor review):
            For the witness/auto backend (WitnessExtract), the external
            module's `_ort_replay` uses a +1e-6 slack (HyZor/WitnessExtract.py:169),
            i.e. small_tol acceptance. A 'falsified' verdict from that
            backend can therefore be a boundary witness, not a hard FAL.
            To prevent paper-grade SAT claims from leaking boundary
            witnesses, this method now takes ONLY the x* and re-runs
            ACT's own ``strict_replay_for_act`` (zero-tolerance ORT eval
            via _eval_unsafe_strict). Witnesses that fail the strict
            replay downgrade to UNKNOWN with phantom_rejected_small_dense=True.

            Caller MUST pass ``net`` and ``assert_layer`` for this gate.
            If either is absent the attempt fails closed as UNKNOWN with
            ``small_dense_lp_strict_replay_unavailable=True``; no external
            falsification may be promoted without local strict replay.
        """
        mode = self.cfg.get("small_dense_lp", "auto")
        if mode == "off" or conv_count != 0:
            return None
        onnx_path = self.cfg.get("onnx_path")
        vnnlib_path = self.cfg.get("vnnlib_path")
        if not onnx_path or not vnnlib_path:
            return None

        module_root = self.cfg.get("small_dense_lp_root")
        if not module_root:
            sibling = Path(__file__).resolve().parents[4] / "HyZor"
            if (
                (sibling / "GlobalTriangleLP.py").is_file()
                and (sibling / "SpecAwareLP.py").is_file()
            ):
                module_root = str(sibling)
        if module_root:
            root = str(Path(module_root))
            if not Path(root).is_dir():
                self._stats["small_dense_lp_error"] = f"missing module root: {root}"
                return None
            if root not in sys.path:
                sys.path.insert(0, root)
            importlib.invalidate_caches()

        try:
            base = importlib.import_module("GlobalTriangleLP")
            if not base.is_small_dense(Path(onnx_path)):
                return None
            refinement_passes: Optional[int] = None
            refinement_policy: Optional[str] = None
            if mode != "base":
                configured_passes = int(
                    self.cfg.get("small_dense_lp_refinement_passes", 0)
                )
                if configured_passes > 0:
                    refinement_passes = configured_passes
                    refinement_policy = "explicit"
                else:
                    # Safenlp/sat_relu audit: this shallow case gained 49
                    # sound CERTs at negligible additional wall time with 20
                    # passes. Keep deeper ACAS/TLL/nn4sys paths at 3 passes.
                    refinement_passes = 3
                    refinement_policy = "default"
                    try:
                        _, hidden_layers, _ = base.extract_layers(Path(onnx_path))
                        total_hidden = sum(int(b.shape[0]) for _, b in hidden_layers)
                        if len(hidden_layers) == 1 and total_hidden <= 128:
                            refinement_passes = 20
                            refinement_policy = "shallow_20"
                    except Exception:
                        pass
            # Mode → backend module:
            #   "base"      → GlobalTriangleLP (verify only)
            #   "specaware" → SpecAwareLP (verify only, no falsification)
            #   "witness" / "auto" → WitnessExtract (SpecAware + ORT-replay-
            #                 validated counterexample search; SOUND because
            #                 ORT replay confirms each FAL).
            # Two-tier reporting note (feedback_two_tier_reporting):
            #   witness mode may produce 'falsified' on instances whose
            #   official_zero=unsat but official_small=sat (the boundary
            #   tolerance regime). These ARE valid falsifications per the
            #   model's float32 precision, but disagree with strict zero-
            #   tolerance verification. Per the advisor's framework these
            #   are P1 watchlist, NOT soundness regressions.
            if mode == "base":
                backend = base
                backend_label = "base"
                verdict, elapsed = backend.verify(
                    Path(onnx_path),
                    Path(vnnlib_path),
                    time_limit_per_lp=float(self.cfg["small_dense_lp_time_limit_s"]),
                )
                witness = None
                y_ort = None
            elif mode == "specaware":
                backend = importlib.import_module("SpecAwareLP")
                backend_label = "specaware"
                verdict, elapsed = backend.verify(
                    Path(onnx_path),
                    Path(vnnlib_path),
                    time_limit_per_lp=float(self.cfg["small_dense_lp_time_limit_s"]),
                    max_refinement_passes=int(refinement_passes),
                )
                witness = None
                y_ort = None
            else:  # "witness" or "auto"
                backend = importlib.import_module("WitnessExtract")
                backend_label = "witness"
                result = backend.verify_with_falsification(
                    Path(onnx_path),
                    Path(vnnlib_path),
                    time_limit_per_lp=float(self.cfg["small_dense_lp_time_limit_s"]),
                    max_refinement_passes=int(refinement_passes),
                    return_witness=True,
                )
                verdict, witness, y_ort, elapsed = result
            self._stats["small_dense_lp_dispatch"] = True
            self._stats["small_dense_lp_backend"] = backend_label
            self._stats["small_dense_lp_verdict"] = verdict
            self._stats["small_dense_lp_elapsed_s"] = elapsed
            if refinement_passes is not None:
                self._stats["small_dense_lp_refinement_passes"] = refinement_passes
                self._stats["small_dense_lp_refinement_policy"] = refinement_policy
            if verdict == "verified":
                self._status = SolveStatus.UNSAT
                return self._status
            if verdict == "falsified":
                # SOUNDNESS GATE: the external backend's ORT replay accepts
                # +1e-6 slack (small_tol). Re-run ACT's strict zero-tol
                # replay before promoting to SAT. See method docstring.
                if witness is None:
                    self._status = SolveStatus.UNKNOWN
                    self._stats["small_dense_lp_no_witness"] = True
                    return self._status
                x_star_np = np.asarray(witness, dtype=np.float64).ravel()
                if net is None or assert_layer is None:
                    # Fail-closed per advisor 2026-05-24 Route C: without
                    # ACT-side context we cannot run the strict zero-tol
                    # replay, so the witness MUST NOT be promoted to SAT
                    # on the strength of the external backend's +1e-6
                    # acceptance. Downgrade to UNKNOWN and flag for audit.
                    self._stats["small_dense_lp_strict_replay_unavailable"] = True
                    self._status = SolveStatus.UNKNOWN
                    return self._status
                # Ensure net carries onnx_path so strict_replay_for_act can
                # take the ORT fast path.
                if self.cfg.get("onnx_path") and not getattr(net, "onnx_path", None):
                    try: net.onnx_path = self.cfg["onnx_path"]
                    except Exception: pass
                try:
                    ok = strict_replay_for_act(
                        net=net, x_star=x_star_np, assert_layer=assert_layer
                    )
                    # R9.3: stash net so _emit_sat_with_receipt can read
                    # the input-box check result it just recorded.
                    self._last_net_for_replay = net
                except Exception as e:
                    ok = False
                    self._stats["small_dense_lp_strict_replay_error"] = (
                        f"{type(e).__name__}: {e}"
                    )
                if ok:
                    sat_ok = self._emit_sat_with_receipt(
                        x_star=x_star_np,
                        assert_layer=assert_layer,
                        source="small_dense_lp_witness",
                    )
                    if sat_ok:
                        self._stats["small_dense_lp_strict_replay_passed"] = True
                    return self._status
                else:
                    self._status = SolveStatus.UNKNOWN
                    self._stats["small_dense_lp_phantom_rejected"] = True
                return self._status
            if (
                verdict == "unknown"
                and not self.cfg["small_dense_lp_fallback_on_unknown"]
            ):
                self._status = SolveStatus.UNKNOWN
                return self._status
        except Exception as exc:
            self._stats["small_dense_lp_error"] = f"{type(exc).__name__}: {exc}"
        return None

    # ----- Real entry: cons walker -----
    def consume_cons(
        self, globalC: ConSet, before: Dict[int, Fact], after: Dict[int, Fact],
        *, net: Net, input_ids: List[int], output_ids: List[int],
        assert_layer: Layer,
    ) -> str:
        if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
            print(
                f"[HZ-PROGRESS] consume_cons start layers={len(net.layers)} "
                f"inputs={len(input_ids)} outputs={len(output_ids)}",
                flush=True,
            )
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
        if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
            print(
                f"[HZ-PROGRESS] input_hz dim={input_hz.dim} ng={input_hz.ng} "
                f"nb={input_hz.nb} nc={input_hz.nc}",
                flush=True,
            )
        for poly_con in global_polys:
            input_hz = hz_intersect_polytope(
                input_hz, poly_con.meta["A"], poly_con.meta["b"])

        # Degenerate input boxes are exact singleton verification problems.
        # This is not a fallback or random witness search: γ(input_hz) has
        # at most one point, so strict ORT replay on that point decides the
        # current query exactly. This rescues sparse VNNLIBs that encode fixed
        # images as zero-width boxes while preserving the normal HZ path for
        # every non-singleton input.
        if (
            net is not None
            and assert_layer is not None
            and int(input_hz.ng) == 0
            and int(input_hz.nb) == 0
        ):
            if not _zero_factor_hz_feasible(input_hz):
                self._status = SolveStatus.UNSAT
                self._stats["singleton_input_empty"] = True
                return self._status
            x_single = input_hz.c.detach().cpu().numpy().reshape(-1)
            try:
                unsafe = strict_replay_for_act(
                    net=net, x_star=x_single, assert_layer=assert_layer,
                )
            except Exception as e:
                self._stats["singleton_replay_error"] = (
                    f"{type(e).__name__}: {e}"
                )
                unsafe = None
            if unsafe is True:
                self._emit_sat_with_receipt(
                    x_star=x_single,
                    assert_layer=assert_layer,
                    source="singleton_input",
                    model_path_fallback=self.cfg.get("onnx_path"),
                )
                return self._status
            if unsafe is False:
                self._status = SolveStatus.UNSAT
                self._stats["singleton_input_certified"] = True
                return self._status

        var_to_hz: Dict[Tuple[int, ...], Any] = {tuple(input_ids): input_hz}

        # ── Pre-scan: detect large_cls_proof_mode + count relus/convs ──
        relu_layer_ids: List[int] = []
        conv_count = 0
        convtranspose_count = 0
        for L in net.layers:
            ku = L.kind.upper()
            if ku == "RELU": relu_layer_ids.append(L.id)
            elif ku in ("CONV", "CONV2D", "CONV1D", "CONV3D"): conv_count += 1
            elif ku in ("CONVTRANSPOSE2D", "CONVTRANSPOSE", "CONV_TRANSPOSE2D"):
                convtranspose_count += 1
        total_relu = len(relu_layer_ids)
        out_dim = len(output_ids)
        self._final_output_ids = tuple(output_ids)
        self._final_softmax_order_only = (
            os.environ.get("ACT_HZ_FINAL_SOFTMAX_ORDER_BYPASS", "1")
            .strip().lower() in ("1", "true", "yes", "on")
            and _assert_is_order_only_zero_threshold(assert_layer)
        )
        if self._final_softmax_order_only:
            self._stats["final_softmax_order_only_spec"] = True
        if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
            print(
                f"[HZ-PROGRESS] prescan conv={conv_count} relu={total_relu} out_dim={out_dim}",
                flush=True,
            )

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
            logger.info(
                "large_cls_proof_mode ACTIVE: conv=%d out_dim=%d relus=%d "
                "(triangle for relu 1..%d, eq_lagr_v8 for last %d)",
                conv_count, out_dim, total_relu,
                total_relu - eq_last, eq_last,
            )
            self._stats["large_cls_active"] = True
            self._stats["total_relu"] = total_relu
            self._stats["eq_last"] = eq_last
        sparse_huge_auto = False
        if large_cls_active and os.environ.get(
            "ACT_HZ_AUTO_SPARSE_HUGE_PROFILE", "1"
        ).strip().lower() in ("1", "true", "yes", "on"):
            sparse_dim_min = int(os.environ.get(
                "ACT_HZ_SPARSE_HUGE_DIM_MIN", "50000"
            ))
            sparse_active_max = int(os.environ.get(
                "ACT_HZ_SPARSE_HUGE_ACTIVE_MAX", "64"
            ))
            sparse_huge_auto = (
                input_hz.dim >= sparse_dim_min
                and input_hz.ng <= sparse_active_max
                and conv_count >= self.cfg["large_cls_conv_threshold"]
            )
            if sparse_huge_auto:
                self._stats["sparse_huge_profile_active"] = True
                self._stats["sparse_huge_input_dim"] = int(input_hz.dim)
                self._stats["sparse_huge_input_ng"] = int(input_hz.ng)
                logger.info(
                    "sparse_huge_profile ACTIVE: input_dim=%d active_root=%d "
                    "(late ReLUs use triangle unless ACT_HZ_LATE_RELU overrides)",
                    input_hz.dim, input_hz.ng,
                )
        self._lc_active = large_cls_active
        self._sparse_huge_profile_active = sparse_huge_auto
        self._convtranspose_triangle_profile_active = (
            convtranspose_count > 0
            and self.cfg["relu_method"] == "eq_lagr_v8"
            and os.environ.get("ACT_HZ_AUTO_CONVTRANSPOSE_TRIANGLE", "1")
            .strip().lower() in ("1", "true", "yes", "on")
        )
        if self._convtranspose_triangle_profile_active:
            self._stats["convtranspose_triangle_profile_active"] = True
            self._stats["convtranspose_count"] = int(convtranspose_count)
            logger.info(
                "convtranspose_triangle_profile ACTIVE: convtranspose=%d relus=%d "
                "(ReLUs use triangle unless ACT_HZ_AUTO_CONVTRANSPOSE_TRIANGLE=0)",
                convtranspose_count, total_relu,
            )
        self._relu_idx_map = relu_idx_map
        self._total_relu = total_relu

        small_dense_status = self._try_small_dense_lp(
            conv_count=conv_count, net=net, assert_layer=assert_layer,
        )
        if small_dense_status is not None:
            return small_dense_status

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
            logger.debug(
                "L%d %s  in: dim=%s ng=%s",
                L.id, tag_for_log, in_dim, in_ng,
            )
            if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
                in_desc = (
                    f"dim={hz_in.dim} ng={hz_in.ng} nb={hz_in.nb} nc={hz_in.nc}"
                    if hz_in is not None else "none"
                )
                print(f"[HZ-PROGRESS] start L{L.id} {L.kind} in={in_desc}", flush=True)
            try:
                # MaxPool: ACT cons_exporter doesn't generate constraints
                # for max-pool (it's not a linear op), so op_con is None
                # and the generic _dispatch fall-through would land in
                # _box_fallback (discarding all HZ correlation). Instead,
                # detect MAXPOOL2D explicitly here and route to the HZ
                # max_pool_node_evaluate via the hz_maxpool2d facade,
                # which preserves stable-winner rows exactly and falls
                # back to interval only on unstable blocks.
                if op_con is None and L.kind == "UPSAMPLE" and hz_in is not None:
                    try:
                        hz_out = _hz_upsample_nearest_nchw(hz_in, L.params)
                        self._stats[f"upsample_exact@{L.id}"] = True
                    except Exception as e:
                        self._stats[f"upsample_fallback@{L.id}"] = (
                            f"{type(e).__name__}: {e}"
                        )
                        hz_out = self._box_fallback(L, after, hz_from_bounds)
                elif op_con is None and L.kind == "MAXPOOL2D" and hz_in is not None:
                    try:
                        from act.back_end.hybridz_tf.hz_routing import hz_maxpool2d
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

                logger.debug(
                    "L%d done  out: dim=%d ng=%d nb=%d nc=%d",
                    L.id, hz_out.dim, hz_out.ng, hz_out.nb, hz_out.nc,
                )
                if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
                    print(
                        f"[HZ-PROGRESS] L{L.id} {L.kind} -> "
                        f"dim={hz_out.dim} ng={hz_out.ng} "
                        f"nb={hz_out.nb} nc={hz_out.nc}",
                        flush=True,
                    )
            except Exception as e:
                # Sound fallback on any per-layer failure
                self._stats[f"error@{L.id}"] = f"{type(e).__name__}: {e}"
                hz_out = self._box_fallback(L, after, hz_from_bounds)
                logger.warning(
                    "L%d FALLBACK (%s): %s", L.id, type(e).__name__, e,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

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
        max_replay_candidates = int(os.environ.get(
            "ACT_HZ_MAX_REPLAY_CANDIDATES", "1"
        ))
        xi_candidates: List[np.ndarray] = []
        try:
            if max_replay_candidates > 1:
                feas, xi_candidates = check_unsafe_candidates_for_act(
                    out_hz, assert_layer,
                    output_ids=output_ids,
                    timeout_s=self.cfg["timeout_s"],
                    max_candidates=max_replay_candidates,
                )
                self._stats["unsafe_lp_candidates_requested"] = int(
                    max_replay_candidates
                )
                self._stats["unsafe_lp_candidates_found"] = int(
                    len(xi_candidates)
                )
            else:
                feas, xi_star = check_unsafe_for_act(
                    out_hz, assert_layer,
                    output_ids=output_ids,
                    timeout_s=self.cfg["timeout_s"]
                )
                if xi_star is not None:
                    xi_candidates = [xi_star]
        except Exception as e:
            self._status = SolveStatus.UNKNOWN
            self._stats["feasibility_error"] = f"{type(e).__name__}: {e}"
            return self._status

        if feas == "infeasible":
            self._status = SolveStatus.UNSAT
            return self._status
        if feas == "timeout" and not xi_candidates:
            self._status = SolveStatus.UNKNOWN
            self._stats["timeout"] = True
            return self._status

        # ─── Phase 3: witness back to input space ───
        replay_attempts = 0
        replay_errors: List[str] = []
        for xi_star in xi_candidates:
            try:
                x_star = lp_witness_to_input(xi_star, input_hz)
            except Exception as e:
                replay_errors.append(f"witness:{type(e).__name__}: {e}")
                continue

            # ─── Phase 4: strict replay ───
            ok = True
            if self.cfg["strict_replay"]:
                try:
                    if self.cfg.get("onnx_path") and not getattr(net, "onnx_path", None):
                        try: net.onnx_path = self.cfg["onnx_path"]
                        except Exception: pass
                    ok = strict_replay_for_act(
                        net=net, x_star=x_star, assert_layer=assert_layer
                    )
                    self._last_net_for_replay = net  # R9.3 — see emit fn
                except Exception as e:
                    ok = False
                    replay_errors.append(f"replay:{type(e).__name__}: {e}")
            replay_attempts += 1
            if not ok:
                continue

            x_star_arr = np.asarray(x_star, dtype=np.float64).ravel()
            self._emit_sat_with_receipt(
                x_star=x_star_arr,
                assert_layer=assert_layer,
                source="hz_walker_lp",
                model_path_fallback=getattr(net, "onnx_path", None),
            )
            # Final cleanup: release intermediate HZ tensors held in var_to_hz
            try:
                del var_to_hz, out_hz, input_hz
                import torch as _t
                if _t.cuda.is_available() and self.cfg.get("device","cpu").startswith("cuda"):
                    _t.cuda.empty_cache()
            except Exception:
                pass
            return self._status

        if replay_attempts > 0:
            self._stats["replay_candidates_tried"] = int(replay_attempts)
            self._stats["phantom_rejected"] = True
            if replay_errors:
                self._stats["replay_errors"] = replay_errors[:4]
            if feas == "timeout":
                self._stats["timeout_after_replay_candidates"] = True
                return self._status
        self._status = SolveStatus.UNKNOWN
        if replay_errors:
            self._stats["witness_error"] = "; ".join(replay_errors[:4])
        return self._status

    # ----- Per-layer dispatch (cons tag -> HyZor or ACT op) -----
    def _dispatch(self, L, op_con, hz_in, multi_in_hzs, before, after, **ops):
        if op_con is None:
            if (
                L.kind == "SIGN"
                and hz_in is not None
                and os.environ.get("ACT_HZ_SIGN_HULL", "0").strip().lower()
                in ("1", "true", "yes", "on")
            ):
                try:
                    hz_out = _hz_sign_convex_hull(hz_in, before[L.id].bounds)
                    self._stats[f"sign_hull@{L.id}"] = True
                    return hz_out
                except Exception as e:
                    self._stats[f"sign_hull_fallback@{L.id}"] = (
                        f"{type(e).__name__}: {e}"
                    )
            if self._can_bypass_final_softmax(L, hz_in):
                self._stats[f"final_softmax_order_bypass@{L.id}"] = True
                return hz_in
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
            elif op == "convtranspose2d":
                out_shape = meta.get("output_shape")
                out_dim = (
                    int(out_shape[1] * out_shape[2] * out_shape[3])
                    if out_shape else 0
                )
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
        if op in ("upsample", "resize"):
            params = dict(getattr(L, "params", {}) or {})
            params.update(meta)
            return _hz_upsample_nearest_nchw(hz_in, params)
        if op == "softmax":
            if self._can_bypass_final_softmax(L, hz_in):
                self._stats[f"final_softmax_order_bypass@{L.id}"] = True
                return hz_in
            return self._box_fallback(L, after, ops["hz_from_bounds"])
        if op == "convtranspose2d":
            params = dict(getattr(L, "params", {}) or {})
            params.update(meta)
            return _hz_convtranspose2d_native(hz_in, params)
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
            # large_cls_proof_mode: triangle for early relus, eq_lagr_v8 for last N.
            # If ACT_HZ_LATE_RELU is set, use that method (e.g.
            # "convex_hull_cont") for late-layer experiments. The raw
            # per-neuron LP hull matches relaxed eq_lagr_v8; the full routed
            # pipelines differ because eq_lagr_v8 also runs PEE.
            method = self.cfg["relu_method"]
            ridx = self._relu_idx_map.get(L.id, 0)
            if getattr(self, "_convtranspose_triangle_profile_active", False):
                method = "triangle"
            if getattr(self, "_lc_active", False):
                eq_last = self.cfg["large_cls_eq_layers"]
                if ridx <= self._total_relu - eq_last:
                    # ACT_HZ_MID_RELU lets experiments substitute the
                    # default triangle for middle-layer HZono ReLUs
                    # (e.g. "selective_chull"). Sound regardless of choice
                    # since all dispatch targets are sound encodings.
                    mid_method = os.environ.get("ACT_HZ_MID_RELU", "").strip()
                    method = mid_method if mid_method else "triangle"
                else:
                    late_method = os.environ.get("ACT_HZ_LATE_RELU", "").strip()
                    if late_method:
                        method = late_method
                    elif getattr(self, "_sparse_huge_profile_active", False):
                        method = "triangle"
            # Expose the current relu_idx so selective_chull dispatches can
            # look up per-layer masks (set by property-direction driver).
            from act.back_end.hybridz_tf.algorithms.relu_methods import (
                set_current_relu_idx,
            )
            set_current_relu_idx(ridx)
            return ops["hz_apply_relu_v8"](hz_in, method=method)
        if op == "lrelu":
            # Accept either ``alpha`` (legacy) or ``negative_slope`` (PyTorch
            # converter); fall back to ONNX/torch default of 0.01.
            alpha = meta.get("alpha", meta.get("negative_slope", 0.01))
            return ops["hz_apply_leaky_relu_v8"](hz_in, alpha=float(alpha))

        # ── ACT ops (sigmoid/tanh K-piece -- ACT innovation) ──
        # Keep the PWL HZ encoding for moderately-wide smooth activations
        # when the incoming HZ is still simple.  This matters for
        # dist_shift: its 784-wide Sigmoid arrives with only a few root/ReLU
        # generators, and box fallback destroys the decisive correlations.
        # If the state is already complex, fail back to box to avoid OOM.
        if op == "sigmoid":
            cap = int(os.environ.get("HYZOR_SIGMOID_DIM_CAP", "2048"))
            max_gens = int(os.environ.get("HYZOR_SIGMOID_MAX_IN_GENS", "128"))
            max_cons = int(os.environ.get("HYZOR_SIGMOID_MAX_IN_CONS", "1024"))
            if (
                int(hz_in.dim) > cap
                or int(hz_in.ng) + int(hz_in.nb) > max_gens
                or int(hz_in.nc) > max_cons
            ):
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            return ops["act_hz_apply_sigmoid"](hz_in, K=self.cfg["sigmoid_K"])
        if op == "tanh":
            cap = int(os.environ.get("HYZOR_TANH_DIM_CAP", "2048"))
            max_gens = int(os.environ.get("HYZOR_TANH_MAX_IN_GENS", "128"))
            max_cons = int(os.environ.get("HYZOR_TANH_MAX_IN_CONS", "1024"))
            if (
                int(hz_in.dim) > cap
                or int(hz_in.ng) + int(hz_in.nb) > max_gens
                or int(hz_in.nc) > max_cons
            ):
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            return ops["act_hz_apply_tanh"](hz_in, K=self.cfg["tanh_K"])

        # ── Shape ops ──
        if op in ("flatten", "reshape", "transpose", "squeeze",
                  "unsqueeze", "tile", "expand"):
            return hz_in
        # SLICE is a linear permutation/selection; use exact HZ transfer
        # when possible (no relaxation), fall back to box on mismatch/error.
        if op == "slice":
            try:
                return _hz_slice_exact(hz_in, meta)
            except Exception as _e:
                self._stats[f"slice_exact_fallback@{L.id}"] = (
                    f"{type(_e).__name__}: {str(_e)[:120]}"
                )
                return self._box_fallback(L, after, ops["hz_from_bounds"])
        # GATHER (ONNX) is row selection along an axis; exact in HZ.
        if op == "gather":
            try:
                return _hz_gather_exact(hz_in, meta)
            except Exception as _e:
                self._stats[f"gather_exact_fallback@{L.id}"] = (
                    f"{type(_e).__name__}: {str(_e)[:120]}"
                )
                return self._box_fallback(L, after, ops["hz_from_bounds"])

        # ── Fallback ──
        return self._box_fallback(L, after, ops["hz_from_bounds"])

    # ----- Helpers -----

    def _can_bypass_final_softmax(self, L, hz_in) -> bool:
        if hz_in is None or not getattr(self, "_final_softmax_order_only", False):
            return False
        if L.kind != "SOFTMAX":
            return False
        out_ids = tuple(getattr(L, "out_vars", ()))
        if out_ids != getattr(self, "_final_output_ids", ()):
            return False
        if int(hz_in.dim) != len(out_ids):
            return False
        axis = int((getattr(L, "params", {}) or {}).get("axis", -1))
        # ACT stores the final classifier vector flattened, so the only
        # sound default bypass is softmax over the last/only feature axis.
        return axis in (-1, 0, 1)

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
        """Apply sound Girard reduction when ``girard_cap`` is achievable.

        For output dimension at least the configured cap, independent
        diagonal slack cannot fit inside that cap; ``reduce_constraints``
        keeps the larger representation rather than shrinking unsoundly.

        Property-facing tail preservation: when ``hz.dim`` is below
        non-zero ``tail_preserve_dim``, skip the reduce so that
        correlation is preserved through the small classifier tail.
        This is always sound (skipping a widening operator is identity).

        Controlled result: tail preservation alone did not improve the
        default eq_lagr_v8 path; paired with convex_hull_cont it materially
        tightened margins. It is therefore an opt-in research policy rather
        than a default verification behavior.
        """
        cap = self.cfg["girard_cap"]
        if int(hz.ng) <= cap:
            return hz
        tail_dim = int(self.cfg.get("tail_preserve_dim", 0) or 0)
        if tail_dim > 0 and int(hz.dim) < tail_dim:
            return hz
        try:
            return hz.reduce_constraints(
                ng_budget=cap,
                constraint_keep_weight=float(
                    self.cfg.get("constraint_keep_weight", 0.0) or 0.0
                ),
            )
        except Exception:
            return hz

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


def verify_once_hz(
    net,
    *,
    solver: "HZVerifier",
    timelimit: Optional[float] = None,
    batch_lane: int = 0,
) -> Tuple[str, Optional[np.ndarray], Dict[str, Any]]:
    """Single-instance HZ verification driver.

    Analyzes ``net`` end-to-end and drives ``HZVerifier.consume_cons`` on
    the resulting cons IR. Unlike ``verify_once`` in ``act.back_end.verifier``,
    this routes through the HZ cons-walker rather than the batched
    ``solve_batch`` API; HZVerifier walks the cons IR natively.

    Args:
        net: ACT Net whose first layer is INPUT, last is ASSERT.
            INPUT_SPEC may be batched ``[B, *shape]``; this driver
            verifies lane ``batch_lane`` only (default 0; raises if B>1
            without an explicit lane).
        solver: ``HZVerifier`` instance.
        timelimit: optional wall-clock budget (seconds).
        batch_lane: index into the batched INPUT_SPEC to verify.

    Returns:
        ``(status, counterexample_input, stats)``
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
            f"verify_once_hz: batch_lane={batch_lane} out of "
            f"range [0, {B})"
        )

    # Exact singleton-input fast path. If the current VNNLIB query has a
    # zero-width input box, the reachable set is one concrete network output.
    # Strict replay of that single input decides the query without running
    # interval/HZ analysis. This is a structured exact HZ degenerate case,
    # not random sampling or a fallback verifier.
    pure_box_input = (
        len(spec_layers) == 1
        and str(getattr(spec_layers[0], "params", {}).get("kind", "")).upper()
        == "BOX"
    )
    if (
        pure_box_input
        and os.environ.get("ACT_HZ_SINGLETON_FASTPATH", "1").strip().lower() in (
            "1", "true", "yes", "on",
        )
    ):
        lane_lb = seed_bounds.lb[batch_lane].flatten()
        lane_ub = seed_bounds.ub[batch_lane].flatten()
        if (
            torch.isfinite(lane_lb).all().item()
            and torch.isfinite(lane_ub).all().item()
            and torch.all(lane_lb == lane_ub).item()
        ):
            try:
                if hasattr(solver, "cfg") and solver.cfg.get("onnx_path"):
                    try:
                        net.onnx_path = solver.cfg["onnx_path"]
                    except Exception:
                        pass
                assert_layer_b1 = _slice_assert_layer_lane(assert_layer, batch_lane)
                x_single = lane_lb.detach().cpu().numpy().reshape(-1)
                unsafe = strict_replay_for_act(
                    net=net, x_star=x_single, assert_layer=assert_layer_b1,
                )
                if unsafe:
                    solver._emit_sat_with_receipt(
                        x_star=x_single,
                        assert_layer=assert_layer_b1,
                        source="singleton_input",
                        model_path_fallback=(
                            solver.cfg.get("onnx_path")
                            if hasattr(solver, "cfg") else None
                        ),
                    )
                else:
                    solver._status = SolveStatus.UNSAT
                    solver._stats["singleton_input_certified"] = True
                return solver._status, (
                    x_single if solver._status == SolveStatus.SAT else None
                ), {
                    "status": solver._status,
                    "ncons": 0,
                    "solver": "hyzor",
                    **(solver.stats() if hasattr(solver, "stats") else {}),
                }
            except Exception as e:
                if hasattr(solver, "_stats"):
                    solver._stats["singleton_fastpath_error"] = (
                        f"{type(e).__name__}: {e}"
                    )

    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    before, after, globalC = analyze(net, entry_id, entry_fact)
    if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
        print(f"[HZ-PROGRESS] analyze done globalC={len(globalC)}", flush=True)
    validate_constraints(globalC, after, net)
    if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
        print("[HZ-PROGRESS] validate_constraints done", flush=True)

    # HyZor's consume_cons predates batch-native analyze and expects
    # single-lane (1-D) bounds in before/after Facts. Slice lane
    # ``batch_lane`` so consume_cons sees the same input shape it always
    # has. globalC may also carry per-batch box rows; rewrite them to
    # the single-lane view.
    before_b1, after_b1 = _slice_facts_lane(before, batch_lane), _slice_facts_lane(after, batch_lane)
    globalC_b1 = _slice_globalC_lane(globalC, batch_lane)
    assert_layer_b1 = _slice_assert_layer_lane(assert_layer, batch_lane)
    if os.environ.get("ACT_HZ_LAYER_PROGRESS", "0") == "1":
        print(f"[HZ-PROGRESS] slice done globalC_b1={len(globalC_b1)}", flush=True)

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


def verify_all_lanes_hz(
    net,
    *,
    solver: "HZVerifier",
    timelimit: Optional[float] = None,
) -> List[Tuple[str, Optional[np.ndarray], Dict[str, Any]]]:
    """Batched-input HZ verification: analyze ONCE, verify each lane.

    Mirrors ``verify_once_hz`` but takes a ``net`` whose INPUT_SPEC carries
    ``B>=1`` lanes and returns a list of ``(status, ce_input, stats)`` of
    length ``B``. The forward analyze pass is shared across lanes (batched
    HZ propagation on GPU); ``consume_cons`` is still per-lane and runs
    sequentially because the cons-walker / LP feasibility path predates
    batch-native input. The solver is reset between lanes via ``begin()``
    so per-lane state doesn't leak.

    Single-lane callers should still use ``verify_once_hz`` — it preserves
    the single-Verdict return shape that older drivers depend on.
    """
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import set_transfer_function_mode
    from act.back_end.verifier import (
        find_entry_layer_id, get_input_ids, get_output_ids,
        gather_input_spec_layers, get_assert_layer,
        seed_from_input_specs, add_all_input_specs, validate_constraints,
    )

    _tf_mode = os.environ.get("HYZOR_TF_MODE", "").strip().lower()
    if _tf_mode in ("interval", "hybridz"):
        set_transfer_function_mode(_tf_mode)

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2:
        seed_bounds = Bounds(
            lb=seed_bounds.lb.unsqueeze(0), ub=seed_bounds.ub.unsqueeze(0)
        )
    B = int(seed_bounds.lb.shape[0])

    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    before, after, globalC = analyze(net, entry_id, entry_fact)
    validate_constraints(globalC, after, net)

    if timelimit is not None and hasattr(solver, "cfg"):
        solver.cfg["timeout_s"] = float(timelimit)

    results: List[Tuple[str, Optional[np.ndarray], Dict[str, Any]]] = []
    for lane in range(B):
        before_b1 = _slice_facts_lane(before, lane)
        after_b1 = _slice_facts_lane(after, lane)
        globalC_b1 = _slice_globalC_lane(globalC, lane)
        assert_layer_b1 = _slice_assert_layer_lane(assert_layer, lane)

        # Reset per-lane solver state. Same HZVerifier object is reused so
        # cfg (relu_method / timeout / strict_replay etc.) stays consistent
        # across lanes, but witness/stats from previous lane don't leak.
        solver.begin()

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
            "batch_lane": lane, "batch_size": B,
        }
        try:
            stats.update(solver.stats() if hasattr(solver, "stats") else {})
        except Exception:
            pass
        results.append((st, ce_input, stats))

    return results


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


def check_unsafe_candidates_for_act(
    out_hz: HZono,
    assert_layer,
    *,
    output_ids=None,
    timeout_s: float = 30.0,
    max_candidates: int = 4,
) -> Tuple[str, List[np.ndarray]]:
    """Return several structured unsafe LP witnesses for strict replay.

    This is the same forward HZ factor LP used by ``check_unsafe_for_act``.
    It does not sample input space and does not use gradients; it simply keeps
    walking later TOP1/MARGIN disjuncts when an earlier LP-feasible assignment
    later proves to be a strict-replay phantom.
    """
    max_candidates = max(1, int(max_candidates))
    kind = _kind_str(assert_layer.params.get("kind"))
    if kind not in ("TOP1_ROBUST", "MARGIN_ROBUST"):
        st, x = check_unsafe_for_act(
            out_hz, assert_layer, output_ids=output_ids, timeout_s=timeout_s
        )
        return st, ([x] if x is not None else [])

    prob = _build_factor_lp(out_hz)
    p, q = prob["p"], prob["q"]
    Gc, Gb, c_vec = prob["Gc"], prob["Gb"], prob["c"]
    t = int(_to_np_f64(assert_layer.params["y_true"]).reshape(-1)[0])
    margin = 0.0
    if kind == "MARGIN_ROBUST":
        margin = float(_to_np_f64(assert_layer.params["margin"]).reshape(-1)[0])
    n_out = c_vec.size

    def _row_to_obj_y(coef: np.ndarray) -> Tuple[np.ndarray, float]:
        obj_row = np.concatenate([coef @ Gc, coef @ Gb], axis=0)
        obj_const = float(coef @ c_vec)
        return obj_row, obj_const

    t0 = time.perf_counter()
    def _remaining():
        return max(0.05, timeout_s - (time.perf_counter() - t0))

    diffY = np.concatenate([Gc, Gb], axis=1)
    diff_rows = diffY - diffY[t:t+1]
    diff_c = c_vec - c_vec[t]
    ub_diff_tight = diff_c + np.abs(diff_rows).sum(axis=1)
    threshold = -margin
    candidates = [
        j for j in range(n_out)
        if j != t and ub_diff_tight[j] >= threshold
    ]
    candidates.sort(key=lambda j: -ub_diff_tight[j])

    witnesses: List[np.ndarray] = []
    saw_timeout = False
    for j in candidates:
        if len(witnesses) >= max_candidates:
            break
        coef = np.zeros(n_out)
        coef[j] = 1.0
        coef[t] = -1.0
        obj_row, obj_const = _row_to_obj_y(coef)
        st, x = _lp_feas_or_minimize(
            prob, obj_row, rhs_threshold=threshold - obj_const,
            sense="maximize", timeout_s=_remaining(),
        )
        if st == "feasible" and x is not None:
            witnesses.append(x)
        elif st == "timeout":
            saw_timeout = True
            break

    if witnesses:
        return ("timeout" if saw_timeout else "feasible"), witnesses
    return ("timeout" if saw_timeout else "infeasible"), []


def lp_witness_to_input(xi_star: np.ndarray, input_hz) -> np.ndarray:
    """Map a factor-space witness xi_star back to input space.

    For a network whose first HZ corresponds to the input box, the
    "input" coordinates are ``c_in + Gc_in @ xi_c + Gb_in @ xi_b`` for
    the relevant factor slots. If ``input_hz`` is a BoxHZ-style HZono
    (diagonal Gc, no Gb), the first p_in entries of ``xi_star`` are the
    factor coordinates of the input pixels.

    A real ``BoxHZ`` stores bounds rather than factor columns. Once that
    representation is materialised or reduced downstream, a final relaxed
    LP witness has no defined inverse map to the original input factors.
    Return its center as a concrete in-box replay candidate; strict replay
    remains the sole authority for reporting SAT.
    """
    if not hasattr(input_hz, "Gc"):
        if hasattr(input_hz, "bounds"):
            lb, ub = input_hz.bounds()
            lb_np = _to_np_f64(lb).reshape(-1)
            ub_np = _to_np_f64(ub).reshape(-1)
            return (lb_np + ub_np) / 2.0
        raise TypeError(
            f"Cannot obtain a concrete input candidate from "
            f"{type(input_hz).__name__}"
        )

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



__all__ = ["strict_replay_for_act", "reportable_verdict_for_cli"]


def reportable_verdict_for_cli(solver: "HZVerifier", internal_status: str) -> str:
    """Translate a solver's internal verdict into a CLI-reportable label.

    Honesty rule (advisor 2026-05-24 Round 4):
        Internal ``_status == SAT`` is the mathematical truth (a real
        adversary was found and replay-confirmed). In formal mode that
        SAT may still NOT be admissible into paper-grade FAL counts if
        its audit ledger entry is incomplete. This helper surfaces that
        distinction:

        non-formal mode → normalize {UNSAT→CERTIFIED, SAT→FALSIFIED}
        formal mode:
            SAT + formal_result=REPORTABLE_FALSIFIED → FALSIFIED
            SAT + formal_result=ERROR_*              → that ERROR_*
            SAT + formal_result missing/None         → ERROR_NO_FORMAL_RESULT
            UNSAT                                    → CERTIFIED
            UNKNOWN                                  → UNKNOWN
            anything else                            → ERROR_UNEXPECTED
    """
    import os as _os
    formal = _os.environ.get("ACT_FAL_RECEIPT_FORMAL", "").strip().lower() \
        in ("1", "true", "yes", "on")
    if internal_status == "UNSAT":
        return "CERTIFIED"
    if internal_status == "UNKNOWN":
        return "UNKNOWN"
    if internal_status != "SAT":
        return f"ERROR_UNEXPECTED_{internal_status}"
    # SAT path
    if not formal:
        return "FALSIFIED"
    fr = solver._stats.get("formal_result")
    if fr == "REPORTABLE_FALSIFIED":
        return "FALSIFIED"
    if fr and fr.startswith("ERROR_"):
        return fr
    return "ERROR_NO_FORMAL_RESULT"


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


def _x_star_in_input_box(net, x_arr) -> tuple[bool, str]:
    """ROUND 9.3 (advisor 2026-05-25 P0): fail-CLOSED input-box gate.

    Returns ``(holds, reason)`` where:
      - ``holds=True``  iff x_arr is elementwise inside [lb, ub] for at
        least one InputSpec layer with a usable bounds pair.
      - ``holds=False`` for ANY of the following (R9.3 fail-closed):
          * no InputSpec layer with usable bounds was found
          * LayerKind import failed
          * lb/ub missing on the (sole) input spec
          * lb/ub shape doesn't match x_flat
          * any (lb, ub) pair contains a NaN
          * x violates any finite bound on the (sole) input spec

    The earlier R9.2 returned True on shape-mismatch or missing spec —
    that was FAIL-OPEN and admitted "validated" witnesses for which the
    box was never actually checked. R9.3 closes that hole: a SAT/FAL is
    only emitted when input-domain validity is positively established.

    ``reason`` is a short tag for the caller to record in the receipt:
        "ok" | "no_input_spec" | "missing_bounds" | "shape_mismatch"
      | "nan_bounds" | "out_of_box" | "layerkind_unavailable"
    """
    try:
        from act.back_end.layer_schema import LayerKind
    except Exception:
        return False, "layerkind_unavailable"
    x_flat = _to_np(x_arr).astype(np.float64, copy=False).reshape(-1)
    found_spec_with_bounds = False
    for L in net.layers:
        if L.kind != LayerKind.INPUT_SPEC.value:
            continue
        lb = L.params.get("lb")
        ub = L.params.get("ub")
        if lb is None or ub is None:
            return False, "missing_bounds"
        # Formal replay runs after propagation on the requested device.
        # CUDA-hosted INPUT_SPEC bounds must be copied to host before the
        # NumPy domain check; otherwise valid GPU witnesses are rejected.
        lb_np = _to_np(lb).astype(np.float64, copy=False).reshape(-1)
        ub_np = _to_np(ub).astype(np.float64, copy=False).reshape(-1)
        if lb_np.shape != x_flat.shape or ub_np.shape != x_flat.shape:
            return False, "shape_mismatch"
        if np.any(np.isnan(lb_np)) or np.any(np.isnan(ub_np)):
            return False, "nan_bounds"
        tol = 0.0
        lo_ok = (x_flat >= lb_np - tol) | ~np.isfinite(lb_np)
        hi_ok = (x_flat <= ub_np + tol) | ~np.isfinite(ub_np)
        if not bool(np.all(lo_ok & hi_ok)):
            return False, "out_of_box"
        found_spec_with_bounds = True
    if not found_spec_with_bounds:
        return False, "no_input_spec"
    return True, "ok"


def strict_replay_for_act(*, net, x_star, assert_layer) -> bool:
    """Strict (zero-tol) witness replay for the ACT verifier.

    Args:
        net: ACT ``Net`` — may carry ``.onnx_path`` for ORT fast path.
        x_star: witness in input space, ``np.ndarray`` of length ``n_in``.
        assert_layer: ACT ASSERT layer carrying spec params.

    Returns:
        ``True`` iff
          (1) x_star lies elementwise inside every declared InputSpec
              box on the net (ROUND 9.2 gate), AND
          (2) the model's output at ``x_star`` violates the spec
              (i.e. is in the unsafe set).
        Used to confirm SAT witnesses; if either fails, the LP cert is
        spurious and the verdict downgrades.
    """
    x_arr = np.asarray(x_star, dtype=np.float64)

    # ROUND 9.3: fail-CLOSED input-box gate. Reject witnesses outside
    # the box AND any case where the box itself cannot be validated
    # (no InputSpec, shape mismatch, NaN bounds, missing bounds, etc.).
    # P1 audit caught 31/49 sat_relu FAL with x outside [0,1]^100 (R9.2
    # caught those); R9.3 also closes the fail-open path that returned
    # True on missing-spec / shape-mismatch.
    holds, reason = _x_star_in_input_box(net, x_arr)
    # Stash on net so callers (write_receipt) can record this in the
    # audit ledger. Best-effort: ignore if net is not a SimpleNamespace.
    try:
        net._last_input_box_check = (holds, reason)
    except Exception:
        pass
    if not holds:
        return False

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
