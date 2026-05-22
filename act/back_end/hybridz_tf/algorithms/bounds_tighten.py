#===- act/back_end/hybridz_tf/algorithms/bounds_tighten.py - HZ Bounds Tightening Cascade -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Three-tier bound tightening: unconstrained interval hull (Tier-1) →
#   batched Adam Lagrangian dual (Tier-2) → QR-reduced HiGHS LP after
#   equality elimination (Tier-3). Sound at every tier.
#
#===---------------------------------------------------------------------===#

"""Bounds-tightening algorithms for HZono (Y2 Stage 4 port).

HyZor's ``HybridZReLU.forward`` for ``eq_lagr_v8`` computes tight
pre-ReLU bounds via a cascade (UNC -> Lagrangian dual -> selective LP)
then calls ``applyReLU_eq_native(external_bounds=(lb, ub))`` with those
bounds. Without this step, ACT's HZono ReLU uses unconstrained interval
bounds and classifies more neurons as "unstable", which produces a
larger HZ that's more likely to give "unknown" verdicts on cifar/tiny
mid-stage layers.

This module ports HyZor's bound cascade to operate on ACT's HZono /
HybridZonotope-like duck-typed objects. The cascade has three tiers:

  Tier 1 - hz_bounds_unconstrained: c ± |Gc|·1 ± |Gb|·1
  Tier 2 - hz_bounds_hz_dual: closed-form Lagrangian dual (Adam)
  Tier 3 - hz_bounds_eq_elim_lp: selective HiGHS LP after QR elim

The cascade picks the cheapest sound bound from the available tools
depending on whether nc > 0, eq_mask has equalities, etc.

Also exports ``hz_intersect_box(hz, lb, ub)`` which appends 2n
inequality rows restricting the HZ to a box.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple

import torch

from act.back_end.solver.solver_hz import HZono, _eq_mask_of

logger = logging.getLogger(__name__)


__all__ = [
    "hz_bounds_unconstrained",
    "hz_intersect_box",
    "hz_bounds_eq_elim_lp",
    "hz_bounds_hz_dual",
]


# ---------------------------------------------------------------------------
# Persistent HiGHS solver (warm-start across many LPs with same constraints)
# ---------------------------------------------------------------------------


class _HighspyWarmSolver:
    """Persistent HiGHS solver with warm-start via objective change.

    Faithful port of HyZor ``_HighspyWarmSolver`` (HybridZonotope.py:20).
    Used by ``hz_bounds_eq_elim_lp`` to solve many LPs that share
    constraints but differ in objective — ~10-20x speedup vs scipy
    cold-start. Falls back gracefully when highspy unavailable.
    """
    _hp = None

    @classmethod
    def _ensure_hp(cls):
        if cls._hp is None:
            try:
                import highspy as _hp
                cls._hp = _hp
            except ImportError:
                return None
        return cls._hp

    def __init__(self, A_ub, b_ub, A_eq, b_eq, bounds_lb, bounds_ub):
        import numpy as np
        hp = self._ensure_hp()
        if hp is None:
            self.h = None
            return
        self._hp = hp
        nv = int(len(bounds_lb))
        A_blocks = []
        rl_blocks = []
        ru_blocks = []
        if A_ub is not None and A_ub.size > 0:
            A_blocks.append(np.asarray(A_ub, dtype=np.float64))
            rl_blocks.append(np.full(A_ub.shape[0], -hp.kHighsInf))
            ru_blocks.append(np.asarray(b_ub, dtype=np.float64))
        if A_eq is not None and A_eq.size > 0:
            A_blocks.append(np.asarray(A_eq, dtype=np.float64))
            rl_blocks.append(np.asarray(b_eq, dtype=np.float64))
            ru_blocks.append(np.asarray(b_eq, dtype=np.float64))
        if A_blocks:
            A_full = np.vstack(A_blocks)
            row_lower = np.concatenate(rl_blocks)
            row_upper = np.concatenate(ru_blocks)
            n_row = int(A_full.shape[0])
        else:
            A_full = np.zeros((0, nv), dtype=np.float64)
            row_lower = np.zeros(0, dtype=np.float64)
            row_upper = np.zeros(0, dtype=np.float64)
            n_row = 0

        # Build CSC sparse triple from dense A_full (column-major).
        col_starts = [0]
        row_indices = []
        values = []
        for j in range(nv):
            col = A_full[:, j]
            nz = np.nonzero(col)[0]
            if nz.size > 0:
                row_indices.extend(nz.tolist())
                values.extend(col[nz].tolist())
            col_starts.append(len(values))

        h = hp.Highs()
        h.silent()
        try:
            h.setOptionValue("presolve", "off")
            h.setOptionValue("solver", "simplex")
        except Exception:
            pass
        lp = hp.HighsLp()
        lp.num_col_ = nv
        lp.num_row_ = n_row
        lp.col_cost_ = [0.0] * nv
        lp.col_lower_ = np.asarray(bounds_lb, dtype=np.float64).tolist()
        lp.col_upper_ = np.asarray(bounds_ub, dtype=np.float64).tolist()
        lp.row_lower_ = row_lower.tolist()
        lp.row_upper_ = row_upper.tolist()
        lp.a_matrix_.format_ = hp.MatrixFormat.kColwise
        lp.a_matrix_.start_ = col_starts
        lp.a_matrix_.index_ = row_indices
        lp.a_matrix_.value_ = values
        h.passModel(lp)
        self.h = h
        self.nv = nv
        self._col_idx = np.arange(nv, dtype=np.int32)

    def is_ok(self):
        return self.h is not None

    def solve_min(self, obj):
        """Returns ``(status, fun)`` where status is
        ``"optimal"|"infeasible"|"fail"``."""
        import numpy as np
        hp = self._hp
        h = self.h
        h.changeColsCost(self.nv, self._col_idx,
                         np.asarray(obj, dtype=np.float64).tolist())
        h.run()
        sm = h.getModelStatus()
        if sm == hp.HighsModelStatus.kOptimal:
            return ("optimal", float(h.getObjectiveValue()))
        if sm in (hp.HighsModelStatus.kInfeasible,
                  hp.HighsModelStatus.kPrimalInfeasible):
            return ("infeasible", None)
        return ("fail", None)


# ---------------------------------------------------------------------------
# Tier 3: hz_bounds_eq_elim_lp (QR-reduced LP)
# ---------------------------------------------------------------------------


@torch.no_grad()
def hz_bounds_eq_elim_lp(
    hz,
    indices: Optional[torch.Tensor] = None,
    base_lb: Optional[torch.Tensor] = None,
    base_ub: Optional[torch.Tensor] = None,
    classify_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bounds via equality-variable elimination + reduced LP.

    Faithful port of HyZor ``HybridZonotope._bounds_eq_elim_lp``
    (HybridZonotope.py:2457).

    For HZ with equality constraints (from eq_lagr encoding):
      1. QR-eliminate dependent ``xi_c`` columns using ``Ac_eq``
      2. Solve a reduced LP per dimension on ``[xi_free; xi_b]``

    The reduced system has ``ng_free = ng - rank(Ac_eq)`` free continuous
    vars, making LP much faster than the full system. Binary domain is
    relaxed to ``[-1, +1]`` (sound LP relaxation).

    Args:
        hz: HZono-like with ``.c, .Gc, .Gb, .Ac, .Ab, .b, .eq_mask``.
        indices: optional 1D tensor of dim indices to bound (default: all).
        base_lb, base_ub: optional fast bounds used in ``classify_only``.
        classify_only: when True with base_lb/ub, only solve the bound
            whose sign is uncertain (saves one LP per neuron when the
            other side is already 0-classified).

    Returns ``(lb, ub)`` as ``(k, 1)`` tensors (``k = len(indices) or n``).
    """
    import numpy as np
    from scipy.optimize import linprog as _linprog
    from scipy.linalg import qr as _qr, solve_triangular
    import os as _os
    import time as _time

    _prof = _os.environ.get("HYZOR_EQ_ELIM_LP_PROF", "0") == "1"
    _t0 = _time.perf_counter()
    _calls = 0
    _saved = 0

    n = int(hz.c.shape[0])
    nc = int(hz.b.shape[0])
    ng = int(hz.Gc.shape[1])
    nb = int(hz.Gb.shape[1])
    dtype = hz.c.dtype
    device = hz.c.device

    if indices is None:
        solve_idx = list(range(n))
    else:
        solve_idx = [int(i) for i in indices.view(-1).detach().cpu().tolist()]

    base_lb_np = (base_lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
                  if base_lb is not None else None)
    base_ub_np = (base_ub.detach().cpu().numpy().reshape(-1).astype(np.float64)
                  if base_ub is not None else None)
    classify_mode = bool(classify_only and base_lb_np is not None and base_ub_np is not None)

    # No constraints OR no equality rows → fall back to unconstrained bounds.
    if nc == 0:
        lb_unc, ub_unc = hz_bounds_unconstrained(hz)
        if indices is None:
            return lb_unc, ub_unc
        idx_t = indices.view(-1).to(device=device)
        return lb_unc[idx_t], ub_unc[idx_t]

    em = (hz.eq_mask if (hz.eq_mask is not None
                          and int(hz.eq_mask.numel()) == nc)
          else torch.zeros(nc, dtype=torch.bool, device=device))
    n_eq = int(em.sum().item())
    if n_eq == 0:
        lb_unc, ub_unc = hz_bounds_unconstrained(hz)
        if indices is None:
            return lb_unc, ub_unc
        idx_t = indices.view(-1).to(device=device)
        return lb_unc[idx_t], ub_unc[idx_t]

    # Move everything to numpy float64.
    Gc = hz.Gc.detach().cpu().double().numpy()
    Gb = hz.Gb.detach().cpu().double().numpy()
    c_vec = hz.c.detach().cpu().double().numpy().ravel()
    Ac = hz.Ac.detach().cpu().double().numpy()
    Ab = hz.Ab.detach().cpu().double().numpy()
    bv = hz.b.detach().cpu().double().numpy().ravel()
    eq_np = em.detach().cpu().numpy()

    Ac_eq = Ac[eq_np]; Ab_eq = Ab[eq_np]; b_eq = bv[eq_np]
    ineq_np = ~eq_np
    Ac_ineq = Ac[ineq_np]; Ab_ineq = Ab[ineq_np]; b_ineq = bv[ineq_np]
    n_ineq = Ac_ineq.shape[0]

    # Pivoted QR on Ac_eq (continuous-block only).
    Q, R, piv = _qr(Ac_eq, pivoting=True)
    rank = int(np.sum(np.abs(np.diag(R[:min(n_eq, ng), :])) > 1e-10))
    if rank == 0:
        lb_unc, ub_unc = hz_bounds_unconstrained(hz)
        if indices is None:
            return lb_unc, ub_unc
        idx_t = indices.view(-1).to(device=device)
        return lb_unc[idx_t], ub_unc[idx_t]

    dep_idx = piv[:rank]
    free_idx = piv[rank:]
    ng_free = len(free_idx)

    R_dep = R[:rank, :rank]
    Qt = Q[:, :rank].T
    Ac_eq_free = Ac_eq[:, free_idx]

    M = solve_triangular(R_dep, Qt)         # (rank, n_eq)
    M_b = M @ b_eq                          # (rank,)
    M_Ab = M @ Ab_eq                        # (rank, nb)
    M_Ac_free = M @ Ac_eq_free              # (rank, ng_free)

    Gc_dep = Gc[:, dep_idx]
    Gc_free = Gc[:, free_idx]

    Gc_red = Gc_free - Gc_dep @ M_Ac_free   # (n, ng_free)
    Gb_red = Gb - Gc_dep @ M_Ab             # (n, nb)
    c_red = c_vec + Gc_dep @ M_b            # (n,)

    # Box constraints on eliminated xi_dep in [-1, +1]:
    A_box_ub_free = -M_Ac_free
    A_box_ub_b = -M_Ab
    b_box_ub = 1.0 - M_b
    A_box_lb_free = M_Ac_free
    A_box_lb_b = M_Ab
    b_box_lb = 1.0 + M_b

    if n_ineq > 0:
        Ac_ineq_dep = Ac_ineq[:, dep_idx]
        Ac_ineq_free = Ac_ineq[:, free_idx]
        Ac_red_free = Ac_ineq_free - Ac_ineq_dep @ M_Ac_free
        Ab_red = Ab_ineq - Ac_ineq_dep @ M_Ab
        b_red_arr = b_ineq - Ac_ineq_dep @ M_b
    else:
        Ac_red_free = np.empty((0, ng_free))
        Ab_red = np.empty((0, nb))
        b_red_arr = np.empty(0)

    nv = ng_free + nb
    A_all = np.vstack([
        np.hstack([A_box_ub_free, A_box_ub_b]),
        np.hstack([A_box_lb_free, A_box_lb_b]),
        np.hstack([Ac_red_free, Ab_red]) if n_ineq > 0 else np.empty((0, nv)),
    ])
    b_all = np.concatenate([b_box_ub, b_box_lb, b_red_arr])
    bounds = [(-1.0, 1.0)] * nv

    lb_out = np.full(len(solve_idx), -np.inf)
    ub_out = np.full(len(solve_idx), np.inf)

    # Optional highspy warm-start.
    _highspy_on = _os.environ.get("HYZOR_SCIPY_LOOP_HIGHSPY", "1") == "1"
    _hp_solver = None
    if _highspy_on:
        bounds_lb_arr = np.array([b[0] for b in bounds], dtype=np.float64)
        bounds_ub_arr = np.array([b[1] for b in bounds], dtype=np.float64)
        try:
            _hp_solver = _HighspyWarmSolver(
                A_ub=A_all, b_ub=b_all, A_eq=None, b_eq=None,
                bounds_lb=bounds_lb_arr, bounds_ub=bounds_ub_arr,
            )
            if not _hp_solver.is_ok():
                _hp_solver = None
        except Exception:
            _hp_solver = None

    for out_pos, i in enumerate(solve_idx):
        obj = np.zeros(nv)
        obj[:ng_free] = Gc_red[i]
        obj[ng_free:] = Gb_red[i]

        def _fallback_lb():
            return c_red[i] - np.abs(Gc_red[i]).sum() - np.abs(Gb_red[i]).sum()

        def _fallback_ub():
            return c_red[i] + np.abs(Gc_red[i]).sum() + np.abs(Gb_red[i]).sum()

        def _solve_lb():
            nonlocal _calls
            _calls += 1
            if _hp_solver is not None:
                st, fun = _hp_solver.solve_min(obj)
                if st == "optimal":
                    return fun + c_red[i]
                if st == "infeasible":
                    return _fallback_lb()
            res = _linprog(obj, A_ub=A_all, b_ub=b_all, bounds=bounds, method='highs')
            return (res.fun + c_red[i]) if res.success else _fallback_lb()

        def _solve_ub():
            nonlocal _calls
            _calls += 1
            if _hp_solver is not None:
                st, fun = _hp_solver.solve_min(-obj)
                if st == "optimal":
                    return -fun + c_red[i]
                if st == "infeasible":
                    return _fallback_ub()
            res = _linprog(-obj, A_ub=A_all, b_ub=b_all, bounds=bounds, method='highs')
            return (-res.fun + c_red[i]) if res.success else _fallback_ub()

        if classify_mode:
            lb0 = float(base_lb_np[out_pos])
            ub0 = float(base_ub_np[out_pos])
            if abs(lb0) <= abs(ub0):
                lb_val = _solve_lb()
                lb_out[out_pos] = lb_val
                if lb_val >= 0.0:
                    ub_out[out_pos] = ub0
                    _saved += 1
                else:
                    ub_out[out_pos] = _solve_ub()
            else:
                ub_val = _solve_ub()
                ub_out[out_pos] = ub_val
                if ub_val <= 0.0:
                    lb_out[out_pos] = lb0
                    _saved += 1
                else:
                    lb_out[out_pos] = _solve_lb()
        else:
            lb_out[out_pos] = _solve_lb()
            ub_out[out_pos] = _solve_ub()

    lb_t = torch.tensor(lb_out, device=device, dtype=dtype).unsqueeze(1)
    ub_t = torch.tensor(ub_out, device=device, dtype=dtype).unsqueeze(1)
    if _prof:
        _elapsed = _time.perf_counter() - _t0
        _mode = "classify" if classify_mode else "full"
        logger.debug(
            "eq_elim_lp prof: mode=%s n_idx=%d lp_calls=%d saved=%d "
            "ng_free=%d nb=%d constraints=%d time=%.3fs",
            _mode, len(solve_idx), _calls, _saved, ng_free,
            nb, A_all.shape[0], _elapsed,
        )
    return lb_t, ub_t


# ---------------------------------------------------------------------------
# Tier 1: hz_bounds_unconstrained
# ---------------------------------------------------------------------------


@torch.no_grad()
def hz_bounds_unconstrained(hz) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cheapest sound bounds: ``c ± |Gc|·1 ± |Gb|·1``.

    Faithful port of HyZor ``HybridZonotope._bounds_unconstrained``
    (HybridZonotope.py:470). Returns ``(lb, ub)`` of shape ``(n, 1)``.
    """
    n = int(hz.c.shape[0])
    dtype = hz.c.dtype
    device = hz.c.device
    if hz.Gc.numel() == 0 and hz.Gb.numel() == 0:
        return hz.c.clone(), hz.c.clone()
    absGc = (hz.Gc.abs().sum(dim=1, keepdim=True)
             if hz.Gc.numel() else torch.zeros((n, 1), dtype=dtype, device=device))
    absGb = (hz.Gb.abs().sum(dim=1, keepdim=True)
             if hz.Gb.numel() else torch.zeros((n, 1), dtype=dtype, device=device))
    rad = absGc + absGb
    return hz.c - rad, hz.c + rad


# ---------------------------------------------------------------------------
# hz_intersect_box
# ---------------------------------------------------------------------------


@torch.no_grad()
def hz_intersect_box(hz, lb: torch.Tensor, ub: torch.Tensor) -> HZono:
    """Intersect HZ with axis-aligned box ``[lb, ub]``. Adds 2n inequality
    rows.

    Faithful port of HyZor ``HybridZonotope.intersect_box``
    (HybridZonotope.py:399).

    Soundness: if ``lb / ub`` come from a sound overapproximation of
    the true reachable set, the intersection still contains all true
    outputs.

    Output: new HZono with same ``c, Gc, Gb`` but ``2n`` new
    inequality rows in ``Ac, Ab, b`` and eq_mask.
    """
    dtype = hz.c.dtype
    device = hz.c.device
    n = int(hz.c.shape[0])

    lb_col = lb.to(device=device, dtype=dtype).view(-1, 1)
    ub_col = ub.to(device=device, dtype=dtype).view(-1, 1)

    # z_i <= ub_i:  Gc[i,:] xi_c + Gb[i,:] xi_b <= ub_i - c_i
    # z_i >= lb_i: -Gc[i,:] xi_c - Gb[i,:] xi_b <= c_i - lb_i
    new_Ac = torch.cat([hz.Gc, -hz.Gc], dim=0)
    new_Ab = torch.cat([hz.Gb, -hz.Gb], dim=0)
    new_b = torch.cat([ub_col - hz.c, hz.c - lb_col], dim=0)

    Ac_all = torch.cat([hz.Ac, new_Ac], dim=0)
    Ab_all = torch.cat([hz.Ab, new_Ab], dim=0)
    b_all = torch.cat([hz.b, new_b], dim=0)

    em_old = _eq_mask_of(hz)
    em_new = torch.cat(
        [em_old, torch.zeros(2 * n, dtype=torch.bool, device=device)]
    )

    out = HZono(
        c=hz.c.clone(), Gc=hz.Gc.clone(), Gb=hz.Gb.clone(),
        Ac=Ac_all, Ab=Ab_all, b=b_all, eq_mask=em_new,
    )
    # ng/nb unchanged by intersect_box; preserve input's base tracking
    # so a prior project_eq_elim's _base_ng doesn't get overwritten by
    # __post_init__'s default of new.ng.
    from act.back_end.solver.solver_hz import _propagate_base
    _propagate_base(hz, out)
    return out


# ---------------------------------------------------------------------------
# Tier 2: closed-form Lagrangian dual ascent (Adam, batched on GPU)
# ---------------------------------------------------------------------------


def hz_bounds_hz_dual(
    hz: HZono,
    *,
    max_iter: int = 150,
    lr: float = 0.1,
    selective_lp: bool = True,
    lp_threshold: float = 0.5,
    unconstrained_lambda: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paper-consistent bound cascade: UNC -> dual ascent -> selective LP.

    Faithful port of HyZor ``HybridZonotope._bounds_hz_dual`` (HZ:2714).
    Three-tier:

      Tier 1 - unconstrained interval hull (fast).
      Tier 2 - batched Adam Lagrangian dual ascent on borderline neurons
               (sound for any non-negative ineq λ, free eq λ).
      Tier 3 - selective ``hz_bounds_eq_elim_lp`` on still-borderline
               neurons whose Tier-2 margin is within ``lp_threshold`` of
               the UNC width (a hint that LP can close the gap).

    Returns ``(lb, ub)`` of shape ``(n, 1)``.
    """
    import os as _os_dual
    n = int(hz.c.shape[0])
    nc = int(hz.b.shape[0])

    # No constraints: UNC is exact.
    if nc == 0:
        return hz_bounds_unconstrained(hz)

    try:
        lp_threshold = float(
            _os_dual.environ.get("HYZOR_HZ_DUAL_LP_THRESHOLD", str(lp_threshold))
        )
    except Exception:
        pass

    device = hz.c.device
    dtype = hz.c.dtype
    c = hz.c.view(-1)
    Gc, Gb = hz.Gc, hz.Gb
    Ac, Ab = hz.Ac, hz.Ab
    bvec = hz.b.view(-1)
    ng = int(Gc.shape[1])
    nb = int(Gb.shape[1])

    AcT = Ac.T.contiguous()
    AbT = Ab.T.contiguous()
    eq_mask = _eq_mask_of(hz) if nc > 0 else torch.zeros(0, dtype=torch.bool, device=device)

    # ---- Tier 1: unconstrained interval hull ----
    lb_unc_raw, ub_unc_raw = hz_bounds_unconstrained(hz)
    lb_unc = lb_unc_raw.view(-1)
    ub_unc = ub_unc_raw.view(-1)

    borderline_mask = (lb_unc < 0) & (ub_unc > 0)
    n_borderline = int(borderline_mask.sum().item())
    if n_borderline == 0:
        return lb_unc.unsqueeze(1), ub_unc.unsqueeze(1)

    # ---- Tier 2: batched Adam Lagrangian dual ascent ----
    _iter_cap = int(_os_dual.environ.get("HYZOR_HZ_DUAL_MAX_ITER_CAP", "50"))
    actual_iter = max(1, min(max_iter, max(1, _iter_cap)))

    border_idx = torch.nonzero(borderline_mask, as_tuple=False).view(-1)
    Gc_b = Gc[border_idx]
    Gb_b = Gb[border_idx]
    c_b = c[border_idx]

    lam_ub = torch.zeros(n_borderline, nc, device=device, dtype=dtype)
    lam_lb = torch.zeros(n_borderline, nc, device=device, dtype=dtype)
    m_ub = torch.zeros_like(lam_ub); v_ub = torch.zeros_like(lam_ub)
    m_lb = torch.zeros_like(lam_lb); v_lb = torch.zeros_like(lam_lb)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    best_ub_b = torch.full(
        (n_borderline,), float("inf"), device=device, dtype=dtype
    )
    best_lb_b = torch.full(
        (n_borderline,), float("-inf"), device=device, dtype=dtype
    )

    has_em = (int(eq_mask.numel()) == nc)
    ineq_cols = (~eq_mask) if has_em else None

    for it in range(actual_iter):
        t_adam = it + 1
        # ---- Upper bound minimization step ----
        adj_gc = Gc_b - lam_ub @ Ac
        adj_gb = Gb_b - lam_ub @ Ab
        ub_val = c_b + (lam_ub @ bvec) + adj_gc.abs().sum(1) + adj_gb.abs().sum(1)
        best_ub_b = torch.minimum(best_ub_b, ub_val)
        sg = bvec.unsqueeze(0) - adj_gc.sign() @ AcT - adj_gb.sign() @ AbT
        m_ub = beta1 * m_ub + (1 - beta1) * sg
        v_ub = beta2 * v_ub + (1 - beta2) * sg * sg
        lam_ub_raw = lam_ub - lr * (m_ub / (1 - beta1 ** t_adam)) / (
            (v_ub / (1 - beta2 ** t_adam)).sqrt() + eps_adam
        )
        if has_em:
            lam_ub = lam_ub_raw.clone()
            lam_ub[:, ineq_cols] = lam_ub_raw[:, ineq_cols].clamp(min=0.0)
        elif unconstrained_lambda:
            lam_ub = lam_ub_raw
        else:
            lam_ub = lam_ub_raw.clamp_(min=0.0)

        # ---- Lower bound maximization step (symmetric) ----
        adj_gc = Gc_b + lam_lb @ Ac
        adj_gb = Gb_b + lam_lb @ Ab
        lb_val = c_b - (lam_lb @ bvec) - adj_gc.abs().sum(1) - adj_gb.abs().sum(1)
        best_lb_b = torch.maximum(best_lb_b, lb_val)
        sg = -bvec.unsqueeze(0) - adj_gc.sign() @ AcT - adj_gb.sign() @ AbT
        m_lb = beta1 * m_lb + (1 - beta1) * sg
        v_lb = beta2 * v_lb + (1 - beta2) * sg * sg
        lam_lb_raw = lam_lb + lr * (m_lb / (1 - beta1 ** t_adam)) / (
            (v_lb / (1 - beta2 ** t_adam)).sqrt() + eps_adam
        )
        if has_em:
            lam_lb = lam_lb_raw.clone()
            lam_lb[:, ineq_cols] = lam_lb_raw[:, ineq_cols].clamp(min=0.0)
        elif unconstrained_lambda:
            lam_lb = lam_lb_raw
        else:
            lam_lb = lam_lb_raw.clamp_(min=0.0)

    # Merge Tier 2 results with UNC (Tier 2 must be no looser than UNC).
    best_lb = lb_unc.clone()
    best_ub = ub_unc.clone()
    best_lb[border_idx] = torch.maximum(best_lb_b, lb_unc[border_idx])
    best_ub[border_idx] = torch.minimum(best_ub_b, ub_unc[border_idx])

    # ---- Tier 3: selective reduced LP on residual borderline set ----
    if selective_lp:
        still_unstable = (best_lb < 0) & (best_ub > 0)
        lp_candidates = torch.nonzero(still_unstable, as_tuple=False).view(-1)
        if lp_candidates.numel() > 0:
            margins = torch.minimum(
                best_lb[lp_candidates].abs(),
                best_ub[lp_candidates].abs(),
            )
            unc_width = (ub_unc - lb_unc)[lp_candidates].clamp(min=1e-12)
            relative_margin = margins / unc_width
            lp_idx = lp_candidates[relative_margin < lp_threshold]
            if lp_idx.numel() > 0 and has_em and bool(eq_mask.any()):
                # eq-elim LP path (Tier-3 of cascade). Only if eq rows present.
                try:
                    lp_lb, lp_ub = hz_bounds_eq_elim_lp(hz, indices=lp_idx)
                    # eq_elim_lp returns shape (k, 1) where k = len(indices).
                    best_lb[lp_idx] = torch.maximum(
                        best_lb[lp_idx], lp_lb.view(-1).to(device=device, dtype=dtype)
                    )
                    best_ub[lp_idx] = torch.minimum(
                        best_ub[lp_idx], lp_ub.view(-1).to(device=device, dtype=dtype)
                    )
                except Exception:
                    pass  # sound fallback to Tier-2 result

    return best_lb.unsqueeze(1), best_ub.unsqueeze(1)
