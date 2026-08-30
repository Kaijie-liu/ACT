# ===- act/back_end/moe/hz_routing.py - Exact HZ Route Guards ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import scipy.sparse as sp

from act.back_end.moe.incremental_hz_solver import IncrementalHZBranchSolver
from act.back_end.solver.solver_hz import (
    HZono,
    SparseHZono,
    hz_add_output_inequalities,
    hz_check_feasibility,
    hz_compute_bounds,
    hz_concat,
    hz_multiply,
    hz_select_rows,
    hz_support_bounds,
    sparse_hz_concat,
    sparse_hz_linear,
)

HZ = HZono | SparseHZono


@dataclass(frozen=True)
class TopKMembershipDomain:
    hz: HZ
    expert: int
    top_k: int
    big_m: Mapping[int, float]
    selection_binaries: int
    big_m_support_mode: str = "fast"
    big_m_support_exact: bool = False
    big_m_upper_status: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteBranch:
    expert: int
    feasibility: str
    conditioned_router: HZ
    guarded_input: HZ | None
    selection_binaries: int
    big_m: Mapping[int, float]
    nodes: int
    elapsed: float


@dataclass(frozen=True)
class CandidateReport:
    candidates: tuple[int, ...]
    infeasible: tuple[int, ...]
    unresolved: tuple[int, ...]
    branches: tuple[RouteBranch, ...]
    minimal: bool


@dataclass(frozen=True)
class TopKSetBranch:
    route_set: tuple[int, ...]
    feasibility: str
    nodes: int
    elapsed: float


@dataclass(frozen=True)
class TopKSetReport:
    """Feasibility report for legal unordered top-k route sets.

    Ties are deliberately inclusive: a set is legal when every selected score
    can be greater than or equal to every unselected score.  ``exact`` is true
    only when the incoming router HZ is exact and every feasibility query was
    decided.
    """

    feasible: tuple[tuple[int, ...], ...]
    infeasible: tuple[tuple[int, ...], ...]
    unresolved: tuple[tuple[int, ...], ...]
    branches: tuple[TopKSetBranch, ...]
    exact: bool


@dataclass(frozen=True)
class TopKSetDomain:
    """An exact tie-inclusive guard for one unordered top-k route set."""

    hz: HZ
    route_set: tuple[int, ...]
    top_k: int


@dataclass(frozen=True)
class LazyTopKSetReport:
    """Incremental no-good-cut enumeration of legal unordered top-k sets."""

    route_sets: tuple[tuple[int, ...], ...]
    complete: bool
    status: str
    reason: str
    solves: int
    no_good_cuts: int
    selector_binaries: int
    elapsed: float
    big_m_support_mode: str
    big_m_support_exact: bool
    big_m_upper_status: tuple[str, ...]
    telemetry: Mapping[str, object]


def _output_width(hz: HZ) -> int:
    return hz.n_out if isinstance(hz, SparseHZono) else int(hz.c.shape[0])


def _fast_linear_upper(hz: HZ, A: torch.Tensor) -> torch.Tensor:
    if isinstance(hz, SparseHZono):
        A_np = A.detach().cpu().double().numpy()
        A_sp = sp.csr_matrix(A_np)
        center = torch.from_numpy(A_np @ hz.c)
        gc = A_sp @ hz.Gc
        gb = A_sp @ hz.Gb
        radius = torch.from_numpy(
            abs(gc).sum(axis=1).A1 + (abs(gb).sum(axis=1).A1 if hz.n_bin else 0.0)
        )
        return center + radius
    transformed = HZono(
        c=A.to(hz.c) @ hz.c,
        Gc=A.to(hz.c) @ hz.Gc,
        Gb=A.to(hz.c) @ hz.Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=hz.eq_mask,
        col_ids=hz.col_ids,
        bcol_ids=hz.bcol_ids,
    )
    return hz_compute_bounds(transformed, exact=False).ub.reshape(-1).cpu()


def _linear_upper_support(
    hz: HZ,
    A: torch.Tensor,
    *,
    mode: str,
    time_limit: float,
) -> tuple[torch.Tensor, bool, tuple[str, ...]]:
    """Return sound upper bounds, optionally retaining all HZ constraints."""

    normalized = str(mode).lower().replace("-", "_")
    if normalized == "fast":
        return (
            _fast_linear_upper(hz, A),
            False,
            tuple("fast_generator" for _ in range(int(A.shape[0]))),
        )
    if normalized not in {"lp", "exact"}:
        raise ValueError("big_m_support_mode must be 'fast', 'lp', or 'exact'")
    transformed = (
        sparse_hz_linear(hz, A.detach().cpu().double().numpy())
        if isinstance(hz, SparseHZono)
        else hz_multiply(hz, A.to(hz.c))
    )
    support = hz_support_bounds(
        transformed,
        range(int(A.shape[0])),
        time_limit=float(time_limit),
        relax_binaries=normalized == "lp",
    )
    return (
        support.bounds.ub.reshape(-1).detach().cpu().double(),
        bool(support.exact),
        tuple(support.upper_status),
    )


def condition_topk_membership(
    hz: HZ,
    expert: int,
    top_k: int,
    *,
    score_rows: Sequence[int] | None = None,
    big_m_padding: float = 1e-9,
    big_m_support_mode: str = "fast",
    big_m_support_time_limit: float = 30.0,
) -> TopKMembershipDomain:
    """Intersect ``hz`` with “expert may belong to a legal top-k set”.

    The encoding uses at most ``E-1`` new binaries and never enumerates
    ``binomial(E, k)`` route sets.  ``M[j,i]`` is a sound upper bound on
    ``r_j-r_i``.  The default ``fast`` mode ignores retained constraints;
    ``lp`` and ``exact`` optimize the support with all HZ constraints and
    fail closed to the generator bound on any undecided side.
    """
    rows = (
        tuple(range(_output_width(hz)))
        if score_rows is None
        else tuple(score_rows)
    )
    experts = len(rows)
    if not 0 <= int(expert) < experts:
        raise IndexError("expert index out of range")
    if not 1 <= int(top_k) <= experts:
        raise ValueError("top_k must lie in [1, num_experts]")
    if not np.isfinite(float(big_m_padding)) or float(big_m_padding) < 0.0:
        raise ValueError("big_m_padding must be finite and non-negative")
    if (
        not np.isfinite(float(big_m_support_time_limit))
        or float(big_m_support_time_limit) < 0.0
    ):
        raise ValueError("big_m_support_time_limit must be finite and non-negative")
    support_mode = str(big_m_support_mode).lower().replace("-", "_")
    if support_mode not in {"fast", "lp", "exact"}:
        raise ValueError("big_m_support_mode must be 'fast', 'lp', or 'exact'")
    if top_k == experts:
        return TopKMembershipDomain(
            hz,
            int(expert),
            int(top_k),
            {},
            0,
            support_mode,
            False,
            (),
        )

    dtype = torch.float64
    A_diff = torch.zeros((experts - 1, _output_width(hz)), dtype=dtype)
    competitors = [j for j in range(experts) if j != int(expert)]
    for row, competitor in enumerate(competitors):
        A_diff[row, rows[competitor]] = 1.0
        A_diff[row, rows[int(expert)]] = -1.0
    safe_upper, support_exact, upper_status = _linear_upper_support(
        hz,
        A_diff,
        mode=support_mode,
        time_limit=big_m_support_time_limit,
    )

    possible = [
        (competitor, row, max(0.0, float(safe_upper[row].item())) + big_m_padding)
        for row, competitor in enumerate(competitors)
        if float(safe_upper[row].item()) > 0.0
    ]
    big_m = {competitor: value for competitor, _, value in possible}

    if top_k == 1:
        conditioned = hz_add_output_inequalities(
            hz,
            A_diff,
            torch.zeros(A_diff.shape[0], dtype=dtype),
        )
        return TopKMembershipDomain(
            conditioned,
            int(expert),
            1,
            big_m,
            0,
            support_mode,
            support_exact,
            upper_status,
        )

    if len(possible) <= top_k - 1:
        return TopKMembershipDomain(
            hz,
            int(expert),
            int(top_k),
            big_m,
            0,
            support_mode,
            support_exact,
            upper_status,
        )

    q = len(possible)
    A = torch.zeros((q + 1, _output_width(hz)), dtype=dtype)
    binary = torch.zeros((q + 1, q), dtype=dtype)
    rhs = torch.zeros(q + 1, dtype=dtype)
    for slot, (_, diff_row, M) in enumerate(possible):
        A[slot] = A_diff[diff_row]
        binary[slot, slot] = -M / 2.0
        rhs[slot] = M / 2.0
    binary[-1] = 0.5
    rhs[-1] = float(top_k - 1) - q / 2.0
    conditioned = hz_add_output_inequalities(
        hz,
        A,
        rhs,
        new_binary_coefficients=binary,
    )
    return TopKMembershipDomain(
        conditioned,
        int(expert),
        int(top_k),
        big_m,
        q,
        support_mode,
        support_exact,
        upper_status,
    )


def guarded_input_domain(
    input_hz: HZ,
    router_hz: HZ,
    expert: int,
    top_k: int,
) -> TopKMembershipDomain:
    """Attach router constraints to input coordinates, then discard scores."""
    if isinstance(input_hz, SparseHZono) != isinstance(router_hz, SparseHZono):
        raise TypeError("input and router HZ representations must match")
    n_input = _output_width(input_hz)
    if isinstance(input_hz, SparseHZono):
        joint = sparse_hz_concat([input_hz, router_hz])
    else:
        joint = hz_concat([input_hz, router_hz])
    if joint is None:
        raise ValueError("cannot construct an empty joint input/router HZ")
    score_rows = range(n_input, n_input + _output_width(router_hz))
    membership = condition_topk_membership(
        joint,
        expert,
        top_k,
        score_rows=score_rows,
    )
    guarded = hz_select_rows(membership.hz, range(n_input))
    return TopKMembershipDomain(
        guarded,
        membership.expert,
        membership.top_k,
        membership.big_m,
        membership.selection_binaries,
        membership.big_m_support_mode,
        membership.big_m_support_exact,
        membership.big_m_upper_status,
    )


def condition_topk_set(
    hz: HZ,
    route_set: Sequence[int],
    *,
    score_rows: Sequence[int] | None = None,
) -> TopKSetDomain:
    """Intersect ``hz`` with one legal unordered, tie-inclusive top-k set."""
    rows = (
        tuple(range(_output_width(hz)))
        if score_rows is None
        else tuple(score_rows)
    )
    experts = len(rows)
    selected = tuple(sorted(int(value) for value in route_set))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("route_set must contain distinct expert indices")
    if any(value < 0 or value >= experts for value in selected):
        raise IndexError("route_set expert index out of range")
    outside = tuple(value for value in range(experts) if value not in selected)
    comparisons = len(selected) * len(outside)
    if comparisons:
        A = torch.zeros((comparisons, _output_width(hz)), dtype=torch.float64)
        row = 0
        for chosen in selected:
            for other in outside:
                A[row, rows[other]] = 1.0
                A[row, rows[chosen]] = -1.0
                row += 1
        conditioned = hz_add_output_inequalities(
            hz,
            A,
            torch.zeros(comparisons, dtype=torch.float64),
        )
    else:
        conditioned = hz
    return TopKSetDomain(conditioned, selected, len(selected))


def guarded_input_topk_set(
    input_hz: HZ,
    router_hz: HZ,
    route_set: Sequence[int],
) -> TopKSetDomain:
    """Attach an exact unordered set guard, then retain only input outputs."""
    if isinstance(input_hz, SparseHZono) != isinstance(router_hz, SparseHZono):
        raise TypeError("input and router HZ representations must match")
    n_input = _output_width(input_hz)
    if isinstance(input_hz, SparseHZono):
        joint = sparse_hz_concat([input_hz, router_hz])
    else:
        joint = hz_concat([input_hz, router_hz])
    if joint is None:
        raise ValueError("cannot construct an empty joint input/router HZ")
    score_rows = range(n_input, n_input + _output_width(router_hz))
    guarded = condition_topk_set(joint, route_set, score_rows=score_rows)
    return TopKSetDomain(
        hz_select_rows(guarded.hz, range(n_input)),
        guarded.route_set,
        guarded.top_k,
    )


def analyze_candidates(
    router_hz: HZ,
    top_k: int,
    *,
    input_hz: HZ | None = None,
    time_limit_per_expert: float = 30.0,
    router_exact: bool = True,
) -> CandidateReport:
    """Compute the smallest candidate set when every HZ MILP is decided."""
    experts = _output_width(router_hz)
    branches: list[RouteBranch] = []
    infeasible: list[int] = []
    unresolved: list[int] = []
    candidates: list[int] = []
    for expert in range(experts):
        membership = condition_topk_membership(router_hz, expert, top_k)
        feasibility = hz_check_feasibility(
            membership.hz, time_limit=time_limit_per_expert
        )
        guarded = None
        if feasibility.status != "infeasible" and input_hz is not None:
            guarded = guarded_input_domain(input_hz, router_hz, expert, top_k).hz
        branch = RouteBranch(
            expert=expert,
            feasibility=feasibility.status,
            conditioned_router=membership.hz,
            guarded_input=guarded,
            selection_binaries=membership.selection_binaries,
            big_m=membership.big_m,
            nodes=feasibility.nodes,
            elapsed=feasibility.elapsed,
        )
        branches.append(branch)
        if feasibility.status == "infeasible":
            infeasible.append(expert)
        else:
            candidates.append(expert)
            if feasibility.status == "unknown":
                unresolved.append(expert)
    return CandidateReport(
        candidates=tuple(candidates),
        infeasible=tuple(infeasible),
        unresolved=tuple(unresolved),
        branches=tuple(branches),
        minimal=bool(router_exact and not unresolved),
    )


def analyze_topk_sets(
    router_hz: HZ,
    top_k: int,
    *,
    time_limit_per_set: float = 30.0,
    router_exact: bool = True,
) -> TopKSetReport:
    """Enumerate all feasible unordered top-k sets for a small router.

    For a proposed set ``S``, legality is encoded without selector binaries as
    ``r_j - r_i <= 0`` for every ``i in S`` and ``j not in S``.  The inclusive
    inequality implements the repository's any-legal-top-k tie policy.
    """
    experts = _output_width(router_hz)
    if not 1 <= int(top_k) <= experts:
        raise ValueError("top_k must lie in [1, num_experts]")
    feasible: list[tuple[int, ...]] = []
    infeasible: list[tuple[int, ...]] = []
    unresolved: list[tuple[int, ...]] = []
    branches: list[TopKSetBranch] = []
    for selected_values in combinations(range(experts), int(top_k)):
        selected = tuple(int(value) for value in selected_values)
        conditioned = condition_topk_set(router_hz, selected).hz
        result = hz_check_feasibility(
            conditioned,
            time_limit=float(time_limit_per_set),
        )
        branches.append(
            TopKSetBranch(
                route_set=selected,
                feasibility=result.status,
                nodes=result.nodes,
                elapsed=result.elapsed,
            )
        )
        if result.status == "feasible":
            feasible.append(selected)
        elif result.status == "infeasible":
            infeasible.append(selected)
        else:
            unresolved.append(selected)
    return TopKSetReport(
        feasible=tuple(feasible),
        infeasible=tuple(infeasible),
        unresolved=tuple(unresolved),
        branches=tuple(branches),
        exact=bool(router_exact and not unresolved),
    )


def _topk_selector_domain(
    router_hz: SparseHZono,
    top_k: int,
    *,
    big_m_padding: float,
    big_m_support_mode: str,
    big_m_support_time_limit: float,
) -> tuple[SparseHZono, bool, tuple[str, ...]]:
    """Encode one tie-inclusive legal top-k selector into a sparse HZ."""

    experts = router_hz.n_out
    pairs = tuple((i, j) for i in range(experts) for j in range(experts) if i != j)
    A_diff = torch.zeros((len(pairs), experts), dtype=torch.float64)
    for row, (selected, outside) in enumerate(pairs):
        A_diff[row, outside] = 1.0
        A_diff[row, selected] = -1.0
    upper, support_exact, upper_status = _linear_upper_support(
        router_hz,
        A_diff,
        mode=big_m_support_mode,
        time_limit=big_m_support_time_limit,
    )
    if not torch.isfinite(upper).all():
        raise ValueError("top-k selector requires finite score-difference bounds")

    rows = len(pairs) + 2
    A = torch.zeros((rows, experts), dtype=torch.float64)
    binary = torch.zeros((rows, experts), dtype=torch.float64)
    rhs = torch.zeros(rows, dtype=torch.float64)
    for row, ((selected, outside), raw_upper) in enumerate(zip(pairs, upper)):
        M = max(0.0, float(raw_upper.item())) + float(big_m_padding)
        A[row] = A_diff[row]
        binary[row, selected] = M / 2.0
        binary[row, outside] = -M / 2.0
        rhs[row] = M

    # beta_i in {-1, 1}, z_i=(beta_i+1)/2, and sum_i z_i = k.
    binary[-2] = 0.5
    rhs[-2] = float(top_k) - experts / 2.0
    binary[-1] = -0.5
    rhs[-1] = -float(top_k) + experts / 2.0
    conditioned = hz_add_output_inequalities(
        router_hz,
        A,
        rhs,
        new_binary_coefficients=binary,
    )
    if not isinstance(conditioned, SparseHZono):  # pragma: no cover - typing guard
        raise TypeError("lazy selector encoding unexpectedly changed HZ representation")
    return conditioned, support_exact, upper_status


def enumerate_topk_sets_lazy(
    router_hz: SparseHZono,
    top_k: int,
    *,
    time_limit: float = 30.0,
    max_sets: int | None = None,
    big_m_padding: float = 1e-9,
    big_m_support_mode: str = "fast",
    big_m_support_time_limit: float = 30.0,
) -> LazyTopKSetReport:
    """Enumerate feasible top-k sets with one incremental HiGHS session.

    The MILP contains one selector per expert and exact tie-inclusive ordering
    implications.  Each validated solution contributes one no-good cut, after
    which the same loaded model is solved again.  ``complete`` is true only
    when HiGHS proves the cut-augmented model infeasible; time or numerical
    limits never masquerade as a complete enumeration.
    """

    if not isinstance(router_hz, SparseHZono):
        raise TypeError("lazy top-k enumeration requires a SparseHZono")
    experts = router_hz.n_out
    if not 1 <= int(top_k) <= experts:
        raise ValueError("top_k must lie in [1, num_experts]")
    if not np.isfinite(float(time_limit)) or float(time_limit) < 0.0:
        raise ValueError("time_limit must be finite and non-negative")
    if max_sets is not None and int(max_sets) <= 0:
        raise ValueError("max_sets must be positive when provided")
    if not np.isfinite(float(big_m_padding)) or float(big_m_padding) < 0.0:
        raise ValueError("big_m_padding must be finite and non-negative")
    if (
        not np.isfinite(float(big_m_support_time_limit))
        or float(big_m_support_time_limit) < 0.0
    ):
        raise ValueError("big_m_support_time_limit must be finite and non-negative")
    support_mode = str(big_m_support_mode).lower().replace("-", "_")
    if support_mode not in {"fast", "lp", "exact"}:
        raise ValueError("big_m_support_mode must be 'fast', 'lp', or 'exact'")

    started = time.monotonic()
    selector_hz, support_exact, upper_status = _topk_selector_domain(
        router_hz,
        int(top_k),
        big_m_padding=float(big_m_padding),
        big_m_support_mode=support_mode,
        big_m_support_time_limit=float(big_m_support_time_limit),
    )
    session = IncrementalHZBranchSolver(
        selector_hz,
        time_limit=float(time_limit),
        relax_binaries=False,
        submit_basis=True,
    )
    selector_start = session.model.n_var - experts
    if selector_start < session.model.n_cont:
        raise RuntimeError("selector binaries are not the final MILP variables")

    route_sets: list[tuple[int, ...]] = []
    cuts: list[sp.csr_matrix] = []
    previous_point: np.ndarray | None = None
    terminal_status = "unknown"
    terminal_reason = "not_run"
    while True:
        if max_sets is not None and len(route_sets) >= int(max_sets):
            terminal_status = "max_sets"
            terminal_reason = "max_sets_reached"
            break
        if cuts:
            extra_A = sp.vstack(cuts, format="csr")
            extra_lb = np.full(len(cuts), -np.inf, dtype=np.float64)
            extra_ub = np.full(len(cuts), int(top_k) - 1.0, dtype=np.float64)
        else:
            extra_A = None
            extra_lb = None
            extra_ub = None
        query = session.find_feasible(
            extra_A=extra_A,
            extra_lb=extra_lb,
            extra_ub=extra_ub,
            mip_start_indices=(
                np.arange(selector_start, dtype=np.int32)
                if previous_point is not None and selector_start
                else None
            ),
            mip_start_values=(
                previous_point[:selector_start]
                if previous_point is not None and selector_start
                else None
            ),
        )
        if query.status == "infeasible":
            terminal_status = "complete"
            terminal_reason = "cut_augmented_model_infeasible"
            break
        if query.status != "optimal" or query.point is None:
            terminal_status = query.status
            terminal_reason = query.reason
            break

        selector = np.asarray(
            query.point[selector_start : selector_start + experts],
            dtype=np.float64,
        )
        selected = tuple(int(i) for i in np.flatnonzero(selector > 0.5))
        if len(selected) != int(top_k) or selected in route_sets:
            terminal_status = "unknown"
            terminal_reason = "invalid_or_duplicate_selector_solution"
            break
        scores = np.asarray(
            session.model.value_center + session.model.value_matrix @ query.point,
            dtype=np.float64,
        ).reshape(-1)
        outside = tuple(i for i in range(experts) if i not in selected)
        tolerance = 1e-7
        if outside and min(scores[list(selected)]) < max(scores[list(outside)]) - tolerance:
            terminal_status = "unknown"
            terminal_reason = "selector_point_failed_tie_inclusive_replay"
            break
        route_sets.append(selected)
        previous_point = query.point.copy()
        cut = sp.csr_matrix(
            (
                np.ones(len(selected), dtype=np.float64),
                (
                    np.zeros(len(selected), dtype=np.int32),
                    np.asarray([selector_start + i for i in selected], dtype=np.int32),
                ),
            ),
            shape=(1, session.model.n_var),
        )
        cuts.append(cut)

    telemetry = session.telemetry()
    return LazyTopKSetReport(
        route_sets=tuple(sorted(route_sets)),
        complete=terminal_status == "complete",
        status=terminal_status,
        reason=terminal_reason,
        solves=telemetry.solves,
        no_good_cuts=len(cuts),
        selector_binaries=experts,
        elapsed=time.monotonic() - started,
        big_m_support_mode=support_mode,
        big_m_support_exact=bool(support_exact),
        big_m_upper_status=upper_status,
        telemetry=telemetry.as_dict(),
    )
