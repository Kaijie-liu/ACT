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
from typing import Mapping, Sequence

import torch
import scipy.sparse as sp

from act.back_end.solver.solver_hz import (
    HZono,
    SparseHZono,
    hz_add_output_inequalities,
    hz_check_feasibility,
    hz_compute_bounds,
    hz_concat,
    hz_select_rows,
    sparse_hz_concat,
)

HZ = HZono | SparseHZono


@dataclass(frozen=True)
class TopKMembershipDomain:
    hz: HZ
    expert: int
    top_k: int
    big_m: Mapping[int, float]
    selection_binaries: int


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


def condition_topk_membership(
    hz: HZ,
    expert: int,
    top_k: int,
    *,
    score_rows: Sequence[int] | None = None,
    big_m_padding: float = 1e-9,
) -> TopKMembershipDomain:
    """Intersect ``hz`` with “expert may belong to a legal top-k set”.

    The encoding uses at most ``E-1`` new binaries and never enumerates
    ``binomial(E, k)`` route sets.  ``M[j,i]`` is obtained from a
    correlation-preserving HZ support bound on ``r_j-r_i``.
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
    if top_k == experts:
        return TopKMembershipDomain(hz, int(expert), int(top_k), {}, 0)

    dtype = torch.float64
    A_diff = torch.zeros((experts - 1, _output_width(hz)), dtype=dtype)
    competitors = [j for j in range(experts) if j != int(expert)]
    for row, competitor in enumerate(competitors):
        A_diff[row, rows[competitor]] = 1.0
        A_diff[row, rows[int(expert)]] = -1.0
    safe_upper = _fast_linear_upper(hz, A_diff)

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
        return TopKMembershipDomain(conditioned, int(expert), 1, big_m, 0)

    if len(possible) <= top_k - 1:
        return TopKMembershipDomain(hz, int(expert), int(top_k), big_m, 0)

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
    return TopKMembershipDomain(conditioned, int(expert), int(top_k), big_m, q)


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
