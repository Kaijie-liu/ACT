# ===- act/back_end/moe/weighted_topk.py - Normalized Top-k Fallback --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Gate-family-agnostic range fallback for normalized weighted top-k MoEs.

The encoding in this module depends only on a sound box containing the gate
weights and on the fact that those weights lie in the probability simplex.  It
does not encode softmax, sigmoid, exponentiation, or division.  For a selected
set ``S`` and an anchor ``b in S`` it uses the identity

``F = E_b + sum_{i in S \\ {b}} lambda_i * (E_i - E_b)``.

Consequently a size-k route needs k-1 McCormick products.  The implementation
is deliberately restricted to normalized non-negative gates.  ``switch_prob``
is an unnormalized scale and is rejected instead of being silently treated as
a convex gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.moe.schema import GateKind
from act.back_end.moe.weighted_top2 import (
    SAFE_WEIGHTED_RANGE,
    UNKNOWN_WEIGHTED_NUMERICAL,
    UNKNOWN_WEIGHTED_RELAXATION,
    UNKNOWN_WEIGHTED_SOLVER_LIMIT,
    McCormickBounds,
    _assert_prefix,
    _column_mapping,
    _product_range,
    _remap_columns,
    mccormick_inequalities,
)
from act.back_end.solver.solver_hz import (
    HZ_NUMERICAL_POLICY,
    HZMinimumResult,
    HZSupportBoundsResult,
    SparseHZono,
    hz_minimize_output,
    hz_support_bounds,
    sparse_empty,
    sparse_hz_linear,
    sparse_hz_pad_frame,
)


class UnsupportedNormalizedGateError(ValueError):
    """Raised when the normalized-simplex lemma does not apply to a gate."""


@dataclass(frozen=True)
class SharedInputExpertsHZ:
    """Outputs of k experts with shared input factors and private ReLU factors."""

    output_hz: SparseHZono
    input_hz: SparseHZono
    route_set: tuple[int, ...]
    expert_rows: tuple[tuple[int, ...], ...]
    shared_continuous: int
    shared_binary: int
    private_continuous: tuple[int, ...]
    private_binary: tuple[int, ...]


@dataclass(frozen=True)
class NormalizedTopKGateBox:
    """A property-independent ``simplex intersection box`` gate abstraction."""

    route_set: tuple[int, ...]
    gate_kind: GateKind
    conditioned_router: SparseHZono = field(repr=False, compare=False)
    router_frame_id: int | None
    router_output_width: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    score_support: HZSupportBoundsResult | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        raw_route_set = tuple(int(value) for value in self.route_set)
        route_set = _canonical_route_set(raw_route_set)
        if raw_route_set != route_set:
            raise ValueError("gate-box route set and bounds must use canonical order")
        object.__setattr__(self, "route_set", route_set)
        object.__setattr__(self, "gate_kind", GateKind(self.gate_kind))
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if len(lower) != len(route_set) or len(upper) != len(route_set):
            raise ValueError("gate bounds must have one entry per selected expert")
        if (
            self.router_frame_id != self.conditioned_router.frame_id
            or self.router_output_width != self.conditioned_router.n_out
        ):
            raise ValueError("gate-box router identity metadata is inconsistent")
        if self.gate_kind == GateKind.SWITCH_PROB:
            raise UnsupportedNormalizedGateError(
                "switch_prob is unnormalized and requires an independent scale term"
            )
        if self.gate_kind == GateKind.HARD_TOP1 and len(route_set) != 1:
            raise UnsupportedNormalizedGateError("hard_top1 requires a singleton route")
        if any(not np.isfinite(value) for value in lower + upper):
            raise ValueError("gate bounds must be finite")
        if any(lo < 0.0 or hi > 1.0 or lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError("gate bounds must satisfy 0 <= lower <= upper <= 1")
        tolerance = 64.0 * np.finfo(np.float64).eps
        if sum(lower) > 1.0 + tolerance or sum(upper) < 1.0 - tolerance:
            raise ValueError("gate box has empty intersection with the simplex")

    @property
    def anchor(self) -> int:
        """Use the final canonical expert as the decomposition anchor."""
        return self.route_set[-1]

    @property
    def free_experts(self) -> tuple[int, ...]:
        return self.route_set[:-1]


@dataclass(frozen=True)
class WeightedTopKRangeEncoding:
    """One property-directed normalized top-k McCormick relaxation."""

    output_hz: SparseHZono
    input_hz: SparseHZono
    route_set: tuple[int, ...]
    anchor: int
    free_experts: tuple[int, ...]
    property_row: tuple[float, ...]
    property_constant: float
    gate_box: NormalizedTopKGateBox
    term_bounds: tuple[McCormickBounds, ...]
    difference_support: HZSupportBoundsResult | None
    mccormick_A: tuple[np.ndarray, ...]
    mccormick_b: tuple[np.ndarray, ...]
    simplex_A: np.ndarray
    simplex_b: np.ndarray


@dataclass(frozen=True)
class WeightedTopKRangeDecision:
    """Solver result whose non-positive relaxation candidates are never unsafe."""

    status: str
    reason: str
    minimum: float | None
    candidate_objective: float | None
    candidate_input: torch.Tensor | None
    solver_status: int | None
    solver_gap: float | None
    elapsed: float
    solver_certified_lower_bound: float | None = None
    solver_bound_kind: str | None = None
    solver_primal_objective: float | None = None
    solver_dual_objective: float | None = None


def _canonical_route_set(route_set: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(sorted(int(value) for value in route_set))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("route set must contain distinct expert indices")
    return selected


def normalized_gate_support(gate_kind: GateKind | str) -> tuple[bool, str]:
    """Return whether the simplex decomposition applies to a concrete gate."""
    kind = GateKind(gate_kind)
    if kind == GateKind.SWITCH_PROB:
        return False, "unnormalized selected-expert scale; independent scale required"
    if kind == GateKind.HARD_TOP1:
        return True, "exact singleton simplex weight"
    if kind == GateKind.SELECTED_SOFTMAX:
        return True, "positive selected weights normalized to the simplex"
    if kind == GateKind.NORMALIZED_SIGMOID:
        return True, "positive sigmoid weights normalized to the simplex"
    raise AssertionError(f"unhandled gate kind {kind.value}")


def shared_input_experts_hz(
    entry: SparseHZono,
    experts: Mapping[int, SparseHZono],
) -> SharedInputExpertsHZ:
    """Merge selected experts while sharing entry factors and separating private ones."""
    route_set = _canonical_route_set(tuple(experts))
    ordered = tuple(experts[index] for index in route_set)
    for expert in ordered:
        _assert_prefix(entry, expert)
    widths = {expert.n_out for expert in ordered}
    if len(widths) != 1:
        raise ValueError("selected experts must have the same output width")
    output_width = widths.pop()

    private_cont = tuple(expert.n_cont - entry.n_cont for expert in ordered)
    private_bin = tuple(expert.n_bin - entry.n_bin for expert in ordered)
    n_cont = entry.n_cont + sum(private_cont)
    n_bin = entry.n_bin + sum(private_bin)
    entry_c_map = np.arange(entry.n_cont, dtype=np.int64)
    entry_b_map = np.arange(entry.n_bin, dtype=np.int64)

    c_maps: list[np.ndarray] = []
    b_maps: list[np.ndarray] = []
    c_cursor, b_cursor = entry.n_cont, entry.n_bin
    for expert in ordered:
        c_maps.append(_column_mapping(entry.n_cont, expert.n_cont, c_cursor))
        b_maps.append(_column_mapping(entry.n_bin, expert.n_bin, b_cursor))
        c_cursor += expert.n_cont - entry.n_cont
        b_cursor += expert.n_bin - entry.n_bin

    def remap(hz, c_map, b_map, c_name, b_name):
        return (
            _remap_columns(getattr(hz, c_name), c_map, n_cont),
            _remap_columns(getattr(hz, b_name), b_map, n_bin),
        )

    entry_Ac, entry_Ab = remap(entry, entry_c_map, entry_b_map, "Ac", "Ab")
    entry_Auc, entry_Aub = remap(
        entry, entry_c_map, entry_b_map, "Auc", "Aub"
    )

    output_Gc = []
    output_Gb = []
    extra_Ac = []
    extra_Ab = []
    extra_b = []
    extra_Auc = []
    extra_Aub = []
    extra_ub = []
    for expert, c_map, b_map in zip(ordered, c_maps, b_maps):
        output_Gc.append(_remap_columns(expert.Gc, c_map, n_cont))
        output_Gb.append(_remap_columns(expert.Gb, b_map, n_bin))
        mapped_Ac, mapped_Ab = remap(expert, c_map, b_map, "Ac", "Ab")
        mapped_Auc, mapped_Aub = remap(expert, c_map, b_map, "Auc", "Aub")
        extra_Ac.append(mapped_Ac[entry.n_eq :])
        extra_Ab.append(mapped_Ab[entry.n_eq :])
        extra_b.append(expert.b[entry.n_eq :])
        extra_Auc.append(mapped_Auc[entry.n_ineq :])
        extra_Aub.append(mapped_Aub[entry.n_ineq :])
        extra_ub.append(expert.ub[entry.n_ineq :])

    output = SparseHZono(
        c=np.concatenate([expert.c for expert in ordered]),
        Gc=sp.vstack(output_Gc, format="csr"),
        Gb=sp.vstack(output_Gb, format="csr"),
        Ac=sp.vstack([entry_Ac, *extra_Ac], format="csr"),
        Ab=sp.vstack([entry_Ab, *extra_Ab], format="csr"),
        b=np.concatenate([entry.b, *extra_b]),
        Auc=sp.vstack([entry_Auc, *extra_Auc], format="csr"),
        Aub=sp.vstack([entry_Aub, *extra_Aub], format="csr"),
        ub=np.concatenate([entry.ub, *extra_ub]),
        frame_id=entry.frame_id,
        exact=entry.exact and all(expert.exact for expert in ordered),
    )
    return SharedInputExpertsHZ(
        output_hz=output,
        input_hz=sparse_hz_pad_frame(entry, n_cont, n_bin),
        route_set=route_set,
        expert_rows=tuple(
            tuple(range(position * output_width, (position + 1) * output_width))
            for position in range(len(route_set))
        ),
        shared_continuous=entry.n_cont,
        shared_binary=entry.n_bin,
        private_continuous=private_cont,
        private_binary=private_bin,
    )


def _outward_unit_interval(value: float, direction: float) -> float:
    return float(np.clip(np.nextafter(float(value), direction), 0.0, 1.0))


def _normalized_log_activation_bounds(
    log_activation_lower: np.ndarray,
    log_activation_upper: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Bound normalized positive activations without numerical exponentiation."""
    count = int(log_activation_lower.size)
    if count == 1:
        return (1.0,), (1.0,)
    lower, upper = [], []
    lower_t = torch.from_numpy(log_activation_lower).double()
    upper_t = torch.from_numpy(log_activation_upper).double()
    for index in range(count):
        low_corner = upper_t.clone()
        low_corner[index] = lower_t[index]
        high_corner = lower_t.clone()
        high_corner[index] = upper_t[index]
        lo = float(torch.softmax(low_corner, dim=0)[index])
        hi = float(torch.softmax(high_corner, dim=0)[index])
        lower.append(_outward_unit_interval(lo, 0.0))
        upper.append(_outward_unit_interval(hi, 1.0))
    return tuple(lower), tuple(upper)


def normalized_gate_box_from_score_bounds(
    conditioned_router: SparseHZono,
    route_set: Sequence[int],
    gate_kind: GateKind | str,
    score_lower: Sequence[float],
    score_upper: Sequence[float],
    *,
    score_support: HZSupportBoundsResult | None = None,
) -> NormalizedTopKGateBox:
    """Build a sound simplex-box abstraction from selected router score bounds."""
    raw_selected = tuple(int(value) for value in route_set)
    selected = _canonical_route_set(raw_selected)
    kind = GateKind(gate_kind)
    supported, reason = normalized_gate_support(kind)
    if not supported:
        raise UnsupportedNormalizedGateError(reason)
    if selected[0] < 0 or selected[-1] >= conditioned_router.n_out:
        raise IndexError("route set is outside the router output")
    lower = np.asarray(score_lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(score_upper, dtype=np.float64).reshape(-1)
    if lower.size != len(selected) or upper.size != len(selected):
        raise ValueError("score bounds must have one entry per selected expert")
    order = np.argsort(np.asarray(raw_selected, dtype=np.int64))
    lower = lower[order]
    upper = upper[order]
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("score bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("score lower bound exceeds upper bound")

    if kind == GateKind.HARD_TOP1:
        weight_lower, weight_upper = (1.0,), (1.0,)
    elif kind == GateKind.SELECTED_SOFTMAX:
        weight_lower, weight_upper = _normalized_log_activation_bounds(lower, upper)
    elif kind == GateKind.NORMALIZED_SIGMOID:
        # log(sigmoid(s)) = -softplus(-s) remains finite when sigmoid(s)
        # itself underflows to zero in floating point.
        log_activation_lower = -torch.nn.functional.softplus(
            -torch.from_numpy(lower).double()
        ).numpy()
        log_activation_upper = -torch.nn.functional.softplus(
            -torch.from_numpy(upper).double()
        ).numpy()
        weight_lower, weight_upper = _normalized_log_activation_bounds(
            log_activation_lower, log_activation_upper
        )
    else:
        raise AssertionError(f"unhandled normalized gate {kind.value}")
    return NormalizedTopKGateBox(
        route_set=selected,
        gate_kind=kind,
        conditioned_router=conditioned_router,
        router_frame_id=conditioned_router.frame_id,
        router_output_width=conditioned_router.n_out,
        lower=weight_lower,
        upper=weight_upper,
        score_support=score_support,
    )


def compute_normalized_topk_gate_box(
    conditioned_router: SparseHZono,
    route_set: Sequence[int],
    gate_kind: GateKind | str,
    *,
    time_limit: float,
    relax_binaries: bool = True,
) -> NormalizedTopKGateBox:
    """Compute selected score supports once, then forget the gate function."""
    selected = _canonical_route_set(route_set)
    kind = GateKind(gate_kind)
    supported, reason = normalized_gate_support(kind)
    if not supported:
        raise UnsupportedNormalizedGateError(reason)
    if kind == GateKind.HARD_TOP1:
        if len(selected) != 1:
            raise UnsupportedNormalizedGateError("hard_top1 requires a singleton route")
        return normalized_gate_box_from_score_bounds(
            conditioned_router, selected, kind, [0.0], [0.0]
        )
    support = hz_support_bounds(
        conditioned_router,
        selected,
        time_limit=float(time_limit),
        relax_binaries=bool(relax_binaries),
    )
    return normalized_gate_box_from_score_bounds(
        conditioned_router,
        selected,
        kind,
        support.bounds.lb.reshape(-1).tolist(),
        support.bounds.ub.reshape(-1).tolist(),
        score_support=support,
    )


def simplex_box_contains(
    weights: Sequence[float],
    gate_box: NormalizedTopKGateBox,
    *,
    tolerance: float = 1e-9,
) -> bool:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size != len(gate_box.route_set):
        return False
    lower = np.asarray(gate_box.lower)
    upper = np.asarray(gate_box.upper)
    return bool(
        np.all(values >= lower - tolerance)
        and np.all(values <= upper + tolerance)
        and abs(float(values.sum()) - 1.0) <= tolerance
    )


def simplex_anchor_contains(
    free_weights: Sequence[float],
    gate_box: NormalizedTopKGateBox,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Check the two inequalities retaining the omitted anchor's box bound."""
    values = np.asarray(free_weights, dtype=np.float64).reshape(-1)
    if values.size != len(gate_box.route_set) - 1:
        return False
    if values.size == 0:
        return True
    total = float(values.sum())
    return bool(
        total <= 1.0 - gate_box.lower[-1] + tolerance
        and total >= 1.0 - gate_box.upper[-1] - tolerance
    )


def decompose_normalized_topk_scalar(
    expert_scalars: Sequence[float],
    weights: Sequence[float],
    *,
    constant: float = 0.0,
) -> tuple[float, float, tuple[float, ...], tuple[float, ...]]:
    """Return direct value, anchor term, differences, and k-1 products."""
    values = np.asarray(expert_scalars, dtype=np.float64).reshape(-1)
    lambdas = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or lambdas.size != values.size:
        raise ValueError("expert values and weights must have the same nonzero size")
    if np.any(lambdas < 0.0) or not np.isclose(lambdas.sum(), 1.0):
        raise ValueError("weights must lie in the probability simplex")
    anchor_value = float(values[-1] + constant)
    differences = tuple(float(value - values[-1]) for value in values[:-1])
    products = tuple(
        float(weight * difference)
        for weight, difference in zip(lambdas[:-1], differences)
    )
    direct = float(lambdas @ values + constant)
    return direct, anchor_value, differences, products


def build_weighted_topk_range(
    experts_hz: SharedInputExpertsHZ,
    conditioned_router: SparseHZono,
    gate_box: NormalizedTopKGateBox,
    property_row: Sequence[float],
    property_constant: float,
    *,
    difference_time_limit: float,
) -> WeightedTopKRangeEncoding:
    """Build k-1 property-directed products under simplex-and-box gate bounds."""
    if experts_hz.route_set != gate_box.route_set:
        raise ValueError("gate box belongs to a different route set")
    if gate_box.conditioned_router is not conditioned_router:
        raise ValueError("gate box belongs to a different conditioned router")
    if (
        gate_box.router_frame_id != conditioned_router.frame_id
        or gate_box.router_output_width != conditioned_router.n_out
        or conditioned_router.frame_id != experts_hz.output_hz.frame_id
    ):
        raise ValueError("router and experts must share the gate-box generator frame")
    supported, reason = normalized_gate_support(gate_box.gate_kind)
    if not supported:
        raise UnsupportedNormalizedGateError(reason)

    classes = len(experts_hz.expert_rows[0])
    if any(len(rows) != classes for rows in experts_hz.expert_rows):
        raise ValueError("selected expert output widths differ")
    q = np.asarray(property_row, dtype=np.float64).reshape(-1)
    if q.size != classes:
        raise ValueError("property row width does not match expert outputs")

    route_count = len(experts_hz.route_set)
    free_count = route_count - 1
    anchor_position = route_count - 1
    scalar_W = np.zeros((route_count, route_count * classes), dtype=np.float64)
    for position in range(route_count):
        start = position * classes
        scalar_W[position, start : start + classes] = q
    expert_scalars = sparse_hz_linear(experts_hz.output_hz, scalar_W)

    expression_W = np.zeros((1 + free_count, route_count), dtype=np.float64)
    expression_W[0, anchor_position] = 1.0
    for position in range(free_count):
        expression_W[1 + position, position] = 1.0
        expression_W[1 + position, anchor_position] = -1.0
    constants = np.zeros(1 + free_count, dtype=np.float64)
    constants[0] = float(property_constant)
    expressions = sparse_hz_linear(expert_scalars, expression_W, constants)

    difference_support = None
    if free_count:
        difference_support = hz_support_bounds(
            expressions,
            range(1, 1 + free_count),
            time_limit=float(difference_time_limit),
            relax_binaries=True,
        )
        difference_lower = (
            difference_support.bounds.lb.reshape(-1).double().numpy()
        )
        difference_upper = (
            difference_support.bounds.ub.reshape(-1).double().numpy()
        )
    else:
        difference_lower = np.zeros(0, dtype=np.float64)
        difference_upper = np.zeros(0, dtype=np.float64)

    base = experts_hz.output_hz
    n_cont = base.n_cont + 2 * free_count
    lambda_slots = tuple(range(base.n_cont, base.n_cont + free_count))
    product_slots = tuple(
        range(base.n_cont + free_count, base.n_cont + 2 * free_count)
    )
    free_lower = np.asarray(gate_box.lower[:-1], dtype=np.float64)
    free_upper = np.asarray(gate_box.upper[:-1], dtype=np.float64)
    lambda_center = (free_lower + free_upper) * 0.5
    lambda_radius = (free_upper - free_lower) * 0.5

    term_bounds: list[McCormickBounds] = []
    local_A: list[np.ndarray] = []
    local_b: list[np.ndarray] = []
    product_center = np.zeros(free_count, dtype=np.float64)
    product_radius = np.zeros(free_count, dtype=np.float64)
    for position in range(free_count):
        product_lower, product_upper = _product_range(
            free_lower[position],
            free_upper[position],
            difference_lower[position],
            difference_upper[position],
        )
        product_center[position] = (product_lower + product_upper) * 0.5
        product_radius[position] = (product_upper - product_lower) * 0.5
        term_bounds.append(
            McCormickBounds(
                float(free_lower[position]),
                float(free_upper[position]),
                float(difference_lower[position]),
                float(difference_upper[position]),
                float(product_lower),
                float(product_upper),
            )
        )
        A, b = mccormick_inequalities(
            free_lower[position],
            free_upper[position],
            difference_lower[position],
            difference_upper[position],
        )
        local_A.append(A)
        local_b.append(b)

    extra_Auc_rows: list[sp.csr_matrix] = []
    extra_Aub_rows: list[sp.csr_matrix] = []
    extra_ub_rows: list[np.ndarray] = []
    for position in range(free_count):
        expression_center = np.asarray(
            [
                lambda_center[position],
                expressions.c[1 + position],
                product_center[position],
            ],
            dtype=np.float64,
        )
        expression_Gc = sp.lil_matrix((3, n_cont), dtype=np.float64)
        expression_Gc[0, lambda_slots[position]] = lambda_radius[position]
        expression_Gc[1, : base.n_cont] = expressions.Gc.getrow(1 + position)
        expression_Gc[2, product_slots[position]] = product_radius[position]
        expression_Gc = expression_Gc.tocsr()
        expression_Gb = sp.vstack(
            [
                sparse_empty(1, base.n_bin),
                expressions.Gb.getrow(1 + position),
                sparse_empty(1, base.n_bin),
            ],
            format="csr",
        )
        A = sp.csr_matrix(local_A[position])
        extra_Auc_rows.append((A @ expression_Gc).tocsr())
        extra_Aub_rows.append((A @ expression_Gb).tocsr())
        extra_ub_rows.append(local_b[position] - local_A[position] @ expression_center)

    # The omitted anchor weight is 1-sum(free weights).  These two rows retain
    # its box bounds and therefore encode the simplex intersection box, not k-1
    # independent weights with an unconstrained residual.
    if free_count:
        simplex_A = np.vstack(
            [np.ones(free_count, dtype=np.float64), -np.ones(free_count, dtype=np.float64)]
        )
        simplex_b = np.asarray(
            [1.0 - gate_box.lower[-1], -(1.0 - gate_box.upper[-1])],
            dtype=np.float64,
        )
        simplex_Gc = sp.lil_matrix((2, n_cont), dtype=np.float64)
        for position, slot in enumerate(lambda_slots):
            simplex_Gc[0, slot] = lambda_radius[position]
            simplex_Gc[1, slot] = -lambda_radius[position]
        free_center_sum = float(lambda_center.sum())
        anchor_lower = float(gate_box.lower[-1])
        anchor_upper = float(gate_box.upper[-1])
        simplex_ub = np.asarray(
            [
                1.0 - anchor_lower - free_center_sum,
                -(1.0 - anchor_upper) + free_center_sum,
            ],
            dtype=np.float64,
        )
        extra_Auc_rows.append(simplex_Gc.tocsr())
        extra_Aub_rows.append(sparse_empty(2, base.n_bin))
        extra_ub_rows.append(simplex_ub)
    else:
        simplex_A = np.zeros((0, 0), dtype=np.float64)
        simplex_b = np.zeros(0, dtype=np.float64)

    padded_Auc = sp.hstack(
        [base.Auc, sparse_empty(base.n_ineq, 2 * free_count)], format="csr"
    )
    output_Gc = sp.lil_matrix((1, n_cont), dtype=np.float64)
    output_Gc[0, : base.n_cont] = expressions.Gc.getrow(0)
    for position, slot in enumerate(product_slots):
        output_Gc[0, slot] = product_radius[position]
    output = SparseHZono(
        c=np.asarray([expressions.c[0] + product_center.sum()]),
        Gc=output_Gc.tocsr(),
        Gb=expressions.Gb.getrow(0).tocsr(),
        Ac=sp.hstack(
            [base.Ac, sparse_empty(base.n_eq, 2 * free_count)], format="csr"
        ),
        Ab=base.Ab.copy(),
        b=base.b.copy(),
        Auc=sp.vstack([padded_Auc, *extra_Auc_rows], format="csr"),
        Aub=sp.vstack([base.Aub, *extra_Aub_rows], format="csr"),
        ub=np.concatenate([base.ub, *extra_ub_rows]),
        frame_id=base.frame_id,
        exact=False if free_count else base.exact,
    )
    return WeightedTopKRangeEncoding(
        output_hz=output,
        input_hz=sparse_hz_pad_frame(
            experts_hz.input_hz, n_cont, base.n_bin
        ),
        route_set=experts_hz.route_set,
        anchor=gate_box.anchor,
        free_experts=gate_box.free_experts,
        property_row=tuple(float(value) for value in q),
        property_constant=float(property_constant),
        gate_box=gate_box,
        term_bounds=tuple(term_bounds),
        difference_support=difference_support,
        mccormick_A=tuple(local_A),
        mccormick_b=tuple(local_b),
        simplex_A=simplex_A,
        simplex_b=simplex_b,
    )


def solve_weighted_topk_range(
    encoding: WeightedTopKRangeEncoding,
    *,
    input_shape: tuple[int, ...],
    time_limit: float,
    tolerance: float = HZ_NUMERICAL_POLICY.safe_positive_margin,
) -> WeightedTopKRangeDecision:
    """Certify only positive lower bounds; relaxation candidates remain UNKNOWN."""
    result: HZMinimumResult = hz_minimize_output(
        encoding.output_hz,
        0,
        input_hz=encoding.input_hz,
        input_shape=input_shape,
        time_limit=float(time_limit),
    )
    if (
        result.status == "optimal"
        and result.minimum is not None
        and result.minimum > float(tolerance)
    ):
        status, reason = "SAFE", SAFE_WEIGHTED_RANGE
    elif result.status == "timeout":
        status, reason = "UNKNOWN", UNKNOWN_WEIGHTED_SOLVER_LIMIT
    elif result.status == "optimal":
        status, reason = "UNKNOWN", UNKNOWN_WEIGHTED_RELAXATION
    else:
        status, reason = "UNKNOWN", UNKNOWN_WEIGHTED_NUMERICAL
    return WeightedTopKRangeDecision(
        status=status,
        reason=reason,
        minimum=result.minimum,
        candidate_objective=result.candidate_objective,
        candidate_input=result.candidate_input,
        solver_status=result.solver_status,
        solver_gap=result.solver_gap,
        elapsed=result.elapsed,
        solver_certified_lower_bound=result.solver_certified_lower_bound,
        solver_bound_kind=result.solver_bound_kind,
        solver_primal_objective=result.solver_primal_objective,
        solver_dual_objective=result.solver_dual_objective,
    )
