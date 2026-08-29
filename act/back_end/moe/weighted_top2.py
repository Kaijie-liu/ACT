# ===- act/back_end/moe/weighted_top2.py - Restricted Top-2 Fallback -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Sound F0 range-only fallback for selected-softmax weighted top-2 MoEs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch

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
from act.front_end.specs import OutKind, OutputSpec


SAFE_WEIGHTED_RANGE = "SAFE_WEIGHTED_RANGE"
SAFE_WEIGHTED_SEGMENTED = "SAFE_WEIGHTED_SEGMENTED"
UNSAFE_FULL_FORWARD_FALLBACK = "UNSAFE_FULL_FORWARD_FALLBACK"
UNKNOWN_WEIGHTED_RELAXATION = "UNKNOWN_WEIGHTED_RELAXATION"
UNKNOWN_WEIGHTED_SOLVER_LIMIT = "UNKNOWN_WEIGHTED_SOLVER_LIMIT"
UNKNOWN_WEIGHTED_NUMERICAL = "UNKNOWN_WEIGHTED_NUMERICAL"


@dataclass(frozen=True)
class SharedInputPairHZ:
    """Two expert outputs in one frame with shared input and private binaries."""

    output_hz: SparseHZono
    input_hz: SparseHZono
    a_rows: tuple[int, ...]
    b_rows: tuple[int, ...]
    shared_continuous: int
    shared_binary: int
    a_private_continuous: int
    b_private_continuous: int
    a_private_binary: int
    b_private_binary: int


@dataclass(frozen=True)
class McCormickBounds:
    lambda_lower: float
    lambda_upper: float
    difference_lower: float
    difference_upper: float
    product_lower: float
    product_upper: float


@dataclass(frozen=True)
class WeightedTop2GateRange:
    """Property-independent gate range for one guarded unordered pair."""

    pair: tuple[int, int]
    conditioned_router: SparseHZono = field(repr=False, compare=False)
    router_frame_id: int | None
    router_output_width: int
    margin_bounds: tuple[float, float]
    lambda_bounds: tuple[float, float]
    margin_support: HZSupportBoundsResult


@dataclass(frozen=True)
class WeightedTop2DifferenceRange:
    """Property-directed disagreement range bound to one conditioned pair."""

    pair: tuple[int, int]
    conditioned_pair_output: SparseHZono = field(repr=False, compare=False)
    pair_frame_id: int | None
    pair_output_width: int
    property_row: tuple[float, ...]
    bounds: tuple[float, float]
    support: HZSupportBoundsResult = field(repr=False, compare=False)


@dataclass(frozen=True)
class WeightedTop2F0Encoding:
    """One property-directed scalar relaxation for one feasible expert pair."""

    output_hz: SparseHZono
    input_hz: SparseHZono
    pair: tuple[int, int]
    property_row: tuple[float, ...]
    property_constant: float
    margin_bounds: tuple[float, float]
    bounds: McCormickBounds
    margin_support: HZSupportBoundsResult
    difference_support: HZSupportBoundsResult
    mccormick_A: np.ndarray
    mccormick_b: np.ndarray


@dataclass(frozen=True)
class WeightedTop2F0Decision:
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


def _remap_columns(matrix, mapping: np.ndarray, width: int):
    source = matrix.tocoo()
    if source.shape[1] != mapping.size:
        raise ValueError("sparse column mapping width mismatch")
    return sp.csr_matrix(
        (source.data, (source.row, mapping[source.col])),
        shape=(source.shape[0], int(width)),
        dtype=np.float64,
    )


def _column_mapping(shared: int, old_width: int, private_start: int) -> np.ndarray:
    if old_width < shared:
        raise ValueError("expert frame is narrower than the shared input frame")
    return np.concatenate(
        [
            np.arange(shared, dtype=np.int64),
            private_start
            + np.arange(old_width - shared, dtype=np.int64),
        ]
    )


def _assert_prefix(entry: SparseHZono, expert: SparseHZono) -> None:
    if entry.frame_id is None or expert.frame_id != entry.frame_id:
        raise ValueError("shared-input expert HZs must use the entry frame")
    if expert.n_cont < entry.n_cont or expert.n_bin < entry.n_bin:
        raise ValueError("expert HZ lost shared generator columns")

    def check(entry_c, entry_b, entry_rhs, expert_c, expert_b, expert_rhs, kind):
        rows = int(entry_c.shape[0])
        if expert_c.shape[0] < rows:
            raise ValueError(f"expert HZ lost shared {kind} constraints")
        dc = (expert_c[:rows, : entry.n_cont] - entry_c).tocsr()
        db = (expert_b[:rows, : entry.n_bin] - entry_b).tocsr()
        dc.eliminate_zeros()
        db.eliminate_zeros()
        private_c = expert_c[:rows, entry.n_cont :]
        private_b = expert_b[:rows, entry.n_bin :]
        if (
            dc.nnz
            or db.nnz
            or private_c.nnz
            or private_b.nnz
            or not np.array_equal(
                np.asarray(expert_rhs[:rows]), np.asarray(entry_rhs)
            )
        ):
            raise ValueError(f"expert HZ changed shared {kind} constraint identity")

    check(entry.Ac, entry.Ab, entry.b, expert.Ac, expert.Ab, expert.b, "equality")
    check(
        entry.Auc,
        entry.Aub,
        entry.ub,
        expert.Auc,
        expert.Aub,
        expert.ub,
        "inequality",
    )


def shared_input_pair_hz(
    entry: SparseHZono,
    expert_a: SparseHZono,
    expert_b: SparseHZono,
) -> SharedInputPairHZ:
    """Merge independently propagated experts without aliasing private factors."""
    _assert_prefix(entry, expert_a)
    _assert_prefix(entry, expert_b)
    if expert_a.n_out != expert_b.n_out:
        raise ValueError("paired experts must have the same output width")

    a_pc = expert_a.n_cont - entry.n_cont
    b_pc = expert_b.n_cont - entry.n_cont
    a_pb = expert_a.n_bin - entry.n_bin
    b_pb = expert_b.n_bin - entry.n_bin
    n_cont = entry.n_cont + a_pc + b_pc
    n_bin = entry.n_bin + a_pb + b_pb
    entry_c_map = np.arange(entry.n_cont, dtype=np.int64)
    entry_b_map = np.arange(entry.n_bin, dtype=np.int64)
    a_c_map = _column_mapping(entry.n_cont, expert_a.n_cont, entry.n_cont)
    b_c_map = _column_mapping(
        entry.n_cont,
        expert_b.n_cont,
        entry.n_cont + a_pc,
    )
    a_b_map = _column_mapping(entry.n_bin, expert_a.n_bin, entry.n_bin)
    b_b_map = _column_mapping(
        entry.n_bin,
        expert_b.n_bin,
        entry.n_bin + a_pb,
    )

    def remap(hz, c_map, b_map, c_name, b_name):
        return (
            _remap_columns(getattr(hz, c_name), c_map, n_cont),
            _remap_columns(getattr(hz, b_name), b_map, n_bin),
        )

    entry_Ac, entry_Ab = remap(entry, entry_c_map, entry_b_map, "Ac", "Ab")
    entry_Auc, entry_Aub = remap(
        entry, entry_c_map, entry_b_map, "Auc", "Aub"
    )
    a_Ac, a_Ab = remap(expert_a, a_c_map, a_b_map, "Ac", "Ab")
    b_Ac, b_Ab = remap(expert_b, b_c_map, b_b_map, "Ac", "Ab")
    a_Auc, a_Aub = remap(expert_a, a_c_map, a_b_map, "Auc", "Aub")
    b_Auc, b_Aub = remap(expert_b, b_c_map, b_b_map, "Auc", "Aub")
    a_Gc = _remap_columns(expert_a.Gc, a_c_map, n_cont)
    b_Gc = _remap_columns(expert_b.Gc, b_c_map, n_cont)
    a_Gb = _remap_columns(expert_a.Gb, a_b_map, n_bin)
    b_Gb = _remap_columns(expert_b.Gb, b_b_map, n_bin)

    output = SparseHZono(
        c=np.concatenate([expert_a.c, expert_b.c]),
        Gc=sp.vstack([a_Gc, b_Gc], format="csr"),
        Gb=sp.vstack([a_Gb, b_Gb], format="csr"),
        Ac=sp.vstack(
            [
                entry_Ac,
                a_Ac[entry.n_eq :],
                b_Ac[entry.n_eq :],
            ],
            format="csr",
        ),
        Ab=sp.vstack(
            [
                entry_Ab,
                a_Ab[entry.n_eq :],
                b_Ab[entry.n_eq :],
            ],
            format="csr",
        ),
        b=np.concatenate(
            [entry.b, expert_a.b[entry.n_eq :], expert_b.b[entry.n_eq :]]
        ),
        Auc=sp.vstack(
            [
                entry_Auc,
                a_Auc[entry.n_ineq :],
                b_Auc[entry.n_ineq :],
            ],
            format="csr",
        ),
        Aub=sp.vstack(
            [
                entry_Aub,
                a_Aub[entry.n_ineq :],
                b_Aub[entry.n_ineq :],
            ],
            format="csr",
        ),
        ub=np.concatenate(
            [entry.ub, expert_a.ub[entry.n_ineq :], expert_b.ub[entry.n_ineq :]]
        ),
        frame_id=entry.frame_id,
        exact=entry.exact and expert_a.exact and expert_b.exact,
    )
    width = expert_a.n_out
    return SharedInputPairHZ(
        output_hz=output,
        input_hz=sparse_hz_pad_frame(entry, n_cont, n_bin),
        a_rows=tuple(range(width)),
        b_rows=tuple(range(width, 2 * width)),
        shared_continuous=entry.n_cont,
        shared_binary=entry.n_bin,
        a_private_continuous=a_pc,
        b_private_continuous=b_pc,
        a_private_binary=a_pb,
        b_private_binary=b_pb,
    )


def mccormick_inequalities(
    lambda_lower: float,
    lambda_upper: float,
    difference_lower: float,
    difference_upper: float,
    *,
    reverse_row: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A @ [lambda, difference, product] <= b`` for the hull."""
    ll, lu = float(lambda_lower), float(lambda_upper)
    dl, du = float(difference_lower), float(difference_upper)
    if ll > lu or dl > du:
        raise ValueError("McCormick bounds are inconsistent")
    A = np.asarray(
        [
            [dl, ll, -1.0],
            [du, lu, -1.0],
            [-dl, -lu, 1.0],
            [-du, -ll, 1.0],
        ],
        dtype=np.float64,
    )
    b = np.asarray(
        [ll * dl, lu * du, -lu * dl, -ll * du],
        dtype=np.float64,
    )
    if reverse_row is not None:
        index = int(reverse_row)
        if index < 0 or index >= 4:
            raise IndexError("McCormick mutation row is out of range")
        A[index] *= -1.0
        b[index] *= -1.0
    return A, b


def mccormick_contains(
    lambda_value: float,
    difference: float,
    product: float,
    A: np.ndarray,
    b: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> bool:
    point = np.asarray([lambda_value, difference, product], dtype=np.float64)
    return bool(np.all(np.asarray(A) @ point <= np.asarray(b) + tolerance))


def _sigmoid_range(lower: float, upper: float) -> tuple[float, float]:
    if lower > upper:
        raise ValueError("sigmoid margin bounds are inconsistent")
    if lower == 0.0 and upper == 0.0:
        return 0.5, 0.5
    values = torch.sigmoid(
        torch.tensor([lower, upper], dtype=torch.float64, device="cpu")
    ).numpy()
    return (
        float(np.nextafter(values[0], 0.0)),
        float(np.nextafter(values[1], 1.0)),
    )


def _product_range(ll: float, lu: float, dl: float, du: float) -> tuple[float, float]:
    corners = np.asarray([ll * dl, ll * du, lu * dl, lu * du])
    return float(corners.min()), float(corners.max())


def _canonical_top2_pair(pair: Sequence[int]) -> tuple[int, int]:
    selected = tuple(sorted(int(value) for value in pair))
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError("weighted top-2 fallback requires one distinct pair")
    return selected


def compute_weighted_top2_gate_range(
    conditioned_router: SparseHZono,
    pair: Sequence[int],
    *,
    time_limit: float,
) -> WeightedTop2GateRange:
    """Compute the guarded router-margin range once for a feasible pair.

    The selected-softmax top-2 gate depends only on the guarded router and the
    unordered expert pair. It is independent of the downstream property row,
    so callers may safely reuse this immutable result for every property on the
    same pair.
    """
    selected = _canonical_top2_pair(pair)
    if selected[0] < 0 or selected[1] >= conditioned_router.n_out:
        raise IndexError("weighted top-2 pair is outside the router output")
    margin_W = np.zeros((1, conditioned_router.n_out), dtype=np.float64)
    margin_W[0, selected[0]] = 1.0
    margin_W[0, selected[1]] = -1.0
    margin_hz = sparse_hz_linear(conditioned_router, margin_W)
    margin_support = hz_support_bounds(
        margin_hz,
        [0],
        time_limit=float(time_limit),
        relax_binaries=False,
    )
    margin_lower = float(margin_support.bounds.lb.item())
    margin_upper = float(margin_support.bounds.ub.item())
    return WeightedTop2GateRange(
        pair=selected,
        conditioned_router=conditioned_router,
        router_frame_id=conditioned_router.frame_id,
        router_output_width=conditioned_router.n_out,
        margin_bounds=(margin_lower, margin_upper),
        lambda_bounds=_sigmoid_range(margin_lower, margin_upper),
        margin_support=margin_support,
    )


def compute_weighted_top2_difference_range(
    pair_hz: SharedInputPairHZ,
    pair: Sequence[int],
    property_row: Sequence[float],
    *,
    time_limit: float,
) -> WeightedTop2DifferenceRange:
    """Compute and identity-bind ``q.T @ (E_a-E_b)`` support."""
    selected = _canonical_top2_pair(pair)
    classes = len(pair_hz.a_rows)
    q = np.asarray(property_row, dtype=np.float64).reshape(-1)
    if q.size != classes or len(pair_hz.b_rows) != classes:
        raise ValueError("property row width does not match expert outputs")
    difference_W = np.zeros((1, 2 * classes), dtype=np.float64)
    difference_W[0, :classes] = q
    difference_W[0, classes:] = -q
    difference_hz = sparse_hz_linear(pair_hz.output_hz, difference_W)
    support = hz_support_bounds(
        difference_hz,
        [0],
        time_limit=float(time_limit),
        relax_binaries=True,
    )
    return WeightedTop2DifferenceRange(
        pair=selected,
        conditioned_pair_output=pair_hz.output_hz,
        pair_frame_id=pair_hz.output_hz.frame_id,
        pair_output_width=pair_hz.output_hz.n_out,
        property_row=tuple(float(value) for value in q),
        bounds=(
            float(support.bounds.lb.item()),
            float(support.bounds.ub.item()),
        ),
        support=support,
    )


def linear_safety_rows(
    output_spec: OutputSpec,
    n_out: int,
) -> tuple[tuple[np.ndarray, float], ...]:
    """Lower ordinary linear properties to ``q.T @ output + c >= 0`` rows."""
    if output_spec.kind == OutKind.UNSAFE_LINEAR:
        raise NotImplementedError("UNSAFE_LINEAR is not a conjunction of safe rows")
    encoded = output_spec.encode_linear(
        B=1,
        n_out=int(n_out),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    C = encoded["C"].detach().cpu().double().numpy()
    thresholds = encoded["thresholds"].detach().cpu().double().numpy().reshape(-1)
    return tuple((-C[row].copy(), float(thresholds[row])) for row in range(C.shape[0]))


def build_weighted_top2_f0(
    pair_hz: SharedInputPairHZ,
    conditioned_router: SparseHZono,
    pair: Sequence[int],
    property_row: Sequence[float],
    property_constant: float,
    *,
    margin_time_limit: float | None = None,
    difference_time_limit: float,
    gate_range: WeightedTop2GateRange | None = None,
    difference_range: WeightedTop2DifferenceRange | None = None,
) -> WeightedTop2F0Encoding:
    """Build the range-only sigmoid + McCormick relaxation for one safe row."""
    selected = _canonical_top2_pair(pair)
    if conditioned_router.frame_id != pair_hz.output_hz.frame_id:
        raise ValueError("router and expert pair must share one generator frame")
    classes = len(pair_hz.a_rows)
    q = np.asarray(property_row, dtype=np.float64).reshape(-1)
    if q.size != classes or len(pair_hz.b_rows) != classes:
        raise ValueError("property row width does not match expert outputs")

    if gate_range is None:
        if margin_time_limit is None:
            raise ValueError("margin_time_limit is required without a gate range")
        gate_range = compute_weighted_top2_gate_range(
            conditioned_router,
            selected,
            time_limit=float(margin_time_limit),
        )
    if gate_range.pair != selected:
        raise ValueError("gate range belongs to a different expert pair")
    if gate_range.conditioned_router is not conditioned_router:
        raise ValueError("gate range belongs to a different conditioned router")
    if (
        gate_range.router_frame_id != conditioned_router.frame_id
        or gate_range.router_output_width != conditioned_router.n_out
    ):
        raise ValueError("gate range belongs to a different router frame")
    margin_lower, margin_upper = gate_range.margin_bounds
    lambda_lower, lambda_upper = gate_range.lambda_bounds
    margin_support = gate_range.margin_support

    W = np.zeros((2, 2 * classes), dtype=np.float64)
    W[0, classes:] = q
    W[1, :classes] = q
    W[1, classes:] = -q
    ud = sparse_hz_linear(
        pair_hz.output_hz,
        W,
        np.asarray([float(property_constant), 0.0]),
    )
    if difference_range is None:
        difference_range = compute_weighted_top2_difference_range(
            pair_hz,
            selected,
            q,
            time_limit=float(difference_time_limit),
        )
    if difference_range.pair != selected:
        raise ValueError("difference range belongs to a different expert pair")
    if difference_range.conditioned_pair_output is not pair_hz.output_hz:
        raise ValueError("difference range belongs to a different conditioned pair")
    if (
        difference_range.pair_frame_id != pair_hz.output_hz.frame_id
        or difference_range.pair_output_width != pair_hz.output_hz.n_out
        or difference_range.property_row != tuple(float(value) for value in q)
    ):
        raise ValueError("difference range belongs to a different property domain")
    difference_support = difference_range.support
    difference_lower, difference_upper = difference_range.bounds
    product_lower, product_upper = _product_range(
        lambda_lower,
        lambda_upper,
        difference_lower,
        difference_upper,
    )
    bounds = McCormickBounds(
        lambda_lower,
        lambda_upper,
        difference_lower,
        difference_upper,
        product_lower,
        product_upper,
    )
    mccormick_A, mccormick_b = mccormick_inequalities(
        lambda_lower,
        lambda_upper,
        difference_lower,
        difference_upper,
    )

    base = pair_hz.output_hz
    n_cont = base.n_cont + 2
    lambda_slot, product_slot = base.n_cont, base.n_cont + 1
    lambda_center = (lambda_lower + lambda_upper) * 0.5
    lambda_radius = (lambda_upper - lambda_lower) * 0.5
    product_center = (product_lower + product_upper) * 0.5
    product_radius = (product_upper - product_lower) * 0.5
    expression_center = np.asarray(
        [lambda_center, ud.c[1], product_center], dtype=np.float64
    )
    expression_Gc = sp.lil_matrix((3, n_cont), dtype=np.float64)
    expression_Gc[0, lambda_slot] = lambda_radius
    expression_Gc[1, : base.n_cont] = ud.Gc.getrow(1)
    expression_Gc[2, product_slot] = product_radius
    expression_Gc = expression_Gc.tocsr()
    expression_Gb = sp.vstack(
        [
            sparse_empty(1, base.n_bin),
            ud.Gb.getrow(1),
            sparse_empty(1, base.n_bin),
        ],
        format="csr",
    )
    extra_Auc = (sp.csr_matrix(mccormick_A) @ expression_Gc).tocsr()
    extra_Aub = (sp.csr_matrix(mccormick_A) @ expression_Gb).tocsr()
    extra_ub = mccormick_b - mccormick_A @ expression_center

    output_Gc = sp.lil_matrix((1, n_cont), dtype=np.float64)
    output_Gc[0, : base.n_cont] = ud.Gc.getrow(0)
    output_Gc[0, product_slot] = product_radius
    output = SparseHZono(
        c=np.asarray([ud.c[0] + product_center]),
        Gc=output_Gc.tocsr(),
        Gb=ud.Gb.getrow(0).tocsr(),
        Ac=sp.hstack(
            [base.Ac, sparse_empty(base.n_eq, 2)], format="csr"
        ),
        Ab=base.Ab.copy(),
        b=base.b.copy(),
        Auc=sp.vstack(
            [
                sp.hstack(
                    [base.Auc, sparse_empty(base.n_ineq, 2)],
                    format="csr",
                ),
                extra_Auc,
            ],
            format="csr",
        ),
        Aub=sp.vstack([base.Aub, extra_Aub], format="csr"),
        ub=np.concatenate([base.ub, extra_ub]),
        frame_id=base.frame_id,
        exact=False,
    )
    return WeightedTop2F0Encoding(
        output_hz=output,
        input_hz=sparse_hz_pad_frame(pair_hz.input_hz, n_cont, base.n_bin),
        pair=selected,
        property_row=tuple(float(value) for value in q),
        property_constant=float(property_constant),
        margin_bounds=(margin_lower, margin_upper),
        bounds=bounds,
        margin_support=margin_support,
        difference_support=difference_support,
        mccormick_A=mccormick_A,
        mccormick_b=mccormick_b,
    )


def solve_weighted_top2_f0(
    encoding: WeightedTop2F0Encoding,
    *,
    input_shape: tuple[int, ...],
    time_limit: float,
    tolerance: float = HZ_NUMERICAL_POLICY.safe_positive_margin,
) -> WeightedTop2F0Decision:
    """Solve F0 without ever treating a relaxation witness as unsafe."""
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
    return WeightedTop2F0Decision(
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
