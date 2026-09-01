# ===- act/back_end/moe/monolithic_f0.py ----------------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-----------------------------------------------------------------===#

"""Single-formulation disjunctive MILP for guarded weighted-top2 F0 branches.

Each input encoding is one feasible unordered top-2 pair and one property row.
The construction embeds their union in a single MILP with one pair selector.
It uses bounded homogenization, not an additional arbitrary big-M. The result
is exact for the union of the supplied F0 relaxations; F0 itself remains a
sound McCormick relaxation of the selected-softmax output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import Bounds as SciPyBounds
from scipy.optimize import LinearConstraint, milp

from act.back_end.solver.solver_hz import (
    HZ_NUMERICAL_POLICY,
    _certified_solver_lower_bound,
    _lower_hz_milp,
    hz_outward_slack,
)


SAFE_MONOLITHIC_WEIGHTED_RANGE = "SAFE_MONOLITHIC_WEIGHTED_RANGE"
UNKNOWN_MONOLITHIC_RELAXATION = "UNKNOWN_MONOLITHIC_RELAXATION"
UNKNOWN_MONOLITHIC_SOLVER_LIMIT = "UNKNOWN_MONOLITHIC_SOLVER_LIMIT"
UNKNOWN_MONOLITHIC_NUMERICAL = "UNKNOWN_MONOLITHIC_NUMERICAL"


@dataclass(frozen=True)
class MonolithicF0Decision:
    status: str
    reason: str
    minimum: float | None
    candidate_objective: float | None
    candidate_input: torch.Tensor | None
    active_pair: tuple[int, int] | None
    solver_status: int | None
    solver_gap: float | None
    solver_nodes: int
    elapsed: float
    pair_count: int
    variables: int
    binary_variables: int
    constraint_rows: int
    solver_certified_lower_bound: float | None
    solver_bound_kind: str | None


@dataclass(frozen=True)
class _MonolithicMILP:
    objective: np.ndarray
    integrality: np.ndarray
    var_lb: np.ndarray
    var_ub: np.ndarray
    A: sp.csr_matrix
    row_lb: np.ndarray
    row_ub: np.ndarray
    local_offsets: tuple[int, ...]
    selector_offset: int
    lowered: tuple[object, ...]


def _shifted_row(
    row: sp.csr_matrix,
    *,
    local_offset: int,
    selector_column: int,
    selector_value: float,
    width: int,
) -> sp.csr_matrix:
    source = row.tocoo()
    columns = np.concatenate(
        [source.col.astype(np.int64) + local_offset, np.asarray([selector_column])]
    )
    data = np.concatenate(
        [source.data.astype(np.float64), np.asarray([selector_value])]
    )
    return sp.csr_matrix(
        (data, (np.zeros(data.size, dtype=np.int64), columns)),
        shape=(1, width),
    )


def _unit_row(
    *,
    variable: int,
    variable_value: float,
    selector: int,
    selector_value: float,
    width: int,
) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.asarray([variable_value, selector_value], dtype=np.float64),
            (
                np.zeros(2, dtype=np.int64),
                np.asarray([variable, selector], dtype=np.int64),
            ),
        ),
        shape=(1, width),
    )


def _build_disjunction(encodings: Sequence[object]) -> _MonolithicMILP:
    if not encodings:
        raise ValueError("monolithic F0 requires at least one pair encoding")
    lowered = tuple(_lower_hz_milp(encoding.output_hz) for encoding in encodings)
    if any(model.value_center.size != 1 for model in lowered):
        raise ValueError("each F0 branch must expose one property scalar")
    offsets: list[int] = []
    cursor = 0
    for model in lowered:
        offsets.append(cursor)
        cursor += int(model.n_var)
    selector_offset = cursor
    width = selector_offset + len(lowered)
    objective = np.zeros(width, dtype=np.float64)
    integrality = np.zeros(width, dtype=np.int32)
    var_lb = np.empty(width, dtype=np.float64)
    var_ub = np.empty(width, dtype=np.float64)
    rows: list[sp.csr_matrix] = []
    row_lb: list[float] = []
    row_ub: list[float] = []

    for branch, (encoding, model, offset) in enumerate(
        zip(encodings, lowered, offsets)
    ):
        selector = selector_offset + branch
        objective[offset : offset + model.n_var] = (
            model.value_matrix.getrow(0).toarray().reshape(-1)
        )
        objective[selector] = float(model.value_center[0])
        integrality[offset : offset + model.n_var] = model.integrality
        integrality[selector] = 1
        var_lb[offset : offset + model.n_cont] = -1.0
        var_ub[offset : offset + model.n_cont] = 1.0
        var_lb[offset + model.n_cont : offset + model.n_var] = 0.0
        var_ub[offset + model.n_cont : offset + model.n_var] = 1.0
        var_lb[selector], var_ub[selector] = 0.0, 1.0

        input_hz = encoding.input_hz
        if input_hz.n_cont != model.n_cont or input_hz.n_bin != model.n_bin:
            raise ValueError("F0 input and output frames have different factor widths")

        for local in range(model.n_cont):
            variable = offset + local
            rows.append(
                _unit_row(
                    variable=variable,
                    variable_value=1.0,
                    selector=selector,
                    selector_value=-1.0,
                    width=width,
                )
            )
            row_lb.append(-np.inf)
            row_ub.append(0.0)
            rows.append(
                _unit_row(
                    variable=variable,
                    variable_value=-1.0,
                    selector=selector,
                    selector_value=-1.0,
                    width=width,
                )
            )
            row_lb.append(-np.inf)
            row_ub.append(0.0)
        for local in range(model.n_cont, model.n_var):
            rows.append(
                _unit_row(
                    variable=offset + local,
                    variable_value=1.0,
                    selector=selector,
                    selector_value=-1.0,
                    width=width,
                )
            )
            row_lb.append(-np.inf)
            row_ub.append(0.0)

        for row_index in range(model.A.shape[0]):
            source = model.A.getrow(row_index)
            upper = float(model.row_ub[row_index])
            lower = float(model.row_lb[row_index])
            if np.isfinite(upper):
                rows.append(
                    _shifted_row(
                        source,
                        local_offset=offset,
                        selector_column=selector,
                        selector_value=-upper,
                        width=width,
                    )
                )
                row_lb.append(-np.inf)
                row_ub.append(0.0)
            if np.isfinite(lower):
                rows.append(
                    _shifted_row(
                        -source,
                        local_offset=offset,
                        selector_column=selector,
                        selector_value=lower,
                        width=width,
                    )
                )
                row_lb.append(-np.inf)
                row_ub.append(0.0)

    selector_columns = np.arange(selector_offset, width, dtype=np.int64)
    rows.append(
        sp.csr_matrix(
            (
                np.ones(len(lowered), dtype=np.float64),
                (np.zeros(len(lowered), dtype=np.int64), selector_columns),
            ),
            shape=(1, width),
        )
    )
    row_lb.append(1.0)
    row_ub.append(1.0)
    return _MonolithicMILP(
        objective=objective,
        integrality=integrality,
        var_lb=var_lb,
        var_ub=var_ub,
        A=sp.vstack(rows, format="csr"),
        row_lb=np.asarray(row_lb, dtype=np.float64),
        row_ub=np.asarray(row_ub, dtype=np.float64),
        local_offsets=tuple(offsets),
        selector_offset=selector_offset,
        lowered=lowered,
    )


def _valid_point(model: _MonolithicMILP, point: np.ndarray) -> bool:
    tolerance = HZ_NUMERICAL_POLICY.feasibility_tolerance
    integrality_tolerance = HZ_NUMERICAL_POLICY.integrality_tolerance
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    if value.size != model.objective.size or not np.all(np.isfinite(value)):
        return False
    if np.any(value < model.var_lb - tolerance) or np.any(value > model.var_ub + tolerance):
        return False
    integer = value[model.integrality != 0]
    if np.any(np.abs(integer - np.rint(integer)) > integrality_tolerance):
        return False
    evaluated = np.asarray(model.A @ value).reshape(-1)
    finite_lower = np.isfinite(model.row_lb)
    finite_upper = np.isfinite(model.row_ub)
    return not (
        np.any(evaluated[finite_lower] < model.row_lb[finite_lower] - tolerance)
        or np.any(evaluated[finite_upper] > model.row_ub[finite_upper] + tolerance)
    )


def _recover_input(encoding, local_point: np.ndarray, input_shape) -> torch.Tensor:
    hz = encoding.input_hz
    continuous = local_point[: hz.n_cont]
    binary = local_point[hz.n_cont : hz.n_cont + hz.n_bin]
    center = np.asarray(hz.c, dtype=np.float64).reshape(-1)
    if hz.n_bin:
        center = center - np.asarray(hz.Gb.sum(axis=1)).reshape(-1)
    value = center + np.asarray(hz.Gc @ continuous).reshape(-1)
    if hz.n_bin:
        value = value + 2.0 * np.asarray(hz.Gb @ binary).reshape(-1)
    if int(np.prod(input_shape)) != value.size:
        raise ValueError("input shape does not match the F0 input HZ")
    full = torch.from_numpy(value.copy()).reshape(input_shape).double()
    if not input_shape or input_shape[0] != 1:
        raise ValueError("monolithic F0 currently requires batch size one")
    return full[0].clone()


def solve_monolithic_weighted_top2_f0(
    encodings: Sequence[object],
    *,
    input_shape: tuple[int, ...],
    time_limit: float,
    tolerance: float = HZ_NUMERICAL_POLICY.safe_positive_margin,
) -> MonolithicF0Decision:
    """Minimize one safety row over the union of all supplied pair branches."""
    started = time.monotonic()
    model = _build_disjunction(encodings)
    try:
        result = milp(
            c=model.objective,
            integrality=model.integrality,
            bounds=SciPyBounds(model.var_lb, model.var_ub),
            constraints=LinearConstraint(model.A, model.row_lb, model.row_ub),
            options={
                "presolve": True,
                "time_limit": max(1e-3, float(time_limit)),
                "mip_rel_gap": HZ_NUMERICAL_POLICY.mip_relative_gap,
            },
        )
    except Exception:
        return MonolithicF0Decision(
            "UNKNOWN", UNKNOWN_MONOLITHIC_NUMERICAL, None, None, None, None,
            None, None, 0, time.monotonic() - started, len(encodings),
            model.objective.size, int(model.integrality.sum()), model.A.shape[0],
            None, None,
        )
    solver_status = int(getattr(result, "status", -1))
    raw_gap = getattr(result, "mip_gap", None)
    gap = float(raw_gap) if raw_gap is not None and np.isfinite(raw_gap) else None
    nodes = int(getattr(result, "mip_node_count", 0) or 0)
    point = getattr(result, "x", None)
    valid = point is not None and _valid_point(model, point)
    candidate_objective = None
    candidate_input = None
    active_pair = None
    if valid:
        point = np.asarray(point, dtype=np.float64)
        candidate_objective = float(model.objective @ point)
        selectors = point[model.selector_offset :]
        branch = int(np.argmax(selectors))
        if selectors[branch] >= 1.0 - HZ_NUMERICAL_POLICY.integrality_tolerance:
            encoding = encodings[branch]
            local = point[
                model.local_offsets[branch] :
                model.local_offsets[branch] + model.lowered[branch].n_var
            ]
            candidate_input = _recover_input(encoding, local, input_shape)
            active_pair = tuple(int(value) for value in encoding.pair)
    certified = _certified_solver_lower_bound(result, model.integrality)
    minimum = None
    bound_value = None
    bound_kind = None
    if certified is not None:
        bound_value, bound_kind = certified
        minimum = float(
            np.nextafter(
                bound_value - hz_outward_slack(bound_value),
                -np.inf,
            )
        )
        if minimum > float(tolerance):
            status, reason = "SAFE", SAFE_MONOLITHIC_WEIGHTED_RANGE
        else:
            status, reason = "UNKNOWN", UNKNOWN_MONOLITHIC_RELAXATION
    elif solver_status == 1:
        status, reason = "UNKNOWN", UNKNOWN_MONOLITHIC_SOLVER_LIMIT
    else:
        status, reason = "UNKNOWN", UNKNOWN_MONOLITHIC_NUMERICAL
    return MonolithicF0Decision(
        status=status,
        reason=reason,
        minimum=minimum,
        candidate_objective=candidate_objective,
        candidate_input=candidate_input,
        active_pair=active_pair,
        solver_status=solver_status,
        solver_gap=gap,
        solver_nodes=nodes,
        elapsed=time.monotonic() - started,
        pair_count=len(encodings),
        variables=model.objective.size,
        binary_variables=int(model.integrality.sum()),
        constraint_rows=model.A.shape[0],
        solver_certified_lower_bound=bound_value,
        solver_bound_kind=bound_kind,
    )
