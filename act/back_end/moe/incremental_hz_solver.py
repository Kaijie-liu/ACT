# ===- incremental_hz_solver.py - Incremental expert HZ solves --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Incremental HiGHS backend for repeated expert support/property queries.

One lowered sparse-HZ branch is loaded once.  Support and property queries then
change only column costs and a reusable pool of temporary rows.  Every highspy
warning is treated as an error: support sides retain their sound generator
fallback, while property queries return UNKNOWN.  A FALSIFIED property result
still requires an exact shared input frame and a solver point that passes ACT's
full HZ feasibility/integrality validation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Sequence

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import (
    HZ_NUMERICAL_POLICY,
    HZMinimumResult,
    HZSolver,
    HZSupportBoundsResult,
    SparseHZono,
    _lower_hz_milp,
    _valid_milp_point,
    hz_outward_slack,
    sparse_hz_fast_bounds,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyResult, VerifyStatus

if TYPE_CHECKING:
    from act.back_end.solver.solver_hz import _HZMILP

try:
    import highspy

    _HAS_HIGHSPY = True
except ImportError:  # pragma: no cover - dependency-light installations
    highspy = None
    _HAS_HIGHSPY = False


@dataclass(frozen=True)
class IncrementalHZTelemetry:
    backend: str
    model_builds: int
    model_build_failures: int
    model_build_seconds: float
    objective_update_calls: int
    objective_coefficients_changed: int
    objective_update_seconds: float
    row_pool_additions: int
    row_update_calls: int
    row_coefficients_changed: int
    row_bound_updates: int
    row_update_seconds: float
    integrality_update_calls: int
    budget_extension_calls: int
    budget_extension_seconds: float
    solves: int
    cold_start_solves: int
    basis_submission_attempts: int
    basis_submissions_accepted: int
    basis_valid_after_solve: int
    status_counts: tuple[tuple[str, int], ...]
    simplex_iterations: int
    ipm_iterations: int
    mip_nodes: int
    solve_seconds: float
    total_seconds: float
    build_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "status_counts": dict(self.status_counts),
            "basis_semantics": (
                "accepted submission only; solver-internal use is not claimed"
            ),
            "warm_start_claimed": False,
            "warnings_fail_closed": True,
            "small_matrix_value": 1e-12,
        }


@dataclass(frozen=True)
class _QueryResult:
    status: str
    objective: float | None
    certified_lower: float | None
    bound_kind: str | None
    point: np.ndarray | None
    solver_status: int | None
    gap: float | None
    primal: float | None
    dual: float | None
    reason: str


class _BackendFailure(RuntimeError):
    pass


class IncrementalHZBranchSolver:
    """Reusable highspy model for one sparse-HZ expert branch."""

    def __init__(
        self,
        hz: SparseHZono,
        *,
        time_limit: float = 300.0,
        relax_binaries: bool = False,
        submit_basis: bool = True,
    ) -> None:
        if not isinstance(hz, SparseHZono):
            raise TypeError("incremental expert backend requires a SparseHZono")
        if not np.isfinite(float(time_limit)) or float(time_limit) < 0.0:
            raise ValueError("time_limit must be finite and non-negative")
        self.hz = hz
        self.model: _HZMILP = _lower_hz_milp(hz)
        self.relax_binaries = bool(relax_binaries)
        self.submit_basis = bool(submit_basis)
        self.started = time.monotonic()
        self.deadline = self.started + float(time_limit)
        self._solver = None
        self._build_error: str | None = None
        self._model_builds = 0
        self._model_build_failures = 0
        self._model_build_seconds = 0.0
        self._objective_calls = 0
        self._objective_coefficients = 0
        self._objective_seconds = 0.0
        self._row_pool_additions = 0
        self._row_calls = 0
        self._row_coefficients = 0
        self._row_bounds = 0
        self._row_seconds = 0.0
        self._integrality_calls = 0
        self._budget_extension_calls = 0
        self._budget_extension_seconds = 0.0
        self._solves = 0
        self._cold_solves = 0
        self._basis_attempts = 0
        self._basis_accepted = 0
        self._basis_valid = 0
        self._simplex_iterations = 0
        self._ipm_iterations = 0
        self._mip_nodes = 0
        self._solve_seconds = 0.0
        self._statuses: Counter[str] = Counter()
        self._last_basis = None
        self._previous_objective: dict[int, float] = {}
        self._scratch_rows: list[int] = []
        self._scratch_coefficients: list[dict[int, float]] = []
        self._fast = sparse_hz_fast_bounds(hz)
        self._build()

    @property
    def available(self) -> bool:
        return self._solver is not None and self._build_error is None

    def extend_budget(self, additional_seconds: float) -> None:
        """Extend the cumulative deadline without rebuilding solver state."""
        seconds = float(additional_seconds)
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError("budget extension must be finite and non-negative")
        self.deadline += seconds
        self._budget_extension_calls += 1
        self._budget_extension_seconds += seconds

    def _require_ok(self, status, operation: str) -> None:
        if status != highspy.HighsStatus.kOk:
            raise _BackendFailure(f"{operation}:{status}")

    def _set_option(self, name: str, value) -> None:
        self._require_ok(self._solver.setOptionValue(name, value), f"option_{name}")

    @staticmethod
    def _reject_dropped_matrix_values(matrix: sp.csr_matrix, source: str) -> None:
        # changeCoeff reports kOk while emitting a warning and dropping these
        # values, unlike addRows which currently returns kWarning.  Detect the
        # lossy case ourselves so warning handling is uniform across APIs.
        nonzero = np.abs(np.asarray(matrix.data, dtype=np.float64))
        if np.any((nonzero > 0.0) & (nonzero <= 1e-12)):
            raise _BackendFailure(
                f"{source}_coefficient_below_small_matrix_value"
            )

    def _build(self) -> None:
        build_started = time.monotonic()
        if not _HAS_HIGHSPY:
            self._build_error = "highspy_unavailable"
            self._model_build_failures = 1
            return
        try:
            solver = highspy.Highs()
            self._solver = solver
            self._set_option("output_flag", False)
            self._set_option("parallel", "off")
            self._set_option("random_seed", 0)
            self._set_option("presolve", "on")
            self._set_option("small_matrix_value", 1e-12)
            self._set_option(
                "primal_feasibility_tolerance",
                HZ_NUMERICAL_POLICY.feasibility_tolerance,
            )
            self._set_option(
                "dual_feasibility_tolerance",
                HZ_NUMERICAL_POLICY.feasibility_tolerance,
            )
            self._set_option(
                "mip_feasibility_tolerance",
                HZ_NUMERICAL_POLICY.integrality_tolerance,
            )
            self._set_option("mip_rel_gap", HZ_NUMERICAL_POLICY.mip_relative_gap)
            self._require_ok(
                solver.addVars(
                    self.model.n_var, self.model.var_lb, self.model.var_ub
                ),
                "add_vars",
            )
            if self.model.n_bin and not self.relax_binaries:
                indices = np.arange(
                    self.model.n_cont, self.model.n_var, dtype=np.int32
                )
                types = np.full(
                    indices.size,
                    highspy.HighsVarType.kInteger,
                    dtype=np.uint8,
                )
                self._require_ok(
                    solver.changeColsIntegrality(indices.size, indices, types),
                    "set_integrality",
                )
                self._integrality_calls += 1
            if self.model.A.shape[0]:
                matrix = self.model.A.tocsr(copy=True)
                matrix.sum_duplicates()
                matrix.sort_indices()
                self._reject_dropped_matrix_values(matrix, "base_row")
                self._require_ok(
                    solver.addRows(
                        matrix.shape[0],
                        self.model.row_lb,
                        self.model.row_ub,
                        matrix.nnz,
                        np.asarray(matrix.indptr, dtype=np.int32),
                        np.asarray(matrix.indices, dtype=np.int32),
                        np.asarray(matrix.data, dtype=np.float64),
                    ),
                    "add_base_rows",
                )
            self._model_builds = 1
        except Exception as exc:
            self._solver = None
            self._build_error = f"{type(exc).__name__}:{exc}"
            self._model_build_failures = 1
            self._statuses["build_failed"] += 1
        finally:
            self._model_build_seconds = time.monotonic() - build_started

    def telemetry(self) -> IncrementalHZTelemetry:
        return IncrementalHZTelemetry(
            backend="highspy_incremental_hz",
            model_builds=self._model_builds,
            model_build_failures=self._model_build_failures,
            model_build_seconds=self._model_build_seconds,
            objective_update_calls=self._objective_calls,
            objective_coefficients_changed=self._objective_coefficients,
            objective_update_seconds=self._objective_seconds,
            row_pool_additions=self._row_pool_additions,
            row_update_calls=self._row_calls,
            row_coefficients_changed=self._row_coefficients,
            row_bound_updates=self._row_bounds,
            row_update_seconds=self._row_seconds,
            integrality_update_calls=self._integrality_calls,
            budget_extension_calls=self._budget_extension_calls,
            budget_extension_seconds=self._budget_extension_seconds,
            solves=self._solves,
            cold_start_solves=self._cold_solves,
            basis_submission_attempts=self._basis_attempts,
            basis_submissions_accepted=self._basis_accepted,
            basis_valid_after_solve=self._basis_valid,
            status_counts=tuple(sorted(self._statuses.items())),
            simplex_iterations=self._simplex_iterations,
            ipm_iterations=self._ipm_iterations,
            mip_nodes=self._mip_nodes,
            solve_seconds=self._solve_seconds,
            total_seconds=time.monotonic() - self.started,
            build_error=self._build_error,
        )

    def _set_objective(self, objective: sp.csr_matrix | np.ndarray) -> None:
        started = time.monotonic()
        vector = sp.csr_matrix(objective).reshape((1, self.model.n_var))
        vector.sum_duplicates()
        current = {
            int(index): float(value)
            for index, value in zip(vector.indices, vector.data)
            if value != 0.0
        }
        changed = sorted(set(self._previous_objective) | set(current))
        indices = np.asarray(changed, dtype=np.int32)
        values = np.asarray([current.get(index, 0.0) for index in changed])
        status = self._solver.changeColsCost(len(changed), indices, values)
        self._objective_calls += 1
        self._objective_coefficients += len(changed)
        self._objective_seconds += time.monotonic() - started
        self._require_ok(status, "change_objective")
        self._previous_objective = current

    def _grow_row_pool(self, count: int) -> None:
        missing = int(count) - len(self._scratch_rows)
        if missing <= 0:
            return
        first = self.model.A.shape[0] + len(self._scratch_rows)
        status = self._solver.addRows(
            missing,
            np.full(missing, -highspy.kHighsInf),
            np.full(missing, highspy.kHighsInf),
            0,
            np.zeros(missing + 1, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )
        self._require_ok(status, "grow_scratch_rows")
        self._scratch_rows.extend(range(first, first + missing))
        self._scratch_coefficients.extend({} for _ in range(missing))
        self._row_pool_additions += missing
        self._last_basis = None

    def _set_extra_rows(
        self,
        matrix: sp.csr_matrix | np.ndarray | None,
        lower: Sequence[float] | None,
        upper: Sequence[float] | None,
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        started = time.monotonic()
        if matrix is None:
            current = sp.csr_matrix((0, self.model.n_var), dtype=np.float64)
            row_lb = np.zeros(0, dtype=np.float64)
            row_ub = np.zeros(0, dtype=np.float64)
        else:
            current = sp.csr_matrix(matrix, dtype=np.float64)
            if current.shape[1] != self.model.n_var:
                raise _BackendFailure("extra_row_width_mismatch")
            row_lb = np.asarray(lower, dtype=np.float64).reshape(-1)
            row_ub = np.asarray(upper, dtype=np.float64).reshape(-1)
            if row_lb.size != current.shape[0] or row_ub.size != current.shape[0]:
                raise _BackendFailure("extra_row_bound_shape_mismatch")
            if np.any(row_lb > row_ub):
                raise _BackendFailure("extra_row_bounds_inconsistent")
            current.sum_duplicates()
            current.sort_indices()
            self._reject_dropped_matrix_values(current, "extra_row")
        self._grow_row_pool(current.shape[0])
        for position, solver_row in enumerate(self._scratch_rows):
            if position < current.shape[0]:
                row = current.getrow(position)
                values = {
                    int(index): float(value)
                    for index, value in zip(row.indices, row.data)
                    if value != 0.0
                }
                lb, ub = float(row_lb[position]), float(row_ub[position])
            else:
                values = {}
                lb, ub = -highspy.kHighsInf, highspy.kHighsInf
            previous = self._scratch_coefficients[position]
            for column in sorted(set(previous) | set(values)):
                status = self._solver.changeCoeff(
                    solver_row, column, values.get(column, 0.0)
                )
                self._row_calls += 1
                self._row_coefficients += 1
                self._require_ok(status, "change_row_coefficient")
            status = self._solver.changeRowBounds(solver_row, lb, ub)
            self._row_calls += 1
            self._row_bounds += 1
            self._require_ok(status, "change_row_bounds")
            self._scratch_coefficients[position] = values
        self._row_seconds += time.monotonic() - started
        return current, row_lb, row_ub

    def _run(
        self,
        objective: sp.csr_matrix | np.ndarray,
        *,
        extra_A: sp.csr_matrix | np.ndarray | None = None,
        extra_lb: Sequence[float] | None = None,
        extra_ub: Sequence[float] | None = None,
    ) -> _QueryResult:
        if self.model.n_var == 0:
            if extra_A is None:
                rows = sp.csr_matrix((0, 0), dtype=np.float64)
                row_lb = np.zeros(0, dtype=np.float64)
                row_ub = np.zeros(0, dtype=np.float64)
            else:
                rows = sp.csr_matrix(extra_A, dtype=np.float64)
                row_lb = np.asarray(extra_lb, dtype=np.float64).reshape(-1)
                row_ub = np.asarray(extra_ub, dtype=np.float64).reshape(-1)
                if rows.shape != (row_lb.size, 0) or row_ub.size != row_lb.size:
                    return _QueryResult(
                        "unknown", None, None, None, None, None, None, None,
                        None, "point_extra_row_shape_mismatch",
                    )
            combined_A = sp.vstack([self.model.A, rows], format="csr")
            combined_lb = np.concatenate([self.model.row_lb, row_lb])
            combined_ub = np.concatenate([self.model.row_ub, row_ub])
            point = np.zeros(0, dtype=np.float64)
            if _valid_milp_point(
                self.model,
                point,
                combined_A,
                combined_lb,
                combined_ub,
                HZ_NUMERICAL_POLICY.feasibility_tolerance,
                HZ_NUMERICAL_POLICY.integrality_tolerance,
            ):
                self._statuses["point_optimal"] += 1
                return _QueryResult(
                    "optimal", 0.0, 0.0, "point_exact", point, 0, 0.0,
                    0.0, 0.0, "point_optimal",
                )
            self._statuses["point_infeasible"] += 1
            return _QueryResult(
                "infeasible", None, None, None, None, 2, None, None, None,
                "point_infeasible",
            )
        if not self.available:
            return _QueryResult(
                "unknown", None, None, None, None, None, None, None, None,
                f"backend_unavailable:{self._build_error}",
            )
        remaining = self.deadline - time.monotonic()
        if remaining <= 0.0:
            self._statuses["budget_exhausted"] += 1
            return _QueryResult(
                "timeout", None, None, None, None, None, None, None, None,
                "budget_exhausted",
            )
        try:
            vector = sp.csr_matrix(objective).reshape((1, self.model.n_var))
            rows, row_lb, row_ub = self._set_extra_rows(
                extra_A, extra_lb, extra_ub
            )
            self._set_objective(vector)
            accepted_basis = False
            if self.submit_basis and self._last_basis is not None:
                self._basis_attempts += 1
                status = self._solver.setBasis(self._last_basis)
                if status != highspy.HighsStatus.kOk:
                    raise _BackendFailure(f"set_basis:{status}")
                accepted_basis = True
                self._basis_accepted += 1
            if not accepted_basis:
                self._cold_solves += 1
            self._set_option("time_limit", max(1e-3, remaining))
            solve_started = time.monotonic()
            run_status = self._solver.run()
            self._solve_seconds += time.monotonic() - solve_started
            self._solves += 1
            if run_status != highspy.HighsStatus.kOk:
                raise _BackendFailure(f"run:{run_status}")
        except Exception as exc:
            reason = f"update_or_run_failed:{type(exc).__name__}:{exc}"
            self._statuses[reason] += 1
            self._last_basis = None
            # A warning can occur after only part of an objective or scratch
            # row was changed.  Never reuse that partially updated model.
            self._solver = None
            self._build_error = reason
            return _QueryResult(
                "unknown", None, None, None, None, None, None, None, None, reason
            )

        status = self._solver.getModelStatus()
        status_name = self._solver.modelStatusToString(status).lower().replace(" ", "_")
        self._statuses[status_name] += 1
        info = self._solver.getInfo()
        self._simplex_iterations += max(0, int(info.simplex_iteration_count))
        self._ipm_iterations += max(0, int(info.ipm_iteration_count))
        self._mip_nodes += max(0, int(info.mip_node_count))
        basis = self._solver.getBasis()
        if bool(getattr(basis, "valid", False)):
            self._basis_valid += 1
            self._last_basis = basis
        else:
            self._last_basis = None

        solution = self._solver.getSolution()
        point = None
        combined_A = sp.vstack([self.model.A, rows], format="csr")
        combined_lb = np.concatenate([self.model.row_lb, row_lb])
        combined_ub = np.concatenate([self.model.row_ub, row_ub])
        if bool(getattr(solution, "value_valid", False)):
            candidate = np.asarray(solution.col_value, dtype=np.float64)
            if _valid_milp_point(
                self.model,
                candidate,
                combined_A,
                combined_lb,
                combined_ub,
                HZ_NUMERICAL_POLICY.feasibility_tolerance,
                HZ_NUMERICAL_POLICY.integrality_tolerance,
            ):
                point = candidate
        solver_status = int(status.value)
        raw_primal = float(info.objective_function_value)
        primal = raw_primal if np.isfinite(raw_primal) else None
        raw_dual = float(info.mip_dual_bound)
        dual = raw_dual if np.isfinite(raw_dual) else None
        raw_gap = float(info.mip_gap)
        gap = raw_gap if np.isfinite(raw_gap) else None
        if status == highspy.HighsModelStatus.kOptimal:
            objective_value = float(self._solver.getObjectiveValue())
            if not np.isfinite(objective_value):
                return _QueryResult(
                    "unknown", None, None, None, point, solver_status, gap,
                    primal, dual, "nonfinite_objective",
                )
            if self.model.n_bin and not self.relax_binaries:
                certified = dual
                bound_kind = "highs_mip_dual_bound"
            else:
                certified = objective_value
                bound_kind = "highs_lp_optimum"
            if certified is None:
                return _QueryResult(
                    "unknown", objective_value, None, None, point, solver_status,
                    gap, primal, dual, "missing_certified_bound",
                )
            return _QueryResult(
                "optimal", objective_value, certified, bound_kind, point,
                solver_status, gap, primal, dual, "optimal",
            )
        if status == highspy.HighsModelStatus.kInfeasible:
            return _QueryResult(
                "infeasible", None, None, None, None, solver_status, gap,
                primal, dual, "infeasible",
            )
        if status == highspy.HighsModelStatus.kTimeLimit:
            query_status, reason = "timeout", "solver_limit"
        else:
            query_status, reason = "unknown", f"solver_{status_name}"
        return _QueryResult(
            query_status, None, None, None, point, solver_status, gap,
            primal, dual, reason,
        )

    def support_bounds(self, rows: Sequence[int]) -> HZSupportBoundsResult:
        selected = tuple(int(row) for row in rows)
        if any(row < 0 or row >= self.hz.n_out for row in selected):
            raise IndexError("HZ support row is out of range")
        fast_lb = self._fast.lb.reshape(-1).double().numpy()
        fast_ub = self._fast.ub.reshape(-1).double().numpy()
        lower = np.asarray([fast_lb[row] for row in selected])
        upper = np.asarray([fast_ub[row] for row in selected])
        lower_status = ["fast_fallback"] * len(selected)
        upper_status = ["fast_fallback"] * len(selected)
        gaps: list[float | None] = [None] * len(selected)
        started = time.monotonic()
        start_solves = self._solves
        complete = self.available
        if not self.available:
            return HZSupportBoundsResult(
                selected,
                Bounds(
                    torch.from_numpy(lower).reshape(1, -1),
                    torch.from_numpy(upper).reshape(1, -1),
                ),
                tuple(lower_status),
                tuple(upper_status),
                tuple(gaps),
                time.monotonic() - started,
                0,
                False,
            )
        for position, row in enumerate(selected):
            objective = self.model.value_matrix.getrow(row)
            if objective.nnz == 0:
                lower[position] = upper[position] = self.model.value_center[row]
                lower_status[position] = upper_status[position] = "constant_exact"
                gaps[position] = 0.0
                continue
            minimum = self._run(objective)
            maximum = self._run(-objective)
            if minimum.status == "optimal" and minimum.certified_lower is not None:
                raw = self.model.value_center[row] + minimum.certified_lower
                lower[position] = np.nextafter(
                    raw - hz_outward_slack(
                        self.model.value_center[row], minimum.certified_lower
                    ),
                    -np.inf,
                )
                lower_status[position] = minimum.bound_kind or "highs_optimal"
            else:
                complete = False
                lower_status[position] = minimum.reason
            if maximum.status == "optimal" and maximum.certified_lower is not None:
                raw = self.model.value_center[row] - maximum.certified_lower
                upper[position] = np.nextafter(
                    raw + hz_outward_slack(
                        self.model.value_center[row], maximum.certified_lower
                    ),
                    np.inf,
                )
                upper_status[position] = maximum.bound_kind or "highs_optimal"
            else:
                complete = False
                upper_status[position] = maximum.reason
            available = [
                value for value in (minimum.gap, maximum.gap) if value is not None
            ]
            gaps[position] = max(available) if available else None
        return HZSupportBoundsResult(
            selected,
            Bounds(
                torch.from_numpy(lower).reshape(1, -1),
                torch.from_numpy(upper).reshape(1, -1),
            ),
            tuple(lower_status),
            tuple(upper_status),
            tuple(gaps),
            time.monotonic() - started,
            self._solves - start_solves,
            bool(complete and not self.relax_binaries and self.hz.exact),
        )

    def minimize_output(
        self,
        row: int,
        *,
        input_hz: SparseHZono | None,
        input_shape: tuple[int, ...] | None,
    ) -> HZMinimumResult:
        started = time.monotonic()
        row = int(row)
        if row < 0 or row >= self.hz.n_out:
            raise IndexError("HZ minimum output row is out of range")
        objective = self.model.value_matrix.getrow(row)
        result = self._run(objective)
        candidate = None
        candidate_objective = None
        if result.point is not None:
            candidate_objective = float(
                self.model.value_center[row]
                + float((objective @ result.point).item())
            )
            if input_hz is not None and input_shape is not None:
                candidate = HZSolver._recover_input(
                    self.model, result.point, input_hz, input_shape, 0
                )
        primal = (
            self.model.value_center[row] + result.primal
            if result.primal is not None
            else candidate_objective
        )
        dual = (
            self.model.value_center[row] + result.dual
            if result.dual is not None
            else None
        )
        if result.status == "optimal" and result.certified_lower is not None:
            raw = self.model.value_center[row] + result.certified_lower
            if result.bound_kind == "point_exact":
                minimum = float(raw)
            else:
                minimum = float(
                    np.nextafter(
                        raw - hz_outward_slack(
                            self.model.value_center[row], result.certified_lower
                        ),
                        -np.inf,
                    )
                )
        else:
            minimum = None
        return HZMinimumResult(
            result.status,
            minimum,
            candidate_objective,
            candidate,
            result.solver_status,
            result.gap,
            time.monotonic() - started,
            result.reason,
            result.certified_lower,
            result.bound_kind,
            primal,
            dual,
        )

    def evaluate_spec(
        self,
        out_spec: OutputSpec,
        *,
        batch_size: int,
        n_out: int,
        input_hz: SparseHZono | None = None,
        input_shape: tuple[int, ...] | None = None,
        tolerance: float = HZ_NUMERICAL_POLICY.feasibility_tolerance,
    ) -> list[VerifyResult]:
        """Mirror HZSolver property semantics with one reusable highspy model."""
        call_started = time.monotonic()
        batch = int(batch_size)
        if self.model.value_center.size != batch * int(n_out):
            return self._unknown_results(batch, "output_shape_mismatch")
        if not self.available:
            return self._unknown_results(
                batch, f"incremental_backend_unavailable:{self._build_error}"
            )
        encoded = out_spec.encode_linear(
            B=batch,
            n_out=int(n_out),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        C = encoded["C"].detach().cpu().double().numpy()
        thresholds = encoded["thresholds"].detach().cpu().double().numpy()
        properties = int(encoded["M"])
        unsafe_linear = encoded["kind"] == OutKind.UNSAFE_LINEAR
        base = self._run(np.zeros(self.model.n_var, dtype=np.float64))
        if base.status != "optimal":
            reason = "empty_hz" if base.status == "infeasible" else "base_unknown"
            return self._unknown_results(batch, reason)

        exact_witness = bool(
            self.hz.exact
            and not self.relax_binaries
            and input_hz is not None
            and self.hz.frame_id is not None
            and self.hz.frame_id == input_hz.frame_id
            and input_shape is not None
        )
        results: list[VerifyResult] = []
        for lane in range(batch):
            start, stop = lane * n_out, (lane + 1) * n_out
            C_lane = C[lane * properties : (lane + 1) * properties]
            threshold = thresholds[lane]
            value_matrix = self.model.value_matrix[start:stop]
            coefficients = (sp.csr_matrix(C_lane) @ value_matrix).tocsr()
            constants = C_lane @ self.model.value_center[start:stop]
            decision: VerifyResult | None = None
            if unsafe_linear:
                expanded = self._run(
                    np.zeros(self.model.n_var),
                    extra_A=coefficients,
                    extra_lb=np.full(properties, -np.inf),
                    extra_ub=threshold + tolerance - constants,
                )
                if expanded.status == "infeasible":
                    decision = VerifyResult(
                        VerifyStatus.CERTIFIED,
                        metadata=self._metadata(lane, "expanded_unsafe_infeasible"),
                    )
                elif expanded.status == "optimal" and exact_witness:
                    witness = self._validated_unsafe_witness(
                        expanded,
                        coefficients,
                        constants,
                        threshold,
                        tolerance,
                        contracted_upper=True,
                    )
                    if witness is not None:
                        counterexample = HZSolver._recover_input(
                            self.model, witness, input_hz, input_shape, lane
                        )
                        if counterexample is not None:
                            decision = VerifyResult(
                                VerifyStatus.FALSIFIED,
                                counterexample=counterexample,
                                metadata=self._metadata(lane, "exact_unsafe_witness"),
                            )
                if decision is None:
                    decision = VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata=self._metadata(lane, "unsafe_region_undecided"),
                    )
            else:
                undecided = False
                for prop in range(properties):
                    expanded = self._run(
                        np.zeros(self.model.n_var),
                        extra_A=coefficients.getrow(prop),
                        extra_lb=[threshold[prop] - tolerance - constants[prop]],
                        extra_ub=[np.inf],
                    )
                    if expanded.status == "infeasible":
                        continue
                    if not exact_witness:
                        undecided = True
                        break
                    witness = None
                    if expanded.status == "optimal" and expanded.point is not None:
                        value = constants[prop] + float(
                            (coefficients.getrow(prop) @ expanded.point).item()
                        )
                        if value >= threshold[prop] + tolerance:
                            witness = expanded.point
                        else:
                            contracted = self._run(
                                np.zeros(self.model.n_var),
                                extra_A=coefficients.getrow(prop),
                                extra_lb=[
                                    threshold[prop] + tolerance - constants[prop]
                                ],
                                extra_ub=[np.inf],
                            )
                            if contracted.status == "optimal":
                                witness = contracted.point
                    if witness is not None:
                        counterexample = HZSolver._recover_input(
                            self.model, witness, input_hz, input_shape, lane
                        )
                        if counterexample is not None:
                            decision = VerifyResult(
                                VerifyStatus.FALSIFIED,
                                counterexample=counterexample,
                                metadata=self._metadata(
                                    lane, "exact_violation_witness"
                                ),
                            )
                            break
                    undecided = True
                    if time.monotonic() >= self.deadline:
                        break
                if decision is None:
                    decision = VerifyResult(
                        VerifyStatus.UNKNOWN if undecided else VerifyStatus.CERTIFIED,
                        metadata=self._metadata(
                            lane,
                            "violation_region_undecided"
                            if undecided
                            else "expanded_violations_infeasible",
                        ),
                    )
            results.append(decision)
        telemetry = self.telemetry().as_dict()
        for result in results:
            result.metadata["incremental_hz"] = telemetry
            result.metadata["elapsed"] = time.monotonic() - call_started
            result.metadata["solves"] = telemetry["solves"]
            result.metadata["nodes"] = telemetry["mip_nodes"]
        return results

    def _validated_unsafe_witness(
        self,
        expanded: _QueryResult,
        coefficients: sp.csr_matrix,
        constants: np.ndarray,
        threshold: np.ndarray,
        tolerance: float,
        *,
        contracted_upper: bool,
    ) -> np.ndarray | None:
        if expanded.point is not None:
            values = constants + np.asarray(coefficients @ expanded.point).reshape(-1)
            if np.all(values <= threshold - tolerance):
                return expanded.point
        if not contracted_upper:
            return None
        contracted = self._run(
            np.zeros(self.model.n_var),
            extra_A=coefficients,
            extra_lb=np.full(threshold.size, -np.inf),
            extra_ub=threshold - tolerance - constants,
        )
        if contracted.status == "optimal" and contracted.point is not None:
            values = constants + np.asarray(coefficients @ contracted.point).reshape(-1)
            if np.all(values <= threshold - tolerance):
                return contracted.point
        return None

    def _metadata(self, lane: int, reason: str) -> dict[str, object]:
        return {
            "lane": int(lane),
            "source": "hybridz_highspy_incremental",
            "representation": "sparse",
            "reason": reason,
        }

    def _unknown_results(self, count: int, reason: str) -> list[VerifyResult]:
        telemetry = self.telemetry().as_dict()
        return [
            VerifyResult(
                VerifyStatus.UNKNOWN,
                metadata={
                    **self._metadata(lane, reason),
                    "incremental_hz": telemetry,
                },
            )
            for lane in range(int(count))
        ]
