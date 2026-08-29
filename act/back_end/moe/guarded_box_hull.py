# ===- act/back_end/moe/guarded_box_hull.py - Guarded box hull ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Coordinate box hulls of route-guarded hybrid zonotopes.

The HiGHS implementation builds one LP and changes only its objective between
coordinate support queries.  Binary HZ variables are deliberately relaxed to
``[0, 1]``: the resulting coordinate box remains a sound outer approximation,
while ``exact`` is reported only for an exact, continuous input HZ and a fully
optimal sweep.

Telemetry distinguishes model reuse and accepted basis submissions from the
stronger, solver-internal claim that a submitted basis was actually used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import (
    HZ_NUMERICAL_POLICY,
    HZono,
    SparseHZono,
    _lower_hz_milp,
    hz_outward_slack,
    sparse_hz_fast_bounds,
)

try:
    import highspy

    _HAS_HIGHSPY = True
except ImportError:  # pragma: no cover - exercised in dependency-light installs
    highspy = None
    _HAS_HIGHSPY = False


@dataclass(frozen=True)
class GuardedHullTelemetry:
    """Auditable counters for one coordinate-support sweep."""

    backend: str
    model_builds: int
    model_build_seconds: float
    objective_update_calls: int
    objective_coefficients_changed: int
    objective_update_seconds: float
    solves: int
    cold_start_solves: int
    basis_submission_attempts: int
    basis_submissions_accepted: int
    basis_valid_after_solve: int
    status_counts: tuple[tuple[str, int], ...]
    simplex_iterations: int
    ipm_iterations: int
    solve_seconds: float
    total_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model_builds": self.model_builds,
            "model_build_seconds": self.model_build_seconds,
            "objective_update_calls": self.objective_update_calls,
            "objective_coefficients_changed": self.objective_coefficients_changed,
            "objective_update_seconds": self.objective_update_seconds,
            "solves": self.solves,
            "cold_start_solves": self.cold_start_solves,
            "basis_submission_attempts": self.basis_submission_attempts,
            "basis_submissions_accepted": self.basis_submissions_accepted,
            "basis_seeded_reoptimization_solves": self.basis_submissions_accepted,
            "basis_valid_after_solve": self.basis_valid_after_solve,
            "status_counts": dict(self.status_counts),
            "simplex_iterations": self.simplex_iterations,
            "ipm_iterations": self.ipm_iterations,
            "solve_seconds": self.solve_seconds,
            "total_seconds": self.total_seconds,
            "basis_semantics": (
                "accepted submission only; does not assert solver-internal use"
            ),
            "warm_start_claimed": False,
        }


@dataclass(frozen=True)
class GuardedBoxHullResult:
    """A sound coordinate box enclosing the guarded domain."""

    bounds: Bounds
    lower_status: tuple[str, ...]
    upper_status: tuple[str, ...]
    domain_status: str
    complete: bool
    exact: bool
    relaxed_binaries: int
    telemetry: GuardedHullTelemetry


@dataclass(frozen=True)
class _LinearHullProblem:
    center: np.ndarray
    objectives: sp.csr_matrix
    A: sp.csr_matrix
    row_lb: np.ndarray
    row_ub: np.ndarray
    var_lb: np.ndarray
    var_ub: np.ndarray
    fallback_lb: np.ndarray
    fallback_ub: np.ndarray
    exact_continuous_domain: bool
    relaxed_binaries: int

    def __post_init__(self) -> None:
        n_out, n_var = self.objectives.shape
        if self.center.shape != (n_out,):
            raise ValueError("center and objective row count differ")
        if self.A.shape[1] != n_var:
            raise ValueError("constraint and objective column count differ")
        if self.row_lb.shape != (self.A.shape[0],):
            raise ValueError("constraint lower-bound shape mismatch")
        if self.row_ub.shape != (self.A.shape[0],):
            raise ValueError("constraint upper-bound shape mismatch")
        if self.var_lb.shape != (n_var,) or self.var_ub.shape != (n_var,):
            raise ValueError("variable-bound shape mismatch")
        if self.fallback_lb.shape != (n_out,) or self.fallback_ub.shape != (n_out,):
            raise ValueError("fallback-bound shape mismatch")


def _hz_problem(hz: HZono | SparseHZono) -> _LinearHullProblem:
    lowered = _lower_hz_milp(hz)
    if isinstance(hz, SparseHZono):
        fast = sparse_hz_fast_bounds(hz)
        is_exact = bool(hz.exact)
    else:
        center = hz.c.detach().cpu().double().numpy().reshape(-1)
        radius = (
            hz.Gc.detach().cpu().double().abs().sum(dim=1)
            + hz.Gb.detach().cpu().double().abs().sum(dim=1)
        ).numpy()
        fast = Bounds(
            torch.from_numpy(center - radius).reshape(1, -1),
            torch.from_numpy(center + radius).reshape(1, -1),
        )
        # Dense HZono carries no provenance bit.  Do not silently label a
        # potentially relaxed reachable set as exact merely because its final
        # support problem is continuous.
        is_exact = False
    return _LinearHullProblem(
        center=np.asarray(lowered.value_center, dtype=np.float64),
        objectives=lowered.value_matrix.tocsr(),
        A=lowered.A.tocsr(),
        row_lb=np.asarray(lowered.row_lb, dtype=np.float64),
        row_ub=np.asarray(lowered.row_ub, dtype=np.float64),
        var_lb=np.asarray(lowered.var_lb, dtype=np.float64),
        var_ub=np.asarray(lowered.var_ub, dtype=np.float64),
        fallback_lb=fast.lb.detach().cpu().double().numpy().reshape(-1),
        fallback_ub=fast.ub.detach().cpu().double().numpy().reshape(-1),
        exact_continuous_domain=bool(is_exact and lowered.n_bin == 0),
        relaxed_binaries=int(lowered.n_bin),
    )


def _outward_lower(center: float, optimum: float) -> float:
    value = float(center) + float(optimum)
    return float(
        np.nextafter(value - hz_outward_slack(center, optimum), -np.inf)
    )


def _outward_upper(center: float, negated_optimum: float) -> float:
    value = float(center) - float(negated_optimum)
    return float(
        np.nextafter(value + hz_outward_slack(center, negated_optimum), np.inf)
    )


def _empty_telemetry(backend: str, started: float) -> GuardedHullTelemetry:
    return GuardedHullTelemetry(
        backend=backend,
        model_builds=0,
        model_build_seconds=0.0,
        objective_update_calls=0,
        objective_coefficients_changed=0,
        objective_update_seconds=0.0,
        solves=0,
        cold_start_solves=0,
        basis_submission_attempts=0,
        basis_submissions_accepted=0,
        basis_valid_after_solve=0,
        status_counts=(),
        simplex_iterations=0,
        ipm_iterations=0,
        solve_seconds=0.0,
        total_seconds=time.monotonic() - started,
    )


def guarded_hz_box_hull_highs(
    hz: HZono | SparseHZono,
    *,
    time_limit: float = 300.0,
    submit_basis: bool = True,
) -> GuardedBoxHullResult:
    """Compute a guarded coordinate hull by reoptimizing one HiGHS model.

    Failed or incomplete coordinate objectives retain the unconstrained HZ
    generator bounds, so the returned box remains sound.  ``submit_basis``
    explicitly submits the last valid basis before a reoptimization; telemetry
    records acceptance but intentionally does not claim internal warm-start use.
    """

    if not _HAS_HIGHSPY:
        raise RuntimeError("highspy is required for the incremental hull backend")
    if not np.isfinite(float(time_limit)) or float(time_limit) < 0.0:
        raise ValueError("time_limit must be finite and non-negative")
    started = time.monotonic()
    problem = _hz_problem(hz)
    lower = problem.fallback_lb.copy()
    upper = problem.fallback_ub.copy()
    lower_status = ["fast_fallback"] * lower.size
    upper_status = ["fast_fallback"] * upper.size
    if lower.size == 0 or time_limit == 0.0:
        telemetry = _empty_telemetry("highspy_incremental_lp", started)
        return GuardedBoxHullResult(
            Bounds(torch.from_numpy(lower).reshape(1, -1), torch.from_numpy(upper).reshape(1, -1)),
            tuple(lower_status),
            tuple(upper_status),
            "unknown" if lower.size else "optimal",
            not lower.size,
            bool(not lower.size and problem.exact_continuous_domain),
            problem.relaxed_binaries,
            telemetry,
        )

    deadline = started + float(time_limit)
    build_started = time.monotonic()
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("solver", "simplex")
    solver.setOptionValue("presolve", "on")
    solver.setOptionValue("parallel", "off")
    solver.setOptionValue("random_seed", 0)
    solver.setOptionValue(
        "primal_feasibility_tolerance",
        HZ_NUMERICAL_POLICY.feasibility_tolerance,
    )
    solver.setOptionValue(
        "dual_feasibility_tolerance",
        HZ_NUMERICAL_POLICY.feasibility_tolerance,
    )
    status = solver.addVars(
        int(problem.var_lb.size), problem.var_lb, problem.var_ub
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS addVars failed: {status}")
    if problem.A.shape[0]:
        A = problem.A.tocsr()
        status = solver.addRows(
            int(A.shape[0]),
            problem.row_lb,
            problem.row_ub,
            int(A.nnz),
            np.asarray(A.indptr, dtype=np.int32),
            np.asarray(A.indices, dtype=np.int32),
            np.asarray(A.data, dtype=np.float64),
        )
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS addRows failed: {status}")
    model_build_seconds = time.monotonic() - build_started

    statuses: Counter[str] = Counter()
    objective_update_calls = 0
    objective_coefficients_changed = 0
    objective_update_seconds = 0.0
    solves = 0
    cold_start_solves = 0
    basis_attempts = 0
    basis_accepted = 0
    basis_valid_after = 0
    simplex_iterations = 0
    ipm_iterations = 0
    solve_seconds = 0.0
    last_basis = None
    previous_indices = np.zeros(0, dtype=np.int32)
    complete = True
    infeasible = False

    def optimize(indices: np.ndarray, values: np.ndarray):
        nonlocal objective_update_calls, objective_coefficients_changed
        nonlocal objective_update_seconds, solves, cold_start_solves
        nonlocal basis_attempts, basis_accepted, basis_valid_after
        nonlocal simplex_iterations, ipm_iterations, solve_seconds, last_basis
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            statuses["budget_exhausted"] += 1
            return None, "budget_exhausted"
        update_started = time.monotonic()
        changed_indices = np.union1d(previous_indices, indices).astype(np.int32)
        changed_values = np.zeros(changed_indices.size, dtype=np.float64)
        if indices.size:
            positions = np.searchsorted(changed_indices, indices)
            changed_values[positions] = values
        update_status = solver.changeColsCost(
            int(changed_indices.size), changed_indices, changed_values
        )
        objective_update_calls += 1
        objective_coefficients_changed += int(changed_indices.size)
        objective_update_seconds += time.monotonic() - update_started
        if update_status != highspy.HighsStatus.kOk:
            statuses["objective_update_error"] += 1
            return None, "objective_update_error"

        accepted_basis = False
        if submit_basis and last_basis is not None:
            basis_attempts += 1
            accepted_basis = solver.setBasis(last_basis) == highspy.HighsStatus.kOk
            basis_accepted += int(accepted_basis)
        if not accepted_basis:
            cold_start_solves += 1
        solver.setOptionValue("time_limit", max(1e-3, remaining))
        solve_started = time.monotonic()
        run_status = solver.run()
        solve_seconds += time.monotonic() - solve_started
        solves += 1
        model_status = solver.getModelStatus()
        status_name = solver.modelStatusToString(model_status).lower().replace(" ", "_")
        if run_status != highspy.HighsStatus.kOk:
            status_name = f"run_error:{status_name}"
        statuses[status_name] += 1
        info = solver.getInfo()
        simplex_iterations += max(0, int(info.simplex_iteration_count))
        ipm_iterations += max(0, int(info.ipm_iteration_count))
        basis = solver.getBasis()
        if bool(getattr(basis, "valid", False)):
            basis_valid_after += 1
            last_basis = basis
        else:
            last_basis = None
        if (
            run_status == highspy.HighsStatus.kOk
            and bool(getattr(info, "valid", False))
            and model_status == highspy.HighsModelStatus.kOptimal
        ):
            value = float(solver.getObjectiveValue())
            if np.isfinite(value):
                return value, "lp_optimal"
            return None, "nonfinite_objective"
        if model_status == highspy.HighsModelStatus.kInfeasible:
            return None, "infeasible"
        return None, status_name

    for row in range(lower.size):
        objective = problem.objectives.getrow(row)
        indices = np.asarray(objective.indices, dtype=np.int32)
        values = np.asarray(objective.data, dtype=np.float64)
        if not indices.size:
            lower[row] = upper[row] = problem.center[row]
            lower_status[row] = upper_status[row] = "constant_exact"
            continue
        minimum, min_status = optimize(indices, values)
        previous_indices = indices
        if min_status == "infeasible":
            infeasible = True
            complete = False
            break
        maximum, max_status = optimize(indices, -values)
        previous_indices = indices
        if max_status == "infeasible":
            infeasible = True
            complete = False
            break
        if minimum is not None:
            lower[row] = _outward_lower(problem.center[row], minimum)
            lower_status[row] = min_status
        else:
            complete = False
            lower_status[row] = min_status
        if maximum is not None:
            upper[row] = _outward_upper(problem.center[row], maximum)
            upper_status[row] = max_status
        else:
            complete = False
            upper_status[row] = max_status
        if time.monotonic() >= deadline:
            complete = False
            break

    telemetry = GuardedHullTelemetry(
        backend="highspy_incremental_lp",
        model_builds=1,
        model_build_seconds=model_build_seconds,
        objective_update_calls=objective_update_calls,
        objective_coefficients_changed=objective_coefficients_changed,
        objective_update_seconds=objective_update_seconds,
        solves=solves,
        cold_start_solves=cold_start_solves,
        basis_submission_attempts=basis_attempts,
        basis_submissions_accepted=basis_accepted,
        basis_valid_after_solve=basis_valid_after,
        status_counts=tuple(sorted(statuses.items())),
        simplex_iterations=simplex_iterations,
        ipm_iterations=ipm_iterations,
        solve_seconds=solve_seconds,
        total_seconds=time.monotonic() - started,
    )
    return GuardedBoxHullResult(
        bounds=Bounds(
            torch.from_numpy(lower).reshape(1, -1),
            torch.from_numpy(upper).reshape(1, -1),
        ),
        lower_status=tuple(lower_status),
        upper_status=tuple(upper_status),
        domain_status=("infeasible" if infeasible else "optimal" if complete else "partial"),
        complete=complete,
        exact=bool(complete and problem.exact_continuous_domain),
        relaxed_binaries=problem.relaxed_binaries,
        telemetry=telemetry,
    )


def guarded_hz_box_hull_scipy(
    hz: HZono | SparseHZono,
    *,
    time_limit: float = 300.0,
) -> GuardedBoxHullResult:
    """Reference SciPy/HiGHS sweep used for differential validation."""

    if not np.isfinite(float(time_limit)) or float(time_limit) < 0.0:
        raise ValueError("time_limit must be finite and non-negative")
    started = time.monotonic()
    problem = _hz_problem(hz)
    lower = problem.fallback_lb.copy()
    upper = problem.fallback_ub.copy()
    lower_status = ["fast_fallback"] * lower.size
    upper_status = ["fast_fallback"] * upper.size
    deadline = started + float(time_limit)
    equality_rows: list[int] = []
    upper_parts: list[sp.csr_matrix] = []
    upper_rhs: list[np.ndarray] = []
    for row in range(problem.A.shape[0]):
        row_lb = problem.row_lb[row]
        row_ub = problem.row_ub[row]
        if np.isfinite(row_lb) and np.isfinite(row_ub) and row_lb == row_ub:
            equality_rows.append(row)
            continue
        if np.isfinite(row_ub):
            upper_parts.append(problem.A.getrow(row))
            upper_rhs.append(np.asarray([row_ub], dtype=np.float64))
        if np.isfinite(row_lb):
            upper_parts.append(-problem.A.getrow(row))
            upper_rhs.append(np.asarray([-row_lb], dtype=np.float64))
    A_eq = problem.A[equality_rows] if equality_rows else None
    b_eq = problem.row_ub[equality_rows] if equality_rows else None
    A_ub = sp.vstack(upper_parts, format="csr") if upper_parts else None
    b_ub = np.concatenate(upper_rhs) if upper_rhs else None
    bounds = list(zip(problem.var_lb, problem.var_ub))
    statuses: Counter[str] = Counter()
    solves = 0
    solve_seconds = 0.0
    complete = True
    infeasible = False

    def optimize(objective: np.ndarray):
        nonlocal solves, solve_seconds
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            statuses["budget_exhausted"] += 1
            return None, "budget_exhausted", 0
        solve_started = time.monotonic()
        result = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"time_limit": max(1e-3, remaining), "presolve": True},
        )
        solve_seconds += time.monotonic() - solve_started
        solves += 1
        status_name = "lp_optimal" if result.status == 0 else f"scipy_status_{result.status}"
        statuses[status_name] += 1
        if result.status == 0 and result.fun is not None and np.isfinite(result.fun):
            return float(result.fun), status_name, int(getattr(result, "nit", 0))
        if result.status == 2:
            return None, "infeasible", int(getattr(result, "nit", 0))
        return None, status_name, int(getattr(result, "nit", 0))

    iterations = 0
    for row in range(lower.size):
        objective = problem.objectives.getrow(row).toarray().reshape(-1)
        if not np.any(objective):
            lower[row] = upper[row] = problem.center[row]
            lower_status[row] = upper_status[row] = "constant_exact"
            continue
        minimum, min_status, nit = optimize(objective)
        iterations += nit
        if min_status == "infeasible":
            infeasible = True
            complete = False
            break
        maximum, max_status, nit = optimize(-objective)
        iterations += nit
        if max_status == "infeasible":
            infeasible = True
            complete = False
            break
        if minimum is not None:
            lower[row] = _outward_lower(problem.center[row], minimum)
            lower_status[row] = min_status
        else:
            complete = False
            lower_status[row] = min_status
        if maximum is not None:
            upper[row] = _outward_upper(problem.center[row], maximum)
            upper_status[row] = max_status
        else:
            complete = False
            upper_status[row] = max_status
    total_seconds = time.monotonic() - started
    telemetry = GuardedHullTelemetry(
        backend="scipy_linprog_reference",
        model_builds=solves,
        model_build_seconds=0.0,
        objective_update_calls=solves,
        objective_coefficients_changed=0,
        objective_update_seconds=0.0,
        solves=solves,
        cold_start_solves=solves,
        basis_submission_attempts=0,
        basis_submissions_accepted=0,
        basis_valid_after_solve=0,
        status_counts=tuple(sorted(statuses.items())),
        simplex_iterations=iterations,
        ipm_iterations=0,
        solve_seconds=solve_seconds,
        total_seconds=total_seconds,
    )
    return GuardedBoxHullResult(
        bounds=Bounds(
            torch.from_numpy(lower).reshape(1, -1),
            torch.from_numpy(upper).reshape(1, -1),
        ),
        lower_status=tuple(lower_status),
        upper_status=tuple(upper_status),
        domain_status=("infeasible" if infeasible else "optimal" if complete else "partial"),
        complete=complete,
        exact=bool(complete and problem.exact_continuous_domain),
        relaxed_binaries=problem.relaxed_binaries,
        telemetry=telemetry,
    )
