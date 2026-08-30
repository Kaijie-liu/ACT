#!/usr/bin/env python3
"""Scratch-only, candidate-only HiGHS row-stream transaction contract.

This module deliberately has no production imports and grants no verifier
authority.  It freezes the minimum API needed by the phase-projection probes:

* one base and at most one updated MIN(-c) solve on the same native owner;
* an optimal primal/row-value/row-dual result, or one base-only dual ray;
* exact streamed-row identity mapping and accounting;
* fail-closed poisoning and BaseException-safe cleanup.

Run this file directly for the CPU-only contract sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import time
from typing import Any, Callable, Literal, Sequence

import highspy
import numpy as np


HIGHS_VERSION = "1.15.0"
HIGHS_GITHASH = "8396001"
SMALL_MATRIX_VALUE = 1.0e-12
LARGE_MATRIX_VALUE = 1.0e15
INFINITE_BOUND = 1.0e20
PRIMAL_TOLERANCE = 1.0e-9
DUAL_TOLERANCE = 1.0e-9


class OwnerContractError(RuntimeError):
    pass


class DeadlineExpired(OwnerContractError):
    pass


class InfeasibleRayDisabled(OwnerContractError):
    pass


class CleanupFailure(OwnerContractError):
    pass


class InjectedPrimary(BaseException):
    pass


class InjectedCleanup(BaseException):
    def __str__(self) -> str:
        raise RuntimeError("hostile cleanup __str__ must never be called")

    def __repr__(self) -> str:
        raise RuntimeError("hostile cleanup __repr__ must never be called")


def _sha_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _copy_exact_vector(value: Any, dtype: np.dtype[Any], name: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.ndim != 1
        or not value.flags.c_contiguous
    ):
        raise OwnerContractError(f"{name} must be a contiguous {dtype} vector")
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _validate_cost_bounds(
    cost: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cost = _copy_exact_vector(cost, np.dtype(np.float64), "cost")
    lower = _copy_exact_vector(lower, np.dtype(np.float64), "column lower")
    upper = _copy_exact_vector(upper, np.dtype(np.float64), "column upper")
    if cost.size == 0 or lower.size != cost.size or upper.size != cost.size:
        raise OwnerContractError("column vectors have inconsistent sizes")
    if not np.all(np.isfinite(cost)) or np.any(np.abs(cost) >= INFINITE_BOUND):
        raise OwnerContractError("objective is outside the finite cost contract")
    if np.any((np.abs(cost) > 0.0) & (np.abs(cost) <= SMALL_MATRIX_VALUE)):
        # Matrix threshold options do not define a sufficiently strong
        # optimization guarantee for tiny costs.  Never delete or perturb the
        # objective: this request is simply ineligible and must become UNKNOWN.
        raise OwnerContractError("objective contains an unsupported tiny coefficient")
    # The phase-projection factor frame is finite.  Keeping this restriction
    # also makes exact row-plus-bound Farkas replay unambiguous.
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
        or np.any(np.abs(lower) >= INFINITE_BOUND)
        or np.any(np.abs(upper) >= INFINITE_BOUND)
    ):
        raise OwnerContractError("column bounds are malformed or non-finite")
    return cost, lower, upper


@dataclass(frozen=True)
class FrozenRows:
    rows: int
    cols: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    full_row_ids: np.ndarray
    content_sha256: str


def freeze_rows(
    *,
    rows: int,
    cols: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    full_row_ids: np.ndarray,
    allow_tiny: bool = False,
) -> FrozenRows:
    if type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0:
        raise OwnerContractError("row batch dimensions are invalid")
    indptr = _copy_exact_vector(indptr, np.dtype(np.int32), "indptr")
    indices = _copy_exact_vector(indices, np.dtype(np.int32), "indices")
    data = _copy_exact_vector(data, np.dtype(np.float64), "data")
    lower = _copy_exact_vector(lower, np.dtype(np.float64), "row lower")
    upper = _copy_exact_vector(upper, np.dtype(np.float64), "row upper")
    full_row_ids = _copy_exact_vector(
        full_row_ids, np.dtype(np.int64), "full row ids"
    )
    nnz = int(data.size)
    if (
        indptr.size != rows + 1
        or int(indptr[0]) != 0
        or int(indptr[-1]) != nnz
        or np.any(indptr[1:] < indptr[:-1])
        or indices.size != nnz
        or lower.size != rows
        or upper.size != rows
        or full_row_ids.size != rows
    ):
        raise OwnerContractError("CSR shape or terminal pointer is malformed")
    if np.unique(full_row_ids).size != rows:
        raise OwnerContractError("full row ids must be unique inside a batch")
    if np.any(indices < 0) or np.any(indices >= cols):
        raise OwnerContractError("CSR column index is outside the frame")
    for row in range(rows):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if stop - start > 1 and np.any(indices[start + 1 : stop] <= indices[start:stop - 1]):
            raise OwnerContractError("CSR rows must have strictly increasing indices")
    if not np.all(np.isfinite(data)):
        raise OwnerContractError("CSR contains NaN or infinity")
    magnitudes = np.abs(data)
    if np.any(magnitudes == 0.0) or np.any(magnitudes >= LARGE_MATRIX_VALUE):
        raise OwnerContractError("CSR contains explicit zero or oversized coefficient")
    if not allow_tiny and np.any(magnitudes <= SMALL_MATRIX_VALUE):
        raise OwnerContractError("CSR contains an unprocessed tiny coefficient")
    if (
        np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(np.isposinf(lower))
        or np.any(np.isneginf(upper))
        or np.any(lower > upper)
    ):
        raise OwnerContractError("row bounds are malformed")
    finite_lower = lower[np.isfinite(lower)]
    finite_upper = upper[np.isfinite(upper)]
    if (
        np.any(np.abs(finite_lower) >= INFINITE_BOUND)
        or np.any(np.abs(finite_upper) >= INFINITE_BOUND)
    ):
        raise OwnerContractError("finite row bound crosses infinite_bound")
    content_sha256 = _sha_arrays(
        indptr, indices, data, lower, upper, full_row_ids
    )
    return FrozenRows(
        rows=rows,
        cols=cols,
        indptr=indptr,
        indices=indices,
        data=data,
        lower=lower,
        upper=upper,
        full_row_ids=full_row_ids,
        content_sha256=content_sha256,
    )


def _assert_frozen(rows: FrozenRows) -> None:
    arrays = (
        rows.indptr,
        rows.indices,
        rows.data,
        rows.lower,
        rows.upper,
        rows.full_row_ids,
    )
    if any(array.flags.writeable for array in arrays):
        raise OwnerContractError("sealed row batch became writeable")
    if _sha_arrays(*arrays) != rows.content_sha256:
        raise OwnerContractError("sealed row batch changed before native load")


def _round_fraction_up(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise OwnerContractError("outward upper rounding overflowed")
    if Fraction.from_float(result) < value:
        result = float(np.nextafter(result, np.inf))
    if Fraction.from_float(result) < value:
        raise OwnerContractError("outward upper rounding failed")
    return result


def _round_fraction_down(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise OwnerContractError("outward lower rounding overflowed")
    if Fraction.from_float(result) > value:
        result = float(np.nextafter(result, -np.inf))
    if Fraction.from_float(result) > value:
        raise OwnerContractError("outward lower rounding failed")
    return result


@dataclass(frozen=True)
class TinyFilterResult:
    rows: FrozenRows
    deleted_nnz: int
    upper_exact: tuple[str | None, ...]
    lower_exact: tuple[str | None, ...]


def filter_tiny_outward(
    source: FrozenRows,
    column_lower: np.ndarray,
    column_upper: np.ndarray,
) -> TinyFilterResult:
    """Delete <=1e-12 matrix terms and outward-project their bound range.

    For L <= kept + deleted <= U, projection onto kept uses
      L' = L - sum(max(a*l, a*u))
      U' = U - sum(min(a*l, a*u)).
    The requested upper-only formula is therefore U - sum(min(...)).
    """

    _assert_frozen(source)
    column_lower = _copy_exact_vector(
        column_lower, np.dtype(np.float64), "tiny-filter column lower"
    )
    column_upper = _copy_exact_vector(
        column_upper, np.dtype(np.float64), "tiny-filter column upper"
    )
    if (
        column_lower.size != source.cols
        or column_upper.size != source.cols
        or not np.all(np.isfinite(column_lower))
        or not np.all(np.isfinite(column_upper))
        or np.any(column_lower > column_upper)
    ):
        raise OwnerContractError("tiny-filter column bounds are malformed")
    new_indptr = [0]
    new_indices: list[int] = []
    new_data: list[float] = []
    new_lower = np.array(source.lower, dtype=np.float64, copy=True)
    new_upper = np.array(source.upper, dtype=np.float64, copy=True)
    lower_exact: list[str | None] = []
    upper_exact: list[str | None] = []
    deleted = 0
    for row in range(source.rows):
        deleted_min = Fraction(0)
        deleted_max = Fraction(0)
        for position in range(int(source.indptr[row]), int(source.indptr[row + 1])):
            coefficient = float(source.data[position])
            column = int(source.indices[position])
            if abs(coefficient) <= SMALL_MATRIX_VALUE:
                a = Fraction.from_float(coefficient)
                lo = a * Fraction.from_float(float(column_lower[column]))
                hi = a * Fraction.from_float(float(column_upper[column]))
                deleted_min += min(lo, hi)
                deleted_max += max(lo, hi)
                deleted += 1
            else:
                new_indices.append(column)
                new_data.append(coefficient)
        if math.isfinite(float(source.lower[row])):
            exact_lower = Fraction.from_float(float(source.lower[row])) - deleted_max
            new_lower[row] = _round_fraction_down(exact_lower)
            lower_exact.append(str(exact_lower))
        else:
            lower_exact.append(None)
        if math.isfinite(float(source.upper[row])):
            exact_upper = Fraction.from_float(float(source.upper[row])) - deleted_min
            new_upper[row] = _round_fraction_up(exact_upper)
            upper_exact.append(str(exact_upper))
        else:
            upper_exact.append(None)
        new_indptr.append(len(new_data))
    filtered = freeze_rows(
        rows=source.rows,
        cols=source.cols,
        indptr=np.asarray(new_indptr, dtype=np.int32),
        indices=np.asarray(new_indices, dtype=np.int32),
        data=np.asarray(new_data, dtype=np.float64),
        lower=np.asarray(new_lower, dtype=np.float64),
        upper=np.asarray(new_upper, dtype=np.float64),
        full_row_ids=np.asarray(source.full_row_ids, dtype=np.int64),
        allow_tiny=False,
    )
    return TinyFilterResult(
        rows=filtered,
        deleted_nnz=deleted,
        upper_exact=tuple(upper_exact),
        lower_exact=tuple(lower_exact),
    )


@dataclass(frozen=True)
class OptimalSolve:
    status: Literal["OPTIMAL"]
    primal: np.ndarray
    row_value: np.ndarray
    row_dual: np.ndarray
    upper_residual: np.ndarray
    full_row_ids: np.ndarray
    minimized_objective: float


@dataclass(frozen=True)
class InfeasibleRaySolve:
    status: Literal["INFEASIBLE_DUAL_RAY"]
    row_ray: np.ndarray
    full_row_ids: np.ndarray
    support_full_row_ids: tuple[int, ...]
    toy_algebraic_oracle_ran: bool
    exact_row_rhs: str | None
    exact_bound_support: str | None
    exact_contradiction_gap: str | None


SolveResult = OptimalSolve | InfeasibleRaySolve


def _readonly_f64(value: Sequence[float]) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True).reshape(-1)
    result.setflags(write=False)
    return result


def _readonly_i64(value: Sequence[int]) -> np.ndarray:
    result = np.array(value, dtype=np.int64, order="C", copy=True).reshape(-1)
    result.setflags(write=False)
    return result


class SafeHighsOwner:
    """One-request HiGHS owner with base then optional updated transaction.

    Exactly one native ``highspy.Highs`` object is constructed.  A base model
    is loaded and solved, ``clearModel`` starts the optional updated model, and
    final request cleanup uses ``clear``.  Only the base model may request one
    dual ray; an infeasible updated model fails closed without any ray call.
    """

    _PINNED_OPTIONS: tuple[tuple[str, Any], ...] = (
        ("output_flag", False),
        ("solver", "simplex"),
        ("presolve", "off"),
        ("simplex_strategy", 1),
        ("simplex_scale_strategy", 2),
        ("threads", 1),
        ("parallel", "off"),
        ("random_seed", 0),
        ("small_matrix_value", SMALL_MATRIX_VALUE),
        ("large_matrix_value", LARGE_MATRIX_VALUE),
        ("infinite_bound", INFINITE_BOUND),
        ("primal_feasibility_tolerance", PRIMAL_TOLERANCE),
        ("dual_feasibility_tolerance", DUAL_TOLERANCE),
    )

    def __init__(
        self,
        *,
        negative_objective: np.ndarray,
        column_lower: np.ndarray,
        column_upper: np.ndarray,
        deadline_monotonic: float,
        highs_factory: Callable[[], Any] = highspy.Highs,
        clock: Callable[[], float] = time.monotonic,
        toy_algebraic_oracle: bool = False,
    ) -> None:
        if type(toy_algebraic_oracle) is not bool:
            raise OwnerContractError("toy algebraic oracle flag must be bool")
        self._cost, self._column_lower, self._column_upper = _validate_cost_bounds(
            negative_objective, column_lower, column_upper
        )
        if not isinstance(deadline_monotonic, float) or not math.isfinite(
            deadline_monotonic
        ):
            raise OwnerContractError("deadline must be a finite monotonic float")
        self._deadline = deadline_monotonic
        self._clock = clock
        self._toy_algebraic_oracle = toy_algebraic_oracle
        self._state = "NEW"
        self._poison_reason: str | None = None
        self._batches: list[FrozenRows] = []
        self._current_load_tag: Literal["base", "updated"] = "base"
        self._dual_ray_used = False
        self._updated_started = False
        self._column_roundtrip_bitwise = False
        self._h: Any | None = None
        self._h = highs_factory()
        try:
            self._configure_and_load_columns()
            self._state = "LOADING_BASE"
        except BaseException as primary:
            self._state = "POISONED"
            self._poison_reason = f"constructor:{type(primary).__name__}"
            try:
                self._clear_backend()
            except BaseException as cleanup:
                try:
                    primary.add_note(
                        "secondary HiGHS cleanup failure was suppressed to preserve primary"
                    )
                except BaseException:
                    pass
            raise

    @property
    def state(self) -> str:
        return self._state

    @property
    def poison_reason(self) -> str | None:
        return self._poison_reason

    def __enter__(self) -> "SafeHighsOwner":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        cleanup: BaseException | None = None
        try:
            self._clear_backend()
        except BaseException as error:
            cleanup = error
        if exc is not None:
            if cleanup is not None:
                try:
                    exc.add_note(
                        "secondary HiGHS cleanup failure was suppressed to preserve primary"
                    )
                except BaseException:
                    pass
            return False
        if cleanup is not None:
            raise cleanup
        return False

    def _poison(self, operation: str, error: BaseException) -> None:
        if self._state != "CLOSED":
            self._state = "POISONED"
            self._poison_reason = f"{operation}:{type(error).__name__}"

    def _require_loading(self) -> None:
        if self._state not in ("LOADING_BASE", "LOADING_UPDATED"):
            raise OwnerContractError(f"owner is not loadable: {self._state}")

    def _remaining(self) -> float:
        remaining = float(self._deadline - self._clock())
        if not math.isfinite(remaining) or remaining <= 0.0:
            raise DeadlineExpired("owner deadline expired")
        return remaining

    @staticmethod
    def _require_ok(status: Any, operation: str) -> None:
        if status != highspy.HighsStatus.kOk:
            # Never stringify or repr a backend-controlled hostile object while
            # already handling a primary solver failure.
            raise OwnerContractError(f"HiGHS {operation} returned non-kOk")

    def _configure_and_load_columns(self) -> None:
        assert self._h is not None
        if self._h.version() != HIGHS_VERSION or self._h.githash() != HIGHS_GITHASH:
            raise OwnerContractError("HiGHS version or git hash differs from the pin")
        self._remaining()
        for name, value in self._PINNED_OPTIONS:
            self._require_ok(self._h.setOptionValue(name, value), f"set {name}")
            status, observed = self._h.getOptionValue(name)
            self._require_ok(status, f"get {name}")
            if observed != value:
                raise OwnerContractError(f"HiGHS option {name} did not round-trip")
        self._require_ok(
            self._h.changeObjectiveSense(highspy.ObjSense.kMinimize),
            "set objective sense",
        )
        sense_status, sense = self._h.getObjectiveSense()
        self._require_ok(sense_status, "get objective sense")
        if sense != highspy.ObjSense.kMinimize:
            raise OwnerContractError("objective sense is not MINIMIZE")
        columns = int(self._cost.size)
        status = self._h.addCols(
            columns,
            self._cost,
            self._column_lower,
            self._column_upper,
            0,
            np.zeros(columns + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
        self._require_ok(status, "addCols")
        if (
            self._h.getNumCol() != columns
            or self._h.getNumRow() != 0
            or self._h.getNumNz() != 0
        ):
            raise OwnerContractError("empty column-frame postcondition failed")
        lp = self._h.getLp()
        observed_cost = np.asarray(lp.col_cost_, dtype=np.float64)
        observed_lower = np.asarray(lp.col_lower_, dtype=np.float64)
        observed_upper = np.asarray(lp.col_upper_, dtype=np.float64)
        if not (
            np.array_equal(observed_cost, self._cost)
            and np.array_equal(observed_lower, self._column_lower)
            and np.array_equal(observed_upper, self._column_upper)
        ):
            raise OwnerContractError(
                "HiGHS changed objective coefficients or column bounds"
            )
        self._column_roundtrip_bitwise = True

    def reload_updated(
        self,
        *,
        negative_objective: np.ndarray,
        column_lower: np.ndarray,
        column_upper: np.ndarray,
    ) -> None:
        """Clear only the base model and load the one allowed updated frame."""

        try:
            if self._state not in (
                "SOLVED_BASE_OPTIMAL",
                "SOLVED_BASE_INFEASIBLE_RAY",
            ):
                raise OwnerContractError("updated reload requires a solved base model")
            if self._updated_started:
                raise OwnerContractError("updated model was already started")
            self._remaining()
            new_cost, new_lower, new_upper = _validate_cost_bounds(
                negative_objective, column_lower, column_upper
            )
            assert self._h is not None
            self._require_ok(self._h.clearModel(), "clearModel")
            if (
                self._h.getNumCol() != 0
                or self._h.getNumRow() != 0
                or self._h.getNumNz() != 0
            ):
                raise OwnerContractError("clearModel postcondition failed")
            # clearModel must retain every pinned request option.  Re-read, do
            # not silently reconfigure a drifting owner.
            for name, expected in self._PINNED_OPTIONS:
                status, observed = self._h.getOptionValue(name)
                self._require_ok(status, f"updated get {name}")
                if observed != expected:
                    raise OwnerContractError("pinned option changed after clearModel")
            sense_status, sense = self._h.getObjectiveSense()
            self._require_ok(sense_status, "updated get objective sense")
            if sense != highspy.ObjSense.kMinimize:
                raise OwnerContractError("objective sense changed after clearModel")
            self._cost, self._column_lower, self._column_upper = (
                new_cost,
                new_lower,
                new_upper,
            )
            self._batches.clear()
            self._current_load_tag = "updated"
            self._updated_started = True
            self._column_roundtrip_bitwise = False
            self._configure_columns_only()
            self._state = "LOADING_UPDATED"
        except BaseException as error:
            self._poison("reload_updated", error)
            raise

    def _configure_columns_only(self) -> None:
        """Load the current frozen column frame without constructing a backend."""

        assert self._h is not None
        columns = int(self._cost.size)
        self._require_ok(
            self._h.addCols(
                columns,
                self._cost,
                self._column_lower,
                self._column_upper,
                0,
                np.zeros(columns + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            ),
            "updated addCols",
        )
        if (
            self._h.getNumCol() != columns
            or self._h.getNumRow() != 0
            or self._h.getNumNz() != 0
        ):
            raise OwnerContractError("updated empty column-frame postcondition failed")
        lp = self._h.getLp()
        if not (
            np.array_equal(np.asarray(lp.col_cost_, dtype=np.float64), self._cost)
            and np.array_equal(
                np.asarray(lp.col_lower_, dtype=np.float64), self._column_lower
            )
            and np.array_equal(
                np.asarray(lp.col_upper_, dtype=np.float64), self._column_upper
            )
        ):
            raise OwnerContractError("updated column frame changed during native load")
        self._column_roundtrip_bitwise = True

    def append_rows(
        self,
        *,
        rows: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        full_row_ids: np.ndarray,
    ) -> None:
        try:
            self._require_loading()
            self._remaining()
            frozen = freeze_rows(
                rows=rows,
                cols=int(self._cost.size),
                indptr=indptr,
                indices=indices,
                data=data,
                lower=lower,
                upper=upper,
                full_row_ids=full_row_ids,
                allow_tiny=False,
            )
            _assert_frozen(frozen)
            prior_rows = sum(batch.rows for batch in self._batches)
            prior_nnz = sum(int(batch.data.size) for batch in self._batches)
            existing_ids = {
                int(row_id) for batch in self._batches for row_id in batch.full_row_ids
            }
            if existing_ids.intersection(int(row_id) for row_id in frozen.full_row_ids):
                raise OwnerContractError("full row ids repeat across streamed batches")
            assert self._h is not None
            status = self._h.addRows(
                frozen.rows,
                frozen.lower,
                frozen.upper,
                int(frozen.data.size),
                frozen.indptr,
                frozen.indices,
                frozen.data,
            )
            self._require_ok(status, "addRows")
            if (
                self._h.getNumCol() != int(self._cost.size)
                or self._h.getNumRow() != prior_rows + frozen.rows
                or self._h.getNumNz() != prior_nnz + int(frozen.data.size)
            ):
                raise OwnerContractError("per-load row/column/nnz postcondition failed")
            _assert_frozen(frozen)
            self._batches.append(frozen)
        except BaseException as error:
            self._poison("append_rows", error)
            raise

    def _row_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = _readonly_f64(
            [float(v) for batch in self._batches for v in batch.lower]
        )
        upper = _readonly_f64(
            [float(v) for batch in self._batches for v in batch.upper]
        )
        row_ids = _readonly_i64(
            [int(v) for batch in self._batches for v in batch.full_row_ids]
        )
        return lower, upper, row_ids

    def _exact_dual_ray_certificate(
        self, ray: np.ndarray
    ) -> tuple[Fraction, Fraction, Fraction]:
        lower, upper, _row_ids = self._row_frame()
        if ray.size != lower.size:
            raise OwnerContractError("dual ray length differs from streamed rows")
        q = [Fraction(0) for _ in range(self._cost.size)]
        row_rhs = Fraction(0)
        row_offset = 0
        for batch in self._batches:
            _assert_frozen(batch)
            for local_row in range(batch.rows):
                y = Fraction.from_float(float(ray[row_offset + local_row]))
                if y > 0:
                    if not math.isfinite(float(batch.lower[local_row])):
                        raise OwnerContractError("positive ray lacks a finite lower row")
                    row_rhs += y * Fraction.from_float(float(batch.lower[local_row]))
                elif y < 0:
                    if not math.isfinite(float(batch.upper[local_row])):
                        raise OwnerContractError("negative ray lacks a finite upper row")
                    row_rhs += y * Fraction.from_float(float(batch.upper[local_row]))
                start, stop = int(batch.indptr[local_row]), int(
                    batch.indptr[local_row + 1]
                )
                for position in range(start, stop):
                    q[int(batch.indices[position])] += y * Fraction.from_float(
                        float(batch.data[position])
                    )
            row_offset += batch.rows
        bound_support = Fraction(0)
        for column, coefficient in enumerate(q):
            lo = coefficient * Fraction.from_float(float(self._column_lower[column]))
            hi = coefficient * Fraction.from_float(float(self._column_upper[column]))
            bound_support += max(lo, hi)
        gap = row_rhs - bound_support
        if gap <= 0:
            raise OwnerContractError("dual ray has no exact row-plus-bound contradiction")
        return row_rhs, bound_support, gap

    def solve(self) -> SolveResult:
        try:
            self._require_loading()
            if not self._batches:
                raise OwnerContractError("cannot solve an empty row frame")
            if not self._column_roundtrip_bitwise:
                raise OwnerContractError("column frame was not bitwise read back")
            remaining = self._remaining()
            assert self._h is not None
            self._require_ok(
                self._h.setOptionValue("time_limit", remaining), "set time_limit"
            )
            option_status, observed_limit = self._h.getOptionValue("time_limit")
            self._require_ok(option_status, "get time_limit")
            if observed_limit != remaining:
                raise OwnerContractError("time_limit did not round-trip")
            run_status = self._h.run()
            self._require_ok(run_status, "run")
            # A solver result that arrives after the request deadline is stale.
            # Do not even read a primal, dual, or ray from it.
            self._remaining()
            model_status = self._h.getModelStatus()
            lower, upper, row_ids = self._row_frame()
            if model_status == highspy.HighsModelStatus.kOptimal:
                self._remaining()
                solution = self._h.getSolution()
                self._remaining()
                if not solution.value_valid or not solution.dual_valid:
                    raise OwnerContractError("optimal solution flags are invalid")
                primal = _readonly_f64(solution.col_value)
                row_value = _readonly_f64(solution.row_value)
                row_dual = _readonly_f64(solution.row_dual)
                if (
                    primal.size != self._cost.size
                    or row_value.size != row_ids.size
                    or row_dual.size != row_ids.size
                    or not np.all(np.isfinite(primal))
                    or not np.all(np.isfinite(row_value))
                    or not np.all(np.isfinite(row_dual))
                ):
                    raise OwnerContractError("optimal primal or dual is malformed")
                if np.any(primal < self._column_lower - PRIMAL_TOLERANCE) or np.any(
                    primal > self._column_upper + PRIMAL_TOLERANCE
                ):
                    raise OwnerContractError("optimal primal violates a column bound")
                if np.any(row_value < lower - PRIMAL_TOLERANCE) or np.any(
                    row_value > upper + PRIMAL_TOLERANCE
                ):
                    raise OwnerContractError("optimal primal violates a streamed row")
                upper_only = np.all(np.isneginf(lower)) and np.all(np.isfinite(upper))
                if upper_only and np.any(row_dual > DUAL_TOLERANCE):
                    raise OwnerContractError("MIN upper-row dual has the wrong sign")
                objective = float(self._h.getObjectiveValue())
                self._remaining()
                if not math.isfinite(objective):
                    raise OwnerContractError("optimal objective is non-finite")
                residual = _readonly_f64(upper - row_value)
                self._state = (
                    "SOLVED_BASE_OPTIMAL"
                    if self._current_load_tag == "base"
                    else "SOLVED_UPDATED_OPTIMAL"
                )
                return OptimalSolve(
                    status="OPTIMAL",
                    primal=primal,
                    row_value=row_value,
                    row_dual=row_dual,
                    upper_residual=residual,
                    full_row_ids=row_ids,
                    minimized_objective=objective,
                )
            if model_status != highspy.HighsModelStatus.kInfeasible:
                raise OwnerContractError("unsupported non-optimal model status")
            if self._current_load_tag == "updated":
                raise InfeasibleRayDisabled(
                    "updated/ineligible infeasible solve must stop without a second ray"
                )
            if self._dual_ray_used:
                raise OwnerContractError("base dual ray was already requested")
            # A base dual ray is requested exactly once and only for an
            # upper-only streamed frame.  Stale getSolution() data is never read.
            if not (np.all(np.isneginf(lower)) and np.all(np.isfinite(upper))):
                raise OwnerContractError("base ray contract requires upper-only rows")
            self._remaining()
            exist_status, exists = self._h.getDualRayExist()
            self._dual_ray_used = True
            self._require_ok(exist_status, "getDualRayExist")
            if exists is not True:
                raise OwnerContractError("infeasible model has no dual ray")
            self._remaining()
            ray_status, has_ray, raw_ray = self._h.getDualRay()
            self._remaining()
            self._require_ok(ray_status, "getDualRay")
            if has_ray is not True:
                raise OwnerContractError("getDualRay did not return a ray")
            ray = _readonly_f64(raw_ray)
            if (
                ray.size != row_ids.size
                or not np.all(np.isfinite(ray))
                or not np.any(ray != 0.0)
                or np.any(ray > 0.0)
            ):
                raise OwnerContractError("upper-only dual ray is malformed")
            row_rhs: Fraction | None = None
            bound_support: Fraction | None = None
            gap: Fraction | None = None
            if self._toy_algebraic_oracle:
                row_rhs, bound_support, gap = self._exact_dual_ray_certificate(ray)
            support = tuple(
                int(row_ids[index]) for index in np.flatnonzero(ray != 0.0)
            )
            self._state = "SOLVED_BASE_INFEASIBLE_RAY"
            return InfeasibleRaySolve(
                status="INFEASIBLE_DUAL_RAY",
                row_ray=ray,
                full_row_ids=row_ids,
                support_full_row_ids=support,
                toy_algebraic_oracle_ran=self._toy_algebraic_oracle,
                exact_row_rhs=None if row_rhs is None else str(row_rhs),
                exact_bound_support=(
                    None if bound_support is None else str(bound_support)
                ),
                exact_contradiction_gap=None if gap is None else str(gap),
            )
        except BaseException as error:
            self._poison("solve", error)
            raise

    def _clear_backend(self) -> None:
        if self._state == "CLOSED":
            return
        backend, self._h = self._h, None
        self._state = "CLOSED"
        if backend is None:
            return
        status = backend.clear()
        if status != highspy.HighsStatus.kOk:
            raise CleanupFailure("HiGHS clear returned non-kOk")


class RecordingHighs:
    def __init__(self) -> None:
        self.inner = highspy.Highs()
        self.add_rows_calls = 0
        self.run_calls = 0
        self.get_solution_calls = 0
        self.get_dual_ray_exist_calls = 0
        self.get_dual_ray_calls = 0
        self.clear_model_calls = 0
        self.clear_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def addRows(self, *args: Any) -> Any:
        self.add_rows_calls += 1
        return self.inner.addRows(*args)

    def run(self) -> Any:
        self.run_calls += 1
        return self.inner.run()

    def getSolution(self) -> Any:
        self.get_solution_calls += 1
        return self.inner.getSolution()

    def getDualRayExist(self) -> Any:
        self.get_dual_ray_exist_calls += 1
        return self.inner.getDualRayExist()

    def getDualRay(self) -> Any:
        self.get_dual_ray_calls += 1
        return self.inner.getDualRay()

    def clearModel(self) -> Any:
        self.clear_model_calls += 1
        return self.inner.clearModel()

    def clear(self) -> Any:
        self.clear_calls += 1
        return self.inner.clear()


class WarningAfterMutationHighs(RecordingHighs):
    def addRows(self, *args: Any) -> Any:
        self.add_rows_calls += 1
        status = self.inner.addRows(*args)
        if status != highspy.HighsStatus.kOk:
            return status
        return highspy.HighsStatus.kWarning


class DoubleFaultHighs(RecordingHighs):
    def __init__(self, primary: BaseException, cleanup: BaseException) -> None:
        super().__init__()
        self.primary = primary
        self.cleanup = cleanup

    def addRows(self, *args: Any) -> Any:
        del args
        self.add_rows_calls += 1
        raise self.primary

    def clear(self) -> Any:
        self.clear_calls += 1
        self.inner.clear()
        raise self.cleanup


class AdvanceAfterRunHighs(RecordingHighs):
    def __init__(self, clock: "ManualClock", late_time: float) -> None:
        super().__init__()
        self.clock = clock
        self.late_time = late_time

    def run(self) -> Any:
        self.run_calls += 1
        status = self.inner.run()
        self.clock.now = self.late_time
        return status


class AdvanceAfterRayExistHighs(RecordingHighs):
    def __init__(self, clock: "ManualClock", late_time: float) -> None:
        super().__init__()
        self.clock = clock
        self.late_time = late_time

    def getDualRayExist(self) -> Any:
        self.get_dual_ray_exist_calls += 1
        result = self.inner.getDualRayExist()
        self.clock.now = self.late_time
        return result


class AdvanceAfterRayHighs(RecordingHighs):
    def __init__(self, clock: "ManualClock", late_time: float) -> None:
        super().__init__()
        self.clock = clock
        self.late_time = late_time

    def getDualRay(self) -> Any:
        self.get_dual_ray_calls += 1
        result = self.inner.getDualRay()
        self.clock.now = self.late_time
        return result


class ManualClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class CountingHighsFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.backend: RecordingHighs | None = None

    def __call__(self) -> RecordingHighs:
        self.calls += 1
        if self.calls != 1:
            raise OwnerContractError("request constructed more than one Highs owner")
        self.backend = RecordingHighs()
        return self.backend


def _upper_batch(
    coefficients: Sequence[Sequence[float]],
    upper: Sequence[float],
    row_ids: Sequence[int],
) -> dict[str, Any]:
    rows = len(coefficients)
    cols = len(coefficients[0])
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for dense_row in coefficients:
        if len(dense_row) != cols:
            raise AssertionError("ragged toy row")
        for column, coefficient in enumerate(dense_row):
            if coefficient != 0.0:
                indices.append(column)
                data.append(float(coefficient))
        indptr.append(len(data))
    return {
        "rows": rows,
        "indptr": np.asarray(indptr, dtype=np.int32),
        "indices": np.asarray(indices, dtype=np.int32),
        "data": np.asarray(data, dtype=np.float64),
        "lower": np.full(rows, -np.inf, dtype=np.float64),
        "upper": np.asarray(upper, dtype=np.float64),
        "full_row_ids": np.asarray(row_ids, dtype=np.int64),
    }


def _new_owner(
    *,
    cost: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    backend: RecordingHighs | None = None,
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
    toy_algebraic_oracle: bool = True,
) -> SafeHighsOwner:
    if deadline is None:
        deadline = float(clock()) + 30.0
    return SafeHighsOwner(
        negative_objective=np.asarray(cost, dtype=np.float64),
        column_lower=np.asarray(lower, dtype=np.float64),
        column_upper=np.asarray(upper, dtype=np.float64),
        deadline_monotonic=float(deadline),
        highs_factory=(lambda: backend) if backend is not None else highspy.Highs,
        clock=clock,
        toy_algebraic_oracle=toy_algebraic_oracle,
    )


def _raw_warning_observation() -> dict[str, Any]:
    h = highspy.Highs()
    try:
        if h.setOptionValue("output_flag", False) != highspy.HighsStatus.kOk:
            raise AssertionError("raw warning output option failed")
        if h.setOptionValue("small_matrix_value", SMALL_MATRIX_VALUE) != highspy.HighsStatus.kOk:
            raise AssertionError("raw warning small option failed")
        status = h.addCols(
            1,
            np.zeros(1, dtype=np.float64),
            np.asarray([-1.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            0,
            np.zeros(2, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
        if status != highspy.HighsStatus.kOk:
            raise AssertionError("raw warning columns failed")
        warning = h.addRows(
            1,
            np.asarray([-np.inf], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            1,
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([SMALL_MATRIX_VALUE], dtype=np.float64),
        )
        result = {
            "status": str(warning),
            "rows_after_warning": h.getNumRow(),
            "nnz_after_warning": h.getNumNz(),
            "partial_mutation_observed": bool(
                warning == highspy.HighsStatus.kWarning
                and h.getNumRow() == 1
                and h.getNumNz() == 0
            ),
        }
        if not result["partial_mutation_observed"]:
            raise AssertionError("real HiGHS kWarning partial mutation changed")
        return result
    finally:
        if h.clear() != highspy.HighsStatus.kOk:
            raise AssertionError("raw warning clear failed")


def run_contract_sentinel() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    # Optimal MIN(-c): both upper rows are tight and have negative duals.
    optimal_backend = RecordingHighs()
    with _new_owner(
        cost=[-2.0, -1.0],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        backend=optimal_backend,
    ) as owner:
        owner.append_rows(**_upper_batch([[1.0, 1.0], [1.0, 0.0]], [1.0, 0.75], [10, 20]))
        optimal = owner.solve()
        assert isinstance(optimal, OptimalSolve)
        assert np.array_equal(optimal.primal, np.asarray([0.75, 0.25]))
        assert np.array_equal(optimal.row_value, np.asarray([1.0, 0.75]))
        assert np.array_equal(optimal.row_dual, np.asarray([-1.0, -1.0]))
        assert np.array_equal(optimal.upper_residual, np.zeros(2))
        assert np.array_equal(optimal.full_row_ids, np.asarray([10, 20]))
        checks["optimal_upper_dual"] = {
            "primal": optimal.primal.tolist(),
            "row_value": optimal.row_value.tolist(),
            "row_dual": optimal.row_dual.tolist(),
            "upper_residual": optimal.upper_residual.tolist(),
            "full_row_ids": optimal.full_row_ids.tolist(),
            "minimized_objective": optimal.minimized_objective,
            "getSolution_calls": optimal_backend.get_solution_calls,
        }
    assert optimal_backend.clear_calls == 1

    # Infeasible upper rows need a variable bound.  Duplicate rows demonstrate
    # why solver row positions must never be treated as full phase-row IDs.
    order_results = []
    for row_ids in ([101, 202], [202, 101]):
        backend = RecordingHighs()
        with _new_owner(
            cost=[0.0], lower=[0.0], upper=[1.0], backend=backend
        ) as owner:
            owner.append_rows(**_upper_batch([[1.0], [1.0]], [-1.0, -1.0], row_ids))
            infeasible = owner.solve()
            assert isinstance(infeasible, InfeasibleRaySolve)
            assert backend.get_dual_ray_exist_calls == 1
            assert backend.get_dual_ray_calls == 1
            assert backend.get_solution_calls == 0
            stale = np.asarray(backend.inner.getSolution().row_dual, dtype=np.float64)
            assert not np.array_equal(stale, infeasible.row_ray)
            order_results.append(
                {
                    "load_order_full_row_ids": list(row_ids),
                    "row_ray": infeasible.row_ray.tolist(),
                    "mapped_support_full_row_ids": list(
                        infeasible.support_full_row_ids
                    ),
                    "toy_algebraic_oracle_ran": infeasible.toy_algebraic_oracle_ran,
                    "exact_row_rhs": infeasible.exact_row_rhs,
                    "exact_bound_support": infeasible.exact_bound_support,
                    "exact_contradiction_gap": infeasible.exact_contradiction_gap,
                    "owner_getSolution_calls": backend.get_solution_calls,
                    "raw_stale_solution_row_dual": stale.tolist(),
                }
            )
        assert backend.clear_calls == 1
    assert order_results[0]["row_ray"] == [-1.0, 0.0]
    assert order_results[1]["row_ray"] == [-1.0, 0.0]
    assert order_results[0]["mapped_support_full_row_ids"] == [101]
    assert order_results[1]["mapped_support_full_row_ids"] == [202]
    checks["infeasible_upper_row_plus_bound_ray"] = {
        "orders": order_results,
        "interpretation": (
            "redundant row order changes semantic ray support; load-time sealed "
            "full-row mapping is mandatory, and stale solution.row_dual is not a ray"
        ),
    }

    # One native owner handles base -> clearModel -> updated.  The infeasible
    # updated solve stops without a second ray or stale getSolution read.
    one_factory = CountingHighsFactory()
    disabled_caught = False
    try:
        with SafeHighsOwner(
            negative_objective=np.asarray([-1.0], dtype=np.float64),
            column_lower=np.asarray([0.0], dtype=np.float64),
            column_upper=np.asarray([1.0], dtype=np.float64),
            deadline_monotonic=time.monotonic() + 30.0,
            highs_factory=one_factory,
        ) as owner:
            owner.append_rows(**_upper_batch([[1.0]], [0.5], [301]))
            base_result = owner.solve()
            assert isinstance(base_result, OptimalSolve)
            owner.reload_updated(
                negative_objective=np.asarray([-1.0], dtype=np.float64),
                column_lower=np.asarray([0.0], dtype=np.float64),
                column_upper=np.asarray([1.0], dtype=np.float64),
            )
            owner.append_rows(**_upper_batch([[1.0]], [-1.0], [303]))
            owner.solve()  # must not query a ray on this updated model
    except InfeasibleRayDisabled:
        disabled_caught = True
    assert one_factory.backend is not None
    disabled_backend = one_factory.backend
    assert disabled_caught
    assert one_factory.calls == 1
    assert disabled_backend.clear_model_calls == 1
    assert disabled_backend.run_calls == 2
    assert disabled_backend.get_dual_ray_exist_calls == 0
    assert disabled_backend.get_dual_ray_calls == 0
    assert disabled_backend.get_solution_calls == 1  # base optimal only
    assert disabled_backend.clear_calls == 1
    checks["updated_infeasible_no_second_ray"] = {
        "caught": disabled_caught,
        "Highs_construction_count": one_factory.calls,
        "clearModel_calls": disabled_backend.clear_model_calls,
        "run_calls": disabled_backend.run_calls,
        "getDualRayExist_calls": disabled_backend.get_dual_ray_exist_calls,
        "getDualRay_calls": disabled_backend.get_dual_ray_calls,
        "getSolution_calls_total": disabled_backend.get_solution_calls,
        "updated_getSolution_calls": disabled_backend.get_solution_calls - 1,
    }

    # Non-symmetric [-3,2] bound: correct U' is b-min = b+3e-12,
    # whereas the rejected b+max expression gives only b+2e-12.
    tiny_source = freeze_rows(
        rows=1,
        cols=2,
        indptr=np.asarray([0, 2], dtype=np.int32),
        indices=np.asarray([0, 1], dtype=np.int32),
        data=np.asarray([1.0, SMALL_MATRIX_VALUE], dtype=np.float64),
        lower=np.asarray([-np.inf], dtype=np.float64),
        upper=np.asarray([0.5], dtype=np.float64),
        full_row_ids=np.asarray([404], dtype=np.int64),
        allow_tiny=True,
    )
    tiny_lower = np.asarray([-1.0, -3.0], dtype=np.float64)
    tiny_upper = np.asarray([1.0, 2.0], dtype=np.float64)
    filtered = filter_tiny_outward(tiny_source, tiny_lower, tiny_upper)
    a = Fraction.from_float(SMALL_MATRIX_VALUE)
    exact_min = min(a * Fraction(-3), a * Fraction(2))
    correct_exact = Fraction.from_float(0.5) - exact_min
    rejected_exact = Fraction.from_float(0.5) + max(
        a * Fraction(-3), a * Fraction(2)
    )
    assert correct_exact > rejected_exact
    assert Fraction.from_float(float(filtered.rows.upper[0])) >= correct_exact
    assert filtered.deleted_nnz == 1
    assert filtered.rows.data.tolist() == [1.0]
    with _new_owner(
        cost=[-1.0, 0.0], lower=[-1.0, -3.0], upper=[1.0, 2.0]
    ) as owner:
        owner.append_rows(
            rows=filtered.rows.rows,
            indptr=filtered.rows.indptr,
            indices=filtered.rows.indices,
            data=filtered.rows.data,
            lower=filtered.rows.lower,
            upper=filtered.rows.upper,
            full_row_ids=filtered.rows.full_row_ids,
        )
        tiny_optimal = owner.solve()
        assert isinstance(tiny_optimal, OptimalSolve)
        assert tiny_optimal.primal[0] == filtered.rows.upper[0]
    checks["tiny_nonsymmetric_outward_projection"] = {
        "column_bounds": [[-1.0, 1.0], [-3.0, 2.0]],
        "deleted_coefficient": SMALL_MATRIX_VALUE,
        "correct_formula": "upper - sum(min(a*l,a*u))",
        "correct_exact_upper": str(correct_exact),
        "stored_outward_upper": float(filtered.rows.upper[0]),
        "rejected_b_plus_max_exact": str(rejected_exact),
        "correct_strictly_exceeds_rejected": bool(correct_exact > rejected_exact),
        "deleted_nnz": filtered.deleted_nnz,
    }

    checks["real_highs_warning_partial_mutation"] = _raw_warning_observation()
    warning_backend = WarningAfterMutationHighs()
    warning_caught = False
    try:
        with _new_owner(
            cost=[0.0], lower=[-1.0], upper=[1.0], backend=warning_backend
        ) as owner:
            owner.append_rows(**_upper_batch([[1.0]], [0.0], [505]))
    except OwnerContractError:
        warning_caught = True
    assert warning_caught
    assert warning_backend.add_rows_calls == 1
    assert warning_backend.clear_calls == 1
    assert warning_backend.inner.getNumRow() == 0
    checks["owner_warning_poison_and_discard"] = {
        "caught": warning_caught,
        "addRows_calls": warning_backend.add_rows_calls,
        "clear_calls": warning_backend.clear_calls,
        "rows_after_clear": warning_backend.inner.getNumRow(),
    }

    # Malformed/noncanonical CSR is rejected, never repaired, before addRows.
    malformed_checks = {}
    for name, mutation in (
        ("nan_data", "nan"),
        ("bad_indptr_terminal", "indptr"),
        ("duplicate_index", "duplicate"),
        ("explicit_zero", "zero"),
        ("int64_index", "int64"),
        ("large_matrix_value", "large"),
    ):
        backend = RecordingHighs()
        caught = False
        try:
            with _new_owner(
                cost=[0.0], lower=[-1.0], upper=[1.0], backend=backend
            ) as owner:
                batch = _upper_batch([[1.0]], [0.0], [606])
                if mutation == "nan":
                    batch["data"] = np.asarray([np.nan], dtype=np.float64)
                elif mutation == "indptr":
                    batch["indptr"] = np.asarray([0, 99], dtype=np.int32)
                elif mutation == "duplicate":
                    batch["indptr"] = np.asarray([0, 2], dtype=np.int32)
                    batch["indices"] = np.asarray([0, 0], dtype=np.int32)
                    batch["data"] = np.asarray([1.0, 2.0], dtype=np.float64)
                elif mutation == "zero":
                    batch["data"] = np.asarray([0.0], dtype=np.float64)
                elif mutation == "int64":
                    batch["indices"] = np.asarray([0], dtype=np.int64)
                elif mutation == "large":
                    batch["data"] = np.asarray(
                        [LARGE_MATRIX_VALUE], dtype=np.float64
                    )
                owner.append_rows(**batch)
        except OwnerContractError:
            caught = True
        assert caught and backend.add_rows_calls == 0 and backend.clear_calls == 1
        malformed_checks[name] = {
            "caught_before_native_addRows": caught,
            "native_addRows_calls": backend.add_rows_calls,
            "clear_calls": backend.clear_calls,
        }
    checks["malformed_prevalidation"] = malformed_checks

    # Tiny, non-finite, and infinite-cost-sized objectives stop before any
    # native Highs construction.  Supported objectives and bounds are also
    # read back bitwise immediately after addCols (exercised by every pass).
    objective_checks = {}
    for name, value in (
        ("tiny_nonzero", SMALL_MATRIX_VALUE),
        ("nan", np.nan),
        ("positive_infinity", np.inf),
        ("infinite_cost_threshold", INFINITE_BOUND),
    ):
        factory = CountingHighsFactory()
        caught = False
        try:
            SafeHighsOwner(
                negative_objective=np.asarray([value], dtype=np.float64),
                column_lower=np.asarray([-1.0], dtype=np.float64),
                column_upper=np.asarray([1.0], dtype=np.float64),
                deadline_monotonic=time.monotonic() + 30.0,
                highs_factory=factory,
            )
        except OwnerContractError:
            caught = True
        assert caught and factory.calls == 0
        objective_checks[name] = {
            "caught_before_Highs_construction": caught,
            "Highs_construction_count": factory.calls,
        }
    checks["objective_and_column_frame_contract"] = {
        "invalid_cases": objective_checks,
        "tiny_objective_rule": (
            "0 < abs(c) <= 1e-12 is UNKNOWN; objective coefficients are never deleted"
        ),
        "supported_cost_and_bounds_addCols_getLp_bitwise_readback": True,
    }

    # An expired deadline poisons and clears without entering run().
    manual_clock = ManualClock(0.0)
    deadline_backend = RecordingHighs()
    deadline_caught = False
    try:
        with _new_owner(
            cost=[0.0],
            lower=[-1.0],
            upper=[1.0],
            backend=deadline_backend,
            clock=manual_clock,
            deadline=1.0,
        ) as owner:
            owner.append_rows(**_upper_batch([[1.0]], [0.0], [707]))
            manual_clock.now = 2.0
            owner.solve()
    except DeadlineExpired:
        deadline_caught = True
    assert deadline_caught and deadline_backend.run_calls == 0
    assert deadline_backend.clear_calls == 1
    checks["deadline_before_run"] = {
        "caught": deadline_caught,
        "run_calls": deadline_backend.run_calls,
        "clear_calls": deadline_backend.clear_calls,
    }

    # A result/ray arriving after the deadline is stale and is not read or
    # returned.  Exercise the three post-call boundaries explicitly.
    late_checks = {}
    late_clock = ManualClock(0.0)
    late_run_backend = AdvanceAfterRunHighs(late_clock, 2.0)
    caught = False
    try:
        with _new_owner(
            cost=[-1.0],
            lower=[-1.0],
            upper=[1.0],
            backend=late_run_backend,
            clock=late_clock,
            deadline=1.0,
        ) as owner:
            owner.append_rows(**_upper_batch([[1.0]], [0.5], [710]))
            owner.solve()
    except DeadlineExpired:
        caught = True
    assert caught and late_run_backend.get_solution_calls == 0
    late_checks["after_run_before_solution"] = {
        "caught": caught,
        "getSolution_calls": late_run_backend.get_solution_calls,
    }

    ray_exist_clock = ManualClock(0.0)
    ray_exist_backend = AdvanceAfterRayExistHighs(ray_exist_clock, 2.0)
    caught = False
    try:
        with _new_owner(
            cost=[0.0],
            lower=[0.0],
            upper=[1.0],
            backend=ray_exist_backend,
            clock=ray_exist_clock,
            deadline=1.0,
        ) as owner:
            owner.append_rows(**_upper_batch([[1.0]], [-1.0], [711]))
            owner.solve()
    except DeadlineExpired:
        caught = True
    assert caught and ray_exist_backend.get_dual_ray_exist_calls == 1
    assert ray_exist_backend.get_dual_ray_calls == 0
    late_checks["after_ray_exist_before_ray"] = {
        "caught": caught,
        "getDualRayExist_calls": ray_exist_backend.get_dual_ray_exist_calls,
        "getDualRay_calls": ray_exist_backend.get_dual_ray_calls,
    }

    ray_clock = ManualClock(0.0)
    ray_backend = AdvanceAfterRayHighs(ray_clock, 2.0)
    caught = False
    try:
        with _new_owner(
            cost=[0.0],
            lower=[0.0],
            upper=[1.0],
            backend=ray_backend,
            clock=ray_clock,
            deadline=1.0,
        ) as owner:
            owner.append_rows(**_upper_batch([[1.0]], [-1.0], [712]))
            owner.solve()
    except DeadlineExpired:
        caught = True
    assert caught and ray_backend.get_dual_ray_exist_calls == 1
    assert ray_backend.get_dual_ray_calls == 1
    late_checks["after_ray_no_late_result"] = {
        "caught": caught,
        "getDualRayExist_calls": ray_backend.get_dual_ray_exist_calls,
        "getDualRay_calls": ray_backend.get_dual_ray_calls,
        "late_ray_returned_to_caller": False,
    }
    checks["deadline_post_call_boundaries"] = late_checks

    # Cleanup failure is secondary: the exact primary BaseException identity wins.
    primary = InjectedPrimary("primary-system-boundary")
    cleanup = InjectedCleanup("secondary-clear")
    double_backend = DoubleFaultHighs(primary, cleanup)
    caught_primary: BaseException | None = None
    try:
        with _new_owner(
            cost=[0.0], lower=[-1.0], upper=[1.0], backend=double_backend
        ) as owner:
            owner.append_rows(**_upper_batch([[1.0]], [0.0], [808]))
    except BaseException as error:
        caught_primary = error
    assert caught_primary is primary
    notes = list(getattr(caught_primary, "__notes__", ()))
    assert any("secondary HiGHS cleanup failure" in note for note in notes)
    assert double_backend.clear_calls == 1
    assert double_backend.inner.getNumCol() == 0
    checks["baseexception_primary_preserved_cleanup_secondary"] = {
        "same_primary_identity": caught_primary is primary,
        "primary_type": type(caught_primary).__name__,
        "cleanup_type": type(cleanup).__name__,
        "secondary_cleanup_noted": any(
            "secondary HiGHS cleanup failure" in note for note in notes
        ),
        "hostile_cleanup_stringification_avoided": True,
        "clear_calls": double_backend.clear_calls,
        "columns_after_inner_clear": double_backend.inner.getNumCol(),
    }

    return {
        "schema": "act.scratch.phase_projection_highs_owner_contract.v1",
        "created_at": "2026-08-14",
        "status": "PASS_CPU_ONLY_SCRATCH_CONTRACT",
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "production_modified": False,
        "benchmark_run": False,
        "gpu_used": False,
        "highs_pin": {
            "version": HIGHS_VERSION,
            "githash": HIGHS_GITHASH,
            "options": dict(SafeHighsOwner._PINNED_OPTIONS),
            "objective_sense": "MINIMIZE(-candidate_objective)",
        },
        "contract": {
            "one_Highs_construction_per_request": True,
            "request_lifecycle": "base load/solve -> clearModel -> optional updated load/solve -> clear",
            "base_infeasible_ray_sequence": "getDualRayExist exactly once, then getDualRay exactly once",
            "updated_infeasible_ray": "disabled; fail closed without a second ray",
            "upper_only_ray_sign": "all components <= 0 and at least one exact nonzero",
            "runtime_ray_validation": "state/shape/finite/upper-sign/exact-nonzero only; candidate selection only",
            "toy_algebraic_oracle": "stored-binary64 Fraction row-plus-bound contradiction, CPU sentinel only",
            "runtime_algebraic_replay": False,
            "row_mapping": "full row ids copied, sealed, hashed, and bound to append order before addRows",
            "objective_tiny": "0 < abs(c) <= 1e-12 fails closed; never delete objective coefficients",
            "deadline_checks": "before/after run solution and dual-ray retrieval; late results are discarded",
            "any_non_kOk": "POISONED; no retry; BaseException-safe clear and discard",
            "candidate_or_ray_authority": False,
        },
        "checks": checks,
        "all_checks_passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_contract_sentinel(), indent=2, sort_keys=True))
