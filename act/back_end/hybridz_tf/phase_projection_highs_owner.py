"""Private, fail-closed HiGHS owner for phase-projection candidates.

This module owns one deterministic HiGHS instance for a base LP and at most
one updated LP.  Solver primals, row duals, and infeasibility rays are selection
data only: nothing returned here has verifier or proof authority.

The module is intentionally not exported by :mod:`act.back_end.hybridz_tf`.
It has no solver fallback, runtime menu, sampling path, or algebraic Farkas
replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import time
from typing import Any, Literal, TypeAlias

import highspy
import numpy as np
import scipy.sparse as sp


__all__: tuple[str, ...] = ()


_HIGHS_VERSION = "1.15.0"
_HIGHS_GITHASH = "8396001"
_SMALL_MATRIX_VALUE = 1.0e-12
_LARGE_MATRIX_VALUE = 1.0e15
_INFINITE_BOUND = 1.0e20
_PRIMAL_TOLERANCE = 1.0e-9
_DUAL_TOLERANCE = 1.0e-9

_PINNED_OPTIONS: tuple[tuple[str, Any], ...] = (
    ("output_flag", False),
    ("solver", "simplex"),
    ("presolve", "off"),
    ("simplex_strategy", 1),
    ("simplex_scale_strategy", 2),
    ("threads", 1),
    ("parallel", "off"),
    ("random_seed", 0),
    ("small_matrix_value", _SMALL_MATRIX_VALUE),
    ("large_matrix_value", _LARGE_MATRIX_VALUE),
    ("infinite_bound", _INFINITE_BOUND),
    ("primal_feasibility_tolerance", _PRIMAL_TOLERANCE),
    ("dual_feasibility_tolerance", _DUAL_TOLERANCE),
)


class HighsOwnerUnknown(RuntimeError):
    """The candidate transaction must stop without making a claim."""


class HighsOwnerCleanupError(HighsOwnerUnknown):
    """Final native cleanup failed after the owner was made unusable."""


class HighsOwnerDeadline(HighsOwnerUnknown):
    """The request deadline expired; any late native result is discarded."""


def _add_secondary_type_note(primary: BaseException, cleanup: BaseException) -> None:
    try:
        primary.add_note(
            "secondary HiGHS cleanup failure type=" f"{type(cleanup).__name__}"
        )
    except BaseException:
        pass


def _copy_vector(value: Any, dtype: np.dtype[Any], name: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.ndim != 1
        or not value.flags.c_contiguous
    ):
        raise HighsOwnerUnknown(f"{name} must be a contiguous {dtype} vector")
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _round_fraction_up(value: Fraction) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise HighsOwnerUnknown("tiny projection overflowed its upper row bound")
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    if not math.isfinite(rounded) or Fraction.from_float(rounded) < value:
        raise HighsOwnerUnknown("tiny projection could not round its upper bound outward")
    return rounded


def _round_fraction_down(value: Fraction) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise HighsOwnerUnknown("tiny projection overflowed its lower row bound")
    if Fraction.from_float(rounded) > value:
        rounded = float(np.nextafter(rounded, -np.inf))
    if not math.isfinite(rounded) or Fraction.from_float(rounded) > value:
        raise HighsOwnerUnknown("tiny projection could not round its lower bound outward")
    return rounded


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _validate_column_bounds(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower_copy = _copy_vector(lower, np.dtype(np.float64), "column lower")
    upper_copy = _copy_vector(upper, np.dtype(np.float64), "column upper")
    if (
        lower_copy.size == 0
        or lower_copy.size != upper_copy.size
        or not np.all(np.isfinite(lower_copy))
        or not np.all(np.isfinite(upper_copy))
        or np.any(lower_copy > upper_copy)
        or np.any(np.abs(lower_copy) >= _INFINITE_BOUND)
        or np.any(np.abs(upper_copy) >= _INFINITE_BOUND)
    ):
        raise HighsOwnerUnknown("column bounds are malformed or outside the finite frame")
    return lower_copy, upper_copy


@dataclass(frozen=True)
class FrozenRows:
    """Owned canonical CSR rows, safely projected over deleted tiny terms.

    If ``L <= A*x <= U`` and a tiny term ``a*x_j`` is removed, projection over
    the supplied column box uses ``L - max(a*l, a*u)`` and
    ``U - min(a*l, a*u)``.  Every operation uses the exact dyadic value of the
    stored binary64 inputs, followed by one outward rounding.  This is a
    candidate-only relaxation, not a proof transformation.
    """

    rows: int
    columns: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_ids: np.ndarray
    column_lower: np.ndarray
    column_upper: np.ndarray
    logical_nnz: int
    deleted_tiny_nnz: int
    content_sha256: str

    @classmethod
    def from_csr(
        cls,
        matrix: sp.csr_matrix,
        *,
        row_lower: np.ndarray,
        row_upper: np.ndarray,
        row_ids: np.ndarray,
        column_lower: np.ndarray,
        column_upper: np.ndarray,
    ) -> "FrozenRows":
        if (
            not sp.isspmatrix_csr(matrix)
            or matrix.dtype != np.dtype(np.float64)
            or matrix.ndim != 2
            or matrix.shape[0] <= 0
            or matrix.shape[1] <= 0
        ):
            raise HighsOwnerUnknown("rows must be a nonempty pre-existing float64 CSR")
        if (
            type(matrix.indptr) is not np.ndarray
            or type(matrix.indices) is not np.ndarray
            or type(matrix.data) is not np.ndarray
            or matrix.indptr.dtype != np.dtype(np.int32)
            or matrix.indices.dtype != np.dtype(np.int32)
            or matrix.data.dtype != np.dtype(np.float64)
            or not matrix.indptr.flags.c_contiguous
            or not matrix.indices.flags.c_contiguous
            or not matrix.data.flags.c_contiguous
        ):
            raise HighsOwnerUnknown("CSR arrays must be contiguous int32/int32/float64")

        rows, columns = (int(matrix.shape[0]), int(matrix.shape[1]))
        nnz = int(matrix.data.size)
        int32_max = int(np.iinfo(np.int32).max)
        if (
            rows > int32_max
            or columns > int32_max
            or nnz > int32_max
            or matrix.indptr.shape != (rows + 1,)
            or int(matrix.indptr[0]) != 0
            or int(matrix.indptr[-1]) != nnz
            or matrix.indices.size != nnz
            or np.any(matrix.indptr[1:] < matrix.indptr[:-1])
            or (nnz and np.any(matrix.indices < 0))
            or (nnz and np.any(matrix.indices >= columns))
        ):
            raise HighsOwnerUnknown("CSR shape, pointers, or indices are malformed")
        for row in range(rows):
            start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
            if stop - start > 1 and np.any(
                matrix.indices[start + 1 : stop] <= matrix.indices[start : stop - 1]
            ):
                raise HighsOwnerUnknown(
                    "CSR rows must have strictly increasing, duplicate-free indices"
                )
        if (
            not np.all(np.isfinite(matrix.data))
            or np.any(matrix.data == 0.0)
            or np.any(np.abs(matrix.data) >= _LARGE_MATRIX_VALUE)
        ):
            raise HighsOwnerUnknown(
                "CSR data contain nonfinite, explicit-zero, or oversized coefficients"
            )

        lower = _copy_vector(row_lower, np.dtype(np.float64), "row lower")
        upper = _copy_vector(row_upper, np.dtype(np.float64), "row upper")
        ids = _copy_vector(row_ids, np.dtype(np.int64), "row ids")
        col_lower, col_upper = _validate_column_bounds(column_lower, column_upper)
        if (
            lower.size != rows
            or upper.size != rows
            or ids.size != rows
            or col_lower.size != columns
            or np.unique(ids).size != rows
            or np.any(np.isnan(lower))
            or np.any(np.isnan(upper))
            or np.any(np.isposinf(lower))
            or np.any(np.isneginf(upper))
            or np.any(lower > upper)
        ):
            raise HighsOwnerUnknown("row bounds or sealed row ids are malformed")
        finite_lower = lower[np.isfinite(lower)]
        finite_upper = upper[np.isfinite(upper)]
        if (
            np.any(np.abs(finite_lower) >= _INFINITE_BOUND)
            or np.any(np.abs(finite_upper) >= _INFINITE_BOUND)
        ):
            raise HighsOwnerUnknown("finite row bound reaches the HiGHS infinity threshold")

        tiny = np.abs(matrix.data) <= _SMALL_MATRIX_VALUE
        deleted = int(np.count_nonzero(tiny))
        adjusted_lower = np.array(lower, dtype=np.float64, order="C", copy=True)
        adjusted_upper = np.array(upper, dtype=np.float64, order="C", copy=True)
        if deleted:
            tiny_positions = np.flatnonzero(tiny)
            tiny_rows = (
                np.searchsorted(matrix.indptr, tiny_positions, side="right") - 1
            )
            deleted_ranges: dict[int, tuple[Fraction, Fraction]] = {}
            for position, row_value in zip(tiny_positions, tiny_rows):
                row = int(row_value)
                column = int(matrix.indices[position])
                coefficient = Fraction.from_float(float(matrix.data[position]))
                lo = coefficient * Fraction.from_float(float(col_lower[column]))
                hi = coefficient * Fraction.from_float(float(col_upper[column]))
                prior_min, prior_max = deleted_ranges.get(
                    row, (Fraction(0), Fraction(0))
                )
                deleted_ranges[row] = (
                    prior_min + min(lo, hi),
                    prior_max + max(lo, hi),
                )
            for row, (deleted_min, deleted_max) in deleted_ranges.items():
                if math.isfinite(float(lower[row])):
                    exact_lower = Fraction.from_float(float(lower[row])) - deleted_max
                    adjusted_lower[row] = _round_fraction_down(exact_lower)
                if math.isfinite(float(upper[row])):
                    exact_upper = Fraction.from_float(float(upper[row])) - deleted_min
                    adjusted_upper[row] = _round_fraction_up(exact_upper)

        kept = ~tiny
        data = np.array(matrix.data[kept], dtype=np.float64, order="C", copy=True)
        indices = np.array(matrix.indices[kept], dtype=np.int32, order="C", copy=True)
        counts = np.diff(matrix.indptr).astype(np.int64, copy=True)
        if deleted:
            counts -= np.bincount(tiny_rows, minlength=rows).astype(np.int64)
        indptr64 = np.empty(rows + 1, dtype=np.int64)
        indptr64[0] = 0
        np.cumsum(counts, out=indptr64[1:])
        if int(indptr64[-1]) > int32_max:
            raise HighsOwnerUnknown("filtered CSR exceeds int32 capacity")
        indptr = np.array(indptr64, dtype=np.int32, order="C", copy=True)

        finite_adjusted_lower = adjusted_lower[np.isfinite(adjusted_lower)]
        finite_adjusted_upper = adjusted_upper[np.isfinite(adjusted_upper)]
        if (
            np.any(adjusted_lower > adjusted_upper)
            or np.any(np.abs(finite_adjusted_lower) >= _INFINITE_BOUND)
            or np.any(np.abs(finite_adjusted_upper) >= _INFINITE_BOUND)
            or (data.size and np.any(np.abs(data) <= _SMALL_MATRIX_VALUE))
        ):
            raise HighsOwnerUnknown("outward tiny projection left the native numeric frame")

        adjusted_lower.setflags(write=False)
        adjusted_upper.setflags(write=False)
        indptr.setflags(write=False)
        indices.setflags(write=False)
        data.setflags(write=False)
        digest = _digest_arrays(
            indptr,
            indices,
            data,
            adjusted_lower,
            adjusted_upper,
            ids,
            col_lower,
            col_upper,
        )
        return cls(
            rows=rows,
            columns=columns,
            indptr=indptr,
            indices=indices,
            data=data,
            lower=adjusted_lower,
            upper=adjusted_upper,
            row_ids=ids,
            column_lower=col_lower,
            column_upper=col_upper,
            logical_nnz=nnz,
            deleted_tiny_nnz=deleted,
            content_sha256=digest,
        )

    @property
    def upper_only(self) -> bool:
        return bool(np.all(np.isneginf(self.lower)) and np.all(np.isfinite(self.upper)))

    def assert_intact(self) -> None:
        arrays = (
            self.indptr,
            self.indices,
            self.data,
            self.lower,
            self.upper,
            self.row_ids,
            self.column_lower,
            self.column_upper,
        )
        if (
            type(self.rows) is not int
            or type(self.columns) is not int
            or type(self.logical_nnz) is not int
            or type(self.deleted_tiny_nnz) is not int
            or self.rows <= 0
            or self.columns <= 0
            or self.indptr.shape != (self.rows + 1,)
            or self.indices.shape != self.data.shape
            or self.lower.shape != (self.rows,)
            or self.upper.shape != (self.rows,)
            or self.row_ids.shape != (self.rows,)
            or self.column_lower.shape != (self.columns,)
            or self.column_upper.shape != (self.columns,)
            or int(self.indptr[0]) != 0
            or int(self.indptr[-1]) != self.data.size
            or self.logical_nnz != self.data.size + self.deleted_tiny_nnz
            or self.deleted_tiny_nnz < 0
        ):
            raise HighsOwnerUnknown("frozen CSR or mapping metadata changed")
        if any(array.flags.writeable or not array.flags.c_contiguous for array in arrays):
            raise HighsOwnerUnknown("frozen CSR or mapping is no longer read-only")
        if _digest_arrays(*arrays) != self.content_sha256:
            raise HighsOwnerUnknown("frozen CSR or mapping content changed")


def _freeze_row_bounds(
    lower: np.ndarray, upper: np.ndarray, rows: int
) -> tuple[np.ndarray, np.ndarray]:
    frozen_lower = _copy_vector(lower, np.dtype(np.float64), "row lower")
    frozen_upper = _copy_vector(upper, np.dtype(np.float64), "row upper")
    if (
        frozen_lower.size != rows
        or frozen_upper.size != rows
        or np.any(np.isnan(frozen_lower))
        or np.any(np.isnan(frozen_upper))
        or np.any(np.isposinf(frozen_lower))
        or np.any(np.isneginf(frozen_upper))
        or np.any(frozen_lower > frozen_upper)
    ):
        raise HighsOwnerUnknown("row bounds are malformed")
    finite_lower = frozen_lower[np.isfinite(frozen_lower)]
    finite_upper = frozen_upper[np.isfinite(frozen_upper)]
    if (
        np.any(np.abs(finite_lower) >= _INFINITE_BOUND)
        or np.any(np.abs(finite_upper) >= _INFINITE_BOUND)
    ):
        raise HighsOwnerUnknown("finite row bound reaches the infinity threshold")
    return frozen_lower, frozen_upper


@dataclass(frozen=True)
class FrozenNewColumns:
    """Sealed auxiliary columns over the already loaded base-row order.

    The matrix is canonical CSC with shape ``(base_rows, K)``.  Tiny matrix
    entries remain sealed here and are projected out, using the new column
    bounds, before any native incremental mutation begins.
    """

    rows: int
    columns: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    cost: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_ids: np.ndarray
    content_sha256: str

    @classmethod
    def from_csc(
        cls,
        matrix: sp.csc_matrix,
        *,
        cost: np.ndarray,
        column_lower: np.ndarray,
        column_upper: np.ndarray,
        existing_row_ids: np.ndarray,
    ) -> "FrozenNewColumns":
        if (
            not sp.isspmatrix_csc(matrix)
            or matrix.dtype != np.dtype(np.float64)
            or matrix.ndim != 2
            or matrix.shape[0] <= 0
            or matrix.shape[1] <= 0
        ):
            raise HighsOwnerUnknown(
                "new columns must be a nonempty pre-existing float64 CSC"
            )
        if (
            type(matrix.indptr) is not np.ndarray
            or type(matrix.indices) is not np.ndarray
            or type(matrix.data) is not np.ndarray
            or matrix.indptr.dtype != np.dtype(np.int32)
            or matrix.indices.dtype != np.dtype(np.int32)
            or matrix.data.dtype != np.dtype(np.float64)
            or not matrix.indptr.flags.c_contiguous
            or not matrix.indices.flags.c_contiguous
            or not matrix.data.flags.c_contiguous
        ):
            raise HighsOwnerUnknown("CSC arrays must be contiguous int32/int32/float64")
        rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
        nnz = int(matrix.data.size)
        int32_max = int(np.iinfo(np.int32).max)
        if (
            rows > int32_max
            or columns > int32_max
            or nnz > int32_max
            or matrix.indptr.shape != (columns + 1,)
            or int(matrix.indptr[0]) != 0
            or int(matrix.indptr[-1]) != nnz
            or matrix.indices.size != nnz
            or np.any(matrix.indptr[1:] < matrix.indptr[:-1])
            or (nnz and np.any(matrix.indices < 0))
            or (nnz and np.any(matrix.indices >= rows))
        ):
            raise HighsOwnerUnknown("CSC shape, pointers, or row indices are malformed")
        for column in range(columns):
            start, stop = int(matrix.indptr[column]), int(matrix.indptr[column + 1])
            if stop - start > 1 and np.any(
                matrix.indices[start + 1 : stop]
                <= matrix.indices[start : stop - 1]
            ):
                raise HighsOwnerUnknown(
                    "CSC columns must have strictly increasing, duplicate-free rows"
                )
        if (
            not np.all(np.isfinite(matrix.data))
            or np.any(matrix.data == 0.0)
            or np.any(np.abs(matrix.data) >= _LARGE_MATRIX_VALUE)
        ):
            raise HighsOwnerUnknown(
                "CSC data contain nonfinite, explicit-zero, or oversized coefficients"
            )
        columns_frame = _freeze_columns(cost, column_lower, column_upper)
        if columns_frame.cost.size != columns:
            raise HighsOwnerUnknown("new-column objective width differs from its CSC")
        row_ids = _copy_vector(
            existing_row_ids, np.dtype(np.int64), "existing row ids"
        )
        if row_ids.size != rows or np.unique(row_ids).size != rows:
            raise HighsOwnerUnknown("new-column existing-row ids are malformed")
        indptr = np.array(matrix.indptr, dtype=np.int32, order="C", copy=True)
        indices = np.array(matrix.indices, dtype=np.int32, order="C", copy=True)
        data = np.array(matrix.data, dtype=np.float64, order="C", copy=True)
        for array in (indptr, indices, data):
            array.setflags(write=False)
        digest = _digest_arrays(
            indptr,
            indices,
            data,
            columns_frame.cost,
            columns_frame.lower,
            columns_frame.upper,
            row_ids,
        )
        return cls(
            rows=rows,
            columns=columns,
            indptr=indptr,
            indices=indices,
            data=data,
            cost=columns_frame.cost,
            lower=columns_frame.lower,
            upper=columns_frame.upper,
            row_ids=row_ids,
            content_sha256=digest,
        )

    def assert_intact(self) -> None:
        arrays = (
            self.indptr,
            self.indices,
            self.data,
            self.cost,
            self.lower,
            self.upper,
            self.row_ids,
        )
        if (
            type(self.rows) is not int
            or type(self.columns) is not int
            or self.rows <= 0
            or self.columns <= 0
            or self.indptr.shape != (self.columns + 1,)
            or self.indices.shape != self.data.shape
            or self.cost.shape != (self.columns,)
            or self.lower.shape != (self.columns,)
            or self.upper.shape != (self.columns,)
            or self.row_ids.shape != (self.rows,)
            or int(self.indptr[0]) != 0
            or int(self.indptr[-1]) != self.data.size
        ):
            raise HighsOwnerUnknown("frozen auxiliary-column metadata changed")
        if any(array.flags.writeable or not array.flags.c_contiguous for array in arrays):
            raise HighsOwnerUnknown("frozen auxiliary columns are no longer read-only")
        if _digest_arrays(*arrays) != self.content_sha256:
            raise HighsOwnerUnknown("frozen auxiliary-column content changed")


@dataclass(frozen=True)
class _ProjectedNewColumns:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray


def _project_new_column_tiny(
    columns: FrozenNewColumns,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
) -> _ProjectedNewColumns:
    """Freeze row bounds and remove tiny CSC entries with outward projection."""

    columns.assert_intact()
    adjusted_lower, adjusted_upper = _freeze_row_bounds(
        row_lower, row_upper, columns.rows
    )
    adjusted_lower = np.array(adjusted_lower, dtype=np.float64, order="C", copy=True)
    adjusted_upper = np.array(adjusted_upper, dtype=np.float64, order="C", copy=True)
    tiny = np.abs(columns.data) <= _SMALL_MATRIX_VALUE
    deleted = int(np.count_nonzero(tiny))
    if deleted:
        tiny_positions = np.flatnonzero(tiny)
        tiny_columns = (
            np.searchsorted(columns.indptr, tiny_positions, side="right") - 1
        )
        deleted_ranges: dict[int, tuple[Fraction, Fraction]] = {}
        for position, column_value in zip(tiny_positions, tiny_columns):
            row = int(columns.indices[position])
            column = int(column_value)
            coefficient = Fraction.from_float(float(columns.data[position]))
            lo = coefficient * Fraction.from_float(float(columns.lower[column]))
            hi = coefficient * Fraction.from_float(float(columns.upper[column]))
            prior_min, prior_max = deleted_ranges.get(
                row, (Fraction(0), Fraction(0))
            )
            deleted_ranges[row] = (
                prior_min + min(lo, hi),
                prior_max + max(lo, hi),
            )
        for row, (deleted_min, deleted_max) in deleted_ranges.items():
            if math.isfinite(float(adjusted_lower[row])):
                exact_lower = (
                    Fraction.from_float(float(adjusted_lower[row])) - deleted_max
                )
                adjusted_lower[row] = _round_fraction_down(exact_lower)
            if math.isfinite(float(adjusted_upper[row])):
                exact_upper = (
                    Fraction.from_float(float(adjusted_upper[row])) - deleted_min
                )
                adjusted_upper[row] = _round_fraction_up(exact_upper)
    kept = ~tiny
    data = np.array(columns.data[kept], dtype=np.float64, order="C", copy=True)
    indices = np.array(columns.indices[kept], dtype=np.int32, order="C", copy=True)
    counts = np.diff(columns.indptr).astype(np.int64, copy=True)
    if deleted:
        counts -= np.bincount(tiny_columns, minlength=columns.columns).astype(np.int64)
    indptr64 = np.empty(columns.columns + 1, dtype=np.int64)
    indptr64[0] = 0
    np.cumsum(counts, out=indptr64[1:])
    indptr = np.array(indptr64, dtype=np.int32, order="C", copy=True)
    finite_lower = adjusted_lower[np.isfinite(adjusted_lower)]
    finite_upper = adjusted_upper[np.isfinite(adjusted_upper)]
    if (
        np.any(adjusted_lower > adjusted_upper)
        or np.any(np.abs(finite_lower) >= _INFINITE_BOUND)
        or np.any(np.abs(finite_upper) >= _INFINITE_BOUND)
        or (data.size and np.any(np.abs(data) <= _SMALL_MATRIX_VALUE))
    ):
        raise HighsOwnerUnknown("new-column tiny projection left the numeric frame")
    for array in (indptr, indices, data, adjusted_lower, adjusted_upper):
        array.setflags(write=False)
    return _ProjectedNewColumns(
        indptr=indptr,
        indices=indices,
        data=data,
        row_lower=adjusted_lower,
        row_upper=adjusted_upper,
    )


@dataclass(frozen=True)
class OptimalSelector:
    factors: np.ndarray
    row_value: np.ndarray
    row_dual: np.ndarray
    row_ids: np.ndarray
    minimized_objective: float


@dataclass(frozen=True)
class InfeasibleRaySelector:
    row_ray: np.ndarray
    row_ids: np.ndarray
    support_row_ids: tuple[int, ...]


@dataclass(frozen=True)
class OptimalCandidate:
    factors: np.ndarray
    minimized_objective: float


@dataclass(frozen=True)
class Unresolved:
    model_status: Any


BaseResult: TypeAlias = OptimalSelector | InfeasibleRaySelector | Unresolved
UpdateResult: TypeAlias = OptimalCandidate | Unresolved


@dataclass(frozen=True)
class _Columns:
    cost: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _freeze_columns(
    cost: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> _Columns:
    frozen_cost = _copy_vector(cost, np.dtype(np.float64), "objective")
    frozen_lower, frozen_upper = _validate_column_bounds(lower, upper)
    if frozen_cost.size != frozen_lower.size:
        raise HighsOwnerUnknown("objective and column bounds have different widths")
    if (
        not np.all(np.isfinite(frozen_cost))
        or np.any(np.abs(frozen_cost) >= _INFINITE_BOUND)
    ):
        raise HighsOwnerUnknown("objective is nonfinite or reaches the infinity threshold")
    tiny_objective = (frozen_cost != 0.0) & (
        np.abs(frozen_cost) <= _SMALL_MATRIX_VALUE
    )
    if np.any(tiny_objective):
        raise HighsOwnerUnknown("tiny nonzero objective coefficient is unsupported")
    return _Columns(frozen_cost, frozen_lower, frozen_upper)


def _readonly_f64(value: Any) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True).reshape(-1)
    result.setflags(write=False)
    return result


def _readonly_i64(value: Any) -> np.ndarray:
    result = np.array(value, dtype=np.int64, order="C", copy=True).reshape(-1)
    result.setflags(write=False)
    return result


class SafeHighsOwner:
    """One request-local Highs owner with one optional deterministic update."""

    def __init__(self, *, deadline_monotonic: float) -> None:
        if type(deadline_monotonic) is not float or not math.isfinite(
            deadline_monotonic
        ):
            raise HighsOwnerUnknown("deadline must be a finite monotonic float")
        self._deadline = deadline_monotonic
        self._state = "READY_BASE"
        self._highs: Any | None = None
        self._base_ray_requested = False
        self._constructed = False
        self._base_columns: _Columns | None = None
        self._base_rows: FrozenRows | None = None

    @property
    def state(self) -> str:
        return self._state

    def __enter__(self) -> "SafeHighsOwner":
        return self

    def __exit__(self, _kind: Any, primary: BaseException | None, _tb: Any) -> bool:
        try:
            self.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            _add_secondary_type_note(primary, cleanup)
        return False

    def _remaining(self) -> float:
        remaining = float(self._deadline - time.monotonic())
        if not math.isfinite(remaining) or remaining <= 0.0:
            raise HighsOwnerDeadline("HiGHS owner deadline expired")
        return remaining

    @staticmethod
    def _require_ok(status: Any, operation: str) -> None:
        if status != highspy.HighsStatus.kOk:
            raise HighsOwnerUnknown(f"HiGHS {operation} returned non-kOk")

    def _guard_live(self, expected: str) -> None:
        if self._state == "CLOSED":
            raise HighsOwnerUnknown("closed HiGHS owner rejects every operation")
        if self._state == "POISONED":
            raise HighsOwnerUnknown("poisoned HiGHS owner rejects every operation")
        if self._state != expected:
            primary = HighsOwnerUnknown("HiGHS owner lifecycle order is invalid")
            self._poison_and_discard(primary)
            raise primary

    def _poison_and_discard(self, primary: BaseException) -> None:
        self._state = "POISONED"
        backend, self._highs = self._highs, None
        if backend is None:
            return
        try:
            status = backend.clear()
            if status != highspy.HighsStatus.kOk:
                cleanup = HighsOwnerCleanupError("HiGHS clear returned non-kOk")
                _add_secondary_type_note(primary, cleanup)
        except BaseException as cleanup:
            _add_secondary_type_note(primary, cleanup)

    def _construct_and_configure(self) -> None:
        if self._constructed or self._highs is not None:
            raise HighsOwnerUnknown("a second HiGHS backend is forbidden")
        self._remaining()
        backend = highspy.Highs()
        self._constructed = True
        self._highs = backend
        self._remaining()
        if backend.version() != _HIGHS_VERSION or backend.githash() != _HIGHS_GITHASH:
            raise HighsOwnerUnknown("highspy/HiGHS version differs from the pin")
        for name, value in _PINNED_OPTIONS:
            self._require_ok(backend.setOptionValue(name, value), f"set {name}")
            status, observed = backend.getOptionValue(name)
            self._require_ok(status, f"get {name}")
            if observed != value:
                raise HighsOwnerUnknown("a pinned HiGHS option did not round-trip")
        self._require_ok(
            backend.changeObjectiveSense(highspy.ObjSense.kMinimize),
            "set objective sense",
        )
        sense_status, sense = backend.getObjectiveSense()
        self._require_ok(sense_status, "get objective sense")
        if sense != highspy.ObjSense.kMinimize:
            raise HighsOwnerUnknown("HiGHS objective sense is not minimize")
        self._remaining()

    def _load(self, columns: _Columns, rows: FrozenRows) -> None:
        rows.assert_intact()
        if (
            rows.columns != columns.cost.size
            or not np.array_equal(rows.column_lower, columns.lower)
            or not np.array_equal(rows.column_upper, columns.upper)
        ):
            raise HighsOwnerUnknown("CSR tiny projection is not bound to this column box")
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")

        self._remaining()
        empty_starts = np.zeros(columns.cost.size + 1, dtype=np.int32)
        self._require_ok(
            backend.addCols(
                int(columns.cost.size),
                columns.cost,
                columns.lower,
                columns.upper,
                0,
                empty_starts,
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            ),
            "addCols",
        )
        self._remaining()
        lp = backend.getLp()
        if not (
            np.array_equal(np.asarray(lp.col_cost_, dtype=np.float64), columns.cost)
            and np.array_equal(
                np.asarray(lp.col_lower_, dtype=np.float64), columns.lower
            )
            and np.array_equal(
                np.asarray(lp.col_upper_, dtype=np.float64), columns.upper
            )
        ):
            raise HighsOwnerUnknown("HiGHS changed objective or column bounds")

        self._remaining()
        self._require_ok(
            backend.addRows(
                rows.rows,
                rows.lower,
                rows.upper,
                int(rows.data.size),
                rows.indptr,
                rows.indices,
                rows.data,
            ),
            "addRows",
        )
        self._remaining()
        if (
            backend.getNumCol() != rows.columns
            or backend.getNumRow() != rows.rows
            or backend.getNumNz() != int(rows.data.size)
        ):
            raise HighsOwnerUnknown("native row/column/nnz postcondition failed")
        lp = backend.getLp()
        if not (
            np.array_equal(np.asarray(lp.row_lower_, dtype=np.float64), rows.lower)
            and np.array_equal(np.asarray(lp.row_upper_, dtype=np.float64), rows.upper)
        ):
            raise HighsOwnerUnknown("HiGHS changed outward row bounds")

    @staticmethod
    def _validate_primal_frame(
        factors: np.ndarray,
        columns: _Columns,
        row_value: np.ndarray,
        row_lower: np.ndarray,
        row_upper: np.ndarray,
    ) -> None:
        if (
            factors.shape != columns.cost.shape
            or row_value.shape != row_lower.shape
            or row_lower.shape != row_upper.shape
            or not np.all(np.isfinite(factors))
            or not np.all(np.isfinite(row_value))
            or np.any(factors < columns.lower - _PRIMAL_TOLERANCE)
            or np.any(factors > columns.upper + _PRIMAL_TOLERANCE)
        ):
            raise HighsOwnerUnknown("optimal primal readback is malformed")
        finite_lower = np.isfinite(row_lower)
        finite_upper = np.isfinite(row_upper)
        if (
            np.any(
                row_value[finite_lower]
                < row_lower[finite_lower] - _PRIMAL_TOLERANCE
            )
            or np.any(
                row_value[finite_upper]
                > row_upper[finite_upper] + _PRIMAL_TOLERANCE
            )
        ):
            raise HighsOwnerUnknown("optimal primal violates a loaded row")

    def _run(self) -> Any:
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")
        remaining = self._remaining()
        self._require_ok(backend.setOptionValue("time_limit", remaining), "set time_limit")
        status, observed = backend.getOptionValue("time_limit")
        self._require_ok(status, "get time_limit")
        if observed != remaining:
            raise HighsOwnerUnknown("HiGHS time_limit did not round-trip")
        self._remaining()
        self._require_ok(backend.run(), "run")
        self._remaining()
        status = backend.getModelStatus()
        self._remaining()
        return status

    def _read_objective(self) -> float:
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")
        self._remaining()
        objective = float(backend.getObjectiveValue())
        self._remaining()
        if not math.isfinite(objective):
            raise HighsOwnerUnknown("optimal objective is nonfinite")
        return objective

    def _read_base_optimal(
        self, columns: _Columns, rows: FrozenRows
    ) -> OptimalSelector:
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")
        self._remaining()
        solution = backend.getSolution()
        self._remaining()
        if solution.value_valid is not True or solution.dual_valid is not True:
            raise HighsOwnerUnknown("base optimal solution flags are invalid")
        factors = _readonly_f64(solution.col_value)
        row_value = _readonly_f64(solution.row_value)
        row_dual = _readonly_f64(solution.row_dual)
        self._validate_primal_frame(
            factors, columns, row_value, rows.lower, rows.upper
        )
        if (
            row_dual.shape != (rows.rows,)
            or not np.all(np.isfinite(row_dual))
            or (rows.upper_only and np.any(row_dual > _DUAL_TOLERANCE))
        ):
            raise HighsOwnerUnknown("base row dual is malformed for minimization")
        return OptimalSelector(
            factors=factors,
            row_value=row_value,
            row_dual=row_dual,
            row_ids=_readonly_i64(rows.row_ids),
            minimized_objective=self._read_objective(),
        )

    def _read_base_ray(self, rows: FrozenRows) -> InfeasibleRaySelector:
        if not rows.upper_only or self._base_ray_requested:
            raise HighsOwnerUnknown("base dual ray requires one upper-only request")
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")
        self._remaining()
        exist_status, exists = backend.getDualRayExist()
        self._base_ray_requested = True
        self._remaining()
        self._require_ok(exist_status, "getDualRayExist")
        if exists is not True:
            raise HighsOwnerUnknown("infeasible base model reports no dual ray")
        self._remaining()
        ray_status, has_ray, raw_ray = backend.getDualRay()
        self._remaining()
        self._require_ok(ray_status, "getDualRay")
        if has_ray is not True:
            raise HighsOwnerUnknown("getDualRay did not return a ray")
        ray = _readonly_f64(raw_ray)
        if (
            ray.shape != (rows.rows,)
            or not np.all(np.isfinite(ray))
            or np.any(ray > 0.0)
            or not np.any(ray != 0.0)
        ):
            raise HighsOwnerUnknown("upper-row dual ray is malformed")
        row_ids = _readonly_i64(rows.row_ids)
        return InfeasibleRaySelector(
            row_ray=ray,
            row_ids=row_ids,
            support_row_ids=tuple(
                int(row_ids[index]) for index in np.flatnonzero(ray != 0.0)
            ),
        )

    def _read_update_optimal(
        self,
        columns: _Columns,
        row_lower: np.ndarray,
        row_upper: np.ndarray,
    ) -> OptimalCandidate:
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")
        self._remaining()
        solution = backend.getSolution()
        self._remaining()
        if solution.value_valid is not True:
            raise HighsOwnerUnknown("updated optimal primal flag is invalid")
        factors = _readonly_f64(solution.col_value)
        row_value = _readonly_f64(solution.row_value)
        self._validate_primal_frame(
            factors, columns, row_value, row_lower, row_upper
        )
        return OptimalCandidate(
            factors=factors,
            minimized_objective=self._read_objective(),
        )

    def _append_columns_and_read_back(
        self,
        *,
        base_columns: _Columns,
        base_rows: FrozenRows,
        new_columns: FrozenNewColumns,
        projected: _ProjectedNewColumns,
    ) -> _Columns:
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")
        full_columns = _freeze_columns(
            np.ascontiguousarray(
                np.concatenate((base_columns.cost, new_columns.cost)),
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                np.concatenate((base_columns.lower, new_columns.lower)),
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                np.concatenate((base_columns.upper, new_columns.upper)),
                dtype=np.float64,
            ),
        )
        new_slice = slice(base_columns.cost.size, full_columns.cost.size)
        sealed_new_cost = full_columns.cost[new_slice]
        sealed_new_lower = full_columns.lower[new_slice]
        sealed_new_upper = full_columns.upper[new_slice]
        self._remaining()
        self._require_ok(
            backend.addCols(
                new_columns.columns,
                sealed_new_cost,
                sealed_new_lower,
                sealed_new_upper,
                int(projected.data.size),
                projected.indptr,
                projected.indices,
                projected.data,
            ),
            "incremental addCols",
        )
        self._remaining()
        expected_nnz = int(base_rows.data.size + projected.data.size)
        if (
            backend.getNumCol() != full_columns.cost.size
            or backend.getNumRow() != base_rows.rows
            or backend.getNumNz() != expected_nnz
        ):
            raise HighsOwnerUnknown("incremental addCols postcondition failed")

        native_indices = np.arange(
            base_columns.cost.size,
            full_columns.cost.size,
            dtype=np.int32,
        )
        self._remaining()
        (
            status,
            returned,
            observed_cost,
            observed_lower,
            observed_upper,
            observed_nnz,
        ) = backend.getCols(new_columns.columns, native_indices)
        self._remaining()
        self._require_ok(status, "incremental getCols")
        if not (
            int(returned) == new_columns.columns
            and int(observed_nnz) == int(projected.data.size)
            and np.array_equal(
                np.asarray(observed_cost, dtype=np.float64), sealed_new_cost
            )
            and np.array_equal(
                np.asarray(observed_lower, dtype=np.float64), sealed_new_lower
            )
            and np.array_equal(
                np.asarray(observed_upper, dtype=np.float64), sealed_new_upper
            )
        ):
            raise HighsOwnerUnknown("incremental new-column frame changed")
        self._remaining()
        entry_status, starts, row_indices, values = backend.getColsEntries(
            new_columns.columns, native_indices
        )
        self._remaining()
        self._require_ok(entry_status, "incremental getColsEntries")
        if not (
            np.array_equal(
                np.asarray(starts, dtype=np.int32), projected.indptr[:-1]
            )
            and np.array_equal(
                np.asarray(row_indices, dtype=np.int32), projected.indices
            )
            and np.array_equal(np.asarray(values, dtype=np.float64), projected.data)
        ):
            raise HighsOwnerUnknown("incremental new-column entries changed")
        return full_columns

    def _change_existing_bounds_and_read_back(
        self,
        *,
        base_rows: FrozenRows,
        projected: _ProjectedNewColumns,
        expected_nnz: int,
    ) -> None:
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")
        row_indices = np.arange(base_rows.rows, dtype=np.int32)
        self._remaining()
        self._require_ok(
            backend.changeRowsBounds(
                base_rows.rows,
                row_indices,
                projected.row_lower,
                projected.row_upper,
            ),
            "incremental changeRowsBounds",
        )
        self._remaining()
        status, returned, lower, upper, observed_nnz = backend.getRows(
            base_rows.rows, row_indices
        )
        self._remaining()
        self._require_ok(status, "incremental getRows")
        if not (
            int(returned) == base_rows.rows
            and int(observed_nnz) == expected_nnz
            and np.array_equal(
                np.asarray(lower, dtype=np.float64), projected.row_lower
            )
            and np.array_equal(
                np.asarray(upper, dtype=np.float64), projected.row_upper
            )
        ):
            raise HighsOwnerUnknown("incremental existing-row bounds changed")

    def _append_rows_and_read_back(
        self,
        *,
        full_columns: _Columns,
        base_rows: FrozenRows,
        appended_rows: FrozenRows,
        existing_nnz: int,
    ) -> None:
        backend = self._highs
        if backend is None:
            raise HighsOwnerUnknown("the single HiGHS backend is unavailable")
        self._remaining()
        self._require_ok(
            backend.addRows(
                appended_rows.rows,
                appended_rows.lower,
                appended_rows.upper,
                int(appended_rows.data.size),
                appended_rows.indptr,
                appended_rows.indices,
                appended_rows.data,
            ),
            "incremental addRows",
        )
        self._remaining()
        expected_rows = base_rows.rows + appended_rows.rows
        expected_nnz = existing_nnz + int(appended_rows.data.size)
        if (
            backend.getNumCol() != full_columns.cost.size
            or backend.getNumRow() != expected_rows
            or backend.getNumNz() != expected_nnz
        ):
            raise HighsOwnerUnknown("incremental addRows postcondition failed")
        native_indices = np.arange(
            base_rows.rows, expected_rows, dtype=np.int32
        )
        self._remaining()
        status, returned, lower, upper, observed_nnz = backend.getRows(
            appended_rows.rows, native_indices
        )
        self._remaining()
        self._require_ok(status, "incremental appended getRows")
        if not (
            int(returned) == appended_rows.rows
            and int(observed_nnz) == int(appended_rows.data.size)
            and np.array_equal(
                np.asarray(lower, dtype=np.float64), appended_rows.lower
            )
            and np.array_equal(
                np.asarray(upper, dtype=np.float64), appended_rows.upper
            )
        ):
            raise HighsOwnerUnknown("incremental appended-row bounds changed")
        self._remaining()
        entry_status, starts, columns, values = backend.getRowsEntries(
            appended_rows.rows, native_indices
        )
        self._remaining()
        self._require_ok(entry_status, "incremental getRowsEntries")
        if not (
            np.array_equal(
                np.asarray(starts, dtype=np.int32), appended_rows.indptr[:-1]
            )
            and np.array_equal(
                np.asarray(columns, dtype=np.int32), appended_rows.indices
            )
            and np.array_equal(
                np.asarray(values, dtype=np.float64), appended_rows.data
            )
        ):
            raise HighsOwnerUnknown("incremental appended-row entries changed")

    def solve_base(
        self,
        *,
        cost: np.ndarray,
        column_lower: np.ndarray,
        column_upper: np.ndarray,
        rows: FrozenRows,
    ) -> BaseResult:
        self._guard_live("READY_BASE")
        try:
            columns = _freeze_columns(cost, column_lower, column_upper)
            rows.assert_intact()
            self._state = "BASE_LOADING"
            self._construct_and_configure()
            self._load(columns, rows)
            status = self._run()
            if status == highspy.HighsModelStatus.kOptimal:
                result: BaseResult = self._read_base_optimal(columns, rows)
                self._base_columns = columns
                self._base_rows = rows
                self._state = "BASE_SOLVED"
                return result
            if status == highspy.HighsModelStatus.kInfeasible:
                result = self._read_base_ray(rows)
                self._base_columns = columns
                self._base_rows = rows
                self._state = "BASE_SOLVED"
                return result
            self._state = "BASE_FAILED"
            return Unresolved(model_status=status)
        except BaseException as primary:
            self._poison_and_discard(primary)
            raise

    def apply_incremental_update(
        self,
        *,
        new_columns: FrozenNewColumns,
        existing_row_lower: np.ndarray,
        existing_row_upper: np.ndarray,
        appended_rows: FrozenRows,
    ) -> UpdateResult:
        """Apply the only allowed in-place mutation and warm re-solve once.

        ``existing_row_lower`` and ``existing_row_upper`` are the caller's
        complete logical updated bounds in the sealed base-row order.  They
        must already include every side-switch/RHS compensation belonging to
        the original base-``x`` coefficients, including any tiny base-``x``
        terms omitted when that updated logical row was formed.  This owner
        only adds the outward compensation for tiny coefficients in
        ``new_columns``.  ``appended_rows`` independently carries its own
        complete tiny projection over the expanded column box.

        All inputs are checked before ``addCols``.  The native sequence is
        fixed: append columns, replace every existing row bound, append one
        row batch, then warm-solve.  Any failure discards the owner; there is
        no rollback, full reload, second ray, or alternate update mode.
        """

        self._guard_live("BASE_SOLVED")
        try:
            base_columns = self._base_columns
            base_rows = self._base_rows
            if (
                base_columns is None
                or base_rows is None
                or new_columns.rows != base_rows.rows
                or not np.array_equal(new_columns.row_ids, base_rows.row_ids)
            ):
                raise HighsOwnerUnknown("incremental update is not bound to base rows")
            base_rows.assert_intact()
            new_columns.assert_intact()
            appended_rows.assert_intact()
            projected = _project_new_column_tiny(
                new_columns, existing_row_lower, existing_row_upper
            )
            combined_lower = np.ascontiguousarray(
                np.concatenate((base_columns.lower, new_columns.lower)),
                dtype=np.float64,
            )
            combined_upper = np.ascontiguousarray(
                np.concatenate((base_columns.upper, new_columns.upper)),
                dtype=np.float64,
            )
            if (
                appended_rows.columns != combined_lower.size
                or not np.array_equal(appended_rows.column_lower, combined_lower)
                or not np.array_equal(appended_rows.column_upper, combined_upper)
                or np.intersect1d(
                    base_rows.row_ids, appended_rows.row_ids, assume_unique=True
                ).size
            ):
                raise HighsOwnerUnknown(
                    "appended rows are not sealed to the expanded column/base-row frame"
                )

            # All validation and outward projection above completes before the
            # first native mutation.  From here onward there is one fixed
            # transaction and no rollback or alternate path.
            self._state = "UPDATE_MUTATING"
            columns = self._append_columns_and_read_back(
                base_columns=base_columns,
                base_rows=base_rows,
                new_columns=new_columns,
                projected=projected,
            )
            existing_nnz = int(base_rows.data.size + projected.data.size)
            self._change_existing_bounds_and_read_back(
                base_rows=base_rows,
                projected=projected,
                expected_nnz=existing_nnz,
            )
            self._append_rows_and_read_back(
                full_columns=columns,
                base_rows=base_rows,
                appended_rows=appended_rows,
                existing_nnz=existing_nnz,
            )
            status = self._run()
            full_row_lower = np.ascontiguousarray(
                np.concatenate((projected.row_lower, appended_rows.lower)),
                dtype=np.float64,
            )
            full_row_upper = np.ascontiguousarray(
                np.concatenate((projected.row_upper, appended_rows.upper)),
                dtype=np.float64,
            )
            self._state = "UPDATE_SOLVED"
            if status == highspy.HighsModelStatus.kOptimal:
                return self._read_update_optimal(
                    columns, full_row_lower, full_row_upper
                )
            # Updated infeasibility and every resource status are UNKNOWN.  In
            # particular, no second dual ray is requested.
            return Unresolved(model_status=status)
        except BaseException as primary:
            self._poison_and_discard(primary)
            raise

    def close(self) -> None:
        if self._state == "CLOSED":
            return
        backend, self._highs = self._highs, None
        self._state = "CLOSED"
        if backend is None:
            return
        try:
            status = backend.clear()
        except BaseException:
            raise
        if status != highspy.HighsStatus.kOk:
            raise HighsOwnerCleanupError("HiGHS clear returned non-kOk")
