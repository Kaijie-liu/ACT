"""Pure-CPU builder for one phase-projection incremental repair.

This private module performs arithmetic assembly only.  It neither constructs
nor calls a solver.  Starting from the exact logical full phase rows and the
already frozen/screened base rows, it appends one complete, topologically
ordered set of signed correction variables.  The existing ``x`` block is
never stacked, copied, or filtered again.

All arithmetic here is candidate-only.  A caller must still use the separate
raw-BOX, zero-width outward forward, and stored-binary64 Fraction terminal
before making any verifier claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.phase_projection_highs_owner import (
    FrozenNewColumns,
    FrozenRows,
    HighsOwnerUnknown,
)


__all__: tuple[str, ...] = ()


_SMALL_MATRIX_VALUE = 1.0e-12
_LARGE_MATRIX_VALUE = 1.0e15
_INFINITE_BOUND = 1.0e20
_F64_EPS = np.finfo(np.float64).eps
_F64_TINY = np.finfo(np.float64).tiny
_SCREEN_ROW_BATCH = 4096
_SCREEN_MAX_DENSE_ELEMENTS = 1_000_000
_MAX_PHASE_ROWS = 200_000
_MAX_INPUT_COLUMNS = 200_000
_MAX_SELECTED_PHASES = 200_000
_MAX_LOGICAL_NNZ = 200_000_000
_MAX_DENSE_DELTA_ELEMENTS = 200_000_000
_MAX_AUTHORITY_SEAL_BYTES = 2_000_000_000


class IncrementalRepairUnknown(RuntimeError):
    """The sole incremental candidate path must stop without a claim."""


@dataclass(frozen=True)
class IncrementalRepairPlan:
    """Owned inputs for the owner's one allowed in-place update.

    ``existing_row_lower`` already contains the exact-dyadic projection of
    deleted base-``x`` terms for every side-switched row.  Tiny entries in
    ``new_columns`` deliberately remain present so the owner applies their
    projection exactly once.  ``appended_rows`` projects its complete
    ``(x, y)`` rows independently.
    """

    new_columns: FrozenNewColumns
    existing_row_lower: np.ndarray
    existing_row_upper: np.ndarray
    appended_rows: FrozenRows
    auxiliary_lower: np.ndarray
    auxiliary_upper: np.ndarray
    updated_active: np.ndarray
    selected_ordinals: np.ndarray
    selected_base_row_positions: np.ndarray
    missing_ordinals: np.ndarray
    definition_row_ids: np.ndarray
    objective_margin_delta: np.ndarray
    objective_minimize_cost: np.ndarray
    missing_rows_appended: int
    definition_rows_appended: int
    existing_x_block_reused: bool
    content_sha256: str

    def assert_intact(self) -> None:
        self.new_columns.assert_intact()
        self.appended_rows.assert_intact()
        arrays = (
            self.existing_row_lower,
            self.existing_row_upper,
            self.auxiliary_lower,
            self.auxiliary_upper,
            self.updated_active,
            self.selected_ordinals,
            self.selected_base_row_positions,
            self.missing_ordinals,
            self.definition_row_ids,
            self.objective_margin_delta,
            self.objective_minimize_cost,
        )
        if (
            any(type(value) is not np.ndarray for value in arrays)
            or any(value.flags.writeable or not value.flags.c_contiguous for value in arrays)
            or self.existing_row_lower.shape != self.existing_row_upper.shape
            or self.auxiliary_lower.shape != self.auxiliary_upper.shape
            or self.selected_ordinals.shape != self.auxiliary_lower.shape
            or self.selected_base_row_positions.shape != self.selected_ordinals.shape
            or self.definition_row_ids.shape != self.selected_ordinals.shape
            or self.objective_margin_delta.shape != self.selected_ordinals.shape
            or self.objective_minimize_cost.shape != self.selected_ordinals.shape
            or self.missing_rows_appended != self.missing_ordinals.size
            or self.definition_rows_appended != self.selected_ordinals.size
            or self.existing_x_block_reused is not True
        ):
            raise IncrementalRepairUnknown("incremental repair plan metadata changed")
        if _digest_arrays(*arrays) != self.content_sha256:
            raise IncrementalRepairUnknown("incremental repair plan content changed")


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise IncrementalRepairUnknown(
            f"incremental repair deadline expired at {stage}"
        )


def _immutable_array(value: Any, *, dtype: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    expected = np.dtype(dtype)
    if raw.dtype != expected:
        raise IncrementalRepairUnknown(f"{name} has the wrong dtype")
    try:
        copied = np.array(raw, dtype=expected, order="C", copy=True)
    except (MemoryError, TypeError, ValueError, OverflowError) as exc:
        raise IncrementalRepairUnknown(f"{name} could not be sealed") from exc
    owner = copied.tobytes(order="C")
    sealed = np.frombuffer(owner, dtype=expected).reshape(copied.shape)
    sealed.setflags(write=False)
    return sealed


def _vector(value: Any, *, dtype: Any, name: str) -> np.ndarray:
    sealed = _immutable_array(value, dtype=dtype, name=name)
    if sealed.ndim != 1:
        raise IncrementalRepairUnknown(f"{name} must be a vector")
    return sealed


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _digest_csr(matrix: sp.csr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(repr(matrix.shape).encode("ascii"))
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _bytes_sealed_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Take one direct immutable snapshot of an already canonical array."""

    if type(value) is not np.ndarray or not value.flags.c_contiguous:
        raise IncrementalRepairUnknown(f"{name} is not a contiguous ndarray")
    try:
        immutable = bytes(memoryview(value).cast("B"))
        sealed = np.frombuffer(immutable, dtype=value.dtype).reshape(value.shape)
    except (MemoryError, TypeError, ValueError, OverflowError) as exc:
        raise IncrementalRepairUnknown(f"{name} authority seal failed") from exc
    sealed.setflags(write=False)
    return sealed


def _csr_arrays_digest(
    shape: tuple[int, int],
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(repr(shape).encode("ascii"))
    for array in (indptr, indices, data):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _seal_logical_csr_authority(
    matrix: Any, *, deadline: float
) -> sp.csr_matrix:
    """Seal the complete logical CSR before any arithmetic reads from it."""

    if (
        not sp.isspmatrix_csr(matrix)
        or matrix.dtype != np.dtype(np.float64)
        or matrix.ndim != 2
        or matrix.shape[0] <= 0
        or matrix.shape[1] <= 0
        or type(matrix.indptr) is not np.ndarray
        or type(matrix.indices) is not np.ndarray
        or type(matrix.data) is not np.ndarray
        or matrix.indptr.dtype != np.dtype(np.int32)
        or matrix.indices.dtype != np.dtype(np.int32)
        or matrix.data.dtype != np.dtype(np.float64)
        or not matrix.indptr.flags.c_contiguous
        or not matrix.indices.flags.c_contiguous
        or not matrix.data.flags.c_contiguous
    ):
        raise IncrementalRepairUnknown(
            "full oriented rows must be a pre-existing float64 CSR frame"
        )
    shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    source_indptr = matrix.indptr
    source_indices = matrix.indices
    source_data = matrix.data
    total_bytes = (
        int(source_indptr.nbytes)
        + int(source_indices.nbytes)
        + int(source_data.nbytes)
    )
    if total_bytes < 0 or total_bytes > _MAX_AUTHORITY_SEAL_BYTES:
        raise IncrementalRepairUnknown("logical CSR authority seal exceeds byte cap")
    _check_deadline(deadline, "before logical CSR authority seal")
    before = _csr_arrays_digest(
        shape, source_indptr, source_indices, source_data
    )
    sealed_indptr = _bytes_sealed_array(
        source_indptr, name="logical CSR indptr"
    )
    sealed_indices = _bytes_sealed_array(
        source_indices, name="logical CSR indices"
    )
    sealed_data = _bytes_sealed_array(source_data, name="logical CSR data")
    sealed = sp.csr_matrix(
        (sealed_data, sealed_indices, sealed_indptr),
        shape=shape,
        dtype=np.float64,
        copy=False,
    )
    snapshot = _csr_arrays_digest(
        shape, sealed.indptr, sealed.indices, sealed.data
    )
    after = _csr_arrays_digest(
        shape, source_indptr, source_indices, source_data
    )
    if (
        matrix.shape != shape
        or matrix.indptr is not source_indptr
        or matrix.indices is not source_indices
        or matrix.data is not source_data
        or before != snapshot
        or before != after
    ):
        raise IncrementalRepairUnknown(
            "logical CSR alias changed while its authority snapshot was sealed"
        )
    _check_deadline(deadline, "after logical CSR authority seal")
    return _validate_logical_csr(sealed)


def _seal_delta_authority(
    value: Any,
    *,
    expected_shape: tuple[int, int],
    deadline: float,
) -> np.ndarray:
    """Seal the full delta once; later caller-alias ABA cannot enter the plan."""

    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.ndim != 2
        or not value.flags.c_contiguous
        or value.shape != expected_shape
        or int(value.nbytes) > _MAX_AUTHORITY_SEAL_BYTES
    ):
        raise IncrementalRepairUnknown(
            "full delta must be a bounded contiguous float64 matrix"
        )
    source = value
    shape = tuple(int(item) for item in source.shape)
    _check_deadline(deadline, "before full delta authority seal")
    before = _digest_arrays(source)
    sealed = _bytes_sealed_array(source, name="full delta")
    snapshot = _digest_arrays(sealed)
    after = _digest_arrays(source)
    if (
        value is not source
        or value.shape != shape
        or before != snapshot
        or before != after
    ):
        raise IncrementalRepairUnknown(
            "full delta alias changed while its authority snapshot was sealed"
        )
    _check_deadline(deadline, "after full delta authority seal")
    return sealed


def _validate_logical_csr(matrix: Any) -> sp.csr_matrix:
    if (
        not sp.isspmatrix_csr(matrix)
        or matrix.dtype != np.dtype(np.float64)
        or matrix.ndim != 2
        or matrix.shape[0] <= 0
        or matrix.shape[1] <= 0
        or type(matrix.indptr) is not np.ndarray
        or type(matrix.indices) is not np.ndarray
        or type(matrix.data) is not np.ndarray
        or matrix.indptr.dtype != np.dtype(np.int32)
        or matrix.indices.dtype != np.dtype(np.int32)
        or matrix.data.dtype != np.dtype(np.float64)
        or not matrix.indptr.flags.c_contiguous
        or not matrix.indices.flags.c_contiguous
        or not matrix.data.flags.c_contiguous
    ):
        raise IncrementalRepairUnknown(
            "full oriented rows must be a pre-existing canonical float64 CSR"
        )
    rows, columns = (int(matrix.shape[0]), int(matrix.shape[1]))
    nnz = int(matrix.data.size)
    if (
        rows > _MAX_PHASE_ROWS
        or columns > _MAX_INPUT_COLUMNS
        or nnz > _MAX_LOGICAL_NNZ
        or matrix.indptr.shape != (rows + 1,)
        or int(matrix.indptr[0]) != 0
        or int(matrix.indptr[-1]) != nnz
        or matrix.indices.shape != matrix.data.shape
        or np.any(matrix.indptr[1:] < matrix.indptr[:-1])
        or (nnz and np.any(matrix.indices < 0))
        or (nnz and np.any(matrix.indices >= columns))
        or (nnz and not np.all(np.isfinite(matrix.data)))
        or (nnz and np.any(matrix.data == 0.0))
        or (nnz and np.any(np.abs(matrix.data) >= _LARGE_MATRIX_VALUE))
    ):
        raise IncrementalRepairUnknown(
            "full oriented CSR exceeds the fixed resource/numeric frame"
        )
    for row in range(rows):
        start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
        if stop - start > 1 and np.any(
            matrix.indices[start + 1 : stop]
            <= matrix.indices[start : stop - 1]
        ):
            raise IncrementalRepairUnknown(
                "full oriented CSR is not duplicate-free and canonical"
            )
    return matrix


def _fraction_down(value: Fraction, *, name: str) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise IncrementalRepairUnknown(f"{name} overflowed")
    if Fraction.from_float(rounded) > value:
        rounded = float(np.nextafter(rounded, -np.inf))
    if not math.isfinite(rounded) or Fraction.from_float(rounded) > value:
        raise IncrementalRepairUnknown(f"{name} could not round outward")
    return rounded


def _fraction_up(value: Fraction, *, name: str) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise IncrementalRepairUnknown(f"{name} overflowed")
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    if not math.isfinite(rounded) or Fraction.from_float(rounded) < value:
        raise IncrementalRepairUnknown(f"{name} could not round outward")
    return rounded


def _validate_base_binding(
    logical: sp.csr_matrix,
    centers: np.ndarray,
    active: np.ndarray,
    keep: np.ndarray,
    full_row_ids: np.ndarray,
    base_rows: FrozenRows,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    deadline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_rows.assert_intact()
    kept = np.flatnonzero(keep).astype(np.int64, copy=False)
    if (
        kept.size == 0
        or base_rows.rows != kept.size
        or base_rows.columns != logical.shape[1]
        or not base_rows.upper_only
        or not np.array_equal(base_rows.row_ids, full_row_ids[kept])
        or not np.array_equal(base_rows.column_lower, x_lower)
        or not np.array_equal(base_rows.column_upper, x_upper)
    ):
        raise IncrementalRepairUnknown(
            "frozen base rows are not bound to the supplied full-row frame"
        )

    orientation = np.where(active, -1.0, 1.0)
    rhs = -orientation * centers
    if (
        not np.all(np.isfinite(rhs))
        or np.any(np.abs(rhs) >= _INFINITE_BOUND)
    ):
        raise IncrementalRepairUnknown("phase row rhs is outside the numeric frame")

    logical_nnz = 0
    deleted_tiny = 0
    for local, raw_ordinal in enumerate(kept):
        if local % 1024 == 0:
            _check_deadline(deadline, "base row binding")
        ordinal = int(raw_ordinal)
        full_start = int(logical.indptr[ordinal])
        full_stop = int(logical.indptr[ordinal + 1])
        coefficients = logical.data[full_start:full_stop]
        columns = logical.indices[full_start:full_stop]
        retained = np.abs(coefficients) > _SMALL_MATRIX_VALUE
        base_start = int(base_rows.indptr[local])
        base_stop = int(base_rows.indptr[local + 1])
        if not (
            np.array_equal(base_rows.indices[base_start:base_stop], columns[retained])
            and np.array_equal(base_rows.data[base_start:base_stop], coefficients[retained])
        ):
            raise IncrementalRepairUnknown(
                "frozen base x row differs from the supplied logical row"
            )
        logical_nnz += int(coefficients.size)
        tiny_positions = np.flatnonzero(~retained)
        deleted_tiny += int(tiny_positions.size)
        exact_upper = Fraction.from_float(float(rhs[ordinal]))
        for position in tiny_positions:
            column = int(columns[position])
            coefficient = Fraction.from_float(float(coefficients[position]))
            lo = coefficient * Fraction.from_float(float(x_lower[column]))
            hi = coefficient * Fraction.from_float(float(x_upper[column]))
            exact_upper -= min(lo, hi)
        expected_upper = _fraction_up(
            exact_upper, name="base upper tiny projection"
        )
        if float(base_rows.upper[local]) != expected_upper:
            raise IncrementalRepairUnknown(
                "frozen base upper bound is not the exact logical tiny projection"
            )
    if (
        base_rows.logical_nnz != logical_nnz
        or base_rows.deleted_tiny_nnz != deleted_tiny
    ):
        raise IncrementalRepairUnknown(
            "frozen base logical/tiny nnz accounting is not bound"
        )
    return kept, orientation, rhs


def _recursive_auxiliary_bounds(
    logical: sp.csr_matrix,
    centers: np.ndarray,
    active: np.ndarray,
    selected: np.ndarray,
    delta: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    deadline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = int(selected.size)
    lower = np.empty(width, dtype=np.float64)
    upper = np.empty(width, dtype=np.float64)
    signs = np.where(active[selected], -1.0, 1.0)
    for index, raw_ordinal in enumerate(selected):
        _check_deadline(deadline, "recursive auxiliary bounds")
        ordinal = int(raw_ordinal)
        exact_lower = Fraction.from_float(float(signs[index] * centers[ordinal]))
        exact_upper = exact_lower
        start, stop = int(logical.indptr[ordinal]), int(logical.indptr[ordinal + 1])
        # The supplied full row is already s_i*q_i, so it can be used directly.
        for column_value, coefficient_value in zip(
            logical.indices[start:stop], logical.data[start:stop]
        ):
            column = int(column_value)
            coefficient = Fraction.from_float(float(coefficient_value))
            lo = coefficient * Fraction.from_float(float(x_lower[column]))
            hi = coefficient * Fraction.from_float(float(x_upper[column]))
            exact_lower += min(lo, hi)
            exact_upper += max(lo, hi)
        # Only an inactive->active correction carries earlier delta columns.
        if signs[index] > 0.0:
            for prior in range(index):
                coefficient_value = float(delta[ordinal, prior])
                if coefficient_value == 0.0:
                    continue
                coefficient = Fraction.from_float(coefficient_value)
                lo = coefficient * Fraction.from_float(float(lower[prior]))
                hi = coefficient * Fraction.from_float(float(upper[prior]))
                exact_lower += min(lo, hi)
                exact_upper += max(lo, hi)
        lower[index] = _fraction_down(
            exact_lower, name="recursive auxiliary lower"
        )
        upper[index] = _fraction_up(
            exact_upper, name="recursive auxiliary upper"
        )
        if (
            lower[index] > upper[index]
            or abs(lower[index]) >= _INFINITE_BOUND
            or abs(upper[index]) >= _INFINITE_BOUND
        ):
            raise IncrementalRepairUnknown(
                "recursive auxiliary bound left the finite owner frame"
            )
    return lower, upper, signs


def _gamma(count: np.ndarray) -> np.ndarray:
    count = np.asarray(count, dtype=np.float64)
    product = count * _F64_EPS
    result = np.full(count.shape, np.inf, dtype=np.float64)
    valid = np.isfinite(product) & (product >= 0.0) & (product < 0.5)
    result[valid] = product[valid] / (1.0 - product[valid])
    return result


def _component_box_upper(
    raw: np.ndarray,
    raw_mass: np.ndarray,
    operation_count: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Inflate a stored-order binary64 box dot to an outward upper bound."""

    raw = np.asarray(raw, dtype=np.float64)
    raw_mass = np.asarray(raw_mass, dtype=np.float64)
    count = np.asarray(operation_count, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    result = np.full(raw.shape, np.inf, dtype=np.float64)
    zero = ~active
    result[zero] = 0.0
    valid = active & np.isfinite(raw) & np.isfinite(raw_mass) & (raw_mass >= 0.0)
    if not np.any(valid):
        return result
    gamma = _gamma(count)
    valid &= np.isfinite(gamma)
    if not np.any(valid):
        return result
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mass_upper = raw_mass / (1.0 - gamma)
        mass_upper = mass_upper + _F64_TINY * np.maximum(1.0, count)
        mass_upper = np.nextafter(mass_upper, np.inf)
        guard = gamma * mass_upper
        guard = guard / (1.0 - _gamma(np.full(count.shape, 4.0)))
        guard = guard + _F64_TINY * 4.0
        guard = np.nextafter(guard, np.inf)
        rounded = np.nextafter(raw + guard, np.inf)
    valid &= np.isfinite(mass_upper) & np.isfinite(guard) & np.isfinite(rounded)
    result[valid] = rounded[valid]
    return result


def _screen_missing_rows(
    logical: sp.csr_matrix,
    orientation: np.ndarray,
    keep: np.ndarray,
    rhs: np.ndarray,
    delta: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    deadline: float,
) -> np.ndarray:
    """Return every omitted row not proved strictly redundant.

    The screen is vectorized in fixed row batches.  It uses stored-order raw
    binary64 products plus an outward absolute-product/gamma error allowance.
    A nonfinite result or equality with the rhs is critical and is appended.
    """

    rows = int(logical.shape[0])
    max_abs_x = np.maximum(np.abs(x_lower), np.abs(x_upper))
    max_abs_y = np.maximum(np.abs(y_lower), np.abs(y_upper))
    append_parts: list[np.ndarray] = []
    row_batch = min(
        _SCREEN_ROW_BATCH,
        max(1, _SCREEN_MAX_DENSE_ELEMENTS // int(delta.shape[1])),
    )
    for first in range(0, rows, row_batch):
        _check_deadline(deadline, "missing-row outward screen")
        stop = min(rows, first + row_batch)
        local_keep = keep[first:stop]
        if np.all(local_keep):
            continue
        row_count = stop - first
        starts = logical.indptr[first:stop].astype(np.int64, copy=False)
        ends = logical.indptr[first + 1 : stop + 1].astype(np.int64, copy=False)
        nnz = ends - starts
        data_start = int(starts[0])
        data_stop = int(ends[-1])
        coefficients = logical.data[data_start:data_stop]
        columns = logical.indices[data_start:data_stop]
        relative = starts - data_start
        nonempty = nnz > 0
        x_raw = np.zeros(row_count, dtype=np.float64)
        x_mass = np.zeros(row_count, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if coefficients.size:
                endpoint = np.where(
                    coefficients >= 0.0, x_upper[columns], x_lower[columns]
                )
                products = coefficients * endpoint
                mass_products = np.abs(coefficients) * max_abs_x[columns]
                x_raw[nonempty] = np.add.reduceat(
                    products, relative[nonempty]
                )
                x_mass[nonempty] = np.add.reduceat(
                    mass_products, relative[nonempty]
                )
        x_upper_bound = _component_box_upper(
            x_raw,
            x_mass,
            2.0 * nnz.astype(np.float64) + 2.0,
            nonempty,
        )

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            y_coefficients = (
                orientation[first:stop, None] * delta[first:stop, :]
            )
            y_endpoint = np.where(
                y_coefficients >= 0.0, y_upper[None, :], y_lower[None, :]
            )
            y_products = y_coefficients * y_endpoint
            y_mass_products = np.abs(y_coefficients) * max_abs_y[None, :]
            y_raw = np.sum(y_products, axis=1, dtype=np.float64)
            y_mass = np.sum(y_mass_products, axis=1, dtype=np.float64)
        y_active = np.any(y_coefficients != 0.0, axis=1)
        y_upper_bound = _component_box_upper(
            y_raw,
            y_mass,
            np.full(row_count, 2.0 * delta.shape[1] + 2.0, dtype=np.float64),
            y_active,
        )

        combined_active = nonempty | y_active
        combined = np.full(row_count, np.inf, dtype=np.float64)
        combined[~combined_active] = 0.0
        valid = (
            combined_active
            & np.isfinite(x_upper_bound)
            & np.isfinite(y_upper_bound)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = np.nextafter(x_upper_bound + y_upper_bound, np.inf)
        valid &= np.isfinite(rounded)
        combined[valid] = rounded[valid]
        # Strictness is intentional: equality/roundoff-critical rows are kept.
        local_append = (~local_keep) & ~(
            np.isfinite(combined) & (combined < rhs[first:stop])
        )
        if np.any(local_append):
            append_parts.append(
                np.flatnonzero(local_append).astype(np.int64) + first
            )
    if not append_parts:
        return np.empty(0, dtype=np.int64)
    return np.ascontiguousarray(np.concatenate(append_parts), dtype=np.int64)


def _definition_row_ids(full_row_ids: np.ndarray, width: int) -> np.ndarray:
    i64 = np.iinfo(np.int64)
    maximum = int(np.max(full_row_ids))
    if maximum <= int(i64.max) - width:
        return np.arange(maximum + 1, maximum + 1 + width, dtype=np.int64)
    minimum = int(np.min(full_row_ids))
    if minimum >= int(i64.min) + width:
        return np.arange(minimum - width, minimum, dtype=np.int64)
    raise IncrementalRepairUnknown("no collision-free definition row-id range remains")


def _build_new_columns(
    kept: np.ndarray,
    orientation: np.ndarray,
    delta: np.ndarray,
    objective_delta: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    base_rows: FrozenRows,
    deadline: float,
) -> FrozenNewColumns:
    row_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    indptr64 = np.empty(delta.shape[1] + 1, dtype=np.int64)
    indptr64[0] = 0
    total = 0
    local_rows = np.arange(kept.size, dtype=np.int32)
    for column in range(delta.shape[1]):
        _check_deadline(deadline, "existing-row auxiliary CSC")
        values = orientation[kept] * delta[kept, column]
        nonzero = values != 0.0
        selected_rows = np.ascontiguousarray(local_rows[nonzero], dtype=np.int32)
        selected_data = np.ascontiguousarray(values[nonzero], dtype=np.float64)
        total += int(selected_data.size)
        if total > np.iinfo(np.int32).max or total > _MAX_LOGICAL_NNZ:
            raise IncrementalRepairUnknown(
                "existing-row auxiliary CSC exceeds the fixed resource cap"
            )
        row_parts.append(selected_rows)
        data_parts.append(selected_data)
        indptr64[column + 1] = total
    indices = (
        np.ascontiguousarray(np.concatenate(row_parts), dtype=np.int32)
        if total
        else np.empty(0, dtype=np.int32)
    )
    data = (
        np.ascontiguousarray(np.concatenate(data_parts), dtype=np.float64)
        if total
        else np.empty(0, dtype=np.float64)
    )
    indptr = np.ascontiguousarray(indptr64, dtype=np.int32)
    matrix = sp.csc_matrix(
        (data, indices, indptr), shape=(kept.size, delta.shape[1]), dtype=np.float64
    )
    return FrozenNewColumns.from_csc(
        matrix,
        cost=np.ascontiguousarray(-objective_delta, dtype=np.float64),
        column_lower=np.ascontiguousarray(y_lower, dtype=np.float64),
        column_upper=np.ascontiguousarray(y_upper, dtype=np.float64),
        existing_row_ids=base_rows.row_ids,
    )


def _append_row_parts(
    index_parts: list[np.ndarray],
    data_parts: list[np.ndarray],
    indptr: list[int],
    x_indices: np.ndarray,
    x_data: np.ndarray,
    y_indices: np.ndarray,
    y_data: np.ndarray,
) -> None:
    if x_data.size:
        index_parts.append(np.ascontiguousarray(x_indices, dtype=np.int32))
        data_parts.append(np.ascontiguousarray(x_data, dtype=np.float64))
    if y_data.size:
        index_parts.append(np.ascontiguousarray(y_indices, dtype=np.int32))
        data_parts.append(np.ascontiguousarray(y_data, dtype=np.float64))
    indptr.append(indptr[-1] + int(x_data.size) + int(y_data.size))
    if indptr[-1] > np.iinfo(np.int32).max or indptr[-1] > _MAX_LOGICAL_NNZ:
        raise IncrementalRepairUnknown("appended CSR exceeds the fixed nnz cap")


def _build_appended_rows(
    logical: sp.csr_matrix,
    centers: np.ndarray,
    orientation: np.ndarray,
    rhs: np.ndarray,
    selected: np.ndarray,
    signs: np.ndarray,
    missing: np.ndarray,
    delta: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    full_row_ids: np.ndarray,
    definition_ids: np.ndarray,
    deadline: float,
) -> FrozenRows:
    n_x = int(logical.shape[1])
    width = int(selected.size)
    index_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    indptr_values = [0]
    lower_values: list[float] = []
    upper_values: list[float] = []
    row_id_values: list[int] = []

    for raw_ordinal in missing:
        _check_deadline(deadline, "append missing phase rows")
        ordinal = int(raw_ordinal)
        start, stop = int(logical.indptr[ordinal]), int(logical.indptr[ordinal + 1])
        y_values = orientation[ordinal] * delta[ordinal, :]
        y_nonzero = y_values != 0.0
        _append_row_parts(
            index_parts,
            data_parts,
            indptr_values,
            logical.indices[start:stop],
            logical.data[start:stop],
            np.ascontiguousarray(
                n_x + np.flatnonzero(y_nonzero), dtype=np.int32
            ),
            y_values[y_nonzero],
        )
        lower_values.append(-np.inf)
        upper_values.append(float(rhs[ordinal]))
        row_id_values.append(int(full_row_ids[ordinal]))

    for index, raw_ordinal in enumerate(selected):
        _check_deadline(deadline, "append correction definitions")
        ordinal = int(raw_ordinal)
        start, stop = int(logical.indptr[ordinal]), int(logical.indptr[ordinal + 1])
        prior_values = np.empty(index, dtype=np.float64)
        if index:
            if signs[index] > 0.0:
                prior_values[:] = -delta[ordinal, :index]
            else:
                prior_values.fill(0.0)
        prior_nonzero = prior_values != 0.0
        y_indices = np.concatenate(
            (
                n_x + np.flatnonzero(prior_nonzero),
                np.asarray([n_x + index], dtype=np.int64),
            )
        )
        y_data = np.concatenate(
            (prior_values[prior_nonzero], np.asarray([1.0], dtype=np.float64))
        )
        _append_row_parts(
            index_parts,
            data_parts,
            indptr_values,
            logical.indices[start:stop],
            -logical.data[start:stop],
            np.ascontiguousarray(y_indices, dtype=np.int32),
            np.ascontiguousarray(y_data, dtype=np.float64),
        )
        definition_rhs = float(signs[index] * centers[ordinal])
        lower_values.append(definition_rhs)
        upper_values.append(definition_rhs)
        row_id_values.append(int(definition_ids[index]))

    total_nnz = int(indptr_values[-1])
    indices = (
        np.ascontiguousarray(np.concatenate(index_parts), dtype=np.int32)
        if total_nnz
        else np.empty(0, dtype=np.int32)
    )
    data = (
        np.ascontiguousarray(np.concatenate(data_parts), dtype=np.float64)
        if total_nnz
        else np.empty(0, dtype=np.float64)
    )
    indptr = np.ascontiguousarray(indptr_values, dtype=np.int32)
    matrix = sp.csr_matrix(
        (data, indices, indptr),
        shape=(missing.size + width, n_x + width),
        dtype=np.float64,
    )
    column_lower = np.ascontiguousarray(
        np.concatenate((x_lower, y_lower)), dtype=np.float64
    )
    column_upper = np.ascontiguousarray(
        np.concatenate((x_upper, y_upper)), dtype=np.float64
    )
    return FrozenRows.from_csr(
        matrix,
        row_lower=np.ascontiguousarray(lower_values, dtype=np.float64),
        row_upper=np.ascontiguousarray(upper_values, dtype=np.float64),
        row_ids=np.ascontiguousarray(row_id_values, dtype=np.int64),
        column_lower=column_lower,
        column_upper=column_upper,
    )


def _flipped_existing_bounds(
    logical: sp.csr_matrix,
    rhs: np.ndarray,
    selected: np.ndarray,
    kept: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    base_rows: FrozenRows,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_to_base = np.full(logical.shape[0], -1, dtype=np.int64)
    full_to_base[kept] = np.arange(kept.size, dtype=np.int64)
    positions = full_to_base[selected]
    if np.any(positions < 0):
        raise IncrementalRepairUnknown(
            "selector contains a phase row absent from the loaded base"
        )
    lower = np.full(base_rows.rows, -np.inf, dtype=np.float64)
    # Unflipped rows reuse the frozen base upper vector bit for bit.
    upper = np.array(base_rows.upper, dtype=np.float64, order="C", copy=True)
    for raw_ordinal, raw_position in zip(selected, positions):
        ordinal = int(raw_ordinal)
        position = int(raw_position)
        exact_lower = Fraction.from_float(float(rhs[ordinal]))
        start, stop = int(logical.indptr[ordinal]), int(logical.indptr[ordinal + 1])
        for column_value, coefficient_value in zip(
            logical.indices[start:stop], logical.data[start:stop]
        ):
            if abs(float(coefficient_value)) > _SMALL_MATRIX_VALUE:
                continue
            column = int(column_value)
            coefficient = Fraction.from_float(float(coefficient_value))
            lo = coefficient * Fraction.from_float(float(x_lower[column]))
            hi = coefficient * Fraction.from_float(float(x_upper[column]))
            exact_lower -= max(lo, hi)
        lower[position] = _fraction_down(
            exact_lower, name="side-switched base-x tiny projection"
        )
        upper[position] = np.inf
    return lower, upper, positions


def _build_incremental_repair(
    *,
    full_oriented_rows: sp.csr_matrix,
    phase_centers: np.ndarray,
    base_active: np.ndarray,
    keep: np.ndarray,
    full_row_ids: np.ndarray,
    base_rows: FrozenRows,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    selected_ordinals: np.ndarray,
    delta: np.ndarray,
    objective_delta: np.ndarray,
    deadline_monotonic: float,
) -> IncrementalRepairPlan:
    if type(deadline_monotonic) is not float or not math.isfinite(
        deadline_monotonic
    ):
        raise IncrementalRepairUnknown("deadline must be a finite monotonic float")
    deadline = deadline_monotonic
    _check_deadline(deadline, "entry")
    logical = _seal_logical_csr_authority(
        full_oriented_rows, deadline=deadline
    )
    logical_digest = _digest_csr(logical)

    centers = _vector(phase_centers, dtype=np.float64, name="phase centers")
    active = _vector(base_active, dtype=np.bool_, name="base active")
    kept_mask = _vector(keep, dtype=np.bool_, name="base keep mask")
    row_ids = _vector(full_row_ids, dtype=np.int64, name="full row ids")
    lower_x = _vector(x_lower, dtype=np.float64, name="x lower")
    upper_x = _vector(x_upper, dtype=np.float64, name="x upper")
    selected = _vector(
        selected_ordinals, dtype=np.int64, name="selected ordinals"
    )
    objective = _vector(
        objective_delta, dtype=np.float64, name="objective delta"
    )

    phase_rows, n_x = logical.shape
    width = int(selected.size)
    raw_delta = _seal_delta_authority(
        delta,
        expected_shape=(phase_rows, width),
        deadline=deadline,
    )
    delta_digest = _digest_arrays(raw_delta)
    if (
        centers.shape != (phase_rows,)
        or active.shape != (phase_rows,)
        or kept_mask.shape != (phase_rows,)
        or row_ids.shape != (phase_rows,)
        or lower_x.shape != (n_x,)
        or upper_x.shape != (n_x,)
        or objective.shape != (width,)
        or width <= 0
        or width > _MAX_SELECTED_PHASES
        or phase_rows * width > _MAX_DENSE_DELTA_ELEMENTS
        or np.unique(row_ids).size != phase_rows
        or np.any(selected < 0)
        or np.any(selected >= phase_rows)
        or (width > 1 and np.any(selected[1:] <= selected[:-1]))
        or not np.all(kept_mask[selected])
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(lower_x))
        or not np.all(np.isfinite(upper_x))
        or np.any(lower_x > upper_x)
        or np.any(np.abs(lower_x) >= _INFINITE_BOUND)
        or np.any(np.abs(upper_x) >= _INFINITE_BOUND)
        or not np.all(np.isfinite(raw_delta))
        or np.any(np.abs(raw_delta) >= _LARGE_MATRIX_VALUE)
        or not np.all(np.isfinite(objective))
        or np.any(np.abs(objective) >= _INFINITE_BOUND)
        or np.any(
            (objective != 0.0) & (np.abs(objective) <= _SMALL_MATRIX_VALUE)
        )
    ):
        raise IncrementalRepairUnknown(
            "incremental repair shapes, order, or numeric frame are invalid"
        )
    # An injected correction cannot affect its own or any preceding phase row.
    for index, raw_ordinal in enumerate(selected):
        if np.any(raw_delta[: int(raw_ordinal) + 1, index] != 0.0):
            raise IncrementalRepairUnknown(
                "full delta violates topological injection causality"
            )

    _check_deadline(deadline, "validated inputs")
    kept, orientation, rhs = _validate_base_binding(
        logical,
        centers,
        active,
        kept_mask,
        row_ids,
        base_rows,
        lower_x,
        upper_x,
        deadline,
    )
    auxiliary_lower, auxiliary_upper, signs = _recursive_auxiliary_bounds(
        logical,
        centers,
        active,
        selected,
        raw_delta,
        lower_x,
        upper_x,
        deadline,
    )
    missing = _screen_missing_rows(
        logical,
        orientation,
        kept_mask,
        rhs,
        raw_delta,
        lower_x,
        upper_x,
        auxiliary_lower,
        auxiliary_upper,
        deadline,
    )
    definitions = _definition_row_ids(row_ids, width)
    new_columns = _build_new_columns(
        kept,
        orientation,
        raw_delta,
        objective,
        auxiliary_lower,
        auxiliary_upper,
        base_rows,
        deadline,
    )
    existing_lower, existing_upper, selected_positions = (
        _flipped_existing_bounds(
            logical,
            rhs,
            selected,
            kept,
            lower_x,
            upper_x,
            base_rows,
        )
    )
    appended = _build_appended_rows(
        logical,
        centers,
        orientation,
        rhs,
        selected,
        signs,
        missing,
        raw_delta,
        lower_x,
        upper_x,
        auxiliary_lower,
        auxiliary_upper,
        row_ids,
        definitions,
        deadline,
    )
    updated_active = np.array(active, dtype=np.bool_, order="C", copy=True)
    updated_active[selected] = ~updated_active[selected]

    _check_deadline(deadline, "final ABA validation")
    base_rows.assert_intact()
    if _digest_csr(logical) != logical_digest or _digest_arrays(raw_delta) != delta_digest:
        raise IncrementalRepairUnknown(
            "sealed logical rows or full delta changed during incremental assembly"
        )

    frozen_existing_lower = _immutable_array(
        existing_lower, dtype=np.float64, name="existing row lower"
    )
    frozen_existing_upper = _immutable_array(
        existing_upper, dtype=np.float64, name="existing row upper"
    )
    frozen_aux_lower = _immutable_array(
        auxiliary_lower, dtype=np.float64, name="auxiliary lower"
    )
    frozen_aux_upper = _immutable_array(
        auxiliary_upper, dtype=np.float64, name="auxiliary upper"
    )
    frozen_updated = _immutable_array(
        updated_active, dtype=np.bool_, name="updated active"
    )
    frozen_selected = _immutable_array(
        selected, dtype=np.int64, name="selected ordinals output"
    )
    frozen_positions = _immutable_array(
        selected_positions, dtype=np.int64, name="selected base positions"
    )
    frozen_missing = _immutable_array(
        missing, dtype=np.int64, name="missing ordinals"
    )
    frozen_definitions = _immutable_array(
        definitions, dtype=np.int64, name="definition row ids"
    )
    frozen_objective = _immutable_array(
        objective, dtype=np.float64, name="objective margin delta"
    )
    frozen_cost = _immutable_array(
        -objective, dtype=np.float64, name="objective minimize cost"
    )
    content = _digest_arrays(
        frozen_existing_lower,
        frozen_existing_upper,
        frozen_aux_lower,
        frozen_aux_upper,
        frozen_updated,
        frozen_selected,
        frozen_positions,
        frozen_missing,
        frozen_definitions,
        frozen_objective,
        frozen_cost,
    )
    plan = IncrementalRepairPlan(
        new_columns=new_columns,
        existing_row_lower=frozen_existing_lower,
        existing_row_upper=frozen_existing_upper,
        appended_rows=appended,
        auxiliary_lower=frozen_aux_lower,
        auxiliary_upper=frozen_aux_upper,
        updated_active=frozen_updated,
        selected_ordinals=frozen_selected,
        selected_base_row_positions=frozen_positions,
        missing_ordinals=frozen_missing,
        definition_row_ids=frozen_definitions,
        objective_margin_delta=frozen_objective,
        objective_minimize_cost=frozen_cost,
        missing_rows_appended=int(frozen_missing.size),
        definition_rows_appended=int(width),
        existing_x_block_reused=True,
        content_sha256=content,
    )
    plan.assert_intact()
    _check_deadline(deadline, "return")
    return plan


def build_incremental_repair(
    *,
    full_oriented_rows: sp.csr_matrix,
    phase_centers: np.ndarray,
    base_active: np.ndarray,
    keep: np.ndarray,
    full_row_ids: np.ndarray,
    base_rows: FrozenRows,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    selected_ordinals: np.ndarray,
    delta: np.ndarray,
    objective_delta: np.ndarray,
    deadline_monotonic: float,
) -> IncrementalRepairPlan:
    """Build the one complete simultaneous phase repair, or fail closed."""

    try:
        return _build_incremental_repair(
            full_oriented_rows=full_oriented_rows,
            phase_centers=phase_centers,
            base_active=base_active,
            keep=keep,
            full_row_ids=full_row_ids,
            base_rows=base_rows,
            x_lower=x_lower,
            x_upper=x_upper,
            selected_ordinals=selected_ordinals,
            delta=delta,
            objective_delta=objective_delta,
            deadline_monotonic=deadline_monotonic,
        )
    except IncrementalRepairUnknown:
        raise
    except HighsOwnerUnknown as exc:
        raise IncrementalRepairUnknown(
            "owner input sealing rejected the incremental repair"
        ) from exc
    except (MemoryError, OverflowError, ValueError, IndexError) as exc:
        raise IncrementalRepairUnknown(
            "incremental repair exhausted its fixed arithmetic frame"
        ) from exc
