"""CPU-only scratch compiler for one direct sparse initial target.

The compiler owns one deterministic arithmetic schedule.  It calls no
optimization engine and publishes no intermediate state.  Every result
remains candidate data; this module grants no verification authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp


__all__ = (
    "DirectSparseInitialTargetInput",
    "DirectSparseInitialTargetReceipt",
    "DirectSparseInitialTargetResult",
    "DirectSparseInitialTargetUnknown",
    "compile_direct_sparse_initial_target",
)


_COLUMN_TILE_WIDTH = 64
_MAX_PHASE_ROWS = 200_000
_MAX_INPUT_FACTORS = 200_000
_MAX_CHANGES = 200_000
_MAX_FULL_LOGICAL_NNZ = 200_000_000
_MAX_COMPILE_BYTES = 2_000_000_000
_MAX_INT32 = 2_147_483_647
_FORMAT = "scratch-direct-sparse-initial-target-v1"


class DirectSparseInitialTargetUnknown(RuntimeError):
    """The sole scratch candidate must stop without a claim."""


@dataclass(frozen=True)
class DirectSparseInitialTargetInput:
    """Already ordered first-cell and first-to-target coefficient frames."""

    first_phase: np.ndarray
    initial_delta_phase: np.ndarray
    first_output: np.ndarray
    initial_delta_output: np.ndarray
    first_active: np.ndarray
    target_active: np.ndarray
    phase_centers: np.ndarray
    target_output_center: np.ndarray
    change_ordinals: np.ndarray
    input_rows: np.ndarray
    physical_rows: np.ndarray
    full_row_ids: np.ndarray
    factor_lower: np.ndarray
    factor_upper: np.ndarray
    assertion_matrix: np.ndarray
    thresholds: np.ndarray
    rival: int


@dataclass(frozen=True)
class DirectSparseInitialTargetReceipt:
    format: str
    phase_rows: int
    input_factors: int
    initial_changes: int
    output_rows: int
    segments: int
    fixed_column_tile_width: int
    chunking_used: bool
    partition_transaction_used: bool
    atomic_publish_count: int
    full_logical_nnz: int
    screened_logical_nnz: int
    compile_bytes: int
    worst_compile_bytes: int
    resource_estimate_kind: str
    loaded_nnz_after_tiny_projection: Optional[int]
    downstream_loading_run: bool
    snapshot_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class DirectSparseInitialTargetResult:
    full_rows: sp.csr_matrix
    rhs: np.ndarray
    phase_centers: np.ndarray
    target_active: np.ndarray
    physical_rows: np.ndarray
    full_row_ids: np.ndarray
    keep: np.ndarray
    screened_rows: sp.csr_matrix
    screened_rhs: np.ndarray
    screened_row_ids: np.ndarray
    objective_coefficient: np.ndarray
    objective_center: float
    receipt: DirectSparseInitialTargetReceipt

    def assert_intact(self) -> None:
        arrays = (
            self.rhs,
            self.phase_centers,
            self.target_active,
            self.physical_rows,
            self.full_row_ids,
            self.keep,
            self.screened_rhs,
            self.screened_row_ids,
            self.objective_coefficient,
        )
        if (
            type(self.full_rows) is not sp.csr_matrix
            or type(self.screened_rows) is not sp.csr_matrix
            or any(type(value) is not np.ndarray for value in arrays)
            or any(value.flags.writeable for value in arrays)
            or any(not value.flags.c_contiguous for value in arrays)
            or not _is_canonical(self.full_rows)
            or not _is_canonical(self.screened_rows)
            or self.full_rows.shape
            != (self.receipt.phase_rows, self.receipt.input_factors)
            or self.screened_rows.shape
            != (int(np.count_nonzero(self.keep)), self.receipt.input_factors)
            or self.full_rows.nnz != self.receipt.full_logical_nnz
            or self.screened_rows.nnz != self.receipt.screened_logical_nnz
        ):
            raise DirectSparseInitialTargetUnknown("published target metadata changed")
        for matrix in (self.full_rows, self.screened_rows):
            if any(
                value.flags.writeable
                for value in (matrix.indptr, matrix.indices, matrix.data)
            ):
                raise DirectSparseInitialTargetUnknown("published CSR became writeable")
        content = _content_digest(
            self.full_rows,
            self.rhs,
            self.phase_centers,
            self.target_active,
            self.physical_rows,
            self.full_row_ids,
            self.keep,
            self.screened_rows,
            self.screened_rhs,
            self.screened_row_ids,
            self.objective_coefficient,
            self.objective_center,
        )
        if content != self.receipt.content_sha256:
            raise DirectSparseInitialTargetUnknown("published target digest changed")


@dataclass(frozen=True)
class _Shape:
    phase_rows: int
    input_factors: int
    changes: int
    outputs: int
    assertions: int
    segments: int
    worst_nnz: int
    worst_bytes: int


@dataclass(frozen=True)
class _OwnedInput:
    first_phase: np.ndarray
    initial_delta_phase: np.ndarray
    first_output: np.ndarray
    initial_delta_output: np.ndarray
    first_active: np.ndarray
    target_active: np.ndarray
    phase_centers: np.ndarray
    target_output_center: np.ndarray
    change_ordinals: np.ndarray
    input_rows: np.ndarray
    physical_rows: np.ndarray
    full_row_ids: np.ndarray
    factor_lower: np.ndarray
    factor_upper: np.ndarray
    assertion_matrix: np.ndarray
    thresholds: np.ndarray
    rival: int
    snapshot_sha256: str


def _checked_product(left: int, right: int, *, name: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise DirectSparseInitialTargetUnknown(f"{name} has invalid factors")
    value = left * right
    if value > _MAX_INT32:
        raise DirectSparseInitialTargetUnknown(f"{name} exceeds int32")
    return value


def _compile_bytes(*, nnz: int, phase_rows: int, input_factors: int,
                   changes: int, outputs: int, segments: int) -> int:
    terms = (
        24 * nnz,
        4 * phase_rows * segments,
        8 * _COLUMN_TILE_WIDTH * (phase_rows + changes + outputs),
        4 * (phase_rows + 1),
        8 * input_factors,
    )
    value = sum(terms)
    if any(term < 0 for term in terms) or value > _MAX_COMPILE_BYTES:
        raise DirectSparseInitialTargetUnknown("compile resource bytes exceed cap")
    return value


def _require_array(value: Any, *, dtype: np.dtype[Any], ndim: int,
                   name: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.ndim != ndim
        or not value.flags.c_contiguous
    ):
        raise DirectSparseInitialTargetUnknown(
            f"{name} must be a contiguous {dtype} rank-{ndim} ndarray"
        )
    return value


def _shape_preflight(source: DirectSparseInitialTargetInput) -> _Shape:
    if type(source) is not DirectSparseInitialTargetInput:
        raise DirectSparseInitialTargetUnknown("source type is not exact")
    fp = _require_array(source.first_phase, dtype=np.dtype(np.float64), ndim=2,
                        name="first_phase")
    dp = _require_array(source.initial_delta_phase,
                        dtype=np.dtype(np.float64), ndim=2,
                        name="initial_delta_phase")
    fo = _require_array(source.first_output, dtype=np.dtype(np.float64), ndim=2,
                        name="first_output")
    do = _require_array(source.initial_delta_output,
                        dtype=np.dtype(np.float64), ndim=2,
                        name="initial_delta_output")
    P, n = map(int, fp.shape)
    k = int(dp.shape[1])
    o = int(fo.shape[0])
    C = _require_array(source.assertion_matrix,
                       dtype=np.dtype(np.float64), ndim=2,
                       name="assertion_matrix")
    m = int(C.shape[0])
    expected = (
        (dp.shape, (P, k), "initial_delta_phase"),
        (fo.shape, (o, n), "first_output"),
        (do.shape, (o, k), "initial_delta_output"),
        (C.shape, (m, o), "assertion_matrix"),
    )
    if any(actual != wanted for actual, wanted, _name in expected):
        bad = next(name for actual, wanted, name in expected if actual != wanted)
        raise DirectSparseInitialTargetUnknown(f"{bad} shape drifted")
    vector_specs = (
        (source.first_active, np.dtype(np.bool_), P, "first_active"),
        (source.target_active, np.dtype(np.bool_), P, "target_active"),
        (source.phase_centers, np.dtype(np.float64), P, "phase_centers"),
        (source.target_output_center, np.dtype(np.float64), o,
         "target_output_center"),
        (source.change_ordinals, np.dtype(np.int64), k, "change_ordinals"),
        (source.input_rows, np.dtype(np.int64), n, "input_rows"),
        (source.full_row_ids, np.dtype(np.int64), P, "full_row_ids"),
        (source.factor_lower, np.dtype(np.float64), n, "factor_lower"),
        (source.factor_upper, np.dtype(np.float64), n, "factor_upper"),
        (source.thresholds, np.dtype(np.float64), m, "thresholds"),
    )
    for value, dtype, length, name in vector_specs:
        array = _require_array(value, dtype=dtype, ndim=1, name=name)
        if array.shape != (length,):
            raise DirectSparseInitialTargetUnknown(f"{name} shape drifted")
    physical = _require_array(source.physical_rows, dtype=np.dtype(np.int64),
                              ndim=2, name="physical_rows")
    if physical.shape != (P, 3):
        raise DirectSparseInitialTargetUnknown("physical_rows shape drifted")
    if (
        P <= 0 or n <= 0 or o <= 0 or m <= 0 or k < 0
        or P > _MAX_PHASE_ROWS or n > _MAX_INPUT_FACTORS
        or k > _MAX_CHANGES
        or type(source.rival) is not int
        or source.rival < 0 or source.rival >= m
    ):
        raise DirectSparseInitialTargetUnknown("dimension or rival cap failed")
    worst_nnz = _checked_product(P, n, name="worst logical nnz")
    if worst_nnz > _MAX_FULL_LOGICAL_NNZ:
        raise DirectSparseInitialTargetUnknown("worst logical nnz exceeds cap")
    segments = (n + _COLUMN_TILE_WIDTH - 1) // _COLUMN_TILE_WIDTH
    worst_bytes = _compile_bytes(
        nnz=worst_nnz, phase_rows=P, input_factors=n, changes=k,
        outputs=o, segments=segments,
    )
    for value in (
        fp, dp, fo, do, source.first_active, source.target_active,
        source.phase_centers, source.target_output_center,
        source.change_ordinals, source.input_rows, physical,
        source.full_row_ids, source.factor_lower, source.factor_upper, C,
        source.thresholds,
    ):
        if int(value.nbytes) < 0 or int(value.nbytes) > _MAX_COMPILE_BYTES:
            raise DirectSparseInitialTargetUnknown("input snapshot exceeds byte cap")
    return _Shape(P, n, k, o, m, segments, worst_nnz, worst_bytes)


def _checkpoint(deadline: float, stage: str,
                callback: Optional[Callable[[str], None]]) -> None:
    if callback is not None:
        callback(stage)
    if time.monotonic() >= deadline:
        raise DirectSparseInitialTargetUnknown(f"deadline expired at {stage}")


def _seal(value: np.ndarray, *, name: str, deadline: float,
          callback: Optional[Callable[[str], None]]) -> np.ndarray:
    _checkpoint(deadline, f"before_snapshot_{name}", callback)
    before = _array_digest(value)
    try:
        immutable_bytes = (
            bytes(memoryview(value).cast("B")) if value.size else b""
        )
        result = np.frombuffer(immutable_bytes, dtype=value.dtype).reshape(
            value.shape
        )
    except (MemoryError, TypeError, ValueError, OverflowError) as exc:
        raise DirectSparseInitialTargetUnknown(f"{name} snapshot failed") from exc
    after = _array_digest(value)
    if before != _array_digest(result) or before != after:
        raise DirectSparseInitialTargetUnknown(f"{name} changed during snapshot")
    if result.flags.writeable or not result.flags.c_contiguous:
        raise DirectSparseInitialTargetUnknown(f"{name} snapshot is mutable")
    return result


def _array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    if value.size:
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _arrays_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(_array_digest(value).encode("ascii"))
    return digest.hexdigest()


def _input_digest(values: tuple[np.ndarray, ...], rival: int) -> str:
    digest = hashlib.sha256()
    digest.update(_arrays_digest(*values).encode("ascii"))
    digest.update(int(rival).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _owned_input(source: DirectSparseInitialTargetInput, shape: _Shape,
                 *, deadline: float,
                 callback: Optional[Callable[[str], None]]) -> _OwnedInput:
    names = (
        "first_phase", "initial_delta_phase", "first_output",
        "initial_delta_output", "first_active", "target_active",
        "phase_centers", "target_output_center", "change_ordinals",
        "input_rows", "physical_rows", "full_row_ids", "factor_lower",
        "factor_upper", "assertion_matrix", "thresholds",
    )
    sealed = tuple(
        _seal(getattr(source, name), name=name, deadline=deadline,
              callback=callback)
        for name in names
    )
    _checkpoint(deadline, "after_owned_snapshot", callback)
    snapshot_sha256 = _input_digest(sealed, source.rival)
    owned = _OwnedInput(*sealed, source.rival, snapshot_sha256)
    _validate_owned(owned, shape)
    _checkpoint(deadline, "after_snapshot_validation", callback)
    if _input_digest(sealed, source.rival) != owned.snapshot_sha256:
        raise DirectSparseInitialTargetUnknown("internal snapshot digest drifted")
    return owned


def _validate_owned(source: _OwnedInput, shape: _Shape) -> None:
    finite = (
        source.first_phase, source.initial_delta_phase, source.first_output,
        source.initial_delta_output, source.phase_centers,
        source.target_output_center, source.factor_lower, source.factor_upper,
        source.assertion_matrix, source.thresholds,
    )
    if any(not np.all(np.isfinite(value)) for value in finite):
        raise DirectSparseInitialTargetUnknown("input contains nonfinite values")
    if np.any(source.factor_lower > source.factor_upper):
        raise DirectSparseInitialTargetUnknown("factor BOX is malformed")
    if any(
        int(source.input_rows[index]) >= int(source.input_rows[index + 1])
        for index in range(shape.input_factors - 1)
    ):
        raise DirectSparseInitialTargetUnknown("input factor order drifted")
    if any(int(source.full_row_ids[index]) != index
           for index in range(shape.phase_rows)):
        raise DirectSparseInitialTargetUnknown("full row IDs drifted")
    previous_layer: Optional[int] = None
    previous_position = -1
    physical_keys: set[tuple[int, int]] = set()
    for layer, position, row in source.physical_rows:
        layer_i, position_i, row_i = int(layer), int(position), int(row)
        if (
            layer_i < 0 or position_i < 0 or row_i < 0
            or (previous_layer is not None and layer_i < previous_layer)
            or (previous_layer == layer_i and position_i != previous_position + 1)
            or (previous_layer != layer_i and position_i != 0)
            or (layer_i, row_i) in physical_keys
        ):
            raise DirectSparseInitialTargetUnknown("physical row order drifted")
        physical_keys.add((layer_i, row_i))
        previous_layer, previous_position = layer_i, position_i
    expected_changes = [
        index for index in range(shape.phase_rows)
        if bool(source.first_active[index]) != bool(source.target_active[index])
    ]
    actual_changes = [int(value) for value in source.change_ordinals]
    if actual_changes != expected_changes:
        raise DirectSparseInitialTargetUnknown("change order or membership drifted")
    for index, ordinal in enumerate(actual_changes):
        if np.any(source.initial_delta_phase[ordinal, index:] != 0.0):
            raise DirectSparseInitialTargetUnknown("future dependency is nonzero")


def _new_empty(shape: tuple[int, ...], dtype: Any, *, deadline: float,
               stage: str,
               callback: Optional[Callable[[str], None]]) -> np.ndarray:
    count = math.prod(shape)
    bytes_needed = count * np.dtype(dtype).itemsize
    if count < 0 or bytes_needed < 0 or bytes_needed > _MAX_COMPILE_BYTES:
        raise DirectSparseInitialTargetUnknown(f"{stage} allocation exceeds cap")
    _checkpoint(deadline, f"before_allocate_{stage}", callback)
    try:
        return np.empty(shape, dtype=dtype, order="C")
    except (MemoryError, ValueError, OverflowError) as exc:
        raise DirectSparseInitialTargetUnknown(f"{stage} allocation failed") from exc


def _new_zeros(shape: tuple[int, ...], dtype: Any, *, deadline: float,
               stage: str,
               callback: Optional[Callable[[str], None]]) -> np.ndarray:
    result = _new_empty(shape, dtype, deadline=deadline, stage=stage,
                        callback=callback)
    result.fill(0)
    return result


def _seal_generated(value: np.ndarray) -> np.ndarray:
    try:
        immutable_bytes = (
            bytes(memoryview(value).cast("B")) if value.size else b""
        )
        result = np.frombuffer(immutable_bytes,
                               dtype=value.dtype).reshape(value.shape)
    except (MemoryError, TypeError, ValueError, OverflowError) as exc:
        raise DirectSparseInitialTargetUnknown("generated snapshot failed") from exc
    if result.flags.writeable or not result.flags.c_contiguous:
        raise DirectSparseInitialTargetUnknown("generated snapshot is mutable")
    return result


def _is_canonical(matrix: sp.csr_matrix) -> bool:
    if (
        type(matrix) is not sp.csr_matrix
        or matrix.dtype != np.dtype(np.float64)
        or matrix.indptr.dtype != np.dtype(np.int32)
        or matrix.indices.dtype != np.dtype(np.int32)
        or not matrix.indptr.flags.c_contiguous
        or not matrix.indices.flags.c_contiguous
        or not matrix.data.flags.c_contiguous
        or not matrix.has_canonical_format
        or np.any(matrix.data == 0.0)
        or not np.all(np.isfinite(matrix.data))
    ):
        return False
    return all(
        np.all(matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]][1:]
               > matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]][:-1])
        for row in range(matrix.shape[0])
    )


def _fraction_row_upper(matrix: sp.csr_matrix, row: int,
                        lower: np.ndarray, upper: np.ndarray) -> Fraction:
    total = Fraction(0)
    for position in range(int(matrix.indptr[row]), int(matrix.indptr[row + 1])):
        coefficient = float(matrix.data[position])
        column = int(matrix.indices[position])
        bound = float(upper[column] if coefficient >= 0.0 else lower[column])
        total += Fraction.from_float(coefficient) * Fraction.from_float(bound)
    return total


def _content_digest(full_rows: sp.csr_matrix, rhs: np.ndarray,
                    phase_centers: np.ndarray, target_active: np.ndarray,
                    physical_rows: np.ndarray, full_row_ids: np.ndarray,
                    keep: np.ndarray, screened_rows: sp.csr_matrix,
                    screened_rhs: np.ndarray, screened_row_ids: np.ndarray,
                    objective_coefficient: np.ndarray,
                    objective_center: float) -> str:
    digest = hashlib.sha256()
    digest.update(_FORMAT.encode("ascii"))
    for matrix in (full_rows, screened_rows):
        digest.update(repr(matrix.shape).encode("ascii"))
        for value in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(_array_digest(value).encode("ascii"))
    for value in (
        rhs, phase_centers, target_active, physical_rows, full_row_ids, keep,
        screened_rhs, screened_row_ids, objective_coefficient,
    ):
        digest.update(_array_digest(value).encode("ascii"))
    digest.update(float(objective_center).hex().encode("ascii"))
    return digest.hexdigest()


def _compile_owned(source: _OwnedInput, shape: _Shape, *, deadline: float,
                   callback: Optional[Callable[[str], None]]) -> DirectSparseInitialTargetResult:
    row_counts_by_segment: list[np.ndarray] = []
    columns_by_segment: list[np.ndarray] = []
    data_by_segment: list[np.ndarray] = []
    objective = _new_empty((shape.input_factors,), np.float64,
                           deadline=deadline, stage="objective",
                           callback=callback)
    try:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise",
                             under="ignore"):
                for segment, c0 in enumerate(
                    range(0, shape.input_factors, _COLUMN_TILE_WIDTH)
                ):
                    _checkpoint(deadline, f"segment_{segment}_start", callback)
                    c1 = min(c0 + _COLUMN_TILE_WIDTH, shape.input_factors)
                    width = c1 - c0
                    U = _new_empty((shape.changes, width), np.float64,
                                   deadline=deadline,
                                   stage=f"segment_{segment}_expansion",
                                   callback=callback)
                    for index, ordinal_raw in enumerate(source.change_ordinals):
                        ordinal = int(ordinal_raw)
                        if bool(source.target_active[ordinal]):
                            U[index] = source.first_phase[ordinal, c0:c1]
                            if index:
                                U[index] += (
                                    source.initial_delta_phase[ordinal, :index]
                                    @ U[:index]
                                )
                        else:
                            U[index] = -source.first_phase[ordinal, c0:c1]
                        if not np.all(np.isfinite(U[index])):
                            raise DirectSparseInitialTargetUnknown(
                                "triangular arithmetic overflowed"
                            )
                    G = source.first_phase[:, c0:c1] + (
                        source.initial_delta_phase @ U
                    )
                    output_block = source.first_output[:, c0:c1] + (
                        source.initial_delta_output @ U
                    )
                    objective_block = (
                        source.assertion_matrix[[source.rival]] @ output_block
                    ).reshape(-1)
                    if not (
                        np.all(np.isfinite(G))
                        and np.all(np.isfinite(output_block))
                        and np.all(np.isfinite(objective_block))
                    ):
                        raise DirectSparseInitialTargetUnknown(
                            "target arithmetic overflowed"
                        )
                    objective[c0:c1] = objective_block
                    for row in range(shape.phase_rows):
                        if bool(source.target_active[row]):
                            G[row] *= -1.0
                    row_counts = _new_empty(
                        (shape.phase_rows,), np.int32, deadline=deadline,
                        stage=f"segment_{segment}_row_counts",
                        callback=callback,
                    )
                    segment_nnz = 0
                    for row in range(shape.phase_rows):
                        count = int(np.count_nonzero(G[row] != 0.0))
                        row_counts[row] = count
                        segment_nnz += count
                    if segment_nnz > _MAX_FULL_LOGICAL_NNZ:
                        raise DirectSparseInitialTargetUnknown(
                            "segment logical nnz exceeds cap"
                        )
                    columns = _new_empty(
                        (segment_nnz,), np.int32, deadline=deadline,
                        stage=f"segment_{segment}_columns", callback=callback,
                    )
                    values = _new_empty(
                        (segment_nnz,), np.float64, deadline=deadline,
                        stage=f"segment_{segment}_data", callback=callback,
                    )
                    position = 0
                    for row in range(shape.phase_rows):
                        local_columns = np.flatnonzero(G[row] != 0.0)
                        count = int(local_columns.size)
                        columns[position:position + count] = local_columns + c0
                        values[position:position + count] = G[row, local_columns]
                        position += count
                    if (
                        not np.all(np.isfinite(values))
                        or np.any(values == 0.0)
                        or columns.size != values.size
                        or int(row_counts.sum(dtype=np.int64)) != values.size
                        or position != segment_nnz
                    ):
                        raise DirectSparseInitialTargetUnknown(
                            "segment sparse record is invalid"
                        )
                    row_counts.setflags(write=False)
                    columns.setflags(write=False)
                    values.setflags(write=False)
                    row_counts_by_segment.append(row_counts)
                    columns_by_segment.append(columns)
                    data_by_segment.append(values)
                    _checkpoint(deadline, f"segment_{segment}_sealed", callback)
                    del U, G, output_block, objective_block, local_columns
        except FloatingPointError as exc:
            raise DirectSparseInitialTargetUnknown(
                "target arithmetic overflowed"
            ) from exc
        full_nnz = sum(int(value.size) for value in data_by_segment)
        if full_nnz > _MAX_FULL_LOGICAL_NNZ or full_nnz > _MAX_INT32:
            raise DirectSparseInitialTargetUnknown("actual logical nnz exceeds cap")
        actual_bytes = _compile_bytes(
            nnz=full_nnz, phase_rows=shape.phase_rows,
            input_factors=shape.input_factors, changes=shape.changes,
            outputs=shape.outputs, segments=shape.segments,
        )
        indptr = _new_zeros((shape.phase_rows + 1,), np.int32,
                            deadline=deadline, stage="full_indptr",
                            callback=callback)
        running = 0
        for row in range(shape.phase_rows):
            running += sum(
                int(counts[row]) for counts in row_counts_by_segment
            )
            if running > _MAX_INT32:
                raise DirectSparseInitialTargetUnknown("indptr exceeds int32")
            indptr[row + 1] = running
        if running != full_nnz:
            raise DirectSparseInitialTargetUnknown("logical nnz accounting drifted")
        indices = _new_empty((full_nnz,), np.int32, deadline=deadline,
                             stage="full_indices", callback=callback)
        data = _new_empty((full_nnz,), np.float64, deadline=deadline,
                          stage="full_data", callback=callback)
        write = 0
        read_offsets = [0 for _ in range(shape.segments)]
        for row in range(shape.phase_rows):
            for segment in range(shape.segments):
                begin = read_offsets[segment]
                width = int(row_counts_by_segment[segment][row])
                end = begin + width
                indices[write:write + width] = columns_by_segment[segment][begin:end]
                data[write:write + width] = data_by_segment[segment][begin:end]
                write += width
                read_offsets[segment] = end
        if write != full_nnz:
            raise DirectSparseInitialTargetUnknown("CSR fill count drifted")
        row_counts_by_segment.clear()
        columns_by_segment.clear()
        data_by_segment.clear()
        _checkpoint(deadline, "before_full_csr_publish", callback)
        sealed_indptr = _seal_generated(indptr)
        sealed_indices = _seal_generated(indices)
        sealed_data = _seal_generated(data)
        full_rows = sp.csr_matrix(
            (sealed_data, sealed_indices, sealed_indptr),
            shape=(shape.phase_rows, shape.input_factors), copy=False,
        )
        for value in (full_rows.indptr, full_rows.indices, full_rows.data):
            value.setflags(write=False)
        del indptr, indices, data, sealed_indptr, sealed_indices, sealed_data
        if not _is_canonical(full_rows):
            raise DirectSparseInitialTargetUnknown("full CSR is noncanonical")
        rhs = np.where(source.target_active, source.phase_centers,
                       -source.phase_centers).astype(np.float64, copy=False)
        if not np.all(np.isfinite(rhs)):
            raise DirectSparseInitialTargetUnknown("rhs overflowed")
        keep = _new_empty((shape.phase_rows,), np.bool_, deadline=deadline,
                          stage="exact_keep", callback=callback)
        for row in range(shape.phase_rows):
            _checkpoint(deadline, f"fraction_row_{row}", callback)
            exact_upper = _fraction_row_upper(
                full_rows, row, source.factor_lower, source.factor_upper
            )
            keep[row] = exact_upper > Fraction.from_float(float(rhs[row]))
        if not np.any(keep):
            raise DirectSparseInitialTargetUnknown(
                "exact screen retained no downstream rows"
            )
        _checkpoint(deadline, "before_screened_csr", callback)
        screened_rows = full_rows[keep]
        if type(screened_rows) is not sp.csr_matrix:
            raise DirectSparseInitialTargetUnknown("screened rows are not CSR")
        screened_rows.data = _seal_generated(screened_rows.data)
        screened_rows.indices = _seal_generated(screened_rows.indices)
        screened_rows.indptr = _seal_generated(screened_rows.indptr)
        for value in (
            screened_rows.indptr, screened_rows.indices, screened_rows.data
        ):
            value.setflags(write=False)
        if not _is_canonical(screened_rows):
            raise DirectSparseInitialTargetUnknown("screened CSR is noncanonical")
        screened_rhs = np.ascontiguousarray(rhs[keep], dtype=np.float64)
        screened_row_ids = np.ascontiguousarray(source.full_row_ids[keep],
                                                dtype=np.int64)
        objective_center = float(
            source.assertion_matrix[source.rival] @ source.target_output_center
            - source.thresholds[source.rival]
        )
        if not math.isfinite(objective_center) or not np.all(np.isfinite(objective)):
            raise DirectSparseInitialTargetUnknown("objective overflowed")
        generated = (
            rhs, source.phase_centers, source.target_active,
            source.physical_rows, source.full_row_ids, keep, screened_rhs,
            screened_row_ids, objective,
        )
        sealed_generated = tuple(_seal_generated(value) for value in generated)
        (
            rhs_s, centers_s, active_s, physical_s, row_ids_s, keep_s,
            screened_rhs_s, screened_ids_s, objective_s,
        ) = sealed_generated
        _checkpoint(deadline, "before_content_digest", callback)
        content = _content_digest(
            full_rows, rhs_s, centers_s, active_s, physical_s, row_ids_s,
            keep_s, screened_rows, screened_rhs_s, screened_ids_s,
            objective_s, objective_center,
        )
        receipt = DirectSparseInitialTargetReceipt(
            _FORMAT, shape.phase_rows, shape.input_factors, shape.changes,
            shape.outputs, shape.segments, _COLUMN_TILE_WIDTH, False, False, 1,
            full_nnz, int(screened_rows.nnz), actual_bytes, shape.worst_bytes,
            "contract_formula_not_observed_peak", None, False,
            source.snapshot_sha256, content,
        )
        result = DirectSparseInitialTargetResult(
            full_rows, rhs_s, centers_s, active_s, physical_s, row_ids_s,
            keep_s, screened_rows, screened_rhs_s, screened_ids_s,
            objective_s, objective_center, receipt,
        )
        result.assert_intact()
        _checkpoint(deadline, "before_atomic_publish", callback)
        return result
    finally:
        row_counts_by_segment.clear()
        columns_by_segment.clear()
        data_by_segment.clear()


def _compile_entry(
    source: DirectSparseInitialTargetInput,
    *,
    deadline_monotonic: float,
    checkpoint: Optional[Callable[[str], None]] = None,
) -> DirectSparseInitialTargetResult:
    _checkpoint(deadline_monotonic, "entry", checkpoint)
    shape = _shape_preflight(source)
    _checkpoint(deadline_monotonic, "after_resource_preflight", checkpoint)
    owned = _owned_input(source, shape, deadline=deadline_monotonic,
                         callback=checkpoint)
    return _compile_owned(owned, shape, deadline=deadline_monotonic,
                          callback=checkpoint)


def compile_direct_sparse_initial_target(
    source: DirectSparseInitialTargetInput,
    *,
    deadline_monotonic: float,
    checkpoint: Optional[Callable[[str], None]] = None,
) -> DirectSparseInitialTargetResult:
    """Compile and atomically return the sole canonical target bundle."""

    if (
        type(deadline_monotonic) is not float
        or not math.isfinite(deadline_monotonic)
        or deadline_monotonic <= 0.0
        or (checkpoint is not None and not callable(checkpoint))
    ):
        raise DirectSparseInitialTargetUnknown("deadline or checkpoint is invalid")
    try:
        return _compile_entry(
            source,
            deadline_monotonic=deadline_monotonic,
            checkpoint=checkpoint,
        )
    except BaseException as primary:
        captured = primary.__traceback__
        if captured is not None:
            traceback.clear_frames(captured)
        primary.__traceback__ = None
        raise primary.with_traceback(None)
