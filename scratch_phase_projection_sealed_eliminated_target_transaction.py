"""CPU-only scratch prototype for one sealed eliminated-target transaction.

This module deliberately is not a production candidate.  It proves a narrow
representation/lifecycle contract:

* the current binary64 triangular-elimination and phase-row arithmetic order;
* one retained, partitioned authority for every logical phase row, including
  rows omitted by the base screen;
* an ephemeral adapter into the existing :class:`FrozenRows` owner ABI;
* one exclusive repair handoff which borrows the same logical-row authority;
* bytes-backed snapshots, content revalidation, resource preflight, and
  deterministic release.

It does not run CUDA, choose phases, certify a terminal result, benchmark a
payment, create an auxiliary base normal form, or provide an alternate solver.
The current production repair builder still requires a global CSR and performs
another authority copy.  This prototype intentionally does not disguise that
remaining integration boundary with a second materialization path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import threading
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import phase_projection_highs_owner as _owner


__all__ = (
    "EliminatedTargetPartitionInput",
    "EliminatedTargetPartitionLayout",
    "EliminatedTargetResourceFrame",
    "RepairHandoff",
    "RepairPartitionView",
    "SealedCsrRows",
    "SealedEliminatedTargetTransaction",
    "SealedEliminatedTargetUnknown",
    "TransactionReceipt",
    "build_sealed_eliminated_target_transaction",
    "preflight_eliminated_target_resource_frame",
)


_SMALL_MATRIX_VALUE = 1.0e-12
_LARGE_MATRIX_VALUE = 1.0e15
_INFINITE_BOUND = 1.0e20
_MAX_PHASE_ROWS = 200_000
_MAX_INPUT_COLUMNS = 200_000
_MAX_CHANGES = 200_000
_MAX_LOGICAL_NNZ = 200_000_000
_MAX_DENSE_DELTA_ELEMENTS = 200_000_000
_MAX_AUTHORITY_BYTES = 2_000_000_000
_FORMAT_VERSION = "scratch-sealed-eliminated-target-v1"


class SealedEliminatedTargetUnknown(RuntimeError):
    """The scratch transaction must stop without an ability/performance claim."""


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _digest_text(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _require_array(
    value: Any,
    *,
    dtype: np.dtype[Any],
    ndim: int,
    name: str,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.ndim != ndim
        or not value.flags.c_contiguous
    ):
        raise SealedEliminatedTargetUnknown(
            f"{name} must be a contiguous {dtype} rank-{ndim} ndarray"
        )
    return value


def _seal_array(value: np.ndarray) -> np.ndarray:
    """Return a C-order view whose ultimate backing object is immutable bytes."""

    raw = value.tobytes(order="C")
    result = np.frombuffer(raw, dtype=value.dtype).reshape(value.shape)
    if result.flags.writeable or not result.flags.c_contiguous:
        raise SealedEliminatedTargetUnknown("bytes-backed array seal failed")
    return result


def _seal_caller_array(
    value: np.ndarray,
    *,
    name: str,
    expected_dtype: np.dtype[Any],
    expected_ndim: int,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Seal one caller alias and reject concurrent mutate/restore TOCTOU."""

    if (
        type(value) is not np.ndarray
        or value.dtype != expected_dtype
        or value.ndim != expected_ndim
        or not value.flags.c_contiguous
        or value.shape != expected_shape
        or int(value.nbytes) < 0
        or int(value.nbytes) > _MAX_AUTHORITY_BYTES
    ):
        raise SealedEliminatedTargetUnknown(
            f"{name} shape or authority bytes changed before seal"
        )
    before = _digest_arrays(value)
    result = _seal_array(value)
    snapshot = _digest_arrays(result)
    after = _digest_arrays(value)
    if (
        before != snapshot
        or snapshot != after
        or type(result) is not np.ndarray
        or result.dtype != expected_dtype
        or result.ndim != expected_ndim
        or not result.flags.c_contiguous
        or result.shape != expected_shape
    ):
        raise SealedEliminatedTargetUnknown(f"{name} changed during authority seal")
    return result


def _canonical(matrix: sp.spmatrix, *, name: str) -> sp.csr_matrix:
    try:
        result = matrix.tocsr()
        result.eliminate_zeros()
        result.sort_indices()
    except Exception as exc:
        raise SealedEliminatedTargetUnknown(f"{name} could not be canonicalized") from exc
    if (
        result.dtype != np.dtype(np.float64)
        or result.ndim != 2
        or not result.has_canonical_format
        or result.indptr.dtype != np.dtype(np.int32)
        or result.indices.dtype != np.dtype(np.int32)
        or not result.indptr.flags.c_contiguous
        or not result.indices.flags.c_contiguous
        or not result.data.flags.c_contiguous
        or np.any(result.data == 0.0)
    ):
        raise SealedEliminatedTargetUnknown(f"{name} is not canonical int32 CSR")
    return result


def _csr_box_upper(
    matrix: sp.csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """The current phase candidate's stored-binary64 screening order."""

    matrix = matrix.tocsr(copy=False)
    contribution = matrix.data * np.where(
        matrix.data >= 0.0,
        upper[matrix.indices],
        lower[matrix.indices],
    )
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    nonempty = np.diff(matrix.indptr) > 0
    if np.any(nonempty):
        result[nonempty] = np.add.reduceat(
            contribution, matrix.indptr[:-1][nonempty]
        )
    if not np.all(np.isfinite(result)):
        raise SealedEliminatedTargetUnknown("phase-row box screening overflowed")
    return result


@dataclass(frozen=True)
class EliminatedTargetPartitionInput:
    """One candidate layer partition to bind to an external frozen layout."""

    partition_id: int
    first_pre: np.ndarray
    delta_pre: np.ndarray
    phase_centers: np.ndarray
    target_active: np.ndarray
    row_ids: np.ndarray
    stream_rows: np.ndarray


@dataclass(frozen=True)
class EliminatedTargetPartitionLayout:
    """Immutable expected layer boundary/order from the upstream source seal."""

    partition_id: int
    row_ids: tuple[int, ...]
    stream_rows: tuple[int, ...]


@dataclass(frozen=True)
class EliminatedTargetResourceFrame:
    """Integer-only preflight frame; it allocates no matrix or dense program."""

    partition_rows: tuple[int, ...]
    input_columns: int
    initial_changes: int
    output_rows: int


@dataclass(frozen=True)
class _ResourceDecision:
    phase_rows: int
    dense_delta_elements: int
    worst_logical_nnz: int
    conservative_authority_bytes: int


def preflight_eliminated_target_resource_frame(
    frame: EliminatedTargetResourceFrame,
) -> _ResourceDecision:
    """Apply the one fixed scratch resource policy before input copies/BLAS."""

    if type(frame) is not EliminatedTargetResourceFrame:
        raise SealedEliminatedTargetUnknown("resource frame type is not exact")
    if type(frame.partition_rows) is not tuple:
        raise SealedEliminatedTargetUnknown("partition resource rows must be a tuple")
    values: tuple[Any, ...] = (
        *frame.partition_rows,
        frame.input_columns,
        frame.initial_changes,
        frame.output_rows,
    )
    if (
        not frame.partition_rows
        or any(type(value) is not int for value in values)
        or any(value <= 0 for value in frame.partition_rows)
        or frame.input_columns <= 0
        or frame.initial_changes <= 0
        or frame.output_rows <= 0
    ):
        raise SealedEliminatedTargetUnknown("resource frame is malformed")

    phase_rows = sum(frame.partition_rows)
    x_width = frame.input_columns
    changes = frame.initial_changes
    output_rows = frame.output_rows
    dense_delta = phase_rows * changes
    worst_nnz = phase_rows * x_width
    dense_input_values = (
        phase_rows * x_width
        + dense_delta
        + output_rows * x_width
        + output_rows * changes
        + changes * x_width
    )
    # Inputs/U plus the largest per-partition target work buffer, the retained
    # partition CSR authority, and the ephemeral base-owner adapter/copy.
    largest_partition = max(frame.partition_rows)
    conservative_bytes = (
        8 * dense_input_values
        + 8 * output_rows * x_width
        + 24 * largest_partition * x_width
        + 60 * worst_nnz
        + 64 * phase_rows
        + 32 * (len(frame.partition_rows) + x_width + changes + output_rows)
    )
    if (
        phase_rows > _MAX_PHASE_ROWS
        or x_width > _MAX_INPUT_COLUMNS
        or changes > _MAX_CHANGES
        or dense_delta > _MAX_DENSE_DELTA_ELEMENTS
        or worst_nnz > _MAX_LOGICAL_NNZ
        or worst_nnz > int(np.iinfo(np.int32).max)
        or conservative_bytes > _MAX_AUTHORITY_BYTES
    ):
        raise SealedEliminatedTargetUnknown(
            "sealed eliminated-target resource frame exceeds its fixed cap"
        )
    return _ResourceDecision(
        phase_rows=phase_rows,
        dense_delta_elements=dense_delta,
        worst_logical_nnz=worst_nnz,
        conservative_authority_bytes=conservative_bytes,
    )


def _seal_expected_layout(
    expected_layout: tuple[EliminatedTargetPartitionLayout, ...],
    *,
    source_binding_sha256: str,
) -> tuple[
    tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...],
    str,
]:
    if (
        type(expected_layout) is not tuple
        or not expected_layout
        or len(expected_layout) > _MAX_PHASE_ROWS
        or type(source_binding_sha256) is not str
        or len(source_binding_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_binding_sha256)
    ):
        raise SealedEliminatedTargetUnknown(
            "expected layout or source SHA-256 binding is malformed"
        )
    snapshot: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    partition_ids: set[int] = set()
    expected_global_row = 0
    for value in expected_layout:
        if type(value) is not EliminatedTargetPartitionLayout:
            raise SealedEliminatedTargetUnknown(
                "expected partition layout is malformed"
            )
        # Capture each frozen-dataclass field exactly once.  The local int and
        # exact tuples are immutable even if hostile code later bypasses the
        # dataclass guard with object.__setattr__.
        partition_id = value.partition_id
        raw_row_ids = value.row_ids
        raw_stream_rows = value.stream_rows
        if (
            type(partition_id) is not int
            or partition_id < 0
            or partition_id in partition_ids
            or type(raw_row_ids) is not tuple
            or type(raw_stream_rows) is not tuple
            or not raw_row_ids
            or len(raw_row_ids) != len(raw_stream_rows)
            or any(type(item) is not int for item in raw_row_ids)
            or any(type(item) is not int for item in raw_stream_rows)
        ):
            raise SealedEliminatedTargetUnknown(
                "expected partition layout is malformed"
            )
        row_ids = raw_row_ids
        stream_rows = raw_stream_rows
        if expected_global_row + len(row_ids) > _MAX_PHASE_ROWS:
            raise SealedEliminatedTargetUnknown(
                "expected partition layout exceeds the phase-row cap"
            )
        expected_ids = tuple(
            range(expected_global_row, expected_global_row + len(row_ids))
        )
        if (
            row_ids != expected_ids
            or any(item < 0 for item in stream_rows)
            or any(
                stream_rows[index + 1] <= stream_rows[index]
                for index in range(len(stream_rows) - 1)
            )
        ):
            raise SealedEliminatedTargetUnknown(
                "expected partition rows are not strictly ordered"
            )
        partition_ids.add(partition_id)
        snapshot.append((partition_id, row_ids, stream_rows))
        expected_global_row += len(row_ids)
    sealed = tuple(snapshot)
    # Revalidate only the immutable composite snapshot before hashing/return.
    sealed_partition_ids = tuple(item[0] for item in sealed)
    sealed_partition_ids_unique = (
        len(set(sealed_partition_ids)) == len(sealed_partition_ids)
    )
    sealed_global_row = 0
    for partition_id, row_ids, stream_rows in sealed:
        if (
            partition_id < 0
            or not sealed_partition_ids_unique
            or row_ids
            != tuple(range(sealed_global_row, sealed_global_row + len(row_ids)))
            or any(item < 0 for item in stream_rows)
            or any(
                stream_rows[index + 1] <= stream_rows[index]
                for index in range(len(stream_rows) - 1)
            )
        ):
            raise SealedEliminatedTargetUnknown(
                "sealed expected partition layout changed"
            )
        sealed_global_row += len(row_ids)
    layout_sha256 = _digest_text(source_binding_sha256, repr(sealed))
    return sealed, layout_sha256


def _preflight_repair_resources(
    *,
    resident_authority_bytes: int,
    phase_rows: int,
    output_rows: int,
    selected_width: int,
    partition_count: int,
) -> int:
    phase_delta_elements = phase_rows * selected_width
    output_delta_elements = output_rows * selected_width
    repair_authority_bytes = (
        8 * phase_delta_elements
        + 16 * output_delta_elements
        + 32 * selected_width
        + 128 * partition_count
        + 64 * phase_rows
    )
    if (
        phase_rows <= 0
        or type(resident_authority_bytes) is not int
        or resident_authority_bytes <= 0
        or output_rows <= 0
        or selected_width <= 0
        or partition_count <= 0
        or phase_delta_elements > _MAX_DENSE_DELTA_ELEMENTS
        or output_delta_elements > _MAX_DENSE_DELTA_ELEMENTS
        or resident_authority_bytes + repair_authority_bytes
        > _MAX_AUTHORITY_BYTES
    ):
        raise SealedEliminatedTargetUnknown(
            "repair authority exceeds its fixed resource cap"
        )
    return resident_authority_bytes + repair_authority_bytes


@dataclass(frozen=True)
class SealedCsrRows:
    """Bytes-backed canonical CSR which can issue disposable SciPy wrappers."""

    rows: int
    columns: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    content_sha256: str

    @classmethod
    def from_csr(cls, matrix: sp.csr_matrix, *, name: str) -> "SealedCsrRows":
        canonical = _canonical(matrix, name=name)
        if (
            canonical.shape[0] <= 0
            or canonical.shape[1] <= 0
            or canonical.nnz > _MAX_LOGICAL_NNZ
            or not np.all(np.isfinite(canonical.data))
            or np.any(np.abs(canonical.data) >= _LARGE_MATRIX_VALUE)
        ):
            raise SealedEliminatedTargetUnknown(
                f"{name} is outside the sealed numeric/resource frame"
            )
        indptr = _seal_array(np.ascontiguousarray(canonical.indptr, dtype=np.int32))
        indices = _seal_array(np.ascontiguousarray(canonical.indices, dtype=np.int32))
        data = _seal_array(np.ascontiguousarray(canonical.data, dtype=np.float64))
        return cls(
            rows=int(canonical.shape[0]),
            columns=int(canonical.shape[1]),
            indptr=indptr,
            indices=indices,
            data=data,
            content_sha256=_digest_arrays(indptr, indices, data),
        )

    def assert_intact(self) -> None:
        if (
            type(self.rows) is not int
            or type(self.columns) is not int
            or self.rows <= 0
            or self.columns <= 0
            or self.indptr.dtype != np.dtype(np.int32)
            or self.indices.dtype != np.dtype(np.int32)
            or self.data.dtype != np.dtype(np.float64)
            or self.indptr.shape != (self.rows + 1,)
            or self.indices.shape != self.data.shape
            or int(self.indptr[0]) != 0
            or int(self.indptr[-1]) != self.data.size
            or any(
                array.flags.writeable or not array.flags.c_contiguous
                for array in (self.indptr, self.indices, self.data)
            )
            or _digest_arrays(self.indptr, self.indices, self.data)
            != self.content_sha256
        ):
            raise SealedEliminatedTargetUnknown("sealed logical CSR changed")

    def as_csr(self) -> sp.csr_matrix:
        self.assert_intact()
        return sp.csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=(self.rows, self.columns),
            dtype=np.float64,
            copy=False,
        )


@dataclass(frozen=True)
class _SealedPartition:
    partition_id: int
    logical_rows: SealedCsrRows
    phase_centers: np.ndarray
    target_active: np.ndarray
    row_ids: np.ndarray
    stream_rows: np.ndarray
    rhs: np.ndarray
    keep: np.ndarray
    content_sha256: str

    def assert_intact(self) -> None:
        self.logical_rows.assert_intact()
        arrays = (
            self.phase_centers,
            self.target_active,
            self.row_ids,
            self.stream_rows,
            self.rhs,
            self.keep,
        )
        rows = self.logical_rows.rows
        if (
            type(self.partition_id) is not int
            or self.phase_centers.shape != (rows,)
            or self.target_active.shape != (rows,)
            or self.row_ids.shape != (rows,)
            or self.stream_rows.shape != (rows,)
            or self.rhs.shape != (rows,)
            or self.keep.shape != (rows,)
            or any(array.flags.writeable for array in arrays)
            or _digest_text(
                self.logical_rows.content_sha256,
                _digest_arrays(*arrays),
                str(self.partition_id),
            )
            != self.content_sha256
        ):
            raise SealedEliminatedTargetUnknown("sealed partition changed")


def _detached_csr_report(rows: SealedCsrRows) -> SealedCsrRows:
    """Copy both a CSR wrapper and its ndarray metadata for public reporting."""

    rows.assert_intact()
    indptr = _seal_array(rows.indptr)
    indices = _seal_array(rows.indices)
    data = _seal_array(rows.data)
    result = SealedCsrRows(
        rows=rows.rows,
        columns=rows.columns,
        indptr=indptr,
        indices=indices,
        data=data,
        content_sha256=_digest_arrays(indptr, indices, data),
    )
    result.assert_intact()
    return result


def _detached_partition_report(partition: _SealedPartition) -> _SealedPartition:
    """Return a diagnostic snapshot which shares no mutable Python metadata."""

    partition.assert_intact()
    logical_rows = _detached_csr_report(partition.logical_rows)
    phase_centers = _seal_array(partition.phase_centers)
    target_active = _seal_array(partition.target_active)
    row_ids = _seal_array(partition.row_ids)
    stream_rows = _seal_array(partition.stream_rows)
    rhs = _seal_array(partition.rhs)
    keep = _seal_array(partition.keep)
    result = _SealedPartition(
        partition_id=partition.partition_id,
        logical_rows=logical_rows,
        phase_centers=phase_centers,
        target_active=target_active,
        row_ids=row_ids,
        stream_rows=stream_rows,
        rhs=rhs,
        keep=keep,
        content_sha256=_digest_text(
            logical_rows.content_sha256,
            _digest_arrays(
                phase_centers,
                target_active,
                row_ids,
                stream_rows,
                rhs,
                keep,
            ),
            str(partition.partition_id),
        ),
    )
    result.assert_intact()
    return result


@dataclass(frozen=True)
class RepairPartitionView:
    """Borrowed repair view; logical rows are the original retained authority."""

    partition_id: int
    logical_rows: SealedCsrRows
    phase_centers: np.ndarray
    target_active: np.ndarray
    keep: np.ndarray
    row_ids: np.ndarray
    stream_rows: np.ndarray
    repair_delta: np.ndarray
    content_sha256: str

    def assert_intact(self) -> None:
        self.logical_rows.assert_intact()
        arrays = (
            self.phase_centers,
            self.target_active,
            self.keep,
            self.row_ids,
            self.stream_rows,
            self.repair_delta,
        )
        if (
            self.repair_delta.ndim != 2
            or self.repair_delta.shape[0] != self.logical_rows.rows
            or any(array.flags.writeable for array in arrays)
            or _digest_text(
                self.logical_rows.content_sha256,
                _digest_arrays(*arrays),
                str(self.partition_id),
            )
            != self.content_sha256
        ):
            raise SealedEliminatedTargetUnknown("repair partition view changed")


@dataclass(frozen=True)
class TransactionReceipt:
    format_version: str
    synthetic_only: bool
    source_binding_sha256: str
    layout_binding_sha256: str
    phase_rows: int
    input_columns: int
    initial_changes: int
    logical_nnz: int
    base_rows: int
    base_logical_nnz: int
    base_loaded_nnz: int
    base_deleted_tiny_nnz: int
    partition_count: int
    conservative_authority_bytes: int
    base_row_space: str
    row_id_rule: str
    partition_binding_rule: str
    base_owner_abi_target: str
    owner_lifecycle_externally_unproven: bool
    selector_authority: str
    terminal_authority: str
    retained_target_pre_container: bool
    retained_target_output: bool
    retained_global_logical_csr: bool
    retained_partitioned_full_logical_rows: bool
    retained_owner_screened_frozen_rows: bool
    retained_screened_scipy_wrapper: bool
    base_auxiliary_normal_form: bool
    repair_handoff_rule: str
    production_repair_global_csr_blocker: bool
    gpu_used: bool
    benchmark_run: bool
    transaction_sha256: str
    content_sha256: str

    def assert_intact(self) -> None:
        integer_values = (
            self.phase_rows,
            self.input_columns,
            self.initial_changes,
            self.logical_nnz,
            self.base_rows,
            self.base_logical_nnz,
            self.base_loaded_nnz,
            self.base_deleted_tiny_nnz,
            self.partition_count,
            self.conservative_authority_bytes,
        )
        boolean_values = (
            self.synthetic_only,
            self.owner_lifecycle_externally_unproven,
            self.retained_target_pre_container,
            self.retained_target_output,
            self.retained_global_logical_csr,
            self.retained_partitioned_full_logical_rows,
            self.retained_owner_screened_frozen_rows,
            self.retained_screened_scipy_wrapper,
            self.base_auxiliary_normal_form,
            self.production_repair_global_csr_blocker,
            self.gpu_used,
            self.benchmark_run,
        )
        string_values = (
            self.format_version,
            self.source_binding_sha256,
            self.layout_binding_sha256,
            self.base_row_space,
            self.row_id_rule,
            self.partition_binding_rule,
            self.base_owner_abi_target,
            self.selector_authority,
            self.terminal_authority,
            self.repair_handoff_rule,
            self.transaction_sha256,
            self.content_sha256,
        )
        if (
            type(self) is not TransactionReceipt
            or any(type(value) is not int or value < 0 for value in integer_values)
            or any(type(value) is not bool for value in boolean_values)
            or any(type(value) is not str for value in string_values)
            or len(self.source_binding_sha256) != 64
            or len(self.layout_binding_sha256) != 64
            or len(self.transaction_sha256) != 64
            or len(self.content_sha256) != 64
            or _transaction_receipt_digest(self) != self.content_sha256
        ):
            raise SealedEliminatedTargetUnknown("transaction receipt changed")


def _transaction_receipt_digest(receipt: TransactionReceipt) -> str:
    values = (
        ("format_version", receipt.format_version),
        ("synthetic_only", receipt.synthetic_only),
        ("source_binding_sha256", receipt.source_binding_sha256),
        ("layout_binding_sha256", receipt.layout_binding_sha256),
        ("phase_rows", receipt.phase_rows),
        ("input_columns", receipt.input_columns),
        ("initial_changes", receipt.initial_changes),
        ("logical_nnz", receipt.logical_nnz),
        ("base_rows", receipt.base_rows),
        ("base_logical_nnz", receipt.base_logical_nnz),
        ("base_loaded_nnz", receipt.base_loaded_nnz),
        ("base_deleted_tiny_nnz", receipt.base_deleted_tiny_nnz),
        ("partition_count", receipt.partition_count),
        ("conservative_authority_bytes", receipt.conservative_authority_bytes),
        ("base_row_space", receipt.base_row_space),
        ("row_id_rule", receipt.row_id_rule),
        ("partition_binding_rule", receipt.partition_binding_rule),
        ("base_owner_abi_target", receipt.base_owner_abi_target),
        (
            "owner_lifecycle_externally_unproven",
            receipt.owner_lifecycle_externally_unproven,
        ),
        ("selector_authority", receipt.selector_authority),
        ("terminal_authority", receipt.terminal_authority),
        ("retained_target_pre_container", receipt.retained_target_pre_container),
        ("retained_target_output", receipt.retained_target_output),
        ("retained_global_logical_csr", receipt.retained_global_logical_csr),
        (
            "retained_partitioned_full_logical_rows",
            receipt.retained_partitioned_full_logical_rows,
        ),
        (
            "retained_owner_screened_frozen_rows",
            receipt.retained_owner_screened_frozen_rows,
        ),
        (
            "retained_screened_scipy_wrapper",
            receipt.retained_screened_scipy_wrapper,
        ),
        ("base_auxiliary_normal_form", receipt.base_auxiliary_normal_form),
        ("repair_handoff_rule", receipt.repair_handoff_rule),
        (
            "production_repair_global_csr_blocker",
            receipt.production_repair_global_csr_blocker,
        ),
        ("gpu_used", receipt.gpu_used),
        ("benchmark_run", receipt.benchmark_run),
        ("transaction_sha256", receipt.transaction_sha256),
    )
    return _digest_text(*(f"{name}={value!r}" for name, value in values))


class RepairHandoff:
    """The transaction's sole, exclusive, one-shot repair lease."""

    def __init__(
        self,
        *,
        transaction: "SealedEliminatedTargetTransaction",
        views: tuple[RepairPartitionView, ...],
        selected_ordinals: np.ndarray,
        selected_base_row_positions: np.ndarray,
        objective_delta: np.ndarray,
        conservative_authority_bytes: int,
    ) -> None:
        self._transaction: SealedEliminatedTargetTransaction | None = transaction
        self._views: tuple[RepairPartitionView, ...] | None = views
        self._selected_ordinals: np.ndarray | None = selected_ordinals
        self._selected_base_row_positions: np.ndarray | None = (
            selected_base_row_positions
        )
        self._objective_delta: np.ndarray | None = objective_delta
        self._conservative_authority_bytes = conservative_authority_bytes
        self._content_sha256 = _digest_text(
            *[view.content_sha256 for view in views],
            _digest_arrays(
                selected_ordinals,
                selected_base_row_positions,
                objective_delta,
            ),
            str(conservative_authority_bytes),
        )
        self._release_lock = threading.RLock()
        self._state = "ACTIVE"

    @property
    def state(self) -> str:
        with self._release_lock:
            return self._state

    def _guard(self) -> None:
        if self._state != "ACTIVE":
            raise SealedEliminatedTargetUnknown("released repair handoff is unusable")

    @property
    def partitions(self) -> tuple[RepairPartitionView, ...]:
        with self._release_lock:
            self._guard()
            assert self._views is not None
            return self._views

    @property
    def selected_ordinals(self) -> np.ndarray:
        with self._release_lock:
            self._guard()
            assert self._selected_ordinals is not None
            return self._selected_ordinals

    @property
    def selected_base_row_positions(self) -> np.ndarray:
        with self._release_lock:
            self._guard()
            assert self._selected_base_row_positions is not None
            return self._selected_base_row_positions

    @property
    def objective_delta(self) -> np.ndarray:
        with self._release_lock:
            self._guard()
            assert self._objective_delta is not None
            return self._objective_delta

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    @property
    def conservative_authority_bytes(self) -> int:
        with self._release_lock:
            self._guard()
            return self._conservative_authority_bytes

    def assert_intact(self) -> None:
        with self._release_lock:
            self._guard()
            assert self._views is not None
            assert self._selected_ordinals is not None
            assert self._selected_base_row_positions is not None
            assert self._objective_delta is not None
            for view in self._views:
                view.assert_intact()
            if (
                type(self._conservative_authority_bytes) is not int
                or self._conservative_authority_bytes <= 0
                or self._conservative_authority_bytes > _MAX_AUTHORITY_BYTES
                or self._selected_ordinals.ndim != 1
                or self._selected_base_row_positions.shape
                != self._selected_ordinals.shape
                or self._objective_delta.shape != self._selected_ordinals.shape
                or any(
                    view.repair_delta.shape[1] != self._selected_ordinals.size
                    for view in self._views
                )
                or any(
                    value.flags.writeable
                    for value in (
                        self._selected_ordinals,
                        self._selected_base_row_positions,
                        self._objective_delta,
                    )
                )
                or _digest_text(
                    *[view.content_sha256 for view in self._views],
                    _digest_arrays(
                        self._selected_ordinals,
                        self._selected_base_row_positions,
                        self._objective_delta,
                    ),
                    str(self._conservative_authority_bytes),
                )
                != self._content_sha256
            ):
                raise SealedEliminatedTargetUnknown("repair handoff changed")

    def release(self) -> None:
        with self._release_lock:
            if self._state == "RELEASED":
                transaction = self._transaction
                if transaction is not None:
                    transaction._force_close_from_handoff(self)
                    self._transaction = None
                return
            transaction = self._transaction
            try:
                if transaction is not None:
                    # The transaction backing is closed before the borrowed
                    # handoff drops its final references.
                    transaction._repair_released(self)
            except BaseException:
                if transaction is not None:
                    try:
                        transaction._force_close_from_handoff(self)
                    except BaseException:
                        # Cleanup must not replace the hostile primary.
                        pass
                raise
            finally:
                self._views = None
                self._selected_ordinals = None
                self._selected_base_row_positions = None
                self._objective_delta = None
                self._transaction = None
                self._state = "RELEASED"

    def __enter__(self) -> "RepairHandoff":
        self.assert_intact()
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> bool:
        self.release()
        return False


class SealedEliminatedTargetTransaction:
    """One base snapshot followed by zero or one exclusive repair handoff."""

    def __init__(
        self,
        *,
        partitions: tuple[_SealedPartition, ...],
        expansion: np.ndarray,
        objective_coeff: np.ndarray,
        objective_center: float,
        property_row: np.ndarray,
        x_lower: np.ndarray,
        x_upper: np.ndarray,
        kept_row_ids: np.ndarray,
        repair_resource_frame: np.ndarray,
        base_rows: _owner.FrozenRows,
        source_binding_sha256: str,
        layout_binding_sha256: str,
        transaction_sha256: str,
        receipt: TransactionReceipt,
    ) -> None:
        self._lifecycle_lock = threading.RLock()
        self._partitions: tuple[_SealedPartition, ...] | None = partitions
        self._expansion: np.ndarray | None = expansion
        self._objective_coeff: np.ndarray | None = objective_coeff
        self._objective_center = objective_center
        self._property_row: np.ndarray | None = property_row
        self._x_lower: np.ndarray | None = x_lower
        self._x_upper: np.ndarray | None = x_upper
        self._kept_row_ids: np.ndarray | None = kept_row_ids
        self._repair_resource_frame: np.ndarray | None = repair_resource_frame
        self._base_rows: _owner.FrozenRows | None = base_rows
        self._source_binding_sha256 = source_binding_sha256
        self._layout_binding_sha256 = layout_binding_sha256
        self._transaction_sha256 = transaction_sha256
        self._receipt = receipt
        self._lease: RepairHandoff | None = None
        self._state = "OPEN"

    @property
    def state(self) -> str:
        with self._lifecycle_lock:
            return self._state

    def _guard_open(self) -> None:
        if self._state != "OPEN":
            raise SealedEliminatedTargetUnknown(
                "transaction is not open for base inspection"
            )

    @property
    def partitions(self) -> tuple[_SealedPartition, ...]:
        with self._lifecycle_lock:
            self._guard_open()
            assert self._partitions is not None
            # These are diagnostic reports, never the private repair authority.
            return tuple(
                _detached_partition_report(partition)
                for partition in self._partitions
            )

    @property
    def expansion(self) -> np.ndarray:
        with self._lifecycle_lock:
            self._guard_open()
            assert self._expansion is not None
            return self._expansion

    @property
    def objective_coeff(self) -> np.ndarray:
        with self._lifecycle_lock:
            self._guard_open()
            assert self._objective_coeff is not None
            return self._objective_coeff

    @property
    def objective_center(self) -> float:
        with self._lifecycle_lock:
            self._guard_open()
            return self._objective_center

    @property
    def base_rows(self) -> _owner.FrozenRows:
        with self._lifecycle_lock:
            self._guard_open()
            assert self._base_rows is not None
            return self._base_rows

    @property
    def receipt(self) -> TransactionReceipt:
        self._receipt.assert_intact()
        return self._receipt

    def assert_intact(self) -> None:
        with self._lifecycle_lock:
            self._guard_open()
            self._assert_authority_intact()

    def _assert_authority_intact(self) -> None:
        assert self._partitions is not None
        assert self._expansion is not None
        assert self._objective_coeff is not None
        assert self._property_row is not None
        assert self._x_lower is not None
        assert self._x_upper is not None
        assert self._kept_row_ids is not None
        assert self._repair_resource_frame is not None
        assert self._base_rows is not None
        self._receipt.assert_intact()
        for partition in self._partitions:
            partition.assert_intact()
        self._base_rows.assert_intact()
        if (
            any(
                value.flags.writeable
                for value in (
                    self._expansion,
                    self._objective_coeff,
                    self._property_row,
                    self._x_lower,
                    self._x_upper,
                    self._kept_row_ids,
                    self._repair_resource_frame,
                )
            )
            or self._repair_resource_frame.dtype != np.dtype(np.int64)
            or self._repair_resource_frame.shape != (5,)
            or np.any(self._repair_resource_frame <= 0)
            or not np.array_equal(self._base_rows.row_ids, self._kept_row_ids)
            or _digest_text(
                *[partition.content_sha256 for partition in self._partitions],
                _digest_arrays(
                    self._expansion,
                    self._objective_coeff,
                    self._property_row,
                    self._x_lower,
                    self._x_upper,
                    self._kept_row_ids,
                    self._repair_resource_frame,
                ),
                self._base_rows.content_sha256,
                float(self._objective_center).hex(),
                self._source_binding_sha256,
                self._layout_binding_sha256,
            )
            != self._transaction_sha256
            or self._receipt.transaction_sha256 != self._transaction_sha256
            or self._receipt.source_binding_sha256 != self._source_binding_sha256
            or self._receipt.layout_binding_sha256 != self._layout_binding_sha256
            or self._receipt.phase_rows != int(self._repair_resource_frame[0])
            or self._receipt.input_columns != int(self._repair_resource_frame[1])
            or self._receipt.partition_count != int(self._repair_resource_frame[3])
            or self._receipt.conservative_authority_bytes
            != int(self._repair_resource_frame[4])
        ):
            raise SealedEliminatedTargetUnknown("transaction authority changed")

    def begin_repair(
        self,
        *,
        selected_ordinals: np.ndarray,
        repair_delta_parts: tuple[np.ndarray, ...],
        repair_delta_output: np.ndarray,
    ) -> RepairHandoff:
        """Consume the only repair opportunity without rebuilding logical rows."""

        self._lifecycle_lock.acquire()
        try:
            self._guard_open()
        except BaseException:
            self._lifecycle_lock.release()
            raise
        try:
            self._assert_authority_intact()
            assert self._partitions is not None
            assert self._property_row is not None
            assert self._kept_row_ids is not None
            assert self._repair_resource_frame is not None
            assert self._base_rows is not None
            phase_rows = int(self._repair_resource_frame[0])
            output_rows = int(self._repair_resource_frame[2])
            partition_count = int(self._repair_resource_frame[3])
            resident_authority_bytes = int(self._repair_resource_frame[4])
            selected_source = _require_array(
                selected_ordinals,
                dtype=np.dtype(np.int64),
                ndim=1,
                name="selected ordinals",
            )
            if (
                type(repair_delta_parts) is not tuple
                or len(repair_delta_parts) != len(self._partitions)
            ):
                raise SealedEliminatedTargetUnknown("repair selection frame is invalid")
            width = int(selected_source.size)
            delta_output_source = _require_array(
                repair_delta_output,
                dtype=np.dtype(np.float64),
                ndim=2,
                name="repair output delta",
            )
            raw_delta_sources: list[np.ndarray] = []
            for partition, raw_delta in zip(self._partitions, repair_delta_parts):
                source = _require_array(
                    raw_delta,
                    dtype=np.dtype(np.float64),
                    ndim=2,
                    name=f"repair delta partition {partition.partition_id}",
                )
                if source.shape != (partition.logical_rows.rows, width):
                    raise SealedEliminatedTargetUnknown(
                        "repair partition delta shape is invalid"
                    )
                raw_delta_sources.append(source)
            if delta_output_source.shape != (self._property_row.size, width):
                raise SealedEliminatedTargetUnknown(
                    "repair output delta shape is invalid"
                )
            repair_authority_bytes = _preflight_repair_resources(
                resident_authority_bytes=resident_authority_bytes,
                phase_rows=phase_rows,
                output_rows=output_rows,
                selected_width=width,
                partition_count=partition_count,
            )

            selected = _seal_caller_array(
                selected_source,
                name="selected ordinals",
                expected_dtype=np.dtype(np.int64),
                expected_ndim=1,
                expected_shape=(width,),
            )
            if (
                selected.size <= 0
                or selected.size > _MAX_CHANGES
                or np.any(selected < 0)
                or np.any(selected >= phase_rows)
                or (
                    selected.size > 1
                    and np.any(selected[1:] <= selected[:-1])
                )
            ):
                raise SealedEliminatedTargetUnknown("repair selection frame is invalid")
            base_positions = np.searchsorted(self._kept_row_ids, selected)
            if (
                np.any(base_positions >= self._kept_row_ids.size)
                or not np.array_equal(self._kept_row_ids[base_positions], selected)
            ):
                raise SealedEliminatedTargetUnknown(
                    "repair selector named a row omitted from the base owner"
                )
            base_positions = _seal_array(
                np.ascontiguousarray(base_positions, dtype=np.int64)
            )

            sealed_delta_output = _seal_caller_array(
                delta_output_source,
                name="repair output delta",
                expected_dtype=np.dtype(np.float64),
                expected_ndim=2,
                expected_shape=(self._property_row.size, width),
            )
            if (
                sealed_delta_output.shape != (self._property_row.size, width)
                or not np.all(np.isfinite(sealed_delta_output))
                or np.any(np.abs(sealed_delta_output) >= _LARGE_MATRIX_VALUE)
            ):
                raise SealedEliminatedTargetUnknown(
                    "repair output delta left the numeric frame"
                )
            objective_delta = np.ascontiguousarray(
                np.asarray(
                    self._property_row[None, :] @ sealed_delta_output,
                    dtype=np.float64,
                ).reshape(-1),
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(objective_delta))
                or np.any(np.abs(objective_delta) >= _INFINITE_BOUND)
                or np.any(
                    (objective_delta != 0.0)
                    & (np.abs(objective_delta) <= _SMALL_MATRIX_VALUE)
                )
            ):
                raise SealedEliminatedTargetUnknown(
                    "repair objective delta left the owner frame"
                )
            objective_delta = _seal_array(objective_delta)

            views: list[RepairPartitionView] = []
            for partition, source in zip(self._partitions, raw_delta_sources):
                delta = _seal_caller_array(
                    source,
                    name=f"repair delta partition {partition.partition_id}",
                    expected_dtype=np.dtype(np.float64),
                    expected_ndim=2,
                    expected_shape=(partition.logical_rows.rows, width),
                )
                if (
                    delta.shape != (partition.logical_rows.rows, width)
                    or not np.all(np.isfinite(delta))
                    or np.any(np.abs(delta) >= _LARGE_MATRIX_VALUE)
                ):
                    raise SealedEliminatedTargetUnknown(
                        "repair partition delta left the numeric frame"
                    )
                for index, ordinal_value in enumerate(selected):
                    prefix = partition.row_ids <= int(ordinal_value)
                    if np.any(delta[prefix, index] != 0.0):
                        raise SealedEliminatedTargetUnknown(
                            "repair delta violates topological injection causality"
                        )
                view_digest = _digest_text(
                    partition.logical_rows.content_sha256,
                    _digest_arrays(
                        partition.phase_centers,
                        partition.target_active,
                        partition.keep,
                        partition.row_ids,
                        partition.stream_rows,
                        delta,
                    ),
                    str(partition.partition_id),
                )
                views.append(
                    RepairPartitionView(
                        partition_id=partition.partition_id,
                        logical_rows=partition.logical_rows,
                        phase_centers=partition.phase_centers,
                        target_active=partition.target_active,
                        keep=partition.keep,
                        row_ids=partition.row_ids,
                        stream_rows=partition.stream_rows,
                        repair_delta=delta,
                        content_sha256=view_digest,
                    )
                )
            self._assert_authority_intact()
            lease = RepairHandoff(
                transaction=self,
                views=tuple(views),
                selected_ordinals=selected,
                selected_base_row_positions=base_positions,
                objective_delta=objective_delta,
                conservative_authority_bytes=repair_authority_bytes,
            )
            lease.assert_intact()
            self._state = "REPAIR_LEASED"
            self._lease = lease
            return lease
        except BaseException:
            self._lease = None
            self._release_backing()
            raise
        finally:
            self._lifecycle_lock.release()

    def _repair_released(self, lease: RepairHandoff) -> None:
        with self._lifecycle_lock:
            if self._lease is not None and self._lease is not lease:
                self._lease = None
                self._release_backing()
                raise SealedEliminatedTargetUnknown("repair lease identity changed")
            self._lease = None
            self._release_backing()

    def _force_close_from_handoff(self, _lease: RepairHandoff) -> None:
        """Idempotent no-validation cleanup used only while releasing a lease."""

        with self._lifecycle_lock:
            self._lease = None
            self._release_backing()

    def _release_backing(self) -> None:
        self._partitions = None
        self._expansion = None
        self._objective_coeff = None
        self._property_row = None
        self._x_lower = None
        self._x_upper = None
        self._kept_row_ids = None
        self._repair_resource_frame = None
        self._base_rows = None
        self._state = "CLOSED"

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._state == "CLOSED":
                return
            lease = self._lease
            if lease is None:
                self._release_backing()
                return
        # Do not hold the transaction lock while taking the handoff lock: a
        # concurrent handoff.release() takes them in the opposite direction.
        try:
            lease.release()
        finally:
            with self._lifecycle_lock:
                if self._state != "CLOSED":
                    self._lease = None
                    self._release_backing()

    def __enter__(self) -> "SealedEliminatedTargetTransaction":
        self.assert_intact()
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> bool:
        self.close()
        return False


def _validate_and_seal_partitions(
    partitions: tuple[EliminatedTargetPartitionInput, ...],
    *,
    expected_layout: tuple[
        tuple[int, tuple[int, ...], tuple[int, ...]], ...
    ],
    input_columns: int,
    initial_changes: int,
) -> tuple[
    tuple[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        ...,
    ],
    int,
]:
    sealed: list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    expected_row = 0
    physical: set[tuple[int, int]] = set()
    if len(partitions) != len(expected_layout):
        raise SealedEliminatedTargetUnknown(
            "partition count differs from the frozen expected layout"
        )
    for index, (partition, layout) in enumerate(zip(partitions, expected_layout)):
        expected_partition_id, expected_ids_tuple, expected_stream_tuple = layout
        rows = len(expected_ids_tuple)
        if type(partition) is not EliminatedTargetPartitionInput:
            raise SealedEliminatedTargetUnknown("partition input type is not exact")
        if (
            type(partition.partition_id) is not int
            or partition.partition_id != expected_partition_id
        ):
            raise SealedEliminatedTargetUnknown(
                "partition id differs from the frozen expected layout"
            )
        first_source = _require_array(
            partition.first_pre,
            dtype=np.dtype(np.float64),
            ndim=2,
            name=f"partition {index} first preactivation",
        )
        delta_source = _require_array(
            partition.delta_pre,
            dtype=np.dtype(np.float64),
            ndim=2,
            name=f"partition {index} initial delta",
        )
        centers_source = _require_array(
            partition.phase_centers,
            dtype=np.dtype(np.float64),
            ndim=1,
            name=f"partition {index} phase centers",
        )
        active_source = _require_array(
            partition.target_active,
            dtype=np.dtype(np.bool_),
            ndim=1,
            name=f"partition {index} target active",
        )
        row_ids_source = _require_array(
            partition.row_ids,
            dtype=np.dtype(np.int64),
            ndim=1,
            name=f"partition {index} row ids",
        )
        stream_rows_source = _require_array(
            partition.stream_rows,
            dtype=np.dtype(np.int64),
            ndim=1,
            name=f"partition {index} stream rows",
        )
        # Every caller alias is sealed before any semantic validation.  All
        # validation below reads only these bytes-backed snapshots.
        first = _seal_caller_array(
            first_source,
            name=f"partition {index} first preactivation",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=2,
            expected_shape=(rows, input_columns),
        )
        delta = _seal_caller_array(
            delta_source,
            name=f"partition {index} initial delta",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=2,
            expected_shape=(rows, initial_changes),
        )
        centers = _seal_caller_array(
            centers_source,
            name=f"partition {index} phase centers",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=1,
            expected_shape=(rows,),
        )
        active = _seal_caller_array(
            active_source,
            name=f"partition {index} target active",
            expected_dtype=np.dtype(np.bool_),
            expected_ndim=1,
            expected_shape=(rows,),
        )
        row_ids = _seal_caller_array(
            row_ids_source,
            name=f"partition {index} row ids",
            expected_dtype=np.dtype(np.int64),
            expected_ndim=1,
            expected_shape=(rows,),
        )
        stream_rows = _seal_caller_array(
            stream_rows_source,
            name=f"partition {index} stream rows",
            expected_dtype=np.dtype(np.int64),
            expected_ndim=1,
            expected_shape=(rows,),
        )
        expected_ids = np.asarray(expected_ids_tuple, dtype=np.int64)
        expected_stream = np.asarray(expected_stream_tuple, dtype=np.int64)
        if (
            rows <= 0
            or first.shape != (rows, input_columns)
            or delta.shape != (rows, initial_changes)
            or centers.shape != (rows,)
            or active.shape != (rows,)
            or row_ids.shape != (rows,)
            or stream_rows.shape != (rows,)
            or not np.array_equal(row_ids, expected_ids)
            or not np.array_equal(stream_rows, expected_stream)
            or not np.array_equal(
                row_ids,
                np.arange(expected_row, expected_row + rows, dtype=np.int64),
            )
            or (rows > 1 and np.any(stream_rows[1:] <= stream_rows[:-1]))
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(delta))
            or not np.all(np.isfinite(centers))
            or np.any(np.abs(first) >= _LARGE_MATRIX_VALUE)
            or np.any(np.abs(delta) >= _LARGE_MATRIX_VALUE)
            or np.any(np.abs(centers) >= _INFINITE_BOUND)
        ):
            raise SealedEliminatedTargetUnknown(
                "partition shape, row order, or numeric frame is invalid"
            )
        for raw_stream_row in stream_rows:
            key = (expected_partition_id, int(raw_stream_row))
            if key in physical:
                raise SealedEliminatedTargetUnknown("physical phase row is duplicated")
            physical.add(key)
        sealed.append(
            (
                expected_partition_id,
                first,
                delta,
                centers,
                active,
                row_ids,
                stream_rows,
            )
        )
        expected_row += rows
    return tuple(sealed), expected_row


def _row_location(
    ordinal: int,
    sealed_inputs: Sequence[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> tuple[np.ndarray, np.ndarray, bool]:
    for _partition_id, first, delta, _centers, active, row_ids, _stream in sealed_inputs:
        start = int(row_ids[0])
        stop = start + int(row_ids.size)
        if start <= ordinal < stop:
            local = ordinal - start
            return first[local], delta[local], bool(active[local])
    raise SealedEliminatedTargetUnknown("phase-change row mapping is absent")


def _freeze_base_rows(
    partitions: tuple[_SealedPartition, ...],
    *,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> _owner.FrozenRows:
    """Use one short-lived adapter because the owner accepts one CSR batch."""

    adapter_parts: list[sp.csr_matrix] = []
    upper_parts: list[np.ndarray] = []
    row_id_parts: list[np.ndarray] = []
    for partition in partitions:
        logical = partition.logical_rows.as_csr()
        adapter_parts.append(logical[partition.keep].tocsr())
        upper_parts.append(np.ascontiguousarray(partition.rhs[partition.keep]))
        row_id_parts.append(np.ascontiguousarray(partition.row_ids[partition.keep]))
    adapter = _canonical(
        sp.vstack(adapter_parts, format="csr"),
        name="sealed eliminated-target owner adapter",
    )
    upper = np.ascontiguousarray(np.concatenate(upper_parts), dtype=np.float64)
    row_ids = np.ascontiguousarray(np.concatenate(row_id_parts), dtype=np.int64)
    if adapter.shape[0] <= 0:
        raise SealedEliminatedTargetUnknown("base screen retained no owner rows")
    result = _owner.FrozenRows.from_csr(
        adapter,
        row_lower=np.full(adapter.shape[0], -np.inf, dtype=np.float64),
        row_upper=upper,
        row_ids=row_ids,
        column_lower=np.ascontiguousarray(x_lower, dtype=np.float64),
        column_upper=np.ascontiguousarray(x_upper, dtype=np.float64),
    )
    result.assert_intact()
    # adapter/parts/upper/ids are intentionally not retained by the caller.
    return result


def build_sealed_eliminated_target_transaction(
    *,
    partitions: tuple[EliminatedTargetPartitionInput, ...],
    expected_layout: tuple[EliminatedTargetPartitionLayout, ...],
    source_binding_sha256: str,
    change_ordinals: np.ndarray,
    change_base_active: np.ndarray,
    change_target_active: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    first_output: np.ndarray,
    delta_output: np.ndarray,
    property_row: np.ndarray,
    target_output_center: np.ndarray,
    threshold: float,
) -> SealedEliminatedTargetTransaction:
    """Build the sole scratch transaction, with no runtime mode or cache."""

    try:
        if type(partitions) is not tuple or not partitions:
            raise SealedEliminatedTargetUnknown("partitions must be a nonempty tuple")
        layout, layout_binding_sha256 = _seal_expected_layout(
            expected_layout,
            source_binding_sha256=source_binding_sha256,
        )
        if len(partitions) != len(layout):
            raise SealedEliminatedTargetUnknown(
                "partitions differ from the frozen expected layout"
            )
        changes_source = _require_array(
            change_ordinals,
            dtype=np.dtype(np.int64),
            ndim=1,
            name="change ordinals",
        )
        base_source = _require_array(
            change_base_active,
            dtype=np.dtype(np.bool_),
            ndim=1,
            name="change base active",
        )
        target_source = _require_array(
            change_target_active,
            dtype=np.dtype(np.bool_),
            ndim=1,
            name="change target active",
        )
        lower_source = _require_array(
            x_lower, dtype=np.dtype(np.float64), ndim=1, name="x lower"
        )
        upper_source = _require_array(
            x_upper, dtype=np.dtype(np.float64), ndim=1, name="x upper"
        )
        first_output_source = _require_array(
            first_output,
            dtype=np.dtype(np.float64),
            ndim=2,
            name="first output",
        )
        delta_output_source = _require_array(
            delta_output,
            dtype=np.dtype(np.float64),
            ndim=2,
            name="initial output delta",
        )
        property_source = _require_array(
            property_row,
            dtype=np.dtype(np.float64),
            ndim=1,
            name="property row",
        )
        center_source = _require_array(
            target_output_center,
            dtype=np.dtype(np.float64),
            ndim=1,
            name="target output center",
        )
        if type(threshold) is not float or not math.isfinite(threshold):
            raise SealedEliminatedTargetUnknown("threshold must be a finite float")

        width = int(changes_source.size)
        x_width = int(lower_source.size)
        output_rows = int(first_output_source.shape[0])
        frame = EliminatedTargetResourceFrame(
            partition_rows=tuple(len(item[1]) for item in layout),
            input_columns=x_width,
            initial_changes=width,
            output_rows=output_rows,
        )
        resources = preflight_eliminated_target_resource_frame(frame)

        sealed_inputs, phase_rows = _validate_and_seal_partitions(
            partitions,
            expected_layout=layout,
            input_columns=x_width,
            initial_changes=width,
        )
        if phase_rows != resources.phase_rows:
            raise SealedEliminatedTargetUnknown("resource/partition row count drifted")
        # Seal every remaining caller alias, then validate only the snapshots.
        changes = _seal_caller_array(
            changes_source,
            name="change ordinals",
            expected_dtype=np.dtype(np.int64),
            expected_ndim=1,
            expected_shape=(width,),
        )
        base_bits = _seal_caller_array(
            base_source,
            name="change base active",
            expected_dtype=np.dtype(np.bool_),
            expected_ndim=1,
            expected_shape=(width,),
        )
        target_bits = _seal_caller_array(
            target_source,
            name="change target active",
            expected_dtype=np.dtype(np.bool_),
            expected_ndim=1,
            expected_shape=(width,),
        )
        lower = _seal_caller_array(
            lower_source,
            name="x lower",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=1,
            expected_shape=(x_width,),
        )
        upper = _seal_caller_array(
            upper_source,
            name="x upper",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=1,
            expected_shape=(x_width,),
        )
        first_out = _seal_caller_array(
            first_output_source,
            name="first output",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=2,
            expected_shape=(output_rows, x_width),
        )
        delta_out = _seal_caller_array(
            delta_output_source,
            name="initial output delta",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=2,
            expected_shape=(output_rows, width),
        )
        property_frozen = _seal_caller_array(
            property_source,
            name="property row",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=1,
            expected_shape=(output_rows,),
        )
        center_frozen = _seal_caller_array(
            center_source,
            name="target output center",
            expected_dtype=np.dtype(np.float64),
            expected_ndim=1,
            expected_shape=(output_rows,),
        )
        if (
            base_bits.shape != (width,)
            or target_bits.shape != (width,)
            or width <= 0
            or changes.shape != (width,)
            or np.any(changes < 0)
            or np.any(changes >= resources.phase_rows)
            or (width > 1 and np.any(changes[1:] <= changes[:-1]))
            or np.any(base_bits == target_bits)
            or upper.shape != (x_width,)
            or x_width <= 0
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
            or np.any(np.abs(lower) >= _INFINITE_BOUND)
            or np.any(np.abs(upper) >= _INFINITE_BOUND)
            or first_out.shape != (output_rows, x_width)
            or delta_out.shape != (output_rows, width)
            or property_frozen.shape != (output_rows,)
            or center_frozen.shape != (output_rows,)
            or not np.all(np.isfinite(first_out))
            or not np.all(np.isfinite(delta_out))
            or not np.all(np.isfinite(property_frozen))
            or not np.all(np.isfinite(center_frozen))
        ):
            raise SealedEliminatedTargetUnknown(
                "transaction shape, phase change, bounds, or output frame is invalid"
            )

        for index, ordinal_value in enumerate(changes):
            ordinal = int(ordinal_value)
            for (
                _partition_id,
                _first,
                delta,
                _centers,
                _active,
                row_ids,
                _stream_rows,
            ) in sealed_inputs:
                prefix = row_ids <= ordinal
                if np.any(delta[prefix, index] != 0.0):
                    raise SealedEliminatedTargetUnknown(
                        "initial delta violates topological injection causality"
                    )

        expansion_work = np.zeros((width, x_width), dtype=np.float64)
        for index, ordinal_value in enumerate(changes):
            ordinal = int(ordinal_value)
            base_q, delta_q, observed_target = _row_location(
                ordinal, sealed_inputs
            )
            base_active = bool(base_bits[index])
            target_active = bool(target_bits[index])
            if observed_target != target_active:
                raise SealedEliminatedTargetUnknown(
                    "phase-change target bit differs from partition authority"
                )
            if (not base_active) and target_active:
                expansion_work[index] = base_q
                if index:
                    expansion_work[index] += np.asarray(
                        delta_q[:index] @ expansion_work[:index],
                        dtype=np.float64,
                    )
            elif base_active and (not target_active):
                expansion_work[index] = -base_q
            else:
                raise SealedEliminatedTargetUnknown("phase change is invalid")
        if not np.all(np.isfinite(expansion_work)):
            raise SealedEliminatedTargetUnknown("triangular expansion overflowed")
        expansion = _seal_array(expansion_work)

        target_output_work = first_out + np.asarray(
            delta_out @ expansion, dtype=np.float64
        )
        objective_work = np.asarray(
            property_frozen[None, :] @ target_output_work, dtype=np.float64
        ).reshape(-1)
        objective_center = float(
            property_frozen @ center_frozen - threshold
        )
        if (
            objective_work.shape != (x_width,)
            or not np.all(np.isfinite(objective_work))
            or np.any(np.abs(objective_work) >= _INFINITE_BOUND)
            or np.any(
                (objective_work != 0.0)
                & (np.abs(objective_work) <= _SMALL_MATRIX_VALUE)
            )
            or not math.isfinite(objective_center)
        ):
            raise SealedEliminatedTargetUnknown(
                "eliminated objective left the owner numeric frame"
            )
        objective = _seal_array(np.ascontiguousarray(objective_work))

        sealed_partitions: list[_SealedPartition] = []
        logical_nnz = 0
        for (
            partition_id,
            first,
            delta,
            centers,
            active,
            row_ids,
            stream_rows,
        ) in sealed_inputs:
            target_work = first + np.asarray(delta @ expansion, dtype=np.float64)
            if not np.all(np.isfinite(target_work)):
                raise SealedEliminatedTargetUnknown(
                    "eliminated phase rows overflowed"
                )
            matrix = sp.csr_matrix(target_work)
            oriented = _canonical(
                matrix.multiply(
                    np.where(active, -1.0, 1.0)[:, None]
                ).tocsr(),
                name=f"eliminated partition {partition_id}",
            )
            logical_rows = SealedCsrRows.from_csr(
                oriented, name=f"sealed partition {partition_id}"
            )
            rhs_work = np.ascontiguousarray(
                np.where(active, centers, -centers), dtype=np.float64
            )
            row_max = _csr_box_upper(logical_rows.as_csr(), lower, upper)
            keep_work = np.ascontiguousarray(row_max > rhs_work, dtype=np.bool_)
            centers_frozen = _seal_array(centers)
            active_frozen = _seal_array(active)
            ids_frozen = _seal_array(row_ids)
            stream_frozen = _seal_array(stream_rows)
            rhs_frozen = _seal_array(rhs_work)
            keep_frozen = _seal_array(keep_work)
            digest = _digest_text(
                logical_rows.content_sha256,
                _digest_arrays(
                    centers_frozen,
                    active_frozen,
                    ids_frozen,
                    stream_frozen,
                    rhs_frozen,
                    keep_frozen,
                ),
                str(partition_id),
            )
            sealed_partition = _SealedPartition(
                partition_id=partition_id,
                logical_rows=logical_rows,
                phase_centers=centers_frozen,
                target_active=active_frozen,
                row_ids=ids_frozen,
                stream_rows=stream_frozen,
                rhs=rhs_frozen,
                keep=keep_frozen,
                content_sha256=digest,
            )
            sealed_partition.assert_intact()
            sealed_partitions.append(sealed_partition)
            logical_nnz += int(logical_rows.data.size)
        partition_authority = tuple(sealed_partitions)
        if not any(np.any(partition.keep) for partition in partition_authority):
            raise SealedEliminatedTargetUnknown("base screen retained no phase row")
        base_rows = _freeze_base_rows(
            partition_authority, x_lower=lower, x_upper=upper
        )
        kept_row_ids = _seal_array(
            np.ascontiguousarray(
                np.concatenate(
                    [
                        partition.row_ids[partition.keep]
                        for partition in partition_authority
                    ]
                ),
                dtype=np.int64,
            )
        )
        if not np.array_equal(base_rows.row_ids, kept_row_ids):
            raise SealedEliminatedTargetUnknown(
                "owner row IDs differ from immutable kept-row authority"
            )
        repair_resource_frame = _seal_array(
            np.ascontiguousarray(
                [
                    phase_rows,
                    x_width,
                    output_rows,
                    len(partition_authority),
                    resources.conservative_authority_bytes,
                ],
                dtype=np.int64,
            )
        )
        transaction_sha256 = _digest_text(
            *[partition.content_sha256 for partition in partition_authority],
            _digest_arrays(
                expansion,
                objective,
                property_frozen,
                lower,
                upper,
                kept_row_ids,
                repair_resource_frame,
            ),
            base_rows.content_sha256,
            float(objective_center).hex(),
            source_binding_sha256,
            layout_binding_sha256,
        )
        provisional_receipt = TransactionReceipt(
            format_version=_FORMAT_VERSION,
            synthetic_only=True,
            source_binding_sha256=source_binding_sha256,
            layout_binding_sha256=layout_binding_sha256,
            phase_rows=phase_rows,
            input_columns=x_width,
            initial_changes=width,
            logical_nnz=logical_nnz,
            base_rows=int(base_rows.rows),
            base_logical_nnz=int(base_rows.logical_nnz),
            base_loaded_nnz=int(base_rows.data.size),
            base_deleted_tiny_nnz=int(base_rows.deleted_tiny_nnz),
            partition_count=len(partition_authority),
            conservative_authority_bytes=resources.conservative_authority_bytes,
            base_row_space="phase_only_unchanged",
            row_id_rule="contiguous_global_phase_ordinal",
            partition_binding_rule=(
                "exact_external_layout_boundary_order_ids_stream_rows_and_source_sha256"
            ),
            base_owner_abi_target=(
                "act.back_end.hybridz_tf.phase_projection_highs_owner."
                "SafeHighsOwner"
            ),
            owner_lifecycle_externally_unproven=True,
            selector_authority="external_unchanged",
            terminal_authority="external_unchanged",
            retained_target_pre_container=False,
            retained_target_output=False,
            retained_global_logical_csr=False,
            retained_partitioned_full_logical_rows=True,
            retained_owner_screened_frozen_rows=True,
            retained_screened_scipy_wrapper=False,
            base_auxiliary_normal_form=False,
            repair_handoff_rule="one_shot_partition_borrow_no_rebuild",
            production_repair_global_csr_blocker=True,
            gpu_used=False,
            benchmark_run=False,
            transaction_sha256=transaction_sha256,
            content_sha256="",
        )
        receipt = replace(
            provisional_receipt,
            content_sha256=_transaction_receipt_digest(provisional_receipt),
        )
        receipt.assert_intact()
        transaction = SealedEliminatedTargetTransaction(
            partitions=partition_authority,
            expansion=expansion,
            objective_coeff=objective,
            objective_center=objective_center,
            property_row=property_frozen,
            x_lower=lower,
            x_upper=upper,
            kept_row_ids=kept_row_ids,
            repair_resource_frame=repair_resource_frame,
            base_rows=base_rows,
            source_binding_sha256=source_binding_sha256,
            layout_binding_sha256=layout_binding_sha256,
            transaction_sha256=transaction_sha256,
            receipt=receipt,
        )
        transaction.assert_intact()
        return transaction
    except SealedEliminatedTargetUnknown:
        raise
    except Exception as exc:
        raise SealedEliminatedTargetUnknown(
            "sealed eliminated-target transaction failed closed"
        ) from exc
