#!/usr/bin/env python3
# ===- exact_sparse_conv_affine_core.py - direct exact CONV affine core --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Disconnected, non-consumable row-local sparse-CONV research reference.

This module fuses the *linear* part of Operator-HZ's ``_affine`` operation for
an injective row-local source.  It traverses each valid CONV geometry/weight
coefficient once per layer snapshot and computes all of the following without
ever constructing the complete CONV matrix and without sparse matrix-matrix
multiplication:

``center_linear``
    The binary64 result of ``W @ source.center`` in canonical CSR order.
``generators``
    The canonical CSR result of ``W @ source.generators``.  Row locality and
    injectivity prove that every retained output coefficient has one product
    contributor, so no sparse reduction is reordered.
``transformed_mass`` and ``propagated_error``
    The same outward binary64 bounds used by Operator-HZ.
``fanin``
    The exact number of stored, nonzero CONV coefficients in every output row.

The separate :func:`finalize_exact_sparse_conv_affine_core` function adds the
snapshotted bias and applies Operator-HZ's arithmetic-error formulas.  Honest
factory-produced inputs have been useful as a bit-for-bit reference for the
established binary64 sparse operator path.  Public source/core/result objects,
however, are self-describing research records rather than authenticated
capabilities: their portable hashes do not prove factory provenance.  No
receipt from this module is authoritative, and no production, proof, verifier,
or verdict path may consume its outputs.

Only :class:`RowLocalNotApplicable` denotes a valid source for which this fast
path is structurally inapplicable.  Malformed CSR, geometry, identifiers,
non-finite arithmetic, and broken internal invariants raise fail-closed core
errors and must never select a fallback.

The module contains no triangle relaxation, branch-and-bound, backward or
dual operation, solver call, dense CONV matrix, torch CONV, full sparse CONV
operator, or SpGEMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import operator
from typing import Any, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import _sparsetools as _scipy_sparsetools


_SOURCE_SCHEMA = "act.exact_sparse_conv_affine.source.v1"
_CORE_SCHEMA = "act.exact_sparse_conv_affine.linear_core.v1"
_RESULT_SCHEMA = "act.exact_sparse_conv_affine.result.v1"
_RECEIPT_SCHEMA = "act.exact_sparse_conv_affine.receipt.v2"
_GEOMETRY_SCHEMA = "act.exact_sparse_conv_affine.geometry.v1"
_INT32_MAX = int(np.iinfo(np.int32).max)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)
_F64_EPS = np.finfo(np.float64).eps
_F64_TINY = np.finfo(np.float64).tiny
_MAX_COEFFICIENT_SLAB_ENTRIES = 1_000_000


class ExactSparseConvAffineCoreError(ValueError):
    """Fail-closed rejection of malformed input or unsafe binary64 arithmetic."""


class ExactSparseConvAffineCoreInternalError(RuntimeError):
    """Fail-closed violation of a private construction invariant."""


class RowLocalNotApplicable(Exception):
    """The sole fallback signal: a valid CSR is not row-local and injective."""


def _private_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Return a C-order ndarray whose immutable backing owner is ``bytes``."""

    normalized = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(normalized.tobytes(order="C"), dtype=dtype).reshape(
        normalized.shape
    )


def _snapshot_f64(value: Any, *, name: str) -> np.ndarray:
    """Take one finite, owning binary64 snapshot without importing torch."""

    source = value
    try:
        detach = getattr(source, "detach", None)
        if callable(detach):
            source = detach()
            cpu = getattr(source, "cpu", None)
            if callable(cpu):
                source = cpu()
            numpy_method = getattr(source, "numpy", None)
            if callable(numpy_method):
                source = numpy_method()
        raw = np.asarray(source)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ExactSparseConvAffineCoreError(
            f"{name} cannot be converted to a host array"
        ) from exc
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ExactSparseConvAffineCoreError(f"{name} must be real-valued")
    try:
        snapshot = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExactSparseConvAffineCoreError(
            f"{name} cannot be represented as binary64"
        ) from exc
    if not np.all(np.isfinite(snapshot)):
        raise ExactSparseConvAffineCoreError(f"{name} contains NaN or infinity")
    return _private_array(snapshot, dtype=np.dtype(np.float64))


def _builtin_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ExactSparseConvAffineCoreError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise ExactSparseConvAffineCoreError(f"{name} must be an integer") from exc
    if result < _INT64_MIN or result > _INT64_MAX:
        raise ExactSparseConvAffineCoreError(f"{name} exceeds signed int64")
    return result


def _pair(value: Any, *, name: str, positive: bool) -> Tuple[int, int]:
    try:
        scalar = _builtin_int(value, name=name)
    except ExactSparseConvAffineCoreError:
        try:
            items = tuple(value)
        except TypeError as exc:
            raise ExactSparseConvAffineCoreError(
                f"{name} must be an integer or a length-two sequence"
            ) from exc
        if len(items) != 2:
            raise ExactSparseConvAffineCoreError(
                f"{name} must be an integer or a length-two sequence"
            )
        result = (
            _builtin_int(items[0], name=f"{name}[0]"),
            _builtin_int(items[1], name=f"{name}[1]"),
        )
    else:
        result = (scalar, scalar)
    if positive and min(result) <= 0:
        raise ExactSparseConvAffineCoreError(f"{name} must be positive")
    if not positive and min(result) < 0:
        raise ExactSparseConvAffineCoreError(f"{name} must be nonnegative")
    return result


def _shape4(value: Any, *, name: str) -> Tuple[int, int, int, int]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ExactSparseConvAffineCoreError(
            f"{name} must be an explicit four-dimensional NCHW shape"
        ) from exc
    if len(items) != 4:
        raise ExactSparseConvAffineCoreError(
            f"{name} must be an explicit four-dimensional NCHW shape"
        )
    shape = tuple(
        _builtin_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(items)
    )
    if min(shape) <= 0:
        raise ExactSparseConvAffineCoreError(f"{name} must be positive")
    return shape  # type: ignore[return-value]


def _checked_i64(value: int, *, name: str) -> int:
    if type(value) is not int or value < _INT64_MIN or value > _INT64_MAX:
        raise ExactSparseConvAffineCoreError(
            f"{name} exceeds checked signed-int64 arithmetic"
        )
    return value


def _checked_product(
    values: Tuple[int, ...], *, limit: int, name: str
) -> int:
    result = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise ExactSparseConvAffineCoreError(
                f"{name} contains a negative or non-builtin factor"
            )
        if value and result > limit // value:
            raise ExactSparseConvAffineCoreError(
                f"{name} exceeds checked limit {limit}"
            )
        result *= value
    return result


def _snapshot_ids(value: Any, *, size: int, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExactSparseConvAffineCoreError(
            f"{name} is not an integer vector"
        ) from exc
    if raw.ndim != 1 or raw.size != size or raw.dtype.kind not in "iu":
        raise ExactSparseConvAffineCoreError(
            f"{name} must be a one-dimensional integer vector of length {size}"
        )
    if (
        raw.dtype.kind == "u"
        and raw.dtype.itemsize >= np.dtype(np.uint64).itemsize
        and raw.size
        and np.any(raw > np.uint64(_INT64_MAX))
    ):
        raise ExactSparseConvAffineCoreError(f"{name} exceeds signed int64")
    ids = np.asarray(raw, dtype=np.int64)
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise ExactSparseConvAffineCoreError(
            f"{name} must contain unique nonnegative identifiers"
        )
    return _private_array(ids, dtype=np.dtype(np.int64))


def _validate_canonical_csr_storage(
    value: Any, *, name: str, copy: bool
) -> sp.csr_matrix:
    """Validate raw CSR bytes without trusting SciPy's cached format flags."""

    if type(value) is not sp.csr_matrix:
        raise ExactSparseConvAffineCoreError(f"{name} must be an exact csr_matrix")
    try:
        shape = value.shape
        raw_data = value.data
        raw_indices = value.indices
        raw_indptr = value.indptr
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExactSparseConvAffineCoreError(
            f"{name} has malformed CSR storage"
        ) from exc
    if (
        type(shape) is not tuple
        or len(shape) != 2
        or any(type(item) is not int for item in shape)
        or min(shape) < 0
        or max(shape) > _INT32_MAX
    ):
        raise ExactSparseConvAffineCoreError(f"{name} has an invalid int32 shape")
    arrays = (raw_data, raw_indices, raw_indptr)
    if any(
        type(array) is not np.ndarray
        or array.ndim != 1
        or not array.flags.c_contiguous
        or any(stride < 0 for stride in array.strides)
        for array in arrays
    ):
        raise ExactSparseConvAffineCoreError(
            f"{name} CSR arrays must be exact C-contiguous one-dimensional ndarrays"
        )
    if raw_data.dtype != np.dtype(np.float64):
        raise ExactSparseConvAffineCoreError(f"{name} data must use native float64")
    allowed_indices = {np.dtype(np.int32), np.dtype(np.int64)}
    if (
        raw_indices.dtype not in allowed_indices
        or raw_indptr.dtype != raw_indices.dtype
    ):
        raise ExactSparseConvAffineCoreError(
            f"{name} indices and indptr must share native int32 or int64 dtype"
        )

    if copy:
        data = _private_array(raw_data, dtype=np.dtype(np.float64))
        indices_wide = _private_array(raw_indices, dtype=raw_indices.dtype)
        indptr_wide = _private_array(raw_indptr, dtype=raw_indptr.dtype)
    else:
        data = raw_data
        indices_wide = raw_indices
        indptr_wide = raw_indptr
        if any(array.flags.writeable for array in arrays):
            raise ExactSparseConvAffineCoreError(
                f"{name} is not an immutable private CSR snapshot"
            )

    n_rows, n_cols = shape
    nnz = int(data.size)
    if (
        nnz > _INT32_MAX
        or indices_wide.size != nnz
        or indptr_wide.size != n_rows + 1
        or indptr_wide.size == 0
        or int(indptr_wide[0]) != 0
        or int(indptr_wide[-1]) != nnz
        or np.any(indptr_wide < 0)
        or np.any(indptr_wide[1:] < indptr_wide[:-1])
    ):
        raise ExactSparseConvAffineCoreError(f"{name} has invalid CSR indptr/storage")
    if (
        not np.all(np.isfinite(data))
        or np.any(data == 0.0)
        or np.any(indices_wide < 0)
        or np.any(indices_wide >= n_cols)
    ):
        raise ExactSparseConvAffineCoreError(
            f"{name} has nonfinite/zero data or out-of-domain indices"
        )
    if nnz > 1:
        same_row = np.ones(nnz - 1, dtype=np.bool_)
        boundaries = indptr_wide[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < nnz)]
        if boundaries.size:
            same_row[np.asarray(boundaries, dtype=np.int64) - 1] = False
        if np.any(same_row & (indices_wide[1:] <= indices_wide[:-1])):
            raise ExactSparseConvAffineCoreError(
                f"{name} columns must be strictly increasing in every row"
            )

    if not copy:
        return value
    indices = _private_array(indices_wide, dtype=np.dtype(np.int32))
    indptr = _private_array(indptr_wide, dtype=np.dtype(np.int32))
    snapshot = sp.csr_matrix(
        (data, indices, indptr), shape=shape, dtype=np.float64, copy=False
    )
    if (
        snapshot.data.flags.writeable
        or snapshot.indices.flags.writeable
        or snapshot.indptr.flags.writeable
        or snapshot.data.dtype != np.dtype(np.float64)
        or snapshot.indices.dtype != np.dtype(np.int32)
        or snapshot.indptr.dtype != np.dtype(np.int32)
    ):
        raise ExactSparseConvAffineCoreInternalError(
            "private CSR construction lost immutable native storage"
        )
    return snapshot


def _digest_array(digest: Any, value: np.ndarray) -> None:
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))


def _digest_csr(digest: Any, value: sp.csr_matrix) -> None:
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    _digest_array(digest, value.indptr)
    _digest_array(digest, value.indices)
    _digest_array(digest, value.data)


def _gamma_ops(op_count: np.ndarray | int, *, name: str) -> np.ndarray:
    count = np.asarray(op_count, dtype=np.float64)
    if np.any(count < 0.0) or not np.all(np.isfinite(count)):
        raise ExactSparseConvAffineCoreError(f"{name} has invalid operation counts")
    product = count * _F64_EPS
    if np.any(product >= 0.5):
        raise ExactSparseConvAffineCoreError(
            f"{name} exceeds the finite binary64 roundoff regime"
        )
    return product / (1.0 - product)


def _inflate_nonnegative(
    rounded: np.ndarray,
    op_count: np.ndarray | int,
    *,
    active: Optional[np.ndarray] = None,
    name: str,
) -> np.ndarray:
    value = np.asarray(rounded, dtype=np.float64)
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ExactSparseConvAffineCoreError(f"{name} is non-finite or negative")
    count = np.broadcast_to(np.asarray(op_count, dtype=np.float64), value.shape)
    gamma = _gamma_ops(count, name=name)
    active_mask = (
        value > 0.0
        if active is None
        else np.broadcast_to(np.asarray(active, dtype=np.bool_), value.shape)
    )
    out = np.zeros_like(value)
    if np.any(active_mask):
        out[active_mask] = value[active_mask] / (1.0 - gamma[active_mask])
        out[active_mask] = out[active_mask] + _F64_TINY * np.maximum(
            1.0, count[active_mask]
        )
        out[active_mask] = np.nextafter(out[active_mask], np.inf)
    if not np.all(np.isfinite(out)):
        raise ExactSparseConvAffineCoreError(f"{name} outward inflation overflowed")
    return out


def _nonnegative_sum_upper(*terms: np.ndarray, name: str) -> np.ndarray:
    if not terms:
        raise ExactSparseConvAffineCoreInternalError(
            f"{name} requires at least one term"
        )
    arrays = [np.asarray(term, dtype=np.float64) for term in terms]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ExactSparseConvAffineCoreError(f"{name} shape mismatch")
    if any(
        not np.all(np.isfinite(array)) or np.any(array < 0.0)
        for array in arrays
    ):
        raise ExactSparseConvAffineCoreError(f"{name} has a nonfinite/negative term")
    rounded = np.zeros(shape, dtype=np.float64)
    active = np.zeros(shape, dtype=np.bool_)
    for array in arrays:
        rounded = rounded + array
        active |= array > 0.0
    return _inflate_nonnegative(
        rounded,
        max(1, 2 * len(arrays)),
        active=active,
        name=name,
    )


def _row_local_l1_upper(
    row_scale: np.ndarray, row_to_column: np.ndarray
) -> np.ndarray:
    active = row_to_column >= 0
    raw = np.zeros(row_scale.shape, dtype=np.float64)
    raw[active] = np.abs(row_scale[active])
    return _inflate_nonnegative(
        raw,
        2.0 * active.astype(np.float64) + 2.0,
        active=active,
        name="row_local_source.G_l1",
    )


@dataclass(frozen=True, eq=False)
class ExactRowLocalAffineSource:
    """Immutable, identity-bound row-local source admitted by this core."""

    center: np.ndarray
    generators: sp.csr_matrix
    error: np.ndarray
    stable_column_ids: np.ndarray
    row_to_generator_column: np.ndarray
    row_scale: np.ndarray
    source_mass: np.ndarray
    positional_mapping_monotone: bool
    digest: str
    schema: str = _SOURCE_SCHEMA

    @property
    def size(self) -> int:
        return int(self.center.size)


@dataclass(frozen=True, eq=False)
class ExactSparseConvLinearCore:
    """Immutable output of the one-pass geometry/weight traversal."""

    center_linear: np.ndarray
    generators: sp.csr_matrix
    transformed_mass: np.ndarray
    propagated_error: np.ndarray
    fanin: np.ndarray
    bias: np.ndarray
    stable_column_ids: np.ndarray
    source_digest: str
    geometry_digest: str
    weight_digest: str
    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    operator_nnz: int
    digest: str
    schema: str = _CORE_SCHEMA

    @property
    def size(self) -> int:
        return int(self.center_linear.size)


@dataclass(frozen=True, eq=False)
class ExactSparseConvAffineResult:
    """Final affine expression, still without property/verdict authority."""

    center: np.ndarray
    generators: sp.csr_matrix
    error: np.ndarray
    stable_column_ids: np.ndarray
    affine_depth: int
    core_digest: str
    digest: str
    schema: str = _RESULT_SCHEMA

    @property
    def size(self) -> int:
        return int(self.center.size)


@dataclass(frozen=True)
class ExactSparseConvAffineCoreReceipt:
    """Non-authoritative accounting for the disconnected research reference."""

    source_digest: str
    geometry_digest: str
    weight_digest: str
    core_digest: str
    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    source_generator_nnz: int
    output_generator_nnz: int
    operator_nnz: int
    geometry_weight_traversals: int = 1
    construction_mode: str = "disconnected_row_local_affine_reference_v2"
    linear_primitive_authoritative: bool = False
    property_proof_authority: bool = False
    verdict_authority: bool = False
    private_binary64_snapshots: bool = True
    stable_ids_bound: bool = True
    full_conv_operator_materialized: bool = False
    transient_operator_sparse_matrix_materialized: bool = False
    uses_compiled_csr_vector_reduction: bool = True
    maximum_coefficient_slab_entries: int = _MAX_COEFFICIENT_SLAB_ENTRIES
    uses_spgemm: bool = False
    uses_dense_conv_matrix: bool = False
    uses_torch_conv: bool = False
    uses_triangle_relaxation: bool = False
    uses_branch_and_bound: bool = False
    uses_backward_or_dual: bool = False
    uses_solver: bool = False
    schema: str = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        digests = (
            self.source_digest,
            self.geometry_digest,
            self.weight_digest,
            self.core_digest,
        )
        if (
            any(type(value) is not str or len(value) != 64 for value in digests)
            or any(
                type(value) is not int or value < 0
                for value in (
                    *self.input_shape,
                    *self.output_shape,
                    self.source_generator_nnz,
                    self.output_generator_nnz,
                    self.operator_nnz,
                )
            )
            or self.geometry_weight_traversals != 1
            or self.construction_mode
            != "disconnected_row_local_affine_reference_v2"
            or self.linear_primitive_authoritative is not False
            or self.property_proof_authority is not False
            or self.verdict_authority is not False
            or self.private_binary64_snapshots is not True
            or self.stable_ids_bound is not True
            or self.full_conv_operator_materialized is not False
            or self.transient_operator_sparse_matrix_materialized is not False
            or self.uses_compiled_csr_vector_reduction is not True
            or self.maximum_coefficient_slab_entries
            != _MAX_COEFFICIENT_SLAB_ENTRIES
            or self.uses_spgemm
            or self.uses_dense_conv_matrix
            or self.uses_torch_conv
            or self.uses_triangle_relaxation
            or self.uses_branch_and_bound
            or self.uses_backward_or_dual
            or self.uses_solver
            or self.schema != _RECEIPT_SCHEMA
        ):
            raise ExactSparseConvAffineCoreInternalError(
                "malformed affine-core receipt"
            )


def _source_digest(source: ExactRowLocalAffineSource) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_SCHEMA.encode("ascii"))
    _digest_array(digest, source.center)
    _digest_csr(digest, source.generators)
    _digest_array(digest, source.error)
    _digest_array(digest, source.stable_column_ids)
    _digest_array(digest, source.row_to_generator_column)
    _digest_array(digest, source.row_scale)
    _digest_array(digest, source.source_mass)
    digest.update(bytes((int(source.positional_mapping_monotone),)))
    return digest.hexdigest()


def _core_digest(core: ExactSparseConvLinearCore) -> str:
    digest = hashlib.sha256()
    digest.update(_CORE_SCHEMA.encode("ascii"))
    for text in (core.source_digest, core.geometry_digest, core.weight_digest):
        digest.update(text.encode("ascii"))
    for shape in (core.input_shape, core.output_shape):
        digest.update(np.asarray(shape, dtype="<i8").tobytes())
    digest.update(np.asarray((core.operator_nnz,), dtype="<i8").tobytes())
    _digest_array(digest, core.center_linear)
    _digest_csr(digest, core.generators)
    _digest_array(digest, core.transformed_mass)
    _digest_array(digest, core.propagated_error)
    _digest_array(digest, core.fanin)
    _digest_array(digest, core.bias)
    _digest_array(digest, core.stable_column_ids)
    return digest.hexdigest()


def _result_digest(result: ExactSparseConvAffineResult) -> str:
    digest = hashlib.sha256()
    digest.update(_RESULT_SCHEMA.encode("ascii"))
    digest.update(result.core_digest.encode("ascii"))
    digest.update(np.asarray((result.affine_depth,), dtype="<i8").tobytes())
    _digest_array(digest, result.center)
    _digest_csr(digest, result.generators)
    _digest_array(digest, result.error)
    _digest_array(digest, result.stable_column_ids)
    return digest.hexdigest()


def prepare_exact_row_local_affine_source(
    center: Any,
    generators: Any,
    error: Any,
    *,
    stable_column_ids: Any,
) -> ExactRowLocalAffineSource:
    """Validate and privately snapshot one row-local, injective affine source."""

    center_snapshot = _snapshot_f64(center, name="source.center").reshape(-1)
    error_snapshot = _snapshot_f64(error, name="source.error").reshape(-1)
    matrix = _validate_canonical_csr_storage(
        generators, name="source.generators", copy=True
    )
    if (
        matrix.shape[0] != center_snapshot.size
        or error_snapshot.size != center_snapshot.size
    ):
        raise ExactSparseConvAffineCoreError(
            "source center/generator/error row counts disagree"
        )
    if np.any(error_snapshot < 0.0):
        raise ExactSparseConvAffineCoreError("source.error must be nonnegative")
    ids = _snapshot_ids(
        stable_column_ids,
        size=int(matrix.shape[1]),
        name="stable_column_ids",
    )

    row_counts = np.diff(matrix.indptr)
    if np.any(row_counts > 1):
        raise RowLocalNotApplicable(
            "valid source is not row-local: one source row has multiple generators"
        )
    nonempty_rows = np.flatnonzero(row_counts == 1)
    positions = matrix.indptr[nonempty_rows]
    mapped_columns = np.asarray(matrix.indices[positions], dtype=np.int64)
    if np.unique(mapped_columns).size != mapped_columns.size:
        raise RowLocalNotApplicable(
            "valid row-local source is not injective across source rows"
        )
    row_to_column = np.full(center_snapshot.size, -1, dtype=np.int64)
    row_scale = np.zeros(center_snapshot.size, dtype=np.float64)
    row_to_column[nonempty_rows] = mapped_columns
    row_scale[nonempty_rows] = matrix.data[positions]
    frozen_row_to_column = _private_array(
        row_to_column, dtype=np.dtype(np.int64)
    )
    frozen_row_scale = _private_array(row_scale, dtype=np.dtype(np.float64))
    l1 = _row_local_l1_upper(frozen_row_scale, frozen_row_to_column)
    source_mass = _nonnegative_sum_upper(
        np.abs(center_snapshot),
        l1,
        error_snapshot,
        name="row_local_source.mass",
    )
    frozen_mass = _private_array(source_mass, dtype=np.dtype(np.float64))
    monotone = bool(
        mapped_columns.size <= 1
        or np.all(mapped_columns[1:] > mapped_columns[:-1])
    )
    provisional = ExactRowLocalAffineSource(
        center=center_snapshot,
        generators=matrix,
        error=error_snapshot,
        stable_column_ids=ids,
        row_to_generator_column=frozen_row_to_column,
        row_scale=frozen_row_scale,
        source_mass=frozen_mass,
        positional_mapping_monotone=monotone,
        digest="",
    )
    digest = _source_digest(provisional)
    return ExactRowLocalAffineSource(
        center=provisional.center,
        generators=provisional.generators,
        error=provisional.error,
        stable_column_ids=provisional.stable_column_ids,
        row_to_generator_column=provisional.row_to_generator_column,
        row_scale=provisional.row_scale,
        source_mass=provisional.source_mass,
        positional_mapping_monotone=provisional.positional_mapping_monotone,
        digest=digest,
    )


def _require_private_array(
    value: Any, *, dtype: np.dtype[Any], shape: Tuple[int, ...], name: str
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
        or value.flags.writeable
        or not np.all(np.isfinite(value))
    ):
        raise ExactSparseConvAffineCoreError(f"{name} is not a valid private snapshot")
    return value


def _validate_source(
    source: Any, *, expected_stable_column_ids: Any
) -> ExactRowLocalAffineSource:
    if type(source) is not ExactRowLocalAffineSource or source.schema != _SOURCE_SCHEMA:
        raise ExactSparseConvAffineCoreError("source is not an affine-core snapshot")
    n_rows = int(getattr(source.center, "size", -1))
    _require_private_array(
        source.center,
        dtype=np.dtype(np.float64),
        shape=(n_rows,),
        name="source.center",
    )
    matrix = _validate_canonical_csr_storage(
        source.generators, name="source.generators", copy=False
    )
    if matrix.shape[0] != n_rows:
        raise ExactSparseConvAffineCoreError("source generator row count changed")
    _require_private_array(
        source.error,
        dtype=np.dtype(np.float64),
        shape=(n_rows,),
        name="source.error",
    )
    _require_private_array(
        source.source_mass,
        dtype=np.dtype(np.float64),
        shape=(n_rows,),
        name="source.source_mass",
    )
    _require_private_array(
        source.row_to_generator_column,
        dtype=np.dtype(np.int64),
        shape=(n_rows,),
        name="source.row_to_generator_column",
    )
    _require_private_array(
        source.row_scale,
        dtype=np.dtype(np.float64),
        shape=(n_rows,),
        name="source.row_scale",
    )
    _require_private_array(
        source.stable_column_ids,
        dtype=np.dtype(np.int64),
        shape=(int(matrix.shape[1]),),
        name="source.stable_column_ids",
    )
    if (
        np.any(source.error < 0.0)
        or np.any(source.source_mass < 0.0)
        or type(source.positional_mapping_monotone) is not bool
        or type(source.digest) is not str
        or len(source.digest) != 64
        or _source_digest(source) != source.digest
    ):
        raise ExactSparseConvAffineCoreError("source snapshot integrity check failed")
    expected = _snapshot_ids(
        expected_stable_column_ids,
        size=int(matrix.shape[1]),
        name="expected_stable_column_ids",
    )
    if not np.array_equal(expected, source.stable_column_ids):
        raise ExactSparseConvAffineCoreError(
            "stable generator identifiers do not match the source snapshot"
        )
    return source


@dataclass(frozen=True, eq=False)
class _ConvGeometry:
    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    kernel: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int
    kernel_gather_by_output: np.ndarray
    input_spatial_by_output: np.ndarray
    digest: str


def _geometry_digest(geometry: _ConvGeometry) -> str:
    digest = hashlib.sha256()
    digest.update(_GEOMETRY_SCHEMA.encode("ascii"))
    for values in (
        geometry.input_shape,
        geometry.output_shape,
        geometry.kernel,
        geometry.stride,
        geometry.padding,
        geometry.dilation,
        (geometry.groups,),
    ):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    _digest_array(digest, geometry.kernel_gather_by_output)
    _digest_array(digest, geometry.input_spatial_by_output)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _cached_geometry(
    input_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int, int, int],
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> _ConvGeometry:
    _batch, _in_ch, in_h, in_w = input_shape
    _out_batch, _out_ch, out_h, out_w = output_shape
    kh, kw = kernel
    output_h = np.arange(out_h, dtype=np.int64)
    output_w = np.arange(out_w, dtype=np.int64)
    gather = np.zeros((out_h * out_w, kh * kw), dtype=np.int32)
    spatial = np.full((out_h * out_w, kh * kw), -1, dtype=np.int32)
    for rr in range(kh):
        input_h = output_h * stride[0] - padding[0] + rr * dilation[0]
        valid_h = (input_h >= 0) & (input_h < in_h)
        if not np.any(valid_h):
            continue
        oh = output_h[valid_h]
        ih = input_h[valid_h]
        for cc in range(kw):
            input_w_values = (
                output_w * stride[1] - padding[1] + cc * dilation[1]
            )
            valid_w = (input_w_values >= 0) & (input_w_values < in_w)
            if not np.any(valid_w):
                continue
            ow = output_w[valid_w]
            iw = input_w_values[valid_w]
            output_spatial = (oh[:, None] * out_w + ow[None, :]).reshape(-1)
            input_spatial = (ih[:, None] * in_w + iw[None, :]).reshape(-1)
            kernel_index = rr * kw + cc
            gather[output_spatial, kernel_index] = kernel_index
            spatial[output_spatial, kernel_index] = input_spatial
    frozen_gather = _private_array(gather, dtype=np.dtype(np.int32))
    frozen_spatial = _private_array(spatial, dtype=np.dtype(np.int32))
    provisional = _ConvGeometry(
        input_shape=input_shape,
        output_shape=output_shape,
        kernel=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        kernel_gather_by_output=frozen_gather,
        input_spatial_by_output=frozen_spatial,
        digest="",
    )
    return _ConvGeometry(
        input_shape=input_shape,
        output_shape=output_shape,
        kernel=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        kernel_gather_by_output=frozen_gather,
        input_spatial_by_output=frozen_spatial,
        digest=_geometry_digest(provisional),
    )


def _checked_output_extent(
    input_extent: int,
    kernel_extent: int,
    stride: int,
    padding: int,
    dilation: int,
    *,
    axis: str,
) -> int:
    twice_padding = _checked_i64(2 * padding, name=f"{axis}.2*padding")
    effective_tail = _checked_i64(
        dilation * (kernel_extent - 1), name=f"{axis}.dilation*(kernel-1)"
    )
    numerator = _checked_i64(
        input_extent + twice_padding, name=f"{axis}.input+2*padding"
    )
    numerator = _checked_i64(
        numerator - effective_tail, name=f"{axis}.padded-effective_tail"
    )
    numerator = _checked_i64(numerator - 1, name=f"{axis}.numerator")
    output = _checked_i64(numerator // stride + 1, name=f"{axis}.output")
    if output <= 0:
        raise ExactSparseConvAffineCoreError(
            f"{axis} geometry has no positive output extent"
        )
    output_tail = _checked_i64(
        (output - 1) * stride, name=f"{axis}.output_tail*stride"
    )
    shifted_tail = _checked_i64(
        output_tail - padding, name=f"{axis}.output_tail-padding"
    )
    _checked_i64(-padding, name=f"{axis}.-padding")
    _checked_i64(
        shifted_tail + effective_tail, name=f"{axis}.max_spatial_index"
    )
    return output


def _layer_snapshot(
    layer: Any,
) -> Tuple[np.ndarray, np.ndarray, _ConvGeometry]:
    try:
        params = layer.params
        weight_value = params["weight"]
        input_shape_value = params["input_shape"]
        output_shape_value = params["output_shape"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ExactSparseConvAffineCoreError(
            "layer must provide weight/input_shape/output_shape parameters"
        ) from exc
    if str(params.get("data_format", "NCHW")).upper() != "NCHW":
        raise ExactSparseConvAffineCoreError("only NCHW CONV is supported")
    if str(params.get("padding_mode", "zeros")).lower() not in {
        "zeros",
        "zero",
        "constant",
    }:
        raise ExactSparseConvAffineCoreError("only zero padding is supported")
    if str(params.get("auto_pad", "NOTSET")).upper() not in {
        "NOTSET",
        "NONE",
        "",
    }:
        raise ExactSparseConvAffineCoreError("automatic padding is unsupported")

    weight = _snapshot_f64(weight_value, name="weight")
    if weight.ndim != 4 or min(weight.shape) <= 0:
        raise ExactSparseConvAffineCoreError(
            "weight must have positive (out_ch,in_ch/group,kh,kw) shape"
        )
    out_ch, in_ch_per_group, kh, kw = (int(item) for item in weight.shape)
    input_shape = _shape4(input_shape_value, name="input_shape")
    output_shape = _shape4(output_shape_value, name="output_shape")
    batch, in_ch, in_h, in_w = input_shape
    out_batch, declared_out_ch, out_h, out_w = output_shape
    groups = _builtin_int(params.get("groups", 1), name="groups")
    stride = _pair(params.get("stride", 1), name="stride", positive=True)
    padding = _pair(params.get("padding", 0), name="padding", positive=False)
    dilation = _pair(params.get("dilation", 1), name="dilation", positive=True)
    if (
        groups <= 0
        or batch != out_batch
        or out_ch != declared_out_ch
        or in_ch_per_group * groups != in_ch
        or out_ch % groups != 0
    ):
        raise ExactSparseConvAffineCoreError(
            "weight/groups/channels/batch disagree with declared geometry"
        )
    expected_h = _checked_output_extent(
        in_h, kh, stride[0], padding[0], dilation[0], axis="height"
    )
    expected_w = _checked_output_extent(
        in_w, kw, stride[1], padding[1], dilation[1], axis="width"
    )
    if (out_h, out_w) != (expected_h, expected_w):
        raise ExactSparseConvAffineCoreError(
            "declared output shape disagrees with CONV geometry"
        )
    _checked_product(input_shape, limit=_INT32_MAX, name="flattened input rows")
    _checked_product(output_shape, limit=_INT32_MAX, name="flattened output rows")
    _checked_product(
        (out_h, out_w, kh, kw),
        limit=_INT32_MAX,
        name="cached spatial schedule",
    )
    try:
        geometry = _cached_geometry(
            input_shape,
            output_shape,
            (kh, kw),
            stride,
            padding,
            dilation,
            groups,
        )
    except ExactSparseConvAffineCoreError:
        raise
    except (MemoryError, OverflowError, ValueError) as exc:
        raise ExactSparseConvAffineCoreError(
            "CONV geometry construction exceeded checked resources"
        ) from exc

    bias_value = params.get("bias")
    output_area = out_h * out_w
    if bias_value is None:
        bias_vector = np.zeros(batch * out_ch * output_area, dtype=np.float64)
    else:
        bias_channels = _snapshot_f64(bias_value, name="bias").reshape(-1)
        if bias_channels.size != out_ch:
            raise ExactSparseConvAffineCoreError(
                "bias size disagrees with output channels"
            )
        bias_vector = np.tile(np.repeat(bias_channels, output_area), batch)
    return (
        weight,
        _private_array(bias_vector, dtype=np.dtype(np.float64)),
        geometry,
    )


def _weight_digest(weight: np.ndarray, geometry_digest: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"act.exact_sparse_conv_affine.weight.v1")
    digest.update(geometry_digest.encode("ascii"))
    _digest_array(digest, weight)
    return digest.hexdigest()


def _sequential_row_sum(terms: np.ndarray) -> np.ndarray:
    """Match CSR SpMV's canonical left-to-right binary64 accumulation."""

    if terms.ndim != 2 or terms.shape[1] <= 0:
        raise ExactSparseConvAffineCoreInternalError("invalid row-reduction slab")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cumulative = np.cumsum(terms, axis=1, dtype=np.float64)
    out = np.asarray(cumulative[:, -1], dtype=np.float64).copy()
    # CSR SpMV starts each row from +0.  NumPy's first cumsum element has no
    # preceding +0, so normalize the only observable difference: signed zero.
    out[out == 0.0] = 0.0
    return out


def apply_exact_sparse_conv_affine_core(
    layer: Any,
    source: ExactRowLocalAffineSource,
    *,
    expected_stable_column_ids: Any,
    return_receipt: bool = False,
) -> ExactSparseConvLinearCore | Tuple[
    ExactSparseConvLinearCore, ExactSparseConvAffineCoreReceipt
]:
    """Apply one CONV snapshot to an admitted source without W or SpGEMM."""

    admitted = _validate_source(
        source, expected_stable_column_ids=expected_stable_column_ids
    )
    weight, bias, geometry = _layer_snapshot(layer)
    batch, in_ch, in_h, in_w = geometry.input_shape
    _out_batch, out_ch, out_h, out_w = geometry.output_shape
    if admitted.size != batch * in_ch * in_h * in_w:
        raise ExactSparseConvAffineCoreError(
            "source row count disagrees with flattened CONV input"
        )
    n_generators = int(admitted.generators.shape[1])
    output_area = out_h * out_w
    input_area = in_h * in_w
    in_ch_per_group = int(weight.shape[1])
    kernel_area = int(weight.shape[2] * weight.shape[3])
    out_ch_per_group = out_ch // geometry.groups
    flat_weight = weight.reshape(out_ch, in_ch_per_group, kernel_area)
    valid = geometry.input_spatial_by_output >= 0
    safe_spatial = np.maximum(geometry.input_spatial_by_output, 0)
    entries_per_output_channel = max(
        1, output_area * in_ch_per_group * kernel_area
    )
    if entries_per_output_channel > _MAX_COEFFICIENT_SLAB_ENTRIES:
        raise ExactSparseConvAffineCoreError(
            "one output-channel coefficient slab exceeds the bounded core cap"
        )

    source_rows_by_group = []
    for group in range(geometry.groups):
        channels = (
            group * in_ch_per_group + np.arange(in_ch_per_group, dtype=np.int64)
        )
        source_rows_by_group.append(
            channels[None, :, None] * input_area + safe_spatial[:, None, :]
        )

    n_output_rows = batch * out_ch * output_area
    center_linear = np.empty(n_output_rows, dtype=np.float64)
    transformed_mass = np.empty(n_output_rows, dtype=np.float64)
    propagated_error = np.empty(n_output_rows, dtype=np.float64)
    fanin = np.empty(n_output_rows, dtype=np.float64)
    generator_counts = np.empty(n_output_rows, dtype=np.int32)
    generator_blocks = []
    operator_nnz = 0
    output_generator_nnz = 0
    nonnegative_rhs = np.column_stack(
        (admitted.source_mass, admitted.error)
    )

    try:
        # Bound the live coefficient slab while amortizing Python dispatch and
        # NumPy allocation across adjacent output channels.  One million
        # entries keeps each binary64 work array at or below roughly 8 MiB.
        output_channel_chunk = max(
            1, _MAX_COEFFICIENT_SLAB_ENTRIES // entries_per_output_channel
        )
        for group in range(geometry.groups):
            group_start = group * out_ch_per_group
            group_stop = group_start + out_ch_per_group
            base_source_rows = source_rows_by_group[group]
            for co_start in range(group_start, group_stop, output_channel_chunk):
                co_stop = min(group_stop, co_start + output_channel_chunk)
                chunk_channels = co_stop - co_start
                # One gather of each weight in this channel chunk.  This slab
                # is consumed by all five core outputs before it is released.
                values = np.take(
                    flat_weight[co_start:co_stop],
                    geometry.kernel_gather_by_output,
                    axis=2,
                ).transpose(0, 2, 1, 3)
                weight_keep = (
                    valid[None, :, None, :] & (values != 0.0)
                )
                flat_values = values.reshape(
                    chunk_channels * output_area, -1
                )
                flat_keep = weight_keep.reshape(
                    chunk_channels * output_area, -1
                )
                local_fanin = np.count_nonzero(flat_keep, axis=1).astype(
                    np.float64, copy=False
                )
                local_operator_nnz = int(
                    np.sum(local_fanin, dtype=np.float64)
                )
                if local_operator_nnz > _INT32_MAX:
                    raise ExactSparseConvAffineCoreError(
                        "one traversal slab exceeds int32 CSR workspace"
                    )
                local_indptr64 = np.empty(
                    chunk_channels * output_area + 1, dtype=np.int64
                )
                local_indptr64[0] = 0
                np.cumsum(
                    local_fanin,
                    dtype=np.int64,
                    out=local_indptr64[1:],
                )
                if int(local_indptr64[-1]) != local_operator_nnz:
                    raise ExactSparseConvAffineCoreInternalError(
                        "operator row counts changed inside a traversal slab"
                    )
                local_indptr = np.asarray(local_indptr64, dtype=np.int32)
                operator_data = np.asarray(
                    flat_values[flat_keep], dtype=np.float64
                )
                repeated_base_rows = np.broadcast_to(
                    base_source_rows.reshape(1, output_area, -1),
                    (chunk_channels, output_area, in_ch_per_group * kernel_area),
                ).reshape(chunk_channels * output_area, -1)
                base_operator_indices = np.asarray(
                    repeated_base_rows[flat_keep], dtype=np.int32
                )

                for n in range(batch):
                    row_start = (n * out_ch + co_start) * output_area
                    row_stop = (n * out_ch + co_stop) * output_area
                    fanin[row_start:row_stop] = local_fanin
                    operator_nnz += local_operator_nnz
                    if operator_nnz > _INT32_MAX:
                        raise ExactSparseConvAffineCoreError(
                            "CONV operator nonzeros exceed int32 CSR domain"
                        )
                    operator_indices = np.asarray(
                        base_operator_indices + n * in_ch * input_area,
                        dtype=np.int32,
                    )
                    local_center = np.zeros(
                        chunk_channels * output_area, dtype=np.float64
                    )
                    # Invoke only SciPy's compiled CSR-vector reduction over
                    # the transient canonical arrays.  No sparse matrix
                    # object (and in particular no CONV W) is constructed.
                    _scipy_sparsetools.csr_matvec(
                        chunk_channels * output_area,
                        admitted.size,
                        local_indptr,
                        operator_indices,
                        operator_data,
                        admitted.center,
                        local_center,
                    )
                    if not np.all(np.isfinite(local_center)):
                        raise ExactSparseConvAffineCoreError(
                            "CONV center propagation overflowed or became NaN"
                        )
                    center_linear[row_start:row_stop] = local_center

                    raw_nonnegative = np.zeros(
                        (chunk_channels * output_area, 2), dtype=np.float64
                    )
                    _scipy_sparsetools.csr_matvecs(
                        chunk_channels * output_area,
                        admitted.size,
                        2,
                        local_indptr,
                        operator_indices,
                        np.abs(operator_data),
                        nonnegative_rhs.ravel(),
                        raw_nonnegative.ravel(),
                    )
                    raw_mass = raw_nonnegative[:, 0]
                    raw_error = raw_nonnegative[:, 1]
                    nonempty_rows = local_fanin > 0.0
                    mass_active = np.zeros(
                        chunk_channels * output_area, dtype=np.bool_
                    )
                    error_active = np.zeros_like(mass_active)
                    if np.any(nonempty_rows):
                        starts = local_indptr64[:-1][nonempty_rows]
                        mass_flags = admitted.source_mass[operator_indices] > 0.0
                        error_flags = admitted.error[operator_indices] > 0.0
                        mass_active[nonempty_rows] = np.logical_or.reduceat(
                            mass_flags, starts
                        )
                        error_active[nonempty_rows] = np.logical_or.reduceat(
                            error_flags, starts
                        )
                    transformed_mass[row_start:row_stop] = _inflate_nonnegative(
                        raw_mass,
                        2.0 * local_fanin + 2.0,
                        active=mass_active,
                        name="conv.transformed_mass",
                    )
                    propagated_error[row_start:row_stop] = _inflate_nonnegative(
                        raw_error,
                        2.0 * local_fanin + 2.0,
                        active=error_active,
                        name="conv.propagated_error",
                    )

                    mapped_columns = admitted.row_to_generator_column[
                        operator_indices
                    ]
                    live = mapped_columns >= 0
                    with np.errstate(
                        over="ignore", invalid="ignore", under="ignore"
                    ):
                        products = (
                            operator_data[live]
                            * admitted.row_scale[operator_indices[live]]
                        )
                    if not np.all(np.isfinite(products)):
                        raise ExactSparseConvAffineCoreError(
                            "row-local generator product overflowed or became NaN"
                        )
                    nonzero_products = products != 0.0
                    retained_positions = np.flatnonzero(live)[nonzero_products]
                    keep_generator = np.zeros(
                        local_operator_nnz, dtype=np.bool_
                    )
                    keep_generator[retained_positions] = True
                    prefix = np.empty(local_operator_nnz + 1, dtype=np.int64)
                    prefix[0] = 0
                    np.cumsum(keep_generator, dtype=np.int64, out=prefix[1:])
                    counts = (
                        prefix[local_indptr64[1:]]
                        - prefix[local_indptr64[:-1]]
                    ).astype(np.int32, copy=False)
                    generator_counts[row_start:row_stop] = counts
                    emitted = int(np.sum(counts, dtype=np.int64))
                    output_generator_nnz += emitted
                    if output_generator_nnz > _INT32_MAX:
                        raise ExactSparseConvAffineCoreError(
                            "output generator nonzeros exceed int32 CSR domain"
                        )
                    if emitted:
                        columns = np.asarray(
                            mapped_columns[keep_generator], dtype=np.int32
                        )
                        data = np.asarray(
                            products[nonzero_products], dtype=np.float64
                        )
                        if not admitted.positional_mapping_monotone:
                            local_rows = np.repeat(
                                np.arange(
                                    chunk_channels * output_area,
                                    dtype=np.int32,
                                ),
                                counts,
                            )
                            order = np.lexsort((columns, local_rows))
                            columns = columns[order]
                            data = data[order]
                        generator_blocks.append(
                            (int(row_start), columns, data)
                        )
    except ExactSparseConvAffineCoreError:
        raise
    except ExactSparseConvAffineCoreInternalError:
        raise
    except (MemoryError, OverflowError, ValueError, IndexError) as exc:
        raise ExactSparseConvAffineCoreError(
            "direct affine-core traversal failed closed"
        ) from exc
    except Exception as exc:
        raise ExactSparseConvAffineCoreInternalError(
            "compiled affine-core traversal violated its internal contract"
        ) from exc

    indptr64 = np.empty(n_output_rows + 1, dtype=np.int64)
    indptr64[0] = 0
    np.cumsum(generator_counts, dtype=np.int64, out=indptr64[1:])
    if int(indptr64[-1]) != output_generator_nnz:
        raise ExactSparseConvAffineCoreInternalError(
            "generator row counts changed after traversal"
        )
    if generator_blocks:
        generator_blocks.sort(key=lambda block: block[0])
        output_indices = np.concatenate(
            [block[1] for block in generator_blocks]
        )
        output_data = np.concatenate([block[2] for block in generator_blocks])
    else:
        output_indices = np.zeros(0, dtype=np.int32)
        output_data = np.zeros(0, dtype=np.float64)
    frozen_data = _private_array(output_data, dtype=np.dtype(np.float64))
    frozen_indices = _private_array(output_indices, dtype=np.dtype(np.int32))
    frozen_indptr = _private_array(indptr64, dtype=np.dtype(np.int32))
    generators = sp.csr_matrix(
        (frozen_data, frozen_indices, frozen_indptr),
        shape=(n_output_rows, n_generators),
        dtype=np.float64,
        copy=False,
    )
    if (
        generators.nnz != output_generator_nnz
        or generators.data.flags.writeable
        or generators.indices.flags.writeable
        or generators.indptr.flags.writeable
    ):
        raise ExactSparseConvAffineCoreInternalError(
            "output generator CSR lost immutable exact storage"
        )
    # Validate canonical order from bytes, never from cached SciPy flags.
    _validate_canonical_csr_storage(
        generators, name="core.generators", copy=False
    )

    frozen_center = _private_array(center_linear, dtype=np.dtype(np.float64))
    frozen_mass = _private_array(transformed_mass, dtype=np.dtype(np.float64))
    frozen_error = _private_array(propagated_error, dtype=np.dtype(np.float64))
    frozen_fanin = _private_array(fanin, dtype=np.dtype(np.float64))
    frozen_ids = _private_array(
        admitted.stable_column_ids, dtype=np.dtype(np.int64)
    )
    weight_digest = _weight_digest(weight, geometry.digest)
    provisional = ExactSparseConvLinearCore(
        center_linear=frozen_center,
        generators=generators,
        transformed_mass=frozen_mass,
        propagated_error=frozen_error,
        fanin=frozen_fanin,
        bias=bias,
        stable_column_ids=frozen_ids,
        source_digest=admitted.digest,
        geometry_digest=geometry.digest,
        weight_digest=weight_digest,
        input_shape=geometry.input_shape,
        output_shape=geometry.output_shape,
        operator_nnz=int(operator_nnz),
        digest="",
    )
    core = ExactSparseConvLinearCore(
        center_linear=provisional.center_linear,
        generators=provisional.generators,
        transformed_mass=provisional.transformed_mass,
        propagated_error=provisional.propagated_error,
        fanin=provisional.fanin,
        bias=provisional.bias,
        stable_column_ids=provisional.stable_column_ids,
        source_digest=provisional.source_digest,
        geometry_digest=provisional.geometry_digest,
        weight_digest=provisional.weight_digest,
        input_shape=provisional.input_shape,
        output_shape=provisional.output_shape,
        operator_nnz=provisional.operator_nnz,
        digest=_core_digest(provisional),
    )
    if not return_receipt:
        return core
    receipt = ExactSparseConvAffineCoreReceipt(
        source_digest=admitted.digest,
        geometry_digest=geometry.digest,
        weight_digest=weight_digest,
        core_digest=core.digest,
        input_shape=geometry.input_shape,
        output_shape=geometry.output_shape,
        source_generator_nnz=int(admitted.generators.nnz),
        output_generator_nnz=int(generators.nnz),
        operator_nnz=int(operator_nnz),
    )
    return core, receipt


def _validate_core(core: Any) -> ExactSparseConvLinearCore:
    if type(core) is not ExactSparseConvLinearCore or core.schema != _CORE_SCHEMA:
        raise ExactSparseConvAffineCoreError("core is not a linear-core snapshot")
    n_rows = int(getattr(core.center_linear, "size", -1))
    for name, value in (
        ("center_linear", core.center_linear),
        ("transformed_mass", core.transformed_mass),
        ("propagated_error", core.propagated_error),
        ("fanin", core.fanin),
        ("bias", core.bias),
    ):
        _require_private_array(
            value,
            dtype=np.dtype(np.float64),
            shape=(n_rows,),
            name=f"core.{name}",
        )
    matrix = _validate_canonical_csr_storage(
        core.generators, name="core.generators", copy=False
    )
    if matrix.shape[0] != n_rows:
        raise ExactSparseConvAffineCoreError("core generator row count changed")
    _require_private_array(
        core.stable_column_ids,
        dtype=np.dtype(np.int64),
        shape=(int(matrix.shape[1]),),
        name="core.stable_column_ids",
    )
    if (
        np.any(core.transformed_mass < 0.0)
        or np.any(core.propagated_error < 0.0)
        or np.any(core.fanin < 0.0)
        or type(core.operator_nnz) is not int
        or core.operator_nnz < 0
        or type(core.digest) is not str
        or len(core.digest) != 64
        or _core_digest(core) != core.digest
    ):
        raise ExactSparseConvAffineCoreError("linear-core integrity check failed")
    return core


def finalize_exact_sparse_conv_affine_core(
    core: ExactSparseConvLinearCore,
    *,
    source_affine_depth: int = 0,
) -> ExactSparseConvAffineResult:
    """Purely add the bound bias and Operator-HZ arithmetic-error allowance."""

    admitted = _validate_core(core)
    depth = _builtin_int(source_affine_depth, name="source_affine_depth")
    if depth < 0:
        raise ExactSparseConvAffineCoreError("source_affine_depth must be nonnegative")
    center = np.asarray(
        admitted.center_linear + admitted.bias, dtype=np.float64
    ).reshape(-1)
    if not np.all(np.isfinite(center)):
        raise ExactSparseConvAffineCoreError(
            "CONV center plus bias overflowed or became NaN"
        )
    arithmetic_mass = _nonnegative_sum_upper(
        admitted.transformed_mass,
        np.abs(admitted.bias),
        name="conv.arithmetic_mass",
    )
    arithmetic_error = (
        _gamma_ops(2.0 * admitted.fanin + 2.0, name="conv.gamma")
        * arithmetic_mass
    )
    arithmetic_error = _inflate_nonnegative(
        arithmetic_error,
        4,
        active=arithmetic_mass > 0.0,
        name="conv.arithmetic_error",
    )
    total_error = _nonnegative_sum_upper(
        admitted.propagated_error,
        arithmetic_error,
        name="conv.total_error",
    )
    frozen_center = _private_array(center, dtype=np.dtype(np.float64))
    frozen_error = _private_array(total_error, dtype=np.dtype(np.float64))
    frozen_ids = _private_array(
        admitted.stable_column_ids, dtype=np.dtype(np.int64)
    )
    provisional = ExactSparseConvAffineResult(
        center=frozen_center,
        generators=admitted.generators,
        error=frozen_error,
        stable_column_ids=frozen_ids,
        affine_depth=depth + 1,
        core_digest=admitted.digest,
        digest="",
    )
    return ExactSparseConvAffineResult(
        center=provisional.center,
        generators=provisional.generators,
        error=provisional.error,
        stable_column_ids=provisional.stable_column_ids,
        affine_depth=provisional.affine_depth,
        core_digest=provisional.core_digest,
        digest=_result_digest(provisional),
    )


def exact_sparse_conv_affine_from_layer(
    layer: Any,
    source: ExactRowLocalAffineSource,
    *,
    expected_stable_column_ids: Any,
    source_affine_depth: int = 0,
    return_receipt: bool = False,
) -> ExactSparseConvAffineResult | Tuple[
    ExactSparseConvAffineResult, ExactSparseConvAffineCoreReceipt
]:
    """Compose the disconnected research core and non-authoritative finalize."""

    core_or_pair = apply_exact_sparse_conv_affine_core(
        layer,
        source,
        expected_stable_column_ids=expected_stable_column_ids,
        return_receipt=return_receipt,
    )
    if return_receipt:
        core, receipt = core_or_pair  # type: ignore[misc]
        return (
            finalize_exact_sparse_conv_affine_core(
                core, source_affine_depth=source_affine_depth
            ),
            receipt,
        )
    return finalize_exact_sparse_conv_affine_core(
        core_or_pair,  # type: ignore[arg-type]
        source_affine_depth=source_affine_depth,
    )


def clear_exact_sparse_conv_affine_geometry_cache() -> None:
    """Clear only this module's immutable value-keyed geometry cache."""

    _cached_geometry.cache_clear()


def exact_sparse_conv_affine_geometry_cache_info() -> Any:
    """Return standard read-only ``functools`` cache accounting."""

    return _cached_geometry.cache_info()


__all__ = [
    "ExactRowLocalAffineSource",
    "ExactSparseConvAffineCoreError",
    "ExactSparseConvAffineCoreInternalError",
    "ExactSparseConvAffineCoreReceipt",
    "ExactSparseConvAffineResult",
    "ExactSparseConvLinearCore",
    "RowLocalNotApplicable",
    "apply_exact_sparse_conv_affine_core",
    "clear_exact_sparse_conv_affine_geometry_cache",
    "exact_sparse_conv_affine_from_layer",
    "exact_sparse_conv_affine_geometry_cache_info",
    "finalize_exact_sparse_conv_affine_core",
    "prepare_exact_row_local_affine_source",
]
