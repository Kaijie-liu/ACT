#!/usr/bin/env python3
# ===- exact_sparse_conv_csr_candidate.py - exact cached CONV CSR --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Disconnected exact sparse-CONV CSR construction candidate.

The production sparse-HybridZ path currently constructs a CONV2D matrix with
Python loops over every output pixel and every kernel coefficient.  This
candidate separates the immutable *spatial incidence topology* from mutable
layer coefficients.  Geometry is cached by value, while every invocation
snapshots the current weight and bias.  Exact CSR row lengths are obtained by
multiplying the per-kernel nonzero-channel counts by the cached spatial-valid
incidence mask.  The constructor then fills preallocated ``indptr``,
``indices``, and ``data`` directly in canonical row/column order; it allocates
no COO row-triplet array and performs no sparse sort or coefficient reduction.
The emitted binary64 affine map is identical to the established constructor.

Only a linear representation is changed: no network or feasible-set
constraint is added, removed, or relaxed.  The module has no proof/verdict
authority and is deliberately disconnected from ``tf_cnn`` and verifier
dispatch.  It contains no triangle relaxation, branch-and-bound, backward
propagation, dual operation, solver call, dense CONV matrix, or torch CONV.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import operator
from typing import Any, Tuple

import numpy as np
import scipy.sparse as sp


_SCHEMA = "act.exact_sparse_conv_csr_candidate.v2"
_TOPOLOGY_SCHEMA = "act.exact_sparse_conv_spatial_topology.v1"
_INT32_MAX = int(np.iinfo(np.int32).max)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


class ExactSparseConvCandidateError(ValueError):
    """Fail-closed rejection of malformed or unsupported CONV geometry."""


class RowLocalGeneratorIneligible(ExactSparseConvCandidateError):
    """The exact fast path is inapplicable; callers must use ordinary SpGEMM."""


def _builtin_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ExactSparseConvCandidateError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ExactSparseConvCandidateError(f"{name} must be an integer") from exc
    result = int(result)
    if result < _INT64_MIN or result > _INT64_MAX:
        raise ExactSparseConvCandidateError(f"{name} exceeds signed int64")
    return result


def _checked_i64(value: int, *, name: str) -> int:
    """Admit one already-integral Python result to signed-int64 arithmetic."""

    if type(value) is not int or value < _INT64_MIN or value > _INT64_MAX:
        raise ExactSparseConvCandidateError(
            f"{name} exceeds checked signed-int64 arithmetic"
        )
    return value


def _checked_nonnegative_product(
    values: Tuple[int, ...],
    *,
    limit: int,
    name: str,
) -> int:
    """Multiply nonnegative builtin integers without exceeding ``limit``."""

    product = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise ExactSparseConvCandidateError(
                f"{name} contains a non-integer or negative factor"
            )
        if value and product > limit // value:
            raise ExactSparseConvCandidateError(
                f"{name} exceeds checked limit {limit}"
            )
        product *= value
    return product


def _checked_conv_output_extent(
    input_extent: int,
    kernel_extent: int,
    stride: int,
    padding: int,
    dilation: int,
    *,
    axis: str,
) -> Tuple[int, int]:
    """Return output/effective extent with every intermediate in int64."""

    twice_padding = _checked_i64(2 * padding, name=f"{axis}.2*padding")
    effective_kernel_tail = _checked_i64(
        dilation * (kernel_extent - 1),
        name=f"{axis}.dilation*(kernel-1)",
    )
    padded_input = _checked_i64(
        input_extent + twice_padding, name=f"{axis}.input+2*padding"
    )
    numerator = _checked_i64(
        padded_input - effective_kernel_tail,
        name=f"{axis}.padded_input-effective_kernel_tail",
    )
    numerator = _checked_i64(numerator - 1, name=f"{axis}.output_numerator")
    output_extent = _checked_i64(
        numerator // stride + 1, name=f"{axis}.output_extent"
    )

    # Mirror the NumPy construction order used by the cached topology and
    # prove its two extreme affine indices cannot wrap signed int64.
    output_stride_tail = _checked_i64(
        max(0, output_extent - 1) * stride,
        name=f"{axis}.output_tail*stride",
    )
    shifted_output_tail = _checked_i64(
        output_stride_tail - padding,
        name=f"{axis}.output_tail*stride-padding",
    )
    _checked_i64(-padding, name=f"{axis}.-padding")
    _checked_i64(
        shifted_output_tail + effective_kernel_tail,
        name=f"{axis}.max_spatial_index",
    )
    return output_extent, effective_kernel_tail


def _pair(value: Any, *, name: str) -> Tuple[int, int]:
    try:
        scalar = _builtin_int(value, name=name)
    except ExactSparseConvCandidateError:
        try:
            items = tuple(value)
        except TypeError as exc:
            raise ExactSparseConvCandidateError(
                f"{name} must be an integer or a length-two sequence"
            ) from exc
        if len(items) != 2:
            raise ExactSparseConvCandidateError(
                f"{name} must be an integer or a length-two sequence"
            )
        return (
            _builtin_int(items[0], name=f"{name}[0]"),
            _builtin_int(items[1], name=f"{name}[1]"),
        )
    return scalar, scalar


def _shape4(value: Any, *, name: str) -> Tuple[int, int, int, int]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ExactSparseConvCandidateError(
            f"{name} must be a three- or four-dimensional NCHW shape"
        ) from exc
    dims = tuple(
        _builtin_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(items)
    )
    if len(dims) == 3:
        dims = (1, *dims)
    if len(dims) != 4 or min(dims) <= 0:
        raise ExactSparseConvCandidateError(
            f"{name} must be a positive three- or four-dimensional NCHW shape"
        )
    return dims


def _numpy_snapshot(value: Any, *, name: str) -> np.ndarray:
    """Take an owning C-order float64 snapshot without importing torch."""

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
        raise ExactSparseConvCandidateError(
            f"{name} cannot be converted to a host array"
        ) from exc
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ExactSparseConvCandidateError(f"{name} must be real-valued")
    try:
        snapshot = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExactSparseConvCandidateError(
            f"{name} cannot be represented as binary64"
        ) from exc
    if not np.all(np.isfinite(snapshot)):
        raise ExactSparseConvCandidateError(
            f"{name} contains NaN or infinity"
        )
    return snapshot


def _readonly_copy(array: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Return an ndarray backed by immutable ``bytes``, not a mutable owner."""

    normalized = np.asarray(array, dtype=dtype, order="C")
    return np.frombuffer(normalized.tobytes(order="C"), dtype=dtype).reshape(
        normalized.shape
    )


@dataclass(frozen=True, eq=False)
class SpatialKernelFrame:
    """Immutable valid spatial incidences for one kernel offset."""

    kernel_row: int
    kernel_col: int
    output_spatial: np.ndarray
    input_spatial: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.kernel_row) is not int
            or type(self.kernel_col) is not int
            or self.kernel_row < 0
            or self.kernel_col < 0
            or type(self.output_spatial) is not np.ndarray
            or type(self.input_spatial) is not np.ndarray
            or self.output_spatial.dtype != np.dtype(np.int32)
            or self.input_spatial.dtype != np.dtype(np.int32)
            or self.output_spatial.ndim != 1
            or self.input_spatial.ndim != 1
            or self.output_spatial.shape != self.input_spatial.shape
            or self.output_spatial.flags.writeable
            or self.input_spatial.flags.writeable
        ):
            raise ExactSparseConvCandidateError("malformed spatial kernel frame")


@dataclass(frozen=True, eq=False)
class ExactConvSpatialTopology:
    """Value-keyed, immutable spatial topology shared by matching layers."""

    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    kernel: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int
    frames: Tuple[SpatialKernelFrame, ...]
    kernel_gather_by_output: np.ndarray
    input_spatial_by_output: np.ndarray
    incidence_count: int
    topology_nbytes: int
    digest: str
    schema: str = _TOPOLOGY_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.input_shape) is not tuple
            or type(self.output_shape) is not tuple
            or type(self.kernel) is not tuple
            or type(self.stride) is not tuple
            or type(self.padding) is not tuple
            or type(self.dilation) is not tuple
            or type(self.groups) is not int
            or type(self.frames) is not tuple
            or type(self.kernel_gather_by_output) is not np.ndarray
            or self.kernel_gather_by_output.dtype != np.dtype(np.int32)
            or self.kernel_gather_by_output.ndim != 2
            or self.kernel_gather_by_output.flags.writeable
            or type(self.input_spatial_by_output) is not np.ndarray
            or self.input_spatial_by_output.dtype != np.dtype(np.int32)
            or self.input_spatial_by_output.shape
            != self.kernel_gather_by_output.shape
            or self.input_spatial_by_output.flags.writeable
            or type(self.incidence_count) is not int
            or type(self.topology_nbytes) is not int
            or type(self.digest) is not str
            or len(self.digest) != 64
            or self.schema != _TOPOLOGY_SCHEMA
        ):
            raise ExactSparseConvCandidateError("malformed CONV topology snapshot")


def _topology_digest(
    *,
    input_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int, int, int],
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    frames: Tuple[SpatialKernelFrame, ...],
    kernel_gather_by_output: np.ndarray,
    input_spatial_by_output: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(_TOPOLOGY_SCHEMA.encode("ascii"))
    for values in (
        input_shape,
        output_shape,
        kernel,
        stride,
        padding,
        dilation,
        (groups,),
    ):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    for frame in frames:
        digest.update(np.asarray((frame.kernel_row, frame.kernel_col), dtype="<i8").tobytes())
        digest.update(frame.output_spatial.tobytes())
        digest.update(frame.input_spatial.tobytes())
    digest.update(kernel_gather_by_output.tobytes())
    digest.update(input_spatial_by_output.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _cached_spatial_topology(
    input_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int, int, int],
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> ExactConvSpatialTopology:
    _batch, _in_ch, in_h, in_w = input_shape
    _out_batch, _out_ch, out_h, out_w = output_shape
    kh, kw = kernel
    output_h = np.arange(out_h, dtype=np.int64)
    output_w = np.arange(out_w, dtype=np.int64)
    frames = []
    incidence_count = 0
    topology_nbytes = 0
    kernel_gather_grid = np.zeros((out_h * out_w, kh * kw), dtype=np.int32)
    input_spatial_grid = np.full(
        (out_h * out_w, kh * kw), -1, dtype=np.int32
    )
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
            output_spatial = _readonly_copy(
                (oh[:, None] * out_w + ow[None, :]).reshape(-1),
                dtype=np.dtype(np.int32),
            )
            input_spatial = _readonly_copy(
                (ih[:, None] * in_w + iw[None, :]).reshape(-1),
                dtype=np.dtype(np.int32),
            )
            frame = SpatialKernelFrame(
                kernel_row=rr,
                kernel_col=cc,
                output_spatial=output_spatial,
                input_spatial=input_spatial,
            )
            frames.append(frame)
            kernel_index = rr * kw + cc
            kernel_gather_grid[output_spatial, kernel_index] = kernel_index
            input_spatial_grid[output_spatial, kernel_index] = input_spatial
            incidence_count += int(output_spatial.size)
            topology_nbytes += int(output_spatial.nbytes + input_spatial.nbytes)
    frozen_frames = tuple(frames)
    frozen_kernel_gather = _readonly_copy(
        kernel_gather_grid, dtype=np.dtype(np.int32)
    )
    frozen_input_spatial = _readonly_copy(
        input_spatial_grid, dtype=np.dtype(np.int32)
    )
    topology_nbytes += int(
        frozen_kernel_gather.nbytes + frozen_input_spatial.nbytes
    )
    return ExactConvSpatialTopology(
        input_shape=input_shape,
        output_shape=output_shape,
        kernel=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        frames=frozen_frames,
        kernel_gather_by_output=frozen_kernel_gather,
        input_spatial_by_output=frozen_input_spatial,
        incidence_count=incidence_count,
        topology_nbytes=topology_nbytes,
        digest=_topology_digest(
            input_shape=input_shape,
            output_shape=output_shape,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            frames=frozen_frames,
            kernel_gather_by_output=frozen_kernel_gather,
            input_spatial_by_output=frozen_input_spatial,
        ),
    )


@dataclass(frozen=True)
class ExactSparseConvCSRReceipt:
    """Non-authoritative accounting for one candidate construction."""

    geometry_digest: str
    matrix_shape: Tuple[int, int]
    matrix_nnz: int
    topology_incidence_count: int
    topology_nbytes: int
    triplet_nbytes: int
    csr_nbytes: int
    peak_workspace_upper_bytes: int
    construction_mode: str = "direct_canonical_csr_v2"
    coo_row_triplets_materialized: bool = False
    immutable_topology_cache: bool = True
    candidate_authoritative: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False
    exact_affine_map: bool = True
    uses_dense_matrix: bool = False
    uses_torch_conv: bool = False
    uses_triangle_relaxation: bool = False
    uses_branch_and_bound: bool = False
    uses_backward_or_dual: bool = False
    uses_solver: bool = False
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.geometry_digest) is not str
            or len(self.geometry_digest) != 64
            or type(self.matrix_shape) is not tuple
            or len(self.matrix_shape) != 2
            or any(type(value) is not int or value < 0 for value in self.matrix_shape)
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.matrix_nnz,
                    self.topology_incidence_count,
                    self.topology_nbytes,
                    self.triplet_nbytes,
                    self.csr_nbytes,
                    self.peak_workspace_upper_bytes,
                )
            )
            or self.construction_mode != "direct_canonical_csr_v2"
            or self.coo_row_triplets_materialized is not False
            or self.immutable_topology_cache is not True
            or type(self.candidate_authoritative) is not bool
            or type(self.proof_authority) is not bool
            or type(self.verdict_authority) is not bool
            or self.candidate_authoritative
            or self.proof_authority
            or self.verdict_authority
            or self.exact_affine_map is not True
            or self.uses_dense_matrix
            or self.uses_torch_conv
            or self.uses_triangle_relaxation
            or self.uses_branch_and_bound
            or self.uses_backward_or_dual
            or self.uses_solver
            or self.schema != _SCHEMA
        ):
            raise ExactSparseConvCandidateError("malformed candidate receipt")


@dataclass(frozen=True, eq=False)
class RowLocalGeneratorPlan:
    """Immutable source-row to stable generator-column/value snapshot."""

    source_row_count: int
    generator_column_count: int
    row_to_generator_column: np.ndarray
    row_scale: np.ndarray
    stable_column_ids: np.ndarray
    mapped_row_count: int
    all_rows_mapped: bool
    positional_mapping_monotone: bool
    digest: str
    schema: str = "act.exact_sparse_conv.row_local_plan.v1"

    def __post_init__(self) -> None:
        if (
            type(self.source_row_count) is not int
            or self.source_row_count < 0
            or type(self.generator_column_count) is not int
            or self.generator_column_count < 0
            or type(self.row_to_generator_column) is not np.ndarray
            or self.row_to_generator_column.dtype != np.dtype(np.int64)
            or self.row_to_generator_column.shape != (self.source_row_count,)
            or self.row_to_generator_column.flags.writeable
            or type(self.row_scale) is not np.ndarray
            or self.row_scale.dtype != np.dtype(np.float64)
            or self.row_scale.shape != (self.source_row_count,)
            or self.row_scale.flags.writeable
            or type(self.stable_column_ids) is not np.ndarray
            or self.stable_column_ids.dtype != np.dtype(np.int64)
            or self.stable_column_ids.shape != (self.generator_column_count,)
            or self.stable_column_ids.flags.writeable
            or type(self.mapped_row_count) is not int
            or self.mapped_row_count < 0
            or self.mapped_row_count > self.source_row_count
            or type(self.all_rows_mapped) is not bool
            or type(self.positional_mapping_monotone) is not bool
            or type(self.digest) is not str
            or len(self.digest) != 64
            or self.schema != "act.exact_sparse_conv.row_local_plan.v1"
        ):
            raise ExactSparseConvCandidateError("malformed row-local plan")


@dataclass(frozen=True)
class RowLocalGeneratorApplyReceipt:
    """Non-authoritative accounting for direct or fused row-local apply."""

    mode: str
    stable_mapping_digest: str
    source_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    source_generator_nnz: int
    conv_operator_nnz: int
    output_generator_nnz: int
    mapped_product_count: int
    candidate_authoritative: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False
    exact_affine_map: bool = True
    row_local_eligible: bool = True
    uses_dense_matrix: bool = False
    uses_torch_conv: bool = False
    uses_triangle_relaxation: bool = False
    uses_branch_and_bound: bool = False
    uses_backward_or_dual: bool = False
    uses_solver: bool = False
    schema: str = "act.exact_sparse_conv.row_local_apply.v1"

    def __post_init__(self) -> None:
        if (
            self.mode not in {
                "standalone_csr_relabel_v1",
                "fused_topology_to_generators_v1",
            }
            or type(self.stable_mapping_digest) is not str
            or len(self.stable_mapping_digest) != 64
            or type(self.source_shape) is not tuple
            or len(self.source_shape) != 2
            or type(self.output_shape) is not tuple
            or len(self.output_shape) != 2
            or any(
                type(value) is not int or value < 0
                for value in (
                    *self.source_shape,
                    *self.output_shape,
                    self.source_generator_nnz,
                    self.conv_operator_nnz,
                    self.output_generator_nnz,
                    self.mapped_product_count,
                )
            )
            or self.candidate_authoritative
            or self.proof_authority
            or self.verdict_authority
            or self.exact_affine_map is not True
            or self.row_local_eligible is not True
            or self.uses_dense_matrix
            or self.uses_torch_conv
            or self.uses_triangle_relaxation
            or self.uses_branch_and_bound
            or self.uses_backward_or_dual
            or self.uses_solver
            or self.schema != "act.exact_sparse_conv.row_local_apply.v1"
        ):
            raise ExactSparseConvCandidateError("malformed row-local receipt")


def _require_canonical_finite_csr(value: Any, *, name: str) -> sp.csr_matrix:
    """Validate raw CSR arrays and return an immutable private snapshot.

    SciPy caches ``has_sorted_indices`` and ``has_canonical_format``.  Direct
    mutation of ``indices``/``indptr`` does not invalidate those flags, so no
    admission decision here consults either property.
    """

    if type(value) is not sp.csr_matrix:
        raise RowLocalGeneratorIneligible(f"{name} must be an exact csr_matrix")
    try:
        raw_shape = value.shape
        raw_data = value.data
        raw_indices = value.indices
        raw_indptr = value.indptr
    except (AttributeError, TypeError, ValueError) as exc:
        raise RowLocalGeneratorIneligible(f"{name} has malformed CSR storage") from exc
    if (
        type(raw_shape) is not tuple
        or len(raw_shape) != 2
        or any(type(dimension) is not int for dimension in raw_shape)
        or min(raw_shape) < 0
        or max(raw_shape) > _INT32_MAX
    ):
        raise RowLocalGeneratorIneligible(f"{name} has an invalid int32 shape")
    arrays = (raw_data, raw_indices, raw_indptr)
    if any(
        type(array) is not np.ndarray
        or array.ndim != 1
        or not array.flags.c_contiguous
        or any(stride < 0 for stride in array.strides)
        for array in arrays
    ):
        raise RowLocalGeneratorIneligible(
            f"{name} CSR arrays must be exact one-dimensional C-contiguous ndarrays"
        )
    if raw_data.dtype != np.dtype(np.float64):
        raise RowLocalGeneratorIneligible(f"{name} data must use native float64")
    allowed_index_dtypes = {np.dtype(np.int32), np.dtype(np.int64)}
    if (
        raw_indices.dtype not in allowed_index_dtypes
        or raw_indptr.dtype != raw_indices.dtype
    ):
        raise RowLocalGeneratorIneligible(
            f"{name} indices and indptr must share native int32 or int64 dtype"
        )

    # Copy first, then validate.  All subsequent reasoning is over these
    # private bytes-backed arrays, never caller-owned storage or cached flags.
    data = _readonly_copy(raw_data, dtype=np.dtype(np.float64))
    indices_wide = _readonly_copy(raw_indices, dtype=raw_indices.dtype)
    indptr_wide = _readonly_copy(raw_indptr, dtype=raw_indptr.dtype)
    n_rows, n_cols = raw_shape
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
        raise RowLocalGeneratorIneligible(f"{name} has invalid CSR indptr/storage")
    if (
        not np.all(np.isfinite(data))
        or np.any(data == 0.0)
        or np.any(indices_wide < 0)
        or np.any(indices_wide >= n_cols)
    ):
        raise RowLocalGeneratorIneligible(
            f"{name} has nonfinite/zero data or out-of-range column indices"
        )
    if nnz > 1:
        # Adjacent entries must increase unless the latter starts a new row.
        same_row = np.ones(nnz - 1, dtype=np.bool_)
        boundaries = indptr_wide[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < nnz)]
        if boundaries.size:
            same_row[np.asarray(boundaries, dtype=np.int64) - 1] = False
        if np.any(
            same_row
            & (indices_wide[1:] <= indices_wide[:-1])
        ):
            raise RowLocalGeneratorIneligible(
                f"{name} column indices must be strictly increasing within each row"
            )

    indices = _readonly_copy(indices_wide, dtype=np.dtype(np.int32))
    indptr = _readonly_copy(indptr_wide, dtype=np.dtype(np.int32))
    snapshot = sp.csr_matrix(
        (data, indices, indptr), shape=raw_shape, dtype=np.float64, copy=False
    )
    if (
        snapshot.shape != raw_shape
        or snapshot.nnz != nnz
        or snapshot.data.dtype != np.dtype(np.float64)
        or snapshot.indices.dtype != np.dtype(np.int32)
        or snapshot.indptr.dtype != np.dtype(np.int32)
        or snapshot.data.flags.writeable
        or snapshot.indices.flags.writeable
        or snapshot.indptr.flags.writeable
    ):
        raise RuntimeError("private CSR snapshot lost exact immutable storage")
    return snapshot


def _prepare_row_local_generator_plan_from_snapshot(
    matrix: sp.csr_matrix,
    *,
    stable_column_ids: Any,
) -> RowLocalGeneratorPlan:
    """Snapshot an injective, at-most-one-entry-per-row admitted ``G``.

    Injectivity is required in addition to row locality.  It proves that each
    emitted output coefficient has exactly one product contributor; hence the
    fast path never changes a floating-point reduction order.
    """

    raw_ids = np.asarray(stable_column_ids)
    if raw_ids.ndim != 1 or raw_ids.size != matrix.shape[1]:
        raise RowLocalGeneratorIneligible(
            "stable_column_ids must name every generator column exactly once"
        )
    if raw_ids.dtype.kind not in "iu" or raw_ids.dtype.kind == "b":
        raise RowLocalGeneratorIneligible("stable_column_ids must be integers")
    stable_ids = _readonly_copy(raw_ids, dtype=np.dtype(np.int64))
    if (
        np.any(stable_ids < 0)
        or np.unique(stable_ids).size != stable_ids.size
    ):
        raise RowLocalGeneratorIneligible(
            "stable_column_ids must be nonnegative and unique"
        )
    row_counts = np.diff(matrix.indptr)
    if np.any(row_counts > 1):
        raise RowLocalGeneratorIneligible(
            "generator matrix is not row-local: a source row has multiple entries"
        )
    nonempty_rows = np.flatnonzero(row_counts == 1)
    positions = matrix.indptr[nonempty_rows]
    generator_columns = np.asarray(matrix.indices[positions], dtype=np.int64)
    if np.unique(generator_columns).size != generator_columns.size:
        raise RowLocalGeneratorIneligible(
            "row-local mapping is not injective across source rows"
        )
    row_to_column = np.full(matrix.shape[0], -1, dtype=np.int64)
    row_scale = np.zeros(matrix.shape[0], dtype=np.float64)
    row_to_column[nonempty_rows] = generator_columns
    row_scale[nonempty_rows] = matrix.data[positions]
    frozen_mapping = _readonly_copy(row_to_column, dtype=np.dtype(np.int64))
    frozen_scale = _readonly_copy(row_scale, dtype=np.dtype(np.float64))
    frozen_ids = stable_ids
    digest = hashlib.sha256()
    digest.update(b"act.exact_sparse_conv.row_local_plan.v1")
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(frozen_mapping.tobytes())
    digest.update(frozen_scale.tobytes())
    digest.update(frozen_ids.tobytes())
    monotone = bool(
        generator_columns.size <= 1
        or np.all(generator_columns[1:] > generator_columns[:-1])
    )
    return RowLocalGeneratorPlan(
        source_row_count=int(matrix.shape[0]),
        generator_column_count=int(matrix.shape[1]),
        row_to_generator_column=frozen_mapping,
        row_scale=frozen_scale,
        stable_column_ids=frozen_ids,
        mapped_row_count=int(nonempty_rows.size),
        all_rows_mapped=bool(nonempty_rows.size == matrix.shape[0]),
        positional_mapping_monotone=monotone,
        digest=digest.hexdigest(),
    )


def prepare_row_local_generator_plan(
    generators: Any,
    *,
    stable_column_ids: Any,
) -> RowLocalGeneratorPlan:
    """Validate and snapshot an injective, row-local generator matrix."""

    matrix = _require_canonical_finite_csr(generators, name="generators")
    return _prepare_row_local_generator_plan_from_snapshot(
        matrix, stable_column_ids=stable_column_ids
    )


def _direct_apply_row_local_plan(
    matrix: sp.csr_matrix,
    plan: RowLocalGeneratorPlan,
) -> Tuple[sp.csr_matrix, int]:
    if matrix.shape[1] != plan.source_row_count:
        raise ExactSparseConvCandidateError(
            "CONV columns disagree with row-local source rows"
        )
    source_rows = np.asarray(matrix.indices, dtype=np.int64)
    mapped_columns = plan.row_to_generator_column[source_rows]
    keep = mapped_columns >= 0
    kept_source_rows = source_rows[keep]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        products = (
            matrix.data[keep] * plan.row_scale[kept_source_rows]
        )
    if not np.all(np.isfinite(products)):
        raise ExactSparseConvCandidateError(
            "row-local generator product overflowed or became NaN"
        )
    nonzero_product = products != 0.0
    if not np.all(nonzero_product):
        kept_positions = np.flatnonzero(keep)[nonzero_product]
        keep = np.zeros(matrix.nnz, dtype=np.bool_)
        keep[kept_positions] = True
        kept_source_rows = source_rows[keep]
        mapped_columns = plan.row_to_generator_column[source_rows]
        products = products[nonzero_product]
    counts_prefix = np.empty(matrix.nnz + 1, dtype=np.int64)
    counts_prefix[0] = 0
    np.cumsum(keep, dtype=np.int64, out=counts_prefix[1:])
    output_indptr64 = counts_prefix[np.asarray(matrix.indptr, dtype=np.int64)]
    if output_indptr64[-1] > _INT32_MAX:
        raise ExactSparseConvCandidateError("row-local output exceeds int32 CSR domain")
    output = sp.csr_matrix(
        (
            np.asarray(products, dtype=np.float64),
            np.asarray(mapped_columns[keep], dtype=np.int32),
            np.asarray(output_indptr64, dtype=np.int32),
        ),
        shape=(matrix.shape[0], plan.generator_column_count),
        dtype=np.float64,
    )
    if not plan.positional_mapping_monotone:
        output.sort_indices()
    if not output.has_canonical_format:
        raise RuntimeError("row-local direct output is not canonical")
    return output, int(np.count_nonzero(keep))


def apply_conv_to_row_local_generators_candidate(
    conv_matrix: Any,
    generators: Any,
    *,
    stable_column_ids: Any,
    return_receipt: bool = False,
) -> sp.csr_matrix | Tuple[sp.csr_matrix, RowLocalGeneratorApplyReceipt]:
    """Apply an already-built CONV CSR to an eligible row-local ``G``."""

    matrix = _require_canonical_finite_csr(conv_matrix, name="conv_matrix")
    generator_matrix = _require_canonical_finite_csr(
        generators, name="generators"
    )
    plan = _prepare_row_local_generator_plan_from_snapshot(
        generator_matrix, stable_column_ids=stable_column_ids
    )
    output, mapped_products = _direct_apply_row_local_plan(matrix, plan)
    if not return_receipt:
        return output
    return output, RowLocalGeneratorApplyReceipt(
        mode="standalone_csr_relabel_v1",
        stable_mapping_digest=plan.digest,
        source_shape=(int(generator_matrix.shape[0]), int(generator_matrix.shape[1])),
        output_shape=(int(output.shape[0]), int(output.shape[1])),
        source_generator_nnz=int(generator_matrix.nnz),
        conv_operator_nnz=int(matrix.nnz),
        output_generator_nnz=int(output.nnz),
        mapped_product_count=mapped_products,
    )


def clear_exact_conv_topology_cache() -> None:
    """Clear only the disconnected candidate's immutable geometry cache."""

    _cached_spatial_topology.cache_clear()


def exact_conv_topology_cache_info() -> Any:
    """Expose standard read-only cache counters for audit tests."""

    return _cached_spatial_topology.cache_info()


def get_exact_conv_spatial_topology(
    *,
    input_shape: Any,
    output_shape: Any,
    kernel: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: Any = 1,
) -> ExactConvSpatialTopology:
    """Validate and return the immutable value-keyed spatial topology."""

    normalized_input = _shape4(input_shape, name="input_shape")
    normalized_output = _shape4(output_shape, name="output_shape")
    normalized_kernel = _pair(kernel, name="kernel")
    normalized_stride = _pair(stride, name="stride")
    normalized_padding = _pair(padding, name="padding")
    normalized_dilation = _pair(dilation, name="dilation")
    normalized_groups = _builtin_int(groups, name="groups")
    if (
        min(normalized_kernel) <= 0
        or min(normalized_stride) <= 0
        or min(normalized_dilation) <= 0
        or min(normalized_padding) < 0
        or normalized_groups <= 0
    ):
        raise ExactSparseConvCandidateError("invalid CONV geometry")
    batch, in_ch, in_h, in_w = normalized_input
    out_batch, out_ch, out_h, out_w = normalized_output
    kh, kw = normalized_kernel
    expected_h, _effective_h = _checked_conv_output_extent(
        in_h,
        kh,
        normalized_stride[0],
        normalized_padding[0],
        normalized_dilation[0],
        axis="height",
    )
    expected_w, _effective_w = _checked_conv_output_extent(
        in_w,
        kw,
        normalized_stride[1],
        normalized_padding[1],
        normalized_dilation[1],
        axis="width",
    )
    if (
        batch != out_batch
        or in_ch % normalized_groups != 0
        or out_ch % normalized_groups != 0
        or (out_h, out_w) != (expected_h, expected_w)
        or expected_h <= 0
        or expected_w <= 0
    ):
        raise ExactSparseConvCandidateError("inconsistent CONV shapes/groups")
    input_area = _checked_nonnegative_product(
        (in_h, in_w), limit=_INT32_MAX, name="input spatial area"
    )
    output_area = _checked_nonnegative_product(
        (out_h, out_w), limit=_INT32_MAX, name="output spatial area"
    )
    kernel_area = _checked_nonnegative_product(
        (kh, kw), limit=_INT32_MAX, name="kernel spatial area"
    )
    n_rows = _checked_nonnegative_product(
        (batch, out_ch, output_area),
        limit=_INT32_MAX,
        name="CONV CSR rows",
    )
    _n_cols = _checked_nonnegative_product(
        (batch, in_ch, input_area),
        limit=_INT32_MAX,
        name="CONV CSR columns",
    )
    _topology_cells = _checked_nonnegative_product(
        (output_area, kernel_area),
        limit=_INT32_MAX,
        name="cached spatial topology cells",
    )
    in_ch_per_group = in_ch // normalized_groups
    _channel_template_cells = _checked_nonnegative_product(
        (output_area, in_ch, kernel_area),
        limit=_INT32_MAX,
        name="direct CSR channel-template cells",
    )
    _full_support_nnz = _checked_nonnegative_product(
        (n_rows, in_ch_per_group, kernel_area),
        limit=_INT32_MAX,
        name="full-support CONV nonzeros",
    )
    try:
        return _cached_spatial_topology(
            normalized_input,
            normalized_output,
            normalized_kernel,
            normalized_stride,
            normalized_padding,
            normalized_dilation,
            normalized_groups,
        )
    except ExactSparseConvCandidateError:
        raise
    except (MemoryError, OverflowError, ValueError) as exc:
        raise ExactSparseConvCandidateError(
            "CONV topology construction exceeded checked geometry/resources"
        ) from exc


def _direct_conv_operator_csr(
    topology: ExactConvSpatialTopology,
    weight: np.ndarray,
) -> sp.csr_matrix:
    """Fill canonical CSR rows directly from a compact spatial template.

    Within each output row, flattening order is input-channel then
    ``(kernel_row, kernel_col)``.  Channels occupy disjoint flattened NCHW
    blocks.  Within one channel, positive dilation makes valid flattened input
    coordinates strictly increase in that kernel order, including across a
    kernel-row boundary.  Removing zero weights therefore preserves strict
    column order and cannot create duplicates.
    """

    batch, in_ch, in_h, in_w = topology.input_shape
    _out_batch, out_ch, out_h, out_w = topology.output_shape
    groups = topology.groups
    in_ch_per_group = int(weight.shape[1])
    out_ch_per_group = out_ch // groups
    output_area = out_h * out_w
    input_area = in_h * in_w
    kernel_area = int(weight.shape[2] * weight.shape[3])
    flat_weight = weight.reshape(out_ch, in_ch_per_group, kernel_area)
    valid_spatial = topology.input_spatial_by_output >= 0

    # Exact row lengths factor into (nonzero input channels per kernel offset)
    # x (spatial validity of that offset).  This small integer product avoids
    # constructing every output-channel value slab twice.
    support_counts = np.count_nonzero(flat_weight != 0.0, axis=1).astype(
        np.int64, copy=False
    )
    row_counts64 = support_counts @ valid_spatial.astype(
        np.int64, copy=False
    ).T
    if row_counts64.size and int(np.max(row_counts64)) > _INT32_MAX:
        raise ExactSparseConvCandidateError("one CONV row exceeds int32 CSR domain")
    row_counts = row_counts64.astype(np.int32, copy=False)
    per_batch_nnz = int(np.sum(row_counts, dtype=np.int64))
    total_nnz = batch * per_batch_nnz
    if total_nnz > _INT32_MAX:
        raise ExactSparseConvCandidateError(
            "CONV nonzero count exceeds int32 CSR domain"
        )
    indptr = np.empty(batch * out_ch * output_area + 1, dtype=np.int32)
    indptr[0] = 0
    tiled_counts = np.broadcast_to(
        row_counts.reshape(1, -1), (batch, out_ch * output_area)
    ).reshape(-1)
    cumulative = np.cumsum(tiled_counts, dtype=np.int64)
    indptr[1:] = cumulative

    indices = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=np.float64)
    safe_input_spatial = np.maximum(topology.input_spatial_by_output, 0)
    channel_columns = []
    for group in range(groups):
        global_channels = (
            group * in_ch_per_group
            + np.arange(in_ch_per_group, dtype=np.int64)
        )
        columns = (
            global_channels[None, :, None] * input_area
            + safe_input_spatial[:, None, :]
        )
        channel_columns.append(columns)

    cursor = 0
    for n in range(batch):
        batch_column_offset = n * in_ch * input_area
        for co in range(out_ch):
            values = np.take(
                flat_weight[co], topology.kernel_gather_by_output, axis=1
            ).transpose(1, 0, 2)
            keep = valid_spatial[:, None, :] & (values != 0.0)
            emitted = int(np.count_nonzero(keep))
            destination = slice(cursor, cursor + emitted)
            group = co // out_ch_per_group
            indices[destination] = (
                channel_columns[group][keep] + batch_column_offset
            )
            data[destination] = values[keep]
            cursor += emitted
    if cursor != total_nnz:
        raise RuntimeError("direct CONV CSR row counts changed during fill")
    matrix = sp.csr_matrix(
        (data, indices, indptr),
        shape=(batch * out_ch * output_area, batch * in_ch * input_area),
        dtype=np.float64,
    )
    if matrix.nnz != total_nnz or not matrix.has_canonical_format:
        raise RuntimeError("direct CONV CSR output is not canonical")
    return matrix


def exact_sparse_conv2d_matrix_from_layer_candidate(
    layer: Any,
    *,
    return_receipt: bool = False,
) -> Tuple[sp.csr_matrix, np.ndarray] | Tuple[
    sp.csr_matrix, np.ndarray, ExactSparseConvCSRReceipt
]:
    """Build an exact canonical NCHW CSR from a current layer snapshot."""

    try:
        params = layer.params
        weight_value = params["weight"]
        input_shape_value = params["input_shape"]
        output_shape_value = params["output_shape"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ExactSparseConvCandidateError(
            "layer must provide weight/input_shape/output_shape parameters"
        ) from exc

    weight = _numpy_snapshot(weight_value, name="weight")
    if weight.ndim != 4 or min(weight.shape) <= 0:
        raise ExactSparseConvCandidateError(
            "weight must have positive (out_ch, in_ch/group, kh, kw) shape"
        )
    out_ch, in_ch_per_group, kh, kw = (int(value) for value in weight.shape)
    groups = _builtin_int(params.get("groups", 1), name="groups")
    topology = get_exact_conv_spatial_topology(
        input_shape=input_shape_value,
        output_shape=output_shape_value,
        kernel=(kh, kw),
        stride=params.get("stride", 1),
        padding=params.get("padding", 0),
        dilation=params.get("dilation", 1),
        groups=groups,
    )
    batch, in_ch, in_h, in_w = topology.input_shape
    out_batch, out_ch_shape, out_h, out_w = topology.output_shape
    if (
        batch != out_batch
        or out_ch != out_ch_shape
        or in_ch_per_group * groups != in_ch
        or out_ch % groups != 0
    ):
        raise ExactSparseConvCandidateError("weight shape disagrees with CONV geometry")

    bias_value = params.get("bias")
    if bias_value is None:
        bias = None
    else:
        bias = _numpy_snapshot(bias_value, name="bias").reshape(-1)
        if bias.size != out_ch:
            raise ExactSparseConvCandidateError("bias size disagrees with output channels")

    output_area = out_h * out_w
    n_rows = batch * out_ch * output_area
    matrix = _direct_conv_operator_csr(topology, weight)
    if bias is None:
        bias_vector = np.zeros(n_rows, dtype=np.float64)
    else:
        bias_vector = np.tile(np.repeat(bias, output_area), batch)

    if not return_receipt:
        return matrix, bias_vector
    # Direct row-major construction has no COO triplet staging arrays.
    triplet_nbytes = 0
    csr_nbytes = int(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    )
    kernel_area = kh * kw
    slab_entries = output_area * in_ch_per_group * kernel_area
    direct_workspace_upper = int(
        weight.nbytes
        + bias_vector.nbytes
        + 32 * slab_entries
        + 8 * output_area * in_ch * kernel_area
        + 12 * out_ch * output_area
        + 12 * batch * out_ch * output_area
    )
    receipt = ExactSparseConvCSRReceipt(
        geometry_digest=topology.digest,
        matrix_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        matrix_nnz=int(matrix.nnz),
        topology_incidence_count=topology.incidence_count,
        topology_nbytes=topology.topology_nbytes,
        triplet_nbytes=triplet_nbytes,
        csr_nbytes=csr_nbytes,
        peak_workspace_upper_bytes=int(
            topology.topology_nbytes
            + csr_nbytes
            + direct_workspace_upper
        ),
    )
    return matrix, bias_vector, receipt


def fused_exact_sparse_conv_row_local_generators_candidate(
    layer: Any,
    generators: Any,
    *,
    stable_column_ids: Any,
    return_receipt: bool = False,
) -> sp.csr_matrix | Tuple[sp.csr_matrix, RowLocalGeneratorApplyReceipt]:
    """Fuse exact CONV topology emission with an injective row-local ``G``.

    This computes only the generator transform.  Center, bias, and numerical
    error propagation remain deliberately outside this disconnected
    candidate.  Unlike the standalone path, no intermediate CONV CSR is
    materialized.
    """

    try:
        params = layer.params
        weight_value = params["weight"]
        input_shape_value = params["input_shape"]
        output_shape_value = params["output_shape"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ExactSparseConvCandidateError(
            "layer must provide weight/input_shape/output_shape parameters"
        ) from exc
    weight = _numpy_snapshot(weight_value, name="weight")
    if weight.ndim != 4 or min(weight.shape) <= 0:
        raise ExactSparseConvCandidateError(
            "weight must have positive (out_ch, in_ch/group, kh, kw) shape"
        )
    out_ch, in_ch_per_group, kh, kw = (int(value) for value in weight.shape)
    groups = _builtin_int(params.get("groups", 1), name="groups")
    topology = get_exact_conv_spatial_topology(
        input_shape=input_shape_value,
        output_shape=output_shape_value,
        kernel=(kh, kw),
        stride=params.get("stride", 1),
        padding=params.get("padding", 0),
        dilation=params.get("dilation", 1),
        groups=groups,
    )
    batch, in_ch, in_h, in_w = topology.input_shape
    out_batch, out_ch_shape, out_h, out_w = topology.output_shape
    if (
        batch != out_batch
        or out_ch != out_ch_shape
        or in_ch_per_group * groups != in_ch
        or out_ch % groups != 0
    ):
        raise ExactSparseConvCandidateError("weight shape disagrees with CONV geometry")

    generator_matrix = _require_canonical_finite_csr(
        generators, name="generators"
    )
    plan = _prepare_row_local_generator_plan_from_snapshot(
        generator_matrix, stable_column_ids=stable_column_ids
    )
    input_rows = batch * in_ch * in_h * in_w
    if plan.source_row_count != input_rows:
        raise ExactSparseConvCandidateError(
            "generator rows disagree with flattened CONV input"
        )

    out_ch_per_group = out_ch // groups
    output_area = out_h * out_w
    input_area = in_h * in_w
    kernel_area = kh * kw
    flat_weight = weight.reshape(out_ch, in_ch_per_group, kernel_area)
    valid_spatial = topology.input_spatial_by_output >= 0
    safe_input_spatial = np.maximum(topology.input_spatial_by_output, 0)
    source_columns_by_group = []
    for group in range(groups):
        global_channels = (
            group * in_ch_per_group
            + np.arange(in_ch_per_group, dtype=np.int64)
        )
        source_columns_by_group.append(
            global_channels[None, :, None] * input_area
            + safe_input_spatial[:, None, :]
        )

    support_counts = np.count_nonzero(flat_weight != 0.0, axis=1).astype(
        np.int64, copy=False
    )
    operator_row_counts64 = support_counts @ valid_spatial.astype(
        np.int64, copy=False
    ).T
    if operator_row_counts64.size and int(np.max(operator_row_counts64)) > _INT32_MAX:
        raise ExactSparseConvCandidateError("one CONV row exceeds int32 CSR domain")
    operator_row_counts = operator_row_counts64.astype(np.int32, copy=False)
    per_batch_operator_nnz = int(np.sum(operator_row_counts64, dtype=np.int64))

    nonzero_weight = np.abs(weight[weight != 0.0])
    nonzero_scale = np.abs(plan.row_scale[plan.row_to_generator_column >= 0])
    product_range_safe = bool(nonzero_weight.size == 0)
    if nonzero_weight.size and nonzero_scale.size:
        # Binary64 multiplication is monotone on nonnegative finite operands.
        # Thus the rounded extrema prove that every live product is finite and
        # nonzero without relying on a wider floating-point implementation.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            minimum_product = np.float64(np.min(nonzero_weight)) * np.float64(
                np.min(nonzero_scale)
            )
            maximum_product = np.float64(np.max(nonzero_weight)) * np.float64(
                np.max(nonzero_scale)
            )
        product_range_safe = bool(
            minimum_product != 0.0 and np.isfinite(maximum_product)
        )
    fast_full_map = bool(plan.all_rows_mapped and product_range_safe)
    if fast_full_map:
        row_counts = np.broadcast_to(
            operator_row_counts[None, :, :],
            (batch, out_ch, output_area),
        ).copy()
    else:
        row_counts = np.empty((batch, out_ch, output_area), dtype=np.int32)
        for co in range(out_ch):
            values = np.take(
                flat_weight[co], topology.kernel_gather_by_output, axis=1
            ).transpose(1, 0, 2)
            weight_keep = valid_spatial[:, None, :] & (values != 0.0)
            group = co // out_ch_per_group
            base_source_columns = source_columns_by_group[group]
            for n in range(batch):
                source_rows = base_source_columns + n * in_ch * input_area
                mapped_columns = plan.row_to_generator_column[source_rows]
                mapped = mapped_columns >= 0
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    products = values * plan.row_scale[source_rows]
                live_products = weight_keep & mapped
                if not np.all(np.isfinite(products[live_products])):
                    raise ExactSparseConvCandidateError(
                        "fused row-local product overflowed or became NaN"
                    )
                keep = live_products & (products != 0.0)
                row_counts[n, co, :] = np.count_nonzero(keep, axis=(1, 2))

    conv_operator_nnz = batch * per_batch_operator_nnz
    output_nnz = int(np.sum(row_counts, dtype=np.int64))
    if max(conv_operator_nnz, output_nnz) > _INT32_MAX:
        raise ExactSparseConvCandidateError(
            "fused CONV nonzero count exceeds int32 CSR domain"
        )
    indptr = np.empty(batch * out_ch * output_area + 1, dtype=np.int32)
    indptr[0] = 0
    indptr[1:] = np.cumsum(row_counts.reshape(-1), dtype=np.int64)
    indices = np.empty(output_nnz, dtype=np.int32)
    data = np.empty(output_nnz, dtype=np.float64)
    cursor = 0
    for n in range(batch):
        batch_column_offset = n * in_ch * input_area
        for co in range(out_ch):
            values = np.take(
                flat_weight[co], topology.kernel_gather_by_output, axis=1
            ).transpose(1, 0, 2)
            weight_keep = valid_spatial[:, None, :] & (values != 0.0)
            group = co // out_ch_per_group
            source_rows = (
                source_columns_by_group[group] + batch_column_offset
            )
            mapped_columns = plan.row_to_generator_column[source_rows]
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                products = values * plan.row_scale[source_rows]
            keep = (
                weight_keep
                & (mapped_columns >= 0)
                & (products != 0.0)
            )
            emitted = int(np.count_nonzero(keep))
            destination = slice(cursor, cursor + emitted)
            indices[destination] = mapped_columns[keep]
            data[destination] = products[keep]
            cursor += emitted
    if cursor != output_nnz:
        raise RuntimeError("fused CONV CSR row counts changed during fill")
    output = sp.csr_matrix(
        (data, indices, indptr),
        shape=(batch * out_ch * output_area, plan.generator_column_count),
        dtype=np.float64,
    )
    if not plan.positional_mapping_monotone:
        output.sort_indices()
    if output.nnz != output_nnz or not output.has_canonical_format:
        raise RuntimeError("fused row-local CONV CSR output is not canonical")
    if not return_receipt:
        return output
    return output, RowLocalGeneratorApplyReceipt(
        mode="fused_topology_to_generators_v1",
        stable_mapping_digest=plan.digest,
        source_shape=(int(generator_matrix.shape[0]), int(generator_matrix.shape[1])),
        output_shape=(int(output.shape[0]), int(output.shape[1])),
        source_generator_nnz=int(generator_matrix.nnz),
        conv_operator_nnz=int(conv_operator_nnz),
        output_generator_nnz=int(output.nnz),
        mapped_product_count=int(output_nnz),
    )


__all__ = [
    "ExactConvSpatialTopology",
    "ExactSparseConvCSRReceipt",
    "ExactSparseConvCandidateError",
    "RowLocalGeneratorApplyReceipt",
    "RowLocalGeneratorIneligible",
    "RowLocalGeneratorPlan",
    "SpatialKernelFrame",
    "apply_conv_to_row_local_generators_candidate",
    "clear_exact_conv_topology_cache",
    "exact_conv_topology_cache_info",
    "exact_sparse_conv2d_matrix_from_layer_candidate",
    "fused_exact_sparse_conv_row_local_generators_candidate",
    "get_exact_conv_spatial_topology",
    "prepare_row_local_generator_plan",
]
