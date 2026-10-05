"""Exact, fail-closed linear operators with compact resident storage.

The implicit convolution descriptor in this module deliberately keeps only the
kernel and convolution geometry resident.  ``to_csr_reference`` exists as a
small-test oracle; normal evaluation and left composition do not expand the
convolution matrix.

``resident_entries`` counts stored coefficient or selection payload entries;
``resident_bytes`` counts their NumPy/SciPy array buffers (including CSR index
buffers).  Constant-size Python geometry metadata is intentionally excluded.
``logical_expanded_nnz`` follows the structural CSR convention and therefore
includes explicitly stored zero-valued kernel positions.
"""

from __future__ import annotations

import hashlib
import math
import operator
from typing import Callable

import numpy as np
import scipy.sparse as sp


__all__ = [
    "CSRLinearOp",
    "DiagonalLinearOp",
    "ImplicitConv2DOp",
]


def _array_digest(*arrays: np.ndarray) -> bytes:
    """Return a stable content digest for immutable operator payloads."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(repr(tuple(int(v) for v in contiguous.shape)).encode())
        digest.update(contiguous.tobytes(order="C"))
    return digest.digest()


def _numeric_array(value, *, name: str, ndim: int) -> np.ndarray:
    """Return an owned finite float64 array, rejecting lossy input kinds."""
    try:
        raw = np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive for foreign arrays
        raise ValueError(f"{name} is not array-like") from exc
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {raw.shape}")
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real numeric, got dtype {raw.dtype}")
    array = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _strict_int(value, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    result = int(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _pair(value, *, name: str, minimum: int) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        first = second = _strict_int(value, name=name, minimum=minimum)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer or an integer pair") from exc
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two integers")
        first = _strict_int(values[0], name=f"{name}[0]", minimum=minimum)
        second = _strict_int(values[1], name=f"{name}[1]", minimum=minimum)
    return first, second


def _canonical_q(Q, *, expected_columns: int) -> sp.csr_matrix:
    """Validate and canonicalize a left-composition matrix."""
    if sp.issparse(Q):
        if Q.ndim != 2:
            raise ValueError(f"Q must be 2-D, got ndim={Q.ndim}")
        if Q.dtype.kind not in "biuf":
            raise ValueError(f"Q must be real numeric, got dtype {Q.dtype}")
        matrix = Q.tocsr().astype(np.float64, copy=True)
    else:
        matrix = sp.csr_matrix(_numeric_array(Q, name="Q", ndim=2))
    if matrix.shape[1] != expected_columns:
        raise ValueError(
            f"Q shape mismatch: {matrix.shape} cannot left-compose "
            f"an operator with {expected_columns} rows"
        )
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("Q contains a non-finite value")
    return matrix


def _checked_cap(max_nnz) -> int:
    if isinstance(max_nnz, (bool, np.bool_)):
        raise ValueError("max_nnz must be a non-negative integer")
    return _strict_int(max_nnz, name="max_nnz", minimum=0)


def _left_compose_rows(
    Q,
    *,
    operator_shape: tuple[int, int],
    row: Callable[[int], tuple[np.ndarray, np.ndarray]],
    max_nnz,
) -> sp.csr_matrix:
    """Compute ``Q @ W`` from an exact row oracle without materializing W."""
    cap = _checked_cap(max_nnz)
    q_csr = _canonical_q(Q, expected_columns=operator_shape[0])
    result_indptr = np.zeros(q_csr.shape[0] + 1, dtype=np.int64)
    result_indices: list[int] = []
    result_data: list[float] = []

    for output_row in range(q_csr.shape[0]):
        accumulator: dict[int, float] = {}
        q_start = int(q_csr.indptr[output_row])
        q_stop = int(q_csr.indptr[output_row + 1])
        for q_position in range(q_start, q_stop):
            operator_row = int(q_csr.indices[q_position])
            q_value = float(q_csr.data[q_position])
            columns, values = row(operator_row)
            for column, operator_value in zip(columns, values, strict=True):
                w_value = float(operator_value)
                if w_value == 0.0:
                    continue
                product = q_value * w_value
                if not math.isfinite(product):
                    raise ValueError("Q @ W produced a non-finite product")
                column = int(column)
                total = accumulator.get(column, 0.0) + product
                if not math.isfinite(total):
                    raise ValueError("Q @ W produced a non-finite sum")
                if total == 0.0:
                    accumulator.pop(column, None)
                else:
                    accumulator[column] = total
                if len(result_data) + len(accumulator) > cap:
                    raise MemoryError(
                        f"Q @ W nnz exceeds max_nnz={cap}"
                    )

        for column in sorted(accumulator):
            result_indices.append(column)
            result_data.append(accumulator[column])
        result_indptr[output_row + 1] = len(result_data)

    result = sp.csr_matrix(
        (
            np.asarray(result_data, dtype=np.float64),
            np.asarray(result_indices, dtype=np.int64),
            result_indptr,
        ),
        shape=(q_csr.shape[0], operator_shape[1]),
    )
    if result.nnz > cap:  # The incremental guard is authoritative; be defensive.
        raise MemoryError(f"Q @ W nnz exceeds max_nnz={cap}")
    if not np.all(np.isfinite(result.data)):
        raise ValueError("Q @ W contains a non-finite value")
    return result


class CSRLinearOp:
    """An exact linear operator backed by canonical CSR storage."""

    def __init__(self, matrix):
        if sp.issparse(matrix):
            if matrix.ndim != 2:
                raise ValueError(f"matrix must be 2-D, got ndim={matrix.ndim}")
            if matrix.dtype.kind not in "biuf":
                raise ValueError(
                    f"matrix must be real numeric, got dtype {matrix.dtype}"
                )
            csr = matrix.tocsr().astype(np.float64, copy=True)
        else:
            csr = sp.csr_matrix(
                _numeric_array(matrix, name="matrix", ndim=2)
            )
        csr.sum_duplicates()
        csr.sort_indices()
        csr.eliminate_zeros()
        if not np.all(np.isfinite(csr.data)):
            raise ValueError("matrix contains a non-finite value")
        self._matrix = csr
        self._content_key = (
            "csr_linear_op_v1",
            self.shape,
            _array_digest(csr.indptr, csr.indices, csr.data),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._matrix.shape[0]), int(self._matrix.shape[1])

    @property
    def resident_entries(self) -> int:
        return int(self._matrix.data.size)

    @property
    def resident_bytes(self) -> int:
        return int(
            self._matrix.data.nbytes
            + self._matrix.indices.nbytes
            + self._matrix.indptr.nbytes
        )

    @property
    def logical_expanded_nnz(self) -> int:
        return int(self._matrix.nnz)

    @property
    def content_key(self) -> tuple:
        return self._content_key

    def to_csr_reference(self) -> sp.csr_matrix:
        return self._matrix.copy()

    def _row(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self._matrix.indptr[index])
        stop = int(self._matrix.indptr[index + 1])
        return self._matrix.indices[start:stop], self._matrix.data[start:stop]

    def matvec(self, vector) -> np.ndarray:
        x = _numeric_array(vector, name="vector", ndim=1)
        if x.size != self.shape[1]:
            raise ValueError(
                f"vector length mismatch: {x.size} vs {self.shape[1]}"
            )
        result = np.asarray(self._matrix @ x, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(result)):
            raise ValueError("W @ vector contains a non-finite value")
        return result

    def left_compose(self, Q, max_nnz) -> sp.csr_matrix:
        return _left_compose_rows(
            Q,
            operator_shape=self.shape,
            row=self._row,
            max_nnz=max_nnz,
        )


class DiagonalLinearOp:
    """An exact square diagonal operator."""

    def __init__(self, diagonal):
        self._diagonal = _numeric_array(
            diagonal, name="diagonal", ndim=1
        )
        self._content_key = (
            "diagonal_linear_op_v1",
            self.shape,
            _array_digest(self._diagonal),
        )

    @property
    def shape(self) -> tuple[int, int]:
        size = int(self._diagonal.size)
        return size, size

    @property
    def resident_entries(self) -> int:
        return int(self._diagonal.size)

    @property
    def resident_bytes(self) -> int:
        return int(self._diagonal.nbytes)

    @property
    def logical_expanded_nnz(self) -> int:
        return int(np.count_nonzero(self._diagonal))

    @property
    def content_key(self) -> tuple:
        return self._content_key

    def to_csr_reference(self) -> sp.csr_matrix:
        nonzero = np.flatnonzero(self._diagonal)
        return sp.csr_matrix(
            (
                self._diagonal[nonzero],
                (nonzero, nonzero),
            ),
            shape=self.shape,
        )

    def _row(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        value = self._diagonal[index]
        if value == 0.0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        return (
            np.asarray([index], dtype=np.int64),
            np.asarray([value], dtype=np.float64),
        )

    def matvec(self, vector) -> np.ndarray:
        x = _numeric_array(vector, name="vector", ndim=1)
        if x.size != self.shape[1]:
            raise ValueError(
                f"vector length mismatch: {x.size} vs {self.shape[1]}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            result = self._diagonal * x
        if not np.all(np.isfinite(result)):
            raise ValueError("W @ vector contains a non-finite value")
        return result

    def left_compose(self, Q, max_nnz) -> sp.csr_matrix:
        return _left_compose_rows(
            Q,
            operator_shape=self.shape,
            row=self._row,
            max_nnz=max_nnz,
        )


class ImplicitConv2DOp:
    """Exact implicit NCHW Conv2D matrix.

    ``kernel`` has PyTorch layout ``(out_channels, in_channels/groups,
    kernel_height, kernel_width)``.  A row mask zeros disabled output rows but
    never removes them, matching ``sparse_conv2d_matrix_from_layer_csr``.
    """

    def __init__(
        self,
        kernel,
        input_shape,
        *,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        row_mask=None,
    ):
        self._kernel = _numeric_array(kernel, name="kernel", ndim=4)
        try:
            raw_input_shape = tuple(input_shape)
        except TypeError as exc:
            raise ValueError("input_shape must be NCHW") from exc
        if len(raw_input_shape) != 4:
            raise ValueError(
                f"input_shape must be NCHW, got {raw_input_shape}"
            )
        self._input_shape = tuple(
            _strict_int(value, name=f"input_shape[{index}]", minimum=1)
            for index, value in enumerate(raw_input_shape)
        )
        self._stride = _pair(stride, name="stride", minimum=1)
        self._padding = _pair(padding, name="padding", minimum=0)
        self._dilation = _pair(dilation, name="dilation", minimum=1)
        self._groups = _strict_int(groups, name="groups", minimum=1)

        batch, channels, height, width = self._input_shape
        out_channels, in_per_group, kernel_height, kernel_width = (
            self._kernel.shape
        )
        if min(self._kernel.shape) <= 0:
            raise ValueError(
                f"kernel dimensions must be positive: {self._kernel.shape}"
            )
        if channels != in_per_group * self._groups:
            raise ValueError(
                "invalid grouped Conv2D input shape: "
                f"C={channels}, ICg={in_per_group}, groups={self._groups}"
            )
        if out_channels % self._groups:
            raise ValueError(
                "invalid grouped Conv2D output shape: "
                f"OC={out_channels}, groups={self._groups}"
            )

        output_height = (
            height
            + 2 * self._padding[0]
            - self._dilation[0] * (kernel_height - 1)
            - 1
        ) // self._stride[0] + 1
        output_width = (
            width
            + 2 * self._padding[1]
            - self._dilation[1] * (kernel_width - 1)
            - 1
        ) // self._stride[1] + 1
        if output_height < 0 or output_width < 0:
            raise ValueError(
                "invalid Conv2D output shape: "
                f"OH={output_height}, OW={output_width}"
            )
        self._output_shape = (
            batch,
            int(out_channels),
            int(output_height),
            int(output_width),
        )

        total_rows = math.prod(self._output_shape)
        total_columns = math.prod(self._input_shape)
        address_limit = int(np.iinfo(np.intp).max)
        if total_rows > address_limit or total_columns > address_limit:
            raise ValueError("Conv2D flattened shape exceeds platform limits")
        if row_mask is None:
            self._row_mask = None
        else:
            raw_mask = np.asarray(row_mask)
            if raw_mask.dtype.kind != "b":
                raise ValueError(
                    f"row_mask must be boolean, got dtype {raw_mask.dtype}"
                )
            mask = np.array(raw_mask, dtype=bool, copy=True).reshape(-1)
            if mask.size != total_rows:
                raise ValueError(
                    f"row_mask size mismatch: {mask.size} vs {total_rows}"
                )
            self._row_mask = mask

        self._logical_expanded_nnz = self._count_expanded_entries()
        mask_key = (
            None
            if self._row_mask is None
            else _array_digest(self._row_mask)
        )
        self._content_key = (
            "implicit_conv2d_op_v1",
            self._input_shape,
            self._output_shape,
            self._stride,
            self._padding,
            self._dilation,
            self._groups,
            _array_digest(self._kernel),
            mask_key,
        )

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return self._output_shape

    @property
    def shape(self) -> tuple[int, int]:
        return (
            math.prod(self._output_shape),
            math.prod(self._input_shape),
        )

    @property
    def resident_entries(self) -> int:
        mask_entries = 0 if self._row_mask is None else self._row_mask.size
        return int(self._kernel.size + mask_entries)

    @property
    def resident_bytes(self) -> int:
        mask_bytes = 0 if self._row_mask is None else self._row_mask.nbytes
        return int(self._kernel.nbytes + mask_bytes)

    @property
    def logical_expanded_nnz(self) -> int:
        return self._logical_expanded_nnz

    @property
    def content_key(self) -> tuple:
        return self._content_key

    def _spatial_stencil_counts(self):
        """Yield structural stencil sizes without allocating expanded rows."""
        _, _, height, width = self._input_shape
        _, in_per_group, kernel_height, kernel_width = self._kernel.shape
        _, _, output_height, output_width = self._output_shape
        for oh in range(output_height):
            for ow in range(output_width):
                valid = 0
                for kh in range(kernel_height):
                    ih = (
                        oh * self._stride[0]
                        - self._padding[0]
                        + kh * self._dilation[0]
                    )
                    if ih < 0 or ih >= height:
                        continue
                    for kw in range(kernel_width):
                        iw = (
                            ow * self._stride[1]
                            - self._padding[1]
                            + kw * self._dilation[1]
                        )
                        if 0 <= iw < width:
                            valid += 1
                yield valid * in_per_group

    def _count_expanded_entries(self) -> int:
        batch, out_channels, output_height, output_width = self._output_shape
        if self._row_mask is None:
            count = batch * out_channels * sum(
                self._spatial_stencil_counts()
            )
        else:
            spatial = output_height * output_width
            mask = self._row_mask.reshape(batch, out_channels, spatial)
            count = sum(
                int(np.count_nonzero(mask[:, :, position])) * entries
                for position, entries in enumerate(
                    self._spatial_stencil_counts()
                )
            )
        if count > int(np.iinfo(np.intp).max):
            raise ValueError("Conv2D expanded nnz exceeds platform limits")
        return count

    def _row(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        total_rows, _ = self.shape
        if index < 0 or index >= total_rows:
            raise IndexError(index)
        if self._row_mask is not None and not self._row_mask[index]:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        batch, channels, height, width = self._input_shape
        out_channels, in_per_group, kernel_height, kernel_width = (
            self._kernel.shape
        )
        _, _, output_height, output_width = self._output_shape
        output_spatial = output_height * output_width
        output_per_batch = out_channels * output_spatial
        batch_index, within_batch = divmod(index, output_per_batch)
        output_channel, spatial_index = divmod(within_batch, output_spatial)
        oh, ow = divmod(spatial_index, output_width)
        out_per_group = out_channels // self._groups
        group = output_channel // out_per_group
        first_channel = group * in_per_group
        input_batch_offset = batch_index * channels * height * width

        columns: list[int] = []
        values: list[float] = []
        for local_channel in range(in_per_group):
            input_channel = first_channel + local_channel
            channel_offset = (
                input_batch_offset + input_channel * height * width
            )
            for kh in range(kernel_height):
                ih = (
                    oh * self._stride[0]
                    - self._padding[0]
                    + kh * self._dilation[0]
                )
                if ih < 0 or ih >= height:
                    continue
                for kw in range(kernel_width):
                    iw = (
                        ow * self._stride[1]
                        - self._padding[1]
                        + kw * self._dilation[1]
                    )
                    if iw < 0 or iw >= width:
                        continue
                    columns.append(channel_offset + ih * width + iw)
                    values.append(
                        float(
                            self._kernel[
                                output_channel, local_channel, kh, kw
                            ]
                        )
                    )
        return (
            np.asarray(columns, dtype=np.int64),
            np.asarray(values, dtype=np.float64),
        )

    def to_csr_reference(self) -> sp.csr_matrix:
        """Expand the exact operator; intended only for focused test oracles."""
        total_rows, total_columns = self.shape
        indptr = np.empty(total_rows + 1, dtype=np.int64)
        indptr[0] = 0
        indices = np.empty(self.logical_expanded_nnz, dtype=np.int64)
        data = np.empty(self.logical_expanded_nnz, dtype=np.float64)
        cursor = 0
        for row_index in range(total_rows):
            row_columns, row_values = self._row(row_index)
            stop = cursor + row_columns.size
            indices[cursor:stop] = row_columns
            data[cursor:stop] = row_values
            cursor = stop
            indptr[row_index + 1] = cursor
        if cursor != self.logical_expanded_nnz:
            raise RuntimeError(
                "internal Conv2D expanded-nnz accounting mismatch"
            )
        return sp.csr_matrix(
            (data, indices, indptr), shape=(total_rows, total_columns)
        )

    def matvec(self, vector) -> np.ndarray:
        x = _numeric_array(vector, name="vector", ndim=1)
        if x.size != self.shape[1]:
            raise ValueError(
                f"vector length mismatch: {x.size} vs {self.shape[1]}"
            )
        result = np.zeros(self.shape[0], dtype=np.float64)
        for row_index in range(self.shape[0]):
            columns, values = self._row(row_index)
            if columns.size:
                total = 0.0
                for column, value in zip(columns, values, strict=True):
                    product = float(value) * float(x[int(column)])
                    if not math.isfinite(product):
                        raise ValueError(
                            "W @ vector produced a non-finite product"
                        )
                    total += product
                    if not math.isfinite(total):
                        raise ValueError(
                            "W @ vector produced a non-finite sum"
                        )
                result[row_index] = total
        if not np.all(np.isfinite(result)):
            raise ValueError("W @ vector contains a non-finite value")
        return result

    def left_compose(self, Q, max_nnz) -> sp.csr_matrix:
        return _left_compose_rows(
            Q,
            operator_shape=self.shape,
            row=self._row,
            max_nnz=max_nnz,
        )
