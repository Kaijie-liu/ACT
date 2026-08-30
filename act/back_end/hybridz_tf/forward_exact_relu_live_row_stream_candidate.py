"""Disconnected exact-ReLU live-row GPU stream candidate.

This module is deliberately not imported by the production verifier.  It
tests one representation only: phase bounds are taken from the established
interval pass, every unstable ReLU receives an exact binary graph, stable
active rows alias their predecessor, stable inactive rows are zero, and
continuous generator columns are propagated through only the affine output
rows that can reach an exact preactivation or the final graph output.  Each
dense preactivation row is stored once in an equality to a constraint-local
normalized factor; the ReLU facets then reference that sparse local factor.

There is no triangle fallback, BaB, backward propagation, dual tightening,
input sampling, PGD, concrete ONNX search, or solver call here.  Unsupported
inputs fail closed.  The ordinary Operator-HZ outward arithmetic envelope is
retained, so changing the CUDA reduction schedule does not silently claim
stored-bit equality with the production CSR path.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import triton
import triton.language as tl

from act.back_end.hybridz_tf import operator_hz as _oh
from act.back_end.hybridz_tf.exact_sparse_conv_csr_candidate import (
    ExactConvSpatialTopology,
    ExactSparseConvCandidateError,
    get_exact_conv_spatial_topology,
)
from act.back_end.solver.solver_hz import hz_fresh_col_ids


_SCHEMA = "act.hybridz.forward_exact_relu_live_row_stream_candidate.v2"
_SUPPORTED = frozenset(
    {"INPUT", "INPUT_SPEC", "CONV2D", "DENSE", "ADD", "RELU", "FLATTEN", "ASSERT"}
)
_FACTOR_BATCH = 64
_MAX_FACTORS = 200_000
_MAX_EXACT_ROWS = 200_000
_MAX_STREAMED_AFFINE_NNZ = 200_000_000
_MAX_OUTPUT_BYTES = 2_000_000_000


class ExactReLULiveRowStreamError(RuntimeError):
    """Fail-closed candidate error."""


@dataclass(frozen=True)
class _PhaseFrame:
    lower: np.ndarray
    upper: np.ndarray
    active: np.ndarray
    inactive: np.ndarray
    exact: np.ndarray
    continuous_columns: np.ndarray
    binary_columns: np.ndarray
    stream_half_widths: np.ndarray
    stream_rows: np.ndarray
    stream_continuous_columns: np.ndarray
    deferred_rows: np.ndarray
    deferred_active: np.ndarray
    deferred_continuous_columns: np.ndarray
    linked_rows: np.ndarray
    linked_continuous_columns: np.ndarray


@dataclass(frozen=True)
class _Shadow:
    center: np.ndarray
    error: np.ndarray
    mass_upper: np.ndarray


@dataclass(frozen=True)
class _AffineSnapshot:
    kind: str
    weight: np.ndarray
    bias: np.ndarray
    input_size: int
    output_size: int
    topology: Optional[ExactConvSpatialTopology]


@dataclass(frozen=True)
class _DeviceCSR:
    indptr: torch.Tensor
    indices: torch.Tensor
    data: torch.Tensor
    rows: int
    columns: int


@dataclass(frozen=True)
class ExactReLULiveRowStreamReceipt:
    schema: str
    status: str
    factor_batch: int
    continuous_factors: int
    binary_factors: int
    exact_rows: int
    source_rows: int
    source_nnz: int
    output_nnz: int
    full_affine_nnz: int
    streamed_affine_nnz: int
    matrix_build_seconds: float
    shadow_seconds: float
    gpu_schedule_seconds: float
    phase_refinement_seconds: float
    gpu_stream_seconds: float
    assembly_seconds: float
    total_seconds: float
    incremental_cuda_peak_bytes: int
    phase_refinement_passes: int
    interval_exact_rows: int
    refined_stable_rows: int
    constraint_local_preactivation_factors: int
    generator_kernel: str = "triton_ordered_csr_dense_v1"
    all_unstable_exact: bool = True
    triangle_rows: int = 0
    second_full_hz_built: bool = False
    runtime_fallbacks: int = 0
    input_sampling_used: bool = False
    pgd_used: bool = False
    concrete_onnx_execution: bool = False
    candidate_authority: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False


@dataclass(frozen=True)
class ExactReLULiveRowStreamResult:
    hz: Any
    input_col_ids: np.ndarray
    input_layer_id: int
    output_layer_id: int
    assert_layer_id: int
    receipt: ExactReLULiveRowStreamReceipt


def _deadline(deadline: Optional[float], stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise ExactReLULiveRowStreamError(
            f"live-row stream deadline expired at {stage}"
        )


def _facts_box(
    facts: Mapping[int, Any], layer_id: int, width: int, *, name: str
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        bounds = facts[int(layer_id)].bounds
        lower = np.ascontiguousarray(
            bounds.lb.detach().cpu().double().numpy(), dtype=np.float64
        ).reshape(-1)
        upper = np.ascontiguousarray(
            bounds.ub.detach().cpu().double().numpy(), dtype=np.float64
        ).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExactReLULiveRowStreamError(f"{name} is unavailable") from exc
    if (
        lower.size != int(width)
        or upper.size != int(width)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
    ):
        raise ExactReLULiveRowStreamError(f"{name} is malformed")
    lower.setflags(write=False)
    upper.setflags(write=False)
    return lower, upper


def _topological(net: Any) -> Tuple[Tuple[Any, ...], Dict[int, Any]]:
    layers = list(net.layers)
    by_id: Dict[int, Any] = {}
    position: Dict[int, int] = {}
    for index, layer in enumerate(layers):
        layer_id = int(layer.id)
        if layer_id in by_id or _oh._kind(layer.kind) not in _SUPPORTED:
            raise ExactReLULiveRowStreamError("unsupported or duplicate layer")
        by_id[layer_id] = layer
        position[layer_id] = index
    indegree = {layer_id: 0 for layer_id in by_id}
    children = {layer_id: [] for layer_id in by_id}
    for layer_id in by_id:
        for parent in net.preds.get(layer_id, []):
            parent = int(parent)
            if parent not in by_id:
                raise ExactReLULiveRowStreamError("graph predecessor is missing")
            indegree[layer_id] += 1
            children[parent].append(layer_id)
    ready = [(position[layer_id], layer_id) for layer_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        _position, layer_id = heapq.heappop(ready)
        ordered.append(by_id[layer_id])
        for child in sorted(children[layer_id], key=position.__getitem__):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (position[child], child))
    if len(ordered) != len(layers):
        raise ExactReLULiveRowStreamError("graph is cyclic")
    return tuple(ordered), by_id


def _preds(net: Any, layer: Any, count: int) -> Tuple[int, ...]:
    values = tuple(int(value) for value in net.preds.get(int(layer.id), []))
    if len(values) != int(count):
        raise ExactReLULiveRowStreamError(
            f"{_oh._kind(layer.kind)} layer {int(layer.id)} needs {count} predecessors"
        )
    return values


def _allocate_ids(count: int) -> np.ndarray:
    if type(count) is not int or count < 0:
        raise ExactReLULiveRowStreamError("factor count is malformed")
    values = hz_fresh_col_ids(count, device="cpu")
    result = np.ascontiguousarray(
        values.detach().cpu().numpy(), dtype=np.int64
    ).reshape(-1)
    if result.size != count or (result.size and np.unique(result).size != count):
        raise ExactReLULiveRowStreamError("stable factor allocator returned malformed ids")
    return result


def _upper_exact_sum(*values: float) -> float:
    exact = sum((Fraction.from_float(float(value)) for value in values), Fraction(0))
    try:
        candidate = float(exact)
    except OverflowError as exc:
        raise ExactReLULiveRowStreamError("exact upper sum overflowed") from exc
    if not math.isfinite(candidate):
        raise ExactReLULiveRowStreamError("exact upper sum is non-finite")
    if Fraction.from_float(candidate) < exact:
        candidate = float(np.nextafter(candidate, np.inf))
    if not math.isfinite(candidate) or Fraction.from_float(candidate) < exact:
        raise ExactReLULiveRowStreamError("exact upper sum could not be rounded outward")
    return candidate


def _pad_columns(matrix: sp.csr_matrix, width: int) -> sp.csr_matrix:
    if matrix.shape[1] > int(width):
        raise ExactReLULiveRowStreamError("CSR column frame shrank")
    if matrix.shape[1] == int(width):
        return matrix
    return sp.csr_matrix(
        (matrix.data, matrix.indices, matrix.indptr),
        shape=(matrix.shape[0], int(width)),
        dtype=np.float64,
        copy=False,
    )


def _canonical(matrix: sp.spmatrix, *, name: str) -> sp.csr_matrix:
    try:
        out = matrix.tocsr()
        out.eliminate_zeros()
        out.sort_indices()
        _oh._require_canonical_csr(out, name=name)
    except Exception as exc:
        raise ExactReLULiveRowStreamError(f"{name} is not canonical") from exc
    return out


def _array_snapshot(value: Any, *, name: str) -> np.ndarray:
    try:
        source = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        raw = np.asarray(source)
        if np.issubdtype(raw.dtype, np.complexfloating):
            raise TypeError("complex input")
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ExactReLULiveRowStreamError(f"{name} is not binary64 data") from exc
    if not np.all(np.isfinite(result)):
        raise ExactReLULiveRowStreamError(f"{name} contains NaN or infinity")
    result.setflags(write=False)
    return result


def _affine_snapshot(layer: Any, *, input_size: int) -> _AffineSnapshot:
    layer_id = int(layer.id)
    kind = _oh._kind(layer.kind)
    try:
        params = layer.params
        weight = _array_snapshot(params["weight"], name=f"weight[{layer_id}]")
    except (AttributeError, KeyError, TypeError) as exc:
        raise ExactReLULiveRowStreamError(
            f"affine parameters are missing at layer {layer_id}"
        ) from exc
    output_size = int(len(layer.out_vars))
    topology: Optional[ExactConvSpatialTopology] = None
    if kind == "DENSE":
        if weight.ndim != 2 or weight.shape != (output_size, int(input_size)):
            raise ExactReLULiveRowStreamError(
                f"DENSE geometry is malformed at layer {layer_id}"
            )
        channel_bias_size = output_size
    elif kind == "CONV2D":
        if weight.ndim != 4 or min(weight.shape) <= 0:
            raise ExactReLULiveRowStreamError(
                f"CONV weight is malformed at layer {layer_id}"
            )
        out_ch, in_ch_per_group, kh, kw = (int(v) for v in weight.shape)
        try:
            topology = get_exact_conv_spatial_topology(
                input_shape=params["input_shape"],
                output_shape=params["output_shape"],
                kernel=(kh, kw),
                stride=params.get("stride", 1),
                padding=params.get("padding", 0),
                dilation=params.get("dilation", 1),
                groups=params.get("groups", 1),
            )
        except (ExactSparseConvCandidateError, KeyError, TypeError) as exc:
            raise ExactReLULiveRowStreamError(
                f"CONV topology failed at layer {layer_id}"
            ) from exc
        batch, in_ch, in_h, in_w = topology.input_shape
        out_batch, topology_out_ch, out_h, out_w = topology.output_shape
        if (
            batch * in_ch * in_h * in_w != int(input_size)
            or out_batch * topology_out_ch * out_h * out_w != output_size
            or topology_out_ch != out_ch
            or in_ch_per_group * topology.groups != in_ch
            or out_ch % topology.groups != 0
        ):
            raise ExactReLULiveRowStreamError(
                f"CONV shape disagrees with graph at layer {layer_id}"
            )
        channel_bias_size = out_ch
    else:
        raise ExactReLULiveRowStreamError("not an affine layer")
    bias_value = params.get("bias")
    if bias_value is None:
        channel_bias = np.zeros(channel_bias_size, dtype=np.float64)
    else:
        channel_bias = _array_snapshot(
            bias_value, name=f"bias[{layer_id}]"
        ).reshape(-1)
        if channel_bias.size != channel_bias_size:
            raise ExactReLULiveRowStreamError(
                f"affine bias is malformed at layer {layer_id}"
            )
    if kind == "CONV2D":
        assert topology is not None
        output_area = topology.output_shape[2] * topology.output_shape[3]
        bias = np.tile(
            np.repeat(channel_bias, output_area), topology.output_shape[0]
        )
    else:
        bias = np.array(channel_bias, dtype=np.float64, copy=True)
    bias = np.ascontiguousarray(bias, dtype=np.float64)
    bias.setflags(write=False)
    return _AffineSnapshot(
        kind=kind,
        weight=weight,
        bias=bias,
        input_size=int(input_size),
        output_size=output_size,
        topology=topology,
    )


def _conv_source_columns(
    snapshot: _AffineSnapshot,
    *,
    output_channel: int,
    output_spatial: np.ndarray,
    batch_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    topology = snapshot.topology
    if topology is None:
        raise ExactReLULiveRowStreamError("CONV snapshot lost its topology")
    batch, in_ch, in_h, in_w = topology.input_shape
    _out_batch, out_ch, _out_h, _out_w = topology.output_shape
    input_area = in_h * in_w
    in_ch_per_group = int(snapshot.weight.shape[1])
    out_ch_per_group = out_ch // topology.groups
    group = int(output_channel) // out_ch_per_group
    channels = group * in_ch_per_group + np.arange(in_ch_per_group, dtype=np.int64)
    spatial = topology.input_spatial_by_output[output_spatial]
    columns = (
        batch_index * in_ch * input_area
        + channels[None, :, None] * input_area
        + np.maximum(spatial[:, None, :], 0)
    )
    valid = spatial[:, None, :] >= 0
    return columns, valid


def _affine_support_forward(
    snapshot: _AffineSnapshot, source_mask: np.ndarray
) -> np.ndarray:
    source_mask = np.asarray(source_mask, dtype=bool).reshape(-1)
    if source_mask.size != snapshot.input_size:
        raise ExactReLULiveRowStreamError("affine support input width mismatch")
    support = snapshot.weight != 0.0
    if snapshot.kind == "DENSE":
        return np.any(support & source_mask[None, :], axis=1)
    topology = snapshot.topology
    assert topology is not None
    batch, _in_ch, _in_h, _in_w = topology.input_shape
    _out_batch, out_ch, out_h, out_w = topology.output_shape
    output_area = out_h * out_w
    flat_support = support.reshape(out_ch, support.shape[1], -1)
    result = np.zeros((batch, out_ch, output_area), dtype=bool)
    spatial_rows = np.arange(output_area, dtype=np.int64)
    for n in range(batch):
        for co in range(out_ch):
            columns, valid = _conv_source_columns(
                snapshot,
                output_channel=co,
                output_spatial=spatial_rows,
                batch_index=n,
            )
            result[n, co] = np.any(
                valid
                & flat_support[co][None, :, :]
                & source_mask[columns],
                axis=(1, 2),
            )
    return result.reshape(-1)


def _affine_support_backward(
    snapshot: _AffineSnapshot, output_mask: np.ndarray
) -> np.ndarray:
    output_mask = np.asarray(output_mask, dtype=bool).reshape(-1)
    if output_mask.size != snapshot.output_size:
        raise ExactReLULiveRowStreamError("affine demand output width mismatch")
    support = snapshot.weight != 0.0
    result = np.zeros(snapshot.input_size, dtype=bool)
    if snapshot.kind == "DENSE":
        selected = np.flatnonzero(output_mask)
        if selected.size:
            result[np.any(support[selected], axis=0)] = True
        return result
    topology = snapshot.topology
    assert topology is not None
    batch, _in_ch, _in_h, _in_w = topology.input_shape
    _out_batch, out_ch, out_h, out_w = topology.output_shape
    output_area = out_h * out_w
    flat_support = support.reshape(out_ch, support.shape[1], -1)
    shaped = output_mask.reshape(batch, out_ch, output_area)
    for n in range(batch):
        for co in range(out_ch):
            spatial_rows = np.flatnonzero(shaped[n, co]).astype(np.int64)
            if not spatial_rows.size:
                continue
            columns, valid = _conv_source_columns(
                snapshot,
                output_channel=co,
                output_spatial=spatial_rows,
                batch_index=n,
            )
            keep = valid & flat_support[co][None, :, :]
            result[columns[keep]] = True
    return result


def _selected_affine_matrix(
    snapshot: _AffineSnapshot,
    selected_rows: np.ndarray,
    source_possible: np.ndarray,
    *,
    name: str,
) -> sp.csr_matrix:
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    source_possible = np.asarray(source_possible, dtype=bool).reshape(-1)
    if (
        source_possible.size != snapshot.input_size
        or (selected_rows.size and (
            np.any(selected_rows < 0)
            or np.any(selected_rows >= snapshot.output_size)
            or np.any(selected_rows[1:] <= selected_rows[:-1])
        ))
    ):
        raise ExactReLULiveRowStreamError(f"{name} selection is malformed")
    if snapshot.kind == "DENSE":
        live_columns = np.flatnonzero(source_possible).astype(np.int64)
        local = sp.csr_matrix(
            snapshot.weight[np.ix_(selected_rows, live_columns)],
            dtype=np.float64,
        )
        matrix = sp.csr_matrix(
            (local.data, live_columns[local.indices], local.indptr),
            shape=(selected_rows.size, snapshot.input_size),
            dtype=np.float64,
        )
        return _canonical(matrix, name=name)
    topology = snapshot.topology
    assert topology is not None
    batch, _in_ch, _in_h, _in_w = topology.input_shape
    _out_batch, out_ch, out_h, out_w = topology.output_shape
    output_area = out_h * out_w
    flat_weight = snapshot.weight.reshape(out_ch, snapshot.weight.shape[1], -1)
    counts = np.zeros(selected_rows.size, dtype=np.int64)
    groups: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    row_batch = selected_rows // (out_ch * output_area)
    remainder = selected_rows % (out_ch * output_area)
    row_channel = remainder // output_area
    row_spatial = remainder % output_area
    for n in range(batch):
        for co in range(out_ch):
            positions = np.flatnonzero((row_batch == n) & (row_channel == co)).astype(np.int64)
            if not positions.size:
                continue
            spatial_rows = row_spatial[positions]
            columns, valid = _conv_source_columns(
                snapshot,
                output_channel=co,
                output_spatial=spatial_rows,
                batch_index=n,
            )
            values = np.take(
                flat_weight[co], topology.kernel_gather_by_output[spatial_rows], axis=1
            ).transpose(1, 0, 2)
            keep = valid & (values != 0.0) & source_possible[columns]
            counts[positions] = np.count_nonzero(keep, axis=(1, 2))
            groups.append((positions, columns, values, keep))
    total = int(np.sum(counts, dtype=np.int64))
    if total > np.iinfo(np.int32).max:
        raise ExactReLULiveRowStreamError(f"{name} exceeds int32 CSR capacity")
    indptr = np.empty(selected_rows.size + 1, dtype=np.int32)
    indptr[0] = 0
    indptr[1:] = np.cumsum(counts, dtype=np.int64)
    indices = np.empty(total, dtype=np.int32)
    data = np.empty(total, dtype=np.float64)
    for positions, columns, values, keep in groups:
        # ``selected_rows`` is strictly increasing and groups are traversed in
        # flattened output order, so every (batch, channel) group occupies one
        # contiguous destination interval.  Boolean indexing preserves the
        # old row-major (ci, kr, kc) order while moving the proportional copy
        # out of the Python per-output-row loop.
        if (
            not positions.size
            or int(positions[-1]) - int(positions[0]) + 1 != positions.size
        ):
            raise ExactReLULiveRowStreamError(
                f"{name} Conv selection lost contiguous group order"
            )
        start = int(indptr[int(positions[0])])
        stop = int(indptr[int(positions[-1]) + 1])
        selected_columns = columns[keep]
        selected_values = values[keep]
        if selected_columns.size != stop - start or selected_values.size != stop - start:
            raise ExactReLULiveRowStreamError(
                f"{name} Conv direct emission count drifted"
            )
        indices[start:stop] = selected_columns
        data[start:stop] = selected_values
    return _canonical(
        sp.csr_matrix(
            (data, indices, indptr),
            shape=(selected_rows.size, snapshot.input_size),
            dtype=np.float64,
        ),
        name=name,
    )


@triton.jit
def _gpu_conv_csr_count_kernel(
    selected_rows,
    source_possible,
    counts,
    in_channels,
    in_height,
    in_width,
    out_channels,
    out_height,
    out_width,
    in_per_group,
    out_per_group,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    pad_height,
    pad_width,
    dilation_height,
    dilation_width,
    kernel_elements,
):
    local_row = tl.program_id(0)
    output_row = tl.load(selected_rows + local_row)
    output_area = out_height * out_width
    batch_stride = out_channels * output_area
    batch_index = output_row // batch_stride
    remainder = output_row - batch_index * batch_stride
    output_channel = remainder // output_area
    output_spatial = remainder - output_channel * output_area
    output_row_spatial = output_spatial // out_width
    output_column_spatial = output_spatial - output_row_spatial * out_width
    group = output_channel // out_per_group
    input_area = in_height * in_width
    count = 0
    flat_kernel = 0
    while flat_kernel < kernel_elements:
        channel_local = flat_kernel // (kernel_height * kernel_width)
        kernel_remainder = flat_kernel - channel_local * kernel_height * kernel_width
        kernel_row = kernel_remainder // kernel_width
        kernel_column = kernel_remainder - kernel_row * kernel_width
        input_row_spatial = (
            output_row_spatial * stride_height
            - pad_height
            + kernel_row * dilation_height
        )
        input_column_spatial = (
            output_column_spatial * stride_width
            - pad_width
            + kernel_column * dilation_width
        )
        valid = (
            (input_row_spatial >= 0)
            & (input_row_spatial < in_height)
            & (input_column_spatial >= 0)
            & (input_column_spatial < in_width)
        )
        input_channel = group * in_per_group + channel_local
        source_row = (
            batch_index * in_channels * input_area
            + input_channel * input_area
            + input_row_spatial * in_width
            + input_column_spatial
        )
        safe_source_row = tl.where(valid, source_row, 0)
        possible = tl.load(
            source_possible + safe_source_row, mask=valid, other=0
        ) != 0
        count += tl.where(valid & possible, 1, 0)
        flat_kernel += 1
    tl.store(counts + local_row, count)


@triton.jit
def _gpu_conv_csr_emit_kernel(
    selected_rows,
    source_possible,
    weight,
    indptr,
    indices,
    data,
    in_channels,
    in_height,
    in_width,
    out_channels,
    out_height,
    out_width,
    in_per_group,
    out_per_group,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    pad_height,
    pad_width,
    dilation_height,
    dilation_width,
    kernel_elements,
):
    local_row = tl.program_id(0)
    output_row = tl.load(selected_rows + local_row)
    output_area = out_height * out_width
    batch_stride = out_channels * output_area
    batch_index = output_row // batch_stride
    remainder = output_row - batch_index * batch_stride
    output_channel = remainder // output_area
    output_spatial = remainder - output_channel * output_area
    output_row_spatial = output_spatial // out_width
    output_column_spatial = output_spatial - output_row_spatial * out_width
    group = output_channel // out_per_group
    input_area = in_height * in_width
    cursor = tl.load(indptr + local_row)
    flat_kernel = 0
    while flat_kernel < kernel_elements:
        channel_local = flat_kernel // (kernel_height * kernel_width)
        kernel_remainder = flat_kernel - channel_local * kernel_height * kernel_width
        kernel_row = kernel_remainder // kernel_width
        kernel_column = kernel_remainder - kernel_row * kernel_width
        input_row_spatial = (
            output_row_spatial * stride_height
            - pad_height
            + kernel_row * dilation_height
        )
        input_column_spatial = (
            output_column_spatial * stride_width
            - pad_width
            + kernel_column * dilation_width
        )
        valid = (
            (input_row_spatial >= 0)
            & (input_row_spatial < in_height)
            & (input_column_spatial >= 0)
            & (input_column_spatial < in_width)
        )
        input_channel = group * in_per_group + channel_local
        source_row = (
            batch_index * in_channels * input_area
            + input_channel * input_area
            + input_row_spatial * in_width
            + input_column_spatial
        )
        safe_source_row = tl.where(valid, source_row, 0)
        possible = tl.load(
            source_possible + safe_source_row, mask=valid, other=0
        ) != 0
        keep = valid & possible
        coefficient = tl.load(
            weight
            + output_channel * in_per_group * kernel_height * kernel_width
            + flat_kernel
        )
        tl.store(indices + cursor, source_row, mask=keep)
        tl.store(data + cursor, coefficient, mask=keep)
        cursor += tl.where(keep, 1, 0)
        flat_kernel += 1


def _gpu_selected_affine_matrix(
    snapshot: _AffineSnapshot,
    selected_rows: np.ndarray,
    source_possible: np.ndarray,
    *,
    name: str,
) -> _DeviceCSR:
    """Emit the exact selected CSR on CUDA in the all-nonzero domain."""

    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    source_possible = np.asarray(source_possible, dtype=bool).reshape(-1)
    if (
        source_possible.size != snapshot.input_size
        or (
            selected_rows.size
            and (
                np.any(selected_rows < 0)
                or np.any(selected_rows >= snapshot.output_size)
                or np.any(selected_rows[1:] <= selected_rows[:-1])
            )
        )
        or np.any(snapshot.weight == 0.0)
    ):
        raise ExactReLULiveRowStreamError(
            f"{name} is outside the GPU all-nonzero CSR domain"
        )
    selected = torch.as_tensor(
        selected_rows, dtype=torch.int64, device="cuda"
    )
    possible = torch.as_tensor(
        source_possible, dtype=torch.uint8, device="cuda"
    )
    rows = int(selected.numel())
    if snapshot.kind == "DENSE":
        live_columns = torch.nonzero(
            possible, as_tuple=False
        ).reshape(-1).to(torch.int64)
        row_width = int(live_columns.numel())
        total = rows * row_width
        if total > np.iinfo(np.int32).max:
            raise ExactReLULiveRowStreamError(
                f"{name} exceeds int32 CSR capacity"
            )
        weight = torch.tensor(
            snapshot.weight, dtype=torch.float64, device="cuda"
        )
        data = weight.index_select(0, selected).index_select(
            1, live_columns
        ).reshape(-1)
        indices = live_columns.repeat(rows)
        indptr = (
            torch.arange(
                0, total + 1, row_width, dtype=torch.int64, device="cuda"
            )
            if row_width
            else torch.zeros(rows + 1, dtype=torch.int64, device="cuda")
        )
        return _DeviceCSR(indptr, indices, data, rows, snapshot.input_size)

    topology = snapshot.topology
    if snapshot.kind != "CONV2D" or topology is None:
        raise ExactReLULiveRowStreamError(
            f"{name} has unsupported affine geometry"
        )
    _batch, in_channels, input_height, input_width = topology.input_shape
    _out_batch, out_channels, output_height, output_width = topology.output_shape
    in_per_group = int(snapshot.weight.shape[1])
    kernel_height = int(snapshot.weight.shape[2])
    kernel_width = int(snapshot.weight.shape[3])
    out_per_group = out_channels // topology.groups
    kernel_elements = in_per_group * kernel_height * kernel_width
    counts = torch.empty(rows, dtype=torch.int64, device="cuda")
    if rows:
        _gpu_conv_csr_count_kernel[(rows,)](
            selected,
            possible,
            counts,
            in_channels,
            input_height,
            input_width,
            out_channels,
            output_height,
            output_width,
            in_per_group,
            out_per_group,
            kernel_height,
            kernel_width,
            topology.stride[0],
            topology.stride[1],
            topology.padding[0],
            topology.padding[1],
            topology.dilation[0],
            topology.dilation[1],
            kernel_elements,
        )
    indptr = torch.empty(rows + 1, dtype=torch.int64, device="cuda")
    indptr[0] = 0
    if rows:
        torch.cumsum(counts, dim=0, out=indptr[1:])
    total = int(indptr[-1].item())
    if total > np.iinfo(np.int32).max:
        raise ExactReLULiveRowStreamError(
            f"{name} exceeds int32 CSR capacity"
        )
    indices = torch.empty(total, dtype=torch.int64, device="cuda")
    data = torch.empty(total, dtype=torch.float64, device="cuda")
    weight = torch.tensor(
        snapshot.weight, dtype=torch.float64, device="cuda"
    )
    if rows:
        _gpu_conv_csr_emit_kernel[(rows,)](
            selected,
            possible,
            weight,
            indptr,
            indices,
            data,
            in_channels,
            input_height,
            input_width,
            out_channels,
            output_height,
            output_width,
            in_per_group,
            out_per_group,
            kernel_height,
            kernel_width,
            topology.stride[0],
            topology.stride[1],
            topology.padding[0],
            topology.padding[1],
            topology.dilation[0],
            topology.dilation[1],
            kernel_elements,
        )
    return _DeviceCSR(indptr, indices, data, rows, snapshot.input_size)


@triton.jit
def _ordered_csr_dense_kernel(
    indptr,
    indices,
    data,
    dense,
    output,
    width,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    active_lane = lane < width
    start = tl.load(indptr + row)
    stop = tl.load(indptr + row + 1)
    accumulator = tl.zeros((BLOCK,), dtype=tl.float64)
    cursor = start
    while cursor < stop:
        source_row = tl.load(indices + cursor)
        coefficient = tl.load(data + cursor)
        value = tl.load(
            dense + source_row * width + lane,
            mask=active_lane,
            other=0.0,
        )
        accumulator += coefficient * value
        cursor += 1
    tl.store(output + row * width + lane, accumulator, mask=active_lane)


def _device_csr(matrix: sp.csr_matrix) -> _DeviceCSR:
    return _DeviceCSR(
        torch.as_tensor(matrix.indptr, dtype=torch.int64, device="cuda"),
        torch.as_tensor(matrix.indices, dtype=torch.int64, device="cuda"),
        torch.as_tensor(matrix.data, dtype=torch.float64, device="cuda"),
        int(matrix.shape[0]),
        int(matrix.shape[1]),
    )


def _ordered_csr_dense(matrix: _DeviceCSR, dense: torch.Tensor) -> torch.Tensor:
    if dense.ndim != 2 or int(dense.shape[0]) != matrix.columns:
        raise ExactReLULiveRowStreamError("ordered CSR input shape drifted")
    width = int(dense.shape[1])
    if width < 1 or width > _FACTOR_BATCH:
        raise ExactReLULiveRowStreamError("ordered CSR factor batch drifted")
    result = torch.empty(
        (matrix.rows, width), dtype=torch.float64, device="cuda"
    )
    if matrix.rows:
        _ordered_csr_dense_kernel[(matrix.rows,)](
            matrix.indptr,
            matrix.indices,
            matrix.data,
            dense,
            result,
            width,
            BLOCK=_FACTOR_BATCH,
            num_warps=2,
        )
    return result


def _positive_gpu_result_upper(
    rounded: np.ndarray,
    fanin: np.ndarray,
    active: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    rounded = np.asarray(rounded, dtype=np.float64).reshape(-1)
    fanin = np.asarray(fanin, dtype=np.float64).reshape(-1)
    active = np.asarray(active, dtype=bool).reshape(-1)
    if (
        rounded.size != fanin.size
        or rounded.size != active.size
        or np.any(rounded < 0.0)
        or not np.all(np.isfinite(rounded))
        or np.any(fanin < 0.0)
        or not np.all(np.isfinite(fanin))
    ):
        raise ExactReLULiveRowStreamError(f"{name} has malformed nonnegative operands")
    try:
        return _oh._inflate_nonnegative(
            rounded,
            2.0 * fanin + 2.0,
            active=active,
            name=name,
        )
    except Exception as exc:
        raise ExactReLULiveRowStreamError(f"{name} could not be rounded outward") from exc


def _gpu_affine_shadow(
    source: _Shadow,
    snapshot: _AffineSnapshot,
    *,
    layer_id: int,
) -> _Shadow:
    device_weight = torch.tensor(
        snapshot.weight, dtype=torch.float64, device="cuda"
    )
    if snapshot.kind == "DENSE":
        source_center = torch.as_tensor(
            source.center.reshape(-1, 1), dtype=torch.float64, device="cuda"
        )
        source_nonnegative = torch.as_tensor(
            np.column_stack((source.mass_upper, source.error)),
            dtype=torch.float64,
            device="cuda",
        )
        raw_center_tensor = torch.matmul(device_weight, source_center)
        raw_nonnegative_tensor = torch.matmul(
            torch.abs(device_weight), source_nonnegative
        )
    else:
        topology = snapshot.topology
        assert topology is not None
        batch, in_ch, in_h, in_w = topology.input_shape
        stride = topology.stride
        padding = topology.padding
        dilation = topology.dilation
        groups = topology.groups
        source_center = torch.as_tensor(
            source.center.reshape(batch, in_ch, in_h, in_w),
            dtype=torch.float64,
            device="cuda",
        )
        raw_center_tensor = torch.nn.functional.conv2d(
            source_center,
            device_weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        nonnegative = np.stack((source.mass_upper, source.error), axis=0)
        source_nonnegative = torch.as_tensor(
            nonnegative.reshape(2 * batch, in_ch, in_h, in_w),
            dtype=torch.float64,
            device="cuda",
        )
        raw_nonnegative_tensor = torch.nn.functional.conv2d(
            source_nonnegative,
            torch.abs(device_weight),
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        raw_nonnegative_tensor = raw_nonnegative_tensor.reshape(
            2, snapshot.output_size
        ).transpose(0, 1)
    raw_center = raw_center_tensor.detach().cpu().numpy().reshape(-1)
    raw_nonnegative = raw_nonnegative_tensor.detach().cpu().numpy().reshape(
        snapshot.output_size, 2
    )
    center = raw_center + snapshot.bias
    fanin = _affine_fanin(snapshot)
    transformed = _positive_gpu_result_upper(
        raw_nonnegative[:, 0],
        fanin,
        _affine_support_forward(snapshot, source.mass_upper > 0.0),
        name=f"live_row.mass[{layer_id}]",
    )
    arithmetic_mass = _oh._nonnegative_sum_upper(
        transformed,
        np.abs(snapshot.bias),
        name=f"live_row.arithmetic_mass[{layer_id}]",
    )
    propagated = _positive_gpu_result_upper(
        raw_nonnegative[:, 1],
        fanin,
        _affine_support_forward(snapshot, source.error > 0.0),
        name=f"live_row.propagated_error[{layer_id}]",
    )
    arithmetic_error = _oh._inflate_nonnegative(
        _oh._gamma_ops(2.0 * fanin + 2.0, name=f"live_row.gamma[{layer_id}]")
        * arithmetic_mass,
        4,
        active=arithmetic_mass > 0.0,
        name=f"live_row.arithmetic_error[{layer_id}]",
    )
    error = _oh._nonnegative_sum_upper(
        propagated, arithmetic_error, name=f"live_row.total_error[{layer_id}]"
    )
    mass = _oh._nonnegative_sum_upper(
        transformed, np.abs(snapshot.bias), arithmetic_error,
        name=f"live_row.output_mass[{layer_id}]",
    )
    if not np.all(np.isfinite(center)):
        raise ExactReLULiveRowStreamError(f"affine layer {layer_id} center overflowed")
    return _Shadow(center, error, mass)


def _affine_fanin(snapshot: _AffineSnapshot) -> np.ndarray:
    support = snapshot.weight != 0.0
    if snapshot.kind == "DENSE":
        return np.count_nonzero(support, axis=1).astype(np.float64)
    topology = snapshot.topology
    assert topology is not None
    _batch, _in_ch, _in_h, _in_w = topology.input_shape
    out_batch, out_ch, _out_h, _out_w = topology.output_shape
    flat_support = support.reshape(out_ch, support.shape[1], -1)
    support_counts = np.count_nonzero(flat_support, axis=1).astype(np.int64)
    valid = topology.input_spatial_by_output >= 0
    counts = support_counts @ valid.astype(np.int64).T
    return np.tile(counts.reshape(-1), out_batch).astype(np.float64)


def _add_shadow(left: _Shadow, right: _Shadow, *, layer_id: int) -> _Shadow:
    if left.center.size != right.center.size:
        raise ExactReLULiveRowStreamError(f"ADD layer {layer_id} shape mismatch")
    center = left.center + right.center
    base_mass = _oh._nonnegative_sum_upper(
        left.mass_upper, right.mass_upper, name=f"live_row.add_mass[{layer_id}]"
    )
    arithmetic = _oh._inflate_nonnegative(
        _oh._gamma_ops(8, name=f"live_row.add_gamma[{layer_id}]") * base_mass,
        4,
        active=base_mass > 0.0,
        name=f"live_row.add_arithmetic_error[{layer_id}]",
    )
    error = _oh._nonnegative_sum_upper(
        left.error, right.error, arithmetic, name=f"live_row.add_error[{layer_id}]"
    )
    mass = _oh._nonnegative_sum_upper(
        base_mass, arithmetic, name=f"live_row.add_output_mass[{layer_id}]"
    )
    if not np.all(np.isfinite(center)):
        raise ExactReLULiveRowStreamError(f"ADD layer {layer_id} center overflowed")
    return _Shadow(center, error, mass)


def _relu_shadow(source: _Shadow, frame: _PhaseFrame, *, layer_id: int) -> _Shadow:
    size = source.center.size
    center = np.zeros(size, dtype=np.float64)
    error = np.zeros(size, dtype=np.float64)
    mass = np.zeros(size, dtype=np.float64)
    center[frame.active] = source.center[frame.active]
    error[frame.active] = source.error[frame.active]
    mass[frame.active] = source.mass_upper[frame.active]
    if frame.exact.size:
        upper_half = 0.5 * frame.upper[frame.exact]
        if not np.all(np.isfinite(upper_half)) or np.any(
            2.0 * upper_half != frame.upper[frame.exact]
        ):
            raise ExactReLULiveRowStreamError(
                f"ReLU {layer_id} has an inexact upper half"
            )
        center[frame.exact] = upper_half
        mass[frame.exact] = frame.upper[frame.exact]
    return _Shadow(center, error, mass)


def _make_phase_frames(
    order: Sequence[Any],
    before: Mapping[int, Any],
    *,
    first_continuous_column: int,
) -> Tuple[Dict[int, _PhaseFrame], int, int]:
    continuous = int(first_continuous_column)
    binary = 0
    frames: Dict[int, _PhaseFrame] = {}
    for layer in order:
        if _oh._kind(layer.kind) != "RELU":
            continue
        layer_id = int(layer.id)
        lower, upper = _facts_box(
            before, layer_id, len(layer.out_vars), name=f"ReLU[{layer_id}].before"
        )
        active = np.flatnonzero((lower >= 0.0) & (upper > 0.0)).astype(np.int64)
        inactive = np.flatnonzero(upper <= 0.0).astype(np.int64)
        exact = np.flatnonzero((lower < 0.0) & (upper > 0.0)).astype(np.int64)
        if active.size + inactive.size + exact.size != lower.size:
            raise ExactReLULiveRowStreamError(f"ReLU {layer_id} phase partition is incomplete")
        if binary + exact.size > _MAX_EXACT_ROWS:
            raise ExactReLULiveRowStreamError("exact-row cap exceeded")
        cont_columns = np.arange(continuous, continuous + exact.size, dtype=np.int64)
        bin_columns = np.arange(binary, binary + exact.size, dtype=np.int64)
        continuous += int(exact.size)
        binary += int(exact.size)
        frames[layer_id] = _PhaseFrame(
            lower,
            upper,
            active,
            inactive,
            exact,
            cont_columns,
            bin_columns,
            0.5 * upper[exact],
            exact.copy(),
            cont_columns.copy(),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    return frames, continuous, binary


def _live_rows(
    net: Any,
    order: Sequence[Any],
    affines: Mapping[int, _AffineSnapshot],
    frames: Mapping[int, _PhaseFrame],
    input_variable_rows: np.ndarray,
    output_layer_id: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    possible = {
        int(layer.id): np.zeros(len(layer.out_vars), dtype=bool) for layer in order
    }
    input_layer = next(layer for layer in order if _oh._kind(layer.kind) == "INPUT")
    possible[int(input_layer.id)][input_variable_rows] = True
    for layer in order:
        layer_id = int(layer.id)
        kind = _oh._kind(layer.kind)
        if kind == "INPUT":
            continue
        predecessors = tuple(int(v) for v in net.preds.get(layer_id, []))
        if kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
            possible[layer_id] = possible[predecessors[0]].copy()
        elif kind in {"CONV2D", "DENSE"}:
            possible[layer_id] = _affine_support_forward(
                affines[layer_id], possible[predecessors[0]]
            )
        elif kind == "ADD":
            possible[layer_id] = possible[predecessors[0]] | possible[predecessors[1]]
        elif kind == "RELU":
            frame = frames[layer_id]
            possible[layer_id][frame.active] = possible[predecessors[0]][frame.active]
            possible[layer_id][frame.exact] = True

    demand = {
        int(layer.id): np.zeros(len(layer.out_vars), dtype=bool) for layer in order
    }
    demand[int(output_layer_id)][:] = True
    for layer in reversed(order):
        layer_id = int(layer.id)
        kind = _oh._kind(layer.kind)
        predecessors = tuple(int(v) for v in net.preds.get(layer_id, []))
        rows = demand[layer_id].copy()
        if kind == "RELU":
            frame = frames[layer_id]
            rows[frame.exact] = True
            demand[layer_id] = rows
            needed = np.zeros(rows.size, dtype=bool)
            needed[frame.active] = rows[frame.active]
            needed[frame.exact] = True
            demand[predecessors[0]] |= needed
        elif kind in {"CONV2D", "DENSE"}:
            demand[predecessors[0]] |= _affine_support_backward(
                affines[layer_id], rows
            )
        elif kind == "ADD":
            for predecessor in predecessors:
                demand[predecessor] |= rows
        elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"} and predecessors:
            demand[predecessors[0]] |= rows
    live = {
        layer_id: np.flatnonzero(demand[layer_id] & possible[layer_id]).astype(np.int64)
        for layer_id in demand
    }
    possible_rows = {
        layer_id: np.asarray(possible[layer_id], dtype=bool)
        for layer_id in possible
    }
    return live, possible_rows


def _build_constraints(
    order: Sequence[Any],
    frames: Mapping[int, _PhaseFrame],
    preactivation_generators: Mapping[int, sp.csr_matrix],
    preactivation_shadows: Mapping[int, _Shadow],
    *,
    n_cont: int,
    n_bin: int,
) -> Tuple[
    sp.csr_matrix,
    sp.csr_matrix,
    np.ndarray,
    sp.csr_matrix,
    sp.csr_matrix,
    np.ndarray,
]:
    equality_continuous_blocks = []
    equality_binary_blocks = []
    equality_rhs_blocks = []
    continuous_blocks = []
    binary_blocks = []
    upper_blocks = []
    for layer in order:
        layer_id = int(layer.id)
        frame = frames.get(layer_id)
        if frame is None or (
            not frame.exact.size and not frame.deferred_rows.size
        ):
            continue
        stream = _pad_columns(preactivation_generators[layer_id], n_cont)
        shadow = preactivation_shadows[layer_id]

        def stream_positions(rows: np.ndarray, *, kind: str) -> np.ndarray:
            positions = np.searchsorted(frame.stream_rows, rows)
            if (
                positions.size != rows.size
                or np.any(positions >= frame.stream_rows.size)
                or not np.array_equal(frame.stream_rows[positions], rows)
            ):
                raise ExactReLULiveRowStreamError(
                    f"ReLU {layer_id} {kind} rows escaped the generator stream"
                )
            return positions.astype(np.int64, copy=False)

        def linked_positions(rows: np.ndarray, *, kind: str) -> np.ndarray:
            positions = np.searchsorted(frame.linked_rows, rows)
            if (
                positions.size != rows.size
                or np.any(positions >= frame.linked_rows.size)
                or not np.array_equal(frame.linked_rows[positions], rows)
            ):
                raise ExactReLULiveRowStreamError(
                    f"ReLU {layer_id} {kind} rows escaped the local link"
                )
            return positions.astype(np.int64, copy=False)

        linked_proxy = sp.csr_matrix(
            (frame.linked_rows.size, n_cont), dtype=np.float64
        )
        if frame.linked_rows.size:
            linked_stream_positions = stream_positions(
                frame.linked_rows, kind="constraint-local"
            )
            linked_pre = stream[linked_stream_positions]
            linked_scale = _oh._row_l1_upper(
                linked_pre, name=f"live_row.link_scale[{layer_id}]"
            )
            linked_scale[linked_scale == 0.0] = 1.0
            if (
                not np.all(np.isfinite(linked_scale))
                or np.any(linked_scale <= 0.0)
            ):
                raise ExactReLULiveRowStreamError(
                    f"ReLU {layer_id} local link scale is invalid"
                )
            local = np.arange(frame.linked_rows.size, dtype=np.int64)
            linked_proxy = sp.csr_matrix(
                (linked_scale, (local, frame.linked_continuous_columns)),
                shape=(frame.linked_rows.size, n_cont),
                dtype=np.float64,
            )
            equality_continuous_blocks.append(
                _canonical(
                    linked_pre - linked_proxy,
                    name=f"live_row.link[{layer_id}]",
                )
            )
            equality_binary_blocks.append(
                sp.csr_matrix((frame.linked_rows.size, n_bin), dtype=np.float64)
            )
            equality_rhs_blocks.append(
                np.zeros(frame.linked_rows.size, dtype=np.float64)
            )

        if frame.exact.size:
            rows = frame.exact
            positions = stream_positions(rows, kind="exact")
            pre = linked_proxy[linked_positions(rows, kind="exact")]
            count = int(rows.size)
            lower_half = -0.5 * frame.lower[rows]
            upper_half = 0.5 * frame.upper[rows]
            output_half = frame.stream_half_widths[positions]
            if (
                not np.all(np.isfinite(lower_half))
                or not np.all(np.isfinite(upper_half))
                or not np.all(np.isfinite(output_half))
                or np.any(2.0 * lower_half != -frame.lower[rows])
                or np.any(2.0 * upper_half != frame.upper[rows])
            ):
                raise ExactReLULiveRowStreamError(
                    f"ReLU {layer_id} has an inexact Big-M half"
                )
            local = np.arange(count, dtype=np.int64)
            y = sp.csr_matrix(
                (output_half, (local, frame.continuous_columns)),
                shape=(count, n_cont),
                dtype=np.float64,
            )
            lower_A = _canonical(pre - y, name=f"live_row.lower[{layer_id}]")
            x_branch_A = _canonical(-pre + y, name=f"live_row.x_branch[{layer_id}]")
            continuous_blocks.append(
                sp.vstack((lower_A, x_branch_A, y), format="csr")
            )
            x_binary = sp.csr_matrix(
                (lower_half, (local, frame.binary_columns)),
                shape=(count, n_bin), dtype=np.float64,
            )
            zero_binary = sp.csr_matrix(
                (-upper_half, (local, frame.binary_columns)),
                shape=(count, n_bin), dtype=np.float64,
            )
            binary_blocks.append(
                sp.vstack(
                    (sp.csr_matrix((count, n_bin)), x_binary, zero_binary),
                    format="csr",
                )
            )
            center = shadow.center[rows]
            error = shadow.error[rows]
            lower_rhs = np.asarray(
                [
                    _upper_exact_sum(-float(cx), float(scale), float(er))
                    for cx, scale, er in zip(center, output_half, error)
                ],
                dtype=np.float64,
            )
            x_rhs = np.asarray(
                [
                    _upper_exact_sum(
                        float(lh), float(cx), -float(scale), float(er)
                    )
                    for lh, cx, scale, er in zip(
                        lower_half, center, output_half, error
                    )
                ],
                dtype=np.float64,
            )
            zero_rhs = np.asarray(
                [
                    _upper_exact_sum(float(uh), -float(scale))
                    for uh, scale in zip(upper_half, output_half)
                ],
                dtype=np.float64,
            )
            upper_blocks.append(
                np.concatenate((lower_rhs, x_rhs, zero_rhs))
            )

        if frame.deferred_rows.size:
            rows = frame.deferred_rows
            positions = stream_positions(rows, kind="deferred-stable")
            count = int(rows.size)
            output_half = frame.stream_half_widths[positions]
            if (
                not np.all(np.isfinite(output_half))
            ):
                raise ExactReLULiveRowStreamError(
                    f"ReLU {layer_id} has an inexact deferred upper half"
                )
            local = np.arange(count, dtype=np.int64)
            y = sp.csr_matrix(
                (output_half, (local, frame.deferred_continuous_columns)),
                shape=(count, n_cont),
                dtype=np.float64,
            )
            center = shadow.center[rows]
            error = shadow.error[rows]
            active = frame.deferred_active
            if active.size != count:
                raise ExactReLULiveRowStreamError(
                    f"ReLU {layer_id} deferred phase metadata drifted"
                )
            if np.any(active):
                active_rows = np.flatnonzero(active).astype(np.int64)
                pre = linked_proxy[
                    linked_positions(rows[active_rows], kind="deferred-active")
                ]
                lower_A = _canonical(
                    pre - y[active_rows],
                    name=f"live_row.deferred_active_lower[{layer_id}]",
                )
                upper_A = _canonical(
                    -pre + y[active_rows],
                    name=f"live_row.deferred_active_upper[{layer_id}]",
                )
                continuous_blocks.append(
                    sp.vstack((lower_A, upper_A), format="csr")
                )
                binary_blocks.append(
                    sp.csr_matrix((2 * active_rows.size, n_bin), dtype=np.float64)
                )
                active_center = center[active_rows]
                active_half = output_half[active_rows]
                active_error = error[active_rows]
                lower_rhs = np.asarray(
                    [
                        _upper_exact_sum(-float(cx), float(uh), float(er))
                        for cx, uh, er in zip(
                            active_center, active_half, active_error
                        )
                    ],
                    dtype=np.float64,
                )
                upper_rhs = np.asarray(
                    [
                        _upper_exact_sum(float(cx), -float(uh), float(er))
                        for cx, uh, er in zip(
                            active_center, active_half, active_error
                        )
                    ],
                    dtype=np.float64,
                )
                upper_blocks.append(np.concatenate((lower_rhs, upper_rhs)))
            if np.any(~active):
                inactive_rows = np.flatnonzero(~active).astype(np.int64)
                continuous_blocks.append(y[inactive_rows])
                binary_blocks.append(
                    sp.csr_matrix((inactive_rows.size, n_bin), dtype=np.float64)
                )
                upper_blocks.append(
                    np.asarray(
                        [
                            _upper_exact_sum(-float(value))
                            for value in output_half[inactive_rows]
                        ],
                        dtype=np.float64,
                    )
                )

    equality_continuous = (
        _canonical(
            sp.vstack(equality_continuous_blocks, format="csr"),
            name="live_row.Ac",
        )
        if equality_continuous_blocks
        else sp.csr_matrix((0, n_cont), dtype=np.float64)
    )
    equality_binary = (
        _canonical(
            sp.vstack(equality_binary_blocks, format="csr"),
            name="live_row.Ab",
        )
        if equality_binary_blocks
        else sp.csr_matrix((0, n_bin), dtype=np.float64)
    )
    equality_rhs = (
        np.ascontiguousarray(np.concatenate(equality_rhs_blocks), dtype=np.float64)
        if equality_rhs_blocks
        else np.zeros(0, dtype=np.float64)
    )
    upper_continuous = (
        _canonical(
            sp.vstack(continuous_blocks, format="csr"), name="live_row.Auc"
        )
        if continuous_blocks
        else sp.csr_matrix((0, n_cont), dtype=np.float64)
    )
    upper_binary = (
        _canonical(sp.vstack(binary_blocks, format="csr"), name="live_row.Aub")
        if binary_blocks
        else sp.csr_matrix((0, n_bin), dtype=np.float64)
    )
    upper_rhs = (
        np.ascontiguousarray(np.concatenate(upper_blocks), dtype=np.float64)
        if upper_blocks
        else np.zeros(0, dtype=np.float64)
    )
    return (
        equality_continuous,
        equality_binary,
        equality_rhs,
        upper_continuous,
        upper_binary,
        upper_rhs,
    )


def _build_shadows(
    net: Any,
    order: Sequence[Any],
    affines: Mapping[int, _AffineSnapshot],
    frames: Mapping[int, _PhaseFrame],
    *,
    input_center: np.ndarray,
    input_radius: np.ndarray,
) -> Tuple[Dict[int, _Shadow], Dict[int, _Shadow], float]:
    input_mass = _oh._nonnegative_sum_upper(
        np.abs(input_center), input_radius, name="live_row.input_mass"
    )
    shadows: Dict[int, _Shadow] = {}
    preactivation: Dict[int, _Shadow] = {}
    elapsed = 0.0
    for layer in order:
        layer_id = int(layer.id)
        kind = _oh._kind(layer.kind)
        if kind == "INPUT":
            shadows[layer_id] = _Shadow(
                input_center.copy(),
                np.zeros(input_center.size, dtype=np.float64),
                input_mass,
            )
        elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
            shadows[layer_id] = shadows[_preds(net, layer, 1)[0]]
        elif kind in {"CONV2D", "DENSE"}:
            source = shadows[_preds(net, layer, 1)[0]]
            started = time.monotonic()
            shadows[layer_id] = _gpu_affine_shadow(
                source,
                affines[layer_id],
                layer_id=layer_id,
            )
            elapsed += time.monotonic() - started
        elif kind == "ADD":
            left, right = _preds(net, layer, 2)
            started = time.monotonic()
            shadows[layer_id] = _add_shadow(
                shadows[left], shadows[right], layer_id=layer_id
            )
            elapsed += time.monotonic() - started
        elif kind == "RELU":
            source = shadows[_preds(net, layer, 1)[0]]
            preactivation[layer_id] = source
            shadows[layer_id] = _relu_shadow(
                source, frames[layer_id], layer_id=layer_id
            )
    return shadows, preactivation, elapsed


def _stream_generators(
    net: Any,
    order: Sequence[Any],
    frames: Mapping[int, _PhaseFrame],
    live_rows: Mapping[int, np.ndarray],
    device_rows: Mapping[int, _DeviceCSR],
    device_row_ids: Mapping[int, torch.Tensor],
    *,
    input_rows: np.ndarray,
    input_radius: np.ndarray,
    n_cont: int,
    assert_layer: Any,
    deadline: Optional[float],
    stage_prefix: str,
    collect_output: bool,
    pointwise: Optional[Mapping[int, np.ndarray]] = None,
) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray], float]:
    preactivation_dense = {
        layer_id: np.empty((frame.stream_rows.size, n_cont), dtype=np.float64)
        for layer_id, frame in frames.items()
    }
    output_dense = (
        np.empty((len(assert_layer.out_vars), n_cont), dtype=np.float64)
        if collect_output
        else None
    )
    successor_uses = {int(layer.id): 0 for layer in order}
    for layer in order:
        for predecessor in net.preds.get(int(layer.id), []):
            successor_uses[int(predecessor)] += 1
    started = time.monotonic()
    finite = torch.ones((), dtype=torch.bool, device="cuda")
    for start in range(0, n_cont, _FACTOR_BATCH):
        _deadline(deadline, f"{stage_prefix}_{start}")
        stop = min(n_cont, start + _FACTOR_BATCH)
        width = stop - start
        values: Dict[int, torch.Tensor] = {}
        remaining_uses = dict(successor_uses)
        for layer in order:
            layer_id = int(layer.id)
            kind = _oh._kind(layer.kind)
            predecessors = tuple(int(v) for v in net.preds.get(layer_id, []))
            if kind == "INPUT":
                value = torch.zeros(
                    (len(layer.out_vars), width), dtype=torch.float64, device="cuda"
                )
                local_columns = np.arange(input_rows.size, dtype=np.int64)
                selected = (local_columns >= start) & (local_columns < stop)
                if np.any(selected):
                    value[
                        torch.as_tensor(input_rows[selected], dtype=torch.int64, device="cuda"),
                        torch.as_tensor(local_columns[selected] - start, dtype=torch.int64, device="cuda"),
                    ] = torch.as_tensor(
                        input_radius[input_rows[selected]], dtype=torch.float64, device="cuda"
                    )
                values[layer_id] = value
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                values[layer_id] = values[_preds(net, layer, 1)[0]]
            elif kind in {"CONV2D", "DENSE"}:
                source = values[_preds(net, layer, 1)[0]]
                selected_value = _ordered_csr_dense(device_rows[layer_id], source)
                value = torch.zeros(
                    (len(layer.out_vars), width), dtype=torch.float64, device="cuda"
                )
                if device_row_ids[layer_id].numel():
                    value[device_row_ids[layer_id]] = selected_value
                values[layer_id] = value
                finite &= torch.isfinite(selected_value).all()
            elif kind in {"SCALE", "BIAS"}:
                if pointwise is None or layer_id not in pointwise:
                    raise ExactReLULiveRowStreamError(
                        f"{kind} generator parameter is unavailable"
                    )
                source = values[_preds(net, layer, 1)[0]]
                if kind == "SCALE":
                    parameter = torch.tensor(
                        pointwise[layer_id],
                        dtype=torch.float64,
                        device="cuda",
                    ).reshape(-1, 1)
                    values[layer_id] = source * parameter
                    finite &= torch.isfinite(values[layer_id]).all()
                else:
                    values[layer_id] = source
            elif kind == "ADD":
                left, right = _preds(net, layer, 2)
                values[layer_id] = values[left] + values[right]
                finite &= torch.isfinite(values[layer_id]).all()
            elif kind == "RELU":
                predecessor = _preds(net, layer, 1)[0]
                source = values[predecessor]
                frame = frames[layer_id]
                stream_device = torch.as_tensor(
                    frame.stream_rows, dtype=torch.int64, device="cuda"
                )
                preactivation_dense[layer_id][:, start:stop] = (
                    source[stream_device].detach().cpu().numpy()
                )
                value = torch.zeros_like(source)
                active_live = np.intersect1d(
                    live_rows[layer_id], frame.active, assume_unique=True
                )
                if active_live.size:
                    active_device = torch.as_tensor(
                        active_live, dtype=torch.int64, device="cuda"
                    )
                    value[active_device] = source[active_device]
                selected = (
                    (frame.stream_continuous_columns >= start)
                    & (frame.stream_continuous_columns < stop)
                )
                if np.any(selected):
                    value[
                        torch.as_tensor(
                            frame.stream_rows[selected], dtype=torch.int64, device="cuda"
                        ),
                        torch.as_tensor(
                            frame.stream_continuous_columns[selected] - start,
                            dtype=torch.int64,
                            device="cuda",
                        ),
                    ] = torch.as_tensor(
                        frame.stream_half_widths[selected],
                        dtype=torch.float64,
                        device="cuda",
                    )
                values[layer_id] = value
            for predecessor in predecessors:
                remaining_uses[predecessor] -= 1
                if remaining_uses[predecessor] == 0:
                    del values[predecessor]
        if output_dense is not None:
            output_dense[:, start:stop] = (
                values[int(assert_layer.id)].detach().cpu().numpy()
            )
        if not bool(finite.item()):
            raise ExactReLULiveRowStreamError("GPU coefficient propagation overflowed")
        del values
    torch.cuda.synchronize()
    return preactivation_dense, output_dense, time.monotonic() - started


def _cube_bounds_from_stream(
    generators: sp.csr_matrix,
    center: np.ndarray,
    error: np.ndarray,
    *,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mass = _oh._nonnegative_sum_upper(
        _oh._row_l1_upper(generators, name=f"{name}.G_l1"),
        error,
        name=f"{name}.radius",
    )
    row_nnz = np.diff(generators.indptr).astype(np.float64)
    eps = np.finfo(np.float64).eps
    k_eps = (row_nnz + 2.0) * eps
    if np.any(k_eps >= 1.0):
        raise ExactReLULiveRowStreamError(f"{name} row is too long")
    guard = (k_eps / (1.0 - k_eps)) * (np.abs(center) + mass)
    guard += np.finfo(np.float64).tiny
    variable = (row_nnz > 0) | (error > 0.0)
    lower = center.copy()
    upper = center.copy()
    lower[variable] = np.nextafter(
        center[variable] - mass[variable] - guard[variable], -np.inf
    )
    upper[variable] = np.nextafter(
        center[variable] + mass[variable] + guard[variable], np.inf
    )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ExactReLULiveRowStreamError(f"{name} bounds are non-finite")
    return lower, upper


def _refine_phase_frames(
    order: Sequence[Any],
    frames: Mapping[int, _PhaseFrame],
    preliminary: Mapping[int, np.ndarray],
    preactivation_shadows: Mapping[int, _Shadow],
    *,
    continuous_count: int,
) -> Tuple[Dict[int, _PhaseFrame], int, int, int]:
    continuous = int(continuous_count)
    binary = 0
    stabilized = 0
    refined: Dict[int, _PhaseFrame] = {}
    for layer in order:
        layer_id = int(layer.id)
        frame = frames.get(layer_id)
        if frame is None:
            continue
        exact_rows = frame.stream_rows
        lower = frame.lower.copy()
        upper = frame.upper.copy()
        if exact_rows.size:
            matrix = _canonical(
                sp.csr_matrix(preliminary[layer_id]),
                name=f"live_row.refine[{layer_id}]",
            )
            shadow = preactivation_shadows[layer_id]
            cube_lower, cube_upper = _cube_bounds_from_stream(
                matrix,
                shadow.center[exact_rows],
                shadow.error[exact_rows],
                name=f"live_row.refine[{layer_id}]",
            )
            lower[exact_rows] = np.maximum(lower[exact_rows], cube_lower)
            upper[exact_rows] = np.minimum(upper[exact_rows], cube_upper)
            if np.any(lower[exact_rows] > upper[exact_rows]):
                raise ExactReLULiveRowStreamError(
                    f"refined ReLU {layer_id} bounds are contradictory"
                )
            tightened_lower = lower[exact_rows]
            tightened_upper = upper[exact_rows]
            active_mask = (tightened_lower >= 0.0) & (tightened_upper > 0.0)
            inactive_mask = tightened_upper <= 0.0
            exact_mask = (tightened_lower < 0.0) & (tightened_upper > 0.0)
            became_active = exact_rows[active_mask]
            became_inactive = exact_rows[inactive_mask]
            remains_exact = exact_rows[exact_mask]
            if became_active.size + became_inactive.size + remains_exact.size != exact_rows.size:
                raise ExactReLULiveRowStreamError(
                    f"refined ReLU {layer_id} phase partition is incomplete"
                )
        else:
            active_mask = np.zeros(0, dtype=bool)
            inactive_mask = np.zeros(0, dtype=bool)
            exact_mask = np.zeros(0, dtype=bool)
            became_active = np.zeros(0, dtype=np.int64)
            became_inactive = np.zeros(0, dtype=np.int64)
            remains_exact = exact_rows
        active = np.sort(np.concatenate((frame.active, became_active))).astype(np.int64)
        inactive = np.sort(np.concatenate((frame.inactive, became_inactive))).astype(np.int64)
        stabilized += int(became_active.size + became_inactive.size)
        cont_columns = frame.stream_continuous_columns[exact_mask]
        bin_columns = np.arange(binary, binary + remains_exact.size, dtype=np.int64)
        binary += int(remains_exact.size)
        deferred_mask = active_mask | inactive_mask
        linked_rows = np.sort(
            np.concatenate((remains_exact, became_active))
        ).astype(np.int64)
        linked_columns = np.arange(
            continuous, continuous + linked_rows.size, dtype=np.int64
        )
        continuous += int(linked_rows.size)
        refined[layer_id] = _PhaseFrame(
            lower,
            upper,
            active,
            inactive,
            remains_exact,
            cont_columns,
            bin_columns,
            frame.stream_half_widths,
            frame.stream_rows,
            frame.stream_continuous_columns,
            exact_rows[deferred_mask],
            active_mask[deferred_mask],
            frame.stream_continuous_columns[deferred_mask],
            linked_rows,
            linked_columns,
        )
    return refined, continuous, binary, stabilized


def build_forward_exact_relu_live_row_stream_candidate(
    net: Any,
    before: Mapping[int, Any],
    after: Mapping[int, Any],
    *,
    deadline: Optional[float] = None,
) -> ExactReLULiveRowStreamResult:
    """Build the disconnected live-row candidate; never falls back."""

    started = time.monotonic()
    if deadline is not None and (
        type(deadline) not in {int, float} or not math.isfinite(float(deadline))
    ):
        raise ExactReLULiveRowStreamError("deadline is malformed")
    deadline = None if deadline is None else float(deadline)
    if not torch.cuda.is_available():
        raise ExactReLULiveRowStreamError("CUDA is required; no fallback exists")
    order, by_id = _topological(net)
    inputs = [layer for layer in order if _oh._kind(layer.kind) == "INPUT"]
    asserts = [layer for layer in order if _oh._kind(layer.kind) == "ASSERT"]
    if len(inputs) != 1 or len(asserts) != 1:
        raise ExactReLULiveRowStreamError("candidate requires one INPUT and one ASSERT")
    input_layer = inputs[0]
    assert_layer = asserts[0]
    output_layer_id = _preds(net, assert_layer, 1)[0]
    input_lower, input_upper = _facts_box(
        after, int(input_layer.id), len(input_layer.out_vars), name="INPUT.after"
    )
    input_center, input_radius = _oh._enclosing_center_radius(
        input_lower, input_upper, name="live_row.input"
    )
    input_rows = np.flatnonzero(input_radius > 0.0).astype(np.int64)
    frames, n_cont, n_bin = _make_phase_frames(
        order, before, first_continuous_column=int(input_rows.size)
    )
    interval_exact_rows = int(n_bin)
    if n_cont > _MAX_FACTORS or n_bin > _MAX_EXACT_ROWS:
        raise ExactReLULiveRowStreamError("factor cap exceeded")

    _deadline(deadline, "matrices")
    matrix_started = time.monotonic()
    affines: Dict[int, _AffineSnapshot] = {}
    for layer in order:
        if _oh._kind(layer.kind) in {"CONV2D", "DENSE"}:
            predecessor = _preds(net, layer, 1)[0]
            affines[int(layer.id)] = _affine_snapshot(
                layer, input_size=len(by_id[predecessor].out_vars)
            )
    full_affine_nnz = int(
        sum(np.sum(_affine_fanin(snapshot), dtype=np.float64) for snapshot in affines.values())
    )
    live_rows, possible_rows = _live_rows(
        net, order, affines, frames, input_rows, output_layer_id
    )
    row_matrices: Dict[int, sp.csr_matrix] = {}
    for layer_id, snapshot in affines.items():
        predecessor = _preds(net, by_id[layer_id], 1)[0]
        row_matrices[layer_id] = _selected_affine_matrix(
            snapshot,
            live_rows[layer_id],
            possible_rows[predecessor],
            name=f"live_row.stream[{layer_id}]",
        )
    streamed_affine_nnz = int(sum(matrix.nnz for matrix in row_matrices.values()))
    if streamed_affine_nnz > _MAX_STREAMED_AFFINE_NNZ:
        raise ExactReLULiveRowStreamError("streamed affine nnz cap exceeded")
    matrix_seconds = time.monotonic() - matrix_started

    _deadline(deadline, "shadow")
    entry_cuda = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    shadows, preactivation_shadows, shadow_seconds = _build_shadows(
        net,
        order,
        affines,
        frames,
        input_center=input_center,
        input_radius=input_radius,
    )

    _deadline(deadline, "schedule")
    schedule_started = time.monotonic()
    device_rows = {layer_id: _device_csr(matrix) for layer_id, matrix in row_matrices.items()}
    device_row_ids = {
        layer_id: torch.as_tensor(rows, dtype=torch.int64, device="cuda")
        for layer_id, rows in live_rows.items()
    }
    torch.cuda.synchronize()
    schedule_seconds = time.monotonic() - schedule_started

    initial_n_cont = int(n_cont)
    initial_buffer_bytes = int(
        (interval_exact_rows + len(assert_layer.out_vars)) * initial_n_cont * 8
    )
    if initial_buffer_bytes > _MAX_OUTPUT_BYTES:
        raise ExactReLULiveRowStreamError("phase-refinement buffer cap exceeded")
    _deadline(deadline, "stream")
    preliminary, output_dense_optional, stream_seconds = _stream_generators(
        net,
        order,
        frames,
        live_rows,
        device_rows,
        device_row_ids,
        input_rows=input_rows,
        input_radius=input_radius,
        n_cont=initial_n_cont,
        assert_layer=assert_layer,
        deadline=deadline,
        stage_prefix="stream",
        collect_output=True,
    )
    if output_dense_optional is None:
        raise RuntimeError("the generator stream did not return an output")
    output_dense = output_dense_optional

    _deadline(deadline, "phase_refinement")
    refinement_started = time.monotonic()
    frames, n_cont, n_bin, refined_stable_rows = _refine_phase_frames(
        order,
        frames,
        preliminary,
        preactivation_shadows,
        continuous_count=initial_n_cont,
    )
    refinement_seconds = time.monotonic() - refinement_started
    if n_cont > _MAX_FACTORS or n_bin > _MAX_EXACT_ROWS:
        raise ExactReLULiveRowStreamError("refined factor cap exceeded")

    output_shadow = shadows[output_layer_id]

    full_input_ids = _allocate_ids(int(input_center.size))
    continuous_ids = [int(full_input_ids[row]) for row in input_rows]
    binary_ids = []
    for layer in order:
        frame = frames.get(int(layer.id))
        if frame is None or not frame.stream_rows.size:
            continue
        continuous_ids.extend(_allocate_ids(int(frame.stream_rows.size)).tolist())
    for layer in order:
        frame = frames.get(int(layer.id))
        if frame is None:
            continue
        continuous_ids.extend(
            _allocate_ids(int(frame.linked_rows.size)).tolist()
        )
        if frame.exact.size:
            binary_ids.extend(_allocate_ids(int(frame.exact.size)).tolist())
    if len(continuous_ids) != n_cont or len(binary_ids) != n_bin:
        raise ExactReLULiveRowStreamError("refined factor allocation drifted")

    exact_rows = int(sum(frame.exact.size for frame in frames.values()))
    incremental_peak = max(0, int(torch.cuda.max_memory_allocated()) - entry_cuda)
    del device_rows

    _deadline(deadline, "assembly")
    assembly_started = time.monotonic()
    preactivation_generators = {
        layer_id: _canonical(sp.csr_matrix(value), name=f"live_row.preactivation[{layer_id}]")
        for layer_id, value in preliminary.items()
    }
    output_G = _canonical(sp.csr_matrix(output_dense), name="live_row.output")
    output_error_rows = np.flatnonzero(output_shadow.error > 0.0).astype(np.int64)
    output_error_columns = _allocate_ids(int(output_error_rows.size))
    continuous_ids.extend(output_error_columns.tolist())
    final_n_cont = len(continuous_ids)
    output_G = _pad_columns(output_G, final_n_cont)
    if output_error_rows.size:
        error_matrix = sp.csr_matrix(
            (
                output_shadow.error[output_error_rows],
                (output_error_rows, np.arange(n_cont, final_n_cont, dtype=np.int64)),
            ),
            shape=(output_shadow.center.size, final_n_cont),
            dtype=np.float64,
        )
        output_G = _canonical(output_G + error_matrix, name="live_row.output_with_error")
    Ac, Ab, equal, Auc, Aub, upper = _build_constraints(
        order, frames, preactivation_generators, preactivation_shadows,
        n_cont=final_n_cont, n_bin=n_bin,
    )
    hz = _oh._assemble_owned_operator_sparse_hz(
        c=np.ascontiguousarray(output_shadow.center, dtype=np.float64),
        Gc=output_G,
        Gb=sp.csr_matrix((output_shadow.center.size, n_bin), dtype=np.float64),
        Ac=_pad_columns(Ac, final_n_cont),
        Ab=Ab,
        b=equal,
        Auc=_pad_columns(Auc, final_n_cont),
        Aub=Aub,
        ub=upper,
        col_ids=np.ascontiguousarray(continuous_ids, dtype=np.int64),
        bcol_ids=np.ascontiguousarray(binary_ids, dtype=np.int64),
    )
    hz.full_col_ids = full_input_ids.copy()
    hz.operator_input_center = input_center.copy()
    hz.operator_input_radius = input_radius.copy()
    assembly_seconds = time.monotonic() - assembly_started
    total_seconds = time.monotonic() - started
    receipt = ExactReLULiveRowStreamReceipt(
        schema=_SCHEMA,
        status="candidate_complete",
        factor_batch=_FACTOR_BATCH,
        continuous_factors=int(final_n_cont),
        binary_factors=int(n_bin),
        exact_rows=exact_rows,
        source_rows=int(hz.n_eq + hz.n_ub),
        source_nnz=int(hz.constraint_nnz),
        output_nnz=int(hz.Gc.nnz),
        full_affine_nnz=full_affine_nnz,
        streamed_affine_nnz=streamed_affine_nnz,
        matrix_build_seconds=float(matrix_seconds),
        shadow_seconds=float(shadow_seconds),
        gpu_schedule_seconds=float(schedule_seconds),
        phase_refinement_seconds=float(refinement_seconds),
        gpu_stream_seconds=float(stream_seconds),
        assembly_seconds=float(assembly_seconds),
        total_seconds=float(total_seconds),
        incremental_cuda_peak_bytes=int(incremental_peak),
        phase_refinement_passes=1,
        interval_exact_rows=int(interval_exact_rows),
        refined_stable_rows=int(refined_stable_rows),
        constraint_local_preactivation_factors=int(
            sum(frame.linked_rows.size for frame in frames.values())
        ),
    )
    return ExactReLULiveRowStreamResult(
        hz=hz,
        input_col_ids=full_input_ids.copy(),
        input_layer_id=int(input_layer.id),
        output_layer_id=int(output_layer_id),
        assert_layer_id=int(assert_layer.id),
        receipt=receipt,
    )


__all__ = [
    "ExactReLULiveRowStreamError",
    "ExactReLULiveRowStreamReceipt",
    "ExactReLULiveRowStreamResult",
    "build_forward_exact_relu_live_row_stream_candidate",
]
