#!/usr/bin/env python3
"""Bounded single-stream float64-only phase-projection probe.

This probe changes only the non-authoritative candidate arithmetic.  It keeps
the verifier-owned raw-BOX, zero-width forward interval, and exact Fraction
property check unchanged.  It does not execute an ONNX input point, sample an
input, run PGD, BaB, backward bounds, or dual tightening.

The projected cell is formed from one generator stream by eliminating a
topologically triangular batch of phase-change auxiliaries.  It is disconnected
from the production verifier.
"""

from __future__ import annotations

from fractions import Fraction
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch
import triton
import triton.language as tl

from act.back_end.analyze import analyze
from act.back_end.core import ConSet, Fact
from act.back_end.hybridz_tf import forward_exact_relu_live_row_stream_candidate as _live
from act.back_end.hybridz_tf import forward_exact_relu_phase_projection_candidate as _projection
from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
from act.back_end.verifier import (
    _ensure_assert_linear_encoding,
    _get_output_layer_id,
    add_all_input_specs,
    find_entry_layer_id,
    gather_input_spec_layers,
    get_assert_layer,
    get_input_ids,
    seed_from_input_specs,
)
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
from act.pipeline.verification.torch2act import TorchToACT
from act.util.device_manager import initialize_device


ONNX = os.environ.get(
    "ACT_PHASE_PROJECTION_ONNX",
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/"
    "CIFAR100_resnet_medium.onnx",
)
VNNLIB = os.environ.get(
    "ACT_PHASE_PROJECTION_VNNLIB",
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/"
    "CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib",
)
CATEGORY = os.environ.get("ACT_PHASE_PROJECTION_CATEGORY", "cifar100_2024")


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


def _gpu_selected_affine_matrix(snapshot, selected_rows, source_possible, *, name):
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
        raise _projection.ExactReLUPhaseProjectionUnknown(
            f"{name} is outside the GPU all-nonzero CSR domain"
        )
    selected = torch.as_tensor(selected_rows, dtype=torch.int64, device="cuda")
    possible = torch.as_tensor(source_possible, dtype=torch.uint8, device="cuda")
    rows = int(selected.numel())
    if snapshot.kind == "DENSE":
        live_columns = torch.nonzero(possible, as_tuple=False).reshape(-1).to(torch.int64)
        row_width = int(live_columns.numel())
        total = rows * row_width
        if total > np.iinfo(np.int32).max:
            raise _projection.ExactReLUPhaseProjectionUnknown(
                f"{name} exceeds int32 CSR capacity"
            )
        weight = torch.as_tensor(snapshot.weight, dtype=torch.float64, device="cuda")
        data = weight.index_select(0, selected).index_select(1, live_columns).reshape(-1)
        indices = live_columns.repeat(rows)
        indptr = (
            torch.arange(
                0, total + 1, row_width, dtype=torch.int64, device="cuda"
            )
            if row_width
            else torch.zeros(rows + 1, dtype=torch.int64, device="cuda")
        )
        return _live._DeviceCSR(indptr, indices, data, rows, snapshot.input_size)

    topology = snapshot.topology
    if topology is None:
        raise _projection.ExactReLUPhaseProjectionUnknown(
            f"{name} CONV lost topology"
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
        raise _projection.ExactReLUPhaseProjectionUnknown(
            f"{name} exceeds int32 CSR capacity"
        )
    indices = torch.empty(total, dtype=torch.int64, device="cuda")
    data = torch.empty(total, dtype=torch.float64, device="cuda")
    weight = torch.as_tensor(snapshot.weight, dtype=torch.float64, device="cuda")
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
    return _live._DeviceCSR(indptr, indices, data, rows, snapshot.input_size)


def _float_affine_shadow(source, snapshot, *, layer_id):
    del layer_id
    weight = torch.as_tensor(snapshot.weight, dtype=torch.float64, device="cuda")
    if snapshot.kind == "DENSE":
        value = torch.matmul(
            weight,
            torch.as_tensor(source.center, dtype=torch.float64, device="cuda"),
        )
    else:
        topology = snapshot.topology
        if topology is None:
            raise RuntimeError("CONV snapshot lost topology")
        batch, channels, height, width = topology.input_shape
        value = torch.nn.functional.conv2d(
            torch.as_tensor(
                source.center.reshape(batch, channels, height, width),
                dtype=torch.float64,
                device="cuda",
            ),
            weight,
            bias=None,
            stride=topology.stride,
            padding=topology.padding,
            dilation=topology.dilation,
            groups=topology.groups,
        )
    center = value.detach().cpu().numpy().reshape(-1) + snapshot.bias
    if not np.all(np.isfinite(center)):
        raise RuntimeError("float candidate center overflowed")
    zero = np.zeros(center.size, dtype=np.float64)
    return _live._Shadow(center, zero, np.abs(center))


def _float_add_shadow(left, right, *, layer_id):
    del layer_id
    center = np.asarray(left.center) + np.asarray(right.center)
    if not np.all(np.isfinite(center)):
        raise RuntimeError("float candidate ADD overflowed")
    zero = np.zeros(center.size, dtype=np.float64)
    return _live._Shadow(center, zero, np.abs(center))


def _float_relu_shadow(source, frame, *, layer_id):
    del layer_id
    center = np.zeros(source.center.size, dtype=np.float64)
    center[frame.active] = source.center[frame.active]
    if frame.exact.size:
        center[frame.exact] = 0.5 * frame.upper[frame.exact]
    zero = np.zeros(center.size, dtype=np.float64)
    return _live._Shadow(center, zero, np.abs(center))


def _fixed_frame(original, selected):
    empty = np.zeros(0, dtype=np.int64)
    active = np.sort(
        np.concatenate((original.active, original.exact[selected]))
    ).astype(np.int64)
    inactive = np.sort(
        np.concatenate((original.inactive, original.exact[~selected]))
    ).astype(np.int64)
    return _live._PhaseFrame(
        original.lower,
        original.upper,
        active,
        inactive,
        empty,
        empty,
        empty,
        np.zeros(0, dtype=np.float64),
        original.exact.copy(),
        empty,
        empty,
        np.zeros(0, dtype=bool),
        empty,
        empty,
        empty,
    )


def _triangular_input_expansion(
    changes, positions, first_pre, delta_pre, *, input_width
):
    """Eliminate ordered phase-change auxiliaries into input coordinates."""

    expansion = np.zeros((len(changes), input_width), dtype=np.float64)
    for index, (layer_id, row, base_active, target_active) in enumerate(changes):
        position = positions[layer_id][row]
        base_q = np.asarray(first_pre[layer_id][position], dtype=np.float64)
        if (not base_active) and target_active:
            expansion[index] = base_q
            if index:
                expansion[index] += np.asarray(
                    delta_pre[layer_id][position, :index] @ expansion[:index],
                    dtype=np.float64,
                )
        elif base_active and (not target_active):
            expansion[index] = -base_q
        else:
            raise _projection.ExactReLUPhaseProjectionUnknown("invalid phase change")
    return expansion


def _csr_box_upper(matrix, lower, upper):
    """Evaluate each CSR row's maximum over one axis-aligned box."""

    matrix = matrix.tocsr(copy=False)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if matrix.shape[1] != lower.size or lower.shape != upper.shape:
        raise ValueError("CSR box dimensions do not match")
    contribution = matrix.data * np.where(
        matrix.data >= 0.0, upper[matrix.indices], lower[matrix.indices]
    )
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    nonempty = np.diff(matrix.indptr) > 0
    if np.any(nonempty):
        result[nonempty] = np.add.reduceat(
            contribution, matrix.indptr[:-1][nonempty]
        )
    return result


def _all_nonzero_affine_support_forward(snapshot, source_mask):
    source_mask = np.asarray(source_mask, dtype=bool).reshape(-1)
    if source_mask.size != snapshot.input_size:
        raise ValueError("affine support input width mismatch")
    if snapshot.kind == "DENSE":
        return np.full(snapshot.output_size, bool(np.any(source_mask)), dtype=bool)
    topology = snapshot.topology
    if topology is None:
        raise ValueError("CONV snapshot lost topology")
    batch, in_channels, input_height, input_width = topology.input_shape
    _out_batch, out_channels, _output_height, _output_width = topology.output_shape
    kernel_height, kernel_width = snapshot.weight.shape[2:]
    source = torch.as_tensor(
        source_mask.reshape(batch, in_channels, input_height, input_width),
        dtype=torch.float64,
        device="cuda",
    )
    kernel = torch.ones(
        (out_channels, snapshot.weight.shape[1], kernel_height, kernel_width),
        dtype=torch.float64,
        device="cuda",
    )
    result = torch.nn.functional.conv2d(
        source,
        kernel,
        stride=topology.stride,
        padding=topology.padding,
        dilation=topology.dilation,
        groups=topology.groups,
    )
    return (result > 0.0).detach().cpu().numpy().reshape(-1)


def _all_nonzero_affine_support_backward(snapshot, output_mask):
    output_mask = np.asarray(output_mask, dtype=bool).reshape(-1)
    if output_mask.size != snapshot.output_size:
        raise ValueError("affine demand output width mismatch")
    if snapshot.kind == "DENSE":
        return np.full(snapshot.input_size, bool(np.any(output_mask)), dtype=bool)
    topology = snapshot.topology
    if topology is None:
        raise ValueError("CONV snapshot lost topology")
    batch, in_channels, input_height, input_width = topology.input_shape
    _out_batch, out_channels, output_height, output_width = topology.output_shape
    kernel_height, kernel_width = snapshot.weight.shape[2:]
    base_height = (
        (output_height - 1) * topology.stride[0]
        - 2 * topology.padding[0]
        + topology.dilation[0] * (kernel_height - 1)
        + 1
    )
    base_width = (
        (output_width - 1) * topology.stride[1]
        - 2 * topology.padding[1]
        + topology.dilation[1] * (kernel_width - 1)
        + 1
    )
    output_padding = (input_height - base_height, input_width - base_width)
    if not (
        0 <= output_padding[0] < topology.stride[0]
        and 0 <= output_padding[1] < topology.stride[1]
    ):
        raise ValueError("CONV transpose output padding is malformed")
    source = torch.as_tensor(
        output_mask.reshape(batch, out_channels, output_height, output_width),
        dtype=torch.float64,
        device="cuda",
    )
    kernel = torch.ones(
        (out_channels, snapshot.weight.shape[1], kernel_height, kernel_width),
        dtype=torch.float64,
        device="cuda",
    )
    result = torch.nn.functional.conv_transpose2d(
        source,
        kernel,
        stride=topology.stride,
        padding=topology.padding,
        output_padding=output_padding,
        dilation=topology.dilation,
        groups=topology.groups,
    )
    return (result > 0.0).detach().cpu().numpy().reshape(-1)


def _all_nonzero_live_rows(
    net, order, affines, frames, input_variable_rows, output_layer_id
):
    possible = {
        int(layer.id): np.zeros(len(layer.out_vars), dtype=bool) for layer in order
    }
    input_layer = next(layer for layer in order if _live._oh._kind(layer.kind) == "INPUT")
    possible[int(input_layer.id)][input_variable_rows] = True
    for layer in order:
        layer_id = int(layer.id)
        kind = _live._oh._kind(layer.kind)
        if kind == "INPUT":
            continue
        predecessors = tuple(int(value) for value in net.preds.get(layer_id, []))
        if kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
            possible[layer_id] = possible[predecessors[0]].copy()
        elif kind in {"CONV2D", "DENSE"}:
            possible[layer_id] = _all_nonzero_affine_support_forward(
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
        kind = _live._oh._kind(layer.kind)
        predecessors = tuple(int(value) for value in net.preds.get(layer_id, []))
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
            demand[predecessors[0]] |= _all_nonzero_affine_support_backward(
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
    return live, possible


def _single_stream_float64_candidate(net, entry, before, after):
    """Return one candidate and timing receipt, or fail closed.

    The projected cell generators are obtained by eliminating a topologically
    triangular batch of phase-change auxiliaries from the first-cell stream.
    No second generator stream and no candidate outward/error envelope is
    constructed.
    """

    started = time.monotonic()
    order, by_id = _live._topological(net)
    input_layer = next(layer for layer in order if _live._oh._kind(layer.kind) == "INPUT")
    assert_layer = next(layer for layer in order if _live._oh._kind(layer.kind) == "ASSERT")
    output_layer_id = _live._preds(net, assert_layer, 1)[0]
    if int(input_layer.id) != int(entry):
        raise _projection.ExactReLUPhaseProjectionUnknown("entry mismatch")

    input_width = len(input_layer.out_vars)
    represented_lower, represented_upper = _live._facts_box(
        after, int(input_layer.id), input_width, name="float_projection.input"
    )
    raw_lower, raw_upper = _projection._raw_box_intersection(order, input_width)
    input_center, input_radius = _live._oh._enclosing_center_radius(
        represented_lower, represented_upper, name="float_projection.input"
    )
    input_rows = np.flatnonzero(input_radius > 0.0).astype(np.int64)
    factor_bounds = _projection._inward_factor_bounds(
        raw_lower,
        raw_upper,
        input_center,
        input_radius,
        input_rows,
        _projection._SOLVER_TOLERANCE,
    )
    factor_lower = np.asarray([bound[0] for bound in factor_bounds])
    factor_upper = np.asarray([bound[1] for bound in factor_bounds])
    original_frames, _n_cont, n_bin = _live._make_phase_frames(
        order, before, first_continuous_column=int(input_rows.size)
    )
    if not input_rows.size or not n_bin:
        raise _projection.ExactReLUPhaseProjectionUnknown(
            "float projection requires input factors and unstable ReLUs"
        )

    setup_started = time.monotonic()
    affines = {}
    for layer in order:
        if _live._oh._kind(layer.kind) in {"CONV2D", "DENSE"}:
            predecessor = _live._preds(net, layer, 1)[0]
            affines[int(layer.id)] = _live._affine_snapshot(
                layer, input_size=len(by_id[predecessor].out_vars)
            )
            if np.any(affines[int(layer.id)].weight == 0.0):
                raise _projection.ExactReLUPhaseProjectionUnknown(
                    "float single-stream path requires all affine weights nonzero"
                )
    live_rows, possible_rows = _all_nonzero_live_rows(
        net, order, affines, original_frames, input_rows, output_layer_id
    )
    device_matrices = {}
    for layer_id, snapshot in affines.items():
        predecessor = _live._preds(net, by_id[layer_id], 1)[0]
        device_matrices[layer_id] = _gpu_selected_affine_matrix(
            snapshot,
            live_rows[layer_id],
            possible_rows[predecessor],
            name=f"float_projection.stream[{layer_id}]",
        )
    device_rows = {
        key: torch.as_tensor(value, dtype=torch.int64, device="cuda")
        for key, value in live_rows.items()
    }
    setup_seconds = time.monotonic() - setup_started

    def centers(assignments):
        shadows = {}
        pre_centers = {}
        selected_map = {}
        frames = {}
        for layer in order:
            layer_id = int(layer.id)
            kind = _live._oh._kind(layer.kind)
            predecessors = tuple(int(v) for v in net.preds.get(layer_id, []))
            if kind == "INPUT":
                zero = np.zeros(input_center.size, dtype=np.float64)
                shadows[layer_id] = _live._Shadow(input_center.copy(), zero, np.abs(input_center))
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                shadows[layer_id] = shadows[predecessors[0]]
            elif kind in {"CONV2D", "DENSE"}:
                shadows[layer_id] = _float_affine_shadow(
                    shadows[predecessors[0]], affines[layer_id], layer_id=layer_id
                )
            elif kind == "ADD":
                shadows[layer_id] = _float_add_shadow(
                    shadows[predecessors[0]], shadows[predecessors[1]], layer_id=layer_id
                )
            elif kind == "RELU":
                source = shadows[predecessors[0]]
                original = original_frames[layer_id]
                pre_centers[layer_id] = source.center[original.exact].copy()
                default = source.center[original.exact] >= 0.0
                selected = default if assignments is None else np.asarray(
                    assignments.get(layer_id, default), dtype=bool
                )
                if selected.shape != (original.exact.size,):
                    raise _projection.ExactReLUPhaseProjectionUnknown(
                        "projected phase shape mismatch"
                    )
                selected_map[layer_id] = selected.copy()
                frame = _fixed_frame(original, selected)
                frames[layer_id] = frame
                shadows[layer_id] = _float_relu_shadow(source, frame, layer_id=layer_id)
            else:
                raise _projection.ExactReLUPhaseProjectionUnknown(
                    f"unsupported float graph kind {kind}"
                )
        return selected_map, pre_centers, shadows[output_layer_id].center, frames

    first_center_started = time.monotonic()
    first_assign, first_pre_center, first_output_center, first_frames = centers(None)
    first_center_seconds = time.monotonic() - first_center_started

    first_pre, first_output, first_stream_seconds = _live._stream_generators(
        net,
        order,
        first_frames,
        live_rows,
        device_matrices,
        device_rows,
        input_rows=input_rows,
        input_radius=input_radius,
        n_cont=int(input_rows.size),
        assert_layer=assert_layer,
        deadline=None,
        stage_prefix="float_projection_first",
        collect_output=True,
    )
    if first_output is None:
        raise _projection.ExactReLUPhaseProjectionUnknown("missing first stream output")

    output_width = len(by_id[output_layer_id].out_vars)
    C, thresholds = _projection._top1_property(assert_layer, output_width)
    first_objective = np.asarray(C @ first_output, dtype=np.float64)
    first_objective_center = np.asarray(
        C @ first_output_center - thresholds, dtype=np.float64
    )
    first_upper = first_objective_center + np.sum(
        np.abs(first_objective), axis=1, dtype=np.float64
    )
    rival = int(np.argmax(first_upper))
    first_coeff = first_objective[rival]
    first_factors = np.where(first_coeff >= 0.0, factor_upper, factor_lower)

    projected = {}
    changes = []
    positions = {}
    for layer in order:
        layer_id = int(layer.id)
        original = original_frames.get(layer_id)
        if original is None or not original.exact.size:
            continue
        value = first_pre_center[layer_id] + np.asarray(
            first_pre[layer_id] @ first_factors, dtype=np.float64
        )
        if np.any(value == 0.0) or not np.all(np.isfinite(value)):
            raise _projection.ExactReLUPhaseProjectionUnknown(
                "float projection hit a zero or nonfinite phase"
            )
        selected = value >= 0.0
        projected[layer_id] = selected
        rows = np.asarray(original.stream_rows, dtype=np.int64)
        positions[layer_id] = {int(row): pos for pos, row in enumerate(rows)}
        for pos in np.flatnonzero(selected != first_assign[layer_id]):
            changes.append(
                (layer_id, int(rows[pos]), bool(first_assign[layer_id][pos]), bool(selected[pos]))
            )

    target_center_started = time.monotonic()
    target_assign, target_pre_center, target_output_center, target_frames = centers(projected)
    target_center_seconds = time.monotonic() - target_center_started
    change_index = {
        (layer_id, row): index
        for index, (layer_id, row, _base, _target) in enumerate(changes)
    }
    width_total = len(changes)
    target_active_live = {}
    changed_by_layer = {}
    exact_device = {}
    for layer in order:
        layer_id = int(layer.id)
        if layer_id not in target_frames:
            continue
        target_active_live[layer_id] = torch.as_tensor(
            np.intersect1d(
                live_rows[layer_id], target_frames[layer_id].active, assume_unique=True
            ),
            dtype=torch.int64,
            device="cuda",
        )
        changed_by_layer[layer_id] = [
            (row, change_index[(layer_id, row)])
            for local_layer, row, _base, _target in changes
            if local_layer == layer_id
        ]
        exact_device[layer_id] = torch.as_tensor(
            original_frames[layer_id].stream_rows,
            dtype=torch.int64,
            device="cuda",
        )

    delta_started = time.monotonic()
    delta_pre_parts = {layer_id: [] for layer_id in original_frames}
    delta_output_parts = []
    for start in range(0, width_total, 64):
        stop = min(width_total, start + 64)
        width = stop - start
        values = {}
        for layer in order:
            layer_id = int(layer.id)
            kind = _live._oh._kind(layer.kind)
            predecessors = tuple(int(v) for v in net.preds.get(layer_id, []))
            if kind == "INPUT":
                values[layer_id] = torch.zeros(
                    (len(layer.out_vars), width), dtype=torch.float64, device="cuda"
                )
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                values[layer_id] = values[predecessors[0]]
            elif kind in {"CONV2D", "DENSE"}:
                selected_value = _live._ordered_csr_dense(
                    device_matrices[layer_id], values[predecessors[0]]
                )
                value = torch.zeros(
                    (len(layer.out_vars), width), dtype=torch.float64, device="cuda"
                )
                if device_rows[layer_id].numel():
                    value[device_rows[layer_id]] = selected_value
                values[layer_id] = value
            elif kind == "ADD":
                values[layer_id] = values[predecessors[0]] + values[predecessors[1]]
            elif kind == "RELU":
                source = values[predecessors[0]]
                delta_pre_parts[layer_id].append(
                    source[exact_device[layer_id]].detach().cpu().numpy()
                )
                value = torch.zeros_like(source)
                active = target_active_live[layer_id]
                if active.numel():
                    value[active] = source[active]
                for row, column in changed_by_layer[layer_id]:
                    value[row] = 0.0
                    if start <= column < stop:
                        value[row, column - start] = 1.0
                values[layer_id] = value
            else:
                raise _projection.ExactReLUPhaseProjectionUnknown(
                    f"unsupported delta graph kind {kind}"
                )
        delta_output_parts.append(
            values[int(assert_layer.id)].detach().cpu().numpy()
        )
    torch.cuda.synchronize()
    delta_pre = {
        layer_id: (
            np.concatenate(parts, axis=1)
            if parts
            else np.empty((original_frames[layer_id].exact.size, 0), dtype=np.float64)
        )
        for layer_id, parts in delta_pre_parts.items()
    }
    delta_output = (
        np.concatenate(delta_output_parts, axis=1)
        if delta_output_parts
        else np.empty((output_width, 0), dtype=np.float64)
    )
    delta_seconds = time.monotonic() - delta_started

    expansion_started = time.monotonic()
    U = _triangular_input_expansion(
        changes,
        positions,
        first_pre,
        delta_pre,
        input_width=int(input_rows.size),
    )

    target_pre = {
        layer_id: np.asarray(first_pre[layer_id], dtype=np.float64)
        + np.asarray(delta_pre[layer_id] @ U, dtype=np.float64)
        for layer_id in first_pre
    }
    target_output = np.asarray(first_output, dtype=np.float64) + np.asarray(
        delta_output @ U, dtype=np.float64
    )
    expansion_seconds = time.monotonic() - expansion_started

    model_started = time.monotonic()
    blocks = []
    rhs = []
    total_rows = 0
    for layer in order:
        layer_id = int(layer.id)
        original = original_frames.get(layer_id)
        if original is None or not original.exact.size:
            continue
        matrix = sp.csr_matrix(target_pre[layer_id])
        selected = target_assign[layer_id]
        blocks.append(
            matrix.multiply(np.where(selected, -1.0, 1.0)[:, None]).tocsr()
        )
        center = target_pre_center[layer_id]
        rhs.append(np.where(selected, center, -center))
        total_rows += int(original.exact.size)
    A = sp.vstack(blocks, format="csr")
    b = np.ascontiguousarray(np.concatenate(rhs), dtype=np.float64)
    row_max = _csr_box_upper(A, factor_lower, factor_upper)
    keep = row_max > b
    screened_A = A[keep].tocsr()
    screened_b = b[keep]
    objective_coeff = np.asarray(C[[rival]] @ target_output, dtype=np.float64).reshape(-1)
    objective_center = float(C[rival] @ target_output_center - thresholds[rival])
    model_seconds = time.monotonic() - model_started

    lp_started = time.monotonic()
    solved = linprog(
        -objective_coeff,
        A_ub=screened_A,
        b_ub=screened_b,
        bounds=factor_bounds,
        method="highs-ds",
        options={
            "presolve": False,
            "time_limit": 10.0,
            "primal_feasibility_tolerance": _projection._SOLVER_TOLERANCE,
        },
    )
    lp_seconds = time.monotonic() - lp_started
    if not solved.success or solved.x is None:
        raise _projection.ExactReLUPhaseProjectionUnknown(
            f"float single-stream LP failed: {solved.message}"
        )
    factors = np.asarray(solved.x, dtype=np.float64).reshape(-1)
    float_margin = float(objective_center + objective_coeff @ factors)
    if not np.isfinite(float_margin) or float_margin <= 0.0:
        raise _projection.ExactReLUPhaseProjectionUnknown(
            f"float screen rejected candidate margin={float_margin!r}"
        )

    decoded = np.asarray(raw_lower, dtype=np.float64).copy()
    for column, raw_row in enumerate(input_rows):
        exact = Fraction.from_float(float(input_center[int(raw_row)]))
        exact += Fraction.from_float(float(input_radius[int(raw_row)])) * Fraction.from_float(
            float(factors[column])
        )
        decoded[int(raw_row)] = float(exact)
    if not (
        np.all(np.isfinite(decoded))
        and np.all(decoded >= raw_lower)
        and np.all(decoded <= raw_upper)
    ):
        raise _projection.ExactReLUPhaseProjectionUnknown(
            "float candidate decoded outside raw BOX"
        )

    terminal_started = time.monotonic()
    input_shape = tuple(int(value) for value in after[int(input_layer.id)].bounds.lb.shape)
    point_lower, point_upper = _projection._singleton_interval_forward(
        net, order, affines, decoded.reshape(input_shape), output_layer_id
    )
    exact_margin = _projection._exact_singleton_margin_lower(
        C[rival], thresholds[rival], point_lower, point_upper
    )
    terminal_seconds = time.monotonic() - terminal_started
    if exact_margin <= 0:
        raise _projection.ExactReLUPhaseProjectionUnknown(
            "unchanged terminal proof rejected float candidate"
        )

    return {
        "status": "singleton_verified",
        "generator_representation": "gpu_emitted_selected_csr_v1",
        "selected_property_row": rival,
        "phase_changes": width_total,
        "phase_rows": total_rows,
        "screened_rows": int(screened_A.shape[0]),
        "raw_nnz": int(A.nnz),
        "screened_nnz": int(screened_A.nnz),
        "float_margin": float_margin,
        "singleton_margin_lower": float(exact_margin),
        "setup_seconds": setup_seconds,
        "first_center_seconds": first_center_seconds,
        "first_stream_seconds": first_stream_seconds,
        "target_center_seconds": target_center_seconds,
        "delta_seconds": delta_seconds,
        "expansion_seconds": expansion_seconds,
        "model_seconds": model_seconds,
        "lp_seconds": lp_seconds,
        "terminal_seconds": terminal_seconds,
        "total_seconds": time.monotonic() - started,
    }


def main() -> None:
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")

    request_started = time.monotonic()
    spec = create_specs_from_paths(ONNX, VNNLIB, category=CATEGORY)
    wrapped = next(iter(synthesize_models_from_specs([spec]).values())).to(
        device=torch.device("cuda"), dtype=torch.float64
    )
    net = TorchToACT(wrapped).run()
    entry = int(find_entry_layer_id(net))
    specs = gather_input_spec_layers(net)
    seed = seed_from_input_specs(specs)
    fact = Fact(bounds=seed, cons=ConSet())
    add_all_input_specs(fact.cons, get_input_ids(net), specs)
    before, after, _ = analyze(net, entry, fact)
    assert_layer = get_assert_layer(net)
    output_layer_id = _get_output_layer_id(net)
    output_layer = next(
        layer for layer in net.layers if int(layer.id) == int(output_layer_id)
    )
    _ensure_assert_linear_encoding(
        assert_layer,
        B=1,
        n_out=len(output_layer.out_vars),
        device=torch.device("cuda"),
        dtype=torch.float64,
    )
    analysis_seconds = time.monotonic() - request_started
    unsupported = tuple(
        (int(layer.id), str(layer.kind))
        for layer in net.layers
        if _live._oh._kind(layer.kind) not in _live._SUPPORTED
    )
    if unsupported:
        raise RuntimeError(f"unsupported synthesized layers: {unsupported}")

    repeats = int(os.environ.get("ACT_FLOAT_PROBE_REPEATS", "3"))
    if repeats < 1 or repeats > 7:
        raise RuntimeError("ACT_FLOAT_PROBE_REPEATS must be in [1, 7]")
    receipts = []
    candidate_started = time.monotonic()
    status = "FALSIFIED"
    reason = None
    for _repeat in range(repeats):
        try:
            receipts.append(
                _single_stream_float64_candidate(net, entry, before, after)
            )
        except _projection.ExactReLUPhaseProjectionUnknown as exc:
            status = "UNKNOWN"
            reason = str(exc)
            break
    receipt = receipts[-1] if receipts else None

    print(
        json.dumps(
            {
                "schema": "act.hybridz.phase_projection_float64_probe.v1",
                "status": status,
                "reason": reason,
                "analysis_seconds": analysis_seconds,
                "candidate_seconds": time.monotonic() - candidate_started,
                "receipt": receipt,
                "repeat_receipts": receipts,
                "scope": {
                    "candidate_authority": False,
                    "candidate_proof_authority": False,
                    "verdict_authority": False,
                    "production_integrated": False,
                    "float32_used": False,
                    "candidate_outward_error_bands_used": False,
                    "intermediate_phase_or_margin_replay_used": False,
                    "generator_streams": 1,
                    "generator_representation": "gpu_emitted_selected_csr_v1",
                    "projected_generators": "triangular_batch_preelimination",
                    "input_sampling_used": False,
                    "onnx_input_execution_used": False,
                    "pgd_used": False,
                    "bab_used": False,
                    "backward_used": False,
                    "dual_tightening_used": False,
                    "terminal_verifier_proof_unchanged": True,
                    "terminal_verifier_proof_executed": status == "FALSIFIED",
                    "terminal_verifier_proof_authority": status == "FALSIFIED",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
