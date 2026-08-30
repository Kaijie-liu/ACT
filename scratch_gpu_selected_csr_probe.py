#!/usr/bin/env python3
"""Disconnected GPU selected-CSR construction probe.

The experiment changes representation only.  It does not sample inputs,
execute an ONNX input point, run PGD, split/BaB, backward bounds, dual
tightening, or create a verdict.  Exact CSR bytes are compared against the
existing CPU builder before the candidate timing is considered.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import triton
import triton.language as tl

import scratch_phase_projection_float64_probe as _probe
from act.back_end.analyze import analyze
from act.back_end.core import ConSet, Fact
from act.back_end.hybridz_tf import forward_exact_relu_live_row_stream_candidate as _live
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


@triton.jit
def _conv_row_count_kernel(
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
def _conv_row_emit_kernel(
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


def _gpu_selected(snapshot, selected_rows, source_possible, *, name):
    del name
    selected = torch.as_tensor(selected_rows, dtype=torch.int64, device="cuda")
    possible = torch.as_tensor(source_possible, dtype=torch.uint8, device="cuda")
    rows = int(selected.numel())
    if snapshot.kind == "DENSE":
        live_columns = torch.nonzero(possible, as_tuple=False).reshape(-1).to(torch.int64)
        row_width = int(live_columns.numel())
        count = rows * row_width
        weight = torch.as_tensor(snapshot.weight, dtype=torch.float64, device="cuda")
        data = weight.index_select(0, selected).index_select(1, live_columns).reshape(-1)
        indices = live_columns.repeat(rows)
        indptr = torch.arange(
            0, count + 1, row_width, dtype=torch.int64, device="cuda"
        ) if row_width else torch.zeros(rows + 1, dtype=torch.int64, device="cuda")
        return _live._DeviceCSR(indptr, indices, data, rows, snapshot.input_size)

    topology = snapshot.topology
    if topology is None:
        raise RuntimeError("CONV lost topology")
    _batch, in_channels, input_height, input_width = topology.input_shape
    _out_batch, out_channels, output_height, output_width = topology.output_shape
    in_per_group = int(snapshot.weight.shape[1])
    kernel_height = int(snapshot.weight.shape[2])
    kernel_width = int(snapshot.weight.shape[3])
    out_per_group = out_channels // topology.groups
    kernel_elements = in_per_group * kernel_height * kernel_width
    counts = torch.empty(rows, dtype=torch.int64, device="cuda")
    if rows:
        _conv_row_count_kernel[(rows,)](
            selected, possible, counts,
            in_channels, input_height, input_width,
            out_channels, output_height, output_width,
            in_per_group, out_per_group, kernel_height, kernel_width,
            topology.stride[0], topology.stride[1],
            topology.padding[0], topology.padding[1],
            topology.dilation[0], topology.dilation[1], kernel_elements,
        )
    indptr = torch.empty(rows + 1, dtype=torch.int64, device="cuda")
    indptr[0] = 0
    if rows:
        torch.cumsum(counts, dim=0, out=indptr[1:])
    total = int(indptr[-1].item())
    indices = torch.empty(total, dtype=torch.int64, device="cuda")
    data = torch.empty(total, dtype=torch.float64, device="cuda")
    weight = torch.as_tensor(snapshot.weight, dtype=torch.float64, device="cuda")
    if rows:
        _conv_row_emit_kernel[(rows,)](
            selected, possible, weight, indptr, indices, data,
            in_channels, input_height, input_width,
            out_channels, output_height, output_width,
            in_per_group, out_per_group, kernel_height, kernel_width,
            topology.stride[0], topology.stride[1],
            topology.padding[0], topology.padding[1],
            topology.dilation[0], topology.dilation[1], kernel_elements,
        )
    return _live._DeviceCSR(indptr, indices, data, rows, snapshot.input_size)


def _load():
    spec = create_specs_from_paths(_probe.ONNX, _probe.VNNLIB, category=_probe.CATEGORY)
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
    output_id = _get_output_layer_id(net)
    output_width = len(next(layer for layer in net.layers if int(layer.id) == output_id).out_vars)
    _ensure_assert_linear_encoding(
        assert_layer, B=1, n_out=output_width,
        device=torch.device("cuda"), dtype=torch.float64,
    )
    return net, entry, before, after


def main():
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    net, entry, before, after = _load()
    order, by_id = _live._topological(net)
    input_layer = next(layer for layer in order if _live._oh._kind(layer.kind) == "INPUT")
    output_layer_id = _live._preds(
        net, next(layer for layer in order if _live._oh._kind(layer.kind) == "ASSERT"), 1
    )[0]
    lower, upper = _live._facts_box(
        after, int(input_layer.id), len(input_layer.out_vars), name="gpu_csr.input"
    )
    _center, radius = _live._oh._enclosing_center_radius(lower, upper, name="gpu_csr.input")
    input_rows = np.flatnonzero(radius > 0.0).astype(np.int64)
    frames, _n_cont, _n_bin = _live._make_phase_frames(
        order, before, first_continuous_column=int(input_rows.size)
    )
    affines = {}
    for layer in order:
        if _live._oh._kind(layer.kind) in {"CONV2D", "DENSE"}:
            predecessor = _live._preds(net, layer, 1)[0]
            snapshot = _live._affine_snapshot(
                layer, input_size=len(by_id[predecessor].out_vars)
            )
            if np.any(snapshot.weight == 0.0):
                raise RuntimeError("GPU CSR probe admits only all-nonzero weights")
            affines[int(layer.id)] = snapshot
    live_rows, possible_rows = _probe._all_nonzero_live_rows(
        net, order, affines, frames, input_rows, output_layer_id
    )

    cpu_started = time.monotonic()
    cpu = {}
    for layer_id, snapshot in affines.items():
        predecessor = _live._preds(net, by_id[layer_id], 1)[0]
        cpu[layer_id] = _live._device_csr(_live._selected_affine_matrix(
            snapshot, live_rows[layer_id], possible_rows[predecessor],
            name=f"gpu_csr.cpu[{layer_id}]",
        ))
    torch.cuda.synchronize()
    cpu_seconds = time.monotonic() - cpu_started

    # Compile all observed geometries before measuring the warm-request path.
    for layer_id, snapshot in affines.items():
        predecessor = _live._preds(net, by_id[layer_id], 1)[0]
        _gpu_selected(
            snapshot, live_rows[layer_id], possible_rows[predecessor],
            name=f"gpu_csr.warm[{layer_id}]",
        )
    torch.cuda.synchronize()
    gpu_runs = []
    gpu = None
    for _ in range(3):
        started = time.monotonic()
        gpu = {}
        for layer_id, snapshot in affines.items():
            predecessor = _live._preds(net, by_id[layer_id], 1)[0]
            gpu[layer_id] = _gpu_selected(
                snapshot, live_rows[layer_id], possible_rows[predecessor],
                name=f"gpu_csr.measured[{layer_id}]",
            )
        torch.cuda.synchronize()
        gpu_runs.append(time.monotonic() - started)
    assert gpu is not None
    mismatches = []
    for layer_id in cpu:
        for field in ("indptr", "indices", "data"):
            left = getattr(cpu[layer_id], field).detach().cpu().numpy()
            right = getattr(gpu[layer_id], field).detach().cpu().numpy()
            if not np.array_equal(left.view(np.uint8), right.view(np.uint8)):
                mismatches.append([layer_id, field, list(left.shape), list(right.shape)])

    original_selected = _live._selected_affine_matrix
    original_device = _live._device_csr
    try:
        _live._selected_affine_matrix = _gpu_selected
        _live._device_csr = lambda value: value
        # First call warms non-construction kernels.  The second is the sole
        # complete-path diagnostic for this early-stop decision.
        _probe._single_stream_float64_candidate(net, entry, before, after)
        receipt = _probe._single_stream_float64_candidate(net, entry, before, after)
    finally:
        _live._selected_affine_matrix = original_selected
        _live._device_csr = original_device

    print(json.dumps({
        "schema": "act.hybridz.gpu_selected_csr_probe.v1",
        "status": "diagnostic_complete" if not mismatches else "rejected",
        "cpu_build_and_schedule_seconds": cpu_seconds,
        "gpu_build_and_schedule_seconds": gpu_runs,
        "gpu_build_and_schedule_median_seconds": float(np.median(gpu_runs)),
        "csr_mismatches": mismatches,
        "receipt": receipt,
        "scope": {
            "candidate_authority": False,
            "proof_authority": False,
            "verdict_authority": False,
            "production_integrated": False,
            "input_sampling_used": False,
            "onnx_input_execution_used": False,
            "pgd_used": False,
            "bab_or_split_used": False,
            "backward_used": False,
            "dual_tightening_used": False,
            "runtime_fallbacks": 0,
        },
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
