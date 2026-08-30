#!/usr/bin/env python3
"""Bounded iid2 component sentinel for a request-local CUDA affine program.

This is deliberately scratch-only.  It compares the frozen 59-path's first
generator stream plus its causally later zero-width terminal replay with one
fixed implementation that pre-seals immutable CUDA weights, absolute weights,
and row schedules inside *each request*.  The terminal accepts only the decoded
binary64 input and never consumes candidate outputs, phases, margins, or error
bands.  Stored arithmetic and the final outward/Fraction authority are not
changed here.

There is no input sampling, concrete ONNX point execution, PGD, BaB/splitting,
backward propagation, dual tightening, retry menu, or production edit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from act.back_end.analyze import analyze
from act.back_end.core import ConSet, Fact
from act.back_end.hybridz_tf import forward_exact_relu_live_row_stream_candidate as live
from act.back_end.hybridz_tf import forward_exact_relu_phase_projection_candidate as phase
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


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts/hybridz_largecls_gates/device_affine_program_iid2_sentinel_20260814.json"
ONNX = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx")
VNNLIB = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib")
WORKERS = 4
PAIRS = 5
MIN_GROUP_SAVING_SECONDS = 0.20


@dataclass(frozen=True)
class DeviceAffine:
    weight: torch.Tensor
    absolute_weight: torch.Tensor
    support_kernel: Optional[torch.Tensor]
    fanin: np.ndarray
    gamma: np.ndarray
    absolute_bias: np.ndarray


@dataclass(frozen=True)
class DeviceProgram:
    affines: Mapping[int, DeviceAffine]
    pointwise: Mapping[int, torch.Tensor]
    stream_rows: Mapping[int, torch.Tensor]
    active_rows: Mapping[int, torch.Tensor]
    input_batches: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    successor_uses: Mapping[int, int]
    build_seconds: float
    cuda_bytes: int


@dataclass(frozen=True)
class Lane:
    matrices: Mapping[int, live._DeviceCSR]
    rows: Mapping[int, torch.Tensor]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def restrictions() -> Dict[str, Any]:
    return {
        "scope": "frozen_cifar100_medium_iid2_component_only",
        "production_modified": False,
        "request_local_device_program": True,
        "cross_request_device_cache": False,
        "candidate_terminal_causal_independence": True,
        "candidate_has_authority": False,
        "input_sampling_used": False,
        "onnx_input_point_execution_used": False,
        "pgd_used": False,
        "bab_or_split_used": False,
        "backward_or_dual_tightening_used": False,
        "runtime_menu_or_retry": False,
    }


def load_context() -> Tuple[Dict[str, Any], Any]:
    """Run the frozen verifier once and capture its successful local state."""

    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    spec = create_specs_from_paths(str(ONNX), str(VNNLIB), category="cifar100_2024")
    wrapped = next(iter(synthesize_models_from_specs([spec]).values()))
    net = TorchToACT(wrapped).run()
    entry = int(find_entry_layer_id(net))
    input_specs = gather_input_spec_layers(net)
    seed = seed_from_input_specs(input_specs)
    fact = Fact(bounds=seed, cons=ConSet())
    add_all_input_specs(fact.cons, get_input_ids(net), input_specs)
    before, after, _ = analyze(net, entry, fact)
    assert_layer = get_assert_layer(net)
    output_id = _get_output_layer_id(net)
    output_width = len(
        next(layer for layer in net.layers if int(layer.id) == output_id).out_vars
    )
    _ensure_assert_linear_encoding(
        assert_layer,
        B=1,
        n_out=output_width,
        device=torch.device("cuda"),
        dtype=torch.float64,
    )

    captured: Dict[str, Any] = {}
    target = phase.build_forward_exact_relu_phase_projection_candidate.__code__

    def tracer(frame, event, arg):
        if frame.f_code is target and event == "return":
            captured.update(frame.f_locals.copy())
        return tracer

    sys.settrace(tracer)
    try:
        result = phase.build_forward_exact_relu_phase_projection_candidate(
            net,
            entry,
            before,
            after,
            deadline=time.monotonic() + 30.0,
            lp_time_limit=30.0,
        )
    finally:
        sys.settrace(None)
    if result.receipt.status != "singleton_verified" or not captured:
        raise RuntimeError("frozen iid2 capture did not reach terminal verification")
    captured["net"] = net
    return captured, result


def clone_lane(c: Mapping[str, Any]) -> Lane:
    matrices = {
        layer_id: live._DeviceCSR(
            matrix.indptr.clone(),
            matrix.indices.clone(),
            matrix.data.clone(),
            matrix.rows,
            matrix.columns,
        )
        for layer_id, matrix in c["device_matrices"].items()
    }
    rows = {layer_id: value.clone() for layer_id, value in c["device_rows"].items()}
    torch.cuda.synchronize()
    return Lane(matrices=matrices, rows=rows)


def build_program(c: Mapping[str, Any]) -> DeviceProgram:
    """Build exactly one request-local immutable device schedule."""

    torch.cuda.synchronize()
    entry_bytes = int(torch.cuda.memory_allocated())
    started = time.monotonic()
    device_affines: Dict[int, DeviceAffine] = {}
    for layer_id, snapshot in c["affines"].items():
        weight = torch.tensor(snapshot.weight, dtype=torch.float64, device="cuda")
        absolute_weight = torch.abs(weight)
        support_kernel = None
        if snapshot.kind == "CONV2D":
            support_kernel = torch.ones_like(weight)
        fanin = live._affine_fanin(snapshot)
        gamma = phase._oh._gamma_ops(
            2.0 * fanin + 2.0, name=f"scratch.program.gamma[{layer_id}]"
        )
        device_affines[layer_id] = DeviceAffine(
            weight=weight,
            absolute_weight=absolute_weight,
            support_kernel=support_kernel,
            fanin=fanin,
            gamma=gamma,
            absolute_bias=np.abs(snapshot.bias),
        )
    pointwise = {
        layer_id: torch.as_tensor(value, dtype=torch.float64, device="cuda")
        for layer_id, value in c["pointwise"].items()
    }
    stream_rows = {
        layer_id: torch.as_tensor(
            frame.stream_rows, dtype=torch.int64, device="cuda"
        )
        for layer_id, frame in c["first_frames"].items()
    }
    active_rows = {
        layer_id: torch.as_tensor(
            np.intersect1d(
                c["live_rows"][layer_id], frame.active, assume_unique=True
            ),
            dtype=torch.int64,
            device="cuda",
        )
        for layer_id, frame in c["first_frames"].items()
    }
    input_batches = []
    input_rows = np.asarray(c["input_rows"], dtype=np.int64)
    input_radius = np.asarray(c["input_radius"], dtype=np.float64)
    for start in range(0, int(input_rows.size), live._FACTOR_BATCH):
        stop = min(int(input_rows.size), start + live._FACTOR_BATCH)
        selected_rows = input_rows[start:stop]
        input_batches.append(
            (
                torch.as_tensor(selected_rows, dtype=torch.int64, device="cuda"),
                torch.arange(stop - start, dtype=torch.int64, device="cuda"),
                torch.as_tensor(
                    input_radius[selected_rows], dtype=torch.float64, device="cuda"
                ),
            )
        )
    successor_uses = {int(layer.id): 0 for layer in c["order"]}
    for layer in c["order"]:
        for predecessor in c["net"].preds.get(int(layer.id), []):
            successor_uses[int(predecessor)] += 1
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    return DeviceProgram(
        affines=device_affines,
        pointwise=pointwise,
        stream_rows=stream_rows,
        active_rows=active_rows,
        input_batches=tuple(input_batches),
        successor_uses=successor_uses,
        build_seconds=elapsed,
        cuda_bytes=max(0, int(torch.cuda.memory_allocated()) - entry_bytes),
    )


def presealed_stream(
    c: Mapping[str, Any], lane: Lane, program: DeviceProgram
) -> Tuple[Dict[int, np.ndarray], np.ndarray, float]:
    """Same ordered-CSR arithmetic, with one device buffer/copy schedule."""

    n_cont = int(np.asarray(c["input_rows"]).size)
    device_pre = {
        layer_id: torch.empty(
            (frame.stream_rows.size, n_cont), dtype=torch.float64, device="cuda"
        )
        for layer_id, frame in c["first_frames"].items()
    }
    output_device = torch.empty(
        (len(c["assert_layer"].out_vars), n_cont),
        dtype=torch.float64,
        device="cuda",
    )
    finite = torch.ones((), dtype=torch.bool, device="cuda")
    started = time.monotonic()
    for batch_index, start in enumerate(range(0, n_cont, live._FACTOR_BATCH)):
        stop = min(n_cont, start + live._FACTOR_BATCH)
        width = stop - start
        values: Dict[int, torch.Tensor] = {}
        remaining_uses = dict(program.successor_uses)
        for layer in c["order"]:
            layer_id = int(layer.id)
            kind = phase._oh._kind(layer.kind)
            predecessors = tuple(
                int(value) for value in c["net"].preds.get(layer_id, [])
            )
            if kind == "INPUT":
                value = torch.zeros(
                    (len(layer.out_vars), width), dtype=torch.float64, device="cuda"
                )
                rows, columns, radii = program.input_batches[batch_index]
                value[rows, columns] = radii
                values[layer_id] = value
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                values[layer_id] = values[predecessors[0]]
            elif kind in {"CONV2D", "DENSE"}:
                source = values[predecessors[0]]
                selected_value = live._ordered_csr_dense(
                    lane.matrices[layer_id], source
                )
                value = torch.zeros(
                    (len(layer.out_vars), width), dtype=torch.float64, device="cuda"
                )
                if lane.rows[layer_id].numel():
                    value[lane.rows[layer_id]] = selected_value
                values[layer_id] = value
                finite = finite & torch.isfinite(selected_value).all()
            elif kind in {"SCALE", "BIAS"}:
                source = values[predecessors[0]]
                if kind == "SCALE":
                    values[layer_id] = source * program.pointwise[layer_id].reshape(-1, 1)
                    finite = finite & torch.isfinite(values[layer_id]).all()
                else:
                    values[layer_id] = source
            elif kind == "ADD":
                left, right = predecessors
                values[layer_id] = values[left] + values[right]
                finite = finite & torch.isfinite(values[layer_id]).all()
            elif kind == "RELU":
                source = values[predecessors[0]]
                device_pre[layer_id][:, start:stop] = source[
                    program.stream_rows[layer_id]
                ]
                value = torch.zeros_like(source)
                active = program.active_rows[layer_id]
                if active.numel():
                    value[active] = source[active]
                values[layer_id] = value
            else:
                raise RuntimeError(f"unexpected generator kind {kind}")
            for predecessor in predecessors:
                remaining_uses[predecessor] -= 1
                if remaining_uses[predecessor] == 0:
                    del values[predecessor]
        output_device[:, start:stop] = values[int(c["assert_layer"].id)]
        del values
    if not bool(finite.item()):
        raise RuntimeError("presealed generator overflowed")
    preactivation = {
        layer_id: value.detach().cpu().numpy()
        for layer_id, value in device_pre.items()
    }
    output = output_device.detach().cpu().numpy()
    torch.cuda.synchronize()
    return preactivation, output, time.monotonic() - started


def _program_support(
    snapshot: Any, affine: DeviceAffine, mass_mask: np.ndarray, error_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if snapshot.kind == "DENSE":
        return (
            np.full(snapshot.output_size, bool(np.any(mass_mask)), dtype=bool),
            np.full(snapshot.output_size, bool(np.any(error_mask)), dtype=bool),
        )
    topology = snapshot.topology
    if topology is None or affine.support_kernel is None:
        raise RuntimeError("program Conv support schedule is missing")
    batch, channels, height, width = topology.input_shape
    source = np.stack((mass_mask, error_mask), axis=0)
    device = torch.as_tensor(
        source.reshape(2 * batch, channels, height, width),
        dtype=torch.float64,
        device="cuda",
    )
    counts = torch.nn.functional.conv2d(
        device,
        affine.support_kernel,
        bias=None,
        stride=topology.stride,
        padding=topology.padding,
        dilation=topology.dilation,
        groups=topology.groups,
    )
    masks = (counts > 0.0).detach().cpu().numpy().reshape(2, snapshot.output_size)
    return masks[0], masks[1]


def _program_affine_shadow(
    source: live._Shadow,
    snapshot: Any,
    affine: DeviceAffine,
    *,
    layer_id: int,
) -> live._Shadow:
    if snapshot.kind == "DENSE":
        source_center = torch.as_tensor(
            source.center.reshape(-1, 1), dtype=torch.float64, device="cuda"
        )
        source_nonnegative = torch.as_tensor(
            np.column_stack((source.mass_upper, source.error)),
            dtype=torch.float64,
            device="cuda",
        )
        raw_center_tensor = torch.matmul(affine.weight, source_center)
        raw_nonnegative_tensor = torch.matmul(
            affine.absolute_weight, source_nonnegative
        )
    else:
        topology = snapshot.topology
        if topology is None:
            raise RuntimeError("program Conv topology is missing")
        batch, channels, height, width = topology.input_shape
        source_center = torch.as_tensor(
            source.center.reshape(batch, channels, height, width),
            dtype=torch.float64,
            device="cuda",
        )
        raw_center_tensor = torch.nn.functional.conv2d(
            source_center,
            affine.weight,
            bias=None,
            stride=topology.stride,
            padding=topology.padding,
            dilation=topology.dilation,
            groups=topology.groups,
        )
        nonnegative = np.stack((source.mass_upper, source.error), axis=0)
        source_nonnegative = torch.as_tensor(
            nonnegative.reshape(2 * batch, channels, height, width),
            dtype=torch.float64,
            device="cuda",
        )
        raw_nonnegative_tensor = torch.nn.functional.conv2d(
            source_nonnegative,
            affine.absolute_weight,
            bias=None,
            stride=topology.stride,
            padding=topology.padding,
            dilation=topology.dilation,
            groups=topology.groups,
        )
        raw_nonnegative_tensor = raw_nonnegative_tensor.reshape(
            2, snapshot.output_size
        ).transpose(0, 1)
    raw_center = raw_center_tensor.detach().cpu().numpy().reshape(-1)
    raw_nonnegative = raw_nonnegative_tensor.detach().cpu().numpy().reshape(
        snapshot.output_size, 2
    )
    center = raw_center + snapshot.bias
    mass_support, error_support = _program_support(
        snapshot,
        affine,
        source.mass_upper > 0.0,
        source.error > 0.0,
    )
    transformed = live._positive_gpu_result_upper(
        raw_nonnegative[:, 0],
        affine.fanin,
        mass_support,
        name=f"scratch.program.mass[{layer_id}]",
    )
    arithmetic_mass = phase._oh._nonnegative_sum_upper(
        transformed,
        affine.absolute_bias,
        name=f"scratch.program.arithmetic_mass[{layer_id}]",
    )
    propagated = live._positive_gpu_result_upper(
        raw_nonnegative[:, 1],
        affine.fanin,
        error_support,
        name=f"scratch.program.propagated_error[{layer_id}]",
    )
    arithmetic_error = phase._oh._inflate_nonnegative(
        affine.gamma * arithmetic_mass,
        4,
        active=arithmetic_mass > 0.0,
        name=f"scratch.program.arithmetic_error[{layer_id}]",
    )
    error = phase._oh._nonnegative_sum_upper(
        propagated,
        arithmetic_error,
        name=f"scratch.program.total_error[{layer_id}]",
    )
    mass = phase._oh._nonnegative_sum_upper(
        transformed,
        affine.absolute_bias,
        arithmetic_error,
        name=f"scratch.program.output_mass[{layer_id}]",
    )
    if not np.all(np.isfinite(center)):
        raise RuntimeError("program affine center overflowed")
    return live._Shadow(center, error, mass)


def presealed_terminal(
    c: Mapping[str, Any], program: DeviceProgram
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Independent outward replay; its only varying input is decoded binary64."""

    started = time.monotonic()
    shadows: Dict[int, live._Shadow] = {}
    flat = np.asarray(c["decoded"], dtype=np.float64).reshape(-1)
    input_mass = np.abs(flat)
    for layer in c["order"]:
        layer_id = int(layer.id)
        kind = phase._oh._kind(layer.kind)
        predecessors = tuple(
            int(value) for value in c["net"].preds.get(layer_id, [])
        )
        if kind == "INPUT":
            shadows[layer_id] = live._Shadow(
                flat.copy(), np.zeros(flat.size, dtype=np.float64), input_mass
            )
        elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
            shadows[layer_id] = shadows[predecessors[0]]
        elif kind in {"CONV2D", "DENSE"}:
            shadows[layer_id] = _program_affine_shadow(
                shadows[predecessors[0]],
                c["affines"][layer_id],
                program.affines[layer_id],
                layer_id=layer_id,
            )
        elif kind in {"SCALE", "BIAS"}:
            source = shadows[predecessors[0]]
            lower = np.nextafter(source.center - source.error, -np.inf)
            upper = np.nextafter(source.center + source.error, np.inf)
            parameter = c["pointwise"][layer_id]
            if kind == "SCALE":
                first = lower * parameter
                second = upper * parameter
                lower = np.nextafter(np.minimum(first, second), -np.inf)
                upper = np.nextafter(np.maximum(first, second), np.inf)
            else:
                lower = np.nextafter(lower + parameter, -np.inf)
                upper = np.nextafter(upper + parameter, np.inf)
            center, error = phase._oh._enclosing_center_radius(
                lower, upper, name=f"scratch.program.{kind.lower()}[{layer_id}]"
            )
            mass = phase._oh._nonnegative_sum_upper(
                np.abs(center),
                error,
                name=f"scratch.program.{kind.lower()}_mass[{layer_id}]",
            )
            shadows[layer_id] = live._Shadow(center, error, mass)
        elif kind == "ADD":
            shadows[layer_id] = live._add_shadow(
                shadows[predecessors[0]], shadows[predecessors[1]], layer_id=layer_id
            )
        elif kind == "RELU":
            source = shadows[predecessors[0]]
            lower = np.nextafter(source.center - source.error, -np.inf)
            upper = np.nextafter(source.center + source.error, np.inf)
            relu_lower = np.maximum(lower, 0.0)
            relu_upper = np.maximum(upper, 0.0)
            center, error = phase._oh._enclosing_center_radius(
                relu_lower,
                relu_upper,
                name=f"scratch.program.relu[{layer_id}]",
            )
            mass = phase._oh._nonnegative_sum_upper(
                np.abs(center),
                error,
                name=f"scratch.program.relu_mass[{layer_id}]",
            )
            shadows[layer_id] = live._Shadow(center, error, mass)
        else:
            raise RuntimeError(f"unexpected terminal kind {kind}")
    output = shadows[int(c["output_layer_id"])]
    lower = np.nextafter(output.center - output.error, -np.inf)
    upper = np.nextafter(output.center + output.error, np.inf)
    return lower, upper, time.monotonic() - started


def current_component(c: Mapping[str, Any], lane: Lane) -> Dict[str, float]:
    started = time.monotonic()
    _pre, _output, stream_seconds = live._stream_generators(
        c["net"],
        c["order"],
        c["first_frames"],
        c["live_rows"],
        lane.matrices,
        lane.rows,
        input_rows=c["input_rows"],
        input_radius=c["input_radius"],
        n_cont=int(np.asarray(c["input_rows"]).size),
        assert_layer=c["assert_layer"],
        deadline=None,
        stage_prefix="scratch_current",
        collect_output=True,
        pointwise=c["pointwise"],
    )
    terminal_started = time.monotonic()
    phase._singleton_interval_forward(
        c["net"],
        c["order"],
        c["affines"],
        np.asarray(c["decoded"]).reshape(c["input_shape"]),
        c["output_layer_id"],
        pointwise=c["pointwise"],
        deadline=None,
    )
    terminal_seconds = time.monotonic() - terminal_started
    return {
        "total_seconds": time.monotonic() - started,
        "program_build_seconds": 0.0,
        "stream_seconds": stream_seconds,
        "terminal_seconds": terminal_seconds,
    }


def proposed_component(c: Mapping[str, Any], lane: Lane) -> Dict[str, float]:
    started = time.monotonic()
    program = build_program(c)
    _pre, _output, stream_seconds = presealed_stream(c, lane, program)
    _lower, _upper, terminal_seconds = presealed_terminal(c, program)
    result = {
        "total_seconds": time.monotonic() - started,
        "program_build_seconds": program.build_seconds,
        "stream_seconds": stream_seconds,
        "terminal_seconds": terminal_seconds,
        "program_cuda_bytes": program.cuda_bytes,
    }
    del program
    return result


def component_oracle(c: Mapping[str, Any], lane: Lane) -> Dict[str, Any]:
    program = build_program(c)
    pre, output, _ = presealed_stream(c, lane, program)
    lower, upper, _ = presealed_terminal(c, program)
    generator_layer_equal = {
        str(layer_id): bool(np.array_equal(pre[layer_id], c["first_pre"][layer_id]))
        for layer_id in pre
    }
    generator_output_equal = bool(np.array_equal(output, c["first_output"]))
    lower_equal = bool(np.array_equal(lower, c["point_lower"]))
    upper_equal = bool(np.array_equal(upper, c["point_upper"]))
    lower_encloses = bool(np.all(lower <= c["point_lower"]))
    upper_encloses = bool(np.all(upper >= c["point_upper"]))
    result = {
        "generator_all_layers_bitwise_equal": bool(all(generator_layer_equal.values())),
        "generator_layer_bitwise_equal": generator_layer_equal,
        "generator_output_bitwise_equal": generator_output_equal,
        "terminal_lower_bitwise_equal": lower_equal,
        "terminal_upper_bitwise_equal": upper_equal,
        "terminal_interval_encloses_frozen": lower_encloses and upper_encloses,
        "terminal_signature_inputs": ["decoded_binary64", "stored_affines", "graph_topology"],
        "candidate_outputs_consumed_by_terminal": False,
        "program_cuda_bytes": program.cuda_bytes,
    }
    del program
    return result


def run_group(function, lanes: Sequence[Lane]) -> Tuple[float, Sequence[Dict[str, float]]]:
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        values = tuple(pool.map(lambda lane: function(lane), lanes))
    return time.monotonic() - started, values


def main() -> None:
    c, frozen_result = load_context()
    lanes = tuple(clone_lane(c) for _ in range(WORKERS))
    oracle = component_oracle(c, lanes[0])
    oracle_pass = bool(
        oracle["generator_all_layers_bitwise_equal"]
        and oracle["generator_output_bitwise_equal"]
        and oracle["terminal_interval_encloses_frozen"]
    )
    if not oracle_pass:
        raise RuntimeError("component equivalence/outward oracle failed")

    current_single = lambda: current_component(c, lanes[0])
    proposed_single = lambda: proposed_component(c, lanes[0])
    current_group = lambda lane: current_component(c, lane)
    proposed_group = lambda lane: proposed_component(c, lane)

    # Fixed warmup: one single and one group for each shape, never measured.
    current_single()
    proposed_single()
    run_group(current_group, lanes)
    run_group(proposed_group, lanes)

    pairs = []
    for pair in range(PAIRS):
        order = ("current", "proposed") if pair % 2 == 0 else ("proposed", "current")
        record: Dict[str, Any] = {"pair": pair, "order": list(order)}
        for name in order:
            if name == "current":
                single = current_single()
                group_seconds, group_values = run_group(current_group, lanes)
            else:
                single = proposed_single()
                group_seconds, group_values = run_group(proposed_group, lanes)
            record[f"{name}_single"] = single
            record[f"{name}_group_wall_seconds"] = group_seconds
            record[f"{name}_group_request_seconds"] = [
                value["total_seconds"] for value in group_values
            ]
        record["single_saving_seconds"] = (
            record["current_single"]["total_seconds"]
            - record["proposed_single"]["total_seconds"]
        )
        record["group_saving_seconds"] = (
            record["current_group_wall_seconds"]
            - record["proposed_group_wall_seconds"]
        )
        pairs.append(record)

    group_savings = [item["group_saving_seconds"] for item in pairs]
    single_savings = [item["single_saving_seconds"] for item in pairs]
    median_group_saving = statistics.median(group_savings)
    median_single_saving = statistics.median(single_savings)
    hard_gate_pass = bool(median_group_saving >= MIN_GROUP_SAVING_SECONDS)
    decision = (
        "COMPONENT_SENTINEL_PASSES_0P20_REOPEN_FULL_STRUCTURAL_DESIGN"
        if hard_gate_pass
        else "STOP_LOSS_COMPONENT_SAVING_BELOW_0P20"
    )
    artifact = {
        "schema": "act.scratch.device_affine_program_iid2_sentinel.v1",
        "scratch_sha256": sha256(Path(__file__).resolve()),
        "created_at": "2026-08-14",
        "status": decision,
        "audit_complete": True,
        "formal_fixed400_unchanged": 59,
        "hard_gate": {
            "metric": "median four-concurrent combined first-stream plus independent terminal wall saving",
            "minimum_saving_seconds": MIN_GROUP_SAVING_SECONDS,
            "observed_median_saving_seconds": median_group_saving,
            "pass": hard_gate_pass,
        },
        "single_observed_median_saving_seconds": median_single_saving,
        "oracle": oracle,
        "frozen_iid2_receipt": {
            "first_stream_seconds": frozen_result.receipt.first_stream_seconds,
            "singleton_seconds": frozen_result.receipt.singleton_seconds,
            "singleton_margin_lower": frozen_result.receipt.singleton_margin_lower,
        },
        "measurements": pairs,
        "method": {
            "pairs": PAIRS,
            "workers": WORKERS,
            "warmup_single_per_path": 1,
            "warmup_group_per_path": 1,
            "proposal_program_constructed_inside_every_timed_request": True,
            "request_lanes_have_distinct_device_csr_and_row_tensors": True,
            "device_program_reused_across_requests": False,
            "full_promotion_gate_claimed": False,
        },
        "static_findings": {
            "current_first_stream_repeated_materialization": [
                "per-batch CUDA stream-row tensors",
                "per-batch active-live intersection and CUDA tensors",
                "per-batch SCALE CUDA tensors",
                "per-layer-per-batch host copies of selected preactivation rows",
            ],
            "current_terminal_repeated_materialization": [
                "full stored weight CPU-to-CUDA transfer at every affine",
                "absolute weight materialization at every affine",
                "fanin reconstruction from stored support at every affine",
                "two CPU convolution-support walks at every affine",
            ],
            "safe_shared_scope": "request-local immutable weights, abs-weights, fanin, support kernels, and index schedule only",
            "forbidden_reuse": "decoded terminal receives no candidate output, phase, margin, shadow, or error band",
        },
        "restrictions": restrictions(),
        "production_files_modified": [],
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
