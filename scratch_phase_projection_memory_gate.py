#!/usr/bin/env python3
"""Fresh-process four-request memory gate for one selected path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import resource
import statistics
import time

import torch

from scratch_phase_projection_float64_probe import _single_stream_float64_candidate
from act.back_end.analyze import analyze
from act.back_end.core import ConSet, Fact
from act.back_end.hybridz_tf.forward_exact_relu_phase_projection_candidate import (
    build_forward_exact_relu_phase_projection_candidate,
)
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


ONNX = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx"
VNNLIB = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib"


def _hwm_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def main():
    mode = os.environ.get("ACT_MEMORY_GATE_MODE")
    if mode not in {"baseline", "candidate"}:
        raise RuntimeError("ACT_MEMORY_GATE_MODE must be baseline or candidate")
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    spec = create_specs_from_paths(ONNX, VNNLIB, category="cifar100_2024")
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
    output_width = len(next(layer for layer in net.layers if int(layer.id) == output_id).out_vars)
    _ensure_assert_linear_encoding(
        assert_layer, B=1, n_out=output_width,
        device=torch.device("cuda"), dtype=torch.float64,
    )
    load_hwm = _hwm_bytes()

    def call():
        if mode == "baseline":
            result = build_forward_exact_relu_phase_projection_candidate(
                net, entry, before, after, deadline=None, lp_time_limit=10.0
            )
            return result.receipt.status
        return _single_stream_float64_candidate(net, entry, before, after)["status"]

    def group():
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = tuple(pool.map(lambda _index: call(), range(4)))
        torch.cuda.synchronize()
        return time.monotonic() - started, outcomes

    for _ in range(2):
        group()
    seconds = []
    cuda_peaks = []
    outcomes = []
    for _ in range(5):
        torch.cuda.reset_peak_memory_stats()
        elapsed, values = group()
        seconds.append(elapsed)
        outcomes.append(values)
        cuda_peaks.append(int(torch.cuda.max_memory_allocated()))
    print(json.dumps({
        "schema": "act.hybridz.phase_projection_memory_gate.v1",
        "mode": mode,
        "workers": 4,
        "warmup_groups": 2,
        "measured_groups": 5,
        "group_seconds": seconds,
        "median_group_seconds": statistics.median(seconds),
        "load_hwm_bytes": load_hwm,
        "final_hwm_bytes": _hwm_bytes(),
        "cuda_peak_bytes": cuda_peaks,
        "median_cuda_peak_bytes": statistics.median(cuda_peaks),
        "outcomes": outcomes,
        "scope": {
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
