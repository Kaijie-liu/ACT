#!/usr/bin/env python3
"""Disposable paired throughput gate for the phase-projection candidate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import statistics
import time

import torch

from scratch_phase_projection_float64_probe import (
    _single_stream_float64_candidate,
)

from act.back_end.analyze import analyze
from act.back_end.core import ConSet, Fact
from act.back_end.hybridz_tf.forward_exact_relu_phase_projection_candidate import (
    ExactReLUPhaseProjectionUnknown,
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
PAIRS = int(os.environ.get("ACT_PHASE_PROJECTION_PAIRS", "5"))
WORKERS = int(os.environ.get("ACT_PHASE_PROJECTION_WORKERS", "4"))


def main() -> None:
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
        assert_layer,
        B=1,
        n_out=output_width,
        device=torch.device("cuda"),
        dtype=torch.float64,
    )

    def candidate():
        started = time.monotonic()
        try:
            result = _single_stream_float64_candidate(net, entry, before, after)
            return time.monotonic() - started, result["status"], result
        except ExactReLUPhaseProjectionUnknown as exc:
            return time.monotonic() - started, "UNKNOWN", str(exc)

    def baseline():
        started = time.monotonic()
        try:
            result = build_forward_exact_relu_phase_projection_candidate(
                net,
                entry,
                before,
                after,
                deadline=started + 10.0,
                lp_time_limit=10.0,
            )
            return time.monotonic() - started, result.receipt.status, None
        except ExactReLUPhaseProjectionUnknown as exc:
            return time.monotonic() - started, "UNKNOWN", str(exc)

    def run_group(function):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            values = tuple(pool.map(lambda _index: function(), range(WORKERS)))
        return time.monotonic() - started, values

    # Two complete warmup groups per path compile kernels and import solvers.
    # They are fixed by the gate protocol and never enter measured pairs.
    for warmup in range(2):
        order = (baseline, candidate) if warmup % 2 == 0 else (candidate, baseline)
        for function in order:
            run_group(function)

    single_baseline = baseline()
    single_candidate = candidate()
    groups = []
    all_conflicts = 0
    for pair in range(PAIRS):
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        measured = {}
        outcomes = {}
        reasons = {}
        for name in order:
            function = baseline if name == "baseline" else candidate
            measured[name], values = run_group(function)
            outcomes[name] = [value[1] for value in values]
            reasons[name] = [value[2] if name == "candidate" else None for value in values]
            all_conflicts += int(len(set(outcomes[name])) != 1)
        groups.append(
            {
                "pair": pair,
                "order": list(order),
                "baseline_seconds": measured["baseline"],
                "candidate_seconds": measured["candidate"],
                "speedup": measured["baseline"] / measured["candidate"],
                "baseline_outcomes": outcomes["baseline"],
                "candidate_outcomes": outcomes["candidate"],
                "candidate_reasons": reasons["candidate"],
            }
        )
    speedups = [item["speedup"] for item in groups]
    bootstrap_medians = []
    for encoded in range(PAIRS**PAIRS):
        value = encoded
        sample = []
        for _slot in range(PAIRS):
            sample.append(speedups[value % PAIRS])
            value //= PAIRS
        bootstrap_medians.append(statistics.median(sample))
    bootstrap_medians.sort()
    lower_index = int(math.floor(0.025 * (len(bootstrap_medians) - 1)))
    print(
        json.dumps(
            {
                "schema": "act.hybridz.phase_projection_perf_gate.v1",
                "scope": {
                    "same_parsed_graph_and_forward_facts": True,
                    "workers": WORKERS,
                    "pairs": PAIRS,
                    "per_call_deadline_seconds": 10.0,
                    "input_sampling_used": False,
                    "pgd_used": False,
                    "concrete_onnx_execution_used": False,
                    "bab_used": False,
                    "backward_used": False,
                    "dual_tightening_used": False,
                    "candidate_generator_representation": "gpu_emitted_selected_csr_v1",
                    "runtime_fallbacks": 0,
                },
                "single": {
                    "baseline_seconds": single_baseline[0],
                    "candidate_seconds": single_candidate[0],
                    "speedup": single_baseline[0] / single_candidate[0],
                    "baseline_outcome": single_baseline[1],
                    "candidate_outcome": single_candidate[1],
                },
                "groups": groups,
                "median_speedup": statistics.median(speedups),
                "paired_bootstrap_95_lower": bootstrap_medians[lower_index],
                "id_conflicts": all_conflicts,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
