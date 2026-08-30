#!/usr/bin/env python3
"""One-shot iid2 bitwise oracle for the isolated device-program component.

This is a disconnected component audit.  It runs exactly one official iid2
request through the frozen phase implementation, captures that request's
already-computed intermediates, and compares the replacement device program
against them.  It never changes a verifier verdict or production source.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest import mock
import weakref

import numpy as np
import torch

from act.back_end.hybridz_tf import phase_projection_device_program as device
import scratch_phase_projection_fprime_single_owner_probe as fprime


ROOT = Path(__file__).resolve().parent
CASE = "cifar100_medium_iid2"
LOCKS = {
    "device_program": (
        ROOT / "act/back_end/hybridz_tf/phase_projection_device_program.py",
        "7f0cce0e461f63ff6599ddd82ad5e61ef7c921eb489ef7bbbf4d60cda9048962",
    ),
    "device_program_test": (
        ROOT / "act/back_end/hybridz_tf/test_phase_projection_device_program.py",
        "06d28254ea60cc20c1ebb0f009124c0a94edcea16a096733e30f3bf23ca5da64",
    ),
    "frozen_phase": (
        ROOT / "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py",
        "4b66470df55edebb595e0e06c6b8a2de5c65496b8671c4d2f2552003d01ea306",
    ),
    "frozen_live": (
        ROOT / "act/back_end/hybridz_tf/forward_exact_relu_live_row_stream_candidate.py",
        "d53c2335c43905097e78bef8311175d7151d7e98293a6152fce62dba00d37511",
    ),
    "verifier": (
        ROOT / "act/back_end/verifier.py",
        "eb3dfc8611ee97262bf71d66b8deee58a6a2544d0934d94ca7b075424dbc3afd",
    ),
    "historical_rejected_artifact": (
        ROOT / "artifacts/hybridz_largecls_gates/phase_projection_device_program_iid2_oracle_20260814.json",
        "617e7dfc5e54d5e8a67d2fbd01d0bcbefa101e090ce343ca586b0323017dce80",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def f64_bits_equal(left, right) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    return (
        left.dtype == right.dtype == np.dtype(np.float64)
        and left.shape == right.shape
        and np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def map_f64_bits_equal(left, right) -> bool:
    return set(left) == set(right) and all(
        f64_bits_equal(left[key], right[key]) for key in sorted(left)
    )


def map_values_equal(left, right) -> bool:
    return set(left) == set(right) and all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in sorted(left)
    )


def map_digest(values) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        array = np.ascontiguousarray(values[key], dtype=np.float64)
        digest.update(int(key).to_bytes(8, "little", signed=True))
        digest.update(len(array.shape).to_bytes(2, "little"))
        for width in array.shape:
            digest.update(int(width).to_bytes(8, "little"))
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def array_digest(values) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def main() -> None:
    actual_locks = {name: sha256(path) for name, (path, _expected) in LOCKS.items()}
    for name, (_path, expected) in LOCKS.items():
        if actual_locks[name] != expected:
            raise RuntimeError(f"source lock drifted: {name}")

    with LOCKS["historical_rejected_artifact"][0].open(
        "r", encoding="utf-8"
    ) as handle:
        historical = json.load(handle)
    if historical.get("files", {}).get("device_program", {}).get("sha256") != (
        "4afeaa96b0f76dfd6e6943ce27627ac6b1c9420941749735f65fdf9bee09af43"
    ):
        raise RuntimeError("historical rejected artifact source drifted")

    helper = fprime.import_frozen_helpers(CASE)
    phase = helper.phase
    helper.initialize_device(device="cuda", dtype="float64")
    helper.set_solver_mode("hybridz")
    helper.set_transfer_function_mode("interval")
    category, onnx, vnnlib = fprime.CONTROLS[CASE]
    sr = helper.create_specs_from_paths(onnx, vnnlib, category=category)
    vm = next(iter(helper.synthesize_models_from_specs([sr]).values()))
    net = helper.TorchToACT(vm).run()
    entry = helper.find_entry_layer_id(net)
    specs = helper.gather_input_spec_layers(net)
    seed = helper.seed_from_input_specs(specs)
    fact = helper.Fact(bounds=seed, cons=helper.ConSet())
    helper.add_all_input_specs(fact.cons, helper.get_input_ids(net), specs)
    before, after, _ = helper.analyze(net, entry, fact)

    captured = {}
    target_code = phase.build_forward_exact_relu_phase_projection_candidate.__code__

    def local_trace(frame, event, arg):
        if event == "return":
            captured.update(frame.f_locals)
        return local_trace

    def global_trace(frame, event, arg):
        if event == "call" and frame.f_code is target_code:
            return local_trace
        return None

    old_deadline = time.monotonic() + 90.0
    sys.settrace(global_trace)
    try:
        old_result = phase.build_forward_exact_relu_phase_projection_candidate(
            net,
            int(entry),
            before,
            after,
            deadline=old_deadline,
            lp_time_limit=30.0,
        )
    finally:
        sys.settrace(None)
    required = {
        "order",
        "affines",
        "pointwise",
        "device_matrices",
        "live_rows",
        "input_rows",
        "input_radius",
        "assert_layer",
        "output_layer_id",
        "centers",
        "projected",
        "first_assign",
        "first_pre_center",
        "first_output_center",
        "first_frames",
        "first_pre",
        "first_output",
        "target_assign",
        "target_pre_center",
        "target_output_center",
        "target_frames",
        "changes",
        "delta_pre",
        "delta_output",
        "decoded",
        "raw_lower",
        "raw_upper",
        "point_lower",
        "point_upper",
    }
    if not required.issubset(captured):
        raise RuntimeError(
            "frozen return capture missed: "
            + ",".join(sorted(required - set(captured)))
        )
    if old_result is None:
        raise RuntimeError("frozen iid2 did not reach its terminal")

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    entry_bytes = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    deadline = time.monotonic() + 90.0
    programs = device.build_request_local_programs(
        net,
        captured["order"],
        captured["affines"],
        captured["pointwise"],
        captured["device_matrices"],
        captured["live_rows"],
        input_rows=captured["input_rows"],
        input_radius=captured["input_radius"],
        assert_layer_id=int(captured["assert_layer"].id),
        output_layer_id=int(captured["output_layer_id"]),
        deadline=deadline,
    )
    weight_refs = []
    storage_disjoint = True
    for layer_id in sorted(programs.candidate.affines):
        candidate_affine = programs.candidate.affines[layer_id]
        terminal_affine = programs.terminal.affines[layer_id]
        tensors = (
            candidate_affine.weight,
            terminal_affine.weight,
            terminal_affine.absolute_weight,
        )
        storage_disjoint = storage_disjoint and len(
            {tensor.untyped_storage().data_ptr() for tensor in tensors}
        ) == 3
        weight_refs.extend(weakref.ref(tensor) for tensor in tensors)

    layer_by_snapshot = {
        id(snapshot): int(layer_id)
        for layer_id, snapshot in captured["affines"].items()
    }

    def new_affine_center(source, snapshot):
        layer_id = layer_by_snapshot[id(snapshot)]
        return device.candidate_affine_center(
            source,
            programs.candidate.affines[layer_id],
            layer_id=layer_id,
        )

    with mock.patch.object(
        phase, "_float_affine_shadow", side_effect=new_affine_center
    ):
        first_new = captured["centers"](None)
        target_new = captured["centers"](captured["projected"])

    fixed_schedule = device.seal_fixed_phase_schedule(
        programs.candidate, captured["first_frames"]
    )
    first_pre_new, first_output_new, _ = device.stream_fixed_cell_generators(
        programs.candidate, fixed_schedule
    )
    delta_schedule = device.seal_delta_schedule(
        programs.candidate,
        captured["first_frames"],
        captured["target_frames"],
        tuple(captured["changes"]),
    )
    delta_pre_new, delta_output_new, _ = device.stream_phase_deltas(
        programs.candidate, delta_schedule
    )

    checks = {
        "first_assign_equal": map_values_equal(first_new[0], captured["first_assign"]),
        "first_pre_center_bitwise_equal": map_f64_bits_equal(
            first_new[1], captured["first_pre_center"]
        ),
        "first_output_center_bitwise_equal": f64_bits_equal(
            first_new[2], captured["first_output_center"]
        ),
        "target_assign_equal": map_values_equal(target_new[0], captured["target_assign"]),
        "target_pre_center_bitwise_equal": map_f64_bits_equal(
            target_new[1], captured["target_pre_center"]
        ),
        "target_output_center_bitwise_equal": f64_bits_equal(
            target_new[2], captured["target_output_center"]
        ),
        "first_stream_preactivation_bitwise_equal": map_f64_bits_equal(
            first_pre_new, captured["first_pre"]
        ),
        "first_stream_output_bitwise_equal": f64_bits_equal(
            first_output_new, captured["first_output"]
        ),
        "delta_preactivation_bitwise_equal": map_f64_bits_equal(
            delta_pre_new, captured["delta_pre"]
        ),
        "delta_output_bitwise_equal": f64_bits_equal(
            delta_output_new, captured["delta_output"]
        ),
    }

    first_affine_id = min(programs.candidate.affines)
    candidate_affine = programs.candidate.affines[first_affine_id]
    terminal_affine = programs.terminal.affines[first_affine_id]
    terminal_weight_before = terminal_affine.weight.clone()
    terminal_absolute_before = terminal_affine.absolute_weight.clone()
    terminal_weight_version = terminal_affine.weight._version
    terminal_absolute_version = terminal_affine.absolute_weight._version
    with torch.no_grad():
        candidate_affine.weight.add_(1.0)
    candidate_mutation_isolated = bool(
        terminal_affine.weight._version == terminal_weight_version
        and terminal_affine.absolute_weight._version == terminal_absolute_version
        and torch.equal(terminal_affine.weight, terminal_weight_before)
        and torch.equal(terminal_affine.absolute_weight, terminal_absolute_before)
    )
    del terminal_weight_before, terminal_absolute_before

    sealed = device.seal_terminal_input(captured["decoded"])
    raw_box_values = sealed.values
    raw_box_ok = bool(
        np.all(raw_box_values >= np.asarray(captured["raw_lower"]).reshape(-1))
        and np.all(raw_box_values <= np.asarray(captured["raw_upper"]).reshape(-1))
    )
    terminal_saw_same_input = []
    frozen_terminal_affine = device._terminal_affine_shadow

    def observe_terminal_input(source, affine, *, layer_id):
        if not terminal_saw_same_input:
            terminal_saw_same_input.append(source.center is raw_box_values)
        return frozen_terminal_affine(source, affine, layer_id=layer_id)

    with mock.patch.object(
        device, "_terminal_affine_shadow", side_effect=observe_terminal_input
    ):
        lower_new, upper_new = device.terminal_interval_forward(
            sealed, programs.terminal
        )
    checks["terminal_lower_bitwise_equal"] = f64_bits_equal(
        lower_new, captured["point_lower"]
    )
    checks["terminal_upper_bitwise_equal"] = f64_bits_equal(
        upper_new, captured["point_upper"]
    )

    safety_checks = {
        "all_candidate_terminal_affine_storages_disjoint": storage_disjoint,
        "candidate_weight_mutation_did_not_change_terminal": candidate_mutation_isolated,
        "raw_box_uses_sealed_input": raw_box_ok,
        "terminal_saw_same_sealed_input_object": terminal_saw_same_input == [True],
        "phase_change_count_is_187": len(captured["changes"]) == 187,
        "delta_batch_count_is_3": (len(captured["changes"]) + 63) // 64 == 3,
        "input_factor_count_is_3072": int(captured["input_rows"].size) == 3072,
    }
    current_bytes = int(torch.cuda.memory_allocated())
    peak_bytes = int(torch.cuda.max_memory_allocated())
    program_reported_bytes = int(programs.candidate.cuda_bytes)

    digests = {
        "first_pre_center": map_digest(first_new[1]),
        "target_pre_center": map_digest(target_new[1]),
        "first_stream_preactivation": map_digest(first_pre_new),
        "delta_preactivation": map_digest(delta_pre_new),
        "terminal_lower": array_digest(lower_new),
        "terminal_upper": array_digest(upper_new),
    }

    del candidate_affine, terminal_affine
    del fixed_schedule, delta_schedule
    del first_new, target_new, first_pre_new, first_output_new
    del delta_pre_new, delta_output_new, lower_new, upper_new
    del new_affine_center, programs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    released_bytes = int(torch.cuda.memory_allocated())
    resources_released = all(reference() is None for reference in weight_refs)
    safety_checks["all_program_affine_tensors_released"] = resources_released
    safety_checks["allocated_bytes_returned_to_entry_or_lower"] = (
        released_bytes <= entry_bytes
    )

    all_pass = all(checks.values()) and all(safety_checks.values())
    receipt = {
        "schema": "act.hybridz.phase_projection_device_program_iid2_oracle.v2",
        "created_at": "2026-08-14",
        "status": (
            "IID2_REPAIRED_COMPONENT_BITWISE_ORACLE_PASS_NOT_PRODUCTION"
            if all_pass
            else "IID2_REPAIRED_COMPONENT_ORACLE_BLOCKED"
        ),
        "case": CASE,
        "all_checks_pass": all_pass,
        "bitwise_checks": checks,
        "safety_checks": safety_checks,
        "digests_v2_keyed_shape_framing": digests,
        "shape": {
            "input_factors": int(captured["input_rows"].size),
            "phase_changes": len(captured["changes"]),
            "delta_batches": (len(captured["changes"]) + 63) // 64,
            "factor_batch": 64,
        },
        "memory": {
            "entry_allocated_bytes": entry_bytes,
            "program_reported_cuda_bytes": program_reported_bytes,
            "current_allocated_bytes_before_release": current_bytes,
            "peak_allocated_bytes": peak_bytes,
            "incremental_peak_over_entry_bytes": max(0, peak_bytes - entry_bytes),
            "released_allocated_bytes_after_gc_and_empty_cache": released_bytes,
            "all_program_affine_tensor_weakrefs_dead": resources_released,
            "full_phase_by_input_device_buffer_used": False,
            "full_phase_by_delta_device_buffer_used": False,
            "scope": "one process, one iid2 component oracle; not a performance gate",
        },
        "source_locks": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": actual_locks[name]}
            for name, (path, _expected) in LOCKS.items()
        },
        "historical_artifact_disposition": {
            "sha256": actual_locks["historical_rejected_artifact"],
            "status": "HISTORICAL_REJECTED_AFTER_P0_REDTEAM",
            "reason": "candidate/terminal CUDA storage and terminal host authority were not isolated",
            "causal_evidence_reused": False,
        },
        "focused_cpu_gate": {
            "warnings_as_errors": True,
            "tests_passed": 16,
            "tests_failed": 0,
        },
        "restrictions": {
            "single_iid2_only": True,
            "gpu_oracle_runs": 1,
            "fixed_or_benchmark_gate_run": False,
            "input_sampling_used": False,
            "onnx_input_point_execution_used": False,
            "pgd_used": False,
            "bab_split_or_enumeration_used": False,
            "backward_bounds_used": False,
            "dual_tightening_used": False,
            "solver_called_by_new_device_module": False,
            "production_integrated": False,
            "formal_fixed400_unchanged": 59,
        },
        "claim_limit": (
            "This readmits only the repaired isolated device-program component "
            "for production wiring review. It is not a performance, fixed-suite, "
            "formal-score, or verifier-authority result."
        ),
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
