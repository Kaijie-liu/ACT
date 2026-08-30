#!/usr/bin/env python3
"""One-shot synthetic CUDA ownership gate for the device program.

No model, ONNX, VNNLIB, benchmark instance, sampled input, solver, or verifier
is used.  The first transaction builds and releases one tiny request-local
program.  The second injects one non-Exception ``BaseException`` immediately
after candidate, terminal, and terminal-absolute affine storage exists.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import torch

from act.back_end.hybridz_tf import (
    forward_exact_relu_live_row_stream_candidate as live,
)
from act.back_end.hybridz_tf import phase_projection_device_program as device


ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "act/back_end/hybridz_tf/phase_projection_device_program.py"
TEST = ROOT / "act/back_end/hybridz_tf/test_phase_projection_device_program.py"
MODULE_SHA256 = "7f0cce0e461f63ff6599ddd82ad5e61ef7c921eb489ef7bbbf4d60cda9048962"
TEST_SHA256 = "06d28254ea60cc20c1ebb0f009124c0a94edcea16a096733e30f3bf23ca5da64"
ALLOWED_ALLOCATED_DELTA_BYTES = 1 << 20


class InjectedBomb(BaseException):
    """Non-Exception primary whose identity must survive cleanup."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def static_reference_audit() -> dict:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned_programs = 0
    deleted_programs = 0
    forbidden_bindings = []
    forbidden = {
        "tensors",
        "candidate_weight",
        "terminal_weight",
        "absolute_weight",
        "candidate_affine",
        "terminal_affine",
    }
    clone_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id == "programs":
                assigned_programs += 1
            if node.id in forbidden:
                forbidden_bindings.append((node.id, node.lineno))
        elif isinstance(node, ast.Delete):
            deleted_programs += sum(
                isinstance(target, ast.Name) and target.id == "programs"
                for target in node.targets
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "clone"
        ):
            clone_calls += 1
    passed = (
        assigned_programs == 1
        and deleted_programs == 1
        and not forbidden_bindings
        and clone_calls == 0
    )
    return {
        "passed": passed,
        "programs_assignments": assigned_programs,
        "programs_deletions": deleted_programs,
        "forbidden_tensor_bindings": forbidden_bindings,
        "tensor_clone_calls": clone_calls,
        "strong_reference_manifest": {
            "success_persistent_owner": ["programs"],
            "success_after_delete": [
                "three weakref.ref objects",
                "three integer storage pointers",
            ],
            "failure_transient_owner": [
                "pair inside observing_seal; its successful callback frame returns before injection"
            ],
            "failure_after_injection": [
                "three weakref.ref objects",
                "the injected exception object and its sanitized traceback",
            ],
        },
    }


def traceback_names(error: BaseException) -> list[str]:
    result = []
    traceback = error.__traceback__
    while traceback is not None:
        result.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    return result


def main() -> None:
    static_audit = static_reference_audit()
    locks = {
        "device_program": sha256(MODULE),
        "device_program_test": sha256(TEST),
    }
    if locks != {
        "device_program": MODULE_SHA256,
        "device_program_test": TEST_SHA256,
    }:
        raise RuntimeError("device-program source lock drifted")
    if not static_audit["passed"]:
        raise RuntimeError("oracle strong-reference static audit failed")
    if not torch.cuda.is_available():
        raise RuntimeError("synthetic release gate requires CUDA")

    layer0 = SimpleNamespace(id=0, kind="INPUT", out_vars=[0, 1])
    layer1 = SimpleNamespace(id=1, kind="DENSE", out_vars=[0, 1])
    layer2 = SimpleNamespace(id=2, kind="RELU", out_vars=[0, 1])
    layer3 = SimpleNamespace(id=3, kind="ASSERT", out_vars=[0, 1])
    order = (layer0, layer1, layer2, layer3)
    net = SimpleNamespace(preds={0: [], 1: [0], 2: [1], 3: [2]})
    snapshot = SimpleNamespace(
        kind="DENSE",
        weight=np.asarray([[2.0, -1.0], [1.0, 3.0]], dtype=np.float64),
        bias=np.asarray([0.125, -0.25], dtype=np.float64),
        input_size=2,
        output_size=2,
        topology=None,
    )
    matrix = live._DeviceCSR(
        indptr=torch.tensor([0, 2, 4], dtype=torch.int64, device="cuda"),
        indices=torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda"),
        data=torch.tensor(
            [2.0, -1.0, 1.0, 3.0], dtype=torch.float64, device="cuda"
        ),
        rows=2,
        columns=2,
    )
    affines = {1: snapshot}
    matrices = {1: matrix}
    live_rows = {
        layer_id: np.asarray([0, 1], dtype=np.int64)
        for layer_id in range(4)
    }
    input_rows = np.asarray([0, 1], dtype=np.int64)
    input_radius = np.asarray([0.5, 0.25], dtype=np.float64)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    entry_allocated = int(torch.cuda.memory_allocated())

    programs = device.build_request_local_programs(
        net,
        order,
        affines,
        {},
        matrices,
        live_rows,
        input_rows=input_rows,
        input_radius=input_radius,
        assert_layer_id=3,
        output_layer_id=2,
        deadline=None,
    )
    success_candidate_ref = weakref.ref(
        programs.candidate.affines[1].weight
    )
    success_terminal_ref = weakref.ref(programs.terminal.affines[1].weight)
    success_absolute_ref = weakref.ref(
        programs.terminal.affines[1].absolute_weight
    )
    success_candidate_ptr = int(
        programs.candidate.affines[1].weight.untyped_storage().data_ptr()
    )
    success_terminal_ptr = int(
        programs.terminal.affines[1].weight.untyped_storage().data_ptr()
    )
    success_absolute_ptr = int(
        programs.terminal.affines[1].absolute_weight.untyped_storage().data_ptr()
    )
    success_storage_disjoint = len(
        {success_candidate_ptr, success_terminal_ptr, success_absolute_ptr}
    ) == 3
    success_peak_allocated = int(torch.cuda.memory_allocated())
    del programs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    success_released_allocated = int(torch.cuda.memory_allocated())
    success_refs_dead = bool(
        success_candidate_ref() is None
        and success_terminal_ref() is None
        and success_absolute_ref() is None
    )
    success_delta = success_released_allocated - entry_allocated
    success_allocated_returned = bool(
        success_delta <= ALLOWED_ALLOCATED_DELTA_BYTES
    )
    if not (
        success_storage_disjoint
        and success_refs_dead
        and success_allocated_returned
    ):
        receipt = {
            "schema": "act.hybridz.phase_projection_device_program.synthetic_release.v1",
            "status": "BLOCKED_AT_SUCCESS_RELEASE",
            "static_reference_audit": static_audit,
            "success": {
                "storage_disjoint": success_storage_disjoint,
                "three_storage_weakrefs_dead": success_refs_dead,
                "entry_allocated_bytes": entry_allocated,
                "peak_allocated_bytes": success_peak_allocated,
                "released_allocated_bytes": success_released_allocated,
                "released_minus_entry_bytes": success_delta,
                "allowed_allocator_constant_bytes": ALLOWED_ALLOCATED_DELTA_BYTES,
            },
            "source_locks": locks,
        }
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        raise SystemExit(2)

    failure_refs: dict[str, weakref.ReferenceType] = {}
    injection_state = {"three_storages_alive_before_raise": False}
    original_seal = device._seal_affine_pair
    original_deadline = device._deadline
    bomb = InjectedBomb("synthetic build interruption")

    def observing_seal(*args, **kwargs):
        pair = original_seal(*args, **kwargs)
        failure_refs["candidate"] = weakref.ref(pair[0].weight)
        failure_refs["terminal"] = weakref.ref(pair[1].weight)
        failure_refs["absolute"] = weakref.ref(pair[1].absolute_weight)
        return pair

    def injected_deadline(deadline, stage):
        if stage == "device affine 1 complete":
            injection_state["three_storages_alive_before_raise"] = bool(
                set(failure_refs) == {"candidate", "terminal", "absolute"}
                and all(reference() is not None for reference in failure_refs.values())
            )
            raise bomb
        return original_deadline(deadline, stage)

    device._seal_affine_pair = observing_seal
    device._deadline = injected_deadline
    failure_caught = False
    failure_identity = False
    failure_traceback = []
    failure_traceback_sanitized = False
    failure_refs_dead_while_exception_alive = False
    failure_allocated = -1
    failure_delta = -1
    failure_exception_alive_during_check = False
    try:
        device.build_request_local_programs(
            net,
            order,
            affines,
            {},
            matrices,
            live_rows,
            input_rows=input_rows,
            input_radius=input_radius,
            assert_layer_id=3,
            output_layer_id=2,
            deadline=None,
        )
    except BaseException as caught:
        failure_caught = True
        failure_identity = caught is bomb
        failure_traceback = traceback_names(caught)
        failure_traceback_sanitized = bool(
            "_build_request_local_programs_impl" not in failure_traceback
            and "_seal_affine_pair" not in failure_traceback
            and "observing_seal" not in failure_traceback
            and "injected_deadline" not in failure_traceback
        )
        device._seal_affine_pair = original_seal
        device._deadline = original_deadline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        failure_exception_alive_during_check = caught is bomb
        failure_refs_dead_while_exception_alive = bool(
            set(failure_refs) == {"candidate", "terminal", "absolute"}
            and all(reference() is None for reference in failure_refs.values())
        )
        failure_allocated = int(torch.cuda.memory_allocated())
        failure_delta = failure_allocated - entry_allocated
    finally:
        device._seal_affine_pair = original_seal
        device._deadline = original_deadline

    failure_allocated_returned = bool(
        failure_delta <= ALLOWED_ALLOCATED_DELTA_BYTES
    )
    failure_pass = bool(
        failure_caught
        and failure_identity
        and injection_state["three_storages_alive_before_raise"]
        and failure_traceback_sanitized
        and failure_exception_alive_during_check
        and failure_refs_dead_while_exception_alive
        and failure_allocated_returned
    )

    del matrices, matrix
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    final_allocated = int(torch.cuda.memory_allocated())

    all_pass = bool(
        static_audit["passed"]
        and success_storage_disjoint
        and success_refs_dead
        and success_allocated_returned
        and failure_pass
    )
    receipt = {
        "schema": "act.hybridz.phase_projection_device_program.synthetic_release.v1",
        "created_at": "2026-08-14",
        "status": "PASS_NOT_PRODUCTION" if all_pass else "BLOCKED",
        "all_checks_pass": all_pass,
        "static_reference_audit": static_audit,
        "success_release": {
            "storage_disjoint": success_storage_disjoint,
            "three_storage_weakrefs_dead": success_refs_dead,
            "entry_allocated_bytes": entry_allocated,
            "peak_allocated_bytes": success_peak_allocated,
            "released_allocated_bytes": success_released_allocated,
            "released_minus_entry_bytes": success_delta,
            "allowed_allocator_constant_bytes": ALLOWED_ALLOCATED_DELTA_BYTES,
            "allocated_returned_within_constant": success_allocated_returned,
        },
        "baseexception_release": {
            "caught": failure_caught,
            "same_primary_identity": failure_identity,
            "three_storages_alive_before_injection": injection_state[
                "three_storages_alive_before_raise"
            ],
            "traceback_frame_names": failure_traceback,
            "traceback_has_no_impl_or_injection_frame": failure_traceback_sanitized,
            "exception_alive_during_release_check": failure_exception_alive_during_check,
            "three_storage_weakrefs_dead_while_exception_alive": failure_refs_dead_while_exception_alive,
            "allocated_bytes_during_live_exception": failure_allocated,
            "allocated_minus_entry_bytes": failure_delta,
            "allocated_returned_within_constant": failure_allocated_returned,
        },
        "final_after_external_synthetic_csr_release_bytes": final_allocated,
        "source_locks": locks,
        "resource_scope": {
            "synthetic_cuda_only": True,
            "model_or_onnx_loaded": False,
            "benchmark_or_fixed_case_run": False,
            "sampling_or_onnx_point_execution": False,
            "pgd": False,
            "solver": False,
            "production_modified": False,
        },
        "claim_limit": "This gate covers only request-local CUDA ownership and BaseException cleanup for the isolated device-program component.",
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
