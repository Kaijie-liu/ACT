#!/usr/bin/env python3
"""One-shot production compatibility sentinel for the F-prime path.

This is an offline gate, not a production dispatcher.  It runs the current
production ``verify_once`` phase-projection path on five frozen controls in a
fixed order.  Every case gets at most one fresh-child attempt in an artifact
lifetime.  There is no input sampling, point ONNX execution, PGD, BaB/split,
backward bound propagation, dual tightening, fallback, retry, or parameter
menu.

The worker instruments the production request-local owner only to count
logical/native owners, solves, ray reads, and cleanup.  All operations are
delegated to the real production classes; no candidate algorithm is copied
here.  Candidate LP/dual data have no authority.  A FALSIFIED result is valid
only when the formal verifier reports its unchanged raw-BOX, zero-width
outward-forward, and stored-binary64 Fraction terminal proof.

``--static-check`` is deliberately CPU-only and creates no artifact.  Running
without arguments is the separately-authorized CUDA action.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
EVENTS_PATH = (
    ROOT
    / "artifacts/hybridz_largecls_gates/"
    "phase_projection_fprime_production_five_case_sentinel_20260814.events.jsonl"
)
SUMMARY_PATH = (
    ROOT
    / "artifacts/hybridz_largecls_gates/"
    "phase_projection_fprime_production_five_case_sentinel_20260814.json"
)
RESULT_PREFIX = "@@ACT_FPRIME_PRODUCTION_SENTINEL_RESULT@@"
WORKER_TOKEN_ENV = "ACT_FPRIME_PRODUCTION_SENTINEL_ATTEMPT_TOKEN"
REQUEST_SECONDS = 10.0
WORKER_TIMEOUT_SECONDS = 45.0


SOURCE_LOCKS: Mapping[str, str] = {
    "act/back_end/config.py": (
        "8d94960cf4686f8d9c894cc18c1483b57092faa52e794905dce9f622c8fddf21"
    ),
    "act/back_end/verifier.py": (
        "eb3dfc8611ee97262bf71d66b8deee58a6a2544d0934d94ca7b075424dbc3afd"
    ),
    "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py": (
        "13625f452c36a1b7844e4385b884471c8a0c82abf015bf2af417257e2c96c23a"
    ),
    "act/back_end/hybridz_tf/forward_exact_relu_live_row_stream_candidate.py": (
        "d53c2335c43905097e78bef8311175d7151d7e98293a6152fce62dba00d37511"
    ),
    "act/back_end/hybridz_tf/operator_hz.py": (
        "2502c009d6b5d37983e3f6d072e802c7734ac9bd106bb2a252e0a562da390fd8"
    ),
    "act/back_end/hybridz_tf/phase_projection_device_program.py": (
        "7f0cce0e461f63ff6599ddd82ad5e61ef7c921eb489ef7bbbf4d60cda9048962"
    ),
    "act/back_end/hybridz_tf/phase_projection_highs_owner.py": (
        "2f5678a5b3d2b098637b27558a8bdbffcc5160ca89cb8b4947f557320d03f5b7"
    ),
    "act/back_end/hybridz_tf/phase_projection_incremental_repair.py": (
        "acc1a98fa47d36c3b0bea7d10bf93af33d41ee9de108b0151ca01cd4822f997e"
    ),
}


CSV_LOCKS: Mapping[str, str] = {
    "cifar100_2024": (
        "aa656d7a73529ba7c41b5618440f543ba4677418bb44115d384b644cc034f9ee"
    ),
    "tinyimagenet_2024": (
        "188058624df1122f32295f99d83380485a7d736212555a5e8214204459c22b7e"
    ),
}


@dataclass(frozen=True)
class Case:
    name: str
    benchmark: str
    iid: int
    model_name: str
    spec_name: str
    onnx_sha256: str
    vnnlib_sha256: str


# The expected result table is an offline compatibility oracle derived from
# ACT's own terminal-verified five-case sentinel, not an external SAT table.
# Neither the expected value nor the case name is passed to the production
# phase algorithm.
CASES: tuple[Case, ...] = (
    Case(
        "cifar100_medium_iid2",
        "cifar100_2024",
        2,
        "CIFAR100_resnet_medium.onnx",
        "CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib",
        "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
        "33e795c8421b7b19125f32415adb9cee09b2f90cb83152c4cd3aa03810e91ec3",
    ),
    Case(
        "cifar100_large_iid153",
        "cifar100_2024",
        153,
        "CIFAR100_resnet_large.onnx",
        "CIFAR100_resnet_large_prop_idx_4652_sidx_1371_eps_0.0039.vnnlib",
        "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16",
        "5b425e64c837085d070e219e8ac0e29012b30f2ef9f8e3af1ec7f5e00bc8e507",
    ),
    Case(
        "tinyimagenet_medium_iid143",
        "tinyimagenet_2024",
        143,
        "TinyImageNet_resnet_medium.onnx",
        "TinyImageNet_resnet_medium_prop_idx_3553_sidx_3392_eps_0.0039.vnnlib",
        "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
        "812bec2c0362d92d123380df161e1da6d5addbc84a27304d0a079090e814f5c7",
    ),
    Case(
        "cifar100_large_iid166",
        "cifar100_2024",
        166,
        "CIFAR100_resnet_large.onnx",
        "CIFAR100_resnet_large_prop_idx_2630_sidx_1753_eps_0.0039.vnnlib",
        "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16",
        "bdbd4493fbcc15ee7518afc86491eb183da4725a0525ad4af82cebc51b121b8c",
    ),
    Case(
        "tinyimagenet_medium_iid153",
        "tinyimagenet_2024",
        153,
        "TinyImageNet_resnet_medium.onnx",
        "TinyImageNet_resnet_medium_prop_idx_2493_sidx_4209_eps_0.0039.vnnlib",
        "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
        "f9c50760f284590d36366ffff8c9d4f628e554c11bc78d7e5d443b473dda8a17",
    ),
)


EXPECTED_STATUS: Mapping[str, str] = {
    "cifar100_medium_iid2": "FALSIFIED",
    "cifar100_large_iid153": "UNKNOWN",
    "tinyimagenet_medium_iid143": "FALSIFIED",
    "cifar100_large_iid166": "UNKNOWN",
    "tinyimagenet_medium_iid153": "FALSIFIED",
}


class SentinelError(RuntimeError):
    """The offline gate cannot safely interpret or continue an attempt."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _source_bundle() -> dict[str, str]:
    observed = {relative: _sha256(ROOT / relative) for relative in SOURCE_LOCKS}
    if observed != dict(SOURCE_LOCKS):
        changed = sorted(
            relative
            for relative, expected in SOURCE_LOCKS.items()
            if observed.get(relative) != expected
        )
        raise SentinelError(
            "production source bundle differs from the frozen sentinel lock: "
            + ",".join(changed)
        )
    return observed


def _bundle_sha256(bundle: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical_json(dict(bundle)).encode("utf-8")).hexdigest()


def _resolve(case: Case) -> tuple[Path, Path]:
    root = BENCHMARK_ROOT / case.benchmark
    csv_path = root / "instances.csv"
    if _sha256(csv_path) != CSV_LOCKS[case.benchmark]:
        raise SentinelError(f"{case.benchmark} instances.csv changed")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if case.iid >= len(rows) or len(rows[case.iid]) != 3:
        raise SentinelError(f"{case.name} instances.csv row is malformed")
    model_rel, spec_rel, _timeout = rows[case.iid]
    model = (root / model_rel).resolve()
    spec = (root / spec_rel).resolve()
    if model.name != case.model_name or spec.name != case.spec_name:
        raise SentinelError(f"{case.name} manifest identity changed")
    if not model.is_file() or not spec.is_file():
        raise SentinelError(f"{case.name} input file is unavailable")
    if _sha256(model) != case.onnx_sha256 or _sha256(spec) != case.vnnlib_sha256:
        raise SentinelError(f"{case.name} input content changed")
    return model, spec


def _identity(bundle: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": "act.hybridz.fprime.production_five_case_sentinel.identity.v1",
        "case_order": [case.name for case in CASES],
        "expected_status": dict(EXPECTED_STATUS),
        "request_seconds": REQUEST_SECONDS,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "worker_timeout_role": "transport_hard_limit_not_request_or_lp_deadline",
        "source_bundle": dict(bundle),
        "source_bundle_sha256": _bundle_sha256(bundle),
        "harness_sha256": _sha256(Path(__file__)),
        "csv_sha256": dict(CSV_LOCKS),
        "input_sha256": {
            case.name: {
                "onnx": case.onnx_sha256,
                "vnnlib": case.vnnlib_sha256,
            }
            for case in CASES
        },
        "production_entry": "act.back_end.verifier.verify_once",
        "runtime_paths": 1,
        "phase_updates_max": 1,
        "resolves_after_base_max": 1,
        "retries": 0,
        "external_labels_read_by_production": False,
        "randomness_scope": "ipc_attempt_capability_only_never_production_or_selector",
    }


def _restriction_fields() -> tuple[str, ...]:
    return (
        "input_sampling_used",
        "pgd_used",
        "concrete_onnx_execution_used",
        "bab_used",
        "backward_used",
        "dual_tightening_used",
    )


def _validate_owner_audit(audit: Mapping[str, Any], *, positive: bool) -> None:
    integer_fields = (
        "logical_owner_instances",
        "logical_owner_close_calls",
        "native_owner_instances",
        "native_run_calls",
        "native_clear_calls",
        "native_clear_model_calls",
        "dual_ray_exist_calls",
        "dual_ray_calls",
    )
    if any(type(audit.get(name)) is not int for name in integer_fields):
        raise SentinelError("owner audit has a malformed counter")
    logical = int(audit["logical_owner_instances"])
    native = int(audit["native_owner_instances"])
    if logical not in {0, 1} or native not in {0, 1} or native > logical:
        raise SentinelError("more than one production owner was observed")
    if int(audit["native_run_calls"]) not in {0, 1, 2}:
        raise SentinelError("production owner exceeded two solves")
    if int(audit["dual_ray_exist_calls"]) not in {0, 1} or int(
        audit["dual_ray_calls"]
    ) not in {0, 1}:
        raise SentinelError("production owner exceeded one dual-ray read")
    if audit["dual_ray_exist_calls"] != audit["dual_ray_calls"]:
        raise SentinelError("dual-ray existence/read counts disagree")
    if int(audit["native_clear_model_calls"]) != 0:
        raise SentinelError("clearModel/reload was observed")
    final_states = audit.get("logical_owner_final_states")
    if type(final_states) is not list or any(value != "CLOSED" for value in final_states):
        raise SentinelError("a logical owner did not finish CLOSED")
    if logical == 0:
        if final_states != [] or any(
            int(audit[name]) != 0 for name in integer_fields[1:]
        ):
            raise SentinelError("owner activity exists without a logical owner")
    else:
        if int(audit["logical_owner_close_calls"]) != 1 or final_states != ["CLOSED"]:
            raise SentinelError("logical owner cleanup count is not exactly one")
        if native == 1 and int(audit["native_clear_calls"]) != 1:
            raise SentinelError("native owner cleanup count is not exactly one")
        if native == 0 and any(
            int(audit[name]) != 0
            for name in (
                "native_run_calls",
                "native_clear_calls",
                "native_clear_model_calls",
                "dual_ray_exist_calls",
                "dual_ray_calls",
            )
        ):
            raise SentinelError("native activity exists without a native owner")
    if int(audit["dual_ray_calls"]) > int(audit["native_run_calls"]):
        raise SentinelError("dual-ray read exists without a completed solve")
    if positive and (logical != 1 or native != 1):
        raise SentinelError("a formal positive did not use exactly one owner")


def _validate_positive_receipt(
    receipt: Mapping[str, Any], audit: Mapping[str, Any]
) -> None:
    if receipt.get("schema") != "act.hybridz.forward_exact_relu_phase_projection_candidate.v3":
        raise SentinelError("formal positive has the wrong candidate schema")
    if receipt.get("status") != "singleton_verified":
        raise SentinelError("formal positive lacks the terminal candidate status")
    if receipt.get("singleton_interval_verified") is not True:
        raise SentinelError("zero-width outward terminal was not verified")
    margin = receipt.get("singleton_margin_lower")
    if type(margin) is not float or not math.isfinite(margin) or margin <= 0.0:
        raise SentinelError("stored-binary64 Fraction terminal margin is not positive")
    false_fields = (
        "candidate_authority",
        "proof_authority",
        "verdict_authority",
        "input_sampling_used",
        "pgd_used",
        "concrete_onnx_execution_used",
        "bab_used",
        "backward_used",
        "dual_tightening_used",
        "dual_ray_authority",
        "dual_selector_authority",
        "second_solver_used",
        "runtime_menu_used",
        "activation_split_used",
        "input_split_used",
        "enumeration_used",
        "cross_request_cache_used",
        "updated_full_target_materialized",
    )
    if any(receipt.get(name) is not False for name in false_fields):
        raise SentinelError("candidate receipt crosses an authority/prohibition boundary")
    integer_fields = (
        "fallbacks",
        "retries",
        "phase_updates",
        "phase_retries",
        "property_rows_selected",
        "property_row_retries",
        "triangle_rows",
        "generator_streams",
        "fixed_cell_generator_streams",
        "owner_instances",
        "repair_updates",
        "owner_solves",
        "resolves_after_base",
        "phase_delta_streams",
        "repair_selected_rows",
        "dual_ray_requests",
    )
    if any(type(receipt.get(name)) is not int for name in integer_fields):
        raise SentinelError("candidate receipt has a non-integer count")
    if receipt.get("fallbacks") != 0 or receipt.get("retries") != 0:
        raise SentinelError("candidate receipt reports a fallback or retry")
    if not (
        receipt.get("phase_updates") == 1
        and receipt.get("phase_retries") == 0
        and receipt.get("property_rows_selected") == 1
        and receipt.get("property_row_retries") == 0
        and receipt.get("all_unstable_exact") is True
        and receipt.get("triangle_rows") == 0
        and receipt.get("generator_streams") == 1
        and receipt.get("fixed_cell_generator_streams") == 1
        and receipt.get("generator_representation")
        == "request_local_device_program_incremental_lowrank_v1"
        and receipt.get("candidate_outward_error_bands_used") is False
        and receipt.get("intermediate_phase_or_margin_replay_used") is False
    ):
        raise SentinelError("candidate receipt violates the frozen single-path shape")
    if receipt.get("owner_instances") != 1:
        raise SentinelError("candidate receipt does not report one owner")
    repair = receipt.get("repair_updates")
    if repair not in {0, 1}:
        raise SentinelError("candidate receipt repair count is malformed")
    if receipt.get("owner_solves") != 1 + repair:
        raise SentinelError("candidate receipt owner solve count is malformed")
    if receipt.get("resolves_after_base") != repair:
        raise SentinelError("candidate receipt resolve count is malformed")
    if receipt.get("same_owner_warm_update_used") is not bool(repair):
        raise SentinelError("candidate receipt warm-update flag is malformed")
    if receipt.get("phase_delta_streams") != 1 + repair:
        raise SentinelError("candidate receipt phase-delta stream count is malformed")
    if receipt.get("same_stored_binary64_input_for_box_and_terminal") is not True:
        raise SentinelError("BOX and terminal did not consume the same sealed input")
    if int(audit["native_run_calls"]) != int(receipt["owner_solves"]):
        raise SentinelError("native and receipt solve counts disagree")
    if int(audit["dual_ray_calls"]) != int(receipt.get("dual_ray_requests", -1)):
        raise SentinelError("native and receipt dual-ray counts disagree")
    digest = receipt.get("repair_selected_row_ids_sha256")
    if type(digest) is not str or len(digest) != 64:
        raise SentinelError("repair selection digest is not 64 hexadecimal digits")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise SentinelError("repair selection digest is not hexadecimal") from exc
    if repair == 0:
        if (
            receipt.get("repair_selected_rows") != 0
            or receipt.get("repair_selector_rule") != "base_positive_none"
            or receipt.get("dual_selector_used") is not False
            or receipt.get("dual_ray_requests") != 0
        ):
            raise SentinelError("base-positive receipt reports a repair")
    else:
        if (
            type(receipt.get("repair_selected_rows")) is not int
            or receipt["repair_selected_rows"] <= 0
            or receipt.get("dual_selector_used") is not True
        ):
            raise SentinelError("repair receipt lacks a sealed nonempty selection")
        selector = receipt.get("repair_selector_rule")
        allowed = {
            "optimal_negative_all_tight_strict_negative_upper_row_dual": (
                "OPTIMAL",
                0,
            ),
            "infeasible_all_exact_nonzero_validated_dual_ray_phase_rows": (
                "INFEASIBLE",
                1,
            ),
        }
        if selector not in allowed:
            raise SentinelError("repair selector is outside the frozen whitelist")
        base_status, ray_requests = allowed[selector]
        if (
            receipt.get("base_model_status") != base_status
            or receipt.get("dual_ray_requests") != ray_requests
        ):
            raise SentinelError("repair selector disagrees with its base status")


def _validated_status(record: Mapping[str, Any]) -> str:
    projection = record.get("phase_projection")
    audit = record.get("owner_audit")
    if type(projection) is not dict or type(audit) is not dict:
        raise SentinelError("worker record lacks projection/owner metadata")
    if projection.get("enabled") is not True or projection.get("configured_seconds") != 10.0:
        raise SentinelError("formal production phase was not enabled for ten seconds")
    if any(projection.get(name) is not False for name in _restriction_fields()):
        raise SentinelError("formal verifier reports a prohibited method")
    status = record.get("status")
    if status == "VerifyStatus.FALSIFIED":
        _validate_owner_audit(audit, positive=True)
        receipt = projection.get("candidate_receipt")
        if type(receipt) is not dict:
            raise SentinelError("formal positive lacks a candidate receipt")
        _validate_positive_receipt(receipt, audit)
        if not (
            record.get("has_counterexample") is True
            and projection.get("status") == "FALSIFIED"
            and projection.get("verifier_owned_proof_authority") is True
            and projection.get("proof_rule")
            == "decoded_input_in_raw_BOX;verifier_owned_zero_width_forward_interval;"
            "exact_Fraction_property_lower_bound_positive"
        ):
            raise SentinelError("formal positive lacks verifier-owned authority")
        return "FALSIFIED"
    if status == "VerifyStatus.UNKNOWN":
        _validate_owner_audit(audit, positive=False)
        if not (
            record.get("has_counterexample") is False
            and projection.get("status") == "UNKNOWN"
            and projection.get("verifier_owned_proof_authority") is False
            and "candidate_receipt" not in projection
            and type(projection.get("reason")) is str
            and bool(projection["reason"])
        ):
            raise SentinelError("formal UNKNOWN crossed the fail-closed boundary")
        return "UNKNOWN"
    raise SentinelError("formal worker returned an unsupported status")


def _revalidate_record(
    record: Mapping[str, Any],
    *,
    case: Case,
    identity: Mapping[str, Any],
    token_sha256: str,
) -> str:
    if record.get("schema") != (
        "act.hybridz.fprime.production_five_case_sentinel.worker.v1"
    ):
        raise SentinelError("worker record schema is not frozen")
    if record.get("case") != case.name:
        raise SentinelError("worker record case identity changed")
    if (
        type(token_sha256) is not str
        or len(token_sha256) != 64
        or record.get("attempt_token_sha256") != token_sha256
    ):
        raise SentinelError("worker record is not bound to its one-shot attempt")
    try:
        int(token_sha256, 16)
    except ValueError as exc:
        raise SentinelError("attempt token digest is not hexadecimal") from exc
    claimed = record.get("validated_status")
    if claimed in {"FALSIFIED", "UNKNOWN"}:
        observed = _validated_status(record)
        if observed != claimed:
            raise SentinelError("worker and parent validation disagree")
        if record.get("source_bundle_sha256") != identity[
            "source_bundle_sha256"
        ]:
            raise SentinelError("worker source identity differs from parent")
        if record.get("input_sha256") != identity["input_sha256"][case.name]:
            raise SentinelError("worker input identity differs from parent")
        return observed
    if claimed == "ERROR" and record.get("has_counterexample") is False:
        return "ERROR"
    raise SentinelError("worker record has an invalid claimed status")


def _authorize_worker(case_name: str, token: str) -> str:
    if type(token) is not str or len(token) < 32:
        raise SentinelError("worker lacks a parent attempt capability")
    token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
    bundle = _source_bundle()
    identity = _identity(bundle)
    events = _read_events(identity)
    if not events:
        raise SentinelError("worker has no persisted parent ledger")
    started, results = _event_state(events, identity)
    if case_name not in started or case_name in results:
        raise SentinelError("worker attempt is absent or already consumed")
    last = events[-1]
    if not (
        last.get("event") == "case_attempt_started"
        and last.get("case") == case_name
        and last.get("attempt_token_sha256") == token_sha256
    ):
        raise SentinelError("worker capability is not bound to the active case")
    return token_sha256


class _OwnerInstrumentation:
    """Transparent counters around the real request-local production owner."""

    def __init__(self, owner_module: Any) -> None:
        self.owner_module = owner_module
        self.real_owner = owner_module.SafeHighsOwner
        self.real_highs = owner_module.highspy.Highs
        self.audit: dict[str, Any] = {
            "logical_owner_instances": 0,
            "logical_owner_close_calls": 0,
            "logical_owner_final_states": [],
            "native_owner_instances": 0,
            "native_run_calls": 0,
            "native_clear_calls": 0,
            "native_clear_model_calls": 0,
            "dual_ray_exist_calls": 0,
            "dual_ray_calls": 0,
        }

    def __enter__(self) -> "_OwnerInstrumentation":
        audit = self.audit
        real_owner = self.real_owner
        real_highs = self.real_highs

        class TrackingOwner(real_owner):
            def __init__(tracked_self: Any, *args: Any, **kwargs: Any) -> None:
                audit["logical_owner_instances"] += 1
                super().__init__(*args, **kwargs)

            def close(tracked_self: Any) -> None:
                audit["logical_owner_close_calls"] += 1
                try:
                    super().close()
                finally:
                    audit["logical_owner_final_states"].append(tracked_self.state)

        class TrackingNative:
            def __init__(tracked_self: Any) -> None:
                audit["native_owner_instances"] += 1
                tracked_self._backend = real_highs()

            def __getattr__(tracked_self: Any, name: str) -> Any:
                return getattr(tracked_self._backend, name)

            def run(tracked_self: Any) -> Any:
                audit["native_run_calls"] += 1
                return tracked_self._backend.run()

            def clear(tracked_self: Any) -> Any:
                audit["native_clear_calls"] += 1
                return tracked_self._backend.clear()

            def clearModel(tracked_self: Any) -> Any:
                audit["native_clear_model_calls"] += 1
                return tracked_self._backend.clearModel()

            def getDualRayExist(tracked_self: Any) -> Any:
                audit["dual_ray_exist_calls"] += 1
                return tracked_self._backend.getDualRayExist()

            def getDualRay(tracked_self: Any) -> Any:
                audit["dual_ray_calls"] += 1
                return tracked_self._backend.getDualRay()

        self.owner_module.SafeHighsOwner = TrackingOwner
        self.owner_module.highspy.Highs = TrackingNative
        return self

    def __exit__(self, _kind: Any, _value: Any, _tb: Any) -> bool:
        self.owner_module.SafeHighsOwner = self.real_owner
        self.owner_module.highspy.Highs = self.real_highs
        return False


def _run_production_case(case: Case) -> dict[str, Any]:
    import torch

    from act.back_end.config import BackendConfig, HybridZConfig
    from act.back_end.hybridz_tf import phase_projection_highs_owner
    from act.back_end.transfer_functions import (
        set_solver_mode,
        set_transfer_function_mode,
    )
    from act.back_end.verifier import verify_once
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
    from act.pipeline.verification.torch2act import TorchToACT
    from act.util.device_manager import initialize_device

    bundle_before = _source_bundle()
    onnx, vnnlib = _resolve(case)
    input_before = {"onnx": _sha256(onnx), "vnnlib": _sha256(vnnlib)}
    started = time.monotonic()
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    specs = create_specs_from_paths(str(onnx), str(vnnlib), category=case.benchmark)
    wrapped = synthesize_models_from_specs([specs])
    if len(wrapped) != 1:
        raise SentinelError("production worker requires exactly one wrapped model")
    model = next(iter(wrapped.values())).to(
        device=torch.device("cuda"), dtype=torch.float64
    )
    net = TorchToACT(model).run()
    instrumentation = _OwnerInstrumentation(phase_projection_highs_owner)
    with instrumentation:
        results = verify_once(
            net,
            backend_cfg=BackendConfig(
                solver="hybridz",
                device="cuda",
                dtype="float64",
                timeout=30.0,
                bab_enabled=False,
                lp_enabled=False,
                hybridz=HybridZConfig(
                    timeout=20.0,
                    engine="operator_hz_objbound",
                    operator_exact_budget=-1,
                    operator_phase_projection_time_limit=REQUEST_SECONDS,
                    operator_materialize_add=True,
                ),
            ),
        )
    torch.cuda.synchronize()
    if len(results) != 1:
        raise SentinelError("production worker returned more than one lane")
    result = results[0]
    input_after = {"onnx": _sha256(onnx), "vnnlib": _sha256(vnnlib)}
    bundle_after = _source_bundle()
    if input_after != input_before or bundle_after != bundle_before:
        raise SentinelError("an input or production source changed during the request")
    projection = result.metadata.get("operator_phase_projection", {})
    record: dict[str, Any] = {
        "schema": "act.hybridz.fprime.production_five_case_sentinel.worker.v1",
        "case": case.name,
        "benchmark": case.benchmark,
        "iid": case.iid,
        "onnx": str(onnx),
        "vnnlib": str(vnnlib),
        "input_sha256": input_before,
        "source_bundle_sha256": _bundle_sha256(bundle_before),
        "status": str(result.status),
        "has_counterexample": result.counterexample is not None,
        "phase_projection": projection,
        "owner_audit": instrumentation.audit,
        "elapsed_seconds": time.monotonic() - started,
        "scope": {
            "production_verify_once": True,
            "disconnected_algorithm_reused": False,
            "input_sampling_used": False,
            "onnx_input_point_execution_used": False,
            "pgd_used": False,
            "bab_split_or_enumeration_used": False,
            "backward_bounds_used": False,
            "dual_tightening_used": False,
            "runtime_fallback_or_menu_used": False,
            "external_label_read_by_production": False,
            "attempts_for_case": 1,
        },
    }
    record["validated_status"] = _validated_status(record)
    return record


def _worker_entry(case_name: str) -> int:
    by_name = {case.name: case for case in CASES}
    if case_name not in by_name:
        raise SentinelError("worker case is outside the frozen manifest")
    token = os.environ.pop(WORKER_TOKEN_ENV, "")
    token_sha256 = (
        hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""
    )
    try:
        token_sha256 = _authorize_worker(case_name, token)
        # Keep every Python-level library diagnostic off the result channel.
        with contextlib.redirect_stdout(sys.stderr):
            record = _run_production_case(by_name[case_name])
    except Exception as exc:
        record = {
            "schema": "act.hybridz.fprime.production_five_case_sentinel.worker.v1",
            "case": case_name,
            "status": "worker_error",
            "has_counterexample": False,
            "validated_status": "ERROR",
            "error_type": type(exc).__name__,
        }
    record["attempt_token_sha256"] = token_sha256
    print(RESULT_PREFIX + _canonical_json(record), flush=True)
    return 0 if record.get("validated_status") in {"FALSIFIED", "UNKNOWN"} else 2


def _append_event(value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "a"
    with EVENTS_PATH.open(mode, encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(dict(value)) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_events(identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not EVENTS_PATH.exists():
        return []
    values = []
    with EVENTS_PATH.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if type(value) is not dict:
                raise SentinelError(f"event line {number} is not an object")
            values.append(value)
    if not values or values[0] != {"event": "run_created", "identity": dict(identity)}:
        raise SentinelError("existing event ledger has a different frozen identity")
    return values


def _event_state(
    events: list[dict[str, Any]], identity: Mapping[str, Any]
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    started: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    by_name = {case.name: case for case in CASES}
    order = {case.name: index for index, case in enumerate(CASES)}
    attempt_tokens: dict[str, str] = {}
    for event in events[1:]:
        kind = event.get("event")
        case = event.get("case")
        if case not in order:
            raise SentinelError("event ledger contains a case outside the manifest")
        if kind == "case_attempt_started":
            if case in started:
                raise SentinelError("event ledger contains a retry")
            if len(started) != order[case]:
                raise SentinelError("event ledger violates the frozen case order")
            if len(results) != len(started):
                raise SentinelError("event ledger continued after an incomplete attempt")
            if started:
                prior = CASES[len(started) - 1].name
                if results[prior].get("validated_status") != EXPECTED_STATUS[prior]:
                    raise SentinelError("event ledger continued after a core blocker")
            token_sha256 = event.get("attempt_token_sha256")
            if (
                event.get("ordinal") != order[case]
                or event.get("source_bundle_sha256")
                != identity["source_bundle_sha256"]
                or type(token_sha256) is not str
                or len(token_sha256) != 64
            ):
                raise SentinelError("attempt event lacks a token digest")
            try:
                int(token_sha256, 16)
            except ValueError as exc:
                raise SentinelError("attempt token digest is not hexadecimal") from exc
            started.add(case)
            attempt_tokens[case] = token_sha256
        elif kind == "case_result":
            if case not in started or case in results:
                raise SentinelError("event ledger has an orphan/duplicate result")
            record = event.get("record")
            if type(record) is not dict or record.get("case") != case:
                raise SentinelError("event ledger result identity is malformed")
            _revalidate_record(
                record,
                case=by_name[case],
                identity=identity,
                token_sha256=attempt_tokens[case],
            )
            results[case] = record
        else:
            raise SentinelError("event ledger contains an unknown event")
    return started, results


def _summary(
    identity: Mapping[str, Any],
    started: set[str],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    incomplete = [name for name in started if name not in results]
    errors = sum(record.get("validated_status") == "ERROR" for record in results.values())
    mismatches = [
        name
        for name, record in results.items()
        if record.get("validated_status") != EXPECTED_STATUS[name]
    ]
    complete = len(results) == len(CASES)
    if incomplete:
        status = "BLOCKED_INCOMPLETE_ATTEMPT_NO_RETRY"
    elif errors:
        status = "FAILED_CLOSED_ERROR"
    elif mismatches:
        status = "STOP_LOSS_COMPATIBILITY_MISMATCH"
    elif complete and not mismatches:
        status = "COMPLETE_COMPATIBLE"
    else:
        status = "IN_PROGRESS"
    return {
        "schema": "act.hybridz.fprime.production_five_case_sentinel.v1",
        "status": status,
        "identity": dict(identity),
        "attempted": len(started),
        "completed": len(results),
        "errors": errors,
        "compatibility_mismatches": mismatches,
        "incomplete_attempts": sorted(incomplete),
        "results": [results[case.name] for case in CASES if case.name in results],
        "events_path": str(EVENTS_PATH.relative_to(ROOT)),
        "scope": {
            "production_verify_once": True,
            "five_cases_once_in_frozen_order": True,
            "disconnected_algorithm_reused": False,
            "timing_authority": False,
            "formal_fixed400_changed": False,
            "input_sampling_used": False,
            "onnx_input_point_execution_used": False,
            "pgd_used": False,
            "bab_split_or_enumeration_used": False,
            "backward_bounds_used": False,
            "dual_tightening_used": False,
            "runtime_fallback_or_menu_used": False,
            "external_labels_read_by_production": False,
        },
    }


def _parent() -> int:
    bundle = _source_bundle()
    for case in CASES:
        _resolve(case)
    identity = _identity(bundle)
    events = _read_events(identity)
    if not events:
        created = {"event": "run_created", "identity": identity}
        _append_event(created, exclusive=True)
        events = [created]
    started, results = _event_state(events, identity)
    if any(case in started and case not in results for case in started):
        _atomic_json(SUMMARY_PATH, _summary(identity, started, results))
        return 2
    if any(
        record.get("validated_status") != EXPECTED_STATUS[name]
        for name, record in results.items()
    ):
        _atomic_json(SUMMARY_PATH, _summary(identity, started, results))
        return 2
    for case in CASES:
        if case.name in results:
            continue
        token = secrets.token_urlsafe(32)
        token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
        attempt = {
            "event": "case_attempt_started",
            "case": case.name,
            "ordinal": len(started),
            "source_bundle_sha256": identity["source_bundle_sha256"],
            "attempt_token_sha256": token_sha256,
        }
        _append_event(attempt)
        events.append(attempt)
        started.add(case.name)
        env = dict(os.environ)
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
                WORKER_TOKEN_ENV: token,
            }
        )
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-case", case.name]
        child_started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            record = {
                "schema": "act.hybridz.fprime.production_five_case_sentinel.worker.v1",
                "case": case.name,
                "status": "worker_timeout",
                "has_counterexample": False,
                "validated_status": "ERROR",
                "error_type": type(exc).__name__,
            }
        else:
            stdout, stderr = completed.stdout, completed.stderr
            marked = [
                line[len(RESULT_PREFIX) :]
                for line in stdout.splitlines()
                if line.startswith(RESULT_PREFIX)
            ]
            if len(marked) == 1:
                try:
                    record = json.loads(marked[0])
                except json.JSONDecodeError:
                    record = {}
            else:
                record = {}
            if (
                type(record) is not dict
                or record.get("case") != case.name
                or completed.returncode not in {0, 2}
            ):
                record = {
                    "schema": "act.hybridz.fprime.production_five_case_sentinel.worker.v1",
                    "case": case.name,
                    "status": "worker_transport_error",
                    "has_counterexample": False,
                    "validated_status": "ERROR",
                    "error_type": "MalformedIsolatedResult",
                }
            else:
                try:
                    _revalidate_record(
                        record,
                        case=case,
                        identity=identity,
                        token_sha256=token_sha256,
                    )
                except Exception as exc:
                    record = {
                        "schema": (
                            "act.hybridz.fprime.production_five_case_sentinel."
                            "worker.v1"
                        ),
                        "case": case.name,
                        "status": "worker_parent_revalidation_error",
                        "has_counterexample": False,
                        "validated_status": "ERROR",
                        "error_type": type(exc).__name__,
                        "attempt_token_sha256": token_sha256,
                    }
        record.setdefault("attempt_token_sha256", token_sha256)
        record["transport"] = {
            "child_wall_seconds": time.monotonic() - child_started,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
            "isolated_result_records": sum(
                line.startswith(RESULT_PREFIX) for line in stdout.splitlines()
            ),
        }
        result_event = {"event": "case_result", "case": case.name, "record": record}
        _append_event(result_event)
        events.append(result_event)
        results[case.name] = record
        _atomic_json(SUMMARY_PATH, _summary(identity, started, results))
        if record.get("validated_status") != EXPECTED_STATUS[case.name]:
            break
    summary = _summary(identity, started, results)
    _atomic_json(SUMMARY_PATH, summary)
    print(_canonical_json(summary), flush=True)
    return 0 if summary["status"] == "COMPLETE_COMPATIBLE" else 2


def _static_check() -> dict[str, Any]:
    bundle = _source_bundle()
    for case in CASES:
        _resolve(case)
    if tuple(case.name for case in CASES) != tuple(EXPECTED_STATUS):
        raise SentinelError("expected status table is not in frozen case order")
    if len(CASES) != 5 or len({case.name for case in CASES}) != 5:
        raise SentinelError("control manifest is not five unique cases")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    if any(name.startswith("scratch_phase_projection") for name in imported):
        raise SentinelError("sentinel imports a disconnected phase probe")
    production_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "verify_once"
    ]
    if len(production_calls) != 1:
        raise SentinelError("worker must contain exactly one verify_once callsite")
    phase_tree = ast.parse(
        (ROOT / "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py")
        .read_text(encoding="utf-8")
    )
    phase_imports = {
        node.module or ""
        for node in ast.walk(phase_tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(phase_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    if "scipy.optimize" in phase_imports or "onnxruntime" in phase_imports:
        raise SentinelError("production phase imports a forbidden second/search backend")

    no_owner = {
        "logical_owner_instances": 0,
        "logical_owner_close_calls": 0,
        "logical_owner_final_states": [],
        "native_owner_instances": 0,
        "native_run_calls": 0,
        "native_clear_calls": 0,
        "native_clear_model_calls": 0,
        "dual_ray_exist_calls": 0,
        "dual_ray_calls": 0,
    }
    one_owner = {
        "logical_owner_instances": 1,
        "logical_owner_close_calls": 1,
        "logical_owner_final_states": ["CLOSED"],
        "native_owner_instances": 1,
        "native_run_calls": 2,
        "native_clear_calls": 1,
        "native_clear_model_calls": 0,
        "dual_ray_exist_calls": 1,
        "dual_ray_calls": 1,
    }
    _validate_owner_audit(no_owner, positive=False)
    _validate_owner_audit(one_owner, positive=True)
    hostile = dict(one_owner)
    hostile["native_owner_instances"] = 2
    try:
        _validate_owner_audit(hostile, positive=True)
    except SentinelError:
        pass
    else:
        raise SentinelError("owner-count validator accepted a second owner")

    impossible_native = dict(one_owner)
    impossible_native.update(
        {
            "native_owner_instances": 0,
            "native_clear_calls": 0,
        }
    )
    try:
        _validate_owner_audit(impossible_native, positive=False)
    except SentinelError:
        pass
    else:
        raise SentinelError("owner validator accepted activity without a native owner")
    impossible_state = dict(no_owner)
    impossible_state["logical_owner_final_states"] = ["CLOSED"]
    try:
        _validate_owner_audit(impossible_state, positive=False)
    except SentinelError:
        pass
    else:
        raise SentinelError("owner validator accepted a state without an owner")

    base_audit = dict(one_owner)
    base_audit.update(
        {
            "native_run_calls": 1,
            "dual_ray_exist_calls": 0,
            "dual_ray_calls": 0,
        }
    )
    positive_receipt: dict[str, Any] = {
        "schema": "act.hybridz.forward_exact_relu_phase_projection_candidate.v3",
        "status": "singleton_verified",
        "singleton_interval_verified": True,
        "singleton_margin_lower": 0.25,
        "candidate_authority": False,
        "proof_authority": False,
        "verdict_authority": False,
        "input_sampling_used": False,
        "pgd_used": False,
        "concrete_onnx_execution_used": False,
        "bab_used": False,
        "backward_used": False,
        "dual_tightening_used": False,
        "dual_ray_authority": False,
        "dual_selector_authority": False,
        "second_solver_used": False,
        "runtime_menu_used": False,
        "activation_split_used": False,
        "input_split_used": False,
        "enumeration_used": False,
        "cross_request_cache_used": False,
        "updated_full_target_materialized": False,
        "fallbacks": 0,
        "retries": 0,
        "phase_updates": 1,
        "phase_retries": 0,
        "property_rows_selected": 1,
        "property_row_retries": 0,
        "all_unstable_exact": True,
        "triangle_rows": 0,
        "generator_streams": 1,
        "fixed_cell_generator_streams": 1,
        "generator_representation": "request_local_device_program_incremental_lowrank_v1",
        "candidate_outward_error_bands_used": False,
        "intermediate_phase_or_margin_replay_used": False,
        "owner_instances": 1,
        "repair_updates": 0,
        "owner_solves": 1,
        "resolves_after_base": 0,
        "same_owner_warm_update_used": False,
        "phase_delta_streams": 1,
        "same_stored_binary64_input_for_box_and_terminal": True,
        "repair_selected_row_ids_sha256": "a" * 64,
        "repair_selected_rows": 0,
        "repair_selector_rule": "base_positive_none",
        "dual_selector_used": False,
        "dual_ray_requests": 0,
    }
    _validate_positive_receipt(positive_receipt, base_audit)
    hostile_receipt = dict(positive_receipt)
    hostile_receipt.update(
        {
            "repair_updates": 1,
            "owner_solves": 2,
            "resolves_after_base": 1,
            "same_owner_warm_update_used": True,
            "phase_delta_streams": 2,
            "repair_selected_rows": 1,
            "repair_selector_rule": "per_instance_selector",
            "dual_selector_used": True,
        }
    )
    try:
        _validate_positive_receipt(hostile_receipt, one_owner)
    except SentinelError:
        pass
    else:
        raise SentinelError("receipt validator accepted a selector outside the whitelist")
    hostile_receipt = dict(positive_receipt)
    hostile_receipt["phase_retries"] = 9
    try:
        _validate_positive_receipt(hostile_receipt, base_audit)
    except SentinelError:
        pass
    else:
        raise SentinelError("receipt validator accepted phase retries")
    hostile_receipt = dict(positive_receipt)
    hostile_receipt["repair_updates"] = False
    try:
        _validate_positive_receipt(hostile_receipt, base_audit)
    except SentinelError:
        pass
    else:
        raise SentinelError("receipt validator accepted bool as an integer count")

    identity = _identity(bundle)
    token_sha256 = "b" * 64
    header = {"event": "run_created", "identity": identity}
    attempt0 = {
        "event": "case_attempt_started",
        "case": CASES[0].name,
        "ordinal": 0,
        "source_bundle_sha256": identity["source_bundle_sha256"],
        "attempt_token_sha256": token_sha256,
    }
    forged = {
        "event": "case_result",
        "case": CASES[0].name,
        "record": {
            "case": CASES[0].name,
            "validated_status": EXPECTED_STATUS[CASES[0].name],
        },
    }
    try:
        _event_state([header, attempt0, forged], identity)
    except SentinelError:
        pass
    else:
        raise SentinelError("resume validator accepted an unauthoritative forged result")
    error_record = {
        "schema": "act.hybridz.fprime.production_five_case_sentinel.worker.v1",
        "case": CASES[0].name,
        "status": "worker_error",
        "has_counterexample": False,
        "validated_status": "ERROR",
        "attempt_token_sha256": token_sha256,
    }
    result0 = {"event": "case_result", "case": CASES[0].name, "record": error_record}
    attempt1 = {
        "event": "case_attempt_started",
        "case": CASES[1].name,
        "ordinal": 1,
        "source_bundle_sha256": identity["source_bundle_sha256"],
        "attempt_token_sha256": "c" * 64,
    }
    try:
        _event_state([header, attempt0, result0, attempt1], identity)
    except SentinelError:
        pass
    else:
        raise SentinelError("resume validator continued after a core blocker")

    authorization_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_authorize_worker"
    ]
    if len(authorization_calls) != 1:
        raise SentinelError("hidden worker lacks its one-shot parent authorization")
    return {
        "schema": "act.hybridz.fprime.production_five_case_sentinel.static.v1",
        "status": "STATIC_CPU_ONLY_PASS",
        "harness_sha256": _sha256(Path(__file__)),
        "source_bundle": bundle,
        "source_bundle_sha256": _bundle_sha256(bundle),
        "case_order": [case.name for case in CASES],
        "verify_once_callsites": 1,
        "owner_validator_selftests": 6,
        "receipt_validator_selftests": 4,
        "resume_validator_selftests": 2,
        "worker_authorization_callsites": 1,
        "disconnected_probe_imports": 0,
        "gpu_initialized": False,
        "artifacts_created": False,
        "events_path_exists": EVENTS_PATH.exists(),
        "summary_path_exists": SUMMARY_PATH.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument("--worker-case", choices=tuple(case.name for case in CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.static_check:
        print(_canonical_json(_static_check()), flush=True)
        return 0
    if args.worker_case is not None:
        return _worker_entry(args.worker_case)
    lock_path = EVENTS_PATH.with_suffix(EVENTS_PATH.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SentinelError("another sentinel parent already owns the run") from exc
        return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
