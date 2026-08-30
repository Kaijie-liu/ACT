#!/usr/bin/env python3
"""Frozen F-prime paired noninferiority gate (offline harness only).

For each of five controls this harness starts one fresh process for the
single-request mode and one fresh process for the four-thread mode.  A worker
loads the formal-59 phase implementation directly from a locked Git blob into
memory, never materializing old source on disk.  It compares that old module
with the current production module through the unchanged production
``verify_once`` entry point.

Each job performs two alternating warmup pairs and five alternating measured
pairs.  The exact paired bootstrap enumerates all 5**5 resamples.  A job passes
only when its paired median old/new speedup is at least 1.0, its exact 95%
lower bound is at least 0.95, and all current/old terminal or UNKNOWN semantics
remain internally authoritative and conflict-free.

This file is a comparator, not a production dispatcher.  Expected outcomes
are offline audit strata and are never supplied to ``verify_once`` or either
candidate implementation.  There is no input sampling, point ONNX execution,
PGD, BaB/split, backward bounds, dual tightening, runtime fallback, retry, or
parameter menu.  ``--static-check`` is CPU-only and creates no artifact.
Running without arguments is the separately authorized CUDA performance gate.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import importlib
import importlib.machinery
import itertools
import json
import math
import os
from pathlib import Path
import resource
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types
from typing import Any, Iterator, Mapping, Sequence

import scratch_phase_projection_fprime_production_five_case_sentinel as five


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts/hybridz_largecls_gates"
BASELINE_JSONL = ARTIFACT_ROOT / "phase_projection_gpu_csr_fixed400_20260814.jsonl"
BASELINE_SUMMARY = (
    ARTIFACT_ROOT / "phase_projection_gpu_csr_fixed400_20260814.summary.json"
)
NEW_ORACLE = (
    ARTIFACT_ROOT / "phase_projection_fprime_production_five_case_sentinel_20260814.json"
)
EVENTS_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_fprime_paired_noninferiority_20260814.events.jsonl"
)
SUMMARY_PATH = (
    ARTIFACT_ROOT / "phase_projection_fprime_paired_noninferiority_20260814.json"
)

OLD_BLOB_OID = "08e9ca2cb1f91b3cfbeae86b2dd5ca0d4349d025"
OLD_BLOB_BYTES = 55_419
OLD_SOURCE_SHA256 = (
    "4b66470df55edebb595e0e06c6b8a2de5c65496b8671c4d2f2552003d01ea306"
)
CURRENT_PHASE_SHA256 = (
    "13625f452c36a1b7844e4385b884471c8a0c82abf015bf2af417257e2c96c23a"
)
FIVE_HELPER_SHA256 = (
    "3d168c9f29fae8343d5f101794d6b829cfbdf48a30dda57cea2cfd30ac595873"
)
OFFLINE_SOURCE_LOCKS: Mapping[str, str] = {
    str(BASELINE_JSONL.relative_to(ROOT)): (
        "749db4e400329598c23c3dd7c9b9863c291eb3d1ba556cdcfcfa879c58487b43"
    ),
    str(BASELINE_SUMMARY.relative_to(ROOT)): (
        "036a87f7005033ad8478af5dbecfce8657d3f02457f654a9b0e63e5b47e2ab41"
    ),
    str(NEW_ORACLE.relative_to(ROOT)): (
        "fc1471a8c41dc548a8386a34a1996fe1fdf2d91f420accdab53803153ac8c681"
    ),
}

CONTROL_ORDER = (
    "cifar100_medium_iid2",
    "cifar100_large_iid153",
    "cifar100_large_iid166",
    "tinyimagenet_medium_iid153",
    "tinyimagenet_medium_iid143",
)
EXPECTED_OLD = {
    "cifar100_medium_iid2": "FALSIFIED",
    "cifar100_large_iid153": "UNKNOWN",
    "cifar100_large_iid166": "UNKNOWN",
    "tinyimagenet_medium_iid153": "UNKNOWN",
    "tinyimagenet_medium_iid143": "UNKNOWN",
}
EXPECTED_NEW = {
    "cifar100_medium_iid2": "FALSIFIED",
    "cifar100_large_iid153": "UNKNOWN",
    "cifar100_large_iid166": "UNKNOWN",
    "tinyimagenet_medium_iid153": "FALSIFIED",
    "tinyimagenet_medium_iid143": "FALSIFIED",
}
MODES: Mapping[str, int] = {"single": 1, "four_thread": 4}
IMPLEMENTATIONS = ("old59", "new_production")
WARMUP_PAIRS = 2
MEASURED_PAIRS = 5
EXACT_BOOTSTRAP_RESAMPLES = MEASURED_PAIRS**MEASURED_PAIRS
MEDIAN_SPEEDUP_MINIMUM = 1.0
BOOTSTRAP_95_LOWER_MINIMUM = 0.95
REQUEST_SECONDS = 10.0
WORKER_TIMEOUT_SECONDS = 300.0
RESULT_PREFIX = "@@ACT_FPRIME_PAIRED_NONINFERIORITY_RESULT@@"
WORKER_TOKEN_ENV = "ACT_FPRIME_PAIRED_NONINFERIORITY_ATTEMPT_TOKEN"
CANONICAL_PHASE_MODULE = (
    "act.back_end.hybridz_tf.forward_exact_relu_phase_projection_candidate"
)
OLD_MEMORY_MODULE = (
    "act.back_end.hybridz_tf._frozen_old59_phase_08e9ca2cb1f91b3c"
)

# Replaced after the deterministic manifest has been reconstructed once.
EXPECTED_MANIFEST_SHA256 = (
    "895a8ae1459c9f4f02654240f4dd8d935decff16128117fe77bde69c4e149dad"
)


class GateError(RuntimeError):
    """The comparator cannot safely interpret or continue the gate."""


class ResultConflict(GateError):
    """A measured/warm outcome disagreed with its frozen internal oracle."""


@dataclass(frozen=True)
class Job:
    name: str
    case_name: str
    mode: str
    workers: int


@dataclass(frozen=True)
class Manifest:
    cases: Mapping[str, five.Case]
    jobs: tuple[Job, ...]
    payload: Mapping[str, Any]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_hex_digest(value: Any) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _git_blob_bytes() -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", OLD_BLOB_OID],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    source = completed.stdout
    if completed.returncode != 0 or completed.stderr or len(source) != OLD_BLOB_BYTES:
        raise GateError("frozen old59 Git blob is unavailable or malformed")
    git_object = b"blob " + str(len(source)).encode("ascii") + b"\0" + source
    if hashlib.sha1(git_object).hexdigest() != OLD_BLOB_OID:
        raise GateError("old59 Git object identity failed independent SHA-1 replay")
    if hashlib.sha256(source).hexdigest() != OLD_SOURCE_SHA256:
        raise GateError("old59 Git blob content SHA-256 changed")
    return source


def _assert_sources() -> bytes:
    for relative, expected in OFFLINE_SOURCE_LOCKS.items():
        if _sha256(ROOT / relative) != expected:
            raise GateError(f"offline comparator source changed: {relative}")
    helper = Path(five.__file__).resolve()
    if helper != ROOT / "scratch_phase_projection_fprime_production_five_case_sentinel.py":
        raise GateError("audited five-case helper resolved outside the workspace")
    if _sha256(helper) != FIVE_HELPER_SHA256:
        raise GateError("audited five-case helper changed")
    bundle = five._source_bundle()
    if (
        bundle.get(
            "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py"
        )
        != CURRENT_PHASE_SHA256
    ):
        raise GateError("current production phase source is not frozen")
    return _git_blob_bytes()


def _zero_owner_audit() -> dict[str, Any]:
    return {
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


def _validate_zero_owner_audit(value: Any) -> None:
    expected = _zero_owner_audit()
    integer_fields = tuple(name for name in expected if name != "logical_owner_final_states")
    if not (
        type(value) is dict
        and set(value) == set(expected)
        and all(type(value.get(name)) is int and value[name] == 0 for name in integer_fields)
        and type(value.get("logical_owner_final_states")) is list
        and value["logical_owner_final_states"] == []
    ):
        raise GateError("old59 call used a current owner or has malformed zero counters")


_OLD_PROJECTION_COMMON_KEYS = frozenset(
    {
        "schema",
        "enabled",
        "status",
        "configured_seconds",
        "elapsed_seconds",
        "verifier_owned_proof_authority",
        "input_sampling_used",
        "pgd_used",
        "concrete_onnx_execution_used",
        "bab_used",
        "backward_used",
        "dual_tightening_used",
    }
)
_OLD_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "selected_property_row",
        "input_factors",
        "phase_rows",
        "initial_phase_changes",
        "lp_rows",
        "lp_nnz",
        "candidate_margin",
        "singleton_margin_lower",
        "setup_seconds",
        "first_center_seconds",
        "first_stream_seconds",
        "target_center_seconds",
        "delta_seconds",
        "expansion_seconds",
        "model_seconds",
        "lp_seconds",
        "singleton_seconds",
        "total_seconds",
        "phase_updates",
        "phase_retries",
        "property_rows_selected",
        "property_row_retries",
        "all_unstable_exact",
        "triangle_rows",
        "input_sampling_used",
        "pgd_used",
        "concrete_onnx_execution_used",
        "bab_used",
        "backward_used",
        "dual_tightening_used",
        "singleton_interval_verified",
        "candidate_authority",
        "proof_authority",
        "verdict_authority",
        "generator_streams",
        "generator_representation",
        "candidate_outward_error_bands_used",
        "intermediate_phase_or_margin_replay_used",
    }
)


def _validate_old_call(record: Mapping[str, Any]) -> str:
    projection = record.get("phase_projection")
    audit = record.get("owner_audit")
    if type(projection) is not dict:
        raise GateError("old59 call lacks projection metadata")
    _validate_zero_owner_audit(audit)
    elapsed = projection.get("elapsed_seconds")
    if not (
        projection.get("schema") == "verifier_operator_phase_projection_v1"
        and projection.get("enabled") is True
        and type(projection.get("configured_seconds")) is float
        and projection["configured_seconds"] == REQUEST_SECONDS
        and type(elapsed) is float
        and math.isfinite(elapsed)
        and elapsed >= 0.0
    ):
        raise GateError("old59 production phase was not enabled for ten seconds")
    if any(projection.get(name) is not False for name in five._restriction_fields()):
        raise GateError("old59 call reports a prohibited method")
    status = record.get("status")
    if status == "VerifyStatus.FALSIFIED":
        if set(projection) != _OLD_PROJECTION_COMMON_KEYS | {
            "candidate_receipt",
            "proof_rule",
        }:
            raise GateError("old59 formal positive projection schema changed")
        receipt = projection.get("candidate_receipt")
        integer_fields = (
            "selected_property_row",
            "input_factors",
            "phase_rows",
            "initial_phase_changes",
            "lp_rows",
            "lp_nnz",
            "phase_updates",
            "phase_retries",
            "property_rows_selected",
            "property_row_retries",
            "triangle_rows",
            "generator_streams",
        )
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
            "candidate_outward_error_bands_used",
            "intermediate_phase_or_margin_replay_used",
        )
        if type(receipt) is not dict:
            raise GateError("old59 formal positive lacks a candidate receipt")
        margin = receipt.get("singleton_margin_lower")
        candidate_margin = receipt.get("candidate_margin")
        timing_fields = (
            "setup_seconds",
            "first_center_seconds",
            "first_stream_seconds",
            "target_center_seconds",
            "delta_seconds",
            "expansion_seconds",
            "model_seconds",
            "lp_seconds",
            "singleton_seconds",
            "total_seconds",
        )
        if not (
            set(receipt) == _OLD_RECEIPT_KEYS
            and receipt.get("schema")
            == "act.hybridz.forward_exact_relu_phase_projection_candidate.v2"
            and receipt.get("status") == "singleton_verified"
            and receipt.get("singleton_interval_verified") is True
            and type(margin) is float
            and math.isfinite(margin)
            and margin > 0.0
            and type(candidate_margin) is float
            and math.isfinite(candidate_margin)
            and candidate_margin > 0.0
            and all(type(receipt.get(name)) is int for name in integer_fields)
            and all(receipt[name] >= 0 for name in integer_fields)
            and all(
                type(receipt.get(name)) is float
                and math.isfinite(receipt[name])
                and receipt[name] >= 0.0
                for name in timing_fields
            )
            and receipt.get("phase_updates") == 1
            and receipt.get("phase_retries") == 0
            and receipt.get("property_rows_selected") == 1
            and receipt.get("property_row_retries") == 0
            and receipt.get("triangle_rows") == 0
            and receipt.get("generator_streams") == 1
            and receipt.get("generator_representation")
            == "gpu_emitted_selected_csr_v1"
            and receipt.get("all_unstable_exact") is True
            and all(receipt.get(name) is False for name in false_fields)
            and record.get("has_counterexample") is True
            and _is_hex_digest(record.get("counterexample_sha256"))
            and projection.get("status") == "FALSIFIED"
            and projection.get("verifier_owned_proof_authority") is True
            and projection.get("proof_rule")
            == "decoded_input_in_raw_BOX;verifier_owned_zero_width_forward_interval;"
            "exact_Fraction_property_lower_bound_positive"
        ):
            raise GateError("old59 formal positive crossed its frozen authority boundary")
        return "FALSIFIED"
    if status == "VerifyStatus.UNKNOWN":
        if set(projection) != _OLD_PROJECTION_COMMON_KEYS | {"reason"}:
            raise GateError("old59 UNKNOWN projection schema changed")
        reason = projection.get("reason")
        if not (
            record.get("has_counterexample") is False
            and record.get("counterexample_sha256") is None
            and projection.get("status") == "UNKNOWN"
            and projection.get("verifier_owned_proof_authority") is False
            and "candidate_receipt" not in projection
            and type(reason) is str
            and bool(reason)
            and not reason.startswith("unexpected_fail_closed:")
        ):
            raise GateError("old59 UNKNOWN crossed the fail-closed boundary")
        return "UNKNOWN"
    raise GateError("old59 call returned an unsupported status")


def _validate_new_call(record: Mapping[str, Any]) -> str:
    observed = five._validated_status(record)
    if observed == "FALSIFIED":
        if not _is_hex_digest(record.get("counterexample_sha256")):
            raise GateError("new production positive lacks its sealed input digest")
    else:
        reason = record["phase_projection"].get("reason")
        unexpected_candidate_reasons = {
            "phase-projection request-local modules are unavailable",
            "phase-projection request-local transaction failed closed",
        }
        if record.get("counterexample_sha256") is not None:
            raise GateError("new production UNKNOWN carries a counterexample digest")
        if (
            type(reason) is not str
            or reason.startswith("unexpected_fail_closed:")
            or reason in unexpected_candidate_reasons
        ):
            raise GateError("new production UNKNOWN masks an unexpected verifier error")
    return observed


def _strip_timing(value: Any) -> Any:
    if type(value) is dict:
        return {
            key: _strip_timing(item)
            for key, item in value.items()
            if "seconds" not in key.lower() and "elapsed" not in key.lower()
        }
    if type(value) is list:
        return [_strip_timing(item) for item in value]
    return value


def _semantic_id(record: Mapping[str, Any]) -> str:
    return _json_digest(
        {
            "status": record.get("validated_status"),
            "has_counterexample": record.get("has_counterexample"),
            "counterexample_sha256": record.get("counterexample_sha256"),
            "phase_projection": _strip_timing(record.get("phase_projection")),
            "owner_audit": record.get("owner_audit"),
        }
    )


def _baseline_oracle() -> dict[str, str]:
    wanted = set(CONTROL_ORDER)
    observed: dict[str, str] = {}
    with BASELINE_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            name = record.get("case")
            if name not in wanted:
                continue
            projection = record.get("phase_projection")
            adapted = {
                "status": record.get("status"),
                "has_counterexample": record.get("has_counterexample"),
                "counterexample_sha256": (
                    "0" * 64 if record.get("has_counterexample") is True else None
                ),
                "phase_projection": projection,
                "owner_audit": _zero_owner_audit(),
            }
            status = _validate_old_call(adapted)
            if status != record.get("validated_status") or name in observed:
                raise GateError("old59 oracle record fails independent replay")
            observed[name] = status
    if observed != EXPECTED_OLD:
        raise GateError("old59 oracle does not match the frozen five controls")
    return observed


def _new_oracle() -> dict[str, str]:
    data = json.loads(NEW_ORACLE.read_text(encoding="utf-8"))
    if data.get("status") != "COMPLETE_COMPATIBLE":
        raise GateError("new production five-case oracle is incomplete")
    observed: dict[str, str] = {}
    for record in data.get("results", []):
        name = record.get("case")
        if name not in set(CONTROL_ORDER) or name in observed:
            raise GateError("new production oracle has a malformed case set")
        status = five._validated_status(record)
        if status != record.get("validated_status"):
            raise GateError("new production oracle fails authority replay")
        observed[name] = status
    if observed != EXPECTED_NEW:
        raise GateError("new production oracle does not match the frozen five controls")
    return observed


def _build_manifest(*, verify_inputs: bool, enforce_digest: bool = True) -> Manifest:
    old_source = _assert_sources()
    old_oracle = _baseline_oracle()
    new_oracle = _new_oracle()
    helper_cases = {case.name: case for case in five.CASES}
    if set(helper_cases) != set(CONTROL_ORDER):
        raise GateError("five-case helper and paired control sets disagree")
    cases: dict[str, five.Case] = {}
    case_payload: list[dict[str, Any]] = []
    verified_paths: dict[Path, str] = {}
    for position, name in enumerate(CONTROL_ORDER):
        case = helper_cases[name]
        onnx, vnnlib = five._resolve(case) if verify_inputs else (
            five.BENCHMARK_ROOT / case.benchmark / "onnx" / case.model_name,
            five.BENCHMARK_ROOT / case.benchmark / "vnnlib" / case.spec_name,
        )
        for path, expected in (
            (onnx, case.onnx_sha256),
            (vnnlib, case.vnnlib_sha256),
        ):
            prior = verified_paths.setdefault(path.resolve(), expected)
            if prior != expected:
                raise GateError("one input path has conflicting frozen hashes")
        cases[name] = case
        case_payload.append(
            {
                "position": position,
                "name": name,
                "benchmark": case.benchmark,
                "iid": case.iid,
                "model_name": case.model_name,
                "spec_name": case.spec_name,
                "onnx_sha256": case.onnx_sha256,
                "vnnlib_sha256": case.vnnlib_sha256,
                "old59_expected": old_oracle[name],
                "new_production_expected": new_oracle[name],
                "expected_outcome_supplied_to_verify_once": False,
                "external_label_supplied_to_verify_once": False,
            }
        )
    jobs = tuple(
        Job(f"{name}__{mode}", name, mode, workers)
        for name in CONTROL_ORDER
        for mode, workers in MODES.items()
    )
    payload: dict[str, Any] = {
        "schema": "act.hybridz.fprime.paired_noninferiority_manifest.v1",
        "controls": case_payload,
        "jobs": [
            {
                "position": position,
                "name": job.name,
                "case": job.case_name,
                "mode": job.mode,
                "workers": job.workers,
            }
            for position, job in enumerate(jobs)
        ],
        "protocol": {
            "fresh_worker_per_case_mode": True,
            "warmup_pairs": WARMUP_PAIRS,
            "measured_pairs": MEASURED_PAIRS,
            "alternating_order": True,
            "exact_bootstrap_resamples": EXACT_BOOTSTRAP_RESAMPLES,
            "median_speedup_minimum": MEDIAN_SPEEDUP_MINIMUM,
            "bootstrap_95_lower_minimum": BOOTSTRAP_95_LOWER_MINIMUM,
            "request_lp_phase_seconds": REQUEST_SECONDS,
            "first_failed_job_stops_parent": True,
            "job_retries": 0,
        },
        "old59": {
            "git_blob_oid": OLD_BLOB_OID,
            "source_bytes": len(old_source),
            "source_sha256": OLD_SOURCE_SHA256,
            "load": "git_cat_file_then_compile_exec_in_memory_no_source_file",
            "module_role": "phase_source_only_under_unchanged_verify_once",
        },
        "new_production_source_bundle": five._source_bundle(),
        "offline_source_sha256": dict(OFFLINE_SOURCE_LOCKS),
        "five_case_helper_sha256": FIVE_HELPER_SHA256,
        "external_sat_labels_used": False,
    }
    digest = _json_digest(payload)
    if enforce_digest and digest != EXPECTED_MANIFEST_SHA256:
        raise GateError("paired manifest differs from its frozen digest: " + digest)
    return Manifest(cases=cases, jobs=jobs, payload=payload, sha256=digest)


def _identity(manifest: Manifest) -> dict[str, Any]:
    bundle = five._source_bundle()
    return {
        "schema": "act.hybridz.fprime.paired_noninferiority.identity.v1",
        "harness_sha256": _sha256(Path(__file__)),
        "manifest_sha256": manifest.sha256,
        "manifest": dict(manifest.payload),
        "five_case_helper_sha256": FIVE_HELPER_SHA256,
        "production_source_bundle": bundle,
        "production_source_bundle_sha256": five._bundle_sha256(bundle),
        "old_git_blob_oid": OLD_BLOB_OID,
        "old_source_sha256": OLD_SOURCE_SHA256,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "worker_timeout_role": "fresh_job_transport_limit_not_request_or_lp_deadline",
        "python_major_minor": [sys.version_info.major, sys.version_info.minor],
        "runtime_paths_per_measured_call": 1,
        "job_attempts_max": 1,
        "external_labels_read_by_production": False,
        "randomness_scope": "ipc_capability_only_bootstrap_is_exact_enumeration",
    }


def _exact_bootstrap(speedups: Sequence[float]) -> tuple[float, int]:
    values = tuple(float(value) for value in speedups)
    if len(values) != 5 or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise GateError("exact bootstrap requires five finite positive speedups")
    medians = sorted(
        statistics.median(values[index] for index in indices)
        for indices in itertools.product(range(5), repeat=5)
    )
    if len(medians) != EXACT_BOOTSTRAP_RESAMPLES:
        raise GateError("exact bootstrap did not enumerate 5**5 resamples")
    lower_index = int(math.floor(0.025 * (len(medians) - 1)))
    return float(medians[lower_index]), lower_index


def _scope() -> dict[str, Any]:
    return {
        "offline_two_implementation_comparator": True,
        "production_verify_once": True,
        "old_source_memory_only": True,
        "fresh_worker_per_case_mode": True,
        "shared_prepared_graph_within_worker": True,
        "input_sampling_used": False,
        "onnx_input_point_execution_used": False,
        "pgd_used": False,
        "bab_split_or_enumeration_used": False,
        "backward_bounds_used": False,
        "dual_tightening_used": False,
        "production_runtime_fallback_or_menu_used": False,
        "job_retries": 0,
        "external_label_read_by_production": False,
        "timing_authority": True,
        "same_process_old_new_memory_attribution_authority": False,
        "resource_measurements_are_diagnostics": True,
        "verdict_authority_unchanged": True,
    }


def _rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    if len(fields) < 2:
        raise GateError("/proc/self/statm is malformed")
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def _hwm_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _PerRequestOwnerInstrumentation:
    """Transparent request-indexed counters around the real production owner."""

    def __init__(self, owner_module: Any) -> None:
        self.owner_module = owner_module
        self.real_owner = owner_module.SafeHighsOwner
        self.real_highs = owner_module.highspy.Highs
        self.local = threading.local()
        # TrackingOwner/TrackingNative update a request audit while already
        # holding this lock.  A re-entrant lock keeps creation of the audit and
        # its counter update one atomic operation without self-deadlocking.
        self.lock = threading.RLock()
        self.audits: dict[int, dict[str, Any]] = {}

    def _request_id(self) -> int:
        value = getattr(self.local, "request_id", None)
        if type(value) is not int:
            raise GateError("owner activity is not bound to a request index")
        return value

    def _audit(self, request_id: int) -> dict[str, Any]:
        with self.lock:
            if request_id not in self.audits:
                self.audits[request_id] = _zero_owner_audit()
            return self.audits[request_id]

    @contextlib.contextmanager
    def request(self, request_id: int) -> Iterator[None]:
        if hasattr(self.local, "request_id"):
            raise GateError("nested owner request instrumentation is forbidden")
        self.local.request_id = request_id
        try:
            yield
        finally:
            del self.local.request_id

    def __enter__(self) -> "_PerRequestOwnerInstrumentation":
        instrumentation = self
        real_owner = self.real_owner
        real_highs = self.real_highs

        class TrackingOwner(real_owner):
            def __init__(tracked_self: Any, *args: Any, **kwargs: Any) -> None:
                request_id = instrumentation._request_id()
                tracked_self._paired_request_id = request_id
                with instrumentation.lock:
                    instrumentation._audit(request_id)["logical_owner_instances"] += 1
                super().__init__(*args, **kwargs)

            def close(tracked_self: Any) -> None:
                request_id = tracked_self._paired_request_id
                with instrumentation.lock:
                    instrumentation._audit(request_id)["logical_owner_close_calls"] += 1
                try:
                    super().close()
                finally:
                    with instrumentation.lock:
                        instrumentation._audit(request_id)[
                            "logical_owner_final_states"
                        ].append(tracked_self.state)

        class TrackingNative:
            def __init__(tracked_self: Any) -> None:
                request_id = instrumentation._request_id()
                tracked_self._paired_request_id = request_id
                with instrumentation.lock:
                    instrumentation._audit(request_id)["native_owner_instances"] += 1
                tracked_self._backend = real_highs()

            def __getattr__(tracked_self: Any, name: str) -> Any:
                return getattr(tracked_self._backend, name)

            def _increment(tracked_self: Any, name: str) -> None:
                with instrumentation.lock:
                    instrumentation._audit(tracked_self._paired_request_id)[name] += 1

            def run(tracked_self: Any) -> Any:
                tracked_self._increment("native_run_calls")
                return tracked_self._backend.run()

            def clear(tracked_self: Any) -> Any:
                tracked_self._increment("native_clear_calls")
                return tracked_self._backend.clear()

            def clearModel(tracked_self: Any) -> Any:
                tracked_self._increment("native_clear_model_calls")
                return tracked_self._backend.clearModel()

            def getDualRayExist(tracked_self: Any) -> Any:
                tracked_self._increment("dual_ray_exist_calls")
                return tracked_self._backend.getDualRayExist()

            def getDualRay(tracked_self: Any) -> Any:
                tracked_self._increment("dual_ray_calls")
                return tracked_self._backend.getDualRay()

        self.owner_module.SafeHighsOwner = TrackingOwner
        self.owner_module.highspy.Highs = TrackingNative
        return self

    def __exit__(self, _kind: Any, _value: Any, _tb: Any) -> bool:
        self.owner_module.SafeHighsOwner = self.real_owner
        self.owner_module.highspy.Highs = self.real_highs
        return False


def _load_old_module(source: bytes) -> Any:
    if OLD_MEMORY_MODULE in sys.modules:
        raise GateError("old59 memory module already exists in this worker")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError("old59 source is not UTF-8") from exc
    code = compile(text, f"<git-blob:{OLD_BLOB_OID}>", "exec", dont_inherit=True)
    module = types.ModuleType(OLD_MEMORY_MODULE)
    module.__file__ = f"<git-blob:{OLD_BLOB_OID}>"
    module.__package__ = "act.back_end.hybridz_tf"
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        OLD_MEMORY_MODULE,
        loader=None,
        origin=module.__file__,
    )
    sys.modules[OLD_MEMORY_MODULE] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(OLD_MEMORY_MODULE, None)
        raise
    if not (
        module._SCHEMA == "act.hybridz.forward_exact_relu_phase_projection_candidate.v2"
        and callable(module.build_forward_exact_relu_phase_projection_candidate)
    ):
        sys.modules.pop(OLD_MEMORY_MODULE, None)
        raise GateError("old59 memory module exports changed")
    return module


@contextlib.contextmanager
def _bind_phase_module(module: Any) -> Iterator[None]:
    package = importlib.import_module("act.back_end.hybridz_tf")
    attribute = CANONICAL_PHASE_MODULE.rsplit(".", 1)[1]
    prior_module = sys.modules.get(CANONICAL_PHASE_MODULE)
    prior_attribute = getattr(package, attribute, None)
    sys.modules[CANONICAL_PHASE_MODULE] = module
    setattr(package, attribute, module)
    try:
        yield
    finally:
        if prior_module is None:
            sys.modules.pop(CANONICAL_PHASE_MODULE, None)
        else:
            sys.modules[CANONICAL_PHASE_MODULE] = prior_module
        if prior_attribute is None:
            try:
                delattr(package, attribute)
            except AttributeError:
                pass
        else:
            setattr(package, attribute, prior_attribute)


def _backend_config() -> Any:
    from act.back_end.config import BackendConfig, HybridZConfig

    return BackendConfig(
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
    )


def _prepare_net(case: five.Case) -> Any:
    import torch

    from act.back_end.transfer_functions import (
        set_solver_mode,
        set_transfer_function_mode,
    )
    from act.back_end.verifier import (
        _ensure_assert_linear_encoding,
        _get_output_layer_id,
        get_assert_layer,
    )
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
    from act.pipeline.verification.torch2act import TorchToACT
    from act.util.device_manager import initialize_device

    onnx, vnnlib = five._resolve(case)
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    specs = create_specs_from_paths(str(onnx), str(vnnlib), category=case.benchmark)
    wrapped = synthesize_models_from_specs([specs])
    if len(wrapped) != 1:
        raise GateError("paired worker requires exactly one wrapped model")
    model = next(iter(wrapped.values())).to(
        device=torch.device("cuda"), dtype=torch.float64
    )
    net = TorchToACT(model).run()
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
    del model, wrapped, specs
    return net


def _call_record(result: Any, audit: Mapping[str, Any], request_index: int) -> dict[str, Any]:
    import numpy as np

    counterexample = result.counterexample
    counterexample_sha256: str | None = None
    if counterexample is not None:
        array = np.ascontiguousarray(
            counterexample.detach().cpu().double().numpy(), dtype=np.float64
        )
        counterexample_sha256 = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return {
        "request_index": request_index,
        "status": str(result.status),
        "has_counterexample": counterexample is not None,
        "counterexample_sha256": counterexample_sha256,
        "phase_projection": result.metadata.get("operator_phase_projection", {}),
        "owner_audit": dict(audit),
    }


def _validate_call(
    record: Mapping[str, Any], implementation: str, expected: str
) -> str:
    observed = (
        _validate_old_call(record)
        if implementation == "old59"
        else _validate_new_call(record)
    )
    if observed != expected:
        raise ResultConflict(
            f"{implementation} observed {observed}, expected frozen {expected}"
        )
    if record.get("validated_status") not in {None, observed}:
        raise GateError("stored call status disagrees with authority replay")
    return observed


def _cuda_metrics(torch: Any) -> dict[str, int]:
    total = int(
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    )
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_total_bytes": total,
    }


def _run_group(
    *,
    implementation: str,
    module: Any,
    net: Any,
    workers: int,
    expected: str,
    verify_once: Any,
    owner_module: Any,
    torch: Any,
) -> dict[str, Any]:
    # Do not collect or empty the CUDA allocator between the A/B members of a
    # pair: doing so changes cache state and invalidates the paired timing.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rss_before = _rss_bytes()
    hwm_before = _hwm_bytes()
    cuda_before = _cuda_metrics(torch)
    instrumentation = _PerRequestOwnerInstrumentation(owner_module)

    def invoke(request_index: int) -> tuple[int, Any]:
        with instrumentation.request(request_index):
            results = verify_once(net, backend_cfg=_backend_config())
        if len(results) != 1:
            raise GateError("production verify_once returned more than one lane")
        return request_index, results[0]

    with _bind_phase_module(module), instrumentation:
        started = time.perf_counter()
        if workers == 1:
            raw_results = [invoke(0)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                raw_results = list(pool.map(invoke, range(workers)))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    cuda_peak = _cuda_metrics(torch)
    call_records = []
    for request_index, result in raw_results:
        audit = instrumentation.audits.get(request_index, _zero_owner_audit())
        record = _call_record(result, audit, request_index)
        observed = _validate_call(record, implementation, expected)
        record["validated_status"] = observed
        record["semantic_id"] = _semantic_id(record)
        call_records.append(record)
    call_records.sort(key=lambda item: item["request_index"])
    if [item["request_index"] for item in call_records] != list(range(workers)):
        raise GateError("group request indices are not exact")
    semantic_ids = [item["semantic_id"] for item in call_records]
    if len(set(semantic_ids)) != 1:
        raise ResultConflict("concurrent calls produced different semantic identities")
    del raw_results
    torch.cuda.synchronize()
    return {
        "implementation": implementation,
        "workers": workers,
        "elapsed_seconds": float(elapsed),
        "calls": call_records,
        "semantic_id": semantic_ids[0],
        "result_conflicts": 0,
        "resources": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": _rss_bytes(),
            "hwm_before_bytes": hwm_before,
            "hwm_after_bytes": _hwm_bytes(),
            "cuda_before": cuda_before,
            "cuda_peak": cuda_peak,
            "cuda_after": _cuda_metrics(torch),
            "cuda_synchronized_before_timing": True,
            "cuda_synchronized_after_timing": True,
            "cuda_peak_reset_before_timing": True,
            "gc_or_empty_cache_inside_pair": False,
        },
    }


def _run_job(job: Job, case: five.Case, manifest: Manifest) -> dict[str, Any]:
    import torch

    from act.back_end.hybridz_tf import phase_projection_highs_owner
    from act.back_end.verifier import verify_once

    old_source = _assert_sources()
    # Resolve and hash once before setup and once after all measurements.  The
    # expected labels remain parent-only and are never part of either call.
    five._resolve(case)
    input_identity = {
        "onnx": case.onnx_sha256,
        "vnnlib": case.vnnlib_sha256,
    }
    current_module = importlib.import_module(CANONICAL_PHASE_MODULE)
    if _sha256(Path(current_module.__file__).resolve()) != CURRENT_PHASE_SHA256:
        raise GateError("current production module changed before timing")
    old_module = _load_old_module(old_source)
    before_setup = {
        "rss_bytes": _rss_bytes(),
        "hwm_bytes": _hwm_bytes(),
    }
    net = _prepare_net(case)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    after_setup = {
        "rss_bytes": _rss_bytes(),
        "hwm_bytes": _hwm_bytes(),
        "cuda": _cuda_metrics(torch),
    }
    expected = {"old59": EXPECTED_OLD[case.name], "new_production": EXPECTED_NEW[case.name]}
    modules = {"old59": old_module, "new_production": current_module}
    stable_ids: dict[str, set[str]] = {name: set() for name in IMPLEMENTATIONS}
    warmups = []
    pairs = []
    maximum_rss = after_setup["rss_bytes"]
    maximum_hwm = after_setup["hwm_bytes"]
    maximum_cuda_allocated = after_setup["cuda"]["allocated_bytes"]
    maximum_cuda_reserved = after_setup["cuda"]["reserved_bytes"]
    result: dict[str, Any] | None = None

    def observe(group: Mapping[str, Any]) -> None:
        nonlocal maximum_rss, maximum_hwm
        nonlocal maximum_cuda_allocated, maximum_cuda_reserved
        resources = group["resources"]
        maximum_rss = max(
            maximum_rss,
            int(resources["rss_before_bytes"]),
            int(resources["rss_after_bytes"]),
        )
        maximum_hwm = max(
            maximum_hwm,
            int(resources["hwm_before_bytes"]),
            int(resources["hwm_after_bytes"]),
        )
        maximum_cuda_allocated = max(
            maximum_cuda_allocated,
            int(resources["cuda_peak"]["max_allocated_bytes"]),
        )
        maximum_cuda_reserved = max(
            maximum_cuda_reserved,
            int(resources["cuda_peak"]["max_reserved_bytes"]),
        )

    try:
        for pair_index in range(WARMUP_PAIRS):
            order = (
                IMPLEMENTATIONS
                if pair_index % 2 == 0
                else tuple(reversed(IMPLEMENTATIONS))
            )
            runs = []
            for implementation in order:
                group = _run_group(
                    implementation=implementation,
                    module=modules[implementation],
                    net=net,
                    workers=job.workers,
                    expected=expected[implementation],
                    verify_once=verify_once,
                    owner_module=phase_projection_highs_owner,
                    torch=torch,
                )
                stable_ids[implementation].add(group["semantic_id"])
                observe(group)
                runs.append(group)
            warmups.append({"pair_index": pair_index, "order": list(order), "runs": runs})

        for pair_index in range(MEASURED_PAIRS):
            order = (
                IMPLEMENTATIONS
                if pair_index % 2 == 0
                else tuple(reversed(IMPLEMENTATIONS))
            )
            runs = []
            by_implementation: dict[str, Mapping[str, Any]] = {}
            for implementation in order:
                group = _run_group(
                    implementation=implementation,
                    module=modules[implementation],
                    net=net,
                    workers=job.workers,
                    expected=expected[implementation],
                    verify_once=verify_once,
                    owner_module=phase_projection_highs_owner,
                    torch=torch,
                )
                stable_ids[implementation].add(group["semantic_id"])
                by_implementation[implementation] = group
                runs.append(group)
                observe(group)
            old_seconds = float(by_implementation["old59"]["elapsed_seconds"])
            new_seconds = float(
                by_implementation["new_production"]["elapsed_seconds"]
            )
            if not (
                math.isfinite(old_seconds)
                and math.isfinite(new_seconds)
                and old_seconds > 0.0
                and new_seconds > 0.0
            ):
                raise GateError("measured pair has a nonpositive/nonfinite duration")
            pairs.append(
                {
                    "pair_index": pair_index,
                    "order": list(order),
                    "runs": runs,
                    "old59_seconds": old_seconds,
                    "new_production_seconds": new_seconds,
                    "speedup": old_seconds / new_seconds,
                }
            )
        if any(len(values) != 1 for values in stable_ids.values()):
            raise ResultConflict("one implementation changed semantic identity across groups")
        speedups = [float(pair["speedup"]) for pair in pairs]
        median_speedup = float(statistics.median(speedups))
        bootstrap_lower, lower_index = _exact_bootstrap(speedups)
        median_passed = median_speedup >= MEDIAN_SPEEDUP_MINIMUM
        bootstrap_passed = bootstrap_lower >= BOOTSTRAP_95_LOWER_MINIMUM
        status = (
            "PASSED"
            if median_passed and bootstrap_passed
            else "REJECTED_NONINFERIORITY"
        )
        # Recheck every immutable input and source after timing.  This is an
        # identity seal, not another algorithm attempt.
        five._resolve(case)
        _assert_sources()
        result = {
            "schema": "act.hybridz.fprime.paired_noninferiority.worker.v1",
            "status": status,
            "passed": status == "PASSED",
            "case": case.name,
            "job": job.name,
            "mode": job.mode,
            "workers": job.workers,
            "input_sha256": input_identity,
            "expected_outcomes": expected,
            "expected_outcomes_supplied_to_verify_once": False,
            "warmup_pairs": warmups,
            "measured_pairs": pairs,
            "performance": {
                "speedups": speedups,
                "median_speedup": median_speedup,
                "median_speedup_minimum": MEDIAN_SPEEDUP_MINIMUM,
                "median_passed": median_passed,
                "exact_bootstrap_resamples": EXACT_BOOTSTRAP_RESAMPLES,
                "exact_bootstrap_lower_index": lower_index,
                "paired_bootstrap_95_lower": bootstrap_lower,
                "bootstrap_95_lower_minimum": BOOTSTRAP_95_LOWER_MINIMUM,
                "bootstrap_passed": bootstrap_passed,
            },
            "result_conflicts": 0,
            "errors": 0,
            "stable_semantic_ids": {
                name: next(iter(values)) for name, values in stable_ids.items()
            },
            "resources": {
                "before_setup": before_setup,
                "after_setup": after_setup,
                "maximum_rss_bytes": maximum_rss,
                "maximum_hwm_bytes": maximum_hwm,
                "maximum_cuda_allocated_bytes": maximum_cuda_allocated,
                "maximum_cuda_reserved_bytes": maximum_cuda_reserved,
                "per_group_cache_cleanup_performed": False,
                "same_process_old_new_memory_attribution_authority": False,
                "final_cleanup": None,
            },
            "old_source": {
                "git_blob_oid": OLD_BLOB_OID,
                "source_sha256": OLD_SOURCE_SHA256,
                "loaded_in_memory": True,
                "temporary_source_written": False,
            },
            "manifest_sha256": manifest.sha256,
            "scope": _scope(),
        }
    finally:
        cleanup_before = {
            "rss_bytes": _rss_bytes(),
            "hwm_bytes": _hwm_bytes(),
            "cuda": _cuda_metrics(torch),
        }
        del net
        sys.modules.pop(OLD_MEMORY_MODULE, None)
        collected = int(gc.collect())
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        cleanup_after = {
            "rss_bytes": _rss_bytes(),
            "hwm_bytes": _hwm_bytes(),
            "cuda": _cuda_metrics(torch),
        }
        if result is not None:
            result["resources"]["final_cleanup"] = {
                "before": cleanup_before,
                "after": cleanup_after,
                "gc_collect_called": True,
                "gc_collected_objects": collected,
                "cuda_empty_cache_called": True,
                "cuda_synchronize_called": True,
                "old_memory_module_removed": OLD_MEMORY_MODULE not in sys.modules,
            }
    if result is None:
        raise GateError("paired job did not produce a result")
    return result


def _validate_resources(resources: Any) -> None:
    if type(resources) is not dict:
        raise GateError("job lacks resource/cleanup metadata")
    integer_fields = (
        "maximum_rss_bytes",
        "maximum_hwm_bytes",
        "maximum_cuda_allocated_bytes",
        "maximum_cuda_reserved_bytes",
    )
    if any(type(resources.get(name)) is not int or resources[name] < 0 for name in integer_fields):
        raise GateError("job resource count is malformed")
    if not (
        resources.get("per_group_cache_cleanup_performed") is False
        and resources.get("same_process_old_new_memory_attribution_authority") is False
    ):
        raise GateError("job cleanup contract was not recorded")
    cleanup = resources.get("final_cleanup")
    if type(cleanup) is not dict or not (
        cleanup.get("gc_collect_called") is True
        and type(cleanup.get("gc_collected_objects")) is int
        and cleanup["gc_collected_objects"] >= 0
        and cleanup.get("cuda_empty_cache_called") is True
        and cleanup.get("cuda_synchronize_called") is True
        and cleanup.get("old_memory_module_removed") is True
    ):
        raise GateError("job final cleanup receipt is malformed")
    for point in ("before", "after"):
        value = cleanup.get(point)
        if type(value) is not dict:
            raise GateError("job final cleanup point is absent")
        if any(
            type(value.get(name)) is not int or value[name] < 0
            for name in ("rss_bytes", "hwm_bytes")
        ):
            raise GateError("job final cleanup host memory is malformed")
        _validate_cuda_metrics(value.get("cuda"))


def _validate_cuda_metrics(value: Any) -> None:
    fields = (
        "allocated_bytes",
        "reserved_bytes",
        "max_allocated_bytes",
        "max_reserved_bytes",
        "device_total_bytes",
    )
    if type(value) is not dict or any(
        type(value.get(name)) is not int or value[name] < 0 for name in fields
    ):
        raise GateError("CUDA resource point is malformed")
    if not (
        value["allocated_bytes"] <= value["reserved_bytes"] <= value["device_total_bytes"]
        and value["max_allocated_bytes"]
        <= value["max_reserved_bytes"]
        <= value["device_total_bytes"]
    ):
        raise GateError("CUDA resource ordering is malformed")


def _validate_group(
    group: Mapping[str, Any],
    *,
    implementation: str,
    workers: int,
    expected: str,
) -> str:
    elapsed = group.get("elapsed_seconds")
    if not (
        group.get("implementation") == implementation
        and type(group.get("workers")) is int
        and group["workers"] == workers
        and type(elapsed) is float
        and math.isfinite(elapsed)
        and elapsed > 0.0
        and group.get("result_conflicts") == 0
    ):
        raise GateError("persisted group shape/timing is malformed")
    calls = group.get("calls")
    if type(calls) is not list or len(calls) != workers:
        raise GateError("persisted group has the wrong call count")
    semantic_ids = []
    for index, record in enumerate(calls):
        if type(record) is not dict or record.get("request_index") != index:
            raise GateError("persisted group request order changed")
        observed = _validate_call(record, implementation, expected)
        if record.get("validated_status") != observed:
            raise GateError("persisted call status changed")
        semantic = _semantic_id(record)
        if record.get("semantic_id") != semantic:
            raise GateError("persisted call semantic digest changed")
        semantic_ids.append(semantic)
    if len(set(semantic_ids)) != 1 or group.get("semantic_id") != semantic_ids[0]:
        raise GateError("persisted concurrent semantic identities conflict")
    resource_record = group.get("resources")
    if type(resource_record) is not dict:
        raise GateError("group resource receipt is missing")
    for field in (
        "rss_before_bytes",
        "rss_after_bytes",
        "hwm_before_bytes",
        "hwm_after_bytes",
    ):
        if type(resource_record.get(field)) is not int or resource_record[field] < 0:
            raise GateError("group host resource point is malformed")
    for point in ("cuda_before", "cuda_peak", "cuda_after"):
        _validate_cuda_metrics(resource_record.get(point))
    if not (
        resource_record.get("cuda_synchronized_before_timing") is True
        and resource_record.get("cuda_synchronized_after_timing") is True
        and resource_record.get("cuda_peak_reset_before_timing") is True
        and resource_record.get("gc_or_empty_cache_inside_pair") is False
    ):
        raise GateError("group timing/cache-state receipt is malformed")
    return semantic_ids[0]


def _validate_job_record(
    record: Mapping[str, Any],
    *,
    job: Job,
    manifest: Manifest,
    identity: Mapping[str, Any],
    token_sha256: str,
) -> str:
    case = manifest.cases[job.case_name]
    if not (
        record.get("schema") == "act.hybridz.fprime.paired_noninferiority.worker.v1"
        and record.get("case") == case.name
        and record.get("job") == job.name
        and record.get("mode") == job.mode
        and type(record.get("workers")) is int
        and record["workers"] == job.workers
        and record.get("manifest_sha256") == manifest.sha256
        and record.get("harness_sha256") == identity["harness_sha256"]
        and record.get("source_bundle_sha256")
        == identity["production_source_bundle_sha256"]
        and record.get("attempt_token_sha256") == token_sha256
        and _is_hex_digest(token_sha256)
        and record.get("input_sha256")
        == {"onnx": case.onnx_sha256, "vnnlib": case.vnnlib_sha256}
        and record.get("scope") == _scope()
    ):
        raise GateError("job result identity/source/scope is malformed")
    _validate_transport(record.get("transport"), formal=record.get("status") != "ERROR")
    if record.get("status") == "ERROR":
        if not (
            record.get("passed") is False
            and type(record.get("error_type")) is str
            and bool(record["error_type"])
        ):
            raise GateError("persisted fail-closed job error is malformed")
        return "ERROR"
    if record.get("status") not in {"PASSED", "REJECTED_NONINFERIORITY"}:
        raise GateError("job has an unsupported status")
    expected = {"old59": EXPECTED_OLD[case.name], "new_production": EXPECTED_NEW[case.name]}
    if not (
        record.get("expected_outcomes") == expected
        and record.get("expected_outcomes_supplied_to_verify_once") is False
        and record.get("result_conflicts") == 0
        and record.get("errors") == 0
    ):
        raise GateError("job outcome oracle/conflict fields changed")
    warmups = record.get("warmup_pairs")
    pairs = record.get("measured_pairs")
    if type(warmups) is not list or len(warmups) != WARMUP_PAIRS:
        raise GateError("job does not contain exactly two warmup pairs")
    if type(pairs) is not list or len(pairs) != MEASURED_PAIRS:
        raise GateError("job does not contain exactly five measured pairs")
    stable: dict[str, set[str]] = {name: set() for name in IMPLEMENTATIONS}

    def validate_pair(pair: Mapping[str, Any], pair_index: int, measured: bool) -> None:
        order = (
            IMPLEMENTATIONS
            if pair_index % 2 == 0
            else tuple(reversed(IMPLEMENTATIONS))
        )
        if pair.get("pair_index") != pair_index or pair.get("order") != list(order):
            raise GateError("pair alternation/order changed")
        runs = pair.get("runs")
        if type(runs) is not list or len(runs) != 2:
            raise GateError("pair lacks exactly two implementation runs")
        by_impl = {}
        for position, implementation in enumerate(order):
            group = runs[position]
            if type(group) is not dict:
                raise GateError("pair run is not an object")
            stable[implementation].add(
                _validate_group(
                    group,
                    implementation=implementation,
                    workers=job.workers,
                    expected=expected[implementation],
                )
            )
            by_impl[implementation] = group
        if measured:
            old_seconds = float(by_impl["old59"]["elapsed_seconds"])
            new_seconds = float(by_impl["new_production"]["elapsed_seconds"])
            speedup = old_seconds / new_seconds
            if not (
                pair.get("old59_seconds") == old_seconds
                and pair.get("new_production_seconds") == new_seconds
                and pair.get("speedup") == speedup
            ):
                raise GateError("persisted pair timing arithmetic changed")

    for index, pair in enumerate(warmups):
        if type(pair) is not dict:
            raise GateError("warmup pair is not an object")
        validate_pair(pair, index, False)
    for index, pair in enumerate(pairs):
        if type(pair) is not dict:
            raise GateError("measured pair is not an object")
        validate_pair(pair, index, True)
    if any(len(values) != 1 for values in stable.values()):
        raise GateError("persisted implementation semantic identity changed")
    if record.get("stable_semantic_ids") != {
        name: next(iter(values)) for name, values in stable.items()
    }:
        raise GateError("stored stable semantic IDs changed")
    speedups = [float(pair["speedup"]) for pair in pairs]
    median = float(statistics.median(speedups))
    lower, lower_index = _exact_bootstrap(speedups)
    performance = record.get("performance")
    if type(performance) is not dict or performance != {
        "speedups": speedups,
        "median_speedup": median,
        "median_speedup_minimum": MEDIAN_SPEEDUP_MINIMUM,
        "median_passed": median >= MEDIAN_SPEEDUP_MINIMUM,
        "exact_bootstrap_resamples": EXACT_BOOTSTRAP_RESAMPLES,
        "exact_bootstrap_lower_index": lower_index,
        "paired_bootstrap_95_lower": lower,
        "bootstrap_95_lower_minimum": BOOTSTRAP_95_LOWER_MINIMUM,
        "bootstrap_passed": lower >= BOOTSTRAP_95_LOWER_MINIMUM,
    }:
        raise GateError("persisted noninferiority arithmetic changed")
    passed = median >= MEDIAN_SPEEDUP_MINIMUM and lower >= BOOTSTRAP_95_LOWER_MINIMUM
    expected_status = "PASSED" if passed else "REJECTED_NONINFERIORITY"
    if record.get("status") != expected_status or record.get("passed") is not passed:
        raise GateError("job status disagrees with exact noninferiority arithmetic")
    expected_returncode = 0 if passed else 3
    if record["transport"]["returncode"] != expected_returncode:
        raise GateError("job return code disagrees with exact noninferiority status")
    _validate_resources(record.get("resources"))
    old_source = record.get("old_source")
    if old_source != {
        "git_blob_oid": OLD_BLOB_OID,
        "source_sha256": OLD_SOURCE_SHA256,
        "loaded_in_memory": True,
        "temporary_source_written": False,
    }:
        raise GateError("old59 memory-source receipt changed")
    return expected_status


def _error_record(
    job: Job,
    manifest: Manifest,
    identity: Mapping[str, Any],
    token_sha256: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "schema": "act.hybridz.fprime.paired_noninferiority.worker.v1",
        "status": "ERROR",
        "passed": False,
        "case": job.case_name,
        "job": job.name,
        "mode": job.mode,
        "workers": job.workers,
        "input_sha256": {
            "onnx": manifest.cases[job.case_name].onnx_sha256,
            "vnnlib": manifest.cases[job.case_name].vnnlib_sha256,
        },
        "error_type": error_type,
        "manifest_sha256": manifest.sha256,
        "harness_sha256": identity["harness_sha256"],
        "source_bundle_sha256": identity["production_source_bundle_sha256"],
        "attempt_token_sha256": token_sha256,
        "scope": _scope(),
    }


def _validate_transport(value: Any, *, formal: bool) -> None:
    if type(value) is not dict:
        raise GateError("job lacks isolated transport metadata")
    integer_fields = (
        "returncode",
        "stdout_bytes",
        "stderr_bytes",
        "stdout_nonempty_lines",
        "stdout_unmarked_nonempty_lines",
        "isolated_result_records",
    )
    if any(type(value.get(name)) is not int for name in integer_fields):
        raise GateError("transport integer metadata is malformed")
    wall = value.get("child_wall_seconds")
    if type(wall) is not float or not math.isfinite(wall) or wall < 0.0:
        raise GateError("transport wall time is malformed")
    if type(value.get("timed_out")) is not bool:
        raise GateError("transport timeout flag is malformed")
    if not (
        _is_hex_digest(value.get("stdout_sha256"))
        and _is_hex_digest(value.get("stderr_sha256"))
    ):
        raise GateError("transport channel digest is malformed")
    if formal and not (
        value["timed_out"] is False
        and value["isolated_result_records"] == 1
        and value["stdout_nonempty_lines"] == 1
        and value["stdout_unmarked_nonempty_lines"] == 0
        and value["returncode"] in {0, 3}
    ):
        raise GateError("formal job did not cross a clean isolated channel")


def _append_event(value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("x" if exclusive else "a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(dict(value)) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_events(identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not EVENTS_PATH.exists():
        return []
    values = []
    with EVENTS_PATH.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"event line {number} is not JSON") from exc
            if type(value) is not dict:
                raise GateError(f"event line {number} is not an object")
            values.append(value)
    if not values or values[0] != {"event": "run_created", "identity": dict(identity)}:
        raise GateError("existing event ledger has a different frozen identity")
    return values


def _event_state(
    events: list[dict[str, Any]], manifest: Manifest, identity: Mapping[str, Any]
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    order = {job.name: index for index, job in enumerate(manifest.jobs)}
    jobs = {job.name: job for job in manifest.jobs}
    started: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    tokens: dict[str, str] = {}
    for event in events[1:]:
        kind = event.get("event")
        name = event.get("job")
        if name not in order:
            raise GateError("ledger contains a job outside the manifest")
        if kind == "job_attempt_started":
            if name in started or len(started) != order[name]:
                raise GateError("ledger contains a retry or job-order violation")
            if len(results) != len(started):
                raise GateError("ledger continued after an incomplete job")
            if any(record.get("status") != "PASSED" for record in results.values()):
                raise GateError("ledger continued after the first failed job")
            token = event.get("attempt_token_sha256")
            if not (
                event.get("ordinal") == order[name]
                and event.get("case") == jobs[name].case_name
                and event.get("mode") == jobs[name].mode
                and event.get("manifest_sha256") == manifest.sha256
                and event.get("source_bundle_sha256")
                == identity["production_source_bundle_sha256"]
                and _is_hex_digest(token)
            ):
                raise GateError("job attempt identity is malformed")
            started.add(name)
            tokens[name] = token
        elif kind == "job_result":
            if name not in started or name in results or len(started) != len(results) + 1:
                raise GateError("ledger has an orphan/duplicate/out-of-order result")
            if name != manifest.jobs[len(results)].name:
                raise GateError("job result order changed")
            record = event.get("record")
            if type(record) is not dict:
                raise GateError("job result is not an object")
            _validate_job_record(
                record,
                job=jobs[name],
                manifest=manifest,
                identity=identity,
                token_sha256=tokens[name],
            )
            results[name] = record
        else:
            raise GateError("ledger contains an unknown event")
    return started, results


def _summary(
    manifest: Manifest,
    identity: Mapping[str, Any],
    started: set[str],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    incomplete = [job.name for job in manifest.jobs if job.name in started and job.name not in results]
    errors = [name for name, record in results.items() if record.get("status") == "ERROR"]
    rejected = [
        name
        for name, record in results.items()
        if record.get("status") == "REJECTED_NONINFERIORITY"
    ]
    passed = [name for name, record in results.items() if record.get("status") == "PASSED"]
    if incomplete:
        status = "BLOCKED_INCOMPLETE_JOB_NO_RETRY"
    elif errors:
        status = "FAILED_CLOSED_ERROR"
    elif rejected:
        status = "STOP_LOSS_NONINFERIORITY_REJECTED"
    elif len(passed) == len(manifest.jobs):
        status = "COMPLETE_PASSED"
    else:
        status = "IN_PROGRESS"
    by_case = {
        case_name: {
            mode: results.get(f"{case_name}__{mode}", {}).get("status", "NOT_RUN")
            for mode in MODES
        }
        for case_name in CONTROL_ORDER
    }
    return {
        "schema": "act.hybridz.fprime.paired_noninferiority.v1",
        "status": status,
        "identity": dict(identity),
        "attempted_jobs": len(started),
        "completed_jobs": len(results),
        "passed_jobs": len(passed),
        "errors": errors,
        "rejected": rejected,
        "incomplete": incomplete,
        "case_mode_status": by_case,
        "results": [results[job.name] for job in manifest.jobs if job.name in results],
        "events_path": str(EVENTS_PATH.relative_to(ROOT)),
        "gate": {
            "jobs_required": 10,
            "each_case_single_and_four_thread": True,
            "each_median_speedup_minimum": MEDIAN_SPEEDUP_MINIMUM,
            "each_bootstrap_95_lower_minimum": BOOTSTRAP_95_LOWER_MINIMUM,
            "errors_maximum": 0,
            "result_conflicts_maximum": 0,
            "no_rerun_for_luck": True,
        },
        "scope": _scope(),
    }


def _authorize_worker(
    job_name: str,
    token: str,
    manifest: Manifest,
    identity: Mapping[str, Any],
) -> str:
    if type(token) is not str or len(token) < 32:
        raise GateError("worker lacks a parent attempt capability")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    events = _read_events(identity)
    if not events:
        raise GateError("worker has no persisted parent ledger")
    started, results = _event_state(events, manifest, identity)
    if job_name not in started or job_name in results:
        raise GateError("worker job is absent or already consumed")
    last = events[-1]
    if not (
        last.get("event") == "job_attempt_started"
        and last.get("job") == job_name
        and last.get("attempt_token_sha256") == digest
    ):
        raise GateError("worker capability is not bound to the active job")
    return digest


def _worker_entry(job_name: str) -> int:
    manifest = _build_manifest(verify_inputs=False)
    jobs = {job.name: job for job in manifest.jobs}
    if job_name not in jobs:
        raise GateError("worker job is outside the frozen manifest")
    job = jobs[job_name]
    identity = _identity(manifest)
    token = os.environ.pop(WORKER_TOKEN_ENV, "")
    token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""
    try:
        token_sha256 = _authorize_worker(job_name, token, manifest, identity)
        with contextlib.redirect_stdout(sys.stderr):
            record = _run_job(job, manifest.cases[job.case_name], manifest)
        record.update(
            {
                "harness_sha256": identity["harness_sha256"],
                "source_bundle_sha256": identity["production_source_bundle_sha256"],
                "attempt_token_sha256": token_sha256,
            }
        )
    except BaseException as exc:
        record = _error_record(
            job, manifest, identity, token_sha256, type(exc).__name__
        )
    print(RESULT_PREFIX + _canonical_json(record), flush=True)
    if record.get("status") == "PASSED":
        return 0
    if record.get("status") == "REJECTED_NONINFERIORITY":
        return 3
    return 2


def _decode_stdout(stdout: str) -> dict[str, Any]:
    nonempty = [line for line in stdout.splitlines() if line.strip()]
    marked = [line[len(RESULT_PREFIX) :] for line in nonempty if line.startswith(RESULT_PREFIX)]
    if len(nonempty) != 1 or len(marked) != 1:
        raise GateError("worker stdout is not one isolated result marker")
    try:
        value = json.loads(marked[0])
    except json.JSONDecodeError as exc:
        raise GateError("worker result marker is not JSON") from exc
    if type(value) is not dict:
        raise GateError("worker result marker is not an object")
    return value


def _transport(
    *,
    stdout: str,
    stderr: str,
    returncode: int,
    timed_out: bool,
    child_wall_seconds: float,
) -> dict[str, Any]:
    nonempty = [line for line in stdout.splitlines() if line.strip()]
    marked = [line for line in nonempty if line.startswith(RESULT_PREFIX)]
    return {
        "child_wall_seconds": float(child_wall_seconds),
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stdout_nonempty_lines": len(nonempty),
        "stdout_unmarked_nonempty_lines": len(nonempty) - len(marked),
        "isolated_result_records": len(marked),
    }


def _parent() -> int:
    manifest = _build_manifest(verify_inputs=True)
    identity = _identity(manifest)
    events = _read_events(identity)
    if not events:
        created = {"event": "run_created", "identity": identity}
        _append_event(created, exclusive=True)
        events = [created]
    started, results = _event_state(events, manifest, identity)
    if any(name in started and name not in results for name in started) or any(
        record.get("status") != "PASSED" for record in results.values()
    ):
        _atomic_json(SUMMARY_PATH, _summary(manifest, identity, started, results))
        return 2
    for job in manifest.jobs:
        if job.name in results:
            continue
        token = secrets.token_urlsafe(32)
        token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
        attempt = {
            "event": "job_attempt_started",
            "job": job.name,
            "case": job.case_name,
            "mode": job.mode,
            "ordinal": len(started),
            "manifest_sha256": manifest.sha256,
            "source_bundle_sha256": identity["production_source_bundle_sha256"],
            "attempt_token_sha256": token_sha256,
        }
        _append_event(attempt)
        events.append(attempt)
        started.add(job.name)
        env = dict(os.environ)
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONUNBUFFERED": "1",
                WORKER_TOKEN_ENV: token,
            }
        )
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-job", job.name]
        stdout = ""
        stderr = ""
        returncode = 2
        timed_out = False
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
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
            record = _decode_stdout(stdout)
            expected_code = {"PASSED": 0, "REJECTED_NONINFERIORITY": 3, "ERROR": 2}.get(
                record.get("status")
            )
            if expected_code is None or returncode != expected_code:
                raise GateError("worker return code disagrees with its status")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            record = _error_record(
                job, manifest, identity, token_sha256, type(exc).__name__
            )
        except BaseException as exc:
            record = _error_record(
                job, manifest, identity, token_sha256, type(exc).__name__
            )
        transport = _transport(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
            child_wall_seconds=time.monotonic() - child_started,
        )
        record["transport"] = transport
        try:
            _assert_sources()
            _validate_job_record(
                record,
                job=job,
                manifest=manifest,
                identity=identity,
                token_sha256=token_sha256,
            )
        except BaseException as exc:
            record = _error_record(
                job, manifest, identity, token_sha256, type(exc).__name__
            )
            record["transport"] = transport
            _validate_job_record(
                record,
                job=job,
                manifest=manifest,
                identity=identity,
                token_sha256=token_sha256,
            )
        event = {"event": "job_result", "job": job.name, "record": record}
        _append_event(event)
        events.append(event)
        results[job.name] = record
        _atomic_json(SUMMARY_PATH, _summary(manifest, identity, started, results))
        if record.get("status") != "PASSED":
            break
    summary = _summary(manifest, identity, started, results)
    _atomic_json(SUMMARY_PATH, summary)
    print(_canonical_json(summary), flush=True)
    return 0 if summary["status"] == "COMPLETE_PASSED" else 2


def _static_error_record(
    job: Job,
    manifest: Manifest,
    identity: Mapping[str, Any],
    token: str,
) -> dict[str, Any]:
    record = _error_record(job, manifest, identity, token, "StaticSyntheticError")
    record["transport"] = _transport(
        stdout="",
        stderr="",
        returncode=2,
        timed_out=False,
        child_wall_seconds=0.1,
    )
    return record


def _static_check() -> dict[str, Any]:
    manifest = _build_manifest(verify_inputs=True)
    identity = _identity(manifest)
    if not (
        len(manifest.cases) == 5
        and len(manifest.jobs) == 10
        and tuple(job.case_name for job in manifest.jobs[::2]) == CONTROL_ORDER
        and all(manifest.jobs[index].mode == "single" for index in range(0, 10, 2))
        and all(manifest.jobs[index].mode == "four_thread" for index in range(1, 10, 2))
    ):
        raise GateError("paired job order/cardinality changed")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_job"
    ]
    authorization_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_authorize_worker"
    ]
    if len(helper_calls) != 1 or len(authorization_calls) != 1:
        raise GateError("worker call/authorization shape changed")
    old_source = _git_blob_bytes()
    old_tree = ast.parse(old_source.decode("utf-8"))
    old_imports = {
        alias.name
        for node in ast.walk(old_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(old_tree)
        if isinstance(node, ast.ImportFrom)
    }
    if any(name in old_imports for name in ("onnxruntime", "random")):
        raise GateError("old blob imports a point-execution/random backend")
    compile(
        old_source.decode("utf-8"),
        f"<git-blob:{OLD_BLOB_OID}>",
        "exec",
        dont_inherit=True,
    )
    lower, index = _exact_bootstrap([1.0] * 5)
    if lower != 1.0 or index != 78 or EXACT_BOOTSTRAP_RESAMPLES != 3125:
        raise GateError("exact 3125 bootstrap contract changed")
    token = "b" * 64
    job0 = manifest.jobs[0]
    header = {"event": "run_created", "identity": identity}
    attempt0 = {
        "event": "job_attempt_started",
        "job": job0.name,
        "case": job0.case_name,
        "mode": job0.mode,
        "ordinal": 0,
        "manifest_sha256": manifest.sha256,
        "source_bundle_sha256": identity["production_source_bundle_sha256"],
        "attempt_token_sha256": token,
    }
    error0 = _static_error_record(job0, manifest, identity, token)
    result0 = {"event": "job_result", "job": job0.name, "record": error0}
    _event_state([header, attempt0, result0], manifest, identity)
    job1 = manifest.jobs[1]
    attempt1 = {
        "event": "job_attempt_started",
        "job": job1.name,
        "case": job1.case_name,
        "mode": job1.mode,
        "ordinal": 1,
        "manifest_sha256": manifest.sha256,
        "source_bundle_sha256": identity["production_source_bundle_sha256"],
        "attempt_token_sha256": "c" * 64,
    }
    try:
        _event_state([header, attempt0, result0, attempt1], manifest, identity)
    except GateError:
        pass
    else:
        raise GateError("resume continued after a failed job")
    forged = {
        "schema": "act.hybridz.fprime.paired_noninferiority.worker.v1",
        "job": job0.name,
        "status": "PASSED",
    }
    try:
        _event_state(
            [header, attempt0, {"event": "job_result", "job": job0.name, "record": forged}],
            manifest,
            identity,
        )
    except GateError:
        pass
    else:
        raise GateError("resume accepted a forged passing job")
    valid_stdout = RESULT_PREFIX + _canonical_json({"status": "ERROR"}) + "\n"
    _decode_stdout(valid_stdout)
    for hostile in ("noise\n" + valid_stdout, valid_stdout + valid_stdout, " " + valid_stdout):
        try:
            _decode_stdout(hostile)
        except GateError:
            pass
        else:
            raise GateError("stdout isolation accepted a hostile channel")
    return {
        "schema": "act.hybridz.fprime.paired_noninferiority.static.v1",
        "status": "STATIC_CPU_ONLY_PASS",
        "harness_sha256": _sha256(Path(__file__)),
        "manifest_sha256": manifest.sha256,
        "production_source_bundle_sha256": identity["production_source_bundle_sha256"],
        "old_git_blob_oid": OLD_BLOB_OID,
        "old_source_sha256": OLD_SOURCE_SHA256,
        "old_source_bytes": len(old_source),
        "old_source_compiled_not_executed": True,
        "temporary_old_source_written": False,
        "controls": 5,
        "jobs": 10,
        "warmup_pairs_per_job": 2,
        "measured_pairs_per_job": 5,
        "exact_bootstrap_resamples": 3125,
        "exact_bootstrap_lower_index": 78,
        "worker_authorization_callsites": 1,
        "worker_run_job_callsites": 1,
        "resume_hostile_selftests": 2,
        "stdout_hostile_selftests": 3,
        "gpu_initialized": False,
        "worker_started": False,
        "artifacts_created": False,
        "events_path_exists": EVENTS_PATH.exists(),
        "summary_path_exists": SUMMARY_PATH.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument("--worker-job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.static_check:
        print(_canonical_json(_static_check()), flush=True)
        return 0
    if args.worker_job is not None:
        return _worker_entry(args.worker_job)
    lock_path = EVENTS_PATH.with_suffix(EVENTS_PATH.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GateError("another paired gate parent already owns the run") from exc
        return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
