#!/usr/bin/env python3
"""One-shot production retention gate for the frozen F-prime path.

This is an offline, serial gate.  It first evaluates the frozen fixed-14
manifest and only then the remaining members of the frozen formal-59 set.
Overlapping cases are reused, so every benchmark instance is attempted at
most once in an artifact lifetime.  Each attempt invokes the current
production ``verify_once`` path through the already audited five-case
sentinel helper; no candidate/search algorithm is copied into this file.

The fixed-14 gate requires its four prior positives plus TinyImageNet iid143
to remain formally FALSIFIED (and therefore at least five positives total).
The retained-59 gate requires all 59 frozen formal positives to remain
FALSIFIED.  Any ERROR or required-positive regression is persisted and stops
the run immediately.  UNKNOWN never acquires proof authority.

``--static-check`` is CPU-only and creates no artifact.  Running without
arguments is the separately authorized CUDA action.
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
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import scratch_phase_projection_fprime_production_five_case_sentinel as five


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts/hybridz_largecls_gates"
FIXED14_SOURCE = (
    ARTIFACT_ROOT / "phase_projection_single_stream_float64_candidate_20260813.json"
)
RETAINED59_SOURCE = (
    ARTIFACT_ROOT / "phase_projection_gpu_csr_fixed400_20260814.jsonl"
)
LEGACY_VALIDATOR_SOURCE = (
    ROOT / "act/pipeline/verification/hybridz_phase_projection_fixed400.py"
)
EVENTS_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_fprime_production_retention_gate_20260814.events.jsonl"
)
SUMMARY_PATH = (
    ARTIFACT_ROOT / "phase_projection_fprime_production_retention_gate_20260814.json"
)
RESULT_PREFIX = "@@ACT_FPRIME_PRODUCTION_RETENTION_RESULT@@"
WORKER_TOKEN_ENV = "ACT_FPRIME_PRODUCTION_RETENTION_ATTEMPT_TOKEN"
WORKER_TIMEOUT_SECONDS = 45.0

OFFLINE_SOURCE_LOCKS: Mapping[str, str] = {
    str(FIXED14_SOURCE.relative_to(ROOT)): (
        "a1323cf69ac5e9f2f7189b8ce4ac96e67cd6e0b354f1c6a044a3a5433b7e16c9"
    ),
    str(RETAINED59_SOURCE.relative_to(ROOT)): (
        "749db4e400329598c23c3dd7c9b9863c291eb3d1ba556cdcfcfa879c58487b43"
    ),
    str(LEGACY_VALIDATOR_SOURCE.relative_to(ROOT)): (
        "d3e1825b9a8f8c8bb7f83cb8f08bdabd68f0c3fa32c379cac8ce08bb4c1e24c1"
    ),
}
FIVE_HELPER_SHA256 = (
    "3d168c9f29fae8343d5f101794d6b829cfbdf48a30dda57cea2cfd30ac595873"
)

# Filled after the deterministic source/input manifest is independently
# reconstructed.  It prevents a fresh artifact from silently accepting a
# different list even when no prior ledger exists.
EXPECTED_MANIFEST_SHA256 = (
    "3a55e4dee4345fead432641059fedd946924875cc26294212b32df2a9146afad"
)

# The fixed-14 source freezes names/order but not input digests.  These locks
# bind those 14 rows to the exact ONNX/VNNLIB inputs.  Retained-59 input locks
# are read from, and cryptographically bound to, the frozen fixed-400 JSONL.
FIXED14_INPUT_LOCKS: Mapping[str, tuple[str, str, str, str]] = {
    "cifar100_medium_iid2": (
        "CIFAR100_resnet_medium.onnx",
        "CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib",
        "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
        "33e795c8421b7b19125f32415adb9cee09b2f90cb83152c4cd3aa03810e91ec3",
    ),
    "cifar100_medium_iid11": (
        "CIFAR100_resnet_medium.onnx",
        "CIFAR100_resnet_medium_prop_idx_4752_sidx_1800_eps_0.0039.vnnlib",
        "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
        "370e0f7173fa4f645c742795c053b7c2cd4b81500778bf927ca8f74297fbe154",
    ),
    "cifar100_large_iid118": (
        "CIFAR100_resnet_large.onnx",
        "CIFAR100_resnet_large_prop_idx_9773_sidx_3718_eps_0.0039.vnnlib",
        "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16",
        "885ee9f008a908094ab00d00124436cc0483454b40c775c57fa9a43b2b3388e1",
    ),
    "tinyimagenet_medium_iid6": (
        "TinyImageNet_resnet_medium.onnx",
        "TinyImageNet_resnet_medium_prop_idx_9262_sidx_880_eps_0.0039.vnnlib",
        "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
        "e43da462a4b9758587532cd311a7e51b6dcb9ede5c656df0a5fbde2a76e7e2f4",
    ),
    "cifar100_medium_iid29": (
        "CIFAR100_resnet_medium.onnx",
        "CIFAR100_resnet_medium_prop_idx_4429_sidx_1471_eps_0.0039.vnnlib",
        "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
        "a0fd1008fcd127ac83840bc00c8f10e995e513bebb22aa9394b2e789d2333720",
    ),
    "cifar100_medium_iid50": (
        "CIFAR100_resnet_medium.onnx",
        "CIFAR100_resnet_medium_prop_idx_913_sidx_2404_eps_0.0039.vnnlib",
        "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
        "295075c963461299d128f9514cefbc8b99082d11e42ee5403c57d39ec689addc",
    ),
    "cifar100_large_iid113": (
        "CIFAR100_resnet_large.onnx",
        "CIFAR100_resnet_large_prop_idx_1439_sidx_8354_eps_0.0039.vnnlib",
        "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16",
        "4425caad857d90470a6db6f60de288d8a685676f2ae5a02076b6cd5baa65b3a3",
    ),
    "cifar100_large_iid110": (
        "CIFAR100_resnet_large.onnx",
        "CIFAR100_resnet_large_prop_idx_1063_sidx_7948_eps_0.0039.vnnlib",
        "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16",
        "f9b77b99fe82813df69b11e9ed2c378798c6d252277ab107ce3dd68429285b76",
    ),
    "cifar100_large_iid153": (
        "CIFAR100_resnet_large.onnx",
        "CIFAR100_resnet_large_prop_idx_4652_sidx_1371_eps_0.0039.vnnlib",
        "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16",
        "5b425e64c837085d070e219e8ac0e29012b30f2ef9f8e3af1ec7f5e00bc8e507",
    ),
    "tinyimagenet_medium_iid0": (
        "TinyImageNet_resnet_medium.onnx",
        "TinyImageNet_resnet_medium_prop_idx_1126_sidx_4974_eps_0.0039.vnnlib",
        "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
        "9497c3bdd8ade3804cfd9fe8d415941a18049753e31eede0adbf4260cdc280c1",
    ),
    "tinyimagenet_medium_iid17": (
        "TinyImageNet_resnet_medium.onnx",
        "TinyImageNet_resnet_medium_prop_idx_2063_sidx_8156_eps_0.0039.vnnlib",
        "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
        "0089ecbe47c9f4322ea8391c358e8a1dc7645e5832f6a35920434f84a07c0f8c",
    ),
    "tinyimagenet_medium_iid93": (
        "TinyImageNet_resnet_medium.onnx",
        "TinyImageNet_resnet_medium_prop_idx_3437_sidx_2708_eps_0.0039.vnnlib",
        "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
        "17428c1298247c3d94956164ca8784dec126b8a59bdfacd1f16556c080752651",
    ),
    "tinyimagenet_medium_iid143": (
        "TinyImageNet_resnet_medium.onnx",
        "TinyImageNet_resnet_medium_prop_idx_3553_sidx_3392_eps_0.0039.vnnlib",
        "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
        "812bec2c0362d92d123380df161e1da6d5addbc84a27304d0a079090e814f5c7",
    ),
    "tinyimagenet_medium_iid159": (
        "TinyImageNet_resnet_medium.onnx",
        "TinyImageNet_resnet_medium_prop_idx_9825_sidx_6883_eps_0.0039.vnnlib",
        "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
        "d8491f029cab63899953e6588c0a4587554c1acc7649e10370e395a9f83271d5",
    ),
}


class GateError(RuntimeError):
    """The offline gate cannot safely interpret or continue a run."""


@dataclass(frozen=True)
class Manifest:
    fixed14: tuple[str, ...]
    fixed14_original_falsified: tuple[str, ...]
    fixed14_required_falsified: tuple[str, ...]
    retained59: tuple[str, ...]
    execution_order: tuple[str, ...]
    cases: Mapping[str, five.Case]
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


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_hex_digest(value: Any) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _assert_offline_sources() -> None:
    for relative, expected in OFFLINE_SOURCE_LOCKS.items():
        if _sha256(ROOT / relative) != expected:
            raise GateError(f"offline manifest source changed: {relative}")
    helper_path = Path(five.__file__).resolve()
    if helper_path != ROOT / "scratch_phase_projection_fprime_production_five_case_sentinel.py":
        raise GateError("audited five-case helper resolved outside the workspace")
    if _sha256(helper_path) != FIVE_HELPER_SHA256:
        raise GateError("audited five-case helper changed")


def _case_parts(name: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"(cifar100_(?:medium|large)|tinyimagenet_medium)_iid([0-9]+)", name)
    if match is None:
        raise GateError(f"malformed case name: {name}")
    family = match.group(1)
    iid = int(match.group(2))
    benchmark = "tinyimagenet_2024" if family.startswith("tinyimagenet") else "cifar100_2024"
    if not 0 <= iid < 200:
        raise GateError(f"case iid is outside the frozen benchmark: {name}")
    return benchmark, family, iid


def _fixed14_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    data = json.loads(FIXED14_SOURCE.read_text(encoding="utf-8"))
    section = data.get("fixed_14")
    if type(section) is not dict:
        raise GateError("fixed-14 source lacks its frozen section")
    candidate = section.get("candidate_terminal_verified")
    previous = section.get("previous_terminal_verified")
    new = section.get("new")
    remaining = section.get("remaining_unknown")
    if not all(type(value) is list for value in (candidate, previous, new, remaining)):
        raise GateError("fixed-14 source has malformed lists")
    if not all(type(name) is str for name in candidate + previous + new + remaining):
        raise GateError("fixed-14 source has a malformed case")
    fixed = tuple(candidate + remaining)
    original = tuple(candidate)
    if not (
        len(fixed) == 14
        and len(set(fixed)) == 14
        and len(original) == 4
        and tuple(previous) == original[:3]
        and tuple(new) == ("tinyimagenet_medium_iid6",)
        and original[-1] == "tinyimagenet_medium_iid6"
    ):
        raise GateError("fixed-14 source no longer describes the frozen 4+10 set")
    if set(FIXED14_INPUT_LOCKS) != set(fixed):
        raise GateError("fixed-14 input locks disagree with the source")
    return fixed, original


def _legacy_validated_status(record: Mapping[str, Any]) -> str:
    """Reproduce the locked fixed-400 selection predicate, offline only."""

    projection = record.get("phase_projection")
    if type(projection) is not dict:
        return "ERROR"
    forbidden = (
        "input_sampling_used",
        "pgd_used",
        "concrete_onnx_execution_used",
        "bab_used",
        "backward_used",
        "dual_tightening_used",
    )
    if any(projection.get(key) is not False for key in forbidden):
        return "ERROR"
    if record.get("status") == "VerifyStatus.FALSIFIED":
        candidate = projection.get("candidate_receipt")
        if type(candidate) is not dict:
            return "ERROR"
        if not (
            record.get("has_counterexample") is True
            and projection.get("status") == "FALSIFIED"
            and projection.get("verifier_owned_proof_authority") is True
            and candidate.get("status") == "singleton_verified"
            and candidate.get("all_unstable_exact") is True
            and type(candidate.get("triangle_rows")) is int
            and candidate.get("triangle_rows") == 0
            and type(candidate.get("phase_retries")) is int
            and candidate.get("phase_retries") == 0
            and type(candidate.get("property_row_retries")) is int
            and candidate.get("property_row_retries") == 0
            and candidate.get("proof_authority") is False
            and candidate.get("verdict_authority") is False
        ):
            return "ERROR"
        return "FALSIFIED"
    if record.get("status") == "VerifyStatus.UNKNOWN":
        if not (
            record.get("has_counterexample") is False
            and projection.get("status") == "UNKNOWN"
            and projection.get("verifier_owned_proof_authority") is False
            and "candidate_receipt" not in projection
        ):
            return "ERROR"
        return "UNKNOWN"
    return "ERROR"


def _retained59_rows() -> tuple[
    tuple[str, ...], dict[str, dict[str, Any]]
]:
    lines = RETAINED59_SOURCE.read_text(encoding="utf-8").splitlines()
    if len(lines) != 400 or any(not line.strip() for line in lines):
        raise GateError("fixed-400 source is not exactly 400 JSONL records")
    retained: list[str] = []
    provenance: dict[str, dict[str, Any]] = {}
    observed: set[str] = set()
    for ordinal, line in enumerate(lines):
        record = json.loads(line)
        if type(record) is not dict:
            raise GateError("fixed-400 source contains a non-object")
        if ordinal < 100:
            expected = f"cifar100_medium_iid{ordinal}"
        elif ordinal < 200:
            expected = f"cifar100_large_iid{ordinal}"
        else:
            expected = f"tinyimagenet_medium_iid{ordinal - 200}"
        name = record.get("case")
        _benchmark, expected_family, expected_iid = _case_parts(expected)
        recomputed = _legacy_validated_status(record)
        if (
            record.get("schema") != "act.hybridz.phase_projection_fixed400.worker.v1"
            or name != expected
            or record.get("family") != expected_family
            or type(record.get("iid")) is not int
            or record.get("iid") != expected_iid
            or name in observed
            or record.get("validated_status") not in {"FALSIFIED", "UNKNOWN"}
            or recomputed != record.get("validated_status")
        ):
            raise GateError("fixed-400 source violates its frozen row order/schema")
        observed.add(name)
        if recomputed != "FALSIFIED":
            continue
        projection = record.get("phase_projection")
        hashes = record.get("input_sha256")
        if not (
            record.get("status") == "VerifyStatus.FALSIFIED"
            and record.get("has_counterexample") is True
            and type(projection) is dict
            and projection.get("status") == "FALSIFIED"
            and projection.get("verifier_owned_proof_authority") is True
            and type(hashes) is dict
            and _is_hex_digest(hashes.get("onnx"))
            and _is_hex_digest(hashes.get("vnnlib"))
        ):
            raise GateError("a retained source row lacks its formal positive identity")
        retained.append(name)
        provenance[name] = {
            "input_lock": (
                Path(str(record.get("onnx"))).name,
                Path(str(record.get("vnnlib"))).name,
                hashes["onnx"],
                hashes["vnnlib"],
            ),
            "source_line": ordinal + 1,
            "source_record_sha256": hashlib.sha256(
                line.encode("utf-8")
            ).hexdigest(),
            "source_family": record.get("family"),
            "source_iid": record.get("iid"),
        }
    if len(retained) != 59 or len(set(retained)) != 59:
        raise GateError("validated FALSIFIED filter did not produce exactly 59 cases")
    return tuple(retained), provenance


def _csv_rows(benchmark: str) -> list[list[str]]:
    path = five.BENCHMARK_ROOT / benchmark / "instances.csv"
    if _sha256(path) != five.CSV_LOCKS[benchmark]:
        raise GateError(f"{benchmark} instances.csv changed")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _build_manifest(*, verify_all_inputs: bool, enforce_digest: bool = True) -> Manifest:
    _assert_offline_sources()
    fixed14, original = _fixed14_names()
    retained59, retained_provenance = _retained59_rows()
    required = tuple(dict.fromkeys((*original, "tinyimagenet_medium_iid143")))
    execution = tuple(dict.fromkeys((*fixed14, *retained59)))
    if not (
        len(required) == 5
        and len(execution) == 69
        and len(set(fixed14) & set(retained59)) == 4
        and set(fixed14) & set(retained59) == set(original)
    ):
        raise GateError("fixed-14/retained-59 overlap is not the frozen four cases")

    rows_by_benchmark = {
        benchmark: _csv_rows(benchmark)
        for benchmark in ("cifar100_2024", "tinyimagenet_2024")
    }
    cases: dict[str, five.Case] = {}
    payload_cases: list[dict[str, Any]] = []
    expected_input_paths: dict[Path, str] = {}
    for name in execution:
        benchmark, family, iid = _case_parts(name)
        retained_entry = retained_provenance.get(name)
        source_lock = (
            retained_entry["input_lock"] if retained_entry is not None else None
        )
        fixed_lock = FIXED14_INPUT_LOCKS.get(name)
        if source_lock is not None and fixed_lock is not None and source_lock != fixed_lock:
            raise GateError(f"overlap input identity disagrees for {name}")
        lock = source_lock or fixed_lock
        if lock is None:
            raise GateError(f"case lacks a frozen input identity: {name}")
        model_name, spec_name, onnx_sha256, vnnlib_sha256 = lock
        rows = rows_by_benchmark[benchmark]
        if iid >= len(rows) or len(rows[iid]) != 3:
            raise GateError(f"instances.csv row is malformed for {name}")
        model_rel, spec_rel, _timeout = rows[iid]
        model_path = (five.BENCHMARK_ROOT / benchmark / model_rel).resolve()
        spec_path = (five.BENCHMARK_ROOT / benchmark / spec_rel).resolve()
        root = (five.BENCHMARK_ROOT / benchmark).resolve()
        expected_size = "large" if family == "cifar100_large" else "medium"
        if not (
            model_path.is_relative_to(root)
            and spec_path.is_relative_to(root)
            and model_path.name == model_name
            and spec_path.name == spec_name
            and expected_size in model_path.name.lower()
        ):
            raise GateError(f"CSV/input manifest identity changed for {name}")
        case = five.Case(
            name,
            benchmark,
            iid,
            model_name,
            spec_name,
            onnx_sha256,
            vnnlib_sha256,
        )
        for path, expected_sha256 in (
            (model_path, onnx_sha256),
            (spec_path, vnnlib_sha256),
        ):
            prior_sha256 = expected_input_paths.setdefault(path, expected_sha256)
            if prior_sha256 != expected_sha256:
                raise GateError(f"input path has conflicting hashes: {path.name}")
        cases[name] = case
        payload_cases.append(
            {
                "execution_position": len(payload_cases),
                "name": name,
                "benchmark": benchmark,
                "family": family,
                "iid": iid,
                "fixed14_position": (
                    fixed14.index(name) if name in set(fixed14) else None
                ),
                "retained59_position": (
                    retained59.index(name) if name in set(retained59) else None
                ),
                "instances_csv_raw_row": list(rows[iid]),
                "onnx_relpath": str(model_path.relative_to(root)),
                "model_name": model_name,
                "vnnlib_relpath": str(spec_path.relative_to(root)),
                "spec_name": spec_name,
                "onnx_sha256": onnx_sha256,
                "vnnlib_sha256": vnnlib_sha256,
                "baseline_expected_status": (
                    "FALSIFIED" if retained_entry is not None else None
                ),
                "baseline_source_line": (
                    retained_entry["source_line"] if retained_entry is not None else None
                ),
                "baseline_source_record_sha256": (
                    retained_entry["source_record_sha256"]
                    if retained_entry is not None
                    else None
                ),
            }
        )
    if verify_all_inputs:
        for path, expected_sha256 in expected_input_paths.items():
            if not path.is_file() or _sha256(path) != expected_sha256:
                raise GateError(f"frozen benchmark input changed: {path.name}")
    payload: dict[str, Any] = {
        "schema": "act.hybridz.fprime.production_retention_manifest.v1",
        "order_source": {
            "fixed14": "fixed_14.candidate_terminal_verified_then_remaining_unknown",
            "retained59": "fixed400_jsonl_line_order_after_locked_legacy_validation",
        },
        "selection_rule_id": "locked_fixed400_validated_status_v1_strict_integer_replay",
        "fixed14": list(fixed14),
        "fixed14_original_falsified": list(original),
        "fixed14_required_falsified": list(required),
        "retained59": list(retained59),
        "execution_order_unique": list(execution),
        "overlap_reused_once": [name for name in fixed14 if name in set(retained59)],
        "cases": payload_cases,
        "offline_source_sha256": dict(OFFLINE_SOURCE_LOCKS),
    }
    manifest_sha256 = _digest(payload)
    if enforce_digest and manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise GateError(
            "retention manifest digest differs from the frozen value: "
            + manifest_sha256
        )
    return Manifest(
        fixed14=fixed14,
        fixed14_original_falsified=original,
        fixed14_required_falsified=required,
        retained59=retained59,
        execution_order=execution,
        cases=cases,
        payload=payload,
        sha256=manifest_sha256,
    )


def _identity(manifest: Manifest) -> dict[str, Any]:
    source_bundle = five._source_bundle()
    return {
        "schema": "act.hybridz.fprime.production_retention_gate.identity.v1",
        "harness_sha256": _sha256(Path(__file__)),
        "five_case_helper_sha256": FIVE_HELPER_SHA256,
        "manifest_sha256": manifest.sha256,
        "manifest": dict(manifest.payload),
        "production_source_bundle": source_bundle,
        "production_source_bundle_sha256": five._bundle_sha256(source_bundle),
        "request_lp_phase_seconds": five.REQUEST_SECONDS,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "worker_timeout_role": "transport_hard_limit_not_request_or_lp_deadline",
        "runtime_paths": 1,
        "attempts_per_unique_case_max": 1,
        "fixed14_before_retained59": True,
        "external_labels_read_by_production": False,
        "randomness_scope": "ipc_attempt_capability_only_never_production_or_selector",
    }


def _case_membership(name: str, manifest: Manifest) -> dict[str, Any]:
    fixed = name in set(manifest.fixed14)
    retained = name in set(manifest.retained59)
    return {
        "primary_stage": "fixed14" if fixed else "retained59",
        "fixed14_member": fixed,
        "retained59_member": retained,
        "fixed14_required_falsified": name
        in set(manifest.fixed14_required_falsified),
        "expected_status_supplied_to_production": False,
        "external_label_supplied_to_production": False,
    }


def _offline_scope() -> dict[str, Any]:
    return {
        "production_verify_once": True,
        "audited_five_case_helper_only": True,
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
    }


def _error_record(
    case: five.Case,
    manifest: Manifest,
    identity: Mapping[str, Any],
    token_sha256: str,
    *,
    status: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "schema": "act.hybridz.fprime.production_retention_gate.worker.v1",
        "case": case.name,
        "benchmark": case.benchmark,
        "iid": case.iid,
        "input_sha256": {
            "onnx": case.onnx_sha256,
            "vnnlib": case.vnnlib_sha256,
        },
        "source_bundle_sha256": identity["production_source_bundle_sha256"],
        "manifest_sha256": manifest.sha256,
        "five_case_helper_sha256": FIVE_HELPER_SHA256,
        "attempt_token_sha256": token_sha256,
        "status": status,
        "has_counterexample": False,
        "validated_status": "ERROR",
        "error_type": error_type,
        "retention_membership": _case_membership(case.name, manifest),
        "scope": _offline_scope(),
    }


def _validate_transport(transport: Any, *, formal: bool) -> None:
    if type(transport) is not dict:
        raise GateError("persisted record lacks isolated transport metadata")
    integer_fields = (
        "returncode",
        "stdout_bytes",
        "stderr_bytes",
        "stdout_nonempty_lines",
        "stdout_unmarked_nonempty_lines",
        "isolated_result_records",
    )
    if any(type(transport.get(name)) is not int for name in integer_fields):
        raise GateError("transport metadata has a malformed integer")
    if type(transport.get("child_wall_seconds")) is not float or not (
        0.0 <= transport["child_wall_seconds"] <= WORKER_TIMEOUT_SECONDS + 5.0
    ):
        raise GateError("transport wall time is malformed")
    if type(transport.get("timed_out")) is not bool:
        raise GateError("transport timeout flag is malformed")
    if not (
        _is_hex_digest(transport.get("stdout_sha256"))
        and _is_hex_digest(transport.get("stderr_sha256"))
    ):
        raise GateError("transport digest is malformed")
    if formal and not (
        transport["returncode"] == 0
        and transport["timed_out"] is False
        and transport["isolated_result_records"] == 1
        and transport["stdout_nonempty_lines"] == 1
        and transport["stdout_unmarked_nonempty_lines"] == 0
    ):
        raise GateError("formal result did not cross a clean isolated channel")


def _revalidate_record(
    record: Mapping[str, Any],
    *,
    case: five.Case,
    manifest: Manifest,
    identity: Mapping[str, Any],
    token_sha256: str,
) -> str:
    if record.get("schema") != "act.hybridz.fprime.production_retention_gate.worker.v1":
        raise GateError("worker record schema is not frozen")
    if record.get("case") != case.name or record.get("benchmark") != case.benchmark:
        raise GateError("worker record case identity changed")
    if type(record.get("iid")) is not int or record["iid"] != case.iid:
        raise GateError("worker record iid identity changed")
    if not _is_hex_digest(token_sha256) or record.get("attempt_token_sha256") != token_sha256:
        raise GateError("worker record is not bound to its one-shot attempt")
    if not (
        record.get("manifest_sha256") == manifest.sha256
        and record.get("five_case_helper_sha256") == FIVE_HELPER_SHA256
        and record.get("source_bundle_sha256")
        == identity["production_source_bundle_sha256"]
        and record.get("input_sha256")
        == {"onnx": case.onnx_sha256, "vnnlib": case.vnnlib_sha256}
        and record.get("retention_membership")
        == _case_membership(case.name, manifest)
        and record.get("scope") == _offline_scope()
    ):
        raise GateError("worker record source/input/scope identity changed")
    claimed = record.get("validated_status")
    if claimed in {"FALSIFIED", "UNKNOWN"}:
        observed = five._validated_status(record)
        if observed != claimed:
            raise GateError("worker and parent authority validation disagree")
        _validate_transport(record.get("transport"), formal=True)
        return observed
    if claimed == "ERROR":
        if not (
            record.get("has_counterexample") is False
            and type(record.get("status")) is str
            and bool(record["status"])
            and type(record.get("error_type")) is str
            and bool(record["error_type"])
        ):
            raise GateError("fail-closed worker error is malformed")
        _validate_transport(record.get("transport"), formal=False)
        return "ERROR"
    raise GateError("worker record has an unsupported validated status")


def _blocking_result(name: str, status: str, manifest: Manifest) -> bool:
    if status == "ERROR":
        return True
    required = set(manifest.fixed14_required_falsified) | set(manifest.retained59)
    return name in required and status != "FALSIFIED"


def _fixed14_passed(results: Mapping[str, Mapping[str, Any]], manifest: Manifest) -> bool:
    if not all(name in results for name in manifest.fixed14):
        return False
    statuses = {name: results[name].get("validated_status") for name in manifest.fixed14}
    return (
        all(statuses[name] == "FALSIFIED" for name in manifest.fixed14_required_falsified)
        and sum(status == "FALSIFIED" for status in statuses.values()) >= 5
        and all(status != "ERROR" for status in statuses.values())
    )


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
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_events(identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not EVENTS_PATH.exists():
        return []
    values: list[dict[str, Any]] = []
    with EVENTS_PATH.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"event ledger line {number} is not JSON") from exc
            if type(value) is not dict:
                raise GateError(f"event ledger line {number} is not an object")
            values.append(value)
    expected_header = {"event": "run_created", "identity": dict(identity)}
    if not values or values[0] != expected_header:
        raise GateError("existing ledger has a different frozen identity")
    return values


def _event_state(
    events: list[dict[str, Any]], manifest: Manifest, identity: Mapping[str, Any]
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    started: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    tokens: dict[str, str] = {}
    order = {name: ordinal for ordinal, name in enumerate(manifest.execution_order)}
    for event in events[1:]:
        kind = event.get("event")
        name = event.get("case")
        if name not in order:
            raise GateError("event ledger contains a case outside the manifest")
        case = manifest.cases[name]
        if kind == "case_attempt_started":
            if name in started or len(started) != order[name]:
                raise GateError("event ledger contains a retry or order violation")
            if len(results) != len(started):
                raise GateError("event ledger continued after an incomplete attempt")
            if any(
                _blocking_result(prior, result.get("validated_status"), manifest)
                for prior, result in results.items()
            ):
                raise GateError("event ledger continued after a core blocker")
            if order[name] >= len(manifest.fixed14) and not _fixed14_passed(results, manifest):
                raise GateError("retained-59 began before fixed-14 passed")
            token = event.get("attempt_token_sha256")
            if not (
                event.get("ordinal") == order[name]
                and event.get("primary_stage")
                == _case_membership(name, manifest)["primary_stage"]
                and event.get("manifest_sha256") == manifest.sha256
                and event.get("source_bundle_sha256")
                == identity["production_source_bundle_sha256"]
                and _is_hex_digest(token)
            ):
                raise GateError("attempt event identity is malformed")
            started.add(name)
            tokens[name] = token
        elif kind == "case_result":
            if name not in started or name in results or len(started) != len(results) + 1:
                raise GateError("event ledger has an orphan/duplicate/out-of-order result")
            if name != manifest.execution_order[len(results)]:
                raise GateError("result order differs from the frozen manifest")
            record = event.get("record")
            if type(record) is not dict:
                raise GateError("event ledger result is not an object")
            _revalidate_record(
                record,
                case=case,
                manifest=manifest,
                identity=identity,
                token_sha256=tokens[name],
            )
            results[name] = record
        else:
            raise GateError("event ledger contains an unknown event")
    return started, results


def _summary(
    manifest: Manifest,
    identity: Mapping[str, Any],
    started: set[str],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    incomplete = [name for name in manifest.execution_order if name in started and name not in results]
    errors = [name for name, record in results.items() if record.get("validated_status") == "ERROR"]
    fixed_required_missing = [
        name
        for name in manifest.fixed14_required_falsified
        if name in results and results[name].get("validated_status") != "FALSIFIED"
    ]
    retained_missing = [
        name
        for name in manifest.retained59
        if name in results and results[name].get("validated_status") != "FALSIFIED"
    ]
    fixed_complete = all(name in results for name in manifest.fixed14)
    fixed_falsified = [
        name for name in manifest.fixed14 if results.get(name, {}).get("validated_status") == "FALSIFIED"
    ]
    retained_complete = all(name in results for name in manifest.retained59)
    retained_falsified = [
        name for name in manifest.retained59 if results.get(name, {}).get("validated_status") == "FALSIFIED"
    ]
    new_falsified = [
        name
        for name in manifest.execution_order
        if name not in set(manifest.retained59)
        and results.get(name, {}).get("validated_status") == "FALSIFIED"
    ]
    if incomplete:
        status = "BLOCKED_INCOMPLETE_ATTEMPT_NO_RETRY"
    elif errors:
        status = "FAILED_CLOSED_ERROR"
    elif fixed_required_missing:
        status = "STOP_LOSS_FIXED14_REQUIRED_FALSIFIED_MISSING"
    elif not fixed_complete:
        status = "IN_PROGRESS_FIXED14"
    elif not _fixed14_passed(results, manifest):
        status = "STOP_LOSS_FIXED14_THRESHOLD_NOT_MET"
    elif retained_missing:
        status = "STOP_LOSS_RETAINED59_REGRESSION"
    elif retained_complete and len(retained_falsified) == 59:
        status = "COMPLETE_RETAINED59"
    else:
        status = "IN_PROGRESS_RETAINED59"
    return {
        "schema": "act.hybridz.fprime.production_retention_gate.v1",
        "status": status,
        "identity": dict(identity),
        "attempted_unique": len(started),
        "completed_unique": len(results),
        "errors": errors,
        "incomplete_attempts": incomplete,
        "fixed14": {
            "complete": fixed_complete,
            "passed": _fixed14_passed(results, manifest),
            "falsified": len(fixed_falsified),
            "falsified_cases": fixed_falsified,
            "required_falsified": list(manifest.fixed14_required_falsified),
            "required_missing": fixed_required_missing,
            "minimum_falsified": 5,
        },
        "retained59": {
            "complete": retained_complete,
            "retained": len(retained_falsified),
            "required": 59,
            "regressions": retained_missing,
            "overlap_reused_not_rerun": [
                name for name in manifest.fixed14 if name in set(manifest.retained59)
            ],
        },
        "new_falsified_outside_retained59": new_falsified,
        "results": [
            results[name] for name in manifest.execution_order if name in results
        ],
        "events_path": str(EVENTS_PATH.relative_to(ROOT)),
        "scope": {
            "production_verify_once": True,
            "strict_serial_fixed14_then_retained59": True,
            "unique_cases_attempted_at_most_once": True,
            "overlap_reused": True,
            "timing_authority": False,
            "formal_fixed400_changed": False,
            "disconnected_algorithm_reused": False,
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


def _authorize_worker(
    name: str,
    token: str,
    manifest: Manifest,
    identity: Mapping[str, Any],
) -> str:
    if type(token) is not str or len(token) < 32:
        raise GateError("worker lacks a parent attempt capability")
    token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
    events = _read_events(identity)
    if not events:
        raise GateError("worker has no persisted parent ledger")
    started, results = _event_state(events, manifest, identity)
    if name not in started or name in results:
        raise GateError("worker attempt is absent or already consumed")
    last = events[-1]
    if not (
        last.get("event") == "case_attempt_started"
        and last.get("case") == name
        and last.get("attempt_token_sha256") == token_sha256
    ):
        raise GateError("worker capability is not bound to the active case")
    return token_sha256


def _decorate_formal_record(
    record: dict[str, Any],
    case: five.Case,
    manifest: Manifest,
    identity: Mapping[str, Any],
    token_sha256: str,
) -> dict[str, Any]:
    record.update(
        {
            "schema": "act.hybridz.fprime.production_retention_gate.worker.v1",
            "case": case.name,
            "benchmark": case.benchmark,
            "iid": case.iid,
            "manifest_sha256": manifest.sha256,
            "five_case_helper_sha256": FIVE_HELPER_SHA256,
            "source_bundle_sha256": identity["production_source_bundle_sha256"],
            "attempt_token_sha256": token_sha256,
            "retention_membership": _case_membership(case.name, manifest),
            "scope": _offline_scope(),
        }
    )
    return record


def _worker_entry(name: str) -> int:
    manifest = _build_manifest(verify_all_inputs=False)
    if name not in manifest.cases:
        raise GateError("worker case is outside the frozen manifest")
    case = manifest.cases[name]
    identity = _identity(manifest)
    token = os.environ.pop(WORKER_TOKEN_ENV, "")
    token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""
    try:
        token_sha256 = _authorize_worker(name, token, manifest, identity)
        with contextlib.redirect_stdout(sys.stderr):
            record = five._run_production_case(case)
        record = _decorate_formal_record(
            record, case, manifest, identity, token_sha256
        )
    except Exception as exc:
        record = _error_record(
            case,
            manifest,
            identity,
            token_sha256,
            status="worker_error",
            error_type=type(exc).__name__,
        )
    print(RESULT_PREFIX + _canonical_json(record), flush=True)
    return 0 if record.get("validated_status") in {"FALSIFIED", "UNKNOWN"} else 2


def _decode_worker_stdout(stdout: str) -> dict[str, Any]:
    nonempty = [line for line in stdout.splitlines() if line.strip()]
    marked = [line[len(RESULT_PREFIX) :] for line in nonempty if line.startswith(RESULT_PREFIX)]
    unmarked = [line for line in nonempty if not line.startswith(RESULT_PREFIX)]
    if len(nonempty) != 1 or len(marked) != 1 or unmarked:
        raise GateError("isolated worker emitted stdout outside its single result marker")
    try:
        value = json.loads(marked[0])
    except json.JSONDecodeError as exc:
        raise GateError("isolated worker marker is not JSON") from exc
    if type(value) is not dict:
        raise GateError("isolated worker marker is not an object")
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
    manifest = _build_manifest(verify_all_inputs=True)
    identity = _identity(manifest)
    events = _read_events(identity)
    if not events:
        created = {"event": "run_created", "identity": identity}
        _append_event(created, exclusive=True)
        events = [created]
    started, results = _event_state(events, manifest, identity)
    if any(name in started and name not in results for name in started) or any(
        _blocking_result(name, record.get("validated_status"), manifest)
        for name, record in results.items()
    ):
        _atomic_json(SUMMARY_PATH, _summary(manifest, identity, started, results))
        return 2

    for name in manifest.execution_order:
        if name in results:
            continue
        case = manifest.cases[name]
        if name not in manifest.fixed14 and not _fixed14_passed(results, manifest):
            raise GateError("retained-59 cannot start before fixed-14 passes")
        token = secrets.token_urlsafe(32)
        token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
        attempt = {
            "event": "case_attempt_started",
            "case": name,
            "ordinal": len(started),
            "primary_stage": _case_membership(name, manifest)["primary_stage"],
            "manifest_sha256": manifest.sha256,
            "source_bundle_sha256": identity["production_source_bundle_sha256"],
            "attempt_token_sha256": token_sha256,
        }
        _append_event(attempt)
        events.append(attempt)
        started.add(name)
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
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-case", name]
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
            stdout, stderr, returncode = (
                completed.stdout,
                completed.stderr,
                completed.returncode,
            )
            record = _decode_worker_stdout(stdout)
            expected_returncode = (
                0
                if record.get("validated_status") in {"FALSIFIED", "UNKNOWN"}
                else 2
            )
            if returncode != expected_returncode:
                raise GateError("worker return code disagrees with its claimed status")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            record = _error_record(
                case,
                manifest,
                identity,
                token_sha256,
                status="worker_timeout",
                error_type=type(exc).__name__,
            )
        except Exception as exc:
            record = _error_record(
                case,
                manifest,
                identity,
                token_sha256,
                status="worker_transport_error",
                error_type=type(exc).__name__,
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
            five._source_bundle()
            five._resolve(case)
            _revalidate_record(
                record,
                case=case,
                manifest=manifest,
                identity=identity,
                token_sha256=token_sha256,
            )
        except Exception as exc:
            record = _error_record(
                case,
                manifest,
                identity,
                token_sha256,
                status="worker_parent_revalidation_error",
                error_type=type(exc).__name__,
            )
            record["transport"] = transport
            _revalidate_record(
                record,
                case=case,
                manifest=manifest,
                identity=identity,
                token_sha256=token_sha256,
            )
        result_event = {"event": "case_result", "case": name, "record": record}
        _append_event(result_event)
        events.append(result_event)
        results[name] = record
        _atomic_json(SUMMARY_PATH, _summary(manifest, identity, started, results))
        if _blocking_result(name, record.get("validated_status"), manifest):
            break
    summary = _summary(manifest, identity, started, results)
    _atomic_json(SUMMARY_PATH, summary)
    print(_canonical_json(summary), flush=True)
    return 0 if summary["status"] == "COMPLETE_RETAINED59" else 2


def _synthetic_unknown(
    case: five.Case,
    manifest: Manifest,
    identity: Mapping[str, Any],
    token_sha256: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": "act.hybridz.fprime.production_retention_gate.worker.v1",
        "case": case.name,
        "benchmark": case.benchmark,
        "iid": case.iid,
        "input_sha256": {"onnx": case.onnx_sha256, "vnnlib": case.vnnlib_sha256},
        "source_bundle_sha256": identity["production_source_bundle_sha256"],
        "manifest_sha256": manifest.sha256,
        "five_case_helper_sha256": FIVE_HELPER_SHA256,
        "attempt_token_sha256": token_sha256,
        "status": "VerifyStatus.UNKNOWN",
        "has_counterexample": False,
        "validated_status": "UNKNOWN",
        "phase_projection": {
            "enabled": True,
            "configured_seconds": 10.0,
            "input_sampling_used": False,
            "pgd_used": False,
            "concrete_onnx_execution_used": False,
            "bab_used": False,
            "backward_used": False,
            "dual_tightening_used": False,
            "status": "UNKNOWN",
            "verifier_owned_proof_authority": False,
            "reason": "synthetic static fail-closed record",
        },
        "owner_audit": {
            "logical_owner_instances": 0,
            "logical_owner_close_calls": 0,
            "logical_owner_final_states": [],
            "native_owner_instances": 0,
            "native_run_calls": 0,
            "native_clear_calls": 0,
            "native_clear_model_calls": 0,
            "dual_ray_exist_calls": 0,
            "dual_ray_calls": 0,
        },
        "retention_membership": _case_membership(case.name, manifest),
        "scope": _offline_scope(),
        "transport": _transport(
            stdout=RESULT_PREFIX + "{}\n",
            stderr="",
            returncode=0,
            timed_out=False,
            child_wall_seconds=0.1,
        ),
    }
    return record


def _static_check() -> dict[str, Any]:
    manifest = _build_manifest(verify_all_inputs=True)
    identity = _identity(manifest)
    if not (
        len(manifest.fixed14) == 14
        and len(manifest.fixed14_required_falsified) == 5
        and len(manifest.retained59) == 59
        and len(manifest.execution_order) == 69
    ):
        raise GateError("static manifest cardinality check failed")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    if any("probe" in name or "disconnected" in name for name in imports):
        raise GateError("retention gate imports a disconnected/probe implementation")
    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "five"
        and node.func.attr == "_run_production_case"
    ]
    if len(helper_calls) != 1:
        raise GateError("retention worker must have one production helper callsite")
    helper_tree = ast.parse(Path(five.__file__).read_text(encoding="utf-8"))
    verify_calls = [
        node
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "verify_once"
    ]
    if len(verify_calls) != 1:
        raise GateError("audited helper no longer has exactly one verify_once callsite")

    token = "b" * 64
    case0 = manifest.cases[manifest.execution_order[0]]
    unknown = _synthetic_unknown(case0, manifest, identity, token)
    _revalidate_record(
        unknown,
        case=case0,
        manifest=manifest,
        identity=identity,
        token_sha256=token,
    )
    header = {"event": "run_created", "identity": identity}
    attempt0 = {
        "event": "case_attempt_started",
        "case": case0.name,
        "ordinal": 0,
        "primary_stage": "fixed14",
        "manifest_sha256": manifest.sha256,
        "source_bundle_sha256": identity["production_source_bundle_sha256"],
        "attempt_token_sha256": token,
    }
    result0 = {"event": "case_result", "case": case0.name, "record": unknown}
    _event_state([header, attempt0, result0], manifest, identity)
    case1 = manifest.cases[manifest.execution_order[1]]
    attempt1 = {
        "event": "case_attempt_started",
        "case": case1.name,
        "ordinal": 1,
        "primary_stage": "fixed14",
        "manifest_sha256": manifest.sha256,
        "source_bundle_sha256": identity["production_source_bundle_sha256"],
        "attempt_token_sha256": "c" * 64,
    }
    try:
        _event_state([header, attempt0, result0, attempt1], manifest, identity)
    except GateError:
        pass
    else:
        raise GateError("resume continued after a required-positive UNKNOWN")
    hostile = dict(unknown)
    hostile["input_sha256"] = {"onnx": "0" * 64, "vnnlib": case0.vnnlib_sha256}
    try:
        _event_state(
            [header, attempt0, {"event": "case_result", "case": case0.name, "record": hostile}],
            manifest,
            identity,
        )
    except GateError:
        pass
    else:
        raise GateError("resume accepted a changed input identity")
    forged = {
        "schema": "act.hybridz.fprime.production_retention_gate.worker.v1",
        "case": case0.name,
        "validated_status": "FALSIFIED",
    }
    try:
        _event_state(
            [header, attempt0, {"event": "case_result", "case": case0.name, "record": forged}],
            manifest,
            identity,
        )
    except GateError:
        pass
    else:
        raise GateError("resume accepted an unauthoritative FALSIFIED record")
    valid_line = RESULT_PREFIX + _canonical_json({"case": case0.name}) + "\n"
    _decode_worker_stdout(valid_line)
    for hostile_stdout in (
        "noise\n" + valid_line,
        valid_line + valid_line,
        " " + valid_line,
    ):
        try:
            _decode_worker_stdout(hostile_stdout)
        except GateError:
            pass
        else:
            raise GateError("stdout isolation accepted a hostile channel")
    authorization_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_authorize_worker"
    ]
    if len(authorization_calls) != 1:
        raise GateError("hidden worker lacks its one-shot parent authorization")
    return {
        "schema": "act.hybridz.fprime.production_retention_gate.static.v1",
        "status": "STATIC_CPU_ONLY_PASS",
        "harness_sha256": _sha256(Path(__file__)),
        "five_case_helper_sha256": FIVE_HELPER_SHA256,
        "production_source_bundle_sha256": identity["production_source_bundle_sha256"],
        "manifest_sha256": manifest.sha256,
        "fixed14": len(manifest.fixed14),
        "fixed14_required_falsified": len(manifest.fixed14_required_falsified),
        "retained59": len(manifest.retained59),
        "unique_execution_cases": len(manifest.execution_order),
        "overlap_reused": len(set(manifest.fixed14) & set(manifest.retained59)),
        "production_helper_callsites": 1,
        "verify_once_callsites_in_locked_helper": 1,
        "resume_hostile_selftests": 3,
        "stdout_hostile_selftests": 3,
        "worker_authorization_callsites": 1,
        "gpu_initialized": False,
        "artifacts_created": False,
        "events_path_exists": EVENTS_PATH.exists(),
        "summary_path_exists": SUMMARY_PATH.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument(
        "--worker-case",
        help=argparse.SUPPRESS,
    )
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
            raise GateError("another retention parent already owns the run") from exc
        return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
