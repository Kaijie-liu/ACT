"""Strict 6 -> 14 -> 40 stop-loss gates for CIFAR100/TinyImageNet HybridZ.

The stage manifests in ``configs/hybridz_largecls_gates.yaml`` are incremental:
Gate-14 consumes a PASS receipt from Gate-6 and runs only eight new sentinels;
Gate-40 consumes a PASS receipt from Gate-14 and runs only 26 new sentinels.
Reference ``S``/``U`` labels are historical diagnostics and never participate
in a verdict or promotion decision.

Each actual verification runs in a fresh child process.  The parent owns the
wall-clock deadline and kills the complete child process group at expiry, so
model parsing/conversion, TorchToACT, propagation, solving, and strict replay
all share one budget.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Mapping, Optional, Sequence
import uuid

import yaml

from act.back_end.config import (
    HybridZConfig,
    normalize_query_dual_feedback_targets,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "hybridz_largecls_gates.yaml"
SCHEMA_VERSION = 3
GATES = (6, 14, 40)
CONCLUSIVE = {"certified", "falsified"}
ERROR_STATUSES = {"verifier_error", "model_infer_failure", "error"}
RUN_FAILURES = {
    "FAIL_P0",
    "FAIL_ERROR",
    "BLOCKED_RESOURCE",
    "BLOCKED_ENGINE",
}
ANSI_RED = "\033[1;31m"
ANSI_RESET = "\033[0m"
CUDA_PEAK_MEMORY_SCHEMA = "act.cuda_peak_memory.v1"
CUDA_PEAK_MEMORY_SUMMARY_SCHEMA = "act.cuda_peak_memory_summary.v1"
QUERY_DUAL_DIAGNOSTIC_FIELD_LIMIT = 160
QUERY_DUAL_DIAGNOSTIC_ERROR_LIMIT = 640
QUERY_DUAL_PIPELINE_SCHEMA = "act.verified_query_dual_feedback.v2"
QUERY_DUAL_STAGE_SCHEMA = "act.verified_query_dual_stage.v2"
QUERY_DUAL_PROPERTY_SCHEMA = "act.verified_query_dual_property.v2"
QUERY_DUAL_CANDIDATE_SCHEMA = "act.query_dual_candidates.v2"
QUERY_DUAL_CANDIDATE_PROTOCOL = "descriptor_only_v2"


class GateConfigError(RuntimeError):
    """A manifest, promotion, or launch invariant failed."""


@dataclass(frozen=True)
class CsvInstance:
    benchmark: str
    iid: int
    family: str
    onnx_rel: str
    vnnlib_rel: str
    csv_timeout: float
    onnx_path: Path
    vnnlib_path: Path


@dataclass(frozen=True)
class Sentinel:
    gate: int
    family: str
    iid: int
    reference_label: str
    query_dual_feedback_targets: tuple[int, ...]
    query_dual_feedback_status: str
    instance: CsvInstance


@dataclass(frozen=True)
class ResultClassification:
    """Structured, mutually-exclusive interpretation of a worker receipt."""

    failure_class: Optional[str]
    reason: Optional[str]
    conclusive: bool

    def __post_init__(self) -> None:
        if self.failure_class is not None and self.failure_class not in RUN_FAILURES:
            raise ValueError(f"invalid failure class: {self.failure_class!r}")
        if self.failure_class is None and self.reason is not None:
            raise ValueError("a non-fatal classification cannot carry a reason")
        if self.failure_class is not None and self.conclusive:
            raise ValueError("a failed classification cannot be conclusive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _query_dual_candidate_policy() -> dict[str, Any]:
    """Fixed production receipt identity for descriptor-only query dual."""

    return {
        "wrapper_schema": "verifier_query_dual_feedback_transaction_v1",
        "pipeline_schema": QUERY_DUAL_PIPELINE_SCHEMA,
        "target_stage_schema": QUERY_DUAL_STAGE_SCHEMA,
        "property_stage_schema": QUERY_DUAL_PROPERTY_SCHEMA,
        "candidate_schema": QUERY_DUAL_CANDIDATE_SCHEMA,
        "candidate_protocol": QUERY_DUAL_CANDIDATE_PROTOCOL,
        "candidate_success_status": "descriptors_generated",
        "target_empty_status": "no_queries_fallback",
        "candidate_bound_source": "none_descriptor_only",
        "replay_chunk_size_binding": (
            "equals_effective_query_dual_feedback_block_size"
        ),
        "replay_max_workspace_bytes_binding": "each_cpu_replay_receipt",
        "conv_channel_chunk_binding": "root_box_certificate",
        "candidate_non_authoritative_audit_fields": [
            "lr_alpha",
            "lr_decay",
            "solver",
            "elapsed_seconds",
            "timings",
        ],
        "pipeline_non_authoritative_audit_fields": [
            "candidate_generator",
            "candidate_solver_factory",
            "dual_solver_default_device",
            "dual_solver_default_dtype",
            "candidate_cuda_device_name",
        ],
        "optimizer_margins_exported": False,
        "gpu_frozen_alpha_replay": False,
        "cpu_independent_replay_required": True,
        "operator_schema": "operator_hz_verified_query_dual_feedback_v1",
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_query_dual_targets(
    value: Any,
    *,
    context: str,
) -> tuple[int, ...]:
    try:
        return normalize_query_dual_feedback_targets(value)
    except ValueError as exc:
        raise GateConfigError(f"{context}: {exc}") from exc


def _query_dual_family_snapshot(
    families: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Canonical family-local target schedules bound into every receipt."""

    snapshot: dict[str, dict[str, Any]] = {}
    for family in sorted(families):
        spec = families[family]
        targets = _gate_query_dual_targets(
            spec.get("query_dual_feedback_targets"),
            context=f"family {family}",
        )
        status = spec.get("query_dual_feedback_status")
        if status not in {"gate1_candidate", "not_promoted"}:
            raise GateConfigError(
                f"family {family} query_dual_feedback_status must be "
                "gate1_candidate or not_promoted"
            )
        if status == "not_promoted" and targets:
            raise GateConfigError(
                f"family {family} is not_promoted but has query-dual targets"
            )
        snapshot[family] = {
            "targets": list(targets),
            "status": str(status),
        }
    return snapshot


def _query_dual_effective_by_family(
    runtime: Mapping[str, Any],
    family_snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive family-effective settings from one explicitly requested knob."""

    requested_steps = int(runtime["query_dual_feedback_steps"])
    requested_seconds = float(runtime["query_dual_feedback_time_limit"])
    effective: dict[str, dict[str, Any]] = {}
    for family in sorted(family_snapshot):
        spec = family_snapshot[family]
        targets = _gate_query_dual_targets(
            spec.get("targets"),
            context=f"family snapshot {family}",
        )
        status = str(spec.get("status"))
        family_enabled = status == "gate1_candidate"
        effective[family] = {
            "targets": list(targets),
            "status": status,
            "requested_steps": requested_steps,
            "requested_time_limit": requested_seconds,
            "effective_steps": requested_steps if family_enabled else 0,
            "effective_time_limit": (
                requested_seconds if family_enabled else 0.0
            ),
            "block_size": int(runtime["query_dual_feedback_block_size"]),
            "device": str(runtime["query_dual_feedback_device"]),
        }
    return effective


def _strict_relative_file(base: Path, raw: str, *, suffix: str) -> Path:
    rel = Path(raw)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        raise GateConfigError(f"instances.csv path is not strict-relative: {raw!r}")
    if rel.suffix.lower() != suffix:
        raise GateConfigError(
            f"instances.csv path {raw!r} must have suffix {suffix!r}"
        )
    base_resolved = base.resolve()
    target = (base_resolved / rel).resolve()
    try:
        target.relative_to(base_resolved)
    except ValueError as exc:
        raise GateConfigError(
            f"instances.csv path escapes benchmark directory: {raw!r}"
        ) from exc
    if not target.is_file():
        raise GateConfigError(f"instances.csv target does not exist: {target}")
    return target


def _family_for_row(
    benchmark: str,
    iid: int,
    model_basename: str,
    families: Mapping[str, Mapping[str, Any]],
) -> str:
    matches = []
    for name, spec in families.items():
        if spec["benchmark"] != benchmark:
            continue
        if not (int(spec["iid_min"]) <= iid <= int(spec["iid_max"])):
            continue
        if spec["model_basename"] != model_basename:
            continue
        matches.append(name)
    if len(matches) != 1:
        raise GateConfigError(
            f"{benchmark} iid={iid} model={model_basename!r} maps to "
            f"{len(matches)} families (expected exactly one): {matches}"
        )
    return matches[0]


def _load_instances_csv(
    benchmark_root: Path,
    benchmark: str,
    families: Mapping[str, Mapping[str, Any]],
) -> list[CsvInstance]:
    category_dir = (benchmark_root / benchmark).resolve()
    csv_path = category_dir / "instances.csv"
    if not csv_path.is_file():
        raise GateConfigError(f"instances.csv not found: {csv_path}")

    parsed: list[CsvInstance] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        for iid, row in enumerate(reader):
            if len(row) != 3 or any(not cell.strip() for cell in row):
                raise GateConfigError(
                    f"{csv_path}: iid={iid} expected exactly three non-empty "
                    f"columns, got {row!r}"
                )
            onnx_rel, vnnlib_rel, raw_timeout = (cell.strip() for cell in row)
            try:
                csv_timeout = float(raw_timeout)
            except ValueError as exc:
                raise GateConfigError(
                    f"{csv_path}: iid={iid} invalid timeout {raw_timeout!r}"
                ) from exc
            if not math.isfinite(csv_timeout) or csv_timeout <= 0.0:
                raise GateConfigError(
                    f"{csv_path}: iid={iid} timeout must be finite and positive"
                )
            onnx_path = _strict_relative_file(
                category_dir, onnx_rel, suffix=".onnx"
            )
            vnnlib_path = _strict_relative_file(
                category_dir, vnnlib_rel, suffix=".vnnlib"
            )
            family = _family_for_row(
                benchmark, iid, Path(onnx_rel).name, families
            )
            expected_prefix = Path(onnx_rel).stem + "_prop_"
            if not Path(vnnlib_rel).name.startswith(expected_prefix):
                raise GateConfigError(
                    f"{csv_path}: iid={iid} VNNLIB basename does not match "
                    f"model family prefix {expected_prefix!r}"
                )
            parsed.append(
                CsvInstance(
                    benchmark=benchmark,
                    iid=iid,
                    family=family,
                    onnx_rel=onnx_rel,
                    vnnlib_rel=vnnlib_rel,
                    csv_timeout=csv_timeout,
                    onnx_path=onnx_path,
                    vnnlib_path=vnnlib_path,
                )
            )

    if not parsed:
        raise GateConfigError(f"{csv_path} is empty")
    return parsed


def _validate_property_micro_rlt_settings(
    settings: Mapping[str, Any],
    *,
    context: str = "",
) -> None:
    """Validate the bounded, default-off C49 parent-relaxation contract."""

    cap = settings.get("property_micro_rlt_product_cap")
    if isinstance(cap, bool) or not isinstance(cap, int):
        raise GateConfigError(
            f"{context}property_micro_rlt_product_cap must be an integer"
        )
    if not 0 <= cap <= 4096:
        raise GateConfigError(
            f"{context}property_micro_rlt_product_cap must lie in [0, 4096]"
        )
    packet_mode = settings.get("property_micro_rlt_packet_mode")
    if (
        not isinstance(packet_mode, str)
        or packet_mode not in {"both", "first", "second"}
    ):
        raise GateConfigError(
            f"{context}property_micro_rlt_packet_mode must be one of "
            "both|first|second"
        )
    if cap <= 0 and packet_mode != "both":
        raise GateConfigError(
            f"{context}property_micro_rlt_packet_mode first/second requires "
            "property micro-RLT to be enabled"
        )
    raw_seconds = settings.get(
        "property_micro_rlt_parent_prefilter_seconds"
    )
    if isinstance(raw_seconds, bool) or not isinstance(
        raw_seconds, (int, float)
    ):
        raise GateConfigError(
            f"{context}property_micro_rlt_parent_prefilter_seconds "
            "must be numeric"
        )
    seconds = float(raw_seconds)
    if not math.isfinite(seconds) or not 0.0 <= seconds <= 10.0:
        raise GateConfigError(
            f"{context}property_micro_rlt_parent_prefilter_seconds "
            "must be finite and lie in [0, 10]"
        )
    if (cap > 0) != (seconds > 0.0):
        raise GateConfigError(
            f"{context}property micro-RLT product cap and parent prefilter "
            "time must be enabled together"
        )
    parent_only = settings.get(
        "property_micro_rlt_parent_only_diagnostic"
    )
    if not isinstance(parent_only, bool):
        raise GateConfigError(
            f"{context}property_micro_rlt_parent_only_diagnostic "
            "must be a boolean"
        )
    if parent_only and cap <= 0:
        raise GateConfigError(
            f"{context}property_micro_rlt_parent_only_diagnostic requires "
            "property micro-RLT to be enabled"
        )
    if cap <= 0:
        return
    if settings.get("engine") != "operator_hz_objbound":
        raise GateConfigError(
            f"{context}property micro-RLT requires "
            "engine=operator_hz_objbound"
        )
    if settings.get("property_tail_upper") is not True:
        raise GateConfigError(
            f"{context}property micro-RLT requires "
            "property_tail_upper=true"
        )
    try:
        exact_budget = int(settings["operator_exact_budget"])
        residual_budget = int(settings["property_residual_budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GateConfigError(
            f"{context}property micro-RLT requires the depth-2 property-tail "
            "phase split with operator_exact_budget="
            "property_residual_budget=2"
        ) from exc
    if exact_budget != 2 or residual_budget != 2:
        raise GateConfigError(
            f"{context}property micro-RLT requires the depth-2 property-tail "
            "phase split with operator_exact_budget="
            "property_residual_budget=2"
        )


def _validate_property_micro_rlt_parent_only_selection(
    *,
    enabled: bool,
    gate: int,
    selected_families: Sequence[str],
    iid: Optional[int],
    packet_mode: str = "both",
) -> None:
    """Keep parent-only and packet-isolation experiments at Gate-1."""

    packet_isolation = packet_mode in {"first", "second"}
    if not enabled and not packet_isolation:
        return
    setting = (
        "property_micro_rlt_parent_only_diagnostic"
        if enabled
        else f"property_micro_rlt_packet_mode={packet_mode}"
    )
    if gate != 6:
        raise GateConfigError(
            f"{setting} requires --gate 6"
        )
    if len(selected_families) != 1:
        raise GateConfigError(
            f"{setting} requires exactly one --family"
        )
    if iid is None:
        raise GateConfigError(
            f"{setting} requires an explicit --iid"
        )


def _validate_operator_phase_clique_selection(
    *,
    time_limit: float,
    gate: int,
    selected_families: Sequence[str],
    iid: Optional[int],
) -> None:
    """Keep the first K4 production trial to one non-promotable iid."""

    if float(time_limit) <= 0.0:
        return
    if gate != 6:
        raise GateConfigError(
            "operator phase cliques initially require --gate 6"
        )
    if len(selected_families) != 1:
        raise GateConfigError(
            "operator phase cliques initially require exactly one --family"
        )
    if iid is None:
        raise GateConfigError(
            "operator phase cliques initially require an explicit --iid"
        )


_RUNTIME_BUILTIN_INTEGER_FIELDS = (
    "default_gate",
    "gpu_index",
    "operator_exact_budget",
    "query_dual_feedback_steps",
    "query_dual_feedback_block_size",
    "preactivation_lp_budget",
    "property_correlation_budget",
    "property_residual_budget",
    "property_residual_max_adjoint_cells",
    "property_residual_pool_per_rival",
    "property_micro_rlt_product_cap",
    "property_tail_alpha_steps",
    "property_tail_alpha_max_cells",
    "property_tail_mixture_grid_bits",
    "property_tail_pairhull_budget",
    "property_tail_suffix_blocks",
    "property_tail_suffix_alpha_steps",
    "gpu_dual_steps",
    "gpu_dual_row_topk",
    "row_workers",
    "total_solver_threads",
    "max_inconclusive_per_family",
)


def _validate_runtime(
    runtime: Mapping[str, Any],
    *,
    query_dual_feedback_targets: Optional[Sequence[int]] = None,
) -> None:
    required = {
        "default_gate",
        "wall_timeout_seconds",
        "device",
        "gpu_index",
        "dtype",
        "engine",
        "operator_exact_budget",
        "operator_phase_clique_time_limit",
        "operator_materialize_add",
        "query_dual_feedback_steps",
        "query_dual_feedback_time_limit",
        "query_dual_feedback_block_size",
        "query_dual_feedback_device",
        "preactivation_lp_budget",
        "preactivation_lp_time_limit",
        "property_residual_budget",
        "property_residual_time_limit",
        "property_residual_max_adjoint_cells",
        "property_residual_pool_per_rival",
        "property_tail_upper",
        "property_micro_rlt_product_cap",
        "property_micro_rlt_packet_mode",
        "property_micro_rlt_parent_prefilter_seconds",
        "property_micro_rlt_parent_only_diagnostic",
        "property_tail_add_source_planes",
        "property_tail_alpha_steps",
        "property_tail_alpha_time_limit",
        "property_tail_alpha_learning_rate",
        "property_tail_alpha_max_cells",
        "property_tail_alpha_device",
        "property_tail_mixture_grid_bits",
        "property_tail_pairhull_budget",
        "property_tail_pairhull_time_limit",
        "property_tail_suffix_blocks",
        "property_tail_suffix_alpha_steps",
        "property_tail_suffix_alpha_time_limit",
        "property_tail_suffix_alpha_device",
        "gpu_dual_steps",
        "gpu_dual_time_limit",
        "gpu_dual_row_topk",
        "gpu_dual_learning_rate",
        "lp_prefilter_fraction",
        "lp_prefilter_max_seconds",
        "row_workers",
        "total_solver_threads",
        "max_inconclusive_per_family",
    }
    missing = sorted(required - set(runtime))
    if missing:
        raise GateConfigError(f"runtime is missing fields: {missing}")
    for name in _RUNTIME_BUILTIN_INTEGER_FIELDS:
        if name in runtime and type(runtime[name]) is not int:
            raise GateConfigError(
                f"{name} must be an integer of the built-in int type; "
                "booleans and numeric coercions are forbidden"
            )
    if int(runtime["default_gate"]) != 6:
        raise GateConfigError("default_gate must be 6")
    timeout = float(runtime["wall_timeout_seconds"])
    if not math.isfinite(timeout) or not (0.0 < timeout <= 100.0):
        raise GateConfigError("wall_timeout_seconds must be in (0, 100]")
    if runtime["device"] != "cuda":
        raise GateConfigError("large-classification gates require device=cuda")
    if int(runtime["gpu_index"]) < 0:
        raise GateConfigError("gpu_index must be non-negative")
    if runtime["dtype"] != "float64":
        raise GateConfigError(
            "strict large-classification gates require dtype=float64; "
            "float32 is diagnostic-only and is forbidden here"
        )
    if not str(runtime["engine"]).strip():
        raise GateConfigError("engine must be non-empty")
    if int(runtime["operator_exact_budget"]) < -1:
        raise GateConfigError(
            "operator_exact_budget must be -1, 0, or a positive integer"
        )
    raw_phase_clique_seconds = runtime[
        "operator_phase_clique_time_limit"
    ]
    if isinstance(raw_phase_clique_seconds, bool) or not isinstance(
        raw_phase_clique_seconds, (int, float)
    ):
        raise GateConfigError(
            "operator_phase_clique_time_limit must be numeric"
        )
    phase_clique_seconds = float(raw_phase_clique_seconds)
    if (
        not math.isfinite(phase_clique_seconds)
        or not 0.0 <= phase_clique_seconds <= 40.0
    ):
        raise GateConfigError(
            "operator_phase_clique_time_limit must be finite and lie in "
            "[0, 40]"
        )
    if not isinstance(runtime["operator_materialize_add"], bool):
        raise GateConfigError("operator_materialize_add must be a boolean")
    query_steps = runtime["query_dual_feedback_steps"]
    if isinstance(query_steps, bool) or not isinstance(query_steps, int):
        raise GateConfigError(
            "query_dual_feedback_steps must be an integer"
        )
    if not 0 <= query_steps <= 64:
        raise GateConfigError(
            "query_dual_feedback_steps must lie in [0, 64]"
        )
    raw_query_seconds = runtime["query_dual_feedback_time_limit"]
    if isinstance(raw_query_seconds, bool) or not isinstance(
        raw_query_seconds, (int, float)
    ):
        raise GateConfigError(
            "query_dual_feedback_time_limit must be numeric"
        )
    query_seconds = float(raw_query_seconds)
    if (
        not math.isfinite(query_seconds)
        or not 0.0 <= query_seconds <= 20.0
    ):
        raise GateConfigError(
            "query_dual_feedback_time_limit must be finite and lie in "
            "[0, 20]"
        )
    query_block_size = runtime["query_dual_feedback_block_size"]
    if (
        isinstance(query_block_size, bool)
        or not isinstance(query_block_size, int)
        or not 1 <= query_block_size <= 4096
    ):
        raise GateConfigError(
            "query_dual_feedback_block_size must be an integer in [1, 4096]"
        )
    if runtime["query_dual_feedback_device"] not in {"cpu", "cuda"}:
        raise GateConfigError(
            "query_dual_feedback_device must be cpu or cuda"
        )
    if query_steps == 0:
        if query_seconds != 0.0:
            raise GateConfigError(
                "disabled query-dual feedback requires "
                "query_dual_feedback_time_limit=0"
            )
    else:
        property_only_bound_replay = bool(
            runtime.get("residual_bound_screen", False)
            and not (
                _gate_query_dual_targets(
                    query_dual_feedback_targets,
                    context="runtime query-dual targets",
                )
                if query_dual_feedback_targets is not None
                else ()
            )
        )
        if query_seconds <= 0.0:
            raise GateConfigError(
                "enabled query-dual feedback requires "
                "query_dual_feedback_time_limit>0"
            )
        if query_dual_feedback_targets is not None:
            targets = _gate_query_dual_targets(
                query_dual_feedback_targets,
                context="runtime query-dual targets",
            )
            if not targets and not property_only_bound_replay:
                raise GateConfigError(
                    "enabled query-dual feedback requires family targets "
                    "or residual_bound_screen property-only replay"
                )
        if (
            not property_only_bound_replay
            and runtime["property_tail_upper"] is not True
        ):
            raise GateConfigError(
                "enabled query-dual feedback requires "
                "property_tail_upper=true"
            )
        if (
            property_only_bound_replay
            and runtime["property_tail_upper"] is not False
        ):
            raise GateConfigError(
                "property-only residual-bound query replay requires "
                "property_tail_upper=false"
            )
        if runtime["engine"] != "operator_hz_objbound":
            raise GateConfigError(
                "enabled query-dual feedback requires "
                "engine=operator_hz_objbound"
            )
        if int(runtime["operator_exact_budget"]) != 0:
            raise GateConfigError(
                "enabled query-dual feedback requires "
                "operator_exact_budget=0"
            )
    if int(runtime["preactivation_lp_budget"]) < 0:
        raise GateConfigError("preactivation_lp_budget must be nonnegative")
    preactivation_seconds = float(runtime["preactivation_lp_time_limit"])
    if (
        not math.isfinite(preactivation_seconds)
        or preactivation_seconds < 0.0
    ):
        raise GateConfigError(
            "preactivation_lp_time_limit must be finite and nonnegative"
        )
    correlation_budget = int(runtime.get("property_correlation_budget", 0))
    if correlation_budget < 0:
        raise GateConfigError(
            "property_correlation_budget must be nonnegative"
        )
    correlation_seconds = float(
        runtime.get("property_correlation_time_limit", 0.0)
    )
    if (
        not math.isfinite(correlation_seconds)
        or correlation_seconds < 0.0
    ):
        raise GateConfigError(
            "property_correlation_time_limit must be finite and "
            "nonnegative"
        )
    if (correlation_budget > 0) != (correlation_seconds > 0.0):
        raise GateConfigError(
            "property correlation budget and time limit must be enabled "
            "together"
        )
    if correlation_budget > 0 and not runtime["operator_materialize_add"]:
        raise GateConfigError(
            "property correlation requires operator_materialize_add=true"
        )
    phase_screen = runtime.get("residual_phase_screen", False)
    if not isinstance(phase_screen, bool):
        raise GateConfigError("residual_phase_screen must be a boolean")
    if phase_screen and not runtime["operator_materialize_add"]:
        raise GateConfigError(
            "residual_phase_screen requires operator_materialize_add=true"
        )
    bound_screen = runtime.get("residual_bound_screen", False)
    if not isinstance(bound_screen, bool):
        raise GateConfigError("residual_bound_screen must be a boolean")
    if bound_screen and not runtime["operator_materialize_add"]:
        raise GateConfigError(
            "residual_bound_screen requires operator_materialize_add=true"
        )
    if phase_screen and bound_screen:
        raise GateConfigError(
            "residual phase-only and bound screens are mutually exclusive"
        )
    if int(runtime["property_residual_budget"]) < 0:
        raise GateConfigError(
            "property_residual_budget must be nonnegative"
        )
    residual_seconds = float(runtime["property_residual_time_limit"])
    if not math.isfinite(residual_seconds) or residual_seconds < 0.0:
        raise GateConfigError(
            "property_residual_time_limit must be finite and nonnegative"
        )
    if int(runtime["property_residual_max_adjoint_cells"]) <= 0:
        raise GateConfigError(
            "property_residual_max_adjoint_cells must be positive"
        )
    if int(runtime["property_residual_pool_per_rival"]) <= 0:
        raise GateConfigError(
            "property_residual_pool_per_rival must be positive"
        )
    if not isinstance(runtime["property_tail_upper"], bool):
        raise GateConfigError("property_tail_upper must be a boolean")
    if not isinstance(runtime["property_tail_add_source_planes"], bool):
        raise GateConfigError(
            "property_tail_add_source_planes must be a boolean"
        )
    if (
        runtime["property_tail_add_source_planes"]
        and not runtime["property_tail_upper"]
    ):
        raise GateConfigError(
            "property_tail_add_source_planes requires "
            "property_tail_upper=true"
        )
    if (
        runtime["property_tail_add_source_planes"]
        and not runtime["operator_materialize_add"]
    ):
        raise GateConfigError(
            "property_tail_add_source_planes requires "
            "operator_materialize_add=true"
        )
    phase_split_mode = bool(
        runtime["property_tail_upper"]
        and int(runtime["operator_exact_budget"]) > 0
        and int(runtime["property_residual_budget"]) > 0
    )
    _validate_property_micro_rlt_settings(runtime)
    if phase_split_mode:
        if not 1 <= int(runtime["operator_exact_budget"]) <= 2:
            raise GateConfigError(
                "property-tail exact phase cover supports depth 1 or 2"
            )
        if int(runtime["property_residual_budget"]) != int(
            runtime["operator_exact_budget"]
        ):
            raise GateConfigError(
                "property-tail exact phase cover requires "
                "property_residual_budget=operator_exact_budget"
            )
        if residual_seconds <= 0.0:
            raise GateConfigError(
                "property-tail exact phase cover requires "
                "property_residual_time_limit>0"
            )
    elif (
        runtime["property_tail_upper"]
        and int(runtime["property_residual_budget"]) > 0
    ):
        raise GateConfigError(
            "property_tail_upper and property_residual_budget are mutually "
            "exclusive candidates"
        )
    if correlation_budget > 0 and (
        int(runtime["property_residual_budget"]) > 0
        or runtime["property_tail_upper"]
    ):
        raise GateConfigError(
            "property correlation, residual normal form, and property tail "
            "are isolated candidate families"
        )
    if phase_clique_seconds > 0.0:
        if runtime["engine"] != "operator_hz_objbound":
            raise GateConfigError(
                "operator phase cliques require "
                "engine=operator_hz_objbound"
            )
        if runtime["operator_materialize_add"] is not True:
            raise GateConfigError(
                "operator phase cliques require "
                "operator_materialize_add=true"
            )
        if int(runtime["operator_exact_budget"]) != 4:
            raise GateConfigError(
                "operator phase cliques require operator_exact_budget=4"
            )
        if int(runtime["property_residual_budget"]) != 4:
            raise GateConfigError(
                "operator phase cliques require "
                "property_residual_budget=4"
            )
        if residual_seconds <= 0.0:
            raise GateConfigError(
                "operator phase cliques require "
                "property_residual_time_limit>0"
            )
        if runtime["property_tail_upper"] is not False:
            raise GateConfigError(
                "operator phase cliques require property_tail_upper=false"
            )
        if correlation_budget != 0:
            raise GateConfigError(
                "operator phase cliques require "
                "property_correlation_budget=0"
            )
        if phase_screen or bound_screen:
            raise GateConfigError(
                "operator phase cliques require residual screens off"
            )
        if int(runtime["preactivation_lp_budget"]) != 0:
            raise GateConfigError(
                "operator phase cliques require preactivation_lp_budget=0"
            )
        if float(runtime["preactivation_lp_time_limit"]) != 0.0:
            raise GateConfigError(
                "operator phase cliques require "
                "preactivation_lp_time_limit=0"
            )
        if int(runtime["property_micro_rlt_product_cap"]) != 0:
            raise GateConfigError(
                "operator phase cliques require "
                "property_micro_rlt_product_cap=0"
            )
        if int(runtime["query_dual_feedback_steps"]) != 0:
            raise GateConfigError(
                "operator phase cliques require "
                "query_dual_feedback_steps=0"
            )
        if (
            int(runtime["gpu_dual_steps"]) != 0
            or float(runtime["gpu_dual_time_limit"]) != 0.0
            or int(runtime["gpu_dual_row_topk"]) != 0
        ):
            raise GateConfigError(
                "operator phase cliques require GPU dual candidates off"
            )
    alpha_steps = int(runtime["property_tail_alpha_steps"])
    if alpha_steps < 0:
        raise GateConfigError(
            "property_tail_alpha_steps must be nonnegative"
        )
    alpha_seconds = float(runtime["property_tail_alpha_time_limit"])
    if not math.isfinite(alpha_seconds) or alpha_seconds < 0.0:
        raise GateConfigError(
            "property_tail_alpha_time_limit must be finite and nonnegative"
        )
    if (alpha_steps > 0) != (alpha_seconds > 0.0):
        raise GateConfigError(
            "property-tail alpha steps and time limit must be enabled "
            "together"
        )
    alpha_lr = float(runtime["property_tail_alpha_learning_rate"])
    if not math.isfinite(alpha_lr) or alpha_lr <= 0.0:
        raise GateConfigError(
            "property_tail_alpha_learning_rate must be finite and positive"
        )
    if int(runtime["property_tail_alpha_max_cells"]) <= 0:
        raise GateConfigError(
            "property_tail_alpha_max_cells must be positive"
        )
    if runtime["property_tail_alpha_device"] not in {"cpu", "cuda"}:
        raise GateConfigError(
            "official property_tail_alpha_device must be cpu or cuda"
        )
    if alpha_steps > 0 and not runtime["property_tail_upper"]:
        raise GateConfigError(
            "property-tail alpha candidates require property_tail_upper"
        )
    if alpha_steps > 0 and int(runtime["operator_exact_budget"]) != 0:
        raise GateConfigError(
            "property-tail alpha candidates currently require "
            "operator_exact_budget=0"
        )
    mixture_grid_bits = runtime["property_tail_mixture_grid_bits"]
    if isinstance(mixture_grid_bits, bool) or not isinstance(
        mixture_grid_bits, int
    ):
        raise GateConfigError(
            "property_tail_mixture_grid_bits must be an integer"
        )
    if not 0 <= mixture_grid_bits <= 24:
        raise GateConfigError(
            "property_tail_mixture_grid_bits must lie in [0, 24]"
        )
    if mixture_grid_bits > 0 and not runtime["property_tail_upper"]:
        raise GateConfigError(
            "property_tail_mixture_grid_bits>0 requires "
            "property_tail_upper=true"
        )
    if mixture_grid_bits > 0 and alpha_steps <= 0:
        raise GateConfigError(
            "property_tail_mixture_grid_bits>0 requires "
            "property_tail_alpha_steps>0"
        )
    if mixture_grid_bits > 0 and alpha_seconds <= 0.0:
        raise GateConfigError(
            "property_tail_mixture_grid_bits>0 requires "
            "property_tail_alpha_time_limit>0"
        )
    if (
        mixture_grid_bits > 0
        and int(runtime["operator_exact_budget"]) != 0
    ):
        raise GateConfigError(
            "property_tail_mixture_grid_bits>0 requires "
            "operator_exact_budget=0"
        )
    pairhull_budget = runtime["property_tail_pairhull_budget"]
    if isinstance(pairhull_budget, bool) or not isinstance(
        pairhull_budget, int
    ):
        raise GateConfigError(
            "property_tail_pairhull_budget must be an integer"
        )
    if not 0 <= pairhull_budget <= 8:
        raise GateConfigError(
            "property_tail_pairhull_budget must lie in [0, 8]"
        )
    raw_pairhull_seconds = runtime["property_tail_pairhull_time_limit"]
    if isinstance(raw_pairhull_seconds, bool) or not isinstance(
        raw_pairhull_seconds, (int, float)
    ):
        raise GateConfigError(
            "property_tail_pairhull_time_limit must be numeric"
        )
    pairhull_seconds = float(raw_pairhull_seconds)
    if (
        not math.isfinite(pairhull_seconds)
        or not 0.0 <= pairhull_seconds <= 1.5
    ):
        raise GateConfigError(
            "property_tail_pairhull_time_limit must be finite and "
            "lie in [0, 1.5]"
        )
    if (pairhull_budget > 0) != (pairhull_seconds > 0.0):
        raise GateConfigError(
            "property-tail PairHull budget and time limit must be "
            "enabled together"
        )
    if pairhull_budget > 0 and not runtime["property_tail_upper"]:
        raise GateConfigError(
            "property_tail_pairhull_budget>0 requires "
            "property_tail_upper=true"
        )
    if (
        pairhull_budget > 0
        and int(runtime["operator_exact_budget"]) != 0
    ):
        raise GateConfigError(
            "property_tail_pairhull_budget>0 requires "
            "operator_exact_budget=0"
        )
    suffix_blocks = runtime["property_tail_suffix_blocks"]
    if isinstance(suffix_blocks, bool) or not isinstance(
        suffix_blocks, int
    ):
        raise GateConfigError(
            "property_tail_suffix_blocks must be an integer"
        )
    if not 0 <= suffix_blocks <= 8:
        raise GateConfigError(
            "property_tail_suffix_blocks must lie in [0, 8]"
        )
    if suffix_blocks > 0 and not runtime["property_tail_upper"]:
        raise GateConfigError(
            "property_tail_suffix_blocks>0 requires "
            "property_tail_upper=true"
        )
    if suffix_blocks > 0 and not runtime["operator_materialize_add"]:
        raise GateConfigError(
            "property_tail_suffix_blocks>0 requires "
            "operator_materialize_add=true"
        )
    if phase_split_mode and not 1 <= suffix_blocks <= 7:
        raise GateConfigError(
            "property-tail exact phase cover requires "
            "property_tail_suffix_blocks in [1, 7]"
        )
    suffix_alpha_steps = runtime["property_tail_suffix_alpha_steps"]
    if isinstance(suffix_alpha_steps, bool) or not isinstance(
        suffix_alpha_steps, int
    ):
        raise GateConfigError(
            "property_tail_suffix_alpha_steps must be an integer"
        )
    if not 0 <= suffix_alpha_steps <= 64:
        raise GateConfigError(
            "property_tail_suffix_alpha_steps must lie in [0, 64]"
        )
    suffix_alpha_seconds = float(
        runtime["property_tail_suffix_alpha_time_limit"]
    )
    if (
        not math.isfinite(suffix_alpha_seconds)
        or not 0.0 <= suffix_alpha_seconds <= 20.0
    ):
        raise GateConfigError(
            "property_tail_suffix_alpha_time_limit must be finite and lie "
            "in [0, 20]"
        )
    if (suffix_alpha_steps > 0) != (suffix_alpha_seconds > 0.0):
        raise GateConfigError(
            "property-tail suffix alpha steps and time limit must be "
            "enabled together"
        )
    if suffix_alpha_steps > 0 and suffix_blocks <= 0:
        raise GateConfigError(
            "property_tail_suffix_alpha_steps>0 requires "
            "property_tail_suffix_blocks>0"
        )
    if runtime["property_tail_suffix_alpha_device"] not in {
        "cpu",
        "cuda",
    }:
        raise GateConfigError(
            "official property_tail_suffix_alpha_device must be cpu or cuda"
        )
    if int(runtime["gpu_dual_steps"]) < 0:
        raise GateConfigError("gpu_dual_steps must be nonnegative")
    gpu_dual_seconds = float(runtime["gpu_dual_time_limit"])
    if not math.isfinite(gpu_dual_seconds) or gpu_dual_seconds < 0.0:
        raise GateConfigError(
            "gpu_dual_time_limit must be finite and nonnegative"
        )
    if int(runtime["gpu_dual_row_topk"]) < 0:
        raise GateConfigError("gpu_dual_row_topk must be nonnegative")
    gpu_dual_lr = float(runtime["gpu_dual_learning_rate"])
    if not math.isfinite(gpu_dual_lr) or gpu_dual_lr <= 0.0:
        raise GateConfigError(
            "gpu_dual_learning_rate must be finite and positive"
        )
    lp_fraction = float(runtime["lp_prefilter_fraction"])
    if not math.isfinite(lp_fraction) or not 0.0 <= lp_fraction <= 1.0:
        raise GateConfigError("lp_prefilter_fraction must lie in [0, 1]")
    lp_max_seconds = float(runtime["lp_prefilter_max_seconds"])
    if not math.isfinite(lp_max_seconds) or lp_max_seconds < 0.0:
        raise GateConfigError(
            "lp_prefilter_max_seconds must be finite and nonnegative"
        )
    row_workers = int(runtime["row_workers"])
    total_threads = int(runtime["total_solver_threads"])
    if not (1 <= row_workers <= 4):
        raise GateConfigError("row_workers must be in [1, 4]")
    if not (1 <= total_threads <= 20):
        raise GateConfigError("total_solver_threads must be in [1, 20]")
    if row_workers > total_threads:
        raise GateConfigError(
            "row_workers cannot exceed total_solver_threads; otherwise the "
            "row-parallel floor is already above the declared total cap"
        )
    if int(runtime["max_inconclusive_per_family"]) < 1:
        raise GateConfigError("max_inconclusive_per_family must be >= 1")


def load_manifest(config_path: Path) -> tuple[dict[str, Any], dict[int, list[Sentinel]], dict[str, Any]]:
    """Load YAML, strictly parse both official CSVs, and validate all gates."""

    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise GateConfigError(f"gate config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
        raise GateConfigError(
            f"config schema_version must equal {SCHEMA_VERSION}"
        )
    runtime = raw.get("runtime")
    families = raw.get("families")
    stages = raw.get("stages")
    if not isinstance(runtime, dict):
        raise GateConfigError("runtime must be a mapping")
    if not isinstance(families, dict) or not families:
        raise GateConfigError("families must be a non-empty mapping")
    if not isinstance(stages, dict):
        raise GateConfigError("stages must be a mapping")
    _validate_runtime(runtime)

    expected_family_names = {
        "cifar100_medium",
        "cifar100_large",
        "tinyimagenet_medium",
    }
    if set(families) != expected_family_names:
        raise GateConfigError(
            f"families must be exactly {sorted(expected_family_names)}"
        )
    for name, spec in families.items():
        if not isinstance(spec, dict):
            raise GateConfigError(f"family {name} must be a mapping")
        for key in (
            "benchmark",
            "model_basename",
            "iid_min",
            "iid_max",
            "query_dual_feedback_targets",
            "query_dual_feedback_status",
        ):
            if key not in spec:
                raise GateConfigError(f"family {name} missing {key}")
        if int(spec["iid_min"]) < 0 or int(spec["iid_max"]) < int(spec["iid_min"]):
            raise GateConfigError(f"family {name} has an invalid iid range")
    query_dual_families = _query_dual_family_snapshot(families)
    _validate_runtime(
        runtime,
        query_dual_feedback_targets=[
            target
            for spec in query_dual_families.values()
            for target in spec["targets"]
        ],
    )

    benchmark_root = Path(raw.get("benchmark_root", "")).expanduser().resolve()
    if not benchmark_root.is_dir():
        raise GateConfigError(f"benchmark_root not found: {benchmark_root}")
    benchmarks = sorted({str(spec["benchmark"]) for spec in families.values()})
    if benchmarks != ["cifar100_2024", "tinyimagenet_2024"]:
        raise GateConfigError(f"unexpected benchmark set: {benchmarks}")

    csv_rows: dict[str, list[CsvInstance]] = {}
    csv_hashes: dict[str, str] = {}
    for benchmark in benchmarks:
        csv_rows[benchmark] = _load_instances_csv(
            benchmark_root, benchmark, families
        )
        csv_path = benchmark_root / benchmark / "instances.csv"
        csv_hashes[benchmark] = _sha256_file(csv_path)

    # Verify every declared family range exists, is in bounds, and is mapped
    # to exactly that family in the official 0-based CSV.
    for name, spec in families.items():
        rows = csv_rows[str(spec["benchmark"])]
        lo, hi = int(spec["iid_min"]), int(spec["iid_max"])
        if hi >= len(rows):
            raise GateConfigError(
                f"family {name} range [{lo}, {hi}] exceeds CSV size {len(rows)}"
            )
        for iid in range(lo, hi + 1):
            if rows[iid].family != name:
                raise GateConfigError(
                    f"family {name} expected iid={iid}, CSV mapped "
                    f"{rows[iid].family}"
                )

    if set(stages) != {str(gate) for gate in GATES}:
        raise GateConfigError(f"stages must be exactly {GATES}")
    expected_previous = {6: None, 14: 6, 40: 14}
    cumulative_keys: set[tuple[str, int]] = set()
    resolved: dict[int, list[Sentinel]] = {}
    for gate in GATES:
        stage = stages[str(gate)]
        if not isinstance(stage, dict) or not isinstance(stage.get("add"), list):
            raise GateConfigError(f"stage {gate} must contain an add list")
        previous = stage.get("previous_gate")
        if previous is not None:
            previous = int(previous)
        if previous != expected_previous[gate]:
            raise GateConfigError(
                f"stage {gate} previous_gate must be {expected_previous[gate]}"
            )
        additions: list[Sentinel] = []
        for index, item in enumerate(stage["add"]):
            if not isinstance(item, dict):
                raise GateConfigError(f"stage {gate} add[{index}] must be a mapping")
            if set(item) != {"family", "iid", "reference_diagnostic_label"}:
                raise GateConfigError(
                    f"stage {gate} add[{index}] has unexpected fields"
                )
            family = str(item["family"])
            iid = int(item["iid"])
            label = str(item["reference_diagnostic_label"])
            if family not in families:
                raise GateConfigError(
                    f"stage {gate} add[{index}] unknown family {family!r}"
                )
            if label not in {"S", "U"}:
                raise GateConfigError(
                    f"stage {gate} add[{index}] diagnostic label must be S or U"
                )
            spec = families[family]
            rows = csv_rows[str(spec["benchmark"])]
            if not (0 <= iid < len(rows)):
                raise GateConfigError(
                    f"stage {gate} add[{index}] iid={iid} is out of range"
                )
            instance = rows[iid]
            if instance.family != family:
                raise GateConfigError(
                    f"stage {gate} iid={iid}: requested family={family}, "
                    f"CSV family={instance.family}"
                )
            key = (family, iid)
            if key in cumulative_keys:
                raise GateConfigError(
                    f"duplicate cumulative sentinel {family}/iid={iid}"
                )
            cumulative_keys.add(key)
            additions.append(
                Sentinel(
                    gate=gate,
                    family=family,
                    iid=iid,
                    reference_label=label,
                    query_dual_feedback_targets=tuple(
                        query_dual_families[family]["targets"]
                    ),
                    query_dual_feedback_status=str(
                        query_dual_families[family]["status"]
                    ),
                    instance=instance,
                )
            )
        expected_size = int(stage.get("cumulative_size", -1))
        if len(cumulative_keys) != expected_size or expected_size != gate:
            raise GateConfigError(
                f"stage {gate}: cumulative sentinel count is "
                f"{len(cumulative_keys)}, declared={expected_size}, expected={gate}"
            )
        resolved[gate] = additions

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": _sha256_file(config_path),
        "csv_sha256": csv_hashes,
        "families": families,
        "query_dual_feedback_families": query_dual_families,
        "stages": stages,
    }
    provenance = {
        "config_path": str(config_path),
        "config_sha256": manifest_payload["config_sha256"],
        "csv_sha256": csv_hashes,
        "manifest_sha256": _sha256_bytes(_canonical_json(manifest_payload)),
        "benchmark_root": str(benchmark_root),
        "query_dual_feedback_families": query_dual_families,
    }
    raw["_resolved_config_path"] = str(config_path)
    raw["_resolved_benchmark_root"] = str(benchmark_root)
    return raw, resolved, provenance


def _source_fingerprint() -> tuple[str, list[dict[str, Any]]]:
    """Hash the complete local ACT source/config surface used by a gate.

    A hand-maintained allow-list silently misses new dispatch, parser, or
    numerical helper modules.  Hash every Python/YAML source under ``act``
    plus the environment declaration instead (runtime logs and documentation
    are deliberately excluded).  Per-file hashes are retained in the receipt
    so the aggregate can be independently rebuilt.
    """

    paths = {
        path.resolve()
        for path in (REPO_ROOT / "act").rglob("*")
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".yaml", ".yml"}
        )
    }
    environment_file = REPO_ROOT / "environment.yml"
    if environment_file.is_file():
        paths.add(environment_file.resolve())
    if not paths:
        raise GateConfigError("source fingerprint found no ACT source files")

    records: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            relative = str(path.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise GateConfigError(
                f"source fingerprint path escaped repository: {path}"
            ) from exc
        stat = path.stat()
        records.append(
            {
                "path": relative,
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(path),
            }
        )
    return _sha256_bytes(_canonical_json(records)), records


def _artifact_fingerprint(
    stages: Mapping[int, Sequence[Sentinel]],
) -> tuple[str, list[dict[str, Any]]]:
    """Hash every ONNX/VNNLIB artifact in the complete 40-sentinel ladder."""

    uses: dict[Path, set[str]] = {}
    kinds: dict[Path, set[str]] = {}
    for gate in GATES:
        for sentinel in stages[gate]:
            key = f"gate{gate}:{sentinel.family}:iid{sentinel.iid}"
            for kind, path in (
                ("onnx", sentinel.instance.onnx_path),
                ("vnnlib", sentinel.instance.vnnlib_path),
            ):
                resolved = path.resolve()
                uses.setdefault(resolved, set()).add(key)
                kinds.setdefault(resolved, set()).add(kind)
    records: list[dict[str, Any]] = []
    for path in sorted(uses, key=lambda item: str(item)):
        if not path.is_file():
            raise GateConfigError(f"gate artifact disappeared: {path}")
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "kinds": sorted(kinds[path]),
                "uses": sorted(uses[path]),
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise GateConfigError("artifact fingerprint found no artifacts")
    return _sha256_bytes(_canonical_json(records)), records


def _fixed_worker_environment(runtime: Mapping[str, Any]) -> dict[str, str]:
    """Return every numerical/HybridZ environment option a worker may see."""

    row_workers = int(runtime["row_workers"])
    total_threads = int(runtime["total_solver_threads"])
    if row_workers > total_threads:
        raise GateConfigError(
            "row_workers exceeds the declared total solver thread cap"
        )
    per_solver_threads = max(1, total_threads // row_workers)
    fixed = {
        "CUDA_VISIBLE_DEVICES": str(int(runtime["gpu_index"])),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Native BLAS teams run inside Python/row workers, so one native
        # thread each is the only portable way to preserve a global cap.
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "HZ_QUERY_WORKERS": str(row_workers),
        "HZ_MILP_THREADS": str(per_solver_threads),
        # HiGHS owns a process-global scheduler.  Every model in one isolated
        # worker must use the same size; mixing LP=1 with MILP=5 makes later
        # h.run() calls fail even though each individual option is valid.
        "HZ_LP_PREFILTER_THREADS": str(per_solver_threads),
        "HZ_TIGHT_THREADS": str(min(row_workers, total_threads)),
        "HZ_MILP_BACKEND": "highs",
        "HZ_HIGHS_OPTIONS": "",
        "HZ_MILP_HEURISTIC": "",
        "HZ_MILP_EQ_SUBST": "0",
        "HZ_MILP_ELIM_SINGLETONS": "0",
        "HZ_MILP_SCALE": "0",
        "HZ_MILP_CUTOFF_ROW": "1",
        "HZ_MILP_EQ_SUBST_MAX_INEQ": "2",
        "HZ_RELU_TIGHT_LP_TIMEOUT": "0",
        "HZ_RELU_TIGHT_ALL_ROWS": "0",
        "HZ_RELU_TIGHT_MAX_ROWS": "0",
        "HZ_LP_PREFILTER_FRACTION": repr(
            float(runtime["lp_prefilter_fraction"])
        ),
        "HZ_LP_PREFILTER_MAX_SECONDS": repr(
            float(runtime["lp_prefilter_max_seconds"])
        ),
        "HZ_EXACT_BASE_WITNESS_MAX_TERMS": "250000",
    }
    if row_workers * per_solver_threads > total_threads:
        raise GateConfigError("row x solver thread product exceeds total cap")
    referenced_hz: set[str] = set()
    pattern = re.compile(
        r"(?:environ\.get|getenv|_env_flag|_env_int)"
        r"\(\s*[\"'](HZ_[A-Z0-9_]+)[\"']"
    )
    for source in (REPO_ROOT / "act").rglob("*.py"):
        if "__pycache__" in source.parts:
            continue
        referenced_hz.update(
            pattern.findall(source.read_text(encoding="utf-8"))
        )
    missing = sorted(referenced_hz - set(fixed))
    if missing:
        raise GateConfigError(
            "strict worker policy does not fix newly referenced HZ options: "
            f"{missing}"
        )
    return fixed


def _gpu_identity(gpu_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=uuid,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3.0,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
        rows = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        if len(rows) != 1:
            raise RuntimeError(f"expected one GPU identity row, got {rows!r}")
        uuid_text, name, driver, total_mib, compute_cap = (
            part.strip() for part in rows[0].split(",", 4)
        )
        return {
            "gpu_index": int(gpu_index),
            "uuid": uuid_text,
            "name": name,
            "driver_version": driver,
            "total_bytes": int(total_mib) * 1024 * 1024,
            "compute_capability": compute_cap,
        }
    except Exception as exc:
        return {
            "gpu_index": int(gpu_index),
            "probe_error": f"{type(exc).__name__}: {exc}",
        }


def _environment_fingerprint(
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Capture the stable execution environment used for promotion equality."""

    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip().lower()
        if name:
            distributions[name] = str(distribution.version)
    executable = Path(sys.executable).resolve()
    relevant_names = (
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    )
    ambient_hz = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("HZ_")
    }
    snapshot = {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": str(executable),
            "executable_sha256": (
                _sha256_file(executable) if executable.is_file() else None
            ),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "distributions": [
            {"name": name, "version": version}
            for name, version in sorted(distributions.items())
        ],
        "gpu": _gpu_identity(int(runtime["gpu_index"])),
        "process_environment": {
            name: os.environ.get(name) for name in relevant_names
        },
        "ambient_hz_options": ambient_hz,
        "fixed_worker_environment": _fixed_worker_environment(runtime),
    }
    return _sha256_bytes(_canonical_json(snapshot)), snapshot


def _engine_connected(
    engine: str,
    *,
    operator_phase_clique_time_limit: float = 0.0,
) -> tuple[bool, str]:
    if engine in {"dense_hz_objbound", "sparse_hz_objbound"}:
        return True, "built_in_hybridz_engine"
    if engine == "operator_hz_objbound":
        verifier_path = REPO_ROOT / "act/back_end/verifier.py"
        operator_path = REPO_ROOT / "act/back_end/hybridz_tf/operator_hz.py"
        if not operator_path.is_file():
            return False, "operator_hz module is absent"
        verifier_source = verifier_path.read_text(encoding="utf-8")
        if "operator_hz_objbound" not in verifier_source:
            return False, "verify_once has no operator_hz_objbound dispatch"
        if float(operator_phase_clique_time_limit) > 0.0:
            pipeline_path = (
                REPO_ROOT
                / "act/back_end/hybridz_tf/"
                "operator_phase_clique_pipeline.py"
            )
            required_verifier_tokens = (
                "maybe_run_operator_phase_clique_pipeline",
                "consume_operator_phase_clique_pipeline_solver_handoff",
                "validate_consumed_operator_phase_clique_solver_build",
                "operator_phase_clique_materialization",
                "operator_phase_clique_solver_handoff",
            )
            if not pipeline_path.is_file():
                return False, "operator phase-clique pipeline is absent"
            if any(
                token not in verifier_source
                for token in required_verifier_tokens
            ):
                return (
                    False,
                    "verify_once has no complete operator phase-clique hook",
                )
            return True, "operator_hz phase-clique dispatch found"
        return True, "operator_hz_objbound dispatch found"
    return False, f"unrecognized HybridZ engine {engine!r}"


def _experiment_fingerprint(
    *,
    provenance: Mapping[str, Any],
    source_sha256: str,
    artifact_sha256: str,
    environment_sha256: str,
    engine: str,
    runtime: Mapping[str, Any],
    query_dual_feedback_families: Optional[
        Mapping[str, Mapping[str, Any]]
    ] = None,
) -> str:
    payload = {
        "manifest_sha256": provenance["manifest_sha256"],
        "source_sha256": source_sha256,
        "artifact_sha256": artifact_sha256,
        "environment_sha256": environment_sha256,
        "engine": engine,
        "wall_timeout_seconds": float(runtime["wall_timeout_seconds"]),
        "device": runtime["device"],
        "gpu_index": int(runtime["gpu_index"]),
        "dtype": runtime["dtype"],
        "cuda_peak_memory_policy": _cuda_peak_memory_policy(),
        "operator_exact_budget": int(runtime["operator_exact_budget"]),
        "operator_phase_clique_time_limit": float(
            runtime["operator_phase_clique_time_limit"]
        ),
        "operator_materialize_add": bool(
            runtime["operator_materialize_add"]
        ),
        "query_dual_feedback_steps": int(
            runtime["query_dual_feedback_steps"]
        ),
        "query_dual_feedback_time_limit": float(
            runtime["query_dual_feedback_time_limit"]
        ),
        "query_dual_feedback_block_size": int(
            runtime["query_dual_feedback_block_size"]
        ),
        "query_dual_feedback_device": str(
            runtime["query_dual_feedback_device"]
        ),
        "query_dual_feedback_families": dict(
            query_dual_feedback_families or {}
        ),
        "query_dual_feedback_effective_by_family": (
            _query_dual_effective_by_family(
                runtime,
                query_dual_feedback_families or {},
            )
        ),
        "query_dual_candidate_policy": _query_dual_candidate_policy(),
        "preactivation_lp_budget": int(
            runtime["preactivation_lp_budget"]
        ),
        "preactivation_lp_time_limit": float(
            runtime["preactivation_lp_time_limit"]
        ),
        "property_correlation_budget": int(
            runtime.get("property_correlation_budget", 0)
        ),
        "property_correlation_time_limit": float(
            runtime.get("property_correlation_time_limit", 0.0)
        ),
        "residual_phase_screen": bool(
            runtime.get("residual_phase_screen", False)
        ),
        "residual_bound_screen": bool(
            runtime.get("residual_bound_screen", False)
        ),
        "property_residual_budget": int(
            runtime["property_residual_budget"]
        ),
        "property_residual_time_limit": float(
            runtime["property_residual_time_limit"]
        ),
        "property_residual_max_adjoint_cells": int(
            runtime["property_residual_max_adjoint_cells"]
        ),
        "property_residual_pool_per_rival": int(
            runtime["property_residual_pool_per_rival"]
        ),
        "property_tail_upper": bool(runtime["property_tail_upper"]),
        "property_micro_rlt_product_cap": int(
            runtime["property_micro_rlt_product_cap"]
        ),
        "property_micro_rlt_packet_mode": str(
            runtime["property_micro_rlt_packet_mode"]
        ),
        "property_micro_rlt_parent_prefilter_seconds": float(
            runtime["property_micro_rlt_parent_prefilter_seconds"]
        ),
        "property_micro_rlt_parent_only_diagnostic": bool(
            runtime["property_micro_rlt_parent_only_diagnostic"]
        ),
        "property_tail_add_source_planes": bool(
            runtime["property_tail_add_source_planes"]
        ),
        "property_tail_alpha_steps": int(
            runtime["property_tail_alpha_steps"]
        ),
        "property_tail_alpha_time_limit": float(
            runtime["property_tail_alpha_time_limit"]
        ),
        "property_tail_alpha_learning_rate": float(
            runtime["property_tail_alpha_learning_rate"]
        ),
        "property_tail_alpha_max_cells": int(
            runtime["property_tail_alpha_max_cells"]
        ),
        "property_tail_alpha_device": str(
            runtime["property_tail_alpha_device"]
        ),
        "property_tail_mixture_grid_bits": int(
            runtime["property_tail_mixture_grid_bits"]
        ),
        "property_tail_pairhull_budget": int(
            runtime["property_tail_pairhull_budget"]
        ),
        "property_tail_pairhull_time_limit": float(
            runtime["property_tail_pairhull_time_limit"]
        ),
        "property_tail_suffix_blocks": int(
            runtime["property_tail_suffix_blocks"]
        ),
        "property_tail_suffix_alpha_steps": int(
            runtime["property_tail_suffix_alpha_steps"]
        ),
        "property_tail_suffix_alpha_time_limit": float(
            runtime["property_tail_suffix_alpha_time_limit"]
        ),
        "property_tail_suffix_alpha_device": str(
            runtime["property_tail_suffix_alpha_device"]
        ),
        "gpu_dual_steps": int(runtime["gpu_dual_steps"]),
        "gpu_dual_time_limit": float(runtime["gpu_dual_time_limit"]),
        "gpu_dual_row_topk": int(runtime["gpu_dual_row_topk"]),
        "gpu_dual_learning_rate": float(
            runtime["gpu_dual_learning_rate"]
        ),
        "lp_prefilter_fraction": float(runtime["lp_prefilter_fraction"]),
        "lp_prefilter_max_seconds": float(
            runtime["lp_prefilter_max_seconds"]
        ),
        "row_workers": int(runtime["row_workers"]),
        "total_solver_threads": int(runtime["total_solver_threads"]),
        "fixed_worker_environment": _fixed_worker_environment(runtime),
    }
    return _sha256_bytes(_canonical_json(payload))


def _read_last_jsonl_record(path: Path) -> dict[str, Any]:
    last: Optional[dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateConfigError(
                    f"{path}: malformed JSONL line {line_number}"
                ) from exc
            if isinstance(value, dict) and value.get("record_type") == "run_end":
                last = value
    if last is None:
        raise GateConfigError(f"{path} contains no run_end receipt")
    return last


def _load_promotion_record(path: Path) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise GateConfigError(f"promotion receipt not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        value = _read_last_jsonl_record(path)
    if not isinstance(value, dict):
        raise GateConfigError(f"promotion receipt must contain a JSON object: {path}")
    if value.get("record_type") == "summary" and isinstance(value.get("run_end"), dict):
        value = value["run_end"]
    return value, _sha256_file(path)


def validate_promotion(
    path: Path,
    *,
    gate: int,
    provenance: Mapping[str, Any],
    experiment_sha256: str,
    source_sha256: str,
    artifact_sha256: str,
    environment_sha256: str,
    expected_families: Sequence[str],
) -> dict[str, Any]:
    previous = {14: 6, 40: 14}.get(gate)
    if previous is None:
        raise GateConfigError("Gate-6 does not accept a promotion receipt")
    record, receipt_sha256 = _load_promotion_record(path)
    try:
        receipt_gate = int(record.get("gate", -1))
    except (TypeError, ValueError):
        receipt_gate = -1
    try:
        receipt_cumulative = int(record.get("cumulative_count", -1))
    except (TypeError, ValueError):
        receipt_cumulative = -1
    receipt_families = record.get("selected_families")
    if not isinstance(receipt_families, list) or not all(
        isinstance(item, str) for item in receipt_families
    ):
        receipt_families = []
    prior_chain = record.get("promotion_chain")
    expected_prior_gates = [] if previous == 6 else [6]
    chain_valid = isinstance(prior_chain, list)
    chain_receipts_unchanged = chain_valid
    if chain_valid:
        chain_gates: list[int] = []
        for item in prior_chain:
            if not isinstance(item, dict):
                chain_valid = False
                break
            try:
                item_gate = int(item.get("gate", -1))
            except (TypeError, ValueError):
                chain_valid = False
                break
            sha_fields = (
                item.get("receipt_sha256"),
                item.get("run_end_sha256"),
            )
            if (
                not isinstance(item.get("run_id"), str)
                or not item["run_id"]
                or not isinstance(item.get("receipt_path"), str)
                or not item["receipt_path"]
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(ch not in "0123456789abcdef" for ch in value)
                    for value in sha_fields
                )
            ):
                chain_valid = False
                break
            chain_gates.append(item_gate)
        chain_valid = chain_valid and chain_gates == expected_prior_gates
        if chain_valid:
            for item in prior_chain:
                try:
                    if _sha256_file(Path(item["receipt_path"])) != item[
                        "receipt_sha256"
                    ]:
                        chain_receipts_unchanged = False
                        break
                except Exception:
                    chain_receipts_unchanged = False
                    break
    expected_delta = {6: 6, 14: 8}[previous]
    integrity = record.get("run_end_integrity")
    checks = {
        "schema_version": record.get("schema_version") == SCHEMA_VERSION,
        "record_type": record.get("record_type") == "run_end",
        "status": record.get("status") == "PASS",
        "gate": receipt_gate == previous,
        "cumulative_count": receipt_cumulative == previous,
        "manifest_sha256": (
            record.get("manifest_sha256") == provenance["manifest_sha256"]
        ),
        "experiment_sha256": (
            record.get("experiment_sha256") == experiment_sha256
        ),
        "source_sha256": record.get("source_sha256") == source_sha256,
        "artifact_sha256": (
            record.get("artifact_sha256") == artifact_sha256
        ),
        "environment_sha256": (
            record.get("environment_sha256") == environment_sha256
        ),
        "all_families": (
            sorted(receipt_families) == sorted(expected_families)
        ),
        "all_expected_completed": (
            record.get("all_expected_completed") is True
        ),
        "all_results_conclusive": (
            record.get("all_results_conclusive") is True
        ),
        "delta_count": record.get("delta_count") == expected_delta,
        "instance_counts": (
            record.get("expected_instance_count") == expected_delta
            and record.get("completed_instance_count") == expected_delta
        ),
        "no_global_failure": (
            record.get("global_failure_class") is None
            and record.get("global_failure_reason") is None
        ),
        "run_end_integrity": (
            isinstance(integrity, Mapping)
            and integrity.get("passed") is True
        ),
        "promoted": not bool(record.get("unpromoted_diagnostic", False)),
        "not_partial": not bool(
            record.get("partial_family_diagnostic", False)
        ),
        "not_diagnostic_only": record.get("diagnostic_only") is False,
        "promotion_eligible": record.get("promotion_eligible") is True,
        "not_parent_only_diagnostic": (
            record.get(
                "property_micro_rlt_parent_only_diagnostic"
            )
            is False
        ),
        "production_micro_rlt_packet_mode": (
            record.get("property_micro_rlt_packet_mode") == "both"
        ),
        "promotion_chain": chain_valid,
        "promotion_chain_receipts_unchanged": (
            chain_valid and chain_receipts_unchanged
        ),
        "promotion_chain_sha256": (
            chain_valid
            and record.get("promotion_chain_sha256")
            == _sha256_bytes(_canonical_json(prior_chain))
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise GateConfigError(
            f"promotion receipt is not an exact Gate-{previous} PASS for this "
            f"experiment; failed checks: {failed}"
        )
    identity = {
        "gate": previous,
        "run_id": record.get("run_id"),
        "path": str(path.expanduser().resolve()),
        "receipt_path": str(path.expanduser().resolve()),
        "receipt_sha256": receipt_sha256,
        "run_end_sha256": _sha256_bytes(_canonical_json(record)),
    }
    return {
        **identity,
        "sha256": receipt_sha256,
        "checks": checks,
        "chain": [*prior_chain, identity],
    }


def _json_safe(
    value: Any,
    *,
    depth: int = 0,
    preserve_lists: bool = False,
) -> Any:
    # Exact proof receipts are intentionally nested (operator batch ->
    # candidate batch -> candidate record -> exact phase enumeration).  Keep
    # the recursion guard well beyond that fixed schema depth so sanitizing a
    # worker result cannot replace checksum-covered proof fields.
    if depth > 32:
        return "<depth-limit>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        preserve_nested_lists = bool(
            preserve_lists
            or _is_sha256(value.get("receipt_sha256"))
        )
        return {
            str(key): _json_safe(
                item,
                depth=depth + 1,
                preserve_lists=preserve_nested_lists,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        limit = len(value) if preserve_lists else 512
        items = [
            _json_safe(
                item,
                depth=depth + 1,
                preserve_lists=preserve_lists,
            )
            for item in value[:limit]
        ]
        if len(value) > limit:
            items.append(f"<{len(value) - limit} items omitted>")
        return items
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, bool, int, float)):
        return _json_safe(enum_value, depth=depth + 1)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item(), depth=depth + 1)
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist(), depth=depth + 1)
        except Exception:
            pass
    return repr(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(value),
                handle,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


class AppendOnlyReceipt:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, value: Mapping[str, Any]) -> None:
        payload = _canonical_json(_json_safe(value)) + b"\n"
        fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _mem_available() -> dict[str, Any]:
    try:
        with Path("/proc/meminfo").open("r", encoding="ascii") as handle:
            fields = {}
            for line in handle:
                key, raw = line.split(":", 1)
                fields[key] = raw.strip()
        kib = int(fields["MemAvailable"].split()[0])
        return {"available_bytes": kib * 1024, "source": "/proc/meminfo"}
    except Exception as exc:
        return {"available_bytes": None, "error": f"{type(exc).__name__}: {exc}"}


def _gpu_memory(gpu_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=uuid,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3.0,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError(f"expected one GPU row, got {rows!r}")
        uuid_text, free_mib, total_mib = (part.strip() for part in rows[0].split(","))
        return {
            "gpu_index": gpu_index,
            "uuid": uuid_text,
            "free_bytes": int(free_mib) * 1024 * 1024,
            "total_bytes": int(total_mib) * 1024 * 1024,
        }
    except Exception as exc:
        return {
            "gpu_index": gpu_index,
            "free_bytes": None,
            "total_bytes": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cuda_peak_memory_policy() -> dict[str, Any]:
    """Stable protocol identity; dynamic measurements live in receipts."""

    return {
        "schema": CUDA_PEAK_MEMORY_SCHEMA,
        "reset_api": "torch.cuda.reset_peak_memory_stats",
        "capture_apis": [
            "torch.cuda.max_memory_allocated",
            "torch.cuda.max_memory_reserved",
        ],
        "reset_point": "after_initialize_device_before_model_parse",
        "capture_scope": "through_worker_result_construction",
        "synchronize_before_capture": True,
        "completed_worker_requires_captured_receipt": True,
        "dynamic_values_in_experiment_fingerprint": False,
    }


def _cuda_peak_memory_unavailable(reason: str) -> dict[str, Any]:
    return {
        "schema": CUDA_PEAK_MEMORY_SCHEMA,
        "observation_status": "unavailable",
        "available": False,
        "unavailable_reason": str(reason),
        "reset_performed": False,
        "logical_device_index": None,
        "device_total_bytes": None,
        "max_memory_allocated_bytes": None,
        "max_memory_reserved_bytes": None,
    }


def _start_cuda_peak_memory(torch_module: Any) -> dict[str, Any]:
    """Reset PyTorch CUDA peaks and record the exact logical device."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return _cuda_peak_memory_unavailable("cuda_unavailable")
    try:
        device_index = int(cuda.current_device())
        device_total = int(
            cuda.get_device_properties(device_index).total_memory
        )
        if device_index < 0 or device_total <= 0:
            raise ValueError("CUDA device index/total memory is invalid")
        cuda.reset_peak_memory_stats(device_index)
    except Exception as exc:
        return {
            **_cuda_peak_memory_unavailable("reset_failed"),
            "observation_status": "reset_error",
            "available": True,
            "capture_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "schema": CUDA_PEAK_MEMORY_SCHEMA,
        "observation_status": "tracking",
        "available": True,
        "unavailable_reason": None,
        "reset_performed": True,
        "logical_device_index": device_index,
        "device_total_bytes": device_total,
        "max_memory_allocated_bytes": None,
        "max_memory_reserved_bytes": None,
    }


def _capture_cuda_peak_memory(
    torch_module: Any,
    tracking: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture peaks without masking an already active worker exception."""

    state = dict(tracking)
    if (
        state.get("available") is not True
        or state.get("reset_performed") is not True
    ):
        return state
    device_index = state.get("logical_device_index")
    try:
        cuda = torch_module.cuda
        cuda.synchronize(device_index)
        allocated = int(cuda.max_memory_allocated(device_index))
        reserved = int(cuda.max_memory_reserved(device_index))
        total = int(state["device_total_bytes"])
        if (
            allocated < 0
            or reserved < allocated
            or reserved > total
        ):
            raise ValueError(
                "CUDA peak counters violate 0 <= allocated <= reserved <= total"
            )
    except Exception as exc:
        state.update(
            {
                "observation_status": "capture_error",
                "max_memory_allocated_bytes": None,
                "max_memory_reserved_bytes": None,
                "capture_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return state
    state.update(
        {
            "observation_status": "captured",
            "max_memory_allocated_bytes": allocated,
            "max_memory_reserved_bytes": reserved,
        }
    )
    state.pop("capture_error", None)
    return state


def _cuda_peak_memory_receipt_valid(
    value: Any,
    *,
    require_captured: bool,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("schema") != CUDA_PEAK_MEMORY_SCHEMA:
        return False
    if not require_captured:
        if value.get("observation_status") == "unavailable":
            return bool(
                value.get("available") is False
                and value.get("reset_performed") is False
                and value.get("logical_device_index") is None
                and value.get("device_total_bytes") is None
                and value.get("max_memory_allocated_bytes") is None
                and value.get("max_memory_reserved_bytes") is None
                and isinstance(value.get("unavailable_reason"), str)
                and bool(value["unavailable_reason"])
            )
    integer_fields = (
        "logical_device_index",
        "device_total_bytes",
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
    )
    if (
        value.get("observation_status") != "captured"
        or value.get("available") is not True
        or value.get("reset_performed") is not True
        or any(
            isinstance(value.get(key), bool)
            or not isinstance(value.get(key), int)
            for key in integer_fields
        )
    ):
        return False
    device_index = int(value["logical_device_index"])
    total = int(value["device_total_bytes"])
    allocated = int(value["max_memory_allocated_bytes"])
    reserved = int(value["max_memory_reserved_bytes"])
    return bool(
        device_index >= 0
        and total > 0
        and 0 <= allocated <= reserved <= total
        and value.get("unavailable_reason") is None
        and "capture_error" not in value
    )


def _summarize_cuda_peak_memory(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    captured: list[Mapping[str, Any]] = []
    unavailable = 0
    errors = 0
    missing = 0
    for item in results:
        observation = item.get("cuda_peak_memory")
        if observation is None:
            child = item.get("result")
            observation = (
                child.get("cuda_peak_memory")
                if isinstance(child, Mapping)
                else None
            )
        if _cuda_peak_memory_receipt_valid(
            observation,
            require_captured=True,
        ):
            assert isinstance(observation, Mapping)
            captured.append(observation)
        elif _cuda_peak_memory_receipt_valid(
            observation,
            require_captured=False,
        ):
            unavailable += 1
        elif isinstance(observation, Mapping):
            errors += 1
        else:
            missing += 1
    totals = sorted(
        {int(item["device_total_bytes"]) for item in captured}
    )
    return {
        "schema": CUDA_PEAK_MEMORY_SUMMARY_SCHEMA,
        "policy": _cuda_peak_memory_policy(),
        "instance_result_count": len(results),
        "captured_count": len(captured),
        "unavailable_count": unavailable,
        "error_count": errors,
        "missing_count": missing,
        "all_instance_results_captured": (
            len(captured) == len(results) and bool(results)
        ),
        "max_memory_allocated_bytes": (
            max(
                int(item["max_memory_allocated_bytes"])
                for item in captured
            )
            if captured
            else None
        ),
        "max_memory_reserved_bytes": (
            max(
                int(item["max_memory_reserved_bytes"])
                for item in captured
            )
            if captured
            else None
        ),
        "observed_device_total_bytes": totals,
        "device_total_bytes": totals[0] if len(totals) == 1 else None,
    }


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield mappings without treating key names as asserted conditions."""

    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_mappings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_mappings(item)


def _metadata_reason_values(metadata: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    reason_keys = {
        "reason",
        "audit_reason",
        "fatal_reason",
        "lp_status",
        "solver_status",
    }
    for mapping in _walk_mappings(metadata):
        for key in reason_keys:
            value = mapping.get(key)
            if isinstance(value, str):
                values.add(value.strip().lower())
    return values


def _metadata_true_flag(
    metadata: Mapping[str, Any],
    names: set[str],
) -> Optional[str]:
    for mapping in _walk_mappings(metadata):
        for key, value in mapping.items():
            if str(key).lower() in names and value is True:
                return str(key).lower()
    return None


def _resource_error(result: Mapping[str, Any]) -> Optional[str]:
    error = result.get("error")
    if not isinstance(error, Mapping):
        return None
    error_type = str(error.get("type", "")).lower()
    message = str(error.get("message", "")).lower()
    resource_tokens = (
        "outofmemory",
        "out of memory",
        "memoryerror",
        "cuda is unavailable",
        "cuda unavailable",
        "no cuda-capable",
        "cuda driver",
        "cudnn_status_alloc_failed",
    )
    joined = f"{error_type} {message}"
    if any(token in joined for token in resource_tokens):
        return f"{error.get('type', 'ResourceError')}:{error.get('message', '')}"
    return None


def _strict_replay_receipt_valid(
    result: Mapping[str, Any],
    receipt: Any,
) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    integrity = result.get("input_integrity")
    expected = (
        integrity.get("expected")
        if isinstance(integrity, Mapping)
        else None
    )
    if not isinstance(expected, Mapping):
        return False
    required_true = (
        "valid_counterexample",
        "property_evaluated",
        "property_holds",
        "ort_executed",
        "raw_spec_evaluated",
        "zero_tolerance_holds",
        "replay_completed",
    )
    return bool(
        receipt.get("authority")
        == "onnxruntime_cpu_raw_vnnlib_zero_tolerance"
        and receipt.get("tolerance") == 0.0
        and all(receipt.get(key) is True for key in required_true)
        and receipt.get("model_sha256") == expected.get("onnx")
        and receipt.get("vnnlib_sha256") == expected.get("vnnlib")
    )


def _classify_result(result: Mapping[str, Any]) -> ResultClassification:
    """Classify a worker receipt with exact, truth-aware P0 semantics."""

    worker_state = result.get("worker_state")
    if worker_state != "completed":
        resource_reason = _resource_error(result)
        if resource_reason is not None:
            return ResultClassification(
                "BLOCKED_RESOURCE", resource_reason, False
            )
        error = result.get("error")
        message = (
            str(error.get("message", ""))
            if isinstance(error, Mapping)
            else ""
        )
        if "engine is not connected" in message.lower():
            return ResultClassification(
                "BLOCKED_ENGINE", f"worker_engine_error:{message}", False
            )
        return ResultClassification(
            "FAIL_ERROR", "worker_failed_or_missing_receipt", False
        )

    status = str(result.get("status", "")).strip().lower()
    metadata_raw = result.get("metadata")
    if not isinstance(metadata_raw, Mapping):
        return ResultClassification(
            "FAIL_ERROR", "metadata_missing_or_not_mapping", False
        )
    metadata = metadata_raw
    if status in ERROR_STATUSES:
        return ResultClassification("FAIL_ERROR", f"verifier:{status}", False)
    if status not in {"certified", "falsified", "unknown", "timeout"}:
        return ResultClassification(
            "FAIL_ERROR", f"unexpected_verdict_status:{status!r}", False
        )

    reasons = _metadata_reason_values(metadata)
    p0_reason_exact = {
        "hybridz_replay_conflict",
        "hybridz_representation_drop",
        "hybridz_operator_build_failed",
        "operator_hz_build_failed",
        "base_feasibility_conflict",
        "lp_base_feasibility_conflict",
        "soundness_conflict",
        "representation_drop",
        "sparse_drop",
    }
    matched_reason = next(
        (
            reason
            for reason in sorted(reasons)
            if reason in p0_reason_exact
            or reason.startswith("hybridz_sparse_drop:")
        ),
        None,
    )
    if matched_reason is not None:
        return ResultClassification(
            "FAIL_P0", f"metadata_reason:{matched_reason}", False
        )
    true_flag = _metadata_true_flag(
        metadata,
        {
            "p0",
            "p0_latched",
            "global_p0_latched",
            "representation_drop",
            "sparse_drop",
            "soundness_conflict",
            "base_feasibility_conflict",
            "lp_base_feasibility_conflict",
        },
    )
    if true_flag is not None:
        return ResultClassification(
            "FAIL_P0", f"metadata_true_flag:{true_flag}", False
        )

    expected_engine = str(result.get("expected_engine", "")).strip()
    solver = str(metadata.get("solver", "")).strip().lower()
    actual_engine = str(metadata.get("engine", "")).strip()
    hz_verdict = str(metadata.get("hz_verdict", "")).strip().upper()
    replay_state = str(metadata.get("hz_independent_replay", "")).strip()
    has_counterexample = result.get("has_counterexample")

    # Any conclusive receipt must be tied to the exact strict HybridZ engine.
    # Otherwise a fallback can masquerade as a gate result.
    if status != "timeout" and (
        solver != "hybridz"
        or not expected_engine
        or actual_engine != expected_engine
    ):
        return ResultClassification(
            "FAIL_P0",
            "verdict_engine_metadata_mismatch",
            False,
        )

    if status == "certified":
        if (
            hz_verdict != "SAFE"
            or has_counterexample is not False
            or metadata.get("hz_has_witness") is not False
            or replay_state == "independent_replay_accepted"
            or "hz_replay_receipt" in metadata
        ):
            return ResultClassification(
                "FAIL_P0", "certified_metadata_inconsistent", False
            )
        return ResultClassification(None, None, True)

    if status == "falsified":
        receipt = metadata.get("hz_replay_receipt")
        replay_valid = _strict_replay_receipt_valid(result, receipt)
        if (
            hz_verdict != "UNSAFE"
            or replay_state != "independent_replay_accepted"
            or has_counterexample is not True
            or metadata.get("hz_has_witness") is not True
            or not replay_valid
        ):
            return ResultClassification(
                "FAIL_P0", "falsified_metadata_inconsistent", False
            )
        return ResultClassification(None, None, True)

    if replay_state == "independent_replay_accepted":
        return ResultClassification(
            "FAIL_P0",
            "accepted_replay_not_reported_falsified",
            False,
        )
    if has_counterexample is True:
        return ResultClassification(
            "FAIL_P0", "inconclusive_receipt_exposes_counterexample", False
        )
    if status == "unknown":
        expected_witness = hz_verdict == "UNSAFE"
        if (
            hz_verdict not in {"UNKNOWN", "UNSAFE"}
            or metadata.get("hz_has_witness") is not expected_witness
        ):
            return ResultClassification(
                "FAIL_P0", "unknown_metadata_inconsistent", False
            )
    # A relaxed HZ may produce a phantom UNSAFE candidate.  Strict replay
    # rejection demotes it to UNKNOWN and remains an ordinary stop-loss.
    return ResultClassification(None, None, False)


def _worker_payload_result(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_json(path, value)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _query_dual_receipt_valid(
    receipt: Any,
    *,
    schema: str,
) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    if not _is_sha256(claimed) or body.get("schema") != schema:
        return False
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest() == claimed


def _checksummed_json_receipt_valid(
    receipt: Any,
    *,
    schema: str,
) -> bool:
    """Validate one strict ASCII checksummed metadata receipt."""

    if not isinstance(receipt, Mapping):
        return False
    try:
        body = dict(receipt)
        claimed = body.pop("receipt_sha256", None)
        if (
            not _is_sha256(claimed)
            or body.get("schema") != schema
        ):
            return False
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError):
        return False
    return hashlib.sha256(encoded).hexdigest() == claimed


_PHASE_CLIQUE_TIMING_STAGES = (
    "raw_top1_seconds",
    "hardness_seconds",
    "focus_and_replay_seconds",
    "literal_selection_seconds",
    "k4_candidate_seconds",
    "materializer_and_recheck_seconds",
    "terminal_seal_seconds",
)

_PHASE_CLIQUE_MATERIALIZER_CAPS = {
    "max_parent_variables": 2_000_000,
    "max_parent_rows": 2_000_000,
    "max_parent_nonzeros": 50_000_000,
    "max_parent_buffer_items": 120_000_000,
    "max_top_literals": 4,
    "max_total_pairs": 6,
    "max_cliques": 1,
    "max_clique_search_nodes": 100_000,
    "max_source_terms": 128,
    "max_multiplier_bits": 256,
    "max_exact_bits": 4096,
    "max_exact_nonzeros": 200_000,
}


def _phase_clique_receipt_number_valid(
    value: Any,
    *,
    positive: bool = False,
) -> bool:
    """Reject bool, nonfinite values, and signed zero in receipt numbers."""

    if type(value) not in {int, float}:
        return False
    try:
        observed = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    if not math.isfinite(observed):
        return False
    if observed == 0.0 and math.copysign(1.0, observed) < 0.0:
        return False
    return observed > 0.0 if positive else observed >= 0.0


def _validate_operator_phase_clique_receipt(
    metadata: Mapping[str, Any],
    *,
    expected_seconds: float,
) -> None:
    """Bind the default-off or enabled K4 transaction to this worker."""

    schema = "act.operator_phase_clique_pipeline.v1"
    receipt = metadata.get(
        "operator_phase_clique_materialization"
    )
    if not _checksummed_json_receipt_valid(
        receipt,
        schema=schema,
    ):
        raise GateConfigError(
            "worker operator phase-clique receipt is missing or has an "
            "invalid checksum"
        )
    assert isinstance(receipt, Mapping)
    common_valid = bool(
        receipt.get("candidate_only") is True
        and receipt.get("proof_authority") is False
        and receipt.get("verdict_path")
        == "hz_objbound_decide_only"
        and receipt.get("candidate_budget_fraction") == 0.40
        and receipt.get("materializer_reserve_fraction") == 0.60
    )
    timings = receipt.get("timings")
    allowed_timing_keys = {
        "total_seconds",
        *_PHASE_CLIQUE_TIMING_STAGES,
    }
    timings_valid = bool(
        isinstance(timings, Mapping)
        and timings
        and "total_seconds" in timings
        and all(type(name) is str for name in timings)
        and set(timings).issubset(allowed_timing_keys)
        and all(
            _phase_clique_receipt_number_valid(value)
            for value in timings.values()
        )
    )
    if not common_valid or not timings_valid:
        raise GateConfigError(
            "worker operator phase-clique receipt has an invalid common "
            "transaction contract"
        )

    if expected_seconds == 0.0:
        expected_body = {
            "schema": schema,
            "enabled": False,
            "status": "no_op_disabled",
            "candidate_attempted": False,
            "candidate_only": True,
            "proof_authority": False,
            "identity_preserved": True,
            "materialized": False,
            "materialization_receipt_sha256": None,
            "verdict_path": "hz_objbound_decide_only",
            "candidate_budget_fraction": 0.40,
            "materializer_reserve_fraction": 0.60,
            "timings": {"total_seconds": 0.0},
        }
        expected_receipt = {
            **expected_body,
            "receipt_sha256": hashlib.sha256(
                json.dumps(
                    expected_body,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest(),
        }
        if _canonical_json(dict(receipt)) != _canonical_json(
            expected_receipt
        ):
            raise GateConfigError(
                "worker emitted a stale or noncanonical operator "
                "phase-clique receipt while the feature was disabled"
            )
        if "operator_phase_clique_solver_handoff" in metadata:
            raise GateConfigError(
                "worker emitted a stale operator phase-clique solver handoff "
                "while the feature was disabled"
            )
        return

    if (
        receipt.get("enabled") is not True
        or receipt.get("candidate_attempted") is not True
    ):
        raise GateConfigError(
            "enabled operator phase-clique experiment did not execute"
        )
    configured_tolerance = max(
        1.0e-6,
        min(0.05, 1.0e-3 * float(expected_seconds)),
    )
    total_seconds = timings.get("total_seconds")
    stage_seconds = sum(
        float(timings.get(name, 0.0))
        for name in _PHASE_CLIQUE_TIMING_STAGES
    )
    if (
        float(total_seconds) > float(expected_seconds) + configured_tolerance
        or any(
            float(value)
            > float(total_seconds) + configured_tolerance
            for value in timings.values()
        )
        or stage_seconds
        > float(total_seconds) + configured_tolerance
    ):
        raise GateConfigError(
            "worker operator phase-clique receipt exceeds its configured "
            "time limit or has impossible segmented timings"
        )
    handoff = metadata.get("operator_phase_clique_solver_handoff")
    handoff_materialized = receipt.get("materialized") is True
    handoff_semantic_digest = receipt.get(
        "fresh_semantic_digest"
        if handoff_materialized
        else "source_parent_semantic_digest"
    )
    handoff_body = {
        "schema": "verifier_operator_phase_clique_solver_handoff_v1",
        "status": "consumed_private",
        "proof_authority": False,
        "one_use_consumed": True,
        "owner_bound": True,
        "pid_bound": True,
        "private_core_readonly": True,
        "solver_hz_is_public_result_hz": False,
        "materialized": handoff_materialized,
        "pipeline_receipt_sha256": receipt.get("receipt_sha256"),
        "semantic_digest": handoff_semantic_digest,
        "verdict_path": "hz_objbound_decide_only",
    }
    expected_handoff = {
        **handoff_body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                handoff_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    }
    if (
        not _is_sha256(handoff_semantic_digest)
        or not _checksummed_json_receipt_valid(
            handoff,
            schema=(
                "verifier_operator_phase_clique_solver_handoff_v1"
            ),
        )
        or _canonical_json(dict(handoff))
        != _canonical_json(expected_handoff)
    ):
        raise GateConfigError(
            "worker operator phase-clique solver handoff is missing, "
            "malformed, or not bound to the pipeline receipt"
        )
    if receipt.get("materialized") is False:
        full_rival_count = receipt.get("full_rival_count")
        focused_encoded_row = receipt.get("focused_encoded_row")
        optional_fallback_digests = (
            "hardness_vector_digest",
            "focused_subset_digest",
            "selection_digest",
            "subset_binding_digest",
        )
        if (
            receipt.get("identity_preserved") is not True
            or receipt.get("status")
            not in {
                "baseline_fallback_error",
                "baseline_fallback_timeout",
                "baseline_fallback_no_k4_clique",
            }
            or receipt.get("materialization_receipt_sha256")
            is not None
            or receipt.get("materialization_receipt") is not None
            or type(receipt.get("fallback_reason")) is not str
            or not receipt.get("fallback_reason")
            or type(receipt.get("failed_stage")) is not str
            or not receipt.get("failed_stage")
            or type(receipt.get("error_type")) is not str
            or not receipt.get("error_type")
            or (
                full_rival_count is not None
                and (
                    type(full_rival_count) is not int
                    or full_rival_count < 1
                )
            )
            or (
                focused_encoded_row is not None
                and (
                    type(focused_encoded_row) is not int
                    or type(full_rival_count) is not int
                    or not 0 <= focused_encoded_row < full_rival_count
                )
            )
            or any(
                receipt.get(name) is not None
                and not _is_sha256(receipt.get(name))
                for name in optional_fallback_digests
            )
            or not _is_sha256(
                receipt.get("source_parent_semantic_digest")
            )
            or not _is_sha256(receipt.get("source_frame_digest"))
            or receipt.get("solver_handoff_status") != "issued"
            or receipt.get("solver_handoff_one_use") is not True
            or receipt.get("solver_handoff_owner_bound") is not True
            or receipt.get("solver_handoff_pid_bound") is not True
            or receipt.get("solver_handoff_private_core_readonly")
            is not True
        ):
            raise GateConfigError(
                "worker operator phase-clique fallback receipt is malformed"
            )
        return

    nested = receipt.get("materialization_receipt")
    digest_fields = (
        "full_batch_sha256",
        "full_live_assert_sha256",
        "full_property_digest",
        "interval_frame_sha256",
        "hardness_vector_digest",
        "focused_subset_digest",
        "selection_digest",
        "focused_property_digest",
        "subset_binding_digest",
        "source_parent_semantic_digest",
        "fresh_semantic_digest",
        "materialization_receipt_sha256",
    )
    source_rows = metadata.get("operator_source_n_ub")
    fresh_rows = metadata.get("operator_n_ub")
    full_rival_count = receipt.get("full_rival_count")
    focused_encoded_row = receipt.get("focused_encoded_row")
    initial_seconds = receipt.get("initial_budget_seconds")
    candidate_seconds = receipt.get("candidate_budget_seconds")
    reserve_seconds = receipt.get(
        "minimum_materializer_reserve_seconds"
    )
    candidate_elapsed = receipt.get(
        "candidate_elapsed_seconds"
    )
    fixed_counts = {
        "focus_count": 1,
        "ranked_literal_count": 4,
        "pair_count": 6,
        "certified_edge_count": 6,
        "clique_count": 1,
        "cut_row_count": 1,
    }
    success_timing_keys = {
        "total_seconds",
        *_PHASE_CLIQUE_TIMING_STAGES,
    }
    candidate_stage_seconds = sum(
        float(timings[name])
        for name in _PHASE_CLIQUE_TIMING_STAGES[:5]
        if name in timings
    )
    materializer_seconds = float(
        timings.get("materializer_and_recheck_seconds", 0.0)
    )
    terminal_seal_seconds = float(
        timings.get("terminal_seal_seconds", 0.0)
    )
    if (
        receipt.get("identity_preserved") is not False
        or receipt.get("materialized") is not True
        or receipt.get("status")
        != "fresh_verified_k4_clique_materialized"
        or any(
            type(receipt.get(name)) is not int
            or receipt.get(name) != expected
            for name, expected in fixed_counts.items()
        )
        or type(full_rival_count) is not int
        or full_rival_count < 1
        or type(focused_encoded_row) is not int
        or not 0 <= focused_encoded_row < full_rival_count
        or any(
            not _is_sha256(receipt.get(name))
            for name in digest_fields
        )
        or type(source_rows) is not int
        or source_rows < 0
        or type(fresh_rows) is not int
        or type(receipt.get("source_upper_rows")) is not int
        or type(receipt.get("fresh_upper_rows")) is not int
        or receipt.get("source_upper_rows") != source_rows
        or receipt.get("fresh_upper_rows") != fresh_rows
        or fresh_rows != source_rows + 1
        or not all(
            _phase_clique_receipt_number_valid(
                value,
                positive=True,
            )
            for value in (
                initial_seconds,
                candidate_seconds,
                reserve_seconds,
            )
        )
        or not _phase_clique_receipt_number_valid(candidate_elapsed)
        or float(initial_seconds)
        > float(expected_seconds) + configured_tolerance
        or not math.isclose(
            float(candidate_seconds),
            0.40 * float(initial_seconds),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(reserve_seconds),
            0.60 * float(initial_seconds),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or float(candidate_elapsed)
        > float(candidate_seconds) + configured_tolerance
        or float(total_seconds)
        > float(initial_seconds) + configured_tolerance
        or set(timings) != success_timing_keys
        or candidate_stage_seconds
        > float(candidate_elapsed) + configured_tolerance
        or (
            float(candidate_elapsed)
            + materializer_seconds
            + terminal_seal_seconds
        )
        > float(total_seconds) + configured_tolerance
        or receipt.get("solver_handoff_status") != "issued"
        or receipt.get("solver_handoff_one_use") is not True
        or receipt.get("solver_handoff_owner_bound") is not True
        or receipt.get("solver_handoff_pid_bound") is not True
        or receipt.get("solver_handoff_private_core_readonly") is not True
        or not _checksummed_json_receipt_valid(
            nested,
            schema=(
                "act.operator_exact_relu_phase_clique_"
                "materialization.v2"
            ),
        )
    ):
        raise GateConfigError(
            "worker operator phase-clique success receipt is malformed or "
            "not bound to the final Operator-HZ"
        )
    assert isinstance(nested, Mapping)
    nested_sha_fields = (
        "verified_snapshot_digest",
        "verified_result_digest",
        "parent_semantic_digest",
        "verified_cut_semantic_digest",
        "fresh_semantic_digest",
        "ordered_source_frame_sha256",
        "source_frame_digest",
        "fresh_frame_digest",
        "selection_digest",
        "focused_property_digest",
        "subset_binding_digest",
        "receipt_sha256",
    )
    caps = nested.get("caps")
    clique_ids = nested.get("clique_ids")
    cut_row_tags = nested.get("cut_row_tags")
    clique_id = (
        clique_ids[0]
        if type(clique_ids) is list and len(clique_ids) == 1
        else None
    )
    expected_tag = (
        f"operator_exact_relu_phase_clique_cut:v1:0:{clique_id}"
        if _is_sha256(clique_id)
        else None
    )
    if (
        nested.get("status")
        != "fresh_verified_clique_cuts_materialized"
        or nested.get("candidate_only") is not True
        or nested.get("proof_authority") is not False
        or nested.get("hardened_exact_result_verifier_passed")
        is not True
        or nested.get("one_use_snapshot_consumed") is not True
        or nested.get("verdict_path") != "hz_objbound_decide_only"
        or nested.get("receipt_sha256")
        != receipt.get("materialization_receipt_sha256")
        or nested.get("parent_semantic_digest")
        != receipt.get("source_parent_semantic_digest")
        or nested.get("fresh_semantic_digest")
        != receipt.get("fresh_semantic_digest")
        or nested.get("selection_digest")
        != receipt.get("selection_digest")
        or nested.get("focused_property_digest")
        != receipt.get("focused_property_digest")
        or nested.get("subset_binding_digest")
        != receipt.get("subset_binding_digest")
        or any(
            not _is_sha256(nested.get(name))
            for name in nested_sha_fields
        )
        or nested.get("verified_cut_semantic_digest")
        != receipt.get("fresh_semantic_digest")
        or type(nested.get("cut_row_count")) is not int
        or nested.get("cut_row_count") != 1
        or type(nested.get("source_upper_rows")) is not int
        or nested.get("source_upper_rows") != source_rows
        or type(nested.get("fresh_upper_rows")) is not int
        or nested.get("fresh_upper_rows") != fresh_rows
        or type(caps) is not dict
        or set(caps) != set(_PHASE_CLIQUE_MATERIALIZER_CAPS)
        or any(
            type(caps.get(name)) is not int
            or caps.get(name) != expected
            for name, expected in _PHASE_CLIQUE_MATERIALIZER_CAPS.items()
        )
        or type(clique_ids) is not list
        or len(clique_ids) != 1
        or not _is_sha256(clique_id)
        or type(cut_row_tags) is not list
        or cut_row_tags != [expected_tag]
        or nested.get("copied_parent_attributes")
        != [
            "full_col_ids",
            "operator_input_center",
            "operator_input_radius",
            "_solver_continuous_column_layer_ids",
        ]
        or nested.get("row_prefix_frames") != "fresh_empty"
        or nested.get("incompatible_receipts")
        != "rejected_not_copied"
        or nested.get("constructive_nonempty_reissued") is not True
        or nested.get("constructive_nonempty_scope")
        != "private_solver_handoff_only"
        or nested.get("public_constructive_nonempty_token") != "absent"
        or nested.get("solver_caches_stats_safe_tokens")
        != "not_copied"
        or nested.get("solver_handoff_one_use") is not True
        or nested.get("solver_handoff_owner_bound") is not True
        or nested.get("solver_handoff_pid_bound") is not True
        or nested.get("solver_handoff_private_core_readonly") is not True
        or nested.get("constructive_nonempty_reason")
        != "operator_hz_redundant_exact_integer_phase_clique_cuts_v1"
        or nested.get("constructive_rule")
        != (
            "full_parent_exact_pair_conflicts_imply_redundant_"
            "integer_clique_rows"
        )
    ):
        raise GateConfigError(
            "worker operator phase-clique nested materialization receipt "
            "does not match the top-level transaction"
        )


def _property_upper_sha256_from_hex(values: Any) -> Optional[str]:
    if not isinstance(values, list) or not values:
        return None
    binary = bytearray()
    try:
        for raw in values:
            if not isinstance(raw, str):
                return None
            value = float.fromhex(raw)
            if not math.isfinite(value):
                return None
            binary.extend(struct.pack("<d", value))
    except (OverflowError, ValueError):
        return None
    digest = hashlib.sha256()
    digest.update(
        json.dumps([len(values)], separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0<f8\0")
    digest.update(binary)
    return digest.hexdigest()


def _safe_query_dual_diagnostic(value: Any, *, limit: int) -> str:
    """Render one untrusted verifier field without terminal/control injection."""

    if value is None:
        raw = "<missing>"
    elif isinstance(value, str):
        raw = value
    else:
        try:
            raw = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            try:
                raw = repr(value)
            except Exception:  # pragma: no cover - hostile third-party object
                raw = f"<unprintable {type(value).__name__}>"
    escaped = raw.encode(
        "unicode_escape", errors="backslashreplace"
    ).decode("ascii")
    if len(escaped) <= limit:
        return escaped

    marker = "...<truncated>..."
    available = max(0, limit - len(marker))
    prefix = (available + 1) // 2
    suffix = available // 2
    return (
        escaped[:prefix]
        + marker
        + (escaped[-suffix:] if suffix else "")
    )


def _query_dual_fallback_diagnostic(
    transaction: Mapping[str, Any],
) -> str:
    """Preserve bounded query/operator fallback root-cause diagnostics."""

    def value(primary: str, alternate: str) -> Any:
        primary_value = transaction.get(primary)
        if primary_value is not None:
            return primary_value
        return transaction.get(alternate)

    fields = (
        (
            "observed_status",
            transaction.get("status"),
            QUERY_DUAL_DIAGNOSTIC_FIELD_LIMIT,
        ),
        (
            "elapsed_seconds",
            transaction.get("elapsed_seconds"),
            QUERY_DUAL_DIAGNOSTIC_FIELD_LIMIT,
        ),
        (
            "error_type",
            value("error_type", "operator_error_type"),
            QUERY_DUAL_DIAGNOSTIC_FIELD_LIMIT,
        ),
        (
            "error_code",
            value("error_code", "operator_error_code"),
            QUERY_DUAL_DIAGNOSTIC_FIELD_LIMIT,
        ),
        (
            "error",
            value("error", "operator_error"),
            QUERY_DUAL_DIAGNOSTIC_ERROR_LIMIT,
        ),
    )
    return "; ".join(
        f"{name}={_safe_query_dual_diagnostic(raw, limit=limit)}"
        for name, raw, limit in fields
    )


def _validate_query_dual_transaction_receipt(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Require an applied all-or-nothing authority or an explicit disable."""

    candidate_policy = _query_dual_candidate_policy()
    targets = _gate_query_dual_targets(
        payload.get("query_dual_feedback_targets"),
        context="worker transaction payload targets",
    )
    expected_steps = payload.get("query_dual_feedback_steps")
    expected_seconds = payload.get("query_dual_feedback_time_limit")
    expected_block = payload.get("query_dual_feedback_block_size")
    expected_device = payload.get("query_dual_feedback_device")
    transaction = metadata.get("query_dual_feedback_transaction")
    if not isinstance(transaction, Mapping):
        raise GateConfigError(
            "worker verifier query_dual_feedback_transaction is missing"
        )
    enabled = isinstance(expected_steps, int) and expected_steps > 0
    if enabled and (
        transaction.get("status") != "applied"
        or transaction.get("proof_authority") is not True
        or transaction.get("source") != "built_in_verify_once"
    ):
        raise GateConfigError(
            "enabled query-dual feedback did not produce an applied "
            "authoritative transaction; "
            f"{_query_dual_fallback_diagnostic(transaction)}"
        )
    common_match = bool(
        transaction.get("schema")
        == "verifier_query_dual_feedback_transaction_v1"
        and transaction.get("targets") == list(targets)
        and transaction.get("steps") == expected_steps
        and transaction.get("time_limit") == expected_seconds
        and transaction.get("block_size") == expected_block
        and transaction.get("device") == expected_device
    )
    if not common_match:
        raise GateConfigError(
            "worker query-dual transaction wrapper disagrees with the "
            "effective family config"
        )

    if not enabled:
        forbidden = {
            "pipeline_receipt",
            "target_stage_receipts",
            "property_stage_receipt",
            "operator_transaction_receipt_sha256",
        }
        if (
            transaction.get("status") != "disabled"
            or transaction.get("proof_authority") is not False
            or transaction.get("source") != "configuration"
            or transaction.get("reason") != "steps_zero"
            or any(key in transaction for key in forbidden)
        ):
            raise GateConfigError(
                "disabled query-dual family lacks an explicit proofless "
                "disabled transaction receipt"
            )
        operator_hz = metadata.get("operator_hz")
        if isinstance(operator_hz, Mapping):
            applied = operator_hz.get("verified_query_dual_feedback")
            if isinstance(applied, Mapping) and (
                applied.get("proof_authority") is True
                or applied.get("transaction_receipt_sha256") is not None
            ):
                raise GateConfigError(
                    "disabled query-dual family exposes operator authority"
                )
        return

    elapsed = transaction.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise GateConfigError(
            "query-dual transaction elapsed_seconds is invalid"
        )
    wrapper_protocol_fields = {
        "pipeline_schema": candidate_policy["pipeline_schema"],
        "target_stage_schema": candidate_policy["target_stage_schema"],
        "property_stage_schema": candidate_policy["property_stage_schema"],
        "candidate_schema": candidate_policy["candidate_schema"],
        "candidate_protocol": candidate_policy["candidate_protocol"],
        "candidate_non_authoritative_audit_fields": candidate_policy[
            "candidate_non_authoritative_audit_fields"
        ],
        "pipeline_non_authoritative_audit_fields": candidate_policy[
            "pipeline_non_authoritative_audit_fields"
        ],
        "replay_chunk_size": expected_block,
    }
    if any(
        transaction.get(key) != value
        for key, value in wrapper_protocol_fields.items()
    ):
        raise GateConfigError(
            "query-dual transaction wrapper does not commit to the fixed "
            "descriptor-only V2 receipt policy"
        )

    pipeline = transaction.get("pipeline_receipt")
    if not _query_dual_receipt_valid(
        pipeline,
        schema=str(candidate_policy["pipeline_schema"]),
    ):
        raise GateConfigError(
            "query-dual pipeline receipt checksum/schema is invalid"
        )
    assert isinstance(pipeline, Mapping)
    pipeline_sha = pipeline["receipt_sha256"]
    if (
        pipeline.get("status") != "verified"
        or pipeline.get("proof_authority") is not True
        or pipeline.get("transaction") != "all_or_nothing"
        or pipeline.get("ordinary_interval_facts_consumed") is not False
        or pipeline.get("target_relu_ids") != list(targets)
        or pipeline.get("steps") != expected_steps
        or pipeline.get("block_size") != expected_block
        or pipeline.get("replay_chunk_size") != expected_block
        or pipeline.get("candidate_device") != expected_device
        or pipeline.get("candidate_device_fallback") is not False
        or pipeline.get("completed_before_deadline") is not True
        or pipeline.get("candidate_schema")
        != candidate_policy["candidate_schema"]
        or pipeline.get("candidate_protocol")
        != candidate_policy["candidate_protocol"]
        or pipeline.get("non_authoritative_audit_fields")
        != candidate_policy["pipeline_non_authoritative_audit_fields"]
    ):
        raise GateConfigError(
            "query-dual pipeline receipt is incomplete or disagrees with "
            "the transaction wrapper"
        )

    target_receipts = transaction.get("target_stage_receipts")
    if (
        not isinstance(target_receipts, list)
        or len(target_receipts) != len(targets)
    ):
        raise GateConfigError(
            "query-dual target-stage coverage is incomplete"
        )
    target_candidate_hashes = pipeline.get(
        "target_candidate_receipt_sha256"
    )
    target_candidate_coverage_hashes = pipeline.get(
        "target_candidate_descriptor_coverage_sha256"
    )
    if (
        not isinstance(target_candidate_hashes, list)
        or len(target_candidate_hashes) != len(targets)
        or not all(_is_sha256(value) for value in target_candidate_hashes)
        or not isinstance(target_candidate_coverage_hashes, list)
        or len(target_candidate_coverage_hashes) != len(targets)
        or not all(
            _is_sha256(value)
            for value in target_candidate_coverage_hashes
        )
    ):
        raise GateConfigError(
            "query-dual pipeline target-candidate V2 hash summary is invalid"
        )
    stage_hashes: list[str] = []
    target_block_count = 0
    strict_improvements = 0
    for index, (target, receipt) in enumerate(
        zip(targets, target_receipts)
    ):
        if not _query_dual_receipt_valid(
            receipt,
            schema=str(candidate_policy["target_stage_schema"]),
        ):
            raise GateConfigError(
                "query-dual target-stage receipt checksum/schema is invalid"
            )
        assert isinstance(receipt, Mapping)
        block_hashes = receipt.get("block_receipt_sha256")
        candidate_hash = receipt.get("candidate_receipt_sha256")
        candidate_coverage_hash = receipt.get(
            "candidate_descriptor_coverage_sha256"
        )
        if (
            receipt.get("status")
            not in {"verified", "verified_no_improvement"}
            or receipt.get("proof_authority") is not True
            or receipt.get("stage_index") != index
            or receipt.get("target_relu_lid") != target
            or receipt.get("commit") != "atomic_whole_stage"
            or not isinstance(block_hashes, list)
            or not all(_is_sha256(value) for value in block_hashes)
            or receipt.get("candidate_schema")
            != candidate_policy["candidate_schema"]
            or receipt.get("candidate_protocol")
            != candidate_policy["candidate_protocol"]
            or not _is_sha256(candidate_hash)
            or not _is_sha256(candidate_coverage_hash)
            or candidate_hash != target_candidate_hashes[index]
            or candidate_coverage_hash
            != target_candidate_coverage_hashes[index]
            or any(
                not _is_sha256(receipt.get(key))
                for key in (
                    "parent_boxes_sha256",
                    "result_boxes_sha256",
                    "candidate_bounds_sha256",
                    "candidate_receipt_sha256",
                    "target_bounds_sha256",
                )
            )
        ):
            raise GateConfigError(
                "query-dual target-stage receipt is incomplete or reordered"
            )
        strict = receipt.get("strict_improvements")
        if isinstance(strict, bool) or not isinstance(strict, int) or strict < 0:
            raise GateConfigError(
                "query-dual target-stage strict improvement count is invalid"
            )
        expected_stage_status = (
            "verified" if strict > 0 else "verified_no_improvement"
        )
        if block_hashes:
            status_semantics_valid = (
                receipt.get("candidate_status")
                == candidate_policy["candidate_success_status"]
                and receipt.get("status") == expected_stage_status
            )
        else:
            status_semantics_valid = (
                receipt.get("candidate_status")
                == candidate_policy["target_empty_status"]
                and receipt.get("status") == "verified_no_improvement"
                and strict == 0
            )
        if not status_semantics_valid:
            raise GateConfigError(
                "query-dual target-stage candidate/block/status semantics "
                "do not match descriptor-only V2"
            )
        stage_hashes.append(str(receipt["receipt_sha256"]))
        target_block_count += len(block_hashes)
        strict_improvements += strict
    if pipeline.get("stage_receipt_sha256") != stage_hashes:
        raise GateConfigError(
            "query-dual pipeline/target-stage checksums disagree"
        )

    property_receipt = transaction.get("property_stage_receipt")
    if not _query_dual_receipt_valid(
        property_receipt,
        schema=str(candidate_policy["property_stage_schema"]),
    ):
        raise GateConfigError(
            "query-dual property receipt checksum/schema is invalid"
        )
    assert isinstance(property_receipt, Mapping)
    property_blocks = property_receipt.get("block_receipt_sha256")
    property_rows = property_receipt.get("property_rows")
    property_candidate_hash = property_receipt.get(
        "candidate_receipt_sha256"
    )
    property_candidate_coverage_hash = property_receipt.get(
        "candidate_descriptor_coverage_sha256"
    )
    if (
        property_receipt.get("status") != "verified"
        or property_receipt.get("proof_authority") is not True
        or property_receipt.get("coverage_complete") is not True
        or property_receipt.get("direction") != "UPPER"
        or property_receipt.get("quantity") != "C_y_minus_threshold"
        or property_receipt.get("objective") != "-C"
        or property_receipt.get("replay_query_bias") != "+threshold"
        or property_receipt.get("upper_reconstruction")
        != "-LB(-C_y+threshold)"
        or property_receipt.get("candidate_schema")
        != candidate_policy["candidate_schema"]
        or property_receipt.get("candidate_protocol")
        != candidate_policy["candidate_protocol"]
        or property_receipt.get("candidate_status")
        != candidate_policy["candidate_success_status"]
        or not _is_sha256(property_receipt.get("parent_boxes_sha256"))
        or not _is_sha256(property_receipt.get("candidate_bounds_sha256"))
        or not _is_sha256(property_candidate_hash)
        or not _is_sha256(property_candidate_coverage_hash)
        or not isinstance(property_blocks, list)
        or not property_blocks
        or not all(_is_sha256(value) for value in property_blocks)
        or isinstance(property_rows, bool)
        or not isinstance(property_rows, int)
        or property_rows <= 0
    ):
        raise GateConfigError(
            "query-dual property receipt coverage is incomplete"
        )
    if (
        pipeline.get("property_receipt_sha256")
        != property_receipt.get("receipt_sha256")
        or pipeline.get("property_candidate_receipt_sha256")
        != property_candidate_hash
        or pipeline.get(
            "property_candidate_descriptor_coverage_sha256"
        )
        != property_candidate_coverage_hash
        or pipeline.get("property_spec_sha256")
        != property_receipt.get("property_spec_sha256")
        or pipeline.get("property_upper_sha256")
        != property_receipt.get("property_upper_sha256")
    ):
        raise GateConfigError(
            "query-dual pipeline/property checksums disagree"
        )

    integer_counts = {
        "root_bounds_count": transaction.get("root_bounds_count"),
        "target_stage_count": transaction.get("target_stage_count"),
        "target_block_count": transaction.get("target_block_count"),
        "property_block_count": transaction.get("property_block_count"),
        "strict_improvements_total": transaction.get(
            "strict_improvements_total"
        ),
        "property_rows": transaction.get("property_rows"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_counts.values()
    ):
        raise GateConfigError("query-dual wrapper counts are invalid")
    if (
        integer_counts["root_bounds_count"] <= 0
        or integer_counts["target_stage_count"] != len(targets)
        or integer_counts["target_block_count"] != target_block_count
        or integer_counts["property_block_count"] != len(property_blocks)
        or integer_counts["strict_improvements_total"]
        != strict_improvements
        or integer_counts["property_rows"] != property_rows
    ):
        raise GateConfigError(
            "query-dual wrapper coverage/counts disagree with stage receipts"
        )

    property_upper_sha = transaction.get("property_upper_sha256")
    if (
        not _is_sha256(property_upper_sha)
        or property_upper_sha != pipeline.get("property_upper_sha256")
        or property_upper_sha
        != _property_upper_sha256_from_hex(
            transaction.get("property_upper_hex")
        )
        or len(transaction.get("property_upper_hex", [])) != property_rows
    ):
        raise GateConfigError(
            "query-dual property upper values/checksum disagree"
        )
    operator_sha = transaction.get(
        "operator_transaction_receipt_sha256"
    )
    operator_hz = metadata.get("operator_hz")
    initial_frame = pipeline.get("initial_preactivation_frame")
    property_only_mode = bool(
        not targets
        and isinstance(initial_frame, Mapping)
        and initial_frame.get("enabled") is True
    )
    operator_receipt = (
        operator_hz.get("verified_query_dual_feedback")
        if isinstance(operator_hz, Mapping)
        else None
    )
    if property_only_mode:
        exported_frame = (
            operator_hz.get("verified_preactivation_frame")
            if isinstance(operator_hz, Mapping)
            else None
        )
        if (
            not _is_sha256(operator_sha)
            or operator_sha != pipeline_sha
            or transaction.get("application_mode")
            != "property_only_post_operator_bound_frame"
            or not isinstance(exported_frame, Mapping)
            or exported_frame.get("schema")
            != "operator_hz_verified_preactivation_frame_v1"
            or exported_frame.get("proof_authority") is not True
            or exported_frame.get(
                "process_local_validation_required"
            )
            is not True
            or initial_frame.get("schema")
            != "query_dual_operator_hz_bound_frame_v1"
            or initial_frame.get("proof_authority") is not True
            or initial_frame.get("intersection_only") is not True
            or initial_frame.get(
                "target_replay_stages_required"
            )
            is not False
            or initial_frame.get("source_receipt_sha256")
            != exported_frame.get("receipt_sha256")
            or initial_frame.get("source_bounds_sha256")
            != exported_frame.get("bounds_sha256")
            or initial_frame.get("source_network_sha256")
            != exported_frame.get("network_sha256")
            or transaction.get(
                "operator_bound_frame_receipt_sha256"
            )
            != exported_frame.get("receipt_sha256")
        ):
            raise GateConfigError(
                "property-only query-dual bound-frame authority is "
                "missing or checksum-mismatched"
            )
    else:
        if (
            not _is_sha256(operator_sha)
            or operator_sha != pipeline_sha
            or not isinstance(operator_receipt, Mapping)
            or operator_receipt.get("schema")
            != "operator_hz_verified_query_dual_feedback_v1"
            or operator_receipt.get("proof_authority") is not True
            or operator_receipt.get("process_local_validation") is not True
            or operator_receipt.get(
                "receipt_rehydration_authority"
            )
            is not False
            or operator_receipt.get("target_relu_ids") != list(targets)
            or operator_receipt.get(
                "transaction_receipt_sha256"
            )
            != pipeline_sha
        ):
            raise GateConfigError(
                "query-dual operator transaction authority is missing or "
                "checksum-mismatched"
            )
        for key in (
            "root_boxes_sha256",
            "final_boxes_sha256",
            "property_spec_sha256",
            "property_upper_sha256",
        ):
            if (
                not _is_sha256(pipeline.get(key))
                or operator_receipt.get(key) != pipeline.get(key)
            ):
                raise GateConfigError(
                    f"query-dual operator/pipeline {key} mismatch"
                )


def _validate_worker_feature_receipts(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Reject missing, forged, or incomplete experimental-feature receipts."""

    expected_phase_clique_seconds = payload.get(
        "operator_phase_clique_time_limit"
    )
    observed_phase_clique_seconds = metadata.get(
        "cfg_operator_phase_clique_time_limit"
    )
    if (
        isinstance(expected_phase_clique_seconds, bool)
        or not isinstance(
            expected_phase_clique_seconds, (int, float)
        )
        or not math.isfinite(float(expected_phase_clique_seconds))
        or isinstance(observed_phase_clique_seconds, bool)
        or not isinstance(
            observed_phase_clique_seconds, (int, float)
        )
        or not math.isfinite(float(observed_phase_clique_seconds))
        or float(observed_phase_clique_seconds)
        != float(expected_phase_clique_seconds)
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_operator_phase_clique_time_limit is missing or disagrees "
            "with the payload"
        )
    _validate_operator_phase_clique_receipt(
        metadata,
        expected_seconds=float(expected_phase_clique_seconds),
    )

    expected_query_targets_raw = payload.get(
        "query_dual_feedback_targets"
    )
    expected_query_targets = _gate_query_dual_targets(
        expected_query_targets_raw,
        context="worker payload query-dual targets",
    )
    if (
        not isinstance(expected_query_targets_raw, list)
        or list(expected_query_targets) != expected_query_targets_raw
    ):
        raise GateConfigError(
            "worker payload query_dual_feedback_targets must be a canonical "
            "unique list"
        )
    observed_query_targets_raw = metadata.get(
        "cfg_query_dual_feedback_targets"
    )
    observed_query_targets = _gate_query_dual_targets(
        observed_query_targets_raw,
        context="worker verifier cfg_query_dual_feedback_targets",
    )
    if (
        not isinstance(observed_query_targets_raw, (list, tuple))
        or tuple(observed_query_targets) != expected_query_targets
        or list(observed_query_targets) != list(observed_query_targets_raw)
    ):
        raise GateConfigError(
            "worker verifier receipt cfg_query_dual_feedback_targets is "
            "missing, noncanonical, or disagrees with the family payload"
        )
    query_fields = (
        (
            "query_dual_feedback_steps",
            "cfg_query_dual_feedback_steps",
            int,
        ),
        (
            "query_dual_feedback_block_size",
            "cfg_query_dual_feedback_block_size",
            int,
        ),
        (
            "query_dual_feedback_device",
            "cfg_query_dual_feedback_device",
            str,
        ),
    )
    for payload_key, metadata_key, expected_type in query_fields:
        expected_value = payload.get(payload_key)
        observed_value = metadata.get(metadata_key)
        if expected_type is int and (
            isinstance(expected_value, bool)
            or isinstance(observed_value, bool)
        ):
            raise GateConfigError(
                f"worker verifier receipt {metadata_key} has invalid type"
            )
        if (
            not isinstance(expected_value, expected_type)
            or not isinstance(observed_value, expected_type)
            or observed_value != expected_value
        ):
            raise GateConfigError(
                f"worker verifier receipt {metadata_key} is missing or "
                "disagrees with the payload"
            )
    expected_query_seconds = payload.get(
        "query_dual_feedback_time_limit"
    )
    observed_query_seconds = metadata.get(
        "cfg_query_dual_feedback_time_limit"
    )
    if (
        isinstance(expected_query_seconds, bool)
        or not isinstance(expected_query_seconds, (int, float))
        or not math.isfinite(float(expected_query_seconds))
        or isinstance(observed_query_seconds, bool)
        or not isinstance(observed_query_seconds, (int, float))
        or not math.isfinite(float(observed_query_seconds))
        or float(observed_query_seconds) != float(expected_query_seconds)
    ):
        raise GateConfigError(
            "worker verifier receipt cfg_query_dual_feedback_time_limit is "
            "missing or disagrees with the payload"
        )
    _validate_query_dual_transaction_receipt(metadata, payload)

    expected_micro_rlt_cap = payload.get(
        "property_micro_rlt_product_cap"
    )
    observed_micro_rlt_cap = metadata.get(
        "cfg_property_micro_rlt_product_cap"
    )
    if (
        isinstance(expected_micro_rlt_cap, bool)
        or not isinstance(expected_micro_rlt_cap, int)
        or isinstance(observed_micro_rlt_cap, bool)
        or not isinstance(observed_micro_rlt_cap, int)
        or observed_micro_rlt_cap != expected_micro_rlt_cap
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_property_micro_rlt_product_cap is missing or disagrees "
            "with the payload"
        )
    expected_packet_mode = payload.get(
        "property_micro_rlt_packet_mode"
    )
    observed_packet_mode = metadata.get(
        "cfg_property_micro_rlt_packet_mode"
    )
    if (
        not isinstance(expected_packet_mode, str)
        or expected_packet_mode not in {"both", "first", "second"}
        or not isinstance(observed_packet_mode, str)
        or observed_packet_mode not in {"both", "first", "second"}
        or observed_packet_mode != expected_packet_mode
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_property_micro_rlt_packet_mode is missing or disagrees "
            "with the payload"
        )
    expected_micro_rlt_seconds = payload.get(
        "property_micro_rlt_parent_prefilter_seconds"
    )
    observed_micro_rlt_seconds = metadata.get(
        "cfg_property_micro_rlt_parent_prefilter_seconds"
    )
    if (
        isinstance(expected_micro_rlt_seconds, bool)
        or not isinstance(expected_micro_rlt_seconds, (int, float))
        or not math.isfinite(float(expected_micro_rlt_seconds))
        or isinstance(observed_micro_rlt_seconds, bool)
        or not isinstance(observed_micro_rlt_seconds, (int, float))
        or not math.isfinite(float(observed_micro_rlt_seconds))
        or float(observed_micro_rlt_seconds)
        != float(expected_micro_rlt_seconds)
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_property_micro_rlt_parent_prefilter_seconds is missing "
            "or disagrees with the payload"
        )
    expected_parent_only = payload.get(
        "property_micro_rlt_parent_only_diagnostic"
    )
    observed_parent_only = metadata.get(
        "cfg_property_micro_rlt_parent_only_diagnostic"
    )
    if (
        not isinstance(expected_parent_only, bool)
        or not isinstance(observed_parent_only, bool)
        or observed_parent_only is not expected_parent_only
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_property_micro_rlt_parent_only_diagnostic is missing or "
            "disagrees with the payload"
        )
    parent_only_receipt = metadata.get(
        "property_micro_rlt_parent_only_diagnostic"
    )
    if not expected_parent_only:
        if parent_only_receipt is not None:
            raise GateConfigError(
                "worker emitted a stale parent-only diagnostic receipt "
                "while the feature was disabled"
            )
    else:
        if not isinstance(parent_only_receipt, Mapping):
            raise GateConfigError(
                "worker parent-only diagnostic receipt is missing"
            )
        parent_call_count = parent_only_receipt.get(
            "parent_call_count"
        )
        if (
            parent_only_receipt.get("schema")
            != (
                "verifier_property_micro_rlt_"
                "parent_only_diagnostic_v1"
            )
            or parent_only_receipt.get("enabled") is not True
            or parent_only_receipt.get("diagnostic_only") is not True
            or parent_only_receipt.get("proof_authority") is not False
            or parent_only_receipt.get("verdict_forced_unknown")
            is not True
            or parent_only_receipt.get("phase_cover_attempted") is not False
            or parent_only_receipt.get("phase_children_created") != 0
            or parent_only_receipt.get("baseline_solver_attempted")
            is not False
            or isinstance(parent_call_count, bool)
            or not isinstance(parent_call_count, int)
            or parent_call_count not in {0, 1}
        ):
            raise GateConfigError(
                "worker parent-only diagnostic contract is malformed"
            )
        parent_only_payload = dict(parent_only_receipt)
        parent_only_sha256 = parent_only_payload.pop(
            "receipt_sha256", None
        )
        if (
            not _is_sha256(parent_only_sha256)
            or _sha256_bytes(_canonical_json(parent_only_payload))
            != parent_only_sha256
        ):
            raise GateConfigError(
                "worker parent-only diagnostic checksum is invalid"
            )

        operator_hz = metadata.get("operator_hz")
        operator_micro_rlt = (
            operator_hz.get("property_micro_rlt")
            if isinstance(operator_hz, Mapping)
            else None
        )
        if not isinstance(operator_micro_rlt, Mapping):
            raise GateConfigError(
                "worker parent-only diagnostic operator receipt is missing"
            )
        operator_payload = dict(operator_micro_rlt)
        operator_sha256 = operator_payload.pop(
            "receipt_sha256", None
        )
        binding_failures = []
        if (
            operator_micro_rlt.get("schema")
            != "operator_hz_property_micro_rlt_v1"
        ):
            binding_failures.append("schema")
        if operator_micro_rlt.get("enabled") is not True:
            binding_failures.append("enabled")
        if (
            operator_micro_rlt.get(
                "requested_product_factor_cap"
            )
            != expected_micro_rlt_cap
        ):
            binding_failures.append("requested_product_factor_cap")
        if not _is_sha256(operator_sha256):
            binding_failures.append("receipt_sha256_shape")
        elif (
            _sha256_bytes(_canonical_json(operator_payload))
            != operator_sha256
        ):
            binding_failures.append("receipt_sha256_content")
        if (
            parent_only_receipt.get("operator_receipt_status")
            != operator_micro_rlt.get("status")
        ):
            binding_failures.append("operator_receipt_status")
        if (
            parent_only_receipt.get("operator_receipt_sha256")
            != operator_sha256
        ):
            binding_failures.append("operator_receipt_sha256")
        if (
            parent_only_receipt.get(
                "operator_live_validation_passed"
            )
            is not operator_micro_rlt.get(
                "live_result_validation_passed"
            )
        ):
            binding_failures.append(
                "operator_live_validation_passed"
            )
        if binding_failures:
            raise GateConfigError(
                "worker parent-only diagnostic/operator binding is invalid: "
                + ",".join(binding_failures)
            )

        parent_prefilter = metadata.get(
            "property_micro_rlt_parent_prefilter"
        )
        phase_split = metadata.get("property_phase_split")
        if (
            not isinstance(parent_prefilter, Mapping)
            or parent_prefilter.get("proof_authority") is not False
            or parent_prefilter.get("parent_call_count")
            != parent_call_count
            or parent_prefilter.get("status")
            != parent_only_receipt.get("parent_prefilter_status")
            or not isinstance(phase_split, Mapping)
            or phase_split.get("proof_authority") is not False
            or phase_split.get("diagnostic_only") is not True
            or phase_split.get("actual_child_count") != 0
            or phase_split.get("phase_enumeration_skipped") is not True
            or phase_split.get("children") != []
            or metadata.get("hz_verdict") != "UNKNOWN"
            or metadata.get("hz_has_witness") is not False
            or metadata.get("reason")
            != "property_micro_rlt_parent_only_diagnostic"
        ):
            raise GateConfigError(
                "worker parent-only diagnostic verifier binding is invalid"
            )

    expected_source_planes = payload.get(
        "property_tail_add_source_planes"
    )
    if not isinstance(expected_source_planes, bool):
        raise GateConfigError(
            "worker property_tail_add_source_planes payload must be boolean"
        )
    observed_source_planes = metadata.get(
        "cfg_property_tail_add_source_planes"
    )
    if (
        not isinstance(observed_source_planes, bool)
        or observed_source_planes is not expected_source_planes
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_property_tail_add_source_planes is missing or disagrees "
            "with the payload"
        )

    expected_mixture_bits = payload.get(
        "property_tail_mixture_grid_bits"
    )
    if isinstance(expected_mixture_bits, bool) or not isinstance(
        expected_mixture_bits, int
    ):
        raise GateConfigError(
            "worker property_tail_mixture_grid_bits payload must be an integer"
        )
    observed_mixture_bits = metadata.get(
        "cfg_property_tail_mixture_grid_bits"
    )
    if (
        isinstance(observed_mixture_bits, bool)
        or not isinstance(observed_mixture_bits, int)
        or observed_mixture_bits != expected_mixture_bits
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_property_tail_mixture_grid_bits is missing or disagrees "
            "with the payload"
        )

    expected_pairhull_budget = payload.get(
        "property_tail_pairhull_budget"
    )
    if isinstance(expected_pairhull_budget, bool) or not isinstance(
        expected_pairhull_budget, int
    ):
        raise GateConfigError(
            "worker property_tail_pairhull_budget payload must be an integer"
        )
    observed_pairhull_budget = metadata.get(
        "cfg_property_tail_pairhull_budget"
    )
    if (
        isinstance(observed_pairhull_budget, bool)
        or not isinstance(observed_pairhull_budget, int)
        or observed_pairhull_budget != expected_pairhull_budget
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_property_tail_pairhull_budget is missing or disagrees "
            "with the payload"
        )

    expected_pairhull_seconds = payload.get(
        "property_tail_pairhull_time_limit"
    )
    if isinstance(expected_pairhull_seconds, bool) or not isinstance(
        expected_pairhull_seconds, (int, float)
    ):
        raise GateConfigError(
            "worker property_tail_pairhull_time_limit payload must be numeric"
        )
    expected_pairhull_seconds = float(expected_pairhull_seconds)
    if not math.isfinite(expected_pairhull_seconds):
        raise GateConfigError(
            "worker property_tail_pairhull_time_limit payload must be finite"
        )
    observed_pairhull_seconds = metadata.get(
        "cfg_property_tail_pairhull_time_limit"
    )
    if (
        isinstance(observed_pairhull_seconds, bool)
        or not isinstance(observed_pairhull_seconds, (int, float))
        or not math.isfinite(float(observed_pairhull_seconds))
        or float(observed_pairhull_seconds) != expected_pairhull_seconds
    ):
        raise GateConfigError(
            "worker verifier receipt "
            "cfg_property_tail_pairhull_time_limit is missing or disagrees "
            "with the payload"
        )

    if "property_tail_suffix_blocks" in payload:
        expected_suffix_blocks = payload["property_tail_suffix_blocks"]
        observed_suffix_blocks = metadata.get(
            "cfg_property_tail_suffix_blocks"
        )
        if (
            isinstance(expected_suffix_blocks, bool)
            or not isinstance(expected_suffix_blocks, int)
            or isinstance(observed_suffix_blocks, bool)
            or not isinstance(observed_suffix_blocks, int)
            or observed_suffix_blocks != expected_suffix_blocks
        ):
            raise GateConfigError(
                "worker verifier receipt cfg_property_tail_suffix_blocks is "
                "missing or disagrees with the payload"
            )
    else:
        expected_suffix_blocks = 0
    suffix_alpha_fields = (
        (
            "property_tail_suffix_alpha_steps",
            "cfg_property_tail_suffix_alpha_steps",
            int,
            0,
        ),
        (
            "property_tail_suffix_alpha_device",
            "cfg_property_tail_suffix_alpha_device",
            str,
            "cuda",
        ),
    )
    for payload_key, metadata_key, expected_type, default in (
        suffix_alpha_fields
    ):
        if payload_key not in payload:
            continue
        expected_value = payload.get(payload_key, default)
        observed_value = metadata.get(metadata_key)
        if (
            expected_type is int
            and (
                isinstance(expected_value, bool)
                or isinstance(observed_value, bool)
            )
        ) or (
            not isinstance(expected_value, expected_type)
            or not isinstance(observed_value, expected_type)
            or observed_value != expected_value
        ):
            raise GateConfigError(
                f"worker verifier receipt {metadata_key} is missing or "
                "disagrees with the payload"
            )
    if "property_tail_suffix_alpha_time_limit" in payload:
        expected_suffix_alpha_seconds = payload[
            "property_tail_suffix_alpha_time_limit"
        ]
        observed_suffix_alpha_seconds = metadata.get(
            "cfg_property_tail_suffix_alpha_time_limit"
        )
        if (
            isinstance(expected_suffix_alpha_seconds, bool)
            or not isinstance(
                expected_suffix_alpha_seconds, (int, float)
            )
            or isinstance(observed_suffix_alpha_seconds, bool)
            or not isinstance(
                observed_suffix_alpha_seconds, (int, float)
            )
            or not math.isfinite(
                float(expected_suffix_alpha_seconds)
            )
            or float(observed_suffix_alpha_seconds)
            != float(expected_suffix_alpha_seconds)
        ):
            raise GateConfigError(
                "worker verifier receipt "
                "cfg_property_tail_suffix_alpha_time_limit is missing or "
                "disagrees with the payload"
            )

    if "operator_hz" not in metadata:
        # An earlier fail-closed result may have no operator receipt.  Preserve
        # that result and its verifier-provided reason for normal classification.
        return
    operator_metadata = metadata["operator_hz"]
    if not isinstance(operator_metadata, Mapping):
        raise GateConfigError(
            "worker operator_hz receipt must be a mapping"
        )

    if expected_source_planes:
        property_tail = operator_metadata.get("property_tail_upper")
        source_receipt = (
            property_tail.get("add_source_planes")
            if isinstance(property_tail, Mapping)
            else None
        )
        if (
            not isinstance(source_receipt, Mapping)
            or source_receipt.get("enabled") is not True
        ):
            raise GateConfigError(
                "worker operator receipt "
                "operator_hz.property_tail_upper.add_source_planes.enabled "
                "must be true"
            )

    if expected_pairhull_budget > 0:
        property_tail = operator_metadata.get("property_tail_upper")
        pairhull_receipt = (
            property_tail.get("pairhull_candidates")
            if isinstance(property_tail, Mapping)
            else None
        )
        alternative_rivals = (
            property_tail.get("alternative_plane_rival_ids")
            if isinstance(property_tail, Mapping)
            else None
        )
        alternative_kinds = (
            property_tail.get("alternative_plane_kinds")
            if isinstance(property_tail, Mapping)
            else None
        )
        rival_count = (
            property_tail.get("baseline_plane_count")
            if isinstance(property_tail, Mapping)
            else None
        )
        try:
            from act.back_end.verifier import (
                _validate_property_tail_pairhull_receipt,
            )

            pairhull_valid = (
                _validate_property_tail_pairhull_receipt(
                    pairhull_receipt,
                    requested_budget=expected_pairhull_budget,
                    requested_time_limit=expected_pairhull_seconds,
                    alternative_rivals=alternative_rivals,
                    alternative_kinds=alternative_kinds,
                    rival_count=rival_count,
                )
            )
        except Exception:
            pairhull_valid = False
        if not pairhull_valid:
            raise GateConfigError(
                "worker operator receipt "
                "operator_hz.property_tail_upper.pairhull_candidates "
                "is missing, forged, incomplete, or disagrees with the "
                "payload"
            )

    if expected_suffix_blocks > 0:
        expected_suffix_alpha_steps = int(
            payload.get("property_tail_suffix_alpha_steps", 0)
        )
        expected_suffix_alpha_time = float(
            payload.get(
                "property_tail_suffix_alpha_time_limit", 0.0
            )
        )
        expected_suffix_alpha_device = str(
            payload.get(
                "property_tail_suffix_alpha_device", "cuda"
            )
        )
        property_tail = operator_metadata.get("property_tail_upper")
        suffix_receipt = (
            property_tail.get("shared_suffix_replay")
            if isinstance(property_tail, Mapping)
            else None
        )
        if (
            not isinstance(suffix_receipt, Mapping)
            or suffix_receipt.get("schema")
            != "operator_hz_property_suffix_replay_v1"
            or suffix_receipt.get("requested_earlier_blocks")
            != expected_suffix_blocks
            or suffix_receipt.get("requested_alpha_steps")
            != expected_suffix_alpha_steps
            or float(
                suffix_receipt.get(
                    "requested_alpha_time_limit", float("nan")
                )
            )
            != expected_suffix_alpha_time
            or suffix_receipt.get("requested_alpha_device")
            != expected_suffix_alpha_device
            or suffix_receipt.get("status")
            not in {"applied", "error_fallback_baseline"}
            or (
                suffix_receipt.get("status") == "applied"
                and suffix_receipt.get("proof_authority") is not True
            )
            or (
                suffix_receipt.get("status")
                == "error_fallback_baseline"
                and suffix_receipt.get("proof_authority") is not False
            )
        ):
            raise GateConfigError(
                "worker operator receipt "
                "operator_hz.property_tail_upper.shared_suffix_replay is "
                "missing, forged, or disagrees with the payload"
            )

    if expected_mixture_bits > 0:
        solver_stats = metadata.get("hz_objbound_stats")
        mixture_receipt = (
            solver_stats.get("safe_row_dyadic_mixture")
            if isinstance(solver_stats, Mapping)
            else None
        )
        if not isinstance(mixture_receipt, Mapping):
            mixture_receipt = metadata.get("safe_row_dyadic_mixture")
        if (
            not isinstance(mixture_receipt, Mapping)
            or mixture_receipt.get("enabled") is not True
        ):
            raise GateConfigError(
                "worker solver receipt "
                "hz_objbound_stats.safe_row_dyadic_mixture.enabled "
                "must be true"
            )
        if (
            mixture_receipt.get("schema")
            != "hz_safe_group_dyadic_mixture_v1"
        ):
            raise GateConfigError(
                "worker solver receipt "
                "safe_row_dyadic_mixture.schema is missing or invalid"
            )
        observed_grid_bits = mixture_receipt.get("grid_bits")
        if (
            isinstance(observed_grid_bits, bool)
            or not isinstance(observed_grid_bits, int)
            or observed_grid_bits != expected_mixture_bits
        ):
            raise GateConfigError(
                "worker solver receipt "
                "safe_row_dyadic_mixture.grid_bits is missing or "
                "disagrees with the payload"
            )
        if mixture_receipt.get("candidate_only") is not True:
            raise GateConfigError(
                "worker solver receipt "
                "safe_row_dyadic_mixture.candidate_only must be true"
            )
        if mixture_receipt.get("proof_authority") is not False:
            raise GateConfigError(
                "worker solver receipt "
                "safe_row_dyadic_mixture.proof_authority must be false"
            )
        mixture_status = mixture_receipt.get("status")
        normalized_status = (
            mixture_status.strip().lower()
            if isinstance(mixture_status, str)
            else ""
        )
        if not normalized_status or normalized_status in {
            "pending",
            "disabled",
        }:
            raise GateConfigError(
                "worker solver receipt "
                "safe_row_dyadic_mixture.status must be an explicit "
                "non-pending, enabled outcome"
            )
        if normalized_status == "generated":
            for audit_field in (
                "stored_dyadic_weights_validated",
                "dyadic_convexity_validated",
                "exact_search_complete",
            ):
                if mixture_receipt.get(audit_field) is not True:
                    raise GateConfigError(
                        "worker generated mixture receipt "
                        f"safe_row_dyadic_mixture.{audit_field} "
                        "must be true"
                    )


def _worker_main(payload_path: Path, result_path: Path) -> int:
    """Fresh-process implementation for exactly one official CSV row."""

    started = time.monotonic()
    phase_times: dict[str, float] = {}
    torch_module: Optional[Any] = None
    cuda_peak_memory = _cuda_peak_memory_unavailable(
        "cuda_not_initialized"
    )
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise GateConfigError("invalid worker payload schema")
        raw_timeout = payload.get("wall_timeout_seconds")
        if isinstance(raw_timeout, bool) or not isinstance(
            raw_timeout, (int, float)
        ):
            raise GateConfigError("worker timeout must be numeric")
        timeout = float(raw_timeout)
        if not (0.0 < timeout <= 100.0):
            raise GateConfigError("worker timeout must be in (0, 100]")
        if payload.get("dtype") != "float64":
            raise GateConfigError(
                "strict gate worker requires dtype=float64"
            )
        if not isinstance(payload.get("operator_materialize_add"), bool):
            raise GateConfigError(
                "worker operator_materialize_add must be boolean"
            )
        raw_phase_clique_seconds = payload.get(
            "operator_phase_clique_time_limit"
        )
        if isinstance(raw_phase_clique_seconds, bool) or not isinstance(
            raw_phase_clique_seconds, (int, float)
        ):
            raise GateConfigError(
                "worker operator_phase_clique_time_limit must be numeric"
            )
        phase_clique_seconds = float(raw_phase_clique_seconds)
        if (
            not math.isfinite(phase_clique_seconds)
            or not 0.0 <= phase_clique_seconds <= 40.0
        ):
            raise GateConfigError(
                "worker operator_phase_clique_time_limit must lie in [0, 40]"
            )
        raw_operator_exact_budget = payload.get(
            "operator_exact_budget"
        )
        if isinstance(raw_operator_exact_budget, bool) or not isinstance(
            raw_operator_exact_budget, int
        ):
            raise GateConfigError(
                "worker operator_exact_budget must be an integer"
            )
        for key in (
            "preactivation_lp_budget",
            "property_correlation_budget",
            "property_residual_budget",
            "query_dual_feedback_steps",
            "query_dual_feedback_block_size",
            "gpu_dual_steps",
            "gpu_dual_row_topk",
        ):
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise GateConfigError(
                    f"worker {key} must be an integer"
                )
        for key in (
            "preactivation_lp_time_limit",
            "property_correlation_time_limit",
            "property_residual_time_limit",
            "query_dual_feedback_time_limit",
            "gpu_dual_time_limit",
        ):
            value = payload.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise GateConfigError(
                    f"worker {key} must be finite numeric"
                )
        for key in (
            "residual_phase_screen",
            "residual_bound_screen",
        ):
            if not isinstance(payload.get(key, False), bool):
                raise GateConfigError(
                    f"worker {key} must be boolean"
                )
        raw_query_targets = payload.get("query_dual_feedback_targets")
        if not isinstance(raw_query_targets, list):
            raise GateConfigError(
                "worker query_dual_feedback_targets must be a canonical list"
            )
        query_targets = _gate_query_dual_targets(
            raw_query_targets,
            context="worker query-dual targets",
        )
        if list(query_targets) != raw_query_targets:
            raise GateConfigError(
                "worker query_dual_feedback_targets must be unique and "
                "canonical"
            )
        query_status = payload.get("query_dual_feedback_status")
        if query_status not in {"gate1_candidate", "not_promoted"}:
            raise GateConfigError(
                "worker query_dual_feedback_status is invalid"
            )
        if query_status == "not_promoted" and query_targets:
            raise GateConfigError(
                "worker not_promoted family cannot carry query-dual targets"
            )
        if not isinstance(payload.get("property_tail_upper"), bool):
            raise GateConfigError(
                "worker property_tail_upper must be boolean"
            )
        if not isinstance(
            payload.get("property_tail_add_source_planes"), bool
        ):
            raise GateConfigError(
                "worker property_tail_add_source_planes must be boolean"
            )
        if (
            payload["property_tail_add_source_planes"]
            and not payload["property_tail_upper"]
        ):
            raise GateConfigError(
                "worker property_tail_add_source_planes requires "
                "property_tail_upper=true"
            )
        if (
            payload["property_tail_add_source_planes"]
            and not payload["operator_materialize_add"]
        ):
            raise GateConfigError(
                "worker property_tail_add_source_planes requires "
                "operator_materialize_add=true"
            )
        _validate_property_micro_rlt_settings(
            payload,
            context="worker ",
        )
        requested_query_steps = payload.get(
            "requested_query_dual_feedback_steps"
        )
        if (
            isinstance(requested_query_steps, bool)
            or not isinstance(requested_query_steps, int)
            or not 0 <= requested_query_steps <= 64
        ):
            raise GateConfigError(
                "worker requested query-dual steps must lie in [0, 64]"
            )
        requested_query_seconds_raw = payload.get(
            "requested_query_dual_feedback_time_limit"
        )
        if (
            isinstance(requested_query_seconds_raw, bool)
            or not isinstance(requested_query_seconds_raw, (int, float))
        ):
            raise GateConfigError(
                "worker requested query-dual time limit must be numeric"
            )
        requested_query_seconds = float(requested_query_seconds_raw)
        if (
            not math.isfinite(requested_query_seconds)
            or not 0.0 <= requested_query_seconds <= 20.0
            or (
                (requested_query_steps > 0)
                != (requested_query_seconds > 0.0)
            )
        ):
            raise GateConfigError(
                "worker requested query-dual steps/time are invalid"
            )
        expected_effective_steps = (
            requested_query_steps
            if query_status == "gate1_candidate"
            else 0
        )
        expected_effective_seconds = (
            requested_query_seconds
            if query_status == "gate1_candidate"
            else 0.0
        )
        if (
            payload.get("query_dual_feedback_steps")
            != expected_effective_steps
            or payload.get("query_dual_feedback_time_limit")
            != expected_effective_seconds
        ):
            raise GateConfigError(
                "worker family-effective query-dual settings disagree with "
                "requested settings and promotion status"
            )
        if requested_query_steps > 0:
            property_only_bound_replay = bool(
                payload.get("residual_bound_screen", False)
                and not query_targets
            )
            if (
                not property_only_bound_replay
                and payload["property_tail_upper"] is not True
            ):
                raise GateConfigError(
                    "worker requested query-dual feedback requires "
                    "property_tail_upper=true"
                )
            if (
                property_only_bound_replay
                and payload["property_tail_upper"] is not False
            ):
                raise GateConfigError(
                    "worker property-only query replay requires "
                    "property_tail_upper=false"
                )
            if payload.get("engine") != "operator_hz_objbound":
                raise GateConfigError(
                    "worker requested query-dual feedback requires the "
                    "operator engine"
                )
            if int(payload.get("operator_exact_budget", -1)) != 0:
                raise GateConfigError(
                    "worker requested query-dual feedback requires "
                    "operator_exact_budget=0"
                )
        # Validate the complete query-dual activation contract before CUDA,
        # model parsing, or any expensive work.
        HybridZConfig(
            engine=str(payload["engine"]),
            operator_exact_budget=raw_operator_exact_budget,
            operator_phase_clique_time_limit=phase_clique_seconds,
            operator_materialize_add=bool(
                payload["operator_materialize_add"]
            ),
            preactivation_lp_budget=payload[
                "preactivation_lp_budget"
            ],
            preactivation_lp_time_limit=payload[
                "preactivation_lp_time_limit"
            ],
            property_correlation_budget=payload.get(
                "property_correlation_budget", 0
            ),
            property_correlation_time_limit=payload.get(
                "property_correlation_time_limit", 0.0
            ),
            residual_phase_screen=payload.get(
                "residual_phase_screen", False
            ),
            residual_bound_screen=payload.get(
                "residual_bound_screen", False
            ),
            property_tail_upper=bool(payload["property_tail_upper"]),
            property_residual_budget=int(
                payload["property_residual_budget"]
            ),
            property_residual_time_limit=float(
                payload["property_residual_time_limit"]
            ),
            property_micro_rlt_product_cap=payload[
                "property_micro_rlt_product_cap"
            ],
            property_micro_rlt_packet_mode=payload[
                "property_micro_rlt_packet_mode"
            ],
            property_micro_rlt_parent_prefilter_seconds=payload[
                "property_micro_rlt_parent_prefilter_seconds"
            ],
            property_micro_rlt_parent_only_diagnostic=payload[
                "property_micro_rlt_parent_only_diagnostic"
            ],
            query_dual_feedback_targets=query_targets,
            query_dual_feedback_steps=payload[
                "query_dual_feedback_steps"
            ],
            query_dual_feedback_time_limit=payload[
                "query_dual_feedback_time_limit"
            ],
            query_dual_feedback_block_size=payload[
                "query_dual_feedback_block_size"
            ],
            query_dual_feedback_device=payload[
                "query_dual_feedback_device"
            ],
            gpu_dual_steps=payload["gpu_dual_steps"],
            gpu_dual_time_limit=payload["gpu_dual_time_limit"],
            gpu_dual_row_topk=payload["gpu_dual_row_topk"],
            property_tail_suffix_blocks=int(
                payload.get("property_tail_suffix_blocks", 0)
            ),
            property_tail_suffix_alpha_steps=int(
                payload.get("property_tail_suffix_alpha_steps", 0)
            ),
            property_tail_suffix_alpha_time_limit=float(
                payload.get(
                    "property_tail_suffix_alpha_time_limit", 0.0
                )
            ),
            property_tail_suffix_alpha_device=str(
                payload.get(
                    "property_tail_suffix_alpha_device", "cuda"
                )
            ),
        )
        if payload.get("property_tail_alpha_device") not in {
            "cpu",
            "cuda",
        }:
            raise GateConfigError(
                "worker property_tail_alpha_device must be cpu or cuda"
            )
        mixture_grid_bits = payload.get(
            "property_tail_mixture_grid_bits"
        )
        if isinstance(mixture_grid_bits, bool) or not isinstance(
            mixture_grid_bits, int
        ):
            raise GateConfigError(
                "worker property_tail_mixture_grid_bits must be an integer"
            )
        if not 0 <= mixture_grid_bits <= 24:
            raise GateConfigError(
                "worker property_tail_mixture_grid_bits must lie in [0, 24]"
            )
        if mixture_grid_bits > 0 and not payload["property_tail_upper"]:
            raise GateConfigError(
                "worker property_tail_mixture_grid_bits>0 requires "
                "property_tail_upper=true"
            )
        if (
            mixture_grid_bits > 0
            and int(payload["property_tail_alpha_steps"]) <= 0
        ):
            raise GateConfigError(
                "worker property_tail_mixture_grid_bits>0 requires "
                "property_tail_alpha_steps>0"
            )
        if (
            mixture_grid_bits > 0
            and float(payload["property_tail_alpha_time_limit"]) <= 0.0
        ):
            raise GateConfigError(
                "worker property_tail_mixture_grid_bits>0 requires "
                "property_tail_alpha_time_limit>0"
            )
        if (
            mixture_grid_bits > 0
            and int(payload["operator_exact_budget"]) != 0
        ):
            raise GateConfigError(
                "worker property_tail_mixture_grid_bits>0 requires "
                "operator_exact_budget=0"
            )
        pairhull_budget = payload.get("property_tail_pairhull_budget")
        if isinstance(pairhull_budget, bool) or not isinstance(
            pairhull_budget, int
        ):
            raise GateConfigError(
                "worker property_tail_pairhull_budget must be an integer"
            )
        if not 0 <= pairhull_budget <= 8:
            raise GateConfigError(
                "worker property_tail_pairhull_budget must lie in [0, 8]"
            )
        raw_pairhull_seconds = payload.get(
            "property_tail_pairhull_time_limit"
        )
        if isinstance(raw_pairhull_seconds, bool) or not isinstance(
            raw_pairhull_seconds, (int, float)
        ):
            raise GateConfigError(
                "worker property_tail_pairhull_time_limit must be numeric"
            )
        pairhull_seconds = float(raw_pairhull_seconds)
        if (
            not math.isfinite(pairhull_seconds)
            or not 0.0 <= pairhull_seconds <= 1.5
        ):
            raise GateConfigError(
                "worker property_tail_pairhull_time_limit must be finite "
                "and lie in [0, 1.5]"
            )
        if (pairhull_budget > 0) != (pairhull_seconds > 0.0):
            raise GateConfigError(
                "worker property-tail PairHull budget and time limit must "
                "be enabled together"
            )
        if pairhull_budget > 0 and not payload["property_tail_upper"]:
            raise GateConfigError(
                "worker property_tail_pairhull_budget>0 requires "
                "property_tail_upper=true"
            )
        if (
            pairhull_budget > 0
            and int(payload["operator_exact_budget"]) != 0
        ):
            raise GateConfigError(
                "worker property_tail_pairhull_budget>0 requires "
                "operator_exact_budget=0"
            )
        suffix_blocks = payload.get("property_tail_suffix_blocks")
        if isinstance(suffix_blocks, bool) or not isinstance(
            suffix_blocks, int
        ):
            raise GateConfigError(
                "worker property_tail_suffix_blocks must be an integer"
            )
        if not 0 <= suffix_blocks <= 8:
            raise GateConfigError(
                "worker property_tail_suffix_blocks must lie in [0, 8]"
            )
        if suffix_blocks > 0 and not payload["property_tail_upper"]:
            raise GateConfigError(
                "worker property_tail_suffix_blocks>0 requires "
                "property_tail_upper=true"
            )
        if suffix_blocks > 0 and not payload["operator_materialize_add"]:
            raise GateConfigError(
                "worker property_tail_suffix_blocks>0 requires "
                "operator_materialize_add=true"
            )
        suffix_alpha_steps = payload.get(
            "property_tail_suffix_alpha_steps"
        )
        if isinstance(suffix_alpha_steps, bool) or not isinstance(
            suffix_alpha_steps, int
        ) or not 0 <= suffix_alpha_steps <= 64:
            raise GateConfigError(
                "worker property_tail_suffix_alpha_steps must lie in [0, 64]"
            )
        suffix_alpha_seconds_raw = payload.get(
            "property_tail_suffix_alpha_time_limit"
        )
        if isinstance(suffix_alpha_seconds_raw, bool) or not isinstance(
            suffix_alpha_seconds_raw, (int, float)
        ):
            raise GateConfigError(
                "worker property_tail_suffix_alpha_time_limit must be numeric"
            )
        suffix_alpha_seconds = float(suffix_alpha_seconds_raw)
        if (
            not math.isfinite(suffix_alpha_seconds)
            or not 0.0 <= suffix_alpha_seconds <= 20.0
            or (
                (suffix_alpha_steps > 0)
                != (suffix_alpha_seconds > 0.0)
            )
        ):
            raise GateConfigError(
                "worker property-tail suffix alpha settings are invalid"
            )
        if suffix_alpha_steps > 0 and suffix_blocks <= 0:
            raise GateConfigError(
                "worker property_tail_suffix_alpha_steps>0 requires "
                "property_tail_suffix_blocks>0"
            )
        if payload.get("property_tail_suffix_alpha_device") not in {
            "cpu",
            "cuda",
        }:
            raise GateConfigError(
                "worker property_tail_suffix_alpha_device must be cpu or cuda"
            )
        row_workers = int(payload["row_workers"])
        total_threads = int(payload["total_solver_threads"])
        if not (1 <= row_workers <= 4 and 1 <= total_threads <= 20):
            raise GateConfigError("worker thread caps are invalid")
        if row_workers > total_threads:
            raise GateConfigError("row workers exceed the total thread cap")
        solver_threads = max(1, total_threads // row_workers)
        if solver_threads * row_workers > total_threads:
            raise GateConfigError("worker solver thread product exceeds cap")

        expected_environment = payload.get("fixed_worker_environment")
        if not isinstance(expected_environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in expected_environment.items()
        ):
            raise GateConfigError("worker fixed environment is invalid")
        canonical_environment = _fixed_worker_environment(payload)
        if expected_environment != canonical_environment:
            raise GateConfigError(
                "worker fixed environment differs from canonical gate policy"
            )
        unexpected_hz = sorted(
            key
            for key in os.environ
            if key.startswith("HZ_") and key not in expected_environment
        )
        mismatched_environment = {
            key: {"expected": value, "actual": os.environ.get(key)}
            for key, value in expected_environment.items()
            if os.environ.get(key) != value
        }
        if unexpected_hz or mismatched_environment:
            raise GateConfigError(
                "worker numerical environment mismatch: "
                f"unexpected_hz={unexpected_hz}, "
                f"mismatched={mismatched_environment}"
            )

        expected_inputs = payload.get("expected_input_sha256")
        if (
            not isinstance(expected_inputs, dict)
            or set(expected_inputs) != {"onnx", "vnnlib"}
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
                for value in expected_inputs.values()
            )
        ):
            raise GateConfigError("worker expected input hashes are invalid")
        input_hash_started = time.monotonic()
        observed_inputs_before = {
            "onnx": _sha256_file(Path(payload["onnx_path"])),
            "vnnlib": _sha256_file(Path(payload["vnnlib_path"])),
        }
        phase_times["input_integrity_before_seconds"] = (
            time.monotonic() - input_hash_started
        )
        if observed_inputs_before != expected_inputs:
            raise GateConfigError(
                "worker input artifact hash mismatch before parse"
            )

        connected, connection_reason = _engine_connected(
            str(payload["engine"]),
            operator_phase_clique_time_limit=float(
                payload["operator_phase_clique_time_limit"]
            ),
        )
        if not connected:
            raise GateConfigError(
                f"HybridZ engine is not connected; refusing fallback: "
                f"{connection_reason}"
            )

        def remaining() -> float:
            return timeout - (time.monotonic() - started)

        def record_phase(name: str, phase_started: float) -> float:
            elapsed = float(max(0.0, time.monotonic() - phase_started))
            phase_times[name] = elapsed
            print(
                "[gate-worker-phase] "
                f"{name}={elapsed:.6f}s "
                f"total={max(0.0, time.monotonic() - started):.6f}s "
                f"remaining={max(0.0, remaining()):.6f}s",
                flush=True,
            )
            return elapsed

        import torch

        torch_module = torch
        if not torch.cuda.is_available():
            cuda_peak_memory = _cuda_peak_memory_unavailable(
                "cuda_unavailable"
            )
            raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
        torch.set_num_threads(total_threads)
        torch.set_num_interop_threads(1)

        from act.back_end.config import BackendConfig
        from act.back_end.transfer_functions import (
            set_solver_mode,
            set_transfer_function_mode,
        )
        from act.back_end.verifier import verify_once
        from act.front_end.model_synthesis import synthesize_models_from_specs
        from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
        from act.pipeline.verification.strict_replay import make_strict_replay
        from act.pipeline.verification.torch2act import TorchToACT
        from act.util.device_manager import initialize_device

        initialize_device("cuda", str(payload["dtype"]))
        cuda_peak_memory = _start_cuda_peak_memory(torch)
        if cuda_peak_memory.get("observation_status") != "tracking":
            raise GateConfigError(
                "CUDA peak-memory reset failed before model parsing: "
                f"{cuda_peak_memory}"
            )
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")

        phase_started = time.monotonic()
        spec_result = create_specs_from_paths(
            payload["onnx_path"],
            payload["vnnlib_path"],
            category=str(payload["family"]),
        )
        record_phase("parse_convert_specs_seconds", phase_started)
        if remaining() <= 0.0:
            raise TimeoutError("budget exhausted during parse/convert/spec creation")

        phase_started = time.monotonic()
        wrapped = synthesize_models_from_specs([spec_result])
        if len(wrapped) != 1:
            raise RuntimeError(
                f"large-classification sentinel expanded to {len(wrapped)} "
                "wrapped models; refusing partial/disjunct verification"
            )
        model = next(iter(wrapped.values()))
        target_dtype = (
            torch.float64
            if str(payload["dtype"]) == "float64"
            else torch.float32
        )
        model = model.to(
            device=torch.device("cuda"),
            dtype=target_dtype,
        )
        record_phase("synthesis_seconds", phase_started)
        if remaining() <= 0.0:
            raise TimeoutError("budget exhausted during model synthesis")

        phase_started = time.monotonic()
        net = TorchToACT(model).run()
        record_phase("torch2act_seconds", phase_started)
        # HiGHS can return a few hundred milliseconds after its requested
        # cutoff.  Keep enough outer-wall slack for the worker to serialize
        # the verifier/GPU receipts instead of being hard-killed with all
        # stage diagnostics lost.  This reserve is orchestration-only and is
        # never added back to the proof solver's budget.
        post_verdict_reserve = max(
            2.0,
            1.5 * phase_times["input_integrity_before_seconds"] + 0.05,
        )
        if remaining() <= post_verdict_reserve:
            raise TimeoutError("no verdict budget remains after TorchToACT")

        verdict_budget = max(0.01, remaining() - post_verdict_reserve)
        hybridz_cfg = HybridZConfig(
            timeout=verdict_budget,
            engine=str(payload["engine"]),
            operator_exact_budget=int(payload["operator_exact_budget"]),
            operator_phase_clique_time_limit=float(
                payload["operator_phase_clique_time_limit"]
            ),
            operator_materialize_add=bool(
                payload["operator_materialize_add"]
            ),
            preactivation_lp_budget=int(
                payload["preactivation_lp_budget"]
            ),
            preactivation_lp_time_limit=float(
                payload["preactivation_lp_time_limit"]
            ),
            property_correlation_budget=int(
                payload.get("property_correlation_budget", 0)
            ),
            property_correlation_time_limit=float(
                payload.get("property_correlation_time_limit", 0.0)
            ),
            residual_phase_screen=bool(
                payload.get("residual_phase_screen", False)
            ),
            residual_bound_screen=bool(
                payload.get("residual_bound_screen", False)
            ),
            property_residual_budget=int(
                payload["property_residual_budget"]
            ),
            property_residual_time_limit=float(
                payload["property_residual_time_limit"]
            ),
            property_residual_max_adjoint_cells=int(
                payload["property_residual_max_adjoint_cells"]
            ),
            property_residual_pool_per_rival=int(
                payload["property_residual_pool_per_rival"]
            ),
            property_tail_upper=bool(payload["property_tail_upper"]),
            property_micro_rlt_product_cap=payload[
                "property_micro_rlt_product_cap"
            ],
            property_micro_rlt_packet_mode=payload[
                "property_micro_rlt_packet_mode"
            ],
            property_micro_rlt_parent_prefilter_seconds=payload[
                "property_micro_rlt_parent_prefilter_seconds"
            ],
            property_micro_rlt_parent_only_diagnostic=payload[
                "property_micro_rlt_parent_only_diagnostic"
            ],
            property_tail_add_source_planes=bool(
                payload["property_tail_add_source_planes"]
            ),
            property_tail_alpha_steps=int(
                payload["property_tail_alpha_steps"]
            ),
            property_tail_alpha_time_limit=float(
                payload["property_tail_alpha_time_limit"]
            ),
            property_tail_alpha_learning_rate=float(
                payload["property_tail_alpha_learning_rate"]
            ),
            property_tail_alpha_max_cells=int(
                payload["property_tail_alpha_max_cells"]
            ),
            property_tail_alpha_device=str(
                payload["property_tail_alpha_device"]
            ),
            property_tail_mixture_grid_bits=int(
                payload["property_tail_mixture_grid_bits"]
            ),
            property_tail_pairhull_budget=int(
                payload["property_tail_pairhull_budget"]
            ),
            property_tail_pairhull_time_limit=float(
                payload["property_tail_pairhull_time_limit"]
            ),
            property_tail_suffix_blocks=int(
                payload["property_tail_suffix_blocks"]
            ),
            property_tail_suffix_alpha_steps=int(
                payload["property_tail_suffix_alpha_steps"]
            ),
            property_tail_suffix_alpha_time_limit=float(
                payload["property_tail_suffix_alpha_time_limit"]
            ),
            property_tail_suffix_alpha_device=str(
                payload["property_tail_suffix_alpha_device"]
            ),
            query_dual_feedback_targets=query_targets,
            query_dual_feedback_steps=int(
                payload["query_dual_feedback_steps"]
            ),
            query_dual_feedback_time_limit=float(
                payload["query_dual_feedback_time_limit"]
            ),
            query_dual_feedback_block_size=int(
                payload["query_dual_feedback_block_size"]
            ),
            query_dual_feedback_device=str(
                payload["query_dual_feedback_device"]
            ),
            gpu_dual_steps=int(payload["gpu_dual_steps"]),
            gpu_dual_time_limit=float(payload["gpu_dual_time_limit"]),
            gpu_dual_row_topk=int(payload["gpu_dual_row_topk"]),
            gpu_dual_learning_rate=float(
                payload["gpu_dual_learning_rate"]
            ),
            lp_prefilter_fraction=float(payload["lp_prefilter_fraction"]),
            lp_prefilter_max_seconds=float(
                payload["lp_prefilter_max_seconds"]
            ),
        )
        backend_cfg = BackendConfig(
            solver="hybridz",
            device="cuda",
            dtype=str(payload["dtype"]),
            timeout=verdict_budget,
            hybridz=hybridz_cfg,
        )
        replay = make_strict_replay(
            payload["onnx_path"], payload["vnnlib_path"]
        )
        phase_started = time.monotonic()
        print(
            "[gate-worker-phase] analysis_solver_replay_seconds=start "
            f"total={max(0.0, time.monotonic() - started):.6f}s "
            f"remaining={max(0.0, remaining()):.6f}s",
            flush=True,
        )
        results = verify_once(
            net,
            backend_cfg=backend_cfg,
            model_fn=model,
            counterexample_replay_fn=replay,
            raw_vnnlib_path=payload["vnnlib_path"],
            expected_raw_vnnlib_sha256=payload[
                "expected_input_sha256"
            ]["vnnlib"],
            fail_fast_on_query_dual_fallback=(
                int(payload["query_dual_feedback_steps"]) > 0
            ),
        )
        record_phase("analysis_solver_replay_seconds", phase_started)
        if len(results) != 1:
            raise RuntimeError(
                f"verify_once returned {len(results)} lanes, expected exactly one"
            )
        result = results[0]
        status = getattr(result.status, "value", str(result.status)).lower()
        if (
            payload["property_micro_rlt_parent_only_diagnostic"]
            and status in CONCLUSIVE
        ):
            raise GateConfigError(
                "property micro-RLT parent-only diagnostic returned a "
                f"forbidden conclusive verdict: {status}"
            )
        input_hash_started = time.monotonic()
        observed_inputs_after = {
            "onnx": _sha256_file(Path(payload["onnx_path"])),
            "vnnlib": _sha256_file(Path(payload["vnnlib_path"])),
        }
        phase_times["input_integrity_after_seconds"] = (
            time.monotonic() - input_hash_started
        )
        if observed_inputs_after != expected_inputs:
            raise GateConfigError(
                "worker input artifact hash changed during verification"
            )
        result_metadata = _json_safe(result.metadata)
        if not isinstance(result_metadata, dict):
            raise GateConfigError("worker result metadata must be a mapping")
        _validate_worker_feature_receipts(result_metadata, payload)
        cuda_peak_memory = _capture_cuda_peak_memory(
            torch,
            cuda_peak_memory,
        )
        if not _cuda_peak_memory_receipt_valid(
            cuda_peak_memory,
            require_captured=True,
        ):
            raise GateConfigError(
                "CUDA peak-memory capture failed at worker completion: "
                f"{cuda_peak_memory}"
            )
        record = {
            "schema_version": SCHEMA_VERSION,
            "worker_state": "completed",
            "status": status,
            "expected_engine": str(payload["engine"]),
            "metadata": result_metadata,
            "has_counterexample": result.counterexample is not None,
            "phase_times": phase_times,
            "worker_elapsed_seconds": time.monotonic() - started,
            "engine_connection": connection_reason,
            "cuda_peak_memory": cuda_peak_memory,
            "query_dual_feedback_config": {
                "family": str(payload["family"]),
                "targets": list(query_targets),
                "status": str(query_status),
                "requested": {
                    "steps": requested_query_steps,
                    "time_limit": requested_query_seconds,
                },
                "effective": {
                    "steps": int(payload["query_dual_feedback_steps"]),
                    "time_limit": float(
                        payload["query_dual_feedback_time_limit"]
                    ),
                },
                "block_size": int(
                    payload["query_dual_feedback_block_size"]
                ),
                "device": str(payload["query_dual_feedback_device"]),
            },
            "input_integrity": {
                "expected": expected_inputs,
                "before": observed_inputs_before,
                "after": observed_inputs_after,
                "unchanged": True,
            },
            "fixed_worker_environment": expected_environment,
            "thread_caps": {
                "row_workers": row_workers,
                "per_solver_threads": solver_threads,
                "total_solver_threads": total_threads,
                "row_x_solver_threads": row_workers * solver_threads,
                "native_threads_per_worker": 1,
                "tight_workers": int(
                    expected_environment["HZ_TIGHT_THREADS"]
                ),
            },
        }
        _worker_payload_result(result_path, record)
        return 0
    except BaseException as exc:
        if torch_module is not None:
            cuda_peak_memory = _capture_cuda_peak_memory(
                torch_module,
                cuda_peak_memory,
            )
        record = {
            "schema_version": SCHEMA_VERSION,
            "worker_state": "error",
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=40),
            },
            "phase_times": phase_times,
            "worker_elapsed_seconds": time.monotonic() - started,
            "cuda_peak_memory": cuda_peak_memory,
        }
        try:
            _worker_payload_result(result_path, record)
        except Exception:
            traceback.print_exc()
        return 2


def _query_dual_worker_payload(
    runtime: Mapping[str, Any],
    *,
    family: str,
    targets: Sequence[int],
    status: str,
) -> dict[str, Any]:
    """Build the requested/effective query settings for one family."""

    normalized_targets = _gate_query_dual_targets(
        targets,
        context=f"worker payload family {family}",
    )
    if status not in {"gate1_candidate", "not_promoted"}:
        raise GateConfigError(
            f"family {family} has invalid query-dual status {status!r}"
        )
    family_enabled = status == "gate1_candidate"
    if status == "not_promoted" and normalized_targets:
        raise GateConfigError(
            f"not_promoted family {family} cannot carry query-dual targets"
        )
    requested_steps = int(runtime["query_dual_feedback_steps"])
    requested_seconds = float(
        runtime["query_dual_feedback_time_limit"]
    )
    return {
        "query_dual_feedback_targets": list(normalized_targets),
        "query_dual_feedback_status": status,
        "requested_query_dual_feedback_steps": requested_steps,
        "requested_query_dual_feedback_time_limit": requested_seconds,
        "query_dual_feedback_steps": (
            requested_steps if family_enabled else 0
        ),
        "query_dual_feedback_time_limit": (
            requested_seconds if family_enabled else 0.0
        ),
        "query_dual_feedback_block_size": int(
            runtime["query_dual_feedback_block_size"]
        ),
        "query_dual_feedback_device": str(
            runtime["query_dual_feedback_device"]
        ),
    }


def _run_child(
    sentinel: Sentinel,
    *,
    runtime: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    run_id: str,
    log_dir: Path,
) -> dict[str, Any]:
    """Launch one isolated worker and enforce its outer wall deadline."""

    timeout = float(runtime["wall_timeout_seconds"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"{run_id}_{sentinel.family}_iid{sentinel.iid:03d}.log"
    )
    memory_before = _mem_available()
    gpu_before = _gpu_memory(int(runtime["gpu_index"]))
    query_dual_payload = _query_dual_worker_payload(
        runtime,
        family=sentinel.family,
        targets=sentinel.query_dual_feedback_targets,
        status=sentinel.query_dual_feedback_status,
    )
    with tempfile.TemporaryDirectory(
        prefix="hybridz-largecls-worker-"
    ) as temp_raw:
        temp_dir = Path(temp_raw)
        payload_path = temp_dir / "payload.json"
        result_path = temp_dir / "result.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "family": sentinel.family,
            "iid": sentinel.iid,
            "onnx_path": str(sentinel.instance.onnx_path),
            "vnnlib_path": str(sentinel.instance.vnnlib_path),
            "wall_timeout_seconds": timeout,
            "gpu_index": int(runtime["gpu_index"]),
            "dtype": runtime["dtype"],
            "engine": runtime["engine"],
            "operator_exact_budget": int(runtime["operator_exact_budget"]),
            "operator_phase_clique_time_limit": float(
                runtime["operator_phase_clique_time_limit"]
            ),
            "operator_materialize_add": bool(
                runtime["operator_materialize_add"]
            ),
            # Family-local targets and effective settings come from one
            # audited derivation; CLI never supplies a global target list.
            **query_dual_payload,
            "preactivation_lp_budget": int(
                runtime["preactivation_lp_budget"]
            ),
            "preactivation_lp_time_limit": float(
                runtime["preactivation_lp_time_limit"]
            ),
            "property_correlation_budget": int(
                runtime.get("property_correlation_budget", 0)
            ),
            "property_correlation_time_limit": float(
                runtime.get("property_correlation_time_limit", 0.0)
            ),
            "residual_phase_screen": bool(
                runtime.get("residual_phase_screen", False)
            ),
            "residual_bound_screen": bool(
                runtime.get("residual_bound_screen", False)
            ),
            "property_residual_budget": int(
                runtime["property_residual_budget"]
            ),
            "property_residual_time_limit": float(
                runtime["property_residual_time_limit"]
            ),
            "property_residual_max_adjoint_cells": int(
                runtime["property_residual_max_adjoint_cells"]
            ),
            "property_residual_pool_per_rival": int(
                runtime["property_residual_pool_per_rival"]
            ),
            "property_tail_upper": bool(runtime["property_tail_upper"]),
            "property_micro_rlt_product_cap": int(
                runtime["property_micro_rlt_product_cap"]
            ),
            "property_micro_rlt_packet_mode": str(
                runtime["property_micro_rlt_packet_mode"]
            ),
            "property_micro_rlt_parent_prefilter_seconds": float(
                runtime["property_micro_rlt_parent_prefilter_seconds"]
            ),
            "property_micro_rlt_parent_only_diagnostic": bool(
                runtime["property_micro_rlt_parent_only_diagnostic"]
            ),
            "property_tail_add_source_planes": bool(
                runtime["property_tail_add_source_planes"]
            ),
            "property_tail_alpha_steps": int(
                runtime["property_tail_alpha_steps"]
            ),
            "property_tail_alpha_time_limit": float(
                runtime["property_tail_alpha_time_limit"]
            ),
            "property_tail_alpha_learning_rate": float(
                runtime["property_tail_alpha_learning_rate"]
            ),
            "property_tail_alpha_max_cells": int(
                runtime["property_tail_alpha_max_cells"]
            ),
            "property_tail_alpha_device": str(
                runtime["property_tail_alpha_device"]
            ),
            "property_tail_mixture_grid_bits": int(
                runtime["property_tail_mixture_grid_bits"]
            ),
            "property_tail_pairhull_budget": int(
                runtime["property_tail_pairhull_budget"]
            ),
            "property_tail_pairhull_time_limit": float(
                runtime["property_tail_pairhull_time_limit"]
            ),
            "property_tail_suffix_blocks": int(
                runtime["property_tail_suffix_blocks"]
            ),
            "property_tail_suffix_alpha_steps": int(
                runtime["property_tail_suffix_alpha_steps"]
            ),
            "property_tail_suffix_alpha_time_limit": float(
                runtime["property_tail_suffix_alpha_time_limit"]
            ),
            "property_tail_suffix_alpha_device": str(
                runtime["property_tail_suffix_alpha_device"]
            ),
            "gpu_dual_steps": int(runtime["gpu_dual_steps"]),
            "gpu_dual_time_limit": float(runtime["gpu_dual_time_limit"]),
            "gpu_dual_row_topk": int(runtime["gpu_dual_row_topk"]),
            "gpu_dual_learning_rate": float(
                runtime["gpu_dual_learning_rate"]
            ),
            "lp_prefilter_fraction": float(runtime["lp_prefilter_fraction"]),
            "lp_prefilter_max_seconds": float(
                runtime["lp_prefilter_max_seconds"]
            ),
            "row_workers": int(runtime["row_workers"]),
            "total_solver_threads": int(runtime["total_solver_threads"]),
            "fixed_worker_environment": _fixed_worker_environment(runtime),
            "expected_input_sha256": {
                "onnx": artifact_hashes[
                    str(sentinel.instance.onnx_path.resolve())
                ],
                "vnnlib": artifact_hashes[
                    str(sentinel.instance.vnnlib_path.resolve())
                ],
            },
        }
        _atomic_json(payload_path, payload)
        # Execute as a module, not by filesystem path.  Direct script
        # execution makes ``sys.path[0]`` the nested verification directory
        # and breaks the worker's absolute ``act.*`` imports even when cwd is
        # the repository root.
        command = [
            sys.executable,
            "-m",
            "act.pipeline.verification.hybridz_largecls_gate",
            "--_worker-payload",
            str(payload_path),
            "--_worker-result",
            str(result_path),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        for key in tuple(env):
            if key.startswith("HZ_"):
                del env[key]
        env.update(_fixed_worker_environment(runtime))
        started = time.monotonic()
        timed_out = False
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        wall_elapsed = time.monotonic() - started

        if timed_out:
            child_result: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "worker_state": "completed",
                "status": "timeout",
                "metadata": {
                    "reason": "outer_total_wall_deadline",
                    "hard_kill": True,
                },
                "phase_times": {},
                "cuda_peak_memory": _cuda_peak_memory_unavailable(
                    "worker_hard_killed_before_receipt"
                ),
            }
        elif not result_path.is_file():
            child_result = {
                "schema_version": SCHEMA_VERSION,
                "worker_state": "error",
                "status": "error",
                "error": {
                    "type": "MissingWorkerReceipt",
                    "message": f"worker exit={process.returncode} produced no result",
                },
                "cuda_peak_memory": _cuda_peak_memory_unavailable(
                    "missing_worker_receipt"
                ),
            }
        else:
            try:
                child_result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                child_result = {
                    "schema_version": SCHEMA_VERSION,
                    "worker_state": "error",
                    "status": "error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": f"invalid worker result: {exc}",
                    },
                    "cuda_peak_memory": _cuda_peak_memory_unavailable(
                        "invalid_worker_receipt"
                    ),
                }
            if process.returncode != 0 and child_result.get("worker_state") == "completed":
                child_result = {
                    **child_result,
                    "worker_state": "error",
                    "error": {
                        "type": "NonzeroWorkerExit",
                        "message": f"worker exit={process.returncode}",
                    },
                }
            if (
                child_result.get("worker_state") == "completed"
                and not _cuda_peak_memory_receipt_valid(
                    child_result.get("cuda_peak_memory"),
                    require_captured=True,
                )
            ):
                child_result = {
                    **child_result,
                    "worker_state": "error",
                    "status": "error",
                    "error": {
                        "type": "InvalidCudaPeakMemoryReceipt",
                        "message": (
                            "completed CUDA worker lacks a valid captured "
                            "peak-memory receipt"
                        ),
                    },
                }

    child_result.setdefault(
        "cuda_peak_memory",
        _cuda_peak_memory_unavailable("worker_receipt_unavailable"),
    )
    child_result.update(
        {
            "parent_wall_seconds": wall_elapsed,
            "outer_timeout_seconds": timeout,
            "outer_timed_out": timed_out,
            "log_path": str(log_path),
            "resources_before": {
                "memory": memory_before,
                "gpu": gpu_before,
            },
            "resources_after": {
                "memory": _mem_available(),
                "gpu": _gpu_memory(int(runtime["gpu_index"])),
            },
        }
    )
    return child_result


def _default_outputs(gate: int) -> tuple[Path, Path]:
    root = REPO_ROOT / "artifacts" / "hybridz_largecls_gates"
    return (
        root / f"gate{gate}.jsonl",
        root / f"gate{gate}.summary.json",
    )


def _runtime_from_args(
    raw: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    query_dual_feedback_targets: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    runtime = dict(raw["runtime"])
    overrides = {
        "wall_timeout_seconds": args.wall_timeout,
        "engine": args.engine,
        "gpu_index": args.gpu_index,
        "dtype": args.dtype,
        "operator_exact_budget": args.operator_exact_budget,
        "operator_phase_clique_time_limit": (
            args.operator_phase_clique_time_limit
        ),
        "operator_materialize_add": args.operator_materialize_add,
        "query_dual_feedback_steps": args.query_dual_feedback_steps,
        "query_dual_feedback_time_limit": (
            args.query_dual_feedback_time_limit
        ),
        "query_dual_feedback_block_size": (
            args.query_dual_feedback_block_size
        ),
        "query_dual_feedback_device": args.query_dual_feedback_device,
        "preactivation_lp_budget": args.preactivation_lp_budget,
        "preactivation_lp_time_limit": args.preactivation_lp_time_limit,
        "property_correlation_budget": getattr(
            args, "property_correlation_budget", None
        ),
        "property_correlation_time_limit": (
            getattr(args, "property_correlation_time_limit", None)
        ),
        "residual_phase_screen": getattr(
            args, "residual_phase_screen", None
        ),
        "residual_bound_screen": getattr(
            args, "residual_bound_screen", None
        ),
        "property_residual_budget": args.property_residual_budget,
        "property_residual_time_limit": args.property_residual_time_limit,
        "property_residual_max_adjoint_cells": (
            args.property_residual_max_adjoint_cells
        ),
        "property_residual_pool_per_rival": (
            args.property_residual_pool_per_rival
        ),
        "property_tail_upper": args.property_tail_upper,
        "property_micro_rlt_product_cap": (
            args.property_micro_rlt_product_cap
        ),
        "property_micro_rlt_packet_mode": (
            args.property_micro_rlt_packet_mode
        ),
        "property_micro_rlt_parent_prefilter_seconds": (
            args.property_micro_rlt_parent_prefilter_seconds
        ),
        "property_micro_rlt_parent_only_diagnostic": (
            args.property_micro_rlt_parent_only_diagnostic
        ),
        "property_tail_add_source_planes": (
            args.property_tail_add_source_planes
        ),
        "property_tail_alpha_steps": args.property_tail_alpha_steps,
        "property_tail_alpha_time_limit": (
            args.property_tail_alpha_time_limit
        ),
        "property_tail_alpha_learning_rate": (
            args.property_tail_alpha_learning_rate
        ),
        "property_tail_alpha_max_cells": (
            args.property_tail_alpha_max_cells
        ),
        "property_tail_alpha_device": args.property_tail_alpha_device,
        "property_tail_mixture_grid_bits": (
            args.property_tail_mixture_grid_bits
        ),
        "property_tail_pairhull_budget": (
            args.property_tail_pairhull_budget
        ),
        "property_tail_pairhull_time_limit": (
            args.property_tail_pairhull_time_limit
        ),
        "property_tail_suffix_blocks": args.property_tail_suffix_blocks,
        "property_tail_suffix_alpha_steps": (
            args.property_tail_suffix_alpha_steps
        ),
        "property_tail_suffix_alpha_time_limit": (
            args.property_tail_suffix_alpha_time_limit
        ),
        "property_tail_suffix_alpha_device": (
            args.property_tail_suffix_alpha_device
        ),
        "gpu_dual_steps": args.gpu_dual_steps,
        "gpu_dual_time_limit": args.gpu_dual_time_limit,
        "gpu_dual_row_topk": args.gpu_dual_row_topk,
        "gpu_dual_learning_rate": args.gpu_dual_learning_rate,
        "lp_prefilter_fraction": args.lp_prefilter_fraction,
        "lp_prefilter_max_seconds": args.lp_prefilter_max_seconds,
        "row_workers": args.row_workers,
        "total_solver_threads": args.total_solver_threads,
    }
    for key, value in overrides.items():
        if value is not None:
            runtime[key] = value
    _validate_runtime(
        runtime,
        query_dual_feedback_targets=query_dual_feedback_targets,
    )
    if float(runtime["operator_phase_clique_time_limit"]) == 0.0:
        runtime["operator_phase_clique_time_limit"] = 0.0
    return runtime


def _print_manifest(
    *,
    gate: int,
    sentinels: Sequence[Sentinel],
    provenance: Mapping[str, Any],
    runtime: Mapping[str, Any],
    engine_connection: tuple[bool, str],
) -> None:
    print(
        f"Gate-{gate} delta={len(sentinels)} cumulative={gate} "
        f"engine={runtime['engine']} connected={engine_connection[0]}"
    )
    print(f"manifest_sha256={provenance['manifest_sha256']}")
    print(f"engine_probe={engine_connection[1]}")
    for sentinel in sentinels:
        print(
            f"{sentinel.family:22s} iid={sentinel.iid:3d} "
            f"ref={sentinel.reference_label} "
            f"query_targets={list(sentinel.query_dual_feedback_targets)} "
            f"query_status={sentinel.query_dual_feedback_status} "
            f"onnx={sentinel.instance.onnx_rel} "
            f"vnnlib={sentinel.instance.vnnlib_rel}"
        )
    print("reference labels are diagnostic only; they are not proof or PASS inputs")


def _run_end_integrity(
    *,
    stages: Mapping[int, Sequence[Sentinel]],
    provenance: Mapping[str, Any],
    source_sha256: str,
    artifact_sha256: str,
    environment_sha256: str,
    runtime: Mapping[str, Any],
    promotion: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild every immutable run input immediately before ``run_end``."""

    observed: dict[str, Any] = {}
    errors: list[str] = []
    try:
        observed_source, _ = _source_fingerprint()
        observed["source_sha256"] = observed_source
    except Exception as exc:
        errors.append(f"source:{type(exc).__name__}:{exc}")
    try:
        observed_artifact, _ = _artifact_fingerprint(stages)
        observed["artifact_sha256"] = observed_artifact
    except Exception as exc:
        errors.append(f"artifact:{type(exc).__name__}:{exc}")
    try:
        observed_environment, _ = _environment_fingerprint(runtime)
        observed["environment_sha256"] = observed_environment
    except Exception as exc:
        errors.append(f"environment:{type(exc).__name__}:{exc}")
    try:
        observed["config_sha256"] = _sha256_file(
            Path(str(provenance["config_path"]))
        )
    except Exception as exc:
        errors.append(f"config:{type(exc).__name__}:{exc}")
    observed_csv: dict[str, str] = {}
    for benchmark in sorted(provenance["csv_sha256"]):
        csv_path = (
            Path(str(provenance["benchmark_root"]))
            / benchmark
            / "instances.csv"
        )
        try:
            observed_csv[benchmark] = _sha256_file(csv_path)
        except Exception as exc:
            errors.append(f"csv:{benchmark}:{type(exc).__name__}:{exc}")
    observed["csv_sha256"] = observed_csv

    expected = {
        "source_sha256": source_sha256,
        "artifact_sha256": artifact_sha256,
        "environment_sha256": environment_sha256,
        "config_sha256": provenance["config_sha256"],
        "csv_sha256": dict(provenance["csv_sha256"]),
    }
    if promotion is not None:
        promotion_path = Path(str(promotion["receipt_path"]))
        expected["promotion_receipt_sha256"] = promotion["receipt_sha256"]
        try:
            observed["promotion_receipt_sha256"] = _sha256_file(
                promotion_path
            )
        except Exception as exc:
            errors.append(f"promotion:{type(exc).__name__}:{exc}")
        expected_chain_receipts = {
            str(item["receipt_path"]): str(item["receipt_sha256"])
            for item in promotion.get("chain", [])
            if isinstance(item, Mapping)
        }
        observed_chain_receipts: dict[str, str] = {}
        for raw_path in sorted(expected_chain_receipts):
            try:
                observed_chain_receipts[raw_path] = _sha256_file(
                    Path(raw_path)
                )
            except Exception as exc:
                errors.append(
                    f"promotion_chain:{raw_path}:{type(exc).__name__}:{exc}"
                )
        expected["promotion_chain_receipts"] = expected_chain_receipts
        observed["promotion_chain_receipts"] = observed_chain_receipts

    checks = {
        key: observed.get(key) == value for key, value in expected.items()
    }
    return {
        "passed": not errors and all(checks.values()),
        "checks": checks,
        "expected": expected,
        "observed": observed,
        "errors": errors,
    }


def run_gate(
    *,
    gate: int,
    sentinels: Sequence[Sentinel],
    stages: Mapping[int, Sequence[Sentinel]],
    all_families: Sequence[str],
    selected_families: Sequence[str],
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
    source_sha256: str,
    source_files: Sequence[Mapping[str, Any]],
    artifact_sha256: str,
    artifact_files: Sequence[Mapping[str, Any]],
    environment_sha256: str,
    environment_snapshot: Mapping[str, Any],
    experiment_sha256: str,
    receipt_path: Path,
    summary_path: Path,
    promotion: Optional[Mapping[str, Any]],
    allow_unpromoted: bool,
) -> int:
    family_snapshot = provenance.get("query_dual_feedback_families")
    if not isinstance(family_snapshot, Mapping):
        raise GateConfigError(
            "run_gate query_dual_feedback_families must be a mapping"
        )
    if any(type(family) is not str for family in family_snapshot):
        raise GateConfigError(
            "run_gate query-dual family names must be strings"
        )
    run_query_targets: list[int] = []
    for family in sorted(family_snapshot):
        spec = family_snapshot[family]
        if not isinstance(spec, Mapping):
            raise GateConfigError(
                "run_gate query-dual family snapshot is malformed"
            )
        raw_targets = spec.get("targets")
        targets = _gate_query_dual_targets(
            raw_targets,
            context=f"run_gate family snapshot {family}",
        )
        if (
            type(raw_targets) is not list
            or list(targets) != raw_targets
            or spec.get("status")
            not in {"gate1_candidate", "not_promoted"}
            or (
                spec.get("status") == "not_promoted"
                and bool(targets)
            )
        ):
            raise GateConfigError(
                f"run_gate family snapshot {family} is noncanonical"
            )
        run_query_targets.extend(targets)
    _validate_runtime(
        runtime,
        query_dual_feedback_targets=run_query_targets,
    )
    receipt = AppendOnlyReceipt(receipt_path)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    parent_only_diagnostic = bool(
        runtime["property_micro_rlt_parent_only_diagnostic"]
    )
    packet_mode = str(runtime["property_micro_rlt_packet_mode"])
    packet_isolation_diagnostic = packet_mode in {"first", "second"}
    phase_clique_diagnostic = bool(
        float(runtime["operator_phase_clique_time_limit"]) > 0.0
    )
    if (parent_only_diagnostic or packet_isolation_diagnostic) and (
        gate != 6
        or len(selected_families) != 1
        or len(sentinels) != 1
    ):
        raise GateConfigError(
            "property micro-RLT parent-only/packet-isolation run contract "
            "requires Gate-6 with exactly one family and one iid"
        )
    if phase_clique_diagnostic and (
        gate != 6
        or len(selected_families) != 1
        or len(sentinels) != 1
    ):
        raise GateConfigError(
            "operator phase-clique trial requires Gate-6 with exactly one "
            "family and one iid"
        )
    unpromoted = bool(gate > 6 and allow_unpromoted and promotion is None)
    partial = (
        sorted(selected_families) != sorted(all_families)
        or len(sentinels) != len(stages[gate])
    )
    diagnostic_only = bool(
        parent_only_diagnostic
        or packet_isolation_diagnostic
        or phase_clique_diagnostic
        or unpromoted
        or partial
    )
    promotion_chain = (
        list(promotion.get("chain", [])) if promotion is not None else []
    )
    query_dual_effective = _query_dual_effective_by_family(
        runtime,
        provenance["query_dual_feedback_families"],
    )
    engine_connected, engine_reason = _engine_connected(
        str(runtime["engine"]),
        operator_phase_clique_time_limit=float(
            runtime["operator_phase_clique_time_limit"]
        ),
    )
    started_wall = time.time()
    run_start = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_start",
        "run_id": run_id,
        "started_at": _utc_now(),
        "gate": gate,
        "delta_count": len(sentinels),
        "cumulative_count": gate,
        "selected_families": list(selected_families),
        "manifest_sha256": provenance["manifest_sha256"],
        "config_sha256": provenance["config_sha256"],
        "source_sha256": source_sha256,
        "source_files": list(source_files),
        "artifact_sha256": artifact_sha256,
        "artifact_files": list(artifact_files),
        "environment_sha256": environment_sha256,
        "environment": dict(environment_snapshot),
        "experiment_sha256": experiment_sha256,
        "runtime": dict(runtime),
        "cuda_peak_memory_policy": _cuda_peak_memory_policy(),
        "query_dual_candidate_policy": _query_dual_candidate_policy(),
        "query_dual_feedback_families": dict(
            provenance["query_dual_feedback_families"]
        ),
        "query_dual_feedback_effective_by_family": query_dual_effective,
        "promotion": promotion,
        "promotion_chain": promotion_chain,
        "unpromoted_diagnostic": unpromoted,
        "partial_family_diagnostic": partial,
        "property_micro_rlt_parent_only_diagnostic": (
            parent_only_diagnostic
        ),
        "property_micro_rlt_packet_mode": packet_mode,
        "operator_phase_clique_diagnostic": phase_clique_diagnostic,
        "diagnostic_only": diagnostic_only,
        "promotion_eligible": not diagnostic_only,
        "ground_truth_loaded": False,
        "reference_diagnostic_label_present": True,
        "reference_labels_are_proof": False,
        "engine_connected": engine_connected,
        "engine_connection_reason": engine_reason,
    }
    receipt.append(run_start)

    if unpromoted:
        print(
            f"{ANSI_RED}UNPROMOTED DIAGNOSTIC ONLY: Gate-{gate} results "
            f"cannot PASS or promote a later gate.{ANSI_RESET}",
            file=sys.stderr,
            flush=True,
        )
    if parent_only_diagnostic:
        print(
            f"{ANSI_RED}MICRO-RLT PARENT-ONLY DIAGNOSTIC: the single-iid "
            f"receipt cannot PASS or promote a later gate.{ANSI_RESET}",
            file=sys.stderr,
            flush=True,
        )
    elif packet_isolation_diagnostic:
        print(
            f"{ANSI_RED}MICRO-RLT PACKET-{packet_mode.upper()} DIAGNOSTIC: "
            f"the single-iid receipt cannot PASS or promote a later "
            f"gate.{ANSI_RESET}",
            file=sys.stderr,
            flush=True,
        )
    if phase_clique_diagnostic:
        print(
            f"{ANSI_RED}OPERATOR K4 PHASE-CLIQUE DIAGNOSTIC: the "
            f"single-iid receipt cannot PASS or promote a later gate."
            f"{ANSI_RESET}",
            file=sys.stderr,
            flush=True,
        )

    results: list[dict[str, Any]] = []
    stopped_families: dict[str, str] = {}
    global_failure_class: Optional[str] = None
    global_failure_reason: Optional[str] = None
    log_dir = receipt.path.parent / "logs"
    artifact_hashes = {
        str(item["path"]): str(item["sha256"]) for item in artifact_files
    }

    if not engine_connected:
        global_failure_class = "BLOCKED_ENGINE"
        global_failure_reason = f"engine_not_connected:{engine_reason}"
    else:
        # Advisory lock prevents two gate runners from propagating on the same
        # physical GPU concurrently.  It is deliberately held for the run.
        lock_path = Path(
            f"/tmp/act_hybridz_largecls_gpu{int(runtime['gpu_index'])}.lock"
        )
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                global_failure_class = "BLOCKED_RESOURCE"
                global_failure_reason = f"gpu_lock_busy:{lock_path}"

            for family in selected_families:
                if global_failure_class is not None:
                    break
                family_rows = [item for item in sentinels if item.family == family]
                inconclusive = 0
                for sentinel in family_rows:
                    if (
                        global_failure_class is not None
                        or family in stopped_families
                    ):
                        break
                    print(
                        f"[Gate-{gate}] {family} iid={sentinel.iid} "
                        f"(ref={sentinel.reference_label}, diagnostic only)",
                        flush=True,
                    )
                    child = _run_child(
                        sentinel,
                        runtime=runtime,
                        artifact_hashes=artifact_hashes,
                        run_id=run_id,
                        log_dir=log_dir,
                    )
                    classification = _classify_result(child)
                    if (
                        parent_only_diagnostic
                        and str(child.get("status", "")).strip().lower()
                        in CONCLUSIVE
                    ):
                        classification = ResultClassification(
                            "FAIL_ERROR",
                            (
                                "parent_only_diagnostic_conclusive_verdict_"
                                "contract_conflict"
                            ),
                            False,
                        )
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "instance_result",
                        "run_id": run_id,
                        "recorded_at": _utc_now(),
                        "gate": gate,
                        "family": family,
                        "benchmark": sentinel.instance.benchmark,
                        "iid": sentinel.iid,
                        "reference_diagnostic_label": sentinel.reference_label,
                        "ground_truth_loaded": False,
                        "reference_diagnostic_label_present": True,
                        "reference_label_used_for_verdict_or_pass": False,
                        "property_micro_rlt_parent_only_diagnostic": (
                            parent_only_diagnostic
                        ),
                        "property_micro_rlt_packet_mode": packet_mode,
                        "operator_phase_clique_diagnostic": (
                            phase_clique_diagnostic
                        ),
                        "diagnostic_only": diagnostic_only,
                        "promotion_eligible": not diagnostic_only,
                        "query_dual_feedback_targets": list(
                            sentinel.query_dual_feedback_targets
                        ),
                        "query_dual_feedback_status": (
                            sentinel.query_dual_feedback_status
                        ),
                        "query_dual_feedback_config": dict(
                            query_dual_effective[family]
                        ),
                        "onnx_path": str(sentinel.instance.onnx_path),
                        "vnnlib_path": str(sentinel.instance.vnnlib_path),
                        "csv_timeout_seconds": sentinel.instance.csv_timeout,
                        "conclusive": classification.conclusive,
                        "failure_class": classification.failure_class,
                        "failure_reason": classification.reason,
                        "fatal_reason": classification.reason,
                        "cuda_peak_memory": child.get(
                            "cuda_peak_memory"
                        ),
                        "result": child,
                    }
                    receipt.append(record)
                    results.append(record)
                    print(
                        f"  -> {child.get('status')} "
                        f"wall={child.get('parent_wall_seconds', 0.0):.3f}s"
                        + (
                            f" {classification.failure_class}="
                            f"{classification.reason}"
                            if classification.failure_class
                            else ""
                        ),
                        flush=True,
                    )
                    if classification.failure_class is not None:
                        global_failure_class = classification.failure_class
                        global_failure_reason = classification.reason
                        stopped_families[family] = str(
                            classification.reason
                        )
                        receipt.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "record_type": "family_stop",
                                "run_id": run_id,
                                "recorded_at": _utc_now(),
                                "gate": gate,
                                "family": family,
                                "reason": classification.reason,
                                "failure_class": (
                                    classification.failure_class
                                ),
                                "global_p0_latched": (
                                    classification.failure_class == "FAIL_P0"
                                ),
                            }
                        )
                        break
                    if not classification.conclusive:
                        inconclusive += 1
                        if inconclusive >= int(
                            runtime["max_inconclusive_per_family"]
                        ):
                            reason = (
                                "STOPLOSS:max_inconclusive_per_family="
                                f"{runtime['max_inconclusive_per_family']}"
                            )
                            stopped_families[family] = reason
                            receipt.append(
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "record_type": "family_stop",
                                    "run_id": run_id,
                                    "recorded_at": _utc_now(),
                                    "gate": gate,
                                    "family": family,
                                    "reason": reason,
                                    "global_p0_latched": False,
                                }
                            )
                            break
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    integrity_end = _run_end_integrity(
        stages=stages,
        provenance=provenance,
        source_sha256=source_sha256,
        artifact_sha256=artifact_sha256,
        environment_sha256=environment_sha256,
        runtime=runtime,
        promotion=promotion,
    )
    if not integrity_end["passed"] and global_failure_class != "FAIL_P0":
        global_failure_class = "FAIL_ERROR"
        global_failure_reason = "run_end_integrity_or_toctou_failure"

    selected_expected = [
        item for item in sentinels if item.family in set(selected_families)
    ]
    completed_keys = {(item["family"], int(item["iid"])) for item in results}
    expected_keys = {(item.family, item.iid) for item in selected_expected}
    all_completed = completed_keys == expected_keys
    all_conclusive = bool(results) and all(
        bool(item["conclusive"]) for item in results
    )
    if global_failure_class is not None:
        status = global_failure_class
    elif parent_only_diagnostic:
        status = "DIAGNOSTIC_PARENT_ONLY"
    elif packet_isolation_diagnostic:
        status = "DIAGNOSTIC_PACKET_MODE"
    elif phase_clique_diagnostic:
        status = "DIAGNOSTIC_OPERATOR_PHASE_CLIQUE"
    elif unpromoted:
        status = "DIAGNOSTIC_UNPROMOTED"
    elif partial:
        status = "DIAGNOSTIC_PARTIAL"
    elif all_completed and all_conclusive and not stopped_families:
        status = "PASS"
    else:
        status = "FAIL_STOPLOSS"

    counts: dict[str, int] = {}
    for item in results:
        result_status = str(item["result"].get("status", "error")).lower()
        counts[result_status] = counts.get(result_status, 0) + 1
    cuda_peak_memory_summary = _summarize_cuda_peak_memory(results)
    run_end = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_end",
        "run_id": run_id,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.time() - started_wall,
        "status": status,
        "gate": gate,
        "delta_count": len(sentinels),
        "cumulative_count": gate,
        "selected_families": list(selected_families),
        "expected_instance_count": len(selected_expected),
        "completed_instance_count": len(results),
        "all_expected_completed": all_completed,
        "all_results_conclusive": all_conclusive,
        "counts": counts,
        "stopped_families": stopped_families,
        "global_failure_class": global_failure_class,
        "global_failure_reason": global_failure_reason,
        "global_fatal": global_failure_reason,
        "manifest_sha256": provenance["manifest_sha256"],
        "config_sha256": provenance["config_sha256"],
        "source_sha256": source_sha256,
        "artifact_sha256": artifact_sha256,
        "environment_sha256": environment_sha256,
        "experiment_sha256": experiment_sha256,
        "engine": runtime["engine"],
        "cuda_peak_memory_policy": _cuda_peak_memory_policy(),
        "cuda_peak_memory_summary": cuda_peak_memory_summary,
        "query_dual_candidate_policy": _query_dual_candidate_policy(),
        "query_dual_feedback_runtime": {
            "steps": int(runtime["query_dual_feedback_steps"]),
            "time_limit": float(
                runtime["query_dual_feedback_time_limit"]
            ),
            "block_size": int(
                runtime["query_dual_feedback_block_size"]
            ),
            "device": str(runtime["query_dual_feedback_device"]),
        },
        "property_micro_rlt_runtime": {
            "product_cap": int(
                runtime["property_micro_rlt_product_cap"]
            ),
            "packet_mode": packet_mode,
            "parent_prefilter_seconds": float(
                runtime["property_micro_rlt_parent_prefilter_seconds"]
            ),
            "parent_only_diagnostic": parent_only_diagnostic,
        },
        "query_dual_feedback_families": dict(
            provenance["query_dual_feedback_families"]
        ),
        "query_dual_feedback_effective_by_family": query_dual_effective,
        "promotion": promotion,
        "promotion_chain": promotion_chain,
        "promotion_chain_sha256": _sha256_bytes(
            _canonical_json(promotion_chain)
        ),
        "run_end_integrity": integrity_end,
        "unpromoted_diagnostic": unpromoted,
        "partial_family_diagnostic": partial,
        "property_micro_rlt_parent_only_diagnostic": (
            parent_only_diagnostic
        ),
        "property_micro_rlt_packet_mode": packet_mode,
        "operator_phase_clique_diagnostic": (
            phase_clique_diagnostic
        ),
        "diagnostic_only": diagnostic_only,
        "promotion_eligible": bool(
            status == "PASS" and not diagnostic_only
        ),
        "ground_truth_loaded": False,
        "reference_diagnostic_label_present": True,
        "reference_labels_are_proof": False,
        "receipt_jsonl": str(receipt.path),
    }
    receipt.append(run_end)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "summary",
        "written_at": _utc_now(),
        "run_start": run_start,
        "run_end": run_end,
        "instance_results": results,
    }
    _atomic_json(summary_path.expanduser().resolve(), summary)
    print(
        f"Gate-{gate} {status}: completed={len(results)}/"
        f"{len(selected_expected)} summary={summary_path}",
        flush=True,
    )
    return 0 if status == "PASS" else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict incremental HybridZ Gate-6/14/40 runner. "
            "Default is Gate-6; no full benchmark mode exists."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gate", type=int, choices=GATES, default=6)
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help=(
            "Run only this family (repeatable). Partial runs are diagnostic "
            "and cannot produce a promotable PASS."
        ),
    )
    parser.add_argument(
        "--iid",
        type=int,
        help=(
            "Run exactly one iid from exactly one --family. This is a "
            "non-promotable Gate-1 diagnostic and is the preferred way to "
            "test a new candidate before Gate-6."
        ),
    )
    parser.add_argument("--promotion-receipt", type=Path)
    parser.add_argument(
        "--allow-unpromoted",
        action="store_true",
        help=(
            "Development diagnostics only: bypass a Gate-14/40 promotion "
            "receipt. Output is marked red and can never PASS."
        ),
    )
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--wall-timeout", type=float)
    parser.add_argument("--engine")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--operator-exact-budget", type=int)
    parser.add_argument(
        "--operator-phase-clique-time-limit",
        type=float,
        help=(
            "Complete raw-property/K4/materialization budget in seconds; "
            "zero disables the candidate."
        ),
    )
    parser.add_argument(
        "--operator-materialize-add",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Materialize every ADD in a fresh constrained frame. "
            "--no-operator-materialize-add is diagnostic until promoted."
        ),
    )
    parser.add_argument("--query-dual-feedback-steps", type=int)
    parser.add_argument("--query-dual-feedback-time-limit", type=float)
    parser.add_argument("--query-dual-feedback-block-size", type=int)
    parser.add_argument(
        "--query-dual-feedback-device",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--preactivation-lp-budget", type=int)
    parser.add_argument("--preactivation-lp-time-limit", type=float)
    parser.add_argument("--property-correlation-budget", type=int)
    parser.add_argument("--property-correlation-time-limit", type=float)
    parser.add_argument(
        "--residual-phase-screen",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--residual-bound-screen",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--property-residual-budget", type=int)
    parser.add_argument("--property-residual-time-limit", type=float)
    parser.add_argument("--property-residual-max-adjoint-cells", type=int)
    parser.add_argument("--property-residual-pool-per-rival", type=int)
    parser.add_argument(
        "--property-tail-upper",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--property-micro-rlt-product-cap", type=int)
    parser.add_argument(
        "--property-micro-rlt-packet-mode",
        choices=("both", "first", "second"),
    )
    parser.add_argument(
        "--property-micro-rlt-parent-prefilter-seconds",
        type=float,
    )
    parser.add_argument(
        "--property-micro-rlt-parent-only-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stop the bounded micro-RLT experiment after its parent "
            "prefilter. Gate-6, exactly one --family, and an explicit --iid "
            "are required; the receipt is diagnostic-only and can never "
            "PASS or promote."
        ),
    )
    parser.add_argument(
        "--property-tail-add-source-planes",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--property-tail-alpha-steps", type=int)
    parser.add_argument("--property-tail-alpha-time-limit", type=float)
    parser.add_argument("--property-tail-alpha-learning-rate", type=float)
    parser.add_argument("--property-tail-alpha-max-cells", type=int)
    parser.add_argument(
        "--property-tail-alpha-device",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--property-tail-mixture-grid-bits", type=int)
    parser.add_argument("--property-tail-pairhull-budget", type=int)
    parser.add_argument("--property-tail-pairhull-time-limit", type=float)
    parser.add_argument("--property-tail-suffix-blocks", type=int)
    parser.add_argument("--property-tail-suffix-alpha-steps", type=int)
    parser.add_argument(
        "--property-tail-suffix-alpha-time-limit", type=float
    )
    parser.add_argument(
        "--property-tail-suffix-alpha-device",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--gpu-dual-steps", type=int)
    parser.add_argument("--gpu-dual-time-limit", type=float)
    parser.add_argument("--gpu-dual-row-topk", type=int)
    parser.add_argument("--gpu-dual-learning-rate", type=float)
    parser.add_argument("--lp-prefilter-fraction", type=float)
    parser.add_argument("--lp-prefilter-max-seconds", type=float)
    parser.add_argument("--row-workers", type=int)
    parser.add_argument("--total-solver-threads", type=int)
    parser.add_argument("--_worker-payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args._worker_payload is not None or args._worker_result is not None:
        if args._worker_payload is None or args._worker_result is None:
            parser.error("both hidden worker paths are required")
        return _worker_main(args._worker_payload, args._worker_result)

    try:
        raw, stages, provenance = load_manifest(args.config)
        family_order = list(raw["families"])
        all_query_targets = [
            target
            for family in family_order
            for target in raw["families"][family][
                "query_dual_feedback_targets"
            ]
        ]
        runtime = _runtime_from_args(
            raw,
            args,
            query_dual_feedback_targets=all_query_targets,
        )
        if args.families:
            unknown = sorted(set(args.families) - set(family_order))
            if unknown:
                raise GateConfigError(f"unknown --family values: {unknown}")
            if len(set(args.families)) != len(args.families):
                raise GateConfigError("duplicate --family values are forbidden")
            selected_families = [
                family for family in family_order if family in set(args.families)
            ]
        else:
            selected_families = family_order
        # Global runtime knobs express the requested experiment.  Family
        # promotion status below derives effective settings per sentinel, so
        # empty not-promoted schedules remain an explicit baseline.
        _validate_runtime(
            runtime,
            query_dual_feedback_targets=all_query_targets,
        )
        _validate_property_micro_rlt_parent_only_selection(
            enabled=bool(
                runtime["property_micro_rlt_parent_only_diagnostic"]
            ),
            gate=args.gate,
            selected_families=selected_families,
            iid=args.iid,
            packet_mode=str(runtime["property_micro_rlt_packet_mode"]),
        )
        _validate_operator_phase_clique_selection(
            time_limit=float(
                runtime["operator_phase_clique_time_limit"]
            ),
            gate=args.gate,
            selected_families=selected_families,
            iid=args.iid,
        )
        sentinels = [
            item for item in stages[args.gate]
            if item.family in set(selected_families)
        ]
        if args.iid is not None:
            if not args.families or len(selected_families) != 1:
                raise GateConfigError(
                    "--iid requires exactly one distinct --family"
                )
            sentinels = [
                item for item in sentinels if item.iid == int(args.iid)
            ]
        if not sentinels:
            raise GateConfigError(
                "selected gate/families/iid contain no sentinels"
            )

        source_sha256, source_files = _source_fingerprint()
        artifact_sha256, artifact_files = _artifact_fingerprint(stages)
        environment_sha256, environment_snapshot = (
            _environment_fingerprint(runtime)
        )
        experiment_sha256 = _experiment_fingerprint(
            provenance=provenance,
            source_sha256=source_sha256,
            artifact_sha256=artifact_sha256,
            environment_sha256=environment_sha256,
            engine=str(runtime["engine"]),
            runtime=runtime,
            query_dual_feedback_families=provenance[
                "query_dual_feedback_families"
            ],
        )
        engine_connection = _engine_connected(
            str(runtime["engine"]),
            operator_phase_clique_time_limit=float(
                runtime["operator_phase_clique_time_limit"]
            ),
        )
        _print_manifest(
            gate=args.gate,
            sentinels=sentinels,
            provenance=provenance,
            runtime=runtime,
            engine_connection=engine_connection,
        )
        if args.list_only:
            if args.gate > 6 and args.promotion_receipt is None:
                print(
                    f"Gate-{args.gate} LIST ONLY: an exact Gate-"
                    f"{14 if args.gate == 40 else 6} PASS receipt will be "
                    "required for dry-run or execution."
                )
            return 0

        promotion: Optional[dict[str, Any]] = None
        if args.gate == 6:
            if args.promotion_receipt is not None or args.allow_unpromoted:
                raise GateConfigError(
                    "Gate-6 neither accepts a promotion receipt nor "
                    "--allow-unpromoted"
                )
        elif args.promotion_receipt is not None:
            promotion = validate_promotion(
                args.promotion_receipt,
                gate=args.gate,
                provenance=provenance,
                experiment_sha256=experiment_sha256,
                source_sha256=source_sha256,
                artifact_sha256=artifact_sha256,
                environment_sha256=environment_sha256,
                expected_families=family_order,
            )
        elif not args.allow_unpromoted:
            raise GateConfigError(
                f"Gate-{args.gate} requires --promotion-receipt pointing to "
                f"an exact Gate-{14 if args.gate == 40 else 6} PASS; "
                "--allow-unpromoted is diagnostic-only"
            )

        if args.dry_run:
            if args.allow_unpromoted and promotion is None:
                print(
                    f"{ANSI_RED}DRY RUN IS UNPROMOTED DIAGNOSTIC ONLY"
                    f"{ANSI_RESET}",
                    file=sys.stderr,
                )
            print(
                f"DRY RUN PASS: manifest/CSV/files/families/promotion validated; "
                f"no verifier child was launched. "
                f"experiment_sha256={experiment_sha256} "
                f"artifact_sha256={artifact_sha256} "
                f"environment_sha256={environment_sha256}"
            )
            return 0

        default_receipt, default_summary = _default_outputs(args.gate)
        receipt_path = args.receipt_jsonl or default_receipt
        summary_path = args.summary_json or default_summary
        if receipt_path.expanduser().resolve() == summary_path.expanduser().resolve():
            raise GateConfigError(
                "--receipt-jsonl and --summary-json must be different files"
            )
        if promotion is not None:
            promotion_path = Path(str(promotion["receipt_path"])).resolve()
            if promotion_path in {
                receipt_path.expanduser().resolve(),
                summary_path.expanduser().resolve(),
            }:
                raise GateConfigError(
                    "gate outputs must not overwrite the promotion receipt"
                )
        return run_gate(
            gate=args.gate,
            sentinels=sentinels,
            stages=stages,
            all_families=family_order,
            selected_families=selected_families,
            runtime=runtime,
            provenance=provenance,
            source_sha256=source_sha256,
            source_files=source_files,
            artifact_sha256=artifact_sha256,
            artifact_files=artifact_files,
            environment_sha256=environment_sha256,
            environment_snapshot=environment_snapshot,
            experiment_sha256=experiment_sha256,
            receipt_path=receipt_path,
            summary_path=summary_path,
            promotion=promotion,
            allow_unpromoted=args.allow_unpromoted,
        )
    except GateConfigError as exc:
        print(f"gate configuration rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
