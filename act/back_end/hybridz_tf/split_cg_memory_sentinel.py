#!/usr/bin/env python3
"""Fail-closed synthetic RSS sentinel for the frozen split-CG candidate.

The sentinel is deliberately diagnostic-only.  Its public runner starts a
fresh Python process, constructs a deterministic canonical split CSR frame,
lets the source frame become resident, and only then invokes
``split_constraint_generation_candidate``.  The candidate is run for one
round with explicit, non-empty seed rows and a complete split-frame scan.

This file is isolated from verifier and benchmark code.  In particular, the
large CIFAR-100 topology is never executed unless the caller supplies the
explicit ``execute_large_profile`` consent bit and every resource preflight
passes.  A checksum detects accidental/tampered diagnostic JSON; it is not a
proof signature.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import ctypes
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import math
import numbers
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp

# The repository package currently emits environment notices on import.  A
# worker's stdout is a strict one-document JSON protocol, so contain those
# unrelated notices locally without changing package initialization.
with contextlib.redirect_stdout(io.StringIO()):
    try:
        from act.back_end.hybridz_tf import (
            split_constraint_generation_candidate as _scg,
        )
    except ModuleNotFoundError:  # Direct execution from an uninstalled checkout.
        _REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
        if str(_REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPOSITORY_ROOT))
        from act.back_end.hybridz_tf import (  # type: ignore[no-redef]
            split_constraint_generation_candidate as _scg,
        )


SCHEMA = "act.hybridz.split_cg_memory_sentinel.v1"
PROFILE_NAME = "cifar100_medium_iid2_v1"
HARD_ABSOLUTE_RSS_CAP_BYTES = 5 * (1 << 29)  # Exactly 2.5 GiB.
MINIMUM_RSS_RESERVE_BYTES = 64 * (1 << 20)
DEFAULT_SELECTED_UPPER_ROWS = 8192
DEFAULT_SCAN_CHUNK_ROWS = 8192
MINIMUM_LARGE_DEADLINE_SECONDS = 120.0
LARGE_NNZ_THRESHOLD = 1_000_000
_INT32_MAX = int(np.iinfo(np.int32).max)
_MAX_SELECTED_UPPER_ROWS = 1_000_000
_MAX_EQUALITY_ROWS = 1_000_000
_MAX_BINARY_CHANGE_COEFFICIENTS = 2_000_000
_POLL_SECONDS = 0.005
_HASH_CHUNK_BYTES = 1 << 20


_PROFILE_DATA = {
    "name": PROFILE_NAME,
    "parent": {
        "n_cont": 52657,
        "n_bin": 4,
        "n_upper": 98974,
        "n_eq": 0,
        "total_constraint_nnz": 10498232,
    },
    "fresh": {
        "n_cont": 52661,
        "n_bin": 4,
        "n_upper": 98975,
        "n_eq": 3,
    },
    # The only measured nnz supplied for this profile is the parent count.
    # It is intentionally retained in the synthetic fresh-shape frame rather
    # than inventing an unmeasured fresh count.
    "synthetic": {
        "n_cont": 52661,
        "n_bin": 4,
        "n_upper": 98975,
        "n_eq": 3,
        "constraint_nnz": 10498232,
    },
    "execution_policy": {
        "first_selected_upper_rows": 8192,
        "separate_followup_selected_upper_rows": 24576,
        "automatic_retry_or_scale_up": False,
    },
}
CIFAR100_MEDIUM_IID2_V1 = MappingProxyType(copy.deepcopy(_PROFILE_DATA))


_LIMITATIONS = (
    "synthetic_canonical_csr_matches_dimensions_and_total_nnz_not_the_real_"
    "coefficient_or_row_degree_distribution",
    "fresh_profile_reuses_the_supplied_parent_total_nnz_because_no_measured_"
    "fresh_total_was_provided",
    "rss_includes_python_scipy_highs_and_allocator_state_not_only_csr_or_"
    "highs_model_bytes",
    "process_peak_rss_is_a_lifetime_high_water_mark_and_can_predate_the_"
    "post_source_baseline",
    "cg_sampled_peak_rss_can_miss_short_lived_spikes_between_samples",
    "the_parent_rss_poll_is_diagnostic_not_a_kernel_hard_limit_and_the_child_"
    "can_overshoot_between_5ms_samples",
    "bounded_v2_stoploss_uses_leaf_aggregate_memory_peak_which_includes_the_"
    "sentinel_parent_and_worker_for_the_service_lifetime",
    "memory_peak_is_not_reset_by_the_sentinel_because_reset_may_be_"
    "unavailable_or_require_authority_not_granted_to_this_process",
    "bounded_v2_mode_leaves_generic_rss_baseline_current_peak_and_delta_"
    "unset_to_prevent_mixing_child_process_rss_with_cgroup_aggregate_memory",
    "malloc_trim_is_libc_and_allocator_dependent_and_does_not_prove_that_all_"
    "native_memory_was_returned",
    "diagnostic_sha256_detects_payload_changes_but_is_not_authentication_or_"
    "proof_authority",
    "this_sentinel_is_not_a_real_benchmark_and_does_not_calibrate_runtime_or_"
    "memory_for_an_actual_network_instance",
)


@dataclass(frozen=True)
class _SyntheticFrame:
    Auc: sp.csr_matrix
    Aub: sp.csr_matrix
    Ac: sp.csr_matrix
    Ab: sp.csr_matrix
    ub: np.ndarray
    b: np.ndarray
    q: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    source_csr_payload_bytes: int
    source_dense_payload_bytes: int
    source_frame_sha256: str
    selected_upper_nnz: int
    equality_nnz: int
    selected_binary_nnz: int
    equality_binary_nnz: int


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _seal_diagnostic(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(dict(diagnostic))
    sealed.pop("diagnostic_sha256", None)
    sealed["diagnostic_sha256"] = _canonical_json_sha256(sealed)
    return sealed


def _safe_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    """Make untrusted API arguments JSON-safe for rejection diagnostics."""

    def safe(value: Any) -> Any:
        if value is None or isinstance(value, (bool, str)):
            return value[:256] if isinstance(value, str) else value
        if isinstance(value, numbers.Integral):
            integer = int(value)
            if -(1 << 63) <= integer <= (1 << 63) - 1:
                return integer
        if isinstance(value, numbers.Real):
            real = float(value)
            if math.isfinite(real):
                return real
        try:
            representation = repr(value)[:256]
        except BaseException:
            representation = "<repr_failed>"
        value_type = type(value)
        return {
            "rejected_type": (
                f"{value_type.__module__}.{value_type.__qualname__}"
            )[:256],
            "repr": representation,
        }

    return {str(key)[:128]: safe(value) for key, value in config.items()}


def _safe_exception_text(exc: BaseException) -> str:
    try:
        detail = str(exc)[:512]
    except BaseException:
        detail = "exception_text_unavailable"
    return f"{type(exc).__name__}:{detail}"


def verify_diagnostic_checksum(diagnostic: Mapping[str, Any]) -> bool:
    """Return whether a diagnostic has an intact canonical JSON checksum."""

    if not isinstance(diagnostic, Mapping):
        return False
    expected = diagnostic.get("diagnostic_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    payload = copy.deepcopy(dict(diagnostic))
    payload.pop("diagnostic_sha256", None)
    try:
        actual = _canonical_json_sha256(payload)
    except (TypeError, ValueError, OverflowError):
        return False
    return hmac.compare_digest(expected, actual)


def get_fixed_profile(
    name: str = PROFILE_NAME,
    *,
    selected_upper_rows: int = DEFAULT_SELECTED_UPPER_ROWS,
) -> dict[str, Any]:
    """Return an independent public copy of the fixed synthetic profile."""

    if name != PROFILE_NAME:
        raise ValueError("unknown_split_cg_memory_profile")
    result = copy.deepcopy(_PROFILE_DATA)
    result["synthetic"]["selected_upper_rows"] = int(selected_upper_rows)
    return result


def _strict_int(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Integral
    ):
        raise ValueError(f"{name}_must_be_an_integer")
    result = int(value)
    if result < lower or result > upper:
        raise ValueError(f"{name}_outside_supported_range")
    return result


def _row_prefix_nnz(
    *, total_nnz: int, total_rows: int, start: int, count: int
) -> int:
    if count <= 0:
        return 0
    base, extra = divmod(total_nnz, total_rows)
    end = start + count
    extra_in_interval = max(0, min(end, extra) - min(start, extra))
    return count * base + extra_in_interval


def estimate_topology_resources(
    *,
    n_cont: int,
    n_bin: int,
    n_upper: int,
    n_eq: int,
    constraint_nnz: int,
    selected_upper_rows: int,
    scan_chunk_rows: int = DEFAULT_SCAN_CHUNK_ROWS,
) -> dict[str, int]:
    """Return deterministic byte/row/nnz estimates without allocating CSR."""

    n_cont = _strict_int(n_cont, "n_cont", 0, _INT32_MAX)
    n_bin = _strict_int(n_bin, "n_bin", 0, _INT32_MAX)
    n_upper = _strict_int(n_upper, "n_upper", 1, _INT32_MAX)
    n_eq = _strict_int(n_eq, "n_eq", 0, _MAX_EQUALITY_ROWS)
    selected = _strict_int(
        selected_upper_rows,
        "selected_upper_rows",
        1,
        min(n_upper, _MAX_SELECTED_UPPER_ROWS),
    )
    scan_chunk_rows = _strict_int(
        scan_chunk_rows, "scan_chunk_rows", 1, 65536
    )
    n_columns = n_cont + n_bin
    if n_columns <= 0 or n_columns > _INT32_MAX:
        raise ValueError("combined_column_count_outside_supported_range")
    total_rows = n_upper + n_eq
    if total_rows <= 0 or total_rows > _INT32_MAX:
        raise ValueError("combined_row_count_outside_supported_range")
    constraint_nnz = _strict_int(
        constraint_nnz, "constraint_nnz", total_rows, _INT32_MAX
    )
    if constraint_nnz > total_rows * n_columns:
        raise ValueError("constraint_nnz_exceeds_csr_capacity")

    selected_upper_nnz = _row_prefix_nnz(
        total_nnz=constraint_nnz,
        total_rows=total_rows,
        start=0,
        count=selected,
    )
    equality_nnz = _row_prefix_nnz(
        total_nnz=constraint_nnz,
        total_rows=total_rows,
        start=n_upper,
        count=n_eq,
    )
    source_csr_payload = (
        constraint_nnz * (8 + 4)
        + 2 * (n_upper + 1) * 4
        + 2 * (n_eq + 1) * 4
    )
    source_dense_payload = (
        n_upper + n_eq + 3 * n_columns
    ) * np.dtype(np.float64).itemsize
    selected_model_nnz = selected_upper_nnz + equality_nnz
    output_bytes = (n_upper + n_eq + n_columns) * 8
    # This is an admission estimate, not a claimed allocator model.  The
    # deliberately generous fixed overhead covers imports and native objects.
    estimated_candidate_increment = (
        selected_model_nnz * 40
        + (selected + n_eq) * 160
        + n_columns * 72
        + output_bytes
        + scan_chunk_rows * 64
        + 128 * (1 << 20)
    )
    return {
        "n_columns": n_columns,
        "total_rows": total_rows,
        "source_csr_payload_bytes": int(source_csr_payload),
        "source_dense_payload_bytes": int(source_dense_payload),
        "selected_upper_nnz": int(selected_upper_nnz),
        "equality_nnz": int(equality_nnz),
        "selected_model_nnz": int(selected_model_nnz),
        "selected_model_rows": int(selected + n_eq),
        "full_scan_rows": int(total_rows),
        "zero_padded_candidate_output_bytes": int(output_bytes),
        "estimated_candidate_increment_bytes": int(
            estimated_candidate_increment
        ),
    }


def _read_status_bytes(path: str, key: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return int(fields[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _current_rss_bytes(pid: Optional[int] = None) -> Optional[int]:
    target = "self" if pid is None else str(int(pid))
    status = _read_status_bytes(f"/proc/{target}/status", "VmRSS")
    if status is not None:
        return status
    try:
        with open(f"/proc/{target}/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


def _peak_rss_bytes() -> Optional[int]:
    status = _read_status_bytes("/proc/self/status", "VmHWM")
    if status is not None:
        return status
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ValueError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes.  The sentinel is Linux-focused,
    # but keep the fallback explicit rather than silently scaling incorrectly.
    return value if sys.platform == "darwin" else value * 1024


def _read_mem_available_bytes() -> Optional[int]:
    return _read_status_bytes("/proc/meminfo", "MemAvailable")


def _read_cgroup_memory() -> dict[str, Any]:
    empty = {
        "detected": False,
        "version": None,
        "leaf": None,
        "current_bytes": None,
        "max_bytes": None,
        "unlimited": False,
        "headroom_bytes": None,
        "ancestor_limits": [],
        "delegation_boundary": None,
        "boundary_complete": False,
        "error": None,
    }
    try:
        membership_lines = Path("/proc/self/cgroup").read_text(
            encoding="ascii"
        ).splitlines()
    except OSError as exc:
        return {
            **empty,
            "error": f"membership_read_failed:{type(exc).__name__}",
        }

    v2_relative: Optional[str] = None
    v1_relative: Optional[str] = None
    for line in membership_lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, relative = fields
        if hierarchy == "0" and controllers == "":
            v2_relative = relative
        elif "memory" in controllers.split(","):
            v1_relative = relative

    if v2_relative is not None:
        root = Path("/sys/fs/cgroup").resolve()
        leaf = (root / v2_relative.lstrip("/")).resolve()
        try:
            leaf.relative_to(root)
        except ValueError:
            return {
                **empty,
                "version": 2,
                "error": "cgroup_leaf_escapes_mount",
            }
        levels: list[dict[str, Any]] = []
        cursor = leaf
        boundary: Optional[str] = None
        error: Optional[str] = None
        while True:
            maximum_path = cursor / "memory.max"
            current_path = cursor / "memory.current"
            has_max = maximum_path.exists()
            has_current = current_path.exists()
            if not has_max and not has_current:
                boundary = str(cursor)
                if cursor != root:
                    error = f"interior_memory_controller_gap:{cursor}"
                break
            if has_max != has_current:
                error = f"partial_memory_controller_files:{cursor}"
                break
            try:
                raw_max = maximum_path.read_text(encoding="ascii").strip()
                current = int(
                    current_path.read_text(encoding="ascii").strip()
                )
                maximum = None if raw_max == "max" else int(raw_max)
            except (OSError, ValueError) as exc:
                error = (
                    "memory_controller_read_failed:"
                    f"{type(exc).__name__}"
                )
                break
            levels.append(
                {
                    "path": str(cursor),
                    "current_bytes": current,
                    "max_bytes": maximum,
                    "unlimited": maximum is None,
                    "headroom_bytes": (
                        None
                        if maximum is None
                        else max(0, maximum - current)
                    ),
                }
            )
            if cursor == root:
                break
            parent = cursor.parent
            if parent == cursor:
                error = "cgroup_ancestor_walk_escaped_mount"
                break
            cursor = parent
        finite_limits = [
            int(level["max_bytes"])
            for level in levels
            if level["max_bytes"] is not None
        ]
        finite_headrooms = [
            int(level["headroom_bytes"])
            for level in levels
            if level["headroom_bytes"] is not None
        ]
        return {
            "detected": bool(levels),
            "version": 2,
            "leaf": str(leaf),
            "current_bytes": (
                levels[0]["current_bytes"] if levels else None
            ),
            "max_bytes": min(finite_limits) if finite_limits else None,
            "unlimited": bool(levels) and not finite_limits,
            "headroom_bytes": (
                min(finite_headrooms) if finite_headrooms else None
            ),
            "ancestor_limits": levels,
            "delegation_boundary": boundary,
            "boundary_complete": (
                error is None and bool(levels)
            ),
            "error": error,
        }

    if v1_relative is not None:
        # The conventional v1 mount is accepted only when the controller pair
        # is complete.  A partial pair is an explicit fail-closed condition.
        leaf = (
            Path("/sys/fs/cgroup/memory") / v1_relative.lstrip("/")
        ).resolve()
        maximum_path = leaf / "memory.limit_in_bytes"
        current_path = leaf / "memory.usage_in_bytes"
        has_max = maximum_path.exists()
        has_current = current_path.exists()
        if has_max != has_current:
            return {
                **empty,
                "version": 1,
                "leaf": str(leaf),
                "error": "partial_memory_controller_files",
            }
        if has_max and has_current:
            try:
                maximum = int(
                    maximum_path.read_text(encoding="ascii").strip()
                )
                current = int(
                    current_path.read_text(encoding="ascii").strip()
                )
            except (OSError, ValueError) as exc:
                return {
                    **empty,
                    "version": 1,
                    "leaf": str(leaf),
                    "error": (
                        "memory_controller_read_failed:"
                        f"{type(exc).__name__}"
                    ),
                }
            unlimited = maximum >= (1 << 60)
            level = {
                "path": str(leaf),
                "current_bytes": current,
                "max_bytes": None if unlimited else maximum,
                "unlimited": unlimited,
                "headroom_bytes": (
                    None if unlimited else max(0, maximum - current)
                ),
            }
            return {
                **empty,
                "detected": True,
                "version": 1,
                "leaf": str(leaf),
                "current_bytes": current,
                "max_bytes": level["max_bytes"],
                "unlimited": unlimited,
                "headroom_bytes": level["headroom_bytes"],
                "ancestor_limits": [level],
                "boundary_complete": True,
            }
        return {
            **empty,
            "version": 1,
            "leaf": str(leaf),
            "delegation_boundary": str(leaf),
        }
    return {**empty, "error": "memory_cgroup_membership_not_found"}


def _read_v2_leaf_aggregate_memory(leaf: Any) -> dict[str, Any]:
    """Read aggregate current/peak for one already-bound cgroup v2 leaf."""

    sampled = time.monotonic()
    result = {
        "readable": False,
        "leaf": str(leaf) if leaf is not None else None,
        "current_bytes": None,
        "peak_bytes": None,
        "sampled_monotonic_hex": sampled.hex(),
        "error": None,
    }
    if not isinstance(leaf, (str, os.PathLike)):
        result["error"] = "cgroup_leaf_path_unavailable"
        return result
    root = Path("/sys/fs/cgroup").resolve()
    candidate = Path(leaf).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        result["error"] = "cgroup_leaf_escapes_trusted_mount"
        return result
    current_path = candidate / "memory.current"
    peak_path = candidate / "memory.peak"
    has_current = current_path.exists()
    has_peak = peak_path.exists()
    if not has_current or not has_peak:
        missing = []
        if not has_current:
            missing.append("memory.current")
        if not has_peak:
            missing.append("memory.peak")
        result["error"] = "aggregate_metric_missing:" + ",".join(missing)
        return result
    try:
        current = int(current_path.read_text(encoding="ascii").strip())
        peak = int(peak_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        result["error"] = (
            "aggregate_metric_read_failed:" f"{type(exc).__name__}"
        )
        return result
    if current < 0 or peak < 0 or peak < current:
        result["error"] = "aggregate_metric_value_invalid"
        return result
    result.update(readable=True, current_bytes=current, peak_bytes=peak)
    return result


def _base_diagnostic(
    config: Mapping[str, Any], estimates: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    estimates = dict(estimates or {})
    return {
        "schema": SCHEMA,
        "status": "not_started",
        "reason": None,
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "primal_feasibility_authority": False,
        "parent_binding": False,
        "parent_binding_authority": False,
        "profile_name": config.get("profile_name"),
        "profile": (
            copy.deepcopy(_PROFILE_DATA)
            if config.get("profile_name") == PROFILE_NAME
            else None
        ),
        "topology": {
            "n_cont": config.get("n_cont"),
            "n_bin": config.get("n_bin"),
            "n_upper": config.get("n_upper"),
            "n_eq": config.get("n_eq"),
            "constraint_nnz": config.get("constraint_nnz"),
            "selected_upper_rows": config.get("selected_upper_rows"),
        },
        "absolute_deadline_monotonic_hex": (
            float(config["deadline"]).hex()
            if isinstance(config.get("deadline"), (int, float))
            and math.isfinite(float(config["deadline"]))
            else None
        ),
        "absolute_rss_cap_bytes": config.get("absolute_rss_cap_bytes"),
        "hard_absolute_rss_cap_bytes": HARD_ABSOLUTE_RSS_CAP_BYTES,
        "rss_reserve_bytes": config.get("rss_reserve_bytes"),
        "effective_rss_limit_bytes": None,
        "allowed_rss_increment_bytes": None,
        "baseline_current_rss_bytes": None,
        "current_rss_bytes": None,
        "post_trim_current_rss_bytes": None,
        "process_peak_rss_bytes": None,
        "peak_rss_bytes": None,
        "current_delta_from_baseline_bytes": None,
        "post_trim_delta_from_baseline_bytes": None,
        "peak_delta_from_baseline_bytes": None,
        "generic_rss_fields_scope": None,
        "child_process_allowed_rss_increment_bytes": None,
        "child_process_baseline_current_rss_bytes": None,
        "child_process_current_after_cg_bytes": None,
        "child_process_current_after_trim_bytes": None,
        "child_process_peak_rss_bytes": None,
        "child_process_current_delta_from_baseline_bytes": None,
        "child_process_post_trim_delta_from_baseline_bytes": None,
        "child_process_peak_delta_bytes": None,
        "cg_sampled_peak_rss_bytes": None,
        "cg_sampled_peak_delta_from_baseline_bytes": None,
        "rss_cap_respected": False,
        "kernel_hard_limit_enforced": False,
        "rss_cap_enforcement": "parent_polling_diagnostic_only",
        "rss_monitor_sample_seconds": _POLL_SECONDS,
        "cgroup_leaf_path": None,
        "cgroup_leaf_memory_current_bytes": None,
        "cgroup_leaf_memory_max_bytes": None,
        "cgroup_ancestor_min_memory_max_bytes": None,
        "cgroup_aggregate_memory_current_start_bytes": None,
        "cgroup_aggregate_memory_current_after_frame_bytes": None,
        "cgroup_aggregate_memory_current_after_cg_bytes": None,
        "cgroup_aggregate_memory_current_terminal_bytes": None,
        "cgroup_aggregate_memory_peak_terminal_bytes": None,
        "cgroup_aggregate_metrics_readable": False,
        "cgroup_aggregate_peak_reset_attempted": False,
        "cgroup_aggregate_peak_scope": (
            "v2_leaf_service_lifetime_including_parent_and_worker"
        ),
        "cgroup_allowed_increment_after_frame_bytes": None,
        "cgroup_aggregate_peak_delta_from_after_frame_bytes": None,
        "stoploss_peak_bytes": None,
        "stoploss_peak_source": None,
        "worker_cgroup_leaf_path": None,
        "source_csr_payload_bytes": estimates.get(
            "source_csr_payload_bytes"
        ),
        "source_dense_payload_bytes": estimates.get(
            "source_dense_payload_bytes"
        ),
        "source_frame_sha256": None,
        "selected_constraint_nnz": estimates.get("selected_model_nnz"),
        "selected_upper_nnz": estimates.get("selected_upper_nnz"),
        "equality_nnz": estimates.get("equality_nnz"),
        "selected_model_rows": estimates.get("selected_model_rows"),
        "full_scan_rows": None,
        "expected_full_scan_rows": estimates.get("full_scan_rows"),
        "candidate_return_status": "not_started",
        "native_model_clear_status": "not_created",
        "allocator_trim_status": "not_attempted",
        "allocator_trim_return_code": None,
        "candidate_receipt_sha256": None,
        "candidate_frame_sha256": None,
        "candidate_status": None,
        "candidate_max_rounds": 1,
        "candidate_full_scan_count": None,
        "parent_observed_peak_rss_bytes": None,
        "parent_observed_cgroup_aggregate_current_peak_bytes": None,
        "parent_terminal_monotonic_hex": None,
        "worker_exit_code": None,
        "worker_signal": None,
        "worker_terminal_monotonic_hex": None,
        "worker_terminal_deadline_respected": False,
        "source_frame_build_elapsed_seconds_hex": None,
        "cg_elapsed_seconds_hex": None,
        "total_elapsed_seconds_hex": None,
        "preflight": None,
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "uses_dense_hstack": False,
        "uses_dense_vstack": False,
        "used_merged_sparse_frame": False,
        "materialized_full_candidate_csr": False,
        "limitations": list(_LIMITATIONS),
    }


def _bind_preflight(
    diagnostic: dict[str, Any], preflight: Mapping[str, Any]
) -> None:
    cgroup = preflight.get("cgroup") or {}
    levels = cgroup.get("ancestor_limits") or []
    leaf = levels[0] if levels else {}
    aggregate = preflight.get("cgroup_aggregate_metrics") or {}
    diagnostic.update(
        preflight=copy.deepcopy(dict(preflight)),
        kernel_hard_limit_enforced=bool(
            preflight.get("kernel_hard_limit_enforced", False)
        ),
        rss_cap_enforcement=(
            "bounded_v2_leaf_kernel_limit_plus_aggregate_memory_monitor"
            if preflight.get("kernel_hard_limit_enforced", False)
            else "parent_process_rss_polling_diagnostic_only"
        ),
        cgroup_leaf_path=cgroup.get("leaf"),
        cgroup_leaf_memory_current_bytes=leaf.get("current_bytes"),
        cgroup_leaf_memory_max_bytes=leaf.get("max_bytes"),
        cgroup_ancestor_min_memory_max_bytes=cgroup.get("max_bytes"),
        cgroup_aggregate_memory_current_start_bytes=aggregate.get(
            "current_bytes"
        ),
        cgroup_aggregate_metrics_readable=bool(
            aggregate.get("readable", False)
        ),
    )


def _validate_config(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    config = dict(raw)
    estimates = estimate_topology_resources(
        n_cont=config.get("n_cont"),
        n_bin=config.get("n_bin"),
        n_upper=config.get("n_upper"),
        n_eq=config.get("n_eq"),
        constraint_nnz=config.get("constraint_nnz"),
        selected_upper_rows=config.get("selected_upper_rows"),
        scan_chunk_rows=config.get("scan_chunk_rows", DEFAULT_SCAN_CHUNK_ROWS),
    )
    for key in (
        "n_cont",
        "n_bin",
        "n_upper",
        "n_eq",
        "constraint_nnz",
        "selected_upper_rows",
        "scan_chunk_rows",
    ):
        config[key] = int(config[key])
    for key in ("absolute_rss_cap_bytes", "rss_reserve_bytes"):
        config[key] = _strict_int(config.get(key), key, 1, 1 << 63)
    if config["absolute_rss_cap_bytes"] > HARD_ABSOLUTE_RSS_CAP_BYTES:
        raise ValueError("absolute_rss_cap_exceeds_2_5_gib_hard_cap")
    if config["rss_reserve_bytes"] < MINIMUM_RSS_RESERVE_BYTES:
        raise ValueError("rss_reserve_below_64_mib_hard_minimum")
    if config["rss_reserve_bytes"] >= config["absolute_rss_cap_bytes"]:
        raise ValueError("rss_reserve_leaves_no_usable_rss")
    deadline = config.get("deadline")
    if isinstance(deadline, (bool, np.bool_)):
        raise ValueError("deadline_must_be_finite_absolute_monotonic_time")
    try:
        deadline = float(deadline)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "deadline_must_be_finite_absolute_monotonic_time"
        ) from exc
    if not math.isfinite(deadline):
        raise ValueError("deadline_must_be_finite_absolute_monotonic_time")
    config["deadline"] = deadline
    execute_large = config.get("execute_large_profile", False)
    if not isinstance(execute_large, (bool, np.bool_)):
        raise ValueError("execute_large_profile_requires_explicit_boolean")
    config["execute_large_profile"] = bool(execute_large)
    profile_name = config.get("profile_name")
    if profile_name not in (None, PROFILE_NAME):
        raise ValueError("unknown_split_cg_memory_profile")
    return config, estimates


def _preflight(
    config: Mapping[str, Any], estimates: Mapping[str, int]
) -> tuple[dict[str, Any], Optional[str]]:
    now = time.monotonic()
    remaining = float(config["deadline"]) - now
    current = _current_rss_bytes()
    host_available = _read_mem_available_bytes()
    cgroup = _read_cgroup_memory()
    effective_limit = (
        int(config["absolute_rss_cap_bytes"])
        - int(config["rss_reserve_bytes"])
    )
    import_rss_estimate = int(current or (96 * (1 << 20)))
    estimated_worker_peak = (
        import_rss_estimate
        + estimates["source_csr_payload_bytes"]
        + estimates["source_dense_payload_bytes"]
        + estimates["estimated_candidate_increment_bytes"]
    )
    is_large = (
        int(config["constraint_nnz"]) >= LARGE_NNZ_THRESHOLD
        or config.get("profile_name") == PROFILE_NAME
    )
    cgroup_headroom = cgroup.get("headroom_bytes")
    ancestor_limits = cgroup.get("ancestor_limits") or []
    leaf_limit = (
        ancestor_limits[0].get("max_bytes") if ancestor_limits else None
    )
    aggregate_metrics = (
        _read_v2_leaf_aggregate_memory(cgroup.get("leaf"))
        if cgroup.get("version") == 2 and cgroup.get("detected")
        else {
            "readable": False,
            "leaf": cgroup.get("leaf"),
            "current_bytes": None,
            "peak_bytes": None,
            "sampled_monotonic_hex": now.hex(),
            "error": "bounded_cgroup_v2_leaf_unavailable",
        }
    )
    kernel_hard_limit_enforced = bool(
        cgroup.get("detected")
        and cgroup.get("version") == 2
        and cgroup.get("boundary_complete")
        and cgroup.get("error") is None
        and leaf_limit is not None
        and int(leaf_limit) <= int(config["absolute_rss_cap_bytes"])
    )
    checks = {
        "large_topology": is_large,
        "explicit_large_execution_consent": bool(
            config.get("execute_large_profile")
        ),
        "deadline_remaining_seconds": max(0.0, remaining),
        "minimum_large_deadline_seconds": MINIMUM_LARGE_DEADLINE_SECONDS,
        "parent_current_rss_bytes": current,
        "host_mem_available_bytes": host_available,
        "cgroup": cgroup,
        "kernel_hard_limit_enforced": kernel_hard_limit_enforced,
        "kernel_hard_limit_leaf_max_bytes": leaf_limit,
        "kernel_hard_limit_requirement": (
            "bounded_current_leaf_memory_max_at_or_below_requested_cap"
        ),
        "cgroup_aggregate_metrics": aggregate_metrics,
        "effective_rss_limit_bytes": effective_limit,
        "estimated_worker_peak_bytes": estimated_worker_peak,
        "absolute_cap_estimate_ok": estimated_worker_peak <= effective_limit,
        "host_available_estimate_ok": (
            host_available is not None
            and estimated_worker_peak <= host_available
        ),
        "cgroup_headroom_estimate_ok": (
            cgroup_headroom is not None
            and estimated_worker_peak <= int(cgroup_headroom)
        ),
    }
    if remaining <= 0.0:
        return checks, "deadline_expired_before_child_start"
    if is_large and not config.get("execute_large_profile"):
        return checks, "large_profile_requires_explicit_execute_flag"
    if not checks["absolute_cap_estimate_ok"]:
        return checks, "absolute_rss_cap_preflight_failed"
    if is_large and remaining < MINIMUM_LARGE_DEADLINE_SECONDS:
        return checks, "large_profile_deadline_preflight_failed"
    if is_large and host_available is None:
        return checks, "large_profile_host_memory_preflight_unavailable"
    if is_large and not checks["host_available_estimate_ok"]:
        return checks, "large_profile_host_memory_preflight_failed"
    if is_large and not cgroup.get("detected"):
        return checks, "large_profile_cgroup_preflight_unavailable"
    if is_large and cgroup.get("version") != 2:
        return checks, "large_profile_requires_cgroup_v2"
    if is_large and cgroup.get("error") is not None:
        return checks, "large_profile_cgroup_preflight_error"
    if is_large and not cgroup.get("boundary_complete"):
        return checks, "large_profile_cgroup_delegation_boundary_unverified"
    if is_large and cgroup.get("unlimited"):
        return checks, "large_profile_cgroup_has_no_finite_hard_limit"
    if is_large and not checks["cgroup_headroom_estimate_ok"]:
        return checks, "large_profile_cgroup_memory_preflight_failed"
    if is_large and not kernel_hard_limit_enforced:
        return checks, "large_profile_kernel_hard_rss_limit_unavailable"
    if is_large and not aggregate_metrics.get("readable"):
        return checks, "large_profile_cgroup_aggregate_metrics_unreadable"
    if is_large and (
        int(aggregate_metrics["current_bytes"]) > effective_limit
        or int(aggregate_metrics["peak_bytes"]) > effective_limit
    ):
        return checks, "large_profile_aggregate_stoploss_already_exceeded"
    return checks, None


def _update_hash_array(digest: Any, array: np.ndarray) -> None:
    view = memoryview(np.ascontiguousarray(array)).cast("B")
    for start in range(0, len(view), _HASH_CHUNK_BYTES):
        digest.update(view[start : start + _HASH_CHUNK_BYTES])


def _csr_from_row_segments(
    *,
    row_counts: np.ndarray,
    row_offset: int,
    n_cont: int,
    n_bin: int,
    deadline: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    rows = int(row_counts.size)
    n_columns = n_cont + n_bin
    cont_counts = np.empty(rows, dtype=np.int32)
    bin_counts = np.empty(rows, dtype=np.int32)
    starts = np.empty(rows, dtype=np.int64)
    for local_row in range(rows):
        if (local_row & 4095) == 0 and time.monotonic() >= deadline:
            raise TimeoutError("deadline_expired_during_csr_topology_plan")
        width = int(row_counts[local_row])
        global_row = row_offset + local_row
        span = n_columns - width + 1
        start = (global_row * 104729 + 8191) % span
        stop = start + width
        continuous = max(0, min(stop, n_cont) - min(start, n_cont))
        starts[local_row] = start
        cont_counts[local_row] = continuous
        bin_counts[local_row] = width - continuous

    cont_indptr = np.empty(rows + 1, dtype=np.int32)
    bin_indptr = np.empty(rows + 1, dtype=np.int32)
    cont_indptr[0] = 0
    bin_indptr[0] = 0
    np.cumsum(cont_counts, dtype=np.int64, out=cont_indptr[1:])
    np.cumsum(bin_counts, dtype=np.int64, out=bin_indptr[1:])
    cont_nnz = int(cont_indptr[-1])
    bin_nnz = int(bin_indptr[-1])
    if cont_nnz > _INT32_MAX or bin_nnz > _INT32_MAX:
        raise ValueError("split_csr_block_nnz_exceeds_int32")
    cont_indices = np.empty(cont_nnz, dtype=np.int32)
    bin_indices = np.empty(bin_nnz, dtype=np.int32)
    cont_data = np.ones(cont_nnz, dtype=np.float64)
    bin_data = np.ones(bin_nnz, dtype=np.float64)
    for local_row in range(rows):
        if (local_row & 4095) == 0 and time.monotonic() >= deadline:
            raise TimeoutError("deadline_expired_during_csr_topology_fill")
        start = int(starts[local_row])
        width = int(row_counts[local_row])
        stop = start + width
        continuous_stop = min(stop, n_cont)
        if start < continuous_stop:
            destination = int(cont_indptr[local_row])
            count = continuous_stop - start
            cont_indices[destination : destination + count] = np.arange(
                start, continuous_stop, dtype=np.int32
            )
        binary_start = max(start, n_cont)
        if binary_start < stop:
            destination = int(bin_indptr[local_row])
            count = stop - binary_start
            bin_indices[destination : destination + count] = np.arange(
                binary_start - n_cont, stop - n_cont, dtype=np.int32
            )
    continuous = sp.csr_matrix(
        (cont_data, cont_indices, cont_indptr),
        shape=(rows, n_cont),
        dtype=np.float64,
        copy=False,
    )
    binary = sp.csr_matrix(
        (bin_data, bin_indices, bin_indptr),
        shape=(rows, n_bin),
        dtype=np.float64,
        copy=False,
    )
    if not continuous.has_canonical_format or not binary.has_canonical_format:
        raise RuntimeError("synthetic_csr_not_canonical")
    return continuous, binary, starts


def _build_synthetic_frame(
    config: Mapping[str, Any], estimates: Mapping[str, int]
) -> _SyntheticFrame:
    deadline = float(config["deadline"])
    n_upper = int(config["n_upper"])
    n_eq = int(config["n_eq"])
    total_rows = n_upper + n_eq
    total_nnz = int(config["constraint_nnz"])
    base, extra = divmod(total_nnz, total_rows)
    row_counts = np.full(total_rows, base, dtype=np.int32)
    if extra:
        row_counts[:extra] += 1
    if np.any(row_counts <= 0):
        raise RuntimeError("synthetic_constraint_row_is_degenerate")
    upper_counts = row_counts[:n_upper]
    equality_counts = row_counts[n_upper:]
    Auc, Aub, _upper_starts = _csr_from_row_segments(
        row_counts=upper_counts,
        row_offset=0,
        n_cont=int(config["n_cont"]),
        n_bin=int(config["n_bin"]),
        deadline=deadline,
    )
    Ac, Ab, _equality_starts = _csr_from_row_segments(
        row_counts=equality_counts,
        row_offset=n_upper,
        n_cont=int(config["n_cont"]),
        n_bin=int(config["n_bin"]),
        deadline=deadline,
    )
    del _upper_starts, _equality_starts
    ub = upper_counts.astype(np.float64) + 1.0
    b = np.zeros(n_eq, dtype=np.float64)
    n_columns = int(config["n_cont"]) + int(config["n_bin"])
    q = np.zeros(n_columns, dtype=np.float64)
    q[0] = 1.0
    lower = np.full(n_columns, -1.0, dtype=np.float64)
    upper = np.full(n_columns, 1.0, dtype=np.float64)
    selected = int(config["selected_upper_rows"])
    selected_binary_nnz = int(Aub.indptr[selected])
    equality_binary_nnz = int(Ab.nnz)
    if (
        selected_binary_nnz + equality_binary_nnz
        > _MAX_BINARY_CHANGE_COEFFICIENTS
    ):
        raise ValueError("selected_binary_change_coefficients_exceed_frozen_cap")

    digest = hashlib.sha256()
    digest.update(b"act.split_cg.synthetic_split_frame.v1\0")
    digest.update(
        json.dumps(
            {
                "n_cont": int(config["n_cont"]),
                "n_bin": int(config["n_bin"]),
                "n_upper": n_upper,
                "n_eq": n_eq,
                "constraint_nnz": total_nnz,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, matrix in (("Auc", Auc), ("Aub", Aub), ("Ac", Ac), ("Ab", Ab)):
        digest.update(name.encode("ascii") + b"\0")
        _update_hash_array(digest, matrix.indptr)
        _update_hash_array(digest, matrix.indices)
        _update_hash_array(digest, matrix.data)
    for name, array in (
        ("ub", ub),
        ("b", b),
        ("q", q),
        ("lower", lower),
        ("upper", upper),
    ):
        digest.update(name.encode("ascii") + b"\0")
        _update_hash_array(digest, array)
    if time.monotonic() >= deadline:
        raise TimeoutError("deadline_expired_after_source_residency_hash")
    actual_csr_payload = sum(
        int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
        for matrix in (Auc, Aub, Ac, Ab)
    )
    actual_dense_payload = sum(
        int(array.nbytes) for array in (ub, b, q, lower, upper)
    )
    if actual_csr_payload != estimates["source_csr_payload_bytes"]:
        raise RuntimeError("source_csr_payload_estimate_mismatch")
    if sum(int(matrix.nnz) for matrix in (Auc, Aub, Ac, Ab)) != total_nnz:
        raise RuntimeError("source_constraint_nnz_mismatch")
    return _SyntheticFrame(
        Auc=Auc,
        Aub=Aub,
        Ac=Ac,
        Ab=Ab,
        ub=ub,
        b=b,
        q=q,
        lower_bounds=lower,
        upper_bounds=upper,
        source_csr_payload_bytes=actual_csr_payload,
        source_dense_payload_bytes=actual_dense_payload,
        source_frame_sha256=digest.hexdigest(),
        selected_upper_nnz=int(Auc.indptr[selected] + Aub.indptr[selected]),
        equality_nnz=int(Ac.nnz + Ab.nnz),
        selected_binary_nnz=selected_binary_nnz,
        equality_binary_nnz=equality_binary_nnz,
    )


class _RSSSampler:
    def __init__(self, initial: int):
        self._stop = threading.Event()
        self.peak = int(initial)
        self._thread = threading.Thread(
            target=self._run, name="split-cg-rss-sampler", daemon=True
        )

    def _run(self) -> None:
        while not self._stop.wait(_POLL_SECONDS):
            current = _current_rss_bytes()
            if current is not None:
                self.peak = max(self.peak, int(current))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        current = _current_rss_bytes()
        if current is not None:
            self.peak = max(self.peak, int(current))
        self._stop.set()
        self._thread.join(timeout=1.0)
        return self.peak


def _malloc_trim() -> tuple[str, Optional[int]]:
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return "unavailable", None
    try:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        result = int(trim(0))
    except (AttributeError, OSError, ValueError):
        return "error", None
    return ("released" if result else "no_release"), result


def _rss_delta(value: Optional[int], baseline: Optional[int]) -> Optional[int]:
    if value is None or baseline is None:
        return None
    return int(value) - int(baseline)


def _candidate_receipt_contract_ok(
    receipt: Mapping[str, Any],
    *,
    expected_full_scan_rows: int,
    deadline: float,
) -> bool:
    return bool(
        receipt.get("status") == "full_scan_candidate_feasible"
        and receipt.get("candidate_only") is True
        and receipt.get("full_split_scan_count") == 1
        and receipt.get("full_split_rows_scanned") == expected_full_scan_rows
        and receipt.get("native_model_closed_before_return") is True
        and receipt.get("proof_authority") is False
        and receipt.get("verdict_authority") is False
        and receipt.get("primal_feasibility_authority") is False
        and receipt.get("parent_binding") is False
        and receipt.get("parent_binding_authority") is False
        and receipt.get("caps", {}).get("max_rounds") == 1
        and receipt.get("absolute_deadline_hex") == float(deadline).hex()
        and receipt.get("uses_sparse_hstack") is False
        and receipt.get("uses_sparse_vstack") is False
        and receipt.get("uses_dense_hstack") is False
        and receipt.get("uses_dense_vstack") is False
        and receipt.get("used_merged_sparse_frame") is False
        and receipt.get("materialized_full_candidate_csr") is False
    )


def _execute_worker(config: Mapping[str, Any]) -> dict[str, Any]:
    worker_started = time.monotonic()
    try:
        checked, estimates = _validate_config(config)
    except Exception as exc:
        diagnostic = _base_diagnostic(_safe_config_summary(config))
        diagnostic.update(
            status="worker_config_rejected",
            reason=_safe_exception_text(exc),
        )
        return _seal_diagnostic(diagnostic)
    diagnostic = _base_diagnostic(checked, estimates)
    worker_preflight, worker_preflight_reason = _preflight(
        checked, estimates
    )
    _bind_preflight(diagnostic, worker_preflight)
    if worker_preflight_reason is not None:
        diagnostic.update(
            status="worker_preflight_rejected",
            reason=worker_preflight_reason,
            candidate_return_status="not_started",
            native_model_clear_status="not_created",
        )
        return _seal_diagnostic(diagnostic)
    effective_limit = (
        checked["absolute_rss_cap_bytes"] - checked["rss_reserve_bytes"]
    )
    worker_cgroup = worker_preflight.get("cgroup") or {}
    worker_leaf = worker_cgroup.get("leaf")
    aggregate_required = bool(
        worker_preflight.get("kernel_hard_limit_enforced", False)
    )
    diagnostic["effective_rss_limit_bytes"] = effective_limit
    frame: Optional[_SyntheticFrame] = None
    candidate = None
    sampler: Optional[_RSSSampler] = None
    baseline: Optional[int] = None
    post_clear: Optional[int] = None
    sampled_peak: Optional[int] = None
    try:
        frame_build_started = time.monotonic()
        frame = _build_synthetic_frame(checked, estimates)
        frame_build_finished = time.monotonic()
        diagnostic["source_frame_build_elapsed_seconds_hex"] = (
            frame_build_finished - frame_build_started
        ).hex()
        aggregate_after_frame = _read_v2_leaf_aggregate_memory(worker_leaf)
        diagnostic.update(
            cgroup_aggregate_memory_current_after_frame_bytes=(
                aggregate_after_frame.get("current_bytes")
            ),
            cgroup_aggregate_metrics_readable=bool(
                aggregate_after_frame.get("readable")
            ),
        )
        if aggregate_required and not aggregate_after_frame.get("readable"):
            raise RuntimeError("cgroup_aggregate_after_frame_unreadable")
        baseline = _current_rss_bytes()
        if baseline is None:
            raise RuntimeError("baseline_current_rss_unavailable")
        allowed = effective_limit - baseline
        aggregate_allowed = (
            effective_limit - int(aggregate_after_frame["current_bytes"])
            if aggregate_after_frame.get("readable")
            else None
        )
        diagnostic.update(
            baseline_current_rss_bytes=baseline,
            allowed_rss_increment_bytes=max(0, allowed),
            child_process_baseline_current_rss_bytes=baseline,
            child_process_allowed_rss_increment_bytes=max(0, allowed),
            source_csr_payload_bytes=frame.source_csr_payload_bytes,
            source_dense_payload_bytes=frame.source_dense_payload_bytes,
            source_frame_sha256=frame.source_frame_sha256,
            selected_constraint_nnz=(
                frame.selected_upper_nnz + frame.equality_nnz
            ),
            selected_upper_nnz=frame.selected_upper_nnz,
            equality_nnz=frame.equality_nnz,
            selected_model_rows=(
                checked["selected_upper_rows"] + checked["n_eq"]
            ),
            cgroup_allowed_increment_after_frame_bytes=(
                max(0, aggregate_allowed)
                if aggregate_allowed is not None
                else None
            ),
        )
        admission_allowed = (
            aggregate_allowed if aggregate_required else allowed
        )
        if admission_allowed < estimates["estimated_candidate_increment_bytes"]:
            diagnostic.update(
                status="worker_cap_preflight_rejected",
                reason="allowed_increment_below_candidate_estimate",
                candidate_return_status="not_called",
                native_model_clear_status="not_created",
            )
        else:
            sampler = _RSSSampler(baseline)
            sampler.start()
            selected_rows = np.arange(
                checked["selected_upper_rows"], dtype=np.int32
            )
            equality_rows = np.arange(checked["n_eq"], dtype=np.int32)
            cg_started = time.monotonic()
            candidate = _scg.propose_split_constraint_generation_candidate(
                Auc=frame.Auc,
                Aub=frame.Aub,
                Ac=frame.Ac,
                Ab=frame.Ab,
                ub=frame.ub,
                b=frame.b,
                q=frame.q,
                lower_bounds=frame.lower_bounds,
                upper_bounds=frame.upper_bounds,
                seed_upper_rows=selected_rows,
                seed_upper_duals=np.zeros(
                    checked["selected_upper_rows"], dtype=np.float64
                ),
                seed_equality_rows=equality_rows,
                seed_equality_duals=np.zeros(
                    checked["n_eq"], dtype=np.float64
                ),
                deadline=checked["deadline"],
                max_rounds=1,
                add_batch=1,
                max_selected_upper_rows=checked["selected_upper_rows"],
                max_equality_rows=checked["n_eq"],
                max_binary_change_coefficients=(
                    frame.selected_binary_nnz + frame.equality_binary_nnz
                ),
                scan_chunk_rows=checked["scan_chunk_rows"],
                threads=1,
            )
            candidate_return_monotonic = time.monotonic()
            diagnostic["cg_elapsed_seconds_hex"] = (
                candidate_return_monotonic - cg_started
            ).hex()
            post_clear = _current_rss_bytes()
            aggregate_after_cg = _read_v2_leaf_aggregate_memory(worker_leaf)
            diagnostic[
                "cgroup_aggregate_memory_current_after_cg_bytes"
            ] = aggregate_after_cg.get("current_bytes")
            if aggregate_required and not aggregate_after_cg.get("readable"):
                raise RuntimeError("cgroup_aggregate_after_cg_unreadable")
            sampled_peak = sampler.stop()
            sampler = None
            receipt = candidate.receipt
            full_scan_rows = int(receipt["full_split_rows_scanned"])
            candidate_ok = (
                candidate_return_monotonic < checked["deadline"]
                and _candidate_receipt_contract_ok(
                    receipt,
                    expected_full_scan_rows=estimates["full_scan_rows"],
                    deadline=checked["deadline"],
                )
            )
            diagnostic.update(
                status="ok" if candidate_ok else "candidate_contract_failed",
                reason=None if candidate_ok else "frozen_candidate_receipt_mismatch",
                current_rss_bytes=post_clear,
                child_process_current_after_cg_bytes=post_clear,
                full_scan_rows=full_scan_rows,
                candidate_return_status=(
                    "returned_candidate_only" if candidate_ok else "returned_mismatch"
                ),
                native_model_clear_status=(
                    "cleared_before_return"
                    if receipt.get("native_model_closed_before_return")
                    else "not_confirmed"
                ),
                candidate_receipt_sha256=receipt.get("receipt_sha256"),
                candidate_frame_sha256=receipt.get(
                    "provided_split_frame_sha256"
                ),
                candidate_status=receipt.get("status"),
                candidate_full_scan_count=receipt.get("full_split_scan_count"),
                uses_sparse_hstack=receipt.get("uses_sparse_hstack"),
                uses_sparse_vstack=receipt.get("uses_sparse_vstack"),
                uses_dense_hstack=receipt.get("uses_dense_hstack"),
                uses_dense_vstack=receipt.get("uses_dense_vstack"),
                used_merged_sparse_frame=receipt.get(
                    "used_merged_sparse_frame"
                ),
                materialized_full_candidate_csr=receipt.get(
                    "materialized_full_candidate_csr"
                ),
            )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        diagnostic.update(
            status="worker_error",
            reason=_safe_exception_text(exc),
            candidate_return_status="failed_closed",
            native_model_clear_status=(
                "candidate_enforces_clear_before_error"
                if frame is not None
                else "not_created"
            ),
        )
    finally:
        if sampler is not None:
            sampled_peak = sampler.stop()
        if post_clear is None:
            post_clear = _current_rss_bytes()
        diagnostic["current_rss_bytes"] = post_clear
        candidate = None
        import gc

        gc.collect()
        trim_status, trim_code = _malloc_trim()
        post_trim = _current_rss_bytes()
        process_peak = _peak_rss_bytes()
        aggregate_terminal = _read_v2_leaf_aggregate_memory(worker_leaf)
        aggregate_current_terminal = aggregate_terminal.get("current_bytes")
        aggregate_peak_terminal = aggregate_terminal.get("peak_bytes")
        diagnostic.update(
            allocator_trim_status=trim_status,
            allocator_trim_return_code=trim_code,
            post_trim_current_rss_bytes=post_trim,
            process_peak_rss_bytes=process_peak,
            child_process_current_after_cg_bytes=post_clear,
            child_process_current_after_trim_bytes=post_trim,
            child_process_peak_rss_bytes=process_peak,
            peak_rss_bytes=(
                aggregate_peak_terminal if aggregate_required else process_peak
            ),
            stoploss_peak_bytes=(
                aggregate_peak_terminal if aggregate_required else process_peak
            ),
            stoploss_peak_source=(
                "cgroup_v2_leaf_aggregate_memory_peak"
                if aggregate_required
                else "process_lifetime_hwm_diagnostic_only"
            ),
            cgroup_aggregate_memory_current_terminal_bytes=(
                aggregate_current_terminal
            ),
            cgroup_aggregate_memory_peak_terminal_bytes=(
                aggregate_peak_terminal
            ),
            cgroup_aggregate_metrics_readable=bool(
                aggregate_terminal.get("readable")
            ),
            cg_sampled_peak_rss_bytes=sampled_peak,
            current_delta_from_baseline_bytes=_rss_delta(post_clear, baseline),
            post_trim_delta_from_baseline_bytes=_rss_delta(post_trim, baseline),
            peak_delta_from_baseline_bytes=(
                max(0, _rss_delta(process_peak, baseline) or 0)
                if process_peak is not None and baseline is not None
                else None
            ),
            cg_sampled_peak_delta_from_baseline_bytes=(
                max(0, _rss_delta(sampled_peak, baseline) or 0)
                if sampled_peak is not None and baseline is not None
                else None
            ),
            child_process_current_delta_from_baseline_bytes=_rss_delta(
                post_clear, baseline
            ),
            child_process_post_trim_delta_from_baseline_bytes=_rss_delta(
                post_trim, baseline
            ),
            child_process_peak_delta_bytes=(
                max(0, _rss_delta(process_peak, baseline) or 0)
                if process_peak is not None and baseline is not None
                else None
            ),
        )
        respected = (
            aggregate_terminal.get("readable") is True
            and aggregate_peak_terminal <= effective_limit
            if aggregate_required
            else process_peak is not None and process_peak <= effective_limit
        )
        diagnostic["rss_cap_respected"] = respected
        if aggregate_required:
            diagnostic.update(
                generic_rss_fields_scope="unset_in_bounded_v2_mode",
                allowed_rss_increment_bytes=None,
                baseline_current_rss_bytes=None,
                current_rss_bytes=None,
                post_trim_current_rss_bytes=None,
                peak_rss_bytes=None,
                current_delta_from_baseline_bytes=None,
                post_trim_delta_from_baseline_bytes=None,
                peak_delta_from_baseline_bytes=None,
                cgroup_aggregate_peak_delta_from_after_frame_bytes=None,
            )
        else:
            diagnostic["generic_rss_fields_scope"] = (
                "child_process_diagnostic_only"
            )
        if diagnostic["status"] == "ok" and not respected:
            diagnostic.update(
                status=(
                    "rss_cap_exceeded"
                    if aggregate_peak_terminal is not None
                    or not aggregate_required
                    else "cgroup_aggregate_metrics_unreadable"
                ),
                reason=(
                    "aggregate_peak_exceeded_effective_rss_limit"
                    if aggregate_required
                    and aggregate_peak_terminal is not None
                    else "aggregate_peak_unreadable"
                    if aggregate_required
                    else "process_peak_exceeded_effective_rss_limit"
                ),
            )
        terminal_monotonic = time.monotonic()
        diagnostic.update(
            worker_terminal_monotonic_hex=terminal_monotonic.hex(),
            total_elapsed_seconds_hex=(
                terminal_monotonic - worker_started
            ).hex(),
            worker_terminal_deadline_respected=(
                terminal_monotonic < checked["deadline"]
            ),
        )
        if (
            terminal_monotonic >= checked["deadline"]
            and diagnostic["status"] == "ok"
        ):
            diagnostic.update(
                status="deadline_exceeded",
                reason="worker_terminal_gate_rejected_late_completion",
            )
    return _seal_diagnostic(diagnostic)


def _worker_command(config: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-config-json",
        json.dumps(config, sort_keys=True, separators=(",", ":")),
    ]


def _is_nonnegative_int(value: Any) -> bool:
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, (bool, np.bool_))
        and int(value) >= 0
    )


def _finite_nonnegative_hex(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = float.fromhex(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(parsed) and parsed >= 0.0


def _validate_child_diagnostic_contract(
    child: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    estimates: Mapping[str, int],
    preflight: Mapping[str, Any],
) -> Optional[str]:
    """Validate a checksummed worker payload before any coercion/arithmetic."""

    try:
        allowed_statuses = {
            "ok",
            "worker_config_rejected",
            "worker_preflight_rejected",
            "worker_cap_preflight_rejected",
            "candidate_contract_failed",
            "worker_error",
            "rss_cap_exceeded",
            "cgroup_aggregate_metrics_unreadable",
            "deadline_exceeded",
            "worker_bootstrap_error",
        }
        if child.get("schema") != SCHEMA:
            return "schema_mismatch"
        if child.get("status") not in allowed_statuses:
            return "status_not_allowed"
        if child.get("reason") is not None and not isinstance(
            child.get("reason"), str
        ):
            return "reason_not_string_or_null"
        for field, expected in (
            ("candidate_only", True),
            ("proof_authority", False),
            ("verdict_authority", False),
            ("primal_feasibility_authority", False),
            ("parent_binding", False),
            ("parent_binding_authority", False),
        ):
            if child.get(field) is not expected:
                return f"authority_field_invalid:{field}"
        expected_leaf = (preflight.get("cgroup") or {}).get("leaf")
        if child.get("cgroup_leaf_path") != expected_leaf:
            return "cgroup_leaf_binding_mismatch"
        if child.get("absolute_deadline_monotonic_hex") != float(
            config["deadline"]
        ).hex():
            return "absolute_deadline_binding_mismatch"
        bounded = bool(preflight.get("kernel_hard_limit_enforced", False))
        if child.get("kernel_hard_limit_enforced") is not bounded:
            return "kernel_limit_binding_mismatch"

        # Error diagnostics may legitimately terminate before frame/CG stage
        # fields exist.  Successful payloads must bind every field consumed by
        # the parent before the parent performs int conversion or arithmetic.
        if child.get("status") != "ok":
            terminal = child.get("worker_terminal_monotonic_hex")
            if terminal is not None and not _finite_nonnegative_hex(terminal):
                return "error_terminal_time_invalid"
            for field in (
                "process_peak_rss_bytes",
                "child_process_baseline_current_rss_bytes",
            ):
                value = child.get(field)
                if value is not None and not _is_nonnegative_int(value):
                    return f"error_metric_invalid:{field}"
            return None

        if child.get("worker_terminal_deadline_respected") is not True:
            return "worker_terminal_deadline_flag_invalid"
        terminal_hex = child.get("worker_terminal_monotonic_hex")
        if not _finite_nonnegative_hex(terminal_hex):
            return "worker_terminal_time_invalid"
        if float.fromhex(terminal_hex) >= float(config["deadline"]):
            return "worker_terminal_after_deadline"
        for field in (
            "source_frame_build_elapsed_seconds_hex",
            "cg_elapsed_seconds_hex",
            "total_elapsed_seconds_hex",
        ):
            if not _finite_nonnegative_hex(child.get(field)):
                return f"elapsed_field_invalid:{field}"
        if child.get("candidate_return_status") != "returned_candidate_only":
            return "candidate_return_status_invalid"
        if child.get("native_model_clear_status") != "cleared_before_return":
            return "native_model_clear_status_invalid"
        if child.get("candidate_status") != "full_scan_candidate_feasible":
            return "candidate_status_invalid"
        if child.get("candidate_max_rounds") != 1:
            return "candidate_max_rounds_invalid"
        if child.get("candidate_full_scan_count") != 1:
            return "candidate_full_scan_count_invalid"
        if child.get("full_scan_rows") != estimates["full_scan_rows"]:
            return "full_scan_rows_invalid"
        for field in (
            "uses_sparse_hstack",
            "uses_sparse_vstack",
            "uses_dense_hstack",
            "uses_dense_vstack",
            "used_merged_sparse_frame",
            "materialized_full_candidate_csr",
        ):
            if child.get(field) is not False:
                return f"merge_contract_invalid:{field}"
        for field in (
            "source_frame_sha256",
            "candidate_receipt_sha256",
            "candidate_frame_sha256",
        ):
            value = child.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                return f"sha256_field_invalid:{field}"
        for field in (
            "process_peak_rss_bytes",
            "child_process_baseline_current_rss_bytes",
            "child_process_current_after_cg_bytes",
            "child_process_current_after_trim_bytes",
            "child_process_peak_rss_bytes",
        ):
            if not _is_nonnegative_int(child.get(field)):
                return f"child_process_metric_invalid:{field}"
        baseline = int(child["child_process_baseline_current_rss_bytes"])
        current = int(child["child_process_current_after_cg_bytes"])
        trimmed = int(child["child_process_current_after_trim_bytes"])
        peak = int(child["child_process_peak_rss_bytes"])
        if child.get("child_process_current_delta_from_baseline_bytes") != (
            current - baseline
        ):
            return "child_current_delta_invalid"
        if child.get(
            "child_process_post_trim_delta_from_baseline_bytes"
        ) != (trimmed - baseline):
            return "child_trim_delta_invalid"
        if child.get("child_process_peak_delta_bytes") != max(
            0, peak - baseline
        ):
            return "child_peak_delta_invalid"
        if bounded:
            for field in (
                "cgroup_aggregate_memory_current_after_frame_bytes",
                "cgroup_aggregate_memory_current_after_cg_bytes",
                "cgroup_aggregate_memory_current_terminal_bytes",
                "cgroup_aggregate_memory_peak_terminal_bytes",
            ):
                if not _is_nonnegative_int(child.get(field)):
                    return f"aggregate_stage_metric_invalid:{field}"
            if child.get("cgroup_aggregate_peak_delta_from_after_frame_bytes") is not None:
                return "aggregate_lifetime_peak_delta_must_be_null"
            for field in (
                "allowed_rss_increment_bytes",
                "baseline_current_rss_bytes",
                "current_rss_bytes",
                "post_trim_current_rss_bytes",
                "peak_rss_bytes",
                "current_delta_from_baseline_bytes",
                "post_trim_delta_from_baseline_bytes",
                "peak_delta_from_baseline_bytes",
            ):
                if child.get(field) is not None:
                    return f"bounded_generic_rss_field_must_be_null:{field}"
        return None
    except Exception as exc:
        return "validator_exception:" + _safe_exception_text(exc)


def _safe_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value[-2048:]
    if isinstance(value, bytes):
        return value[-2048:].decode("utf-8", errors="replace")
    if value is None:
        return ""
    try:
        return repr(value)[-2048:]
    except BaseException:
        return "<output_unrepresentable>"


def _best_effort_stop_process(process: Any) -> None:
    """Bounded cleanup for a worker whose normal control path failed."""

    try:
        returncode = process.poll()
    except BaseException:
        returncode = None
    if returncode is not None:
        return
    try:
        process.terminate()
    except BaseException:
        pass
    try:
        process.wait(timeout=0.5)
        return
    except BaseException:
        pass
    try:
        process.kill()
    except BaseException:
        pass
    try:
        process.wait(timeout=0.5)
    except BaseException:
        pass


def _abnormal_child_diagnostic(
    config: Mapping[str, Any],
    estimates: Mapping[str, int],
    preflight: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    observed_peak: Optional[int],
    returncode: Optional[int],
    stderr: str = "",
    aggregate_start: Optional[Mapping[str, Any]] = None,
    aggregate_terminal: Optional[Mapping[str, Any]] = None,
    parent_terminal_monotonic: Optional[float] = None,
) -> dict[str, Any]:
    diagnostic = _base_diagnostic(config, estimates)
    effective = (
        int(config["absolute_rss_cap_bytes"])
        - int(config["rss_reserve_bytes"])
    )
    diagnostic.update(
        status=status,
        reason=reason,
        effective_rss_limit_bytes=effective,
        parent_observed_peak_rss_bytes=observed_peak,
        worker_exit_code=returncode,
        worker_signal=(
            -returncode
            if returncode is not None and returncode < 0
            else None
        ),
        candidate_return_status="child_did_not_return_valid_diagnostic",
        native_model_clear_status="not_observable_after_abnormal_child",
        child_stderr=_safe_output_text(stderr),
    )
    _bind_preflight(diagnostic, preflight)
    aggregate_required = bool(
        preflight.get("kernel_hard_limit_enforced", False)
    )
    start = aggregate_start or {}
    terminal = aggregate_terminal or {}
    diagnostic.update(
        cgroup_aggregate_memory_current_start_bytes=start.get(
            "current_bytes"
        ),
        cgroup_aggregate_memory_current_terminal_bytes=terminal.get(
            "current_bytes"
        ),
        cgroup_aggregate_memory_peak_terminal_bytes=terminal.get(
            "peak_bytes"
        ),
        cgroup_aggregate_metrics_readable=bool(terminal.get("readable")),
        stoploss_peak_bytes=(
            terminal.get("peak_bytes") if aggregate_required else observed_peak
        ),
        peak_rss_bytes=(
            terminal.get("peak_bytes") if aggregate_required else observed_peak
        ),
        stoploss_peak_source=(
            "cgroup_v2_leaf_aggregate_memory_peak"
            if aggregate_required
            else "process_rss_poll_diagnostic_only"
        ),
        parent_terminal_monotonic_hex=(
            parent_terminal_monotonic.hex()
            if parent_terminal_monotonic is not None
            else None
        ),
        child_process_peak_rss_bytes=observed_peak,
        generic_rss_fields_scope=(
            "unset_in_bounded_v2_mode"
            if aggregate_required
            else "child_process_diagnostic_only"
        ),
    )
    if aggregate_required:
        diagnostic.update(
            allowed_rss_increment_bytes=None,
            baseline_current_rss_bytes=None,
            current_rss_bytes=None,
            post_trim_current_rss_bytes=None,
            peak_rss_bytes=None,
            current_delta_from_baseline_bytes=None,
            post_trim_delta_from_baseline_bytes=None,
            peak_delta_from_baseline_bytes=None,
            cgroup_aggregate_peak_delta_from_after_frame_bytes=None,
        )
    return _seal_diagnostic(diagnostic)


def _run_worker_process(
    config: Mapping[str, Any],
    estimates: Mapping[str, int],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    effective_limit = (
        int(config["absolute_rss_cap_bytes"])
        - int(config["rss_reserve_bytes"])
    )
    aggregate_required = bool(
        preflight.get("kernel_hard_limit_enforced", False)
    )
    cgroup = preflight.get("cgroup") or {}
    leaf = cgroup.get("leaf")
    aggregate_start = _read_v2_leaf_aggregate_memory(leaf)
    if aggregate_required and not aggregate_start.get("readable"):
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status="cgroup_aggregate_metrics_unreadable",
            reason="aggregate_metrics_unreadable_before_child_start",
            observed_peak=None,
            returncode=None,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_start,
            parent_terminal_monotonic=time.monotonic(),
        )
    try:
        process = subprocess.Popen(
            _worker_command(config),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
        )
    except Exception as exc:
        terminal = time.monotonic()
        aggregate_terminal = _read_v2_leaf_aggregate_memory(leaf)
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status="child_launch_failed",
            reason=_safe_exception_text(exc),
            observed_peak=None,
            returncode=None,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_terminal,
            parent_terminal_monotonic=terminal,
        )
    observed_peak = 0  # Child /proc RSS: diagnostic only when bounded.
    observed_aggregate_current_peak = int(
        aggregate_start.get("current_bytes") or 0
    )
    killed_status: Optional[str] = None
    control_error: Optional[str] = None
    while True:
        try:
            polled_returncode = process.poll()
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            control_error = "child_poll_failed:" + _safe_exception_text(exc)
            _best_effort_stop_process(process)
            break
        if polled_returncode is not None:
            break
        current = _current_rss_bytes(process.pid)
        if current is not None:
            observed_peak = max(observed_peak, current)
            if not aggregate_required and current > effective_limit:
                killed_status = "rss_cap_exceeded"
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    try:
                        exited = process.poll() is not None
                    except BaseException:
                        exited = False
                    if not exited:
                        control_error = (
                            "child_kill_failed:" + _safe_exception_text(exc)
                        )
                        _best_effort_stop_process(process)
                break
        if aggregate_required:
            aggregate_sample = _read_v2_leaf_aggregate_memory(leaf)
            if not aggregate_sample.get("readable"):
                killed_status = "cgroup_aggregate_metrics_unreadable"
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    try:
                        exited = process.poll() is not None
                    except BaseException:
                        exited = False
                    if not exited:
                        control_error = (
                            "child_kill_failed:" + _safe_exception_text(exc)
                        )
                        _best_effort_stop_process(process)
                break
            aggregate_current = int(aggregate_sample["current_bytes"])
            aggregate_peak = int(aggregate_sample["peak_bytes"])
            observed_aggregate_current_peak = max(
                observed_aggregate_current_peak, aggregate_current
            )
            if (
                aggregate_current > effective_limit
                or aggregate_peak > effective_limit
            ):
                killed_status = "rss_cap_exceeded"
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    try:
                        exited = process.poll() is not None
                    except BaseException:
                        exited = False
                    if not exited:
                        control_error = (
                            "child_kill_failed:" + _safe_exception_text(exc)
                        )
                        _best_effort_stop_process(process)
                break
        if time.monotonic() >= float(config["deadline"]):
            killed_status = "deadline_exceeded"
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except OSError as exc:
                try:
                    exited = process.poll() is not None
                except BaseException:
                    exited = False
                if not exited:
                    control_error = (
                        "child_kill_failed:" + _safe_exception_text(exc)
                    )
                    _best_effort_stop_process(process)
            break
        time.sleep(_POLL_SECONDS)
    try:
        stdout, stderr = process.communicate()
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _best_effort_stop_process(process)
        terminal = time.monotonic()
        aggregate_terminal = _read_v2_leaf_aggregate_memory(leaf)
        try:
            returncode = process.returncode
        except BaseException:
            returncode = None
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status="child_communicate_failed",
            reason=_safe_exception_text(exc),
            observed_peak=observed_peak or None,
            returncode=returncode,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_terminal,
            parent_terminal_monotonic=terminal,
        )
    parent_terminal_monotonic = time.monotonic()
    returncode = process.returncode
    aggregate_terminal = _read_v2_leaf_aggregate_memory(leaf)
    if control_error is not None:
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status="child_control_failed",
            reason=control_error,
            observed_peak=observed_peak or None,
            returncode=returncode,
            stderr=stderr,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_terminal,
            parent_terminal_monotonic=parent_terminal_monotonic,
        )
    if (
        killed_status is None
        and parent_terminal_monotonic >= float(config["deadline"])
    ):
        killed_status = "deadline_exceeded"
    if (
        killed_status is None
        and aggregate_required
        and not aggregate_terminal.get("readable")
    ):
        killed_status = "cgroup_aggregate_metrics_unreadable"
    if killed_status is not None:
        reason_by_status = {
            "rss_cap_exceeded": "parent_aggregate_stoploss_exceeded",
            "deadline_exceeded": (
                "parent_terminal_gate_rejected_late_completion"
            ),
            "cgroup_aggregate_metrics_unreadable": (
                "aggregate_metrics_became_unreadable"
            ),
        }
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status=killed_status,
            reason=reason_by_status[killed_status],
            observed_peak=observed_peak or None,
            returncode=returncode,
            stderr=stderr,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_terminal,
            parent_terminal_monotonic=parent_terminal_monotonic,
        )
    lines = [line for line in stdout.splitlines() if line.strip()]
    if returncode != 0 or len(lines) != 1:
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status="child_abnormal_exit",
            reason=(
                f"worker_exit_{returncode}"
                if returncode != 0
                else "worker_did_not_emit_exactly_one_json_document"
            ),
            observed_peak=observed_peak or None,
            returncode=returncode,
            stderr=stderr,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_terminal,
            parent_terminal_monotonic=parent_terminal_monotonic,
        )
    try:
        child = json.loads(lines[0])
    except (TypeError, json.JSONDecodeError):
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status="child_abnormal_output",
            reason="worker_output_is_not_json",
            observed_peak=observed_peak or None,
            returncode=returncode,
            stderr=stderr,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_terminal,
            parent_terminal_monotonic=parent_terminal_monotonic,
        )
    if not isinstance(child, dict) or not verify_diagnostic_checksum(child):
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status="child_diagnostic_checksum_invalid",
            reason="worker_diagnostic_checksum_failed",
            observed_peak=observed_peak or None,
            returncode=returncode,
            stderr=stderr,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_terminal,
            parent_terminal_monotonic=parent_terminal_monotonic,
        )
    contract_error = _validate_child_diagnostic_contract(
        child,
        config=config,
        estimates=estimates,
        preflight=preflight,
    )
    if contract_error is not None:
        return _abnormal_child_diagnostic(
            config,
            estimates,
            preflight,
            status="child_diagnostic_schema_invalid",
            reason=contract_error,
            observed_peak=observed_peak or None,
            returncode=returncode,
            stderr=stderr,
            aggregate_start=aggregate_start,
            aggregate_terminal=aggregate_terminal,
            parent_terminal_monotonic=parent_terminal_monotonic,
        )
    worker_leaf = child.get("cgroup_leaf_path")
    worker_terminal_ok = False
    try:
        worker_terminal = float.fromhex(
            str(child.get("worker_terminal_monotonic_hex"))
        )
        worker_terminal_ok = bool(
            child.get("worker_terminal_deadline_respected") is True
            and worker_terminal < float(config["deadline"])
        )
    except (TypeError, ValueError, OverflowError):
        worker_terminal = None
    child["parent_observed_peak_rss_bytes"] = observed_peak or None
    child["parent_observed_cgroup_aggregate_current_peak_bytes"] = (
        observed_aggregate_current_peak or None
    )
    child["parent_terminal_monotonic_hex"] = (
        parent_terminal_monotonic.hex()
    )
    child["worker_exit_code"] = returncode
    child["worker_signal"] = None
    child["worker_cgroup_leaf_path"] = worker_leaf
    _bind_preflight(child, preflight)
    process_peak = child.get("process_peak_rss_bytes")
    combined_peak = max(
        int(process_peak or 0), int(observed_peak or 0)
    )
    child_baseline = child.get("child_process_baseline_current_rss_bytes")
    if child_baseline is None:
        child_baseline = child.get("baseline_current_rss_bytes")
    if combined_peak:
        child["child_process_peak_rss_bytes"] = combined_peak
        child["child_process_peak_delta_bytes"] = (
            max(0, combined_peak - int(child_baseline))
            if child_baseline is not None
            else None
        )
    if aggregate_required:
        aggregate_peak = aggregate_terminal.get("peak_bytes")
        child.update(
            cgroup_aggregate_memory_current_start_bytes=(
                aggregate_start.get("current_bytes")
            ),
            cgroup_aggregate_memory_current_terminal_bytes=(
                aggregate_terminal.get("current_bytes")
            ),
            cgroup_aggregate_memory_peak_terminal_bytes=aggregate_peak,
            cgroup_aggregate_metrics_readable=bool(
                aggregate_terminal.get("readable")
            ),
            peak_rss_bytes=None,
            stoploss_peak_bytes=aggregate_peak,
            stoploss_peak_source=(
                "cgroup_v2_leaf_aggregate_memory_peak"
            ),
            rss_cap_respected=bool(
                aggregate_terminal.get("readable")
                and aggregate_peak <= effective_limit
            ),
            generic_rss_fields_scope="unset_in_bounded_v2_mode",
            allowed_rss_increment_bytes=None,
            baseline_current_rss_bytes=None,
            current_rss_bytes=None,
            post_trim_current_rss_bytes=None,
            current_delta_from_baseline_bytes=None,
            post_trim_delta_from_baseline_bytes=None,
            peak_delta_from_baseline_bytes=None,
            cgroup_aggregate_peak_delta_from_after_frame_bytes=None,
        )
    else:
        child["peak_rss_bytes"] = combined_peak or None
        child["stoploss_peak_bytes"] = combined_peak or None
        child["stoploss_peak_source"] = (
            "combined_process_hwm_and_parent_poll_diagnostic_only"
        )
        child["rss_cap_respected"] = bool(
            combined_peak and combined_peak <= effective_limit
        )
        child["generic_rss_fields_scope"] = (
            "child_process_diagnostic_only"
        )
        child["peak_delta_from_baseline_bytes"] = child.get(
            "child_process_peak_delta_bytes"
        )
    if child.get("status") == "ok" and not worker_terminal_ok:
        child["status"] = "deadline_exceeded"
        child["reason"] = "worker_terminal_deadline_contract_failed"
    if (
        child.get("status") == "ok"
        and aggregate_required
        and worker_leaf != leaf
    ):
        child["status"] = "candidate_contract_failed"
        child["reason"] = "worker_cgroup_leaf_binding_mismatch"
    if (
        child.get("status") == "ok"
        and aggregate_required
        and (
            child.get("cgroup_aggregate_memory_current_after_frame_bytes")
            is None
            or child.get("cgroup_aggregate_memory_current_after_cg_bytes")
            is None
        )
    ):
        child["status"] = "candidate_contract_failed"
        child["reason"] = "worker_aggregate_stage_metrics_missing"
    if child.get("status") == "ok" and not child["rss_cap_respected"]:
        child["status"] = "rss_cap_exceeded"
        child["reason"] = (
            "aggregate_peak_exceeded_effective_rss_limit"
            if aggregate_required
            else "combined_parent_or_child_peak_exceeded_limit"
        )
    return _seal_diagnostic(child)


def run_split_cg_memory_sentinel(
    *,
    n_cont: int,
    n_bin: int,
    n_upper: int,
    n_eq: int,
    constraint_nnz: int,
    selected_upper_rows: int,
    deadline: float,
    absolute_rss_cap_bytes: int = HARD_ABSOLUTE_RSS_CAP_BYTES,
    rss_reserve_bytes: int = MINIMUM_RSS_RESERVE_BYTES,
    scan_chunk_rows: int = DEFAULT_SCAN_CHUNK_ROWS,
    execute_large_profile: bool = False,
    profile_name: Optional[str] = None,
) -> dict[str, Any]:
    """Run one synthetic sentinel in an independently monitored child.

    ``deadline`` is an absolute ``time.monotonic()`` value shared by parent
    and child.  Every return path is a checksummed, non-authoritative JSON
    dictionary, including validation, preflight, timeout, cap, and child
    failure paths.
    """

    raw = {
        "n_cont": n_cont,
        "n_bin": n_bin,
        "n_upper": n_upper,
        "n_eq": n_eq,
        "constraint_nnz": constraint_nnz,
        "selected_upper_rows": selected_upper_rows,
        "deadline": deadline,
        "absolute_rss_cap_bytes": absolute_rss_cap_bytes,
        "rss_reserve_bytes": rss_reserve_bytes,
        "scan_chunk_rows": scan_chunk_rows,
        "execute_large_profile": execute_large_profile,
        "profile_name": profile_name,
    }
    try:
        config, estimates = _validate_config(raw)
    except Exception as exc:
        diagnostic = _base_diagnostic(_safe_config_summary(raw))
        diagnostic.update(
            status="config_rejected", reason=_safe_exception_text(exc)
        )
        return _seal_diagnostic(diagnostic)
    preflight, reason = _preflight(config, estimates)
    if reason is not None:
        diagnostic = _base_diagnostic(config, estimates)
        diagnostic.update(
            status="preflight_rejected",
            reason=reason,
            effective_rss_limit_bytes=preflight[
                "effective_rss_limit_bytes"
            ],
            candidate_return_status="not_started",
            native_model_clear_status="not_created",
        )
        _bind_preflight(diagnostic, preflight)
        return _seal_diagnostic(diagnostic)
    return _run_worker_process(config, estimates, preflight)


def run_fixed_profile(
    name: str = PROFILE_NAME,
    *,
    selected_upper_rows: int = DEFAULT_SELECTED_UPPER_ROWS,
    deadline: float,
    absolute_rss_cap_bytes: int = HARD_ABSOLUTE_RSS_CAP_BYTES,
    rss_reserve_bytes: int = MINIMUM_RSS_RESERVE_BYTES,
    scan_chunk_rows: int = DEFAULT_SCAN_CHUNK_ROWS,
    execute_large_profile: bool = False,
) -> dict[str, Any]:
    """Run (or, by default, refuse to run) the named fixed profile."""

    profile = get_fixed_profile(
        name, selected_upper_rows=selected_upper_rows
    )
    synthetic = profile["synthetic"]
    return run_split_cg_memory_sentinel(
        n_cont=synthetic["n_cont"],
        n_bin=synthetic["n_bin"],
        n_upper=synthetic["n_upper"],
        n_eq=synthetic["n_eq"],
        constraint_nnz=synthetic["constraint_nnz"],
        selected_upper_rows=synthetic["selected_upper_rows"],
        deadline=deadline,
        absolute_rss_cap_bytes=absolute_rss_cap_bytes,
        rss_reserve_bytes=rss_reserve_bytes,
        scan_chunk_rows=scan_chunk_rows,
        execute_large_profile=execute_large_profile,
        profile_name=name,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only split-CG synthetic RSS sentinel"
    )
    parser.add_argument("--profile", choices=(PROFILE_NAME,))
    parser.add_argument("--n-cont", type=int)
    parser.add_argument("--n-bin", type=int)
    parser.add_argument("--n-upper", type=int)
    parser.add_argument("--n-eq", type=int)
    parser.add_argument("--constraint-nnz", type=int)
    parser.add_argument("--selected-upper-rows", type=int)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument(
        "--absolute-rss-cap-bytes",
        type=int,
        default=HARD_ABSOLUTE_RSS_CAP_BYTES,
    )
    parser.add_argument(
        "--rss-reserve-bytes",
        type=int,
        default=MINIMUM_RSS_RESERVE_BYTES,
    )
    parser.add_argument(
        "--scan-chunk-rows", type=int, default=DEFAULT_SCAN_CHUNK_ROWS
    )
    parser.add_argument("--execute-large-profile", action="store_true")
    parser.add_argument("--_worker-config-json", help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker_config_json is not None:
        try:
            config = json.loads(args._worker_config_json)
            diagnostic = _execute_worker(config)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            diagnostic = _seal_diagnostic(
                {
                    **_base_diagnostic({}),
                    "status": "worker_bootstrap_error",
                    "reason": f"{type(exc).__name__}:{str(exc)[:512]}",
                }
            )
        print(json.dumps(diagnostic, sort_keys=True, separators=(",", ":")))
        return 0

    custom_values = (
        args.n_cont,
        args.n_bin,
        args.n_upper,
        args.n_eq,
        args.constraint_nnz,
    )
    custom_requested = any(value is not None for value in custom_values)
    deadline_seconds = args.deadline_seconds
    if deadline_seconds is None:
        deadline_seconds = 600.0 if not custom_requested else 15.0
    deadline = time.monotonic() + float(deadline_seconds)
    if not custom_requested:
        diagnostic = run_fixed_profile(
            args.profile or PROFILE_NAME,
            selected_upper_rows=(
                args.selected_upper_rows
                if args.selected_upper_rows is not None
                else DEFAULT_SELECTED_UPPER_ROWS
            ),
            deadline=deadline,
            absolute_rss_cap_bytes=args.absolute_rss_cap_bytes,
            rss_reserve_bytes=args.rss_reserve_bytes,
            scan_chunk_rows=args.scan_chunk_rows,
            execute_large_profile=args.execute_large_profile,
        )
    elif not all(value is not None for value in custom_values):
        diagnostic = _seal_diagnostic(
            {
                **_base_diagnostic({"deadline": deadline}),
                "status": "config_rejected",
                "reason": "custom_topology_requires_all_dimension_and_nnz_arguments",
            }
        )
    else:
        diagnostic = run_split_cg_memory_sentinel(
            n_cont=args.n_cont,
            n_bin=args.n_bin,
            n_upper=args.n_upper,
            n_eq=args.n_eq,
            constraint_nnz=args.constraint_nnz,
            selected_upper_rows=(
                args.selected_upper_rows
                if args.selected_upper_rows is not None
                else min(DEFAULT_SELECTED_UPPER_ROWS, args.n_upper)
            ),
            deadline=deadline,
            absolute_rss_cap_bytes=args.absolute_rss_cap_bytes,
            rss_reserve_bytes=args.rss_reserve_bytes,
            scan_chunk_rows=args.scan_chunk_rows,
            execute_large_profile=args.execute_large_profile,
        )
    print(json.dumps(diagnostic, sort_keys=True, separators=(",", ":")))
    return 0 if diagnostic.get("status") == "ok" else 2


__all__ = [
    "CIFAR100_MEDIUM_IID2_V1",
    "DEFAULT_SELECTED_UPPER_ROWS",
    "HARD_ABSOLUTE_RSS_CAP_BYTES",
    "MINIMUM_RSS_RESERVE_BYTES",
    "PROFILE_NAME",
    "estimate_topology_resources",
    "get_fixed_profile",
    "run_fixed_profile",
    "run_split_cg_memory_sentinel",
    "verify_diagnostic_checksum",
]


if __name__ == "__main__":
    raise SystemExit(main())
