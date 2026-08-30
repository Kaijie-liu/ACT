#!/usr/bin/env python3
"""Fail-closed memory sentinel for the K3 pair and scheduled stages.

This module is deliberately isolated from verifier and benchmark entrypoints.
It constructs a deterministic *synthetic* ``OperatorHZBuild`` with four exact
ReLU binaries, runs the real 12-query K3 pair candidate, and then runs the real
shared-context eight-pattern scheduled conditional-bound producer.  It never
issues or materializes a fresh HZ.

The fixed CIFAR-100-sized recipe is disabled by default.  Large execution
requires an explicit consent bit, a complete cgroup-v2 memory-controller walk,
a finite leaf ``memory.max`` no larger than 2.5 GiB, and at least 64 MiB of
reserved headroom.  The owned-array payload reported here is measured from
NumPy/SciPy buffers; it is not an allocator or native-solver upper bound.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import hmac
import io
import itertools
import json
import math
import numbers
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp


# Package initialization currently prints environment notices.  A worker's
# stdout is a strict one-document JSON protocol, so contain unrelated notices.
with contextlib.redirect_stdout(io.StringIO()):
    try:
        from act.back_end.hybridz_tf.adaptive_phase_forest import RivalSpec
        from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
            derive_operator_exact_relu_property_phase_literals,
        )
        from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_objective_bounds as _bounds,
        )
        from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
            build_k3_pair_first_schedule,
        )
        from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
            OperatorPhaseConditionedScheduledStop,
            OperatorPhaseConditionedScheduledStopPolicy,
            build_scheduled_complete_operator_phase_conditioned_objective_bounds,
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure,
        )
        from act.back_end.hybridz_tf.operator_phase_conditioned_pair_infeasibility import (
            PairLocalCaps,
            run_phase_conditioned_pair_infeasibility_candidate,
        )
        from act.back_end.solver.solver_hz import SparseHZono
    except ModuleNotFoundError:  # Direct execution from an uninstalled checkout.
        _REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
        if str(_REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPOSITORY_ROOT))
        from act.back_end.hybridz_tf.adaptive_phase_forest import RivalSpec
        from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
            derive_operator_exact_relu_property_phase_literals,
        )
        from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_objective_bounds as _bounds,
        )
        from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
            build_k3_pair_first_schedule,
        )
        from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
            OperatorPhaseConditionedScheduledStop,
            OperatorPhaseConditionedScheduledStopPolicy,
            build_scheduled_complete_operator_phase_conditioned_objective_bounds,
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure,
        )
        from act.back_end.hybridz_tf.operator_phase_conditioned_pair_infeasibility import (
            PairLocalCaps,
            run_phase_conditioned_pair_infeasibility_candidate,
        )
        from act.back_end.solver.solver_hz import SparseHZono


SCHEMA = "act.hybridz.pcoh_k3_pair_scheduled_memory_sentinel.v1"
PROFILE_NAME = "cifar100_medium_iid2_k3_pair_scheduled_v1"
HARD_ABSOLUTE_RSS_CAP_BYTES = 5 * (1 << 29)  # Exactly 2.5 GiB.
MINIMUM_RSS_RESERVE_BYTES = 64 * (1 << 20)
MINIMUM_LARGE_DEADLINE_SECONDS = 120.0
POLL_SECONDS = 0.005
MAX_WORKER_OUTPUT_BYTES = 1 << 20
LARGE_NNZ_THRESHOLD = 1_000_000
_INT32_MAX = int(np.iinfo(np.int32).max)
_MODES = frozenset({"complete", "early_stop"})
_CHECKPOINT_KEYS = frozenset(
    {
        "stage",
        "pattern",
        "sampled_monotonic_hex",
        "process_current_rss_bytes",
        "process_peak_rss_bytes",
        "cgroup_v2_leaf",
        "cgroup_current_bytes",
        "cgroup_peak_bytes",
        "cgroup_leaf_max_bytes",
        "cgroup_effective_max_bytes",
        "cgroup_effective_headroom_bytes",
        "cgroup_boundary_complete",
        "cgroup_error",
    }
)


_PROFILE_DATA = {
    "name": PROFILE_NAME,
    "topology": {
        "n_cont": 52657,
        "n_bin": 4,
        "n_upper": 98974,
        "n_eq": 0,
        "constraint_nnz": 10498232,
    },
    "k3_recipe": {
        "selected_stable_bits": 3,
        "signed_pair_queries": 12,
        "scheduled_patterns": 8,
        "exact_relu_rows": 12,
        "fresh_issue_called": False,
        "fresh_materialization_called": False,
        "allocator_increment_bound_bytes": None,
        "allocator_increment_bound_available": False,
    },
}
CIFAR100_MEDIUM_IID2_K3_PAIR_SCHEDULED_V1 = MappingProxyType(
    copy.deepcopy(_PROFILE_DATA)
)


_LIMITATIONS = (
    "synthetic_operator_hz_matches_fixed_dimensions_and_total_constraint_nnz_"
    "but_not_real_coefficients_row_degrees_or_network_provenance",
    "filler_upper_rows_are_deterministic_cube_redundant_rows_and_are_not_a_"
    "model_of_the_real_parent_constraint_distribution",
    "owned_numeric_payload_bytes_counts_numpy_and_scipy_buffers_only_and_is_"
    "not_an_allocator_or_native_solver_upper_bound",
    "python_fraction_highs_and_allocator_transients_have_no_predeclared_byte_"
    "upper_bound_in_this_sentinel",
    "cgroup_memory_peak_is_a_leaf_service_lifetime_high_water_mark_and_is_not_"
    "reset_by_this_sentinel",
    "cgroup_memory_current_includes_every_process_in_the_leaf_not_only_the_"
    "worker",
    "the_parent_5ms_stoploss_can_miss_short_spikes_but_the_required_large_"
    "profile_memory_max_is_a_kernel_hard_limit",
    "trusted_no_external_cgroup_migration_required_because_a_delegated_actor_"
    "could_move_the_worker_away_and_back_between_5ms_pid_membership_samples",
    "per_pattern_samples_wrap_the_existing_internal_pattern_builder_only_for_"
    "observation_and_restore_it_in_finally",
    "diagnostic_sha256_detects_payload_changes_but_is_not_authentication_or_"
    "proof_authority",
    "this_is_not_a_real_network_benchmark_and_cannot_calibrate_real_accuracy_"
    "runtime_or_memory",
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _seal(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(diagnostic))
    result.pop("diagnostic_sha256", None)
    result["diagnostic_sha256"] = _canonical_sha256(result)
    return result


def verify_diagnostic_checksum(diagnostic: Mapping[str, Any]) -> bool:
    try:
        if not isinstance(diagnostic, Mapping):
            return False
        expected = diagnostic.get("diagnostic_sha256")
        if (
            type(expected) is not str
            or len(expected) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected
            )
        ):
            return False
        body = copy.deepcopy(dict(diagnostic))
        body.pop("diagnostic_sha256", None)
        actual = _canonical_sha256(body)
        return hmac.compare_digest(expected, actual)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return False


def get_fixed_profile(name: str = PROFILE_NAME) -> dict[str, Any]:
    if type(name) is not str or name != PROFILE_NAME:
        raise ValueError("unknown_k3_pair_scheduled_memory_profile")
    return copy.deepcopy(_PROFILE_DATA)


def _strict_int(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Integral
    ):
        raise ValueError(f"{name}_must_be_builtin_integer")
    result = int(value)
    if result < lower or result > upper:
        raise ValueError(f"{name}_outside_supported_range")
    return result


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
    value = _read_status_bytes(f"/proc/{target}/status", "VmRSS")
    if value is not None:
        return value
    try:
        with open(f"/proc/{target}/statm", "r", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


def _peak_rss_bytes() -> Optional[int]:
    value = _read_status_bytes("/proc/self/status", "VmHWM")
    if value is not None:
        return value
    try:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    return raw if sys.platform == "darwin" else raw * 1024


def _read_mem_available_bytes() -> Optional[int]:
    return _read_status_bytes("/proc/meminfo", "MemAvailable")


def _empty_cgroup(error: Optional[str] = None) -> dict[str, Any]:
    return {
        "detected": False,
        "version": None,
        "leaf": None,
        "leaf_current_bytes": None,
        "leaf_peak_bytes": None,
        "leaf_max_bytes": None,
        "effective_max_bytes": None,
        "effective_headroom_bytes": None,
        "ancestor_limits": [],
        "boundary_complete": False,
        "error": error,
    }


def _read_cgroup_v2(pid: Optional[int] = None) -> dict[str, Any]:
    """Read every cgroup-v2 memory ancestor from the leaf to mount root."""

    if pid is None:
        membership_path = Path("/proc/self/cgroup")
    elif type(pid) is not int or pid <= 0:
        return _empty_cgroup("cgroup_pid_invalid")
    else:
        membership_path = Path(f"/proc/{pid}/cgroup")
    try:
        membership = membership_path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        return _empty_cgroup(
            f"membership_read_failed:{type(exc).__name__}:"
            f"{membership_path}"
        )
    relative = None
    for line in membership:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            relative = fields[2]
            break
    if relative is None:
        return _empty_cgroup("cgroup_v2_membership_not_found")
    root = Path("/sys/fs/cgroup").resolve()
    leaf = (root / relative.lstrip("/")).resolve()
    try:
        leaf.relative_to(root)
    except ValueError:
        return {
            **_empty_cgroup("cgroup_leaf_escapes_mount"),
            "version": 2,
            "leaf": str(leaf),
        }
    levels: list[dict[str, Any]] = []
    cursor = leaf
    while True:
        maximum_path = cursor / "memory.max"
        current_path = cursor / "memory.current"
        if not maximum_path.exists() or not current_path.exists():
            return {
                **_empty_cgroup(f"memory_controller_gap:{cursor}"),
                "version": 2,
                "leaf": str(leaf),
                "ancestor_limits": levels,
            }
        try:
            raw_max = maximum_path.read_text(encoding="ascii").strip()
            current = int(current_path.read_text(encoding="ascii").strip())
            maximum = None if raw_max == "max" else int(raw_max)
        except (OSError, ValueError) as exc:
            return {
                **_empty_cgroup(
                    f"memory_controller_read_failed:{type(exc).__name__}"
                ),
                "version": 2,
                "leaf": str(leaf),
                "ancestor_limits": levels,
            }
        if current < 0 or (maximum is not None and maximum <= 0):
            return {
                **_empty_cgroup("memory_controller_value_invalid"),
                "version": 2,
                "leaf": str(leaf),
                "ancestor_limits": levels,
            }
        levels.append(
            {
                "path": str(cursor),
                "current_bytes": current,
                "max_bytes": maximum,
                "headroom_bytes": (
                    None if maximum is None else max(0, maximum - current)
                ),
            }
        )
        if cursor == root:
            break
        parent = cursor.parent
        if parent == cursor:
            return {
                **_empty_cgroup("cgroup_ancestor_walk_escaped_mount"),
                "version": 2,
                "leaf": str(leaf),
                "ancestor_limits": levels,
            }
        cursor = parent
    peak = None
    peak_error = None
    try:
        peak = int((leaf / "memory.peak").read_text(encoding="ascii").strip())
        if peak < int(levels[0]["current_bytes"]):
            peak_error = "leaf_memory_peak_invalid"
            peak = None
    except (OSError, ValueError) as exc:
        peak_error = f"leaf_memory_peak_read_failed:{type(exc).__name__}"
    finite_maxima = [
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
        "detected": True,
        "version": 2,
        "leaf": str(leaf),
        "leaf_current_bytes": levels[0]["current_bytes"],
        "leaf_peak_bytes": peak,
        "leaf_max_bytes": levels[0]["max_bytes"],
        "effective_max_bytes": min(finite_maxima) if finite_maxima else None,
        "effective_headroom_bytes": (
            min(finite_headrooms) if finite_headrooms else None
        ),
        "ancestor_limits": levels,
        "boundary_complete": True,
        "error": peak_error,
    }


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _derive_cgroup_contract(
    cgroup: Any,
    *,
    require_peak: bool,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Strictly replay a cgroup snapshot from its complete ancestor records."""

    if not isinstance(cgroup, Mapping):
        return None, "cgroup_not_mapping"
    if (
        cgroup.get("detected") is not True
        or type(cgroup.get("version")) is not int
        or cgroup.get("version") != 2
    ):
        return None, "cgroup_v2_not_detected"
    if cgroup.get("boundary_complete") is not True:
        return None, "cgroup_boundary_not_complete"
    if cgroup.get("error") is not None:
        return None, "cgroup_snapshot_has_error"
    leaf = cgroup.get("leaf")
    levels = cgroup.get("ancestor_limits")
    if type(leaf) is not str or not leaf.startswith("/sys/fs/cgroup"):
        return None, "cgroup_leaf_invalid"
    if type(levels) is not list or not levels:
        return None, "cgroup_ancestor_records_missing"
    root = Path("/sys/fs/cgroup")
    expected_path = Path(leaf)
    if str(expected_path) != leaf or ".." in expected_path.parts:
        return None, "cgroup_leaf_not_canonical"
    maxima: list[int] = []
    headrooms: list[int] = []
    for index, level in enumerate(levels):
        if type(level) is not dict or set(level) != {
            "path",
            "current_bytes",
            "max_bytes",
            "headroom_bytes",
        }:
            return None, f"cgroup_ancestor_record_invalid:{index}"
        if level.get("path") != str(expected_path):
            return None, f"cgroup_ancestor_path_chain_invalid:{index}"
        current = level.get("current_bytes")
        maximum = level.get("max_bytes")
        headroom = level.get("headroom_bytes")
        if not _is_nonnegative_int(current):
            return None, f"cgroup_ancestor_current_invalid:{index}"
        if maximum is None:
            if headroom is not None:
                return None, f"cgroup_unlimited_headroom_not_null:{index}"
        else:
            if type(maximum) is not int or maximum <= 0:
                return None, f"cgroup_ancestor_max_invalid:{index}"
            expected_headroom = max(0, maximum - current)
            if type(headroom) is not int or headroom != expected_headroom:
                return None, f"cgroup_ancestor_headroom_invalid:{index}"
            maxima.append(maximum)
            headrooms.append(expected_headroom)
        if expected_path == root:
            if index != len(levels) - 1:
                return None, "cgroup_records_continue_above_root"
        else:
            expected_path = expected_path.parent
    if expected_path != root or levels[-1]["path"] != str(root):
        return None, "cgroup_ancestor_records_do_not_reach_mount_root"
    if not maxima:
        return None, "cgroup_has_no_finite_ancestor_limit"
    first = levels[0]
    if (
        type(cgroup.get("leaf_current_bytes")) is not int
        or cgroup.get("leaf_current_bytes") != first["current_bytes"]
    ):
        return None, "cgroup_leaf_current_summary_mismatch"
    if (
        type(cgroup.get("leaf_max_bytes")) is not int
        or cgroup.get("leaf_max_bytes") != first["max_bytes"]
    ):
        return None, "cgroup_leaf_max_summary_mismatch"
    if first["max_bytes"] is None:
        return None, "cgroup_leaf_memory_max_not_finite"
    effective_max = min(maxima)
    effective_headroom = min(headrooms)
    if (
        type(cgroup.get("effective_max_bytes")) is not int
        or cgroup.get("effective_max_bytes") != effective_max
    ):
        return None, "cgroup_effective_max_summary_mismatch"
    if (
        type(cgroup.get("effective_headroom_bytes")) is not int
        or cgroup.get("effective_headroom_bytes") != effective_headroom
    ):
        return None, "cgroup_effective_headroom_summary_mismatch"
    peak = cgroup.get("leaf_peak_bytes")
    if require_peak and (
        not _is_nonnegative_int(peak) or peak < first["current_bytes"]
    ):
        return None, "cgroup_leaf_peak_invalid"
    return {
        "leaf": leaf,
        "leaf_current_bytes": int(first["current_bytes"]),
        "leaf_peak_bytes": peak,
        "leaf_max_bytes": int(first["max_bytes"]),
        "effective_max_bytes": effective_max,
        "effective_headroom_bytes": effective_headroom,
    }, None


def estimate_owned_numeric_payload_bytes(
    *,
    n_cont: int,
    n_bin: int,
    n_upper: int,
    n_eq: int,
    constraint_nnz: int,
) -> dict[str, int]:
    """Return exact planned numeric-buffer bytes, never an allocator bound."""

    n_cont = _strict_int(n_cont, "n_cont", 8, _INT32_MAX)
    n_bin = _strict_int(n_bin, "n_bin", 4, 4)
    n_upper = _strict_int(n_upper, "n_upper", 12, _INT32_MAX - 1)
    n_eq = _strict_int(n_eq, "n_eq", 0, 0)
    constraint_nnz = _strict_int(
        constraint_nnz, "constraint_nnz", 28, _INT32_MAX
    )
    filler_nnz = constraint_nnz - 28
    if filler_nnz > (n_upper - 12) * n_cont:
        raise ValueError("constraint_nnz_exceeds_sentinel_recipe_capacity")
    auc_nnz = constraint_nnz - 8
    aub_nnz = 8
    csr_payload = (
        (auc_nnz + aub_nnz + 8) * (8 + 4)
        + 4 * (n_upper + 1)
        + 4 * (n_upper + 1)
        + 4 * (3 + 1)
        + 4 * (3 + 1)
        + 4 * (n_eq + 1)
        + 4 * (n_eq + 1)
    )
    dense_payload = (
        3 + n_eq + n_upper + n_cont + n_bin + 4
    ) * 8
    planning_payload = n_upper * (8 + 4 + 4)
    return {
        "exact_relu_constraint_nnz": 28,
        "filler_constraint_nnz": filler_nnz,
        "auc_nnz": auc_nnz,
        "aub_nnz": aub_nnz,
        "gc_nnz": 8,
        "planned_csr_buffer_bytes": int(csr_payload),
        "planned_dense_id_buffer_bytes": int(dense_payload),
        "planned_row_count_work_buffer_bytes": int(planning_payload),
        "planned_owned_numeric_payload_bytes": int(
            csr_payload + dense_payload
        ),
        "allocator_increment_bound_available": False,
    }


def _validate_config(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    config = dict(raw)
    for key, lower, upper in (
        ("n_cont", 8, _INT32_MAX),
        ("n_bin", 4, 4),
        ("n_upper", 12, _INT32_MAX - 1),
        ("n_eq", 0, 0),
        ("constraint_nnz", 28, _INT32_MAX),
        ("absolute_rss_cap_bytes", 1, HARD_ABSOLUTE_RSS_CAP_BYTES),
        ("rss_reserve_bytes", MINIMUM_RSS_RESERVE_BYTES, 1 << 63),
    ):
        config[key] = _strict_int(config.get(key), key, lower, upper)
    if config["rss_reserve_bytes"] >= config["absolute_rss_cap_bytes"]:
        raise ValueError("rss_reserve_leaves_no_usable_limit")
    estimates = estimate_owned_numeric_payload_bytes(
        n_cont=config["n_cont"],
        n_bin=config["n_bin"],
        n_upper=config["n_upper"],
        n_eq=config["n_eq"],
        constraint_nnz=config["constraint_nnz"],
    )
    mode = config.get("mode")
    if type(mode) is not str or mode not in _MODES:
        raise ValueError("mode_must_be_complete_or_early_stop")
    execute = config.get("execute_large_profile", False)
    if type(execute) is not bool:
        raise ValueError("execute_large_profile_requires_builtin_boolean")
    config["execute_large_profile"] = execute
    deadline = config.get("deadline")
    if isinstance(deadline, bool) or type(deadline) not in {int, float}:
        raise ValueError("deadline_must_be_finite_absolute_monotonic_time")
    deadline = float(deadline)
    if not math.isfinite(deadline):
        raise ValueError("deadline_must_be_finite_absolute_monotonic_time")
    config["deadline"] = deadline
    profile_name = config.get("profile_name")
    if profile_name not in (None, PROFILE_NAME):
        raise ValueError("unknown_k3_pair_scheduled_memory_profile")
    if profile_name == PROFILE_NAME:
        expected = _PROFILE_DATA["topology"]
        if any(config[key] != expected[key] for key in expected):
            raise ValueError("fixed_profile_topology_mismatch")
    config["large_topology"] = bool(
        profile_name == PROFILE_NAME
        or config["constraint_nnz"] >= LARGE_NNZ_THRESHOLD
        or config["n_cont"] > 4096
        or config["n_upper"] > 8192
    )
    return config, estimates


def _preflight(
    config: Mapping[str, Any], estimates: Mapping[str, int]
) -> tuple[dict[str, Any], Optional[str]]:
    now = time.monotonic()
    remaining = float(config["deadline"]) - now
    cgroup = _read_cgroup_v2()
    host_available = _read_mem_available_bytes()
    large = bool(config["large_topology"])
    cgroup_contract, cgroup_contract_error = _derive_cgroup_contract(
        cgroup, require_peak=True
    )
    effective_limit = (
        None
        if cgroup_contract is None
        else cgroup_contract["effective_max_bytes"]
    )
    effective_headroom = (
        None
        if cgroup_contract is None
        else cgroup_contract["effective_headroom_bytes"]
    )
    leaf_max = (
        None
        if cgroup_contract is None
        else cgroup_contract["leaf_max_bytes"]
    )
    numeric_payload = int(estimates["planned_owned_numeric_payload_bytes"])
    checks = {
        "sampled_monotonic_hex": now.hex(),
        "remaining_seconds_hex": max(0.0, remaining).hex(),
        "large_topology": large,
        "explicit_large_execution_consent": bool(
            config["execute_large_profile"]
        ),
        "minimum_large_deadline_seconds": MINIMUM_LARGE_DEADLINE_SECONDS,
        "hard_absolute_rss_cap_bytes": HARD_ABSOLUTE_RSS_CAP_BYTES,
        "requested_absolute_rss_cap_bytes": config[
            "absolute_rss_cap_bytes"
        ],
        "rss_reserve_bytes": config["rss_reserve_bytes"],
        "host_mem_available_bytes": host_available,
        "cgroup": cgroup,
        "cgroup_contract_error": cgroup_contract_error,
        "complete_cgroup_v2_ancestor_walk": cgroup_contract_error is None,
        "finite_leaf_memory_max": leaf_max is not None,
        "leaf_memory_max_at_or_below_2_5_gib": (
            leaf_max is not None
            and int(leaf_max) <= HARD_ABSOLUTE_RSS_CAP_BYTES
        ),
        "leaf_memory_max_at_or_below_requested_cap": (
            leaf_max is not None
            and int(leaf_max) <= int(config["absolute_rss_cap_bytes"])
        ),
        "effective_memory_max_bytes": effective_limit,
        "effective_headroom_bytes": effective_headroom,
        "planned_owned_numeric_payload_bytes": numeric_payload,
        "allocator_increment_bound_available": False,
        "numeric_payload_plus_reserve_fits_headroom": (
            effective_headroom is not None
            and numeric_payload + int(config["rss_reserve_bytes"])
            <= int(effective_headroom)
        ),
    }
    if remaining <= 0.0:
        return checks, "deadline_expired_before_worker_start"
    if large and not config["execute_large_profile"]:
        return checks, "large_profile_requires_explicit_execute_flag"
    if large and remaining < MINIMUM_LARGE_DEADLINE_SECONDS:
        return checks, "large_profile_deadline_preflight_failed"
    if large and not checks["complete_cgroup_v2_ancestor_walk"]:
        return checks, "large_profile_requires_complete_cgroup_v2_ancestors"
    if large and not checks["finite_leaf_memory_max"]:
        return checks, "large_profile_requires_finite_leaf_memory_max"
    if large and not checks["leaf_memory_max_at_or_below_2_5_gib"]:
        return checks, "large_profile_leaf_memory_max_exceeds_2_5_gib"
    if large and not checks["leaf_memory_max_at_or_below_requested_cap"]:
        return checks, "large_profile_leaf_memory_max_exceeds_requested_cap"
    if large and effective_limit is None:
        return checks, "large_profile_effective_memory_max_unavailable"
    if large and not checks["numeric_payload_plus_reserve_fits_headroom"]:
        return checks, "large_profile_numeric_payload_reserve_preflight_failed"
    return checks, None


def _base_diagnostic(
    config: Mapping[str, Any], estimates: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "not_started",
        "reason": None,
        "diagnostic_only": True,
        "candidate_only": True,
        "synthetic_topology": True,
        "real_network_instance": False,
        "production_ready": False,
        "proof_authority": False,
        "verdict_authority": False,
        "primal_feasibility_authority": False,
        "parent_binding_authority": False,
        "ground_truth_loaded": False,
        "full_parent_lp_called": False,
        "fresh_issue_called": False,
        "fresh_materialization_called": False,
        "partial_certificates_returned": False,
        "trusted_no_external_cgroup_migration_required": True,
        "profile_name": config.get("profile_name"),
        "mode": config.get("mode"),
        "large_topology": config.get("large_topology"),
        "topology": {
            key: config.get(key)
            for key in (
                "n_cont",
                "n_bin",
                "n_upper",
                "n_eq",
                "constraint_nnz",
            )
        },
        "recipe": {
            "selected_stable_bits": 3,
            "signed_pair_queries": 12,
            "scheduled_patterns": 8,
            "exact_relu_rows": 12,
            "filler_row_semantics": "cube_redundant_continuous_rows",
            "pair_api": (
                "run_phase_conditioned_pair_infeasibility_candidate"
            ),
            "scheduled_api": (
                "build_scheduled_complete_operator_phase_conditioned_"
                "objective_bounds"
            ),
            "complete_stop_policy": "empty",
            "early_stop_policy": "unconditional_schedule_index_0",
            "fresh_stage_excluded": True,
            "allocator_increment_bound_available": False,
            "parent_cgroup_membership_monitor": "/proc/<worker_pid>/cgroup",
            "trusted_no_external_cgroup_migration_required": True,
        },
        "resource_estimates": copy.deepcopy(dict(estimates)),
        "absolute_deadline_monotonic_hex": (
            float(config["deadline"]).hex()
            if isinstance(config.get("deadline"), (int, float))
            and math.isfinite(float(config["deadline"]))
            else None
        ),
        "absolute_rss_cap_bytes": config.get("absolute_rss_cap_bytes"),
        "rss_reserve_bytes": config.get("rss_reserve_bytes"),
        "preflight": None,
        "owned_numeric_payload_bytes": None,
        "source_semantic_digest": None,
        "selection_digest": None,
        "pair_bundle_sha256": None,
        "pair_query_count": None,
        "pair_models_closed": None,
        "certified_pair_conflicts": None,
        "evaluation_schedule": None,
        "scheduled_bundle_sha256": None,
        "scheduled_stop_record_sha256": None,
        "scheduled_patterns_completed": None,
        "scheduled_candidate_dual_accepted": None,
        "scheduled_local_lp_actual_calls": None,
        "conditional_checker_actual_calls": None,
        "scheduled_telemetry_sha256": None,
        "memory_checkpoints": [],
        "memory_trace_sha256": None,
        "parent_observed_child_peak_rss_bytes": None,
        "parent_observed_cgroup_current_peak_bytes": None,
        "worker_exit_code": None,
        "worker_signal": None,
        "worker_stderr": None,
        "worker_cleanup": None,
        "timings": {},
        "limitations": list(_LIMITATIONS),
    }


def _safe_rejection(raw: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    try:
        exception_text = str(exc)[:512]
    except BaseException:
        exception_text = "exception_text_unavailable"
    safe = {}
    for key, value in raw.items():
        try:
            safe_key = str(key)[:128]
        except BaseException:
            safe_key = "key_text_unavailable"
        if value is None or type(value) in {str, bool, int}:
            safe[safe_key] = value
        elif type(value) is float and math.isfinite(value):
            safe[safe_key] = value
        elif type(value) is float:
            safe[safe_key] = {
                "rejected_type": "builtins.float",
                "reason": "nonfinite_float",
            }
        else:
            safe[safe_key] = {
                "rejected_type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                )[:256]
            }
    return _seal(
        {
            "schema": SCHEMA,
            "status": "config_rejected",
            "reason": f"{type(exc).__name__}:{exception_text}",
            "diagnostic_only": True,
            "candidate_only": True,
            "proof_authority": False,
            "verdict_authority": False,
            "primal_feasibility_authority": False,
            "parent_binding_authority": False,
            "ground_truth_loaded": False,
            "full_parent_lp_called": False,
            "fresh_issue_called": False,
            "fresh_materialization_called": False,
            "partial_certificates_returned": False,
            "trusted_no_external_cgroup_migration_required": True,
            "rejected_config": safe,
            "limitations": list(_LIMITATIONS),
        }
    )


def _checkpoint(stage: str, pattern: Optional[Sequence[int]] = None) -> dict[str, Any]:
    cgroup = _read_cgroup_v2()
    return {
        "stage": stage,
        "pattern": None if pattern is None else [int(item) for item in pattern],
        "sampled_monotonic_hex": time.monotonic().hex(),
        "process_current_rss_bytes": _current_rss_bytes(),
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "cgroup_v2_leaf": cgroup.get("leaf"),
        "cgroup_current_bytes": cgroup.get("leaf_current_bytes"),
        "cgroup_peak_bytes": cgroup.get("leaf_peak_bytes"),
        "cgroup_leaf_max_bytes": cgroup.get("leaf_max_bytes"),
        "cgroup_effective_max_bytes": cgroup.get("effective_max_bytes"),
        "cgroup_effective_headroom_bytes": cgroup.get(
            "effective_headroom_bytes"
        ),
        "cgroup_boundary_complete": cgroup.get("boundary_complete"),
        "cgroup_error": cgroup.get("error"),
    }


def _owned_numeric_payload_bytes(build: OperatorHZBuild) -> int:
    hz = build.hz
    total = 0
    for matrix in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub):
        if matrix is not None:
            total += sum(
                int(array.nbytes)
                for array in (matrix.data, matrix.indices, matrix.indptr)
            )
    for array in (
        hz.c,
        hz.b,
        hz.ub,
        hz.col_ids,
        hz.bcol_ids,
        build.input_col_ids,
    ):
        if array is not None:
            total += int(np.asarray(array).nbytes)
    return total


def _build_synthetic_operator_hz(
    config: Mapping[str, Any], estimates: Mapping[str, int]
) -> OperatorHZBuild:
    """Build the frozen exact-row/filler recipe with no network authority."""

    n_cont = int(config["n_cont"])
    n_bin = int(config["n_bin"])
    n_upper = int(config["n_upper"])
    auc_nnz = int(estimates["auc_nnz"])
    filler_nnz = int(estimates["filler_constraint_nnz"])
    deadline = float(config["deadline"])

    row_counts = np.zeros(n_upper, dtype=np.int32)
    row_counts[:8] = 2
    row_counts[8:12] = 1
    filler_rows = n_upper - 12
    base, extra = divmod(filler_nnz, filler_rows)
    if base + (1 if extra else 0) > n_cont:
        raise ValueError("filler_row_width_exceeds_continuous_columns")
    row_counts[12:] = base
    if extra:
        row_counts[12 : 12 + extra] += 1
    indptr64 = np.empty(n_upper + 1, dtype=np.int64)
    indptr64[0] = 0
    np.cumsum(row_counts, dtype=np.int64, out=indptr64[1:])
    if int(indptr64[-1]) != auc_nnz or auc_nnz > _INT32_MAX:
        raise RuntimeError("auc_recipe_nnz_mismatch")
    auc_indptr = indptr64.astype(np.int32, copy=True)
    del indptr64
    auc_indices = np.empty(auc_nnz, dtype=np.int32)
    auc_data = np.ones(auc_nnz, dtype=np.float64)

    for bit in range(4):
        start = int(auc_indptr[bit])
        auc_indices[start : start + 2] = (bit, 4 + bit)
        auc_data[start : start + 2] = (1.0, -1.0)
        row = 4 + bit
        start = int(auc_indptr[row])
        auc_indices[start : start + 2] = (bit, 4 + bit)
        auc_data[start : start + 2] = (-1.0, 1.0)
        row = 8 + bit
        start = int(auc_indptr[row])
        auc_indices[start] = 4 + bit

    for row in range(12, n_upper):
        if (row & 1023) == 0 and time.monotonic() >= deadline:
            raise TimeoutError("deadline_expired_during_filler_csr_build")
        start = int(auc_indptr[row])
        stop = int(auc_indptr[row + 1])
        width = stop - start
        if width:
            first = (row * 104729 + 8191) % (n_cont - width + 1)
            auc_indices[start:stop] = np.arange(
                first, first + width, dtype=np.int32
            )
    Auc = sp.csr_matrix(
        (auc_data, auc_indices, auc_indptr),
        shape=(n_upper, n_cont),
        copy=False,
    )
    if Auc.has_canonical_format is not True or Auc.nnz != auc_nnz:
        raise RuntimeError("auc_recipe_not_canonical")

    aub_counts = np.zeros(n_upper, dtype=np.int32)
    aub_counts[4:12] = 1
    aub_indptr = np.empty(n_upper + 1, dtype=np.int32)
    aub_indptr[0] = 0
    np.cumsum(aub_counts, dtype=np.int64, out=aub_indptr[1:])
    aub_indices = np.arange(8, dtype=np.int32) % 4
    aub_data = np.concatenate(
        (np.ones(4, dtype=np.float64), -np.ones(4, dtype=np.float64))
    )
    Aub = sp.csr_matrix(
        (aub_data, aub_indices, aub_indptr),
        shape=(n_upper, n_bin),
        copy=False,
    )
    if Aub.has_canonical_format is not True or Aub.nnz != 8:
        raise RuntimeError("aub_recipe_not_canonical")

    ub = row_counts.astype(np.float64, copy=True)
    ub[:12] = 4.0
    gc_indptr = np.asarray((0, 0, 4, 8), dtype=np.int32)
    gc_indices = np.asarray(tuple(range(4, 8)) * 2, dtype=np.int32)
    gc_data = np.asarray((1.0,) * 4 + (0.5,) * 4, dtype=np.float64)
    Gc = sp.csr_matrix(
        (gc_data, gc_indices, gc_indptr), shape=(3, n_cont), copy=False
    )
    hz = SparseHZono(
        c=np.zeros(3, dtype=np.float64),
        Gc=Gc,
        Gb=sp.csr_matrix((3, n_bin), dtype=np.float64),
        Ac=sp.csr_matrix((0, n_cont), dtype=np.float64),
        Ab=sp.csr_matrix((0, n_bin), dtype=np.float64),
        b=np.empty(0, dtype=np.float64),
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        col_ids=np.arange(n_cont, dtype=np.int64),
        bcol_ids=np.arange(n_cont, n_cont + n_bin, dtype=np.int64),
    )
    tags = tuple(
        [f"relu_exact_lower:{bit}" for bit in range(4)]
        + [f"relu_exact_x_branch:{bit}" for bit in range(4)]
        + [f"relu_exact_zero_branch:{bit}" for bit in range(4)]
        + ["synthetic_memory_filler"] * (n_upper - 12)
    )
    setattr(hz, "_solver_constraint_row_tags", tags)
    build = OperatorHZBuild(
        hz=hz,
        input_col_ids=np.arange(4, dtype=np.int64),
        input_layer_id=0,
        output_layer_id=1,
        assert_layer_id=2,
        metadata={
            "schema": "act.hybridz.k3_memory_synthetic_operator_hz.v1",
            "synthetic_only": True,
            "proof_authority": False,
        },
        property_upper_output=False,
    )
    if hz.constraint_nnz != int(config["constraint_nnz"]):
        raise RuntimeError("synthetic_constraint_nnz_mismatch")
    return build


def _pair_caps() -> PairLocalCaps:
    return PairLocalCaps(
        max_stable_bits=3,
        max_signed_pair_queries=12,
        max_local_rows=6,
        max_local_nonzeros=200_000,
        max_source_terms=6,
        max_multiplier_bits=256,
        max_exact_bits=4096,
        max_exact_nonzeros=200_000,
    )


def _execute_worker(raw: Mapping[str, Any]) -> dict[str, Any]:
    worker_started = time.monotonic()
    try:
        config, estimates = _validate_config(raw)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _safe_rejection(raw, exc)
    diagnostic = _base_diagnostic(config, estimates)
    preflight, reason = _preflight(config, estimates)
    diagnostic["preflight"] = preflight
    if reason is not None:
        diagnostic.update(status="worker_preflight_rejected", reason=reason)
        return _seal(diagnostic)

    checkpoints: list[dict[str, Any]] = []
    original_pattern_builder = None
    hook_installed = False
    try:
        build_started = time.monotonic()
        build = _build_synthetic_operator_hz(config, estimates)
        build_seconds = time.monotonic() - build_started
        actual_payload = _owned_numeric_payload_bytes(build)
        if actual_payload != estimates["planned_owned_numeric_payload_bytes"]:
            raise RuntimeError("owned_numeric_payload_plan_mismatch")
        checkpoints.append(_checkpoint("baseline"))

        rivals = (
            RivalSpec(
                rival_id=10,
                objective=(-1.0, 1.0, 0.0),
                threshold=0.0,
                assert_digest="a" * 64,
            ),
            RivalSpec(
                rival_id=20,
                objective=(-1.0, 0.0, 1.0),
                threshold=0.0,
                assert_digest="b" * 64,
            ),
        )
        selection_timeout = 30.0 if config["large_topology"] else 5.0
        selection = derive_operator_exact_relu_property_phase_literals(
            build,
            rivals,
            max_rivals=2,
            max_binaries=4,
            max_work_items=5_000_000,
            timeout_seconds=selection_timeout,
        )
        stable_ids = tuple(
            mapping.stable_bcol_id for mapping in selection.mappings[:3]
        )
        if len(selection.mappings) != 4 or len(stable_ids) != 3:
            raise RuntimeError("synthetic_phase_selection_shape_mismatch")

        pair_started = time.monotonic()
        pair = run_phase_conditioned_pair_infeasibility_candidate(
            build,
            rivals,
            selection,
            stable_bit_ids=stable_ids,
            deadline=float(config["deadline"]),
            caps=_pair_caps(),
        )
        pair_seconds = time.monotonic() - pair_started
        if (
            len(pair.records) != 12
            or not all(record.model_closed for record in pair.records)
        ):
            raise RuntimeError("pair_stage_contract_mismatch")
        checkpoints.append(_checkpoint("after_pair"))
        schedule = build_k3_pair_first_schedule(
            pair, preferred_third_phase=1
        )
        if len(schedule.evaluation_schedule) != 8:
            raise RuntimeError("k3_schedule_pattern_count_mismatch")
        checkpoints.append(_checkpoint("pre_s"))

        original_pattern_builder = _bounds._build_bound_from_verified_context

        def observed_pattern_builder(*args: Any, **kwargs: Any) -> Any:
            certificate = original_pattern_builder(*args, **kwargs)
            pattern = kwargs.get("pattern")
            checkpoints.append(
                _checkpoint(
                    f"pattern_{len([item for item in checkpoints if item['stage'].startswith('pattern_')])}",
                    pattern,
                )
            )
            return certificate

        _bounds._build_bound_from_verified_context = observed_pattern_builder
        hook_installed = True
        policy = (
            OperatorPhaseConditionedScheduledStopPolicy()
            if config["mode"] == "complete"
            else OperatorPhaseConditionedScheduledStopPolicy(
                stop_after_pattern_indices=(0,)
            )
        )
        scheduled_started = time.monotonic()
        scheduled = None
        stop_record = None
        try:
            scheduled = (
                build_scheduled_complete_operator_phase_conditioned_objective_bounds(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    stable_bit_ids=stable_ids,
                    evaluation_schedule=schedule.evaluation_schedule,
                    deadline=float(config["deadline"]),
                    stop_policy=policy,
                    candidate_timeout_seconds=1.0,
                )
            )
        except OperatorPhaseConditionedScheduledStop as stop:
            stop_record = stop.record
        scheduled_seconds = time.monotonic() - scheduled_started
        _bounds._build_bound_from_verified_context = original_pattern_builder
        hook_installed = False

        if config["mode"] == "complete":
            if (
                scheduled is None
                or stop_record is not None
                or not verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                    scheduled
                )
                or len(scheduled.certificates) != 8
            ):
                raise RuntimeError("scheduled_complete_contract_mismatch")
            telemetry = scheduled.telemetry
            status = "ok_complete"
            scheduled_bundle_sha = scheduled.bundle_sha256
            stop_sha = None
        else:
            if (
                scheduled is not None
                or stop_record is None
                or stop_record.triggering_schedule_index != 0
                or stop_record.telemetry.get("patterns_completed") != 1
            ):
                raise RuntimeError("scheduled_early_stop_contract_mismatch")
            telemetry = stop_record.telemetry
            status = "ok_early_stop"
            scheduled_bundle_sha = None
            stop_sha = stop_record.record_sha256
        expected_pattern_samples = 8 if config["mode"] == "complete" else 1
        completed = telemetry.get("patterns_completed")
        accepted = telemetry.get("candidate_dual_accepted")
        scheduled_lp = telemetry.get("linprog_actual_calls")
        producer_checker = telemetry.get("split_checker_evaluations")
        telemetry_sha = telemetry.get("telemetry_sha256")
        if (
            type(completed) is not int
            or type(accepted) is not int
            or type(scheduled_lp) is not int
            or type(producer_checker) is not int
            or not 0 <= accepted <= completed
            or not 0 <= scheduled_lp <= completed
            or producer_checker != 1 + completed + accepted
            or producer_checker > 17
            or telemetry.get("actual_call_site_counters") is not True
            or type(telemetry_sha) is not str
            or len(telemetry_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in telemetry_sha
            )
        ):
            raise RuntimeError("scheduled_actual_counter_binding_failed")
        pattern_samples = [
            item for item in checkpoints if item["stage"].startswith("pattern_")
        ]
        if len(pattern_samples) != expected_pattern_samples:
            raise RuntimeError("per_pattern_memory_checkpoint_count_mismatch")
        checkpoints.append(_checkpoint("terminal"))
        diagnostic.update(
            status=status,
            reason=None,
            owned_numeric_payload_bytes=actual_payload,
            source_semantic_digest=selection.parent_semantic_digest,
            selection_digest=selection.selection_digest,
            pair_bundle_sha256=pair.bundle_sha256,
            pair_query_count=len(pair.records),
            pair_models_closed=all(record.model_closed for record in pair.records),
            certified_pair_conflicts=sum(
                record.status == "certified_conflict" for record in pair.records
            ),
            evaluation_schedule=[
                list(pattern) for pattern in schedule.evaluation_schedule
            ],
            scheduled_bundle_sha256=scheduled_bundle_sha,
            scheduled_stop_record_sha256=stop_sha,
            scheduled_patterns_completed=completed,
            scheduled_candidate_dual_accepted=accepted,
            scheduled_local_lp_actual_calls=scheduled_lp,
            conditional_checker_actual_calls=producer_checker,
            scheduled_telemetry_sha256=telemetry_sha,
            memory_checkpoints=checkpoints,
            memory_trace_sha256=_canonical_sha256(
                {"memory_checkpoints": checkpoints}
            ),
            timings={
                "source_build_seconds_hex": build_seconds.hex(),
                "pair_seconds_hex": pair_seconds.hex(),
                "scheduled_seconds_hex": scheduled_seconds.hex(),
                "worker_total_seconds_hex": (
                    time.monotonic() - worker_started
                ).hex(),
            },
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        diagnostic.update(
            status="worker_error",
            reason=_safe_exception_label(exc),
            memory_checkpoints=checkpoints,
            memory_trace_sha256=_canonical_sha256(
                {"memory_checkpoints": checkpoints}
            ),
            timings={
                "worker_total_seconds_hex": (
                    time.monotonic() - worker_started
                ).hex()
            },
        )
    finally:
        if hook_installed and original_pattern_builder is not None:
            _bounds._build_bound_from_verified_context = original_pattern_builder
    return _seal(diagnostic)


def _worker_command(config: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-config-json",
        json.dumps(config, sort_keys=True, separators=(",", ":")),
    ]


def _stop_process(process: subprocess.Popen[Any]) -> dict[str, Any]:
    """Attempt TERM/KILL and return an explicit, non-authoritative reap record."""

    warnings: list[str] = []
    record: dict[str, Any] = {
        "schema": "act.hybridz.k3_memory_worker_cleanup.v1",
        "termination_required": True,
        "initial_poll_exit_code": None,
        "sigterm_attempted": False,
        "sigterm_method": None,
        "sigkill_attempted": False,
        "sigkill_method": None,
        "reaped": False,
        "exit_code": None,
        "cleanup_error": None,
        "cleanup_warnings": warnings,
    }
    try:
        initial = process.poll()
    except BaseException as exc:
        initial = None
        warnings.append("initial_poll_failed:" + _safe_exception_label(exc))
    if type(initial) is int:
        record.update(
            termination_required=False,
            initial_poll_exit_code=initial,
            reaped=True,
            exit_code=initial,
        )
        return record
    if initial is not None:
        warnings.append("initial_poll_exit_code_invalid")

    record["sigterm_attempted"] = True
    try:
        os.killpg(process.pid, signal.SIGTERM)
        record["sigterm_method"] = "process_group"
    except BaseException as exc:
        warnings.append("sigterm_process_group_failed:" + _safe_exception_label(exc))
        try:
            process.terminate()
            record["sigterm_method"] = "process_terminate_fallback"
        except BaseException as fallback:
            warnings.append(
                "sigterm_fallback_failed:" + _safe_exception_label(fallback)
            )
    try:
        exit_code = process.wait(timeout=1.0)
        if type(exit_code) is int:
            record.update(reaped=True, exit_code=exit_code)
            return record
        warnings.append("wait_after_sigterm_exit_code_invalid")
    except subprocess.TimeoutExpired:
        warnings.append("wait_after_sigterm_timeout")
    except BaseException as exc:
        warnings.append("wait_after_sigterm_failed:" + _safe_exception_label(exc))

    record["sigkill_attempted"] = True
    try:
        os.killpg(process.pid, signal.SIGKILL)
        record["sigkill_method"] = "process_group"
    except BaseException as exc:
        warnings.append("sigkill_process_group_failed:" + _safe_exception_label(exc))
        try:
            process.kill()
            record["sigkill_method"] = "process_kill_fallback"
        except BaseException as fallback:
            warnings.append(
                "sigkill_fallback_failed:" + _safe_exception_label(fallback)
            )
    try:
        exit_code = process.wait(timeout=1.0)
        if type(exit_code) is int:
            record.update(reaped=True, exit_code=exit_code)
            return record
        warnings.append("final_wait_exit_code_invalid")
    except subprocess.TimeoutExpired:
        warnings.append("final_wait_after_sigkill_timeout")
    except BaseException as exc:
        warnings.append("final_wait_failed:" + _safe_exception_label(exc))
    record["cleanup_error"] = "worker_not_reaped_after_term_and_kill"
    return record


def _safe_exception_label(exc: BaseException) -> str:
    try:
        detail = str(exc)[:512]
    except BaseException:
        detail = "exception_text_unavailable"
    return f"{type(exc).__name__}:{detail}"


def _cleanup_process_noexcept(process: Any) -> dict[str, Any]:
    """Best-effort worker cleanup that never masks the triggering exception."""

    try:
        record = _stop_process(process)
        if type(record) is not dict or record.get("reaped") not in {
            True,
            False,
        }:
            raise RuntimeError("cleanup_record_invalid")
        return record
    except BaseException as exc:
        return {
            "schema": "act.hybridz.k3_memory_worker_cleanup.v1",
            "termination_required": True,
            "initial_poll_exit_code": None,
            "sigterm_attempted": False,
            "sigterm_method": None,
            "sigkill_attempted": False,
            "sigkill_method": None,
            "reaped": False,
            "exit_code": None,
            "cleanup_error": "cleanup_state_machine_failed:"
            + _safe_exception_label(exc),
            "cleanup_warnings": [],
        }


def _validate_checkpoint_shape(item: Any) -> Optional[str]:
    if type(item) is not dict:
        return "checkpoint_not_builtin_dict"
    if set(item) != _CHECKPOINT_KEYS:
        return "checkpoint_keyset_mismatch"
    if type(item["stage"]) is not str:
        return "checkpoint_stage_not_builtin_string"
    pattern = item["pattern"]
    if pattern is not None and (
        type(pattern) is not list
        or len(pattern) != 3
        or any(type(value) is not int or value not in {-1, 1} for value in pattern)
    ):
        return "checkpoint_pattern_invalid"
    timestamp = item["sampled_monotonic_hex"]
    if type(timestamp) is not str:
        return "checkpoint_sample_time_invalid"
    try:
        sampled = float.fromhex(timestamp)
    except (TypeError, ValueError, OverflowError):
        return "checkpoint_sample_time_invalid"
    if not math.isfinite(sampled) or sampled < 0.0:
        return "checkpoint_sample_time_invalid"
    for field in ("process_current_rss_bytes", "process_peak_rss_bytes"):
        if not _is_nonnegative_int(item[field]):
            return f"checkpoint_metric_invalid:{field}"
    if item["process_peak_rss_bytes"] < item["process_current_rss_bytes"]:
        return "checkpoint_process_peak_below_current"
    if item["cgroup_v2_leaf"] is not None and type(
        item["cgroup_v2_leaf"]
    ) is not str:
        return "checkpoint_cgroup_leaf_invalid"
    for field in (
        "cgroup_current_bytes",
        "cgroup_peak_bytes",
        "cgroup_leaf_max_bytes",
        "cgroup_effective_max_bytes",
        "cgroup_effective_headroom_bytes",
    ):
        if item[field] is not None and not _is_nonnegative_int(item[field]):
            return f"checkpoint_metric_invalid:{field}"
    if type(item["cgroup_boundary_complete"]) is not bool:
        return "checkpoint_cgroup_boundary_flag_invalid"
    if item["cgroup_error"] is not None and type(item["cgroup_error"]) is not str:
        return "checkpoint_cgroup_error_invalid"
    return None


def _validate_large_checkpoint_binding(
    item: Any,
    *,
    expected: Mapping[str, int | str],
) -> Optional[str]:
    shape_error = _validate_checkpoint_shape(item)
    if shape_error is not None:
        return shape_error
    for field in (
        "process_current_rss_bytes",
        "process_peak_rss_bytes",
        "cgroup_current_bytes",
        "cgroup_peak_bytes",
        "cgroup_effective_headroom_bytes",
    ):
        if not _is_nonnegative_int(item.get(field)):
            return f"checkpoint_metric_invalid:{field}"
    if item["process_peak_rss_bytes"] < item["process_current_rss_bytes"]:
        return "checkpoint_process_peak_below_current"
    if item["cgroup_peak_bytes"] < item["cgroup_current_bytes"]:
        return "checkpoint_cgroup_peak_below_current"
    if item.get("cgroup_v2_leaf") != expected["leaf"]:
        return "checkpoint_cgroup_leaf_binding_mismatch"
    for field, expected_key in (
        ("cgroup_leaf_max_bytes", "leaf_max_bytes"),
        ("cgroup_effective_max_bytes", "effective_max_bytes"),
    ):
        value = item.get(field)
        if type(value) is not int or value != expected[expected_key]:
            return f"checkpoint_cgroup_limit_binding_mismatch:{field}"
        if value <= 0 or value > HARD_ABSOLUTE_RSS_CAP_BYTES:
            return f"checkpoint_cgroup_limit_outside_hard_cap:{field}"
    if item["cgroup_current_bytes"] > item["cgroup_leaf_max_bytes"]:
        return "checkpoint_cgroup_current_above_leaf_max"
    if item.get("cgroup_boundary_complete") is not True:
        return "checkpoint_cgroup_boundary_incomplete"
    if item.get("cgroup_error") is not None:
        return "checkpoint_cgroup_error"
    return None


def _valid_success_contract(
    child: Mapping[str, Any],
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> Optional[str]:
    if child.get("schema") != SCHEMA:
        return "child_schema_mismatch"
    for field, expected in (
        ("diagnostic_only", True),
        ("candidate_only", True),
        ("proof_authority", False),
        ("verdict_authority", False),
        ("primal_feasibility_authority", False),
        ("parent_binding_authority", False),
        ("ground_truth_loaded", False),
        ("full_parent_lp_called", False),
        ("fresh_issue_called", False),
        ("fresh_materialization_called", False),
        ("partial_certificates_returned", False),
        ("trusted_no_external_cgroup_migration_required", True),
    ):
        if child.get(field) is not expected:
            return f"child_authority_contract_invalid:{field}"
    if child.get("mode") != config["mode"]:
        return "child_mode_binding_mismatch"
    if child.get("large_topology") is not config["large_topology"]:
        return "child_large_topology_binding_mismatch"
    if child.get("topology") != {
        key: config[key]
        for key in ("n_cont", "n_bin", "n_upper", "n_eq", "constraint_nnz")
    }:
        return "child_topology_binding_mismatch"
    expected_status = (
        "ok_complete" if config["mode"] == "complete" else "ok_early_stop"
    )
    if child.get("status") != expected_status:
        return None  # Preserve a sealed fail-closed worker diagnostic.
    checkpoints = child.get("memory_checkpoints")
    if type(checkpoints) is not list:
        return "child_memory_checkpoints_not_list"
    for index, item in enumerate(checkpoints):
        checkpoint_error = _validate_checkpoint_shape(item)
        if checkpoint_error is not None:
            return f"child_memory_checkpoint_invalid:{index}:{checkpoint_error}"
    expected_stages = ["baseline", "after_pair", "pre_s"] + [
        f"pattern_{index}"
        for index in range(8 if config["mode"] == "complete" else 1)
    ] + ["terminal"]
    if [item["stage"] for item in checkpoints] != expected_stages:
        return "child_memory_checkpoint_stage_mismatch"
    schedule = child.get("evaluation_schedule")
    if (
        type(schedule) is not list
        or len(schedule) != 8
        or any(
            type(pattern) is not list
            or len(pattern) != 3
            or any(
                type(value) is not int or value not in {-1, 1}
                for value in pattern
            )
            for pattern in schedule
        )
    ):
        return "child_evaluation_schedule_invalid"
    for index in range(8 if config["mode"] == "complete" else 1):
        if checkpoints[3 + index]["pattern"] != schedule[index]:
            return "child_memory_checkpoint_pattern_binding_mismatch"
    if any(
        item["pattern"] is not None
        for item in (*checkpoints[:3], checkpoints[-1])
    ):
        return "child_nonpattern_checkpoint_has_pattern"
    if child.get("pair_query_count") != 12:
        return "child_pair_query_count_mismatch"
    if child.get("pair_models_closed") is not True:
        return "child_pair_model_close_mismatch"
    expected_patterns = 8 if config["mode"] == "complete" else 1
    if child.get("scheduled_patterns_completed") != expected_patterns:
        return "child_scheduled_pattern_count_mismatch"
    accepted = child.get("scheduled_candidate_dual_accepted")
    scheduled_lp = child.get("scheduled_local_lp_actual_calls")
    checker_calls = child.get("conditional_checker_actual_calls")
    telemetry_sha = child.get("scheduled_telemetry_sha256")
    if (
        type(accepted) is not int
        or type(scheduled_lp) is not int
        or type(checker_calls) is not int
        or not 0 <= accepted <= expected_patterns
        or not 0 <= scheduled_lp <= expected_patterns
        or checker_calls != 1 + expected_patterns + accepted
        or checker_calls > 17
        or type(telemetry_sha) is not str
        or len(telemetry_sha) != 64
        or any(
            character not in "0123456789abcdef"
            for character in telemetry_sha
        )
    ):
        return "child_scheduled_actual_counter_binding_mismatch"
    if child.get("memory_trace_sha256") != _canonical_sha256(
        {"memory_checkpoints": checkpoints}
    ):
        return "child_memory_trace_checksum_mismatch"
    if config["large_topology"]:
        parent_contract, parent_contract_error = _derive_cgroup_contract(
            preflight.get("cgroup"), require_peak=True
        )
        if parent_contract_error is not None or parent_contract is None:
            return "parent_entry_cgroup_contract_invalid"
        child_preflight = child.get("preflight")
        if not isinstance(child_preflight, Mapping):
            return "child_entry_preflight_missing"
        child_contract, child_contract_error = _derive_cgroup_contract(
            child_preflight.get("cgroup"), require_peak=True
        )
        if child_contract_error is not None or child_contract is None:
            return "child_entry_cgroup_contract_invalid"
        for field in ("leaf", "leaf_max_bytes", "effective_max_bytes"):
            if child_contract[field] != parent_contract[field]:
                return f"child_entry_cgroup_binding_mismatch:{field}"
        if (
            parent_contract["leaf_max_bytes"] > HARD_ABSOLUTE_RSS_CAP_BYTES
            or parent_contract["effective_max_bytes"]
            > HARD_ABSOLUTE_RSS_CAP_BYTES
        ):
            return "entry_cgroup_limit_exceeds_hard_cap"
        for item in checkpoints:
            error = _validate_large_checkpoint_binding(
                item, expected=parent_contract
            )
            if error is not None:
                return "child_large_" + error
    return None


def _contract_validation_error_barrier(
    child: Mapping[str, Any],
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> Optional[str]:
    """Turn every validator failure into a sealed protocol rejection."""

    try:
        return _valid_success_contract(child, config, preflight)
    except BaseException as exc:
        return "child_contract_validator_exception:" + _safe_exception_label(exc)


def _run_worker_process(
    config: Mapping[str, Any],
    estimates: Mapping[str, int],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostic = _base_diagnostic(config, estimates)
    diagnostic["preflight"] = copy.deepcopy(dict(preflight))
    observed_child_peak = 0
    observed_cgroup_peak = 0
    stop_reason = None
    process: Optional[subprocess.Popen[Any]] = None
    monitor_exception: Optional[BaseException] = None
    cleanup_record: Optional[dict[str, Any]] = None
    stdout = b""
    stderr = b""
    expected_cgroup, expected_cgroup_error = _derive_cgroup_contract(
        preflight.get("cgroup"), require_peak=True
    )
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        try:
            process = subprocess.Popen(
                _worker_command(config),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            while process.poll() is None:
                now = time.monotonic()
                child_rss = _current_rss_bytes(process.pid)
                if child_rss is not None:
                    observed_child_peak = max(observed_child_peak, child_rss)
                cgroup = _read_cgroup_v2(process.pid)
                current = cgroup.get("leaf_current_bytes")
                if type(current) is int:
                    observed_cgroup_peak = max(observed_cgroup_peak, current)
                if now >= float(config["deadline"]):
                    stop_reason = "worker_deadline_exceeded"
                    break
                if config["large_topology"]:
                    live_contract, live_contract_error = (
                        _derive_cgroup_contract(cgroup, require_peak=True)
                    )
                    if (
                        expected_cgroup_error is not None
                        or expected_cgroup is None
                        or live_contract_error is not None
                        or live_contract is None
                    ):
                        # A short worker can exit between the loop poll and
                        # its /proc/<pid>/cgroup read.  That is normal only if
                        # an immediate second poll confirms termination.
                        if process.poll() is not None:
                            break
                        stop_reason = (
                            "worker_pid_cgroup_contract_unavailable_during_worker"
                        )
                        break
                    if any(
                        live_contract[field] != expected_cgroup[field]
                        for field in (
                            "leaf",
                            "leaf_max_bytes",
                            "effective_max_bytes",
                        )
                    ):
                        stop_reason = "cgroup_binding_changed_during_worker"
                        break
                    if (
                        live_contract["leaf_max_bytes"]
                        > HARD_ABSOLUTE_RSS_CAP_BYTES
                        or live_contract["effective_max_bytes"]
                        > HARD_ABSOLUTE_RSS_CAP_BYTES
                        or live_contract["leaf_max_bytes"]
                        > int(config["absolute_rss_cap_bytes"])
                    ):
                        stop_reason = "worker_cgroup_limit_relaxed_during_worker"
                        break
                    if current >= (
                        expected_cgroup["effective_max_bytes"]
                        - int(config["rss_reserve_bytes"])
                    ):
                        stop_reason = "cgroup_reserve_stoploss_reached"
                        break
                if (
                    child_rss is not None
                    and child_rss
                    >= int(config["absolute_rss_cap_bytes"])
                    - int(config["rss_reserve_bytes"])
                ):
                    stop_reason = "child_process_rss_stoploss_reached"
                    break
                time.sleep(POLL_SECONDS)
            if stop_reason is None:
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    stop_reason = "worker_exit_wait_timeout"
        except BaseException as exc:
            monitor_exception = exc
        finally:
            if process is not None and (
                monitor_exception is not None
                or stop_reason is not None
                or process.returncode is None
            ):
                cleanup_record = _cleanup_process_noexcept(process)
            try:
                stdout_file.seek(0)
                stdout = stdout_file.read(MAX_WORKER_OUTPUT_BYTES + 1)
            except BaseException as exc:
                if monitor_exception is None:
                    monitor_exception = exc
            try:
                stderr_file.seek(0)
                stderr = stderr_file.read(MAX_WORKER_OUTPUT_BYTES + 1)
            except BaseException as exc:
                if monitor_exception is None:
                    monitor_exception = exc

    if (
        cleanup_record is not None
        and stop_reason is None
        and monitor_exception is None
    ):
        stop_reason = "worker_required_unplanned_cleanup"
    cleanup_failed = bool(
        cleanup_record is not None
        and cleanup_record.get("reaped") is not True
    )
    if monitor_exception is not None:
        if isinstance(monitor_exception, (KeyboardInterrupt, SystemExit)):
            raise monitor_exception
        diagnostic.update(
            status=(
                "worker_cleanup_failed"
                if cleanup_failed
                else "worker_start_failed"
                if process is None
                else "worker_monitor_error"
            ),
            reason=(
                _safe_exception_label(monitor_exception)
                + (
                    ":worker_not_reaped"
                    if cleanup_failed
                    else ""
                )
            ),
            parent_observed_child_peak_rss_bytes=observed_child_peak,
            parent_observed_cgroup_current_peak_bytes=observed_cgroup_peak,
            worker_exit_code=(None if process is None else process.returncode),
            worker_signal=(
                -process.returncode
                if process is not None
                and process.returncode is not None
                and process.returncode < 0
                else None
            ),
            worker_stderr=stderr[:4096].decode(
                "utf-8", errors="replace"
            ),
            worker_cleanup=cleanup_record,
        )
        return _seal(diagnostic)
    if process is None:
        diagnostic.update(
            status="worker_start_failed",
            reason="worker_process_unavailable_without_exception",
        )
        return _seal(diagnostic)
    exit_code = process.returncode
    if stop_reason is not None:
        diagnostic.update(
            status=(
                "worker_cleanup_failed"
                if cleanup_failed
                else "worker_stopped"
            ),
            reason=(
                stop_reason
                + (":worker_not_reaped" if cleanup_failed else "")
            ),
            parent_observed_child_peak_rss_bytes=observed_child_peak,
            parent_observed_cgroup_current_peak_bytes=observed_cgroup_peak,
            worker_exit_code=exit_code,
            worker_signal=(-exit_code if exit_code is not None and exit_code < 0 else None),
            worker_stderr=stderr[:4096].decode("utf-8", errors="replace"),
            worker_cleanup=cleanup_record,
        )
        return _seal(diagnostic)
    if type(exit_code) is not int:
        diagnostic.update(
            status="worker_protocol_error",
            reason="worker_exit_code_not_builtin_integer",
            worker_exit_code=None,
        )
        return _seal(diagnostic)
    if len(stdout) > MAX_WORKER_OUTPUT_BYTES or len(stderr) > MAX_WORKER_OUTPUT_BYTES:
        diagnostic.update(
            status="worker_protocol_error",
            reason="worker_output_exceeds_1_mib",
            worker_exit_code=exit_code,
        )
        return _seal(diagnostic)
    try:
        child = json.loads(stdout.decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        MemoryError,
    ) as exc:
        diagnostic.update(
            status="worker_protocol_error",
            reason=f"worker_json_invalid:{type(exc).__name__}",
            worker_exit_code=exit_code,
            worker_stderr=stderr[:4096].decode("utf-8", errors="replace"),
        )
        return _seal(diagnostic)
    try:
        checksum_ok = bool(
            isinstance(child, Mapping)
            and verify_diagnostic_checksum(child)
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        diagnostic.update(
            status="worker_protocol_error",
            reason="worker_checksum_verifier_exception:"
            + _safe_exception_label(exc),
            worker_exit_code=exit_code,
        )
        return _seal(diagnostic)
    if not checksum_ok:
        diagnostic.update(
            status="worker_protocol_error",
            reason="worker_checksum_invalid",
            worker_exit_code=exit_code,
        )
        return _seal(diagnostic)
    contract_error = _contract_validation_error_barrier(
        child, config, preflight
    )
    if contract_error is not None:
        diagnostic.update(
            status="worker_protocol_error",
            reason=contract_error,
            worker_exit_code=exit_code,
        )
        return _seal(diagnostic)
    result = copy.deepcopy(dict(child))
    result.update(
        parent_observed_child_peak_rss_bytes=observed_child_peak,
        parent_observed_cgroup_current_peak_bytes=observed_cgroup_peak,
        worker_exit_code=exit_code,
        worker_signal=(-exit_code if exit_code is not None and exit_code < 0 else None),
        worker_stderr=stderr[:4096].decode("utf-8", errors="replace"),
    )
    if exit_code != 0 and result.get("status", "").startswith("ok_"):
        result.update(
            status="worker_protocol_error",
            reason="successful_payload_with_nonzero_worker_exit",
        )
    return _seal(result)


def run_k3_pair_scheduled_memory_sentinel(
    *,
    n_cont: int,
    n_bin: int,
    n_upper: int,
    n_eq: int,
    constraint_nnz: int,
    mode: str,
    deadline: float,
    execute_large_profile: bool = False,
    absolute_rss_cap_bytes: int = HARD_ABSOLUTE_RSS_CAP_BYTES,
    rss_reserve_bytes: int = MINIMUM_RSS_RESERVE_BYTES,
    profile_name: Optional[str] = None,
) -> dict[str, Any]:
    """Run one isolated worker, or return a sealed preflight rejection."""

    raw = {
        "n_cont": n_cont,
        "n_bin": n_bin,
        "n_upper": n_upper,
        "n_eq": n_eq,
        "constraint_nnz": constraint_nnz,
        "mode": mode,
        "deadline": deadline,
        "execute_large_profile": execute_large_profile,
        "absolute_rss_cap_bytes": absolute_rss_cap_bytes,
        "rss_reserve_bytes": rss_reserve_bytes,
        "profile_name": profile_name,
    }
    try:
        config, estimates = _validate_config(raw)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _safe_rejection(raw, exc)
    diagnostic = _base_diagnostic(config, estimates)
    preflight, reason = _preflight(config, estimates)
    diagnostic["preflight"] = preflight
    if reason is not None:
        diagnostic.update(status="preflight_rejected", reason=reason)
        return _seal(diagnostic)
    return _run_worker_process(config, estimates, preflight)


def run_fixed_profile(
    *,
    mode: str,
    deadline: float,
    execute_large_profile: bool = False,
    absolute_rss_cap_bytes: int = HARD_ABSOLUTE_RSS_CAP_BYTES,
    rss_reserve_bytes: int = MINIMUM_RSS_RESERVE_BYTES,
) -> dict[str, Any]:
    topology = _PROFILE_DATA["topology"]
    return run_k3_pair_scheduled_memory_sentinel(
        **topology,
        mode=mode,
        deadline=deadline,
        execute_large_profile=execute_large_profile,
        absolute_rss_cap_bytes=absolute_rss_cap_bytes,
        rss_reserve_bytes=rss_reserve_bytes,
        profile_name=PROFILE_NAME,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(_MODES), default="complete")
    parser.add_argument("--deadline-seconds", type=float, default=300.0)
    parser.add_argument("--execute-large-profile", action="store_true")
    parser.add_argument(
        "--_worker-config-json", help=argparse.SUPPRESS, default=None
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker_config_json is not None:
        try:
            raw = json.loads(args._worker_config_json)
            if not isinstance(raw, Mapping):
                raise ValueError("worker_config_not_mapping")
            diagnostic = _execute_worker(raw)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            diagnostic = _safe_rejection(
                {"worker_config": "unavailable"}, exc
            )
        sys.stdout.write(
            json.dumps(
                diagnostic,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        sys.stdout.flush()
        return 0 if diagnostic.get("status", "").startswith("ok_") else 2
    if not math.isfinite(args.deadline_seconds) or args.deadline_seconds <= 0:
        result = _safe_rejection(
            {"deadline_seconds": args.deadline_seconds},
            ValueError("deadline_seconds_must_be_finite_and_positive"),
        )
    else:
        result = run_fixed_profile(
            mode=args.mode,
            deadline=time.monotonic() + args.deadline_seconds,
            execute_large_profile=args.execute_large_profile,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("status", "").startswith("ok_") else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CIFAR100_MEDIUM_IID2_K3_PAIR_SCHEDULED_V1",
    "HARD_ABSOLUTE_RSS_CAP_BYTES",
    "MINIMUM_RSS_RESERVE_BYTES",
    "PROFILE_NAME",
    "SCHEMA",
    "estimate_owned_numeric_payload_bytes",
    "get_fixed_profile",
    "run_fixed_profile",
    "run_k3_pair_scheduled_memory_sentinel",
    "verify_diagnostic_checksum",
]
