#!/usr/bin/env python3
# ===- constraint_block_dag_memory_sentinel.py - bounded RSS gate -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Fail-closed fresh-process RSS sentinel for the disconnected RANGE/DAG.

The sentinel compares the same deterministic, medium-size synthetic guarded
band with the candidate pipeline forced to retain two LE facets and with the
candidate pipeline allowed to retain one native RANGE row.  It records two
independent fresh-process collections: source build/seal, legacy full expanded
replay, and exact full streaming replay.  Only source plus streaming can set
the candidate RSS prerequisite; legacy expanded replay is always closed for
promotion.  The module is deliberately
disconnected from Operator-HZ, solvers, the verifier, and all real/large-model
entry points.

The public entry point has no geometry, repeat, memory-cap, or wall-cap
arguments.  It always runs three cold repeats per mode and per collection in
alternating order (eighteen fresh workers total), with a strict 20 second
aggregate deadline, a 512 MiB process-RSS stop loss, and a 32 MiB retained-
numeric-payload cap.  A diagnostic checksum detects protocol corruption; it
is neither authentication nor proof authority.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from fractions import Fraction
import gc
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp


# Import the frozen candidate by its exact neighbouring file.  Importing the
# top-level ACT package would initialise unrelated ML/solver modules and make
# their hundreds of MiB part of this candidate-only RSS experiment.
_CANDIDATE_PATH = Path(__file__).resolve().with_name(
    "constraint_block_dag_candidate.py"
)
_CANDIDATE_MODULE_NAME = "_act_constraint_block_dag_candidate_for_rss_sentinel"
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "4dcace661ea6886c755ff7848cb6de5f1f440742fdcc0f1d6a69dd713ad03f44"
)
_CANDIDATE_SOURCE_BYTES = _CANDIDATE_PATH.read_bytes()
_CANDIDATE_SOURCE_SHA256 = hashlib.sha256(_CANDIDATE_SOURCE_BYTES).hexdigest()
if _CANDIDATE_SOURCE_SHA256 != EXPECTED_CANDIDATE_SOURCE_SHA256:
    raise ImportError("frozen_constraint_block_dag_candidate_sha256_mismatch")
# Compile and execute the exact bytes that were hashed.  A path-backed loader
# would reopen the file and leave a hash/check-to-exec replacement window.
_dag = types.ModuleType(_CANDIDATE_MODULE_NAME)
_dag.__file__ = str(_CANDIDATE_PATH)
_dag.__package__ = ""
sys.modules[_CANDIDATE_MODULE_NAME] = _dag
with contextlib.redirect_stdout(io.StringIO()):
    exec(
        compile(
            _CANDIDATE_SOURCE_BYTES,
            str(_CANDIDATE_PATH),
            "exec",
            dont_inherit=True,
        ),
        _dag.__dict__,
        _dag.__dict__,
    )
del _CANDIDATE_SOURCE_BYTES


SCHEMA = "act.constraint_block_dag_memory_sentinel.v2"
CHILD_SCHEMA = "act.constraint_block_dag_memory_worker.v2"
CONFIG_SCHEMA = "act.constraint_block_dag_memory_worker_config.v2"
PROFILE_NAME = "bounded_add_range_medium_synthetic_scale20_v1"
_TOY_PROFILE_NAME = "bounded_add_range_toy_synthetic_v1"

COLD_REPEATS = 3
HARD_WALL_SECONDS = 20.0
HARD_RSS_CAP_BYTES = 512 * (1 << 20)
HARD_RETAINED_PAYLOAD_CAP_BYTES = 32 * (1 << 20)
MAX_WORKER_STDOUT_BYTES = 256 * (1 << 10)
MAX_WORKER_STDERR_BYTES = 256 * (1 << 10)
POLL_SECONDS = 0.005
_TERM_WAIT_SECONDS = 0.20
_KILL_WAIT_SECONDS = 0.50
_MODES = ("dual_le", "range")
_STAGES = (
    "source_build_seal",
    "full_build_replay",
    "full_build_stream_replay",
)
STREAM_MAX_ROWS = 128

_PROFILES: dict[str, dict[str, int]] = {
    # ``scale20`` is only a descriptive synthetic scale.  This matrix is not
    # extracted from, or asserted representative of, any real dataset.
    PROFILE_NAME: {
        "pair_count": 2_048,
        "columns": 2_618,
        "low_width": 73,
        "high_width": 74,
        "high_rows": 1_706,
    },
    # Hidden, fixed test geometry.  It cannot be selected by the public API.
    _TOY_PROFILE_NAME: {
        "pair_count": 8,
        "columns": 96,
        "low_width": 5,
        "high_width": 6,
        "high_rows": 5,
    },
}

_FALSE_AUTHORITY = {
    "authority": False,
    "proof_authority": False,
    "verdict_authority": False,
    "production_authority": False,
    "production_integration": False,
    "production_ready": False,
    "promotion_authority": False,
    "real_model_called": False,
    "real_model_allowed": False,
    "large_model_called": False,
    "large_model_allowed": False,
    "triangle_relaxation_called": False,
    "branch_and_bound_called": False,
    "backward_called": False,
    "dual_called": False,
}

_CHILD_SUCCESS_KEYS = frozenset(
    {
        "schema",
        "status",
        "reason",
        "diagnostic_only",
        "candidate_only",
        "synthetic_only",
        *_FALSE_AUTHORITY.keys(),
        "profile_name",
        "mode",
        "measurement_stage",
        "repeat_index",
        "order_index",
        "nonce",
        "config_sha256",
        "candidate_source_sha256",
        "absolute_deadline_monotonic_hex",
        "geometry",
        "entry",
        "terminal",
        "rss_current_delta_bytes",
        "rss_current_growth_bytes",
        "rss_peak_delta_bytes",
        "cgroup_current_delta_bytes",
        "retained_payload_bytes",
        "source_rows",
        "source_nnz",
        "virtual_facet_rows",
        "ranged_rows",
        "fallback_pairs",
        "structure_complete",
        "replay_complete",
        "replay_kind",
        "fraction_membership_complete",
        "stream_batch_count",
        "stream_max_rows",
        "stream_peak_batch_rows",
        "candidate_receipt_safe",
        "replay_sha256",
        "candidate_receipt_sha256",
        "worker_stage_wall_seconds_hex",
        "diagnostic_sha256",
    }
)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reject_json_constant(token: str) -> Any:
    raise ValueError("nonfinite_json_constant_rejected:" + token)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_object_key_rejected:" + key[:128])
        result[key] = value
    return result


def _strict_json_loads(raw: Any) -> Any:
    """Parse one JSON document while rejecting duplicate keys and NaN/Inf."""

    if type(raw) not in {str, bytes, bytearray}:
        raise TypeError("strict_json_input_type_invalid")
    return json.loads(
        raw,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


def _finite_hex_float(value: Any, *, name: str) -> float:
    if type(value) is not str:
        raise ValueError(name + "_must_be_exact_hex_string")
    try:
        result = float.fromhex(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + "_hex_invalid") from None
    if not math.isfinite(result):
        raise ValueError(name + "_nonfinite")
    return result


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result.pop("diagnostic_sha256", None)
    result["diagnostic_sha256"] = _sha256(result)
    return result


def verify_diagnostic_checksum(diagnostic: Mapping[str, Any]) -> bool:
    """Return whether a diagnostic has intact canonical-JSON bytes."""

    try:
        if not isinstance(diagnostic, Mapping):
            return False
        expected = diagnostic.get("diagnostic_sha256")
        if (
            type(expected) is not str
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            return False
        body = copy.deepcopy(dict(diagnostic))
        body.pop("diagnostic_sha256", None)
        return hmac.compare_digest(expected, _sha256(body))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return False


def _safe_exception_label(exc: BaseException) -> str:
    try:
        text = str(exc).replace("\n", " ")[:384]
    except BaseException:
        text = "exception_text_unavailable"
    return f"{type(exc).__name__}:{text}"


def _closed(reason: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "closed",
        "reason": str(reason)[:512],
        "diagnostic_only": True,
        "candidate_only": True,
        "synthetic_only": True,
        "rss_gate_passed": False,
        "source_stage_rss_gate_passed": False,
        "full_rss_gate_passed": False,
        "stream_full_rss_gate_passed": False,
        "full_expanded_gate_closed": True,
        **_FALSE_AUTHORITY,
        "profile_name": PROFILE_NAME,
        "cold_repeats_required": COLD_REPEATS,
        "hard_wall_seconds_hex": HARD_WALL_SECONDS.hex(),
        "hard_rss_cap_bytes": HARD_RSS_CAP_BYTES,
        "hard_retained_payload_cap_bytes": HARD_RETAINED_PAYLOAD_CAP_BYTES,
        "candidate_source_sha256": _CANDIDATE_SOURCE_SHA256,
        "runs": [],
        "summaries": None,
        "gate_checks": None,
    }
    body.update(extra)
    return _seal(body)


def _read_status_bytes(pid: Optional[int], key: str) -> Optional[int]:
    if key not in {"VmRSS", "VmHWM"}:
        return None
    target = "self" if pid is None else str(pid)
    try:
        with open(f"/proc/{target}/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    fields = line.split()
                    if len(fields) != 3 or fields[2] != "kB":
                        return None
                    kib = int(fields[1])
                    return kib * 1024 if kib >= 0 else None
    except (OSError, ValueError):
        return None
    return None


def _read_cgroup_v2_current() -> dict[str, Any]:
    """Read optional cgroup-v2 aggregate current memory for this worker."""

    empty = {
        "available": False,
        "version": None,
        "leaf": None,
        "current_bytes": None,
        "error": None,
    }
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        return {**empty, "error": "membership_read_failed:" + type(exc).__name__}
    relative = None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            relative = fields[2]
            break
    if relative is None:
        return {**empty, "error": "cgroup_v2_membership_unavailable"}
    root = Path("/sys/fs/cgroup").resolve()
    leaf = (root / relative.lstrip("/")).resolve()
    try:
        leaf.relative_to(root)
    except ValueError:
        return {
            **empty,
            "version": 2,
            "leaf": str(leaf),
            "error": "cgroup_leaf_escapes_mount",
        }
    try:
        raw = (leaf / "memory.current").read_text(encoding="ascii").strip()
        current = int(raw)
    except (OSError, ValueError) as exc:
        return {
            **empty,
            "version": 2,
            "leaf": str(leaf),
            "error": "memory_current_unreadable:" + type(exc).__name__,
        }
    if current < 0:
        return {
            **empty,
            "version": 2,
            "leaf": str(leaf),
            "error": "memory_current_negative",
        }
    return {
        "available": True,
        "version": 2,
        "leaf": str(leaf),
        "current_bytes": current,
        "error": None,
    }


def _sample_memory(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "sampled_monotonic_hex": time.monotonic().hex(),
        "vmrss_bytes": _read_status_bytes(None, "VmRSS"),
        "vmhwm_bytes": _read_status_bytes(None, "VmHWM"),
        "cgroup_v2": _read_cgroup_v2_current(),
    }


def _profile_geometry(name: str) -> dict[str, int]:
    if type(name) is not str or name not in _PROFILES:
        raise ValueError("unknown_fixed_synthetic_profile")
    result = dict(_PROFILES[name])
    pair_count = result["pair_count"]
    columns = result["columns"]
    low = result["low_width"]
    high = result["high_width"]
    high_rows = result["high_rows"]
    if not (
        1 <= pair_count <= 2_048
        and 1 <= low <= high <= 74
        and high <= columns <= 2_618
        and 0 <= high_rows <= pair_count
    ):
        raise ValueError("fixed_synthetic_profile_exceeds_hard_geometry")
    result["forward_nnz"] = high_rows * high + (pair_count - high_rows) * low
    return result


def _synthetic_forward(profile_name: str) -> sp.csr_matrix:
    geometry = _profile_geometry(profile_name)
    rows = geometry["pair_count"]
    columns = geometry["columns"]
    widths = np.full(rows, geometry["low_width"], dtype=np.int64)
    widths[: geometry["high_rows"]] = geometry["high_width"]
    indptr = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(widths, dtype=np.int64))
    )
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    data = np.empty(int(indptr[-1]), dtype=np.float64)
    cursor = 0
    for row, width in enumerate(widths.tolist()):
        start = (row * 97) % columns
        row_indices = np.sort(
            (start + np.arange(width, dtype=np.int64) * 17) % columns
        )
        if np.unique(row_indices).size != width:
            row_indices = np.arange(width, dtype=np.int64)
        stop = cursor + width
        indices[cursor:stop] = row_indices
        ordinal = np.arange(width, dtype=np.int64)
        values = ((ordinal % 29) + 1).astype(np.float64) / 32.0
        values[(ordinal + row) % 2 == 1] *= -1.0
        data[cursor:stop] = values
        cursor = stop
    result = sp.csr_matrix(
        (data, indices, indptr),
        shape=(rows, columns),
        dtype=np.float64,
    )
    result.sort_indices()
    if result.nnz != geometry["forward_nnz"]:
        raise RuntimeError("synthetic_forward_nnz_mismatch")
    return result


def _csr_exact_equal(left: sp.csr_matrix, right: sp.csr_matrix) -> bool:
    return bool(
        type(left) is sp.csr_matrix
        and type(right) is sp.csr_matrix
        and left.shape == right.shape
        and left.dtype == right.dtype == np.dtype(np.float64)
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and left.data.view(np.uint64).tobytes()
        == right.data.view(np.uint64).tobytes()
    )


def _replay_digest(replay: Any) -> str:
    digest = hashlib.sha256()
    for matrix in (replay.A_cont, replay.A_bin):
        digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
        digest.update(np.asarray(matrix.indptr, dtype=np.int64).tobytes())
        digest.update(np.asarray(matrix.indices, dtype=np.int64).tobytes())
        digest.update(
            np.asarray(matrix.data, dtype=np.float64)
            .view(np.uint64)
            .tobytes()
        )
    digest.update(np.asarray(replay.upper, dtype=np.float64).view(np.uint64).tobytes())
    for tag in replay.row_tags:
        encoded = tag.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _construct_and_seal(
    profile_name: str, mode: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    geometry = _profile_geometry(profile_name)
    forward = _synthetic_forward(profile_name)
    reverse = forward.copy()
    reverse.data = np.negative(reverse.data)
    reverse.sort_indices()
    pair_count = geometry["pair_count"]
    owner = _dag.ExactConstraintOwner()
    owner.allocate_continuous(geometry["columns"])
    frame = owner.frame()
    arena = owner.new_arena()
    empty_bin = sp.csr_matrix((pair_count, 0), dtype=np.float64)
    upper = np.full(pair_count, 8.0, dtype=np.float64)
    appended = arena._append_guarded_band(
        arena.empty_view,
        frame=frame,
        forward_cont=forward,
        forward_bin=empty_bin,
        forward_upper=upper,
        reverse_cont=reverse,
        reverse_bin=empty_bin,
        reverse_upper=upper.copy(),
        layer_id=7,
        family="add_materialize",
        allow_range=(mode == "range"),
    )
    program = arena.seal(appended.view, final_frame=frame)
    expected_rows = pair_count if mode == "range" else 2 * pair_count
    expected_nnz = forward.nnz if mode == "range" else 2 * forward.nnz
    expected_ranged = pair_count if mode == "range" else 0
    expected_fallback = 0 if mode == "range" else pair_count
    structure_complete = bool(
        program.block_count == 1
        and program.source_rows == expected_rows
        and program.source_nnz == expected_nnz
        and program.virtual_facet_rows == 2 * pair_count
        and program.ranged_rows == expected_ranged
        and program.fallback_pairs == expected_fallback
    )

    receipt = dict(program.receipt)
    receipt_safe = bool(
        all(receipt.get(key) is False for key in (
            "proof_authority",
            "verdict_authority",
            "authenticity_authority",
            "production_integration",
            "triangle_relaxation_called",
            "branch_and_bound_called",
            "backward_called",
            "dual_called",
            "real_model_called",
            "large_model_called",
        ))
        and receipt.get("candidate_only") is True
        and receipt.get("bytes_backed_source") is True
        and receipt.get("virtual_facets_replayable") is True
        and receipt.get("stream_virtual_facets_replayable") is True
        and receipt.get("full_expanded_gate_closed") is True
    )
    metrics = {
        "geometry": geometry,
        "retained_payload_bytes": int(program.numeric_payload_bytes),
        "source_rows": int(program.source_rows),
        "source_nnz": int(program.source_nnz),
        "virtual_facet_rows": int(program.virtual_facet_rows),
        "ranged_rows": int(program.ranged_rows),
        "fallback_pairs": int(program.fallback_pairs),
        "structure_complete": structure_complete,
        "candidate_receipt_safe": receipt_safe,
        "candidate_receipt_sha256": _sha256(receipt),
    }
    # The caller owns this bounded context.  The full-stage collection replays
    # while it is live (the original measurement flow); the source collection
    # releases it before taking its terminal checkpoint.
    context = {
        "appended": appended,
        "arena": arena,
        "frame": frame,
        "owner": owner,
        "forward": forward,
        "reverse": reverse,
        "empty_bin": empty_bin,
        "upper": upper,
    }
    return program, metrics, context


def _validate_replay(
    program: Any,
    profile_name: str,
    build_context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Validate expanded facets after the build/seal terminal checkpoint.

    Replay is a consumer of the sealed source, not part of source publication.
    The parent still monitors its RSS and deadline, but its common expanded
    two-LE frame does not contaminate the build/seal ``VmHWM`` delta.
    """

    geometry = _profile_geometry(profile_name)
    pair_count = geometry["pair_count"]
    if build_context is None:
        forward = _synthetic_forward(profile_name)
        reverse = forward.copy()
        reverse.data = np.negative(reverse.data)
        reverse.sort_indices()
        upper = np.full(pair_count, 8.0, dtype=np.float64)
    else:
        forward = build_context["forward"]
        reverse = build_context["reverse"]
        upper = build_context["upper"]
    replay = program.replay_virtual_facets()
    expected_cont = sp.vstack((forward, reverse), format="csr")
    expected_bin = sp.csr_matrix((2 * pair_count, 0), dtype=np.float64)
    expected_upper = np.concatenate((upper, upper))
    expected_tags = tuple(
        ["add_materialize:7:forward"] * pair_count
        + ["add_materialize:7:reverse"] * pair_count
    )
    replay_complete = bool(
        _csr_exact_equal(replay.A_cont, expected_cont)
        and _csr_exact_equal(replay.A_bin, expected_bin)
        and replay.upper.view(np.uint64).tobytes()
        == expected_upper.view(np.uint64).tobytes()
        and replay.row_tags == expected_tags
        and len(replay.row_ids) == 2 * pair_count
        and len(set(replay.row_ids)) == 2 * pair_count
        and tuple(replay.continuous_ids) == tuple(program.continuous_ids)
        and tuple(replay.binary_ids) == tuple(program.binary_ids)
    )
    result = {
        "replay_complete": replay_complete,
        "replay_sha256": _replay_digest(replay),
        "replay_kind": "legacy_full_expanded",
        "fraction_membership_complete": False,
        "stream_batch_count": 0,
        "stream_max_rows": 0,
        "stream_peak_batch_rows": 0,
    }
    del replay, expected_cont, expected_bin, expected_upper
    if build_context is None:
        del forward, reverse, upper
    gc.collect()
    return result


def _expected_synthetic_row(
    geometry: Mapping[str, int], row: int, *, reverse: bool
) -> tuple[np.ndarray, np.ndarray]:
    width = (
        geometry["high_width"]
        if row < geometry["high_rows"]
        else geometry["low_width"]
    )
    columns = geometry["columns"]
    start = (row * 97) % columns
    indices = np.sort(
        (start + np.arange(width, dtype=np.int64) * 17) % columns
    )
    if np.unique(indices).size != width:
        indices = np.arange(width, dtype=np.int64)
    ordinal = np.arange(width, dtype=np.int64)
    values = ((ordinal % 29) + 1).astype(np.float64) / 32.0
    values[(ordinal + row) % 2 == 1] *= -1.0
    if reverse:
        bits = values.view(np.uint64).copy()
        bits ^= np.uint64(1 << 63)
        values = bits.view(np.float64)
    return indices, values


def _exact_fraction_membership(
    indices: np.ndarray, data: np.ndarray, upper: np.float64
) -> bool:
    scaled = np.rint(np.asarray(data, dtype=np.float64) * 32.0)
    reconstructed = (scaled / 32.0).astype(np.float64, copy=False)
    if (
        reconstructed.view(np.uint64).tobytes()
        != np.asarray(data, dtype=np.float64).view(np.uint64).tobytes()
    ):
        raise RuntimeError("stream_fraction_coefficient_not_exact_dyadic")
    numerators = scaled.astype(np.int64)
    point_numerators = (np.asarray(indices, dtype=np.int64) % 5) - 2
    lhs = Fraction(int(np.dot(numerators, point_numerators)), 256)
    rhs = Fraction.from_float(float(upper))
    return bool(lhs <= rhs)


def _update_stream_row_digest(
    digest: Any,
    *,
    row_offset: int,
    cont_indices: np.ndarray,
    cont_data: np.ndarray,
    bin_indices: np.ndarray,
    bin_data: np.ndarray,
    upper: np.float64,
    tag: str,
) -> None:
    digest.update(b"row")
    digest.update(int(row_offset).to_bytes(8, "little", signed=False))
    for indices, data in (
        (cont_indices, cont_data),
        (bin_indices, bin_data),
    ):
        digest.update(int(indices.size).to_bytes(8, "little", signed=False))
        digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        digest.update(
            np.asarray(data, dtype=np.float64).view(np.uint64).tobytes()
        )
    digest.update(np.asarray([upper], dtype=np.float64).view(np.uint64).tobytes())
    encoded_tag = tag.encode("utf-8")
    digest.update(len(encoded_tag).to_bytes(4, "little"))
    digest.update(encoded_tag)


def _validate_stream_replay(
    program: Any,
    profile_name: str,
    *,
    absolute_deadline_monotonic: float,
) -> dict[str, Any]:
    """Consume and validate bounded batches without any expanded frame."""

    geometry = _profile_geometry(profile_name)
    pair_count = geometry["pair_count"]
    total_rows = 2 * pair_count
    iterator = None
    batch = None
    Ac = None
    Ab = None
    upper = None
    digest = hashlib.sha256()
    digest.update(b"act.exact_virtual_facet_stream.v1")
    row_offset = 0
    batch_count = 0
    peak_batch_rows = 0
    seen_ids: set[tuple[str, int]] = set()
    replay_complete = True
    fraction_complete = True
    factor_frame_checked = False
    try:
        iterator = _dag.iter_virtual_facet_batches(
            program, max_rows=STREAM_MAX_ROWS
        )
        for batch in iterator:
            if time.monotonic() >= absolute_deadline_monotonic:
                raise TimeoutError("stream_replay_deadline_expired")
            current_rss = _read_status_bytes(None, "VmRSS")
            current_hwm = _read_status_bytes(None, "VmHWM")
            if (
                current_rss is None
                or current_hwm is None
                or current_rss >= HARD_RSS_CAP_BYTES
                or current_hwm >= HARD_RSS_CAP_BYTES
            ):
                raise MemoryError("stream_replay_rss_reaches_hard_cap")
            if (
                type(batch) is not _dag.VirtualFacetBatch
                or batch.bytes_backed is not True
                or batch.proof_authority is not False
                or batch.verdict_authority is not False
                or batch.row_offset != row_offset
                or batch.total_rows != total_rows
                or not 1 <= batch.row_count <= STREAM_MAX_ROWS
            ):
                raise RuntimeError("stream_batch_contract_invalid")
            if not factor_frame_checked:
                continuous_ids = batch.continuous_ids
                binary_ids = batch.binary_ids
                replay_complete = bool(
                    len(continuous_ids) == geometry["columns"]
                    and all(value.kind == "continuous" for value in continuous_ids)
                    and binary_ids == ()
                )
                factor_frame_checked = True
                del continuous_ids, binary_ids
            Ac = batch.A_cont
            Ab = batch.A_bin
            upper = batch.upper
            row_ids = batch.row_ids
            row_tags = batch.row_tags
            if (
                Ac.shape != (batch.row_count, geometry["columns"])
                or Ab.shape != (batch.row_count, 0)
                or Ab.nnz != 0
                or upper.shape != (batch.row_count,)
                or len(row_ids) != batch.row_count
                or len(row_tags) != batch.row_count
            ):
                raise RuntimeError("stream_batch_shape_invalid")
            for local_row in range(batch.row_count):
                global_row = row_offset + local_row
                reverse = global_row >= pair_count
                source_row = global_row - pair_count if reverse else global_row
                expected_indices, expected_data = _expected_synthetic_row(
                    geometry, source_row, reverse=reverse
                )
                start, stop = int(Ac.indptr[local_row]), int(
                    Ac.indptr[local_row + 1]
                )
                actual_indices = Ac.indices[start:stop]
                actual_data = Ac.data[start:stop]
                bstart, bstop = int(Ab.indptr[local_row]), int(
                    Ab.indptr[local_row + 1]
                )
                actual_bin_indices = Ab.indices[bstart:bstop]
                actual_bin_data = Ab.data[bstart:bstop]
                expected_upper = np.float64(8.0)
                expected_tag = (
                    "add_materialize:7:reverse"
                    if reverse
                    else "add_materialize:7:forward"
                )
                row_id = row_ids[local_row]
                id_key = (row_id.kind, row_id.value)
                row_bits_ok = bool(
                    np.array_equal(actual_indices, expected_indices)
                    and actual_data.view(np.uint64).tobytes()
                    == expected_data.view(np.uint64).tobytes()
                    and actual_bin_indices.size == 0
                    and actual_bin_data.size == 0
                    and np.asarray([upper[local_row]], dtype=np.float64)
                    .view(np.uint64)
                    .tobytes()
                    == np.asarray([expected_upper], dtype=np.float64)
                    .view(np.uint64)
                    .tobytes()
                    and type(row_id) is _dag.StableObjectID
                    and row_id.kind == "facet"
                    and type(row_id.value) is int
                    and id_key not in seen_ids
                    and row_tags[local_row] == expected_tag
                )
                replay_complete = replay_complete and row_bits_ok
                seen_ids.add(id_key)
                actual_member = _exact_fraction_membership(
                    actual_indices, actual_data, upper[local_row]
                )
                expected_member = _exact_fraction_membership(
                    expected_indices, expected_data, expected_upper
                )
                fraction_complete = bool(
                    fraction_complete and actual_member == expected_member
                )
                _update_stream_row_digest(
                    digest,
                    row_offset=global_row,
                    cont_indices=actual_indices,
                    cont_data=actual_data,
                    bin_indices=actual_bin_indices,
                    bin_data=actual_bin_data,
                    upper=upper[local_row],
                    tag=row_tags[local_row],
                )
            row_offset += batch.row_count
            batch_count += 1
            peak_batch_rows = max(peak_batch_rows, batch.row_count)
            batch = None
            Ac = None
            Ab = None
            upper = None
        replay_complete = bool(
            replay_complete
            and row_offset == total_rows
            and len(seen_ids) == total_rows
            and factor_frame_checked
        )
    finally:
        if iterator is not None:
            iterator.close()
        batch = None
        Ac = None
        Ab = None
        upper = None
        iterator = None
        gc.collect()
    digest.update(b"rows")
    digest.update(int(row_offset).to_bytes(8, "little", signed=False))
    return {
        "replay_complete": replay_complete,
        "replay_sha256": digest.hexdigest(),
        "replay_kind": "bounded_exact_stream",
        "fraction_membership_complete": fraction_complete,
        "stream_batch_count": batch_count,
        "stream_max_rows": STREAM_MAX_ROWS,
        "stream_peak_batch_rows": peak_batch_rows,
    }


def _config_body(
    *,
    profile_name: str,
    mode: str,
    measurement_stage: str,
    repeat_index: int,
    order_index: int,
    nonce: str,
    absolute_deadline_monotonic: float,
) -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "profile_name": profile_name,
        "mode": mode,
        "measurement_stage": measurement_stage,
        "repeat_index": repeat_index,
        "order_index": order_index,
        "nonce": nonce,
        "absolute_deadline_monotonic_hex": absolute_deadline_monotonic.hex(),
        "hard_rss_cap_bytes": HARD_RSS_CAP_BYTES,
        "hard_retained_payload_cap_bytes": HARD_RETAINED_PAYLOAD_CAP_BYTES,
        "candidate_source_sha256": _CANDIDATE_SOURCE_SHA256,
    }


def _make_worker_config(
    *,
    profile_name: str,
    mode: str,
    measurement_stage: str = "source_build_seal",
    repeat_index: int,
    order_index: int,
    nonce: str,
    absolute_deadline_monotonic: float,
) -> dict[str, Any]:
    body = _config_body(
        profile_name=profile_name,
        mode=mode,
        measurement_stage=measurement_stage,
        repeat_index=repeat_index,
        order_index=order_index,
        nonce=nonce,
        absolute_deadline_monotonic=absolute_deadline_monotonic,
    )
    body["config_sha256"] = _sha256(body)
    return body


def _validate_worker_config(raw: Any) -> dict[str, Any]:
    if type(raw) is not dict:
        raise ValueError("worker_config_must_be_exact_dict")
    required = {
        "schema",
        "profile_name",
        "mode",
        "measurement_stage",
        "repeat_index",
        "order_index",
        "nonce",
        "absolute_deadline_monotonic_hex",
        "hard_rss_cap_bytes",
        "hard_retained_payload_cap_bytes",
        "candidate_source_sha256",
        "config_sha256",
    }
    if set(raw) != required:
        raise ValueError("worker_config_keys_mismatch")
    expected = raw.get("config_sha256")
    body = dict(raw)
    body.pop("config_sha256", None)
    if (
        type(expected) is not str
        or len(expected) != 64
        or not hmac.compare_digest(expected, _sha256(body))
    ):
        raise ValueError("worker_config_checksum_invalid")
    profile_name = raw["profile_name"]
    _profile_geometry(profile_name)
    if (
        raw["schema"] != CONFIG_SCHEMA
        or raw["mode"] not in _MODES
        or raw["measurement_stage"] not in _STAGES
    ):
        raise ValueError("worker_config_schema_or_mode_invalid")
    if (
        type(raw["repeat_index"]) is not int
        or not 0 <= raw["repeat_index"] < COLD_REPEATS
        or type(raw["order_index"]) is not int
        or not 0 <= raw["order_index"] < len(_STAGES) * 2 * COLD_REPEATS
        or type(raw["nonce"]) is not str
        or len(raw["nonce"]) != 32
        or any(character not in "0123456789abcdef" for character in raw["nonce"])
        or raw["hard_rss_cap_bytes"] != HARD_RSS_CAP_BYTES
        or raw["hard_retained_payload_cap_bytes"]
        != HARD_RETAINED_PAYLOAD_CAP_BYTES
        or raw["candidate_source_sha256"] != EXPECTED_CANDIDATE_SOURCE_SHA256
        or _CANDIDATE_SOURCE_SHA256 != EXPECTED_CANDIDATE_SOURCE_SHA256
    ):
        raise ValueError("worker_config_binding_invalid")
    deadline = _finite_hex_float(
        raw["absolute_deadline_monotonic_hex"], name="worker_deadline"
    )
    result = dict(raw)
    result["absolute_deadline_monotonic"] = deadline
    return result


def _worker_failure(config: Any, reason: str) -> dict[str, Any]:
    safe = config if type(config) is dict else {}
    return _seal(
        {
            "schema": CHILD_SCHEMA,
            "status": "closed",
            "reason": reason[:512],
            "diagnostic_only": True,
            "candidate_only": True,
            "synthetic_only": True,
            **_FALSE_AUTHORITY,
            "profile_name": safe.get("profile_name"),
            "mode": safe.get("mode"),
            "measurement_stage": safe.get("measurement_stage"),
            "repeat_index": safe.get("repeat_index"),
            "order_index": safe.get("order_index"),
            "nonce": safe.get("nonce"),
            "config_sha256": safe.get("config_sha256"),
            "candidate_source_sha256": safe.get("candidate_source_sha256"),
            "absolute_deadline_monotonic_hex": safe.get(
                "absolute_deadline_monotonic_hex"
            ),
        }
    )


def _execute_worker(raw_config: Any) -> dict[str, Any]:
    """Execute one fixed mode in the current (already fresh) process."""

    try:
        config = _validate_worker_config(raw_config)
        deadline = config["absolute_deadline_monotonic"]
        worker_entry_now = time.monotonic()
        if worker_entry_now >= deadline:
            raise TimeoutError("absolute_deadline_expired_at_worker_entry")
        if deadline - worker_entry_now > HARD_WALL_SECONDS:
            raise ValueError("worker_deadline_attempts_to_exceed_hard_wall")
        gc.collect()
        entry = _sample_memory("entry")
        if entry["vmrss_bytes"] is None or entry["vmhwm_bytes"] is None:
            raise RuntimeError("proc_status_entry_memory_unavailable")
        if (
            entry["vmrss_bytes"] >= HARD_RSS_CAP_BYTES
            or entry["vmhwm_bytes"] >= HARD_RSS_CAP_BYTES
        ):
            raise MemoryError("entry_rss_reaches_hard_cap")
        started = time.monotonic()
        program, metrics, build_context = _construct_and_seal(
            config["profile_name"], config["mode"]
        )
        replay_metrics = None
        if config["measurement_stage"] == "full_build_replay":
            replay_metrics = _validate_replay(
                program, config["profile_name"], build_context
            )
        build_context.clear()
        del build_context
        gc.collect()
        if config["measurement_stage"] == "full_build_stream_replay":
            replay_metrics = _validate_stream_replay(
                program,
                config["profile_name"],
                absolute_deadline_monotonic=deadline,
            )
            gc.collect()
        terminal = _sample_memory("terminal")
        if terminal["vmrss_bytes"] is None or terminal["vmhwm_bytes"] is None:
            raise RuntimeError("proc_status_terminal_memory_unavailable")
        if (
            terminal["vmrss_bytes"] >= HARD_RSS_CAP_BYTES
            or terminal["vmhwm_bytes"] >= HARD_RSS_CAP_BYTES
        ):
            raise MemoryError("terminal_rss_reaches_hard_cap")
        if config["measurement_stage"] == "source_build_seal":
            replay_metrics = _validate_replay(program, config["profile_name"])
            # Replay is outside the source-stage delta but remains inside the
            # worker's hard memory/deadline envelope.
            replay_guard_rss = _read_status_bytes(None, "VmRSS")
            replay_guard_hwm = _read_status_bytes(None, "VmHWM")
            if (
                replay_guard_rss is None
                or replay_guard_hwm is None
                or replay_guard_rss >= HARD_RSS_CAP_BYTES
                or replay_guard_hwm >= HARD_RSS_CAP_BYTES
            ):
                raise MemoryError("post_terminal_replay_reaches_hard_cap")
        if replay_metrics is None:
            raise RuntimeError("replay_validation_not_executed")
        metrics.update(replay_metrics)
        if time.monotonic() >= deadline:
            raise TimeoutError("absolute_deadline_expired_after_replay_validation")
        retained = metrics["retained_payload_bytes"]
        if retained >= HARD_RETAINED_PAYLOAD_CAP_BYTES:
            raise MemoryError("retained_payload_reaches_hard_cap")
        current_delta = terminal["vmrss_bytes"] - entry["vmrss_bytes"]
        peak_delta = terminal["vmhwm_bytes"] - entry["vmhwm_bytes"]
        if peak_delta < 0:
            raise RuntimeError("vmhwm_decreased_within_worker")
        entry_cgroup = entry["cgroup_v2"]
        terminal_cgroup = terminal["cgroup_v2"]
        cgroup_delta = None
        if (
            entry_cgroup.get("available") is True
            and terminal_cgroup.get("available") is True
            and entry_cgroup.get("leaf") == terminal_cgroup.get("leaf")
        ):
            cgroup_delta = (
                terminal_cgroup["current_bytes"] - entry_cgroup["current_bytes"]
            )
        del program
        result = {
            "schema": CHILD_SCHEMA,
            "status": "ok",
            "reason": None,
            "diagnostic_only": True,
            "candidate_only": True,
            "synthetic_only": True,
            **_FALSE_AUTHORITY,
            "profile_name": config["profile_name"],
            "mode": config["mode"],
            "measurement_stage": config["measurement_stage"],
            "repeat_index": config["repeat_index"],
            "order_index": config["order_index"],
            "nonce": config["nonce"],
            "config_sha256": config["config_sha256"],
            "candidate_source_sha256": config["candidate_source_sha256"],
            "absolute_deadline_monotonic_hex": config[
                "absolute_deadline_monotonic_hex"
            ],
            "geometry": metrics["geometry"],
            "entry": entry,
            "terminal": terminal,
            "rss_current_delta_bytes": current_delta,
            "rss_current_growth_bytes": max(0, current_delta),
            "rss_peak_delta_bytes": peak_delta,
            "cgroup_current_delta_bytes": cgroup_delta,
            "retained_payload_bytes": retained,
            "source_rows": metrics["source_rows"],
            "source_nnz": metrics["source_nnz"],
            "virtual_facet_rows": metrics["virtual_facet_rows"],
            "ranged_rows": metrics["ranged_rows"],
            "fallback_pairs": metrics["fallback_pairs"],
            "structure_complete": metrics["structure_complete"],
            "replay_complete": metrics["replay_complete"],
            "replay_kind": metrics["replay_kind"],
            "fraction_membership_complete": metrics[
                "fraction_membership_complete"
            ],
            "stream_batch_count": metrics["stream_batch_count"],
            "stream_max_rows": metrics["stream_max_rows"],
            "stream_peak_batch_rows": metrics[
                "stream_peak_batch_rows"
            ],
            "candidate_receipt_safe": metrics["candidate_receipt_safe"],
            "replay_sha256": metrics["replay_sha256"],
            "candidate_receipt_sha256": metrics[
                "candidate_receipt_sha256"
            ],
            "worker_stage_wall_seconds_hex": (time.monotonic() - started).hex(),
        }
        return _seal(result)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        return _worker_failure(raw_config, _safe_exception_label(exc))


def _worker_command(config: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-config-json",
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False),
    ]


def _cleanup_process_noexcept(process: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "termination_required": True,
        "terminate_attempted": False,
        "kill_attempted": False,
        "reaped": False,
        "exit_code": None,
        "cleanup_error": None,
    }
    try:
        initial = process.poll()
    except BaseException:
        initial = None
    if type(initial) is int:
        record.update(reaped=True, exit_code=initial)
        return record
    try:
        record["terminate_attempted"] = True
        process.terminate()
    except BaseException:
        pass
    try:
        code = process.wait(timeout=_TERM_WAIT_SECONDS)
        if type(code) is int:
            record.update(reaped=True, exit_code=code)
            return record
    except BaseException:
        pass
    try:
        record["kill_attempted"] = True
        process.kill()
    except BaseException:
        pass
    try:
        code = process.wait(timeout=_KILL_WAIT_SECONDS)
        if type(code) is int:
            record.update(reaped=True, exit_code=code)
            return record
    except BaseException:
        pass
    record["cleanup_error"] = "worker_not_reaped_after_term_and_kill"
    return record


def _validate_cgroup_sample(value: Any) -> Optional[str]:
    if type(value) is not dict or set(value) != {
        "available", "version", "leaf", "current_bytes", "error"
    }:
        return "cgroup_shape_invalid"
    if type(value["available"]) is not bool:
        return "cgroup_available_invalid"
    if value["available"]:
        leaf_valid = False
        if type(value["leaf"]) is str:
            try:
                leaf_path = Path(value["leaf"])
                leaf_path.relative_to(Path("/sys/fs/cgroup"))
                leaf_valid = leaf_path.is_absolute() and ".." not in leaf_path.parts
            except ValueError:
                leaf_valid = False
        if (
            value["version"] != 2
            or not leaf_valid
            or type(value["current_bytes"]) is not int
            or value["current_bytes"] < 0
            or value["error"] is not None
        ):
            return "cgroup_available_fields_invalid"
    elif (
        value["current_bytes"] is not None
        or value["version"] not in {None, 2}
        or value["leaf"] is not None and type(value["leaf"]) is not str
        or value["error"] is not None and type(value["error"]) is not str
    ):
        return "cgroup_unavailable_fields_invalid"
    return None


def _validate_memory_sample(value: Any, stage: str) -> Optional[str]:
    if type(value) is not dict or set(value) != {
        "stage", "sampled_monotonic_hex", "vmrss_bytes", "vmhwm_bytes", "cgroup_v2"
    }:
        return "memory_sample_shape_invalid"
    if value["stage"] != stage:
        return "memory_sample_stage_invalid"
    try:
        sampled = _finite_hex_float(
            value["sampled_monotonic_hex"], name="memory_sample_time"
        )
    except ValueError:
        return "memory_sample_time_invalid"
    for key in ("vmrss_bytes", "vmhwm_bytes"):
        if type(value[key]) is not int or not 0 <= value[key] < HARD_RSS_CAP_BYTES:
            return "memory_sample_rss_invalid"
    if value["vmhwm_bytes"] < value["vmrss_bytes"]:
        return "memory_sample_hwm_below_current"
    return _validate_cgroup_sample(value["cgroup_v2"])


def _validate_child_success(child: Any, config: Mapping[str, Any]) -> Optional[str]:
    if type(child) is not dict:
        return "worker_document_not_exact_dict"
    if not verify_diagnostic_checksum(child):
        return "worker_checksum_invalid"
    if set(child) != _CHILD_SUCCESS_KEYS:
        return "worker_success_schema_keys_mismatch"
    if child.get("schema") != CHILD_SCHEMA:
        return "worker_schema_invalid"
    if child.get("status") != "ok" or child.get("reason") is not None:
        return "worker_reported_closed"
    for key, expected in _FALSE_AUTHORITY.items():
        if child.get(key) is not expected:
            return "worker_authority_flag_invalid:" + key
    for key in ("diagnostic_only", "candidate_only", "synthetic_only"):
        if child.get(key) is not True:
            return "worker_scope_flag_invalid:" + key
    for key in (
        "profile_name", "mode", "measurement_stage", "repeat_index",
        "order_index", "nonce",
        "config_sha256", "candidate_source_sha256",
        "absolute_deadline_monotonic_hex"
    ):
        if child.get(key) != config.get(key):
            return "worker_config_binding_mismatch:" + key
    entry_error = _validate_memory_sample(child.get("entry"), "entry")
    terminal_error = _validate_memory_sample(child.get("terminal"), "terminal")
    if entry_error or terminal_error:
        return entry_error or terminal_error
    entry = child["entry"]
    terminal = child["terminal"]
    try:
        entry_time = _finite_hex_float(
            entry["sampled_monotonic_hex"], name="entry_sample_time"
        )
        terminal_time = _finite_hex_float(
            terminal["sampled_monotonic_hex"], name="terminal_sample_time"
        )
        deadline = _finite_hex_float(
            config["absolute_deadline_monotonic_hex"], name="parent_deadline"
        )
    except ValueError:
        return "worker_sample_deadline_binding_invalid"
    if not entry_time <= terminal_time < deadline:
        return "worker_sample_order_or_deadline_invalid"
    if (
        type(child.get("rss_current_delta_bytes")) is not int
        or child["rss_current_delta_bytes"]
        != terminal["vmrss_bytes"] - entry["vmrss_bytes"]
        or type(child.get("rss_current_growth_bytes")) is not int
        or child["rss_current_growth_bytes"]
        != max(0, child["rss_current_delta_bytes"])
        or type(child.get("rss_peak_delta_bytes")) is not int
        or child["rss_peak_delta_bytes"]
        != terminal["vmhwm_bytes"] - entry["vmhwm_bytes"]
        or child["rss_peak_delta_bytes"] < 0
    ):
        return "worker_rss_delta_invalid"
    geometry = _profile_geometry(config["profile_name"])
    if child.get("geometry") != geometry:
        return "worker_geometry_invalid"
    mode = config["mode"]
    pair_count = geometry["pair_count"]
    forward_nnz = geometry["forward_nnz"]
    expected = {
        "source_rows": pair_count if mode == "range" else 2 * pair_count,
        "source_nnz": forward_nnz if mode == "range" else 2 * forward_nnz,
        "virtual_facet_rows": 2 * pair_count,
        "ranged_rows": pair_count if mode == "range" else 0,
        "fallback_pairs": 0 if mode == "range" else pair_count,
    }
    if any(child.get(key) != value for key, value in expected.items()):
        return "worker_structure_counts_invalid"
    if any(
        child.get(key) is not True
        for key in ("structure_complete", "replay_complete", "candidate_receipt_safe")
    ):
        return "worker_structure_or_replay_incomplete"
    if config["measurement_stage"] == "full_build_stream_replay":
        expected_batches = (2 * pair_count + STREAM_MAX_ROWS - 1) // STREAM_MAX_ROWS
        if (
            child.get("replay_kind") != "bounded_exact_stream"
            or child.get("fraction_membership_complete") is not True
            or type(child.get("stream_batch_count")) is not int
            or child["stream_batch_count"] != expected_batches
            or child.get("stream_max_rows") != STREAM_MAX_ROWS
            or type(child.get("stream_peak_batch_rows")) is not int
            or not 1 <= child["stream_peak_batch_rows"] <= STREAM_MAX_ROWS
        ):
            return "worker_stream_replay_contract_invalid"
    elif (
        child.get("replay_kind") != "legacy_full_expanded"
        or child.get("fraction_membership_complete") is not False
        or child.get("stream_batch_count") != 0
        or child.get("stream_max_rows") != 0
        or child.get("stream_peak_batch_rows") != 0
    ):
        return "worker_expanded_replay_contract_invalid"
    retained = child.get("retained_payload_bytes")
    if type(retained) is not int or not 0 < retained < HARD_RETAINED_PAYLOAD_CAP_BYTES:
        return "worker_retained_payload_invalid"
    for key in ("replay_sha256", "candidate_receipt_sha256"):
        value = child.get(key)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return "worker_digest_invalid:" + key
    cgroup_delta = child.get("cgroup_current_delta_bytes")
    entry_cgroup = entry["cgroup_v2"]
    terminal_cgroup = terminal["cgroup_v2"]
    expected_cgroup_delta = None
    if (
        entry_cgroup["available"]
        and terminal_cgroup["available"]
        and entry_cgroup["leaf"] == terminal_cgroup["leaf"]
    ):
        expected_cgroup_delta = (
            terminal_cgroup["current_bytes"] - entry_cgroup["current_bytes"]
        )
    if cgroup_delta != expected_cgroup_delta:
        return "worker_cgroup_delta_invalid"
    try:
        wall = _finite_hex_float(
            child.get("worker_stage_wall_seconds_hex"), name="worker_stage_wall"
        )
    except ValueError:
        return "worker_wall_time_invalid"
    if not math.isfinite(wall) or wall < 0 or wall >= HARD_WALL_SECONDS:
        return "worker_wall_time_outside_cap"
    return None


def _run_one_child(config: Mapping[str, Any]) -> dict[str, Any]:
    """Launch, monitor, cap, parse, and reap one fresh worker."""

    process: Any = None
    cleanup: Optional[dict[str, Any]] = None
    stop_reason: Optional[str] = None
    caught: Optional[BaseException] = None
    parent_peak_rss = 0
    stdout = b""
    stderr = b""
    exit_code: Optional[int] = None
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
            deadline = _finite_hex_float(
                config["absolute_deadline_monotonic_hex"],
                name="parent_monitor_deadline",
            )
            while process.poll() is None:
                rss = _read_status_bytes(process.pid, "VmRSS")
                if type(rss) is int:
                    parent_peak_rss = max(parent_peak_rss, rss)
                    if rss >= HARD_RSS_CAP_BYTES:
                        stop_reason = "worker_rss_hard_cap_reached"
                        break
                if os.fstat(stdout_file.fileno()).st_size > MAX_WORKER_STDOUT_BYTES:
                    stop_reason = "worker_stdout_cap_exceeded"
                    break
                if os.fstat(stderr_file.fileno()).st_size > MAX_WORKER_STDERR_BYTES:
                    stop_reason = "worker_stderr_cap_exceeded"
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "worker_absolute_deadline_exceeded"
                    break
                time.sleep(POLL_SECONDS)
            if stop_reason is None:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    exit_code = process.wait(timeout=min(1.0, remaining))
                except subprocess.TimeoutExpired:
                    stop_reason = "worker_reap_wait_deadline"
        except BaseException as exc:
            caught = exc
        finally:
            if process is not None and (
                caught is not None
                or stop_reason is not None
                or process.returncode is None
            ):
                cleanup = _cleanup_process_noexcept(process)
                if exit_code is None and type(cleanup.get("exit_code")) is int:
                    exit_code = cleanup["exit_code"]
            elif process is not None:
                cleanup = {
                    "termination_required": False,
                    "terminate_attempted": False,
                    "kill_attempted": False,
                    "reaped": True,
                    "exit_code": process.returncode,
                    "cleanup_error": None,
                }
            try:
                stdout_file.seek(0)
                stdout = stdout_file.read(MAX_WORKER_STDOUT_BYTES + 1)
                stderr_file.seek(0)
                stderr = stderr_file.read(MAX_WORKER_STDERR_BYTES + 1)
            except BaseException as exc:
                if caught is None:
                    caught = exc
    if isinstance(caught, (KeyboardInterrupt, SystemExit)):
        raise caught
    if caught is not None:
        return _closed(
            "worker_launch_or_monitor_error:" + _safe_exception_label(caught),
            worker_cleanup=cleanup,
        )
    if cleanup is None or cleanup.get("reaped") is not True:
        return _closed(
            "worker_cleanup_failed:worker_not_reaped",
            worker_cleanup=cleanup,
        )
    if stop_reason is not None:
        return _closed(stop_reason, worker_cleanup=cleanup)
    if len(stdout) > MAX_WORKER_STDOUT_BYTES or len(stderr) > MAX_WORKER_STDERR_BYTES:
        return _closed("worker_output_cap_exceeded_after_exit", worker_cleanup=cleanup)
    try:
        child = _strict_json_loads(stdout.decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ) as exc:
        return _closed(
            "worker_json_invalid:" + type(exc).__name__, worker_cleanup=cleanup
        )
    try:
        validation_error = _validate_child_success(child, config)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        return _closed(
            "worker_contract_validator_exception:"
            + _safe_exception_label(exc),
            worker_cleanup=cleanup,
        )
    if validation_error is not None:
        return _closed(validation_error, worker_cleanup=cleanup)
    if exit_code != 0:
        return _closed(
            "successful_worker_payload_with_nonzero_exit", worker_cleanup=cleanup
        )
    result = copy.deepcopy(child)
    result["parent_observed_peak_rss_bytes"] = parent_peak_rss
    result["worker_exit_code"] = exit_code
    result["worker_stderr"] = stderr[:4096].decode("utf-8", errors="replace")
    result["worker_cleanup"] = cleanup
    return _seal(result)


def _metric_summary(runs: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    values = [int(run[key]) for run in runs]
    return {"median_bytes": int(statistics.median(values)), "worst_bytes": max(values)}


def _summarize_mode(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cgroup_values = [
        run["cgroup_current_delta_bytes"]
        for run in runs
        if type(run.get("cgroup_current_delta_bytes")) is int
    ]
    return {
        "cold_repeats": len(runs),
        "entry_current_rss_bytes": _metric_summary(
            [{"value": run["entry"]["vmrss_bytes"]} for run in runs], "value"
        ),
        "terminal_current_rss_bytes": _metric_summary(
            [{"value": run["terminal"]["vmrss_bytes"]} for run in runs], "value"
        ),
        "entry_peak_rss_bytes": _metric_summary(
            [{"value": run["entry"]["vmhwm_bytes"]} for run in runs], "value"
        ),
        "terminal_peak_rss_bytes": _metric_summary(
            [{"value": run["terminal"]["vmhwm_bytes"]} for run in runs], "value"
        ),
        "current_rss_delta_bytes": _metric_summary(runs, "rss_current_delta_bytes"),
        "current_rss_growth_bytes": _metric_summary(runs, "rss_current_growth_bytes"),
        "peak_rss_delta_bytes": _metric_summary(runs, "rss_peak_delta_bytes"),
        "retained_payload_bytes": _metric_summary(runs, "retained_payload_bytes"),
        "parent_observed_peak_rss_bytes": _metric_summary(
            runs, "parent_observed_peak_rss_bytes"
        ),
        "cgroup_v2_current_delta_bytes": (
            {
                "available_for_all_repeats": True,
                "median_bytes": int(statistics.median(cgroup_values)),
                "worst_bytes": max(cgroup_values),
            }
            if len(cgroup_values) == len(runs)
            else {
                "available_for_all_repeats": False,
                "median_bytes": None,
                "worst_bytes": None,
            }
        ),
    }


def _stage_receipt(
    measurement_stage: str,
    stage_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped = {
        mode: [run for run in stage_runs if run["mode"] == mode]
        for mode in _MODES
    }
    summaries = {mode: _summarize_mode(grouped[mode]) for mode in _MODES}
    baseline_peak = summaries["dual_le"]["peak_rss_delta_bytes"]["worst_bytes"]
    range_peak = summaries["range"]["peak_rss_delta_bytes"]["worst_bytes"]
    baseline_payload_floor = min(
        run["retained_payload_bytes"] for run in grouped["dual_le"]
    )
    range_payload_worst = max(
        run["retained_payload_bytes"] for run in grouped["range"]
    )
    peak_ratio = range_peak / baseline_peak if baseline_peak > 0 else None
    payload_ratio = range_payload_worst / baseline_payload_floor
    expected_order = [
        "dual_le", "range", "range", "dual_le", "dual_le", "range"
    ]
    checks = {
        "three_cold_repeats_per_mode": all(
            len(grouped[mode]) == COLD_REPEATS for mode in _MODES
        ),
        "alternating_order_exact": [run["mode"] for run in stage_runs]
        == expected_order,
        "all_workers_reaped": all(
            run["worker_cleanup"].get("reaped") is True for run in stage_runs
        ),
        "all_structure_and_replay_complete": all(
            run["structure_complete"] is True
            and run["replay_complete"] is True
            and run["candidate_receipt_safe"] is True
            for run in stage_runs
        ),
        "dual_and_range_replay_digests_equal": len(
            {run["replay_sha256"] for run in stage_runs}
        )
        == 1,
        "stream_fraction_membership_complete": bool(
            measurement_stage != "full_build_stream_replay"
            or all(
                run["fraction_membership_complete"] is True
                and run["replay_kind"] == "bounded_exact_stream"
                for run in stage_runs
            )
        ),
        "baseline_worst_peak_delta_positive": baseline_peak > 0,
        "range_worst_peak_delta_at_most_0_80_baseline": bool(
            baseline_peak > 0 and range_peak <= 0.80 * baseline_peak
        ),
        "range_retained_payload_at_most_0_60_baseline": bool(
            payload_ratio <= 0.60
        ),
        "all_payloads_below_32_mib": all(
            run["retained_payload_bytes"] < HARD_RETAINED_PAYLOAD_CAP_BYTES
            for run in stage_runs
        ),
        "all_process_rss_below_512_mib": all(
            run["entry"]["vmrss_bytes"] < HARD_RSS_CAP_BYTES
            and run["entry"]["vmhwm_bytes"] < HARD_RSS_CAP_BYTES
            and run["terminal"]["vmrss_bytes"] < HARD_RSS_CAP_BYTES
            and run["terminal"]["vmhwm_bytes"] < HARD_RSS_CAP_BYTES
            and run["parent_observed_peak_rss_bytes"] < HARD_RSS_CAP_BYTES
            for run in stage_runs
        ),
    }
    diagnostic_checks_passed = all(checks.values())
    expanded_closed = measurement_stage == "full_build_replay"
    return {
        "measurement_stage": measurement_stage,
        "stage_gate_passed": bool(
            diagnostic_checks_passed and not expanded_closed
        ),
        "diagnostic_checks_passed": diagnostic_checks_passed,
        "promotion_gate_closed": expanded_closed,
        "candidate_only": True,
        "diagnostic_only": True,
        **_FALSE_AUTHORITY,
        "runs": list(stage_runs),
        "summaries": summaries,
        "range_to_dual_le_worst_peak_delta_ratio": peak_ratio,
        "range_to_dual_le_retained_payload_ratio": payload_ratio,
        "gate_checks": checks,
    }


def run_constraint_block_dag_memory_sentinel(
    *, absolute_deadline_monotonic: float
) -> dict[str, Any]:
    """Run the immutable fixed comparison; every failure returns ``closed``."""

    started = time.monotonic()
    try:
        if type(absolute_deadline_monotonic) is not float or not math.isfinite(
            absolute_deadline_monotonic
        ):
            raise ValueError("absolute_deadline_must_be_finite_builtin_float")
        if absolute_deadline_monotonic <= started:
            raise ValueError("absolute_deadline_must_be_in_the_future")
        # A later caller deadline is clamped, never honoured.  Thus no caller
        # can loosen the aggregate hard wall.
        deadline = min(absolute_deadline_monotonic, started + HARD_WALL_SECONDS)
        geometry = _profile_geometry(PROFILE_NAME)
        runs: list[dict[str, Any]] = []
        order_index = 0
        for measurement_stage in _STAGES:
            for repeat_index in range(COLD_REPEATS):
                order = (
                    _MODES
                    if repeat_index % 2 == 0
                    else tuple(reversed(_MODES))
                )
                for mode in order:
                    if time.monotonic() >= deadline:
                        return _closed(
                            "aggregate_absolute_deadline_exceeded",
                            absolute_deadline_monotonic_hex=deadline.hex(),
                            geometry=geometry,
                            runs=runs,
                        )
                    config = _make_worker_config(
                        profile_name=PROFILE_NAME,
                        mode=mode,
                        measurement_stage=measurement_stage,
                        repeat_index=repeat_index,
                        order_index=order_index,
                        nonce=secrets.token_hex(16),
                        absolute_deadline_monotonic=deadline,
                    )
                    run = _run_one_child(config)
                    if (
                        run.get("schema") != CHILD_SCHEMA
                        or run.get("status") != "ok"
                    ):
                        return _closed(
                            "fresh_worker_failed:"
                            + str(run.get("reason"))[:384],
                            absolute_deadline_monotonic_hex=deadline.hex(),
                            geometry=geometry,
                            runs=runs,
                            failed_run=run,
                        )
                    runs.append(run)
                    order_index += 1
        stage_receipts = {
            stage: _stage_receipt(
                stage,
                [run for run in runs if run["measurement_stage"] == stage],
            )
            for stage in _STAGES
        }
        aggregate_wall_ok = time.monotonic() - started < HARD_WALL_SECONDS
        source_stage_gate_passed = bool(
            stage_receipts["source_build_seal"]["stage_gate_passed"]
            and aggregate_wall_ok
        )
        stream_full_rss_gate_passed = bool(
            stage_receipts["full_build_stream_replay"]["stage_gate_passed"]
            and aggregate_wall_ok
        )
        # Promotion of this disconnected representation requires both the
        # source-stage and exact streaming full-stage gates.  Legacy expanded
        # replay remains diagnostic and explicitly closed regardless of its
        # observed ratio.
        rss_gate_passed = bool(
            source_stage_gate_passed and stream_full_rss_gate_passed
        )
        gate_checks = {
            "eighteen_fresh_workers_completed": len(runs)
            == len(_STAGES) * len(_MODES) * COLD_REPEATS,
            "aggregate_wall_below_20_seconds": aggregate_wall_ok,
            "source_stage_collection_complete": stage_receipts[
                "source_build_seal"
            ]["gate_checks"]["three_cold_repeats_per_mode"],
            "full_build_replay_collection_complete": stage_receipts[
                "full_build_replay"
            ]["gate_checks"]["three_cold_repeats_per_mode"],
            "full_build_stream_replay_collection_complete": stage_receipts[
                "full_build_stream_replay"
            ]["gate_checks"]["three_cold_repeats_per_mode"],
            "source_stage_prerequisite_passed": source_stage_gate_passed,
            "stream_full_prerequisite_passed": stream_full_rss_gate_passed,
            "legacy_full_expanded_gate_closed": True,
        }
        return _seal(
            {
                "schema": SCHEMA,
                "status": "rss_gate_passed" if rss_gate_passed else "closed",
                "reason": None if rss_gate_passed else "one_or_more_rss_gates_closed",
                "diagnostic_only": True,
                "candidate_only": True,
                "synthetic_only": True,
                "rss_gate_passed": rss_gate_passed,
                "source_stage_rss_gate_passed": source_stage_gate_passed,
                "stream_full_rss_gate_passed": stream_full_rss_gate_passed,
                "full_rss_gate_passed": False,
                "full_expanded_gate_closed": True,
                "production_promotion_claim": False,
                **_FALSE_AUTHORITY,
                "profile_name": PROFILE_NAME,
                "candidate_source_sha256": _CANDIDATE_SOURCE_SHA256,
                "geometry": geometry,
                "cold_repeats_required": COLD_REPEATS,
                "execution_order": [
                    {
                        "measurement_stage": run["measurement_stage"],
                        "mode": run["mode"],
                        "repeat_index": run["repeat_index"],
                    }
                    for run in runs
                ],
                "absolute_deadline_monotonic_hex": deadline.hex(),
                "hard_wall_seconds_hex": HARD_WALL_SECONDS.hex(),
                "hard_rss_cap_bytes": HARD_RSS_CAP_BYTES,
                "hard_retained_payload_cap_bytes": HARD_RETAINED_PAYLOAD_CAP_BYTES,
                "runs": runs,
                "summaries": {
                    stage: stage_receipts[stage]["summaries"]
                    for stage in _STAGES
                },
                "source_build_seal_receipt": stage_receipts[
                    "source_build_seal"
                ],
                "full_build_replay_receipt": stage_receipts[
                    "full_build_replay"
                ],
                "full_build_stream_replay_receipt": stage_receipts[
                    "full_build_stream_replay"
                ],
                "gate_checks": gate_checks,
                "aggregate_wall_seconds_hex": (time.monotonic() - started).hex(),
                "limitations": [
                    "fixed_deterministic_synthetic_coefficients_not_a_real_network",
                    "candidate_forced_dual_le_baseline_not_production_operator",
                    "vmhwm_is_process_lifetime_high_water_mark",
                    "cgroup_v2_current_is_optional_and_aggregate_not_process_local",
                    "allocator_and_kernel_sampling_can_make_small_rss_deltas_noisy",
                    "source_build_seal_terminal_precedes_replay_validation_"
                    "and_is_not_a_full_gate",
                    "full_build_replay_is_legacy_diagnostic_and_its_"
                    "promotion_gate_is_always_closed",
                    "overall_candidate_rss_prerequisite_requires_source_"
                    "and_full_stream_stage_gates",
                    "diagnostic_sha256_is_corruption_detection_not_authentication",
                ],
            }
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        return _closed("config_or_runner_error:" + _safe_exception_label(exc))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-seconds", type=float, default=HARD_WALL_SECONDS)
    parser.add_argument("--_worker-config-json", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker_config_json is not None:
        try:
            raw = _strict_json_loads(args._worker_config_json)
            result = _execute_worker(raw)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            result = _worker_failure({}, _safe_exception_label(exc))
        sys.stdout.write(
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        sys.stdout.flush()
        return 0 if result.get("status") == "ok" else 2
    if (
        type(args.deadline_seconds) is not float
        or not math.isfinite(args.deadline_seconds)
        or args.deadline_seconds <= 0
    ):
        result = _closed("deadline_seconds_must_be_finite_and_positive")
    else:
        # Longer CLI requests are still clamped by the public runner.
        result = run_constraint_block_dag_memory_sentinel(
            absolute_deadline_monotonic=time.monotonic()
            + min(args.deadline_seconds, HARD_WALL_SECONDS)
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("rss_gate_passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHILD_SCHEMA",
    "COLD_REPEATS",
    "EXPECTED_CANDIDATE_SOURCE_SHA256",
    "HARD_RETAINED_PAYLOAD_CAP_BYTES",
    "HARD_RSS_CAP_BYTES",
    "HARD_WALL_SECONDS",
    "PROFILE_NAME",
    "SCHEMA",
    "STREAM_MAX_ROWS",
    "run_constraint_block_dag_memory_sentinel",
    "verify_diagnostic_checksum",
]
