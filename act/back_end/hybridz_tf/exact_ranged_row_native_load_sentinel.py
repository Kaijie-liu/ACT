#!/usr/bin/env python3
"""Disconnected HiGHS loader sentinel for exact ranged-row compaction.

This module measures one narrow question: does loading one native ranged row
cost materially less than loading the two bitwise-opposite upper rows from
which it was replayed?  It is not wired into Operator-HZ, verification, or any
real-model runner.  The original :class:`SignedUpperSource` remains the only
replay source; a digest is diagnostic and is never identity or proof authority.

The synthetic generator preserves the C89 ADD-materialization proportions:
40,960 signed pairs, 52,359 continuous columns and 3,024,384 nonzeros in the
forward half (73.8375 per row).  A scale divisor makes the default sentinel
small.  Fixed row, column, nnz and payload caps prohibit the full C89 shape and
other large or real-model use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import statistics
import time
from types import MappingProxyType
from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_ranged_row_compaction import (
    ExactRangedRowCandidate,
    SignedUpperSource,
    fold_exact_signed_upper_pairs,
    validate_exact_ranged_candidate,
)

try:  # Optional native dependency; importing this disconnected module is safe.
    import highspy as _highspy
except ImportError:  # pragma: no cover - exercised only in environments without HiGHS
    _highspy = None


_C89_PAIR_COUNT = 40_960
_C89_CONTINUOUS_COLUMNS = 52_359
_C89_FORWARD_NNZ = 3_024_384
_C89_HIGH_NNZ_ROWS = 34_304  # 34,304*74 + 6,656*73 = 3,024,384

_MAX_SOURCE_ROWS = 16_384
_MAX_COLUMNS = 16_384
_MAX_SOURCE_NNZ = 1_500_000
_MAX_PAYLOAD_BYTES = 96 * 1024 * 1024
_MIN_BENCHMARK_PAIRS = 512
_MAX_BENCHMARK_WARMUPS = 31
_MAX_BENCHMARK_REPEATS = 31
_NATIVE_LOADER_SPEEDUP_GATE = 1.5
_C89_FRAME_FACTORY_TOKEN = object()
_C89_METADATA_KEYS = frozenset(
    {
        "schema",
        "source_sha256",
        "synthetic_only",
        "real_model_allowed",
        "large_model_allowed",
        "scale_divisor",
        "c89_reference_pair_count",
        "c89_reference_columns",
        "c89_reference_forward_nnz",
        "pair_count",
        "source_rows",
        "columns",
        "source_constraint_nnz",
        "source_payload_bytes",
        "row_cap",
        "column_cap",
        "nnz_cap",
        "payload_cap_bytes",
        "triangle_relaxation_called",
        "branch_and_bound_called",
        "backward_called",
        "dual_called",
        "proof_authority",
        "verdict_authority",
        "production_integration",
    }
)
_C89_METADATA_FALSE_KEYS = frozenset(
    {
        "real_model_allowed",
        "large_model_allowed",
        "triangle_relaxation_called",
        "branch_and_bound_called",
        "backward_called",
        "dual_called",
        "proof_authority",
        "verdict_authority",
        "production_integration",
    }
)
_C89_METADATA_INT_KEYS = frozenset(
    {
        "scale_divisor",
        "c89_reference_pair_count",
        "c89_reference_columns",
        "c89_reference_forward_nnz",
        "pair_count",
        "source_rows",
        "columns",
        "source_constraint_nnz",
        "source_payload_bytes",
        "row_cap",
        "column_cap",
        "nnz_cap",
        "payload_cap_bytes",
    }
)


class ExactRangedNativeLoadSentinelError(RuntimeError):
    """Fail-closed error raised only by this disconnected sentinel."""


@dataclass(frozen=True)
class _C89FrameFactorySeal:
    token: Any
    source_sha256: str
    metadata_sha256: str


def _metadata_sha256(value: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ExactRangedNativeLoadSentinelError(
            "c89_ratio_metadata_is_not_canonical_json"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _exact_nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ExactRangedNativeLoadSentinelError(
            f"{name}_must_be_a_{qualifier}_builtin_int"
        )
    return value


def _exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ExactRangedNativeLoadSentinelError(f"{name}_must_be_a_builtin_bool")
    return value


def _readonly_float64_vector(value: Any, width: int, name: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.ndim != 1
        or value.size != width
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise ExactRangedNativeLoadSentinelError(
            f"{name}_must_be_a_finite_contiguous_exact_float64_vector"
        )
    out = np.frombuffer(value.tobytes(order="C"), dtype=np.float64)
    out.setflags(write=False)
    return out


def _payload_bytes(matrix: sp.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _require_source_caps(source: SignedUpperSource) -> None:
    if type(source) is not SignedUpperSource:
        raise ExactRangedNativeLoadSentinelError("source_has_wrong_exact_type")
    rows = int(source.A_cont.shape[0])
    columns = int(source.A_cont.shape[1] + source.A_bin.shape[1])
    nnz = int(source.A_cont.nnz + source.A_bin.nnz)
    payload = _payload_bytes(source.A_cont) + _payload_bytes(source.A_bin)
    if rows > _MAX_SOURCE_ROWS:
        raise ExactRangedNativeLoadSentinelError("source_row_cap_exceeded")
    if columns > _MAX_COLUMNS:
        raise ExactRangedNativeLoadSentinelError("source_column_cap_exceeded")
    if nnz > _MAX_SOURCE_NNZ:
        raise ExactRangedNativeLoadSentinelError("source_nnz_cap_exceeded")
    if payload > _MAX_PAYLOAD_BYTES:
        raise ExactRangedNativeLoadSentinelError("source_payload_cap_exceeded")


@dataclass(frozen=True)
class C89RatioSyntheticFrame:
    """Bounded synthetic source and continuous audit-domain column bounds."""

    source: SignedUpperSource
    column_lower: np.ndarray
    column_upper: np.ndarray
    objective: np.ndarray
    metadata: Any
    _factory_seal: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self._factory_seal) is not _C89FrameFactorySeal
            or self._factory_seal.token is not _C89_FRAME_FACTORY_TOKEN
        ):
            raise ExactRangedNativeLoadSentinelError(
                "c89_ratio_frame_must_come_from_factory"
            )
        if type(self.source) is not SignedUpperSource:
            raise ExactRangedNativeLoadSentinelError("source_has_wrong_exact_type")
        try:
            source = SignedUpperSource(
                self.source.A_cont,
                self.source.A_bin,
                self.source.upper,
                tuple(self.source.row_tags),
                source_sha256=self.source.source_sha256,
                schema=self.source.schema,
            )
        except (AttributeError, TypeError, ValueError, OverflowError, MemoryError) as exc:
            raise ExactRangedNativeLoadSentinelError(
                "c89_ratio_source_snapshot_failed"
            ) from exc
        _require_source_caps(source)
        columns = int(source.A_cont.shape[1] + source.A_bin.shape[1])
        lower = _readonly_float64_vector(
            self.column_lower, columns, "column_lower"
        )
        upper = _readonly_float64_vector(
            self.column_upper, columns, "column_upper"
        )
        objective = _readonly_float64_vector(self.objective, columns, "objective")
        if (
            not np.array_equal(lower.view(np.uint64), np.full(columns, -1.0).view(np.uint64))
            or not np.array_equal(upper.view(np.uint64), np.full(columns, 1.0).view(np.uint64))
            or not np.array_equal(objective.view(np.uint64), np.zeros(columns).view(np.uint64))
        ):
            raise ExactRangedNativeLoadSentinelError(
                "c89_ratio_audit_domain_vectors_are_invalid"
            )
        metadata = _validated_c89_metadata(source, self.metadata)
        if (
            self._factory_seal.source_sha256 != source.source_sha256
            or self._factory_seal.metadata_sha256 != _metadata_sha256(metadata)
        ):
            raise ExactRangedNativeLoadSentinelError(
                "c89_ratio_factory_binding_is_stale"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "column_lower", lower)
        object.__setattr__(self, "column_upper", upper)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


def _validated_c89_metadata(
    source: SignedUpperSource, value: Any
) -> Dict[str, Any]:
    if type(value) is MappingProxyType:
        metadata = dict(value)
    elif type(value) is dict:
        metadata = dict(value)
    else:
        raise ExactRangedNativeLoadSentinelError(
            "c89_ratio_metadata_has_wrong_exact_type"
        )
    if (
        not all(type(key) is str for key in metadata)
        or frozenset(metadata) != _C89_METADATA_KEYS
        or type(metadata.get("schema")) is not str
        or metadata["schema"] != "act.forward_exact.c89_ratio_synthetic.v1"
        or type(metadata.get("source_sha256")) is not str
        or metadata["source_sha256"] != source.source_sha256
        or type(metadata.get("synthetic_only")) is not bool
        or metadata["synthetic_only"] is not True
        or any(
            type(metadata.get(key)) is not bool or metadata[key] is not False
            for key in _C89_METADATA_FALSE_KEYS
        )
        or any(
            type(metadata.get(key)) is not int or metadata[key] < 0
            for key in _C89_METADATA_INT_KEYS
        )
        or metadata["scale_divisor"] <= 0
    ):
        raise ExactRangedNativeLoadSentinelError(
            "c89_ratio_metadata_schema_or_type_is_invalid"
        )
    divisor = metadata["scale_divisor"]
    pair_count = (_C89_PAIR_COUNT + divisor - 1) // divisor
    columns = max(
        (_C89_CONTINUOUS_COLUMNS + divisor - 1) // divisor,
        74,
    )
    high_width_rows = (pair_count * _C89_HIGH_NNZ_ROWS) // _C89_PAIR_COUNT
    forward_nnz = pair_count * 73 + high_width_rows
    expected = {
        "c89_reference_pair_count": _C89_PAIR_COUNT,
        "c89_reference_columns": _C89_CONTINUOUS_COLUMNS,
        "c89_reference_forward_nnz": _C89_FORWARD_NNZ,
        "pair_count": pair_count,
        "source_rows": 2 * pair_count,
        "columns": columns,
        "source_constraint_nnz": 2 * forward_nnz,
        "source_payload_bytes": (
            _payload_bytes(source.A_cont) + _payload_bytes(source.A_bin)
        ),
        "row_cap": _MAX_SOURCE_ROWS,
        "column_cap": _MAX_COLUMNS,
        "nnz_cap": _MAX_SOURCE_NNZ,
        "payload_cap_bytes": _MAX_PAYLOAD_BYTES,
    }
    if any(metadata[key] != expected_value for key, expected_value in expected.items()):
        raise ExactRangedNativeLoadSentinelError(
            "c89_ratio_metadata_formula_is_invalid"
        )
    if (
        source.A_cont.shape != (2 * pair_count, columns)
        or source.A_bin.shape != (2 * pair_count, 0)
        or source.A_bin.nnz != 0
        or source.A_cont.nnz != 2 * forward_nnz
        or source.upper.size != 2 * pair_count
        or not np.array_equal(
            source.upper.view(np.uint64),
            np.full(2 * pair_count, 8.0, dtype=np.float64).view(np.uint64),
        )
        or source.row_tags
        != tuple(
            ["c89_ratio:add_materialize:forward"] * pair_count
            + ["c89_ratio:add_materialize:reverse"] * pair_count
        )
    ):
        raise ExactRangedNativeLoadSentinelError(
            "c89_ratio_source_shape_or_payload_is_invalid"
        )
    return metadata


def _snapshot_c89_frame(value: Any) -> C89RatioSyntheticFrame:
    if type(value) is not C89RatioSyntheticFrame:
        raise ExactRangedNativeLoadSentinelError("frame_has_wrong_exact_type")
    # Do not treat the carried seal or module-private token as provenance: a
    # hostile Python caller can import them and use object.__setattr__.  First
    # make strict snapshots of the live payload while deliberately ignoring the
    # seal, then rebuild the unique deterministic factory frame selected by the
    # sole scale_divisor and compare every payload bit.
    try:
        if type(value.source) is not SignedUpperSource:
            raise ExactRangedNativeLoadSentinelError("source_has_wrong_exact_type")
        source = SignedUpperSource(
            value.source.A_cont,
            value.source.A_bin,
            value.source.upper,
            tuple(value.source.row_tags),
            source_sha256=value.source.source_sha256,
            schema=value.source.schema,
        )
        _require_source_caps(source)
        columns = int(source.A_cont.shape[1] + source.A_bin.shape[1])
        lower = _readonly_float64_vector(value.column_lower, columns, "column_lower")
        upper = _readonly_float64_vector(value.column_upper, columns, "column_upper")
        objective = _readonly_float64_vector(value.objective, columns, "objective")
        metadata = _validated_c89_metadata(source, value.metadata)
        expected = make_c89_ratio_signed_upper_source(
            scale_divisor=metadata["scale_divisor"]
        )
    except ExactRangedNativeLoadSentinelError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise ExactRangedNativeLoadSentinelError(
            "c89_ratio_live_payload_snapshot_failed"
        ) from exc
    if not _c89_frame_payload_bitwise_equal(
        source=source,
        lower=lower,
        upper=upper,
        objective=objective,
        metadata=metadata,
        expected=expected,
    ):
        raise ExactRangedNativeLoadSentinelError(
            "c89_ratio_payload_does_not_match_deterministic_factory"
        )
    return expected


def _csr_payload_bitwise_equal(left: Any, right: Any) -> bool:
    return bool(
        type(left) is sp.csr_matrix
        and type(right) is sp.csr_matrix
        and left.shape == right.shape
        and left.dtype == right.dtype == np.dtype(np.float64)
        and left.indices.dtype == right.indices.dtype == np.dtype(np.int32)
        and left.indptr.dtype == right.indptr.dtype == np.dtype(np.int32)
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data.view(np.uint64), right.data.view(np.uint64))
    )


def _c89_frame_payload_bitwise_equal(
    *,
    source: SignedUpperSource,
    lower: np.ndarray,
    upper: np.ndarray,
    objective: np.ndarray,
    metadata: Dict[str, Any],
    expected: C89RatioSyntheticFrame,
) -> bool:
    return bool(
        source.schema == expected.source.schema
        and source.source_sha256 == expected.source.source_sha256
        and source.row_tags == expected.source.row_tags
        and _csr_payload_bitwise_equal(source.A_cont, expected.source.A_cont)
        and _csr_payload_bitwise_equal(source.A_bin, expected.source.A_bin)
        and np.array_equal(
            source.upper.view(np.uint64), expected.source.upper.view(np.uint64)
        )
        and np.array_equal(
            lower.view(np.uint64), expected.column_lower.view(np.uint64)
        )
        and np.array_equal(
            upper.view(np.uint64), expected.column_upper.view(np.uint64)
        )
        and np.array_equal(
            objective.view(np.uint64), expected.objective.view(np.uint64)
        )
        and json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        == json.dumps(
            dict(expected.metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def make_c89_ratio_signed_upper_source(
    *, scale_divisor: int = 40
) -> C89RatioSyntheticFrame:
    """Create a capped synthetic source with C89 ADD row/column/nnz ratios.

    ``scale_divisor=1`` would be the measured C89 exact0 shape and is rejected
    by the fixed caps.  This function has no path accepting ONNX, image, dataset
    or network objects.
    """

    divisor = _exact_nonnegative_int(scale_divisor, "scale_divisor", positive=True)
    pair_count = (_C89_PAIR_COUNT + divisor - 1) // divisor
    columns = (_C89_CONTINUOUS_COLUMNS + divisor - 1) // divisor
    columns = max(columns, 74)
    source_rows = 2 * pair_count
    if source_rows > _MAX_SOURCE_ROWS:
        raise ExactRangedNativeLoadSentinelError("source_row_cap_exceeded")
    if columns > _MAX_COLUMNS:
        raise ExactRangedNativeLoadSentinelError("source_column_cap_exceeded")

    # Bresenham-style scaling preserves the exact 73/74 average without random
    # state.  Each row has strictly increasing, unique canonical indices.
    forward_indptr = np.empty(pair_count + 1, dtype=np.int32)
    forward_indptr[0] = 0
    row_widths = np.empty(pair_count, dtype=np.int32)
    for row in range(pair_count):
        before = (row * _C89_HIGH_NNZ_ROWS) // _C89_PAIR_COUNT
        after = ((row + 1) * _C89_HIGH_NNZ_ROWS) // _C89_PAIR_COUNT
        row_widths[row] = 74 if after > before else 73
    np.cumsum(row_widths, dtype=np.int32, out=forward_indptr[1:])
    forward_nnz = int(forward_indptr[-1])
    if 2 * forward_nnz > _MAX_SOURCE_NNZ:
        raise ExactRangedNativeLoadSentinelError("source_nnz_cap_exceeded")

    indices = np.empty(forward_nnz, dtype=np.int32)
    data = np.empty(forward_nnz, dtype=np.float64)
    dyadic = np.asarray((0.5, -0.25, 0.125, -0.0625), dtype=np.float64)
    for row in range(pair_count):
        width = int(row_widths[row])
        begin = int(forward_indptr[row])
        end = int(forward_indptr[row + 1])
        start_column = (row * 97) % (columns - width + 1)
        indices[begin:end] = np.arange(
            start_column, start_column + width, dtype=np.int32
        )
        data[begin:end] = dyadic[(row + np.arange(width)) & 3]
    forward = sp.csr_matrix(
        (data, indices, forward_indptr), shape=(pair_count, columns), copy=False
    )
    signed = sp.vstack((forward, -forward), format="csr")
    signed.sort_indices()
    signed_upper = np.concatenate(
        (
            np.full(pair_count, 8.0, dtype=np.float64),
            np.full(pair_count, 8.0, dtype=np.float64),
        )
    )
    tags = tuple(
        ["c89_ratio:add_materialize:forward"] * pair_count
        + ["c89_ratio:add_materialize:reverse"] * pair_count
    )
    source = SignedUpperSource(
        A_cont=signed,
        A_bin=sp.csr_matrix((source_rows, 0), dtype=np.float64),
        upper=signed_upper,
        row_tags=tags,
    )
    _require_source_caps(source)
    lower = np.full(columns, -1.0, dtype=np.float64)
    upper = np.full(columns, 1.0, dtype=np.float64)
    objective = np.zeros(columns, dtype=np.float64)
    for value in (lower, upper, objective):
        value.setflags(write=False)
    metadata: Dict[str, Any] = {
        "schema": "act.forward_exact.c89_ratio_synthetic.v1",
        "source_sha256": source.source_sha256,
        "synthetic_only": True,
        "real_model_allowed": False,
        "large_model_allowed": False,
        "scale_divisor": divisor,
        "c89_reference_pair_count": _C89_PAIR_COUNT,
        "c89_reference_columns": _C89_CONTINUOUS_COLUMNS,
        "c89_reference_forward_nnz": _C89_FORWARD_NNZ,
        "pair_count": pair_count,
        "source_rows": source_rows,
        "columns": columns,
        "source_constraint_nnz": int(source.A_cont.nnz),
        "source_payload_bytes": (
            _payload_bytes(source.A_cont) + _payload_bytes(source.A_bin)
        ),
        "row_cap": _MAX_SOURCE_ROWS,
        "column_cap": _MAX_COLUMNS,
        "nnz_cap": _MAX_SOURCE_NNZ,
        "payload_cap_bytes": _MAX_PAYLOAD_BYTES,
        "triangle_relaxation_called": False,
        "branch_and_bound_called": False,
        "backward_called": False,
        "dual_called": False,
        "proof_authority": False,
        "verdict_authority": False,
        "production_integration": False,
    }
    seal = _C89FrameFactorySeal(
        token=_C89_FRAME_FACTORY_TOKEN,
        source_sha256=source.source_sha256,
        metadata_sha256=_metadata_sha256(metadata),
    )
    return C89RatioSyntheticFrame(
        source=source,
        column_lower=lower,
        column_upper=upper,
        objective=objective,
        metadata=metadata,
        _factory_seal=seal,
    )


def _combined_csr(A_cont: sp.csr_matrix, A_bin: sp.csr_matrix) -> sp.csr_matrix:
    if A_bin.shape[1] == 0:
        return A_cont
    matrix = sp.hstack((A_cont, A_bin), format="csr")
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.indices.dtype != np.dtype(np.int32) or matrix.indptr.dtype != np.dtype(np.int32):
        raise ExactRangedNativeLoadSentinelError("combined_csr_requires_int32_indices")
    return matrix


def _require_live_canonical_csr(matrix: Any) -> sp.csr_matrix:
    if (
        type(matrix) is not sp.csr_matrix
        or matrix.dtype != np.dtype(np.float64)
        or matrix.ndim != 2
        or type(matrix.data) is not np.ndarray
        or type(matrix.indices) is not np.ndarray
        or type(matrix.indptr) is not np.ndarray
        or matrix.data.dtype != np.dtype(np.float64)
        or matrix.indices.dtype != np.dtype(np.int32)
        or matrix.indptr.dtype != np.dtype(np.int32)
        or matrix.data.ndim != 1
        or matrix.indices.ndim != 1
        or matrix.indptr.ndim != 1
        or not matrix.data.flags.c_contiguous
        or not matrix.indices.flags.c_contiguous
        or not matrix.indptr.flags.c_contiguous
        or matrix.indptr.size != matrix.shape[0] + 1
        or int(matrix.indptr[0]) != 0
        or np.any(matrix.indptr[1:] < matrix.indptr[:-1])
        or int(matrix.indptr[-1]) != matrix.data.size
        or matrix.indices.size != matrix.data.size
        or (
            matrix.indices.size
            and (
                np.any(matrix.indices < 0)
                or np.any(matrix.indices >= matrix.shape[1])
            )
        )
        or not matrix.has_canonical_format
        or not matrix.has_sorted_indices
        or (matrix.nnz and not np.all(np.isfinite(matrix.data)))
        or (matrix.nnz and np.any(matrix.data == 0.0))
    ):
        raise ExactRangedNativeLoadSentinelError(
            "matrix_must_be_finite_canonical_float64_csr_with_int32_indices"
        )
    for row in range(matrix.shape[0]):
        begin = int(matrix.indptr[row])
        end = int(matrix.indptr[row + 1])
        if end - begin > 1 and np.any(
            matrix.indices[begin + 1 : end] <= matrix.indices[begin : end - 1]
        ):
            raise ExactRangedNativeLoadSentinelError(
                "matrix_row_indices_must_be_strictly_increasing"
            )
    if matrix.shape[0] > _MAX_SOURCE_ROWS:
        raise ExactRangedNativeLoadSentinelError("matrix_row_cap_exceeded")
    if matrix.shape[1] > _MAX_COLUMNS:
        raise ExactRangedNativeLoadSentinelError("matrix_column_cap_exceeded")
    if matrix.nnz > _MAX_SOURCE_NNZ:
        raise ExactRangedNativeLoadSentinelError("matrix_nnz_cap_exceeded")
    if _payload_bytes(matrix) > _MAX_PAYLOAD_BYTES:
        raise ExactRangedNativeLoadSentinelError("matrix_payload_cap_exceeded")
    return matrix


@dataclass(frozen=True)
class NativeFrameLoadResult:
    rows: int
    constraint_nnz: int
    native_row_load_seconds: float
    model_status: str
    objective_value: float | None
    primal: np.ndarray
    receipt: Any


def _new_highs() -> Any:
    if _highspy is None:
        raise ExactRangedNativeLoadSentinelError("highspy_backend_unavailable")
    return _highspy.Highs()


def _require_highs_ok(status: Any, operation: str) -> None:
    if _highspy is None or status != _highspy.HighsStatus.kOk:
        raise ExactRangedNativeLoadSentinelError(
            f"highs_{operation}_failed:{status}"
        )


def _load_native_frame(
    *,
    matrix: sp.csr_matrix,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    column_lower: np.ndarray,
    column_upper: np.ndarray,
    objective: np.ndarray,
    solve: bool,
    route: str,
) -> NativeFrameLoadResult:
    solve = _exact_bool(solve, "solve")
    if type(route) is not str or route not in ("baseline_upper", "candidate_range"):
        raise ExactRangedNativeLoadSentinelError("route_is_invalid")
    matrix = _require_live_canonical_csr(matrix)
    columns = int(matrix.shape[1])
    column_lower = _readonly_float64_vector(column_lower, columns, "column_lower")
    column_upper = _readonly_float64_vector(column_upper, columns, "column_upper")
    objective = _readonly_float64_vector(objective, columns, "objective")
    if np.any(column_lower > column_upper):
        raise ExactRangedNativeLoadSentinelError("column_bounds_are_contradictory")
    if (
        type(row_lower) is not np.ndarray
        or type(row_upper) is not np.ndarray
        or row_lower.dtype != np.dtype(np.float64)
        or row_upper.dtype != np.dtype(np.float64)
        or row_lower.ndim != 1
        or row_upper.ndim != 1
        or row_lower.size != matrix.shape[0]
        or row_upper.size != matrix.shape[0]
        or not row_lower.flags.c_contiguous
        or not row_upper.flags.c_contiguous
        or np.any(np.isnan(row_lower))
        or np.any(np.isposinf(row_lower))
        or np.any(np.isnan(row_upper))
        or np.any(np.isneginf(row_upper))
        or np.any(row_lower > row_upper)
    ):
        raise ExactRangedNativeLoadSentinelError("row_bounds_are_malformed")

    highs = None
    pending: BaseException | None = None
    result: NativeFrameLoadResult | None = None
    try:
        highs = _new_highs()
        _require_highs_ok(highs.setOptionValue("output_flag", False), "set_output_flag")
        _require_highs_ok(highs.setOptionValue("threads", 1), "set_threads")
        empty_indices = np.empty(0, dtype=np.int32)
        empty_values = np.empty(0, dtype=np.float64)
        empty_starts = np.zeros(columns + 1, dtype=np.int32)
        _require_highs_ok(
            highs.addCols(
                columns,
                objective,
                column_lower,
                column_upper,
                0,
                empty_starts,
                empty_indices,
                empty_values,
            ),
            "add_columns",
        )
        load_started = time.perf_counter_ns()
        status = highs.addRows(
            matrix.shape[0],
            row_lower,
            row_upper,
            matrix.nnz,
            matrix.indptr,
            matrix.indices,
            matrix.data,
        )
        load_seconds = (time.perf_counter_ns() - load_started) * 1.0e-9
        _require_highs_ok(status, "add_rows")
        model_status = "not_run"
        objective_value = None
        primal = np.empty(0, dtype=np.float64)
        if solve:
            _require_highs_ok(highs.run(), "run")
            model_status = str(highs.getModelStatus())
            objective_value = float(highs.getObjectiveValue())
            primal = np.asarray(highs.getSolution().col_value, dtype=np.float64)
            if primal.ndim != 1 or primal.size != columns or not np.all(np.isfinite(primal)):
                raise ExactRangedNativeLoadSentinelError("highs_primal_is_malformed")
        primal = np.array(primal, dtype=np.float64, copy=True)
        primal.setflags(write=False)
        receipt = MappingProxyType(
            {
                "schema": "act.forward_exact.native_ranged_load.v1",
                "route": route,
                "candidate_only": True,
                "native_solver_run": solve,
                "continuous_audit_columns_only": True,
                "integrality_loaded": False,
                "triangle_relaxation_called": False,
                "branch_and_bound_called": False,
                "backward_called": False,
                "dual_called": False,
                "proof_authority": False,
                "verdict_authority": False,
                "production_integration": False,
            }
        )
        result = NativeFrameLoadResult(
            rows=int(matrix.shape[0]),
            constraint_nnz=int(matrix.nnz),
            native_row_load_seconds=float(load_seconds),
            model_status=model_status,
            objective_value=objective_value,
            primal=primal,
            receipt=receipt,
        )
    except BaseException as exc:  # clear native memory even for interrupts/system exit
        pending = exc
    finally:
        if highs is not None:
            try:
                close_status = highs.clear()
                _require_highs_ok(close_status, "clear")
            except BaseException as close_exc:
                if pending is None:
                    pending = ExactRangedNativeLoadSentinelError(
                        f"native_model_close_failed:{type(close_exc).__name__}"
                    )
    if pending is not None:
        if type(pending) is ExactRangedNativeLoadSentinelError:
            raise pending
        raise ExactRangedNativeLoadSentinelError(
            f"native_model_operation_failed:{type(pending).__name__}"
        ) from pending
    if result is None:  # defensive; all successful routes assign a result
        raise ExactRangedNativeLoadSentinelError("native_model_result_missing")
    return result


@dataclass(frozen=True)
class NativeEquivalenceResult:
    source_sha256: str
    candidate_sha256: str
    folding_seconds: float
    baseline: NativeFrameLoadResult
    candidate: NativeFrameLoadResult
    receipt: Any


def run_native_ranged_equivalence_sentinel(
    *,
    source: SignedUpperSource,
    column_lower: np.ndarray,
    column_upper: np.ndarray,
    objective: np.ndarray,
) -> NativeEquivalenceResult:
    """Replay, load and solve both exact encodings in continuous audit space."""

    _require_source_caps(source)
    source_digest = source.source_sha256
    fold_started = time.perf_counter_ns()
    candidate = fold_exact_signed_upper_pairs(source)
    folding_seconds = (time.perf_counter_ns() - fold_started) * 1.0e-9
    if not validate_exact_ranged_candidate(source, candidate):
        raise ExactRangedNativeLoadSentinelError("candidate_replay_failed")
    if source.source_sha256 != source_digest:
        raise ExactRangedNativeLoadSentinelError("source_changed_during_replay")
    baseline_matrix = _combined_csr(source.A_cont, source.A_bin)
    candidate_matrix = _combined_csr(candidate.A_cont, candidate.A_bin)
    source_lower = np.full(source.upper.size, -_highs_inf(), dtype=np.float64)
    baseline = _load_native_frame(
        matrix=baseline_matrix,
        row_lower=source_lower,
        row_upper=source.upper,
        column_lower=column_lower,
        column_upper=column_upper,
        objective=objective,
        solve=True,
        route="baseline_upper",
    )
    compact = _load_native_frame(
        matrix=candidate_matrix,
        row_lower=candidate.lower,
        row_upper=candidate.upper,
        column_lower=column_lower,
        column_upper=column_upper,
        objective=objective,
        solve=True,
        route="candidate_range",
    )
    receipt = MappingProxyType(
        {
            "schema": "act.forward_exact.native_ranged_equivalence.v1",
            "source_frame_required_for_replay": True,
            "source_frame_retained_outside_candidate": True,
            "source_rows": baseline.rows,
            "candidate_rows": compact.rows,
            "source_constraint_nnz": baseline.constraint_nnz,
            "candidate_constraint_nnz": compact.constraint_nnz,
            "candidate_only": True,
            "real_model_run": False,
            "large_model_run": False,
            "triangle_relaxation_called": False,
            "branch_and_bound_called": False,
            "backward_called": False,
            "dual_called": False,
            "proof_authority": False,
            "verdict_authority": False,
            "production_integration": False,
        }
    )
    return NativeEquivalenceResult(
        source_sha256=source_digest,
        candidate_sha256=candidate.candidate_sha256,
        folding_seconds=float(folding_seconds),
        baseline=baseline,
        candidate=compact,
        receipt=receipt,
    )


def _highs_inf() -> float:
    if _highspy is None:
        raise ExactRangedNativeLoadSentinelError("highspy_backend_unavailable")
    return float(_highspy.kHighsInf)


@dataclass(frozen=True)
class NativeLoadBenchmarkResult:
    warmup_pairs: int
    measured_pairs: int
    folding_preprocess_median_seconds: float
    baseline_native_load_median_seconds: float
    candidate_native_load_median_seconds: float
    candidate_total_median_seconds: float
    native_loader_paired_median_speedup: float
    total_paired_median_speedup: float
    folding_to_candidate_load_median_ratio: float
    native_loader_speedup_gate: float
    native_loader_candidate_supported: bool
    receipt: Any


def benchmark_native_ranged_row_loader(
    frame: C89RatioSyntheticFrame,
    *,
    warmup_pairs: int = 2,
    measured_pairs: int = 7,
) -> NativeLoadBenchmarkResult:
    """Run single-thread paired load-only timings; no solver ``run`` is called."""

    frame = _snapshot_c89_frame(frame)
    warmups = _exact_nonnegative_int(warmup_pairs, "warmup_pairs")
    repeats = _exact_nonnegative_int(measured_pairs, "measured_pairs", positive=True)
    if warmups < 2:
        raise ExactRangedNativeLoadSentinelError("at_least_two_warmup_pairs_required")
    if warmups > _MAX_BENCHMARK_WARMUPS:
        raise ExactRangedNativeLoadSentinelError("warmup_pair_cap_exceeded")
    if repeats < 7:
        raise ExactRangedNativeLoadSentinelError("at_least_seven_measured_pairs_required")
    if repeats > _MAX_BENCHMARK_REPEATS:
        raise ExactRangedNativeLoadSentinelError("measured_pair_cap_exceeded")
    _require_source_caps(frame.source)
    if frame.source.upper.size // 2 < _MIN_BENCHMARK_PAIRS:
        raise ExactRangedNativeLoadSentinelError("benchmark_pair_floor_not_met")

    folding: list[float] = []
    baseline_load: list[float] = []
    candidate_load: list[float] = []
    total: list[float] = []
    native_speedups: list[float] = []
    total_speedups: list[float] = []
    fold_ratios: list[float] = []
    source_digest = frame.source.source_sha256
    for pair_index in range(warmups + repeats):
        fold_started = time.perf_counter_ns()
        candidate = fold_exact_signed_upper_pairs(frame.source)
        fold_seconds = (time.perf_counter_ns() - fold_started) * 1.0e-9
        if not validate_exact_ranged_candidate(frame.source, candidate):
            raise ExactRangedNativeLoadSentinelError("candidate_replay_failed")
        if frame.source.source_sha256 != source_digest:
            raise ExactRangedNativeLoadSentinelError("source_changed_during_benchmark")
        source_matrix = _combined_csr(frame.source.A_cont, frame.source.A_bin)
        compact_matrix = _combined_csr(candidate.A_cont, candidate.A_bin)
        source_lower = np.full(frame.source.upper.size, -_highs_inf(), dtype=np.float64)

        def load_baseline() -> NativeFrameLoadResult:
            return _load_native_frame(
                matrix=source_matrix,
                row_lower=source_lower,
                row_upper=frame.source.upper,
                column_lower=frame.column_lower,
                column_upper=frame.column_upper,
                objective=frame.objective,
                solve=False,
                route="baseline_upper",
            )

        def load_candidate() -> NativeFrameLoadResult:
            return _load_native_frame(
                matrix=compact_matrix,
                row_lower=candidate.lower,
                row_upper=candidate.upper,
                column_lower=frame.column_lower,
                column_upper=frame.column_upper,
                objective=frame.objective,
                solve=False,
                route="candidate_range",
            )

        # Alternate within-pair order to reduce systematic first/second bias.
        if pair_index & 1:
            compact_result = load_candidate()
            baseline_result = load_baseline()
        else:
            baseline_result = load_baseline()
            compact_result = load_candidate()
        if pair_index < warmups:
            continue
        b = baseline_result.native_row_load_seconds
        c = compact_result.native_row_load_seconds
        if b <= 0.0 or c <= 0.0 or fold_seconds <= 0.0:
            raise ExactRangedNativeLoadSentinelError("nonpositive_timing_observed")
        folding.append(fold_seconds)
        baseline_load.append(b)
        candidate_load.append(c)
        total.append(fold_seconds + c)
        native_speedups.append(b / c)
        total_speedups.append(b / (fold_seconds + c))
        fold_ratios.append(fold_seconds / c)

    native_speedup = float(statistics.median(native_speedups))
    receipt = MappingProxyType(
        {
            "schema": "act.forward_exact.native_ranged_load_benchmark.v1",
            "synthetic_only": True,
            "single_thread": True,
            "paired_alternating_order": True,
            "native_load_is_addRows_only": True,
            "folding_preprocess_reported_separately": True,
            "total_includes_folding_and_candidate_addRows": True,
            "gate_applies_only_to_native_loader": True,
            "real_model_run": False,
            "large_model_run": False,
            "solver_run_called": False,
            "triangle_relaxation_called": False,
            "branch_and_bound_called": False,
            "backward_called": False,
            "dual_called": False,
            "proof_authority": False,
            "verdict_authority": False,
            "production_integration": False,
        }
    )
    return NativeLoadBenchmarkResult(
        warmup_pairs=warmups,
        measured_pairs=repeats,
        folding_preprocess_median_seconds=float(statistics.median(folding)),
        baseline_native_load_median_seconds=float(statistics.median(baseline_load)),
        candidate_native_load_median_seconds=float(statistics.median(candidate_load)),
        candidate_total_median_seconds=float(statistics.median(total)),
        native_loader_paired_median_speedup=native_speedup,
        total_paired_median_speedup=float(statistics.median(total_speedups)),
        folding_to_candidate_load_median_ratio=float(statistics.median(fold_ratios)),
        native_loader_speedup_gate=_NATIVE_LOADER_SPEEDUP_GATE,
        native_loader_candidate_supported=bool(
            native_speedup >= _NATIVE_LOADER_SPEEDUP_GATE
        ),
        receipt=receipt,
    )


__all__ = [
    "C89RatioSyntheticFrame",
    "ExactRangedNativeLoadSentinelError",
    "NativeEquivalenceResult",
    "NativeFrameLoadResult",
    "NativeLoadBenchmarkResult",
    "benchmark_native_ranged_row_loader",
    "make_c89_ratio_signed_upper_source",
    "run_native_ranged_equivalence_sentinel",
]
