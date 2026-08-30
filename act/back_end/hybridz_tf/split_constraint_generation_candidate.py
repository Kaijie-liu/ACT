#!/usr/bin/env python3
"""Bounded split-frame constraint-generation LP candidates.

This module deliberately has no verifier authority.  It minimizes ``-q``
over a relaxed HybridZ factor frame and returns HiGHS' raw minimization row
duals plus its full primal point.  A separate, independent checker must replay
either object before it can affect a proof, a verdict, or an accepted bound.

The loader never constructs ``[Auc Aub]``, ``[Ac Ab]``, or a stacked parent
matrix.  Continuous row blocks are passed to ``Highs.addRows`` separately;
binary coefficients are injected with ``Highs.changeCoeff``.  Only explicitly
selected upper rows are materialized.  Equality rows are all loaded in v1,
subject to a caller-provided hard cap.  Omitted upper rows are scanned in
bounded chunks and the deterministic largest violations are appended.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import numbers
import struct
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import _sparsetools

try:  # Candidate-only optional dependency.
    import highspy as _highspy
except Exception:  # pragma: no cover - exercised by fail-closed tests.
    _highspy = None


_SCHEMA = "act.hybridz.split_constraint_generation_candidate.v1"
_BACKEND = "highspy_split_blocks_constraint_generation_v1"
_INT32_MAX = int(np.iinfo(np.int32).max)
_HASH_CHUNK_BYTES = 1 << 20
_MAX_ROUNDS_HARD = 256
_MAX_BATCH_HARD = 65536
_MAX_SELECTED_UPPER_HARD = 1_000_000
_MAX_EQUALITY_HARD = 1_000_000
_MAX_BINARY_CHANGE_HARD = 2_000_000
_MAX_SCAN_CHUNK_HARD = 65536


class SplitConstraintGenerationCandidateError(ValueError):
    """The candidate could not be produced; no proof fact is emitted."""


class _DeadlineExpired(SplitConstraintGenerationCandidateError):
    pass


@dataclass(frozen=True)
class SplitConstraintGenerationCandidate:
    """One closed-model, non-authoritative split-frame LP proposal.

    ``upper_row_dual`` and ``equality_row_dual`` retain HiGHS' minimization
    sign convention and each follows its provided source block's original row
    order.  All three arrays are independent, C-contiguous, read-only binary64
    arrays.  Missing upper rows are represented by exact zero dual entries.
    Binary-labelled factor columns remain continuous and use the caller's
    bounds; this candidate never imposes integrality.
    """

    upper_row_dual: np.ndarray
    equality_row_dual: np.ndarray
    factor_primal: np.ndarray
    solver_minimization_objective: float
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    verdict_authority: bool = False
    primal_feasibility_authority: bool = False


@dataclass(frozen=True)
class _CSRView:
    name: str
    matrix: sp.csr_matrix
    rows: int
    columns: int
    nnz: int
    sha256: str


@dataclass(frozen=True)
class _ScanResult:
    top_omitted: Tuple[Tuple[int, float, float], ...]
    omitted_violated_rows: int
    selected_upper_violated_rows: int
    equality_violated_rows: int
    max_upper_violation: float
    max_equality_violation: float
    rows_scanned: int
    maximum_dense_chunk_rows: int


def _deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise _DeadlineExpired(f"deadline_expired_during_{stage}")


def _strict_cap(
    value: Any,
    *,
    name: str,
    lower: int,
    upper: int,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Integral)
    ):
        raise SplitConstraintGenerationCandidateError(
            f"{name}_must_be_an_integer"
        )
    result = int(value)
    if result < lower or result > upper:
        raise SplitConstraintGenerationCandidateError(
            f"{name}_outside_hard_range"
        )
    return result


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _update_array_bytes(digest: Any, array: np.ndarray) -> None:
    view = memoryview(array).cast("B")
    for start in range(0, len(view), _HASH_CHUNK_BYTES):
        digest.update(view[start : start + _HASH_CHUNK_BYTES])


def _array_sha256(array: np.ndarray, *, tag: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"act.ndarray.raw.v1\0")
    digest.update(tag.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(struct.pack(">Q", int(array.size)))
    _update_array_bytes(digest, array)
    return digest.hexdigest()


def _validate_dense_f64(
    values: Any,
    *,
    name: str,
    size: Optional[int],
    deadline: float,
) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise SplitConstraintGenerationCandidateError(
            f"{name}_must_be_a_binary64_ndarray"
        )
    array = values
    if (
        array.dtype != np.dtype(np.float64)
        or array.ndim != 1
        or (size is not None and int(array.size) != int(size))
        or not array.flags.c_contiguous
    ):
        raise SplitConstraintGenerationCandidateError(
            f"{name}_must_be_canonical_binary64_vector"
        )
    for start in range(0, int(array.size), 65536):
        _deadline(deadline, f"validate_{name}")
        if not np.all(np.isfinite(array[start : start + 65536])):
            raise SplitConstraintGenerationCandidateError(
                f"{name}_contains_nonfinite_value"
            )
    _deadline(deadline, f"validate_{name}")
    return array


def _validate_csr(
    matrix: Any,
    *,
    name: str,
    rows: int,
    columns: int,
    deadline: float,
) -> _CSRView:
    _deadline(deadline, f"validate_{name}")
    if (
        not sp.isspmatrix_csr(matrix)
        or matrix.dtype != np.dtype(np.float64)
        or matrix.shape != (rows, columns)
        or not matrix.has_canonical_format
    ):
        raise SplitConstraintGenerationCandidateError(
            f"{name}_must_be_canonical_binary64_csr"
        )
    if rows > _INT32_MAX or columns > _INT32_MAX:
        raise SplitConstraintGenerationCandidateError(
            f"{name}_dimension_exceeds_int32"
        )
    indptr = np.asarray(matrix.indptr)
    indices = np.asarray(matrix.indices)
    data = np.asarray(matrix.data)
    if (
        indptr.dtype != np.dtype(np.int32)
        or indices.dtype != np.dtype(np.int32)
        or data.dtype != np.dtype(np.float64)
        or indptr.ndim != 1
        or indices.ndim != 1
        or data.ndim != 1
        or not indptr.flags.c_contiguous
        or not indices.flags.c_contiguous
        or not data.flags.c_contiguous
        or int(indptr.size) != rows + 1
        or int(indices.size) != int(data.size)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != int(data.size)
        or int(data.size) > _INT32_MAX
    ):
        raise SplitConstraintGenerationCandidateError(
            f"{name}_csr_storage_not_canonical_int32_binary64"
        )
    for start in range(0, int(data.size), 65536):
        _deadline(deadline, f"validate_{name}_data")
        block = data[start : start + 65536]
        if not np.all(np.isfinite(block)):
            raise SplitConstraintGenerationCandidateError(
                f"{name}_contains_nonfinite_value"
            )
        if np.any(block == 0.0):
            raise SplitConstraintGenerationCandidateError(
                f"{name}_contains_explicit_zero"
            )
    digest = hashlib.sha256()
    digest.update(b"act.canonical_csr_f64_i32.v1\0")
    digest.update(name.encode("ascii"))
    digest.update(struct.pack(">QQQ", rows, columns, int(data.size)))
    for tag, array in (
        (b"indptr\0", indptr),
        (b"indices\0", indices),
        (b"data\0", data),
    ):
        _deadline(deadline, f"hash_{name}")
        digest.update(tag)
        _update_array_bytes(digest, array)
    _deadline(deadline, f"hash_{name}")
    return _CSRView(
        name=name,
        matrix=matrix,
        rows=rows,
        columns=columns,
        nnz=int(data.size),
        sha256=digest.hexdigest(),
    )


def _validate_seed(
    rows: Sequence[int],
    duals: Any,
    *,
    row_count: int,
    name: str,
) -> Tuple[Tuple[int, ...], np.ndarray]:
    raw_rows = np.asarray(rows)
    if raw_rows.ndim != 1:
        raise SplitConstraintGenerationCandidateError(
            f"{name}_rows_must_be_one_dimensional"
        )
    checked = []
    seen = set()
    for raw in raw_rows:
        if (
            isinstance(raw, (bool, np.bool_))
            or not isinstance(raw, numbers.Integral)
        ):
            raise SplitConstraintGenerationCandidateError(
                f"{name}_row_is_not_an_integer"
            )
        row = int(raw)
        if row < 0 or row >= row_count:
            raise SplitConstraintGenerationCandidateError(
                f"{name}_row_out_of_range"
            )
        if row in seen:
            raise SplitConstraintGenerationCandidateError(
                f"{name}_rows_not_unique"
            )
        seen.add(row)
        checked.append(row)
    try:
        dual_array = np.array(
            duals, dtype=np.float64, order="C", copy=True
        ).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SplitConstraintGenerationCandidateError(
            f"{name}_duals_invalid"
        ) from exc
    if (
        int(dual_array.size) != len(checked)
        or not np.all(np.isfinite(dual_array))
    ):
        raise SplitConstraintGenerationCandidateError(
            f"{name}_duals_invalid"
        )
    dual_array.setflags(write=False)
    return tuple(checked), dual_array


def _selected_continuous_arrays(
    matrix: sp.csr_matrix,
    rows: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize only the requested row subset, preserving row order."""

    count = len(rows)
    starts = np.empty(count + 1, dtype=np.int32)
    starts[0] = 0
    total = 0
    for offset, row in enumerate(rows):
        total += int(matrix.indptr[row + 1] - matrix.indptr[row])
        if total > _INT32_MAX:
            raise SplitConstraintGenerationCandidateError(
                "selected_continuous_nnz_exceeds_int32"
            )
        starts[offset + 1] = total
    indices = np.empty(total, dtype=np.int32)
    data = np.empty(total, dtype=np.float64)
    destination = 0
    for row in rows:
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        width = stop - start
        if width:
            indices[destination : destination + width] = matrix.indices[
                start:stop
            ]
            data[destination : destination + width] = matrix.data[
                start:stop
            ]
        destination += width
    return starts, indices, data


def _chunk_activity(
    continuous: sp.csr_matrix,
    binary: sp.csr_matrix,
    continuous_primal: np.ndarray,
    binary_primal: np.ndarray,
    start: int,
    stop: int,
    *,
    row_kind: str,
) -> np.ndarray:
    size = stop - start
    activity = np.zeros(size, dtype=np.float64)
    for split_kind, matrix, vector in (
        ("continuous", continuous, continuous_primal),
        ("binary", binary, binary_primal),
    ):
        source_start = int(matrix.indptr[start])
        source_stop = int(matrix.indptr[stop])
        if source_stop == source_start:
            continue
        # SciPy's CSR slice path copies the selected sparse data.  Borrow the
        # original indices/data and allocate only a bounded, chunk-sized row
        # pointer rebased to zero.  ``csr_matvec`` accumulates into activity.
        chunk_indptr = np.subtract(
            matrix.indptr[start : stop + 1],
            source_start,
            dtype=np.int32,
        )
        contribution = np.zeros(size, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            _sparsetools.csr_matvec(
                size,
                int(matrix.shape[1]),
                chunk_indptr,
                matrix.indices[source_start:source_stop],
                matrix.data[source_start:source_stop],
                vector,
                contribution,
            )
        if not np.all(np.isfinite(contribution)):
            raise SplitConstraintGenerationCandidateError(
                "nonfinite_split_scan_stage:"
                f"{row_kind}_{split_kind}_matvec"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            np.add(activity, contribution, out=activity)
        if not np.all(np.isfinite(activity)):
            raise SplitConstraintGenerationCandidateError(
                "nonfinite_split_scan_stage:"
                f"{row_kind}_combined_activity"
            )
    return activity


def _require_finite_scan_stage(
    values: np.ndarray, *, stage: str
) -> None:
    if not np.all(np.isfinite(values)):
        raise SplitConstraintGenerationCandidateError(
            f"nonfinite_split_scan_stage:{stage}"
        )


def _scan_split_frame(
    *,
    Auc: sp.csr_matrix,
    Aub: sp.csr_matrix,
    Ac: sp.csr_matrix,
    Ab: sp.csr_matrix,
    ub: np.ndarray,
    b: np.ndarray,
    primal: np.ndarray,
    n_continuous: int,
    selected_upper: set[int],
    top_cap: int,
    scan_chunk_rows: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    deadline: float,
) -> _ScanResult:
    continuous_primal = primal[:n_continuous]
    binary_primal = primal[n_continuous:]
    heap: list[Tuple[float, int, float]] = []
    omitted_violated = 0
    selected_violated = 0
    equality_violated = 0
    max_upper = 0.0
    max_equality = 0.0
    rows_scanned = 0
    maximum_dense_chunk_rows = 0

    for start in range(0, int(ub.size), scan_chunk_rows):
        _deadline(deadline, "scan_upper_rows")
        stop = min(start + scan_chunk_rows, int(ub.size))
        activity = _chunk_activity(
            Auc,
            Aub,
            continuous_primal,
            binary_primal,
            start,
            stop,
            row_kind="upper",
        )
        _require_finite_scan_stage(
            activity, stage="upper_activity"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            upper_residual = activity - ub[start:stop]
        _require_finite_scan_stage(
            upper_residual, stage="upper_residual"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            violation = np.maximum(upper_residual, 0.0)
        _require_finite_scan_stage(
            violation, stage="upper_violation"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            scale = 1.0 + np.maximum(
                np.abs(activity), np.abs(ub[start:stop])
            )
        _require_finite_scan_stage(scale, stage="upper_scale")
        with np.errstate(over="ignore", invalid="ignore"):
            relative = violation / scale
        _require_finite_scan_stage(
            relative, stage="upper_relative_violation"
        )
        local_rows = np.flatnonzero(
            (violation > absolute_tolerance)
            & (relative > relative_tolerance)
        )
        if violation.size:
            max_upper = max(max_upper, float(np.max(violation)))
        for local in local_rows:
            row = start + int(local)
            value = float(violation[local])
            score = float(relative[local])
            if row in selected_upper:
                selected_violated += 1
                continue
            omitted_violated += 1
            if top_cap <= 0:
                continue
            entry = (score, -row, value)
            if len(heap) < top_cap:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
        rows_scanned += stop - start
        maximum_dense_chunk_rows = max(
            maximum_dense_chunk_rows, stop - start
        )
        _deadline(deadline, "scan_upper_rows")

    for start in range(0, int(b.size), scan_chunk_rows):
        _deadline(deadline, "scan_equality_rows")
        stop = min(start + scan_chunk_rows, int(b.size))
        activity = _chunk_activity(
            Ac,
            Ab,
            continuous_primal,
            binary_primal,
            start,
            stop,
            row_kind="equality",
        )
        _require_finite_scan_stage(
            activity, stage="equality_activity"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            equality_residual = activity - b[start:stop]
        _require_finite_scan_stage(
            equality_residual, stage="equality_residual"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            violation = np.abs(equality_residual)
        _require_finite_scan_stage(
            violation, stage="equality_violation"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            scale = 1.0 + np.maximum(
                np.abs(activity), np.abs(b[start:stop])
            )
        _require_finite_scan_stage(scale, stage="equality_scale")
        with np.errstate(over="ignore", invalid="ignore"):
            relative = violation / scale
        _require_finite_scan_stage(
            relative, stage="equality_relative_violation"
        )
        violated = (
            (violation > absolute_tolerance)
            & (relative > relative_tolerance)
        )
        equality_violated += int(np.count_nonzero(violated))
        if violation.size:
            max_equality = max(
                max_equality, float(np.max(violation))
            )
        rows_scanned += stop - start
        maximum_dense_chunk_rows = max(
            maximum_dense_chunk_rows, stop - start
        )
        _deadline(deadline, "scan_equality_rows")

    top = tuple(
        (int(-entry[1]), float(entry[0]), float(entry[2]))
        for entry in sorted(heap, key=lambda item: (-item[0], -item[1]))
    )
    return _ScanResult(
        top_omitted=top,
        omitted_violated_rows=omitted_violated,
        selected_upper_violated_rows=selected_violated,
        equality_violated_rows=equality_violated,
        max_upper_violation=max_upper,
        max_equality_violation=max_equality,
        rows_scanned=rows_scanned,
        maximum_dense_chunk_rows=maximum_dense_chunk_rows,
    )


def _new_highs_model():
    if _highspy is None:
        raise SplitConstraintGenerationCandidateError(
            "highspy_backend_unavailable"
        )
    return _highspy.Highs()


def _require_highs_ok(status: Any, operation: str) -> None:
    if _highspy is None or status != _highspy.HighsStatus.kOk:
        raise SplitConstraintGenerationCandidateError(
            f"highs_{operation}_failed:{status}"
        )


def _close_highs_model(highs: Any) -> None:
    try:
        status = highs.clear()
    except BaseException as exc:
        raise SplitConstraintGenerationCandidateError(
            f"native_model_close_failed:{type(exc).__name__}"
        ) from exc
    if _highspy is None or status != _highspy.HighsStatus.kOk:
        raise SplitConstraintGenerationCandidateError(
            f"native_model_close_failed:{status}"
        )


def propose_split_constraint_generation_candidate(
    *,
    Auc: sp.csr_matrix,
    Aub: sp.csr_matrix,
    Ac: sp.csr_matrix,
    Ab: sp.csr_matrix,
    ub: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    seed_upper_rows: Sequence[int] = (),
    seed_upper_duals: Sequence[float] = (),
    seed_equality_rows: Sequence[int] = (),
    seed_equality_duals: Sequence[float] = (),
    deadline: float,
    max_rounds: int = 24,
    add_batch: int = 1024,
    max_selected_upper_rows: int = 24576,
    max_equality_rows: int = 65536,
    max_binary_change_coefficients: int = 65536,
    scan_chunk_rows: int = 8192,
    absolute_violation_tolerance: float = 5.0e-8,
    relative_violation_tolerance: float = 5.0e-10,
    threads: int = 1,
) -> SplitConstraintGenerationCandidate:
    """Return a bounded, split-block constraint-generation LP proposal.

    The explicit seed row order is preserved exactly, including seed rows
    whose supplied dual is zero.  Seed duals are receipt-bound diagnostics;
    HiGHS v1 does not use them as a basis or warm start.  All equality rows are
    loaded in their source order and must fit ``max_equality_rows``.

    Row dual outputs retain HiGHS' raw minimization sign.  The primal, duals,
    objective, solver status, and complete-frame scan are candidates only.
    Nothing returned by this function certifies feasibility or a bound.
    """

    started = time.monotonic()
    if isinstance(deadline, (bool, np.bool_)):
        raise SplitConstraintGenerationCandidateError(
            "deadline_must_be_finite_absolute_monotonic_time"
        )
    try:
        deadline = float(deadline)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SplitConstraintGenerationCandidateError(
            "deadline_must_be_finite_absolute_monotonic_time"
        ) from exc
    if not math.isfinite(deadline):
        raise SplitConstraintGenerationCandidateError(
            "deadline_must_be_finite_absolute_monotonic_time"
        )
    _deadline(deadline, "entry")

    caps = {
        "max_rounds": _strict_cap(
            max_rounds,
            name="max_rounds",
            lower=1,
            upper=_MAX_ROUNDS_HARD,
        ),
        "add_batch": _strict_cap(
            add_batch,
            name="add_batch",
            lower=1,
            upper=_MAX_BATCH_HARD,
        ),
        "max_selected_upper_rows": _strict_cap(
            max_selected_upper_rows,
            name="max_selected_upper_rows",
            lower=1,
            upper=_MAX_SELECTED_UPPER_HARD,
        ),
        "max_equality_rows": _strict_cap(
            max_equality_rows,
            name="max_equality_rows",
            lower=0,
            upper=_MAX_EQUALITY_HARD,
        ),
        "max_binary_change_coefficients": _strict_cap(
            max_binary_change_coefficients,
            name="max_binary_change_coefficients",
            lower=0,
            upper=_MAX_BINARY_CHANGE_HARD,
        ),
        "scan_chunk_rows": _strict_cap(
            scan_chunk_rows,
            name="scan_chunk_rows",
            lower=1,
            upper=_MAX_SCAN_CHUNK_HARD,
        ),
        "threads": _strict_cap(
            threads, name="threads", lower=1, upper=64
        ),
    }
    try:
        absolute_tolerance = float(absolute_violation_tolerance)
        relative_tolerance = float(relative_violation_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SplitConstraintGenerationCandidateError(
            "violation_tolerance_invalid"
        ) from exc
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise SplitConstraintGenerationCandidateError(
            "violation_tolerance_invalid"
        )

    ub = _validate_dense_f64(
        ub, name="ub", size=None, deadline=deadline
    )
    b = _validate_dense_f64(
        b, name="b", size=None, deadline=deadline
    )
    n_upper = int(ub.size)
    n_equality = int(b.size)
    if n_equality > caps["max_equality_rows"]:
        raise SplitConstraintGenerationCandidateError(
            "all_equality_rows_exceed_v1_cap"
        )
    if not sp.isspmatrix_csr(Auc) or not sp.isspmatrix_csr(Aub):
        raise SplitConstraintGenerationCandidateError(
            "upper_split_blocks_must_be_csr"
        )
    n_continuous = int(Auc.shape[1])
    n_binary = int(Aub.shape[1])
    if n_continuous + n_binary <= 0:
        raise SplitConstraintGenerationCandidateError(
            "factor_frame_must_have_a_column"
        )
    q = _validate_dense_f64(
        q,
        name="q",
        size=n_continuous + n_binary,
        deadline=deadline,
    )
    lower_bounds = _validate_dense_f64(
        lower_bounds,
        name="lower_bounds",
        size=q.size,
        deadline=deadline,
    )
    upper_bounds = _validate_dense_f64(
        upper_bounds,
        name="upper_bounds",
        size=q.size,
        deadline=deadline,
    )
    if np.any(lower_bounds > upper_bounds):
        raise SplitConstraintGenerationCandidateError(
            "factor_lower_bound_exceeds_upper_bound"
        )

    views = {
        "Auc": _validate_csr(
            Auc,
            name="Auc",
            rows=n_upper,
            columns=n_continuous,
            deadline=deadline,
        ),
        "Aub": _validate_csr(
            Aub,
            name="Aub",
            rows=n_upper,
            columns=n_binary,
            deadline=deadline,
        ),
        "Ac": _validate_csr(
            Ac,
            name="Ac",
            rows=n_equality,
            columns=n_continuous,
            deadline=deadline,
        ),
        "Ab": _validate_csr(
            Ab,
            name="Ab",
            rows=n_equality,
            columns=n_binary,
            deadline=deadline,
        ),
    }
    seed_upper_rows, seed_upper_duals = _validate_seed(
        seed_upper_rows,
        seed_upper_duals,
        row_count=n_upper,
        name="seed_upper",
    )
    seed_equality_rows, seed_equality_duals = _validate_seed(
        seed_equality_rows,
        seed_equality_duals,
        row_count=n_equality,
        name="seed_equality",
    )
    if len(seed_upper_rows) > caps["max_selected_upper_rows"]:
        raise SplitConstraintGenerationCandidateError(
            "explicit_seed_upper_rows_exceed_selection_cap"
        )
    seed_upper_binary_nnz = sum(
        int(Aub.indptr[row + 1] - Aub.indptr[row])
        for row in seed_upper_rows
    )
    if (
        views["Ab"].nnz + seed_upper_binary_nnz
        > caps["max_binary_change_coefficients"]
    ):
        raise SplitConstraintGenerationCandidateError(
            "initial_loaded_binary_change_coefficient_cap_exceeded"
        )

    dense_hashes = {
        "ub": _array_sha256(ub, tag="ub"),
        "b": _array_sha256(b, tag="b"),
        "q": _array_sha256(q, tag="q"),
        "lower_bounds": _array_sha256(
            lower_bounds, tag="lower_bounds"
        ),
        "upper_bounds": _array_sha256(
            upper_bounds, tag="upper_bounds"
        ),
        "seed_upper_duals": _array_sha256(
            seed_upper_duals, tag="seed_upper_duals"
        ),
        "seed_equality_duals": _array_sha256(
            seed_equality_duals, tag="seed_equality_duals"
        ),
    }
    source_binding = {
        "schema": "act.hybridz.provided_split_frame_binding.v1",
        "blocks": {
            name: {
                "shape": [view.rows, view.columns],
                "nnz": view.nnz,
                "sha256": view.sha256,
            }
            for name, view in views.items()
        },
        "dense_sha256": dense_hashes,
        "seed_upper_row_order": list(seed_upper_rows),
        "seed_equality_row_order": list(seed_equality_rows),
        "n_continuous": n_continuous,
        "n_binary_relaxed": n_binary,
        "n_upper": n_upper,
        "n_equality": n_equality,
    }
    provided_split_frame_sha256 = _canonical_json_sha256(
        source_binding
    )
    _deadline(deadline, "source_binding")

    selected = list(seed_upper_rows)
    selected_set = set(selected)
    upper_model_positions: list[int] = []
    physical_loaded_continuous_nnz = 0
    physical_loaded_binary_nnz = 0
    maximum_materialized_upper_continuous_nnz = 0
    continuous_add_rows_calls = 0
    binary_change_coeff_calls = 0
    physical_model_rows = 0
    equality_model_start = len(selected)
    rounds: list[Mapping[str, Any]] = []
    full_scan_count = 0
    full_scan_rows = 0
    maximum_scan_dense_rows = 0
    last_primal: Optional[np.ndarray] = None
    last_upper_dual: Optional[np.ndarray] = None
    last_equality_dual: Optional[np.ndarray] = None
    last_objective: Optional[float] = None
    last_selected: Tuple[int, ...] = ()
    last_loaded_continuous_nnz = 0
    last_loaded_binary_nnz = 0
    last_model_rows = 0
    terminal_status = "not_started"
    primal_candidate_status = "not_available"
    deadline_reason: Optional[str] = None
    highs = None
    primary_error: Optional[BaseException] = None
    close_error: Optional[BaseException] = None

    def add_upper_rows(rows_to_add: Sequence[int]) -> None:
        nonlocal physical_loaded_continuous_nnz
        nonlocal physical_loaded_binary_nnz
        nonlocal maximum_materialized_upper_continuous_nnz
        nonlocal continuous_add_rows_calls
        nonlocal binary_change_coeff_calls
        nonlocal physical_model_rows
        if not rows_to_add:
            return
        binary_to_add = sum(
            int(Aub.indptr[row + 1] - Aub.indptr[row])
            for row in rows_to_add
        )
        if (
            physical_loaded_binary_nnz + binary_to_add
            > caps["max_binary_change_coefficients"]
        ):
            raise SplitConstraintGenerationCandidateError(
                "loaded_binary_change_coefficient_cap_exceeded"
            )
        _deadline(deadline, "materialize_selected_upper_rows")
        starts, indices, data = _selected_continuous_arrays(
            Auc, rows_to_add
        )
        maximum_materialized_upper_continuous_nnz = max(
            maximum_materialized_upper_continuous_nnz,
            int(data.size),
        )
        row_indices = np.fromiter(
            rows_to_add, dtype=np.int64, count=len(rows_to_add)
        )
        row_offset = physical_model_rows
        _require_highs_ok(
            highs.addRows(
                len(rows_to_add),
                np.full(
                    len(rows_to_add),
                    -_highspy.kHighsInf,
                    dtype=np.float64,
                ),
                ub[row_indices],
                int(data.size),
                starts,
                indices,
                data,
            ),
            "add_selected_upper_continuous_rows",
        )
        continuous_add_rows_calls += 1
        physical_model_rows += len(rows_to_add)
        physical_loaded_continuous_nnz += int(data.size)
        for local_row, source_row in enumerate(rows_to_add):
            start = int(Aub.indptr[source_row])
            stop = int(Aub.indptr[source_row + 1])
            for position in range(start, stop):
                if (binary_change_coeff_calls & 255) == 0:
                    _deadline(deadline, "inject_selected_upper_binary")
                _require_highs_ok(
                    highs.changeCoeff(
                        row_offset + local_row,
                        n_continuous + int(Aub.indices[position]),
                        float(Aub.data[position]),
                    ),
                    "change_selected_upper_binary_coefficient",
                )
                binary_change_coeff_calls += 1
                physical_loaded_binary_nnz += 1
        upper_model_positions.extend(
            range(row_offset, row_offset + len(rows_to_add))
        )
        _deadline(deadline, "add_selected_upper_rows")

    def add_all_equalities() -> None:
        nonlocal physical_loaded_continuous_nnz
        nonlocal physical_loaded_binary_nnz
        nonlocal continuous_add_rows_calls
        nonlocal binary_change_coeff_calls
        nonlocal physical_model_rows
        if n_equality == 0:
            return
        _deadline(deadline, "add_equality_rows")
        row_offset = physical_model_rows
        _require_highs_ok(
            highs.addRows(
                n_equality,
                b,
                b,
                views["Ac"].nnz,
                Ac.indptr,
                Ac.indices,
                Ac.data,
            ),
            "add_all_equality_continuous_rows",
        )
        continuous_add_rows_calls += 1
        physical_model_rows += n_equality
        physical_loaded_continuous_nnz += views["Ac"].nnz
        for source_row in range(n_equality):
            start = int(Ab.indptr[source_row])
            stop = int(Ab.indptr[source_row + 1])
            for position in range(start, stop):
                if (binary_change_coeff_calls & 255) == 0:
                    _deadline(deadline, "inject_equality_binary")
                _require_highs_ok(
                    highs.changeCoeff(
                        row_offset + source_row,
                        n_continuous + int(Ab.indices[position]),
                        float(Ab.data[position]),
                    ),
                    "change_equality_binary_coefficient",
                )
                binary_change_coeff_calls += 1
                physical_loaded_binary_nnz += 1
        _deadline(deadline, "add_equality_rows")

    try:
        highs = _new_highs_model()
        HS = _highspy.HighsStatus
        _require_highs_ok(
            highs.setOptionValue("output_flag", False),
            "set_output_flag",
        )
        _require_highs_ok(
            highs.setOptionValue("presolve", "on"),
            "set_presolve",
        )
        _require_highs_ok(
            highs.setOptionValue("solver", "simplex"),
            "set_solver",
        )
        _require_highs_ok(
            highs.setOptionValue("threads", caps["threads"]),
            "set_threads",
        )
        _deadline(deadline, "add_factor_columns")
        _require_highs_ok(
            highs.addCols(
                int(q.size),
                np.negative(q),
                lower_bounds,
                upper_bounds,
                0,
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            ),
            "add_factor_columns",
        )
        add_upper_rows(selected)
        equality_model_start = physical_model_rows
        add_all_equalities()
        if (
            int(highs.getNumRow()) != physical_model_rows
            or int(highs.getNumCol()) != int(q.size)
            or int(highs.getNumNz())
            != physical_loaded_continuous_nnz
            + physical_loaded_binary_nnz
        ):
            raise SplitConstraintGenerationCandidateError(
                "highs_split_loader_topology_mismatch"
            )

        for round_index in range(caps["max_rounds"]):
            _deadline(deadline, "before_candidate_solve")
            remaining = deadline - time.monotonic()
            _require_highs_ok(
                highs.setOptionValue(
                    "time_limit", max(1.0e-3, remaining)
                ),
                "set_time_limit",
            )
            run_status = highs.run()
            _deadline(deadline, "candidate_solve")
            if run_status != HS.kOk:
                raise SplitConstraintGenerationCandidateError(
                    f"highs_candidate_run_failed:{run_status}"
                )
            if highs.getModelStatus() != _highspy.HighsModelStatus.kOptimal:
                raise SplitConstraintGenerationCandidateError(
                    "highs_candidate_model_not_optimal:"
                    f"{highs.getModelStatus()}"
                )
            solution = highs.getSolution()
            if not bool(getattr(solution, "value_valid", False)):
                raise SplitConstraintGenerationCandidateError(
                    "highs_candidate_primal_invalid"
                )
            if not bool(getattr(solution, "dual_valid", False)):
                raise SplitConstraintGenerationCandidateError(
                    "highs_candidate_dual_invalid"
                )
            primal = np.asarray(
                solution.col_value, dtype=np.float64
            ).reshape(-1)
            raw_dual = np.asarray(
                solution.row_dual, dtype=np.float64
            ).reshape(-1)
            if (
                int(primal.size) != int(q.size)
                or int(raw_dual.size) != physical_model_rows
                or not np.all(np.isfinite(primal))
                or not np.all(np.isfinite(raw_dual))
            ):
                raise SplitConstraintGenerationCandidateError(
                    "highs_candidate_solution_shape_or_finiteness_invalid"
                )
            objective = float(
                highs.getInfo().objective_function_value
            )
            if not math.isfinite(objective):
                raise SplitConstraintGenerationCandidateError(
                    "highs_candidate_objective_nonfinite"
                )

            current_upper_dual = np.zeros(
                n_upper, dtype=np.float64
            )
            if selected:
                current_upper_dual[
                    np.fromiter(
                        selected,
                        dtype=np.int64,
                        count=len(selected),
                    )
                ] = raw_dual[
                    np.fromiter(
                        upper_model_positions,
                        dtype=np.int64,
                        count=len(upper_model_positions),
                    )
                ]
            current_equality_dual = np.array(
                raw_dual[
                    equality_model_start : equality_model_start
                    + n_equality
                ],
                dtype=np.float64,
                order="C",
                copy=True,
            )
            current_primal = np.array(
                primal, dtype=np.float64, order="C", copy=True
            )
            current_upper_dual.setflags(write=False)
            current_equality_dual.setflags(write=False)
            current_primal.setflags(write=False)
            last_primal = current_primal
            last_upper_dual = current_upper_dual
            last_equality_dual = current_equality_dual
            last_objective = objective
            last_selected = tuple(selected)
            last_loaded_continuous_nnz = (
                physical_loaded_continuous_nnz
            )
            last_loaded_binary_nnz = physical_loaded_binary_nnz
            last_model_rows = physical_model_rows

            remaining_cap = (
                caps["max_selected_upper_rows"] - len(selected)
            )
            top_cap = min(caps["add_batch"], max(0, remaining_cap))
            round_record = {
                "round": round_index + 1,
                "selected_upper_rows_at_solve": len(selected),
                "loaded_rows_at_solve": physical_model_rows,
                "loaded_continuous_nnz_at_solve": (
                    physical_loaded_continuous_nnz
                ),
                "loaded_binary_nnz_at_solve": (
                    physical_loaded_binary_nnz
                ),
                "solver_minimization_objective_hex": objective.hex(),
                "omitted_upper_violated_rows": None,
                "selected_upper_violated_rows": None,
                "equality_violated_rows": None,
                "max_upper_violation_hex": None,
                "max_equality_violation_hex": None,
                "added_upper_rows": [],
                "complete_split_scan_candidate_only": False,
            }
            rounds.append(round_record)
            scan = _scan_split_frame(
                Auc=Auc,
                Aub=Aub,
                Ac=Ac,
                Ab=Ab,
                ub=ub,
                b=b,
                primal=current_primal,
                n_continuous=n_continuous,
                selected_upper=selected_set,
                top_cap=top_cap,
                scan_chunk_rows=caps["scan_chunk_rows"],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                deadline=deadline,
            )
            full_scan_count += 1
            full_scan_rows += scan.rows_scanned
            maximum_scan_dense_rows = max(
                maximum_scan_dense_rows,
                scan.maximum_dense_chunk_rows,
            )
            round_record.update(
                {
                    "omitted_upper_violated_rows": (
                        scan.omitted_violated_rows
                    ),
                    "selected_upper_violated_rows": (
                        scan.selected_upper_violated_rows
                    ),
                    "equality_violated_rows": (
                        scan.equality_violated_rows
                    ),
                    "max_upper_violation_hex": (
                        float(scan.max_upper_violation).hex()
                    ),
                    "max_equality_violation_hex": (
                        float(scan.max_equality_violation).hex()
                    ),
                    "complete_split_scan_candidate_only": True,
                }
            )
            if (
                scan.selected_upper_violated_rows
                or scan.equality_violated_rows
            ):
                terminal_status = "loaded_row_violation_detected"
                primal_candidate_status = (
                    "loaded_split_row_violation_detected_candidate_only"
                )
                break
            if scan.omitted_violated_rows == 0:
                terminal_status = "full_scan_candidate_feasible"
                primal_candidate_status = (
                    "no_split_row_violation_detected_candidate_only"
                )
                break
            primal_candidate_status = (
                "omitted_upper_violation_detected_candidate_only"
            )
            if remaining_cap <= 0:
                terminal_status = "upper_row_cap_reached"
                break
            if round_index + 1 >= caps["max_rounds"]:
                terminal_status = "round_cap_reached"
                break
            remaining_binary = (
                caps["max_binary_change_coefficients"]
                - physical_loaded_binary_nnz
            )
            new_rows = []
            for row, _score, _value in scan.top_omitted:
                row_binary_nnz = int(
                    Aub.indptr[row + 1] - Aub.indptr[row]
                )
                if row_binary_nnz <= remaining_binary:
                    new_rows.append(row)
                    remaining_binary -= row_binary_nnz
            if not new_rows:
                terminal_status = (
                    "binary_change_coefficient_cap_reached"
                    if scan.top_omitted
                    else "no_new_upper_rows"
                )
                break
            add_upper_rows(new_rows)
            selected.extend(new_rows)
            selected_set.update(new_rows)
            round_record["added_upper_rows"] = list(new_rows)
            if (
                int(highs.getNumRow()) != physical_model_rows
                or int(highs.getNumNz())
                != physical_loaded_continuous_nnz
                + physical_loaded_binary_nnz
            ):
                raise SplitConstraintGenerationCandidateError(
                    "highs_incremental_loader_topology_mismatch"
                )
        else:  # Defensive: the explicit final-round branch should win.
            terminal_status = "round_cap_reached"
    except _DeadlineExpired as exc:
        if last_primal is None:
            primary_error = exc
        else:
            terminal_status = "deadline_exhausted_after_candidate"
            primal_candidate_status = (
                "not_completely_scanned_before_deadline_candidate_only"
            )
            deadline_reason = str(exc)
    except BaseException as exc:  # Close native state before propagating.
        primary_error = exc
    finally:
        if highs is not None:
            try:
                _close_highs_model(highs)
            except BaseException as exc:
                close_error = exc
            finally:
                highs = None

    if primary_error is not None:
        if close_error is not None:
            raise close_error from primary_error
        if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
            raise primary_error
        if isinstance(
            primary_error, SplitConstraintGenerationCandidateError
        ):
            raise primary_error
        raise SplitConstraintGenerationCandidateError(
            "constraint_generation_backend_error:"
            f"{type(primary_error).__name__}"
        ) from primary_error
    if close_error is not None:
        raise close_error
    if (
        last_primal is None
        or last_upper_dual is None
        or last_equality_dual is None
        or last_objective is None
    ):
        raise SplitConstraintGenerationCandidateError(
            "constraint_generation_returned_without_candidate"
        )

    receipt = {
        "schema": _SCHEMA,
        "status": terminal_status,
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "primal_feasibility_authority": False,
        "parent_binding_authority": False,
        "backend": _BACKEND,
        "highs_version": "{}.{}.{}".format(
            _highspy.HIGHS_VERSION_MAJOR,
            _highspy.HIGHS_VERSION_MINOR,
            _highspy.HIGHS_VERSION_PATCH,
        ),
        "objective_convention": (
            "highs_minimize_negative_q_raw_minimization_duals"
        ),
        "upper_dual_row_order": "provided_Auc_Aub_original_row_order",
        "equality_dual_row_order": "provided_Ac_Ab_original_row_order",
        "binary_factor_semantics": (
            "continuous_relaxation_with_provided_bounds_no_integrality"
        ),
        "provided_split_frame_binding": source_binding,
        "provided_split_frame_sha256": provided_split_frame_sha256,
        "parent_binding": False,
        "caps": dict(caps),
        "absolute_violation_tolerance_hex": absolute_tolerance.hex(),
        "relative_violation_tolerance_hex": relative_tolerance.hex(),
        "absolute_deadline_hex": deadline.hex(),
        "seed_duals_used_as_warm_start": False,
        "explicit_zero_dual_seed_rows_retained": True,
        "selected_upper_row_order": list(last_selected),
        "rounds": rounds,
        "rounds_completed": len(rounds),
        "full_split_scan_count": full_scan_count,
        "full_split_rows_scanned": full_scan_rows,
        "full_split_scan_is_candidate_telemetry_only": True,
        "primal_candidate_status": primal_candidate_status,
        "deadline_reason": deadline_reason,
        "n_continuous": n_continuous,
        "n_binary_relaxed": n_binary,
        "n_upper": n_upper,
        "n_equality": n_equality,
        "loaded_upper_rows_at_candidate_solve": len(last_selected),
        "loaded_equality_rows_at_candidate_solve": n_equality,
        "loaded_rows_at_candidate_solve": last_model_rows,
        "loaded_continuous_nnz_at_candidate_solve": (
            last_loaded_continuous_nnz
        ),
        "loaded_binary_nnz_at_candidate_solve": (
            last_loaded_binary_nnz
        ),
        "loaded_nnz_at_candidate_solve": (
            last_loaded_continuous_nnz + last_loaded_binary_nnz
        ),
        "physical_rows_before_close": physical_model_rows,
        "physical_continuous_nnz_before_close": (
            physical_loaded_continuous_nnz
        ),
        "physical_binary_nnz_before_close": (
            physical_loaded_binary_nnz
        ),
        "discarded_unsolved_model_mutation": (
            physical_model_rows != last_model_rows
            or physical_loaded_continuous_nnz
            != last_loaded_continuous_nnz
            or physical_loaded_binary_nnz != last_loaded_binary_nnz
        ),
        "continuous_add_rows_calls": continuous_add_rows_calls,
        "binary_change_coefficient_calls": binary_change_coeff_calls,
        "maximum_materialized_upper_continuous_nnz": (
            maximum_materialized_upper_continuous_nnz
        ),
        "maximum_scan_dense_rows": maximum_scan_dense_rows,
        "maximum_scan_dense_binary64_bytes": (
            maximum_scan_dense_rows * 8
        ),
        "zero_padded_output_binary64_bytes": (
            (n_upper + n_equality + q.size) * 8
        ),
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "uses_dense_hstack": False,
        "uses_dense_vstack": False,
        "used_merged_sparse_frame": False,
        "materialized_full_candidate_csr": False,
        "upper_row_dual_sha256": _array_sha256(
            last_upper_dual, tag="upper_row_dual_raw_minimization"
        ),
        "equality_row_dual_sha256": _array_sha256(
            last_equality_dual,
            tag="equality_row_dual_raw_minimization",
        ),
        "factor_primal_sha256": _array_sha256(
            last_primal, tag="factor_primal_candidate"
        ),
        "solver_minimization_objective_hex": last_objective.hex(),
        "native_model_closed_before_return": True,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    receipt["receipt_sha256"] = _canonical_json_sha256(receipt)
    return SplitConstraintGenerationCandidate(
        upper_row_dual=last_upper_dual,
        equality_row_dual=last_equality_dual,
        factor_primal=last_primal,
        solver_minimization_objective=last_objective,
        receipt=receipt,
    )


__all__ = [
    "SplitConstraintGenerationCandidate",
    "SplitConstraintGenerationCandidateError",
    "propose_split_constraint_generation_candidate",
]
