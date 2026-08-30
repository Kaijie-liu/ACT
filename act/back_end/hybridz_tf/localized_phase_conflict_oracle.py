#!/usr/bin/env python3
"""Localized, candidate-only phase-pair conflict proposals.

This module deliberately has no verifier or pipeline integration.  It removes
rows from a *proposal* LP, never columns or variable bounds, and gives HiGHS no
proof authority.  A localized infeasibility ray becomes an edge only after it
is restored to the full ``upper_then_equality`` source frame and accepted by
the existing sparse ``Fraction`` replay against the live full parent.

The public entry point is default-off.  Its row selector uses pattern-only CSC
incidence and deterministic cumulative row prefixes at the strict 64/256/1024/
4096 tiers.  Every returned object is frozen and the complete tier telemetry is
checksummed.  Any malformed input, resource exhaustion, missing ray, rejected
replay, or terminal parent mutation returns no edge (or raises during initial
caller-contract validation).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

try:  # Candidate-only optional dependency.
    import highspy as _highspy
except Exception:  # pragma: no cover - exercised by environments without it.
    _highspy = None

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    ExactDualRayConflictCertificate,
    _ordered_source_frame_digest,
    exact_certificate_from_highs_dual_ray_candidate,
    verify_exact_dual_ray_conflict_certificate,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
    _literal_binding_digest,
    _ordered_pair,
    _stable_position_map,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _highs_process_threads,
)


class LocalizedPhaseConflictOracleError(ValueError):
    """Malformed caller contract; no localized candidate may be consumed."""


@dataclass(frozen=True, order=True)
class RowRef:
    """Authoritative original source-row identity (never an Operator tag id)."""

    kind: str
    local_row: int


@dataclass(frozen=True)
class LocalizedTierTelemetry:
    """Immutable telemetry for one cumulative localized LP prefix."""

    tier_row_cap: int
    expansion_depth: int
    ordered_row_refs: Tuple[RowRef, ...]
    ordered_global_row_ids: Tuple[int, ...]
    row_mapping_sha256: str
    selected_columns: int
    selected_source_nonzeros: int
    model_columns: int
    model_nonzeros: int
    build_seconds: float
    solve_seconds: float
    status: str
    ray_nonzero_rows: int
    exact_replay_status: str
    telemetry_sha256: str


@dataclass(frozen=True)
class LocalizedPhaseConflictOracleResult:
    """One non-authoritative localized proposal and optional exact edge proof."""

    status: str
    reason: str
    literals: Tuple[PhaseLiteral, PhaseLiteral]
    certificate: Optional[ExactDualRayConflictCertificate]
    edge_accepted: bool
    parent_semantic_digest: str
    terminal_parent_semantic_digest: Optional[str]
    property_digest: str
    ordered_source_frame_sha256: str
    terminal_source_frame_sha256: Optional[str]
    parent_unchanged: bool
    row_tiers: Tuple[int, ...]
    max_selected_nnz: int
    max_source_terms: int
    pattern_peak_byte_cap: int
    snapshot_buffer_byte_cap: int
    snapshot_buffer_bytes: int
    snapshot_seconds: float
    incidence_row_cap: int
    frontier_column_cap: int
    candidate_budget_seconds: float
    tiers: Tuple[LocalizedTierTelemetry, ...]
    telemetry_sha256: str
    result_sha256: str
    proof_authority: bool = False


@dataclass(frozen=True)
class _PatternBlock:
    """Owned CSC incidence arrays; coefficient data is intentionally absent."""

    column_indptr: np.ndarray
    column_rows: np.ndarray
    n_rows: int
    n_columns: int


@dataclass(frozen=True)
class _IncidenceFrame:
    upper_continuous: _PatternBlock
    upper_binary: _PatternBlock
    equality_continuous: _PatternBlock
    equality_binary: _PatternBlock
    n_upper: int
    n_equality: int
    n_continuous: int
    n_binary: int


@dataclass(frozen=True)
class _TierSolve:
    status: str
    ordered_global_row_ids: Tuple[int, ...]
    row_mapping_sha256: str
    local_ray: Optional[Tuple[float, ...]]
    model_nonzeros: int
    build_seconds: float
    solve_seconds: float


_STRICT_ROW_TIERS = (64, 256, 1024, 4096)
_HARD_MAX_SELECTED_NNZ = 1_000_000
_HARD_PATTERN_PEAK_BYTES = 256 * 1024 * 1024
_HARD_SNAPSHOT_BUFFER_BYTES = 384 * 1024 * 1024
_HARD_INCIDENCE_ROWS = _STRICT_ROW_TIERS[-1]
_HARD_FRONTIER_COLUMNS = 262_144
_TINY_PROPOSAL_COEFFICIENT = 1.0e-12


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise LocalizedPhaseConflictOracleError(f"{name}_not_integer")
    return int(value)


def _check_deadline(deadline: float, reason: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(reason)


def _validate_canonical_csr(
    matrix: Any,
    *,
    shape: Tuple[int, int],
    name: str,
) -> sp.csr_matrix:
    if type(matrix) is not sp.csr_matrix or matrix.shape != shape:
        raise LocalizedPhaseConflictOracleError(f"{name}_not_exact_csr")
    indptr = np.asarray(matrix.indptr)
    indices = np.asarray(matrix.indices)
    data = np.asarray(matrix.data)
    if (
        matrix.dtype != np.dtype(np.float64)
        or indptr.ndim != 1
        or indices.ndim != 1
        or data.ndim != 1
        or not np.issubdtype(indptr.dtype, np.integer)
        or not np.issubdtype(indices.dtype, np.integer)
        or int(indptr.size) != shape[0] + 1
        or int(indices.size) != int(data.size)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != int(indices.size)
        or np.any(indptr[1:] < indptr[:-1])
        or (indices.size and (np.any(indices < 0) or np.any(indices >= shape[1])))
        or (data.size and (not np.all(np.isfinite(data)) or np.any(data == 0.0)))
    ):
        raise LocalizedPhaseConflictOracleError(f"{name}_malformed_csr")
    if indices.size > 1:
        nonincreasing = indices[1:] <= indices[:-1]
        row_cuts = np.unique(indptr[1:-1])
        row_cuts = row_cuts[
            (row_cuts > 0) & (row_cuts < indices.size)
        ]
        nonincreasing[row_cuts - 1] = False
        if np.any(nonincreasing):
            raise LocalizedPhaseConflictOracleError(
                f"{name}_duplicate_or_unsorted_columns"
            )
    return matrix


def _constraint_blocks(
    hz: SparseHZono,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    n_cont = hz.n_cont
    n_bin = hz.n_bin
    upper_cont = (
        sp.csr_matrix((0, n_cont), dtype=np.float64)
        if hz.Auc is None
        else hz.Auc
    )
    upper_bin = (
        sp.csr_matrix((0, n_bin), dtype=np.float64)
        if hz.Aub is None
        else hz.Aub
    )
    blocks = (
        _validate_canonical_csr(
            upper_cont,
            shape=(hz.n_ub, n_cont),
            name="Auc",
        ),
        _validate_canonical_csr(
            upper_bin,
            shape=(hz.n_ub, n_bin),
            name="Aub",
        ),
        _validate_canonical_csr(
            hz.Ac,
            shape=(hz.n_eq, n_cont),
            name="Ac",
        ),
        _validate_canonical_csr(
            hz.Ab,
            shape=(hz.n_eq, n_bin),
            name="Ab",
        ),
    )
    return blocks


def _exact_dense_array(
    value: Any,
    *,
    dtype: np.dtype,
    length: int,
    name: str,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(dtype)
        or value.ndim != 1
        or int(value.size) != int(length)
        or not value.flags.c_contiguous
        or (
            np.issubdtype(value.dtype, np.floating)
            and not np.all(np.isfinite(value))
        )
    ):
        raise LocalizedPhaseConflictOracleError(
            f"parent_{name}_not_exact_dense_array"
        )
    return value


def _copy_array_with_deadline(
    value: np.ndarray,
    *,
    deadline: float,
    stage: str,
) -> np.ndarray:
    _check_deadline(deadline, f"deadline_before_{stage}")
    copied = np.empty(value.shape, dtype=value.dtype, order="C")
    chunk = max(1, (1 << 20) // max(1, int(value.dtype.itemsize)))
    for start in range(0, int(value.size), chunk):
        _check_deadline(deadline, f"deadline_during_{stage}")
        stop = min(int(value.size), start + chunk)
        np.copyto(copied[start:stop], value[start:stop], casting="no")
    _check_deadline(deadline, f"deadline_after_{stage}")
    return copied


def _copy_csr_with_deadline(
    matrix: sp.csr_matrix,
    *,
    deadline: float,
    stage: str,
) -> sp.csr_matrix:
    data = _copy_array_with_deadline(
        np.asarray(matrix.data),
        deadline=deadline,
        stage=f"{stage}_data",
    )
    indices = _copy_array_with_deadline(
        np.asarray(matrix.indices),
        deadline=deadline,
        stage=f"{stage}_indices",
    )
    indptr = _copy_array_with_deadline(
        np.asarray(matrix.indptr),
        deadline=deadline,
        stage=f"{stage}_indptr",
    )
    return sp.csr_matrix(
        (data, indices, indptr),
        shape=matrix.shape,
        dtype=np.float64,
        copy=False,
    )


def _private_parent_snapshot(
    hz: SparseHZono,
    *,
    expected_parent_digest: str,
    deadline: float,
) -> Tuple[SparseHZono, int, float]:
    """Copy every semantic core buffer and bind the copy to the caller seal.

    The digest is computed on the private copy itself.  Consequently a live
    parent mutation that races any individual copy either leaves the private
    content equal to the expected parent or causes a digest mismatch; later
    LP and Fraction reads never touch caller-owned buffers.
    """

    started = time.monotonic()
    live_vars = vars(hz)
    if any("conditional" in name.lower() for name in live_vars):
        raise LocalizedPhaseConflictOracleError(
            "parent_conditional_metadata_unsupported"
        )
    c = _exact_dense_array(
        live_vars.get("c"),
        dtype=np.dtype(np.float64),
        length=int(np.asarray(live_vars.get("c")).size),
        name="c",
    )
    n_out = int(c.size)
    Gc = live_vars.get("Gc")
    Gb = live_vars.get("Gb")
    if type(Gc) is not sp.csr_matrix or type(Gb) is not sp.csr_matrix:
        raise LocalizedPhaseConflictOracleError(
            "parent_generator_blocks_not_exact_csr"
        )
    n_cont = int(Gc.shape[1])
    n_bin = int(Gb.shape[1])
    Gc = _validate_canonical_csr(
        Gc, shape=(n_out, n_cont), name="Gc"
    )
    Gb = _validate_canonical_csr(
        Gb, shape=(n_out, n_bin), name="Gb"
    )
    b_raw = live_vars.get("b")
    ub_raw = live_vars.get("ub")
    if type(b_raw) is not np.ndarray or b_raw.ndim != 1:
        raise LocalizedPhaseConflictOracleError("parent_b_not_exact_array")
    n_eq = int(b_raw.size)
    n_ub = 0 if ub_raw is None else int(np.asarray(ub_raw).size)
    b = _exact_dense_array(
        b_raw,
        dtype=np.dtype(np.float64),
        length=n_eq,
        name="b",
    )
    ub = None
    if ub_raw is not None:
        ub = _exact_dense_array(
            ub_raw,
            dtype=np.dtype(np.float64),
            length=n_ub,
            name="ub",
        )
    Ac = _validate_canonical_csr(
        live_vars.get("Ac"), shape=(n_eq, n_cont), name="Ac"
    )
    Ab = _validate_canonical_csr(
        live_vars.get("Ab"), shape=(n_eq, n_bin), name="Ab"
    )
    raw_Auc = live_vars.get("Auc")
    raw_Aub = live_vars.get("Aub")
    if (raw_Auc is None) != (raw_Aub is None):
        raise LocalizedPhaseConflictOracleError(
            "parent_upper_blocks_presence_mismatch"
        )
    if raw_Auc is None:
        if n_ub != 0:
            raise LocalizedPhaseConflictOracleError(
                "parent_upper_rhs_without_matrix"
            )
        Auc = None
        Aub = None
    else:
        Auc = _validate_canonical_csr(
            raw_Auc, shape=(n_ub, n_cont), name="Auc"
        )
        Aub = _validate_canonical_csr(
            raw_Aub, shape=(n_ub, n_bin), name="Aub"
        )
    col_ids = _exact_dense_array(
        live_vars.get("col_ids"),
        dtype=np.dtype(np.int64),
        length=n_cont,
        name="col_ids",
    )
    bcol_ids = _exact_dense_array(
        live_vars.get("bcol_ids"),
        dtype=np.dtype(np.int64),
        length=n_bin,
        name="bcol_ids",
    )
    if (
        (col_ids.size and np.any(col_ids < 0))
        or (bcol_ids.size and np.any(bcol_ids < 0))
        or np.unique(col_ids).size != col_ids.size
        or np.unique(bcol_ids).size != bcol_ids.size
    ):
        raise LocalizedPhaseConflictOracleError(
            "parent_stable_ids_invalid"
        )

    dense_values = [c, b, col_ids, bcol_ids]
    if ub is not None:
        dense_values.append(ub)
    sparse_values = [Gc, Gb, Ac, Ab]
    if Auc is not None:
        sparse_values.extend([Auc, Aub])
    buffer_bytes = sum(int(value.nbytes) for value in dense_values)
    buffer_bytes += sum(
        int(matrix.data.nbytes)
        + int(matrix.indices.nbytes)
        + int(matrix.indptr.nbytes)
        for matrix in sparse_values
    )
    if buffer_bytes > _HARD_SNAPSHOT_BUFFER_BYTES:
        raise MemoryError("snapshot_buffer_byte_cap_exceeded")
    _check_deadline(deadline, "deadline_before_private_snapshot_copy")
    copied_dense = {
        "c": _copy_array_with_deadline(c, deadline=deadline, stage="copy_c"),
        "b": _copy_array_with_deadline(b, deadline=deadline, stage="copy_b"),
        "ub": (
            None
            if ub is None
            else _copy_array_with_deadline(
                ub, deadline=deadline, stage="copy_ub"
            )
        ),
        "col_ids": _copy_array_with_deadline(
            col_ids, deadline=deadline, stage="copy_col_ids"
        ),
        "bcol_ids": _copy_array_with_deadline(
            bcol_ids, deadline=deadline, stage="copy_bcol_ids"
        ),
    }
    copied_sparse = {
        name: _copy_csr_with_deadline(
            matrix, deadline=deadline, stage=f"copy_{name}"
        )
        for name, matrix in (
            ("Gc", Gc),
            ("Gb", Gb),
            ("Ac", Ac),
            ("Ab", Ab),
        )
    }
    if Auc is not None:
        copied_sparse["Auc"] = _copy_csr_with_deadline(
            Auc, deadline=deadline, stage="copy_Auc"
        )
        copied_sparse["Aub"] = _copy_csr_with_deadline(
            Aub, deadline=deadline, stage="copy_Aub"
        )
    else:
        copied_sparse["Auc"] = None
        copied_sparse["Aub"] = None
    snapshot = SparseHZono(
        c=copied_dense["c"],
        Gc=copied_sparse["Gc"],
        Gb=copied_sparse["Gb"],
        Ac=copied_sparse["Ac"],
        Ab=copied_sparse["Ab"],
        b=copied_dense["b"],
        Auc=copied_sparse["Auc"],
        Aub=copied_sparse["Aub"],
        ub=copied_dense["ub"],
        col_ids=copied_dense["col_ids"],
        bcol_ids=copied_dense["bcol_ids"],
    )
    for value in (
        snapshot.c,
        snapshot.b,
        snapshot.ub,
        snapshot.col_ids,
        snapshot.bcol_ids,
    ):
        if value is not None:
            value.setflags(write=False)
    for matrix in (
        snapshot.Gc,
        snapshot.Gb,
        snapshot.Ac,
        snapshot.Ab,
        snapshot.Auc,
        snapshot.Aub,
    ):
        if matrix is None:
            continue
        matrix.data.setflags(write=False)
        matrix.indices.setflags(write=False)
        matrix.indptr.setflags(write=False)
    private_digest = sparse_hz_semantic_digest(snapshot)
    _check_deadline(deadline, "deadline_after_private_snapshot_digest")
    if private_digest != expected_parent_digest:
        raise LocalizedPhaseConflictOracleError(
            "private_snapshot_parent_digest_mismatch"
        )
    return snapshot, int(buffer_bytes), float(time.monotonic() - started)


def _pattern_peak_bytes(matrix: sp.csr_matrix) -> int:
    # Conservative bound includes int8 CSR data, complete temporary CSC data,
    # indices/indptr, and the owned incidence copies retained after conversion.
    index_bytes = max(8, int(matrix.indices.dtype.itemsize))
    return int(
        2 * matrix.nnz
        + 2 * index_bytes * matrix.nnz
        + 2 * index_bytes * (matrix.shape[1] + 1)
    )


def _pattern_block(
    matrix: sp.csr_matrix,
    *,
    deadline: float,
) -> _PatternBlock:
    _check_deadline(deadline, "deadline_before_pattern_conversion")
    pattern_data = np.ones(int(matrix.nnz), dtype=np.int8)
    pattern = sp.csr_matrix(
        (pattern_data, matrix.indices, matrix.indptr),
        shape=matrix.shape,
        copy=False,
    )
    csc = pattern.tocsc(copy=True)
    _check_deadline(deadline, "deadline_after_pattern_conversion")
    indptr = np.array(csc.indptr, dtype=np.int64, copy=True)
    rows = np.array(csc.indices, dtype=np.int64, copy=True)
    indptr.setflags(write=False)
    rows.setflags(write=False)
    return _PatternBlock(
        column_indptr=indptr,
        column_rows=rows,
        n_rows=int(matrix.shape[0]),
        n_columns=int(matrix.shape[1]),
    )


def _build_incidence_frame(
    hz: SparseHZono,
    blocks: Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    *,
    deadline: float,
) -> _IncidenceFrame:
    estimated_peak = sum(_pattern_peak_bytes(matrix) for matrix in blocks)
    if estimated_peak > _HARD_PATTERN_PEAK_BYTES:
        raise MemoryError("pattern_peak_byte_cap_exceeded")
    built = []
    for matrix in blocks:
        _check_deadline(deadline, "deadline_before_pattern_block")
        built.append(_pattern_block(matrix, deadline=deadline))
        _check_deadline(deadline, "deadline_after_pattern_block")
    return _IncidenceFrame(
        upper_continuous=built[0],
        upper_binary=built[1],
        equality_continuous=built[2],
        equality_binary=built[3],
        n_upper=hz.n_ub,
        n_equality=hz.n_eq,
        n_continuous=hz.n_cont,
        n_binary=hz.n_bin,
    )


def _global_row_id(frame: _IncidenceFrame, ref: RowRef) -> int:
    if ref.kind == "upper" and 0 <= ref.local_row < frame.n_upper:
        return int(ref.local_row)
    if ref.kind == "equality" and 0 <= ref.local_row < frame.n_equality:
        return int(frame.n_upper + ref.local_row)
    raise LocalizedPhaseConflictOracleError("row_ref_out_of_range")


def _row_ref(frame: _IncidenceFrame, global_row: int) -> RowRef:
    row = _strict_int(global_row, name="global_row")
    if 0 <= row < frame.n_upper:
        return RowRef("upper", row)
    if frame.n_upper <= row < frame.n_upper + frame.n_equality:
        return RowRef("equality", row - frame.n_upper)
    raise LocalizedPhaseConflictOracleError("global_row_out_of_range")


def _validate_global_row_map(
    frame: _IncidenceFrame,
    rows: Sequence[int],
) -> Tuple[int, ...]:
    result = tuple(_strict_int(row, name="mapped_global_row") for row in rows)
    if any(
        row < 0 or row >= frame.n_upper + frame.n_equality
        for row in result
    ):
        raise LocalizedPhaseConflictOracleError("mapped_global_row_out_of_range")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise LocalizedPhaseConflictOracleError("row_map_not_strictly_increasing")
    return result


def _mapping_digest(
    *,
    parent_digest: str,
    source_frame_digest: str,
    rows: Sequence[int],
) -> str:
    return _sha256(
        {
            "schema": "act.localized_phase_conflict.row_map.v1",
            "parent_semantic_digest": parent_digest,
            "ordered_source_frame_sha256": source_frame_digest,
            "row_order": "upper_then_equality",
            "ordered_global_row_ids": [int(row) for row in rows],
        }
    )


def _rows_for_columns(
    frame: _IncidenceFrame,
    columns: Sequence[int],
    *,
    deadline: float,
) -> set[int]:
    total_rows = frame.n_upper + frame.n_equality
    if total_rows > 2_000_000:
        raise MemoryError("parent_row_cap_exceeded")
    reached = np.zeros(total_rows, dtype=np.bool_)
    for offset, raw_column in enumerate(sorted(set(int(item) for item in columns))):
        if offset % 256 == 0:
            _check_deadline(deadline, "deadline_during_column_incidence")
        if raw_column < 0 or raw_column >= frame.n_continuous + frame.n_binary:
            raise LocalizedPhaseConflictOracleError("incidence_column_out_of_range")
        if raw_column < frame.n_continuous:
            column = raw_column
            blocks = (
                (frame.upper_continuous, 0),
                (frame.equality_continuous, frame.n_upper),
            )
        else:
            column = raw_column - frame.n_continuous
            blocks = (
                (frame.upper_binary, 0),
                (frame.equality_binary, frame.n_upper),
            )
        for block, row_offset in blocks:
            start = int(block.column_indptr[column])
            stop = int(block.column_indptr[column + 1])
            for chunk_start in range(start, stop, 4096):
                _check_deadline(
                    deadline, "deadline_during_row_postings"
                )
                chunk_stop = min(stop, chunk_start + 4096)
                reached[
                    row_offset
                    + block.column_rows[chunk_start:chunk_stop]
                ] = True
    _check_deadline(deadline, "deadline_after_column_incidence")
    ordered = np.flatnonzero(reached)
    if ordered.size > _HARD_INCIDENCE_ROWS:
        ordered = ordered[:_HARD_INCIDENCE_ROWS]
    return {int(row) for row in ordered.tolist()}


def _columns_for_rows(
    hz: SparseHZono,
    blocks: Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    rows: Sequence[int],
    *,
    deadline: float,
) -> set[int]:
    upper_cont, upper_bin, eq_cont, eq_bin = blocks
    output: set[int] = set()
    for offset, global_row in enumerate(rows):
        if offset % 256 == 0:
            _check_deadline(deadline, "deadline_during_row_incidence")
        if global_row < hz.n_ub:
            local = global_row
            row_blocks = ((upper_cont, 0), (upper_bin, hz.n_cont))
        else:
            local = global_row - hz.n_ub
            row_blocks = ((eq_cont, 0), (eq_bin, hz.n_cont))
        for matrix, column_offset in row_blocks:
            start = int(matrix.indptr[local])
            stop = int(matrix.indptr[local + 1])
            for chunk_start in range(start, stop, 4096):
                _check_deadline(
                    deadline, "deadline_during_column_postings"
                )
                chunk_stop = min(stop, chunk_start + 4096)
                output.update(
                    column_offset + int(column)
                    for column in matrix.indices[
                        chunk_start:chunk_stop
                    ]
                )
                if len(output) > _HARD_FRONTIER_COLUMNS:
                    raise MemoryError("frontier_column_cap_exceeded")
    _check_deadline(deadline, "deadline_after_row_incidence")
    return output


def _row_source_nnz(
    hz: SparseHZono,
    blocks: Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    global_row: int,
) -> int:
    upper_cont, upper_bin, eq_cont, eq_bin = blocks
    if global_row < hz.n_ub:
        local = global_row
        return int(
            upper_cont.indptr[local + 1]
            - upper_cont.indptr[local]
            + upper_bin.indptr[local + 1]
            - upper_bin.indptr[local]
        )
    local = global_row - hz.n_ub
    return int(
        eq_cont.indptr[local + 1]
        - eq_cont.indptr[local]
        + eq_bin.indptr[local + 1]
        - eq_bin.indptr[local]
    )


def _select_prefix(
    hz: SparseHZono,
    blocks: Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    discovered_depth: dict[int, int],
    *,
    row_cap: int,
    nnz_cap: int,
) -> Tuple[Tuple[int, ...], int]:
    priority = sorted(discovered_depth, key=lambda row: (discovered_depth[row], row))
    chosen = []
    total_nnz = 0
    for row in priority:
        if len(chosen) >= row_cap:
            break
        row_nnz = _row_source_nnz(hz, blocks, row)
        if total_nnz + row_nnz > nnz_cap:
            break
        chosen.append(row)
        total_nnz += row_nnz
    return tuple(sorted(chosen)), int(total_nnz)


def _candidate_matrix_and_bounds(
    hz: SparseHZono,
    blocks: Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    rows: Tuple[int, ...],
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    upper_cont, upper_bin, eq_cont, eq_bin = blocks
    upper_rows = np.asarray([row for row in rows if row < hz.n_ub], dtype=np.int64)
    eq_rows = np.asarray([row - hz.n_ub for row in rows if row >= hz.n_ub], dtype=np.int64)
    upper = sp.hstack(
        [upper_cont[upper_rows], upper_bin[upper_rows]], format="csr"
    )
    equality = sp.hstack([eq_cont[eq_rows], eq_bin[eq_rows]], format="csr")
    matrix = sp.vstack([upper, equality], format="csr")
    # Candidate-quality simplification only.  The exact replay below always
    # reads the untouched full parent, including these tiny coefficients.
    if matrix.nnz:
        matrix = matrix.copy()
        matrix.data[np.abs(matrix.data) <= _TINY_PROPOSAL_COEFFICIENT] = 0.0
        matrix.eliminate_zeros()
        matrix.sort_indices()
    upper_rhs = (
        np.zeros(0, dtype=np.float64)
        if hz.ub is None
        else np.asarray(hz.ub, dtype=np.float64)
    )
    lower = np.concatenate(
        [
            np.full(upper_rows.size, -_highspy.kHighsInf),
            np.asarray(hz.b[eq_rows], dtype=np.float64),
        ]
    )
    upper_bound = np.concatenate(
        [
            np.asarray(upper_rhs[upper_rows], dtype=np.float64),
            np.asarray(hz.b[eq_rows], dtype=np.float64),
        ]
    )
    source_nnz = sum(_row_source_nnz(hz, blocks, row) for row in rows)
    return matrix, lower, upper_bound, int(source_nnz)


def _solve_tier(
    hz: SparseHZono,
    blocks: Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    pair: Tuple[PhaseLiteral, PhaseLiteral],
    rows: Tuple[int, ...],
    *,
    parent_digest: str,
    source_frame_digest: str,
    deadline: float,
) -> _TierSolve:
    mapping = _validate_global_row_map(
        _IncidenceFrame(
            _PatternBlock(np.empty(0), np.empty(0), hz.n_ub, hz.n_cont),
            _PatternBlock(np.empty(0), np.empty(0), hz.n_ub, hz.n_bin),
            _PatternBlock(np.empty(0), np.empty(0), hz.n_eq, hz.n_cont),
            _PatternBlock(np.empty(0), np.empty(0), hz.n_eq, hz.n_bin),
            hz.n_ub,
            hz.n_eq,
            hz.n_cont,
            hz.n_bin,
        ),
        rows,
    )
    map_digest = _mapping_digest(
        parent_digest=parent_digest,
        source_frame_digest=source_frame_digest,
        rows=mapping,
    )
    if _highspy is None:
        return _TierSolve("backend_unavailable", mapping, map_digest, None, 0, 0.0, 0.0)
    build_start = time.monotonic()
    _check_deadline(deadline, "deadline_before_local_model")
    matrix, row_lower, row_upper, _ = _candidate_matrix_and_bounds(hz, blocks, mapping)
    positions = _stable_position_map(hz)
    n_variables = hz.n_cont + hz.n_bin
    column_lower = -np.ones(n_variables, dtype=np.float64)
    column_upper = np.ones(n_variables, dtype=np.float64)
    for literal in pair:
        column = hz.n_cont + positions[literal.stable_bcol_id]
        column_lower[column] = float(literal.phase)
        column_upper[column] = float(literal.phase)
    highs = _highspy.Highs()

    def require_ok(status: Any, operation: str) -> None:
        if status != _highspy.HighsStatus.kOk:
            raise LocalizedPhaseConflictOracleError(f"highs_{operation}_failed")

    require_ok(highs.setOptionValue("output_flag", False), "output_flag")
    require_ok(highs.setOptionValue("solver", "simplex"), "solver")
    require_ok(highs.setOptionValue("presolve", "off"), "presolve")
    require_ok(highs.setOptionValue("threads", int(_highs_process_threads())), "threads")
    require_ok(
        highs.setOptionValue("small_matrix_value", _TINY_PROPOSAL_COEFFICIENT),
        "small_matrix_value",
    )
    empty_i32 = np.zeros(0, dtype=np.int32)
    empty_f64 = np.zeros(0, dtype=np.float64)
    require_ok(
        highs.addCols(
            n_variables,
            np.zeros(n_variables, dtype=np.float64),
            column_lower,
            column_upper,
            0,
            empty_i32,
            empty_i32,
            empty_f64,
        ),
        "add_columns",
    )
    require_ok(
        highs.addRows(
            len(mapping),
            row_lower,
            row_upper,
            int(matrix.nnz),
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data.astype(np.float64),
        ),
        "add_rows",
    )
    if (
        int(highs.getNumCol()) != n_variables
        or int(highs.getNumRow()) != len(mapping)
        or int(highs.getNumNz()) != int(matrix.nnz)
    ):
        raise LocalizedPhaseConflictOracleError("highs_topology_postcondition_failed")
    _check_deadline(deadline, "deadline_after_local_model")
    build_seconds = time.monotonic() - build_start
    solve_start = time.monotonic()
    remaining = deadline - solve_start
    if remaining <= 0.0:
        raise TimeoutError("deadline_before_local_solve")
    require_ok(highs.setOptionValue("time_limit", float(remaining)), "time_limit")
    run_status = highs.run()
    solve_seconds = time.monotonic() - solve_start
    if time.monotonic() >= deadline or run_status != _highspy.HighsStatus.kOk:
        return _TierSolve(
            "feasible_or_unknown", mapping, map_digest, None, int(matrix.nnz), build_seconds, solve_seconds
        )
    if highs.getModelStatus() != _highspy.HighsModelStatus.kInfeasible:
        return _TierSolve(
            "feasible_or_unknown", mapping, map_digest, None, int(matrix.nnz), build_seconds, solve_seconds
        )
    ray_status, ray_exists = highs.getDualRayExist()
    if ray_status != _highspy.HighsStatus.kOk or not ray_exists:
        return _TierSolve(
            "infeasible_without_ray", mapping, map_digest, None, int(matrix.nnz), build_seconds, solve_seconds
        )
    ray_status, ray_exists, raw_ray = highs.getDualRay()
    if ray_status != _highspy.HighsStatus.kOk or not ray_exists or time.monotonic() >= deadline:
        return _TierSolve(
            "infeasible_without_ray", mapping, map_digest, None, int(matrix.nnz), build_seconds, solve_seconds
        )
    ray = np.asarray(raw_ray, dtype=np.float64).reshape(-1)
    if ray.size != len(mapping) or not np.all(np.isfinite(ray)) or not np.any(ray != 0.0):
        return _TierSolve(
            "infeasible_without_ray", mapping, map_digest, None, int(matrix.nnz), build_seconds, solve_seconds
        )
    return _TierSolve(
        "infeasible_with_ray",
        mapping,
        map_digest,
        tuple(float(item) for item in ray),
        int(matrix.nnz),
        build_seconds,
        solve_seconds,
    )


def _zero_pad_ray(
    local_ray: Sequence[float],
    rows: Sequence[int],
    *,
    full_rows: int,
) -> np.ndarray:
    mapping = tuple(_strict_int(row, name="ray_global_row") for row in rows)
    if any(right <= left for left, right in zip(mapping, mapping[1:])):
        raise LocalizedPhaseConflictOracleError("ray_row_map_not_strictly_increasing")
    ray = np.asarray(local_ray, dtype=np.float64).reshape(-1)
    if (
        ray.size != len(mapping)
        or not np.all(np.isfinite(ray))
        or any(row < 0 or row >= full_rows for row in mapping)
    ):
        raise LocalizedPhaseConflictOracleError("localized_ray_or_map_malformed")
    full = np.zeros(full_rows, dtype=np.float64)
    full[np.asarray(mapping, dtype=np.int64)] = ray
    return full


def _tier_payload(record: LocalizedTierTelemetry, *, include_digest: bool) -> dict[str, Any]:
    payload = {
        "schema": "act.localized_phase_conflict.tier.v1",
        "tier_row_cap": record.tier_row_cap,
        "expansion_depth": record.expansion_depth,
        "ordered_row_refs": [[ref.kind, ref.local_row] for ref in record.ordered_row_refs],
        "ordered_global_row_ids": list(record.ordered_global_row_ids),
        "row_mapping_sha256": record.row_mapping_sha256,
        "selected_columns": record.selected_columns,
        "selected_source_nonzeros": record.selected_source_nonzeros,
        "model_columns": record.model_columns,
        "model_nonzeros": record.model_nonzeros,
        "build_seconds_hex": float(record.build_seconds).hex(),
        "solve_seconds_hex": float(record.solve_seconds).hex(),
        "status": record.status,
        "ray_nonzero_rows": record.ray_nonzero_rows,
        "exact_replay_status": record.exact_replay_status,
    }
    if include_digest:
        payload["telemetry_sha256"] = record.telemetry_sha256
    return payload


def _make_tier(
    *,
    cap: int,
    depth: int,
    frame: _IncidenceFrame,
    rows: Tuple[int, ...],
    mapping_digest: str,
    selected_columns: int,
    source_nnz: int,
    model_columns: int,
    solve: _TierSolve,
    replay: str,
) -> LocalizedTierTelemetry:
    placeholder = LocalizedTierTelemetry(
        tier_row_cap=cap,
        expansion_depth=depth,
        ordered_row_refs=tuple(_row_ref(frame, row) for row in rows),
        ordered_global_row_ids=rows,
        row_mapping_sha256=mapping_digest,
        selected_columns=selected_columns,
        selected_source_nonzeros=source_nnz,
        model_columns=model_columns,
        model_nonzeros=solve.model_nonzeros,
        build_seconds=float(solve.build_seconds),
        solve_seconds=float(solve.solve_seconds),
        status=solve.status,
        ray_nonzero_rows=(0 if solve.local_ray is None else sum(value != 0.0 for value in solve.local_ray)),
        exact_replay_status=replay,
        telemetry_sha256="",
    )
    return LocalizedTierTelemetry(
        **{
            **placeholder.__dict__,
            "telemetry_sha256": _sha256(_tier_payload(placeholder, include_digest=False)),
        }
    )


def _result_payload(
    result: LocalizedPhaseConflictOracleResult,
    *,
    include_digest: bool,
) -> dict[str, Any]:
    payload = {
        "schema": "act.localized_phase_conflict.result.v1",
        "status": result.status,
        "reason": result.reason,
        "literals": [
            [literal.stable_bcol_id, literal.phase, literal.binding_digest]
            for literal in result.literals
        ],
        "certificate_sha256": (
            None if result.certificate is None else result.certificate.certificate_sha256
        ),
        "edge_accepted": result.edge_accepted,
        "parent_semantic_digest": result.parent_semantic_digest,
        "terminal_parent_semantic_digest": result.terminal_parent_semantic_digest,
        "property_digest": result.property_digest,
        "ordered_source_frame_sha256": result.ordered_source_frame_sha256,
        "terminal_source_frame_sha256": result.terminal_source_frame_sha256,
        "parent_unchanged": result.parent_unchanged,
        "row_tiers": list(result.row_tiers),
        "max_selected_nnz": result.max_selected_nnz,
        "max_source_terms": result.max_source_terms,
        "pattern_peak_byte_cap": result.pattern_peak_byte_cap,
        "snapshot_buffer_byte_cap": result.snapshot_buffer_byte_cap,
        "snapshot_buffer_bytes": result.snapshot_buffer_bytes,
        "snapshot_seconds_hex": float(result.snapshot_seconds).hex(),
        "incidence_row_cap": result.incidence_row_cap,
        "frontier_column_cap": result.frontier_column_cap,
        "candidate_budget_seconds_hex": float(result.candidate_budget_seconds).hex(),
        "tiers": [_tier_payload(tier, include_digest=True) for tier in result.tiers],
        "telemetry_sha256": result.telemetry_sha256,
        "proof_authority": result.proof_authority,
    }
    if include_digest:
        payload["result_sha256"] = result.result_sha256
    return payload


def _finish_result(
    *,
    status: str,
    reason: str,
    pair: Tuple[PhaseLiteral, PhaseLiteral],
    certificate: Optional[ExactDualRayConflictCertificate],
    parent_digest: str,
    property_digest: str,
    source_frame_digest: str,
    row_tiers: Tuple[int, ...],
    max_selected_nnz: int,
    max_source_terms: int,
    snapshot_buffer_bytes: int,
    snapshot_seconds: float,
    candidate_budget_seconds: float,
    tiers: Tuple[LocalizedTierTelemetry, ...],
    hz: SparseHZono,
    deadline: float,
    seal_terminal: bool = True,
    live_hz: Optional[SparseHZono] = None,
) -> LocalizedPhaseConflictOracleResult:
    terminal_parent: Optional[str] = None
    terminal_source: Optional[str] = None
    unchanged = False
    if seal_terminal:
        try:
            _check_deadline(deadline, "deadline_before_terminal_parent_seal")
            terminal_parent = sparse_hz_semantic_digest(hz)
            _check_deadline(deadline, "deadline_after_terminal_parent_seal")
            terminal_source = _ordered_source_frame_digest(
                hz,
                parent_digest=terminal_parent,
                deadline=deadline,
            )
            unchanged = (
                terminal_parent == parent_digest
                and terminal_source == source_frame_digest
            )
            if not unchanged:
                status = "parent_mutated"
                reason = "terminal_parent_or_source_frame_mismatch"
                certificate = None
            elif certificate is not None:
                # Re-read the no-alias private parent after candidate replay.
                # Caller-owned ABA mutations cannot reach these buffers; the
                # second check also catches accidental private mutation.
                if not verify_exact_dual_ray_conflict_certificate(
                    hz,
                    certificate,
                    property_digest=property_digest,
                    deadline=deadline,
                    max_source_terms=max_source_terms,
                ):
                    certificate = None
                    status = "no_certified_conflict"
                    reason = "terminal_full_parent_fraction_replay_rejected"
                _check_deadline(
                    deadline, "deadline_before_post_replay_parent_seal"
                )
                post_parent = sparse_hz_semantic_digest(hz)
                _check_deadline(
                    deadline, "deadline_after_post_replay_parent_seal"
                )
                post_source = _ordered_source_frame_digest(
                    hz,
                    parent_digest=post_parent,
                    deadline=deadline,
                )
                if (
                    post_parent != parent_digest
                    or post_source != source_frame_digest
                ):
                    terminal_parent = post_parent
                    terminal_source = post_source
                    unchanged = False
                    certificate = None
                    status = "parent_mutated"
                    reason = "post_replay_parent_or_source_frame_mismatch"
            if unchanged and live_hz is not None and live_hz is not hz:
                _check_deadline(
                    deadline, "deadline_before_live_parent_terminal_seal"
                )
                live_terminal_parent = sparse_hz_semantic_digest(live_hz)
                _check_deadline(
                    deadline, "deadline_after_live_parent_terminal_seal"
                )
                live_terminal_source = _ordered_source_frame_digest(
                    live_hz,
                    parent_digest=live_terminal_parent,
                    deadline=deadline,
                )
                if (
                    live_terminal_parent != parent_digest
                    or live_terminal_source != source_frame_digest
                ):
                    unchanged = False
                    certificate = None
                    status = "parent_mutated"
                    reason = "live_parent_terminal_seal_mismatch"
        except TimeoutError as exc:
            certificate = None
            unchanged = False
            status = "deadline_expired"
            reason = str(exc)
        except Exception as exc:
            certificate = None
            unchanged = False
            status = "terminal_seal_failed"
            reason = type(exc).__name__ + ":" + str(exc)
    edge = certificate is not None and unchanged
    if status == "certified_conflict" and not edge:
        status = "no_certified_conflict"
        reason = "certificate_missing_or_terminal_seal_failed"
    telemetry_digest = _sha256(
        {
            "schema": "act.localized_phase_conflict.telemetry.v1",
            "tiers": [_tier_payload(tier, include_digest=True) for tier in tiers],
        }
    )
    placeholder = LocalizedPhaseConflictOracleResult(
        status=status,
        reason=reason,
        literals=pair,
        certificate=certificate,
        edge_accepted=edge,
        parent_semantic_digest=parent_digest,
        terminal_parent_semantic_digest=terminal_parent,
        property_digest=property_digest,
        ordered_source_frame_sha256=source_frame_digest,
        terminal_source_frame_sha256=terminal_source,
        parent_unchanged=unchanged,
        row_tiers=row_tiers,
        max_selected_nnz=max_selected_nnz,
        max_source_terms=max_source_terms,
        pattern_peak_byte_cap=_HARD_PATTERN_PEAK_BYTES,
        snapshot_buffer_byte_cap=_HARD_SNAPSHOT_BUFFER_BYTES,
        snapshot_buffer_bytes=snapshot_buffer_bytes,
        snapshot_seconds=float(snapshot_seconds),
        incidence_row_cap=_HARD_INCIDENCE_ROWS,
        frontier_column_cap=_HARD_FRONTIER_COLUMNS,
        candidate_budget_seconds=float(candidate_budget_seconds),
        tiers=tiers,
        telemetry_sha256=telemetry_digest,
        result_sha256="",
    )
    return LocalizedPhaseConflictOracleResult(
        **{
            **placeholder.__dict__,
            "result_sha256": _sha256(_result_payload(placeholder, include_digest=False)),
        }
    )


def run_localized_phase_conflict_oracle_candidate(
    hz: SparseHZono,
    pair: Tuple[PhaseLiteral, PhaseLiteral],
    *,
    property_digest: str,
    parent_digest: str,
    source_frame_digest: str,
    deadline: float,
    enabled: bool = False,
    row_tiers: Tuple[int, ...] = _STRICT_ROW_TIERS,
    max_selected_nnz: int = _HARD_MAX_SELECTED_NNZ,
    max_source_terms: int = 128,
) -> LocalizedPhaseConflictOracleResult:
    """Propose one localized pair edge; exact full-parent replay is mandatory."""

    if type(hz) is not SparseHZono:
        raise LocalizedPhaseConflictOracleError("parent_not_exact_sparse_hz")
    if type(enabled) is not bool:
        raise LocalizedPhaseConflictOracleError("enabled_not_bool")
    if type(pair) is not tuple or len(pair) != 2 or not all(
        type(literal) is PhaseLiteral for literal in pair
    ):
        raise LocalizedPhaseConflictOracleError("pair_not_exact_two_phase_literals")
    if not _valid_sha256(property_digest) or not _valid_sha256(parent_digest) or not _valid_sha256(source_frame_digest):
        raise LocalizedPhaseConflictOracleError("caller_digest_malformed")
    if isinstance(deadline, (bool, np.bool_)):
        raise LocalizedPhaseConflictOracleError("deadline_not_float")
    try:
        deadline_value = float(deadline)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LocalizedPhaseConflictOracleError("deadline_not_float") from exc
    if not math.isfinite(deadline_value):
        raise LocalizedPhaseConflictOracleError("deadline_nonfinite")
    tiers_value = tuple(_strict_int(value, name="row_tier") for value in row_tiers)
    if tiers_value != _STRICT_ROW_TIERS:
        raise LocalizedPhaseConflictOracleError("row_tiers_not_strict_default")
    nnz_cap = _strict_int(max_selected_nnz, name="max_selected_nnz")
    if nnz_cap < 1 or nnz_cap > _HARD_MAX_SELECTED_NNZ:
        raise LocalizedPhaseConflictOracleError("max_selected_nnz_out_of_range")
    source_cap = _strict_int(max_source_terms, name="max_source_terms")
    if source_cap < 1 or source_cap > 256:
        raise LocalizedPhaseConflictOracleError("max_source_terms_out_of_range")

    try:
        ordered_pair = _ordered_pair(*pair)
    except Exception as exc:
        raise LocalizedPhaseConflictOracleError("pair_not_distinct") from exc
    start = time.monotonic()
    remaining = deadline_value - start
    candidate_budget = max(0.0, 0.25 * remaining)
    candidate_deadline = start + candidate_budget
    if not enabled:
        return _finish_result(
            status="disabled",
            reason="localized_candidate_default_off",
            pair=ordered_pair,
            certificate=None,
            parent_digest=parent_digest,
            property_digest=property_digest,
            source_frame_digest=source_frame_digest,
            row_tiers=tiers_value,
            max_selected_nnz=nnz_cap,
            max_source_terms=source_cap,
            snapshot_buffer_bytes=0,
            snapshot_seconds=0.0,
            candidate_budget_seconds=candidate_budget,
            tiers=(),
            hz=hz,
            deadline=deadline_value,
            seal_terminal=False,
        )
    if remaining <= 0.0:
        return _finish_result(
            status="deadline_expired",
            reason="deadline_expired_before_candidate",
            pair=ordered_pair,
            certificate=None,
            parent_digest=parent_digest,
            property_digest=property_digest,
            source_frame_digest=source_frame_digest,
            row_tiers=tiers_value,
            max_selected_nnz=nnz_cap,
            max_source_terms=source_cap,
            snapshot_buffer_bytes=0,
            snapshot_seconds=0.0,
            candidate_budget_seconds=candidate_budget,
            tiers=(),
            hz=hz,
            deadline=deadline_value,
            seal_terminal=False,
        )

    snapshot_buffer_bytes = 0
    snapshot_seconds = 0.0
    try:
        _check_deadline(
            candidate_deadline, "deadline_before_parent_validation"
        )
        live_parent = sparse_hz_semantic_digest(hz)
        _check_deadline(
            candidate_deadline, "deadline_after_parent_digest"
        )
        if live_parent != parent_digest:
            raise LocalizedPhaseConflictOracleError(
                "parent_digest_mismatch"
            )
        private_hz, snapshot_buffer_bytes, snapshot_seconds = (
            _private_parent_snapshot(
                hz,
                expected_parent_digest=parent_digest,
                deadline=candidate_deadline,
            )
        )
        blocks = _constraint_blocks(private_hz)
        _check_deadline(
            candidate_deadline, "deadline_after_private_parent_validation"
        )
        private_source = _ordered_source_frame_digest(
            private_hz,
            parent_digest=parent_digest,
            deadline=candidate_deadline,
        )
        if private_source != source_frame_digest:
            raise LocalizedPhaseConflictOracleError(
                "source_frame_digest_mismatch"
            )
    except Exception as exc:
        if isinstance(exc, TimeoutError) or time.monotonic() >= candidate_deadline:
            return _finish_result(
                status="deadline_expired",
                reason="deadline_during_parent_validation",
                pair=ordered_pair,
                certificate=None,
                parent_digest=parent_digest,
                property_digest=property_digest,
                source_frame_digest=source_frame_digest,
                row_tiers=tiers_value,
                max_selected_nnz=nnz_cap,
                max_source_terms=source_cap,
                snapshot_buffer_bytes=snapshot_buffer_bytes,
                snapshot_seconds=snapshot_seconds,
                candidate_budget_seconds=candidate_budget,
                tiers=(),
                hz=hz,
                deadline=deadline_value,
                seal_terminal=False,
            )
        raise
    positions = _stable_position_map(private_hz)
    for literal in ordered_pair:
        if (
            type(literal.stable_bcol_id) is not int
            or type(literal.phase) is not int
            or literal.phase not in {-1, 1}
            or literal.stable_bcol_id not in positions
            or literal.binding_digest
            != _literal_binding_digest(
                parent_digest=parent_digest,
                property_digest=property_digest,
                stable_bcol_id=literal.stable_bcol_id,
                phase=literal.phase,
            )
        ):
            raise LocalizedPhaseConflictOracleError("literal_binding_invalid")

    records: list[LocalizedTierTelemetry] = []
    certificate = None
    status = "no_certified_conflict"
    reason = "all_localized_tiers_exhausted_without_exact_edge"
    try:
        frame = _build_incidence_frame(
            private_hz, blocks, deadline=candidate_deadline
        )
        selected_binary_columns = tuple(
            private_hz.n_cont + positions[literal.stable_bcol_id]
            for literal in ordered_pair
        )
        seed_rows = _rows_for_columns(
            frame, selected_binary_columns, deadline=candidate_deadline
        )
        discovered_depth = {row: 0 for row in seed_rows}
        known_columns = set(selected_binary_columns)
        previous_rows: Optional[Tuple[int, ...]] = None
        for depth, cap in enumerate(tiers_value):
            _check_deadline(candidate_deadline, "localized_candidate_budget_expired")
            rows, source_nnz = _select_prefix(
                private_hz,
                blocks,
                discovered_depth,
                row_cap=cap,
                nnz_cap=nnz_cap,
            )
            if not rows or rows == previous_rows:
                reason = "localized_incidence_saturated_without_exact_edge"
                break
            previous_rows = rows
            expected_map_digest = _mapping_digest(
                parent_digest=parent_digest,
                source_frame_digest=source_frame_digest,
                rows=rows,
            )
            solve = _solve_tier(
                private_hz,
                blocks,
                ordered_pair,
                rows,
                parent_digest=parent_digest,
                source_frame_digest=source_frame_digest,
                deadline=candidate_deadline,
            )
            if solve.ordered_global_row_ids != rows or solve.row_mapping_sha256 != expected_map_digest:
                raise LocalizedPhaseConflictOracleError("solver_row_map_binding_mismatch")
            replay_status = "not_attempted"
            if solve.local_ray is not None:
                full_ray = _zero_pad_ray(
                    solve.local_ray,
                    solve.ordered_global_row_ids,
                    full_rows=private_hz.n_ub + private_hz.n_eq,
                )
                certificate = exact_certificate_from_highs_dual_ray_candidate(
                    private_hz,
                    ordered_pair,
                    full_ray,
                    parent_digest=parent_digest,
                    property_digest=property_digest,
                    source_frame_digest=source_frame_digest,
                    deadline=deadline_value,
                    max_source_terms=source_cap,
                )
                replay_status = "accepted" if certificate is not None else "rejected"
            records.append(
                _make_tier(
                    cap=cap,
                    depth=depth,
                    frame=frame,
                    rows=rows,
                    mapping_digest=expected_map_digest,
                    selected_columns=len(known_columns),
                    source_nnz=source_nnz,
                    model_columns=private_hz.n_cont + private_hz.n_bin,
                    solve=solve,
                    replay=replay_status,
                )
            )
            if certificate is not None:
                status = "certified_conflict"
                reason = "localized_ray_accepted_by_full_parent_fraction_replay"
                break
            expanded_columns = _columns_for_rows(
                private_hz, blocks, rows, deadline=candidate_deadline
            )
            known_columns.update(expanded_columns)
            expanded_rows = _rows_for_columns(
                frame, known_columns, deadline=candidate_deadline
            )
            for row in sorted(expanded_rows):
                discovered_depth.setdefault(row, depth + 1)
    except TimeoutError as exc:
        status = "deadline_expired"
        reason = str(exc)
        certificate = None
    except MemoryError as exc:
        status = "resource_cap"
        reason = str(exc)
        certificate = None
    except (
        LocalizedPhaseConflictOracleError,
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ) as exc:
        status = "candidate_error"
        reason = type(exc).__name__ + ":" + str(exc)
        certificate = None

    return _finish_result(
        status=status,
        reason=reason,
        pair=ordered_pair,
        certificate=certificate,
        parent_digest=parent_digest,
        property_digest=property_digest,
        source_frame_digest=source_frame_digest,
        row_tiers=tiers_value,
        max_selected_nnz=nnz_cap,
        max_source_terms=source_cap,
        snapshot_buffer_bytes=snapshot_buffer_bytes,
        snapshot_seconds=snapshot_seconds,
        candidate_budget_seconds=candidate_budget,
        tiers=tuple(records),
        hz=private_hz,
        deadline=deadline_value,
        live_hz=hz,
    )


__all__ = [
    "LocalizedPhaseConflictOracleError",
    "LocalizedPhaseConflictOracleResult",
    "LocalizedTierTelemetry",
    "RowRef",
    "run_localized_phase_conflict_oracle_candidate",
]
