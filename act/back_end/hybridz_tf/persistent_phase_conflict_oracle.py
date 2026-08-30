#!/usr/bin/env python3
"""Persistent HiGHS dual-ray candidates for exact phase conflicts.

This module is an isolated, candidate-only successor to the pair-LP loop in
``property_phase_conflict_clique``.  One continuous HZ relaxation is loaded
into HiGHS, a pair is fixed by changing two column bounds, and the retained
simplex basis is reused for the next pair.

HiGHS never has proof authority here.  An infeasibility dual ray merely
proposes nonnegative source-row multipliers.  A separate sparse
``fractions.Fraction`` replay reads the original live ``SparseHZono`` rows,
reconstructs the exact box cancellation, and accepts a conflict only when the
resulting contradiction is strictly negative.  Results and timing telemetry
remain ``proof_authority=False`` and are not connected to the verifier/BaB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
import secrets
import threading
import time
from typing import Any, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp

try:  # Candidate-only optional backend.
    import highspy as _highspy
except Exception:  # pragma: no cover - exercised by fail-closed tests.
    _highspy = None

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    ordered_property_digest,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
    _copy_parent_with_clique_cut,
    _derive_property_literals,
    _highs_property_upper,
    _literal_binding_digest,
    _ordered_pair,
    _stable_position_map,
    _strict_safe_candidate,
    _variable_bounds,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _highs_process_threads,
)


class PersistentConflictOracleError(ValueError):
    """Malformed or unsupported candidate invocation; no edge is emitted."""


@dataclass(frozen=True)
class ExactSourceTermV2:
    """One canonical nonnegative source-row multiplier."""

    global_row_index: int
    kind: str
    local_row_index: int
    numerator: int
    denominator: int
    source_row_sha256: str

    @property
    def multiplier(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class ExactDualRayConflictCertificate:
    """Exact source proof; bound terms are deterministically rebuilt."""

    literals: Tuple[PhaseLiteral, PhaseLiteral]
    parent_semantic_digest: str
    property_digest: str
    ordered_source_frame_sha256: str
    source_terms: Tuple[ExactSourceTermV2, ...]
    contradiction_numerator: int
    contradiction_denominator: int
    rationalization: str
    certificate_sha256: str
    arithmetic: str = "sparse_Fraction_exact_replay_v2"
    proof_authority: bool = False

    @property
    def contradiction(self) -> Fraction:
        return Fraction(
            self.contradiction_numerator,
            self.contradiction_denominator,
        )


@dataclass(frozen=True)
class PersistentPairRecord:
    """One canonical pair outcome from the persistent candidate engine."""

    literals: Tuple[PhaseLiteral, PhaseLiteral]
    status: str
    ray_nonzero_rows: int
    certificate_sha256: Optional[str]
    rationalization: Optional[str]


@dataclass(frozen=True)
class PersistentConflictOracleResult:
    """Exact edge candidates plus diagnostic, non-authoritative telemetry."""

    status: str
    reason: str
    literals: Tuple[PhaseLiteral, ...]
    records: Tuple[PersistentPairRecord, ...]
    certificates: Tuple[ExactDualRayConflictCertificate, ...]
    parent_semantic_digest: str
    property_digest: str
    ordered_source_frame_sha256: str
    telemetry: Mapping[str, Any]
    proof_authority: bool = False


@dataclass(frozen=True)
class PersistentPCPCCInvocationSpec:
    """Caller-owned, pre-registered config for one full candidate run."""

    nonce: str
    gate_id: str
    parent_semantic_digest: str
    property_digest: str
    deadline: float
    caps: Tuple[Tuple[str, int], ...]
    invocation_sha256: str


@dataclass(frozen=True)
class PersistentPCPCCResult:
    """Full tightness toy around the persistent exact edge oracle."""

    status: str
    hz: Optional[SparseHZono]
    invocation: PersistentPCPCCInvocationSpec
    oracle_result: PersistentConflictOracleResult
    pre_property_uppers: Tuple[Optional[float], ...]
    post_property_uppers: Tuple[Optional[float], ...]
    wall_seconds: float
    proof_authority: bool = False
    _live_capability: object = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class NativeSplitRowObjectiveDualProposal:
    """One closed-model, non-authoritative HiGHS dual proposal.

    The two arrays retain HiGHS' minimization-dual convention.  They are
    independent, read-only binary64 copies of the original upper/equality
    row slices; consumers must still replay them through an independent
    certificate checker before drawing any conclusion.
    """

    upper_row_dual: np.ndarray
    equality_row_dual: np.ndarray
    solver_minimization_objective: float
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    verdict_authority: bool = False


_SOURCE_KINDS = {"upper", "equality_pos", "equality_neg"}
_RECORD_STATUSES = {
    "certified_conflict",
    "feasible_or_unknown",
    "infeasible_without_ray",
    "exact_replay_rejected",
}
_RATIONALIZATION_DENOMINATORS = (16, 256, 4096, 65536)
_RATIONALIZATION_NAMES = {
    "raw_f64_exact_dyadic",
    "normalized_f64_exact_dyadic",
    *{
        f"normalized_limit_denominator_{denominator}"
        for denominator in _RATIONALIZATION_DENOMINATORS
    },
    *{
        (
            "normalized_capped_support_limit_denominator_"
            f"{denominator}"
        )
        for denominator in _RATIONALIZATION_DENOMINATORS
    },
}
_PROCESS_KEY = secrets.token_bytes(32)
_CAPABILITY_SENTINEL = object()
_LIVE_CAPABILITY_LOCK = threading.Lock()
_MAX_LIVE_RESULTS = 256
_MAX_LIVE_RESULT_AGE_SECONDS = 300.0
_MAX_CLAIMED_INVOCATIONS = 512
_CANDIDATE_DUST_ABS = 1.0e-12
_MAX_BINARY_CHANGE_COEFFICIENTS = 65536
_CANDIDATE_SCAN_CHUNK = 1 << 18
_INT32_MAX = int(np.iinfo(np.int32).max)
_NATIVE_OBJECTIVE_PROPOSAL_SCHEMA = (
    "act.hybridz.native_split_row_objective_dual_proposal.v1"
)
_NATIVE_OBJECTIVE_PROPOSAL_BACKEND = (
    "highspy_one_shot_simplex_presolve_split_rows_v1"
)


@dataclass(frozen=True)
class _CandidateCSRBlock:
    """Validated CSR view used only while the HiGHS model is loaded."""

    name: str
    rows: int
    columns: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    kept_nonzeros: int


def _candidate_deadline(deadline: float, operation: str) -> None:
    if time.monotonic() >= deadline:
        raise PersistentConflictOracleError(
            f"deadline_expired_during_{operation}"
        )


def _validated_candidate_block(
    matrix: sp.csr_matrix,
    *,
    rows: int,
    columns: int,
    name: str,
    deadline: float,
) -> _CandidateCSRBlock:
    """Return a canonical source-block view without a whole-model stack.

    Live ``SparseHZono`` matrices are canonical CSR in normal operation, so
    this path only borrows their arrays.  A malformed/noncanonical block is
    canonicalized in isolation to preserve the legacy proposal semantics;
    it is never combined with another full-size sparse matrix.
    """

    _candidate_deadline(deadline, f"validate_{name}")
    if (
        not sp.isspmatrix_csr(matrix)
        or matrix.shape != (rows, columns)
    ):
        raise PersistentConflictOracleError(
            f"candidate_{name}_shape_or_format_invalid"
        )
    if (
        rows < 0
        or columns < 0
        or rows > _INT32_MAX
        or columns > _INT32_MAX
    ):
        raise PersistentConflictOracleError(
            f"candidate_{name}_dimension_exceeds_int32"
        )

    indptr = np.asarray(matrix.indptr)
    indices = np.asarray(matrix.indices)
    data = np.asarray(matrix.data)
    if (
        indptr.ndim != 1
        or indices.ndim != 1
        or data.ndim != 1
        or indptr.size != rows + 1
        or indices.size != data.size
        or not np.issubdtype(indptr.dtype, np.integer)
        or not np.issubdtype(indices.dtype, np.integer)
        or int(data.size) > _INT32_MAX
        or int(indptr[0]) != 0
        or int(indptr[-1]) != int(data.size)
    ):
        raise PersistentConflictOracleError(
            f"candidate_{name}_csr_structure_invalid"
        )
    if np.any(indptr[1:] < indptr[:-1]):
        raise PersistentConflictOracleError(
            f"candidate_{name}_row_pointer_invalid"
        )

    canonical = True
    for row in range(rows):
        if row % 1024 == 0:
            _candidate_deadline(deadline, f"validate_{name}")
        start = int(indptr[row])
        stop = int(indptr[row + 1])
        row_indices = indices[start:stop]
        if row_indices.size:
            if (
                int(np.min(row_indices)) < 0
                or int(np.max(row_indices)) >= columns
            ):
                raise PersistentConflictOracleError(
                    f"candidate_{name}_column_out_of_range"
                )
            if row_indices.size > 1 and np.any(
                row_indices[1:] <= row_indices[:-1]
            ):
                canonical = False
    if not canonical:
        canonical_matrix = matrix.copy()
        canonical_matrix.sum_duplicates()
        canonical_matrix.sort_indices()
        indptr = np.asarray(canonical_matrix.indptr)
        indices = np.asarray(canonical_matrix.indices)
        data = np.asarray(canonical_matrix.data)
        if int(data.size) > _INT32_MAX:
            raise PersistentConflictOracleError(
                f"candidate_{name}_nonzeros_exceed_int32"
            )

    kept_nonzeros = 0
    for start in range(0, int(data.size), _CANDIDATE_SCAN_CHUNK):
        _candidate_deadline(deadline, f"scan_{name}")
        chunk = data[start : start + _CANDIDATE_SCAN_CHUNK]
        if not np.all(np.isfinite(chunk)):
            raise PersistentConflictOracleError(
                "candidate_matrix_nonfinite"
            )
        kept_nonzeros += int(
            np.count_nonzero(np.abs(chunk) > _CANDIDATE_DUST_ABS)
        )
    _candidate_deadline(deadline, f"scan_{name}")
    return _CandidateCSRBlock(
        name=name,
        rows=rows,
        columns=columns,
        indptr=indptr,
        indices=indices,
        data=data,
        kept_nonzeros=kept_nonzeros,
    )


def _compact_candidate_block(
    block: _CandidateCSRBlock,
    *,
    deadline: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize only when dust removal is actually required."""

    if block.kept_nonzeros == int(block.data.size):
        return (
            block.indptr.astype(np.int32, copy=False),
            block.indices.astype(np.int32, copy=False),
            block.data.astype(np.float64, copy=False),
        )

    starts_i64 = np.empty(block.rows + 1, dtype=np.int64)
    kept_indices = np.empty(
        block.kept_nonzeros, dtype=np.int32
    )
    kept_data = np.empty(
        block.kept_nonzeros, dtype=np.float64
    )
    cursor = 0
    starts_i64[0] = 0
    for row in range(block.rows):
        if row % 1024 == 0:
            _candidate_deadline(
                deadline, f"compact_{block.name}"
            )
        start = int(block.indptr[row])
        stop = int(block.indptr[row + 1])
        values = block.data[start:stop]
        keep = np.abs(values) > _CANDIDATE_DUST_ABS
        count = int(np.count_nonzero(keep))
        if count:
            kept_indices[cursor : cursor + count] = (
                block.indices[start:stop][keep].astype(
                    np.int32, copy=False
                )
            )
            kept_data[cursor : cursor + count] = values[
                keep
            ].astype(np.float64, copy=False)
            cursor += count
        starts_i64[row + 1] = cursor
    if cursor != block.kept_nonzeros:
        raise PersistentConflictOracleError(
            f"candidate_{block.name}_compaction_postcondition_failed"
        )
    return (
        starts_i64.astype(np.int32, copy=False),
        kept_indices.astype(np.int32, copy=False),
        kept_data.astype(np.float64, copy=False),
    )


def _copy_candidate_row(
    block: _CandidateCSRBlock,
    row: int,
    *,
    column_offset: int,
    output_indices: np.ndarray,
    output_data: np.ndarray,
    cursor: int,
) -> int:
    start = int(block.indptr[row])
    stop = int(block.indptr[row + 1])
    values = block.data[start:stop]
    keep = np.abs(values) > _CANDIDATE_DUST_ABS
    count = int(np.count_nonzero(keep))
    if count:
        source_indices = block.indices[start:stop][keep]
        shifted = source_indices.astype(np.int64, copy=False)
        if column_offset:
            shifted = shifted + int(column_offset)
        if (
            int(np.min(shifted)) < 0
            or int(np.max(shifted)) > _INT32_MAX
        ):
            raise PersistentConflictOracleError(
                "candidate_merged_column_exceeds_int32"
            )
        output_indices[cursor : cursor + count] = shifted.astype(
            np.int32, copy=False
        )
        output_data[cursor : cursor + count] = values[keep].astype(
            np.float64, copy=False
        )
    return cursor + count


def _merged_candidate_arrays(
    *,
    upper_continuous: _CandidateCSRBlock,
    upper_binary: _CandidateCSRBlock,
    equality_continuous: _CandidateCSRBlock,
    equality_binary: _CandidateCSRBlock,
    continuous_columns: int,
    deadline: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-array fallback for models with too many binary edits."""

    rows = upper_continuous.rows + equality_continuous.rows
    nonzeros = sum(
        block.kept_nonzeros
        for block in (
            upper_continuous,
            upper_binary,
            equality_continuous,
            equality_binary,
        )
    )
    if nonzeros > _INT32_MAX:
        raise PersistentConflictOracleError(
            "candidate_merged_nonzeros_exceed_int32"
        )
    starts_i64 = np.empty(rows + 1, dtype=np.int64)
    indices = np.empty(nonzeros, dtype=np.int32)
    data = np.empty(nonzeros, dtype=np.float64)
    starts_i64[0] = 0
    cursor = 0
    output_row = 0
    for continuous, binary in (
        (upper_continuous, upper_binary),
        (equality_continuous, equality_binary),
    ):
        for local_row in range(continuous.rows):
            if output_row % 1024 == 0:
                _candidate_deadline(
                    deadline, "merge_candidate_rows"
                )
            cursor = _copy_candidate_row(
                continuous,
                local_row,
                column_offset=0,
                output_indices=indices,
                output_data=data,
                cursor=cursor,
            )
            cursor = _copy_candidate_row(
                binary,
                local_row,
                column_offset=continuous_columns,
                output_indices=indices,
                output_data=data,
                cursor=cursor,
            )
            output_row += 1
            starts_i64[output_row] = cursor
    if output_row != rows or cursor != nonzeros:
        raise PersistentConflictOracleError(
            "candidate_merged_topology_postcondition_failed"
        )
    return (
        starts_i64.astype(np.int32, copy=False),
        indices.astype(np.int32, copy=False),
        data.astype(np.float64, copy=False),
    )


class _LivePersistentPCPCCCapability:
    """Opaque issuer-only handle for one exact full candidate."""

    __slots__ = ("_identity",)

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _CAPABILITY_SENTINEL:
            raise TypeError("persistent PC-PCC capability is issuer-only")
        self._identity = secrets.token_hex(32)


@dataclass(frozen=True)
class _LivePersistentPCPCCRecord:
    capability: _LivePersistentPCPCCCapability
    result_ref: Any
    parent: SparseHZono
    rivals: Sequence[RivalSpec]
    invocation: PersistentPCPCCInvocationSpec
    hz: Optional[SparseHZono]
    oracle_result: PersistentConflictOracleResult
    pre_property_uppers: Tuple[Optional[float], ...]
    post_property_uppers: Tuple[Optional[float], ...]
    deadline: float
    issued_at: float
    process_id: int
    snapshot_hmac_sha256: str


_LIVE_PERSISTENT_PC_PCC_RESULTS: dict[
    int, _LivePersistentPCPCCRecord
] = {}
_CLAIMED_PERSISTENT_PC_PCC_INVOCATIONS: dict[str, float] = {}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(payload: Any) -> str:
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
        raise PersistentConflictOracleError(f"{name}_not_integer")
    return int(value)


def _fraction_payload(value: Fraction) -> Tuple[int, int]:
    return int(value.numerator), int(value.denominator)


def _fraction_within_bits(value: Fraction, maximum: int) -> bool:
    return (
        value.numerator.bit_length() <= maximum
        and value.denominator.bit_length() <= maximum
    )


def _term_payload(term: ExactSourceTermV2) -> Mapping[str, Any]:
    return {
        "global_row_index": int(term.global_row_index),
        "kind": term.kind,
        "local_row_index": int(term.local_row_index),
        "numerator": int(term.numerator),
        "denominator": int(term.denominator),
        "source_row_sha256": term.source_row_sha256,
    }


def _literal_payload(literal: PhaseLiteral) -> Mapping[str, Any]:
    return {
        "stable_bcol_id": int(literal.stable_bcol_id),
        "phase": int(literal.phase),
        "binding_digest": literal.binding_digest,
    }


def _certificate_payload(
    certificate: ExactDualRayConflictCertificate,
    *,
    include_digest: bool,
) -> Mapping[str, Any]:
    payload = {
        "schema": "act.pc_pcc.exact_dual_ray_certificate.v2",
        "literals": [
            _literal_payload(literal)
            for literal in certificate.literals
        ],
        "parent_semantic_digest": (
            certificate.parent_semantic_digest
        ),
        "property_digest": certificate.property_digest,
        "ordered_source_frame_sha256": (
            certificate.ordered_source_frame_sha256
        ),
        "source_terms": [
            _term_payload(term)
            for term in certificate.source_terms
        ],
        "contradiction": [
            int(certificate.contradiction_numerator),
            int(certificate.contradiction_denominator),
        ],
        "rationalization": certificate.rationalization,
        "arithmetic": certificate.arithmetic,
        "proof_authority": certificate.proof_authority,
    }
    if include_digest:
        payload["certificate_sha256"] = (
            certificate.certificate_sha256
        )
    return payload


def _certificate_digest(
    certificate: ExactDualRayConflictCertificate,
) -> str:
    return _canonical_sha256(
        _certificate_payload(certificate, include_digest=False)
    )


def _exact_sparse_source_row(
    hz: SparseHZono,
    kind: str,
    local_row: int,
    *,
    deadline: float,
    max_nonzeros: int,
) -> Tuple[
    Tuple[Tuple[int, Fraction], ...],
    Fraction,
    str,
    int,
]:
    """Read one source orientation from the original CSR frame exactly."""

    local_row = _strict_int(local_row, name="local_row")
    max_nonzeros = _strict_int(
        max_nonzeros, name="max_nonzeros"
    )
    if (
        max_nonzeros < 0
        or not math.isfinite(deadline)
        or time.monotonic() >= deadline
    ):
        raise PersistentConflictOracleError(
            "source_row_budget_or_deadline_invalid"
        )
    if kind == "upper":
        if local_row < 0 or local_row >= hz.n_ub:
            raise PersistentConflictOracleError(
                "upper_source_row_out_of_range"
            )
        continuous = hz.Auc
        binary = hz.Aub
        rhs_float = float(hz.ub[local_row])
        sign = 1
        global_row = local_row
    elif kind in {"equality_pos", "equality_neg"}:
        if local_row < 0 or local_row >= hz.n_eq:
            raise PersistentConflictOracleError(
                "equality_source_row_out_of_range"
            )
        continuous = hz.Ac
        binary = hz.Ab
        rhs_float = float(hz.b[local_row])
        sign = 1 if kind == "equality_pos" else -1
        global_row = hz.n_ub + local_row
    else:
        raise PersistentConflictOracleError("source_kind_invalid")
    if not math.isfinite(rhs_float):
        raise PersistentConflictOracleError("source_rhs_nonfinite")

    coefficients_list: list[Tuple[int, Fraction]] = []
    segment_digests = []
    raw_nonzeros = 0
    expected_rows = hz.n_ub if kind == "upper" else hz.n_eq
    for matrix, offset, expected_columns in (
        (continuous, 0, hz.n_cont),
        (binary, hz.n_cont, hz.n_bin),
    ):
        if (
            not sp.isspmatrix_csr(matrix)
            or matrix.shape
            != (expected_rows, expected_columns)
            or np.asarray(matrix.indptr).ndim != 1
            or np.asarray(matrix.indices).ndim != 1
            or np.asarray(matrix.data).ndim != 1
            or int(matrix.indptr.size) != expected_rows + 1
            or int(matrix.indices.size) != int(matrix.data.size)
        ):
            raise PersistentConflictOracleError(
                "source_matrix_structure_invalid"
            )
        start = int(matrix.indptr[local_row])
        stop = int(matrix.indptr[local_row + 1])
        if (
            start < 0
            or stop < start
            or stop > int(matrix.indices.size)
        ):
            raise PersistentConflictOracleError(
                "source_row_pointer_invalid"
            )
        raw_nonzeros += stop - start
        if raw_nonzeros > max_nonzeros:
            raise PersistentConflictOracleError(
                "source_row_nonzero_cap_exceeded"
            )
        raw_indices = np.asarray(
            matrix.indices[start:stop], dtype=np.int64
        )
        raw_data = np.asarray(
            matrix.data[start:stop], dtype=np.float64
        )
        segment_digests.append(
            _canonical_sha256(
                {
                    "schema": (
                        "act.pc_pcc.exact_source_csr_segment.v3"
                    ),
                    "column_offset": int(offset),
                    "column_count": int(expected_columns),
                    "stored_nonzeros": int(stop - start),
                    "indices_sha256": hashlib.sha256(
                        np.ascontiguousarray(
                            raw_indices
                        ).tobytes()
                    ).hexdigest(),
                    "data_sha256": hashlib.sha256(
                        np.ascontiguousarray(raw_data).tobytes()
                    ).hexdigest(),
                }
            )
        )
        if time.monotonic() >= deadline:
            raise PersistentConflictOracleError(
                "deadline_expired_after_source_segment_hash"
            )
        previous_raw_column = -1
        for position in range(start, stop):
            if (
                (position - start) % 1024 == 0
                and time.monotonic() >= deadline
            ):
                raise PersistentConflictOracleError(
                    "deadline_expired_in_source_row"
                )
            value = float(matrix.data[position])
            if not math.isfinite(value):
                raise PersistentConflictOracleError(
                    "source_coefficient_nonfinite"
                )
            raw_column = int(matrix.indices[position])
            if (
                raw_column < 0
                or raw_column >= expected_columns
                or raw_column <= previous_raw_column
            ):
                raise PersistentConflictOracleError(
                    "source_column_range_or_order_invalid"
                )
            previous_raw_column = raw_column
            column = offset + raw_column
            exact_value = Fraction(sign) * Fraction.from_float(value)
            if exact_value:
                coefficients_list.append((column, exact_value))
        if time.monotonic() >= deadline:
            raise PersistentConflictOracleError(
                "deadline_expired_after_source_matrix"
            )
    coefficients = tuple(coefficients_list)
    if time.monotonic() >= deadline:
        raise PersistentConflictOracleError(
            "deadline_expired_after_source_coefficients"
        )
    rhs = Fraction(sign) * Fraction.from_float(rhs_float)
    digest = _canonical_sha256(
        {
            "schema": "act.pc_pcc.exact_source_row.v3",
            "kind": kind,
            "global_row_index": int(global_row),
            "local_row_index": int(local_row),
            "n_variables": int(hz.n_cont + hz.n_bin),
            "csr_segment_sha256": segment_digests,
            "rhs_float_hex": rhs_float.hex(),
        }
    )
    if time.monotonic() >= deadline:
        raise PersistentConflictOracleError(
            "deadline_expired_after_source_row"
        )
    return coefficients, rhs, digest, raw_nonzeros


def _ordered_source_frame_digest(
    hz: SparseHZono,
    *,
    parent_digest: str,
    deadline: float,
) -> str:
    """Bind row orientation to the already strict full-parent CSR digest."""

    if time.monotonic() >= deadline:
        raise PersistentConflictOracleError(
            "deadline_expired_before_source_frame_digest"
        )
    digest = _canonical_sha256(
        {
            "schema": "act.pc_pcc.ordered_source_frame.v3",
            "parent_semantic_digest": parent_digest,
            "row_order": "upper_then_equality",
            "n_upper": int(hz.n_ub),
            "n_equality": int(hz.n_eq),
            "n_continuous": int(hz.n_cont),
            "n_binary": int(hz.n_bin),
        }
    )
    if time.monotonic() >= deadline:
        raise PersistentConflictOracleError(
            "deadline_expired_after_source_frame_digest"
        )
    return digest


def _global_row_index(
    hz: SparseHZono,
    *,
    kind: str,
    local_row: int,
) -> int:
    if kind == "upper":
        return int(local_row)
    if kind in {"equality_pos", "equality_neg"}:
        return int(hz.n_ub + local_row)
    raise PersistentConflictOracleError("source_kind_invalid")


def _build_exact_certificate(
    hz: SparseHZono,
    literals: Tuple[PhaseLiteral, PhaseLiteral],
    *,
    weighted_sources: Sequence[Tuple[str, int, Fraction]],
    parent_digest: str,
    property_digest: str,
    source_frame_digest: str,
    rationalization: str,
    deadline: float,
    max_source_terms: int,
    max_multiplier_bits: int,
    max_exact_bits: int,
    max_exact_nonzeros: int,
) -> Optional[ExactDualRayConflictCertificate]:
    if (
        not weighted_sources
        or len(weighted_sources) > max_source_terms
    ):
        return None
    bounds = _variable_bounds(hz, literals)
    combined: dict[int, Fraction] = {}
    beta = Fraction(0)
    source_terms = []
    previous_global = -1
    exact_nonzeros = 0

    for kind, local_row, multiplier in weighted_sources:
        if time.monotonic() >= deadline:
            return None
        if (
            kind not in _SOURCE_KINDS
            or not isinstance(multiplier, Fraction)
            or multiplier <= 0
            or not _fraction_within_bits(
                multiplier, max_multiplier_bits
            )
        ):
            return None
        global_row = _global_row_index(
            hz, kind=kind, local_row=local_row
        )
        if global_row <= previous_global:
            return None
        previous_global = global_row
        coefficients, rhs, row_digest, row_nonzeros = (
            _exact_sparse_source_row(
                hz,
                kind,
                local_row,
                deadline=deadline,
                max_nonzeros=(
                    max_exact_nonzeros - exact_nonzeros
                ),
            )
        )
        exact_nonzeros += row_nonzeros
        if exact_nonzeros > max_exact_nonzeros:
            return None
        for offset, (column, coefficient) in enumerate(
            coefficients
        ):
            if (
                offset % 256 == 0
                and time.monotonic() >= deadline
            ):
                return None
            value = (
                combined.get(column, Fraction(0))
                + multiplier * coefficient
            )
            if not _fraction_within_bits(value, max_exact_bits):
                return None
            if value:
                combined[column] = value
            else:
                combined.pop(column, None)
        beta += multiplier * rhs
        if not _fraction_within_bits(beta, max_exact_bits):
            return None
        numerator, denominator = _fraction_payload(multiplier)
        source_terms.append(
            ExactSourceTermV2(
                global_row_index=global_row,
                kind=kind,
                local_row_index=int(local_row),
                numerator=numerator,
                denominator=denominator,
                source_row_sha256=row_digest,
            )
        )

    minimum = Fraction(0)
    for offset, (column, coefficient) in enumerate(
        sorted(combined.items())
    ):
        if offset % 64 == 0 and time.monotonic() >= deadline:
            return None
        lower, upper = bounds[column]
        minimum += coefficient * (
            lower if coefficient >= 0 else upper
        )
        if not _fraction_within_bits(minimum, max_exact_bits):
            return None
    contradiction = beta - minimum
    if (
        contradiction >= 0
        or not _fraction_within_bits(
            contradiction, max_exact_bits
        )
    ):
        return None
    contradiction_num, contradiction_den = _fraction_payload(
        contradiction
    )
    placeholder = ExactDualRayConflictCertificate(
        literals=_ordered_pair(*literals),
        parent_semantic_digest=parent_digest,
        property_digest=property_digest,
        ordered_source_frame_sha256=source_frame_digest,
        source_terms=tuple(source_terms),
        contradiction_numerator=contradiction_num,
        contradiction_denominator=contradiction_den,
        rationalization=rationalization,
        certificate_sha256="",
    )
    return ExactDualRayConflictCertificate(
        **{
            **placeholder.__dict__,
            "certificate_sha256": _certificate_digest(placeholder),
        }
    )


def _verify_exact_certificate_with_source_frame(
    hz: SparseHZono,
    certificate: ExactDualRayConflictCertificate,
    *,
    property_digest: str,
    parent_digest: str,
    source_frame_digest: str,
    deadline: float,
    max_source_terms: int = 128,
    max_multiplier_bits: int = 256,
    max_exact_bits: int = 4096,
    max_exact_nonzeros: int = 200000,
) -> bool:
    """Replay one certificate after a batch sealed the live source frame."""

    try:
        if (
            not isinstance(hz, SparseHZono)
            or not isinstance(
                certificate, ExactDualRayConflictCertificate
            )
            or certificate.proof_authority is not False
            or certificate.arithmetic
            != "sparse_Fraction_exact_replay_v2"
            or certificate.parent_semantic_digest
            != parent_digest
            or certificate.property_digest != property_digest
            or not _valid_sha256(
                certificate.ordered_source_frame_sha256
            )
            or certificate.ordered_source_frame_sha256
            != source_frame_digest
            or not _valid_sha256(
                certificate.certificate_sha256
            )
            or _certificate_digest(certificate)
            != certificate.certificate_sha256
            or not isinstance(certificate.source_terms, tuple)
            or not certificate.source_terms
            or len(certificate.source_terms) > max_source_terms
            or certificate.rationalization
            not in _RATIONALIZATION_NAMES
        ):
            return False
        literals = _ordered_pair(*certificate.literals)
        if literals != certificate.literals:
            return False
        for literal in literals:
            if literal.binding_digest != _literal_binding_digest(
                parent_digest=certificate.parent_semantic_digest,
                property_digest=property_digest,
                stable_bcol_id=literal.stable_bcol_id,
                phase=literal.phase,
            ):
                return False

        bounds = _variable_bounds(hz, literals)
        combined: dict[int, Fraction] = {}
        beta = Fraction(0)
        exact_nonzeros = 0
        previous_global = -1
        for term in certificate.source_terms:
            if time.monotonic() >= deadline:
                return False
            if (
                not isinstance(term, ExactSourceTermV2)
                or type(term.global_row_index) is not int
                or type(term.local_row_index) is not int
                or type(term.numerator) is not int
                or type(term.denominator) is not int
                or term.kind not in _SOURCE_KINDS
                or term.numerator <= 0
                or term.denominator <= 0
                or math.gcd(term.numerator, term.denominator) != 1
                or not _valid_sha256(term.source_row_sha256)
            ):
                return False
            multiplier = term.multiplier
            if not _fraction_within_bits(
                multiplier, max_multiplier_bits
            ):
                return False
            global_row = _global_row_index(
                hz,
                kind=term.kind,
                local_row=term.local_row_index,
            )
            if (
                global_row != term.global_row_index
                or global_row <= previous_global
            ):
                return False
            previous_global = global_row
            coefficients, rhs, live_digest, row_nonzeros = (
                _exact_sparse_source_row(
                    hz,
                    term.kind,
                    term.local_row_index,
                    deadline=deadline,
                    max_nonzeros=(
                        max_exact_nonzeros - exact_nonzeros
                    ),
                )
            )
            if live_digest != term.source_row_sha256:
                return False
            exact_nonzeros += row_nonzeros
            if exact_nonzeros > max_exact_nonzeros:
                return False
            for offset, (column, coefficient) in enumerate(
                coefficients
            ):
                if (
                    offset % 256 == 0
                    and time.monotonic() >= deadline
                ):
                    return False
                value = (
                    combined.get(column, Fraction(0))
                    + multiplier * coefficient
                )
                if not _fraction_within_bits(
                    value, max_exact_bits
                ):
                    return False
                if value:
                    combined[column] = value
                else:
                    combined.pop(column, None)
            beta += multiplier * rhs
            if not _fraction_within_bits(beta, max_exact_bits):
                return False

        minimum = Fraction(0)
        for offset, (column, coefficient) in enumerate(
            sorted(combined.items())
        ):
            if (
                offset % 64 == 0
                and time.monotonic() >= deadline
            ):
                return False
            lower, upper = bounds[column]
            minimum += coefficient * (
                lower if coefficient >= 0 else upper
            )
            if not _fraction_within_bits(
                minimum, max_exact_bits
            ):
                return False
        contradiction = beta - minimum
        return (
            contradiction < 0
            and _fraction_within_bits(
                contradiction, max_exact_bits
            )
            and contradiction == certificate.contradiction
            and math.gcd(
                certificate.contradiction_numerator,
                certificate.contradiction_denominator,
            )
            == 1
            and certificate.contradiction_denominator > 0
        )
    except (
        PersistentConflictOracleError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def verify_exact_dual_ray_conflict_certificate(
    hz: SparseHZono,
    certificate: ExactDualRayConflictCertificate,
    *,
    property_digest: str,
    deadline: Optional[float] = None,
    max_source_terms: int = 128,
    max_multiplier_bits: int = 256,
    max_exact_bits: int = 4096,
    max_exact_nonzeros: int = 200000,
) -> bool:
    """Repeat an exact sparse replay against the live parent HZ."""

    try:
        deadline_value = (
            time.monotonic() + 60.0
            if deadline is None
            else float(deadline)
        )
        if (
            not math.isfinite(deadline_value)
            or not isinstance(hz, SparseHZono)
        ):
            return False
        parent_digest = sparse_hz_semantic_digest(hz)
        source_frame_digest = _ordered_source_frame_digest(
            hz,
            parent_digest=parent_digest,
            deadline=deadline_value,
        )
        return _verify_exact_certificate_with_source_frame(
            hz,
            certificate,
            property_digest=property_digest,
            parent_digest=parent_digest,
            source_frame_digest=source_frame_digest,
            deadline=deadline_value,
            max_source_terms=max_source_terms,
            max_multiplier_bits=max_multiplier_bits,
            max_exact_bits=max_exact_bits,
            max_exact_nonzeros=max_exact_nonzeros,
        )
    except (
        PersistentConflictOracleError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _ray_source_attempts(
    hz: SparseHZono,
    raw_ray: np.ndarray,
    *,
    deadline: float,
    max_source_terms: int,
    max_multiplier_bits: int,
) -> Tuple[
    Tuple[str, Tuple[Tuple[str, int, Fraction], ...]], ...
]:
    try:
        ray = np.asarray(
            raw_ray, dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return ()
    expected_rows = hz.n_ub + hz.n_eq
    if (
        ray.size != expected_rows
        or not np.all(np.isfinite(ray))
        or not np.any(ray != 0.0)
        or time.monotonic() >= deadline
        or type(max_source_terms) is not int
        or max_source_terms < 1
        or type(max_multiplier_bits) is not int
        or max_multiplier_bits < 1
    ):
        return ()
    maximum = float(np.max(np.abs(ray)))
    if not math.isfinite(maximum) or maximum <= 0.0:
        return ()

    normalized = np.abs(ray / maximum)
    absolute = np.abs(ray)
    nonzero_rows = np.flatnonzero(ray != 0.0)
    strict_upper_signs = not np.any(
        ray[: hz.n_ub] > 0.0
    )

    output = []
    seen = set()

    def materialize(
        name: str,
        rows: np.ndarray,
        magnitudes: np.ndarray,
        *,
        denominator: Optional[int],
        zero_is_omitted: bool,
    ) -> bool:
        """Materialize at most the capped support; return False on timeout."""

        weighted_sources = []
        for offset, raw_global_row in enumerate(rows):
            if (
                offset % 256 == 0
                and time.monotonic() >= deadline
            ):
                return False
            global_row = int(raw_global_row)
            signed_value = float(ray[global_row])
            magnitude = float(magnitudes[global_row])
            if denominator is not None:
                multiplier = Fraction.from_float(
                    magnitude
                ).limit_denominator(int(denominator))
            else:
                multiplier = Fraction.from_float(magnitude)
            if multiplier == 0 and zero_is_omitted:
                continue
            if multiplier <= 0 or not _fraction_within_bits(
                multiplier, max_multiplier_bits
            ):
                return True
            if global_row < hz.n_ub:
                if signed_value >= 0.0:
                    return True
                kind = "upper"
                local_row = global_row
            else:
                local_row = global_row - hz.n_ub
                kind = (
                    "equality_pos"
                    if signed_value < 0.0
                    else "equality_neg"
                )
            weighted_sources.append(
                (kind, int(local_row), multiplier)
            )
        if (
            not weighted_sources
            or len(weighted_sources) > max_source_terms
        ):
            return True
        identity = tuple(
            (
                kind,
                row,
                multiplier.numerator,
                multiplier.denominator,
            )
            for kind, row, multiplier in weighted_sources
        )
        if identity in seen:
            return True
        seen.add(identity)
        output.append((name, tuple(weighted_sources)))
        return time.monotonic() < deadline

    # Strict dyadic attempts preserve every nonzero ray row.  Count support
    # with NumPy before constructing any Fractions so a dense large ray fails
    # the source cap in O(n) native code rather than six Python scans.
    if (
        strict_upper_signs
        and int(nonzero_rows.size) <= max_source_terms
    ):
        for name, magnitudes in (
            ("raw_f64_exact_dyadic", absolute),
            ("normalized_f64_exact_dyadic", normalized),
        ):
            if not materialize(
                name,
                nonzero_rows,
                magnitudes,
                denominator=None,
                zero_is_omitted=False,
            ):
                return tuple(output)

    for denominator in _RATIONALIZATION_DENOMINATORS:
        if time.monotonic() >= deadline:
            break
        # ``limit_denominator(D)`` maps sufficiently tiny magnitudes to zero.
        # Dropping those rows is candidate sparsification only; every retained
        # source set still needs an exact Fraction contradiction.
        threshold = 0.5 / float(denominator)
        retained = np.flatnonzero(normalized > threshold)
        if retained.size:
            wrong_upper = retained[
                (retained < hz.n_ub)
                & (ray[retained] >= 0.0)
            ]
            if wrong_upper.size:
                continue
        if int(retained.size) <= max_source_terms:
            selected = retained
            name = (
                f"normalized_limit_denominator_{denominator}"
            )
        else:
            retained_values = normalized[retained]
            kth = int(retained_values.size) - max_source_terms
            cutoff = float(
                np.partition(retained_values, kth)[kth]
            )
            greater = retained[retained_values > cutoff]
            equal = retained[retained_values == cutoff]
            remaining = max_source_terms - int(greater.size)
            selected = np.sort(
                np.concatenate([greater, equal[:remaining]])
            )
            name = (
                "normalized_capped_support_limit_denominator_"
                f"{denominator}"
            )
        if not materialize(
            name,
            selected,
            normalized,
            denominator=denominator,
            zero_is_omitted=True,
        ):
            break
    return tuple(output)


def exact_certificate_from_highs_dual_ray_candidate(
    hz: SparseHZono,
    literals: Tuple[PhaseLiteral, PhaseLiteral],
    raw_ray: np.ndarray,
    *,
    parent_digest: str,
    property_digest: str,
    source_frame_digest: str,
    deadline: float,
    max_source_terms: int = 128,
    max_multiplier_bits: int = 256,
    max_exact_bits: int = 4096,
    max_exact_nonzeros: int = 200000,
) -> Optional[ExactDualRayConflictCertificate]:
    """Try a fixed rationalization ladder with exact replay."""

    for name, weighted_sources in _ray_source_attempts(
        hz,
        raw_ray,
        deadline=deadline,
        max_source_terms=max_source_terms,
        max_multiplier_bits=max_multiplier_bits,
    ):
        if time.monotonic() >= deadline:
            return None
        certificate = _build_exact_certificate(
            hz,
            literals,
            weighted_sources=weighted_sources,
            parent_digest=parent_digest,
            property_digest=property_digest,
            source_frame_digest=source_frame_digest,
            rationalization=name,
            deadline=deadline,
            max_source_terms=max_source_terms,
            max_multiplier_bits=max_multiplier_bits,
            max_exact_bits=max_exact_bits,
            max_exact_nonzeros=max_exact_nonzeros,
        )
        if (
            certificate is not None
            and _verify_exact_certificate_with_source_frame(
                hz,
                certificate,
                property_digest=property_digest,
                parent_digest=parent_digest,
                source_frame_digest=source_frame_digest,
                deadline=deadline,
                max_source_terms=max_source_terms,
                max_multiplier_bits=max_multiplier_bits,
                max_exact_bits=max_exact_bits,
                max_exact_nonzeros=max_exact_nonzeros,
            )
        ):
            return certificate
    return None


class _PersistentHighsPairLP:
    """One mutable HiGHS model; callers must not share it across threads.

    ``solve_base_relaxation=False`` is the bounded candidate-first mode.  It
    loads exactly the same parent model but postpones the first simplex run
    until a pair has been fixed.  This is useful when the unfixed production
    relaxation is much harder than the small number of pair queries.  It
    changes proposal performance only: every accepted edge is still
    authorized solely by the full-parent exact ``Fraction`` replay above.
    """

    def __init__(
        self,
        hz: SparseHZono,
        *,
        deadline: float,
        solve_base_relaxation: bool = True,
        candidate_presolve: bool = False,
        require_split_load: bool = False,
    ) -> None:
        if _highspy is None:
            raise PersistentConflictOracleError(
                "highspy_unavailable"
            )
        if type(solve_base_relaxation) is not bool:
            raise PersistentConflictOracleError(
                "solve_base_relaxation_not_bool"
            )
        if type(candidate_presolve) is not bool:
            raise PersistentConflictOracleError(
                "candidate_presolve_not_bool"
            )
        if type(require_split_load) is not bool:
            raise PersistentConflictOracleError(
                "require_split_load_not_bool"
            )
        if solve_base_relaxation and candidate_presolve:
            raise PersistentConflictOracleError(
                "candidate_presolve_requires_candidate_first"
            )
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise PersistentConflictOracleError(
                "deadline_nonfinite"
            )
        _candidate_deadline(deadline, "candidate_model_load")
        self._hz = hz
        self._positions = _stable_position_map(hz)
        self._candidate_presolve = candidate_presolve
        self._require_split_load = require_split_load
        self._threads = int(_highs_process_threads())
        self._candidate_rows = int(hz.n_ub + hz.n_eq)
        self._candidate_columns = int(hz.n_cont + hz.n_bin)
        self._candidate_nonzeros = 0
        self._candidate_load_mode = "uninitialized"
        self._last_highs_run_time = 0.0
        self._closed = False
        self._highs = None
        self.solve_calls = 0
        self.objective_solve_calls = 0
        self.base_solve_calls = 0
        self._candidate_first = not solve_base_relaxation
        self.bound_update_calls = 0
        self.ray_calls = 0
        try:
            self._highs = _highspy.Highs()
            self._configure_highs()
            self._load_candidate_model(hz, deadline=deadline)
            if solve_base_relaxation:
                self._set_time_ceiling(deadline)
                run_status = self._highs.run()
                self.base_solve_calls += 1
                self._require_base_optimal(
                    run_status, deadline=deadline
                )
        except BaseException:
            self._release_highs(suppress_errors=True)
            raise

    def _configure_highs(self) -> None:
        self._require_ok(
            self._highs.setOptionValue("output_flag", False),
            "set_output_flag",
        )
        self._require_ok(
            self._highs.setOptionValue("solver", "simplex"),
            "set_solver",
        )
        self._require_ok(
            self._highs.setOptionValue(
                "presolve",
                "on" if self._candidate_presolve else "off",
            ),
            "set_presolve",
        )
        self._require_ok(
            self._highs.setOptionValue("threads", self._threads),
            "set_threads",
        )
        self._require_ok(
            self._highs.setOptionValue(
                "small_matrix_value", _CANDIDATE_DUST_ABS
            ),
            "set_small_matrix_value",
        )

    def _load_candidate_model(
        self, hz: SparseHZono, *, deadline: float
    ) -> None:
        if (
            self._candidate_rows > _INT32_MAX
            or self._candidate_columns > _INT32_MAX
        ):
            raise PersistentConflictOracleError(
                "candidate_topology_exceeds_int32"
            )
        empty_i32 = np.array([], dtype=np.int32)
        empty_f64 = np.array([], dtype=np.float64)
        n_variables = self._candidate_columns
        self._require_ok(
            self._highs.addCols(
                n_variables,
                np.zeros(n_variables, dtype=np.float64),
                -np.ones(n_variables, dtype=np.float64),
                np.ones(n_variables, dtype=np.float64),
                0,
                empty_i32,
                empty_i32,
                empty_f64,
            ),
            "add_columns",
        )
        _candidate_deadline(deadline, "add_candidate_columns")

        upper_continuous = _validated_candidate_block(
            hz.Auc,
            rows=hz.n_ub,
            columns=hz.n_cont,
            name="Auc",
            deadline=deadline,
        )
        upper_binary = _validated_candidate_block(
            hz.Aub,
            rows=hz.n_ub,
            columns=hz.n_bin,
            name="Aub",
            deadline=deadline,
        )
        equality_continuous = _validated_candidate_block(
            hz.Ac,
            rows=hz.n_eq,
            columns=hz.n_cont,
            name="Ac",
            deadline=deadline,
        )
        equality_binary = _validated_candidate_block(
            hz.Ab,
            rows=hz.n_eq,
            columns=hz.n_bin,
            name="Ab",
            deadline=deadline,
        )
        self._candidate_nonzeros = int(
            upper_continuous.kept_nonzeros
            + upper_binary.kept_nonzeros
            + equality_continuous.kept_nonzeros
            + equality_binary.kept_nonzeros
        )
        if self._candidate_nonzeros > _INT32_MAX:
            raise PersistentConflictOracleError(
                "candidate_nonzeros_exceed_int32"
            )

        upper_bound = np.asarray(hz.ub).astype(
            np.float64, copy=False
        )
        equality_bound = np.asarray(hz.b).astype(
            np.float64, copy=False
        )
        if (
            upper_bound.ndim != 1
            or upper_bound.size != hz.n_ub
            or equality_bound.ndim != 1
            or equality_bound.size != hz.n_eq
            or np.any(np.isnan(upper_bound))
            or np.any(np.isnan(equality_bound))
        ):
            raise PersistentConflictOracleError(
                "candidate_row_bound_nan_or_shape_invalid"
            )

        binary_nonzeros = int(
            upper_binary.kept_nonzeros
            + equality_binary.kept_nonzeros
        )
        if (
            self._require_split_load
            and binary_nonzeros
            > _MAX_BINARY_CHANGE_COEFFICIENTS
        ):
            raise PersistentConflictOracleError(
                "candidate_binary_nonzeros_exceed_split_load_cap"
            )
        if binary_nonzeros <= _MAX_BINARY_CHANGE_COEFFICIENTS:
            self._candidate_load_mode = (
                "split_continuous_rows_binary_change_coeff_v1"
            )
            self._add_continuous_block(
                upper_continuous,
                lower=np.full(
                    hz.n_ub,
                    -_highspy.kHighsInf,
                    dtype=np.float64,
                ),
                upper=upper_bound,
                deadline=deadline,
            )
            self._add_continuous_block(
                equality_continuous,
                lower=equality_bound,
                upper=equality_bound,
                deadline=deadline,
            )
            self._inject_binary_block(
                upper_binary,
                row_offset=0,
                column_offset=hz.n_cont,
                deadline=deadline,
            )
            self._inject_binary_block(
                equality_binary,
                row_offset=hz.n_ub,
                column_offset=hz.n_cont,
                deadline=deadline,
            )
        else:
            self._candidate_load_mode = (
                "single_merged_csr_binary_cap_fallback_v1"
            )
            starts, indices, data = _merged_candidate_arrays(
                upper_continuous=upper_continuous,
                upper_binary=upper_binary,
                equality_continuous=equality_continuous,
                equality_binary=equality_binary,
                continuous_columns=hz.n_cont,
                deadline=deadline,
            )
            row_lower = np.concatenate(
                [
                    np.full(
                        hz.n_ub,
                        -_highspy.kHighsInf,
                        dtype=np.float64,
                    ),
                    equality_bound,
                ]
            )
            row_upper = np.concatenate(
                [upper_bound, equality_bound]
            )
            self._require_ok(
                self._highs.addRows(
                    self._candidate_rows,
                    row_lower,
                    row_upper,
                    self._candidate_nonzeros,
                    starts.astype(np.int32, copy=False),
                    indices.astype(np.int32, copy=False),
                    data.astype(np.float64, copy=False),
                ),
                "add_merged_rows",
            )
            _candidate_deadline(deadline, "add_merged_rows")

        if (
            int(self._highs.getNumCol())
            != self._candidate_columns
            or int(self._highs.getNumRow())
            != self._candidate_rows
            or int(self._highs.getNumNz())
            != self._candidate_nonzeros
        ):
            raise PersistentConflictOracleError(
                "candidate_topology_postcondition_failed"
            )
        _candidate_deadline(deadline, "candidate_topology_check")

    def _add_continuous_block(
        self,
        block: _CandidateCSRBlock,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        deadline: float,
    ) -> None:
        if block.rows == 0:
            return
        starts, indices, data = _compact_candidate_block(
            block, deadline=deadline
        )
        self._require_ok(
            self._highs.addRows(
                block.rows,
                np.asarray(lower).astype(np.float64, copy=False),
                np.asarray(upper).astype(np.float64, copy=False),
                block.kept_nonzeros,
                starts.astype(np.int32, copy=False),
                indices.astype(np.int32, copy=False),
                data.astype(np.float64, copy=False),
            ),
            f"add_{block.name}_rows",
        )
        _candidate_deadline(deadline, f"add_{block.name}_rows")

    def _inject_binary_block(
        self,
        block: _CandidateCSRBlock,
        *,
        row_offset: int,
        column_offset: int,
        deadline: float,
    ) -> None:
        injected = 0
        for row in range(block.rows):
            start = int(block.indptr[row])
            stop = int(block.indptr[row + 1])
            for position in range(start, stop):
                value = float(block.data[position])
                if abs(value) <= _CANDIDATE_DUST_ABS:
                    continue
                if injected % 256 == 0:
                    _candidate_deadline(
                        deadline, f"inject_{block.name}"
                    )
                column = column_offset + int(
                    block.indices[position]
                )
                if column < 0 or column > _INT32_MAX:
                    raise PersistentConflictOracleError(
                        "candidate_binary_column_exceeds_int32"
                    )
                self._require_ok(
                    self._highs.changeCoeff(
                        row_offset + row, column, value
                    ),
                    f"change_{block.name}_coefficient",
                )
                injected += 1
                _candidate_deadline(
                    deadline, f"inject_{block.name}"
                )
        if injected != block.kept_nonzeros:
            raise PersistentConflictOracleError(
                f"candidate_{block.name}_injection_postcondition_failed"
            )

    def _release_highs(self, *, suppress_errors: bool) -> None:
        highs = getattr(self, "_highs", None)
        if highs is None:
            self._closed = True
            return
        try:
            try:
                self._last_highs_run_time = float(
                    highs.getRunTime()
                )
            except Exception:
                pass
            clear = getattr(highs, "clear", None)
            status = clear() if callable(clear) else None
            if (
                not suppress_errors
                and status is not None
                and status != _highspy.HighsStatus.kOk
            ):
                raise PersistentConflictOracleError(
                    "highs_clear_failed"
                )
        except Exception:
            if not suppress_errors:
                raise
        finally:
            self._highs = None
            self._closed = True

    def close(self) -> None:
        """Release the native HiGHS model deterministically."""

        self._release_highs(suppress_errors=False)

    def __enter__(self) -> "_PersistentHighsPairLP":
        if self._closed or self._highs is None:
            raise PersistentConflictOracleError(
                "highs_model_closed"
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._release_highs(suppress_errors=exc_type is not None)
        return False

    def __del__(self) -> None:  # pragma: no cover - GC timing varies.
        try:
            self._release_highs(suppress_errors=True)
        except Exception:
            pass

    def _require_open_highs(self):
        if self._closed or self._highs is None:
            raise PersistentConflictOracleError(
                "highs_model_closed"
            )
        return self._highs

    @staticmethod
    def _require_ok(status: Any, operation: str) -> None:
        if status != _highspy.HighsStatus.kOk:
            raise PersistentConflictOracleError(
                f"highs_{operation}_failed"
            )

    def _set_time_ceiling(self, deadline: float) -> None:
        highs = self._require_open_highs()
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0.0:
            raise PersistentConflictOracleError(
                "deadline_expired_before_highs"
            )
        # HiGHS applies time_limit to cumulative runtime for a persistent
        # object, not to the next run in isolation.
        ceiling = float(highs.getRunTime()) + remaining
        self._require_ok(
            highs.setOptionValue("time_limit", ceiling),
            "set_time_limit",
        )

    def _require_base_optimal(
        self, run_status: Any, *, deadline: float
    ) -> None:
        """Keep resource exhaustion distinct from solver non-optimality."""

        if time.monotonic() >= deadline:
            raise PersistentConflictOracleError(
                "deadline_expired_during_base_relaxation"
            )
        self._require_ok(run_status, "run_base")
        if (
            self._highs.getModelStatus()
            != _highspy.HighsModelStatus.kOptimal
        ):
            raise PersistentConflictOracleError(
                "base_relaxation_not_optimal"
            )

    def probe(
        self,
        pair: Tuple[PhaseLiteral, PhaseLiteral],
        *,
        deadline: float,
    ) -> Tuple[str, Optional[np.ndarray]]:
        self._require_open_highs()
        columns = np.asarray(
            [
                self._hz.n_cont
                + self._positions[literal.stable_bcol_id]
                for literal in pair
            ],
            dtype=np.int32,
        )
        phases = np.asarray(
            [literal.phase for literal in pair],
            dtype=np.float64,
        )
        if columns[0] == columns[1]:
            raise PersistentConflictOracleError(
                "pair_reuses_literal_column"
            )
        restored = False
        self._require_ok(
            self._highs.changeColsBounds(
                2, columns, phases, phases
            ),
            "fix_pair_bounds",
        )
        self.bound_update_calls += 1
        try:
            self._set_time_ceiling(deadline)
            run_status = self._highs.run()
            self.solve_calls += 1
            if (
                time.monotonic() >= deadline
                or run_status != _highspy.HighsStatus.kOk
            ):
                return "feasible_or_unknown", None
            model_status = self._highs.getModelStatus()
            if (
                model_status
                != _highspy.HighsModelStatus.kInfeasible
            ):
                return "feasible_or_unknown", None
            ray_status, ray_exists = self._highs.getDualRayExist()
            if (
                ray_status != _highspy.HighsStatus.kOk
                or (not ray_exists and not self._candidate_presolve)
            ):
                return "infeasible_without_ray", None
            # HiGHS 1.14 may lazily materialize an original-row dual ray
            # after presolve: ``getDualRayExist`` can therefore be false
            # immediately after an infeasible run while ``getDualRay``
            # successfully computes and maps the ray.  The ray remains only
            # a proposal; callers accept an edge solely after exact Fraction
            # replay against the untouched full parent row frame.
            ray_status, ray_exists, raw_ray = (
                self._highs.getDualRay()
            )
            self.ray_calls += 1
            if (
                time.monotonic() >= deadline
                or ray_status != _highspy.HighsStatus.kOk
                or not ray_exists
            ):
                return "infeasible_without_ray", None
            ray = np.array(raw_ray, dtype=np.float64, copy=True)
            return "infeasible_with_ray", ray
        finally:
            restore_status = self._highs.changeColsBounds(
                2,
                columns,
                -np.ones(2, dtype=np.float64),
                np.ones(2, dtype=np.float64),
            )
            restored = restore_status == _highspy.HighsStatus.kOk
            self.bound_update_calls += 1
            if not restored:
                raise PersistentConflictOracleError(
                    "restore_pair_bounds_failed"
                )

    def propose_objective_duals(
        self,
        maximization_factor_objective: np.ndarray,
        *,
        deadline: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve one objective without changing pair-query accounting.

        This internal method intentionally returns only raw proposals.  The
        public one-shot wrapper below owns deterministic model closure and
        constructs the checksummed receipt only after ``close`` succeeds.
        """

        highs = self._require_open_highs()
        if self._candidate_load_mode != (
            "split_continuous_rows_binary_change_coeff_v1"
        ):
            raise PersistentConflictOracleError(
                "objective_proposal_requires_split_load"
            )
        if not self._candidate_presolve:
            raise PersistentConflictOracleError(
                "objective_proposal_requires_presolve"
            )
        if self.objective_solve_calls != 0:
            raise PersistentConflictOracleError(
                "objective_proposal_not_one_shot"
            )
        objective = np.asarray(
            maximization_factor_objective, dtype=np.float64
        ).reshape(-1)
        if (
            objective.size != self._candidate_columns
            or not np.all(np.isfinite(objective))
        ):
            raise PersistentConflictOracleError(
                "objective_proposal_factor_objective_invalid"
            )
        objective = np.ascontiguousarray(objective)
        cost = np.negative(objective)
        columns = np.arange(
            self._candidate_columns, dtype=np.int32
        )
        _candidate_deadline(deadline, "objective_change_cost")
        self._require_ok(
            highs.changeColsCost(
                self._candidate_columns, columns, cost
            ),
            "change_objective_cost",
        )
        _candidate_deadline(deadline, "objective_before_run")
        self._set_time_ceiling(deadline)
        run_status = highs.run()
        self.objective_solve_calls += 1
        _candidate_deadline(deadline, "objective_solver_run")
        self._require_ok(run_status, "run_objective")
        if (
            highs.getModelStatus()
            != _highspy.HighsModelStatus.kOptimal
        ):
            raise PersistentConflictOracleError(
                "objective_proposal_model_not_optimal"
            )
        solution = highs.getSolution()
        if not bool(getattr(solution, "dual_valid", False)):
            raise PersistentConflictOracleError(
                "objective_proposal_dual_invalid"
            )
        raw_row_dual = np.asarray(
            solution.row_dual, dtype=np.float64
        ).reshape(-1)
        if raw_row_dual.size != self._candidate_rows:
            raise PersistentConflictOracleError(
                "objective_proposal_original_row_shape_invalid"
            )
        if not np.all(np.isfinite(raw_row_dual)):
            raise PersistentConflictOracleError(
                "objective_proposal_row_dual_nonfinite"
            )
        solver_objective = float(
            highs.getInfo().objective_function_value
        )
        if not math.isfinite(solver_objective):
            raise PersistentConflictOracleError(
                "objective_proposal_solver_objective_nonfinite"
            )
        upper_row_dual = np.array(
            raw_row_dual[: self._hz.n_ub],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        equality_row_dual = np.array(
            raw_row_dual[self._hz.n_ub :],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        upper_row_dual.setflags(write=False)
        equality_row_dual.setflags(write=False)
        _candidate_deadline(deadline, "objective_dual_copy")
        return (
            upper_row_dual,
            equality_row_dual,
            solver_objective,
        )

    @property
    def telemetry(self) -> Mapping[str, Any]:
        result = {
            "backend": (
                "highspy_persistent_simplex_presolve_lazy_dual_ray_v2"
                if self._candidate_presolve
                else "highspy_persistent_simplex_dual_ray_v1"
            ),
            "highs_version": "{}.{}.{}".format(
                _highspy.HIGHS_VERSION_MAJOR,
                _highspy.HIGHS_VERSION_MINOR,
                _highspy.HIGHS_VERSION_PATCH,
            ),
            "row_order": "upper_then_equality",
            "presolve": (
                "on" if self._candidate_presolve else "off"
            ),
            "threads": self._threads,
            "model_builds": 1,
            "candidate_rows": int(self._candidate_rows),
            "candidate_columns": int(self._candidate_columns),
            "candidate_nonzeros": int(self._candidate_nonzeros),
            "candidate_load_mode": self._candidate_load_mode,
            "binary_change_coefficient_cap": int(
                _MAX_BINARY_CHANGE_COEFFICIENTS
            ),
            "solve_calls": int(self.solve_calls),
            "bound_update_calls": int(self.bound_update_calls),
            "dual_ray_calls": int(self.ray_calls),
            "highs_cumulative_run_time_seconds": (
                float(self._highs.getRunTime())
                if self._highs is not None
                else float(self._last_highs_run_time)
            ),
        }
        if self._candidate_first:
            result["base_solve_calls"] = int(
                self.base_solve_calls
            )
        return result


def _native_objective_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(
        np.asarray(values, dtype=np.float64).reshape(-1)
    )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _native_objective_negative_sha256(values: np.ndarray) -> str:
    """Hash ``-values`` with only one bounded dense temporary."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, int(array.size), 65536):
        chunk = np.negative(array[start : start + 65536])
        digest.update(
            np.ascontiguousarray(chunk).tobytes(order="C")
        )
    return digest.hexdigest()


def _native_objective_all_finite(
    values: np.ndarray,
    *,
    deadline: float,
    name: str,
) -> bool:
    array = np.asarray(values).reshape(-1)
    for start in range(0, int(array.size), 65536):
        _candidate_deadline(deadline, f"validate_{name}")
        if not np.all(np.isfinite(array[start : start + 65536])):
            return False
    _candidate_deadline(deadline, f"validate_{name}")
    return True


def _require_native_objective_parent(
    hz: SparseHZono, *, deadline: float
) -> None:
    """Enforce the split checker's zero-copy canonical HZ contract."""

    if not isinstance(hz, SparseHZono):
        raise PersistentConflictOracleError(
            "objective_proposal_parent_not_sparse_hz"
        )
    if hz.Auc is None or hz.Aub is None or hz.ub is None:
        raise PersistentConflictOracleError(
            "objective_proposal_upper_frame_absent"
        )
    blocks = (
        ("Gc", hz.Gc, hz.n_out, hz.n_cont),
        ("Gb", hz.Gb, hz.n_out, hz.n_bin),
        ("Auc", hz.Auc, hz.n_ub, hz.n_cont),
        ("Aub", hz.Aub, hz.n_ub, hz.n_bin),
        ("Ac", hz.Ac, hz.n_eq, hz.n_cont),
        ("Ab", hz.Ab, hz.n_eq, hz.n_bin),
    )
    for name, matrix, rows, columns in blocks:
        _candidate_deadline(deadline, f"validate_{name}")
        if (
            not sp.isspmatrix_csr(matrix)
            or matrix.dtype != np.dtype(np.float64)
            or matrix.shape != (rows, columns)
            or not matrix.has_canonical_format
            or np.asarray(matrix.indptr).ndim != 1
            or np.asarray(matrix.indices).ndim != 1
            or np.asarray(matrix.data).ndim != 1
            or int(matrix.indptr.size) != rows + 1
            or int(matrix.indices.size) != int(matrix.data.size)
        ):
            raise PersistentConflictOracleError(
                f"objective_proposal_{name}_not_canonical_binary64_csr"
            )
        if not _native_objective_all_finite(
            matrix.data,
            deadline=deadline,
            name=f"{name}_data",
        ):
            raise PersistentConflictOracleError(
                f"objective_proposal_{name}_nonfinite"
            )
    dense_frames = (
        ("center", hz.c, hz.n_out),
        ("upper_bound", hz.ub, hz.n_ub),
        ("equality_bound", hz.b, hz.n_eq),
    )
    for name, values, size in dense_frames:
        array = np.asarray(values)
        if (
            array.dtype != np.dtype(np.float64)
            or array.ndim != 1
            or int(array.size) != size
            or not _native_objective_all_finite(
                array, deadline=deadline, name=name
            )
        ):
            raise PersistentConflictOracleError(
                f"objective_proposal_{name}_invalid"
            )


def propose_native_split_row_objective_duals(
    hz: SparseHZono,
    maximization_factor_objective: np.ndarray,
    *,
    deadline: float,
) -> NativeSplitRowObjectiveDualProposal:
    """Propose original-frame row duals for one maximization objective.

    HiGHS minimizes ``-maximization_factor_objective`` over the relaxed HZ
    factor frame.  The solver result has no proof or verdict authority.  This
    function returns only after the model is optimal, its original-row dual
    is valid and finite, the absolute deadline still holds, and native HiGHS
    storage has been explicitly closed.  Models which exceed the bounded
    binary ``changeCoeff`` route fail before the merged-loader fallback.
    """

    if isinstance(deadline, (bool, np.bool_)):
        raise PersistentConflictOracleError(
            "objective_proposal_deadline_invalid"
        )
    try:
        deadline = float(deadline)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PersistentConflictOracleError(
            "objective_proposal_deadline_invalid"
        ) from exc
    if not math.isfinite(deadline):
        raise PersistentConflictOracleError(
            "objective_proposal_deadline_invalid"
        )
    _candidate_deadline(deadline, "objective_proposal_entry")
    _require_native_objective_parent(hz, deadline=deadline)
    try:
        objective = np.array(
            maximization_factor_objective,
            dtype=np.float64,
            order="C",
            copy=True,
        ).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PersistentConflictOracleError(
            "objective_proposal_factor_objective_invalid"
        ) from exc
    if (
        objective.size != hz.n_cont + hz.n_bin
        or not _native_objective_all_finite(
            objective,
            deadline=deadline,
            name="factor_objective",
        )
    ):
        raise PersistentConflictOracleError(
            "objective_proposal_factor_objective_invalid"
        )

    oracle = None
    try:
        oracle = _PersistentHighsPairLP(
            hz,
            deadline=deadline,
            solve_base_relaxation=False,
            candidate_presolve=True,
            require_split_load=True,
        )
        (
            upper_row_dual,
            equality_row_dual,
            solver_objective,
        ) = oracle.propose_objective_duals(
            objective, deadline=deadline
        )
        telemetry = dict(oracle.telemetry)
        pair_solve_calls = int(oracle.solve_calls)
        objective_solve_calls = int(
            oracle.objective_solve_calls
        )
        if pair_solve_calls != 0 or objective_solve_calls != 1:
            raise PersistentConflictOracleError(
                "objective_proposal_solve_accounting_invalid"
            )
    except PersistentConflictOracleError:
        raise
    except Exception as exc:
        raise PersistentConflictOracleError(
            f"objective_proposal_backend_error:{type(exc).__name__}"
        ) from exc
    finally:
        if oracle is not None:
            oracle.close()

    if not oracle._closed or oracle._highs is not None:
        raise PersistentConflictOracleError(
            "objective_proposal_native_model_not_closed"
        )
    _candidate_deadline(deadline, "objective_proposal_return")
    receipt = {
        "schema": _NATIVE_OBJECTIVE_PROPOSAL_SCHEMA,
        "status": "optimal_dual_candidate",
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "backend": _NATIVE_OBJECTIVE_PROPOSAL_BACKEND,
        "highs_version": str(telemetry["highs_version"]),
        "presolve": "on",
        "row_order": "upper_then_equality",
        "candidate_load_mode": str(
            telemetry["candidate_load_mode"]
        ),
        "binary_change_coefficient_cap": int(
            telemetry["binary_change_coefficient_cap"]
        ),
        "candidate_rows": int(telemetry["candidate_rows"]),
        "candidate_columns": int(
            telemetry["candidate_columns"]
        ),
        "candidate_nonzeros": int(
            telemetry["candidate_nonzeros"]
        ),
        "n_continuous": int(hz.n_cont),
        "n_binary": int(hz.n_bin),
        "n_upper": int(hz.n_ub),
        "n_equality": int(hz.n_eq),
        "objective_convention": (
            "highs_minimize_cost_equals_negative_max_factor_objective"
        ),
        "maximization_factor_objective_size": int(
            objective.size
        ),
        "maximization_factor_objective_sha256": (
            _native_objective_array_sha256(objective)
        ),
        "solver_cost_sha256": (
            _native_objective_negative_sha256(objective)
        ),
        "upper_row_dual_size": int(upper_row_dual.size),
        "equality_row_dual_size": int(
            equality_row_dual.size
        ),
        "upper_row_dual_sha256": (
            _native_objective_array_sha256(upper_row_dual)
        ),
        "equality_row_dual_sha256": (
            _native_objective_array_sha256(equality_row_dual)
        ),
        "solver_minimization_objective_hex": (
            float(solver_objective).hex()
        ),
        "pair_solve_calls": pair_solve_calls,
        "objective_solve_calls": objective_solve_calls,
        "native_model_closed_before_return": True,
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "used_merged_sparse_frame": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return NativeSplitRowObjectiveDualProposal(
        upper_row_dual=upper_row_dual,
        equality_row_dual=equality_row_dual,
        solver_minimization_objective=float(solver_objective),
        receipt=receipt,
    )


def _validate_caps(
    *,
    max_literals: int,
    max_pairs: int,
    max_source_terms: int,
    max_multiplier_bits: int,
    max_exact_bits: int,
    max_exact_nonzeros: int,
) -> Mapping[str, int]:
    caps = {
        "max_literals": _strict_int(
            max_literals, name="max_literals"
        ),
        "max_pairs": _strict_int(max_pairs, name="max_pairs"),
        "max_source_terms": _strict_int(
            max_source_terms, name="max_source_terms"
        ),
        "max_multiplier_bits": _strict_int(
            max_multiplier_bits, name="max_multiplier_bits"
        ),
        "max_exact_bits": _strict_int(
            max_exact_bits, name="max_exact_bits"
        ),
        "max_exact_nonzeros": _strict_int(
            max_exact_nonzeros, name="max_exact_nonzeros"
        ),
    }
    hard_limits = {
        "max_literals": 64,
        "max_pairs": 2016,
        "max_source_terms": 256,
        "max_multiplier_bits": 2048,
        "max_exact_bits": 16384,
        "max_exact_nonzeros": 1000000,
    }
    if any(
        value < 1 or value > hard_limits[name]
        for name, value in caps.items()
    ):
        raise PersistentConflictOracleError(
            "candidate_cap_out_of_range"
        )
    return caps


def _invocation_payload(
    invocation: PersistentPCPCCInvocationSpec,
    *,
    include_digest: bool,
) -> Mapping[str, Any]:
    payload = {
        "schema": "act.pc_pcc.persistent_invocation.v1",
        "nonce": invocation.nonce,
        "gate_id": invocation.gate_id,
        "parent_semantic_digest": (
            invocation.parent_semantic_digest
        ),
        "property_digest": invocation.property_digest,
        "deadline_hex": float(invocation.deadline).hex(),
        "caps": [
            [name, int(value)]
            for name, value in invocation.caps
        ],
    }
    if include_digest:
        payload["invocation_sha256"] = (
            invocation.invocation_sha256
        )
    return payload


def make_persistent_pc_pcc_invocation_spec(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    *,
    deadline: float,
    gate_id: str,
    max_literals: int = 16,
    max_pairs: int = 120,
    max_source_terms: int = 128,
    max_multiplier_bits: int = 256,
    max_exact_bits: int = 4096,
    max_exact_nonzeros: int = 200000,
) -> PersistentPCPCCInvocationSpec:
    """Create the caller-owned config object shared by run and consume."""

    if not isinstance(hz, SparseHZono):
        raise PersistentConflictOracleError(
            "invocation_parent_not_sparse_hz"
        )
    deadline = float(deadline)
    if (
        not math.isfinite(deadline)
        or deadline <= time.monotonic()
    ):
        raise PersistentConflictOracleError(
            "invocation_deadline_invalid"
        )
    if (
        not isinstance(gate_id, str)
        or not 1 <= len(gate_id) <= 128
        or any(
            ord(character) < 32 or ord(character) > 126
            for character in gate_id
        )
    ):
        raise PersistentConflictOracleError(
            "invocation_gate_id_invalid"
        )
    caps = _validate_caps(
        max_literals=max_literals,
        max_pairs=max_pairs,
        max_source_terms=max_source_terms,
        max_multiplier_bits=max_multiplier_bits,
        max_exact_bits=max_exact_bits,
        max_exact_nonzeros=max_exact_nonzeros,
    )
    placeholder = PersistentPCPCCInvocationSpec(
        nonce=secrets.token_hex(32),
        gate_id=gate_id,
        parent_semantic_digest=sparse_hz_semantic_digest(hz),
        property_digest=ordered_property_digest(rivals),
        deadline=deadline,
        caps=tuple(sorted(caps.items())),
        invocation_sha256="",
    )
    return PersistentPCPCCInvocationSpec(
        **{
            **placeholder.__dict__,
            "invocation_sha256": _canonical_sha256(
                _invocation_payload(
                    placeholder, include_digest=False
                )
            ),
        }
    )


def _validated_invocation_caps(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    invocation: PersistentPCPCCInvocationSpec,
) -> Mapping[str, int]:
    if (
        not isinstance(
            invocation, PersistentPCPCCInvocationSpec
        )
        or not isinstance(invocation.nonce, str)
        or len(invocation.nonce) != 64
        or any(
            character not in "0123456789abcdef"
            for character in invocation.nonce
        )
        or not isinstance(invocation.gate_id, str)
        or not 1 <= len(invocation.gate_id) <= 128
        or any(
            ord(character) < 32 or ord(character) > 126
            for character in invocation.gate_id
        )
        or type(invocation.deadline) is not float
        or not math.isfinite(invocation.deadline)
        or invocation.parent_semantic_digest
        != sparse_hz_semantic_digest(hz)
        or invocation.property_digest
        != ordered_property_digest(rivals)
        or not isinstance(invocation.caps, tuple)
        or not _valid_sha256(invocation.invocation_sha256)
        or _canonical_sha256(
            _invocation_payload(
                invocation, include_digest=False
            )
        )
        != invocation.invocation_sha256
    ):
        raise PersistentConflictOracleError(
            "invocation_semantics_invalid"
        )
    caps = dict(invocation.caps)
    if (
        len(caps) != len(invocation.caps)
        or tuple(sorted(caps.items())) != invocation.caps
    ):
        raise PersistentConflictOracleError(
            "invocation_caps_noncanonical"
        )
    return _validate_caps(**caps)


def _claim_persistent_pc_pcc_invocation(
    invocation: PersistentPCPCCInvocationSpec,
) -> None:
    """Atomically make one pre-registered invocation single-run."""

    now = time.monotonic()
    if now >= invocation.deadline:
        raise PersistentConflictOracleError(
            "invocation_deadline_expired_before_claim"
        )
    with _LIVE_CAPABILITY_LOCK:
        stale = [
            nonce
            for nonce, deadline in (
                _CLAIMED_PERSISTENT_PC_PCC_INVOCATIONS.items()
            )
            if now >= deadline
        ]
        for nonce in stale:
            _CLAIMED_PERSISTENT_PC_PCC_INVOCATIONS.pop(
                nonce, None
            )
        if (
            invocation.nonce
            in _CLAIMED_PERSISTENT_PC_PCC_INVOCATIONS
        ):
            raise PersistentConflictOracleError(
                "invocation_already_claimed"
            )
        if (
            len(_CLAIMED_PERSISTENT_PC_PCC_INVOCATIONS)
            >= _MAX_CLAIMED_INVOCATIONS
        ):
            raise PersistentConflictOracleError(
                "invocation_registry_capacity_exceeded"
            )
        _CLAIMED_PERSISTENT_PC_PCC_INVOCATIONS[
            invocation.nonce
        ] = invocation.deadline


def run_persistent_conflict_oracle_candidate(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    *,
    deadline: float,
    max_literals: int = 16,
    max_pairs: int = 120,
    max_source_terms: int = 128,
    max_multiplier_bits: int = 256,
    max_exact_bits: int = 4096,
    max_exact_nonzeros: int = 200000,
) -> PersistentConflictOracleResult:
    """Discover an exact conflict graph with one persistent HiGHS model."""

    started = time.perf_counter()
    if not isinstance(hz, SparseHZono):
        raise PersistentConflictOracleError("parent_not_sparse_hz")
    deadline = float(deadline)
    if not math.isfinite(deadline):
        raise PersistentConflictOracleError("deadline_nonfinite")
    caps = _validate_caps(
        max_literals=max_literals,
        max_pairs=max_pairs,
        max_source_terms=max_source_terms,
        max_multiplier_bits=max_multiplier_bits,
        max_exact_bits=max_exact_bits,
        max_exact_nonzeros=max_exact_nonzeros,
    )
    parent_digest = sparse_hz_semantic_digest(hz)
    property_digest = ordered_property_digest(rivals)
    literals = _derive_property_literals(
        hz,
        rivals,
        parent_digest=parent_digest,
        property_digest=property_digest,
    )
    pair_count = len(literals) * (len(literals) - 1) // 2
    if (
        len(literals) < 2
        or len(literals) > caps["max_literals"]
        or pair_count > caps["max_pairs"]
    ):
        raise PersistentConflictOracleError(
            "literal_or_pair_cap_rejected"
        )
    source_frame_digest = _ordered_source_frame_digest(
        hz,
        parent_digest=parent_digest,
        deadline=deadline,
    )
    oracle = _PersistentHighsPairLP(hz, deadline=deadline)
    records = []
    certificates = []
    reason = "processed_all_pairs"

    for left_index, left in enumerate(literals):
        for right in literals[left_index + 1 :]:
            if time.monotonic() >= deadline:
                reason = "deadline_expired_during_pairs"
                break
            pair = _ordered_pair(left, right)
            candidate_status, raw_ray = oracle.probe(
                pair, deadline=deadline
            )
            if candidate_status == "infeasible_with_ray":
                ray_nonzero_rows = int(
                    np.count_nonzero(raw_ray)
                )
                certificate = (
                    exact_certificate_from_highs_dual_ray_candidate(
                        hz,
                        pair,
                        raw_ray,
                        parent_digest=parent_digest,
                        property_digest=property_digest,
                        source_frame_digest=source_frame_digest,
                        deadline=deadline,
                        max_source_terms=caps[
                            "max_source_terms"
                        ],
                        max_multiplier_bits=caps[
                            "max_multiplier_bits"
                        ],
                        max_exact_bits=caps["max_exact_bits"],
                        max_exact_nonzeros=caps[
                            "max_exact_nonzeros"
                        ],
                    )
                )
                if certificate is None:
                    status = "exact_replay_rejected"
                    certificate_digest = None
                    rationalization = None
                else:
                    status = "certified_conflict"
                    certificate_digest = (
                        certificate.certificate_sha256
                    )
                    rationalization = (
                        certificate.rationalization
                    )
                    certificates.append(certificate)
            else:
                ray_nonzero_rows = 0
                certificate_digest = None
                rationalization = None
                status = candidate_status
            records.append(
                PersistentPairRecord(
                    literals=pair,
                    status=status,
                    ray_nonzero_rows=ray_nonzero_rows,
                    certificate_sha256=certificate_digest,
                    rationalization=rationalization,
                )
            )
            if time.monotonic() >= deadline:
                raise PersistentConflictOracleError(
                    "deadline_expired_after_pair_replay"
                )
        if reason != "processed_all_pairs":
            break

    expected_pairs = pair_count
    if len(records) < expected_pairs:
        result_status = "stopped_without_complete_graph"
    elif len(certificates) == expected_pairs:
        result_status = "complete_conflict_graph_candidate"
    else:
        result_status = "incomplete_conflict_graph_candidate"
    telemetry = {
        **oracle.telemetry,
        "caps": dict(caps),
        "expected_pairs": int(expected_pairs),
        "processed_pairs": int(len(records)),
        "certified_edges": int(len(certificates)),
        "phase_children_minted": 0,
        "wall_seconds": float(time.perf_counter() - started),
        "proof_role": "candidate_generation_only",
    }
    oracle.close()
    result = PersistentConflictOracleResult(
        status=result_status,
        reason=reason,
        literals=literals,
        records=tuple(records),
        certificates=tuple(certificates),
        parent_semantic_digest=parent_digest,
        property_digest=property_digest,
        ordered_source_frame_sha256=source_frame_digest,
        telemetry=telemetry,
    )
    if sparse_hz_semantic_digest(hz) != parent_digest:
        raise PersistentConflictOracleError(
            "parent_mutated_during_candidate"
        )
    if not _verify_generated_oracle_shell(
        hz, rivals, result
    ):
        raise PersistentConflictOracleError(
            "persistent_result_self_audit_failed"
        )
    return result


def _verify_generated_oracle_shell(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PersistentConflictOracleResult,
) -> bool:
    """Check closure after each emitted certificate passed exact replay."""

    try:
        if (
            not isinstance(hz, SparseHZono)
            or not isinstance(
                result, PersistentConflictOracleResult
            )
            or result.proof_authority is not False
            or result.parent_semantic_digest
            != sparse_hz_semantic_digest(hz)
            or result.property_digest
            != ordered_property_digest(rivals)
            or not _valid_sha256(
                result.ordered_source_frame_sha256
            )
            or not isinstance(result.records, tuple)
            or not isinstance(result.certificates, tuple)
            or not isinstance(result.telemetry, Mapping)
        ):
            return False
        raw_caps = result.telemetry.get("caps")
        if not isinstance(raw_caps, dict):
            return False
        caps = _validate_caps(**raw_caps)
        if (
            dict(caps) != raw_caps
            or len(result.literals) > caps["max_literals"]
        ):
            return False
        expected_literals = _derive_property_literals(
            hz,
            rivals,
            parent_digest=result.parent_semantic_digest,
            property_digest=result.property_digest,
        )
        if result.literals != expected_literals:
            return False
        expected_pairs = tuple(
            _ordered_pair(left, right)
            for index, left in enumerate(expected_literals)
            for right in expected_literals[index + 1 :]
        )
        if len(result.records) > len(expected_pairs):
            return False
        if len(expected_pairs) > caps["max_pairs"]:
            return False
        certified_records = []
        for expected_pair, record in zip(
            expected_pairs, result.records
        ):
            if (
                not isinstance(record, PersistentPairRecord)
                or record.literals != expected_pair
                or record.status not in _RECORD_STATUSES
                or type(record.ray_nonzero_rows) is not int
                or record.ray_nonzero_rows < 0
            ):
                return False
            if record.status == "certified_conflict":
                if (
                    not _valid_sha256(
                        record.certificate_sha256
                    )
                    or record.rationalization is None
                ):
                    return False
                certified_records.append(record)
            elif (
                record.certificate_sha256 is not None
                or record.rationalization is not None
            ):
                return False
        if len(certified_records) != len(result.certificates):
            return False
        for record, certificate in zip(
            certified_records, result.certificates
        ):
            if (
                not isinstance(
                    certificate,
                    ExactDualRayConflictCertificate,
                )
                or certificate.literals != record.literals
                or certificate.parent_semantic_digest
                != result.parent_semantic_digest
                or certificate.property_digest
                != result.property_digest
                or certificate.ordered_source_frame_sha256
                != result.ordered_source_frame_sha256
                or certificate.certificate_sha256
                != record.certificate_sha256
                or certificate.rationalization
                != record.rationalization
                or _certificate_digest(certificate)
                != certificate.certificate_sha256
            ):
                return False
        all_processed = len(result.records) == len(expected_pairs)
        complete = (
            bool(expected_pairs)
            and all_processed
            and len(result.certificates) == len(expected_pairs)
        )
        if result.status == "complete_conflict_graph_candidate":
            valid_status = (
                complete
                and result.reason == "processed_all_pairs"
            )
        elif result.status == "incomplete_conflict_graph_candidate":
            valid_status = (
                all_processed
                and not complete
                and result.reason == "processed_all_pairs"
            )
        elif result.status == "stopped_without_complete_graph":
            valid_status = (
                not all_processed
                and result.reason
                == "deadline_expired_during_pairs"
            )
        else:
            return False
        return (
            valid_status
            and result.telemetry.get("expected_pairs")
            == len(expected_pairs)
            and result.telemetry.get("processed_pairs")
            == len(result.records)
            and result.telemetry.get("certified_edges")
            == len(result.certificates)
            and result.telemetry.get("phase_children_minted") == 0
            and result.telemetry.get("model_builds") == 1
            and result.telemetry.get("solve_calls")
            == len(result.records)
        )
    except (
        PersistentConflictOracleError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def verify_persistent_conflict_oracle_result(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PersistentConflictOracleResult,
    *,
    deadline: Optional[float] = None,
) -> bool:
    """Validate exact edge closure; runtime telemetry has no proof role."""

    try:
        deadline_value = (
            time.monotonic() + 60.0
            if deadline is None
            else float(deadline)
        )
        if (
            not math.isfinite(deadline_value)
            or not isinstance(hz, SparseHZono)
            or not isinstance(
                result, PersistentConflictOracleResult
            )
            or result.proof_authority is not False
            or result.status
            not in {
                "complete_conflict_graph_candidate",
                "incomplete_conflict_graph_candidate",
                "stopped_without_complete_graph",
            }
            or result.parent_semantic_digest
            != sparse_hz_semantic_digest(hz)
            or result.property_digest
            != ordered_property_digest(rivals)
            or not _valid_sha256(
                result.ordered_source_frame_sha256
            )
            or not isinstance(result.records, tuple)
            or not isinstance(result.certificates, tuple)
            or not isinstance(result.telemetry, Mapping)
        ):
            return False
        raw_caps = result.telemetry.get("caps")
        if not isinstance(raw_caps, dict):
            return False
        caps = _validate_caps(**raw_caps)
        if (
            dict(caps) != raw_caps
            or len(result.literals) > caps["max_literals"]
        ):
            return False
        expected_literals = _derive_property_literals(
            hz,
            rivals,
            parent_digest=result.parent_semantic_digest,
            property_digest=result.property_digest,
        )
        if result.literals != expected_literals:
            return False
        live_frame = _ordered_source_frame_digest(
            hz,
            parent_digest=result.parent_semantic_digest,
            deadline=deadline_value,
        )
        if live_frame != result.ordered_source_frame_sha256:
            return False
        expected_pairs = tuple(
            _ordered_pair(left, right)
            for index, left in enumerate(expected_literals)
            for right in expected_literals[index + 1 :]
        )
        if len(result.records) > len(expected_pairs):
            return False
        if len(expected_pairs) > caps["max_pairs"]:
            return False
        certified = []
        for expected_pair, record in zip(
            expected_pairs, result.records
        ):
            if (
                not isinstance(record, PersistentPairRecord)
                or record.literals != expected_pair
                or record.status not in _RECORD_STATUSES
                or type(record.ray_nonzero_rows) is not int
                or record.ray_nonzero_rows < 0
            ):
                return False
            if record.status == "certified_conflict":
                if (
                    not _valid_sha256(
                        record.certificate_sha256
                    )
                    or record.rationalization is None
                ):
                    return False
                certified.append(record)
            elif (
                record.certificate_sha256 is not None
                or record.rationalization is not None
            ):
                return False
        if len(certified) != len(result.certificates):
            return False
        for record, certificate in zip(
            certified, result.certificates
        ):
            if (
                not isinstance(
                    certificate,
                    ExactDualRayConflictCertificate,
                )
                or certificate.literals != record.literals
                or certificate.certificate_sha256
                != record.certificate_sha256
                or certificate.rationalization
                != record.rationalization
                or not _verify_exact_certificate_with_source_frame(
                    hz,
                    certificate,
                    property_digest=result.property_digest,
                    parent_digest=result.parent_semantic_digest,
                    source_frame_digest=live_frame,
                    deadline=deadline_value,
                    max_source_terms=caps[
                        "max_source_terms"
                    ],
                    max_multiplier_bits=caps[
                        "max_multiplier_bits"
                    ],
                    max_exact_bits=caps["max_exact_bits"],
                    max_exact_nonzeros=caps[
                        "max_exact_nonzeros"
                    ],
                )
            ):
                return False
        all_processed = len(result.records) == len(expected_pairs)
        complete = (
            bool(expected_pairs)
            and all_processed
            and len(result.certificates) == len(expected_pairs)
        )
        if result.status == "complete_conflict_graph_candidate":
            valid_status = (
                complete
                and result.reason == "processed_all_pairs"
            )
        elif result.status == "incomplete_conflict_graph_candidate":
            valid_status = (
                all_processed
                and not complete
                and result.reason == "processed_all_pairs"
            )
        else:
            valid_status = (
                not all_processed
                and result.reason
                == "deadline_expired_during_pairs"
            )
        return (
            valid_status
            and sparse_hz_semantic_digest(hz)
            == result.parent_semantic_digest
        )
    except (
        PersistentConflictOracleError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _optional_float_hex(value: Optional[float]) -> Optional[str]:
    return None if value is None else float(value).hex()


def _persistent_pc_pcc_live_snapshot(
    *,
    result: PersistentPCPCCResult,
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    deadline: float,
    issued_at: float,
) -> Mapping[str, Any]:
    oracle = result.oracle_result
    return {
        "schema": "act.pc_pcc.persistent_live_candidate.v1",
        "process_id": os.getpid(),
        "result_object_id": id(result),
        "parent_object_id": id(parent),
        "rivals_object_id": id(rivals),
        "cut_object_id": (
            None if result.hz is None else id(result.hz)
        ),
        "invocation_object_id": id(result.invocation),
        "invocation": _invocation_payload(
            result.invocation, include_digest=True
        ),
        "oracle_object_id": id(oracle),
        "oracle_record_tuple_object_id": id(oracle.records),
        "oracle_certificate_tuple_object_id": id(
            oracle.certificates
        ),
        "pre_tuple_object_id": id(result.pre_property_uppers),
        "post_tuple_object_id": id(result.post_property_uppers),
        "parent_semantic_digest": sparse_hz_semantic_digest(
            parent
        ),
        "property_digest": ordered_property_digest(rivals),
        "cut_semantic_digest": (
            None
            if result.hz is None
            else sparse_hz_semantic_digest(result.hz)
        ),
        "status": result.status,
        "proof_authority": result.proof_authority,
        "wall_seconds_hex": float(result.wall_seconds).hex(),
        "deadline_hex": float(deadline).hex(),
        "issued_at_hex": float(issued_at).hex(),
        "pre_property_uppers": [
            _optional_float_hex(value)
            for value in result.pre_property_uppers
        ],
        "post_property_uppers": [
            _optional_float_hex(value)
            for value in result.post_property_uppers
        ],
        "oracle": {
            "status": oracle.status,
            "reason": oracle.reason,
            "proof_authority": oracle.proof_authority,
            "parent_semantic_digest": (
                oracle.parent_semantic_digest
            ),
            "property_digest": oracle.property_digest,
            "ordered_source_frame_sha256": (
                oracle.ordered_source_frame_sha256
            ),
            "literal_bindings": [
                _literal_payload(literal)
                for literal in oracle.literals
            ],
            "records": [
                {
                    "literals": [
                        _literal_payload(literal)
                        for literal in record.literals
                    ],
                    "status": record.status,
                    "ray_nonzero_rows": record.ray_nonzero_rows,
                    "certificate_sha256": (
                        record.certificate_sha256
                    ),
                    "rationalization": record.rationalization,
                }
                for record in oracle.records
            ],
            "certificate_sha256": [
                certificate.certificate_sha256
                for certificate in oracle.certificates
            ],
            "telemetry_sha256": _canonical_sha256(
                oracle.telemetry
            ),
        },
    }


def _issue_live_persistent_pc_pcc_result(
    *,
    result: PersistentPCPCCResult,
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    invocation: PersistentPCPCCInvocationSpec,
) -> PersistentPCPCCResult:
    deadline = invocation.deadline
    capability = _LivePersistentPCPCCCapability(
        _CAPABILITY_SENTINEL
    )
    live_result = PersistentPCPCCResult(
        status=result.status,
        hz=result.hz,
        invocation=invocation,
        oracle_result=result.oracle_result,
        pre_property_uppers=result.pre_property_uppers,
        post_property_uppers=result.post_property_uppers,
        wall_seconds=result.wall_seconds,
        proof_authority=False,
        _live_capability=capability,
    )
    if not _verify_persistent_pc_pcc_shell(
        parent, rivals, live_result
    ):
        raise PersistentConflictOracleError(
            "persistent_live_issue_shell_audit_failed"
        )
    issued_at = time.monotonic()
    if issued_at >= deadline:
        raise PersistentConflictOracleError(
            "deadline_expired_before_persistent_live_issue"
        )
    snapshot = _persistent_pc_pcc_live_snapshot(
        result=live_result,
        parent=parent,
        rivals=rivals,
        deadline=deadline,
        issued_at=issued_at,
    )
    snapshot_mac = hmac.new(
        _PROCESS_KEY,
        _canonical_bytes(snapshot),
        digestmod=hashlib.sha256,
    ).hexdigest()
    capability_key = id(capability)

    def cleanup(_reference, *, key=capability_key) -> None:
        with _LIVE_CAPABILITY_LOCK:
            _LIVE_PERSISTENT_PC_PCC_RESULTS.pop(key, None)

    record = _LivePersistentPCPCCRecord(
        capability=capability,
        result_ref=weakref.ref(live_result, cleanup),
        parent=parent,
        rivals=rivals,
        invocation=invocation,
        hz=live_result.hz,
        oracle_result=live_result.oracle_result,
        pre_property_uppers=live_result.pre_property_uppers,
        post_property_uppers=live_result.post_property_uppers,
        deadline=float(deadline),
        issued_at=issued_at,
        process_id=os.getpid(),
        snapshot_hmac_sha256=snapshot_mac,
    )
    with _LIVE_CAPABILITY_LOCK:
        _sweep_live_persistent_pc_pcc_results(issued_at)
        if (
            len(_LIVE_PERSISTENT_PC_PCC_RESULTS)
            >= _MAX_LIVE_RESULTS
        ):
            raise PersistentConflictOracleError(
                "persistent_live_registry_capacity_exceeded"
            )
        _LIVE_PERSISTENT_PC_PCC_RESULTS[capability_key] = record
    return live_result


def _sweep_live_persistent_pc_pcc_results(now: float) -> None:
    """Drop dead/expired records while the registry lock is held."""

    stale = [
        key
        for key, record in _LIVE_PERSISTENT_PC_PCC_RESULTS.items()
        if (
            record.result_ref() is None
            or now >= record.deadline
            or now - record.issued_at
            >= _MAX_LIVE_RESULT_AGE_SECONDS
        )
    ]
    for key in stale:
        _LIVE_PERSISTENT_PC_PCC_RESULTS.pop(key, None)


def _consume_live_persistent_pc_pcc_result(
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PersistentPCPCCResult,
    invocation: PersistentPCPCCInvocationSpec,
) -> bool:
    if not isinstance(result, PersistentPCPCCResult):
        return False
    capability = result._live_capability
    now = time.monotonic()
    with _LIVE_CAPABILITY_LOCK:
        _sweep_live_persistent_pc_pcc_results(now)
        record = _LIVE_PERSISTENT_PC_PCC_RESULTS.pop(
            id(capability), None
        )
    if (
        record is None
        or not isinstance(
            capability, _LivePersistentPCPCCCapability
        )
        or record.capability is not capability
        or record.result_ref() is not result
        or record.parent is not parent
        or record.rivals is not rivals
        or record.invocation is not invocation
        or result.invocation is not invocation
        or record.hz is not result.hz
        or record.oracle_result is not result.oracle_result
        or record.pre_property_uppers
        is not result.pre_property_uppers
        or record.post_property_uppers
        is not result.post_property_uppers
        or record.process_id != os.getpid()
        or now >= record.deadline
        or now - record.issued_at
        >= _MAX_LIVE_RESULT_AGE_SECONDS
    ):
        return False
    try:
        caps = _validated_invocation_caps(
            parent, rivals, invocation
        )
        if (
            dict(caps)
            != result.oracle_result.telemetry.get("caps")
        ):
            return False
        snapshot = _persistent_pc_pcc_live_snapshot(
            result=result,
            parent=parent,
            rivals=rivals,
            deadline=record.deadline,
            issued_at=record.issued_at,
        )
        live_mac = hmac.new(
            _PROCESS_KEY,
            _canonical_bytes(snapshot),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return (
            hmac.compare_digest(
                live_mac, record.snapshot_hmac_sha256
            )
            and _verify_persistent_pc_pcc_shell(
                parent, rivals, result
            )
        )
    except (
        PersistentConflictOracleError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def revoke_persistent_pc_pcc_result(
    result: PersistentPCPCCResult,
) -> bool:
    """Explicitly revoke one unconsumed live result."""

    if not isinstance(result, PersistentPCPCCResult):
        return False
    capability = result._live_capability
    now = time.monotonic()
    with _LIVE_CAPABILITY_LOCK:
        _sweep_live_persistent_pc_pcc_results(now)
        record = _LIVE_PERSISTENT_PC_PCC_RESULTS.pop(
            id(capability), None
        )
    return bool(
        record is not None
        and isinstance(
            capability, _LivePersistentPCPCCCapability
        )
        and record.capability is capability
        and record.result_ref() is result
    )


def run_persistent_pc_pcc_candidate(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    *,
    invocation: PersistentPCPCCInvocationSpec,
) -> PersistentPCPCCResult:
    """Run the same tightness toy as PC-PCC with persistent pair solves."""

    started = time.perf_counter()
    caps = _validated_invocation_caps(
        hz, rivals, invocation
    )
    deadline = invocation.deadline
    if time.monotonic() >= deadline:
        raise PersistentConflictOracleError(
            "invocation_deadline_expired_before_run"
        )
    _claim_persistent_pc_pcc_invocation(invocation)
    pre_uppers = []
    for rival in rivals:
        pre_uppers.append(
            _highs_property_upper(
                hz, rival, deadline=deadline
            )
        )
        if time.monotonic() >= deadline:
            raise PersistentConflictOracleError(
                "deadline_expired_in_pre_telemetry"
            )
    oracle_result = run_persistent_conflict_oracle_candidate(
        hz,
        rivals,
        deadline=deadline,
        **caps,
    )
    cut_hz = None
    post_uppers = []
    if (
        oracle_result.status
        == "complete_conflict_graph_candidate"
    ):
        cut_hz = _copy_parent_with_clique_cut(
            hz, oracle_result.literals
        )
        for rival in rivals:
            post_uppers.append(
                _highs_property_upper(
                    cut_hz, rival, deadline=deadline
                )
            )
            if time.monotonic() >= deadline:
                raise PersistentConflictOracleError(
                    "deadline_expired_in_post_telemetry"
                )
    pre_tuple = tuple(pre_uppers)
    post_tuple = tuple(post_uppers)
    if cut_hz is None:
        status = "incomplete_conflict_graph_candidate"
    elif (
        not _strict_safe_candidate(pre_tuple, rivals)
        and _strict_safe_candidate(post_tuple, rivals)
    ):
        status = "unknown_to_safe_candidate"
    elif _strict_safe_candidate(post_tuple, rivals):
        status = "safe_candidate"
    else:
        status = "cut_candidate"
    result = PersistentPCPCCResult(
        status=status,
        hz=cut_hz,
        invocation=invocation,
        oracle_result=oracle_result,
        pre_property_uppers=pre_tuple,
        post_property_uppers=post_tuple,
        wall_seconds=float(time.perf_counter() - started),
    )
    return _issue_live_persistent_pc_pcc_result(
        result=result,
        parent=hz,
        rivals=rivals,
        invocation=invocation,
    )


def _valid_property_telemetry(
    values: Any,
    *,
    expected_count: int,
) -> bool:
    return (
        isinstance(values, tuple)
        and len(values) == expected_count
        and all(
            value is None
            or (type(value) is float and math.isfinite(value))
            for value in values
        )
    )


def _verify_persistent_pc_pcc_shell(
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PersistentPCPCCResult,
) -> bool:
    """Check the cut wrapper after its oracle result was exact-audited."""

    try:
        if (
            not isinstance(parent, SparseHZono)
            or not isinstance(result, PersistentPCPCCResult)
            or not isinstance(
                result.invocation,
                PersistentPCPCCInvocationSpec,
            )
            or result.proof_authority is not False
            or result.status
            not in {
                "unknown_to_safe_candidate",
                "safe_candidate",
                "cut_candidate",
                "incomplete_conflict_graph_candidate",
            }
            or type(result.wall_seconds) is not float
            or not math.isfinite(result.wall_seconds)
            or result.wall_seconds < 0.0
            or not _valid_property_telemetry(
                result.pre_property_uppers,
                expected_count=len(rivals),
            )
        ):
            return False
        invocation_caps = _validated_invocation_caps(
            parent, rivals, result.invocation
        )
        if (
            dict(invocation_caps)
            != result.oracle_result.telemetry.get("caps")
        ):
            return False
        complete = (
            result.oracle_result.status
            == "complete_conflict_graph_candidate"
        )
        if not complete:
            return (
                result.status
                == "incomplete_conflict_graph_candidate"
                and result.hz is None
                and result.post_property_uppers == ()
            )
        if not _valid_property_telemetry(
            result.post_property_uppers,
            expected_count=len(rivals),
        ):
            return False
        expected_cut = _copy_parent_with_clique_cut(
            parent, result.oracle_result.literals
        )
        if (
            not isinstance(result.hz, SparseHZono)
            or sparse_hz_semantic_digest(result.hz)
            != sparse_hz_semantic_digest(expected_cut)
        ):
            return False
        pre_safe = _strict_safe_candidate(
            result.pre_property_uppers, rivals
        )
        post_safe = _strict_safe_candidate(
            result.post_property_uppers, rivals
        )
        if not pre_safe and post_safe:
            expected_status = "unknown_to_safe_candidate"
        elif post_safe:
            expected_status = "safe_candidate"
        else:
            expected_status = "cut_candidate"
        return result.status == expected_status
    except (
        PersistentConflictOracleError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def verify_persistent_pc_pcc_structural_result(
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PersistentPCPCCResult,
    *,
    deadline: Optional[float] = None,
) -> bool:
    """Repeat the exact cut audit; runtime telemetry remains diagnostic."""

    try:
        deadline_value = (
            time.monotonic() + 60.0
            if deadline is None
            else float(deadline)
        )
        return (
            math.isfinite(deadline_value)
            and isinstance(result, PersistentPCPCCResult)
            and verify_persistent_conflict_oracle_result(
                parent,
                rivals,
                result.oracle_result,
                deadline=deadline_value,
            )
            and _verify_persistent_pc_pcc_shell(
                parent, rivals, result
            )
        )
    except (
        PersistentConflictOracleError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def verify_persistent_pc_pcc_result(
    parent: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PersistentPCPCCResult,
    *,
    invocation: PersistentPCPCCInvocationSpec,
) -> bool:
    """Consume the exact live candidate without repeating Fraction replay."""

    return _consume_live_persistent_pc_pcc_result(
        parent, rivals, result, invocation
    )


__all__ = [
    "ExactDualRayConflictCertificate",
    "ExactSourceTermV2",
    "NativeSplitRowObjectiveDualProposal",
    "PersistentConflictOracleError",
    "PersistentConflictOracleResult",
    "PersistentPCPCCInvocationSpec",
    "PersistentPCPCCResult",
    "PersistentPairRecord",
    "exact_certificate_from_highs_dual_ray_candidate",
    "make_persistent_pc_pcc_invocation_spec",
    "propose_native_split_row_objective_duals",
    "revoke_persistent_pc_pcc_result",
    "run_persistent_conflict_oracle_candidate",
    "run_persistent_pc_pcc_candidate",
    "verify_exact_dual_ray_conflict_certificate",
    "verify_persistent_conflict_oracle_result",
    "verify_persistent_pc_pcc_result",
    "verify_persistent_pc_pcc_structural_result",
]
