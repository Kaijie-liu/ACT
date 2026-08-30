#!/usr/bin/env python3
"""Pair-local exact infeasibility candidates for PCOH pattern coverage.

This module is intentionally disconnected from the HybridZ pipeline and all
verdict paths.  For ``1 <= k <= 4`` caller-selected exact-ReLU binary ids it
enumerates the canonical signed literal pairs.  Each numerical proposal LP
contains exactly the six original upper rows belonging to the two ReLU
mappings, every original factor column with its ordinary ``[-1, 1]`` box,
and only the queried two binary columns fixed to their signed phases.

HiGHS is proposal-only.  An infeasibility ray is zero-padded back to the
complete ``upper_then_equality`` source-row frame and accepted only by the
existing live-parent sparse-Fraction Farkas replayer.  A full pattern is
labelled ``certified_empty_by_pair`` only when it contains such an accepted
edge.  Every other pattern is ``not_certified_empty`` -- never "feasible".

No full-parent constraint matrix, private parent snapshot, sparse hstack, or
sparse vstack is constructed here.  The result remains ``proof_authority``
false and merely exposes exact certificate references for a future one-use
``eta_p = -1`` materializer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from numbers import Integral
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

try:  # Candidate-only optional dependency.
    import highspy as _highspy
except Exception:  # pragma: no cover
    _highspy = None

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    OperatorExactReLUPhaseMapping,
    OperatorExactReLUPhaseSelection,
    verify_operator_exact_relu_property_phase_selection,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    ExactDualRayConflictCertificate,
    _exact_sparse_source_row,
    _ordered_source_frame_digest,
    _verify_exact_certificate_with_source_frame,
    exact_certificate_from_highs_dual_ray_candidate,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
    _literal_binding_digest,
)
from act.back_end.solver.solver_hz import SparseHZono


class PhaseConditionedPairInfeasibilityError(ValueError):
    """Malformed, stale, over-budget, or unsupported caller contract."""


@dataclass(frozen=True)
class PairLocalCaps:
    max_stable_bits: int = 4
    max_signed_pair_queries: int = 24
    max_local_rows: int = 6
    max_local_nonzeros: int = 200_000
    max_source_terms: int = 6
    max_multiplier_bits: int = 256
    max_exact_bits: int = 4096
    max_exact_nonzeros: int = 200_000


@dataclass(frozen=True)
class SignedPairRecord:
    """One signed pair query and its exact-certificate reference."""

    pair: Tuple[Tuple[int, int], Tuple[int, int]]
    source_upper_rows: Tuple[int, ...]
    source_row_sha256: Tuple[str, ...]
    local_row_map_sha256: str
    status: str
    local_model_rows: int
    local_model_columns: int
    local_model_nonzeros: int
    raw_ray_nonzero_rows: int
    certificate_sha256: Optional[str]
    model_closed: bool
    solver_threads: int
    record_sha256: str
    proof_authority: bool = False


@dataclass(frozen=True)
class PatternCoverage:
    """One complete signed pattern and an optional certified conflict edge."""

    pattern: Tuple[int, ...]
    status: str
    witness_pair: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    certificate_sha256: Optional[str]
    eta_fixed_value: Optional[int]
    coverage_sha256: str
    proof_authority: bool = False


@dataclass(frozen=True)
class PairInfeasibilityBundle:
    """Canonical pair certificates and all-pattern coverage."""

    status: str
    stable_bit_ids: Tuple[int, ...]
    parent_semantic_digest: str
    terminal_parent_semantic_digest: str
    property_digest: str
    selection_digest: str
    operator_row_tag_digest: str
    ordered_source_frame_sha256: str
    caps: PairLocalCaps
    records: Tuple[SignedPairRecord, ...]
    certificates: Tuple[ExactDualRayConflictCertificate, ...]
    coverage: Tuple[PatternCoverage, ...]
    receipt: Dict[str, Any]
    bundle_sha256: str
    proof_authority: bool = False


@dataclass(frozen=True)
class _LocalSolveOutcome:
    status: str
    raw_ray: Optional[np.ndarray]
    model_rows: int
    model_columns: int
    model_nonzeros: int
    model_closed: bool


_DEFAULT_CAPS = PairLocalCaps()
_HARD_CAPS = PairLocalCaps(
    max_stable_bits=4,
    max_signed_pair_queries=24,
    max_local_rows=6,
    max_local_nonzeros=1_000_000,
    max_source_terms=6,
    max_multiplier_bits=2048,
    max_exact_bits=16384,
    max_exact_nonzeros=1_000_000,
)
_VALID_RECORD_STATUSES = frozenset(
    {
        "certified_conflict",
        "feasible_or_unknown",
        "infeasible_without_ray",
        "exact_replay_rejected",
        "deadline_expired",
        "candidate_error",
        "model_close_failed",
    }
)


def _canonical_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PhaseConditionedPairInfeasibilityError(
            "noncanonical_json_payload"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PhaseConditionedPairInfeasibilityError(f"{name}_not_integer")
    return int(value)


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"deadline_expired:{stage}")


def _caps_payload(caps: PairLocalCaps) -> Dict[str, int]:
    return {
        "max_stable_bits": caps.max_stable_bits,
        "max_signed_pair_queries": caps.max_signed_pair_queries,
        "max_local_rows": caps.max_local_rows,
        "max_local_nonzeros": caps.max_local_nonzeros,
        "max_source_terms": caps.max_source_terms,
        "max_multiplier_bits": caps.max_multiplier_bits,
        "max_exact_bits": caps.max_exact_bits,
        "max_exact_nonzeros": caps.max_exact_nonzeros,
    }


def _normalize_caps(caps: Any) -> PairLocalCaps:
    if type(caps) is not PairLocalCaps:
        raise PhaseConditionedPairInfeasibilityError("caps_wrong_type")
    values = {}
    for name, hard in _caps_payload(_HARD_CAPS).items():
        value = _strict_int(getattr(caps, name), name=f"caps_{name}")
        if value < 1 or value > hard:
            raise PhaseConditionedPairInfeasibilityError(
                f"caps_{name}_out_of_range"
            )
        values[name] = value
    if values["max_local_rows"] != 6:
        raise PhaseConditionedPairInfeasibilityError(
            "caps_local_rows_must_equal_six"
        )
    if values["max_source_terms"] > values["max_local_rows"]:
        raise PhaseConditionedPairInfeasibilityError(
            "caps_source_terms_exceed_local_rows"
        )
    return PairLocalCaps(**values)


def _stable_ids(values: Sequence[Any], *, caps: PairLocalCaps) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise PhaseConditionedPairInfeasibilityError(
            "stable_bit_ids_not_sequence"
        )
    result = tuple(
        _strict_int(value, name=f"stable_bit_id_{offset}")
        for offset, value in enumerate(values)
    )
    if not 1 <= len(result) <= min(4, caps.max_stable_bits):
        raise PhaseConditionedPairInfeasibilityError(
            "stable_bit_count_out_of_range"
        )
    if any(value < 0 for value in result) or len(set(result)) != len(result):
        raise PhaseConditionedPairInfeasibilityError(
            "stable_bit_ids_negative_or_duplicate"
        )
    return tuple(sorted(result))


def _selection_caps_kwargs(selection: OperatorExactReLUPhaseSelection) -> Dict[str, Any]:
    caps = selection.caps
    return {
        "max_rivals": caps.max_rivals,
        "max_binaries": caps.max_binaries,
        "max_work_items": caps.max_work_items,
        "timeout_seconds": caps.timeout_seconds,
    }


def _validate_source_matrix(
    matrix: Any,
    *,
    shape: Tuple[int, int],
    name: str,
) -> sp.csr_matrix:
    if type(matrix) is not sp.csr_matrix or matrix.shape != shape:
        raise PhaseConditionedPairInfeasibilityError(f"{name}_not_exact_csr")
    if matrix.dtype != np.dtype(np.float64):
        raise PhaseConditionedPairInfeasibilityError(f"{name}_dtype_invalid")
    indptr = np.asarray(matrix.indptr)
    indices = np.asarray(matrix.indices)
    data = np.asarray(matrix.data)
    if (
        indptr.ndim != 1
        or indices.ndim != 1
        or data.ndim != 1
        or int(indptr.size) != shape[0] + 1
        or int(indptr[0]) != 0
        or int(indptr[-1]) != int(indices.size)
        or int(indices.size) != int(data.size)
        or np.any(indptr[1:] < indptr[:-1])
        or (indices.size and (np.any(indices < 0) or np.any(indices >= shape[1])))
        or (data.size and (not np.all(np.isfinite(data)) or np.any(data == 0.0)))
    ):
        raise PhaseConditionedPairInfeasibilityError(f"{name}_malformed")
    if indices.size > 1:
        nonincreasing = indices[1:] <= indices[:-1]
        cuts = np.asarray(indptr[1:-1], dtype=np.int64)
        cuts = cuts[(cuts > 0) & (cuts < indices.size)]
        nonincreasing[cuts - 1] = False
        if np.any(nonincreasing):
            raise PhaseConditionedPairInfeasibilityError(
                f"{name}_noncanonical_columns"
            )
    return matrix


def _mapping_rows(mapping: OperatorExactReLUPhaseMapping) -> Tuple[int, int, int]:
    return (
        int(mapping.lower_upper_row),
        int(mapping.x_branch_upper_row),
        int(mapping.zero_branch_upper_row),
    )


def _source_binding(
    hz: SparseHZono,
    left: OperatorExactReLUPhaseMapping,
    right: OperatorExactReLUPhaseMapping,
    *,
    parent_digest: str,
    source_frame_digest: str,
    deadline: float,
    caps: PairLocalCaps,
) -> Tuple[Tuple[int, ...], Tuple[str, ...], str]:
    rows = tuple(sorted((*_mapping_rows(left), *_mapping_rows(right))))
    if len(rows) != 6 or len(set(rows)) != 6:
        raise PhaseConditionedPairInfeasibilityError(
            "pair_does_not_map_to_six_unique_rows"
        )
    digests = []
    nonzeros = 0
    for row in rows:
        _, _, digest, count = _exact_sparse_source_row(
            hz,
            "upper",
            row,
            deadline=deadline,
            max_nonzeros=caps.max_exact_nonzeros - nonzeros,
        )
        nonzeros += count
        if nonzeros > caps.max_exact_nonzeros:
            raise PhaseConditionedPairInfeasibilityError(
                "pair_source_nonzero_cap_exceeded"
            )
        digests.append(digest)
    payload = {
        "schema": "act.hybridz_pc_pair_local_row_map.v1",
        "parent_semantic_digest": parent_digest,
        "ordered_source_frame_sha256": source_frame_digest,
        "row_order": "ascending_original_upper_row_id",
        "source_upper_rows": list(rows),
        "source_row_sha256": digests,
        "local_rows": 6,
    }
    return rows, tuple(digests), _canonical_sha256(payload)


def _literal(
    *, parent_digest: str, property_digest: str, stable_id: int, phase: int
) -> PhaseLiteral:
    return PhaseLiteral(
        stable_bcol_id=stable_id,
        phase=phase,
        binding_digest=_literal_binding_digest(
            parent_digest=parent_digest,
            property_digest=property_digest,
            stable_bcol_id=stable_id,
            phase=phase,
        ),
    )


def _replay_exact_certificate_on_precomputed_frame(
    hz: SparseHZono,
    certificate: ExactDualRayConflictCertificate,
    *,
    property_digest: str,
    parent_digest: str,
    source_frame_digest: str,
    deadline: float,
    caps: PairLocalCaps,
) -> bool:
    """Replay exactly without re-hashing the already sealed full frame."""

    return _verify_exact_certificate_with_source_frame(
        hz,
        certificate,
        property_digest=property_digest,
        parent_digest=parent_digest,
        source_frame_digest=source_frame_digest,
        deadline=deadline,
        max_source_terms=caps.max_source_terms,
        max_multiplier_bits=caps.max_multiplier_bits,
        max_exact_bits=caps.max_exact_bits,
        max_exact_nonzeros=caps.max_exact_nonzeros,
    )


def _pair_payload(pair: Tuple[Tuple[int, int], Tuple[int, int]]) -> list[list[int]]:
    return [list(pair[0]), list(pair[1])]


def _record_payload(record: SignedPairRecord, *, include_digest: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": "act.hybridz_pc_signed_pair_record.v1",
        "pair": _pair_payload(record.pair),
        "source_upper_rows": list(record.source_upper_rows),
        "source_row_sha256": list(record.source_row_sha256),
        "local_row_map_sha256": record.local_row_map_sha256,
        "status": record.status,
        "local_model_rows": record.local_model_rows,
        "local_model_columns": record.local_model_columns,
        "local_model_nonzeros": record.local_model_nonzeros,
        "raw_ray_nonzero_rows": record.raw_ray_nonzero_rows,
        "certificate_sha256": record.certificate_sha256,
        "model_closed": record.model_closed,
        "solver_threads": record.solver_threads,
        "proof_authority": record.proof_authority,
    }
    if include_digest:
        payload["record_sha256"] = record.record_sha256
    return payload


def _make_record(
    *,
    pair: Tuple[Tuple[int, int], Tuple[int, int]],
    source_rows: Tuple[int, ...],
    source_digests: Tuple[str, ...],
    row_map_digest: str,
    status: str,
    outcome: _LocalSolveOutcome,
    raw_ray_nonzeros: int,
    certificate_sha256: Optional[str],
) -> SignedPairRecord:
    placeholder = SignedPairRecord(
        pair=pair,
        source_upper_rows=source_rows,
        source_row_sha256=source_digests,
        local_row_map_sha256=row_map_digest,
        status=status,
        local_model_rows=outcome.model_rows,
        local_model_columns=outcome.model_columns,
        local_model_nonzeros=outcome.model_nonzeros,
        raw_ray_nonzero_rows=raw_ray_nonzeros,
        certificate_sha256=certificate_sha256,
        model_closed=outcome.model_closed,
        solver_threads=1,
        record_sha256="",
        proof_authority=False,
    )
    return replace(
        placeholder,
        record_sha256=_canonical_sha256(
            _record_payload(placeholder, include_digest=False)
        ),
    )


def _coverage_payload(item: PatternCoverage, *, include_digest: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": "act.hybridz_pc_pattern_pair_coverage.v1",
        "pattern": list(item.pattern),
        "status": item.status,
        "witness_pair": (
            None if item.witness_pair is None else _pair_payload(item.witness_pair)
        ),
        "certificate_sha256": item.certificate_sha256,
        "eta_fixed_value": item.eta_fixed_value,
        "proof_authority": item.proof_authority,
    }
    if include_digest:
        payload["coverage_sha256"] = item.coverage_sha256
    return payload


def _build_coverage(
    stable_ids: Tuple[int, ...],
    certified: Mapping[
        Tuple[Tuple[int, int], Tuple[int, int]], str
    ],
) -> Tuple[PatternCoverage, ...]:
    output = []
    for pattern in itertools.product((-1, 1), repeat=len(stable_ids)):
        assignment = dict(zip(stable_ids, pattern))
        witnesses = tuple(
            sorted(
                pair
                for pair in certified
                if assignment[pair[0][0]] == pair[0][1]
                and assignment[pair[1][0]] == pair[1][1]
            )
        )
        witness = witnesses[0] if witnesses else None
        certificate = None if witness is None else certified[witness]
        placeholder = PatternCoverage(
            pattern=tuple(int(value) for value in pattern),
            status=(
                "not_certified_empty"
                if witness is None
                else "certified_empty_by_pair"
            ),
            witness_pair=witness,
            certificate_sha256=certificate,
            eta_fixed_value=None if witness is None else -1,
            coverage_sha256="",
            proof_authority=False,
        )
        output.append(
            replace(
                placeholder,
                coverage_sha256=_canonical_sha256(
                    _coverage_payload(placeholder, include_digest=False)
                ),
            )
        )
    return tuple(output)


def _zero_pad_local_ray(
    raw_ray: np.ndarray,
    source_rows: Tuple[int, ...],
    *,
    full_rows: int,
) -> np.ndarray:
    ray = np.asarray(raw_ray, dtype=np.float64).reshape(-1)
    if (
        ray.size != 6
        or len(source_rows) != 6
        or len(set(source_rows)) != 6
        or not np.all(np.isfinite(ray))
        or full_rows < 6
        or any(row < 0 or row >= full_rows for row in source_rows)
    ):
        raise PhaseConditionedPairInfeasibilityError(
            "local_ray_or_row_map_invalid"
        )
    full = np.zeros(full_rows, dtype=np.float64)
    full[np.asarray(source_rows, dtype=np.int64)] = ray
    return full


def _require_highs_ok(status: Any, operation: str) -> None:
    if status != _highspy.HighsStatus.kOk:
        raise RuntimeError(f"highs_{operation}_failed")


def _solve_local_pair(
    hz: SparseHZono,
    *,
    binary_positions: Tuple[int, int],
    phases: Tuple[int, int],
    source_rows: Tuple[int, ...],
    deadline: float,
    caps: PairLocalCaps,
) -> _LocalSolveOutcome:
    """Build, solve, and explicitly close one six-row candidate model."""

    if _highspy is None:
        return _LocalSolveOutcome(
            "candidate_error", None, 0, 0, 0, True
        )
    _check_deadline(deadline, "before_local_model")
    if (
        len(binary_positions) != 2
        or len(set(binary_positions)) != 2
        or any(position < 0 or position >= hz.n_bin for position in binary_positions)
        or any(phase not in {-1, 1} for phase in phases)
        or len(source_rows) != 6
        or len(set(source_rows)) != 6
        or any(row < 0 or row >= hz.n_ub for row in source_rows)
    ):
        raise PhaseConditionedPairInfeasibilityError(
            "local_pair_shape_invalid"
        )
    if (
        type(hz.Auc) is not sp.csr_matrix
        or hz.Auc.shape != (hz.n_ub, hz.n_cont)
        or type(hz.Aub) is not sp.csr_matrix
        or hz.Aub.shape != (hz.n_ub, hz.n_bin)
    ):
        raise PhaseConditionedPairInfeasibilityError(
            "parent_upper_blocks_invalid"
        )
    if (
        type(hz.ub) is not np.ndarray
        or hz.ub.dtype != np.dtype(np.float64)
        or hz.ub.ndim != 1
        or hz.ub.size != hz.n_ub
    ):
        raise PhaseConditionedPairInfeasibilityError("upper_rhs_invalid")

    rows = np.asarray(source_rows, dtype=np.int64)
    # These are the only copied constraint buffers: exactly six original rows.
    continuous = hz.Auc[rows, :].tocsr(copy=True)
    binary = hz.Aub[rows, :].tocsr(copy=True)
    rhs = np.asarray(hz.ub[rows], dtype=np.float64).copy()
    _validate_source_matrix(
        continuous, shape=(6, hz.n_cont), name="local_Auc"
    )
    _validate_source_matrix(
        binary, shape=(6, hz.n_bin), name="local_Aub"
    )
    if not np.all(np.isfinite(rhs)):
        raise PhaseConditionedPairInfeasibilityError(
            "local_upper_rhs_nonfinite"
        )
    local_nonzeros = int(continuous.nnz + binary.nnz)
    if local_nonzeros > caps.max_local_nonzeros:
        raise PhaseConditionedPairInfeasibilityError(
            "local_nonzero_cap_exceeded"
        )
    if int(continuous.shape[0]) != 6 or int(binary.shape[0]) != 6:
        raise PhaseConditionedPairInfeasibilityError("local_row_count_mismatch")

    highs = None
    closed = False
    outcome = _LocalSolveOutcome(
        "candidate_error", None, 6, hz.n_cont + hz.n_bin, local_nonzeros, False
    )
    try:
        highs = _highspy.Highs()
        for name, value in (
            ("output_flag", False),
            ("solver", "simplex"),
            ("presolve", "off"),
            ("threads", 1),
            ("small_matrix_value", 1.0e-12),
        ):
            _require_highs_ok(highs.setOptionValue(name, value), f"set_{name}")
        n_variables = hz.n_cont + hz.n_bin
        lower = -np.ones(n_variables, dtype=np.float64)
        upper = np.ones(n_variables, dtype=np.float64)
        for position, phase in zip(binary_positions, phases):
            column = hz.n_cont + position
            lower[column] = float(phase)
            upper[column] = float(phase)
        empty_i32 = np.zeros(0, dtype=np.int32)
        empty_f64 = np.zeros(0, dtype=np.float64)
        _require_highs_ok(
            highs.addCols(
                n_variables,
                np.zeros(n_variables, dtype=np.float64),
                lower,
                upper,
                0,
                empty_i32,
                empty_i32,
                empty_f64,
            ),
            "add_columns",
        )
        _check_deadline(deadline, "after_local_columns")
        _require_highs_ok(
            highs.addRows(
                6,
                np.full(6, -_highspy.kHighsInf, dtype=np.float64),
                rhs,
                int(continuous.nnz),
                np.asarray(continuous.indptr, dtype=np.int32),
                np.asarray(continuous.indices, dtype=np.int32),
                np.asarray(continuous.data, dtype=np.float64),
            ),
            "add_continuous_rows",
        )
        injected = 0
        for local_row in range(6):
            start = int(binary.indptr[local_row])
            stop = int(binary.indptr[local_row + 1])
            for offset in range(start, stop):
                value = float(binary.data[offset])
                if abs(value) <= 1.0e-12:
                    continue
                _require_highs_ok(
                    highs.changeCoeff(
                        local_row,
                        hz.n_cont + int(binary.indices[offset]),
                        value,
                    ),
                    "inject_binary_coefficient",
                )
                injected += 1
                if injected % 64 == 0:
                    _check_deadline(deadline, "inject_binary_coefficients")
        expected_model_nnz = int(
            np.count_nonzero(np.abs(continuous.data) > 1.0e-12) + injected
        )
        if (
            int(highs.getNumRow()) != 6
            or int(highs.getNumCol()) != n_variables
            or int(highs.getNumNz()) != expected_model_nnz
        ):
            raise RuntimeError("highs_local_topology_postcondition_failed")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("deadline_expired:before_local_solve")
        _require_highs_ok(
            highs.setOptionValue("time_limit", float(remaining)),
            "set_time_limit",
        )
        run_status = highs.run()
        if time.monotonic() >= deadline:
            raise TimeoutError("deadline_expired:during_local_solve")
        if run_status != _highspy.HighsStatus.kOk:
            outcome = _LocalSolveOutcome(
                "feasible_or_unknown",
                None,
                6,
                n_variables,
                expected_model_nnz,
                False,
            )
        elif highs.getModelStatus() != _highspy.HighsModelStatus.kInfeasible:
            outcome = _LocalSolveOutcome(
                "feasible_or_unknown",
                None,
                6,
                n_variables,
                expected_model_nnz,
                False,
            )
        else:
            ray_status, ray_exists, raw_ray = highs.getDualRay()
            if (
                ray_status != _highspy.HighsStatus.kOk
                or not ray_exists
            ):
                outcome = _LocalSolveOutcome(
                    "infeasible_without_ray",
                    None,
                    6,
                    n_variables,
                    expected_model_nnz,
                    False,
                )
            else:
                ray = np.asarray(raw_ray, dtype=np.float64).reshape(-1).copy()
                if ray.size != 6 or not np.all(np.isfinite(ray)):
                    outcome = _LocalSolveOutcome(
                        "infeasible_without_ray",
                        None,
                        6,
                        n_variables,
                        expected_model_nnz,
                        False,
                    )
                else:
                    outcome = _LocalSolveOutcome(
                        "infeasible_with_ray",
                        ray,
                        6,
                        n_variables,
                        expected_model_nnz,
                        False,
                    )
    except TimeoutError:
        outcome = _LocalSolveOutcome(
            "deadline_expired",
            None,
            outcome.model_rows,
            outcome.model_columns,
            outcome.model_nonzeros,
            False,
        )
    except Exception:
        outcome = _LocalSolveOutcome(
            "candidate_error",
            None,
            outcome.model_rows,
            outcome.model_columns,
            outcome.model_nonzeros,
            False,
        )
    finally:
        if highs is not None:
            try:
                try:
                    clear_status = highs.clear()
                    closed = clear_status == _highspy.HighsStatus.kOk
                except Exception:
                    closed = False
            finally:
                highs = None
        else:
            closed = True
    if not closed:
        return _LocalSolveOutcome(
            "model_close_failed",
            None,
            outcome.model_rows,
            outcome.model_columns,
            outcome.model_nonzeros,
            False,
        )
    return replace(outcome, model_closed=True)


def _bundle_payload(bundle: PairInfeasibilityBundle, *, include_digest: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": "act.hybridz_pc_pair_infeasibility_bundle.v1",
        "status": bundle.status,
        "stable_bit_ids": list(bundle.stable_bit_ids),
        "parent_semantic_digest": bundle.parent_semantic_digest,
        "terminal_parent_semantic_digest": bundle.terminal_parent_semantic_digest,
        "property_digest": bundle.property_digest,
        "selection_digest": bundle.selection_digest,
        "operator_row_tag_digest": bundle.operator_row_tag_digest,
        "ordered_source_frame_sha256": bundle.ordered_source_frame_sha256,
        "caps": _caps_payload(bundle.caps),
        "records": [
            _record_payload(record, include_digest=True) for record in bundle.records
        ],
        "certificate_sha256": [
            certificate.certificate_sha256 for certificate in bundle.certificates
        ],
        "coverage": [
            _coverage_payload(item, include_digest=True) for item in bundle.coverage
        ],
        "receipt": bundle.receipt,
        "proof_authority": bundle.proof_authority,
    }
    if include_digest:
        payload["bundle_sha256"] = bundle.bundle_sha256
    return payload


def _make_receipt(
    *,
    stable_ids: Tuple[int, ...],
    records: Tuple[SignedPairRecord, ...],
    coverage: Tuple[PatternCoverage, ...],
    parent_digest: str,
    terminal_parent_digest: str,
    property_digest: str,
    selection_digest: str,
    source_frame_digest: str,
    caps: PairLocalCaps,
) -> Dict[str, Any]:
    receipt: Dict[str, Any] = {
        "schema": "act.hybridz_pc_pair_infeasibility_receipt.v1",
        "algorithm": "six_original_relu_rows_per_signed_pair",
        "proof_authority": False,
        "producer_proof_authority": False,
        "highs_proof_authority": False,
        "exact_fraction_replay_required": True,
        "certificate_replays_reuse_precomputed_source_frame": True,
        "source_frame_digest_computations": 1,
        "full_parent_milp_used": False,
        "full_parent_csr_loaded_into_candidate": False,
        "full_parent_snapshot_created": False,
        "sparse_hstack_used": False,
        "sparse_vstack_used": False,
        "solver_threads": 1,
        "absolute_deadline": True,
        "explicit_model_close_required": True,
        "unknown_never_labelled_feasible": True,
        "empty_pattern_eta_fixed_value": -1,
        "empty_pattern_finite_upper_placeholder_hex": 0.0.hex(),
        "stable_bit_ids": list(stable_ids),
        "signed_pair_queries": len(records),
        "certified_pair_conflicts": sum(
            record.status == "certified_conflict" for record in records
        ),
        "patterns": len(coverage),
        "certified_empty_patterns": sum(
            item.status == "certified_empty_by_pair" for item in coverage
        ),
        "not_certified_empty_patterns": sum(
            item.status == "not_certified_empty" for item in coverage
        ),
        "all_models_closed": all(record.model_closed for record in records),
        "parent_semantic_digest": parent_digest,
        "terminal_parent_semantic_digest": terminal_parent_digest,
        "property_digest": property_digest,
        "selection_digest": selection_digest,
        "ordered_source_frame_sha256": source_frame_digest,
        "caps": _caps_payload(caps),
        "record_sha256": [record.record_sha256 for record in records],
        "coverage_sha256": [item.coverage_sha256 for item in coverage],
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def run_phase_conditioned_pair_infeasibility_candidate(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    stable_bit_ids: Sequence[Any],
    deadline: float,
    caps: PairLocalCaps = _DEFAULT_CAPS,
) -> PairInfeasibilityBundle:
    """Enumerate signed pair conflicts and derive exact empty-pattern coverage."""

    normalized_caps = _normalize_caps(caps)
    try:
        deadline_value = float(deadline)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhaseConditionedPairInfeasibilityError("deadline_invalid") from exc
    if isinstance(deadline, bool) or not math.isfinite(deadline_value):
        raise PhaseConditionedPairInfeasibilityError("deadline_invalid")
    if time.monotonic() >= deadline_value:
        raise PhaseConditionedPairInfeasibilityError(
            "deadline_expired:entry"
        )
    if (
        type(build) is not OperatorHZBuild
        or type(build.hz) is not SparseHZono
        or type(selection) is not OperatorExactReLUPhaseSelection
    ):
        raise PhaseConditionedPairInfeasibilityError(
            "build_or_selection_wrong_type"
        )
    selection_remaining = deadline_value - time.monotonic()
    if selection_remaining <= 0.0 or selection.caps.timeout_seconds > selection_remaining:
        raise PhaseConditionedPairInfeasibilityError(
            "deadline_too_short_for_selection_verification"
        )
    if not verify_operator_exact_relu_property_phase_selection(
        build,
        rivals,
        selection,
        **_selection_caps_kwargs(selection),
    ):
        raise PhaseConditionedPairInfeasibilityError(
            "selection_live_verification_failed"
        )
    _check_deadline(deadline_value, "after_selection_verification")
    stable_ids = _stable_ids(stable_bit_ids, caps=normalized_caps)
    mappings = {
        mapping.stable_bcol_id: mapping for mapping in selection.mappings
    }
    if any(stable_id not in mappings for stable_id in stable_ids):
        raise PhaseConditionedPairInfeasibilityError(
            "stable_bit_id_missing_from_verified_selection"
        )
    query_count = 4 * math.comb(len(stable_ids), 2)
    if query_count > normalized_caps.max_signed_pair_queries:
        raise PhaseConditionedPairInfeasibilityError(
            "signed_pair_query_cap_exceeded"
        )
    hz = build.hz
    parent_digest = sparse_hz_semantic_digest(hz)
    if parent_digest != selection.parent_semantic_digest:
        raise PhaseConditionedPairInfeasibilityError(
            "selection_parent_digest_stale"
        )
    source_frame_digest = _ordered_source_frame_digest(
        hz,
        parent_digest=parent_digest,
        deadline=deadline_value,
    )
    if hz.bcol_ids is None:
        raise PhaseConditionedPairInfeasibilityError("stable_binary_ids_missing")
    bcol_ids = np.asarray(hz.bcol_ids)
    if (
        bcol_ids.dtype != np.dtype(np.int64)
        or bcol_ids.ndim != 1
        or bcol_ids.size != hz.n_bin
        or len(set(int(value) for value in bcol_ids.tolist())) != hz.n_bin
    ):
        raise PhaseConditionedPairInfeasibilityError(
            "stable_binary_ids_malformed"
        )
    positions = {int(value): index for index, value in enumerate(bcol_ids.tolist())}

    structural: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[str, ...], str]] = {}
    for left_id, right_id in itertools.combinations(stable_ids, 2):
        structural[(left_id, right_id)] = _source_binding(
            hz,
            mappings[left_id],
            mappings[right_id],
            parent_digest=parent_digest,
            source_frame_digest=source_frame_digest,
            deadline=deadline_value,
            caps=normalized_caps,
        )
    _check_deadline(deadline_value, "after_pair_source_bindings")

    records = []
    certificates = []
    certified: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str] = {}
    for left_id, right_id in itertools.combinations(stable_ids, 2):
        source_rows, source_digests, row_map_digest = structural[
            (left_id, right_id)
        ]
        for left_phase, right_phase in itertools.product((-1, 1), repeat=2):
            pair_key = (
                (left_id, int(left_phase)),
                (right_id, int(right_phase)),
            )
            if time.monotonic() >= deadline_value:
                outcome = _LocalSolveOutcome(
                    "deadline_expired", None, 0, 0, 0, True
                )
            else:
                try:
                    outcome = _solve_local_pair(
                        hz,
                        binary_positions=(positions[left_id], positions[right_id]),
                        phases=(left_phase, right_phase),
                        source_rows=source_rows,
                        deadline=deadline_value,
                        caps=normalized_caps,
                    )
                except TimeoutError:
                    outcome = _LocalSolveOutcome(
                        "deadline_expired", None, 0, 0, 0, True
                    )
                except Exception:
                    outcome = _LocalSolveOutcome(
                        "candidate_error", None, 0, 0, 0, True
                    )

            status = outcome.status
            certificate = None
            raw_nonzeros = 0
            if not outcome.model_closed:
                status = "model_close_failed"
            elif (
                outcome.status == "infeasible_with_ray"
                and outcome.raw_ray is not None
                and outcome.model_rows == 6
                and outcome.model_columns == hz.n_cont + hz.n_bin
            ):
                raw_nonzeros = int(np.count_nonzero(outcome.raw_ray))
                try:
                    full_ray = _zero_pad_local_ray(
                        outcome.raw_ray,
                        source_rows,
                        full_rows=hz.n_ub + hz.n_eq,
                    )
                    literals = (
                        _literal(
                            parent_digest=parent_digest,
                            property_digest=selection.property_digest,
                            stable_id=left_id,
                            phase=left_phase,
                        ),
                        _literal(
                            parent_digest=parent_digest,
                            property_digest=selection.property_digest,
                            stable_id=right_id,
                            phase=right_phase,
                        ),
                    )
                    certificate = exact_certificate_from_highs_dual_ray_candidate(
                        hz,
                        literals,
                        full_ray,
                        parent_digest=parent_digest,
                        property_digest=selection.property_digest,
                        source_frame_digest=source_frame_digest,
                        deadline=deadline_value,
                        max_source_terms=normalized_caps.max_source_terms,
                        max_multiplier_bits=normalized_caps.max_multiplier_bits,
                        max_exact_bits=normalized_caps.max_exact_bits,
                        max_exact_nonzeros=normalized_caps.max_exact_nonzeros,
                    )
                    if certificate is not None and not _replay_exact_certificate_on_precomputed_frame(
                        hz,
                        certificate,
                        property_digest=selection.property_digest,
                        parent_digest=parent_digest,
                        source_frame_digest=source_frame_digest,
                        deadline=deadline_value,
                        caps=normalized_caps,
                    ):
                        certificate = None
                except Exception:
                    certificate = None
                if certificate is None:
                    status = "exact_replay_rejected"
                else:
                    status = "certified_conflict"
                    certified[pair_key] = certificate.certificate_sha256
                    certificates.append(certificate)
            elif outcome.raw_ray is not None:
                status = "candidate_error"
            elif status not in _VALID_RECORD_STATUSES:
                status = "candidate_error"
            records.append(
                _make_record(
                    pair=pair_key,
                    source_rows=source_rows,
                    source_digests=source_digests,
                    row_map_digest=row_map_digest,
                    status=status,
                    outcome=outcome,
                    raw_ray_nonzeros=raw_nonzeros,
                    certificate_sha256=(
                        None if certificate is None else certificate.certificate_sha256
                    ),
                )
            )

    terminal_parent = sparse_hz_semantic_digest(hz)
    if terminal_parent != parent_digest:
        raise PhaseConditionedPairInfeasibilityError(
            "parent_mutated_during_pair_candidate"
        )
    # One terminal live replay prevents a serializable bundle from becoming
    # authority after its source frame changes.
    for certificate in certificates:
        if not _replay_exact_certificate_on_precomputed_frame(
            hz,
            certificate,
            property_digest=selection.property_digest,
            parent_digest=parent_digest,
            source_frame_digest=source_frame_digest,
            deadline=deadline_value,
            caps=normalized_caps,
        ):
            raise PhaseConditionedPairInfeasibilityError(
                "terminal_exact_certificate_replay_failed"
            )
    coverage = _build_coverage(stable_ids, certified)
    records_tuple = tuple(records)
    certificates_tuple = tuple(certificates)
    status = (
        "complete"
        if all(record.status not in {"deadline_expired", "candidate_error", "model_close_failed"} for record in records_tuple)
        else "partial"
    )
    receipt = _make_receipt(
        stable_ids=stable_ids,
        records=records_tuple,
        coverage=coverage,
        parent_digest=parent_digest,
        terminal_parent_digest=terminal_parent,
        property_digest=selection.property_digest,
        selection_digest=selection.selection_digest,
        source_frame_digest=source_frame_digest,
        caps=normalized_caps,
    )
    placeholder = PairInfeasibilityBundle(
        status=status,
        stable_bit_ids=stable_ids,
        parent_semantic_digest=parent_digest,
        terminal_parent_semantic_digest=terminal_parent,
        property_digest=selection.property_digest,
        selection_digest=selection.selection_digest,
        operator_row_tag_digest=selection.operator_row_tag_digest,
        ordered_source_frame_sha256=source_frame_digest,
        caps=normalized_caps,
        records=records_tuple,
        certificates=certificates_tuple,
        coverage=coverage,
        receipt=receipt,
        bundle_sha256="",
        proof_authority=False,
    )
    return replace(
        placeholder,
        bundle_sha256=_canonical_sha256(
            _bundle_payload(placeholder, include_digest=False)
        ),
    )


def verify_phase_conditioned_pair_infeasibility_bundle(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    bundle: Any,
    *,
    deadline: float,
) -> bool:
    """Strictly verify structure, live source rows, certificates, and coverage."""

    try:
        if (
            type(bundle) is not PairInfeasibilityBundle
            or bundle.proof_authority is not False
            or bundle.status not in {"complete", "partial"}
            or type(bundle.receipt) is not dict
            or not _valid_sha256(bundle.bundle_sha256)
            or type(build) is not OperatorHZBuild
            or type(build.hz) is not SparseHZono
            or type(selection) is not OperatorExactReLUPhaseSelection
        ):
            return False
        deadline_value = float(deadline)
        remaining = deadline_value - time.monotonic()
        if (
            not math.isfinite(deadline_value)
            or remaining <= 0.0
            or selection.caps.timeout_seconds > remaining
        ):
            return False
        caps = _normalize_caps(bundle.caps)
        stable_ids = _stable_ids(bundle.stable_bit_ids, caps=caps)
        if stable_ids != bundle.stable_bit_ids:
            return False
        if not verify_operator_exact_relu_property_phase_selection(
            build,
            rivals,
            selection,
            **_selection_caps_kwargs(selection),
        ):
            return False
        _check_deadline(deadline_value, "bundle_after_selection_verification")
        hz = build.hz
        parent_digest = sparse_hz_semantic_digest(hz)
        if (
            parent_digest != bundle.parent_semantic_digest
            or parent_digest != bundle.terminal_parent_semantic_digest
            or bundle.property_digest != selection.property_digest
            or bundle.selection_digest != selection.selection_digest
            or bundle.operator_row_tag_digest != selection.operator_row_tag_digest
        ):
            return False
        source_frame = _ordered_source_frame_digest(
            hz, parent_digest=parent_digest, deadline=deadline_value
        )
        if source_frame != bundle.ordered_source_frame_sha256:
            return False
        mappings = {mapping.stable_bcol_id: mapping for mapping in selection.mappings}
        expected_query_keys = tuple(
            ((left, lp), (right, rp))
            for left, right in itertools.combinations(stable_ids, 2)
            for lp, rp in itertools.product((-1, 1), repeat=2)
        )
        if (
            type(bundle.records) is not tuple
            or tuple(record.pair for record in bundle.records) != expected_query_keys
            or len(bundle.records) > caps.max_signed_pair_queries
            or type(bundle.certificates) is not tuple
            or type(bundle.coverage) is not tuple
        ):
            return False
        certificate_by_sha = {}
        for certificate in bundle.certificates:
            if (
                type(certificate) is not ExactDualRayConflictCertificate
                or certificate.certificate_sha256 in certificate_by_sha
                or not _replay_exact_certificate_on_precomputed_frame(
                    hz,
                    certificate,
                    property_digest=selection.property_digest,
                    parent_digest=parent_digest,
                    source_frame_digest=source_frame,
                    deadline=deadline_value,
                    caps=caps,
                )
            ):
                return False
            certificate_by_sha[certificate.certificate_sha256] = certificate
        certified: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str] = {}
        for record in bundle.records:
            if (
                type(record) is not SignedPairRecord
                or record.proof_authority is not False
                or record.status not in _VALID_RECORD_STATUSES
                or record.solver_threads != 1
                or record.model_closed is not True
                or not _valid_sha256(record.record_sha256)
                or _canonical_sha256(
                    _record_payload(record, include_digest=False)
                ) != record.record_sha256
            ):
                return False
            left_id = record.pair[0][0]
            right_id = record.pair[1][0]
            rows, digests, row_map = _source_binding(
                hz,
                mappings[left_id],
                mappings[right_id],
                parent_digest=parent_digest,
                source_frame_digest=source_frame,
                deadline=deadline_value,
                caps=caps,
            )
            if (
                record.source_upper_rows != rows
                or record.source_row_sha256 != digests
                or record.local_row_map_sha256 != row_map
            ):
                return False
            if record.status == "certified_conflict":
                if (
                    record.certificate_sha256 not in certificate_by_sha
                    or record.local_model_rows != 6
                    or record.local_model_columns != hz.n_cont + hz.n_bin
                    or record.raw_ray_nonzero_rows < 1
                ):
                    return False
                certificate = certificate_by_sha[record.certificate_sha256]
                certificate_pair = tuple(
                    (literal.stable_bcol_id, literal.phase)
                    for literal in certificate.literals
                )
                if certificate_pair != record.pair:
                    return False
                certified[record.pair] = record.certificate_sha256
            elif record.certificate_sha256 is not None:
                return False
        if tuple(certificate_by_sha) != tuple(
            record.certificate_sha256
            for record in bundle.records
            if record.status == "certified_conflict"
        ):
            return False
        expected_coverage = _build_coverage(stable_ids, certified)
        if bundle.coverage != expected_coverage:
            return False
        expected_receipt = _make_receipt(
            stable_ids=stable_ids,
            records=bundle.records,
            coverage=bundle.coverage,
            parent_digest=parent_digest,
            terminal_parent_digest=parent_digest,
            property_digest=selection.property_digest,
            selection_digest=selection.selection_digest,
            source_frame_digest=source_frame,
            caps=caps,
        )
        if bundle.receipt != expected_receipt:
            return False
        expected_status = (
            "complete"
            if all(record.status not in {"deadline_expired", "candidate_error", "model_close_failed"} for record in bundle.records)
            else "partial"
        )
        if bundle.status != expected_status:
            return False
        return (
            _canonical_sha256(_bundle_payload(bundle, include_digest=False))
            == bundle.bundle_sha256
        )
    except (
        PhaseConditionedPairInfeasibilityError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
        TimeoutError,
    ):
        return False


__all__ = [
    "PairInfeasibilityBundle",
    "PairLocalCaps",
    "PatternCoverage",
    "PhaseConditionedPairInfeasibilityError",
    "SignedPairRecord",
    "run_phase_conditioned_pair_infeasibility_candidate",
    "verify_phase_conditioned_pair_infeasibility_bundle",
]
