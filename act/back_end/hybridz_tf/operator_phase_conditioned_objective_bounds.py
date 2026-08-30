#!/usr/bin/env python3
"""Toy-first replayable conditional objective bounds for PCOH.

This module deliberately has no verifier or pipeline integration.  For a
complete assignment of at most four *selected* exact-ReLU binaries it retains
only the three canonical Big-M upper rows identified by the already existing
live phase-selection verifier::

    relu_exact_lower, relu_exact_x_branch, relu_exact_zero_branch

All other upper rows and every equality row are omitted.  The resulting set
is therefore a relaxation of the live parent conditioned on the selected
binary signs.  A small local LP may propose row duals, but its status,
objective, and primal point have no authority.  The sole numerical authority
is ``_hz_independent_split_block_lp_lagrangian_upper``, replayed on the live
``c/Gc/Gb``, the copied local rows, and signed ``[-1,+1]`` factor bounds.

Two pattern-local bounds and one bundle-scoped bound are compared for every
pattern:

* a candidate-dual local-row bound, when a finite sign-legal dual exists;
* a zero-dual local-row/fixed-pattern cube bound; and
* a zero-dual global factor-cube baseline, independently checked once while
  preparing the immutable bundle context and then replay-identically reused.

The smallest stored outward binary64 bound is retained.  LP infeasibility is
never used to delete a pattern, and every candidate failure falls back to the
two zero-dual certificates.  A verifier re-runs the live selection verifier,
re-extracts every source row, reconstructs the exact stored-binary64
objective with :class:`fractions.Fraction`, and replays the checker.  The
bundle-scoped exact formation also seals a packed factor-objective envelope;
every pattern reuses that envelope without revisiting ``c/Gc/Gb/C``.  This is
an intentionally bounded toy-first implementation; it never loads or joins
the full parent constraint CSR and never calls sparse ``hstack``/``vstack``.
Complete production and replay share one immutable verified context and
perform one terminal live-parent semantic re-hash before returning a bundle
or constructing any external proof handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import itertools
import json
import math
from numbers import Integral, Real
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize as spo
import scipy.sparse as sp

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    ordered_property_digest,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    OperatorExactReLUPhaseMapping,
    OperatorExactReLUPhaseSelection,
    verify_operator_exact_relu_property_phase_selection,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    ExternalPatternUpperBound,
    ObjectiveBinding,
    bind_external_pattern_upper_bound,
    build_objective_binding,
    verify_objective_binding,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _HZPreformedFactorObjectiveEnvelope,
    _hz_form_exact_factor_objective_envelope_from_live_split_blocks,
    _hz_independent_split_block_lp_lagrangian_upper,
    _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope,
    _hz_read_exact_objective_binding_material_from_factor_envelope,
)


_CERTIFICATE_SCHEMA = (
    "act.operator_phase_conditioned_objective_bound.v2"
)
_RECEIPT_SCHEMA = (
    "act.operator_phase_conditioned_objective_bound_receipt.v2"
)
_ROW_SCHEMA = "act.operator_phase_conditioned_local_upper_row.v1"
_ROW_SET_SCHEMA = "act.operator_phase_conditioned_local_row_set.v1"
_PATTERN_SCHEMA = "act.operator_phase_conditioned_pattern.v1"
_VERIFIED_CONTEXT_SCHEMA = (
    "act.operator_phase_conditioned_verified_bundle_context.v2"
)
_DUAL_SCHEMA = "act.operator_phase_conditioned_candidate_dual.v1"
_CHECKER_BUNDLE_SCHEMA = (
    "act.operator_phase_conditioned_checker_bundle.v2"
)
_REPLAY_BUNDLE_SCHEMA = (
    "act.operator_phase_conditioned_complete_replay_bundle.v2"
)
_SCHEDULED_COMPLETE_SCHEMA = (
    "act.operator_phase_conditioned_scheduled_complete_bundle.v2"
)
_SCHEDULED_RECEIPT_SCHEMA = (
    "act.operator_phase_conditioned_scheduled_complete_receipt.v2"
)
_SCHEDULED_TELEMETRY_SCHEMA = (
    "act.operator_phase_conditioned_scheduled_complete_telemetry.v2"
)
_SCHEDULED_STOP_POLICY_SCHEMA = (
    "act.operator_phase_conditioned_scheduled_stop_policy.v2"
)
_SCHEDULED_STOP_SCHEMA = (
    "act.operator_phase_conditioned_scheduled_stop_record.v2"
)
_SPLIT_CHECKER_SCHEMA = (
    "hz_lp_lagrangian_preformed_objective_split_blocks_longdouble_v1"
)
_SPLIT_CHECKER_ROUTE = (
    "native_hz_preformed_objective_split_csr_no_generator_read_v1"
)

_DEFAULT_CERTIFICATE_TIMEOUT_SECONDS = 10.0
_DEFAULT_CANDIDATE_TIMEOUT_SECONDS = 1.0
_DEFAULT_MAX_CANDIDATE_ACTIVE_COLUMNS = 200_000
_DEFAULT_MAX_CANDIDATE_DENSE_ENTRIES = 2_000_000
_MAX_TIMEOUT_SECONDS = 60.0
_MAX_EXACT_OBJECTIVE_NONZEROS = 5_000_000

_SCHEDULED_CANDIDATE_STATUSES = frozenset(
    {
        "optimal",
        "deadline_fallback",
        "active_column_cap_fallback",
        "dense_entry_cap_fallback",
        "objective_conversion_fallback",
        "infeasible_no_authority_fallback",
        "nonoptimal_fallback",
        "missing_dual_fallback",
        "dual_shape_fallback",
        "solver_error_fallback",
        "candidate_nonfinite_dual_fallback",
        "candidate_illegal_dual_fallback",
        "candidate_malformed_fallback",
    }
)
_SCHEDULED_PATTERN_CALL_TRACE_KEYS = frozenset(
    {
        "pattern",
        "linprog_eligible",
        "linprog_called",
        "linprog_completed",
        "normalized_candidate_status",
        "candidate_dual_accepted",
    }
)
_SCHEDULED_TELEMETRY_KEYS = frozenset(
    {
        "schema",
        "status",
        "candidate_only",
        "full_parent_lp_called",
        "proof_authority",
        "verdict_authority",
        "stable_bits",
        "local_upper_rows",
        "expected_pattern_count",
        "patterns_started",
        "patterns_completed",
        "completed_patterns_in_execution_order",
        "observed_upper_exact_in_execution_order",
        "candidate_call_trace_in_execution_order",
        "context_formations",
        "exact_objective_expansions",
        "source_row_hash_passes",
        "candidate_proposal_invocations",
        "linprog_attempted",
        "linprog_actual_calls",
        "linprog_completed_calls",
        "candidate_statuses_in_execution_order",
        "candidate_status_counts",
        "candidate_dual_accepted",
        "split_checker_evaluations",
        "candidate_checker_evaluations",
        "zero_checker_evaluations",
        "global_checker_evaluations",
        "global_checker_cache_hits",
        "context_seconds",
        "pattern_seconds_in_execution_order",
        "total_seconds",
        "terminal_parent_seal_attempts",
        "terminal_parent_seal_completions",
        "actual_call_site_counters",
        "candidate_solver_status_authority",
        "candidate_solver_objective_authority",
        "global_checker_reused_per_pattern",
        "telemetry_sha256",
    }
)

SignedPattern = Tuple[int, ...]
PatternAssignment = Tuple[Tuple[int, int], ...]


class OperatorPhaseConditionedObjectiveBoundError(ValueError):
    """Malformed, stale, timed-out, or non-replayable conditional bound."""


@dataclass(frozen=True)
class OperatorPhaseConditionedObjectiveBoundCertificate:
    """One replayable finite upper bound for one complete signed pattern.

    The producer records checker replay material, but this object itself has
    no proof or verdict authority.  Authority is conferred only by the
    complete-cover live replay API below, which returns core
    :class:`ExternalPatternUpperBound` handles after all ``2**k`` patterns
    succeed.
    """

    schema: str
    parent_semantic_digest: str
    operator_row_tag_digest: str
    selection_digest: str
    property_digest: str
    rival_id: int
    rival_binding_digest: str
    objective_binding: ObjectiveBinding
    objective_envelope_sha256: str
    stable_bit_ids: Tuple[int, ...]
    pattern: SignedPattern
    assignments: PatternAssignment
    local_upper_row_ids: Tuple[int, ...]
    local_upper_row_sha256: Tuple[str, ...]
    omitted_upper_rows: int
    omitted_equality_rows: int
    local_row_set_sha256: str
    verified_context_sha256: str
    pattern_sha256: str
    dual_sha256: str
    raw_upper_dual: Tuple[float, ...]
    raw_equality_dual: Tuple[float, ...]
    candidate_status: str
    candidate_dual_accepted: bool
    candidate_checked_upper: Optional[float]
    zero_dual_fixed_upper: float
    global_cube_upper_exact: Fraction
    global_cube_upper: float
    selected_source: str
    upper_stored: float
    candidate_checker_sha256: Optional[str]
    zero_checker_sha256: str
    global_checker_sha256: str
    checker_bundle_sha256: str
    certificate_sha256: str
    receipt: Mapping[str, Any]
    proof_authority: bool = False


@dataclass(frozen=True)
class ReplayedOperatorPhaseConditionedObjectiveBounds:
    """Complete live-replayed ``2**k`` cover ready for the PCOH core."""

    schema: str
    parent_semantic_digest: str
    stable_bit_ids: Tuple[int, ...]
    objective_binding: ObjectiveBinding
    pattern_bounds: Tuple[ExternalPatternUpperBound, ...]
    baseline_upper_stored: float
    certificate_sha256: Tuple[str, ...]
    replay_bundle_sha256: str
    receipt: Mapping[str, Any]
    proof_authority: bool = True


@dataclass(frozen=True)
class OperatorPhaseConditionedScheduledStopPolicy:
    """Pure-data stop policy for the scheduled complete producer.

    Indices address positions in ``evaluation_schedule`` rather than the
    canonical pattern order.  An unconditional stop is useful for bounded
    toy sentinels.  A threshold stop compares the stored binary64 upper as an
    exact :class:`Fraction`; equality does not stop.
    """

    stop_after_pattern_indices: Tuple[int, ...] = ()
    strict_upper_threshold: Optional[Fraction] = None
    threshold_pattern_indices: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ScheduledOperatorPhaseConditionedObjectiveBounds:
    """Canonical complete cover produced in a caller-selected schedule.

    The certificates themselves remain non-authoritative.  As with the
    legacy complete producer, a later live replay is required before the
    PCOH core can mint external pattern-bound handles.
    """

    schema: str
    parent_semantic_digest: str
    stable_bit_ids: Tuple[int, ...]
    canonical_patterns: Tuple[SignedPattern, ...]
    evaluation_schedule: Tuple[SignedPattern, ...]
    stop_policy: OperatorPhaseConditionedScheduledStopPolicy
    certificates: Tuple[
        OperatorPhaseConditionedObjectiveBoundCertificate, ...
    ]
    telemetry: Mapping[str, Any]
    receipt: Mapping[str, Any]
    bundle_sha256: str
    full_parent_lp_called: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False


@dataclass(frozen=True)
class OperatorPhaseConditionedScheduledStopRecord:
    """Non-authoritative record for a policy stop with no partial cover."""

    schema: str
    status: str
    reason: str
    parent_semantic_digest: str
    stable_bit_ids: Tuple[int, ...]
    evaluation_schedule: Tuple[SignedPattern, ...]
    stop_policy: OperatorPhaseConditionedScheduledStopPolicy
    triggering_schedule_index: int
    triggering_pattern: SignedPattern
    completed_internal_pattern_count: int
    strict_upper_threshold: Optional[Fraction]
    observed_upper_exact: Optional[Fraction]
    telemetry: Mapping[str, Any]
    record_sha256: str
    partial_certificates_returned: bool = False
    external_pattern_bounds_bound: int = 0
    full_parent_lp_called: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False
    structural_self_consistency_only: bool = True
    provenance_authority: bool = False
    authenticity_authority: bool = False
    future_live_owner_anchor_required: bool = True


class OperatorPhaseConditionedScheduledStop(
    OperatorPhaseConditionedObjectiveBoundError
):
    """Raised after a sealed policy stop; ``record`` exposes no certificate."""

    def __init__(
        self, record: OperatorPhaseConditionedScheduledStopRecord
    ) -> None:
        if type(record) is not OperatorPhaseConditionedScheduledStopRecord:
            raise TypeError("scheduled_stop_record_wrong_type")
        self.record = record
        super().__init__(f"scheduled_non_authoritative_stop:{record.reason}")


@dataclass(frozen=True)
class _CandidateSolve:
    status: str
    raw_upper_dual: Tuple[float, ...] = ()
    raw_equality_dual: Tuple[float, ...] = ()


@dataclass
class _ScheduledBuildTelemetry:
    """Mutable counters used only at the actual scheduled execution sites."""

    context_formations: int = 0
    exact_objective_expansions: int = 0
    source_row_hash_passes: int = 0
    candidate_proposal_invocations: int = 0
    linprog_attempted: int = 0
    linprog_actual_calls: int = 0
    linprog_completed_calls: int = 0
    candidate_statuses: list[str] = field(default_factory=list)
    candidate_call_traces: list[
        Tuple[SignedPattern, bool, bool, bool, str, bool]
    ] = field(default_factory=list)
    candidate_dual_accepted: int = 0
    split_checker_evaluations: int = 0
    candidate_checker_evaluations: int = 0
    zero_checker_evaluations: int = 0
    global_checker_evaluations: int = 0
    global_checker_cache_hits: int = 0
    patterns_started: int = 0
    patterns_completed: int = 0
    completed_patterns: list[SignedPattern] = field(default_factory=list)
    observed_upper_exact: list[Fraction] = field(default_factory=list)
    pattern_seconds: list[float] = field(default_factory=list)
    context_seconds: float = 0.0
    total_seconds: float = 0.0
    terminal_parent_seal_attempts: int = 0
    terminal_parent_seal_completions: int = 0


@dataclass(frozen=True)
class _VerifiedBundleContext:
    build: OperatorHZBuild
    hz: SparseHZono
    rival: RivalSpec
    objective_binding: ObjectiveBinding
    objective_envelope: _HZPreformedFactorObjectiveEnvelope
    objective_formation_receipt: Mapping[str, Any]
    parent_semantic_digest: str
    property_digest: str
    stable_bit_ids: Tuple[int, ...]
    mappings: Tuple[OperatorExactReLUPhaseMapping, ...]
    local_upper_row_ids: Tuple[int, ...]
    local_upper_row_sha256: Tuple[str, ...]
    omitted_upper_rows: int
    omitted_equality_rows: int
    local_row_set_sha256: str
    verified_context_sha256: str
    local_Auc: sp.csr_matrix
    local_Aub: sp.csr_matrix
    local_ub: np.ndarray
    empty_Ac: sp.csr_matrix
    empty_Ab: sp.csr_matrix
    empty_b: np.ndarray
    continuous_lb: np.ndarray
    continuous_ub: np.ndarray
    binary_cube_lb: np.ndarray
    binary_cube_ub: np.ndarray
    candidate_continuous_positions: np.ndarray
    candidate_continuous_q: np.ndarray
    candidate_binary_positions: np.ndarray
    candidate_binary_q: np.ndarray
    global_cube_upper_exact: Fraction
    global_cube_upper: float
    global_checker_sha256: str


@dataclass(frozen=True)
class _LiveContext:
    shared: _VerifiedBundleContext
    build: OperatorHZBuild
    hz: SparseHZono
    rival: RivalSpec
    objective_binding: ObjectiveBinding
    objective_envelope: _HZPreformedFactorObjectiveEnvelope
    objective_formation_receipt: Mapping[str, Any]
    parent_semantic_digest: str
    property_digest: str
    stable_bit_ids: Tuple[int, ...]
    pattern: SignedPattern
    assignments: PatternAssignment
    mappings: Tuple[OperatorExactReLUPhaseMapping, ...]
    local_upper_row_ids: Tuple[int, ...]
    local_upper_row_sha256: Tuple[str, ...]
    omitted_upper_rows: int
    omitted_equality_rows: int
    local_row_set_sha256: str
    verified_context_sha256: str
    pattern_sha256: str
    local_Auc: sp.csr_matrix
    local_Aub: sp.csr_matrix
    local_ub: np.ndarray
    empty_Ac: sp.csr_matrix
    empty_Ab: sp.csr_matrix
    empty_b: np.ndarray
    continuous_lb: np.ndarray
    continuous_ub: np.ndarray
    binary_lb: np.ndarray
    binary_ub: np.ndarray
    candidate_continuous_positions: np.ndarray
    candidate_continuous_q: np.ndarray
    candidate_binary_positions: np.ndarray
    candidate_binary_q: np.ndarray
    global_cube_upper_exact: Fraction
    global_cube_upper: float
    global_checker_sha256: str


def _canonical_form(value: Any) -> Any:
    """Convert a strict built-in payload to deterministic JSON primitives."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise OperatorPhaseConditionedObjectiveBoundError(
                "canonical_payload_nonfinite_float"
            )
        return {"__binary64_hex__": value.hex()}
    if type(value) is Fraction:
        return {
            "__fraction__": [value.numerator, value.denominator]
        }
    if isinstance(value, np.generic):
        return _canonical_form(value.item())
    if type(value) in {tuple, list}:
        return [_canonical_form(item) for item in value]
    if isinstance(value, (dict, MappingProxyType)):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise OperatorPhaseConditionedObjectiveBoundError(
                    "canonical_payload_nonstring_key"
                )
            result[key] = _canonical_form(item)
        return result
    raise OperatorPhaseConditionedObjectiveBoundError(
        f"canonical_payload_unsupported_{type(value).__name__}"
    )


def _canonical_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            _canonical_form(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "canonical_payload_encoding_failed"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze receipt payloads without changing scalar values."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) in {tuple, list}:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _is_strict_deep_frozen_canonical_payload(value: Any) -> bool:
    """Accept only the immutable tree shape emitted by ``_deep_freeze``."""

    if value is None or type(value) in {str, bool, int, Fraction}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is tuple:
        return all(
            _is_strict_deep_frozen_canonical_payload(item)
            for item in value
        )
    if type(value) is MappingProxyType:
        return all(
            type(key) is str
            and _is_strict_deep_frozen_canonical_payload(item)
            for key, item in value.items()
        )
    return False


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_timeout(value: Any, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"{name}_not_builtin_numeric"
        )
    result = float(value)
    if (
        not math.isfinite(result)
        or result <= 0.0
        or result > _MAX_TIMEOUT_SECONDS
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"{name}_out_of_range"
        )
    return result


def _strict_positive_cap(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"{name}_invalid"
        )
    return value


def _check_deadline(deadline: float, *, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"certificate_deadline_exhausted_{stage}"
        )


def _strict_stable_bit_ids(values: Any) -> Tuple[int, ...]:
    if type(values) is not tuple or not 1 <= len(values) <= 4:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "stable_bit_ids_must_be_builtin_tuple_k1_to_k4"
        )
    result = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
        ):
            raise OperatorPhaseConditionedObjectiveBoundError(
                "stable_bit_id_noncanonical"
            )
        result.append(int(value))
    canonical = tuple(result)
    if tuple(sorted(canonical)) != canonical or len(set(canonical)) != len(
        canonical
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "stable_bit_ids_not_strictly_sorted_unique"
        )
    return canonical


def _strict_pattern(value: Any, *, k: int) -> SignedPattern:
    if type(value) is not tuple or len(value) != k:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "pattern_not_complete_builtin_tuple"
        )
    result = []
    for phase in value:
        if (
            isinstance(phase, bool)
            or not isinstance(phase, Integral)
            or int(phase) not in {-1, 1}
        ):
            raise OperatorPhaseConditionedObjectiveBoundError(
                "pattern_phase_not_signed_integer"
            )
        result.append(int(phase))
    return tuple(result)


def _canonical_patterns(k: int) -> Tuple[SignedPattern, ...]:
    return tuple(
        tuple(int(value) for value in pattern)
        for pattern in itertools.product((-1, 1), repeat=k)
    )


def _strict_evaluation_schedule(
    value: Any,
    *,
    k: int,
) -> Tuple[SignedPattern, ...]:
    """Accept exactly one permutation of the complete canonical cover."""

    if type(value) is not tuple:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_evaluation_schedule_not_builtin_tuple"
        )
    expected = _canonical_patterns(k)
    if len(value) != len(expected):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_evaluation_schedule_wrong_length"
        )
    schedule = tuple(_strict_pattern(item, k=k) for item in value)
    if len(set(schedule)) != len(schedule) or set(schedule) != set(expected):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_evaluation_schedule_not_complete_permutation"
        )
    return schedule


def _strict_schedule_indices(
    value: Any,
    *,
    pattern_count: int,
    name: str,
) -> Tuple[int, ...]:
    if type(value) is not tuple:
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"{name}_not_builtin_tuple"
        )
    if any(type(item) is not int for item in value):
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"{name}_contains_non_builtin_int"
        )
    result = tuple(value)
    if (
        result != tuple(sorted(result))
        or len(set(result)) != len(result)
        or any(item < 0 or item >= pattern_count for item in result)
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"{name}_not_canonical_or_out_of_range"
        )
    return result


def _strict_scheduled_stop_policy(
    value: Any,
    *,
    pattern_count: int,
) -> OperatorPhaseConditionedScheduledStopPolicy:
    if type(value) is not OperatorPhaseConditionedScheduledStopPolicy:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_stop_policy_wrong_type"
        )
    unconditional = _strict_schedule_indices(
        value.stop_after_pattern_indices,
        pattern_count=pattern_count,
        name="stop_after_pattern_indices",
    )
    threshold_indices = _strict_schedule_indices(
        value.threshold_pattern_indices,
        pattern_count=pattern_count,
        name="threshold_pattern_indices",
    )
    if set(unconditional).intersection(threshold_indices):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_stop_policy_index_sets_overlap"
        )
    threshold = value.strict_upper_threshold
    if threshold is not None:
        if type(threshold) is not Fraction:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "strict_upper_threshold_not_exact_fraction"
            )
        if (
            threshold.numerator.bit_length() > 4096
            or threshold.denominator.bit_length() > 4096
        ):
            raise OperatorPhaseConditionedObjectiveBoundError(
                "strict_upper_threshold_exact_bits_exceeded"
            )
        if not threshold_indices:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "strict_upper_threshold_has_no_pattern_indices"
            )
    elif threshold_indices:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "threshold_pattern_indices_without_threshold"
        )
    return OperatorPhaseConditionedScheduledStopPolicy(
        stop_after_pattern_indices=unconditional,
        strict_upper_threshold=threshold,
        threshold_pattern_indices=threshold_indices,
    )


def _scheduled_trace_control_flow_allowed(
    *,
    status: str,
    eligible: bool,
    called: bool,
    completed: bool,
) -> bool:
    """Replay the only LP call-state combinations reachable by the solver."""

    if status not in _SCHEDULED_CANDIDATE_STATUSES:
        return False
    state = (eligible, called, completed)
    allowed = {
        "optimal": {(False, False, False), (True, True, True)},
        "deadline_fallback": {
            (False, False, False),
            (True, False, False),
            (True, True, True),
        },
        "active_column_cap_fallback": {(False, False, False)},
        "dense_entry_cap_fallback": {(False, False, False)},
        "objective_conversion_fallback": {(True, False, False)},
        "infeasible_no_authority_fallback": {(True, True, True)},
        "nonoptimal_fallback": {(True, True, True)},
        "missing_dual_fallback": {(True, True, True)},
        "dual_shape_fallback": {(True, True, True)},
        "solver_error_fallback": {
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        },
        "candidate_nonfinite_dual_fallback": {
            (True, True, True),
        },
        "candidate_illegal_dual_fallback": {
            (True, True, True),
        },
        # The normalization guard can name this status, but the concrete
        # private solver always returns a well-formed ``_CandidateSolve``.
        # A scheduled producer therefore fails closed instead of inventing a
        # call provenance for an unreachable malformed return.
        "candidate_malformed_fallback": set(),
    }
    return state in allowed[status]


def _scheduled_stop_policy_payload(
    policy: OperatorPhaseConditionedScheduledStopPolicy,
    *,
    pattern_count: int,
    include_digest: bool,
) -> Dict[str, Any]:
    """Canonical pure-data policy payload bound into scheduled receipts."""

    checked = _strict_scheduled_stop_policy(
        policy, pattern_count=pattern_count
    )
    payload: Dict[str, Any] = {
        "schema": _SCHEDULED_STOP_POLICY_SCHEMA,
        "stop_after_pattern_indices": (
            checked.stop_after_pattern_indices
        ),
        "strict_upper_threshold": checked.strict_upper_threshold,
        "threshold_pattern_indices": checked.threshold_pattern_indices,
    }
    if include_digest:
        payload["policy_sha256"] = _canonical_sha256(payload)
    return payload


def _scheduled_stop_policy_mapping(
    policy: OperatorPhaseConditionedScheduledStopPolicy,
    *,
    pattern_count: int,
) -> Mapping[str, Any]:
    frozen = _deep_freeze(
        _scheduled_stop_policy_payload(
            policy,
            pattern_count=pattern_count,
            include_digest=True,
        )
    )
    if type(frozen) is not MappingProxyType:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_stop_policy_freeze_failed"
        )
    return frozen


def _strict_live_float64_csr(
    matrix: Any,
    *,
    shape: Tuple[int, int],
    name: str,
) -> sp.csr_matrix:
    if (
        type(matrix) is not sp.csr_matrix
        or matrix.dtype != np.dtype(np.float64)
        or matrix.shape != shape
        or not matrix.has_canonical_format
        or np.asarray(matrix.indptr).ndim != 1
        or np.asarray(matrix.indices).ndim != 1
        or np.asarray(matrix.data).ndim != 1
        or int(matrix.indptr.size) != shape[0] + 1
        or int(matrix.indices.size) != int(matrix.data.size)
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"{name}_not_canonical_binary64_csr"
        )
    for start in range(0, int(matrix.nnz), 65536):
        if not np.all(np.isfinite(matrix.data[start : start + 65536])):
            raise OperatorPhaseConditionedObjectiveBoundError(
                f"{name}_nonfinite"
            )
    return matrix


def _extract_rows_without_stack(
    matrix: sp.csr_matrix,
    row_ids: Tuple[int, ...],
) -> sp.csr_matrix:
    """Copy only named CSR row segments; never join parent row blocks."""

    data_parts = []
    index_parts = []
    indptr = np.zeros(len(row_ids) + 1, dtype=matrix.indptr.dtype)
    cursor = 0
    for offset, row in enumerate(row_ids):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        data_parts.append(matrix.data[start:stop].copy())
        index_parts.append(matrix.indices[start:stop].copy())
        cursor += stop - start
        indptr[offset + 1] = cursor
    if cursor:
        data = np.concatenate(data_parts)
        indices = np.concatenate(index_parts)
    else:
        data = np.empty(0, dtype=np.float64)
        indices = np.empty(0, dtype=matrix.indices.dtype)
    result = sp.csr_matrix(
        (data, indices, indptr),
        shape=(len(row_ids), matrix.shape[1]),
        copy=False,
    )
    if not result.has_canonical_format:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "local_row_copy_lost_canonical_format"
        )
    return result


def _row_payload(
    *,
    row_id: int,
    Auc: sp.csr_matrix,
    Aub: sp.csr_matrix,
    ub: np.ndarray,
) -> Dict[str, Any]:
    c_start = int(Auc.indptr[row_id])
    c_stop = int(Auc.indptr[row_id + 1])
    b_start = int(Aub.indptr[row_id])
    b_stop = int(Aub.indptr[row_id + 1])
    return {
        "schema": _ROW_SCHEMA,
        "source_upper_row_id": row_id,
        "continuous_indices": tuple(
            int(value) for value in Auc.indices[c_start:c_stop]
        ),
        "continuous_values_hex": tuple(
            float(value).hex() for value in Auc.data[c_start:c_stop]
        ),
        "binary_indices": tuple(
            int(value) for value in Aub.indices[b_start:b_stop]
        ),
        "binary_values_hex": tuple(
            float(value).hex() for value in Aub.data[b_start:b_stop]
        ),
        "upper_bound_hex": float(ub[row_id]).hex(),
    }


def _exact_factor_terms(
    *,
    matrix: sp.csr_matrix,
    stable_ids: np.ndarray,
    output_weights: Tuple[float, ...],
    name: str,
) -> Tuple[Tuple[int, Fraction], ...]:
    if (
        type(stable_ids) is not np.ndarray
        or stable_ids.dtype != np.dtype(np.int64)
        or stable_ids.ndim != 1
        or stable_ids.size != matrix.shape[1]
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            f"{name}_stable_ids_malformed"
        )
    accumulated: Dict[int, Fraction] = {}
    visited = 0
    for output_row, raw_weight in enumerate(output_weights):
        weight = Fraction.from_float(raw_weight)
        if weight == 0:
            continue
        start = int(matrix.indptr[output_row])
        stop = int(matrix.indptr[output_row + 1])
        visited += stop - start
        if visited > _MAX_EXACT_OBJECTIVE_NONZEROS:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "exact_objective_nonzero_cap_exceeded"
            )
        for offset in range(start, stop):
            position = int(matrix.indices[offset])
            coefficient = weight * Fraction.from_float(
                float(matrix.data[offset])
            )
            accumulated[position] = (
                accumulated.get(position, Fraction(0)) + coefficient
            )
    terms = []
    for position in sorted(accumulated):
        coefficient = accumulated[position]
        if coefficient != 0:
            terms.append((int(stable_ids[position]), coefficient))
    return tuple(terms)


def _build_exact_objective_binding(
    *,
    hz: SparseHZono,
    rival: RivalSpec,
    parent_semantic_digest: str,
) -> ObjectiveBinding:
    weights = tuple(rival.objective)
    center = -Fraction.from_float(rival.threshold)
    for weight, value in zip(weights, hz.c.tolist()):
        center += Fraction.from_float(weight) * Fraction.from_float(
            float(value)
        )
    continuous_terms = _exact_factor_terms(
        matrix=hz.Gc,
        stable_ids=hz.col_ids,
        output_weights=weights,
        name="continuous_objective",
    )
    binary_terms = _exact_factor_terms(
        matrix=hz.Gb,
        stable_ids=hz.bcol_ids,
        output_weights=weights,
        name="binary_objective",
    )
    binding = build_objective_binding(
        objective_id=(
            f"rival:{rival.rival_id}:{rival.binding_digest}"
        ),
        parent_semantic_digest=parent_semantic_digest,
        center=center,
        continuous_terms=continuous_terms,
        binary_terms=binary_terms,
    )
    if not verify_objective_binding(binding):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "objective_binding_self_verification_failed"
        )
    return binding


def _freeze_owned_csr(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Freeze the three owned CSR buffers of one local-only matrix."""

    for value in (matrix.data, matrix.indices, matrix.indptr):
        value.setflags(write=False)
    return matrix


def _prepare_verified_bundle_context(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    deadline: float,
    telemetry: Optional[_ScheduledBuildTelemetry] = None,
) -> _VerifiedBundleContext:
    """Perform the expensive live verification and source binding once."""

    if telemetry is not None:
        if type(telemetry) is not _ScheduledBuildTelemetry:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_telemetry_wrong_type"
            )
        telemetry.context_formations += 1

    if (
        type(build) is not OperatorHZBuild
        or type(build.hz) is not SparseHZono
        or type(selection) is not OperatorExactReLUPhaseSelection
        or type(rivals) is not tuple
        or type(focused_rival_id) is not int
        or focused_rival_id < 0
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "live_input_top_level_noncanonical"
        )
    stable_ids = _strict_stable_bit_ids(stable_bit_ids)
    try:
        caps = selection.caps
        live_selection_ok = (
            verify_operator_exact_relu_property_phase_selection(
                build,
                rivals,
                selection,
                max_rivals=caps.max_rivals,
                max_binaries=caps.max_binaries,
                max_work_items=caps.max_work_items,
                timeout_seconds=caps.timeout_seconds,
            )
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        live_selection_ok = False
    if not live_selection_ok:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "live_phase_selection_verification_failed"
        )
    _check_deadline(deadline, stage="selection_verification")

    hz = build.hz
    try:
        parent_digest = sparse_hz_semantic_digest(hz)
        property_digest = ordered_property_digest(rivals)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "live_digest_recomputation_failed"
        ) from exc
    if (
        parent_digest != selection.parent_semantic_digest
        or property_digest != selection.property_digest
        or not _valid_sha256(selection.operator_row_tag_digest)
        or not _valid_sha256(selection.selection_digest)
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "selection_live_binding_mismatch"
        )
    focused = tuple(
        rival for rival in rivals if rival.rival_id == focused_rival_id
    )
    if len(focused) != 1:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "focused_rival_not_unique_in_verified_batch"
        )
    rival = focused[0]

    _strict_live_float64_csr(
        hz.Gc, shape=(hz.n_out, hz.n_cont), name="Gc"
    )
    _strict_live_float64_csr(
        hz.Gb, shape=(hz.n_out, hz.n_bin), name="Gb"
    )
    _strict_live_float64_csr(
        hz.Auc, shape=(hz.n_ub, hz.n_cont), name="Auc"
    )
    _strict_live_float64_csr(
        hz.Aub, shape=(hz.n_ub, hz.n_bin), name="Aub"
    )
    if (
        type(hz.ub) is not np.ndarray
        or hz.ub.dtype != np.dtype(np.float64)
        or hz.ub.ndim != 1
        or hz.ub.size != hz.n_ub
        or not np.all(np.isfinite(hz.ub))
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "upper_bounds_not_canonical_binary64"
        )

    by_id = {mapping.stable_bcol_id: mapping for mapping in selection.mappings}
    if len(by_id) != len(selection.mappings):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "selection_mapping_ids_not_unique"
        )
    try:
        mappings = tuple(by_id[stable_id] for stable_id in stable_ids)
    except KeyError as exc:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "selected_stable_bit_missing_mapping"
        ) from exc
    if any(
        type(mapping) is not OperatorExactReLUPhaseMapping
        for mapping in mappings
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "selected_mapping_wrong_type"
        )

    row_ids = tuple(
        row
        for mapping in mappings
        for row in (
            mapping.lower_upper_row,
            mapping.x_branch_upper_row,
            mapping.zero_branch_upper_row,
        )
    )
    if (
        len(row_ids) != 3 * len(stable_ids)
        or len(row_ids) > 12
        or len(set(row_ids)) != len(row_ids)
        or any(type(row) is not int or row < 0 or row >= hz.n_ub for row in row_ids)
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "local_upper_row_ids_invalid_or_reused"
        )
    if telemetry is not None:
        telemetry.source_row_hash_passes += 1
    row_hashes = tuple(
        _canonical_sha256(
            _row_payload(
                row_id=row,
                Auc=hz.Auc,
                Aub=hz.Aub,
                ub=hz.ub,
            )
        )
        for row in row_ids
    )
    row_set_sha = _canonical_sha256(
        {
            "schema": _ROW_SET_SCHEMA,
            "parent_semantic_digest": parent_digest,
            "operator_row_tag_digest": selection.operator_row_tag_digest,
            "selection_digest": selection.selection_digest,
            "stable_bit_ids": stable_ids,
            "source_upper_row_ids": row_ids,
            "source_upper_row_sha256": row_hashes,
            "retained_upper_rows": len(row_ids),
            "omitted_upper_rows": hz.n_ub - len(row_ids),
            "retained_equality_rows": 0,
            "omitted_equality_rows": hz.n_eq,
            "objective_equality_substitution": False,
        }
    )
    local_Auc = _freeze_owned_csr(
        _extract_rows_without_stack(hz.Auc, row_ids)
    )
    local_Aub = _freeze_owned_csr(
        _extract_rows_without_stack(hz.Aub, row_ids)
    )
    local_ub = np.asarray(hz.ub[list(row_ids)], dtype=np.float64).copy()
    local_ub.setflags(write=False)

    objective_id = f"rival:{rival.rival_id}:{rival.binding_digest}"
    if telemetry is not None:
        telemetry.exact_objective_expansions += 1
    objective_envelope, objective_formation_receipt = (
        _hz_form_exact_factor_objective_envelope_from_live_split_blocks(
            c=hz.c,
            Gc=hz.Gc,
            Gb=hz.Gb,
            C_row=np.asarray(rival.objective, dtype=np.float64),
            threshold=rival.threshold,
            continuous_col_ids=hz.col_ids,
            binary_col_ids=hz.bcol_ids,
            objective_id=objective_id,
            parent_semantic_digest=parent_digest,
            deadline=deadline,
        )
    )
    if (
        type(objective_envelope)
        is not _HZPreformedFactorObjectiveEnvelope
        or not isinstance(objective_formation_receipt, Mapping)
        or objective_formation_receipt.get("status") != "formed"
        or objective_formation_receipt.get("proof_authority") is not False
        or objective_formation_receipt.get("verdict_authority") is not False
        or objective_formation_receipt.get("pcoh_authorization") is not False
        or objective_formation_receipt.get("production_ready") is not False
        or objective_formation_receipt.get("objective_expansion_count") != 1
        or objective_formation_receipt.get("exact_expansion_pass_count") != 1
        or objective_formation_receipt.get("source_hash_pass_count") != 1
        or objective_formation_receipt.get("generator_validation_pass_count")
        != 1
        or objective_formation_receipt.get("parent_semantic_digest")
        != parent_digest
        or objective_formation_receipt.get("objective_id") != objective_id
        or objective_formation_receipt.get("envelope_sha256")
        != objective_envelope.envelope_sha256
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "exact_objective_envelope_formation_failed"
        )
    try:
        (
            objective_center,
            objective_continuous_terms,
            objective_binary_terms,
            sealed_binding_sha,
        ) = _hz_read_exact_objective_binding_material_from_factor_envelope(
            objective_envelope,
            expected_parent_semantic_digest=parent_digest,
            expected_objective_id=objective_id,
        )
        objective_binding = build_objective_binding(
            objective_id=objective_id,
            parent_semantic_digest=parent_digest,
            center=objective_center,
            continuous_terms=objective_continuous_terms,
            binary_terms=objective_binary_terms,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "exact_objective_binding_material_failed"
        ) from exc
    if (
        not verify_objective_binding(objective_binding)
        or objective_binding.objective_binding_sha256 != sealed_binding_sha
        or objective_envelope.objective_binding_sha256 != sealed_binding_sha
        or objective_formation_receipt.get("objective_binding_sha256")
        != sealed_binding_sha
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "exact_objective_binding_cross_check_failed"
        )
    _check_deadline(deadline, stage="exact_objective_formation")
    global_exact = objective_binding.center + sum(
        (
            abs(coefficient)
            for _, coefficient in (
                objective_binding.continuous_terms
                + objective_binding.binary_terms
            )
        ),
        Fraction(0),
    )
    continuous_lb = np.full(hz.n_cont, -1.0, dtype=np.float64)
    continuous_ub = np.full(hz.n_cont, 1.0, dtype=np.float64)
    binary_cube_lb = np.full(hz.n_bin, -1.0, dtype=np.float64)
    binary_cube_ub = np.full(hz.n_bin, 1.0, dtype=np.float64)
    for value in (
        continuous_lb,
        continuous_ub,
        binary_cube_lb,
        binary_cube_ub,
    ):
        value.setflags(write=False)
    candidate_continuous_positions = np.flatnonzero(
        objective_envelope.q_continuous_hat != 0.0
    ).astype(np.int64, copy=False)
    candidate_continuous_q = np.asarray(
        objective_envelope.q_continuous_hat[
            candidate_continuous_positions
        ],
        dtype=np.float64,
    )
    candidate_binary_positions = np.flatnonzero(
        objective_envelope.q_binary_hat != 0.0
    ).astype(np.int64, copy=False)
    candidate_binary_q = np.asarray(
        objective_envelope.q_binary_hat[candidate_binary_positions],
        dtype=np.float64,
    )
    for value in (
        candidate_continuous_positions,
        candidate_continuous_q,
        candidate_binary_positions,
        candidate_binary_q,
    ):
        value.setflags(write=False)
    empty_Ac = _freeze_owned_csr(
        sp.csr_matrix((0, hz.n_cont), dtype=np.float64)
    )
    empty_Ab = _freeze_owned_csr(
        sp.csr_matrix((0, hz.n_bin), dtype=np.float64)
    )
    empty_b = np.empty(0, dtype=np.float64)
    empty_b.setflags(write=False)
    global_cube_upper, global_checker_sha256 = (
        _run_preformed_checker_blocks(
            objective_envelope=objective_envelope,
            objective_binding=objective_binding,
            parent_semantic_digest=parent_digest,
            Auc=empty_Ac,
            Aub=empty_Ab,
            Ac=empty_Ac,
            Ab=empty_Ab,
            ub=empty_b,
            b=empty_b,
            continuous_lb=continuous_lb,
            continuous_ub=continuous_ub,
            binary_lb=binary_cube_lb,
            binary_ub=binary_cube_ub,
            raw_upper_dual=(),
            deadline=deadline,
            telemetry=telemetry,
            checker_kind="global",
        )
    )
    if Fraction.from_float(global_cube_upper) < global_exact:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "shared_global_checker_bound_below_exact_fraction_cube"
        )
    verified_context_sha = _canonical_sha256(
        {
            "schema": _VERIFIED_CONTEXT_SCHEMA,
            "parent_semantic_digest": parent_digest,
            "property_digest": property_digest,
            "selection_digest": selection.selection_digest,
            "operator_row_tag_digest": selection.operator_row_tag_digest,
            "focused_rival_id": rival.rival_id,
            "focused_rival_binding_digest": rival.binding_digest,
            "objective_binding_sha256": (
                objective_binding.objective_binding_sha256
            ),
            "objective_envelope_sha256": (
                objective_envelope.envelope_sha256
            ),
            "objective_source_sha256": (
                objective_envelope.objective_source_sha256
            ),
            "objective_stable_ids_sha256": (
                objective_envelope.stable_ids_sha256
            ),
            "global_cube_upper": global_cube_upper,
            "global_checker_sha256": global_checker_sha256,
            "global_checker_bundle_evaluations": 1,
            "stable_bit_ids": stable_ids,
            "local_upper_row_ids": row_ids,
            "local_upper_row_sha256": row_hashes,
            "local_row_set_sha256": row_set_sha,
            "local_Auc_shape": tuple(int(x) for x in local_Auc.shape),
            "local_Aub_shape": tuple(int(x) for x in local_Aub.shape),
            "local_Auc_nnz": int(local_Auc.nnz),
            "local_Aub_nnz": int(local_Aub.nnz),
            "retained_equality_rows": 0,
            "objective_equality_substitution": False,
        }
    )
    _check_deadline(deadline, stage="verified_context_completion")
    return _VerifiedBundleContext(
        build=build,
        hz=hz,
        rival=rival,
        objective_binding=objective_binding,
        objective_envelope=objective_envelope,
        objective_formation_receipt=objective_formation_receipt,
        parent_semantic_digest=parent_digest,
        property_digest=property_digest,
        stable_bit_ids=stable_ids,
        mappings=mappings,
        local_upper_row_ids=row_ids,
        local_upper_row_sha256=row_hashes,
        omitted_upper_rows=hz.n_ub - len(row_ids),
        omitted_equality_rows=hz.n_eq,
        local_row_set_sha256=row_set_sha,
        verified_context_sha256=verified_context_sha,
        local_Auc=local_Auc,
        local_Aub=local_Aub,
        local_ub=local_ub,
        empty_Ac=empty_Ac,
        empty_Ab=empty_Ab,
        empty_b=empty_b,
        continuous_lb=continuous_lb,
        continuous_ub=continuous_ub,
        binary_cube_lb=binary_cube_lb,
        binary_cube_ub=binary_cube_ub,
        candidate_continuous_positions=candidate_continuous_positions,
        candidate_continuous_q=candidate_continuous_q,
        candidate_binary_positions=candidate_binary_positions,
        candidate_binary_q=candidate_binary_q,
        global_cube_upper_exact=global_exact,
        global_cube_upper=global_cube_upper,
        global_checker_sha256=global_checker_sha256,
    )


def _materialize_pattern_context(
    shared: _VerifiedBundleContext,
    pattern: SignedPattern,
    *,
    deadline: float,
) -> _LiveContext:
    """Create only signed bounds and pattern binding from a verified context."""

    if type(shared) is not _VerifiedBundleContext:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "verified_bundle_context_wrong_type"
        )
    signed_pattern = _strict_pattern(
        pattern, k=len(shared.stable_bit_ids)
    )
    binary_lb = shared.binary_cube_lb.copy()
    binary_ub = shared.binary_cube_ub.copy()
    for mapping, phase in zip(shared.mappings, signed_pattern):
        binary_lb[mapping.binary_position] = float(phase)
        binary_ub[mapping.binary_position] = float(phase)
    binary_lb.setflags(write=False)
    binary_ub.setflags(write=False)
    assignments = tuple(
        (stable_id, phase)
        for stable_id, phase in zip(
            shared.stable_bit_ids, signed_pattern
        )
    )
    pattern_sha = _canonical_sha256(
        {
            "schema": _PATTERN_SCHEMA,
            "verified_context_sha256": shared.verified_context_sha256,
            "parent_semantic_digest": shared.parent_semantic_digest,
            "stable_bit_ids": shared.stable_bit_ids,
            "pattern": signed_pattern,
        }
    )
    _check_deadline(deadline, stage="pattern_context_materialization")
    return _LiveContext(
        shared=shared,
        build=shared.build,
        hz=shared.hz,
        rival=shared.rival,
        objective_binding=shared.objective_binding,
        objective_envelope=shared.objective_envelope,
        objective_formation_receipt=shared.objective_formation_receipt,
        parent_semantic_digest=shared.parent_semantic_digest,
        property_digest=shared.property_digest,
        stable_bit_ids=shared.stable_bit_ids,
        pattern=signed_pattern,
        assignments=assignments,
        mappings=shared.mappings,
        local_upper_row_ids=shared.local_upper_row_ids,
        local_upper_row_sha256=shared.local_upper_row_sha256,
        omitted_upper_rows=shared.omitted_upper_rows,
        omitted_equality_rows=shared.omitted_equality_rows,
        local_row_set_sha256=shared.local_row_set_sha256,
        verified_context_sha256=shared.verified_context_sha256,
        pattern_sha256=pattern_sha,
        local_Auc=shared.local_Auc,
        local_Aub=shared.local_Aub,
        local_ub=shared.local_ub,
        empty_Ac=shared.empty_Ac,
        empty_Ab=shared.empty_Ab,
        empty_b=shared.empty_b,
        continuous_lb=shared.continuous_lb,
        continuous_ub=shared.continuous_ub,
        binary_lb=binary_lb,
        binary_ub=binary_ub,
        candidate_continuous_positions=(
            shared.candidate_continuous_positions
        ),
        candidate_continuous_q=shared.candidate_continuous_q,
        candidate_binary_positions=shared.candidate_binary_positions,
        candidate_binary_q=shared.candidate_binary_q,
        global_cube_upper_exact=shared.global_cube_upper_exact,
        global_cube_upper=shared.global_cube_upper,
        global_checker_sha256=shared.global_checker_sha256,
    )


def _terminal_parent_seal(
    shared: _VerifiedBundleContext,
    *,
    deadline: float,
    telemetry: Optional[_ScheduledBuildTelemetry] = None,
) -> str:
    """Re-hash the live parent once before exposing a completed result."""

    if type(shared) is not _VerifiedBundleContext:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "terminal_parent_seal_context_wrong_type"
        )
    if telemetry is not None:
        if type(telemetry) is not _ScheduledBuildTelemetry:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_telemetry_wrong_type"
            )
        telemetry.terminal_parent_seal_attempts += 1
    _check_deadline(deadline, stage="before_terminal_parent_semantic_digest")
    if shared.build.hz is not shared.hz:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "terminal_parent_object_identity_mismatch"
        )
    try:
        terminal_digest = sparse_hz_semantic_digest(shared.build.hz)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "terminal_parent_semantic_digest_failed"
        ) from exc
    _check_deadline(deadline, stage="after_terminal_parent_semantic_digest")
    if terminal_digest != shared.parent_semantic_digest:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "terminal_parent_semantic_digest_mismatch"
        )
    if telemetry is not None:
        telemetry.terminal_parent_seal_completions += 1
    return terminal_digest


def _prepare_live_context(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    pattern: SignedPattern,
    deadline: float,
) -> _LiveContext:
    """Compatibility path for the existing single-pattern public APIs."""

    shared = _prepare_verified_bundle_context(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_bit_ids=stable_bit_ids,
        deadline=deadline,
    )
    return _materialize_pattern_context(
        shared,
        pattern,
        deadline=deadline,
    )


def _objective_position_maps(
    context: _LiveContext,
) -> Tuple[Dict[int, Fraction], Dict[int, Fraction]]:
    continuous_positions = {
        int(stable_id): position
        for position, stable_id in enumerate(context.hz.col_ids.tolist())
    }
    binary_positions = {
        int(stable_id): position
        for position, stable_id in enumerate(context.hz.bcol_ids.tolist())
    }
    continuous = {
        continuous_positions[stable_id]: coefficient
        for stable_id, coefficient in context.objective_binding.continuous_terms
    }
    binary = {
        binary_positions[stable_id]: coefficient
        for stable_id, coefficient in context.objective_binding.binary_terms
    }
    return continuous, binary


def _solve_local_candidate(
    context: _LiveContext,
    *,
    deadline: float,
    max_active_columns: int,
    max_dense_entries: int,
    telemetry: Optional[_ScheduledBuildTelemetry] = None,
) -> _CandidateSolve:
    """Propose a dual from a local dense active-column LP, without authority."""

    if time.monotonic() >= deadline:
        return _CandidateSolve(status="deadline_fallback")
    try:
        continuous_q = {
            int(position): float(coefficient)
            for position, coefficient in zip(
                context.candidate_continuous_positions,
                context.candidate_continuous_q,
            )
        }
        binary_q = {
            int(position): float(coefficient)
            for position, coefficient in zip(
                context.candidate_binary_positions,
                context.candidate_binary_q,
            )
        }
        active_continuous = set(continuous_q)
        active_binary = set(binary_q)
        active_continuous.update(
            int(value) for value in context.local_Auc.indices.tolist()
        )
        active_binary.update(
            int(value) for value in context.local_Aub.indices.tolist()
        )
        continuous_positions = tuple(sorted(active_continuous))
        binary_positions = tuple(sorted(active_binary))
        active_count = len(continuous_positions) + len(binary_positions)
        dense_entries = len(context.local_upper_row_ids) * active_count
        if active_count > max_active_columns:
            return _CandidateSolve(status="active_column_cap_fallback")
        if dense_entries > max_dense_entries:
            return _CandidateSolve(status="dense_entry_cap_fallback")
        if active_count == 0:
            return _CandidateSolve(
                status="optimal",
                raw_upper_dual=tuple(
                    0.0 for _ in context.local_upper_row_ids
                ),
            )

        if telemetry is not None:
            if type(telemetry) is not _ScheduledBuildTelemetry:
                raise OperatorPhaseConditionedObjectiveBoundError(
                    "scheduled_telemetry_wrong_type"
                )
            telemetry.linprog_attempted += 1

        continuous_local = {
            position: offset
            for offset, position in enumerate(continuous_positions)
        }
        binary_offset = len(continuous_positions)
        binary_local = {
            position: binary_offset + offset
            for offset, position in enumerate(binary_positions)
        }
        objective = np.zeros(active_count, dtype=np.float64)
        for position, coefficient in continuous_q.items():
            objective[continuous_local[position]] = float(coefficient)
        for position, coefficient in binary_q.items():
            objective[binary_local[position]] = float(coefficient)
        if not np.all(np.isfinite(objective)):
            return _CandidateSolve(status="objective_conversion_fallback")

        upper = np.zeros(
            (len(context.local_upper_row_ids), active_count),
            dtype=np.float64,
        )
        for row in range(upper.shape[0]):
            start = int(context.local_Auc.indptr[row])
            stop = int(context.local_Auc.indptr[row + 1])
            for offset in range(start, stop):
                upper[row, continuous_local[int(
                    context.local_Auc.indices[offset]
                )]] = float(context.local_Auc.data[offset])
            start = int(context.local_Aub.indptr[row])
            stop = int(context.local_Aub.indptr[row + 1])
            for offset in range(start, stop):
                upper[row, binary_local[int(
                    context.local_Aub.indices[offset]
                )]] = float(context.local_Aub.data[offset])

        bounds = [(-1.0, 1.0)] * active_count
        selected_by_position = {
            mapping.binary_position: phase
            for mapping, phase in zip(context.mappings, context.pattern)
        }
        for position, local_position in binary_local.items():
            if position in selected_by_position:
                phase = float(selected_by_position[position])
                bounds[local_position] = (phase, phase)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return _CandidateSolve(status="deadline_fallback")
        if telemetry is not None:
            telemetry.linprog_actual_calls += 1
        result = spo.linprog(
            -objective,
            A_ub=upper,
            b_ub=context.local_ub,
            bounds=bounds,
            method="highs",
            options={"time_limit": max(1.0e-6, remaining)},
        )
        if telemetry is not None:
            telemetry.linprog_completed_calls += 1
        if time.monotonic() >= deadline:
            return _CandidateSolve(status="deadline_fallback")
        if not bool(result.success) or int(result.status) != 0:
            if int(result.status) == 2:
                return _CandidateSolve(
                    status="infeasible_no_authority_fallback"
                )
            return _CandidateSolve(status="nonoptimal_fallback")
        marginals = getattr(
            getattr(result, "ineqlin", None), "marginals", None
        )
        if marginals is None:
            return _CandidateSolve(status="missing_dual_fallback")
        raw = np.asarray(marginals, dtype=np.float64).reshape(-1)
        if raw.size != len(context.local_upper_row_ids):
            return _CandidateSolve(status="dual_shape_fallback")
        return _CandidateSolve(
            status="optimal",
            raw_upper_dual=tuple(float(value) for value in raw.tolist()),
            raw_equality_dual=(),
        )
    except Exception:
        return _CandidateSolve(status="solver_error_fallback")


def _strict_checker_result(
    upper: Any,
    receipt: Any,
    *,
    require_clean_dual: bool,
    expected_parent_semantic_digest: str,
    expected_exact_objective_sha256: str,
    expected_objective_envelope_sha256: str,
    expected_objective_binding_sha256: str,
) -> Tuple[float, str]:
    raw_upper_is_finite_scalar = (
        isinstance(upper, (float, np.floating))
        and not isinstance(upper, (bool, np.bool_))
        and bool(np.isfinite(upper))
    )
    if (
        not raw_upper_is_finite_scalar
        or not isinstance(receipt, dict)
        or receipt.get("schema") != _SPLIT_CHECKER_SCHEMA
        or receipt.get("status") != "verified_upper"
        or receipt.get("route") != _SPLIT_CHECKER_ROUTE
        or receipt.get("proof_authority") is not True
        or receipt.get("verdict_authority") is not False
        or receipt.get("pcoh_authorization") is not False
        or receipt.get("generator_source_read_count") != 0
        or receipt.get("envelope_rehash_bytes") != 0
        or receipt.get("objective_formation_reused") is not True
        or receipt.get("objective_binding_cross_checked") is not True
        or receipt.get("parent_semantic_digest")
        != expected_parent_semantic_digest
        or receipt.get("exact_objective_sha256")
        != expected_exact_objective_sha256
        or receipt.get("objective_envelope_sha256")
        != expected_objective_envelope_sha256
        or receipt.get("objective_binding_sha256")
        != expected_objective_binding_sha256
        or receipt.get("uses_sparse_hstack") is not False
        or receipt.get("uses_sparse_vstack") is not False
        or receipt.get("assembled_sparse_nnz") != 0
        or type(receipt.get("upper")) is not float
        or not math.isfinite(receipt["upper"])
        or receipt.get("upper_float64_rounding")
        != "toward_positive_infinity_from_longdouble_v1"
        or np.longdouble(receipt["upper"]) < np.longdouble(upper)
        or (require_clean_dual and receipt.get("illegal_sign_projected") != 0)
        or (require_clean_dual and receipt.get("nonfinite_dual_zeroed") != 0)
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "split_block_checker_did_not_verify_clean_finite_upper"
        )
    return float(receipt["upper"]), _canonical_sha256(receipt)


def _run_preformed_checker_blocks(
    *,
    objective_envelope: _HZPreformedFactorObjectiveEnvelope,
    objective_binding: ObjectiveBinding,
    parent_semantic_digest: str,
    Auc: sp.csr_matrix,
    Aub: sp.csr_matrix,
    Ac: sp.csr_matrix,
    Ab: sp.csr_matrix,
    ub: np.ndarray,
    b: np.ndarray,
    continuous_lb: np.ndarray,
    continuous_ub: np.ndarray,
    binary_lb: np.ndarray,
    binary_ub: np.ndarray,
    raw_upper_dual: Tuple[float, ...],
    deadline: float,
    telemetry: Optional[_ScheduledBuildTelemetry] = None,
    checker_kind: Optional[str] = None,
) -> Tuple[float, str]:
    if telemetry is not None:
        if type(telemetry) is not _ScheduledBuildTelemetry:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_telemetry_wrong_type"
            )
        if checker_kind not in {"candidate", "zero", "global"}:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_checker_kind_invalid"
            )
        telemetry.split_checker_evaluations += 1
        if checker_kind == "candidate":
            telemetry.candidate_checker_evaluations += 1
        elif checker_kind == "zero":
            telemetry.zero_checker_evaluations += 1
        else:
            telemetry.global_checker_evaluations += 1
    upper, receipt = (
        _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope(
            objective_envelope=objective_envelope,
            expected_parent_semantic_digest=parent_semantic_digest,
            expected_exact_objective_sha256=(
                objective_envelope.exact_objective_sha256
            ),
            expected_objective_binding_sha256=(
                objective_binding.objective_binding_sha256
            ),
            Auc=Auc,
            Aub=Aub,
            Ac=Ac,
            Ab=Ab,
            ub=ub,
            b=b,
            continuous_lb=continuous_lb,
            continuous_ub=continuous_ub,
            binary_lb=binary_lb,
            binary_ub=binary_ub,
            upper_row_dual=np.asarray(raw_upper_dual, dtype=np.float64),
            equality_row_dual=np.empty(0, dtype=np.float64),
            deadline=deadline,
        )
    )
    return _strict_checker_result(
        upper,
        receipt,
        require_clean_dual=True,
        expected_parent_semantic_digest=parent_semantic_digest,
        expected_exact_objective_sha256=(
            objective_envelope.exact_objective_sha256
        ),
        expected_objective_envelope_sha256=(
            objective_envelope.envelope_sha256
        ),
        expected_objective_binding_sha256=(
            objective_binding.objective_binding_sha256
        ),
    )


def _run_checker(
    context: _LiveContext,
    *,
    raw_upper_dual: Tuple[float, ...],
    use_local_rows: bool,
    fix_pattern: bool,
    deadline: float,
    telemetry: Optional[_ScheduledBuildTelemetry] = None,
    checker_kind: Optional[str] = None,
) -> Tuple[float, str]:
    if not use_local_rows and not fix_pattern:
        if raw_upper_dual:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "cached_global_checker_requires_zero_dual"
            )
        if telemetry is not None:
            if type(telemetry) is not _ScheduledBuildTelemetry:
                raise OperatorPhaseConditionedObjectiveBoundError(
                    "scheduled_telemetry_wrong_type"
                )
            if checker_kind != "global":
                raise OperatorPhaseConditionedObjectiveBoundError(
                    "scheduled_cached_checker_kind_invalid"
                )
            telemetry.global_checker_cache_hits += 1
        _check_deadline(deadline, stage="cached_global_checker_reuse")
        return (
            context.global_cube_upper,
            context.global_checker_sha256,
        )
    if use_local_rows:
        Auc = context.local_Auc
        Aub = context.local_Aub
        ub = context.local_ub
    else:
        Auc = context.empty_Ac
        Aub = context.empty_Ab
        ub = context.empty_b
    if fix_pattern:
        binary_lb = context.binary_lb
        binary_ub = context.binary_ub
    else:
        binary_lb = context.shared.binary_cube_lb
        binary_ub = context.shared.binary_cube_ub
    return _run_preformed_checker_blocks(
        objective_envelope=context.objective_envelope,
        objective_binding=context.objective_binding,
        parent_semantic_digest=context.parent_semantic_digest,
        Auc=Auc,
        Aub=Aub,
        Ac=context.empty_Ac,
        Ab=context.empty_Ab,
        ub=ub,
        b=context.empty_b,
        continuous_lb=context.continuous_lb,
        continuous_ub=context.continuous_ub,
        binary_lb=binary_lb,
        binary_ub=binary_ub,
        raw_upper_dual=raw_upper_dual,
        deadline=deadline,
        telemetry=telemetry,
        checker_kind=checker_kind,
    )


def _normalize_candidate(
    candidate: Any,
    *,
    local_rows: int,
) -> Tuple[str, bool, Tuple[float, ...]]:
    if type(candidate) is not _CandidateSolve:
        return "candidate_malformed_fallback", False, ()
    status = candidate.status
    if type(status) is not str or status not in {
        "optimal",
        "deadline_fallback",
        "active_column_cap_fallback",
        "dense_entry_cap_fallback",
        "objective_conversion_fallback",
        "infeasible_no_authority_fallback",
        "nonoptimal_fallback",
        "missing_dual_fallback",
        "dual_shape_fallback",
        "solver_error_fallback",
        "candidate_nonfinite_dual_fallback",
        "candidate_illegal_dual_fallback",
    }:
        return "candidate_malformed_fallback", False, ()
    if status != "optimal":
        return status, False, ()
    if (
        type(candidate.raw_upper_dual) is not tuple
        or type(candidate.raw_equality_dual) is not tuple
        or candidate.raw_equality_dual
        or len(candidate.raw_upper_dual) != local_rows
        or any(type(value) is not float for value in candidate.raw_upper_dual)
    ):
        return "candidate_malformed_fallback", False, ()
    if any(not math.isfinite(value) for value in candidate.raw_upper_dual):
        return "candidate_nonfinite_dual_fallback", False, ()
    # HiGHS/SciPy minimization marginals for upper-only rows must be <= 0.
    if any(value > 0.0 for value in candidate.raw_upper_dual):
        return "candidate_illegal_dual_fallback", False, ()
    return status, True, candidate.raw_upper_dual


def _dual_sha256(
    *,
    accepted: bool,
    raw_upper_dual: Tuple[float, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema": _DUAL_SCHEMA,
            "candidate_dual_accepted": accepted,
            "raw_upper_dual_hex": tuple(
                value.hex() for value in raw_upper_dual
            ),
            "raw_equality_dual_hex": (),
            "dual_convention": "minimization_row_dual",
        }
    )


def _selected_bound(
    *,
    candidate: Optional[float],
    zero: float,
    global_cube: float,
) -> Tuple[str, float]:
    choices = []
    if candidate is not None:
        choices.append((candidate, 0, "candidate_local_dual"))
    choices.extend(
        (
            (zero, 1, "zero_dual_fixed_pattern"),
            (global_cube, 2, "global_cube_baseline"),
        )
    )
    value, _, source = min(choices, key=lambda item: (item[0], item[1]))
    return source, value


def _certificate_payload(
    certificate: OperatorPhaseConditionedObjectiveBoundCertificate,
) -> Dict[str, Any]:
    return {
        "schema": certificate.schema,
        "parent_semantic_digest": certificate.parent_semantic_digest,
        "operator_row_tag_digest": certificate.operator_row_tag_digest,
        "selection_digest": certificate.selection_digest,
        "property_digest": certificate.property_digest,
        "rival_id": certificate.rival_id,
        "rival_binding_digest": certificate.rival_binding_digest,
        "objective_binding_sha256": (
            certificate.objective_binding.objective_binding_sha256
        ),
        "objective_envelope_sha256": (
            certificate.objective_envelope_sha256
        ),
        "stable_bit_ids": certificate.stable_bit_ids,
        "pattern": certificate.pattern,
        "assignments": certificate.assignments,
        "local_upper_row_ids": certificate.local_upper_row_ids,
        "local_upper_row_sha256": certificate.local_upper_row_sha256,
        "omitted_upper_rows": certificate.omitted_upper_rows,
        "omitted_equality_rows": certificate.omitted_equality_rows,
        "local_row_set_sha256": certificate.local_row_set_sha256,
        "verified_context_sha256": certificate.verified_context_sha256,
        "pattern_sha256": certificate.pattern_sha256,
        "dual_sha256": certificate.dual_sha256,
        "raw_upper_dual_hex": tuple(
            value.hex() for value in certificate.raw_upper_dual
        ),
        "raw_equality_dual_hex": tuple(
            value.hex() for value in certificate.raw_equality_dual
        ),
        "candidate_status": certificate.candidate_status,
        "candidate_dual_accepted": certificate.candidate_dual_accepted,
        "candidate_checked_upper": certificate.candidate_checked_upper,
        "zero_dual_fixed_upper": certificate.zero_dual_fixed_upper,
        "global_cube_upper_exact": certificate.global_cube_upper_exact,
        "global_cube_upper": certificate.global_cube_upper,
        "selected_source": certificate.selected_source,
        "upper_stored": certificate.upper_stored,
        "candidate_checker_sha256": certificate.candidate_checker_sha256,
        "zero_checker_sha256": certificate.zero_checker_sha256,
        "global_checker_sha256": certificate.global_checker_sha256,
        "checker_bundle_sha256": certificate.checker_bundle_sha256,
        "proof_authority": certificate.proof_authority,
    }


def _make_receipt(
    certificate: OperatorPhaseConditionedObjectiveBoundCertificate,
) -> Mapping[str, Any]:
    payload: Dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "status": "verified_finite_conditional_upper",
        "proof_authority": False,
        "verdict_authority": False,
        "complete_cover_live_replay_required_for_authority": True,
        "candidate_solver_authority": False,
        "candidate_solver_status_used_for_proof": False,
        "candidate_solver_objective_used_for_proof": False,
        "infeasibility_authority": False,
        "pattern_deletion": False,
        "parent_semantic_digest": certificate.parent_semantic_digest,
        "operator_row_tag_digest": certificate.operator_row_tag_digest,
        "selection_digest": certificate.selection_digest,
        "property_digest": certificate.property_digest,
        "objective_binding_sha256": (
            certificate.objective_binding.objective_binding_sha256
        ),
        "objective_envelope_sha256": (
            certificate.objective_envelope_sha256
        ),
        "pattern_sha256": certificate.pattern_sha256,
        "local_row_set_sha256": certificate.local_row_set_sha256,
        "verified_context_sha256": certificate.verified_context_sha256,
        "dual_sha256": certificate.dual_sha256,
        "checker_bundle_sha256": certificate.checker_bundle_sha256,
        "certificate_sha256": certificate.certificate_sha256,
        "retained_upper_rows": len(certificate.local_upper_row_ids),
        "retained_upper_row_limit": 12,
        "retained_equality_rows": 0,
        "omitted_equality_rows": certificate.omitted_equality_rows,
        "omitted_upper_rows": certificate.omitted_upper_rows,
        "all_parent_equality_rows_omitted_as_relaxation": True,
        "all_other_upper_rows_omitted_as_relaxation": True,
        "relaxation_relation": (
            "conditioned_parent_subset_of_local_three_rows_box"
        ),
        "objective_equality_substitution": False,
        "threshold_application_count": 1,
        "selected_binary_bounds": "signed_exact_minus_one_or_plus_one",
        "unselected_factor_bounds": "box_minus_one_plus_one",
        "full_parent_constraint_csr_loaded_by_candidate": False,
        "candidate_active_column_compression": True,
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "checker_route": _SPLIT_CHECKER_ROUTE,
        "objective_envelope_formation_count": 1,
        "objective_envelope_production_ready": False,
        "objective_generator_validation_pass_count": 1,
        "objective_source_hash_pass_count": 1,
        "objective_exact_expansion_pass_count": 1,
        "objective_binding_built_from_sealed_exact_material": True,
        "objective_binding_sha256_cross_checked": True,
        "preformed_checker_generator_source_read_count": 0,
        "preformed_checker_envelope_rehash_bytes": 0,
        "global_checker_bundle_evaluation_count": 1,
        "global_checker_reused_per_pattern": True,
        "selected_source": certificate.selected_source,
        "upper_stored_hex": certificate.upper_stored.hex(),
        "baseline_nonregression_checked": True,
        "deterministic_pattern_parallelism": "disabled_v1",
        "bundle_scope_immutable_verified_context": True,
        "context_preparation_count": 1,
        "selection_verification_count": 1,
        "exact_objective_expansion_count": 1,
        "context_semantic_digest_count": 1,
        "terminal_parent_semantic_digest_count": 1,
        "source_row_hashing_pass_count": 1,
        "per_pattern_context_reverification": False,
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return MappingProxyType(payload)


def _make_certificate(
    context: _LiveContext,
    selection: OperatorExactReLUPhaseSelection,
    *,
    candidate_status: str,
    candidate_accepted: bool,
    raw_upper_dual: Tuple[float, ...],
    candidate_upper: Optional[float],
    candidate_checker_sha: Optional[str],
    zero_upper: float,
    zero_checker_sha: str,
    global_upper: float,
    global_checker_sha: str,
) -> OperatorPhaseConditionedObjectiveBoundCertificate:
    dual_sha = _dual_sha256(
        accepted=candidate_accepted,
        raw_upper_dual=raw_upper_dual,
    )
    checker_bundle_sha = _canonical_sha256(
        {
            "schema": _CHECKER_BUNDLE_SCHEMA,
            "candidate_checker_sha256": candidate_checker_sha,
            "zero_checker_sha256": zero_checker_sha,
            "global_checker_sha256": global_checker_sha,
            "candidate_checked_upper": candidate_upper,
            "zero_dual_fixed_upper": zero_upper,
            "global_cube_upper": global_upper,
        }
    )
    selected_source, selected_upper = _selected_bound(
        candidate=candidate_upper,
        zero=zero_upper,
        global_cube=global_upper,
    )
    provisional = OperatorPhaseConditionedObjectiveBoundCertificate(
        schema=_CERTIFICATE_SCHEMA,
        parent_semantic_digest=context.parent_semantic_digest,
        operator_row_tag_digest=selection.operator_row_tag_digest,
        selection_digest=selection.selection_digest,
        property_digest=context.property_digest,
        rival_id=context.rival.rival_id,
        rival_binding_digest=context.rival.binding_digest,
        objective_binding=context.objective_binding,
        objective_envelope_sha256=(
            context.objective_envelope.envelope_sha256
        ),
        stable_bit_ids=context.stable_bit_ids,
        pattern=context.pattern,
        assignments=context.assignments,
        local_upper_row_ids=context.local_upper_row_ids,
        local_upper_row_sha256=context.local_upper_row_sha256,
        omitted_upper_rows=context.omitted_upper_rows,
        omitted_equality_rows=context.omitted_equality_rows,
        local_row_set_sha256=context.local_row_set_sha256,
        verified_context_sha256=context.verified_context_sha256,
        pattern_sha256=context.pattern_sha256,
        dual_sha256=dual_sha,
        raw_upper_dual=raw_upper_dual,
        raw_equality_dual=(),
        candidate_status=candidate_status,
        candidate_dual_accepted=candidate_accepted,
        candidate_checked_upper=candidate_upper,
        zero_dual_fixed_upper=zero_upper,
        global_cube_upper_exact=context.global_cube_upper_exact,
        global_cube_upper=global_upper,
        selected_source=selected_source,
        upper_stored=selected_upper,
        candidate_checker_sha256=candidate_checker_sha,
        zero_checker_sha256=zero_checker_sha,
        global_checker_sha256=global_checker_sha,
        checker_bundle_sha256=checker_bundle_sha,
        certificate_sha256="",
        receipt=MappingProxyType({}),
        proof_authority=False,
    )
    certificate_sha = _canonical_sha256(_certificate_payload(provisional))
    with_sha = OperatorPhaseConditionedObjectiveBoundCertificate(
        **{
            **provisional.__dict__,
            "certificate_sha256": certificate_sha,
        }
    )
    return OperatorPhaseConditionedObjectiveBoundCertificate(
        **{
            **with_sha.__dict__,
            "receipt": _make_receipt(with_sha),
        }
    )


def _build_bound_from_verified_context(
    shared: _VerifiedBundleContext,
    selection: OperatorExactReLUPhaseSelection,
    *,
    pattern: SignedPattern,
    deadline: float,
    candidate_timeout_seconds: float,
    max_candidate_active_columns: int,
    max_candidate_dense_entries: int,
    telemetry: Optional[_ScheduledBuildTelemetry] = None,
) -> OperatorPhaseConditionedObjectiveBoundCertificate:
    """Run only pattern-local proposal and checker work on shared context."""

    context = _materialize_pattern_context(
        shared,
        pattern,
        deadline=deadline,
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "certificate_deadline_exhausted_before_pattern_candidate"
        )
    reserve = min(0.25, remaining / 2.0)
    candidate_deadline = min(
        time.monotonic() + candidate_timeout_seconds,
        deadline - reserve,
    )
    if telemetry is not None:
        if type(telemetry) is not _ScheduledBuildTelemetry:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_telemetry_wrong_type"
            )
        telemetry.candidate_proposal_invocations += 1
        linprog_before = (
            telemetry.linprog_attempted,
            telemetry.linprog_actual_calls,
            telemetry.linprog_completed_calls,
        )
    else:
        linprog_before = (0, 0, 0)
    candidate = _solve_local_candidate(
        context,
        deadline=candidate_deadline,
        max_active_columns=max_candidate_active_columns,
        max_dense_entries=max_candidate_dense_entries,
        telemetry=telemetry,
    )
    candidate_status, accepted, raw_dual = _normalize_candidate(
        candidate,
        local_rows=len(context.local_upper_row_ids),
    )
    if telemetry is not None:
        linprog_delta = tuple(
            after - before
            for after, before in zip(
                (
                    telemetry.linprog_attempted,
                    telemetry.linprog_actual_calls,
                    telemetry.linprog_completed_calls,
                ),
                linprog_before,
            )
        )
        if any(value not in {0, 1} for value in linprog_delta):
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_pattern_linprog_delta_invalid"
            )
        eligible, called, completed = tuple(
            bool(value) for value in linprog_delta
        )
        if not _scheduled_trace_control_flow_allowed(
            status=candidate_status,
            eligible=eligible,
            called=called,
            completed=completed,
        ):
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_pattern_linprog_control_flow_invalid"
            )
        telemetry.candidate_call_traces.append(
            (
                context.pattern,
                eligible,
                called,
                completed,
                candidate_status,
                accepted,
            )
        )
        telemetry.candidate_statuses.append(candidate_status)
        telemetry.candidate_dual_accepted += int(accepted)
    candidate_upper: Optional[float] = None
    candidate_checker_sha: Optional[str] = None
    if accepted:
        candidate_upper, candidate_checker_sha = _run_checker(
            context,
            raw_upper_dual=raw_dual,
            use_local_rows=True,
            fix_pattern=True,
            deadline=deadline,
            telemetry=telemetry,
            checker_kind="candidate",
        )

    zero_dual = tuple(0.0 for _ in context.local_upper_row_ids)
    zero_upper, zero_checker_sha = _run_checker(
        context,
        raw_upper_dual=zero_dual,
        use_local_rows=True,
        fix_pattern=True,
        deadline=deadline,
        telemetry=telemetry,
        checker_kind="zero",
    )
    global_upper, global_checker_sha = _run_checker(
        context,
        raw_upper_dual=(),
        use_local_rows=False,
        fix_pattern=False,
        deadline=deadline,
        telemetry=telemetry,
        checker_kind="global",
    )
    if Fraction.from_float(global_upper) < context.global_cube_upper_exact:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "global_checker_bound_below_exact_fraction_cube"
        )
    _check_deadline(deadline, stage="final_certificate_binding")
    return _make_certificate(
        context,
        selection,
        candidate_status=candidate_status,
        candidate_accepted=accepted,
        raw_upper_dual=raw_dual,
        candidate_upper=candidate_upper,
        candidate_checker_sha=candidate_checker_sha,
        zero_upper=zero_upper,
        zero_checker_sha=zero_checker_sha,
        global_upper=global_upper,
        global_checker_sha=global_checker_sha,
    )


def build_operator_phase_conditioned_objective_bound(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    pattern: SignedPattern,
    certificate_timeout_seconds: float = (
        _DEFAULT_CERTIFICATE_TIMEOUT_SECONDS
    ),
    candidate_timeout_seconds: float = _DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
    max_candidate_active_columns: int = (
        _DEFAULT_MAX_CANDIDATE_ACTIVE_COLUMNS
    ),
    max_candidate_dense_entries: int = (
        _DEFAULT_MAX_CANDIDATE_DENSE_ENTRIES
    ),
) -> OperatorPhaseConditionedObjectiveBoundCertificate:
    """Produce one local-row conditional bound and its replay receipt."""

    certificate_timeout = _strict_timeout(
        certificate_timeout_seconds,
        name="certificate_timeout_seconds",
    )
    candidate_timeout = _strict_timeout(
        candidate_timeout_seconds,
        name="candidate_timeout_seconds",
    )
    active_cap = _strict_positive_cap(
        max_candidate_active_columns,
        name="max_candidate_active_columns",
    )
    dense_cap = _strict_positive_cap(
        max_candidate_dense_entries,
        name="max_candidate_dense_entries",
    )
    deadline = time.monotonic() + certificate_timeout
    shared = _prepare_verified_bundle_context(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_bit_ids=stable_bit_ids,
        deadline=deadline,
    )
    result = _build_bound_from_verified_context(
        shared,
        selection,
        pattern=pattern,
        deadline=deadline,
        candidate_timeout_seconds=candidate_timeout,
        max_candidate_active_columns=active_cap,
        max_candidate_dense_entries=dense_cap,
    )
    _terminal_parent_seal(shared, deadline=deadline)
    return result


def _strict_certificate_fields(
    certificate: Any,
) -> OperatorPhaseConditionedObjectiveBoundCertificate:
    if (
        type(certificate)
        is not OperatorPhaseConditionedObjectiveBoundCertificate
        or certificate.schema != _CERTIFICATE_SCHEMA
        or certificate.proof_authority is not False
        or not _valid_sha256(certificate.parent_semantic_digest)
        or not _valid_sha256(certificate.operator_row_tag_digest)
        or not _valid_sha256(certificate.selection_digest)
        or not _valid_sha256(certificate.property_digest)
        or type(certificate.rival_id) is not int
        or certificate.rival_id < 0
        or not _valid_sha256(certificate.rival_binding_digest)
        or not verify_objective_binding(certificate.objective_binding)
        or not _valid_sha256(certificate.objective_envelope_sha256)
        or type(certificate.stable_bit_ids) is not tuple
        or type(certificate.pattern) is not tuple
        or type(certificate.assignments) is not tuple
        or type(certificate.local_upper_row_ids) is not tuple
        or type(certificate.local_upper_row_sha256) is not tuple
        or len(certificate.local_upper_row_ids)
        != len(certificate.local_upper_row_sha256)
        or len(certificate.local_upper_row_ids) > 12
        or type(certificate.omitted_upper_rows) is not int
        or certificate.omitted_upper_rows < 0
        or type(certificate.omitted_equality_rows) is not int
        or certificate.omitted_equality_rows < 0
        or any(
            type(row) is not int or row < 0
            for row in certificate.local_upper_row_ids
        )
        or any(
            not _valid_sha256(value)
            for value in certificate.local_upper_row_sha256
        )
        or not _valid_sha256(certificate.local_row_set_sha256)
        or not _valid_sha256(certificate.verified_context_sha256)
        or not _valid_sha256(certificate.pattern_sha256)
        or not _valid_sha256(certificate.dual_sha256)
        or type(certificate.raw_upper_dual) is not tuple
        or type(certificate.raw_equality_dual) is not tuple
        or any(type(value) is not float for value in certificate.raw_upper_dual)
        or any(type(value) is not float for value in certificate.raw_equality_dual)
        or certificate.raw_equality_dual
        or type(certificate.candidate_status) is not str
        or type(certificate.candidate_dual_accepted) is not bool
        or (
            certificate.candidate_checked_upper is not None
            and type(certificate.candidate_checked_upper) is not float
        )
        or type(certificate.zero_dual_fixed_upper) is not float
        or type(certificate.global_cube_upper_exact) is not Fraction
        or type(certificate.global_cube_upper) is not float
        or type(certificate.selected_source) is not str
        or type(certificate.upper_stored) is not float
        or (
            certificate.candidate_checker_sha256 is not None
            and not _valid_sha256(certificate.candidate_checker_sha256)
        )
        or not _valid_sha256(certificate.zero_checker_sha256)
        or not _valid_sha256(certificate.global_checker_sha256)
        or not _valid_sha256(certificate.checker_bundle_sha256)
        or not _valid_sha256(certificate.certificate_sha256)
        or type(certificate.receipt) is not MappingProxyType
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "certificate_top_level_noncanonical"
        )
    float_values = (
        certificate.raw_upper_dual
        + tuple(
            value
            for value in (
                certificate.candidate_checked_upper,
                certificate.zero_dual_fixed_upper,
                certificate.global_cube_upper,
                certificate.upper_stored,
            )
            if value is not None
        )
    )
    if any(not math.isfinite(value) for value in float_values):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "certificate_nonfinite_numeric_field"
        )
    return certificate


def _receipt_payload_matches(
    certificate: OperatorPhaseConditionedObjectiveBoundCertificate,
) -> bool:
    expected = _make_receipt(
        OperatorPhaseConditionedObjectiveBoundCertificate(
            **{
                **certificate.__dict__,
                "receipt": MappingProxyType({}),
            }
        )
    )
    return _canonical_form(expected) == _canonical_form(certificate.receipt)


def _verify_bound_from_verified_context(
    shared: _VerifiedBundleContext,
    selection: OperatorExactReLUPhaseSelection,
    checked: OperatorPhaseConditionedObjectiveBoundCertificate,
    *,
    deadline: float,
) -> bool:
    """Replay one pattern without re-verifying or re-hashing shared input."""

    context = _materialize_pattern_context(
        shared,
        checked.pattern,
        deadline=deadline,
    )
    if checked.candidate_dual_accepted:
        if (
            checked.candidate_status != "optimal"
            or len(checked.raw_upper_dual)
            != len(context.local_upper_row_ids)
            or any(value > 0.0 for value in checked.raw_upper_dual)
        ):
            return False
        candidate_upper, candidate_checker_sha = _run_checker(
            context,
            raw_upper_dual=checked.raw_upper_dual,
            use_local_rows=True,
            fix_pattern=True,
            deadline=deadline,
        )
    else:
        if checked.raw_upper_dual:
            return False
        candidate_upper = None
        candidate_checker_sha = None
    zero_upper, zero_checker_sha = _run_checker(
        context,
        raw_upper_dual=tuple(
            0.0 for _ in context.local_upper_row_ids
        ),
        use_local_rows=True,
        fix_pattern=True,
        deadline=deadline,
    )
    global_upper, global_checker_sha = _run_checker(
        context,
        raw_upper_dual=(),
        use_local_rows=False,
        fix_pattern=False,
        deadline=deadline,
    )
    if Fraction.from_float(global_upper) < context.global_cube_upper_exact:
        return False
    expected = _make_certificate(
        context,
        selection,
        candidate_status=checked.candidate_status,
        candidate_accepted=checked.candidate_dual_accepted,
        raw_upper_dual=checked.raw_upper_dual,
        candidate_upper=candidate_upper,
        candidate_checker_sha=candidate_checker_sha,
        zero_upper=zero_upper,
        zero_checker_sha=zero_checker_sha,
        global_upper=global_upper,
        global_checker_sha=global_checker_sha,
    )
    if (
        _canonical_sha256(_certificate_payload(checked))
        != checked.certificate_sha256
        or not _receipt_payload_matches(checked)
    ):
        return False
    return (
        _canonical_form(_certificate_payload(checked))
        == _canonical_form(_certificate_payload(expected))
        and _canonical_form(checked.receipt)
        == _canonical_form(expected.receipt)
        and checked.certificate_sha256 == expected.certificate_sha256
    )


def verify_operator_phase_conditioned_objective_bound(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    certificate: OperatorPhaseConditionedObjectiveBoundCertificate,
    *,
    certificate_timeout_seconds: float = (
        _DEFAULT_CERTIFICATE_TIMEOUT_SECONDS
    ),
) -> bool:
    """Re-extract live rows and independently replay one certificate."""

    try:
        checked = _strict_certificate_fields(certificate)
        timeout = _strict_timeout(
            certificate_timeout_seconds,
            name="certificate_timeout_seconds",
        )
        deadline = time.monotonic() + timeout
        shared = _prepare_verified_bundle_context(
            build,
            rivals,
            selection,
            focused_rival_id=checked.rival_id,
            stable_bit_ids=checked.stable_bit_ids,
            deadline=deadline,
        )
        verified = _verify_bound_from_verified_context(
            shared,
            selection,
            checked,
            deadline=deadline,
        )
        if not verified:
            return False
        _terminal_parent_seal(shared, deadline=deadline)
        return True
    except (
        OperatorPhaseConditionedObjectiveBoundError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _build_complete_operator_phase_conditioned_objective_bounds_until(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    deadline: float,
    candidate_timeout_seconds: float = _DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
) -> Tuple[OperatorPhaseConditionedObjectiveBoundCertificate, ...]:
    """Private complete producer using a caller-owned absolute deadline."""

    stable_ids = _strict_stable_bit_ids(stable_bit_ids)
    if type(deadline) not in {int, float} or not math.isfinite(float(deadline)):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "complete_bundle_absolute_deadline_invalid"
        )
    bundle_deadline = float(deadline)
    candidate_timeout = _strict_timeout(
        candidate_timeout_seconds,
        name="candidate_timeout_seconds",
    )
    shared = _prepare_verified_bundle_context(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_bit_ids=stable_ids,
        deadline=bundle_deadline,
    )
    completed = []
    for pattern in itertools.product((-1, 1), repeat=len(stable_ids)):
        remaining = bundle_deadline - time.monotonic()
        if remaining <= 0.0:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "complete_bundle_deadline_exhausted_no_partial_output"
            )
        completed.append(
            _build_bound_from_verified_context(
                shared,
                selection,
                pattern=tuple(pattern),
                deadline=bundle_deadline,
                candidate_timeout_seconds=min(
                    candidate_timeout,
                    max(1.0e-6, remaining / 2.0),
                ),
                max_candidate_active_columns=(
                    _DEFAULT_MAX_CANDIDATE_ACTIVE_COLUMNS
                ),
                max_candidate_dense_entries=(
                    _DEFAULT_MAX_CANDIDATE_DENSE_ENTRIES
                ),
            )
        )
        if time.monotonic() >= bundle_deadline:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "complete_bundle_deadline_exhausted_no_partial_output"
            )
    result = tuple(completed)
    expected = tuple(itertools.product((-1, 1), repeat=len(stable_ids)))
    if tuple(item.pattern for item in result) != expected:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "complete_pattern_production_internal_mismatch"
        )
    _terminal_parent_seal(shared, deadline=bundle_deadline)
    return result


def build_complete_operator_phase_conditioned_objective_bounds(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    certificate_timeout_seconds: float = (
        _DEFAULT_CERTIFICATE_TIMEOUT_SECONDS
    ),
    candidate_timeout_seconds: float = _DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
) -> Tuple[OperatorPhaseConditionedObjectiveBoundCertificate, ...]:
    """Duration wrapper around the shared absolute-deadline producer."""

    total_timeout = _strict_timeout(
        certificate_timeout_seconds,
        name="complete_bundle_timeout_seconds",
    )
    return _build_complete_operator_phase_conditioned_objective_bounds_until(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_bit_ids=stable_bit_ids,
        deadline=time.monotonic() + total_timeout,
        candidate_timeout_seconds=candidate_timeout_seconds,
    )


def _scheduled_telemetry_mapping(
    counters: _ScheduledBuildTelemetry,
    *,
    stable_bits: int,
    expected_pattern_count: int,
    status: str,
) -> Mapping[str, Any]:
    """Freeze counters after a successful terminal parent seal."""

    if (
        type(counters) is not _ScheduledBuildTelemetry
        or status not in {"complete", "stopped_by_policy"}
        or counters.context_formations != 1
        or counters.exact_objective_expansions != 1
        or counters.source_row_hash_passes != 1
        or counters.patterns_started != counters.patterns_completed
        or counters.patterns_completed != len(counters.completed_patterns)
        or counters.patterns_completed != len(counters.pattern_seconds)
        or counters.patterns_completed != len(counters.observed_upper_exact)
        or counters.patterns_completed != len(counters.candidate_call_traces)
        or counters.candidate_proposal_invocations
        != counters.patterns_completed
        or len(counters.candidate_statuses) != counters.patterns_completed
        or any(
            type(trace) is not tuple
            or len(trace) != 6
            or type(trace[0]) is not tuple
            or any(type(value) is not bool for value in trace[1:4])
            or type(trace[4]) is not str
            or type(trace[5]) is not bool
            or not _scheduled_trace_control_flow_allowed(
                status=trace[4],
                eligible=trace[1],
                called=trace[2],
                completed=trace[3],
            )
            or trace[5] is not (trace[4] == "optimal")
            for trace in counters.candidate_call_traces
        )
        or tuple(trace[0] for trace in counters.candidate_call_traces)
        != tuple(counters.completed_patterns)
        or tuple(trace[4] for trace in counters.candidate_call_traces)
        != tuple(counters.candidate_statuses)
        or any(
            type(value) is not Fraction
            for value in counters.observed_upper_exact
        )
        or counters.linprog_attempted
        != sum(
            int(trace[1]) for trace in counters.candidate_call_traces
        )
        or counters.linprog_actual_calls
        != sum(
            int(trace[2]) for trace in counters.candidate_call_traces
        )
        or counters.linprog_completed_calls
        != sum(
            int(trace[3]) for trace in counters.candidate_call_traces
        )
        or counters.candidate_dual_accepted
        != sum(
            int(trace[5]) for trace in counters.candidate_call_traces
        )
        or not (
            0
            <= counters.linprog_completed_calls
            <= counters.linprog_actual_calls
            <= counters.linprog_attempted
            <= counters.candidate_proposal_invocations
        )
        or not (
            0
            <= counters.candidate_dual_accepted
            <= counters.patterns_completed
        )
        or counters.candidate_checker_evaluations
        != counters.candidate_dual_accepted
        or counters.zero_checker_evaluations != counters.patterns_completed
        or counters.global_checker_evaluations != 1
        or counters.global_checker_cache_hits != counters.patterns_completed
        or counters.split_checker_evaluations
        != (
            counters.candidate_checker_evaluations
            + counters.zero_checker_evaluations
            + counters.global_checker_evaluations
        )
        or counters.terminal_parent_seal_attempts != 1
        or counters.terminal_parent_seal_completions != 1
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            for value in (
                counters.context_seconds,
                counters.total_seconds,
                *counters.pattern_seconds,
            )
        )
        or counters.context_seconds > counters.total_seconds
        or math.fsum(
            (counters.context_seconds, *counters.pattern_seconds)
        )
        > math.nextafter(counters.total_seconds, math.inf)
        or (
            status == "complete"
            and counters.patterns_completed != expected_pattern_count
        )
        or (
            status == "stopped_by_policy"
            and not 0 < counters.patterns_completed <= expected_pattern_count
        )
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_telemetry_internal_invariant_failed"
        )
    status_counts: Dict[str, int] = {}
    for candidate_status in counters.candidate_statuses:
        status_counts[candidate_status] = (
            status_counts.get(candidate_status, 0) + 1
        )
    payload: Dict[str, Any] = {
        "schema": _SCHEDULED_TELEMETRY_SCHEMA,
        "status": status,
        "candidate_only": True,
        "full_parent_lp_called": False,
        "proof_authority": False,
        "verdict_authority": False,
        "stable_bits": stable_bits,
        "local_upper_rows": 3 * stable_bits,
        "expected_pattern_count": expected_pattern_count,
        "patterns_started": counters.patterns_started,
        "patterns_completed": counters.patterns_completed,
        "completed_patterns_in_execution_order": tuple(
            counters.completed_patterns
        ),
        "observed_upper_exact_in_execution_order": tuple(
            counters.observed_upper_exact
        ),
        "candidate_call_trace_in_execution_order": tuple(
            {
                "pattern": trace[0],
                "linprog_eligible": trace[1],
                "linprog_called": trace[2],
                "linprog_completed": trace[3],
                "normalized_candidate_status": trace[4],
                "candidate_dual_accepted": trace[5],
            }
            for trace in counters.candidate_call_traces
        ),
        "context_formations": counters.context_formations,
        "exact_objective_expansions": (
            counters.exact_objective_expansions
        ),
        "source_row_hash_passes": counters.source_row_hash_passes,
        "candidate_proposal_invocations": (
            counters.candidate_proposal_invocations
        ),
        "linprog_attempted": counters.linprog_attempted,
        "linprog_actual_calls": counters.linprog_actual_calls,
        "linprog_completed_calls": counters.linprog_completed_calls,
        "candidate_statuses_in_execution_order": tuple(
            counters.candidate_statuses
        ),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "candidate_dual_accepted": counters.candidate_dual_accepted,
        "split_checker_evaluations": counters.split_checker_evaluations,
        "candidate_checker_evaluations": (
            counters.candidate_checker_evaluations
        ),
        "zero_checker_evaluations": counters.zero_checker_evaluations,
        "global_checker_evaluations": counters.global_checker_evaluations,
        "global_checker_cache_hits": counters.global_checker_cache_hits,
        "context_seconds": counters.context_seconds,
        "pattern_seconds_in_execution_order": tuple(
            counters.pattern_seconds
        ),
        "total_seconds": counters.total_seconds,
        "terminal_parent_seal_attempts": (
            counters.terminal_parent_seal_attempts
        ),
        "terminal_parent_seal_completions": (
            counters.terminal_parent_seal_completions
        ),
        "actual_call_site_counters": True,
        "candidate_solver_status_authority": False,
        "candidate_solver_objective_authority": False,
        "global_checker_reused_per_pattern": True,
    }
    payload["telemetry_sha256"] = _canonical_sha256(payload)
    frozen = _deep_freeze(payload)
    if type(frozen) is not MappingProxyType:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_telemetry_freeze_failed"
        )
    return frozen


def _first_scheduled_policy_trigger(
    policy: OperatorPhaseConditionedScheduledStopPolicy,
    observed_upper_exact: Tuple[Fraction, ...],
    *,
    pattern_count: int,
) -> Optional[Tuple[int, str]]:
    """Replay one policy prefix and return its unique first trigger."""

    checked = _strict_scheduled_stop_policy(
        policy, pattern_count=pattern_count
    )
    for index, observed in enumerate(observed_upper_exact):
        if type(observed) is not Fraction:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_observed_upper_trace_not_exact_fraction"
            )
        if index in checked.stop_after_pattern_indices:
            return index, "stop_after_pattern_index_reached"
        if (
            index in checked.threshold_pattern_indices
            and checked.strict_upper_threshold is not None
            and observed > checked.strict_upper_threshold
        ):
            return index, "strict_upper_threshold_exceeded"
    return None


def _verify_scheduled_telemetry_structure(
    telemetry: Any,
    *,
    stable_ids: Tuple[int, ...],
    evaluation_schedule: Tuple[SignedPattern, ...],
    status: str,
    execution_certificates: Optional[
        Tuple[OperatorPhaseConditionedObjectiveBoundCertificate, ...]
    ] = None,
) -> bool:
    """Strictly replay one frozen telemetry payload and every aggregate."""

    if (
        type(telemetry) is not MappingProxyType
        or set(telemetry.keys()) != _SCHEDULED_TELEMETRY_KEYS
        or not _is_strict_deep_frozen_canonical_payload(telemetry)
        or status not in {"complete", "stopped_by_policy"}
    ):
        return False
    body = dict(telemetry)
    telemetry_sha = body.pop("telemetry_sha256", None)
    if (
        not _valid_sha256(telemetry_sha)
        or _canonical_sha256(body) != telemetry_sha
    ):
        return False

    pattern_count = len(evaluation_schedule)
    completed_count = telemetry.get("patterns_completed")
    if (
        type(completed_count) is not int
        or (
            status == "complete"
            and completed_count != pattern_count
        )
        or (
            status == "stopped_by_policy"
            and not 0 < completed_count <= pattern_count
        )
    ):
        return False
    completed_patterns = telemetry.get(
        "completed_patterns_in_execution_order"
    )
    observed = telemetry.get("observed_upper_exact_in_execution_order")
    traces = telemetry.get("candidate_call_trace_in_execution_order")
    statuses = telemetry.get("candidate_statuses_in_execution_order")
    status_counts = telemetry.get("candidate_status_counts")
    pattern_seconds = telemetry.get("pattern_seconds_in_execution_order")
    if (
        type(completed_patterns) is not tuple
        or completed_patterns != evaluation_schedule[:completed_count]
        or type(observed) is not tuple
        or len(observed) != completed_count
        or any(
            type(value) is not Fraction
            or value.numerator.bit_length() > 4096
            or value.denominator.bit_length() > 4096
            for value in observed
        )
        or type(traces) is not tuple
        or len(traces) != completed_count
        or type(statuses) is not tuple
        or len(statuses) != completed_count
        or type(status_counts) is not MappingProxyType
        or type(pattern_seconds) is not tuple
        or len(pattern_seconds) != completed_count
    ):
        return False

    derived_statuses = []
    derived_accepted = 0
    derived_attempted = 0
    derived_called = 0
    derived_completed = 0
    for index, trace in enumerate(traces):
        if (
            type(trace) is not MappingProxyType
            or set(trace.keys()) != _SCHEDULED_PATTERN_CALL_TRACE_KEYS
            or not _is_strict_deep_frozen_canonical_payload(trace)
            or trace.get("pattern") != evaluation_schedule[index]
        ):
            return False
        eligible = trace.get("linprog_eligible")
        called = trace.get("linprog_called")
        completed = trace.get("linprog_completed")
        normalized_status = trace.get("normalized_candidate_status")
        accepted = trace.get("candidate_dual_accepted")
        if (
            any(
                type(value) is not bool
                for value in (eligible, called, completed, accepted)
            )
            or type(normalized_status) is not str
            or not _scheduled_trace_control_flow_allowed(
                status=normalized_status,
                eligible=eligible,
                called=called,
                completed=completed,
            )
            or accepted is not (normalized_status == "optimal")
        ):
            return False
        derived_statuses.append(normalized_status)
        derived_accepted += int(accepted)
        derived_attempted += int(eligible)
        derived_called += int(called)
        derived_completed += int(completed)

    derived_status_tuple = tuple(derived_statuses)
    derived_status_counts = {
        candidate_status: derived_status_tuple.count(candidate_status)
        for candidate_status in sorted(set(derived_status_tuple))
    }
    expected_scalars: Dict[str, Any] = {
        "schema": _SCHEDULED_TELEMETRY_SCHEMA,
        "status": status,
        "candidate_only": True,
        "full_parent_lp_called": False,
        "proof_authority": False,
        "verdict_authority": False,
        "stable_bits": len(stable_ids),
        "local_upper_rows": 3 * len(stable_ids),
        "expected_pattern_count": pattern_count,
        "patterns_started": completed_count,
        "candidate_proposal_invocations": completed_count,
        "linprog_attempted": derived_attempted,
        "linprog_actual_calls": derived_called,
        "linprog_completed_calls": derived_completed,
        "candidate_dual_accepted": derived_accepted,
        "split_checker_evaluations": (
            1 + completed_count + derived_accepted
        ),
        "candidate_checker_evaluations": derived_accepted,
        "zero_checker_evaluations": completed_count,
        "global_checker_evaluations": 1,
        "global_checker_cache_hits": completed_count,
        "context_formations": 1,
        "exact_objective_expansions": 1,
        "source_row_hash_passes": 1,
        "terminal_parent_seal_attempts": 1,
        "terminal_parent_seal_completions": 1,
        "actual_call_site_counters": True,
        "candidate_solver_status_authority": False,
        "candidate_solver_objective_authority": False,
        "global_checker_reused_per_pattern": True,
    }
    if any(
        type(telemetry.get(key)) is not type(expected)
        or telemetry.get(key) != expected
        for key, expected in expected_scalars.items()
    ):
        return False
    if (
        statuses != derived_status_tuple
        or tuple(status_counts.keys()) != tuple(derived_status_counts.keys())
        or any(type(value) is not int for value in status_counts.values())
        or dict(status_counts) != derived_status_counts
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            for value in (
                telemetry.get("context_seconds"),
                telemetry.get("total_seconds"),
                *pattern_seconds,
            )
        )
        or telemetry.get("context_seconds")
        > telemetry.get("total_seconds")
        or math.fsum(
            (
                telemetry.get("context_seconds"),
                *pattern_seconds,
            )
        )
        > math.nextafter(telemetry.get("total_seconds"), math.inf)
    ):
        return False

    if execution_certificates is not None:
        if (
            type(execution_certificates) is not tuple
            or len(execution_certificates) != completed_count
        ):
            return False
        for index, certificate in enumerate(execution_certificates):
            if (
                certificate.pattern != evaluation_schedule[index]
                or certificate.candidate_status != statuses[index]
                or certificate.candidate_dual_accepted
                is not traces[index].get("candidate_dual_accepted")
                or Fraction.from_float(certificate.upper_stored)
                != observed[index]
            ):
                return False
    return True


def _scheduled_bundle_payload(
    result: ScheduledOperatorPhaseConditionedObjectiveBounds,
    *,
    include_digest: bool,
) -> Dict[str, Any]:
    stop_policy = _scheduled_stop_policy_mapping(
        result.stop_policy, pattern_count=len(result.evaluation_schedule)
    )
    payload: Dict[str, Any] = {
        "schema": result.schema,
        "parent_semantic_digest": result.parent_semantic_digest,
        "stable_bit_ids": result.stable_bit_ids,
        "canonical_patterns": result.canonical_patterns,
        "evaluation_schedule": result.evaluation_schedule,
        "stop_policy_sha256": stop_policy.get("policy_sha256"),
        "certificate_sha256": tuple(
            item.certificate_sha256 for item in result.certificates
        ),
        "telemetry_sha256": result.telemetry.get("telemetry_sha256"),
        "receipt_sha256": result.receipt.get("receipt_sha256"),
        "full_parent_lp_called": result.full_parent_lp_called,
        "proof_authority": result.proof_authority,
        "verdict_authority": result.verdict_authority,
    }
    if include_digest:
        payload["bundle_sha256"] = result.bundle_sha256
    return payload


def _scheduled_receipt(
    *,
    parent_semantic_digest: str,
    stable_bit_ids: Tuple[int, ...],
    canonical_patterns: Tuple[SignedPattern, ...],
    evaluation_schedule: Tuple[SignedPattern, ...],
    stop_policy: OperatorPhaseConditionedScheduledStopPolicy,
    certificates: Tuple[
        OperatorPhaseConditionedObjectiveBoundCertificate, ...
    ],
    telemetry: Mapping[str, Any],
) -> Mapping[str, Any]:
    stop_policy_mapping = _scheduled_stop_policy_mapping(
        stop_policy, pattern_count=len(evaluation_schedule)
    )
    schedule_sha256 = _canonical_sha256(
        {
            "schema": _SCHEDULED_COMPLETE_SCHEMA,
            "stable_bit_ids": stable_bit_ids,
            "canonical_patterns": canonical_patterns,
            "evaluation_schedule": evaluation_schedule,
        }
    )
    payload: Dict[str, Any] = {
        "schema": _SCHEDULED_RECEIPT_SCHEMA,
        "status": "complete_canonical_cover_built_in_registered_schedule",
        "candidate_only": True,
        "full_parent_lp_called": False,
        "proof_authority": False,
        "verdict_authority": False,
        "public_sha_role": "structural_self_consistency_only",
        "provenance_authority": False,
        "authenticity_authority": False,
        "future_live_owner_anchor_required": True,
        "candidate_solver_status_authority": False,
        "candidate_solver_objective_authority": False,
        "single_shared_verified_context": True,
        "single_caller_owned_absolute_deadline": True,
        "complete_schedule_permutation_required": True,
        "output_restored_to_canonical_pattern_order": True,
        "no_partial_output_on_failure_or_policy_stop": True,
        "external_pattern_bounds_bound": 0,
        "live_replay_required_before_external_binding": True,
        "parent_semantic_digest": parent_semantic_digest,
        "stable_bit_ids": stable_bit_ids,
        "canonical_patterns": canonical_patterns,
        "evaluation_schedule": evaluation_schedule,
        "evaluation_schedule_sha256": schedule_sha256,
        "stop_policy": stop_policy_mapping,
        "stop_policy_sha256": stop_policy_mapping.get("policy_sha256"),
        "certificate_sha256": tuple(
            item.certificate_sha256 for item in certificates
        ),
        "telemetry": telemetry,
        "telemetry_sha256": telemetry.get("telemetry_sha256"),
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    frozen = _deep_freeze(payload)
    if type(frozen) is not MappingProxyType:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_receipt_freeze_failed"
        )
    return frozen


def _scheduled_stop_payload(
    record: OperatorPhaseConditionedScheduledStopRecord,
    *,
    include_digest: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for name in (
        OperatorPhaseConditionedScheduledStopRecord.__dataclass_fields__
    ):
        if name == "record_sha256":
            continue
        if name == "stop_policy":
            payload[name] = _scheduled_stop_policy_payload(
                record.stop_policy,
                pattern_count=len(record.evaluation_schedule),
                include_digest=True,
            )
        else:
            payload[name] = getattr(record, name)
    if include_digest:
        payload["record_sha256"] = record.record_sha256
    return payload


def _make_scheduled_stop_record(
    *,
    reason: str,
    parent_semantic_digest: str,
    stable_bit_ids: Tuple[int, ...],
    evaluation_schedule: Tuple[SignedPattern, ...],
    stop_policy: OperatorPhaseConditionedScheduledStopPolicy,
    triggering_schedule_index: int,
    triggering_pattern: SignedPattern,
    completed_internal_pattern_count: int,
    strict_upper_threshold: Optional[Fraction],
    observed_upper_exact: Optional[Fraction],
    telemetry: Mapping[str, Any],
) -> OperatorPhaseConditionedScheduledStopRecord:
    provisional = OperatorPhaseConditionedScheduledStopRecord(
        schema=_SCHEDULED_STOP_SCHEMA,
        status="stopped_by_non_authoritative_policy",
        reason=reason,
        parent_semantic_digest=parent_semantic_digest,
        stable_bit_ids=stable_bit_ids,
        evaluation_schedule=evaluation_schedule,
        stop_policy=stop_policy,
        triggering_schedule_index=triggering_schedule_index,
        triggering_pattern=triggering_pattern,
        completed_internal_pattern_count=completed_internal_pattern_count,
        strict_upper_threshold=strict_upper_threshold,
        observed_upper_exact=observed_upper_exact,
        telemetry=telemetry,
        record_sha256="",
        partial_certificates_returned=False,
        external_pattern_bounds_bound=0,
        full_parent_lp_called=False,
        proof_authority=False,
        verdict_authority=False,
        structural_self_consistency_only=True,
        provenance_authority=False,
        authenticity_authority=False,
        future_live_owner_anchor_required=True,
    )
    return OperatorPhaseConditionedScheduledStopRecord(
        **{
            **provisional.__dict__,
            "record_sha256": _canonical_sha256(
                _scheduled_stop_payload(provisional, include_digest=False)
            ),
        }
    )


def _build_scheduled_complete_operator_phase_conditioned_objective_bounds_until(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    evaluation_schedule: Tuple[SignedPattern, ...],
    deadline: float,
    stop_policy: OperatorPhaseConditionedScheduledStopPolicy = (
        OperatorPhaseConditionedScheduledStopPolicy()
    ),
    candidate_timeout_seconds: float = _DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
) -> ScheduledOperatorPhaseConditionedObjectiveBounds:
    """Run a complete permutation on one context and return canonical order."""

    stable_ids = _strict_stable_bit_ids(stable_bit_ids)
    schedule = _strict_evaluation_schedule(
        evaluation_schedule, k=len(stable_ids)
    )
    policy = _strict_scheduled_stop_policy(
        stop_policy, pattern_count=len(schedule)
    )
    if (
        isinstance(deadline, bool)
        or type(deadline) not in {int, float}
        or not math.isfinite(float(deadline))
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_bundle_absolute_deadline_invalid"
        )
    bundle_deadline = float(deadline)
    if time.monotonic() >= bundle_deadline:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_bundle_absolute_deadline_expired_no_partial_output"
        )
    candidate_timeout = _strict_timeout(
        candidate_timeout_seconds,
        name="scheduled_candidate_timeout_seconds",
    )
    counters = _ScheduledBuildTelemetry()
    started = time.monotonic()
    context_started = time.monotonic()
    shared = _prepare_verified_bundle_context(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_bit_ids=stable_ids,
        deadline=bundle_deadline,
        telemetry=counters,
    )
    counters.context_seconds = float(time.monotonic() - context_started)
    completed: list[
        OperatorPhaseConditionedObjectiveBoundCertificate
    ] = []
    for schedule_index, pattern in enumerate(schedule):
        remaining = bundle_deadline - time.monotonic()
        if remaining <= 0.0:
            completed.clear()
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_bundle_deadline_exhausted_no_partial_output"
            )
        counters.patterns_started += 1
        pattern_started = time.monotonic()
        certificate = _build_bound_from_verified_context(
            shared,
            selection,
            pattern=pattern,
            deadline=bundle_deadline,
            candidate_timeout_seconds=min(
                candidate_timeout,
                max(1.0e-6, remaining / 2.0),
            ),
            max_candidate_active_columns=(
                _DEFAULT_MAX_CANDIDATE_ACTIVE_COLUMNS
            ),
            max_candidate_dense_entries=(
                _DEFAULT_MAX_CANDIDATE_DENSE_ENTRIES
            ),
            telemetry=counters,
        )
        counters.pattern_seconds.append(
            float(time.monotonic() - pattern_started)
        )
        observed = Fraction.from_float(certificate.upper_stored)
        counters.patterns_completed += 1
        counters.completed_patterns.append(pattern)
        counters.observed_upper_exact.append(observed)
        completed.append(certificate)
        threshold_stop = bool(
            schedule_index in policy.threshold_pattern_indices
            and policy.strict_upper_threshold is not None
            and observed > policy.strict_upper_threshold
        )
        unconditional_stop = (
            schedule_index in policy.stop_after_pattern_indices
        )
        if threshold_stop or unconditional_stop:
            completed_internal_pattern_count = len(completed)
            _terminal_parent_seal(
                shared, deadline=bundle_deadline, telemetry=counters
            )
            completed.clear()
            certificate = None
            _check_deadline(
                bundle_deadline, stage="scheduled_policy_stop_return"
            )
            counters.total_seconds = float(time.monotonic() - started)
            telemetry = _scheduled_telemetry_mapping(
                counters,
                stable_bits=len(stable_ids),
                expected_pattern_count=len(schedule),
                status="stopped_by_policy",
            )
            record = _make_scheduled_stop_record(
                reason=(
                    "strict_upper_threshold_exceeded"
                    if threshold_stop
                    else "stop_after_pattern_index_reached"
                ),
                parent_semantic_digest=shared.parent_semantic_digest,
                stable_bit_ids=stable_ids,
                evaluation_schedule=schedule,
                stop_policy=policy,
                triggering_schedule_index=schedule_index,
                triggering_pattern=pattern,
                completed_internal_pattern_count=(
                    completed_internal_pattern_count
                ),
                strict_upper_threshold=(
                    policy.strict_upper_threshold
                    if threshold_stop
                    else None
                ),
                observed_upper_exact=observed,
                telemetry=telemetry,
            )
            raise OperatorPhaseConditionedScheduledStop(record)
        if time.monotonic() >= bundle_deadline:
            completed.clear()
            raise OperatorPhaseConditionedObjectiveBoundError(
                "scheduled_bundle_deadline_exhausted_no_partial_output"
            )

    by_pattern = {item.pattern: item for item in completed}
    canonical = _canonical_patterns(len(stable_ids))
    if len(by_pattern) != len(canonical) or set(by_pattern) != set(canonical):
        completed.clear()
        raise OperatorPhaseConditionedObjectiveBoundError(
            "scheduled_complete_pattern_production_internal_mismatch"
        )
    certificates = tuple(by_pattern[pattern] for pattern in canonical)
    _terminal_parent_seal(
        shared, deadline=bundle_deadline, telemetry=counters
    )
    counters.total_seconds = float(time.monotonic() - started)
    telemetry = _scheduled_telemetry_mapping(
        counters,
        stable_bits=len(stable_ids),
        expected_pattern_count=len(canonical),
        status="complete",
    )
    receipt = _scheduled_receipt(
        parent_semantic_digest=shared.parent_semantic_digest,
        stable_bit_ids=stable_ids,
        canonical_patterns=canonical,
        evaluation_schedule=schedule,
        stop_policy=policy,
        certificates=certificates,
        telemetry=telemetry,
    )
    provisional = ScheduledOperatorPhaseConditionedObjectiveBounds(
        schema=_SCHEDULED_COMPLETE_SCHEMA,
        parent_semantic_digest=shared.parent_semantic_digest,
        stable_bit_ids=stable_ids,
        canonical_patterns=canonical,
        evaluation_schedule=schedule,
        stop_policy=policy,
        certificates=certificates,
        telemetry=telemetry,
        receipt=receipt,
        bundle_sha256="",
        full_parent_lp_called=False,
        proof_authority=False,
        verdict_authority=False,
    )
    result = ScheduledOperatorPhaseConditionedObjectiveBounds(
        **{
            **provisional.__dict__,
            "bundle_sha256": _canonical_sha256(
                _scheduled_bundle_payload(
                    provisional, include_digest=False
                )
            ),
        }
    )
    _check_deadline(bundle_deadline, stage="scheduled_bundle_return")
    return result


def build_scheduled_complete_operator_phase_conditioned_objective_bounds(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    evaluation_schedule: Tuple[SignedPattern, ...],
    deadline: float,
    stop_policy: OperatorPhaseConditionedScheduledStopPolicy = (
        OperatorPhaseConditionedScheduledStopPolicy()
    ),
    candidate_timeout_seconds: float = _DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
) -> ScheduledOperatorPhaseConditionedObjectiveBounds:
    """Public absolute-deadline scheduled producer; never returns partials."""

    return _build_scheduled_complete_operator_phase_conditioned_objective_bounds_until(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_bit_ids=stable_bit_ids,
        evaluation_schedule=evaluation_schedule,
        deadline=deadline,
        stop_policy=stop_policy,
        candidate_timeout_seconds=candidate_timeout_seconds,
    )


def verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
    result: Any,
) -> bool:
    """Digest/shape replay only; it grants no live or proof authority."""

    try:
        if (
            type(result)
            is not ScheduledOperatorPhaseConditionedObjectiveBounds
            or result.schema != _SCHEDULED_COMPLETE_SCHEMA
            or result.full_parent_lp_called is not False
            or result.proof_authority is not False
            or result.verdict_authority is not False
            or not _valid_sha256(result.parent_semantic_digest)
            or not _valid_sha256(result.bundle_sha256)
            or type(result.telemetry) is not MappingProxyType
            or type(result.receipt) is not MappingProxyType
        ):
            return False
        stable_ids = _strict_stable_bit_ids(result.stable_bit_ids)
        canonical = _canonical_patterns(len(stable_ids))
        schedule = _strict_evaluation_schedule(
            result.evaluation_schedule, k=len(stable_ids)
        )
        stop_policy = _strict_scheduled_stop_policy(
            result.stop_policy, pattern_count=len(schedule)
        )
        if (
            result.canonical_patterns != canonical
            or type(result.certificates) is not tuple
            or len(result.certificates) != len(canonical)
            or tuple(item.pattern for item in result.certificates)
            != canonical
        ):
            return False
        checked = tuple(
            _strict_certificate_fields(item) for item in result.certificates
        )
        if any(
            item.parent_semantic_digest != result.parent_semantic_digest
            or item.stable_bit_ids != stable_ids
            or not _receipt_payload_matches(item)
            or _canonical_sha256(_certificate_payload(item))
            != item.certificate_sha256
            for item in checked
        ):
            return False
        telemetry = result.telemetry
        by_pattern = {item.pattern: item for item in checked}
        execution_certificates = tuple(
            by_pattern[pattern] for pattern in schedule
        )
        if (
            not _verify_scheduled_telemetry_structure(
                telemetry,
                stable_ids=stable_ids,
                evaluation_schedule=schedule,
                status="complete",
                execution_certificates=execution_certificates,
            )
            or _first_scheduled_policy_trigger(
                stop_policy,
                tuple(
                    Fraction.from_float(item.upper_stored)
                    for item in execution_certificates
                ),
                pattern_count=len(schedule),
            )
            is not None
        ):
            return False
        expected_receipt = _scheduled_receipt(
            parent_semantic_digest=result.parent_semantic_digest,
            stable_bit_ids=stable_ids,
            canonical_patterns=canonical,
            evaluation_schedule=schedule,
            stop_policy=stop_policy,
            certificates=checked,
            telemetry=telemetry,
        )
        return bool(
            _is_strict_deep_frozen_canonical_payload(result.receipt)
            and set(result.receipt.keys())
            == set(expected_receipt.keys())
            and
            _canonical_form(expected_receipt)
            == _canonical_form(result.receipt)
            and _canonical_sha256(
                _scheduled_bundle_payload(result, include_digest=False)
            )
            == result.bundle_sha256
        )
    except (
        OperatorPhaseConditionedObjectiveBoundError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def verify_operator_phase_conditioned_scheduled_stop_record(
    record: Any,
) -> bool:
    """Structural replay for a non-authoritative, certificate-free stop."""

    try:
        if (
            type(record) is not OperatorPhaseConditionedScheduledStopRecord
            or record.schema != _SCHEDULED_STOP_SCHEMA
            or record.status != "stopped_by_non_authoritative_policy"
            or record.reason
            not in {
                "strict_upper_threshold_exceeded",
                "stop_after_pattern_index_reached",
            }
            or not _valid_sha256(record.parent_semantic_digest)
            or not _valid_sha256(record.record_sha256)
            or record.partial_certificates_returned is not False
            or record.external_pattern_bounds_bound != 0
            or record.full_parent_lp_called is not False
            or record.proof_authority is not False
            or record.verdict_authority is not False
            or record.structural_self_consistency_only is not True
            or record.provenance_authority is not False
            or record.authenticity_authority is not False
            or record.future_live_owner_anchor_required is not True
            or type(record.telemetry) is not MappingProxyType
        ):
            return False
        stable_ids = _strict_stable_bit_ids(record.stable_bit_ids)
        schedule = _strict_evaluation_schedule(
            record.evaluation_schedule, k=len(stable_ids)
        )
        stop_policy = _strict_scheduled_stop_policy(
            record.stop_policy, pattern_count=len(schedule)
        )
        index = record.triggering_schedule_index
        if (
            type(index) is not int
            or index < 0
            or index >= len(schedule)
            or record.triggering_pattern != schedule[index]
            or record.completed_internal_pattern_count != index + 1
            or not _verify_scheduled_telemetry_structure(
                record.telemetry,
                stable_ids=stable_ids,
                evaluation_schedule=schedule,
                status="stopped_by_policy",
            )
        ):
            return False
        observed_trace = record.telemetry.get(
            "observed_upper_exact_in_execution_order"
        )
        first_trigger = _first_scheduled_policy_trigger(
            stop_policy,
            observed_trace,
            pattern_count=len(schedule),
        )
        if (
            first_trigger != (index, record.reason)
            or record.observed_upper_exact != observed_trace[index]
        ):
            return False
        if record.reason == "strict_upper_threshold_exceeded":
            if (
                index not in stop_policy.threshold_pattern_indices
                or record.strict_upper_threshold
                != stop_policy.strict_upper_threshold
                or type(record.strict_upper_threshold) is not Fraction
                or type(record.observed_upper_exact) is not Fraction
                or record.observed_upper_exact
                <= record.strict_upper_threshold
            ):
                return False
        elif (
            index not in stop_policy.stop_after_pattern_indices
            or record.strict_upper_threshold is not None
        ):
            return False
        return bool(
            _canonical_sha256(
                _scheduled_stop_payload(record, include_digest=False)
            )
            == record.record_sha256
        )
    except (
        OperatorPhaseConditionedObjectiveBoundError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _replay_complete_operator_phase_conditioned_objective_bounds_until(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    certificates: Tuple[
        OperatorPhaseConditionedObjectiveBoundCertificate, ...
    ],
    *,
    deadline: float,
) -> ReplayedOperatorPhaseConditionedObjectiveBounds:
    """Live-replay a complete cover under a caller-owned absolute deadline.

    No handle is constructed until every pattern has replayed successfully.
    Thus a timeout or malformed member cannot expose a partial cover.
    """

    if type(certificates) is not tuple or not certificates:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "complete_replay_certificates_not_nonempty_tuple"
        )
    if type(deadline) not in {int, float} or not math.isfinite(float(deadline)):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "complete_replay_absolute_deadline_invalid"
        )
    replay_deadline = float(deadline)
    first = _strict_certificate_fields(certificates[0])
    stable_ids = _strict_stable_bit_ids(first.stable_bit_ids)
    expected_patterns = tuple(
        itertools.product((-1, 1), repeat=len(stable_ids))
    )
    if any(
        type(item)
        is not OperatorPhaseConditionedObjectiveBoundCertificate
        for item in certificates
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "complete_replay_member_wrong_type"
        )
    checked_certificates = tuple(
        _strict_certificate_fields(item) for item in certificates
    )
    if (
        len(certificates) != len(expected_patterns)
        or tuple(item.pattern for item in checked_certificates)
        != expected_patterns
        or len({item.certificate_sha256 for item in checked_certificates})
        != len(certificates)
    ):
        raise OperatorPhaseConditionedObjectiveBoundError(
            "complete_replay_pattern_cover_noncanonical"
        )
    shared = _prepare_verified_bundle_context(
        build,
        rivals,
        selection,
        focused_rival_id=first.rival_id,
        stable_bit_ids=stable_ids,
        deadline=replay_deadline,
    )
    for checked in checked_certificates:
        remaining = replay_deadline - time.monotonic()
        if remaining <= 0.0:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "complete_replay_deadline_exhausted_no_partial_output"
            )
        if (
            checked.stable_bit_ids != stable_ids
            or checked.parent_semantic_digest
            != first.parent_semantic_digest
            or checked.rival_id != first.rival_id
            or checked.rival_binding_digest
            != first.rival_binding_digest
            or checked.objective_binding.objective_binding_sha256
            != first.objective_binding.objective_binding_sha256
            or checked.objective_envelope_sha256
            != first.objective_envelope_sha256
            or not _verify_bound_from_verified_context(
                shared,
                selection,
                checked,
                deadline=replay_deadline,
            )
        ):
            raise OperatorPhaseConditionedObjectiveBoundError(
                "complete_replay_member_failed_live_verification"
            )
        if time.monotonic() >= replay_deadline:
            raise OperatorPhaseConditionedObjectiveBoundError(
                "complete_replay_deadline_exhausted_no_partial_output"
            )

    _terminal_parent_seal(shared, deadline=replay_deadline)

    # Construct the opaque core handles only after the all-pattern barrier.
    if time.monotonic() >= replay_deadline:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "complete_replay_deadline_exhausted_no_partial_output"
        )
    pattern_bounds = tuple(
        bind_external_pattern_upper_bound(
            assignments=certificate.assignments,
            upper_exact=Fraction.from_float(certificate.upper_stored),
            upper_stored=certificate.upper_stored,
            parent_semantic_digest=certificate.parent_semantic_digest,
            objective_binding_sha256=(
                certificate.objective_binding.objective_binding_sha256
            ),
            certificate_schema=certificate.schema,
            certificate_sha256=certificate.certificate_sha256,
            upstream_proof_authority=True,
            independently_certified=True,
        )
        for certificate in checked_certificates
    )
    baseline = max(
        certificate.upper_stored for certificate in checked_certificates
    )
    bundle_payload = {
        "schema": _REPLAY_BUNDLE_SCHEMA,
        "parent_semantic_digest": first.parent_semantic_digest,
        "stable_bit_ids": stable_ids,
        "verified_context_sha256": shared.verified_context_sha256,
        "objective_binding_sha256": (
            first.objective_binding.objective_binding_sha256
        ),
        "objective_envelope_sha256": first.objective_envelope_sha256,
        "certificate_sha256": tuple(
            item.certificate_sha256 for item in checked_certificates
        ),
        "external_descriptor_sha256": tuple(
            item.descriptor_sha256 for item in pattern_bounds
        ),
        "baseline_upper_stored": baseline,
        "complete_pattern_cover": True,
        "no_partial_output_on_failure": True,
        "single_absolute_replay_deadline": True,
        "context_preparation_count": 1,
        "selection_verification_count": 1,
        "exact_objective_expansion_count": 1,
        "objective_envelope_formation_count": 1,
        "objective_generator_validation_pass_count": 1,
        "objective_source_hash_pass_count": 1,
        "objective_exact_expansion_pass_count": 1,
        "objective_binding_built_from_sealed_exact_material": True,
        "preformed_checker_generator_source_read_count": 0,
        "preformed_checker_envelope_rehash_bytes": 0,
        "global_checker_bundle_evaluation_count": 1,
        "global_checker_reused_per_pattern": True,
        "objective_envelope_production_ready": False,
        "context_semantic_digest_count": 1,
        "terminal_parent_semantic_digest_count": 1,
        "source_row_hashing_pass_count": 1,
        "proof_authority": True,
    }
    bundle_sha = _canonical_sha256(bundle_payload)
    receipt_payload: Dict[str, Any] = {
        **bundle_payload,
        "replay_bundle_sha256": bundle_sha,
        "live_selection_reverified_per_pattern": False,
        "live_rows_reextracted_per_pattern": False,
        "shared_immutable_verified_context": True,
        "split_checker_replayed_per_pattern": True,
        "candidate_solver_authority": False,
        "verdict_authority": False,
        "proof_authority_scope": (
            "live_split_checker_issuer_for_external_pattern_bounds_only"
        ),
    }
    receipt_payload["receipt_sha256"] = _canonical_sha256(receipt_payload)
    result = ReplayedOperatorPhaseConditionedObjectiveBounds(
        schema=_REPLAY_BUNDLE_SCHEMA,
        parent_semantic_digest=first.parent_semantic_digest,
        stable_bit_ids=stable_ids,
        objective_binding=first.objective_binding,
        pattern_bounds=pattern_bounds,
        baseline_upper_stored=baseline,
        certificate_sha256=tuple(
            item.certificate_sha256 for item in checked_certificates
        ),
        replay_bundle_sha256=bundle_sha,
        receipt=MappingProxyType(receipt_payload),
        proof_authority=True,
    )
    if time.monotonic() >= replay_deadline:
        raise OperatorPhaseConditionedObjectiveBoundError(
            "complete_replay_deadline_exhausted_no_partial_output"
        )
    return result


def replay_complete_operator_phase_conditioned_objective_bounds(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    certificates: Tuple[
        OperatorPhaseConditionedObjectiveBoundCertificate, ...
    ],
    *,
    certificate_timeout_seconds: float = (
        _DEFAULT_CERTIFICATE_TIMEOUT_SECONDS
    ),
) -> ReplayedOperatorPhaseConditionedObjectiveBounds:
    """Duration wrapper around the shared absolute-deadline replayer."""

    total_timeout = _strict_timeout(
        certificate_timeout_seconds,
        name="complete_replay_timeout_seconds",
    )
    return _replay_complete_operator_phase_conditioned_objective_bounds_until(
        build,
        rivals,
        selection,
        certificates,
        deadline=time.monotonic() + total_timeout,
    )


__all__ = [
    "OperatorPhaseConditionedObjectiveBoundCertificate",
    "OperatorPhaseConditionedObjectiveBoundError",
    "OperatorPhaseConditionedScheduledStop",
    "OperatorPhaseConditionedScheduledStopPolicy",
    "OperatorPhaseConditionedScheduledStopRecord",
    "ReplayedOperatorPhaseConditionedObjectiveBounds",
    "ScheduledOperatorPhaseConditionedObjectiveBounds",
    "build_complete_operator_phase_conditioned_objective_bounds",
    "build_operator_phase_conditioned_objective_bound",
    "build_scheduled_complete_operator_phase_conditioned_objective_bounds",
    "replay_complete_operator_phase_conditioned_objective_bounds",
    "verify_operator_phase_conditioned_scheduled_stop_record",
    "verify_operator_phase_conditioned_objective_bound",
    "verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure",
]
