#!/usr/bin/env python3
"""Pair-first, verdict-free K3 PCOH build diagnostic.

This module is deliberately separate from the frozen K2 build-only route.  It
selects one third exact-ReLU bit with exact dyadic arithmetic, proves the
complete signed-pair cover before choosing an evaluation schedule, and then
uses the scheduled complete conditional-bound producer.  A scheduled policy
stop exposes no partial certificates and never reaches fresh materialization.

The only successful terminal object is a receipt-only diagnostic.  It owns no
``SparseHZono`` or ``OperatorHZBuild`` and has no proof, verdict, ground-truth,
or full-parent-LP authority.  Same-process verification additionally requires
an identity anchor held in a weak registry; detached JSON verification is only
structural and requires a separately retained digest anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
import threading
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import weakref

import numpy as np

import act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_fresh_materializer as _fresh_materializer_module

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    ExactDyadicRivalCoefficient,
    OperatorExactReLUPhaseMapping,
    OperatorExactReLUPhaseSelection,
    verify_operator_exact_relu_property_phase_selection,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.operator_phase_conditioned_build_only import (
    PhaseConditionedBuildOnlyError,
    _live_resource_snapshot,
    _payload_bytes,
    _resource_postflight,
    _resource_preflight,
    _shape,
    _sparse_margin_preflight,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
    OperatorPhaseConditionedScheduledStop,
    OperatorPhaseConditionedScheduledStopPolicy,
    ScheduledOperatorPhaseConditionedObjectiveBounds,
    build_scheduled_complete_operator_phase_conditioned_objective_bounds,
    verify_operator_phase_conditioned_scheduled_stop_record,
    verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_fresh_materializer import (
    PCOHFreshBuildIssuance,
    PCOHFreshMaterializedTightnessSummary,
    PCOHFreshMaterializationCaps,
    consume_live_phase_conditioned_objective_hull_fresh_build,
    discard_live_phase_conditioned_objective_hull_fresh_build,
    issue_live_phase_conditioned_objective_hull_fresh_build,
    verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_row_materializer import (
    PCOHRowMaterializationCaps,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_pair_infeasibility import (
    PairInfeasibilityBundle,
    PairLocalCaps,
    run_phase_conditioned_pair_infeasibility_candidate,
    verify_phase_conditioned_pair_infeasibility_bundle,
)
from act.back_end.solver.solver_hz import SparseHZono


_SCHEMA = "act.hybridz_pcoh_k3_build_only_diagnostic.v1"
_RECEIPT_SCHEMA = "act.hybridz_pcoh_k3_build_only_receipt.v1"
_STOP_SCHEMA = "act.hybridz_pcoh_k3_build_only_stop.v1"
_STOP_RECEIPT_SCHEMA = "act.hybridz_pcoh_k3_build_only_stop_receipt.v1"
_RESOURCE_STOP_SCHEMA = "act.hybridz_pcoh_k3_build_only_resource_stop.v1"
_RESOURCE_STOP_RECEIPT_SCHEMA = (
    "act.hybridz_pcoh_k3_build_only_resource_stop_receipt.v1"
)
_RESOURCE_GATE_REJECTION_SCHEMA = (
    "act.hybridz_pcoh_k3_resource_gate_rejection.v1"
)
_RANKING_SCHEMA = "act.hybridz_pcoh_k3_third_bit_ranking.v1"
_SCHEDULE_SCHEMA = "act.hybridz_pcoh_k3_pair_first_schedule.v1"

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_K3_PATTERN_COUNT = 8
_K3_PAIR_QUERY_COUNT = 12
_K3_LOCAL_LP_UPPER_BOUND = 20
_K3_CONDITIONAL_CHECKER_UPPER_BOUND = 34
_K3_STRONG_TARGET = Fraction(
    191135223185129307,
    1759218604441600,
)

_RESOURCE_STOP_EXECUTION_KEYS = frozenset(
    {
        "pair_local_lp_actual_calls",
        "scheduled_local_lp_actual_calls",
        "local_lp_actual_calls",
        "local_lp_actual_call_cap",
        "pair_exact_conflict_certificates",
        "pair_exact_conflict_certificates_strictly_replayed",
        "scheduled_patterns_completed",
        "scheduled_candidate_dual_accepted",
        "scheduled_producer_checker_actual_calls",
        "fresh_live_replay_checker_actual_calls",
        "conditional_checker_actual_calls",
        "conditional_checker_actual_call_cap",
        "fresh_live_replay_performed",
        "scheduled_actual_call_site_counters",
        "scheduled_telemetry_sha256",
    }
)
_RESOURCE_STOP_EXECUTION_INT_KEYS = frozenset(
    {
        "pair_local_lp_actual_calls",
        "scheduled_local_lp_actual_calls",
        "local_lp_actual_calls",
        "local_lp_actual_call_cap",
        "pair_exact_conflict_certificates",
        "pair_exact_conflict_certificates_strictly_replayed",
        "scheduled_patterns_completed",
        "scheduled_candidate_dual_accepted",
        "scheduled_producer_checker_actual_calls",
        "fresh_live_replay_checker_actual_calls",
        "conditional_checker_actual_calls",
        "conditional_checker_actual_call_cap",
    }
)
_RESOURCE_CONDITION_KEYS = frozenset(
    {
        "current_plus_budget_strictly_below_process_cap",
        "entry_peak_not_above_process_cap",
        "mem_available_at_least_fixed_reserve",
        "cgroup_unbounded_or_has_fixed_reserve",
    }
)
_RESOURCE_PASS_RECEIPT_KEYS = frozenset(
    {
        "current_rss_bytes",
        "peak_rss_bytes",
        "mem_available_bytes",
        "cgroup_limit_status",
        "cgroup_headroom_bytes",
        "static_additional_rss_budget_bytes",
        "forecast_rss_bytes",
        "max_process_rss_bytes",
        "min_mem_available_bytes",
        "min_cgroup_headroom_bytes",
        "conditions",
        "passed",
        "measurement_source",
        "caller_supplied",
    }
)
_RESOURCE_GATE_REJECTION_KEYS = frozenset(
    {
        "schema",
        "stage",
        "reason",
        "current_rss_bytes",
        "peak_rss_bytes",
        "mem_available_bytes",
        "cgroup_limit_status",
        "cgroup_headroom_bytes",
        "static_additional_rss_budget_bytes",
        "forecast_rss_bytes",
        "max_process_rss_bytes",
        "min_mem_available_bytes",
        "min_cgroup_headroom_bytes",
        "conditions",
        "failed_conditions",
        "measurement_source",
        "caller_supplied",
        "passed",
        "rejection_sha256",
    }
)
_RESOURCE_STOP_TIMING_KEYS = frozenset(
    {
        "pair_and_strict_replay_seconds",
        "scheduled_producer_seconds",
        "total_seconds",
    }
)
_RESOURCE_STOP_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "stage",
        "reason",
        "diagnostic_only",
        "candidate_only",
        "production_ready",
        "solver_handoff_ready",
        "partial_certificates_returned",
        "conditional_certificate_payload_returned",
        "fresh_issue_called",
        "fresh_build_returned",
        "fresh_descriptor_returned",
        "fresh_registry_entries_created",
        "fresh_registry_state_before",
        "fresh_registry_state_terminal",
        "ground_truth_parameter_accepted",
        "full_parent_lp_called",
        "proof_authority",
        "verdict_authority",
        "provenance_authority",
        "authenticity_authority",
        "source_semantic_digest",
        "terminal_source_semantic_digest",
        "source_terminal_semantic_seal_read_count",
        "source_dimensions",
        "source_payload_bytes",
        "focused_rival_id",
        "focused_rival_binding_digest",
        "selected_output_positions",
        "selected_generator_nonzeros",
        "retained_k2_stable_bit_ids",
        "stable_bit_ids",
        "third_stable_bit_id",
        "third_coefficient_exact",
        "preferred_third_phase",
        "ranking_sha256",
        "pair_bundle_sha256",
        "pair_query_count",
        "canonical_pattern_count",
        "active_pattern_mask",
        "certified_empty_pattern_count",
        "evaluation_schedule",
        "threshold_pattern_indices",
        "strong_target_exact",
        "strict_comparison",
        "schedule_sha256",
        "scheduled_bundle_completed",
        "scheduled_bundle_sha256",
        "completed_conditional_certificate_count",
        "execution_telemetry",
        "resource_entry_preflight",
        "resource_pre_s_preflight",
        "resource_pre_fresh_preflight",
        "resource_gate_rejection",
        "caps",
        "timings",
        "receipt_sha256",
    }
)

SignedPattern = Tuple[int, ...]


class PhaseConditionedK3BuildOnlyError(ValueError):
    """The K3 diagnostic failed closed without returning a partial cover."""


def _default_pair_caps() -> PairLocalCaps:
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


def _default_fresh_caps() -> PCOHFreshMaterializationCaps:
    return PCOHFreshMaterializationCaps(
        max_parent_variables=60_000,
        max_parent_rows=105_000,
        max_parent_nonzeros=11_000_000,
        max_parent_buffer_items=24_000_000,
        max_tag_bytes=8_000_000,
        max_registry_entries=1,
        capability_ttl_seconds=15.0,
        row_caps=PCOHRowMaterializationCaps(
            max_parent_continuous_columns=60_000,
            max_parent_binary_columns=4,
            max_eta_columns=8,
            max_rows=12,
            max_total_exact_nonzeros=70_000,
            max_exact_bits=4096,
        ),
    )


@dataclass(frozen=True)
class PCOHK3BuildOnlyCaps:
    """Fixed caps for the first pair-first K3 diagnostic profile."""

    max_stable_bits: int = 3
    max_selected_output_terms: int = 2
    max_selected_generator_nonzeros: int = 512
    max_source_payload_bytes: int = 160 * _MIB
    max_fresh_payload_bytes: int = 192 * _MIB
    max_fresh_payload_delta_bytes: int = 32 * _MIB
    static_additional_rss_budget_bytes: int = 384 * _MIB
    max_process_rss_bytes: int = 5 * _GIB // 2
    min_mem_available_bytes: int = 896 * _MIB
    min_cgroup_headroom_bytes: int = 896 * _MIB
    candidate_timeout_seconds: float = 1.0
    pair_caps: PairLocalCaps = field(default_factory=_default_pair_caps)
    fresh_caps: PCOHFreshMaterializationCaps = field(
        default_factory=_default_fresh_caps
    )


@dataclass(frozen=True)
class PCOHK3ThirdBitPlan:
    """Exact ranking result for the retained K2 pair plus one third bit."""

    retained_k2_stable_bit_ids: Tuple[int, int]
    stable_bit_ids: Tuple[int, int, int]
    third_stable_bit_id: int
    third_coefficient: Fraction
    preferred_third_phase: int
    preferred_phase_source: str
    ranking: Tuple[Tuple[int, Fraction], ...]
    ranking_sha256: str


@dataclass(frozen=True)
class PCOHK3PairFirstSchedule:
    """Canonical active mask and pair-first scheduled execution policy."""

    canonical_patterns: Tuple[SignedPattern, ...]
    active_pattern_mask: Tuple[bool, ...]
    evaluation_schedule: Tuple[SignedPattern, ...]
    threshold_pattern_indices: Tuple[int, ...]
    worst_k2_children: Tuple[SignedPattern, SignedPattern]
    stop_policy: OperatorPhaseConditionedScheduledStopPolicy
    schedule_sha256: str


@dataclass(frozen=True, eq=False)
class PCOHK3BuildOnlyDiagnostic:
    """Terminal receipt-only result; it owns no build or HZ."""

    schema: str
    status: str
    source_semantic_digest: str
    terminal_source_semantic_digest: str
    focused_rival_id: int
    retained_k2_stable_bit_ids: Tuple[int, int]
    stable_bit_ids: Tuple[int, int, int]
    third_stable_bit_id: int
    third_coefficient_exact: Tuple[int, int]
    preferred_third_phase: int
    ranking_sha256: str
    pair_bundle_sha256: str
    active_pattern_mask: Tuple[bool, ...]
    evaluation_schedule: Tuple[SignedPattern, ...]
    threshold_pattern_indices: Tuple[int, ...]
    scheduled_bundle_sha256: str
    conditional_certificate_sha256: Tuple[str, ...]
    fresh_issuance_sha256: str
    fresh_semantic_digest: str
    source_dimensions: Tuple[int, int, int, int, int]
    fresh_dimensions: Tuple[int, int, int, int, int]
    materialized_tightness_summary: Mapping[str, Any]
    execution_telemetry: Mapping[str, Any]
    receipt: Mapping[str, Any]
    diagnostic_sha256: str
    ground_truth_loaded: bool = False
    full_parent_lp_called: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "ground_truth_loaded",
            "full_parent_lp_called",
            "proof_authority",
            "verdict_authority",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"K3 diagnostic authority firewall:{name}")
        if type(self.materialized_tightness_summary) is not MappingProxyType:
            object.__setattr__(
                self,
                "materialized_tightness_summary",
                _deep_freeze_receipt(self.materialized_tightness_summary),
            )
        if type(self.execution_telemetry) is not MappingProxyType:
            object.__setattr__(
                self,
                "execution_telemetry",
                _deep_freeze_receipt(self.execution_telemetry),
            )
        if type(self.receipt) is not MappingProxyType:
            object.__setattr__(
                self, "receipt", _deep_freeze_receipt(self.receipt)
            )


@dataclass(frozen=True, eq=False)
class PCOHK3BuildOnlyStopDiagnostic:
    """Sealed S policy stop; no partial certificate or fresh object escapes."""

    schema: str
    status: str
    reason: str
    source_semantic_digest: str
    terminal_source_semantic_digest: str
    focused_rival_id: int
    retained_k2_stable_bit_ids: Tuple[int, int]
    stable_bit_ids: Tuple[int, int, int]
    third_stable_bit_id: int
    third_coefficient_exact: Tuple[int, int]
    preferred_third_phase: int
    ranking_sha256: str
    pair_bundle_sha256: str
    active_pattern_mask: Tuple[bool, ...]
    evaluation_schedule: Tuple[SignedPattern, ...]
    threshold_pattern_indices: Tuple[int, ...]
    scheduled_stop_record_sha256: str
    triggering_schedule_index: int
    triggering_pattern: SignedPattern
    observed_upper_exact: Tuple[int, int]
    execution_telemetry: Mapping[str, Any]
    receipt: Mapping[str, Any]
    stop_sha256: str
    partial_certificates_returned: bool = False
    fresh_issue_called: bool = False
    fresh_build_returned: bool = False
    ground_truth_loaded: bool = False
    full_parent_lp_called: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "partial_certificates_returned",
            "fresh_issue_called",
            "fresh_build_returned",
            "ground_truth_loaded",
            "full_parent_lp_called",
            "proof_authority",
            "verdict_authority",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"K3 stop authority firewall:{name}")
        if type(self.execution_telemetry) is not MappingProxyType:
            object.__setattr__(
                self,
                "execution_telemetry",
                _deep_freeze_receipt(self.execution_telemetry),
            )
        if type(self.receipt) is not MappingProxyType:
            object.__setattr__(
                self, "receipt", _deep_freeze_receipt(self.receipt)
            )


@dataclass(frozen=True, eq=False)
class PCOHK3BuildOnlyResourceStopDiagnostic:
    """Sealed resource refusal; no certificate, fresh build, or descriptor escapes."""

    schema: str
    status: str
    stage: str
    reason: str
    source_semantic_digest: str
    terminal_source_semantic_digest: str
    focused_rival_id: int
    retained_k2_stable_bit_ids: Tuple[int, int]
    stable_bit_ids: Tuple[int, int, int]
    third_stable_bit_id: int
    third_coefficient_exact: Tuple[int, int]
    preferred_third_phase: int
    ranking_sha256: str
    pair_bundle_sha256: str
    active_pattern_mask: Tuple[bool, ...]
    evaluation_schedule: Tuple[SignedPattern, ...]
    threshold_pattern_indices: Tuple[int, ...]
    scheduled_bundle_sha256: Optional[str]
    completed_conditional_certificate_count: int
    execution_telemetry: Mapping[str, Any]
    receipt: Mapping[str, Any]
    resource_stop_sha256: str
    partial_certificates_returned: bool = False
    conditional_certificate_payload_returned: bool = False
    fresh_issue_called: bool = False
    fresh_build_returned: bool = False
    fresh_descriptor_returned: bool = False
    ground_truth_loaded: bool = False
    full_parent_lp_called: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False
    provenance_authority: bool = False
    authenticity_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "partial_certificates_returned",
            "conditional_certificate_payload_returned",
            "fresh_issue_called",
            "fresh_build_returned",
            "fresh_descriptor_returned",
            "ground_truth_loaded",
            "full_parent_lp_called",
            "proof_authority",
            "verdict_authority",
            "provenance_authority",
            "authenticity_authority",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"K3 resource stop authority firewall:{name}")
        if type(self.execution_telemetry) is not MappingProxyType:
            object.__setattr__(
                self,
                "execution_telemetry",
                _deep_freeze_receipt(self.execution_telemetry),
            )
        if type(self.receipt) is not MappingProxyType:
            object.__setattr__(
                self, "receipt", _deep_freeze_receipt(self.receipt)
            )


PCOHK3BuildOnlyOutcome = Union[
    PCOHK3BuildOnlyDiagnostic,
    PCOHK3BuildOnlyStopDiagnostic,
    PCOHK3BuildOnlyResourceStopDiagnostic,
]


@dataclass(frozen=True)
class _FreshInspection:
    issuance_sha256: str
    fresh_semantic_digest: str
    fresh_dimensions: Tuple[int, int, int, int, int]
    fresh_payload_bytes: int
    fresh_payload_delta_bytes: int
    equality_rows_added: int
    upper_rows_added: int
    materialized_tightness_summary: Mapping[str, Any]
    materialized_tightness_summary_sha256: str
    live_verifier_valid_after_consume: bool
    issue_seconds: float
    consume_seconds: float


@dataclass(frozen=True)
class _RegisteredOutcome:
    process_id: int
    field_names: Tuple[str, ...]
    original_values: Tuple[Any, ...]


_OUTCOME_REGISTRY_LOCK = threading.Lock()
_OUTCOME_REGISTRY: "weakref.WeakKeyDictionary[Any, _RegisteredOutcome]" = (
    weakref.WeakKeyDictionary()
)
_FRESH_DISCARD_AUTHORITY = (
    discard_live_phase_conditioned_objective_hull_fresh_build
)


def _canonical_form(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PhaseConditionedK3BuildOnlyError(
                "canonical_nonfinite_float"
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
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PhaseConditionedK3BuildOnlyError(
                    "canonical_mapping_key_not_string"
                )
            output[key] = _canonical_form(item)
        return output
    raise PhaseConditionedK3BuildOnlyError(
        f"canonical_unsupported:{type(value).__name__}"
    )


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            _canonical_form(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PhaseConditionedK3BuildOnlyError(
            "canonical_encoding_failed"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _receipt_value(value: Any) -> Any:
    """Detach a trusted object into strict immutable receipt primitives."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PhaseConditionedK3BuildOnlyError(
                "receipt_nonfinite_float"
            )
        return value
    if type(value) is Fraction:
        return _fraction_pair(value)
    if isinstance(value, np.generic):
        return _receipt_value(value.item())
    if type(value) in {tuple, list}:
        return tuple(_receipt_value(item) for item in value)
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PhaseConditionedK3BuildOnlyError(
                    "receipt_mapping_key_not_string"
                )
            output[key] = _receipt_value(item)
        return output
    raise PhaseConditionedK3BuildOnlyError(
        f"receipt_unsupported:{type(value).__name__}"
    )


def _deep_freeze_receipt(value: Any) -> Any:
    detached = _receipt_value(value)

    def freeze(item: Any) -> Any:
        if type(item) is dict:
            return MappingProxyType(
                {key: freeze(value) for key, value in item.items()}
            )
        if type(item) is tuple:
            return tuple(freeze(value) for value in item)
        return item

    return freeze(detached)


def _builtin_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _builtin_value(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_builtin_value(item) for item in value]
    if type(value) is Fraction:
        return [value.numerator, value.denominator]
    if isinstance(value, np.generic):
        return _builtin_value(value.item())
    return value


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fixed_profile_types_exact(value: Any, fixed: Any) -> bool:
    if type(value) is not type(fixed):
        return False
    fields = getattr(type(fixed), "__dataclass_fields__", None)
    if fields is None:
        return bool(
            type(value) is not float
            or (math.isfinite(value) and math.isfinite(fixed))
        )
    field_names = tuple(fields)
    if tuple(vars(value)) != field_names or tuple(vars(fixed)) != field_names:
        return False
    return all(
        _fixed_profile_types_exact(
            getattr(value, name), getattr(fixed, name)
        )
        for name in field_names
    )


def _normalize_caps(value: Any) -> PCOHK3BuildOnlyCaps:
    fixed = PCOHK3BuildOnlyCaps()
    if type(value) is not PCOHK3BuildOnlyCaps:
        raise PhaseConditionedK3BuildOnlyError("caps_wrong_type")
    if not _fixed_profile_types_exact(value, fixed):
        raise PhaseConditionedK3BuildOnlyError(
            "k3_caps_field_type_or_shape_not_exact"
        )
    if value != fixed:
        raise PhaseConditionedK3BuildOnlyError(
            "k3_caps_must_match_fixed_profile"
        )
    if (
        value.max_stable_bits != 3
        or value.max_selected_output_terms != 2
        or value.pair_caps.max_stable_bits != 3
        or value.pair_caps.max_signed_pair_queries != 12
        or value.fresh_caps.row_caps.max_eta_columns != 8
        or value.fresh_caps.row_caps.max_rows != 12
        or type(value.candidate_timeout_seconds) is not float
        or not 0.0 < value.candidate_timeout_seconds <= 5.0
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "fixed_k3_caps_internal_invariant_failed"
        )
    # Never retain caller-owned dataclass/nested-cap identities across the
    # transaction; ``frozen=True`` is not an ABA or object.__setattr__ seal.
    return fixed


def _selection_caps_kwargs(
    selection: OperatorExactReLUPhaseSelection,
) -> Dict[str, Any]:
    caps = selection.caps
    return {
        "max_rivals": caps.max_rivals,
        "max_binaries": caps.max_binaries,
        "max_work_items": caps.max_work_items,
        "timeout_seconds": caps.timeout_seconds,
    }


def _fraction_pair(value: Fraction) -> Tuple[int, int]:
    if type(value) is not Fraction:
        raise PhaseConditionedK3BuildOnlyError("fraction_not_exact")
    return (value.numerator, value.denominator)


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise PhaseConditionedK3BuildOnlyError(
            f"deadline_exhausted:{stage}:no_partial_output"
        )


def _deadline(value: Any) -> float:
    if (
        isinstance(value, bool)
        or type(value) not in {int, float}
        or not math.isfinite(float(value))
        or time.monotonic() >= float(value)
    ):
        raise PhaseConditionedK3BuildOnlyError("absolute_deadline_invalid")
    return float(value)


def _focused_rival(
    rivals: Sequence[RivalSpec], focused_rival_id: int
) -> RivalSpec:
    if type(rivals) not in {tuple, list}:
        raise PhaseConditionedK3BuildOnlyError("rivals_not_sequence")
    matches = tuple(
        rival
        for rival in rivals
        if type(rival) is RivalSpec and rival.rival_id == focused_rival_id
    )
    if len(matches) != 1:
        raise PhaseConditionedK3BuildOnlyError(
            "focused_rival_not_unique"
        )
    return matches[0]


def _mapping_focused_coefficient(
    mapping: OperatorExactReLUPhaseMapping,
    *,
    focused_rival_id: int,
) -> Fraction:
    if type(mapping) is not OperatorExactReLUPhaseMapping:
        raise PhaseConditionedK3BuildOnlyError("mapping_wrong_type")
    matches = tuple(
        item
        for item in mapping.rival_coefficients
        if type(item) is ExactDyadicRivalCoefficient
        and item.rival_id == focused_rival_id
    )
    if len(matches) != 1:
        raise PhaseConditionedK3BuildOnlyError(
            "focused_mapping_coefficient_not_unique"
        )
    item = matches[0]
    exact = Fraction(item.numerator, item.denominator)
    if (
        item.denominator <= 0
        or exact.numerator != item.numerator
        or exact.denominator != item.denominator
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "focused_mapping_coefficient_noncanonical"
        )
    return exact


def select_k3_third_bit_plan(
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    retained_k2_stable_bit_ids: Tuple[int, int],
) -> PCOHK3ThirdBitPlan:
    """Choose the third bit by exact magnitude, then stable id."""

    if type(selection) is not OperatorExactReLUPhaseSelection:
        raise PhaseConditionedK3BuildOnlyError("selection_wrong_type")
    if (
        type(retained_k2_stable_bit_ids) is not tuple
        or len(retained_k2_stable_bit_ids) != 2
        or any(type(value) is not int or value < 0 for value in retained_k2_stable_bit_ids)
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "retained_k2_stable_bit_ids_invalid"
        )
    retained = tuple(retained_k2_stable_bit_ids)
    mapping_by_id = {
        mapping.stable_bcol_id: mapping for mapping in selection.mappings
    }
    canonical_ids = tuple(sorted(mapping_by_id))
    if (
        len(mapping_by_id) != len(selection.mappings)
        or len(canonical_ids) < 3
        or retained != canonical_ids[:2]
        or retained != tuple(sorted(retained))
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "retained_k2_ids_not_lowest_verified_pair"
        )
    ranked = []
    for stable_id in canonical_ids[2:]:
        coefficient = _mapping_focused_coefficient(
            mapping_by_id[stable_id],
            focused_rival_id=focused_rival_id,
        )
        ranked.append((stable_id, coefficient))
    ranked.sort(key=lambda item: (-abs(item[1]), item[0]))
    if not ranked:
        raise PhaseConditionedK3BuildOnlyError(
            "no_remaining_verified_mapping_for_third_bit"
        )
    third_id, coefficient = ranked[0]
    if coefficient > 0:
        preferred_phase = 1
        phase_source = "positive_exact_focused_coefficient"
    elif coefficient < 0:
        preferred_phase = -1
        phase_source = "negative_exact_focused_coefficient"
    else:
        preferred_phase = 1
        phase_source = "zero_coefficient_positive_tie"
    stable_ids = tuple(sorted((*retained, third_id)))
    if stable_ids[:2] != retained or len(stable_ids) != 3:
        raise PhaseConditionedK3BuildOnlyError(
            "third_bit_did_not_preserve_k2_prefix"
        )
    payload = {
        "schema": _RANKING_SCHEMA,
        "retained_k2_stable_bit_ids": retained,
        "stable_bit_ids": stable_ids,
        "third_stable_bit_id": third_id,
        "third_coefficient_exact": _fraction_pair(coefficient),
        "preferred_third_phase": preferred_phase,
        "preferred_phase_source": phase_source,
        "ranking": tuple(
            (stable_id, *_fraction_pair(value))
            for stable_id, value in ranked
        ),
        "ranking_rule": "abs_exact_dyadic_desc_then_stable_id_asc",
    }
    return PCOHK3ThirdBitPlan(
        retained_k2_stable_bit_ids=retained,
        stable_bit_ids=stable_ids,
        third_stable_bit_id=third_id,
        third_coefficient=coefficient,
        preferred_third_phase=preferred_phase,
        preferred_phase_source=phase_source,
        ranking=tuple(ranked),
        ranking_sha256=_canonical_sha256(payload),
    )


def build_k3_pair_first_schedule(
    pair_bundle: PairInfeasibilityBundle,
    *,
    preferred_third_phase: int,
) -> PCOHK3PairFirstSchedule:
    """Put the worst K2 descendants first and threshold every active row."""

    if (
        type(pair_bundle) is not PairInfeasibilityBundle
        or pair_bundle.status != "complete"
        or len(pair_bundle.stable_bit_ids) != 3
        or preferred_third_phase not in {-1, 1}
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "pair_bundle_or_preferred_phase_invalid"
        )
    canonical = tuple(
        tuple(int(value) for value in pattern)
        for pattern in itertools.product((-1, 1), repeat=3)
    )
    if (
        type(pair_bundle.coverage) is not tuple
        or tuple(item.pattern for item in pair_bundle.coverage) != canonical
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "pair_coverage_not_canonical_k3"
        )
    active_by_pattern: Dict[SignedPattern, bool] = {}
    for item in pair_bundle.coverage:
        if item.status == "certified_empty_by_pair":
            active_by_pattern[item.pattern] = False
        elif item.status == "not_certified_empty":
            active_by_pattern[item.pattern] = True
        else:
            raise PhaseConditionedK3BuildOnlyError(
                "pair_coverage_status_invalid"
            )
    if not any(active_by_pattern.values()):
        raise PhaseConditionedK3BuildOnlyError(
            "all_k3_patterns_certified_empty:no_verdict"
        )
    preferred = (1, 1, int(preferred_third_phase))
    opposite = (1, 1, -int(preferred_third_phase))
    children = (preferred, opposite)
    active_children = tuple(
        pattern for pattern in children if active_by_pattern[pattern]
    )
    empty_children = tuple(
        pattern for pattern in children if not active_by_pattern[pattern]
    )
    remaining_active = tuple(
        pattern
        for pattern in canonical
        if pattern not in children and active_by_pattern[pattern]
    )
    remaining_empty = tuple(
        pattern
        for pattern in canonical
        if pattern not in children and not active_by_pattern[pattern]
    )
    schedule = (
        active_children
        + empty_children
        + remaining_active
        + remaining_empty
    )
    if (
        len(schedule) != _K3_PATTERN_COUNT
        or len(set(schedule)) != _K3_PATTERN_COUNT
        or set(schedule) != set(canonical)
        or set(schedule[:2]) != set(children)
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "k3_schedule_internal_invariant_failed"
        )
    threshold_indices = tuple(
        index
        for index, pattern in enumerate(schedule)
        if active_by_pattern[pattern]
    )
    policy = OperatorPhaseConditionedScheduledStopPolicy(
        stop_after_pattern_indices=(),
        strict_upper_threshold=_K3_STRONG_TARGET,
        threshold_pattern_indices=threshold_indices,
    )
    active_mask = tuple(active_by_pattern[pattern] for pattern in canonical)
    payload = {
        "schema": _SCHEDULE_SCHEMA,
        "canonical_patterns": canonical,
        "active_pattern_mask": active_mask,
        "active_means": "not_certified_empty_by_exact_signed_pair",
        "evaluation_schedule": schedule,
        "threshold_pattern_indices": threshold_indices,
        "worst_k2_children": children,
        "strict_upper_threshold_exact": _fraction_pair(_K3_STRONG_TARGET),
        "strict_comparison": "observed_upper_exact_gt_target",
    }
    return PCOHK3PairFirstSchedule(
        canonical_patterns=canonical,
        active_pattern_mask=active_mask,
        evaluation_schedule=schedule,
        threshold_pattern_indices=threshold_indices,
        worst_k2_children=children,
        stop_policy=policy,
        schedule_sha256=_canonical_sha256(payload),
    )


def _strict_pair_bundle(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    bundle: Any,
    *,
    stable_ids: Tuple[int, int, int],
    deadline: float,
) -> PairInfeasibilityBundle:
    if (
        type(bundle) is not PairInfeasibilityBundle
        or bundle.status != "complete"
        or bundle.stable_bit_ids != stable_ids
        or bundle.proof_authority is not False
        or len(bundle.records) != _K3_PAIR_QUERY_COUNT
        or len(bundle.coverage) != _K3_PATTERN_COUNT
        or any(record.model_closed is not True for record in bundle.records)
        or any(record.local_model_rows != 6 for record in bundle.records)
        or any(
            record.local_model_columns != build.hz.n_cont + build.hz.n_bin
            for record in bundle.records
        )
        or not verify_phase_conditioned_pair_infeasibility_bundle(
            build,
            rivals,
            selection,
            bundle,
            deadline=deadline,
        )
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "k3_pair_bundle_complete_strict_replay_failed"
        )
    _check_deadline(deadline, "after_k3_pair_strict_replay")
    return bundle


def _scheduled_stop_record_payload(record: Any) -> Mapping[str, Any]:
    policy = record.stop_policy
    payload = {
        "schema": record.schema,
        "status": record.status,
        "reason": record.reason,
        "parent_semantic_digest": record.parent_semantic_digest,
        "stable_bit_ids": record.stable_bit_ids,
        "evaluation_schedule": record.evaluation_schedule,
        "stop_policy": {
            "stop_after_pattern_indices": policy.stop_after_pattern_indices,
            "strict_upper_threshold": policy.strict_upper_threshold,
            "threshold_pattern_indices": policy.threshold_pattern_indices,
        },
        "triggering_schedule_index": record.triggering_schedule_index,
        "triggering_pattern": record.triggering_pattern,
        "completed_internal_pattern_count": (
            record.completed_internal_pattern_count
        ),
        "strict_upper_threshold": record.strict_upper_threshold,
        "observed_upper_exact": record.observed_upper_exact,
        "telemetry": record.telemetry,
        "record_sha256": record.record_sha256,
        "partial_certificates_returned": (
            record.partial_certificates_returned
        ),
        "external_pattern_bounds_bound": (
            record.external_pattern_bounds_bound
        ),
        "full_parent_lp_called": record.full_parent_lp_called,
        "proof_authority": record.proof_authority,
        "verdict_authority": record.verdict_authority,
        "structural_self_consistency_only": (
            record.structural_self_consistency_only
        ),
        "provenance_authority": record.provenance_authority,
        "authenticity_authority": record.authenticity_authority,
        "future_live_owner_anchor_required": (
            record.future_live_owner_anchor_required
        ),
    }
    frozen = _deep_freeze_receipt(payload)
    if type(frozen) is not MappingProxyType:
        raise PhaseConditionedK3BuildOnlyError(
            "scheduled_stop_payload_freeze_failed"
        )
    return frozen


def _clear_exception_traceback(exc: BaseException) -> None:
    cursor = exc.__traceback__
    while cursor is not None:
        frame = cursor.tb_frame
        cursor = cursor.tb_next
        try:
            frame.clear()
        except RuntimeError:
            pass
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None


def _validate_scheduled_complete(
    result: Any,
    *,
    source_digest: str,
    stable_ids: Tuple[int, int, int],
    schedule: PCOHK3PairFirstSchedule,
) -> ScheduledOperatorPhaseConditionedObjectiveBounds:
    if (
        type(result) is not ScheduledOperatorPhaseConditionedObjectiveBounds
        or result.parent_semantic_digest != source_digest
        or result.stable_bit_ids != stable_ids
        or result.canonical_patterns != schedule.canonical_patterns
        or result.evaluation_schedule != schedule.evaluation_schedule
        or result.stop_policy != schedule.stop_policy
        or len(result.certificates) != _K3_PATTERN_COUNT
        or tuple(item.pattern for item in result.certificates)
        != schedule.canonical_patterns
        or not verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
            result
        )
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "scheduled_k3_complete_structure_failed"
        )
    telemetry = result.telemetry
    accepted = telemetry.get("candidate_dual_accepted")
    producer_checker = telemetry.get("split_checker_evaluations")
    linprog_actual = telemetry.get("linprog_actual_calls")
    if (
        telemetry.get("patterns_completed") != _K3_PATTERN_COUNT
        or telemetry.get("expected_pattern_count") != _K3_PATTERN_COUNT
        or telemetry.get("local_upper_rows") != 9
        or type(accepted) is not int
        or not 0 <= accepted <= _K3_PATTERN_COUNT
        or producer_checker != 1 + _K3_PATTERN_COUNT + accepted
        or type(linprog_actual) is not int
        or not 0 <= linprog_actual <= _K3_PATTERN_COUNT
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "scheduled_k3_actual_telemetry_failed"
        )
    return result


def _validate_scheduled_stop(
    stop: OperatorPhaseConditionedScheduledStop,
    *,
    source_digest: str,
    stable_ids: Tuple[int, int, int],
    schedule: PCOHK3PairFirstSchedule,
) -> Any:
    record = stop.record
    if (
        not verify_operator_phase_conditioned_scheduled_stop_record(record)
        or record.reason != "strict_upper_threshold_exceeded"
        or record.parent_semantic_digest != source_digest
        or record.stable_bit_ids != stable_ids
        or record.evaluation_schedule != schedule.evaluation_schedule
        or record.stop_policy != schedule.stop_policy
        or record.triggering_schedule_index
        not in schedule.threshold_pattern_indices
        or record.triggering_pattern
        != schedule.evaluation_schedule[record.triggering_schedule_index]
        or record.strict_upper_threshold != _K3_STRONG_TARGET
        or type(record.observed_upper_exact) is not Fraction
        or record.observed_upper_exact <= _K3_STRONG_TARGET
        or record.partial_certificates_returned is not False
        or record.external_pattern_bounds_bound != 0
        or record.full_parent_lp_called is not False
        or record.proof_authority is not False
        or record.verdict_authority is not False
        or record.structural_self_consistency_only is not True
        or record.provenance_authority is not False
        or record.authenticity_authority is not False
        or record.future_live_owner_anchor_required is not True
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "scheduled_k3_stop_record_failed_live_owner_binding"
        )
    return record


def _discard_unconsumed_fresh_issuance(issuance: Any) -> Tuple[bool, str]:
    """Try the observable binding first, then the captured cleanup authority."""

    detail = ""
    try:
        public_result = discard_live_phase_conditioned_objective_hull_fresh_build(
            issuance, issuance.capability
        )
    except BaseException as exc:
        public_result = False
        detail = "public_interrupted:" + type(exc).__name__
        _clear_exception_traceback(exc)
    if public_result is True:
        return True, detail
    if not detail:
        detail = "public_rejected"
    try:
        fallback_result = _FRESH_DISCARD_AUTHORITY(
            issuance, issuance.capability
        )
    except BaseException as exc:
        fallback_result = False
        fallback_detail = "fallback_interrupted:" + type(exc).__name__
        _clear_exception_traceback(exc)
    else:
        fallback_detail = (
            "fallback_succeeded"
            if fallback_result is True
            else "fallback_rejected"
        )
    return fallback_result is True, detail + ":" + fallback_detail


def _materialized_summary_payload(
    summary: PCOHFreshMaterializedTightnessSummary,
) -> Mapping[str, Any]:
    if type(summary) is not PCOHFreshMaterializedTightnessSummary:
        raise PhaseConditionedK3BuildOnlyError(
            "fresh_tightness_summary_wrong_type"
        )
    return _deep_freeze_receipt(
        {
            name: getattr(summary, name)
            for name in PCOHFreshMaterializedTightnessSummary.__dataclass_fields__
        }
    )


def _issue_consume_inspect_release(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_ids: Tuple[int, int, int],
    certificates: Tuple[Any, ...],
    pair_bundle: PairInfeasibilityBundle,
    active_pattern_mask: Tuple[bool, ...],
    deadline: float,
    caps: PCOHK3BuildOnlyCaps,
    source_dimensions: Tuple[int, int, int, int, int],
    source_payload_bytes: int,
    source_semantic_digest: str,
) -> Tuple[Optional[_FreshInspection], Optional[str]]:
    """Own the one-shot fresh capability and never return a build or HZ."""

    issuance = None
    fresh_build = None
    inspection = None
    error = None
    consumed = False
    started = time.monotonic()
    try:
        issuance = issue_live_phase_conditioned_objective_hull_fresh_build(
            build,
            rivals,
            selection,
            focused_rival_id=focused_rival_id,
            stable_bit_ids=stable_ids,
            conditional_certificates=certificates,
            pair_bundle=pair_bundle,
            deadline=deadline,
            caps=caps.fresh_caps,
        )
        issue_seconds = float(time.monotonic() - started)
        if (
            type(issuance) is not PCOHFreshBuildIssuance
            or issuance.parent_semantic_digest != source_semantic_digest
            or issuance.terminal_parent_semantic_digest
            != source_semantic_digest
            or issuance.proof_authority is not False
            or issuance.verdict_authority is not False
            or type(issuance.receipt) is not MappingProxyType
            or not _valid_sha256(issuance.issuance_sha256)
            or not verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
                issuance
            )
        ):
            raise PhaseConditionedK3BuildOnlyError(
                "fresh_live_strict_verification_failed"
            )
        _check_deadline(deadline, "after_fresh_live_strict_verification")
        receipt = issuance.receipt
        summary = issuance.materialized_tightness_summary
        summary_payload = _materialized_summary_payload(summary)
        registered_summary = receipt.get("materialized_tightness_summary")
        conditional_sha256 = tuple(
            certificate.certificate_sha256 for certificate in certificates
        )
        expected_patterns = tuple(
            tuple(pattern)
            for pattern in itertools.product((-1, 1), repeat=3)
        )
        if (
            type(registered_summary) is not MappingProxyType
            or _canonical_form(registered_summary)
            != _canonical_form(summary_payload)
            or summary.parent_semantic_digest != source_semantic_digest
            or summary.stable_bit_ids != stable_ids
            or summary.canonical_patterns != expected_patterns
            or summary.active_pattern_mask != active_pattern_mask
            or summary.conditional_certificate_sha256
            != conditional_sha256
            or summary.diagnostic_only is not True
            or summary.full_parent_lp_called is not False
            or summary.proof_authority is not False
            or summary.verdict_authority is not False
            or not _valid_sha256(summary.summary_sha256)
            or receipt.get("materialized_tightness_summary_sha256")
            != summary.summary_sha256
            or receipt.get("materialized_tightness_full_parent_lp_called")
            is not False
        ):
            raise PhaseConditionedK3BuildOnlyError(
                "fresh_tightness_cross_binding_failed"
            )
        fresh_build = consume_live_phase_conditioned_objective_hull_fresh_build(
            issuance, issuance.capability, deadline=deadline
        )
        consumed = True
        consume_seconds = float(time.monotonic() - started - issue_seconds)
        if (
            type(fresh_build) is not OperatorHZBuild
            or type(fresh_build.hz) is not SparseHZono
            or fresh_build.constructive_nonempty_seal is not None
            or fresh_build.property_upper_output is not False
            or fresh_build.property_upper_row_groups != ()
            or fresh_build.verified_preactivation_frame is not None
            or fresh_build.metadata.get("candidate_only") is not True
            or fresh_build.metadata.get("proof_authority") is not False
            or fresh_build.metadata.get("verdict_authority") is not False
            or fresh_build.metadata.get("production_ready") is not False
        ):
            raise PhaseConditionedK3BuildOnlyError(
                "fresh_build_authority_firewall_failed"
            )
        empty_count = active_pattern_mask.count(False)
        expected_dimensions = (
            source_dimensions[0],
            source_dimensions[1] + _K3_PATTERN_COUNT,
            source_dimensions[2],
            source_dimensions[3] + 4 + empty_count,
            source_dimensions[4] + 1,
        )
        fresh_dimensions = _shape(fresh_build.hz)
        source_payload_receipt = receipt.get("source_payload_bytes")
        fresh_payload = receipt.get("fresh_payload_bytes")
        fresh_delta = receipt.get("fresh_payload_delta_bytes")
        if (
            len(issuance.eta_col_ids) != _K3_PATTERN_COUNT
            or len(set(issuance.eta_col_ids)) != _K3_PATTERN_COUNT
            or len(issuance.equality_row_tags) != 4 + empty_count
            or len(issuance.upper_row_tags) != 1
            or fresh_dimensions != expected_dimensions
            or tuple(receipt.get("source_dimensions", ()))
            != source_dimensions
            or tuple(receipt.get("fresh_dimensions", ()))
            != fresh_dimensions
            or source_payload_receipt != source_payload_bytes
            or type(fresh_payload) is not int
            or type(fresh_delta) is not int
            or fresh_payload - source_payload_bytes != fresh_delta
            or fresh_payload > caps.max_fresh_payload_bytes
            or fresh_delta < 0
            or fresh_delta > caps.max_fresh_payload_delta_bytes
            or receipt.get("source_buffers_borrowed_by_fresh") is not False
            or receipt.get("fresh_buffers_readonly") is not True
            or receipt.get("uses_sparse_hstack") is not False
            or receipt.get("uses_sparse_vstack") is not False
            or receipt.get("constructive_nonempty_inherited") is not False
            or receipt.get("proof_authority") is not False
            or receipt.get("verdict_authority") is not False
            or receipt.get("production_ready") is not False
        ):
            raise PhaseConditionedK3BuildOnlyError(
                "fresh_shape_payload_or_lifecycle_failed"
            )
        fresh_digest = sparse_hz_semantic_digest(fresh_build.hz)
        if fresh_digest != issuance.fresh_semantic_digest:
            raise PhaseConditionedK3BuildOnlyError(
                "fresh_terminal_semantic_digest_mismatch"
            )
        live_after_consume = (
            verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
                issuance
            )
        )
        if live_after_consume is not False:
            raise PhaseConditionedK3BuildOnlyError(
                "fresh_capability_remained_live_after_consume"
            )
        inspection = _FreshInspection(
            issuance_sha256=issuance.issuance_sha256,
            fresh_semantic_digest=fresh_digest,
            fresh_dimensions=fresh_dimensions,
            fresh_payload_bytes=fresh_payload,
            fresh_payload_delta_bytes=fresh_delta,
            equality_rows_added=len(issuance.equality_row_tags),
            upper_rows_added=len(issuance.upper_row_tags),
            materialized_tightness_summary=summary_payload,
            materialized_tightness_summary_sha256=summary.summary_sha256,
            live_verifier_valid_after_consume=live_after_consume,
            issue_seconds=issue_seconds,
            consume_seconds=consume_seconds,
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}:{str(exc)[:240]}"
        _clear_exception_traceback(exc)
    finally:
        if issuance is not None and not consumed:
            cleaned, detail = _discard_unconsumed_fresh_issuance(issuance)
            if cleaned is not True:
                inspection = None
                error = (
                    str(error)[:180]
                    + ":fresh_registry_cleanup_failed:"
                    + detail[:160]
                )
            elif detail:
                error = (
                    str(error)[:180]
                    + ":fresh_registry_cleanup_recovered:"
                    + detail[:160]
                )
        fresh_build = None
        issuance = None
    return inspection, error


def _caps_payload(caps: PCOHK3BuildOnlyCaps) -> Mapping[str, Any]:
    return _deep_freeze_receipt(
        {
            "max_stable_bits": caps.max_stable_bits,
            "max_selected_output_terms": caps.max_selected_output_terms,
            "max_selected_generator_nonzeros": (
                caps.max_selected_generator_nonzeros
            ),
            "max_source_payload_bytes": caps.max_source_payload_bytes,
            "max_fresh_payload_bytes": caps.max_fresh_payload_bytes,
            "max_fresh_payload_delta_bytes": (
                caps.max_fresh_payload_delta_bytes
            ),
            "static_additional_rss_budget_bytes": (
                caps.static_additional_rss_budget_bytes
            ),
            "max_process_rss_bytes": caps.max_process_rss_bytes,
            "min_mem_available_bytes": caps.min_mem_available_bytes,
            "min_cgroup_headroom_bytes": caps.min_cgroup_headroom_bytes,
            "candidate_timeout_seconds": caps.candidate_timeout_seconds,
            "pair_caps": vars(caps.pair_caps),
            "fresh_caps": {
                **{
                    name: getattr(caps.fresh_caps, name)
                    for name in PCOHFreshMaterializationCaps.__dataclass_fields__
                    if name != "row_caps"
                },
                "row_caps": vars(caps.fresh_caps.row_caps),
            },
            "pattern_cap": _K3_PATTERN_COUNT,
            "pair_query_cap": _K3_PAIR_QUERY_COUNT,
            "eta_cap": _K3_PATTERN_COUNT,
            "dynamic_equality_rows": "4+certified_empty_pattern_count",
            "local_lp_actual_call_cap": _K3_LOCAL_LP_UPPER_BOUND,
            "conditional_checker_actual_call_cap": (
                _K3_CONDITIONAL_CHECKER_UPPER_BOUND
            ),
        }
    )


def _outcome_payload(result: PCOHK3BuildOnlyOutcome, *, include_digest: bool) -> Dict[str, Any]:
    if type(result) is PCOHK3BuildOnlyDiagnostic:
        excluded = {"diagnostic_sha256"}
        digest_name = "diagnostic_sha256"
    elif type(result) is PCOHK3BuildOnlyStopDiagnostic:
        excluded = {"stop_sha256"}
        digest_name = "stop_sha256"
    elif type(result) is PCOHK3BuildOnlyResourceStopDiagnostic:
        excluded = {"resource_stop_sha256"}
        digest_name = "resource_stop_sha256"
    else:
        raise PhaseConditionedK3BuildOnlyError("outcome_wrong_type")
    payload = {
        name: getattr(result, name)
        for name in result.__dataclass_fields__
        if name not in excluded
    }
    if include_digest:
        payload[digest_name] = getattr(result, digest_name)
    return payload


def _register_outcome(result: PCOHK3BuildOnlyOutcome) -> None:
    field_names = tuple(result.__dataclass_fields__)
    if tuple(vars(result)) != field_names:
        raise PhaseConditionedK3BuildOnlyError(
            "outcome_field_shape_not_canonical"
        )
    with _OUTCOME_REGISTRY_LOCK:
        _OUTCOME_REGISTRY[result] = _RegisteredOutcome(
            process_id=os.getpid(),
            field_names=field_names,
            original_values=tuple(getattr(result, name) for name in field_names),
        )


def _receipt_checksum_valid(receipt: Any) -> bool:
    if type(receipt) is not MappingProxyType:
        return False
    body = dict(receipt)
    digest = body.pop("receipt_sha256", None)
    return bool(_valid_sha256(digest) and _canonical_sha256(body) == digest)


def _resource_receipt_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("passed") is True
        and value.get("caller_supplied") is False
        and type(value.get("current_rss_bytes")) is int
        and value.get("current_rss_bytes") >= 0
        and type(value.get("peak_rss_bytes")) is int
        and value.get("peak_rss_bytes") >= 0
    )


def _exact_int_sequence(
    value: Any,
    *,
    length: Optional[int] = None,
    nonnegative: bool = False,
) -> bool:
    if type(value) not in {tuple, list}:
        return False
    if length is not None and len(value) != length:
        return False
    return all(
        type(item) is int and (not nonnegative or item >= 0)
        for item in value
    )


def _exact_bool_sequence(value: Any, *, length: int) -> bool:
    return bool(
        type(value) in {tuple, list}
        and len(value) == length
        and all(type(item) is bool for item in value)
    )


def _exact_k3_schedule(value: Any) -> bool:
    if type(value) not in {tuple, list} or len(value) != _K3_PATTERN_COUNT:
        return False
    normalized = []
    for pattern in value:
        if (
            not _exact_int_sequence(pattern, length=3)
            or any(phase not in {-1, 1} for phase in pattern)
        ):
            return False
        normalized.append(tuple(pattern))
    return bool(
        len(set(normalized)) == _K3_PATTERN_COUNT
        and set(normalized) == set(itertools.product((-1, 1), repeat=3))
    )


def _k3_empty_mask_matches_pair_certificate_count(
    active_pattern_mask: Any, certificate_count: Any
) -> bool:
    if (
        not _exact_bool_sequence(
            active_pattern_mask, length=_K3_PATTERN_COUNT
        )
        or type(certificate_count) is not int
        or not 0 <= certificate_count <= _K3_PAIR_QUERY_COUNT
    ):
        return False
    canonical = tuple(itertools.product((-1, 1), repeat=3))
    empty_vertices = frozenset(
        index
        for index, active in enumerate(active_pattern_mask)
        if not active
    )
    edges = tuple(
        frozenset(
            index
            for index, pattern in enumerate(canonical)
            if pattern[left] == left_phase
            and pattern[right] == right_phase
        )
        for left, right in itertools.combinations(range(3), 2)
        for left_phase, right_phase in itertools.product((-1, 1), repeat=2)
    )
    eligible_edges = tuple(edge for edge in edges if edge <= empty_vertices)
    if certificate_count == 0:
        return not empty_vertices
    if certificate_count > len(eligible_edges):
        return False
    return any(
        frozenset().union(*chosen_edges) == empty_vertices
        for chosen_edges in itertools.combinations(
            eligible_edges, certificate_count
        )
    )


def _strict_resource_pass_receipt_valid(
    value: Any, *, caps: PCOHK3BuildOnlyCaps
) -> bool:
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_PASS_RECEIPT_KEYS:
        return False
    integer_names = (
        "current_rss_bytes",
        "peak_rss_bytes",
        "mem_available_bytes",
        "static_additional_rss_budget_bytes",
        "forecast_rss_bytes",
        "max_process_rss_bytes",
        "min_mem_available_bytes",
        "min_cgroup_headroom_bytes",
    )
    if any(
        type(value.get(name)) is not int or value.get(name) < 0
        for name in integer_names
    ):
        return False
    conditions = value.get("conditions")
    if (
        not isinstance(conditions, Mapping)
        or set(conditions) != _RESOURCE_CONDITION_KEYS
        or any(conditions.get(name) is not True for name in _RESOURCE_CONDITION_KEYS)
        or type(value.get("cgroup_limit_status")) is not str
        or value.get("cgroup_limit_status") not in {"bounded", "unbounded"}
        or type(value.get("measurement_source")) is not str
        or value.get("measurement_source")
        != "live_proc_status_meminfo_and_cgroup_v2"
        or value.get("passed") is not True
        or value.get("caller_supplied") is not False
        or value.get("static_additional_rss_budget_bytes")
        != caps.static_additional_rss_budget_bytes
        or value.get("max_process_rss_bytes") != caps.max_process_rss_bytes
        or value.get("min_mem_available_bytes")
        != caps.min_mem_available_bytes
        or value.get("min_cgroup_headroom_bytes")
        != caps.min_cgroup_headroom_bytes
        or value.get("forecast_rss_bytes")
        != value.get("current_rss_bytes")
        + caps.static_additional_rss_budget_bytes
        or value.get("peak_rss_bytes") < value.get("current_rss_bytes")
        or value.get("forecast_rss_bytes") >= caps.max_process_rss_bytes
        or value.get("peak_rss_bytes") > caps.max_process_rss_bytes
        or value.get("mem_available_bytes") < caps.min_mem_available_bytes
    ):
        return False
    if value.get("cgroup_limit_status") == "bounded":
        return bool(
            type(value.get("cgroup_headroom_bytes")) is int
            and value.get("cgroup_headroom_bytes") >= caps.min_cgroup_headroom_bytes
        )
    return value.get("cgroup_headroom_bytes") is None


def _strict_resource_stop_execution_valid(
    telemetry: Any, *, stage: str
) -> bool:
    if (
        not isinstance(telemetry, Mapping)
        or set(telemetry) != _RESOURCE_STOP_EXECUTION_KEYS
        or any(
            type(telemetry.get(name)) is not int
            or telemetry.get(name) < 0
            for name in _RESOURCE_STOP_EXECUTION_INT_KEYS
        )
        or telemetry.get("pair_local_lp_actual_calls")
        != _K3_PAIR_QUERY_COUNT
        or telemetry.get("local_lp_actual_call_cap")
        != _K3_LOCAL_LP_UPPER_BOUND
        or telemetry.get("conditional_checker_actual_call_cap")
        != _K3_CONDITIONAL_CHECKER_UPPER_BOUND
        or telemetry.get("fresh_live_replay_performed") is not False
        or telemetry.get("fresh_live_replay_checker_actual_calls") != 0
        or not 0
        <= telemetry.get("pair_exact_conflict_certificates")
        <= _K3_PAIR_QUERY_COUNT
        or telemetry.get("pair_exact_conflict_certificates_strictly_replayed")
        != telemetry.get("pair_exact_conflict_certificates")
    ):
        return False
    if stage == "pre_scheduled":
        return bool(
            telemetry.get("scheduled_local_lp_actual_calls") == 0
            and telemetry.get("local_lp_actual_calls") == _K3_PAIR_QUERY_COUNT
            and telemetry.get("scheduled_patterns_completed") == 0
            and telemetry.get("scheduled_candidate_dual_accepted") == 0
            and telemetry.get("scheduled_producer_checker_actual_calls") == 0
            and telemetry.get("conditional_checker_actual_calls") == 0
            and telemetry.get("scheduled_actual_call_site_counters") is None
            and telemetry.get("scheduled_telemetry_sha256") is None
        )
    if stage != "pre_fresh_materialization":
        return False
    scheduled_lp = telemetry.get("scheduled_local_lp_actual_calls")
    accepted = telemetry.get("scheduled_candidate_dual_accepted")
    return bool(
        0 <= scheduled_lp <= _K3_PATTERN_COUNT
        and telemetry.get("local_lp_actual_calls")
        == _K3_PAIR_QUERY_COUNT + scheduled_lp
        and telemetry.get("scheduled_patterns_completed") == _K3_PATTERN_COUNT
        and 0 <= accepted <= _K3_PATTERN_COUNT
        and telemetry.get("scheduled_producer_checker_actual_calls")
        == 1 + _K3_PATTERN_COUNT + accepted
        and telemetry.get("conditional_checker_actual_calls")
        == telemetry.get("scheduled_producer_checker_actual_calls")
        and telemetry.get("scheduled_actual_call_site_counters") is True
        and _valid_sha256(telemetry.get("scheduled_telemetry_sha256"))
    )


def _strict_resource_stop_timings_valid(value: Any, *, stage: str) -> bool:
    if (
        not isinstance(value, Mapping)
        or set(value) != _RESOURCE_STOP_TIMING_KEYS
        or any(
            type(value.get(name)) is not float
            or not math.isfinite(value.get(name))
            or value.get(name) < 0.0
            or (
                value.get(name) == 0.0
                and math.copysign(1.0, value.get(name)) < 0.0
            )
            for name in _RESOURCE_STOP_TIMING_KEYS
        )
        or math.fsum(
            (
                value.get("pair_and_strict_replay_seconds"),
                value.get("scheduled_producer_seconds"),
            )
        )
        > math.nextafter(value.get("total_seconds"), math.inf)
    ):
        return False
    return bool(
        stage != "pre_scheduled"
        or value.get("scheduled_producer_seconds") == 0.0
    )


def _strict_resource_stop_contract_valid(
    payload: Any, receipt: Any
) -> bool:
    if not isinstance(payload, Mapping) or not isinstance(receipt, Mapping):
        return False
    expected_payload_keys = set(
        PCOHK3BuildOnlyResourceStopDiagnostic.__dataclass_fields__
    ) - {"resource_stop_sha256"}
    if (
        set(payload) != expected_payload_keys
        or set(receipt) != _RESOURCE_STOP_RECEIPT_KEYS
        or payload.get("receipt") is not receipt
    ):
        return False
    stage = payload.get("stage")
    if (
        type(stage) is not str
        or stage not in {"pre_scheduled", "pre_fresh_materialization"}
        or type(payload.get("schema")) is not str
        or payload.get("schema") != _RESOURCE_STOP_SCHEMA
        or type(payload.get("status")) is not str
        or payload.get("status") != "stopped_by_resource_gate_no_partial_output"
        or type(payload.get("reason")) is not str
        or not payload.get("reason").startswith("resource_preflight_stop_loss:")
        or not _valid_sha256(payload.get("source_semantic_digest"))
        or payload.get("terminal_source_semantic_digest")
        != payload.get("source_semantic_digest")
        or type(payload.get("focused_rival_id")) is not int
        or payload.get("focused_rival_id") < 0
        or not _exact_int_sequence(
            payload.get("retained_k2_stable_bit_ids"),
            length=2,
            nonnegative=True,
        )
        or not _exact_int_sequence(
            payload.get("stable_bit_ids"), length=3, nonnegative=True
        )
        or tuple(payload.get("retained_k2_stable_bit_ids"))
        != tuple(payload.get("stable_bit_ids"))[:2]
        or tuple(sorted(payload.get("stable_bit_ids")))
        != tuple(payload.get("stable_bit_ids"))
        or len(set(payload.get("stable_bit_ids"))) != 3
        or type(payload.get("third_stable_bit_id")) is not int
        or payload.get("third_stable_bit_id")
        != tuple(payload.get("stable_bit_ids"))[2]
        or not _exact_int_sequence(
            payload.get("third_coefficient_exact"), length=2
        )
        or payload.get("third_coefficient_exact")[1] <= 0
        or type(payload.get("preferred_third_phase")) is not int
        or payload.get("preferred_third_phase") not in {-1, 1}
        or not _valid_sha256(payload.get("ranking_sha256"))
        or not _valid_sha256(payload.get("pair_bundle_sha256"))
        or not _exact_bool_sequence(
            payload.get("active_pattern_mask"), length=_K3_PATTERN_COUNT
        )
        or not _exact_k3_schedule(payload.get("evaluation_schedule"))
        or not _exact_int_sequence(
            payload.get("threshold_pattern_indices"), nonnegative=True
        )
        or type(payload.get("completed_conditional_certificate_count"))
        is not int
        or not _strict_resource_stop_execution_valid(
            payload.get("execution_telemetry"), stage=stage
        )
    ):
        return False
    coefficient_pair = tuple(payload.get("third_coefficient_exact"))
    coefficient = Fraction(*coefficient_pair)
    expected_phase = -1 if coefficient < 0 else 1
    if (
        _fraction_pair(coefficient) != coefficient_pair
        or any(abs(value).bit_length() > 4096 for value in coefficient_pair)
        or payload.get("preferred_third_phase") != expected_phase
    ):
        return False
    top_false_fields = (
        "partial_certificates_returned",
        "conditional_certificate_payload_returned",
        "fresh_issue_called",
        "fresh_build_returned",
        "fresh_descriptor_returned",
        "ground_truth_loaded",
        "full_parent_lp_called",
        "proof_authority",
        "verdict_authority",
        "provenance_authority",
        "authenticity_authority",
    )
    if any(payload.get(name) is not False for name in top_false_fields):
        return False
    canonical = tuple(itertools.product((-1, 1), repeat=3))
    evaluation_schedule = tuple(
        tuple(pattern) for pattern in payload.get("evaluation_schedule")
    )
    active_pattern_mask = tuple(payload.get("active_pattern_mask"))
    active_by_pattern = dict(zip(canonical, active_pattern_mask))
    preferred_child = (
        1,
        1,
        payload.get("preferred_third_phase"),
    )
    opposite_child = (1, 1, -payload.get("preferred_third_phase"))
    worst_k2_children = (preferred_child, opposite_child)
    active_children = tuple(
        pattern for pattern in worst_k2_children if active_by_pattern[pattern]
    )
    empty_children = tuple(
        pattern for pattern in worst_k2_children if not active_by_pattern[pattern]
    )
    remaining_active = tuple(
        pattern
        for pattern in canonical
        if pattern not in worst_k2_children and active_by_pattern[pattern]
    )
    remaining_empty = tuple(
        pattern
        for pattern in canonical
        if pattern not in worst_k2_children and not active_by_pattern[pattern]
    )
    expected_evaluation_schedule = (
        active_children
        + empty_children
        + remaining_active
        + remaining_empty
    )
    expected_threshold_indices = tuple(
        index
        for index, pattern in enumerate(expected_evaluation_schedule)
        if active_by_pattern[pattern]
    )
    schedule_payload = {
        "schema": _SCHEDULE_SCHEMA,
        "canonical_patterns": canonical,
        "active_pattern_mask": active_pattern_mask,
        "active_means": "not_certified_empty_by_exact_signed_pair",
        "evaluation_schedule": expected_evaluation_schedule,
        "threshold_pattern_indices": expected_threshold_indices,
        "worst_k2_children": worst_k2_children,
        "strict_upper_threshold_exact": _fraction_pair(_K3_STRONG_TARGET),
        "strict_comparison": "observed_upper_exact_gt_target",
    }
    if (
        not any(active_pattern_mask)
        or evaluation_schedule != expected_evaluation_schedule
        or tuple(payload.get("threshold_pattern_indices"))
        != expected_threshold_indices
        or not _valid_sha256(receipt.get("schedule_sha256"))
        or receipt.get("schedule_sha256")
        != _canonical_sha256(schedule_payload)
    ):
        return False
    if stage == "pre_scheduled":
        if (
            payload.get("scheduled_bundle_sha256") is not None
            or payload.get("completed_conditional_certificate_count") != 0
        ):
            return False
    elif (
        not _valid_sha256(payload.get("scheduled_bundle_sha256"))
        or payload.get("completed_conditional_certificate_count")
        != _K3_PATTERN_COUNT
    ):
        return False

    receipt_true_fields = ("diagnostic_only", "candidate_only")
    receipt_false_fields = (
        "production_ready",
        "solver_handoff_ready",
        "partial_certificates_returned",
        "conditional_certificate_payload_returned",
        "fresh_issue_called",
        "fresh_build_returned",
        "fresh_descriptor_returned",
        "ground_truth_parameter_accepted",
        "full_parent_lp_called",
        "proof_authority",
        "verdict_authority",
        "provenance_authority",
        "authenticity_authority",
    )
    if (
        any(receipt.get(name) is not True for name in receipt_true_fields)
        or any(receipt.get(name) is not False for name in receipt_false_fields)
        or type(receipt.get("schema")) is not str
        or receipt.get("schema") != _RESOURCE_STOP_RECEIPT_SCHEMA
        or type(receipt.get("status")) is not str
        or receipt.get("status") != payload.get("status")
        or type(receipt.get("stage")) is not str
        or receipt.get("stage") != stage
        or type(receipt.get("reason")) is not str
        or receipt.get("reason") != payload.get("reason")
        or type(receipt.get("fresh_registry_entries_created")) is not int
        or receipt.get("fresh_registry_entries_created") != 0
        or not _exact_int_sequence(
            receipt.get("fresh_registry_state_before"),
            length=2,
            nonnegative=True,
        )
        or not _exact_int_sequence(
            receipt.get("fresh_registry_state_terminal"),
            length=2,
            nonnegative=True,
        )
        or tuple(receipt.get("fresh_registry_state_before"))
        != tuple(receipt.get("fresh_registry_state_terminal"))
        or not _valid_sha256(receipt.get("source_semantic_digest"))
        or receipt.get("source_semantic_digest")
        != payload.get("source_semantic_digest")
        or receipt.get("terminal_source_semantic_digest")
        != payload.get("terminal_source_semantic_digest")
        or type(receipt.get("source_terminal_semantic_seal_read_count"))
        is not int
        or receipt.get("source_terminal_semantic_seal_read_count") != 2
        or not _exact_int_sequence(
            receipt.get("source_dimensions"), length=5, nonnegative=True
        )
        or receipt.get("source_dimensions")[0] < 1
        or receipt.get("source_dimensions")[2] < 3
        or receipt.get("source_dimensions")[4] < 9
        or type(receipt.get("source_payload_bytes")) is not int
        or not 0
        < receipt.get("source_payload_bytes")
        <= PCOHK3BuildOnlyCaps().max_source_payload_bytes
        or type(receipt.get("focused_rival_id")) is not int
        or receipt.get("focused_rival_id") != payload.get("focused_rival_id")
        or not _valid_sha256(receipt.get("focused_rival_binding_digest"))
        or not _exact_int_sequence(
            receipt.get("selected_output_positions"), nonnegative=True
        )
        or not 0 < len(receipt.get("selected_output_positions")) <= 2
        or len(set(receipt.get("selected_output_positions")))
        != len(receipt.get("selected_output_positions"))
        or tuple(receipt.get("selected_output_positions"))
        != tuple(sorted(receipt.get("selected_output_positions")))
        or any(
            position >= receipt.get("source_dimensions")[0]
            for position in receipt.get("selected_output_positions")
        )
        or type(receipt.get("selected_generator_nonzeros")) is not int
        or not 0
        <= receipt.get("selected_generator_nonzeros")
        <= PCOHK3BuildOnlyCaps().max_selected_generator_nonzeros
        or not _exact_int_sequence(
            receipt.get("retained_k2_stable_bit_ids"),
            length=2,
            nonnegative=True,
        )
        or tuple(receipt.get("retained_k2_stable_bit_ids"))
        != tuple(payload.get("retained_k2_stable_bit_ids"))
        or not _exact_int_sequence(
            receipt.get("stable_bit_ids"), length=3, nonnegative=True
        )
        or tuple(receipt.get("stable_bit_ids"))
        != tuple(payload.get("stable_bit_ids"))
        or type(receipt.get("third_stable_bit_id")) is not int
        or receipt.get("third_stable_bit_id")
        != payload.get("third_stable_bit_id")
        or not _exact_int_sequence(
            receipt.get("third_coefficient_exact"), length=2
        )
        or tuple(receipt.get("third_coefficient_exact"))
        != tuple(payload.get("third_coefficient_exact"))
        or type(receipt.get("preferred_third_phase")) is not int
        or receipt.get("preferred_third_phase")
        != payload.get("preferred_third_phase")
        or not _valid_sha256(receipt.get("ranking_sha256"))
        or receipt.get("ranking_sha256") != payload.get("ranking_sha256")
        or not _valid_sha256(receipt.get("pair_bundle_sha256"))
        or receipt.get("pair_bundle_sha256")
        != payload.get("pair_bundle_sha256")
        or type(receipt.get("pair_query_count")) is not int
        or receipt.get("pair_query_count") != _K3_PAIR_QUERY_COUNT
        or type(receipt.get("canonical_pattern_count")) is not int
        or receipt.get("canonical_pattern_count") != _K3_PATTERN_COUNT
        or not _exact_bool_sequence(
            receipt.get("active_pattern_mask"), length=_K3_PATTERN_COUNT
        )
        or tuple(receipt.get("active_pattern_mask")) != active_pattern_mask
        or type(receipt.get("certified_empty_pattern_count")) is not int
        or receipt.get("certified_empty_pattern_count")
        != active_pattern_mask.count(False)
        or not _exact_k3_schedule(receipt.get("evaluation_schedule"))
        or tuple(tuple(pattern) for pattern in receipt.get("evaluation_schedule"))
        != evaluation_schedule
        or not _exact_int_sequence(
            receipt.get("threshold_pattern_indices"), nonnegative=True
        )
        or tuple(receipt.get("threshold_pattern_indices"))
        != expected_threshold_indices
        or not _exact_int_sequence(
            receipt.get("strong_target_exact"), length=2
        )
        or tuple(receipt.get("strong_target_exact"))
        != _fraction_pair(_K3_STRONG_TARGET)
        or type(receipt.get("strict_comparison")) is not str
        or receipt.get("strict_comparison")
        != "observed_upper_exact_gt_target"
        or not _valid_sha256(receipt.get("schedule_sha256"))
        or type(receipt.get("completed_conditional_certificate_count"))
        is not int
        or receipt.get("completed_conditional_certificate_count")
        != payload.get("completed_conditional_certificate_count")
        or receipt.get("scheduled_bundle_sha256")
        != payload.get("scheduled_bundle_sha256")
        or not _strict_resource_stop_execution_valid(
            receipt.get("execution_telemetry"), stage=stage
        )
        or _canonical_sha256(receipt.get("execution_telemetry"))
        != _canonical_sha256(payload.get("execution_telemetry"))
        or not _strict_resource_pass_receipt_valid(
            receipt.get("resource_entry_preflight"),
            caps=PCOHK3BuildOnlyCaps(),
        )
        or _canonical_sha256(receipt.get("caps"))
        != _canonical_sha256(_caps_payload(PCOHK3BuildOnlyCaps()))
        or not _strict_resource_stop_timings_valid(
            receipt.get("timings"), stage=stage
        )
    ):
        return False
    pair_certificate_count = payload.get("execution_telemetry").get(
        "pair_exact_conflict_certificates"
    )
    if not _k3_empty_mask_matches_pair_certificate_count(
        receipt.get("active_pattern_mask"), pair_certificate_count
    ):
        return False
    rejection = receipt.get("resource_gate_rejection")
    if not _resource_gate_rejection_valid(
        rejection, stage=stage, caps=PCOHK3BuildOnlyCaps()
    ):
        return False
    if stage == "pre_scheduled":
        stage_rejection = receipt.get("resource_pre_s_preflight")
        if (
            receipt.get("scheduled_bundle_completed") is not False
            or receipt.get("scheduled_bundle_sha256") is not None
            or receipt.get("completed_conditional_certificate_count") != 0
            or receipt.get("resource_pre_fresh_preflight") is not None
            or not _resource_gate_rejection_valid(
                stage_rejection,
                stage=stage,
                caps=PCOHK3BuildOnlyCaps(),
            )
        ):
            return False
    else:
        stage_rejection = receipt.get("resource_pre_fresh_preflight")
        if (
            receipt.get("scheduled_bundle_completed") is not True
            or not _valid_sha256(receipt.get("scheduled_bundle_sha256"))
            or receipt.get("completed_conditional_certificate_count")
            != _K3_PATTERN_COUNT
            or not _strict_resource_pass_receipt_valid(
                receipt.get("resource_pre_s_preflight"),
                caps=PCOHK3BuildOnlyCaps(),
            )
            or not _resource_gate_rejection_valid(
                stage_rejection,
                stage=stage,
                caps=PCOHK3BuildOnlyCaps(),
            )
        ):
            return False
    if (
        _canonical_sha256(stage_rejection) != _canonical_sha256(rejection)
        or not _valid_sha256(receipt.get("receipt_sha256"))
    ):
        return False
    receipt_body = dict(receipt)
    receipt_sha256 = receipt_body.pop("receipt_sha256")
    return _canonical_sha256(receipt_body) == receipt_sha256


def _fresh_registry_state() -> Tuple[int, int]:
    """Read active/reserved fresh capability counts under their owner lock."""

    with _fresh_materializer_module._REGISTRY_LOCK:
        return (
            len(_fresh_materializer_module._REGISTRY),
            len(_fresh_materializer_module._REGISTRY_RESERVATIONS),
        )


def _resource_gate_rejection(
    snapshot: Mapping[str, Any],
    *,
    stage: str,
    reason: str,
    caps: PCOHK3BuildOnlyCaps,
) -> Mapping[str, Any]:
    if stage not in {"pre_scheduled", "pre_fresh_materialization"}:
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_stage_not_supported"
        )
    current = snapshot.get("current_rss_bytes")
    peak = snapshot.get("peak_rss_bytes")
    available = snapshot.get("mem_available_bytes")
    cgroup_status = snapshot.get("cgroup_limit_status")
    cgroup_headroom = snapshot.get("cgroup_headroom_bytes")
    if any(type(value) is not int or value < 0 for value in (current, peak, available)):
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_snapshot_integer_invalid"
        )
    if peak < current:
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_snapshot_peak_below_current"
        )
    if cgroup_status == "bounded":
        if type(cgroup_headroom) is not int or cgroup_headroom < 0:
            raise PhaseConditionedK3BuildOnlyError(
                "resource_stop_cgroup_headroom_invalid"
            )
        cgroup_passed = cgroup_headroom >= caps.min_cgroup_headroom_bytes
    elif cgroup_status == "unbounded":
        if cgroup_headroom is not None:
            raise PhaseConditionedK3BuildOnlyError(
                "resource_stop_unbounded_cgroup_has_headroom"
            )
        cgroup_passed = True
    else:
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_cgroup_status_invalid"
        )
    forecast = current + caps.static_additional_rss_budget_bytes
    conditions = {
        "current_plus_budget_strictly_below_process_cap": (
            forecast < caps.max_process_rss_bytes
        ),
        "entry_peak_not_above_process_cap": (
            peak <= caps.max_process_rss_bytes
        ),
        "mem_available_at_least_fixed_reserve": (
            available >= caps.min_mem_available_bytes
        ),
        "cgroup_unbounded_or_has_fixed_reserve": cgroup_passed,
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    expected_reason = "resource_preflight_stop_loss:" + ",".join(failed)
    if not failed or reason != expected_reason:
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_reason_not_bound_to_live_snapshot"
        )
    body: Dict[str, Any] = {
        "schema": _RESOURCE_GATE_REJECTION_SCHEMA,
        "stage": stage,
        "reason": reason,
        "current_rss_bytes": current,
        "peak_rss_bytes": peak,
        "mem_available_bytes": available,
        "cgroup_limit_status": cgroup_status,
        "cgroup_headroom_bytes": cgroup_headroom,
        "static_additional_rss_budget_bytes": (
            caps.static_additional_rss_budget_bytes
        ),
        "forecast_rss_bytes": forecast,
        "max_process_rss_bytes": caps.max_process_rss_bytes,
        "min_mem_available_bytes": caps.min_mem_available_bytes,
        "min_cgroup_headroom_bytes": caps.min_cgroup_headroom_bytes,
        "conditions": conditions,
        "failed_conditions": failed,
        "measurement_source": snapshot.get("measurement_source"),
        "caller_supplied": snapshot.get("caller_supplied"),
        "passed": False,
    }
    if (
        body["measurement_source"]
        != "live_proc_status_meminfo_and_cgroup_v2"
        or body["caller_supplied"] is not False
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_snapshot_not_live_kernel_measurement"
        )
    body["rejection_sha256"] = _canonical_sha256(body)
    return _deep_freeze_receipt(body)


def _resource_gate_rejection_valid(
    value: Any,
    *,
    stage: str,
    caps: PCOHK3BuildOnlyCaps,
) -> bool:
    try:
        if (
            not isinstance(value, Mapping)
            or set(value) != _RESOURCE_GATE_REJECTION_KEYS
        ):
            return False
        conditions = value.get("conditions")
        failed_conditions = value.get("failed_conditions")
        if (
            not isinstance(conditions, Mapping)
            or set(conditions) != _RESOURCE_CONDITION_KEYS
            or any(
                type(conditions.get(name)) is not bool
                for name in _RESOURCE_CONDITION_KEYS
            )
            or type(failed_conditions) not in {tuple, list}
            or not failed_conditions
            or any(type(name) is not str for name in failed_conditions)
        ):
            return False
        body = dict(value)
        rejection_sha256 = body.pop("rejection_sha256", None)
        if (
            not _valid_sha256(rejection_sha256)
            or _canonical_sha256(body) != rejection_sha256
            or body.get("schema") != _RESOURCE_GATE_REJECTION_SCHEMA
            or body.get("stage") != stage
            or body.get("passed") is not False
        ):
            return False
        replay = _resource_gate_rejection(
            body,
            stage=stage,
            reason=body.get("reason"),
            caps=caps,
        )
        return _canonical_sha256(replay) == _canonical_sha256(value)
    except (
        PhaseConditionedK3BuildOnlyError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _observable_resource_gate(
    snapshot: Mapping[str, Any],
    *,
    stage: str,
    caps: PCOHK3BuildOnlyCaps,
) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    if (
        snapshot.get("measurement_source")
        != "live_proc_status_meminfo_and_cgroup_v2"
        or snapshot.get("caller_supplied") is not False
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "resource_gate_snapshot_not_live_kernel_measurement"
        )
    try:
        passed = _resource_preflight(
            current_rss_bytes=snapshot.get("current_rss_bytes"),
            peak_rss_bytes=snapshot.get("peak_rss_bytes"),
            mem_available_bytes=snapshot.get("mem_available_bytes"),
            cgroup_limit_status=snapshot.get("cgroup_limit_status"),
            cgroup_headroom_bytes=snapshot.get("cgroup_headroom_bytes"),
            caps=caps,
        )
    except PhaseConditionedBuildOnlyError as exc:
        reason = str(exc)
        if not reason.startswith("resource_preflight_stop_loss:"):
            _clear_exception_traceback(exc)
            raise PhaseConditionedK3BuildOnlyError(
                "resource_gate_measurement_or_contract_failed:" + reason[:240]
            ) from None
        try:
            rejection = _resource_gate_rejection(
                snapshot, stage=stage, reason=reason, caps=caps
            )
        finally:
            _clear_exception_traceback(exc)
        return None, rejection
    passed = _deep_freeze_receipt(
        {
            **dict(passed),
            "measurement_source": snapshot.get("measurement_source"),
            "caller_supplied": False,
        }
    )
    if not _resource_receipt_valid(passed):
        raise PhaseConditionedK3BuildOnlyError(
            "resource_gate_pass_receipt_invalid"
        )
    return passed, None


def _pre_scheduled_execution_telemetry(
    pair_bundle: PairInfeasibilityBundle,
) -> Mapping[str, Any]:
    pair_lp = len(pair_bundle.records)
    if pair_lp != _K3_PAIR_QUERY_COUNT:
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_pair_actual_counter_binding_failed"
        )
    return _deep_freeze_receipt(
        {
            "pair_local_lp_actual_calls": pair_lp,
            "scheduled_local_lp_actual_calls": 0,
            "local_lp_actual_calls": pair_lp,
            "local_lp_actual_call_cap": _K3_LOCAL_LP_UPPER_BOUND,
            "pair_exact_conflict_certificates": len(pair_bundle.certificates),
            "pair_exact_conflict_certificates_strictly_replayed": (
                len(pair_bundle.certificates)
            ),
            "scheduled_patterns_completed": 0,
            "scheduled_candidate_dual_accepted": 0,
            "scheduled_producer_checker_actual_calls": 0,
            "fresh_live_replay_checker_actual_calls": 0,
            "conditional_checker_actual_calls": 0,
            "conditional_checker_actual_call_cap": (
                _K3_CONDITIONAL_CHECKER_UPPER_BOUND
            ),
            "fresh_live_replay_performed": False,
            "scheduled_actual_call_site_counters": None,
            "scheduled_telemetry_sha256": None,
        }
    )


def _common_execution_telemetry(
    *,
    pair_bundle: PairInfeasibilityBundle,
    scheduled_telemetry: Mapping[str, Any],
    fresh_replay_performed: bool,
) -> Mapping[str, Any]:
    accepted = scheduled_telemetry.get("candidate_dual_accepted")
    producer_checker = scheduled_telemetry.get("split_checker_evaluations")
    scheduled_lp = scheduled_telemetry.get("linprog_actual_calls")
    completed = scheduled_telemetry.get("patterns_completed")
    if (
        type(accepted) is not int
        or type(producer_checker) is not int
        or type(scheduled_lp) is not int
        or type(completed) is not int
        or producer_checker != 1 + completed + accepted
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "scheduled_actual_counter_binding_failed"
        )
    pair_lp = len(pair_bundle.records)
    fresh_checker = (
        1 + _K3_PATTERN_COUNT + accepted
        if fresh_replay_performed
        else 0
    )
    total_lp = pair_lp + scheduled_lp
    total_checker = producer_checker + fresh_checker
    if (
        pair_lp != _K3_PAIR_QUERY_COUNT
        or total_lp > _K3_LOCAL_LP_UPPER_BOUND
        or total_checker > _K3_CONDITIONAL_CHECKER_UPPER_BOUND
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "k3_actual_counter_cap_exceeded"
        )
    return _deep_freeze_receipt(
        {
            "pair_local_lp_actual_calls": pair_lp,
            "scheduled_local_lp_actual_calls": scheduled_lp,
            "local_lp_actual_calls": total_lp,
            "local_lp_actual_call_cap": _K3_LOCAL_LP_UPPER_BOUND,
            "pair_exact_conflict_certificates": len(pair_bundle.certificates),
            "pair_exact_conflict_certificates_strictly_replayed": (
                len(pair_bundle.certificates)
            ),
            "scheduled_patterns_completed": completed,
            "scheduled_candidate_dual_accepted": accepted,
            "scheduled_producer_checker_actual_calls": producer_checker,
            "fresh_live_replay_checker_actual_calls": fresh_checker,
            "conditional_checker_actual_calls": total_checker,
            "conditional_checker_actual_call_cap": (
                _K3_CONDITIONAL_CHECKER_UPPER_BOUND
            ),
            "fresh_live_replay_performed": fresh_replay_performed,
            "scheduled_actual_call_site_counters": (
                scheduled_telemetry.get("actual_call_site_counters")
            ),
            "scheduled_telemetry_sha256": (
                scheduled_telemetry.get("telemetry_sha256")
            ),
        }
    )


def _make_stop_outcome(
    *,
    started: float,
    source_digest: str,
    source_dimensions: Tuple[int, int, int, int, int],
    source_payload: int,
    focused_rival_id: int,
    rival: RivalSpec,
    selected_outputs: Tuple[int, ...],
    selected_generator_nonzeros: int,
    plan: PCOHK3ThirdBitPlan,
    pair_bundle: PairInfeasibilityBundle,
    schedule: PCOHK3PairFirstSchedule,
    record: Any,
    pair_seconds: float,
    scheduled_seconds: float,
    entry_resource: Mapping[str, Any],
    pre_s_resource: Mapping[str, Any],
    terminal_resource: Mapping[str, Any],
    terminal_source_digest: str,
    caps: PCOHK3BuildOnlyCaps,
    deadline: float,
) -> PCOHK3BuildOnlyStopDiagnostic:
    execution = _common_execution_telemetry(
        pair_bundle=pair_bundle,
        scheduled_telemetry=record.telemetry,
        fresh_replay_performed=False,
    )
    stop_record_payload = _scheduled_stop_record_payload(record)
    receipt: Dict[str, Any] = {
        "schema": _STOP_RECEIPT_SCHEMA,
        "status": "stopped_by_strong_target_no_partial_output",
        "reason": record.reason,
        "diagnostic_only": True,
        "candidate_only": True,
        "partial_certificates_returned": False,
        "fresh_issue_called": False,
        "fresh_build_returned": False,
        "fresh_registry_entries_created": 0,
        "ground_truth_parameter_accepted": False,
        "full_parent_lp_called": False,
        "proof_authority": False,
        "verdict_authority": False,
        "source_semantic_digest": source_digest,
        "terminal_source_semantic_digest": terminal_source_digest,
        "source_dimensions": source_dimensions,
        "source_payload_bytes": source_payload,
        "focused_rival_id": focused_rival_id,
        "focused_rival_binding_digest": rival.binding_digest,
        "selected_output_positions": selected_outputs,
        "selected_generator_nonzeros": selected_generator_nonzeros,
        "retained_k2_stable_bit_ids": plan.retained_k2_stable_bit_ids,
        "stable_bit_ids": plan.stable_bit_ids,
        "third_stable_bit_id": plan.third_stable_bit_id,
        "third_coefficient_exact": _fraction_pair(plan.third_coefficient),
        "preferred_third_phase": plan.preferred_third_phase,
        "ranking_sha256": plan.ranking_sha256,
        "pair_bundle_sha256": pair_bundle.bundle_sha256,
        "pair_query_count": len(pair_bundle.records),
        "canonical_pattern_count": len(schedule.canonical_patterns),
        "active_pattern_mask": schedule.active_pattern_mask,
        "certified_empty_pattern_count": schedule.active_pattern_mask.count(False),
        "evaluation_schedule": schedule.evaluation_schedule,
        "threshold_pattern_indices": schedule.threshold_pattern_indices,
        "strong_target_exact": _fraction_pair(_K3_STRONG_TARGET),
        "strict_comparison": "observed_upper_exact_gt_target",
        "schedule_sha256": schedule.schedule_sha256,
        "scheduled_stop_record": stop_record_payload,
        "scheduled_stop_record_sha256": record.record_sha256,
        "execution_telemetry": execution,
        "resource_entry_preflight": entry_resource,
        "resource_pre_s_preflight": pre_s_resource,
        "resource_terminal_postflight": terminal_resource,
        "caps": _caps_payload(caps),
        "timings": {
            "pair_and_strict_replay_seconds": pair_seconds,
            "scheduled_producer_seconds": scheduled_seconds,
            "total_seconds": float(time.monotonic() - started),
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    kwargs = {
        "schema": _STOP_SCHEMA,
        "status": "stopped_by_strong_target_no_partial_output",
        "reason": record.reason,
        "source_semantic_digest": source_digest,
        "terminal_source_semantic_digest": terminal_source_digest,
        "focused_rival_id": focused_rival_id,
        "retained_k2_stable_bit_ids": plan.retained_k2_stable_bit_ids,
        "stable_bit_ids": plan.stable_bit_ids,
        "third_stable_bit_id": plan.third_stable_bit_id,
        "third_coefficient_exact": _fraction_pair(plan.third_coefficient),
        "preferred_third_phase": plan.preferred_third_phase,
        "ranking_sha256": plan.ranking_sha256,
        "pair_bundle_sha256": pair_bundle.bundle_sha256,
        "active_pattern_mask": schedule.active_pattern_mask,
        "evaluation_schedule": schedule.evaluation_schedule,
        "threshold_pattern_indices": schedule.threshold_pattern_indices,
        "scheduled_stop_record_sha256": record.record_sha256,
        "triggering_schedule_index": record.triggering_schedule_index,
        "triggering_pattern": record.triggering_pattern,
        "observed_upper_exact": _fraction_pair(record.observed_upper_exact),
        "execution_telemetry": execution,
        "receipt": receipt,
        "stop_sha256": "",
        "partial_certificates_returned": False,
        "fresh_issue_called": False,
        "fresh_build_returned": False,
        "ground_truth_loaded": False,
        "full_parent_lp_called": False,
        "proof_authority": False,
        "verdict_authority": False,
    }
    provisional = PCOHK3BuildOnlyStopDiagnostic(**kwargs)
    result = PCOHK3BuildOnlyStopDiagnostic(
        **{
            **kwargs,
            "stop_sha256": _canonical_sha256(
                _outcome_payload(provisional, include_digest=False)
            ),
        }
    )
    _check_deadline(deadline, "before_k3_stop_anchor_and_return")
    _register_outcome(result)
    return result


def _make_resource_stop_outcome(
    *,
    build: OperatorHZBuild,
    started: float,
    stage: str,
    resource_gate_rejection: Mapping[str, Any],
    source_digest: str,
    source_dimensions: Tuple[int, int, int, int, int],
    source_payload: int,
    focused_rival_id: int,
    rival: RivalSpec,
    selected_outputs: Tuple[int, ...],
    selected_generator_nonzeros: int,
    plan: PCOHK3ThirdBitPlan,
    pair_bundle: PairInfeasibilityBundle,
    schedule: PCOHK3PairFirstSchedule,
    scheduled: Optional[ScheduledOperatorPhaseConditionedObjectiveBounds],
    pair_seconds: float,
    scheduled_seconds: float,
    entry_resource: Mapping[str, Any],
    pre_s_resource: Mapping[str, Any],
    fresh_registry_state_before: Tuple[int, int],
    caps: PCOHK3BuildOnlyCaps,
    deadline: float,
) -> PCOHK3BuildOnlyResourceStopDiagnostic:
    if stage == "pre_scheduled":
        if scheduled is not None or scheduled_seconds != 0.0:
            raise PhaseConditionedK3BuildOnlyError(
                "pre_scheduled_resource_stop_has_scheduled_owner"
            )
        execution = _pre_scheduled_execution_telemetry(pair_bundle)
        scheduled_bundle_sha256: Optional[str] = None
        completed_conditional_certificate_count = 0
        pre_fresh_resource: Optional[Mapping[str, Any]] = None
    elif stage == "pre_fresh_materialization":
        if type(scheduled) is not ScheduledOperatorPhaseConditionedObjectiveBounds:
            raise PhaseConditionedK3BuildOnlyError(
                "pre_fresh_resource_stop_missing_scheduled_owner"
            )
        execution = _common_execution_telemetry(
            pair_bundle=pair_bundle,
            scheduled_telemetry=scheduled.telemetry,
            fresh_replay_performed=False,
        )
        scheduled_bundle_sha256 = scheduled.bundle_sha256
        completed_conditional_certificate_count = len(scheduled.certificates)
        if (
            not _valid_sha256(scheduled_bundle_sha256)
            or completed_conditional_certificate_count != _K3_PATTERN_COUNT
        ):
            raise PhaseConditionedK3BuildOnlyError(
                "pre_fresh_resource_stop_scheduled_binding_failed"
            )
        pre_fresh_resource = resource_gate_rejection
    else:
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_stage_not_supported"
        )
    if not _resource_gate_rejection_valid(
        resource_gate_rejection, stage=stage, caps=caps
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_rejection_receipt_invalid"
        )
    if (
        type(fresh_registry_state_before) is not tuple
        or len(fresh_registry_state_before) != 2
        or any(
            type(value) is not int or value < 0
            for value in fresh_registry_state_before
        )
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_fresh_registry_entry_state_invalid"
        )
    reason = resource_gate_rejection.get("reason")
    if type(reason) is not str:
        raise PhaseConditionedK3BuildOnlyError(
            "resource_stop_reason_wrong_type"
        )

    # No fresh capability can be issued between the terminal registry read and
    # the process-local outcome anchor.  The source has no analogous owner
    # lock, so bracket receipt construction with two independent semantic reads.
    with _fresh_materializer_module._REGISTRY_LOCK:
        fresh_registry_state_terminal = (
            len(_fresh_materializer_module._REGISTRY),
            len(_fresh_materializer_module._REGISTRY_RESERVATIONS),
        )
        if fresh_registry_state_terminal != fresh_registry_state_before:
            raise PhaseConditionedK3BuildOnlyError(
                "fresh_registry_changed_during_resource_stop"
            )
        terminal_source_before = sparse_hz_semantic_digest(build.hz)
        if terminal_source_before != source_digest:
            raise PhaseConditionedK3BuildOnlyError(
                "terminal_source_digest_changed_on_resource_stop"
            )
        receipt: Dict[str, Any] = {
            "schema": _RESOURCE_STOP_RECEIPT_SCHEMA,
            "status": "stopped_by_resource_gate_no_partial_output",
            "stage": stage,
            "reason": reason,
            "diagnostic_only": True,
            "candidate_only": True,
            "production_ready": False,
            "solver_handoff_ready": False,
            "partial_certificates_returned": False,
            "conditional_certificate_payload_returned": False,
            "fresh_issue_called": False,
            "fresh_build_returned": False,
            "fresh_descriptor_returned": False,
            "fresh_registry_entries_created": 0,
            "fresh_registry_state_before": fresh_registry_state_before,
            "fresh_registry_state_terminal": fresh_registry_state_terminal,
            "ground_truth_parameter_accepted": False,
            "full_parent_lp_called": False,
            "proof_authority": False,
            "verdict_authority": False,
            "provenance_authority": False,
            "authenticity_authority": False,
            "source_semantic_digest": source_digest,
            "terminal_source_semantic_digest": terminal_source_before,
            "source_terminal_semantic_seal_read_count": 2,
            "source_dimensions": source_dimensions,
            "source_payload_bytes": source_payload,
            "focused_rival_id": focused_rival_id,
            "focused_rival_binding_digest": rival.binding_digest,
            "selected_output_positions": selected_outputs,
            "selected_generator_nonzeros": selected_generator_nonzeros,
            "retained_k2_stable_bit_ids": plan.retained_k2_stable_bit_ids,
            "stable_bit_ids": plan.stable_bit_ids,
            "third_stable_bit_id": plan.third_stable_bit_id,
            "third_coefficient_exact": _fraction_pair(plan.third_coefficient),
            "preferred_third_phase": plan.preferred_third_phase,
            "ranking_sha256": plan.ranking_sha256,
            "pair_bundle_sha256": pair_bundle.bundle_sha256,
            "pair_query_count": len(pair_bundle.records),
            "canonical_pattern_count": len(schedule.canonical_patterns),
            "active_pattern_mask": schedule.active_pattern_mask,
            "certified_empty_pattern_count": (
                schedule.active_pattern_mask.count(False)
            ),
            "evaluation_schedule": schedule.evaluation_schedule,
            "threshold_pattern_indices": schedule.threshold_pattern_indices,
            "strong_target_exact": _fraction_pair(_K3_STRONG_TARGET),
            "strict_comparison": "observed_upper_exact_gt_target",
            "schedule_sha256": schedule.schedule_sha256,
            "scheduled_bundle_completed": scheduled is not None,
            "scheduled_bundle_sha256": scheduled_bundle_sha256,
            "completed_conditional_certificate_count": (
                completed_conditional_certificate_count
            ),
            "execution_telemetry": execution,
            "resource_entry_preflight": entry_resource,
            "resource_pre_s_preflight": pre_s_resource,
            "resource_pre_fresh_preflight": pre_fresh_resource,
            "resource_gate_rejection": resource_gate_rejection,
            "caps": _caps_payload(caps),
            "timings": {
                "pair_and_strict_replay_seconds": pair_seconds,
                "scheduled_producer_seconds": scheduled_seconds,
                "total_seconds": float(time.monotonic() - started),
            },
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        kwargs = {
            "schema": _RESOURCE_STOP_SCHEMA,
            "status": "stopped_by_resource_gate_no_partial_output",
            "stage": stage,
            "reason": reason,
            "source_semantic_digest": source_digest,
            "terminal_source_semantic_digest": terminal_source_before,
            "focused_rival_id": focused_rival_id,
            "retained_k2_stable_bit_ids": plan.retained_k2_stable_bit_ids,
            "stable_bit_ids": plan.stable_bit_ids,
            "third_stable_bit_id": plan.third_stable_bit_id,
            "third_coefficient_exact": _fraction_pair(plan.third_coefficient),
            "preferred_third_phase": plan.preferred_third_phase,
            "ranking_sha256": plan.ranking_sha256,
            "pair_bundle_sha256": pair_bundle.bundle_sha256,
            "active_pattern_mask": schedule.active_pattern_mask,
            "evaluation_schedule": schedule.evaluation_schedule,
            "threshold_pattern_indices": schedule.threshold_pattern_indices,
            "scheduled_bundle_sha256": scheduled_bundle_sha256,
            "completed_conditional_certificate_count": (
                completed_conditional_certificate_count
            ),
            "execution_telemetry": execution,
            "receipt": receipt,
            "resource_stop_sha256": "",
            "partial_certificates_returned": False,
            "conditional_certificate_payload_returned": False,
            "fresh_issue_called": False,
            "fresh_build_returned": False,
            "fresh_descriptor_returned": False,
            "ground_truth_loaded": False,
            "full_parent_lp_called": False,
            "proof_authority": False,
            "verdict_authority": False,
            "provenance_authority": False,
            "authenticity_authority": False,
        }
        provisional = PCOHK3BuildOnlyResourceStopDiagnostic(**kwargs)
        result = PCOHK3BuildOnlyResourceStopDiagnostic(
            **{
                **kwargs,
                "resource_stop_sha256": _canonical_sha256(
                    _outcome_payload(provisional, include_digest=False)
                ),
            }
        )
        _check_deadline(deadline, "before_k3_resource_stop_source_reseal")
        terminal_source_after = sparse_hz_semantic_digest(build.hz)
        if terminal_source_after != source_digest:
            raise PhaseConditionedK3BuildOnlyError(
                "terminal_source_digest_changed_on_resource_stop"
            )
        _check_deadline(deadline, "before_k3_resource_stop_anchor_and_return")
        _register_outcome(result)
        return result


def _run_phase_conditioned_objective_hull_k3_build_only(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    retained_k2_stable_bit_ids: Tuple[int, int],
    deadline: float,
    caps: PCOHK3BuildOnlyCaps,
) -> PCOHK3BuildOnlyOutcome:
    started = time.monotonic()
    deadline_value = _deadline(deadline)
    normalized_caps = _normalize_caps(caps)
    if (
        type(build) is not OperatorHZBuild
        or type(build.hz) is not SparseHZono
        or type(selection) is not OperatorExactReLUPhaseSelection
        or type(focused_rival_id) is not int
        or focused_rival_id < 0
    ):
        raise PhaseConditionedK3BuildOnlyError("top_level_input_wrong_type")
    live = _live_resource_snapshot()
    entry_resource = _resource_preflight(
        current_rss_bytes=live["current_rss_bytes"],
        peak_rss_bytes=live["peak_rss_bytes"],
        mem_available_bytes=live["mem_available_bytes"],
        cgroup_limit_status=live["cgroup_limit_status"],
        cgroup_headroom_bytes=live["cgroup_headroom_bytes"],
        caps=normalized_caps,
    )
    entry_resource = _deep_freeze_receipt(
        {**dict(entry_resource), "measurement_source": live["measurement_source"], "caller_supplied": False}
    )
    if not verify_operator_exact_relu_property_phase_selection(
        build, rivals, selection, **_selection_caps_kwargs(selection)
    ):
        raise PhaseConditionedK3BuildOnlyError(
            "selection_live_replay_failed"
        )
    _check_deadline(deadline_value, "after_selection_live_replay")
    source_dimensions = _shape(build.hz)
    source_payload = _payload_bytes(build)
    if source_payload > normalized_caps.max_source_payload_bytes:
        raise PhaseConditionedK3BuildOnlyError(
            "source_payload_cap_exceeded_before_fraction"
        )
    source_digest = sparse_hz_semantic_digest(build.hz)
    if source_digest != selection.parent_semantic_digest:
        raise PhaseConditionedK3BuildOnlyError("selection_parent_digest_stale")
    rival = _focused_rival(rivals, focused_rival_id)
    if (
        len(rival.objective) != build.hz.n_out
        or any(type(value) is not float or not math.isfinite(value) for value in rival.objective)
        or type(rival.threshold) is not float
        or not math.isfinite(rival.threshold)
    ):
        raise PhaseConditionedK3BuildOnlyError("focused_rival_malformed")
    selected_outputs, selected_generator_nonzeros = _sparse_margin_preflight(
        build.hz,
        rival,
        caps=normalized_caps,
        deadline=deadline_value,
    )
    plan = select_k3_third_bit_plan(
        selection,
        focused_rival_id=focused_rival_id,
        retained_k2_stable_bit_ids=retained_k2_stable_bit_ids,
    )

    pair_started = time.monotonic()
    pair_bundle = run_phase_conditioned_pair_infeasibility_candidate(
        build,
        rivals,
        selection,
        stable_bit_ids=plan.stable_bit_ids,
        deadline=deadline_value,
        caps=normalized_caps.pair_caps,
    )
    pair_bundle = _strict_pair_bundle(
        build,
        rivals,
        selection,
        pair_bundle,
        stable_ids=plan.stable_bit_ids,
        deadline=deadline_value,
    )
    pair_seconds = float(time.monotonic() - pair_started)
    if (
        pair_bundle.parent_semantic_digest != source_digest
        or pair_bundle.terminal_parent_semantic_digest != source_digest
        or pair_bundle.selection_digest != selection.selection_digest
        or pair_bundle.caps != normalized_caps.pair_caps
    ):
        raise PhaseConditionedK3BuildOnlyError("pair_bundle_owner_binding_failed")
    schedule = build_k3_pair_first_schedule(
        pair_bundle, preferred_third_phase=plan.preferred_third_phase
    )
    fresh_registry_state_before = _fresh_registry_state()
    live = _live_resource_snapshot()
    pre_s_resource, pre_s_rejection = _observable_resource_gate(
        live,
        stage="pre_scheduled",
        caps=normalized_caps,
    )
    if pre_s_rejection is not None:
        if pre_s_resource is not None:
            raise PhaseConditionedK3BuildOnlyError(
                "pre_scheduled_resource_gate_ambiguous"
            )
        return _make_resource_stop_outcome(
            build=build,
            started=started,
            stage="pre_scheduled",
            resource_gate_rejection=pre_s_rejection,
            source_digest=source_digest,
            source_dimensions=source_dimensions,
            source_payload=source_payload,
            focused_rival_id=focused_rival_id,
            rival=rival,
            selected_outputs=selected_outputs,
            selected_generator_nonzeros=selected_generator_nonzeros,
            plan=plan,
            pair_bundle=pair_bundle,
            schedule=schedule,
            scheduled=None,
            pair_seconds=pair_seconds,
            scheduled_seconds=0.0,
            entry_resource=entry_resource,
            pre_s_resource=pre_s_rejection,
            fresh_registry_state_before=fresh_registry_state_before,
            caps=normalized_caps,
            deadline=deadline_value,
        )
    if pre_s_resource is None:
        raise PhaseConditionedK3BuildOnlyError(
            "pre_scheduled_resource_gate_missing_pass_receipt"
        )
    scheduled_started = time.monotonic()
    try:
        scheduled = build_scheduled_complete_operator_phase_conditioned_objective_bounds(
            build,
            rivals,
            selection,
            focused_rival_id=focused_rival_id,
            stable_bit_ids=plan.stable_bit_ids,
            evaluation_schedule=schedule.evaluation_schedule,
            deadline=deadline_value,
            stop_policy=schedule.stop_policy,
            candidate_timeout_seconds=normalized_caps.candidate_timeout_seconds,
        )
    except OperatorPhaseConditionedScheduledStop as stop:
        scheduled_seconds = float(time.monotonic() - scheduled_started)
        record = _validate_scheduled_stop(
            stop,
            source_digest=source_digest,
            stable_ids=plan.stable_bit_ids,
            schedule=schedule,
        )
        terminal_source = sparse_hz_semantic_digest(build.hz)
        if terminal_source != source_digest:
            raise PhaseConditionedK3BuildOnlyError(
                "terminal_source_digest_changed_on_stop"
            )
        terminal_resource = _resource_postflight(
            _live_resource_snapshot(), caps=normalized_caps
        )
        _check_deadline(deadline_value, "before_k3_stop_return")
        return _make_stop_outcome(
            started=started,
            source_digest=source_digest,
            source_dimensions=source_dimensions,
            source_payload=source_payload,
            focused_rival_id=focused_rival_id,
            rival=rival,
            selected_outputs=selected_outputs,
            selected_generator_nonzeros=selected_generator_nonzeros,
            plan=plan,
            pair_bundle=pair_bundle,
            schedule=schedule,
            record=record,
            pair_seconds=pair_seconds,
            scheduled_seconds=scheduled_seconds,
            entry_resource=entry_resource,
            pre_s_resource=pre_s_resource,
            terminal_resource=terminal_resource,
            terminal_source_digest=terminal_source,
            caps=normalized_caps,
            deadline=deadline_value,
        )
    scheduled_seconds = float(time.monotonic() - scheduled_started)
    scheduled = _validate_scheduled_complete(
        scheduled,
        source_digest=source_digest,
        stable_ids=plan.stable_bit_ids,
        schedule=schedule,
    )
    _check_deadline(deadline_value, "before_fresh_resource_gate")
    fresh_registry_state_before = _fresh_registry_state()
    live = _live_resource_snapshot()
    pre_fresh_resource, pre_fresh_rejection = _observable_resource_gate(
        live,
        stage="pre_fresh_materialization",
        caps=normalized_caps,
    )
    if pre_fresh_rejection is not None:
        if pre_fresh_resource is not None:
            raise PhaseConditionedK3BuildOnlyError(
                "pre_fresh_resource_gate_ambiguous"
            )
        return _make_resource_stop_outcome(
            build=build,
            started=started,
            stage="pre_fresh_materialization",
            resource_gate_rejection=pre_fresh_rejection,
            source_digest=source_digest,
            source_dimensions=source_dimensions,
            source_payload=source_payload,
            focused_rival_id=focused_rival_id,
            rival=rival,
            selected_outputs=selected_outputs,
            selected_generator_nonzeros=selected_generator_nonzeros,
            plan=plan,
            pair_bundle=pair_bundle,
            schedule=schedule,
            scheduled=scheduled,
            pair_seconds=pair_seconds,
            scheduled_seconds=scheduled_seconds,
            entry_resource=entry_resource,
            pre_s_resource=pre_s_resource,
            fresh_registry_state_before=fresh_registry_state_before,
            caps=normalized_caps,
            deadline=deadline_value,
        )
    if pre_fresh_resource is None:
        raise PhaseConditionedK3BuildOnlyError(
            "pre_fresh_resource_gate_missing_pass_receipt"
        )
    inspection, inspection_error = _issue_consume_inspect_release(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_ids=plan.stable_bit_ids,
        certificates=scheduled.certificates,
        pair_bundle=pair_bundle,
        active_pattern_mask=schedule.active_pattern_mask,
        deadline=deadline_value,
        caps=normalized_caps,
        source_dimensions=source_dimensions,
        source_payload_bytes=source_payload,
        source_semantic_digest=source_digest,
    )
    if inspection_error is not None or type(inspection) is not _FreshInspection:
        raise PhaseConditionedK3BuildOnlyError(
            "fresh_issue_consume_inspect_release_failed:"
            + str(inspection_error)[:260]
        ) from None
    terminal_source = sparse_hz_semantic_digest(build.hz)
    if terminal_source != source_digest:
        raise PhaseConditionedK3BuildOnlyError(
            "terminal_source_digest_changed_after_fresh"
        )
    terminal_resource = _resource_postflight(
        _live_resource_snapshot(), caps=normalized_caps
    )
    _check_deadline(deadline_value, "before_k3_success_return")
    execution = _common_execution_telemetry(
        pair_bundle=pair_bundle,
        scheduled_telemetry=scheduled.telemetry,
        fresh_replay_performed=True,
    )
    certificate_sha256 = tuple(
        certificate.certificate_sha256
        for certificate in scheduled.certificates
    )
    receipt: Dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "status": "k3_build_only_materialized_validated_consumed_and_released",
        "diagnostic_only": True,
        "candidate_only": True,
        "production_ready": False,
        "solver_handoff_ready": False,
        "ground_truth_parameter_accepted": False,
        "full_parent_lp_called": False,
        "proof_authority": False,
        "verdict_authority": False,
        "fresh_build_returned": False,
        "fresh_build_reference_released_before_return": True,
        "fresh_issue_called": True,
        "fresh_strict_verified_before_consume": True,
        "fresh_consumed_once": True,
        "fresh_inspected_after_consume": True,
        "fresh_live_verifier_valid_after_consume": False,
        "source_semantic_digest": source_digest,
        "terminal_source_semantic_digest": terminal_source,
        "source_dimensions": source_dimensions,
        "fresh_dimensions": inspection.fresh_dimensions,
        "source_payload_bytes": source_payload,
        "fresh_payload_bytes": inspection.fresh_payload_bytes,
        "fresh_payload_delta_bytes": inspection.fresh_payload_delta_bytes,
        "focused_rival_id": focused_rival_id,
        "focused_rival_binding_digest": rival.binding_digest,
        "selected_output_positions": selected_outputs,
        "selected_generator_nonzeros": selected_generator_nonzeros,
        "retained_k2_stable_bit_ids": plan.retained_k2_stable_bit_ids,
        "stable_bit_ids": plan.stable_bit_ids,
        "third_stable_bit_id": plan.third_stable_bit_id,
        "third_coefficient_exact": _fraction_pair(plan.third_coefficient),
        "preferred_third_phase": plan.preferred_third_phase,
        "preferred_phase_source": plan.preferred_phase_source,
        "ranking": tuple(
            (stable_id, *_fraction_pair(value))
            for stable_id, value in plan.ranking
        ),
        "ranking_sha256": plan.ranking_sha256,
        "pair_bundle_sha256": pair_bundle.bundle_sha256,
        "pair_query_count": len(pair_bundle.records),
        "pair_record_status": tuple(record.status for record in pair_bundle.records),
        "pair_models_closed": all(record.model_closed for record in pair_bundle.records),
        "canonical_patterns": schedule.canonical_patterns,
        "canonical_pattern_count": len(schedule.canonical_patterns),
        "active_pattern_mask": schedule.active_pattern_mask,
        "certified_empty_pattern_count": schedule.active_pattern_mask.count(False),
        "evaluation_schedule": schedule.evaluation_schedule,
        "worst_k2_children": schedule.worst_k2_children,
        "threshold_pattern_indices": schedule.threshold_pattern_indices,
        "strong_target_exact": _fraction_pair(_K3_STRONG_TARGET),
        "strict_comparison": "observed_upper_exact_gt_target",
        "schedule_sha256": schedule.schedule_sha256,
        "scheduled_bundle_sha256": scheduled.bundle_sha256,
        "conditional_certificate_sha256": certificate_sha256,
        "fresh_issuance_sha256": inspection.issuance_sha256,
        "fresh_semantic_digest": inspection.fresh_semantic_digest,
        "eta_columns_added": _K3_PATTERN_COUNT,
        "equality_rows_added": inspection.equality_rows_added,
        "equality_rows_formula": "4+certified_empty_pattern_count",
        "upper_rows_added": inspection.upper_rows_added,
        "materialized_tightness_summary": inspection.materialized_tightness_summary,
        "materialized_tightness_summary_sha256": inspection.materialized_tightness_summary_sha256,
        "execution_telemetry": execution,
        "resource_entry_preflight": entry_resource,
        "resource_pre_s_preflight": pre_s_resource,
        "resource_pre_fresh_preflight": pre_fresh_resource,
        "resource_terminal_postflight": terminal_resource,
        "caps": _caps_payload(normalized_caps),
        "timings": {
            "pair_and_strict_replay_seconds": pair_seconds,
            "scheduled_producer_seconds": scheduled_seconds,
            "fresh_issue_seconds": inspection.issue_seconds,
            "fresh_consume_seconds": inspection.consume_seconds,
            "total_seconds": float(time.monotonic() - started),
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    kwargs = {
        "schema": _SCHEMA,
        "status": "k3_build_only_materialized_validated_consumed_and_released",
        "source_semantic_digest": source_digest,
        "terminal_source_semantic_digest": terminal_source,
        "focused_rival_id": focused_rival_id,
        "retained_k2_stable_bit_ids": plan.retained_k2_stable_bit_ids,
        "stable_bit_ids": plan.stable_bit_ids,
        "third_stable_bit_id": plan.third_stable_bit_id,
        "third_coefficient_exact": _fraction_pair(plan.third_coefficient),
        "preferred_third_phase": plan.preferred_third_phase,
        "ranking_sha256": plan.ranking_sha256,
        "pair_bundle_sha256": pair_bundle.bundle_sha256,
        "active_pattern_mask": schedule.active_pattern_mask,
        "evaluation_schedule": schedule.evaluation_schedule,
        "threshold_pattern_indices": schedule.threshold_pattern_indices,
        "scheduled_bundle_sha256": scheduled.bundle_sha256,
        "conditional_certificate_sha256": certificate_sha256,
        "fresh_issuance_sha256": inspection.issuance_sha256,
        "fresh_semantic_digest": inspection.fresh_semantic_digest,
        "source_dimensions": source_dimensions,
        "fresh_dimensions": inspection.fresh_dimensions,
        "materialized_tightness_summary": inspection.materialized_tightness_summary,
        "execution_telemetry": execution,
        "receipt": receipt,
        "diagnostic_sha256": "",
        "ground_truth_loaded": False,
        "full_parent_lp_called": False,
        "proof_authority": False,
        "verdict_authority": False,
    }
    provisional = PCOHK3BuildOnlyDiagnostic(**kwargs)
    result = PCOHK3BuildOnlyDiagnostic(
        **{
            **kwargs,
            "diagnostic_sha256": _canonical_sha256(
                _outcome_payload(provisional, include_digest=False)
            ),
        }
    )
    _check_deadline(deadline_value, "before_k3_success_anchor_and_return")
    _register_outcome(result)
    return result


def run_phase_conditioned_objective_hull_k3_build_only(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    retained_k2_stable_bit_ids: Tuple[int, int],
    deadline: float,
    caps: PCOHK3BuildOnlyCaps = PCOHK3BuildOnlyCaps(),
) -> PCOHK3BuildOnlyOutcome:
    """Run one pair-first K3 diagnostic with no HZ, proof, or verdict output."""

    try:
        return _run_phase_conditioned_objective_hull_k3_build_only(
            build,
            rivals,
            selection,
            focused_rival_id=focused_rival_id,
            retained_k2_stable_bit_ids=retained_k2_stable_bit_ids,
            deadline=deadline,
            caps=caps,
        )
    except PhaseConditionedK3BuildOnlyError:
        raise
    except BaseException as exc:
        detail = f"{type(exc).__name__}:{str(exc)[:240]}"
        _clear_exception_traceback(exc)
        raise PhaseConditionedK3BuildOnlyError(
            "k3_transaction_failed_closed:" + detail
        ) from None


def _resource_stop_structure_valid(
    result: PCOHK3BuildOnlyResourceStopDiagnostic,
) -> bool:
    receipt = result.receipt
    telemetry = result.execution_telemetry
    stage = result.stage
    if not _strict_resource_stop_contract_valid(
        _outcome_payload(result, include_digest=False), receipt
    ):
        return False
    rejection = receipt.get("resource_gate_rejection")
    forbidden_receipt_fields = {
        "conditional_certificate_sha256",
        "conditional_certificate_payload",
        "fresh_issuance_sha256",
        "fresh_semantic_digest",
        "fresh_dimensions",
        "fresh_payload_bytes",
        "fresh_payload_delta_bytes",
        "materialized_tightness_summary",
        "materialized_tightness_summary_sha256",
        "descriptor_representation_sha256",
    }
    before = tuple(receipt.get("fresh_registry_state_before", ()))
    terminal = tuple(receipt.get("fresh_registry_state_terminal", ()))
    if (
        set(receipt) != _RESOURCE_STOP_RECEIPT_KEYS
        or set(telemetry) != _RESOURCE_STOP_EXECUTION_KEYS
        or stage not in {"pre_scheduled", "pre_fresh_materialization"}
        or receipt.get("schema") != _RESOURCE_STOP_RECEIPT_SCHEMA
        or receipt.get("status")
        != "stopped_by_resource_gate_no_partial_output"
        or receipt.get("stage") != stage
        or receipt.get("reason") != result.reason
        or not _resource_gate_rejection_valid(
            rejection, stage=stage, caps=PCOHK3BuildOnlyCaps()
        )
        or rejection.get("reason") != result.reason
        or receipt.get("diagnostic_only") is not True
        or receipt.get("candidate_only") is not True
        or receipt.get("production_ready") is not False
        or receipt.get("solver_handoff_ready") is not False
        or receipt.get("ground_truth_parameter_accepted") is not False
        or receipt.get("full_parent_lp_called") is not False
        or receipt.get("proof_authority") is not False
        or receipt.get("verdict_authority") is not False
        or receipt.get("provenance_authority") is not False
        or receipt.get("authenticity_authority") is not False
        or result.provenance_authority is not False
        or result.authenticity_authority is not False
        or receipt.get("terminal_source_semantic_digest")
        != result.terminal_source_semantic_digest
        or result.partial_certificates_returned is not False
        or result.conditional_certificate_payload_returned is not False
        or result.fresh_issue_called is not False
        or result.fresh_build_returned is not False
        or result.fresh_descriptor_returned is not False
        or receipt.get("partial_certificates_returned") is not False
        or receipt.get("conditional_certificate_payload_returned") is not False
        or receipt.get("fresh_issue_called") is not False
        or receipt.get("fresh_build_returned") is not False
        or receipt.get("fresh_descriptor_returned") is not False
        or receipt.get("fresh_registry_entries_created") != 0
        or receipt.get("source_terminal_semantic_seal_read_count") != 2
        or any(name in receipt for name in forbidden_receipt_fields)
        or len(before) != 2
        or any(type(value) is not int or value < 0 for value in before)
        or terminal != before
        or receipt.get("scheduled_bundle_sha256")
        != result.scheduled_bundle_sha256
        or receipt.get("completed_conditional_certificate_count")
        != result.completed_conditional_certificate_count
        or telemetry.get("fresh_live_replay_performed") is not False
        or telemetry.get("fresh_live_replay_checker_actual_calls") != 0
        or not _resource_receipt_valid(
            receipt.get("resource_entry_preflight")
        )
    ):
        return False
    if stage == "pre_scheduled":
        return bool(
            result.scheduled_bundle_sha256 is None
            and result.completed_conditional_certificate_count == 0
            and receipt.get("scheduled_bundle_completed") is False
            and telemetry.get("scheduled_local_lp_actual_calls") == 0
            and telemetry.get("local_lp_actual_calls")
            == _K3_PAIR_QUERY_COUNT
            and telemetry.get("scheduled_patterns_completed") == 0
            and telemetry.get("scheduled_candidate_dual_accepted") == 0
            and telemetry.get("scheduled_producer_checker_actual_calls") == 0
            and telemetry.get("conditional_checker_actual_calls") == 0
            and _canonical_form(receipt.get("resource_pre_s_preflight"))
            == _canonical_form(rejection)
            and receipt.get("resource_pre_fresh_preflight") is None
        )
    return bool(
        _valid_sha256(result.scheduled_bundle_sha256)
        and result.completed_conditional_certificate_count == _K3_PATTERN_COUNT
        and receipt.get("scheduled_bundle_completed") is True
        and _resource_receipt_valid(
            receipt.get("resource_pre_s_preflight")
        )
        and _canonical_form(receipt.get("resource_pre_fresh_preflight"))
        == _canonical_form(rejection)
        and type(telemetry.get("scheduled_local_lp_actual_calls")) is int
        and 0 <= telemetry.get("scheduled_local_lp_actual_calls") <= _K3_PATTERN_COUNT
        and telemetry.get("local_lp_actual_calls")
        == _K3_PAIR_QUERY_COUNT
        + telemetry.get("scheduled_local_lp_actual_calls")
        and telemetry.get("scheduled_patterns_completed") == _K3_PATTERN_COUNT
        and type(telemetry.get("scheduled_candidate_dual_accepted")) is int
        and 0 <= telemetry.get("scheduled_candidate_dual_accepted") <= _K3_PATTERN_COUNT
        and telemetry.get("scheduled_producer_checker_actual_calls")
        == 1
        + _K3_PATTERN_COUNT
        + telemetry.get("scheduled_candidate_dual_accepted")
        and telemetry.get("conditional_checker_actual_calls")
        == telemetry.get("scheduled_producer_checker_actual_calls")
    )


def _structural_outcome_valid(result: Any) -> bool:
    try:
        canonical = tuple(itertools.product((-1, 1), repeat=3))
        if type(result) is PCOHK3BuildOnlyDiagnostic:
            digest = result.diagnostic_sha256
            expected_schema = _SCHEMA
            expected_status = "k3_build_only_materialized_validated_consumed_and_released"
        elif type(result) is PCOHK3BuildOnlyStopDiagnostic:
            digest = result.stop_sha256
            expected_schema = _STOP_SCHEMA
            expected_status = "stopped_by_strong_target_no_partial_output"
        elif type(result) is PCOHK3BuildOnlyResourceStopDiagnostic:
            digest = result.resource_stop_sha256
            expected_schema = _RESOURCE_STOP_SCHEMA
            expected_status = "stopped_by_resource_gate_no_partial_output"
        else:
            return False
        if (
            result.schema != expected_schema
            or result.status != expected_status
            or not _valid_sha256(digest)
            or _canonical_sha256(_outcome_payload(result, include_digest=False))
            != digest
            or result.source_semantic_digest
            != result.terminal_source_semantic_digest
            or not _valid_sha256(result.source_semantic_digest)
            or not _valid_sha256(result.ranking_sha256)
            or not _valid_sha256(result.pair_bundle_sha256)
            or result.retained_k2_stable_bit_ids != result.stable_bit_ids[:2]
            or len(result.stable_bit_ids) != 3
            or tuple(sorted(result.stable_bit_ids)) != result.stable_bit_ids
            or result.third_stable_bit_id not in result.stable_bit_ids[2:]
            or len(result.active_pattern_mask) != _K3_PATTERN_COUNT
            or any(type(value) is not bool for value in result.active_pattern_mask)
            or len(result.evaluation_schedule) != _K3_PATTERN_COUNT
            or set(result.evaluation_schedule) != set(canonical)
            or result.threshold_pattern_indices
            != tuple(
                index
                for index, pattern in enumerate(result.evaluation_schedule)
                if result.active_pattern_mask[canonical.index(pattern)]
            )
            or type(result.receipt) is not MappingProxyType
            or type(result.execution_telemetry) is not MappingProxyType
            or not _receipt_checksum_valid(result.receipt)
            or result.ground_truth_loaded is not False
            or result.full_parent_lp_called is not False
            or result.proof_authority is not False
            or result.verdict_authority is not False
        ):
            return False
        telemetry = result.execution_telemetry
        if (
            telemetry.get("pair_local_lp_actual_calls")
            != _K3_PAIR_QUERY_COUNT
            or type(telemetry.get("local_lp_actual_calls")) is not int
            or telemetry.get("local_lp_actual_calls")
            > _K3_LOCAL_LP_UPPER_BOUND
            or type(telemetry.get("conditional_checker_actual_calls"))
            is not int
            or telemetry.get("conditional_checker_actual_calls")
            > _K3_CONDITIONAL_CHECKER_UPPER_BOUND
            or result.receipt.get("source_semantic_digest")
            != result.source_semantic_digest
            or tuple(result.receipt.get("active_pattern_mask", ()))
            != result.active_pattern_mask
            or tuple(result.receipt.get("evaluation_schedule", ()))
            != result.evaluation_schedule
            or tuple(result.receipt.get("threshold_pattern_indices", ()))
            != result.threshold_pattern_indices
            or tuple(result.receipt.get("strong_target_exact", ()))
            != _fraction_pair(_K3_STRONG_TARGET)
            or result.receipt.get("proof_authority") is not False
            or result.receipt.get("verdict_authority") is not False
            or result.receipt.get("full_parent_lp_called") is not False
            or result.receipt.get("focused_rival_id")
            != result.focused_rival_id
            or tuple(result.receipt.get("retained_k2_stable_bit_ids", ()))
            != result.retained_k2_stable_bit_ids
            or tuple(result.receipt.get("stable_bit_ids", ()))
            != result.stable_bit_ids
            or result.receipt.get("third_stable_bit_id")
            != result.third_stable_bit_id
            or tuple(result.receipt.get("third_coefficient_exact", ()))
            != result.third_coefficient_exact
            or result.receipt.get("preferred_third_phase")
            != result.preferred_third_phase
            or result.receipt.get("ranking_sha256")
            != result.ranking_sha256
            or result.receipt.get("pair_bundle_sha256")
            != result.pair_bundle_sha256
            or result.receipt.get("pair_query_count")
            != _K3_PAIR_QUERY_COUNT
            or result.receipt.get("canonical_pattern_count")
            != _K3_PATTERN_COUNT
            or result.receipt.get("certified_empty_pattern_count")
            != result.active_pattern_mask.count(False)
            or _canonical_form(result.receipt.get("execution_telemetry"))
            != _canonical_form(result.execution_telemetry)
            or _canonical_form(result.receipt.get("caps"))
            != _canonical_form(_caps_payload(PCOHK3BuildOnlyCaps()))
            or not _resource_receipt_valid(
                result.receipt.get("resource_entry_preflight")
            )
            or (
                type(result) is not PCOHK3BuildOnlyResourceStopDiagnostic
                and not _resource_receipt_valid(
                    result.receipt.get("resource_pre_s_preflight")
                )
            )
            or (
                type(result) is not PCOHK3BuildOnlyResourceStopDiagnostic
                and not _resource_receipt_valid(
                    result.receipt.get("resource_terminal_postflight")
                )
            )
        ):
            return False
        if type(result) is PCOHK3BuildOnlyResourceStopDiagnostic:
            return _resource_stop_structure_valid(result)
        if type(result) is PCOHK3BuildOnlyStopDiagnostic:
            stop_record = result.receipt.get("scheduled_stop_record")
            return bool(
                result.partial_certificates_returned is False
                and result.fresh_issue_called is False
                and result.fresh_build_returned is False
                and result.observed_upper_exact[1] > 0
                and Fraction(*result.observed_upper_exact) > _K3_STRONG_TARGET
                and result.triggering_schedule_index
                in result.threshold_pattern_indices
                and result.triggering_pattern
                == result.evaluation_schedule[result.triggering_schedule_index]
                and result.receipt.get("partial_certificates_returned") is False
                and result.receipt.get("fresh_issue_called") is False
                and result.receipt.get("fresh_build_returned") is False
                and result.receipt.get("scheduled_stop_record_sha256")
                == result.scheduled_stop_record_sha256
                and isinstance(stop_record, Mapping)
                and stop_record.get("record_sha256")
                == result.scheduled_stop_record_sha256
                and stop_record.get("partial_certificates_returned") is False
                and stop_record.get("full_parent_lp_called") is False
                and stop_record.get("proof_authority") is False
                and stop_record.get("verdict_authority") is False
                and stop_record.get("structural_self_consistency_only") is True
                and stop_record.get("future_live_owner_anchor_required") is True
            )
        empty_count = result.active_pattern_mask.count(False)
        source = result.source_dimensions
        fresh = result.fresh_dimensions
        return bool(
            len(result.conditional_certificate_sha256) == _K3_PATTERN_COUNT
            and all(_valid_sha256(value) for value in result.conditional_certificate_sha256)
            and _valid_sha256(result.scheduled_bundle_sha256)
            and _valid_sha256(result.fresh_issuance_sha256)
            and _valid_sha256(result.fresh_semantic_digest)
            and len(source) == 5
            and len(fresh) == 5
            and fresh
            == (source[0], source[1] + 8, source[2], source[3] + 4 + empty_count, source[4] + 1)
            and type(result.materialized_tightness_summary) is MappingProxyType
            and tuple(result.materialized_tightness_summary.get("active_pattern_mask", ()))
            == result.active_pattern_mask
            and result.receipt.get("fresh_build_returned") is False
            and result.receipt.get("fresh_build_reference_released_before_return") is True
            and result.receipt.get("fresh_live_verifier_valid_after_consume") is False
            and result.receipt.get("equality_rows_added") == 4 + empty_count
            and result.receipt.get("upper_rows_added") == 1
            and result.receipt.get("eta_columns_added") == 8
            and tuple(result.receipt.get("source_dimensions", ())) == source
            and tuple(result.receipt.get("fresh_dimensions", ())) == fresh
            and result.receipt.get("scheduled_bundle_sha256")
            == result.scheduled_bundle_sha256
            and tuple(result.receipt.get("conditional_certificate_sha256", ()))
            == result.conditional_certificate_sha256
            and result.receipt.get("fresh_issuance_sha256")
            == result.fresh_issuance_sha256
            and result.receipt.get("fresh_semantic_digest")
            == result.fresh_semantic_digest
            and result.receipt.get("materialized_tightness_summary_sha256")
            == result.materialized_tightness_summary.get("summary_sha256")
            and _canonical_form(
                result.receipt.get("materialized_tightness_summary")
            )
            == _canonical_form(result.materialized_tightness_summary)
            and _resource_receipt_valid(
                result.receipt.get("resource_pre_fresh_preflight")
            )
        )
    except (
        PhaseConditionedK3BuildOnlyError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def verify_phase_conditioned_objective_hull_k3_build_only_outcome(
    result: Any,
) -> bool:
    """Verify structure plus the process-local identity/immutability anchor."""

    try:
        if not _structural_outcome_valid(result):
            return False
        with _OUTCOME_REGISTRY_LOCK:
            registered = _OUTCOME_REGISTRY.get(result)
        fields = tuple(result.__dataclass_fields__)
        if (
            type(registered) is not _RegisteredOutcome
            or registered.process_id != os.getpid()
            or registered.field_names != fields
            or tuple(vars(result)) != fields
        ):
            return False
        for name, original in zip(fields, registered.original_values):
            observed = getattr(result, name)
            if isinstance(original, Mapping):
                if observed is not original:
                    return False
            elif observed != original:
                return False
        return True
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False


def export_phase_conditioned_objective_hull_k3_build_only_detached(
    result: Any,
) -> Mapping[str, Any]:
    """Export JSON primitives only after same-process verification succeeds."""

    if not verify_phase_conditioned_objective_hull_k3_build_only_outcome(result):
        raise PhaseConditionedK3BuildOnlyError(
            "detached_export_requires_valid_process_local_anchor"
        )
    return _builtin_value(_outcome_payload(result, include_digest=True))


def _detached_resource_stop_valid(
    payload: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    if not _strict_resource_stop_contract_valid(payload, receipt):
        return False
    stage = payload.get("stage")
    telemetry = payload.get("execution_telemetry")
    rejection = receipt.get("resource_gate_rejection")
    expected_fields = set(
        PCOHK3BuildOnlyResourceStopDiagnostic.__dataclass_fields__
    ) - {"resource_stop_sha256"}
    forbidden_receipt_fields = {
        "conditional_certificate_sha256",
        "conditional_certificate_payload",
        "fresh_issuance_sha256",
        "fresh_semantic_digest",
        "fresh_dimensions",
        "fresh_payload_bytes",
        "fresh_payload_delta_bytes",
        "materialized_tightness_summary",
        "materialized_tightness_summary_sha256",
        "descriptor_representation_sha256",
    }
    if (
        set(payload) != expected_fields
        or set(receipt) != _RESOURCE_STOP_RECEIPT_KEYS
        or not isinstance(telemetry, Mapping)
        or set(telemetry) != _RESOURCE_STOP_EXECUTION_KEYS
        or stage not in {"pre_scheduled", "pre_fresh_materialization"}
        or payload.get("status")
        != "stopped_by_resource_gate_no_partial_output"
        or payload.get("source_semantic_digest")
        != payload.get("terminal_source_semantic_digest")
        or not _valid_sha256(payload.get("source_semantic_digest"))
        or receipt.get("schema") != _RESOURCE_STOP_RECEIPT_SCHEMA
        or receipt.get("status") != payload.get("status")
        or receipt.get("stage") != stage
        or receipt.get("reason") != payload.get("reason")
        or not _resource_gate_rejection_valid(
            rejection, stage=stage, caps=PCOHK3BuildOnlyCaps()
        )
        or rejection.get("reason") != payload.get("reason")
        or receipt.get("diagnostic_only") is not True
        or receipt.get("candidate_only") is not True
        or receipt.get("production_ready") is not False
        or receipt.get("solver_handoff_ready") is not False
        or receipt.get("ground_truth_parameter_accepted") is not False
        or receipt.get("provenance_authority") is not False
        or receipt.get("authenticity_authority") is not False
        or payload.get("provenance_authority") is not False
        or payload.get("authenticity_authority") is not False
        or _canonical_form(receipt.get("execution_telemetry"))
        != _canonical_form(telemetry)
        or _canonical_form(receipt.get("caps"))
        != _canonical_form(_caps_payload(PCOHK3BuildOnlyCaps()))
        or receipt.get("scheduled_bundle_sha256")
        != payload.get("scheduled_bundle_sha256")
        or receipt.get("completed_conditional_certificate_count")
        != payload.get("completed_conditional_certificate_count")
        or tuple(receipt.get("fresh_registry_state_before", ()))
        != tuple(receipt.get("fresh_registry_state_terminal", ()))
        or len(tuple(receipt.get("fresh_registry_state_before", ()))) != 2
        or any(
            type(value) is not int or value < 0
            for value in receipt.get("fresh_registry_state_before", ())
        )
        or receipt.get("fresh_registry_entries_created") != 0
        or receipt.get("source_terminal_semantic_seal_read_count") != 2
        or any(name in receipt for name in forbidden_receipt_fields)
        or not _resource_receipt_valid(
            receipt.get("resource_entry_preflight")
        )
        or payload.get("partial_certificates_returned") is not False
        or payload.get("conditional_certificate_payload_returned") is not False
        or payload.get("fresh_issue_called") is not False
        or payload.get("fresh_build_returned") is not False
        or payload.get("fresh_descriptor_returned") is not False
        or receipt.get("partial_certificates_returned") is not False
        or receipt.get("conditional_certificate_payload_returned") is not False
        or receipt.get("fresh_issue_called") is not False
        or receipt.get("fresh_build_returned") is not False
        or receipt.get("fresh_descriptor_returned") is not False
        or telemetry.get("pair_local_lp_actual_calls")
        != _K3_PAIR_QUERY_COUNT
        or telemetry.get("fresh_live_replay_performed") is not False
        or telemetry.get("fresh_live_replay_checker_actual_calls") != 0
    ):
        return False
    if stage == "pre_scheduled":
        return bool(
            payload.get("scheduled_bundle_sha256") is None
            and payload.get("completed_conditional_certificate_count") == 0
            and receipt.get("scheduled_bundle_completed") is False
            and telemetry.get("scheduled_local_lp_actual_calls") == 0
            and telemetry.get("local_lp_actual_calls")
            == _K3_PAIR_QUERY_COUNT
            and telemetry.get("scheduled_patterns_completed") == 0
            and telemetry.get("scheduled_producer_checker_actual_calls") == 0
            and telemetry.get("conditional_checker_actual_calls") == 0
            and _canonical_form(receipt.get("resource_pre_s_preflight"))
            == _canonical_form(rejection)
            and receipt.get("resource_pre_fresh_preflight") is None
        )
    scheduled_lp = telemetry.get("scheduled_local_lp_actual_calls")
    accepted = telemetry.get("scheduled_candidate_dual_accepted")
    return bool(
        _valid_sha256(payload.get("scheduled_bundle_sha256"))
        and payload.get("completed_conditional_certificate_count")
        == _K3_PATTERN_COUNT
        and receipt.get("scheduled_bundle_completed") is True
        and _resource_receipt_valid(receipt.get("resource_pre_s_preflight"))
        and _canonical_form(receipt.get("resource_pre_fresh_preflight"))
        == _canonical_form(rejection)
        and type(scheduled_lp) is int
        and 0 <= scheduled_lp <= _K3_PATTERN_COUNT
        and telemetry.get("local_lp_actual_calls")
        == _K3_PAIR_QUERY_COUNT + scheduled_lp
        and telemetry.get("scheduled_patterns_completed") == _K3_PATTERN_COUNT
        and type(accepted) is int
        and 0 <= accepted <= _K3_PATTERN_COUNT
        and telemetry.get("scheduled_producer_checker_actual_calls")
        == 1 + _K3_PATTERN_COUNT + accepted
        and telemetry.get("conditional_checker_actual_calls")
        == telemetry.get("scheduled_producer_checker_actual_calls")
    )


def verify_detached_phase_conditioned_objective_hull_k3_build_only(
    payload: Any,
    *,
    expected_sha256: str,
) -> bool:
    """Check detached self-consistency; never reconstruct authority or provenance."""

    try:
        if not isinstance(payload, Mapping) or not _valid_sha256(expected_sha256):
            return False
        copied = dict(payload)
        schema = copied.get("schema")
        if schema == _SCHEMA:
            digest_name = "diagnostic_sha256"
        elif schema == _STOP_SCHEMA:
            digest_name = "stop_sha256"
        elif schema == _RESOURCE_STOP_SCHEMA:
            digest_name = "resource_stop_sha256"
        else:
            return False
        embedded = copied.pop(digest_name, None)
        receipt = copied.get("receipt")
        if embedded != expected_sha256 or not isinstance(receipt, Mapping):
            return False
        receipt_body = dict(receipt)
        receipt_digest = receipt_body.pop("receipt_sha256", None)
        common_valid = bool(
            _canonical_sha256(copied) == expected_sha256
            and _valid_sha256(receipt_digest)
            and _canonical_sha256(receipt_body) == receipt_digest
            and copied.get("ground_truth_loaded") is False
            and copied.get("full_parent_lp_called") is False
            and copied.get("proof_authority") is False
            and copied.get("verdict_authority") is False
            and receipt.get("proof_authority") is False
            and receipt.get("verdict_authority") is False
            and receipt.get("full_parent_lp_called") is False
            and len(copied.get("active_pattern_mask", ())) == 8
            and len(copied.get("evaluation_schedule", ())) == 8
            and receipt.get("pair_query_count") == 12
        )
        if not common_valid:
            return False
        if schema == _RESOURCE_STOP_SCHEMA:
            return _detached_resource_stop_valid(copied, receipt)
        return True
    except (
        PhaseConditionedK3BuildOnlyError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


__all__ = [
    "PCOHK3BuildOnlyCaps",
    "PCOHK3BuildOnlyDiagnostic",
    "PCOHK3BuildOnlyOutcome",
    "PCOHK3BuildOnlyResourceStopDiagnostic",
    "PCOHK3BuildOnlyStopDiagnostic",
    "PCOHK3PairFirstSchedule",
    "PCOHK3ThirdBitPlan",
    "PhaseConditionedK3BuildOnlyError",
    "build_k3_pair_first_schedule",
    "export_phase_conditioned_objective_hull_k3_build_only_detached",
    "run_phase_conditioned_objective_hull_k3_build_only",
    "select_k3_third_bit_plan",
    "verify_detached_phase_conditioned_objective_hull_k3_build_only",
    "verify_phase_conditioned_objective_hull_k3_build_only_outcome",
]
