#!/usr/bin/env python3
"""Verdict-free build-only transaction for one live PCOH candidate.

This module is the only supported bridge from the toy-proven PCOH components
to a real-model *build* sentinel.  It deliberately returns no ``SparseHZono``
and never calls a verifier, ``hz_base_feasibility``, or
``hz_objbound_decide``.  Local HiGHS solves may propose duals/rays, but all
numeric authority remains with the existing exact/split replay checkers and
the final result still has neither proof nor verdict authority.

The transaction uses one caller-owned absolute deadline and fails before
exact-Fraction objective formation when a focused objective is not the small,
sparse two-output margin expected by the first CIFAR100 sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple
import weakref

import numpy as np

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    OperatorExactReLUPhaseSelection,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
    _build_complete_operator_phase_conditioned_objective_bounds_until,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_fresh_materializer import (
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
    PairLocalCaps,
    run_phase_conditioned_pair_infeasibility_candidate,
)
from act.back_end.solver.solver_hz import SparseHZono


_SCHEMA = "act.hybridz_pcoh_build_only_diagnostic.v2"
_RECEIPT_SCHEMA = "act.hybridz_pcoh_build_only_receipt.v2"
_FRESH_ISSUANCE_SCHEMA = (
    "act.hybridz_pc_objective_hull_fresh_build_issuance.toy.v2"
)
_FRESH_RECEIPT_SCHEMA = (
    "act.hybridz_pc_objective_hull_fresh_build_receipt.toy.v2"
)
_MATERIALIZED_TIGHTNESS_SCHEMA = (
    "act.hybridz_pc_materialized_tightness_summary.toy.v1"
)
_CONDITIONAL_CHECKER_ROUTE = (
    "native_hz_preformed_objective_split_csr_no_generator_read_v1"
)
_CONDITIONAL_CERTIFICATE_SCHEMA = (
    "act.operator_phase_conditioned_objective_bound.v2"
)
# Keep the imported identity as the fail-closed cleanup authority.  Tests and
# callers may wrap the public module binding for observation; a wrapper failure
# must not strand the already-registered private fresh build.
_FRESH_DISCARD_CLEANUP_AUTHORITY = (
    discard_live_phase_conditioned_objective_hull_fresh_build
)
_MIB = 1024 * 1024
_GIB = 1024 * _MIB


class PhaseConditionedBuildOnlyError(ValueError):
    """The diagnostic transaction failed closed without returning an HZ."""


def _default_pair_caps() -> PairLocalCaps:
    return PairLocalCaps(
        max_stable_bits=2,
        max_signed_pair_queries=4,
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
        capability_ttl_seconds=10.0,
        row_caps=PCOHRowMaterializationCaps(
            max_parent_continuous_columns=60_000,
            max_parent_binary_columns=4,
            max_eta_columns=4,
            max_rows=8,
            max_total_exact_nonzeros=70_000,
            max_exact_bits=4096,
        ),
    )


@dataclass(frozen=True)
class PCOHBuildOnlyCaps:
    """Caps for the first sparse-margin, one-instance build sentinel."""

    max_stable_bits: int = 2
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


@dataclass(frozen=True, eq=False)
class PCOHBuildOnlyDiagnostic:
    """Immutable receipt-only result; it intentionally owns no fresh HZ."""

    schema: str
    status: str
    source_semantic_digest: str
    terminal_source_semantic_digest: str
    focused_rival_id: int
    stable_bit_ids: Tuple[int, ...]
    conditional_certificate_sha256: Tuple[str, ...]
    pair_bundle_sha256: str
    fresh_issuance_sha256: str
    fresh_semantic_digest: str
    source_dimensions: Tuple[int, int, int, int, int]
    fresh_dimensions: Tuple[int, int, int, int, int]
    materialized_tightness_summary: Mapping[str, Any]
    receipt: Mapping[str, Any]
    diagnostic_sha256: str
    full_parent_lp_called: bool = False
    proof_authority: bool = False
    verdict_authority: bool = False

    def __post_init__(self) -> None:
        if self.full_parent_lp_called is not False:
            raise ValueError("build-only diagnostic never runs a full parent LP")
        if self.proof_authority is not False:
            raise ValueError("build-only diagnostic never has proof authority")
        if self.verdict_authority is not False:
            raise ValueError("build-only diagnostic never has verdict authority")
        if type(self.materialized_tightness_summary) is not MappingProxyType:
            object.__setattr__(
                self,
                "materialized_tightness_summary",
                MappingProxyType(dict(self.materialized_tightness_summary)),
            )
        if type(self.receipt) is not MappingProxyType:
            object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))


@dataclass(frozen=True)
class _RegisteredDiagnostic:
    process_id: int
    field_names: Tuple[str, ...]
    original_values: Tuple[Any, ...]


@dataclass(frozen=True)
class _FreshInspection:
    fresh_issuance_schema: str
    fresh_receipt_schema: str
    fresh_issuance_sha256: str
    fresh_receipt_sha256: str
    adapter_candidate_sha256: str
    descriptor_representation_sha256: str
    row_frame_sha256: str
    materialized_tightness_summary: Mapping[str, Any]
    issue_seconds: float
    consume_seconds: float
    fresh_payload_bytes: int
    fresh_payload_delta_bytes: int
    fresh_dimensions: Tuple[int, int, int, int, int]
    fresh_semantic_digest: str


_DIAGNOSTIC_REGISTRY_LOCK = threading.Lock()
_DIAGNOSTIC_REGISTRY: "weakref.WeakKeyDictionary[PCOHBuildOnlyDiagnostic, _RegisteredDiagnostic]" = weakref.WeakKeyDictionary()
_RESULT_FIELD_NAMES = tuple(PCOHBuildOnlyDiagnostic.__dataclass_fields__)


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise PhaseConditionedBuildOnlyError(
            f"deadline_exhausted:{stage}:no_diagnostic"
        )


def _deadline(value: Any) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise PhaseConditionedBuildOnlyError("absolute_deadline_invalid")
    result = float(value)
    if not math.isfinite(result) or time.monotonic() >= result:
        raise PhaseConditionedBuildOnlyError("absolute_deadline_invalid")
    return result


def _canonical_form(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PhaseConditionedBuildOnlyError("canonical_nonfinite_float")
        return {"__binary64_hex__": value.hex()}
    if isinstance(value, np.generic):
        return _canonical_form(value.item())
    if type(value) in {tuple, list}:
        return [_canonical_form(item) for item in value]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PhaseConditionedBuildOnlyError(
                    "canonical_mapping_key_not_string"
                )
            result[key] = _canonical_form(item)
        return result
    raise PhaseConditionedBuildOnlyError(
        f"canonical_unsupported:{type(value).__name__}"
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonical_form(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _deep_freeze_tightness_payload(value: Any) -> Any:
    """Detach the registered receipt payload without changing exact values."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) in {tuple, list}:
        return tuple(_deep_freeze_tightness_payload(item) for item in value)
    if isinstance(value, Mapping):
        frozen: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PhaseConditionedBuildOnlyError(
                    "tightness_payload_nonstring_key"
                )
            frozen[key] = _deep_freeze_tightness_payload(item)
        return MappingProxyType(frozen)
    raise PhaseConditionedBuildOnlyError(
        f"tightness_payload_unsupported:{type(value).__name__}"
    )


def _canonical_fraction_pair(value: Any, *, name: str) -> Fraction:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int for item in value)
        or value[1] <= 0
    ):
        raise PhaseConditionedBuildOnlyError(
            f"tightness_{name}_fraction_pair_invalid"
        )
    exact = Fraction(value[0], value[1])
    if (exact.numerator, exact.denominator) != value:
        raise PhaseConditionedBuildOnlyError(
            f"tightness_{name}_fraction_pair_noncanonical"
        )
    return exact


def _canonical_finite_hex(value: Any, *, name: str) -> Tuple[float, Fraction]:
    if type(value) is not str or len(value) > 32:
        raise PhaseConditionedBuildOnlyError(
            f"tightness_{name}_hex_invalid"
        )
    try:
        stored = float.fromhex(value)
    except (OverflowError, ValueError) as exc:
        raise PhaseConditionedBuildOnlyError(
            f"tightness_{name}_hex_invalid"
        ) from exc
    if not math.isfinite(stored) or stored.hex() != value:
        raise PhaseConditionedBuildOnlyError(
            f"tightness_{name}_hex_noncanonical_or_nonfinite"
        )
    return stored, Fraction.from_float(stored)


def _require_minimal_outward_hex(
    stored: float, exact: Fraction, *, name: str
) -> None:
    stored_exact = Fraction.from_float(stored)
    if stored_exact < exact:
        raise PhaseConditionedBuildOnlyError(
            f"tightness_{name}_not_outward"
        )
    predecessor = float(np.nextafter(stored, -np.inf))
    if math.isfinite(predecessor) and Fraction.from_float(predecessor) >= exact:
        raise PhaseConditionedBuildOnlyError(
            f"tightness_{name}_not_minimal_outward"
        )


def _strict_materialized_tightness_payload(
    payload: Any,
    *,
    source_semantic_digest: str,
    stable_bit_ids: Tuple[int, ...],
    conditional_certificate_sha256: Tuple[str, ...],
    adapter_candidate_sha256: str,
    descriptor_representation_sha256: str,
    row_frame_sha256: str,
) -> None:
    """Replay the detached summary structure, digest, and all disclosed joins."""

    if type(payload) is not MappingProxyType:
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_payload_not_frozen_mapping"
        )
    expected_fields = frozenset(
        PCOHFreshMaterializedTightnessSummary.__dataclass_fields__
    )
    if frozenset(payload) != expected_fields:
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_payload_field_set_invalid"
        )
    if (
        payload.get("schema") != _MATERIALIZED_TIGHTNESS_SCHEMA
        or payload.get("status") != "sound_materialized_structural_upper"
        or payload.get("diagnostic_only") is not True
        or payload.get("full_parent_lp_called") is not False
        or payload.get("proof_authority") is not False
        or payload.get("verdict_authority") is not False
        or payload.get("parent_semantic_digest") != source_semantic_digest
        or payload.get("stable_bit_ids") != stable_bit_ids
        or payload.get("adapter_candidate_sha256")
        != adapter_candidate_sha256
        or payload.get("descriptor_representation_sha256")
        != descriptor_representation_sha256
        or payload.get("row_frame_sha256") != row_frame_sha256
        or payload.get("conditional_certificate_schema")
        != _CONDITIONAL_CERTIFICATE_SCHEMA
        or payload.get("conditional_checker_route")
        != _CONDITIONAL_CHECKER_ROUTE
    ):
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_header_or_cross_binding_invalid"
        )
    digest_names = (
        "parent_semantic_digest",
        "adapter_candidate_sha256",
        "descriptor_representation_sha256",
        "row_frame_sha256",
        "objective_binding_sha256",
        "objective_envelope_sha256",
        "global_checker_sha256",
        "summary_sha256",
    )
    if any(not _valid_sha256(payload.get(name)) for name in digest_names):
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_digest_field_invalid"
        )

    expected_patterns = tuple(
        tuple(int(value) for value in pattern)
        for pattern in itertools.product((-1, 1), repeat=len(stable_bit_ids))
    )
    active_mask = payload.get("active_pattern_mask")
    if (
        len(stable_bit_ids) != 2
        or payload.get("canonical_patterns") != expected_patterns
        or type(active_mask) is not tuple
        or len(active_mask) != len(expected_patterns)
        or any(type(value) is not bool for value in active_mask)
        or not any(active_mask)
    ):
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_pattern_cover_invalid"
        )
    empty_digests = payload.get("empty_evidence_descriptor_sha256")
    certificate_digests = payload.get("conditional_certificate_sha256")
    pattern_digests = payload.get("conditional_pattern_sha256")
    selected_sources = payload.get("conditional_selected_source")
    pattern_upper_hex = payload.get("pattern_upper_hex")
    linked_support_pairs = payload.get("linked_support_exact")
    direct_support_pairs = payload.get("direct_eta_support_exact")
    pattern_count = len(expected_patterns)
    if (
        type(empty_digests) is not tuple
        or len(empty_digests) != active_mask.count(False)
        or any(not _valid_sha256(value) for value in empty_digests)
        or len(set(empty_digests)) != len(empty_digests)
        or certificate_digests != conditional_certificate_sha256
        or type(pattern_digests) is not tuple
        or len(pattern_digests) != pattern_count
        or any(not _valid_sha256(value) for value in pattern_digests)
        or len(set(pattern_digests)) != pattern_count
        or type(selected_sources) is not tuple
        or len(selected_sources) != pattern_count
        or any(
            type(value) is not str
            or value
            not in {
                "candidate_local_dual",
                "zero_dual_fixed_pattern",
                "global_cube_baseline",
            }
            for value in selected_sources
        )
        or type(pattern_upper_hex) is not tuple
        or len(pattern_upper_hex) != pattern_count
        or type(linked_support_pairs) is not tuple
        or len(linked_support_pairs) != pattern_count
        or type(direct_support_pairs) is not tuple
        or len(direct_support_pairs) != pattern_count
    ):
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_pattern_fields_invalid"
        )

    global_stored, global_upper = _canonical_finite_hex(
        payload.get("global_cube_upper_hex"), name="global_cube_upper"
    )
    global_exact = _canonical_fraction_pair(
        payload.get("global_cube_upper_exact"), name="global_cube_upper_exact"
    )
    if global_upper < global_exact:
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_global_cube_not_outward"
        )
    pattern_uppers = tuple(
        _canonical_finite_hex(value, name=f"pattern_upper_{index}")[1]
        for index, value in enumerate(pattern_upper_hex)
    )
    active_uppers = tuple(
        value
        for value, active in zip(pattern_uppers, active_mask)
        if active
    )
    _, ideal = _canonical_finite_hex(
        payload.get("ideal_union_upper_hex"), name="ideal_union_upper"
    )
    if ideal != max(active_uppers) or ideal > global_upper:
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_ideal_union_invalid"
        )

    center = _canonical_fraction_pair(
        payload.get("objective_center_exact"), name="objective_center"
    )
    raw_rhs = _canonical_fraction_pair(
        payload.get("row_raw_rhs_exact"), name="row_raw_rhs"
    )
    _, stored_rhs = _canonical_finite_hex(
        payload.get("row_stored_rhs_hex"), name="row_stored_rhs"
    )
    total_guard = _canonical_fraction_pair(
        payload.get("row_total_coefficient_guard_exact"),
        name="row_total_coefficient_guard",
    )
    free_mismatch = _canonical_fraction_pair(
        payload.get("free_parent_mismatch_exact"), name="free_parent_mismatch"
    )
    all_mismatch = _canonical_fraction_pair(
        payload.get("all_parent_mismatch_exact"), name="all_parent_mismatch"
    )
    if (
        total_guard < 0
        or free_mismatch < 0
        or all_mismatch < free_mismatch
        or stored_rhs < raw_rhs + total_guard
    ):
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_rounding_guard_invalid"
        )
    linked_support = tuple(
        _canonical_fraction_pair(value, name=f"linked_support_{index}")
        for index, value in enumerate(linked_support_pairs)
    )
    direct_support = tuple(
        _canonical_fraction_pair(value, name=f"direct_support_{index}")
        for index, value in enumerate(direct_support_pairs)
    )
    active_linked = tuple(
        value for value, active in zip(linked_support, active_mask) if active
    )
    active_direct = tuple(
        value for value, active in zip(direct_support, active_mask) if active
    )
    linked = center + stored_rhs + free_mismatch + max(active_linked)
    direct = center + stored_rhs + all_mismatch + max(active_direct)
    guarded = ideal + (stored_rhs - raw_rhs) + total_guard
    disclosed_linked = _canonical_fraction_pair(
        payload.get("materialized_linked_upper_exact"),
        name="materialized_linked_upper",
    )
    disclosed_direct = _canonical_fraction_pair(
        payload.get("materialized_direct_upper_exact"),
        name="materialized_direct_upper",
    )
    disclosed_guarded = _canonical_fraction_pair(
        payload.get("materialized_guard_upper_exact"),
        name="materialized_guard_upper",
    )
    linked_stored, linked_outward = _canonical_finite_hex(
        payload.get("materialized_linked_upper_hex"),
        name="materialized_linked_upper",
    )
    direct_stored, direct_outward = _canonical_finite_hex(
        payload.get("materialized_direct_upper_hex"),
        name="materialized_direct_upper",
    )
    guarded_stored, guarded_outward = _canonical_finite_hex(
        payload.get("materialized_guard_upper_hex"),
        name="materialized_guard_upper",
    )
    if (
        disclosed_linked != linked
        or disclosed_direct != direct
        or disclosed_guarded != guarded
        or not (ideal <= linked <= direct <= guarded)
        or linked_outward < linked
        or direct_outward < direct
        or guarded_outward < guarded
    ):
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_upper_chain_invalid"
        )
    _require_minimal_outward_hex(
        linked_stored, linked, name="materialized_linked_upper"
    )
    _require_minimal_outward_hex(
        direct_stored, direct, name="materialized_direct_upper"
    )
    _require_minimal_outward_hex(
        guarded_stored, guarded, name="materialized_guard_upper"
    )
    rounding_tax = _canonical_fraction_pair(
        payload.get("rounding_tax_exact"), name="rounding_tax"
    )
    if rounding_tax != linked - ideal or rounding_tax < 0:
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_rounding_tax_invalid"
        )
    final_stored, final_upper = _canonical_finite_hex(
        payload.get("final_structural_upper_hex"),
        name="final_structural_upper",
    )
    if (
        final_stored.hex() != min(global_stored, linked_stored).hex()
        or final_upper > global_upper
    ):
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_final_upper_invalid"
        )
    summary_body = dict(payload)
    summary_sha256 = summary_body.pop("summary_sha256", None)
    if (
        not _valid_sha256(summary_sha256)
        or _canonical_sha256(summary_body) != summary_sha256
    ):
        raise PhaseConditionedBuildOnlyError(
            "materialized_tightness_summary_digest_invalid"
        )


def verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
    payload: Any,
    *,
    expected_source_semantic_digest: str,
    expected_stable_bit_ids: Sequence[int],
    expected_conditional_certificate_sha256: Sequence[str],
    expected_summary_sha256: Any = None,
) -> bool:
    """Verify JSON using the caller-held summary SHA as a trusted anchor.

    The payload cannot authenticate its own coherently rehashed opaque digests;
    consequently ``expected_summary_sha256`` is mandatory even though ``None``
    is accepted at the Python boundary to provide a fail-closed boolean result.
    """

    try:
        if not _valid_sha256(expected_source_semantic_digest):
            return False
        if (
            type(expected_stable_bit_ids) not in {tuple, list}
            or len(expected_stable_bit_ids) != 2
            or any(
                type(value) is not int or value < 0
                for value in expected_stable_bit_ids
            )
        ):
            return False
        stable_ids = tuple(expected_stable_bit_ids)
        if (
            stable_ids != tuple(sorted(stable_ids))
            or len(set(stable_ids)) != len(stable_ids)
        ):
            return False
        if (
            type(expected_conditional_certificate_sha256)
            not in {tuple, list}
            or len(expected_conditional_certificate_sha256) != 4
            or any(
                not _valid_sha256(value)
                for value in expected_conditional_certificate_sha256
            )
        ):
            return False
        conditional_sha256 = tuple(
            expected_conditional_certificate_sha256
        )
        if len(set(conditional_sha256)) != len(conditional_sha256):
            return False
        if not _valid_sha256(expected_summary_sha256):
            return False
        frozen = _deep_freeze_tightness_payload(payload)
        if type(frozen) is not MappingProxyType:
            return False
        if frozen.get("summary_sha256") != expected_summary_sha256:
            return False
        adapter_candidate_sha256 = frozen.get("adapter_candidate_sha256")
        descriptor_representation_sha256 = frozen.get(
            "descriptor_representation_sha256"
        )
        row_frame_sha256 = frozen.get("row_frame_sha256")
        _strict_materialized_tightness_payload(
            frozen,
            source_semantic_digest=expected_source_semantic_digest,
            stable_bit_ids=stable_ids,
            conditional_certificate_sha256=conditional_sha256,
            adapter_candidate_sha256=adapter_candidate_sha256,
            descriptor_representation_sha256=(
                descriptor_representation_sha256
            ),
            row_frame_sha256=row_frame_sha256,
        )
        return True
    except (
        PhaseConditionedBuildOnlyError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _strict_positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PhaseConditionedBuildOnlyError(f"{name}_invalid")
    return value


def _normalize_caps(caps: Any) -> PCOHBuildOnlyCaps:
    if type(caps) is not PCOHBuildOnlyCaps:
        raise PhaseConditionedBuildOnlyError("caps_wrong_type")
    for name in (
        "max_stable_bits",
        "max_selected_output_terms",
        "max_selected_generator_nonzeros",
        "max_source_payload_bytes",
        "max_fresh_payload_bytes",
        "max_fresh_payload_delta_bytes",
        "static_additional_rss_budget_bytes",
        "max_process_rss_bytes",
        "min_mem_available_bytes",
        "min_cgroup_headroom_bytes",
    ):
        _strict_positive_int(getattr(caps, name), name=name)
    if (
        type(caps.candidate_timeout_seconds) is not float
        or not math.isfinite(caps.candidate_timeout_seconds)
        or not 0.0 < caps.candidate_timeout_seconds <= 5.0
        or type(caps.pair_caps) is not PairLocalCaps
        or type(caps.fresh_caps) is not PCOHFreshMaterializationCaps
        or type(caps.fresh_caps.row_caps) is not PCOHRowMaterializationCaps
    ):
        raise PhaseConditionedBuildOnlyError("nested_or_timeout_caps_invalid")
    if (
        caps.max_stable_bits > 4
        or caps.max_selected_output_terms != 2
        or caps.pair_caps.max_stable_bits < caps.max_stable_bits
        or caps.pair_caps.max_signed_pair_queries
        < 4 * math.comb(caps.max_stable_bits, 2)
        or caps.fresh_caps.row_caps.max_eta_columns
        < 2 ** caps.max_stable_bits
    ):
        raise PhaseConditionedBuildOnlyError("caps_exceed_build_only_contract")
    if caps != PCOHBuildOnlyCaps():
        raise PhaseConditionedBuildOnlyError(
            "build_only_caps_must_match_fixed_preregistered_profile"
        )
    return caps


def _caps_payload(caps: PCOHBuildOnlyCaps) -> Dict[str, Any]:
    return {
        "max_stable_bits": caps.max_stable_bits,
        "max_selected_output_terms": caps.max_selected_output_terms,
        "max_selected_generator_nonzeros": (
            caps.max_selected_generator_nonzeros
        ),
        "max_source_payload_bytes": caps.max_source_payload_bytes,
        "max_fresh_payload_bytes": caps.max_fresh_payload_bytes,
        "max_fresh_payload_delta_bytes": caps.max_fresh_payload_delta_bytes,
        "static_additional_rss_budget_bytes": (
            caps.static_additional_rss_budget_bytes
        ),
        "max_process_rss_bytes": caps.max_process_rss_bytes,
        "min_mem_available_bytes": caps.min_mem_available_bytes,
        "min_cgroup_headroom_bytes": caps.min_cgroup_headroom_bytes,
        "candidate_timeout_seconds_hex": caps.candidate_timeout_seconds.hex(),
        "pair_caps": {
            name: getattr(caps.pair_caps, name)
            for name in caps.pair_caps.__dataclass_fields__
        },
        "fresh_caps": {
            "max_parent_variables": caps.fresh_caps.max_parent_variables,
            "max_parent_rows": caps.fresh_caps.max_parent_rows,
            "max_parent_nonzeros": caps.fresh_caps.max_parent_nonzeros,
            "max_parent_buffer_items": caps.fresh_caps.max_parent_buffer_items,
            "max_tag_bytes": caps.fresh_caps.max_tag_bytes,
            "max_registry_entries": caps.fresh_caps.max_registry_entries,
            "capability_ttl_seconds_hex": (
                caps.fresh_caps.capability_ttl_seconds.hex()
            ),
            "row_caps": {
                name: getattr(caps.fresh_caps.row_caps, name)
                for name in caps.fresh_caps.row_caps.__dataclass_fields__
            },
        },
    }


def _parse_kib_field(path: Path, field: str) -> int:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PhaseConditionedBuildOnlyError(
            f"resource_read_failed:{path.name}"
        ) from exc
    prefix = field + ":"
    matches = tuple(line for line in lines if line.startswith(prefix))
    if len(matches) != 1:
        raise PhaseConditionedBuildOnlyError(
            f"resource_field_missing_or_duplicate:{field}"
        )
    parts = matches[0].split()
    if len(parts) != 3 or parts[0] != prefix or parts[2] != "kB":
        raise PhaseConditionedBuildOnlyError(
            f"resource_field_malformed:{field}"
        )
    try:
        kib = int(parts[1], 10)
    except ValueError as exc:
        raise PhaseConditionedBuildOnlyError(
            f"resource_field_malformed:{field}"
        ) from exc
    if kib < 0:
        raise PhaseConditionedBuildOnlyError(
            f"resource_field_negative:{field}"
        )
    return kib * 1024


def _live_cgroup_memory() -> Tuple[str, Any]:
    try:
        lines = Path("/proc/self/cgroup").read_text(
            encoding="ascii"
        ).splitlines()
    except (OSError, UnicodeError) as exc:
        raise PhaseConditionedBuildOnlyError(
            "resource_cgroup_membership_unreadable"
        ) from exc
    unified = tuple(line for line in lines if line.startswith("0::"))
    if len(unified) != 1:
        raise PhaseConditionedBuildOnlyError(
            "resource_cgroup_v2_membership_unavailable"
        )
    relative = unified[0][3:].lstrip("/")
    if any(part in {"", ".", ".."} for part in Path(relative).parts):
        raise PhaseConditionedBuildOnlyError(
            "resource_cgroup_path_malformed"
        )
    root = Path("/sys/fs/cgroup").resolve(strict=True)
    group = (root / relative).resolve(strict=True)
    try:
        group.relative_to(root)
    except ValueError as exc:
        raise PhaseConditionedBuildOnlyError(
            "resource_cgroup_path_escaped_root"
        ) from exc
    ancestors = []
    cursor = group
    while True:
        ancestors.append(cursor)
        if cursor == root:
            break
        cursor = cursor.parent
    bounded_headroom = []
    saw_memory_controller = False
    for index, ancestor in enumerate(ancestors):
        current_path = ancestor / "memory.current"
        maximum_path = ancestor / "memory.max"
        current_exists = current_path.is_file()
        maximum_exists = maximum_path.is_file()
        if not current_exists and not maximum_exists:
            if index == 0:
                raise PhaseConditionedBuildOnlyError(
                    "resource_cgroup_leaf_memory_controller_unavailable"
                )
            break
        if current_exists != maximum_exists:
            raise PhaseConditionedBuildOnlyError(
                "resource_cgroup_memory_files_incomplete"
            )
        saw_memory_controller = True
        try:
            raw_current = current_path.read_text(
                encoding="ascii"
            ).strip()
            raw_max = maximum_path.read_text(
                encoding="ascii"
            ).strip()
        except (OSError, UnicodeError) as exc:
            raise PhaseConditionedBuildOnlyError(
                "resource_cgroup_memory_unreadable"
            ) from exc
        try:
            current = int(raw_current, 10)
        except ValueError as exc:
            raise PhaseConditionedBuildOnlyError(
                "resource_cgroup_current_malformed"
            ) from exc
        if current < 0:
            raise PhaseConditionedBuildOnlyError(
                "resource_cgroup_current_negative"
            )
        if raw_max == "max":
            continue
        try:
            maximum = int(raw_max, 10)
        except ValueError as exc:
            raise PhaseConditionedBuildOnlyError(
                "resource_cgroup_max_malformed"
            ) from exc
        if maximum < 0:
            raise PhaseConditionedBuildOnlyError(
                "resource_cgroup_max_negative"
            )
        bounded_headroom.append(max(0, maximum - current))
    if not saw_memory_controller:
        raise PhaseConditionedBuildOnlyError(
            "resource_cgroup_memory_controller_unavailable"
        )
    if not bounded_headroom:
        return "unbounded", None
    return "bounded", min(bounded_headroom)


def _live_resource_snapshot() -> Mapping[str, Any]:
    cgroup_status, cgroup_headroom = _live_cgroup_memory()
    return MappingProxyType(
        {
            "current_rss_bytes": _parse_kib_field(
                Path("/proc/self/status"), "VmRSS"
            ),
            "peak_rss_bytes": _parse_kib_field(
                Path("/proc/self/status"), "VmHWM"
            ),
            "mem_available_bytes": _parse_kib_field(
                Path("/proc/meminfo"), "MemAvailable"
            ),
            "cgroup_limit_status": cgroup_status,
            "cgroup_headroom_bytes": cgroup_headroom,
            "measurement_source": (
                "live_proc_status_meminfo_and_cgroup_v2"
            ),
            "caller_supplied": False,
        }
    )


def _resource_preflight(
    *,
    current_rss_bytes: Any,
    peak_rss_bytes: Any,
    mem_available_bytes: Any,
    cgroup_limit_status: Any,
    cgroup_headroom_bytes: Any,
    caps: PCOHBuildOnlyCaps,
) -> Mapping[str, Any]:
    for name, value in (
        ("current_rss_bytes", current_rss_bytes),
        ("peak_rss_bytes", peak_rss_bytes),
        ("mem_available_bytes", mem_available_bytes),
    ):
        if type(value) is not int or value < 0:
            raise PhaseConditionedBuildOnlyError(
                f"resource_preflight_{name}_invalid"
            )
    if cgroup_limit_status not in {"bounded", "unbounded"}:
        raise PhaseConditionedBuildOnlyError(
            "resource_preflight_cgroup_status_unavailable"
        )
    if cgroup_limit_status == "bounded":
        if type(cgroup_headroom_bytes) is not int or cgroup_headroom_bytes < 0:
            raise PhaseConditionedBuildOnlyError(
                "resource_preflight_cgroup_headroom_invalid"
            )
        cgroup_passed = (
            cgroup_headroom_bytes >= caps.min_cgroup_headroom_bytes
        )
    else:
        if cgroup_headroom_bytes is not None:
            raise PhaseConditionedBuildOnlyError(
                "resource_preflight_unbounded_cgroup_has_headroom"
            )
        cgroup_passed = True
    rss_forecast = current_rss_bytes + caps.static_additional_rss_budget_bytes
    conditions = {
        "current_plus_budget_strictly_below_process_cap": (
            rss_forecast < caps.max_process_rss_bytes
        ),
        "entry_peak_not_above_process_cap": (
            peak_rss_bytes <= caps.max_process_rss_bytes
        ),
        "mem_available_at_least_fixed_reserve": (
            mem_available_bytes >= caps.min_mem_available_bytes
        ),
        "cgroup_unbounded_or_has_fixed_reserve": cgroup_passed,
    }
    if not all(conditions.values()):
        raise PhaseConditionedBuildOnlyError(
            "resource_preflight_stop_loss:" + ",".join(
                name for name, passed in conditions.items() if not passed
            )
        )
    return MappingProxyType(
        {
            "current_rss_bytes": current_rss_bytes,
            "peak_rss_bytes": peak_rss_bytes,
            "mem_available_bytes": mem_available_bytes,
            "cgroup_limit_status": cgroup_limit_status,
            "cgroup_headroom_bytes": cgroup_headroom_bytes,
            "static_additional_rss_budget_bytes": (
                caps.static_additional_rss_budget_bytes
            ),
            "forecast_rss_bytes": rss_forecast,
            "max_process_rss_bytes": caps.max_process_rss_bytes,
            "min_mem_available_bytes": caps.min_mem_available_bytes,
            "min_cgroup_headroom_bytes": caps.min_cgroup_headroom_bytes,
            "conditions": MappingProxyType(dict(conditions)),
            "passed": True,
        }
    )


def _resource_postflight(
    snapshot: Mapping[str, Any], *, caps: PCOHBuildOnlyCaps
) -> Mapping[str, Any]:
    current = snapshot.get("current_rss_bytes")
    peak = snapshot.get("peak_rss_bytes")
    if type(current) is not int or type(peak) is not int:
        raise PhaseConditionedBuildOnlyError(
            "resource_postflight_rss_missing"
        )
    conditions = {
        "terminal_current_within_process_cap": (
            current <= caps.max_process_rss_bytes
        ),
        "observed_peak_within_process_cap": (
            peak <= caps.max_process_rss_bytes
        ),
        "live_kernel_measurement_not_caller_supplied": (
            snapshot.get("caller_supplied") is False
        ),
    }
    if not all(conditions.values()):
        raise PhaseConditionedBuildOnlyError(
            "resource_postflight_stop_loss:" + ",".join(
                name for name, passed in conditions.items() if not passed
            )
        )
    return MappingProxyType(
        {**dict(snapshot), "conditions": MappingProxyType(conditions), "passed": True}
    )


def _payload_bytes(build: OperatorHZBuild) -> int:
    if type(build) is not OperatorHZBuild or type(build.hz) is not SparseHZono:
        raise PhaseConditionedBuildOnlyError("source_build_wrong_type")
    hz = build.hz
    values = [hz.c, hz.b, hz.ub, hz.col_ids, hz.bcol_ids]
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        matrix = getattr(hz, name)
        values.extend((matrix.data, matrix.indices, matrix.indptr))
    for name in (
        "full_col_ids",
        "operator_input_center",
        "operator_input_radius",
        "_solver_continuous_column_layer_ids",
    ):
        values.append(getattr(hz, name))
    values.append(build.input_col_ids)
    seen = set()
    total = 0
    for value in values:
        if type(value) is not np.ndarray:
            raise PhaseConditionedBuildOnlyError("source_payload_not_ndarray")
        if id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
    return total


def _shape(hz: SparseHZono) -> Tuple[int, int, int, int, int]:
    return (hz.n_out, hz.n_cont, hz.n_bin, hz.n_eq, hz.n_ub)


def _focused_rival(
    rivals: Sequence[RivalSpec], focused_rival_id: int, output_count: int
) -> RivalSpec:
    if type(rivals) not in {tuple, list} or not rivals:
        raise PhaseConditionedBuildOnlyError("rivals_not_nonempty_sequence")
    matches = tuple(
        rival
        for rival in rivals
        if type(rival) is RivalSpec and rival.rival_id == focused_rival_id
    )
    if len(matches) != 1:
        raise PhaseConditionedBuildOnlyError("focused_rival_not_unique")
    rival = matches[0]
    if len(rival.objective) != output_count:
        raise PhaseConditionedBuildOnlyError("focused_objective_width_mismatch")
    if any(type(value) is not float or not math.isfinite(value) for value in rival.objective):
        raise PhaseConditionedBuildOnlyError("focused_objective_noncanonical")
    if type(rival.threshold) is not float or not math.isfinite(rival.threshold):
        raise PhaseConditionedBuildOnlyError("focused_threshold_noncanonical")
    return rival


def _sparse_margin_preflight(
    hz: SparseHZono,
    rival: RivalSpec,
    *,
    caps: PCOHBuildOnlyCaps,
    deadline: float,
) -> Tuple[Tuple[int, ...], int]:
    selected_outputs = tuple(
        index for index, value in enumerate(rival.objective) if value != 0.0
    )
    if len(selected_outputs) != caps.max_selected_output_terms:
        raise PhaseConditionedBuildOnlyError(
            "focused_objective_not_exact_two_output_margin"
        )
    selected_weights = tuple(rival.objective[index] for index in selected_outputs)
    if tuple(sorted(selected_weights)) != (-1.0, 1.0):
        raise PhaseConditionedBuildOnlyError(
            "focused_objective_not_unit_signed_class_margin"
        )
    touched = 0
    for output in selected_outputs:
        touched += int(hz.Gc.indptr[output + 1] - hz.Gc.indptr[output])
        touched += int(hz.Gb.indptr[output + 1] - hz.Gb.indptr[output])
    _check_deadline(deadline, "after_sparse_margin_preflight")
    if touched > caps.max_selected_generator_nonzeros:
        raise PhaseConditionedBuildOnlyError(
            "focused_generator_nonzero_cap_exceeded_before_fraction"
        )
    return selected_outputs, touched


def _stable_ids(
    selection: OperatorExactReLUPhaseSelection,
    values: Any,
    *,
    caps: PCOHBuildOnlyCaps,
) -> Tuple[int, ...]:
    if type(values) is not tuple:
        raise PhaseConditionedBuildOnlyError("stable_bit_ids_not_tuple")
    if len(values) != caps.max_stable_bits:
        raise PhaseConditionedBuildOnlyError("stable_bit_count_out_of_cap")
    if any(type(value) is not int or value < 0 for value in values):
        raise PhaseConditionedBuildOnlyError("stable_bit_ids_malformed")
    result = tuple(values)
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise PhaseConditionedBuildOnlyError("stable_bit_ids_not_canonical")
    mapping_ids = {mapping.stable_bcol_id for mapping in selection.mappings}
    if any(value not in mapping_ids for value in result):
        raise PhaseConditionedBuildOnlyError("stable_bit_missing_from_selection")
    return result


def _diagnostic_payload(
    result: PCOHBuildOnlyDiagnostic, *, include_digest: bool
) -> Dict[str, Any]:
    payload = {
        "schema": result.schema,
        "status": result.status,
        "source_semantic_digest": result.source_semantic_digest,
        "terminal_source_semantic_digest": (
            result.terminal_source_semantic_digest
        ),
        "focused_rival_id": result.focused_rival_id,
        "stable_bit_ids": result.stable_bit_ids,
        "conditional_certificate_sha256": (
            result.conditional_certificate_sha256
        ),
        "pair_bundle_sha256": result.pair_bundle_sha256,
        "fresh_issuance_sha256": result.fresh_issuance_sha256,
        "fresh_semantic_digest": result.fresh_semantic_digest,
        "source_dimensions": result.source_dimensions,
        "fresh_dimensions": result.fresh_dimensions,
        "materialized_tightness_summary": (
            result.materialized_tightness_summary
        ),
        "receipt": result.receipt,
        "full_parent_lp_called": result.full_parent_lp_called,
        "proof_authority": result.proof_authority,
        "verdict_authority": result.verdict_authority,
    }
    if include_digest:
        payload["diagnostic_sha256"] = result.diagnostic_sha256
    return payload


def _clear_nested_exception_traceback(exc: BaseException) -> None:
    """Remove inner frames that may have owned a private registry build."""

    current_code = _clear_nested_exception_traceback.__code__
    traceback_cursor = exc.__traceback__
    while traceback_cursor is not None:
        frame = traceback_cursor.tb_frame
        traceback_cursor = traceback_cursor.tb_next
        if frame.f_code is current_code:
            continue
        try:
            frame.clear()
        except RuntimeError:
            pass
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None


def _discard_unconsumed_fresh_issuance(issuance: Any) -> Tuple[bool, str]:
    """Use the public cleanup first and the captured authority as fallback."""

    public_detail = ""
    try:
        public_result = discard_live_phase_conditioned_objective_hull_fresh_build(
            issuance,
            issuance.capability,
        )
    except BaseException as public_exc:
        public_result = False
        public_detail = "public_discard_interrupted:" + type(public_exc).__name__
        _clear_nested_exception_traceback(public_exc)
    if public_result is True:
        return True, public_detail
    if not public_detail:
        public_detail = "public_discard_rejected"
    try:
        fallback_result = _FRESH_DISCARD_CLEANUP_AUTHORITY(
            issuance,
            issuance.capability,
        )
    except BaseException as fallback_exc:
        fallback_result = False
        fallback_detail = (
            "fallback_discard_interrupted:" + type(fallback_exc).__name__
        )
        _clear_nested_exception_traceback(fallback_exc)
    else:
        fallback_detail = (
            "fallback_discard_succeeded"
            if fallback_result is True
            else "fallback_discard_rejected"
        )
    return fallback_result is True, public_detail + ":" + fallback_detail


def _issue_consume_inspect_release(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_ids: Tuple[int, ...],
    certificates: Tuple[Any, ...],
    pair_bundle: Any,
    deadline: float,
    caps: PCOHBuildOnlyCaps,
    source_shape: Tuple[int, int, int, int, int],
    source_payload: int,
    source_semantic_digest: str,
) -> Tuple[Any, Any]:
    """Never raise and never return a fresh HZ, even on hostile failures."""

    stage_started = time.monotonic()
    issuance = None
    fresh_build = None
    inspection = None
    error = None
    consumed = False
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
        issue_seconds = time.monotonic() - stage_started
        if (
            issuance.schema != _FRESH_ISSUANCE_SCHEMA
            or type(issuance.receipt) is not MappingProxyType
            or issuance.receipt.get("schema") != _FRESH_RECEIPT_SCHEMA
            or not verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
                issuance
            )
        ):
            raise PhaseConditionedBuildOnlyError(
                "fresh_materialized_tightness_strict_verification_failed"
            )
        _check_deadline(deadline, "after_fresh_tightness_strict_verification")
        receipt = issuance.receipt
        summary_object = issuance.materialized_tightness_summary
        registered_summary_payload = receipt.get(
            "materialized_tightness_summary"
        )
        expected_summary_payload = {
            name: getattr(summary_object, name)
            for name in PCOHFreshMaterializedTightnessSummary.__dataclass_fields__
        }
        if (
            type(summary_object) is not PCOHFreshMaterializedTightnessSummary
            or type(registered_summary_payload) is not MappingProxyType
            or _canonical_form(registered_summary_payload)
            != _canonical_form(expected_summary_payload)
            or receipt.get("materialized_tightness_summary_schema")
            != summary_object.schema
            or receipt.get("materialized_tightness_summary_sha256")
            != summary_object.summary_sha256
            or receipt.get("materialized_tightness_full_parent_lp_called")
            is not False
        ):
            raise PhaseConditionedBuildOnlyError(
                "fresh_registered_tightness_payload_rejected"
            )
        tightness_payload = _deep_freeze_tightness_payload(
            registered_summary_payload
        )
        conditional_sha256 = tuple(
            certificate.certificate_sha256 for certificate in certificates
        )
        _strict_materialized_tightness_payload(
            tightness_payload,
            source_semantic_digest=source_semantic_digest,
            stable_bit_ids=stable_ids,
            conditional_certificate_sha256=conditional_sha256,
            adapter_candidate_sha256=issuance.adapter_candidate_sha256,
            descriptor_representation_sha256=(
                issuance.descriptor_representation_sha256
            ),
            row_frame_sha256=issuance.row_frame_sha256,
        )
        fresh_receipt_sha256 = receipt.get("receipt_sha256")
        if not _valid_sha256(fresh_receipt_sha256):
            raise PhaseConditionedBuildOnlyError(
                "fresh_registered_receipt_digest_invalid"
            )
        fresh_build = consume_live_phase_conditioned_objective_hull_fresh_build(
            issuance,
            issuance.capability,
            deadline=deadline,
        )
        consumed = True
        consume_seconds = time.monotonic() - stage_started - issue_seconds
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
            raise PhaseConditionedBuildOnlyError(
                "fresh_verdict_firewall_rejected"
            )
        source_payload_receipt = receipt.get("source_payload_bytes")
        fresh_payload = receipt.get("fresh_payload_bytes")
        fresh_delta = receipt.get("fresh_payload_delta_bytes")
        fresh_shape = _shape(fresh_build.hz)
        expected_fresh_shape = (
            source_shape[0],
            source_shape[1] + 2 ** len(stable_ids),
            source_shape[2],
            source_shape[3] + len(issuance.equality_row_tags),
            source_shape[4] + len(issuance.upper_row_tags),
        )
        if (
            type(source_payload_receipt) is not int
            or source_payload_receipt != source_payload
            or type(fresh_payload) is not int
            or type(fresh_delta) is not int
            or fresh_payload - source_payload != fresh_delta
            or fresh_payload > caps.max_fresh_payload_bytes
            or fresh_delta > caps.max_fresh_payload_delta_bytes
            or fresh_shape != expected_fresh_shape
            or tuple(receipt.get("source_dimensions", ())) != source_shape
            or tuple(receipt.get("fresh_dimensions", ())) != fresh_shape
            or len(issuance.eta_col_ids) != 4
            or not 3 <= len(issuance.equality_row_tags) <= 7
            or len(issuance.upper_row_tags) != 1
            or receipt.get("production_ready") is not False
            or receipt.get("proof_authority") is not False
            or receipt.get("verdict_authority") is not False
            or receipt.get("constructive_nonempty_inherited") is not False
            or receipt.get("source_buffers_borrowed_by_fresh") is not False
            or receipt.get("fresh_buffers_readonly") is not True
            or receipt.get("uses_sparse_hstack") is not False
            or receipt.get("uses_sparse_vstack") is not False
        ):
            raise PhaseConditionedBuildOnlyError(
                "fresh_receipt_or_shape_rejected"
            )
        fresh_digest = sparse_hz_semantic_digest(fresh_build.hz)
        if fresh_digest != issuance.fresh_semantic_digest:
            raise PhaseConditionedBuildOnlyError(
                "fresh_terminal_digest_mismatch"
            )
        inspection = _FreshInspection(
            fresh_issuance_schema=issuance.schema,
            fresh_receipt_schema=receipt["schema"],
            fresh_issuance_sha256=issuance.issuance_sha256,
            fresh_receipt_sha256=fresh_receipt_sha256,
            adapter_candidate_sha256=issuance.adapter_candidate_sha256,
            descriptor_representation_sha256=(
                issuance.descriptor_representation_sha256
            ),
            row_frame_sha256=issuance.row_frame_sha256,
            materialized_tightness_summary=tightness_payload,
            issue_seconds=float(issue_seconds),
            consume_seconds=float(consume_seconds),
            fresh_payload_bytes=fresh_payload,
            fresh_payload_delta_bytes=fresh_delta,
            fresh_dimensions=fresh_shape,
            fresh_semantic_digest=fresh_digest,
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}:{str(exc)[:240]}"
        _clear_nested_exception_traceback(exc)
    finally:
        if issuance is not None and not consumed:
            discarded, cleanup_detail = _discard_unconsumed_fresh_issuance(
                issuance
            )
            if discarded is not True:
                inspection = None
                error = (
                    str(error)[:180]
                    + ":fresh_registry_cleanup_failed:"
                    + cleanup_detail[:160]
                )
            elif cleanup_detail:
                error = (
                    str(error)[:180]
                    + ":fresh_registry_cleanup_recovered:"
                    + cleanup_detail[:160]
                )
        fresh_build = None
        issuance = None
    return inspection, error


def run_phase_conditioned_objective_hull_build_only(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    deadline: float,
    caps: PCOHBuildOnlyCaps = PCOHBuildOnlyCaps(),
) -> PCOHBuildOnlyDiagnostic:
    """Materialize, consume, validate, and release one fresh candidate HZ."""

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
        raise PhaseConditionedBuildOnlyError("top_level_input_wrong_type")
    live_resources = _live_resource_snapshot()
    resource_preflight = _resource_preflight(
        current_rss_bytes=live_resources["current_rss_bytes"],
        peak_rss_bytes=live_resources["peak_rss_bytes"],
        mem_available_bytes=live_resources["mem_available_bytes"],
        cgroup_limit_status=live_resources["cgroup_limit_status"],
        cgroup_headroom_bytes=live_resources["cgroup_headroom_bytes"],
        caps=normalized_caps,
    )
    resource_preflight = MappingProxyType(
        {
            **dict(resource_preflight),
            "measurement_source": live_resources["measurement_source"],
            "caller_supplied": live_resources["caller_supplied"],
        }
    )
    stable_ids = _stable_ids(selection, stable_bit_ids, caps=normalized_caps)
    hz = build.hz
    source_shape = _shape(hz)
    source_payload = _payload_bytes(build)
    if source_payload > normalized_caps.max_source_payload_bytes:
        raise PhaseConditionedBuildOnlyError(
            "source_payload_cap_exceeded_before_fraction"
        )
    source_digest = sparse_hz_semantic_digest(hz)
    if source_digest != selection.parent_semantic_digest:
        raise PhaseConditionedBuildOnlyError("selection_parent_digest_stale")
    rival = _focused_rival(rivals, focused_rival_id, hz.n_out)
    selected_outputs, selected_generator_nonzeros = _sparse_margin_preflight(
        hz, rival, caps=normalized_caps, deadline=deadline_value
    )

    stage_started = time.monotonic()
    certificates = _build_complete_operator_phase_conditioned_objective_bounds_until(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_bit_ids=stable_ids,
        deadline=deadline_value,
        candidate_timeout_seconds=normalized_caps.candidate_timeout_seconds,
    )
    conditional_seconds = time.monotonic() - stage_started
    if len(certificates) != 2 ** len(stable_ids):
        raise PhaseConditionedBuildOnlyError("conditional_cover_incomplete")
    _check_deadline(deadline_value, "after_conditional_cover")

    stage_started = time.monotonic()
    pair_bundle = run_phase_conditioned_pair_infeasibility_candidate(
        build,
        rivals,
        selection,
        stable_bit_ids=stable_ids,
        deadline=deadline_value,
        caps=normalized_caps.pair_caps,
    )
    pair_seconds = time.monotonic() - stage_started
    expected_pair_queries = 4 * math.comb(len(stable_ids), 2)
    if (
        len(pair_bundle.records) != expected_pair_queries
        or pair_bundle.status != "complete"
        or any(not record.model_closed for record in pair_bundle.records)
        or any(
            record.status
            in {"deadline_expired", "candidate_error", "model_close_failed"}
            for record in pair_bundle.records
        )
    ):
        raise PhaseConditionedBuildOnlyError("pair_query_cover_incomplete")
    _check_deadline(deadline_value, "after_pair_bundle")

    inspection, inspection_error = _issue_consume_inspect_release(
        build,
        rivals,
        selection,
        focused_rival_id=focused_rival_id,
        stable_ids=stable_ids,
        certificates=certificates,
        pair_bundle=pair_bundle,
        deadline=deadline_value,
        caps=normalized_caps,
        source_shape=source_shape,
        source_payload=source_payload,
        source_semantic_digest=source_digest,
    )
    if inspection_error is not None or type(inspection) is not _FreshInspection:
        raise PhaseConditionedBuildOnlyError(
            "fresh_issue_consume_inspect_release_failed:"
            + str(inspection_error)[:260]
        ) from None
    fresh_issuance_schema = inspection.fresh_issuance_schema
    fresh_receipt_schema = inspection.fresh_receipt_schema
    fresh_issuance_sha256 = inspection.fresh_issuance_sha256
    fresh_receipt_sha256 = inspection.fresh_receipt_sha256
    adapter_candidate_sha256 = inspection.adapter_candidate_sha256
    descriptor_representation_sha256 = (
        inspection.descriptor_representation_sha256
    )
    row_frame_sha256 = inspection.row_frame_sha256
    tightness_summary = inspection.materialized_tightness_summary
    issue_seconds = inspection.issue_seconds
    consume_seconds = inspection.consume_seconds
    fresh_payload = inspection.fresh_payload_bytes
    fresh_delta = inspection.fresh_payload_delta_bytes
    fresh_shape = inspection.fresh_dimensions
    fresh_digest = inspection.fresh_semantic_digest
    terminal_source = sparse_hz_semantic_digest(build.hz)
    _check_deadline(deadline_value, "after_fresh_release_and_source_seal")
    if terminal_source != source_digest:
        raise PhaseConditionedBuildOnlyError("terminal_source_digest_changed")
    resource_postflight = _resource_postflight(
        _live_resource_snapshot(), caps=normalized_caps
    )
    _check_deadline(deadline_value, "after_live_resource_postflight")

    timings = {
        "conditional_seconds": float(conditional_seconds),
        "pair_seconds": float(pair_seconds),
        "fresh_issue_seconds": float(issue_seconds),
        "fresh_consume_seconds": float(consume_seconds),
        "total_seconds": float(time.monotonic() - started),
    }
    receipt_payload: Dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "status": "build_only_materialized_validated_and_released",
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only_sentinel_ready": True,
        "solver_handoff_ready": False,
        "production_ready": False,
        "proof_authority": False,
        "verdict_authority": False,
        "full_parent_lp_called": False,
        "fresh_build_returned": False,
        "fresh_build_released_before_return": True,
        "final_verifier_solver_called": False,
        "hz_base_feasibility_called": False,
        "hz_objbound_decide_called": False,
        "full_parent_lp_solver_called": False,
        "ground_truth_parameter_accepted": False,
        "local_candidate_solver_value_authority": False,
        "local_candidates_require_independent_replay": True,
        "shared_absolute_deadline": True,
        "source_semantic_digest": source_digest,
        "terminal_source_semantic_digest": terminal_source,
        "focused_rival_id": focused_rival_id,
        "focused_rival_binding_digest": rival.binding_digest,
        "selected_output_positions": selected_outputs,
        "selected_output_term_count": len(selected_outputs),
        "selected_generator_nonzeros": selected_generator_nonzeros,
        "stable_bit_ids": stable_ids,
        "pattern_count": len(certificates),
        "pair_query_count": len(pair_bundle.records),
        "pair_bundle_status": pair_bundle.status,
        "pair_models_closed": all(
            record.model_closed for record in pair_bundle.records
        ),
        "pair_record_status": tuple(
            record.status for record in pair_bundle.records
        ),
        "source_dimensions": source_shape,
        "fresh_dimensions": fresh_shape,
        "fresh_issuance_schema": fresh_issuance_schema,
        "fresh_receipt_schema": fresh_receipt_schema,
        "fresh_receipt_sha256": fresh_receipt_sha256,
        "fresh_adapter_candidate_sha256": adapter_candidate_sha256,
        "fresh_descriptor_representation_sha256": (
            descriptor_representation_sha256
        ),
        "fresh_row_frame_sha256": row_frame_sha256,
        "materialized_tightness_summary": tightness_summary,
        "materialized_tightness_summary_schema": tightness_summary["schema"],
        "materialized_tightness_summary_sha256": (
            tightness_summary["summary_sha256"]
        ),
        "materialized_tightness_full_parent_lp_called": False,
        "materialized_tightness_strict_verified_before_consume": True,
        "materialized_tightness_live_verifier_valid_after_consume": False,
        "source_payload_bytes": source_payload,
        "fresh_payload_bytes": fresh_payload,
        "fresh_payload_delta_bytes": fresh_delta,
        "static_additional_rss_budget_bytes": (
            normalized_caps.static_additional_rss_budget_bytes
        ),
        "resource_preflight": resource_preflight,
        "resource_postflight": resource_postflight,
        "phase_entry_rss_gate_required_and_enforced": True,
        "phase_entry_rss_gate_formula": (
            "current_rss+static_additional_rss_budget<2.5GiB"
        ),
        "conditional_certificate_sha256": tuple(
            item.certificate_sha256 for item in certificates
        ),
        "pair_bundle_sha256": pair_bundle.bundle_sha256,
        "fresh_issuance_sha256": fresh_issuance_sha256,
        "fresh_semantic_digest": fresh_digest,
        "caps": _caps_payload(normalized_caps),
        "timings": timings,
    }
    receipt_payload["receipt_sha256"] = _canonical_sha256(receipt_payload)
    placeholder = PCOHBuildOnlyDiagnostic(
        schema=_SCHEMA,
        status="build_only_materialized_validated_and_released",
        source_semantic_digest=source_digest,
        terminal_source_semantic_digest=terminal_source,
        focused_rival_id=focused_rival_id,
        stable_bit_ids=stable_ids,
        conditional_certificate_sha256=tuple(
            item.certificate_sha256 for item in certificates
        ),
        pair_bundle_sha256=pair_bundle.bundle_sha256,
        fresh_issuance_sha256=fresh_issuance_sha256,
        fresh_semantic_digest=fresh_digest,
        source_dimensions=source_shape,
        fresh_dimensions=fresh_shape,
        materialized_tightness_summary=tightness_summary,
        receipt=MappingProxyType(receipt_payload),
        diagnostic_sha256="",
        full_parent_lp_called=False,
        proof_authority=False,
        verdict_authority=False,
    )
    result = PCOHBuildOnlyDiagnostic(
        **{
            **placeholder.__dict__,
            "diagnostic_sha256": _canonical_sha256(
                _diagnostic_payload(placeholder, include_digest=False)
            ),
        }
    )
    _check_deadline(deadline_value, "before_build_only_diagnostic_return")
    with _DIAGNOSTIC_REGISTRY_LOCK:
        _DIAGNOSTIC_REGISTRY[result] = _RegisteredDiagnostic(
            process_id=os.getpid(),
            field_names=tuple(vars(result)),
            original_values=tuple(
                getattr(result, name) for name in _RESULT_FIELD_NAMES
            ),
        )
    return result


def verify_phase_conditioned_objective_hull_build_only_diagnostic(
    result: Any,
) -> bool:
    """Strict structural replay; this verifier never grants authority."""

    try:
        if type(result) is not PCOHBuildOnlyDiagnostic:
            return False
        with _DIAGNOSTIC_REGISTRY_LOCK:
            registered = _DIAGNOSTIC_REGISTRY.get(result)
        if (
            type(registered) is not _RegisteredDiagnostic
            or registered.process_id != os.getpid()
            or tuple(vars(result)) != registered.field_names
            or registered.field_names != _RESULT_FIELD_NAMES
        ):
            return False
        for name, original in zip(
            _RESULT_FIELD_NAMES, registered.original_values
        ):
            observed = getattr(result, name)
            if name in {"receipt", "materialized_tightness_summary"}:
                if observed is not original:
                    return False
            elif observed != original:
                return False
        if (
            result.schema != _SCHEMA
            or result.status != "build_only_materialized_validated_and_released"
            or result.full_parent_lp_called is not False
            or result.proof_authority is not False
            or result.verdict_authority is not False
            or type(result.receipt) is not MappingProxyType
            or type(result.materialized_tightness_summary)
            is not MappingProxyType
            or not _valid_sha256(result.diagnostic_sha256)
            or not _valid_sha256(result.source_semantic_digest)
            or result.terminal_source_semantic_digest
            != result.source_semantic_digest
            or not _valid_sha256(result.pair_bundle_sha256)
            or not _valid_sha256(result.fresh_issuance_sha256)
            or not _valid_sha256(result.fresh_semantic_digest)
            or any(
                not _valid_sha256(value)
                for value in result.conditional_certificate_sha256
            )
        ):
            return False
        receipt = result.receipt
        receipt_body = dict(receipt)
        receipt_sha = receipt_body.pop("receipt_sha256", None)
        source_shape = result.source_dimensions
        fresh_shape = result.fresh_dimensions
        if (
            len(result.stable_bit_ids) != 2
            or tuple(sorted(result.stable_bit_ids)) != result.stable_bit_ids
            or len(result.conditional_certificate_sha256) != 4
            or len(set(result.conditional_certificate_sha256)) != 4
            or len(source_shape) != 5
            or len(fresh_shape) != 5
            or any(type(value) is not int or value < 0 for value in source_shape)
            or any(type(value) is not int or value < 0 for value in fresh_shape)
            or fresh_shape[0] != source_shape[0]
            or fresh_shape[1] != source_shape[1] + 4
            or fresh_shape[2] != source_shape[2]
            or not 3 <= fresh_shape[3] - source_shape[3] <= 7
            or fresh_shape[4] != source_shape[4] + 1
        ):
            return False
        preflight = receipt.get("resource_preflight")
        postflight = receipt.get("resource_postflight")
        adapter_candidate_sha256 = receipt.get(
            "fresh_adapter_candidate_sha256"
        )
        descriptor_representation_sha256 = receipt.get(
            "fresh_descriptor_representation_sha256"
        )
        row_frame_sha256 = receipt.get("fresh_row_frame_sha256")
        _strict_materialized_tightness_payload(
            result.materialized_tightness_summary,
            source_semantic_digest=result.source_semantic_digest,
            stable_bit_ids=result.stable_bit_ids,
            conditional_certificate_sha256=(
                result.conditional_certificate_sha256
            ),
            adapter_candidate_sha256=adapter_candidate_sha256,
            descriptor_representation_sha256=(
                descriptor_representation_sha256
            ),
            row_frame_sha256=row_frame_sha256,
        )
        return (
            _valid_sha256(receipt_sha)
            and _canonical_sha256(receipt_body) == receipt_sha
            and receipt.get("schema") == _RECEIPT_SCHEMA
            and receipt.get("status")
            == "build_only_materialized_validated_and_released"
            and receipt.get("diagnostic_only") is True
            and receipt.get("candidate_only") is True
            and receipt.get("build_only_sentinel_ready") is True
            and receipt.get("solver_handoff_ready") is False
            and receipt.get("production_ready") is False
            and receipt.get("proof_authority") is False
            and receipt.get("verdict_authority") is False
            and receipt.get("full_parent_lp_called") is False
            and receipt.get("fresh_build_returned") is False
            and receipt.get("fresh_build_released_before_return") is True
            and receipt.get("final_verifier_solver_called") is False
            and receipt.get("hz_base_feasibility_called") is False
            and receipt.get("hz_objbound_decide_called") is False
            and receipt.get("full_parent_lp_solver_called") is False
            and receipt.get("ground_truth_parameter_accepted") is False
            and receipt.get("phase_entry_rss_gate_required_and_enforced")
            is True
            and isinstance(preflight, Mapping)
            and preflight.get("passed") is True
            and preflight.get("caller_supplied") is False
            and isinstance(postflight, Mapping)
            and postflight.get("passed") is True
            and postflight.get("caller_supplied") is False
            and receipt.get("caps") == _caps_payload(PCOHBuildOnlyCaps())
            and receipt.get("pattern_count") == 4
            and receipt.get("pair_query_count") == 4
            and receipt.get("pair_bundle_status") == "complete"
            and receipt.get("pair_models_closed") is True
            and receipt.get("source_semantic_digest")
            == result.source_semantic_digest
            and receipt.get("terminal_source_semantic_digest")
            == result.terminal_source_semantic_digest
            and receipt.get("focused_rival_id") == result.focused_rival_id
            and tuple(receipt.get("stable_bit_ids", ()))
            == result.stable_bit_ids
            and tuple(receipt.get("source_dimensions", ()))
            == result.source_dimensions
            and tuple(receipt.get("fresh_dimensions", ()))
            == result.fresh_dimensions
            and tuple(receipt.get("conditional_certificate_sha256", ()))
            == result.conditional_certificate_sha256
            and receipt.get("pair_bundle_sha256") == result.pair_bundle_sha256
            and receipt.get("fresh_issuance_sha256")
            == result.fresh_issuance_sha256
            and receipt.get("fresh_semantic_digest")
            == result.fresh_semantic_digest
            and receipt.get("fresh_issuance_schema")
            == _FRESH_ISSUANCE_SCHEMA
            and receipt.get("fresh_receipt_schema") == _FRESH_RECEIPT_SCHEMA
            and _valid_sha256(receipt.get("fresh_receipt_sha256"))
            and _valid_sha256(adapter_candidate_sha256)
            and _valid_sha256(descriptor_representation_sha256)
            and _valid_sha256(row_frame_sha256)
            and receipt.get("materialized_tightness_summary")
            is result.materialized_tightness_summary
            and receipt.get("materialized_tightness_summary_schema")
            == _MATERIALIZED_TIGHTNESS_SCHEMA
            and receipt.get("materialized_tightness_summary_sha256")
            == result.materialized_tightness_summary.get("summary_sha256")
            and receipt.get(
                "materialized_tightness_full_parent_lp_called"
            )
            is False
            and receipt.get(
                "materialized_tightness_strict_verified_before_consume"
            )
            is True
            and receipt.get(
                "materialized_tightness_live_verifier_valid_after_consume"
            )
            is False
            and type(receipt.get("source_payload_bytes")) is int
            and 0 <= receipt.get("source_payload_bytes")
            <= PCOHBuildOnlyCaps().max_source_payload_bytes
            and type(receipt.get("fresh_payload_bytes")) is int
            and 0 <= receipt.get("fresh_payload_bytes")
            <= PCOHBuildOnlyCaps().max_fresh_payload_bytes
            and type(receipt.get("fresh_payload_delta_bytes")) is int
            and 0 <= receipt.get("fresh_payload_delta_bytes")
            <= PCOHBuildOnlyCaps().max_fresh_payload_delta_bytes
            and receipt.get("fresh_payload_bytes")
            - receipt.get("source_payload_bytes")
            == receipt.get("fresh_payload_delta_bytes")
            and _canonical_sha256(
                _diagnostic_payload(result, include_digest=False)
            )
            == result.diagnostic_sha256
        )
    except (
        PhaseConditionedBuildOnlyError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


__all__ = [
    "PCOHBuildOnlyCaps",
    "PCOHBuildOnlyDiagnostic",
    "PhaseConditionedBuildOnlyError",
    "run_phase_conditioned_objective_hull_build_only",
    "verify_phase_conditioned_objective_hull_build_only_diagnostic",
    "verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload",
]
