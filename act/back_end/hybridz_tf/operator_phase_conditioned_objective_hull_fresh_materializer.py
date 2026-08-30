#!/usr/bin/env python3
"""Toy-first one-copy materialization of a live-replayed PCOH descriptor.

The public issuer accepts raw conditional certificates and a raw pair bundle;
it never accepts a caller-supplied descriptor or a proof boolean.  It invokes
the live adapter, lowers the resulting exact rows through the independently
guarded binary64 row materializer, and copies the live parent directly into
one final-size detached :class:`SparseHZono`.

This is intentionally not wired into the verifier pipeline.  In particular it
does not forge or inherit a constructive-nonempty token.  The issued receipt
and opaque capability have neither proof nor verdict authority.  The sole
fresh build is kept in a process-local one-use registry until consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
import secrets
import threading
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.operator_phase_conditioned_live_adapter import (
    LivePhaseConditionedObjectiveHullCandidate,
    build_live_phase_conditioned_objective_hull_candidate,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
    OperatorPhaseConditionedObjectiveBoundCertificate,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    outward_float64,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_row_materializer import (
    PCOHBinary64LocalRowFrame,
    PCOHRowMaterializationCaps,
    materialize_phase_conditioned_objective_hull_row_frame,
    verify_phase_conditioned_objective_hull_row_frame,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_reserve_fresh_col_ids_above,
)


_ISSUANCE_SCHEMA = (
    "act.hybridz_pc_objective_hull_fresh_build_issuance.toy.v2"
)
_CAPABILITY_SCHEMA = (
    "act.hybridz_pc_objective_hull_fresh_build_capability.toy.v1"
)
_RECEIPT_SCHEMA = (
    "act.hybridz_pc_objective_hull_fresh_build_receipt.toy.v2"
)
_MATERIALIZED_TIGHTNESS_SCHEMA = (
    "act.hybridz_pc_materialized_tightness_summary.toy.v1"
)
_CONDITIONAL_CERTIFICATE_SCHEMA = (
    "act.operator_phase_conditioned_objective_bound.v2"
)
_CONDITIONAL_CHECKER_ROUTE = (
    "native_hz_preformed_objective_split_csr_no_generator_read_v1"
)
_FRAME_SCHEMA = "act.hybridz_pc_objective_hull_owned_hz_frame.toy.v1"
_EQ_TAG_PREFIX = "operator_pcoh_eq:toy:v1"
_UB_TAG_PREFIX = "operator_pcoh_upper:toy:v1"
_PRODUCER_CAPABILITY = object()
_CORE_DENSE_NAMES = ("c", "b", "ub", "col_ids", "bcol_ids")
_CORE_CSR_NAMES = ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")
_PROVENANCE_NAMES = (
    "full_col_ids",
    "operator_input_center",
    "operator_input_radius",
    "_solver_continuous_column_layer_ids",
)
_CORE_NAMES = frozenset((*_CORE_DENSE_NAMES, *_CORE_CSR_NAMES))
_FRESH_ATTRIBUTE_WHITELIST = _CORE_NAMES.union(
    _PROVENANCE_NAMES,
    {
        "_solver_constraint_row_tags",
        "_solver_row_constraint_prefix_frames",
    },
)
_BUILD_FIELD_WHITELIST = frozenset(
    {
        "hz",
        "input_col_ids",
        "input_layer_id",
        "output_layer_id",
        "assert_layer_id",
        "metadata",
        "property_upper_output",
        "property_upper_row_groups",
        "verified_preactivation_frame",
        "constructive_nonempty_seal",
        "performance_diagnostic",
    }
)
_ALLOWED_SOURCE_MODES = (False, (), None)
_COPY_CHUNK_BYTES = 1 << 20
_ID_VALIDATION_CHUNK = max(
    1, _COPY_CHUNK_BYTES // np.dtype(np.int64).itemsize
)


class PhaseConditionedObjectiveHullFreshMaterializationError(ValueError):
    """The live source cannot safely produce one detached toy fresh build."""


@dataclass(frozen=True)
class PCOHFreshMaterializationCaps:
    max_parent_variables: int = 6_000_000
    max_parent_rows: int = 6_000_000
    max_parent_nonzeros: int = 30_000_000
    max_parent_buffer_items: int = 100_000_000
    max_tag_bytes: int = 1_000_000
    max_registry_entries: int = 16
    capability_ttl_seconds: float = 30.0
    row_caps: PCOHRowMaterializationCaps = field(
        default_factory=PCOHRowMaterializationCaps
    )


@dataclass(frozen=True)
class PCOHFreshMaterializedTightnessSummary:
    """Receipt-only structural upper for the materialized binary64 PCOH.

    Every binary64 value is carried by its canonical ``float.hex()`` text.
    Exact aggregates use canonical numerator/denominator pairs.  The summary
    is diagnostic-only: it neither solves the full parent LP nor grants proof
    or verdict authority.
    """

    schema: str
    status: str
    parent_semantic_digest: str
    adapter_candidate_sha256: str
    descriptor_representation_sha256: str
    row_frame_sha256: str
    stable_bit_ids: Tuple[int, ...]
    canonical_patterns: Tuple[Tuple[int, ...], ...]
    active_pattern_mask: Tuple[bool, ...]
    empty_evidence_descriptor_sha256: Tuple[str, ...]
    conditional_certificate_schema: str
    conditional_certificate_sha256: Tuple[str, ...]
    conditional_pattern_sha256: Tuple[str, ...]
    conditional_selected_source: Tuple[str, ...]
    conditional_checker_route: str
    objective_binding_sha256: str
    objective_envelope_sha256: str
    global_checker_sha256: str
    global_cube_upper_exact: Tuple[int, int]
    global_cube_upper_hex: str
    pattern_upper_hex: Tuple[str, ...]
    objective_center_exact: Tuple[int, int]
    row_raw_rhs_exact: Tuple[int, int]
    row_stored_rhs_hex: str
    row_total_coefficient_guard_exact: Tuple[int, int]
    free_parent_mismatch_exact: Tuple[int, int]
    all_parent_mismatch_exact: Tuple[int, int]
    linked_support_exact: Tuple[Tuple[int, int], ...]
    direct_eta_support_exact: Tuple[Tuple[int, int], ...]
    ideal_union_upper_hex: str
    materialized_linked_upper_exact: Tuple[int, int]
    materialized_linked_upper_hex: str
    materialized_direct_upper_exact: Tuple[int, int]
    materialized_direct_upper_hex: str
    materialized_guard_upper_exact: Tuple[int, int]
    materialized_guard_upper_hex: str
    rounding_tax_exact: Tuple[int, int]
    final_structural_upper_hex: str
    diagnostic_only: bool
    full_parent_lp_called: bool
    proof_authority: bool
    verdict_authority: bool
    summary_sha256: str

    def __post_init__(self) -> None:
        if self.diagnostic_only is not True:
            raise ValueError("materialized tightness summary is diagnostic-only")
        if self.full_parent_lp_called is not False:
            raise ValueError("materialized tightness summary never runs a full LP")
        if self.proof_authority is not False:
            raise ValueError("materialized tightness summary has no proof authority")
        if self.verdict_authority is not False:
            raise ValueError("materialized tightness summary has no verdict authority")


@dataclass(frozen=True, eq=False)
class PCOHFreshBuildCapability:
    schema: str
    token: str
    process_id: int
    expires_monotonic: float
    parent_semantic_digest: str
    fresh_semantic_digest: str
    fresh_frame_sha256: str
    descriptor_representation_sha256: str
    proof_authority: bool = False
    verdict_authority: bool = False
    _producer_capability: Any = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.proof_authority is not False:
            raise ValueError("fresh capability never has proof authority")
        if self.verdict_authority is not False:
            raise ValueError("fresh capability never has verdict authority")


@dataclass(frozen=True, eq=False)
class PCOHFreshBuildIssuance:
    schema: str
    parent_semantic_digest: str
    terminal_parent_semantic_digest: str
    source_frame_sha256: str
    terminal_source_frame_sha256: str
    fresh_parent_prefix_semantic_digest: str
    fresh_parent_prefix_frame_sha256: str
    fresh_semantic_digest: str
    fresh_frame_sha256: str
    adapter_candidate_sha256: str
    descriptor_representation_sha256: str
    row_frame_sha256: str
    eta_col_ids: Tuple[int, ...]
    equality_row_tags: Tuple[str, ...]
    upper_row_tags: Tuple[str, ...]
    materialized_tightness_summary: PCOHFreshMaterializedTightnessSummary
    receipt: Mapping[str, Any]
    capability: PCOHFreshBuildCapability
    issuance_sha256: str
    proof_authority: bool = False
    verdict_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority is not False:
            raise ValueError("fresh issuance never has proof authority")
        if self.verdict_authority is not False:
            raise ValueError("fresh issuance never has verdict authority")


@dataclass
class _RegistryRecord:
    capability_ref: "weakref.ReferenceType[PCOHFreshBuildCapability]"
    issuance_ref: "weakref.ReferenceType[PCOHFreshBuildIssuance]"
    private_build: OperatorHZBuild
    owned_identity: Tuple[int, ...]
    process_id: int
    expires_monotonic: float
    fresh_semantic_digest: str
    fresh_frame_sha256: str
    issuance_sha256: str
    metadata_identity: int
    metadata_sha256: str
    build_field_names: Tuple[str, ...]
    critical_build_fields: Tuple[Any, ...]


@dataclass(frozen=True)
class _RegistryReservation:
    process_id: int
    expires_monotonic: float


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: Dict[str, _RegistryRecord] = {}
_REGISTRY_RESERVATIONS: Dict[str, _RegistryReservation] = {}

def _canonical_form(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "canonical_nonfinite_float"
            )
        return {"__binary64_hex__": value.hex()}
    if isinstance(value, np.generic):
        return _canonical_form(value.item())
    if type(value) in {tuple, list}:
        return [_canonical_form(item) for item in value]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PhaseConditionedObjectiveHullFreshMaterializationError(
                    "canonical_nonstring_key"
                )
            result[key] = _canonical_form(item)
        return result
    raise PhaseConditionedObjectiveHullFreshMaterializationError(
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
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "canonical_encoding_failed"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) in {tuple, list}:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fraction_pair(value: Fraction) -> Tuple[int, int]:
    if type(value) is not Fraction:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "tightness_fraction_not_exact"
        )
    return (int(value.numerator), int(value.denominator))


def _strict_fraction_pair(value: Any, *, name: str) -> Fraction:
    if (
        type(value) is not tuple
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"tightness_{name}_fraction_pair_invalid"
        )
    exact = Fraction(value[0], value[1])
    if (exact.numerator, exact.denominator) != value:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"tightness_{name}_fraction_pair_noncanonical"
        )
    return exact


def _strict_float_hex(value: Any, *, name: str) -> Tuple[float, Fraction]:
    if type(value) is not str or len(value) > 32:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"tightness_{name}_hex_invalid"
        )
    try:
        stored = float.fromhex(value)
    except (OverflowError, ValueError) as exc:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"tightness_{name}_hex_invalid"
        ) from exc
    if not math.isfinite(stored) or stored.hex() != value:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"tightness_{name}_hex_noncanonical_or_nonfinite"
        )
    return stored, Fraction.from_float(stored)


def _outward_hex(value: Fraction, *, name: str) -> str:
    try:
        return outward_float64(value).hex()
    except Exception as exc:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"tightness_{name}_has_no_finite_outward_binary64"
        ) from exc


def _materialized_tightness_payload(
    summary: PCOHFreshMaterializedTightnessSummary,
    *,
    include_digest: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        name: getattr(summary, name)
        for name in PCOHFreshMaterializedTightnessSummary.__dataclass_fields__
        if name != "summary_sha256"
    }
    if include_digest:
        payload["summary_sha256"] = summary.summary_sha256
    return payload


def _strict_replay_materialized_tightness_summary(
    summary: Any,
) -> None:
    """Recompute every disclosed scalar from canonical hex/exact aggregates."""

    if type(summary) is not PCOHFreshMaterializedTightnessSummary:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_summary_wrong_type"
        )
    if (
        summary.schema != _MATERIALIZED_TIGHTNESS_SCHEMA
        or summary.status != "sound_materialized_structural_upper"
        or summary.diagnostic_only is not True
        or summary.full_parent_lp_called is not False
        or summary.proof_authority is not False
        or summary.verdict_authority is not False
        or not _valid_sha256(summary.parent_semantic_digest)
        or not _valid_sha256(summary.adapter_candidate_sha256)
        or not _valid_sha256(summary.descriptor_representation_sha256)
        or not _valid_sha256(summary.row_frame_sha256)
        or summary.conditional_certificate_schema
        != _CONDITIONAL_CERTIFICATE_SCHEMA
        or summary.conditional_checker_route != _CONDITIONAL_CHECKER_ROUTE
        or not _valid_sha256(summary.objective_binding_sha256)
        or not _valid_sha256(summary.objective_envelope_sha256)
        or not _valid_sha256(summary.global_checker_sha256)
        or not _valid_sha256(summary.summary_sha256)
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_summary_header_invalid"
        )
    stable_ids = summary.stable_bit_ids
    if (
        type(stable_ids) is not tuple
        or not 1 <= len(stable_ids) <= 4
        or any(type(value) is not int or value < 0 for value in stable_ids)
        or tuple(sorted(stable_ids)) != stable_ids
        or len(set(stable_ids)) != len(stable_ids)
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_stable_ids_invalid"
        )
    expected_patterns = tuple(
        tuple(int(value) for value in pattern)
        for pattern in itertools.product((-1, 1), repeat=len(stable_ids))
    )
    pattern_count = len(expected_patterns)
    if (
        summary.canonical_patterns != expected_patterns
        or type(summary.active_pattern_mask) is not tuple
        or len(summary.active_pattern_mask) != pattern_count
        or any(type(value) is not bool for value in summary.active_pattern_mask)
        or not any(summary.active_pattern_mask)
        or type(summary.empty_evidence_descriptor_sha256) is not tuple
        or len(summary.empty_evidence_descriptor_sha256)
        != summary.active_pattern_mask.count(False)
        or len(set(summary.empty_evidence_descriptor_sha256))
        != len(summary.empty_evidence_descriptor_sha256)
        or any(
            not _valid_sha256(value)
            for value in summary.empty_evidence_descriptor_sha256
        )
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_pattern_cover_invalid"
        )
    tuple_fields = (
        summary.conditional_certificate_sha256,
        summary.conditional_pattern_sha256,
        summary.conditional_selected_source,
        summary.pattern_upper_hex,
        summary.linked_support_exact,
        summary.direct_eta_support_exact,
    )
    if any(type(value) is not tuple or len(value) != pattern_count for value in tuple_fields):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_pattern_field_shape_invalid"
        )
    if (
        len(set(summary.conditional_certificate_sha256)) != pattern_count
        or len(set(summary.conditional_pattern_sha256)) != pattern_count
        or any(
            not _valid_sha256(value)
            for value in (
                *summary.conditional_certificate_sha256,
                *summary.conditional_pattern_sha256,
            )
        )
        or any(
            type(value) is not str
            or value not in {
                "candidate_local_dual",
                "zero_dual_fixed_pattern",
                "global_cube_baseline",
            }
            for value in summary.conditional_selected_source
        )
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_conditional_binding_invalid"
        )

    global_stored, global_upper = _strict_float_hex(
        summary.global_cube_upper_hex, name="global_cube_upper"
    )
    global_exact = _strict_fraction_pair(
        summary.global_cube_upper_exact, name="global_cube_upper_exact"
    )
    if global_upper < global_exact:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_global_cube_not_outward"
        )
    pattern_uppers = tuple(
        _strict_float_hex(value, name=f"pattern_upper_{index}")[1]
        for index, value in enumerate(summary.pattern_upper_hex)
    )
    active_uppers = tuple(
        upper
        for upper, active in zip(pattern_uppers, summary.active_pattern_mask)
        if active
    )
    ideal_stored, ideal = _strict_float_hex(
        summary.ideal_union_upper_hex, name="ideal_union_upper"
    )
    if ideal != max(active_uppers) or ideal > global_upper:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_ideal_union_invalid"
        )
    center = _strict_fraction_pair(
        summary.objective_center_exact, name="objective_center"
    )
    raw_rhs = _strict_fraction_pair(
        summary.row_raw_rhs_exact, name="row_raw_rhs"
    )
    _, stored_rhs = _strict_float_hex(
        summary.row_stored_rhs_hex, name="row_stored_rhs"
    )
    total_guard = _strict_fraction_pair(
        summary.row_total_coefficient_guard_exact,
        name="row_total_coefficient_guard",
    )
    free_mismatch = _strict_fraction_pair(
        summary.free_parent_mismatch_exact, name="free_parent_mismatch"
    )
    all_mismatch = _strict_fraction_pair(
        summary.all_parent_mismatch_exact, name="all_parent_mismatch"
    )
    if (
        total_guard < 0
        or free_mismatch < 0
        or all_mismatch < free_mismatch
        or stored_rhs < raw_rhs + total_guard
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_rounding_guard_invalid"
        )
    linked_support = tuple(
        _strict_fraction_pair(value, name=f"linked_support_{index}")
        for index, value in enumerate(summary.linked_support_exact)
    )
    direct_support = tuple(
        _strict_fraction_pair(value, name=f"direct_eta_support_{index}")
        for index, value in enumerate(summary.direct_eta_support_exact)
    )
    active_linked = tuple(
        value
        for value, active in zip(linked_support, summary.active_pattern_mask)
        if active
    )
    active_direct = tuple(
        value
        for value, active in zip(direct_support, summary.active_pattern_mask)
        if active
    )
    linked = center + stored_rhs + free_mismatch + max(active_linked)
    direct = center + stored_rhs + all_mismatch + max(active_direct)
    guard = ideal + (stored_rhs - raw_rhs) + total_guard
    stored_linked, linked_outward = _strict_float_hex(
        summary.materialized_linked_upper_hex,
        name="materialized_linked_upper",
    )
    stored_direct, direct_outward = _strict_float_hex(
        summary.materialized_direct_upper_hex,
        name="materialized_direct_upper",
    )
    stored_guard, guard_outward = _strict_float_hex(
        summary.materialized_guard_upper_hex,
        name="materialized_guard_upper",
    )
    if (
        _strict_fraction_pair(
            summary.materialized_linked_upper_exact,
            name="materialized_linked_upper_exact",
        )
        != linked
        or _strict_fraction_pair(
            summary.materialized_direct_upper_exact,
            name="materialized_direct_upper_exact",
        )
        != direct
        or _strict_fraction_pair(
            summary.materialized_guard_upper_exact,
            name="materialized_guard_upper_exact",
        )
        != guard
        or stored_linked.hex() != _outward_hex(linked, name="linked_replay")
        or stored_direct.hex() != _outward_hex(direct, name="direct_replay")
        or stored_guard.hex() != _outward_hex(guard, name="guard_replay")
        or not (ideal <= linked <= direct <= guard)
        or linked_outward < linked
        or direct_outward < direct
        or guard_outward < guard
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_upper_chain_invalid"
        )
    rounding_tax = _strict_fraction_pair(
        summary.rounding_tax_exact, name="rounding_tax"
    )
    if rounding_tax != linked - ideal or rounding_tax < 0:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_rounding_tax_invalid"
        )
    final_stored, final_upper = _strict_float_hex(
        summary.final_structural_upper_hex,
        name="final_structural_upper",
    )
    expected_final = min(global_stored, stored_linked)
    if final_stored.hex() != expected_final.hex() or final_upper > global_upper:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_final_upper_invalid"
        )
    expected_sha = _canonical_sha256(
        _materialized_tightness_payload(summary, include_digest=False)
    )
    if expected_sha != summary.summary_sha256:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "materialized_tightness_summary_sha256_invalid"
        )


def _deadline(value: Any) -> float:
    if type(value) not in {int, float}:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "deadline_not_builtin_numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "deadline_nonfinite"
        )
    return result


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"deadline_exhausted:{stage}"
        )


def _caps_payload(caps: PCOHFreshMaterializationCaps) -> Dict[str, Any]:
    row = caps.row_caps
    return {
        "max_parent_variables": caps.max_parent_variables,
        "max_parent_rows": caps.max_parent_rows,
        "max_parent_nonzeros": caps.max_parent_nonzeros,
        "max_parent_buffer_items": caps.max_parent_buffer_items,
        "max_tag_bytes": caps.max_tag_bytes,
        "max_registry_entries": caps.max_registry_entries,
        "capability_ttl_seconds": caps.capability_ttl_seconds,
        "row_caps": {
            "max_parent_continuous_columns": (
                row.max_parent_continuous_columns
            ),
            "max_parent_binary_columns": row.max_parent_binary_columns,
            "max_eta_columns": row.max_eta_columns,
            "max_rows": row.max_rows,
            "max_total_exact_nonzeros": row.max_total_exact_nonzeros,
            "max_exact_bits": row.max_exact_bits,
        },
    }


def _normalize_caps(value: Any) -> PCOHFreshMaterializationCaps:
    if type(value) is not PCOHFreshMaterializationCaps:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "caps_wrong_type"
        )
    integer_values = (
        value.max_parent_variables,
        value.max_parent_rows,
        value.max_parent_nonzeros,
        value.max_parent_buffer_items,
        value.max_tag_bytes,
        value.max_registry_entries,
    )
    if any(type(item) is not int or item < 1 for item in integer_values):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "caps_integer_invalid"
        )
    if (
        value.max_parent_variables > 20_000_000
        or value.max_parent_rows > 20_000_000
        or value.max_parent_nonzeros > 100_000_000
        or value.max_parent_buffer_items > 400_000_000
        or value.max_tag_bytes > 16_000_000
        or value.max_registry_entries > 64
        or type(value.capability_ttl_seconds) is not float
        or not (0.01 <= value.capability_ttl_seconds <= 300.0)
        or type(value.row_caps) is not PCOHRowMaterializationCaps
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "caps_hard_limit_exceeded"
        )
    # The row module performs its own complete caps normalization.  Touch all
    # fields here so forged nested objects fail before any core allocation.
    row_values = tuple(_caps_payload(value)["row_caps"].values())
    if any(type(item) is not int or item < 1 for item in row_values):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "row_caps_invalid"
        )
    return value


def _strict_dense(
    value: Any,
    *,
    name: str,
    dtype: np.dtype,
    length: int,
    finite: bool,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(dtype)
        or value.ndim != 1
        or int(value.size) != int(length)
        or not value.flags.c_contiguous
        or (finite and not np.all(np.isfinite(value)))
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"source_dense_malformed:{name}"
        )
    return value


def _strict_csr(
    value: Any,
    *,
    name: str,
    shape: Tuple[int, int],
) -> sp.csr_matrix:
    if (
        type(value) is not sp.csr_matrix
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not value.has_canonical_format
        or not np.all(np.isfinite(value.data))
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"source_csr_malformed:{name}"
        )
    for buffer_name in ("data", "indices", "indptr"):
        buffer = vars(value).get(buffer_name)
        if (
            type(buffer) is not np.ndarray
            or buffer.ndim != 1
            or not buffer.flags.c_contiguous
        ):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                f"source_csr_buffer_malformed:{name}:{buffer_name}"
            )
    if value.indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"source_csr_index_dtype:{name}"
        )
    if value.indptr.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"source_csr_indptr_dtype:{name}"
        )
    return value


def _scan_nonnegative_stable_ids(
    value: np.ndarray,
    *,
    name: str,
    error: str,
    deadline: float,
) -> bool:
    """Return whether ``value`` is strictly increasing without a full copy."""

    strictly_increasing = True
    previous_last: Optional[int] = None
    for start in range(0, int(value.size), _ID_VALIDATION_CHUNK):
        chunk = value[start : start + _ID_VALIDATION_CHUNK]
        has_negative = bool(np.any(chunk < 0))
        _check_deadline(deadline, f"{name}_nonnegative_scan")
        if has_negative:
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                error
            )
        if previous_last is not None and int(chunk[0]) <= previous_last:
            strictly_increasing = False
        if chunk.size > 1:
            has_nonincrease = bool(np.any(chunk[1:] <= chunk[:-1]))
            _check_deadline(deadline, f"{name}_strict_order_scan")
            strictly_increasing = strictly_increasing and not has_nonincrease
        previous_last = int(chunk[-1])
    return strictly_increasing


def _sorted_unique_stable_id_copy(
    value: np.ndarray,
    *,
    name: str,
    error: str,
    deadline: float,
) -> np.ndarray:
    """Validate an unordered vector with exactly one full int64 copy."""

    sorted_copy = value.copy()
    _check_deadline(deadline, f"{name}_sort_copy")
    sorted_copy.sort(kind="quicksort")
    _check_deadline(deadline, f"{name}_inplace_sort")
    previous_last: Optional[int] = None
    for start in range(0, int(sorted_copy.size), _ID_VALIDATION_CHUNK):
        chunk = sorted_copy[start : start + _ID_VALIDATION_CHUNK]
        duplicate = (
            previous_last is not None and int(chunk[0]) == previous_last
        )
        if chunk.size > 1:
            duplicate = duplicate or bool(
                np.any(chunk[1:] == chunk[:-1])
            )
            _check_deadline(deadline, f"{name}_duplicate_scan")
        previous_last = int(chunk[-1])
        if duplicate:
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                error
            )
    sorted_copy.setflags(write=False)
    return sorted_copy


def _validate_stable_id_vector(
    value: np.ndarray,
    *,
    name: str,
    error: str,
    deadline: float,
    retain_sorted: bool,
) -> Optional[np.ndarray]:
    """Validate IDs, taking the zero-copy increasing fast path when possible."""

    increasing = _scan_nonnegative_stable_ids(
        value, name=name, error=error, deadline=deadline
    )
    if increasing:
        return value if retain_sorted else None
    sorted_copy = _sorted_unique_stable_id_copy(
        value, name=name, error=error, deadline=deadline
    )
    return sorted_copy if retain_sorted else None


def _reject_continuous_binary_id_overlap(
    continuous_ids: np.ndarray,
    sorted_binary_ids: np.ndarray,
    *,
    deadline: float,
) -> None:
    """Search in bounded chunks; never allocate a full intersection array."""

    if not continuous_ids.size or not sorted_binary_ids.size:
        return
    upper = int(sorted_binary_ids.size) - 1
    for start in range(0, int(continuous_ids.size), _ID_VALIDATION_CHUNK):
        chunk = continuous_ids[start : start + _ID_VALIDATION_CHUNK]
        positions = np.searchsorted(sorted_binary_ids, chunk, side="left")
        _check_deadline(deadline, "source_stable_id_overlap_search")
        np.minimum(positions, upper, out=positions)
        _check_deadline(deadline, "source_stable_id_overlap_clip")
        overlap = bool(np.any(sorted_binary_ids[positions] == chunk))
        _check_deadline(deadline, "source_stable_id_overlap_compare")
        if overlap:
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "source_continuous_binary_ids_overlap"
            )


@dataclass(frozen=True)
class _SourceLayout:
    build: OperatorHZBuild
    hz: SparseHZono
    tags: Tuple[str, ...]
    prefix_frames: Dict[Any, Any]
    provenance: Mapping[str, np.ndarray]
    input_col_ids: np.ndarray
    source_object_identity: Tuple[int, ...]
    payload_bytes: int
    buffer_items: int
    nonzeros: int
    tag_bytes: int
    dropped_attribute_names: Tuple[str, ...]


def _owned_objects(
    build: OperatorHZBuild,
    *,
    provenance: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[Any, ...]:
    hz = build.hz
    objects = [build, build.input_col_ids, hz]
    for name in _CORE_DENSE_NAMES:
        objects.append(getattr(hz, name))
    for name in _CORE_CSR_NAMES:
        matrix = getattr(hz, name)
        objects.extend((matrix, matrix.data, matrix.indices, matrix.indptr))
    source = (
        {name: getattr(hz, name) for name in _PROVENANCE_NAMES}
        if provenance is None
        else provenance
    )
    objects.extend(source[name] for name in _PROVENANCE_NAMES)
    objects.append(getattr(hz, "_solver_constraint_row_tags"))
    objects.append(getattr(hz, "_solver_row_constraint_prefix_frames"))
    return tuple(objects)


def _payload_bytes(build: OperatorHZBuild) -> int:
    arrays = []
    hz = build.hz
    arrays.extend(getattr(hz, name) for name in _CORE_DENSE_NAMES)
    for name in _CORE_CSR_NAMES:
        matrix = getattr(hz, name)
        arrays.extend((matrix.data, matrix.indices, matrix.indptr))
    arrays.extend(getattr(hz, name) for name in _PROVENANCE_NAMES)
    arrays.append(build.input_col_ids)
    seen = set()
    total = 0
    for value in arrays:
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += int(value.nbytes)
    return total


def _validate_source(
    build: Any,
    *,
    caps: PCOHFreshMaterializationCaps,
    deadline: float,
) -> _SourceLayout:
    _check_deadline(deadline, "source_validation_entry")
    if type(build) is not OperatorHZBuild or type(build.hz) is not SparseHZono:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_build_wrong_type"
        )
    if (
        build.property_upper_output is not _ALLOWED_SOURCE_MODES[0]
        or build.property_upper_row_groups != _ALLOWED_SOURCE_MODES[1]
        or build.verified_preactivation_frame is not _ALLOWED_SOURCE_MODES[2]
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_property_mode_unsupported"
        )
    if any(
        type(getattr(build, name)) is not int or getattr(build, name) < 0
        for name in ("input_layer_id", "output_layer_id", "assert_layer_id")
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_layer_ids_malformed"
        )
    hz = build.hz
    O, C, B, E, R = hz.n_out, hz.n_cont, hz.n_bin, hz.n_eq, hz.n_ub
    if (
        C + B > caps.max_parent_variables
        or O + E + R > caps.max_parent_rows
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_dimension_cap_exceeded"
        )
    _strict_dense(hz.c, name="c", dtype=np.float64, length=O, finite=True)
    _strict_dense(hz.b, name="b", dtype=np.float64, length=E, finite=True)
    _strict_dense(hz.ub, name="ub", dtype=np.float64, length=R, finite=True)
    col_ids = _strict_dense(
        hz.col_ids, name="col_ids", dtype=np.int64, length=C, finite=False
    )
    bcol_ids = _strict_dense(
        hz.bcol_ids,
        name="bcol_ids",
        dtype=np.int64,
        length=B,
        finite=False,
    )
    # Increasing IDs need no full allocation.  An unordered vector is checked
    # by one in-place-sorted int64 copy; copies are never live concurrently.
    _validate_stable_id_vector(
        col_ids,
        name="col_ids",
        error="source_stable_ids_invalid:col_ids",
        deadline=deadline,
        retain_sorted=False,
    )
    sorted_binary_ids = _validate_stable_id_vector(
        bcol_ids,
        name="bcol_ids",
        error="source_stable_ids_invalid:bcol_ids",
        deadline=deadline,
        retain_sorted=True,
    )
    if sorted_binary_ids is None:  # retain_sorted makes this unreachable.
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_binary_stable_id_validation_internal_error"
        )
    _reject_continuous_binary_id_overlap(
        col_ids, sorted_binary_ids, deadline=deadline
    )
    matrices = {
        "Gc": _strict_csr(hz.Gc, name="Gc", shape=(O, C)),
        "Gb": _strict_csr(hz.Gb, name="Gb", shape=(O, B)),
        "Ac": _strict_csr(hz.Ac, name="Ac", shape=(E, C)),
        "Ab": _strict_csr(hz.Ab, name="Ab", shape=(E, B)),
        "Auc": _strict_csr(hz.Auc, name="Auc", shape=(R, C)),
        "Aub": _strict_csr(hz.Aub, name="Aub", shape=(R, B)),
    }
    nonzeros = sum(int(matrix.nnz) for matrix in matrices.values())
    if nonzeros > caps.max_parent_nonzeros:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_nonzero_cap_exceeded"
        )
    input_ids = _strict_dense(
        build.input_col_ids,
        name="build_input_col_ids",
        dtype=np.int64,
        length=int(build.input_col_ids.size)
        if type(build.input_col_ids) is np.ndarray
        else -1,
        finite=False,
    )
    N = int(input_ids.size)
    _validate_stable_id_vector(
        input_ids,
        name="build_input_col_ids",
        error="source_input_stable_ids_invalid",
        deadline=deadline,
        retain_sorted=False,
    )
    provenance = {
        "full_col_ids": _strict_dense(
            getattr(hz, "full_col_ids", None),
            name="full_col_ids",
            dtype=np.int64,
            length=N,
            finite=False,
        ),
        "operator_input_center": _strict_dense(
            getattr(hz, "operator_input_center", None),
            name="operator_input_center",
            dtype=np.float64,
            length=N,
            finite=True,
        ),
        "operator_input_radius": _strict_dense(
            getattr(hz, "operator_input_radius", None),
            name="operator_input_radius",
            dtype=np.float64,
            length=N,
            finite=True,
        ),
        "_solver_continuous_column_layer_ids": _strict_dense(
            getattr(hz, "_solver_continuous_column_layer_ids", None),
            name="continuous_column_layer_ids",
            dtype=np.int64,
            length=C,
            finite=False,
        ),
    }
    if (
        not np.array_equal(input_ids, provenance["full_col_ids"])
        or np.any(provenance["operator_input_radius"] < 0.0)
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_input_provenance_mismatch"
        )
    tags = getattr(hz, "_solver_constraint_row_tags", None)
    prefix_frames = getattr(hz, "_solver_row_constraint_prefix_frames", None)
    if (
        type(tags) is not tuple
        or len(tags) != E + R
        or any(type(tag) is not str for tag in tags)
        or type(prefix_frames) is not dict
        or prefix_frames
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_row_metadata_malformed_or_unsupported"
        )
    try:
        encoded_tags = tuple(tag.encode("ascii") for tag in tags)
    except UnicodeEncodeError as exc:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_row_tag_nonascii"
        ) from exc
    tag_bytes = sum(len(value) for value in encoded_tags)
    if tag_bytes > caps.max_tag_bytes:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_tag_cap_exceeded"
        )
    if any("conditional" in name.lower() for name in vars(hz)):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_conditional_metadata_unsupported"
        )
    copied_names = _CORE_NAMES.union(
        _PROVENANCE_NAMES,
        {
            "_solver_constraint_row_tags",
            "_solver_row_constraint_prefix_frames",
        },
    )
    dropped = tuple(sorted(name for name in vars(hz) if name not in copied_names))
    objects = _owned_objects(build, provenance=provenance)
    buffer_items = sum(
        int(value.size)
        for value in objects
        if type(value) is np.ndarray
    )
    if buffer_items > caps.max_parent_buffer_items:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "source_buffer_item_cap_exceeded"
        )
    _check_deadline(deadline, "source_validation_complete")
    return _SourceLayout(
        build=build,
        hz=hz,
        tags=tags,
        prefix_frames=prefix_frames,
        provenance=MappingProxyType(provenance),
        input_col_ids=input_ids,
        source_object_identity=tuple(id(value) for value in objects),
        payload_bytes=_payload_bytes(build),
        buffer_items=buffer_items,
        nonzeros=nonzeros,
        tag_bytes=tag_bytes,
        dropped_attribute_names=dropped,
    )


def _hash_array(
    digest: "hashlib._Hash",
    *,
    name: str,
    value: np.ndarray,
    deadline: float,
) -> None:
    digest.update(name.encode("ascii") + b"\0")
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    raw = value.view(np.uint8).reshape(-1)
    for start in range(0, int(raw.size), _COPY_CHUNK_BYTES):
        _check_deadline(deadline, f"frame_hash:{name}")
        digest.update(memoryview(raw[start : start + _COPY_CHUNK_BYTES]))


def _frame_digest(
    *,
    semantic_digest: str,
    input_layer_id: int,
    output_layer_id: int,
    assert_layer_id: int,
    property_upper_output: bool,
    property_upper_row_groups: Tuple[Tuple[int, ...], ...],
    input_col_ids: np.ndarray,
    provenance: Mapping[str, np.ndarray],
    tags: Tuple[str, ...],
    deadline: float,
) -> str:
    _check_deadline(deadline, "frame_digest_entry")
    digest = hashlib.sha256()
    digest.update(_FRAME_SCHEMA.encode("ascii") + b"\0")
    digest.update(semantic_digest.encode("ascii"))
    digest.update(
        np.asarray(
            [input_layer_id, output_layer_id, assert_layer_id], dtype="<i8"
        ).tobytes()
    )
    digest.update(b"1" if property_upper_output else b"0")
    digest.update(_canonical_sha256(property_upper_row_groups).encode("ascii"))
    _hash_array(
        digest,
        name="build_input_col_ids",
        value=input_col_ids,
        deadline=deadline,
    )
    for name in _PROVENANCE_NAMES:
        _hash_array(
            digest,
            name=name,
            value=provenance[name],
            deadline=deadline,
        )
    digest.update(len(tags).to_bytes(8, "little", signed=False))
    for index, tag in enumerate(tags):
        if index % 256 == 0:
            _check_deadline(deadline, "frame_digest_tags")
        encoded = tag.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    _check_deadline(deadline, "frame_digest_complete")
    return digest.hexdigest()


def _source_frame_digest(
    layout: _SourceLayout,
    *,
    semantic_digest: str,
    deadline: float,
) -> str:
    build = layout.build
    return _frame_digest(
        semantic_digest=semantic_digest,
        input_layer_id=build.input_layer_id,
        output_layer_id=build.output_layer_id,
        assert_layer_id=build.assert_layer_id,
        property_upper_output=False,
        property_upper_row_groups=(),
        input_col_ids=layout.input_col_ids,
        provenance=layout.provenance,
        tags=layout.tags,
        deadline=deadline,
    )


def _copy_array(
    value: np.ndarray,
    *,
    deadline: float,
    stage: str,
    tail: Optional[np.ndarray] = None,
) -> np.ndarray:
    if tail is not None and (
        type(tail) is not np.ndarray
        or tail.dtype != value.dtype
        or tail.ndim != 1
        or not tail.flags.c_contiguous
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"tail_malformed:{stage}"
        )
    tail_size = 0 if tail is None else int(tail.size)
    result = np.empty(int(value.size) + tail_size, dtype=value.dtype)
    chunk_items = max(1, _COPY_CHUNK_BYTES // max(1, value.dtype.itemsize))
    for start in range(0, int(value.size), chunk_items):
        _check_deadline(deadline, f"copy:{stage}")
        stop = min(int(value.size), start + chunk_items)
        np.copyto(result[start:stop], value[start:stop], casting="no")
    if tail is not None:
        offset = int(value.size)
        for start in range(0, tail_size, chunk_items):
            _check_deadline(deadline, f"copy_tail:{stage}")
            stop = min(tail_size, start + chunk_items)
            np.copyto(
                result[offset + start : offset + stop],
                tail[start:stop],
                casting="no",
            )
    return result


def _wrap_owned_csr(
    *,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: Tuple[int, int],
    stage: str,
) -> sp.csr_matrix:
    result = sp.csr_matrix(
        (data, indices, indptr),
        shape=shape,
        dtype=np.float64,
        copy=False,
    )
    if not result.has_canonical_format:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"owned_csr_noncanonical:{stage}"
        )
    for expected, observed in (
        (data, result.data),
        (indices, result.indices),
        (indptr, result.indptr),
    ):
        if int(expected.size) and not np.shares_memory(expected, observed):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                f"owned_csr_reallocated:{stage}"
            )
    return result


def _copy_csr_with_tail(
    parent: sp.csr_matrix,
    tail: sp.csr_matrix,
    *,
    columns: int,
    deadline: float,
    stage: str,
) -> sp.csr_matrix:
    if (
        type(tail) is not sp.csr_matrix
        or tail.dtype != np.dtype(np.float64)
        or tail.shape[1] != columns
        or not tail.has_canonical_format
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"tail_csr_malformed:{stage}"
        )
    total_nnz = int(parent.nnz) + int(tail.nnz)
    index_dtype = parent.indices.dtype
    indptr_dtype = parent.indptr.dtype
    index_limit = int(np.iinfo(index_dtype).max)
    pointer_limit = int(np.iinfo(indptr_dtype).max)
    if columns - 1 > index_limit or total_nnz > pointer_limit:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            f"csr_integer_capacity_exceeded:{stage}"
        )
    data = np.empty(total_nnz, dtype=np.float64)
    indices = np.empty(total_nnz, dtype=index_dtype)
    indptr = np.empty(parent.shape[0] + tail.shape[0] + 1, dtype=indptr_dtype)
    parent_nnz = int(parent.nnz)
    chunk = max(1, _COPY_CHUNK_BYTES // 8)
    for start in range(0, parent_nnz, chunk):
        _check_deadline(deadline, f"csr_parent_copy:{stage}")
        stop = min(parent_nnz, start + chunk)
        np.copyto(data[start:stop], parent.data[start:stop], casting="no")
        np.copyto(
            indices[start:stop], parent.indices[start:stop], casting="safe"
        )
    tail_nnz = int(tail.nnz)
    for start in range(0, tail_nnz, chunk):
        _check_deadline(deadline, f"csr_tail_copy:{stage}")
        stop = min(tail_nnz, start + chunk)
        np.copyto(
            data[parent_nnz + start : parent_nnz + stop],
            tail.data[start:stop],
            casting="no",
        )
        np.copyto(
            indices[parent_nnz + start : parent_nnz + stop],
            tail.indices[start:stop],
            casting="safe",
        )
    np.copyto(
        indptr[: parent.shape[0] + 1], parent.indptr, casting="safe"
    )
    if tail.shape[0]:
        appended = parent_nnz + tail.indptr[1:].astype(
            indptr_dtype, copy=False
        )
        np.copyto(indptr[parent.shape[0] + 1 :], appended, casting="safe")
    return _wrap_owned_csr(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=(parent.shape[0] + tail.shape[0], columns),
        stage=stage,
    )


def _copy_csr_shape(
    parent: sp.csr_matrix,
    *,
    shape: Tuple[int, int],
    deadline: float,
    stage: str,
) -> sp.csr_matrix:
    empty_tail = sp.csr_matrix((0, shape[1]), dtype=np.float64)
    return _copy_csr_with_tail(
        parent,
        empty_tail,
        columns=shape[1],
        deadline=deadline,
        stage=stage,
    )


def _reserve_eta_ids(
    parent_col_ids: np.ndarray,
    parent_bcol_ids: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    if type(count) is not int or not 1 <= count <= 16:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "eta_count_out_of_range"
        )
    parent_max = max(
        -1,
        -1 if parent_col_ids.size == 0 else int(np.max(parent_col_ids)),
        -1 if parent_bcol_ids.size == 0 else int(np.max(parent_bcol_ids)),
    )
    try:
        reserved = hz_reserve_fresh_col_ids_above(
            count,
            lower_bound_exclusive=parent_max,
            device="cpu",
        )
        result = reserved.detach().cpu().numpy().astype(
            np.int64, copy=True
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "eta_id_int64_overflow_or_global_reservation_failed"
        ) from exc
    values = tuple(int(value) for value in result.tolist())
    if (
        len(values) != count
        or len(set(values)) != count
        or any(value < 0 or value <= parent_max for value in values)
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "eta_id_reservation_not_exact_or_disjoint"
        )
    return result


def _row_tags(
    candidate: LivePhaseConditionedObjectiveHullCandidate,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    descriptor = candidate.descriptor
    suffix = descriptor.representation_sha256[:16]
    equality = tuple(
        f"{_EQ_TAG_PREFIX}:{index}:{row.name}:{suffix}"
        for index, row in enumerate(descriptor.equality_rows)
    )
    upper = tuple(
        f"{_UB_TAG_PREFIX}:{index}:{row.name}:{suffix}"
        for index, row in enumerate(descriptor.upper_rows)
    )
    try:
        for tag in (*equality, *upper):
            tag.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "pc_objective_hull_row_tag_nonascii"
        ) from exc
    return equality, upper


def _assemble_owned_hz(
    dense: Mapping[str, np.ndarray],
    sparse: Mapping[str, sp.csr_matrix],
) -> SparseHZono:
    if set(dense) != set(_CORE_DENSE_NAMES) or set(sparse) != set(
        _CORE_CSR_NAMES
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "owned_hz_fields_malformed"
        )
    hz = object.__new__(SparseHZono)
    for name in _CORE_DENSE_NAMES:
        object.__setattr__(hz, name, dense[name])
    for name in _CORE_CSR_NAMES:
        object.__setattr__(hz, name, sparse[name])
    return hz


def _audit_output_caps(
    layout: _SourceLayout,
    frame: PCOHBinary64LocalRowFrame,
    *,
    caps: PCOHFreshMaterializationCaps,
    new_tag_bytes: int,
) -> None:
    hz = layout.hz
    q = int(frame.equality_rhs.size)
    u = int(frame.upper_rhs.size)
    m = int(frame.eta_columns)
    tail_nnz = sum(
        int(matrix.nnz)
        for matrix in (
            frame.equality_continuous_eta,
            frame.equality_binary,
            frame.upper_continuous_eta,
            frame.upper_binary,
        )
    )
    # Added dense items: b Q, ub U, col ids M, provenance layer ids M.
    # Added CSR items: data+indices per nnz and two indptr entries per row.
    added_items = 2 * m + q + u + 2 * tail_nnz + 2 * q + 2 * u
    if (
        hz.n_cont + hz.n_bin + m > caps.max_parent_variables
        or hz.n_out + hz.n_eq + hz.n_ub + q + u
        > caps.max_parent_rows
        or layout.nonzeros + tail_nnz > caps.max_parent_nonzeros
        or layout.buffer_items + added_items > caps.max_parent_buffer_items
        or layout.tag_bytes + new_tag_bytes > caps.max_tag_bytes
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "fresh_output_cap_exceeded"
        )


def _freeze_fresh(hz: SparseHZono, input_col_ids: np.ndarray) -> None:
    for name in _CORE_DENSE_NAMES:
        getattr(hz, name).setflags(write=False)
    for name in _CORE_CSR_NAMES:
        matrix = getattr(hz, name)
        for value in (matrix.data, matrix.indices, matrix.indptr):
            value.setflags(write=False)
    for name in _PROVENANCE_NAMES:
        getattr(hz, name).setflags(write=False)
    input_col_ids.setflags(write=False)


def _csr_prefix_view(
    matrix: sp.csr_matrix,
    *,
    rows: int,
    columns: int,
) -> sp.csr_matrix:
    nnz = int(matrix.indptr[rows])
    data = matrix.data[:nnz]
    indices = matrix.indices[:nnz]
    indptr = matrix.indptr[: rows + 1]
    result = sp.csr_matrix(
        (
            data,
            indices,
            indptr,
        ),
        shape=(rows, columns),
        dtype=np.float64,
        copy=False,
    )
    for expected, observed in (
        (data, result.data),
        (indices, result.indices),
        (indptr, result.indptr),
    ):
        if int(expected.size) and not np.shares_memory(expected, observed):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_parent_prefix_view_reallocated"
            )
    return result


def _parent_prefix_hz(
    fresh: SparseHZono,
    *,
    O: int,
    C: int,
    B: int,
    E: int,
    R: int,
) -> SparseHZono:
    dense = {
        "c": fresh.c,
        "b": fresh.b[:E],
        "ub": fresh.ub[:R],
        "col_ids": fresh.col_ids[:C],
        "bcol_ids": fresh.bcol_ids,
    }
    sparse = {
        "Gc": _csr_prefix_view(fresh.Gc, rows=O, columns=C),
        "Gb": _csr_prefix_view(fresh.Gb, rows=O, columns=B),
        "Ac": _csr_prefix_view(fresh.Ac, rows=E, columns=C),
        "Ab": _csr_prefix_view(fresh.Ab, rows=E, columns=B),
        "Auc": _csr_prefix_view(fresh.Auc, rows=R, columns=C),
        "Aub": _csr_prefix_view(fresh.Aub, rows=R, columns=B),
    }
    return _assemble_owned_hz(dense, sparse)


def _assert_detached(
    layout: _SourceLayout,
    fresh_build: OperatorHZBuild,
) -> None:
    source_arrays = tuple(
        value
        for value in _owned_objects(layout.build, provenance=layout.provenance)
        if type(value) is np.ndarray
    )
    fresh_arrays = tuple(
        value for value in _owned_objects(fresh_build) if type(value) is np.ndarray
    )
    if any(
        np.shares_memory(source, fresh)
        for source in source_arrays
        for fresh in fresh_arrays
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "fresh_source_buffer_alias"
        )
    if any(value.flags.writeable for value in fresh_arrays):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "fresh_buffer_writeable"
        )


def _build_detached_fresh(
    layout: _SourceLayout,
    candidate: LivePhaseConditionedObjectiveHullCandidate,
    frame: PCOHBinary64LocalRowFrame,
    *,
    eta_ids: np.ndarray,
    deadline: float,
    caps: PCOHFreshMaterializationCaps,
) -> Tuple[
    OperatorHZBuild,
    str,
    str,
    str,
    Tuple[str, ...],
    Tuple[str, ...],
    str,
]:
    hz = layout.hz
    O, C, B, E, R = hz.n_out, hz.n_cont, hz.n_bin, hz.n_eq, hz.n_ub
    M = int(frame.eta_columns)
    Q = int(frame.equality_rhs.size)
    U = int(frame.upper_rhs.size)
    if (
        M != int(eta_ids.size)
        or frame.parent_continuous_columns != C
        or frame.parent_binary_columns != B
        or frame.equality_continuous_eta.shape != (Q, C + M)
        or frame.equality_binary.shape != (Q, B)
        or frame.upper_continuous_eta.shape != (U, C + M)
        or frame.upper_binary.shape != (U, B)
        or U != 1
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "row_frame_shape_cross_binding_failed"
        )
    equality_tags, upper_tags = _row_tags(candidate)
    if len(equality_tags) != Q or len(upper_tags) != U:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "row_tag_count_mismatch"
        )
    new_tag_bytes = sum(
        len(tag.encode("ascii")) for tag in (*equality_tags, *upper_tags)
    )
    _audit_output_caps(
        layout, frame, caps=caps, new_tag_bytes=new_tag_bytes
    )
    _check_deadline(deadline, "before_final_size_allocations")

    dense = {
        "c": _copy_array(hz.c, deadline=deadline, stage="c"),
        "b": _copy_array(
            hz.b,
            tail=frame.equality_rhs,
            deadline=deadline,
            stage="b_with_pcoh_tail",
        ),
        "ub": _copy_array(
            hz.ub,
            tail=frame.upper_rhs,
            deadline=deadline,
            stage="ub_with_pcoh_tail",
        ),
        "col_ids": _copy_array(
            hz.col_ids,
            tail=eta_ids,
            deadline=deadline,
            stage="col_ids_with_eta_tail",
        ),
        "bcol_ids": _copy_array(
            hz.bcol_ids, deadline=deadline, stage="bcol_ids"
        ),
    }
    sparse = {
        "Gc": _copy_csr_shape(
            hz.Gc,
            shape=(O, C + M),
            deadline=deadline,
            stage="Gc_eta_zero_extension",
        ),
        "Gb": _copy_csr_shape(
            hz.Gb, shape=(O, B), deadline=deadline, stage="Gb"
        ),
        "Ac": _copy_csr_with_tail(
            hz.Ac,
            frame.equality_continuous_eta,
            columns=C + M,
            deadline=deadline,
            stage="Ac_pcoh_equalities",
        ),
        "Ab": _copy_csr_with_tail(
            hz.Ab,
            frame.equality_binary,
            columns=B,
            deadline=deadline,
            stage="Ab_pcoh_equalities",
        ),
        "Auc": _copy_csr_with_tail(
            hz.Auc,
            frame.upper_continuous_eta,
            columns=C + M,
            deadline=deadline,
            stage="Auc_pcoh_upper",
        ),
        "Aub": _copy_csr_with_tail(
            hz.Aub,
            frame.upper_binary,
            columns=B,
            deadline=deadline,
            stage="Aub_pcoh_upper",
        ),
    }
    fresh_hz = _assemble_owned_hz(dense, sparse)

    fresh_input_ids = _copy_array(
        layout.input_col_ids,
        deadline=deadline,
        stage="build_input_col_ids",
    )
    fresh_provenance = {
        "full_col_ids": _copy_array(
            layout.provenance["full_col_ids"],
            deadline=deadline,
            stage="full_col_ids",
        ),
        "operator_input_center": _copy_array(
            layout.provenance["operator_input_center"],
            deadline=deadline,
            stage="operator_input_center",
        ),
        "operator_input_radius": _copy_array(
            layout.provenance["operator_input_radius"],
            deadline=deadline,
            stage="operator_input_radius",
        ),
        "_solver_continuous_column_layer_ids": _copy_array(
            layout.provenance["_solver_continuous_column_layer_ids"],
            tail=np.full(M, -1, dtype=np.int64),
            deadline=deadline,
            stage="continuous_layer_ids_with_eta_tail",
        ),
    }
    for name, value in fresh_provenance.items():
        setattr(fresh_hz, name, value)
    fresh_tags = (
        layout.tags[:E]
        + equality_tags
        + layout.tags[E:]
        + upper_tags
    )
    setattr(fresh_hz, "_solver_constraint_row_tags", fresh_tags)
    setattr(fresh_hz, "_solver_row_constraint_prefix_frames", {})

    # No source nonempty marker, solver cache, conditional state, or arbitrary
    # metadata is copied.  This build deliberately remains candidate-only.
    fresh_build = OperatorHZBuild(
        hz=fresh_hz,
        input_col_ids=fresh_input_ids,
        input_layer_id=layout.build.input_layer_id,
        output_layer_id=layout.build.output_layer_id,
        assert_layer_id=layout.build.assert_layer_id,
        metadata=_deep_freeze(
            {
                "schema": "act.hybridz_pcoh_private_fresh_build.toy.v1",
                "candidate_only": True,
                "proof_authority": False,
                "verdict_authority": False,
                "production_ready": False,
                "descriptor_representation_sha256": (
                    candidate.descriptor.representation_sha256
                ),
            }
        ),
        property_upper_output=False,
        property_upper_row_groups=(),
        verified_preactivation_frame=None,
        constructive_nonempty_seal=None,
        performance_diagnostic=None,
    )
    _freeze_fresh(fresh_hz, fresh_input_ids)

    prefix = _parent_prefix_hz(
        fresh_hz, O=O, C=C, B=B, E=E, R=R
    )
    prefix_digest = sparse_hz_semantic_digest(prefix)
    copied_source_tags = fresh_tags[:E] + fresh_tags[E + Q : E + Q + R]
    prefix_provenance = {
        "full_col_ids": fresh_hz.full_col_ids,
        "operator_input_center": fresh_hz.operator_input_center,
        "operator_input_radius": fresh_hz.operator_input_radius,
        "_solver_continuous_column_layer_ids": (
            fresh_hz._solver_continuous_column_layer_ids[:C]
        ),
    }
    prefix_frame = _frame_digest(
        semantic_digest=prefix_digest,
        input_layer_id=fresh_build.input_layer_id,
        output_layer_id=fresh_build.output_layer_id,
        assert_layer_id=fresh_build.assert_layer_id,
        property_upper_output=False,
        property_upper_row_groups=(),
        input_col_ids=fresh_build.input_col_ids,
        provenance=prefix_provenance,
        tags=copied_source_tags,
        deadline=deadline,
    )
    fresh_digest = sparse_hz_semantic_digest(fresh_hz)
    fresh_frame = _frame_digest(
        semantic_digest=fresh_digest,
        input_layer_id=fresh_build.input_layer_id,
        output_layer_id=fresh_build.output_layer_id,
        assert_layer_id=fresh_build.assert_layer_id,
        property_upper_output=False,
        property_upper_row_groups=(),
        input_col_ids=fresh_build.input_col_ids,
        provenance=fresh_provenance,
        tags=fresh_tags,
        deadline=deadline,
    )
    _assert_detached(layout, fresh_build)
    if set(vars(fresh_hz)) != _FRESH_ATTRIBUTE_WHITELIST:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "fresh_hz_attribute_whitelist_violation"
        )
    _check_deadline(deadline, "fresh_construction_complete")
    return (
        fresh_build,
        prefix_digest,
        prefix_frame,
        fresh_digest,
        equality_tags,
        upper_tags,
        fresh_frame,
    )


def _critical_build_fields(build: OperatorHZBuild) -> Tuple[Any, ...]:
    return (
        build.input_layer_id,
        build.output_layer_id,
        build.assert_layer_id,
        build.property_upper_output,
        build.property_upper_row_groups,
        build.verified_preactivation_frame,
        build.constructive_nonempty_seal,
        build.performance_diagnostic,
    )


def _strict_private_metadata_sha256(build: OperatorHZBuild) -> str:
    metadata = build.metadata
    if (
        type(metadata) is not MappingProxyType
        or set(metadata) != {
            "schema",
            "candidate_only",
            "proof_authority",
            "verdict_authority",
            "production_ready",
            "descriptor_representation_sha256",
        }
        or metadata.get("schema")
        != "act.hybridz_pcoh_private_fresh_build.toy.v1"
        or metadata.get("candidate_only") is not True
        or metadata.get("proof_authority") is not False
        or metadata.get("verdict_authority") is not False
        or metadata.get("production_ready") is not False
        or not _valid_sha256(
            metadata.get("descriptor_representation_sha256")
        )
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "fresh_private_metadata_malformed"
        )
    return _canonical_sha256(metadata)


def _sweep_registry_locked() -> None:
    now = time.monotonic()
    pid = os.getpid()
    stale = tuple(
        token
        for token, record in _REGISTRY.items()
        if (
            record.process_id != pid
            or record.expires_monotonic <= now
            or record.capability_ref() is None
            or record.issuance_ref() is None
        )
    )
    for token in stale:
        _REGISTRY.pop(token, None)
    stale_reservations = tuple(
        nonce
        for nonce, reservation in _REGISTRY_RESERVATIONS.items()
        if (
            reservation.process_id != pid
            or reservation.expires_monotonic <= now
        )
    )
    for nonce in stale_reservations:
        _REGISTRY_RESERVATIONS.pop(nonce, None)


def _reserve_registry_slot(
    *,
    max_entries: int,
    deadline: float,
) -> str:
    """Reserve capacity before any live replay or parent-sized allocation."""

    _check_deadline(deadline, "before_registry_reservation")
    nonce = secrets.token_hex(32)
    with _REGISTRY_LOCK:
        _sweep_registry_locked()
        if len(_REGISTRY) + len(_REGISTRY_RESERVATIONS) >= max_entries:
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_registry_capacity_exceeded_before_live_replay"
            )
        if nonce in _REGISTRY_RESERVATIONS:
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_registry_reservation_nonce_collision"
            )
        _REGISTRY_RESERVATIONS[nonce] = _RegistryReservation(
            process_id=os.getpid(),
            expires_monotonic=deadline,
        )
    try:
        _check_deadline(deadline, "registry_reservation_complete")
    except BaseException:
        with _REGISTRY_LOCK:
            _REGISTRY_RESERVATIONS.pop(nonce, None)
        raise
    return nonce


def _release_registry_reservation(nonce: str) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY_RESERVATIONS.pop(nonce, None)


def _build_materialized_tightness_summary(
    candidate: LivePhaseConditionedObjectiveHullCandidate,
    row_frame: PCOHBinary64LocalRowFrame,
    conditional_certificates: Tuple[Any, ...],
    *,
    deadline: float,
) -> PCOHFreshMaterializedTightnessSummary:
    """Cross-bind replayed conditionals and support the stored PCOH row."""

    _check_deadline(deadline, "tightness_summary_entry")
    descriptor = candidate.descriptor
    patterns = descriptor.patterns
    stable_ids = descriptor.stable_bit_ids
    if (
        type(conditional_certificates) is not tuple
        or len(conditional_certificates) != len(patterns)
        or len(patterns) != 2 ** len(stable_ids)
        or len(row_frame.upper_row_guards) != 1
        or row_frame.upper_continuous_eta.shape[0] != 1
        or row_frame.upper_binary.shape[0] != 1
        or row_frame.upper_rhs.size != 1
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "tightness_summary_source_shape_invalid"
        )
    empty_patterns = {
        tuple(int(phase) for _, phase in evidence.assignments)
        for evidence in descriptor.empty_pattern_evidence
    }
    active_mask = tuple(pattern not in empty_patterns for pattern in patterns)
    if not any(active_mask):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "tightness_summary_all_patterns_certified_empty"
        )
    binding = descriptor.objective_binding
    first = conditional_certificates[0]
    if type(first) is not OperatorPhaseConditionedObjectiveBoundCertificate:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "tightness_summary_conditional_wrong_type"
        )
    global_upper = first.global_cube_upper
    global_exact = first.global_cube_upper_exact
    objective_envelope_sha = first.objective_envelope_sha256
    global_checker_sha = first.global_checker_sha256
    pattern_upper_hex = []
    certificate_sha = []
    pattern_sha = []
    selected_source = []
    for index, (pattern, bound, certificate) in enumerate(
        zip(patterns, descriptor.pattern_bounds, conditional_certificates)
    ):
        if index % 4 == 0:
            _check_deadline(deadline, "tightness_conditional_cross_binding")
        assignments = tuple(zip(stable_ids, pattern))
        if (
            type(certificate)
            is not OperatorPhaseConditionedObjectiveBoundCertificate
            or certificate.schema != _CONDITIONAL_CERTIFICATE_SCHEMA
            or certificate.pattern != pattern
            or certificate.assignments != assignments
            or bound.assignments != assignments
            or certificate.certificate_sha256 != bound.certificate_sha256
            or certificate.certificate_sha256
            != candidate.conditional_certificate_sha256[index]
            or certificate.upper_stored.hex() != bound.upper_stored.hex()
            or bound.upper_exact != Fraction.from_float(certificate.upper_stored)
            or certificate.objective_binding != binding
            or certificate.objective_binding.objective_binding_sha256
            != binding.objective_binding_sha256
            or certificate.objective_envelope_sha256
            != objective_envelope_sha
            or certificate.global_cube_upper.hex() != global_upper.hex()
            or certificate.global_cube_upper_exact != global_exact
            or certificate.global_checker_sha256 != global_checker_sha
            or certificate.receipt.get("checker_route")
            != _CONDITIONAL_CHECKER_ROUTE
            or certificate.receipt.get("upper_stored_hex")
            != certificate.upper_stored.hex()
            or certificate.receipt.get("objective_binding_sha256")
            != binding.objective_binding_sha256
            or certificate.receipt.get("objective_envelope_sha256")
            != objective_envelope_sha
            or certificate.receipt.get("certificate_sha256")
            != certificate.certificate_sha256
            or certificate.receipt.get("baseline_nonregression_checked")
            is not True
            or certificate.proof_authority is not False
        ):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "tightness_summary_conditional_cross_binding_failed"
            )
        pattern_upper_hex.append(certificate.upper_stored.hex())
        certificate_sha.append(certificate.certificate_sha256)
        pattern_sha.append(certificate.pattern_sha256)
        selected_source.append(certificate.selected_source)
    if (
        not math.isfinite(global_upper)
        or Fraction.from_float(global_upper) < global_exact
        or tuple(certificate_sha) != candidate.conditional_certificate_sha256
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "tightness_summary_global_or_cover_invalid"
        )

    guard = row_frame.upper_row_guards[0]
    if guard.row_index != 0 or guard.row_name != "phase_conditioned_objective_upper":
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "tightness_summary_guard_row_invalid"
        )
    selected_mismatch = {stable_id: Fraction(0) for stable_id in stable_ids}
    eta_stored = [Fraction(0) for _ in patterns]
    free_parent_mismatch = Fraction(0)
    all_parent_mismatch = Fraction(0)
    recomputed_guard = Fraction(0)
    for offset, error in enumerate(guard.coefficient_errors):
        if offset % 4096 == 0:
            _check_deadline(deadline, "tightness_coefficient_support")
        stored = Fraction.from_float(error.stored)
        mismatch = error.exact - stored
        if abs(mismatch) != error.absolute_error:
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "tightness_summary_coefficient_error_invalid"
            )
        recomputed_guard += abs(mismatch)
        if error.group == "parent_continuous":
            free_parent_mismatch += abs(mismatch)
            all_parent_mismatch += abs(mismatch)
        elif error.group == "parent_binary":
            all_parent_mismatch += abs(mismatch)
            if error.identifier in selected_mismatch:
                selected_mismatch[error.identifier] = mismatch
            else:
                free_parent_mismatch += abs(mismatch)
        elif error.group == "eta":
            if not 0 <= error.identifier < len(patterns):
                raise PhaseConditionedObjectiveHullFreshMaterializationError(
                    "tightness_summary_eta_identifier_invalid"
                )
            eta_stored[error.identifier] = stored
        else:
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "tightness_summary_coefficient_group_invalid"
            )
    if recomputed_guard != guard.total_coefficient_guard:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "tightness_summary_total_guard_mismatch"
        )

    linked_support = []
    direct_support = []
    for pattern_index, pattern in enumerate(patterns):
        eta_vertex = tuple(
            Fraction(1 if index == pattern_index else -1)
            for index in range(len(patterns))
        )
        eta_support = -sum(
            (coefficient * value for coefficient, value in zip(eta_stored, eta_vertex)),
            Fraction(0),
        )
        selected_support = sum(
            (
                selected_mismatch[stable_id] * phase
                for stable_id, phase in zip(stable_ids, pattern)
            ),
            Fraction(0),
        )
        direct_support.append(eta_support)
        linked_support.append(eta_support + selected_support)
    active_uppers = tuple(
        Fraction.from_float(float.fromhex(value))
        for value, active in zip(pattern_upper_hex, active_mask)
        if active
    )
    active_linked = tuple(
        value
        for value, active in zip(linked_support, active_mask)
        if active
    )
    active_direct = tuple(
        value
        for value, active in zip(direct_support, active_mask)
        if active
    )
    ideal = max(active_uppers)
    center = binding.center
    stored_rhs = Fraction.from_float(guard.stored_rhs)
    linked = center + stored_rhs + free_parent_mismatch + max(active_linked)
    direct = center + stored_rhs + all_parent_mismatch + max(active_direct)
    guarded = (
        ideal
        + (stored_rhs - guard.raw_rhs_exact)
        + guard.total_coefficient_guard
    )
    if not (
        ideal <= Fraction.from_float(global_upper)
        and ideal <= linked <= direct <= guarded
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "tightness_summary_sound_upper_chain_failed"
        )
    linked_hex = _outward_hex(linked, name="materialized_linked_upper")
    direct_hex = _outward_hex(direct, name="materialized_direct_upper")
    guarded_hex = _outward_hex(guarded, name="materialized_guard_upper")
    final = min(global_upper, float.fromhex(linked_hex))
    provisional = PCOHFreshMaterializedTightnessSummary(
        schema=_MATERIALIZED_TIGHTNESS_SCHEMA,
        status="sound_materialized_structural_upper",
        parent_semantic_digest=candidate.parent_semantic_digest,
        adapter_candidate_sha256=candidate.candidate_sha256,
        descriptor_representation_sha256=descriptor.representation_sha256,
        row_frame_sha256=row_frame.frame_sha256,
        stable_bit_ids=stable_ids,
        canonical_patterns=patterns,
        active_pattern_mask=active_mask,
        empty_evidence_descriptor_sha256=tuple(
            evidence.descriptor_sha256
            for evidence in descriptor.empty_pattern_evidence
        ),
        conditional_certificate_schema=_CONDITIONAL_CERTIFICATE_SCHEMA,
        conditional_certificate_sha256=tuple(certificate_sha),
        conditional_pattern_sha256=tuple(pattern_sha),
        conditional_selected_source=tuple(selected_source),
        conditional_checker_route=_CONDITIONAL_CHECKER_ROUTE,
        objective_binding_sha256=binding.objective_binding_sha256,
        objective_envelope_sha256=objective_envelope_sha,
        global_checker_sha256=global_checker_sha,
        global_cube_upper_exact=_fraction_pair(global_exact),
        global_cube_upper_hex=global_upper.hex(),
        pattern_upper_hex=tuple(pattern_upper_hex),
        objective_center_exact=_fraction_pair(center),
        row_raw_rhs_exact=_fraction_pair(guard.raw_rhs_exact),
        row_stored_rhs_hex=guard.stored_rhs.hex(),
        row_total_coefficient_guard_exact=_fraction_pair(
            guard.total_coefficient_guard
        ),
        free_parent_mismatch_exact=_fraction_pair(free_parent_mismatch),
        all_parent_mismatch_exact=_fraction_pair(all_parent_mismatch),
        linked_support_exact=tuple(_fraction_pair(value) for value in linked_support),
        direct_eta_support_exact=tuple(_fraction_pair(value) for value in direct_support),
        ideal_union_upper_hex=float(ideal).hex(),
        materialized_linked_upper_exact=_fraction_pair(linked),
        materialized_linked_upper_hex=linked_hex,
        materialized_direct_upper_exact=_fraction_pair(direct),
        materialized_direct_upper_hex=direct_hex,
        materialized_guard_upper_exact=_fraction_pair(guarded),
        materialized_guard_upper_hex=guarded_hex,
        rounding_tax_exact=_fraction_pair(linked - ideal),
        final_structural_upper_hex=final.hex(),
        diagnostic_only=True,
        full_parent_lp_called=False,
        proof_authority=False,
        verdict_authority=False,
        summary_sha256="",
    )
    summary = PCOHFreshMaterializedTightnessSummary(
        **{
            **provisional.__dict__,
            "summary_sha256": _canonical_sha256(
                _materialized_tightness_payload(
                    provisional, include_digest=False
                )
            ),
        }
    )
    _strict_replay_materialized_tightness_summary(summary)
    _check_deadline(deadline, "tightness_summary_complete")
    return summary


def _issuance_payload(issuance: PCOHFreshBuildIssuance) -> Dict[str, Any]:
    capability = issuance.capability
    return {
        "schema": issuance.schema,
        "parent_semantic_digest": issuance.parent_semantic_digest,
        "terminal_parent_semantic_digest": (
            issuance.terminal_parent_semantic_digest
        ),
        "source_frame_sha256": issuance.source_frame_sha256,
        "terminal_source_frame_sha256": (
            issuance.terminal_source_frame_sha256
        ),
        "fresh_parent_prefix_semantic_digest": (
            issuance.fresh_parent_prefix_semantic_digest
        ),
        "fresh_parent_prefix_frame_sha256": (
            issuance.fresh_parent_prefix_frame_sha256
        ),
        "fresh_semantic_digest": issuance.fresh_semantic_digest,
        "fresh_frame_sha256": issuance.fresh_frame_sha256,
        "adapter_candidate_sha256": issuance.adapter_candidate_sha256,
        "descriptor_representation_sha256": (
            issuance.descriptor_representation_sha256
        ),
        "row_frame_sha256": issuance.row_frame_sha256,
        "eta_col_ids": issuance.eta_col_ids,
        "equality_row_tags": issuance.equality_row_tags,
        "upper_row_tags": issuance.upper_row_tags,
        "materialized_tightness_summary": _materialized_tightness_payload(
            issuance.materialized_tightness_summary,
            include_digest=True,
        ),
        "receipt": issuance.receipt,
        "capability": {
            "schema": capability.schema,
            "token": capability.token,
            "process_id": capability.process_id,
            "expires_monotonic": capability.expires_monotonic,
            "parent_semantic_digest": capability.parent_semantic_digest,
            "fresh_semantic_digest": capability.fresh_semantic_digest,
            "fresh_frame_sha256": capability.fresh_frame_sha256,
            "descriptor_representation_sha256": (
                capability.descriptor_representation_sha256
            ),
            "proof_authority": capability.proof_authority,
            "verdict_authority": capability.verdict_authority,
        },
        "proof_authority": issuance.proof_authority,
        "verdict_authority": issuance.verdict_authority,
    }


def verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness(
    issuance: Any,
) -> bool:
    """Strictly reverify one live, unconsumed issuance and its summary.

    This is an identity-bound process-local diagnostic verifier.  It never
    returns or inspects the private fresh build, never solves an LP, and never
    grants proof or verdict authority.  Verification intentionally becomes
    false after consume/discard/expiry.
    """

    try:
        if (
            type(issuance) is not PCOHFreshBuildIssuance
            or type(issuance.capability) is not PCOHFreshBuildCapability
        ):
            return False
        # Sweep and authenticate the exact weak owner before parsing any
        # attacker-controlled replacement fields.  If a wrapper discarded
        # the genuine issuance and returned ``replace(issuance, ...)``, this
        # pass removes the now-dead registry record instead of stranding it.
        with _REGISTRY_LOCK:
            _sweep_registry_locked()
            entry_record = _REGISTRY.get(issuance.capability.token)
            if (
                entry_record is None
                or entry_record.process_id != os.getpid()
                or entry_record.capability_ref() is not issuance.capability
                or entry_record.issuance_ref() is not issuance
                or entry_record.issuance_sha256 != issuance.issuance_sha256
            ):
                return False
        if (
            issuance.schema != _ISSUANCE_SCHEMA
            or issuance.proof_authority is not False
            or issuance.verdict_authority is not False
            or type(issuance.receipt) is not MappingProxyType
            or issuance.receipt.get("schema") != _RECEIPT_SCHEMA
            or issuance.capability._producer_capability
            is not _PRODUCER_CAPABILITY
            or issuance.capability.process_id != os.getpid()
            or issuance.capability.proof_authority is not False
            or issuance.capability.verdict_authority is not False
        ):
            return False
        summary = issuance.materialized_tightness_summary
        _strict_replay_materialized_tightness_summary(summary)
        expected_summary_payload = _materialized_tightness_payload(
            summary, include_digest=True
        )
        if (
            summary.parent_semantic_digest
            != issuance.parent_semantic_digest
            or summary.adapter_candidate_sha256
            != issuance.adapter_candidate_sha256
            or summary.descriptor_representation_sha256
            != issuance.descriptor_representation_sha256
            or summary.row_frame_sha256 != issuance.row_frame_sha256
            or issuance.receipt.get("materialized_tightness_summary_schema")
            != summary.schema
            or issuance.receipt.get("materialized_tightness_summary_sha256")
            != summary.summary_sha256
            or issuance.receipt.get(
                "materialized_tightness_full_parent_lp_called"
            )
            is not False
            or _canonical_form(
                issuance.receipt.get("materialized_tightness_summary")
            )
            != _canonical_form(expected_summary_payload)
        ):
            return False
        receipt_payload = dict(issuance.receipt)
        receipt_sha = receipt_payload.pop("receipt_sha256", None)
        if (
            not _valid_sha256(receipt_sha)
            or _canonical_sha256(receipt_payload) != receipt_sha
            or not _valid_sha256(issuance.issuance_sha256)
            or _canonical_sha256(_issuance_payload(issuance))
            != issuance.issuance_sha256
        ):
            return False
        with _REGISTRY_LOCK:
            _sweep_registry_locked()
            record = _REGISTRY.get(issuance.capability.token)
            return bool(
                record is not None
                and record.process_id == os.getpid()
                and record.expires_monotonic > time.monotonic()
                and record.capability_ref() is issuance.capability
                and record.issuance_ref() is issuance
                and record.issuance_sha256 == issuance.issuance_sha256
                and issuance.capability.expires_monotonic
                == record.expires_monotonic
            )
    except Exception:
        return False


def _issue_live_phase_conditioned_objective_hull_fresh_build_reserved(
    build: OperatorHZBuild,
    rivals: Sequence[Any],
    selection: Any,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    conditional_certificates: Tuple[Any, ...],
    pair_bundle: Any,
    deadline: float,
    reservation_nonce: str,
    caps: PCOHFreshMaterializationCaps = PCOHFreshMaterializationCaps(),
) -> PCOHFreshBuildIssuance:
    """Live replay raw evidence and issue one opaque fresh-build capability."""

    deadline_value = _deadline(deadline)
    normalized_caps = _normalize_caps(caps)
    _check_deadline(deadline_value, "issue_entry")
    layout = _validate_source(
        build, caps=normalized_caps, deadline=deadline_value
    )
    source_semantic = sparse_hz_semantic_digest(layout.hz)
    source_frame = _source_frame_digest(
        layout, semantic_digest=source_semantic, deadline=deadline_value
    )
    _check_deadline(deadline_value, "before_live_adapter")
    try:
        candidate = build_live_phase_conditioned_objective_hull_candidate(
            build,
            rivals,
            selection,
            focused_rival_id=focused_rival_id,
            stable_bit_ids=stable_bit_ids,
            conditional_certificates=conditional_certificates,
            pair_bundle=pair_bundle,
            deadline=deadline_value,
        )
    except Exception as exc:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "live_adapter_replay_failed"
        ) from exc
    if (
        type(candidate) is not LivePhaseConditionedObjectiveHullCandidate
        or candidate.proof_authority is not False
        or candidate.verdict_authority is not False
        or candidate.parent_semantic_digest != source_semantic
        or candidate.terminal_parent_semantic_digest != source_semantic
        or candidate.stable_bit_ids != tuple(sorted(stable_bit_ids))
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "live_adapter_cross_binding_failed"
        )
    _check_deadline(deadline_value, "before_row_materializer")
    try:
        row_frame = materialize_phase_conditioned_objective_hull_row_frame(
            candidate.descriptor,
            live_parent_semantic_digest=source_semantic,
            parent_col_ids=layout.hz.col_ids,
            parent_bcol_ids=layout.hz.bcol_ids,
            deadline=deadline_value,
            caps=normalized_caps.row_caps,
        )
    except Exception as exc:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "guarded_row_materialization_failed"
        ) from exc
    if (
        type(row_frame) is not PCOHBinary64LocalRowFrame
        or row_frame.proof_authority is not False
        or row_frame.verdict_authority is not False
        or row_frame.parent_semantic_digest != source_semantic
        or row_frame.descriptor_representation_sha256
        != candidate.descriptor.representation_sha256
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "guarded_row_frame_cross_binding_failed"
        )
    if not verify_phase_conditioned_objective_hull_row_frame(
        row_frame,
        candidate.descriptor,
        live_parent_semantic_digest=source_semantic,
        parent_col_ids=layout.hz.col_ids,
        parent_bcol_ids=layout.hz.bcol_ids,
        deadline=deadline_value,
        caps=normalized_caps.row_caps,
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "guarded_row_frame_strict_replay_failed"
        )
    materialized_tightness = _build_materialized_tightness_summary(
        candidate,
        row_frame,
        conditional_certificates,
        deadline=deadline_value,
    )
    eta_ids = _reserve_eta_ids(
        layout.hz.col_ids,
        layout.hz.bcol_ids,
        count=len(candidate.descriptor.eta_columns),
    )
    (
        fresh_build,
        prefix_digest,
        prefix_frame,
        fresh_digest,
        equality_tags,
        upper_tags,
        fresh_frame,
    ) = _build_detached_fresh(
        layout,
        candidate,
        row_frame,
        eta_ids=eta_ids,
        deadline=deadline_value,
        caps=normalized_caps,
    )
    if prefix_digest != source_semantic or prefix_frame != source_frame:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "fresh_parent_prefix_seal_mismatch"
        )

    # Third seal: the adapter has sealed the source once, the copied prefix
    # has just been sealed independently, and now the live source is re-read
    # after every fresh byte and receipt-relevant hash is complete.
    _check_deadline(deadline_value, "before_outer_terminal_source_seal")
    if (
        build is not layout.build
        or build.hz is not layout.hz
        or build.property_upper_output is not False
        or build.property_upper_row_groups != ()
        or build.verified_preactivation_frame is not None
        or type(
            getattr(build.hz, "_solver_row_constraint_prefix_frames", None)
        )
        is not dict
        or bool(build.hz._solver_row_constraint_prefix_frames)
        or tuple(id(value) for value in _owned_objects(build))
        != layout.source_object_identity
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "outer_terminal_source_identity_changed"
        )
    terminal_semantic = sparse_hz_semantic_digest(build.hz)
    terminal_frame = _source_frame_digest(
        layout,
        semantic_digest=terminal_semantic,
        deadline=deadline_value,
    )
    if terminal_semantic != source_semantic or terminal_frame != source_frame:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "outer_terminal_source_seal_mismatch"
        )
    _check_deadline(deadline_value, "after_outer_terminal_source_seal")

    fresh_payload = _payload_bytes(fresh_build)
    receipt: Dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "status": "toy_fresh_build_registered",
        "production_ready": False,
        "production_blockers": (
            "fresh_producer_constructive_nonempty_seal_not_issued",
            "not_integrated_with_verifier_pipeline",
        ),
        "eta_id_allocator_route": (
            "solver_hz.hz_reserve_fresh_col_ids_above"
        ),
        "eta_id_allocator_global_lock_shared": True,
        "eta_id_reservation_non_reusable": True,
        "proof_authority": False,
        "verdict_authority": False,
        "candidate_only": True,
        "raw_certificates_replayed_inside_issue": True,
        "caller_descriptor_accepted": False,
        "external_proof_booleans_used_as_authority": False,
        "guarded_binary64_row_frame_used": True,
        "direct_live_to_final_detached_copy": True,
        "parent_snapshot_count": 0,
        "final_core_allocation_count": 1,
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "uses_full_concatenate": False,
        "source_buffers_borrowed_by_fresh": False,
        "fresh_buffers_readonly": True,
        "three_seals": (
            "live_adapter_terminal_parent",
            "fresh_parent_prefix_semantic_and_frame",
            "outer_live_parent_semantic_and_frame",
        ),
        "shared_absolute_deadline_fail_closed": True,
        "deadline_enforcement": "cooperative_bulk_scans",
        "hard_wall_deadline_guaranteed": False,
        "caps": _caps_payload(normalized_caps),
        "caps_sha256": _canonical_sha256(_caps_payload(normalized_caps)),
        "parent_semantic_digest": source_semantic,
        "source_frame_sha256": source_frame,
        "terminal_parent_semantic_digest": terminal_semantic,
        "terminal_source_frame_sha256": terminal_frame,
        "fresh_parent_prefix_semantic_digest": prefix_digest,
        "fresh_parent_prefix_frame_sha256": prefix_frame,
        "fresh_semantic_digest": fresh_digest,
        "fresh_frame_sha256": fresh_frame,
        "adapter_candidate_sha256": candidate.candidate_sha256,
        "descriptor_representation_sha256": (
            candidate.descriptor.representation_sha256
        ),
        "row_frame_sha256": row_frame.frame_sha256,
        "materialized_tightness_summary": (
            _materialized_tightness_payload(
                materialized_tightness, include_digest=True
            )
        ),
        "materialized_tightness_summary_sha256": (
            materialized_tightness.summary_sha256
        ),
        "materialized_tightness_summary_schema": (
            materialized_tightness.schema
        ),
        "materialized_tightness_full_parent_lp_called": False,
        "row_frame_guard_sha256": tuple(
            _canonical_sha256(
                {
                    "row_index": guard.row_index,
                    "row_name": guard.row_name,
                    "total_coefficient_guard": (
                        guard.total_coefficient_guard.numerator,
                        guard.total_coefficient_guard.denominator,
                    ),
                    "stored_rhs": guard.stored_rhs,
                }
            )
            for guard in row_frame.upper_row_guards
        ),
        "eta_col_ids": tuple(int(value) for value in eta_ids.tolist()),
        "source_dimensions": (
            layout.hz.n_out,
            layout.hz.n_cont,
            layout.hz.n_bin,
            layout.hz.n_eq,
            layout.hz.n_ub,
        ),
        "fresh_dimensions": (
            fresh_build.hz.n_out,
            fresh_build.hz.n_cont,
            fresh_build.hz.n_bin,
            fresh_build.hz.n_eq,
            fresh_build.hz.n_ub,
        ),
        "source_payload_bytes": layout.payload_bytes,
        "fresh_payload_bytes": fresh_payload,
        "fresh_payload_delta_bytes": fresh_payload - layout.payload_bytes,
        "dropped_source_attributes": layout.dropped_attribute_names,
        "copied_provenance_attributes": _PROVENANCE_NAMES,
        "constructive_nonempty_inherited": False,
        "constructive_nonempty_reissued": False,
        "conditional_metadata_copied": False,
        "row_tag_layout": "old_eq+pcoh_eq+old_ub+pcoh_ub",
        "equality_row_tags": equality_tags,
        "upper_row_tags": upper_tags,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    frozen_receipt = _deep_freeze(receipt)
    now = time.monotonic()
    expires = min(
        deadline_value, now + normalized_caps.capability_ttl_seconds
    )
    if expires <= now:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "no_capability_lifetime_remaining"
        )
    capability = PCOHFreshBuildCapability(
        schema=_CAPABILITY_SCHEMA,
        token=secrets.token_hex(32),
        process_id=os.getpid(),
        expires_monotonic=expires,
        parent_semantic_digest=source_semantic,
        fresh_semantic_digest=fresh_digest,
        fresh_frame_sha256=fresh_frame,
        descriptor_representation_sha256=(
            candidate.descriptor.representation_sha256
        ),
        proof_authority=False,
        verdict_authority=False,
        _producer_capability=_PRODUCER_CAPABILITY,
    )
    placeholder = PCOHFreshBuildIssuance(
        schema=_ISSUANCE_SCHEMA,
        parent_semantic_digest=source_semantic,
        terminal_parent_semantic_digest=terminal_semantic,
        source_frame_sha256=source_frame,
        terminal_source_frame_sha256=terminal_frame,
        fresh_parent_prefix_semantic_digest=prefix_digest,
        fresh_parent_prefix_frame_sha256=prefix_frame,
        fresh_semantic_digest=fresh_digest,
        fresh_frame_sha256=fresh_frame,
        adapter_candidate_sha256=candidate.candidate_sha256,
        descriptor_representation_sha256=(
            candidate.descriptor.representation_sha256
        ),
        row_frame_sha256=row_frame.frame_sha256,
        eta_col_ids=tuple(int(value) for value in eta_ids.tolist()),
        equality_row_tags=equality_tags,
        upper_row_tags=upper_tags,
        materialized_tightness_summary=materialized_tightness,
        receipt=frozen_receipt,
        capability=capability,
        issuance_sha256="",
        proof_authority=False,
        verdict_authority=False,
    )
    issuance = PCOHFreshBuildIssuance(
        **{
            **placeholder.__dict__,
            "issuance_sha256": _canonical_sha256(
                _issuance_payload(placeholder)
            ),
        }
    )
    if set(vars(fresh_build)) != _BUILD_FIELD_WHITELIST:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "fresh_private_build_field_whitelist_violation"
        )
    record = _RegistryRecord(
        capability_ref=weakref.ref(capability),
        issuance_ref=weakref.ref(issuance),
        private_build=fresh_build,
        owned_identity=tuple(id(value) for value in _owned_objects(fresh_build)),
        process_id=os.getpid(),
        expires_monotonic=expires,
        fresh_semantic_digest=fresh_digest,
        fresh_frame_sha256=fresh_frame,
        issuance_sha256=issuance.issuance_sha256,
        metadata_identity=id(fresh_build.metadata),
        metadata_sha256=_strict_private_metadata_sha256(fresh_build),
        build_field_names=tuple(sorted(vars(fresh_build))),
        critical_build_fields=_critical_build_fields(fresh_build),
    )
    _check_deadline(deadline_value, "before_registry_issue")
    with _REGISTRY_LOCK:
        _sweep_registry_locked()
        reservation = _REGISTRY_RESERVATIONS.pop(
            reservation_nonce, None
        )
        if (
            reservation is None
            or reservation.process_id != os.getpid()
            or reservation.expires_monotonic != deadline_value
            or reservation.expires_monotonic <= time.monotonic()
        ):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_registry_reservation_missing_or_expired"
            )
        if capability.token in _REGISTRY:
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_capability_token_collision"
            )
        _REGISTRY[capability.token] = record
    try:
        _check_deadline(deadline_value, "registry_issue_complete")
    except BaseException as exc:
        with _REGISTRY_LOCK:
            _REGISTRY.pop(capability.token, None)
        # The record has already taken ownership of ``fresh_build`` at this
        # point.  A KeyboardInterrupt/SystemExit must therefore follow the
        # same cleanup path as an ordinary exception: otherwise the module
        # registry retains the private HZ even though no issuance reached the
        # caller.  Drop every local private owner before propagating a safe
        # exception and erase the original nested traceback so it cannot be
        # used to recover the record or build from frame locals.
        record = None
        fresh_build = None
        interrupted = not isinstance(exc, Exception)
        interrupted_name = type(exc).__name__
        if interrupted:
            _clear_private_validation_traceback(exc)
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_registry_issue_interrupted:" + interrupted_name
            ) from None
        raise
    return issuance


def issue_live_phase_conditioned_objective_hull_fresh_build(
    build: OperatorHZBuild,
    rivals: Sequence[Any],
    selection: Any,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    conditional_certificates: Tuple[Any, ...],
    pair_bundle: Any,
    deadline: float,
    caps: PCOHFreshMaterializationCaps = PCOHFreshMaterializationCaps(),
) -> PCOHFreshBuildIssuance:
    """Reserve capacity, then perform the one-shot live materialization."""

    deadline_value = _deadline(deadline)
    normalized_caps = _normalize_caps(caps)
    _check_deadline(deadline_value, "issue_entry_before_reservation")
    reservation_nonce = _reserve_registry_slot(
        max_entries=normalized_caps.max_registry_entries,
        deadline=deadline_value,
    )
    try:
        return _issue_live_phase_conditioned_objective_hull_fresh_build_reserved(
            build,
            rivals,
            selection,
            focused_rival_id=focused_rival_id,
            stable_bit_ids=stable_bit_ids,
            conditional_certificates=conditional_certificates,
            pair_bundle=pair_bundle,
            deadline=deadline_value,
            caps=normalized_caps,
            reservation_nonce=reservation_nonce,
        )
    finally:
        _release_registry_reservation(reservation_nonce)


def _take_owned_registry_record(
    issuance: PCOHFreshBuildIssuance,
    capability: PCOHFreshBuildCapability,
) -> Tuple[Optional[_RegistryRecord], Optional[str]]:
    """Validate exact live owners before atomically popping their record."""

    candidate = None
    error = None
    try:
        with _REGISTRY_LOCK:
            _sweep_registry_locked()
            candidate = _REGISTRY.get(capability.token)
            if candidate is None:
                error = "fresh_capability_missing_consumed_or_expired"
            else:
                now = time.monotonic()
                owner_valid = bool(
                    candidate.process_id == os.getpid()
                    and capability.process_id == os.getpid()
                    and candidate.expires_monotonic > now
                    and capability.expires_monotonic
                    == candidate.expires_monotonic
                    and candidate.capability_ref() is capability
                    and candidate.issuance_ref() is issuance
                    and issuance.capability is capability
                    and capability._producer_capability
                    is _PRODUCER_CAPABILITY
                    and capability.proof_authority is False
                    and capability.verdict_authority is False
                    and issuance.proof_authority is False
                    and issuance.verdict_authority is False
                    and issuance.issuance_sha256
                    == candidate.issuance_sha256
                    and _canonical_sha256(_issuance_payload(issuance))
                    == issuance.issuance_sha256
                )
                if owner_valid:
                    candidate = _REGISTRY.pop(capability.token)
                else:
                    candidate = None
                    error = "fresh_capability_owner_or_receipt_mismatch"
    except Exception:
        candidate = None
        error = "fresh_capability_owner_or_receipt_mismatch"
    return candidate, error


def _clear_private_validation_traceback(exc: BaseException) -> None:
    current_code = _clear_private_validation_traceback.__code__
    cursor = exc.__traceback__
    while cursor is not None:
        frame = cursor.tb_frame
        cursor = cursor.tb_next
        if frame.f_code is current_code:
            continue
        try:
            frame.clear()
        except RuntimeError:
            pass
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None


def _validate_taken_registry_record(
    record: _RegistryRecord,
    issuance: PCOHFreshBuildIssuance,
    capability: PCOHFreshBuildCapability,
    *,
    deadline: float,
) -> Tuple[Optional[OperatorHZBuild], Optional[str]]:
    """Validate a popped private build without ever propagating its frame."""

    private_build = None
    result = None
    error = None
    try:
        private_build = record.private_build
        if (
            set(vars(private_build)) != _BUILD_FIELD_WHITELIST
            or tuple(sorted(vars(private_build))) != record.build_field_names
            or _critical_build_fields(private_build)
            != record.critical_build_fields
            or id(private_build.metadata) != record.metadata_identity
            or private_build.property_upper_output is not False
            or private_build.property_upper_row_groups != ()
            or private_build.verified_preactivation_frame is not None
            or private_build.constructive_nonempty_seal is not None
            or private_build.performance_diagnostic is not None
            or set(vars(private_build.hz)) != _FRESH_ATTRIBUTE_WHITELIST
            or type(
                getattr(
                    private_build.hz,
                    "_solver_row_constraint_prefix_frames",
                    None,
                )
            )
            is not dict
            or bool(private_build.hz._solver_row_constraint_prefix_frames)
        ):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_private_mode_or_attribute_whitelist_changed"
            )
        if (
            _strict_private_metadata_sha256(private_build)
            != record.metadata_sha256
        ):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_private_metadata_digest_changed"
            )
        if tuple(id(value) for value in _owned_objects(private_build)) != (
            record.owned_identity
        ):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_private_owner_identity_changed"
            )
        fresh_digest = sparse_hz_semantic_digest(private_build.hz)
        tags = private_build.hz._solver_constraint_row_tags
        provenance = {
            name: getattr(private_build.hz, name)
            for name in _PROVENANCE_NAMES
        }
        fresh_frame = _frame_digest(
            semantic_digest=fresh_digest,
            input_layer_id=private_build.input_layer_id,
            output_layer_id=private_build.output_layer_id,
            assert_layer_id=private_build.assert_layer_id,
            property_upper_output=False,
            property_upper_row_groups=(),
            input_col_ids=private_build.input_col_ids,
            provenance=provenance,
            tags=tags,
            deadline=deadline,
        )
        if (
            fresh_digest != record.fresh_semantic_digest
            or fresh_digest != issuance.fresh_semantic_digest
            or fresh_digest != capability.fresh_semantic_digest
            or fresh_frame != record.fresh_frame_sha256
            or fresh_frame != issuance.fresh_frame_sha256
            or fresh_frame != capability.fresh_frame_sha256
        ):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_private_terminal_digest_mismatch"
            )
        arrays = tuple(
            value
            for value in _owned_objects(private_build)
            if type(value) is np.ndarray
        )
        if any(value.flags.writeable for value in arrays):
            raise PhaseConditionedObjectiveHullFreshMaterializationError(
                "fresh_private_terminal_buffer_writeable"
            )
        _check_deadline(deadline, "consume_complete")
        result = private_build
    except BaseException as exc:
        error = str(exc)[:300]
        _clear_private_validation_traceback(exc)
    finally:
        record = None
        private_build = None
    return result, error


def consume_live_phase_conditioned_objective_hull_fresh_build(
    issuance: PCOHFreshBuildIssuance,
    capability: PCOHFreshBuildCapability,
    *,
    deadline: float,
) -> OperatorHZBuild:
    """Atomically consume the unique private fresh build exactly once."""

    deadline_value = _deadline(deadline)
    _check_deadline(deadline_value, "consume_entry")
    if (
        type(issuance) is not PCOHFreshBuildIssuance
        or type(capability) is not PCOHFreshBuildCapability
        or issuance.capability is not capability
    ):
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            "consume_input_wrong_type_or_cross_binding"
        )
    record, take_error = _take_owned_registry_record(issuance, capability)
    if take_error is not None or record is None:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            str(take_error)
        ) from None
    try:
        private_build, validation_error = _validate_taken_registry_record(
            record,
            issuance,
            capability,
            deadline=deadline_value,
        )
    except BaseException as exc:
        private_build = None
        validation_error = f"private_validation_interrupted:{type(exc).__name__}"
        _clear_private_validation_traceback(exc)
    record = None
    if validation_error is not None or private_build is None:
        raise PhaseConditionedObjectiveHullFreshMaterializationError(
            str(validation_error)
        ) from None
    return private_build


def discard_live_phase_conditioned_objective_hull_fresh_build(
    issuance: PCOHFreshBuildIssuance,
    capability: PCOHFreshBuildCapability,
) -> bool:
    """Drop an unconsumed private build without returning any solver object.

    This cleanup-only operation intentionally has no deadline: its sole use is
    fail-closed exception handling after an issuance succeeded but the normal
    deadline-bearing consume path could not begin.  The opaque token and exact
    owner identities are still required.  A mismatched call may remove only
    the caller-presented secret token and never exposes the registered build.
    """

    if (
        type(issuance) is not PCOHFreshBuildIssuance
        or type(capability) is not PCOHFreshBuildCapability
        or issuance.capability is not capability
        or capability.process_id != os.getpid()
        or capability._producer_capability is not _PRODUCER_CAPABILITY
        or capability.proof_authority is not False
        or capability.verdict_authority is not False
    ):
        return False
    with _REGISTRY_LOCK:
        record = _REGISTRY.get(capability.token)
        if (
            record is None
            or record.process_id != os.getpid()
            or record.capability_ref() is not capability
            or record.issuance_ref() is not issuance
            or record.issuance_sha256 != issuance.issuance_sha256
        ):
            return False
        _REGISTRY.pop(capability.token)
    return True


__all__ = [
    "PCOHFreshBuildCapability",
    "PCOHFreshBuildIssuance",
    "PCOHFreshMaterializedTightnessSummary",
    "PCOHFreshMaterializationCaps",
    "PhaseConditionedObjectiveHullFreshMaterializationError",
    "consume_live_phase_conditioned_objective_hull_fresh_build",
    "discard_live_phase_conditioned_objective_hull_fresh_build",
    "issue_live_phase_conditioned_objective_hull_fresh_build",
    "verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness",
]
