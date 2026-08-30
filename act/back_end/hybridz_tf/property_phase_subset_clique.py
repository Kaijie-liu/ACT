#!/usr/bin/env python3
"""Exact subset-clique closure for property-selected phase literals.

The signed-support grouping stage is only a proposal.  This module converts
each eligible group into freshly bound ``PhaseLiteral`` objects, checks every
one of its ``k(k-1)/2`` pairs against the complete parent HybridZ relaxation,
and appends a clique row only when every pair has an exact Fraction dual-ray
certificate.  Omitted zero-effect binaries are never assumed fixed or
checked, and never appear in the emitted row.

This remains an isolated candidate with ``proof_authority=False``.  It is not
connected to verifier or BaB verdict paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    ExactDualRayConflictCertificate,
    ExactSourceTermV2,
    PersistentPairRecord,
    _PersistentHighsPairLP,
    _ordered_source_frame_digest,
    _verify_exact_certificate_with_source_frame,
    exact_certificate_from_highs_dual_ray_candidate,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
    _copy_parent_with_clique_cut,
    _literal_binding_digest,
    _ordered_pair,
)
from act.back_end.hybridz_tf.property_phase_literal_groups import (
    PropertyLiteralGroup,
    PropertyLiteralGroupingResult,
    verify_property_literal_grouping_result,
)
from act.back_end.solver.solver_hz import SparseHZono


class PropertyPhaseSubsetCliqueError(ValueError):
    """Malformed, incomplete, or expired subset candidate invocation."""


@dataclass(frozen=True)
class SubsetCliqueClosure:
    """One group-bound complete or incomplete exact pair closure."""

    group_id: str
    subset_property_digest: str
    literals: Tuple[PhaseLiteral, ...]
    omitted_zero_bcol_ids: Tuple[int, ...]
    pair_records: Tuple[PersistentPairRecord, ...]
    certificates: Tuple[
        ExactDualRayConflictCertificate, ...
    ]
    complete: bool
    cut_applied: bool
    proof_authority: bool = False


@dataclass(frozen=True)
class PropertyPhaseSubsetCliqueResult:
    """Exact subset cuts plus non-authoritative runtime telemetry."""

    status: str
    hz: Optional[SparseHZono]
    parent_semantic_digest: str
    ordered_property_digest: str
    grouping_receipt_sha256: str
    closures: Tuple[SubsetCliqueClosure, ...]
    telemetry: Mapping[str, Any]
    proof_authority: bool = False


_DEFAULT_MAX_CUT_GROUPS = 16
_DEFAULT_MAX_LITERALS_PER_GROUP = 64
_DEFAULT_MAX_TOTAL_PAIRS = 512
_DEFAULT_MAX_SOURCE_TERMS = 128
_DEFAULT_MAX_MULTIPLIER_BITS = 256
_DEFAULT_MAX_EXACT_BITS = 4096
_DEFAULT_MAX_EXACT_NONZEROS = 200000
_DEFAULT_MAX_PARENT_VARIABLES = 2_000_000
_DEFAULT_MAX_PARENT_ROWS = 2_000_000
_DEFAULT_MAX_PARENT_NONZEROS = 50_000_000
_DEFAULT_MAX_PARENT_BUFFER_ITEMS = 120_000_000

_HARD_LIMITS = {
    "max_cut_groups": 64,
    "max_literals_per_group": 64,
    "max_total_pairs": 2016,
    "max_source_terms": 256,
    "max_multiplier_bits": 2048,
    "max_exact_bits": 16384,
    "max_exact_nonzeros": 1000000,
    "max_parent_variables": 10_000_000,
    "max_parent_rows": 10_000_000,
    "max_parent_nonzeros": 500_000_000,
    "max_parent_buffer_items": 1_200_000_000,
}


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
    return (
        type(value) is str
        and len(value) == 64
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


def _exact_literal_payload(
    literal: Any,
) -> Optional[Tuple[int, int, str]]:
    """Return a comparison-safe literal payload or reject subclasses."""

    if (
        type(literal) is not PhaseLiteral
        or type(literal.stable_bcol_id) is not int
        or literal.stable_bcol_id < 0
        or type(literal.phase) is not int
        or literal.phase not in {-1, 1}
        or not _valid_sha256(literal.binding_digest)
    ):
        return None
    return (
        literal.stable_bcol_id,
        literal.phase,
        literal.binding_digest,
    )


def _exact_literal_tuple_payload(
    literals: Any,
) -> Optional[Tuple[Tuple[int, int, str], ...]]:
    if type(literals) is not tuple:
        return None
    payload = tuple(
        _exact_literal_payload(literal) for literal in literals
    )
    if any(item is None for item in payload):
        return None
    return payload  # type: ignore[return-value]


def _exact_certificate_shape(
    certificate: Any,
) -> bool:
    """Reject equality gadgets before exact arithmetic replay."""

    if (
        type(certificate)
        is not ExactDualRayConflictCertificate
        or certificate.proof_authority is not False
        or _exact_literal_tuple_payload(
            certificate.literals
        )
        is None
        or not _valid_sha256(
            certificate.parent_semantic_digest
        )
        or not _valid_sha256(certificate.property_digest)
        or not _valid_sha256(
            certificate.ordered_source_frame_sha256
        )
        or not _valid_sha256(certificate.certificate_sha256)
        or type(certificate.source_terms) is not tuple
        or not certificate.source_terms
        or type(certificate.contradiction_numerator) is not int
        or type(certificate.contradiction_denominator) is not int
        or certificate.contradiction_denominator <= 0
        or type(certificate.rationalization) is not str
        or type(certificate.arithmetic) is not str
    ):
        return False
    return all(
        type(term) is ExactSourceTermV2
        and type(term.global_row_index) is int
        and type(term.kind) is str
        and type(term.local_row_index) is int
        and type(term.numerator) is int
        and type(term.denominator) is int
        and _valid_sha256(term.source_row_sha256)
        for term in certificate.source_terms
    )


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise PropertyPhaseSubsetCliqueError(
            f"{name}_not_integer"
        )
    return int(value)


def _caps(
    *,
    max_cut_groups: int,
    max_literals_per_group: int,
    max_total_pairs: int,
    max_source_terms: int,
    max_multiplier_bits: int,
    max_exact_bits: int,
    max_exact_nonzeros: int,
    max_parent_variables: int,
    max_parent_rows: int,
    max_parent_nonzeros: int,
    max_parent_buffer_items: int,
) -> Mapping[str, int]:
    values = {
        "max_cut_groups": _strict_int(
            max_cut_groups, name="max_cut_groups"
        ),
        "max_literals_per_group": _strict_int(
            max_literals_per_group,
            name="max_literals_per_group",
        ),
        "max_total_pairs": _strict_int(
            max_total_pairs, name="max_total_pairs"
        ),
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
        "max_parent_variables": _strict_int(
            max_parent_variables,
            name="max_parent_variables",
        ),
        "max_parent_rows": _strict_int(
            max_parent_rows,
            name="max_parent_rows",
        ),
        "max_parent_nonzeros": _strict_int(
            max_parent_nonzeros,
            name="max_parent_nonzeros",
        ),
        "max_parent_buffer_items": _strict_int(
            max_parent_buffer_items,
            name="max_parent_buffer_items",
        ),
    }
    if any(
        value < 1 or value > _HARD_LIMITS[name]
        for name, value in values.items()
    ):
        raise PropertyPhaseSubsetCliqueError(
            "subset_clique_cap_out_of_range"
        )
    return values


def _check_parent_work_caps(
    hz: SparseHZono,
    *,
    caps: Mapping[str, int],
) -> None:
    """Bound every whole-parent C/CSR operation before it starts."""

    if type(hz) is not SparseHZono:
        raise PropertyPhaseSubsetCliqueError(
            "parent_not_exact_sparse_hz"
        )
    variables = int(hz.n_cont) + int(hz.n_bin)
    rows = int(hz.n_out) + int(hz.n_eq) + int(hz.n_ub)
    matrices = (
        hz.Gc,
        hz.Gb,
        hz.Ac,
        hz.Ab,
        hz.Auc,
        hz.Aub,
    )
    if any(
        not hasattr(matrix, "nnz")
        or not hasattr(matrix, "data")
        or not hasattr(matrix, "indices")
        or not hasattr(matrix, "indptr")
        for matrix in matrices
    ):
        raise PropertyPhaseSubsetCliqueError(
            "parent_sparse_matrix_missing"
        )
    nonzeros = sum(int(matrix.nnz) for matrix in matrices)
    dense_arrays = (
        hz.c,
        hz.b,
        hz.ub,
        hz.col_ids,
        hz.bcol_ids,
    )
    if any(not hasattr(value, "size") for value in dense_arrays):
        raise PropertyPhaseSubsetCliqueError(
            "parent_dense_array_missing"
        )
    buffer_items = sum(
        int(matrix.data.size)
        + int(matrix.indices.size)
        + int(matrix.indptr.size)
        for matrix in matrices
    ) + sum(int(value.size) for value in dense_arrays)
    if (
        variables < 0
        or rows < 0
        or nonzeros < 0
        or variables > caps["max_parent_variables"]
        or rows > caps["max_parent_rows"]
        or nonzeros > caps["max_parent_nonzeros"]
        or buffer_items < 0
        or buffer_items > caps["max_parent_buffer_items"]
    ):
        raise PropertyPhaseSubsetCliqueError(
            "parent_work_cap_exceeded"
        )


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise PropertyPhaseSubsetCliqueError(
            f"deadline_expired_{stage}"
        )


def _subset_property_digest(
    *,
    parent_digest: str,
    property_digest: str,
    group: PropertyLiteralGroup,
) -> str:
    return _canonical_sha256(
        {
            "schema": (
                "act.pc_pcc.property_phase_subset_binding.v1"
            ),
            "parent_semantic_digest": parent_digest,
            "ordered_property_digest": property_digest,
            "group_id": group.group_id,
            "rival_binding_digests": list(
                group.rival_binding_digests
            ),
            "signature": [
                [
                    int(literal.stable_bcol_id),
                    int(literal.phase),
                ]
                for literal in group.literals
            ],
            "omitted_zero_bcol_ids": list(
                group.omitted_zero_bcol_ids
            ),
        }
    )


def _phase_literals(
    *,
    parent_digest: str,
    subset_property_digest: str,
    group: PropertyLiteralGroup,
) -> Tuple[PhaseLiteral, ...]:
    return tuple(
        PhaseLiteral(
            stable_bcol_id=int(literal.stable_bcol_id),
            phase=int(literal.phase),
            binding_digest=_literal_binding_digest(
                parent_digest=parent_digest,
                property_digest=subset_property_digest,
                stable_bcol_id=int(
                    literal.stable_bcol_id
                ),
                phase=int(literal.phase),
            ),
        )
        for literal in group.literals
    )


def _eligible_groups(
    grouping: PropertyLiteralGroupingResult,
    *,
    caps: Mapping[str, int],
) -> Tuple[PropertyLiteralGroup, ...]:
    groups = tuple(
        group
        for group in grouping.groups
        if len(group.literals) >= 2
    )
    if len(groups) > caps["max_cut_groups"]:
        raise PropertyPhaseSubsetCliqueError(
            "cut_group_cap_exceeded"
        )
    total_pairs = 0
    for group in groups:
        literal_count = len(group.literals)
        if literal_count > caps["max_literals_per_group"]:
            raise PropertyPhaseSubsetCliqueError(
                "group_literal_cap_exceeded"
            )
        total_pairs += literal_count * (literal_count - 1) // 2
    if total_pairs > caps["max_total_pairs"]:
        raise PropertyPhaseSubsetCliqueError(
            "total_pair_cap_exceeded"
        )
    return groups


def _grouping_receipt_digest(
    grouping: PropertyLiteralGroupingResult,
) -> str:
    digest = grouping.receipt.get("receipt_sha256")
    if not _valid_sha256(digest):
        raise PropertyPhaseSubsetCliqueError(
            "grouping_receipt_digest_invalid"
        )
    return digest


def run_property_phase_subset_clique_candidate(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    grouping: PropertyLiteralGroupingResult,
    *,
    deadline: float,
    grouping_max_rivals: int = 128,
    grouping_max_binaries: int = 2048,
    grouping_max_groups: int = 128,
    grouping_timeout_seconds: float = 5.0,
    max_cut_groups: int = _DEFAULT_MAX_CUT_GROUPS,
    max_literals_per_group: int = (
        _DEFAULT_MAX_LITERALS_PER_GROUP
    ),
    max_total_pairs: int = _DEFAULT_MAX_TOTAL_PAIRS,
    max_source_terms: int = _DEFAULT_MAX_SOURCE_TERMS,
    max_multiplier_bits: int = _DEFAULT_MAX_MULTIPLIER_BITS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_exact_nonzeros: int = _DEFAULT_MAX_EXACT_NONZEROS,
    max_parent_variables: int = (
        _DEFAULT_MAX_PARENT_VARIABLES
    ),
    max_parent_rows: int = _DEFAULT_MAX_PARENT_ROWS,
    max_parent_nonzeros: int = (
        _DEFAULT_MAX_PARENT_NONZEROS
    ),
    max_parent_buffer_items: int = (
        _DEFAULT_MAX_PARENT_BUFFER_ITEMS
    ),
) -> PropertyPhaseSubsetCliqueResult:
    """Check complete exact closures and append only proven subset cuts."""

    if not isinstance(hz, SparseHZono):
        raise PropertyPhaseSubsetCliqueError(
            "parent_not_sparse_hz"
        )
    deadline = float(deadline)
    if not math.isfinite(deadline):
        raise PropertyPhaseSubsetCliqueError(
            "deadline_nonfinite"
        )
    caps = _caps(
        max_cut_groups=max_cut_groups,
        max_literals_per_group=max_literals_per_group,
        max_total_pairs=max_total_pairs,
        max_source_terms=max_source_terms,
        max_multiplier_bits=max_multiplier_bits,
        max_exact_bits=max_exact_bits,
        max_exact_nonzeros=max_exact_nonzeros,
        max_parent_variables=max_parent_variables,
        max_parent_rows=max_parent_rows,
        max_parent_nonzeros=max_parent_nonzeros,
        max_parent_buffer_items=max_parent_buffer_items,
    )
    _check_parent_work_caps(hz, caps=caps)
    _check_deadline(deadline, "before_grouping_audit")
    if (
        deadline - time.monotonic()
        < float(grouping_timeout_seconds)
    ):
        raise PropertyPhaseSubsetCliqueError(
            "deadline_cannot_cover_grouping_audit"
        )
    if not verify_property_literal_grouping_result(
        hz,
        rivals,
        grouping,
        max_rivals=grouping_max_rivals,
        max_binaries=grouping_max_binaries,
        max_groups=grouping_max_groups,
        timeout_seconds=grouping_timeout_seconds,
    ):
        raise PropertyPhaseSubsetCliqueError(
            "grouping_result_invalid"
        )
    _check_deadline(deadline, "after_grouping_audit")
    groups = _eligible_groups(grouping, caps=caps)
    parent_digest = sparse_hz_semantic_digest(hz)
    _check_deadline(deadline, "after_parent_digest")
    if parent_digest != grouping.parent_semantic_digest:
        raise PropertyPhaseSubsetCliqueError(
            "grouping_parent_digest_stale"
        )
    _check_deadline(deadline, "after_parent_digest")
    frame_digest = None
    oracle = None
    if groups:
        frame_digest = _ordered_source_frame_digest(
            hz,
            parent_digest=parent_digest,
            deadline=deadline,
        )
        _check_deadline(deadline, "after_source_frame")
        oracle = _PersistentHighsPairLP(hz, deadline=deadline)
        _check_deadline(deadline, "after_model_build")
    closures = []
    cut_hz: Optional[SparseHZono] = None
    total_certificates = 0
    total_pairs = 0

    for group in groups:
        _check_deadline(deadline, "before_group")
        subset_digest = _subset_property_digest(
            parent_digest=parent_digest,
            property_digest=grouping.ordered_property_digest,
            group=group,
        )
        literals = _phase_literals(
            parent_digest=parent_digest,
            subset_property_digest=subset_digest,
            group=group,
        )
        pair_records = []
        certificates = []
        for left_index, left in enumerate(literals):
            for right in literals[left_index + 1 :]:
                _check_deadline(deadline, "before_pair")
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
                            property_digest=subset_digest,
                            source_frame_digest=frame_digest,
                            deadline=deadline,
                            max_source_terms=caps[
                                "max_source_terms"
                            ],
                            max_multiplier_bits=caps[
                                "max_multiplier_bits"
                            ],
                            max_exact_bits=caps[
                                "max_exact_bits"
                            ],
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
                pair_records.append(
                    PersistentPairRecord(
                        literals=pair,
                        status=status,
                        ray_nonzero_rows=ray_nonzero_rows,
                        certificate_sha256=certificate_digest,
                        rationalization=rationalization,
                    )
                )
                total_pairs += 1
        complete = bool(
            pair_records
            and all(
                record.status == "certified_conflict"
                for record in pair_records
            )
        )
        if complete:
            cut_hz = _copy_parent_with_clique_cut(
                hz if cut_hz is None else cut_hz,
                literals,
            )
        total_certificates += len(certificates)
        closures.append(
            SubsetCliqueClosure(
                group_id=group.group_id,
                subset_property_digest=subset_digest,
                literals=literals,
                omitted_zero_bcol_ids=(
                    group.omitted_zero_bcol_ids
                ),
                pair_records=tuple(pair_records),
                certificates=tuple(certificates),
                complete=complete,
                cut_applied=complete,
            )
        )
    _check_deadline(deadline, "after_all_groups")
    if sparse_hz_semantic_digest(hz) != parent_digest:
        raise PropertyPhaseSubsetCliqueError(
            "parent_mutated_during_candidate"
        )
    _check_deadline(deadline, "after_parent_terminal_digest")
    complete_count = sum(
        closure.complete for closure in closures
    )
    telemetry = {
        "schema": "act.pc_pcc.property_phase_subset_clique.v1",
        "caps": dict(caps),
        "grouping_caps": {
            "max_rivals": int(grouping_max_rivals),
            "max_binaries": int(grouping_max_binaries),
            "max_groups": int(grouping_max_groups),
            "timeout_seconds": float(
                grouping_timeout_seconds
            ),
        },
        "eligible_group_count": len(groups),
        "complete_group_count": int(complete_count),
        "pair_count": int(total_pairs),
        "exact_certificate_count": int(total_certificates),
        "model_builds": 0 if oracle is None else 1,
        "oracle": {} if oracle is None else dict(oracle.telemetry),
        "proof_authority": False,
    }
    return PropertyPhaseSubsetCliqueResult(
        status=(
            "subset_cut_candidate"
            if cut_hz is not None
            else "no_complete_subset_clique"
        ),
        hz=cut_hz,
        parent_semantic_digest=parent_digest,
        ordered_property_digest=(
            grouping.ordered_property_digest
        ),
        grouping_receipt_sha256=_grouping_receipt_digest(
            grouping
        ),
        closures=tuple(closures),
        telemetry=telemetry,
    )


def _closure_shape_is_exact(
    closure: Any,
    *,
    group: PropertyLiteralGroup,
    parent_digest: str,
    property_digest: str,
) -> bool:
    if (
        type(closure) is not SubsetCliqueClosure
        or closure.proof_authority is not False
        or not _valid_sha256(closure.group_id)
        or not _valid_sha256(
            closure.subset_property_digest
        )
        or type(closure.literals) is not tuple
        or type(closure.omitted_zero_bcol_ids) is not tuple
        or type(closure.pair_records) is not tuple
        or type(closure.certificates) is not tuple
        or type(closure.complete) is not bool
        or type(closure.cut_applied) is not bool
    ):
        return False
    literal_payload = _exact_literal_tuple_payload(
        closure.literals
    )
    if (
        literal_payload is None
        or any(
            type(stable_id) is not int or stable_id < 0
            for stable_id in closure.omitted_zero_bcol_ids
        )
    ):
        return False
    expected_digest = _subset_property_digest(
        parent_digest=parent_digest,
        property_digest=property_digest,
        group=group,
    )
    expected_literals = _phase_literals(
        parent_digest=parent_digest,
        subset_property_digest=expected_digest,
        group=group,
    )
    expected_literal_payload = _exact_literal_tuple_payload(
        expected_literals
    )
    if expected_literal_payload is None:
        return False
    return bool(
        closure.group_id == group.group_id
        and closure.subset_property_digest == expected_digest
        and literal_payload == expected_literal_payload
        and closure.omitted_zero_bcol_ids
        == group.omitted_zero_bcol_ids
    )


def _telemetry_is_exact(
    telemetry: Any,
    *,
    hz: SparseHZono,
    caps: Mapping[str, int],
    grouping_max_rivals: int,
    grouping_max_binaries: int,
    grouping_max_groups: int,
    grouping_timeout_seconds: float,
    group_count: int,
    complete_count: int,
    pair_count: int,
    certificate_count: int,
) -> bool:
    """Validate diagnostics without allowing them to affect proof identity."""

    expected_keys = {
        "schema",
        "caps",
        "grouping_caps",
        "eligible_group_count",
        "complete_group_count",
        "pair_count",
        "exact_certificate_count",
        "model_builds",
        "oracle",
        "proof_authority",
    }
    if (
        type(telemetry) is not dict
        or any(type(key) is not str for key in telemetry)
        or set(telemetry) != expected_keys
        or type(telemetry["schema"]) is not str
        or telemetry["schema"]
        != "act.pc_pcc.property_phase_subset_clique.v1"
        or telemetry["proof_authority"] is not False
    ):
        return False
    live_caps = telemetry["caps"]
    if (
        type(live_caps) is not dict
        or any(type(key) is not str for key in live_caps)
        or set(live_caps) != set(caps)
        or any(
            type(live_caps[name]) is not int
            or live_caps[name] != caps[name]
            for name in caps
        )
    ):
        return False
    grouping_caps = telemetry["grouping_caps"]
    expected_grouping_caps = {
        "max_rivals": _strict_int(
            grouping_max_rivals,
            name="grouping_max_rivals",
        ),
        "max_binaries": _strict_int(
            grouping_max_binaries,
            name="grouping_max_binaries",
        ),
        "max_groups": _strict_int(
            grouping_max_groups,
            name="grouping_max_groups",
        ),
        "timeout_seconds": float(
            grouping_timeout_seconds
        ),
    }
    if (
        type(grouping_caps) is not dict
        or any(
            type(key) is not str for key in grouping_caps
        )
        or set(grouping_caps) != set(expected_grouping_caps)
        or any(
            type(grouping_caps[name])
            is not type(expected_grouping_caps[name])
            or grouping_caps[name]
            != expected_grouping_caps[name]
            for name in expected_grouping_caps
        )
    ):
        return False
    expected_counts = {
        "eligible_group_count": group_count,
        "complete_group_count": complete_count,
        "pair_count": pair_count,
        "exact_certificate_count": certificate_count,
        "model_builds": 0 if group_count == 0 else 1,
    }
    if any(
        type(telemetry[name]) is not int
        or telemetry[name] != value
        for name, value in expected_counts.items()
    ):
        return False
    oracle = telemetry["oracle"]
    if group_count == 0:
        return type(oracle) is dict and not oracle
    oracle_keys = {
        "backend",
        "highs_version",
        "row_order",
        "presolve",
        "threads",
        "model_builds",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "solve_calls",
        "bound_update_calls",
        "dual_ray_calls",
        "highs_cumulative_run_time_seconds",
    }
    if (
        type(oracle) is not dict
        or any(type(key) is not str for key in oracle)
        or set(oracle) != oracle_keys
        or type(oracle["backend"]) is not str
        or oracle["backend"]
        != "highspy_persistent_simplex_dual_ray_v1"
        or type(oracle["highs_version"]) is not str
        or type(oracle["row_order"]) is not str
        or oracle["row_order"] != "upper_then_equality"
        or type(oracle["presolve"]) is not str
        or oracle["presolve"] != "off"
    ):
        return False
    integer_fields = (
        "threads",
        "model_builds",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "solve_calls",
        "bound_update_calls",
        "dual_ray_calls",
    )
    if any(
        type(oracle[name]) is not int or oracle[name] < 0
        for name in integer_fields
    ):
        return False
    runtime = oracle["highs_cumulative_run_time_seconds"]
    expected_nonzeros = int(
        hz.Auc.nnz + hz.Aub.nnz + hz.Ac.nnz + hz.Ab.nnz
    )
    return bool(
        oracle["threads"] >= 1
        and oracle["model_builds"] == 1
        and oracle["candidate_rows"] == hz.n_ub + hz.n_eq
        and oracle["candidate_columns"] == hz.n_cont + hz.n_bin
        and oracle["candidate_nonzeros"] == expected_nonzeros
        and oracle["solve_calls"] == pair_count
        and oracle["bound_update_calls"] == 2 * pair_count
        and oracle["dual_ray_calls"] <= pair_count
        and type(runtime) is float
        and math.isfinite(runtime)
        and runtime >= 0.0
    )


def verify_property_phase_subset_clique_result(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    grouping: PropertyLiteralGroupingResult,
    result: PropertyPhaseSubsetCliqueResult,
    *,
    deadline: Optional[float] = None,
    grouping_max_rivals: int = 128,
    grouping_max_binaries: int = 2048,
    grouping_max_groups: int = 128,
    grouping_timeout_seconds: float = 5.0,
    max_cut_groups: int = _DEFAULT_MAX_CUT_GROUPS,
    max_literals_per_group: int = (
        _DEFAULT_MAX_LITERALS_PER_GROUP
    ),
    max_total_pairs: int = _DEFAULT_MAX_TOTAL_PAIRS,
    max_source_terms: int = _DEFAULT_MAX_SOURCE_TERMS,
    max_multiplier_bits: int = _DEFAULT_MAX_MULTIPLIER_BITS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_exact_nonzeros: int = _DEFAULT_MAX_EXACT_NONZEROS,
    max_parent_variables: int = (
        _DEFAULT_MAX_PARENT_VARIABLES
    ),
    max_parent_rows: int = _DEFAULT_MAX_PARENT_ROWS,
    max_parent_nonzeros: int = (
        _DEFAULT_MAX_PARENT_NONZEROS
    ),
    max_parent_buffer_items: int = (
        _DEFAULT_MAX_PARENT_BUFFER_ITEMS
    ),
) -> bool:
    """Recheck every exact edge and reconstruct all emitted subset rows."""

    try:
        deadline_value = (
            time.monotonic() + 60.0
            if deadline is None
            else float(deadline)
        )
        if (
            not math.isfinite(deadline_value)
            or type(result)
            is not PropertyPhaseSubsetCliqueResult
            or result.proof_authority is not False
            or type(result.status) is not str
            or not _valid_sha256(
                result.parent_semantic_digest
            )
            or not _valid_sha256(
                result.ordered_property_digest
            )
            or not _valid_sha256(
                result.grouping_receipt_sha256
            )
            or type(result.closures) is not tuple
            or type(result.telemetry) is not dict
            or (
                result.hz is not None
                and type(result.hz) is not SparseHZono
            )
            or result.status
            not in {
                "subset_cut_candidate",
                "no_complete_subset_clique",
            }
        ):
            return False
        caps = _caps(
            max_cut_groups=max_cut_groups,
            max_literals_per_group=max_literals_per_group,
            max_total_pairs=max_total_pairs,
            max_source_terms=max_source_terms,
            max_multiplier_bits=max_multiplier_bits,
            max_exact_bits=max_exact_bits,
            max_exact_nonzeros=max_exact_nonzeros,
            max_parent_variables=max_parent_variables,
            max_parent_rows=max_parent_rows,
            max_parent_nonzeros=max_parent_nonzeros,
            max_parent_buffer_items=(
                max_parent_buffer_items
            ),
        )
        _check_parent_work_caps(hz, caps=caps)
        _check_deadline(deadline_value, "before_grouping_audit")
        if (
            deadline_value - time.monotonic()
            < float(grouping_timeout_seconds)
            or not verify_property_literal_grouping_result(
                hz,
                rivals,
                grouping,
                max_rivals=grouping_max_rivals,
                max_binaries=grouping_max_binaries,
                max_groups=grouping_max_groups,
                timeout_seconds=grouping_timeout_seconds,
            )
        ):
            return False
        _check_deadline(deadline_value, "after_grouping_audit")
        groups = _eligible_groups(grouping, caps=caps)
        parent_digest = sparse_hz_semantic_digest(hz)
        _check_deadline(deadline_value, "after_parent_digest")
        if (
            result.parent_semantic_digest != parent_digest
            or result.parent_semantic_digest
            != grouping.parent_semantic_digest
            or result.ordered_property_digest
            != grouping.ordered_property_digest
            or result.grouping_receipt_sha256
            != _grouping_receipt_digest(grouping)
            or len(result.closures) != len(groups)
        ):
            return False
        source_frame_digest = None
        if groups:
            source_frame_digest = _ordered_source_frame_digest(
                hz,
                parent_digest=parent_digest,
                deadline=deadline_value,
            )
            _check_deadline(
                deadline_value, "after_source_frame"
            )
        expected_certificate_count = 0
        expected_pair_count = 0
        complete_count = 0
        reconstructed: Optional[SparseHZono] = None
        seen_pair_keys = set()
        seen_certificate_digests = set()
        for group, closure in zip(groups, result.closures):
            _check_deadline(deadline_value, "verify_group")
            if not _closure_shape_is_exact(
                closure,
                group=group,
                parent_digest=parent_digest,
                property_digest=(
                    grouping.ordered_property_digest
                ),
            ):
                return False
            expected_subset_digest = _subset_property_digest(
                parent_digest=parent_digest,
                property_digest=(
                    grouping.ordered_property_digest
                ),
                group=group,
            )
            expected_literals = _phase_literals(
                parent_digest=parent_digest,
                subset_property_digest=(
                    expected_subset_digest
                ),
                group=group,
            )
            expected_pairs = tuple(
                _ordered_pair(left, right)
                for left_index, left in enumerate(
                    expected_literals
                )
                for right in expected_literals[
                    left_index + 1 :
                ]
            )
            if len(closure.pair_records) != len(expected_pairs):
                return False
            certificate_index = 0
            statuses = []
            for expected_pair, record in zip(
                expected_pairs, closure.pair_records
            ):
                expected_pair_payload = (
                    _exact_literal_tuple_payload(
                        expected_pair
                    )
                )
                record_pair_payload = (
                    _exact_literal_tuple_payload(
                        getattr(record, "literals", None)
                    )
                )
                if (
                    expected_pair_payload is None
                    or record_pair_payload is None
                ):
                    return False
                pair_key = (
                    expected_subset_digest,
                    tuple(
                        (item[0], item[1], item[2])
                        for item in expected_pair_payload
                    ),
                )
                if (
                    type(record) is not PersistentPairRecord
                    or len(record_pair_payload) != 2
                    or record_pair_payload
                    != expected_pair_payload
                    or pair_key in seen_pair_keys
                    or type(record.status) is not str
                    or record.status
                    not in {
                        "certified_conflict",
                        "feasible_or_unknown",
                        "infeasible_without_ray",
                        "exact_replay_rejected",
                    }
                    or type(record.ray_nonzero_rows) is not int
                    or record.ray_nonzero_rows < 0
                    or record.ray_nonzero_rows
                    > hz.n_ub + hz.n_eq
                ):
                    return False
                seen_pair_keys.add(pair_key)
                statuses.append(record.status)
                if record.status == "certified_conflict":
                    if (
                        certificate_index
                        >= len(closure.certificates)
                    ):
                        return False
                    certificate = closure.certificates[
                        certificate_index
                    ]
                    certificate_index += 1
                    certificate_pair_payload = (
                        _exact_literal_tuple_payload(
                            getattr(
                                certificate,
                                "literals",
                                None,
                            )
                        )
                    )
                    if (
                        not _exact_certificate_shape(certificate)
                        or certificate_pair_payload is None
                        or certificate_pair_payload
                        != expected_pair_payload
                        or not _valid_sha256(
                            record.certificate_sha256
                        )
                        or record.certificate_sha256
                        != certificate.certificate_sha256
                        or certificate.certificate_sha256
                        in seen_certificate_digests
                        or type(record.rationalization) is not str
                        or record.rationalization
                        != certificate.rationalization
                        or source_frame_digest is None
                        or not _verify_exact_certificate_with_source_frame(
                            hz,
                            certificate,
                            property_digest=(
                                expected_subset_digest
                            ),
                            parent_digest=parent_digest,
                            source_frame_digest=(
                                source_frame_digest
                            ),
                            deadline=deadline_value,
                            max_source_terms=caps[
                                "max_source_terms"
                            ],
                            max_multiplier_bits=caps[
                                "max_multiplier_bits"
                            ],
                            max_exact_bits=caps[
                                "max_exact_bits"
                            ],
                            max_exact_nonzeros=caps[
                                "max_exact_nonzeros"
                            ],
                        )
                    ):
                        return False
                    seen_certificate_digests.add(
                        certificate.certificate_sha256
                    )
                elif (
                    record.certificate_sha256 is not None
                    or record.rationalization is not None
                ):
                    return False
            if certificate_index != len(
                closure.certificates
            ):
                return False
            complete = bool(
                statuses
                and all(
                    status == "certified_conflict"
                    for status in statuses
                )
            )
            if (
                closure.complete is not complete
                or closure.cut_applied is not complete
            ):
                return False
            if complete:
                reconstructed = _copy_parent_with_clique_cut(
                    hz if reconstructed is None else reconstructed,
                    expected_literals,
                )
                complete_count += 1
            expected_pair_count += len(expected_pairs)
            expected_certificate_count += certificate_index
        expected_status = (
            "subset_cut_candidate"
            if reconstructed is not None
            else "no_complete_subset_clique"
        )
        if result.status != expected_status:
            return False
        if reconstructed is None:
            if result.hz is not None:
                return False
        elif (
            type(result.hz) is not SparseHZono
            or sparse_hz_semantic_digest(result.hz)
            != sparse_hz_semantic_digest(reconstructed)
        ):
            return False
        _check_deadline(deadline_value, "after_result_digest")
        if not _telemetry_is_exact(
            result.telemetry,
            hz=hz,
            caps=caps,
            grouping_max_rivals=grouping_max_rivals,
            grouping_max_binaries=grouping_max_binaries,
            grouping_max_groups=grouping_max_groups,
            grouping_timeout_seconds=(
                grouping_timeout_seconds
            ),
            group_count=len(groups),
            complete_count=complete_count,
            pair_count=expected_pair_count,
            certificate_count=expected_certificate_count,
        ):
            return False
        if sparse_hz_semantic_digest(hz) != parent_digest:
            return False
        _check_deadline(
            deadline_value, "after_parent_terminal_digest"
        )
        _check_deadline(deadline_value, "verify_complete")
        return True
    except (
        PropertyPhaseSubsetCliqueError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


__all__ = [
    "PropertyPhaseSubsetCliqueError",
    "PropertyPhaseSubsetCliqueResult",
    "SubsetCliqueClosure",
    "run_property_phase_subset_clique_candidate",
    "verify_property_phase_subset_clique_result",
]
