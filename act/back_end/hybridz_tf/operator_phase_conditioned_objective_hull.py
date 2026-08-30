#!/usr/bin/env python3
"""Toy-first exact core for a phase-conditioned objective hull (PCOH).

This module is deliberately isolated from the HybridZ pipeline and verdict
code.  It constructs a *descriptor* for the following extended formulation,
but does not allocate columns in a live :class:`SparseHZono`::

    lambda_p = (eta_p + 1) / 2,       eta_p in [-1, 1]
    sum_p lambda_p = 1
    xi_i = sum_p phase(p, i) lambda_p
    objective <= sum_p U_p lambda_p

All ``2**k`` signed patterns and their upper-bound handles are present and
``1 <= k <= 4``.  An externally replayed exact infeasibility certificate may
mark a pattern empty; its slot remains canonical, but an explicit equality
fixes ``eta_p = -1`` and hence ``lambda_p = 0``.  The objective row then uses
only patterns not certified empty, avoiding a numerically fragile cancellation
of an empty pattern's unused upper bound.

In the ACT
HybridZ convention the selected phase factors ``xi_i`` are binary variables
in ``{-1, 1}``.  Consequently the formulation is one-hot at every integer
phase assignment: the only feasible lambda has weight one on that pattern.

The emitted equality rows use the equivalent exact-Fraction form::

    sum_p eta_p = 2 - 2**k
    2 xi_i - sum_p phase(p, i) eta_p = 0

For an objective ``c + gc*xi_c + gb*xi_b`` the upper row is::

    gc*xi_c + gb*xi_b - 1/2 sum_p U_p eta_p
        <= 1/2 sum_p U_p - c

``U_p`` in this row is the exact dyadic value of the supplied outward
binary64 bound, not the potentially tighter pre-rounding rational bound.

Trust boundary
--------------
PCOH v2 consumes opaque, externally certified conditional upper bounds and
optional opaque exact-empty evidence.  It
checks their complete pattern coverage, binding digests, finite outward
storage, witness inclusion, and deterministic checksums, but it cannot replay
an arbitrary external proof.  The resulting descriptor therefore always has
``proof_authority=False`` and ``verdict_authority=False`` and must not
authorize a verifier result.  A future pipeline integration must pair it with
a concrete upstream certificate replayer.  The toy tests in the companion
module use controlled exact-Fraction oracles; they do not pretend that these
structural binders certify an opaque proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


_OBJECTIVE_SCHEMA = "act.hybridz_pc_objective_binding.v1"
_EXTERNAL_BOUND_SCHEMA = "act.hybridz_pc_external_pattern_upper.v1"
_EXTERNAL_EMPTY_SCHEMA = (
    "act.hybridz_pc_external_certified_empty_pattern.v1"
)
_DESCRIPTOR_SCHEMA = "act.hybridz_phase_conditioned_objective_hull.v2"
_RECEIPT_SCHEMA = "act.hybridz_phase_conditioned_objective_hull_receipt.v2"

_ZERO = Fraction(0)
_HALF = Fraction(1, 2)

ExactTerm = Tuple[int, Fraction]
SignedPattern = Tuple[int, ...]
PatternAssignment = Tuple[Tuple[int, int], ...]


class PhaseConditionedObjectiveHullError(ValueError):
    """Malformed, incomplete, stale, or tampered PCOH input/descriptor."""


@dataclass(frozen=True)
class ObjectiveBinding:
    """Exact stored objective coefficients bound to one parent HZ digest."""

    schema: str
    objective_id: str
    parent_semantic_digest: str
    center: Fraction
    continuous_terms: Tuple[ExactTerm, ...]
    binary_terms: Tuple[ExactTerm, ...]
    objective_binding_sha256: str


@dataclass(frozen=True)
class ExternalPatternUpperBound:
    """Structural handle for one externally certified conditional bound.

    ``certificate_sha256`` names an opaque certificate owned by the upstream
    producer.  Creating this handle does *not* certify ``upper_exact``.
    """

    schema: str
    assignments: PatternAssignment
    upper_exact: Fraction
    upper_stored: float
    parent_semantic_digest: str
    objective_binding_sha256: str
    certificate_schema: str
    certificate_sha256: str
    upstream_proof_authority: bool
    independently_certified: bool
    descriptor_sha256: str


@dataclass(frozen=True)
class ExternalCertifiedEmptyPattern:
    """Structural handle for one externally replayed empty pattern.

    The full pattern remains present in the PCOH descriptor and still owns a
    conditional upper-bound handle.  This object only records why its local
    ``lambda`` may be fixed to zero.  As with
    :class:`ExternalPatternUpperBound`, the booleans and opaque certificate
    digest are structural bindings rather than proof authority inside this
    module.  A live integration layer must replay the named certificate.
    """

    schema: str
    assignments: PatternAssignment
    witness_literals: PatternAssignment
    parent_semantic_digest: str
    property_digest: str
    selection_digest: str
    operator_row_tag_digest: str
    ordered_source_frame_sha256: str
    source_bundle_sha256: str
    coverage_sha256: str
    source_record_sha256: str
    local_row_map_sha256: str
    certificate_schema: str
    certificate_sha256: str
    eta_fixed_value: int
    upstream_exact_replay_authority: bool
    independently_exact_certified: bool
    descriptor_sha256: str


@dataclass(frozen=True)
class EtaColumn:
    """One descriptor-local continuous HZ factor and its lambda meaning."""

    local_index: int
    pattern: SignedPattern
    lower: Fraction = Fraction(-1)
    upper: Fraction = Fraction(1)
    lambda_scale: Fraction = _HALF
    lambda_shift: Fraction = _HALF


@dataclass(frozen=True)
class ExactHZLinearRow:
    """One exact row over stable parent ids and descriptor-local eta ids."""

    name: str
    sense: str
    parent_continuous_terms: Tuple[ExactTerm, ...]
    parent_binary_terms: Tuple[ExactTerm, ...]
    eta_terms: Tuple[ExactTerm, ...]
    rhs: Fraction


@dataclass(frozen=True)
class PhaseConditionedObjectiveHullDescriptor:
    """Canonical exact extended-formulation descriptor and audit receipt."""

    schema: str
    parent_semantic_digest: str
    objective_binding: ObjectiveBinding
    stable_bit_ids: Tuple[int, ...]
    patterns: Tuple[SignedPattern, ...]
    pattern_bounds: Tuple[ExternalPatternUpperBound, ...]
    empty_pattern_evidence: Tuple[ExternalCertifiedEmptyPattern, ...]
    eta_columns: Tuple[EtaColumn, ...]
    equality_rows: Tuple[ExactHZLinearRow, ...]
    upper_rows: Tuple[ExactHZLinearRow, ...]
    baseline_upper_stored: float
    representation_sha256: str
    receipt: Dict[str, Any]
    proof_authority: bool = False
    verdict_authority: bool = False


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
        raise PhaseConditionedObjectiveHullError(
            "payload_is_not_canonical_json"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, name: str) -> str:
    if not _valid_sha256(value):
        raise PhaseConditionedObjectiveHullError(f"{name}_invalid_sha256")
    return value


def _strict_id(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PhaseConditionedObjectiveHullError(f"{name}_not_integer")
    result = int(value)
    if result < 0:
        raise PhaseConditionedObjectiveHullError(f"{name}_negative")
    return result


def _fraction(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise PhaseConditionedObjectiveHullError(f"{name}_bool")
    if type(value) is Fraction:
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Real):
        stored = float(value)
        if not math.isfinite(stored):
            raise PhaseConditionedObjectiveHullError(f"{name}_nonfinite")
        return Fraction.from_float(stored)
    raise PhaseConditionedObjectiveHullError(f"{name}_not_rational")


def _strict_fraction(value: Any, *, name: str) -> Fraction:
    if type(value) is not Fraction:
        raise PhaseConditionedObjectiveHullError(f"{name}_not_exact_fraction")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PhaseConditionedObjectiveHullError(f"{name}_not_real")
    result = float(value)
    if not math.isfinite(result):
        raise PhaseConditionedObjectiveHullError(f"{name}_nonfinite")
    return result


def _rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def outward_float64(value: Fraction) -> float:
    """Return the least readily available finite binary64 not below value."""

    exact = _strict_fraction(value, name="outward_value")
    try:
        stored = float(exact)
    except OverflowError as exc:
        raise PhaseConditionedObjectiveHullError(
            "outward_value_has_no_finite_binary64_upper"
        ) from exc
    if stored == math.inf:
        raise PhaseConditionedObjectiveHullError(
            "outward_value_has_no_finite_binary64_upper"
        )
    if stored == -math.inf:
        stored = -float.fromhex("0x1.fffffffffffffp+1023")
    if Fraction.from_float(stored) < exact:
        stored = math.nextafter(stored, math.inf)
    if not math.isfinite(stored) or Fraction.from_float(stored) < exact:
        raise PhaseConditionedObjectiveHullError(
            "outward_binary64_conversion_failed"
        )
    return stored


def _canonical_terms(
    values: Sequence[Sequence[Any]], *, name: str
) -> Tuple[ExactTerm, ...]:
    if isinstance(values, (str, bytes)):
        raise PhaseConditionedObjectiveHullError(f"{name}_not_sequence")
    result = []
    seen = set()
    try:
        iterator = enumerate(values)
    except TypeError as exc:
        raise PhaseConditionedObjectiveHullError(f"{name}_not_sequence") from exc
    for offset, item in iterator:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise PhaseConditionedObjectiveHullError(
                f"{name}_{offset}_not_pair"
            )
        if len(item) != 2:
            raise PhaseConditionedObjectiveHullError(
                f"{name}_{offset}_not_pair"
            )
        stable_id = _strict_id(item[0], name=f"{name}_{offset}_id")
        coefficient = _fraction(item[1], name=f"{name}_{offset}_coefficient")
        if stable_id in seen:
            raise PhaseConditionedObjectiveHullError(f"{name}_duplicate_id")
        if coefficient == 0:
            raise PhaseConditionedObjectiveHullError(f"{name}_explicit_zero")
        seen.add(stable_id)
        result.append((stable_id, coefficient))
    return tuple(sorted(result))


def _term_payload(terms: Tuple[ExactTerm, ...]) -> list[list[Any]]:
    return [[stable_id, _rational_text(value)] for stable_id, value in terms]


def _objective_payload(binding: ObjectiveBinding) -> Dict[str, Any]:
    return {
        "schema": binding.schema,
        "objective_id": binding.objective_id,
        "parent_semantic_digest": binding.parent_semantic_digest,
        "center": _rational_text(binding.center),
        "continuous_terms": _term_payload(binding.continuous_terms),
        "binary_terms": _term_payload(binding.binary_terms),
    }


def build_objective_binding(
    *,
    objective_id: str,
    parent_semantic_digest: str,
    center: Any,
    continuous_terms: Sequence[Sequence[Any]] = (),
    binary_terms: Sequence[Sequence[Any]] = (),
) -> ObjectiveBinding:
    """Canonicalize an exact objective over stable parent generator ids."""

    if type(objective_id) is not str or not objective_id or len(objective_id) > 256:
        raise PhaseConditionedObjectiveHullError("objective_id_invalid")
    if any(ord(character) < 32 or ord(character) > 126 for character in objective_id):
        raise PhaseConditionedObjectiveHullError("objective_id_not_printable_ascii")
    parent = _require_sha256(
        parent_semantic_digest, name="parent_semantic_digest"
    )
    binding = ObjectiveBinding(
        schema=_OBJECTIVE_SCHEMA,
        objective_id=objective_id,
        parent_semantic_digest=parent,
        center=_fraction(center, name="objective_center"),
        continuous_terms=_canonical_terms(
            continuous_terms, name="objective_continuous_terms"
        ),
        binary_terms=_canonical_terms(
            binary_terms, name="objective_binary_terms"
        ),
        objective_binding_sha256="",
    )
    return ObjectiveBinding(
        **{
            **binding.__dict__,
            "objective_binding_sha256": _canonical_sha256(
                _objective_payload(binding)
            ),
        }
    )


def verify_objective_binding(binding: Any) -> bool:
    """Fail closed unless ``binding`` is the exact canonical immutable form."""

    try:
        if (
            type(binding) is not ObjectiveBinding
            or binding.schema != _OBJECTIVE_SCHEMA
            or type(binding.objective_id) is not str
            or not binding.objective_id
            or len(binding.objective_id) > 256
            or any(
                ord(character) < 32 or ord(character) > 126
                for character in binding.objective_id
            )
            or not _valid_sha256(binding.parent_semantic_digest)
            or type(binding.center) is not Fraction
            or type(binding.continuous_terms) is not tuple
            or type(binding.binary_terms) is not tuple
            or not _valid_sha256(binding.objective_binding_sha256)
        ):
            return False
        for terms in (binding.continuous_terms, binding.binary_terms):
            if tuple(sorted(terms)) != terms:
                return False
            seen = set()
            for item in terms:
                if (
                    type(item) is not tuple
                    or len(item) != 2
                    or type(item[0]) is not int
                    or item[0] < 0
                    or type(item[1]) is not Fraction
                    or item[1] == 0
                    or item[0] in seen
                ):
                    return False
                seen.add(item[0])
        return (
            _canonical_sha256(_objective_payload(binding))
            == binding.objective_binding_sha256
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _canonical_assignments(
    assignments: Sequence[Sequence[Any]],
) -> PatternAssignment:
    if isinstance(assignments, (str, bytes)):
        raise PhaseConditionedObjectiveHullError("assignments_not_sequence")
    result = []
    seen = set()
    for offset, item in enumerate(assignments):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise PhaseConditionedObjectiveHullError(
                f"assignment_{offset}_not_pair"
            )
        stable_id = _strict_id(item[0], name=f"assignment_{offset}_id")
        phase = item[1]
        if isinstance(phase, bool) or not isinstance(phase, Integral):
            raise PhaseConditionedObjectiveHullError(
                f"assignment_{offset}_phase_not_integer"
            )
        phase = int(phase)
        if phase not in {-1, 1}:
            raise PhaseConditionedObjectiveHullError(
                f"assignment_{offset}_phase_not_signed"
            )
        if stable_id in seen:
            raise PhaseConditionedObjectiveHullError("assignment_duplicate_id")
        seen.add(stable_id)
        result.append((stable_id, phase))
    if not result:
        raise PhaseConditionedObjectiveHullError("assignments_empty")
    return tuple(sorted(result))


def _external_bound_payload(
    bound: ExternalPatternUpperBound,
) -> Dict[str, Any]:
    return {
        "schema": bound.schema,
        "assignments": [list(item) for item in bound.assignments],
        "upper_exact": _rational_text(bound.upper_exact),
        "upper_stored_hex": bound.upper_stored.hex(),
        "parent_semantic_digest": bound.parent_semantic_digest,
        "objective_binding_sha256": bound.objective_binding_sha256,
        "certificate_schema": bound.certificate_schema,
        "certificate_sha256": bound.certificate_sha256,
        "upstream_proof_authority": bound.upstream_proof_authority,
        "independently_certified": bound.independently_certified,
    }


def bind_external_pattern_upper_bound(
    *,
    assignments: Sequence[Sequence[Any]],
    upper_exact: Fraction,
    upper_stored: Any,
    parent_semantic_digest: str,
    objective_binding_sha256: str,
    certificate_schema: str,
    certificate_sha256: str,
    upstream_proof_authority: bool,
    independently_certified: bool,
) -> ExternalPatternUpperBound:
    """Bind, but do not certify, one externally proven conditional bound."""

    exact = _strict_fraction(upper_exact, name="upper_exact")
    stored = _finite_float(upper_stored, name="upper_stored")
    if Fraction.from_float(stored) < exact:
        raise PhaseConditionedObjectiveHullError("upper_stored_not_outward")
    if type(certificate_schema) is not str or not certificate_schema:
        raise PhaseConditionedObjectiveHullError("certificate_schema_invalid")
    if len(certificate_schema) > 256 or any(
        ord(character) < 32 or ord(character) > 126
        for character in certificate_schema
    ):
        raise PhaseConditionedObjectiveHullError("certificate_schema_invalid")
    if upstream_proof_authority is not True:
        raise PhaseConditionedObjectiveHullError(
            "upstream_proof_authority_required"
        )
    if independently_certified is not True:
        raise PhaseConditionedObjectiveHullError(
            "independent_pattern_certificate_required"
        )
    bound = ExternalPatternUpperBound(
        schema=_EXTERNAL_BOUND_SCHEMA,
        assignments=_canonical_assignments(assignments),
        upper_exact=exact,
        upper_stored=stored,
        parent_semantic_digest=_require_sha256(
            parent_semantic_digest, name="bound_parent_semantic_digest"
        ),
        objective_binding_sha256=_require_sha256(
            objective_binding_sha256, name="bound_objective_binding"
        ),
        certificate_schema=certificate_schema,
        certificate_sha256=_require_sha256(
            certificate_sha256, name="certificate"
        ),
        upstream_proof_authority=True,
        independently_certified=True,
        descriptor_sha256="",
    )
    return ExternalPatternUpperBound(
        **{
            **bound.__dict__,
            "descriptor_sha256": _canonical_sha256(
                _external_bound_payload(bound)
            ),
        }
    )


def verify_external_pattern_upper_bound(bound: Any) -> bool:
    """Verify the structural binding, not the opaque upstream proof."""

    try:
        if (
            type(bound) is not ExternalPatternUpperBound
            or bound.schema != _EXTERNAL_BOUND_SCHEMA
            or type(bound.assignments) is not tuple
            or not bound.assignments
            or type(bound.upper_exact) is not Fraction
            or type(bound.upper_stored) is not float
            or not math.isfinite(bound.upper_stored)
            or Fraction.from_float(bound.upper_stored) < bound.upper_exact
            or not _valid_sha256(bound.parent_semantic_digest)
            or not _valid_sha256(bound.objective_binding_sha256)
            or type(bound.certificate_schema) is not str
            or not bound.certificate_schema
            or len(bound.certificate_schema) > 256
            or any(
                ord(character) < 32 or ord(character) > 126
                for character in bound.certificate_schema
            )
            or not _valid_sha256(bound.certificate_sha256)
            or bound.upstream_proof_authority is not True
            or bound.independently_certified is not True
            or not _valid_sha256(bound.descriptor_sha256)
        ):
            return False
        canonical = _canonical_assignments(bound.assignments)
        return (
            canonical == bound.assignments
            and _canonical_sha256(_external_bound_payload(bound))
            == bound.descriptor_sha256
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _external_empty_payload(
    evidence: ExternalCertifiedEmptyPattern,
) -> Dict[str, Any]:
    return {
        "schema": evidence.schema,
        "assignments": [list(item) for item in evidence.assignments],
        "witness_literals": [
            list(item) for item in evidence.witness_literals
        ],
        "parent_semantic_digest": evidence.parent_semantic_digest,
        "property_digest": evidence.property_digest,
        "selection_digest": evidence.selection_digest,
        "operator_row_tag_digest": evidence.operator_row_tag_digest,
        "ordered_source_frame_sha256": (
            evidence.ordered_source_frame_sha256
        ),
        "source_bundle_sha256": evidence.source_bundle_sha256,
        "coverage_sha256": evidence.coverage_sha256,
        "source_record_sha256": evidence.source_record_sha256,
        "local_row_map_sha256": evidence.local_row_map_sha256,
        "certificate_schema": evidence.certificate_schema,
        "certificate_sha256": evidence.certificate_sha256,
        "eta_fixed_value": evidence.eta_fixed_value,
        "upstream_exact_replay_authority": (
            evidence.upstream_exact_replay_authority
        ),
        "independently_exact_certified": (
            evidence.independently_exact_certified
        ),
    }


def _validate_empty_witness_subset(
    *,
    assignments: PatternAssignment,
    witness_literals: PatternAssignment,
) -> None:
    assignment_by_id = dict(assignments)
    if len(witness_literals) > len(assignments):
        raise PhaseConditionedObjectiveHullError(
            "empty_witness_larger_than_pattern"
        )
    for stable_id, phase in witness_literals:
        if stable_id not in assignment_by_id:
            raise PhaseConditionedObjectiveHullError(
                "empty_witness_id_not_in_pattern"
            )
        if assignment_by_id[stable_id] != phase:
            raise PhaseConditionedObjectiveHullError(
                "empty_witness_phase_not_in_pattern"
            )


def bind_external_certified_empty_pattern(
    *,
    assignments: Sequence[Sequence[Any]],
    witness_literals: Sequence[Sequence[Any]],
    parent_semantic_digest: str,
    property_digest: str,
    selection_digest: str,
    operator_row_tag_digest: str,
    ordered_source_frame_sha256: str,
    source_bundle_sha256: str,
    coverage_sha256: str,
    source_record_sha256: str,
    local_row_map_sha256: str,
    certificate_schema: str,
    certificate_sha256: str,
    eta_fixed_value: int,
    upstream_exact_replay_authority: bool,
    independently_exact_certified: bool,
) -> ExternalCertifiedEmptyPattern:
    """Bind, but do not replay, one exact empty-pattern certificate."""

    canonical_assignments = _canonical_assignments(assignments)
    canonical_witness = _canonical_assignments(witness_literals)
    _validate_empty_witness_subset(
        assignments=canonical_assignments,
        witness_literals=canonical_witness,
    )
    if type(eta_fixed_value) is not int or eta_fixed_value != -1:
        raise PhaseConditionedObjectiveHullError(
            "empty_eta_fixed_value_must_be_minus_one"
        )
    if (
        type(certificate_schema) is not str
        or not certificate_schema
        or len(certificate_schema) > 256
        or any(
            ord(character) < 32 or ord(character) > 126
            for character in certificate_schema
        )
    ):
        raise PhaseConditionedObjectiveHullError(
            "empty_certificate_schema_invalid"
        )
    if upstream_exact_replay_authority is not True:
        raise PhaseConditionedObjectiveHullError(
            "empty_upstream_exact_replay_authority_required"
        )
    if independently_exact_certified is not True:
        raise PhaseConditionedObjectiveHullError(
            "empty_independent_exact_certificate_required"
        )
    evidence = ExternalCertifiedEmptyPattern(
        schema=_EXTERNAL_EMPTY_SCHEMA,
        assignments=canonical_assignments,
        witness_literals=canonical_witness,
        parent_semantic_digest=_require_sha256(
            parent_semantic_digest,
            name="empty_parent_semantic_digest",
        ),
        property_digest=_require_sha256(
            property_digest, name="empty_property_digest"
        ),
        selection_digest=_require_sha256(
            selection_digest, name="empty_selection_digest"
        ),
        operator_row_tag_digest=_require_sha256(
            operator_row_tag_digest,
            name="empty_operator_row_tag_digest",
        ),
        ordered_source_frame_sha256=_require_sha256(
            ordered_source_frame_sha256,
            name="empty_ordered_source_frame",
        ),
        source_bundle_sha256=_require_sha256(
            source_bundle_sha256, name="empty_source_bundle"
        ),
        coverage_sha256=_require_sha256(
            coverage_sha256, name="empty_coverage"
        ),
        source_record_sha256=_require_sha256(
            source_record_sha256, name="empty_source_record"
        ),
        local_row_map_sha256=_require_sha256(
            local_row_map_sha256, name="empty_local_row_map"
        ),
        certificate_schema=certificate_schema,
        certificate_sha256=_require_sha256(
            certificate_sha256, name="empty_certificate"
        ),
        eta_fixed_value=-1,
        upstream_exact_replay_authority=True,
        independently_exact_certified=True,
        descriptor_sha256="",
    )
    return ExternalCertifiedEmptyPattern(
        **{
            **evidence.__dict__,
            "descriptor_sha256": _canonical_sha256(
                _external_empty_payload(evidence)
            ),
        }
    )


def verify_external_certified_empty_pattern(evidence: Any) -> bool:
    """Verify only the canonical structural binding of empty evidence."""

    try:
        if (
            type(evidence) is not ExternalCertifiedEmptyPattern
            or evidence.schema != _EXTERNAL_EMPTY_SCHEMA
            or type(evidence.assignments) is not tuple
            or type(evidence.witness_literals) is not tuple
            or not evidence.assignments
            or not evidence.witness_literals
            or not _valid_sha256(evidence.parent_semantic_digest)
            or not _valid_sha256(evidence.property_digest)
            or not _valid_sha256(evidence.selection_digest)
            or not _valid_sha256(evidence.operator_row_tag_digest)
            or not _valid_sha256(evidence.ordered_source_frame_sha256)
            or not _valid_sha256(evidence.source_bundle_sha256)
            or not _valid_sha256(evidence.coverage_sha256)
            or not _valid_sha256(evidence.source_record_sha256)
            or not _valid_sha256(evidence.local_row_map_sha256)
            or type(evidence.certificate_schema) is not str
            or not evidence.certificate_schema
            or len(evidence.certificate_schema) > 256
            or any(
                ord(character) < 32 or ord(character) > 126
                for character in evidence.certificate_schema
            )
            or not _valid_sha256(evidence.certificate_sha256)
            or type(evidence.eta_fixed_value) is not int
            or evidence.eta_fixed_value != -1
            or evidence.upstream_exact_replay_authority is not True
            or evidence.independently_exact_certified is not True
            or not _valid_sha256(evidence.descriptor_sha256)
        ):
            return False
        canonical_assignments = _canonical_assignments(
            evidence.assignments
        )
        canonical_witness = _canonical_assignments(
            evidence.witness_literals
        )
        _validate_empty_witness_subset(
            assignments=canonical_assignments,
            witness_literals=canonical_witness,
        )
        return (
            canonical_assignments == evidence.assignments
            and canonical_witness == evidence.witness_literals
            and _canonical_sha256(_external_empty_payload(evidence))
            == evidence.descriptor_sha256
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _canonical_stable_ids(values: Sequence[Any]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise PhaseConditionedObjectiveHullError("stable_bit_ids_not_sequence")
    result = tuple(
        _strict_id(value, name=f"stable_bit_ids_{offset}")
        for offset, value in enumerate(values)
    )
    if not 1 <= len(result) <= 4:
        raise PhaseConditionedObjectiveHullError("stable_bit_count_out_of_range")
    if len(set(result)) != len(result):
        raise PhaseConditionedObjectiveHullError("stable_bit_ids_duplicate")
    return tuple(sorted(result))


def _row_payload(row: ExactHZLinearRow) -> Dict[str, Any]:
    return {
        "name": row.name,
        "sense": row.sense,
        "parent_continuous_terms": _term_payload(
            row.parent_continuous_terms
        ),
        "parent_binary_terms": _term_payload(row.parent_binary_terms),
        "eta_terms": _term_payload(row.eta_terms),
        "rhs": _rational_text(row.rhs),
    }


def _representation_payload(
    *,
    parent_semantic_digest: str,
    objective_binding: ObjectiveBinding,
    stable_bit_ids: Tuple[int, ...],
    patterns: Tuple[SignedPattern, ...],
    pattern_bounds: Tuple[ExternalPatternUpperBound, ...],
    empty_pattern_evidence: Tuple[ExternalCertifiedEmptyPattern, ...],
    eta_columns: Tuple[EtaColumn, ...],
    equality_rows: Tuple[ExactHZLinearRow, ...],
    upper_rows: Tuple[ExactHZLinearRow, ...],
    baseline_upper_stored: float,
) -> Dict[str, Any]:
    return {
        "schema": _DESCRIPTOR_SCHEMA,
        "parent_semantic_digest": parent_semantic_digest,
        "objective_binding_sha256": (
            objective_binding.objective_binding_sha256
        ),
        "stable_bit_ids": list(stable_bit_ids),
        "patterns": [list(pattern) for pattern in patterns],
        "pattern_bound_descriptor_sha256": [
            bound.descriptor_sha256 for bound in pattern_bounds
        ],
        "empty_pattern_evidence_descriptor_sha256": [
            evidence.descriptor_sha256
            for evidence in empty_pattern_evidence
        ],
        "eta_columns": [
            {
                "local_index": column.local_index,
                "pattern": list(column.pattern),
                "lower": _rational_text(column.lower),
                "upper": _rational_text(column.upper),
                "lambda_scale": _rational_text(column.lambda_scale),
                "lambda_shift": _rational_text(column.lambda_shift),
            }
            for column in eta_columns
        ],
        "equality_rows": [_row_payload(row) for row in equality_rows],
        "upper_rows": [_row_payload(row) for row in upper_rows],
        "baseline_upper_stored_hex": baseline_upper_stored.hex(),
    }


def _make_receipt(
    *,
    representation_payload: Dict[str, Any],
    representation_sha256: str,
    pattern_bounds: Tuple[ExternalPatternUpperBound, ...],
    empty_pattern_evidence: Tuple[ExternalCertifiedEmptyPattern, ...],
    stable_bit_ids: Tuple[int, ...],
    baseline_upper_stored: float,
) -> Dict[str, Any]:
    receipt: Dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "algorithm": "full_pattern_lambda_eta_extended_hull",
        "proof_authority": False,
        "verdict_authority": False,
        "candidate_only": True,
        "upstream_bounds_required_independently_certified": True,
        "upstream_certificates_replayed": False,
        "upstream_empty_certificates_replayed": False,
        "external_certificate_digests_bound": True,
        "arithmetic": "exact_fraction_over_stored_binary64_bounds",
        "phase_encoding": "signed_binary_minus_one_plus_one",
        "eta_interval": "[-1,1]",
        "lambda_from_eta": "lambda=(eta+1)/2",
        "complete_pattern_cover": True,
        "complete_pattern_bound_cover": True,
        "canonical_pattern_slots_retained": True,
        "pattern_deletion": False,
        "integer_phase_forces_one_hot": True,
        "certified_empty_lambda_fixed_zero": True,
        "certified_empty_eta_fixed_minus_one": True,
        "objective_row_empty_terms_eliminated_exactly": True,
        "baseline_nonregression_checked": True,
        "baseline_upper_stored_hex": baseline_upper_stored.hex(),
        "stable_bits": len(stable_bit_ids),
        "patterns": len(pattern_bounds),
        "certified_empty_patterns": len(empty_pattern_evidence),
        "not_certified_empty_patterns": (
            len(pattern_bounds) - len(empty_pattern_evidence)
        ),
        "eta_columns": len(pattern_bounds),
        "empty_eta_fix_rows": len(empty_pattern_evidence),
        "equality_rows": (
            len(stable_bit_ids) + 1 + len(empty_pattern_evidence)
        ),
        "upper_rows": 1,
        "parent_semantic_digest": representation_payload[
            "parent_semantic_digest"
        ],
        "objective_binding_sha256": representation_payload[
            "objective_binding_sha256"
        ],
        "pattern_bound_descriptor_sha256": [
            bound.descriptor_sha256 for bound in pattern_bounds
        ],
        "external_certificate_sha256": [
            bound.certificate_sha256 for bound in pattern_bounds
        ],
        "empty_pattern_evidence_descriptor_sha256": [
            evidence.descriptor_sha256
            for evidence in empty_pattern_evidence
        ],
        "empty_external_certificate_sha256": [
            evidence.certificate_sha256
            for evidence in empty_pattern_evidence
        ],
        "empty_coverage_sha256": [
            evidence.coverage_sha256
            for evidence in empty_pattern_evidence
        ],
        "empty_source_record_sha256": [
            evidence.source_record_sha256
            for evidence in empty_pattern_evidence
        ],
        "empty_local_row_map_sha256": [
            evidence.local_row_map_sha256
            for evidence in empty_pattern_evidence
        ],
        "empty_assignments": [
            [list(item) for item in evidence.assignments]
            for evidence in empty_pattern_evidence
        ],
        "empty_witness_literals": [
            [list(item) for item in evidence.witness_literals]
            for evidence in empty_pattern_evidence
        ],
        "empty_source_join": (
            None
            if not empty_pattern_evidence
            else {
                "property_digest": (
                    empty_pattern_evidence[0].property_digest
                ),
                "selection_digest": (
                    empty_pattern_evidence[0].selection_digest
                ),
                "operator_row_tag_digest": (
                    empty_pattern_evidence[0].operator_row_tag_digest
                ),
                "ordered_source_frame_sha256": (
                    empty_pattern_evidence[0].ordered_source_frame_sha256
                ),
                "source_bundle_sha256": (
                    empty_pattern_evidence[0].source_bundle_sha256
                ),
            }
        ),
        "representation_sha256": representation_sha256,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def build_phase_conditioned_objective_hull(
    *,
    stable_bit_ids: Sequence[Any],
    pattern_bounds: Sequence[ExternalPatternUpperBound],
    objective_binding: ObjectiveBinding,
    parent_semantic_digest: str,
    baseline_upper_stored: Any,
    empty_pattern_evidence: Sequence[
        ExternalCertifiedEmptyPattern
    ] = (),
) -> PhaseConditionedObjectiveHullDescriptor:
    """Build the canonical exact descriptor after strict structural checks."""

    stable_ids = _canonical_stable_ids(stable_bit_ids)
    parent = _require_sha256(
        parent_semantic_digest, name="live_parent_semantic_digest"
    )
    if not verify_objective_binding(objective_binding):
        raise PhaseConditionedObjectiveHullError("objective_binding_invalid")
    if objective_binding.parent_semantic_digest != parent:
        raise PhaseConditionedObjectiveHullError("objective_binding_stale_parent")
    baseline = _finite_float(
        baseline_upper_stored, name="baseline_upper_stored"
    )
    if isinstance(pattern_bounds, (str, bytes)):
        raise PhaseConditionedObjectiveHullError("pattern_bounds_not_sequence")
    if isinstance(empty_pattern_evidence, (str, bytes)):
        raise PhaseConditionedObjectiveHullError(
            "empty_pattern_evidence_not_sequence"
        )

    expected_patterns = tuple(
        itertools.product((-1, 1), repeat=len(stable_ids))
    )
    by_pattern: Dict[SignedPattern, ExternalPatternUpperBound] = {}
    certificate_digests = set()
    for offset, bound in enumerate(pattern_bounds):
        if not verify_external_pattern_upper_bound(bound):
            raise PhaseConditionedObjectiveHullError(
                f"pattern_bound_{offset}_invalid"
            )
        if bound.parent_semantic_digest != parent:
            raise PhaseConditionedObjectiveHullError(
                f"pattern_bound_{offset}_stale_parent"
            )
        if (
            bound.objective_binding_sha256
            != objective_binding.objective_binding_sha256
        ):
            raise PhaseConditionedObjectiveHullError(
                f"pattern_bound_{offset}_stale_objective"
            )
        assignment_ids = tuple(item[0] for item in bound.assignments)
        if assignment_ids != stable_ids:
            raise PhaseConditionedObjectiveHullError(
                f"pattern_bound_{offset}_wrong_stable_ids"
            )
        pattern = tuple(item[1] for item in bound.assignments)
        if pattern in by_pattern:
            raise PhaseConditionedObjectiveHullError("duplicate_pattern_bound")
        if bound.certificate_sha256 in certificate_digests:
            raise PhaseConditionedObjectiveHullError(
                "pattern_certificates_not_independent"
            )
        if Fraction.from_float(bound.upper_stored) > Fraction.from_float(
            baseline
        ):
            raise PhaseConditionedObjectiveHullError(
                f"pattern_bound_{offset}_regresses_baseline"
            )
        by_pattern[pattern] = bound
        certificate_digests.add(bound.certificate_sha256)

    if set(by_pattern) != set(expected_patterns):
        missing = set(expected_patterns).difference(by_pattern)
        extra = set(by_pattern).difference(expected_patterns)
        if missing:
            raise PhaseConditionedObjectiveHullError(
                "incomplete_pattern_cover"
            )
        if extra:
            raise PhaseConditionedObjectiveHullError("unexpected_pattern")
        raise PhaseConditionedObjectiveHullError("pattern_cover_mismatch")
    if len(by_pattern) != 2 ** len(stable_ids):
        raise PhaseConditionedObjectiveHullError("pattern_count_mismatch")

    ordered_bounds = tuple(by_pattern[pattern] for pattern in expected_patterns)
    by_empty_pattern: Dict[
        SignedPattern, ExternalCertifiedEmptyPattern
    ] = {}
    empty_descriptor_digests = set()
    empty_coverage_digests = set()
    empty_source_join = None
    for offset, evidence in enumerate(empty_pattern_evidence):
        if not verify_external_certified_empty_pattern(evidence):
            raise PhaseConditionedObjectiveHullError(
                f"empty_pattern_evidence_{offset}_invalid"
            )
        if evidence.parent_semantic_digest != parent:
            raise PhaseConditionedObjectiveHullError(
                f"empty_pattern_evidence_{offset}_stale_parent"
            )
        assignment_ids = tuple(item[0] for item in evidence.assignments)
        if assignment_ids != stable_ids:
            raise PhaseConditionedObjectiveHullError(
                f"empty_pattern_evidence_{offset}_wrong_stable_ids"
            )
        pattern = tuple(item[1] for item in evidence.assignments)
        if pattern in by_empty_pattern:
            raise PhaseConditionedObjectiveHullError(
                "duplicate_empty_pattern_evidence"
            )
        if evidence.descriptor_sha256 in empty_descriptor_digests:
            raise PhaseConditionedObjectiveHullError(
                "duplicate_empty_evidence_descriptor"
            )
        if evidence.coverage_sha256 in empty_coverage_digests:
            raise PhaseConditionedObjectiveHullError(
                "empty_coverage_not_unique_per_pattern"
            )
        source_join = (
            evidence.property_digest,
            evidence.selection_digest,
            evidence.operator_row_tag_digest,
            evidence.ordered_source_frame_sha256,
            evidence.source_bundle_sha256,
        )
        if empty_source_join is None:
            empty_source_join = source_join
        elif source_join != empty_source_join:
            raise PhaseConditionedObjectiveHullError(
                "empty_evidence_source_join_mismatch"
            )
        by_empty_pattern[pattern] = evidence
        empty_descriptor_digests.add(evidence.descriptor_sha256)
        empty_coverage_digests.add(evidence.coverage_sha256)

    ordered_empty_evidence = tuple(
        by_empty_pattern[pattern]
        for pattern in expected_patterns
        if pattern in by_empty_pattern
    )
    empty_patterns = frozenset(by_empty_pattern)
    eta_columns = tuple(
        EtaColumn(
            local_index=index,
            pattern=pattern,
            lower=Fraction(-1),
            upper=(
                Fraction(-1)
                if pattern in empty_patterns
                else Fraction(1)
            ),
        )
        for index, pattern in enumerate(expected_patterns)
    )
    eta_count = len(eta_columns)
    normalization = ExactHZLinearRow(
        name="lambda_normalization",
        sense="eq",
        parent_continuous_terms=(),
        parent_binary_terms=(),
        eta_terms=tuple((index, Fraction(1)) for index in range(eta_count)),
        rhs=Fraction(2 - eta_count),
    )
    links = tuple(
        ExactHZLinearRow(
            name=f"stable_bit_link:{stable_id}",
            sense="eq",
            parent_continuous_terms=(),
            parent_binary_terms=((stable_id, Fraction(2)),),
            eta_terms=tuple(
                (index, Fraction(-pattern[bit_offset]))
                for index, pattern in enumerate(expected_patterns)
            ),
            rhs=_ZERO,
        )
        for bit_offset, stable_id in enumerate(stable_ids)
    )
    empty_fixes = tuple(
        ExactHZLinearRow(
            name=f"certified_empty_eta_fix:{index}",
            sense="eq",
            parent_continuous_terms=(),
            parent_binary_terms=(),
            eta_terms=((index, Fraction(1)),),
            rhs=Fraction(-1),
        )
        for index, pattern in enumerate(expected_patterns)
        if pattern in empty_patterns
    )
    stored_bounds = tuple(
        Fraction.from_float(bound.upper_stored) for bound in ordered_bounds
    )
    active_stored_bounds = tuple(
        (index, upper)
        for index, (pattern, upper) in enumerate(
            zip(expected_patterns, stored_bounds)
        )
        if pattern not in empty_patterns
    )
    objective_upper = ExactHZLinearRow(
        name="phase_conditioned_objective_upper",
        sense="le",
        parent_continuous_terms=objective_binding.continuous_terms,
        parent_binary_terms=objective_binding.binary_terms,
        eta_terms=tuple(
            (index, -_HALF * upper)
            for index, upper in active_stored_bounds
            if upper != 0
        ),
        rhs=(
            _HALF
            * sum((upper for _, upper in active_stored_bounds), _ZERO)
            - objective_binding.center
        ),
    )
    equality_rows = (normalization, *links, *empty_fixes)
    upper_rows = (objective_upper,)
    representation_payload = _representation_payload(
        parent_semantic_digest=parent,
        objective_binding=objective_binding,
        stable_bit_ids=stable_ids,
        patterns=expected_patterns,
        pattern_bounds=ordered_bounds,
        empty_pattern_evidence=ordered_empty_evidence,
        eta_columns=eta_columns,
        equality_rows=equality_rows,
        upper_rows=upper_rows,
        baseline_upper_stored=baseline,
    )
    representation_sha256 = _canonical_sha256(representation_payload)
    receipt = _make_receipt(
        representation_payload=representation_payload,
        representation_sha256=representation_sha256,
        pattern_bounds=ordered_bounds,
        empty_pattern_evidence=ordered_empty_evidence,
        stable_bit_ids=stable_ids,
        baseline_upper_stored=baseline,
    )
    return PhaseConditionedObjectiveHullDescriptor(
        schema=_DESCRIPTOR_SCHEMA,
        parent_semantic_digest=parent,
        objective_binding=objective_binding,
        stable_bit_ids=stable_ids,
        patterns=expected_patterns,
        pattern_bounds=ordered_bounds,
        empty_pattern_evidence=ordered_empty_evidence,
        eta_columns=eta_columns,
        equality_rows=equality_rows,
        upper_rows=upper_rows,
        baseline_upper_stored=baseline,
        representation_sha256=representation_sha256,
        receipt=receipt,
        proof_authority=False,
        verdict_authority=False,
    )


def verify_phase_conditioned_objective_hull(
    descriptor: Any,
    *,
    live_parent_semantic_digest: str,
    live_objective_binding: ObjectiveBinding,
) -> bool:
    """Strictly replay every structural coefficient and digest.

    This function raises on any mismatch.  A successful return remains a
    structural candidate check, not proof authority for the opaque upstream
    pattern certificates.
    """

    if type(descriptor) is not PhaseConditionedObjectiveHullDescriptor:
        raise PhaseConditionedObjectiveHullError("descriptor_wrong_type")
    if descriptor.schema != _DESCRIPTOR_SCHEMA:
        raise PhaseConditionedObjectiveHullError("descriptor_schema_mismatch")
    if descriptor.proof_authority is not False:
        raise PhaseConditionedObjectiveHullError(
            "descriptor_proof_authority_must_be_false"
        )
    if descriptor.verdict_authority is not False:
        raise PhaseConditionedObjectiveHullError(
            "descriptor_verdict_authority_must_be_false"
        )
    parent = _require_sha256(
        live_parent_semantic_digest, name="live_parent_semantic_digest"
    )
    if descriptor.parent_semantic_digest != parent:
        raise PhaseConditionedObjectiveHullError("descriptor_stale_parent")
    if not verify_objective_binding(live_objective_binding):
        raise PhaseConditionedObjectiveHullError(
            "live_objective_binding_invalid"
        )
    if descriptor.objective_binding != live_objective_binding:
        raise PhaseConditionedObjectiveHullError("descriptor_stale_objective")
    if type(descriptor.receipt) is not dict:
        raise PhaseConditionedObjectiveHullError("receipt_wrong_type")

    expected = build_phase_conditioned_objective_hull(
        stable_bit_ids=descriptor.stable_bit_ids,
        pattern_bounds=descriptor.pattern_bounds,
        objective_binding=descriptor.objective_binding,
        parent_semantic_digest=descriptor.parent_semantic_digest,
        baseline_upper_stored=descriptor.baseline_upper_stored,
        empty_pattern_evidence=descriptor.empty_pattern_evidence,
    )
    field_names = (
        "schema",
        "parent_semantic_digest",
        "objective_binding",
        "stable_bit_ids",
        "patterns",
        "pattern_bounds",
        "empty_pattern_evidence",
        "eta_columns",
        "equality_rows",
        "upper_rows",
        "baseline_upper_stored",
        "representation_sha256",
        "receipt",
        "proof_authority",
        "verdict_authority",
    )
    for field_name in field_names:
        if getattr(descriptor, field_name) != getattr(expected, field_name):
            raise PhaseConditionedObjectiveHullError(
                f"descriptor_tamper:{field_name}"
            )
    return True


def evaluate_exact_hz_row_lhs(
    row: ExactHZLinearRow,
    *,
    continuous_values: Optional[Mapping[int, Fraction]] = None,
    binary_values: Optional[Mapping[int, Fraction]] = None,
    eta_values: Optional[Mapping[int, Fraction]] = None,
) -> Fraction:
    """Evaluate a descriptor row exactly; intended for controlled toys."""

    if type(row) is not ExactHZLinearRow:
        raise PhaseConditionedObjectiveHullError("row_wrong_type")
    continuous_values = {} if continuous_values is None else continuous_values
    binary_values = {} if binary_values is None else binary_values
    eta_values = {} if eta_values is None else eta_values
    total = _ZERO
    for terms, values, name in (
        (row.parent_continuous_terms, continuous_values, "continuous"),
        (row.parent_binary_terms, binary_values, "binary"),
        (row.eta_terms, eta_values, "eta"),
    ):
        if not isinstance(values, Mapping):
            raise PhaseConditionedObjectiveHullError(
                f"{name}_values_not_mapping"
            )
        for stable_id, coefficient in terms:
            if stable_id not in values:
                raise PhaseConditionedObjectiveHullError(
                    f"{name}_value_missing:{stable_id}"
                )
            total += coefficient * _strict_fraction(
                values[stable_id], name=f"{name}_value_{stable_id}"
            )
    return total


__all__ = [
    "EtaColumn",
    "ExactHZLinearRow",
    "ExternalCertifiedEmptyPattern",
    "ExternalPatternUpperBound",
    "ObjectiveBinding",
    "PhaseConditionedObjectiveHullDescriptor",
    "PhaseConditionedObjectiveHullError",
    "bind_external_certified_empty_pattern",
    "bind_external_pattern_upper_bound",
    "build_objective_binding",
    "build_phase_conditioned_objective_hull",
    "evaluate_exact_hz_row_lhs",
    "outward_float64",
    "verify_external_certified_empty_pattern",
    "verify_external_pattern_upper_bound",
    "verify_objective_binding",
    "verify_phase_conditioned_objective_hull",
]
