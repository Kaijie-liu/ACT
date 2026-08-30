#!/usr/bin/env python3
"""Toy-only one-shot live adapter for the PCOH v2 descriptor.

The adapter deliberately stops before materialization or any verifier path.
It accepts the raw complete conditional-bound certificates and a pair-local
infeasibility bundle, then performs all authority-bearing checks itself under
one caller-owned absolute deadline:

* the pair bundle is replayed by its strict live verifier;
* the complete conditional cover is replayed through the shared-context
  absolute-deadline helper;
* empty patterns are re-derived from the replayed exact pair certificates,
  never from caller-supplied coverage/status/proof booleans; and
* the PCOH v2 exact-Fraction descriptor is built and structurally replayed.

The returned wrapper is an immutable, digest-bound *candidate*.  It owns no
HZ, solver object, verdict token, constructive-nonempty seal, or materializer
capability.  Consequently it is intentionally reusable and has no one-use
registry.  A future materializer that transfers a private mutable solver HZ
must add its own one-use ownership capability and must replay these live
sources again.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from numbers import Integral, Real
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    OperatorExactReLUPhaseSelection,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
    OperatorPhaseConditionedObjectiveBoundCertificate,
    _replay_complete_operator_phase_conditioned_objective_bounds_until,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    ExternalCertifiedEmptyPattern,
    PhaseConditionedObjectiveHullDescriptor,
    bind_external_certified_empty_pattern,
    build_phase_conditioned_objective_hull,
    verify_external_certified_empty_pattern,
    verify_phase_conditioned_objective_hull,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_pair_infeasibility import (
    PairInfeasibilityBundle,
    SignedPairRecord,
    verify_phase_conditioned_pair_infeasibility_bundle,
)
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    ExactDualRayConflictCertificate,
)
from act.back_end.solver.solver_hz import SparseHZono


_SCHEMA = "act.hybridz_pc_combined_live_adapter_candidate.v1"
_RECEIPT_SCHEMA = "act.hybridz_pc_combined_live_adapter_receipt.v1"
_EXACT_PAIR_CERTIFICATE_SCHEMA = "act.pc_pcc.exact_dual_ray_certificate.v2"
_MAX_STABLE_BITS = 4
_MAX_PATTERNS = 16
_MAX_SIGNED_PAIR_QUERIES = 24

SignedPattern = Tuple[int, ...]
PatternAssignment = Tuple[Tuple[int, int], ...]
SignedPair = Tuple[Tuple[int, int], Tuple[int, int]]


class PhaseConditionedLiveAdapterError(ValueError):
    """The one-shot live replay/cross-binding contract failed closed."""


@dataclass(frozen=True)
class LivePhaseConditionedObjectiveHullCandidate:
    """Digest-bound, non-authoritative result of the one-shot live adapter."""

    schema: str
    parent_semantic_digest: str
    terminal_parent_semantic_digest: str
    property_digest: str
    selection_digest: str
    operator_row_tag_digest: str
    ordered_source_frame_sha256: str
    focused_rival_id: int
    stable_bit_ids: Tuple[int, ...]
    conditional_replay_bundle_sha256: str
    conditional_certificate_sha256: Tuple[str, ...]
    pair_bundle_sha256: str
    pair_certificate_sha256: Tuple[str, ...]
    empty_pattern_evidence: Tuple[ExternalCertifiedEmptyPattern, ...]
    descriptor: PhaseConditionedObjectiveHullDescriptor
    receipt: Mapping[str, Any]
    candidate_sha256: str
    proof_authority: bool = False
    verdict_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority is not False:
            raise ValueError("combined candidate never has proof authority")
        if self.verdict_authority is not False:
            raise ValueError("combined candidate never has verdict authority")
        if type(self.receipt) is not MappingProxyType:
            object.__setattr__(
                self,
                "receipt",
                MappingProxyType(dict(self.receipt)),
            )


def _canonical_form(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PhaseConditionedLiveAdapterError(
                "canonical_payload_nonfinite_float"
            )
        return {"__binary64_hex__": value.hex()}
    if type(value) in {tuple, list}:
        return [_canonical_form(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_form(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise PhaseConditionedLiveAdapterError(
        f"canonical_payload_unsupported:{type(value).__name__}"
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
        raise PhaseConditionedLiveAdapterError(
            "canonical_payload_invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _deadline(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PhaseConditionedLiveAdapterError("deadline_not_real")
    deadline = float(value)
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise PhaseConditionedLiveAdapterError("deadline_expired_or_nonfinite")
    return deadline


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise PhaseConditionedLiveAdapterError(
            f"deadline_expired:{stage}:no_output"
        )


def _stable_ids(values: Any) -> Tuple[int, ...]:
    if type(values) is not tuple:
        raise PhaseConditionedLiveAdapterError(
            "stable_bit_ids_not_exact_tuple"
        )
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise PhaseConditionedLiveAdapterError(
                "stable_bit_id_not_integer"
            )
        stable_id = int(value)
        if stable_id < 0:
            raise PhaseConditionedLiveAdapterError("stable_bit_id_negative")
        result.append(stable_id)
    result_tuple = tuple(result)
    if not 1 <= len(result_tuple) <= _MAX_STABLE_BITS:
        raise PhaseConditionedLiveAdapterError(
            "stable_bit_count_out_of_range"
        )
    if tuple(sorted(result_tuple)) != result_tuple:
        raise PhaseConditionedLiveAdapterError(
            "stable_bit_ids_not_canonical"
        )
    if len(set(result_tuple)) != len(result_tuple):
        raise PhaseConditionedLiveAdapterError("stable_bit_ids_duplicate")
    return result_tuple


def _focused_rival_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PhaseConditionedLiveAdapterError(
            "focused_rival_id_not_integer"
        )
    result = int(value)
    if result < 0:
        raise PhaseConditionedLiveAdapterError("focused_rival_id_negative")
    return result


def _expected_patterns(stable_ids: Tuple[int, ...]) -> Tuple[SignedPattern, ...]:
    patterns = tuple(
        tuple(int(value) for value in pattern)
        for pattern in itertools.product((-1, 1), repeat=len(stable_ids))
    )
    if len(patterns) > _MAX_PATTERNS:
        raise PhaseConditionedLiveAdapterError("pattern_cap_exceeded")
    return patterns


def _preflight_conditional_certificates(
    certificates: Any,
    *,
    stable_ids: Tuple[int, ...],
    focused_rival_id: int,
    expected_patterns: Tuple[SignedPattern, ...],
) -> Tuple[OperatorPhaseConditionedObjectiveBoundCertificate, ...]:
    if type(certificates) is not tuple:
        raise PhaseConditionedLiveAdapterError(
            "conditional_certificates_not_exact_tuple"
        )
    if (
        len(certificates) != len(expected_patterns)
        or len(certificates) > _MAX_PATTERNS
    ):
        raise PhaseConditionedLiveAdapterError(
            "conditional_certificate_count_mismatch"
        )
    if any(
        type(item) is not OperatorPhaseConditionedObjectiveBoundCertificate
        for item in certificates
    ):
        raise PhaseConditionedLiveAdapterError(
            "conditional_certificate_wrong_type"
        )
    checked = tuple(certificates)
    if tuple(item.pattern for item in checked) != expected_patterns:
        raise PhaseConditionedLiveAdapterError(
            "conditional_patterns_not_canonical_complete_cover"
        )
    if any(
        item.stable_bit_ids != stable_ids
        or item.rival_id != focused_rival_id
        for item in checked
    ):
        raise PhaseConditionedLiveAdapterError(
            "conditional_intent_binding_mismatch"
        )
    if len({item.certificate_sha256 for item in checked}) != len(checked):
        raise PhaseConditionedLiveAdapterError(
            "conditional_certificate_digest_not_unique"
        )
    return checked


def _assignments(
    stable_ids: Tuple[int, ...], pattern: SignedPattern
) -> PatternAssignment:
    return tuple(zip(stable_ids, pattern))


def _certificate_witness(
    certificate: ExactDualRayConflictCertificate,
) -> SignedPair:
    witness = tuple(
        (int(literal.stable_bcol_id), int(literal.phase))
        for literal in certificate.literals
    )
    if len(witness) != 2:
        raise PhaseConditionedLiveAdapterError(
            "pair_certificate_literal_count_mismatch"
        )
    return (witness[0], witness[1])


def _rederive_empty_evidence(
    pair_bundle: PairInfeasibilityBundle,
    *,
    stable_ids: Tuple[int, ...],
    expected_patterns: Tuple[SignedPattern, ...],
) -> Tuple[ExternalCertifiedEmptyPattern, ...]:
    """Derive pattern emptiness only from exact-replayed certificates.

    The strict pair verifier has already replayed every certificate and
    proved the record/coverage digests.  This function intentionally ignores
    bundle/record/coverage proof booleans and does not use coverage status to
    decide emptiness.
    """

    if len(pair_bundle.records) > _MAX_SIGNED_PAIR_QUERIES:
        raise PhaseConditionedLiveAdapterError("signed_pair_query_cap_exceeded")
    certificate_entries = []
    for certificate in pair_bundle.certificates:
        if type(certificate) is not ExactDualRayConflictCertificate:
            raise PhaseConditionedLiveAdapterError(
                "pair_certificate_wrong_type_after_verifier"
            )
        witness = _certificate_witness(certificate)
        if (
            tuple(sorted(witness)) != witness
            or len({item[0] for item in witness}) != 2
            or any(item[0] not in stable_ids or item[1] not in {-1, 1} for item in witness)
        ):
            raise PhaseConditionedLiveAdapterError(
                "pair_certificate_witness_noncanonical"
            )
        records = tuple(
            record
            for record in pair_bundle.records
            if record.certificate_sha256 == certificate.certificate_sha256
        )
        if len(records) != 1 or type(records[0]) is not SignedPairRecord:
            raise PhaseConditionedLiveAdapterError(
                "pair_certificate_record_join_mismatch"
            )
        record = records[0]
        if record.pair != witness:
            raise PhaseConditionedLiveAdapterError(
                "pair_certificate_record_witness_mismatch"
            )
        certificate_entries.append((witness, certificate, record))
    certificate_entries.sort(
        key=lambda item: (item[0], item[1].certificate_sha256)
    )

    coverage_by_pattern = {
        item.pattern: item for item in pair_bundle.coverage
    }
    if (
        len(coverage_by_pattern) != len(expected_patterns)
        or tuple(item.pattern for item in pair_bundle.coverage)
        != expected_patterns
    ):
        raise PhaseConditionedLiveAdapterError(
            "pair_coverage_not_canonical_complete"
        )

    evidence = []
    for pattern in expected_patterns:
        assignment_by_id = dict(_assignments(stable_ids, pattern))
        matches = tuple(
            entry
            for entry in certificate_entries
            if all(
                assignment_by_id[stable_id] == phase
                for stable_id, phase in entry[0]
            )
        )
        if not matches:
            continue
        witness, certificate, record = matches[0]
        coverage = coverage_by_pattern[pattern]
        # These fields do not confer authority.  They are checked only to bind
        # the unique, strict-verifier-validated coverage digest to the newly
        # derived witness and pattern.
        if (
            coverage.witness_pair != witness
            or coverage.certificate_sha256 != certificate.certificate_sha256
            or coverage.eta_fixed_value != -1
        ):
            raise PhaseConditionedLiveAdapterError(
                "rederived_empty_coverage_join_mismatch"
            )
        bound = bind_external_certified_empty_pattern(
            assignments=_assignments(stable_ids, pattern),
            witness_literals=witness,
            parent_semantic_digest=pair_bundle.parent_semantic_digest,
            property_digest=pair_bundle.property_digest,
            selection_digest=pair_bundle.selection_digest,
            operator_row_tag_digest=pair_bundle.operator_row_tag_digest,
            ordered_source_frame_sha256=(
                pair_bundle.ordered_source_frame_sha256
            ),
            source_bundle_sha256=pair_bundle.bundle_sha256,
            coverage_sha256=coverage.coverage_sha256,
            source_record_sha256=record.record_sha256,
            local_row_map_sha256=record.local_row_map_sha256,
            certificate_schema=_EXACT_PAIR_CERTIFICATE_SCHEMA,
            certificate_sha256=certificate.certificate_sha256,
            eta_fixed_value=-1,
            # Both booleans are minted locally from the successful live exact
            # replay event above; no caller-provided boolean is consulted.
            upstream_exact_replay_authority=True,
            independently_exact_certified=True,
        )
        if not verify_external_certified_empty_pattern(bound):
            raise PhaseConditionedLiveAdapterError(
                "new_empty_evidence_structural_verification_failed"
            )
        evidence.append(bound)
    if len(evidence) > _MAX_PATTERNS:
        raise PhaseConditionedLiveAdapterError("empty_evidence_cap_exceeded")
    return tuple(evidence)


def _make_receipt(
    *,
    parent_digest: str,
    terminal_parent_digest: str,
    property_digest: str,
    selection_digest: str,
    operator_row_tag_digest: str,
    source_frame_digest: str,
    focused_rival_id: int,
    stable_ids: Tuple[int, ...],
    conditional_bundle_sha256: str,
    conditional_certificate_sha256: Tuple[str, ...],
    pair_bundle_sha256: str,
    pair_certificate_sha256: Tuple[str, ...],
    evidence: Tuple[ExternalCertifiedEmptyPattern, ...],
    descriptor: PhaseConditionedObjectiveHullDescriptor,
) -> MappingProxyType:
    receipt: Dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "algorithm": "one_shot_live_conditional_pair_pc_objective_hull_v2",
        "proof_authority": False,
        "verdict_authority": False,
        "candidate_only": True,
        "single_caller_owned_absolute_deadline": True,
        "no_partial_output_on_failure": True,
        "conditional_complete_cover_live_replayed": True,
        "conditional_shared_verified_context": True,
        "pair_bundle_strict_live_verified": True,
        "empty_evidence_rederived_from_exact_certificates": True,
        "external_coverage_status_used_as_authority": False,
        "external_proof_booleans_used_as_authority": False,
        "all_pattern_upper_bound_handles_retained": True,
        "empty_pattern_placeholder_used": False,
        "terminal_parent_digest_rechecked": True,
        "adapter_full_parent_semantic_digest_computations": 1,
        "redundant_intermediate_parent_scans": False,
        "materialized_hz": False,
        "solver_handoff_capability_issued": False,
        "constructive_nonempty_capability_issued": False,
        "one_use_registry_used": False,
        "one_use_registry_required_for_pure_descriptor": False,
        "future_mutable_solver_handoff_requires_one_use_registry": True,
        "stable_bit_hard_cap": _MAX_STABLE_BITS,
        "pattern_hard_cap": _MAX_PATTERNS,
        "signed_pair_query_hard_cap": _MAX_SIGNED_PAIR_QUERIES,
        "stable_bits": len(stable_ids),
        "patterns": len(descriptor.patterns),
        "conditional_certificates": len(conditional_certificate_sha256),
        "pair_certificates": len(pair_certificate_sha256),
        "certified_empty_patterns": len(evidence),
        "not_certified_empty_patterns": (
            len(descriptor.patterns) - len(evidence)
        ),
        "parent_semantic_digest": parent_digest,
        "terminal_parent_semantic_digest": terminal_parent_digest,
        "property_digest": property_digest,
        "selection_digest": selection_digest,
        "operator_row_tag_digest": operator_row_tag_digest,
        "ordered_source_frame_sha256": source_frame_digest,
        "focused_rival_id": focused_rival_id,
        "stable_bit_ids": stable_ids,
        "objective_binding_sha256": (
            descriptor.objective_binding.objective_binding_sha256
        ),
        "conditional_replay_bundle_sha256": conditional_bundle_sha256,
        "conditional_certificate_sha256": conditional_certificate_sha256,
        "pattern_bound_descriptor_sha256": tuple(
            bound.descriptor_sha256 for bound in descriptor.pattern_bounds
        ),
        "pair_bundle_sha256": pair_bundle_sha256,
        "pair_certificate_sha256": pair_certificate_sha256,
        "empty_evidence_descriptor_sha256": tuple(
            item.descriptor_sha256 for item in evidence
        ),
        "empty_coverage_sha256": tuple(
            item.coverage_sha256 for item in evidence
        ),
        "descriptor_representation_sha256": descriptor.representation_sha256,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return MappingProxyType(receipt)


def _candidate_payload(
    candidate: LivePhaseConditionedObjectiveHullCandidate,
    *,
    include_digest: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": candidate.schema,
        "parent_semantic_digest": candidate.parent_semantic_digest,
        "terminal_parent_semantic_digest": (
            candidate.terminal_parent_semantic_digest
        ),
        "property_digest": candidate.property_digest,
        "selection_digest": candidate.selection_digest,
        "operator_row_tag_digest": candidate.operator_row_tag_digest,
        "ordered_source_frame_sha256": (
            candidate.ordered_source_frame_sha256
        ),
        "focused_rival_id": candidate.focused_rival_id,
        "stable_bit_ids": candidate.stable_bit_ids,
        "conditional_replay_bundle_sha256": (
            candidate.conditional_replay_bundle_sha256
        ),
        "conditional_certificate_sha256": (
            candidate.conditional_certificate_sha256
        ),
        "pair_bundle_sha256": candidate.pair_bundle_sha256,
        "pair_certificate_sha256": candidate.pair_certificate_sha256,
        "empty_pattern_evidence_descriptor_sha256": tuple(
            item.descriptor_sha256
            for item in candidate.empty_pattern_evidence
        ),
        "descriptor_schema": candidate.descriptor.schema,
        "descriptor_representation_sha256": (
            candidate.descriptor.representation_sha256
        ),
        "receipt_sha256": candidate.receipt.get("receipt_sha256"),
        "proof_authority": candidate.proof_authority,
        "verdict_authority": candidate.verdict_authority,
    }
    if include_digest:
        payload["candidate_sha256"] = candidate.candidate_sha256
    return payload


def build_live_phase_conditioned_objective_hull_candidate(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    selection: OperatorExactReLUPhaseSelection,
    *,
    focused_rival_id: int,
    stable_bit_ids: Tuple[int, ...],
    conditional_certificates: Tuple[
        OperatorPhaseConditionedObjectiveBoundCertificate, ...
    ],
    pair_bundle: PairInfeasibilityBundle,
    deadline: float,
) -> LivePhaseConditionedObjectiveHullCandidate:
    """Live-replay and cross-bind one complete toy PCOH v2 candidate.

    No public replay bundle is accepted: the conditional handles are minted
    inside this call from the raw certificates.  The sole pair input is
    accepted only after its strict verifier succeeds on the same live parent.
    """

    deadline_value = _deadline(deadline)
    stable_ids = _stable_ids(stable_bit_ids)
    focused_id = _focused_rival_id(focused_rival_id)
    expected_patterns = _expected_patterns(stable_ids)
    if (
        type(build) is not OperatorHZBuild
        or type(build.hz) is not SparseHZono
        or type(selection) is not OperatorExactReLUPhaseSelection
        or type(pair_bundle) is not PairInfeasibilityBundle
    ):
        raise PhaseConditionedLiveAdapterError(
            "build_selection_or_pair_bundle_wrong_type"
        )
    checked_certificates = _preflight_conditional_certificates(
        conditional_certificates,
        stable_ids=stable_ids,
        focused_rival_id=focused_id,
        expected_patterns=expected_patterns,
    )
    if pair_bundle.stable_bit_ids != stable_ids:
        raise PhaseConditionedLiveAdapterError(
            "pair_bundle_stable_ids_mismatch"
        )

    # The strict pair verifier is the first live parent seal.  Reuse the
    # selection-bound expected digest here instead of scanning the full HZ an
    # extra time; pair replay below proves that it is the current live digest.
    entry_parent = selection.parent_semantic_digest
    if not _valid_sha256(entry_parent):
        raise PhaseConditionedLiveAdapterError(
            "selection_parent_semantic_digest_invalid"
        )
    if (
        selection.parent_semantic_digest != entry_parent
        or pair_bundle.parent_semantic_digest != entry_parent
        or any(
            item.parent_semantic_digest != entry_parent
            for item in checked_certificates
        )
    ):
        raise PhaseConditionedLiveAdapterError(
            "entry_parent_cross_binding_mismatch"
        )
    if any(
        item.property_digest != selection.property_digest
        or item.selection_digest != selection.selection_digest
        or item.operator_row_tag_digest != selection.operator_row_tag_digest
        for item in checked_certificates
    ):
        raise PhaseConditionedLiveAdapterError(
            "conditional_selection_cross_binding_mismatch"
        )

    # Verify the relatively cheap pair source before replaying all 2**k
    # conditional bounds.  A stale/forged pair input therefore fails early.
    if not verify_phase_conditioned_pair_infeasibility_bundle(
        build,
        rivals,
        selection,
        pair_bundle,
        deadline=deadline_value,
    ):
        raise PhaseConditionedLiveAdapterError(
            "pair_bundle_live_verification_failed"
        )
    _check_deadline(deadline_value, "after_pair_bundle_replay")
    if (
        pair_bundle.terminal_parent_semantic_digest != entry_parent
        or pair_bundle.property_digest != selection.property_digest
        or pair_bundle.selection_digest != selection.selection_digest
        or pair_bundle.operator_row_tag_digest
        != selection.operator_row_tag_digest
    ):
        raise PhaseConditionedLiveAdapterError(
            "pair_bundle_terminal_cross_binding_failed"
        )

    try:
        replayed = (
            _replay_complete_operator_phase_conditioned_objective_bounds_until(
                build,
                rivals,
                selection,
                checked_certificates,
                deadline=deadline_value,
            )
        )
    except Exception as exc:
        raise PhaseConditionedLiveAdapterError(
            "conditional_complete_live_replay_failed"
        ) from exc
    _check_deadline(deadline_value, "after_conditional_complete_replay")
    conditional_certificate_sha = tuple(
        item.certificate_sha256 for item in checked_certificates
    )
    if (
        replayed.parent_semantic_digest != entry_parent
        or replayed.stable_bit_ids != stable_ids
        or replayed.certificate_sha256 != conditional_certificate_sha
        or replayed.objective_binding.parent_semantic_digest != entry_parent
        or len(replayed.pattern_bounds) != len(expected_patterns)
        or tuple(
            tuple(phase for _, phase in bound.assignments)
            for bound in replayed.pattern_bounds
        )
        != expected_patterns
        or tuple(
            tuple(stable_id for stable_id, _ in bound.assignments)
            for bound in replayed.pattern_bounds
        )
        != tuple(stable_ids for _ in expected_patterns)
    ):
        raise PhaseConditionedLiveAdapterError(
            "conditional_replay_cross_binding_failed"
        )

    evidence = _rederive_empty_evidence(
        pair_bundle,
        stable_ids=stable_ids,
        expected_patterns=expected_patterns,
    )
    _check_deadline(deadline_value, "after_empty_evidence_derivation")
    try:
        descriptor = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=replayed.pattern_bounds,
            objective_binding=replayed.objective_binding,
            parent_semantic_digest=entry_parent,
            baseline_upper_stored=replayed.baseline_upper_stored,
            empty_pattern_evidence=evidence,
        )
        if not verify_phase_conditioned_objective_hull(
            descriptor,
            live_parent_semantic_digest=entry_parent,
            live_objective_binding=replayed.objective_binding,
        ):
            raise PhaseConditionedLiveAdapterError(
                "pc_objective_hull_structural_replay_returned_false"
            )
    except PhaseConditionedLiveAdapterError:
        raise
    except Exception as exc:
        raise PhaseConditionedLiveAdapterError(
            "pc_objective_hull_build_or_structural_replay_failed"
        ) from exc
    _check_deadline(deadline_value, "after_pc_objective_hull_replay")

    if (
        descriptor.parent_semantic_digest != entry_parent
        or descriptor.empty_pattern_evidence != evidence
        or descriptor.pattern_bounds != replayed.pattern_bounds
    ):
        raise PhaseConditionedLiveAdapterError(
            "descriptor_cross_binding_failed"
        )
    # Bind the expected terminal value now, but defer the one outer terminal
    # full-parent scan until every receipt/digest byte is ready to issue.
    terminal_parent = entry_parent

    pair_certificate_sha = tuple(
        item.certificate_sha256 for item in pair_bundle.certificates
    )
    receipt = _make_receipt(
        parent_digest=entry_parent,
        terminal_parent_digest=terminal_parent,
        property_digest=selection.property_digest,
        selection_digest=selection.selection_digest,
        operator_row_tag_digest=selection.operator_row_tag_digest,
        source_frame_digest=pair_bundle.ordered_source_frame_sha256,
        focused_rival_id=focused_id,
        stable_ids=stable_ids,
        conditional_bundle_sha256=replayed.replay_bundle_sha256,
        conditional_certificate_sha256=conditional_certificate_sha,
        pair_bundle_sha256=pair_bundle.bundle_sha256,
        pair_certificate_sha256=pair_certificate_sha,
        evidence=evidence,
        descriptor=descriptor,
    )
    placeholder = LivePhaseConditionedObjectiveHullCandidate(
        schema=_SCHEMA,
        parent_semantic_digest=entry_parent,
        terminal_parent_semantic_digest=terminal_parent,
        property_digest=selection.property_digest,
        selection_digest=selection.selection_digest,
        operator_row_tag_digest=selection.operator_row_tag_digest,
        ordered_source_frame_sha256=(
            pair_bundle.ordered_source_frame_sha256
        ),
        focused_rival_id=focused_id,
        stable_bit_ids=stable_ids,
        conditional_replay_bundle_sha256=replayed.replay_bundle_sha256,
        conditional_certificate_sha256=conditional_certificate_sha,
        pair_bundle_sha256=pair_bundle.bundle_sha256,
        pair_certificate_sha256=pair_certificate_sha,
        empty_pattern_evidence=evidence,
        descriptor=descriptor,
        receipt=receipt,
        candidate_sha256="",
        proof_authority=False,
        verdict_authority=False,
    )
    result = LivePhaseConditionedObjectiveHullCandidate(
        **{
            **placeholder.__dict__,
            "candidate_sha256": _canonical_sha256(
                _candidate_payload(placeholder, include_digest=False)
            ),
        }
    )
    _check_deadline(deadline_value, "before_terminal_parent_seal")
    observed_terminal_parent = sparse_hz_semantic_digest(build.hz)
    if observed_terminal_parent != entry_parent:
        raise PhaseConditionedLiveAdapterError(
            "terminal_parent_semantic_digest_mismatch:no_output"
        )
    _check_deadline(deadline_value, "after_terminal_parent_seal")
    return result


def verify_live_phase_conditioned_objective_hull_candidate_structure(
    build: OperatorHZBuild,
    selection: OperatorExactReLUPhaseSelection,
    candidate: LivePhaseConditionedObjectiveHullCandidate,
) -> bool:
    """Check live/digest structure only; never replay upstream certificates.

    This is intentionally not an authority API.  A future consumer must call
    the one-shot builder again with the raw conditional certificates and pair
    bundle before materialization.
    """

    try:
        if (
            type(build) is not OperatorHZBuild
            or type(build.hz) is not SparseHZono
            or type(selection) is not OperatorExactReLUPhaseSelection
            or type(candidate) is not LivePhaseConditionedObjectiveHullCandidate
            or candidate.schema != _SCHEMA
            or candidate.proof_authority is not False
            or candidate.verdict_authority is not False
            or type(candidate.receipt) is not MappingProxyType
            or not _valid_sha256(candidate.candidate_sha256)
        ):
            return False
        stable_ids = _stable_ids(candidate.stable_bit_ids)
        focused_id = _focused_rival_id(candidate.focused_rival_id)
        expected_patterns = _expected_patterns(stable_ids)
        parent = sparse_hz_semantic_digest(build.hz)
        if (
            parent != candidate.parent_semantic_digest
            or parent != candidate.terminal_parent_semantic_digest
            or selection.parent_semantic_digest != parent
            or selection.property_digest != candidate.property_digest
            or selection.selection_digest != candidate.selection_digest
            or selection.operator_row_tag_digest
            != candidate.operator_row_tag_digest
            or candidate.descriptor.stable_bit_ids != stable_ids
            or candidate.descriptor.empty_pattern_evidence
            != candidate.empty_pattern_evidence
            or candidate.descriptor.proof_authority is not False
            or candidate.descriptor.verdict_authority is not False
            or not candidate.descriptor.objective_binding.objective_id.startswith(
                f"rival:{focused_id}:"
            )
            or len(candidate.conditional_certificate_sha256)
            != len(expected_patterns)
            or len(set(candidate.conditional_certificate_sha256))
            != len(expected_patterns)
            or len(candidate.pair_certificate_sha256)
            > _MAX_SIGNED_PAIR_QUERIES
            or any(
                not _valid_sha256(value)
                for value in (
                    candidate.parent_semantic_digest,
                    candidate.terminal_parent_semantic_digest,
                    candidate.property_digest,
                    candidate.selection_digest,
                    candidate.operator_row_tag_digest,
                    candidate.ordered_source_frame_sha256,
                    candidate.conditional_replay_bundle_sha256,
                    candidate.pair_bundle_sha256,
                    *candidate.conditional_certificate_sha256,
                    *candidate.pair_certificate_sha256,
                )
            )
        ):
            return False
        if any(
            not verify_external_certified_empty_pattern(item)
            for item in candidate.empty_pattern_evidence
        ):
            return False
        if not verify_phase_conditioned_objective_hull(
            candidate.descriptor,
            live_parent_semantic_digest=parent,
            live_objective_binding=candidate.descriptor.objective_binding,
        ):
            return False
        expected_receipt = _make_receipt(
            parent_digest=candidate.parent_semantic_digest,
            terminal_parent_digest=candidate.terminal_parent_semantic_digest,
            property_digest=candidate.property_digest,
            selection_digest=candidate.selection_digest,
            operator_row_tag_digest=candidate.operator_row_tag_digest,
            source_frame_digest=candidate.ordered_source_frame_sha256,
            focused_rival_id=focused_id,
            stable_ids=stable_ids,
            conditional_bundle_sha256=(
                candidate.conditional_replay_bundle_sha256
            ),
            conditional_certificate_sha256=(
                candidate.conditional_certificate_sha256
            ),
            pair_bundle_sha256=candidate.pair_bundle_sha256,
            pair_certificate_sha256=candidate.pair_certificate_sha256,
            evidence=candidate.empty_pattern_evidence,
            descriptor=candidate.descriptor,
        )
        return (
            _canonical_form(candidate.receipt)
            == _canonical_form(expected_receipt)
            and _canonical_sha256(
                _candidate_payload(candidate, include_digest=False)
            )
            == candidate.candidate_sha256
        )
    except (
        PhaseConditionedLiveAdapterError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


__all__ = [
    "LivePhaseConditionedObjectiveHullCandidate",
    "PhaseConditionedLiveAdapterError",
    "build_live_phase_conditioned_objective_hull_candidate",
    "verify_live_phase_conditioned_objective_hull_candidate_structure",
]
