#!/usr/bin/env python3
"""Candidate-only signed-support grouping for multi-rival PC-PCC.

Different output rivals need not worsen under the same phase of every
HybridZ binary factor.  This module groups rivals by their complete,
stored-binary64 exact-dyadic signed support over stable ``bcol_ids``.
Zero-effect factors are recorded explicitly as omitted rather than silently
assigned a phase.

Grouping is only a deterministic proposal mechanism.  It proves no cut:
every selected literal pair must still be checked against the complete
parent HZ by an independent exact conflict oracle before a clique inequality
can be emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    ordered_property_digest,
    rival_spec_binding_digest,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    _stable_position_map,
)
from act.back_end.solver.solver_hz import SparseHZono


class PropertyLiteralGroupingError(ValueError):
    """Malformed candidate input; no literal group may be consumed."""


_DEFAULT_MAX_RIVALS = 128
_DEFAULT_MAX_BINARIES = 2048
_DEFAULT_MAX_GROUPS = 128
_DEFAULT_TIMEOUT_SECONDS = 5.0

_HARD_CAPS = {
    "max_rivals": 256,
    "max_binaries": 16384,
    "max_groups": 256,
}
_HARD_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class GroupedPhaseLiteral:
    """One stable signed literal explicitly bound to its rival group."""

    stable_bcol_id: int
    phase: int
    binding_digest: str


@dataclass(frozen=True)
class PropertyLiteralGroup:
    """Rivals with one identical complete signed-support signature."""

    group_id: str
    rivals: Tuple[RivalSpec, ...]
    rival_binding_digests: Tuple[str, ...]
    literals: Tuple[GroupedPhaseLiteral, ...]
    omitted_zero_bcol_ids: Tuple[int, ...]
    eligible_pair_count: int
    proof_authority: bool = False

    @property
    def cut_eligible(self) -> bool:
        return len(self.literals) >= 2


@dataclass(frozen=True)
class PropertyLiteralGroupingResult:
    """All deterministic groups and a structural diagnostic receipt."""

    parent_semantic_digest: str
    ordered_property_digest: str
    groups: Tuple[PropertyLiteralGroup, ...]
    receipt: Mapping[str, Any]
    proof_authority: bool = False


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
    if type(value) is not str or len(value) != 64:
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
        raise PropertyLiteralGroupingError(f"{name}_not_integer")
    return int(value)


def _validate_caps(
    *,
    max_rivals: int,
    max_binaries: int,
    max_groups: int,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    caps = {
        "max_rivals": _strict_int(
            max_rivals, name="max_rivals"
        ),
        "max_binaries": _strict_int(
            max_binaries, name="max_binaries"
        ),
        "max_groups": _strict_int(
            max_groups, name="max_groups"
        ),
    }
    if any(
        value < 1 or value > _HARD_CAPS[name]
        for name, value in caps.items()
    ):
        raise PropertyLiteralGroupingError(
            "grouping_cap_out_of_range"
        )
    if isinstance(timeout_seconds, (bool, np.bool_)):
        raise PropertyLiteralGroupingError(
            "grouping_timeout_not_numeric"
        )
    try:
        normalized_timeout = float(timeout_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PropertyLiteralGroupingError(
            "grouping_timeout_not_numeric"
        ) from exc
    if (
        not math.isfinite(normalized_timeout)
        or normalized_timeout <= 0.0
        or normalized_timeout > _HARD_TIMEOUT_SECONDS
    ):
        raise PropertyLiteralGroupingError(
            "grouping_timeout_out_of_range"
        )
    caps["timeout_seconds"] = normalized_timeout
    return caps


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise PropertyLiteralGroupingError(
            "grouping_deadline_expired"
        )


def _normalized_rivals(
    rivals: Sequence[RivalSpec],
    *,
    output_width: int,
    max_rivals: int,
    deadline: float,
) -> Tuple[RivalSpec, ...]:
    if (
        isinstance(rivals, (str, bytes))
        or not isinstance(rivals, Sequence)
        or not rivals
        or len(rivals) > max_rivals
    ):
        raise PropertyLiteralGroupingError(
            "rival_sequence_or_cap_invalid"
        )
    output = []
    seen_ids = set()
    for rival in rivals:
        _check_deadline(deadline)
        if type(rival) is not RivalSpec:
            raise PropertyLiteralGroupingError(
                "rival_wrong_type"
            )
        rival_id = _strict_int(
            rival.rival_id, name="rival_id"
        )
        try:
            objective = np.asarray(
                rival.objective, dtype=np.float64
            )
            threshold = float(rival.threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PropertyLiteralGroupingError(
                "rival_numeric_conversion_failed"
            ) from exc
        if (
            rival_id < 0
            or rival_id in seen_ids
            or objective.ndim != 1
            or objective.size != output_width
            or not np.all(np.isfinite(objective))
            or not math.isfinite(threshold)
            or not _valid_sha256(rival.assert_digest)
        ):
            raise PropertyLiteralGroupingError(
                "rival_semantics_invalid"
            )
        seen_ids.add(rival_id)
        output.append(
            RivalSpec(
                rival_id=rival_id,
                objective=tuple(
                    float(value) for value in objective.tolist()
                ),
                threshold=threshold,
                assert_digest=rival.assert_digest,
            )
        )
    # This independently rechecks the exact RivalSpec binding contract.
    ordered_property_digest(output)
    _check_deadline(deadline)
    return tuple(output)


def _literal_binding_digest(
    *,
    parent_digest: str,
    group_id: str,
    stable_bcol_id: int,
    phase: int,
) -> str:
    return _canonical_sha256(
        {
            "schema": "act.pc_pcc.grouped_phase_literal.v2",
            "parent_semantic_digest": parent_digest,
            "group_id": group_id,
            "stable_bcol_id": int(stable_bcol_id),
            "phase": int(phase),
        }
    )


def _group_id(
    *,
    rival_bindings: Sequence[str],
    signature: Sequence[Tuple[int, int]],
    omitted_zero_ids: Sequence[int],
) -> str:
    return _canonical_sha256(
        {
            "schema": "act.pc_pcc.rival_signed_support_group.v2",
            "rival_binding_digests": list(rival_bindings),
            "signature": [
                [int(stable_id), int(phase)]
                for stable_id, phase in signature
            ],
            "omitted_zero_bcol_ids": [
                int(stable_id)
                for stable_id in omitted_zero_ids
            ],
        }
    )


def _dyadic_ratio(value: float) -> Tuple[int, int]:
    """Return ``numerator, denominator_power`` for one finite binary64."""

    numerator, denominator = float(value).as_integer_ratio()
    return int(numerator), int(denominator.bit_length() - 1)


def _exact_dyadic_effect_sign(
    objective: Sequence[Tuple[int, int]],
    generator: Sequence[Tuple[int, int, int]],
    *,
    deadline: float,
) -> int:
    """Sign of the exact real dot product of the stored binary64 values."""

    total_numerator = 0
    denominator_power = 0
    initialized = False
    for index, (row, generator_numerator, generator_power) in enumerate(
        generator
    ):
        if index % 64 == 0:
            _check_deadline(deadline)
        objective_numerator, objective_power = objective[row]
        product_numerator = (
            objective_numerator * generator_numerator
        )
        if product_numerator == 0:
            continue
        product_power = objective_power + generator_power
        if not initialized:
            total_numerator = product_numerator
            denominator_power = product_power
            initialized = True
        elif product_power > denominator_power:
            total_numerator = (
                total_numerator
                << (product_power - denominator_power)
            ) + product_numerator
            denominator_power = product_power
        else:
            total_numerator += product_numerator << (
                denominator_power - product_power
            )
    return (
        1
        if total_numerator > 0
        else -1
        if total_numerator < 0
        else 0
    )


def _sparse_dyadic_generators(
    hz: SparseHZono,
    stable_ids: Sequence[int],
    positions: Mapping[int, int],
    *,
    deadline: float,
) -> Tuple[Tuple[Tuple[int, int, int], ...], ...]:
    """Read each physical column once and retain exact nonzero dyadics."""

    _check_deadline(deadline)
    matrix = hz.Gb.tocsc(copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if (
        matrix.shape != (hz.n_out, hz.n_bin)
        or not np.all(np.isfinite(matrix.data))
    ):
        raise PropertyLiteralGroupingError(
            "binary_generator_nonfinite_or_wrong_shape"
        )
    generators = []
    for index, stable_id in enumerate(stable_ids):
        if index % 64 == 0:
            _check_deadline(deadline)
        position = positions[stable_id]
        start = int(matrix.indptr[position])
        stop = int(matrix.indptr[position + 1])
        terms = []
        for row, raw_value in zip(
            matrix.indices[start:stop],
            matrix.data[start:stop],
        ):
            value = float(raw_value)
            if value == 0.0:
                continue
            numerator, power = _dyadic_ratio(value)
            terms.append((int(row), numerator, power))
        generators.append(tuple(terms))
    _check_deadline(deadline)
    return tuple(generators)


def _derive_groups(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    *,
    parent_digest: str,
    caps: Mapping[str, Any],
    deadline: float,
) -> Tuple[PropertyLiteralGroup, ...]:
    _check_deadline(deadline)
    positions = _stable_position_map(hz)
    if hz.n_bin > caps["max_binaries"]:
        raise PropertyLiteralGroupingError(
            "binary_cap_exceeded"
        )
    normalized = _normalized_rivals(
        rivals,
        output_width=hz.n_out,
        max_rivals=caps["max_rivals"],
        deadline=deadline,
    )
    stable_ids = tuple(sorted(positions))
    generators = _sparse_dyadic_generators(
        hz,
        stable_ids,
        positions,
        deadline=deadline,
    )

    buckets: dict[
        Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]],
        list[RivalSpec],
    ] = {}
    for rival in normalized:
        _check_deadline(deadline)
        objective = tuple(
            _dyadic_ratio(value) for value in rival.objective
        )
        signature = []
        omitted = []
        for binary_index, (stable_id, generator) in enumerate(
            zip(stable_ids, generators)
        ):
            if binary_index % 32 == 0:
                _check_deadline(deadline)
            phase = _exact_dyadic_effect_sign(
                objective,
                generator,
                deadline=deadline,
            )
            if phase > 0:
                signature.append((stable_id, 1))
            elif phase < 0:
                signature.append((stable_id, -1))
            else:
                omitted.append(stable_id)
        key = (tuple(signature), tuple(omitted))
        buckets.setdefault(key, []).append(rival)
    if len(buckets) > caps["max_groups"]:
        raise PropertyLiteralGroupingError("group_cap_exceeded")

    groups = []
    for signature, omitted in sorted(buckets):
        _check_deadline(deadline)
        members = tuple(
            sorted(
                buckets[(signature, omitted)],
                key=lambda rival: (
                    int(rival.rival_id),
                    rival_spec_binding_digest(rival),
                ),
            )
        )
        rival_bindings = tuple(
            rival_spec_binding_digest(rival)
            for rival in members
        )
        group_id = _group_id(
            rival_bindings=rival_bindings,
            signature=signature,
            omitted_zero_ids=omitted,
        )
        literals = tuple(
            GroupedPhaseLiteral(
                stable_bcol_id=stable_id,
                phase=phase,
                binding_digest=_literal_binding_digest(
                    parent_digest=parent_digest,
                    group_id=group_id,
                    stable_bcol_id=stable_id,
                    phase=phase,
                ),
            )
            for stable_id, phase in signature
        )
        groups.append(
            PropertyLiteralGroup(
                group_id=group_id,
                rivals=members,
                rival_binding_digests=rival_bindings,
                literals=literals,
                omitted_zero_bcol_ids=omitted,
                eligible_pair_count=(
                    len(literals) * (len(literals) - 1) // 2
                ),
            )
        )
    _check_deadline(deadline)
    return tuple(groups)


def _exact_json_primitive_copy(
    value: Any,
    *,
    deadline: float,
    remaining_nodes: list[int],
    depth: int = 0,
) -> Any:
    """Copy only exact JSON primitive/container types.

    This deliberately rejects subclasses with attacker-controlled equality.
    """

    _check_deadline(deadline)
    remaining_nodes[0] -= 1
    if remaining_nodes[0] < 0 or depth > 64:
        raise PropertyLiteralGroupingError(
            "receipt_structure_cap_exceeded"
        )
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PropertyLiteralGroupingError(
                "receipt_nonfinite_float"
            )
        return value
    if type(value) is list:
        return [
            _exact_json_primitive_copy(
                item,
                deadline=deadline,
                remaining_nodes=remaining_nodes,
                depth=depth + 1,
            )
            for item in value
        ]
    if type(value) is dict:
        copied = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PropertyLiteralGroupingError(
                    "receipt_key_wrong_type"
                )
            copied[key] = _exact_json_primitive_copy(
                item,
                deadline=deadline,
                remaining_nodes=remaining_nodes,
                depth=depth + 1,
            )
        return copied
    raise PropertyLiteralGroupingError(
        "receipt_value_wrong_type"
    )


def _exact_rival_payload(
    rival: Any,
    *,
    output_width: int,
) -> Mapping[str, Any]:
    if (
        type(rival) is not RivalSpec
        or type(rival.rival_id) is not int
        or rival.rival_id < 0
        or type(rival.objective) is not tuple
        or len(rival.objective) != output_width
        or any(type(value) is not float for value in rival.objective)
        or not all(math.isfinite(value) for value in rival.objective)
        or type(rival.threshold) is not float
        or not math.isfinite(rival.threshold)
        or not _valid_sha256(rival.assert_digest)
    ):
        raise PropertyLiteralGroupingError(
            "result_rival_wrong_type_or_semantics"
        )
    return {
        "rival_id": rival.rival_id,
        "objective_f64_hex": [
            value.hex() for value in rival.objective
        ],
        "threshold_f64_hex": rival.threshold.hex(),
        "assert_digest": rival.assert_digest,
    }


def _exact_group_payload(
    groups: Any,
    *,
    parent_digest: str,
    stable_ids: Sequence[int],
    rival_ids: Sequence[int],
    output_width: int,
    max_groups: int,
    deadline: float,
) -> Any:
    """Validate and rebuild groups using only exact built-in primitives."""

    if (
        type(groups) is not tuple
        or not groups
        or len(groups) > max_groups
    ):
        raise PropertyLiteralGroupingError(
            "result_groups_wrong_type_or_empty"
        )
    stable_id_tuple = tuple(stable_ids)
    stable_id_set = set(stable_id_tuple)
    seen_rival_ids = []
    payload = []
    for group in groups:
        _check_deadline(deadline)
        if (
            type(group) is not PropertyLiteralGroup
            or group.proof_authority is not False
            or not _valid_sha256(group.group_id)
            or type(group.rivals) is not tuple
            or not group.rivals
            or len(group.rivals) > len(rival_ids)
            or type(group.rival_binding_digests) is not tuple
            or type(group.literals) is not tuple
            or len(group.literals) > len(stable_id_tuple)
            or type(group.omitted_zero_bcol_ids) is not tuple
            or len(group.omitted_zero_bcol_ids)
            > len(stable_id_tuple)
            or type(group.eligible_pair_count) is not int
        ):
            raise PropertyLiteralGroupingError(
                "result_group_wrong_type_or_shape"
            )

        rival_payloads = []
        computed_bindings = []
        member_sort_keys = []
        for rival in group.rivals:
            _check_deadline(deadline)
            rival_payloads.append(
                _exact_rival_payload(
                    rival,
                    output_width=output_width,
                )
            )
            binding = rival_spec_binding_digest(rival)
            computed_bindings.append(binding)
            member_sort_keys.append((rival.rival_id, binding))
            seen_rival_ids.append(rival.rival_id)
        if (
            len(group.rival_binding_digests)
            != len(computed_bindings)
            or any(
                not _valid_sha256(binding)
                for binding in group.rival_binding_digests
            )
            or tuple(computed_bindings)
            != group.rival_binding_digests
            or tuple(member_sort_keys)
            != tuple(sorted(member_sort_keys))
        ):
            raise PropertyLiteralGroupingError(
                "result_rival_binding_mismatch"
            )

        signature = []
        literal_payloads = []
        for literal in group.literals:
            _check_deadline(deadline)
            if (
                type(literal) is not GroupedPhaseLiteral
                or type(literal.stable_bcol_id) is not int
                or literal.stable_bcol_id < 0
                or type(literal.phase) is not int
                or literal.phase not in {-1, 1}
                or not _valid_sha256(literal.binding_digest)
            ):
                raise PropertyLiteralGroupingError(
                    "result_literal_wrong_type_or_semantics"
                )
            expected_binding = _literal_binding_digest(
                parent_digest=parent_digest,
                group_id=group.group_id,
                stable_bcol_id=literal.stable_bcol_id,
                phase=literal.phase,
            )
            if literal.binding_digest != expected_binding:
                raise PropertyLiteralGroupingError(
                    "result_literal_binding_mismatch"
                )
            signature.append(
                (literal.stable_bcol_id, literal.phase)
            )
            literal_payloads.append(
                {
                    "stable_bcol_id": literal.stable_bcol_id,
                    "phase": literal.phase,
                    "binding_digest": literal.binding_digest,
                }
            )
        literal_ids = tuple(
            stable_id for stable_id, _ in signature
        )
        if (
            literal_ids != tuple(sorted(literal_ids))
            or len(set(literal_ids)) != len(literal_ids)
        ):
            raise PropertyLiteralGroupingError(
                "result_literal_ids_not_canonical"
            )

        omitted = group.omitted_zero_bcol_ids
        if (
            any(
                type(stable_id) is not int or stable_id < 0
                for stable_id in omitted
            )
            or omitted != tuple(sorted(omitted))
            or len(set(omitted)) != len(omitted)
            or set(literal_ids) & set(omitted)
            or set(literal_ids) | set(omitted) != stable_id_set
            or len(literal_ids) + len(omitted)
            != len(stable_id_tuple)
        ):
            raise PropertyLiteralGroupingError(
                "result_stable_id_partition_invalid"
            )
        eligible_pair_count = (
            len(signature) * (len(signature) - 1) // 2
        )
        if group.eligible_pair_count != eligible_pair_count:
            raise PropertyLiteralGroupingError(
                "result_eligible_pair_count_invalid"
            )
        expected_group_id = _group_id(
            rival_bindings=computed_bindings,
            signature=signature,
            omitted_zero_ids=omitted,
        )
        if group.group_id != expected_group_id:
            raise PropertyLiteralGroupingError(
                "result_group_id_mismatch"
            )
        payload.append(
            {
                "group_id": group.group_id,
                "rivals": rival_payloads,
                "rival_binding_digests": computed_bindings,
                "literals": literal_payloads,
                "omitted_zero_bcol_ids": list(omitted),
                "eligible_pair_count": eligible_pair_count,
                "proof_authority": False,
            }
        )
    _check_deadline(deadline)
    if (
        len(set(seen_rival_ids)) != len(seen_rival_ids)
        or tuple(sorted(seen_rival_ids))
        != tuple(sorted(rival_ids))
    ):
        raise PropertyLiteralGroupingError(
            "result_rival_partition_invalid"
        )
    return payload


def _receipt(
    *,
    parent_digest: str,
    property_digest: str,
    groups: Sequence[PropertyLiteralGroup],
    caps: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = {
        "schema": "act.pc_pcc.property_literal_groups.v2",
        "proof_authority": False,
        "role": "candidate_selection_only",
        "receipt_integrity": (
            "unkeyed_sha256_diagnostic_not_authentication"
        ),
        "caps_binding": (
            "caller_expected_caps_required_by_verifier"
        ),
        "invocation_provenance": (
            "not_provided_by_serialized_receipt"
        ),
        "effect_arithmetic": (
            "stored_f64_exact_dyadic_integer_accumulation"
        ),
        "deadline_policy": "monotonic_fail_closed_loop_checks",
        "parent_semantic_digest": parent_digest,
        "ordered_property_digest": property_digest,
        "caps": dict(caps),
        "group_count": len(groups),
        "cut_eligible_group_count": sum(
            group.cut_eligible for group in groups
        ),
        "groups": [
            {
                "group_id": group.group_id,
                "rival_binding_digests": list(
                    group.rival_binding_digests
                ),
                "literals": [
                    {
                        "stable_bcol_id": literal.stable_bcol_id,
                        "phase": literal.phase,
                        "binding_digest": (
                            literal.binding_digest
                        ),
                    }
                    for literal in group.literals
                ],
                "omitted_zero_bcol_ids": list(
                    group.omitted_zero_bcol_ids
                ),
                "eligible_pair_count": (
                    group.eligible_pair_count
                ),
            }
            for group in groups
        ],
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload


def derive_property_literal_groups_candidate(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    *,
    max_rivals: int = _DEFAULT_MAX_RIVALS,
    max_binaries: int = _DEFAULT_MAX_BINARIES,
    max_groups: int = _DEFAULT_MAX_GROUPS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> PropertyLiteralGroupingResult:
    """Derive deterministic explicit literal subsets; prove no cut."""

    if not isinstance(hz, SparseHZono):
        raise PropertyLiteralGroupingError("parent_not_sparse_hz")
    caps = _validate_caps(
        max_rivals=max_rivals,
        max_binaries=max_binaries,
        max_groups=max_groups,
        timeout_seconds=timeout_seconds,
    )
    if hz.n_bin > caps["max_binaries"]:
        raise PropertyLiteralGroupingError(
            "binary_cap_exceeded"
        )
    deadline = time.monotonic() + caps["timeout_seconds"]
    try:
        normalized = _normalized_rivals(
            rivals,
            output_width=hz.n_out,
            max_rivals=caps["max_rivals"],
            deadline=deadline,
        )
        _check_deadline(deadline)
        parent_digest = sparse_hz_semantic_digest(hz)
        _check_deadline(deadline)
        property_digest = ordered_property_digest(normalized)
        groups = _derive_groups(
            hz,
            normalized,
            parent_digest=parent_digest,
            caps=caps,
            deadline=deadline,
        )
        _check_deadline(deadline)
    except PropertyLiteralGroupingError:
        raise
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ) as exc:
        raise PropertyLiteralGroupingError(
            "grouping_parent_or_property_invalid"
        ) from exc
    result = PropertyLiteralGroupingResult(
        parent_semantic_digest=parent_digest,
        ordered_property_digest=property_digest,
        groups=groups,
        receipt=_receipt(
            parent_digest=parent_digest,
            property_digest=property_digest,
            groups=groups,
            caps=caps,
        ),
    )
    if not verify_property_literal_grouping_result(
        hz,
        rivals,
        result,
        max_rivals=caps["max_rivals"],
        max_binaries=caps["max_binaries"],
        max_groups=caps["max_groups"],
        timeout_seconds=caps["timeout_seconds"],
    ):
        raise PropertyLiteralGroupingError(
            "grouping_self_audit_failed"
        )
    return result


def verify_property_literal_grouping_result(
    hz: SparseHZono,
    rivals: Sequence[RivalSpec],
    result: PropertyLiteralGroupingResult,
    *,
    max_rivals: int = _DEFAULT_MAX_RIVALS,
    max_binaries: int = _DEFAULT_MAX_BINARIES,
    max_groups: int = _DEFAULT_MAX_GROUPS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Repeat derivation with caller-expected caps; prove no run provenance."""

    try:
        expected_caps = _validate_caps(
            max_rivals=max_rivals,
            max_binaries=max_binaries,
            max_groups=max_groups,
            timeout_seconds=timeout_seconds,
        )
        deadline = time.monotonic() + expected_caps[
            "timeout_seconds"
        ]
        if (
            not isinstance(hz, SparseHZono)
            or type(result) is not PropertyLiteralGroupingResult
            or result.proof_authority is not False
            or type(result.groups) is not tuple
            or type(result.receipt) is not dict
            or hz.n_bin > expected_caps["max_binaries"]
            or not _valid_sha256(result.parent_semantic_digest)
            or not _valid_sha256(result.ordered_property_digest)
        ):
            return False
        normalized = _normalized_rivals(
            rivals,
            output_width=hz.n_out,
            max_rivals=expected_caps["max_rivals"],
            deadline=deadline,
        )
        _check_deadline(deadline)
        live_parent_digest = sparse_hz_semantic_digest(hz)
        _check_deadline(deadline)
        live_property_digest = ordered_property_digest(normalized)
        if (
            result.parent_semantic_digest != live_parent_digest
            or result.ordered_property_digest
            != live_property_digest
        ):
            return False

        receipt_node_budget = 1024 + expected_caps[
            "max_groups"
        ] * (
            32
            + 8 * int(hz.n_bin)
            + 8 * len(normalized)
        )
        receipt = _exact_json_primitive_copy(
            result.receipt,
            deadline=deadline,
            remaining_nodes=[receipt_node_budget],
        )
        receipt_digest = receipt.get("receipt_sha256")
        if not _valid_sha256(receipt_digest):
            return False
        payload = dict(receipt)
        del payload["receipt_sha256"]
        if _canonical_sha256(payload) != receipt_digest:
            return False
        caps = receipt.get("caps")
        if (
            type(caps) is not dict
            or set(caps)
            != {
                "max_rivals",
                "max_binaries",
                "max_groups",
                "timeout_seconds",
            }
        ):
            return False
        validated_caps = _validate_caps(**caps)
        if _canonical_bytes(validated_caps) != _canonical_bytes(
            expected_caps
        ):
            return False
        positions = _stable_position_map(hz)
        stable_ids = tuple(sorted(positions))
        actual_group_payload = _exact_group_payload(
            result.groups,
            parent_digest=result.parent_semantic_digest,
            stable_ids=stable_ids,
            rival_ids=tuple(
                rival.rival_id for rival in normalized
            ),
            output_width=hz.n_out,
            max_groups=expected_caps["max_groups"],
            deadline=deadline,
        )
        expected_groups = _derive_groups(
            hz,
            normalized,
            parent_digest=result.parent_semantic_digest,
            caps=expected_caps,
            deadline=deadline,
        )
        expected_group_payload = _exact_group_payload(
            expected_groups,
            parent_digest=result.parent_semantic_digest,
            stable_ids=stable_ids,
            rival_ids=tuple(
                rival.rival_id for rival in normalized
            ),
            output_width=hz.n_out,
            max_groups=expected_caps["max_groups"],
            deadline=deadline,
        )
        expected_receipt = _receipt(
            parent_digest=result.parent_semantic_digest,
            property_digest=result.ordered_property_digest,
            groups=expected_groups,
            caps=expected_caps,
        )
        _check_deadline(deadline)
        return (
            _canonical_bytes(actual_group_payload)
            == _canonical_bytes(expected_group_payload)
            and _canonical_bytes(receipt)
            == _canonical_bytes(expected_receipt)
        )
    except (
        PropertyLiteralGroupingError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


__all__ = [
    "GroupedPhaseLiteral",
    "PropertyLiteralGroup",
    "PropertyLiteralGroupingError",
    "PropertyLiteralGroupingResult",
    "derive_property_literal_groups_candidate",
    "verify_property_literal_grouping_result",
]
