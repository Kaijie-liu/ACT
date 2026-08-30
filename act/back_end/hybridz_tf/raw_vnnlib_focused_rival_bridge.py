#!/usr/bin/env python3
"""Strict candidate-only raw TOP1 full-batch to focused-rival bridge.

Large-class TOP1 properties commonly contain 99 or 199 ordered rivals.  The
exact Operator-HZ phase selector can deliberately focus on a strict subset,
but that subset must not be chosen from ground truth, detached row numbers,
or an unbound array of floating-point scores.

This module implements a narrow two-stage boundary:

1. A caller supplies the *complete* live interval-upper-violation vector as
   canonical exact dyadics ``(numerator, denominator)`` plus a digest of the
   live output-interval snapshot from which it was computed.  Issuance binds
   every vector position to the owner-bound consumed raw batch, encoded row,
   competitor/rival ID, exact ``RivalSpec`` binding, full live ASSERT digest,
   and full ordered-property digest.
2. A deterministic exact ranking selects one rival by default, or an
   explicitly pre-registered subset of at most four rivals.  Ties are broken
   by encoded row and stable rival ID.  The selected ``RivalSpec`` objects are
   the exact objects owned by the consumed batch and can be passed directly
   to ``operator_exact_relu_phase_literals`` and
   ``operator_exact_relu_phase_cliques``.

The hardness values are scheduling inputs only.  They carry no bound or
verdict authority, and this bridge does not assert that a caller-supplied
interval digest came from a particular bound engine.  The intended production
call point is immediately after the live output lower/upper arrays and the
standard interval violation vector are computed in
``property_residual_targets.select_property_residual_targets`` (or its sparse
query sibling).  Convert each finite binary64 hardness with
``float(value).as_integer_ratio()`` and bind a canonical digest of the exact
lower/upper snapshots.

All public artifacts are immutable, candidate-only, ``proof_authority=False``
objects.  Verification reconstructs trusted primitive payloads and never
invokes candidate-controlled dataclass equality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import os
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    ordered_property_digest,
    rival_spec_binding_digest,
)
from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
    ConsumedRivalBatch,
    validate_consumed_raw_vnnlib_rival_batch,
)


_HARDNESS_SCHEMA = (
    "act.raw_vnnlib_focused_rival_exact_hardness.v1"
)
_FOCUSED_SCHEMA = "act.raw_vnnlib_focused_rival_selection.v1"
_FULL_BINDING_SCHEMA = (
    "act.raw_vnnlib_full_live_assert_property_binding.v1"
)
_METHOD = (
    "caller_bound_binary64_interval_upper_violation_"
    "exact_dyadic_v1"
)
_RESIDUAL_PROPERTY_KIND = "TOP1_ROBUST"
_HARDEST_FOCUS_METHOD = (
    "exact_hardness_descending_default_singleton_or_subset_v1"
)
_RESIDUAL_JOINT_FOCUS_METHOD = (
    "caller_bound_residual_joint_focus_encoded_row_v1"
)

_DEFAULT_MAX_RIVALS = 512
_DEFAULT_MAX_FOCUS = 4
_DEFAULT_MAX_EXACT_BITS = 4096
_DEFAULT_MAX_WORK_ITEMS = 5_000_000

_HARD_MAX_RIVALS = 10_000
_HARD_MAX_FOCUS = 4
_HARD_MAX_EXACT_BITS = 65_536
_HARD_MAX_WORK_ITEMS = 100_000_000
_MAX_INT64 = (1 << 63) - 1


class RawFocusedRivalBridgeError(ValueError):
    """Fail-closed focused-rival bridge rejection."""


@dataclass(frozen=True, eq=False)
class RawFocusedRivalCaps:
    """Caller-bound limits shared by issuance, selection, and replay."""

    max_rivals: int
    max_focus: int
    max_exact_bits: int
    max_work_items: int


@dataclass(frozen=True, eq=False)
class ExactRawRivalHardness:
    """One exact vector position bound to its raw rival identity."""

    encoded_row: int
    competitor_class: int
    rival_id: int
    rival_spec_binding_digest: str
    upper_numerator: int
    upper_denominator: int

    @property
    def upper(self) -> Fraction:
        return Fraction(self.upper_numerator, self.upper_denominator)


@dataclass(frozen=True, eq=False)
class RawRivalExactHardnessReceipt:
    """Immutable owner-bound receipt for the complete hardness vector."""

    full_batch_sha256: str
    full_live_assert_sha256: str
    full_property_digest: str
    full_live_assert_property_digest: str
    live_interval_bounds_sha256: str
    method: str
    shape: Tuple[int, ...]
    entries: Tuple[ExactRawRivalHardness, ...]
    caps: RawFocusedRivalCaps
    vector_digest: str
    receipt_sha256: str
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    _batch_owner: object = field(
        default=None,
        repr=False,
        compare=False,
    )
    _process_id: int = field(
        default=0,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, eq=False)
class RankedRawRivalHardness:
    """One deterministic exact rank over the complete raw batch."""

    rank: int
    encoded_row: int
    competitor_class: int
    rival_id: int
    rival_spec_binding_digest: str
    upper_numerator: int
    upper_denominator: int

    @property
    def upper(self) -> Fraction:
        return Fraction(self.upper_numerator, self.upper_denominator)


@dataclass(frozen=True, eq=False)
class RawFocusedRivalSelection:
    """Focused live ``RivalSpec`` subset plus complete ranking evidence."""

    status: str
    full_batch_sha256: str
    full_live_assert_sha256: str
    full_property_digest: str
    full_live_assert_property_digest: str
    live_interval_bounds_sha256: str
    hardness_vector_digest: str
    hardness_method: str
    method: str
    focus_count: int
    explicit_encoded_focus_row: int | None
    residual_selector_property_sha256: str | None
    residual_selector_receipt_sha256: str | None
    residual_joint_focus_rival_id: int | None
    residual_selector_receipt: Mapping[str, Any] | None
    ranked_entries: Tuple[RankedRawRivalHardness, ...]
    focused_entries: Tuple[RankedRawRivalHardness, ...]
    focused_rivals: Tuple[RivalSpec, ...]
    caps: RawFocusedRivalCaps
    focused_subset_digest: str
    receipt_sha256: str
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    _batch_owner: object = field(
        default=None,
        repr=False,
        compare=False,
    )
    _hardness_owner: object = field(
        default=None,
        repr=False,
        compare=False,
    )
    _process_id: int = field(
        default=0,
        repr=False,
        compare=False,
    )

    @property
    def rivals(self) -> Tuple[RivalSpec, ...]:
        """Alias used by the existing Operator-HZ focused APIs."""

        return self.focused_rivals


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


def _deep_freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {
                key: _deep_freeze(item)
                for key, item in value.items()
            }
        )
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    if type(value) is tuple:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _frozen_exact_match(
    actual: Any,
    expected: Any,
    *,
    deadline: float,
) -> bool:
    if time.monotonic() >= deadline:
        return False
    if type(expected) is dict:
        if (
            type(actual) is not MappingProxyType
            or len(actual) != len(expected)
        ):
            return False
        actual_keys = tuple(actual.keys())
        if (
            time.monotonic() >= deadline
            or len(actual_keys) != len(expected)
            or any(type(key) is not str for key in actual_keys)
            or set(actual_keys) != set(expected)
        ):
            return False
        return all(
            _frozen_exact_match(
                actual[key],
                expected[key],
                deadline=deadline,
            )
            for key in expected
        )
    if type(expected) in {list, tuple}:
        if type(actual) is not tuple or len(actual) != len(expected):
            return False
        return all(
            _frozen_exact_match(
                item,
                expected_item,
                deadline=deadline,
            )
            for item, expected_item in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _normalize_deadline(deadline: Any) -> float:
    if (
        type(deadline) not in {int, float}
        or type(deadline) is bool
    ):
        raise RawFocusedRivalBridgeError(
            "focused_rival_deadline_not_builtin_numeric"
        )
    normalized = float(deadline)
    if not math.isfinite(normalized):
        raise RawFocusedRivalBridgeError(
            "focused_rival_deadline_not_finite"
        )
    return normalized


def _check_deadline(deadline: float, *, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise RawFocusedRivalBridgeError(
            f"focused_rival_deadline_expired_{stage}"
        )


def _strict_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise RawFocusedRivalBridgeError(f"{name}_not_builtin_int")
    return value


def _normalize_caps(
    *,
    max_rivals: int,
    max_focus: int,
    max_exact_bits: int,
    max_work_items: int,
) -> RawFocusedRivalCaps:
    rivals = _strict_int(max_rivals, name="max_rivals")
    focus = _strict_int(max_focus, name="max_focus")
    exact_bits = _strict_int(
        max_exact_bits, name="max_exact_bits"
    )
    work_items = _strict_int(
        max_work_items, name="max_work_items"
    )
    if rivals < 1 or rivals > _HARD_MAX_RIVALS:
        raise RawFocusedRivalBridgeError(
            "max_rivals_out_of_range"
        )
    if focus < 1 or focus > _HARD_MAX_FOCUS:
        raise RawFocusedRivalBridgeError(
            "max_focus_out_of_range"
        )
    if exact_bits < 1 or exact_bits > _HARD_MAX_EXACT_BITS:
        raise RawFocusedRivalBridgeError(
            "max_exact_bits_out_of_range"
        )
    if (
        work_items < 1
        or work_items > _HARD_MAX_WORK_ITEMS
    ):
        raise RawFocusedRivalBridgeError(
            "max_work_items_out_of_range"
        )
    return RawFocusedRivalCaps(
        max_rivals=rivals,
        max_focus=focus,
        max_exact_bits=exact_bits,
        max_work_items=work_items,
    )


def _caps_payload(caps: Any) -> Mapping[str, int]:
    if (
        type(caps) is not RawFocusedRivalCaps
        or type(caps.max_rivals) is not int
        or type(caps.max_focus) is not int
        or type(caps.max_exact_bits) is not int
        or type(caps.max_work_items) is not int
    ):
        raise RawFocusedRivalBridgeError(
            "focused_rival_caps_noncanonical"
        )
    normalized = _normalize_caps(
        max_rivals=caps.max_rivals,
        max_focus=caps.max_focus,
        max_exact_bits=caps.max_exact_bits,
        max_work_items=caps.max_work_items,
    )
    return {
        "max_rivals": normalized.max_rivals,
        "max_focus": normalized.max_focus,
        "max_exact_bits": normalized.max_exact_bits,
        "max_work_items": normalized.max_work_items,
    }


def _same_caps(
    actual: Any, expected: RawFocusedRivalCaps
) -> bool:
    try:
        return _canonical_bytes(_caps_payload(actual)) == (
            _canonical_bytes(_caps_payload(expected))
        )
    except (
        RawFocusedRivalBridgeError,
        AttributeError,
        TypeError,
        ValueError,
    ):
        return False


def _consume_work(
    work: list[int],
    amount: int,
    *,
    caps: RawFocusedRivalCaps,
    deadline: float,
    stage: str,
) -> None:
    if type(amount) is not int or amount < 0:
        raise RawFocusedRivalBridgeError(
            "focused_rival_work_accounting_invalid"
        )
    work[0] += amount
    if work[0] > caps.max_work_items:
        raise RawFocusedRivalBridgeError(
            f"focused_rival_work_cap_exceeded_{stage}"
        )
    _check_deadline(deadline, stage=stage)


def _live_rivals(
    batch: Any,
    *,
    caps: RawFocusedRivalCaps,
    deadline: float,
    work: list[int],
) -> Tuple[RivalSpec, ...]:
    _check_deadline(deadline, stage="before_batch_validation")
    if type(batch) is not ConsumedRivalBatch:
        raise RawFocusedRivalBridgeError(
            "consumed_raw_batch_owner_invalid"
        )
    # Inspect only the exact dataclass storage before invoking the complete
    # owner/content validator.  This makes a caller's small ``max_rivals``
    # cap an O(1) precondition rather than paying to hash an out-of-scope
    # full batch first.
    rivals = vars(batch).get("_rivals")
    if (
        type(rivals) is not tuple
        or not rivals
        or len(rivals) > caps.max_rivals
    ):
        raise RawFocusedRivalBridgeError(
            "consumed_raw_batch_rival_cap_or_type_invalid"
        )
    if any(type(rival) is not RivalSpec for rival in rivals):
        raise RawFocusedRivalBridgeError(
            "consumed_raw_batch_rival_cap_or_type_invalid"
        )
    _check_deadline(deadline, stage="before_full_owner_validation")
    if not validate_consumed_raw_vnnlib_rival_batch(batch):
        raise RawFocusedRivalBridgeError(
            "consumed_raw_batch_owner_invalid"
        )
    _check_deadline(deadline, stage="after_full_owner_validation")
    output_width = len(rivals[0].objective)
    _consume_work(
        work,
        len(rivals) * (output_width + 1),
        caps=caps,
        deadline=deadline,
        stage="full_batch_binding",
    )
    return rivals


def _full_binding_digest(
    *,
    batch_sha256: str,
    live_assert_sha256: str,
    property_digest: str,
) -> str:
    if not (
        _valid_sha256(batch_sha256)
        and _valid_sha256(live_assert_sha256)
        and _valid_sha256(property_digest)
    ):
        raise RawFocusedRivalBridgeError(
            "full_live_assert_property_binding_malformed"
        )
    return _canonical_sha256(
        {
            "schema": _FULL_BINDING_SCHEMA,
            "full_batch_sha256": batch_sha256,
            "full_live_assert_sha256": live_assert_sha256,
            "full_property_digest": property_digest,
        }
    )


def _residual_selector_property_digest(
    rivals: Tuple[RivalSpec, ...],
    *,
    deadline: float,
) -> str:
    """Reproduce ``property_residual_targets._binary64_sha256`` exactly."""

    _check_deadline(
        deadline, stage="before_residual_property_digest"
    )
    C = np.ascontiguousarray(
        tuple(rival.objective for rival in rivals),
        dtype=np.float64,
    )
    thresholds = np.ascontiguousarray(
        tuple(rival.threshold for rival in rivals),
        dtype=np.float64,
    )
    if (
        C.ndim != 2
        or thresholds.ndim != 1
        or C.shape[0] != len(rivals)
        or thresholds.size != len(rivals)
        or not np.all(np.isfinite(C))
        or not np.all(np.isfinite(thresholds))
    ):
        raise RawFocusedRivalBridgeError(
            "full_raw_property_binary64_snapshot_invalid"
        )
    digest = hashlib.sha256()
    for value in (C, thresholds):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(
            np.asarray(array.shape, dtype=np.int64).tobytes()
        )
        digest.update(array.tobytes())
    digest.update(_RESIDUAL_PROPERTY_KIND.encode("utf-8"))
    _check_deadline(
        deadline, stage="after_residual_property_digest"
    )
    return digest.hexdigest()


def _canonical_dyadic_pair(
    value: Any,
    *,
    max_exact_bits: int,
    encoded_row: int,
) -> Tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise RawFocusedRivalBridgeError(
            f"hardness_entry_{encoded_row}_not_builtin_int_pair"
        )
    numerator, denominator = value
    if (
        denominator <= 0
        or (denominator & (denominator - 1)) != 0
        or math.gcd(abs(numerator), denominator) != 1
        or (numerator == 0 and denominator != 1)
        or numerator.bit_length() > max_exact_bits
        or denominator.bit_length() > max_exact_bits
    ):
        raise RawFocusedRivalBridgeError(
            f"hardness_entry_{encoded_row}_noncanonical_or_too_large"
        )
    return numerator, denominator


def _entry_payload(entry: Any) -> Mapping[str, Any]:
    if (
        type(entry) is not ExactRawRivalHardness
        or type(entry.encoded_row) is not int
        or type(entry.competitor_class) is not int
        or type(entry.rival_id) is not int
        or type(entry.rival_spec_binding_digest) is not str
        or type(entry.upper_numerator) is not int
        or type(entry.upper_denominator) is not int
    ):
        raise RawFocusedRivalBridgeError(
            "hardness_entry_runtime_type_invalid"
        )
    return {
        "encoded_row": entry.encoded_row,
        "competitor_class": entry.competitor_class,
        "rival_id": entry.rival_id,
        "rival_spec_binding_digest": (
            entry.rival_spec_binding_digest
        ),
        "upper_numerator": entry.upper_numerator,
        "upper_denominator": entry.upper_denominator,
    }


def _ranked_payload(entry: Any) -> Mapping[str, Any]:
    if (
        type(entry) is not RankedRawRivalHardness
        or type(entry.rank) is not int
        or type(entry.encoded_row) is not int
        or type(entry.competitor_class) is not int
        or type(entry.rival_id) is not int
        or type(entry.rival_spec_binding_digest) is not str
        or type(entry.upper_numerator) is not int
        or type(entry.upper_denominator) is not int
    ):
        raise RawFocusedRivalBridgeError(
            "ranked_hardness_runtime_type_invalid"
        )
    return {
        "rank": entry.rank,
        "encoded_row": entry.encoded_row,
        "competitor_class": entry.competitor_class,
        "rival_id": entry.rival_id,
        "rival_spec_binding_digest": (
            entry.rival_spec_binding_digest
        ),
        "upper_numerator": entry.upper_numerator,
        "upper_denominator": entry.upper_denominator,
    }


def _trusted_entries(
    rivals: Tuple[RivalSpec, ...],
    exact_upper_violations: Any,
    *,
    caps: RawFocusedRivalCaps,
    deadline: float,
    work: list[int],
) -> Tuple[ExactRawRivalHardness, ...]:
    if (
        type(exact_upper_violations) is not tuple
        or len(exact_upper_violations) != len(rivals)
    ):
        raise RawFocusedRivalBridgeError(
            "hardness_vector_shape_or_container_invalid"
        )
    entries = []
    seen_rival_ids = set()
    for encoded_row, (rival, raw_value) in enumerate(
        zip(rivals, exact_upper_violations)
    ):
        _consume_work(
            work,
            1,
            caps=caps,
            deadline=deadline,
            stage="hardness_vector_entry",
        )
        if (
            type(rival.rival_id) is not int
            or rival.rival_id < 0
            or rival.rival_id > _MAX_INT64
            or rival.rival_id in seen_rival_ids
        ):
            raise RawFocusedRivalBridgeError(
                "raw_rival_id_contract_invalid"
            )
        seen_rival_ids.add(rival.rival_id)
        numerator, denominator = _canonical_dyadic_pair(
            raw_value,
            max_exact_bits=caps.max_exact_bits,
            encoded_row=encoded_row,
        )
        binding = rival_spec_binding_digest(rival)
        if not _valid_sha256(binding):
            raise RawFocusedRivalBridgeError(
                "raw_rival_binding_digest_invalid"
            )
        entries.append(
            ExactRawRivalHardness(
                encoded_row=encoded_row,
                competitor_class=rival.rival_id,
                rival_id=rival.rival_id,
                rival_spec_binding_digest=binding,
                upper_numerator=numerator,
                upper_denominator=denominator,
            )
        )
    return tuple(entries)


def _validate_entry_sequence(
    entries: Any,
    rivals: Tuple[RivalSpec, ...],
    *,
    caps: RawFocusedRivalCaps,
    deadline: float,
    work: list[int],
) -> Tuple[Mapping[str, Any], ...]:
    if type(entries) is not tuple or len(entries) != len(rivals):
        raise RawFocusedRivalBridgeError(
            "hardness_receipt_entry_shape_invalid"
        )
    payloads = []
    for encoded_row, (entry, rival) in enumerate(
        zip(entries, rivals)
    ):
        _consume_work(
            work,
            1,
            caps=caps,
            deadline=deadline,
            stage="hardness_receipt_entry",
        )
        payload = _entry_payload(entry)
        numerator, denominator = _canonical_dyadic_pair(
            (
                payload["upper_numerator"],
                payload["upper_denominator"],
            ),
            max_exact_bits=caps.max_exact_bits,
            encoded_row=encoded_row,
        )
        binding = rival_spec_binding_digest(rival)
        if (
            payload["encoded_row"] != encoded_row
            or payload["competitor_class"] != rival.rival_id
            or payload["rival_id"] != rival.rival_id
            or payload["rival_spec_binding_digest"] != binding
            or numerator != payload["upper_numerator"]
            or denominator != payload["upper_denominator"]
        ):
            raise RawFocusedRivalBridgeError(
                "hardness_receipt_entry_binding_mismatch"
            )
        payloads.append(payload)
    return tuple(payloads)


def _hardness_content_payload(
    *,
    full_batch_sha256: str,
    full_live_assert_sha256: str,
    full_property_digest: str,
    full_live_assert_property_digest: str,
    live_interval_bounds_sha256: str,
    shape: Tuple[int, ...],
    entry_payloads: Sequence[Mapping[str, Any]],
    caps_payload: Mapping[str, int],
) -> Mapping[str, Any]:
    return {
        "schema": _HARDNESS_SCHEMA,
        "full_batch_sha256": full_batch_sha256,
        "full_live_assert_sha256": full_live_assert_sha256,
        "full_property_digest": full_property_digest,
        "full_live_assert_property_digest": (
            full_live_assert_property_digest
        ),
        "live_interval_bounds_sha256": (
            live_interval_bounds_sha256
        ),
        "method": _METHOD,
        "shape": list(shape),
        "entries": list(entry_payloads),
        "caps": dict(caps_payload),
    }


def _hardness_receipt_payload(
    *,
    content_payload: Mapping[str, Any],
    vector_digest: str,
) -> Mapping[str, Any]:
    return {
        **content_payload,
        "status": "full_exact_hardness_vector_bound_candidate",
        "candidate_only": True,
        "proof_authority": False,
        "owner_bound_consumed_raw_batch": True,
        "vector_digest": vector_digest,
    }


def issue_raw_rival_exact_hardness_receipt(
    batch: ConsumedRivalBatch,
    exact_upper_violations: Sequence[Tuple[int, int]],
    *,
    live_interval_bounds_sha256: str,
    deadline: float,
    max_rivals: int = _DEFAULT_MAX_RIVALS,
    max_focus: int = _DEFAULT_MAX_FOCUS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_work_items: int = _DEFAULT_MAX_WORK_ITEMS,
) -> RawRivalExactHardnessReceipt:
    """Bind one complete exact hardness vector to a live raw batch.

    ``exact_upper_violations`` must be a built-in tuple whose length exactly
    equals the full raw rival count.  Every item is a reduced built-in
    ``(numerator, positive_denominator)`` pair.  No float, NumPy scalar,
    missing row, or reordered identity is accepted.
    """

    deadline_value = _normalize_deadline(deadline)
    caps = _normalize_caps(
        max_rivals=max_rivals,
        max_focus=max_focus,
        max_exact_bits=max_exact_bits,
        max_work_items=max_work_items,
    )
    _check_deadline(deadline_value, stage="hardness_issue_begin")
    if not _valid_sha256(live_interval_bounds_sha256):
        raise RawFocusedRivalBridgeError(
            "live_interval_bounds_digest_invalid"
        )
    work = [0]
    rivals = _live_rivals(
        batch,
        caps=caps,
        deadline=deadline_value,
        work=work,
    )
    initial_batch_sha256 = batch.batch_sha256
    entries = _trusted_entries(
        rivals,
        exact_upper_violations,
        caps=caps,
        deadline=deadline_value,
        work=work,
    )
    property_digest = ordered_property_digest(rivals)
    full_digest = _full_binding_digest(
        batch_sha256=initial_batch_sha256,
        live_assert_sha256=batch.live_assert_sha256,
        property_digest=property_digest,
    )
    caps_payload = _caps_payload(caps)
    content = _hardness_content_payload(
        full_batch_sha256=initial_batch_sha256,
        full_live_assert_sha256=batch.live_assert_sha256,
        full_property_digest=property_digest,
        full_live_assert_property_digest=full_digest,
        live_interval_bounds_sha256=live_interval_bounds_sha256,
        shape=(len(entries),),
        entry_payloads=tuple(
            _entry_payload(entry) for entry in entries
        ),
        caps_payload=caps_payload,
    )
    vector_digest = _canonical_sha256(content)
    receipt_payload = _hardness_receipt_payload(
        content_payload=content,
        vector_digest=vector_digest,
    )
    receipt_sha256 = _canonical_sha256(receipt_payload)
    _check_deadline(
        deadline_value, stage="before_terminal_batch_validation"
    )
    if (
        not validate_consumed_raw_vnnlib_rival_batch(batch)
        or batch.batch_sha256 != initial_batch_sha256
    ):
        raise RawFocusedRivalBridgeError(
            "consumed_raw_batch_changed_during_hardness_issue"
        )
    _check_deadline(
        deadline_value, stage="hardness_issue_completion"
    )
    return RawRivalExactHardnessReceipt(
        full_batch_sha256=initial_batch_sha256,
        full_live_assert_sha256=batch.live_assert_sha256,
        full_property_digest=property_digest,
        full_live_assert_property_digest=full_digest,
        live_interval_bounds_sha256=(
            live_interval_bounds_sha256
        ),
        method=_METHOD,
        shape=(len(entries),),
        entries=entries,
        caps=caps,
        vector_digest=vector_digest,
        receipt_sha256=receipt_sha256,
        receipt=_deep_freeze(receipt_payload),
        _batch_owner=batch,
        _process_id=os.getpid(),
    )


def _hardness_exact_payload(
    batch: ConsumedRivalBatch,
    result: Any,
    *,
    expected_exact_upper_violations: Any,
    expected_live_interval_bounds_sha256: Any,
    expected_caps: RawFocusedRivalCaps,
    deadline: float,
) -> Mapping[str, Any]:
    if (
        type(result) is not RawRivalExactHardnessReceipt
        or result._batch_owner is not batch
        or type(result._process_id) is not int
        or result._process_id != os.getpid()
        or type(result.full_batch_sha256) is not str
        or type(result.full_live_assert_sha256) is not str
        or type(result.full_property_digest) is not str
        or type(result.full_live_assert_property_digest) is not str
        or type(result.live_interval_bounds_sha256) is not str
        or type(result.method) is not str
        or type(result.shape) is not tuple
        or len(result.shape) != 1
        or type(result.shape[0]) is not int
        or type(result.vector_digest) is not str
        or type(result.receipt_sha256) is not str
        or type(result.proof_authority) is not bool
        or result.proof_authority is not False
        or not _same_caps(result.caps, expected_caps)
    ):
        raise RawFocusedRivalBridgeError(
            "hardness_receipt_runtime_contract_invalid"
        )
    if not _valid_sha256(expected_live_interval_bounds_sha256):
        raise RawFocusedRivalBridgeError(
            "expected_live_interval_bounds_digest_invalid"
        )
    if (
        result.method != _METHOD
        or not _valid_sha256(result.full_batch_sha256)
        or not _valid_sha256(result.full_live_assert_sha256)
        or not _valid_sha256(result.full_property_digest)
        or not _valid_sha256(
            result.full_live_assert_property_digest
        )
        or not _valid_sha256(
            result.live_interval_bounds_sha256
        )
        or not _valid_sha256(result.vector_digest)
        or not _valid_sha256(result.receipt_sha256)
        or result.live_interval_bounds_sha256
        != expected_live_interval_bounds_sha256
    ):
        raise RawFocusedRivalBridgeError(
            "hardness_receipt_digest_or_method_invalid"
        )
    work = [0]
    rivals = _live_rivals(
        batch,
        caps=expected_caps,
        deadline=deadline,
        work=work,
    )
    property_digest = ordered_property_digest(rivals)
    full_digest = _full_binding_digest(
        batch_sha256=batch.batch_sha256,
        live_assert_sha256=batch.live_assert_sha256,
        property_digest=property_digest,
    )
    if (
        result.shape != (len(rivals),)
        or result.full_batch_sha256 != batch.batch_sha256
        or result.full_live_assert_sha256
        != batch.live_assert_sha256
        or result.full_property_digest != property_digest
        or result.full_live_assert_property_digest != full_digest
    ):
        raise RawFocusedRivalBridgeError(
            "hardness_receipt_full_batch_binding_mismatch"
        )
    entry_payloads = _validate_entry_sequence(
        result.entries,
        rivals,
        caps=expected_caps,
        deadline=deadline,
        work=work,
    )
    expected_entries = _trusted_entries(
        rivals,
        expected_exact_upper_violations,
        caps=expected_caps,
        deadline=deadline,
        work=work,
    )
    expected_entry_payloads = tuple(
        _entry_payload(entry) for entry in expected_entries
    )
    if _canonical_bytes(entry_payloads) != _canonical_bytes(
        expected_entry_payloads
    ):
        raise RawFocusedRivalBridgeError(
            "hardness_receipt_expected_live_vector_mismatch"
        )
    content = _hardness_content_payload(
        full_batch_sha256=batch.batch_sha256,
        full_live_assert_sha256=batch.live_assert_sha256,
        full_property_digest=property_digest,
        full_live_assert_property_digest=full_digest,
        live_interval_bounds_sha256=(
            result.live_interval_bounds_sha256
        ),
        shape=(len(rivals),),
        entry_payloads=entry_payloads,
        caps_payload=_caps_payload(expected_caps),
    )
    vector_digest = _canonical_sha256(content)
    receipt_payload = _hardness_receipt_payload(
        content_payload=content,
        vector_digest=vector_digest,
    )
    receipt_sha256 = _canonical_sha256(receipt_payload)
    if (
        result.vector_digest != vector_digest
        or result.receipt_sha256 != receipt_sha256
        or not _frozen_exact_match(
            result.receipt,
            receipt_payload,
            deadline=deadline,
        )
    ):
        raise RawFocusedRivalBridgeError(
            "hardness_receipt_payload_mismatch"
        )
    _check_deadline(deadline, stage="hardness_payload_complete")
    return content


def verify_raw_rival_exact_hardness_receipt(
    batch: ConsumedRivalBatch,
    result: RawRivalExactHardnessReceipt,
    *,
    expected_exact_upper_violations: Sequence[
        Tuple[int, int]
    ],
    expected_live_interval_bounds_sha256: str,
    deadline: float,
    max_rivals: int = _DEFAULT_MAX_RIVALS,
    max_focus: int = _DEFAULT_MAX_FOCUS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_work_items: int = _DEFAULT_MAX_WORK_ITEMS,
) -> bool:
    """Replay against independent live inputs, never candidate equality.

    The expected vector and interval-frame digest are required again at the
    verification boundary.  A self-consistent candidate receipt therefore
    cannot substitute a different same-shape ranking vector.
    """

    try:
        deadline_value = _normalize_deadline(deadline)
        caps = _normalize_caps(
            max_rivals=max_rivals,
            max_focus=max_focus,
            max_exact_bits=max_exact_bits,
            max_work_items=max_work_items,
        )
        initial_batch_sha256 = batch.batch_sha256
        _hardness_exact_payload(
            batch,
            result,
            expected_exact_upper_violations=(
                expected_exact_upper_violations
            ),
            expected_live_interval_bounds_sha256=(
                expected_live_interval_bounds_sha256
            ),
            expected_caps=caps,
            deadline=deadline_value,
        )
        _check_deadline(
            deadline_value,
            stage="before_hardness_terminal_batch_validation",
        )
        return (
            validate_consumed_raw_vnnlib_rival_batch(batch)
            and batch.batch_sha256 == initial_batch_sha256
            and time.monotonic() < deadline_value
        )
    except (
        RawFocusedRivalBridgeError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _rank_entries(
    entries: Tuple[ExactRawRivalHardness, ...],
    *,
    caps: RawFocusedRivalCaps,
    deadline: float,
) -> Tuple[RankedRawRivalHardness, ...]:
    if len(entries) > caps.max_rivals:
        raise RawFocusedRivalBridgeError(
            "ranked_entry_count_exceeds_cap"
        )
    decorated = []
    for entry in entries:
        _check_deadline(deadline, stage="ranking_input")
        decorated.append(
            (
                Fraction(
                    entry.upper_numerator,
                    entry.upper_denominator,
                ),
                entry,
            )
        )
    ordered = sorted(
        decorated,
        key=lambda item: (
            -item[0],
            item[1].encoded_row,
            item[1].rival_id,
            item[1].rival_spec_binding_digest,
        ),
    )
    ranked = tuple(
        RankedRawRivalHardness(
            rank=rank,
            encoded_row=entry.encoded_row,
            competitor_class=entry.competitor_class,
            rival_id=entry.rival_id,
            rival_spec_binding_digest=(
                entry.rival_spec_binding_digest
            ),
            upper_numerator=entry.upper_numerator,
            upper_denominator=entry.upper_denominator,
        )
        for rank, (_upper, entry) in enumerate(ordered)
    )
    _check_deadline(deadline, stage="ranking_complete")
    return ranked


def _strict_json_snapshot(
    value: Any,
    *,
    caps: RawFocusedRivalCaps,
    deadline: float,
    work: list[int],
    depth: int = 0,
) -> Any:
    """Copy one caller receipt through a small exact JSON boundary."""

    if depth > 64:
        raise RawFocusedRivalBridgeError(
            "residual_selector_receipt_depth_exceeded"
        )
    _consume_work(
        work,
        1,
        caps=caps,
        deadline=deadline,
        stage="residual_selector_receipt_snapshot",
    )
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if abs(value) > _MAX_INT64:
            raise RawFocusedRivalBridgeError(
                "residual_selector_receipt_integer_out_of_range"
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RawFocusedRivalBridgeError(
                "residual_selector_receipt_float_nonfinite"
            )
        return value
    if type(value) is str:
        _consume_work(
            work,
            len(value.encode("utf-8")),
            caps=caps,
            deadline=deadline,
            stage="residual_selector_receipt_string",
        )
        return value
    if type(value) in {list, tuple}:
        return [
            _strict_json_snapshot(
                item,
                caps=caps,
                deadline=deadline,
                work=work,
                depth=depth + 1,
            )
            for item in value
        ]
    if type(value) in {dict, MappingProxyType}:
        snapshot = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RawFocusedRivalBridgeError(
                    "residual_selector_receipt_key_not_string"
                )
            _consume_work(
                work,
                len(key.encode("utf-8")),
                caps=caps,
                deadline=deadline,
                stage="residual_selector_receipt_key",
            )
            snapshot[key] = _strict_json_snapshot(
                item,
                caps=caps,
                deadline=deadline,
                work=work,
                depth=depth + 1,
            )
        return snapshot
    raise RawFocusedRivalBridgeError(
        "residual_selector_receipt_non_json_builtin"
    )


def _prepare_focus_method(
    batch: ConsumedRivalBatch,
    *,
    focus_count: int,
    explicit_encoded_focus_row: Any,
    residual_selector_receipt: Any,
    residual_selector_property_sha256: Any,
    caps: RawFocusedRivalCaps,
    deadline: float,
) -> Tuple[
    str,
    int | None,
    str | None,
    str | None,
    int | None,
    Mapping[str, Any] | None,
]:
    """Validate either default hardness ranking or residual joint focus."""

    explicit_mode = explicit_encoded_focus_row is not None
    if not explicit_mode:
        if (
            residual_selector_receipt is not None
            or residual_selector_property_sha256 is not None
        ):
            raise RawFocusedRivalBridgeError(
                "partial_residual_joint_focus_binding"
            )
        return (
            _HARDEST_FOCUS_METHOD,
            None,
            None,
            None,
            None,
            None,
        )
    if focus_count != 1:
        raise RawFocusedRivalBridgeError(
            "residual_joint_focus_requires_singleton"
        )
    encoded_row = _strict_int(
        explicit_encoded_focus_row,
        name="explicit_encoded_focus_row",
    )
    rivals = batch.rivals
    if encoded_row < 0 or encoded_row >= len(rivals):
        raise RawFocusedRivalBridgeError(
            "explicit_encoded_focus_row_out_of_range"
        )
    if (
        type(residual_selector_property_sha256) is not str
        or not _valid_sha256(
            residual_selector_property_sha256
        )
        or type(residual_selector_receipt) is not dict
    ):
        raise RawFocusedRivalBridgeError(
            "residual_selector_binding_input_invalid"
        )
    work = [0]
    snapshot = _strict_json_snapshot(
        residual_selector_receipt,
        caps=caps,
        deadline=deadline,
        work=work,
    )
    if type(snapshot) is not dict:
        raise RawFocusedRivalBridgeError(
            "residual_selector_receipt_not_mapping"
        )
    # ``property_residual_targets`` currently names processed rivals by the
    # encoded C-row index.  Raw ``RivalSpec.rival_id`` instead names the
    # competitor class.  They coincide only accidentally for some labels, so
    # first authenticate the residual row ID and only then map that row
    # through the full consumed raw batch to its competitor/RivalSpec.
    mapped_rival = rivals[encoded_row]
    mapped_binding = rival_spec_binding_digest(mapped_rival)
    expected_residual_property_sha256 = (
        _residual_selector_property_digest(
            rivals, deadline=deadline
        )
    )
    joint_id = snapshot.get("joint_focus_rival_id")
    rival_ids = snapshot.get("rival_ids")
    if (
        snapshot.get("proof_authority") is not False
        or snapshot.get("status") != "selected"
        or snapshot.get("selection_policy")
        != "facility_first_then_same_rival_joint"
        or type(snapshot.get("targets_selected")) is not int
        or snapshot["targets_selected"] < 1
        or type(joint_id) is not int
        or joint_id != encoded_row
        or type(rival_ids) is not list
        or not rival_ids
        or any(type(value) is not int for value in rival_ids)
        or len(set(rival_ids)) != len(rival_ids)
        or joint_id not in rival_ids
        or snapshot.get("property_sha256")
        != residual_selector_property_sha256
        or residual_selector_property_sha256
        != expected_residual_property_sha256
        or not _valid_sha256(mapped_binding)
    ):
        raise RawFocusedRivalBridgeError(
            "residual_selector_joint_focus_contract_invalid"
        )
    receipt_sha256 = _canonical_sha256(snapshot)
    _check_deadline(
        deadline, stage="residual_selector_binding_complete"
    )
    return (
        _RESIDUAL_JOINT_FOCUS_METHOD,
        encoded_row,
        residual_selector_property_sha256,
        receipt_sha256,
        joint_id,
        _deep_freeze(snapshot),
    )


def _focused_content_payload(
    *,
    hardness: RawRivalExactHardnessReceipt,
    method: str,
    focus_count: int,
    explicit_encoded_focus_row: int | None,
    residual_selector_property_sha256: str | None,
    residual_selector_receipt_sha256: str | None,
    residual_joint_focus_rival_id: int | None,
    ranked_payloads: Sequence[Mapping[str, Any]],
    focused_payloads: Sequence[Mapping[str, Any]],
    caps_payload: Mapping[str, int],
    status: str,
) -> Mapping[str, Any]:
    return {
        "schema": _FOCUSED_SCHEMA,
        "status": status,
        "full_batch_sha256": hardness.full_batch_sha256,
        "full_live_assert_sha256": (
            hardness.full_live_assert_sha256
        ),
        "full_property_digest": hardness.full_property_digest,
        "full_live_assert_property_digest": (
            hardness.full_live_assert_property_digest
        ),
        "live_interval_bounds_sha256": (
            hardness.live_interval_bounds_sha256
        ),
        "hardness_vector_digest": hardness.vector_digest,
        "hardness_method": hardness.method,
        "method": method,
        "focus_count": focus_count,
        "explicit_encoded_focus_row": explicit_encoded_focus_row,
        "residual_selector_property_sha256": (
            residual_selector_property_sha256
        ),
        "residual_selector_receipt_sha256": (
            residual_selector_receipt_sha256
        ),
        "residual_joint_focus_rival_id": (
            residual_joint_focus_rival_id
        ),
        "ranked_entries": list(ranked_payloads),
        "focused_entries": list(focused_payloads),
        "focused_rival_bindings": [
            {
                "encoded_row": payload["encoded_row"],
                "competitor_class": payload[
                    "competitor_class"
                ],
                "rival_id": payload["rival_id"],
                "rival_spec_binding_digest": payload[
                    "rival_spec_binding_digest"
                ],
            }
            for payload in focused_payloads
        ],
        "caps": dict(caps_payload),
    }


def _focused_receipt_payload(
    *,
    content_payload: Mapping[str, Any],
    focused_subset_digest: str,
) -> Mapping[str, Any]:
    return {
        **content_payload,
        "candidate_only": True,
        "proof_authority": False,
        "owner_bound_consumed_raw_batch": True,
        "selection_coverage": "explicit_focused_subset_only",
        "focused_subset_digest": focused_subset_digest,
    }


def select_raw_focused_rivals(
    batch: ConsumedRivalBatch,
    hardness: RawRivalExactHardnessReceipt,
    *,
    focus_count: int = 1,
    explicit_encoded_focus_row: int | None = None,
    residual_selector_receipt: Mapping[str, Any] | None = None,
    residual_selector_property_sha256: str | None = None,
    expected_exact_upper_violations: Sequence[
        Tuple[int, int]
    ],
    expected_live_interval_bounds_sha256: str,
    deadline: float,
    max_rivals: int = _DEFAULT_MAX_RIVALS,
    max_focus: int = _DEFAULT_MAX_FOCUS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_work_items: int = _DEFAULT_MAX_WORK_ITEMS,
) -> RawFocusedRivalSelection:
    """Select the hardest subset or one receipt-bound residual focus row.

    Selection replays the hardness receipt against the independently supplied
    expected live vector/frame before consulting any rank.
    """

    deadline_value = _normalize_deadline(deadline)
    caps = _normalize_caps(
        max_rivals=max_rivals,
        max_focus=max_focus,
        max_exact_bits=max_exact_bits,
        max_work_items=max_work_items,
    )
    normalized_focus_count = _strict_int(
        focus_count, name="focus_count"
    )
    if (
        normalized_focus_count < 1
        or normalized_focus_count > caps.max_focus
    ):
        raise RawFocusedRivalBridgeError(
            "focus_count_out_of_registered_range"
        )
    _hardness_exact_payload(
        batch,
        hardness,
        expected_exact_upper_violations=(
            expected_exact_upper_violations
        ),
        expected_live_interval_bounds_sha256=(
            expected_live_interval_bounds_sha256
        ),
        expected_caps=caps,
        deadline=deadline_value,
    )
    if normalized_focus_count > len(hardness.entries):
        raise RawFocusedRivalBridgeError(
            "focus_count_exceeds_full_batch"
        )
    (
        method,
        bound_encoded_row,
        bound_residual_property_sha256,
        bound_residual_receipt_sha256,
        bound_joint_rival_id,
        frozen_residual_receipt,
    ) = _prepare_focus_method(
        batch,
        focus_count=normalized_focus_count,
        explicit_encoded_focus_row=explicit_encoded_focus_row,
        residual_selector_receipt=residual_selector_receipt,
        residual_selector_property_sha256=(
            residual_selector_property_sha256
        ),
        caps=caps,
        deadline=deadline_value,
    )
    initial_batch_sha256 = batch.batch_sha256
    ranked = _rank_entries(
        hardness.entries,
        caps=caps,
        deadline=deadline_value,
    )
    if bound_encoded_row is None:
        focused = ranked[:normalized_focus_count]
    else:
        focused = tuple(
            item
            for item in ranked
            if item.encoded_row == bound_encoded_row
        )
        if len(focused) != 1:
            raise RawFocusedRivalBridgeError(
                "residual_joint_focus_rank_mapping_missing"
            )
    live_rivals = batch.rivals
    focused_rivals = tuple(
        live_rivals[item.encoded_row] for item in focused
    )
    status = (
        "focused_singleton_candidate"
        if normalized_focus_count == 1
        else "focused_subset_candidate"
    )
    ranked_payloads = tuple(
        _ranked_payload(item) for item in ranked
    )
    focused_payloads = tuple(
        _ranked_payload(item) for item in focused
    )
    content = _focused_content_payload(
        hardness=hardness,
        method=method,
        focus_count=normalized_focus_count,
        explicit_encoded_focus_row=bound_encoded_row,
        residual_selector_property_sha256=(
            bound_residual_property_sha256
        ),
        residual_selector_receipt_sha256=(
            bound_residual_receipt_sha256
        ),
        residual_joint_focus_rival_id=bound_joint_rival_id,
        ranked_payloads=ranked_payloads,
        focused_payloads=focused_payloads,
        caps_payload=_caps_payload(caps),
        status=status,
    )
    subset_digest = _canonical_sha256(content)
    receipt_payload = _focused_receipt_payload(
        content_payload=content,
        focused_subset_digest=subset_digest,
    )
    receipt_sha256 = _canonical_sha256(receipt_payload)
    _check_deadline(
        deadline_value,
        stage="before_focus_terminal_batch_validation",
    )
    if (
        not validate_consumed_raw_vnnlib_rival_batch(batch)
        or batch.batch_sha256 != initial_batch_sha256
    ):
        raise RawFocusedRivalBridgeError(
            "consumed_raw_batch_changed_during_focus"
        )
    _check_deadline(deadline_value, stage="focus_completion")
    return RawFocusedRivalSelection(
        status=status,
        full_batch_sha256=hardness.full_batch_sha256,
        full_live_assert_sha256=(
            hardness.full_live_assert_sha256
        ),
        full_property_digest=hardness.full_property_digest,
        full_live_assert_property_digest=(
            hardness.full_live_assert_property_digest
        ),
        live_interval_bounds_sha256=(
            hardness.live_interval_bounds_sha256
        ),
        hardness_vector_digest=hardness.vector_digest,
        hardness_method=hardness.method,
        method=method,
        focus_count=normalized_focus_count,
        explicit_encoded_focus_row=bound_encoded_row,
        residual_selector_property_sha256=(
            bound_residual_property_sha256
        ),
        residual_selector_receipt_sha256=(
            bound_residual_receipt_sha256
        ),
        residual_joint_focus_rival_id=bound_joint_rival_id,
        residual_selector_receipt=frozen_residual_receipt,
        ranked_entries=ranked,
        focused_entries=focused,
        focused_rivals=focused_rivals,
        caps=caps,
        focused_subset_digest=subset_digest,
        receipt_sha256=receipt_sha256,
        receipt=_deep_freeze(receipt_payload),
        _batch_owner=batch,
        _hardness_owner=hardness,
        _process_id=os.getpid(),
    )


def _validate_ranked_sequence(
    ranked: Any,
    trusted_entries: Tuple[ExactRawRivalHardness, ...],
    *,
    caps: RawFocusedRivalCaps,
    deadline: float,
) -> Tuple[Mapping[str, Any], ...]:
    if type(ranked) is not tuple or len(ranked) != len(
        trusted_entries
    ):
        raise RawFocusedRivalBridgeError(
            "ranked_sequence_shape_invalid"
        )
    trusted_by_row = {
        entry.encoded_row: entry for entry in trusted_entries
    }
    seen_rows = set()
    payloads = []
    previous_key = None
    for expected_rank, item in enumerate(ranked):
        _check_deadline(deadline, stage="ranked_sequence_entry")
        payload = _ranked_payload(item)
        if (
            payload["rank"] != expected_rank
            or payload["encoded_row"] in seen_rows
            or payload["encoded_row"] not in trusted_by_row
        ):
            raise RawFocusedRivalBridgeError(
                "ranked_sequence_rank_or_row_invalid"
            )
        seen_rows.add(payload["encoded_row"])
        trusted = trusted_by_row[payload["encoded_row"]]
        if (
            payload["competitor_class"]
            != trusted.competitor_class
            or payload["rival_id"] != trusted.rival_id
            or payload["rival_spec_binding_digest"]
            != trusted.rival_spec_binding_digest
            or payload["upper_numerator"]
            != trusted.upper_numerator
            or payload["upper_denominator"]
            != trusted.upper_denominator
        ):
            raise RawFocusedRivalBridgeError(
                "ranked_sequence_entry_binding_mismatch"
            )
        exact_upper = Fraction(
            payload["upper_numerator"],
            payload["upper_denominator"],
        )
        ordering_key = (
            -exact_upper,
            payload["encoded_row"],
            payload["rival_id"],
            payload["rival_spec_binding_digest"],
        )
        if previous_key is not None and ordering_key < previous_key:
            raise RawFocusedRivalBridgeError(
                "ranked_sequence_order_invalid"
            )
        previous_key = ordering_key
        payloads.append(payload)
    return tuple(payloads)


def _focused_exact_payload(
    batch: ConsumedRivalBatch,
    hardness: RawRivalExactHardnessReceipt,
    result: Any,
    *,
    expected_caps: RawFocusedRivalCaps,
    expected_focus_count: int,
    deadline: float,
) -> Mapping[str, Any]:
    if (
        type(result) is not RawFocusedRivalSelection
        or result._batch_owner is not batch
        or result._hardness_owner is not hardness
        or type(result._process_id) is not int
        or result._process_id != os.getpid()
        or type(result.status) is not str
        or type(result.full_batch_sha256) is not str
        or type(result.full_live_assert_sha256) is not str
        or type(result.full_property_digest) is not str
        or type(result.full_live_assert_property_digest) is not str
        or type(result.live_interval_bounds_sha256) is not str
        or type(result.hardness_vector_digest) is not str
        or type(result.hardness_method) is not str
        or type(result.method) is not str
        or type(result.focus_count) is not int
        or type(result.focused_subset_digest) is not str
        or type(result.receipt_sha256) is not str
        or type(result.proof_authority) is not bool
        or result.proof_authority is not False
        or not _same_caps(result.caps, expected_caps)
    ):
        raise RawFocusedRivalBridgeError(
            "focused_selection_runtime_contract_invalid"
        )
    if (
        result.focus_count != expected_focus_count
        or result.focus_count < 1
        or result.focus_count > expected_caps.max_focus
        or result.hardness_method != hardness.method
        or result.full_batch_sha256
        != hardness.full_batch_sha256
        or result.full_live_assert_sha256
        != hardness.full_live_assert_sha256
        or result.full_property_digest
        != hardness.full_property_digest
        or result.full_live_assert_property_digest
        != hardness.full_live_assert_property_digest
        or result.live_interval_bounds_sha256
        != hardness.live_interval_bounds_sha256
        or result.hardness_vector_digest
        != hardness.vector_digest
    ):
        raise RawFocusedRivalBridgeError(
            "focused_selection_parent_binding_mismatch"
        )
    expected_status = (
        "focused_singleton_candidate"
        if expected_focus_count == 1
        else "focused_subset_candidate"
    )
    if result.status != expected_status:
        raise RawFocusedRivalBridgeError(
            "focused_selection_status_mismatch"
        )
    ranked_payloads = _validate_ranked_sequence(
        result.ranked_entries,
        hardness.entries,
        caps=expected_caps,
        deadline=deadline,
    )
    if result.residual_selector_receipt is None:
        thawed_residual_receipt = None
    else:
        receipt_work = [0]
        thawed_residual_receipt = _strict_json_snapshot(
            result.residual_selector_receipt,
            caps=expected_caps,
            deadline=deadline,
            work=receipt_work,
        )
        if type(thawed_residual_receipt) is not dict:
            raise RawFocusedRivalBridgeError(
                "focused_residual_receipt_not_mapping"
            )
    (
        expected_method,
        expected_encoded_row,
        expected_residual_property_sha256,
        expected_residual_receipt_sha256,
        expected_joint_rival_id,
        expected_frozen_residual_receipt,
    ) = _prepare_focus_method(
        batch,
        focus_count=expected_focus_count,
        explicit_encoded_focus_row=(
            result.explicit_encoded_focus_row
        ),
        residual_selector_receipt=thawed_residual_receipt,
        residual_selector_property_sha256=(
            result.residual_selector_property_sha256
        ),
        caps=expected_caps,
        deadline=deadline,
    )
    if (
        result.method != expected_method
        or result.residual_selector_receipt_sha256
        != expected_residual_receipt_sha256
        or result.residual_joint_focus_rival_id
        != expected_joint_rival_id
    ):
        raise RawFocusedRivalBridgeError(
            "focused_selection_method_binding_mismatch"
        )
    if expected_frozen_residual_receipt is None:
        if result.residual_selector_receipt is not None:
            raise RawFocusedRivalBridgeError(
                "unexpected_residual_selector_receipt"
            )
    elif not _frozen_exact_match(
        result.residual_selector_receipt,
        thawed_residual_receipt,
        deadline=deadline,
    ):
        raise RawFocusedRivalBridgeError(
            "residual_selector_receipt_snapshot_mismatch"
        )
    if (
        type(result.focused_entries) is not tuple
        or len(result.focused_entries) != expected_focus_count
        or type(result.focused_rivals) is not tuple
        or len(result.focused_rivals) != expected_focus_count
    ):
        raise RawFocusedRivalBridgeError(
            "focused_selection_subset_shape_invalid"
        )
    live_rivals = batch.rivals
    focused_payloads = []
    if expected_encoded_row is None:
        expected_focused_payloads = ranked_payloads[
            :expected_focus_count
        ]
    else:
        expected_focused_payloads = tuple(
            payload
            for payload in ranked_payloads
            if payload["encoded_row"] == expected_encoded_row
        )
        if len(expected_focused_payloads) != 1:
            raise RawFocusedRivalBridgeError(
                "residual_joint_focus_expected_row_missing"
            )
    for index in range(expected_focus_count):
        _check_deadline(deadline, stage="focused_subset_entry")
        item = result.focused_entries[index]
        item_payload = _ranked_payload(item)
        expected_payload = expected_focused_payloads[index]
        if _canonical_bytes(item_payload) != _canonical_bytes(
            expected_payload
        ):
            raise RawFocusedRivalBridgeError(
                "focused_entry_not_ranked_prefix"
            )
        expected_rival = live_rivals[
            expected_payload["encoded_row"]
        ]
        actual_rival = result.focused_rivals[index]
        if (
            actual_rival is not expected_rival
            or type(actual_rival) is not RivalSpec
            or rival_spec_binding_digest(actual_rival)
            != expected_payload["rival_spec_binding_digest"]
        ):
            raise RawFocusedRivalBridgeError(
                "focused_rival_live_identity_mismatch"
            )
        focused_payloads.append(expected_payload)
    content = _focused_content_payload(
        hardness=hardness,
        method=expected_method,
        focus_count=expected_focus_count,
        explicit_encoded_focus_row=expected_encoded_row,
        residual_selector_property_sha256=(
            expected_residual_property_sha256
        ),
        residual_selector_receipt_sha256=(
            expected_residual_receipt_sha256
        ),
        residual_joint_focus_rival_id=expected_joint_rival_id,
        ranked_payloads=ranked_payloads,
        focused_payloads=tuple(focused_payloads),
        caps_payload=_caps_payload(expected_caps),
        status=expected_status,
    )
    subset_digest = _canonical_sha256(content)
    receipt_payload = _focused_receipt_payload(
        content_payload=content,
        focused_subset_digest=subset_digest,
    )
    receipt_sha256 = _canonical_sha256(receipt_payload)
    if (
        result.focused_subset_digest != subset_digest
        or result.receipt_sha256 != receipt_sha256
        or not _frozen_exact_match(
            result.receipt,
            receipt_payload,
            deadline=deadline,
        )
    ):
        raise RawFocusedRivalBridgeError(
            "focused_selection_payload_mismatch"
        )
    _check_deadline(deadline, stage="focused_payload_complete")
    return content


def verify_raw_focused_rival_selection(
    batch: ConsumedRivalBatch,
    hardness: RawRivalExactHardnessReceipt,
    result: RawFocusedRivalSelection,
    *,
    expected_focus_count: int = 1,
    expected_exact_upper_violations: Sequence[
        Tuple[int, int]
    ],
    expected_live_interval_bounds_sha256: str,
    deadline: float,
    max_rivals: int = _DEFAULT_MAX_RIVALS,
    max_focus: int = _DEFAULT_MAX_FOCUS,
    max_exact_bits: int = _DEFAULT_MAX_EXACT_BITS,
    max_work_items: int = _DEFAULT_MAX_WORK_ITEMS,
) -> bool:
    """Live primitive replay including the independent hardness inputs."""

    try:
        deadline_value = _normalize_deadline(deadline)
        caps = _normalize_caps(
            max_rivals=max_rivals,
            max_focus=max_focus,
            max_exact_bits=max_exact_bits,
            max_work_items=max_work_items,
        )
        normalized_focus_count = _strict_int(
            expected_focus_count, name="expected_focus_count"
        )
        if (
            normalized_focus_count < 1
            or normalized_focus_count > caps.max_focus
        ):
            return False
        initial_batch_sha256 = batch.batch_sha256
        _hardness_exact_payload(
            batch,
            hardness,
            expected_exact_upper_violations=(
                expected_exact_upper_violations
            ),
            expected_live_interval_bounds_sha256=(
                expected_live_interval_bounds_sha256
            ),
            expected_caps=caps,
            deadline=deadline_value,
        )
        _focused_exact_payload(
            batch,
            hardness,
            result,
            expected_caps=caps,
            expected_focus_count=normalized_focus_count,
            deadline=deadline_value,
        )
        _check_deadline(
            deadline_value,
            stage="before_focus_verify_terminal_batch_validation",
        )
        return (
            validate_consumed_raw_vnnlib_rival_batch(batch)
            and batch.batch_sha256 == initial_batch_sha256
            and time.monotonic() < deadline_value
        )
    except (
        RawFocusedRivalBridgeError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


__all__ = [
    "ExactRawRivalHardness",
    "RankedRawRivalHardness",
    "RawFocusedRivalBridgeError",
    "RawFocusedRivalCaps",
    "RawFocusedRivalSelection",
    "RawRivalExactHardnessReceipt",
    "issue_raw_rival_exact_hardness_receipt",
    "select_raw_focused_rivals",
    "verify_raw_focused_rival_selection",
    "verify_raw_rival_exact_hardness_receipt",
]
