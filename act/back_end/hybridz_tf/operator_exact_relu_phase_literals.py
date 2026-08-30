#!/usr/bin/env python3
"""Candidate-only Operator-HZ exact-ReLU property phase selection.

The graph-output Operator-HZ representation deliberately stores exact ReLU
phase binaries in local Big-M constraints rather than in ``Gb``.  Consequently
the generic ``objective @ Gb`` phase selector sees a zero column even when a
final property depends directly on the corresponding ReLU output.

This isolated adapter re-derives the missing association from the live sparse
rows:

* one ``relu_exact_x_branch:<layer>`` row with a positive binary coefficient;
* one ``relu_exact_zero_branch:<layer>`` row with a negative binary
  coefficient and exactly one continuous ReLU-output factor; and
* the unique opposite ``relu_exact_lower:<layer>`` continuous row.

The association is accepted only when all three rows have the canonical
stored-binary64 structure and stable continuous/binary IDs are valid.  For
each ordered :class:`RivalSpec`, the direct output-column coefficient is then
accumulated with ``Fraction.from_float``.  A unanimous positive sign proposes
the active literal, a unanimous negative sign proposes the inactive literal,
and an exact all-zero coefficient produces an explicit omission.

This module has no cut or verdict integration.  Every returned object has
``proof_authority=False``.  Missing/ambiguous rows, non-canonical sparse
storage, a missing direct suffix column, property-tail output, or disagreeing
rival signs fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    ordered_property_digest,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
)
from act.back_end.solver.solver_hz import SparseHZono


@dataclass(frozen=True)
class ExactDyadicRivalCoefficient:
    """One exact coefficient of a mapped ReLU-output factor."""

    rival_id: int
    numerator: int
    denominator: int

    @property
    def value(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class OperatorExactReLUPhaseMapping:
    """Strict row/column association for one live exact-ReLU binary."""

    stable_bcol_id: int
    binary_position: int
    relu_layer_id: int
    stable_output_col_id: int
    output_continuous_position: int
    lower_upper_row: int
    x_branch_upper_row: int
    zero_branch_upper_row: int
    rival_coefficients: Tuple[ExactDyadicRivalCoefficient, ...]
    selected_phase: Optional[int]


@dataclass(frozen=True)
class OperatorExactReLUZeroOmission:
    """An explicit exact-zero property omission, never an inferred literal."""

    stable_bcol_id: int
    stable_output_col_id: int
    rival_coefficients: Tuple[ExactDyadicRivalCoefficient, ...]
    reason: str = "all_ordered_rival_coefficients_exactly_zero"
    proof_authority: bool = False


@dataclass(frozen=True)
class OperatorExactReLUPhaseSelectionCaps:
    """Caller-bound resource limits for derivation and live verification."""

    max_rivals: int
    max_binaries: int
    max_work_items: int
    timeout_seconds: float


@dataclass(frozen=True)
class OperatorExactReLUPhaseSelection:
    """Non-authoritative phase candidates bound to one live parent/property."""

    status: str
    mappings: Tuple[OperatorExactReLUPhaseMapping, ...]
    literals: Tuple[PhaseLiteral, ...]
    zero_omissions: Tuple[OperatorExactReLUZeroOmission, ...]
    parent_semantic_digest: str
    operator_row_tag_digest: str
    property_digest: str
    caps: OperatorExactReLUPhaseSelectionCaps
    selection_digest: str
    arithmetic: str = "Fraction.from_float_exact_dyadic"
    proof_authority: bool = False


class OperatorExactReLUPhaseLiteralError(ValueError):
    """The live Operator-HZ structure cannot safely drive this candidate."""


_EXACT_TAG = re.compile(
    r"relu_exact_(lower|x_branch|zero_branch):([0-9]+)"
)
_EXACT_TAG_PREFIX = "relu_exact_"
_ZERO_OMISSION_REASON = (
    "all_ordered_rival_coefficients_exactly_zero"
)

_DEFAULT_MAX_RIVALS = 128
_DEFAULT_MAX_BINARIES = 16384
_DEFAULT_MAX_WORK_ITEMS = 5_000_000
_DEFAULT_TIMEOUT_SECONDS = 5.0

_HARD_MAX_RIVALS = 256
_HARD_MAX_BINARIES = 65536
_HARD_MAX_WORK_ITEMS = 50_000_000
_HARD_TIMEOUT_SECONDS = 60.0
_HARD_EXACT_INTEGER_BITS = 8192
_MAX_INT64 = int(np.iinfo(np.int64).max)


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


def _strict_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise OperatorExactReLUPhaseLiteralError(f"{name}_not_integer")
    return value


def _normalize_caps(
    *,
    max_rivals: int,
    max_binaries: int,
    max_work_items: int,
    timeout_seconds: float,
) -> OperatorExactReLUPhaseSelectionCaps:
    normalized_max_rivals = _strict_int(
        max_rivals, name="max_rivals"
    )
    normalized_max_binaries = _strict_int(
        max_binaries, name="max_binaries"
    )
    normalized_max_work_items = _strict_int(
        max_work_items, name="max_work_items"
    )
    if (
        normalized_max_rivals < 1
        or normalized_max_rivals > _HARD_MAX_RIVALS
        or normalized_max_binaries < 1
        or normalized_max_binaries > _HARD_MAX_BINARIES
        or normalized_max_work_items < 1
        or normalized_max_work_items > _HARD_MAX_WORK_ITEMS
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "selection_resource_cap_out_of_range"
        )
    if type(timeout_seconds) not in {int, float}:
        raise OperatorExactReLUPhaseLiteralError(
            "selection_timeout_not_builtin_numeric"
        )
    normalized_timeout = float(timeout_seconds)
    if (
        not math.isfinite(normalized_timeout)
        or normalized_timeout <= 0.0
        or normalized_timeout > _HARD_TIMEOUT_SECONDS
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "selection_timeout_out_of_range"
        )
    return OperatorExactReLUPhaseSelectionCaps(
        max_rivals=normalized_max_rivals,
        max_binaries=normalized_max_binaries,
        max_work_items=normalized_max_work_items,
        timeout_seconds=normalized_timeout,
    )


def _caps_payload(
    caps: OperatorExactReLUPhaseSelectionCaps,
) -> dict[str, Any]:
    if (
        type(caps) is not OperatorExactReLUPhaseSelectionCaps
        or type(caps.max_rivals) is not int
        or type(caps.max_binaries) is not int
        or type(caps.max_work_items) is not int
        or type(caps.timeout_seconds) is not float
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "selection_caps_noncanonical"
        )
    normalized = _normalize_caps(
        max_rivals=caps.max_rivals,
        max_binaries=caps.max_binaries,
        max_work_items=caps.max_work_items,
        timeout_seconds=caps.timeout_seconds,
    )
    return {
        "max_rivals": normalized.max_rivals,
        "max_binaries": normalized.max_binaries,
        "max_work_items": normalized.max_work_items,
        "timeout_seconds_f64_hex": (
            normalized.timeout_seconds.hex()
        ),
    }


def _check_deadline(deadline: float, *, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise OperatorExactReLUPhaseLiteralError(
            f"selection_deadline_expired_{stage}"
        )


def _consume_work(
    work: list[int],
    amount: int,
    *,
    caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
    stage: str,
) -> None:
    if type(amount) is not int or amount < 0:
        raise OperatorExactReLUPhaseLiteralError(
            "selection_work_accounting_invalid"
        )
    work[0] += amount
    if work[0] > caps.max_work_items:
        raise OperatorExactReLUPhaseLiteralError(
            f"selection_work_cap_exceeded_{stage}"
        )
    _check_deadline(deadline, stage=stage)


def _valid_stable_ids(
    value: Any,
    *,
    expected: int,
    name: str,
) -> Tuple[int, ...]:
    if value is None:
        raise OperatorExactReLUPhaseLiteralError(f"{name}_missing")
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.int64)
        or array.ndim != 1
        or int(array.size) != int(expected)
        or (array.size and np.any(array < 0))
    ):
        raise OperatorExactReLUPhaseLiteralError(f"{name}_malformed")
    result = tuple(int(item) for item in array.tolist())
    if len(set(result)) != len(result):
        raise OperatorExactReLUPhaseLiteralError(f"{name}_duplicate")
    return result


def _row_entries(
    matrix: sp.csr_matrix,
    row: int,
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    start = int(matrix.indptr[row])
    stop = int(matrix.indptr[row + 1])
    return (
        tuple(int(item) for item in matrix.indices[start:stop]),
        tuple(float(item) for item in matrix.data[start:stop]),
    )


def _rows_are_exact_negatives(
    left: Tuple[Tuple[int, ...], Tuple[float, ...]],
    right: Tuple[Tuple[int, ...], Tuple[float, ...]],
) -> bool:
    left_indices, left_values = left
    right_indices, right_values = right
    return (
        left_indices == right_indices
        and len(left_values) == len(right_values)
        and all(
            Fraction.from_float(left_value)
            == -Fraction.from_float(right_value)
            for left_value, right_value in zip(left_values, right_values)
        )
    )


def _row_signature(
    entries: Tuple[Tuple[int, ...], Tuple[float, ...]],
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    indices, values = entries
    return indices, tuple(value.hex() for value in values)


def _negative_row_signature(
    entries: Tuple[Tuple[int, ...], Tuple[float, ...]],
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    indices, values = entries
    return indices, tuple((-value).hex() for value in values)


def _stored_scales_within_ulps(
    left: float,
    right: float,
    *,
    max_ulps: int,
) -> bool:
    """Compare two positive builder scales across bounded outward rounding."""

    if (
        not np.isfinite(left)
        or not np.isfinite(right)
        or left <= 0.0
        or right <= 0.0
    ):
        return False
    lower = min(float(left), float(right))
    upper = max(float(left), float(right))
    for _ in range(int(max_ulps) + 1):
        if lower == upper:
            return True
        lower = float(np.nextafter(lower, np.inf))
    return False


def _fraction_payload(
    coefficients: Tuple[ExactDyadicRivalCoefficient, ...],
) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(
        (
            int(item.rival_id),
            int(item.numerator),
            int(item.denominator),
        )
        for item in coefficients
    )


def _mapping_payload(
    mapping: OperatorExactReLUPhaseMapping,
) -> dict[str, Any]:
    return {
        "stable_bcol_id": int(mapping.stable_bcol_id),
        "binary_position": int(mapping.binary_position),
        "relu_layer_id": int(mapping.relu_layer_id),
        "stable_output_col_id": int(mapping.stable_output_col_id),
        "output_continuous_position": int(
            mapping.output_continuous_position
        ),
        "lower_upper_row": int(mapping.lower_upper_row),
        "x_branch_upper_row": int(mapping.x_branch_upper_row),
        "zero_branch_upper_row": int(mapping.zero_branch_upper_row),
        "rival_coefficients": _fraction_payload(
            mapping.rival_coefficients
        ),
        "selected_phase": mapping.selected_phase,
    }


def _literal_binding_digest(
    *,
    parent_digest: str,
    property_digest: str,
    stable_bcol_id: int,
    phase: int,
) -> str:
    """Use the existing PC-PCC literal binding schema without cut coupling."""

    return _canonical_sha256(
        {
            "schema": "act.pc_pcc.literal.v1",
            "parent_semantic_digest": parent_digest,
            "property_digest": property_digest,
            "stable_bcol_id": int(stable_bcol_id),
            "phase": int(phase),
        }
    )


def _make_literal(
    *,
    parent_digest: str,
    property_digest: str,
    stable_bcol_id: int,
    phase: int,
) -> PhaseLiteral:
    if phase not in {-1, 1}:
        raise OperatorExactReLUPhaseLiteralError(
            "selected_phase_not_pm1"
        )
    return PhaseLiteral(
        stable_bcol_id=int(stable_bcol_id),
        phase=int(phase),
        binding_digest=_literal_binding_digest(
            parent_digest=parent_digest,
            property_digest=property_digest,
            stable_bcol_id=stable_bcol_id,
            phase=phase,
        ),
    )


def _normalize_rivals(
    rivals: Sequence[RivalSpec],
    *,
    output_width: int,
    caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
    work: list[int],
) -> Tuple[Tuple[RivalSpec, ...], str]:
    if (
        type(rivals) is not tuple
        or not rivals
        or len(rivals) > caps.max_rivals
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "rival_sequence_or_cap_invalid"
        )
    _consume_work(
        work,
        len(rivals) * int(output_width),
        caps=caps,
        deadline=deadline,
        stage="rival_objectives",
    )
    normalized = []
    seen_ids = set()
    for rival in rivals:
        _check_deadline(deadline, stage="rival_normalization")
        if (
            type(rival) is not RivalSpec
            or type(rival.rival_id) is not int
            or rival.rival_id < 0
            or rival.rival_id > _MAX_INT64
            or rival.rival_id in seen_ids
            or type(rival.objective) is not tuple
            or len(rival.objective) != output_width
            or any(
                type(value) is not float
                or not math.isfinite(value)
                for value in rival.objective
            )
            or type(rival.threshold) is not float
            or not math.isfinite(rival.threshold)
            or not _valid_sha256(rival.assert_digest)
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "rival_binding_contract_invalid"
            )
        seen_ids.add(rival.rival_id)
        normalized.append(
            RivalSpec(
                rival_id=rival.rival_id,
                objective=tuple(rival.objective),
                threshold=rival.threshold,
                assert_digest=rival.assert_digest,
            )
        )
    normalized_tuple = tuple(normalized)
    try:
        property_digest = ordered_property_digest(normalized_tuple)
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ) as exc:
        raise OperatorExactReLUPhaseLiteralError(
            "ordered_property_malformed"
        ) from exc
    _check_deadline(deadline, stage="ordered_property_digest")
    return normalized_tuple, property_digest


def _operator_row_tags(
    hz: SparseHZono,
    *,
    caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
    work: list[int],
) -> Tuple[Tuple[str, ...], str]:
    raw_tags = getattr(hz, "_solver_constraint_row_tags", None)
    if (
        type(raw_tags) is not tuple
        or len(raw_tags) != hz.n_eq + hz.n_ub
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "operator_constraint_row_tags_malformed"
        )
    _consume_work(
        work,
        len(raw_tags),
        caps=caps,
        deadline=deadline,
        stage="operator_row_tags",
    )
    if any(type(tag) is not str for tag in raw_tags):
        raise OperatorExactReLUPhaseLiteralError(
            "operator_constraint_row_tags_malformed"
        )
    tags = raw_tags
    if any(tag.startswith(_EXACT_TAG_PREFIX) for tag in tags[: hz.n_eq]):
        raise OperatorExactReLUPhaseLiteralError(
            "exact_relu_tag_in_equality_rows"
        )
    digest = _canonical_sha256(
        {
            "schema": "act.operator_exact_relu_row_tags.v1",
            "n_eq": int(hz.n_eq),
            "n_ub": int(hz.n_ub),
            "tags": tags,
        }
    )
    return tags, digest


def _parse_exact_upper_rows(
    hz: SparseHZono,
    tags: Tuple[str, ...],
    *,
    caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
    work: list[int],
) -> dict[Tuple[int, str], Tuple[int, ...]]:
    grouped: dict[Tuple[int, str], list[int]] = {}
    tagged_row_count = 0
    for upper_row, tag in enumerate(tags[hz.n_eq :]):
        _check_deadline(deadline, stage="exact_row_tag_scan")
        match = _EXACT_TAG.fullmatch(tag)
        if match is None:
            if tag.startswith(_EXACT_TAG_PREFIX):
                raise OperatorExactReLUPhaseLiteralError(
                    "exact_relu_row_tag_noncanonical"
                )
            continue
        kind, raw_layer = match.groups()
        layer_id = int(raw_layer)
        if layer_id < 0 or str(layer_id) != raw_layer:
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_layer_id_noncanonical"
            )
        grouped.setdefault((layer_id, kind), []).append(upper_row)
        tagged_row_count += 1
        if tagged_row_count > 3 * caps.max_binaries:
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_tagged_row_cap_exceeded"
            )
    result = {
        key: tuple(rows)
        for key, rows in grouped.items()
    }
    exact_rows = {
        row
        for rows in result.values()
        for row in rows
    }
    exact_nnz = sum(
        int(hz.Auc.indptr[row + 1] - hz.Auc.indptr[row])
        + int(hz.Aub.indptr[row + 1] - hz.Aub.indptr[row])
        for row in exact_rows
    )
    _consume_work(
        work,
        exact_nnz,
        caps=caps,
        deadline=deadline,
        stage="exact_tagged_row_nnz",
    )
    return result


def _exact_objective_coefficients(
    hz: SparseHZono,
    rivals: Tuple[RivalSpec, ...],
    *,
    continuous_position: int,
    caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
    work: list[int],
) -> Tuple[ExactDyadicRivalCoefficient, ...]:
    column = hz.Gc.getcol(continuous_position).tocoo(copy=False)
    stored = tuple(
        (int(row), float(value))
        for row, value in zip(column.row.tolist(), column.data.tolist())
        if float(value) != 0.0
    )
    if not stored:
        # A preceding materialization can remove every direct suffix
        # coefficient.  Treating that as an exact zero would silently select
        # an early layer using no property information.
        raise OperatorExactReLUPhaseLiteralError(
            "exact_relu_output_has_no_direct_suffix_coefficient"
        )
    _consume_work(
        work,
        len(stored) * len(rivals),
        caps=caps,
        deadline=deadline,
        stage="direct_suffix_exact_accumulation",
    )
    result = []
    for rival in rivals:
        _check_deadline(
            deadline, stage="direct_suffix_rival_accumulation"
        )
        objective = tuple(float(value) for value in rival.objective)
        coefficient = sum(
            (
                Fraction.from_float(objective[row])
                * Fraction.from_float(value)
                for row, value in stored
            ),
            Fraction(0),
        )
        result.append(
            ExactDyadicRivalCoefficient(
                rival_id=int(rival.rival_id),
                numerator=int(coefficient.numerator),
                denominator=int(coefficient.denominator),
            )
        )
    return tuple(result)


def _derive_mappings(
    hz: SparseHZono,
    rivals: Tuple[RivalSpec, ...],
    *,
    tags: Tuple[str, ...],
    caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
    work: list[int],
) -> Tuple[OperatorExactReLUPhaseMapping, ...]:
    if hz.n_bin < 1:
        raise OperatorExactReLUPhaseLiteralError(
            "operator_hz_has_no_exact_relu_binaries"
        )
    if hz.n_bin > caps.max_binaries:
        raise OperatorExactReLUPhaseLiteralError(
            "operator_hz_binary_cap_exceeded"
        )
    if hz.Gb.nnz != 0:
        raise OperatorExactReLUPhaseLiteralError(
            "operator_hz_binary_output_is_not_zero"
        )
    if hz.Ab.nnz != 0:
        raise OperatorExactReLUPhaseLiteralError(
            "exact_relu_binary_used_in_equality"
        )
    if hz.Auc is None or hz.Aub is None or hz.ub is None:
        raise OperatorExactReLUPhaseLiteralError(
            "operator_hz_upper_rows_missing"
        )

    stable_continuous_ids = _valid_stable_ids(
        hz.col_ids,
        expected=hz.n_cont,
        name="stable_continuous_ids",
    )
    stable_binary_ids = _valid_stable_ids(
        hz.bcol_ids,
        expected=hz.n_bin,
        name="stable_binary_ids",
    )
    grouped = _parse_exact_upper_rows(
        hz,
        tags,
        caps=caps,
        deadline=deadline,
        work=work,
    )

    binary_rows: list[list[int]] = [
        [] for _ in range(hz.n_bin)
    ]
    _consume_work(
        work,
        int(hz.Aub.nnz),
        caps=caps,
        deadline=deadline,
        stage="binary_row_incidence",
    )
    for row in range(hz.n_ub):
        _check_deadline(deadline, stage="binary_row_scan")
        binary_indices, _ = _row_entries(hz.Aub, row)
        for position in binary_indices:
            binary_rows[position].append(row)

    lower_lookup: dict[
        Tuple[int, Tuple[int, ...], Tuple[str, ...]],
        list[int],
    ] = {}
    for (layer_id, kind), lower_rows in grouped.items():
        if kind != "lower":
            continue
        for lower_row in lower_rows:
            _check_deadline(deadline, stage="lower_row_index")
            lower_binary_indices, _ = _row_entries(
                hz.Aub, lower_row
            )
            if lower_binary_indices:
                raise OperatorExactReLUPhaseLiteralError(
                    "exact_relu_lower_row_has_binary"
                )
            indices, values = _row_signature(
                _row_entries(hz.Auc, lower_row)
            )
            lower_lookup.setdefault(
                (layer_id, indices, values), []
            ).append(lower_row)

    consumed_lower: set[int] = set()
    consumed_x: set[int] = set()
    consumed_zero: set[int] = set()
    output_positions: set[int] = set()
    mappings = []

    for binary_position, stable_binary_id in enumerate(
        stable_binary_ids
    ):
        _check_deadline(deadline, stage="exact_binary_mapping")
        rows = tuple(binary_rows[binary_position])
        if len(rows) != 2:
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_binary_row_count_not_two"
            )
        classified = {}
        layer_id = None
        for row in rows:
            tag = tags[hz.n_eq + row]
            match = _EXACT_TAG.fullmatch(tag)
            if match is None or match.group(1) not in {
                "x_branch",
                "zero_branch",
            }:
                raise OperatorExactReLUPhaseLiteralError(
                    "exact_relu_binary_has_noncanonical_row"
                )
            row_layer = int(match.group(2))
            if layer_id is None:
                layer_id = row_layer
            elif layer_id != row_layer:
                raise OperatorExactReLUPhaseLiteralError(
                    "exact_relu_binary_rows_cross_layers"
                )
            kind = match.group(1)
            if kind in classified:
                raise OperatorExactReLUPhaseLiteralError(
                    "exact_relu_binary_row_kind_ambiguous"
                )
            classified[kind] = row
        if set(classified) != {"x_branch", "zero_branch"}:
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_binary_branch_pair_incomplete"
            )
        if layer_id is None:
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_binary_layer_missing"
            )
        x_row = int(classified["x_branch"])
        zero_row = int(classified["zero_branch"])

        x_binary_indices, x_binary_values = _row_entries(
            hz.Aub, x_row
        )
        zero_binary_indices, zero_binary_values = _row_entries(
            hz.Aub, zero_row
        )
        if (
            x_binary_indices != (binary_position,)
            or zero_binary_indices != (binary_position,)
            or len(x_binary_values) != 1
            or len(zero_binary_values) != 1
            or not (x_binary_values[0] > 0.0)
            or not (zero_binary_values[0] < 0.0)
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_binary_branch_coefficients_noncanonical"
            )

        zero_continuous = _row_entries(hz.Auc, zero_row)
        zero_indices, zero_values = zero_continuous
        if len(zero_indices) != 1 or not (zero_values[0] > 0.0):
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_zero_branch_output_noncanonical"
            )
        output_position = zero_indices[0]
        # ``upper_half`` and the normalized y radius are independently
        # computed from the same outward upper bound.  They can differ by one
        # final binary64 rounding step, but no wider structural mismatch is
        # accepted.
        if not _stored_scales_within_ulps(
            zero_values[0],
            -zero_binary_values[0],
            max_ulps=2,
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_zero_branch_scale_mismatch"
            )

        x_continuous = _row_entries(hz.Auc, x_row)
        x_indices, x_values = x_continuous
        if len(x_indices) < 2 or output_position not in x_indices:
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_x_branch_output_missing"
            )
        x_output_value = x_values[x_indices.index(output_position)]
        if (
            Fraction.from_float(x_output_value)
            != Fraction.from_float(zero_values[0])
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_branch_output_scale_mismatch"
            )

        negative_indices, negative_values = _negative_row_signature(
            x_continuous
        )
        lower_candidates = lower_lookup.get(
            (layer_id, negative_indices, negative_values), ()
        )
        if len(lower_candidates) != 1:
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_lower_row_missing_or_ambiguous"
            )
        lower_row = int(lower_candidates[0])
        if (
            lower_row in consumed_lower
            or x_row in consumed_x
            or zero_row in consumed_zero
            or output_position in output_positions
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "exact_relu_row_or_output_mapping_reused"
            )

        coefficients = _exact_objective_coefficients(
            hz,
            rivals,
            continuous_position=output_position,
            caps=caps,
            deadline=deadline,
            work=work,
        )
        signs = tuple(
            1 if item.value > 0 else -1 if item.value < 0 else 0
            for item in coefficients
        )
        if all(sign == 1 for sign in signs):
            phase: Optional[int] = 1
        elif all(sign == -1 for sign in signs):
            phase = -1
        elif all(sign == 0 for sign in signs):
            phase = None
        else:
            raise OperatorExactReLUPhaseLiteralError(
                "ordered_rivals_disagree_on_exact_relu_phase"
            )

        mappings.append(
            OperatorExactReLUPhaseMapping(
                stable_bcol_id=int(stable_binary_id),
                binary_position=int(binary_position),
                relu_layer_id=int(layer_id),
                stable_output_col_id=int(
                    stable_continuous_ids[output_position]
                ),
                output_continuous_position=int(output_position),
                lower_upper_row=lower_row,
                x_branch_upper_row=x_row,
                zero_branch_upper_row=zero_row,
                rival_coefficients=coefficients,
                selected_phase=phase,
            )
        )
        consumed_lower.add(lower_row)
        consumed_x.add(x_row)
        consumed_zero.add(zero_row)
        output_positions.add(output_position)

    all_lower = {
        row
        for (layer_id, kind), rows in grouped.items()
        if kind == "lower"
        for row in rows
    }
    all_x = {
        row
        for (layer_id, kind), rows in grouped.items()
        if kind == "x_branch"
        for row in rows
    }
    all_zero = {
        row
        for (layer_id, kind), rows in grouped.items()
        if kind == "zero_branch"
        for row in rows
    }
    if (
        consumed_lower != all_lower
        or consumed_x != all_x
        or consumed_zero != all_zero
        or len(consumed_lower) != hz.n_bin
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "exact_relu_tagged_rows_not_bijectively_consumed"
        )
    _check_deadline(deadline, stage="mapping_completion")
    return tuple(
        sorted(mappings, key=lambda item: item.stable_bcol_id)
    )


def _derive_operator_exact_relu_property_phase_literals_with_caps(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    *,
    caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
) -> OperatorExactReLUPhaseSelection:
    """Internal derivation under caller-validated caps and one deadline."""

    work = [0]
    if type(build) is not OperatorHZBuild:
        raise OperatorExactReLUPhaseLiteralError(
            "build_not_operator_hz_result"
        )
    if build.property_upper_output is not False:
        raise OperatorExactReLUPhaseLiteralError(
            "property_upper_output_unsupported"
        )
    hz = build.hz
    if type(hz) is not SparseHZono:
        raise OperatorExactReLUPhaseLiteralError(
            "operator_hz_parent_not_sparse_hz"
        )
    try:
        parent_digest = sparse_hz_semantic_digest(hz)
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ) as exc:
        raise OperatorExactReLUPhaseLiteralError(
            "operator_hz_parent_semantics_malformed"
        ) from exc
    _check_deadline(deadline, stage="parent_semantic_digest")
    if hz.n_bin > caps.max_binaries:
        raise OperatorExactReLUPhaseLiteralError(
            "operator_hz_binary_cap_exceeded"
        )
    tags, row_tag_digest = _operator_row_tags(
        hz,
        caps=caps,
        deadline=deadline,
        work=work,
    )
    normalized_rivals, property_digest = _normalize_rivals(
        rivals,
        output_width=hz.n_out,
        caps=caps,
        deadline=deadline,
        work=work,
    )
    mappings = _derive_mappings(
        hz,
        normalized_rivals,
        tags=tags,
        caps=caps,
        deadline=deadline,
        work=work,
    )

    literals = tuple(
        _make_literal(
            parent_digest=parent_digest,
            property_digest=property_digest,
            stable_bcol_id=mapping.stable_bcol_id,
            phase=int(mapping.selected_phase),
        )
        for mapping in mappings
        if mapping.selected_phase is not None
    )
    omissions = tuple(
        OperatorExactReLUZeroOmission(
            stable_bcol_id=mapping.stable_bcol_id,
            stable_output_col_id=mapping.stable_output_col_id,
            rival_coefficients=mapping.rival_coefficients,
        )
        for mapping in mappings
        if mapping.selected_phase is None
    )
    status = (
        "selected_with_exact_zero_omissions"
        if omissions
        else "selected"
    )
    selection_payload = {
        "schema": "act.operator_exact_relu_phase_selection.v2",
        "status": status,
        "parent_semantic_digest": parent_digest,
        "operator_row_tag_digest": row_tag_digest,
        "property_digest": property_digest,
        "caps": _caps_payload(caps),
        "mappings": tuple(_mapping_payload(item) for item in mappings),
        "literals": tuple(
            (
                int(item.stable_bcol_id),
                int(item.phase),
                item.binding_digest,
            )
            for item in literals
        ),
        "zero_omissions": tuple(
            (
                int(item.stable_bcol_id),
                int(item.stable_output_col_id),
                _fraction_payload(item.rival_coefficients),
                item.reason,
            )
            for item in omissions
        ),
        "arithmetic": "Fraction.from_float_exact_dyadic",
        "proof_authority": False,
    }
    _check_deadline(deadline, stage="selection_payload")
    return OperatorExactReLUPhaseSelection(
        status=status,
        mappings=mappings,
        literals=literals,
        zero_omissions=omissions,
        parent_semantic_digest=parent_digest,
        operator_row_tag_digest=row_tag_digest,
        property_digest=property_digest,
        caps=caps,
        selection_digest=_canonical_sha256(selection_payload),
        proof_authority=False,
    )


def derive_operator_exact_relu_property_phase_literals(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    *,
    max_rivals: int = _DEFAULT_MAX_RIVALS,
    max_binaries: int = _DEFAULT_MAX_BINARIES,
    max_work_items: int = _DEFAULT_MAX_WORK_ITEMS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> OperatorExactReLUPhaseSelection:
    """Derive exact-ReLU phase candidates from one live Operator-HZ build.

    The result is diagnostic and selection-only.  It must not authorize a cut,
    pruning, SAFE, or UNSAFE verdict.  Caps and timeout are bound into the
    returned selection and must be supplied identically to the live verifier.
    """

    caps = _normalize_caps(
        max_rivals=max_rivals,
        max_binaries=max_binaries,
        max_work_items=max_work_items,
        timeout_seconds=timeout_seconds,
    )
    return _derive_operator_exact_relu_property_phase_literals_with_caps(
        build,
        rivals,
        caps=caps,
        deadline=time.monotonic() + caps.timeout_seconds,
    )


def _exact_coefficient_payload(
    coefficients: Any,
    *,
    rival_ids: Tuple[int, ...],
    caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
    work: list[int],
) -> Tuple[Tuple[int, int, int], ...]:
    """Validate nested coefficient records without candidate equality."""

    if (
        type(coefficients) is not tuple
        or len(coefficients) != len(rival_ids)
        or len(coefficients) > caps.max_rivals
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "result_coefficient_tuple_noncanonical"
        )
    _consume_work(
        work,
        len(coefficients),
        caps=caps,
        deadline=deadline,
        stage="result_coefficients",
    )
    payload = []
    for coefficient in coefficients:
        if type(coefficient) is not ExactDyadicRivalCoefficient:
            raise OperatorExactReLUPhaseLiteralError(
                "result_coefficient_wrong_type"
            )
        rival_id = coefficient.rival_id
        numerator = coefficient.numerator
        denominator = coefficient.denominator
        if (
            type(rival_id) is not int
            or type(numerator) is not int
            or type(denominator) is not int
            or rival_id < 0
            or rival_id > _MAX_INT64
            or denominator <= 0
            or abs(numerator).bit_length()
            > _HARD_EXACT_INTEGER_BITS
            or denominator.bit_length()
            > _HARD_EXACT_INTEGER_BITS
            or math.gcd(abs(numerator), denominator) != 1
            or (numerator == 0 and denominator != 1)
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "result_coefficient_noncanonical"
            )
        payload.append((rival_id, numerator, denominator))
        _check_deadline(deadline, stage="result_coefficient")
    payload_tuple = tuple(payload)
    if tuple(item[0] for item in payload_tuple) != rival_ids:
        raise OperatorExactReLUPhaseLiteralError(
            "result_coefficient_rival_binding_mismatch"
        )
    return payload_tuple


def _exact_result_payload(
    result: Any,
    *,
    hz: SparseHZono,
    rival_ids: Tuple[int, ...],
    expected_caps: OperatorExactReLUPhaseSelectionCaps,
    deadline: float,
) -> dict[str, Any]:
    """Rebuild one result solely from exact built-in primitive fields."""

    if type(result) is not OperatorExactReLUPhaseSelection:
        raise OperatorExactReLUPhaseLiteralError(
            "result_wrong_exact_type"
        )
    if hz.n_bin > expected_caps.max_binaries:
        raise OperatorExactReLUPhaseLiteralError(
            "result_parent_binary_cap_exceeded"
        )
    if (
        type(result.status) is not str
        or type(result.mappings) is not tuple
        or type(result.literals) is not tuple
        or type(result.zero_omissions) is not tuple
        or type(result.arithmetic) is not str
        or result.proof_authority is not False
        or not _valid_sha256(result.parent_semantic_digest)
        or not _valid_sha256(result.operator_row_tag_digest)
        or not _valid_sha256(result.property_digest)
        or not _valid_sha256(result.selection_digest)
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "result_top_level_noncanonical"
        )
    if result.arithmetic != "Fraction.from_float_exact_dyadic":
        raise OperatorExactReLUPhaseLiteralError(
            "result_arithmetic_mismatch"
        )
    actual_caps_payload = _caps_payload(result.caps)
    expected_caps_payload = _caps_payload(expected_caps)
    if _canonical_bytes(actual_caps_payload) != _canonical_bytes(
        expected_caps_payload
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "result_caps_binding_mismatch"
        )
    if (
        len(result.mappings) < 1
        or len(result.mappings) > expected_caps.max_binaries
        or len(result.literals) > len(result.mappings)
        or len(result.zero_omissions) > len(result.mappings)
        or len(result.literals) + len(result.zero_omissions)
        != len(result.mappings)
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "result_collection_sizes_invalid"
        )

    work = [0]
    _consume_work(
        work,
        12 * len(result.mappings)
        + 4 * len(result.literals)
        + 6 * len(result.zero_omissions),
        caps=expected_caps,
        deadline=deadline,
        stage="result_structure",
    )
    mapping_payloads = []
    mapping_coefficients: dict[
        int, Tuple[Tuple[int, int, int], ...]
    ] = {}
    mapping_output_ids: dict[int, int] = {}
    selected_pairs = []
    omitted_ids = []
    binary_positions = []
    output_positions = []
    lower_rows = []
    x_rows = []
    zero_rows = []
    mapping_ids = []
    for mapping in result.mappings:
        if type(mapping) is not OperatorExactReLUPhaseMapping:
            raise OperatorExactReLUPhaseLiteralError(
                "result_mapping_wrong_type"
            )
        integer_fields = (
            mapping.stable_bcol_id,
            mapping.binary_position,
            mapping.relu_layer_id,
            mapping.stable_output_col_id,
            mapping.output_continuous_position,
            mapping.lower_upper_row,
            mapping.x_branch_upper_row,
            mapping.zero_branch_upper_row,
        )
        if (
            any(type(value) is not int for value in integer_fields)
            or mapping.stable_bcol_id < 0
            or mapping.stable_bcol_id > _MAX_INT64
            or mapping.binary_position < 0
            or mapping.binary_position >= hz.n_bin
            or mapping.relu_layer_id < 0
            or mapping.relu_layer_id > _MAX_INT64
            or mapping.stable_output_col_id < 0
            or mapping.stable_output_col_id > _MAX_INT64
            or mapping.output_continuous_position < 0
            or mapping.output_continuous_position >= hz.n_cont
            or mapping.lower_upper_row < 0
            or mapping.lower_upper_row >= hz.n_ub
            or mapping.x_branch_upper_row < 0
            or mapping.x_branch_upper_row >= hz.n_ub
            or mapping.zero_branch_upper_row < 0
            or mapping.zero_branch_upper_row >= hz.n_ub
            or (
                mapping.selected_phase is not None
                and (
                    type(mapping.selected_phase) is not int
                    or mapping.selected_phase not in {-1, 1}
                )
            )
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "result_mapping_noncanonical"
            )
        coefficients = _exact_coefficient_payload(
            mapping.rival_coefficients,
            rival_ids=rival_ids,
            caps=expected_caps,
            deadline=deadline,
            work=work,
        )
        mapping_ids.append(mapping.stable_bcol_id)
        binary_positions.append(mapping.binary_position)
        output_positions.append(mapping.output_continuous_position)
        lower_rows.append(mapping.lower_upper_row)
        x_rows.append(mapping.x_branch_upper_row)
        zero_rows.append(mapping.zero_branch_upper_row)
        mapping_coefficients[mapping.stable_bcol_id] = coefficients
        mapping_output_ids[
            mapping.stable_bcol_id
        ] = mapping.stable_output_col_id
        if mapping.selected_phase is None:
            omitted_ids.append(mapping.stable_bcol_id)
        else:
            selected_pairs.append(
                (
                    mapping.stable_bcol_id,
                    mapping.selected_phase,
                )
            )
        mapping_payloads.append(
            {
                "stable_bcol_id": mapping.stable_bcol_id,
                "binary_position": mapping.binary_position,
                "relu_layer_id": mapping.relu_layer_id,
                "stable_output_col_id": (
                    mapping.stable_output_col_id
                ),
                "output_continuous_position": (
                    mapping.output_continuous_position
                ),
                "lower_upper_row": mapping.lower_upper_row,
                "x_branch_upper_row": mapping.x_branch_upper_row,
                "zero_branch_upper_row": (
                    mapping.zero_branch_upper_row
                ),
                "rival_coefficients": coefficients,
                "selected_phase": mapping.selected_phase,
            }
        )
        _check_deadline(deadline, stage="result_mapping")

    mapping_id_tuple = tuple(mapping_ids)
    if (
        mapping_id_tuple != tuple(sorted(mapping_id_tuple))
        or len(set(mapping_id_tuple)) != len(mapping_id_tuple)
        or len(set(binary_positions)) != len(binary_positions)
        or set(binary_positions) != set(range(hz.n_bin))
        or len(set(output_positions)) != len(output_positions)
        or len(set(lower_rows)) != len(lower_rows)
        or len(set(x_rows)) != len(x_rows)
        or len(set(zero_rows)) != len(zero_rows)
        or set(lower_rows) & set(x_rows)
        or set(lower_rows) & set(zero_rows)
        or set(x_rows) & set(zero_rows)
    ):
        raise OperatorExactReLUPhaseLiteralError(
            "result_mapping_partition_noncanonical"
        )

    literal_payloads = []
    literal_pairs = []
    for literal in result.literals:
        if type(literal) is not PhaseLiteral:
            raise OperatorExactReLUPhaseLiteralError(
                "result_literal_wrong_type"
            )
        if (
            type(literal.stable_bcol_id) is not int
            or literal.stable_bcol_id < 0
            or literal.stable_bcol_id > _MAX_INT64
            or type(literal.phase) is not int
            or literal.phase not in {-1, 1}
            or not _valid_sha256(literal.binding_digest)
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "result_literal_noncanonical"
            )
        expected_binding = _literal_binding_digest(
            parent_digest=result.parent_semantic_digest,
            property_digest=result.property_digest,
            stable_bcol_id=literal.stable_bcol_id,
            phase=literal.phase,
        )
        if literal.binding_digest != expected_binding:
            raise OperatorExactReLUPhaseLiteralError(
                "result_literal_binding_mismatch"
            )
        literal_pairs.append(
            (literal.stable_bcol_id, literal.phase)
        )
        literal_payloads.append(
            (
                literal.stable_bcol_id,
                literal.phase,
                literal.binding_digest,
            )
        )
        _check_deadline(deadline, stage="result_literal")
    if tuple(literal_pairs) != tuple(selected_pairs):
        raise OperatorExactReLUPhaseLiteralError(
            "result_literal_mapping_mismatch"
        )

    omission_payloads = []
    omission_ids = []
    for omission in result.zero_omissions:
        if type(omission) is not OperatorExactReLUZeroOmission:
            raise OperatorExactReLUPhaseLiteralError(
                "result_omission_wrong_type"
            )
        if (
            type(omission.stable_bcol_id) is not int
            or omission.stable_bcol_id < 0
            or omission.stable_bcol_id > _MAX_INT64
            or type(omission.stable_output_col_id) is not int
            or omission.stable_output_col_id < 0
            or omission.stable_output_col_id > _MAX_INT64
            or type(omission.reason) is not str
            or omission.reason != _ZERO_OMISSION_REASON
            or omission.proof_authority is not False
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "result_omission_noncanonical"
            )
        coefficients = _exact_coefficient_payload(
            omission.rival_coefficients,
            rival_ids=rival_ids,
            caps=expected_caps,
            deadline=deadline,
            work=work,
        )
        stable_id = omission.stable_bcol_id
        if (
            stable_id not in mapping_coefficients
            or mapping_output_ids[stable_id]
            != omission.stable_output_col_id
            or mapping_coefficients[stable_id] != coefficients
            or any(
                numerator != 0 or denominator != 1
                for _, numerator, denominator in coefficients
            )
        ):
            raise OperatorExactReLUPhaseLiteralError(
                "result_omission_mapping_mismatch"
            )
        omission_ids.append(stable_id)
        omission_payloads.append(
            (
                stable_id,
                omission.stable_output_col_id,
                coefficients,
                omission.reason,
            )
        )
        _check_deadline(deadline, stage="result_omission")
    if tuple(omission_ids) != tuple(omitted_ids):
        raise OperatorExactReLUPhaseLiteralError(
            "result_omission_partition_mismatch"
        )

    expected_status = (
        "selected_with_exact_zero_omissions"
        if omission_payloads
        else "selected"
    )
    if result.status != expected_status:
        raise OperatorExactReLUPhaseLiteralError(
            "result_status_mismatch"
        )
    payload = {
        "schema": "act.operator_exact_relu_phase_selection.v2",
        "status": result.status,
        "parent_semantic_digest": result.parent_semantic_digest,
        "operator_row_tag_digest": (
            result.operator_row_tag_digest
        ),
        "property_digest": result.property_digest,
        "caps": actual_caps_payload,
        "mappings": tuple(mapping_payloads),
        "literals": tuple(literal_payloads),
        "zero_omissions": tuple(omission_payloads),
        "arithmetic": result.arithmetic,
        "proof_authority": False,
    }
    if _canonical_sha256(payload) != result.selection_digest:
        raise OperatorExactReLUPhaseLiteralError(
            "result_selection_digest_mismatch"
        )
    _check_deadline(deadline, stage="result_payload")
    return payload


def verify_operator_exact_relu_property_phase_selection(
    build: OperatorHZBuild,
    rivals: Sequence[RivalSpec],
    result: OperatorExactReLUPhaseSelection,
    *,
    max_rivals: int = _DEFAULT_MAX_RIVALS,
    max_binaries: int = _DEFAULT_MAX_BINARIES,
    max_work_items: int = _DEFAULT_MAX_WORK_ITEMS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Live re-derivation without invoking candidate-controlled equality."""

    try:
        caps = _normalize_caps(
            max_rivals=max_rivals,
            max_binaries=max_binaries,
            max_work_items=max_work_items,
            timeout_seconds=timeout_seconds,
        )
        deadline = time.monotonic() + caps.timeout_seconds
        if (
            type(build) is not OperatorHZBuild
            or type(build.hz) is not SparseHZono
        ):
            return False
        normalization_work = [0]
        normalized_rivals, _ = _normalize_rivals(
            rivals,
            output_width=build.hz.n_out,
            caps=caps,
            deadline=deadline,
            work=normalization_work,
        )
        rival_ids = tuple(
            rival.rival_id for rival in normalized_rivals
        )
        actual_payload = _exact_result_payload(
            result,
            hz=build.hz,
            rival_ids=rival_ids,
            expected_caps=caps,
            deadline=deadline,
        )
        expected = (
            _derive_operator_exact_relu_property_phase_literals_with_caps(
                build,
                normalized_rivals,
                caps=caps,
                deadline=deadline,
            )
        )
        expected_payload = _exact_result_payload(
            expected,
            hz=build.hz,
            rival_ids=rival_ids,
            expected_caps=caps,
            deadline=deadline,
        )
        _check_deadline(deadline, stage="verification_completion")
        return _canonical_bytes(actual_payload) == _canonical_bytes(
            expected_payload
        )
    except (
        OperatorExactReLUPhaseLiteralError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


__all__ = [
    "ExactDyadicRivalCoefficient",
    "OperatorExactReLUPhaseLiteralError",
    "OperatorExactReLUPhaseMapping",
    "OperatorExactReLUPhaseSelection",
    "OperatorExactReLUPhaseSelectionCaps",
    "OperatorExactReLUZeroOmission",
    "derive_operator_exact_relu_property_phase_literals",
    "verify_operator_exact_relu_property_phase_selection",
]
