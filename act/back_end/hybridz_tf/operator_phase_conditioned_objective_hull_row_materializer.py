#!/usr/bin/env python3
"""Toy-first sound binary64 row formation for an exact PCOH descriptor.

This module does not modify a :class:`SparseHZono` and has no verifier or
verdict authority.  It maps the exact-Fraction rows of a structurally verified
phase-conditioned objective hull into a local split CSR frame:

``continuous_eta``
    Parent continuous columns followed by descriptor-local eta columns.

``binary``
    Parent binary columns in the caller's live stable-id order.

Equality coefficients and right-hand sides must be finite and bit-exact in
binary64.  For an upper row ``a*x <= b``, every coefficient is rounded to a
finite binary64 ``a_hat`` and the exact box error
``sum(abs(a_hat - a))`` is added to the right-hand side.  All parent and eta
variables lie in ``[-1, 1]``, so

``a*x <= b  =>  a_hat*x <= b + sum(abs(a_hat-a))``.

The guarded exact right-hand side is finally rounded toward positive infinity.
No sparse hstack/vstack or full parent constraint frame is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    ExactHZLinearRow,
    PhaseConditionedObjectiveHullDescriptor,
    outward_float64,
    verify_phase_conditioned_objective_hull,
)


_FRAME_SCHEMA = "act.hybridz_pc_objective_hull_binary64_row_frame.v1"
_RECEIPT_SCHEMA = (
    "act.hybridz_pc_objective_hull_binary64_row_frame_receipt.v1"
)


class PhaseConditionedObjectiveHullRowMaterializationError(ValueError):
    """An exact descriptor row cannot be safely represented in binary64."""


@dataclass(frozen=True)
class PCOHRowMaterializationCaps:
    max_parent_continuous_columns: int = 5_000_000
    max_parent_binary_columns: int = 65_536
    max_eta_columns: int = 16
    max_rows: int = 64
    max_total_exact_nonzeros: int = 5_000_000
    max_exact_bits: int = 16_384


@dataclass(frozen=True)
class CoefficientFormationError:
    """Exact error for one upper-row coefficient conversion."""

    group: str
    identifier: int
    exact: Fraction
    stored: float
    absolute_error: Fraction


@dataclass(frozen=True)
class UpperRowFormationGuard:
    """Exact coefficient and RHS formation record for one upper row."""

    row_index: int
    row_name: str
    coefficient_errors: Tuple[CoefficientFormationError, ...]
    total_coefficient_guard: Fraction
    raw_rhs_exact: Fraction
    guarded_rhs_exact: Fraction
    stored_rhs: float
    rhs_outward_error: Fraction


@dataclass(frozen=True)
class PCOHBinary64LocalRowFrame:
    """Four-block local CSR frame; never a live or verdict-capable HZ."""

    schema: str
    descriptor_representation_sha256: str
    parent_semantic_digest: str
    objective_binding_sha256: str
    parent_col_ids_sha256: str
    parent_bcol_ids_sha256: str
    parent_continuous_columns: int
    parent_binary_columns: int
    eta_columns: int
    equality_continuous_eta: sp.csr_matrix
    equality_binary: sp.csr_matrix
    upper_continuous_eta: sp.csr_matrix
    upper_binary: sp.csr_matrix
    equality_rhs: np.ndarray
    upper_rhs: np.ndarray
    upper_row_guards: Tuple[UpperRowFormationGuard, ...]
    block_sha256: Mapping[str, str]
    frame_sha256: str
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    verdict_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority is not False:
            raise ValueError("row frame never has proof authority")
        if self.verdict_authority is not False:
            raise ValueError("row frame never has verdict authority")


_DEFAULT_CAPS = PCOHRowMaterializationCaps()
_HARD_CAPS = PCOHRowMaterializationCaps(
    max_parent_continuous_columns=20_000_000,
    max_parent_binary_columns=1_000_000,
    max_eta_columns=16,
    max_rows=128,
    max_total_exact_nonzeros=20_000_000,
    max_exact_bits=65_536,
)

_ID_VALIDATION_CHUNK = 65_536


def _canonical_form(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PhaseConditionedObjectiveHullRowMaterializationError(
                "canonical_nonfinite_float"
            )
        return {"__binary64_hex__": value.hex()}
    if type(value) is Fraction:
        return {"__fraction__": [value.numerator, value.denominator]}
    if isinstance(value, np.generic):
        return _canonical_form(value.item())
    if type(value) in {tuple, list}:
        return [_canonical_form(item) for item in value]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PhaseConditionedObjectiveHullRowMaterializationError(
                    "canonical_nonstring_key"
                )
            result[key] = _canonical_form(item)
        return result
    raise PhaseConditionedObjectiveHullRowMaterializationError(
        f"canonical_unsupported_{type(value).__name__}"
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
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "canonical_encoding_failed"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) in {tuple, list}:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _caps_payload(caps: PCOHRowMaterializationCaps) -> Dict[str, int]:
    return {
        "max_parent_continuous_columns": (
            caps.max_parent_continuous_columns
        ),
        "max_parent_binary_columns": caps.max_parent_binary_columns,
        "max_eta_columns": caps.max_eta_columns,
        "max_rows": caps.max_rows,
        "max_total_exact_nonzeros": caps.max_total_exact_nonzeros,
        "max_exact_bits": caps.max_exact_bits,
    }


def _normalize_caps(caps: Any) -> PCOHRowMaterializationCaps:
    if type(caps) is not PCOHRowMaterializationCaps:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "caps_wrong_type"
        )
    values = _caps_payload(caps)
    hard = _caps_payload(_HARD_CAPS)
    for name, value in values.items():
        if type(value) is not int or value < 1 or value > hard[name]:
            raise PhaseConditionedObjectiveHullRowMaterializationError(
                f"cap_invalid:{name}"
            )
    return caps


def _strict_deadline(value: Any) -> float:
    if type(value) not in {int, float}:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "deadline_not_builtin_numeric"
        )
    deadline = float(value)
    if not math.isfinite(deadline):
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "deadline_nonfinite"
        )
    return deadline


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            f"deadline_exhausted:{stage}"
        )


@dataclass(frozen=True)
class _StableIdLookup:
    """Array-backed stable-id lookup without per-parent-id Python objects."""

    snapshot: np.ndarray
    indirect_order: np.ndarray | None
    name: str

    def position(self, identifier: int, *, deadline: float) -> int | None:
        if self.indirect_order is None:
            # NumPy's binary search is the only bulk operation in this path.
            candidate = int(np.searchsorted(self.snapshot, identifier))
            _check_deadline(deadline, f"{self.name}_searchsorted")
            if (
                candidate >= self.snapshot.size
                or int(self.snapshot[candidate]) != identifier
            ):
                return None
            return candidate

        # Keep the unsorted fallback at two int64-sized arrays: the immutable
        # caller-order snapshot and an argsort permutation.  Indexing the
        # permutation one scalar at a time avoids materializing a third,
        # full-size sorted-id vector.
        lower = 0
        upper = int(self.indirect_order.size)
        while lower < upper:
            middle = (lower + upper) // 2
            original = int(self.indirect_order[middle])
            if int(self.snapshot[original]) < identifier:
                lower = middle + 1
            else:
                upper = middle
        _check_deadline(deadline, f"{self.name}_indirect_search")
        if lower >= self.indirect_order.size:
            return None
        original = int(self.indirect_order[lower])
        if int(self.snapshot[original]) != identifier:
            return None
        return original


def _strict_id_array(
    value: Any,
    *,
    name: str,
    maximum_columns: int,
    deadline: float,
) -> Tuple[np.ndarray, _StableIdLookup]:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.int64)
        or value.ndim != 1
        or value.size > maximum_columns
        or not value.flags.c_contiguous
    ):
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            f"{name}_not_canonical_int64_vector"
        )
    snapshot = value.copy()
    _check_deadline(deadline, f"{name}_snapshot")
    snapshot.setflags(write=False)

    strictly_increasing = True
    previous_last = None
    for start in range(0, int(snapshot.size), _ID_VALIDATION_CHUNK):
        chunk = snapshot[start : start + _ID_VALIDATION_CHUNK]
        has_negative = bool(np.any(chunk < 0))
        _check_deadline(deadline, f"{name}_nonnegative_validation")
        if has_negative:
            raise PhaseConditionedObjectiveHullRowMaterializationError(
                f"{name}_negative_or_duplicate"
            )
        if previous_last is not None and int(chunk[0]) <= previous_last:
            strictly_increasing = False
        if chunk.size > 1:
            chunk_increasing = not bool(
                np.any(chunk[1:] <= chunk[:-1])
            )
            _check_deadline(deadline, f"{name}_order_validation")
            strictly_increasing = strictly_increasing and chunk_increasing
        previous_last = int(chunk[-1])

    if strictly_increasing:
        return snapshot, _StableIdLookup(snapshot, None, name)

    # Quicksort's returned permutation is the only full-size fallback
    # allocation.  Validate duplicates through bounded gathered chunks rather
    # than building a second full sorted copy.
    indirect_order = np.argsort(snapshot, kind="quicksort")
    _check_deadline(deadline, f"{name}_argsort")
    previous_last = None
    for start in range(0, int(snapshot.size), _ID_VALIDATION_CHUNK):
        order_chunk = indirect_order[start : start + _ID_VALIDATION_CHUNK]
        sorted_chunk = snapshot[order_chunk]
        _check_deadline(deadline, f"{name}_sorted_chunk_gather")
        duplicate = (
            previous_last is not None
            and int(sorted_chunk[0]) == previous_last
        )
        if sorted_chunk.size > 1:
            duplicate = duplicate or bool(
                np.any(sorted_chunk[1:] == sorted_chunk[:-1])
            )
            _check_deadline(deadline, f"{name}_duplicate_validation")
        previous_last = int(sorted_chunk[-1])
        if duplicate:
            raise PhaseConditionedObjectiveHullRowMaterializationError(
                f"{name}_negative_or_duplicate"
            )
    indirect_order.setflags(write=False)
    return snapshot, _StableIdLookup(snapshot, indirect_order, name)


def _id_vector_sha256(
    name: str, value: np.ndarray, *, deadline: float
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{name}:int64_le:v1:".encode("ascii"))
    digest.update(np.asarray(value.size, dtype="<i8").tobytes())
    for start in range(0, int(value.size), _ID_VALIDATION_CHUNK):
        chunk = value[start : start + _ID_VALIDATION_CHUNK]
        little_endian = chunk.astype("<i8", copy=False)
        digest.update(memoryview(little_endian).cast("B"))
        _check_deadline(deadline, f"{name}_sha256")
    return digest.hexdigest()


def _fraction_bits(value: Fraction) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


def _check_fraction_bits(
    value: Fraction,
    *,
    name: str,
    caps: PCOHRowMaterializationCaps,
) -> None:
    if type(value) is not Fraction:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            f"{name}_not_fraction"
        )
    if _fraction_bits(value) > caps.max_exact_bits:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            f"{name}_exact_bit_cap_exceeded"
        )


def _fraction_to_finite_binary64(
    value: Fraction,
    *,
    name: str,
) -> float:
    try:
        stored = float(value)
    except (OverflowError, ValueError) as exc:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            f"{name}_binary64_overflow"
        ) from exc
    if not math.isfinite(stored):
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            f"{name}_binary64_nonfinite"
        )
    return stored


def _fraction_to_exact_binary64(
    value: Fraction,
    *,
    name: str,
) -> float:
    stored = _fraction_to_finite_binary64(value, name=name)
    if Fraction.from_float(stored) != value:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            f"{name}_not_bit_exact_binary64"
        )
    return stored


def _freeze_csr(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    if not matrix.has_canonical_format:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "constructed_csr_not_canonical"
        )
    for value in (matrix.data, matrix.indices, matrix.indptr):
        value.setflags(write=False)
    return matrix


def _rows_to_csr(
    rows: Sequence[Sequence[Tuple[int, float]]],
    *,
    columns: int,
    deadline: float,
) -> sp.csr_matrix:
    indptr = [0]
    indices = []
    data = []
    for row in rows:
        _check_deadline(deadline, "csr_row_formation")
        previous = -1
        for column, coefficient in sorted(row):
            if (
                type(column) is not int
                or column < 0
                or column >= columns
                or column <= previous
                or type(coefficient) is not float
                or not math.isfinite(coefficient)
            ):
                raise PhaseConditionedObjectiveHullRowMaterializationError(
                    "local_row_entry_noncanonical"
                )
            previous = column
            if coefficient != 0.0:
                indices.append(column)
                data.append(coefficient)
                if len(indices) % 4096 == 0:
                    _check_deadline(deadline, "csr_entry_formation")
        indptr.append(len(indices))
    return _freeze_csr(
        sp.csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(len(rows), columns),
            dtype=np.float64,
        )
    )


def _csr_sha256(name: str, matrix: sp.csr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(f"{name}:canonical_csr_binary64_v1:".encode("ascii"))
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(
        matrix.indptr.astype("<i8", copy=False).tobytes(order="C")
    )
    digest.update(
        matrix.indices.astype("<i8", copy=False).tobytes(order="C")
    )
    digest.update(matrix.data.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _rhs_sha256(name: str, value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(f"{name}:binary64_vector_v1:".encode("ascii"))
    digest.update(np.asarray(value.size, dtype="<i8").tobytes())
    digest.update(value.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _coefficient_error_payload(
    item: CoefficientFormationError,
) -> Dict[str, Any]:
    return {
        "group": item.group,
        "identifier": item.identifier,
        "exact": item.exact,
        "stored": item.stored,
        "absolute_error": item.absolute_error,
    }


def _guard_payload(item: UpperRowFormationGuard) -> Dict[str, Any]:
    return {
        "row_index": item.row_index,
        "row_name": item.row_name,
        "coefficient_errors": tuple(
            _coefficient_error_payload(error)
            for error in item.coefficient_errors
        ),
        "total_coefficient_guard": item.total_coefficient_guard,
        "raw_rhs_exact": item.raw_rhs_exact,
        "guarded_rhs_exact": item.guarded_rhs_exact,
        "stored_rhs": item.stored_rhs,
        "rhs_outward_error": item.rhs_outward_error,
    }


def _convert_row_terms(
    row: ExactHZLinearRow,
    *,
    row_index: int,
    equality: bool,
    continuous_positions: _StableIdLookup,
    binary_positions: _StableIdLookup,
    parent_continuous_columns: int,
    eta_columns: int,
    caps: PCOHRowMaterializationCaps,
    deadline: float,
    visited_nonzeros: int,
) -> Tuple[
    Tuple[Tuple[int, float], ...],
    Tuple[Tuple[int, float], ...],
    Tuple[CoefficientFormationError, ...],
    int,
]:
    continuous_eta = []
    binary = []
    errors = []
    groups = (
        (
            "parent_continuous",
            row.parent_continuous_terms,
            continuous_positions,
            0,
            continuous_eta,
        ),
        (
            "parent_binary",
            row.parent_binary_terms,
            binary_positions,
            0,
            binary,
        ),
        (
            "eta",
            row.eta_terms,
            None,
            parent_continuous_columns,
            continuous_eta,
        ),
    )
    for group, terms, position_map, offset, target in groups:
        for identifier, exact in terms:
            visited_nonzeros += 1
            if visited_nonzeros > caps.max_total_exact_nonzeros:
                raise PhaseConditionedObjectiveHullRowMaterializationError(
                    "total_exact_nonzero_cap_exceeded"
                )
            if visited_nonzeros % 4096 == 0:
                _check_deadline(deadline, "coefficient_conversion")
            _check_fraction_bits(
                exact,
                name=f"{group}_coefficient_{row_index}_{identifier}",
                caps=caps,
            )
            if group == "eta":
                if (
                    type(identifier) is not int
                    or identifier < 0
                    or identifier >= eta_columns
                ):
                    raise PhaseConditionedObjectiveHullRowMaterializationError(
                        "eta_identifier_out_of_range"
                    )
                position = identifier
            else:
                position = position_map.position(
                    identifier, deadline=deadline
                )
                if position is None:
                    raise PhaseConditionedObjectiveHullRowMaterializationError(
                        f"{group}_stable_id_missing:{identifier}"
                    )
            if equality:
                stored = _fraction_to_exact_binary64(
                    exact,
                    name=f"equality_{group}_{row_index}_{identifier}",
                )
            else:
                stored = _fraction_to_finite_binary64(
                    exact,
                    name=f"upper_{group}_{row_index}_{identifier}",
                )
                stored_exact = Fraction.from_float(stored)
                errors.append(
                    CoefficientFormationError(
                        group=group,
                        identifier=identifier,
                        exact=exact,
                        stored=stored,
                        absolute_error=abs(stored_exact - exact),
                    )
                )
            target.append((offset + position, stored))
    return (
        tuple(continuous_eta),
        tuple(binary),
        tuple(errors),
        visited_nonzeros,
    )


def _accumulate_coefficient_guard(
    errors: Sequence[CoefficientFormationError],
    *,
    row_index: int,
    caps: PCOHRowMaterializationCaps,
    deadline: float,
) -> Fraction:
    """Accumulate exactly without allowing an unchecked denominator blow-up."""

    total = Fraction(0)
    for error_index, item in enumerate(errors):
        total += item.absolute_error
        _check_fraction_bits(
            total,
            name=f"upper_total_guard_{row_index}",
            caps=caps,
        )
        if error_index % 4096 == 0:
            _check_deadline(deadline, "coefficient_guard_accumulation")
    return total


def _frame_payload(
    *,
    descriptor: PhaseConditionedObjectiveHullDescriptor,
    parent_digest: str,
    col_ids_sha: str,
    bcol_ids_sha: str,
    parent_continuous_columns: int,
    parent_binary_columns: int,
    eta_columns: int,
    block_sha256: Mapping[str, str],
    equality_rhs_sha256: str,
    upper_rhs_sha256: str,
    guards: Tuple[UpperRowFormationGuard, ...],
) -> Dict[str, Any]:
    return {
        "schema": _FRAME_SCHEMA,
        "descriptor_representation_sha256": (
            descriptor.representation_sha256
        ),
        "parent_semantic_digest": parent_digest,
        "objective_binding_sha256": (
            descriptor.objective_binding.objective_binding_sha256
        ),
        "parent_col_ids_sha256": col_ids_sha,
        "parent_bcol_ids_sha256": bcol_ids_sha,
        "parent_continuous_columns": parent_continuous_columns,
        "parent_binary_columns": parent_binary_columns,
        "eta_columns": eta_columns,
        "block_sha256": dict(block_sha256),
        "equality_rhs_sha256": equality_rhs_sha256,
        "upper_rhs_sha256": upper_rhs_sha256,
        "upper_row_guard_sha256": tuple(
            _canonical_sha256(_guard_payload(item)) for item in guards
        ),
        "proof_authority": False,
        "verdict_authority": False,
    }


def materialize_phase_conditioned_objective_hull_row_frame(
    descriptor: PhaseConditionedObjectiveHullDescriptor,
    *,
    live_parent_semantic_digest: str,
    parent_col_ids: np.ndarray,
    parent_bcol_ids: np.ndarray,
    deadline: float,
    caps: PCOHRowMaterializationCaps = _DEFAULT_CAPS,
) -> PCOHBinary64LocalRowFrame:
    """Form a sound local binary64 row frame without touching a live HZ."""

    normalized_caps = _normalize_caps(caps)
    deadline_value = _strict_deadline(deadline)
    _check_deadline(deadline_value, "entry")
    if type(descriptor) is not PhaseConditionedObjectiveHullDescriptor:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "descriptor_wrong_type"
        )
    if not _valid_sha256(live_parent_semantic_digest):
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "live_parent_semantic_digest_invalid"
        )
    try:
        descriptor_ok = verify_phase_conditioned_objective_hull(
            descriptor,
            live_parent_semantic_digest=live_parent_semantic_digest,
            live_objective_binding=descriptor.objective_binding,
        )
    except Exception as exc:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "descriptor_structural_verification_failed"
        ) from exc
    if descriptor_ok is not True:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "descriptor_structural_verification_failed"
        )
    _check_deadline(deadline_value, "descriptor_verification")

    col_ids, continuous_positions = _strict_id_array(
        parent_col_ids,
        name="parent_col_ids",
        maximum_columns=normalized_caps.max_parent_continuous_columns,
        deadline=deadline_value,
    )
    bcol_ids, binary_positions = _strict_id_array(
        parent_bcol_ids,
        name="parent_bcol_ids",
        maximum_columns=normalized_caps.max_parent_binary_columns,
        deadline=deadline_value,
    )
    eta_columns = len(descriptor.eta_columns)
    if (
        eta_columns < 1
        or eta_columns > normalized_caps.max_eta_columns
        or tuple(column.local_index for column in descriptor.eta_columns)
        != tuple(range(eta_columns))
    ):
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "eta_column_frame_noncanonical_or_over_cap"
        )
    total_rows = len(descriptor.equality_rows) + len(descriptor.upper_rows)
    if total_rows > normalized_caps.max_rows:
        raise PhaseConditionedObjectiveHullRowMaterializationError(
            "row_cap_exceeded"
        )

    equality_continuous_eta_rows = []
    equality_binary_rows = []
    equality_rhs = []
    upper_continuous_eta_rows = []
    upper_binary_rows = []
    upper_rhs = []
    guards = []
    visited_nonzeros = 0

    for row_index, row in enumerate(descriptor.equality_rows):
        if type(row) is not ExactHZLinearRow or row.sense != "eq":
            raise PhaseConditionedObjectiveHullRowMaterializationError(
                "equality_row_type_or_sense_invalid"
            )
        continuous_eta, binary, _, visited_nonzeros = _convert_row_terms(
            row,
            row_index=row_index,
            equality=True,
            continuous_positions=continuous_positions,
            binary_positions=binary_positions,
            parent_continuous_columns=col_ids.size,
            eta_columns=eta_columns,
            caps=normalized_caps,
            deadline=deadline_value,
            visited_nonzeros=visited_nonzeros,
        )
        _check_fraction_bits(
            row.rhs,
            name=f"equality_rhs_{row_index}",
            caps=normalized_caps,
        )
        stored_rhs = _fraction_to_exact_binary64(
            row.rhs, name=f"equality_rhs_{row_index}"
        )
        equality_continuous_eta_rows.append(continuous_eta)
        equality_binary_rows.append(binary)
        equality_rhs.append(stored_rhs)

    for row_index, row in enumerate(descriptor.upper_rows):
        if type(row) is not ExactHZLinearRow or row.sense != "le":
            raise PhaseConditionedObjectiveHullRowMaterializationError(
                "upper_row_type_or_sense_invalid"
            )
        continuous_eta, binary, errors, visited_nonzeros = _convert_row_terms(
            row,
            row_index=row_index,
            equality=False,
            continuous_positions=continuous_positions,
            binary_positions=binary_positions,
            parent_continuous_columns=col_ids.size,
            eta_columns=eta_columns,
            caps=normalized_caps,
            deadline=deadline_value,
            visited_nonzeros=visited_nonzeros,
        )
        _check_fraction_bits(
            row.rhs,
            name=f"upper_rhs_{row_index}",
            caps=normalized_caps,
        )
        total_guard = _accumulate_coefficient_guard(
            errors,
            row_index=row_index,
            caps=normalized_caps,
            deadline=deadline_value,
        )
        guarded_rhs = row.rhs + total_guard
        _check_fraction_bits(
            guarded_rhs,
            name=f"upper_guarded_rhs_{row_index}",
            caps=normalized_caps,
        )
        try:
            stored_rhs = outward_float64(guarded_rhs)
        except Exception as exc:
            raise PhaseConditionedObjectiveHullRowMaterializationError(
                f"upper_guarded_rhs_{row_index}_has_no_finite_outward_binary64"
            ) from exc
        stored_rhs_exact = Fraction.from_float(stored_rhs)
        if stored_rhs_exact < guarded_rhs:
            raise PhaseConditionedObjectiveHullRowMaterializationError(
                "upper_rhs_outward_conversion_failed"
            )
        guard = UpperRowFormationGuard(
            row_index=row_index,
            row_name=row.name,
            coefficient_errors=errors,
            total_coefficient_guard=total_guard,
            raw_rhs_exact=row.rhs,
            guarded_rhs_exact=guarded_rhs,
            stored_rhs=stored_rhs,
            rhs_outward_error=stored_rhs_exact - guarded_rhs,
        )
        upper_continuous_eta_rows.append(continuous_eta)
        upper_binary_rows.append(binary)
        upper_rhs.append(stored_rhs)
        guards.append(guard)

    _check_deadline(deadline_value, "before_csr_formation")
    continuous_eta_columns = int(col_ids.size) + eta_columns
    matrices = {
        "equality_continuous_eta": _rows_to_csr(
            equality_continuous_eta_rows,
            columns=continuous_eta_columns,
            deadline=deadline_value,
        ),
        "equality_binary": _rows_to_csr(
            equality_binary_rows,
            columns=int(bcol_ids.size),
            deadline=deadline_value,
        ),
        "upper_continuous_eta": _rows_to_csr(
            upper_continuous_eta_rows,
            columns=continuous_eta_columns,
            deadline=deadline_value,
        ),
        "upper_binary": _rows_to_csr(
            upper_binary_rows,
            columns=int(bcol_ids.size),
            deadline=deadline_value,
        ),
    }
    _check_deadline(deadline_value, "after_csr_formation")
    equality_rhs_array = np.asarray(equality_rhs, dtype=np.float64)
    upper_rhs_array = np.asarray(upper_rhs, dtype=np.float64)
    equality_rhs_array.setflags(write=False)
    upper_rhs_array.setflags(write=False)
    block_sha = {
        name: _csr_sha256(name, matrix)
        for name, matrix in matrices.items()
    }
    equality_rhs_sha = _rhs_sha256("equality_rhs", equality_rhs_array)
    upper_rhs_sha = _rhs_sha256("upper_rhs", upper_rhs_array)
    col_ids_sha = _id_vector_sha256(
        "parent_col_ids", col_ids, deadline=deadline_value
    )
    bcol_ids_sha = _id_vector_sha256(
        "parent_bcol_ids", bcol_ids, deadline=deadline_value
    )
    guards_tuple = tuple(guards)
    frame_payload = _frame_payload(
        descriptor=descriptor,
        parent_digest=live_parent_semantic_digest,
        col_ids_sha=col_ids_sha,
        bcol_ids_sha=bcol_ids_sha,
        parent_continuous_columns=int(col_ids.size),
        parent_binary_columns=int(bcol_ids.size),
        eta_columns=eta_columns,
        block_sha256=block_sha,
        equality_rhs_sha256=equality_rhs_sha,
        upper_rhs_sha256=upper_rhs_sha,
        guards=guards_tuple,
    )
    frame_sha = _canonical_sha256(frame_payload)
    receipt: Dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "status": "sound_binary64_local_row_frame",
        "proof_authority": False,
        "verdict_authority": False,
        "candidate_only": True,
        "descriptor_structurally_reverified": True,
        "upstream_certificates_replayed": False,
        "algorithm": "exact_coefficient_l1_box_guard_then_outward_rhs_v1",
        "variable_domain_used_for_guard": "all_parent_and_eta_in_closed_minus_one_plus_one",
        "equality_binary64_policy": "finite_bit_exact_or_fail_closed",
        "upper_coefficient_policy": "nearest_finite_binary64_with_exact_fraction_error",
        "upper_rhs_policy": "outward_float64_of_exact_rhs_plus_exact_l1_coefficient_error",
        "column_layout": "continuous_eta_parent_prefix_then_eta_local_tail",
        "uses_sparse_hstack": False,
        "uses_sparse_vstack": False,
        "full_parent_constraint_frame_loaded": False,
        "absolute_deadline": True,
        "caps": _caps_payload(normalized_caps),
        "descriptor_representation_sha256": (
            descriptor.representation_sha256
        ),
        "parent_semantic_digest": live_parent_semantic_digest,
        "objective_binding_sha256": (
            descriptor.objective_binding.objective_binding_sha256
        ),
        "parent_col_ids_sha256": col_ids_sha,
        "parent_bcol_ids_sha256": bcol_ids_sha,
        "parent_continuous_columns": int(col_ids.size),
        "parent_binary_columns": int(bcol_ids.size),
        "eta_columns": eta_columns,
        "equality_rows": len(descriptor.equality_rows),
        "upper_rows": len(descriptor.upper_rows),
        "total_exact_nonzeros": visited_nonzeros,
        "block_shapes": {
            name: matrix.shape for name, matrix in matrices.items()
        },
        "block_nonzeros": {
            name: int(matrix.nnz) for name, matrix in matrices.items()
        },
        "block_sha256": block_sha,
        "equality_rhs_sha256": equality_rhs_sha,
        "upper_rhs_sha256": upper_rhs_sha,
        "upper_row_formation": tuple(
            _guard_payload(item) for item in guards_tuple
        ),
        "frame_sha256": frame_sha,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _check_deadline(deadline_value, "authorization")
    return PCOHBinary64LocalRowFrame(
        schema=_FRAME_SCHEMA,
        descriptor_representation_sha256=descriptor.representation_sha256,
        parent_semantic_digest=live_parent_semantic_digest,
        objective_binding_sha256=(
            descriptor.objective_binding.objective_binding_sha256
        ),
        parent_col_ids_sha256=col_ids_sha,
        parent_bcol_ids_sha256=bcol_ids_sha,
        parent_continuous_columns=int(col_ids.size),
        parent_binary_columns=int(bcol_ids.size),
        eta_columns=eta_columns,
        equality_continuous_eta=matrices["equality_continuous_eta"],
        equality_binary=matrices["equality_binary"],
        upper_continuous_eta=matrices["upper_continuous_eta"],
        upper_binary=matrices["upper_binary"],
        equality_rhs=equality_rhs_array,
        upper_rhs=upper_rhs_array,
        upper_row_guards=guards_tuple,
        block_sha256=_deep_freeze(block_sha),
        frame_sha256=frame_sha,
        receipt=_deep_freeze(receipt),
        proof_authority=False,
        verdict_authority=False,
    )


def _csr_exactly_equal(left: Any, right: Any) -> bool:
    return (
        type(left) is sp.csr_matrix
        and type(right) is sp.csr_matrix
        and left.shape == right.shape
        and left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and left.has_canonical_format
        and right.has_canonical_format
        and not left.data.flags.writeable
        and not left.indices.flags.writeable
        and not left.indptr.flags.writeable
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _binary64_vector_exactly_equal(left: Any, right: Any) -> bool:
    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and left.ndim == 1
        and right.ndim == 1
        and not left.flags.writeable
        and np.array_equal(left, right)
    )


def verify_phase_conditioned_objective_hull_row_frame(
    frame: Any,
    descriptor: PhaseConditionedObjectiveHullDescriptor,
    *,
    live_parent_semantic_digest: str,
    parent_col_ids: np.ndarray,
    parent_bcol_ids: np.ndarray,
    deadline: float,
    caps: PCOHRowMaterializationCaps = _DEFAULT_CAPS,
) -> bool:
    """Strictly reconstruct and compare a non-authoritative local row frame."""

    try:
        if (
            type(frame) is not PCOHBinary64LocalRowFrame
            or frame.schema != _FRAME_SCHEMA
            or frame.proof_authority is not False
            or frame.verdict_authority is not False
            or not _valid_sha256(frame.frame_sha256)
            or not isinstance(frame.receipt, Mapping)
            or not isinstance(frame.block_sha256, Mapping)
        ):
            return False
        expected = materialize_phase_conditioned_objective_hull_row_frame(
            descriptor,
            live_parent_semantic_digest=live_parent_semantic_digest,
            parent_col_ids=parent_col_ids,
            parent_bcol_ids=parent_bcol_ids,
            deadline=deadline,
            caps=caps,
        )
        scalar_fields = (
            "schema",
            "descriptor_representation_sha256",
            "parent_semantic_digest",
            "objective_binding_sha256",
            "parent_col_ids_sha256",
            "parent_bcol_ids_sha256",
            "parent_continuous_columns",
            "parent_binary_columns",
            "eta_columns",
            "upper_row_guards",
            "frame_sha256",
            "proof_authority",
            "verdict_authority",
        )
        if any(
            getattr(frame, name) != getattr(expected, name)
            for name in scalar_fields
        ):
            return False
        for name in (
            "equality_continuous_eta",
            "equality_binary",
            "upper_continuous_eta",
            "upper_binary",
        ):
            if not _csr_exactly_equal(
                getattr(frame, name), getattr(expected, name)
            ):
                return False
        return (
            _binary64_vector_exactly_equal(
                frame.equality_rhs, expected.equality_rhs
            )
            and _binary64_vector_exactly_equal(
                frame.upper_rhs, expected.upper_rhs
            )
            and _canonical_form(frame.block_sha256)
            == _canonical_form(expected.block_sha256)
            and _canonical_form(frame.receipt)
            == _canonical_form(expected.receipt)
        )
    except Exception:
        return False


__all__ = [
    "CoefficientFormationError",
    "PCOHBinary64LocalRowFrame",
    "PCOHRowMaterializationCaps",
    "PhaseConditionedObjectiveHullRowMaterializationError",
    "UpperRowFormationGuard",
    "materialize_phase_conditioned_objective_hull_row_frame",
    "verify_phase_conditioned_objective_hull_row_frame",
]
