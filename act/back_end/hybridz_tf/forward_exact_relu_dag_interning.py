#!/usr/bin/env python3
# ===- forward_exact_relu_dag_interning.py - exact DAG compaction --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Toy-only forward exact-ReLU predicate and constraint interning.

This module is intentionally disconnected from ``tf_mlp``, the solver, the
verifier, and configuration.  It demonstrates two exact set-preserving
operations:

* immutable stable-ID rows are stored once across residual DAG views; and
* byte-for-byte/Fraction-identical unstable ReLU predicates reuse one exact
  compressed graph handle.

The represented network graph is unchanged.  Only redundant encoder-private
phase labels, lift factors, and rows are quotiented.  No triangle relaxation,
backward pass, dual construction, or branch-and-bound operation appears here.
All exported receipts remain permanently non-authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import time
from typing import Any, Dict, Sequence, Tuple


Q = Fraction
ZERO = Q(0)
ONE = Q(1)
HALF = Q(1, 2)

_AFFINE_SCHEMA = "act.forward_exact_relu.affine.v1"
_ROW_SCHEMA = "act.forward_exact_relu.row.v1"
_ROW_VIEW_SCHEMA = "act.forward_exact_relu.row_view.v1"
_HANDLE_SCHEMA = "act.forward_exact_relu.handle.v1"
_BUILD_SCHEMA = "act.forward_exact_relu.fixed_toy_build.v1"
_PHASE_SCHEMA = "act.forward_exact_relu.phase_projection.v1"
_SYNTHETIC_SCHEMA = "act.forward_exact_relu.synthetic.v1"


def _q(value: Any) -> Fraction:
    if type(value) is Fraction:
        return value
    if type(value) is int:
        return Fraction(value)
    raise ValueError("exact values must be builtin int or Fraction, never bool")


def _qtext(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


ExactTerm = Tuple[int, Fraction]


def _normalize_terms(values: Any, *, name: str) -> Tuple[ExactTerm, ...]:
    if type(values) is not tuple:
        raise ValueError(f"{name} must be an exact tuple")
    combined: Dict[int, Fraction] = {}
    for item in values:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError(f"{name} contains a malformed term")
        stable_id, raw_coefficient = item
        if type(stable_id) is not int or stable_id < 0:
            raise ValueError(f"{name} stable ids must be nonnegative builtin ints")
        coefficient = _q(raw_coefficient)
        if coefficient == 0:
            raise ValueError(f"{name} must not store explicit zero coefficients")
        combined[stable_id] = combined.get(stable_id, ZERO) + coefficient
    normalized = tuple(
        (stable_id, coefficient)
        for stable_id, coefficient in sorted(combined.items())
        if coefficient != 0
    )
    if len(normalized) != len(values):
        raise ValueError(f"{name} must be sorted, unique, and zero-free")
    return normalized


def _terms_payload(values: Sequence[ExactTerm]) -> list[list[Any]]:
    return [[stable_id, _qtext(coefficient)] for stable_id, coefficient in values]


def _affine_payload(
    constant: Fraction,
    continuous_terms: Tuple[ExactTerm, ...],
    binary_terms: Tuple[ExactTerm, ...],
) -> dict[str, Any]:
    return {
        "schema": _AFFINE_SCHEMA,
        "constant": _qtext(constant),
        "continuous_terms": _terms_payload(continuous_terms),
        "binary_terms": _terms_payload(binary_terms),
    }


@dataclass(frozen=True)
class ExactAffineForm:
    """One exact affine expression over stable continuous and binary IDs."""

    constant: Fraction
    continuous_terms: Tuple[ExactTerm, ...] = ()
    binary_terms: Tuple[ExactTerm, ...] = ()
    semantic_digest: str = ""
    schema: str = _AFFINE_SCHEMA

    def __post_init__(self) -> None:
        constant = _q(self.constant)
        continuous = _normalize_terms(
            self.continuous_terms, name="continuous_terms"
        )
        binary = _normalize_terms(self.binary_terms, name="binary_terms")
        if (
            type(self.schema) is not str
            or self.schema != _AFFINE_SCHEMA
            or type(self.semantic_digest) is not str
            or (self.semantic_digest != "" and not _is_sha256(self.semantic_digest))
            or not set(stable_id for stable_id, _ in continuous).isdisjoint(
                stable_id for stable_id, _ in binary
            )
        ):
            raise ValueError("affine schema or digest is malformed")
        payload = _affine_payload(constant, continuous, binary)
        observed = _digest(payload)
        if self.semantic_digest != "" and self.semantic_digest != observed:
            raise ValueError("affine digest is stale")
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "continuous_terms", continuous)
        object.__setattr__(self, "binary_terms", binary)
        object.__setattr__(self, "semantic_digest", observed)

    def scale(self, scalar: int | Fraction) -> "ExactAffineForm":
        value = _q(scalar)
        if value == 0:
            return ExactAffineForm(ZERO)
        return ExactAffineForm(
            self.constant * value,
            tuple((stable_id, coefficient * value) for stable_id, coefficient in self.continuous_terms),
            tuple((stable_id, coefficient * value) for stable_id, coefficient in self.binary_terms),
        )

    def add(self, other: "ExactAffineForm") -> "ExactAffineForm":
        left = _snapshot_affine(self)
        right = _snapshot_affine(other)
        continuous: Dict[int, Fraction] = dict(left.continuous_terms)
        binary: Dict[int, Fraction] = dict(left.binary_terms)
        for stable_id, coefficient in right.continuous_terms:
            continuous[stable_id] = continuous.get(stable_id, ZERO) + coefficient
        for stable_id, coefficient in right.binary_terms:
            binary[stable_id] = binary.get(stable_id, ZERO) + coefficient
        return ExactAffineForm(
            left.constant + right.constant,
            tuple((key, value) for key, value in sorted(continuous.items()) if value),
            tuple((key, value) for key, value in sorted(binary.items()) if value),
        )

    def subtract(self, other: "ExactAffineForm") -> "ExactAffineForm":
        return self.add(other.scale(-1))

    def shift(self, value: int | Fraction) -> "ExactAffineForm":
        return self.add(ExactAffineForm(_q(value)))

    @property
    def support_nnz(self) -> int:
        return len(self.continuous_terms) + len(self.binary_terms)


def _affine_state_is_exact(value: Any) -> bool:
    return bool(
        type(value) is ExactAffineForm
        and type(value.constant) is Fraction
        and type(value.continuous_terms) is tuple
        and type(value.binary_terms) is tuple
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is int
            and item[0] >= 0
            and type(item[1]) is Fraction
            and item[1] != 0
            for item in value.continuous_terms + value.binary_terms
        )
        and tuple(item[0] for item in value.continuous_terms)
        == tuple(sorted(item[0] for item in value.continuous_terms))
        and tuple(item[0] for item in value.binary_terms)
        == tuple(sorted(item[0] for item in value.binary_terms))
        and _is_sha256(value.semantic_digest)
        and type(value.schema) is str
        and value.schema == _AFFINE_SCHEMA
    )


def _snapshot_affine(value: Any) -> ExactAffineForm:
    if not _affine_state_is_exact(value):
        raise ValueError("live affine state is malformed")
    constant = value.constant
    continuous = tuple((item[0], item[1]) for item in value.continuous_terms)
    binary = tuple((item[0], item[1]) for item in value.binary_terms)
    digest = value.semantic_digest
    schema = value.schema
    return ExactAffineForm(
        constant,
        continuous,
        binary,
        semantic_digest=digest,
        schema=schema,
    )


def _affine_truth_key(
    value: ExactAffineForm,
) -> Tuple[Fraction, Tuple[ExactTerm, ...], Tuple[ExactTerm, ...]]:
    snapshot = _snapshot_affine(value)
    return (
        snapshot.constant,
        snapshot.continuous_terms,
        snapshot.binary_terms,
    )


def _canonical_affine_payload(value: ExactAffineForm) -> dict[str, Any]:
    snapshot = _snapshot_affine(value)
    return _affine_payload(
        snapshot.constant,
        snapshot.continuous_terms,
        snapshot.binary_terms,
    )


def _validate_predicate_snapshot(value: ExactAffineForm) -> None:
    if not _affine_state_is_exact(value):
        raise ValueError("predicate snapshot is malformed")


def _row_payload(
    *,
    stable_row_id: int,
    row_tag: str,
    kind: str,
    continuous_terms: Tuple[ExactTerm, ...],
    binary_terms: Tuple[ExactTerm, ...],
    rhs: Fraction,
) -> dict[str, Any]:
    return {
        "schema": _ROW_SCHEMA,
        "stable_row_id": stable_row_id,
        "row_tag": row_tag,
        "kind": kind,
        "continuous_terms": _terms_payload(continuous_terms),
        "binary_terms": _terms_payload(binary_terms),
        "rhs": _qtext(rhs),
    }


@dataclass(frozen=True)
class ExactConstraintRow:
    """One immutable exact equality or upper row bound to a stable row ID."""

    stable_row_id: int
    row_tag: str
    kind: str
    continuous_terms: Tuple[ExactTerm, ...]
    binary_terms: Tuple[ExactTerm, ...]
    rhs: Fraction
    row_digest: str = ""
    schema: str = _ROW_SCHEMA

    def __post_init__(self) -> None:
        continuous = _normalize_terms(
            self.continuous_terms, name="row continuous_terms"
        )
        binary = _normalize_terms(
            self.binary_terms, name="row binary_terms"
        )
        rhs = _q(self.rhs)
        if (
            type(self.stable_row_id) is not int
            or self.stable_row_id < 0
            or type(self.row_tag) is not str
            or not self.row_tag
            or type(self.kind) is not str
            or self.kind not in {"eq", "le"}
            or not continuous + binary
            or not set(stable_id for stable_id, _ in continuous).isdisjoint(
                stable_id for stable_id, _ in binary
            )
            or type(self.row_digest) is not str
            or (self.row_digest != "" and not _is_sha256(self.row_digest))
            or type(self.schema) is not str
            or self.schema != _ROW_SCHEMA
        ):
            raise ValueError("constraint row identity, kind, or schema is malformed")
        payload = _row_payload(
            stable_row_id=self.stable_row_id,
            row_tag=self.row_tag,
            kind=self.kind,
            continuous_terms=continuous,
            binary_terms=binary,
            rhs=rhs,
        )
        observed = _digest(payload)
        if self.row_digest != "" and self.row_digest != observed:
            raise ValueError("constraint row digest is stale")
        object.__setattr__(self, "continuous_terms", continuous)
        object.__setattr__(self, "binary_terms", binary)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "row_digest", observed)

    @property
    def nnz(self) -> int:
        return len(self.continuous_terms) + len(self.binary_terms)


def _row_state_is_exact(value: Any) -> bool:
    return bool(
        type(value) is ExactConstraintRow
        and type(value.stable_row_id) is int
        and value.stable_row_id >= 0
        and type(value.row_tag) is str
        and bool(value.row_tag)
        and type(value.kind) is str
        and value.kind in {"eq", "le"}
        and type(value.rhs) is Fraction
        and type(value.continuous_terms) is tuple
        and type(value.binary_terms) is tuple
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is int
            and item[0] >= 0
            and type(item[1]) is Fraction
            and item[1] != 0
            for item in value.continuous_terms + value.binary_terms
        )
        and _is_sha256(value.row_digest)
        and type(value.schema) is str
        and value.schema == _ROW_SCHEMA
    )


def _snapshot_row(value: Any) -> ExactConstraintRow:
    if not _row_state_is_exact(value):
        raise ValueError("live constraint row state is malformed")
    return ExactConstraintRow(
        stable_row_id=value.stable_row_id,
        row_tag=value.row_tag,
        kind=value.kind,
        continuous_terms=tuple((item[0], item[1]) for item in value.continuous_terms),
        binary_terms=tuple((item[0], item[1]) for item in value.binary_terms),
        rhs=value.rhs,
        row_digest=value.row_digest,
        schema=value.schema,
    )


def _row_truth_key(
    value: ExactConstraintRow,
) -> Tuple[
    int,
    str,
    str,
    Tuple[ExactTerm, ...],
    Tuple[ExactTerm, ...],
    Fraction,
]:
    snapshot = _snapshot_row(value)
    return (
        snapshot.stable_row_id,
        snapshot.row_tag,
        snapshot.kind,
        snapshot.continuous_terms,
        snapshot.binary_terms,
        snapshot.rhs,
    )


def _canonical_row_payload(value: ExactConstraintRow) -> dict[str, Any]:
    snapshot = _snapshot_row(value)
    return _row_payload(
        stable_row_id=snapshot.stable_row_id,
        row_tag=snapshot.row_tag,
        kind=snapshot.kind,
        continuous_terms=snapshot.continuous_terms,
        binary_terms=snapshot.binary_terms,
        rhs=snapshot.rhs,
    )


def _validate_row_snapshot(value: ExactConstraintRow) -> None:
    if not _row_state_is_exact(value):
        raise ValueError("constraint row snapshot is malformed")


class ExactConstraintArena:
    """Private immutable-row arena; repeated stable rows are stored once."""

    def __init__(self) -> None:
        self._rows_by_id: Dict[int, ExactConstraintRow] = {}

    def intern(self, row: ExactConstraintRow) -> ExactConstraintRow:
        return self.intern_many((row,))[0]

    def intern_many(
        self, rows: Tuple[ExactConstraintRow, ...]
    ) -> Tuple[ExactConstraintRow, ...]:
        """Atomically validate and intern an exact row batch."""

        if type(rows) is not tuple or not rows:
            raise ValueError("row batch must be a nonempty exact tuple")
        snapshots = tuple(_snapshot_row(row) for row in rows)
        for snapshot in snapshots:
            _validate_row_snapshot(snapshot)
        # Re-snapshot after callbacks, then reject duplicate request IDs rather
        # than giving input order any semantic meaning.
        snapshots = tuple(_snapshot_row(row) for row in snapshots)
        row_ids = tuple(row.stable_row_id for row in snapshots)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("row batch contains duplicate stable row ids")

        staged = dict(self._rows_by_id)
        for snapshot in snapshots:
            existing = staged.get(snapshot.stable_row_id)
            if existing is not None:
                # Digests select a cheap bucket only.  Full canonical Fraction
                # state is the truth even under a forced SHA collision.
                if _row_truth_key(existing) != _row_truth_key(snapshot):
                    raise ValueError(
                        "stable row id collides with different exact content"
                    )
                continue
            staged[snapshot.stable_row_id] = _snapshot_row(snapshot)
        self._rows_by_id = staged
        return tuple(
            _snapshot_row(self._rows_by_id[row_id]) for row_id in row_ids
        )

    def rows_for_ids(self, row_ids: Sequence[int]) -> Tuple[ExactConstraintRow, ...]:
        if (
            type(row_ids) is not tuple
            or any(type(value) is not int or value < 0 for value in row_ids)
            or len(set(row_ids)) != len(row_ids)
        ):
            raise ValueError("row id request must be an exact builtin-int tuple")
        try:
            return tuple(_snapshot_row(self._rows_by_id[value]) for value in row_ids)
        except KeyError as error:
            raise ValueError("row id is not live in this arena") from error

    @property
    def rows(self) -> Tuple[ExactConstraintRow, ...]:
        return tuple(
            _snapshot_row(self._rows_by_id[key])
            for key in sorted(self._rows_by_id)
        )

    @property
    def row_count(self) -> int:
        return len(self._rows_by_id)

    @property
    def nnz(self) -> int:
        return sum(_snapshot_row(row).nnz for row in self._rows_by_id.values())

    @property
    def stable_variable_ids(self) -> frozenset[int]:
        return frozenset(
            stable_id
            for row in self.rows
            for stable_id, _ in row.continuous_terms + row.binary_terms
        )

    @property
    def stable_row_ids(self) -> frozenset[int]:
        return frozenset(self._rows_by_id)


@dataclass(frozen=True)
class ExactDAGRowView:
    """Immutable row-ID set with associative/idempotent DAG union."""

    row_ids: frozenset[int]
    schema: str = _ROW_VIEW_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.row_ids) is not frozenset
            or any(type(value) is not int or value < 0 for value in self.row_ids)
            or type(self.schema) is not str
            or self.schema != _ROW_VIEW_SCHEMA
        ):
            raise ValueError("DAG row view is malformed")

    @classmethod
    def from_handle(cls, handle: "ExactReLUHandle") -> "ExactDAGRowView":
        snapshot = _snapshot_handle(handle)
        return cls(frozenset(snapshot.row_ids))

    def union(self, *others: "ExactDAGRowView") -> "ExactDAGRowView":
        left = _snapshot_row_view(self)
        accumulated = set(left.row_ids)
        for other in others:
            accumulated.update(_snapshot_row_view(other).row_ids)
        return ExactDAGRowView(frozenset(accumulated))

    def materialize(
        self, arena: ExactConstraintArena
    ) -> Tuple[ExactConstraintRow, ...]:
        if type(arena) is not ExactConstraintArena:
            raise TypeError("row view requires the exact arena type")
        snapshot = _snapshot_row_view(self)
        return arena.rows_for_ids(tuple(sorted(snapshot.row_ids)))


def _snapshot_row_view(value: Any) -> ExactDAGRowView:
    if type(value) is not ExactDAGRowView:
        raise ValueError("DAG row view has the wrong exact type")
    row_ids = value.row_ids
    schema = value.schema
    if type(row_ids) is not frozenset:
        raise ValueError("DAG row view IDs must be a builtin frozenset")
    return ExactDAGRowView(frozenset(row_ids), schema=schema)


def _handle_payload(
    *,
    predicate: ExactAffineForm,
    lower: Fraction,
    upper: Fraction,
    representative_node_id: int,
    xi1_stable_id: int,
    xi2_stable_id: int,
    phase_stable_id: int,
    row_ids: Tuple[int, int, int],
    output: ExactAffineForm,
) -> dict[str, Any]:
    return {
        "schema": _HANDLE_SCHEMA,
        "predicate_digest": predicate.semantic_digest,
        "predicate": _canonical_affine_payload(predicate),
        "bounds": [_qtext(lower), _qtext(upper)],
        "representative_node_id": representative_node_id,
        "xi_ids": [xi1_stable_id, xi2_stable_id],
        "phase_id": phase_stable_id,
        "row_ids": list(row_ids),
        "output_digest": output.semantic_digest,
        "output": _canonical_affine_payload(output),
        "proof_authority": False,
        "verdict_authority": False,
    }


@dataclass(frozen=True)
class ExactReLUHandle:
    predicate: ExactAffineForm
    predicate_digest: str
    lower: Fraction
    upper: Fraction
    representative_node_id: int
    xi1_stable_id: int
    xi2_stable_id: int
    phase_stable_id: int
    row_ids: Tuple[int, int, int]
    output: ExactAffineForm
    handle_digest: str
    schema: str = _HANDLE_SCHEMA

    def __post_init__(self) -> None:
        predicate = _snapshot_affine(self.predicate)
        output = _snapshot_affine(self.output)
        if (
            not _is_sha256(self.predicate_digest)
            or self.predicate_digest != predicate.semantic_digest
            or type(self.lower) is not Fraction
            or type(self.upper) is not Fraction
            or not self.lower < 0 < self.upper
            or type(self.representative_node_id) is not int
            or self.representative_node_id < 0
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.xi1_stable_id,
                    self.xi2_stable_id,
                    self.phase_stable_id,
                )
            )
            or len({self.xi1_stable_id, self.xi2_stable_id, self.phase_stable_id}) != 3
            or type(self.row_ids) is not tuple
            or len(self.row_ids) != 3
            or any(type(value) is not int or value < 0 for value in self.row_ids)
            or len(set(self.row_ids)) != 3
            or not _is_sha256(self.handle_digest)
            or type(self.schema) is not str
            or self.schema != _HANDLE_SCHEMA
        ):
            raise ValueError("exact ReLU handle is malformed")
        expected = _digest(
            _handle_payload(
                predicate=predicate,
                lower=self.lower,
                upper=self.upper,
                representative_node_id=self.representative_node_id,
                xi1_stable_id=self.xi1_stable_id,
                xi2_stable_id=self.xi2_stable_id,
                phase_stable_id=self.phase_stable_id,
                row_ids=self.row_ids,
                output=output,
            )
        )
        if self.handle_digest != expected:
            raise ValueError("exact ReLU handle digest is stale")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "output", output)

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False


def _snapshot_handle(value: Any) -> ExactReLUHandle:
    if type(value) is not ExactReLUHandle:
        raise ValueError("live exact ReLU handle has the wrong type")
    fields = (
        value.predicate,
        value.predicate_digest,
        value.lower,
        value.upper,
        value.representative_node_id,
        value.xi1_stable_id,
        value.xi2_stable_id,
        value.phase_stable_id,
        value.row_ids,
        value.output,
        value.handle_digest,
        value.schema,
    )
    predicate = _snapshot_affine(fields[0])
    output = _snapshot_affine(fields[9])
    return ExactReLUHandle(
        predicate=predicate,
        predicate_digest=fields[1],
        lower=fields[2],
        upper=fields[3],
        representative_node_id=fields[4],
        xi1_stable_id=fields[5],
        xi2_stable_id=fields[6],
        phase_stable_id=fields[7],
        row_ids=fields[8],
        output=output,
        handle_digest=fields[10],
        schema=fields[11],
    )


@dataclass(frozen=True)
class ReLUInternResult:
    handle: ExactReLUHandle
    created: bool

    def __post_init__(self) -> None:
        if type(self.created) is not bool:
            raise ValueError("intern result creation flag must be a builtin bool")
        object.__setattr__(self, "handle", _snapshot_handle(self.handle))


class ExactReLUInterner:
    """Create or reuse exact compressed ReLU graph blocks at kappa=1."""

    def __init__(
        self,
        arena: ExactConstraintArena,
        *,
        reuse_predicates: bool,
        next_continuous_id: int,
        next_binary_id: int,
        next_row_id: int,
    ) -> None:
        if type(arena) is not ExactConstraintArena:
            raise TypeError("arena must use the exact arena type")
        if type(reuse_predicates) is not bool:
            raise ValueError("reuse_predicates must be a builtin bool")
        starts = (next_continuous_id, next_binary_id, next_row_id)
        if any(type(value) is not int or value < 0 for value in starts):
            raise ValueError("stable allocator starts must be nonnegative builtin ints")
        self._arena = arena
        self._reuse = reuse_predicates
        self._next_cont = next_continuous_id
        self._next_bin = next_binary_id
        self._next_row = next_row_id
        self._by_bucket: Dict[
            Tuple[str, Fraction, Fraction],
            Tuple[Tuple[ExactAffineForm, ExactReLUHandle], ...],
        ] = {}

    def _new_handle(
        self,
        predicate: ExactAffineForm,
        lower: Fraction,
        upper: Fraction,
        semantic_node_id: int,
    ) -> ExactReLUHandle:
        xi1 = self._next_cont
        xi2 = self._next_cont + 1
        phase = self._next_bin
        rows = (self._next_row, self._next_row + 1, self._next_row + 2)

        candidates = (xi1, xi2, phase)
        occupied = set(self._arena.stable_variable_ids)
        occupied.update(
            stable_id
            for stable_id, _ in (
                predicate.continuous_terms + predicate.binary_terms
            )
        )
        if len(set(candidates)) != 3 or set(candidates) & occupied:
            raise ValueError(
                "fresh ReLU continuous/binary stable IDs collide with live support"
            )
        if set(rows) & set(self._arena.stable_row_ids):
            raise ValueError("fresh ReLU row IDs collide with the live arena")

        continuous: Dict[int, Fraction] = {
            stable_id: -coefficient
            for stable_id, coefficient in predicate.continuous_terms
        }
        continuous[xi1] = lower / 2
        continuous[xi2] = -upper / 2
        binary: Dict[int, Fraction] = {
            stable_id: -coefficient
            for stable_id, coefficient in predicate.binary_terms
        }
        binary[phase] = lower / 2
        link = ExactConstraintRow(
            rows[0],
            f"relu:{semantic_node_id}:link",
            "eq",
            tuple((key, value) for key, value in sorted(continuous.items()) if value),
            tuple((key, value) for key, value in sorted(binary.items()) if value),
            predicate.constant - upper / 2,
        )
        inactive = ExactConstraintRow(
            rows[1],
            f"relu:{semantic_node_id}:inactive_projection",
            "le",
            ((xi1, Q(-1)),),
            ((phase, Q(-1)),),
            ZERO,
        )
        active = ExactConstraintRow(
            rows[2],
            f"relu:{semantic_node_id}:active_projection",
            "le",
            ((xi2, Q(-1)),),
            ((phase, ONE),),
            ZERO,
        )
        # Continuous factors xi1/xi2 lie in [-1,1] and the HZ phase factor z
        # lies in {-1,+1}.  z=+1 forces xi2=1, hence y=0 and the link spans
        # p in [lower,0].  z=-1 forces xi1=1, hence y=p and the link spans
        # p in [0,upper].  The union is exactly the ReLU graph.
        output = ExactAffineForm(
            upper / 2,
            ((xi2, -upper / 2),),
        )
        payload = _handle_payload(
            predicate=predicate,
            lower=lower,
            upper=upper,
            representative_node_id=semantic_node_id,
            xi1_stable_id=xi1,
            xi2_stable_id=xi2,
            phase_stable_id=phase,
            row_ids=rows,
            output=output,
        )
        handle = ExactReLUHandle(
            predicate=predicate,
            predicate_digest=predicate.semantic_digest,
            lower=lower,
            upper=upper,
            representative_node_id=semantic_node_id,
            xi1_stable_id=xi1,
            xi2_stable_id=xi2,
            phase_stable_id=phase,
            row_ids=rows,
            output=output,
            handle_digest=_digest(payload),
        )
        # The three graph rows appear atomically, and allocator cursors advance
        # only after every exact object and collision check has succeeded.
        self._arena.intern_many((link, inactive, active))
        self._next_cont += 2
        self._next_bin += 1
        self._next_row += 3
        return handle

    def intern(
        self,
        predicate: ExactAffineForm,
        lower: int | Fraction,
        upper: int | Fraction,
        *,
        semantic_node_id: int,
    ) -> ReLUInternResult:
        snapshot = _snapshot_affine(predicate)
        _validate_predicate_snapshot(snapshot)
        # Consume a new private copy after validation, closing callback ABA on
        # the first private object as well as on the caller-owned predicate.
        snapshot = _snapshot_affine(snapshot)
        lower_q = _q(lower)
        upper_q = _q(upper)
        if (
            not lower_q < 0 < upper_q
            or type(semantic_node_id) is not int
            or semantic_node_id < 0
        ):
            raise ValueError("only exact unstable ReLU predicates are interned")
        bucket_key = (snapshot.semantic_digest, lower_q, upper_q)
        if self._reuse:
            truth = _affine_truth_key(snapshot)
            for stored_predicate, stored_handle in self._by_bucket.get(
                bucket_key, ()
            ):
                if _affine_truth_key(stored_predicate) == truth:
                    return ReLUInternResult(stored_handle, False)
        handle = self._new_handle(snapshot, lower_q, upper_q, semantic_node_id)
        if self._reuse:
            entry = (_snapshot_affine(snapshot), _snapshot_handle(handle))
            self._by_bucket[bucket_key] = (
                *self._by_bucket.get(bucket_key, ()),
                entry,
            )
        return ReLUInternResult(handle, True)


def _fixed_build_payload(
    *,
    compact: bool,
    physical_relu_count: int,
    phase_handle_count: int,
    phase_assignment_count: int,
    alias_count: int,
    unique_rows: Tuple[ExactConstraintRow, ...],
    physical_rows: Tuple[ExactConstraintRow, ...],
    dag_row_view: ExactDAGRowView,
    node_handle_digests: Tuple[Tuple[str, str], ...],
    output: ExactAffineForm,
) -> dict[str, Any]:
    return {
        "schema": _BUILD_SCHEMA,
        "compact": compact,
        "physical_relu_count": physical_relu_count,
        "phase_handle_count": phase_handle_count,
        "phase_assignment_count": phase_assignment_count,
        "alias_count": alias_count,
        "unique_rows": [_canonical_row_payload(row) for row in unique_rows],
        "physical_rows": [_canonical_row_payload(row) for row in physical_rows],
        "dag_row_view": sorted(dag_row_view.row_ids),
        "node_handle_digests": [list(item) for item in node_handle_digests],
        "exact_output_form": _canonical_affine_payload(output),
        "proof_authority": False,
        "verdict_authority": False,
    }


@dataclass(frozen=True)
class FixedToyBuild:
    compact: bool
    physical_relu_count: int
    phase_handle_count: int
    phase_assignment_count: int
    alias_count: int
    unique_rows: Tuple[ExactConstraintRow, ...]
    physical_rows: Tuple[ExactConstraintRow, ...]
    dag_row_view: ExactDAGRowView
    node_handle_digests: Tuple[Tuple[str, str], ...]
    exact_output_form: ExactAffineForm
    receipt_sha256: str
    schema: str = _BUILD_SCHEMA

    def __post_init__(self) -> None:
        if type(self.unique_rows) is not tuple or type(self.physical_rows) is not tuple:
            raise ValueError("fixed toy row collections must be exact tuples")
        unique_rows = tuple(_snapshot_row(row) for row in self.unique_rows)
        physical_rows = tuple(_snapshot_row(row) for row in self.physical_rows)
        dag_row_view = _snapshot_row_view(self.dag_row_view)
        output = _snapshot_affine(self.exact_output_form)
        if (
            type(self.compact) is not bool
            or type(self.physical_relu_count) is not int
            or self.physical_relu_count != 4
            or type(self.phase_handle_count) is not int
            or self.phase_handle_count not in {2, 4}
            or type(self.phase_assignment_count) is not int
            or self.phase_assignment_count != 2 ** self.phase_handle_count
            or type(self.alias_count) is not int
            or self.alias_count != self.physical_relu_count - self.phase_handle_count
            or len({row.stable_row_id for row in unique_rows}) != len(unique_rows)
            or len({row.stable_row_id for row in physical_rows}) != len(physical_rows)
            or tuple(row.stable_row_id for row in physical_rows)
            != tuple(sorted(dag_row_view.row_ids))
            or frozenset(row.stable_row_id for row in unique_rows)
            != dag_row_view.row_ids
            or type(self.node_handle_digests) is not tuple
            or len(self.node_handle_digests) != 4
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not _is_sha256(item[1])
                for item in self.node_handle_digests
            )
            or tuple(item[0] for item in self.node_handle_digests)
            != ("a", "l", "r", "b")
            or not _is_sha256(self.receipt_sha256)
            or type(self.schema) is not str
            or self.schema != _BUILD_SCHEMA
        ):
            raise ValueError("fixed toy build receipt is malformed")
        expected = _digest(
            _fixed_build_payload(
                compact=self.compact,
                physical_relu_count=self.physical_relu_count,
                phase_handle_count=self.phase_handle_count,
                phase_assignment_count=self.phase_assignment_count,
                alias_count=self.alias_count,
                unique_rows=unique_rows,
                physical_rows=physical_rows,
                dag_row_view=dag_row_view,
                node_handle_digests=self.node_handle_digests,
                output=output,
            )
        )
        if self.receipt_sha256 != expected:
            raise ValueError("fixed toy build receipt is stale")
        object.__setattr__(self, "unique_rows", unique_rows)
        object.__setattr__(self, "physical_rows", physical_rows)
        object.__setattr__(self, "dag_row_view", dag_row_view)
        object.__setattr__(self, "exact_output_form", output)

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False

    @property
    def unique_row_count(self) -> int:
        return len(self.unique_rows)

    @property
    def unique_nnz(self) -> int:
        return sum(_snapshot_row(row).nnz for row in self.unique_rows)

    @property
    def physical_row_count(self) -> int:
        return len(self.physical_rows)

    @property
    def physical_nnz(self) -> int:
        return sum(_snapshot_row(row).nnz for row in self.physical_rows)


def build_fixed_residual_fanout_toy(
    *, compact: bool, stable_id_offset: int = 0
) -> FixedToyBuild:
    """Build the frozen four-ReLU residual/fanout exact-constraint toy."""

    if type(compact) is not bool:
        raise ValueError("compact must be a builtin bool")
    if type(stable_id_offset) is not int or stable_id_offset < 0:
        raise ValueError("stable_id_offset must be a nonnegative builtin int")
    base = stable_id_offset * 10_000
    root = ExactAffineForm(ZERO, ((base + 101, ONE),))
    arena = ExactConstraintArena()
    interner = ExactReLUInterner(
        arena,
        reuse_predicates=compact,
        next_continuous_id=base + 1_001,
        next_binary_id=base + 2_001,
        next_row_id=base + 3_001,
    )

    a_result = interner.intern(root, -1, 1, semantic_node_id=base + 11)
    a = a_result.handle.output
    shared_t = a.shift(-HALF)
    left_result = interner.intern(
        shared_t, -HALF, HALF, semantic_node_id=base + 12
    )
    right_result = interner.intern(
        shared_t, -HALF, HALF, semantic_node_id=base + 13
    )
    # Residual join: a + (x-a) is coefficient-wise exactly x.
    skip = root.subtract(a)
    joined = a.add(skip)
    if _affine_truth_key(joined) != _affine_truth_key(root):
        raise AssertionError("exact residual cancellation did not recover x")
    b_result = interner.intern(joined, -1, 1, semantic_node_id=base + 14)

    left = left_result.handle.output
    right = right_result.handle.output
    b = b_result.handle.output
    output = a.subtract(b).add(left).subtract(right)
    node_results = (
        ("a", a_result),
        ("l", left_result),
        ("r", right_result),
        ("b", b_result),
    )
    handles = {result.handle.row_ids for _, result in node_results}
    node_handle_digests = tuple(
        (name, result.handle.handle_digest) for name, result in node_results
    )

    # Generic immutable DAG views make prefix/fanout sharing structural.  The
    # two branch views are joined by associative/idempotent frozenset union;
    # no toy-specific row tuple is selected here.
    a_view = ExactDAGRowView.from_handle(a_result.handle)
    left_view = a_view.union(ExactDAGRowView.from_handle(left_result.handle))
    right_view = a_view.union(ExactDAGRowView.from_handle(right_result.handle))
    joined_view = left_view.union(right_view)
    final_view = joined_view.union(ExactDAGRowView.from_handle(b_result.handle))
    physical_rows = final_view.materialize(arena)
    unique_rows = arena.rows
    if frozenset(row.stable_row_id for row in unique_rows) != final_view.row_ids:
        raise AssertionError("reachable DAG row union does not cover the arena")
    payload = _fixed_build_payload(
        compact=compact,
        physical_relu_count=4,
        phase_handle_count=len(handles),
        phase_assignment_count=2 ** len(handles),
        alias_count=4 - len(handles),
        unique_rows=unique_rows,
        physical_rows=physical_rows,
        dag_row_view=final_view,
        node_handle_digests=node_handle_digests,
        output=output,
    )
    return FixedToyBuild(
        compact=compact,
        physical_relu_count=4,
        phase_handle_count=len(handles),
        phase_assignment_count=2 ** len(handles),
        alias_count=4 - len(handles),
        unique_rows=unique_rows,
        physical_rows=physical_rows,
        dag_row_view=final_view,
        node_handle_digests=node_handle_digests,
        exact_output_form=output,
        receipt_sha256=_digest(payload),
    )


@dataclass(frozen=True)
class FixedToyPoint:
    x: Fraction
    a: Fraction
    t_left: Fraction
    left: Fraction
    t_right: Fraction
    right: Fraction
    skip: Fraction
    joined: Fraction
    b: Fraction
    q: Fraction

    def __post_init__(self) -> None:
        values = (
            self.x,
            self.a,
            self.t_left,
            self.left,
            self.t_right,
            self.right,
            self.skip,
            self.joined,
            self.b,
            self.q,
        )
        if any(type(value) is not Fraction for value in values):
            raise ValueError("fixed toy point must contain only Fractions")
        expected_a = max(ZERO, self.x)
        expected_t = expected_a - HALF
        expected_left = max(ZERO, expected_t)
        expected_skip = self.x - expected_a
        expected_joined = expected_a + expected_skip
        expected_b = max(ZERO, expected_joined)
        expected_q = expected_a - expected_b
        if (
            not -1 <= self.x <= 1
            or self.a != expected_a
            or self.t_left != expected_t
            or self.t_right != expected_t
            or self.left != expected_left
            or self.right != expected_left
            or self.skip != expected_skip
            or self.joined != expected_joined
            or self.b != expected_b
            or self.q != expected_q
        ):
            raise ValueError("fixed toy point is not on the canonical graph")


def _snapshot_fixed_toy_point(value: Any) -> FixedToyPoint:
    if type(value) is not FixedToyPoint:
        raise TypeError("raw evaluator requires an exact fixed toy point")
    fields = (
        value.x,
        value.a,
        value.t_left,
        value.left,
        value.t_right,
        value.right,
        value.skip,
        value.joined,
        value.b,
        value.q,
    )
    return FixedToyPoint(*fields)


def evaluate_fixed_toy(x: int | Fraction) -> FixedToyPoint:
    value = _q(x)
    if not -1 <= value <= 1:
        raise ValueError("toy input lies outside [-1,1]")
    a = max(ZERO, value)
    t = a - HALF
    left = max(ZERO, t)
    right = max(ZERO, t)
    skip = value - a
    joined = a + skip
    b = max(ZERO, joined)
    return FixedToyPoint(
        value, a, t, left, t, right, skip, joined, b,
        a - b + left - right,
    )


def exact_fixed_toy_jacobian(
    x: int | Fraction,
) -> Tuple[Tuple[str, Fraction], ...]:
    value = _q(x)
    if value in (ZERO, HALF) or not -1 < value < 1:
        raise ValueError("Jacobian probe must be interior and away from kinks")
    da = ZERO if value < 0 else ONE
    dt = da
    dl = dt if value > HALF else ZERO
    dr = dl
    dskip = ONE - da
    djoined = da + dskip
    db = djoined if value > 0 else ZERO
    dq = da - db + dl - dr
    return (
        ("x", ONE),
        ("a", da),
        ("t_left", dt),
        ("left", dl),
        ("t_right", dt),
        ("right", dr),
        ("skip", dskip),
        ("joined", djoined),
        ("b", db),
        ("q", dq),
    )


@dataclass(frozen=True)
class _AffineScalar:
    slope: Fraction
    intercept: Fraction

    def value(self, x: Fraction) -> Fraction:
        return self.slope * x + self.intercept

    def add(self, other: "_AffineScalar") -> "_AffineScalar":
        return _AffineScalar(self.slope + other.slope, self.intercept + other.intercept)

    def subtract(self, other: "_AffineScalar") -> "_AffineScalar":
        return _AffineScalar(self.slope - other.slope, self.intercept - other.intercept)


@dataclass(frozen=True)
class PhaseProjection:
    assignment: Tuple[int, ...]
    feasible: bool
    lower: Fraction | None
    upper: Fraction | None
    lower_values: Tuple[Fraction, ...]
    upper_values: Tuple[Fraction, ...]
    compact: bool
    schema: str = _PHASE_SCHEMA

    def __post_init__(self) -> None:
        expected_width = 2 if self.compact is True else 4
        if (
            type(self.assignment) is not tuple
            or len(self.assignment) != expected_width
            or any(type(value) is not int or value not in {0, 1} for value in self.assignment)
            or type(self.feasible) is not bool
            or type(self.compact) is not bool
            or type(self.lower_values) is not tuple
            or type(self.upper_values) is not tuple
            or type(self.schema) is not str
            or self.schema != _PHASE_SCHEMA
        ):
            raise ValueError("phase projection identity is malformed")
        if self.feasible:
            if (
                type(self.lower) is not Fraction
                or type(self.upper) is not Fraction
                or self.lower > self.upper
                or len(self.lower_values) != 10
                or len(self.upper_values) != 10
                or any(type(value) is not Fraction for value in self.lower_values + self.upper_values)
            ):
                raise ValueError("feasible phase projection is malformed")
        elif not (
            self.lower is None
            and self.upper is None
            and type(self.lower_values) is tuple
            and len(self.lower_values) == 0
            and type(self.upper_values) is tuple
            and len(self.upper_values) == 0
        ):
            raise ValueError("infeasible phase projection carries values")


def _snapshot_phase_projection(value: Any) -> PhaseProjection:
    if type(value) is not PhaseProjection:
        raise ValueError("phase projection has the wrong exact type")
    fields = (
        value.assignment,
        value.feasible,
        value.lower,
        value.upper,
        value.lower_values,
        value.upper_values,
        value.compact,
        value.schema,
    )
    return PhaseProjection(*fields)


def _snapshot_projection_tuple(values: Any) -> Tuple[PhaseProjection, ...]:
    if type(values) is not tuple:
        raise ValueError("phase projection collection must be an exact tuple")
    return tuple(_snapshot_phase_projection(value) for value in values)


def _restrict_sign(
    expression: _AffineScalar,
    lower: Fraction,
    upper: Fraction,
    *,
    active: bool,
) -> Tuple[Fraction, Fraction] | None:
    slope, intercept = expression.slope, expression.intercept
    if slope == 0:
        valid = intercept >= 0 if active else intercept <= 0
        return (lower, upper) if valid else None
    root = -intercept / slope
    if active:
        if slope > 0:
            lower = max(lower, root)
        else:
            upper = min(upper, root)
    else:
        if slope > 0:
            upper = min(upper, root)
        else:
            lower = max(lower, root)
    return None if lower > upper else (lower, upper)


def _phase_relu(
    preactivation: _AffineScalar,
    phase: int,
    interval: Tuple[Fraction, Fraction],
) -> Tuple[_AffineScalar, Tuple[Fraction, Fraction]] | None:
    restricted = _restrict_sign(
        preactivation, interval[0], interval[1], active=bool(phase)
    )
    if restricted is None:
        return None
    return (
        preactivation if phase else _AffineScalar(ZERO, ZERO),
        restricted,
    )


def _projection_for_assignment(
    assignment: Tuple[int, ...], *, compact: bool
) -> PhaseProjection:
    expected = 2 if compact else 4
    if type(assignment) is not tuple or len(assignment) != expected:
        raise ValueError("phase assignment has the wrong width")
    x = _AffineScalar(ONE, ZERO)
    interval = (Q(-1), Q(1))
    if compact:
        phase_a, phase_t = assignment
        first = _phase_relu(x, phase_a, interval)
        if first is None:
            return PhaseProjection(assignment, False, None, None, (), (), compact)
        a, interval = first
        t = a.add(_AffineScalar(ZERO, -HALF))
        second = _phase_relu(t, phase_t, interval)
        if second is None:
            return PhaseProjection(assignment, False, None, None, (), (), compact)
        left, interval = second
        right = left
        skip = x.subtract(a)
        joined = a.add(skip)
        b = a
    else:
        phase_a, phase_left, phase_right, phase_b = assignment
        first = _phase_relu(x, phase_a, interval)
        if first is None:
            return PhaseProjection(assignment, False, None, None, (), (), compact)
        a, interval = first
        t = a.add(_AffineScalar(ZERO, -HALF))
        second = _phase_relu(t, phase_left, interval)
        if second is None:
            return PhaseProjection(assignment, False, None, None, (), (), compact)
        left, interval = second
        third = _phase_relu(t, phase_right, interval)
        if third is None:
            return PhaseProjection(assignment, False, None, None, (), (), compact)
        right, interval = third
        skip = x.subtract(a)
        joined = a.add(skip)
        fourth = _phase_relu(joined, phase_b, interval)
        if fourth is None:
            return PhaseProjection(assignment, False, None, None, (), (), compact)
        b, interval = fourth
    q = a.subtract(b).add(left).subtract(right)
    expressions = (x, a, t, left, t, right, skip, joined, b, q)
    lower_values = tuple(expression.value(interval[0]) for expression in expressions)
    upper_values = tuple(expression.value(interval[1]) for expression in expressions)
    return PhaseProjection(
        assignment,
        True,
        interval[0],
        interval[1],
        lower_values,
        upper_values,
        compact,
    )


def enumerate_phase_projections(*, compact: bool) -> Tuple[PhaseProjection, ...]:
    if type(compact) is not bool:
        raise ValueError("compact must be a builtin bool")
    width = 2 if compact else 4
    return tuple(
        _projection_for_assignment(tuple(bits), compact=compact)
        for bits in itertools.product((0, 1), repeat=width)
    )


def projection_probe_points(
    baseline: Sequence[PhaseProjection], compact: Sequence[PhaseProjection]
) -> Tuple[Fraction, ...]:
    baseline_snapshot = _snapshot_projection_tuple(baseline)
    compact_snapshot = _snapshot_projection_tuple(compact)
    endpoints = {Q(-1), Q(1)}
    for projection in baseline_snapshot + compact_snapshot:
        if projection.feasible:
            endpoints.add(projection.lower)  # type: ignore[arg-type]
            endpoints.add(projection.upper)  # type: ignore[arg-type]
    ordered = sorted(endpoints)
    probes = set(ordered)
    probes.update((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    return tuple(sorted(probes))


def projection_values_at(
    projections: Sequence[PhaseProjection], x: int | Fraction
) -> Tuple[Tuple[Fraction, ...], ...]:
    value = _q(x)
    observed = set()
    for projection in _snapshot_projection_tuple(projections):
        if not projection.feasible:
            continue
        lower = projection.lower
        upper = projection.upper
        assert lower is not None and upper is not None
        if not lower <= value <= upper:
            continue
        if lower == upper:
            observed.add(projection.lower_values)
        else:
            weight = (value - lower) / (upper - lower)
            observed.add(
                tuple(
                    first + weight * (second - first)
                    for first, second in zip(
                        projection.lower_values, projection.upper_values
                    )
                )
            )
    return tuple(sorted(observed))


RAW_MARGIN = Q(1, 16)


def raw_unsafe_margins(point: FixedToyPoint) -> Tuple[Fraction, Fraction]:
    snapshot = _snapshot_fixed_toy_point(point)
    return snapshot.q - RAW_MARGIN, -snapshot.q - RAW_MARGIN


def raw_vnnlib_assert() -> str:
    return "(assert (or (>= Y_0 0.0625) (<= Y_0 -0.0625)))"


@dataclass(frozen=True)
class SyntheticCompactionStats:
    blocks: int
    compact: bool
    unique_rows: int
    unique_nnz: int
    materialized_rows: int
    materialized_nnz: int
    content_sha256: str
    elapsed_seconds: float
    schema: str = _SYNTHETIC_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.blocks) is not int
            or self.blocks <= 0
            or type(self.compact) is not bool
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.unique_rows,
                    self.unique_nnz,
                    self.materialized_rows,
                    self.materialized_nnz,
                )
            )
            or not _is_sha256(self.content_sha256)
            or type(self.elapsed_seconds) is not float
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
            or type(self.schema) is not str
            or self.schema != _SYNTHETIC_SCHEMA
        ):
            raise ValueError("synthetic compaction receipt is malformed")

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False


def build_synthetic_residual_family(
    *, blocks: int = 64, compact: bool
) -> SyntheticCompactionStats:
    """Build and stream-hash a fixed family; used only by the toy wall gate."""

    if type(blocks) is not int or blocks <= 0 or blocks > 256:
        raise ValueError("synthetic block count must lie in [1,256]")
    if type(compact) is not bool:
        raise ValueError("compact must be a builtin bool")
    unique_rows = unique_nnz = materialized_rows = materialized_nnz = 0
    builds: list[FixedToyBuild] = []
    for block in range(blocks):
        build = build_fixed_residual_fanout_toy(
            compact=compact, stable_id_offset=block + 1
        )
        builds.append(build)
        unique_rows += build.unique_row_count
        unique_nnz += build.unique_nnz
        materialized_rows += build.physical_row_count
        materialized_nnz += build.physical_nnz

    # Measure the row load/replay phase itself.  Graph construction is outside
    # this timer because it is common setup rather than the row-view cost whose
    # >=1.5x stop gate this synthetic benchmark is intended to measure.
    started = time.perf_counter()
    hasher = hashlib.sha256()
    for build in builds:
        # These are repeated row-stream passes used by a load/replay path.
        # Interning eliminates the same physical duplication from every pass.
        for stage in range(32):
            hasher.update(f"row-pass:{stage}".encode("ascii"))
            for row in build.physical_rows:
                hasher.update(row.row_digest.encode("ascii"))
                for stable_id, coefficient in row.continuous_terms + row.binary_terms:
                    hasher.update(str(stable_id).encode("ascii"))
                    hasher.update(_qtext(coefficient).encode("ascii"))
                hasher.update(_qtext(row.rhs).encode("ascii"))
    elapsed = float(time.perf_counter() - started)
    return SyntheticCompactionStats(
        blocks=blocks,
        compact=compact,
        unique_rows=unique_rows,
        unique_nnz=unique_nnz,
        materialized_rows=materialized_rows,
        materialized_nnz=materialized_nnz,
        content_sha256=hasher.hexdigest(),
        elapsed_seconds=elapsed,
    )


__all__ = [
    "ExactAffineForm",
    "ExactConstraintArena",
    "ExactConstraintRow",
    "ExactDAGRowView",
    "ExactReLUHandle",
    "ExactReLUInterner",
    "FixedToyBuild",
    "FixedToyPoint",
    "PhaseProjection",
    "RAW_MARGIN",
    "ReLUInternResult",
    "SyntheticCompactionStats",
    "build_fixed_residual_fanout_toy",
    "build_synthetic_residual_family",
    "enumerate_phase_projections",
    "evaluate_fixed_toy",
    "exact_fixed_toy_jacobian",
    "projection_probe_points",
    "projection_values_at",
    "raw_unsafe_margins",
    "raw_vnnlib_assert",
]
