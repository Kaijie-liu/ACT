#!/usr/bin/env python3
# ===- constraint_block_dag_candidate.py - exact source DAG candidate --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Disconnected exact constraint-source RANGE/DAG candidate.

This module is deliberately not imported by Operator-HZ, ``SparseHZono``, a
solver, the verifier, or configuration.  It tests a source representation in
which a guarded equality band

``A*x <= u_forward`` and ``-A*x <= u_reverse``

may be stored as one native ranged row

``-u_reverse <= A*x <= u_forward``.

Compaction is permitted independently for every row and only when the two
canonical stored coefficient rows are bit-for-bit sign negations.  Otherwise
the original two upper rows are retained.  The compact source keeps stable
IDs and tags for both *virtual facets*, so an independent replay reconstructs
the original two-row frame exactly without retaining its second CSR payload.

The arena also supplies a small immutable block-set DAG.  Views are bound to
one opaque arena token, and union is associative, commutative, and idempotent.
Cross-arena union fails closed.  Digests select interning buckets only; full
canonical bytes decide equality.

Mutable owners and arenas are explicitly thread-confined: every mutating or
sealing entry point must run on the creating thread.  Sealed programs contain
only immutable source authority and remain safe to replay from another thread.

All numeric authority is held in builtin immutable ``bytes`` objects.  Array
and CSR accessors create detached/read-only views of those bytes.  Every
receipt and result remains non-authoritative and disconnected from production.
There is no triangle relaxation, branch-and-bound implementation, backward
pass, dual tightening, verifier call, or real/large-model entry point here.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import math
import statistics
import threading
import time
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp


_SCHEMA = "act.exact_constraint_block_dag_candidate.v1"
_PROGRAM_SCHEMA = "act.exact_constraint_program_candidate.v1"
_PERF_SCHEMA = "act.exact_constraint_program_c89_ratio_perf.v1"
_FACTOR_KINDS = frozenset({"continuous", "binary"})
_OBJECT_KINDS = frozenset(
    {"block", "facet", "le_row", "range_row", "payload", "view"}
)
_FAMILY_ADD_MATERIALIZE = "add_materialize"

_GLOBAL_ID_LOCK = threading.Lock()
_GLOBAL_NEXT_ID: Dict[str, int] = {
    "factor": 0,
    "block": 0,
    "facet": 0,
    "le_row": 0,
    "range_row": 0,
    "payload": 0,
    "view": 0,
}
# Factor identities have process-wide provenance, including identities imported
# from a future Operator owner.  This table is append-only: sealing, failure,
# and owner collection never make an identity reusable.
_GLOBAL_CLAIMED_FACTOR_KINDS: Dict[int, str] = {}
_INT64_MAX = int(np.iinfo(np.int64).max)

_PROGRAM_REGISTRY_LOCK = threading.Lock()
_PROGRAM_REGISTRY: Dict[int, Tuple[Any, ...]] = {}


class ConstraintBlockDAGCandidateError(ValueError):
    """Fail-closed error raised only by this disconnected candidate."""


class ConstraintArenaMismatch(ConstraintBlockDAGCandidateError):
    """A row/block/view from a different arena was presented to this arena."""


def _exact_nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ConstraintBlockDAGCandidateError(
            f"{name} must be a nonnegative builtin int"
        )
    return value


def _stable_object_key(
    value: Any,
    *,
    allowed_kinds: frozenset[str],
) -> Tuple[str, int]:
    if (
        type(value) is not StableObjectID
        or type(value.kind) is not str
        or value.kind not in allowed_kinds
        or type(value.value) is not int
        or value.value < 0
        or value.value > _INT64_MAX
    ):
        raise ConstraintBlockDAGCandidateError(
            "stable object identity was substituted or mutated"
        )
    return value.kind, value.value


def _reserve_ids(
    namespace: str,
    count: int,
    *,
    lower_bound_exclusive: Optional[int] = None,
) -> Tuple[int, int]:
    if type(namespace) is not str or namespace not in _GLOBAL_NEXT_ID:
        raise ConstraintBlockDAGCandidateError("unknown typed ID namespace")
    count = _exact_nonnegative_int(count, name="ID count")
    if lower_bound_exclusive is None:
        floor = -1
    else:
        floor = _exact_nonnegative_int(
            lower_bound_exclusive, name="ID lower bound"
        )
    with _GLOBAL_ID_LOCK:
        start = max(int(_GLOBAL_NEXT_ID[namespace]), floor + 1)
        if count and (
            start > _INT64_MAX or count - 1 > _INT64_MAX - start
        ):
            raise ConstraintBlockDAGCandidateError(
                f"{namespace} ID reservation exceeds signed int64"
            )
        stop = start + count
        # Reservations are intentionally never rolled back.  A failed build
        # burns its interval so stale handles cannot collide with later ones.
        _GLOBAL_NEXT_ID[namespace] = stop
    return start, stop


def _claim_imported_factor_ids(
    continuous_ids: Tuple[int, ...],
    binary_ids: Tuple[int, ...],
) -> None:
    """Atomically claim imported raw IDs in the global typed registry."""

    claims = tuple((value, "continuous") for value in continuous_ids) + tuple(
        (value, "binary") for value in binary_ids
    )
    if not claims:
        return
    with _GLOBAL_ID_LOCK:
        collisions = tuple(
            (value, _GLOBAL_CLAIMED_FACTOR_KINDS[value], kind)
            for value, kind in claims
            if value in _GLOBAL_CLAIMED_FACTOR_KINDS
        )
        if collisions:
            raise ConstraintBlockDAGCandidateError(
                "imported factor ID was already allocated or imported"
            )
        # Claim and floor advancement are one transaction.  If subsequent
        # owner construction fails, these identities stay burned.
        for value, kind in claims:
            _GLOBAL_CLAIMED_FACTOR_KINDS[value] = kind
        _GLOBAL_NEXT_ID["factor"] = max(
            int(_GLOBAL_NEXT_ID["factor"]),
            max(value for value, _kind in claims) + 1,
        )


def _reserve_factor_ids(kind: str, count: int) -> Tuple[int, int]:
    """Atomically allocate and type-claim fresh raw factor identities."""

    if type(kind) is not str or kind not in _FACTOR_KINDS:
        raise ConstraintBlockDAGCandidateError("unknown factor kind")
    count = _exact_nonnegative_int(count, name="factor ID count")
    with _GLOBAL_ID_LOCK:
        start = int(_GLOBAL_NEXT_ID["factor"])
        if count and (
            start > _INT64_MAX or count - 1 > _INT64_MAX - start
        ):
            raise ConstraintBlockDAGCandidateError(
                "factor ID reservation exceeds signed int64"
            )
        stop = start + count
        if any(
            value in _GLOBAL_CLAIMED_FACTOR_KINDS
            for value in range(start, stop)
        ):
            # The floor always advances beyond imports, so this indicates
            # internal registry corruption rather than a recoverable retry.
            raise ConstraintBlockDAGCandidateError(
                "global factor allocator reached an already claimed ID"
            )
        for value in range(start, stop):
            _GLOBAL_CLAIMED_FACTOR_KINDS[value] = kind
        _GLOBAL_NEXT_ID["factor"] = stop
    return start, stop


@dataclass(frozen=True, order=True)
class StableFactorID:
    """One owner-validated globally non-reused factor identity."""

    kind: str
    value: int

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not str
            or self.kind not in _FACTOR_KINDS
            or type(self.value) is not int
            or self.value < 0
            or self.value > _INT64_MAX
        ):
            raise ConstraintBlockDAGCandidateError(
                "malformed stable factor ID"
            )


@dataclass(frozen=True, order=True)
class StableObjectID:
    """Typed globally non-reused block, row, facet, payload, or view ID."""

    kind: str
    value: int

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not str
            or self.kind not in _OBJECT_KINDS
            or type(self.value) is not int
            or self.value < 0
            or self.value > _INT64_MAX
        ):
            raise ConstraintBlockDAGCandidateError(
                "malformed stable object ID"
            )


def _fresh_object_id(kind: str) -> StableObjectID:
    start, _stop = _reserve_ids(kind, 1)
    return StableObjectID(kind, start)


class _OwnerToken:
    """Opaque process-local identity; equality is intentionally object identity."""


class _ArenaToken:
    """Opaque arena identity; equal-looking tokens are never interchangeable."""


@dataclass(frozen=True)
class ExactFactorFrame:
    """Immutable append-only factor-frame snapshot owned by one owner."""

    continuous_ids: Tuple[StableFactorID, ...]
    binary_ids: Tuple[StableFactorID, ...]
    _owner_token: Any
    _version: int

    def __post_init__(self) -> None:
        if (
            type(self.continuous_ids) is not tuple
            or type(self.binary_ids) is not tuple
            or type(self._owner_token) is not _OwnerToken
            or type(self._version) is not int
            or self._version < 0
            or any(
                type(value) is not StableFactorID
                or value.kind != expected
                for expected, values in (
                    ("continuous", self.continuous_ids),
                    ("binary", self.binary_ids),
                )
                for value in values
            )
        ):
            raise ConstraintBlockDAGCandidateError(
                "malformed factor-frame snapshot"
            )
        raw = tuple(
            value.value for value in self.continuous_ids + self.binary_ids
        )
        if len(set(raw)) != len(raw):
            raise ConstraintBlockDAGCandidateError(
                "factor-frame IDs are not globally disjoint"
            )


class ExactConstraintOwner:
    """Thread-confined allocator and live registry for one candidate arena."""

    def __init__(
        self,
        *,
        imported_continuous_ids: Tuple[int, ...] = (),
        imported_binary_ids: Tuple[int, ...] = (),
    ) -> None:
        if type(imported_continuous_ids) is not tuple or type(
            imported_binary_ids
        ) is not tuple:
            raise ConstraintBlockDAGCandidateError(
                "imported factor IDs must be exact tuples"
            )
        raw_cont = tuple(
            _exact_nonnegative_int(value, name="continuous factor ID")
            for value in imported_continuous_ids
        )
        raw_bin = tuple(
            _exact_nonnegative_int(value, name="binary factor ID")
            for value in imported_binary_ids
        )
        raw = raw_cont + raw_bin
        if any(value > _INT64_MAX for value in raw) or len(set(raw)) != len(raw):
            raise ConstraintBlockDAGCandidateError(
                "imported factor IDs collide or exceed signed int64"
            )
        _claim_imported_factor_ids(raw_cont, raw_bin)
        # Keep the Thread object alive and compare identity.  Integer thread
        # identifiers may be reused after a thread exits.
        self._thread = threading.current_thread()
        self._token = _OwnerToken()
        self._continuous = [
            StableFactorID("continuous", value) for value in raw_cont
        ]
        self._binary = [StableFactorID("binary", value) for value in raw_bin]
        self._registry: Dict[int, str] = {
            item.value: item.kind for item in self._continuous + self._binary
        }
        self._frames: Dict[int, Tuple[ExactFactorFrame, Tuple[Any, ...]]] = {}
        self._version = 0
        self._arena_created = False
        self._sealed = False

    def _require_thread(self) -> None:
        if threading.current_thread() is not self._thread:
            raise ConstraintBlockDAGCandidateError(
                "mutable constraint owner/arena is thread-confined"
            )

    def _require_open(self) -> None:
        self._require_thread()
        if self._sealed:
            raise ConstraintBlockDAGCandidateError(
                "constraint owner is already sealed"
            )

    def _allocate(self, kind: str, count: int) -> Tuple[StableFactorID, ...]:
        self._require_open()
        if kind not in _FACTOR_KINDS:
            raise ConstraintBlockDAGCandidateError("unknown factor kind")
        count = _exact_nonnegative_int(count, name=f"{kind} factor count")
        start, stop = _reserve_factor_ids(kind, count)
        values = tuple(StableFactorID(kind, raw) for raw in range(start, stop))
        target = self._continuous if kind == "continuous" else self._binary
        for value in values:
            if value.value in self._registry:
                raise ConstraintBlockDAGCandidateError(
                    "global factor allocator reused a live ID"
                )
            self._registry[value.value] = kind
            target.append(value)
        if values:
            self._version += 1
        return values

    def allocate_continuous(self, count: int) -> Tuple[StableFactorID, ...]:
        return self._allocate("continuous", count)

    def allocate_binary(self, count: int) -> Tuple[StableFactorID, ...]:
        return self._allocate("binary", count)

    def frame(self) -> ExactFactorFrame:
        self._require_open()
        frame = ExactFactorFrame(
            tuple(self._continuous),
            tuple(self._binary),
            self._token,
            int(self._version),
        )
        truth = _factor_frame_key(frame)
        self._frames[id(frame)] = (frame, truth)
        return frame

    def _validate_frame(self, frame: Any) -> ExactFactorFrame:
        self._require_thread()
        if type(frame) is not ExactFactorFrame:
            raise ConstraintBlockDAGCandidateError(
                "factor frame has the wrong exact type"
            )
        record = self._frames.get(id(frame))
        if (
            record is None
            or record[0] is not frame
            or frame._owner_token is not self._token
            or _factor_frame_key(frame) != record[1]
        ):
            raise ConstraintBlockDAGCandidateError(
                "factor frame is forged, stale, or belongs to another owner"
            )
        for expected, values in (
            ("continuous", frame.continuous_ids),
            ("binary", frame.binary_ids),
        ):
            for value in values:
                if self._registry.get(value.value) != expected:
                    raise ConstraintBlockDAGCandidateError(
                        "factor frame references a non-live typed factor"
                    )
        return frame

    def new_arena(self) -> "ExactConstraintArena":
        self._require_open()
        if self._arena_created:
            raise ConstraintBlockDAGCandidateError(
                "the disconnected owner permits exactly one arena"
            )
        self._arena_created = True
        return ExactConstraintArena(self, _factory_token=self._token)

    def _seal(self) -> None:
        self._require_open()
        self._sealed = True

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def production_integration(self) -> bool:
        return False


def _factor_frame_key(frame: ExactFactorFrame) -> Tuple[Any, ...]:
    if (
        type(frame) is not ExactFactorFrame
        or type(frame.continuous_ids) is not tuple
        or type(frame.binary_ids) is not tuple
        or type(frame._owner_token) is not _OwnerToken
        or type(frame._version) is not int
        or frame._version < 0
    ):
        raise ConstraintBlockDAGCandidateError(
            "factor frame was mutated after construction"
        )
    for expected, values in (
        ("continuous", frame.continuous_ids),
        ("binary", frame.binary_ids),
    ):
        if any(
            type(value) is not StableFactorID
            or type(value.kind) is not str
            or value.kind != expected
            or type(value.value) is not int
            or value.value < 0
            or value.value > _INT64_MAX
            for value in values
        ):
            raise ConstraintBlockDAGCandidateError(
                "factor frame contains a mutated factor identity"
            )
    return (
        tuple((value.kind, value.value) for value in frame.continuous_ids),
        tuple((value.kind, value.value) for value in frame.binary_ids),
        int(frame._version),
    )


@dataclass(frozen=True)
class _FrozenCSRBytes:
    """Canonical CSR held only as immutable builtin bytes."""

    rows: int
    columns: int
    data_bytes: bytes
    indices_bytes: bytes
    indptr_bytes: bytes

    @classmethod
    def from_csr(cls, value: Any, *, name: str) -> "_FrozenCSRBytes":
        if type(value) is not sp.csr_matrix or value.dtype != np.dtype(
            np.float64
        ):
            raise ConstraintBlockDAGCandidateError(
                f"{name} must be an exact float64 csr_matrix"
            )
        rows, columns = (int(value.shape[0]), int(value.shape[1]))
        # Copy first, then validate only the detached arrays.  In particular,
        # ``ascontiguousarray`` is not sufficient here: for normal CSR input
        # it aliases caller-owned contiguous storage and leaves a validation /
        # serialization TOCTOU window.  A writer racing the copy may influence
        # which snapshot is selected, but cannot mutate it after the copy; any
        # malformed values captured by the copy are rejected below.
        data = np.array(value.data, dtype=np.float64, order="C", copy=True)
        indices = np.array(value.indices, dtype=np.int64, order="C", copy=True)
        indptr = np.array(value.indptr, dtype=np.int64, order="C", copy=True)
        if (
            data.ndim != 1
            or indices.ndim != 1
            or indptr.ndim != 1
            or data.size != indices.size
            or indptr.size != rows + 1
            or int(indptr[0]) != 0
            or int(indptr[-1]) != int(data.size)
            or np.any(indptr[1:] < indptr[:-1])
            or (
                indices.size
                and (
                    np.any(indices < 0)
                    or np.any(indices >= columns)
                )
            )
            or (
                data.size
                and (
                    not np.all(np.isfinite(data))
                    or np.any(data == 0.0)
                )
            )
        ):
            raise ConstraintBlockDAGCandidateError(
                f"{name} is malformed, non-finite, or contains explicit zero"
            )
        # Do not trust scipy's cached canonical flags.  Validate every row.
        for row in range(rows):
            start, stop = int(indptr[row]), int(indptr[row + 1])
            if stop - start > 1 and np.any(
                indices[start + 1 : stop]
                <= indices[start : stop - 1]
            ):
                raise ConstraintBlockDAGCandidateError(
                    f"{name} row indices are not strictly increasing"
                )
        return cls(
            rows,
            columns,
            bytes(data.tobytes(order="C")),
            bytes(indices.tobytes(order="C")),
            bytes(indptr.tobytes(order="C")),
        )

    @property
    def nnz(self) -> int:
        return len(self.data_bytes) // np.dtype(np.float64).itemsize

    @property
    def payload_bytes(self) -> int:
        return (
            len(self.data_bytes)
            + len(self.indices_bytes)
            + len(self.indptr_bytes)
        )

    def csr(self, *, columns: Optional[int] = None) -> sp.csr_matrix:
        target_columns = self.columns if columns is None else int(columns)
        if target_columns < self.columns:
            raise ConstraintBlockDAGCandidateError(
                "cannot shrink a sealed CSR factor frame"
            )
        data = np.frombuffer(self.data_bytes, dtype=np.float64)
        indices = np.frombuffer(self.indices_bytes, dtype=np.int64)
        indptr = np.frombuffer(self.indptr_bytes, dtype=np.int64)
        result = sp.csr_matrix(
            (data, indices, indptr),
            shape=(self.rows, target_columns),
            dtype=np.float64,
            copy=False,
        )
        return result

    def key(self) -> Tuple[Any, ...]:
        if (
            type(self.rows) is not int
            or self.rows < 0
            or type(self.columns) is not int
            or self.columns < 0
            or type(self.data_bytes) is not bytes
            or type(self.indices_bytes) is not bytes
            or type(self.indptr_bytes) is not bytes
        ):
            raise ConstraintBlockDAGCandidateError(
                "frozen CSR authority is not exact builtin bytes"
            )
        return (
            self.rows,
            self.columns,
            self.data_bytes,
            self.indices_bytes,
            self.indptr_bytes,
        )


@dataclass(frozen=True)
class _FrozenFloat64Bytes:
    length: int
    raw: bytes

    @classmethod
    def from_vector(
        cls,
        value: Any,
        *,
        name: str,
        finite: bool,
    ) -> "_FrozenFloat64Bytes":
        if type(value) is not np.ndarray or value.dtype != np.dtype(
            np.float64
        ) or value.ndim != 1:
            raise ConstraintBlockDAGCandidateError(
                f"{name} must be an exact rank-one float64 ndarray"
            )
        # Always detach before validation; a normal contiguous ndarray would
        # otherwise remain caller-owned until ``tobytes``.
        snapshot = np.array(value, dtype=np.float64, order="C", copy=True)
        if snapshot.ndim != 1:
            raise ConstraintBlockDAGCandidateError(
                f"{name} changed shape while being snapshotted"
            )
        if np.any(np.isnan(snapshot)) or (finite and not np.all(np.isfinite(snapshot))):
            raise ConstraintBlockDAGCandidateError(
                f"{name} contains a forbidden non-finite value"
            )
        return cls(int(snapshot.size), bytes(snapshot.tobytes(order="C")))

    def array(self) -> np.ndarray:
        return np.frombuffer(self.raw, dtype=np.float64)

    @property
    def payload_bytes(self) -> int:
        return len(self.raw)

    def key(self) -> Tuple[Any, ...]:
        if (
            type(self.length) is not int
            or self.length < 0
            or type(self.raw) is not bytes
        ):
            raise ConstraintBlockDAGCandidateError(
                "frozen vector authority is not exact builtin bytes"
            )
        return (self.length, self.raw)


@dataclass(frozen=True)
class _BandPayload:
    frame_continuous: Tuple[StableFactorID, ...]
    frame_binary: Tuple[StableFactorID, ...]
    A_cont: _FrozenCSRBytes
    A_bin: _FrozenCSRBytes
    lower: _FrozenFloat64Bytes
    upper: _FrozenFloat64Bytes
    original_rows: int
    ranged_mask: Tuple[bool, ...]
    reverse_positions: Tuple[int, ...]

    def __post_init__(self) -> None:
        stored_rows = self.A_cont.rows
        if (
            self.A_bin.rows != stored_rows
            or self.lower.length != stored_rows
            or self.upper.length != stored_rows
            or self.A_cont.columns != len(self.frame_continuous)
            or self.A_bin.columns != len(self.frame_binary)
            or type(self.original_rows) is not int
            or self.original_rows <= 0
            or stored_rows < self.original_rows
            or type(self.ranged_mask) is not tuple
            or len(self.ranged_mask) != self.original_rows
            or any(type(value) is not bool for value in self.ranged_mask)
            or type(self.reverse_positions) is not tuple
            or len(self.reverse_positions) != self.original_rows
        ):
            raise ConstraintBlockDAGCandidateError("malformed band payload")
        expected_reverse = self.original_rows
        for paired, position in zip(self.ranged_mask, self.reverse_positions):
            if type(position) is not int:
                raise ConstraintBlockDAGCandidateError(
                    "malformed reverse-row position"
                )
            if paired:
                if position != -1:
                    raise ConstraintBlockDAGCandidateError(
                        "ranged row unexpectedly stores a reverse payload"
                    )
            else:
                if position != expected_reverse:
                    raise ConstraintBlockDAGCandidateError(
                        "fallback reverse rows are not canonical"
                    )
                expected_reverse += 1
        if expected_reverse != stored_rows:
            raise ConstraintBlockDAGCandidateError(
                "fallback reverse-row count is inconsistent"
            )
        lower = self.lower.array()
        upper = self.upper.array()
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ConstraintBlockDAGCandidateError("band bounds contain NaN")
        if np.any(np.isposinf(lower)) or np.any(np.isneginf(upper)):
            raise ConstraintBlockDAGCandidateError(
                "band bounds contain the wrong infinite direction"
            )
        # ``lower > upper`` is a valid exact encoding of an empty band.  Do
        # not erase that proof-source contradiction or silently relax it.

    @property
    def stored_rows(self) -> int:
        return int(self.A_cont.rows)

    @property
    def stored_nnz(self) -> int:
        return int(self.A_cont.nnz + self.A_bin.nnz)

    @property
    def numeric_payload_bytes(self) -> int:
        return int(
            self.A_cont.payload_bytes
            + self.A_bin.payload_bytes
            + self.lower.payload_bytes
            + self.upper.payload_bytes
        )

    def key(self) -> Tuple[Any, ...]:
        if (
            type(self.frame_continuous) is not tuple
            or type(self.frame_binary) is not tuple
            or type(self.A_cont) is not _FrozenCSRBytes
            or type(self.A_bin) is not _FrozenCSRBytes
            or type(self.lower) is not _FrozenFloat64Bytes
            or type(self.upper) is not _FrozenFloat64Bytes
            or type(self.original_rows) is not int
            or type(self.ranged_mask) is not tuple
            or any(type(value) is not bool for value in self.ranged_mask)
            or type(self.reverse_positions) is not tuple
            or any(type(value) is not int for value in self.reverse_positions)
        ):
            raise ConstraintBlockDAGCandidateError(
                "band payload contains a substituted nested object"
            )
        for expected, values in (
            ("continuous", self.frame_continuous),
            ("binary", self.frame_binary),
        ):
            if any(
                type(value) is not StableFactorID
                or type(value.kind) is not str
                or value.kind != expected
                or type(value.value) is not int
                for value in values
            ):
                raise ConstraintBlockDAGCandidateError(
                    "band payload contains a substituted factor identity"
                )
        return (
            tuple((value.kind, value.value) for value in self.frame_continuous),
            tuple((value.kind, value.value) for value in self.frame_binary),
            _FrozenCSRBytes.key(self.A_cont),
            _FrozenCSRBytes.key(self.A_bin),
            _FrozenFloat64Bytes.key(self.lower),
            _FrozenFloat64Bytes.key(self.upper),
            self.original_rows,
            self.ranged_mask,
            self.reverse_positions,
        )


def _payload_digest(key: Tuple[Any, ...]) -> str:
    """Bucket selector only; callers must compare ``key`` in full."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if type(value) is tuple:
            digest.update(b"(")
            for item in value:
                update(item)
            digest.update(b")")
        elif type(value) is bytes:
            digest.update(b"b")
            digest.update(len(value).to_bytes(8, "little", signed=False))
            digest.update(value)
        elif type(value) is str:
            encoded = value.encode("utf-8")
            digest.update(b"s")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
        elif type(value) is bool:
            digest.update(b"t" if value else b"f")
        elif type(value) is int:
            digest.update(b"i")
            digest.update(int(value).to_bytes(16, "little", signed=True))
        else:  # pragma: no cover - keys above are closed over exact builtins.
            raise TypeError(f"unsupported digest key type {type(value).__name__}")

    update(key)
    return digest.hexdigest()


def _view_digest(key: Tuple[Tuple[str, int], ...]) -> str:
    return _payload_digest(("view", key))


@dataclass(frozen=True)
class _InternedPayload:
    payload_id: StableObjectID
    payload: _BandPayload
    full_key: Tuple[Any, ...]

    def validate_live(self) -> None:
        # ``frozen=True`` is not an authority boundary against
        # ``object.__setattr__``.  Recompute the complete bytes key whenever a
        # payload is reused, sealed, or replayed.
        if (
            type(self.payload) is not _BandPayload
            or type(self.full_key) is not tuple
            or _BandPayload.key(self.payload) != self.full_key
        ):
            raise ConstraintBlockDAGCandidateError(
                "interned payload was mutated after publication"
            )
        _stable_object_key(
            self.payload_id, allowed_kinds=frozenset({"payload"})
        )


@dataclass(frozen=True)
class ConstraintRowHandle:
    """Stable source-row or virtual-facet handle bound to one arena token."""

    row_id: StableObjectID
    block_id: StableObjectID
    original_row: int
    role: str
    tag: str
    _arena_token: Any

    def __post_init__(self) -> None:
        if (
            type(self.row_id) is not StableObjectID
            or self.row_id.kind
            not in {"facet", "le_row", "range_row"}
            or type(self.block_id) is not StableObjectID
            or self.block_id.kind != "block"
            or type(self.original_row) is not int
            or self.original_row < 0
            or type(self.role) is not str
            or self.role not in {"forward", "reverse", "range", "le"}
            or type(self.tag) is not str
            or not self.tag
            or type(self._arena_token) is not _ArenaToken
        ):
            raise ConstraintBlockDAGCandidateError("malformed row handle")


@dataclass(frozen=True)
class ConstraintBlockHandle:
    block_id: StableObjectID
    _arena_token: Any

    def __post_init__(self) -> None:
        if (
            type(self.block_id) is not StableObjectID
            or self.block_id.kind != "block"
            or type(self._arena_token) is not _ArenaToken
        ):
            raise ConstraintBlockDAGCandidateError("malformed block handle")


@dataclass(frozen=True)
class ConstraintView:
    """Canonical immutable block-occurrence set for one exact arena."""

    block_ids: Tuple[StableObjectID, ...]
    view_id: StableObjectID
    _arena_token: Any

    def __post_init__(self) -> None:
        if (
            type(self.block_ids) is not tuple
            or any(
                type(value) is not StableObjectID or value.kind != "block"
                for value in self.block_ids
            )
            or self.block_ids != tuple(sorted(set(self.block_ids)))
            or type(self.view_id) is not StableObjectID
            or self.view_id.kind != "view"
            or type(self._arena_token) is not _ArenaToken
        ):
            raise ConstraintBlockDAGCandidateError("malformed constraint view")


@dataclass(frozen=True)
class _BlockOccurrence:
    block_id: StableObjectID
    payload: _InternedPayload
    base_tag: str
    layer_id: int
    source_row_handles: Tuple[ConstraintRowHandle, ...]
    forward_facets: Tuple[ConstraintRowHandle, ...]
    reverse_facets: Tuple[ConstraintRowHandle, ...]

    def key(self) -> Tuple[Any, ...]:
        def row_key(row: ConstraintRowHandle) -> Tuple[Any, ...]:
            if (
                type(row) is not ConstraintRowHandle
                or type(row.row_id) is not StableObjectID
                or type(row.block_id) is not StableObjectID
                or type(row.original_row) is not int
                or type(row.role) is not str
                or type(row.tag) is not str
                or type(row._arena_token) is not _ArenaToken
            ):
                raise ConstraintBlockDAGCandidateError(
                    "block contains a substituted row handle"
                )
            row_id_key = _stable_object_key(
                row.row_id,
                allowed_kinds=frozenset({"facet", "le_row", "range_row"}),
            )
            row_block_key = _stable_object_key(
                row.block_id, allowed_kinds=frozenset({"block"})
            )
            return (
                row_id_key[0],
                row_id_key[1],
                row_block_key[0],
                row_block_key[1],
                row.original_row,
                row.role,
                row.tag,
            )

        if (
            type(self.payload) is not _InternedPayload
            or type(self.base_tag) is not str
            or type(self.layer_id) is not int
            or type(self.source_row_handles) is not tuple
            or type(self.forward_facets) is not tuple
            or type(self.reverse_facets) is not tuple
        ):
            raise ConstraintBlockDAGCandidateError(
                "block occurrence contains a substituted nested object"
            )
        block_id_key = _stable_object_key(
            self.block_id, allowed_kinds=frozenset({"block"})
        )
        _InternedPayload.validate_live(self.payload)
        payload_id_key = _stable_object_key(
            self.payload.payload_id, allowed_kinds=frozenset({"payload"})
        )
        return (
            block_id_key[0],
            block_id_key[1],
            payload_id_key[0],
            payload_id_key[1],
            self.payload.full_key,
            _BandPayload.key(self.payload.payload),
            self.base_tag,
            self.layer_id,
            tuple(row_key(row) for row in self.source_row_handles),
            tuple(row_key(row) for row in self.forward_facets),
            tuple(row_key(row) for row in self.reverse_facets),
        )


@dataclass(frozen=True)
class GuardedBandAppend:
    """Result of one exact source-band transaction."""

    view: ConstraintView
    block: ConstraintBlockHandle
    source_rows: Tuple[ConstraintRowHandle, ...]
    forward_facets: Tuple[ConstraintRowHandle, ...]
    reverse_facets: Tuple[ConstraintRowHandle, ...]
    ranged_rows: int
    fallback_pairs: int
    stored_nnz: int
    virtual_nnz: int

    @property
    def candidate_only(self) -> bool:
        return True

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False

    @property
    def production_integration(self) -> bool:
        return False


def _csr_bitwise_negative_rows(
    forward: _FrozenCSRBytes,
    reverse: _FrozenCSRBytes,
) -> np.ndarray:
    if forward.rows != reverse.rows or forward.columns != reverse.columns:
        raise ConstraintBlockDAGCandidateError(
            "forward/reverse CSR shapes differ"
        )
    rows = forward.rows
    result = np.ones(rows, dtype=bool)
    f_indptr = np.frombuffer(forward.indptr_bytes, dtype=np.int64)
    r_indptr = np.frombuffer(reverse.indptr_bytes, dtype=np.int64)
    f_indices = np.frombuffer(forward.indices_bytes, dtype=np.int64)
    r_indices = np.frombuffer(reverse.indices_bytes, dtype=np.int64)
    f_data = np.frombuffer(forward.data_bytes, dtype=np.float64)
    r_data = np.frombuffer(reverse.data_bytes, dtype=np.float64)

    # The common direct-band path has exactly identical CSR structure.  Keep
    # it vectorized so validating a C89-ratio block is O(nnz) in C loops.
    if np.array_equal(f_indptr, r_indptr) and np.array_equal(f_indices, r_indices):
        equal_bits = np.equal(
            np.negative(f_data).view(np.uint64), r_data.view(np.uint64)
        )
        if not np.all(equal_bits):
            bad_entries = np.flatnonzero(~equal_bits)
            bad_rows = np.searchsorted(
                f_indptr[1:], bad_entries, side="right"
            )
            result[bad_rows] = False
        return result

    # Mixed fallback is uncommon and bounded.  Compare exact row slices.
    for row in range(rows):
        fs, fe = int(f_indptr[row]), int(f_indptr[row + 1])
        rs, re = int(r_indptr[row]), int(r_indptr[row + 1])
        if (
            fe - fs != re - rs
            or not np.array_equal(f_indices[fs:fe], r_indices[rs:re])
            or not np.array_equal(
                np.negative(f_data[fs:fe]).view(np.uint64),
                r_data[rs:re].view(np.uint64),
            )
        ):
            result[row] = False
    return result


def _take_csr_rows(
    matrix: sp.csr_matrix,
    rows: np.ndarray,
) -> sp.csr_matrix:
    if rows.size == 0:
        return sp.csr_matrix((0, matrix.shape[1]), dtype=np.float64)
    result = matrix[rows, :].tocsr()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _build_band_payload(
    *,
    frame: ExactFactorFrame,
    forward_cont: Any,
    forward_bin: Any,
    forward_upper: Any,
    reverse_cont: Any,
    reverse_bin: Any,
    reverse_upper: Any,
    allow_range: bool,
) -> Tuple[_BandPayload, int]:
    if type(allow_range) is not bool:
        raise ConstraintBlockDAGCandidateError("allow_range must be bool")
    # Snapshot every caller-owned value before comparison.  Concurrent or
    # subsequent mutation of the inputs cannot alter the published source.
    fc = _FrozenCSRBytes.from_csr(forward_cont, name="forward_cont")
    fb = _FrozenCSRBytes.from_csr(forward_bin, name="forward_bin")
    rc = _FrozenCSRBytes.from_csr(reverse_cont, name="reverse_cont")
    rb = _FrozenCSRBytes.from_csr(reverse_bin, name="reverse_bin")
    fu = _FrozenFloat64Bytes.from_vector(
        forward_upper, name="forward_upper", finite=True
    )
    ru = _FrozenFloat64Bytes.from_vector(
        reverse_upper, name="reverse_upper", finite=True
    )
    rows = fc.rows
    if (
        rows <= 0
        or fb.rows != rows
        or rc.rows != rows
        or rb.rows != rows
        or fu.length != rows
        or ru.length != rows
        or fc.columns != len(frame.continuous_ids)
        or rc.columns != len(frame.continuous_ids)
        or fb.columns != len(frame.binary_ids)
        or rb.columns != len(frame.binary_ids)
    ):
        raise ConstraintBlockDAGCandidateError(
            "guarded-band shape does not match its factor frame"
        )
    paired = (
        _csr_bitwise_negative_rows(fc, rc)
        & _csr_bitwise_negative_rows(fb, rb)
        if allow_range
        else np.zeros(rows, dtype=bool)
    )
    fallback_rows = np.flatnonzero(~paired).astype(np.int64, copy=False)
    if fallback_rows.size:
        fc_matrix = fc.csr()
        fb_matrix = fb.csr()
        source_cont = sp.vstack(
            (fc_matrix, _take_csr_rows(rc.csr(), fallback_rows)),
            format="csr",
        )
        source_bin = sp.vstack(
            (fb_matrix, _take_csr_rows(rb.csr(), fallback_rows)),
            format="csr",
        )
        frozen_source_cont = _FrozenCSRBytes.from_csr(
            source_cont, name="source_cont"
        )
        frozen_source_bin = _FrozenCSRBytes.from_csr(
            source_bin, name="source_bin"
        )
    else:
        # Transfer the already-sealed forward payload directly.  Rebuilding a
        # second CSR/bytes image here would be the post-hoc copy this source
        # design exists to avoid.
        frozen_source_cont = fc
        frozen_source_bin = fb

    f_upper = fu.array()
    r_upper = ru.array()
    lower_first = np.full(rows, -np.inf, dtype=np.float64)
    lower_first[paired] = np.negative(r_upper[paired])
    lower = np.concatenate(
        (lower_first, np.full(fallback_rows.size, -np.inf, dtype=np.float64))
    )
    upper = np.concatenate((f_upper, r_upper[fallback_rows]))
    reverse_positions = np.full(rows, -1, dtype=np.int64)
    reverse_positions[fallback_rows] = np.arange(
        rows, rows + fallback_rows.size, dtype=np.int64
    )
    payload = _BandPayload(
        tuple(frame.continuous_ids),
        tuple(frame.binary_ids),
        frozen_source_cont,
        frozen_source_bin,
        _FrozenFloat64Bytes.from_vector(lower, name="source_lower", finite=False),
        _FrozenFloat64Bytes.from_vector(upper, name="source_upper", finite=True),
        rows,
        tuple(bool(value) for value in paired.tolist()),
        tuple(int(value) for value in reverse_positions.tolist()),
    )
    virtual_nnz = int(fc.nnz + fb.nnz + rc.nnz + rb.nnz)
    return payload, virtual_nnz


class ExactConstraintArena:
    """One owner-bound immutable-payload arena with canonical set views."""

    def __init__(self, owner: Any, *, _factory_token: Any) -> None:
        if (
            type(owner) is not ExactConstraintOwner
            or _factory_token is not owner._token
        ):
            raise ConstraintBlockDAGCandidateError(
                "arena must be created by its exact owner"
            )
        self._owner = owner
        self._token = _ArenaToken()
        self._payload_buckets: Dict[str, list[_InternedPayload]] = {}
        self._blocks: Dict[StableObjectID, _BlockOccurrence] = {}
        self._view_buckets: Dict[str, list[ConstraintView]] = {}
        self._view_truth: Dict[int, Tuple[Any, ...]] = {}
        self._sealed = False
        self._empty_view = self._intern_view(())

    def _require_open(self) -> None:
        self._owner._require_thread()
        if self._sealed:
            raise ConstraintBlockDAGCandidateError("arena is already sealed")

    def _intern_payload(self, payload: _BandPayload) -> _InternedPayload:
        full_key = payload.key()
        bucket_key = _payload_digest(full_key)
        bucket = self._payload_buckets.setdefault(bucket_key, [])
        for existing in bucket:
            existing.validate_live()
            # The digest is deliberately insufficient.  This comparison is
            # over all canonical IDs, shapes, CSR bytes, bounds, and maps.
            if existing.full_key == full_key:
                return existing
        interned = _InternedPayload(
            _fresh_object_id("payload"), payload, full_key
        )
        bucket.append(interned)
        return interned

    def _intern_view(
        self, block_ids: Tuple[StableObjectID, ...]
    ) -> ConstraintView:
        normalized = tuple(sorted(set(block_ids)))
        key = tuple((value.kind, value.value) for value in normalized)
        bucket = self._view_buckets.setdefault(_view_digest(key), [])
        for existing in bucket:
            if tuple(
                (value.kind, value.value) for value in existing.block_ids
            ) == key:
                return existing
        view = ConstraintView(
            normalized, _fresh_object_id("view"), self._token
        )
        self._view_truth[id(view)] = (
            view,
            view.block_ids,
            view.view_id,
            key,
        )
        bucket.append(view)
        return view

    def _validate_view(
        self, view: Any
    ) -> Tuple[StableObjectID, ...]:
        if type(view) is not ConstraintView:
            raise ConstraintBlockDAGCandidateError(
                "constraint view has the wrong exact type"
            )
        if view._arena_token is not self._token:
            raise ConstraintArenaMismatch(
                "constraint views from different arenas cannot be unioned"
            )
        record = self._view_truth.get(id(view))
        if (
            record is None
            or record[0] is not view
            or type(view.block_ids) is not tuple
            or view.block_ids is not record[1]
            or type(view.view_id) is not StableObjectID
            or view.view_id is not record[2]
        ):
            raise ConstraintBlockDAGCandidateError(
                "constraint view is forged or was mutated after interning"
            )
        _stable_object_key(
            view.view_id, allowed_kinds=frozenset({"view"})
        )
        for block_id in view.block_ids:
            _stable_object_key(
                block_id, allowed_kinds=frozenset({"block"})
            )
        key = tuple((value.kind, value.value) for value in view.block_ids)
        if (
            record[3] != key
            or any(value not in self._blocks for value in view.block_ids)
        ):
            raise ConstraintBlockDAGCandidateError(
                "constraint view is forged or stale"
            )
        # Consumers receive the canonical registry tuple, never a field read
        # from the caller-owned handle after validation.
        return record[1]

    @property
    def empty_view(self) -> ConstraintView:
        return self._empty_view

    def union(self, *views: ConstraintView) -> ConstraintView:
        self._require_open()
        if type(views) is not tuple or not views:
            raise ConstraintBlockDAGCandidateError(
                "union requires at least one exact view"
            )
        ids = []
        for view in views:
            ids.extend(self._validate_view(view))
        return self._intern_view(tuple(ids))

    def append_guarded_band(
        self,
        view: ConstraintView,
        *,
        frame: ExactFactorFrame,
        forward_cont: sp.csr_matrix,
        forward_bin: sp.csr_matrix,
        forward_upper: np.ndarray,
        reverse_cont: sp.csr_matrix,
        reverse_bin: sp.csr_matrix,
        reverse_upper: np.ndarray,
        layer_id: int,
        family: str = _FAMILY_ADD_MATERIALIZE,
    ) -> GuardedBandAppend:
        return self._append_guarded_band(
            view,
            frame=frame,
            forward_cont=forward_cont,
            forward_bin=forward_bin,
            forward_upper=forward_upper,
            reverse_cont=reverse_cont,
            reverse_bin=reverse_bin,
            reverse_upper=reverse_upper,
            layer_id=layer_id,
            family=family,
            allow_range=True,
        )

    def _append_guarded_band(
        self,
        view: ConstraintView,
        *,
        frame: ExactFactorFrame,
        forward_cont: sp.csr_matrix,
        forward_bin: sp.csr_matrix,
        forward_upper: np.ndarray,
        reverse_cont: sp.csr_matrix,
        reverse_bin: sp.csr_matrix,
        reverse_upper: np.ndarray,
        layer_id: int,
        family: str,
        allow_range: bool,
    ) -> GuardedBandAppend:
        self._require_open()
        current_ids = self._validate_view(view)
        frame = self._owner._validate_frame(frame)
        layer_id = _exact_nonnegative_int(layer_id, name="layer_id")
        if type(family) is not str or family != _FAMILY_ADD_MATERIALIZE:
            raise ConstraintBlockDAGCandidateError(
                "only the internal add_materialize family is accepted"
            )
        payload, virtual_nnz = _build_band_payload(
            frame=frame,
            forward_cont=forward_cont,
            forward_bin=forward_bin,
            forward_upper=forward_upper,
            reverse_cont=reverse_cont,
            reverse_bin=reverse_bin,
            reverse_upper=reverse_upper,
            allow_range=allow_range,
        )
        interned = self._intern_payload(payload)
        block_id = _fresh_object_id("block")
        base_tag = f"{family}:{layer_id}"
        facet_start, facet_stop = _reserve_ids(
            "facet", 2 * payload.original_rows
        )
        facets = tuple(
            StableObjectID("facet", value)
            for value in range(facet_start, facet_stop)
        )
        forward_facets = tuple(
            ConstraintRowHandle(
                facets[row],
                block_id,
                row,
                "forward",
                f"{base_tag}:forward",
                self._token,
            )
            for row in range(payload.original_rows)
        )
        reverse_facets = tuple(
            ConstraintRowHandle(
                facets[payload.original_rows + row],
                block_id,
                row,
                "reverse",
                f"{base_tag}:reverse",
                self._token,
            )
            for row in range(payload.original_rows)
        )

        source_handles = []
        fallback_reverse_handles = []
        for row, ranged in enumerate(payload.ranged_mask):
            kind = "range_row" if ranged else "le_row"
            row_id = _fresh_object_id(kind)
            source_handles.append(
                ConstraintRowHandle(
                    row_id,
                    block_id,
                    row,
                    "range" if ranged else "le",
                    f"range:{base_tag}" if ranged else f"{base_tag}:forward",
                    self._token,
                )
            )
        for row, ranged in enumerate(payload.ranged_mask):
            if ranged:
                continue
            fallback_reverse_handles.append(
                ConstraintRowHandle(
                    _fresh_object_id("le_row"),
                    block_id,
                    row,
                    "le",
                    f"{base_tag}:reverse",
                    self._token,
                )
            )
        source_handles.extend(fallback_reverse_handles)
        occurrence = _BlockOccurrence(
            block_id,
            interned,
            base_tag,
            layer_id,
            tuple(source_handles),
            forward_facets,
            reverse_facets,
        )
        self._blocks[block_id] = occurrence
        result_view = self._intern_view(current_ids + (block_id,))
        return GuardedBandAppend(
            result_view,
            ConstraintBlockHandle(block_id, self._token),
            occurrence.source_row_handles,
            forward_facets,
            reverse_facets,
            int(sum(payload.ranged_mask)),
            int(payload.original_rows - sum(payload.ranged_mask)),
            payload.stored_nnz,
            virtual_nnz,
        )

    def seal(
        self,
        view: ConstraintView,
        *,
        final_frame: ExactFactorFrame,
    ) -> "ExactConstraintProgram":
        self._require_open()
        selected_ids = self._validate_view(view)
        frame = self._owner._validate_frame(final_frame)
        final_cont = tuple(frame.continuous_ids)
        final_bin = tuple(frame.binary_ids)
        occurrences = tuple(self._blocks[value] for value in selected_ids)
        for occurrence in occurrences:
            occurrence.payload.validate_live()
            payload = occurrence.payload.payload
            if (
                final_cont[: len(payload.frame_continuous)]
                != payload.frame_continuous
                or final_bin[: len(payload.frame_binary)]
                != payload.frame_binary
            ):
                raise ConstraintBlockDAGCandidateError(
                    "sealed block is not a prefix of the final factor frame"
                )
        self._sealed = True
        self._owner._seal()
        program = ExactConstraintProgram(
            occurrences,
            frame,
            self._token,
            _factory_token=self._owner._token,
        )
        _register_program(program)
        return program

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def production_integration(self) -> bool:
        return False


@dataclass(frozen=True)
class NativeConstraintBlock:
    """Detached native ranged/LE block reconstructed from sealed bytes."""

    A_cont: sp.csr_matrix
    A_bin: sp.csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    row_ids: Tuple[StableObjectID, ...]
    row_tags: Tuple[str, ...]
    block_id: StableObjectID

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False


@dataclass(frozen=True)
class VirtualFacetFrame:
    """Independent expanded two-LE replay of one sealed program."""

    A_cont: sp.csr_matrix
    A_bin: sp.csr_matrix
    upper: np.ndarray
    row_ids: Tuple[StableObjectID, ...]
    row_tags: Tuple[str, ...]
    continuous_ids: Tuple[StableFactorID, ...]
    binary_ids: Tuple[StableFactorID, ...]

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False


class VirtualFacetBatch:
    """One immutable bytes-backed slice of legacy virtual-facet order."""

    __slots__ = (
        "_A_cont",
        "_A_bin",
        "_upper",
        "_row_id_keys",
        "_row_tags",
        "_continuous_id_keys",
        "_binary_id_keys",
        "_row_offset",
        "_total_rows",
        "_sealed",
    )

    def __init__(
        self,
        *,
        A_cont: _FrozenCSRBytes,
        A_bin: _FrozenCSRBytes,
        upper: _FrozenFloat64Bytes,
        row_id_keys: Tuple[Tuple[str, int], ...],
        row_tags: Tuple[str, ...],
        continuous_id_keys: Tuple[Tuple[str, int], ...],
        binary_id_keys: Tuple[Tuple[str, int], ...],
        row_offset: int,
        total_rows: int,
    ) -> None:
        rows = A_cont.rows if type(A_cont) is _FrozenCSRBytes else -1
        if (
            type(A_cont) is not _FrozenCSRBytes
            or type(A_bin) is not _FrozenCSRBytes
            or type(upper) is not _FrozenFloat64Bytes
            or A_bin.rows != rows
            or upper.length != rows
            or not 0 < rows <= 256
            or type(row_id_keys) is not tuple
            or len(row_id_keys) != rows
            or any(
                type(item) is not tuple
                or len(item) != 2
                or item[0] != "facet"
                or type(item[0]) is not str
                or type(item[1]) is not int
                or item[1] < 0
                for item in row_id_keys
            )
            or type(row_tags) is not tuple
            or len(row_tags) != rows
            or any(type(item) is not str or not item for item in row_tags)
            or type(continuous_id_keys) is not tuple
            or type(binary_id_keys) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or item[0] != expected
                or type(item[0]) is not str
                or type(item[1]) is not int
                or item[1] < 0
                for expected, values in (
                    ("continuous", continuous_id_keys),
                    ("binary", binary_id_keys),
                )
                for item in values
            )
            or A_cont.columns != len(continuous_id_keys)
            or A_bin.columns != len(binary_id_keys)
            or type(row_offset) is not int
            or type(total_rows) is not int
            or row_offset < 0
            or total_rows <= 0
            or row_offset + rows > total_rows
        ):
            raise ConstraintBlockDAGCandidateError(
                "malformed virtual-facet batch"
            )
        self._A_cont = A_cont
        self._A_bin = A_bin
        self._upper = upper
        self._row_id_keys = row_id_keys
        self._row_tags = row_tags
        self._continuous_id_keys = continuous_id_keys
        self._binary_id_keys = binary_id_keys
        self._row_offset = row_offset
        self._total_rows = total_rows
        self._sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("virtual-facet batch is immutable")
        object.__setattr__(self, name, value)

    @property
    def A_cont(self) -> sp.csr_matrix:
        return self._A_cont.csr()

    @property
    def A_bin(self) -> sp.csr_matrix:
        return self._A_bin.csr()

    @property
    def upper(self) -> np.ndarray:
        result = self._upper.array()
        result.setflags(write=False)
        return result

    @property
    def row_ids(self) -> Tuple[StableObjectID, ...]:
        return tuple(StableObjectID(kind, value) for kind, value in self._row_id_keys)

    @property
    def row_tags(self) -> Tuple[str, ...]:
        return self._row_tags

    @property
    def continuous_ids(self) -> Tuple[StableFactorID, ...]:
        return tuple(
            StableFactorID(kind, value)
            for kind, value in self._continuous_id_keys
        )

    @property
    def binary_ids(self) -> Tuple[StableFactorID, ...]:
        return tuple(
            StableFactorID(kind, value) for kind, value in self._binary_id_keys
        )

    @property
    def row_offset(self) -> int:
        return self._row_offset

    @property
    def row_count(self) -> int:
        return self._upper.length

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def bytes_backed(self) -> bool:
        return True

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False


class ExactConstraintProgram:
    """Sealed, bytes-backed, non-authoritative source program."""

    def __init__(
        self,
        blocks: Tuple[_BlockOccurrence, ...],
        frame: ExactFactorFrame,
        arena_token: Any,
        *,
        _factory_token: Any,
    ) -> None:
        if (
            type(blocks) is not tuple
            or type(frame) is not ExactFactorFrame
            or type(arena_token) is not _ArenaToken
            or type(_factory_token) is not _OwnerToken
        ):
            raise ConstraintBlockDAGCandidateError(
                "program must be sealed by an exact arena"
            )
        self._blocks = blocks
        self._frame = frame
        self._arena_token = arena_token
        self._factory_token = _factory_token
        self._schema = _PROGRAM_SCHEMA

    @property
    def candidate_only(self) -> bool:
        return True

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False

    @property
    def authenticity_authority(self) -> bool:
        return False

    @property
    def production_integration(self) -> bool:
        return False

    @property
    def schema(self) -> str:
        return _PROGRAM_SCHEMA

    @property
    def continuous_ids(self) -> Tuple[StableFactorID, ...]:
        _validate_program(self)
        return tuple(self._frame.continuous_ids)

    @property
    def binary_ids(self) -> Tuple[StableFactorID, ...]:
        _validate_program(self)
        return tuple(self._frame.binary_ids)

    @property
    def block_count(self) -> int:
        _validate_program(self)
        return len(self._blocks)

    @property
    def source_rows(self) -> int:
        _validate_program(self)
        return int(
            sum(block.payload.payload.stored_rows for block in self._blocks)
        )

    @property
    def virtual_facet_rows(self) -> int:
        _validate_program(self)
        return int(
            sum(
                2 * block.payload.payload.original_rows
                for block in self._blocks
            )
        )

    @property
    def source_nnz(self) -> int:
        _validate_program(self)
        return int(
            sum(block.payload.payload.stored_nnz for block in self._blocks)
        )

    @property
    def numeric_payload_bytes(self) -> int:
        _validate_program(self)
        seen = set()
        total = 0
        for block in self._blocks:
            key = block.payload.payload_id
            if key in seen:
                continue
            seen.add(key)
            total += block.payload.payload.numeric_payload_bytes
        return int(total)

    @property
    def ranged_rows(self) -> int:
        _validate_program(self)
        return int(
            sum(sum(block.payload.payload.ranged_mask) for block in self._blocks)
        )

    @property
    def fallback_pairs(self) -> int:
        _validate_program(self)
        return int(
            sum(
                block.payload.payload.original_rows
                - sum(block.payload.payload.ranged_mask)
                for block in self._blocks
            )
        )

    @property
    def receipt(self) -> Mapping[str, Any]:
        _validate_program(self)
        return MappingProxyType(
            {
                "schema": _SCHEMA,
                "program_schema": _PROGRAM_SCHEMA,
                "candidate_only": True,
                "proof_authority": False,
                "verdict_authority": False,
                "authenticity_authority": False,
                "production_integration": False,
                "hash_is_identity_authority": False,
                "bytes_backed_source": True,
                "virtual_facets_replayable": True,
                "stream_virtual_facets_replayable": True,
                "full_expanded_gate_closed": True,
                "block_count": int(self.block_count),
                "source_rows": int(self.source_rows),
                "virtual_facet_rows": int(self.virtual_facet_rows),
                "source_nnz": int(self.source_nnz),
                "numeric_payload_bytes": int(self.numeric_payload_bytes),
                "ranged_rows": int(self.ranged_rows),
                "fallback_pairs": int(self.fallback_pairs),
                "triangle_relaxation_called": False,
                "branch_and_bound_called": False,
                "backward_called": False,
                "dual_called": False,
                "real_model_called": False,
                "large_model_called": False,
            }
        )

    def native_blocks(self) -> Tuple[NativeConstraintBlock, ...]:
        return native_blocks(self)

    def replay_virtual_facets(self) -> VirtualFacetFrame:
        return replay_virtual_facets(self)

    def iter_virtual_facet_batches(
        self, *, max_rows: int
    ) -> Iterator[VirtualFacetBatch]:
        return iter_virtual_facet_batches(self, max_rows=max_rows)


def _program_graph_objects(program: ExactConstraintProgram) -> Tuple[Any, ...]:
    """Return the exact sealed object graph whose identities are authoritative."""

    if (
        type(program) is not ExactConstraintProgram
        or type(program._blocks) is not tuple
        or type(program._frame) is not ExactFactorFrame
        or type(program._arena_token) is not _ArenaToken
        or type(program._factory_token) is not _OwnerToken
        or type(program._schema) is not str
    ):
        raise ConstraintBlockDAGCandidateError(
            "sealed program contains a substituted top-level object"
        )
    _factor_frame_key(program._frame)
    graph = [
        program._blocks,
        program._frame,
        program._frame.continuous_ids,
        program._frame.binary_ids,
        program._frame._owner_token,
        program._arena_token,
        program._factory_token,
    ]
    graph.extend(program._frame.continuous_ids)
    graph.extend(program._frame.binary_ids)
    for block in program._blocks:
        if type(block) is not _BlockOccurrence:
            raise ConstraintBlockDAGCandidateError(
                "sealed program contains a noncanonical block"
            )
        _BlockOccurrence.key(block)
        interned = block.payload
        payload = interned.payload
        graph.extend(
            (
                block,
                block.block_id,
                interned,
                interned.payload_id,
                interned.full_key,
                payload,
                payload.frame_continuous,
                payload.frame_binary,
                payload.A_cont,
                payload.A_cont.data_bytes,
                payload.A_cont.indices_bytes,
                payload.A_cont.indptr_bytes,
                payload.A_bin,
                payload.A_bin.data_bytes,
                payload.A_bin.indices_bytes,
                payload.A_bin.indptr_bytes,
                payload.lower,
                payload.lower.raw,
                payload.upper,
                payload.upper.raw,
                payload.ranged_mask,
                payload.reverse_positions,
                block.source_row_handles,
                block.forward_facets,
                block.reverse_facets,
            )
        )
        graph.extend(payload.frame_continuous)
        graph.extend(payload.frame_binary)
        for handles in (
            block.source_row_handles,
            block.forward_facets,
            block.reverse_facets,
        ):
            for handle in handles:
                if (
                    type(handle) is not ConstraintRowHandle
                    or handle._arena_token is not program._arena_token
                    or handle.block_id is not block.block_id
                ):
                    raise ConstraintBlockDAGCandidateError(
                        "sealed block contains a foreign or substituted row"
                    )
                graph.extend(
                    (
                        handle,
                        handle.row_id,
                        handle.block_id,
                        handle._arena_token,
                    )
                )
    return tuple(graph)


def _program_key_from_validated(
    program: ExactConstraintProgram,
) -> Tuple[Any, ...]:
    return (
        program._schema,
        _factor_frame_key(program._frame),
        tuple(_BlockOccurrence.key(block) for block in program._blocks),
        id(program._arena_token),
        id(program._factory_token),
    )


def _program_key(program: ExactConstraintProgram) -> Tuple[Any, ...]:
    _program_graph_objects(program)
    return _program_key_from_validated(program)


def _register_program(program: ExactConstraintProgram) -> None:
    object_id = id(program)

    def cleanup(reference: Any) -> None:
        with _PROGRAM_REGISTRY_LOCK:
            current = _PROGRAM_REGISTRY.get(object_id)
            # ABA guard: a delayed stale weakref callback must never delete a
            # newer record that happens to reuse the same CPython object ID.
            if current is not None and current[0] is reference:
                _PROGRAM_REGISTRY.pop(object_id, None)

    graph = _program_graph_objects(program)
    record = (
        weakref.ref(program, cleanup),
        graph,
        _program_key_from_validated(program),
    )
    with _PROGRAM_REGISTRY_LOCK:
        _PROGRAM_REGISTRY[object_id] = record


def _validated_program_record(value: Any) -> Tuple[Any, ...]:
    if type(value) is not ExactConstraintProgram:
        raise ConstraintBlockDAGCandidateError(
            "constraint program has the wrong exact type"
        )
    with _PROGRAM_REGISTRY_LOCK:
        record = _PROGRAM_REGISTRY.get(id(value))
    try:
        graph = _program_graph_objects(value)
        key = _program_key_from_validated(value)
    except (ConstraintBlockDAGCandidateError, AttributeError, TypeError):
        graph = ()
        key = ()
    graph_matches = bool(
        record is not None
        and len(graph) == len(record[1])
        and all(current is sealed for current, sealed in zip(graph, record[1]))
    )
    if (
        record is None
        or record[0]() is not value
        or not graph_matches
        or value._schema != _PROGRAM_SCHEMA
        or key != record[2]
    ):
        raise ConstraintBlockDAGCandidateError(
            "constraint program is forged or was mutated after sealing"
        )
    return record


def _validate_program(value: Any) -> ExactConstraintProgram:
    _validated_program_record(value)
    return value


@dataclass(frozen=True)
class _ReplayBlockCapture:
    A_cont: _FrozenCSRBytes
    A_bin: _FrozenCSRBytes
    lower: _FrozenFloat64Bytes
    upper: _FrozenFloat64Bytes
    original_rows: int
    reverse_positions: Tuple[int, ...]
    forward_facet_keys: Tuple[Tuple[Any, ...], ...]
    reverse_facet_keys: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class _ProgramReplayCapture:
    blocks: Tuple[_ReplayBlockCapture, ...]
    continuous_id_keys: Tuple[Tuple[str, int], ...]
    binary_id_keys: Tuple[Tuple[str, int], ...]
    total_rows: int


def _capture_program_replay(value: Any) -> _ProgramReplayCapture:
    """Validate once, then detach replay from the public program handle."""

    record = _validated_program_record(value)
    canonical_blocks = record[1][0]
    canonical_frame_key = record[2][1]
    canonical_block_keys = record[2][2]
    blocks = []
    total_rows = 0
    for block, block_key in zip(canonical_blocks, canonical_block_keys):
        payload = block.payload.payload
        forward_facet_keys = block_key[-2]
        reverse_facet_keys = block_key[-1]
        if (
            type(forward_facet_keys) is not tuple
            or type(reverse_facet_keys) is not tuple
            or len(forward_facet_keys) != payload.original_rows
            or len(reverse_facet_keys) != payload.original_rows
        ):
            raise ConstraintBlockDAGCandidateError(
                "registry row-key truth is malformed"
            )
        blocks.append(
            _ReplayBlockCapture(
                _FrozenCSRBytes(
                    payload.A_cont.rows,
                    payload.A_cont.columns,
                    payload.A_cont.data_bytes,
                    payload.A_cont.indices_bytes,
                    payload.A_cont.indptr_bytes,
                ),
                _FrozenCSRBytes(
                    payload.A_bin.rows,
                    payload.A_bin.columns,
                    payload.A_bin.data_bytes,
                    payload.A_bin.indices_bytes,
                    payload.A_bin.indptr_bytes,
                ),
                _FrozenFloat64Bytes(
                    payload.lower.length, payload.lower.raw
                ),
                _FrozenFloat64Bytes(
                    payload.upper.length, payload.upper.raw
                ),
                payload.original_rows,
                payload.reverse_positions,
                forward_facet_keys,
                reverse_facet_keys,
            )
        )
        total_rows += 2 * payload.original_rows
    return _ProgramReplayCapture(
        tuple(blocks),
        canonical_frame_key[0],
        canonical_frame_key[1],
        int(total_rows),
    )


def _bitwise_sign_flip(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float64).view(np.uint64).copy()
    bits ^= np.uint64(1 << 63)
    return bits.view(np.float64)


class _VirtualFacetBatchIterator:
    """Closeable O(max_rows + batch_nnz) replay cursor."""

    __slots__ = (
        "_capture",
        "_max_rows",
        "_block_index",
        "_phase",
        "_row",
        "_offset",
        "_active_cont",
        "_active_bin",
        "_active_lower",
        "_active_upper",
        "_closed",
        "_lock",
        "__weakref__",
    )

    def __init__(
        self, capture: _ProgramReplayCapture, max_rows: int
    ) -> None:
        self._capture: Optional[_ProgramReplayCapture] = capture
        self._max_rows = max_rows
        self._block_index = 0
        self._phase = 0
        self._row = 0
        self._offset = 0
        self._active_cont: Optional[sp.csr_matrix] = None
        self._active_bin: Optional[sp.csr_matrix] = None
        self._active_lower: Optional[np.ndarray] = None
        self._active_upper: Optional[np.ndarray] = None
        self._closed = False
        self._lock = threading.Lock()

    def __iter__(self) -> "_VirtualFacetBatchIterator":
        return self

    def _close_locked(self) -> None:
        self._capture = None
        self._active_cont = None
        self._active_bin = None
        self._active_lower = None
        self._active_upper = None
        self._closed = True

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def __enter__(self) -> "_VirtualFacetBatchIterator":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> bool:
        self.close()
        return False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def __del__(self) -> None:  # pragma: no cover - deterministic close preferred.
        self._capture = None
        self._active_cont = None
        self._active_bin = None
        self._active_lower = None
        self._active_upper = None
        self._closed = True

    def __next__(self) -> VirtualFacetBatch:
        with self._lock:
            if self._closed or self._capture is None:
                raise StopIteration
            try:
                return self._next_locked()
            except StopIteration:
                self._close_locked()
                raise
            except BaseException:
                # KeyboardInterrupt and SystemExit are intentionally not
                # swallowed, but their captured graph references are dropped.
                self._close_locked()
                raise

    def _load_block(
        self, block: _ReplayBlockCapture, capture: _ProgramReplayCapture
    ) -> None:
        self._active_cont = block.A_cont.csr(
            columns=len(capture.continuous_id_keys)
        )
        self._active_bin = block.A_bin.csr(
            columns=len(capture.binary_id_keys)
        )
        self._active_lower = block.lower.array()
        self._active_upper = block.upper.array()

    def _advance_boundary(self, block: _ReplayBlockCapture) -> bool:
        if self._row < block.original_rows:
            return False
        if self._phase == 0:
            self._phase = 1
            self._row = 0
            return True
        self._block_index += 1
        self._phase = 0
        self._row = 0
        self._active_cont = None
        self._active_bin = None
        self._active_lower = None
        self._active_upper = None
        return True

    def _next_locked(self) -> VirtualFacetBatch:
        capture = self._capture
        if capture is None or self._offset >= capture.total_rows:
            raise StopIteration
        cont_rows = []
        bin_rows = []
        upper_bits = []
        row_id_keys = []
        row_tags = []
        batch_offset = self._offset
        while len(upper_bits) < self._max_rows:
            if self._block_index >= len(capture.blocks):
                break
            block = capture.blocks[self._block_index]
            if self._active_cont is None:
                self._load_block(block, capture)
            if self._advance_boundary(block):
                continue
            Ac = self._active_cont
            Ab = self._active_bin
            lower = self._active_lower
            upper = self._active_upper
            if Ac is None or Ab is None or lower is None or upper is None:
                raise ConstraintBlockDAGCandidateError(
                    "stream replay lost its active block snapshot"
                )
            row = self._row
            if self._phase == 0:
                position = row
                ci, cd = _csr_row(Ac, position)
                bi, bd = _csr_row(Ab, position)
                cont_rows.append((ci.copy(), cd.copy()))
                bin_rows.append((bi.copy(), bd.copy()))
                upper_bits.append(int(upper.view(np.uint64)[position]))
                facet_key = block.forward_facet_keys[row]
            else:
                position = block.reverse_positions[row]
                if position < 0:
                    ci, cd = _csr_row(Ac, row)
                    bi, bd = _csr_row(Ab, row)
                    cont_rows.append((ci.copy(), _bitwise_sign_flip(cd)))
                    bin_rows.append((bi.copy(), _bitwise_sign_flip(bd)))
                    upper_bits.append(
                        int(lower.view(np.uint64)[row]) ^ (1 << 63)
                    )
                else:
                    ci, cd = _csr_row(Ac, position)
                    bi, bd = _csr_row(Ab, position)
                    cont_rows.append((ci.copy(), cd.copy()))
                    bin_rows.append((bi.copy(), bd.copy()))
                    upper_bits.append(int(upper.view(np.uint64)[position]))
                facet_key = block.reverse_facet_keys[row]
            if (
                type(facet_key) is not tuple
                or len(facet_key) != 7
                or facet_key[0] != "facet"
                or type(facet_key[0]) is not str
                or type(facet_key[1]) is not int
                or type(facet_key[6]) is not str
                or not facet_key[6]
            ):
                raise ConstraintBlockDAGCandidateError(
                    "stream replay encountered malformed registry row truth"
                )
            row_id_keys.append((facet_key[0], facet_key[1]))
            row_tags.append(facet_key[6])
            self._row += 1
            self._offset += 1
        if not upper_bits:
            raise StopIteration
        cont = _assemble_rows(
            cont_rows, columns=len(capture.continuous_id_keys)
        )
        binary = _assemble_rows(
            bin_rows, columns=len(capture.binary_id_keys)
        )
        upper_array = np.asarray(upper_bits, dtype=np.uint64).view(np.float64)
        batch = VirtualFacetBatch(
            A_cont=_FrozenCSRBytes.from_csr(cont, name="stream_batch_cont"),
            A_bin=_FrozenCSRBytes.from_csr(binary, name="stream_batch_bin"),
            upper=_FrozenFloat64Bytes.from_vector(
                upper_array, name="stream_batch_upper", finite=True
            ),
            row_id_keys=tuple(row_id_keys),
            row_tags=tuple(row_tags),
            continuous_id_keys=capture.continuous_id_keys,
            binary_id_keys=capture.binary_id_keys,
            row_offset=batch_offset,
            total_rows=capture.total_rows,
        )
        if self._offset == capture.total_rows:
            self._close_locked()
        return batch


def iter_virtual_facet_batches(
    program: ExactConstraintProgram,
    *,
    max_rows: int,
) -> Iterator[VirtualFacetBatch]:
    """Replay exact virtual facets without constructing a full expanded CSR."""

    if type(max_rows) is not int or not 1 <= max_rows <= 256:
        raise ConstraintBlockDAGCandidateError(
            "max_rows must be an exact builtin int in [1, 256]"
        )
    # This synchronous capture is the sole registry/program validation for the
    # returned iterator.  Subsequent batches read only captured canonical
    # source references and bounded local buffers.
    capture = _capture_program_replay(program)
    return _VirtualFacetBatchIterator(capture, max_rows)


def _source_row_tags(block: _BlockOccurrence) -> Tuple[str, ...]:
    return tuple(handle.tag for handle in block.source_row_handles)


def native_blocks(
    program: ExactConstraintProgram,
) -> Tuple[NativeConstraintBlock, ...]:
    source = _validate_program(program)
    n_cont = len(source._frame.continuous_ids)
    n_bin = len(source._frame.binary_ids)
    result = []
    for block in source._blocks:
        payload = block.payload.payload
        lower = payload.lower.array()
        upper = payload.upper.array()
        lower.setflags(write=False)
        upper.setflags(write=False)
        result.append(
            NativeConstraintBlock(
                payload.A_cont.csr(columns=n_cont),
                payload.A_bin.csr(columns=n_bin),
                lower,
                upper,
                tuple(handle.row_id for handle in block.source_row_handles),
                _source_row_tags(block),
                block.block_id,
            )
        )
    return tuple(result)


def _csr_row(
    matrix: sp.csr_matrix,
    row: int,
) -> Tuple[np.ndarray, np.ndarray]:
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    return matrix.indices[start:stop], matrix.data[start:stop]


def _assemble_rows(
    rows: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    columns: int,
) -> sp.csr_matrix:
    indptr = np.empty(len(rows) + 1, dtype=np.int64)
    indptr[0] = 0
    for index, (cols, data) in enumerate(rows):
        if cols.size != data.size:
            raise ConstraintBlockDAGCandidateError(
                "virtual facet row has mismatched indices/data"
            )
        indptr[index + 1] = indptr[index] + cols.size
    total = int(indptr[-1])
    indices = np.empty(total, dtype=np.int64)
    data = np.empty(total, dtype=np.float64)
    cursor = 0
    for cols, values in rows:
        stop = cursor + cols.size
        indices[cursor:stop] = cols
        data[cursor:stop] = values
        cursor = stop
    matrix = sp.csr_matrix(
        (data, indices, indptr),
        shape=(len(rows), int(columns)),
        dtype=np.float64,
    )
    if not matrix.has_canonical_format or not matrix.has_sorted_indices:
        raise ConstraintBlockDAGCandidateError(
            "virtual facet replay produced noncanonical CSR"
        )
    return matrix


def replay_virtual_facets(program: ExactConstraintProgram) -> VirtualFacetFrame:
    source = _validate_program(program)
    cont_rows = []
    bin_rows = []
    uppers = []
    row_ids = []
    row_tags = []
    n_cont = len(source._frame.continuous_ids)
    n_bin = len(source._frame.binary_ids)
    for block in source._blocks:
        payload = block.payload.payload
        Ac = payload.A_cont.csr(columns=n_cont)
        Ab = payload.A_bin.csr(columns=n_bin)
        lower = payload.lower.array()
        upper = payload.upper.array()
        # Preserve the legacy block order: every forward row, then every
        # reverse row.  Coefficients and RHS values are reconstructed by exact
        # sign-bit negation only; no arithmetic summation or tolerance occurs.
        for row, handle in enumerate(block.forward_facets):
            ci, cd = _csr_row(Ac, row)
            bi, bd = _csr_row(Ab, row)
            cont_rows.append((ci.copy(), cd.copy()))
            bin_rows.append((bi.copy(), bd.copy()))
            uppers.append(float(upper[row]))
            row_ids.append(handle.row_id)
            row_tags.append(handle.tag)
        for row, handle in enumerate(block.reverse_facets):
            position = payload.reverse_positions[row]
            if position < 0:
                ci, cd = _csr_row(Ac, row)
                bi, bd = _csr_row(Ab, row)
                cont_rows.append((ci.copy(), np.negative(cd)))
                bin_rows.append((bi.copy(), np.negative(bd)))
                uppers.append(float(np.negative(lower[row])))
            else:
                ci, cd = _csr_row(Ac, position)
                bi, bd = _csr_row(Ab, position)
                cont_rows.append((ci.copy(), cd.copy()))
                bin_rows.append((bi.copy(), bd.copy()))
                uppers.append(float(upper[position]))
            row_ids.append(handle.row_id)
            row_tags.append(handle.tag)
    upper_array = np.asarray(uppers, dtype=np.float64)
    upper_array.setflags(write=False)
    return VirtualFacetFrame(
        _assemble_rows(cont_rows, columns=n_cont),
        _assemble_rows(bin_rows, columns=n_bin),
        upper_array,
        tuple(row_ids),
        tuple(row_tags),
        tuple(source._frame.continuous_ids),
        tuple(source._frame.binary_ids),
    )


def _c89_ratio_matrix(
    *,
    pair_count: int,
    columns: int,
) -> sp.csr_matrix:
    if pair_count <= 0 or columns < 74:
        raise ConstraintBlockDAGCandidateError(
            "bounded C89-ratio geometry is invalid"
        )
    # C89 ADD forward rows average 73.8375 nonzeros.  Scale the exact high-row
    # proportion without constructing a real network or its full matrix.
    high_rows = (pair_count * 34_304) // 40_960
    widths = np.full(pair_count, 73, dtype=np.int64)
    widths[:high_rows] = 74
    indptr = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(widths, dtype=np.int64))
    )
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    data = np.empty(int(indptr[-1]), dtype=np.float64)
    cursor = 0
    for row, width in enumerate(widths.tolist()):
        start = (row * 97) % columns
        row_indices = np.sort(
            (start + np.arange(width, dtype=np.int64) * 17) % columns
        )
        # columns>=74 and gcd(17, common synthetic widths) produces unique
        # values for the bounded geometries accepted below.
        if np.unique(row_indices).size != width:
            row_indices = np.arange(width, dtype=np.int64)
        stop = cursor + width
        indices[cursor:stop] = row_indices
        ordinal = np.arange(width, dtype=np.int64)
        values = ((ordinal % 29) + 1).astype(np.float64) / 32.0
        values[(ordinal + row) % 2 == 1] *= -1.0
        data[cursor:stop] = values
        cursor = stop
    return sp.csr_matrix(
        (data, indices, indptr),
        shape=(pair_count, columns),
        dtype=np.float64,
    )


def _prepare_benchmark_emitter(
    forward: sp.csr_matrix,
) -> Tuple[
    ExactFactorFrame,
    ExactConstraintArena,
    sp.csr_matrix,
    np.ndarray,
]:
    owner = ExactConstraintOwner()
    owner.allocate_continuous(int(forward.shape[1]))
    frame = owner.frame()
    arena = owner.new_arena()
    empty_bin = sp.csr_matrix((forward.shape[0], 0), dtype=np.float64)
    upper = np.full(forward.shape[0], 8.0, dtype=np.float64)
    return frame, arena, empty_bin, upper


def _emit_benchmark_program(
    *,
    prepared: Tuple[
        ExactFactorFrame,
        ExactConstraintArena,
        sp.csr_matrix,
        np.ndarray,
    ],
    forward: sp.csr_matrix,
    reverse: sp.csr_matrix,
    allow_range: bool,
) -> ExactConstraintProgram:
    frame, arena, empty_bin, upper = prepared
    result = arena._append_guarded_band(
        arena.empty_view,
        frame=frame,
        forward_cont=forward,
        forward_bin=empty_bin,
        forward_upper=upper,
        reverse_cont=reverse,
        reverse_bin=empty_bin,
        reverse_upper=upper.copy(),
        layer_id=7,
        family=_FAMILY_ADD_MATERIALIZE,
        allow_range=allow_range,
    )
    return arena.seal(result.view, final_frame=frame)


def benchmark_bounded_c89_ratio(
    *,
    scale_divisor: int = 80,
    warmups: int = 2,
    repeats: int = 9,
) -> Mapping[str, Any]:
    """Measure bounded dual-LE versus direct-RANGE source emission.

    This benchmark constructs no ACT network and invokes no solver.  Its only
    stage is owner/arena creation, immutable source publication, interning, and
    sealing.  Hard caps prohibit the full C89 shape or a real/large model.
    A failed timing gate is reported as ``closed``.  This receipt deliberately
    does not measure peak RSS and is not a full promotion gate; its baseline is
    the same candidate pipeline forced to dual LE, not production Operator-HZ.
    """

    scale_divisor = _exact_nonnegative_int(
        scale_divisor, name="scale_divisor"
    )
    warmups = _exact_nonnegative_int(warmups, name="warmups")
    repeats = _exact_nonnegative_int(repeats, name="repeats")
    if scale_divisor < 40 or repeats < 5 or warmups > 9 or repeats > 21:
        raise ConstraintBlockDAGCandidateError(
            "bounded benchmark requires divisor>=40, 0<=warmups<=9, 5<=repeats<=21"
        )
    pair_count = (40_960 + scale_divisor - 1) // scale_divisor
    columns = max((52_359 + scale_divisor - 1) // scale_divisor, 74)
    if pair_count > 1_024 or columns > 1_310:
        raise ConstraintBlockDAGCandidateError(
            "bounded benchmark geometry exceeds disconnected caps"
        )
    forward = _c89_ratio_matrix(pair_count=pair_count, columns=columns)
    reverse = forward.copy()
    reverse.data = np.negative(reverse.data)
    reverse.sort_indices()
    nonpair = reverse.copy()
    if nonpair.nnz:
        # One non-opposite coefficient per row makes every pair use the exact
        # dual-LE fallback while retaining identical geometry and payload size.
        positions = nonpair.indptr[:-1]
        nonpair.data[positions] = np.nextafter(
            nonpair.data[positions], np.inf
        )

    baseline_times = []
    range_times = []
    fallback_baseline_times = []
    fallback_candidate_times = []
    baseline_program = None
    range_program = None
    for iteration in range(warmups + repeats):
        # Alternate order to reduce monotonic thermal/order bias.
        operations = (
            (
                ("range", reverse, True),
                ("baseline", reverse, False),
                ("fallback", nonpair, True),
                ("fallback_baseline", nonpair, False),
            )
            if iteration % 2 == 0
            else (
                ("fallback_baseline", nonpair, False),
                ("fallback", nonpair, True),
                ("baseline", reverse, False),
                ("range", reverse, True),
            )
        )
        # The factor owner/frame already exists when Operator-HZ emits an ADD
        # relation.  Prepare those common objects outside the measured region;
        # the isolated gate measures only source publication, interning, and
        # sealing for the same live frame.
        prepared = {
            name: _prepare_benchmark_emitter(forward)
            for name, _right, _allow_range in operations
        }
        measured: Dict[str, int] = {}
        products: Dict[str, ExactConstraintProgram] = {}
        for name, right, allow_range in operations:
            started = time.perf_counter_ns()
            product = _emit_benchmark_program(
                prepared=prepared[name],
                forward=forward,
                reverse=right,
                allow_range=allow_range,
            )
            measured[name] = max(1, time.perf_counter_ns() - started)
            products[name] = product
        if iteration >= warmups:
            range_times.append(measured["range"])
            baseline_times.append(measured["baseline"])
            fallback_candidate_times.append(measured["fallback"])
            fallback_baseline_times.append(measured["fallback_baseline"])
        baseline_program = products["baseline"]
        range_program = products["range"]
        gc.collect()
    assert baseline_program is not None and range_program is not None
    baseline_median = float(statistics.median(baseline_times))
    range_median = float(statistics.median(range_times))
    fallback_median = float(statistics.median(fallback_candidate_times))
    fallback_baseline_median = float(
        statistics.median(fallback_baseline_times)
    )
    speedup = baseline_median / max(1.0, range_median)
    fallback_slowdown = fallback_median / max(1.0, fallback_baseline_median)
    payload_ratio = range_program.numeric_payload_bytes / max(
        1, baseline_program.numeric_payload_bytes
    )
    speed_ok = speedup >= 1.50
    payload_ok = payload_ratio <= 0.60
    fallback_ok = fallback_slowdown <= 1.10
    status = "passed" if speed_ok and payload_ok and fallback_ok else "closed"
    reasons = []
    if not speed_ok:
        reasons.append("range_emission_speedup_below_1.50")
    if not payload_ok:
        reasons.append("range_payload_ratio_above_0.60")
    if not fallback_ok:
        reasons.append("fallback_slowdown_above_1.10")
    return MappingProxyType(
        {
            "schema": _PERF_SCHEMA,
            "status": status,
            "closed_reasons": tuple(reasons),
            "candidate_only": True,
            "proof_authority": False,
            "verdict_authority": False,
            "production_integration": False,
            "gate_scope": "candidate_isolated_timing_and_retained_payload_only",
            "baseline_kind": "candidate_forced_dual_le_same_pipeline",
            "rss_measured": False,
            "full_promotion_gate": False,
            "promotion_authority": False,
            "production_baseline": False,
            "real_model_allowed": False,
            "large_model_allowed": False,
            "measured_stage": "source_publish+intern+seal_on_prepared_owner_frame",
            "scale_divisor": int(scale_divisor),
            "pair_count": int(pair_count),
            "columns": int(columns),
            "forward_nnz": int(forward.nnz),
            "baseline_median_ns": baseline_median,
            "range_median_ns": range_median,
            "range_speedup": float(speedup),
            "fallback_baseline_median_ns": fallback_baseline_median,
            "fallback_candidate_median_ns": fallback_median,
            "fallback_slowdown": float(fallback_slowdown),
            "baseline_payload_bytes": int(
                baseline_program.numeric_payload_bytes
            ),
            "range_payload_bytes": int(range_program.numeric_payload_bytes),
            "payload_ratio": float(payload_ratio),
            "speed_gate": 1.50,
            "payload_ratio_gate": 0.60,
            "fallback_slowdown_gate": 1.10,
            "range_source_rows": int(range_program.source_rows),
            "baseline_source_rows": int(baseline_program.source_rows),
            "range_source_nnz": int(range_program.source_nnz),
            "baseline_source_nnz": int(baseline_program.source_nnz),
            "virtual_facet_rows": int(range_program.virtual_facet_rows),
            "triangle_relaxation_called": False,
            "branch_and_bound_called": False,
            "backward_called": False,
            "dual_called": False,
        }
    )


__all__ = [
    "ConstraintArenaMismatch",
    "ConstraintBlockDAGCandidateError",
    "ConstraintBlockHandle",
    "ConstraintRowHandle",
    "ConstraintView",
    "ExactConstraintArena",
    "ExactConstraintOwner",
    "ExactConstraintProgram",
    "ExactFactorFrame",
    "GuardedBandAppend",
    "NativeConstraintBlock",
    "StableFactorID",
    "StableObjectID",
    "VirtualFacetBatch",
    "VirtualFacetFrame",
    "benchmark_bounded_c89_ratio",
    "iter_virtual_facet_batches",
    "native_blocks",
    "replay_virtual_facets",
]
