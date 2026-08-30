#!/usr/bin/env python3
# ===- constraint_program.py - exact constraint source program --------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Immutable exact constraint representation and replay core.

This module owns only the representation and replay of linear constraints.  It
does not prove a property, choose a solver status, relax ReLU, branch, run a
backward pass, or tighten a dual.  In particular, a :class:`ConstraintProgram`
is not a verification certificate.

Factor identities remain owned by the caller's existing allocator.  The
``ExternalFactorAllocatorAdapter`` captures that allocator and its callbacks at
bind time; this module never creates a second factor-ID namespace.  Its own
process-wide counter is restricted to occurrence identities (blocks, rows,
facets, views, and prepared transactions), whose intervals are deliberately
burned on failure.

Mutable owners, arenas, and prepared transactions are bound to their creating
``threading.Thread`` object.  Sealed programs and their replay iterators contain
only immutable builtin ``bytes`` authority and are safe for concurrent readers.
Digests select interning buckets only.  Equality and authenticity always use
the complete canonical bytes and captured object graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import sys
import threading
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp


_SCHEMA = "act.constraint_program.v1"
_INT64_MAX = int(np.iinfo(np.int64).max)
_FACTORY_TOKEN = object()
_PROGRAM_FACTORY_TOKEN = object()


class ConstraintProgramError(ValueError):
    """Fail-closed contract error for the production representation core."""


class ConstraintArenaMismatch(ConstraintProgramError):
    """A frame, view, or prepared append belongs to another owner/arena."""


class ConstraintTransactionError(ConstraintProgramError):
    """A prepared append is stale, consumed, reordered, or forged."""


class ExternalAllocatorContractError(ConstraintProgramError):
    """The captured external factor allocator violated its typed contract."""


class FactorKind(Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"


class ConstraintFamily(Enum):
    ADD_MATERIALIZE = "add_materialize"


class _BlockKind(Enum):
    LE = "le"
    GUARDED_BAND = "guarded_band"


class _ReservationPhase(Enum):
    RESERVED = "reserved"
    INITIALIZING = "initializing"
    PUBLISHING = "publishing"
    POISONED = "poisoned"


# A native lock is an identity-only, weak-referenceable token whose exact
# runtime type cannot be changed with ``object.__setattr__(..., "__class__")``.
# No code ever acquires it; its native immutability closes the class-rebinding
# hole of a normal Python sentinel object.
_NamespaceIdentity = type(threading.Lock())


class _OwnerToken:
    pass


class _ArenaToken:
    pass


def _exact_count(value: Any, *, name: str, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ConstraintProgramError(f"{name} must be a {qualifier} builtin int")
    if value > _INT64_MAX:
        raise ConstraintProgramError(f"{name} exceeds signed int64")
    return value


def _exact_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ConstraintProgramError(f"{name} must be a nonempty builtin str")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ConstraintProgramError(
            f"{name} must be encodable as strict UTF-8"
        ) from error
    return value


class ExternalFactorID:
    """Typed factor identity issued by a captured external allocator.

    Public construction is intentionally rejected.  Frames and replay batches
    return detached instances created from registry-captured truth.
    """

    __slots__ = ("_kind", "_raw_id", "_namespace_identity", "_sealed")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ExternalFactorID":
        if cls is not ExternalFactorID or _kwargs.pop("_token", None) is not _FACTORY_TOKEN:
            raise TypeError("ExternalFactorID cannot be self-signed")
        if _args or _kwargs:
            raise TypeError("malformed internal ExternalFactorID construction")
        return object.__new__(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ExternalFactorID is immutable")
        object.__setattr__(self, name, value)

    @property
    def kind(self) -> FactorKind:
        return self._kind

    @property
    def raw_id(self) -> int:
        return self._raw_id

    @property
    def namespace_identity(self) -> Any:
        return self._namespace_identity

    def __hash__(self) -> int:
        return hash((self._kind, self._raw_id, id(self._namespace_identity)))

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is ExternalFactorID
            and self._kind is other._kind
            and self._raw_id == other._raw_id
            and self._namespace_identity is other._namespace_identity
        )

    def __lt__(self, other: Any) -> bool:
        if type(other) is not ExternalFactorID:
            return NotImplemented
        return (self._kind.value, self._raw_id, id(self._namespace_identity)) < (
            other._kind.value,
            other._raw_id,
            id(other._namespace_identity),
        )

    def __repr__(self) -> str:
        return f"ExternalFactorID({self._kind.value!r}, {self._raw_id})"


def _new_factor_id(
    kind: FactorKind, raw_id: int, namespace: _NamespaceIdentity
) -> ExternalFactorID:
    result = ExternalFactorID(_token=_FACTORY_TOKEN)
    object.__setattr__(result, "_kind", kind)
    object.__setattr__(result, "_raw_id", raw_id)
    object.__setattr__(result, "_namespace_identity", namespace)
    object.__setattr__(result, "_sealed", True)
    return result


def _factor_key(value: Any) -> Tuple[str, int, _NamespaceIdentity]:
    if (
        type(value) is not ExternalFactorID
        or type(value._kind) is not FactorKind
        or type(value._raw_id) is not int
        or value._raw_id < 0
        or value._raw_id > _INT64_MAX
        or type(value._namespace_identity) is not _NamespaceIdentity
        or value._sealed is not True
    ):
        raise ConstraintProgramError("factor identity was forged or mutated")
    return value._kind.value, value._raw_id, value._namespace_identity


def _validate_external_keys(
    value: Any,
    *,
    kind: FactorKind,
    name: str,
) -> Tuple[Tuple[str, int, _NamespaceIdentity], ...]:
    if type(value) is not tuple:
        raise ConstraintProgramError(f"{name} must be an exact builtin tuple")
    result = []
    raw_ids = set()
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 3
            or item[0] != kind.value
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] < 0
            or item[1] > _INT64_MAX
            or type(item[2]) is not _NamespaceIdentity
            or item[1] in raw_ids
        ):
            raise ConstraintProgramError(f"{name} contains a forged factor key")
        raw_ids.add(item[1])
        result.append(item)
    return tuple(result)


class ConstraintObjectID:
    """Globally non-reused occurrence identity (never a factor identity)."""

    __slots__ = ("_kind", "_value", "_sealed")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ConstraintObjectID":
        if cls is not ConstraintObjectID or _kwargs.pop("_token", None) is not _FACTORY_TOKEN:
            raise TypeError("ConstraintObjectID cannot be self-signed")
        if _args or _kwargs:
            raise TypeError("malformed internal ConstraintObjectID construction")
        return object.__new__(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ConstraintObjectID is immutable")
        object.__setattr__(self, name, value)

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def value(self) -> int:
        return self._value

    def __hash__(self) -> int:
        return hash((self._kind, self._value))

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is ConstraintObjectID
            and self._kind == other._kind
            and self._value == other._value
        )

    def __lt__(self, other: Any) -> bool:
        if type(other) is not ConstraintObjectID:
            return NotImplemented
        return (self._value, self._kind) < (other._value, other._kind)

    def __repr__(self) -> str:
        return f"ConstraintObjectID({self._kind!r}, {self._value})"


_OCCURRENCE_LOCK = threading.Lock()
_NEXT_OCCURRENCE_ID = 0


def _reserve_occurrence_ids(kind: str, count: int) -> Tuple[ConstraintObjectID, ...]:
    global _NEXT_OCCURRENCE_ID
    _exact_text(kind, name="occurrence kind")
    count = _exact_count(count, name="occurrence count")
    with _OCCURRENCE_LOCK:
        start = _NEXT_OCCURRENCE_ID
        if count and count - 1 > _INT64_MAX - start:
            raise ConstraintProgramError("occurrence ID reservation exceeds signed int64")
        _NEXT_OCCURRENCE_ID = start + count
    result = []
    for raw in range(start, start + count):
        item = ConstraintObjectID(_token=_FACTORY_TOKEN)
        object.__setattr__(item, "_kind", kind)
        object.__setattr__(item, "_value", raw)
        object.__setattr__(item, "_sealed", True)
        result.append(item)
    return tuple(result)


def _object_key(value: Any, *, kinds: Tuple[str, ...]) -> Tuple[str, int]:
    if (
        type(value) is not ConstraintObjectID
        or type(value._kind) is not str
        or value._kind not in kinds
        or type(value._value) is not int
        or value._value < 0
        or value._value > _INT64_MAX
        or value._sealed is not True
    ):
        raise ConstraintProgramError("occurrence identity was forged or mutated")
    return value._kind, value._value


@dataclass(frozen=True)
class _AdapterRecord:
    reference: Any
    allocator_obj: Any
    allocate_continuous: Callable[[int], Tuple[int, ...]]
    allocate_binary: Callable[[int], Tuple[int, ...]]
    live_ids_snapshot: Callable[[], Tuple[Tuple[int, ...], Tuple[int, ...]]]
    namespace: _NamespaceIdentity
    namespace_reference: Any
    commit_token: _NamespaceIdentity
    thread: threading.Thread


@dataclass(frozen=True)
class _AdapterRegistryEntry:
    reference: Any
    record: _AdapterRecord
    graph: Tuple[Any, ...]


@dataclass(frozen=True)
class _ReservationState:
    phase: _ReservationPhase
    epoch: int = 0
    reentrancy_detected: bool = False
    external_touched: bool = False
    commit_token: Any = None


@dataclass(frozen=True)
class _ReservationRecord:
    reference: Any
    thread: threading.Thread
    state: _ReservationState
    state_key: Tuple[Any, ...]
    graph: Tuple[Any, ...]


def _reservation_state_key(state: _ReservationState) -> Tuple[Any, ...]:
    if (
        type(state) is not _ReservationState
        or type(state.phase) is not _ReservationPhase
        or type(state.epoch) is not int
        or state.epoch < 0
        or type(state.reentrancy_detected) is not bool
        or type(state.external_touched) is not bool
        or (
            state.phase is not _ReservationPhase.INITIALIZING
            and (state.reentrancy_detected or state.external_touched)
        )
        or (
            state.phase is _ReservationPhase.PUBLISHING
            and type(state.commit_token) is not _NamespaceIdentity
        )
        or (
            state.phase is not _ReservationPhase.PUBLISHING
            and state.commit_token is not None
        )
    ):
        raise ConstraintProgramError("construction reservation state is malformed")
    return (
        state.phase.value,
        state.epoch,
        state.reentrancy_detected,
        state.external_touched,
        state.commit_token,
    )


def _make_reservation_record(
    reference: Any,
    thread: threading.Thread,
    state: _ReservationState,
) -> _ReservationRecord:
    if (
        not isinstance(reference, weakref.ReferenceType)
        or not isinstance(thread, threading.Thread)
    ):
        raise ConstraintProgramError("construction reservation anchor is malformed")
    return _ReservationRecord(
        reference,
        thread,
        state,
        _reservation_state_key(state),
        (reference, thread, state),
    )


def _validate_reservation_record(
    record: Any,
    value: Any,
    *,
    name: str,
) -> _ReservationRecord:
    if type(record) is not _ReservationRecord:
        raise ConstraintProgramError(f"{name} reservation record was substituted")
    current_key = _reservation_state_key(record.state)
    current_graph = (record.reference, record.thread, record.state)
    if (
        record.reference() is not value
        or current_key != record.state_key
        or len(record.graph) != len(current_graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_graph, record.graph)
        )
    ):
        raise ConstraintProgramError(f"{name} reservation was forged or rebound")
    return record


def _validate_adapter_reservation(
    record: Any,
    adapter: Any,
) -> _ReservationRecord:
    try:
        return _validate_reservation_record(record, adapter, name="adapter")
    except ExternalAllocatorContractError:
        raise
    except ConstraintProgramError as error:
        raise ExternalAllocatorContractError(str(error)) from error


def _adapter_graph(record: _AdapterRecord) -> Tuple[Any, ...]:
    if (
        type(record) is not _AdapterRecord
        or type(record.namespace) is not _NamespaceIdentity
        or not isinstance(record.namespace_reference, weakref.ReferenceType)
        or record.namespace_reference() is not record.namespace
        or type(record.commit_token) is not _NamespaceIdentity
        or not isinstance(record.thread, threading.Thread)
        or not callable(record.allocate_continuous)
        or not callable(record.allocate_binary)
        or not callable(record.live_ids_snapshot)
    ):
        raise ExternalAllocatorContractError(
            "captured allocator record was substituted"
        )
    return (
        record,
        record.reference,
        record.allocator_obj,
        record.allocate_continuous,
        record.allocate_binary,
        record.live_ids_snapshot,
        record.namespace,
        record.namespace_reference,
        record.commit_token,
        record.thread,
    )


_ADAPTER_LOCK = threading.Lock()
_ADAPTER_REGISTRY: Dict[int, _AdapterRegistryEntry] = {}
_ADAPTER_RESERVATIONS: Dict[int, _ReservationRecord] = {}
_ALLOCATOR_BINDINGS: Dict[int, Tuple[Any, Any, Any]] = {}


class ExternalFactorAllocatorAdapter:
    """One captured adapter for an existing typed factor allocator."""

    __slots__ = ("_sealed", "__weakref__")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ExternalFactorAllocatorAdapter":
        if cls is not ExternalFactorAllocatorAdapter or _kwargs.pop("_token", None) is not _FACTORY_TOKEN:
            raise TypeError("allocator adapters must be created by reserve() or bind()")
        if _args or _kwargs:
            raise TypeError("malformed internal adapter construction")
        return object.__new__(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("allocator adapter is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def reserve(cls) -> "ExternalFactorAllocatorAdapter":
        """Return one exact, uninitialized, non-authoritative adapter handle."""

        if cls is not ExternalFactorAllocatorAdapter:
            raise TypeError("mutable/subclassed allocator adapters are rejected")
        return _reserve_adapter_handle()

    def initialize(
        self,
        allocator_obj: Any,
        *,
        allocate_continuous: Callable[[int], Tuple[int, ...]],
        allocate_binary: Callable[[int], Tuple[int, ...]],
        live_ids_snapshot: Callable[[], Tuple[Tuple[int, ...], Tuple[int, ...]]],
    ) -> None:
        _initialize_reserved_adapter(
            self,
            allocator_obj,
            allocate_continuous=allocate_continuous,
            allocate_binary=allocate_binary,
            live_ids_snapshot=live_ids_snapshot,
        )

    @classmethod
    def bind(
        cls,
        allocator_obj: Any,
        *,
        allocate_continuous: Callable[[int], Tuple[int, ...]],
        allocate_binary: Callable[[int], Tuple[int, ...]],
        live_ids_snapshot: Callable[[], Tuple[Tuple[int, ...], Tuple[int, ...]]],
    ) -> "ExternalFactorAllocatorAdapter":
        if cls is not ExternalFactorAllocatorAdapter:
            raise TypeError("mutable/subclassed allocator adapters are rejected")
        adapter = cls.reserve()
        adapter.initialize(
            allocator_obj,
            allocate_continuous=allocate_continuous,
            allocate_binary=allocate_binary,
            live_ids_snapshot=live_ids_snapshot,
        )
        return adapter

    @property
    def namespace_identity(self) -> Any:
        return _adapter_record(self).namespace


def _reserve_adapter_handle() -> ExternalFactorAllocatorAdapter:
    adapter = ExternalFactorAllocatorAdapter(_token=_FACTORY_TOKEN)
    object.__setattr__(adapter, "_sealed", True)
    object_id = id(adapter)

    def cleanup(reference: Any) -> None:
        for _attempt in range(4):
            try:
                with _ADAPTER_LOCK:
                    reserved = _ADAPTER_RESERVATIONS.get(object_id)
                    if reserved is not None and reserved.reference is reference:
                        _ADAPTER_RESERVATIONS.pop(object_id, None)
                    current = _ADAPTER_REGISTRY.get(object_id)
                    if current is not None and current.reference is reference:
                        _ADAPTER_REGISTRY.pop(object_id, None)
                return
            except BaseException:
                continue

    reference = weakref.ref(adapter, cleanup)
    reservation = _make_reservation_record(
        reference,
        threading.current_thread(),
        _ReservationState(_ReservationPhase.RESERVED),
    )
    try:
        with _ADAPTER_LOCK:
            if (
                id(adapter) in _ADAPTER_RESERVATIONS
                or id(adapter) in _ADAPTER_REGISTRY
            ):
                raise ExternalAllocatorContractError(
                    "adapter reservation object ID was reused"
                )
            _ADAPTER_RESERVATIONS[id(adapter)] = reservation
    except BaseException:
        for _attempt in range(4):
            try:
                with _ADAPTER_LOCK:
                    current = _ADAPTER_RESERVATIONS.get(id(adapter))
                    if current is not None and current.reference is reference:
                        _ADAPTER_RESERVATIONS.pop(id(adapter), None)
                break
            except BaseException:
                continue
        raise
    return adapter


def _adapter_initialization_epoch(
    adapter: ExternalFactorAllocatorAdapter,
) -> int:
    if type(adapter) is not ExternalFactorAllocatorAdapter:
        raise ExternalAllocatorContractError("allocator adapter has the wrong exact type")
    with _ADAPTER_LOCK:
        initialized = _ADAPTER_REGISTRY.get(id(adapter))
        reservation = _ADAPTER_RESERVATIONS.get(id(adapter))
        if initialized is not None and initialized.reference() is adapter:
            raise ExternalAllocatorContractError("allocator adapter is already initialized")
        if reservation is None:
            raise ExternalAllocatorContractError(
                "allocator adapter is forged, stale, or not reserved"
            )
        reservation = _validate_adapter_reservation(reservation, adapter)
        if getattr(adapter, "_sealed", None) is not True:
            raise ExternalAllocatorContractError(
                "allocator adapter reservation was rebound"
            )
        if threading.current_thread() is not reservation.thread:
            raise ExternalAllocatorContractError(
                "allocator adapter initialization is Thread-object confined"
            )
        state = reservation.state
        if state.phase is _ReservationPhase.INITIALIZING:
            replacement = _ReservationState(
                state.phase,
                state.epoch,
                True,
                state.external_touched,
            )
            _ADAPTER_RESERVATIONS[id(adapter)] = _make_reservation_record(
                reservation.reference,
                reservation.thread,
                replacement,
            )
            raise ExternalAllocatorContractError(
                "allocator adapter initialization cannot reenter"
            )
        if state.phase is _ReservationPhase.POISONED:
            raise ExternalAllocatorContractError(
                "allocator adapter reservation is poisoned"
            )
        if state.phase is not _ReservationPhase.RESERVED:
            raise ExternalAllocatorContractError(
                "allocator adapter reservation state is invalid"
            )
        return state.epoch + 1


def _activate_adapter_initialization(
    adapter: ExternalFactorAllocatorAdapter, epoch: int
) -> None:
    with _ADAPTER_LOCK:
        reservation = _ADAPTER_RESERVATIONS.get(id(adapter))
        if reservation is None:
            raise ExternalAllocatorContractError(
                "adapter reservation vanished before initialization"
            )
        reservation = _validate_adapter_reservation(reservation, adapter)
        state = reservation.state
        if (
            threading.current_thread() is not reservation.thread
            or state.phase is not _ReservationPhase.RESERVED
            or state.epoch + 1 != epoch
        ):
            raise ExternalAllocatorContractError(
                "adapter reservation changed before initialization"
            )
        _ADAPTER_RESERVATIONS[id(adapter)] = _make_reservation_record(
            reservation.reference,
            reservation.thread,
            _ReservationState(_ReservationPhase.INITIALIZING, epoch),
        )


def _poison_adapter_initialization(
    adapter: ExternalFactorAllocatorAdapter,
    epoch: int,
    *,
    staged_entry: Optional[_AdapterRegistryEntry] = None,
    allocator_obj: Any = None,
) -> None:
    allocator_id = None if staged_entry is None else id(allocator_obj)
    for _attempt in range(4):
        try:
            with _ADAPTER_LOCK:
                current_entry = _ADAPTER_REGISTRY.get(id(adapter))
                current_binding = (
                    None
                    if allocator_id is None
                    else _ALLOCATOR_BINDINGS.get(allocator_id)
                )
                entry_matches = bool(
                    staged_entry is not None and current_entry is staged_entry
                )
                binding_matches = bool(
                    staged_entry is not None
                    and _adapter_binding_matches(
                        current_binding,
                        allocator_obj,
                        staged_entry.record.namespace_reference,
                        staged_entry.record.commit_token,
                    )
                )
                # Once both authority records are exact, publication is a
                # complete NEW.  Cleanup must never revoke it merely because
                # an exception interrupted the final reservation removal or
                # a repair helper's return event.
                if entry_matches and binding_matches:
                    _ADAPTER_RESERVATIONS.pop(id(adapter), None)
                    return
                reservation = _ADAPTER_RESERVATIONS.get(id(adapter))
                if reservation is not None:
                    reservation = _validate_adapter_reservation(
                        reservation, adapter
                    )
                if staged_entry is not None and reservation is not None:
                    state = reservation.state
                    must_complete = bool(
                        state.phase in (
                            _ReservationPhase.INITIALIZING,
                            _ReservationPhase.PUBLISHING,
                        )
                        and state.epoch == epoch
                        and (
                            state.phase is _ReservationPhase.INITIALIZING
                            or state.commit_token
                            is staged_entry.record.commit_token
                        )
                        and (current_entry is None or entry_matches)
                        and binding_matches
                    )
                    if must_complete and allocator_id is not None:
                        _ADAPTER_REGISTRY[id(adapter)] = staged_entry
                        _ALLOCATOR_BINDINGS[allocator_id] = (
                            allocator_obj,
                            staged_entry.record.namespace_reference,
                            staged_entry.record.commit_token,
                        )
                        _ADAPTER_RESERVATIONS.pop(id(adapter), None)
                        return
                if entry_matches:
                    _ADAPTER_REGISTRY.pop(id(adapter), None)
                if binding_matches and allocator_id is not None:
                    _ALLOCATOR_BINDINGS.pop(allocator_id, None)
                if reservation is not None:
                    reference = reservation.reference
                    thread = reservation.thread
                    poison_epoch = max(epoch, reservation.state.epoch)
                elif staged_entry is not None:
                    reference = staged_entry.reference
                    thread = staged_entry.record.thread
                    poison_epoch = epoch
                else:
                    return
                if reference() is adapter:
                    _ADAPTER_RESERVATIONS[id(adapter)] = _make_reservation_record(
                        reference,
                        thread,
                        _ReservationState(
                            _ReservationPhase.POISONED,
                            poison_epoch,
                        ),
                    )
            return
        except BaseException:
            continue


def _adapter_binding_matches(
    binding: Any,
    allocator_obj: Any,
    namespace_reference: Any,
    commit_token: Any,
) -> bool:
    return bool(
        type(binding) is tuple
        and len(binding) == 3
        and binding[0] is allocator_obj
        and binding[1] is namespace_reference
        and binding[2] is commit_token
    )


def _publish_adapter_initialization(
    adapter: ExternalFactorAllocatorAdapter,
    epoch: int,
    entry: _AdapterRegistryEntry,
    allocator_obj: Any,
) -> None:
    allocator_id = id(allocator_obj)
    record = entry.record
    with _ADAPTER_LOCK:
        reservation = _ADAPTER_RESERVATIONS.get(id(adapter))
        if reservation is None:
            raise ExternalAllocatorContractError(
                "adapter reservation vanished before publication"
            )
        reservation = _validate_adapter_reservation(reservation, adapter)
        state = reservation.state
        binding = _ALLOCATOR_BINDINGS.get(allocator_id)
        if (
            state.phase is not _ReservationPhase.INITIALIZING
            or state.epoch != epoch
            or state.reentrancy_detected
            or state.external_touched
            or _ADAPTER_REGISTRY.get(id(adapter)) is not None
            or binding is not None
        ):
            raise ExternalAllocatorContractError(
                "adapter initialization changed before publication"
            )
        # Installing the token-authenticated binding is the durable commit
        # intent and also reserves the allocator namespace against a competing
        # initializer.  Readers still reject it until the exact entry is
        # installed and the reservation guard is removed.  Before this write
        # a failure poisons with no lease; after it bounded repair may only
        # complete this already-staged pair and never invokes user code.
        _ALLOCATOR_BINDINGS[allocator_id] = (
            allocator_obj,
            record.namespace_reference,
            record.commit_token,
        )
        _ADAPTER_RESERVATIONS[id(adapter)] = _make_reservation_record(
            reservation.reference,
            reservation.thread,
            _ReservationState(
                _ReservationPhase.PUBLISHING,
                epoch,
                commit_token=record.commit_token,
            ),
        )
        _ADAPTER_REGISTRY[id(adapter)] = entry
        _ADAPTER_RESERVATIONS.pop(id(adapter), None)


def _repair_adapter_initialization(
    adapter: ExternalFactorAllocatorAdapter,
    epoch: int,
    entry: _AdapterRegistryEntry,
    allocator_obj: Any,
) -> bool:
    allocator_id = id(allocator_obj)
    record = entry.record
    for _attempt in range(4):
        try:
            with _ADAPTER_LOCK:
                current_entry = _ADAPTER_REGISTRY.get(id(adapter))
                current_binding = _ALLOCATOR_BINDINGS.get(allocator_id)
                reservation = _ADAPTER_RESERVATIONS.get(id(adapter))
                entry_matches = current_entry is entry
                binding_matches = bool(
                    _adapter_binding_matches(
                        current_binding,
                        allocator_obj,
                        record.namespace_reference,
                        record.commit_token,
                    )
                )
                if entry_matches and binding_matches:
                    _ADAPTER_RESERVATIONS.pop(id(adapter), None)
                    return True
                can_complete = False
                if reservation is not None:
                    reservation = _validate_adapter_reservation(
                        reservation, adapter
                    )
                    state = reservation.state
                    can_complete = bool(
                        state.phase in (
                            _ReservationPhase.INITIALIZING,
                            _ReservationPhase.PUBLISHING,
                        )
                        and state.epoch == epoch
                        and (
                            state.phase is _ReservationPhase.INITIALIZING
                            or state.commit_token is record.commit_token
                        )
                        and (current_entry is None or entry_matches)
                        and binding_matches
                    )
                if can_complete:
                    _ADAPTER_REGISTRY[id(adapter)] = entry
                    _ALLOCATOR_BINDINGS[allocator_id] = (
                        allocator_obj,
                        record.namespace_reference,
                        record.commit_token,
                    )
                    _ADAPTER_RESERVATIONS.pop(id(adapter), None)
                    return True
                if entry_matches:
                    _ADAPTER_REGISTRY.pop(id(adapter), None)
                if binding_matches:
                    _ALLOCATOR_BINDINGS.pop(allocator_id, None)
                if reservation is not None:
                    _ADAPTER_RESERVATIONS[id(adapter)] = _make_reservation_record(
                        reservation.reference,
                        reservation.thread,
                        _ReservationState(_ReservationPhase.POISONED, epoch),
                    )
                return False
        except BaseException:
            continue
    _poison_adapter_initialization(
        adapter,
        epoch,
        staged_entry=entry,
        allocator_obj=allocator_obj,
    )
    return False


def _adapter_initialization_is_complete(
    adapter: ExternalFactorAllocatorAdapter,
    entry: _AdapterRegistryEntry,
) -> bool:
    record = entry.record
    with _ADAPTER_LOCK:
        return bool(
            _ADAPTER_RESERVATIONS.get(id(adapter)) is None
            and _ADAPTER_REGISTRY.get(id(adapter)) is entry
            and _adapter_binding_matches(
                _ALLOCATOR_BINDINGS.get(id(record.allocator_obj)),
                record.allocator_obj,
                record.namespace_reference,
                record.commit_token,
            )
        )


def _initialize_reserved_adapter(
    adapter: ExternalFactorAllocatorAdapter,
    allocator_obj: Any,
    *,
    allocate_continuous: Callable[[int], Tuple[int, ...]],
    allocate_binary: Callable[[int], Tuple[int, ...]],
    live_ids_snapshot: Callable[[], Tuple[Tuple[int, ...], Tuple[int, ...]]],
) -> None:
    epoch = _adapter_initialization_epoch(adapter)
    staged_entry: Optional[_AdapterRegistryEntry] = None
    try:
        _activate_adapter_initialization(adapter, epoch)
        if allocator_obj is None or not callable(allocate_continuous) or not callable(
            allocate_binary
        ) or not callable(live_ids_snapshot):
            raise ExternalAllocatorContractError(
                "allocator object and all three captured callbacks are required"
            )
        with _ADAPTER_LOCK:
            reservation = _validate_adapter_reservation(
                _ADAPTER_RESERVATIONS.get(id(adapter)), adapter
            )
        reference = reservation.reference
        namespace = _NamespaceIdentity()
        allocator_id = id(allocator_obj)
        commit_token = _NamespaceIdentity()

        def cleanup_namespace(current_reference: Any) -> None:
            for _attempt in range(4):
                try:
                    with _ADAPTER_LOCK:
                        current = _ALLOCATOR_BINDINGS.get(allocator_id)
                        if _adapter_binding_matches(
                            current,
                            allocator_obj,
                            current_reference,
                            commit_token,
                        ):
                            _ALLOCATOR_BINDINGS.pop(allocator_id, None)
                    return
                except BaseException:
                    continue

        namespace_reference = weakref.ref(namespace, cleanup_namespace)
        record = _AdapterRecord(
            reference,
            allocator_obj,
            allocate_continuous,
            allocate_binary,
            live_ids_snapshot,
            namespace,
            namespace_reference,
            commit_token,
            reservation.thread,
        )
        staged_entry = _AdapterRegistryEntry(
            reference,
            record,
            _adapter_graph(record),
        )
        _publish_adapter_initialization(
            adapter,
            epoch,
            staged_entry,
            allocator_obj,
        )
    except BaseException:
        completed = False
        if staged_entry is not None:
            try:
                completed = _repair_adapter_initialization(
                    adapter,
                    epoch,
                    staged_entry,
                    allocator_obj,
                )
            except BaseException:
                try:
                    completed = _adapter_initialization_is_complete(
                        adapter,
                        staged_entry,
                    )
                except BaseException:
                    completed = False
        if not completed:
            for _attempt in range(4):
                try:
                    _poison_adapter_initialization(
                        adapter,
                        epoch,
                        staged_entry=staged_entry,
                        allocator_obj=allocator_obj,
                    )
                    break
                except BaseException:
                    continue
        raise


def _adapter_record(value: Any) -> _AdapterRecord:
    if type(value) is not ExternalFactorAllocatorAdapter:
        raise ExternalAllocatorContractError("allocator adapter has the wrong exact type")
    with _ADAPTER_LOCK:
        entry = _ADAPTER_REGISTRY.get(id(value))
        reservation = _ADAPTER_RESERVATIONS.get(id(value))
    if type(entry) is not _AdapterRegistryEntry or reservation is not None:
        raise ExternalAllocatorContractError("allocator adapter is forged or stale")
    record = entry.record
    current_graph = _adapter_graph(record)
    with _ADAPTER_LOCK:
        binding = _ALLOCATOR_BINDINGS.get(id(record.allocator_obj))
    if (
        entry.reference() is not value
        or record.reference is not entry.reference
        or value._sealed is not True
        or type(record.namespace) is not _NamespaceIdentity
        or not _adapter_binding_matches(
            binding,
            record.allocator_obj,
            record.namespace_reference,
            record.commit_token,
        )
        or len(current_graph) != len(entry.graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_graph, entry.graph)
        )
    ):
        raise ExternalAllocatorContractError("allocator adapter is forged or stale")
    return record


def _validate_raw_ids(value: Any, *, name: str) -> Tuple[int, ...]:
    if type(value) is not tuple:
        raise ExternalAllocatorContractError(f"{name} must be an exact builtin tuple")
    result = []
    for item in value:
        if type(item) is not int or item < 0 or item > _INT64_MAX:
            raise ExternalAllocatorContractError(
                f"{name} contains a non-builtin or invalid signed-int64 ID"
            )
        result.append(item)
    if len(set(result)) != len(result):
        raise ExternalAllocatorContractError(f"{name} contains duplicate IDs")
    return tuple(result)


def _validate_snapshot(value: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if type(value) is not tuple or len(value) != 2:
        raise ExternalAllocatorContractError(
            "live_ids_snapshot must return exactly (cont_tuple, bin_tuple)"
        )
    continuous = _validate_raw_ids(value[0], name="continuous live IDs")
    binary = _validate_raw_ids(value[1], name="binary live IDs")
    if set(continuous).intersection(binary):
        raise ExternalAllocatorContractError(
            "continuous and binary IDs collide in the external namespace"
        )
    return continuous, binary


def _snapshot_is_append_only(
    old: Tuple[Tuple[int, ...], Tuple[int, ...]],
    new: Tuple[Tuple[int, ...], Tuple[int, ...]],
) -> bool:
    return (
        new[0][: len(old[0])] == old[0]
        and new[1][: len(old[1])] == old[1]
    )


class FactorFrame:
    """Owner-authenticated append-only snapshot of the external namespace."""

    __slots__ = (
        "_continuous_ids",
        "_binary_ids",
        "_owner_token",
        "_version",
        "_sealed",
        "__weakref__",
    )

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "FactorFrame":
        if cls is not FactorFrame or _kwargs.pop("_token", None) is not _FACTORY_TOKEN:
            raise TypeError("FactorFrame cannot be self-signed")
        if _args or _kwargs:
            raise TypeError("malformed internal FactorFrame construction")
        return object.__new__(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("FactorFrame is immutable")
        object.__setattr__(self, name, value)

    @property
    def continuous_ids(self) -> Tuple[ExternalFactorID, ...]:
        return self._continuous_ids

    @property
    def binary_ids(self) -> Tuple[ExternalFactorID, ...]:
        return self._binary_ids

    @property
    def version(self) -> int:
        return self._version


@dataclass(frozen=True)
class _OwnerData:
    snapshot: Tuple[Tuple[int, ...], Tuple[int, ...]]
    id_cache: Dict[Tuple[FactorKind, int], ExternalFactorID]
    claimed: set[Tuple[FactorKind, int]]
    frames: Dict[int, Tuple[Any, Tuple[Any, ...], Tuple[Any, ...]]]
    version: int = 0
    arena_created: bool = False
    arena_id: Optional[int] = None
    sealed: bool = False
    poisoned: bool = False
    discarded: bool = False


class _OwnerState:
    """Stable owner anchor with one atomically replaceable mutable-data root."""

    def __init__(
        self,
        reference: Any,
        adapter: ExternalFactorAllocatorAdapter,
        adapter_record: _AdapterRecord,
        thread: threading.Thread,
        token: _OwnerToken,
        snapshot: Tuple[Tuple[int, ...], Tuple[int, ...]],
        id_cache: Dict[Tuple[FactorKind, int], ExternalFactorID],
        claimed: set[Tuple[FactorKind, int]],
        frames: Dict[int, Tuple[Any, Tuple[Any, ...], Tuple[Any, ...]]],
    ) -> None:
        self.reference = reference
        self.adapter = adapter
        self.adapter_record = adapter_record
        self.thread = thread
        self.token = token
        self._data = _OwnerData(snapshot, id_cache, claimed, frames)

    def _get(self, name: str) -> Any:
        return getattr(self._data, name)

    def _set(self, name: str, value: Any) -> None:
        if getattr(self._data, name) is value:
            return
        self._data = replace(self._data, **{name: value})

    snapshot = property(lambda self: self._get("snapshot"), lambda self, value: self._set("snapshot", value))
    id_cache = property(lambda self: self._get("id_cache"), lambda self, value: self._set("id_cache", value))
    claimed = property(lambda self: self._get("claimed"), lambda self, value: self._set("claimed", value))
    frames = property(lambda self: self._get("frames"), lambda self, value: self._set("frames", value))
    version = property(lambda self: self._get("version"), lambda self, value: self._set("version", value))
    arena_created = property(lambda self: self._get("arena_created"), lambda self, value: self._set("arena_created", value))
    arena_id = property(lambda self: self._get("arena_id"), lambda self, value: self._set("arena_id", value))
    sealed = property(lambda self: self._get("sealed"), lambda self, value: self._set("sealed", value))
    poisoned = property(lambda self: self._get("poisoned"), lambda self, value: self._set("poisoned", value))
    discarded = property(lambda self: self._get("discarded"), lambda self, value: self._set("discarded", value))


def _owner_anchor_graph(state: _OwnerState) -> Tuple[Any, ...]:
    if (
        type(state) is not _OwnerState
        or type(state.token) is not _OwnerToken
        or not isinstance(state.thread, threading.Thread)
        or type(state.adapter) is not ExternalFactorAllocatorAdapter
        or type(state.adapter_record) is not _AdapterRecord
    ):
        raise ConstraintProgramError("owner registry anchor was substituted")
    return (
        state,
        state.reference,
        state.adapter,
        state.adapter_record,
        state.thread,
        state.token,
    )


@dataclass(frozen=True)
class _OwnerRegistryEntry:
    reference: Any
    state: _OwnerState
    anchor_graph: Tuple[Any, ...]
    mutable_key: Tuple[Any, ...]
    mutable_graph: Tuple[Any, ...]
    operation_active: bool = False
    operation_epoch: int = 0
    reentrancy_detected: bool = False
    rollback_data: Optional[_OwnerData] = None
    external_touched: bool = False


def _owner_mutable_state(
    state: _OwnerState,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    snapshot = _validate_snapshot(state.snapshot)
    if (
        type(state.id_cache) is not dict
        or type(state.claimed) is not set
        or type(state.frames) is not dict
        or type(state.version) is not int
        or state.version < 0
        or type(state.arena_created) is not bool
        or (
            state.arena_id is not None
            and (type(state.arena_id) is not int or state.arena_id < 0)
        )
        or state.arena_created is not (state.arena_id is not None)
        or type(state.sealed) is not bool
        or type(state.poisoned) is not bool
        or type(state.discarded) is not bool
        or (state.sealed and state.discarded)
        or type(state._data) is not _OwnerData
    ):
        raise ConstraintProgramError("owner mutable registry state is malformed")
    expected_keys = {
        *((FactorKind.CONTINUOUS, raw) for raw in snapshot[0]),
        *((FactorKind.BINARY, raw) for raw in snapshot[1]),
    }
    if set(state.id_cache) != expected_keys:
        raise ConstraintProgramError(
            "owner factor cache does not exactly match its live snapshot"
        )
    cache_items = []
    cache_graph = [state.id_cache]
    for key in sorted(expected_keys, key=lambda item: (item[0].value, item[1])):
        if (
            type(key) is not tuple
            or len(key) != 2
            or type(key[0]) is not FactorKind
            or type(key[1]) is not int
        ):
            raise ConstraintProgramError("owner factor cache key is malformed")
        factor = state.id_cache[key]
        factor_key = _factor_key(factor)
        if (
            factor_key[0] != key[0].value
            or factor_key[1] != key[1]
            or factor_key[2] is not state.adapter_record.namespace
        ):
            raise ConstraintProgramError(
                "owner factor cache escaped its captured allocator namespace"
            )
        cache_items.append((key[0].value, key[1]))
        cache_graph.append(factor)
    claimed_items = []
    for key in state.claimed:
        if (
            type(key) is not tuple
            or len(key) != 2
            or type(key[0]) is not FactorKind
            or type(key[1]) is not int
            or key not in expected_keys
        ):
            raise ConstraintProgramError("owner claimed-factor set is malformed")
        claimed_items.append((key[0].value, key[1]))
    claimed_items.sort()
    frame_items = []
    frame_graph = [state.frames]
    for object_id, record in sorted(state.frames.items()):
        if (
            type(object_id) is not int
            or type(record) is not tuple
            or len(record) != 3
            or type(record[0]) is not FactorFrame
            or id(record[0]) != object_id
            or type(record[1]) is not tuple
            or type(record[2]) is not tuple
            or _frame_key(record[0]) != record[1]
        ):
            raise ConstraintArenaMismatch("owner factor-frame registry is malformed")
        current_graph = (
            record[0]._continuous_ids,
            record[0]._binary_ids,
            record[0]._owner_token,
            *record[0]._continuous_ids,
            *record[0]._binary_ids,
        )
        if (
            len(current_graph) != len(record[2])
            or any(
                current is not sealed
                for current, sealed in zip(current_graph, record[2])
            )
        ):
            raise ConstraintArenaMismatch("owner factor-frame graph was rebound")
        frame_items.append((object_id, record[1]))
        frame_graph.extend((record, record[0], record[1], record[2], *record[2]))
    key = (
        snapshot,
        tuple(cache_items),
        tuple(claimed_items),
        tuple(frame_items),
        state.version,
        state.arena_created,
        state.arena_id,
        state.sealed,
        state.poisoned,
        state.discarded,
    )
    graph = (
        state._data,
        state.snapshot,
        state.id_cache,
        state.claimed,
        state.frames,
        *cache_graph,
        *frame_graph,
    )
    return key, graph


def _publish_owner_state(state: _OwnerState) -> None:
    with _OWNER_LOCK:
        entry = _OWNER_REGISTRY.get(id(state.reference()))
        if entry is None or entry.state is not state or entry.reference is not state.reference:
            raise ConstraintProgramError("owner registry publication lost its owner")
        if entry.operation_active:
            # Active operations own a copy-on-write data root.  Only their
            # final registry swap may publish it.
            return
    mutable_key, mutable_graph = _owner_mutable_state(state)
    with _OWNER_LOCK:
        entry = _OWNER_REGISTRY.get(id(state.reference()))
        if (
            entry is None
            or entry.state is not state
            or entry.reference is not state.reference
            or entry.operation_active
        ):
            raise ConstraintProgramError("owner changed during publication")
        current_anchor = _owner_anchor_graph(state)
        if (
            len(current_anchor) != len(entry.anchor_graph)
            or any(
                current is not sealed
                for current, sealed in zip(current_anchor, entry.anchor_graph)
            )
        ):
            raise ConstraintProgramError("owner anchor changed during publication")
        _OWNER_REGISTRY[id(state.reference())] = _OwnerRegistryEntry(
            entry.reference,
            state,
            entry.anchor_graph,
            mutable_key,
            mutable_graph,
            False,
            entry.operation_epoch,
            False,
            None,
            False,
        )


_OWNER_LOCK = threading.Lock()
_OWNER_REGISTRY: Dict[int, _OwnerRegistryEntry] = {}
_OWNER_RESERVATIONS: Dict[int, _ReservationRecord] = {}


class ConstraintProgramOwner:
    """Thread-confined owner that delegates every factor allocation."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ConstraintProgramOwner":
        if cls is not ConstraintProgramOwner:
            raise TypeError("mutable/subclassed owners are rejected")
        return _reserve_owner_handle()

    @classmethod
    def reserve(cls) -> "ConstraintProgramOwner":
        """Return one exact, uninitialized, non-authoritative owner handle."""

        if cls is not ConstraintProgramOwner:
            raise TypeError("mutable/subclassed owners are rejected")
        return _reserve_owner_handle()

    def __init__(self, adapter: ExternalFactorAllocatorAdapter) -> None:
        self.initialize(adapter)

    def initialize(self, adapter: ExternalFactorAllocatorAdapter) -> None:
        _initialize_reserved_owner(self, adapter)

    def _allocate(self, kind: FactorKind, count: int) -> Tuple[ExternalFactorID, ...]:
        state = _require_owner_open(self)
        count = _exact_count(count, name=f"{kind.value} factor count")
        operation_epoch = _begin_owner_operation(state)
        callback = (
            state.adapter_record.allocate_continuous
            if kind is FactorKind.CONTINUOUS
            else state.adapter_record.allocate_binary
        )
        try:
            _activate_owner_operation(state, operation_epoch)
            try:
                # Linearize against external live truth *before* asking for
                # new IDs.  The owner-level operation guard also prevents any
                # captured callback from sealing or mutating its arena.
                pre_snapshot = _validate_snapshot(
                    _call_external(
                        state,
                        operation_epoch,
                        state.adapter_record.live_ids_snapshot,
                    )
                )
                if not _snapshot_is_append_only(state.snapshot, pre_snapshot):
                    raise ExternalAllocatorContractError(
                        "external factor snapshot is not append-only before allocation"
                    )
                pre_additions: Dict[
                    Tuple[FactorKind, int], ExternalFactorID
                ] = {}
                for current_kind, values in (
                    (FactorKind.CONTINUOUS, pre_snapshot[0]),
                    (FactorKind.BINARY, pre_snapshot[1]),
                ):
                    for raw in values:
                        key = (current_kind, raw)
                        if key not in state.id_cache:
                            pre_additions[key] = _new_factor_id(
                                current_kind,
                                raw,
                                state.adapter_record.namespace,
                            )
                if pre_snapshot != state.snapshot:
                    state.id_cache.update(pre_additions)
                    state.snapshot = pre_snapshot
                    state.version += 1
                    _publish_owner_state(state)
                _checkpoint_owner_operation(state, operation_epoch)
                issued_raw = _validate_raw_ids(
                    _call_external(state, operation_epoch, callback, count),
                    name=f"issued {kind.value} IDs",
                )
                if len(issued_raw) != count:
                    raise ExternalAllocatorContractError(
                        "external allocator returned the wrong number of factor IDs"
                    )
                new_snapshot = _validate_snapshot(
                    _call_external(
                        state,
                        operation_epoch,
                        state.adapter_record.live_ids_snapshot,
                    )
                )
                if not _snapshot_is_append_only(pre_snapshot, new_snapshot):
                    raise ExternalAllocatorContractError(
                        "external factor snapshot removed, reordered, or retyped an ID"
                    )
                expected_values = new_snapshot[
                    0 if kind is FactorKind.CONTINUOUS else 1
                ]
                wrong_values = new_snapshot[
                    1 if kind is FactorKind.CONTINUOUS else 0
                ]
                if (
                    any(raw not in expected_values for raw in issued_raw)
                    or any(raw in wrong_values for raw in issued_raw)
                    or any((kind, raw) in state.claimed for raw in issued_raw)
                    or any(
                        raw in pre_snapshot[0] or raw in pre_snapshot[1]
                        for raw in issued_raw
                    )
                ):
                    raise ExternalAllocatorContractError(
                        "issued IDs are stale, mistyped, absent from live truth, or reused"
                    )
            except BaseException:
                # The external allocator may already have burned IDs.  Never
                # call it again from this transaction or infer rollback.
                state.poisoned = True
                _publish_owner_state(state)
                raise
            old_snapshot = state.snapshot
            state.snapshot = new_snapshot
            state.claimed.update((kind, raw) for raw in issued_raw)
            for current_kind, values in (
                (FactorKind.CONTINUOUS, new_snapshot[0]),
                (FactorKind.BINARY, new_snapshot[1]),
            ):
                for raw in values:
                    state.id_cache.setdefault(
                        (current_kind, raw),
                        _new_factor_id(
                            current_kind,
                            raw,
                            state.adapter_record.namespace,
                        )
                    )
            if count or new_snapshot != old_snapshot:
                state.version += 1
            _publish_owner_state(state)
            _checkpoint_owner_operation(state, operation_epoch)
            _assert_owner_operation(state, operation_epoch)
            _commit_owner_operation(state, operation_epoch)
            return tuple(state.id_cache[(kind, raw)] for raw in issued_raw)
        finally:
            _end_owner_operation(state, operation_epoch)

    def allocate_continuous(self, count: int) -> Tuple[ExternalFactorID, ...]:
        return self._allocate(FactorKind.CONTINUOUS, count)

    def allocate_binary(self, count: int) -> Tuple[ExternalFactorID, ...]:
        return self._allocate(FactorKind.BINARY, count)

    def frame(self) -> FactorFrame:
        state = _require_owner_open(self)
        operation_epoch = _begin_owner_operation(state)
        try:
            _activate_owner_operation(state, operation_epoch)
            try:
                snapshot = _validate_snapshot(
                    _call_external(
                        state,
                        operation_epoch,
                        state.adapter_record.live_ids_snapshot,
                    )
                )
                if not _snapshot_is_append_only(state.snapshot, snapshot):
                    raise ExternalAllocatorContractError(
                        "external factor snapshot is not append-only"
                    )
            except BaseException:
                state.poisoned = True
                _publish_owner_state(state)
                raise
            if snapshot != state.snapshot:
                state.snapshot = snapshot
                state.version += 1
            for kind, values in (
                (FactorKind.CONTINUOUS, snapshot[0]),
                (FactorKind.BINARY, snapshot[1]),
            ):
                for raw in values:
                    state.id_cache.setdefault(
                        (kind, raw),
                        _new_factor_id(kind, raw, state.adapter_record.namespace),
                    )
            _checkpoint_owner_operation(state, operation_epoch)
            frame = FactorFrame(_token=_FACTORY_TOKEN)
            continuous = tuple(
                state.id_cache[(FactorKind.CONTINUOUS, raw)]
                for raw in snapshot[0]
            )
            binary = tuple(
                state.id_cache[(FactorKind.BINARY, raw)] for raw in snapshot[1]
            )
            object.__setattr__(frame, "_continuous_ids", continuous)
            object.__setattr__(frame, "_binary_ids", binary)
            object.__setattr__(frame, "_owner_token", state.token)
            object.__setattr__(frame, "_version", int(state.version))
            object.__setattr__(frame, "_sealed", True)
            truth = _frame_key(frame)
            graph = (
                frame._continuous_ids,
                frame._binary_ids,
                frame._owner_token,
                *frame._continuous_ids,
                *frame._binary_ids,
            )
            state.frames[id(frame)] = (frame, truth, graph)
            _publish_owner_state(state)
            _assert_owner_operation(state, operation_epoch)
            _commit_owner_operation(state, operation_epoch)
            return frame
        finally:
            _end_owner_operation(state, operation_epoch)

    def new_arena(self) -> "ConstraintArena":
        state = _require_owner_open(self)
        operation_epoch = _begin_owner_operation(state)
        staged_arena: Optional[ConstraintArena] = None
        staged_state: Optional[_ArenaState] = None
        rollback_data: Optional[_OwnerData] = None
        try:
            _activate_owner_operation(state, operation_epoch)
            active = _owner_operation_entry(state)
            if active is None or type(active.rollback_data) is not _OwnerData:
                raise ConstraintProgramError("new-arena rollback root was lost")
            rollback_data = active.rollback_data
            if state.arena_created:
                with _ARENA_LOCK:
                    existing = (
                        None
                        if state.arena_id is None
                        else _ARENA_REGISTRY.get(state.arena_id)
                    )
                existing_arena = None if existing is None else existing.reference()
                if (
                    existing_arena is not None
                    and existing.state.owner_state is state
                ):
                    if _arena_state(existing_arena) is not existing.state:
                        raise ConstraintProgramError(
                            "owner arena recovery binding was substituted"
                        )
                    _assert_owner_operation(state, operation_epoch)
                    _commit_owner_operation(state, operation_epoch)
                    return existing_arena
                # A prior call may have completed publication but lost its
                # return value to an asynchronous exception.  Once that
                # handle is genuinely gone, its weakref cleanup removes the
                # registry entry and the owner may safely recreate one arena.
                state.arena_created = False
                state.arena_id = None
            state.arena_created = True
            staged_arena, staged_state, staged_entry = _stage_arena(self, state)
            state.arena_id = id(staged_arena)
            _publish_owner_state(state)
            _register_staged_arena(staged_arena, staged_state, staged_entry)
            _assert_owner_operation(state, operation_epoch)
            _commit_owner_operation(state, operation_epoch)
            return staged_arena
        except BaseException:
            if staged_arena is not None and staged_state is not None:
                for _attempt in range(4):
                    try:
                        _unregister_staged_arena(staged_arena, staged_state)
                    except BaseException:
                        pass
                    with _ARENA_LOCK:
                        current = _ARENA_REGISTRY.get(id(staged_arena))
                    if current is None or current.state is not staged_state:
                        break
            if rollback_data is not None:
                repair_error: Optional[BaseException] = None
                for _attempt in range(4):
                    try:
                        _restore_owner_operation(
                            state,
                            operation_epoch,
                            rollback_data,
                        )
                        repair_error = None
                        break
                    except BaseException as error:
                        if repair_error is None:
                            repair_error = error
                        continue
                if repair_error is not None:
                    raise repair_error
            raise
        finally:
            _end_owner_operation(state, operation_epoch)

    @property
    def representation_authority(self) -> bool:
        _owner_state(self)
        return False

    @property
    def discarded(self) -> bool:
        state = _owner_state(self)
        _require_owner_thread(state)
        return state.discarded

    @property
    def proof_authority(self) -> bool:
        _owner_state(self)
        return False


def _reserve_owner_handle() -> ConstraintProgramOwner:
    owner = object.__new__(ConstraintProgramOwner)
    object_id = id(owner)

    def cleanup(reference: Any) -> None:
        for _attempt in range(4):
            try:
                with _OWNER_LOCK:
                    reserved = _OWNER_RESERVATIONS.get(object_id)
                    if reserved is not None and reserved.reference is reference:
                        _OWNER_RESERVATIONS.pop(object_id, None)
                    current = _OWNER_REGISTRY.get(object_id)
                    if current is not None and current.reference is reference:
                        _OWNER_REGISTRY.pop(object_id, None)
                return
            except BaseException:
                continue

    reference = weakref.ref(owner, cleanup)
    reservation = _make_reservation_record(
        reference,
        threading.current_thread(),
        _ReservationState(_ReservationPhase.RESERVED),
    )
    try:
        with _OWNER_LOCK:
            if id(owner) in _OWNER_RESERVATIONS or id(owner) in _OWNER_REGISTRY:
                raise ConstraintProgramError("owner reservation object ID was reused")
            _OWNER_RESERVATIONS[id(owner)] = reservation
    except BaseException:
        for _attempt in range(4):
            try:
                with _OWNER_LOCK:
                    current = _OWNER_RESERVATIONS.get(id(owner))
                    if current is not None and current.reference is reference:
                        _OWNER_RESERVATIONS.pop(id(owner), None)
                break
            except BaseException:
                continue
        raise
    return owner


def _owner_initialization_epoch(owner: ConstraintProgramOwner) -> int:
    if type(owner) is not ConstraintProgramOwner:
        raise ConstraintProgramError("owner has the wrong exact type")
    with _OWNER_LOCK:
        initialized = _OWNER_REGISTRY.get(id(owner))
        reservation = _OWNER_RESERVATIONS.get(id(owner))
        if initialized is not None and initialized.reference() is owner:
            raise ConstraintProgramError("constraint owner is already initialized")
        if reservation is None:
            raise ConstraintProgramError("constraint owner is forged, stale, or not reserved")
        reservation = _validate_reservation_record(
            reservation, owner, name="owner"
        )
        if threading.current_thread() is not reservation.thread:
            raise ConstraintProgramError(
                "owner initialization is Thread-object confined"
            )
        state = reservation.state
        if state.phase is _ReservationPhase.INITIALIZING:
            _OWNER_RESERVATIONS[id(owner)] = _make_reservation_record(
                reservation.reference,
                reservation.thread,
                _ReservationState(
                    state.phase,
                    state.epoch,
                    True,
                    state.external_touched,
                ),
            )
            raise ConstraintProgramError("owner initialization cannot reenter")
        if state.phase is _ReservationPhase.POISONED:
            raise ConstraintProgramError("owner reservation is poisoned")
        if state.phase is not _ReservationPhase.RESERVED:
            raise ConstraintProgramError("owner reservation state is invalid")
        return state.epoch + 1


def _activate_owner_initialization(owner: ConstraintProgramOwner, epoch: int) -> None:
    with _OWNER_LOCK:
        reservation = _OWNER_RESERVATIONS.get(id(owner))
        if reservation is None:
            raise ConstraintProgramError(
                "owner reservation vanished before initialization"
            )
        reservation = _validate_reservation_record(
            reservation, owner, name="owner"
        )
        state = reservation.state
        if (
            threading.current_thread() is not reservation.thread
            or state.phase is not _ReservationPhase.RESERVED
            or state.epoch + 1 != epoch
        ):
            raise ConstraintProgramError(
                "owner reservation changed before initialization"
            )
        _OWNER_RESERVATIONS[id(owner)] = _make_reservation_record(
            reservation.reference,
            reservation.thread,
            _ReservationState(_ReservationPhase.INITIALIZING, epoch),
        )


def _mark_owner_initialization_external(
    owner: ConstraintProgramOwner, epoch: int
) -> None:
    with _OWNER_LOCK:
        reservation = _validate_reservation_record(
            _OWNER_RESERVATIONS.get(id(owner)),
            owner,
            name="owner",
        )
        state = reservation.state
        if (
            state.phase is not _ReservationPhase.INITIALIZING
            or state.epoch != epoch
            or state.reentrancy_detected
            or state.external_touched
        ):
            raise ExternalAllocatorContractError(
                "owner initialization callback escaped its reservation"
            )
        _OWNER_RESERVATIONS[id(owner)] = _make_reservation_record(
            reservation.reference,
            reservation.thread,
            _ReservationState(
                state.phase,
                state.epoch,
                state.reentrancy_detected,
                True,
            ),
        )


def _checkpoint_owner_initialization(
    owner: ConstraintProgramOwner, epoch: int
) -> _ReservationRecord:
    with _OWNER_LOCK:
        reservation = _validate_reservation_record(
            _OWNER_RESERVATIONS.get(id(owner)),
            owner,
            name="owner",
        )
        state = reservation.state
        if (
            state.phase is not _ReservationPhase.INITIALIZING
            or state.epoch != epoch
            or state.reentrancy_detected
            or not state.external_touched
        ):
            raise ExternalAllocatorContractError(
                "owner initialization callback attempted reentrant mutation"
            )
        checkpoint = _make_reservation_record(
            reservation.reference,
            reservation.thread,
            _ReservationState(_ReservationPhase.INITIALIZING, epoch),
        )
        _OWNER_RESERVATIONS[id(owner)] = checkpoint
        return checkpoint


def _poison_owner_initialization(
    owner: ConstraintProgramOwner,
    epoch: int,
    *,
    staged_entry: Optional[_OwnerRegistryEntry] = None,
) -> None:
    for _attempt in range(4):
        try:
            with _OWNER_LOCK:
                current = _OWNER_REGISTRY.get(id(owner))
                # Installing this exact immutable registry entry is the owner
                # construction linearization point.  A later interruption may
                # leave a reservation residue, but cleanup must complete NEW,
                # never revoke the usable handle.
                if staged_entry is not None and current is staged_entry:
                    _OWNER_RESERVATIONS.pop(id(owner), None)
                    return
                reservation = _OWNER_RESERVATIONS.get(id(owner))
                if reservation is not None:
                    reservation = _validate_reservation_record(
                        reservation, owner, name="owner"
                    )
                if staged_entry is not None and current is None:
                    can_complete = reservation is None
                    if reservation is not None:
                        state = reservation.state
                        can_complete = bool(
                            state.phase is _ReservationPhase.INITIALIZING
                            and state.epoch == epoch
                            and not state.reentrancy_detected
                            and not state.external_touched
                        )
                    if can_complete:
                        _OWNER_REGISTRY[id(owner)] = staged_entry
                        _OWNER_RESERVATIONS.pop(id(owner), None)
                        return
                if reservation is not None:
                    _OWNER_RESERVATIONS[id(owner)] = _make_reservation_record(
                        reservation.reference,
                        reservation.thread,
                        _ReservationState(
                            _ReservationPhase.POISONED,
                            max(epoch, reservation.state.epoch),
                        ),
                    )
            return
        except BaseException:
            continue


def _publish_owner_initialization(
    owner: ConstraintProgramOwner,
    epoch: int,
    entry: _OwnerRegistryEntry,
) -> None:
    with _OWNER_LOCK:
        reservation = _validate_reservation_record(
            _OWNER_RESERVATIONS.get(id(owner)),
            owner,
            name="owner",
        )
        state = reservation.state
        if (
            state.phase is not _ReservationPhase.INITIALIZING
            or state.epoch != epoch
            or state.reentrancy_detected
            or state.external_touched
            or _OWNER_REGISTRY.get(id(owner)) is not None
        ):
            raise ConstraintProgramError(
                "owner initialization changed before publication"
            )
        _OWNER_REGISTRY[id(owner)] = entry
        _OWNER_RESERVATIONS.pop(id(owner), None)


def _repair_owner_initialization(
    owner: ConstraintProgramOwner,
    epoch: int,
    entry: _OwnerRegistryEntry,
) -> bool:
    for _attempt in range(4):
        try:
            with _OWNER_LOCK:
                current = _OWNER_REGISTRY.get(id(owner))
                reservation = _OWNER_RESERVATIONS.get(id(owner))
                if current is entry:
                    _OWNER_RESERVATIONS.pop(id(owner), None)
                    return True
                can_complete = False
                if reservation is not None:
                    reservation = _validate_reservation_record(
                        reservation, owner, name="owner"
                    )
                    state = reservation.state
                    can_complete = bool(
                        state.phase is _ReservationPhase.INITIALIZING
                        and state.epoch == epoch
                        and not state.reentrancy_detected
                        and not state.external_touched
                        and current is None
                    )
                if can_complete:
                    _OWNER_REGISTRY[id(owner)] = entry
                    _OWNER_RESERVATIONS.pop(id(owner), None)
                    return True
                if reservation is not None:
                    _OWNER_RESERVATIONS[id(owner)] = _make_reservation_record(
                        reservation.reference,
                        reservation.thread,
                        _ReservationState(_ReservationPhase.POISONED, epoch),
                    )
                return False
        except BaseException:
            continue
    _poison_owner_initialization(
        owner,
        epoch,
        staged_entry=entry,
    )
    return False


def _initialize_reserved_owner(
    owner: ConstraintProgramOwner,
    adapter: ExternalFactorAllocatorAdapter,
) -> None:
    epoch = _owner_initialization_epoch(owner)
    staged_entry: Optional[_OwnerRegistryEntry] = None
    try:
        _activate_owner_initialization(owner, epoch)
        adapter_record = _adapter_record(adapter)
        with _OWNER_LOCK:
            reservation = _validate_reservation_record(
                _OWNER_RESERVATIONS.get(id(owner)),
                owner,
                name="owner",
            )
        if (
            threading.current_thread() is not reservation.thread
            or reservation.thread is not adapter_record.thread
        ):
            raise ExternalAllocatorContractError(
                "adapter and owner must be initialized on the reserving Thread object"
            )
        _mark_owner_initialization_external(owner, epoch)
        snapshot = _validate_snapshot(adapter_record.live_ids_snapshot())
        if _adapter_record(adapter) is not adapter_record:
            raise ExternalAllocatorContractError(
                "captured allocator adapter changed during owner initialization"
            )
        reservation = _checkpoint_owner_initialization(owner, epoch)
        cache: Dict[Tuple[FactorKind, int], ExternalFactorID] = {}
        for kind, values in (
            (FactorKind.CONTINUOUS, snapshot[0]),
            (FactorKind.BINARY, snapshot[1]),
        ):
            for raw in values:
                cache[(kind, raw)] = _new_factor_id(
                    kind, raw, adapter_record.namespace
                )
        state = _OwnerState(
            reservation.reference,
            adapter,
            adapter_record,
            reservation.thread,
            _OwnerToken(),
            snapshot,
            cache,
            set(),
            {},
        )
        mutable_key, mutable_graph = _owner_mutable_state(state)
        staged_entry = _OwnerRegistryEntry(
            reservation.reference,
            state,
            _owner_anchor_graph(state),
            mutable_key,
            mutable_graph,
        )
        _publish_owner_initialization(owner, epoch, staged_entry)
    except BaseException:
        completed = False
        if staged_entry is not None:
            try:
                completed = _repair_owner_initialization(
                    owner,
                    epoch,
                    staged_entry,
                )
            except BaseException:
                with _OWNER_LOCK:
                    completed = _OWNER_REGISTRY.get(id(owner)) is staged_entry
        if not completed:
            for _attempt in range(4):
                try:
                    _poison_owner_initialization(
                        owner,
                        epoch,
                        staged_entry=staged_entry,
                    )
                    break
                except BaseException:
                    continue
        raise


def _owner_state(value: Any) -> _OwnerState:
    if type(value) is not ConstraintProgramOwner:
        raise ConstraintProgramError("owner has the wrong exact type")
    with _OWNER_LOCK:
        entry = _OWNER_REGISTRY.get(id(value))
        reservation = _OWNER_RESERVATIONS.get(id(value))
        # A complete main entry wins over an interrupted cleanup of its
        # reservation guard.  Before the main swap, the guard rejects every
        # accessor and makes swallowed same-thread reentrancy sticky.
        if (
            (entry is None or entry.reference() is not value)
            and reservation is not None
        ):
            reservation = _validate_reservation_record(
                reservation,
                value,
                name="owner",
            )
            reservation_state = reservation.state
            if (
                reservation_state.phase is _ReservationPhase.INITIALIZING
                and threading.current_thread() is reservation.thread
                and not reservation_state.reentrancy_detected
            ):
                _OWNER_RESERVATIONS[id(value)] = _make_reservation_record(
                    reservation.reference,
                    reservation.thread,
                    _ReservationState(
                        reservation_state.phase,
                        reservation_state.epoch,
                        True,
                        reservation_state.external_touched,
                    ),
                )
            raise ConstraintProgramError(
                "constraint owner is reserved or uninitialized"
            )
    if entry is None or entry.reference() is not value:
        raise ConstraintProgramError("owner is forged or stale")
    if (
        type(entry.operation_active) is not bool
        or type(entry.operation_epoch) is not int
        or entry.operation_epoch < 0
        or type(entry.reentrancy_detected) is not bool
        or (entry.reentrancy_detected and not entry.operation_active)
        or (
            entry.operation_active
            and type(entry.rollback_data) is not _OwnerData
        )
        or (not entry.operation_active and entry.rollback_data is not None)
        or type(entry.external_touched) is not bool
        or (entry.external_touched and not entry.operation_active)
    ):
        raise ConstraintProgramError("owner operation registry was substituted")
    state = entry.state
    current_graph = _owner_anchor_graph(state)
    if (
        len(current_graph) != len(entry.anchor_graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_graph, entry.anchor_graph)
        )
    ):
        raise ConstraintProgramError("owner registry anchor was rebound")
    if not entry.operation_active:
        mutable_key, mutable_graph = _owner_mutable_state(state)
        if (
            mutable_key != entry.mutable_key
            or len(mutable_graph) != len(entry.mutable_graph)
            or any(
                current is not sealed
                for current, sealed in zip(mutable_graph, entry.mutable_graph)
            )
        ):
            raise ConstraintProgramError("owner mutable registry state was rebound")
    current_adapter = _adapter_record(state.adapter)
    if current_adapter is not state.adapter_record:
        raise ExternalAllocatorContractError("captured allocator adapter was rebound")
    return state


def _require_owner_thread(state: _OwnerState) -> None:
    if threading.current_thread() is not state.thread:
        raise ConstraintProgramError("mutable owner/arena is Thread-object confined")


def _mark_owner_reentrancy(state: _OwnerState) -> None:
    """Record an attempted nested mutation even if its caller swallows it."""
    with _OWNER_LOCK:
        entry = _OWNER_REGISTRY.get(id(state.reference()))
        if (
            entry is None
            or entry.state is not state
            or not entry.operation_active
        ):
            raise ConstraintProgramError("owner reentrancy guard is not active")
        _OWNER_REGISTRY[id(state.reference())] = replace(
            entry,
            reentrancy_detected=True,
        )


def _require_owner_open(value: Any) -> _OwnerState:
    state = _owner_state(value)
    _require_owner_thread(state)
    with _OWNER_LOCK:
        entry = _OWNER_REGISTRY.get(id(value))
    if entry is not None and entry.operation_active:
        _mark_owner_reentrancy(state)
        raise ConstraintProgramError(
            "owner/arena mutation cannot reenter an active owner operation"
        )
    if state.discarded:
        raise ConstraintProgramError("constraint owner is discarded")
    if state.sealed:
        raise ConstraintProgramError("constraint owner is sealed")
    if state.poisoned:
        raise ConstraintProgramError("constraint owner is poisoned by an allocator failure")
    return state


def _begin_owner_operation(state: _OwnerState) -> int:
    """Reserve an epoch without mutating shared state.

    Activation is deliberately a second call made *inside* each caller's
    try/finally.  A tracing exception on this function's return therefore
    cannot strand an active owner before the caller knows the epoch.
    """

    owner = state.reference()
    if owner is None:
        raise ConstraintProgramError("owner vanished before operation start")
    with _OWNER_LOCK:
        entry = _OWNER_REGISTRY.get(id(owner))
        if (
            entry is None
            or entry.reference() is not owner
            or entry.state is not state
            or entry.operation_active
        ):
            raise ConstraintProgramError("owner operation is already active")
        return entry.operation_epoch + 1


def _activate_owner_operation(state: _OwnerState, epoch: int) -> None:
    owner = state.reference()
    if owner is None:
        raise ConstraintProgramError("owner vanished before operation activation")
    old_data = state._data
    working_data = replace(
        old_data,
        id_cache=dict(old_data.id_cache),
        claimed=set(old_data.claimed),
        frames=dict(old_data.frames),
    )
    with _OWNER_LOCK:
        entry = _OWNER_REGISTRY.get(id(owner))
        if (
            entry is None
            or entry.reference() is not owner
            or entry.state is not state
            or entry.operation_active
            or entry.operation_epoch + 1 != epoch
        ):
            raise ConstraintProgramError("owner operation epoch could not activate")
        _OWNER_REGISTRY[id(owner)] = replace(
            entry,
            operation_active=True,
            operation_epoch=epoch,
            reentrancy_detected=False,
            rollback_data=old_data,
            external_touched=False,
        )
    state._data = working_data


def _assert_owner_operation(state: _OwnerState, epoch: int) -> None:
    owner = state.reference()
    with _OWNER_LOCK:
        entry = None if owner is None else _OWNER_REGISTRY.get(id(owner))
    if (
        type(epoch) is not int
        or entry is None
        or entry.reference() is not owner
        or entry.state is not state
        or not entry.operation_active
        or entry.operation_epoch != epoch
        or entry.reentrancy_detected
    ):
        raise ExternalAllocatorContractError(
            "external allocator callback attempted reentrant mutation"
        )
    current_anchor = _owner_anchor_graph(state)
    if (
        len(current_anchor) != len(entry.anchor_graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_anchor, entry.anchor_graph)
        )
        or _adapter_record(state.adapter) is not state.adapter_record
    ):
        raise ExternalAllocatorContractError(
            "external allocator callback rebound its captured owner/adapter graph"
        )


def _owner_operation_entry(state: _OwnerState) -> Optional[_OwnerRegistryEntry]:
    owner = state.reference()
    with _OWNER_LOCK:
        entry = None if owner is None else _OWNER_REGISTRY.get(id(owner))
    if entry is not None and entry.state is not state:
        raise ConstraintProgramError("owner operation entry changed identity")
    return entry


def _commit_owner_operation(state: _OwnerState, epoch: int) -> None:
    owner = state.reference()
    if owner is None:
        raise ConstraintProgramError("owner vanished before final operation swap")
    mutable_key, mutable_graph = _owner_mutable_state(state)
    with _OWNER_LOCK:
        current = _OWNER_REGISTRY.get(id(owner))
        if (
            current is None
            or current.state is not state
            or not current.operation_active
            or current.operation_epoch != epoch
        ):
            raise ConstraintProgramError("owner operation changed before final swap")
        if current.reentrancy_detected:
            # A nested mutation may have been rejected and swallowed by a
            # final-swap hook.  The sticky bit is authoritative until this
            # epoch is compensated; publishing the working root would let the
            # outer operation succeed despite the allocator-contract breach.
            raise ExternalAllocatorContractError(
                "external allocator callback attempted reentrant mutation"
            )
        _OWNER_REGISTRY[id(owner)] = _OwnerRegistryEntry(
            current.reference,
            state,
            current.anchor_graph,
            mutable_key,
            mutable_graph,
            False,
            epoch,
            False,
            None,
            False,
        )


def _restore_owner_operation(
    state: _OwnerState,
    epoch: int,
    rollback: _OwnerData,
    *,
    poisoned: bool = False,
) -> None:
    """Restore an operation root, whether its final swap ran or not.

    This is intentionally an epoch-scoped compensating transition.  It is
    used only before any public authority is registered (new-arena/seal error
    paths), so a post-swap asynchronous exception can still converge to OLD.
    """

    if type(rollback) is not _OwnerData:
        raise ConstraintProgramError("owner rollback root was lost")
    owner = state.reference()
    if owner is None:
        raise ConstraintProgramError("owner vanished before rollback")
    target = replace(rollback, poisoned=True) if poisoned else rollback
    state._data = target
    mutable_key, mutable_graph = _owner_mutable_state(state)
    with _OWNER_LOCK:
        current = _OWNER_REGISTRY.get(id(owner))
        if (
            current is None
            or current.state is not state
            or current.operation_epoch != epoch
        ):
            raise ConstraintProgramError("owner epoch changed before rollback")
        _OWNER_REGISTRY[id(owner)] = _OwnerRegistryEntry(
            current.reference,
            state,
            current.anchor_graph,
            mutable_key,
            mutable_graph,
            False,
            epoch,
            False,
            None,
            False,
        )


def _end_owner_operation(state: _OwnerState, epoch: int) -> None:
    owner = state.reference()
    if owner is None:
        return
    body_failed = sys.exc_info()[0] is not None
    repair_error: Optional[BaseException] = None
    for _attempt in range(4):
        reentrancy_finalized = False
        try:
            with _OWNER_LOCK:
                entry = _OWNER_REGISTRY.get(id(owner))
            if (
                entry is None
                or entry.state is not state
                or entry.operation_epoch != epoch
            ):
                if repair_error is not None:
                    raise repair_error
                return
            if not entry.operation_active:
                # A dedicated publication helper (seal/new_arena) may already
                # have completed or compensated this epoch.
                if repair_error is not None and not body_failed:
                    raise repair_error
                return
            if entry.reentrancy_detected:
                rollback = entry.rollback_data
                if type(rollback) is not _OwnerData:
                    raise ConstraintProgramError("owner rollback root was lost")
                _restore_owner_operation(
                    state,
                    epoch,
                    rollback,
                    poisoned=bool(state.poisoned or entry.external_touched),
                )
                reentrancy_finalized = True
            elif body_failed:
                rollback = entry.rollback_data
                if type(rollback) is not _OwnerData:
                    raise ConstraintProgramError("owner rollback root was lost")
                _restore_owner_operation(
                    state,
                    epoch,
                    rollback,
                    poisoned=bool(state.poisoned or entry.external_touched),
                )
            else:
                _commit_owner_operation(state, epoch)
        except BaseException as error:
            if repair_error is None:
                repair_error = error
            # Retry from registry truth.  A one-shot asynchronous exception
            # may have happened immediately before or after the atomic entry
            # replacement; the next pass distinguishes those cases.
            continue
        if reentrancy_finalized:
            if not body_failed:
                raise ExternalAllocatorContractError(
                    "external allocator callback attempted reentrant mutation"
                )
            return
        if repair_error is not None and not body_failed:
            raise repair_error
        return
    if repair_error is not None:
        raise repair_error
    raise ConstraintProgramError("owner operation could not reach an idle state")


def _checkpoint_owner_operation(state: _OwnerState, epoch: int) -> None:
    """Make fully reconciled external truth the operation's rollback root.

    The active working root remains private.  A distinct container graph is
    installed after the checkpoint so later frame/seal mutations cannot alter
    the rollback root in place.
    """

    owner = state.reference()
    if owner is None:
        raise ConstraintProgramError("owner vanished before reconciliation checkpoint")
    _assert_owner_operation(state, epoch)
    checkpoint = replace(
        state._data,
        id_cache=dict(state.id_cache),
        claimed=set(state.claimed),
        frames=dict(state.frames),
    )
    working = replace(
        checkpoint,
        id_cache=dict(checkpoint.id_cache),
        claimed=set(checkpoint.claimed),
        frames=dict(checkpoint.frames),
    )
    with _OWNER_LOCK:
        entry = _OWNER_REGISTRY.get(id(owner))
        if (
            entry is None
            or entry.state is not state
            or not entry.operation_active
            or entry.operation_epoch != epoch
            or entry.reentrancy_detected
        ):
            raise ConstraintProgramError(
                "owner changed before reconciliation checkpoint"
            )
        _OWNER_REGISTRY[id(owner)] = replace(
            entry,
            rollback_data=checkpoint,
            external_touched=False,
        )
    state._data = working


def _mark_external_call(state: _OwnerState, epoch: int) -> None:
    owner = state.reference()
    with _OWNER_LOCK:
        entry = None if owner is None else _OWNER_REGISTRY.get(id(owner))
        if (
            entry is None
            or entry.state is not state
            or not entry.operation_active
            or entry.operation_epoch != epoch
            or entry.reentrancy_detected
        ):
            raise ExternalAllocatorContractError(
                "external callback escaped its owner operation"
            )
        _OWNER_REGISTRY[id(owner)] = replace(entry, external_touched=True)


def _call_external(
    state: _OwnerState,
    epoch: int,
    callback: Callable[..., Any],
    *args: Any,
) -> Any:
    _assert_owner_operation(state, epoch)
    _mark_external_call(state, epoch)
    result = callback(*args)
    _assert_owner_operation(state, epoch)
    return result


def _frame_key(value: Any) -> Tuple[Any, ...]:
    if (
        type(value) is not FactorFrame
        or type(value._continuous_ids) is not tuple
        or type(value._binary_ids) is not tuple
        or type(value._owner_token) is not _OwnerToken
        or type(value._version) is not int
        or value._version < 0
        or value._sealed is not True
    ):
        raise ConstraintProgramError("factor frame was forged or mutated")
    continuous = tuple(_factor_key(item) for item in value._continuous_ids)
    binary = tuple(_factor_key(item) for item in value._binary_ids)
    if any(item[0] != FactorKind.CONTINUOUS.value for item in continuous) or any(
        item[0] != FactorKind.BINARY.value for item in binary
    ):
        raise ConstraintProgramError("factor frame has a mistyped factor")
    raw = tuple(item[1] for item in continuous + binary)
    if len(set(raw)) != len(raw):
        raise ConstraintProgramError("factor frame raw IDs are not disjoint")
    namespaces = tuple(item[2] for item in continuous + binary)
    if namespaces and any(item is not namespaces[0] for item in namespaces):
        raise ConstraintProgramError("factor frame crosses external namespaces")
    return continuous, binary, value._version, value._owner_token


def _validate_frame_state(state: _OwnerState, frame: Any) -> Tuple[Any, ...]:
    if type(frame) is not FactorFrame:
        raise ConstraintProgramError("factor frame has the wrong exact type")
    record = state.frames.get(id(frame))
    current = _frame_key(frame)
    if (
        record is None
        or record[0] is not frame
        or record[1] != current
        or frame._owner_token is not state.token
        or frame._continuous_ids is not record[2][0]
        or frame._binary_ids is not record[2][1]
        or frame._owner_token is not record[2][2]
        or len(record[2]) != 3 + len(frame._continuous_ids) + len(frame._binary_ids)
        or any(
            current is not sealed
            for current, sealed in zip(
                frame._continuous_ids + frame._binary_ids,
                record[2][3:],
            )
        )
    ):
        raise ConstraintArenaMismatch("factor frame is forged, stale, or foreign")
    return record[1]


def _validate_frame(owner: ConstraintProgramOwner, frame: Any) -> Tuple[Any, ...]:
    return _validate_frame_state(_require_owner_open(owner), frame)


def _validate_final_frame(
    owner: ConstraintProgramOwner,
    frame: Any,
    *,
    owner_state: Optional[_OwnerState] = None,
    operation_epoch: Optional[int] = None,
) -> Tuple[Any, ...]:
    """Poll external live truth and require the exact latest owner frame."""

    if owner_state is None or type(operation_epoch) is not int:
        raise ConstraintProgramError(
            "final-frame validation requires an active sealed-operation guard"
        )
    state = owner_state
    if _owner_state(owner) is not state:
        raise ConstraintProgramError("final-frame owner operation was substituted")
    epoch = operation_epoch
    _assert_owner_operation(state, epoch)
    frame_key = _validate_frame_state(state, frame)
    try:
        snapshot = _validate_snapshot(
            _call_external(
                state,
                epoch,
                state.adapter_record.live_ids_snapshot,
            )
        )
        if not _snapshot_is_append_only(state.snapshot, snapshot):
            raise ExternalAllocatorContractError(
                "external factor snapshot is not append-only at seal"
            )
        additions: Dict[Tuple[FactorKind, int], ExternalFactorID] = {}
        for kind, values in (
            (FactorKind.CONTINUOUS, snapshot[0]),
            (FactorKind.BINARY, snapshot[1]),
        ):
            for raw in values:
                key = (kind, raw)
                if key not in state.id_cache:
                    additions[key] = _new_factor_id(
                        kind, raw, state.adapter_record.namespace
                    )
    except BaseException:
        state.poisoned = True
        _publish_owner_state(state)
        raise
    if snapshot != state.snapshot:
        state.id_cache.update(additions)
        state.snapshot = snapshot
        state.version += 1
        _publish_owner_state(state)
    _checkpoint_owner_operation(state, epoch)
    frame_cont_raw = tuple(item[1] for item in frame_key[0])
    frame_bin_raw = tuple(item[1] for item in frame_key[1])
    if (
        frame_cont_raw != state.snapshot[0]
        or frame_bin_raw != state.snapshot[1]
        or frame_key[2] != state.version
    ):
        raise ConstraintProgramError(
            "final factor frame is stale relative to external live truth"
        )
    _assert_owner_operation(state, epoch)
    return frame_key


@dataclass(frozen=True)
class _FrozenCSR:
    rows: int
    columns: int
    data_bytes: bytes
    indices_bytes: bytes
    indptr_bytes: bytes

    @classmethod
    def from_value(cls, value: Any, *, name: str) -> "_FrozenCSR":
        if type(value) is not sp.csr_matrix or value.dtype != np.dtype(np.float64):
            raise ConstraintProgramError(
                f"{name} must be an exact float64 scipy.sparse.csr_matrix"
            )
        rows, columns = int(value.shape[0]), int(value.shape[1])
        data = np.array(value.data, dtype=np.float64, order="C", copy=True)
        indices = np.array(value.indices, dtype=np.int64, order="C", copy=True)
        indptr = np.array(value.indptr, dtype=np.int64, order="C", copy=True)
        if (
            rows < 0
            or columns < 0
            or data.ndim != 1
            or indices.ndim != 1
            or indptr.ndim != 1
            or data.size != indices.size
            or indptr.size != rows + 1
            or int(indptr[0]) != 0
            or int(indptr[-1]) != int(data.size)
            or np.any(indptr[1:] < indptr[:-1])
            or (indices.size and (np.any(indices < 0) or np.any(indices >= columns)))
            or (data.size and (not np.all(np.isfinite(data)) or np.any(data == 0.0)))
        ):
            raise ConstraintProgramError(
                f"{name} is malformed, non-finite, or contains explicit zero"
            )
        for row in range(rows):
            start, stop = int(indptr[row]), int(indptr[row + 1])
            if stop - start > 1 and np.any(indices[start + 1 : stop] <= indices[start : stop - 1]):
                raise ConstraintProgramError(f"{name} is not canonical CSR")
        return cls(
            rows,
            columns,
            bytes(data.tobytes(order="C")),
            bytes(indices.tobytes(order="C")),
            bytes(indptr.tobytes(order="C")),
        )

    @property
    def nnz(self) -> int:
        return len(self.data_bytes) // 8

    @property
    def payload_bytes(self) -> int:
        return len(self.data_bytes) + len(self.indices_bytes) + len(self.indptr_bytes)

    def key(self) -> Tuple[Any, ...]:
        if (
            type(self.rows) is not int
            or type(self.columns) is not int
            or type(self.data_bytes) is not bytes
            or type(self.indices_bytes) is not bytes
            or type(self.indptr_bytes) is not bytes
        ):
            raise ConstraintProgramError("frozen CSR authority was substituted")
        return (
            self.rows,
            self.columns,
            self.data_bytes,
            self.indices_bytes,
            self.indptr_bytes,
        )

    def csr(self, *, columns: Optional[int] = None) -> sp.csr_matrix:
        target = self.columns if columns is None else int(columns)
        if target < self.columns:
            raise ConstraintProgramError("cannot shrink a sealed factor frame")
        data = np.frombuffer(self.data_bytes, dtype=np.float64)
        indices = np.frombuffer(self.indices_bytes, dtype=np.int64)
        indptr = np.frombuffer(self.indptr_bytes, dtype=np.int64)
        result = sp.csr_matrix(
            (data, indices, indptr),
            shape=(self.rows, target),
            dtype=np.float64,
            copy=False,
        )
        result.data.setflags(write=False)
        result.indices.setflags(write=False)
        result.indptr.setflags(write=False)
        return result


@dataclass(frozen=True)
class _FrozenVector:
    length: int
    raw: bytes

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        name: str,
        finite: bool,
        lower_bound: bool = False,
    ) -> "_FrozenVector":
        if (
            type(value) is not np.ndarray
            or value.dtype != np.dtype(np.float64)
            or value.ndim != 1
        ):
            raise ConstraintProgramError(
                f"{name} must be an exact rank-one float64 ndarray"
            )
        snapshot = np.array(value, dtype=np.float64, order="C", copy=True)
        if snapshot.ndim != 1 or np.any(np.isnan(snapshot)):
            raise ConstraintProgramError(f"{name} changed shape or contains NaN")
        if finite and not np.all(np.isfinite(snapshot)):
            raise ConstraintProgramError(f"{name} must be finite")
        if lower_bound and np.any(np.isposinf(snapshot)):
            raise ConstraintProgramError(f"{name} contains +inf")
        return cls(int(snapshot.size), bytes(snapshot.tobytes(order="C")))

    @classmethod
    def from_bits(cls, bits: Sequence[int], *, name: str) -> "_FrozenVector":
        array = np.asarray(tuple(bits), dtype=np.uint64).view(np.float64)
        return cls.from_value(array, name=name, finite=False, lower_bound=True)

    def array(self) -> np.ndarray:
        result = np.frombuffer(self.raw, dtype=np.float64)
        result.setflags(write=False)
        return result

    @property
    def payload_bytes(self) -> int:
        return len(self.raw)

    def key(self) -> Tuple[Any, ...]:
        if type(self.length) is not int or type(self.raw) is not bytes:
            raise ConstraintProgramError("frozen bound authority was substituted")
        return self.length, self.raw


def _flip_bits(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float64).view(np.uint64).copy()
    bits ^= np.uint64(1 << 63)
    return bits.view(np.float64)


def _csr_row_bits_equal_negative(a: _FrozenCSR, b: _FrozenCSR) -> Tuple[bool, ...]:
    if a.rows != b.rows or a.columns != b.columns:
        raise ConstraintProgramError("guarded-band coefficient shapes differ")
    ai = np.frombuffer(a.indices_bytes, dtype=np.int64)
    bi = np.frombuffer(b.indices_bytes, dtype=np.int64)
    ap = np.frombuffer(a.indptr_bytes, dtype=np.int64)
    bp = np.frombuffer(b.indptr_bytes, dtype=np.int64)
    ad = np.frombuffer(a.data_bytes, dtype=np.float64).view(np.uint64)
    bd = np.frombuffer(b.data_bytes, dtype=np.float64).view(np.uint64)
    result = []
    sign = np.uint64(1 << 63)
    for row in range(a.rows):
        a0, a1 = int(ap[row]), int(ap[row + 1])
        b0, b1 = int(bp[row]), int(bp[row + 1])
        result.append(
            a1 - a0 == b1 - b0
            and np.array_equal(ai[a0:a1], bi[b0:b1])
            and np.array_equal(ad[a0:a1] ^ sign, bd[b0:b1])
        )
    return tuple(result)


def _frozen_rows(value: _FrozenCSR, positions: Sequence[int]) -> _FrozenCSR:
    source = value.csr()
    rows = []
    for position in positions:
        start, stop = int(source.indptr[position]), int(source.indptr[position + 1])
        rows.append((source.indices[start:stop], source.data[start:stop]))
    return _FrozenCSR.from_value(
        _assemble_rows(rows, columns=value.columns), name="selected frozen rows"
    )


def _stack_frozen(first: _FrozenCSR, second: _FrozenCSR) -> _FrozenCSR:
    if first.columns != second.columns:
        raise ConstraintProgramError("cannot stack different factor frames")
    rows = []
    for source in (first.csr(), second.csr()):
        for row in range(source.shape[0]):
            start, stop = int(source.indptr[row]), int(source.indptr[row + 1])
            rows.append((source.indices[start:stop], source.data[start:stop]))
    return _FrozenCSR.from_value(
        _assemble_rows(rows, columns=first.columns), name="stacked frozen rows"
    )


@dataclass(frozen=True)
class _Payload:
    block_kind: _BlockKind
    frame_cont_keys: Tuple[Tuple[str, int, _NamespaceIdentity], ...]
    frame_bin_keys: Tuple[Tuple[str, int, _NamespaceIdentity], ...]
    A_cont: _FrozenCSR
    A_bin: _FrozenCSR
    lower: _FrozenVector
    upper: _FrozenVector
    virtual_rows: int
    virtual_nnz: int
    legacy_positions: Tuple[int, ...]
    legacy_flip: Tuple[bool, ...]
    legacy_bound_from_lower: Tuple[bool, ...]
    ranged_mask: Tuple[bool, ...]
    reverse_positions: Tuple[int, ...]

    def key(self) -> Tuple[Any, ...]:
        if (
            type(self.block_kind) is not _BlockKind
            or type(self.frame_cont_keys) is not tuple
            or type(self.frame_bin_keys) is not tuple
            or type(self.A_cont) is not _FrozenCSR
            or type(self.A_bin) is not _FrozenCSR
            or type(self.lower) is not _FrozenVector
            or type(self.upper) is not _FrozenVector
            or type(self.virtual_rows) is not int
            or type(self.virtual_nnz) is not int
            or type(self.legacy_positions) is not tuple
            or type(self.legacy_flip) is not tuple
            or type(self.legacy_bound_from_lower) is not tuple
            or type(self.ranged_mask) is not tuple
            or type(self.reverse_positions) is not tuple
        ):
            raise ConstraintProgramError("constraint payload graph was substituted")
        continuous_keys = _validate_external_keys(
            self.frame_cont_keys,
            kind=FactorKind.CONTINUOUS,
            name="payload continuous factor keys",
        )
        binary_keys = _validate_external_keys(
            self.frame_bin_keys,
            kind=FactorKind.BINARY,
            name="payload binary factor keys",
        )
        if {item[1] for item in continuous_keys}.intersection(
            item[1] for item in binary_keys
        ):
            raise ConstraintProgramError("payload factor raw IDs collide across kinds")
        namespaces = tuple(item[2] for item in continuous_keys + binary_keys)
        if namespaces and any(item is not namespaces[0] for item in namespaces):
            raise ConstraintProgramError("payload crosses external factor namespaces")
        rows = self.A_cont.rows
        if (
            self.A_bin.rows != rows
            or self.lower.length != rows
            or self.upper.length != rows
            or self.A_cont.columns != len(self.frame_cont_keys)
            or self.A_bin.columns != len(self.frame_bin_keys)
            or len(self.legacy_positions) != self.virtual_rows
            or len(self.legacy_flip) != self.virtual_rows
            or len(self.legacy_bound_from_lower) != self.virtual_rows
            or any(type(item) is not int or not 0 <= item < rows for item in self.legacy_positions)
            or any(type(item) is not bool for item in self.legacy_flip)
            or any(type(item) is not bool for item in self.legacy_bound_from_lower)
            or any(type(item) is not bool for item in self.ranged_mask)
            or any(type(item) is not int for item in self.reverse_positions)
        ):
            raise ConstraintProgramError("constraint payload dimensions are inconsistent")
        return (
            self.block_kind.value,
            tuple((kind, raw, id(namespace)) for kind, raw, namespace in self.frame_cont_keys),
            tuple((kind, raw, id(namespace)) for kind, raw, namespace in self.frame_bin_keys),
            self.A_cont.key(),
            self.A_bin.key(),
            self.lower.key(),
            self.upper.key(),
            self.virtual_rows,
            self.virtual_nnz,
            self.legacy_positions,
            self.legacy_flip,
            self.legacy_bound_from_lower,
            self.ranged_mask,
            self.reverse_positions,
        )

    @property
    def source_rows(self) -> int:
        return self.A_cont.rows

    @property
    def source_nnz(self) -> int:
        return self.A_cont.nnz + self.A_bin.nnz

    @property
    def payload_bytes(self) -> int:
        return (
            self.A_cont.payload_bytes
            + self.A_bin.payload_bytes
            + self.lower.payload_bytes
            + self.upper.payload_bytes
        )


def _digest_key(value: Any) -> str:
    """Return a bucket selector; never use this result as identity authority."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if type(item) is tuple:
            digest.update(b"(")
            for child in item:
                visit(child)
            digest.update(b")")
        elif type(item) is bytes:
            digest.update(b"b")
            digest.update(len(item).to_bytes(8, "little"))
            digest.update(item)
        elif type(item) is str:
            raw = item.encode("utf-8")
            digest.update(b"s")
            digest.update(len(raw).to_bytes(8, "little"))
            digest.update(raw)
        elif type(item) is bool:
            digest.update(b"1" if item else b"0")
        elif type(item) is int:
            digest.update(b"i")
            digest.update(item.to_bytes(16, "little", signed=True))
        elif item is None:
            digest.update(b"n")
        else:
            raise TypeError(f"unsupported canonical key item {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _build_le_payload(
    frame_key: Tuple[Any, ...], A_cont: Any, A_bin: Any, upper: Any
) -> _Payload:
    fc = _FrozenCSR.from_value(A_cont, name="A_cont")
    fb = _FrozenCSR.from_value(A_bin, name="A_bin")
    ub = _FrozenVector.from_value(upper, name="upper", finite=True)
    rows = fc.rows
    if (
        rows <= 0
        or fb.rows != rows
        or ub.length != rows
        or fc.columns != len(frame_key[0])
        or fb.columns != len(frame_key[1])
    ):
        raise ConstraintProgramError("LE block shape does not match its factor frame")
    lower = np.full(rows, -np.inf, dtype=np.float64)
    return _Payload(
        _BlockKind.LE,
        frame_key[0],
        frame_key[1],
        fc,
        fb,
        _FrozenVector.from_value(lower, name="LE lower", finite=False, lower_bound=True),
        ub,
        rows,
        fc.nnz + fb.nnz,
        tuple(range(rows)),
        (False,) * rows,
        (False,) * rows,
        (),
        (),
    )


def _build_guarded_band_payload(
    frame_key: Tuple[Any, ...],
    forward_cont: Any,
    forward_bin: Any,
    forward_upper: Any,
    reverse_cont: Any,
    reverse_bin: Any,
    reverse_upper: Any,
) -> _Payload:
    fc = _FrozenCSR.from_value(forward_cont, name="forward_cont")
    fb = _FrozenCSR.from_value(forward_bin, name="forward_bin")
    rc = _FrozenCSR.from_value(reverse_cont, name="reverse_cont")
    rb = _FrozenCSR.from_value(reverse_bin, name="reverse_bin")
    fu = _FrozenVector.from_value(forward_upper, name="forward_upper", finite=True)
    ru = _FrozenVector.from_value(reverse_upper, name="reverse_upper", finite=True)
    rows = fc.rows
    if (
        rows <= 0
        or fb.rows != rows
        or rc.rows != rows
        or rb.rows != rows
        or fu.length != rows
        or ru.length != rows
        or fc.columns != len(frame_key[0])
        or rc.columns != len(frame_key[0])
        or fb.columns != len(frame_key[1])
        or rb.columns != len(frame_key[1])
    ):
        raise ConstraintProgramError("guarded band shape does not match its factor frame")
    paired_cont = _csr_row_bits_equal_negative(fc, rc)
    paired_bin = _csr_row_bits_equal_negative(fb, rb)
    ranged = tuple(a and b for a, b in zip(paired_cont, paired_bin))
    fallback = tuple(index for index, value in enumerate(ranged) if not value)
    if fallback:
        source_cont = _stack_frozen(fc, _frozen_rows(rc, fallback))
        source_bin = _stack_frozen(fb, _frozen_rows(rb, fallback))
    else:
        source_cont, source_bin = fc, fb
    fu_bits = fu.array().view(np.uint64)
    ru_bits = ru.array().view(np.uint64)
    neg_inf_bits = int(np.asarray([-np.inf], dtype=np.float64).view(np.uint64)[0])
    lower_bits = [
        int(ru_bits[row]) ^ (1 << 63) if ranged[row] else neg_inf_bits
        for row in range(rows)
    ] + [neg_inf_bits] * len(fallback)
    upper_bits = [int(item) for item in fu_bits] + [int(ru_bits[row]) for row in fallback]
    reverse_positions = [-1] * rows
    for offset, row in enumerate(fallback):
        reverse_positions[row] = rows + offset
    legacy_positions = tuple(range(rows)) + tuple(
        row if ranged[row] else reverse_positions[row] for row in range(rows)
    )
    legacy_flip = (False,) * rows + tuple(bool(ranged[row]) for row in range(rows))
    legacy_from_lower = (False,) * rows + tuple(bool(ranged[row]) for row in range(rows))
    return _Payload(
        _BlockKind.GUARDED_BAND,
        frame_key[0],
        frame_key[1],
        source_cont,
        source_bin,
        _FrozenVector.from_bits(lower_bits, name="guarded-band lower"),
        _FrozenVector.from_bits(upper_bits, name="guarded-band upper"),
        2 * rows,
        fc.nnz + fb.nnz + rc.nnz + rb.nnz,
        legacy_positions,
        legacy_flip,
        legacy_from_lower,
        ranged,
        tuple(reverse_positions),
    )


class ConstraintView:
    __slots__ = ("_block_ids", "_view_id", "_arena_token", "_sealed")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ConstraintView":
        if cls is not ConstraintView or _kwargs.pop("_token", None) is not _FACTORY_TOKEN:
            raise TypeError("ConstraintView cannot be self-signed")
        if _args or _kwargs:
            raise TypeError("malformed internal ConstraintView construction")
        return object.__new__(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ConstraintView is immutable")
        object.__setattr__(self, name, value)

    @property
    def block_ids(self) -> Tuple[ConstraintObjectID, ...]:
        return self._block_ids

    @property
    def view_id(self) -> ConstraintObjectID:
        return self._view_id


class PreparedAppend:
    """Opaque one-use capability for an arena-private staged append."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "PreparedAppend":
        if cls is not PreparedAppend or _kwargs.pop("_token", None) is not _FACTORY_TOKEN:
            raise TypeError("PreparedAppend cannot be self-signed")
        if _args or _kwargs:
            raise TypeError("malformed internal PreparedAppend construction")
        return object.__new__(cls)


@dataclass(frozen=True)
class ConstraintAppend:
    """Convenience metadata handle; the sealed program remains authority."""

    view: ConstraintView
    block_id: ConstraintObjectID
    source_row_ids: Tuple[ConstraintObjectID, ...]
    legacy_facet_ids: Tuple[ConstraintObjectID, ...]
    source_rows: int
    virtual_rows: int
    source_nnz: int
    virtual_nnz: int
    ranged_rows: int
    fallback_pairs: int
    append_ordinal: int

    @property
    def representation_authority(self) -> bool:
        # This public dataclass is intentionally easy to inspect and therefore
        # never self-authenticating.  Its ``view`` is validated independently
        # by the arena; only a sealed ConstraintProgram/batch is authority.
        return False

    @property
    def replay_authority(self) -> bool:
        return False

    @property
    def proof_authority(self) -> bool:
        return False


@dataclass(frozen=True)
class _Occurrence:
    block_key: Tuple[str, int]
    ordinal: int
    payload: _Payload
    payload_key: Tuple[Any, ...]
    payload_digest: str
    source_row_keys: Tuple[Tuple[str, int], ...]
    source_tags: Tuple[str, ...]
    legacy_row_keys: Tuple[Tuple[str, int], ...]
    legacy_tags: Tuple[str, ...]
    family: Optional[ConstraintFamily]
    layer_id: int

    def key(self) -> Tuple[Any, ...]:
        if (
            type(self.payload) is not _Payload
            or type(self.payload_key) is not tuple
            or self.payload.key() != self.payload_key
            or type(self.payload_digest) is not str
            or not self.payload_digest
        ):
            raise ConstraintProgramError("occurrence payload was mutated after staging")
        return (
            self.block_key,
            self.ordinal,
            self.payload_key,
            self.source_row_keys,
            self.source_tags,
            self.legacy_row_keys,
            self.legacy_tags,
            None if self.family is None else self.family.value,
            self.layer_id,
        )


def _payload_graph(payload: _Payload) -> Tuple[Any, ...]:
    payload.key()
    return (
        payload,
        payload.block_kind,
        payload.frame_cont_keys,
        payload.frame_bin_keys,
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
        payload.legacy_positions,
        payload.legacy_flip,
        payload.legacy_bound_from_lower,
        payload.ranged_mask,
        payload.reverse_positions,
        *payload.frame_cont_keys,
        *payload.frame_bin_keys,
    )


def _occurrence_graph(occurrence: _Occurrence) -> Tuple[Any, ...]:
    occurrence.key()
    return (
        occurrence,
        occurrence.payload,
        occurrence.payload_key,
        occurrence.payload_digest,
        occurrence.source_row_keys,
        occurrence.source_tags,
        occurrence.legacy_row_keys,
        occurrence.legacy_tags,
        occurrence.family,
        *_payload_graph(occurrence.payload),
    )


@dataclass
class _PreparedState:
    reference: Any
    object_id: int
    base_view: ConstraintView
    base_ids: Tuple[Tuple[str, int], ...]
    frame: FactorFrame
    frame_key: Tuple[Any, ...]
    occurrence: _Occurrence
    occurrence_graph: Tuple[Any, ...]
    staged_view: ConstraintView
    staged_view_truth: Tuple[Any, ...]
    staged_view_graph: Tuple[Any, ...]
    result: ConstraintAppend
    sequence: int
    status: str


@dataclass(frozen=True)
class _ArenaData:
    blocks: Dict[Tuple[str, int], _Occurrence]
    views: Dict[int, Tuple[Any, Tuple[Any, ...], Tuple[Any, ...]]]
    view_buckets: Dict[str, list[ConstraintView]]
    payload_buckets: Dict[
        str, list[Tuple[_Payload, Tuple[Any, ...], Tuple[Any, ...]]]
    ]
    prepared: Dict[int, _PreparedState]
    pending: list[int]
    next_sequence: int
    sealed: bool
    discarded: bool = False


class _ArenaState:
    """Stable arena anchor with one copy-on-write mutable-data root."""

    def __init__(
        self,
        reference: Any,
        arena_id: int,
        owner: ConstraintProgramOwner,
        owner_state: _OwnerState,
        token: _ArenaToken,
        blocks: Dict[Tuple[str, int], _Occurrence],
        views: Dict[int, Tuple[Any, Tuple[Any, ...], Tuple[Any, ...]]],
        view_buckets: Dict[str, list[ConstraintView]],
        payload_buckets: Dict[
            str, list[Tuple[_Payload, Tuple[Any, ...], Tuple[Any, ...]]]
        ],
        prepared: Dict[int, _PreparedState],
        pending: list[int],
        next_sequence: int,
        sealed: bool,
        empty_view: ConstraintView,
    ) -> None:
        self.reference = reference
        self.arena_id = arena_id
        self.owner = owner
        self.owner_state = owner_state
        self.token = token
        self.empty_view = empty_view
        self._data = _ArenaData(
            blocks,
            views,
            view_buckets,
            payload_buckets,
            prepared,
            pending,
            next_sequence,
            sealed,
        )

    def _get(self, name: str) -> Any:
        return getattr(self._data, name)

    def _set(self, name: str, value: Any) -> None:
        if getattr(self._data, name) is value:
            return
        self._data = replace(self._data, **{name: value})

    blocks = property(lambda self: self._get("blocks"), lambda self, value: self._set("blocks", value))
    views = property(lambda self: self._get("views"), lambda self, value: self._set("views", value))
    view_buckets = property(lambda self: self._get("view_buckets"), lambda self, value: self._set("view_buckets", value))
    payload_buckets = property(lambda self: self._get("payload_buckets"), lambda self, value: self._set("payload_buckets", value))
    prepared = property(lambda self: self._get("prepared"), lambda self, value: self._set("prepared", value))
    pending = property(lambda self: self._get("pending"), lambda self, value: self._set("pending", value))
    next_sequence = property(lambda self: self._get("next_sequence"), lambda self, value: self._set("next_sequence", value))
    sealed = property(lambda self: self._get("sealed"), lambda self, value: self._set("sealed", value))
    discarded = property(lambda self: self._get("discarded"), lambda self, value: self._set("discarded", value))


def _arena_anchor_graph(state: _ArenaState) -> Tuple[Any, ...]:
    if (
        type(state) is not _ArenaState
        or type(state.arena_id) is not int
        or type(state.owner) is not ConstraintProgramOwner
        or type(state.owner_state) is not _OwnerState
        or type(state.token) is not _ArenaToken
        or type(state.empty_view) is not ConstraintView
    ):
        raise ConstraintProgramError("arena registry anchor was substituted")
    return (
        state,
        state.reference,
        state.arena_id,
        state.owner,
        state.owner_state,
        state.token,
        state.empty_view,
    )


@dataclass(frozen=True)
class _ArenaRegistryEntry:
    reference: Any
    state: _ArenaState
    anchor_graph: Tuple[Any, ...]
    committed_key: Tuple[Any, ...]
    committed_graph: Tuple[Any, ...]


def _prepared_key_graph(
    transaction: _PreparedState,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    if (
        type(transaction) is not _PreparedState
        or not isinstance(transaction.reference, weakref.ReferenceType)
        or type(transaction.object_id) is not int
        or transaction.object_id < 0
        or type(transaction.base_view) is not ConstraintView
        or type(transaction.base_ids) is not tuple
        or type(transaction.frame) is not FactorFrame
        or type(transaction.frame_key) is not tuple
        or type(transaction.occurrence) is not _Occurrence
        or type(transaction.occurrence_graph) is not tuple
        or type(transaction.staged_view) is not ConstraintView
        or type(transaction.staged_view_truth) is not tuple
        or type(transaction.staged_view_graph) is not tuple
        or type(transaction.result) is not ConstraintAppend
        or type(transaction.sequence) is not int
        or transaction.sequence < 0
        or transaction.status != "prepared"
    ):
        raise ConstraintTransactionError("prepared transaction graph is malformed")
    live = transaction.reference()
    if live is not None and (
        type(live) is not PreparedAppend or id(live) != transaction.object_id
    ):
        raise ConstraintTransactionError("prepared weak capability was rebound")
    base_truth = _view_key(transaction.base_view)
    base_graph = _view_graph(transaction.base_view)
    frame_truth = _frame_key(transaction.frame)
    occurrence_truth = transaction.occurrence.key()
    occurrence_graph = _occurrence_graph(transaction.occurrence)
    staged_truth = _view_key(transaction.staged_view)
    staged_graph = _view_graph(transaction.staged_view)
    if (
        base_truth[0] != transaction.base_ids
        or frame_truth != transaction.frame_key
        or transaction.sequence != transaction.occurrence.ordinal
        or transaction.result.view is not transaction.staged_view
        or transaction.result.append_ordinal != transaction.sequence
        or staged_truth != transaction.staged_view_truth
        or len(base_graph) == 0
        or len(occurrence_graph) != len(transaction.occurrence_graph)
        or any(
            current is not sealed
            for current, sealed in zip(
                occurrence_graph, transaction.occurrence_graph
            )
        )
        or len(staged_graph) != len(transaction.staged_view_graph)
        or any(
            current is not sealed
            for current, sealed in zip(
                staged_graph, transaction.staged_view_graph
            )
        )
    ):
        raise ConstraintTransactionError("prepared transaction graph was rebound")
    key = (
        transaction.object_id,
        transaction.base_ids,
        transaction.frame_key,
        occurrence_truth,
        transaction.staged_view_truth,
        transaction.sequence,
        transaction.status,
    )
    graph = (
        transaction,
        transaction.reference,
        transaction.base_view,
        transaction.base_ids,
        transaction.frame,
        transaction.frame_key,
        transaction.occurrence,
        transaction.occurrence_graph,
        *transaction.occurrence_graph,
        transaction.staged_view,
        transaction.staged_view_truth,
        transaction.staged_view_graph,
        *transaction.staged_view_graph,
        transaction.result,
        *base_graph,
    )
    return key, graph


def _arena_committed_state(
    state: _ArenaState,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    if (
        type(state._data) is not _ArenaData
        or type(state.blocks) is not dict
        or type(state.views) is not dict
        or type(state.view_buckets) is not dict
        or type(state.payload_buckets) is not dict
        or type(state.prepared) is not dict
        or type(state.pending) is not list
        or type(state.next_sequence) is not int
        or state.next_sequence < 0
        or type(state.sealed) is not bool
        or type(state.discarded) is not bool
        or (state.sealed and state.discarded)
    ):
        raise ConstraintProgramError("arena committed registry state is malformed")
    items = []
    graph = [state._data, state.blocks]
    ordinals = set()
    for block_key, occurrence in sorted(
        state.blocks.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        if (
            type(block_key) is not tuple
            or len(block_key) != 2
            or block_key[0] != "block"
            or type(block_key[1]) is not int
            or type(occurrence) is not _Occurrence
            or occurrence.block_key != block_key
            or occurrence.ordinal in ordinals
            or not 0 <= occurrence.ordinal < state.next_sequence
        ):
            raise ConstraintProgramError(
                "arena committed block key/order is malformed"
            )
        ordinals.add(occurrence.ordinal)
        occurrence_key = occurrence.key()
        occurrence_graph = _occurrence_graph(occurrence)
        items.append((block_key, occurrence_key))
        # ``occurrence_key`` and ``occurrence_graph`` are freshly-created
        # wrapper tuples on every validation.  Their values belong in the
        # equality key above, but only their stable members may participate in
        # the identity graph.
        graph.extend((occurrence, *occurrence_graph))
    view_items = []
    graph.extend((state.views, state.view_buckets, state.payload_buckets))
    for object_id, record in sorted(state.views.items()):
        if (
            type(object_id) is not int
            or type(record) is not tuple
            or len(record) != 3
            or type(record[0]) is not ConstraintView
            or id(record[0]) != object_id
            or type(record[1]) is not tuple
            or type(record[2]) is not tuple
        ):
            raise ConstraintProgramError("arena view registry is malformed")
        truth = _view_key(record[0])
        current_graph = _view_graph(record[0])
        if (
            truth != record[1]
            or truth[2] is not state.token
            or any(key not in state.blocks for key in truth[0])
            or len(current_graph) != len(record[2])
            or any(
                current is not sealed
                for current, sealed in zip(current_graph, record[2])
            )
        ):
            raise ConstraintProgramError("arena view registry graph was rebound")
        view_items.append((object_id, truth))
        graph.extend((record, record[0], record[1], record[2], *record[2]))

    bucket_view_ids = []
    seen_views = set()
    for digest, bucket in sorted(state.view_buckets.items()):
        if type(digest) is not str or not digest or type(bucket) is not list:
            raise ConstraintProgramError("arena view bucket is malformed")
        current_ids = []
        for item in bucket:
            record = state.views.get(id(item))
            if (
                type(item) is not ConstraintView
                or record is None
                or record[0] is not item
                or id(item) in seen_views
            ):
                raise ConstraintProgramError("arena view bucket graph was rebound")
            seen_views.add(id(item))
            current_ids.append(id(item))
            graph.append(item)
        bucket_view_ids.append((digest, tuple(current_ids)))
        graph.append(bucket)
    if seen_views != set(state.views):
        raise ConstraintProgramError("arena view registry and buckets disagree")

    payload_items = []
    for digest, bucket in sorted(state.payload_buckets.items()):
        if type(digest) is not str or not digest or type(bucket) is not list:
            raise ConstraintProgramError("arena payload bucket is malformed")
        keys = []
        for record in bucket:
            if (
                type(record) is not tuple
                or len(record) != 3
                or type(record[0]) is not _Payload
                or type(record[1]) is not tuple
                or type(record[2]) is not tuple
            ):
                raise ConstraintProgramError("arena payload record is malformed")
            current_key = record[0].key()
            current_graph = _payload_graph(record[0])
            if (
                current_key != record[1]
                or len(current_graph) != len(record[2])
                or any(
                    current is not sealed
                    for current, sealed in zip(current_graph, record[2])
                )
            ):
                raise ConstraintProgramError("arena payload graph was rebound")
            keys.append(current_key)
            graph.extend((record, record[0], record[1], record[2], *record[2]))
        if len(keys) != len(set(keys)):
            raise ConstraintProgramError("arena payload bucket contains duplicates")
        payload_items.append((digest, tuple(keys)))
        graph.append(bucket)

    prepared_items = []
    prepared_graph = [state.prepared, state.pending]
    sequences = set(ordinals)
    for object_id, transaction in sorted(
        state.prepared.items(), key=lambda item: item[1].sequence
    ):
        if object_id != transaction.object_id or transaction.sequence in sequences:
            raise ConstraintTransactionError(
                "arena prepared sequence/key is malformed"
            )
        sequences.add(transaction.sequence)
        transaction_key, transaction_graph = _prepared_key_graph(transaction)
        prepared_items.append((object_id, transaction_key))
        prepared_graph.extend(transaction_graph)
    if (
        any(type(item) is not int for item in state.pending)
        or len(state.pending) != len(set(state.pending))
        or tuple(state.pending)
        != tuple(
            object_id
            for object_id, _transaction in sorted(
                state.prepared.items(), key=lambda item: item[1].sequence
            )
        )
        or any(sequence >= state.next_sequence for sequence in sequences)
        or ((state.sealed or state.discarded) and (state.pending or state.prepared))
    ):
        raise ConstraintTransactionError(
            "arena pending/prepared order is not a bijection"
        )
    if state.discarded and (
        state.blocks
        or set(state.views) != {id(state.empty_view)}
        or state.payload_buckets
        or len(state.view_buckets) != 1
        or seen_views != {id(state.empty_view)}
    ):
        raise ConstraintProgramError(
            "discarded arena retained committed replay state"
        )
    graph.extend(prepared_graph)
    return (
        (
            tuple(items),
            tuple(view_items),
            tuple(bucket_view_ids),
            tuple(payload_items),
            tuple(prepared_items),
            tuple(state.pending),
            state.next_sequence,
            state.sealed,
            state.discarded,
        ),
        tuple(graph),
    )


def _publish_arena_state(state: _ArenaState) -> None:
    committed_key, committed_graph = _arena_committed_state(state)
    with _ARENA_LOCK:
        entry = _ARENA_REGISTRY.get(state.arena_id)
        if entry is None or entry.state is not state or entry.reference is not state.reference:
            raise ConstraintProgramError("arena registry publication lost its arena")
        current_anchor = _arena_anchor_graph(state)
        if (
            len(current_anchor) != len(entry.anchor_graph)
            or any(
                current is not sealed
                for current, sealed in zip(current_anchor, entry.anchor_graph)
            )
        ):
            raise ConstraintProgramError("arena anchor changed during publication")
        _ARENA_REGISTRY[state.arena_id] = _ArenaRegistryEntry(
            entry.reference,
            state,
            entry.anchor_graph,
            committed_key,
            committed_graph,
        )


_ARENA_LOCK = threading.Lock()
_ARENA_REGISTRY: Dict[int, _ArenaRegistryEntry] = {}


def _begin_arena_work(state: _ArenaState) -> Tuple[_ArenaData, _ArenaData]:
    """Build a private arena root without publishing or installing it."""

    old = state._data
    prepared = {
        object_id: replace(transaction)
        for object_id, transaction in old.prepared.items()
    }
    working = _ArenaData(
        dict(old.blocks),
        dict(old.views),
        {key: list(bucket) for key, bucket in old.view_buckets.items()},
        {key: list(bucket) for key, bucket in old.payload_buckets.items()},
        prepared,
        list(old.pending),
        old.next_sequence,
        old.sealed,
        old.discarded,
    )
    return old, working


def _activate_arena_work(
    state: _ArenaState, old: _ArenaData, working: _ArenaData
) -> None:
    if state._data is not old or type(working) is not _ArenaData:
        raise ConstraintProgramError("arena changed before work-root activation")
    state._data = working


def _restore_arena_work(
    state: _ArenaState,
    old: _ArenaData,
    *,
    preserve_sequence: bool = False,
    consume: Tuple[int, ...] = (),
) -> None:
    next_sequence = (
        max(old.next_sequence, state.next_sequence)
        if preserve_sequence
        else old.next_sequence
    )
    if consume:
        prepared = {
            object_id: replace(transaction)
            for object_id, transaction in old.prepared.items()
        }
        pending = list(old.pending)
        for object_id in consume:
            transaction = prepared.pop(object_id, None)
            if transaction is not None:
                transaction.status = "aborted"
            pending = [item for item in pending if item != object_id]
    else:
        prepared = old.prepared
        pending = old.pending
    target = _ArenaData(
        old.blocks,
        old.views,
        old.view_buckets,
        old.payload_buckets,
        prepared,
        pending,
        next_sequence,
        old.sealed,
        old.discarded,
    )
    repair_error: Optional[BaseException] = None
    for _attempt in range(4):
        try:
            state._data = target
            _publish_arena_state(state)
            return
        except BaseException as error:
            if repair_error is None:
                repair_error = error
            continue
    if repair_error is not None:
        raise repair_error
    raise ConstraintProgramError("arena rollback could not publish its root")


def _commit_arena_work(state: _ArenaState) -> None:
    _publish_arena_state(state)


@dataclass(frozen=True)
class _SealedArenaRecord:
    arena_reference: Any
    state: _ArenaState
    view: ConstraintView
    view_truth: Tuple[Any, ...]
    view_graph: Tuple[Any, ...]
    frame: FactorFrame
    frame_key: Tuple[Any, ...]
    frame_graph: Tuple[Any, ...]
    program: Any
    program_state: Any
    graph: Tuple[Any, ...]


_SEALED_ARENA_LOCK = threading.Lock()
_SEALED_ARENA_REGISTRY: Dict[int, _SealedArenaRecord] = {}


class ConstraintArena:
    """Thread-confined transactional source arena."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ConstraintArena":
        if cls is not ConstraintArena or _kwargs.pop("_token", None) is not _FACTORY_TOKEN:
            raise TypeError("ConstraintArena must be created by its owner")
        if _args or _kwargs:
            raise TypeError("malformed internal ConstraintArena construction")
        return object.__new__(cls)

    @property
    def empty_view(self) -> ConstraintView:
        return _require_arena_open(self).empty_view

    def union(self, *views: ConstraintView) -> ConstraintView:
        state = _require_arena_open(self)
        operation_epoch = _begin_owner_operation(state.owner_state)
        old_arena_data: Optional[_ArenaData] = None
        try:
            _activate_owner_operation(state.owner_state, operation_epoch)
            old_arena_data, working_arena_data = _begin_arena_work(state)
            _activate_arena_work(state, old_arena_data, working_arena_data)
            if type(views) is not tuple or not views:
                raise ConstraintProgramError("union requires at least one exact view")
            keys = []
            for view in views:
                keys.extend(_validate_view(state, view))
            normalized = tuple(
                sorted(set(keys), key=lambda item: (item[1], item[0]))
            )
            result = _intern_live_view(state, normalized)
            _assert_owner_operation(state.owner_state, operation_epoch)
            _commit_arena_work(state)
            _commit_owner_operation(state.owner_state, operation_epoch)
            return result
        except BaseException:
            if old_arena_data is not None:
                _restore_arena_work(state, old_arena_data)
            raise
        finally:
            _end_owner_operation(state.owner_state, operation_epoch)

    def prepare_le(
        self,
        view: ConstraintView,
        *,
        frame: FactorFrame,
        A_cont: sp.csr_matrix,
        A_bin: sp.csr_matrix,
        upper: np.ndarray,
        tag: str,
        layer_id: int = 0,
    ) -> PreparedAppend:
        return self._prepare_le_with_tag(
            view,
            frame=frame,
            A_cont=A_cont,
            A_bin=A_bin,
            upper=upper,
            tag=tag,
            layer_id=layer_id,
            complete_tag=False,
        )

    def prepare_le_exact_tag(
        self,
        view: ConstraintView,
        *,
        frame: FactorFrame,
        A_cont: sp.csr_matrix,
        A_bin: sp.csr_matrix,
        upper: np.ndarray,
        tag: str,
        layer_id: int = 0,
    ) -> PreparedAppend:
        """Stage ordinary LE rows under one caller-complete exact tag.

        Unlike :meth:`prepare_le`, this entry point never appends a layer
        suffix.  The exact builtin string is frozen into every native and
        legacy row record and therefore into the sealed program digest.
        """

        return self._prepare_le_with_tag(
            view,
            frame=frame,
            A_cont=A_cont,
            A_bin=A_bin,
            upper=upper,
            tag=tag,
            layer_id=layer_id,
            complete_tag=True,
        )

    def _prepare_le_with_tag(
        self,
        view: ConstraintView,
        *,
        frame: FactorFrame,
        A_cont: sp.csr_matrix,
        A_bin: sp.csr_matrix,
        upper: np.ndarray,
        tag: str,
        layer_id: int,
        complete_tag: bool,
    ) -> PreparedAppend:
        # Validate caller text before owner/arena activation, occurrence-ID
        # reservation, or append-sequence publication.  This shared entry
        # covers both the legacy suffixing API and the exact-tag API.
        tag = _exact_text(tag, name="LE tag")
        state = _require_arena_open(self)
        operation_epoch = _begin_owner_operation(state.owner_state)
        old_arena_data: Optional[_ArenaData] = None
        try:
            _activate_owner_operation(state.owner_state, operation_epoch)
            old_arena_data, working_arena_data = _begin_arena_work(state)
            _activate_arena_work(state, old_arena_data, working_arena_data)
            base_ids = _validate_view(state, view)
            frame_key = _validate_frame_state(state.owner_state, frame)
            if type(complete_tag) is not bool:
                raise ConstraintProgramError(
                    "complete-tag mode must be an exact builtin bool"
                )
            layer_id = _exact_count(layer_id, name="layer_id")
            payload = _build_le_payload(frame_key, A_cont, A_bin, upper)
            frozen_tag = tag if complete_tag else f"{tag}:{layer_id}"
            source_tags = (frozen_tag,) * payload.source_rows
            legacy_tags = source_tags
            result = _stage_append(
                state,
                view,
                base_ids,
                frame,
                frame_key,
                payload,
                source_tags,
                legacy_tags,
                family=None,
                layer_id=layer_id,
            )
            _assert_owner_operation(state.owner_state, operation_epoch)
            _commit_arena_work(state)
            _commit_owner_operation(state.owner_state, operation_epoch)
            return result
        except BaseException:
            transaction = (
                state.prepared.get(id(result))
                if "result" in locals()
                else None
            )
            if transaction is not None:
                transaction.status = "aborted"
            if old_arena_data is not None:
                _restore_arena_work(
                    state,
                    old_arena_data,
                    preserve_sequence=True,
                )
            raise
        finally:
            _end_owner_operation(state.owner_state, operation_epoch)

    def prepare_guarded_band(
        self,
        view: ConstraintView,
        *,
        frame: FactorFrame,
        forward_cont: sp.csr_matrix,
        forward_bin: sp.csr_matrix,
        forward_upper: np.ndarray,
        reverse_cont: sp.csr_matrix,
        reverse_bin: sp.csr_matrix,
        reverse_upper: np.ndarray,
        layer_id: int,
        family: ConstraintFamily,
    ) -> PreparedAppend:
        state = _require_arena_open(self)
        operation_epoch = _begin_owner_operation(state.owner_state)
        old_arena_data: Optional[_ArenaData] = None
        try:
            _activate_owner_operation(state.owner_state, operation_epoch)
            old_arena_data, working_arena_data = _begin_arena_work(state)
            _activate_arena_work(state, old_arena_data, working_arena_data)
            base_ids = _validate_view(state, view)
            frame_key = _validate_frame_state(state.owner_state, frame)
            layer_id = _exact_count(layer_id, name="layer_id")
            if (
                type(family) is not ConstraintFamily
                or family is not ConstraintFamily.ADD_MATERIALIZE
            ):
                raise ConstraintProgramError(
                    "guarded RANGE accepts only exact ConstraintFamily.ADD_MATERIALIZE"
                )
            payload = _build_guarded_band_payload(
                frame_key,
                forward_cont,
                forward_bin,
                forward_upper,
                reverse_cont,
                reverse_bin,
                reverse_upper,
            )
            original_rows = payload.virtual_rows // 2
            base_tag = f"{family.value}:{layer_id}"
            source_tags = tuple(
                f"range:{base_tag}"
                if payload.ranged_mask[row]
                else f"{base_tag}:forward"
                for row in range(original_rows)
            ) + tuple(
                f"{base_tag}:reverse"
                for row, paired in enumerate(payload.ranged_mask)
                if not paired
            )
            legacy_tags = (f"{base_tag}:forward",) * original_rows + (
                f"{base_tag}:reverse",
            ) * original_rows
            result = _stage_append(
                state,
                view,
                base_ids,
                frame,
                frame_key,
                payload,
                source_tags,
                legacy_tags,
                family=family,
                layer_id=layer_id,
            )
            _assert_owner_operation(state.owner_state, operation_epoch)
            _commit_arena_work(state)
            _commit_owner_operation(state.owner_state, operation_epoch)
            return result
        except BaseException:
            transaction = (
                state.prepared.get(id(result))
                if "result" in locals()
                else None
            )
            if transaction is not None:
                transaction.status = "aborted"
            if old_arena_data is not None:
                _restore_arena_work(
                    state,
                    old_arena_data,
                    preserve_sequence=True,
                )
            raise
        finally:
            _end_owner_operation(state.owner_state, operation_epoch)

    def commit(self, prepared: PreparedAppend) -> ConstraintAppend:
        state = _require_arena_open(self)
        operation_epoch = _begin_owner_operation(state.owner_state)
        old_arena_data: Optional[_ArenaData] = None
        transaction: Optional[_PreparedState] = None
        publication_ready = False
        owner_swap_started = False
        try:
            _activate_owner_operation(state.owner_state, operation_epoch)
            old_arena_data, working_arena_data = _begin_arena_work(state)
            _activate_arena_work(state, old_arena_data, working_arena_data)
            try:
                transaction = _validate_prepared(state, prepared)
                result = _commit_validated(state, prepared, transaction)
                _assert_owner_operation(state.owner_state, operation_epoch)
                publication_ready = True
                _commit_arena_work(state)
                owner_swap_started = True
                _commit_owner_operation(state.owner_state, operation_epoch)
                return result
            except BaseException:
                if old_arena_data is None:
                    pass
                elif owner_swap_started:
                    _restore_arena_work(
                        state,
                        old_arena_data,
                        consume=(id(prepared),),
                    )
                elif publication_ready:
                    try:
                        _commit_arena_work(state)
                    except BaseException:
                        _restore_arena_work(
                            state,
                            old_arena_data,
                            consume=(id(prepared),),
                        )
                else:
                    _restore_arena_work(
                        state,
                        old_arena_data,
                        consume=(id(prepared),),
                    )
                raise
        finally:
            _end_owner_operation(state.owner_state, operation_epoch)

    def abort(self, prepared: PreparedAppend) -> None:
        state = _require_arena_open(self)
        operation_epoch = _begin_owner_operation(state.owner_state)
        old_arena_data: Optional[_ArenaData] = None
        publication_ready = False
        owner_swap_started = False
        try:
            _activate_owner_operation(state.owner_state, operation_epoch)
            old_arena_data, working_arena_data = _begin_arena_work(state)
            _activate_arena_work(state, old_arena_data, working_arena_data)
            try:
                transaction = _validate_prepared(state, prepared)
                _abort_transaction(state, transaction)
                _assert_owner_operation(state.owner_state, operation_epoch)
                publication_ready = True
                _commit_arena_work(state)
                owner_swap_started = True
                _commit_owner_operation(state.owner_state, operation_epoch)
            except BaseException:
                if old_arena_data is None:
                    pass
                elif owner_swap_started:
                    _restore_arena_work(
                        state,
                        old_arena_data,
                        consume=(id(prepared),),
                    )
                elif publication_ready:
                    try:
                        _commit_arena_work(state)
                    except BaseException:
                        _restore_arena_work(
                            state,
                            old_arena_data,
                            consume=(id(prepared),),
                        )
                else:
                    _restore_arena_work(
                        state,
                        old_arena_data,
                        consume=(id(prepared),),
                    )
                raise
        finally:
            _end_owner_operation(state.owner_state, operation_epoch)

    def append_le(self, view: ConstraintView, **kwargs: Any) -> ConstraintAppend:
        prepared = self.prepare_le(view, **kwargs)
        try:
            return self.commit(prepared)
        except BaseException:
            _abort_if_live(self, prepared)
            raise

    def append_le_exact_tag(
        self, view: ConstraintView, **kwargs: Any
    ) -> ConstraintAppend:
        prepared = self.prepare_le_exact_tag(view, **kwargs)
        try:
            return self.commit(prepared)
        except BaseException:
            _abort_if_live(self, prepared)
            raise

    def append_guarded_band(self, view: ConstraintView, **kwargs: Any) -> ConstraintAppend:
        prepared = self.prepare_guarded_band(view, **kwargs)
        try:
            return self.commit(prepared)
        except BaseException:
            _abort_if_live(self, prepared)
            raise

    @property
    def discarded(self) -> bool:
        state = _arena_state(self)
        _require_owner_thread(state.owner_state)
        return state.discarded

    def discard(self) -> None:
        """Irrevocably close an unsealed build arena without a program.

        A successful discard consumes every live prepared capability and
        freezes the owner/arena pair in one non-authoritative terminal state.
        Factor IDs, row IDs, and append ordinals remain burned.  Repeating
        discard is harmless; discarding an arena that already published a
        sealed program is rejected.
        """

        state = _arena_state(self)
        _require_owner_thread(state.owner_state)
        owner_entry = _owner_operation_entry(state.owner_state)
        if owner_entry is not None and owner_entry.operation_active:
            _mark_owner_reentrancy(state.owner_state)
            raise ConstraintProgramError(
                "discard cannot reenter an active owner operation"
            )
        if state.discarded or state.owner_state.discarded:
            if (
                state.discarded
                and state.owner_state.discarded
                and not state.sealed
                and not state.owner_state.sealed
            ):
                return
            raise ConstraintProgramError(
                "owner/arena discard publication is incomplete"
            )
        if state.sealed or state.owner_state.sealed:
            if state.sealed and state.owner_state.sealed:
                raise ConstraintProgramError(
                    "a sealed constraint arena cannot be discarded"
                )
            raise ConstraintProgramError(
                "owner/arena seal publication is incomplete"
            )

        operation_epoch = _begin_owner_operation(state.owner_state)
        old_arena_data: Optional[_ArenaData] = None
        rollback_owner_data: Optional[_OwnerData] = None
        try:
            _activate_owner_operation(state.owner_state, operation_epoch)
            active = _owner_operation_entry(state.owner_state)
            if active is None or type(active.rollback_data) is not _OwnerData:
                raise ConstraintProgramError("discard rollback root was lost")
            rollback_owner_data = active.rollback_data
            old_arena_data, working_arena_data = _begin_arena_work(state)
            _activate_arena_work(state, old_arena_data, working_arena_data)
            if (
                state.sealed
                or state.owner_state.sealed
                or state.discarded
                or state.owner_state.discarded
            ):
                raise ConstraintProgramError(
                    "constraint arena changed before discard publication"
                )
            # Drop every block/payload/view except the arena's identity-bound
            # empty view, and consume all prepared capabilities.  Numeric and
            # occurrence IDs are still process-burned, while ``next_sequence``
            # preserves the arena-local append-order burn watermark.
            empty_record = state.views.get(id(state.empty_view))
            empty_truth = _view_key(state.empty_view)
            empty_graph = _view_graph(state.empty_view)
            if (
                empty_record is None
                or empty_record[0] is not state.empty_view
                or empty_record[1] != empty_truth
                or len(empty_record[2]) != len(empty_graph)
                or any(
                    current is not sealed
                    for current, sealed in zip(empty_graph, empty_record[2])
                )
            ):
                raise ConstraintProgramError(
                    "discard lost the arena's authenticated empty view"
                )
            state.blocks = {}
            state.views = {id(state.empty_view): empty_record}
            state.view_buckets = {
                _digest_key(("view", empty_truth[0])): [state.empty_view]
            }
            state.payload_buckets = {}
            state.prepared = {}
            state.pending = []
            state.discarded = True
            state.owner_state.discarded = True
            _assert_owner_operation(state.owner_state, operation_epoch)
            _complete_discard_publication(state, operation_epoch)
        except BaseException:
            # Preserve the triggering BaseException.  A bounded compensating
            # pass converges any pre/post-swap interruption to the complete
            # OLD state; if the public return itself was interrupted, the
            # already-complete NEW discarded state remains valid.
            if old_arena_data is not None:
                for _attempt in range(4):
                    try:
                        _restore_arena_work(state, old_arena_data)
                        break
                    except BaseException:
                        continue
            if rollback_owner_data is not None:
                for _attempt in range(4):
                    try:
                        _restore_owner_operation(
                            state.owner_state,
                            operation_epoch,
                            rollback_owner_data,
                        )
                        break
                    except BaseException:
                        continue
            try:
                _end_owner_operation(state.owner_state, operation_epoch)
            except BaseException:
                pass
            raise
        _end_owner_operation(state.owner_state, operation_epoch)

    def close(self) -> None:
        """Alias for :meth:`discard` for build-scope cleanup code."""

        self.discard()

    def seal(self, view: ConstraintView, *, final_frame: FactorFrame) -> "ConstraintProgram":
        state = _arena_state(self)
        _require_owner_thread(state.owner_state)
        owner_entry = _owner_operation_entry(state.owner_state)
        if owner_entry is not None and owner_entry.operation_active:
            _mark_owner_reentrancy(state.owner_state)
            raise ConstraintProgramError(
                "seal cannot reenter an active owner operation"
            )
        if state.discarded or state.owner_state.discarded:
            if not state.discarded or not state.owner_state.discarded:
                raise ConstraintProgramError(
                    "owner/arena discard publication is incomplete"
                )
            raise ConstraintProgramError("discarded arena cannot be sealed")
        if state.sealed or state.owner_state.sealed:
            if not state.sealed or not state.owner_state.sealed:
                raise ConstraintProgramError("owner/arena seal publication is incomplete")
            return _sealed_arena_program(self, state, view, final_frame)
        if state.owner_state.poisoned:
            raise ConstraintProgramError(
                "constraint owner is poisoned by an allocator failure"
            )
        operation_epoch = _begin_owner_operation(state.owner_state)
        old_arena_data: Optional[_ArenaData] = None
        rollback_owner_data: Optional[_OwnerData] = None
        program: Optional[ConstraintProgram] = None
        program_state: Optional[_ProgramState] = None
        try:
            _activate_owner_operation(state.owner_state, operation_epoch)
            active = _owner_operation_entry(state.owner_state)
            if active is None or type(active.rollback_data) is not _OwnerData:
                raise ConstraintProgramError("seal rollback root was lost")
            rollback_owner_data = active.rollback_data
            old_arena_data, working_arena_data = _begin_arena_work(state)
            _activate_arena_work(state, old_arena_data, working_arena_data)
            if state.pending or state.prepared:
                raise ConstraintTransactionError(
                    "all prepared appends must be resolved before seal"
                )
            selected = _validate_view(state, view)
            committed_key, committed_graph = _arena_committed_state(state)
            selected_truth = _view_key(view)
            selected_graph = _view_graph(view)
            frame_key = _validate_final_frame(
                state.owner,
                final_frame,
                owner_state=state.owner_state,
                operation_epoch=operation_epoch,
            )
            active = _owner_operation_entry(state.owner_state)
            if active is None or type(active.rollback_data) is not _OwnerData:
                raise ConstraintProgramError("seal reconciliation root was lost")
            rollback_owner_data = active.rollback_data
            _assert_owner_operation(state.owner_state, operation_epoch)
            current_key, current_graph = _arena_committed_state(state)
            if (
                _arena_state(self) is not state
                or state.sealed
                or state.owner_state.sealed
                or state.discarded
                or state.owner_state.discarded
                or state.owner_state.poisoned
                or state.pending
                or state.prepared
                or current_key != committed_key
                or len(current_graph) != len(committed_graph)
                or any(
                    current is not sealed
                    for current, sealed in zip(current_graph, committed_graph)
                )
                or _validate_view(state, view) != selected
                or _view_key(view) != selected_truth
                or len(_view_graph(view)) != len(selected_graph)
                or any(
                    current is not sealed
                    for current, sealed in zip(_view_graph(view), selected_graph)
                )
            ):
                raise ConstraintTransactionError(
                    "arena changed while validating its final factor frame"
                )
            occurrences = tuple(
                sorted(
                    (state.blocks[key] for key in selected),
                    key=lambda item: item.ordinal,
                )
            )
            for occurrence in occurrences:
                occurrence.key()
                payload = occurrence.payload
                if (
                    frame_key[0][: len(payload.frame_cont_keys)]
                    != payload.frame_cont_keys
                    or frame_key[1][: len(payload.frame_bin_keys)]
                    != payload.frame_bin_keys
                ):
                    raise ConstraintProgramError(
                        "a source block is not a prefix of the final factor frame"
                    )
            _assert_owner_operation(state.owner_state, operation_epoch)
            program, program_state = _stage_program(occurrences, frame_key)
            return _complete_seal_publication(
                self,
                state,
                view,
                final_frame,
                program,
                program_state,
                operation_epoch,
            )
        except BaseException:
            registered = False
            registration_checked = program is None or program_state is None
            if program is not None and program_state is not None:
                for _attempt in range(4):
                    try:
                        registered = _is_registered_program(
                            program, program_state
                        )
                        registration_checked = True
                        break
                    except BaseException:
                        continue
            if not registration_checked:
                # Uncertainty must never revoke an object that may already
                # have crossed the final authority registry swap.
                registered = True
            if not registered:
                if program_state is not None:
                    for _attempt in range(4):
                        try:
                            with _SEALED_ARENA_LOCK:
                                record = _SEALED_ARENA_REGISTRY.get(state.arena_id)
                                if (
                                    record is not None
                                    and record.state is state
                                    and record.program_state is program_state
                                ):
                                    _SEALED_ARENA_REGISTRY.pop(state.arena_id, None)
                        except BaseException:
                            pass
                        with _SEALED_ARENA_LOCK:
                            current_record = _SEALED_ARENA_REGISTRY.get(
                                state.arena_id
                            )
                        if (
                            current_record is None
                            or current_record.program_state is not program_state
                        ):
                            break
                if old_arena_data is not None:
                    for _attempt in range(4):
                        try:
                            _restore_arena_work(state, old_arena_data)
                            break
                        except BaseException:
                            continue
                if rollback_owner_data is not None:
                    active = _owner_operation_entry(state.owner_state)
                    poisoned = bool(
                        state.owner_state.poisoned
                        or (
                            active is not None
                            and active.operation_active
                            and active.external_touched
                        )
                    )
                    repair_error: Optional[BaseException] = None
                    for _attempt in range(4):
                        try:
                            _restore_owner_operation(
                                state.owner_state,
                                operation_epoch,
                                rollback_owner_data,
                                poisoned=poisoned,
                            )
                            repair_error = None
                            break
                        except BaseException as error:
                            if repair_error is None:
                                repair_error = error
                            continue
                    if repair_error is not None:
                        raise repair_error
            raise
        finally:
            _end_owner_operation(state.owner_state, operation_epoch)

    @property
    def representation_authority(self) -> bool:
        _arena_state(self)
        return False

    @property
    def proof_authority(self) -> bool:
        return False


def _commit_validated(
    state: _ArenaState,
    prepared: PreparedAppend,
    transaction: _PreparedState,
) -> ConstraintAppend:
    if not state.pending or state.pending[0] != id(prepared):
        raise ConstraintTransactionError(
            "prepared appends must commit/abort in prepare sequence"
        )
    occurrence = transaction.occurrence
    missing = object()
    payload_bucket: Optional[
        list[Tuple[_Payload, Tuple[Any, ...], Tuple[Any, ...]]]
    ] = None
    payload_bucket_created = False
    payload_record: Optional[
        Tuple[_Payload, Tuple[Any, ...], Tuple[Any, ...]]
    ] = None
    payload_append_attempted = False
    block_previous: Any = missing
    block_write_attempted = False
    view_previous: Any = missing
    view_write_attempted = False
    view_bucket: Optional[list[ConstraintView]] = None
    view_bucket_created = False
    view_append_attempted = False
    view_bucket_key: Optional[str] = None
    try:
        payload_bucket = state.payload_buckets.get(occurrence.payload_digest)
        if payload_bucket is None:
            payload_bucket = []
            payload_bucket_created = True
            state.payload_buckets[occurrence.payload_digest] = payload_bucket
        interned_payload = occurrence.payload
        for existing, full_key, sealed_graph in payload_bucket:
            current_graph = _payload_graph(existing)
            if (
                existing.key() != full_key
                or len(current_graph) != len(sealed_graph)
                or any(
                    current is not sealed
                    for current, sealed in zip(current_graph, sealed_graph)
                )
            ):
                raise ConstraintTransactionError(
                    "interned payload authority was mutated"
                )
            if full_key == occurrence.payload_key:
                interned_payload = existing
                break
        final_occurrence = _Occurrence(
            occurrence.block_key,
            occurrence.ordinal,
            interned_payload,
            occurrence.payload_key,
            occurrence.payload_digest,
            occurrence.source_row_keys,
            occurrence.source_tags,
            occurrence.legacy_row_keys,
            occurrence.legacy_tags,
            occurrence.family,
            occurrence.layer_id,
        )
        final_occurrence.key()
        if interned_payload is occurrence.payload and not any(
            key == occurrence.payload_key
            for _payload, key, _graph in payload_bucket
        ):
            payload_record = (
                interned_payload,
                occurrence.payload_key,
                _payload_graph(interned_payload),
            )
            payload_append_attempted = True
            payload_bucket.append(payload_record)
        block_previous = state.blocks.get(occurrence.block_key, missing)
        block_write_attempted = True
        state.blocks[occurrence.block_key] = final_occurrence
        view_previous = state.views.get(id(transaction.staged_view), missing)
        view_write_attempted = True
        state.views[id(transaction.staged_view)] = (
            transaction.staged_view,
            transaction.staged_view_truth,
            transaction.staged_view_graph,
        )
        view_bucket_key = _digest_key(
            ("view", transaction.staged_view_truth[0])
        )
        view_bucket = state.view_buckets.get(view_bucket_key)
        if view_bucket is None:
            view_bucket = []
            view_bucket_created = True
            state.view_buckets[view_bucket_key] = view_bucket
        view_append_attempted = True
        view_bucket.append(transaction.staged_view)
        state.prepared.pop(id(prepared), None)
        if state.pending and state.pending[0] == id(prepared):
            state.pending.pop(0)
        transaction.status = "committed"
        return transaction.result
    except BaseException:
        if view_append_attempted and view_bucket is not None:
            for index in range(len(view_bucket) - 1, -1, -1):
                if view_bucket[index] is transaction.staged_view:
                    view_bucket.pop(index)
                    break
        if (
            view_bucket_created
            and view_bucket_key is not None
            and state.view_buckets.get(view_bucket_key) is view_bucket
            and not view_bucket
        ):
            state.view_buckets.pop(view_bucket_key, None)
        if view_write_attempted:
            if view_previous is missing:
                if state.views.get(id(transaction.staged_view), missing) is not missing:
                    state.views.pop(id(transaction.staged_view), None)
            else:
                state.views[id(transaction.staged_view)] = view_previous
        if block_write_attempted:
            if block_previous is missing:
                state.blocks.pop(occurrence.block_key, None)
            else:
                state.blocks[occurrence.block_key] = block_previous
        if payload_append_attempted and payload_bucket is not None and payload_record is not None:
            for index in range(len(payload_bucket) - 1, -1, -1):
                if payload_bucket[index] is payload_record:
                    payload_bucket.pop(index)
                    break
        if (
            payload_bucket_created
            and payload_bucket is not None
            and state.payload_buckets.get(occurrence.payload_digest) is payload_bucket
            and not payload_bucket
        ):
            state.payload_buckets.pop(occurrence.payload_digest, None)
        raise


def _stage_arena(
    owner: ConstraintProgramOwner, owner_state: _OwnerState
) -> Tuple[ConstraintArena, _ArenaState, _ArenaRegistryEntry]:
    arena = ConstraintArena(_token=_FACTORY_TOKEN)
    object_id = id(arena)

    def cleanup(reference: Any) -> None:
        for _attempt in range(4):
            try:
                with _ARENA_LOCK:
                    current = _ARENA_REGISTRY.get(object_id)
                    if current is not None and current.reference is reference:
                        _ARENA_REGISTRY.pop(object_id, None)
                with _SEALED_ARENA_LOCK:
                    sealed = _SEALED_ARENA_REGISTRY.get(object_id)
                    if sealed is not None and sealed.arena_reference is reference:
                        _SEALED_ARENA_REGISTRY.pop(object_id, None)
                return
            except BaseException:
                continue

    reference = weakref.ref(arena, cleanup)
    token = _ArenaToken()
    empty = _make_view(token, ())
    truth = _view_key(empty)
    graph = _view_graph(empty)
    state = _ArenaState(
        reference,
        object_id,
        owner,
        owner_state,
        token,
        {},
        {id(empty): (empty, truth, graph)},
        {_digest_key(("view", truth[0])): [empty]},
        {},
        {},
        [],
        0,
        False,
        empty,
    )
    committed_key, committed_graph = _arena_committed_state(state)
    entry = _ArenaRegistryEntry(
        reference,
        state,
        _arena_anchor_graph(state),
        committed_key,
        committed_graph,
    )
    return arena, state, entry


def _register_staged_arena(
    arena: ConstraintArena,
    state: _ArenaState,
    entry: _ArenaRegistryEntry,
) -> None:
    if (
        type(arena) is not ConstraintArena
        or type(state) is not _ArenaState
        or type(entry) is not _ArenaRegistryEntry
        or entry.state is not state
        or entry.reference() is not arena
        or state.reference() is not arena
        or state.arena_id != id(arena)
    ):
        raise ConstraintProgramError("staged arena provenance was substituted")
    with _ARENA_LOCK:
        current = _ARENA_REGISTRY.get(id(arena))
        if (
            current is not None
            and current.reference() is not None
            and (
                current.reference() is not arena
                or current.state is not state
            )
        ):
            raise ConstraintProgramError("arena registry ID was reused")
        _ARENA_REGISTRY[id(arena)] = entry


def _unregister_staged_arena(
    arena: ConstraintArena, state: _ArenaState
) -> None:
    with _ARENA_LOCK:
        current = _ARENA_REGISTRY.get(id(arena))
        if current is not None and (
            current.reference() is arena and current.state is state
        ):
            _ARENA_REGISTRY.pop(id(arena), None)
    with _SEALED_ARENA_LOCK:
        sealed = _SEALED_ARENA_REGISTRY.get(id(arena))
        if sealed is not None and sealed.state is state:
            _SEALED_ARENA_REGISTRY.pop(id(arena), None)


def _prune_dead_prepared(state: _ArenaState) -> None:
    dead = tuple(
        object_id
        for object_id, transaction in state.prepared.items()
        if transaction.reference() is None
    )
    if not dead:
        return
    dead_set = set(dead)
    old = state._data
    target = _ArenaData(
        old.blocks,
        old.views,
        old.view_buckets,
        old.payload_buckets,
        {
            object_id: transaction
            for object_id, transaction in old.prepared.items()
            if object_id not in dead_set
        },
        [object_id for object_id in old.pending if object_id not in dead_set],
        old.next_sequence,
        old.sealed,
        old.discarded,
    )
    repair_error: Optional[BaseException] = None
    for _attempt in range(4):
        try:
            state._data = target
            _publish_arena_state(state)
            return
        except BaseException as error:
            if repair_error is None:
                repair_error = error
            continue
    if repair_error is not None:
        raise repair_error
    raise ConstraintProgramError("dead prepared capabilities could not be pruned")


def _arena_state(value: Any) -> _ArenaState:
    if type(value) is not ConstraintArena:
        raise ConstraintProgramError("arena has the wrong exact type")
    with _ARENA_LOCK:
        entry = _ARENA_REGISTRY.get(id(value))
    if entry is None or entry.reference() is not value:
        raise ConstraintProgramError("arena is forged or stale")
    state = entry.state
    current_graph = _arena_anchor_graph(state)
    if (
        state.arena_id != id(value)
        or len(current_graph) != len(entry.anchor_graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_graph, entry.anchor_graph)
        )
    ):
        raise ConstraintProgramError("arena registry anchor was rebound")
    owner_entry = _owner_operation_entry(state.owner_state)
    if owner_entry is None or not owner_entry.operation_active:
        committed_key, committed_graph = _arena_committed_state(state)
        if (
            committed_key != entry.committed_key
            or len(committed_graph) != len(entry.committed_graph)
            or any(
                current is not sealed
                for current, sealed in zip(committed_graph, entry.committed_graph)
            )
        ):
            raise ConstraintProgramError("arena committed registry state was rebound")
    if _owner_state(state.owner) is not state.owner_state:
        raise ConstraintArenaMismatch("arena owner binding is no longer authentic")
    if (
        state.owner_state.arena_created is not True
        or state.owner_state.arena_id != id(value)
    ):
        raise ConstraintArenaMismatch(
            "arena is not the owner's uniquely published arena"
        )
    if (
        (owner_entry is None or not owner_entry.operation_active)
        and (
            state.sealed is not state.owner_state.sealed
            or state.discarded is not state.owner_state.discarded
        )
    ):
        raise ConstraintProgramError(
            "owner/arena terminal-state publication is incomplete"
        )
    if (
        (owner_entry is None or not owner_entry.operation_active)
        and threading.current_thread() is state.owner_state.thread
    ):
        _prune_dead_prepared(state)
    empty_record = state.views.get(id(state.empty_view))
    empty_truth = _view_key(state.empty_view)
    empty_graph = _view_graph(state.empty_view)
    if (
        state.empty_view._arena_token is not state.token
        or empty_record is None
        or empty_record[0] is not state.empty_view
        or empty_truth[0] != ()
        or empty_record[1] != empty_truth
        or len(empty_record[2]) != len(empty_graph)
        or any(
            current is not sealed
            for current, sealed in zip(empty_graph, empty_record[2])
        )
    ):
        raise ConstraintArenaMismatch("arena empty-view binding is no longer authentic")
    return state


def _require_arena_open(value: Any) -> _ArenaState:
    state = _arena_state(value)
    _require_owner_thread(state.owner_state)
    owner_entry = _owner_operation_entry(state.owner_state)
    if owner_entry is not None and owner_entry.operation_active:
        _mark_owner_reentrancy(state.owner_state)
        raise ConstraintProgramError(
            "arena mutation cannot reenter an active owner operation"
        )
    if state.discarded or state.owner_state.discarded:
        if not state.discarded or not state.owner_state.discarded:
            raise ConstraintProgramError(
                "owner/arena discard publication is incomplete"
            )
        raise ConstraintProgramError("constraint arena is discarded")
    if state.sealed or state.owner_state.sealed:
        raise ConstraintProgramError("constraint arena is sealed")
    if state.owner_state.poisoned:
        raise ConstraintProgramError("constraint owner is poisoned")
    return state


def _make_view(
    token: _ArenaToken, block_keys: Tuple[Tuple[str, int], ...]
) -> ConstraintView:
    view = ConstraintView(_token=_FACTORY_TOKEN)
    ids = []
    for kind, value in block_keys:
        item = ConstraintObjectID(_token=_FACTORY_TOKEN)
        object.__setattr__(item, "_kind", kind)
        object.__setattr__(item, "_value", value)
        object.__setattr__(item, "_sealed", True)
        ids.append(item)
    view_id = _reserve_occurrence_ids("view", 1)[0]
    object.__setattr__(view, "_block_ids", tuple(ids))
    object.__setattr__(view, "_view_id", view_id)
    object.__setattr__(view, "_arena_token", token)
    object.__setattr__(view, "_sealed", True)
    return view


def _view_key(value: Any) -> Tuple[Any, ...]:
    if (
        type(value) is not ConstraintView
        or type(value._block_ids) is not tuple
        or type(value._view_id) is not ConstraintObjectID
        or type(value._arena_token) is not _ArenaToken
        or value._sealed is not True
    ):
        raise ConstraintProgramError("constraint view was forged or mutated")
    block_keys = tuple(_object_key(item, kinds=("block",)) for item in value._block_ids)
    if block_keys != tuple(sorted(set(block_keys), key=lambda item: (item[1], item[0]))):
        raise ConstraintProgramError("constraint view is not a canonical set")
    return block_keys, _object_key(value._view_id, kinds=("view",)), value._arena_token


def _view_graph(value: ConstraintView) -> Tuple[Any, ...]:
    _view_key(value)
    return (
        value._block_ids,
        value._view_id,
        value._arena_token,
        *value._block_ids,
    )


def _validate_view(state: _ArenaState, value: Any) -> Tuple[Tuple[str, int], ...]:
    if type(value) is not ConstraintView:
        raise ConstraintProgramError("constraint view has the wrong exact type")
    if value._arena_token is not state.token:
        raise ConstraintArenaMismatch("constraint view belongs to another arena")
    record = state.views.get(id(value))
    current = _view_key(value)
    graph = _view_graph(value)
    graph_matches = bool(
        record is not None
        and len(record[2]) == len(graph)
        and all(current_item is sealed_item for current_item, sealed_item in zip(graph, record[2]))
    )
    if record is None or record[0] is not value or record[1] != current or not graph_matches:
        raise ConstraintProgramError("constraint view is forged, stale, or mutated")
    if any(key not in state.blocks for key in current[0]):
        raise ConstraintProgramError("constraint view references an uncommitted block")
    return current[0]


def _intern_live_view(
    state: _ArenaState, keys: Tuple[Tuple[str, int], ...]
) -> ConstraintView:
    digest = _digest_key(("view", keys))
    bucket = state.view_buckets.setdefault(digest, [])
    for existing in bucket:
        existing_keys = _validate_view(state, existing)
        if existing_keys == keys:
            return existing
    view = _make_view(state.token, keys)
    truth = _view_key(view)
    state.views[id(view)] = (view, truth, _view_graph(view))
    bucket.append(view)
    return view


def _stage_append(
    state: _ArenaState,
    base_view: ConstraintView,
    base_ids: Tuple[Tuple[str, int], ...],
    frame: FactorFrame,
    frame_key: Tuple[Any, ...],
    payload: _Payload,
    source_tags: Tuple[str, ...],
    legacy_tags: Tuple[str, ...],
    *,
    family: Optional[ConstraintFamily],
    layer_id: int,
) -> PreparedAppend:
    payload_key = payload.key()
    payload_digest = _digest_key(payload_key)
    block_id = _reserve_occurrence_ids("block", 1)[0]
    source_ids = tuple(
        _reserve_occurrence_ids(
            "range_row" if payload.block_kind is _BlockKind.GUARDED_BAND and index < len(payload.ranged_mask) and payload.ranged_mask[index] else "le_row",
            1,
        )[0]
        for index in range(payload.source_rows)
    )
    facet_ids = _reserve_occurrence_ids("facet", payload.virtual_rows)
    if len(source_tags) != payload.source_rows or len(legacy_tags) != payload.virtual_rows:
        raise ConstraintProgramError("row metadata count does not match staged payload")
    block_key = _object_key(block_id, kinds=("block",))
    source_keys = tuple(
        _object_key(item, kinds=("range_row", "le_row")) for item in source_ids
    )
    facet_keys = tuple(_object_key(item, kinds=("facet",)) for item in facet_ids)
    ordinal = state.next_sequence
    # The arena operation owns a copy-on-write data root.  Its failure path
    # publishes only this burned ordinal together with the original arena
    # containers; no staged capability can leak.
    state.next_sequence = ordinal + 1

    transaction: Optional[_PreparedState] = None
    object_id: Optional[int] = None
    try:
        occurrence = _Occurrence(
            block_key,
            ordinal,
            payload,
            payload_key,
            payload_digest,
            source_keys,
            source_tags,
            facet_keys,
            legacy_tags,
            family,
            layer_id,
        )
        occurrence.key()
        if _digest_key(payload_key) != payload_digest:
            raise ConstraintProgramError("payload digest construction failed")
        occurrence_graph = _occurrence_graph(occurrence)
        staged_keys = tuple(
            sorted(
                set(base_ids + (block_key,)),
                key=lambda item: (item[1], item[0]),
            )
        )
        staged_view = _make_view(state.token, staged_keys)
        staged_truth = _view_key(staged_view)
        staged_graph = _view_graph(staged_view)
        prepared = PreparedAppend(_token=_FACTORY_TOKEN)
        result = ConstraintAppend(
            staged_view,
            block_id,
            source_ids,
            facet_ids,
            payload.source_rows,
            payload.virtual_rows,
            payload.source_nnz,
            payload.virtual_nnz,
            int(sum(payload.ranged_mask)),
            int(len(payload.ranged_mask) - sum(payload.ranged_mask)),
            ordinal,
        )
        object_id = id(prepared)

        # The weakref callback deliberately performs no arena mutation.  A
        # handle can die on any thread or during interpreter teardown; the
        # next owner-thread arena lookup prunes dead capabilities through the
        # same copy-on-write publication path as explicit abort.
        reference = weakref.ref(prepared)
        transaction = _PreparedState(
            reference,
            object_id,
            base_view,
            base_ids,
            frame,
            frame_key,
            occurrence,
            occurrence_graph,
            staged_view,
            staged_truth,
            staged_graph,
            result,
            ordinal,
            "prepared",
        )
        state.prepared[object_id] = transaction
        state.pending.append(object_id)
        return prepared
    except BaseException:
        # No staged publication survives a failed prepare.  The occurrence IDs
        # and ordinal remain burned by design.
        if transaction is not None:
            try:
                _abort_transaction(state, transaction)
            except BaseException:
                pass
        elif object_id is not None:
            state.prepared.pop(object_id, None)
            while object_id in state.pending:
                state.pending.remove(object_id)
        raise


def _validate_prepared(state: _ArenaState, value: Any) -> _PreparedState:
    if type(value) is not PreparedAppend:
        raise ConstraintTransactionError("prepared capability has the wrong exact type")
    transaction = state.prepared.get(id(value))
    if (
        transaction is None
        or transaction.reference() is not value
        or transaction.object_id != id(value)
        or transaction.status != "prepared"
    ):
        raise ConstraintTransactionError("prepared capability is forged, stale, or consumed")
    try:
        if _validate_view(state, transaction.base_view) != transaction.base_ids:
            raise ConstraintTransactionError("prepared base view changed after capture")
        if (
            _validate_frame_state(state.owner_state, transaction.frame)
            != transaction.frame_key
        ):
            raise ConstraintTransactionError("prepared factor frame changed after capture")
        current_occurrence_graph = _occurrence_graph(transaction.occurrence)
        if (
            len(current_occurrence_graph) != len(transaction.occurrence_graph)
            or any(
                current is not sealed
                for current, sealed in zip(
                    current_occurrence_graph, transaction.occurrence_graph
                )
            )
        ):
            raise ConstraintTransactionError(
                "prepared occurrence graph was rebound after capture"
            )
        current_graph = _view_graph(transaction.staged_view)
        if (
            _view_key(transaction.staged_view) != transaction.staged_view_truth
            or len(current_graph) != len(transaction.staged_view_graph)
            or any(
                current is not sealed
                for current, sealed in zip(current_graph, transaction.staged_view_graph)
            )
        ):
            raise ConstraintTransactionError("prepared result view was mutated")
        return transaction
    except BaseException:
        _abort_transaction(state, transaction)
        raise


def _abort_transaction(state: _ArenaState, transaction: _PreparedState) -> None:
    if transaction.status != "prepared":
        return
    object_id = transaction.object_id

    def repair() -> None:
        state.prepared.pop(object_id, None)
        while object_id in state.pending:
            state.pending.remove(object_id)

    try:
        transaction.status = "aborted"
        repair()
    except BaseException:
        # Both collections are non-authoritative indexes of the same consumed
        # capability.  Repair both after any instruction-boundary interruption
        # so seal cannot be left behind a permanent ghost pending entry.
        try:
            transaction.status = "aborted"
            repair()
        except BaseException:
            try:
                state.prepared.pop(object_id, None)
                state.pending[:] = [
                    item for item in state.pending if item != object_id
                ]
            except BaseException:
                pass
        raise


def _abort_if_live(arena: ConstraintArena, prepared: PreparedAppend) -> None:
    try:
        state = _arena_state(arena)
        transaction = state.prepared.get(id(prepared))
        if transaction is not None and transaction.reference() is prepared:
            _abort_transaction(state, transaction)
    except BaseException:
        pass


@dataclass(frozen=True)
class _ProgramState:
    reference: Any
    occurrences: Tuple[_Occurrence, ...]
    frame_cont_keys: Tuple[Tuple[str, int, _NamespaceIdentity], ...]
    frame_bin_keys: Tuple[Tuple[str, int, _NamespaceIdentity], ...]
    occurrence_keys: Tuple[Tuple[Any, ...], ...]
    occurrence_graphs: Tuple[Tuple[Any, ...], ...]
    digest: str
    source_rows: int
    virtual_rows: int
    source_nnz: int
    virtual_nnz: int
    ranged_rows: int
    fallback_pairs: int
    numeric_payload_bytes: int


def _program_accounting(
    occurrences: Tuple[_Occurrence, ...]
) -> Tuple[int, int, int, int, int, int, int]:
    seen_payloads = set()
    payload_bytes = 0
    for occurrence in occurrences:
        identity = id(occurrence.payload)
        if identity not in seen_payloads:
            seen_payloads.add(identity)
            payload_bytes += occurrence.payload.payload_bytes
    return (
        int(sum(item.payload.source_rows for item in occurrences)),
        int(sum(item.payload.virtual_rows for item in occurrences)),
        int(sum(item.payload.source_nnz for item in occurrences)),
        int(sum(item.payload.virtual_nnz for item in occurrences)),
        int(sum(sum(item.payload.ranged_mask) for item in occurrences)),
        int(
            sum(
                len(item.payload.ranged_mask) - sum(item.payload.ranged_mask)
                for item in occurrences
            )
        ),
        int(payload_bytes),
    )


def _program_graph(state: _ProgramState) -> Tuple[Any, ...]:
    if (
        type(state) is not _ProgramState
        or type(state.occurrences) is not tuple
        or type(state.frame_cont_keys) is not tuple
        or type(state.frame_bin_keys) is not tuple
        or type(state.occurrence_keys) is not tuple
        or type(state.occurrence_graphs) is not tuple
        or type(state.digest) is not str
        or not state.digest
        or any(
            type(value) is not int or value < 0
            for value in (
                state.source_rows,
                state.virtual_rows,
                state.source_nnz,
                state.virtual_nnz,
                state.ranged_rows,
                state.fallback_pairs,
                state.numeric_payload_bytes,
            )
        )
    ):
        raise ConstraintProgramError("sealed program registry state was substituted")
    continuous_keys = _validate_external_keys(
        state.frame_cont_keys,
        kind=FactorKind.CONTINUOUS,
        name="program continuous factor keys",
    )
    binary_keys = _validate_external_keys(
        state.frame_bin_keys,
        kind=FactorKind.BINARY,
        name="program binary factor keys",
    )
    if {item[1] for item in continuous_keys}.intersection(
        item[1] for item in binary_keys
    ):
        raise ConstraintProgramError("program factor raw IDs collide across kinds")
    namespaces = tuple(item[2] for item in continuous_keys + binary_keys)
    if namespaces and any(item is not namespaces[0] for item in namespaces):
        raise ConstraintProgramError("program crosses external factor namespaces")
    graph = [
        state,
        state.reference,
        state.occurrences,
        state.frame_cont_keys,
        state.frame_bin_keys,
        state.occurrence_keys,
        state.occurrence_graphs,
        state.digest,
        state.source_rows,
        state.virtual_rows,
        state.source_nnz,
        state.virtual_nnz,
        state.ranged_rows,
        state.fallback_pairs,
        state.numeric_payload_bytes,
    ]
    graph.extend(state.occurrences)
    graph.extend(state.frame_cont_keys)
    graph.extend(state.frame_bin_keys)
    graph.extend(state.occurrence_keys)
    graph.extend(state.occurrence_graphs)
    for occurrence_graph in state.occurrence_graphs:
        graph.extend(occurrence_graph)
    return tuple(graph)


@dataclass(frozen=True)
class _ProgramRecord:
    reference: Any
    state: _ProgramState
    graph: Tuple[Any, ...]


_PROGRAM_LOCK = threading.Lock()
_PROGRAM_REGISTRY: Dict[int, _ProgramRecord] = {}


class ConstraintProgram:
    """Sealed bytes-backed representation/replay authority (not a proof)."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ConstraintProgram":
        if cls is not ConstraintProgram or _kwargs.pop("_token", None) is not _PROGRAM_FACTORY_TOKEN:
            raise TypeError("ConstraintProgram cannot be self-signed")
        if _args or _kwargs:
            raise TypeError("malformed internal ConstraintProgram construction")
        return object.__new__(cls)

    @property
    def schema(self) -> str:
        _program_state(self)
        return _SCHEMA

    @property
    def continuous_ids(self) -> Tuple[ExternalFactorID, ...]:
        state = _program_state(self)
        return tuple(_new_factor_id(FactorKind.CONTINUOUS, raw, namespace) for _kind, raw, namespace in state.frame_cont_keys)

    @property
    def binary_ids(self) -> Tuple[ExternalFactorID, ...]:
        state = _program_state(self)
        return tuple(_new_factor_id(FactorKind.BINARY, raw, namespace) for _kind, raw, namespace in state.frame_bin_keys)

    @property
    def block_count(self) -> int:
        return len(_program_state(self).occurrences)

    @property
    def source_rows(self) -> int:
        return _program_state(self).source_rows

    @property
    def virtual_facet_rows(self) -> int:
        return _program_state(self).virtual_rows

    @property
    def source_nnz(self) -> int:
        return _program_state(self).source_nnz

    @property
    def virtual_facet_nnz(self) -> int:
        return _program_state(self).virtual_nnz

    @property
    def ranged_rows(self) -> int:
        return _program_state(self).ranged_rows

    @property
    def fallback_pairs(self) -> int:
        return _program_state(self).fallback_pairs

    @property
    def numeric_payload_bytes(self) -> int:
        return _program_state(self).numeric_payload_bytes

    @property
    def append_ordinals(self) -> Tuple[int, ...]:
        return tuple(item.ordinal for item in _program_state(self).occurrences)

    @property
    def digest(self) -> str:
        return _program_state(self).digest

    @property
    def block_digests(self) -> Tuple[str, ...]:
        return tuple(item.payload_digest for item in _program_state(self).occurrences)

    @property
    def receipt(self) -> Mapping[str, Any]:
        state = _program_state(self)
        return MappingProxyType(
            {
                "schema": _SCHEMA,
                "representation_authority": True,
                "replay_authority": True,
                "proof_authority": False,
                "verdict_authority": False,
                "solver_status_authority": False,
                "authenticity_from_digest": False,
                "hash_is_bucket_only": True,
                "bytes_backed_source": True,
                "consumer_integration": False,
                "block_count": len(state.occurrences),
                "source_rows": state.source_rows,
                "virtual_facet_rows": state.virtual_rows,
                "source_nnz": state.source_nnz,
                "virtual_facet_nnz": state.virtual_nnz,
                "ranged_rows": state.ranged_rows,
                "fallback_pairs": state.fallback_pairs,
                "numeric_payload_bytes": state.numeric_payload_bytes,
                "program_digest": state.digest,
                "block_digests": tuple(item.payload_digest for item in state.occurrences),
                "triangle_relaxation_called": False,
                "branch_and_bound_called": False,
                "backward_called": False,
                "dual_called": False,
                "solver_proof_called": False,
                "real_model_called": False,
                "large_model_called": False,
            }
        )

    @property
    def representation_authority(self) -> bool:
        _program_state(self)
        return True

    @property
    def replay_authority(self) -> bool:
        _program_state(self)
        return True

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def verdict_authority(self) -> bool:
        return False

    @property
    def solver_status_authority(self) -> bool:
        return False

    def iter_native_batches(self, *, max_rows: int) -> Iterator["NativeConstraintBatch"]:
        return iter_native_batches(self, max_rows=max_rows)

    def iter_legacy_facet_batches(self, *, max_rows: int) -> Iterator["LegacyFacetBatch"]:
        return iter_legacy_facet_batches(self, max_rows=max_rows)


def _stage_program(
    occurrences: Tuple[_Occurrence, ...], frame_key: Tuple[Any, ...]
) -> Tuple[ConstraintProgram, _ProgramState]:
    occurrence_keys = tuple(item.key() for item in occurrences)
    occurrence_graphs = tuple(_occurrence_graph(item) for item in occurrences)
    canonical_key = (
        _SCHEMA,
        tuple((kind, raw, id(namespace)) for kind, raw, namespace in frame_key[0]),
        tuple((kind, raw, id(namespace)) for kind, raw, namespace in frame_key[1]),
        occurrence_keys,
    )
    digest = _digest_key(canonical_key)
    accounting = _program_accounting(occurrences)
    program = ConstraintProgram(_token=_PROGRAM_FACTORY_TOKEN)
    object_id = id(program)

    def cleanup(reference: Any) -> None:
        for _attempt in range(4):
            try:
                with _PROGRAM_LOCK:
                    current = _PROGRAM_REGISTRY.get(object_id)
                    if current is not None and current.reference is reference:
                        _PROGRAM_REGISTRY.pop(object_id, None)
                return
            except BaseException:
                continue

    reference = weakref.ref(program, cleanup)
    state = _ProgramState(
        reference,
        occurrences,
        frame_key[0],
        frame_key[1],
        occurrence_keys,
        occurrence_graphs,
        digest,
        *accounting,
    )
    _program_graph(state)
    return program, state


def _register_staged_program(
    program: ConstraintProgram, state: _ProgramState
) -> None:
    if (
        type(program) is not ConstraintProgram
        or type(state) is not _ProgramState
        or state.reference() is not program
    ):
        raise ConstraintProgramError("staged program provenance was substituted")
    with _SEALED_ARENA_LOCK:
        provenance = tuple(
            record
            for record in _SEALED_ARENA_REGISTRY.values()
            if record.program() is program and record.program_state is state
        )
    if len(provenance) != 1:
        raise ConstraintProgramError(
            "program registration lacks one sealed-arena provenance record"
        )
    sealed_record = provenance[0]
    sealed_graph = _sealed_arena_record_graph(sealed_record)
    owner_entry = _owner_operation_entry(sealed_record.state.owner_state)
    with _ARENA_LOCK:
        arena_entry = _ARENA_REGISTRY.get(sealed_record.state.arena_id)
    if (
        sealed_record.arena_reference() is None
        or not sealed_record.state.sealed
        or not sealed_record.state.owner_state.sealed
        or sealed_record.state.discarded
        or sealed_record.state.owner_state.discarded
        or owner_entry is None
        or owner_entry.operation_active
        or not owner_entry.state.sealed
        or arena_entry is None
        or arena_entry.state is not sealed_record.state
        or len(sealed_graph) != len(sealed_record.graph)
        or any(
            current is not sealed
            for current, sealed in zip(sealed_graph, sealed_record.graph)
        )
    ):
        raise ConstraintProgramError(
            "program registration crossed an incomplete seal publication"
        )
    record = _ProgramRecord(state.reference, state, _program_graph(state))
    with _PROGRAM_LOCK:
        existing = _PROGRAM_REGISTRY.get(id(program))
        if (
            existing is not None
            and existing.reference() is not None
            and (
                existing.reference() is not program
                or existing.state is not state
            )
        ):
            raise ConstraintProgramError("program registry ID was reused")
        _PROGRAM_REGISTRY[id(program)] = record


def _is_registered_program(
    program: ConstraintProgram, state: _ProgramState
) -> bool:
    with _PROGRAM_LOCK:
        record = _PROGRAM_REGISTRY.get(id(program))
    return bool(
        record is not None
        and record.reference() is program
        and record.state is state
    )


def _sealed_arena_record_graph(record: _SealedArenaRecord) -> Tuple[Any, ...]:
    if type(record) is not _SealedArenaRecord:
        raise ConstraintProgramError("sealed arena record was substituted")
    return (
        record.arena_reference,
        record.state,
        record.view,
        record.view_truth,
        record.view_graph,
        *record.view_graph,
        record.frame,
        record.frame_key,
        record.frame_graph,
        *record.frame_graph,
        record.program,
        record.program_state,
    )


def _complete_discard_publication(
    state: _ArenaState, operation_epoch: int
) -> None:
    """Publish one terminal non-authoritative owner/arena discard pair."""

    owner_entry = _owner_operation_entry(state.owner_state)
    if (
        owner_entry is None
        or not owner_entry.operation_active
        or owner_entry.operation_epoch != operation_epoch
        or not state.discarded
        or not state.owner_state.discarded
        or state.sealed
        or state.owner_state.sealed
        or state.pending
        or state.prepared
    ):
        raise ConstraintProgramError("discard publication state is incomplete")
    with _SEALED_ARENA_LOCK:
        if _SEALED_ARENA_REGISTRY.get(state.arena_id) is not None:
            raise ConstraintProgramError(
                "discard cannot cross a sealed-program publication"
            )
    # As with seal, the arena registry is installed first and the final idle
    # owner swap is the effective pair linearization point.  Unlike seal,
    # there is deliberately no program-registry publication afterward.
    _publish_arena_state(state)
    _assert_owner_operation(state.owner_state, operation_epoch)
    _commit_owner_operation(state.owner_state, operation_epoch)


def _complete_seal_publication(
    arena: ConstraintArena,
    state: _ArenaState,
    view: ConstraintView,
    frame: FactorFrame,
    program: ConstraintProgram,
    program_state: _ProgramState,
    operation_epoch: int,
) -> ConstraintProgram:
    view_record = state.views.get(id(view))
    frame_record = state.owner_state.frames.get(id(frame))
    if (
        view_record is None
        or view_record[0] is not view
        or frame_record is None
        or frame_record[0] is not frame
    ):
        raise ConstraintProgramError("seal publication lost its captured inputs")
    with _SEALED_ARENA_LOCK:
        record = _SEALED_ARENA_REGISTRY.get(state.arena_id)
        if record is None:
            provisional = _SealedArenaRecord(
                state.reference,
                state,
                view,
                view_record[1],
                view_record[2],
                frame,
                frame_record[1],
                frame_record[2],
                program_state.reference,
                program_state,
                (),
            )
            record = _SealedArenaRecord(
                provisional.arena_reference,
                provisional.state,
                provisional.view,
                provisional.view_truth,
                provisional.view_graph,
                provisional.frame,
                provisional.frame_key,
                provisional.frame_graph,
                provisional.program,
                provisional.program_state,
                _sealed_arena_record_graph(provisional),
            )
            _SEALED_ARENA_REGISTRY[state.arena_id] = record
        elif (
            record.arena_reference() is not arena
            or record.state is not state
            or record.view is not view
            or record.frame is not frame
            or record.program() is not program
            or record.program_state is not program_state
        ):
            raise ConstraintProgramError("arena has a conflicting sealed program")
    current_graph = _sealed_arena_record_graph(record)
    if (
        len(current_graph) != len(record.graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_graph, record.graph)
        )
    ):
        raise ConstraintProgramError("sealed arena publication graph was rebound")
    owner_entry = _owner_operation_entry(state.owner_state)
    if owner_entry is not None and not owner_entry.operation_active:
        if (
            owner_entry.operation_epoch == operation_epoch
            and state.sealed
            and state.owner_state.sealed
            and not state.discarded
            and not state.owner_state.discarded
            and _program_state(program) is program_state
        ):
            return program
        raise ConstraintProgramError("seal publication reached a conflicting idle owner")
    if state.discarded or state.owner_state.discarded:
        raise ConstraintProgramError("discarded arena cannot publish a program")
    state.sealed = True
    state.owner_state.sealed = True
    owner_entry = _owner_operation_entry(state.owner_state)
    if (
        owner_entry is None
        or not owner_entry.operation_active
        or owner_entry.operation_epoch != operation_epoch
    ):
        raise ConstraintProgramError("seal operation epoch changed before publication")
    # Install arena truth first, then the final idle+sealed owner truth.  That
    # owner publication is the pair's effective linearization point; the
    # program registry remains strictly last.
    _publish_arena_state(state)
    _commit_owner_operation(state.owner_state, operation_epoch)
    _register_staged_program(program, program_state)
    return program


def _sealed_arena_program(
    arena: ConstraintArena,
    state: _ArenaState,
    view: ConstraintView,
    frame: FactorFrame,
) -> ConstraintProgram:
    with _SEALED_ARENA_LOCK:
        record = _SEALED_ARENA_REGISTRY.get(state.arena_id)
    if record is None:
        raise ConstraintProgramError("sealed arena lost its program publication")
    current_graph = _sealed_arena_record_graph(record)
    if (
        record.arena_reference() is not arena
        or record.state is not state
        or record.view is not view
        or record.frame is not frame
        or not state.sealed
        or not state.owner_state.sealed
        or state.discarded
        or state.owner_state.discarded
        or len(current_graph) != len(record.graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_graph, record.graph)
        )
        or _validate_view(state, view) != record.view_truth[0]
        or _view_key(view) != record.view_truth
        or _validate_frame_state(state.owner_state, frame) != record.frame_key
    ):
        raise ConstraintProgramError("sealed arena retrieval was forged or mismatched")
    while True:
        program = record.program()
        if program is not None:
            # The sealed record chooses the one recoverable handle *before*
            # that handle can become authoritative.  If a BaseException hit
            # between record CAS and registry publication, any reader repairs
            # the same chosen handle rather than staging a competitor.
            if not _is_registered_program(program, record.program_state):
                _register_staged_program(program, record.program_state)
            if _program_state(program) is not record.program_state:
                raise ConstraintProgramError("sealed arena program graph was rebound")
            return program

        staged_program, staged_state = _stage_program(
            record.program_state.occurrences,
            record.frame_key,
        )
        provisional = _SealedArenaRecord(
            record.arena_reference,
            record.state,
            record.view,
            record.view_truth,
            record.view_graph,
            record.frame,
            record.frame_key,
            record.frame_graph,
            staged_state.reference,
            staged_state,
            (),
        )
        replacement = _SealedArenaRecord(
            provisional.arena_reference,
            provisional.state,
            provisional.view,
            provisional.view_truth,
            provisional.view_graph,
            provisional.frame,
            provisional.frame_key,
            provisional.frame_graph,
            provisional.program,
            provisional.program_state,
            _sealed_arena_record_graph(provisional),
        )
        with _SEALED_ARENA_LOCK:
            current = _SEALED_ARENA_REGISTRY.get(state.arena_id)
            if current is record:
                _SEALED_ARENA_REGISTRY[state.arena_id] = replacement
                record = replacement
            else:
                if current is None:
                    raise ConstraintProgramError(
                        "sealed arena record vanished during retrieval"
                    )
                record = current
        current_graph = _sealed_arena_record_graph(record)
        if (
            record.arena_reference() is not arena
            or record.state is not state
            or record.view is not view
            or record.frame is not frame
            or len(current_graph) != len(record.graph)
            or any(
                current_item is not sealed_item
                for current_item, sealed_item in zip(current_graph, record.graph)
            )
        ):
            raise ConstraintProgramError(
                "sealed arena record changed to an inauthentic value"
            )


def _program_state(value: Any) -> _ProgramState:
    if type(value) is not ConstraintProgram:
        raise ConstraintProgramError("constraint program has the wrong exact type")
    with _PROGRAM_LOCK:
        record = _PROGRAM_REGISTRY.get(id(value))
    if record is None or record.reference() is not value:
        raise ConstraintProgramError("constraint program is forged or stale")
    state = record.state
    current_program_graph = _program_graph(state)
    if (
        len(current_program_graph) != len(record.graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_program_graph, record.graph)
        )
    ):
        raise ConstraintProgramError("sealed program registry graph was rebound")
    if (
        state.source_rows,
        state.virtual_rows,
        state.source_nnz,
        state.virtual_nnz,
        state.ranged_rows,
        state.fallback_pairs,
        state.numeric_payload_bytes,
    ) != _program_accounting(state.occurrences):
        raise ConstraintProgramError(
            "sealed program accounting does not match its occurrence graph"
        )
    current = tuple(item.key() for item in state.occurrences)
    current_graphs = tuple(_occurrence_graph(item) for item in state.occurrences)
    graphs_match = bool(
        len(current_graphs) == len(state.occurrence_graphs)
        and all(
            len(current_graph) == len(sealed_graph)
            and all(
                current_item is sealed_item
                for current_item, sealed_item in zip(current_graph, sealed_graph)
            )
            for current_graph, sealed_graph in zip(
                current_graphs, state.occurrence_graphs
            )
        )
    )
    if current != state.occurrence_keys or not graphs_match:
        raise ConstraintProgramError("sealed constraint graph was mutated")
    if type(state.digest) is not str or not state.digest:
        raise ConstraintProgramError("sealed constraint digest was substituted")
    return state


def _id_from_key(key: Tuple[str, int]) -> ConstraintObjectID:
    result = ConstraintObjectID(_token=_FACTORY_TOKEN)
    object.__setattr__(result, "_kind", key[0])
    object.__setattr__(result, "_value", key[1])
    object.__setattr__(result, "_sealed", True)
    return result


@dataclass(frozen=True)
class _BatchState:
    A_cont: _FrozenCSR
    A_bin: _FrozenCSR
    lower: Optional[_FrozenVector]
    upper: _FrozenVector
    row_keys: Tuple[Tuple[str, int], ...]
    row_tags: Tuple[str, ...]
    block_keys: Tuple[Tuple[str, int], ...]
    append_ordinals: Tuple[int, ...]
    cont_keys: Tuple[Tuple[str, int, _NamespaceIdentity], ...]
    bin_keys: Tuple[Tuple[str, int, _NamespaceIdentity], ...]
    row_offset: int
    total_rows: int


def _batch_key(state: _BatchState) -> Tuple[Any, ...]:
    if (
        type(state) is not _BatchState
        or type(state.A_cont) is not _FrozenCSR
        or type(state.A_bin) is not _FrozenCSR
        or (state.lower is not None and type(state.lower) is not _FrozenVector)
        or type(state.upper) is not _FrozenVector
        or type(state.row_keys) is not tuple
        or type(state.row_tags) is not tuple
        or type(state.block_keys) is not tuple
        or type(state.append_ordinals) is not tuple
        or type(state.cont_keys) is not tuple
        or type(state.bin_keys) is not tuple
        or type(state.row_offset) is not int
        or type(state.total_rows) is not int
    ):
        raise ConstraintProgramError("replay batch graph was substituted")
    continuous_keys = _validate_external_keys(
        state.cont_keys,
        kind=FactorKind.CONTINUOUS,
        name="batch continuous factor keys",
    )
    binary_keys = _validate_external_keys(
        state.bin_keys,
        kind=FactorKind.BINARY,
        name="batch binary factor keys",
    )
    if {item[1] for item in continuous_keys}.intersection(
        item[1] for item in binary_keys
    ):
        raise ConstraintProgramError("batch factor raw IDs collide across kinds")
    namespaces = tuple(item[2] for item in continuous_keys + binary_keys)
    if namespaces and any(item is not namespaces[0] for item in namespaces):
        raise ConstraintProgramError("batch crosses external factor namespaces")
    rows = state.A_cont.rows
    if (
        rows <= 0
        or rows > 256
        or state.A_bin.rows != rows
        or state.upper.length != rows
        or (state.lower is not None and state.lower.length != rows)
        or len(state.row_keys) != rows
        or len(state.row_tags) != rows
        or len(state.block_keys) != rows
        or len(state.append_ordinals) != rows
        or state.A_cont.columns != len(state.cont_keys)
        or state.A_bin.columns != len(state.bin_keys)
        or state.row_offset < 0
        or state.total_rows <= 0
        or state.row_offset + rows > state.total_rows
        or any(type(item) is not str or not item for item in state.row_tags)
        or any(type(item) is not int or item < 0 for item in state.append_ordinals)
    ):
        raise ConstraintProgramError("replay batch dimensions are inconsistent")
    for key in state.row_keys:
        if (
            type(key) is not tuple
            or len(key) != 2
            or key[0] not in {"le_row", "range_row", "facet"}
            or type(key[1]) is not int
            or key[1] < 0
        ):
            raise ConstraintProgramError("replay batch has a malformed row identity")
    for key in state.block_keys:
        if type(key) is not tuple or len(key) != 2 or key[0] != "block" or type(key[1]) is not int or key[1] < 0:
            raise ConstraintProgramError("replay batch has a malformed block identity")
    return (
        state.A_cont.key(),
        state.A_bin.key(),
        None if state.lower is None else state.lower.key(),
        state.upper.key(),
        state.row_keys,
        state.row_tags,
        state.block_keys,
        state.append_ordinals,
        tuple((kind, raw, id(namespace)) for kind, raw, namespace in state.cont_keys),
        tuple((kind, raw, id(namespace)) for kind, raw, namespace in state.bin_keys),
        state.row_offset,
        state.total_rows,
    )


def _batch_graph(state: _BatchState) -> Tuple[Any, ...]:
    _batch_key(state)
    graph = [
        state,
        state.A_cont,
        state.A_cont.data_bytes,
        state.A_cont.indices_bytes,
        state.A_cont.indptr_bytes,
        state.A_bin,
        state.A_bin.data_bytes,
        state.A_bin.indices_bytes,
        state.A_bin.indptr_bytes,
        state.lower,
        state.upper,
        state.upper.raw,
        state.row_keys,
        state.row_tags,
        state.block_keys,
        state.append_ordinals,
        state.cont_keys,
        state.bin_keys,
    ]
    if state.lower is not None:
        graph.append(state.lower.raw)
    graph.extend(state.row_keys)
    graph.extend(state.block_keys)
    graph.extend(state.cont_keys)
    graph.extend(state.bin_keys)
    return tuple(graph)


@dataclass(frozen=True)
class _BatchRecord:
    reference: Any
    state: _BatchState
    full_key: Tuple[Any, ...]
    graph: Tuple[Any, ...]


_BATCH_LOCK = threading.Lock()
_BATCH_REGISTRY: Dict[int, _BatchRecord] = {}


def _register_batch(value: Any, state: _BatchState) -> None:
    object_id = id(value)

    def cleanup(reference: Any) -> None:
        for _attempt in range(4):
            try:
                with _BATCH_LOCK:
                    current = _BATCH_REGISTRY.get(object_id)
                    if current is not None and current.reference is reference:
                        _BATCH_REGISTRY.pop(object_id, None)
                return
            except BaseException:
                continue

    reference = weakref.ref(value, cleanup)
    record = _BatchRecord(reference, state, _batch_key(state), _batch_graph(state))
    with _BATCH_LOCK:
        existing = _BATCH_REGISTRY.get(object_id)
        if existing is not None and existing.reference() is not None:
            raise ConstraintProgramError("replay batch registry ID was reused")
        _BATCH_REGISTRY[object_id] = record


def _validated_batch(value: Any) -> _BatchState:
    if type(value) not in {NativeConstraintBatch, LegacyFacetBatch}:
        raise ConstraintProgramError("replay batch has the wrong exact type")
    with _BATCH_LOCK:
        record = _BATCH_REGISTRY.get(id(value))
    if (
        record is None
        or record.reference() is not value
        or value._state is not record.state
        or value._sealed is not True
    ):
        raise ConstraintProgramError("replay batch is forged or stale")
    current_graph = _batch_graph(record.state)
    if (
        _batch_key(record.state) != record.full_key
        or len(current_graph) != len(record.graph)
        or any(
            current is not sealed
            for current, sealed in zip(current_graph, record.graph)
        )
    ):
        raise ConstraintProgramError("replay batch graph was mutated")
    return record.state


class _ImmutableBatch:
    __slots__ = ("_state", "_sealed", "__weakref__")

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("constraint replay batch is immutable")
        object.__setattr__(self, name, value)

    @property
    def A_cont(self) -> sp.csr_matrix:
        return _validated_batch(self).A_cont.csr()

    @property
    def A_bin(self) -> sp.csr_matrix:
        return _validated_batch(self).A_bin.csr()

    @property
    def upper(self) -> np.ndarray:
        return _validated_batch(self).upper.array()

    @property
    def row_ids(self) -> Tuple[ConstraintObjectID, ...]:
        return tuple(_id_from_key(key) for key in _validated_batch(self).row_keys)

    @property
    def row_tags(self) -> Tuple[str, ...]:
        return _validated_batch(self).row_tags

    @property
    def block_ids(self) -> Tuple[ConstraintObjectID, ...]:
        return tuple(_id_from_key(key) for key in _validated_batch(self).block_keys)

    @property
    def append_ordinals(self) -> Tuple[int, ...]:
        return _validated_batch(self).append_ordinals

    @property
    def continuous_ids(self) -> Tuple[ExternalFactorID, ...]:
        state = _validated_batch(self)
        return tuple(_new_factor_id(FactorKind.CONTINUOUS, raw, namespace) for _kind, raw, namespace in state.cont_keys)

    @property
    def binary_ids(self) -> Tuple[ExternalFactorID, ...]:
        state = _validated_batch(self)
        return tuple(_new_factor_id(FactorKind.BINARY, raw, namespace) for _kind, raw, namespace in state.bin_keys)

    @property
    def row_offset(self) -> int:
        return _validated_batch(self).row_offset

    @property
    def row_count(self) -> int:
        return _validated_batch(self).upper.length

    @property
    def total_rows(self) -> int:
        return _validated_batch(self).total_rows

    @property
    def bytes_backed(self) -> bool:
        _validated_batch(self)
        return True

    @property
    def representation_authority(self) -> bool:
        _validated_batch(self)
        return False

    @property
    def replay_authority(self) -> bool:
        _validated_batch(self)
        return False

    @property
    def proof_authority(self) -> bool:
        return False


class NativeConstraintBatch(_ImmutableBatch):
    __slots__ = ()

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("NativeConstraintBatch cannot be self-signed")

    @property
    def lower(self) -> np.ndarray:
        state = _validated_batch(self)
        if state.lower is None:
            raise ConstraintProgramError("native batch lost its lower bounds")
        return state.lower.array()


class LegacyFacetBatch(_ImmutableBatch):
    __slots__ = ()

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("LegacyFacetBatch cannot be self-signed")


def _new_batch(state: _BatchState, *, native: bool) -> _ImmutableBatch:
    _batch_key(state)
    if native:
        if state.lower is None:
            raise ConstraintProgramError("native batches require lower bounds")
        value = object.__new__(NativeConstraintBatch)
    else:
        if state.lower is not None:
            raise ConstraintProgramError("legacy LE batches cannot carry lower bounds")
        value = object.__new__(LegacyFacetBatch)
    object.__setattr__(value, "_state", state)
    object.__setattr__(value, "_sealed", True)
    _register_batch(value, state)
    return value


def _assemble_rows(
    rows: Sequence[Tuple[np.ndarray, np.ndarray]], *, columns: int
) -> sp.csr_matrix:
    indptr = np.empty(len(rows) + 1, dtype=np.int64)
    indptr[0] = 0
    for index, (indices, data) in enumerate(rows):
        if indices.ndim != 1 or data.ndim != 1 or indices.size != data.size:
            raise ConstraintProgramError("replay row arrays are malformed")
        indptr[index + 1] = indptr[index] + indices.size
    total = int(indptr[-1])
    indices = np.empty(total, dtype=np.int64)
    data = np.empty(total, dtype=np.float64)
    cursor = 0
    for row_indices, row_data in rows:
        stop = cursor + row_indices.size
        indices[cursor:stop] = row_indices
        data[cursor:stop] = row_data
        cursor = stop
    result = sp.csr_matrix(
        (data, indices, indptr),
        shape=(len(rows), int(columns)),
        dtype=np.float64,
    )
    if not result.has_canonical_format or not result.has_sorted_indices:
        raise ConstraintProgramError("replay constructed noncanonical CSR")
    return result


def _matrix_row(matrix: sp.csr_matrix, row: int) -> Tuple[np.ndarray, np.ndarray]:
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    return matrix.indices[start:stop], matrix.data[start:stop]


@dataclass(frozen=True)
class _ReplayOccurrenceCapture:
    block_key: Tuple[str, int]
    ordinal: int
    A_cont: _FrozenCSR
    A_bin: _FrozenCSR
    lower: _FrozenVector
    upper: _FrozenVector
    source_rows: int
    virtual_rows: int
    legacy_positions: Tuple[int, ...]
    legacy_flip: Tuple[bool, ...]
    legacy_bound_from_lower: Tuple[bool, ...]
    source_row_keys: Tuple[Tuple[str, int], ...]
    source_tags: Tuple[str, ...]
    legacy_row_keys: Tuple[Tuple[str, int], ...]
    legacy_tags: Tuple[str, ...]


@dataclass(frozen=True)
class _ReplayCapture:
    occurrences: Tuple[_ReplayOccurrenceCapture, ...]
    cont_keys: Tuple[Tuple[str, int, _NamespaceIdentity], ...]
    bin_keys: Tuple[Tuple[str, int, _NamespaceIdentity], ...]
    total_rows: int
    native: bool


def _capture_occurrence(value: _Occurrence) -> _ReplayOccurrenceCapture:
    value.key()
    payload = value.payload
    return _ReplayOccurrenceCapture(
        value.block_key,
        value.ordinal,
        _FrozenCSR(
            payload.A_cont.rows,
            payload.A_cont.columns,
            payload.A_cont.data_bytes,
            payload.A_cont.indices_bytes,
            payload.A_cont.indptr_bytes,
        ),
        _FrozenCSR(
            payload.A_bin.rows,
            payload.A_bin.columns,
            payload.A_bin.data_bytes,
            payload.A_bin.indices_bytes,
            payload.A_bin.indptr_bytes,
        ),
        _FrozenVector(payload.lower.length, payload.lower.raw),
        _FrozenVector(payload.upper.length, payload.upper.raw),
        payload.source_rows,
        payload.virtual_rows,
        payload.legacy_positions,
        payload.legacy_flip,
        payload.legacy_bound_from_lower,
        value.source_row_keys,
        value.source_tags,
        value.legacy_row_keys,
        value.legacy_tags,
    )


def _replay_capture_truth(
    capture: _ReplayCapture,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    if (
        type(capture) is not _ReplayCapture
        or type(capture.occurrences) is not tuple
        or type(capture.cont_keys) is not tuple
        or type(capture.bin_keys) is not tuple
        or type(capture.total_rows) is not int
        or capture.total_rows < 0
        or type(capture.native) is not bool
    ):
        raise ConstraintProgramError("replay capture was substituted")
    continuous_keys = _validate_external_keys(
        capture.cont_keys,
        kind=FactorKind.CONTINUOUS,
        name="capture continuous factor keys",
    )
    binary_keys = _validate_external_keys(
        capture.bin_keys,
        kind=FactorKind.BINARY,
        name="capture binary factor keys",
    )
    if {item[1] for item in continuous_keys}.intersection(
        item[1] for item in binary_keys
    ):
        raise ConstraintProgramError("capture factor raw IDs collide across kinds")
    namespaces = tuple(item[2] for item in continuous_keys + binary_keys)
    if namespaces and any(item is not namespaces[0] for item in namespaces):
        raise ConstraintProgramError("capture crosses external factor namespaces")
    occurrence_keys = []
    graph = [
        capture,
        capture.occurrences,
        capture.cont_keys,
        capture.bin_keys,
    ]
    computed_total = 0
    for occurrence in capture.occurrences:
        if (
            type(occurrence) is not _ReplayOccurrenceCapture
            or type(occurrence.block_key) is not tuple
            or len(occurrence.block_key) != 2
            or occurrence.block_key[0] != "block"
            or type(occurrence.ordinal) is not int
            or occurrence.ordinal < 0
            or type(occurrence.A_cont) is not _FrozenCSR
            or type(occurrence.A_bin) is not _FrozenCSR
            or type(occurrence.lower) is not _FrozenVector
            or type(occurrence.upper) is not _FrozenVector
            or occurrence.A_bin.rows != occurrence.A_cont.rows
            or occurrence.lower.length != occurrence.A_cont.rows
            or occurrence.upper.length != occurrence.A_cont.rows
            # A source block may have been authored against an earlier
            # append-only factor-frame prefix.  Replay pads trailing columns
            # to the sealed final frame in ``_iterator_load``.
            or occurrence.A_cont.columns > len(capture.cont_keys)
            or occurrence.A_bin.columns > len(capture.bin_keys)
            or type(occurrence.source_rows) is not int
            or occurrence.source_rows != occurrence.A_cont.rows
            or type(occurrence.virtual_rows) is not int
            or occurrence.virtual_rows <= 0
            or len(occurrence.source_row_keys) != occurrence.source_rows
            or len(occurrence.source_tags) != occurrence.source_rows
            or len(occurrence.legacy_positions) != occurrence.virtual_rows
            or len(occurrence.legacy_flip) != occurrence.virtual_rows
            or len(occurrence.legacy_bound_from_lower) != occurrence.virtual_rows
            or len(occurrence.legacy_row_keys) != occurrence.virtual_rows
            or len(occurrence.legacy_tags) != occurrence.virtual_rows
        ):
            raise ConstraintProgramError("replay occurrence capture is malformed")
        key = (
            occurrence.block_key,
            occurrence.ordinal,
            occurrence.A_cont.key(),
            occurrence.A_bin.key(),
            occurrence.lower.key(),
            occurrence.upper.key(),
            occurrence.source_rows,
            occurrence.virtual_rows,
            occurrence.legacy_positions,
            occurrence.legacy_flip,
            occurrence.legacy_bound_from_lower,
            occurrence.source_row_keys,
            occurrence.source_tags,
            occurrence.legacy_row_keys,
            occurrence.legacy_tags,
        )
        occurrence_keys.append(key)
        graph.extend(
            (
                occurrence,
                occurrence.A_cont,
                occurrence.A_cont.data_bytes,
                occurrence.A_cont.indices_bytes,
                occurrence.A_cont.indptr_bytes,
                occurrence.A_bin,
                occurrence.A_bin.data_bytes,
                occurrence.A_bin.indices_bytes,
                occurrence.A_bin.indptr_bytes,
                occurrence.lower,
                occurrence.lower.raw,
                occurrence.upper,
                occurrence.upper.raw,
                occurrence.legacy_positions,
                occurrence.legacy_flip,
                occurrence.legacy_bound_from_lower,
                occurrence.source_row_keys,
                occurrence.source_tags,
                occurrence.legacy_row_keys,
                occurrence.legacy_tags,
            )
        )
        computed_total += (
            occurrence.source_rows if capture.native else occurrence.virtual_rows
        )
    if computed_total != capture.total_rows:
        raise ConstraintProgramError("replay capture total-row count is inconsistent")
    return (
        (
            tuple(occurrence_keys),
            tuple((kind, raw, id(namespace)) for kind, raw, namespace in capture.cont_keys),
            tuple((kind, raw, id(namespace)) for kind, raw, namespace in capture.bin_keys),
            capture.total_rows,
            capture.native,
        ),
        tuple(graph),
    )


@dataclass(frozen=True)
class _IteratorState:
    reference: Any
    capture: Optional[_ReplayCapture]
    max_rows: int
    block_index: int
    row_index: int
    offset: int
    closed: bool
    lock: threading.Lock


_ITERATOR_REGISTRY_LOCK = threading.Lock()
_ITERATOR_REGISTRY: Dict[int, Tuple[Any, ...]] = {}


def _iterator_cursor_truth(
    state: _IteratorState,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    if (
        type(state) is not _IteratorState
        or not isinstance(state.reference, weakref.ReferenceType)
        or type(state.max_rows) is not int
        or not 1 <= state.max_rows <= 256
        or type(state.block_index) is not int
        or state.block_index < 0
        or type(state.row_index) is not int
        or state.row_index < 0
        or type(state.offset) is not int
        or state.offset < 0
        or type(state.closed) is not bool
        or not isinstance(state.lock, type(threading.Lock()))
        or (state.capture is None) is not state.closed
        or (state.capture is not None and type(state.capture) is not _ReplayCapture)
    ):
        raise ConstraintProgramError("replay iterator cursor is malformed")
    return (
        (
            state.max_rows,
            state.block_index,
            state.row_index,
            state.offset,
            state.closed,
            state.capture is None,
        ),
        (
            state,
            state.reference,
            state.capture,
            state.lock,
        ),
    )


def _iterator_entry(value: Any) -> Tuple[Any, ...]:
    if type(value) is not _BatchIterator:
        raise ConstraintProgramError("replay iterator has the wrong exact type")
    with _ITERATOR_REGISTRY_LOCK:
        entry = _ITERATOR_REGISTRY.get(id(value))
    if (
        type(entry) is not tuple
        or len(entry) != 8
        or entry[0]() is not value
        or type(entry[1]) is not _IteratorState
        or entry[1].reference is not entry[0]
        or entry[1].lock is not entry[2]
    ):
        raise ConstraintProgramError("replay iterator is forged or stale")
    cursor_key, cursor_graph = _iterator_cursor_truth(entry[1])
    if entry[3] is None:
        capture_key, capture_graph = None, ()
        capture_binding_valid = (
            entry[1].closed
            and entry[1].capture is None
            and entry[4] is None
            and entry[5] == ()
        )
    else:
        capture_binding_valid = type(entry[3]) is _ReplayCapture
        if capture_binding_valid:
            capture_key, capture_graph = _replay_capture_truth(entry[3])
        else:
            capture_key, capture_graph = None, ()
    if (
        not capture_binding_valid
        or capture_key != entry[4]
        or len(capture_graph) != len(entry[5])
        or any(
            current is not sealed
            for current, sealed in zip(capture_graph, entry[5])
        )
        or cursor_key != entry[6]
        or len(cursor_graph) != len(entry[7])
        or any(
            current is not sealed
            for current, sealed in zip(cursor_graph, entry[7])
        )
        or (
            entry[1].capture is not None
            and entry[1].capture is not entry[3]
        )
    ):
        raise ConstraintProgramError("replay iterator graph was rebound")
    return entry


def _iterator_lock(value: Any) -> Any:
    if type(value) is not _BatchIterator:
        raise ConstraintProgramError("replay iterator has the wrong exact type")
    with _ITERATOR_REGISTRY_LOCK:
        entry = _ITERATOR_REGISTRY.get(id(value))
    if type(entry) is not tuple or len(entry) != 8 or entry[0]() is not value:
        raise ConstraintProgramError("replay iterator is forged or stale")
    return entry[2]


def _publish_iterator_state(value: Any, state: _IteratorState) -> None:
    cursor_key, cursor_graph = _iterator_cursor_truth(state)
    with _ITERATOR_REGISTRY_LOCK:
        entry = _ITERATOR_REGISTRY.get(id(value))
        if (
            type(entry) is not tuple
            or len(entry) != 8
            or entry[0]() is not value
            or entry[2] is not state.lock
        ):
            raise ConstraintProgramError("replay iterator publication lost its cursor")
        if state.capture is None:
            source_capture, capture_key, capture_graph = None, None, ()
        else:
            source_capture = entry[3]
            capture_key = entry[4]
            capture_graph = entry[5]
            if source_capture is not state.capture:
                raise ConstraintProgramError(
                    "replay iterator changed its captured source"
                )
        _ITERATOR_REGISTRY[id(value)] = (
            entry[0],
            state,
            entry[2],
            source_capture,
            capture_key,
            capture_graph,
            cursor_key,
            cursor_graph,
        )


def _iterator_state(value: Any) -> _IteratorState:
    return _iterator_entry(value)[1]


class _BatchIterator:
    """Registry-owned closeable replay cursor with no mutable public graph."""

    __slots__ = ("__weakref__",)

    def __init__(self, capture: _ReplayCapture, max_rows: int) -> None:
        if type(self) is not _BatchIterator or type(capture) is not _ReplayCapture:
            raise ConstraintProgramError("replay iterator must be created internally")
        object_id = id(self)

        def cleanup(reference: Any) -> None:
            for _attempt in range(4):
                try:
                    with _ITERATOR_REGISTRY_LOCK:
                        current = _ITERATOR_REGISTRY.get(object_id)
                        if (
                            type(current) is tuple
                            and len(current) == 8
                            and current[0] is reference
                        ):
                            _ITERATOR_REGISTRY.pop(object_id, None)
                    return
                except BaseException:
                    continue

        reference = weakref.ref(self, cleanup)
        state = _IteratorState(
            reference,
            capture,
            max_rows,
            0,
            0,
            0,
            False,
            threading.Lock(),
        )
        capture_key, capture_graph = _replay_capture_truth(capture)
        cursor_key, cursor_graph = _iterator_cursor_truth(state)
        entry = (
            reference,
            state,
            state.lock,
            capture,
            capture_key,
            capture_graph,
            cursor_key,
            cursor_graph,
        )
        with _ITERATOR_REGISTRY_LOCK:
            previous = _ITERATOR_REGISTRY.get(object_id)
            if (
                previous is not None
                and type(previous) is tuple
                and len(previous) == 8
                and previous[0]() is not None
            ):
                raise ConstraintProgramError("replay iterator registry ID was reused")
            _ITERATOR_REGISTRY[object_id] = entry

    def __iter__(self) -> "_BatchIterator":
        lock = _iterator_lock(self)
        with lock:
            _iterator_state(self)
        return self

    @property
    def closed(self) -> bool:
        lock = _iterator_lock(self)
        with lock:
            state = _iterator_state(self)
            return state.closed

    def close(self) -> None:
        lock = _iterator_lock(self)
        with lock:
            state = _iterator_state(self)
            working = state
            try:
                working = _iterator_close_locked(working)
                _publish_iterator_state(self, working)
            except BaseException:
                # Preserve the triggering exception, but repair any
                # instruction-boundary gap between cursor mutation and its
                # registry publication.
                for _attempt in range(4):
                    try:
                        current = _iterator_state(self)
                        closed = replace(current, capture=None, closed=True)
                        _publish_iterator_state(self, closed)
                        break
                    except BaseException:
                        continue
                raise

    def __enter__(self) -> "_BatchIterator":
        lock = _iterator_lock(self)
        with lock:
            _iterator_state(self)
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> bool:
        self.close()
        return False

    def __next__(self) -> Any:
        lock = _iterator_lock(self)
        with lock:
            state = _iterator_state(self)
            if state.closed or state.capture is None:
                raise StopIteration
            try:
                result, working = _iterator_next_locked(state)
                if (
                    working.capture is not None
                    and working.offset >= working.capture.total_rows
                ):
                    working = _iterator_close_locked(working)
                _publish_iterator_state(self, working)
                return result
            except BaseException:
                # This catches StopIteration, replay construction failures,
                # and asynchronous exceptions at every point from cursor
                # advance through end-close and registry publication.
                for _attempt in range(4):
                    try:
                        current = _iterator_state(self)
                        closed = replace(current, capture=None, closed=True)
                        _publish_iterator_state(self, closed)
                        break
                    except BaseException:
                        continue
                raise


def _iterator_close_locked(state: _IteratorState) -> _IteratorState:
    return replace(state, capture=None, closed=True)


def _iterator_next_locked(
    state: _IteratorState,
) -> Tuple[Any, _IteratorState]:
    capture = state.capture
    if capture is None or state.offset >= capture.total_rows:
        raise StopIteration
    block_index = state.block_index
    row_index = state.row_index
    offset = state.offset
    cont_rows = []
    bin_rows = []
    lower_bits = []
    upper_bits = []
    row_keys = []
    row_tags = []
    block_keys = []
    ordinals = []
    batch_offset = offset
    loaded_block_index = -1
    active_cont: Optional[sp.csr_matrix] = None
    active_bin: Optional[sp.csr_matrix] = None
    active_lower: Optional[np.ndarray] = None
    active_upper: Optional[np.ndarray] = None
    while (
        len(upper_bits) < state.max_rows
        and block_index < len(capture.occurrences)
    ):
        occurrence = capture.occurrences[block_index]
        limit = occurrence.source_rows if capture.native else occurrence.virtual_rows
        if row_index >= limit:
            block_index += 1
            row_index = 0
            continue
        if loaded_block_index != block_index:
            active_cont = occurrence.A_cont.csr(columns=len(capture.cont_keys))
            active_bin = occurrence.A_bin.csr(columns=len(capture.bin_keys))
            active_lower = occurrence.lower.array()
            active_upper = occurrence.upper.array()
            loaded_block_index = block_index
        Ac = active_cont
        Ab = active_bin
        lower = active_lower
        upper = active_upper
        if Ac is None or Ab is None or lower is None or upper is None:
            raise ConstraintProgramError("replay lost its captured block")
        logical = row_index
        if capture.native:
            position = logical
            flip = False
            from_lower = False
            row_key = occurrence.source_row_keys[logical]
            row_tag = occurrence.source_tags[logical]
        else:
            position = occurrence.legacy_positions[logical]
            flip = occurrence.legacy_flip[logical]
            from_lower = occurrence.legacy_bound_from_lower[logical]
            row_key = occurrence.legacy_row_keys[logical]
            row_tag = occurrence.legacy_tags[logical]
        ci, cd = _matrix_row(Ac, position)
        bi, bd = _matrix_row(Ab, position)
        cont_rows.append((ci.copy(), _flip_bits(cd) if flip else cd.copy()))
        bin_rows.append((bi.copy(), _flip_bits(bd) if flip else bd.copy()))
        if capture.native:
            lower_bits.append(int(lower.view(np.uint64)[position]))
            upper_bits.append(int(upper.view(np.uint64)[position]))
        else:
            bound = lower if from_lower else upper
            bound_bits = int(bound.view(np.uint64)[position])
            upper_bits.append(bound_bits ^ (1 << 63) if from_lower else bound_bits)
        row_keys.append(row_key)
        row_tags.append(row_tag)
        block_keys.append(occurrence.block_key)
        ordinals.append(occurrence.ordinal)
        row_index += 1
        offset += 1
    if not upper_bits:
        raise StopIteration
    cont = _FrozenCSR.from_value(
        _assemble_rows(cont_rows, columns=len(capture.cont_keys)),
        name="native replay A_cont" if capture.native else "legacy replay A_cont",
    )
    binary = _FrozenCSR.from_value(
        _assemble_rows(bin_rows, columns=len(capture.bin_keys)),
        name="native replay A_bin" if capture.native else "legacy replay A_bin",
    )
    upper_vector = _FrozenVector.from_bits(upper_bits, name="replay upper")
    lower_vector = (
        _FrozenVector.from_bits(lower_bits, name="replay lower")
        if capture.native
        else None
    )
    batch_state = _BatchState(
        cont,
        binary,
        lower_vector,
        upper_vector,
        tuple(row_keys),
        tuple(row_tags),
        tuple(block_keys),
        tuple(ordinals),
        capture.cont_keys,
        capture.bin_keys,
        batch_offset,
        capture.total_rows,
    )
    return (
        _new_batch(batch_state, native=capture.native),
        replace(
            state,
            block_index=block_index,
            row_index=row_index,
            offset=offset,
        ),
    )


def _make_iterator(
    program: ConstraintProgram, *, max_rows: int, native: bool
) -> _BatchIterator:
    if type(max_rows) is not int or not 1 <= max_rows <= 256:
        raise ConstraintProgramError("max_rows must be an exact builtin int in [1, 256]")
    state = _program_state(program)
    total = state.source_rows if native else state.virtual_rows
    capture = _ReplayCapture(
        tuple(_capture_occurrence(item) for item in state.occurrences),
        state.frame_cont_keys,
        state.frame_bin_keys,
        total,
        native,
    )
    return _BatchIterator(capture, max_rows)


def iter_native_batches(
    program: ConstraintProgram, *, max_rows: int
) -> Iterator[NativeConstraintBatch]:
    """Stream native RANGE/LE rows in global append order."""

    return _make_iterator(program, max_rows=max_rows, native=True)


def iter_legacy_facet_batches(
    program: ConstraintProgram, *, max_rows: int
) -> Iterator[LegacyFacetBatch]:
    """Stream independent legacy LE facets in exact block/row order."""

    return _make_iterator(program, max_rows=max_rows, native=False)


__all__ = (
    "ConstraintAppend",
    "ConstraintArena",
    "ConstraintArenaMismatch",
    "ConstraintFamily",
    "ConstraintObjectID",
    "ConstraintProgram",
    "ConstraintProgramError",
    "ConstraintProgramOwner",
    "ConstraintTransactionError",
    "ConstraintView",
    "ExternalAllocatorContractError",
    "ExternalFactorAllocatorAdapter",
    "ExternalFactorID",
    "FactorFrame",
    "FactorKind",
    "LegacyFacetBatch",
    "NativeConstraintBatch",
    "PreparedAppend",
    "iter_legacy_facet_batches",
    "iter_native_batches",
)
