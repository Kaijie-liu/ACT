#!/usr/bin/env python3
# ===- query_dual_replay_v51_session.py - Sealed V5.1 session -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Root-owned, fail-closed transaction around the V5.1 replay candidate.

This module deliberately does not construct, borrow, or reflect into a V3
``QueryDualReplaySession``.  It borrows only the process-local frozen graph
owned by a root :class:`QueryDualBoxCertificate`, builds independent sealed
cones and bounds frames, and drives the internal V5.1 replay observer.

The observer is the sole source of absorption evidence.  Every affine
execution immediately creates a frame-owned expectation and compact trace
from its live arrays.  Public receipts are never used to reconstruct a trace.
The resulting session remains a research candidate and can never issue proof
authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Sequence, Tuple

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51 as replay_v51
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv_v51
from act.back_end.hybridz_tf import query_dual_scalar_guard_v51 as scalar_v51
from act.back_end.hybridz_tf import query_dual_v51_authority as authority
from act.back_end.hybridz_tf.query_dual_blas_contract import (
    QueryDualBlasContract,
    QueryDualBlasContractError,
    validate_query_dual_blas_contract,
)
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    QueryDualBoxCertificate,
    QueryDualBoxError,
    _borrow_sealed_query_dual_graph,
    verify_query_dual_box_certificate,
)


SCHEMA = "act.query_dual_replay_v51_session_candidate.v1"
FRAME_SCHEMA = "act.query_dual_replay_v51_session_frame.v1"
STAGE_SCHEMA = "act.query_dual_replay_v51_session_stage.v1"
NUMERIC_PROTOCOL = "root_owned_compact_guard_transaction_v5_1"

_SESSION_CAPABILITY = object()
_RESULT_CAPABILITY = object()
_SESSION_PID = os.getpid()
_SESSION_REGISTRY: weakref.WeakValueDictionary[
    str, "QueryDualReplayV51Session"
] = weakref.WeakValueDictionary()
_SESSION_SEALS: Dict[str, "_SessionRuntime"] = {}
_SESSION_LOCK = threading.Lock()
_RESULT_REGISTRY: weakref.WeakValueDictionary[
    str, "QueryDualReplayV51SessionCandidateResult"
] = weakref.WeakValueDictionary()
_RESULT_SEALS: Dict[str, Tuple[int, int, str]] = {}
_RESULT_LOCK = threading.Lock()


class QueryDualReplayV51SessionError(RuntimeError):
    """Stable fail-closed error for the V5.1 transaction."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


class QueryDualReplayV51SessionTimeout(QueryDualReplayV51SessionError):
    """The one absolute session deadline expired."""

    def __init__(self, message: str = "V5.1 session deadline expired"):
        super().__init__("DEADLINE_EXPIRED", message)


def _fail(code: str, message: str) -> NoReturn:
    raise QueryDualReplayV51SessionError(code, message)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    return value


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _deep_freeze(item)
                for key, item in sorted(
                    value.items(), key=lambda pair: str(pair[0])
                )
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, name: str) -> str:
    if not _is_sha256(value):
        _fail("INVALID_BINDING", f"{name} must be a lowercase SHA-256")
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_manifest() -> Mapping[str, str]:
    paths = {
        "session": Path(__file__).resolve(),
        "replay_v51": Path(replay_v51.__file__).resolve(),
        "scalar_v51": Path(scalar_v51.__file__).resolve(),
        "conv_v51": Path(conv_v51.__file__).resolve(),
        "authority": Path(authority.__file__).resolve(),
    }
    return MappingProxyType(
        {
            name: _file_sha256(path)
            for name, path in sorted(paths.items())
        }
    )


def _capture_numeric_contract(
    blas_contract: QueryDualBlasContract,
) -> Mapping[str, Any]:
    """Capture only deterministic platform/source fields.

    The expensive live BLAS re-probe is performed separately at commit.
    """

    if not validate_query_dual_blas_contract(blas_contract):
        _fail("INVALID_BLAS_CONTRACT", "BLAS contract is invalid")
    try:
        frozen_platform = frozen._check_numeric_platform()
        scalar_platform = scalar_v51.check_v51_platform()
    except scalar_v51.QueryDualScalarGuardV51Error as exc:
        raise QueryDualReplayV51SessionError(
            exc.code, str(exc)
        ) from exc
    implementation = _implementation_manifest()
    body: Dict[str, Any] = {
        "schema": "act.query_dual_replay_v51_session_platform.v1",
        "frozen_v3": dict(frozen_platform),
        "v51_scalar_sha256": scalar_platform.sha256,
        "v51_scalar": dict(scalar_platform.items),
        "blas_contract_sha256": blas_contract.content_sha256,
        "required_blas_threads": int(blas_contract.required_threads),
        "implementation": dict(implementation),
    }
    body["implementation_sha256"] = _json_sha256(
        dict(implementation)
    )
    body["content_sha256"] = _json_sha256(body)
    return _deep_freeze(body)


def _drop_result_seal(nonce: str) -> None:
    if os.getpid() != _SESSION_PID:
        return
    with _RESULT_LOCK:
        _RESULT_SEALS.pop(nonce, None)


def _drop_session_seal(nonce: str) -> None:
    if os.getpid() != _SESSION_PID:
        return
    with _SESSION_LOCK:
        runtime = _SESSION_SEALS.pop(nonce, None)
    if runtime is None:
        return
    runtime.operation_lock.acquire()
    try:
        with runtime.state_lock:
            runtime.failed = True
            runtime.closed = True
        for frame in runtime.frames.values():
            frame._catalog.clear()
        runtime.frames.clear()
        runtime.frame_identity.clear()
        runtime.frame_seals.clear()
        runtime.pending.clear()
        runtime.pending_seals.clear()
    finally:
        runtime.operation_lock.release()


@dataclass(frozen=True)
class QueryDualReplayV51SessionPendingResult:
    """Provisional, non-authoritative value for one stage use."""

    lower_bounds: np.ndarray = field(repr=False, compare=False)
    stage_token: str
    proof_authority: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(self.lower_bounds)
        if (
            self.proof_authority is not False
            or values.dtype != np.float64
            or values.ndim != 1
            or values.flags.writeable
            or not np.all(np.isfinite(values))
            or not isinstance(self.stage_token, str)
            or not self.stage_token
        ):
            raise ValueError("malformed V5.1 pending candidate")


@dataclass(frozen=True)
class QueryDualReplayV51SessionCandidateResult:
    """Committed transaction result which permanently denies authority."""

    lower_bounds: np.ndarray = field(repr=False, compare=False)
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    _nonce: str = field(default="", repr=False, compare=False)
    _capability: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.lower_bounds)
        if (
            self.proof_authority is not False
            or values.dtype != np.float64
            or values.ndim != 1
            or values.flags.writeable
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("malformed V5.1 session candidate")


def validate_query_dual_replay_v51_session_candidate(
    value: Any,
) -> bool:
    """Validate a process-local session candidate, never proof authority."""

    try:
        if (
            os.getpid() != _SESSION_PID
            or not isinstance(
                value, QueryDualReplayV51SessionCandidateResult
            )
            or value.proof_authority is not False
            or value._capability is not _RESULT_CAPABILITY
            or type(value.receipt) is not MappingProxyType
        ):
            return False
        with _RESULT_LOCK:
            seal = _RESULT_SEALS.get(value._nonce)
            if (
                _RESULT_REGISTRY.get(value._nonce) is not value
                or seal is None
                or seal[0] != id(value.lower_bounds)
                or seal[1] != id(value.receipt)
            ):
                return False
        bounds = np.asarray(value.lower_bounds)
        body = dict(value.receipt)
        claimed = str(body.pop("receipt_sha256"))
        return bool(
            bounds.dtype == np.float64
            and bounds.ndim == 1
            and not bounds.flags.writeable
            and np.all(np.isfinite(bounds))
            and body.get("schema") == SCHEMA
            and body.get("proof_authority") is False
            and body.get("status") == "session_candidate"
            and body.get("numeric_protocol") == NUMERIC_PROTOCOL
            and body.get("lower_bounds_sha256")
            == frozen._array_digest(bounds)
            and hmac.compare_digest(_json_sha256(body), claimed)
            and hmac.compare_digest(seal[2], claimed)
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


@dataclass(frozen=True)
class _CatalogEntry:
    """One frame-local semantic alias and optional reusable material."""

    key_sha256: str
    stage_use_sha256: str
    cone_start_lid: Optional[int]
    layer_id: int
    predecessor_id: int
    operator_kind: str
    branch: str
    box_semantics: str
    source_lb_sha256: str
    source_ub_sha256: str
    weight_sha256: str
    geometry_sha256: str
    catalog_content_sha256: str
    support_content_sha256: str
    branch_evidence_sha256: str
    implementation_sha256: str
    alias: authority.SupportCatalogAlias
    material_kind: str
    material: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class QueryDualReplayV51BoundsFrame:
    """Opaque immutable bounds snapshot with a private mutable catalog."""

    _session_nonce: str = field(repr=False)
    _frame_nonce: str = field(repr=False)
    _bounds: Mapping[int, frozen._Box] = field(repr=False)
    _content_sha256: str = field(repr=False)
    _bounds_manifest_sha256: str = field(repr=False)
    _parent_chain_sha256: str = field(repr=False)
    _binding: authority.V51FrameBinding = field(repr=False)
    _owner: Any = field(repr=False, compare=False)
    _catalog: Dict[str, _CatalogEntry] = field(
        repr=False, compare=False
    )
    _capability: Any = field(repr=False, compare=False)

    @property
    def frame_content_sha256(self) -> str:
        return self._content_sha256

    @property
    def catalog_entry_count(self) -> int:
        return len(self._catalog)


@dataclass(frozen=True)
class _PendingStage:
    public: QueryDualReplayV51SessionPendingResult
    stage_use_index: int
    frame_nonce: str
    frame_sha256: str
    ledger: authority.CompactGuardLedgerCertificate
    stage_receipt: Mapping[str, Any]
    content_sha256: str


@dataclass
class _SessionRuntime:
    """Registry-external immutable identity plus transaction state."""

    session_id: int
    net: Any
    root_certificate: QueryDualBoxCertificate
    root_graph: Any
    full_layers: Mapping[int, frozen._FrozenLayer]
    contexts: Mapping[Optional[int], frozen._SealedCone]
    stage_uses: Tuple[authority.StageUse, ...]
    deadline: frozen._Deadline
    deadline_end: float
    blas_contract: QueryDualBlasContract
    numeric_contract: Mapping[str, Any]
    crosswalk: Mapping[str, Any]
    operation_lock: threading.Lock
    frame_capability: Any
    frames: Dict[str, QueryDualReplayV51BoundsFrame]
    frame_identity: Dict[str, Tuple[int, int, int, int]]
    pending: List[_PendingStage]
    static_identity: Tuple[Any, ...]
    stage_seals: Tuple[Tuple[int, str], ...]
    frame_seals: Dict[str, Tuple[Any, ...]] = field(
        default_factory=dict
    )
    pending_seals: List[Tuple[Any, ...]] = field(
        default_factory=list
    )
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False
    failed: bool = False
    poisoned: bool = False


def _stage_record(value: authority.StageUse) -> Mapping[str, Any]:
    return {
        "use_index": value.use_index,
        "stage_kind": value.stage_kind,
        "stage_index": value.stage_index,
        "target_relu_lid": value.target_relu_lid,
        "cone_start_lid": value.cone_start_lid,
        "stage_use_sha256": value.stage_use_sha256,
    }


def _start_record(value: Optional[int]) -> Any:
    return "ASSERT_PREDECESSOR" if value is None else int(value)


def _bool_array(value: Any, *, rows: int, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.bool_).reshape(-1)
    if result.shape != (rows,):
        _fail("INVALID_EXECUTION", f"{name} has the wrong row count")
    return result


def _geometry_body(layer: frozen._FrozenLayer) -> Mapping[str, Any]:
    if layer.kind == "DENSE":
        return {
            "operator": "DENSE",
            "weight_shape": list(layer.params["weight"].shape),
        }
    if layer.kind == "CONV2D":
        return {
            "operator": "CONV2D",
            "weight_shape": list(layer.params["weight"].shape),
            "input_shape": list(layer.params["input_shape"]),
            "output_shape": list(layer.params["output_shape"]),
            "stride": list(layer.params["stride"]),
            "padding": list(layer.params["padding"]),
            "dilation": list(layer.params["dilation"]),
            "groups": int(layer.params["groups"]),
        }
    _fail("INVALID_EXECUTION", "catalog requested for non-affine layer")


class QueryDualReplayV51Session:
    """One-shot root-owned V5.1 transaction."""

    def __init__(
        self,
        *,
        token: Any,
        net: Any,
        root_certificate: QueryDualBoxCertificate,
        root_graph: Any,
        full_layers: Mapping[int, frozen._FrozenLayer],
        contexts: Mapping[Optional[int], frozen._SealedCone],
        stage_uses: Tuple[authority.StageUse, ...],
        deadline: frozen._Deadline,
        blas_contract: QueryDualBlasContract,
        numeric_contract: Mapping[str, Any],
    ):
        if token is not _SESSION_CAPABILITY:
            _fail("INVALID_SESSION", "missing V5.1 session capability")
        self._net = net
        self._root_certificate = root_certificate
        self._root_graph = root_graph
        self._full_layers = MappingProxyType(dict(full_layers))
        self._contexts = MappingProxyType(dict(contexts))
        self._stage_uses = tuple(stage_uses)
        self._deadline = deadline
        self._deadline_identity = id(deadline)
        self._deadline_end = deadline.end
        self._blas_contract = blas_contract
        self._numeric_contract = numeric_contract
        self._nonce = secrets.token_hex(32)
        self._frame_capability = object()
        self._frames: Dict[str, QueryDualReplayV51BoundsFrame] = {}
        self._frame_identity: Dict[str, Tuple[int, int, int, int]] = {}
        self._pending: List[_PendingStage] = []
        self._operation_lock = threading.Lock()
        self._closed = False
        self._failed = False
        self._crosswalk = MappingProxyType(
            frozen._sealed_crosswalk(
                root_certificate, root_graph, self._contexts
            )
        )
        self._crosswalk_sha256 = frozen._json_digest(
            dict(self._crosswalk)
        )
        self._required_bounds_ids = frozenset(
            lid
            for cone in self._contexts.values()
            for lid in cone.reverse_order
            if cone.layers[lid].kind != "INPUT"
        )
        self._static_identity = (
            id(self._root_certificate),
            id(self._root_graph),
            id(self._full_layers),
            id(self._contexts),
            id(self._stage_uses),
            id(self._deadline),
            id(self._blas_contract),
            id(self._numeric_contract),
            id(self._crosswalk),
            id(self._operation_lock),
            tuple(
                (key, id(cone), id(cone.layers))
                for key, cone in self._contexts.items()
            ),
        )
        self._static_manifest_commit_validations = 0
        self._catalog_builds = 0
        self._catalog_hits = 0
        runtime = _SessionRuntime(
            session_id=id(self),
            net=self._net,
            root_certificate=self._root_certificate,
            root_graph=self._root_graph,
            full_layers=self._full_layers,
            contexts=self._contexts,
            stage_uses=self._stage_uses,
            deadline=self._deadline,
            deadline_end=float(self._deadline_end),
            blas_contract=self._blas_contract,
            numeric_contract=self._numeric_contract,
            crosswalk=self._crosswalk,
            operation_lock=self._operation_lock,
            frame_capability=self._frame_capability,
            frames=self._frames,
            frame_identity=self._frame_identity,
            pending=self._pending,
            static_identity=self._static_identity,
            stage_seals=tuple(
                (id(value), value.stage_use_sha256)
                for value in self._stage_uses
            ),
        )
        with _SESSION_LOCK:
            _SESSION_REGISTRY[self._nonce] = self
            _SESSION_SEALS[self._nonce] = runtime
        weakref.finalize(self, _drop_session_seal, self._nonce)
        try:
            self._deadline.check(force=True)
        except frozen.QueryDualReplayTimeout as exc:
            with runtime.state_lock:
                runtime.failed = True
                runtime.closed = True
            with _SESSION_LOCK:
                _SESSION_REGISTRY.pop(self._nonce, None)
                _SESSION_SEALS.pop(self._nonce, None)
            raise QueryDualReplayV51SessionTimeout() from exc

    @property
    def unique_context_count(self) -> int:
        return len(self._contexts)

    @property
    def static_manifest_commit_validations(self) -> int:
        return self._static_manifest_commit_validations

    @property
    def catalog_build_count(self) -> int:
        return self._catalog_builds

    @property
    def catalog_hit_count(self) -> int:
        return self._catalog_hits

    @staticmethod
    def _pending_seal(value: _PendingStage) -> Tuple[Any, ...]:
        return (
            id(value),
            id(value.public),
            id(value.public.lower_bounds),
            value.public.stage_token,
            value.stage_use_index,
            value.frame_nonce,
            value.frame_sha256,
            id(value.ledger),
            value.ledger.content_sha256,
            id(value.stage_receipt),
            value.stage_receipt["receipt_sha256"],
            value.content_sha256,
        )

    @staticmethod
    def _frame_seal(
        value: QueryDualReplayV51BoundsFrame,
    ) -> Tuple[Any, ...]:
        return (
            id(value),
            value._session_nonce,
            value._frame_nonce,
            id(value._bounds),
            value._content_sha256,
            value._bounds_manifest_sha256,
            value._parent_chain_sha256,
            id(value._binding),
            value._binding.binding_sha256,
            id(value._owner),
            id(value._catalog),
            id(value._capability),
        )

    def _raw_runtime(
        self, *, require_registered: bool = True
    ) -> _SessionRuntime:
        if os.getpid() != _SESSION_PID:
            _fail(
                "PROCESS_MISMATCH",
                "V5.1 session cannot be inherited across fork",
            )
        try:
            nonce = self._nonce
            with _SESSION_LOCK:
                runtime = _SESSION_SEALS.get(nonce)
                registered = _SESSION_REGISTRY.get(nonce)
        except (AttributeError, TypeError) as exc:
            raise QueryDualReplayV51SessionError(
                "INVALID_SESSION", "session registry lookup failed"
            ) from exc
        if (
            runtime is None
            or runtime.session_id != id(self)
            or (require_registered and registered is not self)
            or (
                not require_registered
                and registered is not None
                and registered is not self
            )
        ):
            _fail("INVALID_SESSION", "session registry identity changed")
        return runtime

    def _runtime(
        self, *, require_registered: bool = True
    ) -> _SessionRuntime:
        try:
            runtime = self._raw_runtime(
                require_registered=require_registered
            )
            if (
                self._net is not runtime.net
                or self._root_certificate is not runtime.root_certificate
                or self._root_graph is not runtime.root_graph
                or self._full_layers is not runtime.full_layers
                or self._contexts is not runtime.contexts
                or self._stage_uses is not runtime.stage_uses
                or self._deadline is not runtime.deadline
                or self._deadline_end != runtime.deadline_end
                or self._deadline.end != runtime.deadline_end
                or self._blas_contract is not runtime.blas_contract
                or self._numeric_contract is not runtime.numeric_contract
                or self._crosswalk is not runtime.crosswalk
                or self._operation_lock is not runtime.operation_lock
                or self._frame_capability is not runtime.frame_capability
                or self._frames is not runtime.frames
                or self._frame_identity is not runtime.frame_identity
                or self._pending is not runtime.pending
                or self._static_identity != runtime.static_identity
                or tuple(
                    (id(value), value.stage_use_sha256)
                    for value in runtime.stage_uses
                )
                != runtime.stage_seals
            ):
                _fail(
                    "INVALID_SESSION",
                    "session identity or immutable seal changed",
                )
            if (
                set(runtime.frames) != set(runtime.frame_identity)
                or set(runtime.frames) != set(runtime.frame_seals)
            ):
                _fail("INVALID_FRAME", "frame registry changed")
            for nonce_value, frame in runtime.frames.items():
                if (
                    runtime.frame_identity.get(nonce_value)
                    != (
                        id(frame._bounds),
                        id(frame._binding),
                        id(frame._owner),
                        id(frame._catalog),
                    )
                    or runtime.frame_seals.get(nonce_value)
                    != self._frame_seal(frame)
                ):
                    _fail("INVALID_FRAME", "frame identity seal changed")
            if (
                len(runtime.pending) != len(runtime.pending_seals)
                or [
                    self._pending_seal(value)
                    for value in runtime.pending
                ]
                != runtime.pending_seals
            ):
                _fail("INVALID_STAGE", "pending stage seal changed")
            return runtime
        except QueryDualReplayV51SessionError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise QueryDualReplayV51SessionError(
                "INVALID_SESSION", "session seal validation failed"
            ) from exc

    def _invalidate(
        self, runtime: Optional[_SessionRuntime] = None
    ) -> None:
        if runtime is None:
            try:
                runtime = self._raw_runtime(require_registered=False)
            except QueryDualReplayV51SessionError:
                return
        with runtime.state_lock:
            runtime.failed = True
            runtime.closed = True
        self._failed = True
        self._closed = True
        with _SESSION_LOCK:
            if _SESSION_REGISTRY.get(self._nonce) is self:
                _SESSION_REGISTRY.pop(self._nonce, None)

    def _enter(self) -> _SessionRuntime:
        runtime = self._raw_runtime()
        if not runtime.operation_lock.acquire(blocking=False):
            with runtime.state_lock:
                runtime.poisoned = True
            self._invalidate(runtime)
            _fail(
                "CONCURRENT_SESSION",
                "concurrent V5.1 session use is forbidden",
            )
        try:
            checked = self._runtime()
            if checked is not runtime:
                _fail(
                    "INVALID_SESSION",
                    "session runtime identity changed during entry",
                )
        except Exception:
            self._invalidate(runtime)
            runtime.operation_lock.release()
            raise
        return runtime

    def _check(
        self, runtime: Optional[_SessionRuntime] = None
    ) -> _SessionRuntime:
        checked = self._runtime()
        if runtime is not None and checked is not runtime:
            _fail("INVALID_SESSION", "session runtime identity changed")
        runtime = checked
        with runtime.state_lock:
            invalid_state = (
                runtime.closed or runtime.failed or runtime.poisoned
            )
        if invalid_state:
            _fail("INVALID_SESSION", "V5.1 session is closed")
        if (
            runtime.deadline.end != runtime.deadline_end
            or runtime.deadline.end is None
        ):
            self._invalidate(runtime)
            _fail("INVALID_DEADLINE", "absolute deadline seal changed")
        try:
            runtime.deadline.check(force=True)
        except frozen.QueryDualReplayTimeout as exc:
            self._invalidate(runtime)
            raise QueryDualReplayV51SessionTimeout() from exc
        identity = (
            id(self._root_certificate),
            id(self._root_graph),
            id(self._full_layers),
            id(self._contexts),
            id(self._stage_uses),
            id(self._deadline),
            id(self._blas_contract),
            id(self._numeric_contract),
            id(self._crosswalk),
            id(self._operation_lock),
            tuple(
                (key, id(cone), id(cone.layers))
                for key, cone in self._contexts.items()
            ),
        )
        if identity != runtime.static_identity:
            self._invalidate(runtime)
            _fail("INVALID_CONTEXT", "static session identity changed")
        return runtime

    def abort(self) -> None:
        """Idempotently release provisional catalog and ledger references."""

        runtime = self._raw_runtime(require_registered=False)
        runtime.operation_lock.acquire()
        try:
            self._invalidate(runtime)
            for frame in runtime.frames.values():
                frame._catalog.clear()
            runtime.frames.clear()
            runtime.frame_identity.clear()
            runtime.frame_seals.clear()
            runtime.pending.clear()
            runtime.pending_seals.clear()
        finally:
            runtime.operation_lock.release()

    def _frame_body(
        self,
        frame: QueryDualReplayV51BoundsFrame,
    ) -> Mapping[str, Any]:
        records = frozen._bounds_manifest(
            self._full_layers, frame._bounds
        )
        return {
            "schema": FRAME_SCHEMA,
            "session_nonce_sha256": hashlib.sha256(
                self._nonce.encode("ascii")
            ).hexdigest(),
            "frame_nonce_sha256": hashlib.sha256(
                frame._frame_nonce.encode("ascii")
            ).hexdigest(),
            "stage_uses": [
                dict(_stage_record(value))
                for value in self._stage_uses
            ],
            "bounds": records,
            "bounds_manifest_sha256": frozen._json_digest(records),
            "parent_chain_sha256": frame._parent_chain_sha256,
            "root_receipt_sha256": str(
                self._root_certificate.receipt["receipt_sha256"]
            ),
        }

    def _check_frame(
        self,
        frame: QueryDualReplayV51BoundsFrame,
        *,
        full: bool = False,
    ) -> None:
        runtime = self._runtime()
        identity = runtime.frame_identity.get(
            getattr(frame, "_frame_nonce", "")
        )
        if (
            not isinstance(frame, QueryDualReplayV51BoundsFrame)
            or frame._session_nonce != self._nonce
            or frame._capability is not self._frame_capability
            or runtime.frames.get(frame._frame_nonce) is not frame
            or identity
            != (
                id(frame._bounds),
                id(frame._binding),
                id(frame._owner),
                id(frame._catalog),
            )
            or runtime.frame_seals.get(frame._frame_nonce)
            != self._frame_seal(frame)
            or not authority.validate_frame_binding(frame._binding)
            or not authority._verify_owner(frame._owner)
            or frame._owner.binding is not frame._binding
        ):
            _fail("INVALID_FRAME", "frame is not owned by this session")
        if not full:
            return
        body = self._frame_body(frame)
        bounds_sha = str(body["bounds_manifest_sha256"])
        content_sha = _json_sha256(body)
        if (
            not hmac.compare_digest(
                bounds_sha, frame._bounds_manifest_sha256
            )
            or not hmac.compare_digest(
                content_sha, frame._content_sha256
            )
            or frame._binding.frame_content_sha256
            != frame._content_sha256
            or frame._binding.bounds_manifest_sha256
            != frame._bounds_manifest_sha256
            or frame._binding.parent_chain_sha256
            != frame._parent_chain_sha256
            or frame._binding.stage_uses != self._stage_uses
            or frame._binding.session_nonce_sha256
            != hashlib.sha256(
                self._nonce.encode("ascii")
            ).hexdigest()
            or frame._binding.frame_nonce_sha256
            != hashlib.sha256(
                frame._frame_nonce.encode("ascii")
            ).hexdigest()
            or frame._binding.deadline_monotonic_hex
            != float(self._deadline.end).hex()
        ):
            _fail("INVALID_FRAME", "frame content seal changed")
        for key, entry in tuple(frame._catalog.items()):
            self._validate_catalog_entry(frame, key, entry)

    def seal_bounds(
        self,
        certified_bounds: Mapping[Any, Any],
        *,
        parent_chain_sha256: Optional[str] = None,
    ) -> QueryDualReplayV51BoundsFrame:
        """Seal one union frame shared by all four target uses and property."""

        runtime = self._enter()
        try:
            self._check(runtime)
            if not isinstance(certified_bounds, Mapping):
                _fail(
                    "INVALID_BOUNDS",
                    "certified_bounds must be a mapping",
                )
            parent_sha = (
                str(self._root_certificate.receipt["receipt_sha256"])
                if parent_chain_sha256 is None
                else _require_sha256(
                    parent_chain_sha256,
                    name="parent_chain_sha256",
                )
            )
            raw: Dict[int, Any] = {}
            for key, value in certified_bounds.items():
                if isinstance(key, bool):
                    _fail("INVALID_BOUNDS", "boolean bounds key")
                try:
                    lid = int(key)
                except Exception as exc:
                    raise QueryDualReplayV51SessionError(
                        "INVALID_BOUNDS", f"invalid bounds key {key!r}"
                    ) from exc
                if lid in raw or lid not in self._full_layers:
                    _fail(
                        "INVALID_BOUNDS",
                        f"duplicate or unknown bounds layer {lid}",
                    )
                raw[lid] = value
            missing = self._required_bounds_ids - set(raw)
            if missing:
                _fail(
                    "MISSING_BOUNDS",
                    f"frame lacks consumed layers {sorted(missing)}",
                )
            boxes: Dict[int, frozen._Box] = {}
            for lid in sorted(self._required_bounds_ids):
                self._deadline.check()
                box = frozen._immutable_box_from_value(
                    raw[lid], layer_id=lid
                )
                if box.lb.size != self._full_layers[lid].width:
                    _fail(
                        "SHAPE_MISMATCH",
                        f"bounds[{lid}] width mismatch",
                    )
                boxes[lid] = box
            nonce = secrets.token_hex(32)
            placeholder = QueryDualReplayV51BoundsFrame(
                _session_nonce=self._nonce,
                _frame_nonce=nonce,
                _bounds=MappingProxyType(boxes),
                _content_sha256="",
                _bounds_manifest_sha256="",
                _parent_chain_sha256=parent_sha,
                _binding=None,  # type: ignore[arg-type]
                _owner=None,
                _catalog={},
                _capability=self._frame_capability,
            )
            body = self._frame_body(placeholder)
            bounds_sha = str(body["bounds_manifest_sha256"])
            content_sha = _json_sha256(body)
            binding = authority.V51FrameBinding(
                session_nonce_sha256=hashlib.sha256(
                    self._nonce.encode("ascii")
                ).hexdigest(),
                frame_nonce_sha256=hashlib.sha256(
                    nonce.encode("ascii")
                ).hexdigest(),
                frame_content_sha256=content_sha,
                bounds_manifest_sha256=bounds_sha,
                root_receipt_sha256=str(
                    self._root_certificate.receipt["receipt_sha256"]
                ),
                parent_chain_sha256=parent_sha,
                deadline_monotonic_hex=float(
                    self._deadline.end
                ).hex(),
                stage_uses=self._stage_uses,
            )
            owner = authority._mint_frame_owner(binding)
            frame = QueryDualReplayV51BoundsFrame(
                _session_nonce=self._nonce,
                _frame_nonce=nonce,
                _bounds=placeholder._bounds,
                _content_sha256=content_sha,
                _bounds_manifest_sha256=bounds_sha,
                _parent_chain_sha256=parent_sha,
                _binding=binding,
                _owner=owner,
                _catalog={},
                _capability=self._frame_capability,
            )
            self._frames[nonce] = frame
            self._frame_identity[nonce] = (
                id(frame._bounds),
                id(frame._binding),
                id(frame._owner),
                id(frame._catalog),
            )
            runtime.frame_seals[nonce] = self._frame_seal(frame)
            self._check(runtime)
            return frame
        except frozen.QueryDualReplayTimeout as exc:
            self._invalidate(runtime)
            raise QueryDualReplayV51SessionTimeout() from exc
        except frozen.QueryDualReplayError as exc:
            self._invalidate(runtime)
            raise QueryDualReplayV51SessionError(
                exc.code, str(exc)
            ) from exc
        except authority.QueryDualV51AuthorityError as exc:
            self._invalidate(runtime)
            if exc.code == "DEADLINE_EXPIRED":
                raise QueryDualReplayV51SessionTimeout(str(exc)) from exc
            raise QueryDualReplayV51SessionError(
                exc.code, str(exc)
            ) from exc
        except Exception:
            self._invalidate(runtime)
            raise
        finally:
            runtime.operation_lock.release()

    def _catalog_description(
        self,
        *,
        frame: QueryDualReplayV51BoundsFrame,
        stage_use: authority.StageUse,
        prepared: frozen._Prepared,
        layer: frozen._FrozenLayer,
        predecessor_id: int,
        branch: str,
        material: Any,
    ) -> Mapping[str, Any]:
        source = frozen._output_box(prepared, predecessor_id)
        semantics = (
            "relu_postactivation_from_preactivation_box_v1"
            if prepared.layers[predecessor_id].kind == "RELU"
            else "output"
        )
        weight_sha = frozen._array_digest(layer.params["weight"])
        geometry_sha = _json_sha256(_geometry_body(layer))
        source_lb_sha = frozen._array_digest(source.lb)
        source_ub_sha = frozen._array_digest(source.ub)
        implementation_sha = str(
            self._numeric_contract["implementation_sha256"]
        )
        branch_evidence_sha = _json_sha256(
            {
                "operator": layer.kind,
                "branch": branch,
                "decision": (
                    "8*count_nonzero<=coefficient_size"
                    if layer.kind == "CONV2D"
                    else "dense_only"
                ),
            }
        )
        if branch == authority.BRANCH_DENSE:
            if not isinstance(material, scalar_v51.DenseV51Support):
                _fail("INVALID_CATALOG", "Dense support is absent")
            catalog_sha = material.diagnostics.sha256
            support_sha = material.support_sha256
            material_kind = "DENSE_SUPPORT"
        elif branch == authority.BRANCH_CONV_DENSE:
            if not isinstance(material, conv_v51.DenseConvV51Plan):
                _fail("INVALID_CATALOG", "dense Conv plan is absent")
            catalog_sha = str(material.manifest["content_sha256"])
            support_sha = frozen._array_digest(material.support)
            material_kind = "CONV_PLAN"
        elif branch == authority.BRANCH_CONV_SPARSE:
            static = {
                "operator": "CONV2D",
                "branch": "frozen_v3_componentwise",
                "weight_sha256": weight_sha,
                "geometry_sha256": geometry_sha,
                "source_lb_sha256": source_lb_sha,
                "source_ub_sha256": source_ub_sha,
            }
            catalog_sha = _json_sha256(
                {"catalog": static, "frame": frame._content_sha256}
            )
            support_sha = _json_sha256(
                {"componentwise_support": static}
            )
            material_kind = "SPARSE_STATIC"
            material = None
        else:
            _fail("INVALID_BRANCH", f"unknown branch {branch}")
        return {
            "stage_use_sha256": stage_use.stage_use_sha256,
            "cone_start_lid": stage_use.cone_start_lid,
            "layer_id": int(layer.id),
            "predecessor_id": int(predecessor_id),
            "operator_kind": layer.kind,
            "branch": branch,
            "box_semantics": semantics,
            "source_lb_sha256": source_lb_sha,
            "source_ub_sha256": source_ub_sha,
            "weight_sha256": weight_sha,
            "geometry_sha256": geometry_sha,
            "catalog_content_sha256": catalog_sha,
            "support_content_sha256": support_sha,
            "branch_evidence_sha256": branch_evidence_sha,
            "implementation_sha256": implementation_sha,
            "material_kind": material_kind,
            "material": material,
        }

    def _ensure_catalog_entry(
        self,
        *,
        frame: QueryDualReplayV51BoundsFrame,
        stage_use: authority.StageUse,
        prepared: frozen._Prepared,
        layer: frozen._FrozenLayer,
        predecessor_id: int,
        branch: str,
        material: Any,
    ) -> _CatalogEntry:
        description = dict(
            self._catalog_description(
                frame=frame,
                stage_use=stage_use,
                prepared=prepared,
                layer=layer,
                predecessor_id=predecessor_id,
                branch=branch,
                material=material,
            )
        )
        material_value = description.pop("material")
        material_kind = str(description.pop("material_kind"))
        key_body = {
            "frame_content_sha256": frame._content_sha256,
            "numeric_platform_sha256": self._numeric_contract[
                "content_sha256"
            ],
            **description,
        }
        key_sha = _json_sha256(key_body)
        cached = frame._catalog.get(key_sha)
        if cached is not None:
            self._validate_catalog_entry(frame, key_sha, cached)
            self._catalog_hits += 1
            return cached
        alias = authority._mint_support_catalog_alias(
            frame._owner,
            stage_use_sha256=str(
                description["stage_use_sha256"]
            ),
            layer_id=int(description["layer_id"]),
            predecessor_id=int(description["predecessor_id"]),
            operator_kind=str(description["operator_kind"]),
            branch=str(description["branch"]),
            box_semantics=str(description["box_semantics"]),
            catalog_content_sha256=str(
                description["catalog_content_sha256"]
            ),
            support_content_sha256=str(
                description["support_content_sha256"]
            ),
            weight_sha256=str(description["weight_sha256"]),
            geometry_sha256=str(description["geometry_sha256"]),
            source_lb_sha256=str(
                description["source_lb_sha256"]
            ),
            source_ub_sha256=str(
                description["source_ub_sha256"]
            ),
            numeric_platform_sha256=str(
                self._numeric_contract["content_sha256"]
            ),
            implementation_sha256=str(
                description["implementation_sha256"]
            ),
            branch_evidence_sha256=str(
                description["branch_evidence_sha256"]
            ),
        )
        entry = _CatalogEntry(
            key_sha256=key_sha,
            alias=alias,
            material_kind=material_kind,
            material=material_value,
            **description,
        )
        frame._catalog[key_sha] = entry
        self._catalog_builds += 1
        return entry

    def _validate_catalog_entry(
        self,
        frame: QueryDualReplayV51BoundsFrame,
        key: str,
        entry: _CatalogEntry,
    ) -> None:
        if (
            not isinstance(entry, _CatalogEntry)
            or key != entry.key_sha256
            or not authority.validate_support_catalog_alias(entry.alias)
            or entry.alias._owner is not frame._owner
        ):
            _fail("INVALID_CATALOG", "catalog alias lost frame ownership")
        description = {
            name: getattr(entry, name)
            for name in (
                "stage_use_sha256",
                "cone_start_lid",
                "layer_id",
                "predecessor_id",
                "operator_kind",
                "branch",
                "box_semantics",
                "source_lb_sha256",
                "source_ub_sha256",
                "weight_sha256",
                "geometry_sha256",
                "catalog_content_sha256",
                "support_content_sha256",
                "branch_evidence_sha256",
                "implementation_sha256",
            )
        }
        expected_key = _json_sha256(
            {
                "frame_content_sha256": frame._content_sha256,
                "numeric_platform_sha256": self._numeric_contract[
                    "content_sha256"
                ],
                **description,
            }
        )
        alias = entry.alias
        if (
            not hmac.compare_digest(expected_key, key)
            or alias.stage_use_sha256 != entry.stage_use_sha256
            or alias.cone_start_lid != entry.cone_start_lid
            or alias.layer_id != entry.layer_id
            or alias.predecessor_id != entry.predecessor_id
            or alias.operator_kind != entry.operator_kind
            or alias.branch != entry.branch
            or alias.box_semantics != entry.box_semantics
            or alias.catalog_content_sha256
            != entry.catalog_content_sha256
            or alias.support_content_sha256
            != entry.support_content_sha256
            or alias.weight_sha256 != entry.weight_sha256
            or alias.geometry_sha256 != entry.geometry_sha256
            or alias.source_lb_sha256 != entry.source_lb_sha256
            or alias.source_ub_sha256 != entry.source_ub_sha256
            or alias.numeric_platform_sha256
            != self._numeric_contract["content_sha256"]
            or alias.implementation_sha256
            != entry.implementation_sha256
            or alias.branch_evidence_sha256
            != entry.branch_evidence_sha256
        ):
            _fail("INVALID_CATALOG", "catalog semantic key changed")
        stage_use = next(
            (
                value
                for value in self._stage_uses
                if value.stage_use_sha256
                == entry.stage_use_sha256
            ),
            None,
        )
        cone = (
            None
            if stage_use is None
            else self._contexts.get(stage_use.cone_start_lid)
        )
        layer = self._full_layers.get(entry.layer_id)
        if (
            stage_use is None
            or cone is None
            or stage_use.cone_start_lid != entry.cone_start_lid
            or entry.layer_id not in cone.layers
            or layer is None
            or layer.kind != entry.operator_kind
            or layer.preds != (entry.predecessor_id,)
            or entry.predecessor_id not in cone.layers
            or frozen._array_digest(layer.params["weight"])
            != entry.weight_sha256
            or _json_sha256(_geometry_body(layer))
            != entry.geometry_sha256
        ):
            _fail("INVALID_CATALOG", "catalog layer binding changed")
        predecessor = cone.layers[entry.predecessor_id]
        raw_source = frame._bounds.get(entry.predecessor_id)
        if raw_source is None:
            _fail("INVALID_CATALOG", "catalog source box disappeared")
        if predecessor.kind == "RELU":
            source_lb = np.maximum(raw_source.lb, 0.0)
            source_ub = np.maximum(raw_source.ub, 0.0)
            semantics = (
                "relu_postactivation_from_preactivation_box_v1"
            )
        else:
            source_lb = raw_source.lb
            source_ub = raw_source.ub
            semantics = "output"
        source_lb = np.ascontiguousarray(source_lb, dtype=np.float64)
        source_ub = np.ascontiguousarray(source_ub, dtype=np.float64)
        expected_branch_evidence = _json_sha256(
            {
                "operator": layer.kind,
                "branch": entry.branch,
                "decision": (
                    "8*count_nonzero<=coefficient_size"
                    if layer.kind == "CONV2D"
                    else "dense_only"
                ),
            }
        )
        if (
            entry.box_semantics != semantics
            or entry.source_lb_sha256
            != frozen._array_digest(source_lb)
            or entry.source_ub_sha256
            != frozen._array_digest(source_ub)
            or entry.branch_evidence_sha256
            != expected_branch_evidence
            or entry.implementation_sha256
            != self._numeric_contract["implementation_sha256"]
        ):
            _fail(
                "INVALID_CATALOG",
                "catalog source/branch/implementation binding changed",
            )
        max_abs = np.ascontiguousarray(
            np.maximum(np.abs(source_lb), np.abs(source_ub)),
            dtype=np.float64,
        )
        if entry.material_kind == "DENSE_SUPPORT":
            support = entry.material
            try:
                scalar_v51._validate_support(
                    support,
                    layer.params["weight"],
                    platform_sha256=scalar_v51.check_v51_platform().sha256,
                )
            except scalar_v51.QueryDualScalarGuardV51Error as exc:
                raise QueryDualReplayV51SessionError(
                    "INVALID_CATALOG", str(exc)
                ) from exc
            if (
                support.support_sha256
                != entry.support_content_sha256
                or support.diagnostics.sha256
                != entry.catalog_content_sha256
                or support.max_abs_sha256
                != scalar_v51._array_sha256(max_abs)
                or support.binding
                != scalar_v51._canonical_binding(
                    {
                        "numeric_protocol": replay_v51.NUMERIC_PROTOCOL,
                        "net_sha256": cone.replay_net_sha256,
                        "bounds_sha256": frozen._json_digest(
                            frozen._bounds_manifest(
                                cone.layers,
                                {
                                    lid: frame._bounds[lid]
                                    for lid in cone.reverse_order
                                    if cone.layers[lid].kind != "INPUT"
                                },
                            )
                        ),
                        "layer_id": str(layer.id),
                        "predecessor_id": str(
                            entry.predecessor_id
                        ),
                        "box_semantics": (
                            "relu_postactivation_from_preactivation_box_v1"
                            if predecessor.kind == "RELU"
                            else "output_box"
                        ),
                    }
                )
            ):
                _fail("INVALID_CATALOG", "Dense support content changed")
        elif entry.material_kind == "CONV_PLAN":
            try:
                conv_v51._validate_plan(
                    entry.material, deadline=self._deadline
                )
            except frozen.QueryDualReplayTimeout as exc:
                raise QueryDualReplayV51SessionTimeout() from exc
            except frozen.QueryDualReplayError as exc:
                raise QueryDualReplayV51SessionError(
                    "INVALID_CATALOG", str(exc)
                ) from exc
            if (
                frozen._array_digest(entry.material.support)
                != entry.support_content_sha256
                or str(entry.material.manifest["content_sha256"])
                != entry.catalog_content_sha256
                or entry.material.layer_id != layer.id
                or frozen._array_digest(entry.material.weight)
                != frozen._array_digest(layer.params["weight"])
                or frozen._array_digest(entry.material.support)
                != frozen._array_digest(max_abs)
                or _geometry_body(layer)
                != {
                    "operator": "CONV2D",
                    "weight_shape": list(
                        entry.material.weight.shape
                    ),
                    "input_shape": list(
                        entry.material.input_shape
                    ),
                    "output_shape": list(
                        entry.material.output_shape
                    ),
                    "stride": list(entry.material.stride),
                    "padding": list(entry.material.padding),
                    "dilation": list(entry.material.dilation),
                    "groups": int(entry.material.groups),
                }
            ):
                _fail("INVALID_CATALOG", "Conv plan content changed")
        elif entry.material_kind == "SPARSE_STATIC":
            static = {
                "operator": "CONV2D",
                "branch": "frozen_v3_componentwise",
                "weight_sha256": entry.weight_sha256,
                "geometry_sha256": entry.geometry_sha256,
                "source_lb_sha256": entry.source_lb_sha256,
                "source_ub_sha256": entry.source_ub_sha256,
            }
            if (
                entry.material is not None
                or entry.catalog_content_sha256
                != _json_sha256(
                    {
                        "catalog": static,
                        "frame": frame._content_sha256,
                    }
                )
                or entry.support_content_sha256
                != _json_sha256(
                    {"componentwise_support": static}
                )
            ):
                _fail(
                    "INVALID_CATALOG",
                    "sparse catalog content changed",
                )
        else:
            _fail("INVALID_CATALOG", "unknown catalog material")

    def _preload_catalog(
        self,
        frame: QueryDualReplayV51BoundsFrame,
        stage_use: authority.StageUse,
    ) -> Tuple[
        Dict[int, scalar_v51.DenseV51Support],
        Dict[int, conv_v51.DenseConvV51Plan],
    ]:
        dense: Dict[int, scalar_v51.DenseV51Support] = {}
        conv: Dict[int, conv_v51.DenseConvV51Plan] = {}
        for key, entry in tuple(frame._catalog.items()):
            if not isinstance(entry, _CatalogEntry):
                _fail("INVALID_CATALOG", "invalid catalog entry type")
            if entry.stage_use_sha256 != stage_use.stage_use_sha256:
                continue
            self._validate_catalog_entry(frame, key, entry)
            if entry.material_kind == "DENSE_SUPPORT":
                if entry.layer_id in dense:
                    _fail("INVALID_CATALOG", "duplicate Dense support")
                dense[entry.layer_id] = entry.material
            elif entry.material_kind == "CONV_PLAN":
                if entry.layer_id in conv:
                    _fail("INVALID_CATALOG", "duplicate Conv plan")
                conv[entry.layer_id] = entry.material
        return dense, conv

    def _observe_execution(
        self,
        *,
        frame: QueryDualReplayV51BoundsFrame,
        stage_use: authority.StageUse,
        prepared: frozen._Prepared,
        context: replay_v51._V51Context,
        spans_by_range: Mapping[Tuple[int, int], authority.QuerySpan],
        expectations: List[authority.AffineExecutionExpectation],
        traces: List[authority.CompactAbsorptionTrace],
        record: Mapping[str, Any],
        nominal: np.ndarray,
        scalar_before: np.ndarray,
        scalar_after: np.ndarray,
        scalar_guard: Optional[np.ndarray] = None,
        componentwise_radius: Optional[np.ndarray] = None,
        componentwise_penalty: Optional[np.ndarray] = None,
    ) -> None:
        """Consume one live observer callback and retain metadata only."""

        rows = int(record.get("query_count", -1))
        start = int(record.get("query_start", -1))
        end = int(record.get("query_end", -1))
        span = spans_by_range.get((start, end))
        binding = context.query_span_bindings.get((start, end))
        index = len(expectations)
        if (
            span is None
            or binding is None
            or int(record.get("execution_index", -1)) != index
            or not replay_v51._verify_execution_record(
                record,
                index=index,
                query_count=prepared.queries.shape[0],
                query_block_sha256=context.query_block_sha256,
                span_binding=binding,
                output_layer_id=prepared.output_id,
            )
            or record.get("nominal_sha256")
            != frozen._array_digest(np.ascontiguousarray(nominal))
        ):
            _fail(
                "INVALID_EXECUTION",
                "observer record/query binding was substituted",
            )
        lid = int(record["layer_id"])
        pred = int(record["predecessor_id"])
        layer = prepared.layers.get(lid)
        if (
            layer is None
            or layer.kind not in {"DENSE", "CONV2D"}
            or layer.preds != (pred,)
            or pred not in prepared.layers
        ):
            _fail("INVALID_EXECUTION", "affine layer binding changed")

        if layer.kind == "DENSE":
            branch = authority.BRANCH_DENSE
            expected_policy = "v51_wide_or_streamed_scalar_once"
            material = context.dense_supports.get(lid)
            if (
                record.get("operator") != "DENSE"
                or "conv_branch" in record
                or scalar_guard is None
                or componentwise_radius is not None
                or componentwise_penalty is not None
            ):
                _fail("POLICY_SUBSTITUTION", "Dense branch/policy changed")
        else:
            nonzero = int(record.get("nonzero_count", -1))
            dense_count = int(record.get("dense_count", -1))
            sparse = 8 * nonzero <= dense_count
            claimed = record.get("conv_branch")
            if (
                record.get("operator") != "CONV2D"
                or nonzero < 0
                or dense_count <= 0
                or int(record.get("threshold_lhs", -1))
                != 8 * nonzero
                or int(record.get("threshold_rhs", -1))
                != dense_count
            ):
                _fail("BRANCH_SUBSTITUTION", "Conv branch evidence changed")
            if sparse:
                branch = authority.BRANCH_CONV_SPARSE
                expected_policy = "frozen_v3_componentwise"
                material = None
                if (
                    claimed != "sparse"
                    or scalar_guard is not None
                    or componentwise_radius is None
                    or componentwise_penalty is None
                ):
                    _fail(
                        "POLICY_SUBSTITUTION",
                        "sparse Conv policy changed",
                    )
            else:
                branch = authority.BRANCH_CONV_DENSE
                expected_policy = "v51_dense_conv_D_plus_A_once"
                material = context.conv_plans.get(lid)
                if (
                    claimed != "dense"
                    or scalar_guard is None
                    or componentwise_radius is not None
                    or componentwise_penalty is not None
                ):
                    _fail(
                        "POLICY_SUBSTITUTION",
                        "dense Conv policy changed",
                    )
        if record.get("policy") != expected_policy:
            _fail("POLICY_SUBSTITUTION", "guard policy changed")

        active = _bool_array(
            record["active_mask"], rows=rows, name="active_mask"
        )
        fallback = _bool_array(
            record["fallback_mask"], rows=rows, name="fallback_mask"
        )
        applied = _bool_array(
            record["scalar_applied_mask"],
            rows=rows,
            name="scalar_applied_mask",
        )
        if branch == authority.BRANCH_CONV_SPARSE:
            scalar_mask = bytes(authority._mask_length(rows))
            componentwise_mask = authority._full_mask(rows)
            if (
                np.any(active)
                or np.any(fallback)
                or np.any(applied)
                or int(record["scalar_guard_policy_count"]) != 0
                or int(
                    record["componentwise_radius_policy_count"]
                )
                != 1
            ):
                _fail(
                    "ACTIVE_MASK_SUBSTITUTION",
                    "sparse row masks changed",
                )
        else:
            scalar_mask = authority._full_mask(rows)
            componentwise_mask = bytes(authority._mask_length(rows))
            if (
                scalar_guard is None
                or int(record["scalar_guard_policy_count"]) != 1
                or int(
                    record["componentwise_radius_policy_count"]
                )
                != 0
                or not np.array_equal(active, applied)
                or np.any(fallback & ~active)
            ):
                _fail(
                    "ACTIVE_MASK_SUBSTITUTION",
                    "scalar activity/fallback masks changed",
                )
            if (
                record.get("scalar_guard_sha256")
                != frozen._array_digest(
                    np.ascontiguousarray(scalar_guard)
                )
                or record.get("scalar_guard_hex")
                != [float(value).hex() for value in scalar_guard]
            ):
                _fail("POLICY_SUBSTITUTION", "scalar guard changed")

        entry = self._ensure_catalog_entry(
            frame=frame,
            stage_use=stage_use,
            prepared=prepared,
            layer=layer,
            predecessor_id=pred,
            branch=branch,
            material=material,
        )
        if branch == authority.BRANCH_DENSE and (
            record.get("support_sha256")
            != entry.support_content_sha256
            or record.get("catalog_sha256")
            != entry.catalog_content_sha256
        ):
            _fail("INVALID_CATALOG", "Dense record/catalog mismatch")
        if branch == authority.BRANCH_CONV_DENSE and (
            record.get("support_sha256")
            != entry.support_content_sha256
            or record.get("plan_sha256")
            != entry.catalog_content_sha256
            or record.get("helper_input_coefficient_sha256")
            != record.get("input_coefficient_sha256")
        ):
            _fail("INVALID_CATALOG", "Conv record/catalog mismatch")

        partition = authority.RowPolicyPartition(
            row_count=rows,
            scalar_mask=scalar_mask,
            componentwise_mask=componentwise_mask,
            active_mask=authority._mask_from_bool(active),
            fallback_mask=authority._mask_from_bool(fallback),
        )
        expectation = authority._mint_affine_execution_expectation(
            frame._owner,
            execution_index=index,
            span=span,
            support_alias=entry.alias,
            partition=partition,
            input_coefficient_sha256=str(
                record["input_coefficient_sha256"]
            ),
        )
        trace = authority._mint_compact_absorption_trace(
            frame._owner,
            expectation,
            nominal=nominal,
            scalar_before=scalar_before,
            scalar_after=scalar_after,
            scalar_guard=scalar_guard,
            componentwise_radius=componentwise_radius,
            componentwise_penalty=componentwise_penalty,
        )
        expectations.append(expectation)
        traces.append(trace)

    def replay(
        self,
        frame: QueryDualReplayV51BoundsFrame,
        *,
        stage_use_index: int,
        query_rows: Optional[Any] = None,
        one_hot: Optional[Any] = None,
        query_bias: Optional[Any] = None,
        alpha_by_relu: Optional[Mapping[Any, Any]] = None,
        expected_net_sha256: Optional[str] = None,
        expected_bounds_sha256: Optional[str] = None,
        expected_query_sha256: Optional[str] = None,
        expected_alpha_sha256: Optional[str] = None,
        chunk_size: int = 1024,
        max_workspace_bytes: int = 512 * 1024 * 1024,
    ) -> QueryDualReplayV51SessionPendingResult:
        """Replay one of the five precommitted stage uses."""

        started = time.monotonic()
        runtime = self._enter()
        try:
            self._check(runtime)
            self._check_frame(frame)
            if (
                isinstance(stage_use_index, bool)
                or not isinstance(stage_use_index, int)
                or stage_use_index < 0
                or stage_use_index >= len(self._stage_uses)
            ):
                _fail("INVALID_STAGE_USE", "unknown stage-use index")
            if (
                isinstance(chunk_size, bool)
                or not isinstance(chunk_size, int)
                or chunk_size <= 0
                or isinstance(max_workspace_bytes, bool)
                or not isinstance(max_workspace_bytes, int)
                or max_workspace_bytes <= 0
            ):
                _fail("INVALID_CHUNK", "invalid replay chunk budget")
            stage_use = self._stage_uses[stage_use_index]
            key = stage_use.cone_start_lid
            cone = self._contexts.get(key)
            if cone is None:
                _fail("INVALID_CONTEXT", "stage cone is not sealed")
            prepared = frozen._prepare_from_sealed(
                cone,
                frame,
                query_rows=query_rows,
                one_hot=one_hot,
                query_bias=query_bias,
                alpha_by_relu=alpha_by_relu,
                deadline=self._deadline,
                expected_net_sha256=expected_net_sha256,
                expected_bounds_sha256=expected_bounds_sha256,
                expected_query_sha256=expected_query_sha256,
                expected_alpha_sha256=expected_alpha_sha256,
            )
            maximum_width = max(
                layer.width for layer in prepared.layers.values()
            )
            bytes_per_query = max(1, maximum_width * 8 * 12)
            memory_limited = max(
                1, max_workspace_bytes // bytes_per_query
            )
            effective_chunk = min(
                chunk_size,
                memory_limited,
                prepared.queries.shape[0],
            )
            chunks = tuple(
                (
                    start,
                    min(
                        prepared.queries.shape[0],
                        start + effective_chunk,
                    ),
                )
                for start in range(
                    0, prepared.queries.shape[0], effective_chunk
                )
            )
            dense_cache, conv_cache = self._preload_catalog(
                frame, stage_use
            )
            context = replay_v51._V51Context(
                prepared=prepared,
                dense_supports=dense_cache,
                conv_plans=conv_cache,
            )
            spans: List[authority.QuerySpan] = []
            spans_by_range: Dict[
                Tuple[int, int], authority.QuerySpan
            ] = {}
            for span_index, (start, end) in enumerate(chunks):
                binding = replay_v51._query_span_binding(
                    context, start, end
                )
                span = authority._mint_query_span(
                    frame._owner,
                    stage_use_sha256=stage_use.stage_use_sha256,
                    span_index=span_index,
                    query_start=start,
                    query_end=end,
                    query_total=prepared.queries.shape[0],
                    query_block_sha256=context.query_block_sha256,
                    query_rows_sha256=str(
                        binding["query_rows_sha256"]
                    ),
                    query_bias_sha256=str(
                        binding["query_bias_sha256"]
                    ),
                    alpha_slice_sha256=str(
                        binding["alpha_slice_sha256"]
                    ),
                )
                spans.append(span)
                spans_by_range[(start, end)] = span

            expectations: List[
                authority.AffineExecutionExpectation
            ] = []
            traces: List[authority.CompactAbsorptionTrace] = []

            def observe(**payload: Any) -> None:
                self._observe_execution(
                    frame=frame,
                    stage_use=stage_use,
                    prepared=prepared,
                    context=context,
                    spans_by_range=spans_by_range,
                    expectations=expectations,
                    traces=traces,
                    **payload,
                )

            context.execution_observer = observe
            stats = frozen._ReplayStats()
            stats.configure_queries(prepared.queries.shape[0])
            values = np.empty(
                prepared.queries.shape[0], dtype=np.float64
            )
            affine_schedule = tuple(
                lid
                for lid in prepared.reverse_order
                if prepared.layers[lid].kind in {"DENSE", "CONV2D"}
            )
            for start, end in chunks:
                self._deadline.check(force=True)
                before = len(expectations)
                values[start:end] = replay_v51._replay_block_v51(
                    context, start, end, stats
                )
                observed = tuple(
                    expectation.support_alias.layer_id
                    for expectation in expectations[before:]
                )
                if observed != affine_schedule:
                    _fail(
                        "MISSING_EXECUTION",
                        "observer affine schedule is incomplete or reordered",
                    )
            if (
                len(context.executions) != len(expectations)
                or len(expectations) != len(traces)
                or len(expectations)
                != len(chunks) * len(affine_schedule)
            ):
                _fail(
                    "MISSING_TRACE",
                    "affine executions and live traces differ",
                )
            replay_v51._span_manifest(
                context.executions, prepared.queries.shape[0]
            )
            replay_v51._query_span_bindings_manifest(
                context, prepared.queries.shape[0]
            )
            ledger = authority._mint_compact_guard_ledger(
                frame._owner, spans, expectations
            )
            for trace in traces:
                ledger.record(trace)
            ledger_certificate = ledger.commit()
            if not authority.validate_compact_guard_ledger_certificate(
                ledger_certificate
            ):
                _fail("INVALID_LEDGER", "committed ledger is invalid")
            if not np.all(np.isfinite(values)):
                _fail("NONFINITE", "non-finite V5.1 lower bounds")
            immutable_values = frozen._immutable_f64_array(
                values, name="V5.1 session pending lower bounds"
            )
            stage_body: Dict[str, Any] = {
                "schema": STAGE_SCHEMA,
                "numeric_protocol": NUMERIC_PROTOCOL,
                "proof_authority": False,
                "frame_content_sha256": frame._content_sha256,
                "frame_binding_sha256": (
                    frame._binding.binding_sha256
                ),
                "stage_use": dict(_stage_record(stage_use)),
                "hashes": dict(prepared.hashes),
                "output_layer_id": prepared.output_id,
                "query_count": immutable_values.size,
                "query_block_sha256": context.query_block_sha256,
                "span_schedule_sha256": ledger_certificate.receipt[
                    "span_schedule_sha256"
                ],
                "ledger_content_sha256": (
                    ledger_certificate.content_sha256
                ),
                "ledger_receipt_sha256": ledger_certificate.receipt[
                    "receipt_sha256"
                ],
                "affine_execution_count": len(expectations),
                "requested_chunk_size": chunk_size,
                "effective_chunk_size": effective_chunk,
                "max_workspace_bytes": max_workspace_bytes,
                "lower_bounds_sha256": frozen._array_digest(
                    immutable_values
                ),
                "stats": dict(replay_v51._stats_record(stats)),
                "catalog_alias_sha256": tuple(
                    ledger_certificate.receipt[
                        "catalog_alias_sha256"
                    ]
                ),
                "elapsed_s_hex": float(
                    time.monotonic() - started
                ).hex(),
            }
            stage_body["receipt_sha256"] = _json_sha256(stage_body)
            token = secrets.token_hex(32)
            public = QueryDualReplayV51SessionPendingResult(
                lower_bounds=immutable_values,
                stage_token=token,
            )
            content_sha = _json_sha256(
                {
                    "stage_token_sha256": hashlib.sha256(
                        token.encode("ascii")
                    ).hexdigest(),
                    "public_identity": id(public),
                    "lower_bounds_sha256": frozen._array_digest(
                        immutable_values
                    ),
                    "stage_receipt_sha256": stage_body[
                        "receipt_sha256"
                    ],
                    "ledger_content_sha256": (
                        ledger_certificate.content_sha256
                    ),
                }
            )
            pending = _PendingStage(
                public=public,
                stage_use_index=stage_use_index,
                frame_nonce=frame._frame_nonce,
                frame_sha256=frame._content_sha256,
                ledger=ledger_certificate,
                stage_receipt=_deep_freeze(stage_body),
                content_sha256=content_sha,
            )
            self._pending.append(pending)
            runtime.pending_seals.append(self._pending_seal(pending))
            self._check(runtime)
            return public
        except frozen.QueryDualReplayTimeout as exc:
            self._invalidate(runtime)
            raise QueryDualReplayV51SessionTimeout() from exc
        except authority.QueryDualV51AuthorityError as exc:
            self._invalidate(runtime)
            if exc.code == "DEADLINE_EXPIRED":
                raise QueryDualReplayV51SessionTimeout(str(exc)) from exc
            raise QueryDualReplayV51SessionError(
                exc.code, str(exc)
            ) from exc
        except frozen.QueryDualReplayError as exc:
            self._invalidate(runtime)
            raise QueryDualReplayV51SessionError(
                exc.code, str(exc)
            ) from exc
        except Exception:
            self._invalidate(runtime)
            raise
        finally:
            runtime.operation_lock.release()

    def _validate_static_manifests(self) -> None:
        manifests = {
            lid: frozen._layer_manifest(layer)
            for lid, layer in self._full_layers.items()
        }
        for key, cone in self._contexts.items():
            if key != cone.start_lid:
                _fail("INVALID_CONTEXT", "cone key changed")
            actual = frozen._json_digest(
                [
                    manifests[lid]
                    for lid in reversed(cone.reverse_order)
                ]
            )
            if (
                actual != cone.manifest_sha256
                or actual != cone.replay_net_sha256
            ):
                _fail("INVALID_CONTEXT", "cone manifest changed")

    def commit(
        self,
    ) -> Tuple[QueryDualReplayV51SessionCandidateResult, ...]:
        """Revalidate every root/frame/catalog/ledger seal and publish."""

        runtime = self._enter()
        try:
            self._check(runtime)
            stage_indices = [
                value.stage_use_index for value in self._pending
            ]
            frame_nonces = {
                value.frame_nonce for value in self._pending
            }
            if (
                stage_indices != list(range(len(self._stage_uses)))
                or len(self._pending) != len(self._stage_uses)
                or len(frame_nonces) != 1
            ):
                _fail(
                    "INCOMPLETE_SESSION",
                    "commit requires stages 0..4 exactly once on one frame",
                )
            if not verify_query_dual_box_certificate(
                self._root_certificate
            ):
                _fail(
                    "INVALID_ROOT_CERTIFICATE",
                    "root certificate changed",
                )
            try:
                graph = _borrow_sealed_query_dual_graph(
                    self._root_certificate, validate_content=True
                )
            except QueryDualBoxError as exc:
                raise QueryDualReplayV51SessionError(
                    "INVALID_ROOT_CERTIFICATE", str(exc)
                ) from exc
            if graph is not self._root_graph:
                _fail(
                    "INVALID_ROOT_CERTIFICATE",
                    "root graph identity changed",
                )
            self._validate_static_manifests()
            crosswalk = frozen._sealed_crosswalk(
                self._root_certificate,
                self._root_graph,
                self._contexts,
            )
            self._static_manifest_commit_validations += 1
            if (
                crosswalk != dict(self._crosswalk)
                or frozen._json_digest(crosswalk)
                != self._crosswalk_sha256
            ):
                _fail("INVALID_CONTEXT", "root/cone crosswalk changed")
            blas_valid = (
                id(self._blas_contract) == runtime.static_identity[6]
                and validate_query_dual_blas_contract(
                    self._blas_contract,
                    recheck_current_platform=True,
                    deadline=self._deadline.end,
                )
            )
            if not blas_valid:
                self._deadline.check(force=True)
                _fail(
                    "BLAS_CONTRACT_MISMATCH",
                    "live BLAS contract differs at commit",
                )
            fresh_numeric = _capture_numeric_contract(
                self._blas_contract
            )
            if _canonical_value(fresh_numeric) != _canonical_value(
                self._numeric_contract
            ):
                _fail(
                    "NUMERIC_PLATFORM_MISMATCH",
                    "numeric/source contract changed",
                )

            validated_frames: set[str] = set()
            for pending in self._pending:
                frame = self._frames.get(pending.frame_nonce)
                if frame is None:
                    _fail("INVALID_FRAME", "pending frame disappeared")
                if pending.frame_nonce not in validated_frames:
                    self._check_frame(frame, full=True)
                    validated_frames.add(pending.frame_nonce)
                body = dict(pending.stage_receipt)
                claimed = str(body.pop("receipt_sha256"))
                public = pending.public
                expected_content = _json_sha256(
                    {
                        "stage_token_sha256": hashlib.sha256(
                            public.stage_token.encode("ascii")
                        ).hexdigest(),
                        "public_identity": id(public),
                        "lower_bounds_sha256": frozen._array_digest(
                            public.lower_bounds
                        ),
                        "stage_receipt_sha256": claimed,
                        "ledger_content_sha256": (
                            pending.ledger.content_sha256
                        ),
                    }
                )
                if (
                    pending.frame_sha256 != frame._content_sha256
                    or body.get("proof_authority") is not False
                    or body.get("schema") != STAGE_SCHEMA
                    or body.get("lower_bounds_sha256")
                    != frozen._array_digest(public.lower_bounds)
                    or not hmac.compare_digest(
                        _json_sha256(body), claimed
                    )
                    or not hmac.compare_digest(
                        expected_content, pending.content_sha256
                    )
                    or not authority.validate_compact_guard_ledger_certificate(
                        pending.ledger
                    )
                    or pending.ledger.content_sha256
                    != body.get("ledger_content_sha256")
                    or pending.ledger.frame_binding
                    is not frame._binding
                ):
                    _fail("INVALID_STAGE", "pending stage changed")
            self._check(runtime)
            if not verify_query_dual_box_certificate(
                self._root_certificate, net=self._net
            ):
                _fail(
                    "LIVE_NET_MISMATCH",
                    "live network changed before commit",
                )
            self._check(runtime)

            session_nonce_sha = hashlib.sha256(
                self._nonce.encode("ascii")
            ).hexdigest()
            results: List[
                QueryDualReplayV51SessionCandidateResult
            ] = []
            registered: List[str] = []
            try:
                for pending in self._pending:
                    frame = self._frames[pending.frame_nonce]
                    body: Dict[str, Any] = {
                        "schema": SCHEMA,
                        "status": "session_candidate",
                        "proof_authority": False,
                        "numeric_protocol": NUMERIC_PROTOCOL,
                        "authority_scope": (
                            "candidate_only_no_solver_verdict"
                        ),
                        "session_nonce_sha256": session_nonce_sha,
                        "root_receipt_sha256": str(
                            self._root_certificate.receipt[
                                "receipt_sha256"
                            ]
                        ),
                        "root_graph_content_sha256": (
                            self._root_graph.content_sha256
                        ),
                        "frame_content_sha256": frame._content_sha256,
                        "frame_binding_sha256": (
                            frame._binding.binding_sha256
                        ),
                        "stage_use": dict(
                            _stage_record(
                                self._stage_uses[
                                    pending.stage_use_index
                                ]
                            )
                        ),
                        "stage_receipt_sha256": (
                            pending.stage_receipt["receipt_sha256"]
                        ),
                        "ledger_content_sha256": (
                            pending.ledger.content_sha256
                        ),
                        "ledger_receipt_sha256": (
                            pending.ledger.receipt["receipt_sha256"]
                        ),
                        "numeric_contract_sha256": (
                            self._numeric_contract["content_sha256"]
                        ),
                        "blas_contract_sha256": (
                            self._blas_contract.content_sha256
                        ),
                        "manifest_crosswalk_sha256": (
                            self._crosswalk_sha256
                        ),
                        "live_net_commit_bound": True,
                        "lower_bounds_sha256": frozen._array_digest(
                            pending.public.lower_bounds
                        ),
                    }
                    body["receipt_sha256"] = _json_sha256(body)
                    nonce = secrets.token_hex(32)
                    result = (
                        QueryDualReplayV51SessionCandidateResult(
                            lower_bounds=pending.public.lower_bounds,
                            receipt=_deep_freeze(body),
                            _nonce=nonce,
                            _capability=_RESULT_CAPABILITY,
                        )
                    )
                    with _RESULT_LOCK:
                        _RESULT_REGISTRY[nonce] = result
                        _RESULT_SEALS[nonce] = (
                            id(result.lower_bounds),
                            id(result.receipt),
                            str(body["receipt_sha256"]),
                        )
                    registered.append(nonce)
                    weakref.finalize(result, _drop_result_seal, nonce)
                    if not validate_query_dual_replay_v51_session_candidate(
                        result
                    ):
                        _fail(
                            "INVALID_RESULT",
                            "candidate publication validation failed",
                        )
                    results.append(result)
                self._check(runtime)
                with runtime.state_lock:
                    if (
                        runtime.closed
                        or runtime.failed
                        or runtime.poisoned
                    ):
                        _fail(
                            "INVALID_SESSION",
                            "concurrent use invalidated session publication",
                        )
                    runtime.deadline.check(force=True)
                    runtime.closed = True
            except Exception:
                with _RESULT_LOCK:
                    for nonce in registered:
                        _RESULT_REGISTRY.pop(nonce, None)
                        _RESULT_SEALS.pop(nonce, None)
                raise
            self._closed = True
            with _SESSION_LOCK:
                if _SESSION_REGISTRY.get(self._nonce) is self:
                    _SESSION_REGISTRY.pop(self._nonce, None)
            for frame in runtime.frames.values():
                frame._catalog.clear()
            runtime.frames.clear()
            runtime.frame_identity.clear()
            runtime.frame_seals.clear()
            runtime.pending.clear()
            runtime.pending_seals.clear()
            return tuple(results)
        except frozen.QueryDualReplayTimeout as exc:
            self._invalidate(runtime)
            raise QueryDualReplayV51SessionTimeout() from exc
        except frozen.QueryDualReplayError as exc:
            self._invalidate(runtime)
            raise QueryDualReplayV51SessionError(
                exc.code, str(exc)
            ) from exc
        except QueryDualBlasContractError as exc:
            self._invalidate(runtime)
            if exc.code == "DEADLINE_EXPIRED":
                raise QueryDualReplayV51SessionTimeout(str(exc)) from exc
            raise QueryDualReplayV51SessionError(
                exc.code, str(exc)
            ) from exc
        except Exception:
            self._invalidate(runtime)
            raise
        finally:
            runtime.operation_lock.release()


def _validate_stage_uses(
    full_layers: Mapping[int, frozen._FrozenLayer],
    stage_uses: Sequence[authority.StageUse],
) -> Tuple[authority.StageUse, ...]:
    if isinstance(stage_uses, (str, bytes)) or not isinstance(
        stage_uses, Sequence
    ):
        _fail("INVALID_STAGE_USE", "stage_uses must be a sequence")
    uses = tuple(stage_uses)
    if (
        len(uses) != 5
        or any(
            not authority.validate_stage_use(value) for value in uses
        )
        or [value.use_index for value in uses] != list(range(5))
        or [value.stage_kind for value in uses]
        != [authority.STAGE_TARGET] * 4
        + [authority.STAGE_PROPERTY]
        or [value.stage_index for value in uses[:4]]
        != list(range(4))
    ):
        _fail(
            "INVALID_STAGE_USE",
            "V5.1 requires four ordered targets then one property use",
        )
    starts: List[int] = []
    for use in uses[:4]:
        target = full_layers.get(int(use.target_relu_lid))
        start = int(use.cone_start_lid)
        if (
            target is None
            or target.kind != "RELU"
            or target.preds != (start,)
            or start not in full_layers
        ):
            _fail(
                "INVALID_STAGE_USE",
                "target use is not the predecessor cone of a ReLU",
            )
        starts.append(start)
    if len(set(starts)) != 4:
        _fail("INVALID_STAGE_USE", "target cone starts must be distinct")
    return uses


def create_query_dual_replay_v51_session(
    net: Any,
    root_certificate: QueryDualBoxCertificate,
    stage_uses: Sequence[authority.StageUse],
    *,
    deadline: float,
    blas_contract: QueryDualBlasContract,
) -> QueryDualReplayV51Session:
    """Create an independent five-stage V5.1 transaction from one root."""

    if os.getpid() != _SESSION_PID:
        _fail(
            "PROCESS_MISMATCH",
            "V5.1 session cannot be inherited across fork",
        )
    if isinstance(deadline, bool):
        _fail("INVALID_DEADLINE", "deadline must be an absolute timestamp")
    try:
        absolute = float(deadline)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QueryDualReplayV51SessionError(
            "INVALID_DEADLINE", "deadline must be finite"
        ) from exc
    if not math.isfinite(absolute):
        _fail("INVALID_DEADLINE", "deadline must be finite")
    timer = frozen._Deadline(end=absolute)
    try:
        timer.check(force=True)
    except frozen.QueryDualReplayTimeout as exc:
        raise QueryDualReplayV51SessionTimeout() from exc
    if not verify_query_dual_box_certificate(root_certificate):
        _fail(
            "INVALID_ROOT_CERTIFICATE",
            "root certificate is invalid",
        )
    try:
        root_graph = _borrow_sealed_query_dual_graph(
            root_certificate, validate_content=False
        )
    except QueryDualBoxError as exc:
        raise QueryDualReplayV51SessionError(
            "INVALID_ROOT_CERTIFICATE", str(exc)
        ) from exc
    try:
        full_layers = MappingProxyType(
            {
                int(layer.id): frozen._replay_layer_from_root(layer)
                for layer in root_graph.layers
            }
        )
        uses = _validate_stage_uses(full_layers, stage_uses)
        manifests = {
            lid: frozen._layer_manifest(layer)
            for lid, layer in full_layers.items()
        }
        contexts: Dict[Optional[int], frozen._SealedCone] = {}
        for start in tuple(
            [value.cone_start_lid for value in uses[:4]] + [None]
        ):
            timer.check(force=True)
            contexts[start] = frozen._sealed_cone(
                full_layers,
                manifests,
                assert_id=int(root_graph.assert_id),
                start_lid=start,
            )
        numeric_contract = _capture_numeric_contract(blas_contract)
        timer.check(force=True)
        return QueryDualReplayV51Session(
            token=_SESSION_CAPABILITY,
            net=net,
            root_certificate=root_certificate,
            root_graph=root_graph,
            full_layers=full_layers,
            contexts=MappingProxyType(contexts),
            stage_uses=uses,
            deadline=timer,
            blas_contract=blas_contract,
            numeric_contract=numeric_contract,
        )
    except frozen.QueryDualReplayTimeout as exc:
        raise QueryDualReplayV51SessionTimeout() from exc
    except frozen.QueryDualReplayError as exc:
        raise QueryDualReplayV51SessionError(
            exc.code, str(exc)
        ) from exc


__all__ = [
    "FRAME_SCHEMA",
    "NUMERIC_PROTOCOL",
    "QueryDualReplayV51BoundsFrame",
    "QueryDualReplayV51Session",
    "QueryDualReplayV51SessionCandidateResult",
    "QueryDualReplayV51SessionError",
    "QueryDualReplayV51SessionPendingResult",
    "QueryDualReplayV51SessionTimeout",
    "SCHEMA",
    "STAGE_SCHEMA",
    "create_query_dual_replay_v51_session",
    "validate_query_dual_replay_v51_session_candidate",
]
