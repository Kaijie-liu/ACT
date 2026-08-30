# ===- query_dual_v51b_material_cache.py - V5.1b Conv cache toy ---===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Isolated, non-authoritative V5.1b physical Conv material-cache toy.

This module is deliberately not wired into the production replay/session
path.  It exercises one architectural claim from the V5.1b pre-registration:
one bounds frame may share a bytes-backed physical
``DenseConvV51Plan`` across overlapping stages while retaining a distinct,
frame-owner-minted semantic alias for every stage use.

The cache key deliberately excludes stage/cone identity and includes the
frame, exact source-box semantics and lower/upper digests, layer/predecessor,
weight, Conv geometry, numeric platform, and implementation.  Admission and
commit perform the exhaustive V5.1a mathematical validator.  Cache hits use
only process-local external identity/content seals.  The public replay helper
still calls ``replay_dense_conv_v51`` and therefore still performs its full
validator; this toy provides no unchecked public numerical entry point.

All public values permanently carry ``proof_authority=False``.  Nothing in
this module can authorize a solver verdict.
"""

from __future__ import annotations

import hashlib
import hmac
import gc
import json
import math
import os
from pathlib import Path
import secrets
import threading
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, NoReturn, Optional, Tuple

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv_v51


SCHEMA = "act.query_dual_v51b_conv_material_cache_candidate.v1"
ALIAS_SCHEMA = "act.query_dual_v51b_conv_material_alias_candidate.v1"
COMMIT_SCHEMA = "act.query_dual_v51b_conv_material_commit_candidate.v1"
NUMERIC_PROTOCOL = "frame_global_physical_conv_core_stage_alias_v51b"

BRANCH = "conv_dense"
OPERATOR = "CONV2D"
BOX_OUTPUT = "output_box_v1"
BOX_RELU_PRE = "relu_preactivation_box_v1"
BOX_RELU_POST = "relu_postactivation_from_preactivation_box_v1"
_BOX_SEMANTICS = frozenset((BOX_OUTPUT, BOX_RELU_PRE, BOX_RELU_POST))

_PID = os.getpid()
_FRAME_CAPABILITY = object()
_ALIAS_CAPABILITY = object()
_FRAME_LOCK = threading.Lock()
_FRAME_REGISTRY: weakref.WeakValueDictionary[
    str, "ConvMaterialFrameCandidate"
] = weakref.WeakValueDictionary()
_FRAME_RUNTIMES: Dict[str, "_FrameRuntime"] = {}


class ConvMaterialCacheError(RuntimeError):
    """Stable fail-closed error for the isolated cache toy."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


class ConvMaterialCacheTimeout(ConvMaterialCacheError):
    """The frame's one absolute deadline expired."""

    def __init__(self) -> None:
        super().__init__("DEADLINE_EXPIRED", "V5.1b frame deadline expired")


def _fail(code: str, message: str) -> NoReturn:
    raise ConvMaterialCacheError(code, message)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical(value.item())
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


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
    return MappingProxyType(
        {
            "material_cache": _file_sha256(Path(__file__).resolve()),
            "conv_v51": _file_sha256(Path(conv_v51.__file__).resolve()),
        }
    )


def _bytes_backed(array: Any) -> bool:
    if not isinstance(array, np.ndarray) or array.flags.writeable:
        return False
    current: Any = array
    seen = set()
    while isinstance(current, np.ndarray):
        if id(current) in seen:
            return False
        seen.add(id(current))
        current = current.base
    return isinstance(current, bytes)


def _immutable_f64(value: Any, *, name: str) -> np.ndarray:
    return frozen._immutable_f64_array(value, name=name)


def _geometry_body_from_layer(
    layer: frozen._FrozenLayer,
) -> Mapping[str, Any]:
    geometry = conv_v51._geometry(layer)
    weight = np.asarray(layer.params["weight"])
    return {
        "operator": OPERATOR,
        "weight_shape": list(weight.shape),
        "input_shape": list(geometry["input_shape"]),
        "output_shape": list(geometry["output_shape"]),
        "stride": list(geometry["stride"]),
        "padding": list(geometry["padding"]),
        "dilation": list(geometry["dilation"]),
        "groups": int(geometry["groups"]),
    }


def _geometry_body_from_plan(
    plan: conv_v51.DenseConvV51Plan,
) -> Mapping[str, Any]:
    return {
        "operator": OPERATOR,
        "weight_shape": list(plan.weight.shape),
        "input_shape": list(plan.input_shape),
        "output_shape": list(plan.output_shape),
        "stride": list(plan.stride),
        "padding": list(plan.padding),
        "dilation": list(plan.dilation),
        "groups": int(plan.groups),
    }


def _source_box(value: frozen._Box) -> frozen._Box:
    if type(value) is not frozen._Box:
        _fail("INVALID_BOX", "predecessor_box must be a frozen replay box")
    lb = _immutable_f64(value.lb, name="V5.1b source lower").reshape(-1)
    ub = _immutable_f64(value.ub, name="V5.1b source upper").reshape(-1)
    if lb.shape != ub.shape or np.any(lb > ub):
        _fail("INVALID_BOX", "source box is malformed")
    return frozen._Box(lb=lb, ub=ub)


def _check_deadline(runtime: "_FrameRuntime") -> None:
    if os.getpid() != _PID or runtime.pid != _PID:
        _fail("FORKED_PROCESS", "cache capability cannot cross a fork")
    try:
        runtime.deadline.check(force=True)
    except frozen.QueryDualReplayTimeout as exc:
        runtime.closed = True
        raise ConvMaterialCacheTimeout() from exc


@dataclass(frozen=True)
class ConvMaterialStageUse:
    """One pre-registered stage/cone identity."""

    stage_use_sha256: str
    cone_start_lid: Optional[int]

    def __post_init__(self) -> None:
        if not _is_sha256(self.stage_use_sha256) or (
            self.cone_start_lid is not None
            and (
                isinstance(self.cone_start_lid, bool)
                or not isinstance(self.cone_start_lid, int)
                or self.cone_start_lid < 0
            )
        ):
            raise ValueError("malformed V5.1b stage use")


@dataclass(frozen=True)
class ConvMaterialStageAliasCandidate:
    """Frame-owner-minted stage-local alias; never proof authority."""

    stage_use_sha256: str
    cone_start_lid: Optional[int]
    frame_content_sha256: str
    physical_key_sha256: str
    physical_core_content_sha256: str
    layer_id: int
    predecessor_id: int
    receipt: Mapping[str, Any]
    proof_authority: bool = False
    _nonce: str = field(default="", repr=False, compare=False)
    _frame_nonce: str = field(default="", repr=False, compare=False)
    _capability: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.proof_authority is not False
            or not _is_sha256(self.stage_use_sha256)
            or not _is_sha256(self.frame_content_sha256)
            or not _is_sha256(self.physical_key_sha256)
            or not _is_sha256(self.physical_core_content_sha256)
            or type(self.receipt) is not MappingProxyType
            or self._capability is not _ALIAS_CAPABILITY
            or not self._nonce
            or not self._frame_nonce
        ):
            raise ValueError("malformed V5.1b material alias candidate")


@dataclass(frozen=True)
class ConvMaterialCommitCandidate:
    """Immutable commit report which explicitly denies proof authority."""

    physical_builds: int
    cross_stage_physical_hits: int
    stage_aliases: int
    execution_alias_lookups: int
    commit_full_validations: int
    physical_core_sha256s: Tuple[str, ...]
    receipt: Mapping[str, Any]
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.proof_authority is not False
            or min(
                self.physical_builds,
                self.cross_stage_physical_hits,
                self.stage_aliases,
                self.execution_alias_lookups,
                self.commit_full_validations,
            )
            < 0
            or self.commit_full_validations != self.physical_builds
            or self.stage_aliases
            != self.physical_builds
            + self.cross_stage_physical_hits
            or len(self.physical_core_sha256s) != self.physical_builds
            or any(not _is_sha256(value) for value in self.physical_core_sha256s)
            or type(self.receipt) is not MappingProxyType
        ):
            raise ValueError("malformed V5.1b commit candidate")


@dataclass(frozen=True)
class _CoreBinding:
    frame_content_sha256: str
    layer_id: int
    predecessor_id: int
    operator_kind: str
    branch: str
    weight_sha256: str
    geometry_sha256: str
    source_lb_sha256: str
    source_ub_sha256: str
    box_semantics: str
    numeric_platform_sha256: str
    implementation_sha256: str
    source_lb: np.ndarray = field(repr=False, compare=False)
    source_ub: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class _PhysicalConvCore:
    key_sha256: str
    content_sha256: str
    plan_content_sha256: str
    binding: _CoreBinding
    plan: conv_v51.DenseConvV51Plan = field(repr=False, compare=False)
    receipt_bytes: bytes = field(repr=False, compare=False)
    proof_authority: bool = False


@dataclass(frozen=True)
class _CoreSeal:
    core_id: int
    binding_id: int
    plan_id: int
    receipt_id: int
    receipt_bytes: bytes
    binding_fast: Tuple[Any, ...]
    plan_fast: Tuple[Any, ...]
    offsets_fast: Tuple[Tuple[Any, ...], ...]
    key_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class _AliasSeal:
    alias_id: int
    receipt_id: int
    receipt_bytes: bytes
    frame_nonce: str
    frame_content_sha256: str
    stage_use_sha256: str
    cone_start_lid: Optional[int]
    physical_key_sha256: str
    physical_core_content_sha256: str
    layer_id: int
    predecessor_id: int


@dataclass(frozen=True)
class _StageSeal:
    stage_id: int
    stage: ConvMaterialStageUse = field(repr=False, compare=False)
    stage_use_sha256: str
    cone_start_lid: Optional[int]
    record_bytes: bytes
    content_sha256: str


@dataclass(frozen=True)
class _FastInputSnapshot:
    token: Tuple[Any, ...]
    caller_signature: Tuple[Any, ...]
    predecessor_id: int
    box_semantics: str
    layer: frozen._FrozenLayer = field(repr=False, compare=False)
    predecessor_box: frozen._Box = field(repr=False, compare=False)
    caller_layer: frozen._FrozenLayer = field(repr=False, compare=False)
    caller_box: frozen._Box = field(repr=False, compare=False)
    content_sha256: str


@dataclass(frozen=True)
class _FastInputCapture:
    """One shallow, root-owned capture of an exact caller object graph."""

    layer: frozen._FrozenLayer = field(repr=False, compare=False)
    predecessor_box: frozen._Box = field(repr=False, compare=False)
    layer_id: int
    layer_kind: str
    layer_preds: Tuple[int, ...]
    layer_width: int
    layer_in_vars: Tuple[Any, ...] = field(repr=False, compare=False)
    layer_out_vars: Tuple[Any, ...] = field(repr=False, compare=False)
    params: Mapping[str, Any] = field(repr=False, compare=False)
    params_backing_id: int
    weight: np.ndarray = field(repr=False, compare=False)
    bias: np.ndarray = field(repr=False, compare=False)
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    groups: int
    lb: np.ndarray = field(repr=False, compare=False)
    ub: np.ndarray = field(repr=False, compare=False)
    predecessor_id: int
    box_semantics: str


@dataclass(frozen=True)
class _FastInputSeal:
    token: Tuple[Any, ...]
    signature: Tuple[Any, ...]
    physical_key_sha256: str
    layer: frozen._FrozenLayer = field(repr=False, compare=False)
    predecessor_box: frozen._Box = field(repr=False, compare=False)
    snapshot: _FastInputSnapshot = field(repr=False, compare=False)


def _build_fast_input_trust_root():
    """Create an append-only MAC registry hidden in a lexical closure."""

    registry: Dict[str, Tuple[Any, ...]] = {}
    retired = set()
    mac_key = secrets.token_bytes(32)
    lock = threading.RLock()

    def create(
        nonce: str,
        *,
        frame_content_sha256: str,
        numeric_platform_sha256: str,
        implementation_sha256: str,
    ) -> Optional[int]:
        context = {
            "schema": "act.query_dual_v51b_fast_trust_context.v1",
            "frame_content_sha256": frame_content_sha256,
            "numeric_platform_sha256": numeric_platform_sha256,
            "implementation_sha256": implementation_sha256,
        }
        if any(
            not _is_sha256(value)
            for value in (
                frame_content_sha256,
                numeric_platform_sha256,
                implementation_sha256,
            )
        ):
            return None
        context_bytes = bytes(_json_bytes(context))
        context_mac = hmac.new(
            mac_key, context_bytes, hashlib.sha256
        ).hexdigest()
        with lock:
            if nonce in registry or nonce in retired:
                return None
            values: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}
            registry[nonce] = (
                frame_content_sha256,
                numeric_platform_sha256,
                implementation_sha256,
                context_bytes,
                context_mac,
                values,
            )
            return id(values)

    def admit_once(
        nonce: str,
        token: Tuple[Any, ...],
        seal: _FastInputSeal,
        *,
        snapshot: _FastInputSnapshot,
    ) -> bool:
        with lock:
            frame = registry.get(nonce)
        if frame is None:
            return False
        (
            frame_sha,
            platform_sha,
            implementation_sha,
            context_bytes,
            context_mac,
            _,
        ) = frame
        if not hmac.compare_digest(
            context_mac,
            hmac.new(
                mac_key, context_bytes, hashlib.sha256
            ).hexdigest(),
        ):
            return False
        try:
            if (
                type(snapshot) is not _FastInputSnapshot
                or seal.snapshot is not snapshot
                or seal.layer is not snapshot.layer
                or seal.predecessor_box
                is not snapshot.predecessor_box
                or seal.signature != snapshot.caller_signature
                or seal.token != snapshot.token
                or snapshot.token
                != (
                    id(snapshot.caller_layer),
                    id(snapshot.caller_box),
                    snapshot.predecessor_id,
                    snapshot.box_semantics,
                )
                or _json_sha256(_snapshot_record(snapshot))
                != snapshot.content_sha256
            ):
                return False
            expected_key = _physical_key_from_immutable_inputs(
                frame_content_sha256=frame_sha,
                numeric_platform_sha256=platform_sha,
                implementation_sha256=implementation_sha,
                layer=snapshot.layer,
                predecessor_box=snapshot.predecessor_box,
                predecessor_id=snapshot.predecessor_id,
                box_semantics=snapshot.box_semantics,
            )
            private_snapshot = _clone_fast_snapshot(snapshot)
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            frozen.QueryDualReplayError,
            ConvMaterialCacheError,
        ):
            return False
        if (
            token != snapshot.token
            or token != seal.token
            or not hmac.compare_digest(
                expected_key, seal.physical_key_sha256
            )
        ):
            return False
        body_bytes = _json_bytes(_fast_input_external_body(seal))
        mac = hmac.new(
            mac_key, body_bytes, hashlib.sha256
        ).hexdigest()
        with lock:
            current = registry.get(nonce)
            if (
                current is not frame
                or not hmac.compare_digest(
                    current[4], context_mac
                )
                or _json_sha256(_snapshot_record(snapshot))
                != snapshot.content_sha256
            ):
                return False
            values = current[5]
            if token in values or token != seal.token:
                return False
            values[token] = (
                bytes(body_bytes),
                mac,
                private_snapshot,
                expected_key,
                id(seal),
                token,
                snapshot,
            )
            return True

    def validate(
        nonce: str,
        token: Tuple[Any, ...],
        seal: _FastInputSeal,
    ) -> bool:
        with lock:
            frame = registry.get(nonce)
            record = None if frame is None else frame[5].get(token)
        if frame is None or record is None:
            return False
        context_bytes = frame[3]
        context_mac = frame[4]
        if not hmac.compare_digest(
            context_mac,
            hmac.new(
                mac_key, context_bytes, hashlib.sha256
            ).hexdigest(),
        ):
            return False
        body_bytes = _json_bytes(_fast_input_external_body(seal))
        expected_mac = hmac.new(
            mac_key, body_bytes, hashlib.sha256
        ).hexdigest()
        return bool(
            token == seal.token
            and record[0] == body_bytes
            and hmac.compare_digest(record[1], expected_mac)
        )

    def commit_validate(
        nonce: str,
        token: Tuple[Any, ...],
        seal: _FastInputSeal,
    ) -> bool:
        with lock:
            frame = registry.get(nonce)
            record = None if frame is None else frame[5].get(token)
        if frame is None or record is None or len(record) != 7:
            return False
        context_mac = hmac.new(
            mac_key, frame[3], hashlib.sha256
        ).hexdigest()
        stored_body_mac = hmac.new(
            mac_key, record[0], hashlib.sha256
        ).hexdigest()
        private_snapshot = record[2]
        try:
            private_content = _json_sha256(
                _snapshot_record(private_snapshot)
            )
            expected_key = _physical_key_from_immutable_inputs(
                frame_content_sha256=frame[0],
                numeric_platform_sha256=frame[1],
                implementation_sha256=frame[2],
                layer=private_snapshot.layer,
                predecessor_box=private_snapshot.predecessor_box,
                predecessor_id=private_snapshot.predecessor_id,
                box_semantics=private_snapshot.box_semantics,
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            frozen.QueryDualReplayError,
            ConvMaterialCacheError,
        ):
            return False
        return bool(
            type(private_snapshot) is _FastInputSnapshot
            and type(seal) is _FastInputSeal
            and token == private_snapshot.token == seal.token
            and private_snapshot.token
            == (
                id(private_snapshot.caller_layer),
                id(private_snapshot.caller_box),
                private_snapshot.predecessor_id,
                private_snapshot.box_semantics,
            )
            and record[4] == id(seal)
            and record[5] == token
            and record[6] is seal.snapshot
            and seal.signature
            == private_snapshot.caller_signature
            and hmac.compare_digest(frame[4], context_mac)
            and hmac.compare_digest(record[1], stored_body_mac)
            and hmac.compare_digest(
                private_snapshot.content_sha256, private_content
            )
            and hmac.compare_digest(record[3], expected_key)
            and hmac.compare_digest(
                expected_key, seal.physical_key_sha256
            )
        )

    def state(
        nonce: str,
    ) -> Optional[Tuple[int, int, frozenset[Tuple[Any, ...]]]]:
        with lock:
            frame = registry.get(nonce)
            if frame is None:
                return None
            values = frame[5]
            if not hmac.compare_digest(
                frame[4],
                hmac.new(
                    mac_key, frame[3], hashlib.sha256
                ).hexdigest(),
            ):
                return None
            return (id(values), len(values), frozenset(values))

    def drop(nonce: str) -> None:
        with lock:
            registry.pop(nonce, None)
            retired.add(nonce)

    return create, admit_once, validate, commit_validate, state, drop


(
    _fast_trust_create,
    _fast_trust_admit_once,
    _fast_trust_validate,
    _fast_trust_commit_validate,
    _fast_trust_state,
    _fast_trust_drop,
) = _build_fast_input_trust_root()


@dataclass
class _FrameRuntime:
    pid: int
    frame_id: int
    nonce: str
    capability: Any
    frame_content_sha256: str
    expected_stage_uses: Tuple[ConvMaterialStageUse, ...]
    expected_stage_set: frozenset[Tuple[str, Optional[int]]]
    stage_seals: Tuple[_StageSeal, ...]
    stages_by_id: Mapping[int, _StageSeal]
    stages_by_record: Mapping[
        Tuple[str, Optional[int]], _StageSeal
    ]
    numeric_platform_sha256: str
    implementation_manifest: Mapping[str, str]
    implementation_sha256: str
    deadline: frozen._Deadline
    deadline_id: int
    deadline_end_hex: str
    cores: Dict[str, _PhysicalConvCore]
    aliases: Dict[str, ConvMaterialStageAliasCandidate]
    aliases_by_use: Dict[
        Tuple[str, Optional[int], str], ConvMaterialStageAliasCandidate
    ]
    core_seals: Dict[str, _CoreSeal]
    alias_seals: Dict[str, _AliasSeal]
    fast_inputs: Dict[Tuple[Any, ...], _FastInputSeal]
    fast_input_external_id: int
    lock: threading.RLock
    static_identity: Tuple[Any, ...] = ()
    physical_builds: int = 0
    cross_stage_physical_hits: int = 0
    execution_alias_lookups: int = 0
    admission_full_validations: int = 0
    alias_mints: int = 0
    fast_input_mints: int = 0
    closed: bool = False
    poisoned: bool = False


def _physical_key_body(
    *,
    frame_content_sha256: str,
    layer_id: int,
    predecessor_id: int,
    weight_sha256: str,
    geometry_sha256: str,
    source_lb_sha256: str,
    source_ub_sha256: str,
    box_semantics: str,
    numeric_platform_sha256: str,
    implementation_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "frame_content_sha256": frame_content_sha256,
        "layer_id": int(layer_id),
        "predecessor_id": int(predecessor_id),
        "operator_kind": OPERATOR,
        "branch": BRANCH,
        "weight_sha256": weight_sha256,
        "conv_geometry_sha256": geometry_sha256,
        "source_lb_sha256": source_lb_sha256,
        "source_ub_sha256": source_ub_sha256,
        "box_semantics": box_semantics,
        "numeric_platform_sha256": numeric_platform_sha256,
        "implementation_sha256": implementation_sha256,
    }


def _binding_key_body(binding: _CoreBinding) -> Mapping[str, Any]:
    return _physical_key_body(
        frame_content_sha256=binding.frame_content_sha256,
        layer_id=binding.layer_id,
        predecessor_id=binding.predecessor_id,
        weight_sha256=binding.weight_sha256,
        geometry_sha256=binding.geometry_sha256,
        source_lb_sha256=binding.source_lb_sha256,
        source_ub_sha256=binding.source_ub_sha256,
        box_semantics=binding.box_semantics,
        numeric_platform_sha256=binding.numeric_platform_sha256,
        implementation_sha256=binding.implementation_sha256,
    )


def _plan_fast_structure(
    plan: conv_v51.DenseConvV51Plan,
) -> Tuple[Any, ...]:
    """Constant-size identity seal; no array hashing or offset traversal."""

    return (
        plan.layer_id,
        plan.input_shape,
        plan.output_shape,
        plan.stride,
        plan.padding,
        plan.dilation,
        plan.groups,
        plan.proof_authority,
        _array_identity(plan.weight),
        _array_identity(plan.support),
        id(plan.offsets),
        len(plan.offsets),
        id(plan.manifest),
        str(plan.manifest.get("content_sha256", "")),
    )


def _offsets_fast_structure(
    plan: conv_v51.DenseConvV51Plan,
) -> Tuple[Tuple[Any, ...], ...]:
    """Identity/metadata seal for every immutable physical offset."""

    return tuple(
        (
            id(offset),
            offset.group,
            offset.kh,
            offset.kw,
            offset.co_start,
            offset.co_end,
            offset.ci_start,
            offset.ci_end,
            _array_identity(offset.output_h_indices),
            _array_identity(offset.output_w_indices),
            _array_identity(offset.targets),
            _array_identity(offset.support_flat),
            _array_identity(offset.channel_support_flat),
            _array_identity(offset.support_activity_flat),
            float(offset.support_sum_upper).hex(),
        )
        for offset in plan.offsets
    )


def _binding_fast_structure(binding: _CoreBinding) -> Tuple[Any, ...]:
    return (
        binding.frame_content_sha256,
        binding.layer_id,
        binding.predecessor_id,
        binding.operator_kind,
        binding.branch,
        binding.weight_sha256,
        binding.geometry_sha256,
        binding.source_lb_sha256,
        binding.source_ub_sha256,
        binding.box_semantics,
        binding.numeric_platform_sha256,
        binding.implementation_sha256,
        _array_identity(binding.source_lb),
        _array_identity(binding.source_ub),
    )


def _array_identity(value: np.ndarray) -> Tuple[Any, ...]:
    return (
        id(value),
        value.dtype.str,
        tuple(value.shape),
        tuple(value.strides),
        bool(value.flags.writeable),
        _bytes_backed(value),
    )


def _fast_input_token(
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    predecessor_id: int,
    box_semantics: str,
) -> Tuple[Any, ...]:
    return (
        id(layer),
        id(predecessor_box),
        int(predecessor_id),
        str(box_semantics),
    )


def _exact_mappingproxy_backing(
    value: Any,
) -> Optional[Dict[str, Any]]:
    """Return an exact-dict proxy backing without invoking mapping methods."""

    if type(value) is not MappingProxyType:
        return None
    referents = gc.get_referents(value)
    if len(referents) != 1 or type(referents[0]) is not dict:
        return None
    return referents[0]


def _capture_fast_input(
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    predecessor_id: int,
    box_semantics: str,
) -> Optional[_FastInputCapture]:
    """Capture exact caller containers once; never call their methods."""

    try:
        if (
            type(layer) is not frozen._FrozenLayer
            or type(predecessor_box) is not frozen._Box
            or type(predecessor_id) is not int
            or type(box_semantics) is not str
        ):
            return None
        layer_dict = object.__getattribute__(layer, "__dict__")
        box_dict = object.__getattribute__(
            predecessor_box, "__dict__"
        )
        if type(layer_dict) is not dict or type(box_dict) is not dict:
            return None
        layer_fields = dict.copy(layer_dict)
        box_fields = dict.copy(box_dict)
        if set(layer_fields) != {
            "id",
            "kind",
            "preds",
            "width",
            "in_vars",
            "out_vars",
            "params",
        } or set(box_fields) != {"lb", "ub"}:
            return None
        params = layer_fields["params"]
        params_backing = _exact_mappingproxy_backing(params)
        if params_backing is None:
            return None
        params_fields = dict.copy(params_backing)
        if set(params_fields) != {
            "weight",
            "bias_channels",
            "input_shape",
            "output_shape",
            "stride",
            "padding",
            "dilation",
            "groups",
        }:
            return None
        capture = _FastInputCapture(
            layer=layer,
            predecessor_box=predecessor_box,
            layer_id=layer_fields["id"],
            layer_kind=layer_fields["kind"],
            layer_preds=layer_fields["preds"],
            layer_width=layer_fields["width"],
            layer_in_vars=layer_fields["in_vars"],
            layer_out_vars=layer_fields["out_vars"],
            params=params,
            params_backing_id=id(params_backing),
            weight=params_fields["weight"],
            bias=params_fields["bias_channels"],
            input_shape=params_fields["input_shape"],
            output_shape=params_fields["output_shape"],
            stride=params_fields["stride"],
            padding=params_fields["padding"],
            dilation=params_fields["dilation"],
            groups=params_fields["groups"],
            lb=box_fields["lb"],
            ub=box_fields["ub"],
            predecessor_id=predecessor_id,
            box_semantics=box_semantics,
        )
        if (
            capture.layer_kind != OPERATOR
            or capture.layer_preds != (predecessor_id,)
            or type(capture.weight) is not np.ndarray
            or type(capture.bias) is not np.ndarray
            or type(capture.lb) is not np.ndarray
            or type(capture.ub) is not np.ndarray
            or type(capture.layer_preds) is not tuple
            or type(capture.layer_in_vars) is not tuple
            or type(capture.layer_out_vars) is not tuple
            or any(
                type(value) is not tuple
                for value in (
                    capture.input_shape,
                    capture.output_shape,
                    capture.stride,
                    capture.padding,
                    capture.dilation,
                )
            )
            or type(capture.groups) is not int
            or type(capture.layer_id) is not int
            or type(capture.layer_kind) is not str
            or type(capture.layer_width) is not int
            or any(
                type(value) is not int
                for value in capture.layer_preds
            )
            or any(
                type(value) is not int
                for values in (
                    capture.input_shape,
                    capture.output_shape,
                    capture.stride,
                    capture.padding,
                    capture.dilation,
                )
                for value in values
            )
            or not all(
                _bytes_backed(value)
                for value in (
                    capture.weight,
                    capture.bias,
                    capture.lb,
                    capture.ub,
                )
            )
        ):
            return None
        return capture
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


def _fast_signature_parts(
    *,
    capture: _FastInputCapture,
) -> Tuple[Any, ...]:
    return (
        id(capture.layer),
        capture.layer_id,
        capture.layer_kind,
        capture.layer_preds,
        capture.layer_width,
        id(capture.layer_in_vars),
        id(capture.layer_out_vars),
        id(capture.params),
        capture.params_backing_id,
        _array_identity(capture.weight),
        _array_identity(capture.bias),
        capture.input_shape,
        capture.output_shape,
        capture.stride,
        capture.padding,
        capture.dilation,
        capture.groups,
        id(capture.predecessor_box),
        _array_identity(capture.lb),
        _array_identity(capture.ub),
        capture.predecessor_id,
        capture.box_semantics,
    )


def _fast_input_signature(
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    predecessor_id: int,
    box_semantics: str,
) -> Optional[Tuple[Any, ...]]:
    """Return an O(1) signature only for immutable-bytes source objects."""

    capture = _capture_fast_input(
        layer,
        predecessor_box,
        predecessor_id,
        box_semantics,
    )
    if capture is None:
        return None
    return _fast_signature_parts(capture=capture)


def _snapshot_record(
    snapshot: _FastInputSnapshot,
) -> Mapping[str, Any]:
    return {
        "schema": "act.query_dual_v51b_fast_input_snapshot.v1",
        "token": snapshot.token,
        "caller_signature": snapshot.caller_signature,
        "predecessor_id": snapshot.predecessor_id,
        "box_semantics": snapshot.box_semantics,
        "layer_id": snapshot.layer.id,
        "weight_sha256": frozen._array_digest(
            snapshot.layer.params["weight"]
        ),
        "geometry_sha256": _json_sha256(
            _geometry_body_from_layer(snapshot.layer)
        ),
        "source_lb_sha256": frozen._array_digest(
            snapshot.predecessor_box.lb
        ),
        "source_ub_sha256": frozen._array_digest(
            snapshot.predecessor_box.ub
        ),
    }


def _snapshot_fast_input(
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    predecessor_id: int,
    box_semantics: str,
) -> _FastInputSnapshot:
    """Capture caller state once into exact, root-owned immutable objects."""

    capture = _capture_fast_input(
        layer,
        predecessor_box,
        predecessor_id,
        box_semantics,
    )
    if capture is None or box_semantics not in _BOX_SEMANTICS:
        _fail(
            "FAST_INPUT_BINDING",
            "caller input must use exact immutable replay types",
        )
    signature = _fast_signature_parts(capture=capture)
    weight = _immutable_f64(
        capture.weight, name="V5.1b snapshot weight"
    )
    bias = _immutable_f64(
        capture.bias, name="V5.1b snapshot bias"
    )
    snapshot_params = MappingProxyType(
        {
            "weight": weight,
            "bias_channels": bias,
            "input_shape": capture.input_shape,
            "output_shape": capture.output_shape,
            "stride": capture.stride,
            "padding": capture.padding,
            "dilation": capture.dilation,
            "groups": capture.groups,
        }
    )
    snapshot_layer = frozen._FrozenLayer(
        id=capture.layer_id,
        kind=capture.layer_kind,
        preds=capture.layer_preds,
        width=capture.layer_width,
        in_vars=capture.layer_in_vars,
        out_vars=capture.layer_out_vars,
        params=snapshot_params,
    )
    snapshot_box = frozen._Box(
        lb=_immutable_f64(
            capture.lb, name="V5.1b snapshot lower"
        ),
        ub=_immutable_f64(
            capture.ub, name="V5.1b snapshot upper"
        ),
    )
    if np.any(snapshot_box.lb > snapshot_box.ub):
        _fail("INVALID_BOX", "source box is malformed")
    token = _fast_input_token(
        capture.layer,
        capture.predecessor_box,
        capture.predecessor_id,
        capture.box_semantics,
    )
    placeholder = _FastInputSnapshot(
        token=token,
        caller_signature=signature,
        predecessor_id=int(predecessor_id),
        box_semantics=str(box_semantics),
        layer=snapshot_layer,
        predecessor_box=snapshot_box,
        caller_layer=layer,
        caller_box=predecessor_box,
        content_sha256="",
    )
    content_sha = _json_sha256(_snapshot_record(placeholder))
    result = _FastInputSnapshot(
        token=token,
        caller_signature=signature,
        predecessor_id=int(predecessor_id),
        box_semantics=str(box_semantics),
        layer=snapshot_layer,
        predecessor_box=snapshot_box,
        caller_layer=layer,
        caller_box=predecessor_box,
        content_sha256=content_sha,
    )
    if (
        _json_sha256(_snapshot_record(result)) != content_sha
        or _fast_input_signature(
            layer,
            predecessor_box,
            predecessor_id,
            box_semantics,
        )
        != signature
    ):
        _fail(
            "FAST_INPUT_BINDING",
            "caller changed during root-owned snapshot",
        )
    return result


def _clone_fast_snapshot(
    snapshot: _FastInputSnapshot,
) -> _FastInputSnapshot:
    """Make a closure-private snapshot independent of the public seal."""

    if type(snapshot) is not _FastInputSnapshot:
        _fail("FAST_INPUT_BINDING", "invalid fast-input snapshot")
    layer = snapshot.layer
    box = snapshot.predecessor_box
    params = MappingProxyType(
        {
            "weight": _immutable_f64(
                layer.params["weight"],
                name="V5.1b trust weight",
            ),
            "bias_channels": _immutable_f64(
                layer.params["bias_channels"],
                name="V5.1b trust bias",
            ),
            "input_shape": tuple(layer.params["input_shape"]),
            "output_shape": tuple(layer.params["output_shape"]),
            "stride": tuple(layer.params["stride"]),
            "padding": tuple(layer.params["padding"]),
            "dilation": tuple(layer.params["dilation"]),
            "groups": int(layer.params["groups"]),
        }
    )
    private_layer = frozen._FrozenLayer(
        id=layer.id,
        kind=layer.kind,
        preds=layer.preds,
        width=layer.width,
        in_vars=layer.in_vars,
        out_vars=layer.out_vars,
        params=params,
    )
    private_box = frozen._Box(
        lb=_immutable_f64(box.lb, name="V5.1b trust lower"),
        ub=_immutable_f64(box.ub, name="V5.1b trust upper"),
    )
    private = _FastInputSnapshot(
        token=snapshot.token,
        caller_signature=snapshot.caller_signature,
        predecessor_id=snapshot.predecessor_id,
        box_semantics=snapshot.box_semantics,
        layer=private_layer,
        predecessor_box=private_box,
        caller_layer=snapshot.caller_layer,
        caller_box=snapshot.caller_box,
        content_sha256=snapshot.content_sha256,
    )
    if (
        _json_sha256(_snapshot_record(private))
        != snapshot.content_sha256
    ):
        _fail(
            "FAST_INPUT_BINDING",
            "closure-private snapshot differs from admission",
        )
    return private


def _physical_key_from_immutable_inputs(
    *,
    frame_content_sha256: str,
    numeric_platform_sha256: str,
    implementation_sha256: str,
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    predecessor_id: int,
    box_semantics: str,
) -> str:
    """Independently reconstruct one semantic key with exact SHA inputs."""

    signature = _fast_input_signature(
        layer,
        predecessor_box,
        predecessor_id,
        box_semantics,
    )
    if signature is None or box_semantics not in _BOX_SEMANTICS:
        _fail(
            "FAST_INPUT_BINDING",
            "fast input is not immutable or semantically valid",
        )
    source = _source_box(predecessor_box)
    body = _physical_key_body(
        frame_content_sha256=frame_content_sha256,
        layer_id=layer.id,
        predecessor_id=predecessor_id,
        weight_sha256=frozen._array_digest(
            layer.params["weight"]
        ),
        geometry_sha256=_json_sha256(
            _geometry_body_from_layer(layer)
        ),
        source_lb_sha256=frozen._array_digest(source.lb),
        source_ub_sha256=frozen._array_digest(source.ub),
        box_semantics=box_semantics,
        numeric_platform_sha256=numeric_platform_sha256,
        implementation_sha256=implementation_sha256,
    )
    return _json_sha256(body)


def _fast_input_external_body(
    seal: _FastInputSeal,
) -> Mapping[str, Any]:
    return {
        "schema": (
            "act.query_dual_v51b_fast_input_external_seal.v1"
        ),
        "fast_input_id": id(seal),
        "token": seal.token,
        "signature": seal.signature,
        "physical_key_sha256": seal.physical_key_sha256,
        "layer_id": id(seal.layer),
        "predecessor_box_id": id(seal.predecessor_box),
        "snapshot_id": id(seal.snapshot),
        "snapshot_content_sha256": seal.snapshot.content_sha256,
    }


def _require_bytes_backed_plan(
    plan: conv_v51.DenseConvV51Plan,
) -> None:
    arrays = [plan.weight, plan.support]
    for offset in plan.offsets:
        arrays.extend(
            (
                offset.output_h_indices,
                offset.output_w_indices,
                offset.targets,
                offset.support_flat,
                offset.channel_support_flat,
                offset.support_activity_flat,
            )
        )
    if not all(_bytes_backed(value) for value in arrays):
        _fail(
            "MUTABLE_CORE",
            "physical Conv plan must have immutable-bytes array backing",
        )


def _core_receipt_body(
    key_sha256: str,
    plan: conv_v51.DenseConvV51Plan,
) -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "proof_authority": False,
        "physical_key_sha256": key_sha256,
        "plan_content_sha256": str(plan.manifest["content_sha256"]),
        "weight_sha256": frozen._array_digest(plan.weight),
        "support_sha256": frozen._array_digest(plan.support),
        "offset_count": len(plan.offsets),
    }


def _make_core_seal(core: _PhysicalConvCore) -> _CoreSeal:
    return _CoreSeal(
        core_id=id(core),
        binding_id=id(core.binding),
        plan_id=id(core.plan),
        receipt_id=id(core.receipt_bytes),
        receipt_bytes=core.receipt_bytes,
        binding_fast=_binding_fast_structure(core.binding),
        plan_fast=_plan_fast_structure(core.plan),
        offsets_fast=_offsets_fast_structure(core.plan),
        key_sha256=core.key_sha256,
        content_sha256=core.content_sha256,
    )


def _stage_record(
    stage_use: ConvMaterialStageUse,
) -> Mapping[str, Any]:
    return {
        "stage_use_sha256": stage_use.stage_use_sha256,
        "cone_start_lid": stage_use.cone_start_lid,
    }


def _make_stage_seal(stage_use: ConvMaterialStageUse) -> _StageSeal:
    record_bytes = bytes(_json_bytes(_stage_record(stage_use)))
    return _StageSeal(
        stage_id=id(stage_use),
        stage=stage_use,
        stage_use_sha256=stage_use.stage_use_sha256,
        cone_start_lid=stage_use.cone_start_lid,
        record_bytes=record_bytes,
        content_sha256=hashlib.sha256(record_bytes).hexdigest(),
    )


def _alias_body(
    runtime: _FrameRuntime,
    stage_seal: _StageSeal,
    core: _PhysicalConvCore,
) -> Dict[str, Any]:
    return {
        "schema": ALIAS_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "proof_authority": False,
        "frame_nonce": runtime.nonce,
        "frame_content_sha256": runtime.frame_content_sha256,
        "stage_use_sha256": stage_seal.stage_use_sha256,
        "stage_content_sha256": stage_seal.content_sha256,
        "cone_start_lid": stage_seal.cone_start_lid,
        "physical_key_sha256": core.key_sha256,
        "physical_core_content_sha256": core.content_sha256,
        "layer_id": core.binding.layer_id,
        "predecessor_id": core.binding.predecessor_id,
    }


def _make_alias_seal(
    alias: ConvMaterialStageAliasCandidate,
) -> _AliasSeal:
    return _AliasSeal(
        alias_id=id(alias),
        receipt_id=id(alias.receipt),
        receipt_bytes=bytes(_json_bytes(dict(alias.receipt))),
        frame_nonce=alias._frame_nonce,
        frame_content_sha256=alias.frame_content_sha256,
        stage_use_sha256=alias.stage_use_sha256,
        cone_start_lid=alias.cone_start_lid,
        physical_key_sha256=alias.physical_key_sha256,
        physical_core_content_sha256=alias.physical_core_content_sha256,
        layer_id=alias.layer_id,
        predecessor_id=alias.predecessor_id,
    )


def _drop_frame(nonce: str) -> None:
    if os.getpid() != _PID:
        return
    with _FRAME_LOCK:
        runtime = _FRAME_RUNTIMES.pop(nonce, None)
        _FRAME_REGISTRY.pop(nonce, None)
    _fast_trust_drop(nonce)
    if runtime is None:
        return
    with runtime.lock:
        runtime.closed = True
        runtime.cores.clear()
        runtime.aliases.clear()
        runtime.aliases_by_use.clear()
        runtime.core_seals.clear()
        runtime.alias_seals.clear()
        runtime.fast_inputs.clear()


def _runtime_for(frame: "ConvMaterialFrameCandidate") -> _FrameRuntime:
    if os.getpid() != _PID:
        _fail("FORKED_PROCESS", "cache frame cannot cross a fork")
    if (
        type(frame) is not ConvMaterialFrameCandidate
        or frame._capability is not _FRAME_CAPABILITY
    ):
        _fail("INVALID_FRAME", "invalid cache frame capability")
    with _FRAME_LOCK:
        runtime = _FRAME_RUNTIMES.get(frame._nonce)
        registered = _FRAME_REGISTRY.get(frame._nonce)
    if runtime is None or registered is not frame:
        _fail("INVALID_FRAME", "cache frame is not externally registered")
    return runtime


def _poison(runtime: _FrameRuntime, code: str, message: str) -> NoReturn:
    runtime.poisoned = True
    runtime.closed = True
    _fail(code, message)


def _checked_fast_trust_state(
    runtime: _FrameRuntime,
) -> Tuple[int, int, frozenset[Tuple[Any, ...]]]:
    state = _fast_trust_state(runtime.nonce)
    if (
        state is None
        or state[0] != runtime.fast_input_external_id
    ):
        _poison(
            runtime,
            "FAST_INPUT_EXTERNAL_SEAL_MISMATCH",
            "external fast-input seal registry changed",
        )
    return state


def _check_static_locked(
    frame: "ConvMaterialFrameCandidate", runtime: _FrameRuntime
) -> None:
    if runtime.poisoned:
        _fail("POISONED_FRAME", "cache frame previously failed closed")
    if runtime.closed:
        _fail("CLOSED_FRAME", "cache frame is already closed")
    if (
        runtime.pid != _PID
        or runtime.frame_id != id(frame)
        or runtime.capability is not _FRAME_CAPABILITY
        or frame._nonce != runtime.nonce
        or frame._frame_content_sha256 != runtime.frame_content_sha256
        or frame._expected_stage_uses is not runtime.expected_stage_uses
        or frame._numeric_platform_sha256
        != runtime.numeric_platform_sha256
        or frame._implementation_manifest
        is not runtime.implementation_manifest
        or frame._implementation_sha256
        != runtime.implementation_sha256
        or frame._deadline is not runtime.deadline
        or id(frame._deadline) != runtime.deadline_id
        or (
            "none"
            if runtime.deadline.end is None
            else float(runtime.deadline.end).hex()
        )
        != runtime.deadline_end_hex
        or frame._cores is not runtime.cores
        or frame._aliases is not runtime.aliases
        or frame._aliases_by_use is not runtime.aliases_by_use
        or frame._lock is not runtime.lock
        or _checked_fast_trust_state(runtime)[0]
        != runtime.fast_input_external_id
        or runtime.static_identity
        != (
            id(runtime.expected_stage_uses),
            id(runtime.stage_seals),
            id(runtime.stages_by_id),
            id(runtime.stages_by_record),
            id(runtime.cores),
            id(runtime.aliases),
            id(runtime.aliases_by_use),
            id(runtime.core_seals),
            id(runtime.alias_seals),
            id(runtime.fast_inputs),
            runtime.fast_input_external_id,
            id(runtime.lock),
        )
    ):
        _poison(
            runtime,
            "FRAME_SEAL_MISMATCH",
            "external frame identity/content seal changed",
        )


def _check_membership_counts_locked(runtime: _FrameRuntime) -> None:
    """O(1) deletion/insertion/counter invariants for every fast operation."""

    fast_trust_state = _checked_fast_trust_state(runtime)
    if (
        runtime.physical_builds < 0
        or runtime.alias_mints < 0
        or runtime.cross_stage_physical_hits < 0
        or runtime.execution_alias_lookups < 0
        or runtime.admission_full_validations < 0
        or runtime.fast_input_mints < 0
        or len(runtime.cores) != runtime.physical_builds
        or len(runtime.core_seals) != runtime.physical_builds
        or runtime.admission_full_validations
        != runtime.physical_builds
        or len(runtime.aliases) != runtime.alias_mints
        or len(runtime.alias_seals) != runtime.alias_mints
        or len(runtime.aliases_by_use) != runtime.alias_mints
        or runtime.cross_stage_physical_hits
        != runtime.alias_mints - runtime.physical_builds
        or len(runtime.fast_inputs) != runtime.fast_input_mints
        or fast_trust_state[1] != runtime.fast_input_mints
    ):
        _poison(
            runtime,
            "MEMBERSHIP_MISMATCH",
            "cache registries, seals, or counters diverged",
        )


def _check_stage_seal_locked(
    runtime: _FrameRuntime,
    stage_use: ConvMaterialStageUse,
) -> _StageSeal:
    seal = runtime.stages_by_id.get(id(stage_use))
    try:
        current_bytes = _json_bytes(_stage_record(stage_use))
        valid = bool(
            isinstance(stage_use, ConvMaterialStageUse)
            and seal is not None
            and seal.stage is stage_use
            and seal.stage_id == id(stage_use)
            and seal.stage_use_sha256 == stage_use.stage_use_sha256
            and seal.cone_start_lid == stage_use.cone_start_lid
            and seal.record_bytes == current_bytes
            and hmac.compare_digest(
                seal.content_sha256,
                hashlib.sha256(current_bytes).hexdigest(),
            )
            and runtime.stages_by_record.get(
                (seal.stage_use_sha256, seal.cone_start_lid)
            )
            is seal
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        valid = False
    if not valid or seal is None:
        _poison(
            runtime,
            "STAGE_SEAL_MISMATCH",
            "stage use was copied, reconstructed, or mutated",
        )
    return seal


def _check_all_stage_seals_locked(runtime: _FrameRuntime) -> None:
    if (
        len(runtime.stage_seals) != len(runtime.expected_stage_uses)
        or len(runtime.stages_by_id) != len(runtime.stage_seals)
        or len(runtime.stages_by_record) != len(runtime.stage_seals)
    ):
        _poison(
            runtime,
            "STAGE_SEAL_MISMATCH",
            "stage seal membership changed",
        )
    for expected, seal in zip(
        runtime.expected_stage_uses, runtime.stage_seals
    ):
        if seal.stage is not expected:
            _poison(
                runtime,
                "STAGE_SEAL_MISMATCH",
                "stage tuple order or identity changed",
            )
        _check_stage_seal_locked(runtime, expected)


def _reconstructed_key(
    runtime: _FrameRuntime, core: _PhysicalConvCore
) -> str:
    plan = core.plan
    binding = core.binding
    body = _physical_key_body(
        frame_content_sha256=runtime.frame_content_sha256,
        layer_id=plan.layer_id,
        predecessor_id=binding.predecessor_id,
        weight_sha256=frozen._array_digest(plan.weight),
        geometry_sha256=_json_sha256(_geometry_body_from_plan(plan)),
        source_lb_sha256=frozen._array_digest(binding.source_lb),
        source_ub_sha256=frozen._array_digest(binding.source_ub),
        box_semantics=binding.box_semantics,
        numeric_platform_sha256=runtime.numeric_platform_sha256,
        implementation_sha256=runtime.implementation_sha256,
    )
    return _json_sha256(body)


def _validate_core_fast_locked(
    runtime: _FrameRuntime, key: str, core: _PhysicalConvCore
) -> None:
    """O(1) core identity check: no array digest and no math validator."""

    seal = runtime.core_seals.get(key)
    try:
        manifest_content = str(core.plan.manifest["content_sha256"])
        valid = bool(
            isinstance(core, _PhysicalConvCore)
            and core.proof_authority is False
            and seal is not None
            and seal.core_id == id(core)
            and seal.binding_id == id(core.binding)
            and seal.plan_id == id(core.plan)
            and seal.receipt_id == id(core.receipt_bytes)
            and seal.receipt_bytes is core.receipt_bytes
            and seal.binding_fast
            == _binding_fast_structure(core.binding)
            and seal.plan_fast == _plan_fast_structure(core.plan)
            and seal.offsets_fast
            == _offsets_fast_structure(core.plan)
            and key == core.key_sha256 == seal.key_sha256
            and core.content_sha256
            == seal.content_sha256
            and core.plan_content_sha256 == manifest_content
            and core.binding.frame_content_sha256
            == runtime.frame_content_sha256
            and core.binding.numeric_platform_sha256
            == runtime.numeric_platform_sha256
            and core.binding.implementation_sha256
            == runtime.implementation_sha256
            and core.binding.operator_kind == OPERATOR
            and core.binding.branch == BRANCH
            and core.binding.box_semantics in _BOX_SEMANTICS
            and _bytes_backed(core.binding.source_lb)
            and _bytes_backed(core.binding.source_ub)
            and _bytes_backed(core.plan.weight)
            and _bytes_backed(core.plan.support)
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        valid = False
    if not valid:
        _poison(
            runtime,
            "CORE_SEAL_MISMATCH",
            "physical Conv core identity/content changed",
        )


def _validate_core_full_content_locked(
    runtime: _FrameRuntime, key: str, core: _PhysicalConvCore
) -> None:
    """Validate content after exactly one caller-owned mathematical check."""

    _validate_core_fast_locked(runtime, key, core)
    try:
        receipt_body = _core_receipt_body(key, core.plan)
        expected_receipt = _json_bytes(receipt_body)
        expected_content = _json_sha256(receipt_body)
        valid = bool(
            core.receipt_bytes == expected_receipt
            and core.content_sha256 == expected_content
            and _json_sha256(_binding_key_body(core.binding)) == key
            and _reconstructed_key(runtime, core) == key
        )
        _require_bytes_backed_plan(core.plan)
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        valid = False
    if not valid:
        _poison(
            runtime,
            "CORE_CONTENT_MISMATCH",
            "physical Conv core failed full content reconstruction",
        )


def _validate_fast_input_locked(
    runtime: _FrameRuntime,
    token: Tuple[Any, ...],
    *,
    layer: frozen._FrozenLayer,
    predecessor_box: frozen._Box,
    predecessor_id: int,
    box_semantics: str,
) -> Optional[_PhysicalConvCore]:
    seal = runtime.fast_inputs.get(token)
    if seal is None:
        return None
    if type(seal) is not _FastInputSeal:
        _poison(
            runtime,
            "FAST_INPUT_SEAL_MISMATCH",
            "fast input registry contains a non-exact seal",
        )
    signature = _fast_input_signature(
        layer, predecessor_box, predecessor_id, box_semantics
    )
    snapshot = seal.snapshot
    if (
        not _fast_trust_validate(runtime.nonce, token, seal)
        or seal.token != token
        or type(snapshot) is not _FastInputSnapshot
        or snapshot.token != token
        or snapshot.caller_layer is not layer
        or snapshot.caller_box is not predecessor_box
        or snapshot.predecessor_id != predecessor_id
        or snapshot.box_semantics != box_semantics
        or seal.layer is not snapshot.layer
        or seal.predecessor_box is not snapshot.predecessor_box
        or signature is None
        or seal.signature != signature
        or signature != snapshot.caller_signature
    ):
        _poison(
            runtime,
            "FAST_INPUT_SEAL_MISMATCH",
            "immutable source identity changed on a cache hit",
        )
    core = runtime.cores.get(seal.physical_key_sha256)
    if core is None:
        _poison(
            runtime,
            "MEMBERSHIP_MISMATCH",
            "fast input references a missing physical core",
        )
    _validate_core_fast_locked(runtime, core.key_sha256, core)
    return core


def _validate_fast_input_full_key_locked(
    runtime: _FrameRuntime,
    token: Tuple[Any, ...],
    seal: _FastInputSeal,
) -> None:
    """Commit-only independent semantic-key reconstruction."""

    if not _fast_trust_commit_validate(
        runtime.nonce, token, seal
    ):
        _poison(
            runtime,
            "FAST_INPUT_SEAL_MISMATCH",
            "closure-private snapshot rejected fast-input key",
        )
    core = runtime.cores.get(seal.physical_key_sha256)
    if core is None:
        _poison(
            runtime,
            "FAST_INPUT_BINDING",
            "closure-private key references no physical core",
        )
    _validate_core_fast_locked(runtime, core.key_sha256, core)


def _validate_alias_locked(
    runtime: _FrameRuntime,
    alias: ConvMaterialStageAliasCandidate,
) -> _PhysicalConvCore:
    try:
        registered = runtime.aliases.get(alias._nonce)
        seal = runtime.alias_seals.get(alias._nonce)
        stage_seal = runtime.stages_by_record.get(
            (alias.stage_use_sha256, alias.cone_start_lid)
        )
        if stage_seal is not None:
            _check_stage_seal_locked(runtime, stage_seal.stage)
        core = runtime.cores.get(alias.physical_key_sha256)
        expected_body = (
            None
            if stage_seal is None or core is None
            else _alias_body(runtime, stage_seal, core)
        )
        if expected_body is not None:
            expected_body["content_sha256"] = _json_sha256(expected_body)
        expected_bytes = (
            b"" if expected_body is None else _json_bytes(expected_body)
        )
        valid = bool(
            isinstance(alias, ConvMaterialStageAliasCandidate)
            and alias.proof_authority is False
            and alias._capability is _ALIAS_CAPABILITY
            and alias._frame_nonce == runtime.nonce
            and registered is alias
            and seal is not None
            and seal.alias_id == id(alias)
            and seal.receipt_id == id(alias.receipt)
            and seal.receipt_bytes == expected_bytes
            and seal.frame_nonce == runtime.nonce == alias._frame_nonce
            and seal.frame_content_sha256
            == runtime.frame_content_sha256
            == alias.frame_content_sha256
            and seal.stage_use_sha256 == alias.stage_use_sha256
            and seal.cone_start_lid == alias.cone_start_lid
            and seal.physical_key_sha256 == alias.physical_key_sha256
            and seal.physical_core_content_sha256
            == alias.physical_core_content_sha256
            and seal.layer_id == alias.layer_id
            and seal.predecessor_id == alias.predecessor_id
            and stage_seal is not None
            and core is not None
            and alias.layer_id == core.binding.layer_id
            and alias.predecessor_id == core.binding.predecessor_id
            and alias.physical_core_content_sha256
            == core.content_sha256
            and type(alias.receipt) is MappingProxyType
            and dict(alias.receipt) == expected_body
            and _json_bytes(dict(alias.receipt)) == expected_bytes
            and runtime.aliases_by_use.get(
                (
                    alias.stage_use_sha256,
                    alias.cone_start_lid,
                    alias.physical_key_sha256,
                )
            )
            is alias
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        valid = False
        core = None
    if not valid or core is None:
        _poison(
            runtime,
            "ALIAS_SEAL_MISMATCH",
            "stage-local alias was copied, transplanted, or mutated",
        )
    _validate_core_fast_locked(runtime, core.key_sha256, core)
    return core


def _check_full_membership_locked(runtime: _FrameRuntime) -> None:
    _check_membership_counts_locked(runtime)
    fast_trust_state = _checked_fast_trust_state(runtime)
    if (
        set(runtime.cores) != set(runtime.core_seals)
        or set(runtime.aliases) != set(runtime.alias_seals)
        or set(runtime.fast_inputs) != set(fast_trust_state[2])
        or {
            alias._nonce for alias in runtime.aliases_by_use.values()
        }
        != set(runtime.aliases)
    ):
        _poison(
            runtime,
            "MEMBERSHIP_MISMATCH",
            "registry key membership changed",
        )
    referenced_cores = set()
    for nonce, alias in runtime.aliases.items():
        if nonce != alias._nonce:
            _poison(
                runtime,
                "MEMBERSHIP_MISMATCH",
                "alias registry key changed",
            )
        core = _validate_alias_locked(runtime, alias)
        referenced_cores.add(core.key_sha256)
    if referenced_cores != set(runtime.cores):
        _poison(
            runtime,
            "MEMBERSHIP_MISMATCH",
            "physical core/alias coverage changed",
        )
    for token, seal in runtime.fast_inputs.items():
        if (
            type(seal) is not _FastInputSeal
            or token != seal.token
            or seal.physical_key_sha256 not in runtime.cores
        ):
            _poison(
                runtime,
                "MEMBERSHIP_MISMATCH",
                "fast input registry changed",
            )
        _validate_fast_input_full_key_locked(
            runtime, token, seal
        )


class ConvMaterialFrameCandidate:
    """One frame-global physical Conv registry with stage-local aliases."""

    __slots__ = (
        "_nonce",
        "_frame_content_sha256",
        "_expected_stage_uses",
        "_numeric_platform_sha256",
        "_implementation_manifest",
        "_implementation_sha256",
        "_deadline",
        "_cores",
        "_aliases",
        "_aliases_by_use",
        "_lock",
        "_capability",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        frame_content_sha256: str,
        expected_stage_uses: Iterable[ConvMaterialStageUse],
        deadline: frozen._Deadline,
    ) -> None:
        if os.getpid() != _PID:
            _fail("FORKED_PROCESS", "cannot create frame after module fork")
        frame_sha = _require_sha256(
            frame_content_sha256, name="frame_content_sha256"
        )
        if not isinstance(deadline, frozen._Deadline):
            _fail("INVALID_DEADLINE", "deadline must be a replay deadline")
        if (
            deadline.end is None
            or not math.isfinite(float(deadline.end))
        ):
            _fail(
                "INVALID_DEADLINE",
                "V5.1b requires one finite absolute deadline",
            )
        try:
            deadline.check(force=True)
        except frozen.QueryDualReplayTimeout as exc:
            raise ConvMaterialCacheTimeout() from exc
        stages = tuple(expected_stage_uses)
        if (
            not stages
            or any(
                not isinstance(value, ConvMaterialStageUse)
                for value in stages
            )
            or len(
                {
                    (value.stage_use_sha256, value.cone_start_lid)
                    for value in stages
                }
            )
            != len(stages)
        ):
            _fail(
                "INVALID_STAGE_SET",
                "expected stage uses must be nonempty and unique",
            )
        platform = dict(conv_v51._wide_platform())
        platform_sha = _json_sha256(platform)
        implementation = _implementation_manifest()
        implementation_sha = _json_sha256(dict(implementation))
        stage_seals = tuple(_make_stage_seal(value) for value in stages)
        stages_by_id = MappingProxyType(
            {value.stage_id: value for value in stage_seals}
        )
        stages_by_record = MappingProxyType(
            {
                (value.stage_use_sha256, value.cone_start_lid): value
                for value in stage_seals
            }
        )
        nonce = secrets.token_hex(32)
        cores: Dict[str, _PhysicalConvCore] = {}
        aliases: Dict[str, ConvMaterialStageAliasCandidate] = {}
        aliases_by_use: Dict[
            Tuple[str, Optional[int], str],
            ConvMaterialStageAliasCandidate,
        ] = {}
        core_seals: Dict[str, _CoreSeal] = {}
        alias_seals: Dict[str, _AliasSeal] = {}
        fast_inputs: Dict[Tuple[Any, ...], _FastInputSeal] = {}
        fast_trust_id = _fast_trust_create(
            nonce,
            frame_content_sha256=frame_sha,
            numeric_platform_sha256=platform_sha,
            implementation_sha256=implementation_sha,
        )
        if fast_trust_id is None:
            _fail(
                "FAST_INPUT_EXTERNAL_SEAL_MISMATCH",
                "could not create append-only fast-input trust root",
            )
        lock = threading.RLock()

        self._nonce = nonce
        self._frame_content_sha256 = frame_sha
        self._expected_stage_uses = stages
        self._numeric_platform_sha256 = platform_sha
        self._implementation_manifest = implementation
        self._implementation_sha256 = implementation_sha
        self._deadline = deadline
        self._cores = cores
        self._aliases = aliases
        self._aliases_by_use = aliases_by_use
        self._lock = lock
        self._capability = _FRAME_CAPABILITY

        runtime = _FrameRuntime(
            pid=_PID,
            frame_id=id(self),
            nonce=nonce,
            capability=_FRAME_CAPABILITY,
            frame_content_sha256=frame_sha,
            expected_stage_uses=stages,
            expected_stage_set=frozenset(
                (value.stage_use_sha256, value.cone_start_lid)
                for value in stages
            ),
            stage_seals=stage_seals,
            stages_by_id=stages_by_id,
            stages_by_record=stages_by_record,
            numeric_platform_sha256=platform_sha,
            implementation_manifest=implementation,
            implementation_sha256=implementation_sha,
            deadline=deadline,
            deadline_id=id(deadline),
            deadline_end_hex=(
                "none"
                if deadline.end is None
                else float(deadline.end).hex()
            ),
            cores=cores,
            aliases=aliases,
            aliases_by_use=aliases_by_use,
            core_seals=core_seals,
            alias_seals=alias_seals,
            fast_inputs=fast_inputs,
            fast_input_external_id=fast_trust_id,
            lock=lock,
        )
        runtime.static_identity = (
            id(runtime.expected_stage_uses),
            id(runtime.stage_seals),
            id(runtime.stages_by_id),
            id(runtime.stages_by_record),
            id(runtime.cores),
            id(runtime.aliases),
            id(runtime.aliases_by_use),
            id(runtime.core_seals),
            id(runtime.alias_seals),
            id(runtime.fast_inputs),
            runtime.fast_input_external_id,
            id(runtime.lock),
        )
        _check_deadline(runtime)
        with _FRAME_LOCK:
            _FRAME_REGISTRY[nonce] = self
            _FRAME_RUNTIMES[nonce] = runtime
        weakref.finalize(self, _drop_frame, nonce)

    def __copy__(self) -> NoReturn:
        _fail("COPY_FORBIDDEN", "cache frame capabilities cannot be copied")

    def __deepcopy__(self, memo: Any) -> NoReturn:
        del memo
        _fail("COPY_FORBIDDEN", "cache frame capabilities cannot be copied")

    @property
    def frame_content_sha256(self) -> str:
        return self._frame_content_sha256

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def counters(self) -> Mapping[str, int]:
        runtime = _runtime_for(self)
        with runtime.lock:
            _check_static_locked(self, runtime)
            _check_membership_counts_locked(runtime)
            _check_deadline(runtime)
            result = MappingProxyType(
                {
                    "physical_builds": runtime.physical_builds,
                    "cross_stage_physical_hits": (
                        runtime.cross_stage_physical_hits
                    ),
                    "execution_alias_lookups": (
                        runtime.execution_alias_lookups
                    ),
                    "stage_aliases": len(runtime.aliases),
                    "admission_full_validations": (
                        runtime.admission_full_validations
                    ),
                }
            )
            _check_deadline(runtime)
            return result

    def admit(
        self,
        *,
        stage_use: ConvMaterialStageUse,
        layer: frozen._FrozenLayer,
        predecessor_id: int,
        predecessor_box: frozen._Box,
        box_semantics: str,
    ) -> ConvMaterialStageAliasCandidate:
        """Admit/reuse one physical plan and mint a stage-local alias."""

        runtime = _runtime_for(self)
        with runtime.lock:
            _check_static_locked(self, runtime)
            _check_membership_counts_locked(runtime)
            _check_deadline(runtime)
            stage_seal = _check_stage_seal_locked(runtime, stage_use)
            if (
                type(layer) is not frozen._FrozenLayer
                or layer.kind != OPERATOR
                or type(layer.params) is not MappingProxyType
                or type(predecessor_box) is not frozen._Box
                or type(predecessor_id) is not int
                or layer.preds != (predecessor_id,)
                or type(box_semantics) is not str
            ):
                _poison(
                    runtime,
                    "INVALID_LAYER",
                    "Conv layer/predecessor binding is malformed",
                )
            semantics = box_semantics
            if semantics not in _BOX_SEMANTICS:
                _poison(
                    runtime,
                    "INVALID_BOX_SEMANTICS",
                    "unknown source-box semantics",
                )
            token = _fast_input_token(
                layer, predecessor_box, predecessor_id, semantics
            )
            core = _validate_fast_input_locked(
                runtime,
                token,
                layer=layer,
                predecessor_box=predecessor_box,
                predecessor_id=predecessor_id,
                box_semantics=semantics,
            )
            core_was_cached = core is not None
            if core is None:
                snapshot = _snapshot_fast_input(
                    layer,
                    predecessor_box,
                    predecessor_id,
                    semantics,
                )
                snapshot_layer = snapshot.layer
                source = snapshot.predecessor_box
                geometry_sha = _json_sha256(
                    _geometry_body_from_layer(snapshot_layer)
                )
                weight_sha = frozen._array_digest(
                    snapshot_layer.params["weight"]
                )
                source_lb_sha = frozen._array_digest(source.lb)
                source_ub_sha = frozen._array_digest(source.ub)
                key_body = _physical_key_body(
                    frame_content_sha256=runtime.frame_content_sha256,
                    layer_id=snapshot_layer.id,
                    predecessor_id=predecessor_id,
                    weight_sha256=weight_sha,
                    geometry_sha256=geometry_sha,
                    source_lb_sha256=source_lb_sha,
                    source_ub_sha256=source_ub_sha,
                    box_semantics=semantics,
                    numeric_platform_sha256=(
                        runtime.numeric_platform_sha256
                    ),
                    implementation_sha256=(
                        runtime.implementation_sha256
                    ),
                )
                key = _json_sha256(key_body)
                core = runtime.cores.get(key)
                core_was_cached = core is not None
                if core is None:
                    try:
                        plan = conv_v51.prepare_dense_conv_v51_plan(
                            snapshot_layer,
                            source,
                            deadline=runtime.deadline,
                        )
                        conv_v51._validate_plan(
                            plan, deadline=runtime.deadline
                        )
                    except frozen.QueryDualReplayTimeout as exc:
                        runtime.closed = True
                        raise ConvMaterialCacheTimeout() from exc
                    except frozen.QueryDualReplayError as exc:
                        _poison(runtime, exc.code, str(exc))
                    except Exception as exc:
                        _poison(
                            runtime,
                            "ADMISSION_VALIDATION_FAILED",
                            f"Conv admission validation failed: {exc}",
                        )
                    _check_deadline(runtime)
                    _require_bytes_backed_plan(plan)
                    binding = _CoreBinding(
                        frame_content_sha256=(
                            runtime.frame_content_sha256
                        ),
                        layer_id=int(snapshot_layer.id),
                        predecessor_id=int(predecessor_id),
                        operator_kind=OPERATOR,
                        branch=BRANCH,
                        weight_sha256=weight_sha,
                        geometry_sha256=geometry_sha,
                        source_lb_sha256=source_lb_sha,
                        source_ub_sha256=source_ub_sha,
                        box_semantics=semantics,
                        numeric_platform_sha256=(
                            runtime.numeric_platform_sha256
                        ),
                        implementation_sha256=(
                            runtime.implementation_sha256
                        ),
                        source_lb=source.lb,
                        source_ub=source.ub,
                    )
                    receipt_body = _core_receipt_body(key, plan)
                    receipt_bytes = bytes(_json_bytes(receipt_body))
                    core = _PhysicalConvCore(
                        key_sha256=key,
                        content_sha256=_json_sha256(receipt_body),
                        plan_content_sha256=str(
                            plan.manifest["content_sha256"]
                        ),
                        binding=binding,
                        plan=plan,
                        receipt_bytes=receipt_bytes,
                    )
                    runtime.cores[key] = core
                    runtime.core_seals[key] = _make_core_seal(core)
                    runtime.physical_builds += 1
                    runtime.admission_full_validations += 1
                    _validate_core_full_content_locked(
                        runtime, key, core
                    )
                else:
                    _validate_core_fast_locked(runtime, key, core)

                if token in runtime.fast_inputs:
                    _poison(
                        runtime,
                        "FAST_INPUT_SEAL_MISMATCH",
                        "duplicate immutable input token",
                    )
                fast_input = _FastInputSeal(
                    token=token,
                    signature=snapshot.caller_signature,
                    physical_key_sha256=core.key_sha256,
                    layer=snapshot.layer,
                    predecessor_box=snapshot.predecessor_box,
                    snapshot=snapshot,
                )
                runtime.fast_inputs[token] = fast_input
                if not _fast_trust_admit_once(
                    runtime.nonce,
                    token,
                    fast_input,
                    snapshot=snapshot,
                ):
                    _poison(
                        runtime,
                        "FAST_INPUT_EXTERNAL_SEAL_MISMATCH",
                        "append-only fast-input registration failed",
                    )
                runtime.fast_input_mints += 1
            key = core.key_sha256
            _validate_core_fast_locked(runtime, key, core)
            _check_deadline(runtime)

            alias_key = (
                stage_seal.stage_use_sha256,
                stage_seal.cone_start_lid,
                key,
            )
            existing = runtime.aliases_by_use.get(alias_key)
            if existing is not None:
                _validate_alias_locked(runtime, existing)
                _check_membership_counts_locked(runtime)
                _check_deadline(runtime)
                return existing
            if core_was_cached:
                runtime.cross_stage_physical_hits += 1
            body = _alias_body(runtime, stage_seal, core)
            body["content_sha256"] = _json_sha256(body)
            alias = ConvMaterialStageAliasCandidate(
                stage_use_sha256=stage_seal.stage_use_sha256,
                cone_start_lid=stage_seal.cone_start_lid,
                frame_content_sha256=runtime.frame_content_sha256,
                physical_key_sha256=key,
                physical_core_content_sha256=core.content_sha256,
                layer_id=core.binding.layer_id,
                predecessor_id=core.binding.predecessor_id,
                receipt=MappingProxyType(body),
                _nonce=secrets.token_hex(32),
                _frame_nonce=runtime.nonce,
                _capability=_ALIAS_CAPABILITY,
            )
            runtime.aliases[alias._nonce] = alias
            runtime.aliases_by_use[alias_key] = alias
            runtime.alias_seals[alias._nonce] = _make_alias_seal(alias)
            runtime.alias_mints += 1
            _validate_alias_locked(runtime, alias)
            _check_membership_counts_locked(runtime)
            _check_deadline(runtime)
            return alias

    def validate_alias(
        self, alias: ConvMaterialStageAliasCandidate
    ) -> bool:
        """Return whether an alias is live and owned by this exact frame."""

        try:
            runtime = _runtime_for(self)
            with runtime.lock:
                _check_static_locked(self, runtime)
                _check_membership_counts_locked(runtime)
                _check_deadline(runtime)
                _validate_alias_locked(runtime, alias)
                _check_membership_counts_locked(runtime)
                _check_deadline(runtime)
                return True
        except ConvMaterialCacheError:
            return False

    def replay_dense_conv_public(
        self,
        alias: ConvMaterialStageAliasCandidate,
        coefficient: Any,
    ) -> conv_v51.DenseConvV51Result:
        """Replay through the existing public helper and its full validator."""

        runtime = _runtime_for(self)
        with runtime.lock:
            _check_static_locked(self, runtime)
            _check_membership_counts_locked(runtime)
            _check_deadline(runtime)
            core = _validate_alias_locked(runtime, alias)
            try:
                result = conv_v51.replay_dense_conv_v51(
                    coefficient, core.plan, deadline=runtime.deadline
                )
            except frozen.QueryDualReplayTimeout as exc:
                runtime.closed = True
                raise ConvMaterialCacheTimeout() from exc
            _validate_core_fast_locked(runtime, core.key_sha256, core)
            _check_deadline(runtime)
            runtime.execution_alias_lookups += 1
            _check_membership_counts_locked(runtime)
            _check_deadline(runtime)
            return result

    def commit(self) -> ConvMaterialCommitCandidate:
        """Full-validate each unique core once and verify alias coverage."""

        runtime = _runtime_for(self)
        with runtime.lock:
            _check_static_locked(self, runtime)
            _check_membership_counts_locked(runtime)
            _check_deadline(runtime)
            if not runtime.cores:
                _poison(
                    runtime, "EMPTY_COMMIT", "no physical core was admitted"
                )
            _check_all_stage_seals_locked(runtime)
            _check_deadline(runtime)
            _check_full_membership_locked(runtime)
            _check_deadline(runtime)
            covered = frozenset(
                (alias.stage_use_sha256, alias.cone_start_lid)
                for alias in runtime.aliases.values()
            )
            _check_deadline(runtime)
            if covered != runtime.expected_stage_set:
                _poison(
                    runtime,
                    "INCOMPLETE_LEDGER",
                    "stage-local alias ledger does not cover the frame",
                )

            validation_count = 0
            for key in sorted(runtime.cores):
                _check_deadline(runtime)
                core = runtime.cores[key]
                _validate_core_fast_locked(runtime, key, core)
                try:
                    conv_v51._validate_plan(
                        core.plan, deadline=runtime.deadline
                    )
                except frozen.QueryDualReplayTimeout as exc:
                    runtime.closed = True
                    raise ConvMaterialCacheTimeout() from exc
                except frozen.QueryDualReplayError as exc:
                    _poison(runtime, exc.code, str(exc))
                except Exception as exc:
                    _poison(
                        runtime,
                        "COMMIT_VALIDATION_FAILED",
                        f"Conv commit validation failed: {exc}",
                    )
                validation_count += 1
                _validate_core_full_content_locked(
                    runtime, key, core
                )
                _check_deadline(runtime)

            implementation_now = _implementation_manifest()
            if (
                dict(implementation_now)
                != dict(runtime.implementation_manifest)
                or _json_sha256(dict(implementation_now))
                != runtime.implementation_sha256
            ):
                _poison(
                    runtime,
                    "IMPLEMENTATION_CHANGED",
                    "implementation source changed during the frame",
                )
            _check_deadline(runtime)
            core_shas = tuple(
                runtime.cores[key].content_sha256
                for key in sorted(runtime.cores)
            )
            sealed_stage_coverage = [
                json.loads(value.record_bytes.decode("ascii"))
                for value in runtime.stage_seals
            ]
            _check_deadline(runtime)
            body: Dict[str, Any] = {
                "schema": COMMIT_SCHEMA,
                "numeric_protocol": NUMERIC_PROTOCOL,
                "proof_authority": False,
                "frame_content_sha256": runtime.frame_content_sha256,
                "numeric_platform_sha256": (
                    runtime.numeric_platform_sha256
                ),
                "implementation_sha256": (
                    runtime.implementation_sha256
                ),
                "physical_builds": runtime.physical_builds,
                "cross_stage_physical_hits": (
                    runtime.cross_stage_physical_hits
                ),
                "execution_alias_lookups": (
                    runtime.execution_alias_lookups
                ),
                "stage_aliases": len(runtime.aliases),
                "commit_full_validations": validation_count,
                "physical_core_sha256s": list(core_shas),
                "stage_coverage": sealed_stage_coverage,
                "status": "candidate_no_authority",
            }
            body["content_sha256"] = _json_sha256(body)
            result = ConvMaterialCommitCandidate(
                physical_builds=runtime.physical_builds,
                cross_stage_physical_hits=(
                    runtime.cross_stage_physical_hits
                ),
                stage_aliases=len(runtime.aliases),
                execution_alias_lookups=(
                    runtime.execution_alias_lookups
                ),
                commit_full_validations=validation_count,
                physical_core_sha256s=core_shas,
                receipt=MappingProxyType(body),
            )
            _check_full_membership_locked(runtime)
            _check_deadline(runtime)
            runtime.closed = True
            return result

    def close(self) -> None:
        runtime = _runtime_for(self)
        with runtime.lock:
            runtime.closed = True


def validate_conv_material_commit_candidate(value: Any) -> bool:
    """Validate an immutable report while continuing to deny authority."""

    try:
        if (
            not isinstance(value, ConvMaterialCommitCandidate)
            or value.proof_authority is not False
            or type(value.receipt) is not MappingProxyType
        ):
            return False
        body = dict(value.receipt)
        claimed = str(body.pop("content_sha256"))
        return bool(
            body.get("schema") == COMMIT_SCHEMA
            and body.get("numeric_protocol") == NUMERIC_PROTOCOL
            and body.get("proof_authority") is False
            and body.get("status") == "candidate_no_authority"
            and int(body.get("physical_builds")) == value.physical_builds
            and int(body.get("cross_stage_physical_hits"))
            == value.cross_stage_physical_hits
            and int(body.get("stage_aliases")) == value.stage_aliases
            and int(body.get("execution_alias_lookups"))
            == value.execution_alias_lookups
            and value.stage_aliases
            == value.physical_builds
            + value.cross_stage_physical_hits
            and int(body.get("commit_full_validations"))
            == value.commit_full_validations
            and tuple(body.get("physical_core_sha256s", ()))
            == value.physical_core_sha256s
            and hmac.compare_digest(_json_sha256(body), claimed)
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


__all__ = [
    "ALIAS_SCHEMA",
    "BOX_OUTPUT",
    "BOX_RELU_POST",
    "BOX_RELU_PRE",
    "BRANCH",
    "COMMIT_SCHEMA",
    "ConvMaterialCacheError",
    "ConvMaterialCacheTimeout",
    "ConvMaterialCommitCandidate",
    "ConvMaterialFrameCandidate",
    "ConvMaterialStageAliasCandidate",
    "ConvMaterialStageUse",
    "NUMERIC_PROTOCOL",
    "OPERATOR",
    "SCHEMA",
    "validate_conv_material_commit_candidate",
]
