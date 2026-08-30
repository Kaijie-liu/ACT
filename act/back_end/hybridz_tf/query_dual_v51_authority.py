#!/usr/bin/env python3
# ===- query_dual_v51_authority.py - V5.1 semantic sidecar ---------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Process-local semantic authority primitives for experimental V5.1 replay.

This module implements only the first three V5.1 integration phases:

* a multi-stage-use immutable frame binding and frame-owned support catalog;
* contiguous query-span and exact per-row guard-policy coverage; and
* a compact absorption ledger which retains hashes and scalar metadata, never
  production-sized nominal or componentwise-radius arrays.

It deliberately does not evaluate a network, inspect BLAS, create a replay
session, or issue lower-bound proof authority.  Every certificate produced
here has ``proof_authority=False``.  A future root-owned V5.1 replay session
must reconstruct expectations from its checked sealed graph and may consume
this sidecar only as one part of its final commit validation.

All objects carrying process-local semantics are protected by identity
registries in addition to immutable content seals.  Copying a dataclass,
rehashing public fields, moving an alias to another frame, or rebuilding a
receipt does not inherit authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import threading
import time
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, NoReturn, Optional, Sequence, Tuple

import numpy as np


NUMERIC_PROTOCOL = "wide_support_structural_activity_v5_1"
FRAME_SCHEMA = "act.query_dual_v51_frame_binding.v2"
CATALOG_ALIAS_SCHEMA = "act.query_dual_v51_support_catalog_alias.v1"
QUERY_SPAN_SCHEMA = "act.query_dual_v51_query_span.v1"
POLICY_PARTITION_SCHEMA = "act.query_dual_v51_row_policy_partition.v1"
EXPECTATION_SCHEMA = "act.query_dual_v51_affine_expectation.v1"
ABSORPTION_TRACE_SCHEMA = "act.query_dual_v51_absorption_trace.v1"
LEDGER_SCHEMA = "act.query_dual_v51_compact_guard_ledger.v1"

STAGE_TARGET = "TARGET"
STAGE_PROPERTY = "PROPERTY"

BRANCH_DENSE = "DENSE"
BRANCH_CONV_DENSE = "CONV2D_DENSE"
BRANCH_CONV_SPARSE = "CONV2D_SPARSE"

POLICY_SCALAR = "scalar_final_guard_once"
POLICY_COMPONENTWISE = "componentwise_radius_once"

_F64 = np.dtype(np.float64)
_AUTHORITY_PID = os.getpid()


class QueryDualV51AuthorityError(RuntimeError):
    """Stable fail-closed error for V5.1 semantic sidecar misuse."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise QueryDualV51AuthorityError(code, message)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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


def _drop_external_seal(
    seals: Dict[str, Any],
    lock: threading.Lock,
    nonce: str,
) -> None:
    """Release registry-external content when its weak object is collected."""

    # Parent finalizers inherited across ``fork`` must never touch a possibly
    # locked parent-thread mutex in the child.  All inherited authorities are
    # invalid there and their parent-side seals are intentionally left alone.
    if os.getpid() != _AUTHORITY_PID:
        return
    with lock:
        seals.pop(nonce, None)


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


def _exact_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        _fail("INVALID_BINDING", f"{name} must be an exact integer")
    result = int(value)
    if result < 0:
        _fail("INVALID_BINDING", f"{name} must be nonnegative")
    return result


def _optional_nonnegative_int(value: Any, *, name: str) -> Optional[int]:
    if value is None:
        return None
    return _exact_nonnegative_int(value, name=name)


def _finite_hex(value: Any, *, name: str, nonnegative: bool) -> str:
    if not isinstance(value, str):
        _fail("INVALID_BINDING", f"{name} must be a binary64 hex string")
    try:
        numeric = float.fromhex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QueryDualV51AuthorityError(
            "INVALID_BINDING", f"{name} is not a binary64 hex string"
        ) from exc
    if not math.isfinite(numeric) or (nonnegative and numeric < 0.0):
        _fail("INVALID_BINDING", f"{name} is outside its finite domain")
    if float(numeric).hex() != value:
        _fail("INVALID_BINDING", f"{name} is not canonical binary64 hex")
    return value


def _deadline_from_hex(value: Any) -> float:
    checked = _finite_hex(
        value, name="deadline_monotonic_hex", nonnegative=True
    )
    return float.fromhex(checked)


def _check_deadline(deadline: float, *, where: str) -> None:
    if time.monotonic() >= float(deadline):
        raise QueryDualV51AuthorityError(
            "DEADLINE_EXPIRED", f"V5.1 authority deadline expired {where}"
        )


def _require_f64_array(
    value: Any,
    *,
    name: str,
    ndim: int,
    nonnegative: bool,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise QueryDualV51AuthorityError(
            "INVALID_NUMERIC", f"{name} cannot be viewed as an array"
        ) from exc
    if (
        array.dtype != _F64
        or not array.dtype.isnative
        or array.ndim != ndim
        or not array.flags.c_contiguous
        or array.size == 0
        or not np.all(np.isfinite(array))
        or (nonnegative and np.any(array < 0.0))
    ):
        _fail(
            "INVALID_NUMERIC",
            f"{name} must be nonempty finite C-order native binary64",
        )
    return array


def _snapshot_f64_array(
    value: Any,
    *,
    name: str,
    ndim: int,
    nonnegative: bool,
) -> np.ndarray:
    """Take one owned, immutable snapshot before validation and hashing."""

    source = _require_f64_array(
        value,
        name=name,
        ndim=ndim,
        nonnegative=nonnegative,
    )
    snapshot = np.frombuffer(
        source.tobytes(order="C"), dtype=np.float64
    ).reshape(source.shape)
    if snapshot.flags.writeable or snapshot.flags.owndata:
        _fail("INVALID_ARRAY", f"{name} snapshot is not bytes-backed")
    _require_f64_array(
        snapshot,
        name=f"{name} snapshot",
        ndim=ndim,
        nonnegative=nonnegative,
    )
    return snapshot


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if (
        array.dtype != _F64
        or not array.dtype.isnative
        or not array.flags.c_contiguous
    ):
        _fail("INVALID_NUMERIC", "only C-order native binary64 may be hashed")
    canonical = array.astype(np.dtype("<f8"), copy=False)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0<f8\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _array_metadata(
    value: np.ndarray,
    *,
    nonnegative: bool,
) -> Dict[str, Any]:
    absolute_max = float(np.max(np.abs(value)))
    metadata: Dict[str, Any] = {
        "shape": [int(item) for item in value.shape],
        "sha256": _array_sha256(value),
        "absolute_max_hex": absolute_max.hex(),
    }
    if nonnegative:
        metadata["maximum_hex"] = float(np.max(value)).hex()
    return metadata


def _mask_length(row_count: int) -> int:
    return (int(row_count) + 7) // 8


def _validate_mask(value: Any, *, row_count: int, name: str) -> bytes:
    if not isinstance(value, bytes):
        _fail("INVALID_MASK", f"{name} must be immutable bytes")
    expected = _mask_length(row_count)
    if len(value) != expected:
        _fail(
            "INVALID_MASK",
            f"{name} has {len(value)} bytes, expected {expected}",
        )
    remainder = row_count & 7
    if remainder and value and (value[-1] >> remainder) != 0:
        _fail("INVALID_MASK", f"{name} has nonzero tail bits")
    return bytes(value)


def _full_mask(row_count: int) -> bytes:
    size = _mask_length(row_count)
    result = bytearray(b"\xff" * size)
    remainder = row_count & 7
    if remainder:
        result[-1] = (1 << remainder) - 1
    return bytes(result)


def _mask_sha256(mask: bytes, *, row_count: int) -> str:
    return _json_sha256(
        {
            "encoding": "little-bit-first-packed-bytes",
            "row_count": int(row_count),
            "bytes_hex": mask.hex(),
        }
    )


def _mask_popcount(mask: bytes) -> int:
    return sum(int(value).bit_count() for value in mask)


def _mask_or(left: bytes, right: bytes) -> bytes:
    return bytes(a | b for a, b in zip(left, right))


def _mask_and(left: bytes, right: bytes) -> bytes:
    return bytes(a & b for a, b in zip(left, right))


def _mask_subset(subset: bytes, superset: bytes) -> bool:
    return all((left & ~right) == 0 for left, right in zip(subset, superset))


def _mask_from_bool(value: Any) -> bytes:
    array = np.asarray(value, dtype=np.bool_).reshape(-1)
    output = bytearray(_mask_length(array.size))
    for index in np.flatnonzero(array):
        output[int(index) // 8] |= 1 << (int(index) & 7)
    return bytes(output)


def _mask_to_bool(mask: bytes, *, row_count: int) -> np.ndarray:
    values = np.zeros(row_count, dtype=np.bool_)
    for index in range(row_count):
        values[index] = bool(mask[index // 8] & (1 << (index & 7)))
    return values


@dataclass(frozen=True)
class StageUse:
    """One ordered consumer of a frame-owned sealed cone."""

    use_index: int
    stage_kind: str
    stage_index: Optional[int]
    target_relu_lid: Optional[int]
    cone_start_lid: Optional[int]
    stage_use_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        use_index = _exact_nonnegative_int(
            self.use_index, name="use_index"
        )
        if self.stage_kind == STAGE_TARGET:
            stage_index = _exact_nonnegative_int(
                self.stage_index, name="stage_index"
            )
            target = _exact_nonnegative_int(
                self.target_relu_lid, name="target_relu_lid"
            )
            cone = _exact_nonnegative_int(
                self.cone_start_lid, name="cone_start_lid"
            )
        elif self.stage_kind == STAGE_PROPERTY:
            if (
                self.stage_index is not None
                or self.target_relu_lid is not None
                or self.cone_start_lid is not None
            ):
                _fail(
                    "INVALID_STAGE_USE",
                    "PROPERTY use requires null stage/target/start",
                )
            stage_index = None
            target = None
            cone = None
        else:
            _fail(
                "INVALID_STAGE_USE",
                "stage_kind must be TARGET or PROPERTY",
            )
        body = {
            "schema": "act.query_dual_v51_stage_use.v1",
            "use_index": use_index,
            "stage_kind": self.stage_kind,
            "stage_index": stage_index,
            "target_relu_lid": target,
            "cone_start_lid": cone,
        }
        object.__setattr__(self, "use_index", use_index)
        object.__setattr__(self, "stage_index", stage_index)
        object.__setattr__(self, "target_relu_lid", target)
        object.__setattr__(self, "cone_start_lid", cone)
        object.__setattr__(
            self, "stage_use_sha256", _json_sha256(body)
        )


def _stage_use_body(value: StageUse) -> Dict[str, Any]:
    return {
        "use_index": value.use_index,
        "stage_kind": value.stage_kind,
        "stage_index": value.stage_index,
        "target_relu_lid": value.target_relu_lid,
        "cone_start_lid": value.cone_start_lid,
        "stage_use_sha256": value.stage_use_sha256,
    }


def validate_stage_use(value: Any) -> bool:
    try:
        if not isinstance(value, StageUse):
            return False
        rebuilt = StageUse(
            use_index=value.use_index,
            stage_kind=value.stage_kind,
            stage_index=value.stage_index,
            target_relu_lid=value.target_relu_lid,
            cone_start_lid=value.cone_start_lid,
        )
        return hmac.compare_digest(
            rebuilt.stage_use_sha256, value.stage_use_sha256
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        QueryDualV51AuthorityError,
    ):
        return False


@dataclass(frozen=True)
class V51FrameBinding:
    """Semantic identity of one bounds frame covering one or more cones."""

    session_nonce_sha256: str
    frame_nonce_sha256: str
    frame_content_sha256: str
    bounds_manifest_sha256: str
    root_receipt_sha256: str
    parent_chain_sha256: str
    deadline_monotonic_hex: str
    stage_uses: Tuple[StageUse, ...]
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "session_nonce_sha256",
            "frame_nonce_sha256",
            "frame_content_sha256",
            "bounds_manifest_sha256",
            "root_receipt_sha256",
            "parent_chain_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _deadline_from_hex(self.deadline_monotonic_hex)
        if not isinstance(self.stage_uses, tuple) or not self.stage_uses:
            _fail(
                "INVALID_FRAME",
                "a V5.1 frame requires at least one stage use",
            )
        uses = tuple(self.stage_uses)
        if any(not validate_stage_use(value) for value in uses):
            _fail("INVALID_FRAME", "frame contains an invalid stage use")
        if [value.use_index for value in uses] != list(range(len(uses))):
            _fail(
                "INVALID_FRAME",
                "stage-use indices must be contiguous and zero-based",
            )
        if len({value.stage_use_sha256 for value in uses}) != len(uses):
            _fail("INVALID_FRAME", "frame contains duplicate stage uses")
        target_indices = [
            value.stage_index
            for value in uses
            if value.stage_kind == STAGE_TARGET
        ]
        if len(set(target_indices)) != len(target_indices):
            _fail("INVALID_FRAME", "frame repeats a target stage index")
        if sum(value.stage_kind == STAGE_PROPERTY for value in uses) > 1:
            _fail("INVALID_FRAME", "frame has more than one property use")
        unique_starts: list[Any] = []
        for value in uses:
            record = (
                "ASSERT_PREDECESSOR"
                if value.cone_start_lid is None
                else value.cone_start_lid
            )
            if record not in unique_starts:
                unique_starts.append(record)
        body = {
            "schema": FRAME_SCHEMA,
            "numeric_protocol": NUMERIC_PROTOCOL,
            "session_nonce_sha256": self.session_nonce_sha256,
            "frame_nonce_sha256": self.frame_nonce_sha256,
            "frame_content_sha256": self.frame_content_sha256,
            "bounds_manifest_sha256": self.bounds_manifest_sha256,
            "root_receipt_sha256": self.root_receipt_sha256,
            "parent_chain_sha256": self.parent_chain_sha256,
            "deadline_monotonic_hex": self.deadline_monotonic_hex,
            "stage_uses": [_stage_use_body(value) for value in uses],
            "stage_use_count": len(uses),
            "unique_cone_starts": unique_starts,
            "unique_cone_count": len(unique_starts),
        }
        object.__setattr__(self, "stage_uses", uses)
        object.__setattr__(self, "binding_sha256", _json_sha256(body))


def validate_frame_binding(value: Any) -> bool:
    try:
        if not isinstance(value, V51FrameBinding):
            return False
        rebuilt = V51FrameBinding(
            session_nonce_sha256=value.session_nonce_sha256,
            frame_nonce_sha256=value.frame_nonce_sha256,
            frame_content_sha256=value.frame_content_sha256,
            bounds_manifest_sha256=value.bounds_manifest_sha256,
            root_receipt_sha256=value.root_receipt_sha256,
            parent_chain_sha256=value.parent_chain_sha256,
            deadline_monotonic_hex=value.deadline_monotonic_hex,
            stage_uses=value.stage_uses,
        )
        return hmac.compare_digest(
            rebuilt.binding_sha256, value.binding_sha256
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        QueryDualV51AuthorityError,
    ):
        return False


_OWNER_CAPABILITY = object()
_OWNER_REGISTRY: weakref.WeakValueDictionary[
    str, "_FrameOwner"
] = weakref.WeakValueDictionary()
_OWNER_SEALS: Dict[
    str,
    Tuple[int, str, int, Tuple[Tuple[int, str], ...]],
] = {}
_OWNER_LOCK = threading.Lock()


@dataclass(frozen=True)
class _FrameOwner:
    binding: V51FrameBinding
    _nonce: str = field(repr=False)
    _capability: Any = field(repr=False, compare=False)

    @property
    def nonce_sha256(self) -> str:
        return hashlib.sha256(self._nonce.encode("ascii")).hexdigest()


def _mint_frame_owner(binding: V51FrameBinding) -> _FrameOwner:
    if os.getpid() != _AUTHORITY_PID:
        _fail(
            "PROCESS_MISMATCH",
            "V5.1 authority cannot be inherited across fork",
        )
    if not validate_frame_binding(binding):
        _fail("INVALID_FRAME", "cannot own an invalid V5.1 frame")
    deadline = _deadline_from_hex(binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before frame owner mint")
    nonce = secrets.token_hex(32)
    owner = _FrameOwner(
        binding=binding,
        _nonce=nonce,
        _capability=_OWNER_CAPABILITY,
    )
    with _OWNER_LOCK:
        _OWNER_REGISTRY[nonce] = owner
        _OWNER_SEALS[nonce] = (
            id(binding),
            binding.binding_sha256,
            id(binding.stage_uses),
            tuple(
                (id(value), value.stage_use_sha256)
                for value in binding.stage_uses
            ),
        )
    weakref.finalize(
        owner, _drop_external_seal, _OWNER_SEALS, _OWNER_LOCK, nonce
    )
    _check_deadline(deadline, where="after frame owner mint")
    return owner


def _verify_owner(value: Any) -> bool:
    try:
        if (
            os.getpid() != _AUTHORITY_PID
            or
            not isinstance(value, _FrameOwner)
            or value._capability is not _OWNER_CAPABILITY
            or not validate_frame_binding(value.binding)
        ):
            return False
        with _OWNER_LOCK:
            seal = _OWNER_SEALS.get(value._nonce)
            return bool(
                _OWNER_REGISTRY.get(value._nonce) is value
                and seal is not None
                and seal[0] == id(value.binding)
                and hmac.compare_digest(
                    seal[1], value.binding.binding_sha256
                )
                and seal[2] == id(value.binding.stage_uses)
                and seal[3]
                == tuple(
                    (id(item), item.stage_use_sha256)
                    for item in value.binding.stage_uses
                )
            )
    except (AttributeError, TypeError, ValueError):
        return False


def _owner_stage_use(
    owner: _FrameOwner, stage_use_sha256: str
) -> StageUse:
    if not _is_sha256(stage_use_sha256):
        _fail(
            "INVALID_STAGE_USE",
            "stage-use identifier must be a SHA-256 string",
        )
    for value in owner.binding.stage_uses:
        if hmac.compare_digest(value.stage_use_sha256, stage_use_sha256):
            return value
    _fail("INVALID_STAGE_USE", "stage use is absent from the frame")


_ALIAS_CAPABILITY = object()
_ALIAS_REGISTRY: weakref.WeakValueDictionary[
    str, "SupportCatalogAlias"
] = weakref.WeakValueDictionary()
_ALIAS_SEALS: Dict[str, Tuple[str, int]] = {}
_ALIAS_BY_KEY: weakref.WeakValueDictionary[
    str, "SupportCatalogAlias"
] = weakref.WeakValueDictionary()
_ALIAS_LOCK = threading.Lock()


@dataclass(frozen=True)
class SupportCatalogAlias:
    """Compact frame/cone/branch binding for one support catalog entry."""

    frame_binding_sha256: str
    owner_nonce_sha256: str
    stage_use_sha256: str
    cone_start_lid: Optional[int]
    layer_id: int
    predecessor_id: int
    operator_kind: str
    branch: str
    box_semantics: str
    catalog_content_sha256: str
    support_content_sha256: str
    weight_sha256: str
    geometry_sha256: str
    source_lb_sha256: str
    source_ub_sha256: str
    numeric_platform_sha256: str
    implementation_sha256: str
    branch_evidence_sha256: str
    catalog_key_sha256: str
    receipt: Mapping[str, Any]
    content_sha256: str
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)


def _alias_body(
    *,
    owner: _FrameOwner,
    stage_use: StageUse,
    layer_id: int,
    predecessor_id: int,
    operator_kind: str,
    branch: str,
    box_semantics: str,
    catalog_content_sha256: str,
    support_content_sha256: str,
    weight_sha256: str,
    geometry_sha256: str,
    source_lb_sha256: str,
    source_ub_sha256: str,
    numeric_platform_sha256: str,
    implementation_sha256: str,
    branch_evidence_sha256: str,
    alias_nonce_sha256: Optional[str],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "schema": CATALOG_ALIAS_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "proof_authority": False,
        "authority_scope": "frame_owned_support_catalog_alias_only",
        "frame_binding_sha256": owner.binding.binding_sha256,
        "frame_content_sha256": owner.binding.frame_content_sha256,
        "owner_nonce_sha256": owner.nonce_sha256,
        "stage_use_sha256": stage_use.stage_use_sha256,
        "cone_start_lid": stage_use.cone_start_lid,
        "layer_id": layer_id,
        "predecessor_id": predecessor_id,
        "operator_kind": operator_kind,
        "branch": branch,
        "box_semantics": box_semantics,
        "catalog_content_sha256": catalog_content_sha256,
        "support_content_sha256": support_content_sha256,
        "weight_sha256": weight_sha256,
        "geometry_sha256": geometry_sha256,
        "source_lb_sha256": source_lb_sha256,
        "source_ub_sha256": source_ub_sha256,
        "numeric_platform_sha256": numeric_platform_sha256,
        "implementation_sha256": implementation_sha256,
        "branch_evidence_sha256": branch_evidence_sha256,
    }
    if alias_nonce_sha256 is not None:
        body["alias_nonce_sha256"] = alias_nonce_sha256
    return body


def _mint_support_catalog_alias(
    owner: _FrameOwner,
    *,
    stage_use_sha256: str,
    layer_id: int,
    predecessor_id: int,
    operator_kind: str,
    branch: str,
    box_semantics: str,
    catalog_content_sha256: str,
    support_content_sha256: str,
    weight_sha256: str,
    geometry_sha256: str,
    source_lb_sha256: str,
    source_ub_sha256: str,
    numeric_platform_sha256: str,
    implementation_sha256: str,
    branch_evidence_sha256: str,
) -> SupportCatalogAlias:
    """Mint or retrieve one exact frame-local catalog alias."""

    if not _verify_owner(owner):
        _fail("INVALID_OWNER", "catalog alias requires a live frame owner")
    deadline = _deadline_from_hex(owner.binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before catalog alias lookup")
    stage_use = _owner_stage_use(owner, stage_use_sha256)
    layer = _exact_nonnegative_int(layer_id, name="layer_id")
    predecessor = _exact_nonnegative_int(
        predecessor_id, name="predecessor_id"
    )
    if type(operator_kind) is not str or type(branch) is not str:
        _fail("INVALID_BRANCH", "operator and branch must be exact strings")
    if type(box_semantics) is not str:
        _fail("INVALID_BINDING", "box semantics must be an exact string")
    if operator_kind == "DENSE":
        if branch != BRANCH_DENSE:
            _fail("INVALID_BRANCH", "DENSE requires the DENSE branch")
    elif operator_kind == "CONV2D":
        if branch not in {BRANCH_CONV_DENSE, BRANCH_CONV_SPARSE}:
            _fail("INVALID_BRANCH", "CONV2D branch is invalid")
    else:
        _fail("INVALID_BRANCH", "operator must be DENSE or CONV2D")
    if box_semantics not in {
        "output",
        "preactivation",
        "relu_postactivation_from_preactivation_box_v1",
    }:
        _fail("INVALID_BINDING", "unsupported predecessor box semantics")
    hashes = {
        name: _require_sha256(value, name=name)
        for name, value in (
            ("catalog_content_sha256", catalog_content_sha256),
            ("support_content_sha256", support_content_sha256),
            ("weight_sha256", weight_sha256),
            ("geometry_sha256", geometry_sha256),
            ("source_lb_sha256", source_lb_sha256),
            ("source_ub_sha256", source_ub_sha256),
            ("numeric_platform_sha256", numeric_platform_sha256),
            ("implementation_sha256", implementation_sha256),
            ("branch_evidence_sha256", branch_evidence_sha256),
        )
    }
    semantic_body = _alias_body(
        owner=owner,
        stage_use=stage_use,
        layer_id=layer,
        predecessor_id=predecessor,
        operator_kind=operator_kind,
        branch=branch,
        box_semantics=box_semantics,
        alias_nonce_sha256=None,
        **hashes,
    )
    catalog_key_sha = _json_sha256(semantic_body)
    registry_key = f"{owner._nonce}:{catalog_key_sha}"
    with _ALIAS_LOCK:
        cached = _ALIAS_BY_KEY.get(registry_key)
    if cached is not None:
        _check_deadline(deadline, where="after catalog alias cache hit")
        if not validate_support_catalog_alias(cached):
            _fail("INVALID_CATALOG", "cached catalog alias was modified")
        _check_deadline(
            deadline, where="after cached catalog alias validation"
        )
        return cached

    nonce = secrets.token_hex(32)
    nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
    body = _alias_body(
        owner=owner,
        stage_use=stage_use,
        layer_id=layer,
        predecessor_id=predecessor,
        operator_kind=operator_kind,
        branch=branch,
        box_semantics=box_semantics,
        alias_nonce_sha256=nonce_sha,
        **hashes,
    )
    content_sha = _json_sha256(body)
    receipt_body = dict(body)
    receipt_body["catalog_key_sha256"] = catalog_key_sha
    receipt_body["content_sha256"] = content_sha
    receipt_body["receipt_sha256"] = _json_sha256(receipt_body)
    alias = SupportCatalogAlias(
        frame_binding_sha256=owner.binding.binding_sha256,
        owner_nonce_sha256=owner.nonce_sha256,
        stage_use_sha256=stage_use.stage_use_sha256,
        cone_start_lid=stage_use.cone_start_lid,
        layer_id=layer,
        predecessor_id=predecessor,
        operator_kind=operator_kind,
        branch=branch,
        box_semantics=box_semantics,
        catalog_key_sha256=catalog_key_sha,
        receipt=_deep_freeze(receipt_body),
        content_sha256=content_sha,
        _nonce=nonce,
        _owner_nonce=owner._nonce,
        _owner=owner,
        _capability=_ALIAS_CAPABILITY,
        **hashes,
    )
    registered = False
    with _ALIAS_LOCK:
        existing = _ALIAS_BY_KEY.get(registry_key)
        if existing is not None:
            alias = existing
        else:
            _ALIAS_REGISTRY[nonce] = alias
            _ALIAS_SEALS[nonce] = (content_sha, id(alias.receipt))
            _ALIAS_BY_KEY[registry_key] = alias
            registered = True
    if registered:
        weakref.finalize(
            alias,
            _drop_external_seal,
            _ALIAS_SEALS,
            _ALIAS_LOCK,
            nonce,
        )
    else:
        _check_deadline(
            deadline, where="after raced catalog alias cache hit"
        )
        if not validate_support_catalog_alias(alias):
            _fail(
                "INVALID_CATALOG",
                "raced cached catalog alias was modified",
            )
    _check_deadline(deadline, where="after catalog alias mint")
    return alias


def validate_support_catalog_alias(value: Any) -> bool:
    try:
        if (
            not isinstance(value, SupportCatalogAlias)
            or value._capability is not _ALIAS_CAPABILITY
            or not _verify_owner(value._owner)
            or value._owner._nonce != value._owner_nonce
            or value.frame_binding_sha256
            != value._owner.binding.binding_sha256
            or value.owner_nonce_sha256 != value._owner.nonce_sha256
        ):
            return False
        with _ALIAS_LOCK:
            seal = _ALIAS_SEALS.get(value._nonce)
            if (
                _ALIAS_REGISTRY.get(value._nonce) is not value
                or seal is None
                or type(value.receipt) is not MappingProxyType
                or seal[1] != id(value.receipt)
                or not hmac.compare_digest(seal[0], value.content_sha256)
            ):
                return False
        stage_use = _owner_stage_use(
            value._owner, value.stage_use_sha256
        )
        if stage_use.cone_start_lid != value.cone_start_lid:
            return False
        hashes = {
            name: _require_sha256(getattr(value, name), name=name)
            for name in (
                "catalog_content_sha256",
                "support_content_sha256",
                "weight_sha256",
                "geometry_sha256",
                "source_lb_sha256",
                "source_ub_sha256",
                "numeric_platform_sha256",
                "implementation_sha256",
                "branch_evidence_sha256",
            )
        }
        semantic_body = _alias_body(
            owner=value._owner,
            stage_use=stage_use,
            layer_id=value.layer_id,
            predecessor_id=value.predecessor_id,
            operator_kind=value.operator_kind,
            branch=value.branch,
            box_semantics=value.box_semantics,
            alias_nonce_sha256=None,
            **hashes,
        )
        if _json_sha256(semantic_body) != value.catalog_key_sha256:
            return False
        body = _alias_body(
            owner=value._owner,
            stage_use=stage_use,
            layer_id=value.layer_id,
            predecessor_id=value.predecessor_id,
            operator_kind=value.operator_kind,
            branch=value.branch,
            box_semantics=value.box_semantics,
            alias_nonce_sha256=hashlib.sha256(
                value._nonce.encode("ascii")
            ).hexdigest(),
            **hashes,
        )
        content_sha = _json_sha256(body)
        receipt = dict(value.receipt)
        claimed = str(receipt.pop("receipt_sha256"))
        expected = dict(body)
        expected["catalog_key_sha256"] = value.catalog_key_sha256
        expected["content_sha256"] = content_sha
        return bool(
            hmac.compare_digest(content_sha, value.content_sha256)
            and set(receipt) == set(expected)
            and all(
                _canonical_value(receipt[key])
                == _canonical_value(expected[key])
                for key in expected
            )
            and hmac.compare_digest(_json_sha256(receipt), claimed)
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        QueryDualV51AuthorityError,
    ):
        return False


_SPAN_CAPABILITY = object()
_SPAN_REGISTRY: weakref.WeakValueDictionary[
    str, "QuerySpan"
] = weakref.WeakValueDictionary()
_SPAN_SEALS: Dict[str, str] = {}
_SPAN_LOCK = threading.Lock()


@dataclass(frozen=True)
class QuerySpan:
    """One frame-owned contiguous slice of a precommitted query block."""

    frame_binding_sha256: str
    owner_nonce_sha256: str
    stage_use_sha256: str
    span_index: int
    query_start: int
    query_end: int
    query_total: int
    query_block_sha256: str
    query_rows_sha256: str
    query_bias_sha256: str
    alpha_slice_sha256: str
    content_sha256: str
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)

    @property
    def row_count(self) -> int:
        return self.query_end - self.query_start


def _span_body(
    *,
    owner: _FrameOwner,
    stage_use: StageUse,
    span_index: int,
    query_start: int,
    query_end: int,
    query_total: int,
    query_block_sha256: str,
    query_rows_sha256: str,
    query_bias_sha256: str,
    alpha_slice_sha256: str,
    span_nonce_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema": QUERY_SPAN_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "proof_authority": False,
        "frame_binding_sha256": owner.binding.binding_sha256,
        "owner_nonce_sha256": owner.nonce_sha256,
        "stage_use_sha256": stage_use.stage_use_sha256,
        "span_nonce_sha256": span_nonce_sha256,
        "span_index": span_index,
        "query_start": query_start,
        "query_end": query_end,
        "query_count": query_end - query_start,
        "query_total": query_total,
        "query_block_sha256": query_block_sha256,
        "query_rows_sha256": query_rows_sha256,
        "query_bias_sha256": query_bias_sha256,
        "alpha_slice_sha256": alpha_slice_sha256,
    }


def _mint_query_span(
    owner: _FrameOwner,
    *,
    stage_use_sha256: str,
    span_index: int,
    query_start: int,
    query_end: int,
    query_total: int,
    query_block_sha256: str,
    query_rows_sha256: str,
    query_bias_sha256: str,
    alpha_slice_sha256: str,
) -> QuerySpan:
    if not _verify_owner(owner):
        _fail("INVALID_OWNER", "query span requires a live frame owner")
    deadline = _deadline_from_hex(owner.binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before query span mint")
    stage_use = _owner_stage_use(owner, stage_use_sha256)
    index = _exact_nonnegative_int(span_index, name="span_index")
    start = _exact_nonnegative_int(query_start, name="query_start")
    end = _exact_nonnegative_int(query_end, name="query_end")
    total = _exact_nonnegative_int(query_total, name="query_total")
    if total <= 0 or end <= start or end > total:
        _fail(
            "INVALID_QUERY_SPAN",
            "query span must be nonempty and lie inside its block",
        )
    hashes = {
        name: _require_sha256(value, name=name)
        for name, value in (
            ("query_block_sha256", query_block_sha256),
            ("query_rows_sha256", query_rows_sha256),
            ("query_bias_sha256", query_bias_sha256),
            ("alpha_slice_sha256", alpha_slice_sha256),
        )
    }
    nonce = secrets.token_hex(32)
    body = _span_body(
        owner=owner,
        stage_use=stage_use,
        span_index=index,
        query_start=start,
        query_end=end,
        query_total=total,
        span_nonce_sha256=hashlib.sha256(
            nonce.encode("ascii")
        ).hexdigest(),
        **hashes,
    )
    content_sha = _json_sha256(body)
    span = QuerySpan(
        frame_binding_sha256=owner.binding.binding_sha256,
        owner_nonce_sha256=owner.nonce_sha256,
        stage_use_sha256=stage_use.stage_use_sha256,
        span_index=index,
        query_start=start,
        query_end=end,
        query_total=total,
        content_sha256=content_sha,
        _nonce=nonce,
        _owner_nonce=owner._nonce,
        _owner=owner,
        _capability=_SPAN_CAPABILITY,
        **hashes,
    )
    with _SPAN_LOCK:
        _SPAN_REGISTRY[nonce] = span
        _SPAN_SEALS[nonce] = content_sha
    weakref.finalize(
        span, _drop_external_seal, _SPAN_SEALS, _SPAN_LOCK, nonce
    )
    _check_deadline(deadline, where="after query span mint")
    return span


def validate_query_span(value: Any) -> bool:
    try:
        if (
            not isinstance(value, QuerySpan)
            or value._capability is not _SPAN_CAPABILITY
            or not _verify_owner(value._owner)
            or value._owner._nonce != value._owner_nonce
            or value.frame_binding_sha256
            != value._owner.binding.binding_sha256
            or value.owner_nonce_sha256 != value._owner.nonce_sha256
        ):
            return False
        with _SPAN_LOCK:
            if (
                _SPAN_REGISTRY.get(value._nonce) is not value
                or not hmac.compare_digest(
                    _SPAN_SEALS.get(value._nonce, ""),
                    value.content_sha256,
                )
            ):
                return False
        stage_use = _owner_stage_use(
            value._owner, value.stage_use_sha256
        )
        if (
            value.query_total <= 0
            or value.query_start < 0
            or value.query_end <= value.query_start
            or value.query_end > value.query_total
        ):
            return False
        body = _span_body(
            owner=value._owner,
            stage_use=stage_use,
            span_index=value.span_index,
            query_start=value.query_start,
            query_end=value.query_end,
            query_total=value.query_total,
            query_block_sha256=_require_sha256(
                value.query_block_sha256, name="query_block_sha256"
            ),
            query_rows_sha256=_require_sha256(
                value.query_rows_sha256, name="query_rows_sha256"
            ),
            query_bias_sha256=_require_sha256(
                value.query_bias_sha256, name="query_bias_sha256"
            ),
            alpha_slice_sha256=_require_sha256(
                value.alpha_slice_sha256, name="alpha_slice_sha256"
            ),
            span_nonce_sha256=hashlib.sha256(
                value._nonce.encode("ascii")
            ).hexdigest(),
        )
        return hmac.compare_digest(
            _json_sha256(body), value.content_sha256
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        QueryDualV51AuthorityError,
    ):
        return False


@dataclass(frozen=True)
class RowPolicyPartition:
    """Exact scalar/componentwise row partition plus activity diagnostics."""

    row_count: int
    scalar_mask: bytes
    componentwise_mask: bytes
    active_mask: bytes
    fallback_mask: bytes
    partition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        rows = _exact_nonnegative_int(self.row_count, name="row_count")
        if rows <= 0:
            _fail("INVALID_MASK", "row policy cannot be empty")
        scalar = _validate_mask(
            self.scalar_mask, row_count=rows, name="scalar_mask"
        )
        componentwise = _validate_mask(
            self.componentwise_mask,
            row_count=rows,
            name="componentwise_mask",
        )
        active = _validate_mask(
            self.active_mask, row_count=rows, name="active_mask"
        )
        fallback = _validate_mask(
            self.fallback_mask, row_count=rows, name="fallback_mask"
        )
        if _mask_and(scalar, componentwise) != bytes(len(scalar)):
            _fail("POLICY_OVERLAP", "row guard policies overlap")
        if _mask_or(scalar, componentwise) != _full_mask(rows):
            _fail("POLICY_GAP", "row guard policies do not cover all rows")
        if not _mask_subset(active, scalar):
            _fail("INVALID_MASK", "active mask must be scalar-policy rows")
        if not _mask_subset(fallback, scalar):
            _fail("INVALID_MASK", "fallback mask must be scalar-policy rows")
        if not _mask_subset(fallback, active):
            _fail("INVALID_MASK", "fallback mask must be active rows")
        body = {
            "schema": POLICY_PARTITION_SCHEMA,
            "row_count": rows,
            "encoding": "little-bit-first-packed-bytes",
            "scalar_mask_hex": scalar.hex(),
            "componentwise_mask_hex": componentwise.hex(),
            "active_mask_hex": active.hex(),
            "fallback_mask_hex": fallback.hex(),
            "scalar_mask_sha256": _mask_sha256(
                scalar, row_count=rows
            ),
            "componentwise_mask_sha256": _mask_sha256(
                componentwise, row_count=rows
            ),
            "active_mask_sha256": _mask_sha256(
                active, row_count=rows
            ),
            "fallback_mask_sha256": _mask_sha256(
                fallback, row_count=rows
            ),
            "scalar_row_count": _mask_popcount(scalar),
            "componentwise_row_count": _mask_popcount(componentwise),
            "active_row_count": _mask_popcount(active),
            "fallback_row_count": _mask_popcount(fallback),
        }
        object.__setattr__(self, "row_count", rows)
        object.__setattr__(self, "scalar_mask", scalar)
        object.__setattr__(self, "componentwise_mask", componentwise)
        object.__setattr__(self, "active_mask", active)
        object.__setattr__(self, "fallback_mask", fallback)
        object.__setattr__(
            self, "partition_sha256", _json_sha256(body)
        )

    @property
    def scalar_row_count(self) -> int:
        return _mask_popcount(self.scalar_mask)

    @property
    def componentwise_row_count(self) -> int:
        return _mask_popcount(self.componentwise_mask)

    @property
    def active_row_count(self) -> int:
        return _mask_popcount(self.active_mask)

    @property
    def fallback_row_count(self) -> int:
        return _mask_popcount(self.fallback_mask)


def _partition_body(value: RowPolicyPartition) -> Dict[str, Any]:
    return {
        "partition_sha256": value.partition_sha256,
        "row_count": value.row_count,
        "scalar_mask_sha256": _mask_sha256(
            value.scalar_mask, row_count=value.row_count
        ),
        "componentwise_mask_sha256": _mask_sha256(
            value.componentwise_mask, row_count=value.row_count
        ),
        "active_mask_sha256": _mask_sha256(
            value.active_mask, row_count=value.row_count
        ),
        "fallback_mask_sha256": _mask_sha256(
            value.fallback_mask, row_count=value.row_count
        ),
        "scalar_row_count": value.scalar_row_count,
        "componentwise_row_count": value.componentwise_row_count,
        "active_row_count": value.active_row_count,
        "fallback_row_count": value.fallback_row_count,
    }


def validate_row_policy_partition(value: Any) -> bool:
    try:
        if not isinstance(value, RowPolicyPartition):
            return False
        rebuilt = RowPolicyPartition(
            row_count=value.row_count,
            scalar_mask=value.scalar_mask,
            componentwise_mask=value.componentwise_mask,
            active_mask=value.active_mask,
            fallback_mask=value.fallback_mask,
        )
        return hmac.compare_digest(
            rebuilt.partition_sha256, value.partition_sha256
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        QueryDualV51AuthorityError,
    ):
        return False


_EXPECTATION_CAPABILITY = object()
_EXPECTATION_REGISTRY: weakref.WeakValueDictionary[
    str, "AffineExecutionExpectation"
] = weakref.WeakValueDictionary()
_EXPECTATION_SEALS: Dict[str, Tuple[str, int, int, int]] = {}
_EXPECTATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class AffineExecutionExpectation:
    """One exact affine execution, span, branch, catalog and row policy."""

    execution_index: int
    span: QuerySpan
    support_alias: SupportCatalogAlias
    partition: RowPolicyPartition
    input_coefficient_sha256: str
    expectation_sha256: str
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)


def _expectation_body(
    *,
    owner: _FrameOwner,
    execution_index: int,
    span: QuerySpan,
    support_alias: SupportCatalogAlias,
    partition: RowPolicyPartition,
    input_coefficient_sha256: str,
    expectation_nonce_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema": EXPECTATION_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "proof_authority": False,
        "frame_binding_sha256": owner.binding.binding_sha256,
        "owner_nonce_sha256": owner.nonce_sha256,
        "expectation_nonce_sha256": expectation_nonce_sha256,
        "execution_index": execution_index,
        "query_span_sha256": span.content_sha256,
        "query_start": span.query_start,
        "query_end": span.query_end,
        "query_block_sha256": span.query_block_sha256,
        "stage_use_sha256": span.stage_use_sha256,
        "layer_id": support_alias.layer_id,
        "predecessor_id": support_alias.predecessor_id,
        "operator_kind": support_alias.operator_kind,
        "branch": support_alias.branch,
        "branch_evidence_sha256": (
            support_alias.branch_evidence_sha256
        ),
        "support_catalog_key_sha256": (
            support_alias.catalog_key_sha256
        ),
        "support_catalog_content_sha256": (
            support_alias.catalog_content_sha256
        ),
        "support_alias_content_sha256": support_alias.content_sha256,
        "input_coefficient_sha256": input_coefficient_sha256,
        **_partition_body(partition),
    }


def _mint_affine_execution_expectation(
    owner: _FrameOwner,
    *,
    execution_index: int,
    span: QuerySpan,
    support_alias: SupportCatalogAlias,
    partition: RowPolicyPartition,
    input_coefficient_sha256: str,
) -> AffineExecutionExpectation:
    if not _verify_owner(owner):
        _fail("INVALID_OWNER", "expectation requires a live owner")
    deadline = _deadline_from_hex(owner.binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before expectation mint")
    if (
        not validate_query_span(span)
        or not validate_support_catalog_alias(support_alias)
        or span._owner_nonce != owner._nonce
        or support_alias._owner_nonce != owner._nonce
        or span.stage_use_sha256 != support_alias.stage_use_sha256
        or not validate_row_policy_partition(partition)
        or partition.row_count != span.row_count
    ):
        _fail(
            "INVALID_EXPECTATION",
            "span/catalog/partition do not share one frame execution",
        )
    if support_alias.branch in {BRANCH_DENSE, BRANCH_CONV_DENSE}:
        if (
            partition.scalar_row_count != partition.row_count
            or partition.componentwise_row_count != 0
        ):
            _fail(
                "POLICY_MISMATCH",
                "dense affine branch requires scalar policy for every row",
            )
    elif support_alias.branch == BRANCH_CONV_SPARSE:
        if (
            partition.componentwise_row_count != partition.row_count
            or partition.scalar_row_count != 0
        ):
            _fail(
                "POLICY_MISMATCH",
                "sparse Conv branch requires componentwise policy",
            )
    else:
        _fail("INVALID_BRANCH", "support alias has an unknown branch")
    index = _exact_nonnegative_int(
        execution_index, name="execution_index"
    )
    coefficient_sha = _require_sha256(
        input_coefficient_sha256,
        name="input_coefficient_sha256",
    )
    nonce = secrets.token_hex(32)
    body = _expectation_body(
        owner=owner,
        execution_index=index,
        span=span,
        support_alias=support_alias,
        partition=partition,
        input_coefficient_sha256=coefficient_sha,
        expectation_nonce_sha256=hashlib.sha256(
            nonce.encode("ascii")
        ).hexdigest(),
    )
    expectation = AffineExecutionExpectation(
        execution_index=index,
        span=span,
        support_alias=support_alias,
        partition=partition,
        input_coefficient_sha256=coefficient_sha,
        expectation_sha256=_json_sha256(body),
        _nonce=nonce,
        _owner_nonce=owner._nonce,
        _owner=owner,
        _capability=_EXPECTATION_CAPABILITY,
    )
    with _EXPECTATION_LOCK:
        _EXPECTATION_REGISTRY[nonce] = expectation
        _EXPECTATION_SEALS[nonce] = (
            expectation.expectation_sha256,
            id(expectation.span),
            id(expectation.support_alias),
            id(expectation.partition),
        )
    weakref.finalize(
        expectation,
        _drop_external_seal,
        _EXPECTATION_SEALS,
        _EXPECTATION_LOCK,
        nonce,
    )
    _check_deadline(deadline, where="after expectation mint")
    return expectation


def validate_affine_execution_expectation(value: Any) -> bool:
    try:
        if (
            not isinstance(value, AffineExecutionExpectation)
            or value._capability is not _EXPECTATION_CAPABILITY
            or not _verify_owner(value._owner)
            or value._owner._nonce != value._owner_nonce
            or not validate_query_span(value.span)
            or not validate_support_catalog_alias(value.support_alias)
            or not validate_row_policy_partition(value.partition)
            or value.span._owner_nonce != value._owner_nonce
            or value.support_alias._owner_nonce != value._owner_nonce
            or value.span.stage_use_sha256
            != value.support_alias.stage_use_sha256
            or value.partition.row_count != value.span.row_count
        ):
            return False
        with _EXPECTATION_LOCK:
            seal = _EXPECTATION_SEALS.get(value._nonce)
            if (
                _EXPECTATION_REGISTRY.get(value._nonce) is not value
                or seal is None
                or seal[1] != id(value.span)
                or seal[2] != id(value.support_alias)
                or seal[3] != id(value.partition)
                or not hmac.compare_digest(
                    seal[0], value.expectation_sha256
                )
            ):
                return False
        body = _expectation_body(
            owner=value._owner,
            execution_index=value.execution_index,
            span=value.span,
            support_alias=value.support_alias,
            partition=value.partition,
            input_coefficient_sha256=_require_sha256(
                value.input_coefficient_sha256,
                name="input_coefficient_sha256",
            ),
            expectation_nonce_sha256=hashlib.sha256(
                value._nonce.encode("ascii")
            ).hexdigest(),
        )
        return hmac.compare_digest(
            _json_sha256(body), value.expectation_sha256
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        QueryDualV51AuthorityError,
    ):
        return False


_TRACE_CAPABILITY = object()
_TRACE_REGISTRY: weakref.WeakValueDictionary[
    str, "CompactAbsorptionTrace"
] = weakref.WeakValueDictionary()
_TRACE_SEALS: Dict[str, Tuple[str, int]] = {}
_TRACE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CompactAbsorptionTrace:
    """Hashes and scalar metadata only; no nominal/radius array is retained."""

    expectation: AffineExecutionExpectation
    nominal_shape: Tuple[int, ...]
    nominal_sha256: str
    nominal_absolute_max_hex: str
    scalar_before_sha256: str
    scalar_after_sha256: str
    scalar_guard_shape: Optional[Tuple[int, ...]]
    scalar_guard_sha256: Optional[str]
    scalar_guard_max_hex: Optional[str]
    componentwise_radius_shape: Optional[Tuple[int, ...]]
    componentwise_radius_sha256: Optional[str]
    componentwise_radius_max_hex: Optional[str]
    componentwise_penalty_shape: Optional[Tuple[int, ...]]
    componentwise_penalty_sha256: Optional[str]
    componentwise_penalty_max_hex: Optional[str]
    trace_sha256: str
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)


def _trace_body(
    *,
    owner: _FrameOwner,
    expectation: AffineExecutionExpectation,
    nominal_shape: Tuple[int, ...],
    nominal_sha256: str,
    nominal_absolute_max_hex: str,
    scalar_before_sha256: str,
    scalar_after_sha256: str,
    scalar_guard_shape: Optional[Tuple[int, ...]],
    scalar_guard_sha256: Optional[str],
    scalar_guard_max_hex: Optional[str],
    componentwise_radius_shape: Optional[Tuple[int, ...]],
    componentwise_radius_sha256: Optional[str],
    componentwise_radius_max_hex: Optional[str],
    componentwise_penalty_shape: Optional[Tuple[int, ...]],
    componentwise_penalty_sha256: Optional[str],
    componentwise_penalty_max_hex: Optional[str],
    trace_nonce_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema": ABSORPTION_TRACE_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "proof_authority": False,
        "authority_scope": "compact_process_local_absorption_trace_only",
        "frame_binding_sha256": owner.binding.binding_sha256,
        "owner_nonce_sha256": owner.nonce_sha256,
        "trace_nonce_sha256": trace_nonce_sha256,
        "expectation_sha256": expectation.expectation_sha256,
        "execution_index": expectation.execution_index,
        "query_span_sha256": expectation.span.content_sha256,
        "partition_sha256": expectation.partition.partition_sha256,
        "nominal_shape": list(nominal_shape),
        "nominal_sha256": nominal_sha256,
        "nominal_absolute_max_hex": nominal_absolute_max_hex,
        "scalar_before_sha256": scalar_before_sha256,
        "scalar_after_sha256": scalar_after_sha256,
        "scalar_guard_shape": (
            None
            if scalar_guard_shape is None
            else list(scalar_guard_shape)
        ),
        "scalar_guard_sha256": scalar_guard_sha256,
        "scalar_guard_max_hex": scalar_guard_max_hex,
        "componentwise_radius_shape": (
            None
            if componentwise_radius_shape is None
            else list(componentwise_radius_shape)
        ),
        "componentwise_radius_sha256": componentwise_radius_sha256,
        "componentwise_radius_max_hex": componentwise_radius_max_hex,
        "componentwise_penalty_shape": (
            None
            if componentwise_penalty_shape is None
            else list(componentwise_penalty_shape)
        ),
        "componentwise_penalty_sha256": componentwise_penalty_sha256,
        "componentwise_penalty_max_hex": componentwise_penalty_max_hex,
        "active_mask_sha256": _mask_sha256(
            expectation.partition.active_mask,
            row_count=expectation.partition.row_count,
        ),
        "fallback_mask_sha256": _mask_sha256(
            expectation.partition.fallback_mask,
            row_count=expectation.partition.row_count,
        ),
        "arrays_retained": False,
    }


def _mint_compact_absorption_trace(
    owner: _FrameOwner,
    expectation: AffineExecutionExpectation,
    *,
    nominal: Any,
    scalar_before: Any,
    scalar_after: Any,
    scalar_guard: Optional[Any] = None,
    componentwise_radius: Optional[Any] = None,
    componentwise_penalty: Optional[Any] = None,
) -> CompactAbsorptionTrace:
    """Validate live arrays, record compact metadata, and retain no arrays."""

    if (
        not _verify_owner(owner)
        or not validate_affine_execution_expectation(expectation)
        or expectation._owner_nonce != owner._nonce
    ):
        _fail("INVALID_TRACE", "trace expectation is not frame-owned")
    deadline = _deadline_from_hex(owner.binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before compact absorption trace")
    rows = expectation.span.row_count
    nominal_array = _snapshot_f64_array(
        nominal, name="nominal", ndim=2, nonnegative=False
    )
    before = _snapshot_f64_array(
        scalar_before,
        name="scalar_before",
        ndim=1,
        nonnegative=False,
    )
    after = _snapshot_f64_array(
        scalar_after,
        name="scalar_after",
        ndim=1,
        nonnegative=False,
    )
    if (
        nominal_array.shape[0] != rows
        or before.shape != (rows,)
        or after.shape != (rows,)
    ):
        _fail("INVALID_TRACE", "trace arrays do not match query span")
    scalar_rows = _mask_to_bool(
        expectation.partition.scalar_mask, row_count=rows
    )
    componentwise_rows = _mask_to_bool(
        expectation.partition.componentwise_mask, row_count=rows
    )
    active_rows = _mask_to_bool(
        expectation.partition.active_mask, row_count=rows
    )

    guard_meta: Optional[Dict[str, Any]] = None
    if np.any(scalar_rows):
        if scalar_guard is None:
            _fail("MISSING_GUARD", "scalar-policy rows need a scalar guard")
        guard = _snapshot_f64_array(
            scalar_guard,
            name="scalar_guard",
            ndim=1,
            nonnegative=True,
        )
        if guard.shape != (rows,) or np.any(guard[~scalar_rows] != 0.0):
            _fail(
                "POLICY_MISMATCH",
                "scalar guard must be zero outside scalar-policy rows",
            )
        actual_active = scalar_rows & (guard != 0.0)
        if not np.array_equal(actual_active, active_rows):
            _fail(
                "ACTIVE_MASK_MISMATCH",
                "active mask differs from nonzero final scalar guard",
            )
        expected_after = np.ascontiguousarray(
            before.copy(), dtype=np.float64
        )
        if np.any(active_rows):
            expected_after[active_rows] = np.nextafter(
                np.asarray(
                    before[active_rows] + (-guard[active_rows]),
                    dtype=np.float64,
                ),
                np.float64(-math.inf),
            )
        if not np.all(np.isfinite(expected_after)):
            _fail(
                "ABSORPTION_MISMATCH",
                "scalar guard absorption is non-finite",
            )
        inactive_scalar = scalar_rows & ~active_rows
        if not np.array_equal(
            after[inactive_scalar].view(np.uint64),
            before[inactive_scalar].view(np.uint64),
        ):
            _fail(
                "ZERO_GUARD_CHANGED",
                "zero-guard scalar rows must remain bit-identical",
            )
        if not np.array_equal(
            after.view(np.uint64), expected_after.view(np.uint64)
        ):
            _fail(
                "ABSORPTION_MISMATCH",
                "scalar guard was not subtracted exactly once row-locally",
            )
        guard_meta = _array_metadata(guard, nonnegative=True)
    elif scalar_guard is not None:
        _fail("DOUBLE_CHARGE", "componentwise execution supplied scalar guard")

    radius_meta: Optional[Dict[str, Any]] = None
    penalty_meta: Optional[Dict[str, Any]] = None
    if np.any(componentwise_rows):
        if componentwise_radius is None or componentwise_penalty is None:
            _fail(
                "MISSING_GUARD",
                "componentwise-policy rows need radius and scalar penalty",
            )
        radius = _snapshot_f64_array(
            componentwise_radius,
            name="componentwise_radius",
            ndim=2,
            nonnegative=True,
        )
        if (
            radius.shape != nominal_array.shape
            or np.any(radius[~componentwise_rows] != 0.0)
        ):
            _fail(
                "POLICY_MISMATCH",
                "componentwise radius is outside its policy rows",
            )
        penalty = _snapshot_f64_array(
            componentwise_penalty,
            name="componentwise_penalty",
            ndim=1,
            nonnegative=True,
        )
        if (
            penalty.shape != (rows,)
            or np.any(penalty[~componentwise_rows] != 0.0)
        ):
            _fail(
                "POLICY_MISMATCH",
                "componentwise penalty is outside its policy rows",
            )
        expected_after = np.ascontiguousarray(
            before.copy(), dtype=np.float64
        )
        if np.any(penalty):
            expected_after = np.nextafter(
                np.asarray(before + (-penalty), dtype=np.float64),
                np.float64(-math.inf),
            )
        if (
            not np.all(np.isfinite(expected_after))
            or not np.array_equal(
                after.view(np.uint64), expected_after.view(np.uint64)
            )
        ):
            _fail(
                "ABSORPTION_MISMATCH",
                "componentwise penalty was not subtracted exactly once",
            )
        radius_meta = _array_metadata(radius, nonnegative=True)
        penalty_meta = _array_metadata(penalty, nonnegative=True)
    elif (
        componentwise_radius is not None
        or componentwise_penalty is not None
    ):
        _fail(
            "DOUBLE_CHARGE",
            "scalar execution supplied a radius or componentwise penalty",
        )

    nominal_meta = _array_metadata(nominal_array, nonnegative=False)
    before_sha = _array_sha256(before)
    after_sha = _array_sha256(after)
    nonce = secrets.token_hex(32)
    trace_kwargs: Dict[str, Any] = {
        "nominal_shape": tuple(nominal_meta["shape"]),
        "nominal_sha256": nominal_meta["sha256"],
        "nominal_absolute_max_hex": nominal_meta["absolute_max_hex"],
        "scalar_before_sha256": before_sha,
        "scalar_after_sha256": after_sha,
        "scalar_guard_shape": (
            None
            if guard_meta is None
            else tuple(guard_meta["shape"])
        ),
        "scalar_guard_sha256": (
            None if guard_meta is None else guard_meta["sha256"]
        ),
        "scalar_guard_max_hex": (
            None if guard_meta is None else guard_meta["maximum_hex"]
        ),
        "componentwise_radius_shape": (
            None
            if radius_meta is None
            else tuple(radius_meta["shape"])
        ),
        "componentwise_radius_sha256": (
            None if radius_meta is None else radius_meta["sha256"]
        ),
        "componentwise_radius_max_hex": (
            None if radius_meta is None else radius_meta["maximum_hex"]
        ),
        "componentwise_penalty_shape": (
            None
            if penalty_meta is None
            else tuple(penalty_meta["shape"])
        ),
        "componentwise_penalty_sha256": (
            None if penalty_meta is None else penalty_meta["sha256"]
        ),
        "componentwise_penalty_max_hex": (
            None if penalty_meta is None else penalty_meta["maximum_hex"]
        ),
    }
    body = _trace_body(
        owner=owner,
        expectation=expectation,
        trace_nonce_sha256=hashlib.sha256(
            nonce.encode("ascii")
        ).hexdigest(),
        **trace_kwargs,
    )
    trace = CompactAbsorptionTrace(
        expectation=expectation,
        trace_sha256=_json_sha256(body),
        _nonce=nonce,
        _owner_nonce=owner._nonce,
        _owner=owner,
        _capability=_TRACE_CAPABILITY,
        **trace_kwargs,
    )
    with _TRACE_LOCK:
        _TRACE_REGISTRY[nonce] = trace
        _TRACE_SEALS[nonce] = (
            trace.trace_sha256,
            id(trace.expectation),
        )
    weakref.finalize(
        trace, _drop_external_seal, _TRACE_SEALS, _TRACE_LOCK, nonce
    )
    _check_deadline(deadline, where="after compact absorption trace")
    return trace


def validate_compact_absorption_trace(value: Any) -> bool:
    try:
        if (
            not isinstance(value, CompactAbsorptionTrace)
            or value._capability is not _TRACE_CAPABILITY
            or not _verify_owner(value._owner)
            or value._owner._nonce != value._owner_nonce
            or not validate_affine_execution_expectation(
                value.expectation
            )
            or value.expectation._owner_nonce != value._owner_nonce
        ):
            return False
        with _TRACE_LOCK:
            seal = _TRACE_SEALS.get(value._nonce)
            if (
                _TRACE_REGISTRY.get(value._nonce) is not value
                or seal is None
                or seal[1] != id(value.expectation)
                or not hmac.compare_digest(seal[0], value.trace_sha256)
            ):
                return False
        for shape in (
            value.nominal_shape,
            value.scalar_guard_shape,
            value.componentwise_radius_shape,
            value.componentwise_penalty_shape,
        ):
            if shape is not None and type(shape) is not tuple:
                return False
        for name in (
            "nominal_sha256",
            "scalar_before_sha256",
            "scalar_after_sha256",
        ):
            _require_sha256(getattr(value, name), name=name)
        _finite_hex(
            value.nominal_absolute_max_hex,
            name="nominal_absolute_max_hex",
            nonnegative=True,
        )
        if value.scalar_guard_shape is None:
            if (
                value.scalar_guard_sha256 is not None
                or value.scalar_guard_max_hex is not None
            ):
                return False
        else:
            _require_sha256(
                value.scalar_guard_sha256,
                name="scalar_guard_sha256",
            )
            _finite_hex(
                value.scalar_guard_max_hex,
                name="scalar_guard_max_hex",
                nonnegative=True,
            )
        if value.componentwise_radius_shape is None:
            if (
                value.componentwise_radius_sha256 is not None
                or value.componentwise_radius_max_hex is not None
                or value.componentwise_penalty_shape is not None
                or value.componentwise_penalty_sha256 is not None
                or value.componentwise_penalty_max_hex is not None
            ):
                return False
        else:
            _require_sha256(
                value.componentwise_radius_sha256,
                name="componentwise_radius_sha256",
            )
            _finite_hex(
                value.componentwise_radius_max_hex,
                name="componentwise_radius_max_hex",
                nonnegative=True,
            )
            if value.componentwise_penalty_shape is None:
                return False
            _require_sha256(
                value.componentwise_penalty_sha256,
                name="componentwise_penalty_sha256",
            )
            _finite_hex(
                value.componentwise_penalty_max_hex,
                name="componentwise_penalty_max_hex",
                nonnegative=True,
            )
        body = _trace_body(
            owner=value._owner,
            expectation=value.expectation,
            nominal_shape=value.nominal_shape,
            nominal_sha256=value.nominal_sha256,
            nominal_absolute_max_hex=value.nominal_absolute_max_hex,
            scalar_before_sha256=value.scalar_before_sha256,
            scalar_after_sha256=value.scalar_after_sha256,
            scalar_guard_shape=value.scalar_guard_shape,
            scalar_guard_sha256=value.scalar_guard_sha256,
            scalar_guard_max_hex=value.scalar_guard_max_hex,
            componentwise_radius_shape=value.componentwise_radius_shape,
            componentwise_radius_sha256=value.componentwise_radius_sha256,
            componentwise_radius_max_hex=(
                value.componentwise_radius_max_hex
            ),
            componentwise_penalty_shape=(
                value.componentwise_penalty_shape
            ),
            componentwise_penalty_sha256=(
                value.componentwise_penalty_sha256
            ),
            componentwise_penalty_max_hex=(
                value.componentwise_penalty_max_hex
            ),
            trace_nonce_sha256=hashlib.sha256(
                value._nonce.encode("ascii")
            ).hexdigest(),
        )
        return hmac.compare_digest(_json_sha256(body), value.trace_sha256)
    except (
        AttributeError,
        TypeError,
        ValueError,
        QueryDualV51AuthorityError,
    ):
        return False


_LEDGER_CAPABILITY = object()
_LEDGER_REGISTRY: weakref.WeakValueDictionary[
    str, "_CompactGuardLedger"
] = weakref.WeakValueDictionary()
_LEDGER_SEALS: Dict[str, "_LedgerRuntime"] = {}
_LEDGER_LOCK = threading.Lock()
_CERTIFICATE_CAPABILITY = object()
_CERTIFICATE_REGISTRY: weakref.WeakValueDictionary[
    str, "CompactGuardLedgerCertificate"
] = weakref.WeakValueDictionary()
_CERTIFICATE_SEALS: Dict[
    str,
    Tuple[
        str,
        int,
        int,
        Tuple[int, ...],
        int,
        Tuple[int, ...],
        int,
        Tuple[int, ...],
        int,
    ],
] = {}
_CERTIFICATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CompactGuardLedgerCertificate:
    """Complete process-local semantic ledger; never lower-bound authority."""

    frame_binding: V51FrameBinding
    owner_nonce_sha256: str
    spans: Tuple[QuerySpan, ...]
    expectations: Tuple[AffineExecutionExpectation, ...]
    traces: Tuple[CompactAbsorptionTrace, ...]
    receipt: Mapping[str, Any]
    content_sha256: str
    proof_authority: bool
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)


@dataclass
class _LedgerRuntime:
    """Registry-external immutable schedule plus mutable one-shot state."""

    ledger_id: int
    owner: _FrameOwner
    spans: Tuple[QuerySpan, ...]
    expectations: Tuple[AffineExecutionExpectation, ...]
    by_key: Mapping[str, AffineExecutionExpectation]
    span_seals: Tuple[Tuple[int, str, str], ...]
    expectation_seals: Tuple[Tuple[int, str, str], ...]
    deadline: float
    deadline_hex: str
    operation_lock: threading.Lock
    recorded: Dict[str, CompactAbsorptionTrace] = field(
        default_factory=dict
    )
    trace_seals: Dict[str, Tuple[int, str, str]] = field(
        default_factory=dict
    )
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    poisoned: bool = False
    closed: bool = False
    failed: bool = False


def _span_record(span: QuerySpan) -> Dict[str, Any]:
    return {
        "span_index": span.span_index,
        "query_start": span.query_start,
        "query_end": span.query_end,
        "query_count": span.row_count,
        "query_total": span.query_total,
        "query_block_sha256": span.query_block_sha256,
        "query_span_sha256": span.content_sha256,
    }


def _expectation_record(
    expectation: AffineExecutionExpectation,
) -> Dict[str, Any]:
    return {
        "execution_index": expectation.execution_index,
        "expectation_sha256": expectation.expectation_sha256,
        "semantic_execution_sha256": _semantic_execution_sha256(
            expectation
        ),
        "query_span_sha256": expectation.span.content_sha256,
        "support_alias_content_sha256": (
            expectation.support_alias.content_sha256
        ),
        "catalog_key_sha256": (
            expectation.support_alias.catalog_key_sha256
        ),
        "branch": expectation.support_alias.branch,
        **_partition_body(expectation.partition),
    }


def _semantic_execution_sha256(
    expectation: AffineExecutionExpectation,
) -> str:
    """Nonce/index-free identity of one scheduled affine occurrence."""

    return _json_sha256(
        {
            "schema": "act.query_dual_v51_semantic_execution.v1",
            "query_span_sha256": expectation.span.content_sha256,
            "support_alias_content_sha256": (
                expectation.support_alias.content_sha256
            ),
            "partition_sha256": expectation.partition.partition_sha256,
            "input_coefficient_sha256": (
                expectation.input_coefficient_sha256
            ),
        }
    )


def _validate_span_partition(
    spans: Tuple[QuerySpan, ...],
    *,
    owner: _FrameOwner,
) -> Tuple[StageUse, str, int]:
    if not spans:
        _fail("INVALID_QUERY_SPAN", "query-span schedule cannot be empty")
    for span in spans:
        if (
            not validate_query_span(span)
            or span._owner_nonce != owner._nonce
        ):
            _fail("INVALID_QUERY_SPAN", "query span is not frame-owned")
    if [span.span_index for span in spans] != list(range(len(spans))):
        _fail(
            "INVALID_QUERY_SPAN",
            "query-span indices must be contiguous and zero-based",
        )
    stage_sha = spans[0].stage_use_sha256
    block_sha = spans[0].query_block_sha256
    total = spans[0].query_total
    cursor = 0
    for span in spans:
        if (
            span.stage_use_sha256 != stage_sha
            or span.query_block_sha256 != block_sha
            or span.query_total != total
        ):
            _fail(
                "INVALID_QUERY_SPAN",
                "spans do not belong to one stage/query block",
            )
        if span.query_start < cursor:
            _fail("QUERY_SPAN_OVERLAP", "query spans overlap")
        if span.query_start > cursor:
            _fail("QUERY_SPAN_GAP", "query spans contain a gap")
        cursor = span.query_end
    if cursor != total:
        _fail("QUERY_SPAN_GAP", "query spans do not cover the block tail")
    return _owner_stage_use(owner, stage_sha), block_sha, total


def _ledger_body(
    *,
    owner: _FrameOwner,
    certificate_nonce_sha256: str,
    spans: Tuple[QuerySpan, ...],
    expectations: Tuple[AffineExecutionExpectation, ...],
    traces: Tuple[CompactAbsorptionTrace, ...],
) -> Dict[str, Any]:
    span_records = [_span_record(value) for value in spans]
    expectation_records = [
        _expectation_record(value) for value in expectations
    ]
    scalar_rows = sum(
        value.partition.scalar_row_count for value in expectations
    )
    componentwise_rows = sum(
        value.partition.componentwise_row_count
        for value in expectations
    )
    active_rows = sum(
        value.partition.active_row_count for value in expectations
    )
    fallback_rows = sum(
        value.partition.fallback_row_count for value in expectations
    )
    return {
        "schema": LEDGER_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "proof_authority": False,
        "authority_scope": "compact_guard_accounting_only",
        "coverage_complete": True,
        "arrays_retained": False,
        "frame_binding_sha256": owner.binding.binding_sha256,
        "owner_nonce_sha256": owner.nonce_sha256,
        "certificate_nonce_sha256": certificate_nonce_sha256,
        "deadline_monotonic_hex": (
            owner.binding.deadline_monotonic_hex
        ),
        "stage_use_sha256": spans[0].stage_use_sha256,
        "query_block_sha256": spans[0].query_block_sha256,
        "query_total": spans[0].query_total,
        "spans": span_records,
        "span_schedule_sha256": _json_sha256(span_records),
        "span_count": len(spans),
        "expectations": expectation_records,
        "expectations_sha256": _json_sha256(expectation_records),
        "execution_count": len(expectations),
        "trace_sha256": [value.trace_sha256 for value in traces],
        "scalar_policy_row_count": scalar_rows,
        "componentwise_policy_row_count": componentwise_rows,
        "active_row_count": active_rows,
        "fallback_row_count": fallback_rows,
        "catalog_alias_sha256": sorted(
            {
                value.support_alias.content_sha256
                for value in expectations
            }
        ),
    }


class _CompactGuardLedger:
    """One-shot exact-coverage ledger whose committed state is metadata-only."""

    def __init__(
        self,
        *,
        authority: Any,
        owner: _FrameOwner,
        spans: Sequence[QuerySpan],
        expectations: Sequence[AffineExecutionExpectation],
    ):
        if authority is not _LEDGER_CAPABILITY or not _verify_owner(owner):
            _fail("INVALID_LEDGER", "ledger requires a live frame owner")
        deadline_hex = owner.binding.deadline_monotonic_hex
        deadline = _deadline_from_hex(deadline_hex)
        _check_deadline(deadline, where="before compact ledger mint")
        if (
            isinstance(spans, (str, bytes))
            or not isinstance(spans, Sequence)
            or isinstance(expectations, (str, bytes))
            or not isinstance(expectations, Sequence)
        ):
            _fail("INVALID_LEDGER", "spans/expectations must be sequences")
        frozen_spans = tuple(spans)
        _validate_span_partition(frozen_spans, owner=owner)
        frozen_expectations = tuple(expectations)
        if not frozen_expectations:
            _fail("INVALID_EXPECTATION", "expectations cannot be empty")
        by_key: Dict[str, AffineExecutionExpectation] = {}
        semantic_keys: set[str] = set()
        span_ids = {id(value) for value in frozen_spans}
        represented_spans: set[int] = set()
        for expected_index, expectation in enumerate(
            frozen_expectations
        ):
            if (
                not validate_affine_execution_expectation(expectation)
                or expectation._owner_nonce != owner._nonce
                or expectation.execution_index != expected_index
                or id(expectation.span) not in span_ids
            ):
                _fail(
                    "INVALID_EXPECTATION",
                    "execution schedule is not contiguous/frame-owned",
                )
            if expectation.expectation_sha256 in by_key:
                _fail(
                    "INVALID_EXPECTATION",
                    "duplicate affine execution expectation",
                )
            semantic_key = _semantic_execution_sha256(expectation)
            if semantic_key in semantic_keys:
                _fail(
                    "DUPLICATE_EXECUTION",
                    "duplicate semantic affine execution occurrence",
                )
            by_key[expectation.expectation_sha256] = expectation
            semantic_keys.add(semantic_key)
            represented_spans.add(id(expectation.span))
        if represented_spans != span_ids:
            _fail(
                "MISSING_EXECUTION",
                "one or more query spans have no affine execution",
            )
        _check_deadline(
            deadline, where="after compact ledger schedule validation"
        )
        nonce = secrets.token_hex(32)
        operation_lock = threading.Lock()
        frozen_by_key = MappingProxyType(by_key)
        self._owner = owner
        self._spans = frozen_spans
        self._expectations = frozen_expectations
        self._by_key = frozen_by_key
        self._deadline = deadline
        self._deadline_hex = deadline_hex
        self._lock = operation_lock
        self._nonce = nonce
        self._capability = _LEDGER_CAPABILITY
        runtime = _LedgerRuntime(
            ledger_id=id(self),
            owner=owner,
            spans=frozen_spans,
            expectations=frozen_expectations,
            by_key=frozen_by_key,
            span_seals=tuple(
                (id(value), value._nonce, value.content_sha256)
                for value in frozen_spans
            ),
            expectation_seals=tuple(
                (
                    id(value),
                    value._nonce,
                    value.expectation_sha256,
                )
                for value in frozen_expectations
            ),
            deadline=deadline,
            deadline_hex=deadline_hex,
            operation_lock=operation_lock,
        )
        with _LEDGER_LOCK:
            _LEDGER_REGISTRY[nonce] = self
            _LEDGER_SEALS[nonce] = runtime
        weakref.finalize(
            self,
            _drop_external_seal,
            _LEDGER_SEALS,
            _LEDGER_LOCK,
            nonce,
        )
        try:
            _check_deadline(deadline, where="after compact ledger mint")
        except Exception:
            with runtime.state_lock:
                runtime.failed = True
                runtime.closed = True
            with _LEDGER_LOCK:
                _LEDGER_REGISTRY.pop(nonce, None)
                _LEDGER_SEALS.pop(nonce, None)
            raise

    def _runtime(self) -> _LedgerRuntime:
        """Recover this exact ledger's external seal and validate its schedule."""

        try:
            nonce = self._nonce
            with _LEDGER_LOCK:
                runtime = _LEDGER_SEALS.get(nonce)
                registered = _LEDGER_REGISTRY.get(nonce)
            if (
                registered is not self
                or runtime is None
                or runtime.ledger_id != id(self)
                or self._capability is not _LEDGER_CAPABILITY
                or self._owner is not runtime.owner
                or self._spans is not runtime.spans
                or self._expectations is not runtime.expectations
                or self._by_key is not runtime.by_key
                or self._lock is not runtime.operation_lock
                or type(self._deadline) is not float
                or self._deadline.hex() != runtime.deadline.hex()
                or self._deadline_hex != runtime.deadline_hex
                or runtime.deadline_hex
                != runtime.owner.binding.deadline_monotonic_hex
                or not _verify_owner(runtime.owner)
            ):
                _fail(
                    "INVALID_LEDGER",
                    "ledger identity or immutable schedule seal changed",
                )
            if (
                tuple(
                    (id(value), value._nonce, value.content_sha256)
                    for value in runtime.spans
                )
                != runtime.span_seals
                or tuple(
                    (
                        id(value),
                        value._nonce,
                        value.expectation_sha256,
                    )
                    for value in runtime.expectations
                )
                != runtime.expectation_seals
            ):
                _fail(
                    "INVALID_LEDGER",
                    "ledger schedule objects changed after mint",
                )
            _validate_span_partition(runtime.spans, owner=runtime.owner)
            span_ids = {id(value) for value in runtime.spans}
            represented_spans: set[int] = set()
            semantic_keys: set[str] = set()
            for index, expectation in enumerate(runtime.expectations):
                semantic_key = _semantic_execution_sha256(expectation)
                if (
                    not validate_affine_execution_expectation(expectation)
                    or expectation._owner_nonce != runtime.owner._nonce
                    or expectation.execution_index != index
                    or id(expectation.span) not in span_ids
                    or runtime.by_key.get(
                        expectation.expectation_sha256
                    )
                    is not expectation
                    or semantic_key in semantic_keys
                ):
                    _fail(
                        "INVALID_LEDGER",
                        "ledger expectation schedule changed after mint",
                    )
                represented_spans.add(id(expectation.span))
                semantic_keys.add(semantic_key)
            if (
                represented_spans != span_ids
                or len(runtime.by_key) != len(runtime.expectations)
            ):
                _fail(
                    "INVALID_LEDGER",
                    "ledger no longer represents every query span",
                )
            return runtime
        except QueryDualV51AuthorityError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise QueryDualV51AuthorityError(
                "INVALID_LEDGER", "ledger seal validation failed"
            ) from exc

    def _invalidate(self, runtime: Optional[_LedgerRuntime] = None) -> None:
        if runtime is None:
            try:
                runtime = self._runtime()
            except QueryDualV51AuthorityError:
                return
        with runtime.state_lock:
            runtime.failed = True
            runtime.closed = True
        runtime.recorded.clear()
        runtime.trace_seals.clear()

    def _poison_concurrent_use(self) -> None:
        """Invalidate an in-flight operation without racing its owned data."""

        try:
            runtime = self._runtime()
        except QueryDualV51AuthorityError:
            return
        with runtime.state_lock:
            if not runtime.closed:
                runtime.poisoned = True

    def _check(self, *, where: str) -> _LedgerRuntime:
        runtime = self._runtime()
        with runtime.state_lock:
            invalid_state = (
                runtime.closed or runtime.failed or runtime.poisoned
            )
        if invalid_state:
            _fail("INVALID_LEDGER", "ledger is closed or lost its owner")
        try:
            _check_deadline(runtime.deadline, where=where)
        except QueryDualV51AuthorityError:
            self._invalidate(runtime)
            raise
        return runtime

    def record(self, trace: CompactAbsorptionTrace) -> None:
        runtime = self._runtime()
        if not runtime.operation_lock.acquire(blocking=False):
            self._poison_concurrent_use()
            _fail("CONCURRENT_LEDGER", "concurrent ledger use is forbidden")
        try:
            runtime = self._check(where="before absorption record")
            if (
                not validate_compact_absorption_trace(trace)
                or trace._owner_nonce != runtime.owner._nonce
            ):
                self._invalidate(runtime)
                _fail("INVALID_TRACE", "trace is not frame-owned")
            key = trace.expectation.expectation_sha256
            expected = runtime.by_key.get(key)
            if expected is None or trace.expectation is not expected:
                self._invalidate(runtime)
                _fail(
                    "UNEXPECTED_EXECUTION",
                    "trace has no frozen execution expectation",
                )
            if key in runtime.recorded:
                self._invalidate(runtime)
                _fail(
                    "DOUBLE_CHARGE",
                    "one affine execution received multiple traces",
                )
            runtime.recorded[key] = trace
            runtime.trace_seals[key] = (
                id(trace),
                trace._nonce,
                trace.trace_sha256,
            )
            self._check(where="after absorption record")
        except Exception:
            self._invalidate(runtime)
            raise
        finally:
            runtime.operation_lock.release()

    def commit(self) -> CompactGuardLedgerCertificate:
        runtime = self._runtime()
        if not runtime.operation_lock.acquire(blocking=False):
            self._poison_concurrent_use()
            _fail("CONCURRENT_LEDGER", "concurrent ledger use is forbidden")
        try:
            runtime = self._check(where="before compact ledger commit")
            missing = [
                value.execution_index
                for value in runtime.expectations
                if value.expectation_sha256 not in runtime.recorded
            ]
            if missing:
                self._invalidate(runtime)
                _fail(
                    "MISSING_CHARGE",
                    f"affine executions lack absorption traces: {missing}",
                )
            ordered = tuple(
                runtime.recorded[value.expectation_sha256]
                for value in runtime.expectations
            )
            for expected, trace in zip(runtime.expectations, ordered):
                if (
                    runtime.trace_seals.get(
                        expected.expectation_sha256
                    )
                    != (id(trace), trace._nonce, trace.trace_sha256)
                    or not validate_compact_absorption_trace(trace)
                    or trace._owner_nonce != runtime.owner._nonce
                    or trace.expectation is not expected
                ):
                    self._invalidate(runtime)
                    _fail(
                        "INVALID_TRACE",
                        "recorded trace changed before ledger commit",
                    )
            nonce = secrets.token_hex(32)
            nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
            body = _ledger_body(
                owner=runtime.owner,
                certificate_nonce_sha256=nonce_sha,
                spans=runtime.spans,
                expectations=runtime.expectations,
                traces=ordered,
            )
            content_sha = _json_sha256(body)
            receipt_body = dict(body)
            receipt_body["content_sha256"] = content_sha
            receipt_body["receipt_sha256"] = _json_sha256(receipt_body)
            certificate = CompactGuardLedgerCertificate(
                frame_binding=runtime.owner.binding,
                owner_nonce_sha256=runtime.owner.nonce_sha256,
                spans=runtime.spans,
                expectations=runtime.expectations,
                traces=ordered,
                receipt=_deep_freeze(receipt_body),
                content_sha256=content_sha,
                proof_authority=False,
                _nonce=nonce,
                _owner_nonce=runtime.owner._nonce,
                _owner=runtime.owner,
                _capability=_CERTIFICATE_CAPABILITY,
            )
            # This state-lock section is the commit linearization point.  A
            # contender arriving before it poisons this operation; a caller
            # arriving after ``_closed`` observes an already committed ledger.
            with runtime.state_lock:
                if runtime.closed or runtime.failed or runtime.poisoned:
                    _fail(
                        "INVALID_LEDGER",
                        "concurrent use poisoned compact ledger commit",
                    )
                try:
                    for expected, trace in zip(
                        runtime.expectations, ordered
                    ):
                        if (
                            runtime.trace_seals.get(
                                expected.expectation_sha256
                            )
                            != (
                                id(trace),
                                trace._nonce,
                                trace.trace_sha256,
                            )
                            or not validate_compact_absorption_trace(
                                trace
                            )
                            or trace.expectation is not expected
                        ):
                            _fail(
                                "INVALID_TRACE",
                                "trace changed during ledger commit",
                            )
                    _check_deadline(
                        runtime.deadline,
                        where="at compact ledger publication",
                    )
                    with _CERTIFICATE_LOCK:
                        _CERTIFICATE_REGISTRY[nonce] = certificate
                        _CERTIFICATE_SEALS[nonce] = (
                            content_sha,
                            id(certificate.frame_binding),
                            id(certificate.spans),
                            tuple(id(value) for value in certificate.spans),
                            id(certificate.expectations),
                            tuple(
                                id(value)
                                for value in certificate.expectations
                            ),
                            id(certificate.traces),
                            tuple(id(value) for value in certificate.traces),
                            id(certificate.receipt),
                        )
                    _check_deadline(
                        runtime.deadline,
                        where="after compact ledger publication",
                    )
                except Exception:
                    with _CERTIFICATE_LOCK:
                        _CERTIFICATE_REGISTRY.pop(nonce, None)
                        _CERTIFICATE_SEALS.pop(nonce, None)
                    runtime.failed = True
                    runtime.closed = True
                    raise
                runtime.closed = True
            weakref.finalize(
                certificate,
                _drop_external_seal,
                _CERTIFICATE_SEALS,
                _CERTIFICATE_LOCK,
                nonce,
            )
            runtime.recorded.clear()
            runtime.trace_seals.clear()
            return certificate
        except Exception:
            self._invalidate(runtime)
            raise
        finally:
            runtime.operation_lock.release()

    def abort(self) -> None:
        runtime = self._runtime()
        with runtime.operation_lock:
            self._check(where="before compact ledger abort")
            self._invalidate(runtime)


def _mint_compact_guard_ledger(
    owner: _FrameOwner,
    spans: Sequence[QuerySpan],
    expectations: Sequence[AffineExecutionExpectation],
) -> _CompactGuardLedger:
    return _CompactGuardLedger(
        authority=_LEDGER_CAPABILITY,
        owner=owner,
        spans=spans,
        expectations=expectations,
    )


def validate_compact_guard_ledger_certificate(value: Any) -> bool:
    try:
        if (
            not isinstance(value, CompactGuardLedgerCertificate)
            or value._capability is not _CERTIFICATE_CAPABILITY
            or value.proof_authority is not False
            or not validate_frame_binding(value.frame_binding)
            or not _verify_owner(value._owner)
            or value._owner._nonce != value._owner_nonce
            or value.owner_nonce_sha256 != value._owner.nonce_sha256
            or value.frame_binding.binding_sha256
            != value._owner.binding.binding_sha256
            or value.frame_binding is not value._owner.binding
            or type(value.spans) is not tuple
            or type(value.expectations) is not tuple
            or type(value.traces) is not tuple
            or type(value.receipt) is not MappingProxyType
            or len(value.expectations) != len(value.traces)
        ):
            return False
        with _CERTIFICATE_LOCK:
            seal = _CERTIFICATE_SEALS.get(value._nonce)
            if (
                _CERTIFICATE_REGISTRY.get(value._nonce) is not value
                or seal is None
                or seal[1] != id(value.frame_binding)
                or seal[2] != id(value.spans)
                or seal[3] != tuple(id(item) for item in value.spans)
                or seal[4] != id(value.expectations)
                or seal[5]
                != tuple(id(item) for item in value.expectations)
                or seal[6] != id(value.traces)
                or seal[7] != tuple(id(item) for item in value.traces)
                or seal[8] != id(value.receipt)
                or not hmac.compare_digest(seal[0], value.content_sha256)
            ):
                return False
        _validate_span_partition(value.spans, owner=value._owner)
        if [item.execution_index for item in value.expectations] != list(
            range(len(value.expectations))
        ):
            return False
        span_ids = {id(span) for span in value.spans}
        represented_spans: set[int] = set()
        semantic_keys: set[str] = set()
        for expectation, trace in zip(value.expectations, value.traces):
            semantic_key = _semantic_execution_sha256(expectation)
            if (
                not validate_affine_execution_expectation(expectation)
                or not validate_compact_absorption_trace(trace)
                or expectation._owner_nonce != value._owner_nonce
                or trace._owner_nonce != value._owner_nonce
                or trace.expectation is not expectation
                or id(expectation.span) not in span_ids
                or semantic_key in semantic_keys
            ):
                return False
            represented_spans.add(id(expectation.span))
            semantic_keys.add(semantic_key)
        if represented_spans != span_ids:
            return False
        body = _ledger_body(
            owner=value._owner,
            certificate_nonce_sha256=hashlib.sha256(
                value._nonce.encode("ascii")
            ).hexdigest(),
            spans=value.spans,
            expectations=value.expectations,
            traces=value.traces,
        )
        content_sha = _json_sha256(body)
        receipt = dict(value.receipt)
        claimed = str(receipt.pop("receipt_sha256"))
        expected = dict(body)
        expected["content_sha256"] = content_sha
        return bool(
            hmac.compare_digest(content_sha, value.content_sha256)
            and set(receipt) == set(expected)
            and all(
                _canonical_value(receipt[key])
                == _canonical_value(expected[key])
                for key in expected
            )
            and hmac.compare_digest(_json_sha256(receipt), claimed)
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        QueryDualV51AuthorityError,
    ):
        return False


__all__ = [
    "ABSORPTION_TRACE_SCHEMA",
    "AffineExecutionExpectation",
    "BRANCH_CONV_DENSE",
    "BRANCH_CONV_SPARSE",
    "BRANCH_DENSE",
    "CATALOG_ALIAS_SCHEMA",
    "CompactAbsorptionTrace",
    "CompactGuardLedgerCertificate",
    "EXPECTATION_SCHEMA",
    "FRAME_SCHEMA",
    "LEDGER_SCHEMA",
    "NUMERIC_PROTOCOL",
    "POLICY_COMPONENTWISE",
    "POLICY_PARTITION_SCHEMA",
    "POLICY_SCALAR",
    "QUERY_SPAN_SCHEMA",
    "QueryDualV51AuthorityError",
    "QuerySpan",
    "RowPolicyPartition",
    "STAGE_PROPERTY",
    "STAGE_TARGET",
    "StageUse",
    "SupportCatalogAlias",
    "V51FrameBinding",
    "validate_affine_execution_expectation",
    "validate_compact_absorption_trace",
    "validate_compact_guard_ledger_certificate",
    "validate_frame_binding",
    "validate_query_span",
    "validate_row_policy_partition",
    "validate_stage_use",
    "validate_support_catalog_alias",
]
