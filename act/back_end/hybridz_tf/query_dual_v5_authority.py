#!/usr/bin/env python3
# ===- query_dual_v5_authority.py - V5 numeric authority sidecar ----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Fail-closed authority primitives for opt-in scalar affine guards.

This module is deliberately independent of ``query_dual_replay.py``.  It
does not evaluate a neural network and it never grants lower-bound proof
authority.  Instead, it provides a small process-local sidecar that a future
V5 replay session can use to prove three accounting facts:

* every box-dependent support belongs to one exact session/frame/stage;
* every expected affine execution uses exactly one guard policy; and
* scalar-compressed and legacy componentwise guards cannot both be charged.

The current V3 session, kernel, schemas, and receipts are therefore untouched.
All constructors that mint process-local authority are private by convention
and additionally protected by identity registries.  Copies, dataclass
replacement, receipt rehashing, and cross-frame reuse do not inherit
authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import threading
import time
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np


NUMERIC_PROTOCOL = "scalar_compressed_affine_roundoff_v5"
AUTHORITY_SCHEMA = "act.query_dual_v5_authority.v1"
SUPPORT_SCHEMA = "act.scalar_affine_support.v1"
LEDGER_SCHEMA = "act.scalar_affine_guard_ledger.v1"

POLICY_SCALAR = "scalar_compressed_once"
POLICY_COMPONENTWISE = "componentwise_radius_once"

BRANCH_DENSE = "DENSE"
BRANCH_CONV_DENSE = "CONV2D_DENSE"
BRANCH_CONV_SPARSE = "CONV2D_SPARSE"

_PROTOCOL_MANIFEST_BODY: Mapping[str, Any] = {
    "schema": AUTHORITY_SCHEMA,
    "numeric_protocol": NUMERIC_PROTOCOL,
    "authority_scope": "guard_accounting_only_not_lower_bound_proof",
    "dense_scalar_formula": (
        "G_up=up(gamma_dot*P_up+tau_dot*B_up)"
    ),
    "tau_formula": "up(k*eta/(1-k*u))",
    "nonnegative_dot_upper": "up((nominal+tau)/(1-gamma))",
    "guard_exclusivity": "exactly_one_of_scalar_or_componentwise",
    "support_scope": "one_process_local_session_frame_stage",
    "conv_branch_policy": "sparse_iff_8_times_nonzero_le_dense",
    "v3_default_mutation": False,
}


class QueryDualV5AuthorityError(RuntimeError):
    """Stable fail-closed error for V5 sidecar misuse."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise QueryDualV5AuthorityError(code, message)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
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
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


PROTOCOL_MANIFEST = _deep_freeze(_PROTOCOL_MANIFEST_BODY)
PROTOCOL_MANIFEST_SHA256 = _json_sha256(_PROTOCOL_MANIFEST_BODY)


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


def _finite_hex(value: Any, *, name: str, nonnegative: bool) -> str:
    if not isinstance(value, str):
        _fail("INVALID_BINDING", f"{name} must be a binary64 hex string")
    try:
        numeric = float.fromhex(value)
    except (TypeError, ValueError) as exc:
        raise QueryDualV5AuthorityError(
            "INVALID_BINDING", f"{name} is not a binary64 hex string"
        ) from exc
    if not math.isfinite(numeric) or (nonnegative and numeric < 0.0):
        _fail("INVALID_BINDING", f"{name} is outside its finite domain")
    if float(numeric).hex() != value:
        _fail("INVALID_BINDING", f"{name} is not in canonical binary64 hex form")
    return value


def _immutable_f64(value: Any, *, name: str, nonnegative: bool) -> np.ndarray:
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except Exception as exc:
        raise QueryDualV5AuthorityError(
            "INVALID_NUMERIC", f"{name}: {exc}"
        ) from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        _fail("INVALID_NUMERIC", f"{name} must be nonempty and finite")
    if nonnegative and np.any(array < 0.0):
        _fail("INVALID_NUMERIC", f"{name} must be nonnegative")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )
    frozen.setflags(write=False)
    return frozen


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0<f8\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _deadline_from_hex(value: str) -> float:
    _finite_hex(value, name="deadline_monotonic_hex", nonnegative=False)
    return float.fromhex(value)


def _check_deadline(deadline: float, *, where: str) -> None:
    if time.monotonic() >= float(deadline):
        raise QueryDualV5AuthorityError(
            "DEADLINE_EXPIRED", f"V5 authority deadline expired {where}"
        )


@dataclass(frozen=True)
class V5FrameBinding:
    """Public semantic identity of one V5 frame; not authority by itself."""

    session_nonce_sha256: str
    frame_nonce_sha256: str
    frame_content_sha256: str
    root_receipt_sha256: str
    parent_chain_sha256: str
    deadline_monotonic_hex: str
    stage_kind: str
    stage_index: Optional[int]
    start_lid: Optional[int]
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "session_nonce_sha256",
            "frame_nonce_sha256",
            "frame_content_sha256",
            "root_receipt_sha256",
            "parent_chain_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _deadline_from_hex(self.deadline_monotonic_hex)
        if self.stage_kind not in {"TARGET", "PROPERTY"}:
            _fail("INVALID_BINDING", "stage_kind must be TARGET or PROPERTY")
        if self.stage_kind == "TARGET":
            index = _exact_nonnegative_int(
                self.stage_index, name="stage_index"
            )
            if self.start_lid is None:
                _fail("INVALID_BINDING", "TARGET frame requires a start_lid")
            start = _exact_nonnegative_int(self.start_lid, name="start_lid")
        else:
            if self.stage_index is not None or self.start_lid is not None:
                _fail(
                    "INVALID_BINDING",
                    "PROPERTY frame requires null stage_index/start_lid",
                )
            index = None
            start = None
        body = {
            "schema": "act.query_dual_v5_frame_binding.v1",
            "numeric_protocol": NUMERIC_PROTOCOL,
            "session_nonce_sha256": self.session_nonce_sha256,
            "frame_nonce_sha256": self.frame_nonce_sha256,
            "frame_content_sha256": self.frame_content_sha256,
            "root_receipt_sha256": self.root_receipt_sha256,
            "parent_chain_sha256": self.parent_chain_sha256,
            "deadline_monotonic_hex": self.deadline_monotonic_hex,
            "stage_kind": self.stage_kind,
            "stage_index": index,
            "start_lid": start,
        }
        object.__setattr__(self, "stage_index", index)
        object.__setattr__(self, "start_lid", start)
        object.__setattr__(self, "binding_sha256", _json_sha256(body))


def validate_frame_binding(value: Any) -> bool:
    try:
        if not isinstance(value, V5FrameBinding):
            return False
        rebuilt = V5FrameBinding(
            session_nonce_sha256=value.session_nonce_sha256,
            frame_nonce_sha256=value.frame_nonce_sha256,
            frame_content_sha256=value.frame_content_sha256,
            root_receipt_sha256=value.root_receipt_sha256,
            parent_chain_sha256=value.parent_chain_sha256,
            deadline_monotonic_hex=value.deadline_monotonic_hex,
            stage_kind=value.stage_kind,
            stage_index=value.stage_index,
            start_lid=value.start_lid,
        )
        return hmac.compare_digest(
            rebuilt.binding_sha256, value.binding_sha256
        )
    except (AttributeError, TypeError, ValueError, QueryDualV5AuthorityError):
        return False


_OWNER_CAPABILITY = object()
_OWNER_REGISTRY: weakref.WeakValueDictionary[
    str, "_FrameLocalSupportOwner"
] = weakref.WeakValueDictionary()
_OWNER_LOCK = threading.Lock()


@dataclass(frozen=True)
class _FrameLocalSupportOwner:
    binding: V5FrameBinding
    _nonce: str = field(repr=False)
    _capability: Any = field(repr=False, compare=False)

    @property
    def nonce_sha256(self) -> str:
        return hashlib.sha256(self._nonce.encode("ascii")).hexdigest()


def _mint_frame_local_support_owner(
    binding: V5FrameBinding,
) -> _FrameLocalSupportOwner:
    if not validate_frame_binding(binding):
        _fail("INVALID_BINDING", "cannot own an invalid V5 frame")
    deadline = _deadline_from_hex(binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before frame owner mint")
    nonce = secrets.token_hex(32)
    owner = _FrameLocalSupportOwner(
        binding=binding,
        _nonce=nonce,
        _capability=_OWNER_CAPABILITY,
    )
    with _OWNER_LOCK:
        _OWNER_REGISTRY[nonce] = owner
    _check_deadline(deadline, where="after frame owner mint")
    return owner


def _verify_owner(owner: Any) -> bool:
    try:
        if (
            not isinstance(owner, _FrameLocalSupportOwner)
            or owner._capability is not _OWNER_CAPABILITY
            or not validate_frame_binding(owner.binding)
        ):
            return False
        with _OWNER_LOCK:
            return _OWNER_REGISTRY.get(owner._nonce) is owner
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ConvBranchEvidence:
    """Canonical proof of the frozen sparse/dense branch threshold."""

    nonzero_count: int
    dense_count: int
    selected_branch: str = field(init=False)
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        nonzero = _exact_nonnegative_int(
            self.nonzero_count, name="nonzero_count"
        )
        dense = _exact_nonnegative_int(self.dense_count, name="dense_count")
        if dense <= 0 or nonzero > dense:
            _fail(
                "INVALID_BRANCH",
                "branch counts require 0 <= nonzero_count <= dense_count",
            )
        branch = (
            BRANCH_CONV_SPARSE
            if nonzero * 8 <= dense
            else BRANCH_CONV_DENSE
        )
        body = {
            "schema": "act.query_dual_v5_conv_branch.v1",
            "policy": "sparse_iff_8_times_nonzero_le_dense",
            "nonzero_count": nonzero,
            "dense_count": dense,
            "threshold_lhs": nonzero * 8,
            "threshold_rhs": dense,
            "selected_branch": branch,
        }
        object.__setattr__(self, "nonzero_count", nonzero)
        object.__setattr__(self, "dense_count", dense)
        object.__setattr__(self, "selected_branch", branch)
        object.__setattr__(self, "evidence_sha256", _json_sha256(body))


def validate_conv_branch_evidence(value: Any) -> bool:
    try:
        if not isinstance(value, ConvBranchEvidence):
            return False
        rebuilt = ConvBranchEvidence(
            nonzero_count=value.nonzero_count,
            dense_count=value.dense_count,
        )
        return bool(
            rebuilt.selected_branch == value.selected_branch
            and hmac.compare_digest(
                rebuilt.evidence_sha256, value.evidence_sha256
            )
        )
    except (AttributeError, TypeError, ValueError, QueryDualV5AuthorityError):
        return False


@dataclass(frozen=True)
class AffineExecutionKey:
    """One exact affine execution inside a frame/query block."""

    execution_index: int
    layer_id: int
    predecessor_id: int
    operator_kind: str
    branch: str
    branch_evidence: Optional[ConvBranchEvidence] = None
    key_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        execution = _exact_nonnegative_int(
            self.execution_index, name="execution_index"
        )
        layer = _exact_nonnegative_int(self.layer_id, name="layer_id")
        predecessor = _exact_nonnegative_int(
            self.predecessor_id, name="predecessor_id"
        )
        if self.operator_kind == "DENSE":
            if self.branch != BRANCH_DENSE or self.branch_evidence is not None:
                _fail("INVALID_BRANCH", "DENSE requires the DENSE branch")
            evidence_sha = None
        elif self.operator_kind == "CONV2D":
            if (
                not validate_conv_branch_evidence(self.branch_evidence)
                or self.branch != self.branch_evidence.selected_branch
            ):
                _fail(
                    "INVALID_BRANCH",
                    "CONV2D branch/evidence mismatch",
                )
            evidence_sha = self.branch_evidence.evidence_sha256
        else:
            _fail(
                "INVALID_BINDING",
                "operator_kind must be DENSE or CONV2D",
            )
        body = {
            "schema": "act.query_dual_v5_affine_execution.v1",
            "execution_index": execution,
            "layer_id": layer,
            "predecessor_id": predecessor,
            "operator_kind": self.operator_kind,
            "branch": self.branch,
            "branch_evidence_sha256": evidence_sha,
        }
        object.__setattr__(self, "execution_index", execution)
        object.__setattr__(self, "layer_id", layer)
        object.__setattr__(self, "predecessor_id", predecessor)
        object.__setattr__(self, "key_sha256", _json_sha256(body))


def validate_execution_key(value: Any) -> bool:
    try:
        if not isinstance(value, AffineExecutionKey):
            return False
        rebuilt = AffineExecutionKey(
            execution_index=value.execution_index,
            layer_id=value.layer_id,
            predecessor_id=value.predecessor_id,
            operator_kind=value.operator_kind,
            branch=value.branch,
            branch_evidence=value.branch_evidence,
        )
        return hmac.compare_digest(rebuilt.key_sha256, value.key_sha256)
    except (AttributeError, TypeError, ValueError, QueryDualV5AuthorityError):
        return False


def _required_policy_for_key(key: AffineExecutionKey) -> str:
    if not validate_execution_key(key):
        _fail("INVALID_EXPECTATION", "execution key is invalid")
    if key.branch in {BRANCH_DENSE, BRANCH_CONV_DENSE}:
        return POLICY_SCALAR
    if key.branch == BRANCH_CONV_SPARSE:
        return POLICY_COMPONENTWISE
    _fail("INVALID_BRANCH", "execution key has no V5 guard policy")


def _scalar_support_branch_matches(
    operator_kind: Any,
    branch: Any,
) -> bool:
    return bool(
        (operator_kind == "DENSE" and branch == BRANCH_DENSE)
        or (
            operator_kind == "CONV2D"
            and branch == BRANCH_CONV_DENSE
        )
    )


_SUPPORT_CAPABILITY = object()
_SUPPORT_REGISTRY: weakref.WeakValueDictionary[
    str, "FrameLocalAffineSupport"
] = weakref.WeakValueDictionary()
_SUPPORT_LOCK = threading.Lock()


@dataclass(frozen=True)
class FrameLocalAffineSupport:
    """Immutable box-dependent support owned by one exact V5 frame."""

    frame_binding: V5FrameBinding
    owner_nonce_sha256: str
    layer_id: int
    predecessor_id: int
    operator_kind: str
    branch: str
    box_semantics: str
    weight_sha256: str
    geometry_sha256: str
    source_lb_sha256: str
    source_ub_sha256: str
    numeric_platform_sha256: str
    implementation_sha256: str
    maxabs: np.ndarray
    support_s: np.ndarray
    box_mass_hex: str
    receipt: Mapping[str, Any]
    content_sha256: str
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameLocalSupportOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)


def _support_body(
    *,
    owner: _FrameLocalSupportOwner,
    layer_id: int,
    predecessor_id: int,
    operator_kind: str,
    branch: str,
    box_semantics: str,
    weight_sha256: str,
    geometry_sha256: str,
    source_lb_sha256: str,
    source_ub_sha256: str,
    numeric_platform_sha256: str,
    implementation_sha256: str,
    maxabs: np.ndarray,
    support_s: np.ndarray,
    box_mass_hex: str,
    support_nonce_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema": SUPPORT_SCHEMA,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        "authority_scope": "frame_local_support_binding_only",
        "frame_binding_sha256": owner.binding.binding_sha256,
        "owner_nonce_sha256": owner.nonce_sha256,
        "support_nonce_sha256": support_nonce_sha256,
        "layer_id": layer_id,
        "predecessor_id": predecessor_id,
        "operator_kind": operator_kind,
        "branch": branch,
        "box_semantics": box_semantics,
        "weight_sha256": weight_sha256,
        "geometry_sha256": geometry_sha256,
        "source_lb_sha256": source_lb_sha256,
        "source_ub_sha256": source_ub_sha256,
        "numeric_platform_sha256": numeric_platform_sha256,
        "implementation_sha256": implementation_sha256,
        "maxabs_shape": list(maxabs.shape),
        "maxabs_sha256": _array_sha256(maxabs),
        "support_s_shape": list(support_s.shape),
        "support_s_sha256": _array_sha256(support_s),
        "box_mass_hex": box_mass_hex,
        "tau_formula": "up(k*eta/(1-k*u))",
        "nonnegative_dot_upper": "up((nominal+tau)/(1-gamma))",
        "candidate_inputs_are_authoritative": False,
        "proof_authority": False,
    }


def _mint_frame_local_affine_support(
    owner: _FrameLocalSupportOwner,
    *,
    layer_id: int,
    predecessor_id: int,
    operator_kind: str,
    branch: str,
    box_semantics: str,
    weight_sha256: str,
    geometry_sha256: str,
    source_lb_sha256: str,
    source_ub_sha256: str,
    numeric_platform_sha256: str,
    implementation_sha256: str,
    maxabs: Any,
    support_s: Any,
    box_mass: float,
) -> FrameLocalAffineSupport:
    """Mint support for a future trusted replay V5 implementation."""

    if not _verify_owner(owner):
        _fail("INVALID_OWNER", "support owner is not live")
    deadline = _deadline_from_hex(owner.binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before support freeze")
    layer = _exact_nonnegative_int(layer_id, name="layer_id")
    predecessor = _exact_nonnegative_int(
        predecessor_id, name="predecessor_id"
    )
    if not _scalar_support_branch_matches(operator_kind, branch):
        _fail(
            "INVALID_BINDING",
            "scalar support requires the matching DENSE/CONV2D_DENSE branch",
        )
    if box_semantics not in {
        "preactivation",
        "output",
        "relu_postactivation_from_preactivation_box_v1",
    }:
        _fail("INVALID_BINDING", "unsupported predecessor box semantics")
    for name, value in (
        ("weight_sha256", weight_sha256),
        ("geometry_sha256", geometry_sha256),
        ("source_lb_sha256", source_lb_sha256),
        ("source_ub_sha256", source_ub_sha256),
        ("numeric_platform_sha256", numeric_platform_sha256),
        ("implementation_sha256", implementation_sha256),
    ):
        _require_sha256(value, name=name)
    maxabs_frozen = _immutable_f64(
        maxabs, name="maxabs", nonnegative=True
    )
    support_frozen = _immutable_f64(
        support_s, name="support_s", nonnegative=True
    )
    numeric_mass = float(box_mass)
    if not math.isfinite(numeric_mass) or numeric_mass < 0.0:
        _fail("INVALID_NUMERIC", "box_mass must be finite and nonnegative")
    mass_hex = numeric_mass.hex()
    nonce = secrets.token_hex(32)
    nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
    body = _support_body(
        owner=owner,
        layer_id=layer,
        predecessor_id=predecessor,
        operator_kind=operator_kind,
        branch=branch,
        box_semantics=box_semantics,
        weight_sha256=weight_sha256,
        geometry_sha256=geometry_sha256,
        source_lb_sha256=source_lb_sha256,
        source_ub_sha256=source_ub_sha256,
        numeric_platform_sha256=numeric_platform_sha256,
        implementation_sha256=implementation_sha256,
        maxabs=maxabs_frozen,
        support_s=support_frozen,
        box_mass_hex=mass_hex,
        support_nonce_sha256=nonce_sha,
    )
    content_sha = _json_sha256(body)
    receipt_body = dict(body)
    receipt_body["content_sha256"] = content_sha
    receipt_body["receipt_sha256"] = _json_sha256(receipt_body)
    support = FrameLocalAffineSupport(
        frame_binding=owner.binding,
        owner_nonce_sha256=owner.nonce_sha256,
        layer_id=layer,
        predecessor_id=predecessor,
        operator_kind=operator_kind,
        branch=branch,
        box_semantics=box_semantics,
        weight_sha256=weight_sha256,
        geometry_sha256=geometry_sha256,
        source_lb_sha256=source_lb_sha256,
        source_ub_sha256=source_ub_sha256,
        numeric_platform_sha256=numeric_platform_sha256,
        implementation_sha256=implementation_sha256,
        maxabs=maxabs_frozen,
        support_s=support_frozen,
        box_mass_hex=mass_hex,
        receipt=_deep_freeze(receipt_body),
        content_sha256=content_sha,
        _nonce=nonce,
        _owner_nonce=owner._nonce,
        _owner=owner,
        _capability=_SUPPORT_CAPABILITY,
    )
    with _SUPPORT_LOCK:
        _SUPPORT_REGISTRY[nonce] = support
    _check_deadline(deadline, where="after support freeze")
    return support


def validate_frame_local_affine_support(value: Any) -> bool:
    try:
        if (
            not isinstance(value, FrameLocalAffineSupport)
            or value._capability is not _SUPPORT_CAPABILITY
            or not validate_frame_binding(value.frame_binding)
            or not _scalar_support_branch_matches(
                value.operator_kind, value.branch
            )
            or value._owner._nonce != value._owner_nonce
            or value.owner_nonce_sha256
            != hashlib.sha256(value._owner_nonce.encode("ascii")).hexdigest()
        ):
            return False
        with _OWNER_LOCK:
            owner = _OWNER_REGISTRY.get(value._owner_nonce)
        with _SUPPORT_LOCK:
            live = _SUPPORT_REGISTRY.get(value._nonce)
        if (
            owner is None
            or value._owner is not owner
            or owner.binding.binding_sha256
            != value.frame_binding.binding_sha256
            or live is not value
        ):
            return False
        maxabs = np.asarray(value.maxabs)
        support_s = np.asarray(value.support_s)
        if (
            maxabs.dtype != np.float64
            or support_s.dtype != np.float64
            or maxabs.flags.writeable
            or support_s.flags.writeable
            or not np.all(np.isfinite(maxabs))
            or not np.all(np.isfinite(support_s))
            or np.any(maxabs < 0.0)
            or np.any(support_s < 0.0)
        ):
            return False
        body = _support_body(
            owner=owner,
            layer_id=value.layer_id,
            predecessor_id=value.predecessor_id,
            operator_kind=value.operator_kind,
            branch=value.branch,
            box_semantics=value.box_semantics,
            weight_sha256=value.weight_sha256,
            geometry_sha256=value.geometry_sha256,
            source_lb_sha256=value.source_lb_sha256,
            source_ub_sha256=value.source_ub_sha256,
            numeric_platform_sha256=value.numeric_platform_sha256,
            implementation_sha256=value.implementation_sha256,
            maxabs=maxabs,
            support_s=support_s,
            box_mass_hex=_finite_hex(
                value.box_mass_hex,
                name="box_mass_hex",
                nonnegative=True,
            ),
            support_nonce_sha256=hashlib.sha256(
                value._nonce.encode("ascii")
            ).hexdigest(),
        )
        content_sha = _json_sha256(body)
        receipt = dict(value.receipt)
        claimed_receipt = str(receipt.pop("receipt_sha256"))
        return bool(
            hmac.compare_digest(content_sha, value.content_sha256)
            and receipt.get("content_sha256") == content_sha
            and all(
                _canonical_value(receipt.get(key))
                == _canonical_value(expected)
                for key, expected in body.items()
            )
            and set(receipt) == set(body) | {"content_sha256"}
            and hmac.compare_digest(_json_sha256(receipt), claimed_receipt)
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        QueryDualV5AuthorityError,
    ):
        return False


@dataclass(frozen=True)
class GuardExecutionExpectation:
    """Frozen policy and support expected for one affine execution."""

    key: AffineExecutionKey
    expected_policy: str
    expected_support_sha256: Optional[str]
    expectation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not validate_execution_key(self.key):
            _fail("INVALID_EXPECTATION", "execution key is invalid")
        required_policy = _required_policy_for_key(self.key)
        if self.expected_policy != required_policy:
            _fail(
                "POLICY_MISMATCH",
                "guard policy is incompatible with the frozen affine branch",
            )
        if self.expected_policy == POLICY_SCALAR:
            support_sha = _require_sha256(
                self.expected_support_sha256,
                name="expected_support_sha256",
            )
        elif self.expected_policy == POLICY_COMPONENTWISE:
            if self.expected_support_sha256 is not None:
                _fail(
                    "INVALID_EXPECTATION",
                    "componentwise execution cannot claim scalar support",
                )
            support_sha = None
        else:
            _fail("INVALID_EXPECTATION", "unknown affine guard policy")
        body = {
            "schema": "act.query_dual_v5_guard_expectation.v1",
            "key_sha256": self.key.key_sha256,
            "expected_policy": self.expected_policy,
            "expected_support_sha256": support_sha,
        }
        object.__setattr__(
            self, "expected_support_sha256", support_sha
        )
        object.__setattr__(
            self, "expectation_sha256", _json_sha256(body)
        )


_RESULT_CAPABILITY = object()
_RESULT_REGISTRY: weakref.WeakValueDictionary[
    str, Union["ScalarGuardedAffineResult", "ComponentwiseAffineResult"]
] = weakref.WeakValueDictionary()
_RESULT_LOCK = threading.Lock()


@dataclass(frozen=True)
class ScalarGuardedAffineResult:
    """Nominal coefficient plus one scalar guard; never a radius."""

    key: AffineExecutionKey
    frame_binding_sha256: str
    owner_nonce_sha256: str
    support: FrameLocalAffineSupport
    nominal: np.ndarray
    scalar_guard: np.ndarray
    trace_sha256: str
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameLocalSupportOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class ComponentwiseAffineResult:
    """Nominal coefficient plus one componentwise radius; never scalar."""

    key: AffineExecutionKey
    frame_binding_sha256: str
    owner_nonce_sha256: str
    nominal: np.ndarray
    radius: np.ndarray
    trace_sha256: str
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameLocalSupportOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)


AffineGuardResult = Union[
    ScalarGuardedAffineResult, ComponentwiseAffineResult
]


def _result_body(result: AffineGuardResult) -> Dict[str, Any]:
    common = {
        "schema": "act.query_dual_v5_affine_guard_result.v1",
        "numeric_protocol": NUMERIC_PROTOCOL,
        "key_sha256": result.key.key_sha256,
        "frame_binding_sha256": result.frame_binding_sha256,
        "owner_nonce_sha256": result.owner_nonce_sha256,
        "result_nonce_sha256": hashlib.sha256(
            result._nonce.encode("ascii")
        ).hexdigest(),
        "nominal_shape": list(result.nominal.shape),
        "nominal_sha256": _array_sha256(result.nominal),
    }
    if isinstance(result, ScalarGuardedAffineResult):
        common.update(
            {
                "policy": POLICY_SCALAR,
                "support_content_sha256": result.support.content_sha256,
                "scalar_guard_shape": list(result.scalar_guard.shape),
                "scalar_guard_sha256": _array_sha256(
                    result.scalar_guard
                ),
                "scalar_guard_applied_count": 1,
                "componentwise_radius_applied_count": 0,
            }
        )
    elif isinstance(result, ComponentwiseAffineResult):
        common.update(
            {
                "policy": POLICY_COMPONENTWISE,
                "support_content_sha256": None,
                "radius_shape": list(result.radius.shape),
                "radius_sha256": _array_sha256(result.radius),
                "scalar_guard_applied_count": 0,
                "componentwise_radius_applied_count": 1,
            }
        )
    else:
        _fail("INVALID_RESULT", "unknown affine guard result type")
    return common


def _mint_scalar_guarded_result(
    owner: _FrameLocalSupportOwner,
    key: AffineExecutionKey,
    support: FrameLocalAffineSupport,
    *,
    nominal: Any,
    scalar_guard: Any,
) -> ScalarGuardedAffineResult:
    if (
        not _verify_owner(owner)
        or not validate_execution_key(key)
        or _required_policy_for_key(key) != POLICY_SCALAR
        or not validate_frame_local_affine_support(support)
        or support._owner_nonce != owner._nonce
        or support.layer_id != key.layer_id
        or support.predecessor_id != key.predecessor_id
        or support.operator_kind != key.operator_kind
        or support.branch != key.branch
    ):
        _fail("INVALID_RESULT", "scalar result/support binding mismatch")
    deadline = _deadline_from_hex(owner.binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before scalar result mint")
    nominal_frozen = _immutable_f64(
        nominal, name="nominal", nonnegative=False
    )
    if nominal_frozen.ndim != 2:
        _fail("INVALID_RESULT", "nominal coefficient must be rank two")
    guard_frozen = _immutable_f64(
        scalar_guard, name="scalar_guard", nonnegative=True
    ).reshape(-1)
    if guard_frozen.size != nominal_frozen.shape[0]:
        _fail(
            "INVALID_RESULT",
            "scalar guard count must equal query-row count",
        )
    nonce = secrets.token_hex(32)
    result = ScalarGuardedAffineResult(
        key=key,
        frame_binding_sha256=owner.binding.binding_sha256,
        owner_nonce_sha256=owner.nonce_sha256,
        support=support,
        nominal=nominal_frozen,
        scalar_guard=guard_frozen,
        trace_sha256="",
        _nonce=nonce,
        _owner_nonce=owner._nonce,
        _owner=owner,
        _capability=_RESULT_CAPABILITY,
    )
    object.__setattr__(result, "trace_sha256", _json_sha256(_result_body(result)))
    with _RESULT_LOCK:
        _RESULT_REGISTRY[nonce] = result
    _check_deadline(deadline, where="after scalar result mint")
    return result


def _mint_componentwise_result(
    owner: _FrameLocalSupportOwner,
    key: AffineExecutionKey,
    *,
    nominal: Any,
    radius: Any,
) -> ComponentwiseAffineResult:
    if (
        not _verify_owner(owner)
        or not validate_execution_key(key)
        or _required_policy_for_key(key) != POLICY_COMPONENTWISE
    ):
        _fail("INVALID_RESULT", "componentwise result owner/key mismatch")
    deadline = _deadline_from_hex(owner.binding.deadline_monotonic_hex)
    _check_deadline(deadline, where="before componentwise result mint")
    nominal_frozen = _immutable_f64(
        nominal, name="nominal", nonnegative=False
    )
    radius_frozen = _immutable_f64(
        radius, name="radius", nonnegative=True
    )
    if nominal_frozen.ndim != 2 or radius_frozen.shape != nominal_frozen.shape:
        _fail(
            "INVALID_RESULT",
            "componentwise radius must match a rank-two nominal coefficient",
        )
    nonce = secrets.token_hex(32)
    result = ComponentwiseAffineResult(
        key=key,
        frame_binding_sha256=owner.binding.binding_sha256,
        owner_nonce_sha256=owner.nonce_sha256,
        nominal=nominal_frozen,
        radius=radius_frozen,
        trace_sha256="",
        _nonce=nonce,
        _owner_nonce=owner._nonce,
        _owner=owner,
        _capability=_RESULT_CAPABILITY,
    )
    object.__setattr__(result, "trace_sha256", _json_sha256(_result_body(result)))
    with _RESULT_LOCK:
        _RESULT_REGISTRY[nonce] = result
    _check_deadline(deadline, where="after componentwise result mint")
    return result


def validate_affine_guard_result(value: Any) -> bool:
    try:
        if (
            not isinstance(
                value,
                (ScalarGuardedAffineResult, ComponentwiseAffineResult),
            )
            or value._capability is not _RESULT_CAPABILITY
            or not validate_execution_key(value.key)
            or value._owner._nonce != value._owner_nonce
            or not _is_sha256(value.frame_binding_sha256)
            or not _is_sha256(value.owner_nonce_sha256)
            or value.owner_nonce_sha256
            != hashlib.sha256(value._owner_nonce.encode("ascii")).hexdigest()
        ):
            return False
        with _OWNER_LOCK:
            owner = _OWNER_REGISTRY.get(value._owner_nonce)
        with _RESULT_LOCK:
            live = _RESULT_REGISTRY.get(value._nonce)
        if (
            owner is None
            or value._owner is not owner
            or live is not value
            or value.frame_binding_sha256
            != owner.binding.binding_sha256
        ):
            return False
        nominal = np.asarray(value.nominal)
        if (
            nominal.dtype != np.float64
            or nominal.ndim != 2
            or nominal.flags.writeable
            or not np.all(np.isfinite(nominal))
        ):
            return False
        if isinstance(value, ScalarGuardedAffineResult):
            guard = np.asarray(value.scalar_guard)
            if (
                not validate_frame_local_affine_support(value.support)
                or value.support._owner_nonce != value._owner_nonce
                or value.support.layer_id != value.key.layer_id
                or value.support.predecessor_id
                != value.key.predecessor_id
                or value.support.operator_kind != value.key.operator_kind
                or value.support.branch != value.key.branch
                or guard.dtype != np.float64
                or guard.ndim != 1
                or guard.size != nominal.shape[0]
                or guard.flags.writeable
                or not np.all(np.isfinite(guard))
                or np.any(guard < 0.0)
            ):
                return False
        else:
            radius = np.asarray(value.radius)
            if (
                radius.dtype != np.float64
                or radius.shape != nominal.shape
                or radius.flags.writeable
                or not np.all(np.isfinite(radius))
                or np.any(radius < 0.0)
            ):
                return False
        return hmac.compare_digest(
            _json_sha256(_result_body(value)), value.trace_sha256
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        QueryDualV5AuthorityError,
    ):
        return False


_LEDGER_CAPABILITY = object()
_CERTIFICATE_CAPABILITY = object()
_CERTIFICATE_REGISTRY: weakref.WeakValueDictionary[
    str, "GuardLedgerCertificate"
] = weakref.WeakValueDictionary()
_CERTIFICATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class GuardLedgerCertificate:
    """Process-local certificate of guard accounting, not proof authority."""

    frame_binding: V5FrameBinding
    owner_nonce_sha256: str
    expectations: Tuple[GuardExecutionExpectation, ...]
    results: Tuple[AffineGuardResult, ...]
    receipt: Mapping[str, Any]
    content_sha256: str
    proof_authority: bool
    _nonce: str = field(repr=False, compare=False)
    _owner_nonce: str = field(repr=False, compare=False)
    _owner: _FrameLocalSupportOwner = field(repr=False, compare=False)
    _capability: Any = field(repr=False, compare=False)


def _expectation_body(value: GuardExecutionExpectation) -> Dict[str, Any]:
    return {
        "key_sha256": value.key.key_sha256,
        "expected_policy": value.expected_policy,
        "expected_support_sha256": value.expected_support_sha256,
        "expectation_sha256": value.expectation_sha256,
    }


class _GuardLedger:
    """One-shot, frame-owned exact coverage ledger."""

    def __init__(
        self,
        *,
        authority: Any,
        owner: _FrameLocalSupportOwner,
        expectations: Sequence[GuardExecutionExpectation],
    ):
        if authority is not _LEDGER_CAPABILITY or not _verify_owner(owner):
            _fail("INVALID_LEDGER", "ledger requires a live frame owner")
        if not isinstance(expectations, Sequence) or isinstance(
            expectations, (str, bytes)
        ):
            _fail("INVALID_EXPECTATION", "expectations must be a sequence")
        frozen = tuple(expectations)
        if not frozen:
            _fail("INVALID_EXPECTATION", "expectations cannot be empty")
        by_key: Dict[str, GuardExecutionExpectation] = {}
        indices = []
        for expectation in frozen:
            if not isinstance(expectation, GuardExecutionExpectation):
                _fail("INVALID_EXPECTATION", "invalid expectation type")
            rebuilt = GuardExecutionExpectation(
                key=expectation.key,
                expected_policy=expectation.expected_policy,
                expected_support_sha256=(
                    expectation.expected_support_sha256
                ),
            )
            if not hmac.compare_digest(
                rebuilt.expectation_sha256,
                expectation.expectation_sha256,
            ):
                _fail("INVALID_EXPECTATION", "expectation seal changed")
            key_sha = expectation.key.key_sha256
            if key_sha in by_key:
                _fail("INVALID_EXPECTATION", "duplicate execution expectation")
            by_key[key_sha] = expectation
            indices.append(expectation.key.execution_index)
        if indices != list(range(len(frozen))):
            _fail(
                "INVALID_EXPECTATION",
                "execution indices must be contiguous and zero-based",
            )
        self._owner = owner
        self._expectations = frozen
        self._by_key = MappingProxyType(by_key)
        self._recorded: Dict[str, AffineGuardResult] = {}
        self._deadline = _deadline_from_hex(
            owner.binding.deadline_monotonic_hex
        )
        self._deadline_hex = owner.binding.deadline_monotonic_hex
        self._lock = threading.Lock()
        self._closed = False
        self._failed = False
        self._capability = _LEDGER_CAPABILITY

    def _invalidate(self) -> None:
        self._failed = True
        self._closed = True
        self._recorded.clear()

    def _check(self, *, where: str) -> None:
        if (
            self._closed
            or self._failed
            or self._capability is not _LEDGER_CAPABILITY
            or not _verify_owner(self._owner)
            or self._deadline_hex
            != self._owner.binding.deadline_monotonic_hex
        ):
            _fail("INVALID_LEDGER", "ledger is closed or lost its owner")
        try:
            _check_deadline(self._deadline, where=where)
        except QueryDualV5AuthorityError:
            self._invalidate()
            raise

    def record(self, result: AffineGuardResult) -> None:
        if not self._lock.acquire(blocking=False):
            self._invalidate()
            _fail("CONCURRENT_LEDGER", "concurrent guard recording is forbidden")
        try:
            self._check(where="before guard record")
            if (
                not validate_affine_guard_result(result)
                or result._owner_nonce != self._owner._nonce
            ):
                self._invalidate()
                _fail("INVALID_RESULT", "guard result is not frame-owned")
            key_sha = result.key.key_sha256
            expectation = self._by_key.get(key_sha)
            if expectation is None:
                self._invalidate()
                _fail(
                    "UNEXPECTED_EXECUTION",
                    "guard result has no frozen execution expectation",
                )
            if key_sha in self._recorded:
                self._invalidate()
                _fail(
                    "DOUBLE_CHARGE",
                    "one affine execution received more than one guard",
                )
            if isinstance(result, ScalarGuardedAffineResult):
                actual_policy = POLICY_SCALAR
                support_sha = result.support.content_sha256
            else:
                actual_policy = POLICY_COMPONENTWISE
                support_sha = None
            if (
                actual_policy != expectation.expected_policy
                or support_sha != expectation.expected_support_sha256
            ):
                self._invalidate()
                _fail(
                    "POLICY_MISMATCH",
                    "guard result differs from its frozen policy/support",
                )
            self._recorded[key_sha] = result
            self._check(where="after guard record")
        except Exception:
            if not self._closed:
                self._invalidate()
            raise
        finally:
            self._lock.release()

    def commit(self) -> GuardLedgerCertificate:
        if not self._lock.acquire(blocking=False):
            self._invalidate()
            _fail("CONCURRENT_LEDGER", "concurrent ledger commit is forbidden")
        try:
            self._check(where="before guard ledger commit")
            missing = [
                expectation.key.execution_index
                for expectation in self._expectations
                if expectation.key.key_sha256 not in self._recorded
            ]
            if missing:
                self._invalidate()
                _fail(
                    "MISSING_GUARD",
                    f"affine executions lack a guard: {missing}",
                )
            ordered = tuple(
                self._recorded[expectation.key.key_sha256]
                for expectation in self._expectations
            )
            scalar_count = sum(
                isinstance(result, ScalarGuardedAffineResult)
                for result in ordered
            )
            componentwise_count = len(ordered) - scalar_count
            nonce = secrets.token_hex(32)
            nonce_sha = hashlib.sha256(nonce.encode("ascii")).hexdigest()
            body = {
                "schema": LEDGER_SCHEMA,
                "numeric_protocol": NUMERIC_PROTOCOL,
                "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
                "authority_scope": "guard_accounting_only",
                "proof_authority": False,
                "coverage_complete": True,
                "guard_exclusivity": (
                    "exactly_one_of_scalar_or_componentwise"
                ),
                "frame_binding_sha256": (
                    self._owner.binding.binding_sha256
                ),
                "owner_nonce_sha256": self._owner.nonce_sha256,
                "ledger_nonce_sha256": nonce_sha,
                "deadline_monotonic_hex": self._deadline_hex,
                "expectations": [
                    _expectation_body(value)
                    for value in self._expectations
                ],
                "expectations_sha256": _json_sha256(
                    [
                        _expectation_body(value)
                        for value in self._expectations
                    ]
                ),
                "result_trace_sha256": [
                    result.trace_sha256 for result in ordered
                ],
                "execution_count": len(ordered),
                "scalar_guard_count": int(scalar_count),
                "componentwise_radius_count": int(componentwise_count),
            }
            content_sha = _json_sha256(body)
            receipt_body = dict(body)
            receipt_body["content_sha256"] = content_sha
            receipt_body["receipt_sha256"] = _json_sha256(receipt_body)
            certificate = GuardLedgerCertificate(
                frame_binding=self._owner.binding,
                owner_nonce_sha256=self._owner.nonce_sha256,
                expectations=self._expectations,
                results=ordered,
                receipt=_deep_freeze(receipt_body),
                content_sha256=content_sha,
                proof_authority=False,
                _nonce=nonce,
                _owner_nonce=self._owner._nonce,
                _owner=self._owner,
                _capability=_CERTIFICATE_CAPABILITY,
            )
            with _CERTIFICATE_LOCK:
                _CERTIFICATE_REGISTRY[nonce] = certificate
            self._closed = True
            self._recorded.clear()
            _check_deadline(self._deadline, where="after guard ledger commit")
            return certificate
        except Exception:
            if not self._closed:
                self._invalidate()
            raise
        finally:
            self._lock.release()

    def abort(self) -> None:
        with self._lock:
            self._invalidate()


def _mint_guard_ledger(
    owner: _FrameLocalSupportOwner,
    expectations: Sequence[GuardExecutionExpectation],
) -> _GuardLedger:
    return _GuardLedger(
        authority=_LEDGER_CAPABILITY,
        owner=owner,
        expectations=expectations,
    )


def validate_guard_ledger_certificate(value: Any) -> bool:
    try:
        if (
            not isinstance(value, GuardLedgerCertificate)
            or value._capability is not _CERTIFICATE_CAPABILITY
            or value.proof_authority is not False
            or not validate_frame_binding(value.frame_binding)
            or value._owner._nonce != value._owner_nonce
            or value.owner_nonce_sha256
            != hashlib.sha256(value._owner_nonce.encode("ascii")).hexdigest()
        ):
            return False
        with _OWNER_LOCK:
            owner = _OWNER_REGISTRY.get(value._owner_nonce)
        with _CERTIFICATE_LOCK:
            live = _CERTIFICATE_REGISTRY.get(value._nonce)
        if (
            owner is None
            or value._owner is not owner
            or live is not value
            or owner.binding.binding_sha256
            != value.frame_binding.binding_sha256
            or value.owner_nonce_sha256 != owner.nonce_sha256
            or len(value.expectations) != len(value.results)
        ):
            return False
        expectation_bodies = []
        scalar_count = 0
        seen = set()
        for expectation, result in zip(value.expectations, value.results):
            rebuilt = GuardExecutionExpectation(
                key=expectation.key,
                expected_policy=expectation.expected_policy,
                expected_support_sha256=(
                    expectation.expected_support_sha256
                ),
            )
            if (
                rebuilt.expectation_sha256
                != expectation.expectation_sha256
                or expectation.key.key_sha256 in seen
                or not validate_affine_guard_result(result)
                or result._owner_nonce != value._owner_nonce
                or result.key.key_sha256 != expectation.key.key_sha256
            ):
                return False
            seen.add(expectation.key.key_sha256)
            if isinstance(result, ScalarGuardedAffineResult):
                actual_policy = POLICY_SCALAR
                support_sha = result.support.content_sha256
                scalar_count += 1
            else:
                actual_policy = POLICY_COMPONENTWISE
                support_sha = None
            if (
                actual_policy != expectation.expected_policy
                or support_sha != expectation.expected_support_sha256
            ):
                return False
            expectation_bodies.append(_expectation_body(expectation))
        body = {
            "schema": LEDGER_SCHEMA,
            "numeric_protocol": NUMERIC_PROTOCOL,
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
            "authority_scope": "guard_accounting_only",
            "proof_authority": False,
            "coverage_complete": True,
            "guard_exclusivity": "exactly_one_of_scalar_or_componentwise",
            "frame_binding_sha256": value.frame_binding.binding_sha256,
            "owner_nonce_sha256": owner.nonce_sha256,
            "ledger_nonce_sha256": hashlib.sha256(
                value._nonce.encode("ascii")
            ).hexdigest(),
            "deadline_monotonic_hex": value.frame_binding.deadline_monotonic_hex,
            "expectations": expectation_bodies,
            "expectations_sha256": _json_sha256(expectation_bodies),
            "result_trace_sha256": [
                result.trace_sha256 for result in value.results
            ],
            "execution_count": len(value.results),
            "scalar_guard_count": int(scalar_count),
            "componentwise_radius_count": int(
                len(value.results) - scalar_count
            ),
        }
        content_sha = _json_sha256(body)
        receipt = dict(value.receipt)
        claimed_receipt = str(receipt.pop("receipt_sha256"))
        return bool(
            hmac.compare_digest(content_sha, value.content_sha256)
            and receipt.get("content_sha256") == content_sha
            and all(
                _canonical_value(receipt.get(key))
                == _canonical_value(expected)
                for key, expected in body.items()
            )
            and set(receipt) == set(body) | {"content_sha256"}
            and hmac.compare_digest(_json_sha256(receipt), claimed_receipt)
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        QueryDualV5AuthorityError,
    ):
        return False


__all__ = [
    "AUTHORITY_SCHEMA",
    "BRANCH_CONV_DENSE",
    "BRANCH_CONV_SPARSE",
    "BRANCH_DENSE",
    "ComponentwiseAffineResult",
    "ConvBranchEvidence",
    "FrameLocalAffineSupport",
    "GuardExecutionExpectation",
    "GuardLedgerCertificate",
    "LEDGER_SCHEMA",
    "NUMERIC_PROTOCOL",
    "POLICY_COMPONENTWISE",
    "POLICY_SCALAR",
    "PROTOCOL_MANIFEST",
    "PROTOCOL_MANIFEST_SHA256",
    "QueryDualV5AuthorityError",
    "SUPPORT_SCHEMA",
    "ScalarGuardedAffineResult",
    "V5FrameBinding",
    "AffineExecutionKey",
    "validate_affine_guard_result",
    "validate_conv_branch_evidence",
    "validate_execution_key",
    "validate_frame_binding",
    "validate_frame_local_affine_support",
    "validate_guard_ledger_certificate",
]
