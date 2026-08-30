#!/usr/bin/env python3
# ===- query_dual_pipeline.py - transactional proof authority -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Transactional query-dual tightening rooted in an independent box proof.

The ordinary ACT ``before``/``after`` facts are intentionally absent from
this API.  The only initial authority is
:func:`query_dual_box_certifier.certify_query_dual_boxes`.  GPU DualSolver
queries merely propose frozen-alpha candidates; every consumed number is
reproved by :func:`query_dual_replay.replay_query_lower_bounds`.

Each target-ReLU stage is evaluated against one immutable parent snapshot and
committed atomically.  Any malformed candidate, replay failure, hash mismatch,
deadline crossing, inconsistent intersection, or live-network mutation aborts
the whole transaction without returning a partially authoritative object.

Receipts are audit trails, not signatures.  A completed bundle is additionally
registered under a process-local, object-identity capability.  Consequently a
JSON reconstruction, deepcopy, or freshly self-hashed forged object cannot
become proof authority.  Downstream consumers must call
``validate_verified_query_dual_feedback`` on the exact live object.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import os
import secrets
import threading
import time
from typing import Any, Callable, Dict, Mapping, NoReturn, Optional, Sequence, Tuple
import weakref

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    QueryDualBoxCertificate,
    QueryDualBoxError,
    QueryDualBoxTimeout,
    certify_query_dual_boxes,
    verify_query_dual_box_certificate,
)
from act.back_end.hybridz_tf.query_dual_candidates import (
    QueryDescriptor,
    QueryDualCandidates,
    generate_query_dual_candidates,
    query_dual_stored_alpha_sha256,
    validate_query_dual_candidates,
    verify_query_dual_candidates_receipt,
)
from act.back_end.hybridz_tf.query_dual_replay import (
    QueryDualReplayError,
    QueryDualReplayResult,
    QueryDualReplayTimeout,
    replay_query_lower_bounds,
    validate_query_dual_replay_result,
)
from act.util.device_manager import get_default_device, get_default_dtype


_SCHEMA = "act.verified_query_dual_feedback.v2"
_STAGE_SCHEMA = "act.verified_query_dual_stage.v2"
_PROPERTY_SCHEMA = "act.verified_query_dual_property.v2"
_CANDIDATE_SCHEMA = "act.query_dual_candidates.v2"
_CANDIDATE_PROTOCOL = "descriptor_only_v2"
_CANDIDATE_NON_AUTHORITATIVE_AUDIT_FIELDS = [
    "lr_alpha",
    "lr_decay",
    "solver",
    "elapsed_seconds",
    "timings",
]
_PIPELINE_NON_AUTHORITATIVE_AUDIT_FIELDS = [
    "candidate_generator",
    "candidate_solver_factory",
    "dual_solver_default_device",
    "dual_solver_default_dtype",
    "candidate_cuda_device_name",
]
_LIVE_LOCK = threading.Lock()
_LIVE_AUTHORITIES: Dict[
    int,
    Tuple[
        "weakref.ReferenceType[VerifiedQueryDualFeedback]",
        str,
        str,
    ],
] = {}
_TRUSTED_CERTIFIER = certify_query_dual_boxes
_TRUSTED_REPLAYER = replay_query_lower_bounds


class QueryDualPipelineError(RuntimeError):
    """Fail-closed transaction error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


class QueryDualPipelineTimeout(QueryDualPipelineError):
    """The shared transaction deadline expired before an atomic result."""

    def __init__(self, message: str = "query-dual transaction deadline expired"):
        super().__init__("DEADLINE_EXPIRED", message)


@dataclass(frozen=True)
class QueryDualAuthorityBlock:
    """One candidate descriptor bridged to one independent replay."""

    block_id: int
    query_kind: str
    start_lid: Optional[int]
    target_relu_lid: Optional[int]
    row_ids: Tuple[int, ...]
    objective_sha256: str
    candidate_alpha_sha256: str
    replay_query_sha256: str
    replay_alpha_sha256: str
    replay_bounds_sha256: str
    replay_net_sha256: str
    alpha_bridge_sha256: str
    lower_bounds: np.ndarray
    replay_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(
            np.asarray(self.lower_bounds, dtype=np.float64).reshape(-1)
        ).copy()
        values.setflags(write=False)
        object.__setattr__(self, "lower_bounds", values)
        object.__setattr__(
            self, "row_ids", tuple(int(value) for value in self.row_ids)
        )
        object.__setattr__(
            self, "replay_receipt", copy.deepcopy(dict(self.replay_receipt))
        )


@dataclass(frozen=True)
class QueryDualTargetStage:
    """Atomic authority record for one target ReLU."""

    stage_index: int
    target_relu_lid: int
    predecessor_lid: int
    predecessor_kind: str
    parent_boxes_sha256: str
    result_boxes_sha256: str
    candidate_bounds_sha256: str
    candidate_receipt: Mapping[str, Any]
    blocks: Tuple[QueryDualAuthorityBlock, ...]
    target_lower: np.ndarray
    target_upper: np.ndarray
    strict_improvements: int
    status: str
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        lower = np.ascontiguousarray(
            np.asarray(self.target_lower, dtype=np.float64).reshape(-1)
        ).copy()
        upper = np.ascontiguousarray(
            np.asarray(self.target_upper, dtype=np.float64).reshape(-1)
        ).copy()
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "target_lower", lower)
        object.__setattr__(self, "target_upper", upper)
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(
            self, "candidate_receipt", copy.deepcopy(dict(self.candidate_receipt))
        )
        object.__setattr__(self, "receipt", copy.deepcopy(dict(self.receipt)))


@dataclass(frozen=True)
class QueryDualPropertyStage:
    """Final ``-C`` replay with ``+threshold`` incorporated as query bias."""

    parent_boxes_sha256: str
    candidate_bounds_sha256: str
    candidate_receipt: Mapping[str, Any]
    blocks: Tuple[QueryDualAuthorityBlock, ...]
    property_upper: np.ndarray
    property_spec_sha256: str
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        upper = np.ascontiguousarray(
            np.asarray(self.property_upper, dtype=np.float64).reshape(-1)
        ).copy()
        upper.setflags(write=False)
        object.__setattr__(self, "property_upper", upper)
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(
            self, "candidate_receipt", copy.deepcopy(dict(self.candidate_receipt))
        )
        object.__setattr__(self, "receipt", copy.deepcopy(dict(self.receipt)))


@dataclass(frozen=True)
class VerifiedQueryDualFeedback:
    """Live, process-local authority bundle consumed by Operator-HZ."""

    root_certificate: QueryDualBoxCertificate
    certified_bounds: Mapping[int, Bounds]
    target_relu_ids: Tuple[int, ...]
    stages: Tuple[QueryDualTargetStage, ...]
    property_stage: QueryDualPropertyStage
    property_upper: np.ndarray
    receipt: Mapping[str, Any]
    provenance_nonce: str
    proof_authority: bool = True

    def __post_init__(self) -> None:
        if self.proof_authority is not True:
            raise ValueError("completed query-dual feedback must be authoritative")
        frozen_bounds = _clone_bounds(self.certified_bounds)
        upper = np.ascontiguousarray(
            np.asarray(self.property_upper, dtype=np.float64).reshape(-1)
        ).copy()
        upper.setflags(write=False)
        object.__setattr__(self, "certified_bounds", frozen_bounds)
        object.__setattr__(
            self,
            "target_relu_ids",
            tuple(int(value) for value in self.target_relu_ids),
        )
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "property_upper", upper)
        object.__setattr__(self, "receipt", copy.deepcopy(dict(self.receipt)))


def _fail(code: str, message: str) -> NoReturn:
    raise QueryDualPipelineError(code, message)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _receipt(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = _json_sha256(result)
    return result


def _verify_receipt(value: Mapping[str, Any], schema: str) -> bool:
    try:
        body = copy.deepcopy(dict(value))
        claimed = str(body.pop("receipt_sha256"))
        return (
            body.get("schema") == schema
            and hmac.compare_digest(_json_sha256(body), claimed)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _array_digest(value: Any) -> str:
    """Replay/certifier binary64 array digest."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0<f8\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _candidate_array_digest(value: Any) -> str:
    """Candidate module's deliberately distinct binary64 array digest."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _as_numpy_f64(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except Exception as exc:
        raise QueryDualPipelineError("INVALID_NUMERIC", f"{name}: {exc}") from exc
    if not np.all(np.isfinite(array)):
        _fail("NONFINITE", f"{name} contains NaN or infinity")
    return array


def _clone_bounds(source: Mapping[int, Bounds]) -> Dict[int, Bounds]:
    result: Dict[int, Bounds] = {}
    for raw_lid, value in source.items():
        lid = int(raw_lid)
        if lid in result or not isinstance(value, Bounds):
            _fail("INVALID_BOUNDS", f"invalid or duplicate bounds key {raw_lid!r}")
        lb = torch.as_tensor(value.lb).detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous().clone()
        ub = torch.as_tensor(value.ub).detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous().clone()
        if (
            lb.shape != ub.shape
            or lb.dim() < 2
            or int(lb.shape[0]) != 1
            or not bool(torch.isfinite(lb).all())
            or not bool(torch.isfinite(ub).all())
            or bool((lb > ub).any())
        ):
            _fail("INVALID_BOUNDS", f"bounds[{lid}] is not one finite box")
        result[lid] = Bounds(lb=lb, ub=ub)
    if not result:
        _fail("INVALID_BOUNDS", "bounds mapping is empty")
    return result


def _apply_operator_hz_preactivation_frame(
    *,
    net: Any,
    root_bounds: Mapping[int, Bounds],
    frame: Any,
    expected_network_sha256: str,
) -> Tuple[Dict[int, Bounds], Dict[str, Any]]:
    """Intersect one live C5 frame into an independent root snapshot."""

    from act.back_end.hybridz_tf.operator_hz import (
        OperatorHZPreactivationFrame,
        validate_operator_hz_preactivation_frame,
    )

    if not isinstance(frame, OperatorHZPreactivationFrame):
        _fail(
            "INVALID_BOUND_FRAME",
            "operator-HZ preactivation frame has the wrong type",
        )
    if not validate_operator_hz_preactivation_frame(
        frame,
        net=net,
        expected_network_sha256=expected_network_sha256,
        require_live_provenance=True,
    ):
        _fail(
            "INVALID_BOUND_FRAME",
            "operator-HZ preactivation frame failed live validation",
        )
    by_id, _preds = _layer_maps(net)
    current = _clone_bounds(root_bounds)
    strict_lower = 0
    strict_upper = 0
    layer_ids = []
    for raw_lid in sorted(frame.bounds):
        lid = int(raw_lid)
        if lid not in by_id or _kind(by_id[lid]) != "RELU":
            _fail(
                "INVALID_BOUND_FRAME",
                f"frame layer {lid} is not a live ReLU",
            )
        if lid not in current:
            _fail(
                "INVALID_BOUND_FRAME",
                f"independent root omits frame ReLU {lid}",
            )
        root_lower, root_upper = _flat_box(current[lid], lid=lid)
        frame_lower = np.ascontiguousarray(
            np.asarray(frame.bounds[lid][0], dtype=np.float64).reshape(-1)
        )
        frame_upper = np.ascontiguousarray(
            np.asarray(frame.bounds[lid][1], dtype=np.float64).reshape(-1)
        )
        if (
            frame_lower.shape != root_lower.shape
            or frame_upper.shape != root_upper.shape
            or not np.all(np.isfinite(frame_lower))
            or not np.all(np.isfinite(frame_upper))
            or np.any(frame_lower > frame_upper)
        ):
            _fail(
                "INVALID_BOUND_FRAME",
                f"frame bounds at ReLU {lid} are malformed",
            )
        lower = np.maximum(root_lower, frame_lower)
        upper = np.minimum(root_upper, frame_upper)
        if np.any(lower > upper):
            _fail(
                "BOUND_FRAME_CONFLICT",
                f"frame/root intersection is empty at ReLU {lid}",
            )
        strict_lower += int(np.count_nonzero(lower > root_lower))
        strict_upper += int(np.count_nonzero(upper < root_upper))
        _replace_box(current, lid, lower, upper)
        layer_ids.append(lid)
    audit = {
        "schema": "query_dual_operator_hz_bound_frame_v1",
        "enabled": True,
        "proof_authority": True,
        "source": "live_operator_hz_preactivation_frame",
        "source_receipt_sha256": frame.receipt["receipt_sha256"],
        "source_bounds_sha256": frame.receipt["bounds_sha256"],
        "source_network_sha256": frame.receipt["network_sha256"],
        "source_layer_ids": layer_ids,
        "strict_lower_rows": int(strict_lower),
        "strict_upper_rows": int(strict_upper),
        "committed_boxes_sha256": _boxes_sha256(net, current),
        "intersection_only": True,
        "target_replay_stages_required": False,
    }
    return current, audit


def _candidate_bounds_on_device(
    source: Mapping[int, Bounds], device: torch.device
) -> Dict[int, Bounds]:
    """Private proofless candidate view; authority remains CPU binary64."""

    result: Dict[int, Bounds] = {}
    for lid, value in _clone_bounds(source).items():
        result[int(lid)] = Bounds(
            lb=value.lb.to(device=device, dtype=torch.float64).contiguous(),
            ub=value.ub.to(device=device, dtype=torch.float64).contiguous(),
        )
    return result


def _validate_real_solver_net_device(
    net: Any, device: torch.device, dtype: torch.dtype
) -> None:
    """Ensure the non-copyable live ACT parameters match DualSolver defaults."""

    if dtype != torch.float64:
        _fail(
            "CANDIDATE_DTYPE_MISMATCH",
            "production query-dual candidates require global torch.float64",
        )
    by_id, _ = _layer_maps(net)
    for lid, layer in by_id.items():
        if _kind(layer) not in {"DENSE", "CONV2D"}:
            continue
        params = getattr(layer, "params", {}) or {}
        weight = params.get("weight")
        if not isinstance(weight, torch.Tensor):
            _fail(
                "CANDIDATE_PARAMETER_DEVICE",
                f"production affine weight at layer {lid} must be a torch tensor",
            )
        tensors = [("weight", weight)]
        bias = params.get("bias")
        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                _fail(
                    "CANDIDATE_PARAMETER_DEVICE",
                    f"production affine bias at layer {lid} must be a torch tensor",
                )
            tensors.append(("bias", bias))
        for name, value in tensors:
            if value.device != device or value.dtype != dtype:
                _fail(
                    "CANDIDATE_PARAMETER_DEVICE",
                    f"layer {lid} {name} is {value.device}/{value.dtype}, "
                    f"expected {device}/{dtype}",
                )


def _flat_box(value: Bounds, *, lid: int) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, Bounds):
        _fail("INVALID_BOUNDS", f"bounds[{lid}] must be Bounds")
    lower = _as_numpy_f64(value.lb, name=f"bounds[{lid}].lb").reshape(-1)
    upper = _as_numpy_f64(value.ub, name=f"bounds[{lid}].ub").reshape(-1)
    if lower.shape != upper.shape or np.any(lower > upper):
        _fail("INVALID_BOUNDS", f"bounds[{lid}] has invalid endpoints")
    return lower, upper


def _layer_maps(net: Any) -> Tuple[Dict[int, Any], Dict[int, Tuple[int, ...]]]:
    layers = getattr(net, "layers", None)
    if not isinstance(layers, Sequence) or not layers:
        _fail("INVALID_GRAPH", "net.layers must be a nonempty sequence")
    by_id: Dict[int, Any] = {}
    for layer in layers:
        lid = int(getattr(layer, "id"))
        if lid in by_id:
            _fail("INVALID_GRAPH", f"duplicate layer id {lid}")
        by_id[lid] = layer
    raw_preds = getattr(net, "preds", None)
    if not isinstance(raw_preds, Mapping):
        _fail("INVALID_GRAPH", "net.preds must be a mapping")
    preds: Dict[int, Tuple[int, ...]] = {}
    for lid in by_id:
        values = tuple(int(value) for value in raw_preds.get(lid, ()))
        if any(value not in by_id for value in values):
            _fail("INVALID_GRAPH", f"layer {lid} has an unknown predecessor")
        preds[lid] = values
    return by_id, preds


def _kind(layer: Any) -> str:
    return str(getattr(getattr(layer, "kind", ""), "value", getattr(layer, "kind", ""))).upper()


def _assert_output_id(
    by_id: Mapping[int, Any], preds: Mapping[int, Tuple[int, ...]]
) -> int:
    assertions = [lid for lid, layer in by_id.items() if _kind(layer) == "ASSERT"]
    if len(assertions) != 1 or len(preds[assertions[0]]) != 1:
        _fail("INVALID_GRAPH", "exactly one unary ASSERT is required")
    return int(preds[assertions[0]][0])


def _ancestor_ids(
    by_id: Mapping[int, Any],
    preds: Mapping[int, Tuple[int, ...]],
    start_lid: Optional[int],
) -> Tuple[int, ...]:
    output = (
        _assert_output_id(by_id, preds)
        if start_lid is None
        else int(start_lid)
    )
    if output not in by_id:
        _fail("INVALID_GRAPH", f"unknown replay start layer {output}")
    seen = {output}
    stack = [output]
    while stack:
        for pred in preds[stack.pop()]:
            if pred not in seen:
                seen.add(pred)
                stack.append(pred)
    return tuple(sorted(seen))


def _candidate_bounds_sha256(bounds: Mapping[int, Bounds]) -> str:
    digest = hashlib.sha256()
    keys = sorted(int(key) for key in bounds)
    digest.update(np.asarray([len(keys)], dtype=np.int64).tobytes())
    for lid in keys:
        value = bounds[lid]
        lb = torch.as_tensor(value.lb).detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        ub = torch.as_tensor(value.ub).detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        if lb.shape != ub.shape or bool((lb > ub).any()):
            _fail("INVALID_BOUNDS", f"candidate bounds[{lid}] is invalid")
        digest.update(np.asarray([lid], dtype=np.int64).tobytes())
        digest.update(np.asarray(lb.shape, dtype=np.int64).tobytes())
        digest.update(lb.numpy().tobytes())
        digest.update(ub.numpy().tobytes())
    return digest.hexdigest()


def _boxes_sha256(
    net: Any,
    bounds: Mapping[int, Bounds],
) -> str:
    by_id, _ = _layer_maps(net)
    records = []
    for lid in sorted(bounds):
        if lid not in by_id:
            _fail("INVALID_BOUNDS", f"bounds includes unknown layer {lid}")
        lower, upper = _flat_box(bounds[lid], lid=lid)
        records.append(
            {
                "id": int(lid),
                "kind": _kind(by_id[lid]),
                "semantics": (
                    "preactivation" if _kind(by_id[lid]) == "RELU" else "output"
                ),
                "shape": list(torch.as_tensor(bounds[lid].lb).shape),
                "lb_sha256": _array_digest(lower),
                "ub_sha256": _array_digest(upper),
            }
        )
    return _json_sha256(records)


def _replay_bounds_sha256(
    net: Any,
    bounds: Mapping[int, Bounds],
    start_lid: Optional[int],
) -> str:
    """Independently reproduce replay's ancestor-cone bounds manifest."""

    by_id, preds = _layer_maps(net)
    records = []
    for lid in _ancestor_ids(by_id, preds, start_lid):
        kind = _kind(by_id[lid])
        if kind == "INPUT":
            continue
        if lid not in bounds:
            _fail("MISSING_BOUNDS", f"replay ancestor {lid} has no certified box")
        lower, upper = _flat_box(bounds[lid], lid=lid)
        records.append(
            {
                "id": int(lid),
                "semantics": "preactivation" if kind == "RELU" else "output",
                "lb_sha256": _array_digest(lower),
                "ub_sha256": _array_digest(upper),
            }
        )
    return _json_sha256(records)


def _replay_query_sha256(
    objectives: np.ndarray,
    bias: np.ndarray,
    start_lid: Optional[int],
    *,
    output_lid: int,
) -> str:
    start_mode = (
        "ASSERT_PREDECESSOR" if start_lid is None else "EXPLICIT_INTERIOR"
    )
    return _json_sha256(
        {
            "rows": _array_digest(objectives),
            "bias": _array_digest(bias),
            "start_layer_id": int(output_lid),
            "start_mode": start_mode,
        }
    )


def _property_spec_sha256(rows: np.ndarray, thresholds: np.ndarray) -> str:
    return _json_sha256(
        {
            "rows_sha256": _array_digest(rows),
            "thresholds_sha256": _array_digest(thresholds),
            "semantics": "upper_bounds_of_Cy_minus_threshold",
        }
    )


def _callable_name(value: Any) -> str:
    if value is None:
        return "none"
    module = str(getattr(value, "__module__", type(value).__module__))
    qualname = str(getattr(value, "__qualname__", type(value).__qualname__))
    return f"{module}.{qualname}"


def _effective_deadline(
    deadline: Optional[float], timeout_s: Optional[float]
) -> Tuple[Optional[float], float]:
    started = time.monotonic()
    ends = []
    for name, value in (("deadline", deadline), ("timeout_s", timeout_s)):
        if value is None:
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            _fail("INVALID_DEADLINE", f"{name} must be finite")
        if name == "timeout_s":
            if float(value) < 0.0:
                _fail("INVALID_DEADLINE", "timeout_s must be nonnegative")
            ends.append(started + float(value))
        else:
            ends.append(float(value))
    return (min(ends) if ends else None), started


def _check_deadline(deadline: Optional[float], where: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise QueryDualPipelineTimeout(f"deadline expired {where}")


def _normalise_property(
    property_rows: Any, thresholds: Any
) -> Tuple[np.ndarray, np.ndarray]:
    rows = _as_numpy_f64(property_rows, name="property_rows")
    threshold = _as_numpy_f64(thresholds, name="thresholds").reshape(-1)
    if rows.ndim != 2 or rows.shape[0] <= 0 or rows.shape[1] <= 0:
        _fail("INVALID_PROPERTY", "property_rows must be a nonempty matrix")
    if threshold.shape != (rows.shape[0],):
        _fail("INVALID_PROPERTY", "threshold count differs from property rows")
    return rows.copy(), threshold.copy()


def _normalise_targets(
    values: Sequence[int], by_id: Mapping[int, Any]
) -> Tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        _fail("INVALID_TARGETS", "target_relu_ids must be an explicit sequence")
    result = []
    for raw in values:
        if (
            isinstance(raw, (bool, np.bool_))
            or not isinstance(raw, (Integral, np.integer))
        ):
            _fail("INVALID_TARGETS", f"invalid target ReLU id {raw!r}")
        lid = int(raw)
        if lid not in by_id or _kind(by_id[lid]) != "RELU":
            _fail("INVALID_TARGETS", f"layer {lid} is not a ReLU")
        if lid in result:
            _fail("INVALID_TARGETS", f"duplicate target ReLU id {lid}")
        result.append(lid)
    return tuple(result)


def _flat_alpha_tree(
    tree: Any,
    *,
    net: Any,
    start_lid: Optional[int],
) -> Mapping[int, torch.Tensor]:
    """Reject generic pytrees; authority V1 accepts real flat ReLU alpha maps."""

    if not isinstance(tree, Mapping) or not tree:
        _fail("INVALID_ALPHA", "candidate alpha must be a nonempty flat mapping")
    by_id, preds = _layer_maps(net)
    ancestors = set(_ancestor_ids(by_id, preds, start_lid))
    result: Dict[int, torch.Tensor] = {}
    for raw_lid, raw_value in tree.items():
        if (
            isinstance(raw_lid, (bool, np.bool_))
            or not isinstance(raw_lid, (Integral, np.integer))
        ):
            _fail("INVALID_ALPHA", f"invalid alpha layer key {raw_lid!r}")
        lid = int(raw_lid)
        if (
            lid in result
            or lid not in ancestors
            or lid not in by_id
            or _kind(by_id[lid]) != "RELU"
        ):
            _fail("INVALID_ALPHA", f"alpha key {lid} is not an ancestor ReLU")
        if not isinstance(raw_value, torch.Tensor):
            _fail("INVALID_ALPHA", f"alpha[{lid}] is not a stored tensor")
        if (
            raw_value.device.type != "cpu"
            or raw_value.dtype != torch.float64
            or not bool(torch.isfinite(raw_value).all())
            or bool((raw_value < 0.0).any())
            or bool((raw_value > 1.0).any())
        ):
            _fail(
                "INVALID_ALPHA",
                f"alpha[{lid}] must be finite CPU binary64 in [0,1]",
            )
        result[lid] = (
            raw_value.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
    return result


def _expected_target_objective(
    row_ids: Tuple[int, ...], width: int
) -> np.ndarray:
    eye = np.zeros((len(row_ids), width), dtype=np.float64)
    eye[np.arange(len(row_ids)), np.asarray(row_ids, dtype=np.int64)] = 1.0
    return np.vstack([eye, -eye])


def _candidate_v2_receipt_semantics(receipt: Mapping[str, Any]) -> bool:
    """Validate descriptor-only receipt semantics without granting authority."""

    try:
        if (
            not verify_query_dual_candidates_receipt(receipt)
            or receipt.get("schema") != _CANDIDATE_SCHEMA
            or receipt.get("protocol") != _CANDIDATE_PROTOCOL
            or receipt.get("non_authoritative_audit_fields")
            != _CANDIDATE_NON_AUTHORITATIVE_AUDIT_FIELDS
            or receipt.get("candidate_only") is not True
            or receipt.get("proof_authority") is not False
            or receipt.get("return_optimized_required") is not True
            or receipt.get("refresh_forward") is not False
            or receipt.get("bounds_source")
            != "caller_frozen_bounds_private_clone"
            or receipt.get("alpha_storage")
            != "cpu_stored_binary64_tree"
            or receipt.get("candidate_bound_source")
            != "none_descriptor_only"
            or receipt.get("optimizer_best_margins_used_as_bounds") is not False
            or receipt.get("optimizer_margins_exported") is not False
            or receipt.get("optimizer_margins_used_for_improvement") is not False
            or receipt.get("gpu_frozen_alpha_replay") is not False
            or receipt.get("cpu_independent_replay_required") is not True
            or receipt.get("all_candidate_updates_replayed_with_stored_alpha")
            is not False
            or receipt.get("all_bounds_replayed_with_stored_alpha") is not False
            or receipt.get("property_lower_dual_replayed") is not False
            or receipt.get("property_upper_only") is not True
            or receipt.get("shared_absolute_deadline")
            is not (receipt.get("deadline_monotonic") is not None)
            or receipt.get("property_only")
            is not (receipt.get("target_relu_lid") is None)
            or receipt.get("caller_bounds_unchanged") is not True
            or receipt.get("strict_target_improvements") != 0
            or receipt.get("strict_property_improvements") != 0
            or receipt.get("improved_target_indices") != []
            or receipt.get("improved_property_indices") != []
            or receipt.get("candidate_target_bounds_sha256")
            != receipt.get("target_bounds_sha256")
            or receipt.get("candidate_property_bounds_sha256")
            != receipt.get("property_baseline_sha256")
        ):
            return False
        status = receipt.get("status")
        records = receipt.get("descriptor_records")
        alpha_hashes = receipt.get("alpha_hashes")
        if not isinstance(records, list) or not isinstance(alpha_hashes, list):
            return False
        if (
            receipt.get("descriptor_records_sha256")
            != _json_sha256(records)
            or receipt.get("descriptor_coverage_sha256")
            != _json_sha256(records)
            or receipt.get("alpha_hashes_sha256")
            != _json_sha256(alpha_hashes)
        ):
            return False
        if status == "descriptors_generated":
            block_count = receipt.get("query_blocks")
            tree_count = receipt.get("alpha_trees")
            if (
                isinstance(block_count, (bool, np.bool_))
                or not isinstance(block_count, (Integral, np.integer))
                or int(block_count) <= 0
                or isinstance(tree_count, (bool, np.bool_))
                or not isinstance(tree_count, (Integral, np.integer))
                or int(tree_count) != int(block_count)
                or len(records) != int(block_count)
                or len(alpha_hashes) != int(block_count)
                or receipt.get("candidate_generated") is not True
                or receipt.get("whole_batch_complete") is not True
                or receipt.get("descriptor_coverage_complete") is not True
                or receipt.get("completed_blocks_discarded") != 0
            ):
                return False
            for index, record in enumerate(records):
                if (
                    not isinstance(record, Mapping)
                    or record.get("block_id") != index
                    or record.get("alpha_tree_index") != index
                    or record.get("bound_source") != "none_descriptor_only"
                    or record.get("alpha_sha256") != alpha_hashes[index]
                    or "optimizer_margin_sha256" in record
                    or "replay_margin_sha256" in record
                ):
                    return False
            return True
        if (
            receipt.get("candidate_generated") is not False
            or receipt.get("query_blocks") != 0
            or receipt.get("alpha_trees") != 0
            or records != []
            or alpha_hashes != []
        ):
            return False
        if status == "no_queries_fallback":
            return bool(
                receipt.get("whole_batch_complete") is True
                and receipt.get("descriptor_coverage_complete") is True
                and receipt.get("completed_blocks_discarded") == 0
            )
        completed = receipt.get("completed_blocks_discarded")
        return bool(
            status
            in {
                "deadline_fallback_frozen_bounds",
                "error_fallback_frozen_bounds",
            }
            and receipt.get("whole_batch_complete") is False
            and receipt.get("descriptor_coverage_complete") is False
            and not isinstance(completed, (bool, np.bool_))
            and isinstance(completed, (Integral, np.integer))
            and int(completed) >= 0
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _check_candidate_common(
    candidate: QueryDualCandidates,
    bounds: Mapping[int, Bounds],
    *,
    target_relu_lid: Optional[int],
    property_rows: Optional[np.ndarray],
    block_size: int,
    steps: int,
    deadline: Optional[float],
) -> None:
    if (
        not validate_query_dual_candidates(candidate)
        or not _candidate_v2_receipt_semantics(candidate.receipt)
    ):
        _fail("INVALID_CANDIDATE", "candidate full-object validation failed")
    receipt = candidate.receipt
    source_hash = _candidate_bounds_sha256(bounds)
    if (
        receipt.get("input_bounds_sha256") != source_hash
        or receipt.get("target_relu_lid") != target_relu_lid
        or receipt.get("block_size") != int(block_size)
        or receipt.get("steps_requested") != int(steps)
        or receipt.get("property_upper_only") is not True
        or receipt.get("deadline_monotonic")
        != (float(deadline) if deadline is not None else None)
    ):
        _fail("CANDIDATE_BINDING", "candidate invocation metadata mismatch")
    expected_rows = (
        0 if property_rows is None else int(property_rows.shape[0])
    )
    if (
        int(receipt.get("property_rows", -1)) != expected_rows
        or (
            property_rows is not None
            and receipt.get("property_rows_sha256")
            != _candidate_array_digest(property_rows)
        )
    ):
        _fail("CANDIDATE_BINDING", "candidate property rows mismatch")


def _replay_descriptor(
    *,
    net: Any,
    parent_bounds: Mapping[int, Bounds],
    descriptor: QueryDescriptor,
    alpha_tree: Any,
    query_bias: np.ndarray,
    expected_objectives: np.ndarray,
    expected_kind: str,
    expected_target: Optional[int],
    expected_start: Optional[int],
    expected_rows: Tuple[int, ...],
    chunk_size: int,
    max_workspace_bytes: int,
    deadline: Optional[float],
) -> QueryDualAuthorityBlock:
    immutable_parent_hash = _boxes_sha256(net, parent_bounds)
    proof_bounds = _clone_bounds(parent_bounds)
    if _boxes_sha256(net, proof_bounds) != immutable_parent_hash:
        _fail("PARENT_TOCTOU", "parent changed while creating replay snapshot")
    objectives = _as_numpy_f64(
        descriptor.objectives, name=f"descriptor[{descriptor.block_id}].objectives"
    )
    if (
        descriptor.query_kind != expected_kind
        or descriptor.target_relu_lid != expected_target
        or descriptor.start_lid != expected_start
        or tuple(descriptor.row_ids) != expected_rows
        or descriptor.M != int(expected_objectives.shape[0])
        or objectives.shape != expected_objectives.shape
        or not np.array_equal(objectives, expected_objectives)
        or descriptor.objective_sha256
        != _candidate_array_digest(expected_objectives)
    ):
        _fail("OBJECTIVE_BINDING", "candidate descriptor semantics mismatch")
    alpha = _flat_alpha_tree(alpha_tree, net=net, start_lid=expected_start)
    if query_dual_stored_alpha_sha256(alpha) != descriptor.alpha_sha256:
        _fail(
            "ALPHA_BINDING",
            "cloned flat alpha differs from the candidate descriptor",
        )
    output_lid = (
        _assert_output_id(*_layer_maps(net))
        if expected_start is None
        else int(expected_start)
    )
    expected_query_hash = _replay_query_sha256(
        expected_objectives,
        query_bias,
        expected_start,
        output_lid=output_lid,
    )
    expected_bounds_hash = _replay_bounds_sha256(
        net, proof_bounds, expected_start
    )
    _check_deadline(deadline, "before independent replay")
    result = _TRUSTED_REPLAYER(
        net,
        proof_bounds,
        start_lid=expected_start,
        query_rows=objectives,
        query_bias=query_bias,
        alpha_by_relu=alpha,
        chunk_size=int(chunk_size),
        max_workspace_bytes=int(max_workspace_bytes),
        deadline=deadline,
    )
    _check_deadline(deadline, "after independent replay")
    if _boxes_sha256(net, parent_bounds) != immutable_parent_hash:
        _fail("PARENT_TOCTOU", "replayer changed the immutable parent boxes")
    if _boxes_sha256(net, proof_bounds) != immutable_parent_hash:
        _fail("PARENT_TOCTOU", "replayer changed its private proof snapshot")
    if query_dual_stored_alpha_sha256(alpha) != descriptor.alpha_sha256:
        _fail("ALPHA_TOCTOU", "replayer changed the frozen alpha clone")
    if not isinstance(result, QueryDualReplayResult):
        _fail("INVALID_REPLAY", "replayer returned a non-authoritative type")
    hashes = result.receipt.get("hashes", {})
    replay_alpha_hash = str(hashes.get("alpha_sha256", ""))
    replay_net_hash = str(hashes.get("net_sha256", ""))
    if not validate_query_dual_replay_result(
        result,
        expected_net_sha256=replay_net_hash,
        expected_bounds_sha256=expected_bounds_hash,
        expected_query_sha256=expected_query_hash,
        expected_alpha_sha256=replay_alpha_hash,
    ):
        _fail("INVALID_REPLAY", "live replay result/receipt binding failed")
    candidate_alpha_hash = str(descriptor.alpha_sha256)
    bridge_hash = _json_sha256(
        {
            "candidate_alpha_sha256": candidate_alpha_hash,
            "replay_alpha_sha256": replay_alpha_hash,
            "objective_sha256": descriptor.objective_sha256,
            "replay_query_sha256": expected_query_hash,
        }
    )
    return QueryDualAuthorityBlock(
        block_id=int(descriptor.block_id),
        query_kind=str(descriptor.query_kind),
        start_lid=descriptor.start_lid,
        target_relu_lid=descriptor.target_relu_lid,
        row_ids=tuple(int(value) for value in descriptor.row_ids),
        objective_sha256=str(descriptor.objective_sha256),
        candidate_alpha_sha256=candidate_alpha_hash,
        replay_query_sha256=expected_query_hash,
        replay_alpha_sha256=replay_alpha_hash,
        replay_bounds_sha256=expected_bounds_hash,
        replay_net_sha256=replay_net_hash,
        alpha_bridge_sha256=bridge_hash,
        lower_bounds=result.lower_bounds,
        replay_receipt=result.receipt,
    )


def _sealed_property_replay_workers() -> int:
    raw_workers = os.environ.get("HZ_QUERY_WORKERS", "1")
    try:
        workers = int(raw_workers)
    except (TypeError, ValueError) as exc:
        raise QueryDualPipelineError(
            "INVALID_WORKERS",
            "HZ_QUERY_WORKERS must be an integer in [1, 32]",
        ) from exc
    if workers <= 0 or workers > 32:
        raise QueryDualPipelineError(
            "INVALID_WORKERS",
            "HZ_QUERY_WORKERS must be an integer in [1, 32]",
        )
    return workers


def _replay_property_descriptors_sealed(
    *,
    net: Any,
    root: QueryDualBoxCertificate,
    parent_bounds: Mapping[int, Bounds],
    candidate: QueryDualCandidates,
    rows: np.ndarray,
    thresholds: np.ndarray,
    chunk_size: int,
    max_workspace_bytes: int,
    deadline: float,
) -> Tuple[Tuple[QueryDualAuthorityBlock, ...], np.ndarray]:
    """Replay property descriptors in one root-owned sealed CPU session."""

    from act.back_end.hybridz_tf.query_dual_pipeline_v3 import (
        _committed_block,
        _pending_descriptor,
    )
    from act.back_end.hybridz_tf.query_dual_replay import (
        create_query_dual_replay_session,
    )

    proof_workers = _sealed_property_replay_workers()

    try:
        session = create_query_dual_replay_session(
            net,
            root,
            (None,),
            deadline=float(deadline),
        )
    except (QueryDualReplayError, QueryDualReplayTimeout) as exc:
        raise QueryDualPipelineError(
            "SEALED_CONTEXT", str(exc)
        ) from exc
    try:
        frame = session.seal_bounds(
            parent_bounds, start_lids=(None,)
        )
        records = candidate.receipt.get("descriptor_records")
        if (
            not isinstance(records, list)
            or len(records) != len(candidate.query_descriptors)
        ):
            _fail(
                "CANDIDATE_BINDING",
                "sealed replay descriptor records are incomplete",
            )
        pending = []
        property_upper = np.empty(rows.shape[0], dtype=np.float64)
        covered = []
        for descriptor_index, descriptor in enumerate(
            candidate.query_descriptors
        ):
            row_ids = tuple(int(value) for value in descriptor.row_ids)
            index = np.asarray(row_ids, dtype=np.int64)
            if (
                not row_ids
                or np.any(index < 0)
                or np.any(index >= rows.shape[0])
            ):
                _fail(
                    "COVERAGE_ERROR",
                    "invalid sealed property row ids",
                )
            item = _pending_descriptor(
                session=session,
                frame=frame,
                net=net,
                parent_bounds=parent_bounds,
                descriptor=descriptor,
                candidate_record=records[descriptor_index],
                alpha_tree=candidate.alpha_trees[
                    descriptor.alpha_tree_index
                ],
                query_bias=thresholds[index],
                expected_objectives=-rows[index],
                expected_kind=(
                    "final_property_negative_c_upper_only"
                ),
                expected_target=None,
                expected_start=None,
                expected_rows=row_ids,
                chunk_size=int(chunk_size),
                max_workspace_bytes=int(max_workspace_bytes),
                deadline=float(deadline),
                proof_workers=int(proof_workers),
            )
            if item.pending.lower_bounds.size != len(row_ids):
                _fail(
                    "SHAPE_MISMATCH",
                    "sealed property replay result count mismatch",
                )
            property_upper[index] = -item.pending.lower_bounds
            covered.extend(row_ids)
            pending.append(item)
        if tuple(covered) != tuple(range(rows.shape[0])):
            _fail(
                "COVERAGE_ERROR",
                "sealed property descriptor coverage/order mismatch",
            )
        committed = session.commit()
        if len(committed) != len(pending):
            _fail(
                "COVERAGE_ERROR",
                "sealed replay commit count mismatch",
            )
        blocks = tuple(
            _committed_block(item, result)
            for item, result in zip(pending, committed)
        )
        if not np.all(np.isfinite(property_upper)):
            _fail(
                "NONFINITE",
                "sealed property replay produced non-finite bounds",
            )
        return blocks, property_upper
    except Exception:
        session.abort()
        raise


def _replace_box(
    bounds: Dict[int, Bounds],
    lid: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    original = bounds[lid]
    shape = tuple(int(value) for value in original.lb.shape)
    if int(np.prod(shape)) != int(lower.size):
        _fail("SHAPE_MISMATCH", f"replacement bounds[{lid}] width mismatch")
    bounds[lid] = Bounds(
        lb=torch.from_numpy(np.ascontiguousarray(lower.copy())).reshape(shape),
        ub=torch.from_numpy(np.ascontiguousarray(upper.copy())).reshape(shape),
    )


def _root_still_valid(certificate: QueryDualBoxCertificate, net: Any) -> None:
    if not verify_query_dual_box_certificate(certificate, net=net):
        _fail("ROOT_TOCTOU", "live net/root certificate changed during transaction")


def _register_live(bundle: VerifiedQueryDualFeedback) -> None:
    object_id = id(bundle)
    nonce = bundle.provenance_nonce

    def cleanup(reference: "weakref.ReferenceType[VerifiedQueryDualFeedback]") -> None:
        del reference
        with _LIVE_LOCK:
            current = _LIVE_AUTHORITIES.get(object_id)
            if current is not None and current[1] == nonce:
                _LIVE_AUTHORITIES.pop(object_id, None)

    reference = weakref.ref(bundle, cleanup)
    with _LIVE_LOCK:
        _LIVE_AUTHORITIES[object_id] = (
            reference,
            nonce,
            str(bundle.receipt["receipt_sha256"]),
        )


def _has_live_capability(bundle: VerifiedQueryDualFeedback) -> bool:
    with _LIVE_LOCK:
        entry = _LIVE_AUTHORITIES.get(id(bundle))
        return bool(
            entry is not None
            and entry[0]() is bundle
            and hmac.compare_digest(entry[1], bundle.provenance_nonce)
            and hmac.compare_digest(
                entry[2], str(bundle.receipt.get("receipt_sha256", ""))
            )
        )


def build_verified_query_dual_feedback(
    net: Any,
    property_rows: Any,
    thresholds: Any,
    *,
    target_relu_ids: Sequence[int],
    steps: int = 8,
    block_size: int = 1024,
    lr_alpha: float = 0.25,
    lr_decay: float = 0.98,
    replay_chunk_size: int = 1024,
    replay_max_workspace_bytes: int = 512 * 1024 * 1024,
    conv_channel_chunk: int = 32,
    candidate_device: str = "cuda",
    deadline: Optional[float] = None,
    timeout_s: Optional[float] = None,
    solver_factory: Optional[Callable[[], Any]] = None,
    candidate_generator: Callable[..., QueryDualCandidates] = (
        generate_query_dual_candidates
    ),
    verified_preactivation_frame: Optional[Any] = None,
) -> VerifiedQueryDualFeedback:
    """Build one all-or-nothing query-dual authority transaction.

    ``target_relu_ids`` is explicit and its order is preserved.  The default
    query and replay block size is 1024.  A custom ``solver_factory`` or
    ``candidate_generator`` affects only non-authoritative proposal generation
    and is recorded in the audit receipt.  The authority-bearing certifier and
    replayer are deliberately not injectable.
    """

    effective_deadline, started = _effective_deadline(deadline, timeout_s)
    rows, threshold = _normalise_property(property_rows, thresholds)
    by_id, preds = _layer_maps(net)
    targets = _normalise_targets(target_relu_ids, by_id)
    if verified_preactivation_frame is not None and targets:
        _fail(
            "INVALID_CONFIG",
            "operator-HZ preactivation frames are property-only; "
            "target_relu_ids must be empty",
        )
    if not isinstance(candidate_device, str) or candidate_device not in {
        "cpu",
        "cuda",
    }:
        _fail("INVALID_CONFIG", "candidate_device must be 'cpu' or 'cuda'")
    if candidate_device == "cuda" and not torch.cuda.is_available():
        _fail(
            "CANDIDATE_DEVICE_UNAVAILABLE",
            "CUDA candidate generation was requested but CUDA is unavailable",
        )
    requested_candidate_device = torch.device(candidate_device)
    default_candidate_device = get_default_device()
    default_candidate_dtype = get_default_dtype()
    if default_candidate_device.type != requested_candidate_device.type:
        _fail(
            "CANDIDATE_DEVICE_MISMATCH",
            "candidate_device "
            f"{requested_candidate_device.type!r} differs from DualSolver default "
            f"device {default_candidate_device.type!r}",
        )
    candidate_torch_device = default_candidate_device
    if (
        candidate_generator is generate_query_dual_candidates
        and solver_factory is None
    ):
        _validate_real_solver_net_device(
            net, candidate_torch_device, default_candidate_dtype
        )
    for name, value in (
        ("steps", steps),
        ("block_size", block_size),
        ("replay_chunk_size", replay_chunk_size),
        ("replay_max_workspace_bytes", replay_max_workspace_bytes),
        ("conv_channel_chunk", conv_channel_chunk),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (Integral, np.integer))
            or int(value) <= 0
        ):
            _fail("INVALID_CONFIG", f"{name} must be a positive integer")
    output_lid = _assert_output_id(by_id, preds)
    _check_deadline(effective_deadline, "before root certification")
    try:
        root = _TRUSTED_CERTIFIER(
            net,
            deadline=effective_deadline,
            conv_channel_chunk=int(conv_channel_chunk),
        )
    except QueryDualBoxTimeout as exc:
        raise QueryDualPipelineTimeout(str(exc)) from exc
    except QueryDualBoxError as exc:
        raise QueryDualPipelineError("ROOT_CERTIFICATION", str(exc)) from exc
    _check_deadline(effective_deadline, "after root certification")
    if not verify_query_dual_box_certificate(root, net=net):
        _fail("ROOT_CERTIFICATION", "root full-object validation failed")
    current = _clone_bounds(root.bounds)
    if output_lid not in current:
        _fail("ROOT_CERTIFICATION", "root certificate omits ASSERT predecessor")
    output_width = _flat_box(current[output_lid], lid=output_lid)[0].size
    if rows.shape[1] != output_width:
        _fail("INVALID_PROPERTY", "property width differs from network output")
    root_boxes_hash = _boxes_sha256(net, current)
    if verified_preactivation_frame is None:
        initial_frame_audit: Dict[str, Any] = {
            "schema": "query_dual_operator_hz_bound_frame_v1",
            "enabled": False,
            "proof_authority": False,
            "source": "none",
            "source_layer_ids": [],
            "strict_lower_rows": 0,
            "strict_upper_rows": 0,
            "committed_boxes_sha256": root_boxes_hash,
            "intersection_only": True,
            "target_replay_stages_required": False,
        }
    else:
        current, initial_frame_audit = (
            _apply_operator_hz_preactivation_frame(
                net=net,
                root_bounds=current,
                frame=verified_preactivation_frame,
                expected_network_sha256=str(
                    root.receipt["hashes"]["net_sha256"]
                ),
            )
        )
    stages = []

    try:
        for stage_index, target_lid in enumerate(targets):
            _check_deadline(effective_deadline, f"before target stage {stage_index}")
            parent = _clone_bounds(current)
            parent_hash = _boxes_sha256(net, parent)
            candidate_kwargs: Dict[str, Any] = {
                "net": net,
                "bounds_dict": _candidate_bounds_on_device(
                    parent, candidate_torch_device
                ),
                "target_relu_lid": int(target_lid),
                "property_rows": None,
                "property_upper_only": True,
                "steps": int(steps),
                "block_size": int(block_size),
                "lr_alpha": float(lr_alpha),
                "lr_decay": float(lr_decay),
                "deadline": effective_deadline,
                "descriptor_only": True,
            }
            if solver_factory is not None:
                candidate_kwargs["solver_factory"] = solver_factory
            candidate = candidate_generator(**candidate_kwargs)
            _check_deadline(effective_deadline, f"after target candidate {target_lid}")
            _root_still_valid(root, net)
            if _boxes_sha256(net, parent) != parent_hash:
                _fail(
                    "PARENT_TOCTOU",
                    f"candidate changed parent boxes at target {target_lid}",
                )
            _check_candidate_common(
                candidate,
                parent,
                target_relu_lid=int(target_lid),
                property_rows=None,
                block_size=int(block_size),
                steps=int(steps),
                deadline=effective_deadline,
            )
            target_preds = preds[target_lid]
            if len(target_preds) != 1:
                _fail("INVALID_TARGETS", f"target ReLU {target_lid} is not unary")
            predecessor = int(target_preds[0])
            predecessor_kind = _kind(by_id[predecessor])
            target_lower, target_upper = _flat_box(
                parent[target_lid], lid=target_lid
            )
            combined_lower = target_lower.copy()
            combined_upper = target_upper.copy()
            if predecessor_kind != "RELU":
                pred_lower, pred_upper = _flat_box(
                    parent[predecessor], lid=predecessor
                )
                if pred_lower.shape != combined_lower.shape:
                    _fail(
                        "SHAPE_MISMATCH",
                        f"target {target_lid}/predecessor widths differ",
                    )
                combined_lower = np.maximum(combined_lower, pred_lower)
                combined_upper = np.minimum(combined_upper, pred_upper)
            pre_replay_lower = combined_lower.copy()
            pre_replay_upper = combined_upper.copy()
            blocks = []
            status = str(candidate.status)
            expected_unstable = tuple(
                int(value)
                for value in np.flatnonzero(
                    (target_lower < 0.0) & (target_upper > 0.0)
                )
            )
            if status == "descriptors_generated":
                seen_rows = []
                for descriptor in candidate.query_descriptors:
                    descriptor_rows = tuple(int(value) for value in descriptor.row_ids)
                    expected_objective = _expected_target_objective(
                        descriptor_rows, target_lower.size
                    )
                    if any(
                        value not in expected_unstable for value in descriptor_rows
                    ):
                        _fail(
                            "OBJECTIVE_BINDING",
                            f"target {target_lid} descriptor queries stable neuron",
                        )
                    block = _replay_descriptor(
                        net=net,
                        parent_bounds=parent,
                        descriptor=descriptor,
                        alpha_tree=candidate.alpha_trees[
                            descriptor.alpha_tree_index
                        ],
                        query_bias=np.zeros(descriptor.M, dtype=np.float64),
                        expected_objectives=expected_objective,
                        expected_kind="relu_unstable_plus_minus_one_hot",
                        expected_target=int(target_lid),
                        expected_start=predecessor,
                        expected_rows=descriptor_rows,
                        chunk_size=int(replay_chunk_size),
                        max_workspace_bytes=int(replay_max_workspace_bytes),
                        deadline=effective_deadline,
                    )
                    _root_still_valid(root, net)
                    count = len(descriptor_rows)
                    replay_lower = block.lower_bounds[:count]
                    replay_upper = -block.lower_bounds[count:]
                    if (
                        block.lower_bounds.size != 2 * count
                        or np.any(replay_lower > replay_upper)
                    ):
                        _fail(
                            "BOUND_CONFLICT",
                            f"target {target_lid} replay lower exceeds upper",
                        )
                    index = np.asarray(descriptor_rows, dtype=np.int64)
                    combined_lower[index] = np.maximum(
                        combined_lower[index], replay_lower
                    )
                    combined_upper[index] = np.minimum(
                        combined_upper[index], replay_upper
                    )
                    if np.any(combined_lower > combined_upper):
                        _fail(
                            "BOUND_CONFLICT",
                            f"target {target_lid} intersection is empty",
                        )
                    blocks.append(block)
                    seen_rows.extend(descriptor_rows)
                if tuple(seen_rows) != expected_unstable:
                    _fail(
                        "COVERAGE_ERROR",
                        f"target {target_lid} descriptor coverage/order mismatch",
                    )
            elif status == "no_queries_fallback":
                if expected_unstable:
                    _fail(
                        "COVERAGE_ERROR",
                        f"target {target_lid} omitted unstable queries",
                    )
            else:
                if status.startswith("deadline_"):
                    raise QueryDualPipelineTimeout(
                        f"candidate deadline fallback at target {target_lid}"
                    )
                _fail(
                    "CANDIDATE_FAILURE",
                    f"candidate status {status!r} at target {target_lid}",
                )
            if np.any(combined_lower > combined_upper):
                _fail("BOUND_CONFLICT", f"target {target_lid} parent boxes conflict")
            prior_lower = target_lower.copy()
            prior_upper = target_upper.copy()
            next_bounds = _clone_bounds(parent)
            _replace_box(
                next_bounds,
                target_lid,
                combined_lower,
                combined_upper,
            )
            if predecessor_kind != "RELU":
                _replace_box(
                    next_bounds,
                    predecessor,
                    combined_lower,
                    combined_upper,
                )
            strict = int(
                np.count_nonzero(
                    (combined_lower > pre_replay_lower)
                    | (combined_upper < pre_replay_upper)
                )
            )
            synchronisation_strict = int(
                np.count_nonzero(
                    (pre_replay_lower > prior_lower)
                    | (pre_replay_upper < prior_upper)
                )
            )
            result_hash = _boxes_sha256(net, next_bounds)
            candidate_bounds_hash = _candidate_bounds_sha256(parent)
            stage_body = {
                "schema": _STAGE_SCHEMA,
                "status": (
                    "verified"
                    if strict > 0
                    else "verified_no_improvement"
                ),
                "proof_authority": True,
                "stage_index": int(stage_index),
                "target_relu_lid": int(target_lid),
                "predecessor_lid": predecessor,
                "predecessor_kind": predecessor_kind,
                "predecessor_synchronised": predecessor_kind != "RELU",
                "relu_key_semantics": "preactivation",
                "parent_boxes_sha256": parent_hash,
                "result_boxes_sha256": result_hash,
                "candidate_bounds_sha256": candidate_bounds_hash,
                "candidate_receipt_sha256": candidate.receipt["receipt_sha256"],
                "candidate_schema": candidate.receipt["schema"],
                "candidate_protocol": candidate.receipt["protocol"],
                "candidate_descriptor_coverage_sha256": candidate.receipt[
                    "descriptor_coverage_sha256"
                ],
                "candidate_status": status,
                "block_receipt_sha256": [
                    block.replay_receipt["receipt_sha256"] for block in blocks
                ],
                "alpha_bridge_sha256": [
                    block.alpha_bridge_sha256 for block in blocks
                ],
                "target_bounds_sha256": _array_digest(
                    np.stack([combined_lower, combined_upper])
                ),
                "pre_replay_target_bounds_sha256": _array_digest(
                    np.stack([pre_replay_lower, pre_replay_upper])
                ),
                "synchronisation_strict_improvements": (
                    synchronisation_strict
                ),
                "strict_improvements": strict,
                "commit": "atomic_whole_stage",
            }
            stage = QueryDualTargetStage(
                stage_index=int(stage_index),
                target_relu_lid=int(target_lid),
                predecessor_lid=predecessor,
                predecessor_kind=predecessor_kind,
                parent_boxes_sha256=parent_hash,
                result_boxes_sha256=result_hash,
                candidate_bounds_sha256=candidate_bounds_hash,
                candidate_receipt=candidate.receipt,
                blocks=tuple(blocks),
                target_lower=combined_lower,
                target_upper=combined_upper,
                strict_improvements=strict,
                status=str(stage_body["status"]),
                receipt=_receipt(stage_body),
            )
            stages.append(stage)
            current = next_bounds

        _check_deadline(effective_deadline, "before property candidate")
        property_parent = _clone_bounds(current)
        property_parent_hash = _boxes_sha256(net, property_parent)
        property_kwargs: Dict[str, Any] = {
            "net": net,
            "bounds_dict": _candidate_bounds_on_device(
                property_parent, candidate_torch_device
            ),
            "target_relu_lid": None,
            "property_rows": rows.copy(),
            "property_upper_only": True,
            "steps": int(steps),
            "block_size": int(block_size),
            "lr_alpha": float(lr_alpha),
            "lr_decay": float(lr_decay),
            "deadline": effective_deadline,
            "descriptor_only": True,
        }
        if solver_factory is not None:
            property_kwargs["solver_factory"] = solver_factory
        property_candidate = candidate_generator(**property_kwargs)
        _check_deadline(effective_deadline, "after property candidate")
        _root_still_valid(root, net)
        if _boxes_sha256(net, property_parent) != property_parent_hash:
            _fail("PARENT_TOCTOU", "property candidate changed parent boxes")
        _check_candidate_common(
            property_candidate,
            property_parent,
            target_relu_lid=None,
            property_rows=rows,
            block_size=int(block_size),
            steps=int(steps),
            deadline=effective_deadline,
        )
        if property_candidate.status != "descriptors_generated":
            if property_candidate.status.startswith("deadline_"):
                raise QueryDualPipelineTimeout("property candidate deadline fallback")
            _fail(
                "NO_PROPERTY_CANDIDATE",
                f"property candidate status {property_candidate.status!r}",
            )
        sealed_property_replay = bool(
            initial_frame_audit["enabled"]
            and effective_deadline is not None
        )
        property_replay_workers = (
            _sealed_property_replay_workers()
            if sealed_property_replay
            else 1
        )
        if sealed_property_replay:
            property_blocks, property_upper = (
                _replay_property_descriptors_sealed(
                    net=net,
                    root=root,
                    parent_bounds=property_parent,
                    candidate=property_candidate,
                    rows=rows,
                    thresholds=threshold,
                    chunk_size=int(replay_chunk_size),
                    max_workspace_bytes=int(
                        replay_max_workspace_bytes
                    ),
                    deadline=float(effective_deadline),
                )
            )
            _root_still_valid(root, net)
        else:
            property_blocks_list = []
            property_upper = np.empty(
                rows.shape[0], dtype=np.float64
            )
            covered = []
            for descriptor in property_candidate.query_descriptors:
                row_ids = tuple(
                    int(value) for value in descriptor.row_ids
                )
                index = np.asarray(row_ids, dtype=np.int64)
                if (
                    not row_ids
                    or np.any(index < 0)
                    or np.any(index >= rows.shape[0])
                ):
                    _fail(
                        "COVERAGE_ERROR",
                        "invalid property row ids",
                    )
                expected_objective = -rows[index]
                query_bias = threshold[index]
                block = _replay_descriptor(
                    net=net,
                    parent_bounds=property_parent,
                    descriptor=descriptor,
                    alpha_tree=property_candidate.alpha_trees[
                        descriptor.alpha_tree_index
                    ],
                    query_bias=query_bias,
                    expected_objectives=expected_objective,
                    expected_kind=(
                        "final_property_negative_c_upper_only"
                    ),
                    expected_target=None,
                    expected_start=None,
                    expected_rows=row_ids,
                    chunk_size=int(replay_chunk_size),
                    max_workspace_bytes=int(
                        replay_max_workspace_bytes
                    ),
                    deadline=effective_deadline,
                )
                _root_still_valid(root, net)
                if block.lower_bounds.size != len(row_ids):
                    _fail(
                        "SHAPE_MISMATCH",
                        "property replay result count mismatch",
                    )
                # The replay proves LB(-C y + threshold); negation is exact.
                property_upper[index] = -block.lower_bounds
                covered.extend(row_ids)
                property_blocks_list.append(block)
            if tuple(covered) != tuple(range(rows.shape[0])):
                _fail(
                    "COVERAGE_ERROR",
                    "property descriptor coverage/order mismatch",
                )
            property_blocks = tuple(property_blocks_list)
        if not np.all(np.isfinite(property_upper)):
            _fail("NONFINITE", "property replay produced non-finite upper bounds")
        property_spec_hash = _property_spec_sha256(rows, threshold)
        candidate_bounds_hash = _candidate_bounds_sha256(property_parent)
        property_body = {
            "schema": _PROPERTY_SCHEMA,
            "status": "verified",
            "proof_authority": True,
            "direction": "UPPER",
            "quantity": "C_y_minus_threshold",
            "objective": "-C",
            "replay_query_bias": "+threshold",
            "upper_reconstruction": "-LB(-C_y+threshold)",
            "parent_boxes_sha256": property_parent_hash,
            "candidate_bounds_sha256": candidate_bounds_hash,
            "candidate_receipt_sha256": property_candidate.receipt[
                "receipt_sha256"
            ],
            "candidate_schema": property_candidate.receipt["schema"],
            "candidate_protocol": property_candidate.receipt["protocol"],
            "candidate_status": property_candidate.status,
            "candidate_descriptor_coverage_sha256": (
                property_candidate.receipt["descriptor_coverage_sha256"]
            ),
            "block_receipt_sha256": [
                block.replay_receipt["receipt_sha256"]
                for block in property_blocks
            ],
            "alpha_bridge_sha256": [
                block.alpha_bridge_sha256 for block in property_blocks
            ],
            "property_spec_sha256": property_spec_hash,
            "property_upper_sha256": _array_digest(property_upper),
            "property_rows": int(rows.shape[0]),
            "coverage_complete": True,
            "replay_execution": (
                "root_owned_sealed_session_v1"
                if sealed_property_replay
                else "standalone_independent_replay_v2"
            ),
            "proof_workers_requested": int(property_replay_workers),
        }
        property_stage = QueryDualPropertyStage(
            parent_boxes_sha256=property_parent_hash,
            candidate_bounds_sha256=candidate_bounds_hash,
            candidate_receipt=property_candidate.receipt,
            blocks=tuple(property_blocks),
            property_upper=property_upper,
            property_spec_sha256=property_spec_hash,
            receipt=_receipt(property_body),
        )
        _root_still_valid(root, net)
        _check_deadline(effective_deadline, "before transaction commit")
        final_boxes_hash = _boxes_sha256(net, current)
        completed = time.monotonic()
        nonce = secrets.token_hex(32)
        body = {
            "schema": _SCHEMA,
            "status": "verified",
            "proof_authority": True,
            "authority_source": (
                (
                    "independent_outward_root_plus_operator_hz_bound_frame_"
                    "plus_independent_frozen_alpha_replay"
                )
                if initial_frame_audit["enabled"]
                else (
                    "independent_outward_root_plus_"
                    "independent_frozen_alpha_replay"
                )
            ),
            "ordinary_interval_facts_consumed": False,
            "transaction": "all_or_nothing",
            "process_local_identity_capability_required": True,
            "provenance_nonce_sha256": hashlib.sha256(
                nonce.encode("ascii")
            ).hexdigest(),
            "root_receipt_sha256": root.receipt["receipt_sha256"],
            "root_net_sha256": root.receipt["hashes"]["net_sha256"],
            "root_input_sha256": root.receipt["hashes"]["input_sha256"],
            "root_boxes_sha256": root_boxes_hash,
            "initial_preactivation_frame": initial_frame_audit,
            "final_boxes_sha256": final_boxes_hash,
            "target_relu_ids": list(targets),
            "stage_receipt_sha256": [
                stage.receipt["receipt_sha256"] for stage in stages
            ],
            "property_receipt_sha256": property_stage.receipt[
                "receipt_sha256"
            ],
            "property_spec_sha256": property_spec_hash,
            "property_upper_sha256": _array_digest(property_upper),
            "candidate_schema": _CANDIDATE_SCHEMA,
            "candidate_protocol": _CANDIDATE_PROTOCOL,
            "target_candidate_receipt_sha256": [
                stage.candidate_receipt["receipt_sha256"]
                for stage in stages
            ],
            "target_candidate_descriptor_coverage_sha256": [
                stage.candidate_receipt["descriptor_coverage_sha256"]
                for stage in stages
            ],
            "property_candidate_receipt_sha256": property_candidate.receipt[
                "receipt_sha256"
            ],
            "property_candidate_descriptor_coverage_sha256": (
                property_candidate.receipt["descriptor_coverage_sha256"]
            ),
            "candidate_generator": _callable_name(candidate_generator),
            "candidate_solver_factory": _callable_name(solver_factory),
            "root_certifier": _callable_name(_TRUSTED_CERTIFIER),
            "independent_replayer": _callable_name(_TRUSTED_REPLAYER),
            "property_replay_execution": (
                "root_owned_sealed_session_v1"
                if sealed_property_replay
                else "standalone_independent_replay_v2"
            ),
            "property_replay_workers": int(property_replay_workers),
            "steps": int(steps),
            "block_size": int(block_size),
            "replay_chunk_size": int(replay_chunk_size),
            "replay_max_workspace_bytes": int(replay_max_workspace_bytes),
            "conv_channel_chunk": int(conv_channel_chunk),
            "candidate_device": candidate_device,
            "dual_solver_default_device": str(default_candidate_device),
            "dual_solver_default_dtype": str(default_candidate_dtype),
            "candidate_device_fallback": False,
            "candidate_cuda_device_name": (
                torch.cuda.get_device_name(candidate_torch_device)
                if candidate_device == "cuda"
                else None
            ),
            # Callable names and device telemetry are forensic context only.
            # They never supply a number to the independently replayed proof.
            "non_authoritative_audit_fields": list(
                _PIPELINE_NON_AUTHORITATIVE_AUDIT_FIELDS
            ),
            "deadline_present": effective_deadline is not None,
            "deadline_monotonic_hex": (
                float(effective_deadline).hex()
                if effective_deadline is not None
                else None
            ),
            "started_monotonic_hex": float(started).hex(),
            "completed_monotonic_hex": float(completed).hex(),
            "completed_before_deadline": (
                effective_deadline is None or completed < effective_deadline
            ),
        }
        bundle = VerifiedQueryDualFeedback(
            root_certificate=root,
            certified_bounds=current,
            target_relu_ids=targets,
            stages=tuple(stages),
            property_stage=property_stage,
            property_upper=property_upper,
            receipt=_receipt(body),
            provenance_nonce=nonce,
        )
        if not _validate_bundle_contents(
            bundle,
            net=net,
            property_rows=rows,
            thresholds=threshold,
            expected_target_relu_ids=targets,
        ):
            _fail("INTERNAL_VALIDATION", "fresh transaction failed validation")
        _check_deadline(
            effective_deadline,
            "after transaction self-validation",
        )
        _register_live(bundle)
        return bundle
    except QueryDualPipelineError:
        raise
    except (QueryDualReplayTimeout, QueryDualBoxTimeout) as exc:
        raise QueryDualPipelineTimeout(str(exc)) from exc
    except (QueryDualReplayError, QueryDualBoxError) as exc:
        raise QueryDualPipelineError("INDEPENDENT_PROOF", str(exc)) from exc
    except Exception as exc:
        raise QueryDualPipelineError(
            "TRANSACTION_ABORTED", f"{type(exc).__name__}: {exc}"
        ) from exc


def _validate_replay_block(
    block: QueryDualAuthorityBlock,
    *,
    net: Any,
    parent_bounds: Mapping[int, Bounds],
    expected_objective: np.ndarray,
    expected_bias: np.ndarray,
    expected_kind: str,
    expected_target: Optional[int],
    expected_start: Optional[int],
    expected_rows: Tuple[int, ...],
    candidate_record: Mapping[str, Any],
) -> bool:
    try:
        expected_order = {
            "relu_unstable_plus_minus_one_hot": (
                "positive_rows_then_negated_rows"
            ),
            "final_property_negative_c_upper_only": (
                "negated_rows_only_for_property_upper_bounds"
            ),
            "final_property_c_minus_c": (
                "positive_rows_then_negated_rows"
            ),
        }.get(expected_kind)
        if (
            not isinstance(block, QueryDualAuthorityBlock)
            or block.query_kind != expected_kind
            or block.target_relu_lid != expected_target
            or block.start_lid != expected_start
            or block.row_ids != expected_rows
            or block.objective_sha256
            != _candidate_array_digest(expected_objective)
            or candidate_record.get("objective_sha256")
            != block.objective_sha256
            or candidate_record.get("alpha_sha256")
            != block.candidate_alpha_sha256
            or candidate_record.get("row_ids") != list(expected_rows)
            or candidate_record.get("query_kind") != expected_kind
            or candidate_record.get("start_lid") != expected_start
            or candidate_record.get("target_relu_lid") != expected_target
            or candidate_record.get("M")
            != int(expected_objective.shape[0])
            or candidate_record.get("objective_order") != expected_order
        ):
            return False
        by_id, preds = _layer_maps(net)
        output_lid = (
            _assert_output_id(by_id, preds)
            if expected_start is None
            else int(expected_start)
        )
        query_hash = _replay_query_sha256(
            expected_objective,
            expected_bias,
            expected_start,
            output_lid=output_lid,
        )
        bounds_hash = _replay_bounds_sha256(net, parent_bounds, expected_start)
        bridge_hash = _json_sha256(
            {
                "candidate_alpha_sha256": block.candidate_alpha_sha256,
                "replay_alpha_sha256": block.replay_alpha_sha256,
                "objective_sha256": block.objective_sha256,
                "replay_query_sha256": query_hash,
            }
        )
        if (
            block.replay_query_sha256 != query_hash
            or block.replay_bounds_sha256 != bounds_hash
            or block.alpha_bridge_sha256 != bridge_hash
        ):
            return False
        replay_result = QueryDualReplayResult(
            lower_bounds=block.lower_bounds,
            receipt=block.replay_receipt,
        )
        return validate_query_dual_replay_result(
            replay_result,
            expected_net_sha256=block.replay_net_sha256,
            expected_bounds_sha256=bounds_hash,
            expected_query_sha256=query_hash,
            expected_alpha_sha256=block.replay_alpha_sha256,
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        QueryDualPipelineError,
    ):
        return False


def _validate_bundle_contents(
    bundle: VerifiedQueryDualFeedback,
    *,
    net: Any,
    property_rows: Any,
    thresholds: Any,
    expected_target_relu_ids: Optional[Sequence[int]],
) -> bool:
    """Full deterministic validation excluding the live capability lookup."""

    try:
        if (
            not isinstance(bundle, VerifiedQueryDualFeedback)
            or bundle.proof_authority is not True
            or not verify_query_dual_box_certificate(
                bundle.root_certificate, net=net
            )
            or not _verify_receipt(bundle.receipt, _SCHEMA)
        ):
            return False
        rows, threshold = _normalise_property(property_rows, thresholds)
        by_id, preds = _layer_maps(net)
        output_lid = _assert_output_id(by_id, preds)
        expected_targets = (
            bundle.target_relu_ids
            if expected_target_relu_ids is None
            else _normalise_targets(expected_target_relu_ids, by_id)
        )
        if bundle.target_relu_ids != expected_targets:
            return False
        receipt = bundle.receipt
        initial_frame = receipt.get("initial_preactivation_frame")
        if not isinstance(initial_frame, Mapping):
            return False
        frame_enabled = initial_frame.get("enabled")
        if not isinstance(frame_enabled, bool):
            return False
        expected_authority_source = (
            (
                "independent_outward_root_plus_operator_hz_bound_frame_"
                "plus_independent_frozen_alpha_replay"
            )
            if frame_enabled
            else (
                "independent_outward_root_plus_"
                "independent_frozen_alpha_replay"
            )
        )
        if (
            receipt.get("status") != "verified"
            or receipt.get("proof_authority") is not True
            or receipt.get("authority_source")
            != expected_authority_source
            or receipt.get("transaction") != "all_or_nothing"
            or receipt.get("candidate_schema") != _CANDIDATE_SCHEMA
            or receipt.get("candidate_protocol") != _CANDIDATE_PROTOCOL
            or receipt.get("ordinary_interval_facts_consumed") is not False
            or receipt.get("process_local_identity_capability_required") is not True
            or receipt.get("candidate_device_fallback") is not False
            or receipt.get("candidate_device") not in {"cpu", "cuda"}
            or receipt.get("non_authoritative_audit_fields")
            != _PIPELINE_NON_AUTHORITATIVE_AUDIT_FIELDS
            or receipt.get("root_certifier")
            != _callable_name(_TRUSTED_CERTIFIER)
            or receipt.get("independent_replayer")
            != _callable_name(_TRUSTED_REPLAYER)
            or receipt.get("target_relu_ids") != list(expected_targets)
            or receipt.get("root_receipt_sha256")
            != bundle.root_certificate.receipt["receipt_sha256"]
            or receipt.get("root_net_sha256")
            != bundle.root_certificate.receipt["hashes"]["net_sha256"]
            or receipt.get("root_input_sha256")
            != bundle.root_certificate.receipt["hashes"]["input_sha256"]
            or receipt.get("conv_channel_chunk")
            != bundle.root_certificate.receipt.get("conv_channel_chunk")
            or receipt.get("provenance_nonce_sha256")
            != hashlib.sha256(bundle.provenance_nonce.encode("ascii")).hexdigest()
        ):
            return False
        for config_key in (
            "steps",
            "block_size",
            "replay_chunk_size",
            "replay_max_workspace_bytes",
            "conv_channel_chunk",
        ):
            config_value = receipt.get(config_key)
            if (
                isinstance(config_value, (bool, np.bool_))
                or not isinstance(config_value, (Integral, np.integer))
                or int(config_value) <= 0
            ):
                return False
        deadline_hex = receipt.get("deadline_monotonic_hex")
        started = float.fromhex(receipt["started_monotonic_hex"])
        completed = float.fromhex(receipt["completed_monotonic_hex"])
        if (
            not math.isfinite(started)
            or not math.isfinite(completed)
            or completed < started
        ):
            return False
        if deadline_hex is None:
            candidate_deadline = None
            if (
                receipt.get("deadline_present") is not False
                or receipt.get("completed_before_deadline") is not True
            ):
                return False
        else:
            deadline = float.fromhex(deadline_hex)
            candidate_deadline = deadline
            if (
                not math.isfinite(deadline)
                or receipt.get("deadline_present") is not True
                or receipt.get("completed_before_deadline")
                is not (completed < deadline)
                or completed >= deadline
            ):
                return False
        expected_property_replay_execution = (
            "root_owned_sealed_session_v1"
            if frame_enabled and candidate_deadline is not None
            else "standalone_independent_replay_v2"
        )
        if (
            receipt.get("property_replay_execution")
            != expected_property_replay_execution
        ):
            return False
        property_replay_workers = receipt.get("property_replay_workers")
        if (
            isinstance(property_replay_workers, bool)
            or not isinstance(property_replay_workers, int)
            or property_replay_workers <= 0
            or property_replay_workers > 32
            or (
                expected_property_replay_execution
                == "standalone_independent_replay_v2"
                and property_replay_workers != 1
            )
        ):
            return False
        current = _clone_bounds(bundle.root_certificate.bounds)
        root_hash = _boxes_sha256(net, current)
        if frame_enabled:
            if expected_targets or bundle.stages:
                return False
            source_layer_ids = initial_frame.get("source_layer_ids")
            expected_relu_ids = sorted(
                int(lid)
                for lid, layer in by_id.items()
                if _kind(layer) == "RELU"
            )
            if source_layer_ids != expected_relu_ids:
                return False
            if set(bundle.certified_bounds) != set(current):
                return False
            strict_lower = 0
            strict_upper = 0
            source_set = set(source_layer_ids)
            for lid in current:
                root_lower, root_upper = _flat_box(
                    current[lid], lid=lid
                )
                final_lower, final_upper = _flat_box(
                    bundle.certified_bounds[lid], lid=lid
                )
                if lid not in source_set:
                    if (
                        not np.array_equal(final_lower, root_lower)
                        or not np.array_equal(final_upper, root_upper)
                    ):
                        return False
                    continue
                if (
                    final_lower.shape != root_lower.shape
                    or final_upper.shape != root_upper.shape
                    or np.any(final_lower < root_lower)
                    or np.any(final_upper > root_upper)
                    or np.any(final_lower > final_upper)
                ):
                    return False
                strict_lower += int(
                    np.count_nonzero(final_lower > root_lower)
                )
                strict_upper += int(
                    np.count_nonzero(final_upper < root_upper)
                )
                _replace_box(
                    current, lid, final_lower, final_upper
                )
            source_hashes = (
                initial_frame.get("source_receipt_sha256"),
                initial_frame.get("source_bounds_sha256"),
                initial_frame.get("source_network_sha256"),
            )
            if (
                initial_frame.get("schema")
                != "query_dual_operator_hz_bound_frame_v1"
                or initial_frame.get("proof_authority") is not True
                or initial_frame.get("source")
                != "live_operator_hz_preactivation_frame"
                or initial_frame.get("source_network_sha256")
                != bundle.root_certificate.receipt["hashes"][
                    "net_sha256"
                ]
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                    for value in source_hashes
                )
                or initial_frame.get("strict_lower_rows")
                != strict_lower
                or initial_frame.get("strict_upper_rows")
                != strict_upper
                or initial_frame.get("intersection_only") is not True
                or initial_frame.get(
                    "target_replay_stages_required"
                )
                is not False
                or initial_frame.get("committed_boxes_sha256")
                != _boxes_sha256(net, current)
            ):
                return False
        elif (
            initial_frame
            != {
                "schema": "query_dual_operator_hz_bound_frame_v1",
                "enabled": False,
                "proof_authority": False,
                "source": "none",
                "source_layer_ids": [],
                "strict_lower_rows": 0,
                "strict_upper_rows": 0,
                "committed_boxes_sha256": root_hash,
                "intersection_only": True,
                "target_replay_stages_required": False,
            }
        ):
            return False
        if (
            receipt.get("root_boxes_sha256") != root_hash
            or len(bundle.stages) != len(expected_targets)
            or receipt.get("stage_receipt_sha256")
            != [stage.receipt["receipt_sha256"] for stage in bundle.stages]
            or receipt.get("target_candidate_receipt_sha256")
            != [
                stage.candidate_receipt["receipt_sha256"]
                for stage in bundle.stages
            ]
            or receipt.get(
                "target_candidate_descriptor_coverage_sha256"
            )
            != [
                stage.candidate_receipt["descriptor_coverage_sha256"]
                for stage in bundle.stages
            ]
        ):
            return False
        for stage_index, (target_lid, stage) in enumerate(
            zip(expected_targets, bundle.stages)
        ):
            if (
                not isinstance(stage, QueryDualTargetStage)
                or not _verify_receipt(stage.receipt, _STAGE_SCHEMA)
                or stage.stage_index != stage_index
                or stage.target_relu_lid != target_lid
                or stage.parent_boxes_sha256 != _boxes_sha256(net, current)
                or stage.candidate_bounds_sha256
                != _candidate_bounds_sha256(current)
                or not verify_query_dual_candidates_receipt(
                    stage.candidate_receipt
                )
                or not _candidate_v2_receipt_semantics(
                    stage.candidate_receipt
                )
            ):
                return False
            candidate_receipt = stage.candidate_receipt
            target_pred = preds[target_lid]
            if len(target_pred) != 1:
                return False
            predecessor = int(target_pred[0])
            predecessor_kind = _kind(by_id[predecessor])
            if (
                stage.predecessor_lid != predecessor
                or stage.predecessor_kind != predecessor_kind
                or candidate_receipt.get("input_bounds_sha256")
                != stage.candidate_bounds_sha256
                or candidate_receipt.get("target_relu_lid") != target_lid
                or candidate_receipt.get("property_rows") != 0
                or candidate_receipt.get("steps_requested")
                != receipt.get("steps")
                or candidate_receipt.get("block_size")
                != receipt.get("block_size")
                or candidate_receipt.get("deadline_monotonic")
                != candidate_deadline
                or stage.receipt.get("candidate_receipt_sha256")
                != candidate_receipt.get("receipt_sha256")
                or stage.receipt.get("candidate_schema")
                != candidate_receipt.get("schema")
                or stage.receipt.get("candidate_protocol")
                != candidate_receipt.get("protocol")
                or stage.receipt.get("candidate_status")
                != candidate_receipt.get("status")
                or stage.receipt.get(
                    "candidate_descriptor_coverage_sha256"
                )
                != candidate_receipt.get("descriptor_coverage_sha256")
            ):
                return False
            base_lower, base_upper = _flat_box(current[target_lid], lid=target_lid)
            lower = base_lower.copy()
            upper = base_upper.copy()
            if predecessor_kind != "RELU":
                pred_lower, pred_upper = _flat_box(
                    current[predecessor], lid=predecessor
                )
                lower = np.maximum(lower, pred_lower)
                upper = np.minimum(upper, pred_upper)
            pre_replay_lower = lower.copy()
            pre_replay_upper = upper.copy()
            unstable = tuple(
                int(value)
                for value in np.flatnonzero(
                    (base_lower < 0.0) & (base_upper > 0.0)
                )
            )
            if (
                candidate_receipt.get("target_start_lid") != predecessor
                or candidate_receipt.get("property_upper_only") is not True
                or candidate_receipt.get("target_width")
                != int(base_lower.size)
                or candidate_receipt.get("target_bounds_sha256")
                != _candidate_array_digest(
                    np.stack([base_lower, base_upper])
                )
                or candidate_receipt.get("property_rows_sha256") is not None
                or candidate_receipt.get("property_output_lid") is not None
                or candidate_receipt.get("property_baseline_sha256")
                != _candidate_array_digest(
                    np.zeros((2, 0), dtype=np.float64)
                )
                or candidate_receipt.get("property_lower_bound_source")
                != "not_requested"
                or candidate_receipt.get("property_upper_bound_source")
                != "not_requested"
                or candidate_receipt.get("unstable_target_neurons")
                != len(unstable)
                or candidate_receipt.get("planned_query_blocks")
                != len(stage.blocks)
            ):
                return False
            records = candidate_receipt.get("descriptor_records", ())
            if len(records) != len(stage.blocks):
                return False
            covered = []
            for block_index, block in enumerate(stage.blocks):
                block_rows = tuple(block.row_ids)
                expected_objective = _expected_target_objective(
                    block_rows, base_lower.size
                )
                if (
                    block.block_id != block_index
                    or any(value not in unstable for value in block_rows)
                    or not _validate_replay_block(
                        block,
                        net=net,
                        parent_bounds=current,
                        expected_objective=expected_objective,
                        expected_bias=np.zeros(
                            expected_objective.shape[0], dtype=np.float64
                        ),
                        expected_kind="relu_unstable_plus_minus_one_hot",
                        expected_target=target_lid,
                        expected_start=predecessor,
                        expected_rows=block_rows,
                        candidate_record=records[block_index],
                    )
                    or block.replay_receipt.get("requested_chunk_size")
                    != receipt.get("replay_chunk_size")
                    or block.replay_receipt.get("max_workspace_bytes")
                    != receipt.get("replay_max_workspace_bytes")
                ):
                    return False
                count = len(block_rows)
                raw_lower = block.lower_bounds[:count]
                raw_upper = -block.lower_bounds[count:]
                if block.lower_bounds.size != 2 * count or np.any(
                    raw_lower > raw_upper
                ):
                    return False
                index = np.asarray(block_rows, dtype=np.int64)
                lower[index] = np.maximum(lower[index], raw_lower)
                upper[index] = np.minimum(upper[index], raw_upper)
                if np.any(lower > upper):
                    return False
                covered.extend(block_rows)
            candidate_status = str(candidate_receipt.get("status", ""))
            if stage.blocks:
                if (
                    candidate_status != "descriptors_generated"
                    or tuple(covered) != unstable
                ):
                    return False
            elif (
                candidate_status != "no_queries_fallback"
                or unstable
            ):
                return False
            strict = int(
                np.count_nonzero(
                    (lower > pre_replay_lower)
                    | (upper < pre_replay_upper)
                )
            )
            synchronisation_strict = int(
                np.count_nonzero(
                    (pre_replay_lower > base_lower)
                    | (pre_replay_upper < base_upper)
                )
            )
            next_bounds = _clone_bounds(current)
            _replace_box(next_bounds, target_lid, lower, upper)
            if predecessor_kind != "RELU":
                _replace_box(next_bounds, predecessor, lower, upper)
            result_hash = _boxes_sha256(net, next_bounds)
            expected_stage_status = (
                "verified" if strict > 0 else "verified_no_improvement"
            )
            if (
                not np.array_equal(stage.target_lower, lower)
                or not np.array_equal(stage.target_upper, upper)
                or stage.strict_improvements != strict
                or stage.status != expected_stage_status
                or stage.receipt.get("status") != expected_stage_status
                or stage.receipt.get("proof_authority") is not True
                or stage.receipt.get("stage_index") != stage_index
                or stage.receipt.get("target_relu_lid") != target_lid
                or stage.receipt.get("predecessor_lid") != predecessor
                or stage.receipt.get("predecessor_kind")
                != predecessor_kind
                or stage.receipt.get("predecessor_synchronised")
                is not (predecessor_kind != "RELU")
                or stage.receipt.get("relu_key_semantics")
                != "preactivation"
                or stage.receipt.get("candidate_bounds_sha256")
                != stage.candidate_bounds_sha256
                or stage.receipt.get("strict_improvements") != strict
                or stage.receipt.get("commit") != "atomic_whole_stage"
                or stage.result_boxes_sha256 != result_hash
                or stage.receipt.get("result_boxes_sha256") != result_hash
                or stage.receipt.get("parent_boxes_sha256")
                != stage.parent_boxes_sha256
                or stage.receipt.get("target_bounds_sha256")
                != _array_digest(np.stack([lower, upper]))
                or stage.receipt.get("pre_replay_target_bounds_sha256")
                != _array_digest(
                    np.stack([pre_replay_lower, pre_replay_upper])
                )
                or stage.receipt.get(
                    "synchronisation_strict_improvements"
                )
                != synchronisation_strict
                or stage.receipt.get("block_receipt_sha256")
                != [
                    block.replay_receipt["receipt_sha256"]
                    for block in stage.blocks
                ]
                or stage.receipt.get("alpha_bridge_sha256")
                != [block.alpha_bridge_sha256 for block in stage.blocks]
            ):
                return False
            current = next_bounds
        if not _verify_receipt(bundle.property_stage.receipt, _PROPERTY_SCHEMA):
            return False
        property_stage = bundle.property_stage
        candidate_receipt = property_stage.candidate_receipt
        property_parent_hash = _boxes_sha256(net, current)
        property_candidate_hash = _candidate_bounds_sha256(current)
        spec_hash = _property_spec_sha256(rows, threshold)
        output_lower, output_upper = _flat_box(
            current[output_lid], lid=output_lid
        )
        positive_rows = np.maximum(rows, 0.0)
        negative_rows = np.minimum(rows, 0.0)
        baseline_lower = (
            positive_rows @ output_lower + negative_rows @ output_upper
        )
        baseline_upper = (
            positive_rows @ output_upper + negative_rows @ output_lower
        )
        property_baseline_hash = _candidate_array_digest(
            np.stack([baseline_lower, baseline_upper])
        )
        empty_target_hash = _candidate_array_digest(
            np.zeros((2, 0), dtype=np.float64)
        )
        if (
            property_stage.parent_boxes_sha256 != property_parent_hash
            or property_stage.candidate_bounds_sha256 != property_candidate_hash
            or property_stage.property_spec_sha256 != spec_hash
            or not verify_query_dual_candidates_receipt(candidate_receipt)
            or not _candidate_v2_receipt_semantics(candidate_receipt)
            or candidate_receipt.get("status") != "descriptors_generated"
            or candidate_receipt.get("input_bounds_sha256")
            != property_candidate_hash
            or candidate_receipt.get("target_relu_lid") is not None
            or candidate_receipt.get("target_start_lid") is not None
            or candidate_receipt.get("target_width") != 0
            or candidate_receipt.get("target_bounds_sha256")
            != empty_target_hash
            or candidate_receipt.get("property_upper_only") is not True
            or candidate_receipt.get("property_rows") != int(rows.shape[0])
            or candidate_receipt.get("property_output_lid") != output_lid
            or candidate_receipt.get("property_baseline_sha256")
            != property_baseline_hash
            or candidate_receipt.get("property_lower_bound_source")
            != "frozen_interval_baseline_not_dual_replayed"
            or candidate_receipt.get("property_upper_bound_source")
            != "baseline_placeholder_no_candidate_bound"
            or candidate_receipt.get("unstable_target_neurons") != 0
            or candidate_receipt.get("steps_requested")
            != receipt.get("steps")
            or candidate_receipt.get("block_size")
            != receipt.get("block_size")
            or candidate_receipt.get("deadline_monotonic")
            != candidate_deadline
            or candidate_receipt.get("planned_query_blocks")
            != len(property_stage.blocks)
            or candidate_receipt.get("property_rows_sha256")
            != _candidate_array_digest(rows)
            or property_stage.receipt.get("candidate_receipt_sha256")
            != candidate_receipt.get("receipt_sha256")
            or property_stage.receipt.get("candidate_schema")
            != candidate_receipt.get("schema")
            or property_stage.receipt.get("candidate_protocol")
            != candidate_receipt.get("protocol")
            or property_stage.receipt.get("candidate_status")
            != candidate_receipt.get("status")
            or property_stage.receipt.get(
                "candidate_descriptor_coverage_sha256"
            )
            != candidate_receipt.get("descriptor_coverage_sha256")
            or property_stage.receipt.get("status") != "verified"
            or property_stage.receipt.get("proof_authority") is not True
            or property_stage.receipt.get("direction") != "UPPER"
            or property_stage.receipt.get("quantity")
            != "C_y_minus_threshold"
            or property_stage.receipt.get("objective") != "-C"
            or property_stage.receipt.get("replay_query_bias")
            != "+threshold"
            or property_stage.receipt.get("upper_reconstruction")
            != "-LB(-C_y+threshold)"
            or property_stage.receipt.get("candidate_bounds_sha256")
            != property_candidate_hash
            or property_stage.receipt.get("property_rows")
            != int(rows.shape[0])
            or property_stage.receipt.get("coverage_complete") is not True
            or property_stage.receipt.get("replay_execution")
            != expected_property_replay_execution
            or property_stage.receipt.get("proof_workers_requested")
            != property_replay_workers
        ):
            return False
        records = candidate_receipt.get("descriptor_records", ())
        if len(records) != len(property_stage.blocks):
            return False
        upper = np.empty(rows.shape[0], dtype=np.float64)
        covered = []
        for block_index, block in enumerate(property_stage.blocks):
            row_ids = tuple(block.row_ids)
            index = np.asarray(row_ids, dtype=np.int64)
            if (
                block.block_id != block_index
                or not row_ids
                or np.any(index < 0)
                or np.any(index >= rows.shape[0])
            ):
                return False
            objective = -rows[index]
            bias = threshold[index]
            if not _validate_replay_block(
                block,
                net=net,
                parent_bounds=current,
                expected_objective=objective,
                expected_bias=bias,
                expected_kind="final_property_negative_c_upper_only",
                expected_target=None,
                expected_start=None,
                expected_rows=row_ids,
                candidate_record=records[block_index],
            ) or (
                block.replay_receipt.get("requested_chunk_size")
                != receipt.get("replay_chunk_size")
                or block.replay_receipt.get("max_workspace_bytes")
                != receipt.get("replay_max_workspace_bytes")
            ):
                return False
            parallelism = block.replay_receipt.get(
                "proof_row_parallelism"
            )
            if (
                not isinstance(parallelism, Mapping)
                or parallelism.get("protocol")
                != "disjoint_objective_rows_v1"
                or parallelism.get("requested_workers")
                != property_replay_workers
                or isinstance(
                    parallelism.get("effective_workers"), bool
                )
                or not isinstance(
                    parallelism.get("effective_workers"), int
                )
                or parallelism.get("effective_workers") <= 0
                or parallelism.get("effective_workers")
                > min(property_replay_workers, len(row_ids))
                or parallelism.get("partial_authority") is not False
            ):
                return False
            upper[index] = -block.lower_bounds
            covered.extend(row_ids)
        if tuple(covered) != tuple(range(rows.shape[0])):
            return False
        final_hash = _boxes_sha256(net, current)
        return bool(
            np.array_equal(property_stage.property_upper, upper)
            and np.array_equal(bundle.property_upper, upper)
            and _boxes_sha256(net, bundle.certified_bounds) == final_hash
            and all(
                np.array_equal(
                    _flat_box(bundle.certified_bounds[lid], lid=lid)[0],
                    _flat_box(current[lid], lid=lid)[0],
                )
                and np.array_equal(
                    _flat_box(bundle.certified_bounds[lid], lid=lid)[1],
                    _flat_box(current[lid], lid=lid)[1],
                )
                for lid in current
            )
            and set(bundle.certified_bounds) == set(current)
            and property_stage.receipt.get("parent_boxes_sha256")
            == property_parent_hash
            and property_stage.receipt.get("property_spec_sha256") == spec_hash
            and property_stage.receipt.get("property_upper_sha256")
            == _array_digest(upper)
            and property_stage.receipt.get("block_receipt_sha256")
            == [
                block.replay_receipt["receipt_sha256"]
                for block in property_stage.blocks
            ]
            and property_stage.receipt.get("alpha_bridge_sha256")
            == [block.alpha_bridge_sha256 for block in property_stage.blocks]
            and receipt.get("final_boxes_sha256") == final_hash
            and receipt.get("property_receipt_sha256")
            == property_stage.receipt["receipt_sha256"]
            and receipt.get("property_spec_sha256") == spec_hash
            and receipt.get("property_upper_sha256") == _array_digest(upper)
            and receipt.get("property_candidate_receipt_sha256")
            == candidate_receipt["receipt_sha256"]
            and receipt.get(
                "property_candidate_descriptor_coverage_sha256"
            )
            == candidate_receipt["descriptor_coverage_sha256"]
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        QueryDualPipelineError,
    ):
        return False


def validate_verified_query_dual_feedback(
    bundle: VerifiedQueryDualFeedback,
    *,
    net: Any,
    property_rows: Any,
    thresholds: Any,
    expected_target_relu_ids: Optional[Sequence[int]] = None,
    require_live_provenance: bool = True,
) -> bool:
    """Validate a live bundle before every downstream authority use.

    ``require_live_provenance`` defaults to true and production consumers must
    never disable it.  The opt-out exists only for deterministic audit tooling
    that wants to distinguish content corruption from capability rejection;
    it does not restore proof authority to a reconstructed object.
    """

    try:
        if (
            isinstance(bundle, VerifiedQueryDualFeedback)
            and isinstance(bundle.receipt, Mapping)
            and bundle.receipt.get("schema")
            == "act.verified_query_dual_feedback.v3"
        ):
            # Lazy import avoids a module cycle: V3 deliberately reuses the
            # frozen V2 dataclasses and process-local capability registry,
            # while keeping both receipt protocols independently validated.
            from act.back_end.hybridz_tf.query_dual_pipeline_v3 import (
                validate_verified_query_dual_feedback_v3,
            )

            return validate_verified_query_dual_feedback_v3(
                bundle,
                net=net,
                property_rows=property_rows,
                thresholds=thresholds,
                expected_target_relu_ids=expected_target_relu_ids,
                require_live_provenance=require_live_provenance,
            )
        if require_live_provenance and not _has_live_capability(bundle):
            return False
        return _validate_bundle_contents(
            bundle,
            net=net,
            property_rows=property_rows,
            thresholds=thresholds,
            expected_target_relu_ids=expected_target_relu_ids,
        )
    except (AttributeError, TypeError, ValueError):
        return False


__all__ = [
    "QueryDualAuthorityBlock",
    "QueryDualPipelineError",
    "QueryDualPipelineTimeout",
    "QueryDualPropertyStage",
    "QueryDualTargetStage",
    "VerifiedQueryDualFeedback",
    "build_verified_query_dual_feedback",
    "validate_verified_query_dual_feedback",
]
