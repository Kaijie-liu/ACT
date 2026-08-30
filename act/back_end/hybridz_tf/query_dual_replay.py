#===- query_dual_replay.py - Independent HybridZ proof replay ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
"""Independent, fail-closed lower-bound replay for a small ACT operator set.

This module deliberately does *not* import or call ``dual_tf`` or
``solver_dual``.  It consumes a frozen set of already-certified boxes and a
stored binary64 alpha candidate, then reconstructs a CROWN-style lower bound
from first principles.

Authority boundary
------------------
The replay proves the query relative to the supplied boxes.  It does not prove
that those boxes were produced soundly; their canonical SHA-256 is therefore a
required part of every receipt, and callers may pin it through
``expected_bounds_sha256``.  Candidate generation has no authority.  Only a
successfully completed replay result carries ``proof_authority=True``.

Numerics
--------
All nominal arithmetic is CPU binary64.  Affine coefficient roundoff is
bounded componentwise with a conservative Higham-gamma model (including an
absolute subnormal allowance), then immediately absorbed into the certified
box of the predecessor as a downward scalar guard.  Ambiguous ReLU upper
lines are built by an exact :class:`fractions.Fraction` endpoint audit and an
outward-rounded intercept.

The first version intentionally supports only:

``INPUT / INPUT_SPEC / CONV2D / RELU / ADD / FLATTEN / DENSE / ASSERT``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import platform
import secrets
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch

from act.back_end.hybridz_tf.query_dual_box_certifier import (
    QueryDualBoxCertificate,
    QueryDualBoxError,
    _borrow_sealed_query_dual_graph,
    verify_query_dual_box_certificate,
)


_SCHEMA = "act.query_dual_replay.v1"
_SEALED_SCHEMA = "act.query_dual_replay.v2"
_AFFINE_SCHEMA = "act.query_dual_affine_lower_plane.v1"
_SEALED_PROTOCOL = "frozen_union_context_v3"
_SUPPORTED = frozenset(
    {
        "INPUT",
        "INPUT_SPEC",
        "CONV2D",
        "RELU",
        "ADD",
        "FLATTEN",
        "DENSE",
        "ASSERT",
    }
)
_U = float(2.0**-53)
_ETA = float(np.nextafter(np.float64(0.0), np.float64(math.inf)))
_SEALED_SESSION_CAPABILITY = object()
_SEALED_SESSION_REGISTRY: weakref.WeakValueDictionary[
    str, "QueryDualReplaySession"
] = weakref.WeakValueDictionary()


class QueryDualReplayError(RuntimeError):
    """Fail-closed replay error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


class QueryDualReplayTimeout(QueryDualReplayError):
    """The replay deadline expired before an authoritative result existed."""

    def __init__(self, message: str = "query-dual replay deadline expired"):
        super().__init__("DEADLINE_EXPIRED", message)


@dataclass(frozen=True)
class QueryDualReplayResult:
    """Authoritative output of a completed independent replay."""

    lower_bounds: np.ndarray
    receipt: Mapping[str, Any]
    proof_authority: bool = True

    def __post_init__(self) -> None:
        if not self.proof_authority:
            raise ValueError("a completed replay result must be authoritative")


@dataclass(frozen=True)
class QueryDualAffineLowerPlane:
    """Certified affine lower inequality at one dominating interior layer.

    For each query row ``r`` the replay proves

    ``scalar[r] + coefficients[r] @ y_stop <= query_expression[r]``

    over the supplied certified bounds.  It does not itself optimize the
    affine left-hand side over the prefix set.
    """

    coefficients: np.ndarray
    scalar: np.ndarray
    receipt: Mapping[str, Any]
    proof_authority: bool = True

    def __post_init__(self) -> None:
        if not self.proof_authority:
            raise ValueError("a completed affine replay must be authoritative")


@dataclass(frozen=True)
class QueryDualReplayPendingResult:
    """Non-authoritative stage output usable only inside a V3 transaction."""

    lower_bounds: np.ndarray
    stage_token: str
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("a pending replay result cannot be authoritative")


@dataclass(frozen=True)
class _Box:
    lb: np.ndarray
    ub: np.ndarray


@dataclass(frozen=True)
class _FrozenLayer:
    id: int
    kind: str
    preds: Tuple[int, ...]
    width: int
    in_vars: Tuple[Any, ...]
    out_vars: Tuple[Any, ...]
    params: Mapping[str, Any]


@dataclass
class _ReplayStats:
    coefficient_guards: int = 0
    scalar_guards: int = 0
    fraction_endpoint_audits: int = 0
    relu_ambiguous_terms: int = 0
    affine_terms: int = 0
    dag_merges: int = 0
    conv_sparse_blocks: int = 0
    conv_dense_blocks: int = 0
    guard_total: float = 0.0
    guard_max: float = 0.0
    guard_by_query: Optional[np.ndarray] = None
    active_start: int = 0
    active_end: int = 0

    def configure_queries(self, query_count: int) -> None:
        self.guard_by_query = np.zeros(int(query_count), dtype=np.float64)

    def begin_block(self, start: int, end: int) -> None:
        self.active_start = int(start)
        self.active_end = int(end)

    def record_guard(self, value: Any, *, coefficient: bool) -> None:
        array = np.asarray(value, dtype=np.float64)
        if np.any(array < 0.0) or not np.all(np.isfinite(array)):
            raise QueryDualReplayError("NUMERIC_GUARD", "invalid roundoff guard")
        if coefficient:
            self.coefficient_guards += int(array.size)
        else:
            self.scalar_guards += int(array.size)
        self.guard_total = float(self.guard_total + float(np.sum(array)))
        self.guard_max = max(
            self.guard_max, float(np.max(array)) if array.size else 0.0
        )
        if (
            self.guard_by_query is not None
            and array.ndim == 1
            and array.size == self.active_end - self.active_start
        ):
            current = self.guard_by_query[self.active_start : self.active_end]
            self.guard_by_query[self.active_start : self.active_end] = (
                _upper_nonnegative_sum(current, array)
            )
        if not math.isfinite(self.guard_total):
            raise QueryDualReplayError("NUMERIC_GUARD", "guard statistics overflow")


@dataclass
class _Deadline:
    end: Optional[float]
    counter: int = 0

    @classmethod
    def build(
        cls,
        deadline: Optional[float],
        timeout_s: Optional[float],
    ) -> "_Deadline":
        now = time.monotonic()
        ends: List[float] = []
        if deadline is not None:
            d = float(deadline)
            if not math.isfinite(d):
                raise QueryDualReplayError("INVALID_DEADLINE", "deadline must be finite")
            ends.append(d)
        if timeout_s is not None:
            t = float(timeout_s)
            if not math.isfinite(t) or t < 0.0:
                raise QueryDualReplayError(
                    "INVALID_DEADLINE", "timeout_s must be finite and nonnegative"
                )
            ends.append(now + t)
        return cls(min(ends) if ends else None)

    def check(self, *, force: bool = False) -> None:
        self.counter += 1
        if not force and (self.counter & 1023) != 0:
            return
        if self.end is not None and time.monotonic() >= self.end:
            raise QueryDualReplayTimeout()


@dataclass
class _Prepared:
    layers: Mapping[int, _FrozenLayer]
    reverse_order: Tuple[int, ...]
    output_id: int
    output_width: int
    start_mode: str
    input_spec_id: int
    bounds: Mapping[int, _Box]
    queries: np.ndarray
    query_bias: np.ndarray
    alpha: Mapping[int, np.ndarray]
    hashes: Mapping[str, str]
    deadline: _Deadline
    relu_lines: MutableMapping[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _SealedCone:
    start_lid: Optional[int]
    layers: Mapping[int, _FrozenLayer]
    reverse_order: Tuple[int, ...]
    output_id: int
    output_width: int
    start_mode: str
    input_spec_id: int
    replay_net_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class _SealedReplayValidationContext:
    """Recomputed root/crosswalk semantics for a V3 authority validator."""

    full_layers: Mapping[int, _FrozenLayer]
    contexts: Mapping[Optional[int], _SealedCone]
    crosswalk: Mapping[str, Any]


@dataclass(frozen=True)
class QueryDualReplayBoundsFrame:
    """Opaque immutable stage-bounds snapshot owned by one V3 session."""

    _session_nonce: str = field(repr=False)
    _frame_nonce: str = field(repr=False)
    _bounds: Mapping[int, _Box] = field(repr=False)
    _start_lids: Tuple[Optional[int], ...] = field(repr=False)
    _content_sha256: str = field(repr=False)
    _capability: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class _PendingStage:
    public: QueryDualReplayPendingResult
    prepared: _Prepared
    base_receipt: Mapping[str, Any]
    frame_nonce: str
    frame_sha256: str
    content_sha256: str


def _kind(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).upper()


def _fail(code: str, message: str) -> NoReturn:
    raise QueryDualReplayError(code, message)


def _as_f64_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise QueryDualReplayError("INVALID_NUMERIC", f"{name}: {exc}") from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64).copy()
    if not np.all(np.isfinite(arr)):
        _fail("NONFINITE", f"{name} contains NaN or infinity")
    arr.setflags(write=False)
    return arr


def _immutable_f64_array(value: Any, *, name: str) -> np.ndarray:
    """Binary64 array whose immutable-bytes backing cannot be made writable."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise QueryDualReplayError("INVALID_NUMERIC", f"{name}: {exc}") from exc
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        _fail("NONFINITE", f"{name} contains NaN or infinity")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )
    frozen.setflags(write=False)
    return frozen


def _as_stored_f64_alpha(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float64:
            _fail("ALPHA_NOT_F64", f"{name} must be stored as torch.float64")
        raw = value.detach().cpu().numpy()
    else:
        raw = np.asarray(value)
        if raw.dtype.kind == "f" and raw.dtype != np.float64:
            _fail("ALPHA_NOT_F64", f"{name} must be stored as numpy.float64")
    arr = _as_f64_array(raw, name=name)
    if arr.size == 0:
        _fail("INVALID_ALPHA", f"{name} is empty")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        _fail("INVALID_ALPHA", f"{name} has values outside [0, 1]")
    return arr


def _array_digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype="<f8")
    h = hashlib.sha256()
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii"))
    h.update(b"\0<f8\0")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _manifest_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("NONFINITE", "network scalar parameter is non-finite")
        return {"float_hex": float(value).hex()}
    if isinstance(value, np.generic):
        return _manifest_scalar(value.item())
    if isinstance(value, (tuple, list)):
        return [_manifest_scalar(v) for v in value]
    return str(value)


def _box_from_value(value: Any, *, layer_id: int) -> _Box:
    if hasattr(value, "bounds"):
        value = value.bounds
    if isinstance(value, Mapping):
        lb_raw, ub_raw = value.get("lb"), value.get("ub")
    else:
        lb_raw, ub_raw = getattr(value, "lb", None), getattr(value, "ub", None)
        if lb_raw is None and isinstance(value, (tuple, list)) and len(value) == 2:
            lb_raw, ub_raw = value
    if lb_raw is None or ub_raw is None:
        _fail("MISSING_BOUNDS", f"layer {layer_id} has no lb/ub pair")
    lb = _as_f64_array(lb_raw, name=f"bounds[{layer_id}].lb").reshape(-1)
    ub = _as_f64_array(ub_raw, name=f"bounds[{layer_id}].ub").reshape(-1)
    lb = np.ascontiguousarray(lb)
    ub = np.ascontiguousarray(ub)
    if lb.shape != ub.shape:
        _fail("INVALID_BOUNDS", f"layer {layer_id} lb/ub shapes differ")
    if np.any(lb > ub):
        _fail("INVALID_BOUNDS", f"layer {layer_id} has lb > ub")
    lb.setflags(write=False)
    ub.setflags(write=False)
    return _Box(lb=lb, ub=ub)


def _immutable_box_from_value(value: Any, *, layer_id: int) -> _Box:
    if hasattr(value, "bounds"):
        value = value.bounds
    if isinstance(value, Mapping):
        lb_raw, ub_raw = value.get("lb"), value.get("ub")
    else:
        lb_raw, ub_raw = getattr(value, "lb", None), getattr(value, "ub", None)
        if lb_raw is None and isinstance(value, (tuple, list)) and len(value) == 2:
            lb_raw, ub_raw = value
    if lb_raw is None or ub_raw is None:
        _fail("MISSING_BOUNDS", f"layer {layer_id} has no lb/ub pair")
    lb = _immutable_f64_array(
        lb_raw, name=f"sealed_bounds[{layer_id}].lb"
    ).reshape(-1)
    ub = _immutable_f64_array(
        ub_raw, name=f"sealed_bounds[{layer_id}].ub"
    ).reshape(-1)
    if lb.shape != ub.shape:
        _fail("INVALID_BOUNDS", f"layer {layer_id} lb/ub shapes differ")
    if np.any(lb > ub):
        _fail("INVALID_BOUNDS", f"layer {layer_id} has lb > ub")
    return _Box(lb=lb, ub=ub)


def _pair(value: Any, *, name: str, allow_zero: bool) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            _fail("INVALID_CONV", f"{name} must be an int or length-2 sequence")
        pair = (int(value[0]), int(value[1]))
    else:
        pair = (int(value), int(value))
    lower = 0 if allow_zero else 1
    if pair[0] < lower or pair[1] < lower:
        _fail("INVALID_CONV", f"{name} entries must be >= {lower}")
    return pair


def _shape3(value: Any, *, name: str) -> Tuple[int, int, int]:
    if value is None:
        _fail("INVALID_CONV", f"{name} is required")
    shape = tuple(int(v) for v in value)
    if len(shape) == 4:
        if shape[0] != 1:
            _fail("BATCH_UNSUPPORTED", f"{name} batch must be one")
        shape = shape[1:]
    if len(shape) != 3 or any(v <= 0 for v in shape):
        _fail("INVALID_CONV", f"{name} must be CHW or 1xCHW")
    return shape


def _broadcast_bias(value: Any, width: int, *, name: str) -> np.ndarray:
    if value is None:
        out = np.zeros(width, dtype=np.float64)
    else:
        raw = _as_f64_array(value, name=name).reshape(-1)
        if raw.size == 1:
            out = np.full(width, float(raw[0]), dtype=np.float64)
        elif raw.size == width:
            out = np.ascontiguousarray(raw.copy())
        else:
            _fail(
                "SHAPE_MISMATCH",
                f"{name} has {raw.size} entries; expected one or {width}",
            )
    out.setflags(write=False)
    return out


def _freeze_layer(layer: Any, preds: Tuple[int, ...]) -> _FrozenLayer:
    lid = int(layer.id)
    kind = _kind(layer.kind)
    if kind not in _SUPPORTED:
        _fail("UNSUPPORTED_OPERATOR", f"layer {lid} kind {kind}")
    in_vars = tuple(getattr(layer, "in_vars", ()) or ())
    out_vars = tuple(getattr(layer, "out_vars", ()) or ())
    width = len(out_vars)
    raw_params = getattr(layer, "params", {}) or {}
    params: Dict[str, Any] = {}

    if kind == "DENSE":
        weight = _as_f64_array(raw_params.get("weight"), name=f"layer {lid} weight")
        if weight.ndim != 2:
            _fail("SHAPE_MISMATCH", f"DENSE layer {lid} weight must be rank two")
        if width == 0:
            width = int(weight.shape[0])
        if weight.shape[0] != width:
            _fail("SHAPE_MISMATCH", f"DENSE layer {lid} output width mismatch")
        params["weight"] = weight
        params["bias"] = _broadcast_bias(
            raw_params.get("bias"), width, name=f"layer {lid} bias"
        )
    elif kind == "CONV2D":
        weight = _as_f64_array(raw_params.get("weight"), name=f"layer {lid} weight")
        if weight.ndim != 4:
            _fail(
                "SHAPE_MISMATCH",
                f"CONV2D layer {lid} weight must be [out,in/groups,kh,kw]",
            )
        input_shape = _shape3(raw_params.get("input_shape"), name="input_shape")
        output_shape = _shape3(raw_params.get("output_shape"), name="output_shape")
        groups = int(raw_params.get("groups", 1))
        if groups <= 0 or weight.shape[0] % groups:
            _fail("INVALID_CONV", f"CONV2D layer {lid} has invalid groups")
        if input_shape[0] != weight.shape[1] * groups:
            _fail("INVALID_CONV", f"CONV2D layer {lid} input channels mismatch")
        if output_shape[0] != weight.shape[0]:
            _fail("INVALID_CONV", f"CONV2D layer {lid} output channels mismatch")
        stride = _pair(
            raw_params.get("stride", 1), name="stride", allow_zero=False
        )
        padding = _pair(
            raw_params.get("padding", 0), name="padding", allow_zero=True
        )
        dilation = _pair(
            raw_params.get("dilation", 1), name="dilation", allow_zero=False
        )
        expected_h = (
            input_shape[1]
            + 2 * padding[0]
            - dilation[0] * (int(weight.shape[2]) - 1)
            - 1
        ) // stride[0] + 1
        expected_w = (
            input_shape[2]
            + 2 * padding[1]
            - dilation[1] * (int(weight.shape[3]) - 1)
            - 1
        ) // stride[1] + 1
        if output_shape[1:] != (expected_h, expected_w):
            _fail(
                "INVALID_CONV",
                f"CONV2D layer {lid} declared output {output_shape[1:]} "
                f"!= formula {(expected_h, expected_w)}",
            )
        if width == 0:
            width = int(np.prod(output_shape))
        if width != int(np.prod(output_shape)):
            _fail("SHAPE_MISMATCH", f"CONV2D layer {lid} output width mismatch")
        if bool(raw_params.get("transposed", False)):
            _fail("UNSUPPORTED_OPERATOR", f"layer {lid} is a transposed convolution")
        if str(raw_params.get("padding_mode", "zeros")).lower() not in {"zeros", "zero"}:
            _fail("UNSUPPORTED_OPERATOR", f"layer {lid} has nonzero padding mode")
        params.update(
            {
                "weight": weight,
                "bias_channels": _broadcast_bias(
                    raw_params.get("bias"),
                    int(weight.shape[0]),
                    name=f"layer {lid} bias",
                ),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }
        )
    elif kind == "ADD":
        if width <= 0:
            _fail("SHAPE_MISMATCH", f"ADD layer {lid} has no output variables")
        add_bias = _broadcast_bias(
            raw_params.get("bias"), width, name=f"layer {lid} bias"
        )
        if np.any(add_bias != 0.0):
            _fail(
                "UNSUPPORTED_OPERATOR",
                f"ADD layer {lid} has a nonzero bias, but ACT ADD semantics "
                "are an unbiased sum",
            )
        params["bias"] = add_bias
    elif kind == "INPUT":
        if width <= 0:
            raw_shape = tuple(int(v) for v in raw_params.get("shape", ()))
            if raw_shape:
                width = int(np.prod(raw_shape[1:] if len(raw_shape) > 1 else raw_shape))
        params["shape"] = tuple(int(v) for v in raw_params.get("shape", ()))
        params["dtype"] = str(raw_params.get("dtype", ""))
    elif kind == "INPUT_SPEC":
        if width <= 0:
            lb = raw_params.get("lb")
            if lb is not None:
                width = int(np.asarray(lb).reshape(-1).size)
        params["kind"] = str(raw_params.get("kind", ""))
    elif kind == "ASSERT":
        params["kind"] = str(raw_params.get("kind", ""))
    elif kind == "FLATTEN":
        params["start_dim"] = int(raw_params.get("start_dim", 1))
        params["end_dim"] = int(raw_params.get("end_dim", -1))
        params["input_shape"] = _manifest_scalar(raw_params.get("input_shape"))
        params["output_shape"] = _manifest_scalar(raw_params.get("output_shape"))

    if width <= 0:
        _fail("SHAPE_MISMATCH", f"layer {lid} has zero output width")
    return _FrozenLayer(
        id=lid,
        kind=kind,
        preds=preds,
        width=width,
        in_vars=in_vars,
        out_vars=out_vars,
        params=params,
    )


def _topology(
    net: Any,
    start_lid: Optional[int],
) -> Tuple[List[Any], Dict[int, Tuple[int, ...]], int, Tuple[int, ...], str]:
    layers = list(getattr(net, "layers", ()) or ())
    if not layers:
        _fail("INVALID_GRAPH", "network has no layers")
    by_id = {int(layer.id): layer for layer in layers}
    if len(by_id) != len(layers):
        _fail("INVALID_GRAPH", "layer ids are not unique")
    raw_preds = getattr(net, "preds", {}) or {}
    preds: Dict[int, Tuple[int, ...]] = {}
    for lid in by_id:
        try:
            preds[lid] = tuple(int(p) for p in (raw_preds.get(lid, ()) or ()))
        except Exception as exc:
            raise QueryDualReplayError(
                "INVALID_GRAPH", f"layer {lid} has malformed predecessors"
            ) from exc

    if start_lid is None:
        assertions = [
            lid for lid, layer in by_id.items() if _kind(layer.kind) == "ASSERT"
        ]
        if len(assertions) != 1:
            _fail("INVALID_GRAPH", "exactly one ASSERT layer is required")
        assert_id = assertions[0]
        if len(preds[assert_id]) != 1:
            _fail("INVALID_GRAPH", "ASSERT must have exactly one predecessor")
        output_id = preds[assert_id][0]
        if output_id not in by_id:
            _fail("INVALID_GRAPH", "ASSERT references an unknown predecessor")
        start_mode = "ASSERT_PREDECESSOR"
    else:
        try:
            output_id = int(start_lid)
        except Exception as exc:
            raise QueryDualReplayError(
                "INVALID_START_LAYER", f"invalid start_lid {start_lid!r}"
            ) from exc
        if output_id not in by_id:
            _fail("INVALID_START_LAYER", f"unknown start_lid {output_id}")
        if _kind(by_id[output_id].kind) == "ASSERT":
            _fail("INVALID_START_LAYER", "start_lid cannot be ASSERT")
        start_mode = "EXPLICIT_INTERIOR"

    state: Dict[int, int] = {}
    topo: List[int] = []

    def visit(lid: int) -> None:
        mark = state.get(lid, 0)
        if mark == 1:
            _fail("INVALID_GRAPH", "cycle in reachable proof graph")
        if mark == 2:
            return
        if len(set(preds[lid])) != len(preds[lid]):
            _fail("INVALID_GRAPH", f"layer {lid} repeats a predecessor")
        if any(parent not in by_id for parent in preds[lid]):
            _fail(
                "INVALID_GRAPH",
                f"reachable layer {lid} references an unknown predecessor",
            )
        state[lid] = 1
        for parent in preds[lid]:
            visit(parent)
        state[lid] = 2
        topo.append(lid)

    visit(output_id)
    reachable = set(topo)
    input_specs = [
        lid for lid in topo if _kind(by_id[lid].kind) == "INPUT_SPEC"
    ]
    if len(input_specs) != 1:
        _fail("INVALID_GRAPH", "reachable proof graph needs exactly one INPUT_SPEC")
    for lid in topo:
        kind = _kind(by_id[lid].kind)
        if kind not in _SUPPORTED:
            _fail("UNSUPPORTED_OPERATOR", f"layer {lid} kind {kind}")
        if kind == "INPUT" and preds[lid]:
            _fail("INVALID_GRAPH", f"INPUT layer {lid} must be a source")
        if kind == "INPUT_SPEC":
            if len(preds[lid]) != 1 or _kind(by_id[preds[lid][0]].kind) != "INPUT":
                _fail("INVALID_GRAPH", f"INPUT_SPEC layer {lid} must follow INPUT")
        elif kind in {"DENSE", "CONV2D", "RELU", "FLATTEN"}:
            if len(preds[lid]) != 1:
                _fail("INVALID_GRAPH", f"{kind} layer {lid} needs one predecessor")
        elif kind == "ADD" and len(preds[lid]) != 2:
            _fail(
                "UNSUPPORTED_OPERATOR",
                f"ADD layer {lid} must have exactly two predecessors in V1",
            )
    if any(_kind(by_id[lid].kind) == "ASSERT" for lid in reachable):
        _fail("INVALID_GRAPH", "ASSERT cannot be inside the replay ancestor cone")
    return layers, preds, output_id, tuple(reversed(topo)), start_mode


def _layer_manifest(layer: _FrozenLayer) -> Mapping[str, Any]:
    params: Dict[str, Any] = {}
    for key, value in sorted(layer.params.items()):
        if isinstance(value, np.ndarray):
            params[key] = {"shape": list(value.shape), "sha256": _array_digest(value)}
        else:
            params[key] = _manifest_scalar(value)
    return {
        "id": layer.id,
        "kind": layer.kind,
        "preds": list(layer.preds),
        "width": layer.width,
        "in_vars": [_manifest_scalar(v) for v in layer.in_vars],
        "out_vars": [_manifest_scalar(v) for v in layer.out_vars],
        "params": params,
    }


def _normalise_queries(
    output_width: int,
    query_rows: Optional[Any],
    one_hot: Optional[Any],
    query_bias: Optional[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    if (query_rows is None) == (one_hot is None):
        _fail("INVALID_QUERY", "provide exactly one of query_rows or one_hot")
    if query_rows is not None:
        rows = _as_f64_array(query_rows, name="query_rows")
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != output_width or rows.shape[0] == 0:
            _fail(
                "SHAPE_MISMATCH",
                f"query_rows must have shape [Q,{output_width}]",
            )
    else:
        signs: Optional[Any] = None
        if isinstance(one_hot, Mapping):
            indices = one_hot.get("indices", one_hot.get("index"))
            signs = one_hot.get("signs", one_hot.get("sign", 1.0))
        else:
            indices = one_hot
        if isinstance(indices, (int, np.integer)):
            index_array = np.asarray([int(indices)], dtype=np.int64)
        else:
            index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        if index_array.size == 0:
            _fail("INVALID_QUERY", "one_hot descriptor is empty")
        if np.any(index_array < 0) or np.any(index_array >= output_width):
            _fail("INVALID_QUERY", "one_hot index is out of range")
        if signs is None:
            sign_array = np.ones(index_array.size, dtype=np.float64)
        else:
            sign_array = np.asarray(signs, dtype=np.float64)
            if sign_array.ndim == 0:
                sign_array = np.full(index_array.size, float(sign_array))
            sign_array = sign_array.reshape(-1)
            if sign_array.size != index_array.size:
                _fail("SHAPE_MISMATCH", "one_hot signs do not match indices")
        if not np.all(np.isfinite(sign_array)) or np.any(sign_array == 0.0):
            _fail("INVALID_QUERY", "one_hot signs must be finite and nonzero")
        rows = np.zeros((index_array.size, output_width), dtype=np.float64)
        rows[np.arange(index_array.size), index_array] = sign_array

    rows = np.ascontiguousarray(rows, dtype=np.float64)
    if query_bias is None:
        biases = np.zeros(rows.shape[0], dtype=np.float64)
    else:
        biases = _as_f64_array(query_bias, name="query_bias").reshape(-1)
        if biases.size == 1:
            biases = np.full(rows.shape[0], float(biases[0]), dtype=np.float64)
        elif biases.size != rows.shape[0]:
            _fail("SHAPE_MISMATCH", "query_bias must be scalar or length Q")
    rows.setflags(write=False)
    biases = np.ascontiguousarray(biases, dtype=np.float64)
    biases.setflags(write=False)
    return rows, biases


def _normalise_alpha(
    layers: Mapping[int, _FrozenLayer],
    bounds: Mapping[int, _Box],
    query_count: int,
    alpha_by_relu: Optional[Mapping[Any, Any]],
) -> Dict[int, np.ndarray]:
    supplied = dict(alpha_by_relu or {})
    converted: Dict[int, Any] = {}
    for key, value in supplied.items():
        try:
            lid = int(key)
        except Exception as exc:
            raise QueryDualReplayError("INVALID_ALPHA", f"invalid alpha key {key!r}") from exc
        if lid in converted:
            _fail("INVALID_ALPHA", f"duplicate alpha key {lid}")
        converted[lid] = value
    relu_ids = {lid for lid, layer in layers.items() if layer.kind == "RELU"}
    unknown = set(converted) - relu_ids
    if unknown:
        _fail("INVALID_ALPHA", f"alpha supplied for non-ReLU layers {sorted(unknown)}")

    result: Dict[int, np.ndarray] = {}
    for lid in sorted(relu_ids):
        width = bounds[lid].lb.size
        if lid not in converted:
            arr = np.asarray(0.0, dtype=np.float64)
            arr.setflags(write=False)
            result[lid] = arr
            continue
        arr = _as_stored_f64_alpha(converted[lid], name=f"alpha[{lid}]")
        if arr.size == 1:
            norm = np.asarray(float(arr.reshape(-1)[0]), dtype=np.float64)
        elif arr.shape == (width,):
            norm = np.ascontiguousarray(arr, dtype=np.float64)
        elif arr.shape == (query_count, width):
            norm = np.ascontiguousarray(arr, dtype=np.float64)
        elif arr.shape == (1, width):
            # Stored [B=1,n] alpha shared by every query.
            norm = np.ascontiguousarray(arr.reshape(width), dtype=np.float64)
        elif arr.shape == (1, query_count, width):
            # Native candidate tree layout [B=1,M=query_count,n].
            norm = np.ascontiguousarray(arr.reshape(query_count, width))
        else:
            _fail(
                "SHAPE_MISMATCH",
                f"alpha[{lid}] must be scalar, [{width}], [1,{width}], "
                f"[{query_count},{width}], or [1,{query_count},{width}]",
            )
        norm.setflags(write=False)
        result[lid] = norm
    return result


def _check_expected(actual: Mapping[str, str], expected: Mapping[str, Optional[str]]) -> None:
    for name, wanted in expected.items():
        if wanted is None:
            continue
        got = actual[name]
        if not hmac.compare_digest(str(got), str(wanted)):
            _fail("HASH_MISMATCH", f"{name}: expected {wanted}, got {got}")


def _prepare(
    net: Any,
    certified_bounds: Mapping[Any, Any],
    *,
    start_lid: Optional[int],
    query_rows: Optional[Any],
    one_hot: Optional[Any],
    query_bias: Optional[Any],
    alpha_by_relu: Optional[Mapping[Any, Any]],
    deadline: _Deadline,
    expected_net_sha256: Optional[str],
    expected_bounds_sha256: Optional[str],
    expected_query_sha256: Optional[str],
    expected_alpha_sha256: Optional[str],
) -> _Prepared:
    deadline.check(force=True)
    raw_layers, pred_map, output_id, reverse_order, start_mode = _topology(
        net, start_lid
    )
    raw_by_id = {int(layer.id): layer for layer in raw_layers}
    reachable = set(reverse_order)
    frozen: Dict[int, _FrozenLayer] = {}
    for lid in reverse_order:
        deadline.check()
        frozen[lid] = _freeze_layer(raw_by_id[lid], pred_map[lid])
    output_width = frozen[output_id].width

    raw_bounds: Dict[int, Any] = {}
    for key, value in certified_bounds.items():
        try:
            lid = int(key)
        except Exception as exc:
            raise QueryDualReplayError(
                "INVALID_BOUNDS", f"invalid bounds key {key!r}"
            ) from exc
        if lid in raw_bounds:
            _fail("INVALID_BOUNDS", f"duplicate bounds key {lid}")
        raw_bounds[lid] = value
    boxes: Dict[int, _Box] = {}
    for lid in reverse_order:
        if frozen[lid].kind == "INPUT":
            continue
        if lid not in raw_bounds:
            _fail("MISSING_BOUNDS", f"proof consumes bounds for layer {lid}")
        box = _box_from_value(raw_bounds[lid], layer_id=lid)
        if box.lb.size != frozen[lid].width:
            _fail(
                "SHAPE_MISMATCH",
                f"bounds[{lid}] width {box.lb.size} != layer width {frozen[lid].width}",
            )
        boxes[lid] = box

    input_specs = [lid for lid in reverse_order if frozen[lid].kind == "INPUT_SPEC"]
    input_spec_id = input_specs[0]
    rows, biases = _normalise_queries(output_width, query_rows, one_hot, query_bias)
    alpha = _normalise_alpha(frozen, boxes, rows.shape[0], alpha_by_relu)

    net_manifest = [_layer_manifest(frozen[lid]) for lid in reversed(reverse_order)]
    bounds_manifest = [
        {
            "id": lid,
            "semantics": "preactivation" if frozen[lid].kind == "RELU" else "output",
            "lb_sha256": _array_digest(boxes[lid].lb),
            "ub_sha256": _array_digest(boxes[lid].ub),
        }
        for lid in sorted(boxes)
    ]
    alpha_manifest = [
        {"id": lid, "shape": list(value.shape), "sha256": _array_digest(value)}
        for lid, value in sorted(alpha.items())
    ]
    hashes = {
        "net_sha256": _json_digest(net_manifest),
        "bounds_sha256": _json_digest(bounds_manifest),
        "query_sha256": _json_digest(
            {
                "rows": _array_digest(rows),
                "bias": _array_digest(biases),
                "start_layer_id": output_id,
                "start_mode": start_mode,
            }
        ),
        "alpha_sha256": _json_digest(alpha_manifest),
    }
    _check_expected(
        hashes,
        {
            "net_sha256": expected_net_sha256,
            "bounds_sha256": expected_bounds_sha256,
            "query_sha256": expected_query_sha256,
            "alpha_sha256": expected_alpha_sha256,
        },
    )
    deadline.check(force=True)
    return _Prepared(
        layers=frozen,
        reverse_order=reverse_order,
        output_id=output_id,
        output_width=output_width,
        start_mode=start_mode,
        input_spec_id=input_spec_id,
        bounds=boxes,
        queries=rows,
        query_bias=biases,
        alpha=alpha,
        hashes=hashes,
        deadline=deadline,
    )


def _replay_layer_from_root(layer: Any) -> _FrozenLayer:
    """Translate one root-frozen layer without consulting the live network."""

    kind = str(layer.kind)
    source = layer.params
    params: Dict[str, Any] = {}
    if kind == "DENSE":
        params["weight"] = source["weight"]
        params["bias"] = source["bias"]
    elif kind == "CONV2D":
        params.update(
            {
                "weight": source["weight"],
                "bias_channels": source["bias"],
                "input_shape": tuple(source["input_shape"]),
                "output_shape": tuple(source["output_shape"]),
                "stride": tuple(source["stride"]),
                "padding": tuple(source["padding"]),
                "dilation": tuple(source["dilation"]),
                "groups": int(source["groups"]),
            }
        )
    elif kind == "ADD":
        params["bias"] = source["bias"]
    elif kind == "INPUT":
        params["shape"] = tuple(source.get("shape", ()))
        params["dtype"] = str(source.get("dtype", ""))
    elif kind == "INPUT_SPEC":
        params["kind"] = str(source.get("kind", ""))
    elif kind == "ASSERT":
        params["kind"] = str(source.get("kind", ""))
    elif kind == "FLATTEN":
        params["start_dim"] = int(source.get("start_dim", 1))
        params["end_dim"] = int(source.get("end_dim", -1))
        params["input_shape"] = source.get("input_shape")
        params["output_shape"] = source.get("output_shape")
    elif kind != "RELU":
        _fail("UNSUPPORTED_OPERATOR", f"root-frozen layer {layer.id} kind {kind}")
    if int(layer.width) <= 0 and kind != "ASSERT":
        _fail("SHAPE_MISMATCH", f"root-frozen layer {layer.id} has zero width")
    return _FrozenLayer(
        id=int(layer.id),
        kind=kind,
        preds=tuple(int(value) for value in layer.preds),
        width=int(layer.width),
        in_vars=tuple(layer.in_vars),
        out_vars=tuple(layer.out_vars),
        params=MappingProxyType(params),
    )


def _sealed_cone(
    full_layers: Mapping[int, _FrozenLayer],
    layer_manifests: Mapping[int, Mapping[str, Any]],
    *,
    assert_id: int,
    start_lid: Optional[int],
) -> _SealedCone:
    if start_lid is None:
        assertion = full_layers.get(int(assert_id))
        if assertion is None or assertion.kind != "ASSERT" or len(assertion.preds) != 1:
            _fail("INVALID_ROOT_CERTIFICATE", "sealed graph has no terminal ASSERT")
        output_id = int(assertion.preds[0])
        start_mode = "ASSERT_PREDECESSOR"
    else:
        if isinstance(start_lid, bool):
            _fail("INVALID_START_LAYER", "start_lid cannot be boolean")
        try:
            output_id = int(start_lid)
        except Exception as exc:
            raise QueryDualReplayError(
                "INVALID_START_LAYER", f"invalid start_lid {start_lid!r}"
            ) from exc
        if output_id not in full_layers:
            _fail("INVALID_START_LAYER", f"unknown start_lid {output_id}")
        if full_layers[output_id].kind == "ASSERT":
            _fail("INVALID_START_LAYER", "start_lid cannot be ASSERT")
        start_mode = "EXPLICIT_INTERIOR"

    state: Dict[int, int] = {}
    topo: List[int] = []

    def visit(lid: int) -> None:
        mark = state.get(lid, 0)
        if mark == 1:
            _fail("INVALID_CONE", "cycle in sealed replay cone")
        if mark == 2:
            return
        layer = full_layers.get(lid)
        if layer is None:
            _fail("INVALID_CONE", f"sealed cone references unknown layer {lid}")
        if layer.kind == "ASSERT":
            _fail("INVALID_CONE", "ASSERT cannot occur inside replay cone")
        if len(set(layer.preds)) != len(layer.preds):
            _fail("INVALID_CONE", f"layer {lid} repeats a predecessor")
        state[lid] = 1
        for parent in layer.preds:
            visit(parent)
        state[lid] = 2
        topo.append(lid)

    visit(output_id)
    input_specs = [
        lid for lid in topo if full_layers[lid].kind == "INPUT_SPEC"
    ]
    if len(input_specs) != 1:
        _fail("INVALID_CONE", "sealed replay cone needs exactly one INPUT_SPEC")
    reverse_order = tuple(reversed(topo))
    layers = MappingProxyType({lid: full_layers[lid] for lid in reverse_order})
    manifest = [
        layer_manifests[lid] for lid in reversed(reverse_order)
    ]
    manifest_sha = _json_digest(manifest)
    return _SealedCone(
        start_lid=start_lid,
        layers=layers,
        reverse_order=reverse_order,
        output_id=output_id,
        output_width=layers[output_id].width,
        start_mode=start_mode,
        input_spec_id=input_specs[0],
        replay_net_sha256=manifest_sha,
        manifest_sha256=manifest_sha,
    )


def _bounds_manifest(
    layers: Mapping[int, _FrozenLayer],
    boxes: Mapping[int, _Box],
) -> List[Mapping[str, Any]]:
    return [
        {
            "id": lid,
            "semantics": "preactivation" if layers[lid].kind == "RELU" else "output",
            "lb_sha256": _array_digest(boxes[lid].lb),
            "ub_sha256": _array_digest(boxes[lid].ub),
        }
        for lid in sorted(boxes)
    ]


def _prepare_from_sealed(
    cone: _SealedCone,
    frame: QueryDualReplayBoundsFrame,
    *,
    query_rows: Optional[Any],
    one_hot: Optional[Any],
    query_bias: Optional[Any],
    alpha_by_relu: Optional[Mapping[Any, Any]],
    deadline: _Deadline,
    expected_net_sha256: Optional[str],
    expected_bounds_sha256: Optional[str],
    expected_query_sha256: Optional[str],
    expected_alpha_sha256: Optional[str],
) -> _Prepared:
    deadline.check(force=True)
    boxes: Dict[int, _Box] = {}
    for lid in cone.reverse_order:
        layer = cone.layers[lid]
        if layer.kind == "INPUT":
            continue
        box = frame._bounds.get(lid)
        if box is None:
            _fail("MISSING_BOUNDS", f"sealed frame has no bounds for layer {lid}")
        if box.lb.size != layer.width:
            _fail(
                "SHAPE_MISMATCH",
                f"sealed bounds[{lid}] width {box.lb.size} != layer width {layer.width}",
            )
        boxes[lid] = box
    rows, biases = _normalise_queries(
        cone.output_width, query_rows, one_hot, query_bias
    )
    alpha = _normalise_alpha(
        cone.layers, boxes, rows.shape[0], alpha_by_relu
    )
    alpha_manifest = [
        {"id": lid, "shape": list(value.shape), "sha256": _array_digest(value)}
        for lid, value in sorted(alpha.items())
    ]
    hashes = {
        "net_sha256": cone.replay_net_sha256,
        "bounds_sha256": _json_digest(_bounds_manifest(cone.layers, boxes)),
        "query_sha256": _json_digest(
            {
                "rows": _array_digest(rows),
                "bias": _array_digest(biases),
                "start_layer_id": cone.output_id,
                "start_mode": cone.start_mode,
            }
        ),
        "alpha_sha256": _json_digest(alpha_manifest),
    }
    _check_expected(
        hashes,
        {
            "net_sha256": expected_net_sha256,
            "bounds_sha256": expected_bounds_sha256,
            "query_sha256": expected_query_sha256,
            "alpha_sha256": expected_alpha_sha256,
        },
    )
    deadline.check(force=True)
    return _Prepared(
        layers=cone.layers,
        reverse_order=cone.reverse_order,
        output_id=cone.output_id,
        output_width=cone.output_width,
        start_mode=cone.start_mode,
        input_spec_id=cone.input_spec_id,
        bounds=MappingProxyType(boxes),
        queries=rows,
        query_bias=biases,
        alpha=MappingProxyType(alpha),
        hashes=MappingProxyType(hashes),
        deadline=deadline,
    )


def _gamma(operations: int) -> float:
    operations = max(1, int(operations))
    product = np.longdouble(operations) * np.longdouble(_U)
    if product >= 0.5:
        _fail("NUMERIC_GUARD", "Higham gamma operation count is too large")
    return float(_longdouble_to_f64_up(product / (np.longdouble(1.0) - product)))


def _underflow_allowance(operations: int) -> float:
    value = np.longdouble(max(1, int(operations))) * np.longdouble(_ETA)
    return float(_longdouble_to_f64_up(value))


def _longdouble_to_f64_up(value: Any) -> Any:
    """Convert a finite long-double expression to a binary64 upper bound.

    Every caller supplies an expression over nonnegative binary64 operands.
    ``np.longdouble`` has strictly more precision than binary64 on the
    supported CPU proof platform; one final binary64 successor therefore
    dominates both the long-double rounding and the narrowing conversion.
    """

    if not _has_wide_longdouble():
        _fail(
            "NUMERIC_PLATFORM",
            "proof replay requires longdouble precision wider than binary64",
        )
    ld = np.asarray(value, dtype=np.longdouble)
    if not np.all(np.isfinite(ld)):
        _fail("NONFINITE", "long-double outward expression is non-finite")
    out = np.asarray(ld, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        _fail("NONFINITE", "outward binary64 conversion overflowed")
    out = np.nextafter(out, math.inf)
    if out.ndim == 0:
        return float(out)
    return np.ascontiguousarray(out)


def _has_wide_longdouble() -> bool:
    return bool(
        np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
        and np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    )


def _check_numeric_platform() -> Mapping[str, Any]:
    """Reject FTZ/DAZ, non-nearest rounding, or a narrow long-double host."""

    if not _has_wide_longdouble():
        _fail(
            "NUMERIC_PLATFORM",
            "proof replay requires longdouble precision wider than binary64",
        )
    eta = np.nextafter(np.float64(0.0), np.float64(math.inf))
    tiny = np.float64(np.finfo(np.float64).tiny)
    half_tiny = np.float64(tiny * np.float64(0.5))
    eta_product = np.float64(eta * np.float64(1.0))
    eta_dot = float(
        (
            np.asarray([[eta]], dtype=np.float64)
            @ np.asarray([[1.0]], dtype=np.float64)
        )[0, 0]
    )
    if (
        eta <= 0.0
        or half_tiny <= 0.0
        or eta_product != eta
        or eta_dot != float(eta)
    ):
        _fail(
            "NUMERIC_PLATFORM",
            "gradual-underflow probe failed (FTZ/DAZ is unsafe for eta guards)",
        )
    half_ulp = np.float64(2.0**-53)
    above_half_ulp = np.nextafter(half_ulp, np.float64(math.inf))
    if (
        np.float64(1.0) + half_ulp != np.float64(1.0)
        or np.float64(1.0) + above_half_ulp == np.float64(1.0)
    ):
        _fail("NUMERIC_PLATFORM", "round-to-nearest-even probe failed")
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "byteorder": __import__("sys").byteorder,
        "binary64_nmant": int(np.finfo(np.float64).nmant),
        "longdouble_nmant": int(np.finfo(np.longdouble).nmant),
        "longdouble_eps": str(np.finfo(np.longdouble).eps),
        "gradual_underflow": True,
        "round_to_nearest_even": True,
        "blas_subnormal_dot": True,
    }


def _upper_gamma_enclosure(abs_nominal: Any, gamma: float, under: float) -> Any:
    numerator = np.asarray(abs_nominal, dtype=np.longdouble) + np.longdouble(under)
    denominator = np.longdouble(1.0) - np.longdouble(gamma)
    return _longdouble_to_f64_up(numerator / denominator)


def _upper_error_from_mass(mass_upper: Any, gamma: float, under: float) -> Any:
    value = (
        np.longdouble(gamma) * np.asarray(mass_upper, dtype=np.longdouble)
        + np.longdouble(under)
    )
    return _longdouble_to_f64_up(value)


def _upper_nonnegative_sum(*values: Any) -> Any:
    if not values:
        return 0.0
    total = np.asarray(values[0], dtype=np.longdouble)
    for value in values[1:]:
        total = total + np.asarray(value, dtype=np.longdouble)
    return _longdouble_to_f64_up(total)


def _require_finite(value: Any, *, where: str) -> None:
    if not np.all(np.isfinite(value)):
        _fail("NONFINITE", f"non-finite arithmetic at {where}")


def _matrix_product_with_error(
    a: np.ndarray,
    weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Direct row-dot matrix product with a componentwise Higham enclosure."""

    if a.ndim != 2 or weight.ndim != 2 or a.shape[1] != weight.shape[0]:
        _fail("SHAPE_MISMATCH", "invalid binary64 matrix-product operands")
    terms = int(a.shape[1])
    ops = 2 * terms + 2
    g = _gamma(ops)
    under = _underflow_allowance(ops)
    nominal = np.asarray(a @ weight, dtype=np.float64)
    abs_nominal = np.asarray(np.abs(a) @ np.abs(weight), dtype=np.float64)
    _require_finite(nominal, where="matrix product")
    _require_finite(abs_nominal, where="absolute matrix product")
    sum_upper = _upper_gamma_enclosure(abs_nominal, g, under)
    radius = _upper_error_from_mass(sum_upper, g, under)
    exact_zero = ~(
        np.any(a != 0.0, axis=1).reshape(-1, 1)
        & np.any(weight != 0.0, axis=0).reshape(1, -1)
    )
    radius[exact_zero] = 0.0
    _require_finite(radius, where="matrix-product error")
    return np.ascontiguousarray(nominal), np.ascontiguousarray(radius)


def _row_dots_with_error(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """One dot per row; ``b`` may be one shared row or match ``a``."""

    if a.ndim != 2:
        _fail("SHAPE_MISMATCH", "row-dot left operand must be rank two")
    if b.ndim == 1:
        if b.size != a.shape[1]:
            _fail("SHAPE_MISMATCH", "shared row-dot operand width")
        shared = True
    elif b.ndim == 2 and b.shape == a.shape:
        shared = False
    else:
        _fail("SHAPE_MISMATCH", "invalid row-dot operands")
    if not np.any(a) or not np.any(b):
        zeros = np.zeros(a.shape[0], dtype=np.float64)
        return zeros, zeros.copy()
    n = int(a.shape[1])
    ops = 2 * n + 2
    g = _gamma(ops)
    under = _underflow_allowance(ops)
    if shared:
        nominal = np.asarray(a @ b, dtype=np.float64)
        abs_nominal = np.asarray(np.abs(a) @ np.abs(b), dtype=np.float64)
        exact_zero = (
            ~np.any(a != 0.0, axis=1)
            if np.any(b != 0.0)
            else np.ones(a.shape[0], dtype=bool)
        )
    else:
        nominal = np.asarray(np.sum(a * b, axis=1), dtype=np.float64)
        abs_nominal = np.asarray(
            np.sum(np.abs(a) * np.abs(b), axis=1), dtype=np.float64
        )
        exact_zero = ~np.any((a != 0.0) & (b != 0.0), axis=1)
    _require_finite(nominal, where="row dots")
    _require_finite(abs_nominal, where="absolute row dots")
    sum_upper = _upper_gamma_enclosure(abs_nominal, g, under)
    radius = _upper_error_from_mass(sum_upper, g, under)
    radius[exact_zero] = 0.0
    _require_finite(radius, where="row-dot error")
    return np.ascontiguousarray(nominal), np.ascontiguousarray(radius)


def _elementwise_product_with_error(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        _fail("SHAPE_MISMATCH", "invalid elementwise product operands")
    ops = 3
    g = _gamma(ops)
    under = _underflow_allowance(ops)
    nominal = np.asarray(a * b, dtype=np.float64)
    abs_nominal = np.asarray(np.abs(a) * np.abs(b), dtype=np.float64)
    _require_finite(nominal, where="elementwise product")
    _require_finite(abs_nominal, where="elementwise absolute product")
    product_upper = _upper_gamma_enclosure(abs_nominal, g, under)
    radius = _upper_error_from_mass(product_upper, g, under)
    _require_finite(radius, where="elementwise product error")
    radius[(a == 0.0) | (b == 0.0)] = 0.0
    return np.ascontiguousarray(nominal), np.ascontiguousarray(radius)


def _binary_add_with_error(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        _fail("SHAPE_MISMATCH", "DAG adjoint widths differ")
    nominal = np.asarray(a + b, dtype=np.float64)
    _require_finite(nominal, where="DAG adjoint merge")
    g = _gamma(2)
    under = _underflow_allowance(2)
    magnitude = _upper_nonnegative_sum(np.abs(a), np.abs(b))
    radius = _upper_error_from_mass(magnitude, g, under)
    radius[(a == 0.0) & (b == 0.0)] = 0.0
    _require_finite(radius, where="DAG merge error")
    return np.ascontiguousarray(nominal), np.ascontiguousarray(radius)


def _down_add(a: Any, b: Any, *, where: str) -> Any:
    value = np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    _require_finite(value, where=where)
    value = np.nextafter(value, -math.inf)
    _require_finite(value, where=where)
    if value.ndim == 0:
        return float(value)
    return np.ascontiguousarray(value)


def _row_dots_lower(
    a: np.ndarray,
    b: np.ndarray,
    stats: _ReplayStats,
) -> np.ndarray:
    nominal, radius = _row_dots_with_error(a, b)
    if np.any(radius):
        stats.record_guard(radius, coefficient=False)
    return _down_add(nominal, -radius, where="downward scalar dot")


def _output_box(prepared: _Prepared, lid: int) -> _Box:
    layer = prepared.layers[lid]
    box = prepared.bounds[lid]
    if layer.kind != "RELU":
        return box
    lb = np.maximum(box.lb, 0.0)
    ub = np.maximum(box.ub, 0.0)
    lb.setflags(write=False)
    ub.setflags(write=False)
    return _Box(lb=lb, ub=ub)


def _absorb_radius(
    scalar: np.ndarray,
    radius: np.ndarray,
    box: _Box,
    stats: _ReplayStats,
) -> np.ndarray:
    if (
        radius.ndim != 2
        or radius.shape[0] != scalar.size
        or radius.shape[1] != box.lb.size
        or np.any(radius < 0.0)
    ):
        _fail("NUMERIC_GUARD", "coefficient radius/box mismatch")
    if not np.any(radius):
        return scalar
    max_abs = np.maximum(np.abs(box.lb), np.abs(box.ub))
    _, raw_error = _row_dots_with_error(radius, max_abs)
    nominal = np.asarray(radius @ max_abs, dtype=np.float64)
    penalty = _upper_nonnegative_sum(nominal, raw_error)
    zero_rows = ~np.any(
        (radius != 0.0) & (max_abs.reshape(1, -1) != 0.0), axis=1
    )
    penalty[zero_rows] = 0.0
    _require_finite(penalty, where="coefficient-error box absorption")
    stats.record_guard(penalty, coefficient=True)
    if not np.any(penalty):
        return scalar
    return _down_add(scalar, -penalty, where="coefficient-error absorption")


def _alpha_block(
    prepared: _Prepared,
    lid: int,
    query_start: int,
    query_end: int,
    width: int,
) -> np.ndarray:
    raw = prepared.alpha[lid]
    if raw.ndim == 0:
        return np.full((query_end - query_start, width), float(raw), dtype=np.float64)
    if raw.ndim == 1:
        return np.broadcast_to(raw.reshape(1, -1), (query_end - query_start, width))
    return np.asarray(raw[query_start:query_end])


def _fraction_to_float_up(value: Fraction) -> float:
    nearest = float(value)
    if not math.isfinite(nearest):
        _fail("NONFINITE", "Fraction-to-float conversion overflow")
    if Fraction.from_float(nearest) < value:
        nearest = float(np.nextafter(np.float64(nearest), np.float64(math.inf)))
    if not math.isfinite(nearest):
        _fail("NONFINITE", "outward Fraction conversion overflow")
    return nearest


def _relu_lines(
    prepared: _Prepared,
    lid: int,
    stats: _ReplayStats,
    required_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cached = prepared.relu_lines.get(lid)
    box = prepared.bounds[lid]
    l, u = box.lb, box.ub
    ambiguous = (l < 0.0) & (u > 0.0)
    if cached is None:
        slope = np.zeros_like(l)
        beta = np.zeros_like(l)
        audited = np.zeros_like(ambiguous)
        prepared.relu_lines[lid] = (slope, beta, audited)
    else:
        slope, beta, audited = cached
    required = ambiguous.copy()
    if required_mask is not None:
        mask = np.asarray(required_mask, dtype=bool).reshape(-1)
        if mask.shape != ambiguous.shape:
            _fail("SHAPE_MISMATCH", f"RELU audit mask at layer {lid}")
        required &= mask
    indices = np.flatnonzero(required & ~audited)
    for offset, index in enumerate(indices):
        if (offset & 1023) == 0:
            prepared.deadline.check(force=True)
        lf = Fraction.from_float(float(l[index]))
        uf = Fraction.from_float(float(u[index]))
        exact_slope = uf / (uf - lf)
        sf = float(exact_slope)
        if not math.isfinite(sf) or sf < 0.0 or sf > 1.0:
            _fail("RELU_AUDIT", f"invalid secant slope at layer {lid}, index {index}")
        slope_fraction = Fraction.from_float(sf)
        required = max(
            Fraction(0),
            -slope_fraction * lf,
            (Fraction(1) - slope_fraction) * uf,
        )
        bf = _fraction_to_float_up(required)
        beta_fraction = Fraction.from_float(bf)
        # Exact endpoint audit of the stored binary64 line.
        if (
            slope_fraction * lf + beta_fraction < 0
            or beta_fraction < 0
            or slope_fraction * uf + beta_fraction < uf
        ):
            _fail("RELU_AUDIT", f"outward secant audit failed at {lid}:{index}")
        slope[index] = sf
        beta[index] = bf
        audited[index] = True
        stats.fraction_endpoint_audits += 1
    return slope, beta, ambiguous


def _conv_output_padding(layer: _FrozenLayer) -> Tuple[int, int]:
    p = layer.params
    _, input_h, input_w = p["input_shape"]
    _, output_h, output_w = p["output_shape"]
    kernel_h, kernel_w = p["weight"].shape[-2:]
    stride_h, stride_w = p["stride"]
    padding_h, padding_w = p["padding"]
    dilation_h, dilation_w = p["dilation"]
    base_h = (
        (output_h - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (kernel_h - 1)
        + 1
    )
    base_w = (
        (output_w - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (kernel_w - 1)
        + 1
    )
    result = (input_h - base_h, input_w - base_w)
    if (
        result[0] < 0
        or result[1] < 0
        or result[0] >= stride_h
        or result[1] >= stride_w
    ):
        _fail("INVALID_CONV", f"layer {layer.id} cannot recover declared input shape")
    return result


def _conv_reverse_with_error(
    coefficient: np.ndarray,
    layer: _FrozenLayer,
    deadline: _Deadline,
    stats: _ReplayStats,
) -> Tuple[np.ndarray, np.ndarray]:
    """Auditable batched Conv2D adjoint.

    The opaque framework convolution backend is intentionally not used.
    Each kernel offset is reduced as a direct channel GEMM (one dot of
    ``out_channels/groups`` terms), then accumulated explicitly into its
    unique input-spatial targets.  The GEMM dot and every accumulation have
    separate componentwise error enclosures.
    """

    p = layer.params
    out_c, out_h, out_w = p["output_shape"]
    in_c, in_h, in_w = p["input_shape"]
    if (
        coefficient.ndim != 2
        or coefficient.shape[1] != out_c * out_h * out_w
    ):
        _fail("SHAPE_MISMATCH", f"CONV2D layer {layer.id} coefficient width")
    # Also proves that the declared geometry admits the standard transpose
    # recovery remainder; _freeze_layer already checked the forward formula.
    _conv_output_padding(layer)
    batch = coefficient.shape[0]
    coeff = coefficient.reshape(batch, out_c, out_h, out_w)
    nonzero_count = int(np.count_nonzero(coeff))
    dense_count = int(coeff.size)
    # A one-hot/interior query is extremely sparse.  Expanding all zero rows
    # through GEMM is both slower and harder on memory, so use an explicitly
    # audited scatter reduction while fewer than 1/8 entries are nonzero.
    if nonzero_count * 8 <= dense_count:
        stats.conv_sparse_blocks += 1
        return _conv_reverse_sparse_with_error(
            coeff, layer, deadline, nonzero_count
        )
    stats.conv_dense_blocks += 1
    nominal = np.zeros((batch, in_c, in_h * in_w), dtype=np.float64)
    radius = np.zeros_like(nominal)
    weight = p["weight"]
    stride_h, stride_w = p["stride"]
    padding_h, padding_w = p["padding"]
    dilation_h, dilation_w = p["dilation"]
    groups = int(p["groups"])
    out_per_group = out_c // groups
    in_per_group = in_c // groups
    for group in range(groups):
        deadline.check(force=True)
        co_start = group * out_per_group
        co_end = co_start + out_per_group
        ci_start = group * in_per_group
        ci_end = ci_start + in_per_group
        coeff_group = coeff[:, co_start:co_end, :, :]
        nominal_group = nominal[:, ci_start:ci_end, :]
        radius_group = radius[:, ci_start:ci_end, :]
        for kh in range(int(weight.shape[2])):
            deadline.check(force=True)
            input_h_indices = (
                np.arange(out_h, dtype=np.int64) * stride_h
                - padding_h
                + kh * dilation_h
            )
            valid_h = (input_h_indices >= 0) & (input_h_indices < in_h)
            if not np.any(valid_h):
                continue
            output_h_indices = np.flatnonzero(valid_h)
            input_h_indices = input_h_indices[valid_h]
            for kw in range(int(weight.shape[3])):
                deadline.check(force=True)
                input_w_indices = (
                    np.arange(out_w, dtype=np.int64) * stride_w
                    - padding_w
                    + kw * dilation_w
                )
                valid_w = (input_w_indices >= 0) & (input_w_indices < in_w)
                if not np.any(valid_w):
                    continue
                output_w_indices = np.flatnonzero(valid_w)
                input_w_indices = input_w_indices[valid_w]
                selected = np.take(coeff_group, output_h_indices, axis=2)
                selected = np.take(selected, output_w_indices, axis=3)
                left = np.ascontiguousarray(
                    selected.transpose(0, 2, 3, 1).reshape(-1, out_per_group)
                )
                weight_slice = np.ascontiguousarray(
                    weight[co_start:co_end, :, kh, kw]
                )
                term, term_radius = _matrix_product_with_error(left, weight_slice)
                nh, nw = len(input_h_indices), len(input_w_indices)
                term = term.reshape(batch, nh, nw, in_per_group).transpose(
                    0, 3, 1, 2
                )
                term_radius = term_radius.reshape(
                    batch, nh, nw, in_per_group
                ).transpose(0, 3, 1, 2)
                targets = (
                    input_h_indices[:, None] * in_w + input_w_indices[None, :]
                ).reshape(-1)
                term = np.ascontiguousarray(term.reshape(batch, in_per_group, -1))
                term_radius = np.ascontiguousarray(
                    term_radius.reshape(batch, in_per_group, -1)
                )
                old = nominal_group[:, :, targets]
                old_radius = radius_group[:, :, targets]
                merged, addition_radius = _binary_add_with_error(old, term)
                combined_radius = _upper_nonnegative_sum(
                    old_radius, term_radius, addition_radius
                )
                nominal_group[:, :, targets] = merged
                radius_group[:, :, targets] = combined_radius
                deadline.check(force=True)

    deadline.check(force=True)
    _require_finite(nominal, where="direct Conv2D reverse")
    _require_finite(radius, where="conv2d reverse error")
    return (
        np.ascontiguousarray(nominal.reshape(batch, -1)),
        np.ascontiguousarray(radius.reshape(batch, -1)),
    )


def _conv_reverse_sparse_with_error(
    coefficient_4d: np.ndarray,
    layer: _FrozenLayer,
    deadline: _Deadline,
    nonzero_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Direct nonzero scatter form of the Conv2D adjoint."""

    p = layer.params
    batch, out_c, out_h, out_w = coefficient_4d.shape
    in_c, in_h, in_w = p["input_shape"]
    weight = p["weight"]
    stride_h, stride_w = p["stride"]
    padding_h, padding_w = p["padding"]
    dilation_h, dilation_w = p["dilation"]
    groups = int(p["groups"])
    out_per_group = out_c // groups
    in_per_group = in_c // groups
    nominal = np.zeros((batch, in_c, in_h * in_w), dtype=np.float64)
    radius = np.zeros_like(nominal)
    if nonzero_count == 0:
        return nominal.reshape(batch, -1), radius.reshape(batch, -1)

    nonzero = np.argwhere(coefficient_4d != 0.0)
    for offset, (batch_index, co, oh, ow) in enumerate(nonzero):
        if (offset & 63) == 0:
            deadline.check(force=True)
        c = float(coefficient_4d[batch_index, co, oh, ow])
        group = int(co) // out_per_group
        ci_start = group * in_per_group
        ci_end = ci_start + in_per_group
        ih0 = int(oh) * stride_h - padding_h
        iw0 = int(ow) * stride_w - padding_w
        left = np.full(in_per_group, c, dtype=np.float64)
        for kh in range(int(weight.shape[2])):
            ih = ih0 + kh * dilation_h
            if ih < 0 or ih >= in_h:
                continue
            for kw in range(int(weight.shape[3])):
                iw = iw0 + kw * dilation_w
                if iw < 0 or iw >= in_w:
                    continue
                target = ih * in_w + iw
                term, term_radius = _elementwise_product_with_error(
                    left, weight[int(co), :, kh, kw]
                )
                old = nominal[int(batch_index), ci_start:ci_end, target]
                old_radius = radius[int(batch_index), ci_start:ci_end, target]
                merged, addition_radius = _binary_add_with_error(old, term)
                combined_radius = _upper_nonnegative_sum(
                    old_radius, term_radius, addition_radius
                )
                nominal[int(batch_index), ci_start:ci_end, target] = merged
                radius[int(batch_index), ci_start:ci_end, target] = combined_radius
    deadline.check(force=True)
    _require_finite(nominal, where="sparse direct Conv2D reverse")
    _require_finite(radius, where="sparse direct Conv2D reverse error")
    return (
        np.ascontiguousarray(nominal.reshape(batch, -1)),
        np.ascontiguousarray(radius.reshape(batch, -1)),
    )


def _push_adjoint(
    prepared: _Prepared,
    pending: MutableMapping[int, np.ndarray],
    lid: int,
    coefficient: np.ndarray,
    scalar: np.ndarray,
    stats: _ReplayStats,
) -> np.ndarray:
    coefficient = np.ascontiguousarray(coefficient, dtype=np.float64)
    _require_finite(coefficient, where=f"adjoint for layer {lid}")
    if (
        coefficient.ndim != 2
        or coefficient.shape[0] != scalar.size
        or coefficient.shape[1] != prepared.layers[lid].width
    ):
        _fail("SHAPE_MISMATCH", f"adjoint width for layer {lid}")
    current = pending.get(lid)
    if current is None:
        pending[lid] = coefficient.copy()
        return scalar
    merged, radius = _binary_add_with_error(current, coefficient)
    scalar = _absorb_radius(scalar, radius, _output_box(prepared, lid), stats)
    pending[lid] = merged
    stats.dag_merges += 1
    return scalar


def _input_support_lower(
    coefficient: np.ndarray,
    box: _Box,
    stats: _ReplayStats,
) -> np.ndarray:
    if coefficient.ndim != 2 or coefficient.shape[1] != box.lb.size:
        _fail("SHAPE_MISMATCH", "input support width mismatch")
    endpoint = np.where(
        coefficient >= 0.0, box.lb.reshape(1, -1), box.ub.reshape(1, -1)
    )
    return _row_dots_lower(coefficient, endpoint, stats)


def _replay_block_core(
    prepared: _Prepared,
    query_start: int,
    query_end: int,
    stats: _ReplayStats,
    *,
    stop_lid: Optional[int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    prepared.deadline.check(force=True)
    batch = query_end - query_start
    stats.begin_block(query_start, query_end)
    pending: Dict[int, np.ndarray] = {
        prepared.output_id: np.ascontiguousarray(
            prepared.queries[query_start:query_end].copy(), dtype=np.float64
        )
    }
    scalar = np.ascontiguousarray(
        prepared.query_bias[query_start:query_end].copy(), dtype=np.float64
    )
    reached_input = False
    stopped_coefficient: Optional[np.ndarray] = None

    for lid in prepared.reverse_order:
        prepared.deadline.check()
        coefficient = pending.pop(lid, None)
        if coefficient is None:
            continue
        if stop_lid is not None and lid == stop_lid:
            if stopped_coefficient is not None:
                _fail("INVALID_GRAPH", "interior stop layer was reached twice")
            stopped_coefficient = np.ascontiguousarray(
                coefficient.copy(), dtype=np.float64
            )
            continue
        layer = prepared.layers[lid]
        kind = layer.kind
        preds = layer.preds

        if kind == "INPUT_SPEC":
            if lid != prepared.input_spec_id:
                _fail("INVALID_GRAPH", "unexpected secondary INPUT_SPEC")
            support = _input_support_lower(coefficient, prepared.bounds[lid], stats)
            scalar = _down_add(scalar, support, where="input support addition")
            reached_input = True
            continue
        if kind == "INPUT":
            _fail("INVALID_GRAPH", "proof path bypassed INPUT_SPEC")
        if kind == "DENSE":
            weight = layer.params["weight"]
            if coefficient.shape != (batch, weight.shape[0]):
                _fail("SHAPE_MISMATCH", f"DENSE layer {lid} adjoint width")
            bias_contribution = _row_dots_lower(
                coefficient, layer.params["bias"], stats
            )
            scalar = _down_add(
                scalar, bias_contribution, where=f"DENSE {lid} bias"
            )
            new_coefficient, radius = _matrix_product_with_error(
                coefficient, weight
            )
            pred = preds[0]
            if new_coefficient.shape[1] != prepared.layers[pred].width:
                _fail("SHAPE_MISMATCH", f"DENSE layer {lid} input width")
            scalar = _absorb_radius(
                scalar, radius, _output_box(prepared, pred), stats
            )
            scalar = _push_adjoint(
                prepared, pending, pred, new_coefficient, scalar, stats
            )
            stats.affine_terms += batch * int(weight.size)
            continue
        if kind == "CONV2D":
            p = layer.params
            out_c, out_h, out_w = p["output_shape"]
            repeated_bias = np.repeat(p["bias_channels"], out_h * out_w)
            bias_contribution = _row_dots_lower(
                coefficient, repeated_bias, stats
            )
            scalar = _down_add(
                scalar, bias_contribution, where=f"CONV2D {lid} bias"
            )
            new_coefficient, radius = _conv_reverse_with_error(
                coefficient, layer, prepared.deadline, stats
            )
            pred = preds[0]
            if new_coefficient.shape[1] != prepared.layers[pred].width:
                _fail("SHAPE_MISMATCH", f"CONV2D layer {lid} input width")
            scalar = _absorb_radius(
                scalar, radius, _output_box(prepared, pred), stats
            )
            scalar = _push_adjoint(
                prepared, pending, pred, new_coefficient, scalar, stats
            )
            stats.affine_terms += batch * int(p["weight"].size) * out_h * out_w
            continue
        if kind == "FLATTEN":
            pred = preds[0]
            if coefficient.shape[1] != prepared.layers[pred].width:
                _fail("SHAPE_MISMATCH", f"FLATTEN layer {lid} is not size preserving")
            scalar = _push_adjoint(
                prepared, pending, pred, coefficient.copy(), scalar, stats
            )
            continue
        if kind == "ADD":
            bias_contribution = _row_dots_lower(
                coefficient, layer.params["bias"], stats
            )
            scalar = _down_add(scalar, bias_contribution, where=f"ADD {lid} bias")
            for pred in preds:
                if prepared.layers[pred].width != coefficient.shape[1]:
                    _fail("SHAPE_MISMATCH", f"ADD layer {lid} predecessor width")
                scalar = _push_adjoint(
                    prepared, pending, pred, coefficient.copy(), scalar, stats
                )
            continue
        if kind == "RELU":
            box = prepared.bounds[lid]
            if coefficient.shape[1] != box.lb.size:
                _fail("SHAPE_MISMATCH", f"RELU layer {lid} adjoint width")
            ambiguous = (box.lb < 0.0) & (box.ub > 0.0)
            off = box.ub <= 0.0
            on = (box.lb >= 0.0) & ~off
            alpha = _alpha_block(
                prepared, lid, query_start, query_end, coefficient.shape[1]
            )
            factor = np.zeros_like(coefficient)
            factor[:, on] = 1.0
            positive_ambiguous = ambiguous.reshape(1, -1) & (coefficient >= 0.0)
            negative_ambiguous = ambiguous.reshape(1, -1) & (coefficient < 0.0)
            slope, beta, _ = _relu_lines(
                prepared,
                lid,
                stats,
                required_mask=np.any(negative_ambiguous, axis=0),
            )
            factor[positive_ambiguous] = alpha[positive_ambiguous]
            slope_block = np.broadcast_to(slope.reshape(1, -1), coefficient.shape)
            factor[negative_ambiguous] = slope_block[negative_ambiguous]
            new_coefficient, radius = _elementwise_product_with_error(
                coefficient, factor
            )
            intercept = np.zeros_like(coefficient)
            beta_block = np.broadcast_to(beta.reshape(1, -1), coefficient.shape)
            intercept[negative_ambiguous] = beta_block[negative_ambiguous]
            intercept_contribution = _row_dots_lower(
                coefficient, intercept, stats
            )
            scalar = _down_add(
                scalar, intercept_contribution, where=f"RELU {lid} intercept"
            )
            pred = preds[0]
            scalar = _absorb_radius(scalar, radius, box, stats)
            scalar = _push_adjoint(
                prepared, pending, pred, new_coefficient, scalar, stats
            )
            stats.relu_ambiguous_terms += batch * int(np.count_nonzero(ambiguous))
            continue
        _fail("UNSUPPORTED_OPERATOR", f"unhandled layer {lid} kind {kind}")

    if pending:
        _fail("INVALID_GRAPH", f"unprocessed adjoints remain at {sorted(pending)}")
    if stop_lid is None:
        if not reached_input:
            _fail("INVALID_GRAPH", "query does not reach INPUT_SPEC")
    else:
        if stopped_coefficient is None:
            _fail("INVALID_GRAPH", "query does not reach the interior stop layer")
        if reached_input:
            _fail(
                "INVALID_GRAPH",
                "interior stop layer does not dominate every query path",
            )
    scalar = np.nextafter(scalar, -math.inf)
    _require_finite(scalar, where="final lower bound")
    if stopped_coefficient is not None:
        _require_finite(
            stopped_coefficient,
            where="interior affine lower coefficient",
        )
    return np.ascontiguousarray(scalar), stopped_coefficient


def _replay_block(
    prepared: _Prepared,
    query_start: int,
    query_end: int,
    stats: _ReplayStats,
) -> np.ndarray:
    scalar, stopped = _replay_block_core(
        prepared,
        query_start,
        query_end,
        stats,
        stop_lid=None,
    )
    if stopped is not None:
        _fail("INTERNAL_ERROR", "full replay unexpectedly returned a stop plane")
    return scalar


def _replay_affine_block(
    prepared: _Prepared,
    query_start: int,
    query_end: int,
    stats: _ReplayStats,
    *,
    stop_lid: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scalar, coefficient = _replay_block_core(
        prepared,
        query_start,
        query_end,
        stats,
        stop_lid=int(stop_lid),
    )
    if coefficient is None:
        _fail("INTERNAL_ERROR", "interior replay returned no affine plane")
    return scalar, coefficient


def _worker_prepared(prepared: _Prepared) -> _Prepared:
    """Clone only mutable replay state for one objective worker.

    Frozen graph, bounds, queries, and alpha arrays are immutable inputs and
    are deliberately shared.  The deadline counter and lazily audited ReLU
    line cache are worker-local so that scheduling cannot change proof state.
    """

    return _Prepared(
        layers=prepared.layers,
        reverse_order=prepared.reverse_order,
        output_id=prepared.output_id,
        output_width=prepared.output_width,
        start_mode=prepared.start_mode,
        input_spec_id=prepared.input_spec_id,
        bounds=prepared.bounds,
        queries=prepared.queries,
        query_bias=prepared.query_bias,
        alpha=prepared.alpha,
        hashes=prepared.hashes,
        deadline=_Deadline(end=prepared.deadline.end),
        relu_lines=prepared.relu_lines,
    )


def _merge_replay_stats(target: _ReplayStats, source: _ReplayStats) -> None:
    """Deterministically merge disjoint-query worker statistics."""

    target.coefficient_guards += source.coefficient_guards
    target.scalar_guards += source.scalar_guards
    target.fraction_endpoint_audits += source.fraction_endpoint_audits
    target.relu_ambiguous_terms += source.relu_ambiguous_terms
    target.affine_terms += source.affine_terms
    target.dag_merges += source.dag_merges
    target.conv_sparse_blocks += source.conv_sparse_blocks
    target.conv_dense_blocks += source.conv_dense_blocks
    target.guard_total = float(
        np.nextafter(
            np.longdouble(target.guard_total) + np.longdouble(source.guard_total),
            np.longdouble(math.inf),
        )
    )
    target.guard_max = max(target.guard_max, source.guard_max)
    if target.guard_by_query is None or source.guard_by_query is None:
        _fail("INTERNAL_ERROR", "parallel replay statistics were not configured")
    target.guard_by_query = _upper_nonnegative_sum(
        target.guard_by_query, source.guard_by_query
    )
    if not math.isfinite(target.guard_total):
        _fail("NUMERIC_GUARD", "parallel guard statistics overflow")


def _replay_objective_workers(
    prepared: _Prepared,
    *,
    requested_workers: int,
    chunk_size: int,
    max_workspace_bytes: int,
) -> Tuple[np.ndarray, _ReplayStats, int, int]:
    """Replay independent objective rows with bounded fixed CPU parallelism.

    The total nominal workspace budget is divided across active workers.  A
    worker owns a contiguous row shard and an independent mutable replay
    state; only immutable graph/material arrays are shared.  Results and
    statistics are merged by shard index, never completion order.
    """

    query_count = int(prepared.queries.shape[0])
    if query_count <= 0:
        _fail("EMPTY_QUERY", "parallel replay requires at least one query")
    maximum_width = max(layer.width for layer in prepared.layers.values())
    bytes_per_query = max(1, maximum_width * 8 * 12)
    workspace_workers = max(1, max_workspace_bytes // bytes_per_query)
    effective_workers = min(requested_workers, query_count, workspace_workers)
    per_worker_workspace = max_workspace_bytes // effective_workers
    memory_limited = max(1, per_worker_workspace // bytes_per_query)
    effective_chunk_size = min(chunk_size, memory_limited, query_count)

    stats = _ReplayStats()
    stats.configure_queries(query_count)
    values = np.empty(query_count, dtype=np.float64)
    if effective_workers == 1:
        for chunk_start in range(0, query_count, effective_chunk_size):
            prepared.deadline.check(force=True)
            chunk_end = min(query_count, chunk_start + effective_chunk_size)
            values[chunk_start:chunk_end] = _replay_block(
                prepared, chunk_start, chunk_end, stats
            )
        return values, stats, effective_chunk_size, effective_workers

    # Fraction endpoint construction is Python-heavy and the same certified
    # secant is consumed by every objective shard.  Audit each reachable ReLU
    # once on the coordinator, freeze the arrays, and share only those
    # immutable lines.  This avoids four duplicate audits without sharing a
    # lazily mutable cache.
    for lid in prepared.reverse_order:
        if prepared.layers[lid].kind != "RELU":
            continue
        _relu_lines(prepared, lid, stats)
    frozen_relu_lines: Dict[
        int, Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for lid, line in prepared.relu_lines.items():
        for value in line:
            value.setflags(write=False)
        frozen_relu_lines[int(lid)] = line
    prepared.relu_lines = MappingProxyType(frozen_relu_lines)

    shard_ranges = []
    for worker_index in range(effective_workers):
        start = (query_count * worker_index) // effective_workers
        end = (query_count * (worker_index + 1)) // effective_workers
        if start < end:
            shard_ranges.append((worker_index, start, end))

    def run_shard(
        worker_index: int, start: int, end: int
    ) -> Tuple[int, int, int, np.ndarray, _ReplayStats]:
        local_prepared = _worker_prepared(prepared)
        local_stats = _ReplayStats()
        local_stats.configure_queries(query_count)
        local_values = np.empty(end - start, dtype=np.float64)
        for chunk_start in range(start, end, effective_chunk_size):
            local_prepared.deadline.check(force=True)
            chunk_end = min(end, chunk_start + effective_chunk_size)
            local_values[
                chunk_start - start : chunk_end - start
            ] = _replay_block(
                local_prepared, chunk_start, chunk_end, local_stats
            )
        local_prepared.deadline.check(force=True)
        return worker_index, start, end, local_values, local_stats

    with ThreadPoolExecutor(
        max_workers=effective_workers,
        thread_name_prefix="hybridz-proof-row",
    ) as executor:
        futures = [
            executor.submit(run_shard, worker_index, start, end)
            for worker_index, start, end in shard_ranges
        ]
        # Calling result in submission order makes receipt statistics
        # deterministic even when worker completion order differs.
        for future in futures:
            worker_index, start, end, local_values, local_stats = future.result()
            if worker_index < 0 or end - start != local_values.size:
                _fail("INTERNAL_ERROR", "parallel replay shard result is malformed")
            values[start:end] = local_values
            _merge_replay_stats(stats, local_stats)
    prepared.deadline.check(force=True)
    return values, stats, effective_chunk_size, effective_workers


def _receipt(
    prepared: _Prepared,
    lower_bounds: np.ndarray,
    stats: _ReplayStats,
    *,
    requested_chunk_size: int,
    effective_chunk_size: int,
    max_workspace_bytes: int,
    platform_contract: Mapping[str, Any],
    elapsed_s: float,
    proof_workers_requested: int = 1,
    proof_workers_effective: int = 1,
) -> Mapping[str, Any]:
    lower_hex = [float(value).hex() for value in lower_bounds]
    body: Dict[str, Any] = {
        "schema": _SCHEMA,
        "status": "verified",
        "proof_authority": True,
        "authority_source": "independent_reverse_topological_replay",
        "candidate_inputs_are_authoritative": False,
        "direction": "LOWER",
        "supported_operators": sorted(_SUPPORTED),
        "start_layer_id": prepared.output_id,
        "start_mode": prepared.start_mode,
        "input_spec_layer_id": prepared.input_spec_id,
        "query_count": int(lower_bounds.size),
        "requested_chunk_size": int(requested_chunk_size),
        "effective_chunk_size": int(effective_chunk_size),
        "max_workspace_bytes": int(max_workspace_bytes),
        "proof_row_parallelism": {
            "protocol": "disjoint_objective_rows_v1",
            "requested_workers": int(proof_workers_requested),
            "effective_workers": int(proof_workers_effective),
            "mutable_state": "worker_local_deadline_relu_cache_and_stats",
            "merge_order": "ascending_contiguous_row_shard",
            "partial_authority": False,
        },
        "numeric_method": {
            "device": "cpu",
            "nominal_dtype": "IEEE-754-binary64",
            "affine_error": "Higham-gamma-plus-subnormal-allowance",
            "coefficient_error": "absorbed-on-certified-predecessor-box",
            "relu_upper": "Fraction-endpoint-audit-plus-upward-intercept",
            "scalar_rounding": "nextafter-toward-negative-infinity",
            "outward_guard_arithmetic": "wider-longdouble-then-binary64-successor",
            "conv2d_adjoint": "audited-direct-sparse-scatter-or-kernel-offset-channel-GEMM",
        },
        "numeric_platform": dict(platform_contract),
        "trusted_assumption": "supplied_bounds_are_certified",
        "hashes": dict(prepared.hashes),
        "lower_bounds_hex": lower_hex,
        "lower_bounds_sha256": _array_digest(lower_bounds),
        "stats": {
            "affine_terms": stats.affine_terms,
            "coefficient_guards": stats.coefficient_guards,
            "scalar_guards": stats.scalar_guards,
            "fraction_endpoint_audits": stats.fraction_endpoint_audits,
            "relu_ambiguous_terms": stats.relu_ambiguous_terms,
            "dag_merges": stats.dag_merges,
            "conv_sparse_blocks": stats.conv_sparse_blocks,
            "conv_dense_blocks": stats.conv_dense_blocks,
            "guard_total_hex": float(stats.guard_total).hex(),
            "guard_max_hex": float(stats.guard_max).hex(),
            "guard_by_query_hex": [
                float(value).hex()
                for value in (
                    stats.guard_by_query
                    if stats.guard_by_query is not None
                    else np.zeros(lower_bounds.size, dtype=np.float64)
                )
            ],
            "guard_by_query_sha256": _array_digest(
                stats.guard_by_query
                if stats.guard_by_query is not None
                else np.zeros(lower_bounds.size, dtype=np.float64)
            ),
        },
        "elapsed_s_hex": float(elapsed_s).hex(),
    }
    body["receipt_sha256"] = _json_digest(body)
    return body


def _start_record(start_lid: Optional[int]) -> Any:
    return "ASSERT_PREDECESSOR" if start_lid is None else int(start_lid)


def _sealed_crosswalk(
    root_certificate: QueryDualBoxCertificate,
    root_graph: Any,
    contexts: Mapping[Optional[int], _SealedCone],
) -> Dict[str, Any]:
    """Build the exact root-to-cone bridge from recomputed cone semantics."""

    entries = []
    for key, cone in contexts.items():
        if key != cone.start_lid:
            _fail("INVALID_CONTEXT", "sealed replay cone start key changed")
        if not hmac.compare_digest(
            cone.manifest_sha256, cone.replay_net_sha256
        ):
            _fail("INVALID_CONTEXT", "sealed replay cone manifest changed")
        entries.append(
            {
                "start_layer": _start_record(cone.start_lid),
                "output_layer_id": cone.output_id,
                "start_mode": cone.start_mode,
                "forward_layer_ids": list(reversed(cone.reverse_order)),
                "replay_net_sha256": cone.replay_net_sha256,
            }
        )
    return {
        "source": "single_owned_cpu_f64_root_snapshot",
        "root_manifest_format": "query_dual_box_raw_parameter_manifest_v1",
        "replay_manifest_format": "query_dual_replay_normalized_cone_manifest_v1",
        "hashes_are_crosswalked_not_compared": True,
        "root_net_sha256": str(root_graph.root_net_sha256),
        "root_snapshot_content_sha256": str(root_graph.content_sha256),
        "root_certificate_receipt_sha256": str(
            root_certificate.receipt["receipt_sha256"]
        ),
        "replay_cones": entries,
    }


class QueryDualReplaySession:
    """One-shot V3 transaction over a root-owned frozen graph.

    Construction, bounds-frame sealing, numerical stage replay, and commit
    are intentionally separate.  Stage results are explicitly
    non-authoritative until ``commit`` performs one final live-network bind.
    """

    def __init__(
        self,
        *,
        authority: Any,
        net: Any,
        root_certificate: QueryDualBoxCertificate,
        root_graph: Any,
        full_layers: Mapping[int, _FrozenLayer],
        contexts: Mapping[Optional[int], _SealedCone],
        timer: _Deadline,
        platform_contract: Mapping[str, Any],
    ):
        if authority is not _SEALED_SESSION_CAPABILITY:
            _fail("INVALID_SESSION", "V3 sessions require a local capability")
        self._net = net
        self._root_certificate = root_certificate
        self._root_graph = root_graph
        self._full_layers = MappingProxyType(dict(full_layers))
        self._contexts = MappingProxyType(dict(contexts))
        self._deadline = timer
        self._deadline_identity_seal = id(timer)
        self._deadline_end_seal = timer.end
        self._platform_contract = MappingProxyType(dict(platform_contract))
        self._authority = _SEALED_SESSION_CAPABILITY
        self._nonce = secrets.token_hex(32)
        self._capability = object()
        self._frames: Dict[str, QueryDualReplayBoundsFrame] = {}
        self._frame_bounds_identities: Dict[str, int] = {}
        self._pending: List[_PendingStage] = []
        self._operation_lock = threading.Lock()
        self._closed = False
        self._failed = False
        self._crosswalk = MappingProxyType(self._build_crosswalk())
        self._crosswalk_sha256 = _json_digest(dict(self._crosswalk))
        self._required_bounds_ids = frozenset(
            lid
            for cone in self._contexts.values()
            for lid in cone.reverse_order
            if cone.layers[lid].kind != "INPUT"
        )
        self._static_identity_seal = (
            id(self._root_certificate),
            id(self._root_graph),
            id(self._full_layers),
            id(self._contexts),
            id(self._crosswalk),
            id(self._deadline),
            id(self._operation_lock),
            tuple(
                (key, id(cone), id(cone.layers))
                for key, cone in self._contexts.items()
            ),
        )
        self._static_manifest_commit_validations = 0
        _SEALED_SESSION_REGISTRY[self._nonce] = self

    @property
    def unique_context_count(self) -> int:
        return len(self._contexts)

    @property
    def static_manifest_commit_validations(self) -> int:
        return self._static_manifest_commit_validations

    def _build_crosswalk(self) -> Dict[str, Any]:
        return _sealed_crosswalk(
            self._root_certificate,
            self._root_graph,
            self._contexts,
        )

    def _validate_static_replay_manifests(self) -> None:
        """Hash every unique frozen layer once, then validate all cone hashes."""

        unique_lids = {
            lid for cone in self._contexts.values() for lid in cone.reverse_order
        }
        manifests = {
            lid: _layer_manifest(self._full_layers[lid])
            for lid in sorted(unique_lids)
        }
        for cone in self._contexts.values():
            actual = _json_digest(
                [
                    manifests[lid]
                    for lid in reversed(cone.reverse_order)
                ]
            )
            if (
                not hmac.compare_digest(actual, cone.manifest_sha256)
                or not hmac.compare_digest(actual, cone.replay_net_sha256)
            ):
                _fail("INVALID_CONTEXT", "sealed replay cone manifest changed")

    def _invalidate(self) -> None:
        self._failed = True
        self._closed = True
        _SEALED_SESSION_REGISTRY.pop(self._nonce, None)

    def abort(self) -> None:
        """Idempotently discard every provisional resource without authority."""

        self._operation_lock.acquire()
        try:
            self._invalidate()
            self._frames.clear()
            self._frame_bounds_identities.clear()
            self._pending.clear()
        finally:
            self._operation_lock.release()

    def _enter_operation(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            self._invalidate()
            _fail("CONCURRENT_SESSION", "concurrent V3 session use is forbidden")

    def _check(self) -> None:
        if (
            self._closed
            or self._failed
            or self._authority is not _SEALED_SESSION_CAPABILITY
            or _SEALED_SESSION_REGISTRY.get(self._nonce) is not self
        ):
            _fail("INVALID_SESSION", "V3 replay session is closed or unregistered")
        if (
            id(self._deadline) != self._deadline_identity_seal
            or self._deadline.end != self._deadline_end_seal
        ):
            self._invalidate()
            _fail("INVALID_DEADLINE", "V3 absolute deadline seal was modified")
        self._deadline.check(force=True)
        identity = (
            id(self._root_certificate),
            id(self._root_graph),
            id(self._full_layers),
            id(self._contexts),
            id(self._crosswalk),
            id(self._deadline),
            id(self._operation_lock),
            tuple(
                (key, id(cone), id(cone.layers))
                for key, cone in self._contexts.items()
            ),
        )
        if identity != self._static_identity_seal:
            self._invalidate()
            _fail("INVALID_CONTEXT", "V3 replay context identity seal was modified")

    def _check_frame(
        self, frame: QueryDualReplayBoundsFrame, *, full: bool = False
    ) -> None:
        if (
            not isinstance(frame, QueryDualReplayBoundsFrame)
            or frame._session_nonce != self._nonce
            or frame._capability is not self._capability
            or self._frames.get(frame._frame_nonce) is not frame
            or self._frame_bounds_identities.get(frame._frame_nonce)
            != id(frame._bounds)
        ):
            _fail("INVALID_FRAME", "bounds frame does not belong to this session")
        if not full:
            return
        records = _bounds_manifest(self._full_layers, frame._bounds)
        actual = _json_digest(
            {
                "session_nonce_sha256": hashlib.sha256(
                    self._nonce.encode("ascii")
                ).hexdigest(),
                "start_layers": [
                    _start_record(value) for value in frame._start_lids
                ],
                "bounds": records,
            }
        )
        if not hmac.compare_digest(actual, frame._content_sha256):
            _fail("INVALID_FRAME", "bounds frame seal was modified")

    def seal_bounds(
        self,
        certified_bounds: Mapping[Any, Any],
        *,
        start_lids: Optional[Sequence[Optional[int]]] = None,
    ) -> QueryDualReplayBoundsFrame:
        """Freeze one stage's complete consumed bounds set exactly once."""

        self._enter_operation()
        try:
            self._check()
            if not isinstance(certified_bounds, Mapping):
                _fail("INVALID_BOUNDS", "certified_bounds must be a mapping")
            if start_lids is None:
                frame_starts = tuple(self._contexts)
            else:
                if isinstance(start_lids, (str, bytes)) or not isinstance(
                    start_lids, Sequence
                ):
                    _fail(
                        "INVALID_FRAME",
                        "frame start_lids must be a nonempty sequence",
                    )
                selected: List[Optional[int]] = []
                for raw_start in start_lids:
                    if raw_start is None:
                        key = None
                    else:
                        if isinstance(raw_start, bool):
                            _fail(
                                "INVALID_START_LAYER",
                                "frame start_lid cannot be boolean",
                            )
                        try:
                            key = int(raw_start)
                        except Exception as exc:
                            raise QueryDualReplayError(
                                "INVALID_START_LAYER",
                                f"invalid frame start_lid {raw_start!r}",
                            ) from exc
                    if key not in self._contexts:
                        _fail(
                            "INVALID_CONTEXT",
                            f"frame start layer {_start_record(key)!r} was not sealed",
                        )
                    if key not in selected:
                        selected.append(key)
                if not selected:
                    _fail("INVALID_FRAME", "frame start_lids must be nonempty")
                frame_starts = tuple(selected)
            required_bounds_ids = frozenset(
                lid
                for key in frame_starts
                for lid in self._contexts[key].reverse_order
                if self._contexts[key].layers[lid].kind != "INPUT"
            )
            raw: Dict[int, Any] = {}
            for key, value in certified_bounds.items():
                try:
                    lid = int(key)
                except Exception as exc:
                    raise QueryDualReplayError(
                        "INVALID_BOUNDS", f"invalid bounds key {key!r}"
                    ) from exc
                if lid in raw:
                    _fail("INVALID_BOUNDS", f"duplicate bounds key {lid}")
                if lid not in self._full_layers:
                    _fail("INVALID_BOUNDS", f"unknown bounds layer {lid}")
                raw[lid] = value
            missing = required_bounds_ids - set(raw)
            if missing:
                _fail(
                    "MISSING_BOUNDS",
                    f"bounds frame is missing consumed layers {sorted(missing)}",
                )
            boxes: Dict[int, _Box] = {}
            for lid in sorted(required_bounds_ids):
                self._deadline.check()
                box = _immutable_box_from_value(raw[lid], layer_id=lid)
                layer = self._full_layers[lid]
                if box.lb.size != layer.width:
                    _fail(
                        "SHAPE_MISMATCH",
                        f"bounds[{lid}] width {box.lb.size} != layer width {layer.width}",
                    )
                boxes[lid] = box
            frame_nonce = secrets.token_hex(32)
            records = _bounds_manifest(self._full_layers, boxes)
            content_sha = _json_digest(
                {
                    "session_nonce_sha256": hashlib.sha256(
                        self._nonce.encode("ascii")
                    ).hexdigest(),
                    "start_layers": [
                        _start_record(value) for value in frame_starts
                    ],
                    "bounds": records,
                }
            )
            frame = QueryDualReplayBoundsFrame(
                _session_nonce=self._nonce,
                _frame_nonce=frame_nonce,
                _bounds=MappingProxyType(boxes),
                _start_lids=frame_starts,
                _content_sha256=content_sha,
                _capability=self._capability,
            )
            self._frames[frame_nonce] = frame
            self._frame_bounds_identities[frame_nonce] = id(frame._bounds)
            self._deadline.check(force=True)
            return frame
        except Exception:
            self._invalidate()
            raise
        finally:
            self._operation_lock.release()

    def replay(
        self,
        frame: QueryDualReplayBoundsFrame,
        *,
        start_lid: Optional[int] = None,
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
        proof_workers: int = 1,
    ) -> QueryDualReplayPendingResult:
        """Run the unchanged numerical core and return a pending stage value."""

        started = time.monotonic()
        self._enter_operation()
        try:
            self._check()
            self._check_frame(frame)
            if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
                _fail("INVALID_CHUNK", "chunk_size must be a positive integer")
            if (
                not isinstance(max_workspace_bytes, int)
                or isinstance(max_workspace_bytes, bool)
                or max_workspace_bytes <= 0
            ):
                _fail(
                    "INVALID_CHUNK",
                    "max_workspace_bytes must be a positive integer",
                )
            if (
                not isinstance(proof_workers, int)
                or isinstance(proof_workers, bool)
                or proof_workers <= 0
                or proof_workers > 32
            ):
                _fail(
                    "INVALID_WORKERS",
                    "proof_workers must be an integer in [1, 32]",
                )
            key: Optional[int]
            if start_lid is None:
                key = None
            else:
                if isinstance(start_lid, bool):
                    _fail("INVALID_START_LAYER", "start_lid cannot be boolean")
                try:
                    key = int(start_lid)
                except Exception as exc:
                    raise QueryDualReplayError(
                        "INVALID_START_LAYER", f"invalid start_lid {start_lid!r}"
                    ) from exc
            cone = self._contexts.get(key)
            if cone is None:
                _fail(
                    "INVALID_CONTEXT",
                    f"start layer {_start_record(key)!r} was not sealed",
                )
            if key not in frame._start_lids:
                _fail(
                    "INVALID_FRAME",
                    f"bounds frame does not cover start {_start_record(key)!r}",
                )
            prepared = _prepare_from_sealed(
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
            values, stats, effective_chunk_size, effective_workers = (
                _replay_objective_workers(
                    prepared,
                    requested_workers=int(proof_workers),
                    chunk_size=int(chunk_size),
                    max_workspace_bytes=int(max_workspace_bytes),
                )
            )
            self._deadline.check(force=True)
            if not np.all(np.isfinite(values)):
                _fail("NONFINITE", "non-finite final lower bounds")
            frozen_values = _immutable_f64_array(
                values, name="pending_lower_bounds"
            )
            base_receipt = _receipt(
                prepared,
                frozen_values,
                stats,
                requested_chunk_size=chunk_size,
                effective_chunk_size=effective_chunk_size,
                max_workspace_bytes=max_workspace_bytes,
                platform_contract=self._platform_contract,
                elapsed_s=time.monotonic() - started,
                proof_workers_requested=int(proof_workers),
                proof_workers_effective=int(effective_workers),
            )
            token = secrets.token_hex(32)
            public = QueryDualReplayPendingResult(
                lower_bounds=frozen_values,
                stage_token=token,
            )
            content_sha = _json_digest(
                {
                    "stage_token_sha256": hashlib.sha256(
                        token.encode("ascii")
                    ).hexdigest(),
                    "frame_sha256": frame._content_sha256,
                    "base_receipt_sha256": base_receipt["receipt_sha256"],
                    "lower_bounds_sha256": _array_digest(frozen_values),
                }
            )
            self._pending.append(
                _PendingStage(
                    public=public,
                    prepared=prepared,
                    base_receipt=MappingProxyType(dict(base_receipt)),
                    frame_nonce=frame._frame_nonce,
                    frame_sha256=frame._content_sha256,
                    content_sha256=content_sha,
                )
            )
            self._deadline.check(force=True)
            return public
        except Exception:
            self._invalidate()
            raise
        finally:
            self._operation_lock.release()

    def commit(self) -> Tuple[QueryDualReplayResult, ...]:
        """Bind the live net once, then promote every pending result."""

        self._enter_operation()
        try:
            self._check()
            if not self._pending:
                _fail("EMPTY_SESSION", "cannot commit a session with no replay stages")
            if not verify_query_dual_box_certificate(self._root_certificate):
                _fail("INVALID_ROOT_CERTIFICATE", "root certificate was modified")
            try:
                graph = _borrow_sealed_query_dual_graph(self._root_certificate)
            except QueryDualBoxError as exc:
                raise QueryDualReplayError(
                    "INVALID_ROOT_CERTIFICATE", str(exc)
                ) from exc
            if graph is not self._root_graph:
                _fail("INVALID_ROOT_CERTIFICATE", "root frozen graph identity changed")
            self._validate_static_replay_manifests()
            actual_crosswalk = self._build_crosswalk()
            self._static_manifest_commit_validations += 1
            if (
                actual_crosswalk != dict(self._crosswalk)
                or not hmac.compare_digest(
                    _json_digest(actual_crosswalk), self._crosswalk_sha256
                )
            ):
                _fail("INVALID_CONTEXT", "V3 replay context seal was modified")
            validated_frames: set[str] = set()
            for pending in self._pending:
                frame = self._frames.get(pending.frame_nonce)
                if frame is None:
                    _fail("INVALID_FRAME", "pending stage lost its bounds frame")
                if pending.frame_nonce not in validated_frames:
                    self._check_frame(frame, full=True)
                    validated_frames.add(pending.frame_nonce)
                public = pending.public
                expected_content = _json_digest(
                    {
                        "stage_token_sha256": hashlib.sha256(
                            public.stage_token.encode("ascii")
                        ).hexdigest(),
                        "frame_sha256": pending.frame_sha256,
                        "base_receipt_sha256": pending.base_receipt[
                            "receipt_sha256"
                        ],
                        "lower_bounds_sha256": _array_digest(
                            public.lower_bounds
                        ),
                    }
                )
                if (
                    not hmac.compare_digest(
                        expected_content, pending.content_sha256
                    )
                    or not validate_query_dual_replay_result(
                        QueryDualReplayResult(
                            lower_bounds=public.lower_bounds,
                            receipt=pending.base_receipt,
                        )
                    )
                ):
                    _fail("INVALID_STAGE", "pending replay stage was modified")
            self._deadline.check(force=True)
            if not verify_query_dual_box_certificate(
                self._root_certificate, net=self._net
            ):
                _fail("LIVE_NET_MISMATCH", "live network changed before V3 commit")
            self._deadline.check(force=True)

            committed: List[QueryDualReplayResult] = []
            nonce_sha = hashlib.sha256(self._nonce.encode("ascii")).hexdigest()
            for pending in self._pending:
                body = dict(pending.base_receipt)
                body.pop("receipt_sha256", None)
                body["schema"] = _SEALED_SCHEMA
                body[
                    "authority_source"
                ] = "independent_reverse_topological_replay_sealed_transaction"
                sealed_context: Dict[str, Any] = {
                    "protocol": _SEALED_PROTOCOL,
                    "session_nonce_sha256": nonce_sha,
                    "live_net_commit_bound": True,
                    "live_net_bind": "root_certificate_full_live_verification_once_at_commit",
                    "network_snapshot_freeze_count": 1,
                    "unique_cone_count": len(self._contexts),
                    "bounds_frame_sha256": pending.frame_sha256,
                    "root_net_sha256": self._root_graph.root_net_sha256,
                    "replay_net_sha256": pending.prepared.hashes["net_sha256"],
                    "manifest_crosswalk": dict(self._crosswalk),
                    "manifest_crosswalk_sha256": self._crosswalk_sha256,
                }
                sealed_context["context_sha256"] = _json_digest(sealed_context)
                body["sealed_context"] = sealed_context
                body["receipt_sha256"] = _json_digest(body)
                result = QueryDualReplayResult(
                    lower_bounds=pending.public.lower_bounds,
                    receipt=MappingProxyType(body),
                )
                committed.append(result)
            self._closed = True
            _SEALED_SESSION_REGISTRY.pop(self._nonce, None)
            return tuple(committed)
        except Exception:
            self._invalidate()
            raise
        finally:
            self._operation_lock.release()


def create_query_dual_replay_session(
    net: Any,
    root_certificate: QueryDualBoxCertificate,
    start_lids: Sequence[Optional[int]],
    *,
    deadline: float,
) -> QueryDualReplaySession:
    """Create a V3 session from the exact graph frozen by the root certifier."""

    if deadline is None:
        _fail("INVALID_DEADLINE", "V3 requires an absolute monotonic deadline")
    try:
        absolute = float(deadline)
    except Exception as exc:
        raise QueryDualReplayError(
            "INVALID_DEADLINE", "deadline must be a finite absolute timestamp"
        ) from exc
    if not math.isfinite(absolute):
        _fail("INVALID_DEADLINE", "deadline must be a finite absolute timestamp")
    timer = _Deadline(end=absolute)
    timer.check(force=True)
    platform_contract = _check_numeric_platform()
    timer.check(force=True)
    if not verify_query_dual_box_certificate(root_certificate):
        _fail("INVALID_ROOT_CERTIFICATE", "root certificate is invalid")
    try:
        # Registry identity plus immutable-bytes/mapping seals are sufficient
        # during construction.  Full content is rehashed once at commit.
        root_graph = _borrow_sealed_query_dual_graph(
            root_certificate, validate_content=False
        )
    except QueryDualBoxError as exc:
        raise QueryDualReplayError("INVALID_ROOT_CERTIFICATE", str(exc)) from exc
    if isinstance(start_lids, (str, bytes)) or not isinstance(
        start_lids, Sequence
    ):
        _fail("INVALID_CONTEXT", "start_lids must be a nonempty sequence")
    requested: List[Optional[int]] = []
    for raw in start_lids:
        if raw is None:
            key = None
        else:
            if isinstance(raw, bool):
                _fail("INVALID_START_LAYER", "start_lid cannot be boolean")
            try:
                key = int(raw)
            except Exception as exc:
                raise QueryDualReplayError(
                    "INVALID_START_LAYER", f"invalid start_lid {raw!r}"
                ) from exc
        if key not in requested:
            requested.append(key)
    if not requested:
        _fail("INVALID_CONTEXT", "start_lids must be nonempty")
    full_layers = MappingProxyType(
        {
            int(layer.id): _replay_layer_from_root(layer)
            for layer in root_graph.layers
        }
    )
    layer_manifests = {
        lid: _layer_manifest(layer) for lid, layer in full_layers.items()
    }
    contexts: Dict[Optional[int], _SealedCone] = {}
    for key in requested:
        timer.check()
        contexts[key] = _sealed_cone(
            full_layers,
            layer_manifests,
            assert_id=int(root_graph.assert_id),
            start_lid=key,
        )
    timer.check(force=True)
    return QueryDualReplaySession(
        authority=_SEALED_SESSION_CAPABILITY,
        net=net,
        root_certificate=root_certificate,
        root_graph=root_graph,
        full_layers=full_layers,
        contexts=contexts,
        timer=timer,
        platform_contract=platform_contract,
    )


def _build_query_dual_replay_validation_context(
    root_certificate: QueryDualBoxCertificate,
    start_lids: Sequence[Optional[int]],
) -> _SealedReplayValidationContext:
    """Recompute every V3 cone and crosswalk field from the sealed root.

    This helper is intentionally separate from receipt verification: a
    receipt SHA is only an integrity checksum, whereas an authority consumer
    must derive the expected cone partition and normalized manifests from the
    process-local root certificate.
    """

    if not verify_query_dual_box_certificate(root_certificate):
        _fail("INVALID_ROOT_CERTIFICATE", "root certificate is invalid")
    try:
        root_graph = _borrow_sealed_query_dual_graph(
            root_certificate, validate_content=True
        )
    except QueryDualBoxError as exc:
        raise QueryDualReplayError(
            "INVALID_ROOT_CERTIFICATE", str(exc)
        ) from exc
    if isinstance(start_lids, (str, bytes)) or not isinstance(
        start_lids, Sequence
    ):
        _fail("INVALID_CONTEXT", "start_lids must be a nonempty sequence")
    requested: List[Optional[int]] = []
    for raw in start_lids:
        if raw is None:
            key = None
        else:
            if isinstance(raw, bool):
                _fail("INVALID_START_LAYER", "start_lid cannot be boolean")
            try:
                key = int(raw)
            except Exception as exc:
                raise QueryDualReplayError(
                    "INVALID_START_LAYER", f"invalid start_lid {raw!r}"
                ) from exc
        if key not in requested:
            requested.append(key)
    if not requested:
        _fail("INVALID_CONTEXT", "start_lids must be nonempty")
    full_layers = MappingProxyType(
        {
            int(layer.id): _replay_layer_from_root(layer)
            for layer in root_graph.layers
        }
    )
    manifests = {
        lid: _layer_manifest(layer) for lid, layer in full_layers.items()
    }
    contexts: Dict[Optional[int], _SealedCone] = {}
    for key in requested:
        contexts[key] = _sealed_cone(
            full_layers,
            manifests,
            assert_id=int(root_graph.assert_id),
            start_lid=key,
        )
    frozen_contexts = MappingProxyType(contexts)
    crosswalk = MappingProxyType(
        _sealed_crosswalk(
            root_certificate,
            root_graph,
            frozen_contexts,
        )
    )
    return _SealedReplayValidationContext(
        full_layers=full_layers,
        contexts=frozen_contexts,
        crosswalk=crosswalk,
    )


def _query_dual_replay_frame_payload(
    context: _SealedReplayValidationContext,
    certified_bounds: Mapping[Any, Any],
    *,
    start_lids: Sequence[Optional[int]],
) -> Mapping[str, Any]:
    """Recompute the session-independent part of a targeted frame seal."""

    if not isinstance(context, _SealedReplayValidationContext):
        _fail("INVALID_CONTEXT", "invalid replay validation context")
    if not isinstance(certified_bounds, Mapping):
        _fail("INVALID_BOUNDS", "certified_bounds must be a mapping")
    if isinstance(start_lids, (str, bytes)) or not isinstance(
        start_lids, Sequence
    ):
        _fail("INVALID_FRAME", "frame start_lids must be a nonempty sequence")
    selected: List[Optional[int]] = []
    for raw in start_lids:
        if raw is None:
            key = None
        else:
            if isinstance(raw, bool):
                _fail("INVALID_START_LAYER", "frame start_lid cannot be boolean")
            try:
                key = int(raw)
            except Exception as exc:
                raise QueryDualReplayError(
                    "INVALID_START_LAYER",
                    f"invalid frame start_lid {raw!r}",
                ) from exc
        if key not in context.contexts:
            _fail(
                "INVALID_CONTEXT",
                f"frame start layer {_start_record(key)!r} was not sealed",
            )
        if key not in selected:
            selected.append(key)
    if not selected:
        _fail("INVALID_FRAME", "frame start_lids must be nonempty")
    required = frozenset(
        lid
        for key in selected
        for lid in context.contexts[key].reverse_order
        if context.contexts[key].layers[lid].kind != "INPUT"
    )
    raw_bounds: Dict[int, Any] = {}
    for raw_lid, value in certified_bounds.items():
        if isinstance(raw_lid, bool):
            _fail("INVALID_BOUNDS", "bounds layer id cannot be boolean")
        try:
            lid = int(raw_lid)
        except Exception as exc:
            raise QueryDualReplayError(
                "INVALID_BOUNDS", f"invalid bounds key {raw_lid!r}"
            ) from exc
        if lid in raw_bounds:
            _fail("INVALID_BOUNDS", f"duplicate bounds key {lid}")
        if lid not in context.full_layers:
            _fail("INVALID_BOUNDS", f"unknown bounds layer {lid}")
        raw_bounds[lid] = value
    missing = required - set(raw_bounds)
    if missing:
        _fail(
            "MISSING_BOUNDS",
            f"bounds frame is missing consumed layers {sorted(missing)}",
        )
    boxes: Dict[int, _Box] = {}
    for lid in sorted(required):
        box = _immutable_box_from_value(raw_bounds[lid], layer_id=lid)
        if box.lb.size != context.full_layers[lid].width:
            _fail(
                "SHAPE_MISMATCH",
                f"bounds[{lid}] width {box.lb.size} != "
                f"layer width {context.full_layers[lid].width}",
            )
        boxes[lid] = box
    return {
        "start_layers": [_start_record(value) for value in selected],
        "bounds": _bounds_manifest(context.full_layers, boxes),
    }


def _query_dual_replay_frame_sha256(
    payload: Mapping[str, Any],
    *,
    session_nonce_sha256: str,
) -> str:
    """Finish a recomputed frame seal once the committed session is known."""

    nonce_sha = str(session_nonce_sha256)
    if (
        len(nonce_sha) != 64
        or any(value not in "0123456789abcdef" for value in nonce_sha)
    ):
        _fail("INVALID_SESSION", "session nonce SHA-256 is malformed")
    if not isinstance(payload, Mapping):
        _fail("INVALID_FRAME", "frame payload is malformed")
    return _json_digest(
        {
            "session_nonce_sha256": nonce_sha,
            "start_layers": list(payload["start_layers"]),
            "bounds": list(payload["bounds"]),
        }
    )


def replay_query_lower_bounds(
    net: Any,
    certified_bounds: Mapping[Any, Any],
    *,
    start_lid: Optional[int] = None,
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
    deadline: Optional[float] = None,
    timeout_s: Optional[float] = None,
) -> QueryDualReplayResult:
    """Replay lower bounds and issue an integrity-checkable proof receipt.

    ``certified_bounds[relu_id]`` must contain the ReLU *pre-activation* box;
    other consumed entries contain layer-output boxes.  Bounds may be
    ``Bounds``, ``Fact``, ``{"lb", "ub"}``, or ``(lb, ub)`` values.

    ``alpha_by_relu`` accepts stored binary64 scalars, vectors, or ``[Q,n]``
    arrays.  Missing ReLU entries deterministically use alpha zero.

    ``start_lid=None`` replays from the ASSERT predecessor.  An explicit
    ``start_lid`` replays an interior one-hot/query row and hashes only that
    layer's ancestor cone.  In particular, a pre-ReLU candidate should pass
    the affine predecessor id, not the target ReLU id.

    ``deadline`` is an absolute :func:`time.monotonic` timestamp.
    """

    started = time.monotonic()
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        _fail("INVALID_CHUNK", "chunk_size must be a positive integer")
    if not isinstance(max_workspace_bytes, int) or max_workspace_bytes <= 0:
        _fail("INVALID_CHUNK", "max_workspace_bytes must be a positive integer")
    timer = _Deadline.build(deadline, timeout_s)
    timer.check(force=True)
    platform_contract = _check_numeric_platform()
    timer.check(force=True)
    prepared = _prepare(
        net,
        certified_bounds,
        start_lid=start_lid,
        query_rows=query_rows,
        one_hot=one_hot,
        query_bias=query_bias,
        alpha_by_relu=alpha_by_relu,
        deadline=timer,
        expected_net_sha256=expected_net_sha256,
        expected_bounds_sha256=expected_bounds_sha256,
        expected_query_sha256=expected_query_sha256,
        expected_alpha_sha256=expected_alpha_sha256,
    )
    maximum_width = max(layer.width for layer in prepared.layers.values())
    # Nominal, absolute mass, radius, relaxation factors, and DAG temporaries
    # coexist at the widest layer.  Twelve arrays per row is conservative.
    bytes_per_query = max(1, maximum_width * 8 * 12)
    memory_limited = max(1, max_workspace_bytes // bytes_per_query)
    effective_chunk_size = min(chunk_size, memory_limited, prepared.queries.shape[0])
    stats = _ReplayStats()
    stats.configure_queries(prepared.queries.shape[0])
    result = np.empty(prepared.queries.shape[0], dtype=np.float64)
    for chunk_start in range(0, result.size, effective_chunk_size):
        timer.check(force=True)
        chunk_end = min(result.size, chunk_start + effective_chunk_size)
        result[chunk_start:chunk_end] = _replay_block(
            prepared, chunk_start, chunk_end, stats
        )
    timer.check(force=True)
    if not np.all(np.isfinite(result)):
        _fail("NONFINITE", "non-finite final lower bounds")
    result.setflags(write=False)
    receipt = _receipt(
        prepared,
        result,
        stats,
        requested_chunk_size=chunk_size,
        effective_chunk_size=effective_chunk_size,
        max_workspace_bytes=max_workspace_bytes,
        platform_contract=platform_contract,
        elapsed_s=time.monotonic() - started,
    )
    timer.check(force=True)
    return QueryDualReplayResult(lower_bounds=result, receipt=receipt)


def replay_query_affine_lower_to_layer(
    net: Any,
    certified_bounds: Mapping[Any, Any],
    *,
    stop_lid: int,
    query_rows: Any,
    query_bias: Optional[Any] = None,
    alpha_by_relu: Optional[Mapping[Any, Any]] = None,
    expected_net_sha256: Optional[str] = None,
    expected_bounds_sha256: Optional[str] = None,
    expected_query_sha256: Optional[str] = None,
    expected_alpha_sha256: Optional[str] = None,
    chunk_size: int = 1024,
    max_workspace_bytes: int = 512 * 1024 * 1024,
    deadline: Optional[float] = None,
    timeout_s: Optional[float] = None,
) -> QueryDualAffineLowerPlane:
    """Replay a lower affine predicate only through a network suffix.

    ``stop_lid`` must dominate every path from the ASSERT predecessor.  The
    returned plane is relative to that layer's output and is useful for
    composing a property-conditioned suffix with a separately constructed
    sound prefix abstraction.
    """

    started = time.monotonic()
    if (
        not isinstance(stop_lid, int)
        or isinstance(stop_lid, bool)
        or stop_lid < 0
    ):
        _fail("INVALID_START_LAYER", "stop_lid must be a nonnegative integer")
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        _fail("INVALID_CHUNK", "chunk_size must be a positive integer")
    if (
        not isinstance(max_workspace_bytes, int)
        or isinstance(max_workspace_bytes, bool)
        or max_workspace_bytes <= 0
    ):
        _fail(
            "INVALID_CHUNK",
            "max_workspace_bytes must be a positive integer",
        )
    timer = _Deadline.build(deadline, timeout_s)
    timer.check(force=True)
    platform_contract = _check_numeric_platform()
    prepared = _prepare(
        net,
        certified_bounds,
        start_lid=None,
        query_rows=query_rows,
        one_hot=None,
        query_bias=query_bias,
        alpha_by_relu=alpha_by_relu,
        deadline=timer,
        expected_net_sha256=expected_net_sha256,
        expected_bounds_sha256=expected_bounds_sha256,
        expected_query_sha256=expected_query_sha256,
        expected_alpha_sha256=expected_alpha_sha256,
    )
    stop = int(stop_lid)
    if stop not in prepared.layers or stop not in prepared.reverse_order:
        _fail("INVALID_START_LAYER", f"stop layer {stop} is not in the query cone")
    if prepared.layers[stop].kind in {"INPUT", "INPUT_SPEC", "ASSERT"}:
        _fail(
            "INVALID_START_LAYER",
            "affine replay stop must be an interior value-producing layer",
        )
    query_count = int(prepared.queries.shape[0])
    stop_width = int(prepared.layers[stop].width)
    maximum_width = max(layer.width for layer in prepared.layers.values())
    bytes_per_query = max(
        1,
        (maximum_width * 12 + stop_width) * 8,
    )
    memory_limited = max(1, max_workspace_bytes // bytes_per_query)
    effective_chunk_size = min(chunk_size, memory_limited, query_count)
    stats = _ReplayStats()
    stats.configure_queries(query_count)
    scalar = np.empty(query_count, dtype=np.float64)
    coefficients = np.empty((query_count, stop_width), dtype=np.float64)
    for chunk_start in range(0, query_count, effective_chunk_size):
        timer.check(force=True)
        chunk_end = min(query_count, chunk_start + effective_chunk_size)
        block_scalar, block_coefficients = _replay_affine_block(
            prepared,
            chunk_start,
            chunk_end,
            stats,
            stop_lid=stop,
        )
        scalar[chunk_start:chunk_end] = block_scalar
        coefficients[chunk_start:chunk_end, :] = block_coefficients
    timer.check(force=True)
    if (
        not np.all(np.isfinite(scalar))
        or not np.all(np.isfinite(coefficients))
    ):
        _fail("NONFINITE", "affine replay produced non-finite values")
    frozen_scalar = _immutable_f64_array(
        scalar, name="affine_lower_scalar"
    )
    frozen_coefficients = _immutable_f64_array(
        coefficients, name="affine_lower_coefficients"
    )
    # Reuse the audited numeric/stats serialization, then replace the
    # full-bound quantity with the affine inequality actually proved.
    base = dict(
        _receipt(
            prepared,
            frozen_scalar,
            stats,
            requested_chunk_size=chunk_size,
            effective_chunk_size=effective_chunk_size,
            max_workspace_bytes=max_workspace_bytes,
            platform_contract=platform_contract,
            elapsed_s=time.monotonic() - started,
        )
    )
    base.pop("receipt_sha256", None)
    base.pop("lower_bounds_hex", None)
    base.pop("lower_bounds_sha256", None)
    base.update(
        {
            "schema": _AFFINE_SCHEMA,
            "status": "verified_affine_lower",
            "authority_source": (
                "independent_reverse_topological_suffix_replay"
            ),
            "quantity": (
                "scalar_plus_coefficients_at_stop_le_query_expression"
            ),
            "full_input_support_not_computed": True,
            "stop_layer_id": stop,
            "stop_layer_kind": prepared.layers[stop].kind,
            "stop_layer_width": stop_width,
            "coefficients_shape": [
                int(value) for value in frozen_coefficients.shape
            ],
            "coefficients_sha256": _array_digest(
                frozen_coefficients
            ),
            "scalar_hex": [
                float(value).hex() for value in frozen_scalar
            ],
            "scalar_sha256": _array_digest(frozen_scalar),
        }
    )
    base["receipt_sha256"] = _json_digest(base)
    timer.check(force=True)
    result = QueryDualAffineLowerPlane(
        coefficients=frozen_coefficients,
        scalar=frozen_scalar,
        receipt=MappingProxyType(base),
    )
    if not validate_query_dual_affine_lower_plane(result):
        _fail("INVALID_RECEIPT", "affine replay self-validation failed")
    return result


def validate_query_dual_affine_lower_plane(
    result: QueryDualAffineLowerPlane,
    *,
    expected_net_sha256: Optional[str] = None,
    expected_bounds_sha256: Optional[str] = None,
    expected_query_sha256: Optional[str] = None,
    expected_alpha_sha256: Optional[str] = None,
) -> bool:
    """Validate immutable arrays, receipt integrity, and optional hash pins."""

    try:
        if (
            not isinstance(result, QueryDualAffineLowerPlane)
            or result.proof_authority is not True
        ):
            return False
        coefficients = result.coefficients
        scalar = result.scalar
        receipt = result.receipt
        if (
            not isinstance(coefficients, np.ndarray)
            or coefficients.dtype != np.float64
            or coefficients.ndim != 2
            or coefficients.flags.writeable
            or not coefficients.flags.c_contiguous
            or not isinstance(scalar, np.ndarray)
            or scalar.dtype != np.float64
            or scalar.ndim != 1
            or scalar.flags.writeable
            or not scalar.flags.c_contiguous
            or coefficients.shape[0] != scalar.size
            or not np.all(np.isfinite(coefficients))
            or not np.all(np.isfinite(scalar))
        ):
            return False
        body = dict(receipt)
        claimed = str(body.pop("receipt_sha256"))
        hashes = receipt.get("hashes")
        return bool(
            receipt.get("schema") == _AFFINE_SCHEMA
            and receipt.get("status") == "verified_affine_lower"
            and receipt.get("proof_authority") is True
            and receipt.get("direction") == "LOWER"
            and receipt.get("query_count") == int(scalar.size)
            and receipt.get("stop_layer_width")
            == int(coefficients.shape[1])
            and receipt.get("coefficients_shape")
            == [int(value) for value in coefficients.shape]
            and receipt.get("coefficients_sha256")
            == _array_digest(coefficients)
            and receipt.get("scalar_sha256") == _array_digest(scalar)
            and receipt.get("scalar_hex")
            == [float(value).hex() for value in scalar]
            and hmac.compare_digest(claimed, _json_digest(body))
            and isinstance(hashes, Mapping)
            and (
                expected_net_sha256 is None
                or hmac.compare_digest(
                    str(hashes.get("net_sha256", "")),
                    str(expected_net_sha256),
                )
            )
            and (
                expected_bounds_sha256 is None
                or hmac.compare_digest(
                    str(hashes.get("bounds_sha256", "")),
                    str(expected_bounds_sha256),
                )
            )
            and (
                expected_query_sha256 is None
                or hmac.compare_digest(
                    str(hashes.get("query_sha256", "")),
                    str(expected_query_sha256),
                )
            )
            and (
                expected_alpha_sha256 is None
                or hmac.compare_digest(
                    str(hashes.get("alpha_sha256", "")),
                    str(expected_alpha_sha256),
                )
            )
        )
    except Exception:
        return False


def verify_query_dual_replay_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_net_sha256: Optional[str] = None,
    expected_bounds_sha256: Optional[str] = None,
    expected_query_sha256: Optional[str] = None,
    expected_alpha_sha256: Optional[str] = None,
) -> bool:
    """Check receipt integrity and pinned hashes (this does not re-run proof)."""

    try:
        body = dict(receipt)
        claimed = str(body.pop("receipt_sha256"))
        schema = body.get("schema")
        if schema not in {_SCHEMA, _SEALED_SCHEMA}:
            return False
        if body.get("status") != "verified" or body.get("proof_authority") is not True:
            return False
        if body.get("candidate_inputs_are_authoritative") is not False:
            return False
        if schema == _SEALED_SCHEMA:
            sealed = dict(body["sealed_context"])
            context_claimed = str(sealed.pop("context_sha256"))
            if (
                body.get("authority_source")
                != (
                    "independent_reverse_topological_replay_"
                    "sealed_transaction"
                )
                or sealed.get("protocol") != _SEALED_PROTOCOL
                or sealed.get("live_net_commit_bound") is not True
                or sealed.get("live_net_bind")
                != (
                    "root_certificate_full_live_verification_"
                    "once_at_commit"
                )
                or isinstance(
                    sealed.get("network_snapshot_freeze_count"), bool
                )
                or not isinstance(
                    sealed.get("network_snapshot_freeze_count"), int
                )
                or sealed.get("network_snapshot_freeze_count") != 1
                or not hmac.compare_digest(
                    _json_digest(sealed), context_claimed
                )
            ):
                return False
            crosswalk = sealed["manifest_crosswalk"]
            if not isinstance(crosswalk, Mapping):
                return False
            replay_cones = crosswalk.get("replay_cones", ())
            if (
                not isinstance(replay_cones, list)
                or not replay_cones
                or isinstance(sealed.get("unique_cone_count"), bool)
                or not isinstance(sealed.get("unique_cone_count"), int)
                or sealed.get("unique_cone_count") != len(replay_cones)
                or set(crosswalk)
                != {
                    "source",
                    "root_manifest_format",
                    "replay_manifest_format",
                    "hashes_are_crosswalked_not_compared",
                    "root_net_sha256",
                    "root_snapshot_content_sha256",
                    "root_certificate_receipt_sha256",
                    "replay_cones",
                }
            ):
                return False
            cone_tokens = []
            for entry in replay_cones:
                if (
                    not isinstance(entry, Mapping)
                    or set(entry)
                    != {
                        "start_layer",
                        "output_layer_id",
                        "start_mode",
                        "forward_layer_ids",
                        "replay_net_sha256",
                    }
                ):
                    return False
                output_id = entry.get("output_layer_id")
                forward_ids = entry.get("forward_layer_ids")
                start_mode = entry.get("start_mode")
                start_layer = entry.get("start_layer")
                if (
                    isinstance(output_id, bool)
                    or not isinstance(output_id, int)
                    or not isinstance(forward_ids, list)
                    or not forward_ids
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        for value in forward_ids
                    )
                    or len(set(forward_ids)) != len(forward_ids)
                    or forward_ids[-1] != output_id
                    or start_mode
                    not in {"ASSERT_PREDECESSOR", "EXPLICIT_INTERIOR"}
                    or (
                        start_mode == "ASSERT_PREDECESSOR"
                        and start_layer != "ASSERT_PREDECESSOR"
                    )
                    or (
                        start_mode == "EXPLICIT_INTERIOR"
                        and (
                            isinstance(start_layer, bool)
                            or not isinstance(start_layer, int)
                            or start_layer != output_id
                        )
                    )
                ):
                    return False
                cone_tokens.append((start_mode, start_layer))
            receipt_start_mode = body.get("start_mode")
            receipt_start_id = body.get("start_layer_id")
            if (
                receipt_start_mode
                not in {"ASSERT_PREDECESSOR", "EXPLICIT_INTERIOR"}
                or isinstance(receipt_start_id, bool)
                or not isinstance(receipt_start_id, int)
            ):
                return False
            expected_token = (
                ("ASSERT_PREDECESSOR", "ASSERT_PREDECESSOR")
                if receipt_start_mode == "ASSERT_PREDECESSOR"
                else ("EXPLICIT_INTERIOR", receipt_start_id)
            )
            matching = [
                entry
                for entry, token in zip(replay_cones, cone_tokens)
                if token == expected_token
            ]
            if (
                crosswalk.get("source")
                != "single_owned_cpu_f64_root_snapshot"
                or crosswalk.get("root_manifest_format")
                != "query_dual_box_raw_parameter_manifest_v1"
                or crosswalk.get("replay_manifest_format")
                != "query_dual_replay_normalized_cone_manifest_v1"
                or crosswalk.get("hashes_are_crosswalked_not_compared") is not True
                or len(set(cone_tokens)) != len(cone_tokens)
                or len(matching) != 1
                or not hmac.compare_digest(
                    _json_digest(crosswalk),
                    str(sealed["manifest_crosswalk_sha256"]),
                )
                or not hmac.compare_digest(
                    str(sealed["root_net_sha256"]),
                    str(crosswalk["root_net_sha256"]),
                )
                or not any(
                    hmac.compare_digest(
                        str(sealed["replay_net_sha256"]),
                        str(entry["replay_net_sha256"]),
                    )
                    for entry in matching
                )
            ):
                return False
        if not hmac.compare_digest(_json_digest(body), claimed):
            return False
        values = np.asarray(
            [float.fromhex(value) for value in body["lower_bounds_hex"]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            return False
        if not hmac.compare_digest(
            _array_digest(values), str(body["lower_bounds_sha256"])
        ):
            return False
        guard_values = np.asarray(
            [
                float.fromhex(value)
                for value in body["stats"]["guard_by_query_hex"]
            ],
            dtype=np.float64,
        )
        if (
            guard_values.shape != values.shape
            or not np.all(np.isfinite(guard_values))
            or np.any(guard_values < 0.0)
            or not hmac.compare_digest(
                _array_digest(guard_values),
                str(body["stats"]["guard_by_query_sha256"]),
            )
        ):
            return False
        hashes = body["hashes"]
        for key, wanted in (
            ("net_sha256", expected_net_sha256),
            ("bounds_sha256", expected_bounds_sha256),
            ("query_sha256", expected_query_sha256),
            ("alpha_sha256", expected_alpha_sha256),
        ):
            if wanted is not None and not hmac.compare_digest(
                str(hashes[key]), str(wanted)
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def validate_query_dual_replay_result(
    result: QueryDualReplayResult,
    *,
    expected_net_sha256: Optional[str] = None,
    expected_bounds_sha256: Optional[str] = None,
    expected_query_sha256: Optional[str] = None,
    expected_alpha_sha256: Optional[str] = None,
) -> bool:
    """Bind a live replay result array to its independently checked receipt.

    Receipt verification alone cannot detect replacement of the in-memory
    ``lower_bounds`` array after replay.  Authority consumers must use this
    validator before intersecting a bound or exporting a property constant.
    """

    try:
        if (
            not isinstance(result, QueryDualReplayResult)
            or result.proof_authority is not True
            or not verify_query_dual_replay_receipt(
                result.receipt,
                expected_net_sha256=expected_net_sha256,
                expected_bounds_sha256=expected_bounds_sha256,
                expected_query_sha256=expected_query_sha256,
                expected_alpha_sha256=expected_alpha_sha256,
            )
        ):
            return False
        values = np.asarray(result.lower_bounds)
        if (
            values.dtype != np.float64
            or values.ndim != 1
            or not np.all(np.isfinite(values))
        ):
            return False
        receipt_values = np.asarray(
            [
                float.fromhex(value)
                for value in result.receipt["lower_bounds_hex"]
            ],
            dtype=np.float64,
        )
        return bool(
            values.shape == receipt_values.shape
            and int(result.receipt["query_count"]) == int(values.size)
            and np.array_equal(values, receipt_values)
            and hmac.compare_digest(
                _array_digest(values),
                str(result.receipt["lower_bounds_sha256"]),
            )
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


# ---------------------------------------------------------------------------
# Exact small-network oracle
# ---------------------------------------------------------------------------


class _TermBudget:
    def __init__(self, maximum: int):
        if int(maximum) <= 0:
            _fail("INVALID_ORACLE_BUDGET", "max_arithmetic_terms must be positive")
        self.maximum = int(maximum)
        self.used = 0

    def add(self, amount: int) -> None:
        self.used += int(amount)
        if self.used > self.maximum:
            _fail(
                "ORACLE_BUDGET",
                f"Fraction oracle exceeded {self.maximum} arithmetic terms",
            )


def _fractions(values: np.ndarray) -> List[Fraction]:
    return [Fraction.from_float(float(value)) for value in values.reshape(-1)]


def _fraction_conv_reverse(
    coefficient: Sequence[Fraction],
    layer: _FrozenLayer,
    budget: _TermBudget,
) -> List[Fraction]:
    p = layer.params
    weight = p["weight"]
    out_c, out_h, out_w = p["output_shape"]
    in_c, in_h, in_w = p["input_shape"]
    stride_h, stride_w = p["stride"]
    padding_h, padding_w = p["padding"]
    dilation_h, dilation_w = p["dilation"]
    groups = p["groups"]
    out_per_group = out_c // groups
    in_per_group = in_c // groups
    result = [Fraction(0) for _ in range(in_c * in_h * in_w)]
    for co in range(out_c):
        group = co // out_per_group
        ci_base = group * in_per_group
        for oh in range(out_h):
            ih0 = oh * stride_h - padding_h
            for ow in range(out_w):
                iw0 = ow * stride_w - padding_w
                c = coefficient[(co * out_h + oh) * out_w + ow]
                if not c:
                    continue
                for ci_local in range(in_per_group):
                    ci = ci_base + ci_local
                    for kh in range(weight.shape[2]):
                        ih = ih0 + kh * dilation_h
                        if ih < 0 or ih >= in_h:
                            continue
                        for kw in range(weight.shape[3]):
                            iw = iw0 + kw * dilation_w
                            if iw < 0 or iw >= in_w:
                                continue
                            budget.add(1)
                            index = (ci * in_h + ih) * in_w + iw
                            result[index] += c * Fraction.from_float(
                                float(weight[co, ci_local, kh, kw])
                            )
    return result


def _fraction_replay_one(
    prepared: _Prepared,
    query_index: int,
    budget: _TermBudget,
) -> Fraction:
    # Reuse only the frozen input snapshot and exact-audited ReLU line.  All
    # arithmetic below is independent exact rational arithmetic.
    dummy_stats = _ReplayStats()
    pending: Dict[int, List[Fraction]] = {
        prepared.output_id: _fractions(prepared.queries[query_index])
    }
    scalar = Fraction.from_float(float(prepared.query_bias[query_index]))
    reached_input = False

    def push(lid: int, values: List[Fraction]) -> None:
        current = pending.get(lid)
        if current is None:
            pending[lid] = values
            return
        if len(current) != len(values):
            _fail("SHAPE_MISMATCH", "Fraction DAG merge width")
        budget.add(len(values))
        pending[lid] = [a + b for a, b in zip(current, values)]

    for lid in prepared.reverse_order:
        prepared.deadline.check()
        coefficient = pending.pop(lid, None)
        if coefficient is None:
            continue
        layer = prepared.layers[lid]
        if layer.kind == "INPUT_SPEC":
            box = prepared.bounds[lid]
            lower, upper = _fractions(box.lb), _fractions(box.ub)
            budget.add(len(coefficient))
            scalar += sum(
                c * (lo if c >= 0 else hi)
                for c, lo, hi in zip(coefficient, lower, upper)
            )
            reached_input = True
        elif layer.kind == "INPUT":
            _fail("INVALID_GRAPH", "Fraction proof bypassed INPUT_SPEC")
        elif layer.kind == "DENSE":
            weight = layer.params["weight"]
            bias = _fractions(layer.params["bias"])
            budget.add(len(coefficient))
            scalar += sum(c * b for c, b in zip(coefficient, bias))
            new = []
            for column in range(weight.shape[1]):
                budget.add(weight.shape[0])
                new.append(
                    sum(
                        coefficient[row]
                        * Fraction.from_float(float(weight[row, column]))
                        for row in range(weight.shape[0])
                    )
                )
            push(layer.preds[0], new)
        elif layer.kind == "CONV2D":
            p = layer.params
            _, out_h, out_w = p["output_shape"]
            bias = [
                Fraction.from_float(float(p["bias_channels"][channel]))
                for channel in range(p["output_shape"][0])
                for _ in range(out_h * out_w)
            ]
            budget.add(len(coefficient))
            scalar += sum(c * b for c, b in zip(coefficient, bias))
            push(
                layer.preds[0],
                _fraction_conv_reverse(coefficient, layer, budget),
            )
        elif layer.kind == "FLATTEN":
            push(layer.preds[0], list(coefficient))
        elif layer.kind == "ADD":
            bias = _fractions(layer.params["bias"])
            budget.add(len(coefficient))
            scalar += sum(c * b for c, b in zip(coefficient, bias))
            for pred in layer.preds:
                push(pred, list(coefficient))
        elif layer.kind == "RELU":
            box = prepared.bounds[lid]
            ambiguous = (box.lb < 0.0) & (box.ub > 0.0)
            required = np.asarray(
                [
                    bool(ambiguous[index] and c < 0)
                    for index, c in enumerate(coefficient)
                ],
                dtype=bool,
            )
            slope, beta, _ = _relu_lines(
                prepared, lid, dummy_stats, required_mask=required
            )
            alpha = _alpha_block(
                prepared, lid, query_index, query_index + 1, len(coefficient)
            )[0]
            new: List[Fraction] = []
            for index, c in enumerate(coefficient):
                budget.add(1)
                if box.ub[index] <= 0.0:
                    factor = Fraction(0)
                elif box.lb[index] >= 0.0:
                    factor = Fraction(1)
                elif c >= 0:
                    factor = Fraction.from_float(float(alpha[index]))
                else:
                    factor = Fraction.from_float(float(slope[index]))
                    scalar += c * Fraction.from_float(float(beta[index]))
                new.append(c * factor)
            push(layer.preds[0], new)
        else:
            _fail("UNSUPPORTED_OPERATOR", f"Fraction oracle layer {lid}")
    if pending or not reached_input:
        _fail("INVALID_GRAPH", "Fraction oracle did not terminate at INPUT_SPEC")
    return scalar


def fraction_replay_lower_bounds(
    net: Any,
    certified_bounds: Mapping[Any, Any],
    *,
    start_lid: Optional[int] = None,
    query_rows: Optional[Any] = None,
    one_hot: Optional[Any] = None,
    query_bias: Optional[Any] = None,
    alpha_by_relu: Optional[Mapping[Any, Any]] = None,
    max_arithmetic_terms: int = 200_000,
    deadline: Optional[float] = None,
    timeout_s: Optional[float] = None,
) -> Tuple[Fraction, ...]:
    """Exact-rational reference replay for controlled toy networks.

    This is an audit oracle, not an authority-bearing production result.  It
    has an explicit arithmetic budget so it cannot accidentally be used on a
    CIFAR-sized graph.
    """

    timer = _Deadline.build(deadline, timeout_s)
    prepared = _prepare(
        net,
        certified_bounds,
        start_lid=start_lid,
        query_rows=query_rows,
        one_hot=one_hot,
        query_bias=query_bias,
        alpha_by_relu=alpha_by_relu,
        deadline=timer,
        expected_net_sha256=None,
        expected_bounds_sha256=None,
        expected_query_sha256=None,
        expected_alpha_sha256=None,
    )
    budget = _TermBudget(max_arithmetic_terms)
    results = tuple(
        _fraction_replay_one(prepared, query_index, budget)
        for query_index in range(prepared.queries.shape[0])
    )
    timer.check(force=True)
    return results


__all__ = [
    "QueryDualReplayBoundsFrame",
    "QueryDualReplayError",
    "QueryDualReplayPendingResult",
    "QueryDualReplayResult",
    "QueryDualReplaySession",
    "QueryDualReplayTimeout",
    "create_query_dual_replay_session",
    "fraction_replay_lower_bounds",
    "replay_query_lower_bounds",
    "validate_query_dual_replay_result",
    "verify_query_dual_replay_receipt",
]
