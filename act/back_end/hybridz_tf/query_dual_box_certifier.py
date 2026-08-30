#!/usr/bin/env python3
# ===- query_dual_box_certifier.py - independent outward boxes -----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Independent outward box anchor for proof-carrying query-dual replay.

This module intentionally does not consume ordinary ACT interval facts.  It
reconstructs a single-lane BOX propagation from the raw network and raw
``INPUT_SPEC`` in CPU binary64.  Affine reductions carry a Higham-style
relative error enclosure plus a minimum-subnormal allowance per operation,
after scalar, NumPy, and Torch gradual-underflow probes all pass.

Supported graph:

``INPUT -> INPUT_SPEC(BOX) -> {DENSE, CONV2D, RELU, ADD, FLATTEN}* -> ASSERT``

The public bounds mapping has the exact convention expected by
``query_dual_replay`` and ``DualSolver``:

* ``INPUT_SPEC`` and every non-ReLU computational layer: layer output box;
* ``RELU``: preactivation box (its internal post-ReLU box is propagated but
  is not exported under the ReLU id);
* ``INPUT`` and ``ASSERT``: no bounds entry.

ADD is deliberately restricted to the operator-HZ semantics used by the
large-classification graphs: exactly two same-width predecessors and no
nonzero bias.  Unsupported topology fails closed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import platform
import secrets
from pathlib import Path
import sys
import time
import weakref
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from act.back_end.core import Bounds


_SCHEMA = "act.query_dual_box_certificate.v1"
_ALGORITHM = "cpu_f64_outward_box_direct_reduction_v1"
_SUPPORTED = frozenset(
    {
        "INPUT",
        "INPUT_SPEC",
        "DENSE",
        "CONV2D",
        "RELU",
        "ADD",
        "FLATTEN",
        "ASSERT",
    }
)
_U = float(2.0**-53)
_TINY = float(np.finfo(np.float64).tiny)
_ETA = float.fromhex("0x0.0000000000001p-1022")
_SEALED_GRAPH_CAPABILITY = object()
_SEALED_GRAPH_REGISTRY: weakref.WeakValueDictionary[
    str, "_SealedFrozenGraph"
] = weakref.WeakValueDictionary()


class QueryDualBoxError(RuntimeError):
    """Fail-closed box certification error with a stable code."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


class QueryDualBoxTimeout(QueryDualBoxError):
    """The deadline expired before a complete certificate existed."""

    def __init__(self, message: str = "outward box certification deadline expired"):
        super().__init__("DEADLINE_EXPIRED", message)


@dataclass(frozen=True)
class QueryDualBoxCertificate:
    """Complete authority-bearing outward box snapshot."""

    bounds: Mapping[int, Bounds]
    semantics: Mapping[int, str]
    receipt: Mapping[str, Any]
    proof_authority: bool = True
    # V3 replay borrows this process-local object through the checked bridge
    # below.  It is deliberately absent from the public receipt and equality.
    _sealed_frozen_graph: Optional[Any] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.proof_authority:
            raise ValueError("a completed box certificate must be authoritative")


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
    raw_params_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _SealedFrozenGraph:
    """Owned immutable graph snapshot, available only through a local seal."""

    layers: Tuple[_FrozenLayer, ...]
    input_spec_id: int
    assert_id: int
    root_net_sha256: str
    content_sha256: str
    nonce: str
    capability: Any = field(repr=False, compare=False)


@dataclass
class _Deadline:
    end: Optional[float]

    @classmethod
    def build(
        cls, deadline: Optional[float], timeout_s: Optional[float]
    ) -> "_Deadline":
        now = time.monotonic()
        ends: List[float] = []
        if deadline is not None:
            value = float(deadline)
            if not math.isfinite(value):
                _fail("INVALID_DEADLINE", "deadline must be finite")
            ends.append(value)
        if timeout_s is not None:
            value = float(timeout_s)
            if not math.isfinite(value) or value < 0.0:
                _fail("INVALID_DEADLINE", "timeout_s must be finite and nonnegative")
            ends.append(now + value)
        return cls(min(ends) if ends else None)

    def check(self, stage: str) -> None:
        if self.end is not None and time.monotonic() >= self.end:
            raise QueryDualBoxTimeout(f"deadline expired during {stage}")


def _fail(code: str, message: str) -> "NoReturn":  # type: ignore[name-defined]
    raise QueryDualBoxError(code, message)


def _kind(value: Any) -> str:
    return str(getattr(value, "value", value)).upper()


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


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0<f8\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_f64(value: Any, *, name: str) -> np.ndarray:
    """Return an owned binary64 view backed by immutable ``bytes``."""

    array = _as_f64(value, name=name)
    raw = array.tobytes(order="C")
    frozen = np.frombuffer(raw, dtype=np.float64).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


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


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _source_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception as exc:
        raise QueryDualBoxError(
            "SOURCE_UNAVAILABLE", f"cannot hash certifier implementation: {exc}"
        ) from exc


def _as_f64(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise QueryDualBoxError("INVALID_NUMERIC", f"{name}: {exc}") from exc
    array = np.ascontiguousarray(array, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        _fail("NONFINITE", f"{name} must be nonempty and finite")
    return array


def _manifest_value(value: Any, *, name: str) -> Any:
    if isinstance(value, torch.Tensor):
        raw_dtype = str(value.dtype)
        array = _as_f64(value, name=name)
        return {
            "kind": "array",
            "original_dtype": raw_dtype,
            "shape": list(array.shape),
            "f64_sha256": _array_digest(array),
        }
    if isinstance(value, np.ndarray):
        raw_dtype = str(value.dtype)
        array = _as_f64(value, name=name)
        return {
            "kind": "array",
            "original_dtype": raw_dtype,
            "shape": list(array.shape),
            "f64_sha256": _array_digest(array),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not math.isfinite(scalar):
            _fail("NONFINITE", f"{name} contains a non-finite scalar")
        return {"float_hex": scalar.hex()}
    if isinstance(value, np.integer):
        return int(value)
    if hasattr(value, "value"):
        return _manifest_value(value.value, name=name)
    if isinstance(value, Mapping):
        return {
            str(key): _manifest_value(item, name=f"{name}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [
            _manifest_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _manifest_value(
                    getattr(value, item.name),
                    name=f"{name}.{item.name}",
                )
                for item in fields(value)
            },
        }
    _fail("UNSUPPORTED_PARAMETER", f"{name} has unsupported type {type(value)!r}")


def _snapshot_manifest_value(value: Any, *, name: str) -> Tuple[Any, Any]:
    """Capture one raw value once for both semantics and root manifest."""

    if isinstance(value, torch.Tensor):
        raw_dtype = str(value.dtype)
        array = _immutable_f64(value, name=name)
        return array, {
            "kind": "array",
            "original_dtype": raw_dtype,
            "shape": list(array.shape),
            "f64_sha256": _array_digest(array),
        }
    if isinstance(value, np.ndarray):
        raw_dtype = str(value.dtype)
        array = _immutable_f64(value, name=name)
        return array, {
            "kind": "array",
            "original_dtype": raw_dtype,
            "shape": list(array.shape),
            "f64_sha256": _array_digest(array),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value, value
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not math.isfinite(scalar):
            _fail("NONFINITE", f"{name} contains a non-finite scalar")
        return scalar, {"float_hex": scalar.hex()}
    if isinstance(value, np.integer):
        scalar = int(value)
        return scalar, scalar
    if hasattr(value, "value"):
        return _snapshot_manifest_value(value.value, name=name)
    if isinstance(value, Mapping):
        owned: Dict[Any, Any] = {}
        manifest: Dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            manifest_key = str(key)
            if manifest_key in manifest:
                _fail(
                    "INVALID_GRAPH",
                    f"{name} has keys that collide after string canonicalization",
                )
            child, child_manifest = _snapshot_manifest_value(
                item, name=f"{name}.{key}"
            )
            owned[key] = child
            manifest[manifest_key] = child_manifest
        return MappingProxyType(owned), manifest
    if isinstance(value, (tuple, list)):
        owned_items = []
        manifest_items = []
        for index, item in enumerate(value):
            child, child_manifest = _snapshot_manifest_value(
                item, name=f"{name}[{index}]"
            )
            owned_items.append(child)
            manifest_items.append(child_manifest)
        return tuple(owned_items), manifest_items
    if is_dataclass(value) and not isinstance(value, type):
        owned_fields: Dict[str, Any] = {}
        manifest_fields: Dict[str, Any] = {}
        for item in fields(value):
            child, child_manifest = _snapshot_manifest_value(
                getattr(value, item.name),
                name=f"{name}.{item.name}",
            )
            owned_fields[item.name] = child
            manifest_fields[item.name] = child_manifest
        return MappingProxyType(owned_fields), {
            "kind": "dataclass",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": manifest_fields,
        }
    _fail("UNSUPPORTED_PARAMETER", f"{name} has unsupported type {type(value)!r}")


def _snapshot_raw_params(
    layer: Any,
) -> Tuple[Mapping[Any, Any], Mapping[str, Any]]:
    params = getattr(layer, "params", {}) or {}
    if not isinstance(params, Mapping):
        _fail("INVALID_GRAPH", f"layer {layer.id} params are not a mapping")
    owned: Dict[Any, Any] = {}
    manifest: Dict[str, Any] = {}
    for key, value in sorted(params.items(), key=lambda pair: str(pair[0])):
        manifest_key = str(key)
        if manifest_key in manifest:
            _fail(
                "INVALID_GRAPH",
                f"layer {layer.id} params keys collide after canonicalization",
            )
        captured, captured_manifest = _snapshot_manifest_value(
            value, name=f"layer[{layer.id}].params[{key!r}]"
        )
        owned[key] = captured
        manifest[manifest_key] = captured_manifest
    return MappingProxyType(owned), MappingProxyType(manifest)


def _pair(value: Any, *, name: str, allow_zero: bool) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            _fail("INVALID_CONV", f"{name} must have length two")
        result = (int(value[0]), int(value[1]))
    else:
        result = (int(value), int(value))
    floor = 0 if allow_zero else 1
    if result[0] < floor or result[1] < floor:
        _fail("INVALID_CONV", f"{name} entries must be >= {floor}")
    return result


def _shape3(value: Any, *, name: str) -> Tuple[int, int, int]:
    if value is None:
        _fail("INVALID_CONV", f"{name} is required")
    shape = tuple(int(item) for item in value)
    if len(shape) == 4:
        if shape[0] != 1:
            _fail("BATCH_UNSUPPORTED", f"{name} metadata batch must be one")
        shape = shape[1:]
    if len(shape) != 3 or any(item <= 0 for item in shape):
        _fail("INVALID_CONV", f"{name} must be CHW or 1xCHW")
    return shape


def _bias(value: Any, width: int, *, name: str) -> np.ndarray:
    if value is None:
        return _immutable_f64(np.zeros(width, dtype=np.float64), name=name)
    raw = _as_f64(value, name=name).reshape(-1)
    if raw.size == 1:
        return _immutable_f64(
            np.full(width, float(raw[0]), dtype=np.float64), name=name
        )
    if raw.size != width:
        _fail("SHAPE_MISMATCH", f"{name} has {raw.size} entries, expected {width}")
    return _immutable_f64(raw, name=name)


def _replay_manifest_scalar(value: Any) -> Any:
    """Freeze the scalar/shape view used by replay's V1 layer manifest."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("NONFINITE", "network scalar parameter is non-finite")
        return float(value)
    if isinstance(value, np.generic):
        return _replay_manifest_scalar(value.item())
    if isinstance(value, (tuple, list)):
        return tuple(_replay_manifest_scalar(item) for item in value)
    return str(value)


def _freeze_layer(layer: Any, preds: Tuple[int, ...]) -> _FrozenLayer:
    lid = int(layer.id)
    kind = _kind(layer.kind)
    if kind not in _SUPPORTED:
        _fail("UNSUPPORTED_OPERATOR", f"layer {lid} kind {kind}")
    in_vars = tuple(getattr(layer, "in_vars", ()) or ())
    out_vars = tuple(getattr(layer, "out_vars", ()) or ())
    width = len(out_vars)
    raw, raw_params_manifest = _snapshot_raw_params(layer)
    params: Dict[str, Any] = {}

    if kind == "INPUT":
        if preds:
            _fail("INVALID_GRAPH", f"INPUT layer {lid} must be a source")
        params["shape"] = tuple(int(value) for value in raw.get("shape", ()))
        params["dtype"] = str(raw.get("dtype", ""))
    elif kind == "INPUT_SPEC":
        if len(preds) != 1:
            _fail("INVALID_GRAPH", f"INPUT_SPEC layer {lid} needs one predecessor")
        if _kind(raw.get("kind")) != "BOX":
            _fail("UNSUPPORTED_INPUT_SPEC", f"INPUT_SPEC layer {lid} is not BOX")
        lb = _as_f64(raw.get("lb"), name=f"INPUT_SPEC[{lid}].lb")
        ub = _as_f64(raw.get("ub"), name=f"INPUT_SPEC[{lid}].ub")
        if lb.ndim < 2 or ub.shape != lb.shape or lb.shape[0] != 1:
            _fail("BATCH_UNSUPPORTED", "V1 requires one batched BOX lane")
        lb, ub = lb.reshape(1, -1), ub.reshape(1, -1)
        if np.any(lb > ub):
            _fail("INVALID_BOUNDS", f"INPUT_SPEC layer {lid} has lb > ub")
        if width == 0:
            width = int(lb.shape[1])
        if width != lb.shape[1]:
            _fail("SHAPE_MISMATCH", f"INPUT_SPEC layer {lid} width mismatch")
        params.update(
            {
                "lb": lb,
                "ub": ub,
                "kind": "BOX",
            }
        )
    elif kind == "DENSE":
        if len(preds) != 1:
            _fail("INVALID_GRAPH", f"DENSE layer {lid} needs one predecessor")
        weight = _as_f64(raw.get("weight"), name=f"DENSE[{lid}].weight")
        if weight.ndim != 2:
            _fail("SHAPE_MISMATCH", f"DENSE layer {lid} weight must be rank two")
        if width == 0:
            width = int(weight.shape[0])
        if width != weight.shape[0]:
            _fail("SHAPE_MISMATCH", f"DENSE layer {lid} output width mismatch")
        params["weight"] = weight
        params["bias"] = _bias(raw.get("bias"), width, name=f"DENSE[{lid}].bias")
    elif kind == "CONV2D":
        if len(preds) != 1:
            _fail("INVALID_GRAPH", f"CONV2D layer {lid} needs one predecessor")
        weight = _as_f64(raw.get("weight"), name=f"CONV2D[{lid}].weight")
        if weight.ndim != 4:
            _fail("SHAPE_MISMATCH", f"CONV2D layer {lid} weight must be rank four")
        input_shape = _shape3(raw.get("input_shape"), name=f"CONV2D[{lid}].input_shape")
        output_shape = _shape3(
            raw.get("output_shape"), name=f"CONV2D[{lid}].output_shape"
        )
        groups = int(raw.get("groups", 1))
        if groups <= 0 or weight.shape[0] % groups:
            _fail("INVALID_CONV", f"CONV2D layer {lid} invalid groups")
        if input_shape[0] != weight.shape[1] * groups:
            _fail("INVALID_CONV", f"CONV2D layer {lid} input channel mismatch")
        if output_shape[0] != weight.shape[0]:
            _fail("INVALID_CONV", f"CONV2D layer {lid} output channel mismatch")
        stride = _pair(raw.get("stride", 1), name="stride", allow_zero=False)
        padding = _pair(raw.get("padding", 0), name="padding", allow_zero=True)
        dilation = _pair(raw.get("dilation", 1), name="dilation", allow_zero=False)
        if bool(raw.get("transposed", False)):
            _fail("UNSUPPORTED_OPERATOR", f"CONV2D layer {lid} is transposed")
        if str(raw.get("padding_mode", "zeros")).lower() not in {"zero", "zeros"}:
            _fail("UNSUPPORTED_OPERATOR", f"CONV2D layer {lid} padding mode")
        kh, kw = weight.shape[-2:]
        expected_h = (
            input_shape[1] + 2 * padding[0] - dilation[0] * (kh - 1) - 1
        ) // stride[0] + 1
        expected_w = (
            input_shape[2] + 2 * padding[1] - dilation[1] * (kw - 1) - 1
        ) // stride[1] + 1
        if output_shape[1:] != (expected_h, expected_w):
            _fail("INVALID_CONV", f"CONV2D layer {lid} declared output shape mismatch")
        if width == 0:
            width = int(np.prod(output_shape))
        if width != int(np.prod(output_shape)):
            _fail("SHAPE_MISMATCH", f"CONV2D layer {lid} output width mismatch")
        params.update(
            {
                "weight": weight,
                "bias": _bias(
                    raw.get("bias"), output_shape[0], name=f"CONV2D[{lid}].bias"
                ),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }
        )
    elif kind == "RELU":
        if len(preds) != 1:
            _fail("INVALID_GRAPH", f"RELU layer {lid} needs one predecessor")
    elif kind == "FLATTEN":
        if len(preds) != 1:
            _fail("INVALID_GRAPH", f"FLATTEN layer {lid} needs one predecessor")
        params["start_dim"] = int(raw.get("start_dim", 1))
        params["end_dim"] = int(raw.get("end_dim", -1))
        params["input_shape"] = _replay_manifest_scalar(raw.get("input_shape"))
        params["output_shape"] = _replay_manifest_scalar(raw.get("output_shape"))
    elif kind == "ADD":
        if len(preds) != 2:
            _fail("UNSUPPORTED_ADD", f"ADD layer {lid} must have exactly two predecessors")
        bias = raw.get("bias")
        if bias is not None:
            bias_value = _as_f64(bias, name=f"ADD[{lid}].bias")
            if np.any(bias_value != 0.0):
                _fail("UNSUPPORTED_ADD", f"ADD layer {lid} has nonzero bias")
        params["bias"] = _bias(bias, width, name=f"ADD[{lid}].bias")
    elif kind == "ASSERT":
        if len(preds) != 1:
            _fail("INVALID_GRAPH", f"ASSERT layer {lid} needs one predecessor")
        params["kind"] = str(raw.get("kind", ""))

    if kind not in {"INPUT", "ASSERT"} and width <= 0:
        _fail("SHAPE_MISMATCH", f"layer {lid} has no output variables")
    return _FrozenLayer(
        id=lid,
        kind=kind,
        preds=preds,
        width=width,
        in_vars=in_vars,
        out_vars=out_vars,
        params=_deep_freeze(params),
        raw_params_manifest=_deep_freeze(raw_params_manifest),
    )


def _freeze_graph(net: Any) -> Tuple[Tuple[_FrozenLayer, ...], int, int]:
    raw_layers = list(getattr(net, "layers", ()) or ())
    if not raw_layers:
        _fail("INVALID_GRAPH", "network has no layers")
    by_id = {int(layer.id): layer for layer in raw_layers}
    if len(by_id) != len(raw_layers):
        _fail("INVALID_GRAPH", "layer ids are not unique")
    positions = {int(layer.id): index for index, layer in enumerate(raw_layers)}
    raw_preds = getattr(net, "preds", {}) or {}
    preds: Dict[int, Tuple[int, ...]] = {}
    successors: Dict[int, List[int]] = {lid: [] for lid in by_id}
    indegree: Dict[int, int] = {}
    for lid in by_id:
        parents = tuple(int(value) for value in (raw_preds.get(lid, ()) or ()))
        if len(set(parents)) != len(parents):
            _fail("INVALID_GRAPH", f"layer {lid} repeats a predecessor")
        if any(parent not in by_id for parent in parents):
            _fail("INVALID_GRAPH", f"layer {lid} references an unknown predecessor")
        preds[lid] = parents
        indegree[lid] = len(parents)
        for parent in parents:
            successors[parent].append(lid)

    ready = sorted(
        (positions[lid], lid) for lid, degree in indegree.items() if degree == 0
    )
    order: List[int] = []
    while ready:
        _, lid = ready.pop(0)
        order.append(lid)
        for successor in sorted(successors[lid], key=positions.__getitem__):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append((positions[successor], successor))
                ready.sort()
    if len(order) != len(raw_layers):
        _fail("INVALID_GRAPH", "network is cyclic")

    assertions = [lid for lid in order if _kind(by_id[lid].kind) == "ASSERT"]
    if len(assertions) != 1 or assertions[0] != order[-1]:
        _fail("INVALID_GRAPH", "V1 requires one terminal ASSERT")
    assert_id = assertions[0]
    if len(preds[assert_id]) != 1:
        _fail("INVALID_GRAPH", "ASSERT must have one predecessor")

    reachable: set[int] = set()
    stack = [assert_id]
    while stack:
        lid = stack.pop()
        if lid in reachable:
            continue
        reachable.add(lid)
        stack.extend(preds[lid])
    if reachable != set(order):
        _fail("INVALID_GRAPH", "V1 rejects layers outside the ASSERT proof cone")

    frozen = tuple(_freeze_layer(by_id[lid], preds[lid]) for lid in order)
    specs = [layer.id for layer in frozen if layer.kind == "INPUT_SPEC"]
    inputs = [layer.id for layer in frozen if layer.kind == "INPUT"]
    if len(specs) != 1 or len(inputs) != 1:
        _fail("INVALID_GRAPH", "V1 requires exactly one INPUT and one INPUT_SPEC")
    spec = next(layer for layer in frozen if layer.id == specs[0])
    if spec.preds != (inputs[0],):
        _fail("INVALID_GRAPH", "INPUT_SPEC must immediately follow INPUT")
    return frozen, specs[0], assert_id


def _network_manifest(layers: Sequence[_FrozenLayer]) -> List[Mapping[str, Any]]:
    return [
        {
            "id": layer.id,
            "kind": layer.kind,
            "preds": list(layer.preds),
            "width": layer.width,
            "in_vars": list(layer.in_vars),
            "out_vars": list(layer.out_vars),
            "raw_params": _deep_thaw(layer.raw_params_manifest),
        }
        for layer in layers
    ]


def query_dual_network_sha256(net: Any) -> str:
    """Hash the exact live graph representation consumed by query-dual.

    This uses the same freeze/manifest path as
    :func:`certify_query_dual_boxes`, without interval propagation.  A
    process-local bound authority can therefore bind itself to the exact
    network later frozen by the independent root certifier.
    """

    layers, _input_spec_id, _assert_id = _freeze_graph(net)
    return _json_digest(_network_manifest(layers))


def _sealed_semantic_manifest(
    layers: Sequence[_FrozenLayer],
) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    for layer in layers:
        params: Dict[str, Any] = {}
        for key, value in sorted(layer.params.items()):
            if isinstance(value, np.ndarray):
                params[key] = {
                    "shape": list(value.shape),
                    "sha256": _array_digest(value),
                }
            else:
                params[key] = _deep_thaw(value)
        records.append(
            {
                "id": layer.id,
                "kind": layer.kind,
                "preds": list(layer.preds),
                "width": layer.width,
                "in_vars": [_manifest_value(value, name="in_var") for value in layer.in_vars],
                "out_vars": [
                    _manifest_value(value, name="out_var") for value in layer.out_vars
                ],
                "params": params,
            }
        )
    return records


def _sealed_graph_digest(
    layers: Sequence[_FrozenLayer],
    *,
    root_net_sha256: str,
    input_spec_id: int,
    assert_id: int,
) -> str:
    return _json_digest(
        {
            "root_net_sha256": str(root_net_sha256),
            "input_spec_id": int(input_spec_id),
            "assert_id": int(assert_id),
            "semantic_layers": _sealed_semantic_manifest(layers),
        }
    )


def _seal_frozen_graph(
    layers: Tuple[_FrozenLayer, ...],
    *,
    root_net_sha256: str,
    input_spec_id: int,
    assert_id: int,
) -> _SealedFrozenGraph:
    nonce = secrets.token_hex(32)
    graph = _SealedFrozenGraph(
        layers=layers,
        input_spec_id=int(input_spec_id),
        assert_id=int(assert_id),
        root_net_sha256=str(root_net_sha256),
        content_sha256=_sealed_graph_digest(
            layers,
            root_net_sha256=root_net_sha256,
            input_spec_id=input_spec_id,
            assert_id=assert_id,
        ),
        nonce=nonce,
        capability=_SEALED_GRAPH_CAPABILITY,
    )
    _SEALED_GRAPH_REGISTRY[nonce] = graph
    return graph


def _borrow_sealed_query_dual_graph(
    certificate: QueryDualBoxCertificate,
    *,
    validate_content: bool = True,
) -> _SealedFrozenGraph:
    """Checked process-local bridge used by the V3 replay session."""

    if not isinstance(certificate, QueryDualBoxCertificate):
        _fail("INVALID_CERTIFICATE", "root certificate has the wrong type")
    graph = certificate._sealed_frozen_graph
    if (
        not isinstance(graph, _SealedFrozenGraph)
        or graph.capability is not _SEALED_GRAPH_CAPABILITY
        or _SEALED_GRAPH_REGISTRY.get(graph.nonce) is not graph
    ):
        _fail("INVALID_CERTIFICATE", "root certificate has no local frozen-graph seal")
    try:
        receipt_net_sha = str(certificate.receipt["hashes"]["net_sha256"])
    except (KeyError, TypeError) as exc:
        raise QueryDualBoxError(
            "INVALID_CERTIFICATE", "root certificate receipt is malformed"
        ) from exc
    if not hmac.compare_digest(graph.root_net_sha256, receipt_net_sha):
        _fail("INVALID_CERTIFICATE", "root frozen graph/receipt mismatch")
    if validate_content:
        actual = _sealed_graph_digest(
            graph.layers,
            root_net_sha256=graph.root_net_sha256,
            input_spec_id=graph.input_spec_id,
            assert_id=graph.assert_id,
        )
        if not hmac.compare_digest(actual, graph.content_sha256):
            _fail("INVALID_CERTIFICATE", "root frozen graph seal was modified")
    return graph


def _check_numeric_platform() -> Mapping[str, Any]:
    smallest = np.float64(_ETA)
    half_tiny = np.float64(_TINY) * np.float64(0.5)
    eta_product = np.float64(smallest * np.float64(1.0))
    gradual = bool(
        smallest != 0.0 and half_tiny != 0.0 and eta_product == smallest
    )
    f64 = np.finfo(np.float64)
    wide = np.finfo(np.longdouble)
    wide_longdouble = bool(
        wide.nmant > f64.nmant and wide.eps < f64.eps and wide.max > f64.max
    )
    probe_width = 4096
    probe_left = np.zeros((1, probe_width), dtype=np.float64)
    probe_right = np.zeros((probe_width, 1), dtype=np.float64)
    probe_left[0, 0] = smallest
    probe_right[0, 0] = 1.0
    numpy_dot = float((probe_left @ probe_right)[0, 0])
    torch_dot = float(
        torch.matmul(
            torch.from_numpy(probe_left),
            torch.from_numpy(probe_right),
        )[0, 0].item()
    )
    torch_eta = torch.tensor([_ETA], dtype=torch.float64, device="cpu")
    torch_clamp = float(torch_eta.clamp(min=0.0)[0].item())
    torch_abs = float(torch_eta.neg().abs()[0].item())
    numpy_gradual = bool(numpy_dot == float(smallest))
    torch_gradual = bool(torch_dot == float(smallest))
    torch_pointwise_gradual = bool(
        torch_clamp == float(smallest) and torch_abs == float(smallest)
    )
    half_ulp = np.float64(2.0**-53)
    above_half_ulp = np.nextafter(half_ulp, np.float64(math.inf))
    nearest_even = bool(
        np.float64(1.0) + half_ulp == np.float64(1.0)
        and np.float64(1.0) + above_half_ulp != np.float64(1.0)
    )
    if (
        not gradual
        or not numpy_gradual
        or not torch_gradual
        or not torch_pointwise_gradual
        or not nearest_even
        or not wide_longdouble
    ):
        _fail(
            "NUMERIC_PLATFORM",
            "requires round-to-nearest-even, gradual float64 underflow in "
            "scalar/NumPy/Torch reductions, and wider longdouble",
        )
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "byteorder": sys.byteorder,
        "float64_gradual_underflow": gradual,
        "numpy_matmul_gradual_underflow_probe": numpy_gradual,
        "torch_matmul_gradual_underflow_probe": torch_gradual,
        "torch_clamp_abs_gradual_underflow_probe": torch_pointwise_gradual,
        "round_to_nearest_even": nearest_even,
        "longdouble_nmant": int(wide.nmant),
        "float64_nmant": int(f64.nmant),
        "wide_longdouble": wide_longdouble,
        "probe_reduction_width": probe_width,
        "underflow_guard_unit": "float64_min_subnormal",
    }


def _next_up(value: np.ndarray) -> np.ndarray:
    result = np.nextafter(np.asarray(value, dtype=np.float64), math.inf)
    if not np.all(np.isfinite(result)):
        _fail("NUMERIC_OVERFLOW", "outward upper operation overflowed")
    return result


def _up_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _next_up(np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64))


def _up_mul(a: np.ndarray, b: Any) -> np.ndarray:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    result = left * right
    active = (left != 0.0) & (right != 0.0)
    widened = result.copy()
    widened[active] = np.nextafter(widened[active], math.inf)
    if not np.all(np.isfinite(widened)):
        _fail("NUMERIC_OVERFLOW", "outward product overflowed")
    return widened


def _up_div(a: np.ndarray, b: float) -> np.ndarray:
    if not math.isfinite(b) or b <= 0.0:
        _fail("NUMERIC_GUARD", "invalid outward divisor")
    return _next_up(np.asarray(a, dtype=np.float64) / np.float64(b))


def _gamma(operations: int) -> float:
    count = max(1, int(operations))
    product = count * _U
    if product >= 0.25:
        _fail("NUMERIC_GUARD", "reduction is too long for a finite error model")
    return float(
        np.nextafter(np.float64(product / (1.0 - product)), np.float64(math.inf))
    )


def _underflow(operations: int, gamma: float) -> float:
    # All arithmetic paths have passed gradual-underflow probes, so the
    # standard per-operation minimum-subnormal allowance applies.
    raw = np.longdouble(max(1, int(operations))) * np.longdouble(_ETA)
    raw /= np.longdouble(1.0) - np.longdouble(gamma)
    value = float(raw)
    if np.longdouble(value) < raw:
        value = float(np.nextafter(np.float64(value), np.float64(math.inf)))
    if not math.isfinite(value):
        _fail("NUMERIC_GUARD", "underflow allowance overflowed")
    return value


def _reduction_radius(
    nominal_mass: np.ndarray, *, operations: int
) -> np.ndarray:
    mass = np.asarray(nominal_mass, dtype=np.float64)
    if np.any(mass < 0.0) or not np.all(np.isfinite(mass)):
        _fail("NUMERIC_GUARD", "invalid absolute reduction mass")
    gamma = _gamma(operations)
    under = _underflow(operations, gamma)
    mass_upper = _up_div(
        _up_add(mass, np.full_like(mass, under)), 1.0 - gamma
    )
    return _up_add(
        _up_mul(mass_upper, gamma), np.full_like(mass_upper, under)
    )


def _finish_enclosure(
    lower_nominal: np.ndarray,
    upper_nominal: np.ndarray,
    lower_mass: np.ndarray,
    upper_mass: np.ndarray,
    *,
    operations: int,
    where: str,
) -> _Box:
    for value in (lower_nominal, upper_nominal, lower_mass, upper_mass):
        if not np.all(np.isfinite(value)):
            _fail("NUMERIC_OVERFLOW", f"{where} produced non-finite arithmetic")
    lower_radius = _reduction_radius(lower_mass, operations=operations)
    upper_radius = _reduction_radius(upper_mass, operations=operations)
    lb = np.nextafter(
        np.asarray(lower_nominal, dtype=np.float64) - lower_radius, -math.inf
    )
    ub = np.nextafter(
        np.asarray(upper_nominal, dtype=np.float64) + upper_radius, math.inf
    )
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        _fail("NUMERIC_OVERFLOW", f"{where} outward endpoints overflowed")
    if np.any(lb > ub):
        _fail("INVALID_BOUNDS", f"{where} produced lb > ub")
    return _Box(np.ascontiguousarray(lb), np.ascontiguousarray(ub))


def _dense_box(layer: _FrozenLayer, source: _Box) -> _Box:
    weight = layer.params["weight"]
    bias = layer.params["bias"].reshape(1, -1)
    if source.lb.ndim != 2 or source.lb.shape[1] != weight.shape[1]:
        _fail("SHAPE_MISMATCH", f"DENSE layer {layer.id} input width")
    positive = np.maximum(weight, 0.0)
    negative = np.minimum(weight, 0.0)
    abs_negative = -negative
    lower_nominal = source.lb @ positive.T
    lower_nominal = lower_nominal + source.ub @ negative.T
    lower_nominal = lower_nominal + bias
    upper_nominal = source.ub @ positive.T
    upper_nominal = upper_nominal + source.lb @ negative.T
    upper_nominal = upper_nominal + bias
    lower_mass = np.abs(source.lb) @ positive.T
    lower_mass = lower_mass + np.abs(source.ub) @ abs_negative.T
    lower_mass = lower_mass + np.abs(bias)
    upper_mass = np.abs(source.ub) @ positive.T
    upper_mass = upper_mass + np.abs(source.lb) @ abs_negative.T
    upper_mass = upper_mass + np.abs(bias)
    return _finish_enclosure(
        lower_nominal,
        upper_nominal,
        lower_mass,
        upper_mass,
        operations=4 * int(weight.shape[1]) + 8,
        where=f"DENSE[{layer.id}]",
    )


def _conv_box(
    layer: _FrozenLayer,
    source: _Box,
    *,
    channel_chunk: int,
    deadline: _Deadline,
) -> _Box:
    params = layer.params
    weight = params["weight"]
    in_c, in_h, in_w = params["input_shape"]
    out_c, out_h, out_w = params["output_shape"]
    if source.lb.shape != (1, in_c * in_h * in_w):
        _fail("SHAPE_MISMATCH", f"CONV2D layer {layer.id} input shape")
    lower_input = torch.from_numpy(source.lb.reshape(1, in_c, in_h, in_w))
    upper_input = torch.from_numpy(source.ub.reshape(1, in_c, in_h, in_w))
    unfold_kwargs = {
        "kernel_size": tuple(int(value) for value in weight.shape[-2:]),
        "dilation": params["dilation"],
        "padding": params["padding"],
        "stride": params["stride"],
    }
    deadline.check(f"CONV2D[{layer.id}].unfold_before")
    lower_patches = F.unfold(lower_input, **unfold_kwargs)
    upper_patches = F.unfold(upper_input, **unfold_kwargs)
    deadline.check(f"CONV2D[{layer.id}].unfold_after")
    spatial = out_h * out_w
    if lower_patches.shape[-1] != spatial:
        _fail("SHAPE_MISMATCH", f"CONV2D layer {layer.id} unfold shape")
    groups = int(params["groups"])
    out_per_group = out_c // groups
    terms = int(weight.shape[1] * weight.shape[2] * weight.shape[3])
    lower_nominal = np.empty((1, out_c, spatial), dtype=np.float64)
    upper_nominal = np.empty_like(lower_nominal)
    bias = params["bias"]
    for group in range(groups):
        patch_start = group * terms
        patch_end = patch_start + terms
        lp = lower_patches[:, patch_start:patch_end, :]
        up = upper_patches[:, patch_start:patch_end, :]
        alp, aup = lp.abs(), up.abs()
        channel_start = group * out_per_group
        channel_end = channel_start + out_per_group
        for start in range(channel_start, channel_end, channel_chunk):
            deadline.check(f"CONV2D[{layer.id}].chunk_before")
            end = min(channel_end, start + channel_chunk)
            local_weight = torch.from_numpy(
                np.ascontiguousarray(
                    weight[start:end].reshape(end - start, terms)
                ).copy()
            )
            positive = local_weight.clamp(min=0.0)
            negative = local_weight.clamp(max=0.0)
            abs_negative = -negative
            lo = torch.matmul(positive, lp) + torch.matmul(negative, up)
            hi = torch.matmul(positive, up) + torch.matmul(negative, lp)
            lo_mass = torch.matmul(positive, alp) + torch.matmul(abs_negative, aup)
            hi_mass = torch.matmul(positive, aup) + torch.matmul(abs_negative, alp)
            local_bias = torch.from_numpy(bias[start:end].copy()).reshape(1, -1, 1)
            lo = lo + local_bias
            hi = hi + local_bias
            lo_mass = lo_mass + local_bias.abs()
            hi_mass = hi_mass + local_bias.abs()
            # Each channel chunk gets its own complete dot/accumulation
            # enclosure.  No later whole-layer guard is trusted to repair a
            # native reduction performed in this chunk.
            chunk_box = _finish_enclosure(
                lo.numpy(),
                hi.numpy(),
                lo_mass.numpy(),
                hi_mass.numpy(),
                operations=4 * terms + 8,
                where=f"CONV2D[{layer.id}].chunk[{start}:{end}]",
            )
            lower_nominal[:, start:end, :] = chunk_box.lb
            upper_nominal[:, start:end, :] = chunk_box.ub
            deadline.check(f"CONV2D[{layer.id}].chunk_after")
    return _Box(
        lower_nominal.reshape(1, -1),
        upper_nominal.reshape(1, -1),
    )


def _add_box(layer: _FrozenLayer, left: _Box, right: _Box) -> _Box:
    if left.lb.shape != right.lb.shape or left.lb.shape != (1, layer.width):
        _fail("SHAPE_MISMATCH", f"ADD layer {layer.id} requires equal widths")
    lower_nominal = left.lb + right.lb
    upper_nominal = left.ub + right.ub
    lower_mass = np.abs(left.lb) + np.abs(right.lb)
    upper_mass = np.abs(left.ub) + np.abs(right.ub)
    return _finish_enclosure(
        lower_nominal,
        upper_nominal,
        lower_mass,
        upper_mass,
        operations=6,
        where=f"ADD[{layer.id}]",
    )


def _box_record(lid: int, kind: str, semantics: str, box: _Box) -> Mapping[str, Any]:
    return {
        "id": int(lid),
        "kind": str(kind),
        "semantics": str(semantics),
        "shape": list(box.lb.shape),
        "lb_sha256": _array_digest(box.lb),
        "ub_sha256": _array_digest(box.ub),
    }


def _live_bounds_records(
    bounds: Mapping[int, Bounds],
    semantics: Mapping[int, str],
    kinds: Mapping[int, str],
) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    for lid in sorted(bounds):
        value = bounds[lid]
        lb = _as_f64(value.lb, name=f"certificate.bounds[{lid}].lb")
        ub = _as_f64(value.ub, name=f"certificate.bounds[{lid}].ub")
        if lb.shape != ub.shape or lb.shape[0] != 1 or np.any(lb > ub):
            _fail("INVALID_CERTIFICATE", f"certificate bounds[{lid}] are invalid")
        records.append(_box_record(lid, kinds[lid], semantics[lid], _Box(lb, ub)))
    return records


def certify_query_dual_boxes(
    net: Any,
    *,
    deadline: Optional[float] = None,
    timeout_s: Optional[float] = None,
    conv_channel_chunk: int = 32,
    expected_net_sha256: Optional[str] = None,
    expected_input_sha256: Optional[str] = None,
    expected_implementation_sha256: Optional[str] = None,
) -> QueryDualBoxCertificate:
    """Produce a complete independent outward box anchor.

    V1 accepts exactly one BOX lane.  It raises on every unsupported,
    incomplete, non-finite, platform, hash, or deadline condition; there is no
    authority-bearing fallback.
    """

    started = time.monotonic()
    if (
        isinstance(conv_channel_chunk, bool)
        or not isinstance(conv_channel_chunk, int)
        or conv_channel_chunk <= 0
    ):
        _fail("INVALID_CHUNK", "conv_channel_chunk must be a positive integer")
    timer = _Deadline.build(deadline, timeout_s)
    timer.check("entry")
    platform = _check_numeric_platform()
    timer.check("platform")
    layers, input_spec_id, assert_id = _freeze_graph(net)
    source_sha = _source_sha256()
    net_manifest = _network_manifest(layers)
    net_sha = _json_digest(net_manifest)
    spec = next(layer for layer in layers if layer.id == input_spec_id)
    input_manifest = {
        "layer_id": input_spec_id,
        "kind": "BOX",
        "lb_sha256": _array_digest(spec.params["lb"]),
        "ub_sha256": _array_digest(spec.params["ub"]),
        "shape": list(spec.params["lb"].shape),
    }
    input_sha = _json_digest(input_manifest)
    for label, actual, expected in (
        ("net_sha256", net_sha, expected_net_sha256),
        ("input_sha256", input_sha, expected_input_sha256),
        ("implementation_sha256", source_sha, expected_implementation_sha256),
    ):
        if expected is not None and not hmac.compare_digest(str(actual), str(expected)):
            _fail("HASH_MISMATCH", f"{label}: expected {expected}, got {actual}")

    outputs: Dict[int, _Box] = {}
    public: Dict[int, _Box] = {}
    semantics: Dict[int, str] = {}
    layer_records: List[Mapping[str, Any]] = []
    kind_by_id = {layer.id: layer.kind for layer in layers}
    for layer in layers:
        timer.check(f"layer[{layer.id}].before")
        kind = layer.kind
        if kind == "INPUT":
            layer_records.append(
                {"id": layer.id, "kind": kind, "semantics": "domain_placeholder"}
            )
            continue
        if kind == "INPUT_SPEC":
            box = _Box(layer.params["lb"].copy(), layer.params["ub"].copy())
            outputs[layer.id] = box
            public[layer.id] = box
            semantics[layer.id] = "output"
        elif kind == "DENSE":
            source = outputs.get(layer.preds[0])
            if source is None:
                _fail("INVALID_GRAPH", f"DENSE layer {layer.id} predecessor unavailable")
            box = _dense_box(layer, source)
            outputs[layer.id] = box
            public[layer.id] = box
            semantics[layer.id] = "output"
        elif kind == "CONV2D":
            source = outputs.get(layer.preds[0])
            if source is None:
                _fail("INVALID_GRAPH", f"CONV2D layer {layer.id} predecessor unavailable")
            box = _conv_box(
                layer,
                source,
                channel_chunk=conv_channel_chunk,
                deadline=timer,
            )
            outputs[layer.id] = box
            public[layer.id] = box
            semantics[layer.id] = "output"
        elif kind == "ADD":
            left = outputs.get(layer.preds[0])
            right = outputs.get(layer.preds[1])
            if left is None or right is None:
                _fail("INVALID_GRAPH", f"ADD layer {layer.id} predecessor unavailable")
            box = _add_box(layer, left, right)
            outputs[layer.id] = box
            public[layer.id] = box
            semantics[layer.id] = "output"
        elif kind == "FLATTEN":
            source = outputs.get(layer.preds[0])
            if source is None or source.lb.size != layer.width:
                _fail("SHAPE_MISMATCH", f"FLATTEN layer {layer.id} changes size")
            box = _Box(source.lb.copy(), source.ub.copy())
            outputs[layer.id] = box
            public[layer.id] = box
            semantics[layer.id] = "output"
        elif kind == "RELU":
            source = outputs.get(layer.preds[0])
            if source is None or source.lb.shape != (1, layer.width):
                _fail("SHAPE_MISMATCH", f"RELU layer {layer.id} input width")
            pre = _Box(source.lb.copy(), source.ub.copy())
            public[layer.id] = pre
            semantics[layer.id] = "preactivation"
            outputs[layer.id] = _Box(
                np.maximum(pre.lb, 0.0), np.maximum(pre.ub, 0.0)
            )
        elif kind == "ASSERT":
            if outputs.get(layer.preds[0]) is None:
                _fail("INVALID_GRAPH", "ASSERT predecessor unavailable")
            layer_records.append(
                {"id": layer.id, "kind": kind, "semantics": "validation_terminal"}
            )
            timer.check(f"layer[{layer.id}].after")
            continue
        else:  # pragma: no cover - frozen graph already rejects this.
            _fail("UNSUPPORTED_OPERATOR", f"layer {layer.id} kind {kind}")
        layer_records.append(
            _box_record(layer.id, kind, semantics[layer.id], public[layer.id])
        )
        timer.check(f"layer[{layer.id}].after")

    expected_keys = {
        layer.id for layer in layers if layer.kind not in {"INPUT", "ASSERT"}
    }
    if set(public) != expected_keys:
        _fail("COVERAGE_ERROR", "outward box coverage is incomplete")
    bounds_records = [
        _box_record(lid, kind_by_id[lid], semantics[lid], public[lid])
        for lid in sorted(public)
    ]
    bounds_sha = _json_digest(bounds_records)
    torch_bounds: Dict[int, Bounds] = {
        lid: Bounds(
            lb=torch.from_numpy(box.lb.copy()).to(dtype=torch.float64),
            ub=torch.from_numpy(box.ub.copy()).to(dtype=torch.float64),
        )
        for lid, box in public.items()
    }
    body: Dict[str, Any] = {
        "schema": _SCHEMA,
        "status": "verified",
        "proof_authority": True,
        "authority_scope": "stored_binary64_network_relative_to_raw_box_input",
        "ordinary_interval_facts_consumed": False,
        "algorithm": _ALGORITHM,
        "supported_operators": sorted(_SUPPORTED),
        "input_spec_layer_id": input_spec_id,
        "assert_layer_id": assert_id,
        "single_lane": True,
        "batch_size": 1,
        "conv_channel_chunk": int(conv_channel_chunk),
        "box_key_semantics": {
            "RELU": "preactivation",
            "INPUT_SPEC": "output",
            "DENSE": "output",
            "CONV2D": "output",
            "ADD": "output",
            "FLATTEN": "output",
            "INPUT": "absent",
            "ASSERT": "absent",
        },
        "numeric_method": {
            "device": "cpu",
            "dtype": "IEEE-754-binary64",
            "dense_reduction": "direct_matrix_product",
            "conv_reduction": "unfold_grouped_channel_chunk_direct_matrix_product",
            "roundoff": "Higham-gamma-plus-min-subnormal-underflow-allowance",
            "conv_guard_scope": "independent-per-channel-chunk",
            "outward_endpoints": "nextafter",
        },
        "numeric_platform": dict(platform),
        "hashes": {
            "implementation_sha256": source_sha,
            "net_sha256": net_sha,
            "input_sha256": input_sha,
            "bounds_sha256": bounds_sha,
        },
        "input_manifest": input_manifest,
        "layer_records": layer_records,
        "bounds_records": bounds_records,
        "coverage_count": len(public),
        "elapsed_s_hex": float(time.monotonic() - started).hex(),
    }
    timer.check("receipt")
    body["receipt_sha256"] = _json_digest(body)
    timer.check("receipt_hash")
    sealed_graph = _seal_frozen_graph(
        layers,
        root_net_sha256=net_sha,
        input_spec_id=input_spec_id,
        assert_id=assert_id,
    )
    certificate = QueryDualBoxCertificate(
        bounds=MappingProxyType(torch_bounds),
        semantics=MappingProxyType(dict(semantics)),
        receipt=MappingProxyType(body),
        _sealed_frozen_graph=sealed_graph,
    )
    timer.check("return")
    return certificate


def verify_query_dual_box_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_net_sha256: Optional[str] = None,
    expected_input_sha256: Optional[str] = None,
    expected_bounds_sha256: Optional[str] = None,
    expected_implementation_sha256: Optional[str] = None,
) -> bool:
    """Integrity-check a receipt.  This does not authenticate or rerun proof."""

    try:
        body = dict(receipt)
        claimed = str(body.pop("receipt_sha256"))
        if (
            body.get("schema") != _SCHEMA
            or body.get("status") != "verified"
            or body.get("proof_authority") is not True
            or body.get("ordinary_interval_facts_consumed") is not False
            or body.get("algorithm") != _ALGORITHM
        ):
            return False
        if not hmac.compare_digest(_json_digest(body), claimed):
            return False
        hashes = body["hashes"]
        for key, expected in (
            ("net_sha256", expected_net_sha256),
            ("input_sha256", expected_input_sha256),
            ("bounds_sha256", expected_bounds_sha256),
            ("implementation_sha256", expected_implementation_sha256),
        ):
            if expected is not None and not hmac.compare_digest(
                str(hashes[key]), str(expected)
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def verify_query_dual_box_certificate(
    certificate: QueryDualBoxCertificate,
    *,
    net: Optional[Any] = None,
    expected_net_sha256: Optional[str] = None,
    expected_input_sha256: Optional[str] = None,
    expected_bounds_sha256: Optional[str] = None,
    expected_implementation_sha256: Optional[str] = None,
) -> bool:
    """Validate the complete live object, not merely its self-hashed receipt."""

    try:
        if not isinstance(certificate, QueryDualBoxCertificate):
            return False
        if certificate.proof_authority is not True:
            return False
        receipt = certificate.receipt
        hashes = receipt["hashes"]
        if not verify_query_dual_box_receipt(
            receipt,
            expected_net_sha256=expected_net_sha256,
            expected_input_sha256=expected_input_sha256,
            expected_bounds_sha256=expected_bounds_sha256,
            expected_implementation_sha256=expected_implementation_sha256,
        ):
            return False
        if not hmac.compare_digest(
            _source_sha256(), str(hashes["implementation_sha256"])
        ):
            return False
        if net is not None:
            layers, input_spec_id, _ = _freeze_graph(net)
            net_sha = _json_digest(_network_manifest(layers))
            spec = next(layer for layer in layers if layer.id == input_spec_id)
            input_sha = _json_digest(
                {
                    "layer_id": input_spec_id,
                    "kind": "BOX",
                    "lb_sha256": _array_digest(spec.params["lb"]),
                    "ub_sha256": _array_digest(spec.params["ub"]),
                    "shape": list(spec.params["lb"].shape),
                }
            )
            if not hmac.compare_digest(net_sha, str(hashes["net_sha256"])):
                return False
            if not hmac.compare_digest(input_sha, str(hashes["input_sha256"])):
                return False
            kinds = {
                layer.id: layer.kind
                for layer in layers
                if layer.kind not in {"INPUT", "ASSERT"}
            }
        else:
            kinds = {
                int(record["id"]): str(record["kind"])
                for record in receipt["bounds_records"]
            }
        semantics = {int(key): str(value) for key, value in certificate.semantics.items()}
        if set(certificate.bounds) != set(semantics) or set(kinds) != set(semantics):
            return False
        live_records = _live_bounds_records(certificate.bounds, semantics, kinds)
        live_sha = _json_digest(live_records)
        if not hmac.compare_digest(live_sha, str(hashes["bounds_sha256"])):
            return False
        if live_records != list(receipt["bounds_records"]):
            return False
        return True
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        QueryDualBoxError,
    ):
        return False


__all__ = [
    "QueryDualBoxCertificate",
    "QueryDualBoxError",
    "QueryDualBoxTimeout",
    "certify_query_dual_boxes",
    "query_dual_network_sha256",
    "verify_query_dual_box_certificate",
    "verify_query_dual_box_receipt",
]
