"""Isolated, default-off runtime-lineage adapter proof for S0-C3.

This module records an event tuple beside each lazy affine term, but treats
that tuple only as an untrusted observation.  Immediately before and after
calling the existing pure C3 planner it reconstructs the complete path from
the current ``layers/preds/succs`` objects, checks graph/value/operator
bijection and current numeric payloads, and compares a private custody/CAS
snapshot.  It cannot execute, materialize, rewrite, publish, or enable a
production path.

The event grammar is intentionally narrow: ACT ``CONV2D`` and ``SCALE``
consume one live ``ImplicitConv2DOp`` or ``DiagonalLinearOp`` respectively;
``BIAS`` and ``ADD`` consume no operator slot.  Every other traversed kind is
a barrier.  A CSR matrix which happens to be diagonal is not Scale evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import weakref
from typing import Any, Final, Mapping, Sequence

import numpy as np
from scipy import sparse as scipy_sparse

try:  # Runtime graph payloads are normally CPU torch tensors.
    import torch
except Exception:  # pragma: no cover - the ACT test environment has torch
    torch = None

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831 import (
    s0_c3_identity_middle_lineage_planner_prototype as c3,
)
from experiments.neural_hz_20260831.s0_c1_pure_tail_planner_prototype import (
    BudgetReservation,
    DescriptorEstimate,
    DescriptorRequest,
    PlannerBudget,
)


RUNTIME_ADAPTER_SCHEMA: Final = "s0_c3_runtime_lineage_adapter_v1"
RUNTIME_EVENT_SCHEMA: Final = "s0_c3_runtime_term_event_v1"
FORMAL_BASELINE: Final = "1870/2413"
RUNTIME_ADAPTER_NO_CLAIMS: Final = (
    "experiment_only_default_off",
    "non_executable_observation_and_planning_only",
    "no_production_import_or_runtime_flag_is_installed",
    "no_graph_repair_materialization_verifier_run_or_score_gain",
    "full_13_family_2413_replay_remains_required",
    "request_lifecycle_cleanup_is_prototype_only_not_production_integrated",
    "compact_source_authority_proxy_has_no_materializer_rebinding_contract",
    "detached_operator_snapshot_copy_cost_not_performance_gated",
    "production_requires_a_request_lock_or_mutation_epoch",
    "no_post_return_concurrency_or_atomic_publication_claim",
    "repeatable_compromise_of_the_trusted_pure_planner_is_out_of_scope",
)

_EVENT_KIND = {
    "CONV2D": "CONV",
    "SCALE": "SCALE",
    "BIAS": "BIAS",
    "ADD": "ADD",
}
_MULTIPLICATIVE = frozenset({"CONV2D", "SCALE"})
_TRANSITION = frozenset({"BIAS", "ADD"})
_SUPPORTED_CONV2D_PARAMS = frozenset(
    {
        "weight",
        "bias",
        "input_shape",
        "output_shape",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "in_channels",
        "out_channels",
    }
)
_SUPPORTED_SCALE_PARAMS = frozenset({"a"})
_SUPPORTED_BIAS_PARAMS = frozenset({"c"})
_LAYER_RAW_STATE_KEYS = frozenset(
    {"id", "kind", "params", "in_vars", "out_vars"}
)
_CUSTODY_MARKER = object()
_ARENA_MARKER = object()
_SOURCE_FIELDS = ("c", "Gc", "Gb", "Ac", "Ab", "b", "Auc", "Aub", "ub")
_SOURCE_STATE_KEYS = frozenset((*_SOURCE_FIELDS, "frame_id", "exact"))
_COMPRESSED_SPARSE_REQUIRED_STATE_KEYS = frozenset(
    {"_shape", "maxprint", "data", "indices", "indptr"}
)
_COMPRESSED_SPARSE_OPTIONAL_STATE_KEYS = frozenset(
    {
        "_has_sorted_indices",
        "_has_canonical_format",
        "_act_hz_zero_free",
    }
)
_AUTHORIZED_SOURCE_LAYER_KINDS = frozenset({"INPUT", "INPUT_SPEC", "RELU"})
_AUTHORIZED_TERMINAL_CONSUMER_KINDS = frozenset({"RELU"})
_SOURCE_ARENA_OWNERS: list[
    tuple[
        weakref.ReferenceType[object],
        weakref.ReferenceType[object],
        object,
    ]
] = []


class RuntimeLineageReject(ValueError):
    """Deterministic, fail-closed adapter rejection."""


@dataclass(frozen=True)
class _LineageCustody:
    marker: object
    canonical_sha256: str
    occurrence_objects: tuple[object, ...]
    operator_objects: tuple[object | None, ...]
    entry_source_layer_id: int
    source_object: object
    arena: object


@dataclass(frozen=True)
class RuntimeLineageEvent:
    """Term-local runtime observation; never authority by itself."""

    schema: str
    graph_sha256: str
    occurrence_layer_id: int
    occurrence_layer: object
    kind: str
    ordered_predecessor_layer_ids: tuple[int, ...]
    selected_input: int
    input_var_tokens: tuple[tuple[int, ...], ...]
    output_var_token: tuple[int, ...]
    operator_position: int
    layer_payload_sha256: str
    operator_occurrence: object | None
    operator_payload_sha256: str
    _custody: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class RuntimeLineageTermView:
    """A lazy term with the parallel graph-event tuple captured at creation."""

    source: object
    operators: tuple[object, ...]
    events: tuple[RuntimeLineageEvent, ...]


@dataclass(frozen=True)
class RuntimeLineageExprView:
    terms: tuple[RuntimeLineageTermView, ...]
    bias: object
    n_out: int
    frame_id: object


@dataclass(frozen=True)
class RuntimeAdapterDecision:
    """Planning-only result which always retains the runtime expression."""

    accepted: bool
    reason: str
    expression: RuntimeLineageExprView
    c3_expression: c3.C3AffineExprView | None
    planner_decision: c3.C3PlanDecision | None
    current_state_sha256: str
    adapter_schema: str = RUNTIME_ADAPTER_SCHEMA
    formal_baseline: str = FORMAL_BASELINE
    gain: int = 0
    default_enabled: bool = False
    execution_enabled: bool = False
    no_claims: tuple[str, ...] = RUNTIME_ADAPTER_NO_CLAIMS


@dataclass(frozen=True)
class _NumericSnapshot:
    canonical: np.ndarray = field(compare=False, repr=False)
    payload: bytes = field(compare=False, repr=False)
    sha256: str


@dataclass(frozen=True)
class _LayerSnapshot:
    layer_id: int
    layer: object = field(compare=False, repr=False)
    kind: str
    params: Mapping[str, Any] = field(compare=False, repr=False)
    in_vars: tuple[int, ...]
    out_vars: tuple[int, ...]
    preds: tuple[int, ...]
    succs: tuple[int, ...]
    input_tokens: tuple[tuple[int, ...], ...]
    layer_payload_sha256: str


@dataclass(frozen=True)
class _GraphSnapshot:
    layers: tuple[_LayerSnapshot, ...]
    graph_sha256: str
    reference_objects: tuple[object, ...] = field(compare=False, repr=False)


@dataclass(frozen=True)
class _SourceBoundarySnapshot:
    by_layer: tuple[tuple[int, object], ...] = field(compare=False, repr=False)
    key_payload: bytes
    reference_objects: tuple[object, ...] = field(compare=False, repr=False)

    def source_at(self, layer_id: int) -> object:
        for candidate, source in self.by_layer:
            if candidate == layer_id:
                return source
        raise RuntimeLineageReject("runtime_source_boundary_missing")


@dataclass(frozen=True)
class _ExpectedAddPrefix:
    add_layer_id: int
    selected_input: int
    entry_layer_id: int
    source: object = field(compare=False, repr=False)
    operator_prefix: tuple[object, ...] = field(compare=False, repr=False)
    occurrence_prefix: tuple[object, ...] = field(compare=False, repr=False)
    canonical_sha256: str


@dataclass(frozen=True)
class _ExpectedTerminalTerm:
    terminal_layer_id: int
    entry_layer_id: int
    source: object = field(compare=False, repr=False)
    operators: tuple[object, ...] = field(compare=False, repr=False)
    occurrences: tuple[object, ...] = field(compare=False, repr=False)
    canonical_sha256: str


@dataclass(frozen=True)
class _RuntimeOperandLineage:
    source: object = field(compare=False, repr=False)
    operators: tuple[object, ...] = field(compare=False, repr=False)
    path_layer_ids: tuple[int, ...]
    events: tuple[RuntimeLineageEvent, ...] = field(compare=False, repr=False)
    custody: _LineageCustody = field(compare=False, repr=False)


@dataclass(frozen=True)
class _RuntimeAffineExprCacheView:
    terms: tuple[_RuntimeOperandLineage, ...] = field(
        compare=False, repr=False
    )
    bias: object = field(compare=False, repr=False)
    n_out: int
    frame_id: object = field(compare=False, repr=False)


@dataclass(frozen=True)
class _RegisteredAddOperands:
    add_layer_id: int
    operand_cache_entries: tuple[
        _RuntimeAffineExprCacheView,
        _RuntimeAffineExprCacheView,
    ] = field(compare=False, repr=False)
    operands: tuple[
        tuple[_RuntimeOperandLineage, ...],
        tuple[_RuntimeOperandLineage, ...],
    ] = field(compare=False, repr=False)
    expected_prefixes: tuple[_ExpectedAddPrefix, ...] = field(
        compare=False, repr=False
    )


@dataclass(frozen=True)
class _RegisteredTerminalExpression:
    terminal_layer_id: int
    terminal_consumer_layer_id: int | None
    cache_entry: _RuntimeAffineExprCacheView = field(
        compare=False, repr=False
    )
    terms: tuple[_RuntimeOperandLineage, ...] = field(
        compare=False, repr=False
    )
    expected_terms: tuple[_ExpectedTerminalTerm, ...] = field(
        compare=False, repr=False
    )


@dataclass(frozen=True)
class _FactorAllocationRecord:
    kind: str
    frame_id: int
    key: tuple[int, ...]
    slots: tuple[int, ...]
    n_cont: int
    n_bin: int


@dataclass(frozen=True)
class _FactorSourcePrefix:
    source: object = field(compare=False, repr=False)
    frame_id: int
    frame_root: object = field(compare=False, repr=False)
    continuous_slots: tuple[object, ...] = field(
        compare=False, repr=False
    )
    binary_slots: tuple[object, ...] = field(compare=False, repr=False)


class _RuntimeFactorAllocator:
    """Request-owned allocator lineage; integer frame ids are not authority."""

    __slots__ = (
        "marker",
        "request_owner",
        "owner_cache",
        "owner_cache_identity",
        "next_frame_id",
        "frame_roots",
        "initial_widths",
        "frame_widths",
        "continuous_slot_tokens",
        "binary_slot_tokens",
        "relu_slots",
        "aux_slots",
        "allocation_history",
        "source_prefixes",
        "sealed_payload",
        "sealed_references",
        "sealed",
    )

    def __init__(
        self, request_owner: object, owner_cache: dict[int, object]
    ) -> None:
        self.marker = _ARENA_MARKER
        self.request_owner = request_owner
        self.owner_cache = owner_cache
        self.owner_cache_identity = owner_cache
        self.next_frame_id = 0
        self.frame_roots: dict[int, object] = {}
        self.initial_widths: dict[int, tuple[int, int]] = {}
        self.frame_widths: dict[int, tuple[int, int]] = {}
        self.continuous_slot_tokens: dict[int, tuple[object, ...]] = {}
        self.binary_slot_tokens: dict[int, tuple[object, ...]] = {}
        self.relu_slots: dict[
            tuple[int, int, int], tuple[int, int, int]
        ] = {}
        self.aux_slots: dict[tuple[int, int], tuple[int, ...]] = {}
        self.allocation_history: list[_FactorAllocationRecord] | tuple[
            _FactorAllocationRecord, ...
        ] = []
        self.source_prefixes: list[_FactorSourcePrefix] | tuple[
            _FactorSourcePrefix, ...
        ] = []
        self.sealed_payload: bytes | None = None
        self.sealed_references: tuple[object, ...] | None = None
        self.sealed = False


class _RuntimeLineageArena:
    """One request/cache capability; frame integers alone are not custody."""

    __slots__ = (
        "__weakref__",
        "marker",
        "nonce",
        "graph_sha256",
        "graph_reference_objects",
        "owner_cache",
        "owner_cache_identity",
        "owner_affine_cache",
        "owner_affine_cache_identity",
        "factor_allocator",
        "factor_allocator_identity",
        "factor_allocator_payload",
        "factor_allocator_references",
        "affine_entries",
        "affine_payloads",
        "affine_reference_snapshots",
        "source_entries",
        "sealed_source_entries",
        "source_payloads",
        "source_reference_snapshots",
        "frame_payload",
        "closed",
    )

    def __init__(
        self,
        graph: _GraphSnapshot,
        owner_cache: Mapping[int, object],
        owner_affine_cache: Mapping[int, object],
        factor_allocator: _RuntimeFactorAllocator,
        marker: object,
    ) -> None:
        if marker is not _ARENA_MARKER:
            raise RuntimeLineageReject("runtime_arena_private_constructor")
        self.marker = marker
        self.nonce = object()
        self.graph_sha256 = graph.graph_sha256
        self.graph_reference_objects = graph.reference_objects
        self.owner_cache = owner_cache
        self.owner_cache_identity = owner_cache
        self.owner_affine_cache = owner_affine_cache
        self.owner_affine_cache_identity = owner_affine_cache
        self.factor_allocator = factor_allocator
        self.factor_allocator_identity = factor_allocator
        (
            self.factor_allocator_payload,
            self.factor_allocator_references,
        ) = _factor_allocator_guard(factor_allocator, require_sealed=True)
        self.affine_entries: list[
            tuple[int, _RuntimeAffineExprCacheView]
        ] = []
        self.affine_payloads: list[bytes] = []
        self.affine_reference_snapshots: list[tuple[object, ...]] = []
        self.source_entries: list[tuple[int, object]] | tuple[
            tuple[int, object], ...
        ] = []
        self.sealed_source_entries: tuple[tuple[int, object], ...] | None = None
        self.source_payloads: list[bytes] | tuple[bytes, ...] = []
        self.source_reference_snapshots: list[
            tuple[object, ...]
        ] | tuple[tuple[object, ...], ...] = []
        self.frame_payload: bytes | None = None
        self.closed = False


class _RuntimeLineageRegistry:
    """Private recorder/capability sealed before adapter use."""

    __slots__ = (
        "marker",
        "arena",
        "graph_sha256",
        "graph_reference_objects",
        "source_boundaries",
        "add_operand_snapshots",
        "terminal_expression_snapshot",
        "sealed_authority_payload",
        "sealed_authority_references",
        "sealed",
    )

    def __init__(
        self,
        graph: _GraphSnapshot,
        source_boundaries: _SourceBoundarySnapshot,
        arena: _RuntimeLineageArena,
        marker: object,
    ) -> None:
        if marker is not _CUSTODY_MARKER:
            raise RuntimeLineageReject("runtime_registry_private_constructor")
        self.marker = marker
        self.arena = arena
        self.graph_sha256 = graph.graph_sha256
        self.graph_reference_objects = graph.reference_objects
        self.source_boundaries = source_boundaries
        self.add_operand_snapshots: list[_RegisteredAddOperands] | tuple[
            _RegisteredAddOperands, ...
        ] = []
        self.terminal_expression_snapshot: (
            _RegisteredTerminalExpression | None
        ) = None
        self.sealed_authority_payload: bytes | None = None
        self.sealed_authority_references: tuple[object, ...] | None = None
        self.sealed = False


@dataclass(frozen=True)
class _DerivedState:
    c3_expression: c3.C3AffineExprView
    term_events: tuple[tuple[RuntimeLineageEvent, ...], ...]
    current_state_sha256: str
    reference_objects: tuple[object, ...] = field(compare=False, repr=False)


@dataclass(frozen=True)
class _DetachedSourceAuthorityProxy:
    """Compact planner-only token; it cannot materialize an HZ source."""

    frame_id: object
    exact: bool
    semantic_sha256: str
    c: tuple[()] = ()
    Gc: tuple[()] = ()
    Gb: tuple[()] = ()
    Ac: tuple[()] = ()
    Ab: tuple[()] = ()
    b: tuple[()] = ()
    Auc: tuple[()] = ()
    Aub: tuple[()] = ()
    ub: tuple[()] = ()


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise RuntimeLineageReject(f"{name}_not_exact_integer")
    if value < minimum:
        raise RuntimeLineageReject(f"{name}_out_of_range")
    return value


def _exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RuntimeLineageReject(f"{name}_not_exact_bool")
    return value


def _exact_str(value: Any, name: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise RuntimeLineageReject(f"{name}_not_exact_string")
    return value


def _exact_bytes(value: Any, name: str, *, nonempty: bool = True) -> bytes:
    if type(value) is not bytes or (nonempty and not value):
        raise RuntimeLineageReject(f"{name}_not_exact_bytes")
    return value


def _exact_sha256(value: Any, name: str) -> str:
    digest = _exact_str(value, name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RuntimeLineageReject(f"{name}_not_canonical_sha256")
    return digest


def _exact_int_tuple(
    value: Any,
    name: str,
    *,
    nonempty: bool = False,
    unique: bool = True,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise RuntimeLineageReject(f"{name}_not_exact_tuple")
    for item in value:
        _exact_int(item, name)
    result = value
    if nonempty and not result:
        raise RuntimeLineageReject(f"{name}_empty")
    if unique and len(result) != len(set(result)):
        raise RuntimeLineageReject(f"{name}_duplicate")
    return result


def _int_tuple(
    value: Any,
    name: str,
    *,
    nonempty: bool = False,
    unique: bool = True,
) -> tuple[int, ...]:
    if type(value) not in (list, tuple):
        raise RuntimeLineageReject(f"{name}_not_list_or_tuple")
    result = tuple(_exact_int(item, name) for item in value)
    if nonempty and not result:
        raise RuntimeLineageReject(f"{name}_empty")
    if unique and len(result) != len(set(result)):
        raise RuntimeLineageReject(f"{name}_duplicate")
    return result


def _pair(value: Any, name: str, *, minimum: int) -> tuple[int, int]:
    if type(value) is int:
        item = _exact_int(value, name, minimum=minimum)
        return item, item
    values = _int_tuple(value, name, unique=False)
    if len(values) != 2 or any(item < minimum for item in values):
        raise RuntimeLineageReject(f"{name}_not_integer_pair")
    return values


def _shape(value: Any, name: str) -> tuple[int, ...]:
    result = _int_tuple(value, name, nonempty=True, unique=False)
    if any(item <= 0 for item in result):
        raise RuntimeLineageReject(f"{name}_nonpositive")
    return result


def _stable_token_payload(value: object, name: str, *, depth: int = 0) -> bytes:
    if depth > 32 or value is None or type(value) is bool:
        raise RuntimeLineageReject(f"{name}_not_stable_token")
    if type(value) is int:
        raw = str(value).encode("ascii")
        return b"I" + len(raw).to_bytes(8, "big") + raw
    if type(value) is str:
        raw = value.encode("utf-8")
        if not raw:
            raise RuntimeLineageReject(f"{name}_not_stable_token")
        return b"S" + len(raw).to_bytes(8, "big") + raw
    if type(value) is bytes:
        if not value:
            raise RuntimeLineageReject(f"{name}_not_stable_token")
        return b"Y" + len(value).to_bytes(8, "big") + value
    if type(value) is tuple and value:
        parts = tuple(
            _stable_token_payload(item, name, depth=depth + 1)
            for item in value
        )
        return b"T" + len(parts).to_bytes(8, "big") + b"".join(
            len(part).to_bytes(8, "big") + part for part in parts
        )
    raise RuntimeLineageReject(f"{name}_not_stable_token")


def _numeric_snapshot(value: Any, name: str, *, ndim: int | None = None) -> _NumericSnapshot:
    if torch is not None and type(value) is torch.Tensor:
        tensor_state = object.__getattribute__(value, "__dict__")
        if type(tensor_state) is not dict or tensor_state:
            raise RuntimeLineageReject(f"{name}_tensor_instance_state")
        device = torch.Tensor.device.__get__(value, torch.Tensor)
        layout = torch.Tensor.layout.__get__(value, torch.Tensor)
        if device.type != "cpu" or layout != torch.strided:
            raise RuntimeLineageReject(f"{name}_tensor_not_cpu_strided")
        detached_tensor = torch.Tensor.detach(value)
        raw = torch.Tensor.numpy(detached_tensor)
        dtype_name = str(torch.Tensor.dtype.__get__(value, torch.Tensor))
    else:
        if type(value) is not np.ndarray:
            raise RuntimeLineageReject(f"{name}_not_exact_numeric_array")
        raw = value
        dtype_name = raw.dtype.str
    if ndim is not None and raw.ndim != ndim:
        raise RuntimeLineageReject(f"{name}_rank")
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise RuntimeLineageReject(f"{name}_not_real_numeric")
    canonical = np.array(raw, dtype="<f8", order="C", copy=True)
    if not np.all(np.isfinite(canonical)):
        raise RuntimeLineageReject(f"{name}_nonfinite")
    header = json.dumps(
        {"dtype": dtype_name, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    payload = (
        len(header).to_bytes(8, "big")
        + header
        + canonical.tobytes(order="C")
    )
    return _NumericSnapshot(canonical, payload, hashlib.sha256(payload).hexdigest())


def _raw_numeric_array_payload(value: Any, name: str) -> bytes:
    if type(value) is not np.ndarray:
        raise RuntimeLineageReject(f"{name}_not_exact_ndarray")
    raw = value
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise RuntimeLineageReject(f"{name}_not_real_numeric_vector")
    owned = np.array(raw, order="C", copy=True)
    if owned.dtype.kind == "f" and not np.all(np.isfinite(owned)):
        raise RuntimeLineageReject(f"{name}_nonfinite")
    header = _metadata_payload([owned.dtype.str, tuple(owned.shape)])
    return len(header).to_bytes(8, "big") + header + owned.tobytes(order="C")


def _payload_digest(parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"s0-c3-runtime-layer-payload-v1\0")
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _metadata_payload(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _parameter_guard(
    value: object, name: str, *, depth: int = 0
) -> tuple[bytes, tuple[object, ...]]:
    if depth > 32:
        raise RuntimeLineageReject(f"{name}_nesting_too_deep")
    if isinstance(value, Mapping):
        if type(value) is not dict:
            raise RuntimeLineageReject(f"{name}_mapping_type")
        if any(type(key) is not str or not key for key in value.keys()):
            raise RuntimeLineageReject(f"{name}_mapping_key_type")
        refs: list[object] = [value]
        parts: list[bytes] = []
        for key in sorted(value.keys()):
            payload, child_refs = _parameter_guard(
                value[key], f"{name}_{key}", depth=depth + 1
            )
            key_payload = key.encode("utf-8")
            parts.append(
                len(key_payload).to_bytes(8, "big")
                + key_payload
                + len(payload).to_bytes(8, "big")
                + payload
            )
            refs.extend(child_refs)
        return b"M" + len(parts).to_bytes(8, "big") + b"".join(parts), tuple(refs)
    if type(value) in (list, tuple):
        refs = [value]
        parts = []
        for index, item in enumerate(value):
            payload, child_refs = _parameter_guard(
                item, f"{name}_{index}", depth=depth + 1
            )
            parts.append(len(payload).to_bytes(8, "big") + payload)
            refs.extend(child_refs)
        tag = b"L" if type(value) is list else b"T"
        return tag + len(parts).to_bytes(8, "big") + b"".join(parts), tuple(refs)
    if type(value) in {
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    }:
        snapshot = _numeric_snapshot(np.asarray(value), name)
        return b"N" + snapshot.payload, (value,)
    if (
        type(value) is np.ndarray
        or type(value) in (scipy_sparse.csr_matrix, scipy_sparse.csc_matrix)
        or (torch is not None and type(value) is torch.Tensor)
    ):
        payload, refs = _object_guard(value, name)
        return b"A" + payload, refs
    if value is None:
        return b"Z", (value,)
    if type(value) is bool:
        return b"B1" if value else b"B0", (value,)
    if type(value) is int:
        payload = str(value).encode("ascii")
        return b"I" + len(payload).to_bytes(8, "big") + payload, (value,)
    if type(value) is float:
        if not np.isfinite(value):
            raise RuntimeLineageReject(f"{name}_nonfinite")
        payload = np.asarray(value, dtype="<f8").tobytes()
        return b"F" + payload, (value,)
    if type(value) in (str, bytes):
        payload = value.encode("utf-8") if type(value) is str else value
        tag = b"S" if type(value) is str else b"Y"
        return tag + len(payload).to_bytes(8, "big") + payload, (value,)
    raise RuntimeLineageReject(f"{name}_unsupported_mutable_type")


def _layer_payload(
    kind: str,
    params: Mapping[str, Any],
    layer_id: int,
    output_width: int,
) -> tuple[str, tuple[object, ...]]:
    prefix = f"layer_{layer_id}"
    if any(type(key) is not str or not key for key in params):
        raise RuntimeLineageReject(f"{prefix}_parameter_key_type")
    supported = (
        _SUPPORTED_CONV2D_PARAMS
        if kind == "CONV2D"
        else _SUPPORTED_SCALE_PARAMS
        if kind == "SCALE"
        else _SUPPORTED_BIAS_PARAMS
        if kind == "BIAS"
        else None
    )
    if supported is not None and (unsupported := set(params) - supported):
        raise RuntimeLineageReject(
            f"{kind.lower()}_semantic_parameter_unsupported:"
            + ",".join(sorted(unsupported))
        )
    complete_payload, complete_refs = _parameter_guard(params, f"{prefix}_params")
    parts: list[bytes] = [kind.encode("ascii"), complete_payload]
    if kind == "CONV2D":
        if "weight" not in params:
            raise RuntimeLineageReject("conv_weight_missing")
        weight = _numeric_snapshot(params["weight"], f"{prefix}_weight", ndim=4)
        input_shape = _shape(params.get("input_shape"), f"{prefix}_input_shape")
        output_shape = _shape(params.get("output_shape"), f"{prefix}_output_shape")
        stride = _pair(params.get("stride", 1), f"{prefix}_stride", minimum=1)
        padding = _pair(params.get("padding", 0), f"{prefix}_padding", minimum=0)
        dilation = _pair(params.get("dilation", 1), f"{prefix}_dilation", minimum=1)
        groups = _exact_int(params.get("groups", 1), f"{prefix}_groups", minimum=1)
        parts.extend(
            (
                weight.payload,
                _metadata_payload(
                    [input_shape, output_shape, stride, padding, dilation, groups]
                ),
            )
        )
        if "bias" in params and params["bias"] is not None:
            parts.append(_numeric_snapshot(params["bias"], f"{prefix}_bias", ndim=1).payload)
    elif kind == "SCALE":
        if "a" not in params:
            raise RuntimeLineageReject("scale_payload_missing")
        parts.append(_numeric_snapshot(params["a"], f"{prefix}_scale", ndim=1).payload)
    elif kind == "BIAS":
        if "c" not in params:
            raise RuntimeLineageReject("bias_payload_missing")
        parts.append(_numeric_snapshot(params["c"], f"{prefix}_bias", ndim=1).payload)
    elif kind == "ADD":
        unsupported = set(params) - {"x_vars", "y_vars", "bias"}
        if unsupported:
            raise RuntimeLineageReject(
                "add_semantic_parameter_unsupported"
            )
        x_vars = _int_tuple(params.get("x_vars"), f"{prefix}_x_vars", nonempty=True)
        y_vars = _int_tuple(params.get("y_vars"), f"{prefix}_y_vars", nonempty=True)
        parts.append(_metadata_payload([x_vars, y_vars]))
        if "bias" in params and params["bias"] is not None:
            raw_add_bias = params["bias"]
            if type(raw_add_bias) in (int, float):
                add_bias = _numeric_snapshot(
                    np.asarray(raw_add_bias, dtype=np.float64),
                    f"{prefix}_add_bias",
                    ndim=0,
                )
            else:
                add_bias = _numeric_snapshot(
                    raw_add_bias,
                    f"{prefix}_add_bias",
                    ndim=1,
                )
                if int(add_bias.canonical.size) != output_width:
                    raise RuntimeLineageReject(
                        "add_inline_bias_shape_unsupported"
                    )
            if np.any(add_bias.canonical != 0.0):
                raise RuntimeLineageReject(
                    "add_inline_bias_transition_not_explicit"
                )
            parts.append(add_bias.payload)
    return _payload_digest(parts), complete_refs


def _freeze_edge_map(
    mapping: Mapping[int, Sequence[int]], layer_count: int, name: str
) -> tuple[tuple[int, ...], ...]:
    if type(mapping) is not dict:
        raise RuntimeLineageReject(f"{name}_not_mapping")
    if set(mapping.keys()) != set(range(layer_count)) or any(type(key) is not int for key in mapping):
        raise RuntimeLineageReject(f"{name}_keys_not_complete_exact_ints")
    result = []
    for layer_id in range(layer_count):
        neighbors = _int_tuple(mapping[layer_id], f"{name}_{layer_id}")
        if any(item >= layer_count for item in neighbors):
            raise RuntimeLineageReject(f"{name}_unknown_layer")
        result.append(neighbors)
    return tuple(result)


def _validate_global_ssa(records: Sequence[_LayerSnapshot]) -> None:
    """Require one current producer for every value in the complete graph."""

    producer: dict[int, int] = {}
    for layer in records:
        missing = tuple(value for value in layer.in_vars if value not in producer)
        if missing:
            raise RuntimeLineageReject("graph_input_variable_without_prior_producer")
        if layer.kind == "ADD":
            expected: list[int] = []
            for operand in layer.input_tokens:
                operand_producers = {producer[value] for value in operand}
                if len(operand_producers) != 1:
                    raise RuntimeLineageReject("add_operand_producer_ambiguity")
                expected.append(next(iter(operand_producers)))
            expected_preds = tuple(expected)
        else:
            ordered: list[int] = []
            for value in layer.in_vars:
                candidate = producer[value]
                if candidate not in ordered:
                    ordered.append(candidate)
            expected_preds = tuple(ordered)
        if expected_preds != layer.preds:
            raise RuntimeLineageReject("graph_predecessor_producer_mismatch")

        duplicate_outputs = tuple(
            value for value in layer.out_vars if value in producer
        )
        exact_wrapper_alias = (
            layer.kind in {"INPUT_SPEC", "ASSERT"}
            and bool(layer.out_vars)
            and layer.out_vars == layer.in_vars
        )
        if duplicate_outputs and not exact_wrapper_alias:
            raise RuntimeLineageReject("graph_duplicate_output_variable_producer")
        for value in layer.out_vars:
            producer[value] = layer.layer_id


def _freeze_graph(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
) -> _GraphSnapshot:
    if type(layers) not in (list, tuple) or not layers:
        raise RuntimeLineageReject("layers_not_nonempty_list_or_tuple")
    frozen_preds = _freeze_edge_map(preds, len(layers), "preds")
    frozen_succs = _freeze_edge_map(succs, len(layers), "succs")
    inverse: list[list[int]] = [[] for _ in layers]
    for layer_id, incoming in enumerate(frozen_preds):
        for predecessor in incoming:
            if predecessor >= layer_id:
                raise RuntimeLineageReject("graph_not_topologically_ordered_dag")
            inverse[predecessor].append(layer_id)
    if tuple(tuple(values) for values in inverse) != frozen_succs:
        raise RuntimeLineageReject("predecessor_successor_asymmetry")

    records: list[_LayerSnapshot] = []
    refs: list[object] = [layers, preds, succs]
    serializable: list[dict[str, Any]] = []
    for layer_id, layer in enumerate(layers):
        layer_type = type(layer)
        try:
            raw_state = object.__getattribute__(layer, "__dict__")
        except Exception as exc:
            raise RuntimeLineageReject("layer_raw_state_unavailable") from exc
        if type(raw_state) is not dict or any(
            type(key) is not str for key in raw_state
        ):
            raise RuntimeLineageReject("layer_raw_state_schema_mismatch")
        if set(raw_state) != _LAYER_RAW_STATE_KEYS:
            raise RuntimeLineageReject("layer_raw_state_schema_mismatch")
        if _exact_int(raw_state["id"], "layer_id") != layer_id:
            raise RuntimeLineageReject("layer_ids_not_contiguous")
        kind = raw_state["kind"]
        if type(kind) is not str or not kind:
            raise RuntimeLineageReject("layer_kind_not_nonempty_exact_str")
        params = raw_state["params"]
        if type(params) is not dict:
            raise RuntimeLineageReject("layer_params_not_mapping")
        raw_in = raw_state["in_vars"]
        raw_out = raw_state["out_vars"]
        in_vars = _int_tuple(raw_in, f"layer_{layer_id}_in_vars")
        out_vars = _int_tuple(raw_out, f"layer_{layer_id}_out_vars")
        payload_sha, payload_refs = _layer_payload(
            kind, params, layer_id, len(out_vars)
        )
        if kind == "ADD":
            x_vars = _int_tuple(params["x_vars"], "add_x_vars", nonempty=True)
            y_vars = _int_tuple(params["y_vars"], "add_y_vars", nonempty=True)
            if in_vars != x_vars + y_vars:
                raise RuntimeLineageReject("add_operands_do_not_equal_in_vars")
            input_tokens = (x_vars, y_vars)
        else:
            input_tokens = (in_vars,) if in_vars else ()
        records.append(
            _LayerSnapshot(
                layer_id,
                layer,
                kind,
                params,
                in_vars,
                out_vars,
                frozen_preds[layer_id],
                frozen_succs[layer_id],
                input_tokens,
                payload_sha,
            )
        )
        refs.extend(
            (
                layer,
                layer_type,
                raw_state,
                params,
                raw_in,
                raw_out,
                preds[layer_id],
                succs[layer_id],
            )
        )
        refs.extend(payload_refs)
        serializable.append(
            {
                "id": layer_id,
                "kind": kind,
                "in_vars": in_vars,
                "out_vars": out_vars,
                "preds": frozen_preds[layer_id],
                "succs": frozen_succs[layer_id],
                "payload": payload_sha,
            }
        )
    _validate_global_ssa(records)
    graph_payload = json.dumps(
        serializable, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return _GraphSnapshot(
        tuple(records), hashlib.sha256(graph_payload).hexdigest(), tuple(refs)
    )


def _freeze_source_boundaries(
    source_by_layer: Mapping[int, object], graph: _GraphSnapshot
) -> _SourceBoundarySnapshot:
    """Freeze the current runtime cache that authorizes exact source cuts."""

    if not isinstance(source_by_layer, Mapping) or not source_by_layer:
        raise RuntimeLineageReject("runtime_source_boundary_map_missing")
    entries: list[tuple[int, object]] = []
    seen_sources: list[object] = []
    for raw_layer_id, source in source_by_layer.items():
        if type(raw_layer_id) is not int:
            raise RuntimeLineageReject("runtime_source_boundary_key_not_exact_int")
        layer_id = _exact_int(raw_layer_id, "runtime_source_boundary_layer")
        if layer_id >= len(graph.layers):
            raise RuntimeLineageReject("runtime_source_boundary_unknown_layer")
        if graph.layers[layer_id].kind not in _AUTHORIZED_SOURCE_LAYER_KINDS:
            raise RuntimeLineageReject(
                "runtime_source_boundary_kind_not_authorized"
            )
        if source is None:
            raise RuntimeLineageReject("runtime_source_boundary_none")
        if any(source is prior for prior in seen_sources):
            raise RuntimeLineageReject("runtime_source_object_aliases_boundaries")
        seen_sources.append(source)
        entries.append((layer_id, source))
    entries.sort(key=lambda item: item[0])
    keys = tuple(layer_id for layer_id, _ in entries)
    if len(keys) != len(set(keys)):
        raise RuntimeLineageReject("runtime_source_boundary_duplicate_layer")
    return _SourceBoundarySnapshot(
        tuple(entries),
        _metadata_payload(keys),
        (source_by_layer, *(source for _, source in entries)),
    )


def _source_boundary_snapshot_guard(
    snapshot: object,
) -> _SourceBoundarySnapshot:
    if type(snapshot) is not _SourceBoundarySnapshot:
        raise RuntimeLineageReject("runtime_source_boundary_snapshot_type")
    if type(snapshot.by_layer) is not tuple or type(
        snapshot.reference_objects
    ) is not tuple:
        raise RuntimeLineageReject("runtime_source_boundary_container_type")
    key_payload = _exact_bytes(
        snapshot.key_payload, "runtime_source_boundary_key_payload"
    )
    entries: list[tuple[int, object]] = []
    for entry in snapshot.by_layer:
        if type(entry) is not tuple or len(entry) != 2:
            raise RuntimeLineageReject("runtime_source_boundary_entry_type")
        layer_id = _exact_int(
            entry[0], "runtime_source_boundary_snapshot_layer"
        )
        entries.append((layer_id, entry[1]))
    keys = tuple(layer_id for layer_id, _ in entries)
    if len(keys) != len(set(keys)) or key_payload != _metadata_payload(keys):
        raise RuntimeLineageReject("runtime_source_boundary_key_payload_mismatch")
    if (
        len(snapshot.reference_objects) != len(entries) + 1
        or type(snapshot.reference_objects[0]) is not dict
        or any(
            snapshot.reference_objects[index + 1] is not source
            for index, (_, source) in enumerate(entries)
        )
    ):
        raise RuntimeLineageReject("runtime_source_boundary_reference_mismatch")
    return snapshot


def _identity_tuple_equal(
    left: tuple[object, ...], right: tuple[object, ...]
) -> bool:
    if type(left) is not tuple or type(right) is not tuple:
        return False
    return len(left) == len(right) and all(
        first is second for first, second in zip(left, right, strict=True)
    )


def _begin_private_runtime_factor_allocator(
    request_owner: object, owner_cache: dict[int, object]
) -> _RuntimeFactorAllocator:
    if request_owner is None or type(owner_cache) is not dict or not owner_cache:
        raise RuntimeLineageReject("runtime_factor_allocator_owner_missing")
    return _RuntimeFactorAllocator(request_owner, owner_cache)


def _factor_source_dimensions(source: object) -> tuple[int, int, int]:
    state = _sparse_hz_source_raw_state(source)
    gc_shape = _validate_sparse_hz_csr(
        state["Gc"], "runtime_source_Gc"
    )
    gb_shape = _validate_sparse_hz_csr(
        state["Gb"], "runtime_source_Gb"
    )
    return (
        _exact_int(state["frame_id"], "runtime_factor_source_frame"),
        gc_shape[1],
        gb_shape[1],
    )


def _require_open_factor_allocator(
    allocator: object,
) -> _RuntimeFactorAllocator:
    if (
        type(allocator) is not _RuntimeFactorAllocator
        or allocator.marker is not _ARENA_MARKER
        or allocator.sealed is not False
        or type(allocator.allocation_history) is not list
        or type(allocator.source_prefixes) is not list
    ):
        raise RuntimeLineageReject("runtime_factor_allocator_not_open")
    if (
        type(allocator.owner_cache) is not dict
        or allocator.owner_cache is not allocator.owner_cache_identity
    ):
        raise RuntimeLineageReject("runtime_factor_allocator_cache_cas_mismatch")
    return allocator


def _append_factor_source_prefix(
    allocator: _RuntimeFactorAllocator, source: object
) -> None:
    if not any(source is value for value in allocator.owner_cache.values()):
        raise RuntimeLineageReject(
            "runtime_factor_source_not_owned_by_current_cache"
        )
    if any(prefix.source is source for prefix in allocator.source_prefixes):
        raise RuntimeLineageReject("runtime_factor_source_prefix_duplicate")
    frame_id, n_cont, n_bin = _factor_source_dimensions(source)
    if frame_id not in allocator.frame_roots:
        raise RuntimeLineageReject("runtime_factor_frame_root_missing")
    if allocator.frame_widths.get(frame_id) != (n_cont, n_bin):
        raise RuntimeLineageReject("runtime_factor_source_not_current_prefix")
    continuous = allocator.continuous_slot_tokens[frame_id]
    binary = allocator.binary_slot_tokens[frame_id]
    if len(continuous) != n_cont or len(binary) != n_bin:
        raise RuntimeLineageReject("runtime_factor_slot_width_mismatch")
    allocator.source_prefixes.append(
        _FactorSourcePrefix(
            source,
            frame_id,
            allocator.frame_roots[frame_id],
            continuous,
            binary,
        )
    )


def _record_private_runtime_factor_frame_root(
    allocator: object, source: object
) -> None:
    private = _require_open_factor_allocator(allocator)
    frame_id, n_cont, n_bin = _factor_source_dimensions(source)
    if frame_id in private.frame_roots:
        raise RuntimeLineageReject("runtime_factor_frame_root_duplicate")
    private.frame_roots[frame_id] = object()
    private.initial_widths[frame_id] = (n_cont, n_bin)
    private.frame_widths[frame_id] = (n_cont, n_bin)
    private.continuous_slot_tokens[frame_id] = tuple(
        object() for _ in range(n_cont)
    )
    private.binary_slot_tokens[frame_id] = tuple(
        object() for _ in range(n_bin)
    )
    private.next_frame_id = max(private.next_frame_id, frame_id + 1)
    private.allocation_history.append(
        _FactorAllocationRecord("ROOT", frame_id, (), (), n_cont, n_bin)
    )
    _append_factor_source_prefix(private, source)


def _record_private_runtime_factor_source_prefix(
    allocator: object, source: object
) -> None:
    _append_factor_source_prefix(
        _require_open_factor_allocator(allocator), source
    )


def _record_private_runtime_factor_relu_allocation(
    allocator: object,
    frame_id: int,
    layer_id: int,
    neuron: int,
    *,
    compact: bool = False,
) -> None:
    private = _require_open_factor_allocator(allocator)
    is_compact = _exact_bool(compact, "runtime_factor_relu_compact")
    frame = _exact_int(frame_id, "runtime_factor_relu_frame")
    layer = _exact_int(layer_id, "runtime_factor_relu_layer")
    unit = _exact_int(neuron, "runtime_factor_relu_neuron")
    key = (frame, layer, unit)
    if frame not in private.frame_roots or key in private.relu_slots:
        raise RuntimeLineageReject("runtime_factor_relu_allocation_invalid")
    n_cont, n_bin = private.frame_widths[frame]
    slots = (
        (n_cont, n_cont, n_bin)
        if is_compact
        else (n_cont, n_cont + 1, n_bin)
    )
    cont_growth = 1 if is_compact else 2
    private.relu_slots[key] = slots
    private.continuous_slot_tokens[frame] = (
        *private.continuous_slot_tokens[frame],
        *(object() for _ in range(cont_growth)),
    )
    private.binary_slot_tokens[frame] = (
        *private.binary_slot_tokens[frame],
        object(),
    )
    private.frame_widths[frame] = (
        n_cont + cont_growth,
        n_bin + 1,
    )
    private.allocation_history.append(
        _FactorAllocationRecord(
            "RELU",
            frame,
            (layer, unit),
            slots,
            n_cont + cont_growth,
            n_bin + 1,
        )
    )


def _record_private_runtime_factor_aux_allocation(
    allocator: object, frame_id: int, layer_id: int, count: int
) -> None:
    private = _require_open_factor_allocator(allocator)
    frame = _exact_int(frame_id, "runtime_factor_aux_frame")
    layer = _exact_int(layer_id, "runtime_factor_aux_layer")
    size = _exact_int(count, "runtime_factor_aux_count", minimum=1)
    key = (frame, layer)
    if frame not in private.frame_roots or key in private.aux_slots:
        raise RuntimeLineageReject("runtime_factor_aux_allocation_invalid")
    n_cont, n_bin = private.frame_widths[frame]
    slots = tuple(range(n_cont, n_cont + size))
    private.aux_slots[key] = slots
    private.continuous_slot_tokens[frame] = (
        *private.continuous_slot_tokens[frame],
        *(object() for _ in range(size)),
    )
    private.frame_widths[frame] = (n_cont + size, n_bin)
    private.allocation_history.append(
        _FactorAllocationRecord(
            "AUX", frame, (layer,), slots, n_cont + size, n_bin
        )
    )


def _record_private_runtime_factor_rebase(
    allocator: object,
    layer_id: int,
    before_source: object,
    after_source: object,
) -> None:
    """Record the exact frontier rebase allocation/ledger-clear event."""

    private = _require_open_factor_allocator(allocator)
    layer = _exact_int(layer_id, "runtime_factor_rebase_layer")
    before_frame, before_cont, before_bin = _factor_source_dimensions(
        before_source
    )
    after_frame, after_cont, after_bin = _factor_source_dimensions(
        after_source
    )
    if (
        before_frame != after_frame
        or private.frame_widths.get(before_frame)
        != (before_cont, before_bin)
        or after_cont <= before_cont
        or after_bin != before_bin
        or not any(
            prefix.source is before_source
            for prefix in private.source_prefixes
        )
    ):
        raise RuntimeLineageReject("runtime_factor_rebase_prefix_mismatch")
    added = after_cont - before_cont
    slots = tuple(range(before_cont, after_cont))
    private.continuous_slot_tokens[before_frame] = (
        *private.continuous_slot_tokens[before_frame],
        *(object() for _ in range(added)),
    )
    private.frame_widths[before_frame] = (after_cont, after_bin)
    private.relu_slots = {
        key: value
        for key, value in private.relu_slots.items()
        if key[0] != before_frame
    }
    private.aux_slots = {
        key: value
        for key, value in private.aux_slots.items()
        if key[0] != before_frame
    }
    private.allocation_history.append(
        _FactorAllocationRecord(
            "REBASE",
            before_frame,
            (layer,),
            slots,
            after_cont,
            after_bin,
        )
    )
    _append_factor_source_prefix(private, after_source)


def _factor_allocation_record_fields(
    record: object,
) -> tuple[str, int, tuple[int, ...], tuple[int, ...], int, int]:
    if type(record) is not _FactorAllocationRecord:
        raise RuntimeLineageReject("runtime_factor_allocation_record_type")
    kind = _exact_str(record.kind, "runtime_factor_allocation_kind")
    if kind not in {"ROOT", "RELU", "AUX", "REBASE"}:
        raise RuntimeLineageReject("runtime_factor_allocation_kind")
    frame_id = _exact_int(record.frame_id, "runtime_factor_record_frame")
    key = _exact_int_tuple(
        record.key,
        "runtime_factor_record_key",
        unique=False,
    )
    slots = _exact_int_tuple(
        record.slots,
        "runtime_factor_record_slots",
        unique=False,
    )
    n_cont = _exact_int(record.n_cont, "runtime_factor_record_n_cont")
    n_bin = _exact_int(record.n_bin, "runtime_factor_record_n_bin")
    return kind, frame_id, key, slots, n_cont, n_bin


def _factor_source_prefix_fields(
    prefix: object,
) -> tuple[object, int, object, tuple[object, ...], tuple[object, ...]]:
    if type(prefix) is not _FactorSourcePrefix:
        raise RuntimeLineageReject("runtime_factor_source_prefix_type")
    frame_id = _exact_int(prefix.frame_id, "runtime_factor_prefix_frame")
    continuous = prefix.continuous_slots
    binary = prefix.binary_slots
    if type(continuous) is not tuple or type(binary) is not tuple:
        raise RuntimeLineageReject("runtime_factor_prefix_slot_container_type")
    return prefix.source, frame_id, prefix.frame_root, continuous, binary


def _factor_allocator_guard(
    allocator: object, *, require_sealed: bool
) -> tuple[bytes, tuple[object, ...]]:
    if type(require_sealed) is not bool:
        raise RuntimeLineageReject("runtime_factor_required_state_not_bool")
    if (
        type(allocator) is not _RuntimeFactorAllocator
        or allocator.marker is not _ARENA_MARKER
        or type(allocator.owner_cache) is not dict
        or allocator.owner_cache is not allocator.owner_cache_identity
    ):
        raise RuntimeLineageReject("runtime_factor_allocator_state")
    if _exact_bool(
        allocator.sealed, "runtime_factor_allocator_sealed"
    ) is not require_sealed:
        raise RuntimeLineageReject("runtime_factor_allocator_state")
    next_frame_id = _exact_int(
        allocator.next_frame_id, "runtime_factor_next_frame"
    )
    if any(type(key) is not int or key < 0 for key in allocator.owner_cache):
        raise RuntimeLineageReject("runtime_factor_owner_cache_key_type")
    container_type = tuple if require_sealed else list
    if (
        type(allocator.allocation_history) is not container_type
        or type(allocator.source_prefixes) is not container_type
    ):
        raise RuntimeLineageReject("runtime_factor_allocator_container_state")
    dictionaries = (
        allocator.frame_roots,
        allocator.initial_widths,
        allocator.frame_widths,
        allocator.continuous_slot_tokens,
        allocator.binary_slot_tokens,
        allocator.relu_slots,
        allocator.aux_slots,
    )
    if any(type(value) is not dict for value in dictionaries):
        raise RuntimeLineageReject("runtime_factor_allocator_ledger_type")
    if any(type(key) is not int or key < 0 for key in allocator.frame_roots):
        raise RuntimeLineageReject("runtime_factor_frame_key_type")
    frame_ids = set(allocator.frame_roots)
    for name, mapping in (
        ("initial_widths", allocator.initial_widths),
        ("frame_widths", allocator.frame_widths),
    ):
        if any(type(key) is not int or key < 0 for key in mapping):
            raise RuntimeLineageReject(f"runtime_factor_{name}_key_type")
        for width in mapping.values():
            values = _exact_int_tuple(
                width, f"runtime_factor_{name}_value", unique=False
            )
            if len(values) != 2:
                raise RuntimeLineageReject(
                    f"runtime_factor_{name}_value_arity"
                )
    for name, mapping in (
        ("continuous_slots", allocator.continuous_slot_tokens),
        ("binary_slots", allocator.binary_slot_tokens),
    ):
        if any(type(key) is not int or key < 0 for key in mapping):
            raise RuntimeLineageReject(f"runtime_factor_{name}_key_type")
        if any(type(value) is not tuple for value in mapping.values()):
            raise RuntimeLineageReject(
                f"runtime_factor_{name}_container_type"
            )
    for key, value in allocator.relu_slots.items():
        if (
            len(_exact_int_tuple(
                key, "runtime_factor_relu_slot_key", unique=False
            ))
            != 3
            or len(_exact_int_tuple(
                value, "runtime_factor_relu_slot_value", unique=False
            ))
            != 3
        ):
            raise RuntimeLineageReject("runtime_factor_relu_slot_schema")
    for key, value in allocator.aux_slots.items():
        if (
            len(_exact_int_tuple(
                key, "runtime_factor_aux_slot_key", unique=False
            ))
            != 2
            or not _exact_int_tuple(
                value, "runtime_factor_aux_slot_value"
            )
        ):
            raise RuntimeLineageReject("runtime_factor_aux_slot_schema")
    if not frame_ids or any(
        set(mapping) != frame_ids
        for mapping in (
            allocator.initial_widths,
            allocator.frame_widths,
            allocator.continuous_slot_tokens,
            allocator.binary_slot_tokens,
        )
    ):
        raise RuntimeLineageReject("runtime_factor_frame_ledger_incomplete")
    if next_frame_id <= max(frame_ids):
        raise RuntimeLineageReject("runtime_factor_next_frame_invalid")
    records: list[object] = [
        next_frame_id,
        tuple(sorted(frame_ids)),
    ]
    refs: list[object] = [
        allocator,
        allocator.request_owner,
        allocator.owner_cache,
        allocator.owner_cache_identity,
        *dictionaries,
        allocator.allocation_history,
        allocator.source_prefixes,
    ]
    replay: dict[int, tuple[int, int]] = {}
    expected_relu_slots: dict[
        tuple[int, int, int], tuple[int, int, int]
    ] = {}
    expected_aux_slots: dict[tuple[int, int], tuple[int, ...]] = {}
    for record in allocator.allocation_history:
        kind, frame_id, key, slots, n_cont, n_bin = (
            _factor_allocation_record_fields(record)
        )
        refs.extend((record, record.key, record.slots))
        if kind == "ROOT":
            if frame_id in replay or key or slots:
                raise RuntimeLineageReject("runtime_factor_root_history_invalid")
            if allocator.initial_widths.get(frame_id) != (
                n_cont,
                n_bin,
            ):
                raise RuntimeLineageReject("runtime_factor_root_width_mismatch")
            replay[frame_id] = (n_cont, n_bin)
        elif kind == "RELU":
            if frame_id not in replay or len(key) != 2:
                raise RuntimeLineageReject("runtime_factor_relu_history_invalid")
            prior_cont, prior_bin = replay[frame_id]
            expected_slots = (
                (prior_cont, prior_cont, prior_bin)
                if slots[:2] == (prior_cont, prior_cont)
                else (prior_cont, prior_cont + 1, prior_bin)
            )
            if slots != expected_slots:
                raise RuntimeLineageReject("runtime_factor_relu_slot_mismatch")
            expected_relu_slots[(frame_id, *key)] = slots
            growth = 1 if slots[0] == slots[1] else 2
            replay[frame_id] = (prior_cont + growth, prior_bin + 1)
        elif kind == "AUX":
            if frame_id not in replay or len(key) != 1:
                raise RuntimeLineageReject("runtime_factor_aux_history_invalid")
            prior_cont, prior_bin = replay[frame_id]
            expected_slots = tuple(
                range(prior_cont, prior_cont + len(slots))
            )
            if not slots or slots != expected_slots:
                raise RuntimeLineageReject("runtime_factor_aux_slot_mismatch")
            expected_aux_slots[(frame_id, *key)] = slots
            replay[frame_id] = (
                prior_cont + len(slots),
                prior_bin,
            )
        elif kind == "REBASE":
            if frame_id not in replay or len(key) != 1:
                raise RuntimeLineageReject("runtime_factor_rebase_history_invalid")
            prior_cont, prior_bin = replay[frame_id]
            expected_slots = tuple(
                range(prior_cont, prior_cont + len(slots))
            )
            if not slots or slots != expected_slots:
                raise RuntimeLineageReject("runtime_factor_rebase_slot_mismatch")
            replay[frame_id] = (
                prior_cont + len(slots),
                prior_bin,
            )
            expected_relu_slots = {
                active_key: value
                for active_key, value in expected_relu_slots.items()
                if active_key[0] != frame_id
            }
            expected_aux_slots = {
                active_key: value
                for active_key, value in expected_aux_slots.items()
                if active_key[0] != frame_id
            }
        else:
            raise RuntimeLineageReject("runtime_factor_allocation_kind")
        if replay[frame_id] != (n_cont, n_bin):
            raise RuntimeLineageReject("runtime_factor_history_width_mismatch")
        records.append(
            (
                kind,
                frame_id,
                key,
                slots,
                n_cont,
                n_bin,
            )
        )
    if replay != allocator.frame_widths:
        raise RuntimeLineageReject("runtime_factor_frame_width_history_mismatch")
    if (
        allocator.relu_slots != expected_relu_slots
        or allocator.aux_slots != expected_aux_slots
    ):
        raise RuntimeLineageReject("runtime_factor_active_slot_ledger_mismatch")
    for frame_id in sorted(frame_ids):
        root = allocator.frame_roots[frame_id]
        initial = _exact_int_tuple(
            allocator.initial_widths[frame_id],
            "runtime_factor_initial_width",
            unique=False,
        )
        current = _exact_int_tuple(
            allocator.frame_widths[frame_id],
            "runtime_factor_current_width",
            unique=False,
        )
        continuous = allocator.continuous_slot_tokens[frame_id]
        binary = allocator.binary_slot_tokens[frame_id]
        if (
            len(continuous) != current[0]
            or len(binary) != current[1]
            or len({id(token) for token in (*continuous, *binary)})
            != len(continuous) + len(binary)
        ):
            raise RuntimeLineageReject("runtime_factor_slot_token_mismatch")
        refs.extend(
            (
                root,
                initial,
                current,
                continuous,
                binary,
                *continuous,
                *binary,
            )
        )
        records.append((frame_id, initial, current))
    cache_sources = tuple(allocator.owner_cache.values())
    if len({id(source) for source in cache_sources}) != len(cache_sources):
        raise RuntimeLineageReject("runtime_factor_owner_cache_source_alias")
    prefixes = tuple(allocator.source_prefixes)
    if len(prefixes) != len(cache_sources) or any(
        not any(prefix.source is source for prefix in prefixes)
        for source in cache_sources
    ):
        raise RuntimeLineageReject("runtime_factor_source_ledger_incomplete")
    for prefix in prefixes:
        (
            prefix_source,
            prefix_frame,
            prefix_root,
            prefix_continuous,
            prefix_binary,
        ) = _factor_source_prefix_fields(prefix)
        frame_id, n_cont, n_bin = _factor_source_dimensions(prefix_source)
        full_continuous = allocator.continuous_slot_tokens.get(frame_id, ())
        full_binary = allocator.binary_slot_tokens.get(frame_id, ())
        if (
            prefix_frame != frame_id
            or prefix_root is not allocator.frame_roots.get(frame_id)
            or len(prefix_continuous) != n_cont
            or len(prefix_binary) != n_bin
            or not _identity_tuple_equal(
                prefix_continuous, full_continuous[:n_cont]
            )
            or not _identity_tuple_equal(
                prefix_binary, full_binary[:n_bin]
            )
        ):
            raise RuntimeLineageReject("runtime_factor_source_prefix_mismatch")
        refs.extend(
            (
                prefix,
                prefix_source,
                prefix_root,
                prefix_continuous,
                prefix_binary,
                *prefix_continuous,
                *prefix_binary,
            )
        )
        records.append((frame_id, n_cont, n_bin))
    return _metadata_payload(records), tuple(refs)


def _seal_private_runtime_factor_allocator(
    allocator: object,
) -> _RuntimeFactorAllocator:
    private = _require_open_factor_allocator(allocator)
    _factor_allocator_guard(private, require_sealed=False)
    private.allocation_history = tuple(private.allocation_history)
    private.source_prefixes = tuple(private.source_prefixes)
    private.sealed = True
    private.sealed_payload, private.sealed_references = (
        _factor_allocator_guard(private, require_sealed=True)
    )
    return private


def _validate_factor_allocator_current(
    allocator: object,
) -> _RuntimeFactorAllocator:
    payload, references = _factor_allocator_guard(
        allocator, require_sealed=True
    )
    sealed_payload = _exact_bytes(
        allocator.sealed_payload, "runtime_factor_sealed_payload"
    )
    if type(allocator.sealed_references) is not tuple:
        raise RuntimeLineageReject(
            "runtime_factor_sealed_references_not_tuple"
        )
    if (
        payload != sealed_payload
        or not _identity_tuple_equal(
            references, allocator.sealed_references
        )
    ):
        raise RuntimeLineageReject("runtime_factor_allocator_cas_mismatch")
    return allocator


def _prune_source_arena_owners(_reference: object | None = None) -> None:
    del _reference
    _SOURCE_ARENA_OWNERS[:] = [
        entry
        for entry in _SOURCE_ARENA_OWNERS
        if entry[0]() is not None and entry[1]() is not None
    ]


def _active_source_arena_owner(source: object) -> object | None:
    _prune_source_arena_owners()
    matches = tuple(
        nonce
        for source_reference, arena_reference, nonce in _SOURCE_ARENA_OWNERS
        if source_reference() is source and arena_reference() is not None
    )
    if len(matches) > 1:
        raise RuntimeLineageReject("runtime_source_owner_ledger_ambiguous")
    return matches[0] if matches else None


def _register_source_arena_owner(
    source: object, arena: _RuntimeLineageArena
) -> None:
    try:
        source_reference = weakref.ref(source, _prune_source_arena_owners)
        arena_reference = weakref.ref(arena, _prune_source_arena_owners)
    except TypeError as exc:
        raise RuntimeLineageReject(
            "runtime_source_or_arena_not_weak_referenceable"
        ) from exc
    _SOURCE_ARENA_OWNERS.append(
        (source_reference, arena_reference, arena.nonce)
    )


def _begin_private_runtime_lineage_arena(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    owner_cache: Mapping[int, object],
    owner_affine_cache: Mapping[int, object],
    factor_allocator: object,
) -> _RuntimeLineageArena:
    """Create a one-request capability; raw mappings are never registries."""

    graph = _freeze_graph(layers, preds, succs)
    private_allocator = _validate_factor_allocator_current(factor_allocator)
    if (
        type(owner_cache) is not dict
        or type(owner_affine_cache) is not dict
        or owner_affine_cache
        or private_allocator.owner_cache is not owner_cache
    ):
        raise RuntimeLineageReject("runtime_owner_cache_missing")
    return _RuntimeLineageArena(
        graph,
        owner_cache,
        owner_affine_cache,
        private_allocator,
        _ARENA_MARKER,
    )


def _validate_arena_current(
    arena: object,
    graph: _GraphSnapshot,
    *,
    require_closed: bool,
    require_complete_cache: bool,
) -> _RuntimeLineageArena:
    if (
        type(arena) is not _RuntimeLineageArena
        or arena.marker is not _ARENA_MARKER
    ):
        raise RuntimeLineageReject("runtime_private_arena_missing")
    if type(require_closed) is not bool or type(require_complete_cache) is not bool:
        raise RuntimeLineageReject("runtime_arena_required_state_not_bool")
    if _exact_bool(arena.closed, "runtime_arena_closed") is not require_closed:
        state = "closed" if require_closed else "open"
        raise RuntimeLineageReject(f"runtime_arena_not_{state}")
    arena_graph_sha = _exact_sha256(
        arena.graph_sha256, "runtime_arena_graph_sha256"
    )
    graph_sha = _exact_sha256(graph.graph_sha256, "runtime_graph_sha256")
    if (
        type(arena.graph_reference_objects) is not tuple
        or type(graph.reference_objects) is not tuple
    ):
        raise RuntimeLineageReject("runtime_arena_graph_reference_state")
    if arena_graph_sha != graph_sha or not _identity_tuple_equal(
        arena.graph_reference_objects, graph.reference_objects
    ):
        raise RuntimeLineageReject("runtime_arena_graph_cas_mismatch")
    if (
        type(arena.owner_cache) is not dict
        or arena.owner_cache is not arena.owner_cache_identity
        or type(arena.owner_affine_cache) is not dict
        or arena.owner_affine_cache is not arena.owner_affine_cache_identity
    ):
        raise RuntimeLineageReject("runtime_owner_cache_cas_mismatch")
    if any(
        type(key) is not int or key < 0
        for key in (*arena.owner_cache.keys(), *arena.owner_affine_cache.keys())
    ):
        raise RuntimeLineageReject("runtime_owner_cache_key_type")
    if (
        type(arena.factor_allocator) is not _RuntimeFactorAllocator
        or arena.factor_allocator is not arena.factor_allocator_identity
        or arena.factor_allocator.owner_cache is not arena.owner_cache
    ):
        raise RuntimeLineageReject("runtime_factor_allocator_identity_mismatch")
    factor_payload = _exact_bytes(
        arena.factor_allocator_payload,
        "runtime_arena_factor_allocator_payload",
    )
    if type(arena.factor_allocator_references) is not tuple:
        raise RuntimeLineageReject(
            "runtime_arena_factor_allocator_references_type"
        )
    current_factor_allocator = _validate_factor_allocator_current(
        arena.factor_allocator
    )
    current_factor_payload, current_factor_references = (
        _factor_allocator_guard(
            current_factor_allocator, require_sealed=True
        )
    )
    if (
        current_factor_payload != factor_payload
        or not _identity_tuple_equal(
            current_factor_references,
            arena.factor_allocator_references,
        )
    ):
        raise RuntimeLineageReject("runtime_factor_allocator_arena_cas_mismatch")
    if (
        type(arena.affine_entries) is not list
        or type(arena.affine_payloads) is not list
        or type(arena.affine_reference_snapshots) is not list
    ):
        raise RuntimeLineageReject("runtime_affine_cache_ledger_state")
    if not (
        len(arena.affine_entries)
        == len(arena.affine_payloads)
        == len(arena.affine_reference_snapshots)
    ):
        raise RuntimeLineageReject("runtime_affine_cache_ledger_cas_mismatch")
    if any(
        type(entry) is not tuple or len(entry) != 2
        for entry in arena.affine_entries
    ):
        raise RuntimeLineageReject("runtime_affine_cache_ledger_entry_type")
    if any(
        type(entry[0]) is not int or entry[0] < 0
        for entry in arena.affine_entries
    ):
        raise RuntimeLineageReject("runtime_affine_cache_ledger_key_type")
    if any(type(payload) is not bytes for payload in arena.affine_payloads):
        raise RuntimeLineageReject("runtime_affine_cache_payload_type")
    if any(
        type(references) is not tuple
        for references in arena.affine_reference_snapshots
    ):
        raise RuntimeLineageReject("runtime_affine_cache_references_type")
    affine_keys = tuple(layer_id for layer_id, _ in arena.affine_entries)
    if len(affine_keys) != len(set(affine_keys)):
        raise RuntimeLineageReject("runtime_affine_cache_ledger_duplicate")
    if set(arena.owner_affine_cache.keys()) != set(affine_keys):
        raise RuntimeLineageReject("runtime_affine_cache_ledger_incomplete")
    for entry_index, (layer_id, cache_entry) in enumerate(
        arena.affine_entries
    ):
        if (
            type(layer_id) is not int
            or type(cache_entry) is not _RuntimeAffineExprCacheView
            or arena.owner_affine_cache.get(layer_id) is not cache_entry
        ):
            raise RuntimeLineageReject("runtime_affine_cache_ledger_cas_mismatch")
        current_payload, current_references = _affine_cache_entry_guard(
            cache_entry, graph
        )
        if (
            current_payload != arena.affine_payloads[entry_index]
            or not _identity_tuple_equal(
                current_references,
                arena.affine_reference_snapshots[entry_index],
            )
        ):
            raise RuntimeLineageReject(
                "runtime_affine_cache_entry_snapshot_mismatch"
            )
    expected_container_type = tuple if require_closed else list
    if (
        type(arena.source_entries) is not expected_container_type
        or type(arena.source_payloads) is not expected_container_type
        or type(arena.source_reference_snapshots) is not expected_container_type
    ):
        raise RuntimeLineageReject("runtime_arena_source_ledger_state")
    if require_closed and arena.source_entries is not arena.sealed_source_entries:
        raise RuntimeLineageReject("runtime_arena_source_ledger_cas_mismatch")
    if not require_closed and arena.sealed_source_entries is not None:
        raise RuntimeLineageReject("runtime_arena_source_ledger_state")
    if require_closed and type(arena.sealed_source_entries) is not tuple:
        raise RuntimeLineageReject("runtime_arena_sealed_source_ledger_type")
    entries = tuple(arena.source_entries)
    payloads = tuple(arena.source_payloads)
    reference_snapshots = tuple(arena.source_reference_snapshots)
    if not (
        len(entries) == len(payloads) == len(reference_snapshots)
    ):
        raise RuntimeLineageReject("runtime_arena_source_ledger_cas_mismatch")
    if any(type(entry) is not tuple or len(entry) != 2 for entry in entries):
        raise RuntimeLineageReject("runtime_arena_source_entry_type")
    if any(type(entry[0]) is not int or entry[0] < 0 for entry in entries):
        raise RuntimeLineageReject("runtime_arena_source_key_type")
    if any(type(payload) is not bytes for payload in payloads):
        raise RuntimeLineageReject("runtime_arena_source_payload_type")
    if any(type(references) is not tuple for references in reference_snapshots):
        raise RuntimeLineageReject("runtime_arena_source_references_type")
    if not entries:
        raise RuntimeLineageReject("runtime_arena_source_ledger_empty")
    keys = tuple(layer_id for layer_id, _ in entries)
    if len(keys) != len(set(keys)):
        raise RuntimeLineageReject("runtime_arena_source_ledger_duplicate")
    if require_complete_cache and set(arena.owner_cache.keys()) != set(keys):
        raise RuntimeLineageReject("runtime_owner_cache_not_fully_recorded")
    frame_payload: bytes | None = None
    if arena.frame_payload is not None:
        _exact_bytes(arena.frame_payload, "runtime_arena_frame_payload")
    seen_sources: list[object] = []
    for source_index, (layer_id, source) in enumerate(entries):
        if arena.owner_cache.get(layer_id) is not source:
            raise RuntimeLineageReject("runtime_owner_cache_cas_mismatch")
        if any(source is prior for prior in seen_sources):
            raise RuntimeLineageReject("runtime_source_object_aliases_boundaries")
        seen_sources.append(source)
        if layer_id >= len(graph.layers):
            raise RuntimeLineageReject("runtime_source_boundary_unknown_layer")
        if graph.layers[layer_id].kind not in _AUTHORIZED_SOURCE_LAYER_KINDS:
            raise RuntimeLineageReject("runtime_source_boundary_kind_not_authorized")
        current_payload, _ = _source_boundary_payload(
            source, graph.layers[layer_id]
        )
        _, current_references = _snapshot_sparse_hz_source(source)
        current_references = (*current_references, source.frame_id)
        if (
            current_payload != payloads[source_index]
            or not _identity_tuple_equal(
                current_references, reference_snapshots[source_index]
            )
        ):
            raise RuntimeLineageReject(
                "runtime_source_arena_snapshot_mismatch"
            )
        current_frame = _stable_token_payload(
            source.frame_id, "runtime_source_frame"
        )
        if frame_payload is None:
            frame_payload = current_frame
        elif frame_payload != current_frame:
            raise RuntimeLineageReject("runtime_arena_mixed_factor_frames")
        owner = _active_source_arena_owner(source)
        if owner is not arena.nonce:
            raise RuntimeLineageReject("runtime_source_cross_arena_custody")
    if frame_payload != arena.frame_payload:
        raise RuntimeLineageReject("runtime_arena_frame_cas_mismatch")
    return arena


def _record_private_runtime_source_boundary(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    arena: object,
    layer_id: int,
    source: object,
) -> None:
    graph = _freeze_graph(layers, preds, succs)
    if (
        type(arena) is not _RuntimeLineageArena
        or arena.marker is not _ARENA_MARKER
        or arena.closed is not False
        or type(arena.source_entries) is not list
    ):
        raise RuntimeLineageReject("runtime_private_open_arena_missing")
    if (
        _exact_sha256(
            arena.graph_sha256, "runtime_arena_graph_sha256"
        )
        != _exact_sha256(graph.graph_sha256, "runtime_graph_sha256")
        or not _identity_tuple_equal(
        arena.graph_reference_objects, graph.reference_objects
        )
    ):
        raise RuntimeLineageReject("runtime_arena_graph_cas_mismatch")
    boundary_id = _exact_int(layer_id, "runtime_source_boundary_layer")
    if boundary_id >= len(graph.layers):
        raise RuntimeLineageReject("runtime_source_boundary_unknown_layer")
    if graph.layers[boundary_id].kind not in _AUTHORIZED_SOURCE_LAYER_KINDS:
        raise RuntimeLineageReject("runtime_source_boundary_kind_not_authorized")
    if type(arena.owner_cache) is not dict or arena.owner_cache.get(
        boundary_id
    ) is not source:
        raise RuntimeLineageReject("runtime_source_not_owned_by_current_cache")
    allocator = _validate_factor_allocator_current(arena.factor_allocator)
    matching_prefixes = tuple(
        prefix
        for prefix in allocator.source_prefixes
        if prefix.source is source
    )
    if len(matching_prefixes) != 1:
        raise RuntimeLineageReject(
            "runtime_source_factor_prefix_authority_missing"
        )
    factor_prefix = matching_prefixes[0]
    source_frame, source_n_cont, source_n_bin = _factor_source_dimensions(
        source
    )
    if (
        factor_prefix.frame_id != source_frame
        or len(factor_prefix.continuous_slots) != source_n_cont
        or len(factor_prefix.binary_slots) != source_n_bin
        or factor_prefix.frame_root
        is not allocator.frame_roots.get(source_frame)
    ):
        raise RuntimeLineageReject("runtime_source_factor_prefix_mismatch")
    if any(boundary_id == prior for prior, _ in arena.source_entries):
        raise RuntimeLineageReject("runtime_source_boundary_duplicate_layer")
    if any(source is prior for _, prior in arena.source_entries):
        raise RuntimeLineageReject("runtime_source_object_aliases_boundaries")
    source_payload, _ = _source_boundary_payload(
        source, graph.layers[boundary_id]
    )
    _, source_references = _snapshot_sparse_hz_source(source)
    frame_payload = _stable_token_payload(source.frame_id, "runtime_source_frame")
    if arena.frame_payload is not None and arena.frame_payload != frame_payload:
        raise RuntimeLineageReject("runtime_arena_mixed_factor_frames")
    owner_nonce = _active_source_arena_owner(source)
    if owner_nonce is not None and owner_nonce is not arena.nonce:
        raise RuntimeLineageReject("runtime_source_cross_arena_custody")
    if owner_nonce is None:
        _register_source_arena_owner(source, arena)
    arena.frame_payload = frame_payload
    arena.source_entries.append((boundary_id, source))
    arena.source_payloads.append(source_payload)
    arena.source_reference_snapshots.append(
        (*source_references, source.frame_id)
    )


def _begin_private_runtime_lineage_registry(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    arena: object,
) -> _RuntimeLineageRegistry:
    """Consume an arena once; a caller-owned raw source map is rejected."""

    graph = _freeze_graph(layers, preds, succs)
    if type(arena) is not _RuntimeLineageArena or arena.marker is not _ARENA_MARKER:
        raise RuntimeLineageReject("runtime_private_arena_missing")
    was_closed = _exact_bool(arena.closed, "runtime_arena_closed")
    private_arena = _validate_arena_current(
        arena,
        graph,
        require_closed=was_closed,
        require_complete_cache=True,
    )
    if not was_closed:
        ordered_sources = sorted(
            zip(
                private_arena.source_entries,
                private_arena.source_payloads,
                private_arena.source_reference_snapshots,
                strict=True,
            ),
            key=lambda item: item[0][0],
        )
        private_arena.source_entries = tuple(
            entry for entry, _, _ in ordered_sources
        )
        private_arena.source_payloads = tuple(
            payload for _, payload, _ in ordered_sources
        )
        private_arena.source_reference_snapshots = tuple(
            references for _, _, references in ordered_sources
        )
        private_arena.sealed_source_entries = private_arena.source_entries
        private_arena.closed = True
    source_boundaries = _freeze_source_boundaries(
        private_arena.owner_cache, graph
    )
    if not all(
        left_id == right_id and left_source is right_source
        for (left_id, left_source), (right_id, right_source) in zip(
            private_arena.source_entries,
            source_boundaries.by_layer,
            strict=True,
        )
    ):
        raise RuntimeLineageReject("runtime_arena_source_ledger_cas_mismatch")
    return _RuntimeLineageRegistry(
        graph, source_boundaries, private_arena, _CUSTODY_MARKER
    )


def _validate_registry_current(
    registry: object, graph: _GraphSnapshot, *, require_sealed: bool
) -> _RuntimeLineageRegistry:
    if (
        type(registry) is not _RuntimeLineageRegistry
        or registry.marker is not _CUSTODY_MARKER
    ):
        raise RuntimeLineageReject("runtime_private_registry_missing")
    if type(require_sealed) is not bool:
        raise RuntimeLineageReject("runtime_registry_required_state_not_bool")
    if _exact_bool(
        registry.sealed, "runtime_registry_sealed"
    ) is not require_sealed:
        expected = "sealed" if require_sealed else "open"
        raise RuntimeLineageReject(f"runtime_registry_not_{expected}")
    registry_graph_sha = _exact_sha256(
        registry.graph_sha256, "runtime_registry_graph_sha256"
    )
    graph_sha = _exact_sha256(graph.graph_sha256, "runtime_graph_sha256")
    if (
        type(registry.graph_reference_objects) is not tuple
        or type(graph.reference_objects) is not tuple
    ):
        raise RuntimeLineageReject("runtime_registry_graph_reference_state")
    if registry_graph_sha != graph_sha or not _identity_tuple_equal(
        registry.graph_reference_objects, graph.reference_objects
    ):
        raise RuntimeLineageReject("runtime_registry_graph_cas_mismatch")
    _validate_arena_current(
        registry.arena,
        graph,
        require_closed=True,
        require_complete_cache=True,
    )
    expected_snapshot_container = tuple if require_sealed else list
    if type(registry.add_operand_snapshots) is not expected_snapshot_container:
        raise RuntimeLineageReject("runtime_add_snapshot_container_type")
    source_boundaries = _source_boundary_snapshot_guard(
        registry.source_boundaries
    )
    mapping = source_boundaries.reference_objects[0]
    if type(mapping) is not dict or mapping is not registry.arena.owner_cache:
        raise RuntimeLineageReject("runtime_registry_source_map_cas_mismatch")
    if set(mapping.keys()) != {
        layer_id for layer_id, _ in source_boundaries.by_layer
    }:
        raise RuntimeLineageReject("runtime_registry_source_map_cas_mismatch")
    for layer_id, source in source_boundaries.by_layer:
        if mapping.get(layer_id) is not source:
            raise RuntimeLineageReject(
                "runtime_registry_source_map_cas_mismatch"
            )
        _source_boundary_payload(source, graph.layers[layer_id])
    for snapshot in registry.add_operand_snapshots:
        _validate_registered_add_operands_current(snapshot, graph, registry)
    if registry.terminal_expression_snapshot is not None:
        _registered_terminal_snapshot_schema(
            registry.terminal_expression_snapshot
        )
        _validate_registered_terminal_expression_current(
            registry.terminal_expression_snapshot, graph, registry
        )
    if require_sealed:
        payload, references = _registry_authority_guard(registry)
        sealed_payload = _exact_bytes(
            registry.sealed_authority_payload,
            "runtime_registry_sealed_authority_payload",
        )
        if type(registry.sealed_authority_references) is not tuple:
            raise RuntimeLineageReject(
                "runtime_registry_sealed_authority_references_type"
            )
        if (
            payload != sealed_payload
            or not _identity_tuple_equal(
                references, registry.sealed_authority_references
            )
        ):
            raise RuntimeLineageReject("runtime_registry_authority_cas_mismatch")
    elif (
        registry.sealed_authority_payload is not None
        or registry.sealed_authority_references is not None
    ):
        raise RuntimeLineageReject("runtime_registry_authority_state")
    return registry


def _seal_private_runtime_lineage_registry(
    registry: _RuntimeLineageRegistry,
) -> _RuntimeLineageRegistry:
    if (
        type(registry) is not _RuntimeLineageRegistry
        or registry.marker is not _CUSTODY_MARKER
        or registry.sealed is not False
        or type(registry.add_operand_snapshots) is not list
    ):
        raise RuntimeLineageReject("runtime_registry_cannot_seal")
    registry.add_operand_snapshots = tuple(registry.add_operand_snapshots)
    registry.sealed = True
    (
        registry.sealed_authority_payload,
        registry.sealed_authority_references,
    ) = _registry_authority_guard(registry)
    return registry


def _sparse_hz_source_raw_state(source: object) -> dict[str, object]:
    if type(source) is not SparseHZono:
        raise RuntimeLineageReject("runtime_source_type_not_sparse_hzono")
    try:
        state = object.__getattribute__(source, "__dict__")
    except Exception as exc:
        raise RuntimeLineageReject("runtime_source_raw_state_unavailable") from exc
    if type(state) is not dict or any(type(key) is not str for key in state):
        raise RuntimeLineageReject("runtime_source_raw_state_type")
    if set(state) != _SOURCE_STATE_KEYS:
        raise RuntimeLineageReject("runtime_source_raw_state_schema")
    if state["exact"] is not True:
        raise RuntimeLineageReject("runtime_source_not_exact")
    _exact_int(state["frame_id"], "runtime_source_frame")
    if any(type(state[name]) is not np.ndarray for name in ("c", "b", "ub")):
        raise RuntimeLineageReject("runtime_source_dense_field_type")
    if any(
        type(state[name]) is not scipy_sparse.csr_matrix
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")
    ):
        raise RuntimeLineageReject("runtime_source_sparse_field_not_csr")
    return state


def _compressed_sparse_raw_state(
    matrix: object, name: str
) -> tuple[
    dict[str, object],
    tuple[int, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if type(matrix) not in (scipy_sparse.csr_matrix, scipy_sparse.csc_matrix):
        raise RuntimeLineageReject(f"{name}_not_exact_compressed_sparse")
    try:
        state = object.__getattribute__(matrix, "__dict__")
    except Exception as exc:
        raise RuntimeLineageReject(f"{name}_raw_state_unavailable") from exc
    if type(state) is not dict or any(type(key) is not str for key in state):
        raise RuntimeLineageReject(f"{name}_raw_state_type")
    keys = set(state)
    if (
        not _COMPRESSED_SPARSE_REQUIRED_STATE_KEYS.issubset(keys)
        or not keys.issubset(
            _COMPRESSED_SPARSE_REQUIRED_STATE_KEYS
            | _COMPRESSED_SPARSE_OPTIONAL_STATE_KEYS
        )
    ):
        raise RuntimeLineageReject(f"{name}_raw_state_schema")
    raw_shape = _exact_int_tuple(
        state["_shape"], f"{name}_shape", unique=False
    )
    if len(raw_shape) != 2:
        raise RuntimeLineageReject(f"{name}_shape_arity")
    shape = (raw_shape[0], raw_shape[1])
    _exact_int(state["maxprint"], f"{name}_maxprint")
    for flag_name in _COMPRESSED_SPARSE_OPTIONAL_STATE_KEYS:
        if flag_name in state:
            _exact_bool(state[flag_name], f"{name}_{flag_name}")
    data = state["data"]
    indices = state["indices"]
    indptr = state["indptr"]
    if (
        type(data) is not np.ndarray
        or type(indices) is not np.ndarray
        or type(indptr) is not np.ndarray
    ):
        raise RuntimeLineageReject(f"{name}_buffers_not_ndarray")
    return state, shape, data, indices, indptr


def _snapshot_sparse_hz_source(
    source: object,
) -> tuple[bytes, tuple[object, ...]]:
    state = _sparse_hz_source_raw_state(source)
    matrices = tuple(
        state[name] for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")
    )
    c = state["c"]
    b = state["b"]
    ub = state["ub"]
    if c.ndim != 1 or b.ndim != 1 or ub.ndim != 1:
        raise RuntimeLineageReject("runtime_source_dense_field_rank")
    if any(array.dtype != np.dtype(np.float64) for array in (c, b, ub)):
        raise RuntimeLineageReject("runtime_source_dense_field_dtype")
    shapes: dict[str, tuple[int, int]] = {}
    for field_name, matrix in zip(
        ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"), matrices, strict=True
    ):
        shapes[field_name] = _validate_sparse_hz_csr(
            matrix, f"runtime_source_{field_name}"
        )
    n_out = int(c.size)
    n_cont = shapes["Gc"][1]
    n_bin = shapes["Gb"][1]
    if shapes["Gc"] != (n_out, n_cont) or shapes["Gb"] != (n_out, n_bin):
        raise RuntimeLineageReject("runtime_source_value_factor_shape")
    if shapes["Ac"] != (b.size, n_cont) or shapes["Ab"] != (b.size, n_bin):
        raise RuntimeLineageReject("runtime_source_equality_factor_shape")
    if shapes["Auc"] != (ub.size, n_cont) or shapes["Aub"] != (ub.size, n_bin):
        raise RuntimeLineageReject("runtime_source_inequality_factor_shape")

    payloads: list[bytes] = []
    refs: list[object] = [source, state]
    for field_name in _SOURCE_FIELDS:
        payload, field_refs = _object_guard(
            state[field_name], f"runtime_source_{field_name}"
        )
        payloads.append(payload)
        refs.extend(field_refs)
    header = _metadata_payload(
        [n_out, n_cont, n_bin, int(b.size), int(ub.size)]
    )
    snapshot = b"".join(
        len(part).to_bytes(8, "big") + part
        for part in (header, *payloads)
    )
    return snapshot, tuple(refs)


def _validate_sparse_hz_csr(
    matrix: scipy_sparse.csr_matrix, name: str
) -> tuple[int, int]:
    """Validate structure rather than trusting SciPy's mutable cached flags."""

    if type(matrix) is not scipy_sparse.csr_matrix:
        raise RuntimeLineageReject(f"{name}_not_exact_csr")
    _, shape, data, indices, indptr = _compressed_sparse_raw_state(
        matrix, name
    )
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise RuntimeLineageReject(f"{name}_csr_buffer_rank")
    if data.dtype != np.dtype(np.float64):
        raise RuntimeLineageReject(f"{name}_csr_data_dtype")
    if indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
        raise RuntimeLineageReject(f"{name}_csr_index_dtype")
    rows, columns = shape
    if rows < 0 or columns < 0 or indptr.size != rows + 1:
        raise RuntimeLineageReject(f"{name}_csr_indptr_shape")
    if indptr.size == 0 or int(indptr[0]) != 0:
        raise RuntimeLineageReject(f"{name}_csr_indptr_origin")
    if np.any(indptr[1:] < indptr[:-1]):
        raise RuntimeLineageReject(f"{name}_csr_indptr_not_monotone")
    nnz = int(indptr[-1])
    if nnz < 0 or data.size != nnz or indices.size != nnz:
        raise RuntimeLineageReject(f"{name}_csr_buffer_length")
    if data.size and not np.all(np.isfinite(data)):
        raise RuntimeLineageReject(f"{name}_csr_data_nonfinite")
    if data.size and np.any(data == 0.0):
        raise RuntimeLineageReject(f"{name}_csr_explicit_zero")
    if indices.size and (np.any(indices < 0) or np.any(indices >= columns)):
        raise RuntimeLineageReject(f"{name}_csr_index_out_of_bounds")
    for row in range(rows):
        start = int(indptr[row])
        stop = int(indptr[row + 1])
        if stop - start > 1 and np.any(
            indices[start + 1 : stop] <= indices[start : stop - 1]
        ):
            raise RuntimeLineageReject(f"{name}_csr_not_canonical")
    return rows, columns


def _source_boundary_payload(
    source: object,
    entry: _LayerSnapshot,
    *,
    expression_frame_payload: bytes | None = None,
) -> tuple[bytes, int]:
    source_snapshot, _ = _snapshot_sparse_hz_source(source)
    state = _sparse_hz_source_raw_state(source)
    frame_id = _exact_int(state["frame_id"], "runtime_source_frame")
    frame_payload = _stable_token_payload(
        frame_id, "runtime_source_frame"
    )
    if (
        expression_frame_payload is not None
        and frame_payload != expression_frame_payload
    ):
        raise RuntimeLineageReject("runtime_source_expression_frame_mismatch")
    n_out = _exact_int(int(state["c"].size), "runtime_source_n_out")
    if n_out != len(entry.out_vars):
        raise RuntimeLineageReject("runtime_source_entry_width_mismatch")
    payload = b"".join(
        (
            b"exact:true",
            len(frame_payload).to_bytes(8, "big"),
            frame_payload,
            n_out.to_bytes(8, "big"),
            len(source_snapshot).to_bytes(8, "big"),
            source_snapshot,
        )
    )
    return payload, n_out


_DIAGONAL_OPERATOR_STATE_KEYS = frozenset({"_diagonal", "_content_key"})
_CONV_OPERATOR_STATE_KEYS = frozenset(
    {
        "_kernel",
        "_input_shape",
        "_output_shape",
        "_stride",
        "_padding",
        "_dilation",
        "_groups",
        "_row_mask",
        "_logical_expanded_nnz",
        "_content_key",
    }
)


def _operator_raw_state(
    operator_value: object, *, expected_keys: frozenset[str], name: str
) -> dict[str, object]:
    try:
        state = object.__getattribute__(operator_value, "__dict__")
    except Exception as exc:
        raise RuntimeLineageReject(f"{name}_raw_state_unavailable") from exc
    if type(state) is not dict or set(state) != expected_keys:
        raise RuntimeLineageReject(f"{name}_raw_state_schema")
    return state


def _pure_conv_logical_nnz(
    *,
    kernel_shape: tuple[int, int, int, int],
    input_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    row_mask: np.ndarray | None,
) -> int:
    batch, _, height, width = input_shape
    out_channels, in_per_group, kernel_height, kernel_width = kernel_shape
    _, _, output_height, output_width = output_shape
    stencil_counts: list[int] = []
    for output_y in range(output_height):
        for output_x in range(output_width):
            valid = 0
            for kernel_y in range(kernel_height):
                input_y = (
                    output_y * stride[0]
                    - padding[0]
                    + kernel_y * dilation[0]
                )
                if input_y < 0 or input_y >= height:
                    continue
                for kernel_x in range(kernel_width):
                    input_x = (
                        output_x * stride[1]
                        - padding[1]
                        + kernel_x * dilation[1]
                    )
                    if 0 <= input_x < width:
                        valid += 1
            stencil_counts.append(valid * in_per_group)
    if row_mask is None:
        result = batch * out_channels * sum(stencil_counts)
    else:
        spatial = output_height * output_width
        mask = row_mask.reshape(batch, out_channels, spatial)
        result = sum(
            int(np.count_nonzero(mask[:, :, position])) * entries
            for position, entries in enumerate(stencil_counts)
        )
    if type(result) is not int or result < 0:
        raise RuntimeLineageReject("runtime_conv_logical_nnz_invalid")
    return result


def _live_operator_reference_objects(
    operator_value: object,
) -> tuple[object, ...]:
    if type(operator_value) is ImplicitConv2DOp:
        state = _operator_raw_state(
            operator_value,
            expected_keys=_CONV_OPERATOR_STATE_KEYS,
            name="runtime_conv_operator",
        )
        return (
            operator_value,
            state,
            state["_kernel"],
            state["_input_shape"],
            state["_output_shape"],
            state["_stride"],
            state["_padding"],
            state["_dilation"],
            state["_groups"],
            state["_row_mask"],
            state["_logical_expanded_nnz"],
            state["_content_key"],
        )
    if type(operator_value) is DiagonalLinearOp:
        state = _operator_raw_state(
            operator_value,
            expected_keys=_DIAGONAL_OPERATOR_STATE_KEYS,
            name="runtime_diagonal_operator",
        )
        return (
            operator_value,
            state,
            state["_diagonal"],
            state["_content_key"],
        )
    raise RuntimeLineageReject("runtime_operator_reference_type")


def _operator_snapshot(operator_value: object, kind: str, layer: _LayerSnapshot) -> str:
    params = layer.params
    if kind == "CONV2D":
        if type(operator_value) is not ImplicitConv2DOp:
            raise RuntimeLineageReject("conv_operator_type_mismatch")
        state = _operator_raw_state(
            operator_value,
            expected_keys=_CONV_OPERATOR_STATE_KEYS,
            name="runtime_conv_operator",
        )
        kernel_array = state["_kernel"]
        if (
            type(kernel_array) is not np.ndarray
            or kernel_array.dtype != np.dtype(np.float64)
            or kernel_array.ndim != 4
            or not kernel_array.flags.c_contiguous
            or not kernel_array.flags.owndata
            or any(dimension <= 0 for dimension in kernel_array.shape)
        ):
            raise RuntimeLineageReject("runtime_conv_kernel_storage_schema")
        kernel = _numeric_snapshot(
            kernel_array, "runtime_conv_kernel", ndim=4
        )
        graph_kernel = _numeric_snapshot(params["weight"], "graph_conv_weight", ndim=4)
        if kernel.canonical.shape != graph_kernel.canonical.shape or not np.array_equal(
            kernel.canonical, graph_kernel.canonical
        ):
            raise RuntimeLineageReject("conv_operator_payload_mismatch")
        if "bias" in params and params["bias"] is not None:
            inline_bias = _numeric_snapshot(
                params["bias"], "graph_conv_inline_bias", ndim=1
            ).canonical
            output_shape_for_bias = _exact_geometry_tuple(
                state["_output_shape"],
                name="runtime_conv_output_shape",
                length=4,
                minimum=1,
            )
            if inline_bias.size != output_shape_for_bias[1]:
                raise RuntimeLineageReject("conv_inline_bias_width_mismatch")
            if np.any(inline_bias != 0.0):
                raise RuntimeLineageReject(
                    "conv_inline_bias_transition_not_explicit"
                )
        graph_metadata = (
            _shape(params["input_shape"], "graph_conv_input_shape"),
            _shape(params["output_shape"], "graph_conv_output_shape"),
            _pair(params.get("stride", 1), "graph_conv_stride", minimum=1),
            _pair(params.get("padding", 0), "graph_conv_padding", minimum=0),
            _pair(params.get("dilation", 1), "graph_conv_dilation", minimum=1),
            _exact_int(params.get("groups", 1), "graph_conv_groups", minimum=1),
        )
        input_shape = _exact_geometry_tuple(
            state["_input_shape"],
            name="runtime_conv_input_shape",
            length=4,
            minimum=1,
        )
        output_shape = _exact_geometry_tuple(
            state["_output_shape"],
            name="runtime_conv_output_shape",
            length=4,
            minimum=1,
        )
        stride = _exact_geometry_tuple(
            state["_stride"],
            name="runtime_conv_stride",
            length=2,
            minimum=1,
        )
        padding = _exact_geometry_tuple(
            state["_padding"],
            name="runtime_conv_padding",
            length=2,
            minimum=0,
        )
        dilation = _exact_geometry_tuple(
            state["_dilation"],
            name="runtime_conv_dilation",
            length=2,
            minimum=1,
        )
        groups = state["_groups"]
        if type(groups) is not int or groups < 1:
            raise RuntimeLineageReject("runtime_conv_groups_schema")
        batch, channels, height, width = input_shape
        out_channels, in_per_group, kernel_height, kernel_width = (
            kernel_array.shape
        )
        if channels != in_per_group * groups or out_channels % groups:
            raise RuntimeLineageReject("runtime_conv_group_geometry_mismatch")
        output_height = (
            height
            + 2 * padding[0]
            - dilation[0] * (kernel_height - 1)
            - 1
        ) // stride[0] + 1
        output_width = (
            width
            + 2 * padding[1]
            - dilation[1] * (kernel_width - 1)
            - 1
        ) // stride[1] + 1
        expected_output_shape = (
            batch,
            int(out_channels),
            int(output_height),
            int(output_width),
        )
        if output_height <= 0 or output_width <= 0 or (
            output_shape != expected_output_shape
        ):
            raise RuntimeLineageReject("runtime_conv_output_geometry_mismatch")
        op_metadata = (
            input_shape,
            output_shape,
            stride,
            padding,
            dilation,
            groups,
        )
        if graph_metadata != op_metadata:
            raise RuntimeLineageReject("conv_operator_geometry_mismatch")
        if "kernel_size" in params and _pair(
            params["kernel_size"], "graph_conv_kernel_size", minimum=1
        ) != tuple(kernel_array.shape[-2:]):
            raise RuntimeLineageReject("conv_operator_kernel_size_mismatch")
        if "in_channels" in params and _exact_int(
            params["in_channels"], "graph_conv_in_channels", minimum=1
        ) != input_shape[1]:
            raise RuntimeLineageReject("conv_operator_in_channels_mismatch")
        if "out_channels" in params and _exact_int(
            params["out_channels"], "graph_conv_out_channels", minimum=1
        ) != output_shape[1]:
            raise RuntimeLineageReject("conv_operator_out_channels_mismatch")
        flattened_shape = (
            int(np.prod(output_shape)),
            int(np.prod(input_shape)),
        )
        if flattened_shape != (len(layer.out_vars), len(layer.in_vars)):
            raise RuntimeLineageReject("conv_operator_variable_shape_mismatch")
        mask = b"none"
        row_mask = state["_row_mask"]
        mask_digest: bytes | None = None
        if row_mask is not None:
            if (
                type(row_mask) is not np.ndarray
                or row_mask.dtype != np.dtype(bool)
                or row_mask.ndim != 1
                or row_mask.size != flattened_shape[0]
                or not row_mask.flags.c_contiguous
            ):
                raise RuntimeLineageReject("conv_operator_row_mask_mismatch")
            mask = row_mask.astype(np.uint8, copy=True).tobytes()
            mask_digest = _detached_array_digest(row_mask)
        expected_logical_nnz = _pure_conv_logical_nnz(
            kernel_shape=tuple(kernel_array.shape),
            input_shape=input_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            row_mask=row_mask,
        )
        logical_nnz = state["_logical_expanded_nnz"]
        if (
            type(logical_nnz) is not int
            or logical_nnz < 0
            or logical_nnz != expected_logical_nnz
        ):
            raise RuntimeLineageReject("runtime_conv_logical_nnz_cache_mismatch")
        expected_content_key = (
            "implicit_conv2d_op_v1",
            input_shape,
            output_shape,
            stride,
            padding,
            dilation,
            groups,
            _detached_array_digest(kernel_array),
            mask_digest,
        )
        content_key = state["_content_key"]
        if (
            type(content_key) is not tuple
            or len(content_key) != 9
            or type(content_key[0]) is not str
            or any(type(content_key[index]) is not tuple for index in range(1, 6))
            or any(
                any(type(item) is not int for item in content_key[index])
                for index in range(1, 6)
            )
            or type(content_key[6]) is not int
            or type(content_key[7]) is not bytes
            or (content_key[8] is not None and type(content_key[8]) is not bytes)
            or content_key != expected_content_key
        ):
            raise RuntimeLineageReject("runtime_conv_content_cache_mismatch")
        return _payload_digest((kernel.payload, _metadata_payload(op_metadata), mask))
    if kind == "SCALE":
        if type(operator_value) is not DiagonalLinearOp:
            raise RuntimeLineageReject("scale_operator_not_diagonal_linear_op")
        state = _operator_raw_state(
            operator_value,
            expected_keys=_DIAGONAL_OPERATOR_STATE_KEYS,
            name="runtime_diagonal_operator",
        )
        diagonal_array = state["_diagonal"]
        if (
            type(diagonal_array) is not np.ndarray
            or diagonal_array.dtype != np.dtype(np.float64)
            or diagonal_array.ndim != 1
            or diagonal_array.size <= 0
            or not diagonal_array.flags.c_contiguous
            or not diagonal_array.flags.owndata
        ):
            raise RuntimeLineageReject("runtime_scale_storage_schema")
        diagonal = _numeric_snapshot(diagonal_array, "runtime_scale", ndim=1)
        graph_scale = _numeric_snapshot(params["a"], "graph_scale", ndim=1)
        if diagonal.canonical.shape != graph_scale.canonical.shape or not np.array_equal(
            diagonal.canonical, graph_scale.canonical
        ):
            raise RuntimeLineageReject("scale_operator_payload_mismatch")
        expected_shape = (int(diagonal_array.size), int(diagonal_array.size))
        if expected_shape != (len(layer.out_vars), len(layer.in_vars)):
            raise RuntimeLineageReject("scale_operator_variable_shape_mismatch")
        expected_content_key = (
            "diagonal_linear_op_v1",
            expected_shape,
            _detached_array_digest(diagonal_array),
        )
        content_key = state["_content_key"]
        if (
            type(content_key) is not tuple
            or len(content_key) != 3
            or type(content_key[0]) is not str
            or type(content_key[1]) is not tuple
            or len(content_key[1]) != 2
            or any(type(item) is not int for item in content_key[1])
            or type(content_key[2]) is not bytes
            or content_key != expected_content_key
        ):
            raise RuntimeLineageReject("runtime_scale_content_cache_mismatch")
        return _payload_digest((diagonal.payload,))
    raise RuntimeLineageReject("transition_cannot_have_operator")


def _bias_values(layer: _LayerSnapshot) -> tuple[float, ...]:
    values = _numeric_snapshot(layer.params["c"], "graph_bias", ndim=1).canonical
    if values.size != len(layer.out_vars) or len(layer.in_vars) != len(layer.out_vars):
        raise RuntimeLineageReject("bias_payload_variable_width_mismatch")
    return tuple(float(item) for item in values)


def _runtime_event_primitive_signature(
    event: object,
) -> tuple[object, ...]:
    """Normalize only exact inert primitives; never invoke hostile equality."""

    if type(event) is not RuntimeLineageEvent:
        raise RuntimeLineageReject("runtime_lineage_event_type")
    schema = _exact_str(event.schema, "runtime_event_schema")
    if schema != RUNTIME_EVENT_SCHEMA:
        raise RuntimeLineageReject("runtime_event_schema_mismatch")
    graph_sha = _exact_sha256(
        event.graph_sha256, "runtime_event_graph_sha256"
    )
    occurrence_id = _exact_int(
        event.occurrence_layer_id, "runtime_event_occurrence"
    )
    kind = _exact_str(event.kind, "runtime_event_kind")
    if kind not in {"CONV", "SCALE", "BIAS", "ADD"}:
        raise RuntimeLineageReject("runtime_event_kind_unsupported")
    predecessors = _exact_int_tuple(
        event.ordered_predecessor_layer_ids,
        "runtime_event_predecessors",
        nonempty=True,
    )
    selected = _exact_int(event.selected_input, "runtime_event_selected_input")
    if selected >= len(predecessors):
        raise RuntimeLineageReject(
            "runtime_lineage_custody_event_selected_input_out_of_range"
        )
    if type(event.input_var_tokens) is not tuple:
        raise RuntimeLineageReject("runtime_event_input_tokens_not_tuple")
    input_tokens = tuple(
        _exact_int_tuple(
            token,
            "runtime_event_input_token",
            nonempty=True,
        )
        for token in event.input_var_tokens
    )
    if len(input_tokens) != len(predecessors):
        raise RuntimeLineageReject("runtime_event_input_token_arity")
    output_token = _exact_int_tuple(
        event.output_var_token,
        "runtime_event_output_token",
        nonempty=True,
    )
    position = _exact_int(
        event.operator_position, "runtime_event_operator_position"
    )
    layer_payload = _exact_sha256(
        event.layer_payload_sha256, "runtime_event_layer_payload"
    )
    operator_payload = _exact_str(
        event.operator_payload_sha256,
        "runtime_event_operator_payload",
        nonempty=False,
    )
    if kind in {"CONV", "SCALE"}:
        _exact_sha256(operator_payload, "runtime_event_operator_payload")
        if event.operator_occurrence is None:
            raise RuntimeLineageReject("runtime_event_operator_missing")
    elif operator_payload or event.operator_occurrence is not None:
        raise RuntimeLineageReject("runtime_transition_has_operator")
    if event.occurrence_layer is None:
        raise RuntimeLineageReject("runtime_event_occurrence_object_missing")
    return (
        schema,
        graph_sha,
        occurrence_id,
        kind,
        predecessors,
        selected,
        input_tokens,
        output_token,
        position,
        layer_payload,
        operator_payload,
    )


def _lineage_custody_guard(custody: object) -> _LineageCustody:
    if type(custody) is not _LineageCustody:
        raise RuntimeLineageReject("runtime_lineage_private_custody_missing")
    if custody.marker is not _CUSTODY_MARKER:
        raise RuntimeLineageReject("runtime_lineage_private_custody_missing")
    _exact_sha256(
        custody.canonical_sha256, "runtime_lineage_custody_sha256"
    )
    if (
        type(custody.occurrence_objects) is not tuple
        or type(custody.operator_objects) is not tuple
    ):
        raise RuntimeLineageReject("runtime_lineage_custody_container_type")
    _exact_int(
        custody.entry_source_layer_id,
        "runtime_lineage_custody_source_layer",
    )
    if custody.source_object is None or type(custody.arena) is not _RuntimeLineageArena:
        raise RuntimeLineageReject("runtime_lineage_custody_authority_missing")
    return custody


def _event_canonical_payload(events: tuple[RuntimeLineageEvent, ...]) -> bytes:
    if type(events) is not tuple:
        raise RuntimeLineageReject("runtime_lineage_events_not_tuple")
    payload = []
    for event in events:
        (
            schema,
            graph_sha,
            occurrence_id,
            kind,
            predecessors,
            selected,
            input_tokens,
            output_token,
            position,
            layer_payload,
            operator_payload,
        ) = _runtime_event_primitive_signature(event)
        payload.append(
            {
                "schema": schema,
                "graph": graph_sha,
                "occurrence": occurrence_id,
                "kind": kind,
                "preds": predecessors,
                "selected": selected,
                "inputs": input_tokens,
                "output": output_token,
                "position": position,
                "layer_payload": layer_payload,
                "operator_payload": operator_payload,
            }
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")


def _prefix_record(
    *,
    source: object,
    entry_layer_id: int,
    operators: tuple[object, ...],
    events: tuple[RuntimeLineageEvent, ...],
    add_event_index: int,
) -> _ExpectedAddPrefix:
    add_event = events[add_event_index]
    prefix_events = events[:add_event_index]
    return _ExpectedAddPrefix(
        add_layer_id=add_event.occurrence_layer_id,
        selected_input=add_event.selected_input,
        entry_layer_id=entry_layer_id,
        source=source,
        operator_prefix=operators[: add_event.operator_position],
        occurrence_prefix=tuple(
            prefix.occurrence_layer for prefix in prefix_events
        ),
        canonical_sha256=hashlib.sha256(
            _event_canonical_payload(prefix_events)
        ).hexdigest(),
    )


def _expected_add_prefix_guard(
    prefix: object,
) -> _ExpectedAddPrefix:
    if type(prefix) is not _ExpectedAddPrefix:
        raise RuntimeLineageReject("runtime_add_expected_prefix_type")
    _exact_int(prefix.add_layer_id, "runtime_add_expected_prefix_layer")
    selected = _exact_int(
        prefix.selected_input, "runtime_add_expected_prefix_selected"
    )
    if selected not in (0, 1):
        raise RuntimeLineageReject(
            "runtime_add_expected_prefix_selected_out_of_range"
        )
    _exact_int(prefix.entry_layer_id, "runtime_add_expected_prefix_entry")
    if (
        prefix.source is None
        or type(prefix.operator_prefix) is not tuple
        or type(prefix.occurrence_prefix) is not tuple
    ):
        raise RuntimeLineageReject("runtime_add_expected_prefix_container_type")
    _exact_sha256(
        prefix.canonical_sha256,
        "runtime_add_expected_prefix_canonical_sha256",
    )
    return prefix


def _prefix_identity_equal(
    left: _ExpectedAddPrefix, right: _ExpectedAddPrefix
) -> bool:
    left = _expected_add_prefix_guard(left)
    right = _expected_add_prefix_guard(right)
    return (
        left.add_layer_id == right.add_layer_id
        and left.selected_input == right.selected_input
        and left.entry_layer_id == right.entry_layer_id
        and left.source is right.source
        and left.canonical_sha256 == right.canonical_sha256
        and _identity_tuple_equal(left.operator_prefix, right.operator_prefix)
        and _identity_tuple_equal(
            left.occurrence_prefix, right.occurrence_prefix
        )
    )


def _terminal_term_record(
    *,
    source: object,
    operators: tuple[object, ...],
    events: tuple[RuntimeLineageEvent, ...],
    terminal_layer_id: int,
) -> _ExpectedTerminalTerm:
    path = _path_from_events(events)
    if path[-1] != terminal_layer_id:
        raise RuntimeLineageReject("runtime_terminal_term_occurrence_mismatch")
    return _ExpectedTerminalTerm(
        terminal_layer_id=terminal_layer_id,
        entry_layer_id=path[0],
        source=source,
        operators=operators,
        occurrences=tuple(event.occurrence_layer for event in events),
        canonical_sha256=hashlib.sha256(
            _event_canonical_payload(events)
        ).hexdigest(),
    )


def _expected_terminal_term_guard(
    expected: object,
) -> _ExpectedTerminalTerm:
    if type(expected) is not _ExpectedTerminalTerm:
        raise RuntimeLineageReject("runtime_terminal_expected_term_type")
    _exact_int(
        expected.terminal_layer_id, "runtime_terminal_expected_layer"
    )
    _exact_int(expected.entry_layer_id, "runtime_terminal_expected_entry")
    if (
        expected.source is None
        or type(expected.operators) is not tuple
        or type(expected.occurrences) is not tuple
    ):
        raise RuntimeLineageReject(
            "runtime_terminal_expected_term_container_type"
        )
    _exact_sha256(
        expected.canonical_sha256,
        "runtime_terminal_expected_canonical_sha256",
    )
    return expected


def _terminal_term_identity_equal(
    left: _ExpectedTerminalTerm, right: _ExpectedTerminalTerm
) -> bool:
    left = _expected_terminal_term_guard(left)
    right = _expected_terminal_term_guard(right)
    return (
        left.terminal_layer_id == right.terminal_layer_id
        and left.entry_layer_id == right.entry_layer_id
        and left.source is right.source
        and left.canonical_sha256 == right.canonical_sha256
        and _identity_tuple_equal(left.operators, right.operators)
        and _identity_tuple_equal(left.occurrences, right.occurrences)
    )


def _require_exact_terminal_term_multiset(
    snapshot: _RegisteredTerminalExpression,
    actual: tuple[_ExpectedTerminalTerm, ...],
) -> None:
    if type(actual) is not tuple or type(snapshot.expected_terms) is not tuple:
        raise RuntimeLineageReject(
            "runtime_terminal_expected_terms_container_type"
        )
    expected = snapshot.expected_terms
    for item in (*expected, *actual):
        _expected_terminal_term_guard(item)
    if len(actual) != len(expected):
        raise RuntimeLineageReject(
            "runtime_terminal_expression_term_multiplicity_mismatch"
        )
    unmatched = list(expected)
    for candidate in actual:
        match = next(
            (
                index
                for index, reference in enumerate(unmatched)
                if _terminal_term_identity_equal(candidate, reference)
            ),
            None,
        )
        if match is None:
            raise RuntimeLineageReject(
                "runtime_terminal_expression_term_mismatch"
            )
        unmatched.pop(match)
    if unmatched:
        raise RuntimeLineageReject(
            "runtime_terminal_expression_term_missing"
        )


def _require_exact_add_prefix_multiset(
    registry: _RuntimeLineageRegistry,
    add_layer_id: int,
    actual: tuple[_ExpectedAddPrefix, ...],
) -> None:
    if type(actual) is not tuple:
        raise RuntimeLineageReject("runtime_add_prefix_multiset_not_tuple")
    for prefix in actual:
        _expected_add_prefix_guard(prefix)
    sealed = _exact_bool(registry.sealed, "runtime_registry_sealed")
    expected_snapshot_container = tuple if sealed else list
    if type(registry.add_operand_snapshots) is not expected_snapshot_container:
        raise RuntimeLineageReject("runtime_add_snapshot_container_type")
    for snapshot in registry.add_operand_snapshots:
        _registered_add_snapshot_schema(snapshot)
    matching_snapshots = tuple(
        snapshot
        for snapshot in registry.add_operand_snapshots
        if snapshot.add_layer_id == add_layer_id
    )
    if len(matching_snapshots) != 1:
        raise RuntimeLineageReject("runtime_add_operand_snapshot_missing_or_duplicate")
    expected = matching_snapshots[0].expected_prefixes
    if type(expected) is not tuple or not expected:
        raise RuntimeLineageReject("runtime_add_expected_prefixes_missing")
    for prefix in expected:
        _expected_add_prefix_guard(prefix)
    if sorted(prefix.selected_input for prefix in expected) != sorted(
        prefix.selected_input for prefix in actual
    ):
        raise RuntimeLineageReject("runtime_add_input_multiplicity_mismatch")
    unmatched = list(expected)
    for candidate in actual:
        match = next(
            (
                index
                for index, reference in enumerate(unmatched)
                if _prefix_identity_equal(candidate, reference)
            ),
            None,
        )
        if match is None:
            raise RuntimeLineageReject(
                "runtime_add_term_prefix_extra_duplicate_or_mismatch"
            )
        unmatched.pop(match)
    if unmatched:
        raise RuntimeLineageReject("runtime_add_term_prefix_missing")


def _record_path(
    graph: _GraphSnapshot,
    registry: _RuntimeLineageRegistry,
    source: object,
    path_layer_ids: Sequence[int],
    operators: tuple[object, ...],
    *,
    allow_empty_events: bool = False,
) -> tuple[RuntimeLineageEvent, ...]:
    source_boundaries = registry.source_boundaries
    path = _int_tuple(path_layer_ids, "path_layer_ids", nonempty=True)
    if len(path) < 2 and not allow_empty_events:
        raise RuntimeLineageReject("path_requires_entry_and_event")
    if type(operators) is not tuple:
        raise RuntimeLineageReject("term_operators_not_tuple")
    if any(layer_id >= len(graph.layers) for layer_id in path):
        raise RuntimeLineageReject("path_unknown_layer")
    if source_boundaries.source_at(path[0]) is not source:
        raise RuntimeLineageReject("runtime_source_boundary_identity_mismatch")
    _source_boundary_payload(source, graph.layers[path[0]])
    events: list[RuntimeLineageEvent] = []
    cursor = 0
    seen_outputs: set[int] = set(graph.layers[path[0]].out_vars)
    for previous_id, layer_id in zip(path, path[1:]):
        layer = graph.layers[layer_id]
        previous = graph.layers[previous_id]
        if layer.kind not in _EVENT_KIND:
            raise RuntimeLineageReject(f"unsupported_runtime_event_kind:{layer.kind}")
        if previous_id not in layer.preds:
            raise RuntimeLineageReject("runtime_event_path_skips_graph_edge")
        selected = layer.preds.index(previous_id)
        if layer.kind == "ADD":
            if len(layer.preds) != 2 or len(layer.input_tokens) != 2:
                raise RuntimeLineageReject("add_event_arity")
            if len(layer.out_vars) != len(layer.input_tokens[0]) or any(
                len(token) != len(layer.out_vars) for token in layer.input_tokens
            ):
                raise RuntimeLineageReject("add_operand_width_mismatch")
            for predecessor_id, token in zip(layer.preds, layer.input_tokens, strict=True):
                if graph.layers[predecessor_id].out_vars != token:
                    raise RuntimeLineageReject("add_operand_predecessor_value_mismatch")
        else:
            if layer.preds != (previous_id,) or layer.input_tokens != (layer.in_vars,):
                raise RuntimeLineageReject("unary_event_arity_or_predecessor_mismatch")
            if previous.out_vars != layer.in_vars:
                raise RuntimeLineageReject("unary_event_value_flow_mismatch")
        if set(layer.out_vars) & seen_outputs:
            raise RuntimeLineageReject("runtime_path_output_value_alias")
        if set(layer.out_vars) & set(layer.in_vars):
            raise RuntimeLineageReject("runtime_event_input_output_alias")
        seen_outputs.update(layer.out_vars)

        operator_occurrence: object | None = None
        operator_payload = ""
        if layer.kind in _MULTIPLICATIVE:
            if cursor >= len(operators):
                raise RuntimeLineageReject("missing_runtime_operator_event")
            operator_occurrence = operators[cursor]
            operator_payload = _operator_snapshot(operator_occurrence, layer.kind, layer)
            cursor += 1
        elif layer.kind in _TRANSITION:
            operator_occurrence = None
        events.append(
            RuntimeLineageEvent(
                RUNTIME_EVENT_SCHEMA,
                graph.graph_sha256,
                layer_id,
                layer.layer,
                _EVENT_KIND[layer.kind],
                layer.preds,
                selected,
                layer.input_tokens,
                layer.out_vars,
                cursor - 1 if layer.kind in _MULTIPLICATIVE else cursor,
                layer.layer_payload_sha256,
                operator_occurrence,
                operator_payload,
            )
        )
    if cursor != len(operators):
        raise RuntimeLineageReject("runtime_operator_without_graph_event")
    bare = tuple(events)
    canonical = hashlib.sha256(_event_canonical_payload(bare)).hexdigest()
    custody = _LineageCustody(
        _CUSTODY_MARKER,
        canonical,
        tuple(event.occurrence_layer for event in bare),
        tuple(event.operator_occurrence for event in bare),
        path[0],
        source,
        registry.arena,
    )
    return tuple(replace(event, _custody=custody) for event in bare)


def capture_runtime_lineage_events(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    path_layer_ids: Sequence[int],
    operators: tuple[object, ...],
    *,
    registry: object,
    source: object,
) -> tuple[RuntimeLineageEvent, ...]:
    """Capture proof vocabulary; the adapter will not trust it later."""

    graph = _freeze_graph(layers, preds, succs)
    private_registry = _validate_registry_current(
        registry, graph, require_sealed=False
    )
    return _record_path(
        graph,
        private_registry,
        source,
        path_layer_ids,
        operators,
    )


def _empty_path_custody(
    registry: _RuntimeLineageRegistry,
    source: object,
    entry_layer_id: int,
) -> _LineageCustody:
    canonical = hashlib.sha256(_event_canonical_payload(())).hexdigest()
    return _LineageCustody(
        _CUSTODY_MARKER,
        canonical,
        (),
        (),
        entry_layer_id,
        source,
        registry.arena,
    )


def _capture_private_runtime_operand_lineage(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    path_layer_ids: Sequence[int],
    operators: tuple[object, ...],
    *,
    registry: object,
    source: object,
) -> _RuntimeOperandLineage:
    """Internal pre-ADD cache recorder, independent of candidate capture."""

    graph = _freeze_graph(layers, preds, succs)
    private_registry = _validate_registry_current(
        registry, graph, require_sealed=False
    )
    path = _int_tuple(path_layer_ids, "operand_path_layer_ids", nonempty=True)
    events = _record_path(
        graph,
        private_registry,
        source,
        path,
        operators,
        allow_empty_events=True,
    )
    custody = (
        events[0]._custody
        if events
        else _empty_path_custody(private_registry, source, path[0])
    )
    if type(custody) is not _LineageCustody:
        raise RuntimeLineageReject("runtime_operand_private_custody_missing")
    return _RuntimeOperandLineage(source, operators, path, events, custody)


def _runtime_operand_lineage_schema(
    lineage: object,
) -> tuple[_RuntimeOperandLineage, tuple[int, ...], _LineageCustody]:
    if type(lineage) is not _RuntimeOperandLineage:
        raise RuntimeLineageReject("runtime_operand_lineage_type")
    if type(lineage.operators) is not tuple or type(lineage.events) is not tuple:
        raise RuntimeLineageReject("runtime_operand_lineage_container_type")
    path = _exact_int_tuple(
        lineage.path_layer_ids, "runtime_operand_path", nonempty=True
    )
    for event in lineage.events:
        _runtime_event_primitive_signature(event)
    custody = _lineage_custody_guard(lineage.custody)
    if lineage.source is None or custody.source_object is not lineage.source:
        raise RuntimeLineageReject("runtime_operand_lineage_source_mismatch")
    if custody.entry_source_layer_id != path[0]:
        raise RuntimeLineageReject("runtime_operand_lineage_entry_mismatch")
    if len(lineage.events) != len(custody.occurrence_objects) or len(
        lineage.events
    ) != len(custody.operator_objects):
        raise RuntimeLineageReject("runtime_operand_lineage_custody_arity")
    if any(event._custody is not custody for event in lineage.events):
        raise RuntimeLineageReject("runtime_operand_lineage_mixed_custody")
    if any(
        event.occurrence_layer is not occurrence
        for event, occurrence in zip(
            lineage.events, custody.occurrence_objects, strict=True
        )
    ) or any(
        event.operator_occurrence is not operator_value
        for event, operator_value in zip(
            lineage.events, custody.operator_objects, strict=True
        )
    ):
        raise RuntimeLineageReject("runtime_operand_lineage_custody_identity")
    canonical = hashlib.sha256(
        _event_canonical_payload(lineage.events)
    ).hexdigest()
    if canonical != custody.canonical_sha256:
        raise RuntimeLineageReject("runtime_operand_lineage_custody_digest")
    return lineage, path, custody


def _validate_operand_lineage_current(
    lineage: object,
    graph: _GraphSnapshot,
    registry: _RuntimeLineageRegistry,
) -> _RuntimeOperandLineage:
    lineage, path, custody = _runtime_operand_lineage_schema(lineage)
    current = _record_path(
        graph,
        registry,
        lineage.source,
        path,
        lineage.operators,
        allow_empty_events=True,
    )
    _same_event_observation(lineage.events, current)
    current_custody = (
        current[0]._custody
        if current
        else _empty_path_custody(registry, lineage.source, path[0])
    )
    current_custody = _lineage_custody_guard(current_custody)
    if (
        custody.arena is not registry.arena
        or custody.source_object is not lineage.source
        or custody.entry_source_layer_id != path[0]
        or custody.canonical_sha256 != current_custody.canonical_sha256
        or not _identity_tuple_equal(
            custody.occurrence_objects, current_custody.occurrence_objects
        )
        or not _identity_tuple_equal(
            custody.operator_objects, current_custody.operator_objects
        )
    ):
        raise RuntimeLineageReject("runtime_operand_private_custody_mismatch")
    return lineage


def _expected_prefix_from_operand(
    lineage: _RuntimeOperandLineage,
    add_layer_id: int,
    selected_input: int,
) -> _ExpectedAddPrefix:
    return _ExpectedAddPrefix(
        add_layer_id=add_layer_id,
        selected_input=selected_input,
        entry_layer_id=lineage.path_layer_ids[0],
        source=lineage.source,
        operator_prefix=lineage.operators,
        occurrence_prefix=tuple(
            event.occurrence_layer for event in lineage.events
        ),
        canonical_sha256=hashlib.sha256(
            _event_canonical_payload(lineage.events)
        ).hexdigest(),
    )


def _affine_cache_entry_guard(
    cache_entry: object,
    graph: _GraphSnapshot,
) -> tuple[bytes, tuple[object, ...]]:
    if type(cache_entry) is not _RuntimeAffineExprCacheView:
        raise RuntimeLineageReject("runtime_affine_cache_entry_type")
    if type(cache_entry.terms) is not tuple or not cache_entry.terms:
        raise RuntimeLineageReject("runtime_affine_cache_terms_missing")
    n_out = _exact_int(
        cache_entry.n_out, "runtime_cached_expression_n_out", minimum=1
    )
    frame_payload = _stable_token_payload(
        _exact_int(
            cache_entry.frame_id, "runtime_cached_expression_frame"
        ),
        "runtime_cached_expression_frame",
    )
    bias_payload = _numeric_snapshot(
        cache_entry.bias, "runtime_cached_expression_bias", ndim=1
    ).payload
    refs: list[object] = [
        cache_entry,
        cache_entry.terms,
        cache_entry.bias,
        cache_entry.frame_id,
    ]
    term_records: list[object] = []
    for lineage in cache_entry.terms:
        lineage, lineage_path, lineage_custody = (
            _runtime_operand_lineage_schema(lineage)
        )
        refs.extend(
            (
                lineage,
                lineage.source,
                lineage.operators,
                lineage.events,
                lineage_custody,
                *lineage.operators,
                *lineage.events,
                *lineage_custody.occurrence_objects,
                *lineage_custody.operator_objects,
            )
        )
        operator_events = tuple(
            event
            for event in lineage.events
            if event.operator_occurrence is not None
        )
        if len(operator_events) != len(lineage.operators) or any(
            event.operator_occurrence is not operator_value
            for event, operator_value in zip(
                operator_events, lineage.operators, strict=True
            )
        ):
            raise RuntimeLineageReject(
                "runtime_affine_cache_operator_event_bijection_mismatch"
            )
        operator_payloads: list[str] = []
        for event, operator_value in zip(
            operator_events, lineage.operators, strict=True
        ):
            layer_id = _exact_int(
                event.occurrence_layer_id,
                "runtime_affine_cache_operator_layer",
            )
            if layer_id >= len(graph.layers):
                raise RuntimeLineageReject(
                    "runtime_affine_cache_operator_layer_unknown"
                )
            layer = graph.layers[layer_id]
            operator_payload = _operator_snapshot(
                operator_value, layer.kind, layer
            )
            if operator_payload != event.operator_payload_sha256:
                raise RuntimeLineageReject(
                    "runtime_affine_cache_operator_payload_mismatch"
                )
            operator_payloads.append(operator_payload)
            refs.extend(_live_operator_reference_objects(operator_value))
        term_records.append(
            (
                lineage_path,
                hashlib.sha256(
                    _event_canonical_payload(lineage.events)
                ).hexdigest(),
                tuple(operator_payloads),
            )
        )
    metadata = _metadata_payload([n_out, frame_payload.hex(), term_records])
    payload = (
        len(metadata).to_bytes(8, "big")
        + metadata
        + len(bias_payload).to_bytes(8, "big")
        + bias_payload
    )
    return payload, tuple(refs)


def _validate_cached_affine_expression_current(
    cache_entry: object,
    producer_layer_id: int,
    graph: _GraphSnapshot,
    registry: _RuntimeLineageRegistry,
) -> _RuntimeAffineExprCacheView:
    if type(cache_entry) is not _RuntimeAffineExprCacheView:
        raise RuntimeLineageReject("runtime_affine_cache_entry_type")
    if type(cache_entry.terms) is not tuple or not cache_entry.terms:
        raise RuntimeLineageReject("runtime_affine_cache_terms_missing")
    n_out = _exact_int(
        cache_entry.n_out, "runtime_cached_expression_n_out", minimum=1
    )
    if producer_layer_id >= len(graph.layers) or n_out != len(
        graph.layers[producer_layer_id].out_vars
    ):
        raise RuntimeLineageReject("runtime_affine_cache_output_width_mismatch")
    frame_payload = _stable_token_payload(
        _exact_int(
            cache_entry.frame_id, "runtime_cached_expression_frame"
        ),
        "runtime_cached_expression_frame",
    )
    if frame_payload != registry.arena.frame_payload:
        raise RuntimeLineageReject("runtime_affine_cache_frame_mismatch")
    bias = _numeric_snapshot(
        cache_entry.bias, "runtime_cached_expression_bias", ndim=1
    )
    if int(bias.canonical.size) != n_out:
        raise RuntimeLineageReject("runtime_affine_cache_bias_width_mismatch")
    for lineage in cache_entry.terms:
        current = _validate_operand_lineage_current(
            lineage, graph, registry
        )
        if current.path_layer_ids[-1] != producer_layer_id:
            raise RuntimeLineageReject(
                "runtime_affine_cache_term_terminal_mismatch"
            )
        source_payload, width = _source_boundary_payload(
            current.source,
            graph.layers[current.path_layer_ids[0]],
            expression_frame_payload=frame_payload,
        )
        del source_payload
        for operator_index, runtime_operator in enumerate(current.operators):
            raw_shape = getattr(runtime_operator, "shape", None)
            if type(raw_shape) is not tuple or len(raw_shape) != 2:
                raise RuntimeLineageReject("runtime_operator_shape_invalid")
            rows = _exact_int(
                raw_shape[0], f"runtime_cached_operator_{operator_index}_rows"
            )
            columns = _exact_int(
                raw_shape[1], f"runtime_cached_operator_{operator_index}_columns"
            )
            if columns != width:
                raise RuntimeLineageReject(
                    "runtime_affine_cache_operator_shape_chain_mismatch"
                )
            width = rows
        if width != n_out:
            raise RuntimeLineageReject(
                "runtime_affine_cache_term_output_width_mismatch"
            )
    return cache_entry


def _record_private_runtime_affine_expression_cache(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    registry: object,
    producer_layer_id: int,
    terms: tuple[_RuntimeOperandLineage, ...],
    bias: object,
    n_out: int,
    frame_id: object,
) -> _RuntimeAffineExprCacheView:
    """Trusted expression-construction hook for the owning runtime cache."""

    graph = _freeze_graph(layers, preds, succs)
    private_registry = _validate_registry_current(
        registry, graph, require_sealed=False
    )
    layer_id = _exact_int(
        producer_layer_id, "runtime_affine_cache_producer_layer"
    )
    if layer_id >= len(graph.layers):
        raise RuntimeLineageReject("runtime_affine_cache_unknown_producer")
    cache = private_registry.arena.owner_affine_cache
    if type(cache) is not dict or cache is not private_registry.arena.owner_affine_cache_identity:
        raise RuntimeLineageReject("runtime_affine_owner_cache_cas_mismatch")
    if layer_id in cache:
        raise RuntimeLineageReject("runtime_affine_cache_entry_already_recorded")
    entry = _RuntimeAffineExprCacheView(terms, bias, n_out, frame_id)
    _validate_cached_affine_expression_current(
        entry, layer_id, graph, private_registry
    )
    entry_payload, entry_references = _affine_cache_entry_guard(entry, graph)
    cache[layer_id] = entry
    private_registry.arena.affine_entries.append((layer_id, entry))
    private_registry.arena.affine_payloads.append(entry_payload)
    private_registry.arena.affine_reference_snapshots.append(
        entry_references
    )
    return entry


def _registered_add_snapshot_schema(
    snapshot: object,
) -> _RegisteredAddOperands:
    if type(snapshot) is not _RegisteredAddOperands:
        raise RuntimeLineageReject("runtime_add_operand_snapshot_type")
    _exact_int(snapshot.add_layer_id, "runtime_add_layer")
    if (
        type(snapshot.operand_cache_entries) is not tuple
        or len(snapshot.operand_cache_entries) != 2
        or any(
            type(entry) is not _RuntimeAffineExprCacheView
            for entry in snapshot.operand_cache_entries
        )
    ):
        raise RuntimeLineageReject("runtime_add_operand_cache_entry_arity")
    if (
        type(snapshot.operands) is not tuple
        or len(snapshot.operands) != 2
        or any(type(terms) is not tuple for terms in snapshot.operands)
    ):
        raise RuntimeLineageReject("runtime_add_operand_cache_arity")
    if type(snapshot.expected_prefixes) is not tuple:
        raise RuntimeLineageReject("runtime_add_expected_prefix_container_type")
    for prefix in snapshot.expected_prefixes:
        _expected_add_prefix_guard(prefix)
    return snapshot


def _validate_registered_add_operands_current(
    snapshot: object,
    graph: _GraphSnapshot,
    registry: _RuntimeLineageRegistry,
) -> _RegisteredAddOperands:
    snapshot = _registered_add_snapshot_schema(snapshot)
    add_layer_id = _exact_int(snapshot.add_layer_id, "runtime_add_layer")
    if add_layer_id >= len(graph.layers):
        raise RuntimeLineageReject("runtime_add_layer_unknown")
    add_layer = graph.layers[add_layer_id]
    if add_layer.kind != "ADD" or len(add_layer.preds) != 2:
        raise RuntimeLineageReject("runtime_add_snapshot_not_binary_add")
    expected: list[_ExpectedAddPrefix] = []
    nested_prefixes_by_add: dict[int, list[_ExpectedAddPrefix]] = {}
    current_entries: list[_RuntimeAffineExprCacheView] = []
    for selected_input, predecessor_id in enumerate(add_layer.preds):
        cache_entry = registry.arena.owner_affine_cache.get(predecessor_id)
        if cache_entry is not snapshot.operand_cache_entries[selected_input]:
            raise RuntimeLineageReject(
                "runtime_add_operand_cache_entry_cas_mismatch"
            )
        current_entry = _validate_cached_affine_expression_current(
            cache_entry, predecessor_id, graph, registry
        )
        if current_entry.terms is not snapshot.operands[selected_input]:
            raise RuntimeLineageReject(
                "runtime_add_operand_term_tuple_cas_mismatch"
            )
        current_entries.append(current_entry)
    if current_entries[0] is current_entries[1]:
        raise RuntimeLineageReject("runtime_add_operand_cache_entry_alias")
    for selected_input, current_entry in enumerate(current_entries):
        operand_terms = current_entry.terms
        if type(operand_terms) is not tuple or not operand_terms:
            raise RuntimeLineageReject("runtime_add_operand_cache_empty")
        for lineage in operand_terms:
            current = _validate_operand_lineage_current(
                lineage, graph, registry
            )
            if current.path_layer_ids[-1] != add_layer.preds[selected_input]:
                raise RuntimeLineageReject(
                    "runtime_add_operand_terminal_predecessor_mismatch"
                )
            expected.append(
                _expected_prefix_from_operand(
                    current, add_layer_id, selected_input
                )
            )
            for event_index, event in enumerate(current.events):
                if event.kind != "ADD":
                    continue
                nested = _prefix_record(
                    source=current.source,
                    entry_layer_id=current.path_layer_ids[0],
                    operators=current.operators,
                    events=current.events,
                    add_event_index=event_index,
                )
                nested_prefixes_by_add.setdefault(
                    nested.add_layer_id, []
                ).append(nested)
    current_expected = tuple(expected)
    if len(current_expected) != len(snapshot.expected_prefixes) or any(
        not _prefix_identity_equal(left, right)
        for left, right in zip(
            current_expected, snapshot.expected_prefixes, strict=True
        )
    ):
        raise RuntimeLineageReject("runtime_add_operand_snapshot_cas_mismatch")
    for nested_add_id, nested_prefixes in nested_prefixes_by_add.items():
        _require_exact_add_prefix_multiset(
            registry, nested_add_id, tuple(nested_prefixes)
        )
    return snapshot


def _record_private_runtime_add_operands(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    registry: object,
    add_layer_id: int,
) -> None:
    """One ADD hook snapshots both current predecessor expression caches."""

    graph = _freeze_graph(layers, preds, succs)
    private_registry = _validate_registry_current(
        registry, graph, require_sealed=False
    )
    if type(private_registry.add_operand_snapshots) is not list:
        raise RuntimeLineageReject("runtime_registry_not_open")
    layer_id = _exact_int(add_layer_id, "runtime_add_layer")
    if any(
        snapshot.add_layer_id == layer_id
        for snapshot in private_registry.add_operand_snapshots
    ):
        raise RuntimeLineageReject("runtime_add_operand_snapshot_duplicate")
    add_layer = graph.layers[layer_id] if layer_id < len(graph.layers) else None
    if add_layer is None or add_layer.kind != "ADD" or len(add_layer.preds) != 2:
        raise RuntimeLineageReject("runtime_add_snapshot_not_binary_add")
    cache_entries = tuple(
        private_registry.arena.owner_affine_cache.get(predecessor_id)
        for predecessor_id in add_layer.preds
    )
    if any(entry is None for entry in cache_entries):
        raise RuntimeLineageReject("runtime_add_operand_cache_entry_missing")
    if len(cache_entries) != 2:
        raise RuntimeLineageReject("runtime_add_operand_cache_entry_arity")
    operands = tuple(entry.terms for entry in cache_entries)
    provisional = _RegisteredAddOperands(
        layer_id, cache_entries, operands, ()
    )
    expected: list[_ExpectedAddPrefix] = []
    for selected_input, operand_terms in enumerate(operands):
        if type(operand_terms) is not tuple or not operand_terms:
            raise RuntimeLineageReject("runtime_add_operand_cache_empty")
        for lineage in operand_terms:
            current = _validate_operand_lineage_current(
                lineage, graph, private_registry
            )
            if current.path_layer_ids[-1] != add_layer.preds[selected_input]:
                raise RuntimeLineageReject(
                    "runtime_add_operand_terminal_predecessor_mismatch"
                )
            expected.append(
                _expected_prefix_from_operand(
                    current, layer_id, selected_input
                )
            )
    provisional = replace(provisional, expected_prefixes=tuple(expected))
    _validate_registered_add_operands_current(
        provisional, graph, private_registry
    )
    private_registry.add_operand_snapshots.append(provisional)


def _terminal_boundary_consumer(
    graph: _GraphSnapshot, terminal_layer_id: int
) -> int | None:
    """Prove that no unrecorded linear event follows the chosen terminal."""

    terminal = graph.layers[terminal_layer_id]
    if not terminal.succs:
        sinks = tuple(
            layer.layer_id for layer in graph.layers if not layer.succs
        )
        if sinks != (terminal_layer_id,):
            raise RuntimeLineageReject("runtime_terminal_graph_sink_not_unique")
        return None
    if len(terminal.succs) != 1:
        raise RuntimeLineageReject("runtime_terminal_consumer_not_unique")
    consumer_id = terminal.succs[0]
    consumer = graph.layers[consumer_id]
    if consumer.kind not in _AUTHORIZED_TERMINAL_CONSUMER_KINDS:
        raise RuntimeLineageReject(
            "runtime_terminal_followed_by_unrecorded_linear_event"
        )
    if (
        consumer.preds != (terminal_layer_id,)
        or consumer.input_tokens != (terminal.out_vars,)
        or len(consumer.in_vars) != len(terminal.out_vars)
        or len(consumer.out_vars) != len(terminal.out_vars)
    ):
        raise RuntimeLineageReject("runtime_terminal_consumer_boundary_mismatch")
    return consumer_id


def _common_last_add_layer_id(
    term_events: Sequence[tuple[RuntimeLineageEvent, ...]],
) -> int:
    marker: tuple[int, tuple[int, ...]] | None = None
    for events in term_events:
        add_positions = tuple(
            index for index, event in enumerate(events) if event.kind == "ADD"
        )
        if not add_positions:
            raise RuntimeLineageReject("runtime_common_add_missing")
        last_add = add_positions[-1]
        current = (
            events[last_add].occurrence_layer_id,
            tuple(event.occurrence_layer_id for event in events[last_add:]),
        )
        if marker is None:
            marker = current
        elif marker != current:
            raise RuntimeLineageReject(
                "runtime_common_add_or_suffix_occurrence_mismatch"
            )
    if marker is None:
        raise RuntimeLineageReject("runtime_terminal_expression_terms_missing")
    return marker[0]


def _registered_terminal_snapshot_schema(
    snapshot: object,
) -> _RegisteredTerminalExpression:
    if type(snapshot) is not _RegisteredTerminalExpression:
        raise RuntimeLineageReject(
            "runtime_terminal_expression_snapshot_type"
        )
    _exact_int(snapshot.terminal_layer_id, "runtime_terminal_layer")
    if snapshot.terminal_consumer_layer_id is not None:
        _exact_int(
            snapshot.terminal_consumer_layer_id,
            "runtime_terminal_consumer_layer",
        )
    if type(snapshot.cache_entry) is not _RuntimeAffineExprCacheView:
        raise RuntimeLineageReject("runtime_terminal_cache_entry_type")
    if type(snapshot.terms) is not tuple:
        raise RuntimeLineageReject("runtime_terminal_terms_container_type")
    if type(snapshot.expected_terms) is not tuple:
        raise RuntimeLineageReject(
            "runtime_terminal_expected_terms_container_type"
        )
    for expected in snapshot.expected_terms:
        _expected_terminal_term_guard(expected)
    return snapshot


def _validate_registered_terminal_expression_current(
    snapshot: object,
    graph: _GraphSnapshot,
    registry: _RuntimeLineageRegistry,
) -> _RegisteredTerminalExpression:
    snapshot = _registered_terminal_snapshot_schema(snapshot)
    terminal_layer_id = _exact_int(
        snapshot.terminal_layer_id, "runtime_terminal_layer"
    )
    if terminal_layer_id >= len(graph.layers):
        raise RuntimeLineageReject("runtime_terminal_layer_unknown")
    current_consumer = _terminal_boundary_consumer(
        graph, terminal_layer_id
    )
    if current_consumer != snapshot.terminal_consumer_layer_id:
        raise RuntimeLineageReject("runtime_terminal_boundary_cas_mismatch")
    cache_entry = registry.arena.owner_affine_cache.get(terminal_layer_id)
    if cache_entry is not snapshot.cache_entry:
        raise RuntimeLineageReject(
            "runtime_terminal_cache_entry_cas_mismatch"
        )
    current_entry = _validate_cached_affine_expression_current(
        cache_entry, terminal_layer_id, graph, registry
    )
    if current_entry.terms is not snapshot.terms:
        raise RuntimeLineageReject("runtime_terminal_term_tuple_cas_mismatch")
    if type(snapshot.expected_terms) is not tuple or not snapshot.expected_terms:
        raise RuntimeLineageReject("runtime_terminal_expected_terms_missing")
    current_expected: list[_ExpectedTerminalTerm] = []
    current_events: list[tuple[RuntimeLineageEvent, ...]] = []
    for lineage in current_entry.terms:
        current = _validate_operand_lineage_current(
            lineage, graph, registry
        )
        current_expected.append(
            _terminal_term_record(
                source=current.source,
                operators=current.operators,
                events=current.events,
                terminal_layer_id=terminal_layer_id,
            )
        )
        current_events.append(current.events)
    if len(current_expected) != len(snapshot.expected_terms) or any(
        not _terminal_term_identity_equal(left, right)
        for left, right in zip(
            current_expected, snapshot.expected_terms, strict=True
        )
    ):
        raise RuntimeLineageReject(
            "runtime_terminal_expression_snapshot_cas_mismatch"
        )
    common_add_layer_id = _common_last_add_layer_id(current_events)
    expected_bias = _expected_bias_from_common_add(
        registry, graph, tuple(current_events), common_add_layer_id
    )
    cached_bias = _numeric_snapshot(
        current_entry.bias, "runtime_terminal_cached_bias", ndim=1
    ).canonical
    if (
        expected_bias.shape != cached_bias.shape
        or not np.array_equal(expected_bias, cached_bias)
    ):
        raise RuntimeLineageReject(
            "runtime_terminal_cache_bias_semantic_mismatch"
        )
    return snapshot


def _record_private_runtime_terminal_expression(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    registry: object,
    terminal_layer_id: int,
) -> None:
    """Snapshot the owning cache's complete expression at a proven boundary."""

    graph = _freeze_graph(layers, preds, succs)
    private_registry = _validate_registry_current(
        registry, graph, require_sealed=False
    )
    if private_registry.terminal_expression_snapshot is not None:
        raise RuntimeLineageReject(
            "runtime_terminal_expression_snapshot_duplicate"
        )
    layer_id = _exact_int(terminal_layer_id, "runtime_terminal_layer")
    if layer_id >= len(graph.layers):
        raise RuntimeLineageReject("runtime_terminal_layer_unknown")
    consumer_id = _terminal_boundary_consumer(graph, layer_id)
    cache_entry = private_registry.arena.owner_affine_cache.get(layer_id)
    if cache_entry is None:
        raise RuntimeLineageReject("runtime_terminal_cache_entry_missing")
    current_entry = _validate_cached_affine_expression_current(
        cache_entry, layer_id, graph, private_registry
    )
    expected_terms = tuple(
        _terminal_term_record(
            source=lineage.source,
            operators=lineage.operators,
            events=lineage.events,
            terminal_layer_id=layer_id,
        )
        for lineage in current_entry.terms
    )
    snapshot = _RegisteredTerminalExpression(
        layer_id,
        consumer_id,
        current_entry,
        current_entry.terms,
        expected_terms,
    )
    _validate_registered_terminal_expression_current(
        snapshot, graph, private_registry
    )
    private_registry.terminal_expression_snapshot = snapshot


def _validate_custody(
    events: tuple[RuntimeLineageEvent, ...],
    *,
    source: object,
    registry: _RuntimeLineageRegistry,
) -> _LineageCustody:
    source_boundaries = registry.source_boundaries
    if type(events) is not tuple or not events:
        raise RuntimeLineageReject("runtime_lineage_events_missing")
    if any(type(event) is not RuntimeLineageEvent for event in events):
        raise RuntimeLineageReject("runtime_lineage_event_type")
    for event in events:
        _runtime_event_primitive_signature(event)
    custody = _lineage_custody_guard(events[0]._custody)
    if any(event._custody is not custody for event in events):
        raise RuntimeLineageReject("runtime_lineage_mixed_custody")
    if hashlib.sha256(_event_canonical_payload(events)).hexdigest() != custody.canonical_sha256:
        raise RuntimeLineageReject("runtime_lineage_custody_digest_mismatch")
    if len(events) != len(custody.occurrence_objects) or any(
        event.occurrence_layer is not reference
        for event, reference in zip(events, custody.occurrence_objects, strict=True)
    ):
        raise RuntimeLineageReject("runtime_lineage_occurrence_custody_mismatch")
    if any(
        event.operator_occurrence is not reference
        for event, reference in zip(events, custody.operator_objects, strict=True)
    ):
        raise RuntimeLineageReject("runtime_lineage_operator_custody_mismatch")
    path = _path_from_events(events)
    if custody.entry_source_layer_id != path[0]:
        raise RuntimeLineageReject("runtime_lineage_source_layer_custody_mismatch")
    if custody.source_object is not source:
        raise RuntimeLineageReject("runtime_lineage_source_object_custody_mismatch")
    if custody.arena is not registry.arena:
        raise RuntimeLineageReject("runtime_lineage_arena_custody_mismatch")
    if source_boundaries.source_at(path[0]) is not source:
        raise RuntimeLineageReject("runtime_source_boundary_identity_mismatch")
    return custody


def _path_from_events(events: tuple[RuntimeLineageEvent, ...]) -> tuple[int, ...]:
    if type(events) is not tuple or not events:
        raise RuntimeLineageReject("runtime_lineage_events_missing")
    for event in events:
        _runtime_event_primitive_signature(event)
    first = events[0]
    selected = _exact_int(first.selected_input, "first_selected_input")
    if selected >= len(first.ordered_predecessor_layer_ids):
        raise RuntimeLineageReject("first_selected_input_out_of_range")
    entry = first.ordered_predecessor_layer_ids[selected]
    path = (entry, *(event.occurrence_layer_id for event in events))
    if len(path) != len(set(path)):
        raise RuntimeLineageReject("duplicate_runtime_graph_event")
    return path


def _same_event_observation(
    observed: tuple[RuntimeLineageEvent, ...], current: tuple[RuntimeLineageEvent, ...]
) -> None:
    if type(observed) is not tuple or type(current) is not tuple:
        raise RuntimeLineageReject("runtime_lineage_events_not_tuple")
    if len(observed) != len(current):
        raise RuntimeLineageReject("runtime_lineage_event_count_changed")
    for left, right in zip(observed, current, strict=True):
        left_signature = _runtime_event_primitive_signature(left)
        right_signature = _runtime_event_primitive_signature(right)
        if left_signature != right_signature:
            raise RuntimeLineageReject("runtime_lineage_current_snapshot_mismatch")
        if left.occurrence_layer is not right.occurrence_layer:
            raise RuntimeLineageReject("runtime_lineage_occurrence_object_mismatch")
        if left.operator_occurrence is not right.operator_occurrence:
            raise RuntimeLineageReject("runtime_lineage_operator_occurrence_mismatch")


def _graph_event(event: RuntimeLineageEvent, graph: _GraphSnapshot) -> c3.GraphEventEvidence:
    occurrence_token = ("layer", graph.graph_sha256, event.occurrence_layer_id)
    producer_tokens = tuple(
        ("layer", graph.graph_sha256, layer_id)
        for layer_id in event.ordered_predecessor_layer_ids
    )
    input_tokens = tuple(("vars", token) for token in event.input_var_tokens)
    output_token = ("vars", event.output_var_token)
    bias = ()
    if event.kind == "BIAS":
        bias = _bias_values(graph.layers[event.occurrence_layer_id])
    return c3.GraphEventEvidence(
        kind=event.kind,
        occurrence_token=occurrence_token,
        producer_occurrence_tokens=producer_tokens,
        graph_predecessor_occurrence_tokens=producer_tokens,
        input_value_tokens=input_tokens,
        output_value_token=output_token,
        operator_position=event.operator_position,
        selected_input=event.selected_input,
        operator_occurrence=event.operator_occurrence,
        bias_payload=bias,
    )


def _object_guard(value: object, name: str) -> tuple[bytes, tuple[object, ...]]:
    refs: list[object] = [value]
    if type(value) in (scipy_sparse.csr_matrix, scipy_sparse.csc_matrix):
        state, shape, raw_data, raw_indices, raw_indptr = (
            _compressed_sparse_raw_state(value, name)
        )
        data = _raw_numeric_array_payload(raw_data, f"{name}_sparse_data")
        indices = _raw_numeric_array_payload(
            raw_indices, f"{name}_sparse_indices"
        )
        indptr = _raw_numeric_array_payload(
            raw_indptr, f"{name}_sparse_indptr"
        )
        format_name = "csr" if type(value) is scipy_sparse.csr_matrix else "csc"
        header = _metadata_payload(
            [format_name, shape]
        )
        payload = b"".join(
            len(part).to_bytes(8, "big") + part
            for part in (header, data, indices, indptr)
        )
        refs.extend(
            (
                state,
                state["_shape"],
                raw_data,
                raw_indices,
                raw_indptr,
            )
        )
        return payload, tuple(refs)
    if type(value) is np.ndarray or (
        torch is not None and type(value) is torch.Tensor
    ):
        return _numeric_snapshot(value, name).payload, tuple(refs)
    if value is None or type(value) in (str, bytes, int, float, bool):
        try:
            payload = repr(value).encode("utf-8")
        except Exception as exc:
            raise RuntimeLineageReject(f"{name}_snapshot_failed") from exc
        return payload, tuple(refs)
    raise RuntimeLineageReject(f"{name}_unsupported_mutable_type")


def _registry_authority_guard(
    registry: _RuntimeLineageRegistry,
) -> tuple[bytes, tuple[object, ...]]:
    """CAS the owning arena and operand-cache containers by identity."""

    if type(registry) is not _RuntimeLineageRegistry or registry.marker is not _CUSTODY_MARKER:
        raise RuntimeLineageReject("runtime_registry_authority_type")
    _exact_bool(registry.sealed, "runtime_registry_sealed")
    _exact_sha256(registry.graph_sha256, "runtime_registry_graph_sha256")
    if type(registry.graph_reference_objects) is not tuple:
        raise RuntimeLineageReject("runtime_registry_graph_references_type")
    arena = registry.arena
    if type(arena) is not _RuntimeLineageArena or arena.marker is not _ARENA_MARKER:
        raise RuntimeLineageReject("runtime_registry_arena_type")
    _exact_bool(arena.closed, "runtime_arena_closed")
    _exact_sha256(arena.graph_sha256, "runtime_arena_graph_sha256")
    if (
        type(arena.graph_reference_objects) is not tuple
        or type(arena.factor_allocator_references) is not tuple
        or type(arena.affine_entries) is not list
        or type(arena.affine_payloads) is not list
        or type(arena.affine_reference_snapshots) is not list
        or type(arena.source_entries) is not tuple
        or type(arena.sealed_source_entries) is not tuple
        or type(arena.source_payloads) is not tuple
        or type(arena.source_reference_snapshots) is not tuple
    ):
        raise RuntimeLineageReject("runtime_registry_arena_container_type")
    _exact_bytes(
        arena.factor_allocator_payload,
        "runtime_arena_factor_allocator_payload",
    )
    if arena.frame_payload is not None:
        _exact_bytes(arena.frame_payload, "runtime_arena_frame_payload")
    if any(type(payload) is not bytes for payload in arena.affine_payloads):
        raise RuntimeLineageReject("runtime_affine_cache_payload_type")
    if any(type(payload) is not bytes for payload in arena.source_payloads):
        raise RuntimeLineageReject("runtime_arena_source_payload_type")
    if any(
        type(entry) is not tuple
        or len(entry) != 2
        or type(entry[0]) is not int
        or entry[0] < 0
        for entry in (*arena.affine_entries, *arena.source_entries)
    ):
        raise RuntimeLineageReject("runtime_registry_arena_entry_type")
    if any(
        type(references) is not tuple
        for references in (
            *arena.affine_reference_snapshots,
            *arena.source_reference_snapshots,
        )
    ):
        raise RuntimeLineageReject("runtime_registry_reference_snapshot_type")
    if type(registry.add_operand_snapshots) not in (list, tuple):
        raise RuntimeLineageReject("runtime_add_snapshot_container_type")
    source_boundaries = _source_boundary_snapshot_guard(
        registry.source_boundaries
    )
    refs: list[object] = [
        registry,
        arena,
        arena.nonce,
        arena.owner_cache,
        arena.owner_cache_identity,
        arena.owner_affine_cache,
        arena.owner_affine_cache_identity,
        arena.factor_allocator,
        arena.factor_allocator_identity,
        arena.factor_allocator_references,
        arena.affine_entries,
        arena.affine_payloads,
        arena.affine_reference_snapshots,
        arena.source_entries,
        arena.sealed_source_entries,
        arena.source_payloads,
        arena.source_reference_snapshots,
        registry.graph_reference_objects,
        arena.graph_reference_objects,
        source_boundaries,
        source_boundaries.by_layer,
        source_boundaries.reference_objects,
        registry.add_operand_snapshots,
        registry.terminal_expression_snapshot,
    ]
    records: list[object] = [
        registry.graph_sha256,
        arena.graph_sha256,
        arena.frame_payload.hex() if arena.frame_payload is not None else None,
        source_boundaries.key_payload.hex(),
        tuple(
            layer_id for layer_id, _ in source_boundaries.by_layer
        ),
        tuple(payload.hex() for payload in arena.source_payloads),
        arena.factor_allocator_payload.hex(),
    ]
    refs.extend(arena.factor_allocator_references)
    for reference_snapshot in arena.source_reference_snapshots:
        refs.append(reference_snapshot)
        refs.extend(reference_snapshot)
    refs.extend(source_boundaries.reference_objects)
    refs.extend(
        source for _, source in source_boundaries.by_layer
    )
    records.append(
        tuple(layer_id for layer_id, _ in arena.affine_entries)
    )
    records.append(tuple(payload.hex() for payload in arena.affine_payloads))
    for reference_snapshot in arena.affine_reference_snapshots:
        refs.append(reference_snapshot)
        refs.extend(reference_snapshot)
    for layer_id, cache_entry in arena.affine_entries:
        if type(cache_entry) is not _RuntimeAffineExprCacheView:
            raise RuntimeLineageReject("runtime_affine_cache_entry_type")
        _exact_int(cache_entry.n_out, "runtime_cached_expression_n_out", minimum=1)
        _exact_int(cache_entry.frame_id, "runtime_cached_expression_frame")
        if type(cache_entry.terms) is not tuple:
            raise RuntimeLineageReject("runtime_affine_cache_terms_container")
        refs.extend(
            (
                cache_entry,
                cache_entry.terms,
                cache_entry.bias,
                cache_entry.frame_id,
            )
        )
        refs.extend(cache_entry.terms)
        records.append(
            (
                layer_id,
                cache_entry.n_out,
                _stable_token_payload(
                    cache_entry.frame_id,
                    "runtime_cached_expression_frame",
                ).hex(),
                _numeric_snapshot(
                    cache_entry.bias,
                    "runtime_cached_expression_bias",
                    ndim=1,
                ).payload.hex(),
                len(cache_entry.terms),
            )
        )
    for snapshot in registry.add_operand_snapshots:
        snapshot = _registered_add_snapshot_schema(snapshot)
        refs.extend(
            (
                snapshot,
                snapshot.operand_cache_entries,
                snapshot.operands,
                snapshot.expected_prefixes,
            )
        )
        records.append(
            (
                snapshot.add_layer_id,
                tuple(len(terms) for terms in snapshot.operands),
                tuple(
                    (
                        prefix.selected_input,
                        prefix.entry_layer_id,
                        prefix.canonical_sha256,
                    )
                    for prefix in snapshot.expected_prefixes
                ),
                tuple(
                    (
                        entry.n_out,
                        _stable_token_payload(
                            entry.frame_id, "runtime_cached_expression_frame"
                        ).hex(),
                        _numeric_snapshot(
                            entry.bias,
                            "runtime_cached_expression_bias",
                            ndim=1,
                        ).payload.hex(),
                    )
                    for entry in snapshot.operand_cache_entries
                ),
            )
        )
        for operand_terms in snapshot.operands:
            refs.append(operand_terms)
            for lineage in operand_terms:
                lineage, _, custody = _runtime_operand_lineage_schema(
                    lineage
                )
                refs.extend(
                    (
                        lineage,
                        lineage.source,
                        lineage.operators,
                        lineage.events,
                        custody,
                        *custody.occurrence_objects,
                        *custody.operator_objects,
                    )
                )
        for cache_entry in snapshot.operand_cache_entries:
            refs.extend(
                (
                    cache_entry,
                    cache_entry.terms,
                    cache_entry.bias,
                    cache_entry.frame_id,
                )
            )
        for prefix in snapshot.expected_prefixes:
            prefix = _expected_add_prefix_guard(prefix)
            refs.extend(
                (
                    prefix,
                    prefix.source,
                    prefix.operator_prefix,
                    prefix.occurrence_prefix,
                    *prefix.operator_prefix,
                    *prefix.occurrence_prefix,
                )
            )
    terminal = registry.terminal_expression_snapshot
    if terminal is None:
        records.append(("terminal_expression", None))
    else:
        terminal = _registered_terminal_snapshot_schema(terminal)
        refs.extend(
            (
                terminal,
                terminal.cache_entry,
                terminal.terms,
                terminal.expected_terms,
                terminal.cache_entry.bias,
                terminal.cache_entry.frame_id,
            )
        )
        refs.extend(terminal.terms)
        records.append(
            (
                "terminal_expression",
                terminal.terminal_layer_id,
                terminal.terminal_consumer_layer_id,
                terminal.cache_entry.n_out,
                _stable_token_payload(
                    terminal.cache_entry.frame_id,
                    "runtime_terminal_cached_frame",
                ).hex(),
                _numeric_snapshot(
                    terminal.cache_entry.bias,
                    "runtime_terminal_cached_bias",
                    ndim=1,
                ).payload.hex(),
                len(terminal.expected_terms),
                tuple(
                    (
                        expected.terminal_layer_id,
                        expected.entry_layer_id,
                        expected.canonical_sha256,
                    )
                    for expected in terminal.expected_terms
                ),
            )
        )
        for expected in terminal.expected_terms:
            expected = _expected_terminal_term_guard(expected)
            refs.extend(
                (
                    expected,
                    expected.source,
                    expected.operators,
                    expected.occurrences,
                    *expected.operators,
                    *expected.occurrences,
                )
            )
    return _metadata_payload(records), tuple(refs)


def _checked_bias_add(
    left: np.ndarray, right: np.ndarray, reason: str
) -> np.ndarray:
    if left.shape != right.shape:
        raise RuntimeLineageReject(f"{reason}_shape_mismatch")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(left + right, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise RuntimeLineageReject(f"{reason}_nonfinite")
    return result


def _unmasked_conv_bias_matvec(
    operator_value: ImplicitConv2DOp, bias: np.ndarray
) -> np.ndarray:
    """Propagate global bias through Conv geometry, never through row support."""

    try:
        unmasked = ImplicitConv2DOp(
            np.array(operator_value._kernel, copy=True),
            tuple(operator_value._input_shape),
            stride=tuple(operator_value._stride),
            padding=tuple(operator_value._padding),
            dilation=tuple(operator_value._dilation),
            groups=int(operator_value._groups),
            row_mask=None,
        )
        result = ImplicitConv2DOp.matvec(unmasked, bias)
    except Exception as exc:
        raise RuntimeLineageReject(
            "runtime_bias_conv_propagation_failed"
        ) from exc
    expected_shape = (_exact_int(operator_value.shape[0], "runtime_bias_conv_rows"),)
    result = np.asarray(result, dtype=np.float64).reshape(-1)
    if result.shape != expected_shape:
        raise RuntimeLineageReject(
            "runtime_bias_conv_propagation_shape_mismatch"
        )
    if not np.all(np.isfinite(result)):
        raise RuntimeLineageReject("runtime_bias_conv_propagation_nonfinite")
    return result


def _expected_bias_from_common_add(
    registry: _RuntimeLineageRegistry,
    graph: _GraphSnapshot,
    term_events: Sequence[tuple[RuntimeLineageEvent, ...]],
    common_add_layer_id: int,
) -> np.ndarray:
    snapshots = tuple(
        snapshot
        for snapshot in registry.add_operand_snapshots
        if snapshot.add_layer_id == common_add_layer_id
    )
    if len(snapshots) != 1:
        raise RuntimeLineageReject(
            "runtime_add_operand_snapshot_missing_or_duplicate"
        )
    snapshot = snapshots[0]
    if len(snapshot.operand_cache_entries) != 2:
        raise RuntimeLineageReject("runtime_add_operand_cache_entry_arity")
    left = _numeric_snapshot(
        snapshot.operand_cache_entries[0].bias,
        "runtime_add_left_cached_bias",
        ndim=1,
    ).canonical
    right = _numeric_snapshot(
        snapshot.operand_cache_entries[1].bias,
        "runtime_add_right_cached_bias",
        ndim=1,
    ).canonical
    bias = _checked_bias_add(left, right, "runtime_add_cached_bias")
    if not term_events:
        raise RuntimeLineageReject("runtime_common_suffix_missing")
    reference_events = term_events[0]
    add_indices = tuple(
        index
        for index, event in enumerate(reference_events)
        if event.kind == "ADD"
        and event.occurrence_layer_id == common_add_layer_id
    )
    if len(add_indices) != 1:
        raise RuntimeLineageReject("runtime_common_add_occurrence_ambiguous")
    for event in reference_events[add_indices[0] + 1 :]:
        if event.kind == "BIAS":
            bias = _checked_bias_add(
                bias,
                np.asarray(
                    _bias_values(graph.layers[event.occurrence_layer_id]),
                    dtype=np.float64,
                ),
                "runtime_suffix_bias",
            )
            continue
        if event.kind == "SCALE":
            operator_value = event.operator_occurrence
            if type(operator_value) is not DiagonalLinearOp:
                raise RuntimeLineageReject(
                    "runtime_bias_scale_operator_type_mismatch"
                )
            try:
                bias = DiagonalLinearOp.matvec(operator_value, bias)
            except Exception as exc:
                raise RuntimeLineageReject(
                    "runtime_bias_scale_propagation_failed"
                ) from exc
            continue
        if event.kind == "CONV":
            operator_value = event.operator_occurrence
            if type(operator_value) is not ImplicitConv2DOp:
                raise RuntimeLineageReject(
                    "runtime_bias_conv_operator_type_mismatch"
                )
            bias = _unmasked_conv_bias_matvec(operator_value, bias)
            continue
        raise RuntimeLineageReject(
            "runtime_bias_suffix_contains_unexpected_add"
        )
    if not np.all(np.isfinite(bias)):
        raise RuntimeLineageReject("runtime_expected_bias_nonfinite")
    return np.asarray(bias, dtype=np.float64).reshape(-1)


def _derive_current(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    registry: object,
    expr: RuntimeLineageExprView,
) -> _DerivedState:
    if type(expr) is not RuntimeLineageExprView:
        raise RuntimeLineageReject("runtime_expression_type")
    if type(expr.terms) is not tuple or not expr.terms:
        raise RuntimeLineageReject("runtime_expression_terms_missing")
    expression_n_out = _exact_int(
        expr.n_out, "runtime_expression_n_out", minimum=1
    )
    expression_frame_payload = _stable_token_payload(
        expr.frame_id, "runtime_expression_frame"
    )
    bias_snapshot = _numeric_snapshot(
        expr.bias, "runtime_expression_bias", ndim=1
    )
    if int(bias_snapshot.canonical.size) != expression_n_out:
        raise RuntimeLineageReject("runtime_expression_bias_width_mismatch")
    graph = _freeze_graph(layers, preds, succs)
    private_registry = _validate_registry_current(
        registry, graph, require_sealed=True
    )
    source_boundaries = private_registry.source_boundaries
    registry_payload, registry_refs = _registry_authority_guard(
        private_registry
    )
    c3_terms: list[c3.C3AffineTermView] = []
    current_events: list[tuple[RuntimeLineageEvent, ...]] = []
    refs: list[object] = [
        expr,
        expr.terms,
        expr.bias,
        *graph.reference_objects,
        *source_boundaries.reference_objects,
        private_registry,
        *registry_refs,
    ]
    guards: list[bytes] = [
        graph.graph_sha256.encode("ascii"),
        source_boundaries.key_payload,
        registry_payload,
    ]
    operator_by_layer: dict[int, object] = {}
    layer_by_operator_identity: list[tuple[object, int]] = []
    last_add_and_suffix: tuple[int, tuple[int, ...]] | None = None
    actual_prefixes_by_add: dict[int, list[_ExpectedAddPrefix]] = {}
    candidate_terminal_components: list[
        tuple[
            object,
            tuple[object, ...],
            tuple[RuntimeLineageEvent, ...],
        ]
    ] = []

    for term_index, term in enumerate(expr.terms):
        if type(term) is not RuntimeLineageTermView:
            raise RuntimeLineageReject("runtime_term_type")
        if type(term.operators) is not tuple:
            raise RuntimeLineageReject("runtime_term_operators_not_tuple")
        _validate_custody(
            term.events,
            source=term.source,
            registry=private_registry,
        )
        path = _path_from_events(term.events)
        rederived = _record_path(
            graph,
            private_registry,
            term.source,
            path,
            term.operators,
        )
        _same_event_observation(term.events, rederived)
        source_payload, width = _source_boundary_payload(
            term.source,
            graph.layers[path[0]],
            expression_frame_payload=expression_frame_payload,
        )
        shape_chain: list[tuple[int, int]] = []
        for operator_index, runtime_operator in enumerate(term.operators):
            raw_shape = getattr(runtime_operator, "shape", None)
            if type(raw_shape) is not tuple or len(raw_shape) != 2:
                raise RuntimeLineageReject("runtime_operator_shape_invalid")
            rows = _exact_int(
                raw_shape[0], f"runtime_operator_{operator_index}_rows"
            )
            columns = _exact_int(
                raw_shape[1], f"runtime_operator_{operator_index}_columns"
            )
            if columns != width:
                raise RuntimeLineageReject(
                    "runtime_operator_shape_chain_mismatch"
                )
            width = rows
            shape_chain.append((rows, columns))
        if width != expression_n_out:
            raise RuntimeLineageReject("runtime_term_output_width_mismatch")
        add_positions = tuple(index for index, event in enumerate(rederived) if event.kind == "ADD")
        if not add_positions:
            raise RuntimeLineageReject("runtime_common_add_missing")
        last_add = add_positions[-1]
        suffix = tuple(event.occurrence_layer_id for event in rederived[last_add:])
        marker = (rederived[last_add].occurrence_layer_id, suffix)
        if last_add_and_suffix is None:
            last_add_and_suffix = marker
        elif marker != last_add_and_suffix:
            raise RuntimeLineageReject("runtime_common_add_or_suffix_occurrence_mismatch")
        for add_event_index in add_positions:
            prefix = _prefix_record(
                source=term.source,
                entry_layer_id=path[0],
                operators=term.operators,
                events=rederived,
                add_event_index=add_event_index,
            )
            actual_prefixes_by_add.setdefault(prefix.add_layer_id, []).append(
                prefix
            )

        for event in rederived:
            if event.operator_occurrence is None:
                continue
            prior = operator_by_layer.setdefault(event.occurrence_layer_id, event.operator_occurrence)
            if prior is not event.operator_occurrence:
                raise RuntimeLineageReject("shared_suffix_operator_object_mismatch")
            for seen_operator, seen_layer in layer_by_operator_identity:
                if event.operator_occurrence is seen_operator and seen_layer != event.occurrence_layer_id:
                    raise RuntimeLineageReject("operator_object_aliases_graph_occurrences")
            layer_by_operator_identity.append((event.operator_occurrence, event.occurrence_layer_id))

        first = rederived[0]
        entry_id = first.ordered_predecessor_layer_ids[first.selected_input]
        certificate = c3.OrderedPathCertificate(
            schema=c3.C3_CERTIFICATE_SCHEMA,
            entry_producer_occurrence_token=("layer", graph.graph_sha256, entry_id),
            entry_value_token=("vars", first.input_var_tokens[first.selected_input]),
            events=tuple(_graph_event(event, graph) for event in rederived),
        )
        c3_terms.append(c3.C3AffineTermView(term.source, term.operators, certificate))
        current_events.append(rederived)
        candidate_terminal_components.append(
            (term.source, term.operators, rederived)
        )
        refs.extend((term, term.source, term.operators, term.events))
        refs.extend(event.occurrence_layer for event in rederived)
        for event in rederived:
            if event.operator_occurrence is not None:
                refs.extend(
                    _live_operator_reference_objects(
                        event.operator_occurrence
                    )
                )
        guards.append(_event_canonical_payload(rederived))
        guards.extend(
            (
                source_payload,
                _metadata_payload(shape_chain),
            )
        )
        refs.append(term.source.frame_id)
        for source_field in _SOURCE_FIELDS:
            if not hasattr(term.source, source_field):
                raise RuntimeLineageReject("runtime_source_semantic_field_missing")
            payload, field_refs = _object_guard(
                getattr(term.source, source_field), f"source_{term_index}_{source_field}"
            )
            guards.append(payload)
            refs.extend(field_refs)

    assert last_add_and_suffix is not None
    for add_layer_id, prefixes in actual_prefixes_by_add.items():
        _require_exact_add_prefix_multiset(
            private_registry, add_layer_id, tuple(prefixes)
        )
    terminal_snapshot = private_registry.terminal_expression_snapshot
    if terminal_snapshot is None:
        raise RuntimeLineageReject(
            "runtime_terminal_expression_snapshot_missing"
        )
    actual_terminal_terms = tuple(
        _terminal_term_record(
            source=source,
            operators=operators,
            events=events,
            terminal_layer_id=terminal_snapshot.terminal_layer_id,
        )
        for source, operators, events in candidate_terminal_components
    )
    _require_exact_terminal_term_multiset(
        terminal_snapshot, actual_terminal_terms
    )
    terminal_entry = terminal_snapshot.cache_entry
    terminal_frame_payload = _stable_token_payload(
        terminal_entry.frame_id, "runtime_terminal_cached_frame"
    )
    terminal_bias_snapshot = _numeric_snapshot(
        terminal_entry.bias, "runtime_terminal_cached_bias", ndim=1
    )
    if expression_n_out != terminal_entry.n_out:
        raise RuntimeLineageReject(
            "runtime_terminal_expression_output_width_mismatch"
        )
    if expression_frame_payload != terminal_frame_payload:
        raise RuntimeLineageReject(
            "runtime_terminal_expression_frame_mismatch"
        )
    if (
        terminal_bias_snapshot.canonical.shape
        != bias_snapshot.canonical.shape
        or not np.array_equal(
            terminal_bias_snapshot.canonical, bias_snapshot.canonical
        )
    ):
        raise RuntimeLineageReject(
            "runtime_terminal_expression_bias_mismatch"
        )
    if expr.bias is not terminal_entry.bias:
        raise RuntimeLineageReject(
            "runtime_terminal_expression_bias_identity_mismatch"
        )
    expected_bias = _expected_bias_from_common_add(
        private_registry,
        graph,
        tuple(current_events),
        last_add_and_suffix[0],
    )
    if (
        expected_bias.shape != bias_snapshot.canonical.shape
        or not np.array_equal(expected_bias, bias_snapshot.canonical)
    ):
        raise RuntimeLineageReject("runtime_expression_bias_semantic_mismatch")
    guards.extend(
        (
            bias_snapshot.payload,
            terminal_bias_snapshot.payload,
            _numeric_snapshot(
                expected_bias, "runtime_expected_expression_bias", ndim=1
            ).payload,
            expression_frame_payload,
            expression_n_out.to_bytes(8, "big"),
        )
    )
    refs.extend((expr.bias, expr.frame_id))
    c3_expr = c3.C3AffineExprView(tuple(c3_terms), expr.bias, expr.n_out, expr.frame_id)
    state_digest = hashlib.sha256()
    for guard in guards:
        state_digest.update(len(guard).to_bytes(8, "big"))
        state_digest.update(guard)
    return _DerivedState(c3_expr, tuple(current_events), state_digest.hexdigest(), tuple(refs))


def _same_references(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return len(left) == len(right) and all(
        first is second for first, second in zip(left, right, strict=True)
    )


def _make_array_read_only(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _detach_sparse_source(source: object) -> _DetachedSourceAuthorityProxy:
    if type(source) is not SparseHZono:
        raise RuntimeLineageReject("runtime_planner_source_type")
    semantic_payload, _ = _snapshot_sparse_hz_source(source)
    return _DetachedSourceAuthorityProxy(
        frame_id=source.frame_id,
        exact=True,
        semantic_sha256=hashlib.sha256(semantic_payload).hexdigest(),
    )


def _detach_runtime_operator(operator_value: object) -> object:
    if type(operator_value) is DiagonalLinearOp:
        detached = DiagonalLinearOp(
            np.array(operator_value._diagonal, copy=True)
        )
        _make_array_read_only(detached._diagonal)
        return detached
    if type(operator_value) is ImplicitConv2DOp:
        row_mask = (
            None
            if operator_value._row_mask is None
            else np.array(operator_value._row_mask, dtype=bool, copy=True)
        )
        detached = ImplicitConv2DOp(
            np.array(operator_value._kernel, copy=True),
            tuple(operator_value._input_shape),
            stride=tuple(operator_value._stride),
            padding=tuple(operator_value._padding),
            dilation=tuple(operator_value._dilation),
            groups=int(operator_value._groups),
            row_mask=row_mask,
        )
        _make_array_read_only(detached._kernel)
        if detached._row_mask is not None:
            detached._row_mask = np.array(
                detached._row_mask, dtype=bool, order="C", copy=True
            )
            _make_array_read_only(detached._row_mask)
        return detached
    raise RuntimeLineageReject("runtime_planner_operator_type")


def _detach_c3_expression(
    expression: c3.C3AffineExprView,
) -> c3.C3AffineExprView:
    source_pairs: list[
        tuple[object, _DetachedSourceAuthorityProxy]
    ] = []
    operator_pairs: list[tuple[object, object]] = []

    def detached_source(source: object) -> _DetachedSourceAuthorityProxy:
        for live, detached in source_pairs:
            if live is source:
                return detached
        detached = _detach_sparse_source(source)
        source_pairs.append((source, detached))
        return detached

    def detached_operator(operator_value: object) -> object:
        for live, detached in operator_pairs:
            if live is operator_value:
                return detached
        detached = _detach_runtime_operator(operator_value)
        operator_pairs.append((operator_value, detached))
        return detached

    terms: list[c3.C3AffineTermView] = []
    for term in expression.terms:
        if type(term) is not c3.C3AffineTermView or type(
            term.certificate
        ) is not c3.OrderedPathCertificate:
            raise RuntimeLineageReject("runtime_planner_certificate_type")
        operators = tuple(
            detached_operator(operator_value)
            for operator_value in term.operators
        )
        events = tuple(
            replace(
                event,
                operator_occurrence=(
                    None
                    if event.operator_occurrence is None
                    else detached_operator(event.operator_occurrence)
                ),
            )
            for event in term.certificate.events
        )
        certificate = replace(term.certificate, events=events)
        terms.append(
            c3.C3AffineTermView(
                detached_source(term.source), operators, certificate
            )
        )
    bias = _make_array_read_only(
        np.array(
            _numeric_snapshot(
                expression.bias, "runtime_planner_expression_bias", ndim=1
            ).canonical,
            dtype=np.float64,
            copy=True,
        )
    )
    return c3.C3AffineExprView(
        tuple(terms), bias, expression.n_out, expression.frame_id
    )


def _exact_float64_read_only_array(
    value: object, *, name: str, ndim: int
) -> _NumericSnapshot:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.ndim != ndim
        or not value.flags.c_contiguous
        or not value.flags.owndata
        or value.flags.writeable
    ):
        raise RuntimeLineageReject(f"{name}_storage_schema")
    return _numeric_snapshot(value, name, ndim=ndim)


def _exact_geometry_tuple(
    value: object, *, name: str, length: int, minimum: int
) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or len(value) != length
        or any(type(item) is not int or item < minimum for item in value)
    ):
        raise RuntimeLineageReject(f"{name}_schema")
    return value


def _detached_array_digest(value: np.ndarray) -> bytes:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(tuple(int(item) for item in value.shape)).encode())
    digest.update(value.tobytes(order="C"))
    return digest.digest()


def _detached_operator_guard(
    operator_value: object,
) -> tuple[bytes, tuple[object, ...]]:
    if type(operator_value) is DiagonalLinearOp:
        state = _operator_raw_state(
            operator_value,
            expected_keys=_DIAGONAL_OPERATOR_STATE_KEYS,
            name="runtime_detached_diagonal",
        )
        diagonal = _exact_float64_read_only_array(
            state["_diagonal"],
            name="runtime_detached_diagonal",
            ndim=1,
        )
        if state["_diagonal"].size <= 0:
            raise RuntimeLineageReject("runtime_detached_diagonal_empty")
        expected_shape = (
            int(state["_diagonal"].size),
            int(state["_diagonal"].size),
        )
        expected_content_key = (
            "diagonal_linear_op_v1",
            expected_shape,
            _detached_array_digest(state["_diagonal"]),
        )
        content_key = state["_content_key"]
        if (
            type(content_key) is not tuple
            or len(content_key) != 3
            or type(content_key[0]) is not str
            or type(content_key[1]) is not tuple
            or len(content_key[1]) != 2
            or any(type(item) is not int for item in content_key[1])
            or type(content_key[2]) is not bytes
            or content_key != expected_content_key
        ):
            raise RuntimeLineageReject(
                "runtime_detached_diagonal_content_cache_mismatch"
            )
        metadata = _metadata_payload(
            ["SCALE", expected_shape, "owned_c_float64_read_only"]
        )
        return (
            len(metadata).to_bytes(8, "big")
            + metadata
            + diagonal.payload,
            (
                operator_value,
                state,
                state["_diagonal"],
                state["_content_key"],
            ),
        )
    if type(operator_value) is ImplicitConv2DOp:
        state = _operator_raw_state(
            operator_value,
            expected_keys=_CONV_OPERATOR_STATE_KEYS,
            name="runtime_detached_conv",
        )
        kernel = _exact_float64_read_only_array(
            state["_kernel"],
            name="runtime_detached_conv_kernel",
            ndim=4,
        )
        if any(dimension <= 0 for dimension in state["_kernel"].shape):
            raise RuntimeLineageReject(
                "runtime_detached_conv_kernel_shape"
            )
        input_shape = _exact_geometry_tuple(
            state["_input_shape"],
            name="runtime_detached_conv_input_shape",
            length=4,
            minimum=1,
        )
        output_shape = _exact_geometry_tuple(
            state["_output_shape"],
            name="runtime_detached_conv_output_shape",
            length=4,
            minimum=1,
        )
        stride = _exact_geometry_tuple(
            state["_stride"],
            name="runtime_detached_conv_stride",
            length=2,
            minimum=1,
        )
        padding = _exact_geometry_tuple(
            state["_padding"],
            name="runtime_detached_conv_padding",
            length=2,
            minimum=0,
        )
        dilation = _exact_geometry_tuple(
            state["_dilation"],
            name="runtime_detached_conv_dilation",
            length=2,
            minimum=1,
        )
        groups = state["_groups"]
        if type(groups) is not int or groups < 1:
            raise RuntimeLineageReject("runtime_detached_conv_groups_schema")
        batch, channels, height, width = input_shape
        out_channels, in_per_group, kernel_height, kernel_width = (
            state["_kernel"].shape
        )
        if channels != in_per_group * groups or out_channels % groups:
            raise RuntimeLineageReject(
                "runtime_detached_conv_group_geometry_mismatch"
            )
        output_height = (
            height
            + 2 * padding[0]
            - dilation[0] * (kernel_height - 1)
            - 1
        ) // stride[0] + 1
        output_width = (
            width
            + 2 * padding[1]
            - dilation[1] * (kernel_width - 1)
            - 1
        ) // stride[1] + 1
        expected_output_shape = (
            batch,
            int(out_channels),
            int(output_height),
            int(output_width),
        )
        if output_height <= 0 or output_width <= 0 or (
            output_shape != expected_output_shape
        ):
            raise RuntimeLineageReject(
                "runtime_detached_conv_output_geometry_mismatch"
            )
        row_mask_payload = b"none"
        mask_digest: bytes | None = None
        refs: list[object] = [
            operator_value,
            state,
            state["_kernel"],
            state["_input_shape"],
            state["_output_shape"],
            state["_stride"],
            state["_padding"],
            state["_dilation"],
            state["_groups"],
        ]
        if state["_row_mask"] is not None:
            mask = state["_row_mask"]
            if (
                type(mask) is not np.ndarray
                or mask.dtype != np.dtype(bool)
                or mask.ndim != 1
                or mask.size != int(np.prod(output_shape))
                or not mask.flags.c_contiguous
                or not mask.flags.owndata
                or mask.flags.writeable
            ):
                raise RuntimeLineageReject(
                    "runtime_detached_conv_row_mask_invalid"
                )
            row_mask_payload = (
                _metadata_payload([tuple(mask.shape), "owned_c_bool_read_only"])
                + mask.astype(np.uint8, copy=True).tobytes(order="C")
            )
            mask_digest = _detached_array_digest(mask)
            refs.append(mask)
        expected_logical_nnz = _pure_conv_logical_nnz(
            kernel_shape=tuple(state["_kernel"].shape),
            input_shape=input_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            row_mask=state["_row_mask"],
        )
        if (
            type(state["_logical_expanded_nnz"]) is not int
            or state["_logical_expanded_nnz"] < 0
            or state["_logical_expanded_nnz"] != expected_logical_nnz
        ):
            raise RuntimeLineageReject(
                "runtime_detached_conv_logical_nnz_cache_mismatch"
            )
        expected_content_key = (
            "implicit_conv2d_op_v1",
            input_shape,
            output_shape,
            stride,
            padding,
            dilation,
            groups,
            _detached_array_digest(state["_kernel"]),
            mask_digest,
        )
        content_key = state["_content_key"]
        if (
            type(content_key) is not tuple
            or len(content_key) != 9
            or type(content_key[0]) is not str
            or any(type(content_key[index]) is not tuple for index in range(1, 6))
            or any(
                any(type(item) is not int for item in content_key[index])
                for index in range(1, 6)
            )
            or type(content_key[6]) is not int
            or type(content_key[7]) is not bytes
            or (
                content_key[8] is not None
                and type(content_key[8]) is not bytes
            )
            or content_key != expected_content_key
        ):
            raise RuntimeLineageReject(
                "runtime_detached_conv_content_cache_mismatch"
            )
        refs.extend(
            (
                state["_logical_expanded_nnz"],
                state["_content_key"],
            )
        )
        metadata = _metadata_payload(
            [
                "CONV",
                (int(np.prod(output_shape)), int(np.prod(input_shape))),
                input_shape,
                output_shape,
                stride,
                padding,
                dilation,
                groups,
                "owned_c_float64_read_only",
                (
                    None
                    if state["_row_mask"] is None
                    else "owned_c_bool_read_only"
                ),
                expected_logical_nnz,
            ]
        )
        return (
            b"".join(
                len(part).to_bytes(8, "big") + part
                for part in (metadata, kernel.payload, row_mask_payload)
            ),
            tuple(refs),
        )
    raise RuntimeLineageReject("runtime_detached_operator_type")


def _certificate_guard(
    certificate: object,
) -> tuple[bytes, tuple[object, ...]]:
    if type(certificate) is not c3.OrderedPathCertificate or type(
        certificate.events
    ) is not tuple or type(certificate.schema) is not str:
        raise RuntimeLineageReject("runtime_detached_certificate_type")
    refs: list[object] = [certificate, certificate.events]
    event_records: list[object] = []
    for event in certificate.events:
        if (
            type(event) is not c3.GraphEventEvidence
            or type(event.kind) is not str
            or type(event.producer_occurrence_tokens) is not tuple
            or type(event.graph_predecessor_occurrence_tokens) is not tuple
            or type(event.input_value_tokens) is not tuple
            or type(event.operator_position) is not int
            or event.operator_position < 0
            or type(event.selected_input) is not int
            or event.selected_input < 0
        ):
            raise RuntimeLineageReject("runtime_detached_graph_event_type")
        if type(event.bias_payload) is not tuple or any(
            type(value) is not float for value in event.bias_payload
        ):
            raise RuntimeLineageReject(
                "runtime_detached_event_bias_container_type"
            )
        bias_payload = _numeric_snapshot(
            np.asarray(event.bias_payload, dtype=np.float64),
            "runtime_detached_event_bias",
            ndim=1,
        ).payload
        event_records.append(
            (
                event.kind,
                _stable_token_payload(
                    event.occurrence_token,
                    "runtime_detached_event_occurrence",
                ).hex(),
                tuple(
                    _stable_token_payload(
                        token, "runtime_detached_event_producer"
                    ).hex()
                    for token in event.producer_occurrence_tokens
                ),
                tuple(
                    _stable_token_payload(
                        token, "runtime_detached_event_graph_predecessor"
                    ).hex()
                    for token in event.graph_predecessor_occurrence_tokens
                ),
                tuple(
                    _stable_token_payload(
                        token, "runtime_detached_event_input_value"
                    ).hex()
                    for token in event.input_value_tokens
                ),
                _stable_token_payload(
                    event.output_value_token,
                    "runtime_detached_event_output_value",
                ).hex(),
                event.operator_position,
                event.selected_input,
                bias_payload.hex(),
            )
        )
        refs.extend(
            (
                event,
                event.occurrence_token,
                event.producer_occurrence_tokens,
                event.graph_predecessor_occurrence_tokens,
                event.input_value_tokens,
                event.output_value_token,
                event.operator_occurrence,
                event.bias_payload,
            )
        )
        refs.extend(event.producer_occurrence_tokens)
        refs.extend(event.graph_predecessor_occurrence_tokens)
        refs.extend(event.input_value_tokens)
    payload = _metadata_payload(
        [
            certificate.schema,
            _stable_token_payload(
                certificate.entry_producer_occurrence_token,
                "runtime_detached_certificate_entry_producer",
            ).hex(),
            _stable_token_payload(
                certificate.entry_value_token,
                "runtime_detached_certificate_entry_value",
            ).hex(),
            event_records,
        ]
    )
    refs.extend(
        (
            certificate.entry_producer_occurrence_token,
            certificate.entry_value_token,
        )
    )
    return payload, tuple(refs)


def _detached_expression_guard(
    expression: object,
) -> tuple[bytes, tuple[object, ...]]:
    if type(expression) is not c3.C3AffineExprView or type(
        expression.terms
    ) is not tuple or not expression.terms:
        raise RuntimeLineageReject("runtime_detached_expression_type")
    if type(expression.n_out) is not int or expression.n_out <= 0:
        raise RuntimeLineageReject("runtime_detached_expression_width_type")
    bias = _exact_float64_read_only_array(
        expression.bias,
        name="runtime_detached_expression_bias",
        ndim=1,
    )
    if expression.bias.shape != (expression.n_out,):
        raise RuntimeLineageReject("runtime_detached_expression_bias_width")
    refs: list[object] = [
        expression,
        expression.terms,
        expression.bias,
        expression.frame_id,
    ]
    records: list[object] = [
        expression.n_out,
        _stable_token_payload(
            expression.frame_id, "runtime_detached_expression_frame"
        ).hex(),
        "owned_c_float64_read_only_bias",
        bias.payload.hex(),
    ]
    operator_payloads: list[bytes] = []
    seen_operators: list[object] = []
    for term in expression.terms:
        if type(term) is not c3.C3AffineTermView or type(
            term.operators
        ) is not tuple or type(term.source) is not _DetachedSourceAuthorityProxy:
            raise RuntimeLineageReject("runtime_detached_term_type")
        source = term.source
        source_payload, source_refs = _detached_source_authority_guard(
            source
        )
        certificate_payload, certificate_refs = _certificate_guard(
            term.certificate
        )
        refs.extend(
            (
                term,
                source,
                term.operators,
                term.certificate,
                *source_refs,
                *certificate_refs,
            )
        )
        term_operator_indices: list[int] = []
        for operator_value in term.operators:
            operator_index = next(
                (
                    index
                    for index, prior in enumerate(seen_operators)
                    if operator_value is prior
                ),
                None,
            )
            if operator_index is None:
                operator_index = len(seen_operators)
                seen_operators.append(operator_value)
                operator_payload, operator_refs = _detached_operator_guard(
                    operator_value
                )
                operator_payloads.append(operator_payload)
                refs.extend(operator_refs)
            term_operator_indices.append(operator_index)
        records.append(
            (
                source_payload,
                tuple(term_operator_indices),
                certificate_payload.hex(),
            )
        )
    records.append(
        tuple(payload.hex() for payload in operator_payloads)
    )
    records.append(_identity_class_pattern(expression))
    return _metadata_payload(records), tuple(refs)


def _operator_binding_signature(
    expression: c3.C3AffineExprView, operator_value: object
) -> tuple[object, ...]:
    positions = tuple(
        (term_index, operator_index)
        for term_index, term in enumerate(expression.terms)
        for operator_index, candidate in enumerate(term.operators)
        if candidate is operator_value
    )
    payload, _ = _detached_operator_guard(operator_value)
    return positions, payload


def _detached_source_authority_guard(
    source: object,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    if type(source) is not _DetachedSourceAuthorityProxy:
        raise RuntimeLineageReject("runtime_detached_source_type")
    if (
        source.exact is not True
        or type(source.semantic_sha256) is not str
        or len(source.semantic_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source.semantic_sha256
        )
    ):
        raise RuntimeLineageReject(
            "runtime_detached_source_authority_mismatch"
        )
    empty_fields: list[object] = []
    for field_name in _SOURCE_FIELDS:
        field_value = getattr(source, field_name)
        if type(field_value) is not tuple or len(field_value) != 0:
            raise RuntimeLineageReject(
                "runtime_detached_source_schema_field_mismatch"
            )
        empty_fields.append(field_value)
    frame_payload = _stable_token_payload(
        source.frame_id, "runtime_detached_source_frame"
    )
    return (
        (
            source.semantic_sha256,
            frame_payload.hex(),
            "compact_empty_predicate_schema_v1",
        ),
        (source, source.frame_id, *empty_fields),
    )


def _source_binding_signature(
    expression: c3.C3AffineExprView, source: object
) -> tuple[object, ...]:
    authority_payload, _ = _detached_source_authority_guard(source)
    positions = tuple(
        term_index
        for term_index, term in enumerate(expression.terms)
        if term.source is source
    )
    return positions, authority_payload


def _certificate_binding_signature(
    expression: c3.C3AffineExprView, certificate: object
) -> tuple[object, ...]:
    positions = tuple(
        term_index
        for term_index, term in enumerate(expression.terms)
        if term.certificate is certificate
    )
    payload, _ = _certificate_guard(certificate)
    return positions, payload


def _exact_dataclass_values(
    value: object, expected_type: type, reason: str
) -> tuple[object, ...]:
    if type(value) is not expected_type:
        raise RuntimeLineageReject(reason)
    values = tuple(
        getattr(value, name)
        for name in expected_type.__dataclass_fields__
    )
    if any(type(item) is not int or item < 0 for item in values):
        raise RuntimeLineageReject(reason)
    return values


def _strict_index_tuple(
    value: object,
    *,
    reason: str,
    upper_bound: int,
) -> tuple[int, ...]:
    if type(value) is not tuple or any(
        type(item) is not int or item < 0 or item >= upper_bound
        for item in value
    ):
        raise RuntimeLineageReject(reason)
    if value != tuple(sorted(set(value))):
        raise RuntimeLineageReject(reason)
    return value


def _strict_sha256_tuple(
    value: object, *, reason: str, expected_size: int | None = None
) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or (expected_size is not None and len(value) != expected_size)
        or any(
            type(item) is not str
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in value
        )
    ):
        raise RuntimeLineageReject(reason)
    return value


def _strict_content_key(
    value: object, *, reason: str
) -> tuple[str, str]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not str for item in value)
        or value[0] != c3.C3_DESCRIPTOR_TAG
        or len(value[1]) != 64
        or any(character not in "0123456789abcdef" for character in value[1])
    ):
        raise RuntimeLineageReject(reason)
    return value


def _planner_decision_semantic_fingerprint(
    decision: c3.C3PlanDecision,
    expression: c3.C3AffineExprView | None = None,
) -> tuple[object, ...]:
    """Address-independent, complete result snapshot relative to its input."""

    if type(decision) is not c3.C3PlanDecision:
        raise RuntimeLineageReject("runtime_planner_decision_type")
    if expression is None:
        expression = decision.expression
    if type(expression) is not c3.C3AffineExprView:
        raise RuntimeLineageReject("runtime_planner_expression_type")
    decision_prefix = (
        decision.accepted,
        decision.reason,
        decision.no_claims,
    )
    if not decision.accepted:
        return (*decision_prefix, None)
    plan = decision.plan
    if type(plan) is not c3.C3PureLineagePlan:
        raise RuntimeLineageReject("runtime_planner_plan_type")
    requests: list[object] = []
    for request in plan.requests:
        if type(request) is not DescriptorRequest:
            raise RuntimeLineageReject("runtime_planner_request_type")
        requests.append(
            (
                request.content_key,
                _operator_binding_signature(expression, request.inner),
                tuple(
                    _operator_binding_signature(expression, middle)
                    for middle in request.middle_ops
                ),
                _operator_binding_signature(expression, request.outer),
                request.selected_rows,
                request.term_indices,
                request.use_count,
                _exact_dataclass_values(
                    request.estimate,
                    DescriptorEstimate,
                    "runtime_planner_estimate_type",
                ),
            )
        )
    uses: list[object] = []
    for use in plan.uses:
        if type(use) is not c3.C3BranchUse:
            raise RuntimeLineageReject("runtime_planner_use_type")
        uses.append(
            (
                use.term_index,
                _source_binding_signature(expression, use.source),
                _operator_binding_signature(expression, use.inner),
                tuple(
                    _operator_binding_signature(expression, diagonal)
                    for diagonal in use.pre_add_diagonals
                ),
                tuple(
                    _operator_binding_signature(expression, diagonal)
                    for diagonal in use.post_add_diagonals
                ),
                _operator_binding_signature(expression, use.outer),
                tuple(
                    _operator_binding_signature(expression, diagonal)
                    for diagonal in use.output_diagonals
                ),
                use.selected_rows,
                use.descriptor_content_key,
                use.pre_add_diagonal_count,
                _certificate_binding_signature(
                    expression, use.certificate
                ),
                use.certificate_digest,
                use.certificate_snapshot_payload,
            )
        )
    return (
        *decision_prefix,
        (
            plan.rule_id,
            _stable_token_payload(
                plan.frame_id, "runtime_planner_plan_frame"
            ),
            plan.common_add_snapshot_payload,
            plan.certificate_digests,
            tuple(requests),
            tuple(uses),
            plan.identity_term_indices,
            plan.zero_support_term_indices,
            _exact_dataclass_values(
                plan.reservation,
                BudgetReservation,
                "runtime_planner_reservation_type",
            ),
            plan.prospective_emission_contributions,
            plan.emission_executed,
            plan.emission_status,
            plan.no_claims,
        ),
    )


def _planner_decision_guard(
    decision: c3.C3PlanDecision,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    semantic = _planner_decision_semantic_fingerprint(decision)
    refs: list[object] = [
        decision,
        decision.expression,
        decision.terms,
        decision.bias,
        decision.plan,
    ]
    refs.extend(decision.terms)
    plan = decision.plan
    if decision.accepted:
        if type(plan) is not c3.C3PureLineagePlan:
            raise RuntimeLineageReject("runtime_planner_plan_type")
        refs.extend(
            (
                plan,
                plan.expression,
                plan.original_terms,
                plan.bias,
                plan.frame_id,
                plan.certificate_digests,
                plan.requests,
                plan.uses,
                plan.identity_term_indices,
                plan.zero_support_term_indices,
                plan.reservation,
                plan.no_claims,
            )
        )
        refs.extend(plan.original_terms)
        for request in plan.requests:
            refs.extend(
                (
                    request,
                    request.content_key,
                    request.inner,
                    request.middle_ops,
                    *request.middle_ops,
                    request.outer,
                    request.selected_rows,
                    request.term_indices,
                    request.estimate,
                )
            )
        for use in plan.uses:
            refs.extend(
                (
                    use,
                    use.source,
                    use.inner,
                    use.pre_add_diagonals,
                    *use.pre_add_diagonals,
                    use.post_add_diagonals,
                    *use.post_add_diagonals,
                    use.outer,
                    use.output_diagonals,
                    *use.output_diagonals,
                    use.selected_rows,
                    use.descriptor_content_key,
                    use.certificate,
                    use.certificate_snapshot_payload,
                )
            )
    elif plan is not None:
        raise RuntimeLineageReject("runtime_rejected_planner_has_plan")
    return semantic, tuple(refs)


def _identity_class_pattern(
    expression: c3.C3AffineExprView,
) -> tuple[object, ...]:
    seen: list[object] = []

    def identity_class(value: object | None) -> int | None:
        if value is None:
            return None
        for index, prior in enumerate(seen):
            if value is prior:
                return index
        seen.append(value)
        return len(seen) - 1

    return tuple(
        (
            identity_class(term.source),
            tuple(identity_class(value) for value in term.operators),
            tuple(
                identity_class(event.operator_occurrence)
                for event in term.certificate.events
            ),
        )
        for term in expression.terms
    )


def _planner_input_semantic_fingerprint(
    expression: c3.C3AffineExprView,
    support: np.ndarray,
    budget: PlannerBudget,
) -> tuple[object, ...]:
    support_array = np.array(support, order="C", copy=True)
    support_payload = (
        _metadata_payload(
            [support_array.dtype.str, tuple(support_array.shape)]
        )
        + support_array.tobytes(order="C")
    )
    return (
        len(expression.terms),
        _identity_class_pattern(expression),
        _numeric_snapshot(
            expression.bias, "runtime_planner_fingerprint_bias", ndim=1
        ).payload,
        expression.n_out,
        _stable_token_payload(
            expression.frame_id, "runtime_planner_fingerprint_frame"
        ),
        support_payload,
        _exact_dataclass_values(
            budget,
            PlannerBudget,
            "runtime_planner_budget_field_type",
        ),
    )


def _require_identity_sequence(
    actual: object, expected: tuple[object, ...], reason: str
) -> None:
    if type(actual) is not tuple or not _identity_tuple_equal(
        actual, expected
    ):
        raise RuntimeLineageReject(reason)


def _require_planner_decision_bound_to_input(
    decision: c3.C3PlanDecision,
    expression: c3.C3AffineExprView,
    output_support: np.ndarray,
    budget: PlannerBudget,
) -> None:
    """Reconstruct every accepted nested plan field from the exact input."""

    if (
        type(decision) is not c3.C3PlanDecision
        or type(decision.accepted) is not bool
        or type(decision.reason) is not str
        or decision.expression is not expression
        or decision.terms is not expression.terms
        or decision.bias is not expression.bias
        or decision.no_claims is not c3.C3_NO_CLAIMS
    ):
        raise RuntimeLineageReject("runtime_planner_input_binding_mismatch")
    if not decision.accepted:
        if decision.plan is not None:
            raise RuntimeLineageReject("runtime_rejected_planner_has_plan")
        return
    if decision.reason != "planned_pure_identity_middle_lineage_only":
        raise RuntimeLineageReject("runtime_planner_accept_reason_mismatch")
    plan = decision.plan
    if (
        type(plan) is not c3.C3PureLineagePlan
        or plan.expression is not expression
        or plan.original_terms is not expression.terms
        or plan.bias is not expression.bias
        or plan.frame_id is not expression.frame_id
        or type(plan.rule_id) is not str
        or plan.rule_id != c3.C3_RULE_ID
        or plan.no_claims is not c3.C3_NO_CLAIMS
    ):
        raise RuntimeLineageReject("runtime_planner_plan_binding_mismatch")
    if (
        type(plan.requests) is not tuple
        or type(plan.uses) is not tuple
        or type(plan.certificate_digests) is not tuple
        or type(plan.identity_term_indices) is not tuple
        or type(plan.zero_support_term_indices) is not tuple
        or type(plan.common_add_snapshot_payload) is not bytes
        or type(plan.prospective_emission_contributions) is not int
        or plan.prospective_emission_contributions < 0
        or type(plan.emission_executed) is not bool
        or type(plan.emission_status) is not str
    ):
        raise RuntimeLineageReject("runtime_planner_plan_field_type")

    try:
        n_out = c3._strict_nonnegative_int(
            expression.n_out, name="runtime_binding_n_out"
        )
        support = c3._c1_normalize_support(output_support, n_out)
        frame_payload = c3._frame_payload(
            expression.frame_id, name="runtime_binding_frame"
        )
        parsed = tuple(
            c3._parse_term(
                term,
                term_index=term_index,
                frame_payload=frame_payload,
                support=support,
                n_out=n_out,
            )
            for term_index, term in enumerate(expression.terms)
        )
        c3._require_one_shared_suffix(parsed)
    except Exception as exc:
        raise RuntimeLineageReject(
            "runtime_planner_plan_reparse_failed"
        ) from exc

    identity_indices = tuple(
        item.term_index for item in parsed if item.identity_skip
    )
    complete = tuple(item for item in parsed if not item.identity_skip)
    zero_support = tuple(
        item.term_index
        for item in complete
        if not item.active_selected_rows
    )
    eligible = tuple(
        item for item in complete if item.active_selected_rows
    )
    if not eligible:
        raise RuntimeLineageReject("runtime_planner_no_expected_use")
    _strict_index_tuple(
        plan.identity_term_indices,
        reason="runtime_planner_identity_indices_type",
        upper_bound=len(expression.terms),
    )
    _strict_index_tuple(
        plan.zero_support_term_indices,
        reason="runtime_planner_zero_support_type",
        upper_bound=len(expression.terms),
    )
    _strict_sha256_tuple(
        plan.certificate_digests,
        reason="runtime_planner_certificate_digests_type",
        expected_size=len(expression.terms),
    )
    if plan.identity_term_indices != identity_indices:
        raise RuntimeLineageReject("runtime_planner_identity_indices_mismatch")
    if plan.zero_support_term_indices != zero_support:
        raise RuntimeLineageReject("runtime_planner_zero_support_mismatch")
    expected_certificate_digests = tuple(
        item.validated_certificate.digest for item in parsed
    )
    if plan.certificate_digests != expected_certificate_digests:
        raise RuntimeLineageReject(
            "runtime_planner_certificate_digests_mismatch"
        )
    if (
        plan.common_add_snapshot_payload
        != parsed[0].validated_certificate.add_snapshot_payload
    ):
        raise RuntimeLineageReject(
            "runtime_planner_common_add_snapshot_mismatch"
        )

    grouped: dict[bytes, list[object]] = {}
    for item in eligible:
        if (
            item.stable_numeric_payload is None
            or item.content_key is None
            or item.representative_key is None
        ):
            raise RuntimeLineageReject(
                "runtime_planner_expected_descriptor_missing"
            )
        grouped.setdefault(item.stable_numeric_payload, []).append(item)
    ordered_groups = tuple(
        group
        for _, group in sorted(
            grouped.items(),
            key=lambda pair: (hashlib.sha256(pair[0]).hexdigest(), pair[0]),
        )
    )
    if len(plan.requests) != len(ordered_groups):
        raise RuntimeLineageReject("runtime_planner_request_count_mismatch")
    expected_requests: list[DescriptorRequest] = []
    for request, group in zip(plan.requests, ordered_groups, strict=True):
        if (
            type(request) is not DescriptorRequest
            or type(request.middle_ops) is not tuple
            or type(request.use_count) is not int
            or request.use_count < 0
            or type(request.estimate) is not DescriptorEstimate
        ):
            raise RuntimeLineageReject("runtime_planner_request_type")
        _strict_content_key(
            request.content_key,
            reason="runtime_planner_request_content_key_type",
        )
        _strict_index_tuple(
            request.selected_rows,
            reason="runtime_planner_request_selected_rows_type",
            upper_bound=n_out,
        )
        _strict_index_tuple(
            request.term_indices,
            reason="runtime_planner_request_term_indices_type",
            upper_bound=len(expression.terms),
        )
        request_estimate_values = _exact_dataclass_values(
            request.estimate,
            DescriptorEstimate,
            "runtime_planner_estimate_type",
        )
        representative = min(
            group, key=lambda item: item.representative_key
        )
        selected_rows = tuple(
            sorted(
                {
                    row
                    for item in group
                    for row in item.selected_rows
                }
            )
        )
        term_indices = tuple(sorted(item.term_index for item in group))
        middle_ops = (
            *representative.pre_add_diagonals,
            *representative.post_add_diagonals,
        )
        if (
            request.content_key != representative.content_key
            or request.inner is not representative.inner
            or request.outer is not representative.outer
            or request.selected_rows != selected_rows
            or request.term_indices != term_indices
            or request.use_count != len(group)
        ):
            raise RuntimeLineageReject(
                "runtime_planner_request_semantic_mismatch"
            )
        _require_identity_sequence(
            request.middle_ops,
            middle_ops,
            "runtime_planner_request_middle_binding_mismatch",
        )
        try:
            expected_estimate = c3._c1_estimate_descriptor(
                representative.inner,
                representative.outer,
                selected_rows,
            )
        except Exception as exc:
            raise RuntimeLineageReject(
                "runtime_planner_estimate_recompute_failed"
            ) from exc
        if request_estimate_values != _exact_dataclass_values(
            expected_estimate,
            DescriptorEstimate,
            "runtime_planner_expected_estimate_type",
        ):
            raise RuntimeLineageReject("runtime_planner_estimate_mismatch")
        expected_requests.append(
            DescriptorRequest(
                content_key=representative.content_key,
                inner=representative.inner,
                middle_ops=middle_ops,
                outer=representative.outer,
                selected_rows=selected_rows,
                term_indices=term_indices,
                use_count=len(group),
                estimate=expected_estimate,
            )
        )

    expected_uses = tuple(sorted(eligible, key=lambda item: item.term_index))
    if len(plan.uses) != len(expected_uses):
        raise RuntimeLineageReject("runtime_planner_use_count_mismatch")
    for use, item in zip(plan.uses, expected_uses, strict=True):
        certificate = item.term.certificate
        if (
            type(use) is not c3.C3BranchUse
            or type(use.term_index) is not int
            or use.term_index < 0
            or use.term_index >= len(expression.terms)
            or type(use.pre_add_diagonals) is not tuple
            or type(use.post_add_diagonals) is not tuple
            or type(use.output_diagonals) is not tuple
            or type(use.pre_add_diagonal_count) is not int
            or use.pre_add_diagonal_count < 0
            or type(use.certificate_digest) is not str
            or len(use.certificate_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in use.certificate_digest
            )
            or type(use.certificate_snapshot_payload) is not bytes
        ):
            raise RuntimeLineageReject(
                "runtime_planner_use_type"
            )
        _strict_index_tuple(
            use.selected_rows,
            reason="runtime_planner_use_selected_rows_type",
            upper_bound=n_out,
        )
        _strict_content_key(
            use.descriptor_content_key,
            reason="runtime_planner_use_content_key_type",
        )
        if (
            use.term_index != item.term_index
            or use.source is not item.source
            or use.inner is not item.inner
            or use.outer is not item.outer
            or use.selected_rows != item.selected_rows
            or use.descriptor_content_key != item.content_key
            or use.pre_add_diagonal_count
            != len(item.pre_add_diagonals)
            or use.certificate is not certificate
            or use.certificate_digest
            != item.validated_certificate.digest
            or use.certificate_snapshot_payload
            != item.validated_certificate.snapshot_payload
        ):
            raise RuntimeLineageReject(
                "runtime_planner_use_semantic_mismatch"
            )
        _require_identity_sequence(
            use.pre_add_diagonals,
            item.pre_add_diagonals,
            "runtime_planner_use_pre_add_binding_mismatch",
        )
        _require_identity_sequence(
            use.post_add_diagonals,
            item.post_add_diagonals,
            "runtime_planner_use_post_add_binding_mismatch",
        )
        _require_identity_sequence(
            use.output_diagonals,
            item.output_diagonals,
            "runtime_planner_use_output_binding_mismatch",
        )
        expected_operators = (
            use.inner,
            *use.pre_add_diagonals,
            *use.post_add_diagonals,
            use.outer,
            *use.output_diagonals,
        )
        _require_identity_sequence(
            item.term.operators,
            expected_operators,
            "runtime_planner_use_operator_partition_mismatch",
        )

    uses_by_key: dict[tuple[str, str], list[c3.C3BranchUse]] = {}
    for use in plan.uses:
        uses_by_key.setdefault(use.descriptor_content_key, []).append(use)
    for request in plan.requests:
        matching_uses = uses_by_key.get(request.content_key, [])
        if (
            tuple(use.term_index for use in matching_uses)
            != request.term_indices
            or len(matching_uses) != request.use_count
            or tuple(
                sorted(
                    {
                        row
                        for use in matching_uses
                        for row in use.selected_rows
                    }
                )
            )
            != request.selected_rows
            or not any(
                use.inner is request.inner
                and use.outer is request.outer
                and _identity_tuple_equal(
                    (
                        *use.pre_add_diagonals,
                        *use.post_add_diagonals,
                    ),
                    request.middle_ops,
                )
                for use in matching_uses
            )
        ):
            raise RuntimeLineageReject(
                "runtime_planner_request_use_binding_mismatch"
            )

    try:
        expected_reservation = c3._c1_reserve_budget(
            tuple(expected_requests), budget
        )
    except Exception as exc:
        raise RuntimeLineageReject(
            "runtime_planner_reservation_recompute_failed"
        ) from exc
    if _exact_dataclass_values(
        plan.reservation,
        BudgetReservation,
        "runtime_planner_reservation_type",
    ) != _exact_dataclass_values(
        expected_reservation,
        BudgetReservation,
        "runtime_planner_expected_reservation_type",
    ):
        raise RuntimeLineageReject("runtime_planner_reservation_mismatch")
    if (
        plan.prospective_emission_contributions
        != sum(
            request.estimate.emission_contributions
            for request in expected_requests
        )
        or plan.emission_executed is not False
        or plan.emission_status
        != "deferred_to_future_actual_support_transaction"
    ):
        raise RuntimeLineageReject("runtime_planner_emission_state_mismatch")


def _reject(expr: RuntimeLineageExprView, reason: str) -> RuntimeAdapterDecision:
    return RuntimeAdapterDecision(False, reason, expr, None, None, "")


def _snapshot_output_support(
    output_support: object, *, expected_width: int
) -> np.ndarray:
    if (
        type(expected_width) is not int
        or expected_width <= 0
        or type(output_support) is not np.ndarray
        or output_support.dtype != np.dtype(bool)
        or output_support.ndim != 1
        or output_support.shape != (expected_width,)
        or not output_support.flags.c_contiguous
    ):
        raise RuntimeLineageReject("runtime_output_support_storage_schema")
    return _make_array_read_only(
        np.array(output_support, dtype=bool, order="C", copy=True)
    )


def plan_s0_c3_from_runtime_lineage(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    expr: RuntimeLineageExprView,
    output_support: Any,
    *,
    registry: object,
    budget: PlannerBudget = PlannerBudget(),
) -> RuntimeAdapterDecision:
    """Rederive current lineage three times and invoke only the pure planner."""

    try:
        if type(expr) is not RuntimeLineageExprView:
            raise RuntimeLineageReject("runtime_expression_type")
        _exact_dataclass_values(
            budget, PlannerBudget, "runtime_planner_budget_field_type"
        )
        _snapshot_output_support(
            output_support, expected_width=expr.n_out
        )
        before = _derive_current(
            layers, preds, succs, registry, expr
        )
        planner_expression = _detach_c3_expression(before.c3_expression)
        (
            planner_expression_payload,
            planner_expression_references,
        ) = _detached_expression_guard(planner_expression)
        planner_support = _snapshot_output_support(
            output_support, expected_width=expr.n_out
        )
        planner_budget = replace(budget)
        planner_input_fingerprint = _planner_input_semantic_fingerprint(
            planner_expression, planner_support, planner_budget
        )
        planner_error: Exception | None = None
        planner_decision: c3.C3PlanDecision | None = None
        try:
            planner_decision = c3.plan_s0_c3_identity_middle_lineage(
                planner_expression,
                planner_support,
                budget=planner_budget,
            )
            _require_planner_decision_bound_to_input(
                planner_decision,
                planner_expression,
                planner_support,
                planner_budget,
            )
            (
                current_planner_expression_payload,
                current_planner_expression_references,
            ) = _detached_expression_guard(planner_expression)
            if (
                current_planner_expression_payload
                != planner_expression_payload
                or not _identity_tuple_equal(
                    current_planner_expression_references,
                    planner_expression_references,
                )
            ):
                raise RuntimeLineageReject(
                    "runtime_planner_detached_input_changed"
                )
            (
                planner_decision_semantic,
                planner_decision_references,
            ) = _planner_decision_guard(planner_decision)
        except Exception as exc:
            planner_error = exc

        try:
            after = _derive_current(
                layers, preds, succs, registry, expr
            )
        except Exception as exc:
            return _reject(expr, f"current_state_revalidation_failed:{type(exc).__name__}")
        if (
            before.current_state_sha256 != after.current_state_sha256
            or not _same_references(before.reference_objects, after.reference_objects)
        ):
            return _reject(expr, "current_state_changed_during_planning")
        if planner_error is not None:
            return _reject(expr, f"runtime_planner_failed:{type(planner_error).__name__}")
        assert planner_decision is not None
        semantic_expression = _detach_c3_expression(after.c3_expression)
        semantic_support = _snapshot_output_support(
            output_support, expected_width=expr.n_out
        )
        semantic_budget = replace(budget)
        semantic_input_fingerprint = _planner_input_semantic_fingerprint(
            semantic_expression, semantic_support, semantic_budget
        )
        try:
            semantic_decision = c3.plan_s0_c3_identity_middle_lineage(
                semantic_expression,
                semantic_support,
                budget=semantic_budget,
            )
            _require_planner_decision_bound_to_input(
                semantic_decision,
                semantic_expression,
                semantic_support,
                semantic_budget,
            )
            if (
                planner_input_fingerprint,
                _planner_decision_semantic_fingerprint(planner_decision),
            ) != (
                semantic_input_fingerprint,
                _planner_decision_semantic_fingerprint(semantic_decision),
            ):
                return _reject(
                    expr, "runtime_planner_semantic_snapshot_mismatch"
                )
        except RuntimeLineageReject:
            raise
        except Exception as exc:
            return _reject(
                expr,
                "runtime_planner_semantic_revalidation_failed:"
                f"{type(exc).__name__}",
            )
        final_support = _snapshot_output_support(
            output_support, expected_width=expr.n_out
        )
        final_budget = replace(budget)
        if _planner_input_semantic_fingerprint(
            planner_expression, final_support, final_budget
        ) != planner_input_fingerprint:
            return _reject(
                expr, "final_request_input_changed_during_planning"
            )
        (
            final_planner_expression_payload,
            final_planner_expression_references,
        ) = _detached_expression_guard(planner_expression)
        if (
            final_planner_expression_payload != planner_expression_payload
            or not _identity_tuple_equal(
                final_planner_expression_references,
                planner_expression_references,
            )
        ):
            return _reject(
                expr, "final_planner_input_changed_during_planning"
            )
        _require_planner_decision_bound_to_input(
            planner_decision,
            planner_expression,
            final_support,
            final_budget,
        )
        (
            final_planner_decision_semantic,
            final_planner_decision_references,
        ) = _planner_decision_guard(planner_decision)
        if (
            final_planner_decision_semantic != planner_decision_semantic
            or not _identity_tuple_equal(
                final_planner_decision_references,
                planner_decision_references,
            )
        ):
            return _reject(
                expr, "final_planner_decision_changed_during_planning"
            )
        try:
            final_state = _derive_current(
                layers, preds, succs, registry, expr
            )
        except Exception as exc:
            return _reject(
                expr,
                "final_state_revalidation_failed:"
                f"{type(exc).__name__}",
            )
        if (
            final_state.current_state_sha256
            != before.current_state_sha256
            or final_state.current_state_sha256
            != after.current_state_sha256
            or not _same_references(
                final_state.reference_objects, before.reference_objects
            )
            or not _same_references(
                final_state.reference_objects, after.reference_objects
            )
        ):
            return _reject(expr, "final_state_changed_during_planning")
        post_final_support = _snapshot_output_support(
            output_support, expected_width=expr.n_out
        )
        post_final_budget = replace(budget)
        if _planner_input_semantic_fingerprint(
            planner_expression,
            post_final_support,
            post_final_budget,
        ) != planner_input_fingerprint:
            return _reject(
                expr, "post_final_request_input_changed_during_planning"
            )
        (
            post_final_expression_payload,
            post_final_expression_references,
        ) = _detached_expression_guard(planner_expression)
        if (
            post_final_expression_payload != planner_expression_payload
            or not _identity_tuple_equal(
                post_final_expression_references,
                planner_expression_references,
            )
        ):
            return _reject(
                expr, "post_final_planner_input_changed_during_planning"
            )
        _require_planner_decision_bound_to_input(
            planner_decision,
            planner_expression,
            post_final_support,
            post_final_budget,
        )
        (
            post_final_decision_semantic,
            post_final_decision_references,
        ) = _planner_decision_guard(planner_decision)
        if (
            post_final_decision_semantic != planner_decision_semantic
            or not _identity_tuple_equal(
                post_final_decision_references,
                planner_decision_references,
            )
        ):
            return _reject(
                expr, "post_final_planner_decision_changed_during_planning"
            )
        if not planner_decision.accepted:
            return _reject(expr, f"c3_planner_rejected:{planner_decision.reason}")
        return RuntimeAdapterDecision(
            True,
            "planned_from_current_runtime_lineage_only",
            expr,
            planner_expression,
            planner_decision,
            before.current_state_sha256,
        )
    except RuntimeLineageReject as exc:
        return _reject(expr, str(exc))
    except Exception as exc:
        return _reject(expr, f"runtime_adapter_failed:{type(exc).__name__}")


__all__ = [
    "FORMAL_BASELINE",
    "RUNTIME_ADAPTER_NO_CLAIMS",
    "RUNTIME_ADAPTER_SCHEMA",
    "RUNTIME_EVENT_SCHEMA",
    "RuntimeAdapterDecision",
    "RuntimeLineageEvent",
    "RuntimeLineageExprView",
    "RuntimeLineageReject",
    "RuntimeLineageTermView",
    "capture_runtime_lineage_events",
    "plan_s0_c3_from_runtime_lineage",
]
