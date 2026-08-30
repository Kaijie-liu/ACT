#!/usr/bin/env python3
"""Default-off raw-TOP1 to fresh Operator-HZ K4 cut pipeline.

This module is an orchestration boundary, not a verdict engine.  When
enabled, it connects the already hardened candidate-only components in one
strict order:

1. consume a complete raw TOP1 VNN-LIB rival batch;
2. bind the caller's complete B=1 output interval and every rival's exact
   binary64 interval hardness;
3. select exactly the residual selector's authenticated joint-focus row;
4. independently replay both the hardness and focused-rival receipts;
5. derive and verify Operator-HZ exact-ReLU literals;
6. search only a top-4/six-pair exact conflict graph; and
7. materialize a verified K4 cut into a fresh Operator-HZ.

Every intermediate object and every receipt in this module is candidate-only
and has ``proof_authority=False``.  The returned public build never reaches a
solver.  An owner/PID/TTL-bound capability transfers exactly one separately
copied, frozen HZ to the verifier; SAFE remains exclusively the responsibility
of ``hz_objbound_decide`` on that private object.

The pipeline uses one caller-owned absolute ``time.monotonic`` deadline.
Candidate work receives at most the first 40 percent of the initial
remaining wall time.  The remaining 60 percent is reserved for hardened
verification, fresh materialization, and terminal receipt checks.  A
timeout, malformed input, rejected receipt, missing K4, or any downstream
error returns the original build by identity with a checksummed fallback
receipt.

The default-off wrapper branches before reading any path, array, receipt,
deadline, caps object, or downstream function.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
import secrets
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple
import weakref

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_clique_materializer import (
    OperatorPhaseCliqueMaterialization,
    _private_solver_build,
    _solver_owned_arrays,
    _solver_owned_objects,
    _source_frame_digest,
    consume_operator_phase_clique_materialization_solver_handoff,
    materialize_verified_operator_phase_clique_cuts,
    validate_operator_phase_clique_materialization_solver_handoff,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_cliques import (
    OperatorCertifiedPhaseClique,
    OperatorExactReLUPhaseCliqueResult,
    OperatorPhaseCliqueCaps,
    RankedOperatorPhase,
    _COMPACT_SUCCESS_STATUS,
    _COMPACT_TELEMETRY_SCHEMA,
    _PROGRESS_SCHEMA,
    _cut_has_exact_private_nonempty_witness,
    run_operator_exact_relu_phase_cliques_candidate,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
    verify_operator_exact_relu_property_phase_selection,
)
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    ExactDualRayConflictCertificate,
    PersistentPairRecord,
    _CANDIDATE_DUST_ABS,
    _MAX_BINARY_CHANGE_COEFFICIENTS,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
)
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuild,
    validate_operator_hz_constructive_nonempty_seal,
)
from act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge import (
    issue_raw_rival_exact_hardness_receipt,
    select_raw_focused_rivals,
    verify_raw_focused_rival_selection,
    verify_raw_rival_exact_hardness_receipt,
)
from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
    consume_raw_vnnlib_top1_candidate,
    issue_raw_vnnlib_top1_candidate,
    validate_consumed_raw_vnnlib_rival_batch,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_constructively_nonempty,
    hz_mark_constructively_nonempty,
)


class OperatorPhaseCliquePipelineError(ValueError):
    """The public pipeline contract itself is malformed."""


class _PipelineFallback(RuntimeError):
    """Internal stop signal that always returns the unchanged baseline."""

    def __init__(self, reason: str, *, timeout: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.timeout = timeout


_PIPELINE_HANDOFF_PRODUCER = object()
_PIPELINE_HANDOFF_REGISTRY_LOCK = threading.Lock()
_PIPELINE_HANDOFF_REGISTRY_CAPACITY = 64
_PIPELINE_HANDOFF_TTL_SECONDS = 60.0


class OperatorPhaseCliquePipelineSolverCapability:
    """Opaque one-use ownership transfer for one private solver HZ."""

    __slots__ = (
        "_token",
        "_process_id",
        "_expires_monotonic",
        "_semantic_digest",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        token: str,
        process_id: int,
        expires_monotonic: float,
        semantic_digest: str,
        _producer_capability: Any,
    ) -> None:
        if _producer_capability is not _PIPELINE_HANDOFF_PRODUCER:
            raise PermissionError(
                "pipeline solver handoff requires its producer"
            )
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_process_id", process_id)
        object.__setattr__(
            self, "_expires_monotonic", expires_monotonic
        )
        object.__setattr__(self, "_semantic_digest", semantic_digest)

    @property
    def token(self) -> str:
        return self._token

    @property
    def process_id(self) -> int:
        return self._process_id

    @property
    def expires_monotonic(self) -> float:
        return self._expires_monotonic

    @property
    def semantic_digest(self) -> str:
        return self._semantic_digest

    @property
    def proof_authority(self) -> bool:
        return False

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("pipeline solver capabilities are immutable")

    def __copy__(self):
        copied = object.__new__(type(self))
        for name in (
            "_token",
            "_process_id",
            "_expires_monotonic",
            "_semantic_digest",
        ):
            object.__setattr__(
                copied, name, object.__getattribute__(self, name)
            )
        return copied

    def __deepcopy__(self, _memo):
        return self.__copy__()


@dataclass(frozen=True)
class OperatorPhaseCliquePipelineCaps:
    """Bounded resources for the first production K4 experiment."""

    max_full_rivals: int = 256
    max_raw_exact_bits: int = 4096
    max_raw_work_items: int = 5_000_000
    max_binaries: int = 16_384
    max_selection_work_items: int = 5_000_000
    max_selection_seconds: float = 5.0
    max_parent_variables: int = 2_000_000
    max_parent_rows: int = 2_000_000
    max_parent_nonzeros: int = 50_000_000
    max_parent_buffer_items: int = 120_000_000
    max_cliques: int = 1
    max_clique_search_nodes: int = 100_000
    max_source_terms: int = 128
    max_multiplier_bits: int = 256
    max_exact_bits: int = 4096
    max_exact_nonzeros: int = 200_000

    def __post_init__(self) -> None:
        integer_fields = (
            "max_full_rivals",
            "max_raw_exact_bits",
            "max_raw_work_items",
            "max_binaries",
            "max_selection_work_items",
            "max_parent_variables",
            "max_parent_rows",
            "max_parent_nonzeros",
            "max_parent_buffer_items",
            "max_cliques",
            "max_clique_search_nodes",
            "max_source_terms",
            "max_multiplier_bits",
            "max_exact_bits",
            "max_exact_nonzeros",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise OperatorPhaseCliquePipelineError(
                    f"{name}_must_be_positive_builtin_int"
                )
        seconds = self.max_selection_seconds
        if (
            type(seconds) not in {int, float}
            or type(seconds) is bool
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0.0
            or float(seconds) > 60.0
        ):
            raise OperatorPhaseCliquePipelineError(
                "max_selection_seconds_invalid"
            )
        object.__setattr__(
            self, "max_selection_seconds", float(seconds)
        )
        if self.max_full_rivals > 10_000:
            raise OperatorPhaseCliquePipelineError(
                "max_full_rivals_exceeds_hard_limit"
            )
        if self.max_cliques > 16:
            raise OperatorPhaseCliquePipelineError(
                "max_cliques_exceeds_hard_limit"
            )


@dataclass(frozen=True)
class OperatorPhaseCliquePipelineResult:
    """An unchanged baseline or one receipt-checked fresh build."""

    build: Any
    enabled: bool
    status: str
    identity_preserved: bool
    materialized: bool
    receipt: Mapping[str, Any]
    materialization: Optional[OperatorPhaseCliqueMaterialization] = None
    solver_handoff_capability: Optional[
        OperatorPhaseCliquePipelineSolverCapability
    ] = None
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.status) is not str
            or type(self.identity_preserved) is not bool
            or type(self.materialized) is not bool
            or self.proof_authority is not False
            or (
                self.solver_handoff_capability is not None
                and type(self.solver_handoff_capability)
                is not OperatorPhaseCliquePipelineSolverCapability
            )
        ):
            raise OperatorPhaseCliquePipelineError(
                "pipeline_result_header_invalid"
            )
        object.__setattr__(
            self,
            "receipt",
            _deep_freeze(_builtin_copy(self.receipt)),
        )


@dataclass(frozen=True)
class _PipelineSolverHandoffRecord:
    capability_ref: "weakref.ReferenceType[OperatorPhaseCliquePipelineSolverCapability]"
    result_ref: "weakref.ReferenceType[OperatorPhaseCliquePipelineResult]"
    source_build_ref: "weakref.ReferenceType[OperatorHZBuild]"
    public_build_ref: "weakref.ReferenceType[OperatorHZBuild]"
    public_hz_ref: "weakref.ReferenceType[SparseHZono]"
    public_owned_identity: Tuple[int, ...]
    private_build: OperatorHZBuild
    private_owned_identity: Tuple[int, ...]
    receipt_object: Mapping[str, Any]
    receipt_sha256: str
    status: str
    materialized: bool
    semantic_digest: str
    expires_monotonic: float
    process_id: int


_PIPELINE_HANDOFF_REGISTRY: dict[
    str, _PipelineSolverHandoffRecord
] = {}


@dataclass(frozen=True)
class _CandidateBindingSnapshot:
    """Builtin-only bindings captured before fresh materialization."""

    full_rival_count: int
    focused_encoded_row: int
    ranked_literal_count: int
    pair_count: int
    certified_edge_count: int
    clique_count: int
    full_batch_sha256: str
    full_live_assert_sha256: str
    full_property_digest: str
    interval_frame_sha256: str
    hardness_vector_digest: str
    focused_subset_digest: str
    selection_digest: str
    focused_property_digest: str
    subset_binding_digest: str
    clique_ids: Tuple[str, ...]
    candidate_result_status: str
    candidate_telemetry_schema: str
    candidate_representation: str
    candidate_descriptor_sha256: str
    candidate_route_summary: Mapping[str, Any]


_PIPELINE_SCHEMA = "act.operator_phase_clique_pipeline.v1"
_INTERVAL_SCHEMA = "act.operator_phase_clique_b1_interval.v1"
_VERDICT_PATH = "hz_objbound_decide_only"
_CANDIDATE_FRACTION = 0.40
_MATERIALIZER_FRACTION = 0.60
_K4_TOP_LITERALS = 4
_K4_TOTAL_PAIRS = 6
_COMPACT_CANDIDATE_REPRESENTATION = (
    "exact_certificates_and_clique_descriptor_only"
)
_DEFAULT_CAPS = OperatorPhaseCliquePipelineCaps()


def _pipeline_clique_caps(
    caps: OperatorPhaseCliquePipelineCaps,
) -> OperatorPhaseCliqueCaps:
    return OperatorPhaseCliqueCaps(
        max_parent_variables=caps.max_parent_variables,
        max_parent_rows=caps.max_parent_rows,
        max_parent_nonzeros=caps.max_parent_nonzeros,
        max_parent_buffer_items=caps.max_parent_buffer_items,
        max_top_literals=_K4_TOP_LITERALS,
        max_total_pairs=_K4_TOTAL_PAIRS,
        max_cliques=caps.max_cliques,
        max_clique_search_nodes=caps.max_clique_search_nodes,
        max_source_terms=caps.max_source_terms,
        max_multiplier_bits=caps.max_multiplier_bits,
        max_exact_bits=caps.max_exact_bits,
        max_exact_nonzeros=caps.max_exact_nonzeros,
    )


def _operator_core_identity(hz: Any) -> Tuple[int, ...]:
    """Match the producer seal's exact flat core-object ordering."""

    if type(hz) is not SparseHZono:
        raise _PipelineFallback("handoff_source_hz_wrong_type")
    live = vars(hz)
    objects = [hz]
    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        value = live.get(name)
        if type(value) is not np.ndarray:
            raise _PipelineFallback(
                f"handoff_source_dense_malformed_{name}"
            )
        objects.append(value)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        matrix = live.get(name)
        if type(matrix) is not sp.csr_matrix:
            raise _PipelineFallback(
                f"handoff_source_csr_malformed_{name}"
            )
        objects.append(matrix)
        for buffer_name in ("data", "indices", "indptr"):
            value = vars(matrix).get(buffer_name)
            if type(value) is not np.ndarray:
                raise _PipelineFallback(
                    "handoff_source_csr_buffer_malformed_"
                    f"{name}_{buffer_name}"
                )
            objects.append(value)
    return tuple(id(value) for value in objects)


def _source_frame_inputs(
    build: OperatorHZBuild,
) -> Tuple[Tuple[str, ...], Mapping[str, np.ndarray]]:
    hz = build.hz
    tags = getattr(hz, "_solver_constraint_row_tags", None)
    if (
        type(tags) is not tuple
        or len(tags) != hz.n_eq + hz.n_ub
        or any(type(tag) is not str for tag in tags)
    ):
        raise _PipelineFallback("handoff_source_tags_malformed")
    provenance = {
        name: getattr(hz, name, None)
        for name in (
            "full_col_ids",
            "operator_input_center",
            "operator_input_radius",
            "_solver_continuous_column_layer_ids",
        )
    }
    if any(type(value) is not np.ndarray for value in provenance.values()):
        raise _PipelineFallback("handoff_source_provenance_malformed")
    return tags, provenance


def _make_pipeline_capability(
    semantic_digest: str,
) -> OperatorPhaseCliquePipelineSolverCapability:
    return OperatorPhaseCliquePipelineSolverCapability(
        token=secrets.token_hex(32),
        process_id=os.getpid(),
        expires_monotonic=(
            time.monotonic() + _PIPELINE_HANDOFF_TTL_SECONDS
        ),
        semantic_digest=semantic_digest,
        _producer_capability=_PIPELINE_HANDOFF_PRODUCER,
    )


def _sweep_pipeline_handoffs_locked(now: float) -> None:
    process_id = os.getpid()
    stale = tuple(
        token
        for token, record in _PIPELINE_HANDOFF_REGISTRY.items()
        if (
            record.process_id != process_id
            or record.expires_monotonic <= now
            or record.capability_ref() is None
            or record.result_ref() is None
            or record.source_build_ref() is None
            or record.public_build_ref() is None
            or record.public_hz_ref() is None
        )
    )
    for token in stale:
        _PIPELINE_HANDOFF_REGISTRY.pop(token, None)


def _register_pipeline_solver_handoff(
    source_build: OperatorHZBuild,
    result: OperatorPhaseCliquePipelineResult,
    private_build: OperatorHZBuild,
    *,
    semantic_digest: str,
) -> None:
    capability = result.solver_handoff_capability
    if (
        type(capability)
        is not OperatorPhaseCliquePipelineSolverCapability
        or not _valid_sha256(semantic_digest)
        or capability.semantic_digest != semantic_digest
    ):
        raise _PipelineFallback("pipeline_handoff_issue_header_invalid")
    try:
        public_objects = _solver_owned_objects(result.build)
        private_objects = _solver_owned_objects(private_build)
    except Exception as exc:
        raise _PipelineFallback("pipeline_handoff_owner_malformed") from exc
    record = _PipelineSolverHandoffRecord(
        capability_ref=weakref.ref(capability),
        result_ref=weakref.ref(result),
        source_build_ref=weakref.ref(source_build),
        public_build_ref=weakref.ref(result.build),
        public_hz_ref=weakref.ref(result.build.hz),
        public_owned_identity=tuple(id(value) for value in public_objects),
        private_build=private_build,
        private_owned_identity=tuple(id(value) for value in private_objects),
        receipt_object=result.receipt,
        receipt_sha256=str(result.receipt["receipt_sha256"]),
        status=result.status,
        materialized=result.materialized,
        semantic_digest=semantic_digest,
        expires_monotonic=capability.expires_monotonic,
        process_id=capability.process_id,
    )
    with _PIPELINE_HANDOFF_REGISTRY_LOCK:
        now = time.monotonic()
        _sweep_pipeline_handoffs_locked(now)
        if (
            now >= capability.expires_monotonic
            or len(_PIPELINE_HANDOFF_REGISTRY)
            >= _PIPELINE_HANDOFF_REGISTRY_CAPACITY
            or capability.token in _PIPELINE_HANDOFF_REGISTRY
        ):
            raise _PipelineFallback("pipeline_handoff_registry_full")
        _PIPELINE_HANDOFF_REGISTRY[capability.token] = record


def _validate_pipeline_solver_handoff_registration(
    source_build: Any,
    result: Any,
) -> bool:
    try:
        if (
            type(source_build) is not OperatorHZBuild
            or type(result) is not OperatorPhaseCliquePipelineResult
        ):
            return False
        capability = result.solver_handoff_capability
        if (
            type(capability)
            is not OperatorPhaseCliquePipelineSolverCapability
            or capability.proof_authority is not False
            or capability.process_id != os.getpid()
            or not _valid_sha256(capability.semantic_digest)
        ):
            return False
        with _PIPELINE_HANDOFF_REGISTRY_LOCK:
            now = time.monotonic()
            _sweep_pipeline_handoffs_locked(now)
            record = _PIPELINE_HANDOFF_REGISTRY.get(capability.token)
            return bool(
                record is not None
                and record.process_id == os.getpid()
                and record.capability_ref() is capability
                and record.result_ref() is result
                and record.source_build_ref() is source_build
                and record.public_build_ref() is result.build
                and record.public_hz_ref() is result.build.hz
                and record.expires_monotonic
                == capability.expires_monotonic
                and record.expires_monotonic > now
                and record.semantic_digest == capability.semantic_digest
                and record.receipt_object is result.receipt
                and record.receipt_sha256
                == result.receipt.get("receipt_sha256")
                and record.status == result.status
                and record.materialized is result.materialized
            )
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_exception_details(
    exc: Exception,
) -> Tuple[bool, str, str]:
    """Classify one exception without invoking candidate formatting hooks."""

    timeout = False
    reason = "downstream_error"
    type_name = "Exception"
    try:
        exception_type = type(exc)
        raw_name = type.__getattribute__(
            exception_type, "__name__"
        )
        if type(raw_name) is str and raw_name:
            type_name = raw_name[:128]
    except BaseException:
        pass
    if type(exc) is _PipelineFallback:
        try:
            raw_timeout = object.__getattribute__(exc, "timeout")
            raw_reason = object.__getattribute__(exc, "reason")
            timeout = raw_timeout is True
            if type(raw_reason) is str and raw_reason:
                reason = raw_reason[:256]
        except BaseException:
            timeout = False
            reason = "pipeline_fallback_unreadable"
        return timeout, reason, type_name
    try:
        timeout = isinstance(exc, TimeoutError)
    except BaseException:
        timeout = False
    tokens = [type_name.lower()]
    try:
        raw_args = object.__getattribute__(exc, "args")
        if type(raw_args) is tuple:
            tokens.extend(
                item.lower()[:256]
                for item in raw_args
                if type(item) is str
            )
    except BaseException:
        pass
    if timeout or any(
        "timeout" in token or "deadline" in token
        for token in tokens
    ):
        timeout = True
        reason = "downstream_timeout"
    return timeout, reason, type_name


def _array_identity(value: Any, *, name: str) -> Tuple[Any, ...]:
    if type(value) is not np.ndarray:
        raise _PipelineFallback(f"terminal_{name}_not_ndarray")
    return (
        name,
        id(value),
        id(value.base) if value.base is not None else None,
        int(value.ctypes.data),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.strides),
        value.dtype.str,
    )


def _csr_identity(value: Any, *, name: str) -> Tuple[Any, ...]:
    if type(value) is not sp.csr_matrix:
        raise _PipelineFallback(f"terminal_{name}_not_exact_csr")
    return (
        name,
        id(value),
        tuple(int(item) for item in value.shape),
        value.dtype.str,
        _array_identity(value.data, name=f"{name}.data"),
        _array_identity(value.indices, name=f"{name}.indices"),
        _array_identity(value.indptr, name=f"{name}.indptr"),
    )


def _hz_core_identity(hz: Any) -> Tuple[Any, ...]:
    """Snapshot exact core object/buffer identities without copying data."""

    if type(hz) is not SparseHZono:
        raise _PipelineFallback("terminal_hz_wrong_type")
    dense = tuple(
        _array_identity(getattr(hz, name), name=name)
        for name in ("c", "b", "ub", "col_ids", "bcol_ids")
    )
    sparse = tuple(
        _csr_identity(getattr(hz, name), name=name)
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")
    )
    return (
        id(hz),
        int(hz.n_out),
        int(hz.n_cont),
        int(hz.n_bin),
        int(hz.n_eq),
        int(hz.n_ub),
        dense,
        sparse,
    )


def _terminal_semantic_pair_seal(
    source_hz: Any,
    fresh_hz: Any,
    *,
    expected_source_digest: str,
    expected_fresh_digest: str,
    deadline: float,
    stage: str,
) -> Tuple[str, str]:
    """Double-digest immutable identities immediately before acceptance."""

    _check_deadline(deadline, stage=f"{stage}_before_identity")
    if (
        source_hz is fresh_hz
        or not _valid_sha256(expected_source_digest)
        or not _valid_sha256(expected_fresh_digest)
    ):
        raise _PipelineFallback(f"{stage}_header_invalid")
    source_identity = _hz_core_identity(source_hz)
    fresh_identity = _hz_core_identity(fresh_hz)
    source_first = sparse_hz_semantic_digest(source_hz)
    fresh_first = sparse_hz_semantic_digest(fresh_hz)
    _check_deadline(deadline, stage=f"{stage}_between_digests")
    if (
        _hz_core_identity(source_hz) != source_identity
        or _hz_core_identity(fresh_hz) != fresh_identity
    ):
        raise _PipelineFallback(f"{stage}_core_identity_changed")
    source_second = sparse_hz_semantic_digest(source_hz)
    fresh_second = sparse_hz_semantic_digest(fresh_hz)
    _check_deadline(deadline, stage=f"{stage}_after_digests")
    if (
        _hz_core_identity(source_hz) != source_identity
        or _hz_core_identity(fresh_hz) != fresh_identity
        or source_first != source_second
        or fresh_first != fresh_second
        or source_second != expected_source_digest
        or fresh_second != expected_fresh_digest
    ):
        raise _PipelineFallback(f"{stage}_semantic_seal_changed")
    return source_second, fresh_second


def _builtin_copy(value: Any) -> Any:
    """Copy an exact builtin JSON/tuple tree without candidate equality."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise OperatorPhaseCliquePipelineError(
                "receipt_contains_nonfinite_float"
            )
        return value
    if type(value) in {dict, MappingProxyType}:
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise OperatorPhaseCliquePipelineError(
                    "receipt_key_not_builtin_string"
                )
            result[key] = _builtin_copy(item)
        return result
    if type(value) is list:
        return [_builtin_copy(item) for item in value]
    if type(value) is tuple:
        return tuple(_builtin_copy(item) for item in value)
    raise OperatorPhaseCliquePipelineError(
        "receipt_contains_non_builtin_value"
    )


def _deep_freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    if type(value) is tuple:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _checksummed_receipt(body: Mapping[str, Any]) -> Mapping[str, Any]:
    copied = _builtin_copy(body)
    if type(copied) is not dict:
        raise OperatorPhaseCliquePipelineError(
            "receipt_body_not_builtin_mapping"
        )
    receipt = dict(copied)
    receipt["receipt_sha256"] = _canonical_sha256(copied)
    return receipt


def _receipt_checksum_valid(receipt: Any) -> bool:
    try:
        copied = _builtin_copy(receipt)
        if (
            type(copied) is not dict
            or not _valid_sha256(copied.get("receipt_sha256"))
        ):
            return False
        observed = copied.pop("receipt_sha256")
        return _canonical_sha256(copied) == observed
    except (
        OperatorPhaseCliquePipelineError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


def _disabled_receipt() -> Mapping[str, Any]:
    return _checksummed_receipt(
        {
            "schema": _PIPELINE_SCHEMA,
            "enabled": False,
            "status": "no_op_disabled",
            "candidate_attempted": False,
            "candidate_only": True,
            "proof_authority": False,
            "identity_preserved": True,
            "materialized": False,
            "materialization_receipt_sha256": None,
            "verdict_path": _VERDICT_PATH,
            "candidate_budget_fraction": _CANDIDATE_FRACTION,
            "materializer_reserve_fraction": _MATERIALIZER_FRACTION,
            "timings": {
                "total_seconds": 0.0,
            },
        }
    )


def _normalize_deadline(value: Any) -> float:
    if (
        type(value) not in {int, float}
        or type(value) is bool
    ):
        raise _PipelineFallback("deadline_not_builtin_numeric")
    deadline = float(value)
    if not math.isfinite(deadline):
        raise _PipelineFallback("deadline_nonfinite")
    if time.monotonic() >= deadline:
        raise _PipelineFallback(
            "deadline_expired_before_pipeline", timeout=True
        )
    return deadline


def _check_deadline(deadline: float, *, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise _PipelineFallback(
            f"deadline_expired_{stage}", timeout=True
        )


def _normalize_caps(value: Any) -> OperatorPhaseCliquePipelineCaps:
    if value is None:
        return _DEFAULT_CAPS
    if type(value) is not OperatorPhaseCliquePipelineCaps:
        raise _PipelineFallback("pipeline_caps_wrong_type")
    # Reconstructing avoids trusting a post-construction ``object.__setattr__``.
    try:
        return OperatorPhaseCliquePipelineCaps(
            **{
                name: getattr(value, name)
                for name in value.__dataclass_fields__
            }
        )
    except (
        AttributeError,
        OperatorPhaseCliquePipelineError,
        TypeError,
        ValueError,
    ) as exc:
        raise _PipelineFallback("pipeline_caps_invalid") from exc


def _fallback_private_solver_build(
    build: Any,
    *,
    audit: Mapping[str, Any],
) -> OperatorHZBuild:
    """Recover an unchanged baseline only through an isolated private copy."""

    if type(build) is not OperatorHZBuild:
        raise _PipelineFallback("fallback_source_build_wrong_type")
    deadline = audit.get("_operation_deadline")
    caps = audit.get("_clique_caps")
    expected_digest = audit.get("source_parent_semantic_digest")
    expected_frame_digest = audit.get("source_frame_digest")
    if (
        type(deadline) is not float
        or not math.isfinite(deadline)
        or type(caps) is not OperatorPhaseCliqueCaps
        or not _valid_sha256(expected_digest)
        or not _valid_sha256(expected_frame_digest)
    ):
        raise _PipelineFallback("fallback_source_seal_unavailable")
    _check_deadline(deadline, stage="fallback_private_begin")

    producer_seal = build.constructive_nonempty_seal
    producer_reason: Optional[str] = None
    if producer_seal is not None:
        owner_identity = _operator_core_identity(build.hz)
        if not validate_operator_hz_constructive_nonempty_seal(
            producer_seal,
            owner_build=build,
            owner_hz=build.hz,
            owner_core_identity=owner_identity,
            private_parent_semantic_digest=expected_digest,
        ):
            # Production phase-clique builds always carry this scalable seal.
            # A present-but-invalid seal indicates mutation/ownership loss and
            # must not fall through to the toy Fraction witness path.
            raise _PipelineFallback(
                "fallback_producer_seal_invalid"
            )
        producer_reason = producer_seal.reason

    try:
        private_build = _private_solver_build(
            build,
            caps=caps,
            expected_fresh_semantic_digest=expected_digest,
            expected_fresh_frame_digest=expected_frame_digest,
            constructive_reason=producer_reason,
            deadline=deadline,
        )
    except Exception as exc:
        raise _PipelineFallback(
            "fallback_private_snapshot_rejected"
        ) from exc
    if producer_reason is None:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0:
            raise _PipelineFallback(
                "fallback_private_nonempty_deadline", timeout=True
            )
        if not _cut_has_exact_private_nonempty_witness(
            private_build.hz,
            caps=caps,
            deadline=deadline,
        ):
            raise _PipelineFallback(
                "fallback_private_nonempty_unproved"
            )
        # The exact private witness path replays stored binary64 constraints
        # with ``Fraction``.  It is toy-only; production reaches this point
        # through the scalable producer seal above.
        hz_mark_constructively_nonempty(
            private_build.hz,
            "operator_hz_exact_private_fallback_snapshot_v1",
        )
    if not hz_constructively_nonempty(private_build.hz):
        raise _PipelineFallback(
            "fallback_private_constructive_token_missing"
        )
    return private_build


def _snapshot_b1_bounds(
    lower: Any,
    upper: Any,
    *,
    output_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Require complete canonical float64 ``[1, n_out]`` snapshots."""

    values = []
    for name, raw in (("lower", lower), ("upper", upper)):
        if (
            type(raw) is not np.ndarray
            or raw.dtype != np.dtype(np.float64)
            or raw.ndim != 2
            or raw.shape != (1, output_width)
            or not raw.flags.c_contiguous
            or not np.all(np.isfinite(raw))
        ):
            raise _PipelineFallback(f"output_{name}_snapshot_invalid")
        values.append(raw.copy(order="C"))
    lower_copy, upper_copy = values
    if np.any(lower_copy > upper_copy):
        raise _PipelineFallback("output_interval_order_invalid")
    return lower_copy, upper_copy


def _exact_interval_upper_violations(
    rivals: Sequence[RivalSpec],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    deadline: float,
) -> Tuple[Tuple[int, int], ...]:
    """Exact dyadic interval scores for all rivals (scheduling only)."""

    if (
        type(rivals) is not tuple
        or lower.ndim != 1
        or upper.ndim != 1
        or lower.shape != upper.shape
    ):
        raise _PipelineFallback("interval_hardness_shape_invalid")
    scores = []
    work = 0
    for rival in rivals:
        _check_deadline(deadline, stage="interval_hardness")
        if (
            type(rival) is not RivalSpec
            or type(rival.objective) is not tuple
            or len(rival.objective) != int(lower.size)
            or any(type(value) is not float for value in rival.objective)
            or any(
                not math.isfinite(value) for value in rival.objective
            )
            or type(rival.threshold) is not float
            or not math.isfinite(rival.threshold)
        ):
            raise _PipelineFallback(
                "interval_hardness_rival_invalid"
            )
        value = -Fraction.from_float(rival.threshold)
        for coefficient, lower_value, upper_value in zip(
            rival.objective, lower, upper
        ):
            endpoint = (
                float(upper_value)
                if coefficient >= 0.0
                else float(lower_value)
            )
            value += (
                Fraction.from_float(coefficient)
                * Fraction.from_float(endpoint)
            )
            work += 1
            if work % 1024 == 0:
                _check_deadline(
                    deadline, stage="interval_hardness_cells"
                )
        scores.append((value.numerator, value.denominator))
    return tuple(scores)


def _interval_frame_digest(
    *,
    build_digest: str,
    batch_sha256: str,
    live_assert_sha256: str,
    property_digest: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> str:
    if not all(
        _valid_sha256(value)
        for value in (
            build_digest,
            batch_sha256,
            live_assert_sha256,
            property_digest,
        )
    ):
        raise _PipelineFallback("interval_frame_parent_digest_invalid")
    digest = hashlib.sha256()
    digest.update(_INTERVAL_SCHEMA.encode("ascii") + b"\0")
    for value in (
        build_digest,
        batch_sha256,
        live_assert_sha256,
        property_digest,
    ):
        digest.update(value.encode("ascii"))
    for name, value in (("lower", lower), ("upper", upper)):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(
            np.asarray(value.shape, dtype=np.int64).tobytes()
        )
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _materializer_kwargs(
    caps: OperatorPhaseCliquePipelineCaps,
    *,
    selection_timeout_seconds: float,
) -> Mapping[str, Any]:
    return {
        "selection_max_rivals": 1,
        "selection_max_binaries": caps.max_binaries,
        "selection_max_work_items": caps.max_selection_work_items,
        "selection_timeout_seconds": selection_timeout_seconds,
        "max_parent_variables": caps.max_parent_variables,
        "max_parent_rows": caps.max_parent_rows,
        "max_parent_nonzeros": caps.max_parent_nonzeros,
        "max_parent_buffer_items": caps.max_parent_buffer_items,
        "max_top_literals": _K4_TOP_LITERALS,
        "max_total_pairs": _K4_TOTAL_PAIRS,
        "max_cliques": caps.max_cliques,
        "max_clique_search_nodes": caps.max_clique_search_nodes,
        "max_source_terms": caps.max_source_terms,
        "max_multiplier_bits": caps.max_multiplier_bits,
        "max_exact_bits": caps.max_exact_bits,
        "max_exact_nonzeros": caps.max_exact_nonzeros,
    }


def _require_k4_candidate(
    result: Any,
    *,
    deadline: float,
) -> OperatorExactReLUPhaseCliqueResult:
    _check_deadline(deadline, stage="k4_candidate_shape")
    telemetry = (
        result.telemetry
        if type(result) is OperatorExactReLUPhaseCliqueResult
        else None
    )
    telemetry_keys = {
        "schema",
        "caps",
        "ranked_literal_count",
        "omitted_zero_count",
        "excluded_selected_count",
        "pair_count",
        "certified_edge_count",
        "exact_certificate_count",
        "clique_count",
        "clique_search_nodes",
        "clique_search_truncated",
        "model_builds",
        "oracle",
        "proof_authority",
    }
    if (
        type(result) is not OperatorExactReLUPhaseCliqueResult
        or result.proof_authority is not False
        or result.status
        != _COMPACT_SUCCESS_STATUS
        or result.hz is not None
        or type(result.caps) is not OperatorPhaseCliqueCaps
        or type(result.ranked_phases) is not tuple
        or len(result.ranked_phases) != _K4_TOP_LITERALS
        or any(
            type(item) is not RankedOperatorPhase
            for item in result.ranked_phases
        )
        or type(result.literals) is not tuple
        or len(result.literals) != _K4_TOP_LITERALS
        or any(type(item) is not PhaseLiteral for item in result.literals)
        or type(result.pair_records) is not tuple
        or len(result.pair_records) != _K4_TOTAL_PAIRS
        or any(
            type(item) is not PersistentPairRecord
            or item.status != "certified_conflict"
            or not _valid_sha256(item.certificate_sha256)
            for item in result.pair_records
        )
        or type(result.certificates) is not tuple
        or len(result.certificates) != _K4_TOTAL_PAIRS
        or any(
            type(item) is not ExactDualRayConflictCertificate
            or item.proof_authority is not False
            or not _valid_sha256(item.certificate_sha256)
            for item in result.certificates
        )
        or type(result.cliques) is not tuple
        or len(result.cliques) != 1
        or type(result.cliques[0]) is not OperatorCertifiedPhaseClique
        or result.cliques[0].proof_authority is not False
        or result.cliques[0].cut_applied is not True
        or type(result.cliques[0].literals) is not tuple
        or len(result.cliques[0].literals) != _K4_TOP_LITERALS
        or type(telemetry) is not dict
        or set(telemetry) != telemetry_keys
        or telemetry.get("schema") != _COMPACT_TELEMETRY_SCHEMA
        or telemetry.get("proof_authority") is not False
        or telemetry.get("ranked_literal_count") != _K4_TOP_LITERALS
        or telemetry.get("pair_count") != _K4_TOTAL_PAIRS
        or telemetry.get("certified_edge_count") != _K4_TOTAL_PAIRS
        or telemetry.get("exact_certificate_count") != _K4_TOTAL_PAIRS
        or telemetry.get("clique_count") != 1
        or telemetry.get("model_builds") != 1
        or type(telemetry.get("caps")) is not dict
        or telemetry.get("caps") != vars(result.caps)
        or type(telemetry.get("oracle")) is not dict
    ):
        raise _PipelineFallback("no_complete_k4_clique")
    _compact_candidate_route_summary(result)
    return result


def _compact_candidate_descriptor_sha256(
    result: OperatorExactReLUPhaseCliqueResult,
) -> str:
    """Bind every compact exact primitive without retaining its objects."""

    try:
        payload = {
            "schema": (
                "act.operator_phase_clique_compact_binding.v1"
            ),
            "status": result.status,
            "telemetry_schema": result.telemetry["schema"],
            "parent_semantic_digest": result.parent_semantic_digest,
            "focused_property_digest": result.focused_property_digest,
            "operator_row_tag_digest": result.operator_row_tag_digest,
            "selection_digest": result.selection_digest,
            "subset_binding_digest": result.subset_binding_digest,
            "ordered_source_frame_sha256": (
                result.ordered_source_frame_sha256
            ),
            "ranked_phases": tuple(
                (
                    item.rank,
                    item.stable_bcol_id,
                    item.phase,
                    item.score_numerator,
                    item.score_denominator,
                )
                for item in result.ranked_phases
            ),
            "literals": tuple(
                (
                    item.stable_bcol_id,
                    item.phase,
                    item.binding_digest,
                )
                for item in result.literals
            ),
            "pairs": tuple(
                (
                    tuple(
                        (
                            literal.stable_bcol_id,
                            literal.phase,
                            literal.binding_digest,
                        )
                        for literal in item.literals
                    ),
                    item.status,
                    item.ray_nonzero_rows,
                    item.certificate_sha256,
                    item.rationalization,
                )
                for item in result.pair_records
            ),
            "certificate_sha256s": tuple(
                item.certificate_sha256
                for item in result.certificates
            ),
            "cliques": tuple(
                (
                    item.clique_id,
                    tuple(
                        (
                            literal.stable_bcol_id,
                            literal.phase,
                            literal.binding_digest,
                        )
                        for literal in item.literals
                    ),
                    item.edge_certificate_sha256s,
                    item.total_score_numerator,
                    item.total_score_denominator,
                )
                for item in result.cliques
            ),
            "caps": dict(result.telemetry["caps"]),
            "proof_authority": False,
        }
        digest = _canonical_sha256(payload)
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise _PipelineFallback(
            "compact_candidate_descriptor_malformed"
        ) from exc
    if not _valid_sha256(digest):
        raise _PipelineFallback(
            "compact_candidate_descriptor_digest_invalid"
        )
    return digest


def _compact_candidate_route_summary(
    result: OperatorExactReLUPhaseCliqueResult,
) -> Mapping[str, Any]:
    """Freeze the exact compact route and bounded oracle telemetry."""

    try:
        oracle = result.telemetry["oracle"]
        expected_oracle_keys = {
            "backend",
            "highs_version",
            "row_order",
            "presolve",
            "threads",
            "model_builds",
            "candidate_rows",
            "candidate_columns",
            "candidate_nonzeros",
            "candidate_load_mode",
            "binary_change_coefficient_cap",
            "base_solve_calls",
            "solve_calls",
            "bound_update_calls",
            "dual_ray_calls",
            "highs_cumulative_run_time_seconds",
        }
        if type(oracle) is not dict or set(oracle) != expected_oracle_keys:
            raise ValueError("oracle_exact_keys")
        status_counts = {
            status: sum(
                record.status == status
                for record in result.pair_records
            )
            for status in (
                "certified_conflict",
                "feasible_or_unknown",
                "infeasible_without_ray",
                "exact_replay_rejected",
            )
        }
        summary = {
            "schema": (
                "act.operator_phase_clique_compact_route.v1"
            ),
            "result_mode": "compact_exact_descriptor_v1",
            "result_status": result.status,
            "telemetry_schema": result.telemetry["schema"],
            "hz_absent": result.hz is None,
            "oracle_backend": oracle["backend"],
            "oracle_presolve": oracle["presolve"],
            "candidate_load_mode": oracle["candidate_load_mode"],
            "binary_change_coefficient_cap": oracle[
                "binary_change_coefficient_cap"
            ],
            "candidate_rows": oracle["candidate_rows"],
            "candidate_columns": oracle["candidate_columns"],
            "candidate_nonzeros": oracle["candidate_nonzeros"],
            "model_builds": oracle["model_builds"],
            "solve_calls": oracle["solve_calls"],
            "base_solve_calls": oracle["base_solve_calls"],
            "pair_count": len(result.pair_records),
            "pair_status_counts": status_counts,
            "completed_pair_count": sum(status_counts.values()),
            "proof_authority": False,
        }
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise _PipelineFallback(
            "compact_candidate_route_malformed"
        ) from exc
    integer_fields = (
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "model_builds",
        "solve_calls",
        "base_solve_calls",
        "pair_count",
        "completed_pair_count",
    )
    if (
        summary["result_status"] != _COMPACT_SUCCESS_STATUS
        or summary["telemetry_schema"]
        != _COMPACT_TELEMETRY_SCHEMA
        or summary["hz_absent"] is not True
        or (
            summary["oracle_backend"],
            summary["oracle_presolve"],
        )
        not in {
            ("highspy_persistent_simplex_dual_ray_v1", "off"),
            (
                "highspy_persistent_simplex_presolve_lazy_dual_ray_v2",
                "on",
            ),
        }
        or summary["candidate_load_mode"]
        not in {
            "split_continuous_rows_binary_change_coeff_v1",
            "single_merged_csr_binary_cap_fallback_v1",
        }
        or type(summary["binary_change_coefficient_cap"]) is not int
        or summary["binary_change_coefficient_cap"]
        != _MAX_BINARY_CHANGE_COEFFICIENTS
        or any(
            type(summary[name]) is not int or summary[name] < 0
            for name in integer_fields
        )
        or summary["candidate_rows"] < 1
        or summary["candidate_columns"] < 1
        or summary["candidate_nonzeros"] < 1
        or summary["model_builds"] != 1
        or summary["solve_calls"] != _K4_TOTAL_PAIRS
        or summary["base_solve_calls"] != 0
        or summary["pair_count"] != _K4_TOTAL_PAIRS
        or summary["completed_pair_count"] != _K4_TOTAL_PAIRS
        or summary["pair_status_counts"]
        != {
            "certified_conflict": _K4_TOTAL_PAIRS,
            "feasible_or_unknown": 0,
            "infeasible_without_ray": 0,
            "exact_replay_rejected": 0,
        }
    ):
        raise _PipelineFallback(
            "compact_candidate_route_contract_invalid"
        )
    return summary


def _compact_route_receipt_is_exact(
    route: Any,
    *,
    source_hz: SparseHZono,
    deadline: float,
) -> bool:
    expected_keys = {
        "schema",
        "result_mode",
        "result_status",
        "telemetry_schema",
        "hz_absent",
        "oracle_backend",
        "oracle_presolve",
        "candidate_load_mode",
        "binary_change_coefficient_cap",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "model_builds",
        "solve_calls",
        "base_solve_calls",
        "pair_count",
        "pair_status_counts",
        "completed_pair_count",
        "proof_authority",
    }
    if (
        type(route) is not dict
        or set(route) != expected_keys
        or route.get("schema")
        != "act.operator_phase_clique_compact_route.v1"
        or route.get("result_mode")
        != "compact_exact_descriptor_v1"
        or route.get("result_status") != _COMPACT_SUCCESS_STATUS
        or route.get("telemetry_schema")
        != _COMPACT_TELEMETRY_SCHEMA
        or route.get("hz_absent") is not True
        or route.get("proof_authority") is not False
        or (
            route.get("oracle_backend"),
            route.get("oracle_presolve"),
        )
        not in {
            ("highspy_persistent_simplex_dual_ray_v1", "off"),
            (
                "highspy_persistent_simplex_presolve_lazy_dual_ray_v2",
                "on",
            ),
        }
        or route.get("candidate_load_mode")
        not in {
            "split_continuous_rows_binary_change_coeff_v1",
            "single_merged_csr_binary_cap_fallback_v1",
        }
        or type(route.get("binary_change_coefficient_cap")) is not int
        or route.get("binary_change_coefficient_cap")
        != _MAX_BINARY_CHANGE_COEFFICIENTS
    ):
        return False
    integer_fields = (
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "model_builds",
        "solve_calls",
        "base_solve_calls",
        "pair_count",
        "completed_pair_count",
    )
    if any(
        type(route.get(name)) is not int or route[name] < 0
        for name in integer_fields
    ):
        return False
    status_counts = route.get("pair_status_counts")
    if (
        type(status_counts) is not dict
        or set(status_counts)
        != {
            "certified_conflict",
            "feasible_or_unknown",
            "infeasible_without_ray",
            "exact_replay_rejected",
        }
        or any(type(value) is not int for value in status_counts.values())
        or status_counts
        != {
            "certified_conflict": _K4_TOTAL_PAIRS,
            "feasible_or_unknown": 0,
            "infeasible_without_ray": 0,
            "exact_replay_rejected": 0,
        }
    ):
        return False
    kept_counts = {}
    for name in ("Auc", "Aub", "Ac", "Ab"):
        data = getattr(source_hz, name).data
        kept = 0
        for start in range(0, int(data.size), 1 << 18):
            if time.monotonic() >= deadline:
                return False
            chunk = data[start : start + (1 << 18)]
            kept += int(
                np.count_nonzero(
                    np.abs(chunk) > _CANDIDATE_DUST_ABS
                )
            )
        kept_counts[name] = kept
    expected_nonzeros = sum(kept_counts.values())
    binary_nonzeros = kept_counts["Aub"] + kept_counts["Ab"]
    expected_load_mode = (
        "split_continuous_rows_binary_change_coeff_v1"
        if binary_nonzeros <= _MAX_BINARY_CHANGE_COEFFICIENTS
        else "single_merged_csr_binary_cap_fallback_v1"
    )
    return bool(
        route["candidate_rows"] == source_hz.n_ub + source_hz.n_eq
        and route["candidate_columns"]
        == source_hz.n_cont + source_hz.n_bin
        and route["candidate_nonzeros"] == expected_nonzeros
        and route["candidate_load_mode"] == expected_load_mode
        and route["model_builds"] == 1
        and route["solve_calls"] == _K4_TOTAL_PAIRS
        and route["base_solve_calls"] == 0
        and route["pair_count"] == _K4_TOTAL_PAIRS
        and route["completed_pair_count"] == _K4_TOTAL_PAIRS
    )


def _snapshot_candidate_bindings(
    *,
    batch: Any,
    hardness: Any,
    focused: Any,
    selection: Any,
    clique_result: OperatorExactReLUPhaseCliqueResult,
    focused_encoded_row: int,
    interval_frame_sha256: str,
    deadline: float,
) -> _CandidateBindingSnapshot:
    """Capture every later receipt value before materializer verification."""

    _check_deadline(deadline, stage="candidate_binding_snapshot")
    try:
        descriptor_sha256 = _compact_candidate_descriptor_sha256(
            clique_result
        )
        route_summary = dict(
            _compact_candidate_route_summary(clique_result)
        )
        full_rival_count = len(batch.rivals)
        clique_ids = tuple(
            clique.clique_id for clique in clique_result.cliques
        )
        certified_edges = sum(
            record.status == "certified_conflict"
            for record in clique_result.pair_records
        )
        snapshot = _CandidateBindingSnapshot(
            full_rival_count=full_rival_count,
            focused_encoded_row=focused_encoded_row,
            ranked_literal_count=len(clique_result.ranked_phases),
            pair_count=len(clique_result.pair_records),
            certified_edge_count=certified_edges,
            clique_count=len(clique_result.cliques),
            full_batch_sha256=batch.batch_sha256,
            full_live_assert_sha256=batch.live_assert_sha256,
            full_property_digest=hardness.full_property_digest,
            interval_frame_sha256=interval_frame_sha256,
            hardness_vector_digest=hardness.vector_digest,
            focused_subset_digest=focused.focused_subset_digest,
            selection_digest=selection.selection_digest,
            focused_property_digest=selection.property_digest,
            subset_binding_digest=(
                clique_result.subset_binding_digest
            ),
            clique_ids=clique_ids,
            candidate_result_status=clique_result.status,
            candidate_telemetry_schema=(
                clique_result.telemetry["schema"]
            ),
            candidate_representation=(
                _COMPACT_CANDIDATE_REPRESENTATION
            ),
            candidate_descriptor_sha256=descriptor_sha256,
            candidate_route_summary=route_summary,
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise _PipelineFallback(
            "candidate_binding_snapshot_failed"
        ) from exc
    integer_values = (
        snapshot.full_rival_count,
        snapshot.focused_encoded_row,
        snapshot.ranked_literal_count,
        snapshot.pair_count,
        snapshot.certified_edge_count,
        snapshot.clique_count,
    )
    digest_values = (
        snapshot.full_batch_sha256,
        snapshot.full_live_assert_sha256,
        snapshot.full_property_digest,
        snapshot.interval_frame_sha256,
        snapshot.hardness_vector_digest,
        snapshot.focused_subset_digest,
        snapshot.selection_digest,
        snapshot.focused_property_digest,
        snapshot.subset_binding_digest,
        snapshot.candidate_descriptor_sha256,
        *snapshot.clique_ids,
    )
    if (
        any(type(value) is not int for value in integer_values)
        or snapshot.full_rival_count < 1
        or snapshot.focused_encoded_row < 0
        or snapshot.focused_encoded_row
        >= snapshot.full_rival_count
        or snapshot.ranked_literal_count != _K4_TOP_LITERALS
        or snapshot.pair_count != _K4_TOTAL_PAIRS
        or snapshot.certified_edge_count != _K4_TOTAL_PAIRS
        or snapshot.clique_count != 1
        or type(snapshot.clique_ids) is not tuple
        or len(snapshot.clique_ids) != 1
        or snapshot.candidate_result_status
        != _COMPACT_SUCCESS_STATUS
        or snapshot.candidate_telemetry_schema
        != _COMPACT_TELEMETRY_SCHEMA
        or snapshot.candidate_representation
        != _COMPACT_CANDIDATE_REPRESENTATION
        or type(snapshot.candidate_route_summary) is not dict
        or snapshot.candidate_route_summary
        != _compact_candidate_route_summary(clique_result)
        or any(not _valid_sha256(value) for value in digest_values)
    ):
        raise _PipelineFallback(
            "candidate_binding_snapshot_invalid"
        )
    _check_deadline(
        deadline, stage="candidate_binding_snapshot_complete"
    )
    return snapshot


def _validate_materialization(
    source_build: OperatorHZBuild,
    materialized: Any,
    *,
    source_digest: str,
    source_n_ub: int,
    selection_digest: str,
    property_digest: str,
    subset_digest: str,
    clique_ids: Tuple[str, ...],
    deadline: float,
) -> Mapping[str, Any]:
    """Recheck the fresh object and the complete materializer v2 receipt."""

    _check_deadline(deadline, stage="materialization_receipt_begin")
    if (
        type(materialized) is not OperatorPhaseCliqueMaterialization
        or materialized.proof_authority is not False
        or type(materialized.build) is not OperatorHZBuild
        or materialized.build is source_build
        or type(materialized.build.hz) is not SparseHZono
        or materialized.build.hz is source_build.hz
        or type(materialized.clique_ids) is not tuple
        or materialized.clique_ids != clique_ids
        or type(materialized.cut_row_tags) is not tuple
        or len(materialized.cut_row_tags) != len(clique_ids)
        or materialized.parent_semantic_digest != source_digest
    ):
        raise _PipelineFallback("materialization_object_invalid")
    receipt = _builtin_copy(materialized.receipt)
    receipt_caps = (
        receipt.get("caps") if type(receipt) is dict else None
    )
    if (
        type(receipt) is not dict
        or not _receipt_checksum_valid(receipt)
        or receipt.get("schema")
        != "act.operator_exact_relu_phase_clique_materialization.v2"
        or receipt.get("status")
        != "fresh_verified_clique_cuts_materialized"
        or receipt.get("candidate_only") is not True
        or receipt.get("proof_authority") is not False
        or receipt.get("hardened_exact_result_verifier_passed")
        is not True
        or receipt.get("one_use_snapshot_consumed") is not True
        or receipt.get("producer_nonempty_seal_verified") is not True
        or receipt.get("verdict_path") != _VERDICT_PATH
        or receipt.get("parent_semantic_digest") != source_digest
        or receipt.get("selection_digest") != selection_digest
        or receipt.get("focused_property_digest") != property_digest
        or receipt.get("subset_binding_digest") != subset_digest
        or tuple(receipt.get("clique_ids", ())) != clique_ids
        or tuple(receipt.get("cut_row_tags", ()))
        != materialized.cut_row_tags
        or receipt.get("cut_row_count") != len(clique_ids)
        or receipt.get("source_upper_rows") != source_n_ub
        or receipt.get("fresh_upper_rows")
        != source_n_ub + len(clique_ids)
        or receipt.get("fresh_upper_rows")
        != materialized.build.hz.n_ub
        or receipt.get("fresh_semantic_digest")
        != materialized.fresh_semantic_digest
        or receipt.get("source_frame_digest")
        != materialized.source_frame_digest
        or receipt.get("fresh_frame_digest")
        != materialized.fresh_frame_digest
        or receipt.get("constructive_nonempty_reissued") is not True
        or receipt.get("constructive_nonempty_scope")
        != "private_solver_handoff_only"
        or receipt.get("public_constructive_nonempty_token")
        != "absent"
        or receipt.get("solver_handoff_one_use") is not True
        or receipt.get("solver_handoff_owner_bound") is not True
        or receipt.get("solver_handoff_pid_bound") is not True
        or receipt.get("solver_handoff_private_core_readonly")
        is not True
        or receipt.get("solver_caches_stats_safe_tokens")
        != "not_copied"
        or receipt.get("public_core_source")
        != "consumed_verified_cut_zero_copy"
        or receipt.get("parent_prefix_core")
        != "strict_readonly_zero_copy_view"
        or receipt.get("parent_prefix_readonly") is not True
        or receipt.get("parent_prefix_aliases_public_cut") is not True
        or receipt.get("public_core_readonly") is not True
        or receipt.get("materializer_full_core_copy_count") != 1
        or receipt.get("private_solver_core")
        != "single_independent_snapshot"
        or receipt.get("public_private_core_no_alias") is not True
        or type(receipt_caps) is not dict
        or receipt_caps.get("max_top_literals")
        != _K4_TOP_LITERALS
        or receipt_caps.get("max_total_pairs")
        != _K4_TOTAL_PAIRS
        or receipt_caps.get("max_cliques") != 1
    ):
        raise _PipelineFallback(
            "materialization_receipt_invalid"
        )
    if (
        hz_constructively_nonempty(materialized.build.hz)
        or not validate_operator_phase_clique_materialization_solver_handoff(
            materialized
        )
    ):
        raise _PipelineFallback(
            "materialization_solver_handoff_invalid"
        )
    tags = getattr(
        materialized.build.hz,
        "_solver_constraint_row_tags",
        None,
    )
    if (
        type(tags) is not tuple
        or tuple(tags[-len(materialized.cut_row_tags) :])
        != materialized.cut_row_tags
    ):
        raise _PipelineFallback("materialization_cut_tags_invalid")
    _terminal_semantic_pair_seal(
        source_build.hz,
        materialized.build.hz,
        expected_source_digest=source_digest,
        expected_fresh_digest=receipt["fresh_semantic_digest"],
        deadline=deadline,
        stage="materialization_terminal",
    )
    return receipt


def _stage_seconds(
    timings: dict[str, float],
    name: str,
    started: float,
) -> None:
    timings[name] = float(max(0.0, time.monotonic() - started))


_FALLBACK_ALLOCATION_FIELDS = (
    "initial_budget_seconds",
    "candidate_budget_seconds",
    "minimum_materializer_reserve_seconds",
)
_FALLBACK_ELAPSED_SEMANTICS = (
    "elapsed_since_pipeline_start_at_failure_unclamped"
)
_CANDIDATE_PROGRESS_KEYS = frozenset(
    {
        "schema",
        "status",
        "candidate_only",
        "proof_authority",
        "verdict_authority",
        "model_load_started",
        "model_loaded",
        "oracle_backend",
        "oracle_presolve",
        "candidate_load_mode",
        "binary_change_coefficient_cap",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "pair_target_count",
        "pair_attempted_count",
        "pair_completed_count",
        "certified_conflict_count",
        "last_pair_index",
        "terminal_complete",
        "candidate_cut_hz_emitted",
        "partial_never_authorizes_edge",
        "materializer_reached",
    }
)
_CANDIDATE_PROGRESS_STATUSES = frozenset(
    {
        "initialized",
        "pair_plan_ready",
        "model_load_started",
        "model_loaded",
        "pair_probe",
        "complete",
    }
)


def _candidate_progress_is_exact(value: Any) -> bool:
    """Validate authority-free K4 progress without using it for proof."""

    if (
        type(value) is not dict
        or set(value) != _CANDIDATE_PROGRESS_KEYS
        or len(value) != len(_CANDIDATE_PROGRESS_KEYS)
        or value.get("schema") != _PROGRESS_SCHEMA
        or value.get("status") not in _CANDIDATE_PROGRESS_STATUSES
        or value.get("candidate_only") is not True
        or value.get("proof_authority") is not False
        or value.get("verdict_authority") is not False
        or type(value.get("model_load_started")) is not bool
        or type(value.get("model_loaded")) is not bool
        or value.get("model_loaded")
        and value.get("model_load_started") is not True
        or type(value.get("terminal_complete")) is not bool
        or type(value.get("candidate_cut_hz_emitted")) is not bool
        or value.get("candidate_cut_hz_emitted") is not False
        or value.get("partial_never_authorizes_edge") is not True
        or value.get("materializer_reached") is not False
    ):
        return False
    integer_names = (
        "pair_target_count",
        "pair_attempted_count",
        "pair_completed_count",
        "certified_conflict_count",
    )
    if any(
        type(value.get(name)) is not int or value[name] < 0
        for name in integer_names
    ):
        return False
    target = value["pair_target_count"]
    attempted = value["pair_attempted_count"]
    completed = value["pair_completed_count"]
    certified = value["certified_conflict_count"]
    last_pair = value["last_pair_index"]
    if (
        target > _K4_TOTAL_PAIRS
        or attempted > target
        or completed > attempted
        or certified > completed
        or (
            last_pair is not None
            and (
                type(last_pair) is not int
                or last_pair < 0
                or last_pair != attempted - 1
            )
        )
        or (attempted == 0) is not (last_pair is None)
    ):
        return False
    loaded_fields = (
        "oracle_backend",
        "oracle_presolve",
        "candidate_load_mode",
        "binary_change_coefficient_cap",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
    )
    if not value["model_loaded"]:
        if any(value.get(name) is not None for name in loaded_fields):
            return False
        if attempted != 0 or completed != 0 or certified != 0:
            return False
    else:
        if (
            value["oracle_backend"]
            != "highspy_persistent_simplex_presolve_lazy_dual_ray_v2"
            or value["oracle_presolve"] != "on"
            or value["candidate_load_mode"]
            not in {
                "split_continuous_rows_binary_change_coeff_v1",
                "single_merged_csr_binary_cap_fallback_v1",
            }
            or value["binary_change_coefficient_cap"]
            != _MAX_BINARY_CHANGE_COEFFICIENTS
            or any(
                type(value.get(name)) is not int or value[name] < 0
                for name in (
                    "candidate_rows",
                    "candidate_columns",
                    "candidate_nonzeros",
                )
            )
        ):
            return False
    terminal = value["terminal_complete"]
    if terminal is not (value["status"] == "complete"):
        return False
    if terminal and (
        completed != target
        or attempted != target
        or (target > 0 and not value["model_loaded"])
    ):
        return False
    return True


def _candidate_progress_payload(
    audit: Mapping[str, Any],
) -> Tuple[bool, Optional[dict[str, Any]]]:
    """Copy only a canonical diagnostic frame into a receipt."""

    value = audit.get("candidate_progress")
    try:
        copied = _builtin_copy(value)
    except (OperatorPhaseCliquePipelineError, TypeError, ValueError):
        return False, None
    if not _candidate_progress_is_exact(copied):
        return False, None
    return True, copied


def _fallback_budget_values_valid(
    values: Tuple[Any, Any, Any],
) -> bool:
    """Validate the diagnostic-only fallback allocation telemetry.

    These values never authorize a proof, handoff, or verdict.  The exact
    40/60 allocation is nevertheless retained and checked so a fallback
    receipt cannot misreport the budget that governed the failed attempt.
    """

    if (
        type(values) is not tuple
        or len(values) != len(_FALLBACK_ALLOCATION_FIELDS)
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            for value in values
        )
    ):
        return False
    initial, candidate, reserve = values
    return bool(
        math.isclose(
            candidate,
            _CANDIDATE_FRACTION * initial,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            reserve,
            _MATERIALIZER_FRACTION * initial,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            candidate + reserve,
            initial,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )


def _fallback_budget_payload(
    audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Copy only canonical allocation values and raw failure elapsed time."""

    allocation_values = tuple(
        audit.get(name) for name in _FALLBACK_ALLOCATION_FIELDS
    )
    if _fallback_budget_values_valid(allocation_values):
        payload: dict[str, Any] = {
            name: value
            for name, value in zip(
                _FALLBACK_ALLOCATION_FIELDS, allocation_values
            )
        }
    else:
        payload = {
            name: None for name in _FALLBACK_ALLOCATION_FIELDS
        }
    elapsed = audit.get("candidate_elapsed_seconds")
    payload["candidate_elapsed_seconds"] = (
        elapsed
        if (
            type(elapsed) is float
            and math.isfinite(elapsed)
            and elapsed >= 0.0
        )
        else None
    )
    payload["candidate_elapsed_semantics"] = (
        _FALLBACK_ELAPSED_SEMANTICS
    )
    return payload


def _fallback_budget_receipt_valid(
    receipt: Mapping[str, Any],
) -> bool:
    """Require a complete allocation plus honest, unclamped elapsed time."""

    allocation_values = tuple(
        receipt.get(name) for name in _FALLBACK_ALLOCATION_FIELDS
    )
    elapsed = receipt.get("candidate_elapsed_seconds")
    timings = receipt.get("timings")
    total = (
        timings.get("total_seconds")
        if type(timings) is dict
        else None
    )
    return bool(
        _fallback_budget_values_valid(allocation_values)
        and type(elapsed) is float
        and math.isfinite(elapsed)
        and elapsed >= 0.0
        and receipt.get("candidate_elapsed_semantics")
        == _FALLBACK_ELAPSED_SEMANTICS
        and type(total) is float
        and math.isfinite(total)
        and total >= 0.0
        and elapsed <= total + 1.0e-9
    )


def _fallback_result(
    build: Any,
    *,
    started: float,
    status: str,
    reason: str,
    stage: str,
    error_type: str,
    timings: Mapping[str, float],
    audit: Mapping[str, Any],
) -> OperatorPhaseCliquePipelineResult:
    timing_copy = {}
    if type(timings) in {dict, MappingProxyType}:
        for key, value in timings.items():
            if (
                type(key) is str
                and type(value) is float
                and math.isfinite(value)
                and value >= 0.0
            ):
                timing_copy[key[:128]] = value
    timing_copy["total_seconds"] = float(
        max(0.0, time.monotonic() - started)
    )
    safe_reason = (
        reason[:256]
        if type(reason) is str and reason
        else "downstream_error"
    )
    safe_stage = (
        stage[:128]
        if type(stage) is str and stage
        else "unknown"
    )
    safe_error_type = (
        error_type[:128]
        if type(error_type) is str and error_type
        else "Exception"
    )
    full_rival_count = audit.get("full_rival_count")
    if type(full_rival_count) is not int:
        full_rival_count = None
    focused_encoded_row = audit.get("focused_encoded_row")
    if type(focused_encoded_row) is not int:
        focused_encoded_row = None

    def safe_digest(name: str) -> Optional[str]:
        value = audit.get(name)
        return value if _valid_sha256(value) else None

    budget_payload = _fallback_budget_payload(audit)
    (
        candidate_progress_available,
        candidate_progress,
    ) = _candidate_progress_payload(audit)
    private_build: Optional[OperatorHZBuild] = None
    solver_capability: Optional[
        OperatorPhaseCliquePipelineSolverCapability
    ] = None
    handoff_status = "rejected"
    try:
        private_build = _fallback_private_solver_build(
            build, audit=audit
        )
        private_digest = safe_digest(
            "source_parent_semantic_digest"
        )
        if private_digest is None:
            raise _PipelineFallback(
                "fallback_private_digest_unavailable"
            )
        solver_capability = _make_pipeline_capability(
            private_digest
        )
        handoff_status = "issued"
    except Exception:
        private_build = None
        solver_capability = None

    body = {
        "schema": _PIPELINE_SCHEMA,
        "enabled": True,
        "status": status,
        "candidate_attempted": True,
        "candidate_only": True,
        "proof_authority": False,
        "identity_preserved": True,
        "materialized": False,
        "fallback_reason": safe_reason,
        "failed_stage": safe_stage,
        "error_type": safe_error_type,
        "full_rival_count": full_rival_count,
        "focused_encoded_row": focused_encoded_row,
        "hardness_vector_digest": safe_digest(
            "hardness_vector_digest"
        ),
        "focused_subset_digest": safe_digest(
            "focused_subset_digest"
        ),
        "selection_digest": safe_digest("selection_digest"),
        "subset_binding_digest": safe_digest(
            "subset_binding_digest"
        ),
        "source_parent_semantic_digest": safe_digest(
            "source_parent_semantic_digest"
        ),
        "source_frame_digest": safe_digest(
            "source_frame_digest"
        ),
        "solver_handoff_status": handoff_status,
        "solver_handoff_one_use": handoff_status == "issued",
        "solver_handoff_owner_bound": handoff_status == "issued",
        "solver_handoff_pid_bound": handoff_status == "issued",
        "solver_handoff_private_core_readonly": (
            handoff_status == "issued"
        ),
        "materialization_receipt_sha256": None,
        "verdict_path": _VERDICT_PATH,
        "candidate_budget_fraction": _CANDIDATE_FRACTION,
        "materializer_reserve_fraction": _MATERIALIZER_FRACTION,
        "candidate_progress_available": (
            candidate_progress_available
        ),
        "candidate_progress": candidate_progress,
        **budget_payload,
        "timings": timing_copy,
    }
    result = OperatorPhaseCliquePipelineResult(
        build=build,
        enabled=True,
        status=status,
        identity_preserved=True,
        materialized=False,
        receipt=_checksummed_receipt(body),
        materialization=None,
        solver_handoff_capability=solver_capability,
    )
    if private_build is not None and solver_capability is not None:
        try:
            _register_pipeline_solver_handoff(
                build,
                result,
                private_build,
                semantic_digest=solver_capability.semantic_digest,
            )
        except Exception:
            rejected_body = dict(body)
            rejected_body.update(
                {
                    "solver_handoff_status": "registration_rejected",
                    "solver_handoff_one_use": False,
                    "solver_handoff_owner_bound": False,
                    "solver_handoff_pid_bound": False,
                    "solver_handoff_private_core_readonly": False,
                }
            )
            return OperatorPhaseCliquePipelineResult(
                build=build,
                enabled=True,
                status=status,
                identity_preserved=True,
                materialized=False,
                receipt=_checksummed_receipt(rejected_body),
                materialization=None,
                solver_handoff_capability=None,
            )
    return result


def _run_enabled_pipeline(
    build: Any,
    *,
    vnnlib_path: Any,
    expected_vnnlib_sha256: Any,
    live_assert_params: Any,
    output_lower: Any,
    output_upper: Any,
    residual_selector_receipt: Any,
    residual_selector_property_sha256: Any,
    deadline: Any,
    caps: Any,
    started: float,
    timings: dict[str, float],
    audit: dict[str, Any],
) -> OperatorPhaseCliquePipelineResult:
    operation_deadline = _normalize_deadline(deadline)
    normalized_caps = _normalize_caps(caps)
    audit["_operation_deadline"] = float(operation_deadline)
    audit["_clique_caps"] = _pipeline_clique_caps(normalized_caps)
    if (
        type(build) is not OperatorHZBuild
        or type(build.hz) is not SparseHZono
    ):
        raise _PipelineFallback("source_build_wrong_type")
    if (
        not _valid_sha256(expected_vnnlib_sha256)
        or type(residual_selector_receipt) is not dict
        or not _valid_sha256(residual_selector_property_sha256)
    ):
        raise _PipelineFallback("raw_or_selector_binding_invalid")
    joint_focus = residual_selector_receipt.get(
        "joint_focus_rival_id"
    )
    if type(joint_focus) is not int:
        raise _PipelineFallback(
            "residual_joint_focus_row_invalid"
        )

    initial_remaining = operation_deadline - started
    if initial_remaining <= 0.0:
        raise _PipelineFallback(
            "deadline_expired_before_allocation", timeout=True
        )
    candidate_budget_seconds = (
        _CANDIDATE_FRACTION * initial_remaining
    )
    candidate_deadline = started + candidate_budget_seconds
    audit.update(
        {
            "initial_budget_seconds": float(initial_remaining),
            "candidate_budget_seconds": float(
                candidate_budget_seconds
            ),
            "minimum_materializer_reserve_seconds": float(
                _MATERIALIZER_FRACTION * initial_remaining
            ),
        }
    )
    _check_deadline(candidate_deadline, stage="candidate_allocation")
    source_digest = sparse_hz_semantic_digest(build.hz)
    source_n_ub = int(build.hz.n_ub)
    audit["source_parent_semantic_digest"] = source_digest
    source_tags, source_provenance = _source_frame_inputs(build)
    source_frame_digest = _source_frame_digest(
        build,
        tags=source_tags,
        provenance=source_provenance,
        parent_semantic_digest=source_digest,
        deadline=candidate_deadline,
    )
    audit["source_frame_digest"] = source_frame_digest

    audit["stage"] = "raw_top1_issue_consume"
    stage_started = time.monotonic()
    candidate = issue_raw_vnnlib_top1_candidate(
        vnnlib_path,
        expected_vnnlib_sha256=expected_vnnlib_sha256,
        live_assert_params=live_assert_params,
        deadline=candidate_deadline,
    )
    batch = consume_raw_vnnlib_top1_candidate(
        candidate,
        live_assert_params=live_assert_params,
        deadline=candidate_deadline,
    )
    if not validate_consumed_raw_vnnlib_rival_batch(batch):
        raise _PipelineFallback("raw_consumed_batch_invalid")
    _stage_seconds(timings, "raw_top1_seconds", stage_started)
    audit["full_rival_count"] = len(batch.rivals)
    if (
        len(batch.rivals) < 1
        or len(batch.rivals) > normalized_caps.max_full_rivals
    ):
        raise _PipelineFallback("raw_full_rival_count_out_of_cap")

    audit["stage"] = "complete_b1_interval_hardness"
    stage_started = time.monotonic()
    lower, upper = _snapshot_b1_bounds(
        output_lower,
        output_upper,
        output_width=int(build.hz.n_out),
    )
    exact_hardness = _exact_interval_upper_violations(
        batch.rivals,
        lower.reshape(-1),
        upper.reshape(-1),
        deadline=candidate_deadline,
    )
    interval_digest = _interval_frame_digest(
        build_digest=source_digest,
        batch_sha256=batch.batch_sha256,
        live_assert_sha256=batch.live_assert_sha256,
        property_digest=residual_selector_property_sha256,
        lower=lower,
        upper=upper,
    )
    hardness = issue_raw_rival_exact_hardness_receipt(
        batch,
        exact_hardness,
        live_interval_bounds_sha256=interval_digest,
        deadline=candidate_deadline,
        max_rivals=normalized_caps.max_full_rivals,
        max_focus=1,
        max_exact_bits=normalized_caps.max_raw_exact_bits,
        max_work_items=normalized_caps.max_raw_work_items,
    )
    _stage_seconds(timings, "hardness_seconds", stage_started)
    audit["hardness_vector_digest"] = hardness.vector_digest
    audit["interval_frame_sha256"] = interval_digest

    audit["stage"] = "residual_joint_focus"
    stage_started = time.monotonic()
    focused = select_raw_focused_rivals(
        batch,
        hardness,
        focus_count=1,
        explicit_encoded_focus_row=joint_focus,
        residual_selector_receipt=residual_selector_receipt,
        residual_selector_property_sha256=(
            residual_selector_property_sha256
        ),
        expected_exact_upper_violations=exact_hardness,
        expected_live_interval_bounds_sha256=interval_digest,
        deadline=candidate_deadline,
        max_rivals=normalized_caps.max_full_rivals,
        max_focus=1,
        max_exact_bits=normalized_caps.max_raw_exact_bits,
        max_work_items=normalized_caps.max_raw_work_items,
    )
    if not verify_raw_rival_exact_hardness_receipt(
        batch,
        hardness,
        expected_exact_upper_violations=exact_hardness,
        expected_live_interval_bounds_sha256=interval_digest,
        deadline=candidate_deadline,
        max_rivals=normalized_caps.max_full_rivals,
        max_focus=1,
        max_exact_bits=normalized_caps.max_raw_exact_bits,
        max_work_items=normalized_caps.max_raw_work_items,
    ):
        raise _PipelineFallback("hardness_receipt_replay_rejected")
    if not verify_raw_focused_rival_selection(
        batch,
        hardness,
        focused,
        expected_focus_count=1,
        expected_exact_upper_violations=exact_hardness,
        expected_live_interval_bounds_sha256=interval_digest,
        deadline=candidate_deadline,
        max_rivals=normalized_caps.max_full_rivals,
        max_focus=1,
        max_exact_bits=normalized_caps.max_raw_exact_bits,
        max_work_items=normalized_caps.max_raw_work_items,
    ):
        raise _PipelineFallback(
            "focused_selection_replay_rejected"
        )
    _stage_seconds(timings, "focus_and_replay_seconds", stage_started)
    audit["focused_encoded_row"] = joint_focus
    audit["focused_subset_digest"] = (
        focused.focused_subset_digest
    )

    audit["stage"] = "operator_exact_relu_literals"
    stage_started = time.monotonic()
    remaining_candidate = candidate_deadline - time.monotonic()
    if remaining_candidate <= 0.006:
        raise _PipelineFallback(
            "candidate_deadline_before_literal_audit",
            timeout=True,
        )
    selection_timeout = min(
        normalized_caps.max_selection_seconds,
        remaining_candidate / 6.0,
    )
    if selection_timeout <= 0.0:
        raise _PipelineFallback(
            "no_literal_selection_budget", timeout=True
        )
    selection = derive_operator_exact_relu_property_phase_literals(
        build,
        focused.rivals,
        max_rivals=1,
        max_binaries=normalized_caps.max_binaries,
        max_work_items=normalized_caps.max_selection_work_items,
        timeout_seconds=selection_timeout,
    )
    if not verify_operator_exact_relu_property_phase_selection(
        build,
        focused.rivals,
        selection,
        max_rivals=1,
        max_binaries=normalized_caps.max_binaries,
        max_work_items=normalized_caps.max_selection_work_items,
        timeout_seconds=selection_timeout,
    ):
        raise _PipelineFallback(
            "operator_literal_selection_rejected"
        )
    _check_deadline(
        candidate_deadline, stage="operator_literal_selection"
    )
    _stage_seconds(timings, "literal_selection_seconds", stage_started)
    audit["selection_digest"] = selection.selection_digest

    audit["stage"] = "exact_k4_candidate"
    stage_started = time.monotonic()
    clique_kwargs = _materializer_kwargs(
        normalized_caps,
        selection_timeout_seconds=selection_timeout,
    )
    candidate_progress: dict[str, Any] = {}
    audit["candidate_progress"] = candidate_progress
    clique_result = run_operator_exact_relu_phase_cliques_candidate(
        build,
        focused.rivals,
        selection,
        deadline=candidate_deadline,
        emit_cut_hz=False,
        diagnostic_progress=candidate_progress,
        **clique_kwargs,
    )
    clique_result = _require_k4_candidate(
        clique_result, deadline=candidate_deadline
    )
    binding_snapshot = _snapshot_candidate_bindings(
        batch=batch,
        hardness=hardness,
        focused=focused,
        selection=selection,
        clique_result=clique_result,
        focused_encoded_row=joint_focus,
        interval_frame_sha256=interval_digest,
        deadline=candidate_deadline,
    )
    _stage_seconds(timings, "k4_candidate_seconds", stage_started)
    audit["subset_binding_digest"] = (
        binding_snapshot.subset_binding_digest
    )
    audit["certified_edge_count"] = (
        binding_snapshot.certified_edge_count
    )
    audit["clique_count"] = binding_snapshot.clique_count
    _check_deadline(candidate_deadline, stage="candidate_terminal")
    audit["candidate_elapsed_seconds"] = float(
        max(0.0, time.monotonic() - started)
    )

    audit["stage"] = "fresh_materializer"
    stage_started = time.monotonic()
    materialized = materialize_verified_operator_phase_clique_cuts(
        build,
        focused.rivals,
        selection,
        clique_result,
        deadline=operation_deadline,
        caps=clique_result.caps,
    )
    fresh_build_snapshot = materialized.build
    fresh_hz_snapshot = fresh_build_snapshot.hz
    nested_receipt = _validate_materialization(
        build,
        materialized,
        source_digest=source_digest,
        source_n_ub=source_n_ub,
        selection_digest=binding_snapshot.selection_digest,
        property_digest=binding_snapshot.focused_property_digest,
        subset_digest=binding_snapshot.subset_binding_digest,
        clique_ids=binding_snapshot.clique_ids,
        deadline=operation_deadline,
    )
    try:
        private_solver_build = (
            consume_operator_phase_clique_materialization_solver_handoff(
                materialized,
                materialized.solver_handoff_capability,
                deadline=operation_deadline,
            )
        )
    except Exception as exc:
        raise _PipelineFallback(
            "materialization_private_solver_handoff_failed"
        ) from exc
    _stage_seconds(
        timings, "materializer_and_recheck_seconds", stage_started
    )
    _check_deadline(operation_deadline, stage="pipeline_success")

    if (
        materialized.build is not fresh_build_snapshot
        or materialized.build.hz is not fresh_hz_snapshot
    ):
        raise _PipelineFallback(
            "materialization_identity_changed_after_validation"
        )
    stage_started = time.monotonic()
    _terminal_semantic_pair_seal(
        build.hz,
        fresh_hz_snapshot,
        expected_source_digest=source_digest,
        expected_fresh_digest=nested_receipt[
            "fresh_semantic_digest"
        ],
        deadline=operation_deadline,
        stage="pipeline_return_terminal",
    )
    _stage_seconds(timings, "terminal_seal_seconds", stage_started)
    _check_deadline(operation_deadline, stage="pipeline_success_terminal")

    total_seconds = float(max(0.0, time.monotonic() - started))
    timing_copy = {
        str(key): float(value) for key, value in timings.items()
    }
    timing_copy["total_seconds"] = total_seconds
    (
        candidate_progress_available,
        candidate_progress_receipt,
    ) = _candidate_progress_payload(audit)
    body = {
        "schema": _PIPELINE_SCHEMA,
        "enabled": True,
        "status": "fresh_verified_k4_clique_materialized",
        "candidate_attempted": True,
        "candidate_only": True,
        "proof_authority": False,
        "identity_preserved": False,
        "materialized": True,
        "full_rival_count": binding_snapshot.full_rival_count,
        "focus_count": 1,
        "focused_encoded_row": (
            binding_snapshot.focused_encoded_row
        ),
        "ranked_literal_count": (
            binding_snapshot.ranked_literal_count
        ),
        "pair_count": binding_snapshot.pair_count,
        "certified_edge_count": (
            binding_snapshot.certified_edge_count
        ),
        "clique_count": binding_snapshot.clique_count,
        "cut_row_count": nested_receipt["cut_row_count"],
        "source_upper_rows": source_n_ub,
        "fresh_upper_rows": nested_receipt["fresh_upper_rows"],
        "source_parent_semantic_digest": source_digest,
        "full_batch_sha256": binding_snapshot.full_batch_sha256,
        "full_live_assert_sha256": (
            binding_snapshot.full_live_assert_sha256
        ),
        "full_property_digest": (
            binding_snapshot.full_property_digest
        ),
        "interval_frame_sha256": (
            binding_snapshot.interval_frame_sha256
        ),
        "hardness_vector_digest": (
            binding_snapshot.hardness_vector_digest
        ),
        "focused_subset_digest": (
            binding_snapshot.focused_subset_digest
        ),
        "selection_digest": binding_snapshot.selection_digest,
        "focused_property_digest": (
            binding_snapshot.focused_property_digest
        ),
        "subset_binding_digest": (
            binding_snapshot.subset_binding_digest
        ),
        "candidate_result_status": (
            binding_snapshot.candidate_result_status
        ),
        "candidate_telemetry_schema": (
            binding_snapshot.candidate_telemetry_schema
        ),
        "candidate_representation": (
            binding_snapshot.candidate_representation
        ),
        "candidate_cut_hz_emitted": False,
        "candidate_progress_available": (
            candidate_progress_available
        ),
        "candidate_progress": candidate_progress_receipt,
        "candidate_descriptor_sha256": (
            binding_snapshot.candidate_descriptor_sha256
        ),
        "candidate_route_summary": dict(
            binding_snapshot.candidate_route_summary
        ),
        "fresh_semantic_digest": nested_receipt[
            "fresh_semantic_digest"
        ],
        "materialization_receipt_sha256": nested_receipt[
            "receipt_sha256"
        ],
        "producer_nonempty_seal_verified": nested_receipt[
            "producer_nonempty_seal_verified"
        ],
        "materialization_receipt": nested_receipt,
        "solver_handoff_status": "issued",
        "solver_handoff_one_use": True,
        "solver_handoff_owner_bound": True,
        "solver_handoff_pid_bound": True,
        "solver_handoff_private_core_readonly": True,
        "verdict_path": _VERDICT_PATH,
        "candidate_budget_fraction": _CANDIDATE_FRACTION,
        "materializer_reserve_fraction": _MATERIALIZER_FRACTION,
        "initial_budget_seconds": audit[
            "initial_budget_seconds"
        ],
        "candidate_budget_seconds": audit[
            "candidate_budget_seconds"
        ],
        "minimum_materializer_reserve_seconds": audit[
            "minimum_materializer_reserve_seconds"
        ],
        "candidate_elapsed_seconds": audit[
            "candidate_elapsed_seconds"
        ],
        "timings": timing_copy,
    }
    receipt = _checksummed_receipt(body)
    solver_handoff_capability = _make_pipeline_capability(
        nested_receipt["fresh_semantic_digest"]
    )
    pipeline_result = OperatorPhaseCliquePipelineResult(
        build=fresh_build_snapshot,
        enabled=True,
        status=body["status"],
        identity_preserved=False,
        materialized=True,
        receipt=receipt,
        materialization=materialized,
        solver_handoff_capability=solver_handoff_capability,
    )
    if (
        materialized.build is not fresh_build_snapshot
        or materialized.build.hz is not fresh_hz_snapshot
        or pipeline_result.build is not fresh_build_snapshot
    ):
        raise _PipelineFallback(
            "materialization_identity_changed_after_validation"
        )
    _register_pipeline_solver_handoff(
        build,
        pipeline_result,
        private_solver_build,
        semantic_digest=nested_receipt["fresh_semantic_digest"],
    )
    return pipeline_result


def run_operator_phase_clique_pipeline(
    build: OperatorHZBuild,
    *,
    vnnlib_path: Any,
    expected_vnnlib_sha256: Any,
    live_assert_params: Any,
    output_lower: Any,
    output_upper: Any,
    residual_selector_receipt: Any,
    residual_selector_property_sha256: Any,
    deadline: Any,
    caps: Any = None,
) -> OperatorPhaseCliquePipelineResult:
    """Run the enabled pipeline, falling back to ``build`` on any failure."""

    return maybe_run_operator_phase_clique_pipeline(
        build,
        enabled=True,
        vnnlib_path=vnnlib_path,
        expected_vnnlib_sha256=expected_vnnlib_sha256,
        live_assert_params=live_assert_params,
        output_lower=output_lower,
        output_upper=output_upper,
        residual_selector_receipt=residual_selector_receipt,
        residual_selector_property_sha256=(
            residual_selector_property_sha256
        ),
        deadline=deadline,
        caps=caps,
    )


def maybe_run_operator_phase_clique_pipeline(
    build: Any,
    *,
    enabled: bool = False,
    vnnlib_path: Any = None,
    expected_vnnlib_sha256: Any = None,
    live_assert_params: Any = None,
    output_lower: Any = None,
    output_upper: Any = None,
    residual_selector_receipt: Any = None,
    residual_selector_property_sha256: Any = None,
    deadline: Any = None,
    caps: Any = None,
) -> OperatorPhaseCliquePipelineResult:
    """Default-off identity wrapper around the complete candidate chain."""

    if type(enabled) is not bool:
        raise OperatorPhaseCliquePipelineError(
            "enabled_must_be_builtin_bool"
        )
    if not enabled:
        return OperatorPhaseCliquePipelineResult(
            build=build,
            enabled=False,
            status="no_op_disabled",
            identity_preserved=True,
            materialized=False,
            receipt=_disabled_receipt(),
            materialization=None,
        )

    started = time.monotonic()
    timings: dict[str, float] = {}
    audit: dict[str, Any] = {"stage": "input_validation"}
    failure: Optional[Tuple[str, str, str, str, float]] = None
    try:
        return _run_enabled_pipeline(
            build,
            vnnlib_path=vnnlib_path,
            expected_vnnlib_sha256=expected_vnnlib_sha256,
            live_assert_params=live_assert_params,
            output_lower=output_lower,
            output_upper=output_upper,
            residual_selector_receipt=residual_selector_receipt,
            residual_selector_property_sha256=(
                residual_selector_property_sha256
            ),
            deadline=deadline,
            caps=caps,
            started=started,
            timings=timings,
            audit=audit,
        )
    except Exception as exc:
        timeout, reason, error_type = _safe_exception_details(exc)
        status = (
            "baseline_fallback_timeout"
            if timeout
            else "baseline_fallback_no_k4_clique"
            if reason == "no_complete_k4_clique"
            else "baseline_fallback_error"
        )
        stage_value = audit.get("stage", "unknown")
        if type(stage_value) is not str:
            stage_value = "unknown"
        candidate_elapsed_seconds = float(
            max(0.0, time.monotonic() - started)
        )
        # Retain builtin diagnostic values only.  In CPython the exception
        # target and its traceback are cleared when this suite is exited,
        # releasing failed candidate frames before the private fallback HZ is
        # allocated below.
        failure = (
            status,
            reason,
            stage_value,
            error_type,
            candidate_elapsed_seconds,
        )

    if failure is None:  # Defensive: the try either returned or populated it.
        raise OperatorPhaseCliquePipelineError(
            "pipeline_failure_classification_missing"
        )
    (
        status,
        reason,
        stage_value,
        error_type,
        candidate_elapsed_seconds,
    ) = failure
    audit["candidate_elapsed_seconds"] = candidate_elapsed_seconds
    return _fallback_result(
        build,
        started=started,
        status=status,
        reason=reason,
        stage=stage_value,
        error_type=error_type,
        timings=timings,
        audit=audit,
    )


def verify_operator_phase_clique_pipeline_result(
    source_build: Any,
    result: Any,
    *,
    deadline: Optional[float] = None,
) -> bool:
    """Live identity/checksum replay for verifier-side integration."""

    try:
        operation_deadline = (
            time.monotonic() + 60.0
            if deadline is None
            else float(deadline)
        )
        _check_deadline(
            operation_deadline, stage="public_pipeline_result_verify"
        )
        if (
            type(result) is not OperatorPhaseCliquePipelineResult
            or result.proof_authority is not False
            or type(result.receipt) is not MappingProxyType
            or not _receipt_checksum_valid(result.receipt)
        ):
            return False
        receipt = _builtin_copy(result.receipt)
        if (
            receipt.get("schema") != _PIPELINE_SCHEMA
            or receipt.get("enabled") is not result.enabled
            or receipt.get("status") != result.status
            or receipt.get("candidate_only") is not True
            or receipt.get("proof_authority") is not False
            or receipt.get("identity_preserved")
            is not result.identity_preserved
            or receipt.get("materialized") is not result.materialized
            or receipt.get("verdict_path") != _VERDICT_PATH
            or receipt.get("candidate_budget_fraction")
            != _CANDIDATE_FRACTION
            or receipt.get("materializer_reserve_fraction")
            != _MATERIALIZER_FRACTION
        ):
            return False
        if not result.enabled:
            return (
                result.build is source_build
                and result.identity_preserved
                and not result.materialized
                and result.materialization is None
                and result.solver_handoff_capability is None
                and result.status == "no_op_disabled"
                and receipt.get("candidate_attempted") is False
                and receipt.get("materialization_receipt_sha256")
                is None
                and _canonical_bytes(receipt)
                == _canonical_bytes(
                    _builtin_copy(_disabled_receipt())
                )
            )
        progress_available = receipt.get(
            "candidate_progress_available"
        )
        candidate_progress = receipt.get("candidate_progress")
        if (
            type(progress_available) is not bool
            or progress_available
            is not (candidate_progress is not None)
            or (
                progress_available
                and not _candidate_progress_is_exact(
                    candidate_progress
                )
            )
        ):
            return False
        if not result.materialized:
            source_digest = receipt.get(
                "source_parent_semantic_digest"
            )
            return (
                result.build is source_build
                and result.identity_preserved
                and result.materialization is None
                and result.status
                in {
                    "baseline_fallback_error",
                    "baseline_fallback_timeout",
                    "baseline_fallback_no_k4_clique",
                }
                and receipt.get("candidate_attempted") is True
                and receipt.get("materialization_receipt_sha256")
                is None
                and receipt.get("materialization_receipt") is None
                and type(receipt.get("fallback_reason")) is str
                and type(receipt.get("failed_stage")) is str
                and type(receipt.get("error_type")) is str
                and _fallback_budget_receipt_valid(receipt)
                and _valid_sha256(source_digest)
                and _valid_sha256(
                    receipt.get("source_frame_digest")
                )
                and receipt.get("solver_handoff_status") == "issued"
                and receipt.get("solver_handoff_one_use") is True
                and receipt.get("solver_handoff_owner_bound") is True
                and receipt.get("solver_handoff_pid_bound") is True
                and receipt.get(
                    "solver_handoff_private_core_readonly"
                )
                is True
                and _validate_pipeline_solver_handoff_registration(
                    source_build, result
                )
                and sparse_hz_semantic_digest(source_build.hz)
                == source_digest
            )
        materialized = result.materialization
        if (
            type(source_build) is not OperatorHZBuild
            or type(source_build.hz) is not SparseHZono
            or type(materialized)
            is not OperatorPhaseCliqueMaterialization
            or materialized.proof_authority is not False
            or result.build is not materialized.build
            or result.build is source_build
            or type(result.build) is not OperatorHZBuild
            or type(result.build.hz) is not SparseHZono
            or result.build.hz is source_build.hz
            or result.identity_preserved
            or result.status
            != "fresh_verified_k4_clique_materialized"
            or receipt.get("candidate_attempted") is not True
            or type(receipt.get("full_rival_count")) is not int
            or receipt.get("full_rival_count") < 1
            or type(receipt.get("focused_encoded_row")) is not int
            or receipt.get("focused_encoded_row") < 0
            or receipt.get("focused_encoded_row")
            >= receipt.get("full_rival_count")
            or receipt.get("focus_count") != 1
            or receipt.get("ranked_literal_count")
            != _K4_TOP_LITERALS
            or receipt.get("pair_count") != _K4_TOTAL_PAIRS
            or receipt.get("certified_edge_count")
            != _K4_TOTAL_PAIRS
            or receipt.get("clique_count") != 1
            or receipt.get("cut_row_count") != 1
            or receipt.get("candidate_result_status")
            != _COMPACT_SUCCESS_STATUS
            or receipt.get("candidate_telemetry_schema")
            != _COMPACT_TELEMETRY_SCHEMA
            or receipt.get("candidate_representation")
            != _COMPACT_CANDIDATE_REPRESENTATION
            or receipt.get("candidate_cut_hz_emitted") is not False
            or progress_available is not True
            or candidate_progress.get("status") != "complete"
            or candidate_progress.get("terminal_complete") is not True
            or candidate_progress.get("pair_target_count")
            != _K4_TOTAL_PAIRS
            or candidate_progress.get("pair_attempted_count")
            != _K4_TOTAL_PAIRS
            or candidate_progress.get("pair_completed_count")
            != _K4_TOTAL_PAIRS
            or candidate_progress.get("certified_conflict_count")
            != _K4_TOTAL_PAIRS
            or receipt.get("producer_nonempty_seal_verified") is not True
            or not _valid_sha256(
                receipt.get("candidate_descriptor_sha256")
            )
            or not _compact_route_receipt_is_exact(
                receipt.get("candidate_route_summary"),
                source_hz=source_build.hz,
                deadline=operation_deadline,
            )
            or materialized.clique_ids is None
            or type(materialized.clique_ids) is not tuple
            or len(materialized.clique_ids) != 1
            or type(materialized.cut_row_tags) is not tuple
            or len(materialized.cut_row_tags) != 1
        ):
            return False
        nested = receipt.get("materialization_receipt")
        materialized_receipt = _builtin_copy(materialized.receipt)
        nested_caps = (
            nested.get("caps") if type(nested) is dict else None
        )
        source_rows = int(source_build.hz.n_ub)
        fresh_rows = int(result.build.hz.n_ub)
        digest_fields = (
            "full_batch_sha256",
            "full_live_assert_sha256",
            "full_property_digest",
            "interval_frame_sha256",
            "hardness_vector_digest",
            "focused_subset_digest",
            "selection_digest",
            "focused_property_digest",
            "subset_binding_digest",
            "source_parent_semantic_digest",
            "fresh_semantic_digest",
            "materialization_receipt_sha256",
        )
        if any(
            not _valid_sha256(receipt.get(name))
            for name in digest_fields
        ):
            return False
        budget_values = tuple(
            receipt.get(name)
            for name in (
                "initial_budget_seconds",
                "candidate_budget_seconds",
                "minimum_materializer_reserve_seconds",
                "candidate_elapsed_seconds",
            )
        )
        if (
            any(type(value) is not float for value in budget_values)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in budget_values
            )
            or not math.isclose(
                budget_values[1],
                _CANDIDATE_FRACTION * budget_values[0],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                budget_values[2],
                _MATERIALIZER_FRACTION * budget_values[0],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or budget_values[3] > budget_values[1] + 1.0e-9
        ):
            return False
        if not (
            type(nested) is dict
            and _receipt_checksum_valid(nested)
            and _canonical_bytes(nested)
            == _canonical_bytes(materialized_receipt)
            and nested.get("proof_authority") is False
            and nested.get("candidate_only") is True
            and nested.get("hardened_exact_result_verifier_passed")
            is True
            and nested.get("one_use_snapshot_consumed") is True
            and nested.get("producer_nonempty_seal_verified") is True
            and nested.get("verdict_path") == _VERDICT_PATH
            and nested.get("constructive_nonempty_reissued") is True
            and nested.get("constructive_nonempty_scope")
            == "private_solver_handoff_only"
            and nested.get("public_constructive_nonempty_token")
            == "absent"
            and nested.get("solver_handoff_one_use") is True
            and nested.get("solver_handoff_owner_bound") is True
            and nested.get("solver_handoff_pid_bound") is True
            and nested.get("solver_handoff_private_core_readonly")
            is True
            and nested.get("solver_caches_stats_safe_tokens")
            == "not_copied"
            and nested.get("public_core_source")
            == "consumed_verified_cut_zero_copy"
            and nested.get("parent_prefix_core")
            == "strict_readonly_zero_copy_view"
            and nested.get("parent_prefix_readonly") is True
            and nested.get("parent_prefix_aliases_public_cut") is True
            and nested.get("public_core_readonly") is True
            and nested.get("materializer_full_core_copy_count") == 1
            and nested.get("private_solver_core")
            == "single_independent_snapshot"
            and nested.get("public_private_core_no_alias") is True
            and type(nested_caps) is dict
            and nested_caps.get("max_top_literals")
            == _K4_TOP_LITERALS
            and nested_caps.get("max_total_pairs")
            == _K4_TOTAL_PAIRS
            and nested_caps.get("max_cliques") == 1
            and nested.get("receipt_sha256")
            == receipt.get("materialization_receipt_sha256")
            and nested.get("producer_nonempty_seal_verified")
            == receipt.get("producer_nonempty_seal_verified")
            and nested.get("parent_semantic_digest")
            == receipt.get("source_parent_semantic_digest")
            and nested.get("parent_semantic_digest")
            == materialized.parent_semantic_digest
            and nested.get("fresh_semantic_digest")
            == receipt.get("fresh_semantic_digest")
            and nested.get("fresh_semantic_digest")
            == materialized.fresh_semantic_digest
            and nested.get("selection_digest")
            == receipt.get("selection_digest")
            and nested.get("focused_property_digest")
            == receipt.get("focused_property_digest")
            and nested.get("subset_binding_digest")
            == receipt.get("subset_binding_digest")
            and tuple(nested.get("clique_ids", ()))
            == materialized.clique_ids
            and tuple(nested.get("cut_row_tags", ()))
            == materialized.cut_row_tags
            and nested.get("cut_row_count") == 1
            and nested.get("source_upper_rows") == source_rows
            and nested.get("fresh_upper_rows") == fresh_rows
            and receipt.get("source_upper_rows") == source_rows
            and receipt.get("fresh_upper_rows") == fresh_rows
            and fresh_rows == source_rows + 1
            and not hz_constructively_nonempty(result.build.hz)
            and receipt.get("solver_handoff_status") == "issued"
            and receipt.get("solver_handoff_one_use") is True
            and receipt.get("solver_handoff_owner_bound") is True
            and receipt.get("solver_handoff_pid_bound") is True
            and receipt.get("solver_handoff_private_core_readonly")
            is True
            and _validate_pipeline_solver_handoff_registration(
                source_build, result
            )
        ):
            return False
        _terminal_semantic_pair_seal(
            source_build.hz,
            result.build.hz,
            expected_source_digest=receipt[
                "source_parent_semantic_digest"
            ],
            expected_fresh_digest=receipt[
                "fresh_semantic_digest"
            ],
            deadline=operation_deadline,
            stage="public_pipeline_verify_terminal",
        )
        # Diagnostic defense-in-depth for a callback that mutates immediately
        # after the pair helper's own final read.  Solver authority is the
        # isolated one-use handoff above, not this inherently non-atomic bool.
        return (
            sparse_hz_semantic_digest(source_build.hz)
            == receipt["source_parent_semantic_digest"]
            and sparse_hz_semantic_digest(result.build.hz)
            == receipt["fresh_semantic_digest"]
        )
    except Exception:
        return False


def consume_operator_phase_clique_pipeline_solver_handoff(
    source_build: Any,
    result: Any,
    *,
    deadline: float,
) -> OperatorHZBuild:
    """Validate then atomically transfer the private HZ exactly once.

    The public result remains candidate-only.  Its HZ may be mutated after
    this function's diagnostic replay without affecting the returned build:
    all semantic and provenance buffers were copied and frozen before the
    result became observable, and no registry or result retains that private
    build after this atomic pop.
    """

    operation_deadline = _normalize_deadline(deadline)
    if (
        type(source_build) is not OperatorHZBuild
        or type(result) is not OperatorPhaseCliquePipelineResult
        or result.enabled is not True
        or result.proof_authority is not False
        or type(result.receipt) is not MappingProxyType
        or not _receipt_checksum_valid(result.receipt)
    ):
        raise OperatorPhaseCliquePipelineError(
            "pipeline solver handoff result malformed"
        )
    capability = result.solver_handoff_capability
    if (
        type(capability)
        is not OperatorPhaseCliquePipelineSolverCapability
        or result.enabled is not True
        or capability.proof_authority is not False
        or capability.process_id != os.getpid()
        or type(capability.token) is not str
        or len(capability.token) != 64
        or any(
            character not in "0123456789abcdef"
            for character in capability.token
        )
        or not _valid_sha256(capability.semantic_digest)
        or type(capability.expires_monotonic) is not float
        or not math.isfinite(capability.expires_monotonic)
    ):
        raise OperatorPhaseCliquePipelineError(
            "pipeline solver handoff capability malformed"
        )
    with _PIPELINE_HANDOFF_REGISTRY_LOCK:
        now = time.monotonic()
        _sweep_pipeline_handoffs_locked(now)
        record = _PIPELINE_HANDOFF_REGISTRY.get(capability.token)
        if (
            record is None
            or record.process_id != os.getpid()
            or record.capability_ref() is not capability
            or record.result_ref() is not result
            or record.source_build_ref() is not source_build
            or record.public_build_ref() is not result.build
            or record.public_hz_ref() is not result.build.hz
            or record.expires_monotonic
            != capability.expires_monotonic
            or record.expires_monotonic <= now
            or operation_deadline <= now
            or record.semantic_digest != capability.semantic_digest
            or record.receipt_object is not result.receipt
            or record.receipt_sha256
            != result.receipt.get("receipt_sha256")
            or record.status != result.status
            or record.materialized is not result.materialized
        ):
            raise OperatorPhaseCliquePipelineError(
                "pipeline solver handoff capability invalid"
            )
        try:
            public_identity = tuple(
                id(value)
                for value in _solver_owned_objects(result.build)
            )
            private_identity = tuple(
                id(value)
                for value in _solver_owned_objects(
                    record.private_build
                )
            )
            private_arrays = _solver_owned_arrays(
                record.private_build
            )
        except Exception:
            public_identity = ()
            private_identity = ()
            private_arrays = ()
        if (
            public_identity != record.public_owned_identity
            or private_identity != record.private_owned_identity
            or not private_arrays
            or any(value.flags.writeable for value in private_arrays)
        ):
            _PIPELINE_HANDOFF_REGISTRY.pop(capability.token, None)
            raise OperatorPhaseCliquePipelineError(
                "pipeline solver handoff owner identity changed"
            )
        _PIPELINE_HANDOFF_REGISTRY.pop(capability.token)
        private_build = record.private_build
    _check_deadline(operation_deadline, stage="solver_handoff_consumed")
    return private_build


def validate_consumed_operator_phase_clique_solver_build(
    result: Any,
    private_build: Any,
) -> bool:
    """Check the verifier-local consumed object immediately before solving."""

    try:
        if (
            type(result) is not OperatorPhaseCliquePipelineResult
            or type(private_build) is not OperatorHZBuild
            or type(private_build.hz) is not SparseHZono
            or private_build is result.build
            or private_build.hz is result.build.hz
            or not hz_constructively_nonempty(private_build.hz)
            or result.solver_handoff_capability is None
        ):
            return False
        expected_digest = (
            result.solver_handoff_capability.semantic_digest
        )
        if not _valid_sha256(expected_digest):
            return False
        private_arrays = _solver_owned_arrays(private_build)
        public_arrays = _solver_owned_arrays(result.build)
        return bool(
            private_arrays
            and all(not value.flags.writeable for value in private_arrays)
            and not any(
                np.shares_memory(public_value, private_value)
                for public_value in public_arrays
                for private_value in private_arrays
            )
            and sparse_hz_semantic_digest(private_build.hz)
            == expected_digest
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
    ):
        return False


__all__ = [
    "OperatorPhaseCliquePipelineCaps",
    "OperatorPhaseCliquePipelineError",
    "OperatorPhaseCliquePipelineResult",
    "OperatorPhaseCliquePipelineSolverCapability",
    "consume_operator_phase_clique_pipeline_solver_handoff",
    "maybe_run_operator_phase_clique_pipeline",
    "run_operator_phase_clique_pipeline",
    "validate_consumed_operator_phase_clique_solver_build",
    "verify_operator_phase_clique_pipeline_result",
]
