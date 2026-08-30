#!/usr/bin/env python3
# ===- exact_sparse_conv_native_schedule_candidate.py --------------------===#
"""Disconnected compact-schedule native CONV experiment.

The candidate stores only compact binary64 CONV weights, per-output-channel
bias, and seventeen int64 geometry scalars.  Its warm C kernel enumerates the
NCHW convolution directly in the same canonical row order as Operator-HZ and
emits center, outward mass/error data, and canonical row-local generators in
one traversal.  It never constructs or caches an expanded sparse CONV
operator or an expanded operator schedule.

Only monotone, injective, row-local generator sources are eligible.  CFFI and
a host compiler are available in the current research environment but are not
declared ACT dependencies, so this module remains a disconnected feasibility
probe.  Its receipts are permanently non-authoritative and it is not imported
by ``operator_hz.py``.

No triangle relaxation, branch-and-bound, backward/dual operation, solver, or
SpGEMM occurs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import operator
import os
import platform
import tempfile
import threading
from typing import Any, Tuple
import weakref

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import exact_sparse_conv_affine_core as _affine


_SCHEDULE_SCHEMA = "act.exact_sparse_conv.compact_native_schedule.v4"
_SOURCE_SCHEMA = "act.exact_sparse_conv.native_monotone_source.v4"
_RESULT_SCHEMA = "act.exact_sparse_conv.compact_native_result.v4"
_RECEIPT_SCHEMA = "act.exact_sparse_conv.compact_native_receipt.v4"
_PROJECTION_SCHEMA = "act.exact_sparse_conv.compact_cache_projection.v4"
_INT32_MAX = int(np.iinfo(np.int32).max)
_I64_MIN = int(np.iinfo(np.int64).min)
_I64_MAX = int(np.iinfo(np.int64).max)
_NPY_MAX_INTP = int(np.iinfo(np.intp).max)
_GEOMETRY_I64_COUNT = 17
_C89_PERSISTENT_SCHEDULE_ONLY_BYTES = 18_516_504
_KNOWN_LARGE_LAYER_OPERATOR_NNZ = 7_929_856
_KNOWN_LARGE_LAYER_OUTPUT_ROWS = 8_192
# Ownership-transfer lower bound: one G data/index/indptr capacity and one
# c/mass/propagated/error output.  Source, IDs, Python/SciPy objects, allocator
# fragmentation, and a measured C89 peak remain outside this static number.
_KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_BYTES = 95_453_188
_KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_MIB = (
    _KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_BYTES / (1024.0 * 1024.0)
)


class ExactConvNativeScheduleError(ValueError):
    """Fail-closed rejection by the disconnected native experiment."""


class NativeKernelUnavailable(ExactConvNativeScheduleError):
    """The environment cannot build/load the strict research kernel."""


class MonotoneRowLocalNotApplicable(Exception):
    """The sole ordinary ineligibility signal for a valid affine source."""


def _private_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    normalized = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    _checked_allocation(
        int(normalized.size),
        int(np.dtype(dtype).itemsize),
        name="private snapshot",
    )
    return np.frombuffer(normalized.tobytes(order="C"), dtype=dtype).reshape(
        normalized.shape
    )


def _adopt_output_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Transfer an internal kernel allocation without making a second copy."""

    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or not value.flags.c_contiguous
    ):
        raise ExactConvNativeScheduleError("native output cannot be transferred")
    base = value.base
    if type(base) is np.ndarray:
        base.setflags(write=False)
    value.setflags(write=False)
    return value


def _snapshot_f64(value: Any, *, name: str) -> np.ndarray:
    source = value
    try:
        detach = getattr(source, "detach", None)
        if callable(detach):
            source = detach()
            cpu = getattr(source, "cpu", None)
            if callable(cpu):
                source = cpu()
            numpy_method = getattr(source, "numpy", None)
            if callable(numpy_method):
                source = numpy_method()
        raw = np.asarray(source)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ExactConvNativeScheduleError(
            f"{name} cannot be converted to a host array"
        ) from exc
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ExactConvNativeScheduleError(f"{name} must be real-valued")
    raw_entries = int(raw.size)
    _checked_allocation(raw_entries, np.dtype(np.float64).itemsize, name=name)
    try:
        snapshot = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExactConvNativeScheduleError(
            f"{name} cannot be represented as binary64"
        ) from exc
    if not np.all(np.isfinite(snapshot)):
        raise ExactConvNativeScheduleError(f"{name} contains NaN or infinity")
    return _private_array(snapshot, dtype=np.dtype(np.float64))


def _builtin_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ExactConvNativeScheduleError(f"{name} must be an integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise ExactConvNativeScheduleError(f"{name} must be an integer") from exc
    if result < _I64_MIN or result > _I64_MAX:
        raise ExactConvNativeScheduleError(f"{name} exceeds signed int64")
    return result


def _pair(value: Any, *, name: str, positive: bool) -> Tuple[int, int]:
    try:
        scalar = _builtin_int(value, name=name)
    except ExactConvNativeScheduleError:
        try:
            items = tuple(value)
        except TypeError as exc:
            raise ExactConvNativeScheduleError(
                f"{name} must be an integer or length-two sequence"
            ) from exc
        if len(items) != 2:
            raise ExactConvNativeScheduleError(
                f"{name} must be an integer or length-two sequence"
            )
        result = (
            _builtin_int(items[0], name=f"{name}[0]"),
            _builtin_int(items[1], name=f"{name}[1]"),
        )
    else:
        result = (scalar, scalar)
    if (positive and min(result) <= 0) or (not positive and min(result) < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ExactConvNativeScheduleError(f"{name} must be {qualifier}")
    return result


def _shape4(value: Any, *, name: str) -> Tuple[int, int, int, int]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ExactConvNativeScheduleError(f"{name} must be length four") from exc
    if len(items) != 4:
        raise ExactConvNativeScheduleError(f"{name} must be length four")
    shape = tuple(
        _builtin_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(items)
    )
    if min(shape) <= 0:
        raise ExactConvNativeScheduleError(f"{name} must be positive")
    return shape  # type: ignore[return-value]


def _checked_i64(value: int, *, name: str) -> int:
    if type(value) is not int or value < _I64_MIN or value > _I64_MAX:
        raise ExactConvNativeScheduleError(f"{name} exceeds signed int64")
    return value


def _checked_add_i64(left: int, right: int, *, name: str) -> int:
    return _checked_i64(left + right, name=name)


def _checked_mul_i64(left: int, right: int, *, name: str) -> int:
    return _checked_i64(left * right, name=name)


def _checked_product(
    values: Tuple[int, ...],
    *,
    name: str,
    limit: int = _INT32_MAX,
) -> int:
    result = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise ExactConvNativeScheduleError(f"{name} has an invalid factor")
        if value and result > int(limit) // value:
            raise ExactConvNativeScheduleError(f"{name} exceeds checked limit")
        result *= value
    return result


def _checked_allocation(entries: int, itemsize: int, *, name: str) -> int:
    if type(entries) is not int or entries < 0:
        raise ExactConvNativeScheduleError(f"{name} has invalid entry count")
    if type(itemsize) is not int or itemsize <= 0:
        raise ExactConvNativeScheduleError(f"{name} has invalid item size")
    if entries > _NPY_MAX_INTP // itemsize:
        raise ExactConvNativeScheduleError(f"{name} exceeds NPY_MAX_INTP bytes")
    return entries * itemsize


def _checked_allocation_sum(terms: Tuple[int, ...], *, name: str) -> int:
    total = 0
    for term in terms:
        if type(term) is not int or term < 0 or total > _NPY_MAX_INTP - term:
            raise ExactConvNativeScheduleError(f"{name} exceeds NPY_MAX_INTP")
        total += term
    return total


def _extent(
    input_extent: int,
    kernel_extent: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    twice_padding = _checked_mul_i64(2, padding, name="2*padding")
    kernel_tail = _checked_mul_i64(
        dilation, kernel_extent - 1, name="dilation*(kernel-1)"
    )
    numerator = _checked_add_i64(
        input_extent, twice_padding, name="input+2*padding"
    )
    numerator = _checked_add_i64(
        numerator, -kernel_tail, name="padded-kernel-tail"
    )
    numerator = _checked_add_i64(numerator, -1, name="extent numerator")
    output = _checked_add_i64(
        numerator // stride, 1, name="extent quotient plus one"
    )
    if output <= 0:
        raise ExactConvNativeScheduleError("CONV has no positive output extent")
    output_tail = _checked_mul_i64(
        output - 1, stride, name="(output-1)*stride"
    )
    shifted_tail = _checked_add_i64(
        output_tail, -padding, name="output-tail-padding"
    )
    _checked_add_i64(
        shifted_tail, kernel_tail, name="maximum input coordinate"
    )
    return output


def _digest_array(digest: Any, value: np.ndarray) -> None:
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))


@dataclass(frozen=True, eq=False)
class ExactConvNativeSchedule:
    """Model-owned compact weights and geometry; never expanded W arrays."""

    geometry: np.ndarray
    weight: np.ndarray
    bias_channels: np.ndarray
    digest: str
    _owner: Any = field(repr=False, compare=False)
    schema: str = _SCHEDULE_SCHEMA

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        g = self.geometry
        return (int(g[0]), int(g[1]), int(g[2]), int(g[3]))

    @property
    def output_shape(self) -> Tuple[int, int, int, int]:
        g = self.geometry
        return (int(g[0]), int(g[4]), int(g[5]), int(g[6]))

    @property
    def input_size(self) -> int:
        return _checked_product(self.input_shape, name="schedule input size")

    @property
    def output_size(self) -> int:
        return _checked_product(self.output_shape, name="schedule output size")

    @property
    def compact_buffer_nbytes(self) -> int:
        return _checked_allocation_sum(
            (
                int(self.geometry.nbytes),
                int(self.weight.nbytes),
                int(self.bias_channels.nbytes),
            ),
            name="compact schedule bytes",
        )


@dataclass(frozen=True, eq=False)
class ExactMonotoneRowLocalSource:
    center: np.ndarray
    source_mass: np.ndarray
    error: np.ndarray
    row_to_generator_column: np.ndarray
    row_scale: np.ndarray
    stable_column_ids: np.ndarray
    generator_columns: int
    digest: str
    schema: str = _SOURCE_SCHEMA

    @property
    def size(self) -> int:
        return int(self.center.size)


@dataclass(frozen=True, eq=False)
class _CapturedSchedule:
    """Lock-issued view backed only by registry-owned immutable arrays."""

    geometry: np.ndarray
    weight: np.ndarray
    bias_channels: np.ndarray
    digest: str
    _owner: Any = field(repr=False, compare=False)
    schema: str = _SCHEDULE_SCHEMA

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        g = self.geometry
        return (int(g[0]), int(g[1]), int(g[2]), int(g[3]))

    @property
    def output_shape(self) -> Tuple[int, int, int, int]:
        g = self.geometry
        return (int(g[0]), int(g[4]), int(g[5]), int(g[6]))

    @property
    def input_size(self) -> int:
        return _checked_product(self.input_shape, name="captured input size")

    @property
    def output_size(self) -> int:
        return _checked_product(self.output_shape, name="captured output size")

    @property
    def compact_buffer_nbytes(self) -> int:
        return _checked_allocation_sum(
            (
                int(self.geometry.nbytes),
                int(self.weight.nbytes),
                int(self.bias_channels.nbytes),
            ),
            name="captured compact schedule bytes",
        )


@dataclass(frozen=True, eq=False)
class _CapturedSource:
    """Lock-issued view backed only by registry-owned immutable arrays."""

    center: np.ndarray
    source_mass: np.ndarray
    error: np.ndarray
    row_to_generator_column: np.ndarray
    row_scale: np.ndarray
    stable_column_ids: np.ndarray
    generator_columns: int
    digest: str
    schema: str = _SOURCE_SCHEMA

    @property
    def size(self) -> int:
        return int(self.center.size)


@dataclass(frozen=True, eq=False)
class ExactConvNativeResult:
    center: np.ndarray
    generators: sp.csr_matrix
    error: np.ndarray
    transformed_mass: np.ndarray
    propagated_error: np.ndarray
    stable_column_ids: np.ndarray
    affine_depth: int
    schedule_digest: str
    source_digest: str
    schema: str = _RESULT_SCHEMA


@dataclass(frozen=True)
class ExactConvNativeReceipt:
    schedule_digest: str
    source_digest: str
    compact_weight_entries: int
    compact_schedule_bytes: int
    output_generator_nnz: int
    output_capacity_nnz: int
    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    construction_mode: str = "cffi_strict_f64_compact_geometry_stream_v4"
    geometry_weight_traversals: int = 1
    expanded_conv_operator_materialized: bool = False
    expanded_conv_operator_cached: bool = False
    expanded_operator_schedule_cached: bool = False
    uses_runtime_compiler: bool = True
    cffi_is_declared_act_dependency: bool = False
    runtime_compile_excluded_from_schedule_source_model_cold_gate: bool = False
    first_response_native_load_included: bool = False
    first_response_gate_passes: bool = False
    reuse_scope: str = "same_process_same_model_only"
    factory_registry_authenticated: bool = True
    factory_registry_reusable: bool = True
    source_snapshot_reused_without_second_copy: bool = True
    output_ownership_transferred_without_freeze_copy: bool = True
    public_output_rebind_protected: bool = False
    c89_persistent_schedule_only_bytes: int = _C89_PERSISTENT_SCHEDULE_ONLY_BYTES
    c89_apply_peak_measured: bool = False
    known_large_layer_operator_nnz: int = _KNOWN_LARGE_LAYER_OPERATOR_NNZ
    known_large_layer_output_rows: int = _KNOWN_LARGE_LAYER_OUTPUT_ROWS
    known_large_layer_freeze_lower_bound_bytes: int = (
        _KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_BYTES
    )
    known_large_layer_freeze_lower_bound_mib: float = (
        _KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_MIB
    )
    network_memory_gate_passes: bool = False
    linear_primitive_authoritative: bool = False
    property_proof_authority: bool = False
    verdict_authority: bool = False
    uses_spgemm: bool = False
    uses_triangle_relaxation: bool = False
    uses_branch_and_bound: bool = False
    uses_backward_or_dual: bool = False
    uses_solver: bool = False
    production_promotion_claim: bool = False
    schema: str = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        false_flags = (
            self.expanded_conv_operator_materialized,
            self.expanded_conv_operator_cached,
            self.expanded_operator_schedule_cached,
            self.cffi_is_declared_act_dependency,
            self.runtime_compile_excluded_from_schedule_source_model_cold_gate,
            self.first_response_native_load_included,
            self.first_response_gate_passes,
            self.c89_apply_peak_measured,
            self.network_memory_gate_passes,
            self.public_output_rebind_protected,
            self.linear_primitive_authoritative,
            self.property_proof_authority,
            self.verdict_authority,
            self.uses_spgemm,
            self.uses_triangle_relaxation,
            self.uses_branch_and_bound,
            self.uses_backward_or_dual,
            self.uses_solver,
            self.production_promotion_claim,
        )
        true_flags = (
            self.uses_runtime_compiler,
            self.factory_registry_authenticated,
            self.factory_registry_reusable,
            self.source_snapshot_reused_without_second_copy,
            self.output_ownership_transferred_without_freeze_copy,
        )
        integer_fields = (
            self.compact_weight_entries,
            self.compact_schedule_bytes,
            self.output_generator_nnz,
            self.output_capacity_nnz,
            self.geometry_weight_traversals,
            self.c89_persistent_schedule_only_bytes,
            self.known_large_layer_operator_nnz,
            self.known_large_layer_output_rows,
            self.known_large_layer_freeze_lower_bound_bytes,
        )
        hashes = (self.schedule_digest, self.source_digest)
        shapes = (self.input_shape, self.output_shape)
        if (
            type(self) is not ExactConvNativeReceipt
            or set(vars(self)) != _RECEIPT_KEYSET
            or any(type(value) is not bool for value in false_flags + true_flags)
            or any(value is not False for value in false_flags)
            or any(value is not True for value in true_flags)
            or any(type(value) is not int for value in integer_fields)
            or any(type(value) is not str for value in hashes)
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or any(
                type(shape) is not tuple
                or len(shape) != 4
                or any(type(item) is not int or item <= 0 for item in shape)
                for shape in shapes
            )
            or type(self.construction_mode) is not str
            or self.construction_mode
            != "cffi_strict_f64_compact_geometry_stream_v4"
            or type(self.reuse_scope) is not str
            or type(self.schema) is not str
            or type(self.known_large_layer_freeze_lower_bound_mib) is not float
            or self.geometry_weight_traversals != 1
            or self.reuse_scope != "same_process_same_model_only"
            or min(integer_fields) < 0
            or self.output_generator_nnz > self.output_capacity_nnz
            or self.c89_persistent_schedule_only_bytes
            != _C89_PERSISTENT_SCHEDULE_ONLY_BYTES
            or self.known_large_layer_operator_nnz
            != _KNOWN_LARGE_LAYER_OPERATOR_NNZ
            or self.known_large_layer_output_rows
            != _KNOWN_LARGE_LAYER_OUTPUT_ROWS
            or self.known_large_layer_freeze_lower_bound_bytes
            != _KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_BYTES
            or self.known_large_layer_freeze_lower_bound_mib
            != _KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_MIB
            or self.schema != _RECEIPT_SCHEMA
        ):
            raise ExactConvNativeScheduleError("invalid compact native receipt")


_RECEIPT_KEYSET = frozenset(ExactConvNativeReceipt.__dataclass_fields__)


def _schedule_digest(schedule: Any) -> str:
    digest = hashlib.sha256()
    digest.update(_SCHEDULE_SCHEMA.encode("ascii"))
    _digest_array(digest, schedule.geometry)
    _digest_array(digest, schedule.weight)
    _digest_array(digest, schedule.bias_channels)
    return digest.hexdigest()


def _source_digest(source: Any) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_SCHEMA.encode("ascii"))
    digest.update(np.asarray((source.generator_columns,), dtype="<i8").tobytes())
    for value in (
        source.center,
        source.source_mass,
        source.error,
        source.row_to_generator_column,
        source.row_scale,
        source.stable_column_ids,
    ):
        _digest_array(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class _ScheduleRegistryRecord:
    reference: Any
    owner: Any
    digest: str
    geometry: np.ndarray
    weight: np.ndarray
    bias_channels: np.ndarray
    schema: str


@dataclass(frozen=True)
class _SourceRegistryRecord:
    reference: Any
    digest: str
    center: np.ndarray
    source_mass: np.ndarray
    error: np.ndarray
    row_to_generator_column: np.ndarray
    row_scale: np.ndarray
    stable_column_ids: np.ndarray
    generator_columns: int
    schema: str


_REGISTRY_LOCK = threading.RLock()
_SCHEDULE_REGISTRY: dict[int, _ScheduleRegistryRecord] = {}
_SOURCE_REGISTRY: dict[int, _SourceRegistryRecord] = {}


def _drop_schedule(reference: Any, key: int) -> None:
    with _REGISTRY_LOCK:
        record = _SCHEDULE_REGISTRY.get(key)
        if record is not None and record.reference is reference:
            _SCHEDULE_REGISTRY.pop(key, None)


def _drop_source(reference: Any, key: int) -> None:
    with _REGISTRY_LOCK:
        record = _SOURCE_REGISTRY.get(key)
        if record is not None and record.reference is reference:
            _SOURCE_REGISTRY.pop(key, None)


def _register_schedule(schedule: ExactConvNativeSchedule, owner: Any) -> None:
    key = id(schedule)
    reference = weakref.ref(
        schedule, lambda current, identity=key: _drop_schedule(current, identity)
    )
    record = _ScheduleRegistryRecord(
        reference=reference,
        owner=owner,
        digest=schedule.digest,
        geometry=schedule.geometry,
        weight=schedule.weight,
        bias_channels=schedule.bias_channels,
        schema=schedule.schema,
    )
    with _REGISTRY_LOCK:
        existing = _SCHEDULE_REGISTRY.get(key)
        if existing is not None and existing.reference() is not None:
            raise ExactConvNativeScheduleError("schedule registry identity collision")
        _SCHEDULE_REGISTRY[key] = record


def _register_source(source: ExactMonotoneRowLocalSource) -> None:
    key = id(source)
    reference = weakref.ref(
        source, lambda current, identity=key: _drop_source(current, identity)
    )
    record = _SourceRegistryRecord(
        reference=reference,
        digest=source.digest,
        center=source.center,
        source_mass=source.source_mass,
        error=source.error,
        row_to_generator_column=source.row_to_generator_column,
        row_scale=source.row_scale,
        stable_column_ids=source.stable_column_ids,
        generator_columns=source.generator_columns,
        schema=source.schema,
    )
    with _REGISTRY_LOCK:
        existing = _SOURCE_REGISTRY.get(key)
        if existing is not None and existing.reference() is not None:
            raise ExactConvNativeScheduleError("source registry identity collision")
        _SOURCE_REGISTRY[key] = record


def _admit_registered_factory_objects(
    owner: Any,
    schedule: Any,
    source: Any,
) -> Tuple[_CapturedSchedule, _CapturedSource]:
    if type(schedule) is not ExactConvNativeSchedule:
        raise ExactConvNativeScheduleError("schedule was not factory-produced")
    if type(source) is not ExactMonotoneRowLocalSource:
        raise ExactConvNativeScheduleError("source was not factory-produced")
    with _REGISTRY_LOCK:
        schedule_record = _SCHEDULE_REGISTRY.get(id(schedule))
        source_record = _SOURCE_REGISTRY.get(id(source))
        if (
            schedule_record is None
            or schedule_record.reference() is not schedule
            or schedule_record.owner is not owner
            or schedule._owner is not owner
            or schedule_record.digest != schedule.digest
            or schedule_record.geometry is not schedule.geometry
            or schedule_record.weight is not schedule.weight
            or schedule_record.bias_channels is not schedule.bias_channels
            or schedule_record.schema != schedule.schema
        ):
            raise ExactConvNativeScheduleError(
                "schedule lacks live factory registry identity"
            )
        if (
            source_record is None
            or source_record.reference() is not source
            or source_record.digest != source.digest
            or source_record.center is not source.center
            or source_record.source_mass is not source.source_mass
            or source_record.error is not source.error
            or source_record.row_to_generator_column
            is not source.row_to_generator_column
            or source_record.row_scale is not source.row_scale
            or source_record.stable_column_ids is not source.stable_column_ids
            or source_record.generator_columns != source.generator_columns
            or source_record.schema != source.schema
        ):
            raise ExactConvNativeScheduleError(
                "source lacks live factory registry identity"
            )
        captured_schedule = _CapturedSchedule(
            geometry=schedule_record.geometry,
            weight=schedule_record.weight,
            bias_channels=schedule_record.bias_channels,
            digest=schedule_record.digest,
            _owner=schedule_record.owner,
            schema=schedule_record.schema,
        )
        captured_source = _CapturedSource(
            center=source_record.center,
            source_mass=source_record.source_mass,
            error=source_record.error,
            row_to_generator_column=source_record.row_to_generator_column,
            row_scale=source_record.row_scale,
            stable_column_ids=source_record.stable_column_ids,
            generator_columns=source_record.generator_columns,
            digest=source_record.digest,
            schema=source_record.schema,
        )
    return captured_schedule, captured_source


def _registry_sizes_for_tests() -> Tuple[int, int]:
    with _REGISTRY_LOCK:
        return len(_SCHEDULE_REGISTRY), len(_SOURCE_REGISTRY)


def _validate_schedule_snapshot(
    schedule: Any,
    *,
    owner: Any,
    require_digest: bool,
    captured: bool = False,
) -> Tuple[int, ...]:
    expected_type = _CapturedSchedule if captured else ExactConvNativeSchedule
    if (
        type(schedule) is not expected_type
        or schedule.schema != _SCHEDULE_SCHEMA
        or schedule._owner is not owner
    ):
        raise ExactConvNativeScheduleError("invalid compact schedule snapshot")
    geometry = _require_private(
        schedule.geometry,
        dtype=np.dtype(np.int64),
        shape=(_GEOMETRY_I64_COUNT,),
        name="schedule.geometry",
    )
    values = tuple(int(value) for value in geometry)
    (
        batch, in_ch, in_h, in_w, out_ch, out_h, out_w,
        in_per_group, kh, kw, stride_h, stride_w,
        pad_h, pad_w, dilation_h, dilation_w, groups,
    ) = values
    if (
        min(values[:12]) <= 0
        or min(values[12:14]) < 0
        or min(values[14:]) <= 0
        or _checked_mul_i64(in_per_group, groups, name="in_per_group*groups")
        != in_ch
        or out_ch % groups != 0
        or out_h != _extent(in_h, kh, stride_h, pad_h, dilation_h)
        or out_w != _extent(in_w, kw, stride_w, pad_w, dilation_w)
    ):
        raise ExactConvNativeScheduleError("compact schedule geometry changed")
    weight_entries = _checked_product(
        (out_ch, in_per_group, kh, kw),
        name="compact weight entries",
        limit=min(_I64_MAX, _NPY_MAX_INTP),
    )
    _require_private(
        schedule.weight,
        dtype=np.dtype(np.float64),
        shape=(out_ch, in_per_group, kh, kw),
        name="schedule.weight",
    )
    _require_private(
        schedule.bias_channels,
        dtype=np.dtype(np.float64),
        shape=(out_ch,),
        name="schedule.bias",
    )
    if (
        schedule.weight.size != weight_entries
        or not np.all(np.isfinite(schedule.weight))
        or not np.all(np.isfinite(schedule.bias_channels))
    ):
        raise ExactConvNativeScheduleError("compact schedule values changed")
    input_size = _checked_product(
        (batch, in_ch, in_h, in_w), name="flattened input"
    )
    output_size = _checked_product(
        (batch, out_ch, out_h, out_w), name="flattened output"
    )
    capacity = _checked_product(
        (output_size, in_per_group, kh, kw),
        name="maximum output generator capacity",
    )
    indptr_entries = _checked_add_i64(
        output_size, 1, name="output indptr entries"
    )
    vector_bytes = _checked_allocation(
        output_size, np.dtype(np.float64).itemsize, name="output vector"
    )
    raw_indptr_bytes = _checked_allocation(
        indptr_entries, np.dtype(np.int32).itemsize, name="raw indptr"
    )
    raw_index_bytes = _checked_allocation(
        capacity, np.dtype(np.int32).itemsize, name="raw indices"
    )
    data_bytes = _checked_allocation(
        capacity, np.dtype(np.float64).itemsize, name="raw data"
    )
    frozen_indptr_bytes = _checked_allocation(
        indptr_entries, np.dtype(np.int32).itemsize, name="frozen indptr"
    )
    frozen_index_bytes = _checked_allocation(
        capacity, np.dtype(np.int32).itemsize, name="frozen indices"
    )
    _checked_allocation_sum(
        (
            8 * vector_bytes,
            raw_indptr_bytes,
            raw_index_bytes,
            2 * data_bytes,
            frozen_indptr_bytes,
            frozen_index_bytes,
        ),
        name="maximum simultaneous raw/freeze numeric bytes",
    )
    if (
        type(schedule.digest) is not str
        or len(schedule.digest) != 64
        or (require_digest and _schedule_digest(schedule) != schedule.digest)
    ):
        raise ExactConvNativeScheduleError("compact schedule digest changed")
    return values


def _validate_source_snapshot(
    source: Any,
    *,
    require_digest: bool,
    captured: bool = False,
) -> Any:
    expected_type = _CapturedSource if captured else ExactMonotoneRowLocalSource
    if (
        type(source) is not expected_type
        or source.schema != _SOURCE_SCHEMA
        or type(source.generator_columns) is not int
        or source.generator_columns < 0
        or source.generator_columns > _INT32_MAX
    ):
        raise ExactConvNativeScheduleError("invalid row-local source snapshot")
    size = int(getattr(source.center, "size", -1))
    _checked_product((size,), name="source rows")
    for name, value, dtype in (
        ("center", source.center, np.dtype(np.float64)),
        ("source_mass", source.source_mass, np.dtype(np.float64)),
        ("error", source.error, np.dtype(np.float64)),
        ("row_scale", source.row_scale, np.dtype(np.float64)),
        ("row_to_column", source.row_to_generator_column, np.dtype(np.int64)),
    ):
        _require_private(value, dtype=dtype, shape=(size,), name=name)
    _require_private(
        source.stable_column_ids,
        dtype=np.dtype(np.int64),
        shape=(source.generator_columns,),
        name="stable ids",
    )
    live = source.row_to_generator_column >= 0
    live_columns = source.row_to_generator_column[live]
    if (
        not np.all(np.isfinite(source.center))
        or not np.all(np.isfinite(source.source_mass))
        or not np.all(np.isfinite(source.error))
        or not np.all(np.isfinite(source.row_scale))
        or np.any(source.source_mass < 0.0)
        or np.any(source.error < 0.0)
        or np.any(source.row_to_generator_column < -1)
        or np.any(live_columns >= source.generator_columns)
        or (live_columns.size > 1 and np.any(live_columns[1:] <= live_columns[:-1]))
        or np.any(source.row_scale[~live] != 0.0)
        or np.any(source.row_scale[live] == 0.0)
        or np.unique(source.stable_column_ids).size != source.generator_columns
        or type(source.digest) is not str
        or len(source.digest) != 64
        or (require_digest and _source_digest(source) != source.digest)
    ):
        raise ExactConvNativeScheduleError("row-local source snapshot changed")
    return source


def prepare_exact_conv_native_schedule(
    owner: Any,
    layer: Any,
) -> ExactConvNativeSchedule:
    """Cold-snapshot compact weights/geometry without constructing W."""

    if owner is None:
        raise ExactConvNativeScheduleError("schedule owner must be a live object")
    try:
        params = layer.params
        weight_value = params["weight"]
        input_shape_value = params["input_shape"]
        output_shape_value = params["output_shape"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ExactConvNativeScheduleError(
            "layer must provide weight/input_shape/output_shape"
        ) from exc
    if str(params.get("data_format", "NCHW")).upper() != "NCHW":
        raise ExactConvNativeScheduleError("only NCHW CONV is supported")
    if str(params.get("padding_mode", "zeros")).lower() not in {
        "zeros", "zero", "constant"
    }:
        raise ExactConvNativeScheduleError("only zero padding is supported")
    if str(params.get("auto_pad", "NOTSET")).upper() not in {"", "NOTSET"}:
        raise ExactConvNativeScheduleError("auto_pad must be NOTSET")

    weight = _snapshot_f64(weight_value, name="weight")
    if weight.ndim != 4 or min(weight.shape) <= 0:
        raise ExactConvNativeScheduleError(
            "weight must have positive (out_ch,in_ch/group,kh,kw) shape"
        )
    out_ch, in_per_group, kh, kw = (int(v) for v in weight.shape)
    input_shape = _shape4(input_shape_value, name="input_shape")
    output_shape = _shape4(output_shape_value, name="output_shape")
    stride = _pair(params.get("stride", 1), name="stride", positive=True)
    padding = _pair(params.get("padding", 0), name="padding", positive=False)
    dilation = _pair(params.get("dilation", 1), name="dilation", positive=True)
    groups = _builtin_int(params.get("groups", 1), name="groups")
    if groups <= 0:
        raise ExactConvNativeScheduleError("groups must be positive")
    batch, in_ch, in_h, in_w = input_shape
    out_batch, declared_out_ch, out_h, out_w = output_shape
    if (
        out_batch != batch
        or declared_out_ch != out_ch
        or _checked_mul_i64(
            in_per_group, groups, name="in_per_group*groups"
        ) != in_ch
        or out_ch % groups != 0
        or out_h != _extent(in_h, kh, stride[0], padding[0], dilation[0])
        or out_w != _extent(in_w, kw, stride[1], padding[1], dilation[1])
    ):
        raise ExactConvNativeScheduleError(
            "weight/output shape disagrees with CONV geometry"
        )
    _checked_product(input_shape, name="flattened input")
    _checked_product(output_shape, name="flattened output")
    _checked_product(
        (batch, out_ch, out_h, out_w, in_per_group, kh, kw),
        name="maximum output generator capacity",
    )

    bias_value = params.get("bias")
    _checked_allocation(
        out_ch, np.dtype(np.float64).itemsize, name="compact bias channels"
    )
    if bias_value is None:
        bias = _private_array(np.zeros(out_ch), dtype=np.dtype(np.float64))
    else:
        bias = _snapshot_f64(bias_value, name="bias").reshape(-1)
        if bias.size != out_ch:
            raise ExactConvNativeScheduleError("bias size disagrees with output")
    geometry = _private_array(
        np.asarray(
            (
                batch, in_ch, in_h, in_w,
                out_ch, out_h, out_w,
                in_per_group, kh, kw,
                stride[0], stride[1],
                padding[0], padding[1],
                dilation[0], dilation[1],
                groups,
            ),
            dtype=np.int64,
        ),
        dtype=np.dtype(np.int64),
    )
    provisional = ExactConvNativeSchedule(
        geometry=geometry,
        weight=weight,
        bias_channels=bias,
        digest="",
        _owner=owner,
    )
    schedule = ExactConvNativeSchedule(
        geometry=geometry,
        weight=weight,
        bias_channels=bias,
        digest=_schedule_digest(provisional),
        _owner=owner,
    )
    _validate_schedule_snapshot(schedule, owner=owner, require_digest=True)
    _register_schedule(schedule, owner)
    return schedule


def prepare_exact_monotone_row_local_source(
    center: Any,
    generators: Any,
    error: Any,
    *,
    stable_column_ids: Any,
) -> ExactMonotoneRowLocalSource:
    """Cold-validate the only source topology admitted by this probe."""

    try:
        source = _affine.prepare_exact_row_local_affine_source(
            center,
            generators,
            error,
            stable_column_ids=stable_column_ids,
        )
    except _affine.RowLocalNotApplicable as exc:
        raise MonotoneRowLocalNotApplicable(str(exc)) from exc
    except Exception as exc:
        raise ExactConvNativeScheduleError(
            "row-local source preparation failed closed"
        ) from exc
    if not source.positional_mapping_monotone:
        raise MonotoneRowLocalNotApplicable(
            "row-local source generator mapping is not globally monotone"
        )
    generator_columns = int(source.generators.shape[1])
    if generator_columns > _INT32_MAX:
        raise ExactConvNativeScheduleError("generator columns exceed int32")
    # The affine source factory already returned private bytes-backed arrays.
    # Reuse those exact owners; the old wrapper copied every derived vector a
    # second time before the first convolution response.
    center_snapshot = source.center
    mass_snapshot = source.source_mass
    error_snapshot = source.error
    mapping_snapshot = source.row_to_generator_column
    scale_snapshot = source.row_scale
    ids_snapshot = source.stable_column_ids
    provisional = ExactMonotoneRowLocalSource(
        center=center_snapshot,
        source_mass=mass_snapshot,
        error=error_snapshot,
        row_to_generator_column=mapping_snapshot,
        row_scale=scale_snapshot,
        stable_column_ids=ids_snapshot,
        generator_columns=generator_columns,
        digest="",
    )
    admitted = ExactMonotoneRowLocalSource(
        center=provisional.center,
        source_mass=provisional.source_mass,
        error=provisional.error,
        row_to_generator_column=provisional.row_to_generator_column,
        row_scale=provisional.row_scale,
        stable_column_ids=provisional.stable_column_ids,
        generator_columns=provisional.generator_columns,
        digest=_source_digest(provisional),
    )
    _validate_source_snapshot(admitted, require_digest=True)
    _register_source(admitted)
    return admitted


def project_compact_schedule_cache_bytes(
    *,
    weight_entries: int,
    bias_entries: int,
    conv_count: int,
    source_cap_bytes: int = 160 * 1024 * 1024,
) -> dict[str, Any]:
    """Project persistent schedule buffers only, never total network memory."""

    raw = (weight_entries, bias_entries, conv_count, source_cap_bytes)
    if any(type(value) is not int for value in raw):
        raise ExactConvNativeScheduleError("projection counts must be integers")
    weights, biases, count, cap = raw
    if min(weights, biases, count, cap) < 0:
        raise ExactConvNativeScheduleError(
            "projection counts must be nonnegative"
        )
    _checked_i64(weights, name="projection weights")
    _checked_i64(biases, name="projection biases")
    _checked_i64(count, name="projection CONV count")
    _checked_i64(cap, name="projection source cap")
    geometry_entries = _checked_product(
        (count, _GEOMETRY_I64_COUNT),
        name="projection geometry entries",
        limit=_NPY_MAX_INTP,
    )
    weight_bytes = _checked_allocation(
        weights, np.dtype(np.float64).itemsize, name="projected weights"
    )
    bias_bytes = _checked_allocation(
        biases, np.dtype(np.float64).itemsize, name="projected biases"
    )
    geometry_bytes = _checked_allocation(
        geometry_entries,
        np.dtype(np.int64).itemsize,
        name="projected geometry",
    )
    total = _checked_allocation_sum(
        (weight_bytes, bias_bytes, geometry_bytes),
        name="persistent schedule projection",
    )
    return {
        "schema": _PROJECTION_SCHEMA,
        "scope": "persistent_compact_schedule_numeric_buffers_only",
        "weight_entries": weights,
        "bias_entries": biases,
        "conv_count": count,
        "weight_f64_bytes": weight_bytes,
        "bias_f64_bytes": bias_bytes,
        "geometry_i64_count_per_conv": _GEOMETRY_I64_COUNT,
        "geometry_i64_bytes": geometry_bytes,
        "expanded_operator_bytes": 0,
        "expanded_operator_schedule_bytes": 0,
        "total_persistent_numeric_buffer_bytes": total,
        "total_persistent_numeric_buffer_mib": total / (1024.0 * 1024.0),
        "persistent_schedule_cap_bytes": cap,
        "persistent_schedule_within_cap": bool(total <= cap),
        "python_object_headers_included": False,
        "network_total_memory_established": False,
        "c89_apply_peak_measured": False,
        "known_large_layer_operator_nnz": _KNOWN_LARGE_LAYER_OPERATOR_NNZ,
        "known_large_layer_output_rows": _KNOWN_LARGE_LAYER_OUTPUT_ROWS,
        "known_large_layer_freeze_lower_bound_bytes": (
            _KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_BYTES
        ),
        "known_large_layer_freeze_lower_bound_mib": (
            _KNOWN_LARGE_LAYER_FREEZE_LOWER_BOUND_MIB
        ),
        "network_memory_gate_passes": False,
        "production_promotion_claim": False,
    }


_CDEF = """
typedef long int int64_t;
typedef int int32_t;
typedef unsigned long int size_t;
int act_exact_conv_compact_fused_v3(
    int64_t batch, int64_t in_ch, int64_t in_h, int64_t in_w,
    int64_t out_ch, int64_t out_h, int64_t out_w,
    int64_t in_per_group, int64_t kh, int64_t kw,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t groups,
    size_t input_rows, size_t output_rows, size_t weight_entries,
    size_t output_capacity, int64_t generator_columns,
    const double *weight, const double *bias,
    const double *center, const double *source_mass, const double *source_error,
    const int64_t *row_to_column, const double *row_scale,
    double *out_center, double *out_mass, double *out_propagated_error,
    double *out_error, int32_t *out_indptr, int32_t *out_indices,
    double *out_data, int64_t *out_emitted);
"""


_C_SOURCE = r"""
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static inline double act_gamma(double count) {
    double product = count * DBL_EPSILON;
    return product / (1.0 - product);
}

static inline double act_inflate(double rounded, double count, int active) {
    if (!active) return 0.0;
    double out = rounded / (1.0 - act_gamma(count));
    out = out + DBL_MIN * (count > 1.0 ? count : 1.0);
    return nextafter(out, INFINITY);
}

static inline int act_size_mul(size_t left, size_t right, size_t *out) {
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *out = left * right;
    return 1;
}

static inline int act_size_add(size_t left, size_t right, size_t *out) {
    if (right > SIZE_MAX - left) return 0;
    *out = left + right;
    return 1;
}

static inline int act_i64_mul_nonnegative(
    int64_t left, int64_t right, int64_t *out
) {
    if (left < 0 || right < 0) return 0;
    if (left != 0 && right > INT64_MAX / left) return 0;
    *out = left * right;
    return 1;
}

static inline int act_i64_add_nonnegative(
    int64_t left, int64_t right, int64_t *out
) {
    if (left < 0 || right < 0 || right > INT64_MAX - left) return 0;
    *out = left + right;
    return 1;
}

int act_exact_conv_compact_fused_v3(
    int64_t batch, int64_t in_ch, int64_t in_h, int64_t in_w,
    int64_t out_ch, int64_t out_h, int64_t out_w,
    int64_t in_per_group, int64_t kh, int64_t kw,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t groups,
    size_t input_rows, size_t output_rows, size_t weight_entries,
    size_t output_capacity, int64_t generator_columns,
    const double *weight, const double *bias,
    const double *center, const double *source_mass, const double *source_error,
    const int64_t *row_to_column, const double *row_scale,
    double *out_center, double *out_mass, double *out_propagated_error,
    double *out_error, int32_t *out_indptr, int32_t *out_indices,
    double *out_data, int64_t *out_emitted) {
    if (
        batch <= 0 || in_ch <= 0 || in_h <= 0 || in_w <= 0
        || out_ch <= 0 || out_h <= 0 || out_w <= 0
        || in_per_group <= 0 || kh <= 0 || kw <= 0
        || stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0
        || dilation_h <= 0 || dilation_w <= 0 || groups <= 0
        || input_rows == 0 || output_rows == 0 || weight_entries == 0
        || output_capacity == 0 || generator_columns < 0
    ) return -1;
    int64_t expected_in_ch = 0;
    if (
        !act_i64_mul_nonnegative(in_per_group, groups, &expected_in_ch)
        || expected_in_ch != in_ch || out_ch % groups != 0
    ) return -8;
    size_t expected_input_rows = 0;
    size_t expected_output_rows = 0;
    size_t expected_weight_entries = 0;
    size_t expected_output_capacity = 0;
    size_t maximum_fanin = 0;
    size_t input_area = 0;
    if (
        !act_size_mul((size_t)in_h, (size_t)in_w, &input_area)
        || !act_size_mul((size_t)batch, (size_t)in_ch, &expected_input_rows)
        || !act_size_mul(expected_input_rows, input_area, &expected_input_rows)
        || !act_size_mul((size_t)batch, (size_t)out_ch, &expected_output_rows)
        || !act_size_mul(expected_output_rows, (size_t)out_h,
                         &expected_output_rows)
        || !act_size_mul(expected_output_rows, (size_t)out_w,
                         &expected_output_rows)
        || !act_size_mul((size_t)out_ch, (size_t)in_per_group,
                         &expected_weight_entries)
        || !act_size_mul(expected_weight_entries, (size_t)kh,
                         &expected_weight_entries)
        || !act_size_mul(expected_weight_entries, (size_t)kw,
                         &expected_weight_entries)
        || !act_size_mul(expected_output_rows, (size_t)in_per_group,
                         &expected_output_capacity)
        || !act_size_mul(expected_output_capacity, (size_t)kh,
                         &expected_output_capacity)
        || !act_size_mul(expected_output_capacity, (size_t)kw,
                         &expected_output_capacity)
        || expected_input_rows != input_rows
        || expected_output_rows != output_rows
        || expected_weight_entries != weight_entries
        || expected_output_capacity != output_capacity
        || !act_size_mul((size_t)in_per_group, (size_t)kh, &maximum_fanin)
        || !act_size_mul(maximum_fanin, (size_t)kw, &maximum_fanin)
        || maximum_fanin > (size_t)INT64_MAX
        || output_rows == SIZE_MAX
    ) return -9;
    size_t emitted = 0;
    size_t output_row = 0;
    int64_t out_per_group = out_ch / groups;
    out_indptr[0] = 0;
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t co = 0; co < out_ch; ++co) {
            int64_t group = co / out_per_group;
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow, ++output_row) {
                    if (output_row >= output_rows) return -2;
                    double center_sum = 0.0;
                    double mass_sum = 0.0;
                    double propagated_sum = 0.0;
                    int64_t mass_active = 0;
                    int64_t propagated_active = 0;
                    int64_t fanin = 0;
                    for (int64_t ci = 0; ci < in_per_group; ++ci) {
                        int64_t global_channel = 0;
                        if (
                            !act_i64_mul_nonnegative(
                                group, in_per_group, &global_channel
                            )
                            || !act_i64_add_nonnegative(
                                global_channel, ci, &global_channel
                            )
                            || global_channel >= in_ch
                        ) return -10;
                        for (int64_t kr = 0; kr < kh; ++kr) {
                            int64_t ih_base = 0;
                            int64_t ih_offset = 0;
                            int64_t ih_shifted = 0;
                            if (
                                !act_i64_mul_nonnegative(
                                    oh, stride_h, &ih_base
                                )
                                || !act_i64_mul_nonnegative(
                                    kr, dilation_h, &ih_offset
                                )
                                || !act_i64_add_nonnegative(
                                    ih_base, ih_offset, &ih_shifted
                                )
                            ) return -11;
                            int64_t ih = ih_shifted - pad_h;
                            if (ih < 0 || ih >= in_h) continue;
                            for (int64_t kc = 0; kc < kw; ++kc) {
                                int64_t iw_base = 0;
                                int64_t iw_offset = 0;
                                int64_t iw_shifted = 0;
                                if (
                                    !act_i64_mul_nonnegative(
                                        ow, stride_w, &iw_base
                                    )
                                    || !act_i64_mul_nonnegative(
                                        kc, dilation_w, &iw_offset
                                    )
                                    || !act_i64_add_nonnegative(
                                        iw_base, iw_offset, &iw_shifted
                                    )
                                ) return -12;
                                int64_t iw = iw_shifted - pad_w;
                                if (iw < 0 || iw >= in_w) continue;
                                size_t weight_position = 0;
                                if (
                                    !act_size_mul(
                                        (size_t)co, (size_t)in_per_group,
                                        &weight_position
                                    )
                                    || !act_size_add(
                                        weight_position, (size_t)ci,
                                        &weight_position
                                    )
                                    || !act_size_mul(
                                        weight_position, (size_t)kh,
                                        &weight_position
                                    )
                                    || !act_size_add(
                                        weight_position, (size_t)kr,
                                        &weight_position
                                    )
                                    || !act_size_mul(
                                        weight_position, (size_t)kw,
                                        &weight_position
                                    )
                                    || !act_size_add(
                                        weight_position, (size_t)kc,
                                        &weight_position
                                    )
                                ) return -13;
                                if (weight_position >= weight_entries) return -3;
                                double coefficient = weight[weight_position];
                                if (coefficient == 0.0) continue;
                                ++fanin;
                                size_t source_row = 0;
                                size_t spatial_row = 0;
                                if (
                                    !act_size_mul(
                                        (size_t)n, (size_t)in_ch, &source_row
                                    )
                                    || !act_size_add(
                                        source_row, (size_t)global_channel,
                                        &source_row
                                    )
                                    || !act_size_mul(
                                        source_row, input_area, &source_row
                                    )
                                    || !act_size_mul(
                                        (size_t)ih, (size_t)in_w, &spatial_row
                                    )
                                    || !act_size_add(
                                        spatial_row, (size_t)iw, &spatial_row
                                    )
                                    || !act_size_add(
                                        source_row, spatial_row, &source_row
                                    )
                                ) return -14;
                                if (source_row >= input_rows) return -4;
                                double magnitude = fabs(coefficient);
                                center_sum = (
                                    center_sum + coefficient * center[source_row]
                                );
                                mass_sum = (
                                    mass_sum
                                    + magnitude * source_mass[source_row]
                                );
                                propagated_sum = (
                                    propagated_sum
                                    + magnitude * source_error[source_row]
                                );
                                mass_active |= source_mass[source_row] > 0.0;
                                propagated_active |= source_error[source_row] > 0.0;
                                int64_t column = row_to_column[source_row];
                                if (column >= 0) {
                                    if (column >= generator_columns) return -5;
                                    double product = (
                                        coefficient * row_scale[source_row]
                                    );
                                    if (product != 0.0) {
                                        if (emitted >= output_capacity) return -6;
                                        if (column > INT32_MAX) return -15;
                                        out_indices[emitted] = (int32_t)column;
                                        out_data[emitted] = product;
                                        ++emitted;
                                    }
                                }
                            }
                        }
                    }
                    double operations = 2.0 * (double)fanin + 2.0;
                    double transformed_mass = act_inflate(
                        mass_sum, operations, mass_active
                    );
                    double propagated_error = act_inflate(
                        propagated_sum, operations, propagated_active
                    );
                    double bias_value = bias[co];
                    double center_value = center_sum + bias_value;
                    double arithmetic_mass = act_inflate(
                        transformed_mass + fabs(bias_value),
                        4.0,
                        (transformed_mass > 0.0) || (bias_value != 0.0)
                    );
                    double arithmetic_error = (
                        act_gamma(operations) * arithmetic_mass
                    );
                    arithmetic_error = act_inflate(
                        arithmetic_error, 4.0, arithmetic_mass > 0.0
                    );
                    double total_error = act_inflate(
                        propagated_error + arithmetic_error,
                        4.0,
                        (propagated_error > 0.0) || (arithmetic_error > 0.0)
                    );
                    out_center[output_row] = center_value;
                    out_mass[output_row] = transformed_mass;
                    out_propagated_error[output_row] = propagated_error;
                    out_error[output_row] = total_error;
                    if (emitted > INT32_MAX) return -16;
                    out_indptr[output_row + 1] = (int32_t)emitted;
                }
            }
        }
    }
    if (output_row != output_rows) return -7;
    *out_emitted = (int64_t)emitted;
    return 0;
}
"""


@lru_cache(maxsize=1)
def _native_backend() -> Tuple[Any, Any]:
    try:
        from cffi import FFI
    except ImportError as exc:  # pragma: no cover
        raise NativeKernelUnavailable("CFFI is not installed") from exc
    ffi = FFI()
    ffi.cdef(_CDEF)
    source_sha = hashlib.sha256(_C_SOURCE.encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(
        tempfile.gettempdir(),
        f"act-conv-compact-{platform.python_version()}-{source_sha}",
    )
    os.makedirs(cache_dir, exist_ok=True)
    try:
        library = ffi.verify(
            _C_SOURCE,
            extra_compile_args=[
                "-O3",
                "-std=c99",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-fexcess-precision=standard",
                "-fwrapv",
                "-fno-strict-overflow",
            ],
            tmpdir=cache_dir,
        )
    except Exception as exc:  # pragma: no cover
        raise NativeKernelUnavailable(
            "strict CFFI compact kernel could not be compiled"
        ) from exc
    if (
        int(ffi.sizeof("int64_t")) != np.dtype(np.int64).itemsize
        or int(ffi.sizeof("int32_t")) != np.dtype(np.int32).itemsize
        or int(ffi.sizeof("size_t")) != np.dtype(np.uintp).itemsize
    ):
        raise NativeKernelUnavailable(
            "CFFI compact kernel ABI does not match native int64/size_t"
        )
    return ffi, library


def native_kernel_environment() -> dict[str, Any]:
    try:
        _native_backend()
    except NativeKernelUnavailable as exc:
        return {
            "available": False,
            "reason": str(exc),
            "cffi_declared_act_dependency": False,
        }
    return {
        "available": True,
        "backend": "cffi.verify+host_c_compiler",
        "strict_flags": (
            "-O3", "-std=c99", "-ffp-contract=off",
            "-fno-fast-math", "-fexcess-precision=standard",
            "-fwrapv", "-fno-strict-overflow",
        ),
        "cffi_declared_act_dependency": False,
        "runtime_compile_is_production_blocker": True,
    }


def _require_private(
    value: Any,
    *,
    dtype: np.dtype[Any],
    shape: Tuple[int, ...],
    name: str,
) -> np.ndarray:
    root = value
    visited: set[int] = set()
    while type(root) is np.ndarray and root.base is not None:
        if id(root) in visited:
            raise ExactConvNativeScheduleError(f"{name} has cyclic storage")
        visited.add(id(root))
        root = root.base
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
        or value.flags.writeable
        or type(root) is not bytes
    ):
        raise ExactConvNativeScheduleError(f"{name} is not a private snapshot")
    return value


def _validate_hot_inputs(
    owner: Any,
    schedule: ExactConvNativeSchedule,
    source: ExactMonotoneRowLocalSource,
    expected_ids: Any,
) -> Tuple[_CapturedSchedule, _CapturedSource, Tuple[int, ...]]:
    captured_schedule, captured_source = _admit_registered_factory_objects(
        owner, schedule, source
    )
    values = _validate_schedule_snapshot(
        captured_schedule,
        owner=owner,
        require_digest=False,
        captured=True,
    )
    _validate_source_snapshot(
        captured_source,
        require_digest=False,
        captured=True,
    )
    input_size = _checked_product(
        (values[0], values[1], values[2], values[3]),
        name="hot input size",
    )
    if captured_source.size != input_size:
        raise ExactConvNativeScheduleError("source/schedule row mismatch")
    try:
        expected = np.asarray(expected_ids)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExactConvNativeScheduleError("expected stable ids are invalid") from exc
    if (
        expected.ndim != 1
        or expected.dtype.kind not in "iu"
        or not np.array_equal(expected, captured_source.stable_column_ids)
    ):
        raise ExactConvNativeScheduleError("source replay validation failed")
    return captured_schedule, captured_source, values


def _invoke_native(
    schedule: _CapturedSchedule,
    source: _CapturedSource,
    geometry: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ffi, library = _native_backend()
    batch, _in_ch, _in_h, _in_w, out_ch, out_h, out_w, in_per, kh, kw, *_ = geometry
    input_size = _checked_product(
        (geometry[0], geometry[1], geometry[2], geometry[3]),
        name="native input rows",
    )
    output_size = _checked_product(
        (batch, out_ch, out_h, out_w), name="native output rows"
    )
    capacity = _checked_product(
        (output_size, in_per, kh, kw), name="native output capacity"
    )
    center = np.empty(output_size, dtype=np.float64)
    mass = np.empty(output_size, dtype=np.float64)
    propagated = np.empty(output_size, dtype=np.float64)
    error = np.empty(output_size, dtype=np.float64)
    indptr = np.empty(output_size + 1, dtype=np.int32)
    indices = np.empty(capacity, dtype=np.int32)
    data = np.empty(capacity, dtype=np.float64)
    const_f64 = lambda value: ffi.from_buffer("const double[]", value)
    const_i64 = lambda value: ffi.from_buffer("const int64_t[]", value)
    out_f64 = lambda value: ffi.from_buffer("double[]", value)
    out_i32 = lambda value: ffi.from_buffer("int32_t[]", value)
    emitted_pointer = ffi.new("int64_t *")
    status = int(
        library.act_exact_conv_compact_fused_v3(
            *geometry,
            input_size,
            output_size,
            int(schedule.weight.size),
            capacity,
            source.generator_columns,
            const_f64(schedule.weight),
            const_f64(schedule.bias_channels),
            const_f64(source.center),
            const_f64(source.source_mass),
            const_f64(source.error),
            const_i64(source.row_to_generator_column),
            const_f64(source.row_scale),
            out_f64(center), out_f64(mass), out_f64(propagated), out_f64(error),
            out_i32(indptr), out_i32(indices), out_f64(data), emitted_pointer,
        )
    )
    emitted = int(emitted_pointer[0])
    if status != 0 or emitted > capacity or int(indptr[-1]) != emitted:
        raise ExactConvNativeScheduleError("native kernel emitted invalid CSR size")
    return center, mass, propagated, error, indptr, indices[:emitted], data[:emitted]


def apply_exact_conv_native_schedule(
    owner: Any,
    schedule: ExactConvNativeSchedule,
    source: ExactMonotoneRowLocalSource,
    *,
    expected_stable_column_ids: Any,
    source_affine_depth: int = 0,
    return_receipt: bool = False,
) -> ExactConvNativeResult | Tuple[ExactConvNativeResult, ExactConvNativeReceipt]:
    """Apply compact geometry once; no expanded W exists in this path."""

    captured_schedule, captured_source, geometry = _validate_hot_inputs(
        owner, schedule, source, expected_stable_column_ids
    )
    depth = _builtin_int(source_affine_depth, name="source_affine_depth")
    if depth < 0:
        raise ExactConvNativeScheduleError("source_affine_depth must be nonnegative")
    center, mass, propagated, error, indptr, indices, data = _invoke_native(
        captured_schedule, captured_source, geometry
    )
    if (
        not all(np.all(np.isfinite(v)) for v in (center, mass, propagated, error, data))
        or np.any(mass < 0.0)
        or np.any(propagated < 0.0)
        or np.any(error < 0.0)
        or np.any(data == 0.0)
        or np.any(indptr < 0)
        or np.any(indptr > _INT32_MAX)
        or np.any(indices < 0)
        or np.any(indices > _INT32_MAX)
        or np.any(indices >= captured_source.generator_columns)
    ):
        raise ExactConvNativeScheduleError("native result is nonfinite or invalid")
    if data.size > 1:
        same_row = np.ones(data.size - 1, dtype=np.bool_)
        boundaries = indptr[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < data.size)]
        if boundaries.size:
            same_row[np.asarray(boundaries, dtype=np.int64) - 1] = False
        if np.any(same_row & (indices[1:] <= indices[:-1])):
            raise ExactConvNativeScheduleError("native generators are not canonical")

    # These arrays are fresh kernel outputs with no external alias.  Transfer
    # their owners into the result instead of retaining raw+frozen copies.
    frozen_center = _adopt_output_array(center, dtype=np.dtype(np.float64))
    frozen_mass = _adopt_output_array(mass, dtype=np.dtype(np.float64))
    frozen_propagated = _adopt_output_array(
        propagated, dtype=np.dtype(np.float64)
    )
    frozen_error = _adopt_output_array(error, dtype=np.dtype(np.float64))
    frozen_indptr = _adopt_output_array(indptr, dtype=np.dtype(np.int32))
    frozen_indices = _adopt_output_array(indices, dtype=np.dtype(np.int32))
    frozen_data = _adopt_output_array(data, dtype=np.dtype(np.float64))
    generators = sp.csr_matrix(
        (frozen_data, frozen_indices, frozen_indptr),
        shape=(
            captured_schedule.output_size,
            captured_source.generator_columns,
        ),
        dtype=np.float64,
        copy=False,
    )
    frozen_ids = captured_source.stable_column_ids
    result = ExactConvNativeResult(
        center=frozen_center,
        generators=generators,
        error=frozen_error,
        transformed_mass=frozen_mass,
        propagated_error=frozen_propagated,
        stable_column_ids=frozen_ids,
        affine_depth=_checked_add_i64(depth, 1, name="result affine depth"),
        schedule_digest=captured_schedule.digest,
        source_digest=captured_source.digest,
    )
    if not return_receipt:
        return result
    geometry_values = tuple(int(v) for v in captured_schedule.geometry)
    capacity = _checked_product(
        (
            captured_schedule.output_size,
            geometry_values[7],
            geometry_values[8],
            geometry_values[9],
        ),
        name="receipt output capacity",
    )
    return result, ExactConvNativeReceipt(
        schedule_digest=captured_schedule.digest,
        source_digest=captured_source.digest,
        compact_weight_entries=int(captured_schedule.weight.size),
        compact_schedule_bytes=captured_schedule.compact_buffer_nbytes,
        output_generator_nnz=int(generators.nnz),
        output_capacity_nnz=int(capacity),
        input_shape=captured_schedule.input_shape,
        output_shape=captured_schedule.output_shape,
    )


__all__ = [
    "ExactConvNativeReceipt",
    "ExactConvNativeResult",
    "ExactConvNativeSchedule",
    "ExactConvNativeScheduleError",
    "ExactMonotoneRowLocalSource",
    "MonotoneRowLocalNotApplicable",
    "NativeKernelUnavailable",
    "apply_exact_conv_native_schedule",
    "native_kernel_environment",
    "prepare_exact_conv_native_schedule",
    "prepare_exact_monotone_row_local_source",
    "project_compact_schedule_cache_bytes",
]
